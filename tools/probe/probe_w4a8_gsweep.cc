// The W4A8 GROUP-SIZE SWEEP (2026-09-23).
//
// Extends the rejected W4A8 probe (docs/probe-w4a8-2026-09-23.md,
// tools/probe/probe_w4a8.{cl,cc}) with the one measurement that record said it
// could not make: the same shape at scale strides other than the checkpoint's
// 64. The rescale FREQUENCY is a function of the scale STRIDE, not the scale
// VALUES, so the speed curve is measurable on the checkpoint that exists.
//
// THIS MEASURES SPEED ONLY.
// At every arm past g64 the weight scales are SYNTHESISED at the coarser stride
// from the checkpoint's own g64 scales (max over the constituent groups). That
// is legitimate for timing -- the kernel issues exactly the instruction stream a
// real coarse-group checkpoint would make it issue -- and it is NOT a real
// quantisation: the g128, g256 and per-channel OUTPUTS ARE WRONG ON PURPOSE.
// This program therefore prints NO error-against-bf16 figure for any arm. The
// published 2.79 % relative L2 came from the activation side with the weights
// unchanged, is unaffected by group size, and is not re-derived here.
//
// Inputs: the SAME capture file the rejected probe used -- the real layer-63
// gate||up int4 weights and the real activations one prefill chunk fed them.
// Same shape: K = 5120, N = 34816, M = 2048.
//
// Arms:
//   bf16 two-pass control   pf_dequant_slab + pf_gemm_T0, the production binaries
//   pw4a8_quant             the activation quantiser (part of every total)
//   pw4a8_gemm              the PUBLISHED g64 kernel, unchanged -- the drift check
//   pw4a8s_g64/g128/g256/gpc  the sweep, identical but for the rescale stride
//
// DISPATCH PROOF ON EVERY ARM: zeKernelGetName on the handle launch() receives,
// plus a signature word the kernel itself writes carrying ITS OWN group stride
// and chunk count, plus an integer-exact CPU oracle recomputed at that arm's
// stride over a 32 x 32 slice at full K. The oracle is the strong one: a kernel
// applying a different stride cannot reproduce it.
//
// PROBE-ONLY: no production file is edited; production stays g64.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <level_zero/ze_api.h>

#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "runtime/prefill/context.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;
constexpr uint32_t kGroup = 64;             // the checkpoint's int4 group
constexpr uint32_t kNs = 1024;              // kernels::kPfSlabWidth
constexpr uint32_t kTile = 256;             // pf_gemm's WG_M / WG_N
constexpr uint32_t kW4Tile = 128;           // the W4A8 tile's WG_N
constexpr uint32_t kSlabs = kN / kNs;       // 34
constexpr int kReplays = 5;

constexpr double kGflop = 2.0 * double(kM) * double(kK) * double(kN) / 1e9;
constexpr double kStudyMult = 1.53;         // the pre-registered bar

struct Header {
  char magic[8];
  uint32_t version, layer, M, K, N, group, pad;
};

const char* env_or_unset(const char* n) {
  const char* v = std::getenv(n);
  return v ? v : "(unset)";
}

std::string kernel_name(ze_kernel_handle_t k) {
  size_t n = 0;
  if (zeKernelGetName(k, &n, nullptr) != ZE_RESULT_SUCCESS || n == 0) return "<unnamed>";
  std::string s(n, '\0');
  if (zeKernelGetName(k, &n, s.data()) != ZE_RESULT_SUCCESS) return "<unnamed>";
  if (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

uint64_t raw_start(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelStart & mask;
}
uint64_t raw_end(const l0::Event& e, uint64_t mask) {
  ze_kernel_timestamp_result_t ts{};
  zeEventQueryKernelTimestamp(e.handle(), &ts);
  return ts.global.kernelEnd & mask;
}

struct Timed {
  double sum_ms = 0.0, even_ms = 0.0, odd_ms = 0.0;
};

// Exactly the published probe's timing discipline: L0 kernel timestamps, best
// of kReplays after one discarded warm-up.
template <class F>
Timed best_of(F run, std::vector<l0::Event>& ev, const l0::TimerCalib& calib) {
  const uint64_t mask = calib.mask();
  Timed best;
  bool have = false;
  for (int r = -1; r < kReplays; ++r) {
    for (l0::Event& e : ev) e.reset();
    const size_t n = run();
    Timed cur;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double us =
          double((raw_end(ev[i], mask) - raw_start(ev[i], mask)) & mask) / calib.cycles_per_us;
      sum += us;
      if (i % 2 == 0) cur.even_ms += us; else cur.odd_ms += us;
    }
    cur.sum_ms = sum / 1000.0;
    cur.even_ms /= 1000.0;
    cur.odd_ms /= 1000.0;
    if (r < 0) continue;
    if (!have || cur.sum_ms < best.sum_ms) { best = cur; have = true; }
  }
  return best;
}

void download(l0::Context& ctx, void* host, const void* dev, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    const size_t now = std::min(kChunk, bytes - off);
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(host) + off, static_cast<const char*>(dev) + off, now);
  }
}

// One sweep arm.
struct Arm {
  const char* entry;       // OpenCL entry point
  uint32_t stride;         // k per scale
  uint32_t subs;           // 64-k chunks per rescale group
  const char* label;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: probe_w4a8_gsweep <capture-file>\n");
      return 2;
    }

    // --- the inputs, byte for byte the rejected probe's -------------------
    const std::string in_path = argv[1];
    std::ifstream f(in_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + in_path);
    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof h);
    if (std::memcmp(h.magic, "B70W4A8", 8) != 0 || h.M != kM || h.K != kK || h.N != kN ||
        h.group != kGroup)
      throw std::runtime_error(in_path + " is not the W4A8 probe's capture file");
    const size_t act_bytes = size_t(kM) * kK * sizeof(uint16_t);
    const size_t q_bytes = size_t(kK / 8) * kN * sizeof(uint32_t);
    const size_t s_bytes = size_t(kK / kGroup) * kN * sizeof(uint16_t);
    std::vector<uint16_t> act(size_t(kM) * kK);
    std::vector<uint32_t> qw(size_t(kK / 8) * kN);
    std::vector<uint16_t> sc(size_t(kK / kGroup) * kN);
    f.read(reinterpret_cast<char*>(act.data()), std::streamsize(act_bytes));
    f.read(reinterpret_cast<char*>(qw.data()), std::streamsize(q_bytes));
    f.read(reinterpret_cast<char*>(sc.data()), std::streamsize(s_bytes));
    if (!f) throw std::runtime_error(in_path + " is truncated");
    f.close();

    std::printf("# W4A8 GROUP-SIZE SWEEP -- SPEED ONLY (see the header comment)\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    l0::Context ctx(0);
    std::printf("# L0 device: %s\n", ctx.name().c_str());
    std::printf("# inputs: %s -- the REAL layer-%u gate||up int4 weights (unchanged bytes) and "
                "the REAL activations one prefill chunk fed them\n", in_path.c_str(), h.layer);
    std::printf("# shape: K=%u N=%u M=%u; %.1f GFLOP per call; bar %.2fx the bf16 control\n",
                kK, kN, kM, kGflop, kStudyMult);
    std::printf("# scales at stride > 64 are SYNTHESISED (max over the constituent g64 groups). "
                "Timing only; those outputs are wrong on purpose and NO error figure is "
                "printed for them.\n");

    static const Arm kArms[] = {
        {"pw4a8s_g64",   64, 1,  "g64 (the checkpoint's stride)"},
        {"pw4a8s_g128", 128, 2,  "g128 (synthesised stride)"},
        {"pw4a8s_g256", 256, 4,  "g256 (synthesised stride)"},
        {"pw4a8s_gpc", 5120, 80, "per-channel (one scale for the whole K)"},
    };
    constexpr size_t kNArms = sizeof(kArms) / sizeof(kArms[0]);

    // --- synthesised coarse scales, one array per arm ---------------------
    std::vector<std::vector<uint16_t>> coarse(kNArms);
    for (size_t i = 0; i < kNArms; ++i) {
      const uint32_t subs = kArms[i].subs;
      const uint32_t G = (kK / kGroup) / subs;
      if (G * subs != kK / kGroup) throw std::runtime_error("arm stride does not divide K");
      coarse[i].assign(size_t(G) * kN, 0);
      // The winning group's ORIGINAL f16 WORD is copied, never a float
      // round-trip: the 27B's scales really do go subnormal (bf16.h says so and
      // loader::assert_quant_invariants counts them) and f32_to_f16 flushes
      // those to zero. At subs == 1 this therefore reproduces the checkpoint's
      // own scale array bit for bit, which the check below enforces.
      for (uint32_t g = 0; g < G; ++g)
        for (uint32_t n = 0; n < kN; ++n) {
          uint16_t best = sc[size_t(g * subs) * kN + n];
          float mx = common::f16_to_f32(best);
          for (uint32_t j = 1; j < subs; ++j) {
            const uint16_t w = sc[size_t(g * subs + j) * kN + n];
            const float v = common::f16_to_f32(w);
            if (v > mx) { mx = v; best = w; }
          }
          coarse[i][size_t(g) * kN + n] = best;
        }
    }
    // The g64 arm's "synthesised" array must be the checkpoint's own scales,
    // bit for bit, or the arm is not the control it claims to be.
    if (std::memcmp(coarse[0].data(), sc.data(), s_bytes) != 0)
      throw std::runtime_error("the g64 arm's scale array is not the checkpoint's own");

    runtime::prefill::Context cx(ctx);

    // --- device buffers ---------------------------------------------------
    const size_t c_bytes = size_t(kM) * kN * sizeof(float);
    l0::Mem da(ctx, l0::MemKind::Device, act_bytes);
    l0::Mem dq(ctx, l0::MemKind::Device, q_bytes);
    l0::Mem ds(ctx, l0::MemKind::Device, s_bytes);
    l0::Mem slab(ctx, l0::MemKind::Device, size_t(kK) * kNs * sizeof(uint16_t));
    l0::Mem c_ctl(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_ref(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_arm(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem xq(ctx, l0::MemKind::Device, size_t(kM) * kK);
    l0::Mem xs(ctx, l0::MemKind::Device, size_t(kM) * sizeof(float));
    l0::Mem sig(ctx, l0::MemKind::Shared, 64);
    std::vector<l0::Mem> dcs;                       // one coarse scale buffer per arm
    dcs.reserve(kNArms);
    for (size_t i = 0; i < kNArms; ++i)
      dcs.emplace_back(ctx, l0::MemKind::Device, coarse[i].size() * sizeof(uint16_t));
    {
      l0::CmdList up = l0::CmdList::immediate(ctx);
      up.copy(da.ptr(), act.data(), act_bytes);
      up.copy(dq.ptr(), qw.data(), q_bytes);
      up.copy(ds.ptr(), sc.data(), s_bytes);
      up.fill(sig.ptr(), 0u, 64);
      for (size_t i = 0; i < kNArms; ++i)
        up.copy(dcs[i].ptr(), coarse[i].data(), coarse[i].size() * sizeof(uint16_t));
    }

    // --- kernels ----------------------------------------------------------
    l0::Module m_dq(ctx, kernels::path(kernels::pf_dequant_slab_variant(kK, kN, 0)));
    l0::Kernel k_dq = m_dq.kernel("pf_dequant_slab");
    l0::Module m_gemm(ctx, kernels::path(kernels::pf_gemm_variant(false)));
    l0::Kernel k_gemm = m_gemm.kernel("pf_gemm");
    l0::Module m_w4(ctx, kernels::path("pw4a8"));
    l0::Kernel k_w4 = m_w4.kernel("pw4a8_gemm");
    l0::Kernel k_qt = m_w4.kernel("pw4a8_quant");
    l0::Module m_sw(ctx, kernels::path("pw4a8s"));
    std::vector<l0::Kernel> k_arm;
    k_arm.reserve(kNArms);
    for (size_t i = 0; i < kNArms; ++i) k_arm.push_back(m_sw.kernel(kArms[i].entry));

    l0::EventPool pool(ctx, 2 * kSlabs);
    std::vector<l0::Event> ev;
    ev.reserve(2 * kSlabs);
    for (uint32_t i = 0; i < 2 * kSlabs; ++i) ev.emplace_back(pool, i);
    const l0::TimerCalib calib = pool.calib();

    const uint32_t lda = kK, ldc = kN, ldxq = kK / 2;
    const uint64_t zero = 0;
    auto run_control = [&]() -> size_t {
      float* out = c_ctl.as<float>();
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        cx.launch(k_dq, kNs / 16, kK / 64, 1,
                  {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0)},
                  &ev[2 * i]);
        cx.launch(k_gemm, kM / kTile, kNs / kTile, 1,
                  {PtrArg(da.ptr()), PtrArg(slab.ptr()), PtrArg(out + n0), arg_val(kM),
                   arg_val(kK), arg_val(kNs), arg_val(lda), arg_val(kNs), arg_val(ldc),
                   arg_val(zero), arg_val(zero), arg_val(zero)},
                  &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };
    auto run_quant = [&]() -> size_t {
      cx.launch(k_qt, kM, 1, 1,
                {PtrArg(da.ptr()), PtrArg(xq.ptr()), PtrArg(xs.ptr()), arg_val(kM), arg_val(kK),
                 arg_val(lda)},
                &ev[0]);
      cx.wait();
      return size_t(1);
    };
    auto run_published = [&]() -> size_t {
      cx.launch(k_w4, kM / kTile, kN / kW4Tile, 1,
                {PtrArg(xq.ptr()), PtrArg(xs.ptr()), PtrArg(dq.ptr()), PtrArg(ds.ptr()),
                 PtrArg(c_ref.ptr()), PtrArg(sig.ptr()), arg_val(kM), arg_val(kK), arg_val(kN),
                 arg_val(ldxq), arg_val(ldc)},
                &ev[0]);
      cx.wait();
      return size_t(1);
    };
    auto make_arm = [&](size_t i) {
      return [&, i]() -> size_t {
        cx.launch(k_arm[i], kM / kTile, kN / kW4Tile, 1,
                  {PtrArg(xq.ptr()), PtrArg(xs.ptr()), PtrArg(dq.ptr()), PtrArg(dcs[i].ptr()),
                   PtrArg(c_arm.ptr()), PtrArg(sig.ptr()), arg_val(kM), arg_val(kK), arg_val(kN),
                   arg_val(ldxq), arg_val(ldc)},
                  &ev[0]);
        cx.wait();
        return size_t(1);
      };
    };

    // --- dispatch proof, part 1: the handles ------------------------------
    std::printf("\n## dispatch proof -- the handles `launch()` receives\n\n");
    std::printf("| arm | module | `zeKernelGetName` on the launch handle |\n|---|---|---|\n");
    std::printf("| control dequant | `%s` | `%s` |\n",
                kernels::path(kernels::pf_dequant_slab_variant(kK, kN, 0)).c_str(),
                kernel_name(k_dq.handle()).c_str());
    std::printf("| control GEMM | `%s` | `%s` |\n", kernels::path(kernels::pf_gemm_variant(false)).c_str(),
                kernel_name(k_gemm.handle()).c_str());
    std::printf("| quantiser | `%s` | `%s` |\n", kernels::path("pw4a8").c_str(),
                kernel_name(k_qt.handle()).c_str());
    std::printf("| published g64 | `%s` | **`%s`** |\n", kernels::path("pw4a8").c_str(),
                kernel_name(k_w4.handle()).c_str());
    for (size_t i = 0; i < kNArms; ++i) {
      std::printf("| sweep %s | `%s` | **`%s`** |\n", kArms[i].label,
                  kernels::path("pw4a8s").c_str(), kernel_name(k_arm[i].handle()).c_str());
      if (kernel_name(k_arm[i].handle()) != kArms[i].entry)
        throw std::runtime_error(std::string("dispatch proof failed for ") + kArms[i].entry);
    }
    if (kernel_name(k_w4.handle()) != "pw4a8_gemm")
      throw std::runtime_error("dispatch proof failed: the published handle is not pw4a8_gemm");

    // --- value pass: run everything once, prove each arm's own stride -----
    run_control();
    run_quant();
    std::vector<int8_t> xq_h(size_t(kM) * kK);
    std::vector<float> xs_h(kM);
    download(ctx, xq_h.data(), xq.ptr(), xq_h.size());
    download(ctx, xs_h.data(), xs.ptr(), kM * sizeof(float));

    {
      l0::CmdList c = l0::CmdList::immediate(ctx);
      c.fill(sig.ptr(), 0u, 64);
    }
    run_published();
    const uint32_t ref_sig = sig.as<uint32_t>()[0];
    std::printf("\n## dispatch proof -- the signature each kernel WRITES\n\n");
    std::printf("| arm | signature word | expected | sig[1] (64-k chunks per rescale) |\n");
    std::printf("|---|---|---|---:|\n");
    std::printf("| published `pw4a8_gemm` | **0x%08X** | 0x57344138 ('W4A8') | - |\n", ref_sig);
    if (ref_sig != 0x57344138u)
      throw std::runtime_error("dispatch proof failed: pw4a8_gemm did not run");

    // the integer-exact CPU oracle, recomputed at EACH ARM'S OWN STRIDE
    constexpr uint32_t kRows = 32, kCols = 32;
    auto oracle = [&](const uint16_t* wsc, uint32_t stride, const float* got) {
      const uint32_t G = kK / stride;
      double worst = 0.0;
      for (uint32_t m = 0; m < kRows; ++m)
        for (uint32_t n = 0; n < kCols; ++n) {
          float acc = 0.0f;
          for (uint32_t g = 0; g < G; ++g) {
            int32_t s32 = 0;
            for (uint32_t j = 0; j < stride; ++j) {
              const uint32_t k = g * stride + j;
              const uint32_t word = qw[size_t(k / 8) * kN + n];
              const int q = int((word >> (4 * (k % 8))) & 0xFu) - 8;
              s32 += int32_t(xq_h[size_t(m) * kK + k]) * q;
            }
            acc = std::fma(float(s32), common::f16_to_f32(wsc[size_t(g) * kN + n]) * xs_h[m], acc);
          }
          worst = std::max(worst, std::fabs(double(acc) - double(got[size_t(m) * kN + n])));
        }
      return worst;
    };

    std::vector<float> slice(size_t(kRows) * kN);
    auto fetch_slice = [&](l0::Mem& mem) {
      for (uint32_t m = 0; m < kRows; ++m)
        download(ctx, slice.data() + size_t(m) * kN, mem.as<float>() + size_t(m) * kN,
                 size_t(kN) * sizeof(float));
    };
    fetch_slice(c_ref);
    const double ref_orc = oracle(sc.data(), kGroup, slice.data());

    struct ArmRun { uint32_t sig0, sig1; double orc; double kernel_ms; };
    std::vector<ArmRun> ar(kNArms);
    for (size_t i = 0; i < kNArms; ++i) {
      {
        l0::CmdList c = l0::CmdList::immediate(ctx);
        c.fill(sig.ptr(), 0u, 64);
      }
      make_arm(i)();
      ar[i].sig0 = sig.as<uint32_t>()[0];
      ar[i].sig1 = sig.as<uint32_t>()[1];
      const uint32_t want = 0x47000000u | kArms[i].stride;
      std::printf("| sweep %s | **0x%08X** | 0x%08X ('G' \\| %u) | **%u** |\n", kArms[i].label,
                  ar[i].sig0, want, kArms[i].stride, ar[i].sig1);
      if (ar[i].sig0 != want || ar[i].sig1 != kArms[i].subs)
        throw std::runtime_error(std::string("dispatch proof failed: ") + kArms[i].entry +
                                 " did not write its own stride");
      fetch_slice(c_arm);
      ar[i].orc = oracle(coarse[i].data(), kArms[i].stride, slice.data());

      // the g64 arm must reproduce the PUBLISHED kernel bit for bit over the
      // whole output: same scales, same weights, same activations, same order
      if (i == 0) {
        constexpr uint32_t kRowBlock = 128;
        const size_t block = size_t(kRowBlock) * kN * sizeof(float);
        l0::Mem ha(ctx, l0::MemKind::Host, block);
        l0::Mem hb(ctx, l0::MemKind::Host, block);
        size_t diff = 0;
        for (uint32_t m0 = 0; m0 < kM; m0 += kRowBlock) {
          download(ctx, ha.ptr(), c_ref.as<float>() + size_t(m0) * kN, block);
          download(ctx, hb.ptr(), c_arm.as<float>() + size_t(m0) * kN, block);
          const uint32_t* a = ha.as<uint32_t>();
          const uint32_t* b = hb.as<uint32_t>();
          for (size_t j = 0; j < size_t(kRowBlock) * kN; ++j) diff += (a[j] != b[j]);
        }
        std::printf("\n`pw4a8s_g64` vs the published `pw4a8_gemm`, all %zu fp32 outputs: "
                    "**%zu differing words** (bit-exact iff 0)\n\n", size_t(kM) * kN, diff);
        if (diff != 0)
          throw std::runtime_error("the g64 sweep arm is not bit-identical to pw4a8_gemm");
      }
    }

    std::printf("\n## dispatch proof -- integer-exact CPU oracle at EACH ARM'S OWN STRIDE "
                "(%u x %u slice, full K)\n\n", kRows, kCols);
    std::printf("A kernel applying any other stride cannot reproduce these. The oracle uses the "
                "same synthesised scale array the arm was handed; it proves the STRIDE, and says "
                "nothing about accuracy.\n\n");
    std::printf("| arm | stride recomputed on the host | max \\|device - host\\| |\n|---|---:|---:|\n");
    std::printf("| published `pw4a8_gemm` | 64 | **%.6g** |\n", ref_orc);
    for (size_t i = 0; i < kNArms; ++i)
      std::printf("| `%s` | %u | **%.6g** |\n", kArms[i].entry, kArms[i].stride, ar[i].orc);
    if (ref_orc != 0.0) throw std::runtime_error("published kernel disagrees with its oracle");
    for (size_t i = 0; i < kNArms; ++i)
      if (ar[i].orc != 0.0)
        throw std::runtime_error(std::string(kArms[i].entry) + " disagrees with its oracle");
    std::fflush(stdout);

    // --- the rates --------------------------------------------------------
    const Timed t_ctl = best_of(run_control, ev, calib);
    const Timed t_qt = best_of(run_quant, ev, calib);
    const Timed t_ref = best_of(run_published, ev, calib);
    for (size_t i = 0; i < kNArms; ++i) ar[i].kernel_ms = best_of(make_arm(i), ev, calib).sum_ms;

    std::printf("\n## the sweep (measured; L0 kernel timestamps, best of %d after one discarded "
                "warm-up)\n\n", kReplays);
    std::printf("| arm | group size | kernel ms | total ms (+ quantiser) | x the bf16 control | "
                "clears %.2fx |\n|---|---:|---:|---:|---:|:--:|\n", kStudyMult);
    std::printf("| **bf16 two-pass control** (34 slabs, %u launches) | - | **%.3f** | - | 1.000x "
                "| - |\n", 2 * kSlabs, t_ctl.sum_ms);
    std::printf("| &nbsp;&nbsp;of which `pf_dequant_slab` | - | %.3f | - | - | - |\n", t_ctl.even_ms);
    std::printf("| &nbsp;&nbsp;of which `pf_gemm_T0` | - | %.3f | - | - | - |\n", t_ctl.odd_ms);
    std::printf("| **activation quantiser** `pw4a8_quant` | - | **%.3f** | - | - | - |\n",
                t_qt.sum_ms);
    std::printf("| published `pw4a8_gemm` (drift check) | 64 | **%.3f** | **%.3f** | **%.3fx** | "
                "%s |\n", t_ref.sum_ms, t_ref.sum_ms + t_qt.sum_ms,
                t_ctl.sum_ms / (t_ref.sum_ms + t_qt.sum_ms),
                t_ctl.sum_ms / (t_ref.sum_ms + t_qt.sum_ms) >= kStudyMult ? "YES" : "no");
    bool any = false;
    for (size_t i = 0; i < kNArms; ++i) {
      const double tot = ar[i].kernel_ms + t_qt.sum_ms;
      const double mult = t_ctl.sum_ms / tot;
      any = any || (mult >= kStudyMult);
      std::printf("| `%s` | %u | **%.3f** | **%.3f** | **%.3fx** | %s |\n", kArms[i].entry,
                  kArms[i].stride, ar[i].kernel_ms, tot, mult, mult >= kStudyMult ? "**YES**" : "no");
    }
    std::printf("\nTOP/s: control %.2f; ", kGflop / t_ctl.sum_ms);
    for (size_t i = 0; i < kNArms; ++i)
      std::printf("%s %.2f%s", kArms[i].entry, kGflop / ar[i].kernel_ms,
                  i + 1 == kNArms ? "\n" : "; ");
    std::printf("\n## verdict (speed only)\n\n");
    if (any)
      std::printf("At least one granularity clears %.2fx. The next step is a CPU ACCURACY study "
                  "at that granularity, not a kernel.\n", kStudyMult);
    else
      std::printf("**No granularity clears %.2fx, per-channel included. The int8 path is closed "
                  "on this hardware.**\n", kStudyMult);
    std::printf("\nThis program measured SPEED ONLY. Scales at stride > 64 are synthesised and "
                "those arms' outputs are wrong on purpose; no accuracy claim is made or implied "
                "by any row above.\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_w4a8_gsweep FAILED: %s\n", ex.what());
    return 2;
  }
}
