// The W4A8 probe (pre-registration:
// docs/specs/2026-09-22-w4a8-dpas-probe-design.md; results:
// docs/probe-w4a8-2026-09-23.md).
//
// ONE shape, design §4: gate||up, K = 5120, N = 34816, M = 2048 -- 730.1 GFLOP,
// against the two-pass bf16 control already measured at 5.341 ms / 136.72
// TFLOP/s (docs/probe-fused-dequant-2026-09-22.md). Break-even 5.341 ms; the
// probe is worth a study at <= 3.5 ms (>= 1.53x) and the instruction's nominal
// promise is 2.67 ms (2.00x). Below 1.53x the probe is REJECTED and recorded.
//
// TWO MODES, because design §5 requires real weights and real activations and
// the model is 19 GB:
//
//   --capture <snapshot> <ids-file> <out-file> [layer]
//       Loads the checkpoint through the production loader, prefills 2048 ids
//       of a golden prompt through runtime::Engine::prefill on the L0 backend,
//       and dumps ONE file holding (a) the gate||up A operand of layer `layer`
//       exactly as that prefill computed it -- the post-attention RMSNorm
//       output the fused gate||up GEMM reads, `PrefillScratch::mixer_out` --
//       and (b) that layer's int4 gate||up weights and f16 group scales, read
//       back off the device. Run once; the file is reused.
//
//   <in-file>
//       The measurement. Uploads the captured tensors and times three things:
//       the bf16 two-pass control (the production pf_dequant_slab + pf_gemm_T0
//       binaries, launch for launch as runtime::prefill::linear_l0 issues
//       them), the W4A8 GEMM, and the activation quantiser alone. Then the
//       error characterisation of §5 against the control's own fp32 output.
//
// PROBE-ONLY: no production file is edited. The control runs the production
// binaries themselves; the two new kernels live in tools/probe/probe_w4a8.cl
// and nothing in the runtime names them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
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
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/engine.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill_backend.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr uint32_t kK = 5120;
constexpr uint32_t kN = 34816;
constexpr uint32_t kM = 2048;               // PrefillScratch::kC
constexpr uint32_t kGroup = 64;             // the checkpoint's int4 group
constexpr uint32_t kNs = 1024;              // kernels::kPfSlabWidth
constexpr uint32_t kTile = 256;             // pf_gemm's WG_M / WG_N
constexpr uint32_t kW4Tile = 128;           // pw4a8_gemm's WG_N
constexpr uint32_t kSlabs = kN / kNs;       // 34
constexpr int kReplays = 5;
constexpr uint32_t kDefaultLayer = 63;

// 730.1 GFLOP per call (derived: 2 x 2048 x 5120 x 34816).
constexpr double kGflop = 2.0 * double(kM) * double(kK) * double(kN) / 1e9;

// Design §4, restated before any measurement and unchanged since.
constexpr double kBreakEvenMs = 5.341;   // the same-shape two-pass bf16 control
constexpr double kStudyMs = 3.5;         // <= this (>= 1.53x) is worth a study
constexpr double kPromiseMs = 2.67;      // the instruction's nominal 2.00x

constexpr uint32_t kSigMagic = 0x57344138u;   // 'W4A8', written by pw4a8_gemm

struct Header {
  char magic[8];          // "B70W4A8\0"
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
  double sum_ms = 0.0, span_ms = 0.0, even_ms = 0.0, odd_ms = 0.0;
  double tflops() const { return kGflop / sum_ms; }
};

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
    cur.span_ms = double((raw_end(ev[n - 1], mask) - raw_start(ev[0], mask)) & mask) /
                  calib.cycles_per_us / 1000.0;
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

// --------------------------------------------------------------------------
// capture: real weights and real activations, through the production path
// --------------------------------------------------------------------------
int capture(const std::string& snap, const std::string& ids_path, const std::string& out_path,
            uint32_t layer) {
  std::printf("# W4A8 probe -- capture (design §5: real weights, real activations)\n");
  std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
  std::printf("# B70_PREFILL_SILU_FUSED=%s (unset/non-zero => the fused gate||up epilogue, "
              "whose A operand is PrefillScratch::mixer_out)\n",
              env_or_unset("B70_PREFILL_SILU_FUSED"));

  std::vector<uint32_t> ids;
  {
    std::ifstream f(ids_path);
    if (!f) throw std::runtime_error("cannot open ids file: " + ids_path);
    for (long long v; f >> v;) ids.push_back(uint32_t(v));
  }
  if (ids.size() < kM) throw std::runtime_error("prompt has only " + std::to_string(ids.size()) +
                                                " ids; need " + std::to_string(kM));
  ids.resize(kM);
  std::printf("# prompt: %s, first %u ids (one full chunk at C = %u)\n", ids_path.c_str(), kM, kM);

  l0::Context ctx(0);
  std::printf("# L0 device: %s\n", ctx.name().c_str());
  constexpr uint32_t kMaxLen = 16384;
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.set_prefill_backend(runtime::PrefillBackend::L0);
  eng.reset();
  eng.prefill(ids, kM);
  std::printf("# prefill done: pos = %u, %zu L0 launches\n", eng.pos(), eng.prefill_launches());

  const runtime::PrefillScratch* pf = eng.prefill_scratch();
  if (pf == nullptr) throw std::runtime_error("no prefill scratch after prefill()");

  const loader::DeviceWeight& w =
      eng.model().linears.at({layer, model::LinearId::GateUp});
  if (w.shape.K != kK || w.shape.N != kN || w.shape.layout != 0)
    throw std::runtime_error("layer " + std::to_string(layer) + " gate||up is not the expected "
                             "K=5120 N=34816 layout-0 int4 weight");
  if (w.scales == nullptr) throw std::runtime_error("layout-0 weight has no separate scales");

  const size_t act_bytes = size_t(kM) * kK * sizeof(uint16_t);
  const size_t q_bytes = size_t(kK / 8) * kN * sizeof(uint32_t);
  const size_t s_bytes = size_t(kK / kGroup) * kN * sizeof(uint16_t);
  if (w.mem.size() < q_bytes || w.scales->size() < s_bytes)
    throw std::runtime_error("device weight smaller than the layout-0 shape implies");

  std::vector<uint16_t> act(size_t(kM) * kK);
  std::vector<uint32_t> qw(size_t(kK / 8) * kN);
  std::vector<uint16_t> sc(size_t(kK / kGroup) * kN);
  // `mixer_out` is bf16 [kC][6144]; the post-attention norm writes the MLP's A
  // operand into it at pitch 5120 (runtime/prefill/step.cc, "the normed
  // activations go to `mixer_out`, not `x`"), so the first kM*kK words ARE the
  // gate||up input of the last layer this chunk walked.
  download(ctx, act.data(), pf->mixer_out.ptr(), act_bytes);
  download(ctx, qw.data(), w.mem.ptr(), q_bytes);
  download(ctx, sc.data(), w.scales->ptr(), s_bytes);

  // A captured activation that is all zeros, or is obviously not an RMSNorm
  // output, means the wrong buffer was read; print the evidence rather than
  // trusting the offset.
  double amax = 0.0, asum = 0.0;
  size_t nz = 0;
  for (uint16_t h : act) {
    const double v = std::fabs(double(common::bf16_to_f32(h)));
    amax = std::max(amax, v);
    asum += v;
    nz += (v != 0.0);
  }
  std::printf("# captured activations: max|x| %.6g, mean|x| %.6g, non-zero %zu/%zu\n", amax,
              asum / double(act.size()), nz, act.size());
  if (nz * 2 < act.size()) throw std::runtime_error("captured activations are mostly zero");

  Header h{};
  std::memcpy(h.magic, "B70W4A8", 8);
  h.version = 1; h.layer = layer; h.M = kM; h.K = kK; h.N = kN; h.group = kGroup;
  std::ofstream f(out_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + out_path);
  f.write(reinterpret_cast<const char*>(&h), sizeof h);
  f.write(reinterpret_cast<const char*>(act.data()), std::streamsize(act_bytes));
  f.write(reinterpret_cast<const char*>(qw.data()), std::streamsize(q_bytes));
  f.write(reinterpret_cast<const char*>(sc.data()), std::streamsize(s_bytes));
  f.close();
  std::printf("# wrote %s (%zu bytes): layer %u gate||up weights + the chunk's activations\n",
              out_path.c_str(), sizeof h + act_bytes + q_bytes + s_bytes, layer);
  return 0;
}

// --------------------------------------------------------------------------
// capture-down: the same, for the LAST layer's down projection
// --------------------------------------------------------------------------
// After prefill, `PrefillScratch::x` still holds the last layer's down A
// operand, silu(gate) * up, bf16 [C][17408] (runtime/prefill/step.cc). One
// slice of it is gone: step_head writes the final norm's ONE output row into x
// at element offset last_row * 5120, so the rows that range touches are
// recorded in the header (`pad` = first clobbered row, `version` 2) for the
// reader to drop. Only the last layer is reachable this way; other layers'
// activations are overwritten by the walk.
int capture_down(const std::string& snap, const std::string& ids_path,
                 const std::string& out_path) {
  constexpr uint32_t kDK = 17408, kDN = 5120, kLast = 63;
  std::printf("# W4A8 probe -- capture-down (layer %u down projection)\n", kLast);
  std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));

  std::vector<uint32_t> ids;
  {
    std::ifstream f(ids_path);
    if (!f) throw std::runtime_error("cannot open ids file: " + ids_path);
    for (long long v; f >> v;) ids.push_back(uint32_t(v));
  }
  if (ids.size() < kM) throw std::runtime_error("prompt has only " + std::to_string(ids.size()) +
                                                " ids; need " + std::to_string(kM));
  ids.resize(kM);

  l0::Context ctx(0);
  constexpr uint32_t kMaxLen = 16384;
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen);
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.set_prefill_backend(runtime::PrefillBackend::L0);
  eng.reset();
  eng.prefill(ids, kM);
  std::printf("# prefill done: pos = %u\n", eng.pos());

  const runtime::PrefillScratch* pf = eng.prefill_scratch();
  if (pf == nullptr) throw std::runtime_error("no prefill scratch after prefill()");
  const loader::DeviceWeight& w = eng.model().linears.at({kLast, model::LinearId::Down});
  if (w.shape.K != kDK || w.shape.N != kDN || w.shape.layout != 0 || w.scales == nullptr)
    throw std::runtime_error("layer 63 down is not the expected K=17408 N=5120 layout-0 int4");

  const size_t act_bytes = size_t(kM) * kDK * sizeof(uint16_t);
  const size_t q_bytes = size_t(kDK / 8) * kDN * sizeof(uint32_t);
  const size_t s_bytes = size_t(kDK / kGroup) * kDN * sizeof(uint16_t);
  std::vector<uint16_t> act(size_t(kM) * kDK);
  std::vector<uint32_t> qw(size_t(kDK / 8) * kDN);
  std::vector<uint16_t> sc(size_t(kDK / kGroup) * kDN);
  download(ctx, act.data(), pf->x.ptr(), act_bytes);
  download(ctx, qw.data(), w.mem.ptr(), q_bytes);
  download(ctx, sc.data(), w.scales->ptr(), s_bytes);

  const uint32_t clobber_row = uint32_t((size_t(kM - 1) * 5120) / kDK);
  double amax = 0.0, asum = 0.0;
  size_t nz = 0;
  for (uint16_t h : act) {
    const double v = std::fabs(double(common::bf16_to_f32(h)));
    amax = std::max(amax, v);
    asum += v;
    nz += (v != 0.0);
  }
  std::printf("# captured activations: max|x| %.6g, mean|x| %.6g, non-zero %zu/%zu; "
              "rows from %u clobbered by step_head\n",
              amax, asum / double(act.size()), nz, act.size(), clobber_row);
  if (nz * 2 < act.size()) throw std::runtime_error("captured activations are mostly zero");

  Header h{};
  std::memcpy(h.magic, "B70W4A8", 8);
  h.version = 2; h.layer = kLast; h.M = kM; h.K = kDK; h.N = kDN; h.group = kGroup;
  h.pad = clobber_row;
  std::ofstream f(out_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + out_path);
  f.write(reinterpret_cast<const char*>(&h), sizeof h);
  f.write(reinterpret_cast<const char*>(act.data()), std::streamsize(act_bytes));
  f.write(reinterpret_cast<const char*>(qw.data()), std::streamsize(q_bytes));
  f.write(reinterpret_cast<const char*>(sc.data()), std::streamsize(s_bytes));
  std::printf("# wrote %s: layer %u down weights + the chunk's activations\n", out_path.c_str(),
              kLast);
  return 0;
}

// --------------------------------------------------------------------------
// the error characterisation of design §5
// --------------------------------------------------------------------------
struct ErrStat {
  double max_rel = 0.0, max_rel_ref = 0.0, max_rel_got = 0.0;
  size_t max_rel_row = 0, max_rel_col = 0;
  double max_rel_sig = 0.0;          // restricted to |ref| >= 1% of the row RMS
  double mean_rel_sig = 0.0;
  size_t sig_count = 0;
  double max_abs = 0.0;
  double l2_num = 0.0, l2_den = 0.0;     // global relative L2
  double worst_cos = 2.0;
  size_t worst_cos_row = 0;
  double mean_cos = 0.0, min_cos = 2.0;
  size_t argmax_moved = 0;               // rows whose max-value column changed
  double top8_overlap = 0.0;             // mean |top8 ∩ top8| / 8 over rows
};

void row_error(const float* ref, const float* got, size_t n, size_t row, ErrStat& e) {
  double ss = 0.0;
  for (size_t j = 0; j < n; ++j) ss += double(ref[j]) * double(ref[j]);
  const double rms = std::sqrt(ss / double(n));
  const double floor = 0.01 * rms;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t j = 0; j < n; ++j) {
    const double r = ref[j], g = got[j];
    const double d = std::fabs(g - r);
    e.max_abs = std::max(e.max_abs, d);
    e.l2_num += d * d;
    e.l2_den += r * r;
    if (r != 0.0) {
      const double rel = d / std::fabs(r);
      if (rel > e.max_rel) {
        e.max_rel = rel; e.max_rel_ref = r; e.max_rel_got = g;
        e.max_rel_row = row; e.max_rel_col = j;
      }
      if (std::fabs(r) >= floor) {
        e.max_rel_sig = std::max(e.max_rel_sig, rel);
        e.mean_rel_sig += rel;
        ++e.sig_count;
      }
    }
    dot += r * g; na += r * r; nb += g * g;
  }
  const double cos = (na > 0.0 && nb > 0.0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 1.0;
  e.mean_cos += cos;
  if (cos < e.min_cos) { e.min_cos = cos; e.worst_cos_row = row; }
  // the argmax-relevant ordering: the row's largest value, and its top-8 set
  size_t ar = 0, ag = 0;
  for (size_t j = 1; j < n; ++j) {
    if (ref[j] > ref[ar]) ar = j;
    if (got[j] > got[ag]) ag = j;
  }
  if (ar != ag) ++e.argmax_moved;
  std::vector<uint32_t> ir(n), ig(n);
  std::iota(ir.begin(), ir.end(), 0u);
  std::iota(ig.begin(), ig.end(), 0u);
  std::partial_sort(ir.begin(), ir.begin() + 8, ir.end(),
                    [&](uint32_t a, uint32_t b) { return ref[a] > ref[b]; });
  std::partial_sort(ig.begin(), ig.begin() + 8, ig.end(),
                    [&](uint32_t a, uint32_t b) { return got[a] > got[b]; });
  std::vector<uint32_t> A(ir.begin(), ir.begin() + 8), B(ig.begin(), ig.begin() + 8);
  std::sort(A.begin(), A.end());
  std::sort(B.begin(), B.end());
  std::vector<uint32_t> inter;
  std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(inter));
  e.top8_overlap += double(inter.size()) / 8.0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--capture") {
      if (argc < 5) {
        std::fprintf(stderr,
                     "usage: probe_w4a8 --capture <snapshot> <ids-file> <out-file> [layer]\n");
        return 2;
      }
      return capture(argv[2], argv[3], argv[4],
                     argc > 5 ? uint32_t(std::atoi(argv[5])) : kDefaultLayer);
    }
    if (argc >= 5 && std::string(argv[1]) == "--capture-down")
      return capture_down(argv[2], argv[3], argv[4]);
    if (argc < 2) {
      std::fprintf(stderr,
                   "usage: probe_w4a8 <capture-file>\n"
                   "       probe_w4a8 --capture-down <snapshot> <ids-file> <out-file>\n"
                   "       probe_w4a8 --capture <snapshot> <ids-file> <out-file> [layer]\n");
      return 2;
    }

    // --- the inputs -------------------------------------------------------
    const std::string in_path = argv[1];
    std::ifstream f(in_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + in_path);
    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof h);
    if (std::memcmp(h.magic, "B70W4A8", 8) != 0 || h.M != kM || h.K != kK || h.N != kN ||
        h.group != kGroup)
      throw std::runtime_error(in_path + " is not this probe's capture file");
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

    std::printf("# W4A8 probe (design 2026-09-22 §4/§5/§6)\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    l0::Context ctx(0);
    std::printf("# L0 device: %s\n", ctx.name().c_str());
    std::printf("# inputs: %s -- REAL layer-%u gate||up weights and the REAL activations one "
                "prefill chunk fed them\n", in_path.c_str(), h.layer);
    std::printf("# shape: K=%u N=%u M=%u group=%u; %.1f GFLOP per call\n", kK, kN, kM, kGroup,
                kGflop);
    std::printf("# pre-registered: break-even %.3f ms; worth a study <= %.2f ms (>= 1.53x); "
                "the instruction's promise %.2f ms (2.00x)\n",
                kBreakEvenMs, kStudyMs, kPromiseMs);

    runtime::prefill::Context cx(ctx);

    // --- the activation dynamic range (design §5) -------------------------
    std::vector<double> dyn(kM);
    {
      std::vector<float> a(kK);
      for (uint32_t m = 0; m < kM; ++m) {
        for (uint32_t k = 0; k < kK; ++k)
          a[k] = std::fabs(common::bf16_to_f32(act[size_t(m) * kK + k]));
        const double mx = *std::max_element(a.begin(), a.end());
        std::nth_element(a.begin(), a.begin() + kK / 2, a.end());
        const double med = a[kK / 2];
        dyn[m] = med > 0.0 ? mx / med : 0.0;
      }
    }

    // --- device buffers ---------------------------------------------------
    const size_t c_bytes = size_t(kM) * kN * sizeof(float);
    l0::Mem da(ctx, l0::MemKind::Device, act_bytes);
    l0::Mem dq(ctx, l0::MemKind::Device, q_bytes);
    l0::Mem ds(ctx, l0::MemKind::Device, s_bytes);
    l0::Mem slab(ctx, l0::MemKind::Device, size_t(kK) * kNs * sizeof(uint16_t));
    l0::Mem c_ctl(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_w4(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem xq(ctx, l0::MemKind::Device, size_t(kM) * kK);          // int8 [M][K]
    l0::Mem xs(ctx, l0::MemKind::Device, size_t(kM) * sizeof(float));
    l0::Mem sig(ctx, l0::MemKind::Shared, 64);
    {
      l0::CmdList up = l0::CmdList::immediate(ctx);
      up.copy(da.ptr(), act.data(), act_bytes);
      up.copy(dq.ptr(), qw.data(), q_bytes);
      up.copy(ds.ptr(), sc.data(), s_bytes);
      up.fill(sig.ptr(), 0u, 64);
    }

    // --- kernels ----------------------------------------------------------
    l0::Module m_dq(ctx, kernels::path(kernels::pf_dequant_slab_variant(kK, kN, 0)));
    l0::Kernel k_dq = m_dq.kernel("pf_dequant_slab");
    l0::Module m_gemm(ctx, kernels::path(kernels::pf_gemm_variant(false)));
    l0::Kernel k_gemm = m_gemm.kernel("pf_gemm");
    l0::Module m_w4(ctx, kernels::path("pw4a8"));
    l0::Kernel k_w4 = m_w4.kernel("pw4a8_gemm");
    l0::Kernel k_qt = m_w4.kernel("pw4a8_quant");

    l0::EventPool pool(ctx, 2 * kSlabs);
    std::vector<l0::Event> ev;
    ev.reserve(2 * kSlabs);
    for (uint32_t i = 0; i < 2 * kSlabs; ++i) ev.emplace_back(pool, i);
    const l0::TimerCalib calib = pool.calib();

    // --- the three arms ---------------------------------------------------
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
    auto run_w4a8 = [&]() -> size_t {
      cx.launch(k_w4, kM / kTile, kN / kW4Tile, 1,
                {PtrArg(xq.ptr()), PtrArg(xs.ptr()), PtrArg(dq.ptr()), PtrArg(ds.ptr()),
                 PtrArg(c_w4.ptr()), PtrArg(sig.ptr()), arg_val(kM), arg_val(kK), arg_val(kN),
                 arg_val(ldxq), arg_val(ldc)},
                &ev[0]);
      cx.wait();
      return size_t(1);
    };

    // --- DISPATCH PROOF, printed before the timed arm runs ----------------
    std::printf("\n## dispatch proof\n\n");
    std::printf("| what | value |\n|---|---|\n");
    std::printf("| module the W4A8 arm opened | `%s` |\n", kernels::path("pw4a8").c_str());
    std::printf("| `zeKernelGetName` on the handle `launch()` receives | **`%s`** |\n",
                kernel_name(k_w4.handle()).c_str());
    std::printf("| `zeKernelGetName`, quantiser handle | `%s` |\n",
                kernel_name(k_qt.handle()).c_str());
    std::printf("| `zeKernelGetName`, control GEMM handle | `%s` |\n",
                kernel_name(k_gemm.handle()).c_str());
    std::printf("| `zeKernelGetName`, control dequant handle | `%s` |\n",
                kernel_name(k_dq.handle()).c_str());
    if (kernel_name(k_w4.handle()) != "pw4a8_gemm")
      throw std::runtime_error("dispatch proof failed: the W4A8 handle is not pw4a8_gemm");

    // correctness/value pass first: a fast wrong kernel is a failed probe
    run_control();
    run_quant();
    run_w4a8();
    const uint32_t sig_v = *sig.as<uint32_t>();
    std::printf("| signature word the executed kernel wrote | **0x%08X** (expected 0x%08X = "
                "'W4A8') |\n", sig_v, kSigMagic);
    if (sig_v != kSigMagic)
      throw std::runtime_error("dispatch proof failed: pw4a8_gemm did not run");
    std::fflush(stdout);

    // --- an integer-exact CPU oracle over a slice -------------------------
    // Proves the kernel computes design §3's arithmetic (and, with it, that the
    // DPAS operand layouts this kernel assumes are the ones the hardware uses):
    // a transposed or mis-packed operand cannot agree to a few ulp by accident.
    std::vector<int8_t> xq_h(size_t(kM) * kK);
    std::vector<float> xs_h(kM);
    download(ctx, xq_h.data(), xq.ptr(), xq_h.size());
    download(ctx, xs_h.data(), xs.ptr(), kM * sizeof(float));
    {
      constexpr uint32_t kRows = 32, kCols = 32;
      std::vector<float> got(size_t(kRows) * kN);
      for (uint32_t m = 0; m < kRows; ++m)
        download(ctx, got.data() + size_t(m) * kN,
                 c_w4.as<float>() + size_t(m) * kN, size_t(kN) * sizeof(float));
      double worst = 0.0, worst_ref = 0.0;
      for (uint32_t m = 0; m < kRows; ++m)
        for (uint32_t n = 0; n < kCols; ++n) {
          float acc = 0.0f;
          for (uint32_t g = 0; g < kK / kGroup; ++g) {
            int32_t s32 = 0;
            for (uint32_t j = 0; j < kGroup; ++j) {
              const uint32_t k = g * kGroup + j;
              const uint32_t word = qw[size_t(k / 8) * kN + n];
              const int q = int((word >> (4 * (k % 8))) & 0xFu) - 8;
              s32 += int32_t(xq_h[size_t(m) * kK + k]) * q;
            }
            const float ws = common::f16_to_f32(sc[size_t(g) * kN + n]);
            acc = std::fma(float(s32), ws * xs_h[m], acc);
          }
          const double d = std::fabs(double(acc) - double(got[size_t(m) * kN + n]));
          if (d > worst) { worst = d; worst_ref = acc; }
        }
      std::printf("\n## the kernel computes design §3's arithmetic (integer-exact CPU oracle, "
                  "%u x %u slice)\n\n", kRows, kCols);
      std::printf("max |device - host| over the slice: **%.6g** (host value there %.6g, "
                  "relative %.3g)\n", worst, worst_ref,
                  worst_ref != 0.0 ? worst / std::fabs(worst_ref) : 0.0);
      if (worst_ref != 0.0 && worst / std::fabs(worst_ref) > 1e-5)
        throw std::runtime_error("the W4A8 kernel does not reproduce the host oracle");
    }

    // --- rates ------------------------------------------------------------
    const Timed t_ctl = best_of(run_control, ev, calib);
    const Timed t_qt = best_of(run_quant, ev, calib);
    const Timed t_w4 = best_of(run_w4a8, ev, calib);
    const double w4_total = t_w4.sum_ms + t_qt.sum_ms;

    std::printf("\n## the three rates (measured; L0 kernel timestamps, best of %d after one "
                "discarded warm-up)\n\n", kReplays);
    std::printf("| arm | launches | kernel ms | TFLOP/s or GB/s | vs the control |\n");
    std::printf("|---|---:|---:|---:|---:|\n");
    std::printf("| **bf16 two-pass control** (34 slabs) | %u | **%.3f** | %.2f TFLOP/s | 1.000x |\n",
                2 * kSlabs, t_ctl.sum_ms, t_ctl.tflops());
    std::printf("| &nbsp;&nbsp;of which `pf_dequant_slab` | %u | %.3f | - | - |\n", kSlabs,
                t_ctl.even_ms);
    std::printf("| &nbsp;&nbsp;of which `pf_gemm_T0` | %u | %.3f | - | - |\n", kSlabs,
                t_ctl.odd_ms);
    std::printf("| **W4A8 GEMM** (`pw4a8_gemm`) | 1 | **%.3f** | %.2f TOP/s | **%.3fx** |\n",
                t_w4.sum_ms, kGflop / t_w4.sum_ms, t_ctl.sum_ms / t_w4.sum_ms);
    std::printf("| **activation quantiser** (`pw4a8_quant`) | 1 | **%.3f** | %.1f GB/s | - |\n",
                t_qt.sum_ms,
                (double(kM) * kK * 3.0 / 1e6) / t_qt.sum_ms);   // 2 B read + 1 B written
    std::printf("| **W4A8 total** (quantiser + GEMM) | 2 | **%.3f** | %.2f TOP/s | **%.3fx** |\n",
                w4_total, kGflop / w4_total, t_ctl.sum_ms / w4_total);
    std::printf("\nAgainst the pre-registered bars: break-even %.3f ms, worth a study <= %.2f ms "
                "(>= 1.53x), the instruction's promise %.2f ms (2.00x). The gap between %.3f ms "
                "and %.2f ms is the per-group rescale.\n",
                kBreakEvenMs, kStudyMs, kPromiseMs, w4_total, kPromiseMs);
    std::fflush(stdout);

    // --- error (design §5) -------------------------------------------------
    ErrStat e;
    {
      constexpr uint32_t kRowBlock = 128;
      const size_t block = size_t(kRowBlock) * kN * sizeof(float);
      l0::Mem hr(ctx, l0::MemKind::Host, block);
      l0::Mem hg(ctx, l0::MemKind::Host, block);
      for (uint32_t m0 = 0; m0 < kM; m0 += kRowBlock) {
        download(ctx, hr.ptr(), c_ctl.as<float>() + size_t(m0) * kN, block);
        download(ctx, hg.ptr(), c_w4.as<float>() + size_t(m0) * kN, block);
        for (uint32_t r = 0; r < kRowBlock; ++r)
          row_error(hr.as<float>() + size_t(r) * kN, hg.as<float>() + size_t(r) * kN, kN,
                    m0 + r, e);
      }
    }
    e.mean_rel_sig /= double(e.sig_count ? e.sig_count : 1);
    e.mean_cos /= double(kM);

    std::printf("\n## error against the bf16 two-pass result, on the same real inputs "
                "(design §5)\n\n");
    std::printf("| metric | value |\n|---|---:|\n");
    std::printf("| max relative error, all %zu outputs | **%.6g** (ref %.6g, W4A8 %.6g, at row "
                "%zu col %zu) |\n", size_t(kM) * kN, e.max_rel, e.max_rel_ref, e.max_rel_got,
                e.max_rel_row, e.max_rel_col);
    std::printf("| max relative error, outputs above 1%% of their row RMS | **%.6g** |\n",
                e.max_rel_sig);
    std::printf("| mean relative error over those %zu outputs | **%.6g** |\n", e.sig_count,
                e.mean_rel_sig);
    std::printf("| max absolute error | %.6g |\n", e.max_abs);
    std::printf("| global relative L2, ‖W4A8-bf16‖/‖bf16‖ | **%.6g** |\n",
                std::sqrt(e.l2_num / e.l2_den));
    std::printf("| per-row cosine: mean | %.9f |\n", e.mean_cos);
    std::printf("| per-row cosine: **worst row** | **%.9f** (row %zu) |\n", e.min_cos,
                e.worst_cos_row);
    std::printf("| rows whose largest output moved to another column | %zu / %u |\n",
                e.argmax_moved, kM);
    std::printf("| mean top-8 column overlap per row | %.4f / 1.0 |\n", e.top8_overlap / kM);

    // --- the activation dynamic range -------------------------------------
    std::vector<double> d = dyn;
    std::sort(d.begin(), d.end());
    std::printf("\n## activation dynamic range per token, max|x| / median|x| (design §5)\n\n");
    std::printf("| statistic over the chunk's %u tokens | value |\n|---|---:|\n", kM);
    std::printf("| min | %.1f |\n", d.front());
    std::printf("| median | **%.1f** |\n", d[kM / 2]);
    std::printf("| p90 | %.1f |\n", d[size_t(0.90 * kM)]);
    std::printf("| p99 | %.1f |\n", d[size_t(0.99 * kM)]);
    std::printf("| **max** | **%.1f** |\n", d.back());
    {
      double lo = 1e30, hi = 0.0;
      for (float s : xs_h) { lo = std::min(lo, double(s)); hi = std::max(hi, double(s)); }
      std::printf("| per-token int8 scale max|x|/127: min / max | %.6g / %.6g |\n", lo, hi);
    }

    // --- §7 decision rule --------------------------------------------------
    const double mult = t_ctl.sum_ms / w4_total;
    std::printf("\n## design §7 decision rule\n\n");
    std::printf("W4A8 total %.3f ms = **%.3fx** the %.3f ms control (bars: 1.53x / 2.00x).\n",
                w4_total, mult, t_ctl.sum_ms);
    if (mult < 1.53)
      std::printf("**§7 branch 1: REJECTED.** Below 1.53x. Recorded, not tuned. W4A16 stands "
                  "and the parity program's S2-S5 remain the whole plan.\n");
    else
      std::printf("**§7 branch 2 or 3: >= 1.53x.** The error table above decides which: small "
                  "error -> write the adoption spec (proving strategy first); large error -> "
                  "report both numbers and stop.\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_w4a8 FAILED: %s\n", ex.what());
    return 2;
  }
}
