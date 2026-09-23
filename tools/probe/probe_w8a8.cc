// The W8A8 SPEED probe (2026-09-23): i8 x i8 k32 DPAS, per-channel int8
// weights requantised per slab from the checkpoint's int4 g64.
//
// Why this exists: the CPU accuracy study (docs/probe-w4a8-2026-09-23.md)
// rejected every int4-weight W4A8 variant on accuracy and left exactly one
// candidate: per-channel int8 weights + per-token int8 activations after a
// 1024-block Hadamard rotation. The earlier 1.75x per-channel rate was the
// i8 x i4 instruction; this measures the i8 x i8 one, which loads twice the
// weight bytes, plus the two costs the path adds: the int4 -> int8 slab
// requant and the rotating activation quantiser.
//
// Inputs: a capture file from `probe_w4a8 --capture` (gate||up, K 5120,
// N 34816) or `--capture-down` (down, K 17408, N 5120): the real layer-63
// weights and the activations one prefill chunk fed them.
//
// Arms (L0 kernel timestamps, best of 5 after one discarded warm-up):
//   control     pf_dequant_slab + pf_gemm_T0 per 1024-column slab, the production pair
//   quant       pw8_quant, per-token int8, no rotation
//   quant_had   pw8_quant_had, the same after the 1024-block Hadamard
//   gemm_full   pw8_gemm over all N on weights requantised beforehand (GEMM ceiling)
//   two_pass    pw8_requant + pw8_gemm per 1024-column slab, the production shape
//
// Correctness, all checked before any time is printed:
//   * the device's int8 weights equal a CPU requant of the same nibbles, every element
//   * pw8_gemm equals an integer-exact CPU oracle on a 32 x 32 slice at full K
//   * two_pass equals gemm_full on every output word
//   * pw8_quant_had equals a CPU Walsh-Hadamard + quant within 1 LSB on 64 rows
//   * two_pass against the bf16 control has the relative L2 the CPU study
//     predicts for this variant (no rotation): about 2.8 % on gate||up
// Later additions (section 14 of the doc): v2 helpers, the rotating weight
// requant (rot1..rot4), interleaved paired timing, and the end-to-end check of
// the rotated path against the bf16 control.
//
// PROBE-ONLY: no production file is edited.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
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

constexpr uint32_t kFileM = 2048;          // rows in a capture file (one prefill chunk)
uint32_t kM = 2048;                         // rows timed: B70_W8A8_M, a multiple of 256 <= 2048
constexpr uint32_t kGroup = 64;
constexpr uint32_t kNs = 1024;              // kernels::kPfSlabWidth
constexpr uint32_t kTile = 256;             // pf_gemm WG_M / WG_N, pw8_gemm WG_M
constexpr uint32_t kW8Tile = 128;           // pw8_gemm WG_N
constexpr int kReplays = 5;
constexpr double kStudyMult = 1.53;         // the bar the W4A8 probe pre-registered

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

template <class F>
Timed best_of(F run, std::vector<l0::Event>& ev, const l0::TimerCalib& calib) {
  const uint64_t mask = calib.mask();
  Timed best;
  bool have = false;
  for (int r = -1; r < kReplays; ++r) {
    for (l0::Event& e : ev) e.reset();
    const size_t n = run();
    Timed cur;
    for (size_t i = 0; i < n; ++i) {
      const double us =
          double((raw_end(ev[i], mask) - raw_start(ev[i], mask)) & mask) / calib.cycles_per_us;
      cur.sum_ms += us;
      if (i % 2 == 0) cur.even_ms += us; else cur.odd_ms += us;
    }
    cur.sum_ms /= 1000.0;
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

int8_t sat_rte(float v) {
  const float r = std::nearbyint(v);          // default rounding mode: to nearest even
  return int8_t(std::max(-128.0f, std::min(127.0f, r)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: probe_w8a8 <capture-file>\n");
      return 2;
    }
    const std::string in_path = argv[1];
    std::ifstream f(in_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + in_path);
    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof h);
    if (const char* mv = std::getenv("B70_W8A8_M")) kM = uint32_t(std::atoi(mv));
    if (kM == 0 || kM > kFileM || kM % 256)
      throw std::runtime_error("B70_W8A8_M must be a multiple of 256 in 256..2048");
    if (std::memcmp(h.magic, "B70W4A8", 8) != 0 || h.M != kFileM || h.group != kGroup)
      throw std::runtime_error(in_path + " is not a W4A8 probe capture file");
    const uint32_t K = h.K, N = h.N;
    if (K % 1024 || N % kNs || K > 20480)
      throw std::runtime_error("shape not supported: K must be a multiple of 1024, N of 1024");
    const uint32_t kSlabs = N / kNs;
    const double gflop = 2.0 * kM * double(K) * double(N) / 1e9;

    const size_t act_bytes = size_t(kM) * K * 2;
    const size_t q_bytes = size_t(K / 8) * N * 4;
    const size_t s_bytes = size_t(K / kGroup) * N * 2;
    std::vector<uint16_t> act(size_t(kM) * K);
    std::vector<uint32_t> qw(size_t(K / 8) * N);
    std::vector<uint16_t> sc(size_t(K / kGroup) * N);
    // the first kM rows of the capture's kFileM: a shorter prompt's chunk
    f.read(reinterpret_cast<char*>(act.data()), std::streamsize(act_bytes));
    f.seekg(std::streamoff(size_t(kFileM - kM) * K * 2), std::ios::cur);
    f.read(reinterpret_cast<char*>(qw.data()), std::streamsize(q_bytes));
    f.read(reinterpret_cast<char*>(sc.data()), std::streamsize(s_bytes));
    if (!f) throw std::runtime_error(in_path + " is truncated");

    std::printf("# W8A8 SPEED probe: i8 x i8 k32, per-channel int8 weights\n");
    std::printf("# ZE_AFFINITY_MASK=%s\n", env_or_unset("ZE_AFFINITY_MASK"));
    l0::Context ctx(0);
    std::printf("# L0 device: %s\n", ctx.name().c_str());
    std::printf("# input %s: layer %u, K=%u N=%u M=%u (of %u), %.1f GFLOP per call, %u slabs\n",
                in_path.c_str(), h.layer, K, N, kM, kFileM, gflop, kSlabs);

    // --- host-side: the column scales (a load-time property of the weights),
    // the CPU requant the device must reproduce, the Hadamard signs ---------
    std::vector<float> cs(N, 0.0f);
    for (uint32_t r = 0; r < K / 8; ++r)
      for (uint32_t n = 0; n < N; ++n) {
        const uint32_t word = qw[size_t(r) * N + n];
        const float s = common::f16_to_f32(sc[size_t(r / 8) * N + n]);
        for (uint32_t i = 0; i < 8; ++i) {
          const float v = float(int((word >> (4 * i)) & 0xFu) - 8) * s;
          cs[n] = std::max(cs[n], std::fabs(v));
        }
      }
    for (float& c : cs) c = c > 0.0f ? c / 127.0f : 1.0f;
    std::vector<float> sgn(K);
    {
      std::mt19937 rng(1024);
      for (float& v : sgn) v = (rng() & 1u) ? 1.0f : -1.0f;
    }

    runtime::prefill::Context cx(ctx);
    const size_t c_bytes = size_t(kM) * N * 4;
    const size_t w8_bytes = size_t(K / 4) * N * 4;
    l0::Mem da(ctx, l0::MemKind::Device, act_bytes);
    l0::Mem dq(ctx, l0::MemKind::Device, q_bytes);
    l0::Mem ds(ctx, l0::MemKind::Device, s_bytes);
    l0::Mem slab(ctx, l0::MemKind::Device, size_t(K) * kNs * 2);     // bf16 control slab
    l0::Mem c_ctl(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem dcs(ctx, l0::MemKind::Device, size_t(N) * 4);
    l0::Mem dsg(ctx, l0::MemKind::Device, size_t(K) * 4);
    l0::Mem w8slab(ctx, l0::MemKind::Device, size_t(K / 4) * kNs * 4);
    l0::Mem w8full(ctx, l0::MemKind::Device, w8_bytes);
    l0::Mem c_full(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_two(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem xq(ctx, l0::MemKind::Device, size_t(kM) * K);
    l0::Mem xs(ctx, l0::MemKind::Device, size_t(kM) * 4);
    l0::Mem xqh(ctx, l0::MemKind::Device, size_t(kM) * K);
    l0::Mem xsh(ctx, l0::MemKind::Device, size_t(kM) * 4);
    l0::Mem sig(ctx, l0::MemKind::Shared, 64);
    // v2 buffers
    std::vector<float> inv(N);
    for (uint32_t n = 0; n < N; ++n) inv[n] = 1.0f / cs[n];
    l0::Mem dinv(ctx, l0::MemKind::Device, size_t(N) * 4);
    l0::Mem w8full2(ctx, l0::MemKind::Device, w8_bytes);
    l0::Mem c_full2(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem c_two2(ctx, l0::MemKind::Device, c_bytes);
    l0::Mem xq2(ctx, l0::MemKind::Device, size_t(kM) * K);
    l0::Mem xs2(ctx, l0::MemKind::Device, size_t(kM) * 4);
    l0::Mem xqh2(ctx, l0::MemKind::Device, size_t(kM) * K);
    l0::Mem xsh2(ctx, l0::MemKind::Device, size_t(kM) * 4);
    // rotated weights: W'_blk = H (d * w_blk) / 32 per column, the CPU reference
    // the device must reproduce bit for bit, and its per-channel scales
    std::vector<int8_t> wrot_ref(size_t(K) * N);
    std::vector<float> cs_rot(N), inv_rot(N);
    {
      std::vector<float> col(K);
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t k = 0; k < K; ++k) {
          const uint32_t word = qw[size_t(k / 8) * N + n];
          col[k] = (float(int((word >> (4 * (k % 8))) & 0xFu) - 8) *
                    common::f16_to_f32(sc[size_t(k / 64) * N + n])) * sgn[k];
        }
        for (uint32_t b = 0; b < K; b += 1024)
          for (uint32_t hh = 1; hh < 1024; hh <<= 1)
            for (uint32_t i = 0; i < 1024; i += 2 * hh)
              for (uint32_t j = i; j < i + hh; ++j) {
                const float a = col[b + j], e = col[b + j + hh];
                col[b + j] = a + e;
                col[b + j + hh] = a - e;
              }
        float amax = 0.0f;
        for (float& v : col) { v *= 1.0f / 32.0f; amax = std::max(amax, std::fabs(v)); }
        cs_rot[n] = amax > 0.0f ? amax / 127.0f : 1.0f;
        inv_rot[n] = 1.0f / cs_rot[n];
        for (uint32_t k = 0; k < K; ++k) wrot_ref[size_t(k) * N + n] = sat_rte(col[k] * inv_rot[n]);
      }
    }
    std::vector<uint32_t> sbits(K / 32, 0u);
    for (uint32_t k = 0; k < K; ++k)
      if (sgn[k] < 0.0f) sbits[k / 32] |= 1u << (k % 32);
    l0::Mem dsbits(ctx, l0::MemKind::Device, size_t(K / 32) * 4);
    // A_g = H64 diag(d_g), int8 row-major [K/64][64 j'][64 j]
    std::vector<int8_t> amat(size_t(K) * 64);
    for (uint32_t gi = 0; gi < K / 64; ++gi)
      for (uint32_t jp = 0; jp < 64; ++jp)
        for (uint32_t j = 0; j < 64; ++j)
          amat[(size_t(gi) * 64 + jp) * 64 + j] =
              int8_t(((__builtin_popcount(jp & j) & 1) ? -1 : 1) * (sgn[64 * gi + j] < 0.0f ? -1 : 1));
    l0::Mem damat(ctx, l0::MemKind::Device, amat.size());
    l0::Mem dcs_rot(ctx, l0::MemKind::Device, size_t(N) * 4);
    l0::Mem dinv_rot(ctx, l0::MemKind::Device, size_t(N) * 4);
    l0::Mem w8rot(ctx, l0::MemKind::Device, w8_bytes);
    l0::Mem c_rot(ctx, l0::MemKind::Device, c_bytes);
    {
      l0::CmdList up = l0::CmdList::immediate(ctx);
      up.copy(da.ptr(), act.data(), act_bytes);
      up.copy(dq.ptr(), qw.data(), q_bytes);
      up.copy(ds.ptr(), sc.data(), s_bytes);
      up.copy(dcs.ptr(), cs.data(), size_t(N) * 4);
      up.copy(dsg.ptr(), sgn.data(), size_t(K) * 4);
      up.copy(dinv.ptr(), inv.data(), size_t(N) * 4);
      up.copy(dcs_rot.ptr(), cs_rot.data(), size_t(N) * 4);
      up.copy(dsbits.ptr(), sbits.data(), size_t(K / 32) * 4);
      up.copy(damat.ptr(), amat.data(), amat.size());
      up.copy(dinv_rot.ptr(), inv_rot.data(), size_t(N) * 4);
      up.fill(sig.ptr(), 0u, 64);
    }

    // The control's dequant in the layout production uses for this shape
    // (B70_W8A8_CTL_LAYOUT; GDN qkv||z is layout 1). With synthetic weights the
    // control's VALUES then mean nothing; its TIME is the production kernel's.
    const unsigned ctl_layout = std::getenv("B70_W8A8_CTL_LAYOUT")
                                    ? unsigned(std::atoi(std::getenv("B70_W8A8_CTL_LAYOUT"))) : 0u;
    l0::Module m_dq(ctx, kernels::path(kernels::pf_dequant_slab_variant(K, N, ctl_layout)));
    l0::Kernel k_dq = m_dq.kernel("pf_dequant_slab");
    l0::Module m_gemm(ctx, kernels::path(kernels::pf_gemm_variant(false)));
    l0::Kernel k_gemm = m_gemm.kernel("pf_gemm");
    l0::Module m_w8(ctx, kernels::path("pw8a8"));
    l0::Kernel k_rq = m_w8.kernel("pw8_requant");
    l0::Kernel k_g8 = m_w8.kernel("pw8_gemm");
    l0::Kernel k_q = m_w8.kernel("pw8_quant");
    l0::Kernel k_qh = m_w8.kernel("pw8_quant_had");
    const uint32_t nblk = K / 1024;
    if (nblk != 5 && nblk != 6 && nblk != 17)
      throw std::runtime_error("v2 quantisers exist for K = 5120, 6144, 17408");
    const std::string q2_name = "pw8_quant2_" + std::to_string(nblk);
    const std::string qh2_name = "pw8_quant_had2_" + std::to_string(nblk);
    // The helpers come from the 128-GRF build of the same source unless
    // B70_W8A8_HELPERS_256=1 (the GEMM always comes from the 256-GRF build).
    const bool helpers256 = std::getenv("B70_W8A8_HELPERS_256") != nullptr;
    l0::Module m_aux(ctx, kernels::path(helpers256 ? "pw8a8" : "pw8a8_g128"));
    std::printf("# helper kernels from %s\n", helpers256 ? "pw8a8 (256 GRF)" : "pw8a8_g128 (128 GRF)");
    l0::Kernel k_rq2 = m_aux.kernel("pw8_requant2");
    // B70_W8A8_ROT = 1 | 2 | 3 (default 3, the DPAS factorisation, which lives
    // in the 256-GRF build beside the GEMM)
    const char* rot_env = std::getenv("B70_W8A8_ROT");
    const int rot_ver = rot_env ? std::atoi(rot_env) : 3;
    const bool rot_v1 = rot_ver == 1;
    const int rv = rot_env ? rot_ver : 2;   // rot2 measured fastest (rot3/rot4: DPAS factorisation)
    const char* rot_name = rv == 1 ? "pw8_requant_rot" : rv == 2 ? "pw8_requant_rot2"
                           : rv == 3 ? "pw8_requant_rot3" : "pw8_requant_rot4";
    std::printf("# rotating requant: %s\n", rot_name);
    const bool rot_g128 = std::getenv("B70_W8A8_ROT_G128") != nullptr;
    if (rv >= 3) std::printf("# rot%d from the %s build\n", rv, rot_g128 ? "128-GRF" : "256-GRF");
    l0::Kernel k_rqr = (rv >= 3 && !rot_g128 ? m_w8 : m_aux).kernel(rot_name);
    l0::Kernel k_q2 = m_aux.kernel(q2_name.c_str());
    l0::Kernel k_qh2 = m_aux.kernel(qh2_name.c_str());
    const std::pair<l0::Kernel*, const char*> handles[] = {
        {&k_rq, "pw8_requant"}, {&k_g8, "pw8_gemm"}, {&k_q, "pw8_quant"}, {&k_qh, "pw8_quant_had"},
        {&k_rq2, "pw8_requant2"}, {&k_rqr, rot_name}, {&k_q2, q2_name.c_str()}, {&k_qh2, qh2_name.c_str()}};
    for (const auto& [k, want] : handles)
      if (kernel_name(k->handle()) != want)
        throw std::runtime_error(std::string("dispatch proof failed for ") + want);

    l0::EventPool pool(ctx, 2 * kSlabs);
    std::vector<l0::Event> ev;
    ev.reserve(2 * kSlabs);
    for (uint32_t i = 0; i < 2 * kSlabs; ++i) ev.emplace_back(pool, i);
    const l0::TimerCalib calib = pool.calib();

    const uint32_t ldxq = K / 2;
    const uint64_t zero = 0;
    auto run_control = [&]() -> size_t {
      float* out = c_ctl.as<float>();
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        cx.launch(k_dq, kNs / 16, K / 64, 1,
                  {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(slab.ptr()), arg_val(n0)}, &ev[2 * i]);
        cx.launch(k_gemm, kM / kTile, kNs / kTile, 1,
                  {PtrArg(da.ptr()), PtrArg(slab.ptr()), PtrArg(out + n0), arg_val(kM), arg_val(K),
                   arg_val(kNs), arg_val(K), arg_val(kNs), arg_val(N), arg_val(zero),
                   arg_val(zero), arg_val(zero)},
                  &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };
    auto run_quant = [&]() -> size_t {
      cx.launch(k_q, kM, 1, 1,
                {PtrArg(da.ptr()), PtrArg(xq.ptr()), PtrArg(xs.ptr()), arg_val(K), arg_val(K)},
                &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_quant_had = [&]() -> size_t {
      cx.launch(k_qh, kM, 1, 1,
                {PtrArg(da.ptr()), PtrArg(dsg.ptr()), PtrArg(xqh.ptr()), PtrArg(xsh.ptr()),
                 arg_val(K), arg_val(K)},
                &ev[0]);
      cx.wait();
      return 1;
    };
    auto requant = [&](uint32_t n0, uint32_t width, void* out, uint32_t ldo, l0::Event* e) {
      cx.launch(k_rq, width / 16, K / 8 / 16, 1,
                {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(dcs.ptr()), PtrArg(out), arg_val(n0),
                 arg_val(N), arg_val(ldo)},
                e);
    };
    auto gemm8w = [&](const void* a, const void* as, const void* w, const float* wsc, uint32_t n0,
                      uint32_t width, uint32_t ldb, float* out, l0::Event* e) {
      cx.launch(k_g8, kM / kTile, width / kW8Tile, 1,
                {PtrArg(a), PtrArg(as), PtrArg(w),
                 PtrArg(wsc + n0), PtrArg(out + n0), PtrArg(sig.ptr()), arg_val(kM),
                 arg_val(K), arg_val(width), arg_val(ldxq), arg_val(ldb), arg_val(N)},
                e);
    };
    auto gemm8x = [&](const void* a, const void* as, const void* w, uint32_t n0, uint32_t width,
                      uint32_t ldb, float* out, l0::Event* e) {
      gemm8w(a, as, w, dcs.as<float>(), n0, width, ldb, out, e);
    };
    auto gemm8 = [&](const void* w, uint32_t n0, uint32_t width, uint32_t ldb, float* out,
                     l0::Event* e) { gemm8x(xq.ptr(), xs.ptr(), w, n0, width, ldb, out, e); };
    auto requant2 = [&](uint32_t n0, uint32_t width, void* out, uint32_t ldo, l0::Event* e) {
      cx.launch(k_rq2, width / 64, K / 512, 1,
                {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(dinv.ptr()), PtrArg(out), arg_val(n0),
                 arg_val(N), arg_val(ldo)},
                e);
    };
    auto run_quant2 = [&]() -> size_t {
      cx.launch(k_q2, kM, 1, 1,
                {PtrArg(da.ptr()), PtrArg(dsg.ptr()), PtrArg(xq2.ptr()), PtrArg(xs2.ptr()),
                 arg_val(K), arg_val(K)},
                &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_quant_had2 = [&]() -> size_t {
      cx.launch(k_qh2, kM, 1, 1,
                {PtrArg(da.ptr()), PtrArg(dsg.ptr()), PtrArg(xqh2.ptr()), PtrArg(xsh2.ptr()),
                 arg_val(K), arg_val(K)},
                &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_requant2_full = [&]() -> size_t {
      requant2(0, N, w8full2.ptr(), N, &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_gemm_full2 = [&]() -> size_t {
      gemm8x(xq2.ptr(), xs2.ptr(), w8full2.ptr(), 0, N, N, c_full2.as<float>(), &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_two_pass2 = [&]() -> size_t {
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        requant2(n0, kNs, w8slab.ptr(), kNs, &ev[2 * i]);
        gemm8x(xq2.ptr(), xs2.ptr(), w8slab.ptr(), n0, kNs, kNs, c_two2.as<float>(), &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };
    auto requant_rot = [&](uint32_t n0, uint32_t width, void* out, uint32_t ldo, l0::Event* e) {
      if (rv >= 3) {
        cx.launch(k_rqr, width / 16, K / 1024, 1,
                  {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(damat.ptr()), PtrArg(dinv_rot.ptr()),
                   PtrArg(out), arg_val(n0), arg_val(N), arg_val(K), arg_val(ldo)},
                  e);
        return;
      }
      cx.launch(k_rqr, width / 16, K / 1024, 1,
                {PtrArg(dq.ptr()), PtrArg(ds.ptr()), rot_v1 ? PtrArg(dsg.ptr()) : PtrArg(dsbits.ptr()),
                 PtrArg(dinv_rot.ptr()),
                 PtrArg(out), arg_val(n0), arg_val(N), arg_val(ldo)},
                e);
    };
    auto run_requant_rot_full = [&]() -> size_t {
      requant_rot(0, N, w8rot.ptr(), N, &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_two_pass_rot = [&]() -> size_t {
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        requant_rot(n0, kNs, w8slab.ptr(), kNs, &ev[2 * i]);
        gemm8w(xqh2.ptr(), xsh2.ptr(), w8slab.ptr(), dcs_rot.as<float>(), n0, kNs, kNs,
               c_rot.as<float>(), &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };
    auto run_requant_full = [&]() -> size_t {
      requant(0, N, w8full.ptr(), N, &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_gemm_full = [&]() -> size_t {
      gemm8(w8full.ptr(), 0, N, N, c_full.as<float>(), &ev[0]);
      cx.wait();
      return 1;
    };
    auto run_two_pass = [&]() -> size_t {
      for (uint32_t i = 0; i < kSlabs; ++i) {
        const uint32_t n0 = i * kNs;
        requant(n0, kNs, w8slab.ptr(), kNs, &ev[2 * i]);
        gemm8(w8slab.ptr(), n0, kNs, kNs, c_two.as<float>(), &ev[2 * i + 1]);
      }
      cx.wait();
      return 2 * kSlabs;
    };

    // --- value pass ---------------------------------------------------------
    run_control();
    run_quant();
    run_quant_had();
    run_requant_full();
    run_gemm_full();
    if (sig.as<uint32_t>()[0] != 0x57384138u) throw std::runtime_error("pw8_gemm did not run");
    run_two_pass();

    // (1) requant: every int8 weight against the CPU
    std::vector<uint32_t> w8h(size_t(K / 4) * N);
    download(ctx, w8h.data(), w8full.ptr(), w8_bytes);
    size_t rq_diff = 0;
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n) {
        const uint32_t word = qw[size_t(k / 8) * N + n];
        const float v = float(int((word >> (4 * (k % 8))) & 0xFu) - 8) *
                        common::f16_to_f32(sc[size_t(k / 64) * N + n]);
        const int8_t want = sat_rte(v / cs[n]);
        const int8_t got = int8_t((w8h[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu);
        rq_diff += (want != got);
      }
    std::printf("\n## checks\n\n| check | result |\n|---|---|\n");
    std::printf("| int8 weights vs CPU requant, all %zu | **%zu differ** |\n", size_t(K) * N, rq_diff);
    if (rq_diff) throw std::runtime_error("pw8_requant disagrees with the CPU");

    // (2) GEMM oracle, 32 x 32 slice, full K
    std::vector<int8_t> xq_h(size_t(kM) * K);
    std::vector<float> xs_h(kM);
    download(ctx, xq_h.data(), xq.ptr(), xq_h.size());
    download(ctx, xs_h.data(), xs.ptr(), kM * 4);
    std::vector<float> cfull(size_t(kM) * N), ctwo(size_t(kM) * N), cctl(size_t(kM) * N);
    download(ctx, cfull.data(), c_full.ptr(), c_bytes);
    download(ctx, ctwo.data(), c_two.ptr(), c_bytes);
    download(ctx, cctl.data(), c_ctl.ptr(), c_bytes);
    double orc = 0.0;
    for (uint32_t m = 0; m < 32; ++m)
      for (uint32_t n = 0; n < 32; ++n) {
        int32_t acc = 0;
        for (uint32_t k = 0; k < K; ++k)
          acc += int32_t(xq_h[size_t(m) * K + k]) *
                 int32_t(int8_t((w8h[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu));
        const float want = float(acc) * (cs[n] * xs_h[m]);
        orc = std::max(orc, double(std::fabs(want - cfull[size_t(m) * N + n])));
      }
    std::printf("| pw8_gemm vs integer-exact oracle, 32 x 32 at full K | **max abs %.6g** |\n", orc);
    if (orc != 0.0) throw std::runtime_error("pw8_gemm disagrees with its oracle");

    // (3) two-pass == full, every word
    size_t tw_diff = 0;
    for (size_t i = 0; i < cfull.size(); ++i)
      tw_diff += (std::memcmp(&cfull[i], &ctwo[i], 4) != 0);
    std::printf("| two-pass vs full, all %zu outputs | **%zu differ** |\n", cfull.size(), tw_diff);
    if (tw_diff) throw std::runtime_error("the slab two-pass is not the full GEMM");

    // (4) the rotating quantiser, 64 rows, against a CPU Walsh-Hadamard
    {
      std::vector<int8_t> xqh_h(size_t(kM) * K);
      std::vector<float> xsh_h(kM);
      download(ctx, xqh_h.data(), xqh.ptr(), xqh_h.size());
      download(ctx, xsh_h.data(), xsh.ptr(), kM * 4);
      int worst = 0;
      size_t off1 = 0;
      double sdiff = 0.0;
      std::vector<float> r(K);
      for (uint32_t m = 0; m < 64; ++m) {
        for (uint32_t k = 0; k < K; ++k) r[k] = common::bf16_to_f32(act[size_t(m) * K + k]) * sgn[k];
        for (uint32_t b = 0; b < K; b += 1024)
          for (uint32_t hh = 1; hh < 1024; hh <<= 1)
            for (uint32_t i = 0; i < 1024; i += 2 * hh)
              for (uint32_t j = i; j < i + hh; ++j) {
                const float a = r[b + j], c = r[b + j + hh];
                r[b + j] = a + c;
                r[b + j + hh] = a - c;
              }
        float amax = 0.0f;
        for (float& v : r) { v *= 1.0f / 32.0f; amax = std::max(amax, std::fabs(v)); }
        const float scale = amax / 127.0f;
        sdiff = std::max(sdiff, std::fabs(double(scale) - xsh_h[m]) / scale);
        for (uint32_t k = 0; k < K; ++k) {
          const int d = std::abs(int(sat_rte(r[k] / scale)) - int(xqh_h[size_t(m) * K + k]));
          worst = std::max(worst, d);
          off1 += (d != 0);
        }
      }
      std::printf("| pw8_quant_had vs CPU FWHT, 64 rows | **max %d LSB, %zu of %zu off, scale rel %.2g** |\n",
                  worst, off1, size_t(64) * K, sdiff);
      if (worst > 1) throw std::runtime_error("pw8_quant_had disagrees with the CPU transform");
    }

    // (5) the error the CPU study predicts for this (unrotated) variant
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < cctl.size(); ++i) {
      const double d = double(ctwo[i]) - double(cctl[i]);
      num += d * d;
      den += double(cctl[i]) * double(cctl[i]);
    }
    std::printf("| two-pass vs bf16 control, global relative L2 (unrotated; CPU study: ~2.8%% "
                "gate||up, ~5.2%% down) | **%.3f%%** |\n", 100.0 * std::sqrt(num / den));

    // --- v2 value pass and checks -------------------------------------------
    run_quant2();
    run_quant_had2();
    run_requant2_full();
    run_gemm_full2();
    run_two_pass2();
    {
      std::vector<uint32_t> w2(size_t(K / 4) * N);
      download(ctx, w2.data(), w8full2.ptr(), w8_bytes);
      size_t d = 0;
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n) {
          const uint32_t word = qw[size_t(k / 8) * N + n];
          const float t = common::f16_to_f32(sc[size_t(k / 64) * N + n]) * inv[n];
          const int8_t want = sat_rte(float(int((word >> (4 * (k % 8))) & 0xFu) - 8) * t);
          d += (want != int8_t((w2[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu));
        }
      std::printf("| v2 int8 weights vs CPU (q * (s * inv_c)), all %zu | **%zu differ** |\n",
                  size_t(K) * N, d);
      if (d) throw std::runtime_error("pw8_requant2 disagrees with the CPU");

      // plain quantiser, every row; rotating quantiser, 64 rows
      std::vector<int8_t> q2(size_t(kM) * K), qh2(size_t(kM) * K);
      std::vector<float> s2(kM), sh2(kM);
      download(ctx, q2.data(), xq2.ptr(), q2.size());
      download(ctx, s2.data(), xs2.ptr(), kM * 4);
      download(ctx, qh2.data(), xqh2.ptr(), qh2.size());
      download(ctx, sh2.data(), xsh2.ptr(), kM * 4);
      size_t dq_plain = 0, dq_had = 0, ds_bad = 0;
      std::vector<float> r(K);
      for (uint32_t m = 0; m < kM; ++m) {
        for (int had = 0; had < 2; ++had) {
          if (had && m >= 64) break;
          for (uint32_t k = 0; k < K; ++k)
            r[k] = common::bf16_to_f32(act[size_t(m) * K + k]) * (had ? sgn[k] : 1.0f);
          if (had) {
            for (uint32_t b = 0; b < K; b += 1024)
              for (uint32_t hh = 1; hh < 1024; hh <<= 1)
                for (uint32_t i = 0; i < 1024; i += 2 * hh)
                  for (uint32_t j = i; j < i + hh; ++j) {
                    const float a = r[b + j], c = r[b + j + hh];
                    r[b + j] = a + c;
                    r[b + j + hh] = a - c;
                  }
            for (float& v : r) v *= 1.0f / 32.0f;
          }
          float amax = 0.0f;
          for (float v : r) amax = std::max(amax, std::fabs(v));
          const float scale = amax / 127.0f;
          const float iv = scale == 0.0f ? 0.0f : 1.0f / scale;
          ds_bad += (scale != (had ? sh2[m] : s2[m]));
          const int8_t* got = (had ? qh2.data() : q2.data()) + size_t(m) * K;
          size_t& dd = had ? dq_had : dq_plain;
          for (uint32_t k = 0; k < K; ++k) dd += (sat_rte(r[k] * iv) != got[k]);
        }
      }
      std::printf("| v2 quantiser vs CPU, all %u rows | **%zu differ** |\n", kM, dq_plain);
      std::printf("| v2 rotating quantiser vs CPU FWHT, 64 rows | **%zu differ** |\n", dq_had);
      std::printf("| v2 per-token scales vs CPU | **%zu differ** |\n", ds_bad);
      if (dq_plain || dq_had || ds_bad) throw std::runtime_error("v2 quantisers disagree with the CPU");

      std::vector<float> f2(size_t(kM) * N), t2(size_t(kM) * N);
      download(ctx, f2.data(), c_full2.ptr(), c_bytes);
      download(ctx, t2.data(), c_two2.ptr(), c_bytes);
      double o2 = 0.0;
      for (uint32_t m = 0; m < 32; ++m)
        for (uint32_t n = 0; n < 32; ++n) {
          int32_t acc = 0;
          for (uint32_t k = 0; k < K; ++k)
            acc += int32_t(q2[size_t(m) * K + k]) *
                   int32_t(int8_t((w2[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu));
          o2 = std::max(o2, double(std::fabs(float(acc) * (cs[n] * s2[m]) - f2[size_t(m) * N + n])));
        }
      size_t td = 0;
      double num2 = 0.0, den2 = 0.0;
      for (size_t i = 0; i < f2.size(); ++i) {
        td += (std::memcmp(&f2[i], &t2[i], 4) != 0);
        const double dd = double(t2[i]) - double(cctl[i]);
        num2 += dd * dd;
        den2 += double(cctl[i]) * double(cctl[i]);
      }
      std::printf("| v2 GEMM vs oracle, 32 x 32 at full K | **max abs %.6g** |\n", o2);
      std::printf("| v2 two-pass vs v2 full, all outputs | **%zu differ** |\n", td);
      std::printf("| v2 two-pass vs bf16 control, relative L2 (unrotated) | **%.3f%%** |\n",
                  100.0 * std::sqrt(num2 / den2));
      if (o2 != 0.0 || td) throw std::runtime_error("v2 GEMM path disagrees");
    }
    {
      run_quant_had2();
      run_requant_rot_full();
      run_two_pass_rot();
      std::vector<uint32_t> wr(size_t(K / 4) * N);
      download(ctx, wr.data(), w8rot.ptr(), w8_bytes);
      size_t d = 0;
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n)
          d += (wrot_ref[size_t(k) * N + n] !=
                int8_t((wr[size_t(k / 4) * N + n] >> (8 * (k % 4))) & 0xFFu));
      std::printf("| ROTATED int8 weights vs CPU H(d*w)/32 requant, all %zu | **%zu differ** |\n",
                  size_t(K) * N, d);
      if (d) throw std::runtime_error("pw8_requant_rot disagrees with the CPU");
      std::vector<float> cr(size_t(kM) * N);
      download(ctx, cr.data(), c_rot.ptr(), c_bytes);
      double num3 = 0.0, den3 = 0.0, worst_cos = 1.0;
      for (uint32_t m = 0; m < kM; ++m) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t n = 0; n < N; ++n) {
          const double a = cr[size_t(m) * N + n], b = cctl[size_t(m) * N + n];
          num3 += (a - b) * (a - b);
          den3 += b * b;
          dot += a * b; na += a * a; nb += b * b;
        }
        if (na > 0.0 && nb > 0.0) worst_cos = std::min(worst_cos, dot / std::sqrt(na * nb));
      }
      std::printf("| **ROTATED path end to end** (rotated int8 x rotated int8) vs bf16 control: "
                  "relative L2 / worst-row cosine | **%.3f%% / %.6f** |\n",
                  100.0 * std::sqrt(num3 / den3), worst_cos);
    }
    std::fflush(stdout);

    // --- rates --------------------------------------------------------------
    // Warm-up burst: the CPU checks above leave the GPU idle for seconds, and the
    // first arm timed after them (the control) measured up to 35 % slow with a
    // single discarded warm-up. ~0.5 s of control launches first, and the
    // control is timed again LAST; the table uses the faster of the two.
    for (int i = 0; i < 100; ++i) run_control();
    const Timed t_ctl_first = best_of(run_control, ev, calib);
    const Timed t_q = best_of(run_quant, ev, calib);
    const Timed t_qh = best_of(run_quant_had, ev, calib);
    const Timed t_rqf = best_of(run_requant_full, ev, calib);
    const Timed t_gf = best_of(run_gemm_full, ev, calib);
    const Timed t_tw = best_of(run_two_pass, ev, calib);
    const Timed t_q2 = best_of(run_quant2, ev, calib);
    const Timed t_qh2 = best_of(run_quant_had2, ev, calib);
    const Timed t_rqf2 = best_of(run_requant2_full, ev, calib);
    const Timed t_tw2 = best_of(run_two_pass2, ev, calib);
    const Timed t_ctl_last = best_of(run_control, ev, calib);
    const Timed t_ctl = t_ctl_first.sum_ms <= t_ctl_last.sum_ms ? t_ctl_first : t_ctl_last;
    std::printf("\n# control timed first %.3f ms, last %.3f ms\n", t_ctl_first.sum_ms,
                t_ctl_last.sum_ms);

    std::printf("\n## rates (measured; L0 kernel timestamps, best of %d after one warm-up)\n\n",
                kReplays);
    std::printf("| arm | ms | TOP/s |\n|---|---:|---:|\n");
    std::printf("| **bf16 control**, %u slabs x 2 launches | **%.3f** | %.1f |\n", kSlabs,
                t_ctl.sum_ms, gflop / t_ctl.sum_ms);
    std::printf("| &nbsp;&nbsp;of which pf_dequant_slab | %.3f | |\n", t_ctl.even_ms);
    std::printf("| &nbsp;&nbsp;of which pf_gemm_T0 | %.3f | |\n", t_ctl.odd_ms);
    std::printf("| pw8_quant (no rotation) | %.3f | |\n", t_q.sum_ms);
    std::printf("| pw8_quant_had (1024-block Hadamard) | %.3f | |\n", t_qh.sum_ms);
    std::printf("| pw8_requant, all N in one launch | %.3f | |\n", t_rqf.sum_ms);
    std::printf("| **pw8_gemm, all N, weights pre-requantised** | **%.3f** | %.1f |\n", t_gf.sum_ms,
                gflop / t_gf.sum_ms);
    std::printf("| **two-pass**, %u slabs x (requant + gemm) | **%.3f** | %.1f |\n", kSlabs,
                t_tw.sum_ms, gflop / t_tw.sum_ms);
    std::printf("| &nbsp;&nbsp;of which pw8_requant | %.3f | |\n", t_tw.even_ms);
    std::printf("| &nbsp;&nbsp;of which pw8_gemm | %.3f | |\n", t_tw.odd_ms);

    std::printf("| **v2** pw8_quant2 (no rotation) | %.3f | |\n", t_q2.sum_ms);
    std::printf("| **v2** pw8_quant_had2 (1024-block Hadamard) | **%.3f** | |\n", t_qh2.sum_ms);
    std::printf("| **v2** pw8_requant2, all N in one launch | %.3f | |\n", t_rqf2.sum_ms);
    std::printf("| **v2 two-pass**, %u slabs x (requant2 + gemm) | **%.3f** | %.1f |\n", kSlabs,
                t_tw2.sum_ms, gflop / t_tw2.sum_ms);
    std::printf("| &nbsp;&nbsp;of which pw8_requant2 | %.3f | |\n", t_tw2.even_ms);
    std::printf("| &nbsp;&nbsp;of which pw8_gemm | %.3f | |\n", t_tw2.odd_ms);
    const double a2 = t_ctl.sum_ms / (t_tw2.sum_ms + t_qh2.sum_ms);
    const double a = t_ctl.sum_ms / (t_tw.sum_ms + t_qh.sum_ms);
    const double b = t_ctl.sum_ms / (t_gf.sum_ms + t_qh.sum_ms);
    const double c = t_ctl.sum_ms / t_gf.sum_ms;
    std::printf("\n## against the bf16 control\n\n| path | ms | x control | clears %.2fx |\n"
                "|---|---:|---:|:--:|\n", kStudyMult);
    std::printf("| **v2 two-pass + v2 rotating quantiser** (the production shape) | **%.3f** | **%.3fx** | %s |\n",
                t_tw2.sum_ms + t_qh2.sum_ms, a2, a2 >= kStudyMult ? "**YES**" : "no");
    std::printf("| v1 two-pass + v1 rotating quantiser | %.3f | %.3fx | %s |\n",
                t_tw.sum_ms + t_qh.sum_ms, a, a >= kStudyMult ? "YES" : "no");
    std::printf("| GEMM on resident int8 weights + rotating quantiser (needs 2x weight memory) | "
                "%.3f | %.3fx | %s |\n", t_gf.sum_ms + t_qh.sum_ms, b, b >= kStudyMult ? "YES" : "no");
    std::printf("| GEMM alone (the ceiling) | %.3f | %.3fx | %s |\n", t_gf.sum_ms, c,
                c >= kStudyMult ? "YES" : "no");

    // Attribution of the rotating requant (rot4x, the same kernel with parts
    // switched off at runtime), one slab-sequence each, full-N single launch too.
    if (rv == 4) {
      l0::Kernel k_x = m_w8.kernel("pw8_requant_rot4x");
      auto run_x = [&](uint32_t mode, bool full) {
        return [&, mode, full]() -> size_t {
          const uint32_t nl = full ? 1 : kSlabs, width = full ? N : kNs;
          for (uint32_t i = 0; i < nl; ++i)
            cx.launch(k_x, width / 16, K / 1024, 1,
                      {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(damat.ptr()), PtrArg(dinv_rot.ptr()),
                       PtrArg(w8slab.ptr()), arg_val(i * kNs), arg_val(N), arg_val(K),
                       arg_val(full ? N : kNs), arg_val(mode)},
                      &ev[i]);
          cx.wait();
          return nl;
        };
      };
      // full-N writes need the full buffer
      auto run_full = [&]() -> size_t {
        cx.launch(k_x, N / 16, K / 1024, 1,
                  {PtrArg(dq.ptr()), PtrArg(ds.ptr()), PtrArg(damat.ptr()), PtrArg(dinv_rot.ptr()),
                   PtrArg(w8rot.ptr()), arg_val(0u), arg_val(N), arg_val(K), arg_val(N), arg_val(0u)},
                  &ev[0]);
        cx.wait();
        return 1;
      };
      for (int i = 0; i < 20; ++i) run_control();
      std::printf("\n## rotating requant attribution (rot4x)\n\n| variant | ms |\n|---|---:|\n");
      std::printf("| slabs, full kernel | %.3f |\n", best_of(run_x(0, false), ev, calib).sum_ms);
      std::printf("| slabs, no global stores | %.3f |\n", best_of(run_x(1, false), ev, calib).sum_ms);
      std::printf("| slabs, no A reads / DPAS | %.3f |\n", best_of(run_x(2, false), ev, calib).sum_ms);
      std::printf("| slabs, neither | %.3f |\n", best_of(run_x(3, false), ev, calib).sum_ms);
      std::printf("| ONE launch over all N | %.3f |\n", best_of(run_full, ev, calib).sum_ms);
      std::printf("| requant2 (no rotation), slabs, for reference | %.3f |\n",
                  [&] { auto f = [&]() -> size_t { for (uint32_t i = 0; i < kSlabs; ++i) requant2(i * kNs, kNs, w8slab.ptr(), kNs, &ev[i]); cx.wait(); return kSlabs; }; return best_of(f, ev, calib).sum_ms; }());
    }

    // Interleaved pairs: the GPU's clock under sustained load differs from its
    // clock after an idle gap, and arms timed at different moments see
    // different clocks. Each round times the control and then the production-
    // shaped int8 path (v2 two-pass + v2 rotating quantiser) back to back.
    {
      constexpr int kRounds = 11;
      std::vector<double> tc, tp, ratio, trot, ratio_rot, rq_plain, rq_rot;
      for (int i = 0; i < 20; ++i) { run_control(); run_two_pass2(); run_two_pass_rot(); }
      for (int r = 0; r < kRounds; ++r) {
        const double c = best_of(run_control, ev, calib).sum_ms;
        const double qh = best_of(run_quant_had2, ev, calib).sum_ms;
        const Timed t2 = best_of(run_two_pass2, ev, calib);
        const Timed tr = best_of(run_two_pass_rot, ev, calib);
        tc.push_back(c);
        tp.push_back(t2.sum_ms + qh);
        ratio.push_back(c / (t2.sum_ms + qh));
        trot.push_back(tr.sum_ms + qh);
        ratio_rot.push_back(c / (tr.sum_ms + qh));
        rq_plain.push_back(t2.even_ms);
        rq_rot.push_back(tr.even_ms);
      }
      auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      auto lo = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
      auto hi = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
      std::printf("\n## interleaved pairs, %d rounds after a warm-up burst (the headline)\n\n",
                  kRounds);
      std::printf("| | median ms | min .. max |\n|---|---:|---:|\n");
      std::printf("| bf16 control | **%.3f** | %.3f .. %.3f |\n", med(tc), lo(tc), hi(tc));
      std::printf("| v2 two-pass + v2 rotating quantiser | **%.3f** | %.3f .. %.3f |\n", med(tp),
                  lo(tp), hi(tp));
      std::printf("| **paired ratio, control / int8** | **%.3fx** | %.3fx .. %.3fx |\n", med(ratio),
                  lo(ratio), hi(ratio));
      std::printf("| &nbsp;&nbsp;of which requant2 (all slabs) | %.3f | %.3f .. %.3f |\n",
                  med(rq_plain), lo(rq_plain), hi(rq_plain));
      std::printf("| **ROTATED: requant_rot two-pass + rotating quantiser** | **%.3f** | %.3f .. %.3f |\n",
                  med(trot), lo(trot), hi(trot));
      std::printf("| &nbsp;&nbsp;of which requant_rot (all slabs) | %.3f | %.3f .. %.3f |\n",
                  med(rq_rot), lo(rq_rot), hi(rq_rot));
      std::printf("| **paired ratio, control / ROTATED int8** | **%.3fx** | %.3fx .. %.3fx |\n",
                  med(ratio_rot), lo(ratio_rot), hi(ratio_rot));
    }
    std::printf("\nThe ROTATED rows rotate both sides: activations in pw8_quant_had2, weights in "
                "the rotating requant (B70_W8A8_ROT selects the version). The unrotated rows are "
                "fast but fail the accuracy bar (docs/probe-w4a8-2026-09-23.md section 14).\n");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_w8a8 FAILED: %s\n", ex.what());
    return 2;
  }
}
