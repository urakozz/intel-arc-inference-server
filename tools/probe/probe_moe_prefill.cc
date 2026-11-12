// probe_moe_prefill - plan 2026-11-12 (Ornith MoE prefill) Task 1: the grouped expert GEMM read straight from the
// int4 expert blocks, and the multi-work-group sort, on REAL routes (a dump of every layer's route rows after a real
// prefill: ornith_prefill_test / qwen4exp_prefill_test --dump-routes). One model's expert shapes, random int4
// layout-1 weights (two copies, alternated by layer-chunk, so no launch finds the last one's weights in L2), random
// bf16 activations; the routes - so the tile tables, the expert row counts and the item tables - are the dump's.
//
//   control   today's production walk per layer-chunk: the gate||up dequant + grouped GEMM in two weight batches,
//             the down dequant + GEMM in one (pf_moe.cl / q4_pf_moe.cl + pf_moe_gemm.cl)
//   F1.*      fused bf16, per-sub-group conversion          (probe_moe_prefill.cl FORM 1)
//   F2.*      fused bf16, the B slab converted once into SLM (FORM 2)
//   F3.*      W4A8, int8 activations x the s4 operand          (FORM 3; quantiser timed beside it)
//   sort      pf_moe_sort / q4_pf_sort against pf_moe_sort_mw's hist + scan + scatter
//   c4        (phase 3 Task 1) F1 with the EXACT integer weights (q - 8) and the g64 scale on the accumulator, per
//             64-k group: acc = fma(tile, scale, acc); checked bitwise against a host oracle of the same order on
//             integer-valued A rows (the DPAS partial sums are then exact whatever its summation order)
//   o9        (phase 3 Task 1, the plan's `o8`) F1.a8b2m1n8.c2x0o1 with A read through row_tok from the chunk's x;
//             bitwise the chosen arm; reported with the gather's own ms
//   shd       (phase 3 Task 1, Ornith) the shared expert out of the item walk: pf_dequant_slab + pf_gemm (SiLU
//             epilogue) for gate||up and pf_dequant_slab + pf_gemm for down on the chunk's rows in token order, the
//             routed items alone in the chosen arm's walk; timed as the sum against the chosen arm's full walk
//
// Gates before any time: the mw sort's four buffers equal the one-work-group sort's (C <= 2048) and the host twin's
// (tests/kernels/pf_moe_ref.h, every C) on the checked layer-chunks; every F1 / F2 arm's h and y equal the control's
// over every used row as raw 32-bit words; every F3 arm's fp32 result (its EPI 2 binary) equals an integer-exact host
// oracle on every row of the checked tiles. An arm that fails is reported and not timed.
//
// Timing: each sample is one arm's whole walk over every layer-chunk of the dump between two global timestamps,
// after a warm-up of every arm; `rounds` rounds with the arms in a rotating order (interleaved). Medians, divided by
// the layer-chunks: ms a layer and chunk. TFLOP/s counts the useful rows only (routed pairs + the shared expert's C).
//
//   probe_moe_prefill --routes <file> --model ornith|flashnext [--C 2048,4096] [--rounds 5] [--check 4]
//                     [--arms all|<tag>,...] [--list]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/prefill/pf_kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "pf_moe_ref.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;
constexpr uint32_t kTm = 32, kNone = 0xFFFFFFFFu, kTileU32 = 136;

struct Model {
  const char* tag;
  uint32_t E, TK, H, I;
  bool q4exp;   // Flash-Next: q4_pf_moe's control binaries, the shared block passed as w_sh
};
const Model kModels[] = {{"ornith", 256, 8, 2048, 512, false}, {"flashnext", 512, 10, 2560, 640, true}};

// One fused arm: probe_moe_prefill.cl's defines. The list is tools/probe/CMakeLists.txt's PMP_ARMS, in the same
// order; a binary the build lacks is skipped with a line.
struct Arm {
  int form, na, nb, wgm, wgn, cvt, xsg, opt;
  std::string tag() const {
    char b[64];
    std::snprintf(b, sizeof b, "F%d.a%db%dm%dn%d.c%dx%do%d", form, na, nb, wgm, wgn, cvt, xsg, opt);
    return b;
  }
  uint32_t mi() const { return 8u * na * wgm; }          // rows a work item
  uint32_t wg_n() const { return 16u * nb * wgn; }
  bool nfirst() const { return opt & 1; }
  bool exact() const { return cvt == 4; }     // not bitwise the control: checked against an integer-order oracle
  bool indexed() const { return form == 1 && (opt & 8); }   // A through row_tok
  bool diag() const { return cvt == 9 || (opt & 4); }   // the DIAGNOSTIC forms: timed, not values
  std::string variant(uint32_t K, uint32_t N, int epi) const {
    char b[96];
    std::snprintf(b, sizeof b, "pmp_F%d_A%dB%dM%dN%d_C%dX%dO%d_K%u_N%u_E%d", form, na, nb, wgm, wgn, cvt, xsg, opt,
                  K, N, epi);
    return b;
  }
};
// Round 1 (2026-11-12, the record's first table) also ran c0 / c1 conversions, 64-row sub-groups (not bitwise:
// dropped) and MI 64 / 256 SLM items; round 2 keeps the fastest shapes and adds the OPT bits.
const Arm kArms[] = {
    {1, 4, 4, 1, 4, 0, 0, 0}, {1, 4, 4, 1, 4, 2, 0, 0}, {1, 4, 4, 1, 4, 2, 0, 1}, {2, 4, 4, 4, 4, 2, 0, 0},
    {2, 4, 4, 4, 4, 2, 0, 1}, {2, 4, 4, 4, 4, 2, 0, 2}, {2, 4, 4, 4, 4, 2, 0, 3}, {2, 4, 4, 8, 4, 2, 0, 3},
    {2, 4, 4, 2, 4, 2, 0, 3}, {3, 4, 4, 1, 4, 0, 0, 0}, {3, 4, 4, 1, 4, 0, 0, 1}, {3, 4, 2, 4, 8, 0, 0, 0},
    {3, 4, 2, 4, 8, 0, 0, 1}, {3, 4, 4, 1, 4, 0, 1, 1}, {3, 4, 2, 4, 8, 0, 1, 1},
    // DIAGNOSTIC (not checked): c9 = no conversion, o4 = no DPAS
    {1, 4, 4, 1, 4, 9, 0, 1}, {1, 4, 4, 1, 4, 2, 0, 5}, {2, 4, 4, 4, 4, 9, 0, 3}, {2, 4, 4, 4, 4, 2, 0, 7},
    // round 3: 64-row sub-groups again (A's row offset in the base: bitwise now), half the conversion per DPAS
    {1, 8, 2, 1, 8, 2, 0, 1}, {1, 8, 2, 2, 8, 2, 0, 1}, {2, 8, 2, 2, 8, 2, 0, 3}, {1, 8, 2, 1, 8, 9, 0, 1},
    // round 4: the inline-vISA conversion (bfe, int -> f32, one mixed-mode bf16 multiply)
    {1, 4, 4, 1, 4, 3, 0, 1}, {1, 8, 2, 1, 8, 3, 0, 1}, {2, 4, 4, 4, 4, 3, 0, 3},
    // phase 3 Task 1: c4 = exact integer weights, the scale on the group's partial sum (NA 4: a tile per atom);
    // o9 = the chosen arm with A read through row_tok
    {1, 4, 2, 1, 8, 4, 0, 1}, {1, 4, 4, 1, 4, 4, 0, 1}, {1, 8, 2, 1, 8, 2, 0, 9},
};
const char* const kChosen = "F1.a8b2m1n8.c2x0o1";

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
uint16_t f32_to_f16(float f) {   // positive normal values only (the scales)
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const int e = int((u >> 23) & 0xFF) - 127 + 15;
  return uint16_t((e << 10) | ((u >> 13) & 0x3FF));
}
float f16_to_f32(uint16_t h) {   // normal values only
  const uint32_t u = (uint32_t(h & 0x8000) << 16) | ((uint32_t((h >> 10) & 0x1F) - 15 + 127) << 23) |
                     (uint32_t(h & 0x3FF) << 13);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
  return v;
}

void upload(l0::Context& ctx, void* dst, const void* src, size_t bytes) {
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(dst, src, bytes);
}
void download(l0::Context& ctx, void* dst, const void* src, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, std::min(kChunk, bytes - off));
  }
}

struct Routes {
  uint32_t layers = 0, n = 0;
  std::vector<uint32_t> rows;   // [layers][n][32]
};
// ornith_prefill_test --dump-routes: "B70ROUTE", u32 version 1, layers, tokens, words (32), then the rows.
Routes read_routes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[8];
  uint32_t h[4];
  f.read(magic, 8);
  f.read(reinterpret_cast<char*>(h), sizeof h);
  if (!f || std::memcmp(magic, "B70ROUTE", 8) != 0 || h[0] != 1 || h[3] != 32)
    throw std::runtime_error(path + ": not a version-1 route dump");
  Routes r;
  r.layers = h[1];
  r.n = h[2];
  r.rows.resize(size_t(r.layers) * r.n * 32);
  f.read(reinterpret_cast<char*>(r.rows.data()), std::streamsize(r.rows.size() * 4));
  if (!f) throw std::runtime_error(path + ": short read");
  return r;
}

uint32_t tmax_of(const Model& m, uint32_t C) {
  return (C * m.TK + m.E * (kTm - 1)) / kTm + (C + kTm - 1) / kTm;
}
// Work items of up to `mi` rows of one block from the tile table (pf_moe_sort's), padded with NONE to the bound
// sum_e ceil(c_e / mi) <= (C TK + E (mi - 1)) / mi + ceil(C / mi): a grid that is a function of C alone.
uint32_t items_bound(const Model& m, uint32_t C, uint32_t mi) {
  return (C * m.TK + m.E * (mi - 1)) / mi + (C + mi - 1) / mi;
}
std::vector<uint32_t> make_items(const std::vector<uint32_t>& tiles, uint32_t ntiles, uint32_t mi, uint32_t bound) {
  std::vector<uint32_t> it;
  for (uint32_t t = 0; t < ntiles;) {
    const uint32_t e = tiles[2 * t], r0 = tiles[2 * t + 1];
    uint32_t n = 1;
    while (t + n < ntiles && n * kTm < mi && tiles[2 * (t + n)] == e) ++n;
    it.insert(it.end(), {e, r0, n * kTm, 0});
    t += n;
  }
  if (it.size() / 4 > bound) throw std::runtime_error("work items exceed their bound");
  while (it.size() / 4 < bound) it.insert(it.end(), {kNone, 0, 0, 0});
  return it;
}

// Integer-exact host oracle of FORM 3's fp32 result (EPI 2) for row r and column n of block e.
float w4a8_ref(const uint32_t* blk, const int8_t* a, const float* xs, uint32_t K, uint32_t n, int xsg) {
  const uint32_t G = K / 64;
  float acc = 0.0f;
  for (uint32_t g = 0; g < G; ++g) {
    const uint32_t* tile = blk + (size_t(n / 16) * G + g) * kTileU32;
    int32_t t = 0;
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t w = tile[j * 16 + n % 16];
      for (uint32_t i = 0; i < 8; ++i) t += int32_t(a[g * 64 + j * 8 + i]) * (int32_t((w >> (4 * i)) & 0xF) - 8);
    }
    const float ws = f16_to_f32(reinterpret_cast<const uint16_t*>(tile + 128)[n % 16]);
    acc = xsg ? std::fma(float(t), ws * xs[g], acc) : std::fma(float(t), ws, acc);
  }
  return xsg ? acc : acc * xs[0];
}

// Host oracle of CVT 4's fp32 result (EPI 2) for A row `a` (small integers) and column n of block `blk`: per 64-k
// group the exact integer partial sum, then acc = fma(sum, scale, acc) - the kernel's order, group ascending.
float c4_ref(const uint32_t* blk, const int8_t* a, uint32_t K, uint32_t n) {
  const uint32_t G = K / 64;
  float acc = 0.0f;
  for (uint32_t g = 0; g < G; ++g) {
    const uint32_t* tile = blk + (size_t(n / 16) * G + g) * kTileU32;
    int32_t t = 0;
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t w = tile[j * 16 + n % 16];
      for (uint32_t i = 0; i < 8; ++i) t += int32_t(a[g * 64 + j * 8 + i]) * (int32_t((w >> (4 * i)) & 0xF) - 8);
    }
    acc = std::fma(float(t), f16_to_f32(reinterpret_cast<const uint16_t*>(tile + 128)[n % 16]), acc);
  }
  return acc;
}

struct LayerChunk {
  const uint32_t* route;   // device
  std::vector<uint32_t> hdr, tiles;
  std::map<uint32_t, l0::Mem*> items;     // mi -> device items
  std::map<uint32_t, l0::Mem*> items_r;   // mi -> the routed experts' items alone (shd)
  l0::Mem* rowtok = nullptr;              // device row_tok of this layer-chunk
  double gu_flop = 0, dn_flop = 0;
};

int run(const Model& m, const Routes& rt, uint32_t C, int rounds, uint32_t check_lc, const std::vector<std::string>& want) {
  l0::Context ctx(0);
  runtime::prefill::Context cx(ctx);
  runtime::prefill::KernelCache kc(ctx);
  const uint32_t E = m.E, TK = m.TK, H = m.H, I = m.I, NG = 2 * I, B = E + 1;
  const uint32_t T = tmax_of(m, C), rows = T * kTm;
  if (rt.n % C) throw std::runtime_error("the dump's tokens are not whole chunks of C");
  const uint32_t nlc = rt.layers * (rt.n / C);
  std::printf("# probe_moe_prefill: %s (E %u, top %u, H %u, I %u), C %u, %u layer-chunks, tmax %u, device %s\n", m.tag,
              E, TK, H, I, C, nlc, T, ctx.name().c_str());
  const pf_moe_ref::Shape shp{E, TK, H, I, 0};
  const size_t gu_u32 = shp.gate_up_words(), dn_u32 = shp.down_words();

  // ---- device data -------------------------------------------------------------------------------------------
  std::mt19937 rng(11);
  auto fill_blocks = [&](std::vector<uint32_t>& w) {
    std::uniform_real_distribution<float> u(0.005f, 0.02f);
    for (size_t t = 0; t < w.size() / kTileU32; ++t) {
      for (uint32_t j = 0; j < 128; ++j) w[t * kTileU32 + j] = rng();
      for (uint32_t j = 0; j < 8; ++j)
        w[t * kTileU32 + 128 + j] = uint32_t(f32_to_f16(u(rng))) | (uint32_t(f32_to_f16(u(rng))) << 16);
    }
  };
  std::vector<uint32_t> wgu(gu_u32 * B), wdn(dn_u32 * B);
  std::vector<std::unique_ptr<l0::Mem>> dgu, ddn;
  for (int c = 0; c < 2; ++c) {
    fill_blocks(wgu);
    fill_blocks(wdn);
    dgu.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, wgu.size() * 4));
    ddn.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, wdn.size() * 4));
    upload(ctx, dgu.back()->ptr(), wgu.data(), wgu.size() * 4);
    upload(ctx, ddn.back()->ptr(), wdn.data(), wdn.size() * 4);
  }   // the host keeps copy 1's words (the oracle reads layer-chunks that use copy 1)
  std::vector<uint16_t> x(size_t(C) * H);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  upload(ctx, dx.ptr(), x.data(), x.size() * 2);
  l0::Mem droute(ctx, l0::MemKind::Device, rt.rows.size() * 4);
  upload(ctx, droute.ptr(), rt.rows.data(), rt.rows.size() * 4);
  const uint32_t hw = kernels::pf_moe::hdr_words(E);
  l0::Mem hdr(ctx, l0::MemKind::Device, hw * 4), tiles(ctx, l0::MemKind::Device, size_t(T) * 8),
      row_tok(ctx, l0::MemKind::Device, size_t(rows) * 4), pair_row(ctx, l0::MemKind::Device, size_t(C) * TK * 4),
      cnt(ctx, l0::MemKind::Device, size_t((C + 63) / 64) * E * 4),
      ditems(ctx, l0::MemKind::Device, size_t(items_bound(m, C, kernels::pf_moe::kItemM)) * 16);
  const uint32_t ib64 = items_bound(m, C, kernels::pf_moe::kItemM);
  l0::Mem A(ctx, l0::MemKind::Device, size_t(rows) * H * 2), h_c(ctx, l0::MemKind::Device, size_t(rows) * I * 2),
      h_f(ctx, l0::MemKind::Device, size_t(rows) * I * 2), y_c(ctx, l0::MemKind::Device, size_t(rows) * H * 2),
      y_f(ctx, l0::MemKind::Device, size_t(rows) * H * 2), cf(ctx, l0::MemKind::Device, size_t(rows) * std::max(NG, H) * 4),
      a8(ctx, l0::MemKind::Device, size_t(rows) * H), h8(ctx, l0::MemKind::Device, size_t(rows) * I),
      xs8(ctx, l0::MemKind::Device, size_t(rows) * (H / 64) * 4), hs8(ctx, l0::MemKind::Device, size_t(rows) * (I / 64) * 4);
  const uint32_t per_gu = (B + 1) / 2;
  l0::Mem wb(ctx, l0::MemKind::Device, std::max(size_t(per_gu) * H * NG, size_t(B) * I * H) * 2);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();
  // shd (Ornith): the shared block is block E of the same weights, one layout-1 [K][N] linear per half, so the
  // production slab dequant takes it as it is. Not for Flash-Next (N 1280 is not a whole 1024-column slab).
  bool have_shd = !m.q4exp && NG % 1024 == 0 && H % 64 == 0 && I % 64 == 0 && H % 1024 == 0 && C % 256 == 0;
  l0::Kernel *k_slab_gu = nullptr, *k_slab_dn = nullptr, *k_dgemm_silu = nullptr, *k_dgemm = nullptr;
  std::unique_ptr<l0::Mem> d_slab, d_hsh, d_ysh;
  if (have_shd) {
    try {
      k_slab_gu = &kc("pmp_shd_slab_K" + std::to_string(H) + "_N" + std::to_string(NG) + "_L1", "pf_dequant_slab");
      k_slab_dn = &kc("pmp_shd_slab_K" + std::to_string(I) + "_N" + std::to_string(H) + "_L1", "pf_dequant_slab");
      k_dgemm_silu = &kc(kernels::pf_gemm_silu_variant(), "pf_gemm");
      k_dgemm = &kc(kernels::pf_gemm_variant(false), "pf_gemm");
      d_slab = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(std::max(H, I)) * 1024 * 2);
      d_hsh = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(C) * I * 2);
      d_ysh = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(C) * H * 4);
    } catch (const std::exception& e) {
      std::printf("# shd: not built (%s), skipped\n", e.what());
      have_shd = false;
    }
  }
  {   // A: random bf16 rows (the timed walks; the checks gather the real rows)
    std::vector<uint16_t> a(size_t(rows) * H);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : a) v = f32_to_bf16(nd(rng));
    upload(ctx, A.ptr(), a.data(), a.size() * 2);
    upload(ctx, h_c.ptr(), a.data(), size_t(rows) * I * 2);
  }

  // ---- kernels -----------------------------------------------------------------------------------------------
  const std::string pv = m.q4exp ? kernels::qwen4exp::pf_moe_variant() : kernels::pf_moe_variant(E, TK, H, I);
  const std::string sv = kernels::pf_moe_sort_mw_variant(E, TK);
  l0::Kernel& k_gemm_gu = kc(kernels::pf_moe_gemm_variant(H, NG, false, true), "pf_moe_gemm");
  l0::Kernel& k_gemm_dn = kc(kernels::pf_moe_gemm_variant(I, H, false, false), "pf_moe_gemm");
  l0::Kernel& k_deq_gu = kc(pv, m.q4exp ? "q4_pf_dequant_gu" : "pf_moe_dequant_gu");
  l0::Kernel& k_deq_dn = kc(pv, m.q4exp ? "q4_pf_dequant_dn" : "pf_moe_dequant_dn");
  l0::Kernel& k_sort = kc(pv, m.q4exp ? "q4_pf_sort" : "pf_moe_sort");
  l0::Kernel& k_gather = kc(pv, m.q4exp ? "q4_pf_gather" : "pf_moe_gather");
  l0::Kernel& k_hist = kc(sv, "pf_moe_hist");
  l0::Kernel& k_scan = kc(sv, "pf_moe_scan");
  l0::Kernel& k_scat = kc(sv, "pf_moe_scatter");
  const uint32_t nblk = (C + 63) / 64;

  auto sort_old = [&](const uint32_t* r) {
    cx.launch(k_sort, 1, 1, 1, {PtrArg(r), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()),
                                PtrArg(pair_row.ptr()), arg_val(C), arg_val(T)});
  };
  auto sort_mw = [&](const uint32_t* r) {
    cx.launch(k_hist, nblk, 1, 1, {PtrArg(r), PtrArg(cnt.ptr()), arg_val(C)});
    cx.launch(k_scan, 1, 1, 1, {PtrArg(cnt.ptr()), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()),
                                PtrArg(ditems.ptr()), arg_val(C), arg_val(T), arg_val(ib64)});
    cx.launch(k_scat, nblk, 1, 1, {PtrArg(r), PtrArg(cnt.ptr()), PtrArg(row_tok.ptr()), PtrArg(pair_row.ptr()),
                                   arg_val(C)});
  };
  const bool have_old = C <= 2048;

  // control: gate||up and down, today's walk
  auto ctl_gu = [&](const l0::Mem& tl, const l0::Mem& hd, int copy, void* hout) {
    for (uint32_t b0 = 0; b0 < B; b0 += per_gu) {
      const uint32_t nb = std::min(per_gu, B - b0);
      if (m.q4exp)
        cx.launch(k_deq_gu, NG / 16, H / 64, nb, {PtrArg(dgu[copy]->ptr()), PtrArg(dgu[copy]->as<uint32_t>() + size_t(E) * gu_u32),
                                                  PtrArg(hd.ptr()), PtrArg(wb.ptr()), arg_val(b0), arg_val(b0 + nb)});
      else
        cx.launch(k_deq_gu, NG / 16, H / 64, nb, {PtrArg(dgu[copy]->ptr()), PtrArg(hd.ptr()), PtrArg(wb.ptr()), arg_val(b0)});
      const uint32_t b1 = b0 + nb;
      cx.launch(k_gemm_gu, T, NG / 256, 1, {PtrArg(tl.ptr()), PtrArg(A.ptr()), PtrArg(nullptr), PtrArg(wb.ptr()),
                                            PtrArg(nullptr), PtrArg(hout), arg_val(b0), arg_val(b1), arg_val(0u)});
    }
  };
  auto ctl_dn = [&](const l0::Mem& tl, const l0::Mem& hd, int copy, const void* hin, void* yout) {
    if (m.q4exp)
      cx.launch(k_deq_dn, H / 16, I / 64, B, {PtrArg(ddn[copy]->ptr()), PtrArg(ddn[copy]->as<uint32_t>() + size_t(E) * dn_u32),
                                              PtrArg(hd.ptr()), PtrArg(wb.ptr()), arg_val(0u), arg_val(B)});
    else
      cx.launch(k_deq_dn, H / 16, I / 64, B, {PtrArg(ddn[copy]->ptr()), PtrArg(hd.ptr()), PtrArg(wb.ptr()), arg_val(0u)});
    cx.launch(k_gemm_dn, T, H / 256, 1, {PtrArg(tl.ptr()), PtrArg(hin), PtrArg(nullptr), PtrArg(wb.ptr()),
                                         PtrArg(nullptr), PtrArg(yout), arg_val(0u), arg_val(B), arg_val(0u)});
  };

  // shd's dense half: the shared expert's C rows in token order (x -> h -> y fp32 [C][H]), two slab walks
  auto shd_dense = [&](int copy) {
    const uint32_t* wg = dgu[copy]->as<uint32_t>() + size_t(E) * gu_u32;
    const uint32_t* wd = ddn[copy]->as<uint32_t>() + size_t(E) * dn_u32;
    cx.launch(*k_slab_gu, 1024 / 16, H / 64, 1, {PtrArg(wg), PtrArg(nullptr), PtrArg(d_slab->ptr()), arg_val(0u)});
    cx.launch(*k_dgemm_silu, C / 256, NG / 256, 1,
              {PtrArg(dx.ptr()), PtrArg(d_slab->ptr()), PtrArg(d_hsh->ptr()), arg_val(C), arg_val(H), arg_val(NG),
               arg_val(H), arg_val(NG), arg_val(I), arg_val(uint64_t(0)), arg_val(uint64_t(0)), arg_val(uint64_t(0))});
    for (uint32_t n0 = 0; n0 < H; n0 += 1024) {
      cx.launch(*k_slab_dn, 1024 / 16, I / 64, 1, {PtrArg(wd), PtrArg(nullptr), PtrArg(d_slab->ptr()), arg_val(n0)});
      cx.launch(*k_dgemm, C / 256, 1024 / 256, 1,
                {PtrArg(d_hsh->ptr()), PtrArg(d_slab->ptr()), PtrArg(d_ysh->as<float>() + n0), arg_val(C), arg_val(I),
                 arg_val(1024u), arg_val(I), arg_val(1024u), arg_val(H), arg_val(uint64_t(0)), arg_val(uint64_t(0)),
                 arg_val(uint64_t(0))});
    }
  };

  // ---- the fused arms that exist in this build ----------------------------------------------------------------
  struct Built {
    Arm a;
    l0::Kernel *gu, *dn, *gu_chk = nullptr, *dn_chk = nullptr, *quant = nullptr, *quant_dn = nullptr;
  };
  std::vector<Built> arms;
  for (const Arm& a : kArms) {
    if (!want.empty() && std::find(want.begin(), want.end(), a.tag()) == want.end()) continue;
    try {
      Built b{a, &kc(a.variant(H, NG, 1), "pmp_gemm"), &kc(a.variant(I, H, 0), "pmp_gemm")};
      if (a.exact()) {
        b.gu_chk = &kc(a.variant(H, NG, 2), "pmp_gemm");
        b.dn_chk = &kc(a.variant(I, H, 2), "pmp_gemm");
      }
      if (a.form == 3) {
        b.gu_chk = &kc(a.variant(H, NG, 2), "pmp_gemm");
        b.dn_chk = &kc(a.variant(I, H, 2), "pmp_gemm");
        b.quant = &kc(a.variant(H, NG, 2), "pmp_quant");
        b.quant_dn = &kc(a.variant(I, H, 2), "pmp_quant");
      }
      arms.push_back(b);
    } catch (const std::exception& e) {
      std::printf("# %s: not built (%s), skipped\n", a.tag().c_str(), e.what());
    }
  }
  std::map<uint32_t, uint32_t> bound;
  // (items, n-blocks) or, OPT bit 1, (n-blocks, items)
#define GRID(b, nit, NN) (b).a.nfirst() ? (NN) / (b).a.wg_n() : (nit), (b).a.nfirst() ? (nit) : (NN) / (b).a.wg_n(), 1
  for (const Built& b : arms) bound[b.a.mi()] = items_bound(m, C, b.a.mi());
  std::map<uint32_t, uint32_t> bound_r;   // the routed experts' items alone: no shared-expert term
  for (const auto& [mi, bd] : bound) bound_r[mi] = (C * TK + E * (mi - 1)) / mi;
  // c4's check inputs: A rows of small integers (exact in bf16; the DPAS partial sums are then exact integers
  // in any summation order), host and device
  std::vector<int8_t> ai8(size_t(rows) * H), hi8(size_t(rows) * I);
  l0::Mem d_ai(ctx, l0::MemKind::Device, size_t(rows) * H * 2), d_hi(ctx, l0::MemKind::Device, size_t(rows) * I * 2);
  const bool any_c4 = std::any_of(arms.begin(), arms.end(), [](const Built& b) { return b.a.exact(); });
  if (any_c4) {
    std::uniform_int_distribution<int> ui(-8, 8);
    std::vector<uint16_t> t(ai8.size());
    for (size_t i = 0; i < ai8.size(); ++i) ai8[i] = int8_t(ui(rng)), t[i] = f32_to_bf16(float(ai8[i]));
    upload(ctx, d_ai.ptr(), t.data(), t.size() * 2);
    t.resize(hi8.size());
    for (size_t i = 0; i < hi8.size(); ++i) hi8[i] = int8_t(ui(rng)), t[i] = f32_to_bf16(float(hi8[i]));
    upload(ctx, d_hi.ptr(), t.data(), t.size() * 2);
  }

  // ---- per layer-chunk: the sort's tables (and the checks) ---------------------------------------------------
  std::vector<LayerChunk> lcs(nlc);
  std::vector<std::unique_ptr<l0::Mem>> keep;
  std::vector<std::unique_ptr<l0::Mem>> lc_tiles, lc_hdr;
  std::set<std::string> failed;
  size_t sort_checks = 0, gemm_checks = 0, w4a8_checks = 0;
  for (uint32_t lc = 0; lc < nlc; ++lc) {
    const uint32_t layer = lc / (rt.n / C), c0 = (lc % (rt.n / C)) * C;
    const size_t roff = (size_t(layer) * rt.n + c0) * 32;
    LayerChunk& L = lcs[lc];
    L.route = droute.as<uint32_t>() + roff;
    const pf_moe_ref::Sorted ref = pf_moe_ref::sort(rt.rows.data() + roff, C, shp);
    // the mw sort, checked against the host twin (and the old sort) on the first check_lc layer-chunks
    sort_mw(L.route);
    cx.wait();
    std::vector<uint32_t> rowtok_h(rows);
    download(ctx, rowtok_h.data(), row_tok.ptr(), rowtok_h.size() * 4);
    keep.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, rowtok_h.size() * 4));
    upload(ctx, keep.back()->ptr(), rowtok_h.data(), rowtok_h.size() * 4);
    L.rowtok = keep.back().get();
    L.hdr.resize(hw);
    L.tiles.resize(size_t(T) * 2);
    download(ctx, L.hdr.data(), hdr.ptr(), hw * 4);
    download(ctx, L.tiles.data(), tiles.ptr(), L.tiles.size() * 4);
    if (lc < check_lc) {
      std::vector<uint32_t> rtk(rows), prw(size_t(C) * TK);
      download(ctx, rtk.data(), row_tok.ptr(), rtk.size() * 4);
      download(ctx, prw.data(), pair_row.ptr(), prw.size() * 4);
      const uint32_t used = ref.rows_used();
      std::vector<uint32_t> its(size_t(ib64) * 4);
      download(ctx, its.data(), ditems.ptr(), its.size() * 4);
      const bool ok = std::equal(L.hdr.begin(), L.hdr.end(), ref.hdr.begin()) && L.tiles == ref.tiles &&
                      std::equal(rtk.begin(), rtk.begin() + used, ref.row_tok.begin()) && prw == ref.pair_row &&
                      its == make_items(L.tiles, L.hdr[0], kernels::pf_moe::kItemM, ib64);
      ++sort_checks;
      if (!ok) throw std::runtime_error("layer-chunk " + std::to_string(lc) + ": the mw sort is not the host twin");
      if (have_old) {
        std::vector<uint32_t> h2(hw), t2(L.tiles.size()), r2(rows), p2(prw.size());
        sort_old(L.route);
        cx.wait();
        download(ctx, h2.data(), hdr.ptr(), hw * 4);
        download(ctx, t2.data(), tiles.ptr(), t2.size() * 4);
        download(ctx, r2.data(), row_tok.ptr(), r2.size() * 4);
        download(ctx, p2.data(), pair_row.ptr(), p2.size() * 4);
        if (h2 != L.hdr || t2 != L.tiles || !std::equal(r2.begin(), r2.begin() + used, rtk.begin()) || p2 != prw)
          throw std::runtime_error("layer-chunk " + std::to_string(lc) + ": the mw sort is not the one-work-group sort");
        ++sort_checks;
      }
    }
    for (uint32_t e = 0; e < E; ++e) L.gu_flop += 2.0 * L.hdr[4 + e] * H * NG, L.dn_flop += 2.0 * L.hdr[4 + e] * I * H;
    L.gu_flop += 2.0 * C * H * NG;
    L.dn_flop += 2.0 * C * I * H;
    lc_tiles.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, L.tiles.size() * 4));
    lc_hdr.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, hw * 4));
    upload(ctx, lc_tiles.back()->ptr(), L.tiles.data(), L.tiles.size() * 4);
    upload(ctx, lc_hdr.back()->ptr(), L.hdr.data(), hw * 4);
    for (const auto& [mi, bd] : bound) {
      const std::vector<uint32_t> it = make_items(L.tiles, L.hdr[0], mi, bd);
      keep.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, it.size() * 4));
      upload(ctx, keep.back()->ptr(), it.data(), it.size() * 4);
      L.items[mi] = keep.back().get();
    }
    if (have_shd) {   // the routed experts' tiles come first; the shared expert's (block E) follow
      uint32_t nr = 0;
      while (nr < L.hdr[0] && L.tiles[2 * nr] < E) ++nr;
      for (const auto& [mi, bd] : bound_r) {
        const std::vector<uint32_t> it = make_items(L.tiles, nr, mi, bd);
        keep.push_back(std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, it.size() * 4));
        upload(ctx, keep.back()->ptr(), it.data(), it.size() * 4);
        L.items_r[mi] = keep.back().get();
      }
    }
    if (lc >= check_lc) continue;
    // ---- the GEMM checks: real gathered rows, copy lc % 2 of the weights ----
    const int copy = int(lc % 2);
    cx.launch(k_gather, rows, 1, 1, {PtrArg(dx.ptr()), PtrArg(lc_hdr[lc]->ptr()), PtrArg(row_tok.ptr()), PtrArg(A.ptr())});
    ctl_gu(*lc_tiles[lc], *lc_hdr[lc], copy, h_c.ptr());
    ctl_dn(*lc_tiles[lc], *lc_hdr[lc], copy, h_c.ptr(), y_c.ptr());
    cx.wait();
    const uint32_t used = L.hdr[2];
    std::vector<uint32_t> hc(size_t(used) * I / 2), yc(size_t(used) * H / 2), hf(hc.size()), yf(yc.size());
    download(ctx, hc.data(), h_c.ptr(), hc.size() * 4);
    download(ctx, yc.data(), y_c.ptr(), yc.size() * 4);
    for (const Built& b : arms) {
      if (b.a.form == 3 || b.a.diag() || b.a.exact() || failed.count(b.a.tag())) continue;
      const uint32_t nit = bound[b.a.mi()];
      std::vector<uint8_t> poison(hc.size() * 4, 0xA5);
      upload(ctx, h_f.ptr(), poison.data(), poison.size());
      // an indexed arm reads the chunk's x through this layer-chunk's row_tok instead of the gathered rows
      cx.launch(*b.gu, GRID(b, nit, NG), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(b.a.indexed() ? dx.ptr() : A.ptr()),
                                                 PtrArg(b.a.indexed() ? L.rowtok->ptr() : nullptr),
                                                 PtrArg(dgu[copy]->ptr()), PtrArg(h_f.ptr()), arg_val(0u)});
      cx.launch(*b.dn, GRID(b, nit, H), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(h_c.ptr()), PtrArg(nullptr),
                                                PtrArg(ddn[copy]->ptr()), PtrArg(y_f.ptr()), arg_val(0u)});
      cx.wait();
      download(ctx, hf.data(), h_f.ptr(), hf.size() * 4);
      download(ctx, yf.data(), y_f.ptr(), yf.size() * 4);
      size_t dh = 0, dy = 0;
      for (size_t i = 0; i < hc.size(); ++i) dh += hc[i] != hf[i];
      for (size_t i = 0; i < yc.size(); ++i) dy += yc[i] != yf[i];
      ++gemm_checks;
      std::printf("| check | lc %u | %s | h %zu of %zu words differ, y %zu of %zu |\n", lc, b.a.tag().c_str(), dh,
                  hc.size(), dy, yc.size());
      if (dh || dy) failed.insert(b.a.tag());
    }
    if (have_shd) {   // the shared expert as dense: h and y against the control's shared rows (token order)
      shd_dense(copy);
      cx.wait();
      const uint32_t rs = L.hdr[1];
      std::vector<uint32_t> hs(size_t(C) * I / 2);
      std::vector<float> ys(size_t(C) * H);
      download(ctx, hs.data(), d_hsh->ptr(), hs.size() * 4);
      download(ctx, ys.data(), d_ysh->ptr(), ys.size() * 4);
      size_t dh = 0, dy = 0;
      double maxdy = 0;
      for (uint32_t t = 0; t < C; ++t) {
        for (uint32_t w = 0; w < I / 2; ++w) dh += hs[size_t(t) * (I / 2) + w] != hc[size_t(rs + t) * (I / 2) + w];
        for (uint32_t n = 0; n < H; ++n) {
          const uint16_t want_y = uint16_t(yc[size_t(rs + t) * (H / 2) + n / 2] >> (16 * (n & 1)));
          const uint16_t got_y = f32_to_bf16(ys[size_t(t) * H + n]);
          dy += want_y != got_y;
          const uint32_t a32 = uint32_t(want_y) << 16;
          float fa;
          std::memcpy(&fa, &a32, 4);
          maxdy = std::max(maxdy, double(std::fabs(fa - ys[size_t(t) * H + n])));
        }
      }
      std::printf("| check | lc %u | shd | h %zu of %zu words differ from the control's shared rows, y bf16 %zu of %zu "
                  "(max |control y - dense y fp32| %.3e) |\n", lc, dh, size_t(C) * I / 2, dy, size_t(C) * H, maxdy);
    }
    // W4A8: the quantisers, then the EPI 2 result on the rows of the first routed tile and the first shared tile
    // against the integer oracle (copy 1 of the weights is the host's)
    if (copy != 1) continue;
    for (const Built& b : arms) {   // CVT 4: the fp32 result on integer-valued rows against the host oracle
      if (!b.a.exact() || failed.count(b.a.tag())) continue;
      const uint32_t nit = bound[b.a.mi()];
      std::vector<float> gu(size_t(used) * NG), dn(size_t(used) * H);
      cx.launch(*b.gu_chk, GRID(b, nit, NG), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(d_ai.ptr()), PtrArg(nullptr),
                                              PtrArg(dgu[copy]->ptr()), PtrArg(cf.ptr()), arg_val(0u)});
      cx.wait();
      download(ctx, gu.data(), cf.ptr(), gu.size() * 4);
      cx.launch(*b.dn_chk, GRID(b, nit, H), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(d_hi.ptr()), PtrArg(nullptr),
                                             PtrArg(ddn[copy]->ptr()), PtrArg(cf.ptr()), arg_val(0u)});
      cx.wait();
      download(ctx, dn.data(), cf.ptr(), dn.size() * 4);
      size_t diff = 0, n = 0;
      for (uint32_t tsel : {0u, L.hdr[0] - 1}) {   // the first tile (a routed expert) and the last (the shared one)
        const uint32_t e = L.tiles[2 * tsel], r0 = L.tiles[2 * tsel + 1];
        for (uint32_t r = r0; r < r0 + kTm; ++r) {
          for (uint32_t col = 0; col < NG; ++col, ++n) {
            const float want_v = c4_ref(wgu.data() + size_t(e) * gu_u32, ai8.data() + size_t(r) * H, H, col);
            diff += std::memcmp(&want_v, &gu[size_t(r) * NG + col], 4) != 0;
          }
          for (uint32_t col = 0; col < H; ++col, ++n) {
            const float want_v = c4_ref(wdn.data() + size_t(e) * dn_u32, hi8.data() + size_t(r) * I, I, col);
            diff += std::memcmp(&want_v, &dn[size_t(r) * H + col], 4) != 0;
          }
        }
      }
      ++w4a8_checks;
      std::printf("| check | lc %u | %s | %zu of %zu fp32 results differ from the integer-order oracle |\n", lc,
                  b.a.tag().c_str(), diff, n);
      if (diff) failed.insert(b.a.tag());
    }
    for (const Built& b : arms) {
      if (b.a.form != 3 || failed.count(b.a.tag())) continue;
      const uint32_t kq = b.a.xsg ? 64u : H, kqd = b.a.xsg ? 64u : I;
      cx.launch(*b.quant, used, 1, 1, {PtrArg(A.ptr()), PtrArg(a8.ptr()), PtrArg(xs8.ptr()), arg_val(used), arg_val(kq)});
      cx.launch(*b.quant_dn, used, 1, 1, {PtrArg(h_c.ptr()), PtrArg(h8.ptr()), PtrArg(hs8.ptr()), arg_val(used), arg_val(kqd)});
      const uint32_t nit = bound[b.a.mi()];
      std::vector<float> gu(size_t(used) * NG), dn(size_t(used) * H);
      std::vector<int8_t> qa(size_t(used) * H), qh(size_t(used) * I);
      std::vector<float> sa(size_t(used) * (H / kq)), sh(size_t(used) * (I / kqd));
      cx.launch(*b.gu_chk, GRID(b, nit, NG), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(a8.ptr()),
                                                     PtrArg(xs8.ptr()), PtrArg(dgu[copy]->ptr()), PtrArg(cf.ptr()), arg_val(0u)});
      cx.wait();
      download(ctx, gu.data(), cf.ptr(), gu.size() * 4);
      cx.launch(*b.dn_chk, GRID(b, nit, H), {PtrArg(L.items[b.a.mi()]->ptr()), PtrArg(h8.ptr()),
                                                    PtrArg(hs8.ptr()), PtrArg(ddn[copy]->ptr()), PtrArg(cf.ptr()), arg_val(0u)});
      cx.wait();
      download(ctx, dn.data(), cf.ptr(), dn.size() * 4);
      download(ctx, qa.data(), a8.ptr(), qa.size());
      download(ctx, qh.data(), h8.ptr(), qh.size());
      download(ctx, sa.data(), xs8.ptr(), sa.size() * 4);
      download(ctx, sh.data(), hs8.ptr(), sh.size() * 4);
      size_t diff = 0, n = 0;
      for (uint32_t tsel : {0u, L.hdr[0] - 1}) {   // the first tile (a routed expert) and the last (the shared one)
        const uint32_t e = L.tiles[2 * tsel], r0 = L.tiles[2 * tsel + 1];
        for (uint32_t r = r0; r < r0 + kTm; ++r) {
          for (uint32_t col = 0; col < NG; ++col, ++n) {
            const float want_v = w4a8_ref(wgu.data() + size_t(e) * gu_u32, qa.data() + size_t(r) * H,
                                          sa.data() + size_t(r) * (H / kq), H, col, b.a.xsg);
            diff += std::memcmp(&want_v, &gu[size_t(r) * NG + col], 4) != 0;
          }
          for (uint32_t col = 0; col < H; ++col, ++n) {
            const float want_v = w4a8_ref(wdn.data() + size_t(e) * dn_u32, qh.data() + size_t(r) * I,
                                          sh.data() + size_t(r) * (I / kqd), I, col, b.a.xsg);
            diff += std::memcmp(&want_v, &dn[size_t(r) * H + col], 4) != 0;
          }
        }
      }
      ++w4a8_checks;
      std::printf("| check | lc %u | %s | %zu of %zu fp32 results differ from the integer oracle |\n", lc,
                  b.a.tag().c_str(), diff, n);
      if (diff) failed.insert(b.a.tag());
    }
  }
  std::printf("# checks: %zu sort, %zu bf16 GEMM, %zu W4A8; failed arms: %zu\n", sort_checks, gemm_checks,
              w4a8_checks, failed.size());
  for (const std::string& f : failed) std::printf("# FAILED (not timed): %s\n", f.c_str());

  // ---- timing ------------------------------------------------------------------------------------------------
  struct Timed {
    std::string name;
    std::function<void()> run;
    double flop;
  };
  double gu_flop = 0, dn_flop = 0;
  for (const LayerChunk& L : lcs) gu_flop += L.gu_flop, dn_flop += L.dn_flop;
  std::vector<Timed> tw;
  tw.push_back({"control.gu", [&] { for (uint32_t lc = 0; lc < nlc; ++lc) ctl_gu(*lc_tiles[lc], *lc_hdr[lc], int(lc % 2), h_f.ptr()); }, gu_flop});
  tw.push_back({"control.dn", [&] { for (uint32_t lc = 0; lc < nlc; ++lc) ctl_dn(*lc_tiles[lc], *lc_hdr[lc], int(lc % 2), h_c.ptr(), y_f.ptr()); }, dn_flop});
  if (have_old) tw.push_back({"sort.old", [&] { for (uint32_t lc = 0; lc < nlc; ++lc) sort_old(lcs[lc].route); }, 0});
  tw.push_back({"sort.mw", [&] { for (uint32_t lc = 0; lc < nlc; ++lc) sort_mw(lcs[lc].route); }, 0});
  for (const Built& b : arms) {
    if (failed.count(b.a.tag())) continue;
    const uint32_t nit = bound[b.a.mi()];
    const bool w8 = b.a.form == 3;
    tw.push_back({b.a.tag() + ".gu", [&, b, nit, w8] {
      for (uint32_t lc = 0; lc < nlc; ++lc)
        cx.launch(*b.gu, GRID(b, nit, NG), {PtrArg(lcs[lc].items[b.a.mi()]->ptr()),
                                                   PtrArg(w8 ? a8.ptr() : b.a.indexed() ? dx.ptr() : A.ptr()),
                                                   PtrArg(b.a.indexed() ? lcs[lc].rowtok->ptr() : xs8.ptr()),
                                                   PtrArg(dgu[lc % 2]->ptr()), PtrArg(h_f.ptr()), arg_val(0u)});
    }, gu_flop});
    tw.push_back({b.a.tag() + ".dn", [&, b, nit, w8] {
      for (uint32_t lc = 0; lc < nlc; ++lc)
        cx.launch(*b.dn, GRID(b, nit, H), {PtrArg(lcs[lc].items[b.a.mi()]->ptr()), PtrArg(w8 ? h8.ptr() : h_c.ptr()),
                                                  PtrArg(hs8.ptr()), PtrArg(ddn[lc % 2]->ptr()), PtrArg(y_f.ptr()), arg_val(0u)});
    }, dn_flop});
    if (w8) {
      const uint32_t kq = b.a.xsg ? 64u : H, kqd = b.a.xsg ? 64u : I;
      tw.push_back({b.a.tag() + ".quant", [&, b, kq, kqd] {
        for (uint32_t lc = 0; lc < nlc; ++lc) {
          const uint32_t used = lcs[lc].hdr[2];
          cx.launch(*b.quant, used, 1, 1, {PtrArg(A.ptr()), PtrArg(a8.ptr()), PtrArg(xs8.ptr()), arg_val(used), arg_val(kq)});
          cx.launch(*b.quant_dn, used, 1, 1, {PtrArg(h_c.ptr()), PtrArg(h8.ptr()), PtrArg(hs8.ptr()), arg_val(used), arg_val(kqd)});
        }
      }, 0});
    }
  }
  // the gather the indexed arm removes, and the shared-expert split: the chosen arm over the routed items alone,
  // then the dense half; "chosen.total" / "shd.total" are the two walks the gate compares
  tw.push_back({"gather", [&] {
    for (uint32_t lc = 0; lc < nlc; ++lc)
      cx.launch(k_gather, rows, 1, 1, {PtrArg(dx.ptr()), PtrArg(lc_hdr[lc]->ptr()), PtrArg(lcs[lc].rowtok->ptr()), PtrArg(A.ptr())});
  }, 0});
  const Built* chosen = nullptr;
  for (const Built& b : arms) if (b.a.tag() == kChosen) chosen = &b;
  if (chosen) {
    const Built cb = *chosen;
    const uint32_t nit = bound[cb.a.mi()];
    auto chosen_walk = [&, cb, nit](bool routed_only) {
      for (uint32_t lc = 0; lc < nlc; ++lc) {
        const l0::Mem* it = routed_only ? lcs[lc].items_r[cb.a.mi()] : lcs[lc].items[cb.a.mi()];
        const uint32_t n_it = routed_only ? bound_r[cb.a.mi()] : nit;
        cx.launch(*cb.gu, GRID(cb, n_it, NG), {PtrArg(it->ptr()), PtrArg(A.ptr()), PtrArg(xs8.ptr()),
                                               PtrArg(dgu[lc % 2]->ptr()), PtrArg(h_f.ptr()), arg_val(0u)});
        cx.launch(*cb.dn, GRID(cb, n_it, H), {PtrArg(it->ptr()), PtrArg(h_c.ptr()), PtrArg(xs8.ptr()),
                                              PtrArg(ddn[lc % 2]->ptr()), PtrArg(y_f.ptr()), arg_val(0u)});
      }
    };
    tw.push_back({"chosen.total", [chosen_walk] { chosen_walk(false); }, gu_flop + dn_flop});
    if (have_shd) {
      tw.push_back({"shd.routed", [chosen_walk] { chosen_walk(true); }, 0});
      tw.push_back({"shd.dense", [&] { for (uint32_t lc = 0; lc < nlc; ++lc) shd_dense(int(lc % 2)); }, 0});
      tw.push_back({"shd.total", [&, cb] {
        for (uint32_t lc = 0; lc < nlc; ++lc) {
          const l0::Mem* it = lcs[lc].items_r[cb.a.mi()];
          const uint32_t n_it = bound_r[cb.a.mi()];
          cx.launch(*cb.gu, GRID(cb, n_it, NG), {PtrArg(it->ptr()), PtrArg(A.ptr()), PtrArg(xs8.ptr()),
                                                 PtrArg(dgu[lc % 2]->ptr()), PtrArg(h_f.ptr()), arg_val(0u)});
          cx.launch(*cb.dn, GRID(cb, n_it, H), {PtrArg(it->ptr()), PtrArg(h_c.ptr()), PtrArg(xs8.ptr()),
                                                PtrArg(ddn[lc % 2]->ptr()), PtrArg(y_f.ptr()), arg_val(0u)});
          shd_dense(int(lc % 2));
        }
      }, 0});
    }
  }
  auto timed = [&](const std::function<void()>& f) {
    cx.timestamp(stamp);
    f();
    cx.timestamp(stamp + 1);
    cx.wait();
    return cx.timestamp_ms(stamp[0], stamp[1]);
  };
  for (Timed& t : tw) t.run();   // warm-up
  cx.wait();
  std::vector<std::vector<double>> ms(tw.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < tw.size(); ++j) {
      const size_t i = (j + size_t(r) * 5) % tw.size();
      ms[i].push_back(timed(tw[i].run));
    }
  std::printf("\n## %s C %u: ms a layer and chunk (median of %d interleaved rounds over %u layer-chunks)\n\n", m.tag,
              C, rounds, nlc);
  std::printf("| arm | ms / layer-chunk | TFLOP/s (useful) | walk ms |\n|---|---:|---:|---:|\n");
  for (size_t i = 0; i < tw.size(); ++i) {
    const double w = median(ms[i]);
    std::printf("| %s | %.4f | %.1f | %.2f |\n", tw[i].name.c_str(), w / nlc,
                tw[i].flop > 0 ? tw[i].flop / (w * 1e-3) / 1e12 : 0.0, w);
    std::printf("RESULT {\"model\":\"%s\",\"C\":%u,\"arm\":\"%s\",\"ms_per_lc\":%.6f,\"walk_ms\":%.4f,\"lc\":%u}\n",
                m.tag, C, tw[i].name.c_str(), w / nlc, w, nlc);
  }
  // ---- the gates of phase 3 Task 1, per layer-chunk (the record copies the verdicts) ----
  std::map<std::string, double> per_lc;
  for (size_t i = 0; i < tw.size(); ++i) per_lc[tw[i].name] = median(ms[i]) / nlc;
  auto have = [&](const std::string& k) { return per_lc.count(k) > 0; };
  const std::string ch = kChosen;
  if (have(ch + ".gu") && have(ch + ".dn")) {
    const double base = per_lc[ch + ".gu"] + per_lc[ch + ".dn"];
    std::printf("GATE {\"C\":%u,\"what\":\"chosen gu+dn\",\"ms\":%.4f}\n", C, base);
    for (const auto& [k, v] : per_lc) {
      const size_t at = k.find(".gu");
      if (k.rfind("F1.", 0) != 0 || at == std::string::npos || !have(k.substr(0, at) + ".dn") ||
          k.find(".c4x") == std::string::npos)
        continue;
      const double tot = v + per_lc[k.substr(0, at) + ".dn"];
      std::printf("GATE {\"C\":%u,\"task\":4,\"arm\":\"%s\",\"ms\":%.4f,\"ratio_vs_chosen\":%.3f,\"limit\":0.88,\"speed\":\"%s\"}\n",
                  C, k.substr(0, at).c_str(), tot, tot / base, tot / base <= 0.88 ? "PASS" : "FAIL");
    }
    if (have("shd.total") && have("chosen.total"))
      std::printf("GATE {\"C\":%u,\"task\":3,\"chosen_ms\":%.4f,\"shd_ms\":%.4f,\"routed_ms\":%.4f,\"dense_ms\":%.4f,\"saves_ms\":%.4f,\"limit_ms\":0.15,\"speed\":\"%s\"}\n",
                  C, per_lc["chosen.total"], per_lc["shd.total"], per_lc["shd.routed"], per_lc["shd.dense"],
                  per_lc["chosen.total"] - per_lc["shd.total"],
                  per_lc["chosen.total"] - per_lc["shd.total"] >= 0.15 ? "PASS" : "FAIL");
    const std::string o9 = "F1.a8b2m1n8.c2x0o9";
    if (have(o9 + ".gu") && have("gather"))
      std::printf("GATE {\"C\":%u,\"task\":6,\"chosen_gu_ms\":%.4f,\"gather_ms\":%.4f,\"indexed_gu_ms\":%.4f,\"saves_ms\":%.4f,\"limit_ms\":0.1,\"speed\":\"%s\"}\n",
                  C, per_lc[ch + ".gu"], per_lc["gather"], per_lc[o9 + ".gu"],
                  per_lc[ch + ".gu"] + per_lc["gather"] - per_lc[o9 + ".gu"],
                  per_lc[ch + ".gu"] + per_lc["gather"] - per_lc[o9 + ".gu"] >= 0.1 ? "PASS" : "FAIL");
  }
  std::fflush(stdout);
  return failed.empty() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string routes, model = "ornith", cs = "2048", arms_s = "all";
    int rounds = 5;
    uint32_t check = 4;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--routes" && i + 1 < argc) routes = argv[++i];
      else if (a == "--model" && i + 1 < argc) model = argv[++i];
      else if (a == "--C" && i + 1 < argc) cs = argv[++i];
      else if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--check" && i + 1 < argc) check = uint32_t(std::atoi(argv[++i]));
      else if (a == "--arms" && i + 1 < argc) arms_s = argv[++i];
      else if (a == "--list") list = true;
      else {
        std::fprintf(stderr, "usage: probe_moe_prefill --routes <file> --model ornith|flashnext [--C 2048,4096] "
                             "[--rounds 5] [--check 4] [--arms all|<tag>,...] [--list]\n");
        return 2;
      }
    }
    if (list) {
      for (const Arm& a : kArms) std::printf("%s\n", a.tag().c_str());
      return 0;
    }
    const Model* m = nullptr;
    for (const Model& x : kModels) if (model == x.tag) m = &x;
    if (!m) throw std::runtime_error("--model: ornith or flashnext");
    if (routes.empty()) throw std::runtime_error("--routes <file> is required");
    const Routes rt = read_routes(routes);
    const std::vector<std::string> want = arms_s == "all" ? std::vector<std::string>{} : split(arms_s);
    int rc = 0;
    for (const std::string& c : split(cs)) rc |= run(*m, rt, uint32_t(std::stoul(c)), rounds, check, want);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_moe_prefill FAILED: %s\n", e.what());
    return 2;
  }
}
