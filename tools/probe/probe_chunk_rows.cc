// probe_chunk_rows - plan 5d P0 (docs/superpowers/plans/2026-10-27-spec5d-prefill-chunk-4096.md
// Task 1): is the PRODUCTION prefill linear path cheaper per row at a 4096-row chunk than at two
// 2048-row chunks? Qwen3.8's five int4 linear shapes, each through
//   h8    pf_quant_had_K<K> once, then per 1024-column slab pf_requant_rot_L<l> + pf_gemm_i8
//         (gate||up: pf_gemm_i8_SILU into the bf16 x row) - runtime/prefill/int8.cc's walk
//   bf16  per slab pf_dequant_slab_K<K>_N<N>_L<l> + pf_gemm (gemm_l0 / gemm_l0_silu) -
//         runtime/prefill/linear_l0.cc's walk
// at M = 4096 (one call) against M = 2048 twice (rows 0..2047, then 2048..4095), the
// production kernels and launch shapes, replicated here only because linear_i8 / linear_l0
// refuse M above PrefillScratch::kC (the thing plan 5d changes).
//
//   probe_chunk_rows [--rounds 5] [--shape all|gu,down,qkvz,out,faqkv]
//
// Weights and activations are random (int4 words uniform, f16 scales U(0.005, 0.02) for layout
// 0 or in-tile for layout 1, activations bf16 N(0, 1), column scales U(0.5, 2) / 127): kernel
// TIMES do not depend on the values. Before any time: the 4096-row output equals the two
// 2048-row outputs bit for bit on both paths (every row is independent: per-token quantiser,
// the same weights) - which is also the check that the kernels index 4096 rows correctly.
//
// Timing: each arm is the GPU span between two global timestamps on the in-order list around
// the arm's launches (gaps between launches included, as the engine pays them); a warm-up
// burst, then `rounds` rounds, the four arms in a rotating order per round (interleaved;
// docs/probe-w4a8-2026-09-23.md §14.4). Medians over rounds, and the median of each round's
// paired ratio 2 x 2048 / 4096. The stop rule (plan 5d Task 1): the summed h8 linear time a
// 4096-token prompt pays (each shape x its layer count) must be >= 5 % lower at M = 4096.
//
// PROBE-ONLY: no production file is edited.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/int8_signs.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::GemmBatch;
using runtime::prefill::PtrArg;

constexpr uint32_t kNs = kernels::kPfSlabWidth;
constexpr uint32_t kBig = 4096, kSmall = 2048;

struct Shape {
  const char* tag;
  const char* name;
  uint32_t K, N, layout, layers;
  bool silu;
};
// Qwen3.8 (64 layers: 48 GDN + 16 full attention): the production slab rows of
// src/kernels/prefill/CMakeLists.txt (PF_SLAB_ROWS) with their layouts.
const Shape kShapes[] = {
    {"gu", "gate||up (SiLU epilogue)", 5120, 34816, 0, 64, true},
    {"down", "down", 17408, 5120, 0, 64, false},
    {"qkvz", "GDN qkv||z", 5120, 16384, 1, 48, false},
    {"out", "out_proj / o_proj", 6144, 5120, 0, 64, false},
    {"faqkv", "FA q||k||v", 5120, 14336, 0, 16, false},
};

void upload(l0::Context& ctx, l0::Mem& dst, const void* src, size_t bytes) {
  l0::CmdList c = l0::CmdList::immediate(ctx);
  c.copy(dst.ptr(), src, bytes);
}
void download(l0::Context& ctx, void* dst, const l0::Mem& src, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dst) + off, static_cast<const char*>(src.ptr()) + off,
           std::min(kChunk, bytes - off));
  }
}
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
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Result {
  double h8_big, h8_two, bf_big, bf_two, r_h8, r_bf;
  double h8_lo, h8_hi, bf_lo, bf_hi;
};

Result run_shape(l0::Context& ctx, runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc,
                 const Shape& sh, int rounds, std::mt19937& rng) {
  const uint32_t K = sh.K, N = sh.N;
  // --- data ---------------------------------------------------------------
  const size_t w_u32 = sh.layout == 0 ? size_t(K / 8) * N : size_t(N / 16) * (K / 64) * 136;
  std::vector<uint32_t> w(w_u32);
  for (uint32_t& v : w) v = rng();
  if (sh.layout == 1)   // the 8 scale words of every 136-word tile: 16 f16 scales
    for (size_t t = 0; t < w_u32 / 136; ++t)
      for (uint32_t j = 0; j < 8; ++j) {
        std::uniform_real_distribution<float> u(0.005f, 0.02f);
        w[t * 136 + 128 + j] = uint32_t(f32_to_f16(u(rng))) | (uint32_t(f32_to_f16(u(rng))) << 16);
      }
  std::vector<uint16_t> sc(sh.layout == 0 ? size_t(K / 64) * N : 0);
  {
    std::uniform_real_distribution<float> u(0.005f, 0.02f);
    for (uint16_t& s : sc) s = f32_to_f16(u(rng));
  }
  std::vector<uint16_t> x(size_t(kBig) * K);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  std::vector<float> ws(N), inv(N);
  {
    std::uniform_real_distribution<float> u(0.5f, 2.0f);
    for (uint32_t n = 0; n < N; ++n) { ws[n] = u(rng) / 127.0f; inv[n] = 1.0f / ws[n]; }
  }
  const std::vector<float> sgn = runtime::prefill::int8_signs(K);
  const std::vector<uint32_t> bits = runtime::prefill::int8_sign_bits(K);

  l0::Mem dw(ctx, l0::MemKind::Device, w.size() * 4);
  l0::Mem dsc(ctx, l0::MemKind::Device, std::max<size_t>(sc.size() * 2, 64));
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  l0::Mem dws(ctx, l0::MemKind::Device, size_t(N) * 4), dinv(ctx, l0::MemKind::Device, size_t(N) * 4);
  l0::Mem dsgn(ctx, l0::MemKind::Device, size_t(K) * 4), dbits(ctx, l0::MemKind::Device, bits.size() * 4);
  upload(ctx, dw, w.data(), w.size() * 4);
  if (!sc.empty()) upload(ctx, dsc, sc.data(), sc.size() * 2);
  upload(ctx, dx, x.data(), x.size() * 2);
  upload(ctx, dws, ws.data(), size_t(N) * 4);
  upload(ctx, dinv, inv.data(), size_t(N) * 4);
  upload(ctx, dsgn, sgn.data(), size_t(K) * 4);
  upload(ctx, dbits, bits.data(), bits.size() * 4);
  // scratch at 4096 rows: int8 xq / xs, the int8 and bf16 slabs, the output (fp32 [M][N], or
  // the bf16 x row [M][N/2] for the SiLU epilogue)
  l0::Mem xq(ctx, l0::MemKind::Device, size_t(kBig) * K), xs(ctx, l0::MemKind::Device, size_t(kBig) * 4);
  l0::Mem w8(ctx, l0::MemKind::Device, size_t(K / 4) * kNs * 4);
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(K) * kNs * 2);
  const size_t out_bytes = sh.silu ? size_t(kBig) * (N / 2) * 2 : size_t(kBig) * N * 4;
  l0::Mem out_big(ctx, l0::MemKind::Device, out_bytes), out_two(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  const void* scales = sh.layout == 0 ? dsc.ptr() : nullptr;
  l0::Kernel& quant = kc(kernels::pf_quant_had_variant(K), "pf_quant_had");
  l0::Kernel& requant = kc(kernels::pf_requant_rot_variant(sh.layout), "pf_requant_rot");
  l0::Kernel& gemm8 = kc(kernels::pf_gemm_i8_variant(sh.silu), "pf_gemm_i8");
  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");

  // the output's row r: element offset of row r (fp32 [M][N] or bf16 [M][N/2])
  auto out_row = [&](l0::Mem& o, uint32_t r) -> char* {
    return static_cast<char*>(o.ptr()) + (sh.silu ? size_t(r) * (N / 2) * 2 : size_t(r) * N * 4);
  };
  // int8.cc's walk at M rows from row r0 of x into o from row r0
  auto h8 = [&](uint32_t r0, uint32_t M, l0::Mem& o) {
    const uint16_t* xa = dx.as<uint16_t>() + size_t(r0) * K;
    char* ob = out_row(o, r0);
    const uint32_t ldxq = K / 2;
    const uint32_t ldc = sh.silu ? N / 2 : N;
    cx.launch(quant, M, 1, 1,
              {PtrArg(xa), PtrArg(dsgn.ptr()), PtrArg(xq.ptr()), PtrArg(xs.ptr()), arg_val(K)});
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      cx.launch(requant, kNs / 16, K / 1024, 1,
                {PtrArg(dw.ptr()), PtrArg(scales), PtrArg(dbits.ptr()), PtrArg(dinv.ptr()),
                 PtrArg(w8.ptr()), arg_val(n0), arg_val(N), arg_val(K), arg_val(kNs)});
      void* c = sh.silu ? static_cast<void*>(reinterpret_cast<uint16_t*>(ob) + n0 / 2)
                        : static_cast<void*>(reinterpret_cast<float*>(ob) + n0);
      cx.launch(gemm8, M / 256, kNs / 128, 1,
                {PtrArg(xq.ptr()), PtrArg(xs.ptr()), PtrArg(w8.ptr()), PtrArg(dws.as<float>() + n0),
                 PtrArg(c), arg_val(M), arg_val(K), arg_val(kNs), arg_val(ldxq), arg_val(kNs),
                 arg_val(ldc)});
    }
  };
  // linear_l0.cc's walk
  auto bf = [&](uint32_t r0, uint32_t M, l0::Mem& o) {
    const uint16_t* xa = dx.as<uint16_t>() + size_t(r0) * K;
    char* ob = out_row(o, r0);
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      cx.launch(dq, kNs / 16, K / 64, 1,
                {PtrArg(dw.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
      if (sh.silu) {
        const GemmBatch b{M, K, kNs, 1, K, kNs, 0, 0, 0, 0};
        runtime::prefill::gemm_l0_silu(cx, kc, b, xa, slab.as<uint16_t>(),
                                       reinterpret_cast<uint16_t*>(ob) + n0 / 2, N / 2);
      } else {
        const GemmBatch b{M, K, kNs, 1, K, kNs, N, 0, 0, 0};
        runtime::prefill::gemm_l0(cx, kc, b, xa, slab.as<uint16_t>(),
                                  reinterpret_cast<float*>(ob) + n0, false);
      }
    }
  };
  // an arm: timestamps around its launches, one host wait, GPU ms
  auto timed = [&](auto&& launches) {
    cx.timestamp(stamp);
    launches();
    cx.timestamp(stamp + 1);
    cx.wait();
    return cx.timestamp_ms(stamp[0], stamp[1]);
  };
  auto h8_big = [&] { h8(0, kBig, out_big); };
  auto h8_two = [&] { h8(0, kSmall, out_two); h8(kSmall, kSmall, out_two); };
  auto bf_big = [&] { bf(0, kBig, out_big); };
  auto bf_two = [&] { bf(0, kSmall, out_two); bf(kSmall, kSmall, out_two); };

  // --- value checks: 4096 rows == 2 x 2048 rows, every output word ----------
  std::vector<uint8_t> a(out_bytes), b(out_bytes);
  for (int path = 0; path < 2; ++path) {
    std::vector<uint8_t> zero(out_bytes, 0xA5);   // poison: a row the big call skips shows
    upload(ctx, out_big, zero.data(), out_bytes);
    upload(ctx, out_two, zero.data(), out_bytes);
    if (path == 0) { h8_big(); cx.wait(); h8_two(); } else { bf_big(); cx.wait(); bf_two(); }
    cx.wait();
    download(ctx, a.data(), out_big, out_bytes);
    download(ctx, b.data(), out_two, out_bytes);
    size_t diff = 0, poisoned = 0;
    const size_t words = out_bytes / 4;
    const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
    const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
    for (size_t i = 0; i < words; ++i) {
      diff += pa[i] != pb[i];
      poisoned += pa[i] == 0xA5A5A5A5u;
    }
    std::printf("| %s | %s: 4096 rows vs 2 x 2048, all %zu words | **%zu differ, %zu unwritten** |\n",
                sh.tag, path == 0 ? "h8" : "bf16", words, diff, poisoned);
    if (diff || poisoned) throw std::runtime_error(std::string(sh.tag) + ": the 4096-row call is not the two 2048-row calls");
  }
  std::fflush(stdout);

  // --- timing ---------------------------------------------------------------
  for (int i = 0; i < 6; ++i) { h8_big(); h8_two(); bf_big(); bf_two(); }
  cx.wait();
  std::vector<double> t[4], r_h8, r_bf;
  for (int r = 0; r < rounds; ++r) {
    double v[4];
    for (int j = 0; j < 4; ++j) {
      const int arm = (j + r) % 4;
      v[arm] = arm == 0 ? timed(h8_big) : arm == 1 ? timed(h8_two) : arm == 2 ? timed(bf_big) : timed(bf_two);
    }
    for (int j = 0; j < 4; ++j) t[j].push_back(v[j]);
    r_h8.push_back(v[1] / v[0]);
    r_bf.push_back(v[3] / v[2]);
  }
  Result res{median(t[0]), median(t[1]), median(t[2]), median(t[3]), median(r_h8), median(r_bf),
             *std::min_element(r_h8.begin(), r_h8.end()), *std::max_element(r_h8.begin(), r_h8.end()),
             *std::min_element(r_bf.begin(), r_bf.end()), *std::max_element(r_bf.begin(), r_bf.end())};
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 5;
    std::string want = "all";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else { std::fprintf(stderr, "usage: probe_chunk_rows [--rounds 5] [--shape all|gu,down,qkvz,out,faqkv]\n"); return 2; }
    }
    if (rounds < 1) throw std::runtime_error("--rounds must be >= 1");
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_chunk_rows (plan 5d P0): production prefill linears, M = 4096 vs 2 x 2048\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n\n", mask ? mask : "(unset)",
                ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    std::mt19937 rng(5);
    std::printf("## checks\n\n| shape | check | result |\n|---|---|---|\n");
    std::vector<std::pair<const Shape*, Result>> rows;
    for (const Shape& sh : kShapes) {
      if (want != "all" && ("," + want + ",").find("," + std::string(sh.tag) + ",") == std::string::npos) continue;
      rows.emplace_back(&sh, run_shape(ctx, cx, kc, sh, rounds, rng));
    }
    std::printf("\n## times (GPU span per arm, medians of %d interleaved rounds; ratio = the median "
                "of each round's 2 x 2048 / 4096)\n\n", rounds);
    std::printf("| shape | layers | h8 4096 ms | h8 2 x 2048 ms | **h8 ratio** (min..max) | bf16 4096 ms | "
                "bf16 2 x 2048 ms | bf16 ratio (min..max) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n");
    double s_h8_big = 0, s_h8_two = 0, s_bf_big = 0, s_bf_two = 0;
    for (const auto& [sh, r] : rows) {
      std::printf("| %s | %u | %.3f | %.3f | **%.3fx** (%.3f..%.3f) | %.3f | %.3f | %.3fx (%.3f..%.3f) |\n",
                  sh->name, sh->layers, r.h8_big, r.h8_two, r.r_h8, r.h8_lo, r.h8_hi, r.bf_big,
                  r.bf_two, r.r_bf, r.bf_lo, r.bf_hi);
      s_h8_big += sh->layers * r.h8_big;
      s_h8_two += sh->layers * r.h8_two;
      s_bf_big += sh->layers * r.bf_big;
      s_bf_two += sh->layers * r.bf_two;
    }
    std::printf("\n## a 4096-token prompt's linears (each shape x its layers; sums of medians)\n\n");
    std::printf("| path | one 4096 chunk ms | two 2048 chunks ms | saving |\n|---|---:|---:|---:|\n");
    const double save_h8 = 1.0 - s_h8_big / s_h8_two, save_bf = 1.0 - s_bf_big / s_bf_two;
    std::printf("| **h8 (l0-int8)** | %.1f | %.1f | **%.1f %%** |\n", s_h8_big, s_h8_two, 100 * save_h8);
    std::printf("| bf16 (l0) | %.1f | %.1f | %.1f %% |\n", s_bf_big, s_bf_two, 100 * save_bf);
    if (rows.size() == sizeof kShapes / sizeof kShapes[0])
      std::printf("\nSTOP RULE (h8 summed linear time >= 5 %% lower at 4096): **%s** (%.1f %%)\n",
                  save_h8 >= 0.05 ? "PASS - the plan continues" : "FAIL - the plan stops, A13 stands",
                  100 * save_h8);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_chunk_rows FAILED: %s\n", e.what());
    return 2;
  }
}
