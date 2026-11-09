// probe_prefill_split - open-work 3k: what a full Qwen3.8 prefill chunk's int4 linears cost on the
// two prefill backends, split into their kernels, and the multiply's achieved TFLOP/s. Record:
// docs/probes/probe-prefill-chunk-split-2026-11-10.md. Production kernels only - nothing here is a
// candidate; the probe changes no runtime path.
//
// 3i's distinct-layer walk (probe_fused_small --weights layers): each of Qwen3.8's five int4 linear
// families gets one live allocation of independent synthetic weights per layer (the model's shapes
// and footprint), visited in ascending order; families run one after another. Arms per family:
//   l0          runtime/prefill/linear_l0.cc's walk at M rows: per 1024-column slab pf_dequant_slab,
//               then gemm_l0 (pf_gemm above 1024 rows; gate||up with its SiLU epilogue)
//   l0_dq       the walk's dequants alone
//   l0_gemm     the walk's GEMMs alone (each slab reads the slab buffer as it stands)
//   i8          runtime/prefill/int8.cc's walk: pf_quant_had once, then per slab pf_requant_rot and
//               pf_gemm_i8 (plain, or gate||up's SiLU form), with each layer's own column scales
//               (pf_colmax_rot + the host finish, as Int8State::compute_scales)
//   i8_quant / i8_requant / i8_gemm   its three kernels alone
// Sanity, before timing: layer 0's i8 output against l0's (the same linear, int8 vs bf16
// arithmetic) must agree to a few per cent RMS - an int8 arm computing nothing would fail it.
//
// Timing (probe_fused_small's): each arm is the GPU span between two global timestamps around its
// whole layer walk on the in-order list, plus host wall; one warm-up pass of every arm, then
// `rounds` rounds with the arms in a rotating order. Medians. TFLOP/s = 2 M K N layers / span.
//
//   probe_prefill_split [--rounds 5] [--rows 2048] [--shape all|gu,down,qkvz,out,faqkv]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/int8.h"
#include "runtime/prefill/int8_signs.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::pad256;
using runtime::prefill::PtrArg;

constexpr uint32_t kNs = kernels::kPfSlabWidth;
constexpr uint32_t kMaxRows = 4096;
constexpr uint32_t kMaxK = 17408;
constexpr uint32_t kRot = 1024;

struct Shape {
  const char* tag;
  const char* name;
  uint32_t K, N, layout, layers;
  bool silu;
};
const Shape kShapes[] = {
    {"gu", "gate||up (SiLU epilogue)", 5120, 34816, 0, 64, true},
    {"down", "down", 17408, 5120, 0, 64, false},
    {"qkvz", "GDN qkv||z", 5120, 16384, 1, 48, false},
    {"out", "out_proj / o_proj", 6144, 5120, 0, 64, false},
    {"faqkv", "FA q||k||v", 5120, 14336, 0, 16, false},
};

struct Layer {
  l0::Mem words;
  std::unique_ptr<l0::Mem> scales;
  std::unique_ptr<l0::Mem> ws, inv;
  Layer(l0::Context& ctx, size_t wb, size_t sb) : words(ctx, l0::MemKind::Device, wb) {
    if (sb) scales = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sb);
  }
  const void* scale_ptr() const { return scales ? scales->ptr() : nullptr; }
};

void copy(l0::Context& ctx, void* dst, const void* src, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off, std::min(kChunk, bytes - off));
  }
}
uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf16_to_f32(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
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
double gflop(const Shape& sh, uint32_t M) { return 2.0 * M * double(sh.K) * sh.N * sh.layers / 1e9; }

// arm -> rows -> median whole-walk GPU ms
using Table = std::map<std::string, std::map<uint32_t, double>>;

Table run_shape(l0::Context& ctx, runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc,
                runtime::prefill::Int8State& q, const Shape& sh, const std::vector<uint32_t>& rows, int rounds,
                std::mt19937& rng) {
  const uint32_t K = sh.K, N = sh.N;
  const size_t w_u32 = sh.layout == 0 ? size_t(K / 8) * N : size_t(N / 16) * (K / 64) * 136;
  const size_t sc_n = sh.layout == 0 ? size_t(K / 64) * N : 0;
  std::vector<uint32_t> w(w_u32);
  std::vector<uint16_t> sc(sc_n);
  auto fill = [&] {
    for (uint32_t& v : w) v = rng();
    std::uniform_real_distribution<float> u(0.005f, 0.02f);
    if (sh.layout == 1)
      for (size_t t = 0; t < w_u32 / 136; ++t)
        for (uint32_t j = 0; j < 8; ++j)
          w[t * 136 + 128 + j] = uint32_t(f32_to_f16(u(rng))) | (uint32_t(f32_to_f16(u(rng))) << 16);
    for (uint16_t& s : sc) s = f32_to_f16(u(rng));
  };
  const l0::Mem& bits = q.sign_bits(K);
  const l0::Mem& sgn = q.signs_f32(K);
  l0::Kernel& colmax_k = kc(kernels::pf_requant_rot_variant(sh.layout), "pf_colmax_rot");
  std::vector<std::unique_ptr<Layer>> bank;
  l0::Mem colmax(ctx, l0::MemKind::Device, size_t(N) * 4);
  std::vector<uint32_t> zero(N, 0u), mx(N);
  std::vector<float> ws(N), inv(N);
  for (uint32_t l = 0; l < sh.layers; ++l) {
    fill();
    bank.push_back(std::make_unique<Layer>(ctx, w.size() * 4, sc.size() * 2));
    Layer& L = *bank.back();
    copy(ctx, L.words.ptr(), w.data(), w.size() * 4);
    if (L.scales) copy(ctx, L.scales->ptr(), sc.data(), sc.size() * 2);
    // Int8State::compute_scales, for this layer's weight
    copy(ctx, colmax.ptr(), zero.data(), zero.size() * 4);
    cx.launch(colmax_k, N / 16, K / kRot, 1,
              {PtrArg(L.words.ptr()), PtrArg(L.scale_ptr()), PtrArg(bits.ptr()), PtrArg(colmax.ptr()),
               arg_val(0u), arg_val(N), arg_val(K)});
    cx.wait();
    copy(ctx, mx.data(), colmax.ptr(), mx.size() * 4);
    for (uint32_t n = 0; n < N; ++n) {
      float m;
      std::memcpy(&m, &mx[n], 4);
      ws[n] = m > 0.0f ? m / 127.0f : 1.0f;
      inv[n] = 1.0f / ws[n];
    }
    L.ws = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(N) * 4);
    L.inv = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(N) * 4);
    copy(ctx, L.ws->ptr(), ws.data(), N * 4);
    copy(ctx, L.inv->ptr(), inv.data(), N * 4);
  }
  std::printf("# %s: %u distinct layers, %zu weight bytes each\n", sh.tag, sh.layers,
              w.size() * 4 + sc.size() * 2);

  std::vector<uint16_t> x(size_t(kMaxRows) * K);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  copy(ctx, dx.ptr(), x.data(), x.size() * 2);
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(K) * kNs * 2);
  const size_t out_bytes = size_t(kMaxRows) * N * 4;
  l0::Mem out_a(ctx, l0::MemKind::Device, out_bytes), out_b(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");
  l0::Kernel& quant = kc(kernels::pf_quant_had_variant(K), "pf_quant_had");
  l0::Kernel& requant = kc(kernels::pf_requant_rot_variant(sh.layout), "pf_requant_rot");
  l0::Kernel& gemm8 = kc(kernels::pf_gemm_i8_variant(sh.silu), "pf_gemm_i8");
  l0::Mem& w8 = q.w8_slab();

  auto deq = [&](uint32_t n0, uint32_t l) {
    cx.launch(dq, kNs / 16, K / 64, 1,
              {PtrArg(bank[l]->words.ptr()), PtrArg(bank[l]->scale_ptr()), PtrArg(slab.ptr()), arg_val(n0)});
  };
  auto gemm16 = [&](uint32_t M, uint32_t n0, l0::Mem& o) {
    if (sh.silu) {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, 0, 0, 0, 0};
      runtime::prefill::gemm_l0_silu(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(),
                                     o.as<uint16_t>() + n0 / 2, N / 2);
    } else {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, N, 0, 0, 0};
      runtime::prefill::gemm_l0(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(), o.as<float>() + n0, false);
    }
  };
  auto q8 = [&](uint32_t M) {
    cx.launch(quant, pad256(M), 1, 1,
              {PtrArg(dx.ptr()), PtrArg(sgn.ptr()), PtrArg(q.xq().ptr()), PtrArg(q.xs().ptr()), arg_val(K)});
  };
  auto rq = [&](uint32_t n0, uint32_t l) {
    cx.launch(requant, kNs / 16, K / kRot, 1,
              {PtrArg(bank[l]->words.ptr()), PtrArg(bank[l]->scale_ptr()), PtrArg(bits.ptr()),
               PtrArg(bank[l]->inv->ptr()), PtrArg(w8.ptr()), arg_val(n0), arg_val(N), arg_val(K), arg_val(kNs)});
  };
  auto g8 = [&](uint32_t M, uint32_t n0, uint32_t l, l0::Mem& o) {
    const uint32_t Mp = pad256(M);
    void* c = sh.silu ? static_cast<void*>(o.as<uint16_t>() + n0 / 2) : static_cast<void*>(o.as<float>() + n0);
    const uint32_t ldc = sh.silu ? N / 2 : N;
    cx.launch(gemm8, Mp / 256, kNs / 128, 1,
              {PtrArg(q.xq().ptr()), PtrArg(q.xs().ptr()), PtrArg(w8.ptr()),
               PtrArg(bank[l]->ws->as<float>() + n0), PtrArg(c), arg_val(Mp), arg_val(K), arg_val(kNs),
               arg_val(K / 2), arg_val(kNs), arg_val(ldc)});
  };
  auto l0_walk = [&](uint32_t M, uint32_t l, l0::Mem& o) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(n0, l);
      gemm16(M, n0, o);
    }
  };
  auto i8_walk = [&](uint32_t M, uint32_t l, l0::Mem& o) {
    q8(M);
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      rq(n0, l);
      g8(M, n0, l, o);
    }
  };

  // --- sanity: layer 0, int8 vs bf16 arithmetic of the same linear --------------------------------
  for (uint32_t M : rows) {
    l0_walk(M, 0, out_a);
    i8_walk(M, 0, out_b);
    cx.wait();
    const size_t n = size_t(M) * (sh.silu ? N / 2 : N);
    std::vector<float> a(n), b(n);
    if (sh.silu) {
      std::vector<uint16_t> ha(n), hb(n);
      copy(ctx, ha.data(), out_a.ptr(), n * 2);
      copy(ctx, hb.data(), out_b.ptr(), n * 2);
      for (size_t i = 0; i < n; ++i) a[i] = bf16_to_f32(ha[i]), b[i] = bf16_to_f32(hb[i]);
    } else {
      copy(ctx, a.data(), out_a.ptr(), n * 4);
      copy(ctx, b.data(), out_b.ptr(), n * 4);
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) num += double(a[i] - b[i]) * (a[i] - b[i]), den += double(a[i]) * a[i];
    const double rel = den > 0 ? std::sqrt(num / den) : 1.0;
    std::printf("SANITY {\"shape\":\"%s\",\"rows\":%u,\"i8_vs_l0_rel_rms\":%.6f,\"ref_rms\":%.6g}\n", sh.tag, M,
                rel, std::sqrt(den / double(n)));
    if (!(den > 0) || !(rel < 0.1))
      throw std::runtime_error(std::string(sh.tag) + ": the int8 walk does not compute the bf16 walk's linear");
  }

  struct Arm {
    std::string name;
    uint32_t M;
    std::function<void()> run;
  };
  std::vector<Arm> arms;
  const uint32_t L = sh.layers;
  for (uint32_t M : rows) {
    arms.push_back({"l0", M, [&, M] { for (uint32_t l = 0; l < L; ++l) l0_walk(M, l, out_a); }});
    arms.push_back({"l0_dq", M, [&] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(n0, l); }});
    arms.push_back({"l0_gemm", M, [&, M] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) gemm16(M, n0, out_a); }});
    arms.push_back({"i8", M, [&, M] { for (uint32_t l = 0; l < L; ++l) i8_walk(M, l, out_b); }});
    arms.push_back({"i8_quant", M, [&, M] { for (uint32_t l = 0; l < L; ++l) q8(M); }});
    arms.push_back({"i8_requant", M, [&] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) rq(n0, l); }});
    arms.push_back({"i8_gemm", M, [&, M] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) g8(M, n0, l, out_b); }});
  }
  auto timed = [&](const std::function<void()>& launches) {
    const auto start = std::chrono::steady_clock::now();
    cx.timestamp(stamp);
    launches();
    cx.timestamp(stamp + 1);
    cx.wait();
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double gpu = cx.timestamp_ms(stamp[0], stamp[1]);
    if (!(gpu > 0 && wall > 0 && std::isfinite(gpu) && std::isfinite(wall)))
      throw std::runtime_error("invalid whole-walk timing");
    return std::make_pair(gpu, wall);
  };
  for (Arm& arm : arms) arm.run();   // warm-up
  cx.wait();
  std::vector<std::vector<double>> t(arms.size()), wall(arms.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < arms.size(); ++j) {
      const size_t i = (j + size_t(r) * 3) % arms.size();
      const auto ms = timed(arms[i].run);
      t[i].push_back(ms.first);
      wall[i].push_back(ms.second);
      std::printf("SAMPLE {\"shape\":\"%s\",\"round\":%d,\"arm\":\"%s\",\"rows\":%u,\"layers\":%u,"
                  "\"gpu_ms\":%.6f,\"wall_ms\":%.6f}\n",
                  sh.tag, r, arms[i].name.c_str(), arms[i].M, L, ms.first, ms.second);
    }
  Table tab;
  for (size_t i = 0; i < arms.size(); ++i) {
    const double gpu = median(t[i]);
    tab[arms[i].name][arms[i].M] = gpu;
    std::printf("WALK {\"shape\":\"%s\",\"arm\":\"%s\",\"rows\":%u,\"layers\":%u,\"gpu_ms\":%.6f,"
                "\"wall_ms\":%.6f,\"gflop\":%.3f,\"tflops\":%.2f}\n",
                sh.tag, arms[i].name.c_str(), arms[i].M, L, gpu, median(wall[i]), gflop(sh, arms[i].M),
                gflop(sh, arms[i].M) / gpu);
  }
  return tab;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(tok);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 5;
    std::string want = "all", rows_s = "2048";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else {
        std::fprintf(stderr, "usage: probe_prefill_split [--rounds 5] [--rows 2048] "
                             "[--shape all|gu,down,qkvz,out,faqkv]\n");
        return 2;
      }
    }
    std::set<std::string> selected;
    for (const auto& s : split(want)) selected.insert(s);
    std::vector<uint32_t> rows;
    for (const auto& r : split(rows_s)) {
      const long m = std::atol(r.c_str());
      if (m < 1 || m > long(kMaxRows)) throw std::runtime_error("--rows: 1.." + std::to_string(kMaxRows));
      rows.push_back(uint32_t(m));
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_prefill_split (3k): synthetic distinct-layer weights at Qwen3.8's shapes\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n", mask ? mask : "(unset)", ctx.name().c_str(),
                rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    runtime::prefill::Int8State q(ctx, kMaxK, kMaxRows);
    std::mt19937 rng(11);
    Table sum;
    std::map<uint32_t, double> fl;
    for (const Shape& sh : kShapes) {
      if (want != "all" && !selected.count(sh.tag)) continue;
      const Table tab = run_shape(ctx, cx, kc, q, sh, rows, rounds, rng);
      for (const auto& [arm, by_m] : tab)
        for (const auto& [M, ms] : by_m) sum[arm][M] += ms;
      for (uint32_t M : rows) fl[M] += gflop(sh, M);
      std::fflush(stdout);
    }
    std::printf("\n## one chunk's int4 linears (sum of measured whole-walk medians, GPU ms; TFLOP/s on the multiply's FLOP)\n\n");
    std::printf("| arm | rows | ms | TFLOP/s |\n|---|---:|---:|---:|\n");
    for (const auto& [arm, by_m] : sum)
      for (const auto& [M, ms] : by_m) {
        std::printf("| %s | %u | %.1f | %.1f |\n", arm.c_str(), M, ms, fl[M] / ms);
        std::printf("CHUNK {\"arm\":\"%s\",\"rows\":%u,\"gpu_ms\":%.3f,\"gflop\":%.1f,\"tflops\":%.2f}\n",
                    arm.c_str(), M, ms, fl[M], fl[M] / ms);
      }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_prefill_split FAILED: %s\n", e.what());
    return 2;
  }
}
