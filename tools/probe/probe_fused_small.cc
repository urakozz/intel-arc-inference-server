// probe_fused_small - open-work 3b: a short prefill chunk's int4 linears read straight from the int4
// weights (pf_gemm_q4.cl, no bf16 slab) against the slab walk. Qwen3.8's five int4 linear shapes at
// M rows, each through
//   l0_now  runtime/prefill/linear_l0.cc's walk as built: per 1024-column slab pf_dequant_slab, then
//           gemm_l0 (pf_small_tile's pf_gemm_small up to 1024 rows, pf_gemm above) - the control
//   dq      the walk's dequants alone (every slab)
//   qXbYsZ  pf_gemm_q4_A<X>_B<Y>_S<Z>_L<layout>: X 8-row blocks x Y 16-column atoms a sub-group, Z
//           sub-groups a work-group, ONE launch over the linear's whole N, rows padded to 8 X only
// gate||up runs its SiLU epilogue (bf16 x), as the engine does. Synthetic weights have the real
// shapes and, by default, independent allocations for every layer. --weights resident reproduces
// the old single-matrix protocol; only that mode extrapolates one measured linear to a chunk.
//
// The gate, before any time: every qXbYsZ output equals the slab walk's bit for bit over the M real
// rows (memcmp as 32-bit words) at every layer, row count and shape - plain for all five, the SiLU epilogue
// for gate||up, and gate||up plain as well (the fp32 C the SiLU epilogue consumes). A difference
// throws before timing that shape.
//
// Timing (probe_small_chunk's): each arm is the GPU span between two global timestamps around its
// complete layer walk on the in-order list, plus its host wall; a complete warm-up pass, then
// `rounds` rounds with the arms in a rotating
// order (interleaved - sequential arms mis-state ratios by up to 35 %). Medians. The l0_now arm runs
// each slab's GEMM right after its dequant. It isolates linears, not surrounding model operators.
// SAMPLE/WALK records retain measured whole-walk times; the per-linear table divides by copies.
//
//   probe_fused_small [--rounds 7] [--rows 1,16,32,64,128,256,512,1024]
//                     [--shape all|gu,down,qkvz,out,faqkv] [--q4 q1b2s1,...]
//                     [--weights layers|resident] [--plan] [--negative-check]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <limits>
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
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr uint32_t kNs = kernels::kPfSlabWidth;
constexpr uint32_t kMaxRows = 1024;

struct Shape {
  const char* tag;
  const char* name;
  uint32_t K, N, layout, layers;
  bool silu;
};
// Qwen3.8 (64 layers: 48 GDN + 16 full attention), probe_small_chunk's table.
const Shape kShapes[] = {
    {"gu", "gate||up (SiLU epilogue)", 5120, 34816, 0, 64, true},
    {"down", "down", 17408, 5120, 0, 64, false},
    {"qkvz", "GDN qkv||z", 5120, 16384, 1, 48, false},
    {"out", "out_proj / o_proj", 6144, 5120, 0, 64, false},
    {"faqkv", "FA q||k||v", 5120, 14336, 0, 16, false},
};

size_t weight_bytes(const Shape& sh) {
  return size_t(sh.K) * sh.N / 2 + size_t(sh.K / 64) * sh.N * 2;
}

void print_plan(const Shape& sh, bool layers) {
  const size_t copies = layers ? sh.layers : 1;
  std::printf("PLAN {\"shape\":\"%s\",\"weight_mode\":\"%s\",\"weight_source\":\"synthetic\","
              "\"layout\":%u,\"model_layers\":%u,\"weight_copies\":%zu,\"measured_linears\":%zu,"
              "\"weight_bytes_per_copy\":%zu,\"distinct_weight_bytes\":%zu,\"extrapolated\":%s}\n",
              sh.tag, layers ? "layers" : "resident", sh.layout, sh.layers, copies, copies,
              weight_bytes(sh), copies * weight_bytes(sh), layers ? "false" : "true");
}

struct DeviceWeights {
  l0::Mem words;
  std::unique_ptr<l0::Mem> scales;
  DeviceWeights(l0::Context& ctx, size_t words_bytes, size_t scales_bytes)
      : words(ctx, l0::MemKind::Device, words_bytes) {
    if (scales_bytes) scales = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, scales_bytes);
  }
  const void* scale_ptr() const { return scales ? scales->ptr() : nullptr; }
};

struct Q4 {
  uint32_t na, nb, sn;
  bool split = false;
  bool slm = false;
  std::string tag() const {
    return "q" + std::to_string(na) + "b" + std::to_string(nb) + "s" + std::to_string(sn) +
           (slm ? "slm" : split ? "split" : "");
  }
  uint32_t rows() const { return 8 * na; }
  uint32_t cols() const { return (split ? 32 : 16) * nb * sn; }
};
uint32_t pad_to(uint32_t v, uint32_t t) { return (v + t - 1) / t * t; }

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

// arm name -> rows -> median ms (one shape)
using Table = std::map<std::string, std::map<uint32_t, double>>;

Table run_shape(l0::Context& ctx, runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc,
                const Shape& sh, const std::vector<uint32_t>& rows, const std::vector<Q4>& q4s, int rounds,
                std::mt19937& rng, size_t& checks, bool layers, bool negative) {
  const uint32_t K = sh.K, N = sh.N;
  const size_t copies = layers ? sh.layers : 1;
  const size_t out_bytes = size_t(kMaxRows) * N * 4;
  const size_t required = copies * weight_bytes(sh) + size_t(kMaxRows) * K * 2 +
                          size_t(K) * kNs * 2 + 2 * out_bytes + 64;
  if (required + (256ul << 20) > ctx.memory_bytes())
    throw std::runtime_error(std::string(sh.tag) + ": weight walk exceeds device memory with 256 MiB reserve");
  // --- data (probe_small_chunk's) -------------------------------------------
  const size_t w_u32 = sh.layout == 0 ? size_t(K / 8) * N : size_t(N / 16) * (K / 64) * 136;
  std::vector<uint32_t> w(w_u32);
  std::vector<uint16_t> sc(sh.layout == 0 ? size_t(K / 64) * N : 0);
  auto fill_weights = [&](std::mt19937& gen) {
    for (uint32_t& v : w) v = gen();
    if (sh.layout == 1)
      for (size_t t = 0; t < w_u32 / 136; ++t)
        for (uint32_t j = 0; j < 8; ++j) {
          std::uniform_real_distribution<float> u(0.005f, 0.02f);
          w[t * 136 + 128 + j] = uint32_t(f32_to_f16(u(gen))) | (uint32_t(f32_to_f16(u(gen))) << 16);
        }
    std::uniform_real_distribution<float> u(0.005f, 0.02f);
    for (uint16_t& s : sc) s = f32_to_f16(u(gen));
  };
  fill_weights(rng);
  std::vector<uint16_t> x(size_t(kMaxRows) * K);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  // Preserve the old first matrix and activations across modes and shapes. Extra layers
  // advance a separate generator, so resident/layers differ only in the weight walk.
  std::mt19937 extra_rng(rng);
  std::vector<DeviceWeights> bank;
  bank.reserve(copies);
  std::set<const void*> addresses;
  size_t allocated = 0;
  for (size_t layer = 0; layer < copies; ++layer) {
    if (layer) fill_weights(extra_rng);
    bank.emplace_back(ctx, w.size() * 4, sc.size() * 2);
    auto& dw = bank.back();
    upload(ctx, dw.words, w.data(), w.size() * 4);
    if (dw.scales) upload(ctx, *dw.scales, sc.data(), sc.size() * 2);
    addresses.insert(dw.words.ptr());
    allocated += dw.words.size() + (dw.scales ? dw.scales->size() : 0);
  }
  if (addresses.size() != copies || allocated != copies * weight_bytes(sh))
    throw std::runtime_error("distinct weight allocation accounting failed");
  std::printf("# %s: %zu distinct weight allocations, %zu bytes; all arms visit layers in ascending order\n",
              sh.tag, copies, allocated);
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  upload(ctx, dx, x.data(), x.size() * 2);
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(K) * kNs * 2);
  // fp32 [kMaxRows][N] holds either form (the SiLU x is a quarter of it)
  l0::Mem out_a(ctx, l0::MemKind::Device, out_bytes), out_b(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");
  auto deq = [&](uint32_t n0, size_t layer) {
    const auto& dw = bank[layer];
    cx.launch(dq, kNs / 16, K / 64, 1,
              {PtrArg(dw.words.ptr()), PtrArg(dw.scale_ptr()), PtrArg(slab.ptr()), arg_val(n0)});
  };
  // the production walk: gemm_l0 / gemm_l0_silu over each slab right after its dequant
  auto walk = [&](uint32_t M, bool silu, l0::Mem& o, size_t layer) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(n0, layer);
      if (silu) {
        const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, 0, 0, 0, 0};
        runtime::prefill::gemm_l0_silu(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(),
                                       o.as<uint16_t>() + n0 / 2, N / 2);
      } else {
        const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, N, 0, 0, 0};
        runtime::prefill::gemm_l0(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(), o.as<float>() + n0, false);
      }
    }
  };
  auto fused = [&](uint32_t M, const Q4& v, bool silu, l0::Mem& o, size_t layer) {
    const auto& dw = bank[layer];
    const uint32_t Mt = pad_to(M, v.rows());
    const uint32_t ldc = silu ? N / 2 : N;
    auto launch = [&](const std::string& variant) {
      l0::Kernel& g = kc(variant, "pf_gemm_q4");
      cx.launch(g, N / v.cols(), Mt / v.rows(), 1,   // columns first
                {PtrArg(dx.ptr()), PtrArg(dw.words.ptr()), PtrArg(dw.scale_ptr()), PtrArg(o.ptr()), arg_val(Mt), arg_val(K),
                 arg_val(N), arg_val(K), arg_val(ldc)});
    };
    if (v.slm) {
      launch(kernels::pf_gemm_q4_slm_variant(v.na, v.sn));
    } else if (v.split) {
      launch(kernels::pf_gemm_q4_split_variant(v.na, v.sn, false));
      launch(kernels::pf_gemm_q4_split_variant(v.na, v.sn, true));
    } else {
      launch(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, sh.layout, silu));
    }
  };
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

  // --- checks: every fused output == the slab walk's over the M real rows, bit for bit ---------
  std::vector<uint8_t> a(out_bytes), b(out_bytes);
  std::vector<bool> forms = {sh.silu};
  if (sh.silu) forms.push_back(false);   // gate||up plain too: the fp32 C under the epilogue
  for (uint32_t M : rows)
    for (bool silu : forms) {
      const auto compatible = [=](const Q4& v) { return (v.split || v.slm) ? silu : !silu || v.nb % 2 == 0; };
      if (std::none_of(q4s.begin(), q4s.end(), compatible)) continue;
      for (size_t layer = 0; layer < copies; ++layer) {
        const size_t row_bytes = silu ? size_t(N / 2) * 2 : size_t(N) * 4;
        walk(M, silu, out_a, layer);
        cx.wait();
        download(ctx, a.data(), out_a, size_t(M) * row_bytes);
        size_t compared = 0;
        for (const Q4& v : q4s) {
          if (!compatible(v)) continue;
          std::vector<uint8_t> poison(size_t(M) * row_bytes, 0xA5);
          upload(ctx, out_b, poison.data(), poison.size());
          fused(M, v, silu, out_b, negative && layer + 1 == copies ? 0 : layer);
          cx.wait();
          download(ctx, b.data(), out_b, size_t(M) * row_bytes);
          const size_t words = size_t(M) * row_bytes / 4;
          const size_t diff = std::memcmp(a.data(), b.data(), words * 4) == 0 ? 0 : [&] {
            size_t d = 0;
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
            for (size_t i = 0; i < words; ++i) d += pa[i] != pb[i];
            return d;
          }();
          ++checks;
          ++compared;
          if (diff) {
            std::printf("| %s | M %u %s: %s vs the slab walk, %zu words | **%zu differ** |\n", sh.tag, M,
                        silu ? "SiLU" : "plain", v.tag().c_str(), words, diff);
            const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
            const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
            for (size_t i = 0, shown = 0; i < words && shown < 4; ++i)
              if (pa[i] != pb[i]) {
                float fa, fb;
                std::memcpy(&fa, &pa[i], 4);
                std::memcpy(&fb, &pb[i], 4);
                std::printf("|  | word %zu: walk %08x (%g), fused %08x (%g) | |\n", i, pa[i], fa, pb[i], fb);
                ++shown;
              }
            throw std::runtime_error(std::string(sh.tag) + ": " + v.tag() + " layer " +
                                     std::to_string(layer) + " is not the slab walk");
          }
        }
        std::printf("| %s | layer %zu M %u %s: %zu tiles vs the slab walk | 0 words differ |\n", sh.tag, layer, M,
                    silu ? "SiLU" : "plain", compared);
      }
    }
  std::fflush(stdout);
  if (negative) throw std::runtime_error("negative check failed: wrong-layer output unexpectedly matched");

  // --- timing (the engine's forms only) --------------------------------------
  struct Arm {
    std::string name;
    uint32_t M;
    std::function<void()> run;
  };
  std::vector<Arm> arms;
  for (uint32_t M : rows) {
    arms.push_back({"l0_now", M, [&, M] { for (size_t l = 0; l < copies; ++l) walk(M, sh.silu, out_a, l); }});
    arms.push_back({"dq", M, [&] {
      for (size_t l = 0; l < copies; ++l)
        for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(n0, l);
    }});
    for (const Q4& v : q4s)
      if (!((v.split || v.slm) ? !sh.silu : sh.silu && v.nb % 2))
        arms.push_back({v.tag(), M, [&, M, v] {
          for (size_t l = 0; l < copies; ++l) fused(M, v, sh.silu, out_b, l);
        }});
  }
  for (Arm& arm : arms) arm.run();
  cx.wait();
  std::vector<std::vector<double>> t(arms.size()), wall(arms.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < arms.size(); ++j) {
      const size_t i = (j + size_t(r) * 7) % arms.size();
      const auto ms = timed(arms[i].run);
      t[i].push_back(ms.first);
      wall[i].push_back(ms.second);
      std::printf("SAMPLE {\"shape\":\"%s\",\"weight_mode\":\"%s\",\"round\":%d,\"arm\":\"%s\","
                  "\"rows\":%u,\"copies\":%zu,\"gpu_ms\":%.9f,\"wall_ms\":%.9f}\n",
                  sh.tag, layers ? "layers" : "resident", r, arms[i].name.c_str(), arms[i].M, copies,
                  ms.first, ms.second);
    }
  Table tab;
  for (size_t i = 0; i < arms.size(); ++i) {
    const double gpu = median(t[i]), host = median(wall[i]);
    tab[arms[i].name][arms[i].M] = gpu / double(copies);
    std::printf("WALK {\"shape\":\"%s\",\"weight_mode\":\"%s\",\"arm\":\"%s\",\"rows\":%u,"
                "\"copies\":%zu,\"gpu_ms\":%.9f,\"wall_ms\":%.9f,\"extrapolated\":%s}\n",
                sh.tag, layers ? "layers" : "resident", arms[i].name.c_str(), arms[i].M,
                copies, gpu, host, layers ? "false" : "true");
  }
  return tab;
}

std::vector<std::string> split(const std::string& s, const char* flag) {
  if (s.empty() || s.front() == ',' || s.back() == ',' || s.find(",,") != std::string::npos)
    throw std::runtime_error(std::string(flag) + ": expected a nonempty comma-separated list");
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(tok);
  return v;
}

uint32_t positive(const std::string& s, const char* flag, uint32_t maximum) {
  if (s.empty() || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }))
    throw std::runtime_error(std::string(flag) + ": expected a positive integer");
  try {
    const auto value = std::stoull(s);
    if (value >= 1 && value <= maximum) return uint32_t(value);
  } catch (const std::exception&) { }
  throw std::runtime_error(std::string(flag) + ": expected 1.." + std::to_string(maximum));
}

void print_table(const Table& tab, const std::vector<uint32_t>& rows, const char* fmt) {
  std::printf("| arm |");
  for (uint32_t M : rows) std::printf(" M %u |", M);
  std::printf("\n|---|");
  for (size_t i = 0; i < rows.size(); ++i) std::printf("---:|");
  std::printf("\n");
  for (const auto& [arm, by_m] : tab) {
    std::printf("| %s |", arm.c_str());
    for (uint32_t M : rows) {
      auto it = by_m.find(M);
      if (it == by_m.end()) std::printf(" - |");
      else std::printf(fmt, it->second);
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 7;
    std::string want = "all", rows_s = "1,16,32,64,128,256,512,1024", q4_s;
    std::string weights = "layers";
    bool plan_only = false, negative = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc)
        rounds = int(positive(argv[++i], "--rounds", uint32_t(std::numeric_limits<int>::max())));
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else if (a == "--q4" && i + 1 < argc) q4_s = argv[++i];
      else if (a == "--weights" && i + 1 < argc) weights = argv[++i];
      else if (a == "--plan") plan_only = true;
      else if (a == "--negative-check") negative = true;
      else {
        std::fprintf(stderr, "usage: probe_fused_small [--rounds 7] [--rows 1,...,1024] "
                             "[--shape all|gu,down,qkvz,out,faqkv] [--q4 q1b2s1,...] "
                             "[--weights layers|resident] [--plan] [--negative-check]\n");
        return 2;
      }
    }
    if (weights != "layers" && weights != "resident")
      throw std::runtime_error("--weights: expected layers or resident");
    const bool layers = weights == "layers";
    if (negative && !layers) throw std::runtime_error("--negative-check needs --weights layers");
    std::set<std::string> selected;
    if (want == "all") {
      for (const auto& sh : kShapes) selected.insert(sh.tag);
    } else {
      for (const auto& name : split(want, "--shape")) {
        const auto found = std::find_if(std::begin(kShapes), std::end(kShapes),
                                         [&](const Shape& sh) { return name == sh.tag; });
        if (found == std::end(kShapes) || !selected.insert(name).second)
          throw std::runtime_error("--shape: unknown or duplicate shape '" + name + "'");
      }
    }
    std::vector<uint32_t> rows;
    std::set<uint32_t> unique_rows;
    for (const std::string& r : split(rows_s, "--rows")) {
      const uint32_t m = positive(r, "--rows", kMaxRows);
      if (!unique_rows.insert(m).second) throw std::runtime_error("--rows: duplicate row count");
      rows.push_back(m);
    }
    std::vector<Q4> q4s;
    if (q4_s.empty()) {   // every tile the build may carry (-DB70_PF_Q4_SWEEP=ON builds them all)
      for (uint32_t na : {1u, 2u, 4u})
        for (uint32_t nb : {1u, 2u, 4u})
          for (uint32_t sn : {1u, 2u, 4u}) q4s.push_back({na, nb, sn});
      for (uint32_t nb : {1u, 2u})
        for (uint32_t sn : {1u, 2u}) q4s.push_back({8, nb, sn});
      q4s.push_back({8, 1, 4, false, true});
    } else {
      std::set<std::string> unique_tiles;
      for (const std::string& v : split(q4_s, "--q4")) {
        unsigned na = 0, nb = 0, sn = 0;
        int used = 0;
        if (std::sscanf(v.c_str(), "q%1ub%1us%1u%n", &na, &nb, &sn, &used) != 3 ||
            (v.substr(size_t(used)) != "" && v.substr(size_t(used)) != "split" &&
             v.substr(size_t(used)) != "slm") ||
            (na != 1 && na != 2 && na != 4 && na != 8) ||
            (nb != 1 && nb != 2 && nb != 4) || (sn != 1 && sn != 2 && sn != 4))
          throw std::runtime_error("--q4: '" + v + "' is not qXbYsZ[split|slm]");
        if (!unique_tiles.insert(v).second) throw std::runtime_error("--q4: duplicate tile");
        const bool is_split = v.substr(size_t(used)) == "split";
        const bool is_slm = v.substr(size_t(used)) == "slm";
        if (is_split && (nb != 1 || na != 8 || (sn != 1 && sn != 2 && sn != 4)))
          throw std::runtime_error("--q4: split tiles are q8b1s1split, q8b1s2split or q8b1s4split");
        if (is_slm && (nb != 1 || na != 8 || (sn != 2 && sn != 4)))
          throw std::runtime_error("--q4: SLM tiles are q8b1s2slm or q8b1s4slm");
        q4s.push_back({na, nb, sn, is_split, is_slm});
      }
    }
    if (plan_only) {
      for (const auto& sh : kShapes) if (selected.count(sh.tag)) print_plan(sh, layers);
      return 0;
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_fused_small (3i): synthetic weights at Qwen3.8's layer shapes; weights=%s\n",
                weights.c_str());
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n\n", mask ? mask : "(unset)",
                ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    std::vector<Q4> built;
    for (const Q4& v : q4s) {
      try {
        if (v.slm) {
          kc(kernels::pf_gemm_q4_slm_variant(v.na, v.sn), "pf_gemm_q4");
          built.push_back(v);
          continue;
        }
        if (v.split) {
          kc(kernels::pf_gemm_q4_split_variant(v.na, v.sn, false), "pf_gemm_q4");
          kc(kernels::pf_gemm_q4_split_variant(v.na, v.sn, true), "pf_gemm_q4");
          built.push_back(v);
          continue;
        }
        kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 0, false), "pf_gemm_q4");
        kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 1, false), "pf_gemm_q4");
        if (v.nb % 2 == 0) kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 0, true), "pf_gemm_q4");
        built.push_back(v);
      } catch (const std::exception&) {
        if (!q4_s.empty()) throw;
        std::printf("# %s: not built, skipped\n", v.tag().c_str());
      }
    }
    q4s = built;
    if (q4s.empty()) throw std::runtime_error("no built q4 candidates");
    std::mt19937 rng(5);
    std::printf("## checks (bitwise)\n\n| shape | check | result |\n|---|---|---|\n");
    Table sum;   // arm -> rows -> ms summed over shapes x layers
    bool all = true;
    size_t checks = 0;
    for (const Shape& sh : kShapes) {
      if (!selected.count(sh.tag)) {
        all = false;
        continue;
      }
      if (std::none_of(q4s.begin(), q4s.end(), [&](const Q4& v) {
            return (v.split || v.slm) ? sh.silu : !sh.silu || v.nb % 2 == 0;
          })) throw std::runtime_error(std::string(sh.tag) + ": no candidate for the timed form");
      print_plan(sh, layers);
      const Table tab = run_shape(ctx, cx, kc, sh, rows, q4s, rounds, rng, checks, layers, negative);
      std::printf("\n### %s (K %u, N %u, layout %u, %u layers): ms per linear "
                  "(whole GPU walk / %u), medians\n\n", sh.name, sh.K, sh.N, sh.layout, sh.layers,
                  layers ? sh.layers : 1);
      print_table(tab, rows, " %.6f |");
      for (const auto& [arm, by_m] : tab)
        for (const auto& [M, ms] : by_m) sum[arm][M] += sh.layers * ms;
      std::fflush(stdout);
    }
    std::printf("\n## bitwise: %zu comparisons, 0 words differ\n", checks);
    std::printf("\n## one chunk's linears%s (%s, ms)\n\n", all ? "" : " (the shapes run only)",
                layers ? "sum of measured complete-walk medians; no extrapolation"
                       : "ESTIMATED: one resident-matrix median x model layers");
    print_table(sum, rows, " %.1f |");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_fused_small FAILED: %s\n", e.what());
    return 2;
  }
}
