// probe_fused_small - open-work 3b: a short prefill chunk's int4 linears read straight from the int4
// weights (pf_gemm_q4.cl, no bf16 slab) against the slab walk. Qwen3.8's five int4 linear shapes at
// M rows, each through
//   l0_now  runtime/prefill/linear_l0.cc's walk as built: per 1024-column slab pf_dequant_slab, then
//           gemm_l0 (pf_small_tile's pf_gemm_small up to 1024 rows, pf_gemm above) - the control
//   dq      the walk's dequants alone (every slab)
//   qXbYsZ  pf_gemm_q4_A<X>_B<Y>_S<Z>_L<layout>: X 8-row blocks x Y 16-column atoms a sub-group, Z
//           sub-groups a work-group, ONE launch over the linear's whole N, rows padded to 8 X only
// gate||up runs its SiLU epilogue (bf16 x), as the engine does. Weights and activations are random
// (probe_small_chunk's; kernel TIMES do not depend on the values).
//
// The gate, before any time: every qXbYsZ output equals the slab walk's bit for bit over the M real
// rows (memcmp as 32-bit words) at every row count and shape - plain for all five, the SiLU epilogue
// for gate||up, and gate||up plain as well (the fp32 C the SiLU epilogue consumes). A difference
// throws: nothing is timed.
//
// Timing (probe_small_chunk's): each arm is the GPU span between two global timestamps around its
// launches on the in-order list; a warm-up pass, then `rounds` rounds with the arms in a rotating
// order (interleaved - sequential arms mis-state ratios by up to 35 %). Medians. The l0_now arm runs
// each slab's GEMM right after its dequant, as the engine does. The summary multiplies each shape
// by its layer count: one chunk's linears.
//
//   probe_fused_small [--rounds 7] [--rows 1,16,32,64,128,256,512,1024]
//                     [--shape all|gu,down,qkvz,out,faqkv] [--q4 q1b2s1,...]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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

struct Q4 {
  uint32_t na, nb, sn;
  std::string tag() const {
    return "q" + std::to_string(na) + "b" + std::to_string(nb) + "s" + std::to_string(sn);
  }
  uint32_t rows() const { return 8 * na; }
  uint32_t cols() const { return 16 * nb * sn; }
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
                std::mt19937& rng, size_t& checks) {
  const uint32_t K = sh.K, N = sh.N;
  // --- data (probe_small_chunk's) -------------------------------------------
  const size_t w_u32 = sh.layout == 0 ? size_t(K / 8) * N : size_t(N / 16) * (K / 64) * 136;
  std::vector<uint32_t> w(w_u32);
  for (uint32_t& v : w) v = rng();
  if (sh.layout == 1)
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
  std::vector<uint16_t> x(size_t(kMaxRows) * K);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  l0::Mem dw(ctx, l0::MemKind::Device, w.size() * 4);
  l0::Mem dsc(ctx, l0::MemKind::Device, std::max<size_t>(sc.size() * 2, 64));
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  upload(ctx, dw, w.data(), w.size() * 4);
  if (!sc.empty()) upload(ctx, dsc, sc.data(), sc.size() * 2);
  upload(ctx, dx, x.data(), x.size() * 2);
  l0::Mem slab(ctx, l0::MemKind::Device, size_t(K) * kNs * 2);
  // fp32 [kMaxRows][N] holds either form (the SiLU x is a quarter of it)
  const size_t out_bytes = size_t(kMaxRows) * N * 4;
  l0::Mem out_a(ctx, l0::MemKind::Device, out_bytes), out_b(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  const void* scales = sh.layout == 0 ? dsc.ptr() : nullptr;
  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");
  auto deq = [&](uint32_t n0) {
    cx.launch(dq, kNs / 16, K / 64, 1, {PtrArg(dw.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
  };
  // the production walk: gemm_l0 / gemm_l0_silu over each slab right after its dequant
  auto walk = [&](uint32_t M, bool silu, l0::Mem& o) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(n0);
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
  auto fused = [&](uint32_t M, const Q4& v, bool silu, l0::Mem& o) {
    const uint32_t Mt = pad_to(M, v.rows());
    l0::Kernel& g = kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, sh.layout, silu), "pf_gemm_q4");
    const uint32_t ldc = silu ? N / 2 : N;
    cx.launch(g, N / v.cols(), Mt / v.rows(), 1,   // columns first
              {PtrArg(dx.ptr()), PtrArg(dw.ptr()), PtrArg(scales), PtrArg(o.ptr()), arg_val(Mt), arg_val(K),
               arg_val(N), arg_val(K), arg_val(ldc)});
  };
  auto timed = [&](const std::function<void()>& launches) {
    cx.timestamp(stamp);
    launches();
    cx.timestamp(stamp + 1);
    cx.wait();
    return cx.timestamp_ms(stamp[0], stamp[1]);
  };

  // --- checks: every fused output == the slab walk's over the M real rows, bit for bit ---------
  std::vector<uint8_t> a(out_bytes), b(out_bytes);
  std::vector<bool> forms = {sh.silu};
  if (sh.silu) forms.push_back(false);   // gate||up plain too: the fp32 C under the epilogue
  for (uint32_t M : rows)
    for (bool silu : forms) {
      const size_t row_bytes = silu ? size_t(N / 2) * 2 : size_t(N) * 4;
      walk(M, silu, out_a);
      cx.wait();
      download(ctx, a.data(), out_a, size_t(M) * row_bytes);
      for (const Q4& v : q4s) {
        if (silu && v.nb % 2) continue;
        std::vector<uint8_t> poison(size_t(M) * row_bytes, 0xA5);
        upload(ctx, out_b, poison.data(), poison.size());
        fused(M, v, silu, out_b);
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
          throw std::runtime_error(std::string(sh.tag) + ": " + v.tag() + " is not the slab walk");
        }
      }
      std::printf("| %s | M %u %s: %zu tiles vs the slab walk | 0 words differ |\n", sh.tag, M,
                  silu ? "SiLU" : "plain", q4s.size());
    }
  std::fflush(stdout);

  // --- timing (the engine's forms only) --------------------------------------
  struct Arm {
    std::string name;
    uint32_t M;
    std::function<void()> run;
  };
  std::vector<Arm> arms;
  for (uint32_t M : rows) {
    arms.push_back({"l0_now", M, [&, M] { walk(M, sh.silu, out_a); }});
    arms.push_back({"dq", M, [&] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(n0); }});
    for (const Q4& v : q4s)
      if (!(sh.silu && v.nb % 2)) arms.push_back({v.tag(), M, [&, M, v] { fused(M, v, sh.silu, out_b); }});
  }
  for (Arm& arm : arms) arm.run();
  cx.wait();
  std::vector<std::vector<double>> t(arms.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < arms.size(); ++j) {
      const size_t i = (j + size_t(r) * 7) % arms.size();
      t[i].push_back(timed(arms[i].run));
    }
  Table tab;
  for (size_t i = 0; i < arms.size(); ++i) tab[arms[i].name][arms[i].M] = median(t[i]);
  return tab;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(tok);
  return v;
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
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else if (a == "--q4" && i + 1 < argc) q4_s = argv[++i];
      else {
        std::fprintf(stderr, "usage: probe_fused_small [--rounds 7] [--rows 1,...,1024] "
                             "[--shape all|gu,down,qkvz,out,faqkv] [--q4 q1b2s1,...]\n");
        return 2;
      }
    }
    if (rounds < 1) throw std::runtime_error("--rounds must be >= 1");
    std::vector<uint32_t> rows;
    for (const std::string& r : split(rows_s)) {
      const uint32_t m = uint32_t(std::atoi(r.c_str()));
      if (m < 1 || m > kMaxRows) throw std::runtime_error("--rows: each in 1.." + std::to_string(kMaxRows));
      rows.push_back(m);
    }
    std::vector<Q4> q4s;
    if (q4_s.empty()) {   // every tile the build may carry (-DB70_PF_Q4_SWEEP=ON builds them all)
      for (uint32_t na : {1u, 2u, 4u})
        for (uint32_t nb : {1u, 2u, 4u})
          for (uint32_t sn : {1u, 2u, 4u}) q4s.push_back({na, nb, sn});
      for (uint32_t nb : {1u, 2u})
        for (uint32_t sn : {1u, 2u}) q4s.push_back({8, nb, sn});
    } else {
      for (const std::string& v : split(q4_s)) {
        unsigned na = 0, nb = 0, sn = 0;
        if (std::sscanf(v.c_str(), "q%ub%us%u", &na, &nb, &sn) != 3)
          throw std::runtime_error("--q4: '" + v + "' is not qXbYsZ");
        q4s.push_back({na, nb, sn});
      }
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_fused_small (open-work 3b): Qwen3.8's prefill linears from the int4 weights\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n\n", mask ? mask : "(unset)",
                ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    std::vector<Q4> built;
    for (const Q4& v : q4s) {
      try {
        kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 0, false), "pf_gemm_q4");
        kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 1, false), "pf_gemm_q4");
        if (v.nb % 2 == 0) kc(kernels::pf_gemm_q4_variant(v.na, v.nb, v.sn, 0, true), "pf_gemm_q4");
        built.push_back(v);
      } catch (const std::exception&) {
        std::printf("# %s: not built, skipped\n", v.tag().c_str());
      }
    }
    q4s = built;
    std::mt19937 rng(5);
    std::printf("## checks (bitwise)\n\n| shape | check | result |\n|---|---|---|\n");
    Table sum;   // arm -> rows -> ms summed over shapes x layers
    bool all = true;
    size_t checks = 0;
    for (const Shape& sh : kShapes) {
      if (want != "all" && ("," + want + ",").find("," + std::string(sh.tag) + ",") == std::string::npos) {
        all = false;
        continue;
      }
      const Table tab = run_shape(ctx, cx, kc, sh, rows, q4s, rounds, rng, checks);
      std::printf("\n### %s (K %u, N %u, layout %u, %u layers): ms per linear, medians\n\n", sh.name, sh.K,
                  sh.N, sh.layout, sh.layers);
      print_table(tab, rows, " %.3f |");
      for (const auto& [arm, by_m] : tab)
        for (const auto& [M, ms] : by_m) sum[arm][M] += sh.layers * ms;
      std::fflush(stdout);
    }
    std::printf("\n## bitwise: %zu comparisons, 0 words differ\n", checks);
    std::printf("\n## one chunk's linears%s (each shape x its layers; sums of medians, ms)\n\n",
                all ? "" : " (the shapes run only)");
    print_table(sum, rows, " %.1f |");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_fused_small FAILED: %s\n", e.what());
    return 2;
  }
}
