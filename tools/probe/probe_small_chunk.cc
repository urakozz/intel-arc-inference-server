// probe_small_chunk - open-work 3a: where a SHORT prefill chunk's linear time goes, and what a
// short-chunk GEMM buys. Qwen3.8's five int4 linear shapes at M rows, each through
//   l0     runtime/prefill/linear_l0.cc's walk (the default backend): per 1024-column slab
//          pf_dequant_slab_K<K>_N<N>_L<l> + pf_gemm (256 x 256 tiles, rows padded to 256: gemm_l0's
//          launch before open-work 3a)
//   dq     the walk's dequants alone (every slab), g256 the walk's GEMMs alone (every slab)
//   l0_now the walk through gemm_l0 as built (pf_small_tile picks the tile up to 1024 rows)
//   aXbYsZ the walk with pf_gemm_small_A<X>_B<Y>_S<Z> for pf_gemm (pf_gemm_small.cl: X 8-row
//          blocks x Y 16-column atoms a sub-group, Z sub-groups a work-group; rows padded to 8 X
//          only) - each slab's GEMM right after its dequant, as the engine runs it (a GEMM-only
//          arm over one resident slab reads B hot from cache and flatters the small tiles)
//   sm     gemm_sm (spec 28b's small-M decode GEMM, bf16 DPAS straight from the int4 weights - no
//          slab) at M = 8 and 16 where the build has the variant; its output is the S-slice form
//          a decode consumer folds and it rounds differently, so this arm is a time only
// Weights and activations are random as in probe_chunk_rows (kernel TIMES do not depend on the
// values). Before any time, every aXbYsZ walk's output equals l0's bit for bit over the M real
// rows (the same DPAS chain per accumulator; pf_gemm_small.cl's header).
//
// Timing: each arm is the GPU span between two global timestamps around its launches on the
// in-order list (launch gaps included, as the engine pays them); a warm-up pass, then `rounds`
// rounds with the arms in a rotating order (interleaved - sequential arms mis-state ratios by up
// to 35 %). Medians. The summary multiplies each shape by its layer count: one chunk's linears.
//
//   probe_small_chunk [--rounds 7] [--rows 16,32,64,128,256,512,1024,2048]
//                     [--shape all|gu,down,qkvz,out,faqkv] [--small a1b2s1,a2b2s1,a4b2s1,a4b4s4]
// (a trailing n, a4b2s4n: the _NF build, the column tiles on grid dimension 0)
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
using runtime::prefill::pad256;
using runtime::prefill::PtrArg;

constexpr uint32_t kNs = kernels::kPfSlabWidth;
constexpr uint32_t kMaxRows = 2048;

struct Shape {
  const char* tag;
  const char* name;
  uint32_t K, N, layout, layers, gemv_s;
  bool silu;
};
// Qwen3.8 (64 layers: 48 GDN + 16 full attention), probe_chunk_rows' rows; gemv_s the decode S
// that names the shape's gemm_sm row (kernels::kSmallMTuning).
const Shape kShapes[] = {
    {"gu", "gate||up (SiLU epilogue)", 5120, 34816, 0, 64, 8, true},
    {"down", "down", 17408, 5120, 0, 64, 4, false},
    {"qkvz", "GDN qkv||z", 5120, 16384, 1, 48, 1, false},
    {"out", "out_proj / o_proj", 6144, 5120, 0, 64, 4, false},
    {"faqkv", "FA q||k||v", 5120, 14336, 0, 16, 2, false},
};

struct Small {
  uint32_t na, nb, sn;
  bool nfast = false;   // the _NF build: grid dimension 0 walks the column tiles
  std::string tag() const {
    return "a" + std::to_string(na) + "b" + std::to_string(nb) + "s" + std::to_string(sn) + (nfast ? "n" : "");
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
                const Shape& sh, const std::vector<uint32_t>& rows, const std::vector<Small>& smalls,
                int rounds, std::mt19937& rng) {
  const uint32_t K = sh.K, N = sh.N;
  // --- data (probe_chunk_rows') ---------------------------------------------
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
  const size_t out_bytes = sh.silu ? size_t(kMaxRows) * (N / 2) * 2 : size_t(kMaxRows) * N * 4;
  l0::Mem out_a(ctx, l0::MemKind::Device, out_bytes), out_b(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem out_sm(ctx, l0::MemKind::Device, size_t(8) * 16 * N * 4);   // gemm_sm: [S][M][N] fp32
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  const void* scales = sh.layout == 0 ? dsc.ptr() : nullptr;
  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");
  const uint32_t ldc = sh.silu ? N / 2 : N;
  auto c_at = [&](l0::Mem& o, uint32_t n0) -> void* {
    return sh.silu ? static_cast<void*>(o.as<uint16_t>() + n0 / 2) : static_cast<void*>(o.as<float>() + n0);
  };
  auto deq = [&](uint32_t n0) {
    cx.launch(dq, kNs / 16, K / 64, 1, {PtrArg(dw.ptr()), PtrArg(scales), PtrArg(slab.ptr()), arg_val(n0)});
  };
  // pf_gemm itself (gemm_l0 now hands M <= kPfSmallMaxRows to pf_gemm_small): gemm_l0's launch
  l0::Kernel& big = kc(sh.silu ? kernels::pf_gemm_silu_variant() : kernels::pf_gemm_variant(false), "pf_gemm");
  auto g256 = [&](uint32_t M, uint32_t n0, l0::Mem& o) {
    const uint32_t Mp = pad256(M), lda = K, ldb = kNs;
    const uint64_t z = 0;
    cx.launch(big, Mp / 256, kNs / 256, 1,
              {PtrArg(dx.ptr()), PtrArg(slab.ptr()), PtrArg(c_at(o, n0)), arg_val(Mp), arg_val(K),
               arg_val(kNs), arg_val(lda), arg_val(ldb), arg_val(ldc), arg_val(z), arg_val(z), arg_val(z)});
  };
  auto gnow = [&](uint32_t M, uint32_t n0, l0::Mem& o) {
    if (sh.silu) {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, 0, 0, 0, 0};
      runtime::prefill::gemm_l0_silu(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(),
                                     static_cast<uint16_t*>(c_at(o, n0)), N / 2);
    } else {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, N, 0, 0, 0};
      runtime::prefill::gemm_l0(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(),
                                static_cast<float*>(c_at(o, n0)), false);
    }
  };
  auto gs = [&](uint32_t M, const Small& v, uint32_t n0, l0::Mem& o) {
    const uint32_t Mt = pad_to(M, v.rows());
    l0::Kernel& g = kc(kernels::pf_gemm_small_variant(v.na, v.nb, v.sn, sh.silu, v.nfast), "pf_gemm_small");
    const uint32_t gm = Mt / v.rows(), gn = kNs / v.cols();
    cx.launch(g, v.nfast ? gn : gm, v.nfast ? gm : gn, 1,
              {PtrArg(dx.ptr()), PtrArg(slab.ptr()), PtrArg(c_at(o, n0)), arg_val(Mt), arg_val(K),
               arg_val(kNs), arg_val(K), arg_val(kNs), arg_val(ldc)});
  };
  auto timed = [&](const std::function<void()>& launches) {
    cx.timestamp(stamp);
    launches();
    cx.timestamp(stamp + 1);
    cx.wait();
    return cx.timestamp_ms(stamp[0], stamp[1]);
  };

  struct Arm {
    std::string name;
    uint32_t M;
    std::function<void()> run;
  };
  std::vector<Arm> arms;
  for (uint32_t M : rows) {
    arms.push_back({"l0", M, [&, M] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) { deq(n0); g256(M, n0, out_a); } }});
    arms.push_back({"dq", M, [&] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(n0); }});
    // the production walk as it is now: gemm_l0 picks the tile (pf_small_tile)
    arms.push_back({"l0_now", M, [&, M] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) { deq(n0); gnow(M, n0, out_b); } }});
    arms.push_back({"g256", M, [&, M] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) g256(M, n0, out_a); }});
    for (const Small& v : smalls)
      arms.push_back({v.tag(), M, [&, M, v] { for (uint32_t n0 = 0; n0 < N; n0 += kNs) { deq(n0); gs(M, v, n0, out_b); } }});
    // gemm_sm where the build has it (Qwen3.8's rows are built at M 2..8 and 16)
    const kernels::SmallMTuning* t = kernels::small_m_tuning(K, N, sh.gemv_s, sh.layout);
    if (t && (M == 8 || M == 16)) {
      const std::string vn = kernels::gemm_sm_variant(M, *t);
      l0::Kernel* g = nullptr;
      try { g = &kc(vn, kernels::gemm_sm_entry(*t)); } catch (const std::exception&) { g = nullptr; }
      if (g)
        arms.push_back({"sm", M, [&, g, t] {
                          cx.launch(*g, N / kernels::gemm_sm_cols(*t), kernels::gemm_sm_grid_y(*t), 1,
                                    {PtrArg(dw.ptr()), PtrArg(scales), PtrArg(dx.ptr()), PtrArg(out_sm.ptr())});
                        }});
      else
        std::printf("| %s | gemm_sm M %u: %s not built | - |\n", sh.tag, M, vn.c_str());
    }
  }

  // --- checks: every small walk == l0 over the M real rows, bit for bit -----
  const size_t row_bytes = sh.silu ? size_t(N / 2) * 2 : size_t(N) * 4;
  std::vector<uint8_t> a(size_t(kMaxRows) * row_bytes), b(a.size());
  for (uint32_t M : rows) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) { deq(n0); g256(M, n0, out_a); }
    cx.wait();
    download(ctx, a.data(), out_a, size_t(M) * row_bytes);
    for (const Small& v : smalls) {
      std::vector<uint8_t> poison(size_t(M) * row_bytes, 0xA5);
      upload(ctx, out_b, poison.data(), poison.size());
      for (uint32_t n0 = 0; n0 < N; n0 += kNs) { deq(n0); gs(M, v, n0, out_b); }
      cx.wait();
      download(ctx, b.data(), out_b, size_t(M) * row_bytes);
      size_t diff = 0;
      const uint32_t* pa = reinterpret_cast<const uint32_t*>(a.data());
      const uint32_t* pb = reinterpret_cast<const uint32_t*>(b.data());
      for (size_t i = 0; i < size_t(M) * row_bytes / 4; ++i) diff += pa[i] != pb[i];
      std::printf("| %s | M %u: %s vs l0, %zu words | **%zu differ** |\n", sh.tag, M, v.tag().c_str(),
                  size_t(M) * row_bytes / 4, diff);
      if (diff) throw std::runtime_error(std::string(sh.tag) + ": " + v.tag() + " is not pf_gemm");
    }
  }
  std::fflush(stdout);

  // --- timing ---------------------------------------------------------------
  for (Arm& arm : arms) arm.run();
  cx.wait();
  std::vector<std::vector<double>> t(arms.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < arms.size(); ++j) {
      const size_t i = (j + size_t(r)) % arms.size();
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
    std::string want = "all", rows_s = "16,32,64,128,256,512,1024,2048";
    // the four tiles the build carries (pf_small_tile's); the 2026-11-08 sweep built NA 1/2/4 x
    // NB 2/4 x SN 1/2/4 - a tile the build lacks is skipped with a line
    std::string small_s = "a1b2s1,a2b2s1,a4b2s1,a4b4s4";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else if (a == "--small" && i + 1 < argc) small_s = argv[++i];
      else {
        std::fprintf(stderr, "usage: probe_small_chunk [--rounds 7] [--rows 16,...,2048] "
                             "[--shape all|gu,down,qkvz,out,faqkv] [--small a2b4s1,...]\n");
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
    std::vector<Small> smalls;
    for (const std::string& v : split(small_s)) {
      unsigned na = 0, nb = 0, sn = 0;
      if (std::sscanf(v.c_str(), "a%ub%us%u", &na, &nb, &sn) != 3)
        throw std::runtime_error("--small: '" + v + "' is not aXbYsZ[n]");
      smalls.push_back({na, nb, sn, v.back() == 'n'});
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_small_chunk (open-work 3a): Qwen3.8's prefill linears at short chunks\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n\n", mask ? mask : "(unset)",
                ctx.name().c_str(), rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    std::vector<Small> built;
    for (const Small& v : smalls) {
      try {
        kc(kernels::pf_gemm_small_variant(v.na, v.nb, v.sn, false, v.nfast), "pf_gemm_small");
        kc(kernels::pf_gemm_small_variant(v.na, v.nb, v.sn, true, v.nfast), "pf_gemm_small");
        built.push_back(v);
      } catch (const std::exception&) {
        std::printf("# %s: not built, skipped\n", v.tag().c_str());
      }
    }
    smalls = built;
    std::mt19937 rng(5);
    std::printf("## checks\n\n| shape | check | result |\n|---|---|---|\n");
    Table sum;   // arm -> rows -> ms summed over shapes x layers
    bool all = true;
    for (const Shape& sh : kShapes) {
      if (want != "all" && ("," + want + ",").find("," + std::string(sh.tag) + ",") == std::string::npos) {
        all = false;
        continue;
      }
      const Table tab = run_shape(ctx, cx, kc, sh, rows, smalls, rounds, rng);
      std::printf("\n### %s (K %u, N %u, %u layers): ms per linear, medians\n\n", sh.name, sh.K, sh.N,
                  sh.layers);
      print_table(tab, rows, " %.3f |");
      for (const auto& [arm, by_m] : tab)
        for (const auto& [M, ms] : by_m) sum[arm][M] += sh.layers * ms;
      std::fflush(stdout);
    }
    std::printf("\n## one chunk's linears%s (each shape x its layers; sums of medians, ms)\n\n",
                all ? "" : " (the shapes run only)");
    print_table(sum, rows, " %.1f |");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_small_chunk FAILED: %s\n", e.what());
    return 2;
  }
}
