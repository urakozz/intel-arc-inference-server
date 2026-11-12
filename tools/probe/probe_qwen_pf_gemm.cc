// probe_qwen_pf_gemm - Qwen3.8's bf16 prefill linears (the `l0` slab walk: per 1024-column slab
// pf_dequant_slab, then pf_gemm) at a full chunk, against candidate arms built from the same sources
// with one define each. Record: docs/probes/probe-qwen-pf-gemm-2026-11-13.md.
//
// The walk is probe_prefill_split's (3i's distinct-layer walk: one live allocation of independent
// synthetic weights per layer of each of the five int4 linear families, visited in ascending order).
// Arms per family and row count:
//   ctl        the production walk (pf_dequant_slab + gemm_l0 / gemm_l0_silu)
//   ctl_dq     its dequants alone          ctl_gemm   its GEMMs alone
//   bs_dq      pf_dequant_slab BLOCK_STORE=1 alone (2D block stores, the same values)
//   r1_gemm    pf_gemm RASTER=1 alone (the work-group order N-fastest, the same tiles)
//   new        bs_dq + r1_gemm as a walk
// --gemm picks the GEMM arms by tag (qpg_gemm_T0[_SILU]_<TAG> binaries; this build has r1 - round 2's
// p1 / p3 / p4 / nb arms were removed after the record); --no-dq drops bs_dq / new / ctl_dq.
// Checks before any timing, layer 0 at every row count: every slab of bs_dq equals ctl_dq's byte for
// byte, and the new walk's output (and every GEMM arm's, after the control's dequant) equals the
// control walk's byte for byte.
//
// Timing: the GPU span between two global timestamps around each arm's whole layer walk on the
// in-order list; one warm-up pass of every arm, then `rounds` rounds with the arm order rotating
// (interleaved). Medians. TFLOP/s = 2 M K N layers / span; % of the 183.45 TFLOP/s bf16 DPAS peak.
//
//   probe_qwen_pf_gemm [--rounds 5] [--rows 4096,2048] [--shape all|gu,down,qkvz,out,faqkv] [--gemm ...] [--no-dq]
#include <algorithm>
#include <cctype>
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
#include <vector>

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
constexpr uint32_t kMaxRows = 4096;
constexpr double kPeak = 183.45;   // TFLOP/s, bf16 DPAS (probe_dpas_rate, docs/probes/probe-prefill-chunk-split-2026-11-10.md)

struct Shape {
  const char* tag;
  uint32_t K, N, layout, layers;
  bool silu;
};
const Shape kShapes[] = {
    {"gu", 5120, 34816, 0, 64, true},   {"down", 17408, 5120, 0, 64, false}, {"qkvz", 5120, 16384, 1, 48, false},
    {"out", 6144, 5120, 0, 64, false},  {"faqkv", 5120, 14336, 0, 16, false},
};

struct Layer {
  l0::Mem words;
  std::unique_ptr<l0::Mem> scales;
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
std::string slab_name(const char* stem, const Shape& sh) {
  return std::string(stem) + "_K" + std::to_string(sh.K) + "_N" + std::to_string(sh.N) + "_L" +
         std::to_string(sh.layout);
}

using Table = std::map<std::string, std::map<uint32_t, double>>;   // arm -> rows -> median GPU ms

Table run_shape(l0::Context& ctx, runtime::prefill::Context& cx, runtime::prefill::KernelCache& kc, const Shape& sh,
                const std::vector<uint32_t>& rows, const std::vector<std::string>& garms, bool with_dq, int rounds,
                std::mt19937& rng) {
  const uint32_t K = sh.K, N = sh.N;
  const size_t w_u32 = sh.layout == 0 ? size_t(K / 8) * N : size_t(N / 16) * (K / 64) * 136;
  const size_t sc_n = sh.layout == 0 ? size_t(K / 64) * N : 0;
  std::vector<uint32_t> w(w_u32);
  std::vector<uint16_t> sc(sc_n);
  std::vector<std::unique_ptr<Layer>> bank;
  for (uint32_t l = 0; l < sh.layers; ++l) {
    for (uint32_t& v : w) v = rng();
    std::uniform_real_distribution<float> u(0.005f, 0.02f);
    if (sh.layout == 1)
      for (size_t t = 0; t < w_u32 / 136; ++t)
        for (uint32_t j = 0; j < 8; ++j)
          w[t * 136 + 128 + j] = uint32_t(f32_to_f16(u(rng))) | (uint32_t(f32_to_f16(u(rng))) << 16);
    for (uint16_t& s : sc) s = f32_to_f16(u(rng));
    bank.push_back(std::make_unique<Layer>(ctx, w.size() * 4, sc.size() * 2));
    copy(ctx, bank.back()->words.ptr(), w.data(), w.size() * 4);
    if (bank.back()->scales) copy(ctx, bank.back()->scales->ptr(), sc.data(), sc.size() * 2);
  }
  std::printf("# %s: K %u N %u layout %u, %u distinct layers\n", sh.tag, K, N, sh.layout, sh.layers);

  std::vector<uint16_t> x(size_t(kMaxRows) * K);
  {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint16_t& v : x) v = f32_to_bf16(nd(rng));
  }
  l0::Mem dx(ctx, l0::MemKind::Device, x.size() * 2);
  copy(ctx, dx.ptr(), x.data(), x.size() * 2);
  const size_t slab_bytes = size_t(K) * kNs * 2;
  l0::Mem slab(ctx, l0::MemKind::Device, slab_bytes), slab2(ctx, l0::MemKind::Device, slab_bytes);
  const size_t out_bytes = size_t(kMaxRows) * N * 4;
  l0::Mem out_a(ctx, l0::MemKind::Device, out_bytes), out_b(ctx, l0::MemKind::Device, out_bytes);
  l0::Mem ts(ctx, l0::MemKind::Host, 64);
  uint64_t* stamp = ts.as<uint64_t>();

  l0::Kernel& dq = kc(kernels::pf_dequant_slab_variant(K, N, sh.layout), "pf_dequant_slab");
  l0::Kernel& dq_bs = kc(slab_name("qpg_dqbs", sh), "pf_dequant_slab");
  std::map<std::string, l0::Kernel*> gk;
  for (const std::string& a : garms) {
    std::string up = a;
    for (char& c : up) c = char(std::toupper(c));
    gk[a] = &kc(std::string(sh.silu ? "qpg_gemm_T0_SILU_" : "qpg_gemm_T0_") + up, "pf_gemm");
  }

  auto deq = [&](l0::Kernel& k, uint32_t n0, uint32_t l, l0::Mem& s) {
    cx.launch(k, kNs / 16, K / 64, 1,
              {PtrArg(bank[l]->words.ptr()), PtrArg(bank[l]->scale_ptr()), PtrArg(s.ptr()), arg_val(n0)});
  };
  auto gemm_ctl = [&](uint32_t M, uint32_t n0, l0::Mem& o) {
    if (sh.silu) {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, 0, 0, 0, 0};
      runtime::prefill::gemm_l0_silu(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(), o.as<uint16_t>() + n0 / 2,
                                     N / 2);
    } else {
      const runtime::prefill::GemmBatch b{M, K, kNs, 1, K, kNs, N, 0, 0, 0};
      runtime::prefill::gemm_l0(cx, kc, b, dx.as<uint16_t>(), slab.as<uint16_t>(), o.as<float>() + n0, false);
    }
  };
  auto gemm_v = [&](l0::Kernel& g, uint32_t M, uint32_t n0, l0::Mem& o) {
    const uint32_t Mp = pad256(M);
    void* c = sh.silu ? static_cast<void*>(o.as<uint16_t>() + n0 / 2) : static_cast<void*>(o.as<float>() + n0);
    const uint32_t ldc = sh.silu ? N / 2 : N;
    const uint64_t z = 0;
    cx.launch(g, Mp / 256, kNs / 256, 1,
              {PtrArg(dx.ptr()), PtrArg(slab.ptr()), PtrArg(c), arg_val(Mp), arg_val(K), arg_val(kNs), arg_val(K),
               arg_val(kNs), arg_val(ldc), arg_val(z), arg_val(z), arg_val(z)});
  };
  auto walk_ctl = [&](uint32_t M, uint32_t l, l0::Mem& o) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(dq, n0, l, slab);
      gemm_ctl(M, n0, o);
    }
  };
  auto walk_new = [&](uint32_t M, uint32_t l, l0::Mem& o) {
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(dq_bs, n0, l, slab);
      gemm_v(*gk.at("r1"), M, n0, o);
    }
  };

  // --- checks: layer 0, bitwise ---------------------------------------------------------------------
  if (with_dq) {
    std::vector<uint8_t> a(slab_bytes), b(slab_bytes);
    size_t bad = 0;
    for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
      deq(dq, n0, 0, slab);
      deq(dq_bs, n0, 0, slab2);
      cx.wait();
      copy(ctx, a.data(), slab.ptr(), slab_bytes);
      copy(ctx, b.data(), slab2.ptr(), slab_bytes);
      for (size_t i = 0; i < slab_bytes; i += 2) bad += std::memcmp(&a[i], &b[i], 2) != 0;
    }
    std::printf("CHECK {\"shape\":\"%s\",\"what\":\"bs_dq slabs\",\"differ\":%zu,\"of\":%zu}\n", sh.tag, bad,
                size_t(N) * K);
    if (bad) throw std::runtime_error(std::string(sh.tag) + ": bs_dq is not bitwise the production dequant");
  }
  std::vector<std::pair<std::string, std::function<void(uint32_t, l0::Mem&)>>> checks;
  if (with_dq && gk.count("r1")) checks.push_back({"new walk", [&](uint32_t M, l0::Mem& o) { walk_new(M, 0, o); }});
  for (const auto& [a, k] : gk)
    checks.push_back({a + "_gemm walk", [&, k = k](uint32_t M, l0::Mem& o) {
                        for (uint32_t n0 = 0; n0 < N; n0 += kNs) {
                          deq(dq, n0, 0, slab);
                          gemm_v(*k, M, n0, o);
                        }
                      }});
  for (const auto& [what, run] : checks)
  for (uint32_t M : rows) {
    const size_t n = size_t(pad256(M)) * N * 4 / (sh.silu ? 4 : 1);   // silu: bf16 [M][N/2]
    copy(ctx, out_a.ptr(), std::vector<uint8_t>(n, 0x5A).data(), n);
    copy(ctx, out_b.ptr(), std::vector<uint8_t>(n, 0xA5).data(), n);
    walk_ctl(M, 0, out_a);
    run(M, out_b);
    cx.wait();
    const size_t live = size_t(M) * N * (sh.silu ? 1 : 4);   // rows < M only
    std::vector<uint8_t> a(live), b(live);
    copy(ctx, a.data(), out_a.ptr(), live);
    copy(ctx, b.data(), out_b.ptr(), live);
    const size_t elem = sh.silu ? 2 : 4;
    size_t bad = 0;
    for (size_t i = 0; i < live; i += elem) bad += std::memcmp(&a[i], &b[i], elem) != 0;
    std::printf("CHECK {\"shape\":\"%s\",\"what\":\"%s output\",\"rows\":%u,\"differ\":%zu,\"of\":%zu}\n",
                sh.tag, what.c_str(), M, bad, live / elem);
    if (bad) throw std::runtime_error(std::string(sh.tag) + ": " + what + " is not bitwise the control walk");
  }

  struct Arm {
    std::string name;
    uint32_t M;
    std::function<void()> run;
  };
  std::vector<Arm> arms;
  const uint32_t L = sh.layers;
  for (uint32_t M : rows) {
    arms.push_back({"ctl_gemm", M, [&, M] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) gemm_ctl(M, n0, out_a); }});
    for (const auto& [a, k] : gk)
      arms.push_back({a + "_gemm", M, [&, M, k = k] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) gemm_v(*k, M, n0, out_b); }});
    if (!with_dq) continue;
    arms.push_back({"ctl", M, [&, M] { for (uint32_t l = 0; l < L; ++l) walk_ctl(M, l, out_a); }});
    if (gk.count("r1")) arms.push_back({"new", M, [&, M] { for (uint32_t l = 0; l < L; ++l) walk_new(M, l, out_b); }});
  }
  if (with_dq) {
    arms.push_back({"ctl_dq", 0, [&] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(dq, n0, l, slab); }});
    arms.push_back({"bs_dq", 0, [&] { for (uint32_t l = 0; l < L; ++l) for (uint32_t n0 = 0; n0 < N; n0 += kNs) deq(dq_bs, n0, l, slab); }});
  }

  auto timed = [&](const std::function<void()>& launches) {
    cx.timestamp(stamp);
    launches();
    cx.timestamp(stamp + 1);
    cx.wait();
    const double gpu = cx.timestamp_ms(stamp[0], stamp[1]);
    if (!(gpu > 0 && std::isfinite(gpu))) throw std::runtime_error("invalid whole-walk timing");
    return gpu;
  };
  for (Arm& arm : arms) arm.run();   // warm-up
  cx.wait();
  std::vector<std::vector<double>> t(arms.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t j = 0; j < arms.size(); ++j) {
      const size_t i = (j + size_t(r) * 3) % arms.size();
      t[i].push_back(timed(arms[i].run));
      std::printf("SAMPLE {\"shape\":\"%s\",\"round\":%d,\"arm\":\"%s\",\"rows\":%u,\"gpu_ms\":%.4f}\n", sh.tag, r,
                  arms[i].name.c_str(), arms[i].M, t[i].back());
    }
  Table tab;
  for (size_t i = 0; i < arms.size(); ++i) {
    const double gpu = median(t[i]);
    tab[arms[i].name][arms[i].M] = gpu;
    const double tf = arms[i].M ? gflop(sh, arms[i].M) / gpu : 0.0;
    std::printf("WALK {\"shape\":\"%s\",\"arm\":\"%s\",\"rows\":%u,\"layers\":%u,\"gpu_ms\":%.4f,\"min_ms\":%.4f,"
                "\"max_ms\":%.4f,\"tflops\":%.2f,\"of_peak\":%.3f}\n",
                sh.tag, arms[i].name.c_str(), arms[i].M, L, gpu, *std::min_element(t[i].begin(), t[i].end()),
                *std::max_element(t[i].begin(), t[i].end()), tf, tf / kPeak);
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
    std::string want = "all", rows_s = "4096,2048", gemm_s = "r1";
    bool with_dq = true;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--shape" && i + 1 < argc) want = argv[++i];
      else if (a == "--rows" && i + 1 < argc) rows_s = argv[++i];
      else if (a == "--gemm" && i + 1 < argc) gemm_s = argv[++i];
      else if (a == "--no-dq") with_dq = false;
      else {
        std::fprintf(stderr, "usage: probe_qwen_pf_gemm [--rounds 5] [--rows 4096,2048] "
                             "[--shape all|gu,down,qkvz,out,faqkv] [--gemm r1] [--no-dq]\n");
        return 2;
      }
    }
    std::set<std::string> selected;
    for (const auto& s : split(want)) selected.insert(s);
    std::vector<uint32_t> rows;
    for (const auto& r : split(rows_s)) {
      const long m = std::atol(r.c_str());
      if (m < 1025 || m > long(kMaxRows)) throw std::runtime_error("--rows: 1025.." + std::to_string(kMaxRows));
      rows.push_back(uint32_t(m));
    }
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_qwen_pf_gemm: synthetic distinct-layer weights at Qwen3.8's shapes\n");
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n", mask ? mask : "(unset)", ctx.name().c_str(),
                rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    std::mt19937 rng(11);
    Table sum;
    for (const Shape& sh : kShapes) {
      if (want != "all" && !selected.count(sh.tag)) continue;
      const Table tab = run_shape(ctx, cx, kc, sh, rows, split(gemm_s), with_dq, rounds, rng);
      for (const auto& [arm, by_m] : tab)
        for (const auto& [M, ms] : by_m) sum[arm][M] += ms;
      std::fflush(stdout);
    }
    std::printf("\n## the selected families summed (medians, GPU ms)\n\n| arm | rows | ms |\n|---|---:|---:|\n");
    for (const auto& [arm, by_m] : sum)
      for (const auto& [M, ms] : by_m) std::printf("| %s | %u | %.1f |\n", arm.c_str(), M, ms);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_qwen_pf_gemm FAILED: %s\n", e.what());
    return 2;
  }
}
