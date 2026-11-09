// probe_gdn_scan_occ - open-work 3m: is the GDN prefill scan (`pf_gdn_scan_dpas_split`, the walk's
// production entry) bound by how many of its work-groups the card holds at once? Record:
// docs/probes/probe-gdn-prefill-2026-11-10.md. The production kernel only - nothing here is a candidate.
//
// The scan's grid is (48 v-heads, 4 state-column chunks) = 192 work-groups of 256 lanes (16 SIMD16
// threads), each walking the chunk's 64-position sub-chunks in sequence with 40,448 B of SLM. One
// Xe-core holds 64 such threads (4 work-groups) by threads and fewer by SLM, so 192 work-groups may
// not all be resident: the launch then runs in waves, each as long as one work-group's whole
// sequential walk. This probe launches the same kernel over the first `h` heads only (grid (h, 4)),
// h = 6 .. 48, at C = 2048 on synthetic inputs of the model's shapes. Flat time while h grows means
// the work-groups run side by side (latency-bound per work-group); a step means a second wave.
//
// Sanity, before timing: the h = 6 launch's `gdn_o` rows and final state for heads 0..5 must equal
// the h = 48 launch's bitwise (same inputs, state reset from the same pristine copy), and every
// output finite - a launch that computed nothing, or computed a different thing, fails it.
//
// Timing: the GPU span between two global timestamps around one launch on the in-order list (the
// state reset is appended before the first timestamp), one warm-up of every arm, then `rounds`
// rounds in a rotating order. Medians.
//
// Open-work 3n: the production entry reads A2 as the packed hi / lo words `pf_gdn_A2_hilo` writes
// (runtime/prefill/gdn_a2_pack.h; packed here on the host from the same fp32 tiles), and the pre-3n
// entry is kept as `pf_gdn_scan_dpas_split_staged` (fp32 A2, split in its SLM). The probe times
// both: a second sanity line requires the two to agree bitwise over all 48 heads, and the arm
// "staged 48" (the pre-3n kernel, all 192 work-groups) runs in the same rotation. (Candidate 2,
// an 8-sub-group entry, was timed here at commit 2d320eca and dropped: the 3m record, section 8.)
//
//   probe_gdn_scan_occ [--rounds 7] [--rows 2048]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/prefill/pf_kernels.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gdn_a2_pack.h"
#include "runtime/prefill/kernels.h"

namespace {

using runtime::prefill::arg_val;
using runtime::prefill::PtrArg;

constexpr uint32_t kHeads = 48, kKHeads = 16, kDim = 128, kConv = 10240, kCt = 64;
constexpr uint32_t kColChunks = 4;   // gdn.cc's kStateColChunks
constexpr uint32_t kMaxRows = 4096;

uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int rounds = 7;
    uint32_t C = 2048;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--rounds" && i + 1 < argc) rounds = std::max(1, std::atoi(argv[++i]));
      else if (a == "--rows" && i + 1 < argc) C = uint32_t(std::atol(argv[++i]));
      else {
        std::fprintf(stderr, "usage: probe_gdn_scan_occ [--rounds 7] [--rows 2048]\n");
        return 2;
      }
    }
    if (C < 1 || C > kMaxRows) throw std::runtime_error("--rows: 1.." + std::to_string(kMaxRows));
    const uint32_t nch = (C + kCt - 1) / kCt;
    const char* mask = std::getenv("ZE_AFFINITY_MASK");
    l0::Context ctx(0);
    std::printf("# probe_gdn_scan_occ (3m): pf_gdn_scan_dpas_split over the first h heads, C = %u\n", C);
    std::printf("# ZE_AFFINITY_MASK=%s, device %s, %d rounds\n", mask ? mask : "(unset)", ctx.name().c_str(),
                rounds);
    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kc(ctx);
    l0::Kernel& scan = kc(kernels::pf_gdn_scan_variant(kKHeads, kHeads), "pf_gdn_scan_dpas_split");
    l0::Kernel& staged = kc(kernels::pf_gdn_scan_variant(kKHeads, kHeads), "pf_gdn_scan_dpas_split_staged");

    // Synthetic inputs at the model's shapes and magnitudes: q / k as l2-normalised rows, w / u and
    // A2 small, the cumulative gate a steady decay restarting every 64 positions, the state small.
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> xb(size_t(C) * kConv), w(size_t(C) * kHeads * kDim), u(w.size());
    for (auto& x : xb) x = f32_to_bf16(nd(rng) * 0.0884f);
    for (auto& x : w) x = f32_to_bf16(nd(rng) * 0.1f);
    for (auto& x : u) x = f32_to_bf16(nd(rng) * 0.1f);
    std::vector<float> a2(size_t(nch) * kHeads * kCt * kCt), g(size_t(C) * kHeads),
        st(size_t(kHeads) * kDim * kDim);
    for (auto& x : a2) x = nd(rng) * 0.05f;
    for (uint32_t m = 0; m < C; ++m)
      for (uint32_t h = 0; h < kHeads; ++h) g[size_t(m) * kHeads + h] = -0.02f * float(m % kCt + 1);
    for (auto& x : st) x = nd(rng) * 0.01f;
    std::vector<uint32_t> a2hl(size_t(nch) * kHeads * runtime::prefill::kA2HlWords);
    runtime::prefill::pack_a2_hilo(a2.data(), size_t(nch) * kHeads, a2hl.data());

    l0::Mem d_xb(ctx, l0::MemKind::Device, xb.size() * 2), d_w(ctx, l0::MemKind::Device, w.size() * 2),
        d_u(ctx, l0::MemKind::Device, u.size() * 2), d_a2(ctx, l0::MemKind::Device, a2.size() * 4),
        d_a2hl(ctx, l0::MemKind::Device, a2hl.size() * 4),
        d_g(ctx, l0::MemKind::Device, g.size() * 4), d_st0(ctx, l0::MemKind::Device, st.size() * 4),
        d_st(ctx, l0::MemKind::Device, st.size() * 4),
        d_o(ctx, l0::MemKind::Device, size_t(C) * kHeads * kDim * 4);
    cx.copy(d_xb.ptr(), xb.data(), xb.size() * 2);
    cx.copy(d_w.ptr(), w.data(), w.size() * 2);
    cx.copy(d_u.ptr(), u.data(), u.size() * 2);
    cx.copy(d_a2.ptr(), a2.data(), a2.size() * 4);
    cx.copy(d_a2hl.ptr(), a2hl.data(), a2hl.size() * 4);
    cx.copy(d_g.ptr(), g.data(), g.size() * 4);
    cx.copy(d_st0.ptr(), st.data(), st.size() * 4);
    cx.wait();

    l0::Mem d_stamp(ctx, l0::MemKind::Host, 64);
    uint64_t* stamp = d_stamp.as<uint64_t>();

    // h in 1..48: the production entry over the first h heads; kStaged: the pre-3n entry over all 48.
    constexpr uint32_t kStaged = 0;
    auto run = [&](uint32_t arm) {
      const bool pre3n = arm == kStaged;
      const uint32_t h = pre3n ? kHeads : arm;
      cx.copy(d_st.ptr(), d_st0.ptr(), st.size() * 4);
      cx.timestamp(stamp);
      cx.launch(pre3n ? staged : scan, h, kColChunks, 1,
                {PtrArg(d_xb.ptr()), PtrArg(d_w.ptr()), PtrArg(d_u.ptr()),
                 PtrArg(pre3n ? d_a2.ptr() : d_a2hl.ptr()), PtrArg(d_g.ptr()), PtrArg(d_st.ptr()),
                 PtrArg(d_o.ptr()), arg_val(C)});
      cx.timestamp(stamp + 1);
      cx.wait();
      const double ms = cx.timestamp_ms(stamp[0], stamp[1]);
      if (!(ms > 0 && std::isfinite(ms))) throw std::runtime_error("invalid timing");
      return ms;
    };

    // Sanity: heads 0..5 of an h = 6 launch against the h = 48 launch, bitwise.
    const uint32_t hs = 6;
    std::vector<float> o48(size_t(C) * kHeads * kDim), s48(st.size()), o6(o48.size()), s6(st.size());
    run(kHeads);
    cx.copy(o48.data(), d_o.ptr(), o48.size() * 4);
    cx.copy(s48.data(), d_st.ptr(), s48.size() * 4);
    cx.wait();
    run(hs);
    cx.copy(o6.data(), d_o.ptr(), o6.size() * 4);
    cx.copy(s6.data(), d_st.ptr(), s6.size() * 4);
    cx.wait();
    size_t diff = 0, nonfinite = 0, nonzero = 0;
    for (uint32_t m = 0; m < C; ++m)
      for (uint32_t h = 0; h < hs; ++h)
        for (uint32_t x = 0; x < kDim; ++x) {
          const size_t i = (size_t(m) * kHeads + h) * kDim + x;
          diff += std::memcmp(&o6[i], &o48[i], 4) != 0;
          nonfinite += !std::isfinite(o48[i]);
          nonzero += o48[i] != 0.0f;
        }
    for (size_t i = 0; i < size_t(hs) * kDim * kDim; ++i) diff += std::memcmp(&s6[i], &s48[i], 4) != 0;
    for (float v : s48) nonfinite += !std::isfinite(v);
    std::printf("SANITY heads 0..%u: h=%u vs h=48 differing words %zu, non-finite %zu, nonzero o %zu of %zu -> %s\n",
                hs - 1, hs, diff, nonfinite, nonzero, size_t(C) * hs * kDim,
                diff == 0 && nonfinite == 0 && nonzero > 0 ? "PASS" : "FAIL");
    if (!(diff == 0 && nonfinite == 0 && nonzero > 0)) return 1;

    // Sanity 2 (3n): the production entry over 48 heads against the pre-3n one, bitwise.
    {
      const uint32_t arm = kStaged;
      std::vector<float> oS(o48.size()), sS(st.size());
      run(arm);
      cx.copy(oS.data(), d_o.ptr(), oS.size() * 4);
      cx.copy(sS.data(), d_st.ptr(), sS.size() * 4);
      cx.wait();
      size_t d2 = 0;
      for (size_t i = 0; i < size_t(C) * kHeads * kDim; ++i) d2 += std::memcmp(&oS[i], &o48[i], 4) != 0;
      for (size_t i = 0; i < sS.size(); ++i) d2 += std::memcmp(&sS[i], &s48[i], 4) != 0;
      std::printf("SANITY 3n: pf_gdn_scan_dpas_split (packed A2) vs %s, 48 heads: "
                  "differing words %zu of %zu -> %s\n", "_staged (fp32 A2)", d2, size_t(C) * kHeads * kDim + sS.size(),
                  d2 == 0 ? "PASS" : "FAIL");
      if (d2 != 0) return 1;
    }

    const std::vector<uint32_t> arms = {6, 12, 18, 24, 32, 36, 48, kStaged};
    for (uint32_t h : arms) run(h);   // warm-up
    std::vector<std::vector<double>> t(arms.size());
    for (int r = 0; r < rounds; ++r)
      for (size_t j = 0; j < arms.size(); ++j) {
        const size_t i = (j + size_t(r) * 3) % arms.size();
        t[i].push_back(run(arms[i]));
      }
    std::printf("\n| heads | work-groups | GPU ms (median of %d) | ms per head | vs 48 heads |\n"
                "|---:|---:|---:|---:|---:|\n", rounds);
    std::vector<double> med(arms.size());
    for (size_t i = 0; i < arms.size(); ++i) med[i] = median(t[i]);
    const double m48 = med[6];
    for (size_t i = 0; i < arms.size(); ++i) {
      const uint32_t h = arms[i] == kStaged ? kHeads : arms[i];
      const char* entry = arms[i] == kStaged ? "staged" : "packed";
      std::printf("| %u (%s) | %u | %.4f | %.4f | %.3f |\n", h, entry, h * kColChunks, med[i], med[i] / h,
                  med[i] / m48);
      std::printf("ARM {\"entry\":\"%s\",\"heads\":%u,\"wgs\":%u,\"gpu_ms\":%.6f,\"min\":%.6f,\"max\":%.6f}\n",
                  entry, h, h * kColChunks, med[i],
                  *std::min_element(t[i].begin(), t[i].end()), *std::max_element(t[i].begin(), t[i].end()));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gdn_scan_occ FAILED: %s\n", e.what());
    return 2;
  }
}
