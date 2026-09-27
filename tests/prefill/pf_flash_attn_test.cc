// pf_flash_attn against an fp64 CPU reference (spec 6 K1, plan 6b Task 1).
//
// Ported from tools/probe/probe_flash_attn.cc (plan 6a): random Q (bf16 N(0,1) x qscale)
// and a random K/V cache (bf16 N(0,1)) in the production layouts, the kernel launched as
// attn_chunk launches it (grid (ceil(C / 8), 4, 1), WG 96; spec 6c), and the fp64 reference on
// the same sampled rows ({0, 1, 7, 8, 63, 64, C/2, C-2, C-1} plus 16 drawn from
// mt19937(pos + C)) x 24 heads. Bar: worst per-(row, head) cosine >= 0.99999, no
// non-finite value in the sampled rows, and rows [C, pad256(C)) finite. The max abs
// error is printed, not gated (docs/probe-flash-attn-2026-09-25.md §3.1).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "pf_harness.h"

namespace {

constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kQRow = kQH * kHD, kKvRow = kKvH * kHD;
constexpr double kBar = 0.99999;

uint32_t pad256(uint32_t x) { return (x + 255u) & ~255u; }

bool run_case(pf_harness::Dev& d, l0::Kernel& k, uint32_t pos, uint32_t C, double qscale) {
  const uint32_t depth = pos + C, rows = pad256(C), max_len = pad256(depth);

  std::mt19937 rng(12345u + pos * 7u + C);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> q(size_t(rows) * kQRow, 0);
  for (size_t i = 0; i < size_t(C) * kQRow; ++i)
    q[i] = common::f32_to_bf16(nd(rng) * float(qscale));
  std::vector<uint16_t> kc(size_t(max_len) * kKvRow, 0), vc(size_t(max_len) * kKvRow, 0);
  for (size_t i = 0; i < size_t(depth) * kKvRow; ++i) kc[i] = common::f32_to_bf16(nd(rng));
  for (size_t i = 0; i < size_t(depth) * kKvRow; ++i) vc[i] = common::f32_to_bf16(nd(rng));

  l0::Mem dq = pf_harness::upload(d.ctx, d.imm, q);
  l0::Mem dk = pf_harness::upload(d.ctx, d.imm, kc);
  l0::Mem dv = pf_harness::upload(d.ctx, d.imm, vc);
  const size_t o_elems = size_t(kQH) * rows * kHD;
  l0::Mem dout(d.ctx, l0::MemKind::Device, o_elems * 4);
  d.imm.fill(dout.ptr(), 0u, o_elems * 4);

  k.arg_ptr(0, dq.ptr());
  k.arg_ptr(1, dk.ptr());
  k.arg_ptr(2, dv.ptr());
  k.arg_ptr(3, dout.ptr());
  k.arg(4, pos);
  k.arg(5, C);
  k.arg(6, rows);
  // Grid (ceil(C / 8), 4, 1): RPW 8 rows, one work-group per kv head, HPW 6 (gz = 1).
  d.run(k, (C + 7u) / 8u, kKvH);
  std::vector<float> out(o_elems);
  pf_harness::download(d.imm, out, dout);

  // The fp64 reference on the sampled rows.
  std::vector<uint32_t> srows = {0, 1, 7, 8, 63, 64, C / 2, C - 2, C - 1};
  {
    std::mt19937 r2(pos + C);
    for (int i = 0; i < 16; ++i) srows.push_back(uint32_t(r2() % C));
  }
  for (uint32_t& m : srows) m = std::min(m, C - 1);
  if (C < 2) srows.assign(1, 0);
  std::sort(srows.begin(), srows.end());
  srows.erase(std::unique(srows.begin(), srows.end()), srows.end());
  std::vector<float> kf(size_t(depth) * kKvRow), vf(size_t(depth) * kKvRow);
  for (size_t i = 0; i < kf.size(); ++i) {
    kf[i] = common::bf16_to_f32(kc[i]);
    vf[i] = common::bf16_to_f32(vc[i]);
  }
  const size_t npairs = srows.size() * kQH;
  std::vector<double> ref(npairs * kHD);
  auto work = [&](size_t t0, size_t t1) {
    std::vector<double> sc(depth);
    for (size_t t = t0; t < t1; ++t) {
      const uint32_t m = srows[t / kQH], h = uint32_t(t % kQH), j = h / 6;
      double qd[kHD];
      for (uint32_t dd = 0; dd < kHD; ++dd)
        qd[dd] = common::bf16_to_f32(q[size_t(m) * kQRow + size_t(h) * kHD + dd]);
      const uint32_t nv = pos + m + 1;
      double mx = -INFINITY;
      for (uint32_t n = 0; n < nv; ++n) {
        const float* kr = &kf[size_t(n) * kKvRow + size_t(j) * kHD];
        double a = 0;
        for (uint32_t dd = 0; dd < kHD; ++dd) a += qd[dd] * double(kr[dd]);
        sc[n] = a / 16.0;
        mx = std::max(mx, sc[n]);
      }
      double sum = 0, o[kHD] = {};
      for (uint32_t n = 0; n < nv; ++n) {
        const double p = std::exp(sc[n] - mx);
        sum += p;
        const float* vr = &vf[size_t(n) * kKvRow + size_t(j) * kHD];
        for (uint32_t dd = 0; dd < kHD; ++dd) o[dd] += p * double(vr[dd]);
      }
      for (uint32_t dd = 0; dd < kHD; ++dd) ref[t * kHD + dd] = o[dd] / sum;
    }
  };
  const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; ++i) th.emplace_back(work, npairs * i / nt, npairs * (i + 1) / nt);
  for (auto& t : th) t.join();

  double worst = 2.0, max_abs = 0.0;
  uint32_t wh = 0, wm = 0;
  size_t nonfinite = 0, pad_nonfinite = 0;
  for (size_t t = 0; t < npairs; ++t) {
    const uint32_t m = srows[t / kQH], h = uint32_t(t % kQH);
    const float* g = &out[(size_t(h) * rows + m) * kHD];
    double dot = 0, na = 0, nb = 0;
    for (uint32_t dd = 0; dd < kHD; ++dd) {
      const double a = g[dd], b = ref[t * kHD + dd];
      if (!std::isfinite(a)) ++nonfinite;
      dot += a * b;
      na += a * a;
      nb += b * b;
      max_abs = std::max(max_abs, std::fabs(a - b));
    }
    const double cs = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    const double csv = std::isfinite(cs) ? cs : -2.0;   // NaN counts as the worst
    if (csv < worst) { worst = csv; wh = h; wm = m; }
  }
  for (uint32_t h = 0; h < kQH; ++h)
    for (uint32_t m = C; m < rows; ++m)
      for (uint32_t dd = 0; dd < kHD; ++dd)
        if (!std::isfinite(out[(size_t(h) * rows + m) * kHD + dd])) ++pad_nonfinite;
  const bool ok = worst >= kBar && nonfinite == 0 && pad_nonfinite == 0;
  std::printf("pf_flash_attn pos %u C %u qscale %g: worst cos %.9f at (h %u, m %u), max abs %.3e, "
              "non-finite %zu, pad rows [%u, %u) non-finite %zu -- %s\n",
              pos, C, qscale, worst, wh, wm, max_abs, nonfinite, C, rows, pad_nonfinite,
              ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}

}  // namespace

int main() {
  pf_harness::Dev d;
  l0::Module mod(d.ctx, kernels::path(kernels::pf_flash_attn_variant()));
  l0::Kernel k = mod.kernel("pf_flash_attn");
  k.group_size(16 * 6);    // reqd_work_group_size: 6 sub-groups of 16 (HPW 6 x RPW 8 / 8)
  bool ok = true;
  ok &= run_case(d, k, 16384, 2048, 1.0);
  ok &= run_case(d, k, 777, 300, 1.0);
  ok &= run_case(d, k, 0, 2048, 30.0);
  ok &= run_case(d, k, 0, 64, 1.0);
  // Spec 6c: the log2-domain max on a lone row (first tile m = -INF, one key), and C = 1
  // at depth -- the tail chunk of a 4097-id prompt -- where 7 of the 8 rows are padding.
  ok &= run_case(d, k, 0, 1, 1.0);
  ok &= run_case(d, k, 4096, 1, 1.0);
  ok &= run_case(d, k, 0, 9, 1.0);
  CHECK(ok);
  std::printf("pf_flash_attn_test: PASS\n");
  return 0;
}
