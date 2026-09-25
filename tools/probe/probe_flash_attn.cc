// probe_flash_attn - spec 6 P1: the flash-attention tile probe.
//
// Plan: docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md.
// Record: docs/probe-flash-attn-2026-09-25.md.
//
//   probe_flash_attn <pos> <C> [--arms all|none|<name>[,<name>...]] [--qscale S]
//                    [--bar B] [--time]
//
// Inputs, in the production layouts (src/runtime/prefill/attn.cc):
//   Q      bf16 [C][24][256]  N(0,1) x qscale, in PrefillScratch::pf_q (rows past C zero)
//   K, V   bf16 [pos + C][4][256]  N(0,1), one layer's KV cache, max_len rows allocated
// The CONTROL is the production composed path, runtime::prefill::attn_chunk on the L0
// backend, writing pf_o fp32 [24][pad256(C)][256]. Every ARM (probe_flash_attn.cl, one
// binary per -D set) writes the same layout into a buffer of its own.
//
// Correctness, before any time is printed: per-(row, head) cosine and max abs error of
// every path against an fp64 CPU reference on sampled rows ({0, 1, 7, 8, 63, 64, C/2,
// C-2, C-1} plus 16 drawn from mt19937(pos + C)) for all 24 heads, and a finiteness scan
// of each arm's rows [C, pad256(C)). An arm under the bar is reported and not timed; the
// probe exits 1 if any arm run is under the bar.
//
// Timing (--time): L0 kernel timestamps summed over a path's launches (Context's
// profiler), a 20-iteration warm-up of every path, then 11 interleaved rounds, each
// timing the control and then every surviving arm; the headline is the median PAIRED
// ratio control / arm with its range (docs/probe-w4a8-2026-09-23.md §14.4).
//
// PROBE-ONLY: no production file is edited.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/bf16.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/buffers.h"
#include "runtime/prefill/attn.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"
#include "runtime/prefill_backend.h"

namespace {

using runtime::PrefillBackend;
using runtime::PrefillScratch;
using runtime::prefill::arg_val;
using runtime::prefill::pad256;
using runtime::prefill::PtrArg;

constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kQRow = kQH * kHD, kKvRow = kKvH * kHD;
constexpr double kPeakTflops = 183.45;   // bf16 DPAS peak (spec 6 §5 F2)

// The arms. Same list as tools/probe/CMakeLists.txt's PFA foreach; nothing checks the two
// against each other, a missing binary is reported by name and skipped.
struct Arm {
  uint32_t kt, rpw, hpw, qreg;
  std::string name() const {
    return "pfa_KT" + std::to_string(kt) + "_R" + std::to_string(rpw) + "_H" +
           std::to_string(hpw) + "_Q" + std::to_string(qreg);
  }
};
const Arm kArms[] = {{32, 16, 6, 1}, {32, 32, 6, 1}, {64, 16, 6, 1}, {32, 16, 3, 1},
                     {32, 16, 1, 1}, {32, 16, 6, 0}, {64, 16, 6, 0}, {32, 32, 3, 1}};

bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

void download(l0::Context& ctx, void* host, const void* dev, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    const size_t now = std::min(kChunk, bytes - off);
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(host) + off, static_cast<const char*>(dev) + off, now);
  }
}
void upload(l0::Context& ctx, void* dev, const void* host, size_t bytes) {
  constexpr size_t kChunk = 64ul << 20;
  for (size_t off = 0; off < bytes; off += kChunk) {
    const size_t now = std::min(kChunk, bytes - off);
    l0::CmdList c = l0::CmdList::immediate(ctx);
    c.copy(static_cast<char*>(dev) + off, static_cast<const char*>(host) + off, now);
  }
}

struct Check {
  double worst_cos = 2.0, max_abs = 0.0;
  uint32_t wh = 0, wm = 0;
  size_t nonfinite = 0;   // in the sampled rows
  size_t pad_nonfinite = 0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::fprintf(stderr,
                   "usage: probe_flash_attn <pos> <C> [--arms all|none|<name>[,...]] "
                   "[--qscale S] [--bar B] [--time]\n");
      return 2;
    }
    const uint32_t pos = uint32_t(std::stoul(argv[1]));
    const uint32_t C = uint32_t(std::stoul(argv[2]));
    std::string arms_arg = "all";
    double qscale = 1.0, bar = 0.99999;
    bool timing = false;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (++i >= argc) throw std::runtime_error(a + " needs a value");
        return argv[i];
      };
      if (a == "--arms") arms_arg = val();
      else if (a == "--qscale") qscale = std::stod(val());
      else if (a == "--bar") bar = std::stod(val());
      else if (a == "--time") timing = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (C == 0 || C > PrefillScratch::kC) throw std::runtime_error("C must be in (0, 2048]");
    const uint32_t depth = pos + C;
    const uint32_t rows = pad256(C);
    const uint32_t max_len = pad256(depth);

    std::vector<Arm> arms;
    if (arms_arg == "all") {
      arms.assign(std::begin(kArms), std::end(kArms));
    } else if (arms_arg != "none") {
      size_t st = 0;
      while (st <= arms_arg.size()) {
        const size_t e = std::min(arms_arg.find(',', st), arms_arg.size());
        const std::string nm = arms_arg.substr(st, e - st);
        bool found = false;
        for (const Arm& a : kArms)
          if (a.name() == nm) { arms.push_back(a); found = true; }
        if (!found) throw std::runtime_error("unknown arm " + nm);
        st = e + 1;
      }
    }

    std::printf("# probe_flash_attn: pos %u, C %u, depth %u, rows %u, max_len %u, qscale %g, "
                "bar %.6f\n", pos, C, depth, rows, max_len, qscale, bar);
    const char* aff = std::getenv("ZE_AFFINITY_MASK");
    std::printf("# ZE_AFFINITY_MASK=%s\n", aff ? aff : "(unset)");
    l0::Context ctx(0);
    std::printf("# L0 device: %s\n", ctx.name().c_str());

    // --- inputs -------------------------------------------------------------
    std::mt19937 rng(12345u + pos * 7u + C);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> q(size_t(rows) * kQRow, 0);
    for (size_t i = 0; i < size_t(C) * kQRow; ++i)
      q[i] = common::f32_to_bf16(nd(rng) * float(qscale));
    std::vector<uint16_t> kc(size_t(max_len) * kKvRow, 0), vc(size_t(max_len) * kKvRow, 0);
    for (size_t i = 0; i < size_t(depth) * kKvRow; ++i) kc[i] = common::f32_to_bf16(nd(rng));
    for (size_t i = 0; i < size_t(depth) * kKvRow; ++i) vc[i] = common::f32_to_bf16(nd(rng));

    PrefillScratch s(ctx, max_len);
    l0::Mem dk(ctx, l0::MemKind::Device, kc.size() * 2);
    l0::Mem dv(ctx, l0::MemKind::Device, vc.size() * 2);
    const size_t o_elems = size_t(kQH) * rows * kHD;
    l0::Mem dout(ctx, l0::MemKind::Device, o_elems * 4);
    upload(ctx, s.pf_q.ptr(), q.data(), q.size() * 2);
    upload(ctx, dk.ptr(), kc.data(), kc.size() * 2);
    upload(ctx, dv.ptr(), vc.data(), vc.size() * 2);

    runtime::prefill::Context cx(ctx);
    runtime::prefill::KernelCache kcache(ctx);
    // Since plan 6b attn_chunk runs pf_flash_attn by default on L0; the control here is,
    // and stays, the composed path.
    runtime::prefill::set_attn_mode_for_test(runtime::prefill::AttnMode::Composed);

    // --- the fp64 reference on sampled (row, head) pairs ----------------------
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
    {
      auto work = [&](size_t t0, size_t t1) {
        std::vector<double> sc(depth);
        for (size_t t = t0; t < t1; ++t) {
          const uint32_t m = srows[t / kQH], h = uint32_t(t % kQH), j = h / 6;
          double qd[kHD];
          for (uint32_t d = 0; d < kHD; ++d)
            qd[d] = common::bf16_to_f32(q[size_t(m) * kQRow + size_t(h) * kHD + d]);
          const uint32_t nv = pos + m + 1;
          double mx = -INFINITY;
          for (uint32_t n = 0; n < nv; ++n) {
            const float* kr = &kf[size_t(n) * kKvRow + size_t(j) * kHD];
            double a = 0;
            for (uint32_t d = 0; d < kHD; ++d) a += qd[d] * double(kr[d]);
            sc[n] = a / 16.0;
            mx = std::max(mx, sc[n]);
          }
          double sum = 0, o[kHD] = {};
          for (uint32_t n = 0; n < nv; ++n) {
            const double p = std::exp(sc[n] - mx);
            sum += p;
            const float* vr = &vf[size_t(n) * kKvRow + size_t(j) * kHD];
            for (uint32_t d = 0; d < kHD; ++d) o[d] += p * double(vr[d]);
          }
          for (uint32_t d = 0; d < kHD; ++d) ref[t * kHD + d] = o[d] / sum;
        }
      };
      const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
      std::vector<std::thread> th;
      for (unsigned i = 0; i < nt; ++i)
        th.emplace_back(work, npairs * i / nt, npairs * (i + 1) / nt);
      for (auto& t : th) t.join();
    }

    std::vector<float> out(o_elems);
    auto check = [&](bool pad_scan) {
      download(ctx, out.data(), dout.ptr(), o_elems * 4);
      Check c;
      for (size_t t = 0; t < npairs; ++t) {
        const uint32_t m = srows[t / kQH], h = uint32_t(t % kQH);
        const float* g = &out[(size_t(h) * rows + m) * kHD];
        double dot = 0, na = 0, nb = 0;
        for (uint32_t d = 0; d < kHD; ++d) {
          const double a = g[d], b = ref[t * kHD + d];
          if (!std::isfinite(a)) ++c.nonfinite;
          dot += a * b; na += a * a; nb += b * b;
          c.max_abs = std::max(c.max_abs, std::fabs(a - b));
        }
        const double cs = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
        const double csv = std::isfinite(cs) ? cs : -2.0;   // NaN counts as the worst
        if (csv < c.worst_cos) {
          c.worst_cos = csv;
          c.wh = h;
          c.wm = m;
        }
      }
      if (pad_scan)
        for (uint32_t h = 0; h < kQH; ++h)
          for (uint32_t m = C; m < rows; ++m)
            for (uint32_t d = 0; d < kHD; ++d)
              if (!std::isfinite(out[(size_t(h) * rows + m) * kHD + d])) ++c.pad_nonfinite;
      return c;
    };

    // --- the composed control ---------------------------------------------------
    auto run_control = [&]() {
      runtime::prefill::attn_chunk(cx, kcache, s, pos, C, s.pf_q.as<uint16_t>(),
                                   dk.as<uint16_t>(), dv.as<uint16_t>(), PrefillBackend::L0);
    };
    run_control();
    cx.wait();
    // pf_o is the control's output; copy into dout's host view through `out`.
    Check cc;
    {
      download(ctx, out.data(), s.pf_o.ptr(), o_elems * 4);
      upload(ctx, dout.ptr(), out.data(), o_elems * 4);
      cc = check(false);
    }
    std::printf("composed: worst cos %.9f at (h %u, m %u), max abs %.3e, non-finite %zu\n",
                cc.worst_cos, cc.wh, cc.wm, cc.max_abs, cc.nonfinite);
    std::fflush(stdout);

    // --- the arms -----------------------------------------------------------------
    struct Live {
      Arm a;
      l0::Kernel* k;
    };
    std::vector<Live> live;
    bool any_fail = false;
    for (const Arm& a : arms) {
      const std::string nm = a.name();
      if (!file_exists(kernels::path(nm))) {
        std::printf("%s: NO BINARY (%s)\n", nm.c_str(), kernels::path(nm).c_str());
        any_fail = true;
        continue;
      }
      l0::Kernel& k = kcache(nm, "pfa");
      const uint32_t gx = (C + a.rpw - 1) / a.rpw, gz = 6 / a.hpw;
      auto run = [&, gx, gz]() {
        cx.launch(k, gx, kKvH, gz,
                  {PtrArg(s.pf_q.ptr()), PtrArg(dk.ptr()), PtrArg(dv.ptr()), PtrArg(dout.ptr()),
                   arg_val(pos), arg_val(C), arg_val(rows)});
      };
      l0::CmdList::immediate(ctx).fill(dout.ptr(), 0u, o_elems * 4);
      run();
      cx.wait();
      const Check c = check(true);
      const bool ok = c.worst_cos >= bar && c.nonfinite == 0 && c.pad_nonfinite == 0;
      std::printf("%s: worst cos %.9f at (h %u, m %u), max abs %.3e, non-finite %zu, "
                  "pad rows non-finite %zu -- %s\n",
                  nm.c_str(), c.worst_cos, c.wh, c.wm, c.max_abs, c.nonfinite, c.pad_nonfinite,
                  ok ? "PASS" : "FAIL");
      std::fflush(stdout);
      if (!ok) { any_fail = true; continue; }
      live.push_back({a, &k});
    }

    // --- timing -------------------------------------------------------------------
    if (timing && !live.empty()) {
      cx.set_profiling(true);
      auto time_ms = [&](const std::function<void()>& f) {
        cx.take_launch_metrics();
        f();
        cx.wait();
        return cx.take_launch_metrics().gpu_ms;
      };
      std::vector<std::function<void()>> runs;
      for (const Live& L : live) {
        const uint32_t gx = (C + L.a.rpw - 1) / L.a.rpw, gz = 6 / L.a.hpw;
        l0::Kernel* k = L.k;
        runs.push_back([&, k, gx, gz]() {
          cx.launch(*k, gx, kKvH, gz,
                    {PtrArg(s.pf_q.ptr()), PtrArg(dk.ptr()), PtrArg(dv.ptr()), PtrArg(dout.ptr()),
                     arg_val(pos), arg_val(C), arg_val(rows)});
        });
      }
      for (int i = 0; i < 20; ++i) {
        time_ms(run_control);
        for (auto& r : runs) time_ms(r);
      }
      constexpr int kRounds = 11;
      std::vector<double> tc;
      std::vector<std::vector<double>> ta(runs.size()), ratio(runs.size());
      for (int r = 0; r < kRounds; ++r) {
        const double c = time_ms(run_control);
        tc.push_back(c);
        for (size_t i = 0; i < runs.size(); ++i) {
          const double t = time_ms(runs[i]);
          ta[i].push_back(t);
          ratio[i].push_back(c / t);
        }
      }
      auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      auto lo = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
      auto hi = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
      // causal FLOPs: 4 x 24 x 256 x sum_m (pos + m + 1)   (derived)
      double keys = 0;
      for (uint32_t m = 0; m < C; ++m) keys += double(pos) + m + 1;
      const double gflop = 4.0 * kQH * kHD * keys / 1e9;
      std::printf("\n## timing (measured; L0 kernel timestamps summed per path), pos %u C %u, "
                  "%d interleaved rounds after a 20-iteration warm-up; causal work %.1f GFLOP "
                  "(derived)\n\n", pos, C, kRounds, gflop);
      std::printf("| path | median ms | min .. max | paired ratio composed / arm (median) | "
                  "range | TFLOP/s (derived) | %% of 183.45 |\n|---|---:|---:|---:|---:|---:|---:|\n");
      std::printf("| composed attn_chunk (L0) | %.3f | %.3f .. %.3f | 1 | | %.1f | %.1f |\n",
                  med(tc), lo(tc), hi(tc), gflop / med(tc), 100.0 * gflop / med(tc) / kPeakTflops);
      for (size_t i = 0; i < runs.size(); ++i) {
        const double mt = med(ta[i]);
        std::printf("| %s | %.3f | %.3f .. %.3f | **%.3fx** | %.3fx .. %.3fx | %.1f | %.1f |\n",
                    live[i].a.name().c_str(), mt, lo(ta[i]), hi(ta[i]), med(ratio[i]),
                    lo(ratio[i]), hi(ratio[i]), gflop / mt, 100.0 * gflop / mt / kPeakTflops);
      }
      cx.set_profiling(false);
    }
    return any_fail ? 1 : 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "probe_flash_attn FAILED: %s\n", ex.what());
    return 2;
  }
}
