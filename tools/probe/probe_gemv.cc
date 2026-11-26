// probe_gemv: the production GEMV at every production shape, both layouts,
// S in {1,2,4,8,16}; GB/s of weight bytes (int4 nibbles + f16 scales), and a
// correctness check against the CPU reference at every configuration.
// Decision rules (spec 1 §4.2): layout = higher GB/s summed over the five
// int4 shapes; S per shape = smallest S within 3% of that shape's best.
//
// The weights are random, not a uniform fill: the B70 compresses device-local
// memory losslessly, so a repeated word reads back far above the 608 GB/s
// theoretical peak (doc 01). Model weights are incompressible; so is this.
//
// `probe_gemv --loads` runs a SECOND, separate thing in the same binary: spec
// 1.7 §3's P1 load-path battery (`run_loads` below). With no argument this
// program is byte-for-byte the instrument that produced
// the earlier record and its output format is unchanged, because
// that output is a committed record and the battery must not perturb it.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "common/int4.h"
#include "common/repack.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

namespace {
const double kPeak = 600.0;     // GB/s, doc 01's roofline denominator
const double kMeasured = 590.0; // GB/s, what probe_bw reads through this launch path

// --- the P1 load-path battery ----------------------------------------------
// One row per (shape, variant). The variants are compiled from
// tools/probe/probe_gemv_loads.cl, which states the message-count model each
// one moves; `msgs` below is that model's number and is printed beside the
// measurement so a reader can check the arithmetic against the result.
//
// `base` is gemv.cl verbatim (every PGL_* knob at 0) and every other row is
// held BYTE-IDENTICAL to it: all of these variants keep the accumulation order
// exactly, so a differing bit is a bug in the variant and not a tolerance
// question. That is this battery's TDD equivalent, together with the
// pre-registered per-shape predictions in the task report.
struct LoadVariant {
  const char* tag;
  uint32_t L;         // layout
  bool s_l0best;      // false: the S the engine binds. true: the S layout 0 picks for itself
  const char* msgs;   // LSC messages per subgroup per k-group (weights + scales + activations)
  const char* what;
};
const LoadVariant kVariants[] = {
    {"base",     1, false, "10",     "gemv.cl verbatim - the control"},
    {"xwide",    1, false, "6",      "activations 8x16 B -> 4x32 B"},
    {"deq",      1, false, "10",     "xor+shift dequant (ALU only)"},
    {"pf1",      1, false, "10+1pf", "prefetch() 1 k-group ahead"},
    {"pf2",      1, false, "10+1pf", "prefetch() 2 ahead (bestla's distance)"},
    {"pf4",      1, false, "10+1pf", "prefetch() 4 ahead"},
    {"pfbuf",    1, false, "10",     "register double-buffer (+8 u32/lane)"},
    {"ballast",  1, false, "10",     "pfbuf's register-pressure control"},
    {"l0base",   0, false, "17",     "GPTQ-native, 8 strided 64 B loads"},
    {"l0b2d",    0, false, "10",     "+ 2D block read 8r16 (one message)"},
    {"l0b2dx",   0, false, "6",      "+ 2D block read and wide activations"},
    {"l0b2d16",  0, false, "9.5",    "+ 2D block read 16r16 (two k-groups)"},
    {"l0deq",    0, false, "17",     "GPTQ-native + xor/shift dequant"},
    {"base",     1, true,  "10",     "layout 1 at layout 0's best S - the S-ONLY option"},
    {"deq",      1, true,  "10",     "layout 1 at that S, + xor/shift dequant"},
    {"l0base",   0, true,  "17",     "GPTQ-native at layout 0's own best S"},
    {"l0b2d",    0, true,  "10",     "+ 2D block read 8r16, same S"},
    {"l0deq",    0, true,  "17",     "+ xor/shift dequant - THE COMPOSED CELL"},
    {"l0b2ddeq", 0, true,  "10",     "+ 2D block read and xor/shift dequant"},
};
struct LoadShape {
  uint32_t K, N, S_prod, S_l0;
  bool blk2d16;
  const char* name;
};
// The six shapes the engine binds, at the S it binds them at (model/qwen35.cc),
// and the S layout 0 picks for itself.
const LoadShape kShapes[] = {
    {6144, 5120, 16, 4, true, "out/o_proj"},   {5120, 14336, 1, 2, true, "q‖k‖v"},
    {5120, 16384, 1, 16, true, "qkv‖z"},       {5120, 34816, 4, 8, true, "gate‖up"},
    {17408, 5120, 16, 4, false, "down"},       {5120, 248320, 1, 1, true, "lm_head int4"},
};

std::string pgl_name(const char* tag, uint32_t K, uint32_t N, uint32_t S, uint32_t L) {
  return "pgl_" + std::string(tag) + "_M1_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_S" + std::to_string(S) + "_L" + std::to_string(L);
}

int run_loads(l0::Context& ctx, l0::Queue& q, l0::Fence& f) {
  std::puts("# probe_gemv --loads - the P1 load-path battery (spec 1.7 §3)\n");
  std::puts("Same arithmetic, same bytes, different message counts. `msgs` is the LSC");
  std::puts("messages one subgroup issues per k-group (64 K elements, 544 weight bytes):");
  std::puts("weights + scales + activations. Rows at the PRODUCTION S are held byte-identical");
  std::puts("to `base` (src/kernels/gemv.cl verbatim); rows at any other S fold a different");
  std::puts("number of fp32 partials and so reorder the sum, and are held to the CPU");
  std::puts("reference's tolerance instead - the `bytes vs base` column says which.");
  std::puts("Timing is the house 8-replay/drop-3");
  std::puts("median over 40 launches cycling NB weight copies past the 24 MB L2.\n");
  bool all_ok = true;
  std::vector<double> warm_first;   // the discarded warm-up config, kept as ramp evidence
  std::vector<double> base_first;  // GB/s of the `base` row, per shape, for the drift control
  for (const LoadShape& sh : kShapes) {
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    std::printf("\n## %s - %u×%u, %.2f MB of weights\n\n", sh.name, sh.K, sh.N,
                double(w.bytes()) / (1u << 20));
    // WARM-UP, discarded. The 8-replay/drop-3 inside `time_list` covers the
    // ramp WITHIN a configuration; it does not cover the device ramping down
    // while the host computes this shape's CPU reference, which takes seconds
    // of GPU idle. Measured 2026-08-26: without this the first recorded
    // configuration of a shape read 355 / 475 / 524 GB/s where the same binary
    // read 534 / 541 / 538 once the device was warm - +50.4% / +13.8% / +2.7%.
    // That is a clock ramp, not a variant, and it would have been attributed to
    // whichever variant happened to be first.
    const std::string warm = pgl_name("base", sh.K, sh.N, sh.S_prod, 1);
    const double warm_gbps =
        double(w.bytes()) /
        (run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, warm.c_str()}, 40).us_per_launch * 1e3);
    std::printf("| variant | L | S | msgs | µs | GB/s | %% of 590 | Δ%% vs base | bytes vs base | what |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
    std::vector<float> base_out;
    double base_gbps = 0;
    warm_first.push_back(warm_gbps);
    for (const LoadVariant& v : kVariants) {
      if (v.s_l0best && sh.S_l0 == sh.S_prod) continue;      // no second S to run
      if (!std::strcmp(v.tag, "l0b2d16") && !sh.blk2d16) continue;  // odd k-groups per slice
      const uint32_t S = v.s_l0best ? sh.S_l0 : sh.S_prod;
      const std::string name = pgl_name(v.tag, sh.K, sh.N, S, v.L);
      GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, S, v.L, name.c_str()}, 40);
      const double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
      const double err = max_abs_err(r.out, ref);
      const char* bytes = "-";
      if (base_out.empty()) {
        base_out = r.out;
        base_gbps = gbps;
        base_first.push_back(gbps);
        bytes = "(the control)";
        if (err > tol) { all_ok = false; bytes = "**WRONG vs CPU ref**"; }
      } else if (S != sh.S_prod) {
        // A different split-K width reorders the sum (S partials folded by the
        // host here, as `prep` does in situ), so this row is held to the
        // reference tolerance, not to bit-identity.
        if (err > tol) { all_ok = false; bytes = "**WRONG vs CPU ref**"; }
        else bytes = "ref ok (S reorders the sum)";
      } else {
        size_t diff = 0;
        for (size_t i = 0; i < base_out.size(); ++i)
          if (std::memcmp(&base_out[i], &r.out[i], sizeof(float)) != 0) ++diff;
        if (diff == 0) {
          bytes = "identical";
        } else {
          all_ok = false;
          bytes = "**DIFFER**";
          std::printf("<!-- %s: %zu of %zu cells differ, max abs err vs ref %.3g (tol %.3g) -->\n",
                      name.c_str(), diff, base_out.size(), err, tol);
        }
      }
      std::printf("| %s | %u | %u | %s | %.1f | %.0f | %.0f%% | %+.1f%% | %s | %s |\n", v.tag, v.L,
                  S, v.msgs, r.us_per_launch, gbps, 100.0 * gbps / kMeasured,
                  100.0 * (gbps / base_gbps - 1.0), bytes, v.what);
      std::fflush(stdout);
    }
  }
  // Drift control (the Task-A lesson): re-measure every `base` row at the end
  // of the battery on the same instrument in the same process. A battery whose
  // control has moved has not measured its variants either.
  // The ramp control. `warm` is the discarded warm-up configuration and `base`
  // is the identical binary run immediately after it: the gap between them is
  // how much of the device's clock ramp this battery would have charged to
  // whichever variant came first.
  std::puts("\n## ramp control - the discarded warm-up beside the recorded `base`\n");
  std::printf("| shape | warm-up (discarded) GB/s | recorded `base` GB/s | ramp |\n|---|---|---|---|\n");
  for (size_t i = 0; i < base_first.size(); ++i)
    std::printf("| %s | %.0f | %.0f | %+.1f%% |\n", kShapes[i].name, warm_first[i], base_first[i],
                100.0 * (base_first[i] / warm_first[i] - 1.0));

  std::puts("\n## drift control - `base` re-measured after the whole battery\n");
  std::printf("| shape | GB/s at battery start | GB/s at battery end | Δ%% |\n|---|---|---|---|\n");
  double worst = 0;
  for (size_t i = 0; i < sizeof(kShapes) / sizeof(kShapes[0]); ++i) {
    const LoadShape& sh = kShapes[i];
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    const std::string name = pgl_name("base", sh.K, sh.N, sh.S_prod, 1);
    // Same discarded warm-up as the battery: regenerating the random weights
    // above is seconds of GPU idle, and without this the control measures the
    // ramp back out of it instead of the drift it is here for.
    run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, name.c_str()}, 40);
    GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, sh.S_prod, 1, name.c_str()}, 40);
    const double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
    const double d = 100.0 * (gbps / base_first[i] - 1.0);
    worst = std::max(worst, d < 0 ? -d : d);
    std::printf("| %s | %.0f | %.0f | %+.2f%% |\n", sh.name, base_first[i], gbps, d);
    std::fflush(stdout);
  }
  std::printf("\nworst |Δ| over the six controls: **%.2f%%** (the standing per-cell "
              "repeatability of this instrument is a median 0.21%%)\n", worst);
  std::printf("\nbyte-identity and CPU-reference check: %s\n",
              all_ok ? "every row passed" : "**A ROW FAILED - see the WRONG/DIFFER cells above**");
  return all_ok ? 0 : 1;
}
// --- PERF-111: k-major tiles ---------------------------------------------------
// `probe_gemv --tiles`: each layout-1 / head gemv against its GEMV_KMAJOR (and GEMV_AHEAD) builds, A B A B. The
// control is the production binary over today's [n_tile][k] tiles; each variant reads the same tiles reordered
// [k][n_tile] (common::tiles_k_major) or, for `_A4`, today's order. Every variant must match the control byte for
// byte (same arithmetic, same order). GB/s counts weight bytes; ~604 is probe_bw's grid-stride read. The int8
// head's control is the explicit n-major build (production follows loader::kInt8HeadKMajor).
struct TileShape {
  const char* name;
  char kind;  // '4' int4 layout 1, '8' gemv_i8w, 'b' gemv_bf16
  uint32_t K, N;
  int launches;
  const char* control;
  std::vector<const char*> variants;  // names ending in _A4 read today's order; every other one k-major
};

int run_tiles(l0::Context& ctx, l0::Queue& q, l0::Fence& f) {
  const TileShape shapes[] = {
      {"qkv‖z int4", '4', 5120, 16384, 40, "gemv_M1_K5120_N16384_S1_L1",
       {"gemv_M1_K5120_N16384_S1_L1_KM", "gemv_M1_K5120_N16384_S1_L1_KM_A2"}},
      {"lm_head int4", '4', 5120, 248320, 10, "gemv_M1_K5120_N248320_S1_L1",
       {"gemv_M1_K5120_N248320_S1_L1_KM", "gemv_M1_K5120_N248320_S1_L1_KM_A2"}},
      {"lm_head int8", '8', 5120, 248320, 10, "pi8h_gemv_i8w_M1_K5120_N248320_NM",
       {"gemv_i8w_M1_K5120_N248320_KM", "gemv_i8w_M1_K5120_N248320_KM_A4", "gemv_i8w_M1_K5120_N248320_A4"}},
      {"lm_head bf16", 'b', 5120, 248320, 10, "gemv_bf16_M1_K5120_N248320",
       {"gemv_bf16_M1_K5120_N248320_KM", "gemv_bf16_M1_K5120_N248320_KM_A4", "gemv_bf16_M1_K5120_N248320_A4"}},
  };
  bool all_ok = true;
  std::puts("# probe_gemv --tiles - k-major tiles\n");
  for (const TileShape& sh : shapes) {
    // Today's tiles and scales as bytes; the k-major copy is the same tiles reordered.
    std::vector<uint8_t> w, scales;
    size_t tile_bytes = 256, k_steps = sh.K / 16;
    if (sh.kind == '4') {
      common::Int4Gptq g = common::Int4Gptq::random(sh.K, sh.N, 42);
      std::vector<uint32_t> t = g.tiled();
      w.assign(reinterpret_cast<const uint8_t*>(t.data()), reinterpret_cast<const uint8_t*>(t.data() + t.size()));
      scales.assign(reinterpret_cast<const uint8_t*>(g.scales.data()),
                    reinterpret_cast<const uint8_t*>(g.scales.data() + g.scales.size()));
      tile_bytes = 544;
      k_steps = sh.K / 64;
    } else if (sh.kind == '8') {
      std::mt19937 rng(42);
      w.resize(size_t(sh.K) * sh.N);
      for (size_t i = 0; i < w.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&w[i], &r, 4);
      }
      std::vector<float> sc(sh.N);
      std::uniform_real_distribution<float> sd(0.001f, 0.01f);
      for (float& v : sc) v = sd(rng);
      scales.assign(reinterpret_cast<const uint8_t*>(sc.data()), reinterpret_cast<const uint8_t*>(sc.data() + sc.size()));
    } else {
      std::vector<uint16_t> t = random_bf16(size_t(sh.K) * sh.N, 3, -0.05f, 0.05f);
      w.assign(reinterpret_cast<const uint8_t*>(t.data()), reinterpret_cast<const uint8_t*>(t.data() + t.size()));
      k_steps = sh.K / 8;
    }
    std::vector<uint8_t> wk(w.size());
    common::tiles_k_major(w.data(), sh.N / 16, k_steps, tile_bytes, wk.data());
    // An int4 layout-1 tile carries its scales, so the scales buffer is bound but unread there.
    const size_t weight_bytes = sh.kind == '4' ? w.size() : w.size() + (sh.kind == '8' ? scales.size() : 0);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);

    l0::CmdList imm = l0::CmdList::immediate(ctx);
    const int NB = std::max(2, int((72u << 20) / w.size()) + 1);
    std::vector<l0::Mem> wb, wkb;
    for (int i = 0; i < NB; ++i) {
      wb.emplace_back(ctx, l0::MemKind::Device, w.size());
      imm.copy(wb.back().ptr(), w.data(), w.size());
      wkb.emplace_back(ctx, l0::MemKind::Device, w.size());
      imm.copy(wkb.back().ptr(), wk.data(), w.size());
    }
    l0::Mem sbuf(ctx, l0::MemKind::Device, std::max<size_t>(scales.size(), 4));
    if (!scales.empty()) imm.copy(sbuf.ptr(), scales.data(), scales.size());
    l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
    imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
    l0::Mem obuf(ctx, l0::MemKind::Device, size_t(sh.N) * 4);
    const char* entry = sh.kind == '4' ? "gemv" : sh.kind == '8' ? "gemv_i8w" : "gemv_bf16";

    // One timed run: the correctness launch over copy 0 (read back), then `launches` launches cycling the copies.
    auto run = [&](const char* name, std::vector<float>& out) {
      const std::string n(name);
      const bool kmajor = n != sh.control && n.substr(n.size() - 3) != "_A4";
      std::vector<l0::Mem>& bufs = kmajor ? wkb : wb;
      l0::Module mod(ctx, kernels::path(n));
      l0::Kernel k = mod.kernel(entry);
      k.group_size(64);
      auto bind = [&](int i) {
        uint32_t a = 0;
        k.arg_ptr(a++, bufs[i].ptr());
        if (sh.kind != 'b') k.arg_ptr(a++, sbuf.ptr());
        k.arg_ptr(a++, xbuf.ptr());
        k.arg_ptr(a++, obuf.ptr());
      };
      {
        l0::CmdList list = l0::CmdList::regular(ctx);
        bind(0);
        list.launch(k, sh.N / 64);
        list.close();
        q.execute(list, &f);
        f.wait();
        out.resize(sh.N);
        imm.copy(out.data(), obuf.ptr(), out.size() * 4);
      }
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int i = 0; i < sh.launches; ++i) { bind(i % NB); list.launch(k, sh.N / 64); }
      list.close();
      return double(weight_bytes) / (time_list(q, f, list, sh.launches) * 1e3);
    };

    std::printf("\n## %s - %u×%u, %.1f MB of weights\n\n", sh.name, sh.K, sh.N, double(weight_bytes) / 1e6);
    std::vector<float> control_out, out;
    run(sh.control, control_out);  // warm-up, discarded
    std::vector<const char*> names{sh.control};
    names.insert(names.end(), sh.variants.begin(), sh.variants.end());
    std::vector<std::vector<double>> gbps(names.size());
    std::vector<bool> same(names.size(), true);
    for (int round = 0; round < 2; ++round)
      for (size_t i = 0; i < names.size(); ++i) {
        gbps[i].push_back(run(names[i], out));
        if (i == 0) control_out = out;
        else if (std::memcmp(out.data(), control_out.data(), out.size() * 4) != 0) same[i] = false;
      }
    std::printf("| kernel | round 1 GB/s | round 2 GB/s | mean | %% of 604 | %% of 638 | Δ%% vs control | bytes |\n");
    std::printf("|---|---|---|---|---|---|---|---|\n");
    const double base = (gbps[0][0] + gbps[0][1]) / 2;
    for (size_t i = 0; i < names.size(); ++i) {
      const double m = (gbps[i][0] + gbps[i][1]) / 2;
      all_ok = all_ok && same[i];
      std::printf("| %s | %.0f | %.0f | %.0f | %.1f%% | %.1f%% | %+.1f%% | %s |\n", names[i], gbps[i][0], gbps[i][1], m,
                  100.0 * m / 604.0, 100.0 * m / 638.0, 100.0 * (m / base - 1.0),
                  i == 0 ? "(the control)" : same[i] ? "identical" : "**DIFFER**");
    }
    std::fflush(stdout);
  }
  std::printf("\nbyte compare against the control: %s\n", all_ok ? "every variant identical" : "**A VARIANT DIFFERS**");
  return all_ok ? 0 : 1;
}
// `probe_gemv --tiles i8head` (also the last part of `--tiles`): every int8 output head at the M its model binds -
// gemv_i8w at decode and where a verify list binds it, gemm_sm_i8w where kSmallMTuning puts the verify rows - in
// today's n-major order (_NM, the control) and k-major (_KM), both built with the order spelled out
// (tools/probe/CMakeLists.txt PI8H_CELLS, this table's order). One weight a shape, random int8 and scales, the
// k-major copy by common::tiles_k_major; M random activation rows. Warm-up, then NM KM NM KM; the KM output must be
// the NM output byte for byte. The gate (PERF-111): no regression at any M a model's verify binds.
struct I8HeadCell {
  const char* model;
  bool gemm;  // gemm_sm_i8w (else gemv_i8w)
  uint32_t K, N, M;
  bool wg;    // gemm_sm's in-work-group split-K (SPLITK 1)
};
const I8HeadCell kI8HeadCells[] = {
    {"Qwen3.8", false, 5120, 248320, 1, false},  {"Qwen3.8", false, 5120, 248320, 4, false},
    {"Qwen3.8", false, 5120, 248320, 8, false},  {"Qwen3.8", false, 5120, 248320, 16, false},
    {"Qwen3.8", true, 5120, 248320, 4, true},    {"Qwen3.8", true, 5120, 248320, 8, true},
    {"Qwen3.8", true, 5120, 248320, 16, true},   {"Qwen3.8", true, 5120, 248320, 32, true},
    {"Qwen3.8", true, 5120, 248320, 64, true},
    {"Qwen3.8 tp", false, 5120, 124160, 1, false}, {"Qwen3.8 tp", true, 5120, 124160, 4, true},
    {"Qwen3.8 tp", true, 5120, 124160, 8, true},
    {"Qwen3.8 32k draft", false, 5120, 32768, 1, false}, {"Qwen3.8 32k draft", true, 5120, 32768, 7, true},
    {"Flash-Next", false, 2560, 248320, 1, false}, {"Flash-Next", true, 2560, 248320, 4, true},
    {"Flash-Next tp", false, 2560, 124160, 1, false}, {"Flash-Next tp", false, 2560, 124160, 4, false},
    {"Ornith", false, 2048, 248320, 1, false},     {"Ornith", true, 2048, 248320, 4, false},
    {"Ornith", true, 2048, 248320, 32, true},     {"Ornith", true, 2048, 248320, 64, true},
    {"Kolibri", false, 2560, 128000, 1, false},    {"Kolibri", false, 2560, 128000, 4, false},
    {"Kolibri", false, 2560, 128000, 8, false},
};

int run_i8head(l0::Context& ctx, l0::Queue& q, l0::Fence& f) {
  // The orders: n-major (the control), k-major, and the K-blocked k-major, B k16 steps of a tile together (_KB<B>).
  struct Order { const char* tag; int block; };   // block 0: n-major, 1: plain k-major
  // A trailing A: the same order, gemm_sm with I8_AHEAD 1.
  const Order orders[] = {{"NM", 0},   {"KM", 1},   {"KB4", 4},   {"KB8", 8},     {"KB16", 16},    {"KB32", 32},
                          {"NMA", 0},  {"KMA", 1},  {"KB8A", 8},  {"KB16A", 16},  {"KB32A", 32}};
  constexpr size_t NO = sizeof(orders) / sizeof(orders[0]);
  bool all_ok = true;
  std::puts("\n# probe_gemv --tiles i8head - the int8 heads, n-major (control) vs k-major and K-blocked k-major\n");
  std::printf("| model | kernel | M | K×N | NM GB/s |");
  for (size_t o = 1; o < NO; ++o) std::printf(" Δ%% %s |", orders[o].tag);
  std::printf(" bytes |\n|---|---|---|---|---|");
  for (size_t o = 1; o < NO; ++o) std::printf("---|");
  std::printf("---|\n");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  uint32_t K = 0, N = 0;
  std::vector<std::vector<l0::Mem>> wb(NO);
  std::unique_ptr<l0::Mem> sbuf;
  size_t weight_bytes = 0;
  int NB = 2;
  for (const I8HeadCell& c : kI8HeadCells) {
    if (c.K != K || c.N != N) {   // a new shape: today's tiles and each reordered copy, NB copies of each
      K = c.K;
      N = c.N;
      for (auto& v : wb) v.clear();
      std::mt19937 rng(42);
      std::vector<uint8_t> w(size_t(K) * N), wo(w.size());
      for (size_t i = 0; i < w.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&w[i], &r, 4);
      }
      std::vector<float> sc(N);
      std::uniform_real_distribution<float> sd(0.001f, 0.01f);
      for (float& v : sc) v = sd(rng);
      NB = std::max(2, int((72u << 20) / w.size()) + 1);
      for (size_t o = 0; o < NO; ++o) {
        if (orders[o].block) common::tiles_k_major(w.data(), N / 16, K / 16, 256, wo.data(), orders[o].block);
        const std::vector<uint8_t>& src = orders[o].block ? wo : w;
        for (int i = 0; i < NB; ++i) {
          wb[o].emplace_back(ctx, l0::MemKind::Device, w.size());
          imm.copy(wb[o].back().ptr(), src.data(), w.size());
        }
      }
      sbuf = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, sc.size() * 4);
      imm.copy(sbuf->ptr(), sc.data(), sc.size() * 4);
      weight_bytes = w.size() + sc.size() * 4;
    }
    std::vector<uint16_t> x = random_bf16(size_t(c.M) * K, 5);
    l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
    imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
    l0::Mem obuf(ctx, l0::MemKind::Device, size_t(c.M) * N * 4);
    const std::string base = std::string(c.gemm ? "pi8h_gemm_sm_i8w" : "pi8h_gemv_i8w") + "_M" + std::to_string(c.M) +
                             "_K" + std::to_string(K) + "_N" + std::to_string(N) + (c.wg ? "_wg" : "");
    const uint32_t wg = c.gemm && c.wg ? 256 : 64, cols = c.gemm && c.wg ? 32 : 64;
    const int launches = 10;
    // One timed run of one order: the correctness launch over copy 0 (read back), then `launches` cycling the copies.
    auto run = [&](size_t o, std::vector<float>& out) {
      l0::Module mod(ctx, kernels::path(base + "_" + orders[o].tag));
      l0::Kernel k = mod.kernel(c.gemm ? "gemm_sm_i8w" : "gemv_i8w");
      k.group_size(wg);
      auto bind = [&](int i) {
        k.arg_ptr(0, wb[o][i].ptr());
        k.arg_ptr(1, sbuf->ptr());
        k.arg_ptr(2, xbuf.ptr());
        k.arg_ptr(3, obuf.ptr());
      };
      {
        l0::CmdList list = l0::CmdList::regular(ctx);
        bind(0);
        list.launch(k, N / cols);
        list.close();
        q.execute(list, &f);
        f.wait();
        out.resize(size_t(c.M) * N);
        imm.copy(out.data(), obuf.ptr(), out.size() * 4);
      }
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int i = 0; i < launches; ++i) { bind(i % NB); list.launch(k, N / cols); }
      list.close();
      return double(weight_bytes) / (time_list(q, f, list, launches) * 1e3);
    };
    std::vector<float> ctl, out;
    for (size_t o = 0; o < NO; ++o) run(o, out);   // warm-up, discarded
    double g[2][NO];
    bool same = true;
    for (int round = 0; round < 2; ++round)
      for (size_t o = 0; o < NO; ++o) {
        g[round][o] = run(o, o == 0 ? ctl : out);
        if (o) same = same && std::memcmp(out.data(), ctl.data(), out.size() * 4) == 0;
      }
    all_ok = all_ok && same;
    const double nm = (g[0][0] + g[1][0]) / 2;
    std::printf("| %s | %s | %u | %u×%u | %.0f |", c.model,
                c.gemm ? (c.wg ? "gemm_sm_i8w wg" : "gemm_sm_i8w") : "gemv_i8w", c.M, K, N, nm);
    for (size_t o = 1; o < NO; ++o) std::printf(" %+.1f%% |", 100.0 * ((g[0][o] + g[1][o]) / 2 / nm - 1.0));
    std::printf(" %s |\n", same ? "identical" : "**DIFFER**");
    std::fflush(stdout);
  }
  std::printf("\nbyte compare against NM: %s\n", all_ok ? "every cell identical" : "**A CELL DIFFERS**");
  return all_ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  if (argc > 1 && std::string(argv[1]) == "--loads") return run_loads(ctx, q, f);
  if (argc > 2 && std::string(argv[1]) == "--tiles" && std::string(argv[2]) == "i8head") return run_i8head(ctx, q, f);
  if (argc > 1 && std::string(argv[1]) == "--tiles") {
    const int rc = run_tiles(ctx, q, f);
    return run_i8head(ctx, q, f) | rc;
  }
  // --variants K N S <kernel>...: each named gemv.cl binary at one shape; the first is the control, every other one
  // must be its output byte for byte (the load knobs keep the arithmetic). Run twice in order, the first pass a warm-up.
  if (argc > 5 && std::string(argv[1]) == "--variants") {
    const uint32_t K = uint32_t(std::atoi(argv[2])), N = uint32_t(std::atoi(argv[3])), S = uint32_t(std::atoi(argv[4]));
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 42);
    std::vector<uint16_t> x = random_bf16(K, 5);
    std::vector<float> base;
    bool ok = true;
    for (int pass = 0; pass < 2; ++pass)
      for (int a = 5; a < argc; ++a) {
        GemvResult r = run_gemv(ctx, q, f, w, x, {1, K, N, S, 0, argv[a]}, 40);
        if (base.empty()) base = r.out;
        const bool same = std::memcmp(base.data(), r.out.data(), base.size() * 4) == 0;
        ok = ok && same;
        if (pass) std::printf("%-34s %7.1f us %5.0f GB/s %s\n", argv[a], r.us_per_launch,
                              double(r.weight_bytes) / (r.us_per_launch * 1e3), same ? "identical" : "**DIFFER**");
      }
    return ok ? 0 : 1;
  }
  // --pair K N S L <control>: the named control binary (gemv's 64 work-items) against the engine's binary of the shape
  // (kernels::gemv_variant, its work-group: PERF-108's _KS builds), one warm-up pair, then two interleaved pairs. A
  // reordered sum is held to the double-precision reference (both errors printed), not to bit-identity.
  if (argc == 7 && std::string(argv[1]) == "--pair") {
    const uint32_t K = uint32_t(std::atoi(argv[2])), N = uint32_t(std::atoi(argv[3])), S = uint32_t(std::atoi(argv[4])),
                   L = uint32_t(std::atoi(argv[5]));
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 42);
    std::vector<uint16_t> x = random_bf16(K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    const std::string cand = kernels::gemv_variant(1, K, N, S, L);
    double us[2] = {0, 0}, err[2] = {0, 0};
    bool same = true;
    for (int pass = 0; pass < 3; ++pass) {
      GemvResult a = run_gemv(ctx, q, f, w, x, {1, K, N, S, L, argv[6]}, 40);
      GemvResult b = run_gemv(ctx, q, f, w, x, {1, K, N, S, L}, 40);
      if (pass == 0) continue;
      us[0] += a.us_per_launch / 2;
      us[1] += b.us_per_launch / 2;
      err[0] = max_abs_err(a.out, ref);
      err[1] = max_abs_err(b.out, ref);
      same = std::memcmp(a.out.data(), b.out.data(), a.out.size() * 4) == 0;
      std::printf("pair %d: %s %.2f us, %s %.2f us\n", pass, argv[6], a.us_per_launch, cand.c_str(), b.us_per_launch);
    }
    const double mb = double(w.bytes()) * 1e-3;
    std::printf("control   %-34s %7.2f us %5.0f GB/s  max abs err vs fp64 %.3g\n", argv[6], us[0], mb / us[0], err[0]);
    std::printf("candidate %-34s %7.2f us %5.0f GB/s  max abs err vs fp64 %.3g (tol %.3g)%s\n", cand.c_str(), us[1],
                mb / us[1], err[1], tol, same ? ", bitwise the control" : "");
    std::printf("candidate / control: %.3f\n", us[1] / us[0]);
    return err[0] <= tol && err[1] <= tol ? 0 : 1;
  }
  const double peak = kPeak;
  struct Shape { uint32_t K, N; const char* name; };
  // Names use U+2016 (‖), not '|': this table is pasted into docs/ as markdown.
  const Shape shapes[] = {{6144, 5120, "out/o_proj"}, {5120, 14336, "q‖k‖v"}, {5120, 16384, "qkv‖z"},
                          {5120, 34816, "gate‖up"}, {17408, 5120, "down"}};
  const uint32_t Ss[] = {1, 2, 4, 8, 16};
  double layout_sum[2] = {0, 0};
  bool all_ok = true;  // any row over tolerance makes the probe exit non-zero
  std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, double>> best;  // (L, shape idx) -> S -> GB/s

  std::printf("| shape | K×N | L | S | µs | GB/s | %% of 600 | max abs err | tol |\n|---|---|---|---|---|---|---|---|---|\n");
  for (uint32_t si = 0; si < 5; ++si) {
    const Shape& sh = shapes[si];
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42 + si);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    for (uint32_t L = 0; L < 2; ++L)
      for (uint32_t S : Ss) {
        GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, S, L}, 40);
        double err = max_abs_err(r.out, ref);
        if (err > tol) all_ok = false;
        double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
        best[{L, si}][S] = gbps;
        std::printf("| %s | %u×%u | %u | %u | %.1f | %.0f | %.0f%% | %.2g | %.2g |%s\n", sh.name, sh.K, sh.N, L, S,
                    r.us_per_launch, gbps, 100.0 * gbps / peak, err, tol, err <= tol ? "" : "  **WRONG**");
        std::fflush(stdout);
      }
  }
  // lm_head, bf16
  {
    std::vector<uint16_t> w_rm = random_bf16(size_t(248320) * 5120, 3, -0.05f, 0.05f);
    common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), 5120, 248320);
    std::vector<uint16_t> x = random_bf16(5120, 5);
    GemvResult r = run_gemv_bf16(ctx, q, f, w, x, 1, 10);
    double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
    std::printf("| lm_head bf16 | 5120×248320 | B | 1 | %.1f | %.0f | %.0f%% | - | - |\n", r.us_per_launch, gbps, 100.0 * gbps / peak);
  }
  std::puts("\nDecision:");
  for (uint32_t L = 0; L < 2; ++L) {
    for (uint32_t si = 0; si < 5; ++si) {
      double b = 0; for (auto& e : best[{L, si}]) b = std::max(b, e.second);
      layout_sum[L] += b;
    }
    std::printf("- layout %u: sum of best GB/s over shapes = %.0f\n", L, layout_sum[L]);
  }
  const uint32_t Lwin = layout_sum[1] > layout_sum[0] ? 1 : 0;
  std::printf("- canonical layout: %u\n", Lwin);
  for (uint32_t si = 0; si < 5; ++si) {
    double b = 0; for (auto& e : best[{Lwin, si}]) b = std::max(b, e.second);
    uint32_t pick = 16;
    for (uint32_t S : Ss) if (best[{Lwin, si}][S] >= 0.97 * b) { pick = S; break; }
    std::printf("- %s (%u×%u): S = %u\n", shapes[si].name, shapes[si].K, shapes[si].N, pick);
  }
  return all_ok ? 0 : 1;
}
