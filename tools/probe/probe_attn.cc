// probe_attn: what does `attn_decode`'s 224 µs/launch buy?
//
// Spec 1.5's re-assessment memo (§5.2) ranks this measurement above every
// design on its menu: `attn_decode` is 3.585 ms/token - 9.9% of the step and
// the largest non-GEMV row - and **four pre-registered cost models have died on
// it** (docs/15 §2 models A and B, §L5 models C and D). Nobody knows what the
// launch spends. This probe does not fit a fifth model. It transplants the
// launch out of the engine and then removes ONE term at a time, so that what
// each term costs is a subtraction between two measurements rather than a
// parameter.
//
// **The base variant is the experiment's control and its first job is to fail
// falsifiably**: `tools/probe/probe_attn.cl` at every switch default is
// `attn_decode` line for line, and if it does not reproduce the in-situ
// 224.046 µs/launch at depth 4096 then the transplant is invalid and no
// ablation below means anything. The report says which happened.
//
// Two properties of the transplant are deliberate and both matter:
//
//   * **The KV is incompressible.** The B70 compresses device-local memory
//     losslessly, so a constant-filled cache reads back far above peak
//     bandwidth (docs/01, and probe_bw says the same). The caches here are
//     filled with random bf16 in a model-like range.
//   * **The L2 is not allowed to stay warm.** One FA layer's live KV at depth
//     4096 is 16.8 MB against 24 MB of L2, so back-to-back launches over ONE
//     cache would run out of a cache the engine never has: in situ, ~45 other
//     kernels read ~1.8 GB between two `attn_decode` launches. `--kvsets N`
//     rotates the launch over N independent KV caches; the default 4 puts
//     50 MB of other traffic between two launches of the same one. `--kvsets 1`
//     is kept because the difference between the two IS one of the terms.
//
// Nothing here is a shipped code path: `src/kernels/attn.cl` is untouched, the
// ablations deliberately compute wrong answers, and the outputs are read only
// so that the compiler cannot delete the work.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

namespace {

// The shape. These are the model's (docs/03) and the loader's default max_len;
// a probe that measures a different shape measures nothing.
constexpr uint32_t kKvHeads = 4;
constexpr uint32_t kQHeads = 24;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kPart = 258;
constexpr uint32_t kWg = 256;

// The list is replayed kReplays times and the first kWarmup replays are thrown
// away - gemv_harness.h's 8 and 3, so every probe in this tree warms up the
// same way. At the default --reps that is 120 warm-up launches and 200 timed
// ones; see the comment at the timing loop for what happens with fewer.
constexpr uint32_t kReplays = 8;
constexpr uint32_t kWarmup = 3;

// The variant battery. `tag` is the binary's suffix and the row label; the
// defines live in tools/probe/CMakeLists.txt beside the same tags. Nothing
// checks the two lists against each other - a tag with no binary throws at
// `l0::Module`, loudly and by name, which is the same guard `ATTN_BLOCK` uses
// across its three homes (src/kernels/CMakeLists.txt).
struct Variant {
  const char* tag;
  uint32_t block;    // PB_BLOCK - decides the grid, so the host has to know it
  const char* what;  // what this row removes, for the report
};
const Variant kVariants[] = {
    {"base", 64, "nothing - the faithful replica, and the control"},
    {"earlyout", 64, "everything: return at the early-out (the grid floor)"},
    {"qstage", 64, "everything but the 6 q-head stagings and their 12 barriers"},
    {"nok", 64, "the K global load (same fma count, same SLM reads)"},
    {"nov", 64, "the V global load (same fma count)"},
    {"noloads", 64, "both global loads"},
    {"hotk", 64, "K's cache misses: same messages, all on block 0 (L1-resident)"},
    {"hotv", 64, "V's cache misses: same messages, all on block 0"},
    {"hotkv", 64, "both loads' cache misses"},
    {"hotk2", 64, "K's misses, with the hot row a function of the wave (no load is hoistable)"},
    {"hotv2", 64, "V's misses, with the hot row a function of the wave (no load is hoistable)"},
    {"hotkv2", 64, "both, hot rows a function of the wave - the LICM control for `hotkv`"},
    {"kt", 64, "K's 2048 B position stride - a `[head][pos][dim]` cache instead"},
    {"vt", 64, "V's 2048 B position stride - a `[head][pos][dim]` cache instead"},
    {"kvt", 64, "both strides (the transposed KV cache)"},
    {"kvt_vec2", 64, "both strides AND half the K messages"},
    {"nopart", 64, "the 1.6 MB of `attn_part` stores (predicated off)"},
    {"vec2", 64, "half the K messages (ushort2: 64 B each)"},
    {"vec4", 64, "three quarters of the K messages (ushort4: 128 B each)"},
    {"natexp", 64, "exp's precision: native_exp instead"},
    {"noexp", 64, "exp entirely (a multiply-add stands in)"},
    {"notree", 64, "the 4-step SLM reduction and its 4 barriers"},
    {"nobar", 64, "all 6 work-group barriers per wave (same instructions)"},
    {"sgtree", 64, "3 of the 4 tree barriers' SCOPE - subgroup, not work-group; same arithmetic"},
    {"nosoftmax", 64, "the whole online update (fmax, exp, ssum, both rescales)"},
    {"depbreak", 64, "the loop-carried chain: waves stop depending on each other"},
    {"prefetch", 64, "the K load's latency: next wave's row issued before this wave's barriers"},
    {"mathonly", 64, "both loads AND exp - what is left is issue and barriers"},
    {"loadsonly", 64, "the online update and the tree - loads, the dot and the V sum remain"},
    {"gqa1", 64, "5 of the 6 q-head passes"},
    {"gqa2", 64, "4 of the 6 q-head passes"},
    {"gqa3", 64, "3 of the 6 q-head passes"},
    {"b2d", 64, "15 of every 16 K messages: the row in TWO 2D block reads (same order)"},
    {"b2d1", 64, "all but one K message: the row in ONE 2D block read (same order)"},
    {"kvt_b2d", 64, "both strides AND the row in two 2D block reads"},
    {"b32", 32, "half the block - the L5 sweep's fourth point, transplanted"},
    {"b128", 128, "- double the block (L5's second point)"},
    {"b256", 256, "- the pre-L5 block (L5's first point)"},
};

uint32_t xs(uint32_t& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

// f32 -> bf16, round-to-nearest-even (common::f32_to_bf16's add-and-shift).
uint16_t rne_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t((u + rounding) >> 16);
}

// Model-like KV: uniform [-1, 1) rounded to bf16. ~10 bits of entropy per
// 16-bit word, so a 64 B cache line holds ~320 random bits and the device's
// lossless compression has nothing to work with - the property probe_bw and
// probe_gemv both depend on.
void fill_kv(std::vector<uint16_t>& v, uint32_t seed) {
  uint32_t s = seed | 1u;
  for (uint16_t& h : v) h = rne_bf16(2.0f * (float(xs(s) >> 8) / 16777216.0f) - 1.0f);
}

struct Row {
  double median = 0, mean = 0, lo = 0, hi = 0;
};

Row summarise(std::vector<double> us) {
  Row r;
  if (us.empty()) return r;
  std::sort(us.begin(), us.end());
  r.median = us[us.size() / 2];
  r.lo = us.front();
  r.hi = us.back();
  for (double x : us) r.mean += x;
  r.mean /= double(us.size());
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t pos = 4096, reps = 40, kvsets = 4;
  std::vector<std::string> only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* f) -> std::string {
      if (++i >= argc) {
        std::fprintf(stderr, "probe_attn: %s needs a value\n", f);
        std::exit(2);
      }
      return argv[i];
    };
    if (a == "--pos") {
      pos = uint32_t(std::stoul(val("--pos")));
    } else if (a == "--reps") {
      reps = uint32_t(std::stoul(val("--reps")));
    } else if (a == "--kvsets") {
      kvsets = uint32_t(std::stoul(val("--kvsets")));
    } else if (a == "--only") {
      only.push_back(val("--only"));
    } else {
      std::fprintf(stderr,
                   "usage: probe_attn [--pos 4096] [--reps 40] [--kvsets 4] [--only <tag>]...\n"
                   "  --pos     context depth the launch attends (ctrl[CTRL_POS])\n"
                   "  --reps    launches per replay; the list is replayed 8x, the first 3\n"
                   "            dropped for the clock ramp, so 40 means 120 warm-up launches\n"
                   "            and 200 timed ones. Below ~100 warm-up launches the number is\n"
                   "            the ramp, not the kernel (measured: 264 vs 206 us).\n"
                   "  --kvsets  independent KV caches the launches rotate over. 4 keeps 50 MB of\n"
                   "            other traffic between two launches of one cache, so the L2 is as\n"
                   "            cold as it is in situ. 1 measures the warm case on purpose.\n"
                   "  --only    run just this variant tag (repeatable)\n");
      return 2;
    }
  }
  if (reps == 0 || kvsets == 0 || pos == 0 || pos > kMaxLen - 1) {
    std::fprintf(stderr, "probe_attn: --pos in [1, %u), --reps and --kvsets >= 1\n", kMaxLen);
    return 2;
  }

  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  // The buffers `attn_decode` binds, at the engine's own sizes.
  const size_t kv_elems = size_t(kMaxLen) * kKvHeads * kHeadDim;
  l0::Mem ctrl(ctx, l0::MemKind::Shared, 128);
  std::memset(ctrl.ptr(), 0, 128);
  ctrl.as<uint32_t>()[0] = pos;  // CTRL_POS
  ctrl.as<uint32_t>()[1] = 1;    // CTRL_NACT

  l0::Mem attn_q(ctx, l0::MemKind::Device, size_t(kQHeads) * kHeadDim * sizeof(float));
  {
    std::vector<float> h(size_t(kQHeads) * kHeadDim);
    uint32_t s = 12345;
    for (float& x : h) x = 2.0f * (float(xs(s) >> 8) / 16777216.0f) - 1.0f;
    imm.copy(attn_q.ptr(), h.data(), h.size() * sizeof(float));
  }

  std::vector<l0::Mem> kv_k, kv_v;
  kv_k.reserve(kvsets);
  kv_v.reserve(kvsets);
  {
    std::vector<uint16_t> h(kv_elems);
    for (uint32_t i = 0; i < kvsets; ++i) {
      kv_k.emplace_back(ctx, l0::MemKind::Device, kv_elems * 2);
      fill_kv(h, 0x9E3779B9u + 7u * i);
      imm.copy(kv_k.back().ptr(), h.data(), h.size() * 2);
      kv_v.emplace_back(ctx, l0::MemKind::Device, kv_elems * 2);
      fill_kv(h, 0x85EBCA6Bu + 7u * i);
      imm.copy(kv_v.back().ptr(), h.data(), h.size() * 2);
    }
  }
  // Sized for the finest block in the battery, so one allocation serves all.
  const uint32_t max_blocks = kMaxLen / 32;
  l0::Mem attn_part(ctx, l0::MemKind::Device,
                    size_t(kQHeads) * max_blocks * kPart * sizeof(float));

  // Output byte-check. Most variants here compute a wrong answer on purpose, so
  // this is a COLUMN and not a gate -- except for `b2d`/`b2d1`, which are the
  // only variants that claim the same values in the same order as `base`. That
  // claim rests on a guess about how a 2D block read distributes its rows over
  // the subgroup, and a guess about a load's data mapping is exactly what
  // silently invalidated three rows of the GEMV battery this hour. So it is
  // checked, and a mismatch makes this probe exit non-zero.
  const size_t part_floats = size_t(kQHeads) * (kMaxLen / 32) * kPart;
  std::vector<float> part_host(part_floats), part_base;
  bool b2d_ok = true;

  l0::EventPool pool(ctx, reps);
  std::vector<l0::Event> events;
  events.reserve(reps);
  for (uint32_t i = 0; i < reps; ++i) events.emplace_back(pool, i);

  std::printf("# probe_attn - `attn_decode`, one term removed at a time\n"
              "pos %u, %u launches/replay x %u timed replays (%u dropped for the clock ramp),\n"
              "%u KV caches rotated, device %s (%u EUs)\n"
              "Per-launch us are kernelStart->kernelEnd from the same timestamp events\n"
              "`b70-decode --profile` uses, so they are directly comparable to the in-situ\n"
              "224.046 us/launch this row exists to reproduce (docs/15 §L5).\n\n",
              pos, reps, kReplays - kWarmup, kWarmup, kvsets, ctx.name().c_str(),
              ctx.eu_count());
  std::printf("| variant | live WGs | us/launch (median) | mean | min .. max | vs base |"
              " wall/reps | bytes | removes |\n|---|---|---|---|---|---|---|---|---|\n");

  double base_us = 0.0;
  for (const Variant& v : kVariants) {
    if (!only.empty() && std::find(only.begin(), only.end(), v.tag) == only.end()) continue;
    const std::string name = std::string("probe_attn_") + v.tag;
    l0::Module mod(ctx, kernels::path(name));
    l0::Kernel k = mod.kernel("probe_attn");
    k.group_size(kWg);
    const uint32_t nblocks = kMaxLen / v.block;
    const uint32_t live = ((pos / v.block) + 1) * kKvHeads;

    // Arguments are resolved at APPEND time (tests/l0/arg_capture_test.cc), so
    // rotating the KV pointer between appends gives launch i its own cache.
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (uint32_t i = 0; i < reps; ++i) {
      const uint32_t set = i % kvsets;
      k.arg_ptr(0, ctrl.ptr());
      k.arg_ptr(1, attn_q.ptr());
      k.arg_ptr(2, kv_k[set].ptr());
      k.arg_ptr(3, kv_v[set].ptr());
      k.arg_ptr(4, attn_part.ptr());
      list.launch(k, kKvHeads, nblocks, 1, &events[i]);
    }
    list.close();

    // **Replay the closed list `kReplays` times and drop the first `kWarmup`** -
    // tests/kernels/gemv_harness.h's convention (8 and 3) and probe_gemv's, and
    // it is not a formality here. MEASURED 2026-08-25: with one warm-up replay
    // of 40 launches this variant read 264.167 µs on one invocation and
    // 206.458 µs on the next; from ~100 warm-up launches onward it locks to
    // 207.2-207.6 µs across reps 100/200/400/800 and two invocations each,
    // i.e. 0.2%. The swing is the device's clock ramp, and a probe that does
    // not out-wait it reports the ramp instead of the kernel.
    std::vector<double> us;
    us.reserve(size_t(reps) * (kReplays - kWarmup));
    double timed_wall_us = 0.0;
    for (uint32_t rep = 0; rep < kReplays; ++rep) {
      for (l0::Event& e : events) e.reset();
      Timer t;
      t.start();
      q.execute(list, &fence);
      fence.wait();
      if (rep < kWarmup) continue;
      timed_wall_us += t.ms() * 1e3;
      for (l0::Event& e : events) us.push_back(e.duration_us());
    }
    const double wall_us = timed_wall_us / double(kReplays - kWarmup);
    const Row r = summarise(us);
    imm.copy(part_host.data(), attn_part.ptr(), part_floats * sizeof(float));
    const char* bytes = "-";
    if (std::strcmp(v.tag, "base") == 0) {
      part_base = part_host;
      bytes = "(the control)";
    } else if (!part_base.empty()) {
      const bool same = std::memcmp(part_base.data(), part_host.data(),
                                    part_floats * sizeof(float)) == 0;
      bytes = same ? "identical" : "differs (by design)";
      // EXACT names, not a prefix. `kvt_b2d` starts with neither and would slip
      // through a `strstr`, and it must: PB_KT/PB_VT reinterpret the same buffer
      // with a different index mapping, so its VALUES differ from base by
      // construction and "differs (by design)" is the correct reading for it.
      // Only these two claim base's bytes.
      const bool claims_base = std::strcmp(v.tag, "b2d") == 0 || std::strcmp(v.tag, "b2d1") == 0;
      if (claims_base && !same) {
        bytes = "**DIFFERS - the mapping guess is WRONG**";
        b2d_ok = false;
      }
    }
    if (std::strcmp(v.tag, "base") == 0) base_us = r.median;
    char rel[32] = "-";
    if (base_us > 0.0 && std::strcmp(v.tag, "base") != 0)
      std::snprintf(rel, sizeof(rel), "%+.1f%%", 100.0 * (r.median - base_us) / base_us);
    // `wall/reps` is the validity check that the launches did NOT overlap: an
    // in-order list runs them back to back, so it must sit just ABOVE the
    // per-launch median (the difference is the inter-launch gap). If it came
    // out well below, the durations would be contended launches and the whole
    // table would be measuring the wrong thing. It is the mean over the timed
    // replays, so it is comparable to the median column directly.
    std::printf("| `%s` | %u | **%.3f** | %.3f | %.3f .. %.3f | %s | %.3f | %s | %s |\n", v.tag,
                live, r.median, r.mean, r.lo, r.hi, rel, wall_us / double(reps), bytes, v.what);
    std::fflush(stdout);
  }
  if (!b2d_ok)
    std::puts("\n**A `b2d` row did not reproduce `base`'s bytes.** Its timing is therefore not a\n"
              "measurement of the same work and must not be read as one.");
  return b2d_ok ? 0 : 1;
}
