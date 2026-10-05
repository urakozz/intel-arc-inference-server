// b70-decode - spec §12, plus spec 1.5 §3.4. Three modes:
//
//   b70-decode <snapshot-or-repo> --ids <file> --n <N> [--prefill [--pp-chunk C]] [--device N] [--max-len 16384]
//   b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]
//   b70-decode <snapshot-or-repo> --profile [--depth 4096] [--steps 32] [--repeats 1]
//                                           [--device N]
//
// The first two run `runtime::Engine`. The third does not: it replays an
// *instrumented* capture and needs one event reset before every replay, which
// is the caller's business by design (runtime/capture.h) and not something an
// Engine that owns its own queue can be asked to do from outside.
//
// stdout is the machine-readable channel and carries nothing but the answer:
// one generated id per line in `--ids` mode, one markdown row in `--bench`
// mode, the anatomy tables in `--profile` mode. Everything else - the device,
// the loader's report, the timings - goes to stderr, so
// `b70-decode … --ids p.ids --n 32 > out` is a file of ids.
//
// There is no tokenizer here (spec 2 owns it): ids in, ids out.
// tools/oracle/tokenize.py is what turns text into an `--ids` file today.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/max_len.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "loader/trained_context.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/profile.h"

namespace {
using model::Qwen35;

// The bench prompt, cycled to --depth: the 42 ids of the committed golden
// prompt `tests/golden/prompts/prose.ids`, baked in so a bench run needs
// nothing but the binary and the checkpoint. Every id is below kVocabUsed, so
// none of them is a row the sampler masks. What matters for a timing run is
// that the ids are fixed and legal, not what they say - decode reads every
// weight per token regardless - but reusing the golden prompt means the depth-N
// bench and the golden gate ingest the same tokens for their first 42 steps.
constexpr uint32_t kBenchPrompt[] = {
    760,   72103, 506, 37119, 557,   11012, 3213, 310,   6512, 279, 61789, 272, 1072,  2272,
    279,   197616, 2271, 13,   469,   68042, 29123, 7247, 383,  279, 1387,  12615, 1345, 279,
    49813, 78911, 1141, 20459, 13,    3113,  7840,  279,  2981, 1000, 381,  16850, 1495, 13};
constexpr size_t kBenchPromptLen = sizeof(kBenchPrompt) / sizeof(kBenchPrompt[0]);

// The MBU line divides by two measured quantities, and only ONE of them is a
// constant:
//   W  - bytes every token reads. **This is a property of the checkpoint, not
//        of the engine**, and since spec 1.6 §5.1 it is no longer one number:
//        15.540 GB with a bf16 `lm_head`, 13.673 GB with a packed one (both
//        measured 2026-08-26). It is therefore read from `LoadReport::
//        read_per_token`, which is what this load actually made resident, and
//        printed beside the percentage so a row always says which W it used.
//        A hardcoded W would have reported the RTN checkpoint's 30.20 t/s as
//        79.5% of the device when the honest figure is 70.0% (measured
//        2026-08-26; 30.20 x 15.53998 / 590 = 79.54 against
//        30.20 x 13.67261 / 590 = 69.99).
//   BW - `tools/probe/probe_bw`, 590 GB/s median at 2 GB through the same
//        Level Zero path this engine submits on (docs/01). That one IS a
//        device constant.
constexpr double kDeviceGBs = 590.0;              // GB/s, measured

void usage() {
  std::fprintf(
      stderr,
      "usage:\n"
       "  b70-decode <snapshot-or-repo> --ids <file> --n <N> [--prefill [--pp-chunk C]\n"
       "                                        [--pp-backend sycl-tla|l0|l0-int8]]\n"
       "                                        [--device N] [--max-len 16384|auto]\n"
      "  b70-decode <snapshot-or-repo> --bench [--depth 4096 | --pp N [--pp-chunk C]\n"
      "                                        [--pp-backend sycl-tla|l0|l0-int8]]\n"
      "                                        [--tg 256] [--device N] [--max-len 16384|auto]\n"
      "  b70-decode <snapshot-or-repo> --profile [--depth 4096] [--steps 32] [--repeats 1]\n"
      "                                          [--device N]\n"
      "\n"
      "  <snapshot-or-repo>  a snapshot directory, or an HF repo id resolved against the local\n"
      "                      cache ($HF_HOME or ~/.cache/huggingface). Never downloads.\n"
      "  --ids <file>   whitespace-separated token ids (tools/oracle/tokenize.py writes them)\n"
       "  --n <N>        ids to generate, greedily; one per line on stdout, t/s on stderr\n"
       "  --prefill      --ids only: run the prompt through Engine::prefill instead of one\n"
       "                 Engine::ingest replay per id; needs the optional prefill component.\n"
      "  --device N     GPU index. Absent: ONEAPI_DEVICE_SELECTOR=level_zero:N, else device 0.\n"
      "  --max-len <L>  KV cache and RoPE capacity: default 16384 (bench rows stay comparable);\n"
      "                 any multiple of 256 up to the trained context (config.json\n"
      "                 max_position_embeddings), refused before the engine if the memory plan\n"
      "                 does not fit; `auto` = the largest that fits (spec 6 §10). Only\n"
      "                 B70_DECODE_ATTN=v1 bakes max_len: 4096, 16384, 32768 and 131072 are compiled,\n"
      "                 any other fails at capture naming the missing binary\n"
      "  --mem-reserve-gb G  device memory the plan leaves free (default 1.5 GB: driver, kernels,\n"
      "                 allocator slack - an estimate the box has to confirm)\n"
      "  --bench        ingest --depth synthetic ids, then time --tg generated ones and print\n"
      "                 a markdown row on stdout\n"
      "  --pp N         --bench only, and exclusive with --depth: prefill N synthetic ids\n"
      "                 through Engine::prefill instead of ingesting them one replay at a\n"
      "                 time, and print a SECOND markdown row with the device-side prefill\n"
      "                 time. The prefilled ids ARE the depth, which is why --depth is\n"
      "                 refused beside it.\n"
       "  --pp-chunk C   --pp or --prefill: positions per prefill chunk (default\n"
       "                 PrefillScratch::kC = 2048, ruling A13). Spec 2 §6.2's multi-chunk\n"
       "                 gate runs at 1024.\n"
       "  --pp-backend B  --pp or --prefill: the GEMM backend, sycl-tla (spec 2), l0 (spec 2.1,\n"
       "                 every GEMM on the Level Zero list) or l0-int8 (spec 5, l0 with every\n"
       "                 int4 linear on the rotated int8 path). Default: l0-int8.\n"
      "  --lm-head H    bf16 (default: the checkpoint's own head, byte-matched with vLLM) or\n"
      "                 int8 (spec 9: quantised per row at load). An int8 run's bench rows\n"
      "                 carry `int8-head` after the sha.\n"
      "  --profile      ingest --depth synthetic ids on a plain list, then replay --steps\n"
      "                 INSTRUMENTED steps and print the per-launch anatomy on stdout.\n"
      "                 Never a bench row: every launch signals a host-visible event\n"
      "                 an unprofiled list does not pay for (spec 1.5 §3.3).\n"
      "  --repeats R    --profile only: R independent sessions of --steps replays each, over\n"
      "                 ONE ingest, each starting from a rewind of pos to the post-ingest\n"
      "                 value. Adds a per-family mean +/- spread table and the 2sigma delta\n"
      "                 this instrument can resolve. Default 1 (the pre-flag report).\n");
}

// The loader reports itself with std::printf - to stdout, which is this CLI's
// machine-readable channel. Rather than change plan 2's file (the report is
// wanted, just not there), point fd 1 at fd 2 for the duration of the load so
// the banner joins the other diagnostics on stderr.
class StdoutToStderr {
 public:
  StdoutToStderr() {
    std::fflush(stdout);
    saved_ = ::dup(1);
    if (saved_ >= 0 && ::dup2(2, 1) < 0) {
      ::close(saved_);
      saved_ = -1;
    }
  }
  ~StdoutToStderr() {
    std::fflush(stdout);
    if (saved_ >= 0) {
      ::dup2(saved_, 1);
      ::close(saved_);
    }
  }
  StdoutToStderr(const StdoutToStderr&) = delete;
  StdoutToStderr& operator=(const StdoutToStderr&) = delete;

 private:
  int saved_ = -1;
};

uint32_t parse_u32(const char* what, const std::string& s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 10)
    throw std::runtime_error(std::string(what) + " expects a non-negative integer, got '" + s + "'");
  const unsigned long long v = std::stoull(s);
  if (v > 0xFFFFFFFFull)
    throw std::runtime_error(std::string(what) + " is out of range: " + s);
  return uint32_t(v);
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  std::string w;
  while (f >> w) {
    const uint32_t id = parse_u32(("--ids file '" + path + "'").c_str(), w);
    // Not kVocabUsed: rows [kVocabUsed, kVocab) exist in the embedding table
    // and a checkpoint may legitimately spell one. Beyond kVocab the gather
    // would read past the allocation, so that is the hard bound.
    if (id >= Qwen35::kVocab)
      throw std::runtime_error("--ids file '" + path + "': id " + std::to_string(id) +
                               " is outside the vocabulary (" + std::to_string(Qwen35::kVocab) +
                               " rows)");
    ids.push_back(id);
  }
  if (!f.eof()) throw std::runtime_error("--ids file '" + path + "' is not whitespace-separated ids");
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}

// One line for both modes, so the ingest and the generate halves are always
// reported the same way.
void report_generate(const runtime::Engine& eng, uint32_t n) {
  std::fprintf(stderr, "generate: %u ids, %.2f t/s, %.2f ms/token (%.1f%% of it inside the fence)\n",
               n, eng.last_tok_per_s(), n ? eng.last_gen_ms() / n : 0.0,
               eng.last_gen_ms() > 0.0 ? 100.0 * eng.last_fence_ms() / eng.last_gen_ms() : 0.0);
}

// --- --profile: the in-situ step anatomy (spec 1.5 §3.4) ---------------------
//
// What this mode measures, and what it does not. Each launch signals a
// kernel-timestamp event, and `kernelStart → kernelEnd` is that kernel's own
// device time: the host-scope flush the signal carries lands *after*
// `kernelEnd`, so a per-kernel number here is directly comparable to a probe's
// per-kernel number. The **gap** (fence wall − Σ durations) is the opposite: it
// contains one such flush per launch, so it is an UPPER BOUND on what an
// unprofiled list pays between kernels. Both statements are printed with the
// numbers, because a number without its provenance is not a measurement.

// One rollup row. `us` is the row's time summed over every profiled step and is
// divided by the step count exactly once, at print time; `launches` is the
// row's launch count in ONE step, because the rollup walks the 774 launch
// indices once and each index's total already covers every step.
struct Agg {
  std::string name;
  double us = 0.0;
  size_t launches = 0;
};

void accumulate(std::vector<Agg>& rows, const std::string& name, double us) {
  for (Agg& a : rows)
    if (a.name == name) {
      a.us += us;
      ++a.launches;
      return;
    }
  rows.push_back(Agg{name, us, 1});
}

// "L12 gemv gemv_M1_K5120_N16384_S1_L1" - the three fields runtime::build
// writes per launch: the layer tag ("--" for the six token-boundary launches -
// embed_gather, the final norm's two, lm_head, and the two argmax stages, named
// in runtime/capture.h), the entry point, and the compiled variant.
struct Label {
  std::string layer, entry, variant;
};
Label split_label(const std::string& s) {
  const size_t a = s.find(' ');
  const size_t b = a == std::string::npos ? a : s.find(' ', a + 1);
  if (a == std::string::npos || b == std::string::npos)
    throw std::runtime_error("profile: unparseable launch label '" + s + "'");
  return Label{s.substr(0, a), s.substr(a + 1, b - a - 1), s.substr(b + 1)};
}

void print_rollup(const char* title, const char* what, std::vector<Agg> rows, uint32_t steps,
                  double total_us) {
  std::sort(rows.begin(), rows.end(), [](const Agg& x, const Agg& y) { return x.us > y.us; });
  std::printf("\n%s\n  %-34s  launches    us/step   share   us/launch\n", title, what);
  for (const Agg& a : rows) {
    const double us = a.us / steps;
    std::printf("  %-34s  %8zu  %9.3f  %5.2f%%  %10.3f\n", a.name.c_str(), a.launches, us,
                100.0 * us / total_us, a.launches ? us / double(a.launches) : 0.0);
  }
}

// --- --repeats: repeated-run averaging (the memo's §5.4) ---------------------
//
// The problem it exists for, measured: the profiler reproduces to 0.006% over
// two runs three minutes apart, but across a day of work on the box the
// *untouched* launches of the step drifted +0.30% / +0.50% / +0.56% across
// spec 1.5's three levers - 0.12 to 0.18 ms per comparison. Two levers in a row
// then missed their in-situ→bench prediction in OPPOSITE directions (L1 by
// +0.087 ms, L5 by −0.319), which is a resolution limit rather than a missing
// constant, and every remaining candidate on the menu is smaller than it
// (docs/superpowers/specs/2026-08-25-spec1.5-reassessment.md §5.4).
//
// A "session" here is `--steps` instrumented replays run after rewinding
// `Control::pos` to the post-ingest value, so **every session profiles the
// identical launch shape** - the same 774 launches at the same positions, the
// same `nb`, the same grid - and the spread across sessions is drift and
// scheduling noise, nothing else. The ingest is paid once for all of them; at
// depth 4096 it is ~170 s and repeating it would dominate the run for a number
// nobody reads.
//
// **What this measures and what it does not.** Sessions are consecutive inside
// one process, so the spread they report is short-timescale drift only. It is a
// LOWER bound on the floor of a before/after comparison between two binaries,
// which is a comparison across two processes and (usually) two build+sync
// cycles. The honest attribution floor is the across-PROCESS spread of the same
// family, obtained by running this command N times; docs/15's instrument
// section records both and quotes the larger one.
struct Stat {
  double mean = 0.0, sd = 0.0, lo = 0.0, hi = 0.0;
  // The sample (n−1) standard deviation: the sessions are a sample of the
  // sessions this box could have run, not the population. 0 by definition at
  // one session, which is why `--repeats 1` prints no spread at all.
  static Stat of(const std::vector<double>& v) {
    Stat s;
    if (v.empty()) return s;
    s.lo = s.hi = v[0];
    for (double x : v) {
      s.mean += x;
      s.lo = std::min(s.lo, x);
      s.hi = std::max(s.hi, x);
    }
    s.mean /= double(v.size());
    if (v.size() < 2) return s;
    double ss = 0.0;
    for (double x : v) ss += (x - s.mean) * (x - s.mean);
    s.sd = std::sqrt(ss / double(v.size() - 1));
    return s;
  }
  // The bar an A/B attribution has to clear. Two binaries measured with R
  // sessions each give two means whose difference has standard error
  // sd·sqrt(2/R) (both sides contribute, both are means of R). At 2σ the
  // resolvable delta is therefore 2·sd·sqrt(2/R) - this is the number docs/15
  // quotes as the instrument's floor, and it is per family because that is the
  // unit an attribution is written in.
  double resolvable_2sigma(size_t sessions) const {
    return sessions < 2 ? 0.0 : 2.0 * sd * std::sqrt(2.0 / double(sessions));
  }
};

// One family's per-session totals, in the order the sessions ran.
struct Series {
  std::string name;
  std::vector<double> us;  // us/step, one entry per session
};

void print_repeat_table(const char* title, const char* what, std::vector<Series> rows,
                        size_t sessions) {
  std::sort(rows.begin(), rows.end(), [](const Series& x, const Series& y) {
    return Stat::of(x.us).mean > Stat::of(y.us).mean;
  });
  std::printf("\n%s\n  %-34s   mean us/step        sd     sd%%        min .. max"
              "    2sigma-resolvable\n",
              title, what);
  for (const Series& r : rows) {
    const Stat s = Stat::of(r.us);
    std::printf("  %-34s  %13.3f  %8.3f  %6.3f%%  %9.3f .. %9.3f  %10.3f\n", r.name.c_str(), s.mean,
                s.sd, s.mean != 0.0 ? 100.0 * s.sd / s.mean : 0.0, s.lo, s.hi,
                s.resolvable_2sigma(sessions));
  }
}

int run_profile(l0::Context& ctx, const loader::LoadedModel& model,
                const std::vector<uint32_t>& ids, uint32_t steps, uint32_t repeats) {
  runtime::DecodeBuffers buffers(ctx, model.max_len, *model.desc);
  // Two lists over ONE set of buffers - the pattern
  // tests/runtime/profile_capture_test.cc proved and the golden gate already
  // used. The plain list does the ingestion (an instrumented one would pay 774
  // host-scope flushes on each of ~4096 tokens for a number nobody reads); the
  // profiled list is the one that is measured. Both bake the same allocations,
  // so whichever is replayed advances the same state.
  //
  // `debug_resid` off on both: the tap is 64 device copies a token and only the
  // golden gate reads it.
  runtime::CapturedStep plain = runtime::build(ctx, model, buffers);
  runtime::ProfileEvents prof(ctx);
  runtime::CapturedStep instr = runtime::build(ctx, model, buffers, nullptr, &prof);
  const size_t n = instr.kernel_count;
  std::fprintf(stderr,
               "engine: %zu kernels, %zu modules, max_len %u, %.2f GB of persistent state"
               ", decode attention %s (profiled list: %zu events)\n",
               n, instr.modules.size(), buffers.max_len, buffers.persistent_bytes() / 1e9,
               runtime::decode_attn_name(runtime::decode_attn()), prof.events.size());

  // One queue for both lists: the ingestion and the measured replays go through
  // the same ordering domain the Engine would use, so nothing about the
  // measurement depends on a second queue landing somewhere else.
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::Control* c = buffers.control.as<runtime::Control>();

  // Engine::reset()'s persistent group, spelled out because this mode does not
  // construct an Engine. Scratch is deliberately not zeroed, for the reason
  // Engine::reset() gives: no step may read scratch it has not first written.
  for (l0::Mem* m : {&buffers.control, &buffers.gdn_state, &buffers.conv_ring, &buffers.kv_k,
                     &buffers.kv_v})
    imm.fill(m->ptr(), 0u, m->size());

  auto replay = [&](runtime::CapturedStep& s) {
    // Engine::replay()'s precondition; here this loop is the caller, so the
    // check is here (the Task-5 ruling recorded in runtime/engine.h).
    if (size_t(c->pos) + size_t(c->n_active) > size_t(buffers.max_len))
      throw std::runtime_error("profile: pos " + std::to_string(c->pos) + " + n_active " +
                               std::to_string(c->n_active) + " exceeds max_len " +
                               std::to_string(buffers.max_len));
    q.execute(s.list, &fence);
    fence.wait();
  };

  const auto t0 = std::chrono::steady_clock::now();
  c->n_active = 1;
  for (uint32_t id : ids) {
    c->cur_token[0] = id;
    replay(plain);
  }
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr,
               "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u - un-instrumented list, so"
               " this half is a normal decode\n",
               ids.size(), ingest_ms, ids.empty() ? 0.0 : ingest_ms / double(ids.size()), c->pos);

  // The measured half. Every event is reset before EVERY replay (re-signalling
  // an un-reset event is undefined) and nothing is queried before the fence
  // (`duration_us()` throws on an unsignalled event, by design).
  //
  // `--repeats R` wraps this in R sessions. Each session first rewinds
  // `Control::pos` to the value the ingest left, so all R sessions profile the
  // SAME launch shape at the SAME positions - the spread across them is drift,
  // not a difference in what was run. Everything the single-session report
  // prints is computed from the grand totals below, so `--repeats 1` (the
  // default) prints the same NUMBERS this mode printed before the flag existed.
  // Not the same TEXT: the header gained a line and two rollup titles were
  // reworded. Numbers unchanged, wording not.
  const uint32_t ingest_pos = c->pos;
  std::vector<double> per_launch(n, 0.0);           // Σ over every replay of every session
  std::vector<std::vector<double>> session_launch;  // [session][launch], Σ over that session
  std::vector<double> session_sum, session_wall;    // us/step, one entry per session
  double sum_all = 0.0, wall_all = 0.0;
  double sum_lo = 0.0, sum_hi = 0.0, wall_lo = 0.0, wall_hi = 0.0;
  bool first_replay = true;
  for (uint32_t r = 0; r < repeats; ++r) {
    // The rewind. `pos` is the only thing that has to move: argmax_stage2
    // advanced it by one per replay, and every launch's grid and every
    // kernel's live-block count is a function of it.
    c->pos = ingest_pos;
    c->n_active = 1;
    std::vector<double> launch_r(n, 0.0);
    double sum_r = 0.0, wall_r = 0.0;
    for (uint32_t s = 0; s < steps; ++s) {
      for (l0::Event& e : prof.events) e.reset();
      const auto s0 = std::chrono::steady_clock::now();
      replay(instr);
      const double wall =
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count();
      double sum = 0.0;
      for (size_t i = 0; i < n; ++i) {
        const double d = prof.events[i].duration_us();
        per_launch[i] += d;
        launch_r[i] += d;
        sum += d;
      }
      if (first_replay) {
        sum_lo = sum_hi = sum;
        wall_lo = wall_hi = wall;
        first_replay = false;
      }
      sum_lo = std::min(sum_lo, sum);
      sum_hi = std::max(sum_hi, sum);
      wall_lo = std::min(wall_lo, wall);
      wall_hi = std::max(wall_hi, wall);
      sum_all += sum;
      wall_all += wall;
      sum_r += sum;
      wall_r += wall;
    }
    session_launch.push_back(std::move(launch_r));
    session_sum.push_back(sum_r / steps);
    session_wall.push_back(wall_r / steps);
    if (repeats > 1)
      std::fprintf(stderr, "session %u/%u: %.3f us Σ, %.3f us wall, pos %u → %u, last id %u\n",
                   r + 1, repeats, session_sum.back(), session_wall.back(), ingest_pos, c->pos,
                   c->out_token[0]);
  }
  const uint32_t replays = steps * repeats;
  const double sum_us = sum_all / replays;    // Σ per-kernel device time, per step
  const double wall_us = wall_all / replays;  // submit + fence, per step

  // --- the report. stdout: it is this mode's answer. ------------------------
  std::printf(
      "# b70-decode --profile - in-situ step anatomy\n"
      "depth %zu, steps %u, repeats %u (%u profiled replays), %zu launches/step, max_len %u,\n"
      "device %s (%u EUs)\n"
      "PROFILE MODE IS NOT BENCH MODE: every launch below signals a host-visible\n"
      "kernel-timestamp event, a per-launch flush an unprofiled list never pays. Per-kernel\n"
      "durations are kernelStart->kernelEnd and EXCLUDE that flush (it lands after kernelEnd),\n"
      "so they are comparable to probe per-kernel numbers; the gap at the bottom INCLUDES it\n"
      "and is an upper bound. The engine's ms/token comes from --bench, never from here.\n"
      "Every `share` below is of the sum of kernel durations, not of the fence wall.\n"
      "%s",
      ids.size(), steps, repeats, replays, n, buffers.max_len, ctx.name().c_str(), ctx.eu_count(),
      repeats > 1 ? "REPEATED-RUN AVERAGING IS ON: the rollups below are the grand mean over every\n"
                    "replay of every session; the repeatability table after them is what prices an\n"
                    "attribution. A session is `steps` replays after a rewind of Control::pos to the\n"
                    "post-ingest value, so every session profiles the identical launch shape.\n"
                  : "");

  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return per_launch[a] > per_launch[b]; });
  const size_t top = std::min<size_t>(30, n);
  std::printf("\ntop %zu launches by mean us/step (measured, mean of %u profiled replays)\n"
              "  rank  launch    us/step   share  label\n",
              top, replays);
  for (size_t r = 0; r < top; ++r) {
    const size_t i = order[r];
    const double us = per_launch[i] / replays;
    std::printf("  %4zu  %6zu  %9.3f  %5.2f%%  %s\n", r + 1, i, us, 100.0 * us / sum_us,
                instr.labels[i].c_str());
  }

  // The three rollups. `family` is the entry point (what docs/12 has a section
  // for); `variant` is the compiled binary (what docs/12's per-shape tables and
  // the probe price); `layer kind` is the GDN / FA / token-boundary split.
  std::vector<model::LayerDesc> layers = model.desc->layer_descs();
  std::vector<Agg> by_family, by_variant, by_kind;
  for (size_t i = 0; i < n; ++i) {
    const Label lab = split_label(instr.labels[i]);
    const double us = per_launch[i];
    accumulate(by_family, lab.entry, us);
    accumulate(by_variant, lab.variant, us);
    if (lab.layer == "--") {
      accumulate(by_kind, "token boundary", us);
    } else {
      const unsigned long idx = std::stoul(lab.layer.substr(1));
      if (idx >= layers.size()) throw std::runtime_error("profile: label names layer " + lab.layer);
      accumulate(by_kind,
                 layers[idx].kind == model::LayerKind::GDN
                     ? "GDN layers (" + std::to_string(model.desc->gdn_layers) + ")"
                     : "FA layers (" + std::to_string(model.desc->fa_layers) + ")",
                 us);
    }
  }
  print_rollup("per-kernel-family rollup (measured, mean of every profiled replay)", "family",
               by_family, replays, sum_us);
  print_rollup("per-variant rollup - the rows docs/12 and probe_gemv price", "variant", by_variant,
               replays, sum_us);
  print_rollup("per-layer-kind rollup", "layer kind", by_kind, replays, sum_us);

  std::printf(
      "\ntotals per step (mean of %u profiled replays; the per-replay spread is in brackets)\n"
      "  sum of %zu kernel durations   %10.3f us   [%.3f .. %.3f]   MEASURED, per-kernel,\n"
      "                                                                   flush-free\n"
      "  fence wall (submit + wait)    %10.3f us   [%.3f .. %.3f]   MEASURED, profiled list\n"
      "  gap = wall - sum              %10.3f us   = %.3f us/launch over %zu launches\n"
      "                                                 UPPER BOUND: includes one host-scope\n"
      "                                                 flush per launch (spec 1.5 §3.3)\n"
      "  sum / wall                    %10.4f\n",
      replays, n, sum_us, sum_lo, sum_hi, wall_us, wall_lo, wall_hi, wall_us - sum_us,
      (wall_us - sum_us) / double(n), n, sum_us / wall_us);
  // --- the repeatability report, and the attribution floor it prices -------
  //
  // This is the memo's §5.4 item. Everything above is a mean; this is what the
  // mean is worth. Each row's series is that family's us/step in each of the R
  // sessions, and `2sigma-resolvable` is the delta a before/after attribution
  // has to beat for this instrument to see it (2*sd*sqrt(2/R), derived).
  if (repeats > 1) {
    std::vector<Series> fam;
    auto series_for = [&fam, repeats](const std::string& name) -> Series& {
      for (Series& x : fam)
        if (x.name == name) return x;
      fam.push_back(Series{name, std::vector<double>(repeats, 0.0)});
      return fam.back();
    };
    for (size_t i = 0; i < n; ++i) {
      Series& row = series_for(split_label(instr.labels[i]).entry);
      for (uint32_t r = 0; r < repeats; ++r) row.us[r] += session_launch[r][i] / steps;
    }
    std::vector<double> gap(repeats, 0.0);
    for (uint32_t r = 0; r < repeats; ++r) gap[r] = session_wall[r] - session_sum[r];
    print_repeat_table("per-family repeatability across the sessions (measured)", "family", fam,
                       repeats);
    const Stat ss = Stat::of(session_sum), sw = Stat::of(session_wall), sg = Stat::of(gap);
    std::printf("\n  %-34s  %13.3f  %8.3f  %6.3f%%  %9.3f .. %9.3f  %10.3f\n"
                "  %-34s  %13.3f  %8.3f  %6.3f%%  %9.3f .. %9.3f  %10.3f\n"
                "  %-34s  %13.3f  %8.3f  %6.3f%%  %9.3f .. %9.3f  %10.3f\n",
                "SUM of kernel durations", ss.mean, ss.sd, 100.0 * ss.sd / ss.mean, ss.lo, ss.hi,
                ss.resolvable_2sigma(repeats), "fence wall (profiled)", sw.mean, sw.sd,
                100.0 * sw.sd / sw.mean, sw.lo, sw.hi, sw.resolvable_2sigma(repeats),
                "gap = wall - SUM (derived)", sg.mean, sg.sd,
                sg.mean != 0.0 ? 100.0 * sg.sd / sg.mean : 0.0, sg.lo, sg.hi,
                sg.resolvable_2sigma(repeats));
    std::printf(
        "\n  The `2sigma-resolvable` column is the ATTRIBUTION FLOOR of this instrument at\n"
        "  --repeats %u: two binaries measured this way have means whose difference carries a\n"
        "  standard error of sd*sqrt(2/R), so a family-level delta below the column is not a\n"
        "  measurement. It is a LOWER bound on the real floor -- these %u sessions are\n"
        "  consecutive inside ONE process, so they see short-timescale drift only, while a\n"
        "  before/after comparison spans two processes and usually a build. The across-process\n"
        "  figure is in docs/15-step-anatomy.md and it is the one to quote.\n",
        repeats, repeats);
  }

  std::printf(
      "\nreading these numbers\n"
      "  - Per-kernel us are in-situ and flush-free: compare them directly with the\n"
      "    probe-transplant floors in docs/12 (`gemv` -> Measured) and with docs/05's table.\n"
      "  - The gap is NOT a dispatch cost this engine pays: subtract the flush\n"
      "    (doc 07 #5's 0.52 us/kernel from probe_replay is the unprofiled estimate).\n"
      "  - The fence wall above is inflated by the same flushes and is NOT ms/token.\n"
      "    The recorded step time is the --bench median in docs/BENCHMARKS.md.\n"
      "  - A mean without a spread is not an attribution: run --repeats R (R >= 5) before\n"
      "    claiming a delta under ~0.3 ms, and read the floor off the table above.\n");
  return 0;
}

int run(int argc, char** argv) {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256, steps = 32, repeats = 1, max_len = 16384;
  // Spec 6 §10: 16384 stays the default here, so bench rows stay comparable; `auto`
  // plans the largest max_len that fits, as b70-serve's default does.
  cli::MaxLenArg max_len_arg{false, max_len};
  size_t mem_reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  uint32_t device = l0::Context::kFromEnv;
  uint32_t pp = 0, pp_chunk = 0;
  bool bench = false, profile = false, have_n = false, prefill = false;
  bool have_depth = false, have_tg = false, have_steps = false, have_repeats = false;
  bool have_pp = false, have_pp_chunk = false;
  std::string pp_backend_arg;
  bool have_pp_backend = false;
  // Spec 9 §3: bf16 by default, so BENCHMARKS rows stay byte-matched with vLLM.
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;

  auto value = [&](int& i, const char* flag) -> std::string {
    if (++i >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else if (a == "--ids") {
      ids_path = value(i, "--ids");
    } else if (a == "--n") {
      n = parse_u32("--n", value(i, "--n"));
      have_n = true;
    } else if (a == "--prefill") {
      prefill = true;
    } else if (a == "--device") {
      device = parse_u32("--device", value(i, "--device"));
      // 0xFFFFFFFF is l0::Context's "ask the environment" sentinel, so it is
      // the one index --device cannot name.
      if (device == l0::Context::kFromEnv)
        throw std::runtime_error("--device is out of range");
    } else if (a == "--max-len") {
      max_len_arg = cli::parse_max_len(value(i, "--max-len"));
      max_len = max_len_arg.value;   // 0 for auto until cli::settle
    } else if (a == "--mem-reserve-gb") {
      mem_reserve = cli::parse_reserve_gb(value(i, "--mem-reserve-gb"));
    } else if (a == "--bench") {
      bench = true;
    } else if (a == "--profile") {
      profile = true;
    } else if (a == "--depth") {
      depth = parse_u32("--depth", value(i, "--depth"));
      have_depth = true;
    } else if (a == "--pp") {
      pp = parse_u32("--pp", value(i, "--pp"));
      have_pp = true;
    } else if (a == "--pp-chunk") {
      pp_chunk = parse_u32("--pp-chunk", value(i, "--pp-chunk"));
      have_pp_chunk = true;
    } else if (a == "--pp-backend") {
      pp_backend_arg = value(i, "--pp-backend");
      have_pp_backend = true;
    } else if (a == "--tg") {
      tg = parse_u32("--tg", value(i, "--tg"));
      have_tg = true;
    } else if (a == "--steps") {
      steps = parse_u32("--steps", value(i, "--steps"));
      have_steps = true;
    } else if (a == "--repeats") {
      repeats = parse_u32("--repeats", value(i, "--repeats"));
      have_repeats = true;
    } else if (a == "--lm-head") {
      const std::string v = value(i, "--lm-head");
      if (!loader::parse_lm_head_form(v, lm_head))
        throw std::runtime_error("--lm-head expects bf16 or int8, got '" + v + "'");
    } else if (!a.empty() && a[0] == '-') {
      usage();
      throw std::runtime_error("unknown option '" + a + "'");
    } else if (path.empty()) {
      path = a;
    } else {
      usage();
      throw std::runtime_error("unexpected extra argument '" + a + "'");
    }
  }

  if (path.empty()) {
    usage();
    throw std::runtime_error("a snapshot directory or HF repo id is required");
  }
  if (prefill && ids_path.empty())
    throw std::runtime_error(
        "--prefill belongs to --ids; --bench uses --pp and --profile has no prompt");
  // Three modes, exactly one of them. `--profile` is exclusive with `--bench`
  // for a reason that is not tidiness: a profiled list signals 774 host-visible
  // events per step, so it can never produce a bench row (spec 1.5 §3.3).
  const int modes = int(!ids_path.empty()) + int(bench) + int(profile);
  if (modes != 1) {
    usage();
    throw std::runtime_error("exactly one of --ids, --bench and --profile");
  }
  const bool synthetic = bench || profile;   // the two modes that cycle the baked prompt
  if (have_n && ids_path.empty())
    throw std::runtime_error("--n belongs to --ids; --bench sizes its run with --depth and --tg,"
                             " --profile with --depth and --steps");
  if (have_tg && !bench)
    throw std::runtime_error("--tg belongs to --bench; --ids sizes its run with --n and --profile"
                             " with --depth and --steps");
  if (have_steps && !profile)
    throw std::runtime_error("--steps belongs to --profile; --bench sizes its run with --depth"
                             " and --tg, --ids with --n");
  if (have_repeats && !profile)
    throw std::runtime_error("--repeats belongs to --profile; it is repeated-run averaging of the"
                             " instrumented replays and nothing else has any");
  if (have_depth && !synthetic)
    throw std::runtime_error("--depth belongs to --bench and --profile; --ids sizes its run"
                             " with --n");
  // The prefill flags (spec 2, interfaces.md's CLI contract). They are pure
  // argument validation and are checked BEFORE l0::Context like every other
  // rejection, so `b70_cli_reject` can grade them without a device - and so a
  // build with -DB70_PREFILL=OFF still refuses them for the right reason
  // rather than as an unknown flag.
  if (have_pp && !bench)
    throw std::runtime_error("--pp belongs to --bench; --ids sizes its run with --n and"
                             " --profile with --depth and --steps");
  if (have_pp && have_depth)
    throw std::runtime_error("--pp and --depth are exclusive: the prefilled ids ARE the depth,"
                             " so naming both would be two answers to one question");
  if (have_pp && pp == 0) throw std::runtime_error("--pp 0 would prefill nothing");
  if (have_pp_chunk && !have_pp && !prefill)
    throw std::runtime_error("--pp-chunk belongs to --pp; it is the prefill chunk width and"
                             " nothing else has one");
  if (have_pp_chunk && pp_chunk == 0)
    throw std::runtime_error("--pp-chunk 0 is not a chunk width; omit it for the default"
                             " PrefillScratch::kC");
  if (have_pp_backend && !have_pp && !prefill)
    throw std::runtime_error("--pp-backend belongs to --pp and --prefill; the decode path has no"
                             " GEMM backend");
  runtime::PrefillBackend pp_backend{};
  if (have_pp_backend && !runtime::parse_prefill_backend(pp_backend_arg, pp_backend))
    throw std::runtime_error("--pp-backend expects sycl-tla, l0 or l0-int8, got '" + pp_backend_arg + "'");
  if (!synthetic && !have_n) {
    usage();
    throw std::runtime_error("--ids needs --n");
  }
  if (!synthetic && n == 0) throw std::runtime_error("--n 0 would generate nothing");
  if (bench && tg == 0) throw std::runtime_error("--tg 0 would time nothing");
  if (profile && steps == 0) throw std::runtime_error("--steps 0 would profile nothing");
  if (profile && repeats == 0) throw std::runtime_error("--repeats 0 would profile nothing");
  // `--pp N` IS the depth (interfaces.md's CLI contract: "--depth is then
  // ignored and the tg row's depth column reads N"). Making it `depth` here
  // rather than threading a second variable means every downstream use - the
  // max_len bound, the id vector, the tg row's depth column - is one number.
  if (have_pp) depth = pp;
  // Both synthetic modes exist to measure a step at a context depth, and the
  // cost of a step depends on that depth (attention's live-block count is the
  // measured example - docs/15). Depth 0 measures a shape nobody runs.
  if (synthetic && depth == 0)
    throw std::runtime_error("--depth 0 would ingest nothing; --bench and --profile both measure"
                             " a step at a context depth");

  // Everything that can be judged without the device or the 19 GB checkpoint is
  // judged first: failing on a typo'd --ids path after a 13-second load is a
  // worse CLI than failing in a millisecond. (With --max-len auto the length is
  // known only after the load and the plan, so the bound waits for it.)
  std::vector<uint32_t> ids;
  // Both synthetic modes ingest to --depth and then replay: --tg generated
  // tokens for --bench, --steps instrumented ones for --profile. Same bound
  // either way - the KV cache and the RoPE table stop at max_len.
  // `--repeats` does NOT enter this bound: every session rewinds `pos` to the
  // post-ingest value, so R sessions reach exactly the same highest position
  // one session does. That is the point of the rewind.
  const uint32_t after = bench ? tg : steps;
  const char* after_flag = bench ? "--tg " : "--steps ";
  const auto check_len = [&](uint32_t len) {
    if (!synthetic && ids.size() + size_t(n) > size_t(len))
      throw std::runtime_error("prompt (" + std::to_string(ids.size()) + " ids) + --n " +
                               std::to_string(n) + " exceeds --max-len " + std::to_string(len));
    if (synthetic && size_t(depth) + size_t(after) > size_t(len))
      throw std::runtime_error("--depth " + std::to_string(depth) + " + " + after_flag +
                               std::to_string(after) + " exceeds --max-len " +
                               std::to_string(len));
  };
  if (!synthetic) ids = read_ids(ids_path);
  // Spec 6 §10: auto plans under the trained context, which bounds the run already.
  uint32_t trained = 0;
  if (max_len_arg.is_auto) {
    trained = loader::trained_context(loader::resolve_snapshot(path));
    cli::check_before_load(max_len_arg, trained, /*require_quantum=*/false);
  }
  check_len(max_len_arg.is_auto ? trained : max_len);
  if (synthetic) {
    ids.resize(depth);
    for (uint32_t i = 0; i < depth; ++i) ids[i] = kBenchPrompt[i % kBenchPromptLen];
  }

  l0::Context ctx(device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", ctx.name().c_str(), ctx.eu_count(),
               device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");

  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(ctx, path, cli::load_len(max_len_arg, trained), /*mtp=*/false, lm_head);
  }();
  // Spec 15c: --pp / --prefill on a model whose prefill is not built (Ornith, spec 15d)
  // stop here with the stage's name, before a plan or an engine is made for it.
  if (have_pp || prefill) model::require_prefill(*model.desc);
  // Spec 6 §10: auto plans and re-tables the model; an explicit length is held to the
  // plan. The prefill scratch is planned only when this run prefills (ruling R7).
  max_len = cli::settle(ctx, model, max_len_arg, mem_reserve,
                        cli::prefill_path(have_pp || prefill,
                                          have_pp_backend ? pp_backend
                                                          : runtime::prefill::default_prefill_backend()));
  if (max_len_arg.is_auto) check_len(max_len);

  // --profile forks here: it replays an instrumented capture and has to reset
  // 774 events before every replay, which is the caller's job by design
  // (runtime/capture.h) - so it runs its own loop rather than an Engine's.
  if (profile) return run_profile(ctx, model, ids, steps, repeats);

  // debug_resid off: the per-layer tap costs 64 device copies a token and only
  // the golden gate (Task 8) reads it.
  runtime::Engine eng(ctx, std::move(model), max_len);
  std::fprintf(stderr,
               "engine: %zu kernels, %zu modules, max_len %u, %.2f GB of persistent state, "
               "decode attention %s\n",
               eng.step().kernel_count, eng.step().modules.size(), eng.max_len(),
               eng.buffers().persistent_bytes() / 1e9,
               runtime::decode_attn_name(runtime::decode_attn()));

  // Ingestion is one replay per prompt token unless `--pp` or `--prefill` is
  // given, in which case it is `Engine::prefill`. The measured window is
  // **the whole call**, and that needs no extra instrumentation to be the
  // interface's "first prefill launch to the first generated id in cur_token":
  // `prefill()` returns only after its final `Context::wait()`, and its last
  // launch is `argmax_stage2`, which is the only writer of `cur_token`.
  // The prefill setup (scratch, Context, and on l0-int8 the one-time rotated
  // column-scale pass, spec 5 T2) happens here, at load, outside the window.
  size_t pp_launches_setup = 0;
  if (have_pp || prefill) {
    if (have_pp_backend) eng.set_prefill_backend(pp_backend);
    std::fprintf(stderr, "prefill backend: %s (SYCL component %s)\n",
                 runtime::prefill_backend_name(eng.prefill_backend()),
                 runtime::prefill::sycl_available() ? "on" : "off");
    eng.prepare_prefill();
    pp_launches_setup = eng.prefill_launches();
  }
  // Spec 6: every device byte this engine holds, before the first prefill.
  std::fprintf(stderr, "%s\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  size_t pp_launches = 0;
  if (have_pp || prefill) {
    if (have_pp) runtime::prefill::profile_reset();
    eng.prefill(ids, pp_chunk);
    if (have_pp) pp_launches = eng.prefill_launches() - pp_launches_setup;
  } else {
    eng.ingest(ids);
  }
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (have_pp)
    std::fprintf(stderr,
                 "pp: %zu ids in %.1f ms (%.2f t/s) -- device-side, loader excluded, first"
                 " prefill launch to the first generated id in cur_token; chunk %u, %zu L0"
                 " launches, pos %u\n",
                 ids.size(), ingest_ms,
                 ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0,
                 pp_chunk ? pp_chunk : runtime::PrefillScratch::kC, pp_launches, eng.pos());
  if (have_pp) runtime::prefill::profile_report("--pp", ingest_ms);
  if (!have_pp && !prefill)
    std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u\n", ids.size(),
                 ingest_ms, ids.empty() ? 0.0 : ingest_ms / double(ids.size()), eng.pos());
  if (prefill)
    std::fprintf(stderr, "prefill: %zu ids, pos %u\n", ids.size(), eng.pos());

  if (!bench) {
    eng.generate(n, [](uint32_t id) {
      std::printf("%u\n", id);
      std::fflush(stdout);   // one id per line, as it is sampled
    });
    report_generate(eng, n);
    return 0;
  }

  // --bench. One shape, one row, and the decomposition that makes the row
  // actionable. Nothing here is tuned or retried: the number printed is the
  // number the run produced. tools/bench_decode.sh is what runs it three times
  // and takes the median; docs/BENCHMARKS.md records that median.
  eng.generate(tg);
  report_generate(eng, tg);

  const double ms_per_token = eng.last_gen_ms() / double(tg);
  const double host_ms = eng.last_gen_ms() - eng.last_fence_ms();
  std::fprintf(stderr,
               "  wall %.1f ms, fence %.1f ms, host %.1f ms  (host = wall - fence: argument-free\n"
               "  replay, so this is submit + the four bytes of shared memory, nothing else)\n"
               "  per token: %.3f ms total = %.3f ms fence + %.3f ms host\n",
               eng.last_gen_ms(), eng.last_fence_ms(), host_ms, ms_per_token,
               eng.last_fence_ms() / double(tg), host_ms / double(tg));

  // MBU against the two measured constants, spelled out so the reader can
  // check the arithmetic without leaving the terminal. The kernel-count line
  // is an *estimate* carried from doc 07 #5 (0.52 µs/kernel, probe_replay) -
  // it is what the captured list costs in dispatch alone, and it is printed
  // beside the fence time to show how little of the fence it can explain.
  const double bytes_per_token = double(eng.model().report.read_per_token) / 1e9;
  const double achieved_gbs = eng.last_tok_per_s() * bytes_per_token;
  const double dispatch_ms = double(eng.step().kernel_count) * 0.52e-3;
  std::fprintf(stderr,
               "  MBU: %.2f t/s x %.3f GB = %.0f GB/s of %.0f GB/s measured = %.1f%%\n"
               "       (W is this checkpoint's own read/token, from the loader report above;\n"
               "        roofline at %.0f GB/s = %.2f t/s = %.3f ms/token)\n"
               "  dispatch floor (estimated): %zu kernels x 0.52 us = %.3f ms/token,"
               " %.1f%% of the fence\n",
               eng.last_tok_per_s(), bytes_per_token, achieved_gbs, kDeviceGBs,
               100.0 * achieved_gbs / kDeviceGBs, kDeviceGBs, kDeviceGBs / bytes_per_token,
               1000.0 * bytes_per_token / kDeviceGBs, eng.step().kernel_count, dispatch_ms,
               100.0 * dispatch_ms / (eng.last_fence_ms() / double(tg)));

  // The row's identity comes from the environment, not from a configure-time
  // git call: tools/box.sh syncs the tree WITHOUT .git, so the box cannot know
  // its own sha. Whoever runs the bench exports it (tools/bench_decode.sh
  // does), and an unset variable prints as `unknown` rather than as a wrong
  // sha in a recorded measurement.
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  // Spec 9: an int8-head row names its head; a bf16 row is byte-identical to before.
  const char* head_tag = lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "";
  std::printf("| b70-decode %s%s | %u | %u | %.2f | %.2f |\n", sha, head_tag, depth, tg,
              eng.last_tok_per_s(), ms_per_token);
  // The tg row above is byte-identical to what it always was, and the pp row
  // is a SECOND line rather than extra columns on it, so every existing parser
  // of docs/BENCHMARKS.md's table keeps working (interfaces.md's CLI contract;
  // tools/bench_decode.sh --pp reads both).
  if (have_pp)
    std::printf("| b70-decode %s%s %s pp | %u | %u | %.1f | %.2f |\n", sha, head_tag,
                runtime::prefill_backend_name(eng.prefill_backend()), depth,
                pp_chunk ? pp_chunk : runtime::PrefillScratch::kC, ingest_ms,
                ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  std::fprintf(stderr,
               "(measured, one run. docs/BENCHMARKS.md records the median of three on an idle box;\n"
               " tools/bench_decode.sh is that harness. B70_GIT_SHA unset prints `unknown`.)\n");
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "b70-decode: %s\n", e.what());
    return 1;
  }
}
