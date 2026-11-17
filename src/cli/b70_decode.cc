// b70-decode - spec §12, plus spec 1.5 §3.4. Three modes:
//
//   b70-decode <snapshot-or-repo> --ids <file> --n <N> [--prefill [--prefill-chunk C]]
//                                         [--max-len 16384] [--spec mtp|dflash]
//   b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256]
//   b70-decode <snapshot-or-repo> --profile [--depth 4096] [--steps 32] [--repeats 1]
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
#include <print>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cli/dflash_args.h"   // spec 19c: --spec dflash
#include "cli/flags.h"
#include "cli/hardware_gate.h"
#include "cli/stdout_to_stderr.h"
#include "cli/kolibri_decode.h"
#include "cli/qwen4exp_decode.h"
#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_decode.h"
#include "cli/tp_decode.h"
#include "cli/spec_args.h"
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
#include "server/adaptive_k.h"

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

// The flags (cli/flags.h: each member is its flag, its default and its --help text).
using cli::flags::Arg;
using cli::flags::Help;
using cli::flags::Section;
struct Flags {
  [[=Help("<file>", "whitespace-separated token ids")]] std::string ids;
  [[=Help("<N>", "ids to generate, greedily; t/s on stderr")]] Arg<uint32_t> n;
  [[=Help("", "run the prompt through the engine's prefill instead of one decode replay per id")]] bool prefill = false;
  [[=Help("S", "--ids only: off, mtp (the MTP head, adaptive K <= --spec-max 1..3;\n"
               "Qwen3.8, Ornith) or dflash (Qwen3.8's DFlash drafter, K <= --spec-max\n"
               "1..7, one card; --draft-model PATH, required). Greedy: the ids equal the plain run's.")]]
  Arg<std::string> spec{"off"};   // a developer tool: the plain list unless asked (cli/spec_args.h)
  [[=Help("K", "the proposer's cap (see --spec).", "the most")]] Arg<uint32_t> spec_max;
  [[=Help("PATH", "--spec dflash's drafter.")]] Arg<std::string> draft_model;   // spec 19c
  // Spec 6 §10: 16384 stays the default here, so bench rows stay comparable; `auto` plans the largest max_len that
  // fits, as b70-serve's default does.
  [[=Help("L", "KV cache and RoPE capacity (16384 keeps bench rows comparable); any\n"
               "multiple of 256 up to the trained context, refused if it does not fit; auto =\n"
               "the largest that fits.")]]
  std::string max_len = "16384";
  [[=Help("P", "auto: one card when the model and the context fit it, else pp; 1;\n"
               "pp: the layers over two cards (--prefill runs the two-card chunk\n"
               "pipeline). Kolibri-1 defaults to two cards.\n"
               "tp (Qwen3.8): tensor parallel, every layer split over two cards\n"
               "(--prefill and --spec mtp / auto run on both cards).")]]
  std::string parallel = "auto";
  [[=Help("auto|N", "pp: device 0 runs layers [0, N); an override, it implies pp.")]] Arg<std::string> pipeline_split{"auto"};
  [[=Help("N|auto", "Kolibri-1 / Qwen3.8-Flash-Next: load only the first N layers\n"
                    "(development mode; Flash-Next also takes auto). Flash-Next without it loads whole: the\n"
                    "experts that do not fit the cards are read from pinned host memory.")]]
  Arg<std::string> layers;   // spec 20c, Kolibri-1; spec 21-QFN-v1c, Qwen3.8-Flash-Next
  [[=Section("\ndeveloper (benchmarks and profiles):\n"),
    =Help("", "ingest --depth synthetic ids, then time --tg generated ones; a markdown row\non stdout.")]]
  bool bench = false;
  [[=Help("D", "--bench / --profile: synthetic ids of context before decoding.")]] Arg<uint32_t> depth{4096};
  [[=Help("N", "--bench: ids to generate and time.")]] Arg<uint32_t> tg{256};
  [[=Help("N", "--bench only, exclusive with --depth: prefill N synthetic ids and print\n"
               "a SECOND row with the device-side prefill time.")]]
  Arg<uint32_t> prefill_length;
  [[=Help("C", "--prefill-length / --prefill: positions per chunk, at most 4096 (the\n"
               "session's chunk cap; above 2048 the buffers hold 4096-row chunks, planned).", "2048")]]
  Arg<uint32_t> prefill_chunk;
  [[=Help("", "ingest --depth synthetic ids, then replay --steps INSTRUMENTED steps and print\n"
              "the per-launch anatomy (never a bench row).")]]
  bool profile = false;
  [[=Help("N", "--profile: instrumented steps.")]] Arg<uint32_t> steps{32};
  [[=Help("R", "--profile: R sessions over one ingest, a per-family mean +/- spread table.")]] Arg<uint32_t> repeats{1};
  [[=Section("\nstill being measured (they go when their numbers decide, docs/open-work.md):\n"),
    =Help("bf16|int8", "the KV cache's form; int8 halves it.", "bf16")]]
  std::string kv_cache;   // spec 12b; empty: runtime::default_kv_cache()
  [[=Help("l0|l0-int8", "--prefill-length / --prefill: bf16, or int8 on the matrix engines.", "l0")]]
  Arg<std::string> prefill_backend;
};

void usage() {
  std::print(
      stderr,
      "usage: b70-decode <snapshot-or-repo> --ids <file> --n <N> [--prefill] [--spec mtp|dflash]\n"
      "                                     [--max-len 16384|auto] [--parallel auto|1|pp|tp]\n"
      "\n"
      "  ids in, ids out (no tokenizer: tools/oracle/tokenize.py turns text into an --ids file);\n"
      "  stdout is one generated id per line, everything else goes to stderr. The model is chosen\n"
      "  from the checkpoint's config.json; the lm_head is int8 and the device memory the plan leaves\n"
      "  free is the engine's, as b70-serve. Cards: Intel Arc Pro B70, one or two; ZE_AFFINITY_MASK\n"
      "  picks which the driver shows.\n"
      "\n"
      "  <snapshot-or-repo>  a snapshot directory, or an HF repo id resolved against the local\n"
      "                 cache ($HF_HOME or ~/.cache/huggingface). Never downloads.\n"
      "{}"
      "  B70_PREFILL_PROFILE=1   per-phase prefill timings on stderr (diagnostic)\n",
      cli::flags::help<Flags>(17));
}

using cli::StdoutToStderr;

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  std::string w;
  while (f >> w) {
    const uint32_t id = cli::flags::parse_u32("--ids file '" + path + "'", w);
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
  std::print(stderr, "generate: {:d} ids, {:.2f} t/s, {:.2f} ms/token ({:.1f}% of it inside the fence)\n",
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
  std::print("\n{}\n  {:<34}  launches    us/step   share   us/launch\n", title, what);
  for (const Agg& a : rows) {
    const double us = a.us / steps;
    std::print("  {:<34}  {:8d}  {:9.3f}  {:5.2f}%  {:10.3f}\n", a.name.c_str(), a.launches, us,
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
// (spec 1.5's reassessment, its section 5.4).
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
  std::print("\n{}\n  {:<34}   mean us/step        sd     sd%        min .. max"
             "    2sigma-resolvable\n",
             title, what);
  for (const Series& r : rows) {
    const Stat s = Stat::of(r.us);
    std::print("  {:<34}  {:13.3f}  {:8.3f}  {:6.3f}%  {:9.3f} .. {:9.3f}  {:10.3f}\n", r.name.c_str(), s.mean,
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
  std::print(stderr,
             "engine: {:d} kernels, {:d} modules, max_len {:d}, {:.2f} GB of persistent state"
             ", decode attention {} (profiled list: {:d} events)\n",
             n, instr.modules.size(), buffers.max_len, buffers.persistent_bytes() / 1e9,
             "v2", prof.events.size());

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
  std::print(stderr,
             "ingest: {:d} ids in {:.1f} ms ({:.2f} ms/token), pos {:d} - un-instrumented list, so"
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
      std::print(stderr, "session {:d}/{:d}: {:.3f} us Σ, {:.3f} us wall, pos {:d} → {:d}, last id {:d}\n",
                 r + 1, repeats, session_sum.back(), session_wall.back(), ingest_pos, c->pos,
                 c->out_token[0]);
  }
  const uint32_t replays = steps * repeats;
  const double sum_us = sum_all / replays;    // Σ per-kernel device time, per step
  const double wall_us = wall_all / replays;  // submit + fence, per step

  // --- the report. stdout: it is this mode's answer. ------------------------
  std::print(
      "# b70-decode --profile - in-situ step anatomy\n"
      "depth {:d}, steps {:d}, repeats {:d} ({:d} profiled replays), {:d} launches/step, max_len {:d},\n"
      "device {} ({:d} EUs)\n"
      "PROFILE MODE IS NOT BENCH MODE: every launch below signals a host-visible\n"
      "kernel-timestamp event, a per-launch flush an unprofiled list never pays. Per-kernel\n"
      "durations are kernelStart->kernelEnd and EXCLUDE that flush (it lands after kernelEnd),\n"
      "so they are comparable to probe per-kernel numbers; the gap at the bottom INCLUDES it\n"
      "and is an upper bound. The engine's ms/token comes from --bench, never from here.\n"
      "Every `share` below is of the sum of kernel durations, not of the fence wall.\n"
      "{}",
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
  std::print("\ntop {:d} launches by mean us/step (measured, mean of {:d} profiled replays)\n"
             "  rank  launch    us/step   share  label\n",
             top, replays);
  for (size_t r = 0; r < top; ++r) {
    const size_t i = order[r];
    const double us = per_launch[i] / replays;
    std::print("  {:4d}  {:6d}  {:9.3f}  {:5.2f}%  {}\n", r + 1, i, us, 100.0 * us / sum_us,
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

  std::print(
      "\ntotals per step (mean of {:d} profiled replays; the per-replay spread is in brackets)\n"
      "  sum of {:d} kernel durations   {:10.3f} us   [{:.3f} .. {:.3f}]   MEASURED, per-kernel,\n"
      "                                                                   flush-free\n"
      "  fence wall (submit + wait)    {:10.3f} us   [{:.3f} .. {:.3f}]   MEASURED, profiled list\n"
      "  gap = wall - sum              {:10.3f} us   = {:.3f} us/launch over {:d} launches\n"
      "                                                 UPPER BOUND: includes one host-scope\n"
      "                                                 flush per launch\n"
      "  sum / wall                    {:10.4f}\n",
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
    std::print("\n  {:<34}  {:13.3f}  {:8.3f}  {:6.3f}%  {:9.3f} .. {:9.3f}  {:10.3f}\n"
               "  {:<34}  {:13.3f}  {:8.3f}  {:6.3f}%  {:9.3f} .. {:9.3f}  {:10.3f}\n"
               "  {:<34}  {:13.3f}  {:8.3f}  {:6.3f}%  {:9.3f} .. {:9.3f}  {:10.3f}\n",
               "SUM of kernel durations", ss.mean, ss.sd, 100.0 * ss.sd / ss.mean, ss.lo, ss.hi,
               ss.resolvable_2sigma(repeats), "fence wall (profiled)", sw.mean, sw.sd,
               100.0 * sw.sd / sw.mean, sw.lo, sw.hi, sw.resolvable_2sigma(repeats),
               "gap = wall - SUM (derived)", sg.mean, sg.sd,
               sg.mean != 0.0 ? 100.0 * sg.sd / sg.mean : 0.0, sg.lo, sg.hi,
               sg.resolvable_2sigma(repeats));
    std::print(
        "\n  The `2sigma-resolvable` column is the ATTRIBUTION FLOOR of this instrument at\n"
        "  --repeats {:d}: two binaries measured this way have means whose difference carries a\n"
        "  standard error of sd*sqrt(2/R), so a family-level delta below the column is not a\n"
        "  measurement. It is a LOWER bound on the real floor -- these {:d} sessions are\n"
        "  consecutive inside ONE process, so they see short-timescale drift only, while a\n"
        "  before/after comparison spans two processes and usually a build. The across-process\n"
        "  figure is in docs/15-step-anatomy.md and it is the one to quote.\n",
        repeats, repeats);
  }

  std::print(
      "\nreading these numbers\n"
      "  - Per-kernel us are in-situ and flush-free: compare them directly with the\n"
      "    probe-transplant floors in docs/12 (`gemv` -> Measured) and with docs/05's table.\n"
      "  - The gap is NOT a dispatch cost this engine pays: subtract the flush\n"
      "    (probe_replay's 0.52 us/kernel is the unprofiled estimate; docs/15 \"The dispatch gap\").\n"
      "  - The fence wall above is inflated by the same flushes and is NOT ms/token.\n"
      "    The recorded step time is the --bench median in docs/BENCHMARKS.md.\n"
      "  - A mean without a spread is not an attribution: run --repeats R (R >= 5) before\n"
      "    claiming a delta under ~0.3 ms, and read the floor off the table above.\n");
  return 0;
}

// --- --spec mtp: greedy speculative decoding in --ids mode (spec 8) ------------------
//
// What b70-serve's loop does for a greedy request, without the server: per iteration, K
// drafts from the head (fixed, or AdaptiveK's choice under `auto` - one policy for the run,
// fed by its own acceptance, as one request is), the verify list at M = K + 1, the longest
// prefix of drafts the main model's argmax agrees with, then commit. The pending id and the
// accepted drafts are this iteration's ids, printed one per line as the plain loop does;
// greedy acceptance keeps them equal to the plain run's ids, which is what the box's
// long-context checks diff (queue row 6). The last positions of max_len fall back to K = 0
// (max_verify_k), and ids past --n are not printed.
//
// Spec 16d: the same loop over runtime::PipelineEngine (`--parallel pp --spec mtp`): the engines share the
// draft / verify / commit contract; the pending id is read through pending_of.
uint32_t pending_of(runtime::Engine& e) {
  return e.buffers().control.as<runtime::Control>()->cur_token[0];
}
uint32_t pending_of(runtime::PipelineEngine& e) { return e.pending(); }
uint32_t pending_of(runtime::TpEngine& e) { return e.pending(); }   // spec 17d

template <class Eng>
int generate_mtp(Eng& eng, uint32_t n, bool mtp_auto, uint32_t mtp_k, loader::LmHeadForm lm_head, bool tp = false) {
  std::optional<server::AdaptiveK> policy;
  if (mtp_auto) {
    server::AdaptiveKOptions o;
    o.max_k = mtp_k;   // --spec-max (default 3, runtime::Engine::kMaxDraft)
    // Spec 17d: tensor parallel's own table (its verify rows cross at M rows, its draft reads half the head).
    o.cost = tp ? server::MtpCost::tp_int8_head()
                : lm_head == loader::LmHeadForm::Int8 ? server::MtpCost::int8_head() : server::MtpCost::bf16_head();
    policy.emplace(o);
  }
  uint32_t printed = 0;
  uint64_t iters = 0, drafted = 0, accepted = 0, at_k[runtime::Engine::kMaxDraft + 1] = {};
  const auto emit = [&](uint32_t id) {
    if (printed == n) return;
    std::print("{:d}\n", id);
    std::fflush(stdout);
    ++printed;
  };
  const auto t0 = std::chrono::steady_clock::now();
  while (printed < n) {
    const uint32_t want = policy ? policy->next() : mtp_k;
    const uint32_t k = std::min(want, eng.max_verify_k());
    const uint32_t x = pending_of(eng);
    uint32_t j = 0;
    if (k == 0) {
      eng.verify(0);
      eng.commit(0, eng.verify_ids()[0]);
      emit(x);
    } else {
      eng.draft(k);
      eng.verify(k);
      while (j < k && eng.draft_ids()[j] == eng.verify_ids()[j]) ++j;
      emit(x);
      for (uint32_t i = 0; i < j; ++i) emit(eng.draft_ids()[i]);
      eng.commit(j, eng.verify_ids()[j]);
    }
    if (policy) policy->observe(want, j);
    ++iters;
    ++at_k[k];
    drafted += k;
    accepted += j;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::print(stderr, "generate: {:d} ids, {:.2f} t/s, {:.2f} ms/token (--spec mtp {}, greedy)\n", n,
             ms > 0.0 ? 1000.0 * n / ms : 0.0, n ? ms / n : 0.0,
             mtp_auto ? "auto" : std::to_string(mtp_k).c_str());
  std::print(stderr,
             "mtp: {:d} iterations (K=0 {:d}, K=1 {:d}, K=2 {:d}, K=3 {:d}), acceptance {:.4f}"
             " ({:d}/{:d}), {:.3f} ids per iteration, pos {:d}\n",
             (unsigned long long)iters, (unsigned long long)at_k[0], (unsigned long long)at_k[1],
             (unsigned long long)at_k[2], (unsigned long long)at_k[3],
             drafted ? double(accepted) / double(drafted) : 0.0, (unsigned long long)accepted,
             (unsigned long long)drafted, iters ? double(n) / double(iters) : 0.0, eng.pos());
  return 0;
}

// --- --spec dflash: greedy speculative decoding with a DFlash drafter in --ids mode (spec 19c) ----------------
//
// generate_mtp's loop with the engine's drafter as the proposer at a fixed K (--spec-max; the last positions of
// max_len fall back through max_verify_k): the drafter's draft list -> verify(K) -> the longest prefix the target's
// argmax agrees with -> commit (which projects the committed rows into the drafter's ring). Greedy, so the ids are
// the plain run's (D2); stderr adds the iterations and the acceptance.
int generate_dflash(runtime::Engine& eng, uint32_t n, uint32_t max_k) {
  runtime::TapDrafter& d = *eng.tap_drafter();
  uint32_t printed = 0;
  uint64_t iters = 0, drafted = 0, accepted = 0, at_k[8] = {};
  const auto emit = [&](uint32_t id) {
    if (printed == n) return;
    std::print("{:d}\n", id);
    std::fflush(stdout);
    ++printed;
  };
  const auto t0 = std::chrono::steady_clock::now();
  while (printed < n) {
    const uint32_t k = std::min(max_k, eng.max_verify_k());
    const uint32_t x = pending_of(eng);
    uint32_t j = 0;
    if (k == 0) {
      eng.verify(0);
      eng.commit(0, eng.verify_ids()[0]);
      emit(x);
    } else {
      const runtime::dflash::DFlashDraftOut& o = d.draft(eng, k, 0.0f, 0);
      const std::vector<uint32_t> drafts(o.ids, o.ids + k);
      eng.verify(k);
      while (j < k && drafts[j] == eng.verify_ids()[j]) ++j;
      emit(x);
      for (uint32_t i = 0; i < j; ++i) emit(drafts[i]);
      eng.commit(j, eng.verify_ids()[j]);
    }
    ++iters;
    ++at_k[k];
    drafted += k;
    accepted += j;
  }
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::print(stderr, "generate: {:d} ids, {:.2f} t/s, {:.2f} ms/token (--spec dflash, K {:d}, greedy)\n", n,
             ms > 0.0 ? 1000.0 * n / ms : 0.0, n ? ms / n : 0.0, max_k);
  std::print(stderr, "dflash: {:d} iterations (K=0..7:", (unsigned long long)iters);
  for (uint64_t c : at_k) std::print(stderr, " {:d}", (unsigned long long)c);
  std::print(stderr, "), acceptance {:.4f} ({:d}/{:d}), {:.3f} ids per iteration, pos {:d}\n",
             drafted ? double(accepted) / double(drafted) : 0.0, (unsigned long long)accepted,
             (unsigned long long)drafted, iters ? double(n) / double(iters) : 0.0, eng.pos());
  return 0;
}

int run(int argc, char** argv) {
  std::string path;
  Flags f;
  if (!cli::flags::parse(argc, argv, f, path, usage)) {
    usage();
    return 0;
  }
  const std::string ids_path = f.ids;
  uint32_t n = f.n.value, depth = f.depth.value, tg = f.tg.value, steps = f.steps.value,
           repeats = f.repeats.value;
  cli::MaxLenArg max_len_arg = cli::parse_max_len(f.max_len);
  uint32_t max_len = max_len_arg.value;   // 0 for auto until cli::settle
  // The zero-config cleanup: the engine decides the device memory the plan leaves free
  // (runtime::kDefaultReserveGb), the lm_head (int8, as b70-serve: spec 9 - loader::LmHeadForm stays the tests'
  // seam for the checkpoint's bf16 head) and the card (card 0 of what Level Zero shows; two cards: 0 and 1).
  const size_t mem_reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const loader::LmHeadForm lm_head = loader::LmHeadForm::Int8;
  const uint32_t prefill_len = f.prefill_length.value, pp_chunk = f.prefill_chunk.value;
  const bool bench = f.bench, profile = f.profile, have_n = f.n.given, prefill = f.prefill;
  const bool have_depth = f.depth.given, have_tg = f.tg.given, have_steps = f.steps.given,
             have_repeats = f.repeats.given;
  const bool have_prefill_len = f.prefill_length.given, have_pp_chunk = f.prefill_chunk.given;
  const std::string pp_backend_arg = f.prefill_backend.value;
  const bool have_pp_backend = f.prefill_backend.given;
  const std::string spec_arg = cli::parse_spec(f.spec.value);
  const bool have_spec = f.spec.given;
  const uint32_t spec_max = f.spec_max.value;
  const bool have_spec_max = f.spec_max.given;
  cli::dflash::Args dflash;
  dflash.model = f.draft_model.value;
  dflash.have_model = f.draft_model.given;
  runtime::KvCache kv_cache = f.kv_cache.empty() ? runtime::default_kv_cache() : cli::parse_kv_cache_arg(f.kv_cache);
  cli::PipelineArgs pipe;
  cli::parse_parallel(f.parallel, pipe);
  if (f.pipeline_split.given) cli::parse_pipeline_split(f.pipeline_split.value, pipe);
  const bool have_layers = f.layers.given;
  const bool layers_auto = have_layers && f.layers.value == "auto";   // spec 21-QFN-v1c: Qwen3.8-Flash-Next's planner N
  const uint32_t layers = have_layers && !layers_auto ? cli::flags::parse_u32("--layers", f.layers.value) : 0;
  if (have_layers && !layers_auto && layers == 0) throw std::runtime_error("--layers 0 would load no layer");

  if (prefill && ids_path.empty())
    throw std::runtime_error(
        "--prefill belongs to --ids; --bench uses --prefill-length and --profile has no prompt");
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
  if (have_spec && spec_arg != "off" && synthetic)
    throw std::runtime_error("--spec belongs to --ids (greedy speculative decoding); --bench and"
                             " --profile time the plain list (mtp_gpu_test --bench times MTP)");
  if (have_spec_max && !have_spec) throw std::runtime_error("--spec-max caps a proposer's drafts: name it with --spec");
  cli::refuse_replica_in_decode(pipe);   // plan 13b: two requests at once is b70-serve's
  cli::resolve_overrides(pipe);   // --pipeline-split under --parallel auto: pp
  const bool pp_given = pipe.given();
  // The model family decides which proposers exist (cli/spec_args.h); --spec lookup is b70-serve's (it verifies
  // a request's own repeats), so b70-decode has mtp and dflash.
  if (spec_arg == "lookup" || spec_arg == "dflash-lookup")
    throw std::runtime_error("--spec " + spec_arg + " is b70-serve's (prompt lookup over a request's own ids); "
                             "b70-decode --ids runs --spec mtp or --spec dflash");
  const bool fam_kol = cli::kolibri::is_kolibri(path);
  const bool fam_q4 = !fam_kol && cli::qwen4exp::is_qwen4exp(path);
  const std::string model_name = fam_kol ? "Kolibri-1" : fam_q4 ? "Qwen3.8-Flash-Next" : "this model";
  const cli::Spec spec =
      cli::resolve_spec(spec_arg, fam_q4 ? cli::SpecFamily::Qwen4Exp : fam_kol ? cli::SpecFamily::None : cli::SpecFamily::Qwen,
                        model_name, have_spec_max, spec_max, dflash.have_model);
  const bool mtp_on = spec.kind == cli::SpecKind::Mtp;
  const bool mtp_auto = mtp_on && spec.adaptive;
  const uint32_t mtp_k = mtp_on ? spec.max_k : 0;
  // Spec 17b: --parallel tp's refusals, before any device (cli/pipeline_args.h check_tp).
  // Spec 17d: --spec mtp / auto run under tensor parallel (the head whole on each card); DFlash only in b70-serve
  // (spec 19 v3c).
  cli::check_tp(pipe, {spec.kind == cli::SpecKind::DFlash || spec.kind == cli::SpecKind::DFlashLookup, profile,
                       fam_kol || fam_q4});
  // The prefill flags (spec 2, interfaces.md's CLI contract). They are pure
  // argument validation and are checked BEFORE l0::Context like every other
  // rejection, so `b70_cli_reject` can grade them without a device - and so a
  // build with -DB70_PREFILL=OFF still refuses them for the right reason
  // rather than as an unknown flag.
  if (have_prefill_len && !bench)
    throw std::runtime_error("--prefill-length belongs to --bench; --ids sizes its run with --n and"
                             " --profile with --depth and --steps");
  if (have_prefill_len && have_depth)
    throw std::runtime_error("--prefill-length and --depth are exclusive: the prefilled ids ARE the"
                             " depth, so naming both would be two answers to one question");
  if (have_prefill_len && prefill_len == 0)
    throw std::runtime_error("--prefill-length 0 would prefill nothing");
  if (have_pp_chunk && !have_prefill_len && !prefill)
    throw std::runtime_error("--prefill-chunk belongs to --prefill-length and --prefill; it is the"
                             " prefill chunk width and nothing else has one");
  if (have_pp_chunk && pp_chunk == 0)
    throw std::runtime_error("--prefill-chunk 0 is not a chunk width; omit it for the default"
                             " (the session's chunk cap)");
  if (have_pp_chunk && pp_chunk > runtime::PrefillScratchDims::kCMax)
    throw std::runtime_error("--prefill-chunk " + std::to_string(pp_chunk) + " exceeds the widest chunk, " +
                             std::to_string(runtime::PrefillScratchDims::kCMax));
  if (have_pp_backend && !have_prefill_len && !prefill)
    throw std::runtime_error("--prefill-backend belongs to --prefill-length and --prefill; the"
                             " decode path has no GEMM backend");
  runtime::PrefillBackend pp_backend{};
  if (have_pp_backend && !runtime::parse_prefill_backend(pp_backend_arg, pp_backend))
    throw std::runtime_error("--prefill-backend expects l0 or l0-int8, got '" +
                             pp_backend_arg + "'");
  if (!synthetic && !have_n) {
    usage();
    throw std::runtime_error("--ids needs --n");
  }
  if (!synthetic && n == 0) throw std::runtime_error("--n 0 would generate nothing");
  if (bench && tg == 0) throw std::runtime_error("--tg 0 would time nothing");
  if (profile && steps == 0) throw std::runtime_error("--steps 0 would profile nothing");
  if (profile && repeats == 0) throw std::runtime_error("--repeats 0 would profile nothing");
  // `--prefill-length N` IS the depth (interfaces.md's CLI contract: "--depth is then
  // ignored and the tg row's depth column reads N"). Making it `depth` here
  // rather than threading a second variable means every downstream use - the
  // max_len bound, the id vector, the tg row's depth column - is one number.
  if (have_prefill_len) depth = prefill_len;
  // Both synthetic modes exist to measure a step at a context depth, and the
  // cost of a step depends on that depth (attention's live-block count is the
  // measured example - docs/15). Depth 0 measures a shape nobody runs.
  if (synthetic && depth == 0)
    throw std::runtime_error("--depth 0 would ingest nothing; --bench and --profile both measure"
                             " a step at a context depth");
  // The hardware gate (cli/hardware_gate.h): one or two Intel Arc Pro B70s, peer access between two - before
  // any model file is read. Its card memory is what the fit checks before a device opens plan against.
  const cli::Hardware hw = cli::hardware_gate();
  // Spec 19c: --spec dflash is the Qwen engine's (--ids, one card).
  dflash.on = spec.kind == cli::SpecKind::DFlash;
  if (dflash.on) dflash.max_k = spec.max_k;
  cli::dflash::check_flags(dflash);
  if (dflash.on) cli::dflash::check_engine(dflash, pipe.on(), kv_cache);
  // Spec 20c: Kolibri-1 (model_type kolibri1) runs runtime::kolibri::KolibriEngine on one or two
  // cards - dispatched here, before the Qwen engine's pipeline and kv-cache rules (its own refusals,
  // cli/kolibri_decode.h, replace them: --parallel defaults to 2 there). A path that does not resolve is
  // not Kolibri - the flows below report it, so every rejection keeps its order.
  if (cli::kolibri::is_kolibri(path)) {
    if (dflash.on)
      throw std::runtime_error("--spec dflash is built for Qwen3.8 beside its drafter; Kolibri-1 has none");
    if (layers_auto) throw std::runtime_error("--layers auto is Qwen3.8-Flash-Next's; Kolibri-1 takes --layers N");
    cli::kolibri::DecodeArgs ka;
    ka.path = path;
    ka.ids_path = ids_path;
    ka.n = n;
    ka.bench = bench;
    ka.depth = depth;
    ka.tg = tg;
    ka.max_len = max_len_arg;
    ka.reserve = mem_reserve;
    ka.card_bytes = hw.card_bytes;
    ka.lm_head = lm_head;
    ka.layers = layers;
    ka.pipe = pipe;
    ka.pp_given = pp_given;
    ka.prefill = prefill;           // spec 20d: --ids through KolibriEngine::prefill
    ka.pp = have_prefill_len;       // --prefill-length N already set depth = N above
    ka.pp_chunk = pp_chunk;
    ka.pp_backend_given = have_pp_backend;
    ka.pp_backend = pp_backend;
    ka.mtp = mtp_on;
    ka.kv8 = kv_cache == runtime::KvCache::Int8;
    ka.profile = profile;
    return cli::kolibri::run_decode<StdoutToStderr>(ka);
  }
  // Spec 21-QFN-v1c: Qwen3.8-Flash-Next (model_type qwen4_exp) runs runtime::qwen4exp::Qwen4ExpEngine on one or two cards -
  // dispatched here, beside Kolibri-1 and before the Qwen engine's pipeline and kv-cache rules (its own refusals,
  // cli/qwen4exp_decode.h: the prefill's other backends (21-QFN-v1d: l0 only), --spec mtp (21-QFN-v1e), int8 KV (decision 8), the whole
  // model without --layers (spec 22)). A path that does not resolve is not this family - the flows below report it.
  if (cli::qwen4exp::is_qwen4exp(path)) {
    if (dflash.on)
      throw std::runtime_error("--spec dflash is built for Qwen3.8 beside its drafter; Qwen3.8-Flash-Next has none");
    cli::qwen4exp::DecodeArgs qa;
    qa.path = path;
    qa.ids_path = ids_path;
    qa.n = n;
    qa.bench = bench;
    qa.depth = depth;
    qa.tg = tg;
    qa.max_len = max_len_arg;
    qa.reserve = mem_reserve;
    qa.card_bytes = hw.card_bytes;
    qa.lm_head = lm_head;
    qa.layers = layers;
    qa.layers_auto = layers_auto;
    qa.pipe = pipe;
    qa.pp_given = pp_given;
    qa.prefill = prefill;           // spec 21-QFN-v1d: --ids through Qwen4ExpEngine::prefill
    qa.pp = have_prefill_len;       // --prefill-length N already set depth = N above
    qa.pp_chunk = pp_chunk;
    qa.pp_backend_given = have_pp_backend;
    qa.pp_backend = pp_backend;
    qa.mtp = spec_arg == "mtp";     // b70-decode decodes it plain: --spec auto (b70-serve's MTP) stays plain here
    qa.kv8 = kv_cache == runtime::KvCache::Int8;
    qa.profile = profile;
    return cli::qwen4exp::run_decode<StdoutToStderr>(qa);
  }
  if (have_layers)
    throw std::runtime_error("--layers belongs to Kolibri-1's and Qwen3.8-Flash-Next's development modes (load only "
                             "the first N layers); this checkpoint is neither model_type kolibri1 nor "
                             "qwen4_exp");
  runtime::PrefillPath pp_path = cli::prefill_path(
      have_prefill_len || prefill,
      have_pp_backend ? pp_backend : runtime::prefill::default_prefill_backend());
  if (have_prefill_len || prefill) cli::warn_int8_prefill_kv8(pp_path.backend, kv_cache);
  if (profile && kv_cache == runtime::KvCache::Int8)
    throw std::runtime_error("--profile captures its own bf16-KV list; it has no --kv-cache int8");
  // Spec 16b: --parallel's refusals are argument-only, so they come before the device too.
  cli::check_pipeline(pipe, {prefill, have_prefill_len, mtp_on, profile});

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

  // Spec 17b: tensor parallel, its own flow (cli/tp_decode.h).
  if (pipe.tp()) {
    cli::TpDecodeArgs ta;
    ta.path = path;
    ta.ids = ids;
    ta.bench = bench;
    ta.n = n;
    ta.depth = depth;
    ta.tg = tg;
    ta.max_len = max_len_arg;
    ta.trained = trained;
    ta.reserve = mem_reserve;
    ta.lm_head = lm_head;
    ta.kv = kv_cache;
    ta.check_len = check_len;
    ta.prefill = prefill || have_prefill_len;   // spec 17c
    ta.bench_prefill = have_prefill_len;
    ta.chunk = pp_chunk;
    ta.backend = pp_path.backend;
    ta.mtp = mtp_on;   // spec 17d: --ids only (synthetic runs refused --spec above)
    ta.run_mtp = [&](runtime::TpEngine& e) { return generate_mtp(e, n, mtp_auto, mtp_k, lm_head, true); };
    return cli::run_tp_decode<StdoutToStderr>(ta);
  }
  // --parallel auto (the zero-config cleanup): one card when the model and the asked context fit it, else
  // pipeline parallel (cli::parallel_auto, from the checkpoint's descriptor before the load).
  if (pipe.mode == cli::Parallel::Auto) {
    pipe.devices = cli::parallel_auto(loader::resolve_snapshot(path), max_len_arg, mtp_on, kv_cache, pp_path,
                                      hw.card_bytes, mem_reserve);
    if (pipe.on()) {
      std::print(stderr, "parallel: auto -> pp (the model and the context do not fit one card)\n");
      if (hw.cards < 2)
        throw std::runtime_error("--parallel auto: this model at this --max-len does not fit one card and Level Zero "
                                 "shows one - a shorter --max-len, or both cards visible (ZE_AFFINITY_MASK=0,1)");
      cli::check_pipeline(pipe, {prefill, have_prefill_len, mtp_on, profile});
      if (dflash.on) cli::dflash::check_engine(dflash, true, kv_cache);
    }
  }
  // Spec 16b: two devices, its own flow (cli/pipeline_decode.h); --parallel 1 goes on below.
  if (pipe.on()) {
    cli::PipelineDecodeArgs pa;
    pa.path = path;
    pa.ids = ids;
    pa.bench = bench;
    pa.n = n;
    pa.depth = depth;
    pa.tg = tg;
    pa.max_len = max_len_arg;
    pa.trained = trained;
    pa.reserve = mem_reserve;
    pa.lm_head = lm_head;
    pa.kv = kv_cache;
    pa.pipe = pipe;
    pa.check_len = check_len;
    pa.prefill = prefill;                  // spec 16c
    pa.bench_prefill = have_prefill_len;   // --prefill-length N already set depth = N
    pa.chunk = pp_chunk;
    pa.pf = pp_path;
    pa.mtp = mtp_on;   // spec 16d: --ids only (synthetic runs refused --spec mtp above)
    pa.run_mtp = [&](runtime::PipelineEngine& e) { return generate_mtp(e, n, mtp_auto, mtp_k, lm_head); };
    return cli::run_pipeline_decode<StdoutToStderr>(pa);
  }

  l0::Context ctx(0u);
  std::print(stderr, "device: {} ({:d} EUs)\n", ctx.name().c_str(), ctx.eu_count());

  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(ctx, path, cli::load_len(max_len_arg, trained), /*mtp=*/mtp_on, lm_head);
  }();
  // Spec 19c: the drafter beside the target, before the plan. Its load line goes to stderr like the target's (box
  // row 55, 2026-10-27: on stdout it put a line ahead of the ids and r55.decode's cmp against the plain run failed).
  std::unique_ptr<cli::dflash::Loaded> drafter_load;
  if (dflash.on) {
    StdoutToStderr redirect;
    cli::dflash::resolve_format(dflash);
    drafter_load = std::make_unique<cli::dflash::Loaded>(cli::dflash::load(ctx, dflash, model));
  }
  // Spec 15c: --prefill-length / --prefill on a model whose prefill is not built stop here with the
  // reason, before a plan or an engine is made for it (spec 15d: Ornith's is built).
  if (have_prefill_len || prefill) model::require_prefill(*model.desc);
  // Plan 5d: the chunk cap --prefill-chunk needs (Ornith above 2048 refused by name), planned.
  const uint32_t asked_cap =
      have_prefill_len || prefill ? runtime::prefill_chunk_cap_for(pp_chunk, *model.desc) : 0;
  if (pp_path.prefill) pp_path.chunk_cap = cli::planned_chunk_cap(asked_cap, *model.desc);   // decode-only: no cap read
  // Spec 6 §10: auto plans and re-tables the model; an explicit length is held to the
  // plan. The prefill scratch is planned only when this run prefills (ruling R7).
  const runtime::DrafterPlan drafter_plan = drafter_load ? drafter_load->plan : runtime::DrafterPlan{};
  max_len = cli::settle(ctx, model, max_len_arg, mem_reserve, pp_path, kv_cache, drafter_plan);
  if (max_len_arg.is_auto) check_len(max_len);
  const uint32_t chunk_cap =
      cli::settle_chunk_cap(ctx, model, {max_len}, mem_reserve, pp_path, kv_cache, drafter_plan, asked_cap);

  // --profile forks here: it replays an instrumented capture and has to reset
  // 774 events before every replay, which is the caller's job by design
  // (runtime/capture.h) - so it runs its own loop rather than an Engine's.
  if (profile) return run_profile(ctx, model, ids, steps, repeats);

  // debug_resid off: the per-layer tap costs 64 device copies a token and only
  // the golden gate (Task 8) reads it.
  std::unique_ptr<runtime::dflash::DFlashDrafter> drafter;
  if (drafter_load)
    drafter = std::make_unique<runtime::dflash::DFlashDrafter>(ctx, std::move(drafter_load->w), drafter_load->head_rows,
                                                               max_len, dflash.max_k);
  runtime::Engine eng(ctx, std::move(model), max_len, /*debug_resid=*/false, kv_cache, drafter.get());
  if (drafter) drafter->attach(eng, /*draft_vocab=*/false);
  std::print(stderr,
             "engine: {:d} kernels, {:d} modules, max_len {:d}, {:.2f} GB of persistent state, "
             "decode attention {}, kv cache {}\n",
             eng.step().kernel_count, eng.step().modules.size(), eng.max_len(),
             eng.buffers().persistent_bytes() / 1e9,
             "v2",
             runtime::kv_cache_name(eng.kv_cache()));

  // Ingestion is one replay per prompt token unless `--prefill-length` or `--prefill` is
  // given, in which case it is `Engine::prefill`. The measured window is
  // **the whole call**, and that needs no extra instrumentation to be the
  // interface's "first prefill launch to the first generated id in cur_token":
  // `prefill()` returns only after its final `Context::wait()`, and its last
  // launch is `argmax_stage2`, which is the only writer of `cur_token`.
  // The prefill setup (scratch, Context, and on l0-int8 the one-time rotated
  // column-scale pass, spec 5 T2) happens here, at load, outside the window.
  size_t pp_launches_setup = 0;
  if (have_prefill_len || prefill) {
    if (have_pp_backend) eng.set_prefill_backend(pp_backend);
    eng.set_prefill_chunk_cap(chunk_cap);   // plan 5d: the cap the plan counted
    std::print(stderr, "prefill backend: {}\n", runtime::prefill_backend_name(eng.prefill_backend()));
    eng.prepare_prefill();
    pp_launches_setup = eng.prefill_launches();
  }
  // Spec 6: every device byte this engine holds, before the first prefill.
  std::print(stderr, "{}\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  size_t pp_launches = 0;
  if (have_prefill_len || prefill) {
    if (have_prefill_len) runtime::prefill::profile_reset();
    eng.prefill(ids, pp_chunk);
    if (have_prefill_len) pp_launches = eng.prefill_launches() - pp_launches_setup;
  } else {
    eng.ingest(ids);
  }
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (have_prefill_len)
    std::print(stderr,
               "pp: {:d} ids in {:.1f} ms ({:.2f} t/s) -- device-side, loader excluded, first"
               " prefill launch to the first generated id in cur_token; chunk {:d}, {:d} L0"
               " launches, pos {:d}\n",
               ids.size(), ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0,
               pp_chunk ? pp_chunk : eng.prefill_chunk_cap(), pp_launches, eng.pos());
  if (have_prefill_len) runtime::prefill::profile_report("--prefill-length", ingest_ms);
  if (!have_prefill_len && !prefill)
    std::print(stderr, "ingest: {:d} ids in {:.1f} ms ({:.2f} ms/token), pos {:d}\n", ids.size(),
               ingest_ms, ids.empty() ? 0.0 : ingest_ms / double(ids.size()), eng.pos());
  if (prefill)
    std::print(stderr, "prefill: {:d} ids, pos {:d}\n", ids.size(), eng.pos());

  if (!bench && mtp_on) return generate_mtp(eng, n, mtp_auto, mtp_k, lm_head);
  if (!bench && drafter) return generate_dflash(eng, n, dflash.max_k);   // spec 19c
  if (!bench) {
    eng.generate(n, [](uint32_t id) {
      std::print("{:d}\n", id);
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
  std::print(stderr,
             "  wall {:.1f} ms, fence {:.1f} ms, host {:.1f} ms  (host = wall - fence: argument-free\n"
             "  replay, so this is submit + the four bytes of shared memory, nothing else)\n"
             "  per token: {:.3f} ms total = {:.3f} ms fence + {:.3f} ms host\n",
             eng.last_gen_ms(), eng.last_fence_ms(), host_ms, ms_per_token,
             eng.last_fence_ms() / double(tg), host_ms / double(tg));

  // MBU against the two measured constants, spelled out so the reader can
  // check the arithmetic without leaving the terminal. The kernel-count line
  // is an *estimate* carried from probe_replay (0.52 µs/kernel; docs/15 "The dispatch gap") -
  // it is what the captured list costs in dispatch alone, and it is printed
  // beside the fence time to show how little of the fence it can explain.
  const double bytes_per_token = double(eng.model().report.read_per_token) / 1e9;
  const double achieved_gbs = eng.last_tok_per_s() * bytes_per_token;
  const double dispatch_ms = double(eng.step().kernel_count) * 0.52e-3;
  std::print(stderr,
             "  MBU: {:.2f} t/s x {:.3f} GB = {:.0f} GB/s of {:.0f} GB/s measured = {:.1f}%\n"
             "       (W is this checkpoint's own read/token, from the loader report above;\n"
             "        roofline at {:.0f} GB/s = {:.2f} t/s = {:.3f} ms/token)\n"
             "  dispatch floor (estimated): {:d} kernels x 0.52 us = {:.3f} ms/token,"
             " {:.1f}% of the fence\n",
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
  // Spec 12b: likewise an int8-KV row (`int8-kv`), so the interleaved pairs sort apart.
  const std::string head_tag = std::string(lm_head == loader::LmHeadForm::Int8 ? " int8-head" : "") +
                               (kv_cache == runtime::KvCache::Int8 ? " int8-kv" : "");
  std::print("| b70-decode {}{} | {:d} | {:d} | {:.2f} | {:.2f} |\n", sha, head_tag.c_str(), depth, tg,
             eng.last_tok_per_s(), ms_per_token);
  // The tg row above is byte-identical to what it always was, and the pp row
  // is a SECOND line rather than extra columns on it, so every existing parser
  // of docs/BENCHMARKS.md's table keeps working (interfaces.md's CLI contract;
  // tools/bench_decode.sh --prefill-length reads both).
  if (have_prefill_len)
    std::print("| b70-decode {}{} {} pp | {:d} | {:d} | {:.1f} | {:.2f} |\n", sha, head_tag.c_str(),
               runtime::prefill_backend_name(eng.prefill_backend()), depth,
               pp_chunk ? pp_chunk : eng.prefill_chunk_cap(), ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  std::print(stderr,
             "(measured, one run. docs/BENCHMARKS.md records the median of three on an idle box;\n"
             " tools/bench_decode.sh is that harness. B70_GIT_SHA unset prints `unknown`.)\n");
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::print(stderr, "b70-decode: {}\n", e.what());
    return 1;
  }
}
