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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cli/bench_prompt.h"
#include "cli/dflash_args.h"   // spec 19c: --spec dflash
#include "cli/flags.h"
#include "cli/hardware_gate.h"
#include "cli/stdout_to_stderr.h"
#include "cli/tap_dump.h"
#include "cli/inkling2_decode.h"
#include "runtime/inkling2/inkling2_engine.h"
#include "runtime/inkling2/inkling2_window.h"
#include "runtime/inkling2/inkling2_pp_engine.h"
#include "runtime/inkling2/inkling2_tp_engine.h"
#include "cli/kolibri_decode.h"
#include "cli/qwen4exp_decode.h"
#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_decode.h"
#include "cli/tp_decode.h"
#include "cli/spec_args.h"
#include "kernels/kernels.h"
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

using cli::kBenchPrompt;
using cli::kBenchPromptLen;

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
  [[=Help("", "print Inkling's memory plan from checkpoint headers (both cards' with --parallel pp or tp)")]] bool plan = false;
  [[=Help("<file>", "whitespace-separated token ids")]] std::string ids;
  [[=Help("<N>", "ids to generate, greedily; t/s on stderr")]] Arg<uint32_t> n;
  [[=Help("", "run the prompt through the engine's prefill instead of one decode replay per id")]] bool prefill = false;
  [[=Help("S", "off, mtp (the MTP head for --ids, adaptive K <= --spec-max 1..3;\n"
               "Qwen3.8, Ornith) or dflash (Qwen3.8's DFlash drafter, K <= --spec-max\n"
               "1..7, one card; --draft-model PATH, required). Inkling accepts mtp or auto\n"
               "for its built-in DSpark drafter in --ids and --bench, and dflash with\n"
               "--draft-model for the DFlash 2 drafter.")]]
  Arg<std::string> spec{"off"};   // a developer tool: the plain list unless asked (cli/spec_args.h)
  [[=Help("K", "the proposer's cap (see --spec).", "the most")]] Arg<uint32_t> spec_max;
  [[=Help("PATH", "--spec dflash's drafter. Inkling DFlash 2: packed int4 g64 export or bf16 release (loaded as int8).")]]
  Arg<std::string> draft_model;   // spec 19c
  // Spec 6 §10: 16384 stays the default here, so bench rows stay comparable; `auto` plans the largest max_len that
  // fits, as b70-serve's default does.
  [[=Help("L", "KV cache and RoPE capacity. The default is auto for Inkling and 16384 for other\n"
               "models. Auto finds the largest length that fits.")]]
  Arg<std::string> max_len{"16384"};
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
  [[=Help("auto|gpu|cpu", "how experts held in host RAM are served: auto picks per step from a\n"
                          "startup measurement (default), gpu reads them over PCIe into the card, cpu\n"
                          "computes them on the CPU.")]]
  std::string host_experts = "auto";   // PERF-124 (runtime/qwen4exp/qwen4exp_cpu_experts.h)
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
              "the per-launch anatomy (never a bench row). Inkling times target steps with 1 to 8"
              " rows, committing one row per step, and its drafter's pass,\n"
              "then prints one step of 1, 2 and 8 rows and one drafter pass timed launch by launch.")]]
  bool profile = false;
  [[=Help("N", "--profile: instrumented steps.")]] Arg<uint32_t> steps{32};
  [[=Help("R", "--profile: R sessions over one ingest, a per-family mean +/- spread table.")]] Arg<uint32_t> repeats{1};
  [[=Help("FILE", "Inkling (one card) or Qwen3.8-Flash-Next (one card or pp), with --ids and --n:\n"
                  "a plain greedy run that writes FILE (<corpus>__<label>.taps.safetensors):\n"
                  "every position's DFlash 2 drafter taps (the prompt's last 2048 and each\n"
                  "generated one), the head's top 64 logits and their log-sum-exp - the input\n"
                  "of an offline drafter acceptance measurement. Qwen3.8-Flash-Next: --ids and\n"
                  "--dump-taps take comma-separated lists, one output a prompt.")]]
  std::string dump_taps;
  [[=Section("\nstill being measured (they go when their numbers decide):\n"),
    =Help("bf16|int8", "the KV cache's form; int8 halves it.", "bf16")]]
  std::string kv_cache;   // spec 12b; empty: runtime::default_kv_cache()
  [[=Help("l0|l0-int8",
          "--prefill-length / --prefill: bf16, or int8 - every prompt linear, the context (it\n"
          "turns on --kv-cache int8) and the attention scores: 40-47 % faster prompts than bf16\n"
          "on Qwen3.8-27B; the most likely next token differs at 3 of 585 positions, 0 of 12\n"
          "long exact-copy tool calls changed.",
          "l0")]]
  Arg<std::string> prefill_backend;
};

void usage() {
  std::print(
      stderr,
      "usage: b70-decode <snapshot-or-repo> --ids <file> --n <N> [--prefill] [--spec mtp|dflash]\n"
      "                                     [--max-len 16384|auto] [--parallel auto|1|pp|tp]\n"
      "       b70-decode <Inkling snapshot-or-repo> --plan [--max-len auto] [--spec auto|off] [--kv-cache int8]\n"
      "                                     [--parallel pp]\n"
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

std::vector<uint32_t> read_ids(const std::string& path, uint32_t vocab = Qwen35::kVocab) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  std::string w;
  while (f >> w) {
    const uint32_t id = cli::flags::parse_u32("--ids file '" + path + "'", w);
    // Not kVocabUsed: rows [kVocabUsed, kVocab) exist in the embedding table
    // and a checkpoint may legitimately spell one. Beyond kVocab the gather
    // would read past the allocation, so that is the hard bound.
    if (id >= vocab)
      throw std::runtime_error("--ids file '" + path + "': id " + std::to_string(id) +
                               " is outside the vocabulary (" + std::to_string(vocab) +
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
// two runs three minutes apart, but across a day of work on the GPU machine the
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
  // sessions this machine could have run, not the population. 0 by definition at
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
      "    The recorded step time is the --bench median.\n"
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
// greedy acceptance keeps them equal to the plain run's ids, which is what the test machine's
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
  if constexpr (std::is_same_v<Eng, runtime::Engine>)
    eng.greedy_head(true);   // greedy ids only: the verify head's candidate pass where the lists carry it
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
  eng.greedy_head(true);   // greedy ids only: the verify head's candidate pass where the lists carry it
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

template <class E>
int run_inkling_on(E& engine, const Flags& f, const std::vector<uint32_t>& ids, uint32_t n, uint32_t cap, bool draft,
                   bool prompt_pass, uint32_t chunk);

// --dump-taps: the prompt by the prompt pass (a block hook copies each chunk's tap rows), then n plain greedy steps
// on the full head (verify of one row: its logits readable), each step's tap row, top 64 logits (the served
// softcapped values) and log-sum-exp over the live vocabulary, written as cli/tap_dump.h's file (DSpark's taps).
int dump_inkling_taps(runtime::inkling2::Engine& engine, const Flags& f, const std::vector<uint32_t>& prompt,
                      uint32_t n, uint32_t chunk, const std::string& snapshot, const std::vector<uint32_t>& tap_ids) {
  using D = model::Inkling2Desc;
  constexpr size_t tap_row = 4 * size_t(D::hidden) * sizeof(uint16_t);
  const auto np = uint32_t(prompt.size());
  cli::TapDump dump(np, n, D::hidden, D::vocab_used, tap_ids);
  const uint32_t tap_from = dump.tap_from, rows = dump.rows;
  std::vector<uint8_t> taps(size_t(rows) * tap_row);
  dump.ids.assign(prompt.begin(), prompt.end());
  std::vector<float> logits(D::vocab_used);
  uint32_t chunk_start = 0;
  engine.set_block_hook([&](uint32_t end, bool) {
    // the chunk's rows at or after tap_from
    const uint32_t first = std::max(chunk_start, tap_from);
    if (end > first) engine.read_taps(true, first - chunk_start, end - first, taps.data() + (first - tap_from) * tap_row);
    chunk_start = end;
  });
  uint32_t next = engine.prefill(prompt, chunk);
  engine.set_block_hook({});
  engine.read_prefill_logits(logits);
  dump.head(np - 1 - tap_from, logits.data());
  dump.greedy[np - 1 - tap_from] = int32_t(next);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t row = np + i - tap_from;
    const uint32_t out = engine.verify({&next, 1})[0];
    engine.read_logits(0, logits);
    engine.read_taps(false, 0, 1, taps.data() + size_t(row) * tap_row);
    engine.commit(1);
    dump.head(row, logits.data());
    dump.greedy[row] = int32_t(out);
    dump.ids.push_back(int32_t(next));
    next = out;
  }
  // tap t of every row: a strided copy out of [rows][4][hidden]
  for (size_t t = 0; t < tap_ids.size(); ++t)
    for (uint32_t r = 0; r < rows; ++r)
      std::memcpy(dump.taps[t].data() + size_t(r) * D::hidden, taps.data() + r * tap_row + t * D::hidden * 2,
                  D::hidden * 2);
  dump.snapshot = snapshot;
  dump.kv = engine.kv8() ? "int8" : "bf16";
  dump.write(f.dump_taps);
  std::print(stderr, "Inkling tap dump {}: {} prompt ids (taps from {}), {} generated, {} kv cache\n", f.dump_taps, np,
             tap_from, n, engine.kv8() ? "int8" : "bf16");
  return 0;
}

int run_inkling(const std::string& path, const Flags& f) {
  const auto spec = f.spec.given ? f.spec.value : "auto";
  if (spec != "off" && spec != "auto" && spec != "mtp" && spec != "dflash")
    throw std::runtime_error("Inkling accepts --spec off, auto, mtp or dflash");
  if ((spec == "dflash") != f.draft_model.given)
    throw std::runtime_error("Inkling --spec dflash takes the DFlash 2 drafter's folder as --draft-model, and only it");
  const bool prompt_pass = !f.ids.empty() || f.prefill_length.given;
  if ((f.prefill && f.ids.empty()) || (f.prefill_length.given && (!f.bench || f.depth.given)))
    throw std::runtime_error("Use --ids for a prompt or --bench --prefill-length to time the prompt pass");
  if ((f.prefill_chunk.given || f.prefill_backend.given) && !prompt_pass)
    throw std::runtime_error("Prompt options require --ids or --bench --prefill-length");
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0;
  if (f.prefill_backend.given && !runtime::parse_prefill_backend(f.prefill_backend.value, backend))
    throw std::runtime_error("--prefill-backend expects l0 or l0-int8, got '" + f.prefill_backend.value + "'");
  const bool int8_pass = backend == runtime::PrefillBackend::L0Int8;
  const uint32_t chunk = f.prefill_chunk.given ? f.prefill_chunk.value : runtime::inkling2::kPrefillRows;
  if (chunk == 0 || chunk > runtime::inkling2::kPrefillRows)
    throw std::runtime_error("Inkling --prefill-chunk must be between 1 and " +
                             std::to_string(runtime::inkling2::kPrefillRows));
  const uint32_t depth = f.prefill_length.given ? f.prefill_length.value : f.depth.value;
  uint32_t planned_chunk = prompt_pass || f.profile ? chunk : 0;   // --profile ingests by the prompt pass
  if (f.layers.given || f.pipeline_split.given ||
      (f.parallel != "auto" && f.parallel != "1" && f.parallel != "pp" && f.parallel != "tp"))
    throw std::runtime_error("Inkling decode runs on one card, --parallel pp or --parallel tp");
  // The int8 prompt pass runs on the int8 cache (its scores read the int8 keys), as on a dense model.
  const auto kv_flag = f.kv_cache.empty() ? runtime::KvCache::Bf16 : cli::parse_kv_cache_arg(f.kv_cache);
  const bool kv8 = cli::int8_pass_kv(backend, true, f.kv_cache, kv_flag) == runtime::KvCache::Int8;
  if (int(f.bench) + int(f.profile) + int(!f.ids.empty()) != 1)
    throw std::runtime_error("Choose one of --bench, --profile or --ids for Inkling decode");
  if ((f.n.given && f.ids.empty()) || (f.tg.given && !f.bench) || (f.depth.given && !f.ids.empty()) ||
      ((f.steps.given || f.repeats.given) && !f.profile) || (f.profile && ((f.spec.given && spec != "dflash") || f.spec_max.given)))
    throw std::runtime_error("Use --steps and --repeats with --profile, --tg with --bench, and --n with --ids");
  const uint32_t n = f.bench ? f.tg.value : f.profile ? f.steps.value : f.n.value;
  const uint32_t cap = f.spec_max.given ? f.spec_max.value : 7;
  if (n == 0 || cap == 0 || cap > 7)
    throw std::runtime_error("Inkling requires a positive generation length and --spec-max from 1 to 7");
  if (f.profile && f.repeats.value == 0) throw std::runtime_error("Inkling --repeats must be positive");

  if (!f.dump_taps.empty() && (f.ids.empty() || f.parallel == "pp" || f.parallel == "tp" || f.spec_max.given ||
                               spec == "dflash"))
    throw std::runtime_error("--dump-taps takes --ids and --n on one card");
  const loader::Inkling2Checkpoint checkpoint(path, spec == "dflash" ? f.draft_model.value : "");
  if (!f.dump_taps.empty() && !checkpoint.has_drafter())
    throw std::runtime_error("--dump-taps needs the checkpoint's mtp.safetensors (the taps exist with a drafter)");
  const bool draft = (f.profile || !f.dump_taps.empty() || spec != "off") && checkpoint.has_drafter();
  if (!f.profile && spec != "off" && !draft) std::print(stderr, "Inkling checkpoint has no mtp.safetensors. Using plain decode.\n");
  auto length = cli::parse_max_len(f.max_len.given ? f.max_len.value : "auto");
  cli::check_before_load(length, model::Inkling2Desc::trained_max_len, false);
  std::vector<uint32_t> ids;
  const size_t needed = f.profile ? size_t(runtime::inkling2::kRows) * (n + 2) : size_t(n);
  if (f.bench || f.profile) {
    if (depth == 0 || depth + needed > model::Inkling2Desc::trained_max_len)
      throw std::runtime_error("Inkling --depth and --tg must fit the trained context");
    ids.resize(depth);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = kBenchPrompt[i % kBenchPromptLen];
  } else {
    ids = read_ids(f.ids, model::Inkling2Desc::vocab);
  }

  l0::Context ctx(0);
  const auto& desc = checkpoint.desc();
  const auto& weights = checkpoint.layout();
  const size_t card = ctx.memory_bytes();
  const size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  const auto one_card_fits = [&] {
    return card > reserve &&
           runtime::inkling2::sizes(desc, weights, length.value, draft, planned_chunk, kv8, int8_pass).memory().total() <=
               card - reserve;
  };
  // --parallel auto: one card when the asked context fits it, else pp.
  const bool tp = f.parallel == "tp";
  const bool pp = f.parallel == "pp" ||
                  (f.parallel == "auto" && !length.is_auto && !one_card_fits() && l0::Context::gpu_count() >= 2);
  if (pp || tp) cli::require_two_devices(l0::Context::gpu_count());
  std::optional<l0::Context> ctx1;
  if (pp || tp) ctx1.emplace(ctx, 1);
  const size_t cards = pp || tp ? std::min(card, ctx1->memory_bytes()) : card;
  // pp without --prefill-chunk: the context at 2,048-row chunks, then 4,096 where both cards still hold it.
  const bool pp_rows = pp && prompt_pass && !f.prefill_chunk.given;
  if (pp_rows) planned_chunk = 2048;
  if (length.is_auto)
    length.value = tp   ? runtime::inkling2::tp_max_len_that_fits(desc, weights, cards, reserve, draft, planned_chunk,
                                                                  kv8, int8_pass)
                   : pp ? runtime::inkling2::pp_max_len_that_fits(desc, weights, cards, reserve, draft, planned_chunk,
                                                                  kv8, int8_pass)
                        : runtime::inkling2::max_len_that_fits(desc, weights, card, reserve, draft, planned_chunk, kv8,
                                                               int8_pass);
  if (ids.empty() || ids.size() + needed > length.value)
    throw std::runtime_error("Set --max-len to fit the prompt and generated tokens");
  if (pp_rows)
    planned_chunk = std::max(planned_chunk, runtime::inkling2::pp_prefill_rows(desc, weights, cards, reserve,
                                                                               length.value, draft, kv8, int8_pass));
  if (tp) {
    const uint32_t fits = runtime::inkling2::tp_max_len_that_fits(desc, weights, cards, reserve, draft, planned_chunk,
                                                                  kv8, int8_pass);
    if (fits < length.value)
      throw std::runtime_error("Inkling does not fit two cards under --parallel tp at --max-len " +
                               std::to_string(length.value) + ". Use --max-len auto or at most " + std::to_string(fits) +
                               ".");
  } else if (pp) {
    const uint32_t fits = runtime::inkling2::pp_max_len_that_fits(desc, weights, cards, reserve, draft, planned_chunk,
                                                                  kv8, int8_pass);
    if (fits < length.value)
      throw std::runtime_error("Inkling does not fit two cards at --max-len " + std::to_string(length.value) +
                               " with the " + (kv8 ? "int8" : "bf16") + " KV cache. Use --max-len auto or at most " +
                               std::to_string(fits) + (kv8 ? "." : ", or add --kv-cache int8."));
  } else if (!one_card_fits()) {
    throw std::runtime_error(cli::inkling2::no_fit(desc, weights, card, reserve, draft, planned_chunk, length.value, kv8,
                                                   int8_pass));
  }

  std::optional<runtime::inkling2::Engine> one;
  std::optional<runtime::inkling2::PpEngine> two;
  std::optional<runtime::inkling2::TpEngine> split;
  if (tp) {
    split.emplace(ctx, *ctx1, checkpoint, length.value, draft, planned_chunk, kv8, int8_pass);
    for (uint32_t i = 0; i < 2; ++i)
      std::print(stderr, "{}\n", runtime::format_memory(("Inkling tp, card " + std::to_string(i)).c_str(),
                                                         split->memory(i), i ? ctx1->memory_bytes() : card));
  } else if (pp) {
    two.emplace(ctx, *ctx1, checkpoint, length.value, draft, planned_chunk, kv8, int8_pass);
    for (uint32_t i = 0; i < 2; ++i)
      std::print(stderr, "{}\n", runtime::format_memory(("Inkling pp, card " + std::to_string(i)).c_str(),
                                                         two->memory(i), i ? ctx1->memory_bytes() : card));
  } else {
    one.emplace(ctx, checkpoint, length.value, draft, planned_chunk, kv8, int8_pass);
    if (!f.dump_taps.empty()) return dump_inkling_taps(*one, f, ids, n, chunk, path, desc.draft.taps);
  }
  if (draft && checkpoint.dflash())
    std::print(stderr, "Inkling DFlash 2 drafter: {}\n",
               weights.draft_format == loader::DraftFormat::Int4G64 ? "packed int4 g64" : "int8 at load");
  if (int8_pass)
    std::print(stderr, "Inkling prompt pass: int8 - the attention linears int8 (rotated), int8 attention scores on the "
               "int8 cache, the pass's intermediate rows in bf16; the experts as in the bf16 pass\n");
  if (tp) return run_inkling_on(*split, f, ids, n, cap, draft, prompt_pass, chunk);
  return pp ? run_inkling_on(*two, f, ids, n, cap, draft, prompt_pass, chunk)
            : run_inkling_on(*one, f, ids, n, cap, draft, prompt_pass, chunk);
}

// run_inkling's loops over one card's engine or the two-card one (the same calls).
template <class E>
int run_inkling_on(E& engine, const Flags& f, const std::vector<uint32_t>& ids, uint32_t n, uint32_t cap, bool draft,
                   bool prompt_pass, uint32_t chunk) {
  constexpr const char* kMode = std::is_same_v<E, runtime::inkling2::PpEngine>   ? " pp"
                               : std::is_same_v<E, runtime::inkling2::TpEngine> ? " tp"
                                                                                : "";
  const auto ingest = [&] {
    uint32_t next = 0;
    for (size_t i = 0; i < ids.size();) {
      const uint32_t rows = uint32_t(std::min<size_t>(runtime::inkling2::kRows, ids.size() - i));
      next = engine.verify({ids.data() + i, rows}).back();
      engine.commit(rows);
      i += rows;
    }
    return next;
  };
  if (f.profile) {
    // Every verify width and the draft pass on one ingest a repeat (the window's cost table), then the anatomy of
    // the 1-, 2- and 8-row steps.
    constexpr uint32_t R = runtime::inkling2::kRows;
    std::array<double, R> ms{};
    double draft_ms = 0.;
    std::array<uint32_t, 8 * model::Inkling2Desc::visit_count> last_route{};
    uint64_t token_shared = 0, token_tried = 0, pass_shared = 0, pass_tried = 0;
    const auto elapsed = [](auto start) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    for (uint32_t repeat = 0; repeat < f.repeats.value; ++repeat) {
      engine.reset();
      engine.prefill(ids, chunk);
      for (uint32_t rows = 1; rows <= R; ++rows)
        for (uint32_t i = 0; i < n + 1; ++i) {
          std::array<uint32_t, R> input{};
          for (uint32_t j = 0; j < rows; ++j) input[j] = kBenchPrompt[(i + j) % kBenchPromptLen];
          const auto start = std::chrono::steady_clock::now();
          engine.verify({input.data(), rows});
          engine.commit(1);
          if (i) ms[rows - 1] += elapsed(start);
          if constexpr (std::is_same_v<E, runtime::inkling2::Engine>) if (rows == 1) {
            // how often a visit's experts were already chosen: the token before, an earlier pass (one card)
            const auto r = engine.routes();
            const auto shared = [&](const auto& a, uint32_t va, const auto& b, uint32_t vb) {
              uint32_t k = 0;
              for (uint32_t x = 0; x < 8; ++x)
                k += std::count(b.begin() + vb * 8, b.begin() + vb * 8 + 8, a[va * 8 + x]);
              return k;
            };
            constexpr uint32_t core = model::Inkling2Desc::core_end - model::Inkling2Desc::core_start;
            for (uint32_t v = 0; i && v < model::Inkling2Desc::visit_count; ++v) {
              token_shared += shared(r, v, last_route, v);
              token_tried += 8;
            }
            for (uint32_t v = model::Inkling2Desc::core_start + core; v < model::Inkling2Desc::core_start + 3 * core;
                 ++v) {
              pass_shared += shared(r, v, r, v - core);
              pass_tried += 8;
            }
            last_route = r;
          }
        }
      for (uint32_t i = 0; draft && i < n + 1; ++i) {
        const auto start = std::chrono::steady_clock::now();
        engine.propose(kBenchPrompt[i % kBenchPromptLen], R - 1);
        if (i) draft_ms += elapsed(start);
      }
    }
    for (auto& value : ms) value /= double(n) * f.repeats.value;
    draft_ms /= double(n) * f.repeats.value;
    for (uint32_t rows = 1; rows <= R; ++rows)
      std::println("Inkling {}-row target with one-row commit, {} kernel launches, {:.3f} ms per step, {:.3f}x plain",
                   rows, engine.launch_count(rows, 1), ms[rows - 1], ms[rows - 1] / ms[0]);
    if (draft) std::println("Inkling draft pass {:.3f} ms, {:.3f}x plain", draft_ms, draft_ms / ms[0]);
    std::println("Inkling experts already chosen: by the token before {:.1f}%, by the previous pass {:.1f}%",
                 100. * token_shared / std::max<uint64_t>(token_tried, 1), 100. * pass_shared / std::max<uint64_t>(pass_tried, 1));
    std::fflush(stdout);
    if constexpr (std::is_same_v<E, runtime::inkling2::Engine>) {   // the launch-by-launch anatomy, one card
      for (const uint32_t rows : {1u, 2u, 4u, R}) {
        std::array<uint32_t, R> input{};
        for (uint32_t j = 0; j < rows; ++j) input[j] = kBenchPrompt[j % kBenchPromptLen];
        engine.profile_target({input.data(), rows});
      }
      if (draft) engine.profile_draft(kBenchPrompt[0]);
    }
    if constexpr (std::is_same_v<E, runtime::inkling2::TpEngine>) {
      for (const uint32_t rows : {1u, R}) {
        std::array<uint32_t, R> input{};
        for (uint32_t j = 0; j < rows; ++j) input[j] = kBenchPrompt[j % kBenchPromptLen];
        engine.profile_target({input.data(), rows});
      }
      if (draft) engine.profile_draft(kBenchPrompt[0]);
    }
    return 0;
  }
  const auto prompt_start = std::chrono::steady_clock::now();
  if constexpr (std::is_same_v<E, runtime::inkling2::PpEngine>) {
    if (!f.prefill_chunk.given)
      chunk = std::min(engine.prefill_rows(), runtime::inkling2::PpEngine::chunk_rows(ids.size()));
  } else if constexpr (!std::is_same_v<E, runtime::inkling2::Engine>) {
    if (!f.prefill_chunk.given) chunk = E::chunk_rows(ids.size());
  }
  uint32_t next = prompt_pass ? engine.prefill(ids, chunk) : ingest();
  if (f.bench || prompt_pass) {   // --ids: on stderr, beside the ids
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prompt_start).count();
    std::print(f.bench ? stdout : stderr, "Inkling{} prompt: {} ids, {:.2f} t/s, {:.2f} ms to first id ({}-row chunks)\n",
               kMode, ids.size(), 1000. * ids.size() / ms, ms, chunk);
  }
  std::vector<uint32_t> output;
  output.reserve(n);
  namespace ri = runtime::inkling2;
  const bool df2 = engine.dflash();
  runtime::inkling2::DraftWindow window(
      std::is_same_v<E, ri::PpEngine>   ? std::span<const ri::DepthCost>(df2 ? ri::kPpDflashCosts.data() : ri::kPpCosts.data(),
                                                                         df2 ? ri::kPpDflashCosts.size() : ri::kPpCosts.size())
      : std::is_same_v<E, ri::TpEngine> ? std::span<const ri::DepthCost>(df2 ? ri::kTpDflashCosts.data() : ri::kTpCosts.data(),
                                                                         df2 ? ri::kTpDflashCosts.size() : ri::kTpCosts.size())
      : df2 ? std::span<const ri::DepthCost>(ri::kOneCardDflashCosts) : std::span<const ri::DepthCost>(ri::kOneCardCosts));
  uint32_t iterations = 0, proposed = 0, accepted = 0, plain_steps = 0;
  std::array<uint32_t, runtime::inkling2::DraftWindow::kMaxDrafts> tested_at{}, accepted_at{};
  const auto start = std::chrono::steady_clock::now();
  while (output.size() < n) {
    const auto previous = uint32_t(output.size());
    const uint32_t drafts = draft ? window.next(std::min(cap, n - previous - 1), engine.position()) : 0;
    next = engine.step_drafted(next, drafts, output);
    const auto kept = uint32_t(output.size()) - previous - 1;
    window.observe(drafts, kept);
    // Per draft position: tested (the earlier drafts accepted) and accepted, the rate a window policy reads.
    for (uint32_t i = 0; i < std::min(kept + 1, drafts); ++i) {
      ++tested_at[i];
      accepted_at[i] += i < kept;
    }
    proposed += drafts;
    accepted += kept;
    plain_steps += drafts == 0;
    ++iterations;
    if (!f.bench) {
      for (size_t i = previous; i < output.size(); ++i) std::println("{}", output[i]);
      std::fflush(stdout);
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const char* drafter = engine.dflash() ? "DFlash 2" : "DSpark";
  std::print(stderr, "Inkling generated {} ids in {} iterations, {:.2f} t/s, {:.2f} ms/token, {} {}\n",
             n, iterations, n * 1000.0 / ms, ms / n, drafter, draft ? "on" : "off");
  if (proposed)
  {
    std::print(stderr, "Inkling {} accepted {} of {} proposed ids ({:.1f}%), {} one-row steps; by position:",
               drafter, accepted, proposed, 100.0 * accepted / proposed, plain_steps);
    for (uint32_t i = 0; i < tested_at.size() && tested_at[i]; ++i)
      std::print(stderr, " {}/{}", accepted_at[i], tested_at[i]);
    std::print(stderr, "\n");
  }
  if (f.bench) {
    const char* sha = std::getenv("B70_GIT_SHA");
    std::println("| b70-decode {} Inkling{} {} {} | {} | {} | {:.2f} | {:.2f} |",
                 sha && *sha ? sha : "unknown", kMode, drafter, draft ? "on" : "off", ids.size(), n, n * 1000.0 / ms, ms / n);
  }
  return 0;
}

int run(int argc, char** argv) {
  std::string path;
  Flags f;
  if (!cli::flags::parse(argc, argv, f, path, usage)) {
    usage();
    return 0;
  }
  // The compiled device binaries: the compiled-in directory (the Docker image, a build tree) or, in an extracted
  // release tarball, the one beside this executable - resolved once here, so a missing one fails at startup.
  (void)kernels::kernel_dir();
  const bool inkling = cli::inkling2::is_inkling2(path);
  if (f.plan) {
    if (!inkling) throw std::runtime_error("--plan requires an Inkling 2 Mini checkpoint");
    return cli::inkling2::plan(path, cli::parse_max_len(f.max_len.given ? f.max_len.value : "auto"),
                               f.spec.given ? f.spec.value : "auto",
                               !f.kv_cache.empty() && cli::parse_kv_cache_arg(f.kv_cache) == runtime::KvCache::Int8,
                               f.parallel == "pp", f.parallel == "tp");
  }
  const auto host_experts = cli::qwen4exp::host_experts_mode(f.host_experts);
  if (f.host_experts != "auto" && !cli::qwen4exp::is_qwen4exp(path))
    throw std::runtime_error(cli::qwen4exp::host_experts_refusal(f.host_experts));
  if (inkling) return run_inkling(path, f);
  const std::string ids_path = f.ids;
  uint32_t n = f.n.value, depth = f.depth.value, tg = f.tg.value, steps = f.steps.value,
           repeats = f.repeats.value;
  cli::MaxLenArg max_len_arg = cli::parse_max_len(f.max_len.value);
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
  // The model family decides which proposers exist (cli/spec_args.h).
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
  cli::check_tp(pipe, {spec.kind == cli::SpecKind::DFlash, profile,
                       fam_kol});   // plan 21v2b: Qwen3.8-Flash-Next runs --parallel tp too (cli/qwen4exp_decode.h)
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
  // PERF-114: l0-int8 on a dense model runs on the int8 context (the KV cache), turned on here.
  const bool int8_dense = have_pp_backend && pp_backend == runtime::PrefillBackend::L0Int8 &&
                          cli::int8_context_model(path);
  if (have_pp_backend) kv_cache = cli::int8_pass_kv(pp_backend, int8_dense, f.kv_cache, kv_cache);
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
    qa.cards = hw.cards;
    qa.lm_head = lm_head;
    qa.layers = layers;
    qa.layers_auto = layers_auto;
    qa.pipe = pipe;
    qa.pp_given = pp_given;
    qa.prefill = prefill || !f.dump_taps.empty();   // spec 21-QFN-v1d: --ids through Qwen4ExpEngine::prefill
    qa.dump_taps = f.dump_taps;
    qa.host_experts = host_experts;
    qa.host_experts_given = f.host_experts != "auto";
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
  if (have_prefill_len || prefill) cli::check_int8_prefill_kv8(pp_path.backend, kv_cache, int8_dense);
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
  cli::dflash::main_candidate_head(ctx, model, mtp_on);   // PERF-123: before the plan
  // Spec 19c: the drafter beside the target, before the plan. Its load line goes to stderr like the target's (validation
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
  if (pp_path.prefill)   // decode-only: no cap read
    pp_path.chunk_cap = cli::planned_chunk_cap(asked_cap, *model.desc, max_len_arg, pp_path.backend);
  // Spec 6 §10: auto plans and re-tables the model; an explicit length is held to the
  // plan. The prefill scratch is planned only when this run prefills (ruling R7).
  const runtime::DrafterPlan drafter_plan = drafter_load ? drafter_load->plan : runtime::DrafterPlan{};
  max_len = cli::settle(ctx, model, max_len_arg, mem_reserve, pp_path, kv_cache, drafter_plan);
  if (max_len_arg.is_auto) check_len(max_len);
  const runtime::PrefillPath settled =
      cli::settle_chunk_cap(ctx, model, max_len, mem_reserve, pp_path, kv_cache, drafter_plan, asked_cap);

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
  if (drafter) drafter->attach(eng);
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
    eng.set_prefill_chunk_cap(settled.chunk_cap);   // plan 5d: the cap the plan counted
    eng.set_prefill_dense_bf16(settled.dense_bf16);   // open-work 38: likewise
    std::print(stderr, "prefill backend: {}\n", runtime::prefill_backend_name(eng.prefill_backend()));
    cli::notice_int8_prefill(eng.prefill_backend(), *eng.model().desc);
    eng.prepare_prefill();
    pp_launches_setup = eng.prefill_launches();
  }
  // Spec 6: every device byte this engine holds, before the first prefill.
  std::print(stderr, "{}\n", eng.memory_line().c_str());
  const auto t0 = std::chrono::steady_clock::now();
  size_t pp_launches = 0;
  if (have_prefill_len || prefill) {
    runtime::prefill::profile_reset();
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
  // --prefill too: the prompt pass with the MTP head or the DFlash drafter attached (--spec), which --bench refuses.
  if (have_prefill_len || prefill)
    runtime::prefill::profile_report(have_prefill_len ? "--prefill-length" : "--prefill", ingest_ms);
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
  // number the run produced. The benchmark harness runs it three times
  // and takes the median; the recorded step time is that median.
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
  // git call: a tree copied to the GPU machine WITHOUT .git cannot know
  // its own sha. Whoever runs the bench exports it (the benchmark harness
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
  // of the benchmark record's table keeps working (interfaces.md's CLI contract;
  // the benchmark harness reads both with --prefill-length).
  if (have_prefill_len)
    std::print("| b70-decode {}{} {} pp | {:d} | {:d} | {:.1f} | {:.2f} |\n", sha, head_tag.c_str(),
               runtime::prefill_backend_name(eng.prefill_backend()), depth,
               pp_chunk ? pp_chunk : eng.prefill_chunk_cap(), ingest_ms,
               ingest_ms > 0.0 ? double(ids.size()) * 1000.0 / ingest_ms : 0.0);
  std::print(stderr,
             "(measured, one run. The recorded figure is the median of three on an idle machine;\n"
             " B70_GIT_SHA names the commit; unset it prints `unknown`.)\n");
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
