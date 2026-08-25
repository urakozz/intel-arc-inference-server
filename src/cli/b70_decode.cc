// b70-decode - spec §12, plus spec 1.5 §3.4. Three modes:
//
//   b70-decode <snapshot-or-repo> --ids <file> --n <N> [--device N] [--max-len 16384]
//   b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]
//   b70-decode <snapshot-or-repo> --profile [--depth 4096] [--steps 32] [--device N]
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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/event.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/engine.h"

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

// Both measured, both cited where the bench prints the MBU it implies:
//   W  - bytes the loader makes resident and every token reads, from the
//        loader's own report (docs/13, 15,539,980,288 B = 15.540 GB).
//   BW - `tools/probe/probe_bw`, 590 GB/s median at 2 GB through the same
//        Level Zero path this engine submits on (docs/01).
constexpr double kBytesPerToken = 15.539980288;   // GB, measured
constexpr double kDeviceGBs = 590.0;              // GB/s, measured

void usage() {
  std::fprintf(
      stderr,
      "usage:\n"
      "  b70-decode <snapshot-or-repo> --ids <file> --n <N> [--device N] [--max-len 16384]\n"
      "  b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]\n"
      "  b70-decode <snapshot-or-repo> --profile [--depth 4096] [--steps 32] [--device N]\n"
      "\n"
      "  <snapshot-or-repo>  a snapshot directory, or an HF repo id resolved against the local\n"
      "                      cache ($HF_HOME or ~/.cache/huggingface). Never downloads.\n"
      "  --ids <file>   whitespace-separated token ids (tools/oracle/tokenize.py writes them)\n"
      "  --n <N>        ids to generate, greedily; one per line on stdout, t/s on stderr\n"
      "  --device N     GPU index. Absent: ONEAPI_DEVICE_SELECTOR=level_zero:N, else device 0.\n"
      "  --max-len <L>  KV cache and RoPE capacity (default 16384; the attention kernels are\n"
      "                 compiled per max_len, so only the compiled ones load)\n"
      "  --bench        ingest --depth synthetic ids, then time --tg generated ones and print\n"
      "                 a markdown row on stdout\n"
      "  --profile      ingest --depth synthetic ids on a plain list, then replay --steps\n"
      "                 INSTRUMENTED steps and print the per-launch anatomy on stdout.\n"
      "                 Never a bench row: every launch signals a host-visible event\n"
      "                 an unprofiled list does not pay for (spec 1.5 §3.3).\n");
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

int run_profile(l0::Context& ctx, const loader::LoadedModel& model,
                const std::vector<uint32_t>& ids, uint32_t steps) {
  runtime::DecodeBuffers buffers(ctx, model.max_len);
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
               " (profiled list: %zu events)\n",
               n, instr.modules.size(), buffers.max_len, buffers.persistent_bytes() / 1e9,
               prof.events.size());

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
  std::vector<double> per_launch(n, 0.0);
  double sum_all = 0.0, wall_all = 0.0;
  double sum_lo = 0.0, sum_hi = 0.0, wall_lo = 0.0, wall_hi = 0.0;
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
      sum += d;
    }
    if (s == 0) {
      sum_lo = sum_hi = sum;
      wall_lo = wall_hi = wall;
    }
    sum_lo = std::min(sum_lo, sum);
    sum_hi = std::max(sum_hi, sum);
    wall_lo = std::min(wall_lo, wall);
    wall_hi = std::max(wall_hi, wall);
    sum_all += sum;
    wall_all += wall;
  }
  const double sum_us = sum_all / steps;    // Σ per-kernel device time, per step
  const double wall_us = wall_all / steps;  // submit + fence, per step

  // --- the report. stdout: it is this mode's answer. ------------------------
  std::printf(
      "# b70-decode --profile - in-situ step anatomy\n"
      "depth %zu, steps %u, %zu launches/step, max_len %u, device %s (%u EUs)\n"
      "PROFILE MODE IS NOT BENCH MODE: every launch below signals a host-visible\n"
      "kernel-timestamp event, a per-launch flush an unprofiled list never pays. Per-kernel\n"
      "durations are kernelStart->kernelEnd and EXCLUDE that flush (it lands after kernelEnd),\n"
      "so they are comparable to probe per-kernel numbers; the gap at the bottom INCLUDES it\n"
      "and is an upper bound. The engine's ms/token comes from --bench, never from here.\n"
      "Every `share` below is of the sum of kernel durations, not of the fence wall.\n",
      ids.size(), steps, n, buffers.max_len, ctx.name().c_str(), ctx.eu_count());

  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return per_launch[a] > per_launch[b]; });
  const size_t top = std::min<size_t>(30, n);
  std::printf("\ntop %zu launches by mean us/step (measured, mean of %u steps)\n"
              "  rank  launch    us/step   share  label\n",
              top, steps);
  for (size_t r = 0; r < top; ++r) {
    const size_t i = order[r];
    const double us = per_launch[i] / steps;
    std::printf("  %4zu  %6zu  %9.3f  %5.2f%%  %s\n", r + 1, i, us, 100.0 * us / sum_us,
                instr.labels[i].c_str());
  }

  // The three rollups. `family` is the entry point (what docs/12 has a section
  // for); `variant` is the compiled binary (what docs/12's per-shape tables and
  // the probe price); `layer kind` is the GDN / FA / token-boundary split.
  std::vector<model::LayerDesc> layers = Qwen35::layers();
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
      accumulate(by_kind, layers[idx].kind == model::LayerKind::GDN ? "GDN layers (48)"
                                                                    : "FA layers (16)",
                 us);
    }
  }
  print_rollup("per-kernel-family rollup (measured, mean of the steps)", "family", by_family, steps,
               sum_us);
  print_rollup("per-variant rollup - the rows docs/12 and probe_gemv price", "variant", by_variant,
               steps, sum_us);
  print_rollup("per-layer-kind rollup", "layer kind", by_kind, steps, sum_us);

  std::printf(
      "\ntotals per step (mean of %u; the per-step spread is the honest error bar)\n"
      "  sum of %zu kernel durations   %10.3f us   [%.3f .. %.3f]   MEASURED, per-kernel,\n"
      "                                                                   flush-free\n"
      "  fence wall (submit + wait)    %10.3f us   [%.3f .. %.3f]   MEASURED, profiled list\n"
      "  gap = wall - sum              %10.3f us   = %.3f us/launch over %zu launches\n"
      "                                                 UPPER BOUND: includes one host-scope\n"
      "                                                 flush per launch (spec 1.5 §3.3)\n"
      "  sum / wall                    %10.4f\n",
      steps, n, sum_us, sum_lo, sum_hi, wall_us, wall_lo, wall_hi, wall_us - sum_us,
      (wall_us - sum_us) / double(n), n, sum_us / wall_us);
  std::printf(
      "\nreading these numbers\n"
      "  - Per-kernel us are in-situ and flush-free: compare them directly with the\n"
      "    probe-transplant floors in docs/12 (`gemv` -> Measured) and with docs/05's table.\n"
      "  - The gap is NOT a dispatch cost this engine pays: subtract the flush\n"
      "    (doc 07 #5's 0.52 us/kernel from probe_replay is the unprofiled estimate).\n"
      "  - The fence wall above is inflated by the same flushes and is NOT ms/token.\n"
      "    The recorded step time is the --bench median in docs/BENCHMARKS.md.\n");
  return 0;
}

int run(int argc, char** argv) {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256, steps = 32, max_len = 16384;
  uint32_t device = l0::Context::kFromEnv;
  bool bench = false, profile = false, have_n = false;
  bool have_depth = false, have_tg = false, have_steps = false;

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
    } else if (a == "--device") {
      device = parse_u32("--device", value(i, "--device"));
      // 0xFFFFFFFF is l0::Context's "ask the environment" sentinel, so it is
      // the one index --device cannot name.
      if (device == l0::Context::kFromEnv)
        throw std::runtime_error("--device is out of range");
    } else if (a == "--max-len") {
      max_len = parse_u32("--max-len", value(i, "--max-len"));
    } else if (a == "--bench") {
      bench = true;
    } else if (a == "--profile") {
      profile = true;
    } else if (a == "--depth") {
      depth = parse_u32("--depth", value(i, "--depth"));
      have_depth = true;
    } else if (a == "--tg") {
      tg = parse_u32("--tg", value(i, "--tg"));
      have_tg = true;
    } else if (a == "--steps") {
      steps = parse_u32("--steps", value(i, "--steps"));
      have_steps = true;
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
  if (have_depth && !synthetic)
    throw std::runtime_error("--depth belongs to --bench and --profile; --ids sizes its run"
                             " with --n");
  if (!synthetic && !have_n) {
    usage();
    throw std::runtime_error("--ids needs --n");
  }
  if (!synthetic && n == 0) throw std::runtime_error("--n 0 would generate nothing");
  if (bench && tg == 0) throw std::runtime_error("--tg 0 would time nothing");
  if (profile && steps == 0) throw std::runtime_error("--steps 0 would profile nothing");
  // Both synthetic modes exist to measure a step at a context depth, and the
  // cost of a step depends on that depth (attention's live-block count is the
  // measured example - docs/15). Depth 0 measures a shape nobody runs.
  if (synthetic && depth == 0)
    throw std::runtime_error("--depth 0 would ingest nothing; --bench and --profile both measure"
                             " a step at a context depth");

  // Everything that can be judged without the device or the 19 GB checkpoint is
  // judged first: failing on a typo'd --ids path after a 13-second load is a
  // worse CLI than failing in a millisecond.
  std::vector<uint32_t> ids;
  if (!synthetic) {
    ids = read_ids(ids_path);
    if (ids.size() + size_t(n) > size_t(max_len))
      throw std::runtime_error("prompt (" + std::to_string(ids.size()) + " ids) + --n " +
                               std::to_string(n) + " exceeds --max-len " + std::to_string(max_len));
  } else {
    // Both synthetic modes ingest to --depth and then replay: --tg generated
    // tokens for --bench, --steps instrumented ones for --profile. Same bound
    // either way - the KV cache and the RoPE table stop at max_len.
    const uint32_t after = bench ? tg : steps;
    const char* after_flag = bench ? "--tg " : "--steps ";
    if (size_t(depth) + size_t(after) > size_t(max_len))
      throw std::runtime_error("--depth " + std::to_string(depth) + " + " + after_flag +
                               std::to_string(after) + " exceeds --max-len " +
                               std::to_string(max_len));
    ids.resize(depth);
    for (uint32_t i = 0; i < depth; ++i) ids[i] = kBenchPrompt[i % kBenchPromptLen];
  }

  l0::Context ctx(device);
  std::fprintf(stderr, "device: %s (%u EUs)%s\n", ctx.name().c_str(), ctx.eu_count(),
               device == l0::Context::kFromEnv ? " [ONEAPI_DEVICE_SELECTOR]" : " [--device]");

  loader::LoadedModel model = [&] {
    StdoutToStderr redirect;
    return loader::load(ctx, path, max_len);
  }();

  // --profile forks here: it replays an instrumented capture and has to reset
  // 774 events before every replay, which is the caller's job by design
  // (runtime/capture.h) - so it runs its own loop rather than an Engine's.
  if (profile) return run_profile(ctx, model, ids, steps);

  // debug_resid off: the per-layer tap costs 64 device copies a token and only
  // the golden gate (Task 8) reads it.
  runtime::Engine eng(ctx, std::move(model), max_len);
  std::fprintf(stderr, "engine: %zu kernels, %zu modules, max_len %u, %.2f GB of persistent state\n",
               eng.step().kernel_count, eng.step().modules.size(), eng.max_len(),
               eng.buffers().persistent_bytes() / 1e9);

  // Ingestion is one replay per prompt token: this plan has no prefill kernel
  // (that is spec 2), so a long prompt costs decode time per id.
  const auto t0 = std::chrono::steady_clock::now();
  eng.ingest(ids);
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u\n", ids.size(),
               ingest_ms, ids.empty() ? 0.0 : ingest_ms / double(ids.size()), eng.pos());

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
  const double achieved_gbs = eng.last_tok_per_s() * kBytesPerToken;
  const double dispatch_ms = double(eng.step().kernel_count) * 0.52e-3;
  std::fprintf(stderr,
               "  MBU: %.2f t/s x %.3f GB = %.0f GB/s of %.0f GB/s measured = %.1f%%\n"
               "  dispatch floor (estimated): %zu kernels x 0.52 us = %.3f ms/token,"
               " %.1f%% of the fence\n",
               eng.last_tok_per_s(), kBytesPerToken, achieved_gbs, kDeviceGBs,
               100.0 * achieved_gbs / kDeviceGBs, eng.step().kernel_count, dispatch_ms,
               100.0 * dispatch_ms / (eng.last_fence_ms() / double(tg)));

  // The row's identity comes from the environment, not from a configure-time
  // git call: tools/box.sh syncs the tree WITHOUT .git, so the box cannot know
  // its own sha. Whoever runs the bench exports it (tools/bench_decode.sh
  // does), and an unset variable prints as `unknown` rather than as a wrong
  // sha in a recorded measurement.
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s | %u | %u | %.2f | %.2f |\n", sha, depth, tg, eng.last_tok_per_s(),
              ms_per_token);
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
