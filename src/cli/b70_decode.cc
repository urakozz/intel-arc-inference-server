// b70-decode - spec §12. Two modes over one runtime::Engine:
//
//   b70-decode <snapshot-or-repo> --ids <file> --n <N> [--device N] [--max-len 16384]
//   b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]
//
// stdout is the machine-readable channel and carries nothing but the answer:
// one generated id per line in `--ids` mode, one markdown row in `--bench`
// mode. Everything else - the device, the loader's report, the timings - goes
// to stderr, so `b70-decode … --ids p.ids --n 32 > out` is a file of ids.
//
// There is no tokenizer here (spec 2 owns it): ids in, ids out.
// tools/oracle/tokenize.py is what turns text into an `--ids` file today.
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;

// The bench prompt, cycled to --depth. The same 16 prose ids the runtime tests
// use (tests/runtime/replay_determinism_test.cc); Task 8 is what commits the
// real .ids files, and Task 9 may point the bench at them. What matters for a
// timing run is that the ids are fixed and legal, not what they say.
constexpr uint32_t kBenchPrompt[] = {760, 72103, 506, 37119, 557, 11012, 3213, 310,
                                     6512, 279, 61789, 272, 1072, 2272, 279, 197616};
constexpr size_t kBenchPromptLen = sizeof(kBenchPrompt) / sizeof(kBenchPrompt[0]);

void usage() {
  std::fprintf(
      stderr,
      "usage:\n"
      "  b70-decode <snapshot-or-repo> --ids <file> --n <N> [--device N] [--max-len 16384]\n"
      "  b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]\n"
      "\n"
      "  <snapshot-or-repo>  a snapshot directory, or an HF repo id resolved against the local\n"
      "                      cache ($HF_HOME or ~/.cache/huggingface). Never downloads.\n"
      "  --ids <file>   whitespace-separated token ids (tools/oracle/tokenize.py writes them)\n"
      "  --n <N>        ids to generate, greedily; one per line on stdout, t/s on stderr\n"
      "  --device N     GPU index. Absent: ONEAPI_DEVICE_SELECTOR=level_zero:N, else device 0.\n"
      "  --max-len <L>  KV cache and RoPE capacity (default 16384; the attention kernels are\n"
      "                 compiled per max_len, so only the compiled ones load)\n"
      "  --bench        ingest --depth synthetic ids, then time --tg generated ones and print\n"
      "                 a markdown row on stdout\n");
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

int run(int argc, char** argv) {
  std::string path, ids_path;
  uint32_t n = 0, depth = 4096, tg = 256, max_len = 16384;
  uint32_t device = l0::Context::kFromEnv;
  bool bench = false, have_n = false, have_bench_size = false;

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
    } else if (a == "--depth") {
      depth = parse_u32("--depth", value(i, "--depth"));
      have_bench_size = true;
    } else if (a == "--tg") {
      tg = parse_u32("--tg", value(i, "--tg"));
      have_bench_size = true;
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
  if (bench == !ids_path.empty()) {
    usage();
    throw std::runtime_error("exactly one of --ids and --bench");
  }
  if (bench && have_n)
    throw std::runtime_error("--n belongs to --ids; --bench sizes its run with --depth and --tg");
  if (!bench && have_bench_size)
    throw std::runtime_error("--depth and --tg belong to --bench; --ids sizes its run with --n");
  if (!bench && !have_n) {
    usage();
    throw std::runtime_error("--ids needs --n");
  }
  if (!bench && n == 0) throw std::runtime_error("--n 0 would generate nothing");
  if (bench && tg == 0) throw std::runtime_error("--tg 0 would time nothing");

  // Everything that can be judged without the device or the 19 GB checkpoint is
  // judged first: failing on a typo'd --ids path after a 13-second load is a
  // worse CLI than failing in a millisecond.
  std::vector<uint32_t> ids;
  if (!bench) {
    ids = read_ids(ids_path);
    if (ids.size() + size_t(n) > size_t(max_len))
      throw std::runtime_error("prompt (" + std::to_string(ids.size()) + " ids) + --n " +
                               std::to_string(n) + " exceeds --max-len " + std::to_string(max_len));
  } else {
    if (size_t(depth) + size_t(tg) > size_t(max_len))
      throw std::runtime_error("--depth " + std::to_string(depth) + " + --tg " +
                               std::to_string(tg) + " exceeds --max-len " +
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

  // --bench. The scaffold only in this task: it runs the depth/tg shape and
  // prints the row, but the number that goes into docs/BENCHMARKS.md is Task
  // 9's to measure (3 runs on an idle box, median) and to record.
  eng.generate(tg);
  report_generate(eng, tg);
  std::fprintf(stderr,
               "  wall %.1f ms, fence %.1f ms, host %.1f ms  (host = wall - fence: argument-free\n"
               "  replay, so this is submit + the four bytes of shared memory, nothing else)\n",
               eng.last_gen_ms(), eng.last_fence_ms(), eng.last_gen_ms() - eng.last_fence_ms());

  // The row's identity comes from the environment, not from a configure-time
  // git call: tools/box.sh syncs the tree WITHOUT .git, so the box cannot know
  // its own sha. Whoever runs the bench exports it (Task 9's
  // tools/bench_decode.sh does), and an unset variable prints as `unknown`
  // rather than as a wrong sha in a recorded measurement.
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode %s | %u | %u | %.2f | %.2f |\n", sha, depth, tg, eng.last_tok_per_s(),
              tg ? eng.last_gen_ms() / tg : 0.0);
  std::fprintf(stderr,
               "(measured, one run. docs/BENCHMARKS.md takes the median of three on an idle box -\n"
               " Task 9. B70_GIT_SHA unset prints `unknown`.)\n");
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
