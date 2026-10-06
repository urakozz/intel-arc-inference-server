// Spec 16b: b70-decode's --pp / --pipeline-parallel-size flags (cli/pipeline_args.h) on the
// host - the parse and every refusal that needs no device - and the 2026-10-06 rename's
// refusals (cli/renamed_flags.h). The CLI's own registrations (cli_reject_pipeline_*,
// cli_reject_pp_*, cli_reject_renamed_*) drive the shipped binary on the box; this is the same
// logic without it.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/pipeline_args.h"
#include "cli/renamed_flags.h"

namespace {

// The message of the runtime_error `f` throws ("" when it does not throw).
std::string message(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}
bool says(const std::function<void()>& f, const char* what) {
  const std::string m = message(f);
  if (m.find(what) == std::string::npos) {
    std::fprintf(stderr, "expected a refusal containing '%s', got '%s'\n", what, m.c_str());
    return false;
  }
  return true;
}

void check_parse() {
  for (const char* flag : {"--pp", "--pipeline-parallel-size"}) {
    CHECK_EQ(cli::parse_pipeline_devices(flag, "1"), 1u);
    CHECK_EQ(cli::parse_pipeline_devices(flag, "2"), 2u);
    for (const char* bad : {"0", "3", "4", "", "two", "-1", "02"}) {
      CHECK(says([&] { cli::parse_pipeline_devices(flag, bad); },
                 "is the pipeline-parallel size, 1 or 2"));
      CHECK(says([&] { cli::parse_pipeline_devices(flag, bad); }, flag));
    }
  }

  cli::PipelineArgs a;
  CHECK(a.split_auto && !a.on() && a.handoff == runtime::PpHandoff::Copy);
  cli::parse_pipeline_split("33", a);
  CHECK(a.have_split && !a.split_auto && a.split == 33u);
  cli::parse_pipeline_split("auto", a);
  CHECK(a.split_auto && a.split == 0u);
  CHECK(says([&] { cli::parse_pipeline_split("0", a); }, "--pipeline-split 0"));
  for (const char* bad : {"", "-3", "1e3", "half", "123456"})
    CHECK(says([&] { cli::parse_pipeline_split(bad, a); }, "--pipeline-split expects auto"));

  cli::PipelineArgs h;
  CHECK(cli::parse_pipeline_handoff("peer", h) == runtime::PpHandoff::Peer && h.have_handoff);
  CHECK(cli::parse_pipeline_handoff("copy", h) == runtime::PpHandoff::Copy);
  CHECK(says([&] { cli::parse_pipeline_handoff("p2p", h); }, "--pipeline-handoff expects copy or peer"));
}

// An old command line's `--bench --pp 4096` (the prefill length before 2026-10-06) must not
// become a 4096-way pipeline: refused, naming --prefill-length. The old prefill flags are
// unknown options that name their new spelling.
void check_rename() {
  for (const char* old_len : {"4096", "64", "2048", "512", "0"}) {
    CHECK(says([&] { cli::parse_pipeline_devices("--pp", old_len); },
               "--pp is the pipeline-parallel size, 1 or 2"));
    CHECK(says([&] { cli::parse_pipeline_devices("--pp", old_len); }, "--prefill-length N"));
  }
  CHECK(cli::unknown_option("--pp-chunk") ==
        "unknown option '--pp-chunk' (renamed: --prefill-chunk C)");
  CHECK(cli::unknown_option("--pp-backend") ==
        "unknown option '--pp-backend' (renamed: --prefill-backend B)");
  CHECK(cli::unknown_option("--pipeline") ==
        "unknown option '--pipeline' (renamed: --pp N, or --pipeline-parallel-size N)");
  CHECK(cli::unknown_option("--frobnicate") == "unknown option '--frobnicate'");
}

void check_refusals() {
  cli::PipelineArgs one;   // --pp 1 (the default) refuses nothing of its own
  cli::check_pipeline(one, {});
  cli::check_pipeline(one, {true, true, true, true, true});
  cli::PipelineArgs sub = one;
  sub.have_split = true;
  CHECK(says([&] { cli::check_pipeline(sub, {}); }, "--pipeline-split belongs to --pp 2"));
  sub = one;
  sub.have_handoff = true;
  CHECK(says([&] { cli::check_pipeline(sub, {}); }, "--pipeline-handoff belongs to --pp 2"));

  cli::PipelineArgs two;
  two.devices = 2;
  cli::check_pipeline(two, {});   // --ids / --bench --depth, nothing else: accepted
  cli::PipelineContext c;
  c.prefill = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "spec 16c"));
  c = {};
  c.bench_prefill = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--bench takes --depth, not --prefill-length"));
  c = {};
  c.mtp = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "spec 16d"));
  c = {};
  c.profile = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--profile"));
  c = {};
  c.device = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--device names one card"));
}

void check_device_refusal() {
  // P4, a missing card: the message names the count and the affinity mask.
  CHECK(says([] { cli::require_two_devices(1); }, "needs two GPUs and Level Zero shows 1"));
  CHECK(says([] { cli::require_two_devices(0); }, "ZE_AFFINITY_MASK"));
  cli::require_two_devices(2);
  cli::require_two_devices(4);
}
}  // namespace

int main() {
  check_parse();
  check_rename();
  check_refusals();
  check_device_refusal();
  std::printf("pipeline_args_test: OK\n");
  return 0;
}
