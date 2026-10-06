// Spec 16b: b70-decode's --pipeline flags (cli/pipeline_args.h) on the host - the parse and
// every refusal that needs no device. The CLI's own registrations (cli_reject_pipeline_*)
// drive the shipped binary on the box; this is the same logic without it.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/pipeline_args.h"

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
  CHECK_EQ(cli::parse_pipeline_devices("1"), 1u);
  CHECK_EQ(cli::parse_pipeline_devices("2"), 2u);
  for (const char* bad : {"0", "3", "4", "", "two", "-1", "02"})
    CHECK(says([&] { cli::parse_pipeline_devices(bad); }, "--pipeline expects 1 or 2"));

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

void check_refusals() {
  cli::PipelineArgs one;   // --pipeline 1 (the default) refuses nothing of its own
  cli::check_pipeline(one, {});
  cli::check_pipeline(one, {true, true, true, true, true});
  cli::PipelineArgs sub = one;
  sub.have_split = true;
  CHECK(says([&] { cli::check_pipeline(sub, {}); }, "--pipeline-split belongs to --pipeline 2"));
  sub = one;
  sub.have_handoff = true;
  CHECK(says([&] { cli::check_pipeline(sub, {}); }, "--pipeline-handoff belongs to --pipeline 2"));

  cli::PipelineArgs two;
  two.devices = 2;
  cli::check_pipeline(two, {});   // --ids / --bench --depth, nothing else: accepted
  cli::PipelineContext c;
  c.prefill = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "spec 16c"));
  c = {};
  c.bench_prefill = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--bench takes --depth, not --pp"));
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
  check_refusals();
  check_device_refusal();
  std::printf("pipeline_args_test: OK\n");
  return 0;
}
