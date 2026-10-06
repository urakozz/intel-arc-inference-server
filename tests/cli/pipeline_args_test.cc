// Spec 16b: b70-decode's --pp / --pipeline-parallel-size flags (cli/pipeline_args.h) on the
// host - the parse and every refusal that needs no device - and the 2026-10-06 rename's
// refusals (cli/renamed_flags.h). The CLI's own registrations (cli_reject_pipeline_*,
// cli_reject_pp_*, cli_reject_renamed_*) drive the shipped binary on the box; this is the same
// logic without it. Spec 16d: and b70-serve's (cli/pipeline_serve.h) - what --pp 2 serves and
// what it refuses before the device (the box's cli_reject_serve_pp_* drive the binary).
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_serve.h"
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
  cli::check_pipeline(one, {true, true, true, true, true, runtime::PrefillBackend::SyclTla, true});
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
  // Spec 16c: --prefill and --bench --prefill-length run the two-card pipeline - on l0 and
  // l0-int8 with the flash attention; sycl-tla and the composed attention are refused.
  for (const runtime::PrefillBackend b : {runtime::PrefillBackend::L0, runtime::PrefillBackend::L0Int8}) {
    c = {};
    c.prefill = true;
    c.prefill_backend = b;
    cli::check_pipeline(two, c);
    c.prefill = false;
    c.bench_prefill = true;
    cli::check_pipeline(two, c);
    c.composed_attn = true;
    CHECK(says([&] { cli::check_pipeline(two, c); }, "flash attention only"));
  }
  c = {};
  c.prefill = true;
  c.prefill_backend = runtime::PrefillBackend::SyclTla;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "not sycl-tla"));
  c.prefill = false;
  c.bench_prefill = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--prefill-backend l0 or l0-int8"));
  c = {};
  c.prefill_backend = runtime::PrefillBackend::SyclTla;   // no prefill: nothing to refuse
  c.composed_attn = true;
  cli::check_pipeline(two, c);
  c = {};
  c.mtp = true;
  cli::check_pipeline(two, c);   // spec 16d lifted 16b's --mtp refusal
  c.prefill = true;
  cli::check_pipeline(two, c);   // ... with the two-card prefill too (the head's KV on device 1)
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

// Spec 16d: b70-serve --pp.
void check_serve() {
  cli::PipelineArgs one, two;
  two.devices = 2;
  cli::ServePipelineContext c;
  // --pp 1 is today's one-card server: nothing refused, whatever the model or flags.
  for (const char* mt : {"qwen3_5", "k2_horizon", "kolibri1", ""}) {
    c = {};
    c.model_type = mt;
    c.device = true;
    c.prefill_replay = true;
    cli::check_serve_pipeline(one, c);
  }
  // ... but the sub-flags alone are refused, as b70-decode refuses them.
  cli::PipelineArgs split_only, handoff_only;
  cli::parse_pipeline_split("30", split_only);
  cli::parse_pipeline_handoff("peer", handoff_only);
  CHECK(says([&] { cli::check_serve_pipeline(split_only, {}); }, "--pipeline-split belongs to --pp 2"));
  CHECK(says([&] { cli::check_serve_pipeline(handoff_only, {}); }, "--pipeline-handoff belongs to --pp 2"));
  // --pp 2 serves the Qwen family with everything else on: MTP, both L0 backends.
  for (const char* mt : {"qwen3_5", "qwen3_5_moe", ""})
    for (runtime::PrefillBackend b : {runtime::PrefillBackend::L0Int8, runtime::PrefillBackend::L0})
      for (bool mtp : {false, true}) {
        c = {};
        c.model_type = mt;
        c.prefill_backend = b;
        c.mtp = mtp;
        cli::check_serve_pipeline(two, c);
      }
  // The refusals, by name.
  c = {};
  c.model_type = "k2_horizon";
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "K2-Horizon runs its own engine"));
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "serve it on one card"));
  c.model_type = "kolibri1";
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "spec 20e"));
  c = {};
  c.device = true;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "--device names one card"));
  c = {};
  c.prefill_backend = runtime::PrefillBackend::SyclTla;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "not sycl-tla"));
  c = {};
  c.composed_attn = true;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "flash attention only"));
  c = {};
  c.prefill_replay = true;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "B70_PREFILL_REPLAY=1"));
  c = {};
  c.prefill_profile = true;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "B70_PREFILL_PROFILE=1"));
  std::printf("b70-serve --pp: --pp 1 untouched, --pp 2 serves the Qwen family, 8 refusals by name\n");
}
}  // namespace

int main() {
  check_parse();
  check_rename();
  check_refusals();
  check_device_refusal();
  check_serve();
  std::printf("pipeline_args_test: OK\n");
  return 0;
}
