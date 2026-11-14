// Spec 16b: the --parallel flag and the pipeline's overrides (cli/pipeline_args.h) on the host -
// the parse and every refusal that needs no device. The CLI's own registrations
// (cli_reject_pipeline_*, cli_reject_pp_*) drive the shipped binary on the box; this is the same
// logic without it. Spec 16d: and b70-serve's (cli/pipeline_serve.h) - what --parallel pp serves
// and what it refuses before the device (the box's cli_reject_serve_pp_* drive the binary). The
// served name (cli/served_name.h), the --spec rule (cli/spec_args.h) and the hardware gate's rule
// (cli/hardware_check.h, on made-up driver reports) are checked here too.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/hardware_check.h"
#include "cli/pipeline_args.h"
#include "cli/pipeline_serve.h"
#include "cli/served_name.h"
#include "cli/spec_args.h"

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
  cli::PipelineArgs p;
  CHECK(p.mode == cli::Parallel::Auto && !p.on() && !p.given());
  cli::parse_parallel("pp", p);
  CHECK(p.mode == cli::Parallel::Pp && p.on() && p.given() && p.devices == 2u);
  cli::parse_parallel("1", p);
  CHECK(p.mode == cli::Parallel::One && !p.on() && p.given());
  cli::parse_parallel("auto", p);
  CHECK(p.mode == cli::Parallel::Auto && !p.given());
  // Spec 17b: tp - tensor parallel, b70-decode's (on() stays false: that is pipeline parallel's flow).
  cli::parse_parallel("tp", p);
  CHECK(p.mode == cli::Parallel::Tp && p.tp() && !p.on() && !p.replica() && p.given() && p.devices == 1u);
  cli::check_tp(p, {});   // plain decode: nothing to refuse
  CHECK(says([&] { cli::check_tp(p, {true, false, false}); }, "--spec dflash / dflash-lookup under tensor parallel are not built"));
  CHECK(says([&] { cli::check_tp(p, {false, true, false}); }, "it has no --parallel tp"));
  CHECK(says([&] { cli::check_tp(p, {false, false, true}); }, "--parallel tp is built for Qwen3.8"));
  CHECK(says([&] { cli::check_tp(p, {false, false, false, true}); }, "--draft-vocab is not built under tensor parallel"));
  CHECK(says([] { cli::require_two_devices_for_tp(1); }, "--parallel tp needs two GPUs and Level Zero shows 1"));
  cli::require_two_devices_for_tp(2);
  {
    cli::PipelineArgs ts = p;
    cli::parse_pipeline_split("20", ts);
    cli::resolve_overrides(ts);
    CHECK(ts.tp());   // an override does not turn tp into pp
    CHECK(says([&] { cli::check_tp(ts, {}); }, "--pipeline-split belongs to --parallel pp"));
  }
  cli::PipelineArgs not_tp;
  cli::check_tp(not_tp, {true, true, true, true});   // not tp: nothing to refuse here
  for (const char* bad : {"0", "2", "3", "", "two", "PP", "TP", "4096"})
    CHECK(says([&] { cli::parse_parallel(bad, p); }, "--parallel expects auto, 1, pp, replica or tp"));
  // Plan 13b: replica - one engine per card, not pipeline parallel (on() stays false).
  cli::parse_parallel("replica", p);
  CHECK(p.mode == cli::Parallel::Replica && p.replica() && !p.on() && p.given() && p.devices == 1u);
  CHECK(says([&] { cli::refuse_replica_in_decode(p); }, "--parallel replica is b70-serve's"));
  CHECK(says([] { cli::check_replica_model("LlamaForCausalLM", "llama"); },
             "--parallel replica serves Qwen3.8 and Ornith only"));
  CHECK(says([] { cli::check_replica_model("", ""); }, "no readable config.json"));
  cli::check_replica_model("Qwen3_5ForConditionalGeneration", "Qwen3.8");     // no throw
  cli::check_replica_model("Qwen3_5MoeForConditionalGeneration", "Ornith");   // no throw: one request a card
  CHECK(says([] { cli::require_two_devices_for_replica(1); }, "--parallel replica needs two GPUs"));
  cli::require_two_devices_for_replica(2);
  cli::check_replica_flags(p, false);
  CHECK(says([&] { cli::check_replica_flags(p, true); }, "--spec dflash-lookup is built for one card"));
  cli::PipelineArgs rs = p;
  cli::parse_pipeline_split("20", rs);
  cli::resolve_overrides(rs);
  CHECK(rs.replica());   // an override does not turn replica into pp
  CHECK(says([&] { cli::check_replica_flags(rs, false); }, "--pipeline-split belongs to --parallel pp"));
  cli::PipelineArgs none;
  cli::refuse_replica_in_decode(none);
  cli::check_replica_flags(none, true);   // not replica: nothing to refuse here
  // An override under auto means pp; under 1 it stays a refusal (check_pipeline below).
  cli::PipelineArgs o;
  cli::parse_pipeline_split("30", o);
  cli::resolve_overrides(o);
  CHECK(o.mode == cli::Parallel::Pp && o.on());
  cli::PipelineArgs o1;
  cli::parse_parallel("1", o1);
  cli::parse_pipeline_split("30", o1);
  cli::resolve_overrides(o1);
  CHECK(o1.mode == cli::Parallel::One && !o1.on());

  cli::PipelineArgs a;
  CHECK(a.split_auto && !a.on());
  cli::parse_pipeline_split("33", a);
  CHECK(a.have_split && !a.split_auto && a.split == 33u);
  cli::parse_pipeline_split("auto", a);
  CHECK(a.split_auto && a.split == 0u);
  CHECK(says([&] { cli::parse_pipeline_split("0", a); }, "--pipeline-split 0"));
  for (const char* bad : {"", "-3", "1e3", "half", "123456"})
    CHECK(says([&] { cli::parse_pipeline_split(bad, a); }, "--pipeline-split expects auto"));
}

// The served name (cli/served_name.h) and the --spec rule (cli/spec_args.h).
void check_names_and_spec() {
  CHECK(cli::served_name("urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ") == "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
  CHECK(cli::served_name("/home/u/.cache/huggingface/hub/models--urakozz--Kolibri-1/snapshots/abc/") ==
        "urakozz/Kolibri-1");
  CHECK(cli::served_name("/data/checkpoints/my-model/") == "my-model");
  CHECK(cli::served_name("./tests/model/k2") == "k2");

  using cli::SpecFamily;
  using cli::SpecKind;
  const cli::Spec q = cli::resolve_spec("auto", SpecFamily::Qwen, "Qwen3.8", false, 0, false);
  CHECK(q.kind == SpecKind::Mtp && q.adaptive && q.max_k == 3u);
  CHECK(cli::resolve_spec("auto", SpecFamily::None, "K2-Horizon", false, 0, false).kind == SpecKind::Off);
  CHECK(cli::resolve_spec("auto", SpecFamily::Qwen4Exp, "Qwen3.8-Flash-Next", false, 0, false).kind == SpecKind::Off);
  const cli::Spec f = cli::resolve_spec("mtp", SpecFamily::Qwen4Exp, "Qwen3.8-Flash-Next", true, 2, false);
  CHECK(f.kind == SpecKind::Mtp && !f.adaptive && f.max_k == 2u);
  const cli::Spec d = cli::resolve_spec("dflash", SpecFamily::Qwen, "Qwen3.8", false, 0, true);
  CHECK(d.kind == SpecKind::DFlash && d.max_k == 7u);
  CHECK(cli::resolve_spec("lookup", SpecFamily::Qwen, "Qwen3.8", true, 1, false).max_k == 1u);
  CHECK(says([] { cli::resolve_spec("mtp", SpecFamily::None, "K2-Horizon", false, 0, false); }, "K2-Horizon has no MTP head"));
  CHECK(says([] { cli::resolve_spec("lookup", SpecFamily::None, "Kolibri-1", false, 0, false); }, "--spec lookup verifies"));
  CHECK(says([] { cli::resolve_spec("dflash", SpecFamily::Qwen4Exp, "Qwen3.8-Flash-Next", false, 0, false); },
             "built for Qwen3.8 beside its drafter"));
  CHECK(says([] { cli::resolve_spec("mtp", SpecFamily::Qwen, "Qwen3.8", true, 4, false); }, "expects 1..3"));
  CHECK(says([] { cli::resolve_spec("dflash", SpecFamily::Qwen, "Qwen3.8", true, 8, false); }, "expects 1..7"));
  CHECK(says([] { cli::resolve_spec("off", SpecFamily::Qwen, "Qwen3.8", true, 2, false); }, "--spec off has none"));
  CHECK(says([] { cli::resolve_spec("mtp", SpecFamily::Qwen, "Qwen3.8", false, 0, true); }, "--draft-model needs --spec dflash"));
  CHECK(says([] { cli::parse_spec("eagle"); }, "--spec expects auto, off, mtp, lookup, dflash or dflash-lookup"));

}

void check_refusals() {
  cli::PipelineArgs one;   // one card (the default) refuses nothing of its own
  cli::check_pipeline(one, {});
  cli::check_pipeline(one, {true, true, true, true});
  cli::PipelineArgs sub = one;
  sub.have_split = true;
  CHECK(says([&] { cli::check_pipeline(sub, {}); }, "--pipeline-split belongs to --parallel pp"));

  cli::PipelineArgs two;
  two.devices = 2;
  cli::check_pipeline(two, {});   // --ids / --bench --depth, nothing else: accepted
  cli::PipelineContext c;
  // Spec 16c: --prefill and --bench --prefill-length run the two-card pipeline (flash attention).
  c = {};
  c.prefill = true;
  cli::check_pipeline(two, c);
  c.prefill = false;
  c.bench_prefill = true;
  cli::check_pipeline(two, c);
  c = {};
  c.mtp = true;
  cli::check_pipeline(two, c);   // spec 16d lifted 16b's --mtp refusal
  c.prefill = true;
  cli::check_pipeline(two, c);   // ... with the two-card prefill too (the head's KV on device 1)
  c = {};
  c.profile = true;
  CHECK(says([&] { cli::check_pipeline(two, c); }, "--profile"));
}

void check_device_refusal() {
  // P4, a missing card: the message names the count and the affinity mask.
  CHECK(says([] { cli::require_two_devices(1); }, "needs two GPUs and Level Zero shows 1"));
  CHECK(says([] { cli::require_two_devices(0); }, "ZE_AFFINITY_MASK"));
  cli::require_two_devices(2);
  cli::require_two_devices(4);
}

// Spec 16d: b70-serve --parallel.
void check_serve() {
  cli::PipelineArgs one, two;
  two.devices = 2;
  cli::ServePipelineContext c;
  // One card: nothing refused, whatever the model or flags.
  for (const char* mt : {"qwen3_5", "k2_horizon", "kolibri1", ""}) {
    c = {};
    c.model_type = mt;
    cli::check_serve_pipeline(one, c);
  }
  // ... but the sub-flag alone is refused, as b70-decode refuses it.
  cli::PipelineArgs split_only;
  cli::parse_pipeline_split("30", split_only);
  CHECK(says([&] { cli::check_serve_pipeline(split_only, {}); }, "--pipeline-split belongs to --parallel pp"));
  // pp serves the Qwen family with everything else on: MTP, either backend.
  for (const char* mt : {"qwen3_5", "qwen3_5_moe", ""})
    for (bool mtp : {false, true}) {
      c = {};
      c.model_type = mt;
      c.mtp = mtp;
      cli::check_serve_pipeline(two, c);
    }
  // The refusals, by name.
  c = {};
  c.model_type = "k2_horizon";
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "K2-Horizon runs its own engine"));
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "serve it on one card"));
  // Spec 20e: Kolibri-1 is served by its own engine (two cards by default) - nothing refused here; its
  // rules are cli::kolibri::check_args's (a --pipeline-split under --parallel auto is Kolibri's default two cards).
  c.model_type = "kolibri1";
  cli::check_serve_pipeline(two, c);
  cli::check_serve_pipeline(split_only, c);
  c = {};
  c.prefill_profile = true;
  CHECK(says([&] { cli::check_serve_pipeline(two, c); }, "B70_PREFILL_PROFILE=1"));
  std::printf("b70-serve --parallel: one card untouched, pp serves the Qwen family, the refusals by name, Kolibri-1 its own\n");
}

// The hardware gate's rule on made-up driver reports: one or two B70s pass, with the smaller card's memory;
// everything else is refused by name.
void check_hardware_rule() {
  const auto b70 = [](size_t bytes = 32530000000ull) {
    l0::GpuInfo g;
    g.vendor_id = 0x8086;
    g.device_id = 0xE223;
    g.name = "Intel(R) Graphics [0xe223]";
    g.xe_cores = 32;
    g.memory_bytes = bytes;
    return g;
  };
  cli::Hardware hw = cli::check_hardware({b70()}, false);
  CHECK(hw.cards == 1 && hw.card_bytes == 32530000000ull);
  hw = cli::check_hardware({b70(32530000000ull), b70(32000000000ull)}, true);
  CHECK(hw.cards == 2 && hw.card_bytes == 32000000000ull);
  CHECK(says([] { cli::check_hardware({}, false); }, "no GPU"));
  CHECK(says([&] { cli::check_hardware({b70(), b70(), b70()}, true); }, "3 GPUs visible"));
  CHECK(says([&] { cli::check_hardware({b70(), b70()}, false); }, "cannot access each other's memory"));
  l0::GpuInfo g = b70();
  g.vendor_id = 0x10DE;
  CHECK(says([&] { cli::check_hardware({g}, false); }, "is not an Intel GPU (vendor 0x10DE)"));
  g = b70();
  g.device_id = 0xE211;
  CHECK(says([&] { cli::check_hardware({g}, false); }, "Intel Arc Pro B60: not yet tested on this box"));
  g.device_id = 0xE222;
  CHECK(says([&] { cli::check_hardware({g}, false); }, "Intel Arc Pro B65: not yet tested on this box"));
  g.device_id = 0x56A0;
  CHECK(says([&] { cli::check_hardware({g}, false); }, "device id 0x56A0) is not a supported card"));
  g = b70();
  g.xe_cores = 20;
  CHECK(says([&] { cli::check_hardware({g}, false); }, "shows 20 Xe-cores"));
  CHECK(says([&] { cli::check_hardware({b70(), g}, true); }, "GPU 1 ("));
  std::printf("hardware gate: one or two B70s pass; no GPU, three, no peer access, another vendor, the B60 / B65, "
              "another device id and another Xe-core count refused by name\n");
}
}  // namespace

// --max-num-seqs (operator 2026-11-11): tp keeps one slot, pp one request, replica 2..8 (2 unasked); auto with N > 1
// is replica on two cards and one card's slots on one; 4 slots a card.
void check_max_num_seqs() {
  cli::PipelineArgs a;
  CHECK_EQ(cli::check_max_num_seqs(a, 1, false), 1u);
  CHECK_EQ(cli::check_max_num_seqs(a, 4, true), 4u);
  cli::PipelineArgs tp;
  cli::parse_parallel("tp", tp);
  CHECK_EQ(cli::check_max_num_seqs(tp, 1, true), 1u);
  CHECK(says([&] { cli::check_max_num_seqs(tp, 2, true); }, "--parallel tp runs one request at a time"));
  cli::PipelineArgs pp;
  cli::parse_parallel("pp", pp);
  CHECK(says([&] { cli::check_max_num_seqs(pp, 2, true); }, "--parallel pp runs one request at a time"));
  cli::PipelineArgs rep;
  cli::parse_parallel("replica", rep);
  CHECK_EQ(cli::check_max_num_seqs(rep, 1, false), 2u);
  CHECK_EQ(cli::check_max_num_seqs(rep, 8, true), 8u);
  CHECK(says([&] { cli::check_max_num_seqs(rep, 1, true); }, "--parallel replica runs a request on each card"));
  // After the hardware gate.
  cli::PipelineArgs two;
  cli::resolve_max_num_seqs(two, 8, 2);
  CHECK(two.replica() && !two.on());
  cli::PipelineArgs one;
  cli::resolve_max_num_seqs(one, 4, 1);
  CHECK(one.mode == cli::Parallel::One && !one.on());
  cli::PipelineArgs auto1;
  cli::resolve_max_num_seqs(auto1, 1, 2);   // N = 1: --parallel auto's own choice later
  CHECK(auto1.mode == cli::Parallel::Auto);
  CHECK(says([&] { cli::PipelineArgs x; cli::resolve_max_num_seqs(x, 5, 1); }, "one card runs at most 4 requests"));
  CHECK(says([&] { cli::PipelineArgs x; cli::resolve_max_num_seqs(x, 9, 2); }, "two cards run at most 8 requests"));
  CHECK(says([&] { cli::PipelineArgs x; cli::parse_parallel("1", x); cli::resolve_max_num_seqs(x, 5, 2); },
             "--parallel replica runs up to 8 on two"));
  // Ornith has no slots: 2 on two cards (replica, or auto before the hardware gate), else 1; Qwen3.8 is not held.
  const std::string orn = "Qwen3_5MoeForConditionalGeneration", q38 = "Qwen3_5ForConditionalGeneration";
  cli::check_one_request_a_card(rep, 2, orn, "Ornith");
  cli::check_one_request_a_card(a, 2, orn, "Ornith");   // auto: replica on two cards, refused after on one
  cli::check_one_request_a_card(rep, 8, q38, "Qwen3.8");
  CHECK(says([&] { cli::check_one_request_a_card(rep, 4, orn, "Ornith"); }, "Ornith runs one request a card"));
  CHECK(says([&] { cli::check_one_request_a_card(a, 3, orn, "Ornith"); }, "--max-num-seqs 3 gives a card several"));
  CHECK(says([&] { cli::check_one_request_a_card(one, 2, orn, "Ornith"); }, "--max-num-seqs 2 on two cards"));
  cli::check_one_request_a_card(one, 1, orn, "Ornith");
}

int main() {
  check_parse();
  check_names_and_spec();
  check_refusals();
  check_device_refusal();
  check_serve();
  check_hardware_rule();
  check_max_num_seqs();
  std::printf("pipeline_args_test: OK\n");
  return 0;
}
