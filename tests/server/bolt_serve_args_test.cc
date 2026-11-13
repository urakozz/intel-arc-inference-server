// Spec 25e (plan 25e Task 4): b70-serve's refusals for Nemotron 3.5 Bolt, before the device. Host only, header-only code: the
// flags b70-serve parsed (cli::bolt::ServeFlags - run() fills it from its own variables), --spec for Bolt (cli::resolve_spec,
// SpecFamily::Bolt - spec 26c), b70-decode's check_args (b70-serve reuses it, with prefill on) and the trained-context cap. Each refusal is a runtime_error whose message names what was refused
// and why; the same refusals through the binary are the box's cli_reject_serve_bolt_* tests. Valid combinations must pass.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/bolt_decode.h"
#include "cli/bolt_serve.h"
#include "cli/spec_args.h"
#include "model/bolt.h"

namespace {

using namespace cli::bolt;

bool refuses(const std::function<void()>& f, const char* expect) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(expect) != std::string::npos) return true;
    std::fprintf(stderr, "refused, but the message lacks '%s': %s\n", expect, e.what());
    return false;
  }
  std::fprintf(stderr, "not refused (wanted '%s')\n", expect);
  return false;
}

DecodeArgs served() {
  DecodeArgs a;
  a.max_len = cli::MaxLenArg{true, 0};   // b70-serve's default: auto
  a.prefill = true;
  return a;
}

}  // namespace

int main() {
  const model::BoltDesc& d = model::bolt();

  // The flags Bolt has no use for.
  ServeFlags none;
  check_serve_flags(none);
  {
    ServeFlags f;
    f.draft_vocab = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "has neither"));
  }

  // Spec 26c: --spec through cli::resolve_spec (SpecFamily::Bolt) - b70-serve's surface for Bolt's proposers.
  using cli::SpecFamily;
  using cli::SpecKind;
  const auto bolt_spec = [](const char* v, bool have_max = false, uint32_t max = 0, bool have_model = false) {
    return cli::resolve_spec(v, SpecFamily::Bolt, "Nemotron 3.5 Bolt", have_max, max, have_model);
  };
  CHECK(bolt_spec("auto").kind == SpecKind::Off);   // b70-serve's default: off for Bolt until box row 65's D5
  CHECK(bolt_spec("off").kind == SpecKind::Off);
  {
    const cli::Spec l = bolt_spec("lookup");
    CHECK(l.kind == SpecKind::Lookup && l.max_k == 7u && l.adaptive);
    const cli::Spec d = bolt_spec("dflash", true, 3, true);
    CHECK(d.kind == SpecKind::DFlash && d.max_k == 3u && d.adaptive);
  }
  CHECK(refuses([&] { bolt_spec("mtp"); }, "no MTP head"));
  CHECK(refuses([&] { bolt_spec("dflash-lookup"); }, "built for Qwen3.8"));
  CHECK(refuses([&] { bolt_spec("lookup", true, 8); }, "expects 1..7"));
  CHECK(refuses([&] { bolt_spec("dflash", true, 0); }, "expects 1..7"));
  CHECK(refuses([&] { bolt_spec("off", true, 2); }, "--spec off has none"));
  CHECK(refuses([&] { bolt_spec("lookup", false, 0, true); }, "--draft-model needs --spec dflash"));

  // b70-decode's refusals, with prefill on (what b70-serve passes).
  check_args(served(), d);   // two cards by default
  {
    DecodeArgs a = served();
    a.pp_backend_given = true;
    a.pp_backend = runtime::PrefillBackend::L0Int8;
    CHECK(refuses([&] { check_args(a, d); }, "int8 prefill for this family is out of scope"));
    a.pp_backend = runtime::PrefillBackend::L0;
    check_args(a, d);
  }
  {
    DecodeArgs a = served();
    a.mtp = true;
    CHECK(refuses([&] { check_args(a, d); }, "no MTP head"));
  }
  {
    DecodeArgs a = served();
    a.spec = true;
    CHECK(refuses([&] { check_args(a, d); }, "--spec"));
  }
  {
    DecodeArgs a = served();
    a.kv8 = true;
    CHECK(refuses([&] { check_args(a, d); }, "Decision 17"));
  }
  {
    DecodeArgs a = served();
    a.pp_given = true;
    a.pipe.devices = 1;
    a.pipe.have_split = true;
    CHECK(refuses([&] { check_args(a, d); }, "--pipeline-split belongs to --parallel pp"));
  }
  {
    DecodeArgs a = served();
    a.pp_given = true;
    a.pipe.devices = 2;
    a.pipe.split_auto = false;
    a.pipe.split = 88;
    CHECK(refuses([&] { check_args(a, d); }, "leaves a device no block"));
    a.pipe.split = 0;
    CHECK(refuses([&] { check_args(a, d); }, "leaves a device no block"));
    a.pipe.split = 48;
    check_args(a, d);
  }

  // The trained context: 262144 is the cap (Decision 14), refused before the device.
  check_serve_max_len(true, 0, d.max_pos);
  check_serve_max_len(false, 262144, d.max_pos);
  check_serve_max_len(false, 16384, d.max_pos);
  CHECK(refuses([&] { check_serve_max_len(false, 262145, d.max_pos); }, "exceeds Nemotron 3.5 Bolt's trained context 262144"));
  CHECK_EQ(d.max_pos, 262144u);

  std::printf("bolt_serve_args_test OK\n");
  return 0;
}
