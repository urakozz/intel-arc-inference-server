// Spec 24e (plan 24e Task 4): b70-serve's refusals for LFM2.5-24B-A2B, before the device. Host only, header-only code
// (cli/lfm2moe_decode.h): the flags b70-serve parsed (cli::lfm2moe::ServeFlags - run() fills it from its own variables),
// b70-decode's check_args (b70-serve reuses it, with prefill on) and the trained-context cap. Each refusal is a
// runtime_error whose message names what was refused and why; the same refusals through the binary are the box's
// cli_reject_serve_lfm2moe_* tests. Valid combinations must pass.
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "check.h"
#include "cli/lfm2moe_decode.h"
#include "model/lfm2moe.h"

namespace {

using namespace cli::lfm2moe;

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
  const model::Lfm2MoeDesc& d = model::lfm2moe();

  // The flags LFM2.5 has no use for.
  ServeFlags none;
  check_serve_flags(none);
  ServeFlags off;
  off.spec = "off";
  check_serve_flags(off);
  {
    ServeFlags f;
    f.mtp = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "no MTP head"));
  }
  {
    ServeFlags f;
    f.spec = "mtp";
    CHECK(refuses([&] { check_serve_flags(f); }, "no MTP head"));
  }
  {
    ServeFlags f;
    f.spec = "lookup";
    CHECK(refuses([&] { check_serve_flags(f); }, "engine does not capture"));
    CHECK(refuses([&] { check_serve_flags(f); }, "M-row verify lists"));
  }
  {
    ServeFlags f;
    f.spec = "dflash";
    CHECK(refuses([&] { check_serve_flags(f); }, "built for Qwen3.8 beside its drafter"));
  }
  {
    ServeFlags f;
    f.proposer_tuning = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "has none"));
  }
  {
    ServeFlags f;
    f.draft_vocab = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "has neither"));
  }
  {
    ServeFlags f;
    f.qwen4exp_flags = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "belong to Qwen3.8-Flash-Next"));
    CHECK(refuses([&] { check_serve_flags(f); }, "B70_LFM_TOKENIZER_JSON"));
  }
  {
    ServeFlags f;
    f.chat_template = true;
    CHECK(refuses([&] { check_serve_flags(f); }, "Muse-Glimmer-30B only"));
  }

  // b70-decode's refusals, as b70-serve calls them.
  check_args(served(), d);
  {
    DecodeArgs a = served();
    a.pp_backend_given = true;
    a.pp_backend = runtime::PrefillBackend::L0;
    check_args(a, d);   // the l0 backend named is fine
    a.pp_backend = runtime::PrefillBackend::L0Int8;
    CHECK(refuses([&] { check_args(a, d); }, "int8 prefill for this family is a later probe"));
    a.pp_backend = runtime::PrefillBackend::SyclTla;
    CHECK(refuses([&] { check_args(a, d); }, "no LFM2 walk"));
  }
  {
    DecodeArgs a = served();
    a.kv8 = true;
    CHECK(refuses([&] { check_args(a, d); }, "decision 11"));
    a = served();
    a.mtp = true;
    CHECK(refuses([&] { check_args(a, d); }, "no MTP head"));
    a = served();
    a.layers = true;
    CHECK(refuses([&] { check_args(a, d); }, "--layers"));
  }
  {
    DecodeArgs a = served();
    a.pipe.have_split = true;
    a.pipe.split = 3;
    a.pipe.split_auto = false;
    CHECK(refuses([&] { check_args(a, d); }, "--pipeline-split belongs to --pp 2"));
    a = served();
    a.pipe.have_handoff = true;
    CHECK(refuses([&] { check_args(a, d); }, "--pipeline-handoff belongs to --pp 2"));
  }
  {
    DecodeArgs a = served();
    a.pipe.devices = 2;
    check_args(a, d);   // --pp 2, split auto
    a.pipe.split_auto = false;
    a.pipe.split = 20;
    a.pipe.have_split = true;
    check_args(a, d);
    a.pipe.split = 0;
    CHECK(refuses([&] { check_args(a, d); }, "leaves a device no layer"));
    a.pipe.split = 40;
    CHECK(refuses([&] { check_args(a, d); }, "leaves a device no layer"));
    a.pipe.split = 39;
    check_args(a, d);
    a.device = 1;
    CHECK(refuses([&] { check_args(a, d); }, "--device names one card"));
  }

  // The trained context.
  {
    DecodeArgs a = served();
    check_serve_max_len(a, d);   // auto
    a.max_len = cli::MaxLenArg{false, 131072};
    check_serve_max_len(a, d);
    a.max_len = cli::MaxLenArg{false, 131073};
    CHECK(refuses([&] { check_serve_max_len(a, d); }, "exceeds LFM2.5-24B-A2B's trained context 131072"));
    a.max_len = cli::MaxLenArg{false, 262144};
    CHECK(refuses([&] { check_serve_max_len(a, d); }, "trained context"));
  }
  std::printf("lfm2moe_serve_args_test OK\n");
  return 0;
}
