#pragma once
// Spec 21e: what b70-serve sets in server::Options for Qwen3.8-Flash-Next (model_type qwen4_exp) - the request
// path is the Qwen family's, unchanged. Host-only (no Level Zero type): b70-serve's dispatch (cli/qwen4exp_serve.h)
// and tests/server/qwen4exp_server_test.cc call the same function, so the test pins what the server runs.
//
//   - the chat format: server::ChatFormat::for_model_type("qwen4_exp") is Kind::Qwen (anything not listed): the
//     template reads enable_thinking only (Qwen3.8's, byte for byte - sha256 c3cf9e34..., chat::Template's
//     fallback), the output opens with reasoning when the prompt ends in "<think>\n", tool calls are Qwen XML
//     (server::OutputStream). The template also reads `reasoning_effort`, which the Qwen kind does not forward
//     (ChatFormat::template_kwargs() is false) - exactly as for Qwen3.8 today; forwarding chat_template_kwargs for
//     the Qwen kind is a server-wide follow-up for both models, not spec 21e's;
//   - EOS: generation_config.json's [248046 <|im_end|>, 248044 <|endoftext|>] (b70-serve reads it for every model);
//   - the sampling defaults: spec 21 decision 11 is OPEN. Until it is ruled the Qwen family's greedy-unless-asked
//     holds (Options::sampling_defaults unset, server/deps.h's Sampling{}); the ruling flips the one line below to
//     generation_config.json's T 1.0 / top-p 0.95 / top-k 20 (Kolibri's dispatch has the same line), and
//     qwen4exp_server_test pins whichever is ruled.
#include <cstdint>
#include <vector>

#include "server/chat_format.h"
#include "server/server.h"

namespace cli::qwen4exp {

inline constexpr const char* kModelType = "qwen4_exp";

inline void chat_options(server::Options& options, const std::vector<uint32_t>& eos) {
  options.chat_format = server::ChatFormat::for_model_type(kModelType);
  options.eos_ids = eos;
  options.sampling_defaults.reset();   // decision 11's interim: greedy unless the request samples
}

}  // namespace cli::qwen4exp
