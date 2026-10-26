#pragma once

// Spec 18d: how the server reads a model's chat requests and outputs - the per-model choice
// behind one Server. b70-serve picks it from config.json's model_type (spec 18 §5.1:
// "b70-serve dispatches on model_type"); the request path asks it, never the model name.
//
//   Qwen        Qwen3.8, Agnes 3.0 Flash (and anything not listed): the template reads
//               enable_thinking only; the output opens with reasoning when the prompt ends
//               in "<think>\n"; tool calls are Qwen XML (server/toolcall.h). Exactly the
//               server before spec 18d.
//   K2Horizon   K2-Horizon (model_type k2_horizon): the template also reads the request's
//               chat_template_kwargs (tool_call_format xml | xml_typed | json,
//               reasoning_effort high | medium | low, tool_presentation_format ...); the
//               output opens with reasoning when the prompt ends in <ifm|think>\n,
//               <ifm|think_fast>\n or <ifm|think_faster>\n; tool calls in the request's
//               tool_call_format (server/toolcall_k2.h). EOS is generation_config.json's
//               [1, 250019], read by b70-serve as for every model.
//   Kolibri     Kolibri-1 (model_type kolibri1, spec 20e): ChatML; the template also reads the
//               request's chat_template_kwargs (reasoning_effort none | minimal | low | medium |
//               high | xhigh | max, enable_thinking, preserve_thinking); the MODEL opens its
//               reasoning with <think> (the prompt never ends in it), so the output parser keys
//               reasoning on the output; tool calls are hermes JSON in <tool_call>
//               (server/toolcall_kolibri.h). The template concatenates content as a string, so a
//               message's list of text parts is joined with "\n" first (vLLM's "string" content
//               format). EOS [127906, 127901] and the sampling defaults (generation_config.json:
//               T 1.0, top-p 0.97, top-k 128 - Options::sampling_defaults) are b70-serve's.
//   Lfm2        LFM2.5-24B-A2B (model_type lfm2_moe, spec 24e): ChatML with BOS first; the template reads the request's
//               chat_template_kwargs (preserve_thinking, a bool: keep the reasoning of turns before the last user
//               message); enable_thinking is accepted and ignored (the template always reasons: the generation prompt
//               ends in "<think>", so the output opens inside reasoning - spec 24 decision 8); tool calls are pythonic,
//               <|tool_call_start|>[fn(k=v)]<|tool_call_end|> (server/toolcall_lfm2.h). A message's list of text parts
//               goes to the template as it is (its parse_content concatenates the parts). EOS [124900], the sampling
//               defaults (T 0.1, top-k 50, repetition penalty 1.05 - Options::sampling_defaults) and the host-side
//               repetition penalty are b70-serve's / the engine adapter's.
//   Bolt        Nemotron 3.5 Bolt (model_type nemotron_h_omni over llm_config.model_type nemotron_h_puzzle, spec 25e): ChatML,
//               the checkpoint's chat_template.jinja (no BOS; the system block is always rendered); the template reads the
//               request's chat_template_kwargs (enable_thinking - default true -, medium_effort, truncate_history_thinking -
//               default true -; force_nonempty_content is the OUTPUT parser's; anything else passes through unchecked); the
//               OpenAI top-level reasoning_effort is mapped (spec 25 decision 13): none / minimal -> enable_thinking false,
//               low / medium -> medium_effort true, high (and xhigh) -> the default, an explicit kwarg always wins. The
//               generation prompt ends in "<think>\n" (thinking on: the output OPENS inside reasoning, the `ultra_v3` rules
//               of server/toolcall_bolt.h) or "<think></think>" (thinking off: everything is content); tool calls are Qwen3-Coder
//               XML (the Qwen parser, server/toolcall.h). A message's list of parts goes to the template as it is (its
//               render_content concatenates the text parts). EOS [2, 11] and the sampling defaults (T 1.0, top-p 0.95 -
//               Options::sampling_defaults) are b70-serve's (src/cli/bolt_serve.h).

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

struct ChatFormat {
  enum class Kind { Qwen, K2Horizon, Kolibri, Lfm2, Bolt };
  Kind kind = Kind::Qwen;

  // config.json's model_type -> the format; anything but "k2_horizon", "kolibri1", "lfm2_moe" and "nemotron_h_omni"
  // reads as Qwen.
  static ChatFormat for_model_type(const std::string& model_type);
  static ChatFormat k2_horizon() { return ChatFormat{Kind::K2Horizon}; }
  static ChatFormat kolibri() { return ChatFormat{Kind::Kolibri}; }
  static ChatFormat lfm2() { return ChatFormat{Kind::Lfm2}; }
  static ChatFormat bolt() { return ChatFormat{Kind::Bolt}; }
  const char* name() const;
  // The request's chat_template_kwargs reach the template beside enable_thinking.
  bool template_kwargs() const {
    return kind == Kind::K2Horizon || kind == Kind::Kolibri || kind == Kind::Lfm2 ||
           kind == Kind::Bolt;
  }
  // The template reads every message's content as a string (spec 20e: Kolibri-1).
  bool string_content() const { return kind == Kind::Kolibri; }
};

// The messages as this format's template reads them: with string_content(), a content that is a
// list of text parts becomes their texts joined with "\n" (vLLM's "string" content format); otherwise the messages
// unchanged.
nlohmann::json template_messages(const ChatFormat& format, const nlohmann::json& messages);

// The request's chat_template_kwargs that this format reads, checked before anything runs
// (a bad value is the client's error, a 400): K2's tool_call_format and reasoning_effort;
// Kolibri's reasoning_effort (a string of its seven, or null) and preserve_thinking (a bool); LFM2's preserve_thinking (a bool);
// Bolt's enable_thinking, medium_effort, truncate_history_thinking and
// force_nonempty_content (booleans). Throws std::invalid_argument naming the field.
void check_template_kwargs(const ChatFormat& format, const nlohmann::json& kwargs);

// The kwargs this format's template gets for a request - its chat_template_kwargs, and the OpenAI top-level
// `reasoning_effort` (null: absent) mapped where a format has one. Spec 25e: for Bolt (decision 13) none | minimal -> enable_thinking false, low | medium -> medium_effort true, high | xhigh -> nothing,
// each unless the kwargs name that key already (an explicit kwarg wins); anything else std::invalid_argument naming
// reasoning_effort. Every other format: the kwargs unchanged (reasoning_effort ignored, as before).
nlohmann::json request_template_kwargs(const ChatFormat& format, const nlohmann::json& kwargs,
                                       const nlohmann::json& reasoning_effort);

// The parser of one chat request's output. `prompt` is the rendered prompt (its ending says
// whether the output opens with reasoning), `kwargs` the request's chat_template_kwargs,
// `tools` its tools (they type the call arguments).
std::unique_ptr<OutputParser> make_output_parser(const ChatFormat& format, const std::string& prompt,
                                                 const nlohmann::json& kwargs,
                                                 const nlohmann::json& tools);

}  // namespace server
