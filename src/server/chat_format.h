#pragma once

// How the server reads a model's chat requests and outputs - the per-model choice
// behind one Server. b70-serve picks it from config.json's model_type; the request path asks
// it, never the model name.
//
//   Qwen        Qwen3.8 (and anything not listed): the template reads
//               enable_thinking only; the output opens with reasoning when the prompt ends
//               in "<think>\n"; tool calls are Qwen XML (server/toolcall.h).
//   Kolibri     Kolibri-1 (model_type kolibri1, spec 20e): ChatML; the template also reads the
//               request's chat_template_kwargs (reasoning_effort none | minimal | low | medium |
//               high | xhigh | max, enable_thinking, preserve_thinking); the MODEL opens its
//               reasoning with <think> (the prompt never ends in it), so the output parser keys
//               reasoning on the output; tool calls are hermes JSON in <tool_call>
//               (server/toolcall_kolibri.h). The template concatenates content as a string, so a
//               message's list of text parts is joined with "\n" first (vLLM's "string" content
//               format). EOS [127906, 127901] and the sampling defaults (generation_config.json:
//               T 1.0, top-p 0.97, top-k 128 - Options::sampling_defaults) are b70-serve's.

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

struct ChatFormat {
  enum class Kind { Qwen, Kolibri };
  Kind kind = Kind::Qwen;

  // config.json's model_type -> the format; anything but "kolibri1" reads as Qwen.
  static ChatFormat for_model_type(const std::string& model_type);
  static ChatFormat kolibri() { return ChatFormat{Kind::Kolibri}; }
  const char* name() const;
  // The request's chat_template_kwargs reach the template beside enable_thinking.
  bool template_kwargs() const { return kind == Kind::Kolibri; }
  // The template reads every message's content as a string (spec 20e: Kolibri-1).
  bool string_content() const { return kind == Kind::Kolibri; }
};

// The messages as this format's template reads them: with string_content(), a content that is a
// list of text parts becomes their texts joined with "\n" (vLLM's "string" content format); otherwise the messages
// unchanged.
nlohmann::ordered_json template_messages(const ChatFormat& format, const nlohmann::ordered_json& messages);

// The request's chat_template_kwargs that this format reads, checked before anything runs
// (a bad value is the client's error, a 400): Kolibri's reasoning_effort (a string of its seven, or null) and preserve_thinking (a bool).
// Throws std::invalid_argument naming the field.
void check_template_kwargs(const ChatFormat& format, const nlohmann::json& kwargs);

// The kwargs this format's template gets for a request - its chat_template_kwargs, and the OpenAI top-level
// `reasoning_effort` (null: absent) mapped where a format has one. Kolibri-1: the template's own `reasoning_effort` variable (one of its seven efforts, else
// std::invalid_argument), unless the kwargs name it. Every other format: the kwargs unchanged (reasoning_effort
// ignored, as before).
nlohmann::json request_template_kwargs(const ChatFormat& format, const nlohmann::json& kwargs,
                                       const nlohmann::json& reasoning_effort);

// The parser of one chat request's output. `prompt` is the rendered prompt (its ending says
// whether the output opens with reasoning), `kwargs` the request's chat_template_kwargs,
// `tools` its tools (they type the call arguments).
std::unique_ptr<OutputParser> make_output_parser(const ChatFormat& format, const std::string& prompt,
                                                 const nlohmann::json& kwargs,
                                                 const nlohmann::json& tools);

}  // namespace server
