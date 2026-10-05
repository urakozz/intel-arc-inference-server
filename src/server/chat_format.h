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

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

struct ChatFormat {
  enum class Kind { Qwen, K2Horizon };
  Kind kind = Kind::Qwen;

  // config.json's model_type -> the format; anything but "k2_horizon" reads as Qwen.
  static ChatFormat for_model_type(const std::string& model_type);
  static ChatFormat k2_horizon() { return ChatFormat{Kind::K2Horizon}; }
  const char* name() const;
  // The request's chat_template_kwargs reach the template beside enable_thinking.
  bool template_kwargs() const { return kind == Kind::K2Horizon; }
};

// The request's chat_template_kwargs that this format reads, checked before anything runs
// (a bad value is the client's error, a 400): K2's tool_call_format and reasoning_effort.
// Throws std::invalid_argument naming the field.
void check_template_kwargs(const ChatFormat& format, const nlohmann::json& kwargs);

// The parser of one chat request's output. `prompt` is the rendered prompt (its ending says
// whether the output opens with reasoning), `kwargs` the request's chat_template_kwargs,
// `tools` its tools (they type the call arguments).
std::unique_ptr<OutputParser> make_output_parser(const ChatFormat& format, const std::string& prompt,
                                                 const nlohmann::json& kwargs,
                                                 const nlohmann::json& tools);

}  // namespace server
