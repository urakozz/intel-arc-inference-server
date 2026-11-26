#pragma once

// Inkling 2 Mini's reasoning, text and tool calls in the generated text (spec 41f Task 2, spec 41 §1.3).
//
// The prompt ends with the last message's <|end_message|>: the model opens its own blocks. A reply is a list of
// blocks, each opened by <|message_model|> and closed by <|end_message|> (all special tokens, present in the text
// as the server detokenises), and the turn ends with <|content_model_end_sampling|> (id 200006, the EOS):
//   <|message_model|><|content_thinking|>REASONING<|end_message|>
//   <|message_model|><|content_text|>TEXT<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>   (several a turn)
// - the blocks the template writes back for an assistant message (reasoning_content, content, tool_calls), so a turn
// parsed here and sent back re-renders as the generated text.
//
// The reading, whole text and streamed alike (the concatenation of the deltas of any split of a text equals the
// parse of the whole, call ids aside):
//   - a thinking block is reasoning, a text block (or <|content_tool_error|> / <|content_invoke_tool_text|>, which
//     the release's response template reads as content) is content, verbatim; several are concatenated;
//   - a call block: NAME up to <|content_invoke_tool_json|>, then the JSON up to <|end_message|>. The JSON must be an
//     object whose "args" (or, drifted, "arguments") is an object or a string holding one; its "name", when it is a
//     string, names the call, else the block's NAME (no whitespace in either). Raw newlines / tabs inside string
//     arguments are repaired (server::escape_raw_controls), and an argument the tool's schema types as a string but
//     written as another JSON value becomes its JSON text (server::parse_json_call's rules). A body that is not such
//     a call is content: NAME, the marker and the body verbatim;
//   - the drift the model shows (basalt's reading): text outside any block is content as it is; a header followed by
//     another marker (not one of the above) opens a content block holding that marker; a NAME closed by
//     <|end_message|> with no call marker is content; a stray <|content_model_end_sampling|> between blocks is
//     dropped;
//   - a block cut by the end of generation: reasoning or content keeps its text; a call (in its NAME or its JSON) is
//     dropped - neither a call nor content - so a truncated call is never executed.
// Holds back only what may still be a marker. Pure host code: no engine, no tokenizer.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

class Inkling2OutputStream : public OutputParser {
 public:
  explicit Inkling2OutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;

 private:
  enum class State { Between, Header, Text, Name, Call };
  void run(std::vector<Delta>& out, bool final);
  void emit(std::vector<Delta>& out, Delta::Kind kind, std::string text);

  State state_ = State::Between;
  Delta::Kind kind_ = Delta::Content;   // Text: reasoning or content
  nlohmann::json tools_;
  std::string buf_;
  std::string name_;   // Call: the block's NAME
  uint32_t calls_ = 0;
  uint64_t seed_;
};

ParsedOutput parse_output_inkling2(const std::string& text, const nlohmann::json& tools);

}  // namespace server
