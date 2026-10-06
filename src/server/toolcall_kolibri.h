#pragma once

// Kolibri-1's reasoning and tool calls in the generated text (spec 20e, spec 20 §4).
//
// Kolibri's chat template (tokenizer_config.json "chat_template") is ChatML with Qwen-style tags,
// and its tool calls are hermes JSON (the vLLM plugin registers Hermes2ProToolParser; spec 20 §10):
//   <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>
// one call per block, several blocks separated by "\n" (the template's render of an assistant turn
// with tool_calls). The tags are added tokens (127907-127910, not special), present in the text as
// the server detokenises. Not Qwen XML (server/toolcall.h) and not K2's tags (server/toolcall_k2.h):
// the tags are Qwen's, the call body K2's json format - server::parse_json_call, shared.
//
// Reasoning is opened BY THE MODEL: with thinking on the prompt ends in "<|im_start|>assistant\n" and
// the output begins "<think>\n...</think>\n\n"; with thinking off (reasoning_effort none, or
// enable_thinking false) the prompt ends in "<think>\n\n</think>\n\n" and the output is the answer.
// So the parser keys reasoning on the OUTPUT, never on the prompt (the Qwen parser keys it on the
// prompt's ending and would put Kolibri's reasoning into content).
//
// The reading, whole text and streamed alike (the concatenation of the deltas of any split of a text
// equals the parse of the whole, call ids aside - spec 7a's rule):
//   - optional leading whitespace, then <think>, opens reasoning: verbatim up to </think>, the newline
//     right after the open tag dropped and the whitespace right before the close tag dropped (the Qwen
//     rule; the template writes reasoning back as "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n");
//     the whitespace after </think> is dropped. Reasoning cut by the end of generation is reasoning.
//     Any other first non-whitespace byte makes the whole output body;
//   - body: text outside calls is content, verbatim, except: whitespace after a call is dropped, a
//     segment's whitespace right before a call is dropped (the template writes "\n" between content
//     and a call and between calls), and content that is only whitespace is dropped (K2's rule);
//   - <tool_call> ... </tool_call> (the first close): the trimmed body through parse_json_call, the
//     call emitted at its close tag; a body that does not parse becomes content verbatim, with its
//     tags and the whitespace before it; a call cut by the end of generation becomes content.
// So a turn rendered by the template parses back to its message (content, reasoning, tool calls),
// and a client that sends a parsed turn back re-renders exactly the generated text - the prefix
// cache's best case. Holds back only what may still be a tag (and trailing whitespace).
// Pure host code: no engine, no tokenizer.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

class KolibriOutputStream : public OutputParser {
 public:
  explicit KolibriOutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;

 private:
  enum class State { Start, ReasoningStart, Reasoning, Body, Call };
  void run(std::vector<Delta>& out, bool final);
  void content(std::vector<Delta>& out, const std::string& text);
  void emit(std::vector<Delta>& out, Delta::Kind kind, std::string text);

  State state_ = State::Start;
  nlohmann::json tools_;
  std::string buf_;
  std::string held_;          // Reasoning / Body: trailing whitespace not yet emitted
  bool skip_ws_ = false;      // Body: drop whitespace until the next non-whitespace byte
  bool content_seen_ = false; // Body: content was emitted (so held whitespace at the end is content)
  uint32_t calls_ = 0;
  uint64_t seed_;
};

ParsedOutput parse_output_kolibri(const std::string& text, const nlohmann::json& tools);

}  // namespace server
