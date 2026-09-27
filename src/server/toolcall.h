#pragma once

// Qwen XML tool calls and reasoning in the generated text (spec 7 §3.5).
// The checkpoint writes
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>
// (vLLM's qwen3_xml / qwen3_coder format; tools/toolcall/score.py is the reference
// reading). With thinking on the prompt ends in "<think>\n", so the output up to
// "</think>" is reasoning. Pure host code: no engine, no tokenizer.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace server {

struct ToolCall {
  std::string id;            // "call_" + 24 hex chars
  std::string name;
  nlohmann::json arguments;  // object
};

struct ParsedOutput {
  std::string reasoning;  // empty when thinking is off or none was produced
  std::string content;    // text outside reasoning and outside tool calls
  std::vector<ToolCall> tool_calls;
};

// thinking: the prompt ended in "<think>\n". tools: the request's "tools" array or null.
ParsedOutput parse_output(const std::string& text, bool thinking, const nlohmann::json& tools);

struct Delta {
  enum Kind { Reasoning, Content, Call } kind = Content;
  std::string text;    // Reasoning / Content
  ToolCall call;       // Call (complete)
  uint32_t index = 0;  // Call: its position among this output's calls
};

// Incremental form of parse_output: the concatenation of the deltas of any split of
// a text equals parse_output of the whole text (call ids aside, which each stream
// draws itself). Holds back only what may still be a tag.
class OutputStream {
 public:
  OutputStream(bool thinking, nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece);  // a detokenised piece
  std::vector<Delta> finish();                         // end of generation

 private:
  enum class State { ReasoningStart, Reasoning, Body, Call };
  void run(std::vector<Delta>& out, bool final);
  void content(std::vector<Delta>& out, const std::string& text);
  bool close_call(std::vector<Delta>& out, const std::string& body, bool final);

  State state_;
  nlohmann::json tools_;
  std::string buf_;
  std::string held_ws_;     // Reasoning: trailing whitespace; Body: leading ws of a segment
  bool skip_ws_ = false;    // Body: drop whitespace until the next non-space (after a tag)
  bool seg_started_ = false;
  uint32_t calls_ = 0;
  uint64_t seed_;
};

// A call's parameters converted by the tool's schema; exposed for tests.
nlohmann::json convert_value(const std::string& value, const std::string& function,
                             const std::string& key, const nlohmann::json& tools);

}  // namespace server
