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
  // An object, keys in the order the model wrote them (ordered_json): the response's arguments string keeps that
  // order, so a client echoing it back is rendered as the turn was generated (the session's prefix stays a prefix).
  nlohmann::ordered_json arguments;
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

// Spec 18d: what the server reads a chat request's generated text with - this file's Qwen
// XML OutputStream, or K2-Horizon's (server/toolcall_k2.h); server/chat_format.h picks one
// per model and request.
struct OutputParser {
  virtual ~OutputParser() = default;
  virtual std::vector<Delta> push(const std::string& piece) = 0;  // a detokenised piece
  virtual std::vector<Delta> finish() = 0;                         // end of generation
};

// Incremental form of parse_output: the concatenation of the deltas of any split of
// a text equals parse_output of the whole text (call ids aside, which each stream
// draws itself). Holds back only what may still be a tag.
class OutputStream : public OutputParser {
 public:
  OutputStream(bool thinking, nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;  // a detokenised piece
  std::vector<Delta> finish() override;                         // end of generation

  // Spec 25e (Bolt, spec 23 decision 16's rule): a call still open when generation ends is DROPPED - neither a call nor
  // content - unless its </function> had closed (only </tool_call> missing: the call is complete). Off (the default): the
  // Qwen behaviour, unchanged - an open call whose function tag is complete is emitted with the parameters that closed.
  void set_drop_cut_calls(bool on) { drop_cut_calls_ = on; }

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
  bool drop_cut_calls_ = false;
};

// A call's parameters converted by the tool's schema (a JSON value keeps its key order); exposed for tests.
nlohmann::ordered_json convert_value(const std::string& value, const std::string& function,
                             const std::string& key, const nlohmann::json& tools);

// Shared with the K2 parser (spec 18d): how the tool's schema types `key` of `function` -
// 1 a string (or string | null), 0 another type, -1 no type (no such tool or key, or no
// "type"); and the call ids, "call_" + 24 hex chars from a per-stream seed.
int schema_string_type(const std::string& function, const std::string& key,
                       const nlohmann::json& tools);
uint64_t new_call_seed();
std::string make_call_id(uint64_t seed, uint32_t index);

// A JSON call body, shared by K2's json format (spec 18d, unchanged) and Kolibri-1's hermes calls
// (spec 20e, server/toolcall_kolibri.h): the trimmed `body` is {"name": <string without whitespace>,
// "arguments": <object | a string holding an object | null | absent>}; an argument the tool's schema
// types as a string but written as another JSON value becomes its JSON text. Fills call.name and
// call.arguments (not call.id); false when the body does not parse as such a call.
// A body (or a string-held arguments object) that does not parse is tried once more after
// escape_raw_controls: models write multi-line string arguments (file contents, patches) with RAW
// newlines / tabs inside the string (Kolibri-1, HF discussions #8 / #9) - invalid JSON that would lose
// the call. Nothing outside string literals changes and no quote or brace is guessed; a body that
// still does not parse is not a call (the stream keeps it as content).
bool parse_json_call(const std::string& body, const nlohmann::json& tools, ToolCall& call);

// The JSON text with every raw control byte (< 0x20) INSIDE a string literal escaped (\n, \r, \t, \b,
// \f, else \u00XX), bytes outside string literals copied unchanged; a literal runs from its opening
// quote through backslash escapes to its closing quote. False when nothing was escaped or the text
// ends inside a literal (cut or unbalanced: not repaired). Exposed for tests.
bool escape_raw_controls(const std::string& text, std::string& out);

}  // namespace server
