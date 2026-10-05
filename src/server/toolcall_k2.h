#pragma once

// K2-Horizon's tool calls and reasoning in the generated text (spec 18d, spec 18 §5.4).
//
// K2's chat template (render_call_instructions / render_tool_calls_block) asks for one of
// three call syntaxes, chosen by the chat_template_kwargs `tool_call_format`:
//   xml (default)  <ifm|tool_calls>\n<ifm|tool_call>NAME\n
//                  <ifm|arg_key>K</ifm|arg_key>\n<ifm|arg_value>V</ifm|arg_value>\n ...
//                  </ifm|tool_call>\n</ifm|tool_calls>
//   xml_typed      the same, with <ifm|arg_type>T</ifm|arg_type>\n between key and value
//   json           <ifm|tool_calls>\n<ifm|tool_call>{"name": N, "arguments": {...}}</ifm|tool_call>
//                  \n</ifm|tool_calls>
// Strings and scalars are written as plain text, arrays and objects as JSON literals. Every
// tag is one special token (250043/44, 250054-61), present in the text because the server
// detokenises with skip_special off. Not Qwen XML: server/toolcall.h stays Qwen's.
//
// Reasoning: the prompt ends in <ifm|think>\n (or <ifm|think_fast>\n / <ifm|think_faster>\n by
// reasoning_effort), so the output opens with reasoning, closed by the matching close tag or,
// implicitly, by <ifm|tool_calls> - as vLLM's k2_horizon reasoning parser reads it.
//
// The reading, whole text and streamed alike (the concatenation of the deltas of any split of
// a text equals the parse of the whole, call ids aside):
//   - reasoning is verbatim - what the template writes back between the think tags on the
//     next turn, so a client's history re-renders the generated text; an open tag the model
//     repeats at the very start is dropped;
//   - content is verbatim text outside reasoning and the call blocks; content that is only
//     whitespace is dropped (held until a non-whitespace byte arrives, as vLLM's parser does);
//   - each call is emitted at its </ifm|tool_call>; one that does not parse (no name, a
//     malformed argument, invalid JSON) becomes content, verbatim with its tags; non-space text
//     between calls inside a block is content; a bare <ifm|tool_call> outside a block is read
//     as a one-call block;
//   - values: a schema-string argument is verbatim; otherwise the trimmed text as JSON, the
//     verbatim text when that does not parse (server::convert_value's rule). xml_typed's stated
//     type applies where the schema has none ("string" / "any": verbatim). json: a schema-string
//     argument written as another JSON value becomes its JSON text; "arguments" may be an
//     object or a string holding one; a call body that is a JSON object is read as json in
//     any configured format;
//   - cut by the end of generation inside a call: xml with a complete name line is emitted with
//     the arguments that were complete (the Qwen parser's rule); anything else is content.
// Pure host code: no engine, no tokenizer.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

enum class K2CallFormat { Xml, XmlTyped, Json };

// "xml" / "xml_typed" / "json" (the template's names); false for anything else.
bool parse_k2_call_format(const std::string& name, K2CallFormat& out);

class K2OutputStream : public OutputParser {
 public:
  // reasoning_open: the think tag the prompt ended in ("<ifm|think>", "<ifm|think_fast>",
  // "<ifm|think_faster>"; without its "\n"), or "" when the output opens with no reasoning.
  K2OutputStream(std::string reasoning_open, K2CallFormat format, nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;

 private:
  enum class State { ReasoningStart, Reasoning, Body, Block, Call };
  void run(std::vector<Delta>& out, bool final);
  void content(std::vector<Delta>& out, const std::string& text);
  void reasoning(std::vector<Delta>& out, const std::string& text);
  void gap(std::vector<Delta>& out, const std::string& text);
  void call(std::vector<Delta>& out, const std::string& body, bool complete, const std::string& raw);

  State state_;
  std::string open_, close_;
  K2CallFormat format_;
  nlohmann::json tools_;
  std::string buf_;
  std::string held_;           // content whitespace before the first non-whitespace byte
  bool content_seen_ = false;
  bool in_block_ = false;      // the current call sits inside <ifm|tool_calls>
  uint32_t calls_ = 0;
  uint64_t seed_;
};

ParsedOutput parse_output_k2(const std::string& text, const std::string& reasoning_open,
                             K2CallFormat format, const nlohmann::json& tools);

}  // namespace server
