#pragma once

// Spec 23e (spec 23 §6.5): Muse-Glimmer-30B's channels and ATEM tool calls in the generated text.
//
// The prompt ends with `<|start|>assistant` (the template's generation prompt, T:209-211), so THE MODEL writes the
// recipient: its output is a sequence of messages, the first without `<|start|>assistant`:
//    to=self<|message|>...reasoning...<|eom|>
//   <|start|>assistant to=fs.read_file<|message|><atem:function_calls>
//   <atem:invoke name="fs.read_file">
//   <atem:parameter name="path">src/main.cc</atem:parameter>
//   </atem:invoke>
//   </atem:function_calls><|eom|>
//   <|start|>assistant to=user<|message|>...the answer...         (<|eot|>: EOS - never pushed by the server)
// The tags are special tokens; the server detokenises with them (tok::Streamer, skip_special false).
//
// The reading (vLLM's selection rule, V/tool_parsers/muse_glimmer_tool_parser.py:_iter_messages - ideas, no code):
//   - a HEADER is `[<|start|>\s*assistant][horizontal whitespace][to=RECIPIENT]<|message|>` (RECIPIENT
//     [A-Za-z0-9_.-]+): found at a `<|message|>`, extended backwards over what may precede it;
//   - a BODY runs from its header to the first `<|eom|>` / `<|eot|>` (consumed), `<|start|>` or the next header's
//     start - so a reasoning block whose <|eom|> is missing does not swallow the tool call after it;
//   - `to=self` bodies are REASONING, `to=user` and recipient-less bodies CONTENT; each body is stripped of ASCII
//     whitespace (transformers' `text` parser strips) and a channel's non-empty bodies are joined with "\n" (spec 23
//     decision 2: nothing the model wrote is dropped - transformers' parse_response keeps only the LAST block);
//     `<atem:invoke>` inside reasoning or content is text, never a call;
//   - every other recipient is a TOOL channel: each complete `<atem:invoke ... name="N">...</atem:invoke>` in it is
//     a call (vLLM's / transformers' patterns: the name from the invoke's `name="..."`, NEVER from the recipient;
//     parameters `<atem:parameter ... name="K" ...>V</atem:parameter>`, the last of a repeated key winning); the rest
//     of a tool body is dropped. The name is normalised (muse_normalize_name: a doubled `ns.ns` with a registered
//     bare `ns` becomes `ns`); the values are typed by the tool's schema FIRST (muse_convert_value, decision 15);
//   - an invoke cut by the end of generation (max_tokens, a stop) is neither a call nor content (decision 16): an
//     agent must never execute a truncated edit; a cut reasoning / content body is kept (stripped);
//   - text outside every body (before the first header, between an <|eom|> and the next header) is dropped - but an
//     output with NO header at all is one content body (the model wrote no framing);
//   - streaming holds back only what may still become a marker or a header: a tail that is a proper prefix of
//     `<|eom|>` / `<|eot|>` / `<|start|>` / `<|message|>` (OVMS's gpt-oss framing idea), a growing ` to=...` tail
//     (vLLM's _OPEN_TAIL_TO_RE idea), a body's trailing whitespace (the strip), and - until a header appears -
//     everything (it may be unframed content). The concatenation of the deltas of any split equals the whole parse
//     (spec 7a's rule; call ids aside, each stream draws its own).
// Pure host code: no engine, no tokenizer.

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/toolcall.h"

namespace server {

class MuseOutputStream : public OutputParser {
 public:
  explicit MuseOutputStream(nlohmann::json tools);
  std::vector<Delta> push(const std::string& piece) override;
  std::vector<Delta> finish() override;

 private:
  enum class Mode { Outside, Body };
  enum class Channel { Reasoning, Content, Tool };
  struct Block {   // a channel's text: bodies stripped and joined with "\n"
    bool any = false;       // the channel has emitted text
    bool started = false;   // this body has emitted text
    std::string held;       // this body's trailing whitespace
  };
  void run(std::vector<Delta>& out, bool final);
  void open_body(const std::string& recipient);
  void body_text(std::vector<Delta>& out, const std::string& text);
  void end_body();
  void tool_calls(std::vector<Delta>& out, size_t end, bool body_done);

  nlohmann::json tools_;
  std::string buf_;
  size_t scanned_ = 0;         // buf_[0, scanned_) holds no marker start
  size_t pending_close_ = 0;   // tool body: an invoke open at buf_'s front, no close starting before this
  Mode mode_ = Mode::Outside;
  Channel channel_ = Channel::Content;
  bool framed_ = false;   // a header has been seen
  Block reasoning_, content_;
  uint32_t calls_ = 0;
  uint64_t seed_;
};

struct MuseParsed {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;
};
// The whole-text parse (MuseOutputStream over one piece).
MuseParsed parse_muse_output(const std::string& text, const nlohmann::json& tools);

// A registered tool's name for an emitted invoke name: unchanged when registered (or no tools); `ns.ns` -> `ns` when
// the bare `ns` is registered (the template renders a bare name's recipient as "ns.*" and the model writes ns.ns -
// vLLM's _normalize_name); anything else unchanged (a unique-leaf match could dispatch the wrong tool).
std::string muse_normalize_name(const std::string& emitted, const nlohmann::json& tools);

// One ATEM parameter's raw text typed by the schema of `function`'s parameter `key` (spec 23 decision 15):
//   - its types (`type` - a string or a list -, `anyOf` / `oneOf`, `nullable`) all "string" (or "null"): the exact
//     text, unstripped ("007", "true", " x " reach the tool unchanged) - except `null` itself when "null" is among
//     them;
//   - other types: JSON-decoded first; a value that is not JSON (or decodes to a string) is then cast type by type
//     as transformers' parse_response(tools=...) casts (integer: "007" -> 7; number; boolean true / 1 / false / 0;
//     null; object / array from JSON); nothing fits: the text (exact when "string" is among the types, else stripped);
//   - no schema (no such tool, key or type): JSON-decoded, else the exact text.
nlohmann::json muse_convert_value(const std::string& raw, const std::string& function, const std::string& key,
                                  const nlohmann::json& tools);

}  // namespace server
