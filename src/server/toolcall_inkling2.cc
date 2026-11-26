#include "server/toolcall_inkling2.h"

#include <algorithm>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

const std::string kHeader = "<|message_model|>";
const std::string kThink = "<|content_thinking|>";
const std::string kText = "<|content_text|>";
const std::string kToolError = "<|content_tool_error|>";
const std::string kToolText = "<|content_invoke_tool_text|>";
const std::string kToolJson = "<|content_invoke_tool_json|>";
const std::string kEnd = "<|end_message|>";
const std::string kEndTurn = "<|content_model_end_sampling|>";

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && is_ws(s[b])) ++b;
  while (e > b && is_ws(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// The text from `at` to the end is a proper prefix of `tag`.
bool prefix_of(const std::string& s, size_t at, const std::string& tag) {
  const size_t n = s.size() - at;
  return n < tag.size() && tag.compare(0, n, s, at, n) == 0;
}

// Longest suffix of s that is a proper prefix of any of the tags.
size_t held_back(const std::string& s, std::initializer_list<const std::string*> tags) {
  size_t keep = 0;
  for (const std::string* t : tags) keep = std::max(keep, tag_prefix_suffix(s, *t));
  return keep;
}

// One call block's JSON -> the call, by parse_json_call's rules over {"name": ..., "arguments": ...}.
bool parse_call(const std::string& block_name, const std::string& body, const json& tools, ToolCall& call) {
  ojson obj = ojson::parse(body, nullptr, false);
  if (obj.is_discarded()) {
    std::string repaired;
    if (escape_raw_controls(body, repaired)) obj = ojson::parse(repaired, nullptr, false);
  }
  if (obj.is_discarded() || !obj.is_object()) return false;
  const ojson* args = obj.contains("args") ? &obj.at("args") : obj.contains("arguments") ? &obj.at("arguments") : nullptr;
  if (args == nullptr || !(args->is_object() || args->is_string())) return false;
  const std::string name = obj.contains("name") && obj.at("name").is_string() ? obj.at("name").get<std::string>()
                                                                               : trim(block_name);
  if (name.empty()) return false;
  ojson call_body = ojson::object();
  call_body["name"] = name;
  call_body["arguments"] = *args;
  return parse_json_call(call_body.dump(), tools, call);
}

}  // namespace

Inkling2OutputStream::Inkling2OutputStream(json tools) : tools_(std::move(tools)), seed_(new_call_seed()) {}

void Inkling2OutputStream::emit(std::vector<Delta>& out, Delta::Kind kind, std::string text) {
  if (text.empty()) return;
  if (!out.empty() && out.back().kind == kind && kind != Delta::Call) {
    out.back().text += text;
    return;
  }
  Delta d;
  d.kind = kind;
  d.text = std::move(text);
  out.push_back(std::move(d));
}

void Inkling2OutputStream::run(std::vector<Delta>& out, bool final) {
  for (;;) {
    switch (state_) {
      case State::Between: {
        if (buf_.empty()) return;
        if (buf_.starts_with(kHeader)) {
          buf_.erase(0, kHeader.size());
          state_ = State::Header;
          continue;
        }
        if (buf_.starts_with(kEndTurn)) {   // the turn's end between blocks: not text
          buf_.erase(0, kEndTurn.size());
          continue;
        }
        if (!final && (prefix_of(buf_, 0, kHeader) || prefix_of(buf_, 0, kEndTurn))) return;
        // Text outside a block: content as it is, up to the next header or turn end.
        const size_t h = std::min(buf_.find(kHeader), buf_.find(kEndTurn));
        if (h != std::string::npos) {
          emit(out, Delta::Content, buf_.substr(0, h));
          buf_.erase(0, h);
          continue;
        }
        const size_t keep = final ? 0 : held_back(buf_, {&kHeader, &kEndTurn});
        emit(out, Delta::Content, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Header: {
        bool waiting = false;
        bool opened = false;
        for (const auto& [tag, kind] : {std::pair{&kThink, Delta::Reasoning}, std::pair{&kText, Delta::Content},
                                        std::pair{&kToolError, Delta::Content}, std::pair{&kToolText, Delta::Content}}) {
          if (buf_.starts_with(*tag)) {
            buf_.erase(0, tag->size());
            kind_ = kind;
            state_ = State::Text;
            opened = true;
            break;
          }
          if (prefix_of(buf_, 0, *tag)) waiting = true;
        }
        if (opened) continue;
        if (!final && (buf_.empty() || waiting || prefix_of(buf_, 0, kToolJson) || prefix_of(buf_, 0, kEnd))) return;
        if (buf_.starts_with("<|") && !buf_.starts_with(kToolJson) && !buf_.starts_with(kEnd)) {
          kind_ = Delta::Content;   // another block's marker: a content block that holds it
          state_ = State::Text;
          continue;
        }
        name_.clear();
        state_ = State::Name;
        continue;
      }
      case State::Text: {
        const size_t e = buf_.find(kEnd);
        if (e != std::string::npos) {
          emit(out, kind_, buf_.substr(0, e));
          buf_.erase(0, e + kEnd.size());
          state_ = State::Between;
          continue;
        }
        const size_t keep = final ? 0 : tag_prefix_suffix(buf_, kEnd);
        emit(out, kind_, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Name: {
        const size_t j = buf_.find(kToolJson), e = buf_.find(kEnd);
        if (j != std::string::npos && (e == std::string::npos || j < e)) {
          name_ = buf_.substr(0, j);
          buf_.erase(0, j + kToolJson.size());
          state_ = State::Call;
          continue;
        }
        if (e != std::string::npos) {   // a name with no call: text
          emit(out, Delta::Content, buf_.substr(0, e));
          buf_.erase(0, e + kEnd.size());
          state_ = State::Between;
          continue;
        }
        if (final) {   // cut inside the call's name: dropped
          buf_.clear();
          state_ = State::Between;
        }
        return;
      }
      case State::Call: {
        const size_t e = buf_.find(kEnd);
        if (e == std::string::npos) {
          if (final) {   // cut inside the call: dropped - neither a call nor content
            buf_.clear();
            state_ = State::Between;
          }
          return;
        }
        const std::string body = buf_.substr(0, e);
        buf_.erase(0, e + kEnd.size());
        ToolCall c;
        if (parse_call(name_, body, tools_, c)) {
          c.id = make_call_id(seed_, calls_);
          Delta d;
          d.kind = Delta::Call;
          d.call = std::move(c);
          d.index = calls_++;
          out.push_back(std::move(d));
        } else {
          emit(out, Delta::Content, name_ + kToolJson + body);   // not a call: verbatim
        }
        state_ = State::Between;
        continue;
      }
    }
  }
}

std::vector<Delta> Inkling2OutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> Inkling2OutputStream::finish() {
  std::vector<Delta> out;
  run(out, true);
  return out;
}

ParsedOutput parse_output_inkling2(const std::string& text, const json& tools) {
  Inkling2OutputStream stream(tools);
  return fold_output(stream, text);
}

}  // namespace server
