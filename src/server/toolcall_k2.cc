#include "server/toolcall_k2.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

const std::string kBlock = "<ifm|tool_calls>";
const std::string kBlockEnd = "</ifm|tool_calls>";
const std::string kCall = "<ifm|tool_call>";
const std::string kCallEnd = "</ifm|tool_call>";
const std::string kArgKey = "<ifm|arg_key>";
const std::string kArgKeyEnd = "</ifm|arg_key>";
const std::string kArgType = "<ifm|arg_type>";
const std::string kArgTypeEnd = "</ifm|arg_type>";
const std::string kArgValue = "<ifm|arg_value>";
const std::string kArgValueEnd = "</ifm|arg_value>";

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

bool all_ws(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](char c) { return is_ws(c); });
}

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && is_ws(s[b])) ++b;
  while (e > b && is_ws(s[e - 1])) --e;
  return s.substr(b, e - b);
}

size_t skip_ws(const std::string& s, size_t at) {
  while (at < s.size() && is_ws(s[at])) ++at;
  return at;
}

bool starts_at(const std::string& s, size_t at, const std::string& tag) {
  return s.compare(at, tag.size(), tag) == 0;
}

// The text from `at` to the end is a proper prefix of `tag` (what a cut may leave).
bool prefix_of(const std::string& s, size_t at, const std::string& tag) {
  const size_t n = s.size() - at;
  return n < tag.size() && tag.compare(0, n, s, at, n) == 0;
}

// Length of the longest suffix of s that is a proper prefix of tag.
size_t tag_prefix_suffix(const std::string& s, const std::string& tag) {
  const size_t max = std::min(s.size(), tag.size() - 1);
  for (size_t n = max; n > 0; --n) {
    if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
  }
  return 0;
}

bool valid_name(const std::string& name) {
  return !name.empty() && std::none_of(name.begin(), name.end(), [](char c) { return is_ws(c); });
}

// The type xml_typed states, as vLLM reads it: "array[object]" -> "array", lower case.
std::string base_type(const std::string& stated) {
  std::string t = trim(stated.substr(0, stated.find('[')));
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return t;
}

json convert(const std::string& value, const std::string& stated, const std::string& function,
             const std::string& key, const json& tools) {
  const int schema = schema_string_type(function, key, tools);
  if (schema == 1) return value;
  if (schema == -1 && !stated.empty()) {
    const std::string t = base_type(stated);
    if (t == "string" || t == "any") return value;
  }
  json parsed = json::parse(trim(value), nullptr, false);
  if (parsed.is_discarded()) return value;
  return parsed;
}

// The text of a call after <ifm|tool_call>, up to </ifm|tool_call> (complete) or the end of
// generation (not complete). False: not a call (the caller writes it out as content).
bool parse_xml(const std::string& body, bool complete, const json& tools, ToolCall& call) {
  size_t name_end = std::min(body.find('\n'), body.find(kArgKey));
  if (name_end == std::string::npos) {
    if (!complete) return false;   // the name line is not finished
    name_end = body.size();        // a call with no arguments: NAME</ifm|tool_call>
  }
  call.name = trim(body.substr(0, name_end));
  if (!valid_name(call.name)) return false;
  call.arguments = json::object();
  size_t pos = name_end;
  for (;;) {
    pos = skip_ws(body, pos);
    if (pos == body.size()) return true;
    // Cut inside an argument: keep the complete ones. Malformed with the call closed: not a call.
    const auto cut = [&](bool truncated_here) { return !complete && truncated_here; };
    if (!starts_at(body, pos, kArgKey)) return cut(prefix_of(body, pos, kArgKey));
    const size_t key_at = pos + kArgKey.size();
    const size_t key_end = body.find(kArgKeyEnd, key_at);
    if (key_end == std::string::npos) return cut(true);
    const std::string key = trim(body.substr(key_at, key_end - key_at));
    if (key.empty()) return false;
    pos = skip_ws(body, key_end + kArgKeyEnd.size());
    std::string stated;
    if (starts_at(body, pos, kArgType)) {
      const size_t type_at = pos + kArgType.size();
      const size_t type_end = body.find(kArgTypeEnd, type_at);
      if (type_end == std::string::npos) return cut(true);
      stated = trim(body.substr(type_at, type_end - type_at));
      pos = skip_ws(body, type_end + kArgTypeEnd.size());
    }
    if (!starts_at(body, pos, kArgValue)) {
      return cut(pos == body.size() || prefix_of(body, pos, kArgValue) || prefix_of(body, pos, kArgType));
    }
    const size_t value_at = pos + kArgValue.size();
    // The value ends at the first </ifm|arg_value> followed (after whitespace) by the next
    // argument or the end of the call; any other is part of the value.
    size_t value_end = std::string::npos;
    for (size_t c = body.find(kArgValueEnd, value_at); c != std::string::npos;
         c = body.find(kArgValueEnd, c + 1)) {
      const size_t after = skip_ws(body, c + kArgValueEnd.size());
      if (after == body.size() || starts_at(body, after, kArgKey)) {
        value_end = c;
        break;
      }
    }
    if (value_end == std::string::npos) return cut(true);
    call.arguments[key] = convert(body.substr(value_at, value_end - value_at), stated, call.name, key, tools);
    pos = value_end + kArgValueEnd.size();
  }
}

bool parse_json(const std::string& body, const json& tools, ToolCall& call) {
  const json j = json::parse(trim(body), nullptr, false);
  if (j.is_discarded() || !j.is_object() || !j.contains("name") || !j.at("name").is_string()) return false;
  call.name = j.at("name").get<std::string>();
  if (!valid_name(call.name)) return false;
  json args = j.contains("arguments") ? j.at("arguments") : json::object();
  if (args.is_string()) args = json::parse(args.get<std::string>(), nullptr, false);
  if (args.is_null()) args = json::object();
  if (args.is_discarded() || !args.is_object()) return false;
  for (auto& [key, value] : args.items()) {
    if (!value.is_string() && schema_string_type(call.name, key, tools) == 1) value = value.dump();
  }
  call.arguments = std::move(args);
  return true;
}

}  // namespace

bool parse_k2_call_format(const std::string& name, K2CallFormat& out) {
  if (name == "xml") out = K2CallFormat::Xml;
  else if (name == "xml_typed") out = K2CallFormat::XmlTyped;
  else if (name == "json") out = K2CallFormat::Json;
  else return false;
  return true;
}

K2OutputStream::K2OutputStream(std::string reasoning_open, K2CallFormat format, json tools)
    : state_(reasoning_open.empty() ? State::Body : State::ReasoningStart),
      open_(std::move(reasoning_open)),
      format_(format),
      tools_(std::move(tools)),
      seed_(new_call_seed()) {
  if (!open_.empty()) close_ = "</" + open_.substr(1);   // <ifm|think> -> </ifm|think>
}

void K2OutputStream::content(std::vector<Delta>& out, const std::string& input) {
  if (input.empty()) return;
  std::string text = input;
  if (!content_seen_) {
    if (all_ws(text)) {
      held_ += text;
      return;
    }
    text = held_ + text;
    held_.clear();
    content_seen_ = true;
  }
  Delta d;
  d.kind = Delta::Content;
  d.text = std::move(text);
  out.push_back(std::move(d));
}

void K2OutputStream::reasoning(std::vector<Delta>& out, const std::string& text) {
  if (text.empty()) return;
  Delta d;
  d.kind = Delta::Reasoning;
  d.text = text;
  out.push_back(std::move(d));
}

void K2OutputStream::gap(std::vector<Delta>& out, const std::string& text) {
  if (!all_ws(text)) content(out, text);
}

void K2OutputStream::call(std::vector<Delta>& out, const std::string& body, bool complete,
                          const std::string& raw) {
  ToolCall c;
  // A JSON body is read as JSON whatever the configured format (the model wrote that one).
  const std::string trimmed = trim(body);
  const bool json_body = format_ == K2CallFormat::Json || (!trimmed.empty() && trimmed.front() == '{');
  const bool ok = json_body ? complete && parse_json(body, tools_, c) : parse_xml(body, complete, tools_, c);
  if (!ok) {
    content(out, raw);
    return;
  }
  c.id = make_call_id(seed_, calls_);
  Delta d;
  d.kind = Delta::Call;
  d.call = std::move(c);
  d.index = calls_++;
  out.push_back(std::move(d));
}

void K2OutputStream::run(std::vector<Delta>& out, bool final) {
  for (;;) {
    switch (state_) {
      case State::ReasoningStart:
        if (starts_at(buf_, 0, open_)) {
          buf_.erase(0, open_.size());
        } else if (!final && prefix_of(buf_, 0, open_)) {
          return;   // may still become the repeated open tag
        }
        state_ = State::Reasoning;
        continue;
      case State::Reasoning: {
        const size_t end = buf_.find(close_);
        const size_t block = buf_.find(kBlock);
        if (end != std::string::npos && (block == std::string::npos || end < block)) {
          reasoning(out, buf_.substr(0, end));
          buf_.erase(0, end + close_.size());
          state_ = State::Body;
          continue;
        }
        if (block != std::string::npos) {   // a call block ends reasoning; the block stays
          reasoning(out, buf_.substr(0, block));
          buf_.erase(0, block);
          state_ = State::Body;
          continue;
        }
        if (final) {
          reasoning(out, buf_);
          buf_.clear();
          return;
        }
        const size_t keep = std::max(tag_prefix_suffix(buf_, close_), tag_prefix_suffix(buf_, kBlock));
        reasoning(out, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Body: {
        const size_t block = buf_.find(kBlock);
        const size_t bare = buf_.find(kCall);
        if (block != std::string::npos && (bare == std::string::npos || block < bare)) {
          content(out, buf_.substr(0, block));
          buf_.erase(0, block + kBlock.size());
          in_block_ = true;
          state_ = State::Block;
          continue;
        }
        if (bare != std::string::npos) {
          content(out, buf_.substr(0, bare));
          buf_.erase(0, bare + kCall.size());
          in_block_ = false;
          state_ = State::Call;
          continue;
        }
        if (final) {
          content(out, buf_);
          buf_.clear();
          return;
        }
        const size_t keep = std::max(tag_prefix_suffix(buf_, kBlock), tag_prefix_suffix(buf_, kCall));
        content(out, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Block: {
        // Between calls: whitespace, the next call or the block's end. The gap is decided
        // once a tag (or the end of generation) arrives, so it is held until then.
        const size_t next = buf_.find(kCall);
        const size_t end = buf_.find(kBlockEnd);
        if (next != std::string::npos && (end == std::string::npos || next < end)) {
          gap(out, buf_.substr(0, next));
          buf_.erase(0, next + kCall.size());
          state_ = State::Call;
          continue;
        }
        if (end != std::string::npos) {
          gap(out, buf_.substr(0, end));
          buf_.erase(0, end + kBlockEnd.size());
          in_block_ = false;
          state_ = State::Body;
          continue;
        }
        if (final) {
          gap(out, buf_);
          buf_.clear();
        }
        return;
      }
      case State::Call: {
        // Closed by the first </ifm|tool_call> after a complete argument (xml) - or after
        // the name alone - or by the first one at all (json).
        bool closed = false;
        for (size_t j = buf_.find(kCallEnd); j != std::string::npos; j = buf_.find(kCallEnd, j + 1)) {
          const std::string body = buf_.substr(0, j);
          const std::string tail = trim(body);
          const bool closes = format_ == K2CallFormat::Json || body.find(kArgKey) == std::string::npos ||
                              (tail.size() >= kArgValueEnd.size() &&
                               tail.compare(tail.size() - kArgValueEnd.size(), kArgValueEnd.size(), kArgValueEnd) == 0);
          if (!closes) continue;
          call(out, body, true, kCall + body + kCallEnd);
          buf_.erase(0, j + kCallEnd.size());
          state_ = in_block_ ? State::Block : State::Body;
          closed = true;
          break;
        }
        if (closed) continue;
        if (final) {
          call(out, buf_, false, kCall + buf_);
          buf_.clear();
          state_ = State::Body;
        }
        return;
      }
    }
  }
}

std::vector<Delta> K2OutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> K2OutputStream::finish() {
  std::vector<Delta> out;
  run(out, true);
  return out;
}

ParsedOutput parse_output_k2(const std::string& text, const std::string& reasoning_open,
                             K2CallFormat format, const json& tools) {
  K2OutputStream stream(reasoning_open, format, tools);
  std::vector<Delta> deltas = stream.push(text);
  std::vector<Delta> tail = stream.finish();
  deltas.insert(deltas.end(), std::make_move_iterator(tail.begin()), std::make_move_iterator(tail.end()));
  ParsedOutput out;
  for (Delta& d : deltas) {
    if (d.kind == Delta::Reasoning) out.reasoning += d.text;
    else if (d.kind == Delta::Content) out.content += d.text;
    else out.tool_calls.push_back(std::move(d.call));
  }
  return out;
}

}  // namespace server
