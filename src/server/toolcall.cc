#include "server/toolcall.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

constexpr const char* kThinkEnd = "</think>";
constexpr const char* kCallBegin = "<tool_call>";
constexpr const char* kCallEnd = "</tool_call>";
constexpr const char* kFunction = "<function=";
constexpr const char* kFunctionEnd = "</function>";
constexpr const char* kParameter = "<parameter=";
constexpr const char* kParameterEnd = "</parameter>";

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

std::string rtrim(const std::string& s) {
  size_t n = s.size();
  while (n > 0 && is_ws(s[n - 1])) --n;
  return s.substr(0, n);
}

std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && is_ws(s[b])) ++b;
  return rtrim(s.substr(b));
}

bool starts_at(const std::string& s, size_t at, const char* tag) {
  return s.compare(at, std::char_traits<char>::length(tag), tag) == 0;
}

// Length of the longest suffix of s that is a proper prefix of tag: text that may
// still become the tag once more pieces arrive.
size_t tag_prefix_suffix(const std::string& s, const std::string& tag) {
  const size_t max = std::min(s.size(), tag.size() - 1);
  for (size_t n = max; n > 0; --n) {
    if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
  }
  return 0;
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t next_seed() {
  static const uint64_t nonce = static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count()) ^
      static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
  static std::atomic<uint64_t> counter{0};
  return splitmix64(nonce + 0x632BE59BD9B4E019ull * ++counter);
}

std::string call_id(uint64_t seed, uint32_t index) {
  const uint64_t a = splitmix64(seed ^ (2ull * index + 1));
  const uint64_t b = splitmix64(a);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "call_%016llx%08llx", static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(b & 0xFFFFFFFFull));
  return buf;
}

const json* schema_type(const std::string& function, const std::string& key, const json& tools) {
  if (!tools.is_array()) return nullptr;
  for (const json& tool : tools) {
    if (!tool.is_object()) continue;
    const json& fn = tool.contains("function") ? tool.at("function") : tool;
    if (!fn.is_object() || !fn.contains("name") || fn.at("name") != function) continue;
    const json* props = nullptr;
    if (fn.contains("parameters") && fn.at("parameters").is_object() &&
        fn.at("parameters").contains("properties")) {
      props = &fn.at("parameters").at("properties");
    }
    if (props == nullptr || !props->is_object() || !props->contains(key)) return nullptr;
    const json& prop = props->at(key);
    if (!prop.is_object() || !prop.contains("type")) return nullptr;
    return &prop.at("type");
  }
  return nullptr;
}

bool is_string_type(const json& type) {
  if (type.is_string()) return type == "string";
  if (!type.is_array() || type.empty()) return false;
  bool has_string = false;
  for (const json& t : type) {
    if (t == "string") has_string = true;
    else if (t != "null") return false;
  }
  return has_string;
}

// A call's text after "<tool_call>" (up to "</tool_call>" or the end of generation).
// False when the function tag is incomplete.
bool parse_call(const std::string& body, const json& tools, ToolCall& call) {
  const size_t f = body.find(kFunction);
  if (f == std::string::npos) return false;
  const size_t name_at = f + std::char_traits<char>::length(kFunction);
  const size_t gt = body.find('>', name_at);
  if (gt == std::string::npos) return false;
  const std::string name = body.substr(name_at, gt - name_at);
  if (name.find('\n') != std::string::npos || trim(name).empty()) return false;
  call.name = trim(name);
  call.arguments = json::object();
  size_t pos = gt + 1;
  for (;;) {
    const size_t p = body.find(kParameter, pos);
    if (p == std::string::npos) break;
    const size_t fend = body.find(kFunctionEnd, pos);
    if (fend != std::string::npos && fend < p) break;
    const size_t key_at = p + std::char_traits<char>::length(kParameter);
    const size_t kgt = body.find('>', key_at);
    if (kgt == std::string::npos) break;
    const std::string key = trim(body.substr(key_at, kgt - key_at));
    const size_t vstart = kgt + 1;
    // The value ends at the first "</parameter>" followed (after whitespace) by the
    // next parameter, the end of the function or the end of the call; any other
    // "</parameter>" is part of the value.
    size_t end = std::string::npos;
    for (size_t c = body.find(kParameterEnd, vstart); c != std::string::npos;
         c = body.find(kParameterEnd, c + 1)) {
      size_t after = c + std::char_traits<char>::length(kParameterEnd);
      while (after < body.size() && is_ws(body[after])) ++after;
      if (after == body.size() || starts_at(body, after, kParameter) ||
          starts_at(body, after, kFunctionEnd)) {
        end = c;
        break;
      }
    }
    if (end == std::string::npos) break;  // cut mid-value: dropped
    std::string value = body.substr(vstart, end - vstart);
    if (!value.empty() && value.front() == '\n') value.erase(0, 1);
    if (!value.empty() && value.back() == '\n') value.pop_back();
    call.arguments[key] = convert_value(value, call.name, key, tools);
    pos = end + std::char_traits<char>::length(kParameterEnd);
  }
  return true;
}

}  // namespace

json convert_value(const std::string& value, const std::string& function, const std::string& key,
                   const json& tools) {
  const json* type = schema_type(function, key, tools);
  if (type != nullptr && is_string_type(*type)) return value;
  json parsed = json::parse(value, nullptr, false);
  if (parsed.is_discarded()) return value;
  return parsed;
}

OutputStream::OutputStream(bool thinking, json tools)
    : state_(thinking ? State::ReasoningStart : State::Body),
      tools_(std::move(tools)),
      seed_(next_seed()) {}

void OutputStream::content(std::vector<Delta>& out, const std::string& input) {
  std::string text = input;
  if (skip_ws_) {
    size_t b = 0;
    while (b < text.size() && is_ws(text[b])) ++b;
    text.erase(0, b);
    if (text.empty()) return;
    skip_ws_ = false;
  }
  if (text.empty()) return;
  if (!seg_started_) {
    // A segment's leading whitespace waits: it is dropped if a call follows directly.
    size_t b = 0;
    while (b < text.size() && is_ws(text[b])) ++b;
    held_ws_ += text.substr(0, b);
    if (b == text.size()) return;
    text = held_ws_ + text.substr(b);
    held_ws_.clear();
    seg_started_ = true;
  }
  Delta d;
  d.kind = Delta::Content;
  d.text = std::move(text);
  out.push_back(std::move(d));
}

bool OutputStream::close_call(std::vector<Delta>& out, const std::string& body, bool) {
  ToolCall call;
  if (!parse_call(body, tools_, call)) return false;
  call.id = call_id(seed_, calls_);
  Delta d;
  d.kind = Delta::Call;
  d.call = std::move(call);
  d.index = calls_++;
  out.push_back(std::move(d));
  return true;
}

void OutputStream::run(std::vector<Delta>& out, bool final) {
  const auto emit_reasoning = [&out](std::string text) {
    if (text.empty()) return;
    Delta d;
    d.kind = Delta::Reasoning;
    d.text = std::move(text);
    out.push_back(std::move(d));
  };
  for (;;) {
    switch (state_) {
      case State::ReasoningStart:
        if (buf_.empty()) {
          if (final) state_ = State::Reasoning;
          else return;
          continue;
        }
        if (buf_.front() == '\n') buf_.erase(0, 1);
        state_ = State::Reasoning;
        continue;
      case State::Reasoning: {
        const size_t i = buf_.find(kThinkEnd);
        if (i != std::string::npos) {
          emit_reasoning(rtrim(held_ws_ + buf_.substr(0, i)));
          held_ws_.clear();
          buf_.erase(0, i + std::char_traits<char>::length(kThinkEnd));
          state_ = State::Body;
          skip_ws_ = true;
          continue;
        }
        if (final) {
          emit_reasoning(rtrim(held_ws_ + buf_));
          held_ws_.clear();
          buf_.clear();
          return;
        }
        const size_t keep = tag_prefix_suffix(buf_, kThinkEnd);
        const std::string safe = buf_.substr(0, buf_.size() - keep);
        buf_.erase(0, safe.size());
        const std::string part = rtrim(safe);
        if (part.empty()) {
          held_ws_ += safe;
        } else {
          emit_reasoning(held_ws_ + part);
          held_ws_ = safe.substr(part.size());
        }
        return;
      }
      case State::Body: {
        const size_t i = buf_.find(kCallBegin);
        if (i != std::string::npos) {
          content(out, buf_.substr(0, i));
          held_ws_.clear();
          seg_started_ = false;
          buf_.erase(0, i + std::char_traits<char>::length(kCallBegin));
          state_ = State::Call;
          continue;
        }
        if (final) {
          content(out, buf_);
          buf_.clear();
          // Whitespace alone: kept only for a text with no calls at all.
          if (!held_ws_.empty() && calls_ == 0 && !skip_ws_) {
            Delta d;
            d.kind = Delta::Content;
            d.text = held_ws_;
            out.push_back(std::move(d));
          }
          held_ws_.clear();
          return;
        }
        const size_t keep = tag_prefix_suffix(buf_, kCallBegin);
        content(out, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Call: {
        // Closed by "</function>" + whitespace + "</tool_call>".
        bool closed = false;
        for (size_t j = buf_.find(kCallEnd); j != std::string::npos; j = buf_.find(kCallEnd, j + 1)) {
          const std::string body = buf_.substr(0, j);
          const std::string tail = rtrim(body);
          const size_t fe = std::char_traits<char>::length(kFunctionEnd);
          if (tail.size() >= fe && tail.compare(tail.size() - fe, fe, kFunctionEnd) == 0) {
            if (!close_call(out, body, true)) content(out, std::string(kCallBegin) + buf_.substr(0, j + 12));
            buf_.erase(0, j + std::char_traits<char>::length(kCallEnd));
            state_ = State::Body;
            skip_ws_ = true;
            seg_started_ = false;
            closed = true;
            break;
          }
        }
        if (closed) continue;
        if (final) {
          if (!close_call(out, buf_, true)) content(out, std::string(kCallBegin) + buf_);
          buf_.clear();
          state_ = State::Body;
          return;
        }
        return;
      }
    }
  }
}

std::vector<Delta> OutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> OutputStream::finish() {
  std::vector<Delta> out;
  run(out, true);
  return out;
}

ParsedOutput parse_output(const std::string& text, bool thinking, const json& tools) {
  OutputStream stream(thinking, tools);
  std::vector<Delta> deltas = stream.push(text);
  std::vector<Delta> tail = stream.finish();
  deltas.insert(deltas.end(), std::make_move_iterator(tail.begin()),
                std::make_move_iterator(tail.end()));
  ParsedOutput out;
  for (Delta& d : deltas) {
    if (d.kind == Delta::Reasoning) out.reasoning += d.text;
    else if (d.kind == Delta::Content) out.content += d.text;
    else out.tool_calls.push_back(std::move(d.call));
  }
  return out;
}

}  // namespace server
