#include "server/toolcall_kolibri.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

const std::string kThink = "<think>";
const std::string kThinkEnd = "</think>";
const std::string kCall = "<tool_call>";
const std::string kCallEnd = "</tool_call>";

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

size_t rtrim_end(const std::string& s) {
  size_t e = s.size();
  while (e > 0 && is_ws(s[e - 1])) --e;
  return e;
}

bool starts_at(const std::string& s, size_t at, const std::string& tag) {
  return s.compare(at, tag.size(), tag) == 0;
}

// The text from `at` to the end is a proper prefix of `tag`.
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

}  // namespace

KolibriOutputStream::KolibriOutputStream(json tools) : tools_(std::move(tools)), seed_(new_call_seed()) {}

void KolibriOutputStream::emit(std::vector<Delta>& out, Delta::Kind kind, std::string text) {
  if (text.empty()) return;
  Delta d;
  d.kind = kind;
  d.text = std::move(text);
  out.push_back(std::move(d));
}

// Body text: whitespace after a call skipped, trailing whitespace held (dropped if a call follows,
// emitted with the next non-whitespace byte, or at the end once content was seen). Any split of the
// same text into calls emits the same bytes.
void KolibriOutputStream::content(std::vector<Delta>& out, const std::string& input) {
  std::string text = input;
  if (skip_ws_) {
    size_t b = 0;
    while (b < text.size() && is_ws(text[b])) ++b;
    text.erase(0, b);
    if (text.empty()) return;
    skip_ws_ = false;
  }
  const size_t e = rtrim_end(text);
  if (e == 0) {
    held_ += text;
    return;
  }
  emit(out, Delta::Content, held_ + text.substr(0, e));
  held_ = text.substr(e);
  content_seen_ = true;
}

void KolibriOutputStream::run(std::vector<Delta>& out, bool final) {
  for (;;) {
    switch (state_) {
      case State::Start: {
        // Optional whitespace, then <think> (the model opens its reasoning) or anything else (body).
        size_t i = 0;
        while (i < buf_.size() && is_ws(buf_[i])) ++i;
        if (i == buf_.size()) {
          if (!final) return;
          state_ = State::Body;
          continue;
        }
        if (starts_at(buf_, i, kThink)) {
          buf_.erase(0, i + kThink.size());
          state_ = State::ReasoningStart;
          continue;
        }
        if (!final && prefix_of(buf_, i, kThink)) return;
        state_ = State::Body;
        continue;
      }
      case State::ReasoningStart:
        if (buf_.empty()) {
          if (!final) return;
          state_ = State::Reasoning;
          continue;
        }
        if (buf_.front() == '\n') buf_.erase(0, 1);
        state_ = State::Reasoning;
        continue;
      case State::Reasoning: {
        const size_t i = buf_.find(kThinkEnd);
        if (i != std::string::npos) {
          const std::string part = buf_.substr(0, i);
          const size_t e = rtrim_end(part);
          if (e > 0) emit(out, Delta::Reasoning, held_ + part.substr(0, e));
          held_.clear();
          buf_.erase(0, i + kThinkEnd.size());
          state_ = State::Body;
          skip_ws_ = true;
          continue;
        }
        if (final) {
          const size_t e = rtrim_end(buf_);
          if (e > 0) emit(out, Delta::Reasoning, held_ + buf_.substr(0, e));
          held_.clear();
          buf_.clear();
          return;
        }
        const size_t keep = tag_prefix_suffix(buf_, kThinkEnd);
        const std::string safe = buf_.substr(0, buf_.size() - keep);
        buf_.erase(0, safe.size());
        const size_t e = rtrim_end(safe);
        if (e == 0) {
          held_ += safe;
        } else {
          emit(out, Delta::Reasoning, held_ + safe.substr(0, e));
          held_ = safe.substr(e);
        }
        return;
      }
      case State::Body: {
        const size_t i = buf_.find(kCall);
        if (i != std::string::npos) {
          // The whitespace before the call stays held: dropped if the call parses, content if not.
          content(out, buf_.substr(0, i));
          buf_.erase(0, i + kCall.size());
          state_ = State::Call;
          continue;
        }
        if (final) {
          content(out, buf_);
          buf_.clear();
          if (content_seen_) emit(out, Delta::Content, held_);
          held_.clear();
          return;
        }
        const size_t keep = tag_prefix_suffix(buf_, kCall);
        content(out, buf_.substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        return;
      }
      case State::Call: {
        const size_t j = buf_.find(kCallEnd);
        if (j != std::string::npos) {
          const std::string body = buf_.substr(0, j);
          buf_.erase(0, j + kCallEnd.size());
          ToolCall c;
          if (parse_json_call(body, tools_, c)) {
            held_.clear();
            c.id = make_call_id(seed_, calls_);
            Delta d;
            d.kind = Delta::Call;
            d.call = std::move(c);
            d.index = calls_++;
            out.push_back(std::move(d));
            skip_ws_ = true;
          } else {
            content(out, kCall + body + kCallEnd);   // not a call: verbatim, with its tags
          }
          state_ = State::Body;
          continue;
        }
        if (final) {   // cut by the end of generation: content
          content(out, kCall + buf_);
          buf_.clear();
          state_ = State::Body;
          continue;
        }
        return;
      }
    }
  }
}

std::vector<Delta> KolibriOutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> KolibriOutputStream::finish() {
  std::vector<Delta> out;
  run(out, true);
  return out;
}

ParsedOutput parse_output_kolibri(const std::string& text, const json& tools) {
  KolibriOutputStream stream(tools);
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
