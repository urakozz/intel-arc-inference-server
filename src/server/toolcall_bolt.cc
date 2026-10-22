#include "server/toolcall_bolt.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace server {
namespace {

constexpr const char* kThinkEnd = "</think>";

// Length of the longest suffix of s that is a proper prefix of tag (text that may still become the tag).
size_t tag_prefix_suffix(const std::string& s, const std::string& tag) {
  const size_t max = std::min(s.size(), tag.size() - 1);
  for (size_t n = max; n > 0; --n) {
    if (s.compare(s.size() - n, n, tag, 0, n) == 0) return n;
  }
  return 0;
}

void append(std::vector<Delta>& out, std::vector<Delta>&& more) {
  out.insert(out.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
}

}  // namespace

BoltOutputStream::BoltOutputStream(bool thinking, bool force_nonempty_content, nlohmann::json tools)
    : phase_(thinking ? Phase::Reasoning : Phase::Body),
      force_(thinking && force_nonempty_content),   // thinking off: ultra_v3 returns before it reads the flag
      body_(false, std::move(tools)) {
  body_.set_drop_cut_calls(true);
}

void BoltOutputStream::reasoning(std::vector<Delta>& out, const std::string& text) {
  if (text.empty()) return;
  Delta d;
  d.kind = Delta::Reasoning;
  d.text = text;
  out.push_back(std::move(d));
}

void BoltOutputStream::content(std::vector<Delta>& out, const std::string& text) {
  if (text.empty()) return;
  Delta d;
  d.kind = Delta::Content;
  d.text = text;
  out.push_back(std::move(d));
}

void BoltOutputStream::run(std::vector<Delta>& out, bool final) {
  if (!force_) {
    const size_t i = buf_.find(kThinkEnd);
    if (i != std::string::npos) {
      reasoning(out, buf_.substr(0, i));
      const std::string rest = buf_.substr(i + std::char_traits<char>::length(kThinkEnd));
      buf_.clear();
      phase_ = Phase::Body;
      if (!rest.empty()) append(out, body_.push(rest));
      return;
    }
    if (final) {
      reasoning(out, buf_);
      buf_.clear();
      return;
    }
    // Everything but a tail that may still become "</think>" is reasoning, verbatim.
    const size_t keep = tag_prefix_suffix(buf_, kThinkEnd);
    reasoning(out, buf_.substr(0, buf_.size() - keep));
    buf_.erase(0, buf_.size() - keep);
    return;
  }
  // force_nonempty_content: the reasoning is held until it is known to have content behind it.
  if (!closed_) {
    const size_t tag = std::char_traits<char>::length(kThinkEnd);
    const size_t i = buf_.find(kThinkEnd, scanned_);
    scanned_ = buf_.size() >= tag - 1 ? buf_.size() - (tag - 1) : 0;   // a tag may still start in the last 7 bytes
    if (i != std::string::npos) {
      closed_ = true;
      reasoning_len_ = i;
      buf_.erase(i, std::char_traits<char>::length(kThinkEnd));   // buf_ = reasoning, then what follows the tag
    }
  }
  if (closed_ && buf_.size() > reasoning_len_) {   // a byte after </think>: there is content, the reasoning stays
    reasoning(out, buf_.substr(0, reasoning_len_));
    const std::string rest = buf_.substr(reasoning_len_);
    buf_.clear();
    phase_ = Phase::Body;
    append(out, body_.push(rest));
    return;
  }
  if (final) {   // never closed, or closed with nothing after: the reasoning is the content
    content(out, buf_);
    buf_.clear();
  }
}

std::vector<Delta> BoltOutputStream::push(const std::string& piece) {
  std::vector<Delta> out;
  if (phase_ == Phase::Body) {
    append(out, body_.push(piece));
    return out;
  }
  buf_ += piece;
  run(out, false);
  return out;
}

std::vector<Delta> BoltOutputStream::finish() {
  std::vector<Delta> out;
  if (phase_ == Phase::Body) {
    append(out, body_.finish());
    return out;
  }
  run(out, true);
  return out;
}

ParsedOutput parse_bolt_output(const std::string& text, bool thinking, bool force_nonempty_content,
                               const nlohmann::json& tools) {
  BoltOutputStream stream(thinking, force_nonempty_content, tools);
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
