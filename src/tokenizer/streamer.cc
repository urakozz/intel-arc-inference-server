#include "tokenizer/streamer.h"

namespace tok {

Streamer::Streamer(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

std::string Streamer::push(uint32_t id) {
  pending_.push_back(id);
  const std::string text = tokenizer_.decode(pending_, false);
  // A terminal U+FFFD marks a multi-byte UTF-8 sequence that is not complete yet.
  if (text.size() >= 3 && text.compare(text.size() - 3, 3, "\xEF\xBF\xBD") == 0) return "";

  std::string output;
  if (text.size() > printed_) output = text.substr(printed_);
  printed_ = text.size();
  // A newline means no earlier byte can change, so the held run is bounded.
  if (text.find('\n') != std::string::npos) {
    pending_.clear();
    printed_ = 0;
  }
  return output;
}

std::string Streamer::flush() {
  const std::string text = pending_.empty() ? std::string() : tokenizer_.decode(pending_, false);
  std::string output = text.size() > printed_ ? text.substr(printed_) : std::string();
  pending_.clear();
  printed_ = 0;
  return output;
}

}  // namespace tok
