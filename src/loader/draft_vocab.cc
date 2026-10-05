#include "loader/draft_vocab.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "common/json.h"

namespace loader {

bool parse_draft_vocab(const std::string& s, uint32_t& size) {
  if (s == "off") { size = 0; return true; }
  if (s == "32k") { size = 32768; return true; }
  if (s == "64k") { size = 65536; return true; }
  if (s == "128k") { size = 131072; return true; }
  return false;
}

std::string draft_vocab_name(uint32_t size) {
  switch (size) {
    case 0: return "off";
    case 32768: return "32k";
    case 65536: return "64k";
    case 131072: return "128k";
    default: return std::to_string(size);
  }
}

std::vector<uint32_t> select_draft_vocab(const std::vector<uint32_t>& added,
                                         const std::vector<uint32_t>& eos,
                                         const std::vector<uint32_t>& ranked, uint32_t vocab_used,
                                         uint32_t size, DraftVocabCounts* counts) {
  if (size == 0) throw std::invalid_argument("select_draft_vocab: |V'| = 0 (off is not a vocabulary)");
  if (size > vocab_used)
    throw std::invalid_argument("select_draft_vocab: |V'| " + std::to_string(size) +
                                " exceeds the " + std::to_string(vocab_used) + " usable ids");
  std::vector<uint8_t> taken(vocab_used, 0);
  std::vector<uint32_t> out;
  out.reserve(size);
  auto take = [&](uint32_t id) {
    if (id >= vocab_used || taken[id]) return false;   // masked by argmax, or already in
    taken[id] = 1;
    out.push_back(id);
    return true;
  };
  DraftVocabCounts c;
  for (uint32_t id : added) c.forced += take(id);
  for (uint32_t id : eos) c.forced += take(id);
  if (out.size() > size)
    throw std::invalid_argument("select_draft_vocab: the " + std::to_string(out.size()) +
                                " added tokens and EOS ids alone exceed |V'| " +
                                std::to_string(size));
  for (size_t i = 0; i < ranked.size() && out.size() < size; ++i) c.ranked += take(ranked[i]);
  // Terminates: size <= vocab_used, so enough untaken ids below vocab_used remain.
  for (uint32_t id = 0; out.size() < size; ++id) c.lowest += take(id);
  std::sort(out.begin(), out.end());
  if (counts) *counts = c;
  return out;
}

std::vector<uint32_t> parse_ranked_ids(const std::string& text, const std::string& what) {
  std::vector<uint32_t> ids;
  std::istringstream lines(text);
  std::string line;
  size_t line_no = 0;
  while (std::getline(lines, line)) {
    ++line_no;
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream words(line);
    std::string w;
    while (words >> w) {
      if (w.size() > 10 || w.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(what + ":" + std::to_string(line_no) + ": '" + w +
                                 "' is not a token id");
      const unsigned long long v = std::stoull(w);
      if (v > 0xFFFFFFFFull)
        throw std::runtime_error(what + ":" + std::to_string(line_no) + ": id " + w +
                                 " is out of range");
      ids.push_back(static_cast<uint32_t>(v));
    }
  }
  if (ids.empty()) throw std::runtime_error(what + ": no ids");
  return ids;
}

std::vector<uint32_t> read_ranked_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --draft-vocab-ids file '" + path + "'");
  std::stringstream ss;
  ss << f.rdbuf();
  return parse_ranked_ids(ss.str(), "--draft-vocab-ids file '" + path + "'");
}

namespace {
// The text of the JSON array that starts at text[open] ('['): brackets are matched
// outside strings only, and a string's escapes are skipped, so a token whose content
// is "[" or "\"]" cannot end it early.
std::string_view array_at(std::string_view text, size_t open) {
  int depth = 0;
  bool in_str = false;
  for (size_t i = open; i < text.size(); ++i) {
    const char c = text[i];
    if (in_str) {
      if (c == '\\') ++i;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '[' || c == '{') ++depth;
    else if ((c == ']' || c == '}') && --depth == 0) return text.substr(open, i + 1 - open);
  }
  throw std::runtime_error("tokenizer.json: the added_tokens array is not closed");
}
}  // namespace

std::vector<uint32_t> added_token_ids(const std::string& text) {
  // Only the `added_tokens` array is parsed, not the whole document: tokenizer.json is
  // ~11 MB, almost all of it the BPE vocabulary and merges, and common::json would build
  // a tree of ~500k values for an array of 33 objects. The key is HF's top-level
  // `"added_tokens":` (its first occurrence; the vocabulary's keys are token strings, and
  // a token spelled exactly `added_tokens` would be followed by a number, not a '[').
  const std::string_view t(text);
  size_t at = 0;
  while ((at = t.find("\"added_tokens\"", at)) != std::string_view::npos) {
    size_t i = at + 14;
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    if (i < t.size() && t[i] == ':') {
      ++i;
      while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
      if (i < t.size() && t[i] == '[') {
        const common::json::Value arr = common::json::parse(array_at(t, i));
        std::vector<uint32_t> ids;
        for (const common::json::Value& e : arr.arr()) {
          const double id = e.at("id").num();
          if (!(id >= 0 && id <= 4294967295.0) || id != double(uint32_t(id)))
            throw std::runtime_error("tokenizer.json: an added token's id is not a u32");
          ids.push_back(static_cast<uint32_t>(id));
        }
        return ids;
      }
    }
    at += 14;
  }
  throw std::runtime_error("tokenizer.json has no \"added_tokens\" array");
}

std::vector<uint32_t> added_token_ids_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open '" + path + "'");
  std::stringstream ss;
  ss << f.rdbuf();
  return added_token_ids(ss.str());
}

}  // namespace loader
