#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct b70_tok;

namespace tok {

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json_path);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  uint32_t vocab_size() const;
  std::vector<uint32_t> encode(std::string_view text, bool add_special = false) const;
  // encode(text) for a text that may start as the last one `memo` saw (a conversation's next prompt). The memo keeps
  // that text, its ids and its split points (b70_tok_encode_ends); the ids through the last split point the two texts
  // share - with the longest added token's bytes after it shared too, so no longer added token can match there - are
  // kept, and only the text after it is encoded. Equal to encode(text) (tokenizer_extend_test).
  struct Memo {
    std::string text;
    std::vector<uint32_t> ids;
    std::vector<std::pair<size_t, size_t>> splits;   // (byte end, ids through it), ascending
  };
  std::vector<uint32_t> encode_extending(std::string_view text, Memo& memo) const;
  std::string decode(const std::vector<uint32_t>& ids, bool skip_special = false) const;
  std::string id_to_token(uint32_t id) const;
  std::optional<uint32_t> token_to_id(std::string_view token) const;

 private:
  b70_tok* h_ = nullptr;
};

std::string default_tokenizer_json();

}  // namespace tok
