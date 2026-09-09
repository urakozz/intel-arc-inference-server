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
  std::string decode(const std::vector<uint32_t>& ids, bool skip_special = false) const;
  std::string id_to_token(uint32_t id) const;
  std::optional<uint32_t> token_to_id(std::string_view token) const;

 private:
  b70_tok* h_ = nullptr;
};

std::string default_tokenizer_json();

}  // namespace tok
