#include "tokenizer/tokenizer.h"

#include <cstdlib>
#include <stdexcept>

#include "tokenizer/b70_tok.h"

namespace tok {
namespace {

[[noreturn]] void fail(const char* what) {
  throw std::runtime_error(std::string("tok::Tokenizer::") + what + ": " +
                           b70_tok_last_error());
}

}  // namespace

Tokenizer::Tokenizer(const std::string& path) {
  if (b70_tok_new(path.c_str(), &h_) != 0) fail("Tokenizer");
}

Tokenizer::~Tokenizer() { b70_tok_free(h_); }

uint32_t Tokenizer::vocab_size() const { return b70_tok_vocab_size(h_); }

std::vector<uint32_t> Tokenizer::encode(std::string_view text, bool add_special) const {
  uint32_t* ids = nullptr;
  size_t n = 0;
  if (b70_tok_encode(h_, text.data(), text.size(), add_special ? 1 : 0, &ids, &n) != 0) {
    fail("encode");
  }
  std::vector<uint32_t> out(ids, ids + n);
  b70_tok_free_buf(ids, n, 1);
  return out;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids, bool skip_special) const {
  char* s = nullptr;
  size_t n = 0;
  if (b70_tok_decode(h_, ids.data(), ids.size(), skip_special ? 1 : 0, &s, &n) != 0) {
    fail("decode");
  }
  std::string out(s, n);
  b70_tok_free_buf(s, n, 0);
  return out;
}

std::string Tokenizer::id_to_token(uint32_t id) const {
  char* s = nullptr;
  size_t n = 0;
  if (b70_tok_id_to_token(h_, id, &s, &n) != 0) fail("id_to_token");
  std::string out(s, n);
  b70_tok_free_buf(s, n, 0);
  return out;
}

std::optional<uint32_t> Tokenizer::token_to_id(std::string_view t) const {
  const int64_t id = b70_tok_token_to_id(h_, t.data(), t.size());
  if (id < 0) return std::nullopt;
  return uint32_t(id);
}

std::string default_tokenizer_json() {
  if (const char* e = std::getenv("B70_TOKENIZER_JSON")) return e;
  return "/home/user/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d/tokenizer.json";
}

}  // namespace tok
