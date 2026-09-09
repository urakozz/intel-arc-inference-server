#include <cstdio>
#include <string>
#include "check.h"
#include "tokenizer/tokenizer.h"

int main() {
  tok::Tokenizer t(tok::default_tokenizer_json());
  CHECK_EQ(t.vocab_size(), 248077u);  // docs/11: 248044 + 33 added
  const std::vector<uint32_t> ids = t.encode("Hello, world!");
  CHECK(!ids.empty());
  CHECK_EQ(t.decode(ids), std::string("Hello, world!"));
  // Special tokens survive encode(add_special=false) → decode(skip_special=false)
  // as literal text, and map to the ids docs/11 records.
  CHECK_EQ(t.token_to_id("<|im_end|>").value_or(0), 248046u);
  CHECK_EQ(t.token_to_id("<|endoftext|>").value_or(0), 248044u);
  const std::vector<uint32_t> s = t.encode("<|im_start|>user\nhi<|im_end|>\n");
  CHECK_EQ(s.front(), 248045u);  // <|im_start|>
  CHECK_EQ(t.decode(s), std::string("<|im_start|>user\nhi<|im_end|>\n"));
  CHECK(t.token_to_id("definitely-not-a-token-xyz") == std::nullopt);
  std::printf("tokenizer_smoke_test OK: vocab %u, %zu ids for the greeting\n", t.vocab_size(), ids.size());
  return 0;
}
