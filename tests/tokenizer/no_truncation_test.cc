// tok::Tokenizer ignores a tokenizer.json's `truncation` and `padding` blocks: the engine
// tokenises a whole prompt, always. The Rust `tokenizers` crate applies both on every encode
// when the file sets them - and urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ's
// tokenizer.json ships `"truncation": {"max_length": 2048, "direction": "Right", ...}` (a
// calibration-time setting saved with the quantised repo), so before this every K2 prompt
// past 2048 ids lost its tail - the question - with no error (the 2026-11-01 "K2 fails past
// 2048 positions" passkey report; the engine itself was never at fault).
//
// Device-free and data-free: a five-word WordLevel tokenizer.json written to a temp file,
// with truncation at 4 ids and fixed padding to 16. Six words must come back as six ids.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "check.h"
#include "tokenizer/tokenizer.h"

int main() {
  const char* json = R"({
  "version": "1.0",
  "truncation": {"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0},
  "padding": {"strategy": {"Fixed": 16}, "direction": "Right", "pad_to_multiple_of": null,
              "pad_id": 5, "pad_type_id": 0, "pad_token": "[UNK]"},
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {"type": "WordLevel",
            "vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "[UNK]": 5},
            "unk_token": "[UNK]"}
})";
  char path[] = "/tmp/b70_no_truncation_XXXXXX";
  const int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  {
    std::ofstream f(path, std::ios::binary);
    f << json;
    CHECK(f.good());
  }
  int rc = 0;
  {
    tok::Tokenizer t(path);
    const std::vector<uint32_t> ids = t.encode("a b c d e a", false);
    const std::vector<uint32_t> want = {0, 1, 2, 3, 4, 0};
    if (ids != want) {
      std::fprintf(stderr, "no_truncation_test: 'a b c d e a' gave %zu ids (want 6: the file's truncation"
                           " at 4 / padding to 16 must not apply)\n", ids.size());
      rc = 1;
    }
  }
  std::remove(path);
  if (rc == 0) std::printf("no_truncation_test OK: 6 words -> 6 ids under truncation 4 / padding 16\n");
  return rc;
}
