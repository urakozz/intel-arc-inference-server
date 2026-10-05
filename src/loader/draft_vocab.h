#pragma once
// Spec 8 §11: a reduced draft vocabulary V' for the MTP head's drafts. Host-only and
// pure (no device, no checkpoint): which ids V' holds, the CLI spelling of its size,
// the ranked-id file `tools/draft_vocab/rank.py` writes, and tokenizer.json's added
// tokens. The loader gathers the chosen rows of the int8 `lm_head` (loader.h,
// `DraftVocab`); tests/loader/draft_vocab_test.cc covers everything here.
//
// **Why only the draft is reduced.** The verify list keeps the full head, bitwise
// (spec 8 §8 A6), so the output never changes: greedy output with V' equals greedy
// output without it, and sampled output keeps the target's distribution (a draft can
// only be a V' id and q = 0 elsewhere, which the acceptance rule handles, M4). What V'
// can cost is acceptance, when the target's next id lies outside it - which is why the
// added tokens (the chat, tool-call and think tags) are always in it.
#include <cstdint>
#include <string>
#include <vector>

namespace loader {

// The sizes the compact head's GEMV and argmax are compiled for
// (src/kernels/CMakeLists.txt, spec 8 §11): 32768 / 65536 / 131072.
inline constexpr uint32_t kDraftVocabSizes[] = {32768, 65536, 131072};

// "off" -> 0, "32k" -> 32768, "64k" -> 65536, "128k" -> 131072; false on anything else.
bool parse_draft_vocab(const std::string& s, uint32_t& size);
// The inverse: "off", "32k", "64k", "128k" (or the number, for any other size).
std::string draft_vocab_name(uint32_t size);

// What the loader needs to build V'. `size` = 0 is off (nothing is gathered).
struct DraftVocabSpec {
  uint32_t size = 0;
  std::vector<uint32_t> added;    // tokenizer.json's added tokens, file order
  std::vector<uint32_t> eos;      // generation_config.json's eos_token_id
  std::vector<uint32_t> ranked;   // --draft-vocab-ids, file order (may be empty)
};

// Where V''s ids came from, for the load report.
struct DraftVocabCounts {
  uint32_t forced = 0;   // added tokens and EOS ids
  uint32_t ranked = 0;   // from the ranked list
  uint32_t lowest = 0;   // the lowest remaining ids
};

// V': `size` distinct ids, sorted ascending, filled in this order until full (spec 8 §11):
//   1. every id of `added`, then of `eos` - always;
//   2. `ranked`, in order;
//   3. the lowest remaining ids from 0 up (byte-level BPE ids follow merge order, which
//      tracks training frequency).
// Every source skips duplicates and ids >= `vocab_used`: the full head's argmax masks
// those (argmax.cl, VOCAB_USED), so the target never emits one and a draft of one could
// only be rejected. Ascending order is what makes the compact argmax's tie rule (lowest
// index) the full head's (lowest id).
//
// Throws std::invalid_argument when the request is impossible: size 0, size >
// vocab_used, or more distinct usable added/EOS ids than `size`.
std::vector<uint32_t> select_draft_vocab(const std::vector<uint32_t>& added,
                                         const std::vector<uint32_t>& eos,
                                         const std::vector<uint32_t>& ranked, uint32_t vocab_used,
                                         uint32_t size, DraftVocabCounts* counts = nullptr);

// The ranked-id file (`--draft-vocab-ids`, tools/draft_vocab/rank.py's output): decimal
// ids separated by whitespace (rank.py writes one per line, most frequent first); a line
// whose first non-blank character is '#' is a comment. Throws on a missing file, a
// token that is not a decimal u32, or a file with no ids.
std::vector<uint32_t> read_ranked_ids(const std::string& path);
// The same, over the file's text (the unit test's entry point).
std::vector<uint32_t> parse_ranked_ids(const std::string& text, const std::string& what);

// The ids of tokenizer.json's `added_tokens` array, in file order (33 on Qwen3.8:
// 248044..248076). Throws if the text is not JSON or has no such array.
std::vector<uint32_t> added_token_ids(const std::string& tokenizer_json_text);
// Reads the file, then the above.
std::vector<uint32_t> added_token_ids_file(const std::string& tokenizer_json_path);

}  // namespace loader
