// Spec 8 §11: the reduced draft vocabulary's host half. No device, no checkpoint:
//   1. the --draft-vocab spelling;
//   2. which ids V' holds (loader::select_draft_vocab): added tokens and EOS always,
//      ascending, distinct, the ranked list honoured in order, ids >= vocab_used never,
//      exactly |V'| ids, and a throw on an impossible request;
//   3. the ranked-id file (tools/draft_vocab/rank.py's output) and tokenizer.json's
//      added tokens;
//   4. the compact head's gather (loader::gather_int8_tiled_rows): compact row j is the
//      full head's row ids[j], byte for byte, in gemv_i8w's tiled layout - equal to
//      quantising the selected bf16 rows directly - whatever the thread count; and the
//      bf16 head's gather (gemv_bf16's layout) equal to tiling the selected rows.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "loader/draft_vocab.h"
#include "loader/lm_head_int8.h"

namespace {

template <class F>
bool throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

bool contains(const std::vector<uint32_t>& v, uint32_t id) {
  return std::binary_search(v.begin(), v.end(), id);
}

// The properties every V' has, whatever its sources.
void check_shape(const std::vector<uint32_t>& v, uint32_t size, uint32_t vocab_used) {
  CHECK_EQ(v.size(), size_t(size));
  for (size_t i = 1; i < v.size(); ++i) CHECK(v[i - 1] < v[i]);   // ascending AND distinct
  CHECK(v.back() < vocab_used);                                  // never a masked id
}

void test_spelling() {
  uint32_t n = 7;
  CHECK(loader::parse_draft_vocab("off", n) && n == 0);
  CHECK(loader::parse_draft_vocab("32k", n) && n == 32768);
  CHECK(loader::parse_draft_vocab("64k", n) && n == 65536);
  CHECK(loader::parse_draft_vocab("128k", n) && n == 131072);
  for (const char* bad : {"", "0", "32K", "32768", "256k", "on", "16k"}) {
    n = 7;
    CHECK(!loader::parse_draft_vocab(bad, n));
    CHECK_EQ(n, 7u);
  }
  for (uint32_t s : {0u, 32768u, 65536u, 131072u}) {
    uint32_t back = 1;
    CHECK(loader::parse_draft_vocab(loader::draft_vocab_name(s), back) && back == s);
  }
  for (uint32_t s : loader::kDraftVocabSizes) CHECK(s % 1024 == 0);   // the argmax's chunk
}

void test_select_small() {
  // vocab_used 100. Added {95, 96, 97}, EOS {96, 98} (one shared with added), a ranked
  // list with a duplicate, a masked id (120), an added id (95) and plain ids.
  const std::vector<uint32_t> added = {95, 96, 97}, eos = {96, 98};
  const std::vector<uint32_t> ranked = {50, 50, 99, 120, 3, 95, 10};
  loader::DraftVocabCounts c;
  const std::vector<uint32_t> v = loader::select_draft_vocab(added, eos, ranked, 100, 32, &c);
  check_shape(v, 32, 100);
  for (uint32_t id : {95u, 96u, 97u, 98u}) CHECK(contains(v, id));   // forced, always
  for (uint32_t id : {50u, 99u, 3u, 10u}) CHECK(contains(v, id));     // ranked, honoured
  CHECK(!contains(v, 120));
  CHECK_EQ(c.forced, 4u);
  CHECK_EQ(c.ranked, 4u);
  CHECK_EQ(c.lowest, 24u);
  // The lowest remaining ids fill the rest: 0..27 minus the ranked 3 and 10, i.e. 26
  // candidates of which the first 24 are taken - 0..25 without 3 and 10.
  for (uint32_t id = 0; id <= 25; ++id) CHECK(contains(v, id));
  for (uint32_t id = 26; id < 95; ++id)
    if (id != 50) CHECK(!contains(v, id));

  // A ranked list longer than the room left: its PREFIX is kept, in file order.
  const std::vector<uint32_t> long_rank = {60, 61, 62, 63, 64, 65, 66, 67};
  const std::vector<uint32_t> w = loader::select_draft_vocab({1}, {2}, long_rank, 100, 5, &c);
  CHECK((w == std::vector<uint32_t>{1, 2, 60, 61, 62}));
  CHECK_EQ(c.forced, 2u);
  CHECK_EQ(c.ranked, 3u);
  CHECK_EQ(c.lowest, 0u);

  // Added or EOS ids at or above vocab_used are masked by the argmax, so never chosen.
  const std::vector<uint32_t> m = loader::select_draft_vocab({150, 7}, {99, 100}, {}, 100, 16, &c);
  check_shape(m, 16, 100);
  CHECK(contains(m, 7) && contains(m, 99) && !contains(m, 100));
  CHECK_EQ(c.forced, 2u);

  // |V'| = vocab_used: every usable id.
  const std::vector<uint32_t> all = loader::select_draft_vocab({5}, {}, {9}, 64, 64);
  for (uint32_t id = 0; id < 64; ++id) CHECK_EQ(all[id], id);
}

void test_select_throws() {
  CHECK(throws([] { loader::select_draft_vocab({}, {}, {}, 100, 0); }));     // off
  CHECK(throws([] { loader::select_draft_vocab({}, {}, {}, 100, 101); }));   // > usable
  CHECK(throws([] { loader::select_draft_vocab({1, 2, 3}, {4}, {}, 100, 3); }));  // forced > size
  // Exactly full of forced ids is possible.
  CHECK((loader::select_draft_vocab({3, 1, 2}, {}, {}, 100, 3) == std::vector<uint32_t>{1, 2, 3}));
}

void test_select_qwen38_shape() {
  // Qwen3.8's numbers: 248077 usable ids, 33 added tokens 248044..248076 (EOS among
  // them), at every compiled size, with and without a ranked list.
  const uint32_t used = 248077;
  std::vector<uint32_t> added;
  for (uint32_t id = 248044; id <= 248076; ++id) added.push_back(id);
  const std::vector<uint32_t> eos = {248046, 248044};
  std::mt19937 g(5);
  std::vector<uint32_t> ranked(70000);
  for (uint32_t& id : ranked) id = g() % 248320;   // some masked, many duplicates
  for (uint32_t size : loader::kDraftVocabSizes) {
    loader::DraftVocabCounts c;
    const std::vector<uint32_t> v = loader::select_draft_vocab(added, eos, {}, used, size, &c);
    check_shape(v, size, used);
    for (uint32_t id : added) CHECK(contains(v, id));
    CHECK_EQ(c.forced, 33u);
    // Without a ranking: the lowest size - 33 ids, then the added tokens.
    CHECK_EQ(v[size - 34], size - 34);
    CHECK_EQ(v[size - 33], 248044u);

    const std::vector<uint32_t> r = loader::select_draft_vocab(added, eos, ranked, used, size, &c);
    check_shape(r, size, used);
    for (uint32_t id : added) CHECK(contains(r, id));
    // Every ranked id up to the point V' filled is in; walk the list as the rule does.
    std::set<uint32_t> seen(added.begin(), added.end());
    uint32_t room = size - 33, honoured = 0;
    for (uint32_t id : ranked) {
      if (room == 0) break;
      if (id >= used || !seen.insert(id).second) continue;
      CHECK(contains(r, id));
      --room;
      ++honoured;
    }
    CHECK_EQ(c.ranked, honoured);
    CHECK_EQ(c.forced + c.ranked + c.lowest, size);
  }
}

void test_ranked_file() {
  const std::vector<uint32_t> ids =
      loader::parse_ranked_ids("# rank.py: 2 files\n\n  42\n7 9\r\n  # indented comment\n0\n", "t");
  CHECK((ids == std::vector<uint32_t>{42, 7, 9, 0}));
  CHECK(throws([] { loader::parse_ranked_ids("12\nabc\n", "t"); }));
  CHECK(throws([] { loader::parse_ranked_ids("-1\n", "t"); }));
  CHECK(throws([] { loader::parse_ranked_ids("4294967296\n", "t"); }));
  CHECK(throws([] { loader::parse_ranked_ids("# only a comment\n\n", "t"); }));
  CHECK(throws([] { loader::read_ranked_ids("/nonexistent/draft-vocab.ids"); }));
}

void test_added_tokens() {
  // A vocabulary key spelled `added_tokens` (followed by a number) before the real
  // array, and contents with brackets and escaped quotes inside the array.
  const std::string json =
      "{\"version\": \"1.0\", \"truncation\": null,\n"
      " \"model\": {\"vocab\": {\"added_tokens\": 3, \"[\": 4}},\n"
      " \"added_tokens\" : [\n"
      "   {\"id\": 248044, \"content\": \"<|endoftext|>\", \"special\": true},\n"
      "   {\"id\": 248058, \"content\": \"]\\\"[\", \"special\": false},\n"
      "   {\"id\": 248076, \"content\": \"<tool_call>\", \"special\": false}\n"
      " ],\n"
      " \"normalizer\": null}";
  CHECK((loader::added_token_ids(json) == std::vector<uint32_t>{248044, 248058, 248076}));
  CHECK(throws([] { loader::added_token_ids("{\"model\": {}}"); }));
  CHECK(throws([] { loader::added_token_ids("{\"added_tokens\": [{\"id\": 1}"); }));
  CHECK(throws([] { loader::added_token_ids("{\"added_tokens\": [{\"id\": -1}]}"); }));
}

std::vector<uint16_t> random_rows(uint32_t N, uint32_t K, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.f, 0.02f);
  std::vector<uint16_t> w(size_t(N) * K);
  for (uint16_t& x : w) x = common::f32_to_bf16(d(g));
  return w;
}

void test_gather() {
  const uint32_t N = 96, K = 256;
  const std::vector<uint16_t> w = random_rows(N, K, 77);
  std::vector<int8_t> full(size_t(N) * K);
  std::vector<float> s(N);
  loader::quantise_int8_tiled(w.data(), K, N, full.data(), s.data(), 3);

  // 32 rows of 96: ascending, with neighbours, a tile's first and last row, the last row.
  const std::vector<uint32_t> ids = loader::select_draft_vocab({95, 80}, {16}, {15, 47, 33}, N, 32);
  CHECK_EQ(ids.size(), size_t(32));
  const uint32_t n = 32;

  // The reference: quantise the selected bf16 rows directly, as their own [n][K] head.
  std::vector<uint16_t> sel(size_t(n) * K);
  for (uint32_t j = 0; j < n; ++j)
    std::memcpy(&sel[size_t(j) * K], &w[size_t(ids[j]) * K], size_t(K) * 2);
  std::vector<int8_t> ref(size_t(n) * K);
  std::vector<float> ref_s(n);
  loader::quantise_int8_tiled(sel.data(), K, n, ref.data(), ref_s.data(), 1);

  for (unsigned threads : {1u, 2u, 8u}) {
    std::vector<int8_t> out(size_t(n) * K, 0x55);
    std::vector<float> out_s(n, -1.f);
    loader::gather_int8_tiled_rows(full.data(), s.data(), K, N, ids.data(), n, out.data(),
                                   out_s.data(), threads);
    CHECK(out == ref);
    CHECK(std::memcmp(out_s.data(), ref_s.data(), n * sizeof(float)) == 0);
    // And element by element through the layout's own index.
    for (uint32_t j = 0; j < n; ++j)
      for (uint32_t k = 0; k < K; ++k)
        CHECK_EQ(out[loader::int8_tiled_index(K, k, j)], full[loader::int8_tiled_index(K, k, ids[j])]);
  }
  std::vector<int8_t> out(size_t(n) * K);
  std::vector<float> out_s(n);
  std::vector<uint32_t> bad = ids;
  bad[5] = N;   // outside the head
  CHECK(throws([&] {
    loader::gather_int8_tiled_rows(full.data(), s.data(), K, N, bad.data(), n, out.data(), out_s.data());
  }));
  CHECK(throws([&] {   // not whole tiles
    loader::gather_int8_tiled_rows(full.data(), s.data(), K, N, ids.data(), 24, out.data(), out_s.data());
  }));
}

// The bf16 head (`--lm-head bf16 --draft-vocab`): gathering rows of the tiled head equals
// tiling the selected row-major rows (common::repack_bf16_tiled, the loader's layout).
void test_gather_bf16() {
  const uint32_t N = 96, K = 256;
  const std::vector<uint16_t> w = random_rows(N, K, 78);
  std::vector<uint16_t> full(size_t(N) * K);
  common::repack_bf16_tiled(w.data(), K, N, full.data());
  const std::vector<uint32_t> ids = loader::select_draft_vocab({95, 80}, {16}, {15, 47, 33}, N, 48);
  const uint32_t n = 48;
  std::vector<uint16_t> sel(size_t(n) * K);
  for (uint32_t j = 0; j < n; ++j)
    std::memcpy(&sel[size_t(j) * K], &w[size_t(ids[j]) * K], size_t(K) * 2);
  std::vector<uint16_t> ref(size_t(n) * K);
  common::repack_bf16_tiled(sel.data(), K, n, ref.data());
  for (unsigned threads : {1u, 3u, 8u}) {
    std::vector<uint16_t> out(size_t(n) * K, 0xABCD);
    loader::gather_bf16_tiled_rows(full.data(), K, N, ids.data(), n, out.data(), threads);
    CHECK(out == ref);
  }
  std::vector<uint16_t> out(size_t(n) * K);
  std::vector<uint32_t> bad = ids;
  bad[0] = N;
  CHECK(throws([&] { loader::gather_bf16_tiled_rows(full.data(), K, N, bad.data(), n, out.data()); }));
  CHECK(throws([&] { loader::gather_bf16_tiled_rows(full.data(), K, N, ids.data(), 40, out.data()); }));
}

}  // namespace

int main() {
  test_spelling();
  test_select_small();
  test_select_throws();
  test_select_qwen38_shape();
  test_ranked_file();
  test_added_tokens();
  test_gather();
  test_gather_bf16();
  std::puts("draft_vocab_test OK");
  return 0;
}
