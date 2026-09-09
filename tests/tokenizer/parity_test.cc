#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "tokenizer/tokenizer.h"

static std::vector<std::string> lines(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  CHECK(f.good());
  std::stringstream ss;
  ss << f.rdbuf();
  std::string all = ss.str();
  std::vector<std::string> out;
  size_t b = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    if (all[i] == '\n') {
      out.emplace_back(all, b, i - b);
      b = i + 1;
    }
  }
  if (b < all.size()) out.emplace_back(all, b);
  return out;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  tok::Tokenizer t(tok::default_tokenizer_json());
  const auto txt = lines(dir + "/corpus.txt");
  const auto ref = lines(dir + "/corpus.ids");
  CHECK_EQ(txt.size(), ref.size());
  CHECK(txt.size() >= 10000);
  size_t enc_bad = 0;
  size_t dec_bad = 0;
  size_t total_ids = 0;
  for (size_t i = 0; i < txt.size(); ++i) {
    std::vector<uint32_t> want;
    std::istringstream is(ref[i]);
    uint32_t v;
    while (is >> v) want.push_back(v);
    const std::vector<uint32_t> got = t.encode(txt[i]);
    total_ids += got.size();
    if (got != want) {
      if (enc_bad < 10) {
        std::fprintf(stderr, "encode mismatch line %zu: %s\n", i + 1, txt[i].c_str());
      }
      ++enc_bad;
    }
    if (t.decode(got) != txt[i]) {
      if (dec_bad < 10) std::fprintf(stderr, "decode mismatch line %zu\n", i + 1);
      ++dec_bad;
    }
  }
  std::printf("parity_test: %zu cases, %zu ids, %zu encode mismatches, %zu decode mismatches\n",
              txt.size(), total_ids, enc_bad, dec_bad);
  CHECK_EQ(enc_bad, size_t(0));
  CHECK_EQ(dec_bad, size_t(0));
  std::printf("parity_test OK\n");
  return 0;
}
