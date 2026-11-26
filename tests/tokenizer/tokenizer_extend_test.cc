// tok::Tokenizer::encode_extending (the serial server's prompts) equals encode on a conversation that grows turn by
// turn, one whose tail is edited, one cut short and an unrelated text, each through one memo:
//   tokenizer_extend_test <tests/tokenizer> <tokenizer.json>=<render prefix> ...
// "default" names tok::default_tokenizer_json() (Qwen3.8's). Each tokenizer's conversation is its model's template
// renders (<tests/tokenizer>/<render prefix>*.txt, sorted) with corpus.txt pieces between them; its added tokens must
// give split points there (the reuse is exercised). A tokenizer.json that does not exist is said and skipped; none at
// all: SKIP (77).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "tokenizer/tokenizer.h"

namespace {

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream out;
  out << file.rdbuf();
  return out.str();
}

// The largest UTF-8 character boundary <= n.
size_t boundary(const std::string& s, size_t n) {
  while (n > 0 && n < s.size() && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: tokenizer_extend_test <tests/tokenizer> <tokenizer.json>=<render prefix> ...\n");
    return 2;
  }
  const std::string dir = argv[1];
  const std::string corpus = slurp(dir + "/corpus.txt");
  CHECK(corpus.size() > 64 * 1024);
  int ran = 0;
  for (int a = 2; a < argc; ++a) {
    const std::string arg = argv[a];
    const size_t eq = arg.rfind('=');
    CHECK(eq != std::string::npos);
    const std::string name = arg.substr(0, eq), prefix = arg.substr(eq + 1);
    const std::string json = name == "default" ? tok::default_tokenizer_json() : name;
    if (!std::filesystem::exists(json)) {
      std::printf("tokenizer_extend_test: %s does not exist, skipped\n", json.c_str());
      continue;
    }
    std::vector<std::string> renders;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      const std::string file = e.path().filename().string();
      if (file.starts_with(prefix) && file.ends_with(".txt")) renders.push_back(e.path().string());
    }
    std::sort(renders.begin(), renders.end());
    CHECK(!renders.empty());

    const tok::Tokenizer t(json);
    tok::Tokenizer::Memo memo;
    size_t checked = 0, splits = 0;
    const auto check = [&](const std::string& text) {
      CHECK(t.encode_extending(text, memo) == t.encode(text));
      CHECK(memo.ids == t.encode(text));
      splits = std::max(splits, memo.splits.size());
      ++checked;
    };
    std::string conv;
    size_t at = 0;
    for (const std::string& path : renders) {   // the conversation grows: a corpus piece, then a render
      const size_t end = boundary(corpus, std::min(corpus.size(), at + 3000));
      conv += corpus.substr(at, end - at) + slurp(path);
      at = end;
      check(conv);
    }
    const std::string full = conv;
    for (size_t back : {1, 7, 40, 500}) {   // the tail edited just after a shared start
      const std::string edited = full.substr(0, boundary(full, full.size() - std::min(full.size(), back))) +
                                 "\nan edited tail " + slurp(renders.front());
      check(edited);
    }
    check(full.substr(0, boundary(full, full.size() / 2)));   // cut short
    check(corpus.substr(0, boundary(corpus, 20000)));          // unrelated
    check(full);                                               // and back
    check(full + slurp(renders.back()));
    CHECK(splits > 0);
    std::printf("tokenizer_extend_test: %s - %zu texts equal encode, up to %zu split points\n", json.c_str(), checked,
                splits);
    ++ran;
  }
  if (ran == 0) {
    std::printf("tokenizer_extend_test: SKIP - no tokenizer.json\n");
    return 77;
  }
  std::printf("tokenizer_extend_test OK: %d tokenizers\n", ran);
  return 0;
}
