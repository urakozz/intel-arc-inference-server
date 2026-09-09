// Spec 3 §2 bar 1: one id at a time equals one-shot decode, with no spurious U+FFFD.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"

namespace {

std::vector<std::string> lines(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  const std::string all = contents.str();
  std::vector<std::string> result;
  size_t begin = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    if (all[i] == '\n') {
      result.emplace_back(all, begin, i - begin);
      begin = i + 1;
    }
  }
  if (begin < all.size()) result.emplace_back(all, begin);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  tok::Tokenizer tokenizer(tok::default_tokenizer_json());
  const auto reference = lines(dir + "/corpus.ids");
  size_t cases = 0;
  size_t bad_concat = 0;
  size_t bad_fffd = 0;
  size_t held_max = 0;
  for (const std::string& line : reference) {
    std::vector<uint32_t> ids;
    std::istringstream input(line);
    uint32_t id = 0;
    while (input >> id) ids.push_back(id);
    if (ids.empty()) continue;
    ++cases;
    const std::string once = tokenizer.decode(ids);
    tok::Streamer streamer(tokenizer);
    std::string concatenated;
    size_t held = 0;
    size_t current_hold = 0;
    for (uint32_t value : ids) {
      const std::string output = streamer.push(value);
      if (output.empty()) {
        ++current_hold;
        held = std::max(held, current_hold);
      } else {
        current_hold = 0;
      }
      concatenated += output;
    }
    concatenated += streamer.flush();
    held_max = std::max(held_max, held);
    if (concatenated != once) {
      if (bad_concat < 5) std::fprintf(stderr, "concat mismatch: %s\n", once.c_str());
      ++bad_concat;
    }
    if (concatenated.find("\xEF\xBF\xBD") != std::string::npos &&
        once.find("\xEF\xBF\xBD") == std::string::npos) {
      ++bad_fffd;
    }
  }
  std::printf("streamer_test: %zu cases, %zu concat mismatches, %zu spurious U+FFFD, "
              "longest hold %zu ids\n",
              cases, bad_concat, bad_fffd, held_max);
  CHECK_EQ(bad_concat, size_t(0));
  CHECK_EQ(bad_fffd, size_t(0));
  CHECK(held_max >= 1);
  std::printf("streamer_test OK\n");
  return 0;
}
