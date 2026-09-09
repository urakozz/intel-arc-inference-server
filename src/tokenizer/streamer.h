#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace tok {

// Incremental detokeniser using the openvino.genai hold-and-flush algorithm.
class Streamer {
 public:
  explicit Streamer(const Tokenizer& tokenizer);

  std::string push(uint32_t id);
  std::string flush();

 private:
  const Tokenizer& tokenizer_;
  std::vector<uint32_t> pending_;
  size_t printed_ = 0;
};

}  // namespace tok
