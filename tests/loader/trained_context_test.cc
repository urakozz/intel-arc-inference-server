// Spec 6 §10: config.json's max_position_embeddings, the cap of --max-len auto and
// the bound of an explicit --max-len. Host only.
//
//   trained_context_test [snapshot-dir/]   with a directory: also its config.json
#include <cstdio>
#include <stdexcept>
#include <string>
#include "check.h"
#include "common/json.h"
#include "loader/trained_context.h"

namespace {
uint32_t of(const std::string& text) { return loader::trained_context(common::json::parse(text)); }

bool throws(const std::string& text) {
  try {
    of(text);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  // Qwen3.8 / Agnes / Ornith: the multimodal layout, text_config wins.
  CHECK_EQ(of(R"({"architectures":["Qwen3_5ForConditionalGeneration"],
                  "text_config":{"max_position_embeddings":262144,"hidden_size":5120}})"),
           262144u);
  CHECK_EQ(of(R"({"max_position_embeddings":4096,"text_config":{"max_position_embeddings":262144}})"),
           262144u);
  // A text-only checkpoint: the top level.
  CHECK_EQ(of(R"({"max_position_embeddings":32768})"), 32768u);
  CHECK_EQ(of(R"({"text_config":{"hidden_size":5120},"max_position_embeddings":40960})"), 40960u);
  // Not declared: 0, the caller decides.
  CHECK_EQ(of(R"({"text_config":{"hidden_size":5120}})"), 0u);
  CHECK_EQ(of(R"({})"), 0u);
  // Declared but not a positive integer that fits uint32_t.
  CHECK(throws(R"({"max_position_embeddings":"262144"})"));
  CHECK(throws(R"({"max_position_embeddings":0})"));
  CHECK(throws(R"({"max_position_embeddings":-1})"));
  CHECK(throws(R"({"max_position_embeddings":1.5})"));
  CHECK(throws(R"({"max_position_embeddings":4294967296})"));
  CHECK(throws(R"({"text_config":{"max_position_embeddings":null}})"));
  CHECK(throws(R"([1,2])"));
  if (argc > 1) {
    std::string dir = argv[1];
    if (!dir.empty() && dir.back() != '/') dir += '/';
    const uint32_t n = loader::trained_context(dir);
    std::printf("%sconfig.json: max_position_embeddings %u\n", dir.c_str(), n);
    CHECK(n != 0);
  }
  std::puts("trained_context_test OK");
  return 0;
}
