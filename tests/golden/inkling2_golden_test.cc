// Read a common-context dump through the same f32 decision/cosine judge as
// the other families. With one file, exercise the judge and schema against
// the original CPU golden. For a candidate, supply f32, bf16 and engine data.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

#include "golden/forced_judge.h"
#include "model/inkling2.h"

using D = model::Inkling2Desc;
using golden::Golden;

namespace {
bool same_experts(const int32_t* a, const int32_t* b) {
  std::array<int32_t, D::top_k> left{}, right{};
  std::copy_n(a, left.size(), left.begin());
  std::copy_n(b, right.size(), right.begin());
  for (const auto id : left) CHECK(id >= 0 && id < int32_t(D::experts));
  for (const auto id : right) CHECK(id >= 0 && id < int32_t(D::experts));
  std::sort(left.begin(), left.end());
  std::sort(right.begin(), right.end());
  return left == right;
}
}  // namespace

int main(int argc, char** argv) {
  CHECK(argc == 2 || argc == 4);
  const Golden ref(argv[1]), twin(argv[argc == 2 ? 1 : 2]), engine(argv[argc == 2 ? 1 : 3]);
  const uint32_t count = uint32_t(ref.dim("tokens", 1, 0));
  const uint32_t rows = uint32_t(ref.dim("logits", 2, 0));
  CHECK(count < rows);
  const uint32_t prompt = rows - count;
  const auto* chosen = engine.i32("tokens", count);
  const auto* tokens = ref.i32("tokens", count);
  for (const auto* g : {&ref, &twin, &engine}) {
    CHECK_EQ(g->dim("logits", 2, 0), rows);
    CHECK_EQ(g->dim("logits", 2, 1), D::vocab_used);
    CHECK(std::equal(tokens, tokens + count, g->i32("tokens", count)));
  }
  if (argc == 4) {
    const auto* ids = ref.i32("prompt_ids", prompt);
    for (const auto* g : {&twin, &engine}) CHECK(std::equal(ids, ids + prompt, g->i32("prompt_ids", prompt)));
  }
  const auto* j = ref.f32("logits", size_t(rows) * D::vocab_used);
  const auto* t = twin.f32("logits", size_t(rows) * D::vocab_used);
  const auto* e = engine.f32("logits", size_t(rows) * D::vocab_used);
  golden::NearMiss rule;
  for (uint32_t i = 0; i < count; ++i) {
    CHECK(chosen[i] >= 0 && chosen[i] < int32_t(D::vocab_used));
    const size_t offset = (size_t(prompt) - 1 + i) * D::vocab_used;
    rule.add(i, j + offset, uint32_t(chosen[i]), e + offset, t + offset, D::vocab_used, D::vocab_used, "Inkling");
    if (chosen[i] == 200006) break;
  }
  CHECK(rule.verdict("Inkling", "candidate"));

  for (uint32_t v = 0; v < D::visit_count; ++v) {
    const std::string layer = ".L" + std::to_string(v);
    const auto* r = ref.i32("route.ids" + layer, size_t(rows) * D::top_k);
    const auto* a = engine.i32("route.ids" + layer, size_t(rows) * D::top_k);
    const auto* b = twin.i32("route.ids" + layer, size_t(rows) * D::top_k);
    uint32_t miss = 0, twin_miss = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      const size_t at = size_t(row) * D::top_k;
      miss += !same_experts(r + at, a + at);
      twin_miss += !same_experts(r + at, b + at);
    }
    CHECK(miss <= twin_miss);
    const auto* x = ref.f32("H" + layer, size_t(rows) * D::hidden);
    const auto* y = engine.f32("H" + layer, size_t(rows) * D::hidden);
    golden::CosRows cos;
    for (uint32_t row = 0; row < rows; ++row) {
      const size_t at = size_t(row) * D::hidden;
      cos.c.push_back(golden::row_cos(x + at, y + at, D::hidden));
    }
    if (v < D::core_start) CHECK(cos.median() >= 0.9999);
    std::printf("  visit %u residual cosine %s, route differences %u / %u\n",
                v, cos.str().c_str(), miss, twin_miss);
  }
}
