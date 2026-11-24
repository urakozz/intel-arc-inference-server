#include <algorithm>
#include <array>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <vector>

#include "check.h"
#include "golden/forced_judge.h"
#include "runtime/inkling2/inkling2_engine.h"

namespace {

std::vector<uint32_t> read_prompt(const char* path) {
  std::ifstream file(path);
  std::vector<uint32_t> prompt;
  for (uint32_t id; file >> id;) prompt.push_back(id);
  CHECK(file.eof() && !prompt.empty());
  return prompt;
}

uint32_t ingest(runtime::inkling2::Engine& engine, std::span<const uint32_t> prompt) {
  uint32_t next = 0;
  for (size_t pos = 0; pos < prompt.size(); pos += runtime::inkling2::kRows) {
    const size_t n = std::min<size_t>(runtime::inkling2::kRows, prompt.size() - pos);
    next = engine.verify(prompt.subspan(pos, n)).back();
    engine.commit(uint32_t(n));
  }
  return next;
}

void measure_acceptance(runtime::inkling2::Engine& engine, const char* path, uint32_t eos) {
  constexpr uint32_t limit = 128;
  const auto prompt = read_prompt(path);
  CHECK(prompt.size() + limit <= engine.max_len());
  engine.reset();
  uint32_t next = ingest(engine, prompt);
  uint32_t proposed = 0, accepted = 0, generated = 0;
  std::vector<uint32_t> output;
  while (output.size() < limit) {
    const auto before = uint32_t(output.size());
    const auto count = std::min(7u, limit - before - 1);
    next = engine.step_drafted(next, count, output);
    const auto end = std::find(output.begin() + before, output.end(), eos);
    generated = uint32_t(end - output.begin()) + (end != output.end());
    proposed += count;
    accepted += generated - before - 1;
    if (end != output.end()) break;
  }
  const auto name = std::filesystem::path(path).filename().string();
  std::printf("Inkling DSpark %s accepted %u/%u drafts (%.1f%%) over %u output tokens through EOS or the sample limit\n",
              name.c_str(), accepted, proposed, proposed ? 100.0 * accepted / proposed : 0.0, generated);
}

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc == 4 || (argc >= 5 && std::string_view(argv[4]) == "draft"));
  const bool draft = argc >= 5;
  const auto prompt = read_prompt(argv[2]);
  const golden::Golden gold(argv[3]);
  const auto gen = gold.dim("tokens", 1, 0);
  const auto rows = gold.dim("logits", 2, 0);
  using D = model::Inkling2Desc;
  CHECK_EQ(rows, prompt.size() + gen);
  const auto* logits = gold.f32("logits", rows * D::vocab_used);
  const auto* tokens = gold.i32("tokens", gen);

  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  l0::Context ctx(0);
  runtime::inkling2::Engine engine(ctx, checkpoint, 4096, draft);
  const auto judge = [&](uint32_t index, uint32_t id) {
    const auto row = uint32_t(prompt.size() - 1 + index);
    const bool ok = golden::row_accepts(logits + size_t(row) * D::vocab_used, id, D::vocab_used, D::vocab_used);
    if (!ok) golden::judge_explain(gold, row, id, D::vocab_used, D::vocab_used, "engine");
    return ok;
  };
  if (draft) {
    const auto count = uint32_t(gen);
    uint32_t next = ingest(engine, prompt), steps = 0;
    std::vector<uint32_t> actual;
    while (actual.size() < count) {
      next = engine.step_drafted(next, std::min(7u, count - uint32_t(actual.size()) - 1), actual);
      ++steps;
    }
    CHECK(steps < count);
    uint32_t greedy_judged = 0;
    for (uint32_t i = 0; i < count; ++i) {
      CHECK(judge(i, actual[i]));
      ++greedy_judged;
      if (actual[i] != uint32_t(tokens[i]) || actual[i] == checkpoint.desc().eos) break;
    }

    // At a near tie the two valid answers can lead to different contexts.
    // Force the golden tokens through the draft verifier to judge the rest.
    engine.reset();
    next = ingest(engine, prompt);
    uint32_t graded = 0;
    while (graded < count) {
      CHECK(judge(graded, next));
      const auto anchor = uint32_t(tokens[graded]);
      if (graded + 1 == count || anchor == checkpoint.desc().eos) {
        ++graded;
        break;
      }
      const auto k = std::min(7u, count - graded - 1);
      std::array<uint32_t, runtime::inkling2::kRows> input{};
      input[0] = anchor;
      const auto draft_ids = engine.propose(anchor, k);
      std::copy(draft_ids.begin(), draft_ids.end(), input.begin() + 1);
      const auto predictions = engine.verify({input.data(), k + 1});
      uint32_t accepted = 1;
      while (accepted <= k && input[accepted] == uint32_t(tokens[graded + accepted]) &&
             input[accepted] != checkpoint.desc().eos) {
        CHECK(judge(graded + accepted, predictions[accepted - 1]));
        ++accepted;
      }
      next = predictions[accepted - 1];
      engine.commit(accepted);
      graded += accepted;
    }
    std::printf("Inkling DSpark generated %u ids in %u verifier steps, %u greedy and %u forced decisions judged\n",
                count, steps, greedy_judged, graded);
    for (int i = 5; i < argc; ++i) measure_acceptance(engine, argv[i], checkpoint.desc().eos);
    return 0;
  }
  uint32_t misses = 0, graded = 0;
  golden::CosRows drift;
  std::vector<float> actual(D::vocab_used);
  for (size_t p = 0; p + 1 < prompt.size();) {
    const auto count = uint32_t(std::min<size_t>(8, prompt.size() - 1 - p));
    engine.verify({prompt.data() + p, count});
    engine.commit(count);
    p += count;
  }
  uint32_t input = prompt.back();
  for (uint32_t i = 0; i < gen; ++i) {
    const auto index = uint32_t(prompt.size() - 1 + i);
    const auto* row = logits + size_t(index) * D::vocab_used;
    const auto next = engine.verify({&input, 1})[0];
    engine.read_logits(0, actual);
    drift.c.push_back(golden::row_cos(row, actual.data(), D::vocab_used));
    if (!judge(i, next)) ++misses;
    ++graded;
    if (uint32_t(tokens[i]) == checkpoint.desc().eos) break;
    engine.commit(1);
    input = uint32_t(tokens[i]);
  }
  std::printf("Inkling CPU golden judged %u/%u decisions, logit cosine min/median %s\n",
              graded - misses, graded, drift.str().c_str());
  CHECK_EQ(misses, 0u);
}
