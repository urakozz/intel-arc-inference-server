#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "golden/forced_judge.h"
#include "runtime/inkling2_prefill_test_access.h"

// inkling2_prefill_test SNAPSHOT GOLDEN [BF16_TWIN] [kv8|int8]: kv8 runs the engine with `--kv-cache int8` (the goldens
// then are inkling_ref.py --kv int8's, and the full visit's cache rows are de-quantised and un-rotated to compare);
// int8 the int8 prompt pass (`--prefill-backend l0-int8`) on that cache.
int main(int argc, char** argv) {
  const bool int8_pass = argc > 3 && std::string(argv[argc - 1]) == "int8";
  const bool kv8 = int8_pass || (argc > 3 && std::string(argv[argc - 1]) == "kv8");
  if (kv8) --argc;
  CHECK(argc == 3 || argc == 4);
  for (int i = 2; i < argc; ++i)
    if (!std::filesystem::exists(argv[i])) {
      std::printf("SKIP: %s is not made yet (tools/oracle/inkling_refs.sh)\n", argv[i]);
      return 77;
    }
  namespace ink = runtime::inkling2;
  using D = model::Inkling2Desc;
  const golden::Golden gold(argv[2]);
  const uint32_t length = uint32_t(gold.dim("prompt_ids", 1, 0));
  const uint32_t gen = uint32_t(gold.dim("tokens", 1, 0));
  const uint32_t rows = uint32_t(gold.dim("logits", 2, 0));
  const auto* tokens = gold.i32("tokens", gen);
  const auto* input = gold.i32("prompt_ids", length);
  const auto* logits = gold.f32("logits", size_t(rows) * D::vocab_used);
  CHECK_EQ(gen, 64u);
  CHECK(length > 0 && (rows == gen + 1 || rows == length + gen));
  std::vector<uint32_t> prompt(input, input + length);
  const bool thin = rows == gen + 1;
  const uint32_t row0 = thin ? 0 : length - 1;
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  l0::Context ctx(0);
  ink::Engine engine(ctx, checkpoint, length + gen + ink::kRows, true, ink::kPrefillRows, kv8, int8_pass);
  const auto& trace = ink::PrefillTestAccess::trace(engine);
  auto list = l0::CmdList::immediate(ctx);

  uint32_t next = engine.prefill(prompt, ink::kPrefillRows);
  auto read = std::vector<float>(D::vocab_used);
  const auto* expected = logits + size_t(row0) * D::vocab_used;
  engine.read_prefill_logits(read);
  const double first = golden::row_cos(expected, read.data(), D::vocab_used);
  std::printf("Inkling last prompt logit cosine %.8f at %u IDs\n", first, length);
  CHECK(golden::row_accepts(expected, next, D::vocab_used, D::vocab_used));

  std::vector<float> prelude(trace.size() / sizeof(float));
  list.copy(prelude.data(), trace.ptr(), trace.size());
  golden::CosRows hidden;
  for (uint32_t v = 0; v < D::core_start; ++v) {
    const auto name = "H.L" + std::to_string(v);
    const auto* want = gold.f32(name, size_t(thin ? 1 : rows) * D::hidden) + size_t(row0) * D::hidden;
    hidden.c.push_back(golden::row_cos(want, prelude.data() + v * D::hidden, D::hidden));
  }
  std::printf("Inkling prelude output cosine min/median %s\n", hidden.str().c_str());
  CHECK(hidden.median() >= .9999);

  if (thin) {
    const auto state = ink::PrefillTestAccess::state(engine);
    const auto scales = ink::PrefillTestAccess::scales(engine);
    const uint32_t sample = std::min(4u, length);
    for (uint32_t v : {0u, D::core_start - 1}) {
      const auto& visit = checkpoint.desc().visits[v];
      const uint32_t stride = visit.window ? ink::kRing : engine.max_len();
      const bool int8 = kv8 && !visit.window;
      for (uint32_t x = 0; x < 2; ++x) {
        const auto* cache = state[(visit.window ? 2 : 0) + x];
        const auto* data = static_cast<const std::byte*>(cache->ptr()) +
                           size_t(visit.kv_slot) * stride * D::kv_n * (int8 ? 1 : 2);
        const auto name = std::string("prefill.") + (x == 0 ? "K.L" : "V.L") + std::to_string(v);
        std::vector<float> actual(sample * D::kv_n);
        for (uint32_t r = 0; r < sample; ++r) {
          const uint32_t pos = (length - sample + r) % stride;
          if (int8) {   // q8 x scale, un-rotated: what the int8 cache gives the attention back
            std::array<int8_t, D::kv_n> codes{};
            std::array<uint16_t, D::kv_heads> s16{};
            list.copy(codes.data(), data + size_t(pos) * D::kv_n, sizeof(codes));
            list.copy(s16.data(), static_cast<const uint16_t*>(scales[x]->ptr()) +
                                      (size_t(visit.kv_slot) * stride + pos) * D::kv_heads, sizeof(s16));
            for (uint32_t h = 0; h < D::kv_heads; ++h) {
              float y[D::head_dim];
              for (uint32_t d = 0; d < D::head_dim; ++d) y[d] = common::kv8::dequant(codes[h * D::head_dim + d], s16[h]);
              common::kv8::unrotate_n<D::head_dim>(y, &actual[(h * sample + r) * D::head_dim]);
            }
            continue;
          }
          std::array<uint16_t, D::kv_n> kv{};
          list.copy(kv.data(), data + size_t(pos) * D::kv_n * 2, sizeof(kv));
          for (uint32_t h = 0; h < D::kv_heads; ++h)
            for (uint32_t d = 0; d < D::head_dim; ++d)
              actual[(h * sample + r) * D::head_dim + d] = common::bf16_to_f32(kv[h * D::head_dim + d]);
        }
        const double cos = golden::row_cos(gold.f32(name, actual.size()), actual.data(), uint32_t(actual.size()));
        std::printf("Inkling %s cosine %.8f\n", name.c_str(), cos);
        CHECK(cos >= golden::kBar);
      }
    }
  }

  std::unique_ptr<golden::Golden> twin;
  if (argc == 4) twin = std::make_unique<golden::Golden>(argv[3]);
  const float* bf16 = twin ? twin->f32("logits", size_t(rows) * D::vocab_used) : nullptr;
  golden::CosRows drift;
  golden::NearMiss near;
  std::vector<float> plain_logits(size_t(gen) * D::vocab_used);
  for (uint32_t i = 0; i < gen; ++i) {
    const auto off = size_t(row0 + i) * D::vocab_used;
    std::copy(read.begin(), read.end(), plain_logits.data() + size_t(i) * D::vocab_used);
    drift.c.push_back(golden::row_cos(logits + off, read.data(), D::vocab_used));
    if (bf16)
      near.add(i, logits + off, next, read.data(), bf16 + off, D::vocab_used, D::vocab_used, "plain");
    else
      CHECK(golden::row_accepts(logits + off, next, D::vocab_used, D::vocab_used));
    const uint32_t token = uint32_t(tokens[i]);
    if (i + 1 == gen || token == checkpoint.desc().eos) break;
    next = engine.verify({&token, 1})[0];
    engine.read_logits(0, read);
    engine.commit(1);
  }
  if (bf16) {
    CHECK(golden::cos_dist_ratio(first, golden::row_cos(expected, bf16 + size_t(row0) * D::vocab_used, D::vocab_used))
          <= golden::kCosSlack);
    CHECK(near.verdict("Inkling prompt pass", "plain"));
  }
  std::printf("Inkling prompt and plain decode logit cosine min/median %s\n", drift.str().c_str());

  engine.reset();
  next = engine.prefill(prompt, ink::kPrefillRows);
  CHECK(golden::row_accepts(expected, next, D::vocab_used, D::vocab_used));
  engine.read_prefill_logits(read);
  golden::NearMiss drafted;
  const auto judge_draft = [&](uint32_t index, uint32_t id) {
    const size_t offset = size_t(index) * D::vocab_used;
    drafted.add(index, expected + offset, id, read.data(), plain_logits.data() + offset,
                D::vocab_used, D::vocab_used, "drafted");
  };
  judge_draft(0, next);
  uint32_t judged = 1;
  for (uint32_t consumed = 0; consumed < gen;) {
    const uint32_t anchor = uint32_t(tokens[consumed]);
    const uint32_t room = std::min(7u, gen - consumed - 1);
    if (room == 0 || anchor == checkpoint.desc().eos) break;

    std::array<uint32_t, ink::kRows> proposal{};
    proposal[0] = anchor;
    const auto drafts = engine.propose(anchor, room);
    std::copy(drafts.begin(), drafts.end(), proposal.begin() + 1);
    const auto verify = engine.verify({proposal.data(), room + 1});
    uint32_t accepted = 1;
    while (accepted <= room && proposal[accepted] == uint32_t(tokens[consumed + accepted]) &&
           proposal[accepted] != checkpoint.desc().eos) {
      ++accepted;
    }
    // Verify predicts the next token after each committed input row.
    for (uint32_t row = 0; row < accepted && judged < gen; ++row) {
      engine.read_logits(row, read);
      judge_draft(judged++, verify[row]);
    }
    engine.commit(accepted);
    consumed += accepted;
  }
  CHECK(drafted.verdict("Inkling prompt pass", "drafted"));
  std::printf("Inkling DSpark after prompt pass judged %u forced decisions\n", judged);
}
