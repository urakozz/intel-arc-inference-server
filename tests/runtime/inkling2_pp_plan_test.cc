// `--parallel pp`'s placement and plan (runtime/inkling2/inkling2_sizes.h): the split at visit 24, what each card
// holds, its bytes against spec 41 §5.3, the crossings a token, and the contexts both cards hold.
#include <cstdio>
#include <string>

#include "check.h"
#include "runtime/inkling2/inkling2_sizes.h"

int main(int argc, char** argv) {
  CHECK_EQ(argc, 2);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  const auto& d = checkpoint.desc();
  const auto& w = checkpoint.layout();
  namespace ink = runtime::inkling2;
  using D = model::Inkling2Desc;
  using L = loader::Inkling2WeightLayout;

  const auto stages = ink::pp_stages(true);
  CHECK_EQ(stages[0].first, 0u);
  CHECK_EQ(stages[0].end, 24u);
  CHECK(stages[0].taps && !stages[0].head());
  CHECK_EQ(stages[1].first, 24u);
  CHECK(stages[1].head() && !stages[1].taps);
  CHECK(!ink::pp_stages(false)[0].taps);

  // A card a visit: one crossing a token, card 0 -> card 1 between visits 23 and 24.
  uint32_t crossings = 0;
  for (uint32_t i = 1; i < D::visit_count; ++i) crossings += (i >= stages[1].first) != (i - 1 >= stages[1].first);
  CHECK_EQ(crossings, 1u);

  // Card 0: layers 0-17 (the prelude, pass 1, pass 2's layers 6-11), the full visits 5, 11, 17, 23; card 1: layers
  // 6-23, the full visits 29, 35, 41, 47. Twenty window visits each; the slots count on from card 0's.
  const auto s0 = ink::stage_share(d, w, stages[0]), s1 = ink::stage_share(d, w, stages[1]);
  for (uint32_t l = 0; l < D::layers; ++l) {
    CHECK_EQ(s0.layers[l], l < 18);
    CHECK_EQ(s1.layers[l], l >= 6);
  }
  CHECK_EQ(s0.full, 4u);
  CHECK_EQ(s0.window, 20u);
  CHECK_EQ(s0.full_base + s0.window_base, 0u);
  CHECK_EQ(s1.full, 4u);
  CHECK_EQ(s1.window, 20u);
  CHECK_EQ(s1.full_base, 4u);
  CHECK_EQ(s1.window_base, 20u);
  size_t twice = 0;   // the core's layers, held by both cards
  for (uint32_t l = D::core_start; l < D::core_end; ++l) twice += w.bytes[L::kFirstLayer + l];
  CHECK_EQ(s0.layer_bytes + s1.layer_bytes, w.layer_bytes() + twice);

  constexpr uint32_t len = 32768, rows = 4096;
  const auto pp = ink::pp_sizes(d, w, len, true, rows, false, false);
  const auto one = ink::sizes(d, w, len, true, rows);
  constexpr size_t row = ink::handoff_row_bytes(true);
  CHECK_EQ(row, size_t(5 * 2304 * 4 + 4 * 2304 * 2));   // 64,512 B a row
  CHECK_EQ(ink::handoff_row_bytes(false), size_t(5 * 2304 * 4));
  // Each card: its visits' caches; together the one card's.
  CHECK_EQ(pp[0].persistent.full_k + pp[1].persistent.full_k, one.persistent.full_k);
  CHECK_EQ(pp[0].persistent.ring_k + pp[1].persistent.ring_k, one.persistent.ring_k);
  CHECK_EQ(pp[0].persistent.conv_attn + pp[1].persistent.conv_attn, one.persistent.conv_attn);
  CHECK_EQ(pp[0].weights.model, w.bytes[L::kModel]);   // the embedding, the head and the shared tables on both
  CHECK_EQ(pp[1].weights.model, w.bytes[L::kModel]);
  CHECK_EQ(pp[0].weights.draft, 0u);
  CHECK_EQ(pp[1].weights.draft, w.bytes[L::kDraft]);
  CHECK_EQ(runtime::gs::bytes(pp[0].draft), 0u);
  CHECK(pp[0].scratch.draft_taps && pp[0].prefill.draft_taps && !pp[0].prefill.draft_projection);
  // The hand-off: card 1 lands a verify's 8 rows and a chunk's; card 0 sends a chunk from its outbox.
  CHECK_EQ(pp[0].handoff.landing + pp[0].handoff.prefill_landing, 0u);
  CHECK_EQ(pp[0].handoff.prefill_outbox, rows * row);
  CHECK_EQ(pp[1].handoff.landing, ink::kRows * row);
  CHECK_EQ(pp[1].handoff.prefill_landing, rows * row);
  CHECK_EQ(pp[1].handoff.prefill_outbox, 0u);
  CHECK_EQ(runtime::gs::bytes(one.handoff), 0u);
  // A B70 as the planner sees it.
  constexpr size_t card = 32'530'000'000ull, reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  for (uint32_t i = 0; i < 2; ++i) {
    const auto label = "card " + std::to_string(i) + ", 32k positions, 4,096-row chunks";
    std::printf("%s\n", runtime::format_memory(label.c_str(), pp[i].memory(), card).c_str());
  }

  // The contexts: 1 Mi with the int8 cache and the drafter at 2,048-row chunks (spec 41 §5.3); bf16, 4,096-row
  // chunks and drafter-off rows printed for the record.
  CHECK_EQ(ink::pp_max_len_that_fits(d, w, card, reserve, true, 2048, true), D::trained_max_len);
  const auto spare = ink::pp_sizes(d, w, D::trained_max_len, true, 2048, true);
  for (uint32_t i = 0; i < 2; ++i)
    std::printf("1 Mi, int8 cache, drafter, 2,048-row chunks: card %u %.2f GB spare\n", i,
                double(card - reserve - spare[i].memory().total()) / 1e9);
  // Each card's 4 full visits in int8 codes and fp16 scales, card 1 the drafter's bf16 full layer (spec 41 §3.2: ~16.5
  // KiB a position and the drafter's 4 KiB).
  for (uint32_t i = 0; i < 2; ++i) {
    CHECK_EQ(spare[i].persistent.full_k, size_t(4) * D::trained_max_len * D::kv_n);
    CHECK_EQ(spare[i].persistent.full_k_scale, size_t(4) * D::trained_max_len * D::kv_heads * 2);
  }
  CHECK_EQ(spare[1].draft.draft_full_k, size_t(D::trained_max_len) * D::kv_n * 2);
  CHECK_EQ(runtime::gs::bytes(spare[0].draft), 0u);
  // The prompt chunk: 2,048 rows at 1 Mi with the drafter (4,096 leaves card 1 short), 4,096 where both cards hold
  // the context.
  CHECK_EQ(ink::pp_prefill_rows(d, w, card, reserve, D::trained_max_len, true, true), 2048u);
  CHECK_EQ(ink::pp_prefill_rows(d, w, card, reserve, 524288, true, true), ink::kPrefillRows);
  CHECK_EQ(ink::pp_prefill_rows(d, w, card, reserve, D::trained_max_len, false, true), ink::kPrefillRows);
  CHECK_EQ(ink::pp_prefill_rows(d, w, card, reserve, D::trained_max_len, true, false), 0u);
  for (const bool kv8 : {false, true})
    for (const uint32_t chunk : {2048u, 4096u})
      for (const bool draft : {true, false})
        std::printf("pp, %s cache, %u-row chunks, drafter %s: %u positions (one card %u)\n", kv8 ? "int8" : "bf16",
                    chunk, draft ? "on" : "off", ink::pp_max_len_that_fits(d, w, card, reserve, draft, chunk, kv8),
                    ink::max_len_that_fits(d, w, card, reserve, draft, chunk, kv8));
}
