#include "check.h"
#include "runtime/inkling2/inkling2_prefill.h"

int main(int argc, char** argv) {
  CHECK_EQ(argc, 2);
  const loader::Inkling2Checkpoint checkpoint(argv[1]);
  const auto& d = checkpoint.desc();
  namespace ink = runtime::inkling2;

  const auto chunks = ink::prefill_chunks(d, 3700, 5000, 2048);
  CHECK_EQ(chunks.size(), 3u);
  CHECK_EQ(chunks[0].position, 3700u);
  CHECK_EQ(chunks[0].rows, 2048u);
  CHECK_EQ(chunks[1].position, 5748u);
  CHECK_EQ(chunks[1].rows, 2048u);
  CHECK_EQ(chunks[2].position, 7796u);
  CHECK_EQ(chunks[2].rows, 904u);
  for (const auto& chunk : chunks) {
    CHECK(chunk.visits.data() == d.visits.data());
    CHECK_EQ(chunk.visits.size(), 48u);
    for (uint32_t i = 0; i < 48; ++i) {
      const auto& visit = chunk.visits[i];
      const uint32_t layer = i < 6 ? i : i < 42 ? 6 + (i - 6) % 12 : i - 24;
      const uint32_t pass = i < 6 ? 0 : i < 42 ? (i - 6) / 12 : 2;
      CHECK_EQ(visit.layer, layer);
      CHECK_EQ(visit.pass, pass);
      CHECK_EQ(visit.conv_slot, i);
      CHECK_EQ(visit.controller, i >= 18 && i < 42);
      CHECK_EQ(visit.block_boundary, i % 6 == 0);
    }
  }

  const auto& weights = checkpoint.layout();
  const auto decode = ink::sizes(d, weights, 8192, true);
  const auto prompt = ink::sizes(d, weights, 8192, true, 2048);
  const auto plain = ink::sizes(d, weights, 8192, false, 2048);
  const auto& p = prompt.prefill;
  CHECK_EQ(runtime::gs::bytes(decode.prefill), 0u);
  CHECK_EQ(p.ids, size_t(8192) * sizeof(uint32_t));
  CHECK_EQ(p.residual, size_t(2048) * 6 * 2304 * sizeof(float));
  CHECK_EQ(p.block_bank, size_t(2048) * 4 * 2304 * sizeof(float));
  CHECK_EQ(p.draft_taps, size_t(2048) * 4 * 2304 * sizeof(uint16_t));
  CHECK_EQ(plain.prefill.draft_taps, 0u);
  CHECK_EQ(plain.prefill.draft_projection, 0u);
  CHECK_EQ(prompt.memory().prefill_scratch, runtime::gs::bytes(p));
  CHECK_EQ(prompt.memory().total() - decode.memory().total(), runtime::gs::bytes(p));
  CHECK_EQ(prompt.persistent.ring_k, decode.persistent.ring_k);
}
