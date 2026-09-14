// Host-only: the sampler function over a synthetic logits row (plan 7d Task
// 5, spec §3.5). Never touches a device -- no Engine, no adapter object is
// instantiated, only `sample()` from serve_adapters.h.
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "check.h"
#include "cli/serve_adapters.h"
#include "server/deps.h"

namespace {

server::Sampling greedy_free_sampling() {
  server::Sampling s;
  s.greedy = false;  // sample() itself does not look at `greedy` -- the
                     // caller (EngineAdapter::step) is what skips it.
  s.temperature = 1.0f;
  s.top_k = 20;
  s.top_p = 0.95f;
  return s;
}

}  // namespace

int main() {
  // 1. One dominant logit, temperature 1e-3: the softmax is so peaked that
  //    the argmax must win essentially every draw.
  {
    constexpr uint32_t kVocab = 100;
    std::vector<float> logits(kVocab, 0.0f);
    const uint32_t dominant = 37;
    logits[dominant] = 100.0f;  // every other id sits at 0

    server::Sampling s = greedy_free_sampling();
    s.temperature = 1e-3f;
    s.top_k = kVocab;  // no cap for this case
    s.top_p = 1.0f;

    std::mt19937_64 rng(12345);
    int hits = 0;
    for (int i = 0; i < 1000; ++i) {
      if (sample(logits.data(), kVocab, s, rng) == dominant) ++hits;
    }
    CHECK_EQ(hits, 1000);
  }

  // 2. top_k 1: whatever temperature/top_p say, only the argmax is ever a
  //    candidate.
  {
    constexpr uint32_t kVocab = 50;
    std::vector<float> logits(kVocab);
    for (uint32_t i = 0; i < kVocab; ++i) logits[i] = static_cast<float>(i);
    const uint32_t argmax = kVocab - 1;  // logits are strictly increasing

    server::Sampling s = greedy_free_sampling();
    s.top_k = 1;
    s.temperature = 1.0f;
    s.top_p = 0.95f;

    std::mt19937_64 rng(777);
    for (int i = 0; i < 200; ++i) {
      CHECK_EQ(sample(logits.data(), kVocab, s, rng), argmax);
    }
  }

  // 3. A flat-in-effect row (a strictly decreasing but near-uniform logit
  //    ramp, so the sort order is unambiguous -- id i has the i-th highest
  //    logit, no ties, no dependence on std::partial_sort's tie-break) with
  //    top_p 0.5: the resulting probabilities are near-uniform (a 1e-4
  //    logit step is negligible next to temperature 1.0), so cumulative
  //    mass 0.5 is crossed just past the 5th of 10 nearly-equal entries --
  //    i.e. top_p keeps ids {0..4}, "the first half of the sorted set", and
  //    every draw must land there.
  {
    constexpr uint32_t kVocab = 10;
    std::vector<float> logits(kVocab);
    for (uint32_t i = 0; i < kVocab; ++i) logits[i] = -1e-4f * static_cast<float>(i);

    server::Sampling s = greedy_free_sampling();
    s.top_k = kVocab;
    s.temperature = 1.0f;
    s.top_p = 0.5f;

    std::mt19937_64 rng(99);
    for (int i = 0; i < 200; ++i) {
      const uint32_t id = sample(logits.data(), kVocab, s, rng);
      CHECK(id < kVocab / 2);
    }
  }

  // 4. Seeded -> reproducible: two independently seeded RNGs with the same
  //    seed, run over the same row and Sampling, draw the identical
  //    sequence.
  {
    constexpr uint32_t kVocab = 30;
    std::vector<float> logits(kVocab);
    for (uint32_t i = 0; i < kVocab; ++i) {
      logits[i] = static_cast<float>((i * 7 + 3) % kVocab);  // an arbitrary shape
    }
    server::Sampling s = greedy_free_sampling();
    s.top_k = 8;
    s.temperature = 0.9f;
    s.top_p = 0.9f;
    s.has_seed = true;
    s.seed = 424242;

    std::mt19937_64 rng_a(s.seed);
    std::mt19937_64 rng_b(s.seed);
    for (int i = 0; i < 50; ++i) {
      const uint32_t a = sample(logits.data(), kVocab, s, rng_a);
      const uint32_t b = sample(logits.data(), kVocab, s, rng_b);
      CHECK_EQ(a, b);
    }
  }

  std::printf("sampling_test: 4/4 cases passed\n");
  return 0;
}
