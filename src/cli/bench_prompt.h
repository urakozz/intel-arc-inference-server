#pragma once
// b70-decode's --bench / --profile prompt, and b70-serve's load-time measurement of Inkling's round costs.
#include <cstddef>
#include <cstdint>

namespace cli {

// The bench prompt, cycled to --depth: the 42 ids of the committed golden
// prompt `tests/golden/prompts/prose.ids`, baked in so a bench run needs
// nothing but the binary and the checkpoint. Every id is below kVocabUsed, so
// none of them is a row the sampler masks. What matters for a timing run is
// that the ids are fixed and legal, not what they say - decode reads every
// weight per token regardless - but reusing the golden prompt means the depth-N
// bench and the golden gate ingest the same tokens for their first 42 steps.
inline constexpr uint32_t kBenchPrompt[] = {
    760,   72103, 506, 37119, 557,   11012, 3213, 310,   6512, 279, 61789, 272, 1072,  2272,
    279,   197616, 2271, 13,   469,   68042, 29123, 7247, 383,  279, 1387,  12615, 1345, 279,
    49813, 78911, 1141, 20459, 13,    3113,  7840,  279,  2981, 1000, 381,  16850, 1495, 13};
inline constexpr size_t kBenchPromptLen = sizeof(kBenchPrompt) / sizeof(kBenchPrompt[0]);

}  // namespace cli
