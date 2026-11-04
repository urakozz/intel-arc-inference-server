#pragma once
// spec_plain_row.h - spec_judge::PlainRow for the card tests: the engine's plain (M = 1) logits row after a prompt and
// the ids decoded after it, by a replay - reset, prefill(prompt) on the engine's own prefill backend (the test's plain
// run prefills the same way), then one ingest() per id (the M = 1 decode list, each id forced). The row is the last
// ingest's logits (the prefill's last row when nothing follows the prompt). The engine's session is clobbered: the
// tests reset it before every run anyway (generate() resets its adapter).
#include <cstdint>
#include <vector>

#include "l0/cmdlist.h"
#include "runtime/engine.h"
#include "spec_judge.h"

namespace spec_judge {

inline PlainRow engine_plain_row(runtime::Engine& e, const std::vector<uint32_t>& prompt, uint32_t vocab) {
  return [&e, prompt, vocab](const std::vector<uint32_t>& generated) {
    e.reset();
    e.prefill(prompt);
    const l0::Mem* src = &e.prefill_scratch()->logits;
    for (uint32_t id : generated) {
      e.ingest({id});
      src = &e.buffers().logits;
    }
    std::vector<float> row(vocab);
    l0::CmdList imm = l0::CmdList::immediate(e.context());
    imm.copy(row.data(), src->ptr(), size_t(vocab) * sizeof(float));
    std::printf("      (plain replay for the bf16-unstable rule: prompt %zu ids + %zu decoded)\n", prompt.size(),
                generated.size());
    return row;
  };
}

}  // namespace spec_judge
