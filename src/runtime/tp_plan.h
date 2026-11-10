#pragma once
// Spec 17b: a tensor-parallel card's memory plan (`--parallel tp`) - runtime::plan over the card's descriptor
// (model::tp_card_desc: half the heads' KV and GDN state, the scratch at the card's shapes; the loaded weights are the
// card's, loader::load_tp's report) plus the crossing's buffers (runtime/tp_protocol.h: the landing buffer, the two
// counter arrays, the state words, the folded row) in decode_state. Host arithmetic only (tests/runtime/tp_plan_test.cc).
// No prefill scratch (17c) and no MTP (17d): a decode-only engine, ruling R7's plan.
#include <array>
#include <cstddef>
#include <cstdint>

#include "model/model_desc.h"
#include "runtime/memory_plan.h"

namespace runtime {

// The crossing's device bytes on one card (TpEngine::Card's allocations).
size_t tp_crossing_bytes(const model::ModelDesc& card, model::TpCrossing c);

// One card's plan at max_len; `model_bytes` the card's loaded weights without the RoPE table.
MemoryPlan tp_card_plan(const model::ModelDesc& card, model::TpCrossing c, uint32_t max_len, size_t model_bytes,
                        KvCache kv = default_kv_cache());

// `--max-len auto` under TP: the largest multiple of kMaxLenQuantum, at most `cap`, that fits BOTH cards (each card's
// plan + `reserve_bytes` <= its device). 0 when not even min(kMinAutoMaxLen, cap) fits.
uint32_t tp_max_len_that_fits(const model::ModelDesc& card, model::TpCrossing c,
                              const std::array<size_t, 2>& model_bytes, const std::array<size_t, 2>& device_bytes,
                              size_t reserve_bytes, uint32_t cap, KvCache kv = default_kv_cache());

}  // namespace runtime
