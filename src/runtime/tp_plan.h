#pragma once
// Spec 17b: a tensor-parallel card's memory plan (`--parallel tp`) - runtime::plan over the card's descriptor
// (model::tp_card_desc: half the heads' KV and GDN state, the scratch at the card's shapes; the loaded weights are the
// card's, loader::load_tp's report) plus the crossing's buffers (runtime/tp_protocol.h: the landing buffer, the two
// counter arrays, the state words, the folded row) in decode_state. Host arithmetic only (tests/runtime/tp_plan_test.cc).
// No MTP (17d). Spec 17c: with `path.prefill` the card's prefill too - the one-card scratch at the card's shapes
// (plan()'s PrefillPath term) and TpEngine's prefill buffers beside it (tp_prefill_extra_bytes); without it a
// decode-only engine, ruling R7's plan.
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>

#include "model/model_desc.h"
#include "runtime/memory_plan.h"

namespace runtime {

// Spec 17c: the chunks a TP prefill has in flight (the wavefront's pair), and where it cuts a prompt of `n` ids from
// absolute position `pos`: {position, rows} per chunk, in order, every two a pair - a pair takes up to 2 x `chunk`
// rows and splits them on the 64-grid (the first chunk min(chunk, half rounded up to 64), never shorter than the
// second), so a 2048-id prompt is two 1024-row chunks whose crossings hide behind each other's compute. `chunk` a
// positive multiple of 64. Host arithmetic (tp_plan_test).
inline constexpr uint32_t kTpPrefillChunks = 2;
std::vector<std::array<uint32_t, 2>> tp_prefill_chunks(uint32_t pos, size_t n, uint32_t chunk);
// The device bytes TpEngine's prefill holds on a card beside its PrefillScratch: the second chunk's residual, each
// chunk's own half of the row-parallel output and its two landing slots.
size_t tp_prefill_extra_bytes(const model::ModelDesc& card, uint32_t chunk_cap);

// The crossing's device bytes on one card (TpEngine::Card's allocations).
size_t tp_crossing_bytes(const model::ModelDesc& card, model::TpCrossing c);

// One card's plan at max_len; `model_bytes` the card's loaded weights without the RoPE table.
MemoryPlan tp_card_plan(const model::ModelDesc& card, model::TpCrossing c, uint32_t max_len, size_t model_bytes,
                        KvCache kv = default_kv_cache(), const PrefillPath& path = PrefillPath{false});

// `--max-len auto` under TP: the largest multiple of kMaxLenQuantum, at most `cap`, that fits BOTH cards (each card's
// plan + `reserve_bytes` <= its device). 0 when not even min(kMinAutoMaxLen, cap) fits.
uint32_t tp_max_len_that_fits(const model::ModelDesc& card, model::TpCrossing c,
                              const std::array<size_t, 2>& model_bytes, const std::array<size_t, 2>& device_bytes,
                              size_t reserve_bytes, uint32_t cap, KvCache kv = default_kv_cache(),
                              const PrefillPath& path = PrefillPath{false});

}  // namespace runtime
