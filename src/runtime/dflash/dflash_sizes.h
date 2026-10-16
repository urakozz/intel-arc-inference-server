#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/dflash_layout.h"
#include "model/dflash.h"
#include "runtime/memory_plan.h"

// Spec 19b Task 3: the DFlash drafter's device buffers as device-free arithmetic - the one formula DFlashBuffers
// allocates by (runtime/dflash/dflash_buffers.cc) and the memory planner counts (runtime::DrafterPlan:
// `--max-len auto` subtracts the drafter's weights, ring, taps and scratch before it sizes the KV cache). Host only
// (tests/runtime/dflash_plan_test.cc).
//
// Rows: the block's M = kernels::dflash::kM (8: anchor + up to 7 drafts) and the head's R = M - 1. `head_rows` is what
// the draft list's head GEMV produces per row: the target's lm_head rows, or a draft vocabulary's |V'|.
//
//   dctl, out    the control block and the readback (shared): runtime/dflash/dflash_control.h
//   rope         fp32 [max_len + M][head_dim]: cos | sin halves at every position a block row can reach
//   ring_k/_v    bf16 [layers][slots][kv heads][head_dim]: a sliding layer's slots = its window (the ring), a full
//                layer's = max_len (no published Qwen3.8 drafter has one; Ornith's v1 does - its stage builds it)
//   taps         bf16 [M][taps x target hidden]: the projection's input rows (19c's verify list writes them)
//   resid, h     fp32 [M][hidden]; x bf16 [M][hidden] (every norm's output and its GEMV copy)
//   xa           bf16 [M][max(hidden, q_n, inter)]: the conv's / attention's / SiLU's GEMV input
//   gout         fp32 [M][max(qkv_n, hidden, 2 inter)]: the block's GEMV outputs
//   co           fp32 [M][kproj_n] (DFlash 2): a conv's kernel projection (one live at a time)
//   q            fp32 [M][q_n]; kb, vb fp32 [M][kv_n] (the block's own K / V - never the ring's)
//   part         fp32 [q heads][ring / 128][M][head_dim + 2]: df_attn_part's blocks
//   kv           fp32 [layers][M][2 kv_n]: the projection's k || v rows
//   logits       fp32 [R][head_rows]; tpart fp32 [R][ceil(head_rows / 4096)][16][2]; cand u32 / unary fp32 [R][16]
//   hid          fp32 [M][selector rank] (DFlash 2)
namespace runtime::dflash {

struct DFlashSizes {
  size_t dctl = 0, out = 0, rope = 0, ring_k = 0, ring_v = 0, taps = 0, resid = 0, h = 0, x = 0, xa = 0, gout = 0,
         co = 0, q = 0, kb = 0, vb = 0, part = 0, kv = 0, logits = 0, tpart = 0, cand = 0, unary = 0, hid = 0;
  size_t total() const {
    return dctl + out + rope + ring_k + ring_v + taps + resid + h + x + xa + gout + co + q + kb + vb + part + kv +
           logits + tpart + cand + unary + hid;
  }
};

// The drafter's RoPE table: [rows][head_dim] fp32, cos in the first half, sin in the second, angle = pos x
// theta^(-2j / head_dim) in double, rounded once (tools/oracle/dflash_ref.py rope(); tests/kernels/dflash_ref.h
// rope_table is the same arithmetic - dflash_plan_test holds the two bitwise).
std::vector<float> rope_table(double theta, uint32_t head_dim, uint32_t rows);

// Every slot count of the ring for layer l: the window, or max_len for a full-attention layer.
uint32_t ring_slots(const model::DFlashDesc& d, uint32_t layer, uint32_t max_len);
DFlashSizes sizes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows, uint32_t max_len);

// The planner's term (runtime/memory_plan.h DrafterPlan): the weights (loader::dflash_weight_bytes) plus the buffers,
// split into what max_len does not move and what it adds per position (the RoPE rows; a full layer's ring rows).
DrafterPlan drafter_plan(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t head_rows,
                         loader::DraftFormat f);

// The draft list's and the projection list's launch counts, which the captures assert on themselves.
// draft: df_embed; per layer 16 (DFlash 2: norm, kernel projection, conv prepare, q||k||v, qkv prep, attention part
// + reduce, o, conv finish, norm, kernel projection, conv prepare, gate||up, SiLU, down, conv finish) or 12 (v1: no
// kernel projections, no conv prepare, an add for each finish); the final norm, the head, top-16 x 2, (v2) the
// selector projection, the walk. Qwen3.8's drafter: 87. project: fc, hidden_norm, per layer the k||v GEMV, the store
// (one launch for every layer): 3 + layers = 8.
size_t draft_launches(const model::DFlashDesc& d);
size_t project_launches(const model::DFlashDesc& d);

// Spec 19d (spec 19 decision 5): what a session snapshot adds with a drafter - its whole ring, ring_k then ring_v
// (sizes(...).ring_k + ring_v: 41,943,040 B for Qwen3.8's drafter, whatever max_len - every layer slides). The
// prefix store's entries grow by it (server::PrefixCache's state_bytes = Engine::state_bytes()).
size_t ring_snapshot_bytes(const model::DFlashDesc& d, uint32_t max_len);
// The drafter's name in the prefix store's keys (TapDrafter::key): FNV-1a 64 over the descriptor's shapes and taps,
// the weight format and `norms` (the drafter's hidden_norm and final_norm weights as loaded - two checkpoints of one
// shape differ there). Never 0 (0 means "no drafter" to server::store_root).
uint64_t drafter_key(const model::DFlashDesc& d, loader::DraftFormat f, const void* norms, size_t norm_bytes);

struct DFlashCtl;
// The one way to set a draft's K, temperature and seed in the control block (before a replay, under the drain rule):
// K must be 1 .. 7, and a DFlash (v1) drafter drafts greedily only - its sampled q is a softmax over the whole head,
// which the 16-candidate readback cannot carry, so T != 0 is refused here (std::invalid_argument naming v1) rather
// than drafted greedily under a sampled request's acceptance (19c's sampled rule would read the wrong q).
void set_draft_params(DFlashCtl& c, const model::DFlashDesc& d, uint32_t k, float temperature, uint64_t seed);

}  // namespace runtime::dflash
