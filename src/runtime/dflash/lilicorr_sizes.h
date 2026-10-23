#pragma once
#include <cstddef>
#include <cstdint>

#include "loader/lilicorr_layout.h"
#include "model/dflash.h"
#include "runtime/bolt/bolt_sizes.h"

// Spec 26b Task 5: the LiLiCorr drafter's device buffers as device-free arithmetic - the one formula LiliDrafter allocates
// by (runtime/dflash/lilicorr_drafter.cc) and Bolt's planner counts (runtime::bolt::BoltReserve: the drafter and its
// embedding replica on the last device, spec 26 Decision 8). Host only (tests/runtime/lilicorr_plan_test.cc).
//
// Rows: the block's M = 8 (anchor + 7 masks, always the whole block), the head's 7 mask rows, the lattice's R = 56.
//   dctl, out     LiliCtl and the readback LiliDraftOut (shared): runtime/dflash/dflash_control.h
//   rope          fp32 [max_len + M][head_dim] (dflash_sizes.h rope_table): every position a block row can reach
//   ring_k / _v   bf16 [layers][1024][kv heads][head_dim] - 5,120 B a position, 5,242,880 B over the ring (spec 26 §4)
//   anchor        fp32 [hidden]: the newest committed row's hidden_norm(fc(taps)) - the projection list writes it, the
//                 draft list's context_proj reads it (Review Focus 5)
//   taps          bf16 [M][6 x target hidden]: the projection's input rows (26c's verify list writes them)
//   resid, h      fp32 [M][hidden]; x bf16 [M][hidden]; xa bf16 [M][max(hidden, q_n, inter)]; gout fp32 [M][max(qkv_n,
//                 hidden, 2 inter)]; co fp32 [M][kproj_n]; q fp32 [M][q_n]; kb, vb fp32 [M][kv_n]
//   part          fp32 [q heads][1024 / 64][M][head_dim + 2]: ll_attn_part's blocks
//   kv            fp32 [layers][M][2 kv_n]: the projection's k || v rows
//   logits        fp32 [7][vocab]; tpart fp32 [7][ceil(vocab / 4096)][2 x 8 + 2]; cand u32 / logit / lp fp32 [7][8]
//   the head's scratch (fp32): tok [R][Hl], pass [7][Hl], feat [R][5], fu / fm / xh / xn / o / fh [R][Hl] each, qkv [R][3 Hl],
//                 u [R][lil_mlp], a / an [Hl], ov [R][2 F], ao [F], lattice [7][8][8]
namespace runtime::dflash {

struct LiliSizes {
  size_t dctl = 0, out = 0, rope = 0, ring_k = 0, ring_v = 0, anchor = 0, taps = 0, resid = 0, h = 0, x = 0, xa = 0,
         gout = 0, co = 0, q = 0, kb = 0, vb = 0, part = 0, kv = 0, logits = 0, tpart = 0, cand = 0, logit = 0, lp = 0,
         head = 0;   // `head`: the LiLiCorr head's scratch, one allocation (lili_head_scratch's offsets)
  size_t total() const {
    return dctl + out + rope + ring_k + ring_v + anchor + taps + resid + h + x + xa + gout + co + q + kb + vb + part + kv +
           logits + tpart + cand + logit + lp + head;
  }
};

// The head's scratch offsets inside LiliSizes::head (floats).
struct LiliHeadScratch {
  size_t tok = 0, pass = 0, feat = 0, fu = 0, fm = 0, xh = 0, xn = 0, qkv = 0, o = 0, u = 0, a = 0, an = 0, fh = 0, ov = 0,
         ao = 0, lat = 0, total = 0;
};
LiliHeadScratch lili_head_scratch(const model::DFlashDesc& d);

// The lists' rows (8) and the ring's slots (the window: 1024).
uint32_t lili_rows(const model::DFlashDesc& d);
uint32_t lili_ring_slots(const model::DFlashDesc& d);
LiliSizes lili_sizes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab, uint32_t max_len);

// The drafter's whole device bytes at max_len: its weights in `f` (loader::lilicorr_weight_bytes at the published NVFP4
// set) plus its buffers; and Bolt's reserve terms on the last device - `drafter` those bytes, `drafter_embed_replica` the
// row-major bf16 [vocab][hidden] embedding the draft list gathers from (1,073,741,824 B for Bolt); `base`'s other terms
// (26c's verify_slots, the scratch) as given. `max_len` sizes the RoPE table: `--max-len auto` passes the cap it sizes to.
size_t lili_drafter_bytes(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab, loader::DraftFormat f,
                          uint32_t max_len);
runtime::bolt::BoltReserve lili_bolt_reserve(const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab,
                                             loader::DraftFormat f, uint32_t max_len,
                                             runtime::bolt::BoltReserve base = {});

// The draft list's and the projection list's launch counts, which the captures assert on themselves. draft: df_embed; per
// layer 16 (df_norm, the attention conv's kernel projection, df_conv_prep, q||k||v, df_qkv_prep, ll_attn_part + reduce, o,
// df_conv_add, df_norm, the MLP conv's projection, df_conv_prep, gate||up, df_silu_mul, down, df_conv_add); the final
// df_norm, the head, ll_topk x 2; the LiLiCorr head (gather, pass, feat, feature up / down, sum, context_proj, valid, per
// head layer 7, the two final norms, the factor, out / in heads, its L2 norm, anchor_out_head, its L2 norm, the scores:
// 16 + 7 x lil_layers); ll_select. The published drafter: 116. project: fc, df_norm (hidden_norm), ll_anchor, per layer
// the k||v GEMV, df_ctx_store: 4 + layers = 9.
size_t lili_draft_launches(const model::DFlashDesc& d);
size_t lili_project_launches(const model::DFlashDesc& d);

// What a session snapshot adds with the drafter (spec 26 §6, 19d's rule): its ring (ring_k then ring_v) and the anchor -
// 5,242,880 + 16,384 B for the published drafter, whatever max_len.
size_t lili_ring_snapshot_bytes(const model::DFlashDesc& d);
// The drafter's name in the prefix store's keys (TapSink::key): dflash_sizes.h drafter_key's FNV-1a 64 with the LiLiCorr
// form and its head's shapes folded in. Never 0.
uint64_t lili_drafter_key(const model::DFlashDesc& d, loader::DraftFormat f, const void* norms, size_t norm_bytes);

}  // namespace runtime::dflash
