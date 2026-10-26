#pragma once
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "runtime/capture.h"   // CapturedStep - the same record
#include "runtime/dflash/dflash_buffers.h"

// Spec 19b Task 3: the DFlash drafter's two lists, each captured once and replayed (spec 19 §5 "Per iteration"):
//
// **The draft list** (build_dflash_draft): one block of M = 8 rows at the anchor's position p = ctrl.pos, the anchor
// ctrl.cur_token[0], K = dctl.k drafts (1..7: rows past K are computed and masked - one list for every K), the
// walk's temperature and seed from dctl. Writes the K drafts into ctrl.cur_token[1..K] (what the verify list reads,
// spec 8 / 19e) and the readback (DFlashDraftOut: per row the 16 candidates - real ids -, their head logits, the
// selector's scores, the choice). In order (dflash_sizes.h draft_launches: 87 for Qwen3.8's DFlash 2 drafter):
//   df_embed (the target's embedding row of the anchor, the mask row, x input_embedding_scale)
//   per layer   df_norm (input_layernorm) - [v2: attention_conv's kernel projection GEMV, df_conv_prep] - q||k||v
//               GEMV - df_qkv_prep (q / k norms, RoPE at p + r; the block's K / V stay in kb / vb) - df_attn_part +
//               df_attn_reduce (the ring's window + the block, non-causal) - o GEMV - df_conv_add (v1: df_add) -
//               df_norm (post_attention_layernorm) - [v2: mlp_conv's kernel projection, df_conv_prep] - gate||up
//               GEMV - df_silu_mul - down GEMV - df_conv_add (v1: df_add)
//   df_norm (the final norm) - the TARGET's lm_head over rows 1..7 (gemv_i8w / gemv_bf16 at M 7: the full head or
//   the draft vocabulary's compact one, spec 8 §11) - df_topk_stage1/2 (the vocab mask; a draft vocabulary's indices
//   mapped to ids) - [v2: the selector's hidden_projection GEMV] - df_select
//
// **The projection list** (build_dflash_project): dctl.proj_n <= 8 tap rows (DFlashBuffers::taps, bf16
// [8][taps x target hidden]) at positions dctl.proj_pos .. + proj_n - 1 into the ring: fc GEMV, df_norm (hidden_norm),
// per layer the k||v rows of its q||k||v weight (the same weight the block uses - no copy), df_ctx_store (k_norm,
// RoPE, bf16, slot pos % ring; rows past proj_n are not stored). 8 launches. It is 19c's commit-side projection of
// the committed rows and its prefill tail; here it is how a test or the D1 tool fills the ring.
//
// The drafter's embedding and head are the TARGET's (Review Focus 3): DFlashTarget points at them.
namespace runtime::dflash {

struct DFlashTarget {
  const l0::Mem* embed = nullptr;               // bf16 [rows][hidden]
  const loader::DeviceWeight* head = nullptr;   // the lm_head (int8 or bf16) or the draft vocabulary's compact head
  const l0::Mem* ids = nullptr;                 // the draft vocabulary's id table (u32, ascending); null: the full head
  uint32_t vocab_rows = 0;                      // the full head's rows (the codebooks' rows)
  uint32_t vocab_used = 0;                      // ids below it are tokens (the full head's mask)
  uint32_t head_rows() const { return head ? head->shape.N : 0; }
};
// The target's full head (`draft_vocab` false) or its draft vocabulary's compact one (LoadedModel::draft_vocab).
DFlashTarget dflash_target(const loader::LoadedModel& m, bool draft_vocab = false);

CapturedStep build_dflash_draft(l0::Context& ctx, const loader::DFlashWeights& w, const DFlashTarget& t,
                                DFlashBuffers& b, const l0::Mem& control);
CapturedStep build_dflash_project(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b);

}  // namespace runtime::dflash
