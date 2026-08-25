#pragma once
#include <cstdint>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "model/qwen35.h"

namespace runtime {
// All decode-step state and scratch. Allocated once; every pointer is baked
// into the captured list. M = 8 capacity from day one (spec §6.5) even
// though this plan compiles M = 1 kernels only.
struct DecodeBuffers {
  static constexpr uint32_t kM = 8;
  static constexpr uint32_t kConvRing = 16;     // ring depth >= M + 3 (spec §9.4)
  static constexpr uint32_t kAttnBlock = 256;   // KV positions per attn_decode work-group
  // **prep_res_norm's two-stage grid** (spec 1.5 lever L1). Stage A
  // (`prep_res_fold`) runs this many work-groups over the hidden row and writes
  // one fp32 sum-of-squares each; stage B (`prep_norm_finish`) folds exactly
  // these, in ascending index order, into the rms. It is the ONE home for the
  // number: `norm_sumsq` below is sized from it, src/runtime/capture.cc
  // launches both grids from it AND puts it in both variant names
  // (`kernels::prep_res_fold_variant`), so a change here that no compiled
  // binary matches throws at capture instead of reducing the wrong count.
  //
  // 20 at kHidden = 5120 is one element per lane in a 256-lane work-group:
  // 320 subgroups against the single-work-group kernel's 16, which is the axis
  // docs/15 §L2 measured as the one that pays.
  static constexpr uint32_t kNormGroups = 20;

  DecodeBuffers(l0::Context& ctx, uint32_t max_len);

  // --- persistent state (survives across tokens) ---
  l0::Mem control;        // shared, sizeof(Control)
  l0::Mem gdn_state;      // fp32 [48 layers][48 heads][128 k][128 v]  = 150.99 MB
  l0::Mem conv_ring;      // bf16 [48 layers][16][10240]               = 15.73 MB
  l0::Mem kv_k, kv_v;     // bf16 [16 layers][max_len][4][256] each    = 536.87 MB each @16384
  // --- per-step scratch (overwritten every token) ---
  l0::Mem resid;          // bf16 [M][5120]
  l0::Mem x;              // bf16 [M][17408]  (prep output; largest K)
  l0::Mem partials;       // fp32 [16][M][34816] (max S x max N)       = 17.83 MB
  l0::Mem ab_out;         // fp32 [M][128]    (a||b GEMV output, S=1)
  l0::Mem norm_sumsq;     // fp32 [kNormGroups][M] (prep_res_fold -> prep_norm_finish)
  l0::Mem gdn_o;          // fp32 [M][48][128] (gdn_step output, pre gated-norm)
  l0::Mem attn_q;         // fp32 [M][24][256] (post norm+rope)
  l0::Mem attn_gate;      // fp32 [M][24][256]
  l0::Mem attn_part;      // fp32 [24][max_len/256][M][258] (m, l, acc[256]) = 12.68 MB @16384
  l0::Mem attn_out;       // bf16 [M][6144]
  l0::Mem logits;         // fp32 [M][248320] = 7.95 MB
  l0::Mem argmax_part;    // fp32+idx pairs, stage-1 output: [M][243][2]
  uint32_t max_len;

  size_t persistent_bytes() const;   // printed at startup, asserted by the test
  size_t scratch_bytes() const;
};
}  // namespace runtime
