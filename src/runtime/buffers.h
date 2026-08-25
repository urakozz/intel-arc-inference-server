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
  // KV positions per `attn_decode` work-group - spec 1.5's lever L5 took it
  // from 256 to 64. It is the ONE home for the number on the host: `attn_part`
  // below is strided by `max_len / kAttnBlock` blocks, capture.cc launches
  // `attn_decode` with exactly that many, and it is the `_B64` half of both
  // compiled variant names (`kernels::attn_decode_variant`), so a value here
  // that no compiled binary matches throws at capture instead of striding
  // `attn_part` at one size while the kernel writes it at another.
  //
  // **64 is measured, not derived - but it is not a knee either.** docs/15 §2's
  // `F + fill·P` fit predicted 214.5 µs/launch at a 128-position block; the
  // retile measured 296.684. So the block size was swept in situ at depth 4096
  // instead - `attn_decode` µs/launch, then the attn family's ms/token:
  //
  //     B256 369.988 / 6.058   B128 296.684 / 4.931
  //     B64  224.046 / 3.839   B32  203.837 / 3.769
  //
  // B32's family total is **0.070 ms/token better than B64**, so there is no
  // crossover to point at: the step-Σ difference that once looked like one
  // (+19.5 µs) is inside the +0.283% drift the untouched launches showed
  // between those two runs. 64 is chosen because the marginal gain has
  // collapsed (−1.127, −1.092, −0.070 ms/token for the three halvings, the last
  // about a third of one run's drift), because `attn_reduce` is on a steep ramp
  // (80 → 127 → 196 → 450 µs/step), and because B32 would double `attn_part`
  // again - 50.7 → 101.4 MB, per-step scratch 77.6 → 128.3 MB - to buy that
  // 0.070 ms. docs/12 `attn` → Measured carries the arithmetic.
  static constexpr uint32_t kAttnBlock = 64;
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
  l0::Mem attn_part;      // fp32 [24][max_len/kAttnBlock][M][258] (m, l, acc[256]) = 50.72 MB @16384
  l0::Mem attn_out;       // bf16 [M][6144]
  l0::Mem logits;         // fp32 [M][248320] = 7.95 MB
  l0::Mem argmax_part;    // fp32+idx pairs, stage-1 output: [M][243][2]
  uint32_t max_len;

  size_t persistent_bytes() const;   // printed at startup, asserted by the test
  size_t scratch_bytes() const;
};
}  // namespace runtime
