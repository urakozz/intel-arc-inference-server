#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

namespace l0 {
class CmdList;
}

namespace runtime {

// The decode step's allocations, split three ways (spec 2 §3.6, plan 6b Task 1)
// **without moving one byte of decode's**:
//
//   PersistentBuffers - everything that survives a token boundary and is SHARED
//                       between the captured decode list and the prefill path.
//   DecodeScratch     - decode's per-step scratch, sized by kM = 8.
//   PrefillScratch    - the prefill path's per-chunk scratch, sized by kC.
//
// `DecodeBuffers` survives as a **view** over the first two, with every public
// field name it ever had, so `capture.cc`'s 129 `res_norm` sites and its 17
// buffer bindings are untouched and the 774/19 launch/module invariants hold
// byte for byte. Its `(ctx, max_len)` constructor still OWNS both groups, so
// every existing construction site compiles and behaves exactly as before.
//
// **Ruling R7: `PrefillScratch` is allocated lazily**, on the first
// `Engine::prefill()`. A decode-only Engine therefore has byte-identical device
// residency to today's, which is what makes spec §6.5 ("decode is untouched")
// checkable rather than argued.

// Everything that survives a token boundary. Layouts are decode's, unchanged.
struct PersistentBuffers {
  static constexpr uint32_t kConvRing = 16;   // ring depth >= M + 3 (spec §9.4)

  // Spec 14: per-layer state is sized from the model descriptor (gdn_layers,
  // fa_layers); there is no default - a caller has to say which model.
  PersistentBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  l0::Mem control;        // shared, sizeof(Control)
  l0::Mem gdn_state;      // fp32 [gdn_layers][48 heads][128 k][128 v]  = 150.99 MB (Qwen3.8: 48)
  l0::Mem conv_ring;      // bf16 [gdn_layers][16][10240]               = 15.73 MB (Qwen3.8)
  l0::Mem kv_k, kv_v;     // bf16 [fa_layers][max_len][4][256] each     = 536.87 MB each @16384 (Qwen3.8: 16)
  uint32_t max_len;

  size_t bytes() const;
  // The five fills Engine::reset() does, in the order the constructor does
  // them. It is one place, so "reset writes exactly what construction wrote"
  // is a property of the code rather than of two lists agreeing.
  void zero(l0::CmdList& imm);
};

// Decode's per-step scratch. Field names, sizes and allocation order unchanged.
struct DecodeScratch {
  static constexpr uint32_t kM = 8;
  // KV positions per `attn_decode` work-group - spec 1.5's lever L5 took it
  // from 256 to 64. It is the ONE home for the number on the host: `attn_part`
  // below is strided by `max_len / kAttnBlock` blocks, capture.cc launches
  // `attn_decode` with exactly that many, and it is the `_B64` half of both
  // compiled variant names (`kernels::attn_decode_variant`), so a value here
  // that no compiled binary matches throws at capture instead of striding
  // `attn_part` at one size while the kernel writes it at another.
  //
  // **One home on the host, three in the tree - and the only thing binding them
  // needs a GPU.** The same 64 also lives in `src/kernels/CMakeLists.txt`
  // (`ATTN_BLOCK`, which compiles it into the kernel and into the binary's
  // name) and in `tests/kernels/attn_ref.h` (`attn_ref::kBlock`, which the
  // reference walks blocks with). Nothing checks the three against each other
  // at compile time: the guard is the missing-binary throw described above, and
  // that throw fires at CAPTURE - it needs a device, a loaded model and a
  // built kernel set, so a CPU-only build or a host-side review sees a
  // disagreement not at all. `kNormGroups` below has exactly the same shape
  // (here, CMake's `FOLD_G`/`NORM_G`, and `prep.cl`'s own `#define FOLD_G 20`
  // fallback). A device-free cross-check - a static_assert or a generated
  // header - is recorded as spec-1.6 work and is deliberately NOT added here.
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
  // Spec 10 (plan 10b): decode attention v2's work-groups per kv head, TGT in
  // src/kernels/attn_v2.cl and `_T32` in its variant name. v2 walks a row of L keys in
  // blocks of max(64, roundup64(ceil(L / 32))) positions, so no row has more than 32
  // partials: v2 uses attn_part's first 24 x 32 x M x 258 floats, which fits inside
  // v1's [24][max_len / kAttnBlock][M][258] allocation for every max_len >= 2048 -
  // attn_part keeps v1's size while v1 stays selectable (buffers_test pins both).
  static constexpr uint32_t kAttnV2Blocks = 32;
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

  DecodeScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  l0::Mem resid;          // bf16 [M][5120]
  l0::Mem x;              // bf16 [M][intermediate]  (prep output; largest K; Qwen3.8 17408)
  l0::Mem partials;       // fp32 [max S][M][max N] (Qwen3.8 [8][M][34816]) = 8.91 MB
  l0::Mem ab_out;         // fp32 [M][128]    (a||b GEMV output, S=1)
  l0::Mem norm_sumsq;     // fp32 [kNormGroups][M] (prep_res_fold -> prep_norm_finish)
  l0::Mem gdn_o;          // fp32 [M][48][128] (gdn_step output, pre gated-norm)
  l0::Mem attn_q;         // fp32 [M][24][256] (post norm+rope)
  l0::Mem attn_gate;      // fp32 [M][24][256]
  l0::Mem attn_part;      // fp32 [24][max_len/kAttnBlock][M][258] (m, l, acc[256]) = 50.72 MB @16384
  l0::Mem attn_out;       // bf16 [M][6144]
  l0::Mem logits;         // fp32 [M][248320] = 7.95 MB
  l0::Mem argmax_part;    // fp32+idx pairs, stage-1 output: [M][243][2]

  size_t bytes() const;
};

// The prefill path's per-chunk scratch. Every size is derived from
// `model::Qwen35`, `kC` and `max_len`; the totals are pinned by
// tests/runtime/buffers_test.cc with the arithmetic spelled out there.
//
// **kC = 2048, not plan 6b's 4096 - ruling A13, and every byte was recomputed
// rather than halved.** Basis (spec §3.3, progress ledger): four of P2's six
// production shapes reach their best rate at M = 2048 and every shape but
// out/o_proj is slower at 4096; the counterweight (the dequant scratch is
// written once per chunk, so its share doubles) was weighed and the two nearly
// cancel over a 4096-token prefill, leaving 2048 ahead on time and **halving
// the activation scratch** on a part already resident at ~17-19 GB.
//
// **The `M = 64` attention route is NOT here - ruling A14 retired it.** Plan
// 6b's `attn_q` / `attn_gate` / `attn_part` (408,944,640 B, of which
// `attn_part` alone was 405,798,912) existed only to feed decode's
// `attn_decode`/`attn_reduce` recompiled at M = 64, and the composed path
// (QKᵀ GEMM + our causal softmax + PV GEMM) deletes both kernels from prefill.
// In their place are plan 6d-composed's six fields - `pf_q`, `pf_attn`,
// `pf_s`, `pf_p`, `pf_o`, `pf_rowsum` - sized here **now**, at that plan's own
// figures, so L3 lands without resizing this struct. `pf_kt` is deliberately
// absent: the batched-GEMM probe (`docs/probe-gemm-batched-2026-09-05.md`)
// measured `transB` native and bitwise identical to a packed operand, so the
// transpose fallback is not built.
struct PrefillScratch {
  static constexpr uint32_t kC = 2048;               // ruling A13 (was 4096)
  static constexpr uint32_t kGdnChunk = 64;          // the FLA intra-chunk size
  // Heads that share one QKᵀ / softmax launch on the composed attention path:
  // one GQA group. A tile spanning a group boundary would need two B pointers
  // in one batched launch, which is why it is 6 and not A14's illustrative 4
  // (plan 6d-composed, "The head tile Lh").
  static constexpr uint32_t kSHeads = 6;
  static constexpr uint32_t kNormGroups = DecodeScratch::kNormGroups;

  PrefillScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  l0::Mem ids;          // uint32 [kC], Host  - pf_embed_gather's input
  l0::Mem resid;        // bf16 [kC][5120]
  l0::Mem x;            // bf16 [kC][intermediate]  (Qwen3.8 17408)
  l0::Mem partials;     // fp32 [kC][max int4 N]    (R1: one S = 1 rectangle; Qwen3.8 34816)
  l0::Mem ab_out;       // fp32 [kC][128]
  l0::Mem norm_sumsq;   // fp32 [kNormGroups][kC]
  l0::Mem gdn_o;        // fp32 [kC][48][128]
  l0::Mem mixer_out;    // bf16 [kC][6144]    (out_proj / o_proj input, R2)
  l0::Mem logits;       // fp32 [1][248320]   (last position only)
  l0::Mem argmax_part;  // fp32 [1][243][2]
  // gdn_chunk's own scratch (plan 6b Tasks 6-8)
  l0::Mem gdn_xb;       // bf16 [kC][10240]  conv+SiLU, then l2normed in place
  l0::Mem gdn_seed;     // bf16 [3][10240]   the ring's last 3 positions
  l0::Mem gdn_g;        // fp32 [kC][48]     the intra-chunk cumulative gate
  l0::Mem gdn_beta;     // fp32 [kC][48]
  l0::Mem gdn_A;        // fp32 [kC/kGdnChunk][48][64][64]  A, then T in place
  l0::Mem gdn_A2;       // fp32 [kC/kGdnChunk][48][64][64]
  l0::Mem gdn_w, gdn_u; // bf16 [kC][48][128] each
  // composed attention (plan 6d-composed Task 5's fields, allocated here)
  l0::Mem pf_q;         // bf16 [kC][24][256]  RoPE'd queries (ruling A9)
  l0::Mem pf_attn;      // bf16 [kC][24][256]  attn_chunk's output, pre-gate
  // pf_s / pf_p (the composed path's S and P, [kSHeads][kC][max_len]) are lazy since
  // spec 6: pf_s_buffer() / pf_p_buffer() below.
  l0::Mem pf_o;         // fp32 [24][kC][256]           O = PV, all heads
  l0::Mem pf_rowsum;    // fp32 [24][kC]
  uint32_t max_len;

  size_t bytes() const;
  // What `attn_chunk` divides by to pick its head tile at the actual depth
  // (plan 6d-composed Task 4 Step 3). One accessor, so the rule reads the
  // allocation instead of a second copy of the constant.
  // 0 until the composed path has run (spec 6).
  size_t pf_s_bytes() const { return pf_s_ ? pf_s_->size() : 0; }

  // Spec 2.1 §3.3: the two backend expansions, allocated on first use (ruling R7's pattern one
  // level down) so a session pays only for the backend it runs.
  l0::Mem& dequant_buffer();   // sycl-tla: bf16 [5120][max int4 N] (Qwen3.8 356,515,840 B)
  l0::Mem& slab_buffer();      // L0: bf16 [intermediate][1024] (Qwen3.8 35,651,584 B = 17408*1024*2)
  // Spec 6 (plan 6b): the composed attention's score scratch, allocated on the first
  // composed `attn_chunk` (sycl-tla, or L0 with B70_PREFILL_ATTN=composed). The default
  // flash path never touches them, so prefill scratch no longer scales with max_len.
  l0::Mem& pf_s_buffer();      // fp32 [kSHeads][kC][max_len]  S = QK^T
  l0::Mem& pf_p_buffer();      // bf16 [kSHeads][kC][max_len]  P = softmax(S)
  size_t lazy_bytes() const;   // whichever of the four exist

 private:
  l0::Context* ctx_;
  const model::ModelDesc* desc_;
  std::unique_ptr<l0::Mem> dequant_, slab_, pf_s_, pf_p_;
};

// Spec 8 (plan 8b): the MTP head's state, allocated ONLY when the model carries the
// head (`loader::load(..., mtp = true)`); an Engine without it allocates none of this.
//
//   hctl       the head's own `Control` block (shared): its embed_gather, attention and
//              argmax kernels read pos / n_active / cur_token from here, so the head runs
//              at its own position (t for the pair (h_t, x[t+1])) and its argmax chains
//              the drafts (cur_token <- draft, pos += 1) with the main kernels unchanged.
//   gdn_spec   GDN state slots 1..kSlots-1 (slot 0 is PersistentBuffers::gdn_state), each
//              [48][48][128][128] fp32 like gdn_state. The verify list's row m writes slot
//              (live + m) % kSlots; `Control::gdn_live` names the live one (P0's commit
//              mechanism: an index, not a 151 MB copy).
//   kv_k/kv_v  the head's own KV cache, bf16 [max_len][4][256] each (the 17th KV layer).
//   hh         bf16 [kM + 1][5120]: row 0 = h_{pos-1}, the main model's post-final-norm
//              hidden at the last consumed position; rows 1..M = the verify rows' hidden.
//   dh         bf16 [5120]: the draft chain's hidden (the head's own post-mtp.norm output).
//   logits     fp32 [kMaxK][kVocab]: draft i's logits (q_i for the host sampler).
struct MtpBuffers {
  static constexpr uint32_t kSlots = 4;   // M <= 4: K <= 3 drafts + the pending token
  static constexpr uint32_t kMaxK = kSlots - 1;

  MtpBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  l0::Mem hctl;
  l0::Mem gdn_spec;
  l0::Mem kv_k, kv_v;
  l0::Mem hh;
  l0::Mem dh;
  l0::Mem logits;
  uint32_t max_len;

  size_t bytes() const;
  // What Engine::reset() zeroes when MTP is on (all of it: the head's KV rows are
  // only ever read after being written, but a reset session must not depend on it).
  void zero(l0::CmdList& imm);
};

// The VIEW. Every public name capture.cc uses, with the same types as before
// except that the l0::Mem members are references into the two groups. The
// (ctx, max_len) constructor still OWNS both groups.
struct DecodeBuffers {
 private:
  // Declared first: the reference members below bind to these.
  std::unique_ptr<PersistentBuffers> own_p_;
  std::unique_ptr<DecodeScratch> own_s_;

 public:
  static constexpr uint32_t kM = DecodeScratch::kM;
  static constexpr uint32_t kConvRing = PersistentBuffers::kConvRing;
  static constexpr uint32_t kAttnBlock = DecodeScratch::kAttnBlock;
  static constexpr uint32_t kAttnV2Blocks = DecodeScratch::kAttnV2Blocks;
  static constexpr uint32_t kNormGroups = DecodeScratch::kNormGroups;

  DecodeBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);  // owning
  DecodeBuffers(PersistentBuffers& p, DecodeScratch& s);    // view (Engine's)

  // --- persistent state (survives across tokens) ---
  l0::Mem &control, &gdn_state, &conv_ring, &kv_k, &kv_v;
  // --- per-step scratch (overwritten every token) ---
  l0::Mem &resid, &x, &partials, &ab_out, &norm_sumsq, &gdn_o, &attn_q, &attn_gate,
      &attn_part, &attn_out, &logits, &argmax_part;
  uint32_t max_len;

  size_t persistent_bytes() const;   // printed at startup, asserted by the test
  size_t scratch_bytes() const;
};
}  // namespace runtime
