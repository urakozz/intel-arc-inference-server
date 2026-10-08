#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "model/model_desc.h"
#include "model/qwen35.h"
#include "runtime/buffer_sizes.h"

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
//
// **The sizes come from runtime/buffer_sizes.h** (spec 6 §10, max_len auto): each
// struct inherits its constants and a static `sizes(max_len, desc)` from a `*Dims`
// base there, and its constructor allocates exactly what `sizes()` returns - the one
// formula the memory planner (runtime/memory_plan.h) adds up before allocating.

// Everything that survives a token boundary. Layouts are decode's, unchanged.
// kConvRing = 16 (ring depth >= M + 3, spec §9.4): PersistentDims.
struct PersistentBuffers : PersistentDims {
  // Spec 14: per-layer state is sized from the model descriptor (gdn_layers,
  // fa_layers); there is no default - a caller has to say which model.
  // Spec 12b: `kv` is the KV cache's form (`--kv-cache`); kv_k / kv_v are laid out by
  // `kv_lay` (runtime/buffer_sizes.h KvLayout) - at bf16 exactly today's allocation.
  PersistentBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                    KvCache kv = default_kv_cache());
  // Spec 16b (pipeline parallel): one stage's state - `gdn_layers` GDN and `fa_layers` FA
  // slices (PersistentDims::stage_sizes), in the same layout and allocation order.
  PersistentBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc, KvCache kv,
                    uint32_t gdn_layers, uint32_t fa_layers);

  l0::Mem control;        // shared, sizeof(Control)
  // Spec 15b: the widths are the descriptor's; the brackets give Qwen3.8's.
  l0::Mem gdn_state;      // fp32 [gdn_layers][v-heads 48][128 k][128 v] = 150.99 MB (Qwen3.8: 48)
  l0::Mem conv_ring;      // bf16 [gdn_layers][16][conv dim 10240]       = 15.73 MB (Qwen3.8)
  l0::Mem kv_k, kv_v;     // bf16 [fa_layers][max_len][kv-heads 4][256] each = 536.87 MB each @16384 (Qwen3.8: 16)
                          // int8 (spec 12b): [fa_layers][max_len][4][256] int8 + [fa_layers][max_len][4] fp16
  uint32_t max_len;
  KvLayout kv_lay;        // the form and the layout of kv_k / kv_v (fa_layers layers)

  size_t bytes() const;
  // The five fills Engine::reset() does, in the order the constructor does
  // them. It is one place, so "reset writes exactly what construction wrote"
  // is a property of the code rather than of two lists agreeing.
  void zero(l0::CmdList& imm);

 private:
  // The public constructor delegates here with sizes(max_len, desc, kv).
  PersistentBuffers(l0::Context& ctx, uint32_t max_len, const PersistentSizes& s,
                    const KvLayout& kv);
};

// Decode's per-step scratch. Field names, sizes and allocation order unchanged.
struct DecodeScratch : DecodeScratchDims {
  // kM, kAttnBlock, kAttnV2Blocks and kNormGroups - each with the measurement that
  // chose it - are DecodeScratchDims's (runtime/buffer_sizes.h).
  DecodeScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  // Spec 15b: hidden, head counts and widths are the descriptor's (Qwen3.8 in brackets).
  l0::Mem resid;          // bf16 [M][hidden 5120]
  l0::Mem x;              // bf16 [M][max(intermediate, 2 x hidden)]  (prep output; Qwen3.8 17408)
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
  // Spec 15c: a MoE model's router logits, route rows and SiLU x up rows - one
  // allocation at runtime::moe_scratch_layout's offsets; null on a dense model.
  std::unique_ptr<l0::Mem> moe;

  size_t bytes() const;

 private:
  DecodeScratch(l0::Context& ctx, const DecodeScratchSizes& s);   // the delegate
};

// The prefill path's per-chunk scratch. Every size is derived from
// `model::Qwen35`, the model descriptor (spec 15b), `kC` and `max_len`; the totals are pinned by
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
struct PrefillScratch : PrefillScratchDims {
  // kC = 2048 (ruling A13), kGdnChunk = 64 and kNormGroups: PrefillScratchDims.

  PrefillScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc);

  l0::Mem ids;          // uint32 [kC], Host  - pf_embed_gather's input
  l0::Mem resid;        // bf16 [kC][5120]
  l0::Mem x;            // bf16 [kC][max(intermediate, 2 x hidden)]  (Qwen3.8 17408)
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
  // pf_s / pf_p (the composed path's S and P, [s_heads()][kC][max_len]) are lazy since
  // spec 6: pf_s_buffer() / pf_p_buffer() below.
  l0::Mem pf_o;         // fp32 [24][kC][256]           O = PV, all heads
  l0::Mem pf_rowsum;    // fp32 [24][kC]
  // Spec 15d: a MoE model's routing, sorted rows, expert activations and the per-chunk
  // expert weight batch - one allocation at runtime::moe_prefill_layout's offsets; null
  // on a dense model (runtime/prefill/moe.cc).
  std::unique_ptr<l0::Mem> moe;
  uint32_t max_len;

  size_t bytes() const;
  // The model this scratch was sized for (spec 15b: its widths are the descriptor's).
  const model::ModelDesc& desc() const { return *desc_; }
  // Heads that share one QKᵀ / softmax launch on the composed attention path:
  // one GQA group (6 on Qwen3.8, 8 on Ornith: ModelDesc::fa_gqa). A tile spanning
  // a group boundary would need two B pointers in one batched launch, which is
  // why it is the group and not A14's illustrative 4 (plan 6d-composed, "The
  // head tile Lh").
  uint32_t s_heads() const { return desc_->fa_gqa(); }
  // What `attn_chunk` divides by to pick its head tile at the actual depth
  // (plan 6d-composed Task 4 Step 3). One accessor, so the rule reads the
  // allocation instead of a second copy of the constant.
  // 0 until the composed path has run (spec 6).
  size_t pf_s_bytes() const { return pf_s_ ? pf_s_->size() : 0; }

  // Spec 2.1 §3.3: the two backend expansions, allocated on first use (ruling R7's pattern one
  // level down) so a session pays only for the backend it runs.
  l0::Mem& dequant_buffer();   // sycl-tla: bf16 [5120][max int4 N] (Qwen3.8 356,515,840 B)
  l0::Mem& slab_buffer();      // L0: bf16 [widest int4 K][1024] (Qwen3.8 35,651,584 B = 17408*1024*2;
                               // spec 15d: Ornith's 4096, out_proj / o_proj)
  // Spec 6 (plan 6b): the composed attention's score scratch, allocated on the first
  // composed `attn_chunk` (sycl-tla, or L0 with B70_PREFILL_ATTN=composed). The default
  // flash path never touches them, so prefill scratch no longer scales with max_len.
  l0::Mem& pf_s_buffer();      // fp32 [s_heads()][kC][max_len]  S = QK^T
  l0::Mem& pf_p_buffer();      // bf16 [s_heads()][kC][max_len]  P = softmax(S)
  size_t lazy_bytes() const;   // whichever of the four exist

 private:
  PrefillScratch(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                 const PrefillScratchSizes& s);   // the delegate
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
//   dv_logits  spec 8 §11, a draft vocabulary only: fp32 [|V'|], the compact head's
//              logits (the draft argmax's input; draft_vocab.cl scatters them into
//              `logits` row i at V''s ids).
struct MtpBuffers : MtpDims {
  // kSlots = 4 (M <= 4: K <= 3 drafts + the pending token), kMaxK = 3: MtpDims.
  // -inf as the fp32 bit pattern zero() fills `logits` with under a draft vocabulary.
  static constexpr uint32_t kNegInfBits = 0xFF800000u;

  // `draft_vocab` = |V'| (spec 8 §11), 0 = the full head drafts (no dv_logits, and
  // `logits` is zeroed as before). Every size is MtpDims::sizes(max_len, desc, draft_vocab).
  // `kv` (spec 12b): the head's KV is in the main cache's form - one layer of KvLayout.
  // `slots` (spec 19a, plan 19a Task 4): the GDN state slots, kSlots (4) for every engine
  // and pipeline stage; tools/probe/probe_mtp_steps alone builds a second set at 8 for its
  // verify lists at M = 5..8 (build_verify binds `gdn_step_slots_M<M>_N8` over them).
  MtpBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
             uint32_t draft_vocab = 0, KvCache kv = default_kv_cache(), uint32_t slots = kSlots);

  l0::Mem hctl;
  l0::Mem gdn_spec;
  l0::Mem kv_k, kv_v;
  l0::Mem hh;
  l0::Mem dh;
  l0::Mem logits;
  std::unique_ptr<l0::Mem> dv_logits;   // null unless draft_vocab > 0
  uint32_t max_len;
  uint32_t draft_vocab = 0;
  KvLayout kv_lay;   // the head's kv_k / kv_v: one layer, the engine's form
  // GDN state slots: slot 0 is gdn_state, gdn_spec holds slots - 1 more. A verify list
  // at M rows needs M <= slots; row m writes slot (live + m) % slots.
  uint32_t slots = kSlots;

  size_t bytes() const;
  // What Engine::reset() zeroes when MTP is on (all of it: the head's KV rows are
  // only ever read after being written, but a reset session must not depend on it).
  // **Under a draft vocabulary `logits` is filled with -inf instead** (spec 8 §11): the
  // draft list writes only V''s entries of its row, so every other entry must read as
  // q = 0 to the host's acceptance - -inf, which filter_probs turns into probability 0 -
  // and nothing else ever writes them, so once per construction and reset is enough.
  void zero(l0::CmdList& imm);

 private:
  MtpBuffers(l0::Context& ctx, uint32_t max_len, const MtpSizes& s,
             uint32_t draft_vocab, const KvLayout& kv, uint32_t slots);   // the delegate
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

  DecodeBuffers(l0::Context& ctx, uint32_t max_len, const model::ModelDesc& desc,
                KvCache kv = default_kv_cache());                         // owning
  DecodeBuffers(PersistentBuffers& p, DecodeScratch& s);    // view (Engine's)

  // --- persistent state (survives across tokens) ---
  l0::Mem &control, &gdn_state, &conv_ring, &kv_k, &kv_v;
  // --- per-step scratch (overwritten every token) ---
  l0::Mem &resid, &x, &partials, &ab_out, &norm_sumsq, &gdn_o, &attn_q, &attn_gate,
      &attn_part, &attn_out, &logits, &argmax_part;
  l0::Mem* moe;   // spec 15c: DecodeScratch::moe, null on a dense model
  uint32_t max_len;
  KvLayout kv_lay;   // spec 12b: kv_k / kv_v's form and layout (PersistentBuffers::kv_lay)

  size_t persistent_bytes() const;   // printed at startup, asserted by the test
  size_t scratch_bytes() const;
};
}  // namespace runtime
