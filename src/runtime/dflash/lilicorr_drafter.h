#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/lilicorr_loader.h"
#include "runtime/capture.h"   // CapturedStep
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/lilicorr_sizes.h"
#include "runtime/tap_drafter.h"

// Spec 26b Task 5: the LiLiCorr drafter (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, model::DFlashForm::LiLiCorr)
// on Bolt's last card (spec 26 Decision 8: device 1 holds the target's lm_head; the drafter reads an embedding replica
// there) - its weights (loader::load_lilicorr: exact / int8 / int4 g64, the head always exact), its buffers
// (runtime/dflash/lilicorr_sizes.h, allocated exactly as the planner counts them) and its two lists, each captured once:
//
// **The draft list** (spec 26 §5.3): one 8-row block at the anchor's position p = ctrl.pos, the anchor ctrl.cur_token[0] -
// ALWAYS the whole block and all 7 slots (dctl.k = 7: the backbone is non-causal over the block and the head scores every
// slot, so a draft of K < 7 is the walk's first K - spec 26 §12 ruling 1, 23f's finding) - in order (lili_draft_launches:
// 116): df_embed; per layer df_norm, the attention conv's kernel projection, df_conv_prep, q||k||v, df_qkv_prep (RoPE at
// p + r), ll_attn_part + ll_attn_reduce (the 1023 window over the 1024-slot ring, the block non-causal, the sinks), o,
// df_conv_add, df_norm, the MLP conv's projection, df_conv_prep, gate||up, df_silu_mul, down, df_conv_add; the final
// df_norm; the TARGET's head over rows 1..7 (gemv_i8w / gemv_bf16 at M 7); ll_topk_stage1/2 (the 8 best raw logits, the
// full-vocabulary log-probs); the LiLiCorr head (ll_lattice.cl - over the candidates' embedding rows, the mask rows'
// final-normed hidden, the log-probs and the ANCHOR row); ll_select (the walk, the readback, ctrl.cur_token[1..7]).
// The exact arm's NVFP4 linears run ll_gemv_bf16_g (their g on the output); every other linear 19b's GEMVs at M 8.
//
// **The projection list** (19b's, + the anchor): dctl.proj_n <= 8 tap rows (taps(): bf16 [8][6 x 4096]) at positions
// dctl.proj_pos .. into the ring - fc, df_norm (hidden_norm), ll_anchor (the LAST projected row's fp32 normed vector into
// the anchor buffer: the next block's anchor is the newest committed row, spec 26 §2.3 - Review Focus 5), per layer the
// k||v rows of its q||k||v weight, df_ctx_store (k_norm, RoPE, ring slot pos % 1024). 9 launches. The block's own K / V
// never reach the ring.
//
// The target side is runtime::TapSink (26c's Bolt verify lists copy the six taps of every row into taps(), its commit
// replays projection(pos, j + 1), its prefill projects the prompt's last window(), its snapshots carry the ring AND the
// anchor - spec 26 §6). The proposer side is this class's own (as MuseDrafter's): LiliDraftOut, not Qwen3.8's 16-candidate
// DFlashDraftOut:
//   1. LiliDrafter d(ctx, std::move(weights), max_len)                 buffers at the engine's max_len; the projection
//   2. d.attach(ctx, embed_replica, head, control)                      the draft list over the target's rows / head / Control
//   per iteration: d.draft(replay, k, T, seed) -> [the verify of cur_token[1..k]] -> acceptance (q = softmax(scores / T) on
//   each slot's 8 candidates, Decision 6) -> commit: projection(pos, j + 1) replayed.
namespace runtime::dflash {

struct LiliBuffers {
  LiliBuffers(l0::Context& ctx, const model::DFlashDesc& d, uint32_t target_hidden, uint32_t vocab, uint32_t max_len);
  model::DFlashDesc desc;
  uint32_t target_hidden, vocab, max_len;
  LiliSizes sz;
  l0::Mem dctl, out;   // shared
  l0::Mem rope, ring_k, ring_v, anchor, taps, resid, h, x, xa, gout, co, q, kb, vb, part, kv, logits, tpart, cand, logit,
      lp, head;
  LiliCtl* ctl() const { return dctl.as<LiliCtl>(); }
  const LiliDraftOut* readback() const { return out.as<LiliDraftOut>(); }
  size_t bytes() const;   // == sz.total()
  // Zero the ring, the anchor and every scratch, reset the control block and the readback, upload the RoPE table: what
  // construction leaves and what a session reset restores (a cold drafter: anchor_valid 0).
  void zero(l0::CmdList& imm);
};

struct LiliTarget {
  const l0::Mem* embed = nullptr;               // bf16 ROW-MAJOR [vocab][hidden] (the replica)
  const loader::DeviceWeight* head = nullptr;   // the lm_head, int8 or bf16 tiles, [vocab][hidden]
};

CapturedStep build_lili_draft(l0::Context& ctx, const loader::LiliWeights& w, const LiliTarget& t, LiliBuffers& b,
                              const l0::Mem& control);
CapturedStep build_lili_project(l0::Context& ctx, const loader::LiliWeights& w, LiliBuffers& b);

class LiliDrafter final : public TapSink {
 public:
  // `max_len`: the engine's (the RoPE rows); `max_k`: the deepest block the engine verifies, 1..7 (`--spec-max`).
  LiliDrafter(l0::Context& ctx, loader::LiliWeights w, uint32_t max_len, uint32_t max_k = 7);

  // The draft list over the target's embedding replica, its lm_head and the Control block whose pos / cur_token[0] are the
  // anchor (cur_token[1..7] get the drafts). Once.
  void attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head, const l0::Mem& control);
  bool attached() const { return draft_ != nullptr; }

  // One block at the Control block's pos for a caller that keeps k drafts (1..max_k()): `replay` runs the draft list on
  // the drafter's queue and waits. The list always runs the whole block and walks all 7 slots at `temperature` / `seed`
  // (dctl.k = draft_list_k = 7): the readback's first k ids are the k drafts, as a list cut at k would give them.
  const LiliDraftOut& draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k, float temperature = 0.0f,
                            uint64_t seed = 0);
  const LiliDraftOut& readback() const { return *b_.readback(); }

  const model::DFlashDesc& desc() const { return w_.desc; }
  const loader::LiliWeights& weights() const { return w_; }
  LiliBuffers& buffers() { return b_; }
  const CapturedStep& project_step() const { return project_; }
  const CapturedStep* draft_step() const { return draft_.get(); }

  // --- TapSink ---
  uint32_t max_k() const override { return max_k_; }
  const std::vector<uint32_t>& tap_layers() const override { return w_.desc.taps; }
  l0::Mem& taps() override { return b_.taps; }
  uint32_t window() const override { return w_.desc.window.at(0); }   // row 0 reads p - 1023 .. p - 1; the anchor p - 1
  l0::CmdList& projection(uint32_t pos, uint32_t n) override;
  void reset(l0::CmdList& imm) override { b_.zero(imm); }
  size_t bytes() const override { return w_.bytes + b_.bytes(); }
  size_t ring_bytes() const override { return b_.ring_k.size() + b_.ring_v.size() + b_.anchor.size(); }
  // ring_k, ring_v, then the anchor (spec 26 §6); a load sets anchor_valid (a snapshot follows a projection)
  void save_ring(l0::CmdList& imm, void* host) const override;
  void load_ring(l0::CmdList& imm, const void* host) override;
  uint64_t key() const override { return key_; }

 private:
  loader::LiliWeights w_;
  LiliBuffers b_;
  uint32_t max_k_;
  uint64_t key_ = 0;
  CapturedStep project_;
  std::unique_ptr<CapturedStep> draft_;
};

}  // namespace runtime::dflash
