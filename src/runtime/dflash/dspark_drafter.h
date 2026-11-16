#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "runtime/capture.h"   // CapturedStep
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/tap_drafter.h"

// Kolibri DSpark: the DSpark drafter (intuit/Kolibri-1-dspark, model::DFlashForm::DSpark) on Kolibri-1's last card, beside
// the target's lm_head, reading an embedding replica there - its bf16
// weights (loader::load_drafter's target-shape overload: the DFlash v1 body plus the Markov head as pred / succ), 19b's
// buffers (runtime::dflash::DFlashBuffers at sizes(): `hid` holds ones) and its two lists, each captured once:
//
// **The draft list** (draft_launches: 66 for the published drafter): one 8-row block at the anchor's position p =
// ctrl.pos, the anchor ctrl.cur_token[0], the mask row below it - ALWAYS the whole block and the 7-row walk (dctl.k 7: the
// attention is causal, so row r reads nothing past itself, and a draft of K < 7 is the walk's first K):
//   df_embed; per layer df_norm, q||k||v, df_qkv_prep (RoPE at p + r), df_attn_part + df_attn_reduce (the CAUSAL build:
//   the ring's 2047 positions before row r's, the block's rows 0..r), o, df_add, df_norm, gate||up, df_silu_mul, down,
//   df_add; the final df_norm; the TARGET's head over rows 0..6 (anchor-first: row r predicts p + r + 1; gemv_i8w /
//   gemv_bf16 at M 7); df_topk_stage1 / 2 (the 16 best of the 127998 token rows a row); df_select's V2 walk with
//   pred = markov_w1, succ = markov_w2 and the hidden rows of ones - S[c] = logit[c] + markov_w2[c] . markov_w1[prev], prev
//   the anchor then the row before's pick (greedy at T 0, else Gumbel at T) - drafts into ctrl.cur_token[1..7] and the
//   readback (DFlashDraftOut: ids, candidates, their head logits, the walk's realized scores).
// **The projection list** (19b's, project_launches: 8): dctl.proj_n <= 8 tap rows (taps(): bf16 [8][5 x 2560]) at positions
// dctl.proj_pos .. into the ring - fc, df_norm (hidden_norm), per layer the k||v rows of its q||k||v weight, df_ctx_store
// (k_norm, RoPE, ring slot pos % 2048).
//
// The target side is runtime::TapSink (KolibriEngine's verify lists copy the five taps of every row into taps(), its commit
// replays projection(pos, j + 1), its prefill projects the prompt's last window(), its snapshots carry the ring). Order:
//   1. DSparkDrafter d(ctx, loader::load_drafter(ctx, path, shape, replica, Bf16), max_len, max_k)
//   2. engine.attach_drafter(d); d.attach(ctx, replica, engine.lm_head(), engine.control_mem(last))
//   per iteration: d.draft(replay, k, T, seed) -> engine.verify(drafts, k) -> acceptance -> engine.commit(j, t).
namespace runtime::dflash {

// `embed`: the target's embedding replica, bf16 ROW-MAJOR [vocab][hidden]; `head`: its lm_head, int8 or bf16 tiles.
CapturedStep build_dspark_draft(l0::Context& ctx, const loader::DFlashWeights& w, const l0::Mem& embed,
                                const loader::DeviceWeight& head, DFlashBuffers& b, const l0::Mem& control);
CapturedStep build_dspark_project(l0::Context& ctx, const loader::DFlashWeights& w, DFlashBuffers& b);

class DSparkDrafter final : public TapSink {
 public:
  // `max_len`: the engine's (the RoPE rows); `max_k`: the deepest block the engine verifies, 1..7.
  DSparkDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t max_len, uint32_t max_k = 7);

  // The draft list over the target's embedding replica, its lm_head and the Control block whose pos / cur_token[0] are the
  // anchor (cur_token[1..7] get the drafts). Once.
  void attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head, const l0::Mem& control);

  // One block at the Control block's pos for a caller that keeps k drafts (1..max_k()): `replay` runs the draft list on the
  // drafter's queue and waits. The list always walks all 7 rows at `temperature` / `seed`; the readback's first k ids are
  // the k drafts, and row j's sampled q is softmax(scores[j] / T) over cand[j] (server::dflash_q).
  const DFlashDraftOut& draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k, float temperature = 0.0f,
                              uint64_t seed = 0);
  const DFlashDraftOut& readback() const { return *b_.readback(); }

  const model::DFlashDesc& desc() const { return w_.desc; }

  // --- TapSink ---
  uint32_t max_k() const override { return max_k_; }
  const std::vector<uint32_t>& tap_layers() const override { return w_.desc.taps; }
  l0::Mem& taps() override { return b_.taps; }
  uint32_t window() const override { return w_.desc.window.at(0); }   // row 0 reads p - 2047 .. p - 1
  l0::CmdList& projection(uint32_t pos, uint32_t n) override;
  void reset(l0::CmdList& imm) override;
  size_t bytes() const override { return w_.bytes + b_.bytes(); }
  size_t ring_bytes() const override { return b_.ring_k.size() + b_.ring_v.size(); }   // ring_k, then ring_v
  void save_ring(l0::CmdList& imm, void* host) const override;
  void load_ring(l0::CmdList& imm, const void* host) override;
  uint64_t key() const override { return key_; }

 private:
  loader::DFlashWeights w_;
  DFlashBuffers b_;
  uint32_t max_k_;
  uint64_t key_ = 0;
  CapturedStep project_;
  std::unique_ptr<CapturedStep> draft_;
};

}  // namespace runtime::dflash
