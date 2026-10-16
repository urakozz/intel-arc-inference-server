#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "runtime/capture.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/engine.h"
#include "runtime/tap_drafter.h"

// Spec 19c: the DFlash drafter in the decode loop (spec 19 §5 "Per iteration") - 19b's weights, buffers and two
// captured lists behind runtime::TapDrafter, so runtime::Engine verifies its drafts, writes its taps and fills its
// ring (runtime/tap_drafter.h has the Engine's side).
//
// Lifetime and order (the Engine borrows the drafter; the drafter's draft list binds the Engine's Control):
//   1. loader::load_drafter(ctx, path, target model, format)            the weights, beside the loaded target
//   2. DFlashDrafter d(ctx, std::move(weights), head rows, max_len)     buffers at the engine's max_len, the
//                                                                       projection list
//   3. runtime::Engine e(ctx, std::move(target), max_len, false, kv, &d)  verify lists with the taps; reset()
//   4. d.attach(e, draft_vocab)                                         the draft list over e's model and Control
// then per iteration: d.draft(e, k, T, seed) -> e.verify(k) -> host acceptance -> e.commit(j, next) (which replays
// the projection of rows 0..j). The drafter must outlive the Engine.
namespace runtime::dflash {

class DFlashDrafter final : public TapDrafter {
 public:
  // `head_rows`: the rows of the target head the draft list reads - the full lm_head's (vocab_size) or a draft
  // vocabulary's |V'| (LoadedModel::draft_vocab). `max_len`: the engine's (the RoPE rows). `max_k`: the deepest
  // block the engine verifies, 1 .. desc.max_k() (7 for DFlash 2's block of 8; `--spec-max`).
  DFlashDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t head_rows, uint32_t max_len, uint32_t max_k);

  // Captures the draft list against the engine's target model (embedding, head: the full one, or with
  // `draft_vocab` the compact one LoadedModel::draft_vocab holds) and its Control block. `e` must have been built
  // with this drafter. Once.
  void attach(Engine& e, bool draft_vocab);
  bool attached() const { return engine_ != nullptr; }

  // One block at the engine's pos: the anchor cur_token[0], K = k drafts into cur_token[1..k] (what e.verify(k)
  // reads), the walk greedy at temperature 0 or Gumbel at `temperature` with `seed` (set_draft_params' rules: a v1
  // drafter drafts greedily only). k must be 1 .. min(max_k(), e.max_verify_k()). Returns the readback: per draft
  // row its id, its 16 candidates (real ids), their head logits and the selector's realized scores - the draft
  // distribution q of row j is softmax(scores[j] / temperature) over cand[j] (server::dflash_q).
  const DFlashDraftOut& draft(Engine& e, uint32_t k, float temperature, uint64_t seed) override;
  bool samples() const override { return w_.desc.v2(); }

  const model::DFlashDesc& desc() const { return w_.desc; }
  const loader::DFlashWeights& weights() const { return w_; }
  DFlashBuffers& buffers() { return b_; }
  const CapturedStep& project_step() const { return project_; }
  const CapturedStep* draft_step() const { return draft_.get(); }
  // The projection's last replay arguments (tests): its first position and its row count.
  uint32_t projected_pos() const { return b_.ctl()->proj_pos; }
  uint32_t projected_n() const { return b_.ctl()->proj_n; }

  // --- TapDrafter ---
  uint32_t max_k() const override { return max_k_; }
  const std::vector<uint32_t>& tap_layers() const override { return w_.desc.taps; }
  l0::Mem& taps() override { return b_.taps; }
  uint32_t window() const override;
  l0::CmdList& projection(uint32_t pos, uint32_t n) override;
  void reset(l0::CmdList& imm) override { b_.zero(imm); }
  size_t bytes() const override { return w_.bytes + b_.bytes(); }

 private:
  loader::DFlashWeights w_;
  DFlashBuffers b_;
  uint32_t max_k_;
  CapturedStep project_;
  std::unique_ptr<CapturedStep> draft_;
  Engine* engine_ = nullptr;
};

}  // namespace runtime::dflash
