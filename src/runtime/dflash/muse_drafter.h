#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "runtime/capture.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/tap_drafter.h"

// Spec 23f: Muse-Glimmer-30B's DFlash drafter (meta-models/Muse-Glimmer-30B-assistant, model::DFlashForm::Muse) beside
// runtime::muse::MuseEngine - spec 19's pieces at Muse's shapes: 19b's weights (loader::load_drafter by the target's
// shape, any of bf16 / int8 / int4 g64), DFlashBuffers at 16 rows, the two lists of dflash_capture.h (the draft list:
// one 16-row block, mg_df_select writing the 16-row readback and the verify's ids; the projection list: up to 16
// committed rows into the 2112-slot ring). The target side is runtime::TapSink (tap_drafter.h): MuseEngine's verify
// lists copy the taps of every row into taps(), its commit replays projection(pos, j + 1), its prefill projects the
// prompt's last window(), its reset zeroes the ring, its snapshots carry the ring (spec 19d decision 5 as ruled).
//
// The proposer side is this class's own (not TapDrafter's 8-row Qwen3.8 calls):
//   1. MuseDrafter d(ctx, std::move(weights), max_len, max_k, q)         buffers at the engine's max_len; projection
//   2. engine.attach_drafter(d)                                          the engine's verify lists with the taps
//   3. d.attach(ctx, embed, head, control)                               the draft list over the target's embedding,
//                                                                        head and Control (pos, cur_token[0])
// then per iteration: d.draft(replay, k) -> [the host's sampled drafts written into vids() and the readback when
// the request samples] -> engine.verify(k) -> acceptance -> engine.commit(j, t).
//
// **The draft distribution** (spec 23 decision 11, P0: raw): q_j = softmax(scores_j / T) over draft row j's 16
// candidates, scores = the candidates' head logits as df_topk left them - `MuseQ::Raw` the raw rows (transformers'
// form), `MuseQ::Capped` the target's 20 tanh(z x 0.196... / 20). The device picks greedily; a sampled request's
// drafts are drawn by the caller from exactly that q (server::dflash_q) and written back with set_draft().
namespace runtime::dflash {

enum class MuseQ { Raw, Capped };
const char* muse_q_name(MuseQ q);
bool parse_muse_q(const std::string& s, MuseQ& out);   // "raw" | "capped"

class MuseDrafter final : public TapSink {
 public:
  // `max_len`: the engine's (the RoPE rows); `max_k`: the deepest block the engine verifies, 1..15 (`--spec-max`).
  MuseDrafter(l0::Context& ctx, loader::DFlashWeights w, uint32_t max_len, uint32_t max_k, MuseQ q = MuseQ::Raw);

  // The draft list over the target's embedding (bf16 [vocab][6656] on this device), its lm_head (int8 or bf16 at
  // 202048 rows) and the Control block whose pos / cur_token[0] are the anchor. Once.
  void attach(l0::Context& ctx, const l0::Mem& embed, const loader::DeviceWeight& head, const l0::Mem& control);
  bool attached() const { return draft_ != nullptr; }

  // One block of k drafts (1..max_k()) at the Control block's pos: `replay` runs the draft list on the target's queue
  // and waits. Greedy (the device's pick). Returns the readback; vids() holds [anchor, d_1 .. d_k].
  const MuseDraftOut& draft(const std::function<void(l0::CmdList&)>& replay, uint32_t k);
  // A sampled request's draft j (< the last draft's k) replaced by `id`, one of row j's candidates (the caller drew it
  // from q_j): the readback's ids / chosen and vids[1 + j]. Throws when id is not a candidate of row j.
  void set_draft(uint32_t j, uint32_t id);
  const MuseDraftOut& readback() const { return *b_.muse_readback(); }
  l0::Mem& vids() { return *b_.vids; }
  MuseQ q() const { return q_; }

  const model::DFlashDesc& desc() const { return w_.desc; }
  const loader::DFlashWeights& weights() const { return w_; }
  DFlashBuffers& buffers() { return b_; }
  const CapturedStep& project_step() const { return project_; }
  const CapturedStep* draft_step() const { return draft_.get(); }

  // --- TapSink ---
  uint32_t max_k() const override { return max_k_; }
  const std::vector<uint32_t>& tap_layers() const override { return w_.desc.taps; }
  l0::Mem& taps() override { return b_.taps; }
  uint32_t window() const override { return w_.desc.window.at(0); }   // row 0 reads p - 2048 .. p - 1
  l0::CmdList& projection(uint32_t pos, uint32_t n) override;
  void reset(l0::CmdList& imm) override { b_.zero(imm); }
  size_t bytes() const override { return w_.bytes + b_.bytes(); }
  size_t ring_bytes() const override { return b_.ring_k.size() + b_.ring_v.size(); }
  void save_ring(l0::CmdList& imm, void* host) const override;
  void load_ring(l0::CmdList& imm, const void* host) override;
  uint64_t key() const override { return key_; }

  static constexpr float kCapMult = 0.19611613513818404f, kCapSoftcap = 20.0f;   // the target's (spec 23 §1)

 private:
  loader::DFlashWeights w_;
  DFlashBuffers b_;
  uint32_t max_k_;
  MuseQ q_;
  uint64_t key_ = 0;
  CapturedStep project_;
  std::unique_ptr<CapturedStep> draft_;
};

}  // namespace runtime::dflash
