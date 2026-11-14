#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "runtime/capture.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/tap_drafter.h"
#include "runtime/tp_engine.h"

// Spec 19 v3c: the DFlash drafter split over the two cards of `--parallel tp` (spec 19 v3 §6.2), behind
// runtime::TpTapDrafter - runtime::TpEngine's side of it is there. Each card holds its half (loader::load_drafter's
// tp_rank: its 16 q / 4 kv heads of every layer's attention and their ring, its half of every MLP, its vocabulary half
// of the candidate head; fc, the norms, the convolutions and the selector whole) and three lists over it:
//   draft   the block over its half (build_dflash_draft_tp: ten all-reduces with the other card, through the target's
//           crossing), its top 16 candidates a row with their global ids and exact scores (shared memory);
//   tail    the selector over the merged 16 (build_dflash_tail) - drafts into the card's own Control;
//   project the committed rows' taps into its half of the ring (build_dflash_project at the card's shapes).
// merge(): per draft row the two halves' 16 into the best 16 by exact score, ties to the lower id - the one-card
// re-rank's order over the union of both halves' candidates (which holds every candidate one card's top 80 holds).
//
// Lifetime: load both halves (loader::load_drafter(ctx_r, path, card r's model, format, ..., tp_rank r)) ->
// TpDFlashDrafter d(ctxs, halves, max_len, max_k) -> runtime::TpEngine e(..., &d) -> d.attach(e). It must outlive e.
namespace runtime::dflash {

class TpDFlashDrafter final : public TpTapDrafter {
 public:
  TpDFlashDrafter(std::array<l0::Context*, 2> ctx, std::array<loader::DFlashWeights, 2> halves, uint32_t max_len,
                  uint32_t max_k, const CaptureOptions& opt = {});
  // Captures each card's draft and tail lists against the engine's card (its target, Control and crossing). Once.
  void attach(TpEngine& e);

  TapSink& card(uint32_t r) override;
  std::array<l0::CmdList*, 2> draft_lists(uint32_t k, float temperature, uint64_t seed) override;
  std::array<l0::CmdList*, 2> merge() override;
  const DFlashDraftOut& readback() const override { return *half_[0]->b.readback(); }
  bool samples() const override { return true; }   // DFlash 2 (the TP lists are built for it)
  size_t ring_bytes() const override;
  void save_ring(void* host) const override;
  void load_ring(const void* host) override;
  uint64_t key() const override { return key_; }

  // Tests and probes.
  const loader::DFlashWeights& weights(uint32_t r) const { return half_.at(r)->w; }
  DFlashBuffers& buffers(uint32_t r) { return half_.at(r)->b; }
  const CapturedStep& draft_step(uint32_t r) const { return *half_.at(r)->draft; }
  const CapturedStep& tail_step(uint32_t r) const { return *half_.at(r)->tail; }
  const CapturedStep& project_step(uint32_t r) const { return half_.at(r)->project; }

 private:
  struct Half final : TapSink {
    Half(l0::Context& c, loader::DFlashWeights w, uint32_t max_len, uint32_t max_k, const CaptureOptions& opt);
    uint32_t max_k() const override { return k_max; }
    const std::vector<uint32_t>& tap_layers() const override { return w.desc.taps; }
    l0::Mem& taps() override { return b.taps; }
    uint32_t window() const override;
    l0::CmdList& projection(uint32_t pos, uint32_t n) override;
    void reset(l0::CmdList& imm) override { b.zero(imm); }
    size_t bytes() const override { return w.bytes + b.bytes(); }
    size_t ring_bytes() const override { return b.ring_k.size() + b.ring_v.size(); }
    void save_ring(l0::CmdList& imm, void* host) const override;
    void load_ring(l0::CmdList& imm, const void* host) override;
    uint64_t key() const override { return 0; }   // the pair's (TpDFlashDrafter::key)

    l0::Context& ctx;
    loader::DFlashWeights w;
    DFlashBuffers b;
    uint32_t k_max;
    CapturedStep project;
    std::unique_ptr<CapturedStep> draft, tail;
    mutable l0::CmdList imm;
  };
  std::array<std::unique_ptr<Half>, 2> half_;
  CaptureOptions opt_;
  uint64_t key_ = 0;
};

}  // namespace runtime::dflash
