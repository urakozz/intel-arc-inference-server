#pragma once
// Spec 26c: the card tests' pieces beside bolt_rig.h - the drafter's tap blocks of a descriptor, a TapSink that RECORDS what
// the engine hands a drafter (bolt_verify_test, bolt_commit_test: the taps every projected position carried, the positions
// every commit / prefill projected) and the engine's whole session snapshot (state + KV) for restoring one state many times.
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "model/bolt.h"
#include "runtime/bolt/bolt_engine.h"
#include "runtime/tap_drafter.h"

namespace bolt_spec_rig {

// The drafter's tap blocks: LiLiCorr's 1 / 18 / 35 / 51 / 68 / 85 on the real model (spec 26 §1); on a 12-block synthetic the
// tiny pair's 1 / 2 / 4 / 6 / 9 / 10 (spec 26 §12 ruling 3: the real taps' kinds).
inline std::vector<uint32_t> tap_layers(const model::BoltDesc& d) {
  if (d.layers == 88) return {1, 18, 35, 51, 68, 85};
  if (d.layers == 12) return {1, 2, 4, 6, 9, 10};
  throw std::runtime_error("bolt_spec_rig: no tap layers for a " + std::to_string(d.layers) + "-block Bolt");
}

// A TapSink that keeps, per projected position, the tap rows the engine handed it (bf16 [taps][hidden]) - it drafts nothing.
// projection(pos, n) reads rows 0..n-1 of taps() at once (the engine has waited for what wrote them) and returns a one-command
// list for the engine to replay; positions projected again (a later chunk, a restore) are overwritten.
class Recorder final : public runtime::TapSink {
 public:
  Recorder(l0::Context& ctx, std::vector<uint32_t> layers, uint32_t hidden)
      : ctx_(ctx), layers_(std::move(layers)), hidden_(hidden),
        taps_(ctx, l0::MemKind::Device, size_t(8) * layers_.size() * hidden * 2), dummy_(ctx, l0::MemKind::Device, 64),
        imm_(l0::CmdList::immediate(ctx)) {}

  uint32_t max_k() const override { return 7; }
  const std::vector<uint32_t>& tap_layers() const override { return layers_; }
  l0::Mem& taps() override { return taps_; }
  uint32_t window() const override { return 1024; }
  l0::CmdList& projection(uint32_t pos, uint32_t n) override {
    if (n == 0 || n > 8) throw std::invalid_argument("Recorder::projection: 1..8 rows");
    const size_t row = layers_.size() * size_t(hidden_) * 2;
    std::vector<uint16_t> h(size_t(n) * row / 2);
    imm_.copy(h.data(), taps_.ptr(), size_t(n) * row);
    for (uint32_t r = 0; r < n; ++r) rows[pos + r].assign(h.begin() + long(r * row / 2), h.begin() + long((r + 1) * row / 2));
    ++projections;
    list_ = std::make_unique<l0::CmdList>(l0::CmdList::regular(ctx_));
    list_->fill(dummy_.ptr(), 0u, 64);
    list_->close();
    return *list_;
  }
  void reset(l0::CmdList&) override { rows.clear(); }
  size_t bytes() const override { return taps_.size() + dummy_.size(); }
  size_t ring_bytes() const override { return 0; }
  void save_ring(l0::CmdList&, void*) const override {}
  void load_ring(l0::CmdList&, const void*) override {}
  uint64_t key() const override { return 0x26C5EC0DEull; }

  std::map<uint32_t, std::vector<uint16_t>> rows;   // position -> its taps [taps][hidden] bf16
  uint64_t projections = 0;

  // The taps rows the last verify wrote (rows 0..M-1), read directly.
  std::vector<uint16_t> read_taps(uint32_t M) {
    std::vector<uint16_t> h(size_t(M) * layers_.size() * hidden_);
    imm_.copy(h.data(), taps_.ptr(), h.size() * 2);
    return h;
  }

 private:
  l0::Context& ctx_;
  std::vector<uint32_t> layers_;
  uint32_t hidden_;
  l0::Mem taps_, dummy_;
  l0::CmdList imm_;
  std::unique_ptr<l0::CmdList> list_;
};

// The engine's session at its pos: its state (with a drafter's ring after it) and the KV of [0, pos).
struct Session {
  uint32_t pos = 0;
  std::vector<uint8_t> state, kv;
};
inline Session save(runtime::bolt::BoltEngine& e) {
  Session s;
  s.pos = e.pos();
  s.state.resize(e.state_bytes());
  e.save_state(s.state.data());
  s.kv.resize(e.kv_bytes(s.pos));
  e.save_kv(0, s.pos, s.kv.data());
  return s;
}
// Restores `s` and sets the pending id (a restore leaves cur_token to the caller).
inline void load(runtime::bolt::BoltEngine& e, const Session& s, uint32_t pending) {
  e.load_state(s.state.data(), s.pos);
  e.load_kv(0, s.pos, s.kv.data());
  e.set_token(pending);
}

}  // namespace bolt_spec_rig
