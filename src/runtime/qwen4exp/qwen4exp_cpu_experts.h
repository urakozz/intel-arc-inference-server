#pragma once
#include <ze_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "cpu/expert_int4.h"
#include "cpu/expert_q4.h"   // Q4Job
#include "l0/context.h"
#include "l0/memory.h"
#include "l0/sync_event.h"
#include "loader/qwen4exp_loader.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"   // kVerifyRows

// PERF-124 step 4: Qwen3.8-Flash-Next's host-RAM experts (the tier's pinned mirrors) computed on the CPU instead of read
// by the card over PCIe, for the decode step and the verify (M = 1..kVerifyRows rows), one card, `pp` or `tp`.
//
// **Under tp** (the model at the tp placement: each card holds its half of every expert, inter 320) each card hands off
// its own half experts at every layer, both cards at once; the CPU computes each half (cpu::q4_expert at kQ4TpInter)
// into that card's cpu_y, and the card's down sends it to the peer as its own partial - the all-reduce and
// q4_moe_finish are unchanged. The host thread serves the two cards' hand-offs of a layer together - it waits for the
// second card's (microseconds behind the first: the cards meet at every layer's all-reduce) and runs both cards' half
// experts in one pool run a phase (cpu::q4_experts); the route's CPU cost is a half expert's in that batch.
//
// **The hand-off, per MoE layer with a mirror** (the capture, ListSpec::cpu): after q4_route the list resets the layer's
// `done` event, copies its M rows of x (bf16) and of the route to host memory and signals `ready`; gate||up runs without
// the CPU's experts (q4_moe.cl / q4_moe_union.cl CPU_ROUTE); the list waits on `done`; the down takes the CPU's fp32
// sums from cpu_y. The host thread polls the running M's `ready` events (each card's layers in order), finds the
// experts in the layer's mirror from its own copy of the card's table rows, gives the first dec[M][n] of the n to the
// CPU (src/cpu/expert_q4.cc, bit for bit the card's), writes cpu_y and signals `done`. The card and the host derive
// the same set from the same rows, route and table, which change only between steps.
//
// **The route** (--host-experts auto): at startup a ~100 ms measurement - one expert on the CPU at 1 and kVerifyRows
// rows, one expert's blocks over the actual PCIe link, the hand-off's own cost - gives each M's per-expert costs; dec
// splits a layer's n misses between the CPU and the link to finish soonest (the CPU's share runs while the card reads
// the rest), and the CPU-route lists run at M where an expert on the CPU plus a hand-off costs less than one over the link
// (2026-11-25: a rule on misses a layer turned 1 row off, and forcing it on measured +7.9 % decode). While decoding,
// the observed CPU time an expert (an EWMA; the CPU is shared) re-decides between steps; a change prints one line. b70-decode / b70-serve --host-experts: auto (the default) = Mode::Adaptive, cpu = Mode::All (every
// host expert on the CPU at every M; also qwen4exp_tier_test's `cpu`), gpu = none of this (the PCIe reads).
namespace runtime::qwen4exp {

class CpuExperts {
 public:
  enum class Mode { Off, Adaptive, All };
  static constexpr uint32_t kMaxN = kVerifyRows * 10;   // distinct experts a layer at most (top 10 a row)

  // `cards`: one context per part of `m` (views of one Level Zero context). Measures, decides, logs one line and
  // starts the host thread. Construct only when some layer has a mirror (has_host_experts).
  CpuExperts(const std::vector<l0::Context*>& cards, const loader::Q4LoadedModel& m, Mode mode);
  ~CpuExperts();
  CpuExperts(const CpuExperts&) = delete;
  CpuExperts& operator=(const CpuExperts&) = delete;
  static bool has_host_experts(const loader::Q4LoadedModel& m);

  // A layer's hand-off on card `card` at M rows (li: the part's layer index; valid when has(card, li)).
  struct Channel {
    ze_event_handle_t ready = nullptr, done = nullptr;
    void* x = nullptr;              // host: bf16 [M][2560]
    void* route = nullptr;          // host: u32 [M][32]
    const float* cpu_y = nullptr;   // host: fp32 [M][11][2560]
    const uint32_t* dec = nullptr;  // host: u32 [kMaxN + 1], this M's
    uint64_t mlo = 0, msz = 0;      // the layer's mirror
  };
  bool has(uint32_t card, uint32_t li) const;
  Channel channel(uint32_t card, uint32_t M, uint32_t li) const;

  bool on(uint32_t M) const { return on_[M]; }
  // Between steps, before lists at M rows run (every card's list at M, each through all its host-RAM layers): re-decide
  // on drift (one line when the route changes), then let the host thread poll M's events (when the CPU-route list runs
  // at M).
  void before_step(uint32_t M);
  // The rows the card's table now holds (the prefill's lend / reclaim; between steps).
  void set_rows(uint32_t card, const std::vector<uint64_t>& rows);
  // After a failed or abandoned step: every event reset, every card back at its first layer.
  void reset();
  // Between steps (b70-serve's reconfigure in place): the route for `mode` from the costs it holds (the startup
  // measurement, as auto's observations updated it); Off reads every host expert over PCIe. Silent - the caller says what changed.
  void set_mode(Mode mode);

 private:
  struct Card {
    uint32_t first = 0, n = 0;                  // the part's layers
    std::vector<uint32_t> host_layers;          // li with a mirror, in order
    std::vector<uint8_t> has;                   // [li]
    std::vector<uint64_t> mlo, msz;             // [li]
    std::vector<uint64_t> rows;                 // [li][1024]: the table rows (gate||up then down addresses)
    std::unique_ptr<l0::SyncEventPool> ev;      // [M - 1][li][ready, done]
    std::unique_ptr<l0::Mem> x, route, y;
    std::array<uint32_t, kVerifyRows + 1> cursor{};   // [M]: index into host_layers of the next hand-off
  };
  uint32_t ev_index(const Card& c, uint32_t M, uint32_t li, uint32_t which) const {
    return ((M - 1) * c.n + li) * 2 + which;
  }
  void measure(const std::vector<l0::Context*>& cards);
  void decide(bool log);
  bool route();   // dec and on_ for mode_ from the costs; true when on_ changed
  // A hand-off's CPU experts appended to `jobs` (q4_moe_union.cl's order); returns how many.
  uint32_t collect(const Card& c, uint32_t M, uint32_t li, std::vector<cpu::Q4Job>& jobs);
  void loop();

  Mode mode_;
  std::vector<Card> cards_;
  std::unique_ptr<l0::Mem> dec_;                // host: u32 [kVerifyRows + 1][kMaxN + 1]
  std::unique_ptr<cpu::Pool> pool_;
  unsigned cores_ = 0;
  // The kernel's instruction set: AVX-512 where the CPU has it, unless B70_CPU_EXPERTS_AVX2=1 (same bits by design).
  bool avx2_forced_ = false;
  cpu::Isa isa_ = cpu::Isa::Avx2;
  // The experts' width on a card (cpu::kQ4Inter, or kQ4TpInter under tp) and the cards that hand off the same layer at
  // once (tp: 2, else 1: the startup measurement's batch).
  uint32_t inter_ = 0, share_ = 1;
  // Costs in microseconds: an expert on the CPU at M rows, one over the link, a hand-off at M rows.
  std::array<double, kVerifyRows + 1> cpu_us_{}, hand_us_{};
  double pcie_us_ = 0.0;
  // The host thread's observations (written before it signals `done`; read between steps).
  std::array<double, kVerifyRows + 1> obs_cpu_{};
  std::array<uint32_t, kVerifyRows + 1> obs_n_{};
  std::array<bool, kVerifyRows + 1> on_{};
  std::atomic<uint32_t> poll_m_{0};             // the M whose events the host thread polls (0: none)
  std::atomic<uint32_t> pending_{0};            // hand-offs of the running step not yet served
  std::atomic<uint64_t> wake_{0};
  std::atomic<bool> quit_{false};
  std::thread th_;
};

}  // namespace runtime::qwen4exp
