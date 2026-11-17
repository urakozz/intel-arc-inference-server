#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/qwen4exp_loader.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/qwen4exp/qwen4exp_buffers.h"
#include "runtime/tp_protocol.h"

// Plan 21v2b Task 2: Qwen3.8-Flash-Next decode with tensor parallelism over two B70s (`--parallel tp`) - Qwen3.8's
// TpEngine arrangement (runtime/tp_engine.h) over this family's lists (runtime/qwen4exp/qwen4exp_capture.h, tp).
//
//   card r   its half of every layer (loader::load_qwen4exp at the tp placement: model::q4_tp_card's shapes), the final
//            mixer and its vocabulary half of the int8 head; the embedding on card 0 (card 1 gathers its row from it
//            over peer access); its own Qwen4ExpBuffers at the card's shapes, Control, crossing buffers and one list.
//
// A token is one replay of each card's list, submitted back to back; the lists order themselves through the crossings
// (src/kernels/tp.cl: tp_send into the peer's landing buffer, tp_sum waiting in the card's own; bounded by
// spin_limit loads, a failure recorded in the card's state words). After both fences (bounded) the host reads the
// state words - any failure throws naming the card and the chunk, and the engine stays failed until reset() - then
// merges the two heads: the larger value, ties to card 0 (its ids are the lower half), written into both Controls'
// out_token and cur_token. Both cards hold the same residual, and it is one card's bit for bit: the crossings are the
// slices form in one card's summation order (runtime/qwen4exp/qwen4exp_capture.h, tp).
//
// At construction: the PLE hash constants on both cards, the expert tier's table check on each card, one discarded
// warm-up step. Not here yet: the MTP head, snapshots.
//
// Plan 21v2b Task 4, prefill (defined in b70_qwen4exp_prefill: qwen4exp_tp_prefill.cc, the one-card engine's
// arrangement): each chunk of at most kPfC positions is the tensor-parallel walk (qwen4exp_prefill.h, Q4TpPrefill) on
// both cards' immediate prefill lists - two crossings a layer, card to card copies and cross-device events - then a
// bounded wait on both; the prompt's last row runs prefill_head_tp on both cards and the host merges the two heads as a
// decode step's. Leaves pos += ids.size() and the first generated id pending, the state decode continues from (to the
// prefill's cosine bar against decode's fill, not bitwise: plan 21d Review Focus 5, and the K halves summed apart).
namespace runtime::qwen4exp {

struct Q4TpPrefillState;   // runtime/qwen4exp/qwen4exp_tp_prefill.cc

struct Q4TpOptions {
  uint32_t timeout_ms = 30000;                  // one step's fence wait
  uint32_t spin_limit = tp::kDefaultSpinLimit;  // flag loads a wait spins before it gives up
};

class Qwen4ExpTpEngine {
 public:
  // `d1` is a view of d0's context; both cards must access each other's memory. `model` was loaded at the tp placement.
  Qwen4ExpTpEngine(l0::Context& d0, l0::Context& d1, loader::Q4LoadedModel model, uint32_t max_len,
                   const Q4TpOptions& opt = {});
  ~Qwen4ExpTpEngine();
  Qwen4ExpTpEngine(const Qwen4ExpTpEngine&) = delete;
  Qwen4ExpTpEngine& operator=(const Qwen4ExpTpEngine&) = delete;

  void reset();
  void ingest(const std::vector<uint32_t>& ids);
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);
  uint32_t pending() const;
  uint32_t pos() const;
  uint32_t vocab() const { return model_.desc.vocab; }
  uint32_t max_len() const { return max_len_; }
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  size_t launches(uint32_t card) const;
  const CapturedStep& step(uint32_t card) const;
  const loader::Q4LoadedModel& model() const { return model_; }
  // The test hooks: card `card`'s 4-stream H after the last replay (bf16 [10240]); fp32 [vocab] logits, both halves;
  // u32 [layers][32] route rows (card 0's - both cards route on the same bits); the crossing's state words.
  std::vector<uint16_t> read_H(uint32_t card);
  std::vector<float> read_logits();
  std::vector<uint32_t> read_routes(uint32_t card);
  std::array<uint32_t, tp::kStateWords> state_words(uint32_t card) const;
  // Probes only: card `card`'s list rebuilt with a timestamp event a launch, both lists replayed `replays` times;
  // each launch's mean device time in us, walk order (step(card).labels).
  std::vector<double> profile_decode(uint32_t card, uint32_t replays);
  // The expert tier's device read-back at construction: block words read through each card's table.
  size_t expert_words_checked() const { return words_checked_; }

  // --- plan 21v2b Task 4: prefill (b70_qwen4exp_prefill) ---------------------------------------------------------------
  // Runs `ids` from pos in chunks of `chunk` (0 = kPfC) on both cards, then the head on the last row: pos +=
  // ids.size() on both, the first generated id pending in cur_token. A crossing or a card that does not finish within
  // the bound releases both lists, throws naming the card and the chunk, and the engine stays failed until reset().
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  // Allocates both cards' prefill scratch, fold slots, down rows and the crossing events, and checks every binary the
  // walk binds exists; prefill() calls it. A CLI calls it before timing the first prefill.
  void prepare_prefill();
  size_t prefill_launches() const;   // both cards' prefill lists, since prepare_prefill
  // u32 [layers][kPfC][32]: every layer's route rows of the last chunk (card 0's; both cards route on the same bits).
  std::vector<uint32_t> read_prefill_routes();

 private:
  struct Card;
  void submit(CapturedStep* l0, CapturedStep* l1);
  void merge_heads();   // the two cards' head halves into both Controls (the larger value, ties to card 0)
  [[noreturn]] void fail(const std::string& what);
  void ple_init();
  void expert_check();
  CapturedStep build_card(uint32_t r, ProfileEvents* prof);

  loader::Q4LoadedModel model_;
  uint32_t max_len_;
  Q4TpOptions opt_;
  std::array<std::unique_ptr<Card>, 2> cards_;
  bool broken_ = false;
  size_t words_checked_ = 0;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0;
  // Plan 21v2b Task 4: releases a prefill list waiting on a crossing that will not come and waits (bounded) for both
  // prefill lists to go idle; reset() calls it. The prefill state is lazy and declared LAST, destroyed before the
  // cards whose buffers it points into; its deleter is set where it is made (b70_qwen4exp_prefill).
  std::function<bool()> pf_settle_;
  std::unique_ptr<Q4TpPrefillState, void (*)(Q4TpPrefillState*)> pf_{nullptr, nullptr};
};

}  // namespace runtime::qwen4exp
