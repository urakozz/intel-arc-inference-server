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

// Plan 21-QFN-v2b Task 2: Qwen3.8-Flash-Next decode with tensor parallelism over two B70s (`--parallel tp`) - Qwen3.8's
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
// warm-up step. Not here yet: prefill (ingest feeds one id a replay), the MTP head, snapshots.
namespace runtime::qwen4exp {

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

 private:
  struct Card;
  void submit(CapturedStep* l0, CapturedStep* l1);
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
};

}  // namespace runtime::qwen4exp
