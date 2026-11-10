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
#include "loader/loader.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/memory_plan.h"
#include "runtime/tp_protocol.h"

// Spec 17b: Qwen3.8 decode with tensor parallelism over two B70s (`b70-decode --parallel tp`).
//
//   card r   its slice of every layer (loader::load_tp: half the GDN and FA heads, half the MLP, its K half of
//            out_proj / o_proj / down, its vocabulary half of the int8 head), the whole embedding and norms, its
//            heads' GDN state / conv ring / KV, one decode list (runtime::build_tp), its own Control.
//
// One token is one replay of each card's list, submitted back to back; the lists order themselves - after each
// row-parallel linear a card pushes its part into the other card's landing buffer and its next residual fold waits for
// the other card's part in its own (src/kernels/tp.cl, runtime/tp_protocol.h). Both cards then hold the same residual,
// bitwise (T1), and with model::TpCrossing::Slices one card's residual, bitwise (T2).
//
// **The head's exchange (operator decision D2: a host merge).** Each card's argmax writes the best id of its half and
// that logit (Control::out_token / out_value) and advances pos. After both fences the host takes the larger value, ties
// to the lower id (card 0's ids are the lower half), and writes the winner into BOTH Controls' out_token and
// cur_token - argmax.cl's comparator, so the id is one card's argmax. That is the one host action per token beside
// the two fence waits, as --parallel pp's Control mirror is.
//
// **Bounded (T4).** Every fence wait is bounded (TpOptions::timeout_ms) and every in-list wait by spin_limit loads,
// a failure recorded in each card's state words and every later wait skipping its spin. A failed step throws naming
// what did not arrive and marks the engine: every later step throws until reset().
//
// Not here (plan 17b): prefill (17c - the prompt is ingested by decode replays), MTP / DFlash / lookup, the restore
// and the server (17d), profiling. save_state / save_kv ARE here, in the one-card host layouts (runtime/tp_gather.h).
namespace runtime {

// Operator decision D1 (plan 17b's review): the crossing a `--parallel tp` engine runs. One constant. The rule
// recorded with the decision - slices (bitwise one card) unless they cost more than ~5 % against folded - chose folded
// on row 79's first measurement (2026-11-10, depth 4096, interleaved, median of 3: one card 33.5 t/s, slices 50.0
// (1.49x), folded 53.9 (1.61x): slices 7.3 % slower). Folded is bitwise between the cards and judged against one card
// (tp_decode_test: identical ids on the gate prompts); Slices stays built and tested, one word to switch back.
constexpr model::TpCrossing kTpDefaultCrossing = model::TpCrossing::Folded;

struct TpOptions {
  uint32_t timeout_ms = 30000;                  // one step's fence wait (a step is ~18 ms)
  uint32_t spin_limit = tp::kDefaultSpinLimit;  // flag loads one wait spins before it gives up
  bool debug_resid = false;                     // the per-layer residual tap on both cards (T1 / T2 tests)
};

class TpEngine {
 public:
  // `cards` are loader::load_tp's rank 0 (on d0) and rank 1 (on d1), one crossing, loaded at `max_len`. `d1` is a
  // view of d0's context. Throws when either card cannot access the other's memory (each card writes the other's
  // landing buffer).
  TpEngine(l0::Context& d0, l0::Context& d1, std::array<loader::LoadedModel, 2> cards, uint32_t max_len,
           const TpOptions& opt = {}, KvCache kv = default_kv_cache());
  ~TpEngine();
  TpEngine(const TpEngine&) = delete;
  TpEngine& operator=(const TpEngine&) = delete;

  // Engine's contract on both cards: zeroes both persistent groups (both Controls), the landing buffers, the counters
  // and the state words; clears a failure.
  void reset();
  void ingest(const std::vector<uint32_t>& ids);
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);   // the pending id, into both Controls
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }

  uint32_t pos() const;
  uint32_t max_len() const { return max_len_; }
  model::TpCrossing crossing() const { return crossing_; }
  KvCache kv_cache() const { return kv_; }
  const CapturedStep& step(uint32_t card) const;
  DecodeBuffers& buffers(uint32_t card);
  const Control& control(uint32_t card) const;
  const loader::LoadedModel& model(uint32_t card) const;
  l0::Context& context(uint32_t card) const;

  // Spec 7's host layouts of ONE card (runtime/tp_gather.h): the same bytes Engine::save_state / save_kv write at the
  // same position on one card - bitwise equal under the slices crossing.
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host) const;
  void save_kv(uint32_t begin, uint32_t end, void* host) const;

  // The last step's full logits row, card 0's half then card 1's (fp32 [kVocab]) - T2's comparison.
  std::vector<float> read_logits() const;
  // The residual tap (debug_resid) of a card's last step, Engine::read_debug_resid's layout.
  std::vector<uint16_t> read_debug_resid(uint32_t card) const;

  MemoryComponents memory_use(uint32_t card) const;
  std::string memory_line(uint32_t card) const;

  // T4's test hook: the next step submits only the other card's list - card `card`'s parts never arrive. The step
  // must throw within the bounds, never hang; reset() recovers.
  void drop_next(uint32_t card) { drop_ = int(card); }

 private:
  struct Card;
  void step_once();
  bool settle(uint32_t card);
  [[noreturn]] void fail(const std::string& what);
  Card& cd(uint32_t card) const;

  uint32_t max_len_;
  TpOptions opt_;
  KvCache kv_;
  model::TpCrossing crossing_;
  tp::Layout layout_;
  std::array<std::unique_ptr<Card>, model::kTpSize> cards_;
  bool broken_ = false;
  int drop_ = -1;
  std::array<bool, model::kTpSize> pending_{};
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime
