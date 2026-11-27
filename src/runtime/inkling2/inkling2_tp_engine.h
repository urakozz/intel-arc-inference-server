#pragma once
#include <array>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/fence.h"
#include "l0/sync_event.h"
#include "runtime/inkling2/inkling2_engine.h"
#include "runtime/inkling2/inkling2_tp_split.h"

namespace runtime::inkling2 {

// `--parallel tp` (plan 41j): one request on two cards, each holding half of every visit's heads and of every
// expert's width (tp_weights); the walk's glue (the depth mixes, the controllers, the norms, the router and its top
// 8, the convolutions over the hidden row), the head and the drafter run whole on both cards on the same rows, so
// both hold the same residual, the same ids and the same drafts. The drafter's layers split as the visits are (12 q /
// 4 kv heads, half the MLP's columns, an all-reduce after o_proj and after down), its head and Markov fold by rows (a
// one-row all-reduce of each card's best a draft): 13 all-reduces a draft pass. Engine's contract, on two cards.
//
// **Decode, verify, the drafted round.** Each card's list is the one-card list form (Engine's verify / plain step /
// drafted round, one submission) at its widths, with the all-reduce after o_proj and after the experts' down
// projection (Engine::record_all_reduce: tp.cl's tp_send into the other card's landing, tp_sum waiting in its own,
// card 0's part first, so both cards sum the same bits). The host submits card 0's list then card 1's and waits on
// both fences with a bound; then it reads both cards' crossing state words (a wait that ran out its bound, or a
// stale chunk, throws naming the card) and holds both cards' ids equal (a difference throws: the cards diverged).
// After any failure the engine stays failed until reset().
//
// **The prompt pass** (prefill, inkling2_tp_prefill.cc): two chunks in flight, the crossings on the copy engine.
//
// The prefix cache: each card's own tensors in Engine's host order, card 0's then card 1's half - the halves of the
// caches, rings and k / v histories; the replicated tensors (the attention and MLP convolution histories, the
// drafter's caches and taps) once, card 0's on save, written to both on load. A tp entry is not a one-card entry.
class TpEngine {
 public:
  // card1 is a view of card0's context; throws unless each card can write the other's memory.
  TpEngine(l0::Context& card0, l0::Context& card1, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len,
           bool draft, uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false,
           uint32_t timeout_ms = 30000, uint32_t prefill_timeout_ms = 120000)
      pre(prefill_rows <= kPrefillRows)
      pre(!int8_pass || kv8);
  ~TpEngine();
  void reset();
  uint32_t append(uint32_t token);
  uint32_t prefill(std::span<const uint32_t> ids, uint32_t chunk)
      pre(pending() == 0 && !ids.empty() && ids.size() <= max_len() - position())
      pre(chunk > 0 && chunk <= prefill_rows_);
  void read_prefill_logits(std::span<float> out) pre(prefill_logits_ && out.size() == model::Inkling2Desc::vocab_used);
  std::span<const uint32_t> verify(std::span<const uint32_t> tokens);
  void commit(uint32_t rows);
  void read_logits(uint32_t row, std::span<float> out);
  // Both cards draw with the same key, so their drafts stay equal.
  std::span<const uint32_t> propose(uint32_t anchor, uint32_t count, float temperature = 0.f, uint64_t seed = 0);
  const dflash::DFlashDraftOut& draft_out() const { return lead().draft_out(); }
  // One target step of `rows` rows, card 0's launch by launch (--profile; the all-reduces' waits show as tp_sum).
  void profile_target(std::span<const uint32_t> tokens)
      pre(pending() == 0 && !tokens.empty() && tokens.size() <= kRows && tokens.size() <= max_len() - position());
  // The draft pass after `anchor`, card 0's launch by launch (--profile with the drafter).
  void profile_draft(uint32_t anchor) pre(pending() == 0 && position() > 0 && drafter());
  uint32_t step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output);
  size_t launch_count(uint32_t rows, uint32_t accepted) const { return lead().launch_count(rows, accepted); }
  bool drafter() const { return lead().drafter(); }
  bool dflash() const { return lead().dflash(); }
  uint64_t packed_draft_key() const { return lead().packed_draft_key(); }
  bool kv8() const { return lead().kv8(); }
  uint32_t prefill_rows() const { return prefill_rows_; }
  uint32_t head_id() const pre(prefill_logits_) { return lead().ctrl_->out_token[0]; }

  static constexpr uint32_t kBlock = Engine::kBlock;
  using BlockHook = Engine::BlockHook;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t positions) const;
  void save_state(void* host) pre(pending() == 0);
  void load_state(const void* host, uint32_t pos) pre(pos <= max_len());
  void save_kv(uint32_t begin, uint32_t end, void* host) pre(pending() == 0 && begin <= end && end <= max_len());
  void load_kv(uint32_t begin, uint32_t end, const void* host) pre(pending() == 0 && begin <= end && end <= max_len());

  uint32_t position() const { return lead().committed_; }
  uint32_t max_len() const { return lead().max_len_; }
  uint32_t pending() const { return lead().pending_; }
  // The prompt pass's chunk: two chunks in flight need several a prompt, so 2,048 rows, and 4,096 from 8k on, where
  // the wider chunk's own speed wins (measured at 16k: 5,896 -> 6,384 t/s). prefill() takes at most this.
  static uint32_t chunk_rows(size_t prompt) { return prompt >= 2 * size_t(kPrefillRows) ? kPrefillRows : 2048; }
  Engine& card(uint32_t i) pre(i < 2) { return *cards_[i]; }
  MemoryComponents memory(uint32_t i) const pre(i < 2);

 private:
  Engine& lead() const { return *cards_[0]; }
  // Both cards' lists, card 0's first, each with its fence; bounded; the crossing's state words checked.
  void run(std::array<l0::CmdList*, 2> lists, const char* what);
  void check_crossings(const char* what);
  [[noreturn]] void fail(const std::string& what);
  void set_position(uint32_t committed, uint32_t pending);
  // Each card's snapshot runs in the host order: f(card, tensor, run, again). Card 1's runs of the replicated tensors
  // are left out (`both`: given too, as `again` after card 0's run of the same bytes - a load writes them twice).
  template <class F>
  void in_host_order(const std::array<std::vector<Engine::SnapRun>, 2>& runs, bool both, F f) const;

  std::array<std::unique_ptr<TpWeights>, 2> split_;
  std::array<std::unique_ptr<Engine>, 2> cards_;
  const uint32_t prefill_rows_;
  const uint64_t timeout_ns_, prefill_timeout_ns_;
  std::array<std::unique_ptr<l0::Fence>, 2> fences_;
  std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, 2> target_;
  std::array<std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, kRows>, 2> commits_;
  std::array<std::array<std::unique_ptr<l0::CmdList>, kRows - 1>, 2> rounds_;
  std::array<std::unique_ptr<l0::CmdList>, 2> plain_;
  bool prefill_logits_ = false, broken_ = false;
  BlockHook block_hook_;
  // The prompt pass's lists (engine members: their events outlive each prompt and are host-reset by the next) and
  // its events, grown to the longest prompt.
  std::array<std::unique_ptr<l0::CmdList>, 2> compute_, copy_;
  std::unique_ptr<l0::SyncEventPool> chunk_events_;
  // Inside a mid-prompt hook: the shadow save_state reads (0 or 1) and the block end it holds; save_kv's copies go on
  // each card's copy engine (Engine::kv_store_) beside the next chunks and are joined before prefill() returns.
  int shadow_ = -1;
  uint32_t shadow_pos_ = 0;
  bool overlap_kv_ = false;
};

}  // namespace runtime::inkling2
