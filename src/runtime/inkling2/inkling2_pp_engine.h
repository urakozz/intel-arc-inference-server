#pragma once
#include <array>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "l0/fence.h"
#include "l0/sync_event.h"
#include "runtime/inkling2/inkling2_engine.h"

namespace runtime::inkling2 {

// `--parallel pp`: Inkling's walk over two cards, split at kPpSplit (inkling2_sizes.h). Card 0 runs visits 0-23 with
// the embedding (layers 0-17), card 1 visits 24-47 with the head and the drafter (layers 6-23); each holds its
// visits' caches, rings and convolution histories. Engine's contract, on two cards.
//
// **Decode and verify.** One regular list a card at each width: card 0's walk, its rows (Engine::handoff_rows) copied
// into card 1's landing and a barrier that signals the hand-off event; card 1's list waits on the event, copies the
// landing into its scratch and walks on, then the head and the argmax. The host writes both cards' Control blocks
// (the ids, pos, n_active), submits card 0's list then card 1's and waits on both fences with a bound; the ids are
// card 1's. A commit replays each card's commit list (its visits' convolution histories; card 1 the drafter's
// context rows). The drafter drafts on card 1 alone; its taps 0 and 1 (visits 4 and 22, card 0) cross with the rows.
//
// **A drafted round**, one submission a card, as one card's (Engine::step_drafted): card 1 drafts, copies the anchor
// and drafts into card 0's StepIo (host memory) and signals `drafted`; card 0 waits on it, takes them into its Control
// and walks the round's rows, hands them off as a verify does and waits on `accepted`; card 1 walks on, ink_accept
// counts the accepted prefix, card 1 commits it, copies the count into card 0's StepIo and signals `accepted`; card 0's
// convolution commit reads the count there. The host reads the ids and the count from card 1's StepIo.
//
// **The prompt pass, two chunks in flight.** Card 0 walks chunk j + 1 while card 1 walks chunk j. Per chunk: card 0's
// walk, then [it waits until the copy engine sent chunk j - 1] its rows into its outbox and a barrier signalling
// `ready`; card 0's copy engine waits on `ready` [and on card 1 having taken chunk j - 1 out of the landing], copies
// the outbox into card 1's landing and signals `sent`; card 1 waits on `sent`, copies the landing into its scratch,
// signals `taken` and walks. Each card runs the one-card launches in the one-card order on the same inputs, so the
// caches and the logits are one card's, bitwise. Every chunk has its own three events, so the host appends the whole
// prompt without waiting and both cards never wait for the host's recording.
//
// Every wait is bounded (timeout_ms a step, prefill_timeout_ms a chunk); a hand-off that does not arrive releases the
// waiting list, throws and marks the engine until reset().
class PpEngine {
 public:
  // card1 is a view of card0's context (l0::Context(card0, 1)); throws when card 0 cannot write card 1's memory.
  PpEngine(l0::Context& card0, l0::Context& card1, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len,
           bool draft, uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false,
           uint32_t timeout_ms = 30000, uint32_t prefill_timeout_ms = 120000)
      pre(prefill_rows <= kPrefillRows)
      pre(!int8_pass || kv8);
  ~PpEngine();
  void reset();
  uint32_t append(uint32_t token);
  uint32_t prefill(std::span<const uint32_t> ids, uint32_t chunk)
      pre(pending() == 0 && !ids.empty() && ids.size() <= max_len() - position())
      pre(chunk > 0 && chunk <= prefill_rows_);
  void read_prefill_logits(std::span<float> out) pre(prefill_logits_ && out.size() == model::Inkling2Desc::vocab_used);
  // The full head (read_logits gives its rows); append and step_drafted verify on card 1's candidate head, as one
  // card's plain and drafted steps do, so their ids are one card's.
  std::span<const uint32_t> verify(std::span<const uint32_t> tokens) { return verify(tokens, false); }
  void commit(uint32_t rows);
  void read_logits(uint32_t row, std::span<float> out);
  std::span<const uint32_t> propose(uint32_t anchor, uint32_t count);
  // One card's drafted round on two cards, one submission a card (the round's comment in the class comment).
  uint32_t step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output);
  size_t launch_count(uint32_t rows, uint32_t accepted) const;
  bool drafter() const { return head().drafter(); }
  bool dflash() const { return head().dflash(); }
  bool kv8() const { return head().kv8(); }
  uint32_t prefill_rows() const { return prefill_rows_; }
  uint32_t head_id() const pre(prefill_logits_) { return head().ctrl_->out_token[0]; }

  // The prefix cache, Engine's calls and host layouts (a snapshot moves between one card and pp unchanged): tensor by
  // tensor, card 0's slots before card 1's. With a hook, the prompt's chunks end at the snapshot points as on one card;
  // at a mid-prompt block end each card copies its state into one of two shadows right after the chunk, and the host
  // calls the hook - save_state reading the shadows, save_kv the final rows - while the next chunks run; the copies of
  // block end h are appended only after the hook stored block end h - 2. A hook that throws drains both cards and
  // marks the engine until reset().
  static constexpr uint32_t kBlock = Engine::kBlock;
  using BlockHook = Engine::BlockHook;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }
  size_t state_bytes() const { return cards_[0]->state_bytes() + head().state_bytes(); }
  size_t kv_bytes(uint32_t positions) const { return cards_[0]->kv_bytes(positions) + head().kv_bytes(positions); }
  void save_state(void* host) pre(pending() == 0);
  void load_state(const void* host, uint32_t pos) pre(pos <= max_len());
  void save_kv(uint32_t begin, uint32_t end, void* host) pre(pending() == 0 && begin <= end && end <= max_len());
  void load_kv(uint32_t begin, uint32_t end, const void* host) pre(pending() == 0 && begin <= end && end <= max_len());

  uint32_t position() const { return head().committed_; }
  uint32_t max_len() const { return head().max_len_; }
  uint32_t pending() const { return head().pending_; }
  // The prompt pass's chunk when none is asked: two chunks in flight need several chunks a prompt, so 2,048 rows,
  // and 4,096 from 32k on, where the wider chunk's own speed wins (measured at 4k / 16k / 32k).
  static uint32_t chunk_rows(size_t prompt) { return prompt >= 8 * size_t(kPrefillRows) ? kPrefillRows : 2048; }
  // One card's engine (its buffers are the tests'), and its plan's bytes.
  Engine& card(uint32_t i) pre(i < 2) { return *cards_[i]; }
  MemoryComponents memory(uint32_t i) const pre(i < 2);

 private:
  friend struct PpProbeAccess;   // tools/probe/probe_ink_pp_step.cc
  Engine& head() const { return *cards_[1]; }
  void set_position(uint32_t committed, uint32_t pending);   // on both cards' engines and Controls
  void run(l0::CmdList& list0, l0::CmdList& list1, const char* what);   // one replay a card, bounded
  [[noreturn]] void fail(const std::string& what);
  // Every run of both cards' snapshot runs in the host order: f(card, run, the run's offset in its card's own order).
  template <class F>
  static void in_host_order(const std::array<Engine*, 2>& cards, const std::array<std::vector<Engine::SnapRun>, 2>& runs,
                            F f);
  // Appends `rows` rows of a card's handoff_rows to `list`: into `buffer` (out) or out of it (in).
  void pack(l0::CmdList& list, uint32_t card, uint32_t rows, bool prompt, std::byte* buffer, bool out);
  std::span<const uint32_t> verify(std::span<const uint32_t> tokens, bool candidates);

  std::array<std::unique_ptr<Engine>, 2> cards_;
  const uint32_t prefill_rows_;
  const uint64_t timeout_ns_, prefill_timeout_ns_;
  std::array<std::unique_ptr<l0::Fence>, 2> fences_;
  std::unique_ptr<l0::SyncEvent> handoff_;   // the decode lists' hand-off
  std::unique_ptr<l0::SyncEvent> drafted_, accepted_;   // a drafted round's crossings from card 1 to card 0
  // The prompt pass's hand-off events, ready / sent / taken a chunk (the class comment); grown to the longest prompt.
  std::unique_ptr<l0::SyncEventPool> chunk_events_;
  // The prompt pass's lists (card 0's compute and copy engine, card 1's compute), made once: the host resets and
  // waits on chunk_events_ only while the lists that signalled them live. On compute-runtime 26.40 a host access to an
  // event after the immediate list that signalled it was destroyed writes into memory the process has reused (a
  // pp test's host vector lost a word, the heap was corrupted; tools/probe/probe_event_pool_check.cc).
  std::array<std::unique_ptr<l0::CmdList>, 3> prompt_lists_;
  // A verify's lists [candidate head][Engine::deep_verify][card][rows - 1].
  std::array<std::array<std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, 2>, 2>, 2> target_;
  std::array<std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, kRows>, 2> commits_;
  // A drafted round's lists [Engine::deep_verify][card][drafts - 1].
  std::array<std::array<std::array<std::unique_ptr<l0::CmdList>, kRows - 1>, 2>, 2> rounds_;
  std::array<size_t, kRows> target_launches_{};
  std::array<std::array<size_t, kRows>, kRows> commit_launches_{};
  bool prefill_logits_ = false, broken_ = false;
  BlockHook block_hook_;
  // Inside a mid-prompt hook: the shadow save_state reads (0 or 1), and the block end it holds.
  int shadow_ = -1;
  uint32_t shadow_pos_ = 0;
};

}  // namespace runtime::inkling2
