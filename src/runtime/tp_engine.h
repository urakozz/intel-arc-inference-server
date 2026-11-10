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
// **Prefill (spec 17c).** prefill() runs the prompt through both cards' prefill walks (runtime/prefill/tp_prefill.cc,
// compiled into b70_prefill_host as Engine::prefill is: a decode-only binary links no prefill code). The prompt is cut
// into chunks of at most the chunk cap and the chunks go in PAIRS: each card walks the pair's two chunks layer by
// layer, half-layer by half-layer (runtime::prefill::step_tp_unit), alternating - chunk A's mixer of layer l, chunk
// B's, A's MLP, B's - and after each row-parallel linear the card's half crosses to the other card on its COPY ENGINE
// while its compute engine runs the other chunk's unit (the wavefront: spec 17 §R2.8's option W, operator decision
// D8). Each unit then folds both halves (pf_tp_res_fold, card 0's first): the cards stay bitwise each other (T1); one
// card's prefill sums the K halves in one chain, so the result is judged against it, not bitwise (plan 17c).
//
// **MTP (spec 17d).** With the MTP head loaded (loader::load_tp's `mtp`) the head is WHOLE on each card at one card's
// shapes: both cards hold the same residual, so both run the same head on the same input - no crossing - and keep the
// same head KV (MtpBuffers' TP form). Engine's MTP contract, per card:
//   draft(k)      each card's draft list i (runtime::build_tp_draft: the head, its lm_head the card's vocabulary
//                 half, tp.cl's argmax) and the host's merge of the two winners (D2's rule) into both head Controls
//                 before list i + 1 - a fence per draft list, as one card's draft() has;
//   verify(k)     each card's verify list at M = k + 1 (runtime::build_tp_verify: the TP walk at M rows with the
//                 crossings, the per-row GDN slots, the head's KV fill), the per-row merge as a decode step's;
//   commit(j, t)  both cards: row j's GDN slot live, row j's hidden into hh row 0, pos, the pending id.
// ingest() / generate() run verify(0) + commit(0) then (pos 0: the plain list and the hidden copy), as Engine's. The
// verify lists bind DPAS attention by depth (spec 28c's second list), as Engine's. Built for the folded crossing
// (kTpDefaultCrossing); the slices form runs plain decode. No draft vocabulary, no DFlash drafter (not built).
//
// **The server's calls (spec 17d).** PipelineEngineAdapterT's interface (cli/pipeline_serve_adapter.h), so b70-serve
// --parallel tp is that adapter over this engine: pending / set_draft_input / host_rows, read_logits_into gathering
// both cards' halves of each row into one full row (operator decision D9: the gather; the sampler then filters the
// whole row on the host exactly as one card's), read_draft_logits_into likewise, and spec 7's snapshots in the
// one-card host layouts both ways (runtime/tp_gather.h: save gathers, load scatters; with MTP the head's hidden row
// and KV layer, card 0's - the cards' are the same bytes), and prefill's block hook (runtime/prefill/tp_prefill.cc).
//
// Not here: DFlash / prompt lookup's own proposer (lookup verifies through the MTP verify lists, so it needs the head
// loaded), profiling, the l0-int8 prefill backend.
namespace runtime {

struct TpPrefill;   // runtime/prefill/tp_prefill.cc

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
  // Spec 17c's timing seam (tp_prefill_test `time`): the prefill appends no crossing and no wait for one - each fold
  // adds whatever its landing slot holds - so the wall is the cards' compute alone. The numbers are WRONG; never set
  // outside that measurement.
  bool prefill_timing_no_crossing = false;
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
  // With MTP each id runs verify(0) + commit(0) (pos 0: the plain list and the hidden copy) - Engine's rule.
  void ingest(const std::vector<uint32_t>& ids);
  // Spec 17c: the prompt by prefill on both cards (the header's **Prefill**), from pos(): afterwards pos() is
  // pos + ids.size() and the first generated id is in both Controls' cur_token, as after ingest(). `chunk` (0 = the
  // cap) bounds a chunk's rows. prepare_prefill() allocates the scratch, the crossing's buffers and events (prefill()
  // calls it; the CLI calls it before its timed window); set_prefill_chunk_cap (2048 or 4096) before it.
  void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);
  void prepare_prefill();
  uint32_t prefill_chunk_cap() const { return pf_cap_; }
  void set_prefill_chunk_cap(uint32_t cap);
  size_t prefill_launches() const;   // the prefill lists' launches so far, both cards'
  // Whether the crossings ran on the copy engines (the wavefront) or, on a device without a copy-only queue group,
  // in order on the compute lists. Valid after prepare_prefill().
  bool prefill_overlaps() const;
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  void set_token(uint32_t id);   // the pending id, into both Controls
  uint32_t pending() const;      // spec 17d: the pending id (both Controls hold it)
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
  // Spec 17d: with MTP the state ends with the head's hidden row h_{pos-1} and the KV carries the head's layer after
  // the last FA layer (Engine's MTP layouts); load_* are the scatters - state into slot 0 of each card (gdn_live 0),
  // pos set, n_active 0, cur_token NOT restored (prefill at least one id after a restore, spec 7 §3.3).
  static constexpr uint32_t kBlock = 2048;   // Engine::kBlock
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t n_pos) const;
  void save_state(void* host) const;
  void save_kv(uint32_t begin, uint32_t end, void* host) const;
  void load_state(const void* host, uint32_t pos);
  void load_kv(uint32_t begin, uint32_t end, const void* host);
  // Spec 7 §3.2 under TP (runtime/prefill/tp_prefill.cc): called after a pair of chunks that ends on a block end of
  // the hook's stride (both cards finished it), and after the prompt's end - Engine's contract. The stride is a
  // pair's rows rounded up to whole blocks (2 x the chunk cap: 4096 at cap 2048 - plan 5d's rule (A) for a chunk
  // wider than a block); a pair never straddles one.
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }

  // --- spec 17d: MTP (the header's **MTP**) ------------------------------------------------------------------------
  static constexpr uint32_t kMaxDraft = MtpBuffers::kMaxK;   // 3
  bool mtp() const { return mtp_; }
  uint32_t max_draft() const { return mtp_ ? kMaxDraft : 0; }
  uint32_t draft_vocab() const { return 0; }
  uint32_t max_verify_k() const;
  void draft(uint32_t k, const std::function<uint32_t(uint32_t)>& pick = {});
  const std::vector<uint32_t>& draft_ids() const { return draft_ids_; }
  void verify(uint32_t k);
  const uint32_t* verify_ids() const;   // the merged argmax of each verify row (both Controls hold it)
  void commit(uint32_t j, uint32_t next_token);
  void set_draft_input(uint32_t i, uint32_t id);   // cur_token[1 + i] of both Controls (prompt lookup's drafts)
  // The last step's logits, `rows` full rows of kVocab: each row card 0's half then card 1's (D9: the gather).
  void read_logits_into(float* host, uint32_t rows = 1) const;
  void read_draft_logits_into(float* host, uint32_t i) const;   // draft i's full row, the same gather
  float* host_rows(size_t floats);   // pinned host rows of the one context (either card's copies reach them)
  const CapturedStep& verify_step(uint32_t card, uint32_t M) const;
  const CapturedStep& draft_step(uint32_t card, uint32_t i) const;
  uint32_t verify_deep_lists() const;   // the per-M lists captured twice (by depth), one card's count
  const MtpBuffers* mtp_buffers(uint32_t card) const;   // tests: a card's head buffers (null without the head)

  // The last step's full logits row, card 0's half then card 1's (fp32 [kVocab]) - T2's comparison.
  std::vector<float> read_logits() const;
  // The residual tap (debug_resid) of a card's last step, Engine::read_debug_resid's layout.
  std::vector<uint16_t> read_debug_resid(uint32_t card) const;

  MemoryComponents memory_use(uint32_t card) const;
  std::string memory_line(uint32_t card) const;

  // T4's test hook: the next step submits only the other card's list - card `card`'s parts never arrive. The step
  // must throw within the bounds, never hang; reset() recovers. Spec 17d: a verify (or draft) step too.
  void drop_next(uint32_t card) { drop_ = int(card); }
  // tp_prefill_test's timing seam (TpOptions::prefill_timing_no_crossing) switched between calls.
  void set_options_for_test(const TpOptions& o) { opt_ = o; }

 private:
  struct Card;
  void step_once();
  // Spec 17d: one step of the given lists (each card's), bounded as step_once; with `crossings` (a decode or verify
  // list) the cards' state words are read and the head's rows merged after the fences (a draft list merges itself).
  void run_lists(const std::array<l0::CmdList*, model::kTpSize>& lists, bool crossings, const char* what);
  void mtp_step1();
  void mtp_normalise_live();
  void require_mtp(const char* what) const;
  void capture_mtp_lists(uint32_t card);
  void merge_heads();
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
  // Spec 17d: MTP's session state (Engine::Seq's for one sequence), the hook, the pinned rows.
  static constexpr uint32_t kNoVerify = 0xFFFFFFFFu;
  bool mtp_ = false;
  uint32_t verify_k_ = kNoVerify, verify_pos_ = 0;
  std::vector<uint32_t> draft_ids_;
  BlockHook block_hook_;
  std::unique_ptr<l0::Mem> host_rows_;
  // Spec 17c: the prefill half (allocated by prepare_prefill; deleted through the pointer prefill code set, so this
  // archive needs none of it), its chunk cap and each card's device bytes (memory_use). Declared after cards_: it
  // goes first.
  uint32_t pf_cap_;
  std::array<size_t, model::kTpSize> pf_bytes_{};
  std::unique_ptr<TpPrefill, void (*)(TpPrefill*)> pf_{nullptr, nullptr};
};

}  // namespace runtime
