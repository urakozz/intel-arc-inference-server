#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <memory>
#include <span>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/event.h"
#include "l0/kernel.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/inkling2/inkling2_buffers.h"
#include "runtime/inkling2/inkling2_prefill.h"

namespace runtime::inkling2 {

class Engine {
 public:
  // kv8: the 8 full visits' cache in int8 (`--kv-cache int8`, PERF-102); the rings and the drafter stay bf16.
  // int8_pass: the int8 prompt pass (`--prefill-backend l0-int8`, plan 41e): int8 scores on the int8 cache's full visits
  // and the glue rows in bf16; it runs on the int8 cache.
  Engine(l0::Context& context, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
         uint32_t prefill_rows = 0, bool kv8 = false, bool int8_pass = false)
      pre(prefill_rows <= kPrefillRows)
      pre(!int8_pass || kv8);
  void reset();
  uint32_t append(uint32_t token);
  // Appends the prompt at the current position and returns the first ID after
  // it. The caller passes live token IDs and an already planned chunk width.
  uint32_t prefill(std::span<const uint32_t> ids, uint32_t chunk)
      pre(pending_ == 0 && !ids.empty() && ids.size() <= max_len_ - committed_)
      pre(chunk > 0 && chunk <= prefill_rows_);
  void read_prefill_logits(std::span<float> out)
      pre(prefill_logits_ && out.size() == model::Inkling2Desc::vocab_used);
  // Until commit, the caller owns the previous position and must not replay
  // another target list. Each prediction follows the corresponding input. `candidates`: the greedy head, the plain
  // step's (record_candidate_head) - the ids only, read_logits then has no full head rows.
  std::span<const uint32_t> verify(std::span<const uint32_t> tokens, bool candidates = false);
  void commit(uint32_t rows);
  void read_logits(uint32_t row, std::span<float> out);
  // The drafter's tap slots (bf16 [rows][4][hidden], DSpark's taps) of rows [first, first + rows): the last verify's,
  // or with `prompt` the last prompt chunk's (b70-decode --dump-taps reads them from a block hook).
  void read_taps(bool prompt, uint32_t first, uint32_t rows, void* host)
      pre(buffers_.weights.draft != nullptr);
  // The DFlash 2 drafter's walk draws at `temperature` (0: the argmax walk) keyed by `seed` (df_select.cl's Gumbel
  // keying); DSpark's head is greedy and ignores both.
  std::span<const uint32_t> propose(uint32_t anchor, uint32_t count, float temperature = 0.f, uint64_t seed = 0);
  // The last DFlash 2 draft pass's readback: its drafts, each row's 16 candidates and the walk's realized scores.
  const dflash::DFlashDraftOut& draft_out() const pre(dflash()) {
    return *reinterpret_cast<const dflash::DFlashDraftOut*>(static_cast<const std::byte*>(
        buffers_.draft.draft_control->ptr()) + sizeof(dflash::DFlashCtl));
  }
  // The anchor and up to max_drafts drafts verified and their accepted prefix committed in one submission (a plain
  // step when no draft fits); appends the committed ids to output and returns the next anchor.
  uint32_t step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output);
  // One target step of `rows` rows at the position, as the plain and drafted steps run it (the greedy head), every
  // launch timed and printed by its stage (--profile); the rows are left uncommitted.
  void profile_target(std::span<const uint32_t> tokens)
      pre(pending_ == 0 && !tokens.empty() && tokens.size() <= kRows && tokens.size() <= max_len_ - committed_);
  // The draft pass after `anchor`, launch by launch (--profile with the drafter).
  void profile_draft(uint32_t anchor) pre(pending_ == 0 && committed_ > 0 && buffers_.weights.draft);
  // The last one-row step's expert ids, 8 a visit in visit order (--profile's route statistics).
  std::array<uint32_t, 8 * model::Inkling2Desc::visit_count> routes();
  size_t launch_count(uint32_t rows, uint32_t accepted) const {
    return target_launches_[rows - 1] + commit_launches_[rows - 1][accepted - 1];
  }
  uint32_t position() const { return committed_; }
  // A verify of `rows` rows at `pos` takes the deep lists (plan 41h Task 2): its full visits' attention on the prompt
  // flash's split form over kVerifySpans key spans (DPAS, every span's work-group busy) in place of kol_attn's rows,
  // whose cost grows with the rows (one fma chain a (head, row) pair). Measured, 4 full visits, int8 cache: the split
  // form 0.95 / 5.4 / 20.5 ms at 16k / 128k / 512k keys for 2 to 8 rows; kol_attn 2 rows 0.76 / 4.5 / 18.2, 3 rows
  // 0.98 / 5.9 / 24.0, 8 rows 2.3 / 15.1 / 61.1. Not bitwise the plain step's attention (q rounded to bf16 for the
  // DPAS, the spans merged): the f32 judge.
  static constexpr uint32_t kVerifySpans = 32, kDeepVerify = 16384;
  static bool deep_verify(uint32_t rows, uint32_t pos) { return rows >= 3 && pos >= kDeepVerify; }
  uint32_t max_len() const { return max_len_; }
  bool drafter() const { return buffers_.weights.draft != nullptr; }
  // The drafter is the DFlash 2 one (plan 41k, loader::Inkling2Checkpoint's `dflash` folder), not the DSpark head.
  bool dflash() const { return desc_.draft.form == model::DFlashForm::Vllm; }
  uint64_t packed_draft_key() const { return weights_.packed_draft_key; }
  bool kv8() const { return kv8_; }
  uint32_t prefill_rows() const { return prefill_rows_; }
  // The id the last prefill's head chose (its argmax over the live vocabulary), what prefill() returned.
  uint32_t head_id() const pre(prefill_logits_) { return ctrl_->out_token[0]; }

  // The prefix cache (plan 41f Task 4), runtime::Engine's calls. Every tensor the sizes structs tag (SnapTensor)
  // is stored; control and the drafter's recurrence (NotSnapshotted) are rebuilt from the restored position:
  //   state  the 40 window rings' rows of [pos - 1023, pos) (zeros for positions before 0; the window reads no
  //          older key), the 48 visits' convolution histories, the drafter's two window rings and its last tap row;
  //   kv     positions [begin, end) of the 8 full visits' K and V (int8 codes and fp16 scales under `kv8`) and of
  //          the drafter's full layer.
  // In host order [tensor][slot][rows][row bytes], tensors in SnapTensor order. Each call runs with no step in
  // flight (a pending verify is the caller's to commit first). load_state leaves the engine at `pos` with nothing
  // pending; load_kv follows it for [0, pos).
  static constexpr uint32_t kBlock = 2048;
  using BlockHook = std::function<void(uint32_t end_pos, bool is_block_end)>;
  // With a hook, prefill() cuts its chunks at the snapshot points (runtime::prefill_chunk_rows over kBlock, the
  // stride of a chunk wider than a block) and calls the hook on the host after each chunk that ends at a block end
  // before the prompt's end (pos == end_pos, no step in flight), then once at the prompt's end (its first id chosen,
  // head_id() readable). Empty: uniform chunks, no call.
  void set_block_hook(BlockHook hook) { block_hook_ = std::move(hook); }
  size_t state_bytes() const;
  size_t kv_bytes(uint32_t positions) const;
  void save_state(void* host) pre(pending_ == 0);
  void load_state(const void* host, uint32_t pos) pre(pos <= max_len_);
  void save_kv(uint32_t begin, uint32_t end, void* host) pre(pending_ == 0 && begin <= end && end <= max_len_);
  void load_kv(uint32_t begin, uint32_t end, const void* host) pre(pending_ == 0 && begin <= end && end <= max_len_);

 private:
  friend struct PrefillTestAccess;
  friend struct PpProbeAccess;   // tools/probe/probe_ink_pp_step.cc
  friend struct ReplayProbeAccess;   // tools/probe/probe_ink_replay.cc
  friend class PpEngine;
  friend class TpEngine;
  // `--parallel pp` (PpEngine): one card's share of the walk. Its weights are not uploaded here - PpEngine streams the
  // checkpoint once into both cards, then calls loaded(). The split is kPpSplit's (handoff_rows says what crosses).
  // `--parallel tp` (TpEngine): card tp.rank's part at its layout (tp_weights'), uploaded by TpEngine as under pp.
  Engine(l0::Context& context, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
         uint32_t prefill_rows, bool kv8, bool int8_pass, const Stage& stage,
         const loader::Inkling2WeightLayout& layout, const Tp& tp = {})
      pre(prefill_rows <= kPrefillRows)
      pre(!int8_pass || kv8)
      pre(stage.whole() || (stage.first == 0 ? stage.end == kPpSplit : stage.first == kPpSplit && stage.head()))
      pre(!tp.on() || stage.whole());
  void loaded();   // after the upload: the int8 pass's rotated linears, then reset()
  // The rows live at the split in this card's scratch, in handoff_row_bytes' order: the stage input (residual 0),
  // the finished blocks 0 and 1, the previous pass's layer-12 input and output (residuals 3 and 4), each `rows` x
  // hidden f32, then the rows' tap slots (bf16 [rows][4][hidden]; size 0 without taps). Decode's verify scratch, or
  // with `prompt` the prompt pass's.
  std::array<std::pair<void*, size_t>, 6> handoff_rows(uint32_t rows, bool prompt) const;
  uint32_t slot(const model::Inkling2Visit& v) const {   // the visit's cache slot on this card
    return v.kv_slot - (v.window ? share_.window_base : share_.full_base);
  }
  template <class T>
  static void argument(l0::Kernel& kernel, uint32_t index, T value) {
    if constexpr (std::is_pointer_v<T> || std::is_null_pointer_v<T>) kernel.arg_ptr(index, value);
    else kernel.arg(index, value);
  }

  template <class... Args>
  void launch(l0::CmdList& list, const std::string& variant, const char* entry, uint32_t group,
              uint32_t gx, uint32_t gy, Args... args) {
    launch_3d(list, variant, entry, group, gx, gy, 1, args...);
  }

  template <class... Args>
  void launch_3d(l0::CmdList& list, const std::string& variant, const char* entry, uint32_t group,
                 uint32_t gx, uint32_t gy, uint32_t gz, Args... args) {
    auto it = modules_.find(variant);
    if (it == modules_.end()) it = add_module(variant);
    kernels_.push_back(it->second->kernel(entry));
    auto& k = kernels_.back();
    k.group_size(group);
    uint32_t arg = 0;
    (argument(k, arg++, args), ...);
    if (!profile_pool_) return list.launch(k, gx, gy, gz);
    profile_names_.push_back(profile_stage_.empty() ? std::string(entry) : profile_stage_ + " " + entry);
    auto& event = profile_events_.emplace_back(*profile_pool_, uint32_t(profile_events_.size()));
    list.launch(k, gx, gy, gz, &event);
  }
  using Modules = std::map<std::string, std::unique_ptr<l0::Module>>;
  Modules::iterator add_module(const std::string& name);
  // candidates: the greedy head (record_candidate_head) - the plain and drafted steps and a greedy verify; the full
  // head's logits are what read_logits returns.
  // deep: a verify for depth (deep_verify), its full visits' attention on the split prompt flash.
  void record_target(l0::CmdList& list, uint32_t rows, bool candidates = false, bool deep = false);
  // `--parallel tp`: the all-reduce of `rows` fp32 hidden rows - this card's `part` into the other card's landing
  // (tp_send), then both cards' parts summed in one order (card 0's first) into `sum` (tp_sum); bounded waits.
  void record_all_reduce(l0::CmdList& list, uint32_t rows, const void* part, void* sum);
  void record_candidate_head(l0::CmdList& list, uint32_t rows);
  StepIo* step_io() const;
  void record_step_inputs(l0::CmdList& list);
  void record_step_outputs(l0::CmdList& list, bool round);
  void build_candidate_head();
  // count: the accepted rows on the device from elsewhere (pp's card 0: card 1's count in its StepIo).
  void record_commit(l0::CmdList& list, uint32_t rows, uint32_t accepted, bool on_device = false,
                     const uint32_t* count = nullptr);
  void set_commit_control(uint32_t rows);
  void record_draft(l0::CmdList& list);
  // `--parallel tp`: the card's first tile and scale of a k-major int8 head cut by rows (its kTpHeadHalf rows from rank
  // x kTpHeadHalf).
  std::pair<void*, void*> head_half(const std::string& name) const;
  void record_draft_context(l0::CmdList& list, uint32_t rows, bool on_device = false);
  // Plan 41k, the DFlash 2 drafter (dflash()): the block of 8 rows at ctrl pos (the anchor, 7 mask rows) through its
  // five layers, the target's int8 head over rows 1..7, the top 16 raw logits and the selector's capped walk - the
  // drafts into ctrl cur_token[1..7], the readback after the draft control; and the context of `rows` committed rows
  // from their tap rows (fc, hidden_norm, each layer's k || v, k_norm + RoPE into its ring slot pos % 2048, the
  // draft control's proj_pos / proj_n).
  void record_dflash_draft(l0::CmdList& list);
  void record_dflash_context(l0::CmdList& list, uint32_t rows);
  // The prompt chunk's rows at or after prefill_end_ - 2048 (the rings' reach), chunk-wide on pf_gemm.
  void record_dflash_prompt_context(l0::CmdList& list, const PrefillChunk& chunk, const PrefillBuffers& s);
  void record_prefill_chunk(l0::CmdList& list, const PrefillChunk& chunk, const uint32_t* ids, bool head);
  // One prompt chunk's walk, recorded in parts: under `--parallel tp` a part ends where an all-reduce crosses (after
  // o_proj, after the experts' sum); on one card and under pp the chunk is one part. `s` is the chunk's scratch (tp
  // runs two chunks in flight, each with its own), `quantized` its int8 pass's cached input.
  struct PrefillWalk {
    PrefillChunk chunk;
    const uint32_t* ids;
    bool head;
    PrefillBuffers* s;
    size_t visit = 0;
    uint32_t phase = 0;   // the visit's next part: 0 its attention, 1 its experts, 2 the rest of the visit
    void *current = nullptr, *next = nullptr, *midpoint = nullptr, *prev_midpoint = nullptr;
    uint32_t finished = 0;
    const void* quantized = nullptr;
    bool done = false;
  };
  struct Crossing {
    const void* part;   // this card's rows (fp32 [rows][hidden])
    void* sum;          // where both cards' rows go, card 0's first
    uint32_t rows;
  };
  PrefillWalk prefill_walk(const PrefillChunk& chunk, const uint32_t* ids, bool head, PrefillBuffers& s);
  // The walk up to its next all-reduce (returned; the next part starts after its sum is in place) or to its end.
  std::optional<Crossing> record_prefill_part(l0::CmdList& list, PrefillWalk& walk);
  // tp: right after the visit's experts' all-reduce and convolution (a part's first launches), the visit's slots of
  // the state a snapshot stores, copied into `shadow` (the state buffers' images, Buffers::tp_shadow) - before the
  // chunk after it rewrites them. `visit` the walk's index.
  void record_visit_shadow(l0::CmdList& list, uint32_t visit, uint32_t shadow);
  // The offset of tensor t's image in one tp shadow (the state tensors' buffers whole, in SnapTensor order).
  size_t shadow_image(SnapTensor t) const;
  // The prompt pass's linear, on the walk's scratch (pf_) and its cached int8 input (quantized_).
  void prefill_linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output);
  void rotate_int8_linears();
  void copy_logits(uint32_t row, std::span<float> out);
  void report_profile(const std::string& what, double wall);
  void profile_list(const std::string& what, const std::function<void(l0::CmdList&)>& record);
  void linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output);
  const void* weight(const std::string& name) const;
  // One contiguous range of a buffer in a snapshot's host order; `zero`: positions before 0 (state only).
  struct SnapRun {
    l0::Mem* mem;
    size_t offset, bytes;
    bool zero;
  };
  std::vector<SnapRun> state_runs(uint32_t pos) const;
  std::vector<SnapRun> kv_runs(uint32_t begin, uint32_t end) const;
  l0::Mem* snapshot_buffer(SnapTensor t) const;

  l0::Context& ctx_;
  const model::Inkling2Desc desc_;
  const loader::Inkling2WeightLayout weights_;
  const uint32_t max_len_;
  const uint32_t prefill_rows_;
  const bool kv8_;
  const bool int8_pass_;
  const Stage stage_;
  const Tp tp_;
  const StageShare share_;
  Buffers buffers_;
  Control* ctrl_;
  l0::Queue queue_;
  Modules modules_;
  std::vector<l0::Kernel> kernels_;
  std::array<std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, 2>, 2> target_;   // [candidates][deep]
  std::unique_ptr<l0::CmdList> draft_list_;
  std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, kRows> commits_;
  std::array<std::unique_ptr<l0::CmdList>, kRows - 1> rounds_, deep_rounds_;   // step_drafted's list a draft count
  std::unique_ptr<l0::CmdList> plain_;                            // append's
  std::array<size_t, kRows> target_launches_{};
  std::array<std::array<size_t, kRows>, kRows> commit_launches_{};
  uint32_t committed_ = 0, pending_ = 0;
  // PERF-90: the positions any step or restore has written since the last reset (a verify's tentative rows too);
  // all of them before the first reset (the allocation's contents are unknown).
  uint32_t written_ = UINT32_MAX;
  BlockHook block_hook_;
  // PERF-89: inside a mid-prompt hook save_kv appends its copies here (the copy engine) and returns; they run beside
  // the next chunks (the positions they read are final) and are joined before prefill() returns.
  bool overlap_kv_ = false;
  std::unique_ptr<l0::CmdList> kv_store_;
  bool prefill_logits_ = false;
  uint32_t prefill_end_ = 0;   // the prompt being recorded ends here (the DFlash 2 drafter's in-window rows)
  bool trace_prelude_ = false;
  // B70_INK_GLUE=bf16 or the int8 pass: the prompt pass keeps its glue rows (messages, controller mixes, expert
  // outputs) in bf16. Lossy; unset or f32 without the int8 pass is the exact path.
  const bool bf16_glue_;
  // B70_PREFILL_PROFILE=2: every prompt-pass launch timed and named by its stage.
  std::unique_ptr<l0::EventPool> profile_pool_;
  std::vector<l0::Event> profile_events_;
  std::vector<std::string> profile_names_;
  std::string profile_stage_;
  // The int8 pass's rotated linears by weight name: (offset in buffers_.weights.prompt_int8, K, N, the stored
  // columns); and the input whose quantised rows int8_rows holds (cleared where the rows are rewritten).
  struct RotatedLinear {
    size_t offset;
    uint32_t K, N, columns;
  };
  std::map<std::string, RotatedLinear> int8_linears_;
  const void* quantized_ = nullptr;
  PrefillBuffers* pf_ = nullptr;   // the scratch of the walk being recorded
  // `--parallel tp`: the other card's landing (its TpBuffers::tp_landing, written by this card's tp_send) and the
  // waits' bound in flag loads; TpEngine sets both before any list is recorded.
  void* peer_landing_ = nullptr;
  uint32_t spin_limit_ = 0;
};

}  // namespace runtime::inkling2
