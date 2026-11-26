#pragma once
#include <array>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/event.h"
#include "l0/kernel.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/inkling2/inkling2_buffers.h"
#include "runtime/inkling2/inkling2_prefill.h"

namespace runtime::inkling2 {

class Engine {
 public:
  // kv8: the 8 full visits' cache in int8 (`--kv-cache int8`, PERF-102); the rings and the drafter stay bf16.
  Engine(l0::Context& context, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft,
         uint32_t prefill_rows = 0, bool kv8 = false)
      pre(prefill_rows <= kPrefillRows);
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
  // another target list. Each prediction follows the corresponding input.
  std::span<const uint32_t> verify(std::span<const uint32_t> tokens);
  void commit(uint32_t rows);
  void read_logits(uint32_t row, std::span<float> out);
  std::span<const uint32_t> propose(uint32_t anchor, uint32_t count);
  uint32_t step_drafted(uint32_t anchor, uint32_t max_drafts, std::vector<uint32_t>& output,
                        double* proposal_ms = nullptr);
  size_t launch_count(uint32_t rows, uint32_t accepted) const {
    return target_launches_[rows - 1] + commit_launches_[rows - 1][accepted - 1];
  }
  uint32_t position() const { return committed_; }
  uint32_t max_len() const { return max_len_; }

 private:
  friend struct PrefillTestAccess;
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
  void record_target(l0::CmdList& list, uint32_t rows);
  void record_commit(l0::CmdList& list, uint32_t rows, uint32_t accepted);
  void set_commit_control(uint32_t rows);
  void record_draft(l0::CmdList& list);
  void record_draft_context(l0::CmdList& list, uint32_t rows);
  void record_prefill_chunk(l0::CmdList& list, const PrefillChunk& chunk, const uint32_t* ids, bool head);
  void prefill_linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output);
  void copy_logits(uint32_t row, std::span<float> out);
  void linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output);
  const void* weight(const std::string& name) const;

  l0::Context& ctx_;
  const model::Inkling2Desc desc_;
  const loader::Inkling2WeightLayout weights_;
  const uint32_t max_len_;
  const uint32_t prefill_rows_;
  const bool kv8_;
  Buffers buffers_;
  Control* ctrl_;
  l0::Queue queue_;
  Modules modules_;
  std::vector<l0::Kernel> kernels_;
  std::array<std::unique_ptr<l0::CmdList>, kRows> target_;
  std::unique_ptr<l0::CmdList> draft_list_;
  std::array<std::array<std::unique_ptr<l0::CmdList>, kRows>, kRows> commits_;
  std::array<size_t, kRows> target_launches_{};
  std::array<std::array<size_t, kRows>, kRows> commit_launches_{};
  uint32_t committed_ = 0, pending_ = 0;
  bool prefill_logits_ = false;
  bool trace_prelude_ = false;
  // B70_INK_GLUE=bf16: the prompt pass keeps its glue rows (messages, controller mixes, expert outputs) in bf16.
  // Lossy, for the judge; unset or f32 is the exact path.
  const bool bf16_glue_;
  // B70_PREFILL_PROFILE=2: every prompt-pass launch timed and named by its stage.
  std::unique_ptr<l0::EventPool> profile_pool_;
  std::vector<l0::Event> profile_events_;
  std::vector<std::string> profile_names_;
  std::string profile_stage_;
};

}  // namespace runtime::inkling2
