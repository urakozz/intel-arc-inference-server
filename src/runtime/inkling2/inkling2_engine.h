#pragma once
#include <array>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"
#include "runtime/inkling2/inkling2_buffers.h"

namespace runtime::inkling2 {

class Engine {
 public:
  Engine(l0::Context& context, const loader::Inkling2Checkpoint& checkpoint, uint32_t max_len, bool draft);
  void reset();
  uint32_t append(uint32_t token);
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
    list.launch(k, gx, gy, gz);
  }
  using Modules = std::map<std::string, std::unique_ptr<l0::Module>>;
  Modules::iterator add_module(const std::string& name);
  void record_target(l0::CmdList& list, uint32_t rows);
  void record_commit(l0::CmdList& list, uint32_t rows, uint32_t accepted);
  void set_commit_control(uint32_t rows);
  void record_draft(l0::CmdList& list);
  void record_draft_context(l0::CmdList& list, uint32_t rows);
  void linear(l0::CmdList& list, const std::string& name, uint32_t rows, const void* input, void* output);
  const void* weight(const std::string& name) const;

  l0::Context& ctx_;
  const model::Inkling2Desc desc_;
  const loader::Inkling2WeightLayout weights_;
  const uint32_t max_len_;
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
};

}  // namespace runtime::inkling2
