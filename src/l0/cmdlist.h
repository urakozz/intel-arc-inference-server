#pragma once
#include <cstddef>
#include <cstdint>
#include "l0/context.h"

namespace l0 {
class Kernel;

// Two flavours. immediate(): synchronous - every append executes and completes
// before returning; used for uploads and tests. regular(): in-order, recorded
// once, closed, executed many times by Queue::execute - the decode list.
class CmdList {
 public:
  static CmdList immediate(Context& ctx, uint32_t ordinal = 0);
  static CmdList regular(Context& ctx, uint32_t ordinal = 0);
  ~CmdList();
  CmdList(CmdList&& o) noexcept;
  CmdList(const CmdList&) = delete;
  CmdList& operator=(const CmdList&) = delete;

  void copy(void* dst, const void* src, size_t bytes);
  void fill(void* dst, uint32_t pattern, size_t bytes);
  // Appends a launch with gx*gy*gz work-groups; the kernel's group size must
  // already be set (Kernel::group_size).
  void launch(Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
  void close();
  void reset();
  bool is_immediate() const { return immediate_; }
  ze_command_list_handle_t handle() const { return l_; }

 private:
  CmdList(ze_command_list_handle_t l, bool immediate) : l_(l), immediate_(immediate) {}
  ze_command_list_handle_t l_ = nullptr;
  bool immediate_;
};
}  // namespace l0
