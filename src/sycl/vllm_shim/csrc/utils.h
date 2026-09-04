#pragma once

// P5's header uses this dependency only for its BMG branch selection in the
// raw-pointer launcher. The torch entry point is not part of this probe.
namespace vllm::xpu {
inline bool is_bmg() { return true; }
}  // namespace vllm::xpu

