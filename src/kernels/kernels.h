#pragma once
#include <string>

#ifndef B70_KERNEL_DIR
#error "B70_KERNEL_DIR must be defined (use b70_target_kernel_dir in CMake)"
#endif

namespace kernels {
// Absolute path of a compiled device binary by variant name.
inline std::string path(const std::string& variant) {
  return std::string(B70_KERNEL_DIR) + "/" + variant + ".bin";
}
inline std::string gemv_variant(unsigned M, unsigned K, unsigned N, unsigned S, unsigned L) {
  return "gemv_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_S" + std::to_string(S) + "_L" + std::to_string(L);
}
inline std::string gemv_bf16_variant(unsigned M, unsigned K, unsigned N) {
  return "gemv_bf16_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N);
}
}  // namespace kernels
