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
// The between-GEMV kernels (src/kernels/prep.cl). Only `prep_res_norm` varies
// in shape: `prep_silu_mul` and `prep_gated_head` have the model's dimensions
// (17408 / gate||up S=4, 48x128 / qkv||z S=1) baked into the source, so `M` is
// their whole variant space.
inline std::string prep_res_norm_variant(unsigned M, unsigned K, unsigned S_PREV) {
  return "prep_res_norm_M" + std::to_string(M) + "_K" + std::to_string(K) + "_SP" +
         std::to_string(S_PREV);
}
inline std::string prep_silu_mul_variant(unsigned M) { return "prep_silu_mul_M" + std::to_string(M); }
inline std::string prep_gated_head_variant(unsigned M) {
  return "prep_gated_head_M" + std::to_string(M);
}
}  // namespace kernels
