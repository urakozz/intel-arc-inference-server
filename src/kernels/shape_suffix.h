#pragma once
#include <string>

// Spec 15b: the variant-name suffixes of kernels that bake a MODEL shape their
// name does not otherwise carry - the hidden size (embed_gather, the bf16 a||b
// GEMV at its K), the GDN head counts (gdn_step, prep_gated_head, the prefill
// GDN chain) and the FA head counts (the attention kernels). Code-free and
// define-free, so model/model_desc.cc (ModelDesc::*_suffix) and kernels.h share
// this ONE home of the rule.
//
// The reference shape is Qwen3.8's - the shape src/kernels/*.cl bake by default
// and every pre-15b binary was built at - and it carries NO suffix, so every
// Qwen3.8 binary (and every Agnes binary: Agnes shares these shapes) keeps its
// name and its ocloc command line. Another model's shapes name a binary that
// exists only once a CMake block builds it with the matching defines (spec
// 15c); until then capture throws on the missing file, the same guard
// `kernels::path` gives every variant.
namespace kernels {

inline constexpr unsigned kRefHidden = 5120;                        // hidden_size
inline constexpr unsigned kRefGdnKHeads = 16, kRefGdnVHeads = 48;   // linear_num_{key,value}_heads
inline constexpr unsigned kRefFaQHeads = 24, kRefFaKvHeads = 4;     // num_{attention,key_value}_heads

inline std::string hidden_suffix(unsigned hidden) {
  return hidden == kRefHidden ? std::string() : "_D" + std::to_string(hidden);
}
inline std::string gdn_suffix(unsigned k_heads, unsigned v_heads) {
  return k_heads == kRefGdnKHeads && v_heads == kRefGdnVHeads
             ? std::string()
             : "_GK" + std::to_string(k_heads) + "V" + std::to_string(v_heads);
}
inline std::string fa_suffix(unsigned q_heads, unsigned kv_heads) {
  return q_heads == kRefFaQHeads && kv_heads == kRefFaKvHeads
             ? std::string()
             : "_Q" + std::to_string(q_heads) + "KV" + std::to_string(kv_heads);
}

}  // namespace kernels
