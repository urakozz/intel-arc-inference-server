#pragma once
#include <cstddef>
#include <cstdint>

#include "model/qwen35.h"

// The per-layer "small tensor" blocks: everything a layer needs that is not a
// GEMV weight, packed into at most two device allocations so plan 3's captured
// command list binds two pointers per layer instead of six.
//
// **This header is the one place these offsets exist.** Three consumers read
// it and none of them may re-derive a number:
//
//   * `src/loader/loader.cc` packs the blocks against these constants;
//   * `src/model/qwen35.cc` builds `model::LayerDesc::small_tensors` - the
//     table that names, types and *places* every small tensor - out of these
//     very constants (fix I3, 2026-08-25), so the loader hardcodes nothing;
//   * **plan 3's GDN / FA / RMSNorm kernel bindings include this header** to
//     find each field inside the block they are handed.
//
// The `static_assert`s below pin every offset *and* every block size. Editing
// this file without telling the kernels fails the build here, not at run time.
//
// The byte offsets below are Qwen3.8's (`kQwen38Small`); another model's
// follow from its hidden size, conv channels and v-heads (`make_small_layout`).
//
// The dtypes are the *device* dtypes, which are deliberately not always the
// checkpoint's - see "What the loader bakes in" in docs/13-loader.md:
//
//   norms  (both layer kinds, 40960 B)
//     [0]      input_layernorm           fp32 (1 + w) [5120]
//     [20480]  post_attention_layernorm  fp32 (1 + w) [5120]
//
//   GDN    (164480 B)
//     [0]      linear_attn.conv1d        fp32 [10240][4]  (source bf16)
//     [163840] -exp(linear_attn.A_log)   fp32 [48]        (source bf16)
//     [164032] linear_attn.dt_bias       fp32 [48]        (source bf16)
//     [164224] linear_attn.norm          bf16 plain w [128]  (RMSNormGated: no
//              +1 - the one norm in the model without it, and the one that
//              stays bf16 because the reference multiplies in the bf16 domain)
//
//   FA     (2048 B)
//     [0]      self_attn.q_norm          fp32 (1 + w) [256]
//     [1024]   self_attn.k_norm          fp32 (1 + w) [256]
//
// The RMSNorm family is stored **fp32 `1 + w`** (controller ruling 2026-08-25):
// the HF reference computes `x·rsqrt(mean(x²)+ε)·(1 + w)` entirely in fp32, so
// casting the multiplier back to bf16 would add a rounding the reference never
// had - up to 0.39% on a multiplier near 1, at ~130 sites.
namespace loader {

// Spec 15b: the norms and GDN blocks follow the model's hidden size, conv
// channels and GDN v-heads, so their offsets are a function of the descriptor
// (`model::ModelDesc::small_layout()`); the FA block depends only on the
// shared head_dim and stays constant. Qwen3.8 and Agnes share every input, and
// `kQwen38Small` below pins their numbers - the ones the kernels' NEGA_OFF /
// DTBIAS_OFF defines (src/kernels/CMakeLists.txt) were derived from.

// The depthwise conv's taps (linear_conv_kernel_dim, 4 on every supported model).
constexpr size_t kConvTaps = 4;

struct SmallLayout {
  // --- norms block: both layernorms, every layer ---------------------------
  size_t norms_off_input, norms_off_post, norms_block_bytes;
  // --- GDN block ------------------------------------------------------------
  size_t gdn_off_conv, gdn_off_nega, gdn_off_dtbias, gdn_off_gated_norm, gdn_block_bytes;
  // --- the one norm that belongs to no layer ----------------------------------
  // `model.language_model.norm.weight`, before lm_head: its own allocation
  // (`LoadedModel::final_norm`), same fp32 (1 + w) bake as the rest.
  size_t final_norm_bytes;
};

// `conv_rows` is the GDN conv's channel count (q, k and v heads x 128).
constexpr SmallLayout make_small_layout(uint32_t hidden, uint32_t conv_rows,
                                        uint32_t gdn_v_heads) {
  SmallLayout l{};
  l.norms_off_input = 0;
  l.norms_off_post = l.norms_off_input + size_t(hidden) * 4;
  l.norms_block_bytes = l.norms_off_post + size_t(hidden) * 4;
  l.gdn_off_conv = 0;
  l.gdn_off_nega = l.gdn_off_conv + size_t(conv_rows) * kConvTaps * 4;
  l.gdn_off_dtbias = l.gdn_off_nega + size_t(gdn_v_heads) * 4;
  l.gdn_off_gated_norm = l.gdn_off_dtbias + size_t(gdn_v_heads) * 4;
  l.gdn_block_bytes = l.gdn_off_gated_norm + size_t(model::Qwen35::kGdnHeadDim) * 2;
  l.final_norm_bytes = size_t(hidden) * 4;
  return l;
}

// Qwen3.8's (and Agnes's) layout: hidden 5120, conv 10240 = (16 + 16 + 48) x 128,
// 48 v-heads. Pinned so an edit to the arithmetic above fails the build here.
constexpr SmallLayout kQwen38Small = make_small_layout(5120, 10240, 48);
static_assert(kQwen38Small.norms_off_input == 0 && kQwen38Small.norms_off_post == 20480,
              "norms block offsets changed");
static_assert(kQwen38Small.norms_block_bytes == 40960, "norms block size changed");
static_assert(kQwen38Small.gdn_off_conv == 0 && kQwen38Small.gdn_off_nega == 163840 &&
                  kQwen38Small.gdn_off_dtbias == 164032 &&
                  kQwen38Small.gdn_off_gated_norm == 164224,
              "GDN block offsets changed");
static_assert(kQwen38Small.gdn_block_bytes == 164480, "GDN block size changed");
static_assert(kQwen38Small.final_norm_bytes == 20480, "final norm size changed");
// Alignment holds for every model: offsets are sums of 4-byte (fp32) arrays,
// and the gated norm (bf16) follows them.
static_assert(kQwen38Small.gdn_off_nega % 4 == 0 && kQwen38Small.gdn_off_dtbias % 4 == 0,
              "fp32 fields must be 4-aligned");

// --- FA block (shared: head_dim 256 on every supported model) ---------------
constexpr size_t kFaOffQNorm = 0;
constexpr size_t kFaOffKNorm = kFaOffQNorm + size_t(model::Qwen35::kFaHeadDim) * 4;
constexpr size_t kFaBlockBytes = kFaOffKNorm + size_t(model::Qwen35::kFaHeadDim) * 4;
static_assert(kFaOffQNorm == 0 && kFaOffKNorm == 1024, "FA block offsets changed");
static_assert(kFaOffQNorm % 4 == 0 && kFaOffKNorm % 4 == 0, "fp32 fields must be 4-aligned");
static_assert(kFaBlockBytes == 2048, "FA block size changed");

}  // namespace loader
