#pragma once
#include <cstddef>

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

// qkv width x conv kernel dim - the GDN depthwise conv's device shape.
constexpr size_t kConvRows = 10240, kConvTaps = 4;

// --- norms block: both layernorms, every layer -------------------------------
constexpr size_t kNormsOffInput = 0;
constexpr size_t kNormsOffPost = kNormsOffInput + size_t(model::Qwen35::kHidden) * 4;
constexpr size_t kNormsBlockBytes = kNormsOffPost + size_t(model::Qwen35::kHidden) * 4;
static_assert(kNormsOffInput == 0 && kNormsOffPost == 20480, "norms block offsets changed");
static_assert(kNormsBlockBytes == 40960, "norms block size changed");

// --- GDN block ---------------------------------------------------------------
constexpr size_t kGdnOffConv = 0;
constexpr size_t kGdnOffNegA = kGdnOffConv + kConvRows * kConvTaps * 4;
constexpr size_t kGdnOffDtBias = kGdnOffNegA + size_t(model::Qwen35::kGdnVHeads) * 4;
constexpr size_t kGdnOffGatedNorm = kGdnOffDtBias + size_t(model::Qwen35::kGdnVHeads) * 4;
constexpr size_t kGdnBlockBytes = kGdnOffGatedNorm + size_t(model::Qwen35::kGdnHeadDim) * 2;
static_assert(kGdnOffConv == 0 && kGdnOffNegA == 163840 && kGdnOffDtBias == 164032 &&
                  kGdnOffGatedNorm == 164224,
              "GDN block offsets changed");
static_assert(kGdnOffNegA % 4 == 0 && kGdnOffDtBias % 4 == 0, "fp32 fields must be 4-aligned");
static_assert(kGdnOffGatedNorm % 2 == 0, "bf16 field must be 2-aligned");
static_assert(kGdnBlockBytes == 164480, "GDN block size changed");

// --- FA block ----------------------------------------------------------------
constexpr size_t kFaOffQNorm = 0;
constexpr size_t kFaOffKNorm = kFaOffQNorm + size_t(model::Qwen35::kFaHeadDim) * 4;
constexpr size_t kFaBlockBytes = kFaOffKNorm + size_t(model::Qwen35::kFaHeadDim) * 4;
static_assert(kFaOffQNorm == 0 && kFaOffKNorm == 1024, "FA block offsets changed");
static_assert(kFaOffQNorm % 4 == 0 && kFaOffKNorm % 4 == 0, "fp32 fields must be 4-aligned");
static_assert(kFaBlockBytes == 2048, "FA block size changed");

// --- the one norm that belongs to no layer -----------------------------------
// `model.language_model.norm.weight`, before lm_head: its own allocation
// (`LoadedModel::final_norm`), same fp32 (1 + w) bake as the rest.
constexpr size_t kFinalNormBytes = size_t(model::Qwen35::kHidden) * 4;
static_assert(kFinalNormBytes == 20480, "final norm size changed");

}  // namespace loader
