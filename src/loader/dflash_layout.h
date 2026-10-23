#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "model/dflash.h"

// Spec 19b: the DFlash drafter's device bytes as one header-only formula, shared by the loader (which asserts its
// allocations add up to it) and the planner (runtime/dflash/dflash_sizes.h: `--max-len auto` counts the drafter).
//
// **Formats** (spec 19 decision 2, plan 19b Task 1): `Bf16` - every linear as shipped, in gemv_bf16's tiled layout
// (common::repack_bf16_tiled); `Int8` - every linear quantised at load per output row, spec 9's quantiser
// (loader::quantise_int8_tiled: s = max|w| / 127, q = rne(w / s)), in gemv_i8w's tiled layout plus fp32 row scales -
// 19a's simulated int8 arm, which equalled bf16's acceptance. Neither touches the selector's two codebooks (gathered
// rows, kept bf16) or the norms (fp32 widenings of the bf16 w, bit-exact).
//
// The linears (rows x cols = N x K), each one device weight:
//   fc          hidden x (taps x target hidden)
//   per layer   q||k||v  qkv_n x hidden          (the context projection runs its k||v rows, no copy)
//               o        hidden x q_n
//               gate||up 2 inter x hidden        (gate rows, then up rows)
//               down     hidden x inter
//               v2: attention_conv / mlp_conv kernel_projection  kproj_n x hidden (two weights)
//   v2          the selector's hidden_projection  sel_rank x hidden
// The small tensors (fp32): hidden_norm and the final norm [hidden] each; per layer input / post norms [2][hidden],
// q_norm || k_norm [2][head_dim] (one [layers][2][head_dim] block), v2 the conv bases [2 convs][2 sides][taps][hidden]
// (one [layers][...] block); the mask row [hidden] fp32 (the target's embedding row, or mask_embedding.safetensors).
// v2: the codebooks, bf16 [vocab][sel_rank] each.
namespace loader {

// Spec 23f adds Int4G64 (spec 23 decision 9: our int4 g64 symmetric RTN, quantised at load by loader/rtn.h's
// rtn_int4_g64 - P0's int4g64 arm, restated in tools/oracle/dflash_accept.py - and stored in gemv.cl's layout 1, the
// n-tile-major subgroup tiles (common/repack.h repack_int4_layout1), so a row range such as the context projection's
// k||v rows of q||k||v is a contiguous sub-range of the weight, as in the other two forms; S 1: the GEMV writes the
// [M][N] rows the df_* kernels read).
//
// Spec 26b adds Exact - the LiLiCorr drafter's NVFP4 linears as W' (bf16, exact) in gemv_bf16's tiles plus the tensor's
// global g as an fp32 output scale per column (loader/nvfp4.h, src/kernels/dflash/ll_gemv_bf16_g.cl); its BF16 linears are
// Bf16's tiles. Only the LiLiCorr loader builds it (parse_lilicorr_draft_format); the other drafters' `--draft-format`
// spellings are unchanged.
enum class DraftFormat { Bf16, Int8, Int4G64, Exact };
const char* draft_format_name(DraftFormat f);                 // "bf16" | "int8" | "int4" | "exact"
bool parse_draft_format(const std::string& s, DraftFormat& out);   // also "int4g64"
// Spec 26b: the LiLiCorr drafter's spellings - "exact" | "int8" | "int4" (also "int4g64"); "bf16" is not one (its NVFP4
// linears have no bf16 form short of folding g, which is not exact - spec 26 §3).
bool parse_lilicorr_draft_format(const std::string& s, DraftFormat& out);

// One linear's device bytes: bf16 K N x 2; int8 K N + 4 N (the scales); int4 g64 layout 1 (N / 16) x (K / 64) tiles
// of 136 u32 (K / 2 N + K / 32 N: the nibbles and the f16 scales, 4.25 bits a weight).
inline size_t dflash_linear_bytes(uint32_t K, uint32_t N, DraftFormat f) {
  if (f == DraftFormat::Int4G64) return size_t(N / 16) * (K / 64) * 136 * 4;
  return (f == DraftFormat::Bf16 || f == DraftFormat::Exact) ? size_t(K) * N * 2 : size_t(K) * N + size_t(N) * 4;
}

struct DFlashBytes {
  size_t linears = 0;      // every linear above
  size_t small = 0;        // norms, q/k norms, conv bases, the mask row
  size_t codebooks = 0;    // v2: pred + succ
  size_t total() const { return linears + small + codebooks; }
};

inline DFlashBytes dflash_weight_bytes(const model::DFlashDesc& d, uint32_t target_hidden, DraftFormat f) {
  DFlashBytes b;
  const uint32_t H = d.hidden;
  b.linears += dflash_linear_bytes(d.fc_k(target_hidden), H, f);
  size_t per_layer = dflash_linear_bytes(H, d.qkv_n(), f) + dflash_linear_bytes(d.q_n(), H, f) +
                     dflash_linear_bytes(H, d.gate_up_n(), f) + dflash_linear_bytes(d.inter, H, f);
  if (d.v2()) per_layer += 2 * dflash_linear_bytes(H, d.kproj_n(), f);
  b.linears += per_layer * d.layers;
  if (d.v2()) b.linears += dflash_linear_bytes(H, d.sel_rank, f);
  b.small = size_t(H) * 4 * 2                                   // hidden_norm, final norm
            + size_t(d.layers) * 2 * H * 4                      // input / post norms
            + size_t(d.layers) * 2 * d.head_dim * 4             // q_norm || k_norm
            + size_t(H) * 4;                                    // the mask row
  if (d.v2()) b.small += size_t(d.layers) * 2 * 2 * d.conv_taps * H * 4;   // conv bases
  if (d.v2()) b.codebooks = size_t(2) * d.vocab * d.sel_rank * 2;
  return b;
}

// Offsets inside the per-layer small blocks (floats).
inline size_t dflash_qk_norm_off(const model::DFlashDesc& d, uint32_t layer, uint32_t which /*0 q, 1 k*/) {
  return (size_t(layer) * 2 + which) * d.head_dim;
}
// conv base [layer][conv (0 attention, 1 mlp)][side (0 prepare, 1 finish)][tap][hidden]
inline size_t dflash_conv_base_off(const model::DFlashDesc& d, uint32_t layer, uint32_t conv, uint32_t side) {
  return ((size_t(layer) * 2 + conv) * 2 + side) * d.conv_taps * d.hidden;
}

}  // namespace loader
