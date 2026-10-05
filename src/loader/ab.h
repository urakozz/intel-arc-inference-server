#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "common/repack.h"
#include "loader/quant.h"
#include "model/model_desc.h"

// The GDN a||b projection (`linear_attn.in_proj_a || in_proj_b`) in either of the two forms
// a checkpoint ships it in - host-only, no Level Zero, so tests/loader/ab_int4_test.cc runs
// it on any host over a synthetic checkpoint. loader::load uploads what these produce.
//
// **a||b's kind is the checkpoint's, like lm_head's** (model/model_desc.h `ab()`). Qwen3.8's
// and Agnes's exports keep in_proj_a / in_proj_b bf16 (their `dynamic` exclusion rules); the
// published Ornith int4 checkpoint (`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`) quantises
// them, int4 g64 sym, like every other GDN linear. The loader classifies the first GDN
// layer's in_proj_a by content (LinearSrc::kind_of: `.qweight`, compressed-tensors'
// `.weight_packed`, or `.weight`) and loads every GDN layer with that row through
// LinearSrc::classify - the one conversion point, so a compressed-tensors int4 a||b repacks to
// the same bytes (ab_int4_test) - marking LinearSrc::suffixes() consumed; a later layer in the
// other form throws by name (load_linear).
//
// The int4 row keeps the bf16 row's device contract: 2 x v-heads real columns (a at
// [0, v-heads), b at [v-heads, 2 v-heads)), ZERO-PADDED to 128 = gdn_step's AB_STRIDE, and the
// GEMV writes fp32 [M][128] straight at `ab_out` (gemv.cl at S = 1 writes out[m * N + n]).
// A padded column is the int4 encoding of zero: every nibble 8 (q - 8 = 0) under a zero f16
// scale, so its dot product is exactly 0 whatever the activations - and no consumer reads it.
//
// **Prefill** runs a||b as a bf16 GEMV (pf_ab_proj, pf_gemv_bf16.cl: 128 columns is no DPAS
// shape, and the slab dequant is 1024-column slabs). For an int4 a||b the loader therefore
// also keeps a bf16 copy, dequantised ONCE at load by the prefill dequant's own arithmetic
// (pf_dequant_slab.cl, the prefill path of every int4 linear): rne_bf16(float(q - 8) x
// float(scale)) - the product exact in fp32, one round to nearest even. pf_ab_proj then runs
// unchanged on it. Decode never reads the copy; the int4 GEMV computes with the exact
// fp32 weights.
namespace loader {

// Eight nibbles of 8: the int4 g64 sym encoding of eight zero weights.
inline constexpr uint32_t kInt4ZeroWord = 0x88888888u;

// The zero columns an int4 row is padded with (owned here; ColSource points into it).
struct Int4Pad {
  std::vector<uint32_t> qweight;   // [K/8][N] of kInt4ZeroWord
  std::vector<uint16_t> scales;    // [K/64][N] of +0.0 f16
  uint32_t N = 0;
};

// The row the checkpoint takes for a||b, from the kind of its in_proj_a: the table's bf16
// row (the same object as desc.linear(AB) - nothing about a bf16 a||b changes) or
// desc.ab(Int4).
const model::FusedLinear& ab_row_for(const model::ModelDesc& d, WKind in_proj_a_kind);

// The output-column map of an int4 fused row over its classified parts: Fuse::Concat or
// Fuse::Interleave16, then - for a row whose pad_n is set (a||b's 64 -> 128 on Ornith) - the
// zero columns up to shape.N, built into `pad`. Concat only: an interleaved row is never
// padded. Throws naming `id` when the parts do not fill the row.
std::vector<common::ColSource> int4_row_cols(const model::FusedLinear& fl,
                                             const std::vector<LinearSrc>& srcs, Int4Pad& pad,
                                             const std::string& id);

// The bf16 copy prefill reads: column n of `cols` (shape.N of them, the padding included)
// dequantised rne_bf16(float(q - 8) * float(scale)) per weight into row n of an [N][K]
// matrix, then tiled as gemv_bf16 / pf_ab_proj read it (common::repack_bf16_tiled).
// `out` holds K x N halves.
void dequant_int4_bf16_tiled(const model::GemvShape& sh, const std::vector<common::ColSource>& cols,
                             uint16_t* out);

}  // namespace loader
