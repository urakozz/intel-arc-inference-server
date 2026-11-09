#pragma once
#include <cstdint>

#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"     // GemmBatch
#include "runtime/prefill/kernels.h"

namespace runtime::prefill {

inline constexpr uint32_t kPfGemmTile = 256;
inline uint32_t pad256(uint32_t v) { return (v + kPfGemmTile - 1) / kPfGemmTile * kPfGemmTile; }
// Open-work 3a: the most rows a GEMM runs as pf_gemm_small (pf_small_tile).
inline constexpr uint32_t kPfSmallMaxRows = 1024;
// Open-work 3b: the most rows an int4 linear runs as pf_gemm_q4 (pf_q4_tile) - plain, and with
// gate||up's SiLU epilogue.
inline constexpr uint32_t kPfQ4MaxRows = 384, kPfQ4SiluMaxRows = 256;

// Open-work 3a: the pf_gemm_small tile (src/kernels/prefill/pf_gemm_small.cl) an untransposed,
// unbatched GEMM of M rows binds - NA 8-row blocks x NB 16-column atoms a sub-group, SN sub-groups
// a work-group - or null: pf_gemm's 256 x 256 tile. pf_gemm gives a 1024-column slab 4 to 16
// work-groups up to 1024 rows, each walking a whole tile, so a short chunk paid a full chunk's
// GEMM time; the small tiles spread the rows that exist over every core. Bit-identical output
// (the same DPAS chain per accumulator), so the choice is speed only - tools/probe/
// probe_small_chunk.cc's measurement on Qwen3.8's linears (docs/probes/probe-small-chunk-2026-11-08.md).
struct PfSmallTile {
  uint32_t na, nb, sn;
  uint32_t rows() const { return 8 * na; }         // the rows a sub-group (and a work-group) owns
  uint32_t cols() const { return 16 * nb * sn; }   // the columns a work-group owns
};
const PfSmallTile* pf_small_tile(uint32_t M);

// Open-work 3b: the pf_gemm_q4 tile (src/kernels/prefill/pf_gemm_q4.cl) an int4 linear of M rows
// runs as - pf_gemm_small reading the int4 words and converting them in registers, one launch over
// the weight's whole N, no bf16 slab - or null: the slab walk (pf_dequant_slab, then gemm_l0, per
// 1024 columns). `silu`: gate||up's fused epilogue (paired atoms, NB even, or NB1 gate/up passes
// sharing bf16 X at 193-256 rows). The slab's dequant is ~110 ms a Qwen3.8 chunk whatever the
// rows; the fused kernel converts each weight once per row
// tile, so it wins for few rows only. Bit-identical to the slab walk (the same conversion, the same
// DPAS chain per accumulator), so the choice is speed only - tools/probe/probe_fused_small.cc
// (docs/probes/probe-fused-small-chunk-2026-11-08.md).
const PfSmallTile* pf_q4_tile(uint32_t M, bool silu);

// One int4 linear through pf_gemm_q4 at tile `t`: C [pad(M, 8 NA)][N] (fp32 at pitch ldc, or
// with `silu` bf16 x [..][N / 2] at pitch ldc = ldx) = A [M][K] (pitch lda) x the weight
// (`qw` / `scales` in loader layout `layout`: 0 GPTQ-native with its f16 scales, 1 tiled with
// them inside). Rows [M, pad(M, 8 NA)) are computed and never read - the caller's buffers hold
// them. With SiLU and NB1, gate and up use separate launches (layout 0): gate bf16 is stored in
// C, then the up pass reads it and writes the same final bf16 x as the paired epilogue. Each
// launch covers 2 * t.cols() weight columns. Throws on the tile grid, K % 64 and block-IO rules.
void gemm_q4(Context& cx, KernelCache& kc, const PfSmallTile& t, uint32_t layout, bool silu, uint32_t M,
             uint32_t K, uint32_t N, const uint16_t* A, uint32_t lda, const void* qw, const void* scales,
             void* C, uint32_t ldc);

// Launch pf_gemm (src/kernels/prefill/pf_gemm.cl) on cx's list for the LOGICAL problem `b`.
// M is padded to 256 here; so is N when transB (the untransposed shapes, a 1024-column slab
// and P·V's N = 256, are already multiples). The padded rows and columns are computed and
// never read -- the CALLER guarantees they lie inside its buffers (spec 2.1 §3.1, §3.4).
// Throws on any 2D block-IO rule the kernel header states: K % 32, 64-byte bases (per batch),
// 16-byte pitches, leading dimensions that cover the matrix. An untransposed L = 1 problem whose
// M has a pf_small_tile runs pf_gemm_small instead, over pad(M, 8 NA) rows only.
void gemm_l0(Context& cx, KernelCache& kc, const GemmBatch& b, const uint16_t* A,
             const uint16_t* B, float* C, bool transB);

// Parity program S2(a): the same GEMM with `pf_silu_mul`'s chain fused into the
// epilogue. `b` describes ONE 1024-column slab of gate||up (L = 1, B is [K][N]),
// `X` is the x row base for that slab -- x + n0/2, since a 32-column gate||up
// block becomes 16 x columns -- and `ldx` is the whole x row pitch (17408), NOT
// the slab's. `b.ldc` is ignored: no fp32 C exists. The slab's GLOBAL column
// base must be a multiple of 32 or the gate/up parity the kernel assumes flips;
// `linear_l0_silu` is the only caller and walks 1024-column slabs.
void gemm_l0_silu(Context& cx, KernelCache& kc, const GemmBatch& b, const uint16_t* A,
                  const uint16_t* B, uint16_t* X, uint32_t ldx);

}  // namespace runtime::prefill
