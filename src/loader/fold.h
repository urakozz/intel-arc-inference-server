#pragma once
#include <cstdint>
#include <vector>

// Spec 14 §2: Agnes's parallel FFN folded into the MLP, on packed GPTQ int4.
//
// GPTQ stores a linear as `qweight` u32 [K/8][N] (eight 4-bit k per word) and
// `scales` f16 [K/64][N]. The fold is two concatenations of those arrays:
//
//   fold_n  gate' = [gate | gate_p], up' = [up | up_p]: the columns of b appended
//           to a's, row by row (both arrays). K must match.
//   fold_k  down' = [down ; down_p]: b's packed rows appended after a's - at
//           `qweight` row a.K/8 and `scales` row a.K/64, which is a g64 group
//           boundary only because a.K % 64 == 0 (17408 = 272 x 64). N must match.
//
// Both are exact: no nibble is moved within a word and no scale is recomputed,
// so the folded tensors dequantise to the stacked dequantisations bit for bit.
// The identity `g_idx` the row join relies on is asserted for every tensor by
// loader::assert_quant_invariants. tools/oracle/agnes_fold.py is the reference
// (tests/loader/agnes_fold_test.cc compares the two on a shared fixture).
// Host-only: no device, no I/O.
namespace loader {

struct PackedInt4View {
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;   // [K/8][N]
  const uint16_t* scales = nullptr;    // [K/64][N] f16
};

struct PackedInt4 {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;   // [K/8][N]
  std::vector<uint16_t> scales;    // [K/64][N]
  PackedInt4View view() const { return {K, N, qweight.data(), scales.data()}; }
};

// Throw std::runtime_error naming the mismatch.
PackedInt4 fold_n(const PackedInt4View& a, const PackedInt4View& b);
PackedInt4 fold_k(const PackedInt4View& a, const PackedInt4View& b);

}  // namespace loader
