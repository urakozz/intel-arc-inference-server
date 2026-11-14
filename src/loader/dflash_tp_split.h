#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "loader/dflash_repack.h"
#include "model/dflash.h"

// Spec 19 v3c: the DFlash drafter split over the two cards of `--parallel tp` (spec 19 v3 §6.2 C1), on the host, over
// a linear in whatever device form the loader built - bf16 tiles, int8 tiles + row scales, int4 g64 layout 1 (quantised
// at load, or an AutoRound export's own packed words): each form is output-row-tile-major ([N/16][K/kt][tile]), so a
// card's output rows are whole 16-row tiles and its input rows whole k-tiles of every row tile. The split runs AFTER
// quantisation, so the two halves are exactly the one-card weight's bytes: the column halves concatenate to it, the
// row halves' partial sums add to its sum (int8's row scales, whole, on both halves; int4's group scales inside the
// tiles).
//   column-parallel (each card its output rows)  q||k||v by head (its q heads, its k heads, its v heads), gate||up
//                                                (its gate rows, its up rows)
//   row-parallel (each card its input rows)      o (its q heads' outputs), down (its half of the MLP)
//   replicated                                   fc, the conv kernel projections, the selector's projection
// Host only (tests/loader/dflash_tp_split_test.cc).
namespace loader {

// Rows [r0, r1) of the output (N), concatenated over `ranges` in order; each bound a multiple of 16.
DFlashHostLinear dflash_take_outputs(const DFlashHostLinear& h, const std::vector<std::array<uint32_t, 2>>& ranges);
// Input rows [k0, k1) of every output row (K), bounds multiples of the form's k-tile (bf16 8, int8 16, int4 64).
DFlashHostLinear dflash_take_inputs(const DFlashHostLinear& h, uint32_t k0, uint32_t k1);
// Card `rank`'s piece of a layer linear of the whole drafter `d` (the rules above; the conv kernel projections whole).
DFlashHostLinear dflash_tp_layer_linear(const model::DFlashDesc& d, DFlashLinear which, uint32_t rank,
                                        const DFlashHostLinear& whole);

}  // namespace loader
