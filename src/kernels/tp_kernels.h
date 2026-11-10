#pragma once
// Spec 17b: the tensor-parallel kernels' variant names (src/kernels/tp.cl; the CMake block "Spec 17b" builds exactly
// these) - kernels.h's rule: every define a binary bakes is in its name, so a host / device disagreement is a missing
// file at capture.
#include <string>

#include "kernels/kernels.h"

namespace kernels {

// The Control word index of the winner's value (runtime::Control::out_value); CTRL_OUTV in the CMake block.
inline constexpr unsigned kCtrlOutValue = 20;
// Rows the crossing's landing and counters are laid out for (tp.cl TP_MROWS): DecodeScratchDims::kM.
inline constexpr unsigned kTpMRows = 8;

// tp_send + tp_res_fold for card `rank`: M rows of `hidden`, FOLD_G work-groups, `s_loc` local split-K slices, the
// slices or the folded form.
inline std::string tp_cross_variant(unsigned M, unsigned hidden, unsigned fold_g, unsigned s_loc, bool folded,
                                    unsigned rank) {
  return "tp_cross_M" + std::to_string(M) + "_K" + std::to_string(hidden) + "_G" + std::to_string(fold_g) + "_S" +
         std::to_string(s_loc) + (folded ? "_F" : "_X") + "_R" + std::to_string(rank);
}
// tp_argmax_stage1 / _stage2 over a card's `rows` logits, `used` of them real, its first id `base`.
inline std::string tp_argmax_variant(unsigned M, unsigned rows, unsigned used, unsigned base) {
  return "tp_argmax_M" + std::to_string(M) + "_N" + std::to_string(rows) + "_U" + std::to_string(used) + "_B" +
         std::to_string(base);
}

}  // namespace kernels
