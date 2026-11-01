#pragma once
// gemm_sm_levers.h - plan 28b's lever alternates of the six Qwen3.8 rows of kernels::kSmallMTuning: the forms the
// production table is measured against (tools/probe/probe_small_m prod) and held to N2 / N3 at every M
// (tests/kernels/gemm_sm_test --levers). tools/probe/CMakeLists.txt builds exactly these at M = 2..8 and 16
// (GEMM_SM_LEVERS there - same order). 28a measured each lever at M = 8 only; 28b tests property R and the cost
// at every M before a lever stays in the table.
#include <vector>

#include "kernels/kernels.h"

namespace gemm_sm {

//                                 K      N      S  L  wg     magic  g256   blk2d
inline const std::vector<kernels::SmallMTuning>& lever_rows() {
  static const std::vector<kernels::SmallMTuning> v = {
      {5120, 16384, 1, 1, false, false, false, false},   // qkv||z  base (shift dequant, 128 GRF)
      {5120, 16384, 1, 1, false, true, true, false},     // qkv||z  magic + 256 GRF (28a's pick; not kept)
      {6144, 5120, 4, 0, false, false, false, true},     // out_proj / o_proj  base
      {5120, 14336, 2, 0, false, false, false, true},    // q||k||v  base
      {5120, 34816, 8, 0, false, false, false, false},   // gate||up  base (grid S 8)
      {5120, 34816, 8, 0, false, true, false, false},    // gate||up  grid S 8 + magic (the first table's row)
      {17408, 5120, 4, 0, false, false, false, true},    // down  base
      {5120, 248320, 1, 2, false, false, false, false},  // the int8 head  base (grid, 4 x 16 columns)
      {5120, 38912, 1, 0, true, true, false, false},     // Agnes gate'||up'  2 x 8 work-group (one slice)
  };
  return v;
}

}  // namespace gemm_sm
