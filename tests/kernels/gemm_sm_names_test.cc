// gemm_sm_names_test - spec 28b: kernels::kSmallMTuning (src/kernels/kernels.h) and the CMake block that
// builds gemm_sm (src/kernels/CMakeLists.txt, "Spec 28b") name the same binaries. Host only, no device.
//
// usage: gemm_sm_names_test <every kernel_gemm_sm_* target the block made>
// Every (row, M) of the table at kernels::gemm_sm_built_m must be built, and the block builds nothing else -
// a row whose knobs the block spells differently is a failure here on the Mac, not a missing file at
// capture on the box.
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "kernels/kernels.h"

int main(int argc, char** argv) {
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  std::set<std::string> want;
  for (const kernels::SmallMTuning& t : kernels::kSmallMTuning)
    for (unsigned M : kernels::gemm_sm_built_m(t)) want.insert(kernels::gemm_sm_variant(M, t));
  // keys are unique: one row per (K, N, S, layout)
  size_t rows = 0;
  for (const kernels::SmallMTuning& t : kernels::kSmallMTuning) {
    ++rows;
    CHECK(kernels::small_m_tuning(t.K, t.N, t.S, t.layout) == &t);
  }
  int bad = 0;
  for (const std::string& n : want)
    if (!built.count(n)) {
      std::fprintf(stderr, "kSmallMTuning names %s, which the CMake block does not build\n", n.c_str());
      ++bad;
    }
  for (const std::string& n : built)
    if (!want.count(n)) {
      std::fprintf(stderr, "the CMake block builds %s, which no kSmallMTuning row names\n", n.c_str());
      ++bad;
    }
  CHECK_EQ(bad, 0);
  // The naming rule, spelled out.
  const kernels::SmallMTuning* gu = kernels::small_m_tuning(5120, 34816, 8, 0);
  CHECK(gu != nullptr);
  CHECK_EQ(kernels::gemm_sm_variant(8, *gu), std::string("gemm_sm_M8_K5120_N34816_S8_L0_mg"));
  const kernels::SmallMTuning* qz = kernels::small_m_tuning(5120, 16384, 1, 1);
  CHECK(qz != nullptr);
  CHECK_EQ(kernels::gemm_sm_variant(4, *qz), std::string("gemm_sm_M4_K5120_N16384_S1_L1_mg_g256"));
  const kernels::SmallMTuning* hd = kernels::small_m_tuning(5120, 248320, 1, 2);
  CHECK(hd != nullptr);
  CHECK_EQ(kernels::gemm_sm_variant(7, *hd), std::string("gemm_sm_i8w_M7_K5120_N248320_wg"));
  CHECK_EQ(std::string(kernels::gemm_sm_entry(*hd)), std::string("gemm_sm_i8w"));
  CHECK_EQ(kernels::gemm_sm_wg_size(*hd), 256u);
  CHECK_EQ(kernels::gemm_sm_cols(*hd), 32u);
  CHECK_EQ(kernels::gemm_sm_out_s(*gu), 8u);
  CHECK_EQ(kernels::gemm_sm_grid_y(*gu), 8u);
  // The routing helpers: Gemv names today's binaries; SmallM a tuned shape's gemm_sm, an untuned one's gemv.
  CHECK_EQ(kernels::int4_linear_variant(kernels::LinearKind::Gemv, 4, 5120, 34816, 8, 0),
           kernels::gemv_variant(4, 5120, 34816, 8, 0));
  CHECK_EQ(kernels::int4_linear_variant(kernels::LinearKind::SmallM, 4, 5120, 34816, 8, 0),
           std::string("gemm_sm_M4_K5120_N34816_S8_L0_mg"));
  CHECK_EQ(kernels::int4_linear_variant(kernels::LinearKind::SmallM, 4, 2048, 128, 1, 1),
           kernels::gemv_variant(4, 2048, 128, 1, 1));   // Ornith's a||b: not tuned
  CHECK_EQ(kernels::int8_linear_variant(kernels::LinearKind::SmallM, 2, 2048, 248320),
           kernels::gemv_i8w_variant(2, 2048, 248320));   // Ornith's head: not tuned
  CHECK_EQ(kernels::int8_linear_variant(kernels::LinearKind::Gemv, 8, 5120, 248320),
           kernels::gemv_i8w_variant(8, 5120, 248320));
  std::printf("gemm_sm_names_test OK: %zu rows, %zu binaries, the block builds exactly these\n", rows, want.size());
  return 0;
}
