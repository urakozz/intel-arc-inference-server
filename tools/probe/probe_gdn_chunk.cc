// P5 - first build fork for Intel's CuTe chunked gated delta rule.
//
// The header is included exactly once. This target intentionally makes no
// timing or correctness claim until its torch and CuTe dependencies compile.
#include <cstdio>

#include "chunk_gated_delta_rule_kernels_xe2.hpp"
#include "tla_pin.h"

#undef printf

int main() {
  std::printf("P5 compile fork: CuTe chunked GDN header parsed; sycl-tla=%s (%s)\n",
              B70_SYCL_TLA_PIN, B70_SYCL_TLA_SHA);
  return 0;
}
