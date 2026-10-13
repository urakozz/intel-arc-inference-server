// Spec 22a (plan Task 2 Step 1): tools/probe/probe_offload_moe.cl is generated from src/kernels/qwen4exp/q4_moe.cl
// by tools/probe/mk_offload_kernel.py (the two expert-address lines as a pointer-table read, nothing else). This
// test reruns the generator in --check mode: a hand edit of the generated file, or a change to q4_moe.cl that was
// not regenerated, FAILs here. A C++ runner rather than a bare `python3` add_test so the host-only configuration
// (cmake/host_only.cmake: a test's command must be a test executable of this tree) runs it on the Mac. Host only.
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef B70_SOURCE_DIR
#error "B70_SOURCE_DIR must be defined (tests/CMakeLists.txt)"
#endif

int main() {
  const std::string cmd = std::string("python3 \"") + B70_SOURCE_DIR + "/tools/probe/mk_offload_kernel.py\" --check";
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::fprintf(stderr, "probe_offload_kernel_drift_test: FAIL (%s exited %d)\n", cmd.c_str(), rc);
    return 1;
  }
  std::printf("probe_offload_kernel_drift_test OK\n");
  return 0;
}
