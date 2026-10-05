// Spec 18b: the binaries K2's decode list binds (kernels::k2::decode_variants - the names
// runtime/k2/k2_capture.cc forms) against the ones src/kernels/CMakeLists.txt's K2 block
// builds (passed as arguments: tests/CMakeLists.txt's B70_K2_DECODE_KERNELS, `kernel_<name>`).
// Host only, no device: a name the capture would ask for that no CMake line builds fails here
// on the Mac instead of at capture on the box.
#include <algorithm>
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "kernels/k2_kernels.h"
#include "model/k2_horizon.h"

int main(int argc, char** argv) {
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  const model::K2Desc& d = model::k2();
  for (bool int8 : {false, true})
    for (const std::string& v : kernels::k2::decode_variants(d, int8)) {
      if (!built.count(v)) {
        std::fprintf(stderr, "the K2 decode list binds %s, which the CMake K2 block does not build\n",
                     v.c_str());
        return 1;
      }
    }
  // And nothing in the list is dead: every built K2 binary is bound by one of the two heads.
  std::set<std::string> bound;
  for (bool int8 : {false, true})
    for (const std::string& v : kernels::k2::decode_variants(d, int8)) bound.insert(v);
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_K2_DECODE_KERNELS) but no K2 list binds it\n", b.c_str());
      return 1;
    }
  std::printf("k2_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
