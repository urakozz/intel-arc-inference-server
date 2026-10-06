// Spec 20c: the binaries a Kolibri-1 decode list binds (kernels::kolibri::decode_variants - the names
// runtime/kolibri/kolibri_capture.cc forms) for every attention arm, head form and attention form,
// against the ones src/kernels/CMakeLists.txt's Kolibri block builds (passed as arguments:
// tests/CMakeLists.txt's B70_KOLIBRI_DECODE_KERNELS, `kernel_<name>`). Host only: a name the capture
// would ask for that no CMake line builds fails here on the Mac, not at capture on the box; and
// nothing built is dead.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/kolibri_kernels.h"
#include "model/kolibri1.h"

int main(int argc, char** argv) {
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  // The descriptor's PROVISIONAL split-K cells are the names' (decode_variants spells S2 / S4).
  const model::Kolibri1Desc& d = model::kolibri1();
  CHECK(d.qkv_s == 2 && d.oproj_s == 4);
  CHECK(d.hidden == kernels::kolibri::kHidden && d.qkv_n() == kernels::kolibri::kQkvN &&
        d.router_n() == kernels::kolibri::kRouterN && d.vocab == kernels::kolibri::kVocab &&
        d.window == kernels::kolibri::kWindow && model::Kolibri1Desc::kRing == kernels::kolibri::kRing);
  std::set<std::string> bound;
  for (model::KolAttnForm a : {model::KolAttnForm::Int4, model::KolAttnForm::Bf16})
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (const std::string& v : kernels::kolibri::decode_variants(a, int8, eager)) bound.insert(v);
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Kolibri decode list binds %s, which the CMake Kolibri block does not build\n", v.c_str());
      return 1;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_KOLIBRI_DECODE_KERNELS) but nothing binds it\n", b.c_str());
      return 1;
    }
  std::printf("kolibri_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
