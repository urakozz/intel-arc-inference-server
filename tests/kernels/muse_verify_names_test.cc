// Spec 23f: the binaries a Muse-Glimmer verify list binds at M = 1..16 rows (kernels::muse::verify_variants - the names
// runtime/muse/muse_capture.cc's build_verify forms), any head and attention form, against the CMake "Spec 23f" block's
// B70_MUSE_VERIFY_M16 list (tests/CMakeLists.txt's B70_MUSE_VERIFY_KERNELS) together with 23c's decode list
// (B70_MUSE_DECODE_KERNELS: the M = 1 names the verify at M = 1 shares with plain decode). Host only: a name a verify
// would ask for that no CMake line builds fails here on the Mac, and nothing the 23f block builds is dead.
//   muse_verify_names_test <23f kernel_...> -- <23c kernel_...>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/muse_kernels.h"

int main(int argc, char** argv) {
  namespace km = kernels::muse;
  std::set<std::string> verify_built, decode_built;
  bool after = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      after = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (after ? decode_built : verify_built).insert(a.substr(7));
  }
  CHECK(!verify_built.empty() && !decode_built.empty());
  std::set<std::string> bound;
  for (unsigned M = 1; M <= km::kVerifyMaxM; ++M)
    for (bool int8 : {false, true})
      for (bool eager : {false, true})
        for (const std::string& v : km::verify_variants(M, int8, eager)) bound.insert(v);
  int bad = 0;
  for (const std::string& v : bound)
    if (!verify_built.count(v) && !decode_built.count(v)) {
      std::fprintf(stderr, "a Muse verify list binds %s, which neither the Spec 23f nor the 23c block builds\n",
                   v.c_str());
      ++bad;
    }
  for (const std::string& b : verify_built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_MUSE_VERIFY_KERNELS) but no verify list binds it\n", b.c_str());
      ++bad;
    }
  // The verify at M = 1 is the plain step's chain: it shares every 23c name but the embedding and the argmax.
  const std::vector<std::string> ov = km::verify_variants(1, true, false), pv = km::decode_variants(true, false);
  const std::set<std::string> one(ov.begin(), ov.end()), plain(pv.begin(), pv.end());
  size_t shared = 0;
  for (const std::string& v : one) shared += plain.count(v);
  CHECK(shared == one.size() - 3);   // mg_pf_embed, mg_vargmax stage 1, stage 2
  if (bad) return 1;
  std::printf("muse_verify_names_test OK: %zu bound names over M = 1..16, %zu 23f binaries\n", bound.size(),
              verify_built.size());
  return 0;
}
