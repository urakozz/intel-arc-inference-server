// Spec 23c: the binaries a Muse-Glimmer decode list binds (kernels::muse::decode_variants - the names
// runtime/muse/muse_capture.cc forms) for every head form and attention form, against the ones
// src/kernels/CMakeLists.txt's Spec 23c block builds (passed as arguments: tests/CMakeLists.txt's
// B70_MUSE_DECODE_KERNELS, `kernel_<name>`). Host only: a name the capture would ask for that no CMake line builds
// fails here on the Mac, not at capture on the box; and nothing built is dead. The descriptor's PROVISIONAL split-K
// cells and widths are the names'.
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/muse_kernels.h"
#include "model/muse_glimmer.h"

int main(int argc, char** argv) {
  namespace km = kernels::muse;
  std::set<std::string> built;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  const model::MuseGlimmerDesc d = model::muse_glimmer();
  CHECK(d.qgkv_s == km::kQgkvS && d.oproj_s == km::kOprojS && d.gate_up_s == km::kGateUpS && d.down_s == km::kDownS);
  CHECK(d.hidden == km::kHidden && d.q_heads == km::kQHeads && d.kv_heads == km::kKvHeads && d.head_dim == km::kHd);
  CHECK(d.qgkv_rows() == km::kQgkvN && d.inter == km::kInter && 2 * d.inter == km::kGateUpN && d.vocab == km::kVocab);
  CHECK(d.window == km::kWindow && model::MuseGlimmerDesc::kRing == km::kRing);
  CHECK(km::kNormG * km::kRowWg == km::kHidden);   // Σ² chunks of exactly 256
  CHECK(km::argmax_groups() <= km::kArgmaxWg);     // argmax_stage2 folds at most 256 groups
  CHECK((km::kWindow + 1 + 63) / 64 + 1 <= km::kAttnTgt);   // a window's absolute 64-key blocks fit TGT
  std::set<std::string> bound;
  for (bool int8 : {false, true})
    for (bool eager : {false, true})
      for (const std::string& v : km::decode_variants(int8, eager)) bound.insert(v);
  int bad = 0;
  for (const std::string& v : bound)
    if (!built.count(v)) {
      std::fprintf(stderr, "the Muse decode list binds %s, which the CMake Spec 23c block does not build\n", v.c_str());
      ++bad;
    }
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (B70_MUSE_DECODE_KERNELS) but nothing binds it\n", b.c_str());
      ++bad;
    }
  if (bad) return 1;
  std::printf("muse_variant_names_test OK: %zu bound names, %zu binaries\n", bound.size(), built.size());
  return 0;
}
