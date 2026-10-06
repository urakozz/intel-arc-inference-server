// Spec 18b: the binaries K2's decode list binds (kernels::k2::decode_variants - the names
// runtime/k2/k2_capture.cc forms) against the ones src/kernels/CMakeLists.txt's K2 block
// builds (passed as arguments: tests/CMakeLists.txt's B70_K2_DECODE_KERNELS, `kernel_<name>`).
// Host only, no device: a name the capture would ask for that no CMake line builds fails here
// on the Mac instead of at capture on the box.
//
// Spec 18c: `k2_variant_names_test prefill kernel_...` does the same for the prefill walk
// (kernels::k2::prefill_variants - the names runtime/k2/k2_prefill.cc forms - and the K1
// test's prefill_test_variants) against B70_K2_PREFILL_KERNELS.
#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "kernels/k2_kernels.h"
#include "model/k2_horizon.h"

int main(int argc, char** argv) {
  int first = 1;
  const bool prefill = argc > 1 && std::string(argv[1]) == "prefill";
  // Spec 18e: `kv8 kernel_...` - the binaries the int8 KV cache adds (decode and prefill, every
  // head and attention form; kernels::k2's names with kv8 = true that the bf16 lists do not
  // have) against B70_K2_KV8_KERNELS.
  const bool kv8 = argc > 1 && std::string(argv[1]) == "kv8";
  if (prefill || kv8) first = 2;
  std::set<std::string> built;
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    CHECK(a.rfind("kernel_", 0) == 0);
    built.insert(a.substr(7));
  }
  CHECK(!built.empty());
  const model::K2Desc& d = model::k2();
  std::vector<std::string> wanted;
  if (kv8) {
    std::set<std::string> bf16, int8;
    for (bool head : {false, true})
      for (bool eager : {false, true}) {
        for (const std::string& v : kernels::k2::decode_variants(d, head, eager, false)) bf16.insert(v);
        for (const std::string& v : kernels::k2::decode_variants(d, head, eager, true)) int8.insert(v);
      }
    for (bool k : {false, true}) {
      std::vector<std::string> l = kernels::k2::prefill_variants(d, k);
      for (const std::string& v : kernels::k2::prefill_test_variants(d, k)) l.push_back(v);
      for (const std::string& v : l) (k ? int8 : bf16).insert(v);
    }
    for (const std::string& v : int8)
      if (!bf16.count(v)) wanted.push_back(v);
  } else if (prefill) {
    wanted = kernels::k2::prefill_variants(d);
    for (const std::string& v : kernels::k2::prefill_test_variants(d)) wanted.push_back(v);
  } else {
    for (bool int8 : {false, true})
      for (bool eager : {false, true})   // B70_K2_ATTN=flash|eager (spec 18 §10.1)
        for (const std::string& v : kernels::k2::decode_variants(d, int8, eager)) wanted.push_back(v);
  }
  for (const std::string& v : wanted)
    if (!built.count(v)) {
      std::fprintf(stderr, "the K2 %s binds %s, which the CMake K2 block does not build\n",
                   kv8 ? "int8 KV cache" : prefill ? "prefill walk (or its K1 test)" : "decode list", v.c_str());
      return 1;
    }
  // And nothing in the list is dead: every built K2 binary is bound.
  const std::set<std::string> bound(wanted.begin(), wanted.end());
  for (const std::string& b : built)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built (%s) but nothing binds it\n", b.c_str(),
                   kv8 ? "B70_K2_KV8_KERNELS" : prefill ? "B70_K2_PREFILL_KERNELS" : "B70_K2_DECODE_KERNELS");
      return 1;
    }
  std::printf("k2_variant_names_test%s OK: %zu bound names, %zu binaries\n", kv8 ? " (kv8)" : prefill ? " (prefill)" : "",
              bound.size(), built.size());
  return 0;
}
