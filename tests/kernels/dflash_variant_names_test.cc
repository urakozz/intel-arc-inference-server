// Spec 19b: every binary the DFlash lists bind (kernels::dflash::draft_variants, every drafter form x head form x
// draft vocabulary, both versions) is built, and the Spec 19b CMake block builds nothing they do not bind. Also:
// kernels::dflash::kQwen38 - the shapes the names spell - is z-lab/Qwen3.8-27B-DFlash2's config beside Qwen3.8.
// Host only (names, no device).
//
//   dflash_variant_names_test <config.json> kernel_<new>... -- kernel_<reused>...
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_kernels.h"
#include "model/dflash.h"
#include "model/model_desc.h"

int main(int argc, char** argv) {
  namespace kd = kernels::dflash;
  CHECK(argc > 2);
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  const kd::Shape& s = kd::kQwen38;
  CHECK(d.hidden == s.H && d.inter == s.I && d.q_heads == s.NH && d.kv_heads == s.NKV && d.head_dim == s.HD);
  CHECK(d.max_window() == s.win && s.ring == s.win && d.conv_group == s.group && d.conv_taps == s.ctaps);
  CHECK(d.sel_rank == s.rank && d.vocab == s.vocab && model::qwen38().vocab_used == s.used);
  CHECK(d.sel_top_k == kd::kTopK && d.block == kd::kM && d.taps.size() == 5);
  CHECK(s.ring % kd::kAttnPb == 0 && kd::kTopK * kd::kQs == kd::kWg && s.rank % kd::kQs == 0);

  std::set<std::string> fresh, reused;
  bool after = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") {
      after = true;
      continue;
    }
    CHECK(a.rfind("kernel_", 0) == 0);
    (after ? reused : fresh).insert(a.substr(7));
  }
  std::set<std::string> bound;
  for (bool v2 : {true, false})
    for (bool int8 : {true, false})
      for (bool int8_head : {true, false})
        for (unsigned dv : {0u, 32768u, 65536u, 131072u})
          for (const std::string& v : kd::draft_variants(v2, int8, int8_head, dv)) bound.insert(v);
  int bad = 0;
  for (const std::string& v : bound)
    if (!fresh.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the DFlash lists bind %s, which no CMake block builds\n", v.c_str());
      ++bad;
    }
  for (const std::string& b : fresh)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built by the Spec 19b block but nothing binds it\n", b.c_str());
      ++bad;
    }
  if (bad) return 1;
  std::printf("dflash_variant_names_test OK: %zu bound names (%zu built by the 19b block, %zu reused)\n", bound.size(),
              fresh.size(), reused.size());
  return 0;
}
