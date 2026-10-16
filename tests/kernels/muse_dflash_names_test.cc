// Spec 23f: every binary Muse-Glimmer's DFlash draft / projection lists bind (kernels::dflash::muse_draft_variants -
// the names runtime/dflash/dflash_capture.cc forms for a Muse-form drafter, every drafter format bf16 / int8 / int4
// g64 x head int8 / bf16) is built - by the CMake "Spec 23f" block's B70_MUSE_DFLASH list or reused from its
// B70_MUSE_VERIFY_M16 list (the head at M = 15) - and the B70_MUSE_DFLASH list builds nothing they do not bind.
// Also: kernels::dflash::kMuse - the shapes the names spell - is meta-models/Muse-Glimmer-30B-assistant's published
// config (tests/model/muse_glimmer/assistant_config.json) beside Muse-Glimmer-30B. Host only (names, no device).
//
//   muse_dflash_names_test <assistant config.json> kernel_<new>... -- kernel_<reused>...
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_kernels.h"
#include "model/dflash.h"

int main(int argc, char** argv) {
  namespace kd = kernels::dflash;
  CHECK(argc > 2);
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  const kd::Shape& s = kd::kMuse;
  CHECK(d.muse() && s.muse);
  CHECK(d.hidden == s.H && d.inter == s.I && d.q_heads == s.NH && d.kv_heads == s.NKV && d.head_dim == s.HD);
  CHECK(d.block == s.M && s.M == kd::kMuseM && d.taps.size() == 5 && d.layers == 5);
  CHECK(d.max_window() + 1 == s.win && s.ring >= s.win && s.ring % s.pb == 0 && d.window_inclusive);
  CHECK(s.vocab == 202048 && s.used == s.vocab && d.vocab == 0);   // the target's head (the config carries none)
  CHECK(d.eps == 1e-5f && d.norm_two_roundings && !d.causal[0]);

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
  CHECK(!fresh.empty() && !reused.empty());
  std::set<std::string> bound;
  for (unsigned fmt : {0u, 1u, 2u})
    for (bool int8_head : {true, false})
      for (const std::string& v : kd::muse_draft_variants(fmt, int8_head)) bound.insert(v);
  int bad = 0;
  for (const std::string& v : bound)
    if (!fresh.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the Muse DFlash lists bind %s, which no CMake block builds\n", v.c_str());
      ++bad;
    }
  for (const std::string& b : fresh)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built by the Spec 23f B70_MUSE_DFLASH list but nothing binds it\n", b.c_str());
      ++bad;
    }
  CHECK(bad == 0);
  std::printf("muse_dflash_names_test: %zu names bound (3 formats x 2 heads), %zu built fresh, %zu reused: PASS\n",
              bound.size(), fresh.size(), reused.size());
  return 0;
}
