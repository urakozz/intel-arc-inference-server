// Spec 26b Task 4: every binary the LiLiCorr draft / projection lists bind (kernels::dflash::lilicorr_draft_variants - the
// names runtime/dflash/lilicorr_drafter.cc forms, every drafter format exact / int8 / int4 g64 x head int8 / bf16) is built
// - by the CMake "Spec 26b" block's B70_LILICORR list or reused from 26a's block (the GEMVs at M 8, the int8 head at M 7)
// - and the B70_LILICORR list builds nothing they do not bind. Also: kernels::dflash::kLiLiCorr - the shapes the names
// spell - is the published config (tests/model/lilicorr/config.json) beside Bolt (model::bolt()). Host only (names, no
// device).
//
//   lilicorr_names_test <lilicorr config.json> kernel_<new>... -- kernel_<reused>...
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "check.h"
#include "common/json.h"
#include "kernels/dflash_kernels.h"
#include "model/bolt.h"
#include "model/dflash.h"

int main(int argc, char** argv) {
  namespace kd = kernels::dflash;
  CHECK(argc > 2);
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  model::check_lilicorr(d);
  const kd::Shape& s = kd::kLiLiCorr;
  const model::BoltDesc& bolt = model::bolt();
  CHECK(d.hidden == s.H && d.inter == s.I && d.q_heads == s.NH && d.kv_heads == s.NKV && d.head_dim == s.HD);
  CHECK(d.block == s.M && d.taps.size() == 6 && d.window[0] == s.win && s.ring == s.win && s.ring % s.pb == 0);
  CHECK(d.conv_group == s.group && d.conv_taps == s.ctaps && d.vocab == s.vocab && bolt.vocab == s.vocab);
  CHECK(bolt.vocab_used == s.used && bolt.hidden == s.H);
  CHECK(d.cand_topk == kd::kLiliTopK && d.lil_slots() == kd::kLiliSlots && d.lil_hidden == kd::kLiliHidden &&
        d.lil_heads == kd::kLiliHeads && d.lil_rows() <= kd::kLiliRmax);
  CHECK((s.NH / s.NKV) % kd::kLiliHpw == 0);

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
      for (const std::string& v : kd::lilicorr_draft_variants(fmt, int8_head)) bound.insert(v);
  int bad = 0;
  for (const std::string& v : bound)
    if (!fresh.count(v) && !reused.count(v)) {
      std::fprintf(stderr, "the LiLiCorr lists bind %s, which no CMake block builds\n", v.c_str());
      ++bad;
    }
  for (const std::string& b : fresh)
    if (!bound.count(b)) {
      std::fprintf(stderr, "%s is built by the Spec 26b B70_LILICORR list but nothing binds it\n", b.c_str());
      ++bad;
    }
  CHECK(bad == 0);
  std::printf("lilicorr_names_test: %zu names bound (3 formats x 2 heads), %zu built fresh, %zu reused: PASS\n",
              bound.size(), fresh.size(), reused.size());
  return 0;
}
