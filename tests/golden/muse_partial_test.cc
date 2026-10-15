// muse_partial_test - spec 23c Task 6: the development mode (--layers N) on the REAL checkpoint, one card: load only
// its first N layers (default 8), ingest each golden prompt one replay at a time and then the golden's own 32 tokens
// (teacher-forced), and compare the residual tap of layers 0..N-1 with the reference's H.L<l> (muse_ref.py run on the
// WHOLE model - layer l's output does not depend on the layers after it - over the rows the golden keeps). When the
// truncated reference exists (muse_oracle.sh real-layers: <L dir>/<prompt>.golden.safetensors, the final norm and the
// head on layer N-1's residual), the head is graded too: MG3's tie-aware gate (muse_golden_test's rule) on its 32 steps.
//
// Bars per layer, over every compared row: cosine median >= 0.9998 and min >= 0.99 (PROPOSED - set from the first box
// run; printed per layer).
//
// argv: <real checkpoint> <oracle-out-muse> [N] [<oracle-out-muse-L<N>>]   (the attention form: B70_MUSE_ATTN)
// Exit 77 (SKIP) without the checkpoint or the golden set (muse_oracle.sh golden).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "runtime/control.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/muse_rig.h"

namespace {
constexpr uint32_t kGen = 32;
constexpr double kMedianBar = 0.9998, kMinBar = 0.99;   // PROPOSED
const char* const kPrompts[] = {"muse_short_prose", "muse_short_code", "muse_4k", "muse_8k", "muse_agentic"};

// The tie-aware gate of one step (muse_golden_test's rule): determined rows exact, undetermined rows in either set.
bool gate_step(const float* raw, const float* cap, uint32_t V, uint32_t got, bool& determined) {
  const golden::GoldenDecision dr = golden::golden_decision(raw, V, V), dc = golden::golden_decision(cap, V, V);
  determined = dr.determined() && dc.determined();
  return determined ? got == dc.set[0] : (dr.contains(got) || dc.contains(got));
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-muse";
  const uint32_t N = argc > 3 ? uint32_t(std::strtoul(argv[3], nullptr, 10)) : 8;
  const std::string ldir = argc > 4 ? argv[4] : gdir + "-L" + std::to_string(N);
  const std::string snap = muse_rig::muse_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s'\n", arg.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(gdir + "/" + p + ".golden.safetensors") && golden::exists(gdir + "/" + p + ".ids"))
      prompts.push_back(p);
  if (prompts.empty()) {
    std::printf("SKIP: no golden set in %s (tools/box_validate/muse_oracle.sh golden)\n", gdir.c_str());
    return 77;
  }
  const model::MuseGlimmerDesc pre = loader::muse_checkpoint_desc(snap, N);
  size_t longest = 0;
  for (const std::string& p : prompts)
    longest = std::max(longest, muse_rig::read_ids(gdir + "/" + p + ".ids", pre.vocab).size());
  muse_rig::Rig rig;
  runtime::muse::MuseOptions o;
  o.layers = N;
  o.debug_tap = true;
  o.max_len = muse_rig::len_for(longest + kGen + 1);
  muse_rig::build(rig, snap, 1, o);
  runtime::muse::MuseEngine& eng = *rig.eng;
  const model::MuseGlimmerDesc& d = eng.weights().desc;
  CHECK_EQ(d.layers, N);
  std::printf("%s: layers [0, %u) of %u (development mode), %s attention, %zu launches a token\n", snap.c_str(), N,
              eng.weights().checkpoint_layers, runtime::muse::muse_attn_name(eng.attention()), eng.launches());

  bool ok = true;
  std::vector<double> sa, sb;
  std::vector<std::vector<double>> cos(N);
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = muse_rig::read_ids(gdir + "/" + pname + ".ids", d.vocab);
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size()), NR = T + kGen;
    const int32_t* gtok = g.i32("tokens", kGen);
    std::vector<const uint16_t*> gh(N, nullptr);
    uint32_t A = 0;
    for (uint32_t l = 0; l < N; ++l) {
      const std::string n = "H.L" + std::to_string(l);
      CHECK(g.has(n));
      A = uint32_t(g.dim(n, 2, 0));
      gh[l] = g.bf16(n, size_t(A) * d.hidden);
    }
    const uint32_t arow0 = NR - A;
    auto tap = [&](uint32_t row) {
      if (row < arow0) return;
      const std::vector<uint16_t> t = eng.read_debug_resid();
      for (uint32_t l = 0; l < N; ++l)
        cos[l].push_back(golden::compare_bf16(t.data() + size_t(l) * d.hidden, gh[l] + size_t(row - arow0) * d.hidden,
                                              d.hidden, sa, sb).cos);
    };
    eng.reset();
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      tap(t);
    }
    for (uint32_t j = 0; j < kGen; ++j) {   // the whole model's continuation, teacher-forced (rows T + j)
      eng.ingest({uint32_t(gtok[j])});
      tap(T + j);
    }
    std::printf("  %s: %u prompt ids + %u teacher-forced, taps from row %u\n", pname.c_str(), T, kGen, arow0);
    // the head on layer N-1's residual, against the truncated reference
    const std::string lg = ldir + "/" + pname + ".golden.safetensors";
    if (!golden::exists(lg)) continue;
    golden::Golden h(lg);
    const uint32_t Lr = uint32_t(h.dim("logits.raw", 2, 0)), lrow0 = NR - Lr;
    const float* raw = h.f32("logits.raw", size_t(Lr) * d.vocab);
    const float* cap = h.f32("logits.cap", size_t(Lr) * d.vocab);
    const int32_t* htok = h.i32("tokens", kGen);
    eng.reset();
    eng.ingest(ids);
    uint32_t pass = 0, det = 0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      bool determined = false;
      pass += gate_step(raw + size_t(row - lrow0) * d.vocab, cap + size_t(row - lrow0) * d.vocab, d.vocab,
                        eng.control(0).cur_token[0], determined) ? 1 : 0;
      det += determined ? 1 : 0;
      eng.ingest({uint32_t(htok[j])});
    }
    std::printf("  %s head (the truncated reference, %s): %u / %u steps in the gate (%u determined)\n", pname.c_str(),
                ldir.c_str(), pass, kGen, det);
    ok = ok && pass == kGen;
  }
  std::printf("per layer (cosine against H.L<l>, bars median %.4f min %.2f PROPOSED):\n", kMedianBar, kMinBar);
  for (uint32_t l = 0; l < N; ++l) {
    std::vector<double> v = cos[l];
    std::sort(v.begin(), v.end());
    const double med = v.empty() ? 0.0 : v[v.size() / 2], mn = v.empty() ? 0.0 : v.front();
    const bool lok = med >= kMedianBar && mn >= kMinBar;
    std::printf("  L%-2u %s: median %.6f min %.6f over %zu rows %s\n", l, d.is_full(l) ? "full   " : "sliding", med, mn,
                v.size(), lok ? "" : "BELOW");
    ok = ok && lok;
  }
  CHECK(ok);
  std::puts("muse_partial_test OK");
  return 0;
}
