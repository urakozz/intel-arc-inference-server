// muse_golden_test - spec 23 §4.2 MG3 (decode): Muse-Glimmer's engine against tools/oracle/muse_ref.py (spec 23a:
// transformers 5.19.0's own model, layer-streamed) - each golden prompt x 32 greedy tokens, teacher-forced - with the
// per-layer residual diagnostic. kolibri_golden_test's arrangement, Muse's semantics.
//
// **The golden files** (muse_ref.py `run`; tools/box_validate/muse_oracle.sh synth | golden):
//   <dir>/<prompt>.ids                   the prompt (Muse's tokenisation, no special tokens)
//   <dir>/<prompt>.golden.safetensors
//     logits.raw   F32  [Lr][202048]     lm_head's bf16 row widened (no multiplier, no cap); rows T + 32 - Lr ..
//     logits.cap   F32  [Lr][202048]     the model's logits: the four-rounding softcap of the raw row
//     tokens       I32  [32]             greedy: argmax over the CAPPED row (generate(do_sample=False)'s rule)
//     H.L<l>       BF16 [A][6656]        the residual after layer l, rows T + 32 - A ..
// The prompts read: muse_short_prose (must exist), muse_short_code, muse_4k, muse_8k, muse_32k, muse_agentic - each
// whose two files exist.
//
// **The gate** (docs/14's tie-aware rule plus spec 23 §1.1's capped-tie rule): ingest the ids one replay at a time,
// then for j = 0..31 the engine's id (the device argmax over its fp32 RAW row) against the golden decision row
// T - 1 + j: a row whose capped maximum is shared by >= 2 ids (the bf16 softcap made them equal) or whose raw maximum
// is shared is UNDETERMINED - the engine's id must be in the union of the two argmax sets; every other row is
// determined - one id, exact (the softcap is monotone: a unique capped maximum is the raw maximum). Then the GOLDEN
// token is fed (teacher forcing).
//
// **The diagnostic** (printed, not gated): per layer the minimum cosine of the engine's residual tap against H.L<l>
// over the rows the golden keeps; the raw logits' worst cosine at the decision rows.
//
// argv: <checkpoint> <golden dir> [bf16] [pp2 | pp2:<split>] [copy | peer]   (the attention form: B70_MUSE_ATTN)
// Exit 77 (SKIP) when the checkpoint or the golden set is absent, or two GPUs are asked for and not there.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "runtime/control.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/muse_rig.h"

namespace {
constexpr uint32_t kGen = 32;
const char* const kPrompts[] = {"muse_short_prose", "muse_short_code", "muse_4k", "muse_8k", "muse_32k", "muse_agentic"};
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "";
  const std::string gdir = argc > 2 ? argv[2] : "oracle-out-muse";
  runtime::muse::MuseOptions o;
  o.debug_tap = true;
  uint32_t devices = 1;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "bf16") {
      o.int8_head = false;
    } else if (a == "int8") {
      o.int8_head = true;
    } else if (a == "pp2") {
      devices = 2;
    } else if (a.rfind("pp2:", 0) == 0) {
      devices = 2;
      o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10));
    } else if (a == "copy" || a == "peer") {
      o.handoff = a == "peer" ? runtime::PpHandoff::Peer : runtime::PpHandoff::Copy;
    } else {
      std::fprintf(stderr, "muse_golden_test: unknown flag %s (bf16, int8, pp2[:split], copy, peer)\n", a.c_str());
      return 2;
    }
  }
  const std::string snap = muse_rig::muse_snapshot(arg);
  if (snap.empty()) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s' (muse_oracle.sh synth-ckpt; the real one: hf download)\n",
                arg.c_str());
    return 77;
  }
  std::vector<std::string> prompts;
  for (const char* p : kPrompts)
    if (golden::exists(gdir + "/" + p + ".golden.safetensors") && golden::exists(gdir + "/" + p + ".ids"))
      prompts.push_back(p);
  if (prompts.empty() || prompts[0] != "muse_short_prose") {
    std::printf("SKIP: %s/muse_short_prose.{ids,golden.safetensors} is absent (tools/box_validate/muse_oracle.sh)\n",
                gdir.c_str());
    return 77;
  }
  if (devices == 2 && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: pp2 needs two GPUs, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::MuseGlimmerDesc pre = loader::muse_checkpoint_desc(snap);
  size_t longest = 0;
  for (const std::string& p : prompts)
    longest = std::max(longest, muse_rig::read_ids(gdir + "/" + p + ".ids", pre.vocab).size());
  o.max_len = muse_rig::len_for(longest + kGen + 1);
  muse_rig::Rig rig;
  muse_rig::build(rig, snap, devices, o);
  runtime::muse::MuseEngine& eng = *rig.eng;
  const model::MuseGlimmerDesc& d = eng.weights().desc;
  std::printf("%s: %u layers, lm_head %s, %s attention (B70_MUSE_ATTN), %u device(s)%s, %zu launches a token, max_len "
              "%u\n", snap.c_str(), d.layers, o.int8_head ? "int8" : "bf16",
              runtime::muse::muse_attn_name(eng.attention()), eng.devices(),
              eng.devices() == 2 ? (std::string(", split ") + std::to_string(eng.split()) + " " +
                                    runtime::pp_handoff_name(eng.handoff())).c_str() : "",
              eng.launches(), eng.max_len());

  bool gate_ok = true;
  std::vector<double> sa, sb;
  for (const std::string& pname : prompts) {
    const std::vector<uint32_t> ids = muse_rig::read_ids(gdir + "/" + pname + ".ids", d.vocab);
    CHECK(!ids.empty());
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size()), N = T + kGen;
    const uint32_t Lr = uint32_t(g.dim("logits.raw", 2, 0));
    CHECK_EQ(g.dim("logits.raw", 2, 1), uint64_t(d.vocab));
    CHECK(Lr >= kGen + 1 && Lr <= N);
    const uint32_t lrow0 = N - Lr;   // the first row the logits keep
    const float* graw = g.f32("logits.raw", size_t(Lr) * d.vocab);
    const float* gcap = g.f32("logits.cap", size_t(Lr) * d.vocab);
    const int32_t* gtok = g.i32("tokens", kGen);
    // the residual taps the golden keeps (rows N - A ..), for the layers the engine has
    std::vector<const uint16_t*> gh(d.layers, nullptr);
    uint32_t A = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string n = "H.L" + std::to_string(l);
      if (!g.has(n)) continue;
      A = uint32_t(g.dim(n, 2, 0));
      CHECK_EQ(g.dim(n, 2, 1), uint64_t(d.hidden));
      gh[l] = g.bf16(n, size_t(A) * d.hidden);
    }
    const uint32_t arow0 = N - A;
    std::printf("\n================ %s: %u prompt ids, %u generated (logits from row %u, taps from row %u) ======\n",
                pname.c_str(), T, kGen, lrow0, arow0);
    std::vector<double> tap_min(d.layers, 1.0);
    auto check_tap = [&](uint32_t row) {
      if (row < arow0 || A == 0) return;
      const std::vector<uint16_t> tap = eng.read_debug_resid();
      for (uint32_t l = 0; l < d.layers; ++l)
        if (gh[l])
          tap_min[l] = std::min(tap_min[l], golden::compare_bf16(tap.data() + size_t(l) * d.hidden,
                                                                 gh[l] + size_t(row - arow0) * d.hidden, d.hidden, sa, sb).cos);
    };
    eng.reset();
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      check_tap(t);
    }
    uint32_t det = 0, det_ok = 0, und = 0, und_ok = 0;
    int first_bad = -1;
    double logit_min = 1.0;
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t row = T - 1 + j;
      const uint32_t got = eng.control(eng.devices() - 1).cur_token[0];
      const std::vector<float> lg = eng.read_logits();
      const float* rraw = graw + size_t(row - lrow0) * d.vocab;
      const float* rcap = gcap + size_t(row - lrow0) * d.vocab;
      logit_min = std::min(logit_min, golden::compare_f32(lg.data(), rraw, d.vocab, sa, sb).cos);
      const golden::GoldenDecision dr = golden::golden_decision(rraw, d.vocab, d.vocab);
      const golden::GoldenDecision dc = golden::golden_decision(rcap, d.vocab, d.vocab);
      if (dr.determined() && dc.determined()) {
        ++det;
        CHECK_EQ(dc.set[0], dr.set[0]);   // the softcap is monotone
        if (got == dc.set[0]) ++det_ok;
        else if (first_bad < 0) first_bad = int(j);
      } else {
        ++und;
        if (dr.contains(got) || dc.contains(got)) ++und_ok;
        else if (first_bad < 0) first_bad = int(j);
      }
      if (got != uint32_t(gtok[j])) {
        std::printf("  step %u: engine %u, golden %u (%s)\n", j, got, uint32_t(gtok[j]),
                    dr.determined() && dc.determined() ? "determined" : "undetermined: a raw or a capped tie");
        golden::print_top5("engine", lg.data(), d.vocab, d.vocab);
        golden::print_top5("golden", rraw, d.vocab, d.vocab);
      }
      eng.ingest({uint32_t(gtok[j])});   // teacher-forced: row T + j
      check_tap(T + j);
    }
    const bool ok = det_ok == det && und_ok == und;
    gate_ok = gate_ok && ok;
    std::printf("  gate: %u / %u determined rows exact, %u / %u undetermined rows in the argmax sets%s; raw logits worst "
                "cosine %.6f\n", det_ok, det, und_ok, und,
                ok ? "" : (" - FIRST BAD STEP " + std::to_string(first_bad)).c_str(), logit_min);
    std::printf("  tap (min cosine over rows %u..%u):", arow0, N - 1);
    for (uint32_t l = 0; l < d.layers; ++l)
      if (gh[l]) std::printf(" %u:%.5f", l, tap_min[l]);
    std::printf("\n");
  }
  CHECK(gate_ok);
  std::puts("muse_golden_test OK");
  return 0;
}
