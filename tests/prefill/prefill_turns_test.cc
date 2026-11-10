// prefill_turns_test - 3l (2): short turns deep in a session, judged by the f32 judge.
//
//   prefill_turns_test <snapshot> <bf16|int8> [depths=16000,32000]
//
// A session: tests/golden/prompts/long32k.ids to `depth` keys (the engine's own chunks), then three turns of
// 16, 64 and 128 ids (the prompt's next ids) as three Engine::prefill calls - each a chunk of its own size, the
// short chunks deep in the cache that the split-K flash attention takes (runtime/prefill/attn.h
// flash_split_span) - then 32 greedy ids. Run twice per depth on one engine: the split OFF (today's one-pass
// launch) and the engine's rule (Auto), the test seam set_flash_split_for_test. The split changes the softmax's
// association, so the two runs are judged, not compared bitwise:
//   B70_DUMP_FORCED=<dir>   dumps each run's context (tests/golden/forced_judge.h: prompt = the session's ids,
//                           forced = its 32 ids; keys turns_qwen3.8_<kv>_d<depth>_<off|split>_<hash>) for
//                           tools/box_validate/f32_refs.sh forced-qwen (QWEN_KEYS_RE=^turns_)
//   B70_FORCED_JUDGE=<dir>  judges both runs: rows the judge rejects (row j decides forced id j, the one-bf16-ulp
//                           near-tie rule), each with its margin, and the logits cosine to the judge (min /
//                           median). PASS per depth: the split run is rejected at no more rows than the OFF run,
//                           or every split rejection is within kNearMissUlp (8) bf16 ulp and its cosine distances
//                           are within kCosSlack (2) x the OFF run's (forced_judge.h's near-miss numbers).
// Without a judge dir the runs are printed (ids, the first difference, the logits cosine split vs OFF) and the
// test SKIPs (77) - it never passes unjudged. The rule must split every turn at these depths (checked).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "forced_judge.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/attn.h"

namespace {
using model::Qwen35;
using runtime::prefill::FlashSplit;

constexpr uint32_t kGen = 32;
constexpr uint32_t kTurns[] = {16, 64, 128};

struct Run {
  std::vector<uint32_t> prompt, ids;
  std::vector<std::vector<float>> logits;   // row j: the decision of ids[j]
};

Run session(runtime::Engine& e, l0::Context& ctx, const std::vector<uint32_t>& base, uint32_t depth, FlashSplit m) {
  Run r;
  e.reset();
  runtime::prefill::set_flash_split_for_test(m);
  r.prompt.assign(base.begin(), base.begin() + depth);
  e.prefill(r.prompt);
  uint32_t at = depth;
  for (uint32_t t : kTurns) {
    if (m == FlashSplit::Auto) CHECK(runtime::prefill::flash_split_span(*e.model().desc, at, t) != 0);
    const std::vector<uint32_t> turn(base.begin() + at, base.begin() + at + t);
    e.prefill(turn, t);
    r.prompt.insert(r.prompt.end(), turn.begin(), turn.end());
    at += t;
  }
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto read = [&](const l0::Mem& m) {
    std::vector<float> v(Qwen35::kVocab);
    imm.copy(v.data(), m.ptr(), v.size() * sizeof(float));
    r.logits.push_back(std::move(v));
  };
  read(e.prefill_scratch()->logits);
  const auto* ctl = e.buffers().control.as<runtime::Control>();
  for (uint32_t j = 0; j < kGen; ++j) {
    const uint32_t id = ctl->cur_token[0];
    r.ids.push_back(id);
    if (j + 1 == kGen) break;
    CHECK_EQ(e.generate(1)[0], id);
    read(e.buffers().logits);
  }
  runtime::prefill::set_flash_split_for_test(FlashSplit::Auto);
  return r;
}

double cos_of(const std::vector<float>& a, const std::vector<float>& b) {
  double ab = 0, aa = 0, bb = 0;
  for (uint32_t i = 0; i < Qwen35::kVocabUsed; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

struct Judged {
  uint32_t misses = 0;
  double worst_ulp = 0;
  golden::CosRows cos;
};

}  // namespace

int main(int argc, char** argv) {
  CHECK(argc >= 3 && argc <= 4);
  const std::string kv_name = argv[2];
  const auto kv = runtime::parse_kv_cache(argv[2]);
  std::vector<uint32_t> depths = {16000, 32000};
  if (argc > 3) {
    depths.clear();
    std::stringstream ss(argv[3]);
    for (std::string t; std::getline(ss, t, ',');) depths.push_back(uint32_t(std::stoul(t)));
  }
  std::ifstream input("tests/golden/prompts/long32k.ids");
  std::vector<uint32_t> base;
  for (uint32_t id; input >> id;) base.push_back(id);
  CHECK(!base.empty());
  uint32_t need = 0;
  for (uint32_t d : depths) need = std::max(need, d + 16 + 64 + 128 + kGen);
  CHECK(need <= base.size() + kGen);
  const uint32_t max_len = (need + 255u) & ~255u;
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, argv[1], max_len), max_len, false, kv);
  const golden::ForcedJudge fj;
  const uint32_t V = Qwen35::kVocab, used = Qwen35::kVocabUsed;
  bool judged_all = true, pass = true;
  for (uint32_t depth : depths) {
    const Run off = session(e, ctx, base, depth, FlashSplit::Off);
    const Run split = session(e, ctx, base, depth, FlashSplit::Auto);
    uint32_t first = kGen;
    for (uint32_t j = 0; j < kGen; ++j)
      if (off.ids[j] != split.ids[j]) { first = j; break; }
    std::printf("depth %u kv %s: turns 16 / 64 / 128 then %u ids; split vs off: %s", depth, kv_name.c_str(), kGen,
                first == kGen ? "ids identical" : "first different id at row ");
    if (first < kGen) std::printf("%u (%u vs %u)", first, split.ids[first], off.ids[first]);
    std::printf("; logits cos split vs off row 0 %.9f, worst of the common rows %.9f\n", cos_of(split.logits[0], off.logits[0]),
                [&] {
                  double w = 1;
                  for (uint32_t j = 0; j <= std::min(first, kGen - 1); ++j) w = std::min(w, cos_of(split.logits[j], off.logits[j]));
                  return w;
                }());
    Judged res[2];
    const Run* runs[2] = {&off, &split};
    const char* names[2] = {"off", "split"};
    for (int a = 0; a < 2; ++a) {
      const Run& r = *runs[a];
      const std::string k = golden::ForcedJudge::key(
          "turns_qwen3.8_" + kv_name + "_d" + std::to_string(depth) + "_" + names[a], r.prompt, r.ids);
      fj.dump(k, r.prompt, r.ids);
      const auto j = fj.open(k, kGen, V);
      if (!j) {
        judged_all = false;
        continue;
      }
      const float* jl = j->f32("logits", size_t(kGen + 1) * V);
      for (uint32_t row = 0; row < kGen; ++row) {
        const float* jr = jl + size_t(row) * V;
        if (!golden::row_accepts(jr, r.ids[row], V, used)) {
          ++res[a].misses;
          const golden::JudgedRow t = golden::judge_row(jr, V, used);
          res[a].worst_ulp = std::max(res[a].worst_ulp, (double(jr[t.top]) - double(jr[r.ids[row]])) / t.ulp);
          golden::judge_explain(*j, row, r.ids[row], V, used, names[a]);
        }
        res[a].cos.c.push_back(golden::row_cos(jr, r.logits[row].data(), used));
      }
      std::printf("  %-5s run: the f32 judge rejects %u of %u ids; logits cos to the judge min / median %s\n", names[a],
                  res[a].misses, kGen, res[a].cos.str().c_str());
    }
    if (!judged_all) continue;
    const double rmin = golden::cos_dist_ratio(res[1].cos.min(), res[0].cos.min());
    const double rmed = golden::cos_dist_ratio(res[1].cos.median(), res[0].cos.median());
    const bool by_count = res[1].misses <= res[0].misses;
    const bool near = res[1].worst_ulp <= golden::kNearMissUlp && rmin <= golden::kCosSlack && rmed <= golden::kCosSlack;
    const bool ok = by_count || near;
    std::printf("  depth %u: split %u vs off %u rejections, cosine distance x%.2f (min) / x%.2f (median) of off's - %s\n",
                depth, res[1].misses, res[0].misses, rmin, rmed,
                ok ? (by_count ? "PASS (no more rejections)" : "PASS by the near-miss rule") : "** FAIL **");
    pass = pass && ok;
  }
  if (!judged_all) {
    std::puts("prefill_turns_test: SKIP - not every run has its f32 judge (B70_DUMP_FORCED, then f32_refs.sh forced-qwen)");
    return 77;
  }
  std::puts(pass ? "prefill_turns_test: PASS" : "prefill_turns_test: FAIL");
  return pass ? 0 : 1;
}
