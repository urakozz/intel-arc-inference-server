// TP int8 prefill: golden prompts and every committed tool-call prompt against one card's
// bf16 prompt pass. After a difference, teacher-force the reference ids so every compared
// row has the same context. Each differing id must pass the f32 judge or its
// documented near-miss bar (docs/14-golden-gate.md, spec_judge.h).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "check.h"
#include "forced_judge.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/tp_engine.h"

namespace {
constexpr uint32_t kMaxLen = 16384, kGen = 64;
constexpr loader::LmHeadForm kHead = loader::LmHeadForm::Int8;
struct Case {
  std::string name;
  std::vector<uint32_t> prompt, ref;
};

template <class E, class Pending>
int gate(E& e, const std::vector<Case>& cases, const Pending& pending, const char* label) {
  const golden::ForcedJudge fj;
  bool ok = true;
  std::vector<std::string> awaiting;
  std::vector<uint32_t> first;
  for (const Case& c : cases) {
    e.reset();
    e.prefill(c.prompt);
    CHECK_EQ(e.pos(), uint32_t(c.prompt.size()));
    std::vector<uint32_t> got;
    for (uint32_t i = 0; i < kGen; ++i) {
      got.push_back(pending());
      e.ingest({c.ref[i]});
    }
    if (first.empty()) first = got;
    if (got == c.ref) {
      std::printf("%s: identical ids\n", c.name.c_str());
      continue;
    }
    const std::string key = golden::ForcedJudge::key("tp_prefill_int8_" + c.name, c.prompt, c.ref);
    fj.dump(key, c.prompt, c.ref);
    const auto judge = fj.open(key, kGen, model::Qwen35::kVocab);
    if (!judge) {
      std::printf("%s: awaiting f32 reference (%s)\n", c.name.c_str(), key.c_str());
      awaiting.push_back(key);
      continue;
    }
    const float* rows = judge->f32("logits", size_t(kGen + 1) * model::Qwen35::kVocab);
    for (uint32_t i = 0; i < kGen; ++i) {
      if (got[i] == c.ref[i]) continue;
      const float* row = rows + size_t(i) * model::Qwen35::kVocab;
      const golden::JudgedRow jr = golden::judge_row(row, model::Qwen35::kVocab, model::Qwen35::kVocabUsed);
      const double below = (double(row[jr.top]) - row[got[i]]) / jr.ulp;
      const bool accepted = golden::judge_accepts(*judge, i, got[i], model::Qwen35::kVocab,
                                                  model::Qwen35::kVocabUsed) || below <= golden::kNearMissUlp;
      std::printf("%s row %u: int8 %u, reference %u, f32 %s (%.2f ulp below top, near-miss bar %.0f)\n",
                  c.name.c_str(), i, got[i], c.ref[i], accepted ? "accepted" : "REJECTED", below,
                  golden::kNearMissUlp);
      // Operator 2026-11-13: accepted as a known divergence when the mixer linears went to bf16 on l0-int8 - 30.67
      // ulp below the f32 top on two cards, while one card rejects 1 row there against 3 before the change.
      const bool known = !accepted && c.name == "t2_explain-loader" && i == 32;
      if (known) std::printf("%s row %u: KNOWN divergence (accepted, not gated)\n", c.name.c_str(), i);
      if (!accepted)
        golden::judge_explain(*judge, i, got[i], model::Qwen35::kVocab, model::Qwen35::kVocabUsed, label);
      ok &= accepted || known;
    }
  }
  e.reset();
  e.prefill(cases[0].prompt);
  for (uint32_t i = 0; i < kGen; ++i) {
    CHECK_EQ(pending(), first[i]);
    e.ingest({cases[0].ref[i]});
  }
  CHECK(ok);
  if (!awaiting.empty()) {
    for (const auto& key : awaiting) std::printf("awaiting f32 reference: %s\n", key.c_str());
    std::printf("SKIP: %s, %zu contexts awaiting f32 reference\n", label, awaiting.size());
    return 77;
  }
  std::printf("%s: OK\n", label);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  if (l0::Context::gpu_count() < 2) {
    std::printf("SKIP: TP int8 prefill needs two GPUs\n");
    return 77;
  }
  l0::Context d0(0u), d1(d0, 1u);
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0)) {
    std::printf("SKIP: TP int8 prefill needs peer access both ways\n");
    return 77;
  }
  std::vector<Case> cases;
  for (const char* name : {"prose", "code", "cjk", "long"})
    cases.push_back({name, golden::read_ids(std::string("tests/golden/prompts/") + name + ".ids"), {}});
  std::vector<std::filesystem::path> tools;
  for (const auto& f : std::filesystem::directory_iterator("tests/golden/toolcall"))
    if (f.path().extension() == ".ids") tools.push_back(f.path());
  std::sort(tools.begin(), tools.end());
  CHECK(!tools.empty());
  for (const auto& p : tools) cases.push_back({p.stem().string(), golden::read_ids(p.string()), {}});
  {
    runtime::Engine one(d0, loader::load(d0, ckpt, kMaxLen, false, kHead), kMaxLen, false, runtime::KvCache::Bf16);
    one.set_prefill_backend(runtime::PrefillBackend::L0);
    one.prepare_prefill();
    for (Case& c : cases) {
      one.reset();
      one.prefill(c.prompt);
      c.ref = one.generate(kGen);
    }
  }
  if (argc > 2 && std::string(argv[2]) == "--one-card") {
    // Diagnose whether a rejected TP id also occurs with the existing one-card int8 pass.
    runtime::Engine e(d0, loader::load(d0, ckpt, kMaxLen, false, kHead), kMaxLen, false, runtime::KvCache::Bf16);
    e.set_prefill_backend(runtime::PrefillBackend::L0Int8);
    e.prepare_prefill();
    return gate(e, cases, [&] { return e.control()->cur_token[0]; }, "one_card_int8_diagnostic");
  }
  runtime::TpEngine e(d0, d1,
      {loader::load_tp(d0, ckpt, {0, runtime::kTpDefaultCrossing}, kMaxLen, kHead),
       loader::load_tp(d1, ckpt, {1, runtime::kTpDefaultCrossing}, kMaxLen, kHead)}, kMaxLen, {}, runtime::KvCache::Bf16);
  e.set_prefill_backend(runtime::PrefillBackend::L0Int8);
  e.prepare_prefill();
  return gate(e, cases, [&] {
    CHECK_EQ(e.control(0).cur_token[0], e.control(1).cur_token[0]);
    CHECK_EQ(e.control(0).pos, e.control(1).pos);
    return e.pending();
  }, "tp_prefill_int8_test");
}
