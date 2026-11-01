// Spec 19c on the card, through EngineAdapter - exactly what b70-serve's loop calls, no HTTP: the engine with the
// DFlash drafter (int8, beside the gate checkpoint's int8 head, bf16 KV).
//
//   D2   greedy, 256 ids, K = 7, 3 and the AdaptiveK policy over MtpCost::dflash (the server's --spec dflash
//        default), against the plain step() loop of the same engine, on the three golden prompts and the 36 A4
//        tool-call prompts. Spec 28 N6 (plan 28b): the M = k + 1 rows run gemm_sm and are no longer bitwise
//        M = 1, so the bar is identity except a first divergence the golden tie rule classes as a near-tie
//        (tests/server/greedy_tie.h): zero non-tie divergences, the near-ties counted (before 28b: identity). Acceptance and ids per iteration printed per
//        prompt (19a's teacher-forced E_K is the expectation: A4 ~7.2 at K = 7, code ~6.2, prose ~2.2).
//   RF1  a run truncated inside an accepted burst (truncate_to) and then continued equals the plain run.
//   S    a seeded sampled run (T 0.7, top-k 20, top-p 0.95) twice from fresh adapters: ids bitwise (the drafter's
//        Gumbel key and every acceptance draw come from the request's RNG); its acceptance printed (D4 itself is
//        the host test dflash_accept_test).
//
// usage: dflash_gpu_test <target snapshot> <drafter>   (cwd = the source tree). SKIP 77 without either.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "greedy_tie.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/engine.h"
#include "server/adaptive_k.h"

namespace {
using golden::read_ids;
constexpr uint32_t kMaxLen = 16384, kGen = 256, kK = 7;

struct Prompt {
  std::string name;
  std::vector<uint32_t> ids;
};
std::vector<Prompt> prompts() {
  std::vector<Prompt> out;
  for (const char* n : {"code", "prose", "cjk"}) out.push_back({n, read_ids(std::string("tests/golden/prompts/") + n + ".ids")});
  std::ifstream f("tests/golden/toolcall/manifest.json");
  CHECK(f.good());
  nlohmann::json manifest;
  f >> manifest;
  for (const auto& e : manifest) {
    const std::string n = e.at("name").get<std::string>();
    out.push_back({n, read_ids("tests/golden/toolcall/" + n + ".ids")});
  }
  return out;
}

struct Gen {
  std::vector<uint32_t> ids;
  uint64_t iters = 0, drafted = 0, accepted = 0;
};

// server.cc's loop minus the text: prefill, then step() (k_fixed == 0 and no policy) or step_many(K) until n ids,
// the last burst truncated to exactly n. `policy`: K per iteration from server::AdaptiveK (the --spec dflash default).
// `steps` (the plain run only): after every step() the row that decided the next id; steps[i] belongs to ids[i].
Gen generate(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s,
             uint32_t k_fixed, bool policy, std::vector<greedy_tie::RefStep>* steps = nullptr,
             runtime::Engine* eng = nullptr) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  if (steps) {
    steps->clear();
    steps->push_back({UINT32_MAX, UINT32_MAX, 1e30});
  }
  std::vector<float> row(steps ? model::Qwen35::kVocab : 0);
  std::unique_ptr<server::AdaptiveK> ak;
  if (policy) {
    server::AdaptiveKOptions o;
    o.max_k = kK;
    o.cost = server::MtpCost::dflash(true, true);
    ak = std::make_unique<server::AdaptiveK>(o);
  }
  while (g.ids.size() < n) {
    if (!policy && k_fixed == 0) {
      g.ids.push_back(a.step(s));
      if (steps) {
        l0::CmdList imm = l0::CmdList::immediate(eng->context());
        imm.copy(row.data(), eng->buffers().logits.ptr(), row.size() * sizeof(float));
        steps->push_back(greedy_tie::ref_step(row.data(), model::Qwen35::kVocabUsed));
      }
      continue;
    }
    const uint32_t k = ak ? ak->next() : k_fixed;
    const std::vector<uint32_t> burst = a.step_many(s, k);
    if (ak) ak->observe(k, std::min<uint32_t>(k, uint32_t(burst.size() - 1)));
    ++g.iters;
    g.drafted += k;
    g.accepted += burst.size() - 1;
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap, dsnap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    dsnap = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
  loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
  const uint32_t head_rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(w), head_rows, kMaxLen, kK);
  runtime::Engine e(ctx, std::move(model), kMaxLen, /*debug_resid=*/false, runtime::KvCache::Bf16, &d);
  d.attach(e, false);
  EngineAdapter a(e, model::Qwen35::kVocabUsed, kK, /*lookup=*/false, /*dflash=*/true);
  bool ok = true;
  server::Sampling greedy;   // greedy = true by default

  // --- D2 -------------------------------------------------------------------------------------------------
  uint64_t tot_iters[3] = {}, tot_ids[3] = {}, fails = 0, ties = 0;
  for (const Prompt& p : prompts()) {
    std::vector<greedy_tie::RefStep> steps;
    const Gen plain = generate(a, p.ids, kGen, greedy, 0, false, &steps, &e);
    const char* arm[3] = {"K=7", "K=3", "auto"};
    for (int i = 0; i < 3; ++i) {
      const Gen g = generate(a, p.ids, kGen, greedy, i == 0 ? kK : i == 1 ? 3 : 0, i == 2);
      const greedy_tie::Result tr = greedy_tie::classify(plain.ids, steps, g.ids);
      char what[96];
      if (tr.v == greedy_tie::Verdict::Same) std::snprintf(what, sizeof what, "identical");
      else if (tr.v == greedy_tie::Verdict::Tie)
        std::snprintf(what, sizeof what, "near-tie at %zu (%.3f bf16 ulp)", tr.at, tr.margin_ulps);
      else
        std::snprintf(what, sizeof what, "DIFFERS at %zu (candidate rank %u, %.3f ulp below top-1; top-2 %.3f)", tr.at,
                      tr.cand_rank, tr.cand_gap, tr.margin_ulps);
      std::printf("D2 %-28s %-4s %s, %llu iterations, %.2f ids an iteration, acceptance %.3f\n", p.name.c_str(),
                  arm[i], what, (unsigned long long)g.iters, double(kGen) / double(std::max<uint64_t>(1, g.iters)),
                  g.drafted ? double(g.accepted) / double(g.drafted) : 0.0);
      fails += tr.v == greedy_tie::Verdict::Diverged;
      ties += tr.v == greedy_tie::Verdict::Tie;
      tot_iters[i] += g.iters;
      tot_ids[i] += kGen;
    }
  }
  std::printf("D2: %llu non-tie divergences, %llu near-ties; ids an iteration K=7 %.2f, K=3 %.2f, auto %.2f\n",
              (unsigned long long)fails, (unsigned long long)ties,
              double(tot_ids[0]) / double(tot_iters[0]), double(tot_ids[1]) / double(tot_iters[1]),
              double(tot_ids[2]) / double(tot_iters[2]));
  ok &= fails == 0;

  // --- RF1: truncate inside a burst, then continue -----------------------------------------------------------
  {
    const std::vector<uint32_t> prompt = read_ids("tests/golden/prompts/code.ids");
    std::vector<greedy_tie::RefStep> steps;
    const Gen plain = generate(a, prompt, 96, greedy, 0, false, &steps, &e);
    a.reset();
    a.prefill(prompt);
    std::vector<uint32_t> got;
    bool truncated = false;
    while (got.size() < 96) {
      std::vector<uint32_t> burst = a.step_many(greedy, kK);
      if (!truncated && burst.size() >= 3) {   // keep two of the burst
        a.truncate_to(a.pos() - uint32_t(burst.size() - 2));
        burst.resize(2);
        truncated = true;
      }
      got.insert(got.end(), burst.begin(), burst.end());
    }
    got.resize(96);
    const greedy_tie::Result tr = greedy_tie::classify(plain.ids, steps, got);
    const bool rf1 = truncated && tr.v != greedy_tie::Verdict::Diverged;
    std::printf("RF1: a burst truncated to 2 ids, then continued: %s\n",
                !truncated                             ? "no burst of 3 to truncate"
                : tr.v == greedy_tie::Verdict::Same    ? "equals the plain run"
                : tr.v == greedy_tie::Verdict::Tie     ? "equals the plain run up to a near-tie"
                                                       : "DIFFERS");
    ok &= rf1;
  }

  // --- S: seeded sampling, reproducible ----------------------------------------------------------------------
  {
    server::Sampling s;
    s.greedy = false;
    s.temperature = 0.7f;
    s.top_k = 20;
    s.top_p = 0.95f;
    s.seed = 42;
    s.has_seed = true;
    const std::vector<uint32_t> prompt = read_ids("tests/golden/prompts/prose.ids");
    Gen g[2];
    for (int r = 0; r < 2; ++r) {
      EngineAdapter fresh(e, model::Qwen35::kVocabUsed, kK, false, true);   // its RNG seeds from the request
      g[r] = generate(fresh, prompt, 128, s, kK, false);
    }
    const bool same = g[0].ids == g[1].ids;
    std::printf("S: seeded sampled K=7 run twice: %s; acceptance %.3f, %.2f ids an iteration\n",
                same ? "bitwise" : "DIFFERS", g[0].drafted ? double(g[0].accepted) / double(g[0].drafted) : 0.0,
                double(128) / double(std::max<uint64_t>(1, g[0].iters)));
    ok &= same;
  }

  std::printf("dflash_gpu_test: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
