// Open-work 27 on the card, through EngineAdapter - what b70-serve --spec auto / mtp's loop calls with prompt lookup
// inside MTP, no HTTP: the engine with the MTP head (the int8 head, bf16 KV), server::LookupInDFlash with no window
// over a server::PromptLookup exactly as Server::generate drives them, and the MTP iterations' per-step draft choice
// (server::StepK over MtpCost::int8_head, step_chosen) as the greedy server runs them.
//
//   D2   greedy, 256 ids, against the plain step() loop of the same engine on the three golden prompts, the 36 A4
//        tool-call prompts and the edit / copy set's 8k / 32k prompts (tests/golden/edit-copy: long repeats, where
//        lookup enters). The bar is every speculative arm's (dflash_lookup_gpu_test's): identity, or - from the first
//        divergence - the f32 judge rejects no more of its ids than of the plain run's (spec 28 N6,
//        tests/server/spec_judge.h), the golden tie rule without a judge; the tie census printed. Per prompt: lookup
//        iterations / ids in lookup, MTP iterations, ids an iteration; per edit / copy prompt also plain --spec mtp's
//        (per-step choice, no lookup) ids an iteration and the host-timed ms an id of plain / mtp / mtp-lookup
//        (busy-box timing, informational: the server A/B is the measurement).
//   H    the head stays warm: after an mtp-lookup run that used lookup (reemit-8k, 256 ids) the head's KV rows of the
//        generated positions against plain --spec mtp's on the same ids - a lookup verify rewrites them as an MTP
//        verify does, so no row is stale (min cosine >= 0.99; the bitwise share printed: the two runs verify at other
//        M, and an M = 1 row is not bitwise an M >= 2 row). A truncate_to inside a lookup burst, then continued: the
//        ids equal the untruncated run's.
//   S    a seeded sampled mtp-lookup run (T 0.7, top-k 20, top-p 0.95; its MTP iterations step_many at AdaptiveK's K,
//        as the server's) twice from fresh adapters: ids bitwise; its lookup share printed.
//
// usage: mtp_lookup_gpu_test <target snapshot>   (cwd = the source tree). SKIP 77 without it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/serve_adapters.h"
#include "golden_common.h"
#include "greedy_tie.h"
#include "spec_judge.h"
#include "spec_plain_row.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "runtime/engine.h"
#include "server/adaptive_k.h"
#include "server/lookup_in_dflash.h"
#include "server/prompt_lookup.h"
#include "server/step_k.h"

namespace {
using golden::read_ids;
constexpr uint32_t kMaxLen = 40960, kGen = 256, kK = 3;

struct Prompt {
  std::string name;
  std::vector<uint32_t> ids;
  bool far = false;   // an edit / copy prompt
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
  for (const char* d : {"8k", "32k"})
    for (const char* k : {"reemit", "rename", "quote"}) {
      const std::string n = std::string(k) + "-" + d;
      out.push_back({n, read_ids("tests/golden/edit-copy/" + n + ".ids"), true});
    }
  return out;
}

struct Gen {
  std::vector<uint32_t> ids;
  uint64_t iters = 0, lookup_iters = 0, lookup_ids = 0;
  double ms = 0;
};

server::AdaptiveKOptions ak_options() {
  server::AdaptiveKOptions o;
  o.max_k = kK;
  o.cost = server::MtpCost::int8_head();
  return o;
}
server::StepKOptions sk_options() {
  server::StepKOptions o;
  o.max_k = kK;
  o.cost = server::MtpCost::int8_head();
  return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// The plain step() loop (mtp = false) or plain --spec mtp (the per-step choice, step_chosen).
// `steps` (the plain loop only): the row that decided each id (greedy_tie.h; steps[0] the prefill's placeholder).
Gen generate_plain(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s,
                   bool mtp, std::vector<greedy_tie::RefStep>* steps = nullptr, runtime::Engine* eng = nullptr) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  if (steps) {
    steps->clear();
    steps->push_back({UINT32_MAX, UINT32_MAX, 1e30});
  }
  std::vector<float> row(steps ? model::Qwen35::kVocab : 0);
  server::DraftCalib calib;
  server::StepK sk(sk_options(), calib);
  const auto t0 = std::chrono::steady_clock::now();
  while (g.ids.size() < n) {
    if (!mtp) {
      g.ids.push_back(a.step(s));
      if (steps) {
        l0::CmdList imm = l0::CmdList::immediate(eng->context());
        imm.copy(row.data(), eng->buffers().logits.ptr(), row.size() * sizeof(float));
        steps->push_back(greedy_tie::ref_step(row.data(), model::Qwen35::kVocabUsed));
      }
      continue;
    }
    const std::vector<uint32_t> burst =
        a.step_chosen(s, sk.begin(), [&](uint32_t j, float p) { return sk.choose(j, p); });
    sk.observe(uint32_t(burst.size() - 1));
    ++g.iters;
    const size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  a.flush_commit();
  g.ms = ms_since(t0);
  return g;
}

// Server::generate's loop with lookup inside MTP, minus the text (open-work 28: with the engine's wide list, the long
// runs' 16-row verifies and the head-confirmed entries, through step_drafts_or_confirmed as the server): greedy requests' MTP iterations by the per-step
// choice, sampled ones by AdaptiveK. `cut`: the first lookup burst of >= 3 ids is truncated to 2.
Gen generate_lookup(EngineAdapter& a, const std::vector<uint32_t>& prompt, uint32_t n, const server::Sampling& s,
                    bool cut = false, bool* did_cut = nullptr) {
  a.reset();
  a.prefill(prompt);
  Gen g;
  server::DraftCalib calib;
  server::StepK sk(sk_options(), calib);
  server::AdaptiveK ak(ak_options());
  server::PromptLookup m;
  m.append(prompt);
  server::LookupInDFlashOptions lo;
  lo.window = 0;
  lo.max_k = kK;
  lo.wide_k = a.lookup_wide_k();
  server::LookupInDFlash lid(lo);
  const auto t0 = std::chrono::steady_clock::now();
  while (g.ids.size() < n) {
    const uint32_t k = s.greedy ? sk.begin() : ak.next();
    uint32_t proposed = 0;
    server::EngineIface::DraftChoice choose;
    if (s.greedy) choose = [&](uint32_t j, float p) { return sk.choose(j, p); };
    std::vector<uint32_t> cand;
    server::EngineIface::LookupConfirm confirm;
    confirm.candidate = [&]() {
      cand = lid.candidate(m);
      return cand;
    };
    confirm.confirmed = [&]() {
      lid.confirmed();
      proposed = uint32_t(cand.size());
    };
    std::vector<uint32_t> burst = a.step_drafts_or_confirmed(
        s, k,
        [&](uint32_t x) {
          m.append(x);
          const std::vector<uint32_t> d = lid.propose(m);
          proposed = uint32_t(d.size());
          return d;
        },
        choose, confirm);
    ++g.iters;
    const uint32_t kept = uint32_t(burst.size() - 1);
    if (proposed > 0) {
      lid.observe(proposed, kept);
      ++g.lookup_iters;
    } else if (s.greedy) {
      sk.observe(kept);
    } else {
      ak.observe(k, std::min(k, kept));
    }
    size_t keep = std::min<size_t>(burst.size(), n - g.ids.size());
    if (cut && proposed > 0 && did_cut && !*did_cut && keep >= 3) {
      keep = 2;
      *did_cut = true;
    }
    if (proposed > 0) g.lookup_ids += keep;
    g.ids.insert(g.ids.end(), burst.begin(), burst.begin() + keep);
    // the matcher holds what the session holds: burst[0] went in through the proposer, the dropped ids never do
    for (size_t i = 1; i < keep; ++i) m.append(burst[i]);
    if (keep < burst.size()) a.truncate_to(a.pos() - uint32_t(burst.size() - keep));
  }
  a.flush_commit();
  g.ms = ms_since(t0);
  return g;
}

float bf16(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// The head's K rows of positions [b, e) (bf16 KV).
std::vector<uint8_t> head_k_rows(EngineAdapter& a, runtime::Engine& e, uint32_t b, uint32_t end) {
  const runtime::MtpBuffers& mb = *e.mtp_buffers();
  const size_t rb = mb.kv_lay.row_bytes();
  std::vector<uint8_t> r(size_t(end - b) * rb);
  a.imm.copy(r.data(), static_cast<const uint8_t*>(mb.kv_k.ptr()) + size_t(b) * rb, r.size());
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snap, kMaxLen, /*mtp=*/true, loader::LmHeadForm::Int8), kMaxLen,
                    /*debug_resid=*/false, runtime::KvCache::Bf16);
  EngineAdapter mtp(e, model::Qwen35::kVocabUsed, kK);
  CHECK(mtp.mtp_lookup() && mtp.chooses_drafts());
  std::printf("%s\nK <= %u, max_len %u\n", e.memory_line().c_str(), kK, kMaxLen);
  bool ok = true;
  server::Sampling greedy;

  // --- D2 -------------------------------------------------------------------------------------------------
  uint64_t fails = 0, far_lookup_iters = 0, tot_iters = 0, tot_lookup_ids = 0, tot_ids = 0, ties = 0, beyond = 0,
           judged = 0;
  const golden::ForcedJudge fj;
  for (const Prompt& p : prompts()) {
    std::vector<greedy_tie::RefStep> steps;
    const Gen plain = generate_plain(mtp, p.ids, kGen, greedy, false, &steps, &e);
    const Gen g = generate_lookup(mtp, p.ids, kGen, greedy);
    const spec_judge::StreamVerdict sv = spec_judge::streams(fj, "mtp_lookup_gpu_int8_" + p.name, p.ids, plain.ids,
                                                             steps, g.ids, uint32_t(model::Qwen35::kVocab),
                                                             uint32_t(model::Qwen35::kVocabUsed),
                                                             spec_judge::engine_plain_row(
                                                                 e, p.ids, uint32_t(model::Qwen35::kVocab)));
    std::printf("D2 %-28s %s, %llu iterations (%llu lookup, %llu ids in lookup), %.2f ids an iteration",
                p.name.c_str(), sv.line().c_str(),
                (unsigned long long)g.iters, (unsigned long long)g.lookup_iters, (unsigned long long)g.lookup_ids,
                double(kGen) / double(std::max<uint64_t>(1, g.iters)));
    if (p.far) {
      const Gen pm = generate_plain(mtp, p.ids, kGen, greedy, true);
      std::printf("; mtp %.2f ids an iteration%s; ms an id plain %.2f mtp %.2f mtp-lookup %.2f",
                  double(kGen) / double(std::max<uint64_t>(1, pm.iters)),
                  pm.ids == plain.ids ? "" : " (mtp DIFFERS)", plain.ms / kGen, pm.ms / kGen, g.ms / kGen);
      far_lookup_iters += g.lookup_iters;
    }
    std::printf("\n");
    fails += !sv.pass;
    ties += sv.tie.v == greedy_tie::Verdict::Tie;
    beyond += sv.tie.v == greedy_tie::Verdict::Diverged;
    judged += sv.judged;
    tot_iters += g.iters;
    tot_lookup_ids += g.lookup_ids;
    tot_ids += kGen;
  }
  std::printf("D2: %llu failing runs (spec 28 N6); tie census %llu near-ties, %llu beyond the 1-ulp rule, %llu judged "
              "by f32; %.2f ids an iteration, %.1f %% of the ids from lookup; lookup iterations on the "
              "edit / copy prompts %llu (busy-box timing above, informational)\n",
              (unsigned long long)fails, (unsigned long long)ties, (unsigned long long)beyond,
              (unsigned long long)judged, double(tot_ids) / double(tot_iters), 100.0 * tot_lookup_ids / tot_ids,
              (unsigned long long)far_lookup_iters);
  ok &= fails == 0;
  ok &= far_lookup_iters > 0;   // the long repeats entered lookup (else D2 proved nothing new)

  // --- H: the warm head ---------------------------------------------------------------------------------------
  {
    const std::vector<uint32_t> prompt = read_ids("tests/golden/edit-copy/reemit-8k.ids");
    const uint32_t b = uint32_t(prompt.size()), end = b + kGen - 1;   // the generated positions, the last one aside
    const Gen pm = generate_plain(mtp, prompt, kGen, greedy, true);
    const std::vector<uint8_t> kv_mtp = head_k_rows(mtp, e, b, end);
    const Gen g = generate_lookup(mtp, prompt, kGen, greedy);
    const std::vector<uint8_t> kv_lk = head_k_rows(mtp, e, b, end);
    const bool same_ids = g.ids == pm.ids;
    const size_t rb = e.mtp_buffers()->kv_lay.row_bytes();
    double min_cos = 1;
    uint32_t bitwise_rows = 0;
    for (uint32_t r = 0; r < end - b; ++r) {
      const uint8_t* x = kv_lk.data() + size_t(r) * rb;
      const uint8_t* y = kv_mtp.data() + size_t(r) * rb;
      bitwise_rows += std::memcmp(x, y, rb) == 0;
      double dot = 0, na = 0, nb = 0;
      for (size_t i = 0; i < rb; i += 2) {
        uint16_t u, v;
        std::memcpy(&u, x + i, 2);
        std::memcpy(&v, y + i, 2);
        const double fx = bf16(u), fy = bf16(v);
        dot += fx * fy;
        na += fx * fx;
        nb += fy * fy;
      }
      min_cos = std::min(min_cos, dot / std::sqrt(std::max(na * nb, 1e-300)));
    }
    std::printf("H: after %llu lookup iterations (%llu ids) the head's K rows of the %u generated positions against "
                "plain --spec mtp's (ids %s): min cosine %.6f, %u rows bitwise\n",
                (unsigned long long)g.lookup_iters, (unsigned long long)g.lookup_ids, end - b,
                same_ids ? "identical" : "DIFFER", min_cos, bitwise_rows);
    ok &= g.lookup_iters > 0 && (!same_ids || min_cos >= 0.99);
    bool did_cut = false;
    const Gen c = generate_lookup(mtp, prompt, kGen, greedy, /*cut=*/true, &did_cut);
    std::printf("H: a lookup burst truncated to 2 ids, then continued: %s\n",
                !did_cut ? "no lookup burst of 3 to truncate" : c.ids == g.ids ? "ids equal the untruncated run"
                                                                               : "ids DIFFER");
    ok &= did_cut && c.ids == g.ids;
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
    const std::vector<uint32_t> prompt = read_ids("tests/golden/edit-copy/reemit-8k.ids");
    Gen g[2];
    for (int r = 0; r < 2; ++r) {
      EngineAdapter fresh(e, model::Qwen35::kVocabUsed, kK);   // its RNG seeds from the request
      g[r] = generate_lookup(fresh, prompt, 128, s);
    }
    const bool same = g[0].ids == g[1].ids;
    std::printf("S: seeded sampled mtp-lookup run twice: %s; %llu lookup iterations, %llu of 128 ids in lookup\n",
                same ? "bitwise" : "DIFFERS", (unsigned long long)g[0].lookup_iters,
                (unsigned long long)g[0].lookup_ids);
    ok &= same;
  }

  std::printf("mtp_lookup_gpu_test: %s\n", ok ? "PASS" : "FAIL");
  return spec_judge::exit_code(ok, "mtp_lookup_gpu_test");
}
