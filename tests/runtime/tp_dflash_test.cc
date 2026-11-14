// Spec 19 v3c: DFlash under tensor parallel over two B70s - runtime::TpEngine with the drafter split over both cards
// (runtime::dflash::TpDFlashDrafter, plan 19v3c Tasks 3-7), the gate checkpoint with its int8 head and bf16 KV.
//
//   plain   --parallel tp without a proposer: per case, prefill and 64 greedy ids one replay at a time, each deciding
//           row kept - the reference the bars judge against.
//   dflash  --parallel tp with the split drafter (argv[2]: int4 g64 when it is an AutoRound export, else int8 at load;
//           the 2-bit candidate head):
//             G1  greedy speculative decoding at K = 7, 3, 1 (draft_tap -> verify(K) -> the longest agreeing prefix ->
//                 commit, its projection deferred to the next draft) against `plain`: identical, or each first
//                 difference a near-tie (<= 1 bf16 ulp) or accepted by the f32 judge of its context (forced_judge.h:
//                 B70_DUMP_FORCED dumps it, tools/box_validate/f32_refs.sh forced-qwen with QWEN_KEYS_RE='^tp_dflash_'
//                 makes it, B70_FORCED_JUDGE reads it); without a judge the test ends SKIP (77) "awaiting", never PASS.
//                 Tokens a round printed per case and K;
//             T1  after each run both Controls agree, and both drafter halves hold the same residual and final-norm
//                 rows bitwise (every all-reduce leaves the same sum on both cards);
//             S1  a snapshot after the prefill (save_state with the ring in one card's layout, save_kv) restored after
//                 reset() gives the same ids and the same drafts every round, bitwise;
//             T4  a draft list with card 1's dropped throws within its bounds; reset() recovers (the prose case's ids
//                 again).
//
// argv: [1] the checkpoint, [2] the drafter, [3] the mode: "gate" (the above) or "cost <depth>" - plan 19v3c Task 1 /
// Task 6's table at a depth: verify(k) + commit at M = k + 1 for k = 0..7 in units of the K = 0 iteration, the round's
// parts (draft_tap, verify, commit) at K = 7, and the crossings' share of a verify at M = 1 and 8 (per-launch events:
// tp_send + tp_res_fold over the list's sum, each card) - interleaved rounds after a warm-up, medians. Exits 77 (SKIP)
// with fewer than two GPUs or without peer access both ways.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "forced_judge.h"
#include "golden_common.h"
#include "greedy_tie.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/dflash/tp_dflash_drafter.h"
#include "runtime/tp_engine.h"

namespace {
constexpr uint32_t kGen = 64;
constexpr loader::LmHeadForm kHead = loader::LmHeadForm::Int8;
constexpr uint32_t kV = model::Qwen35::kVocab, kUsed = model::Qwen35::kVocabUsed;
std::string g_ckpt, g_drafter;

struct Case {
  std::string name;
  std::vector<uint32_t> ids;
};

struct Run {
  std::vector<uint32_t> ids;
  std::vector<std::vector<float>> rows;     // rows[i] decided ids[i] (rows[0]: the prefill's, not recorded)
  std::vector<std::vector<uint32_t>> drafts;   // each round's drafts
  uint64_t rounds = 0, accepted = 0;
};

// The two cards' slices, at max_len.
std::array<loader::LoadedModel, 2> load_cards(l0::Context& d0, l0::Context& d1, uint32_t max_len) {
  const model::TpCrossing c = runtime::kTpDefaultCrossing;
  return {loader::load_tp(d0, g_ckpt, {0, c}, max_len, kHead, false),
          loader::load_tp(d1, g_ckpt, {1, c}, max_len, kHead, false)};
}

// The engine with the split drafter (the drafter first: the engine borrows it).
struct Tp {
  std::unique_ptr<runtime::dflash::TpDFlashDrafter> drafter;
  std::unique_ptr<runtime::TpEngine> e;
  Tp(l0::Context& d0, l0::Context& d1, uint32_t max_len) {
    std::array<loader::LoadedModel, 2> cards = load_cards(d0, d1, max_len);
    // The drafter's own form: an AutoRound int4 export's packed linears (int4 g64 layout 1), else int8 at load.
    const loader::DraftFormat f =
        loader::DFlashCheckpoint(g_drafter).packed() ? loader::DraftFormat::Int4G64 : loader::DraftFormat::Int8;
    std::array<loader::DFlashWeights, 2> halves{loader::load_drafter(d0, g_drafter, cards[0], f, 0, true, 2, 80, 0),
                                                loader::load_drafter(d1, g_drafter, cards[1], f, 0, true, 2, 80, 1)};
    drafter = std::make_unique<runtime::dflash::TpDFlashDrafter>(std::array<l0::Context*, 2>{&d0, &d1},
                                                                  std::move(halves), max_len, 7);
    e = std::make_unique<runtime::TpEngine>(d0, d1, std::move(cards), max_len, runtime::TpOptions{},
                                            runtime::KvCache::Bf16, drafter.get());
    drafter->attach(*e);
  }
};

// Plain TP greedy ids one replay at a time, each deciding row kept.
Run plain(l0::Context& d0, l0::Context& d1, const std::vector<uint32_t>& prompt, uint32_t max_len) {
  runtime::TpEngine e(d0, d1, load_cards(d0, d1, max_len), max_len, {}, runtime::KvCache::Bf16);
  float* pinned = e.host_rows(kV);
  Run r;
  e.prefill(prompt);
  for (uint32_t i = 0; i < kGen; ++i) {
    if (i == 0) {
      r.rows.emplace_back();
    } else {
      e.read_logits_into(pinned, 1);
      r.rows.emplace_back(pinned, pinned + kV);
    }
    r.ids.push_back(e.generate(1).at(0));
  }
  return r;
}

// Greedy speculative decoding at a fixed K: draft_tap -> verify -> the agreeing prefix -> commit (deferred).
Run speculative(runtime::TpEngine& e, uint32_t K) {
  Run r;
  while (r.ids.size() < kGen) {
    const uint32_t k = std::min(K, e.max_verify_k());
    const uint32_t x = e.pending();
    uint32_t j = 0;
    if (k == 0) {
      e.verify(0);
      e.commit(0, e.verify_ids()[0], true);
    } else {
      const runtime::dflash::DFlashDraftOut& o = e.draft_tap(k, 0.0f, 0);
      e.verify(k);
      std::vector<uint32_t> d(o.ids, o.ids + k);
      while (j < k && d[j] == e.verify_ids()[j]) ++j;
      e.commit(j, e.verify_ids()[j], true);
      r.drafts.push_back(d);
      ++r.rounds;
      r.accepted += j;
      r.ids.push_back(x);
      for (uint32_t i = 0; i < j && r.ids.size() < kGen; ++i) r.ids.push_back(d[i]);
      continue;
    }
    r.ids.push_back(x);
  }
  r.ids.resize(kGen);
  e.flush_projection();
  return r;
}

// T1: one Control, and both drafter halves' residual and final-norm rows bitwise.
void t1(runtime::TpEngine& e, runtime::dflash::TpDFlashDrafter& d) {
  const runtime::Control& a = e.control(0);
  const runtime::Control& b = e.control(1);
  CHECK_EQ(a.pos, b.pos);
  CHECK_EQ(a.cur_token[0], b.cur_token[0]);
  CHECK_EQ(a.gdn_live, b.gdn_live);
  for (const char* which : {"resid", "x"}) {
    std::vector<uint8_t> h[2];
    for (uint32_t r = 0; r < 2; ++r) {
      const l0::Mem& m = std::string(which) == "resid" ? d.buffers(r).resid : d.buffers(r).x;
      h[r].resize(m.size());
      l0::CmdList imm = l0::CmdList::immediate(e.context(r));
      imm.copy(h[r].data(), m.ptr(), m.size());
    }
    if (h[0] != h[1]) std::printf("T1 FAIL: the drafter halves' %s rows differ between the cards\n", which);
    CHECK(h[0] == h[1]);
  }
}

struct Judged {
  bool ok = true, awaiting = false;
};

// tp_mtp_test's judge: the tie rule, then the f32 judge of the context for a non-tie divergence.
Judged judge(const std::string& what, const std::string& tag, const std::vector<uint32_t>& prompt, const Run& ref,
             const Run& got) {
  std::vector<greedy_tie::RefStep> steps;
  for (const std::vector<float>& row : ref.rows)
    steps.push_back(row.empty() ? greedy_tie::RefStep{} : greedy_tie::ref_step(row.data(), kUsed));
  const greedy_tie::Result v = greedy_tie::classify(ref.ids, steps, got.ids);
  const char* verdict = v.v == greedy_tie::Verdict::Same  ? "identical ids"
                        : v.v == greedy_tie::Verdict::Tie ? "a near-tie divergence (<= 1 bf16 ulp)"
                                                          : "DIVERGED (not a near-tie)";
  std::printf("%s: %s", what.c_str(), verdict);
  if (v.v != greedy_tie::Verdict::Same)
    std::printf(" at id %zu (ref %u, got %u, margin %.2f ulp, candidate rank %u)", v.at, ref.ids[v.at], got.ids[v.at],
                v.margin_ulps, v.cand_rank);
  if (got.rounds)
    std::printf("; %.3f tokens a round (%llu rounds, %llu accepted)",
                double(got.rounds + got.accepted) / double(got.rounds), (unsigned long long)got.rounds,
                (unsigned long long)got.accepted);
  std::printf("\n");
  Judged j;
  if (v.v != greedy_tie::Verdict::Diverged) return j;
  if (v.at == 0) {
    j.ok = false;
    return j;
  }
  const golden::ForcedJudge fj;
  const std::string k = golden::ForcedJudge::key(tag, prompt, got.ids);
  fj.dump(k, prompt, got.ids);
  const std::unique_ptr<golden::Golden> g = fj.open(k, kGen, kV);
  if (!g) {
    std::printf("  AWAITING the f32 judge of %s (B70_DUMP_FORCED, then tools/box_validate/f32_refs.sh forced-qwen "
                "with QWEN_KEYS_RE='^tp_dflash_'; B70_FORCED_JUDGE=<the int8-head judges' dir>)\n", k.c_str());
    j.awaiting = true;
    return j;
  }
  const uint32_t row = uint32_t(v.at);
  const bool got_ok = golden::judge_accepts(*g, row, got.ids[row], kV, kUsed);
  const bool ref_ok = golden::judge_accepts(*g, row, ref.ids[row], kV, kUsed);
  if (!got_ok) golden::judge_explain(*g, row, got.ids[row], kV, kUsed, "got");
  std::printf("  the f32 judge at id %u: %u %s, the reference's %u %s\n", row, got.ids[row],
              got_ok ? "accepted" : "REJECTED", ref.ids[row], ref_ok ? "accepted" : "rejected");
  j.ok = got_ok;
  return j;
}

// `only`: G1 on the named case only (a diagnostic run); empty: every case.
int gate(l0::Context& d0, l0::Context& d1, const std::string& only) {
  constexpr uint32_t kMaxLen = 16384;
  std::vector<uint32_t> six = golden::read_ids("tests/golden/prompts/long32k.ids");
  six.resize(6000);
  const std::vector<Case> cases{{"prose", golden::read_ids("tests/golden/prompts/prose.ids")},
                                {"code", golden::read_ids("tests/golden/prompts/code.ids")},
                                {"long", golden::read_ids("tests/golden/prompts/long.ids")},
                                {"long32k[:6000]", six}};
  std::vector<Run> ref;
  for (const Case& c : cases)
    ref.push_back(only.empty() || c.name == only || c.name == "prose" || c.name == "long" ? plain(d0, d1, c.ids, kMaxLen)
                                                                                         : Run{});
  std::printf("plain --parallel tp: %zu cases x %u ids\n", cases.size(), kGen);

  Tp tp(d0, d1, kMaxLen);
  runtime::TpEngine& e = *tp.e;
  for (uint32_t r = 0; r < 2; ++r) std::printf("%s\n", e.memory_line(r).c_str());
  bool ok = true, awaiting = false;
  for (size_t i = 0; i < cases.size(); ++i)
    for (uint32_t K : {7u, 3u, 1u}) {
      if (!only.empty() && cases[i].name != only) continue;
      e.reset();
      e.prefill(cases[i].ids);
      const Run got = speculative(e, K);
      t1(e, *tp.drafter);
      const Judged j = judge("G1 " + cases[i].name + " K = " + std::to_string(K),
                             "tp_dflash_" + cases[i].name + "_k" + std::to_string(K), cases[i].ids, ref[i], got);
      ok = ok && j.ok;
      awaiting = awaiting || j.awaiting;
    }

  // S1: the snapshot after the prefill, restored: the same ids and drafts every round, bitwise.
  try {
    const std::vector<uint32_t>& p = cases[2].ids;
    e.reset();
    e.prefill(p);
    const uint32_t pos = e.pos(), pending = e.pending();
    std::vector<uint8_t> state(e.state_bytes()), kv(e.kv_bytes(pos));
    e.save_state(state.data());
    e.save_kv(0, pos, kv.data());
    std::printf("S1 cold run from pos %u\n", pos);
    const Run cold = speculative(e, 7);
    std::printf("S1 restore\n");
    e.reset();
    e.load_state(state.data(), pos);
    e.load_kv(0, pos, kv.data());
    e.set_token(pending);
    const Run warm = speculative(e, 7);
    const bool same = cold.ids == warm.ids && cold.drafts == warm.drafts;
    std::printf("S1 a restore after the prefill: %s\n", same ? "the same ids and drafts, bitwise" : "DIFFERENT");
    ok = ok && same;
  } catch (const std::exception& ex) {
    std::printf("S1 FAIL: threw at pos %u: %s\n", e.pos(), ex.what());
    ok = false;
  }

  // T4: a dropped draft list fails within its bounds; reset() recovers.
  {
    e.reset();
    e.prefill(cases[0].ids);
    e.drop_next(1);
    bool threw = false;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      e.draft_tap(7, 0.0f, 0);
    } catch (const std::exception& ex) {
      threw = true;
      std::printf("T4 the dropped draft list: %s\n", ex.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("T4 threw %s after %.2f s\n", threw ? "yes" : "NO", s);
    ok = ok && threw;
    try {
      e.reset();
      e.prefill(cases[0].ids);
      const Run again = speculative(e, 7);
      const Judged j = judge("T4 prose after reset, K = 7", "tp_dflash_prose_k7", cases[0].ids, ref[0], again);
      ok = ok && j.ok;
      awaiting = awaiting || j.awaiting;
    } catch (const std::exception& ex) {
      std::printf("T4 FAIL: the run after reset threw at pos %u: %s\n", e.pos(), ex.what());
      ok = false;
    }
  }
  if (!ok) {
    std::printf("tp_dflash_test FAIL\n");
    return 1;
  }
  if (awaiting) {
    std::printf("tp_dflash_test SKIP: every bar but the f32 judges above passed; awaiting their references\n");
    return 77;
  }
  std::printf("tp_dflash_test PASS\n");
  return 0;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// The crossings' share of a verify at M rows: per card, the summed durations of its tp_send and tp_res_fold launches
// against the list's summed durations (medians of 8 replays; the fold's time includes its wait).
void crossings(runtime::TpEngine& e, uint32_t M) {
  std::array<std::unique_ptr<runtime::ProfileEvents>, 2> ev;
  std::array<runtime::CaptureOptions, 2> opt;
  for (uint32_t r = 0; r < 2; ++r) {
    ev[r] = std::make_unique<runtime::ProfileEvents>(e.context(r));
    opt[r].profile = ev[r].get();
    opt[r].profile_m = M;
  }
  e.recapture_verify(opt);
  std::vector<double> cross[2], all[2];
  for (int i = 0; i < 9; ++i) {
    for (uint32_t r = 0; r < 2; ++r)
      for (auto& x : ev[r]->events) x.reset();
    for (uint32_t m = 0; m + 1 < M; ++m) e.set_draft_input(m, e.pending());
    e.verify(M - 1);
    e.commit(0, e.verify_ids()[0], true);
    if (i == 0) continue;   // warm-up
    for (uint32_t r = 0; r < 2; ++r) {
      const runtime::CapturedStep& s = e.verify_step(r, M);
      double c = 0, a = 0;
      for (size_t l = 0; l < s.labels.size(); ++l) {
        const double us = ev[r]->events[l].duration_us();
        a += us;
        if (s.labels[l].find(" tp_send ") != std::string::npos || s.labels[l].find(" tp_res_fold ") != std::string::npos)
          c += us;
      }
      cross[r].push_back(c / 1000.0);
      all[r].push_back(a / 1000.0);
    }
  }
  for (uint32_t r = 0; r < 2; ++r)
    std::printf("  crossings at M = %u, card %u: %.3f ms of %.3f ms summed launch time (%.1f %%)\n", M, r,
                median(cross[r]), median(all[r]), 100.0 * median(cross[r]) / median(all[r]));
  e.recapture_verify({});
}

int cost(l0::Context& d0, l0::Context& d1, uint32_t depth) {
  const uint32_t len = std::max(32768u, (depth + 2048 + 255) / 256 * 256);
  Tp tp(d0, d1, len);
  runtime::TpEngine& e = *tp.e;
  const std::vector<uint32_t> prose = golden::read_ids("tests/golden/prompts/prose.ids");
  std::vector<uint32_t> ids;
  while (ids.size() < depth) ids.push_back(prose[ids.size() % prose.size()]);
  e.prefill(ids);
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  constexpr int kReps = 16, kRounds = 5;
  std::vector<double> v[8], dr, dv, dc;
  for (int round = 0; round <= kRounds; ++round) {   // round 0 warms up
    for (uint32_t k = 0; k <= 7; ++k) {
      const auto t0 = Clock::now();
      for (int i = 0; i < kReps; ++i) {
        for (uint32_t m = 0; m < k; ++m) e.set_draft_input(m, e.pending());
        e.verify(k);
        e.commit(0, e.verify_ids()[0], true);
      }
      if (round) v[k].push_back(ms(t0) / kReps);
    }
    // The round at K = 7: the draft (deferred projection, both cards' draft lists, the host merge), the verify (the
    // tails first), the commit.
    double a = 0, b = 0, c = 0;
    for (int i = 0; i < kReps; ++i) {
      const auto t0 = Clock::now();
      e.draft_tap(7, 0.0f, 0);
      const auto t1 = Clock::now();
      e.verify(7);
      const auto t2 = Clock::now();
      e.commit(0, e.verify_ids()[0], true);
      const auto t3 = Clock::now();
      a += std::chrono::duration<double, std::milli>(t1 - t0).count();
      b += std::chrono::duration<double, std::milli>(t2 - t1).count();
      c += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }
    if (round) {
      dr.push_back(a / kReps);
      dv.push_back(b / kReps);
      dc.push_back(c / kReps);
    }
  }
  const double base = median(v[0]);
  std::printf("tp dflash cost at depth %u (bf16 KV): K = 0 iteration %.3f ms (verify M = 1 + commit)\n", depth, base);
  std::printf("  verify at M = 1..8 (units of the K = 0 iteration):");
  for (uint32_t k = 0; k <= 7; ++k) std::printf(" %.3f", median(v[k]) / base);
  std::printf("\n  ms:");
  for (uint32_t k = 0; k <= 7; ++k) std::printf(" %.3f", median(v[k]));
  std::printf("\n  the round at K = 7: draft_tap %.3f ms (the deferred projection, both draft lists, the merge), verify "
              "%.3f ms (the tails first), commit %.3f ms; %.3f ms in all (medians of %d rounds of %d)\n",
              median(dr), median(dv), median(dc), median(dr) + median(dv) + median(dc), kRounds, kReps);
  crossings(e, 1);
  crossings(e, 8);
  // The crossing lever's A/B: the tap verify lists at M = 8 with fp32 against bf16 messages (the production lists),
  // interleaved rounds after a warm-up - each round recaptures both cards' lists in its form.
  std::vector<double> ab[2];
  for (int round = 0; round <= kRounds; ++round)
    for (int f = 0; f < 2; ++f) {
      std::array<runtime::CaptureOptions, 2> o;
      o[0].tp_fp32_crossing = o[1].tp_fp32_crossing = f == 0;
      e.recapture_verify(o);
      const auto t0 = Clock::now();
      for (int i = 0; i < kReps; ++i) {
        for (uint32_t m = 0; m < 7; ++m) e.set_draft_input(m, e.pending());
        e.verify(7);
        e.commit(0, e.verify_ids()[0], true);
      }
      if (round) ab[f].push_back(ms(t0) / kReps);
    }
  e.recapture_verify({});
  std::printf("  verify at M = 8 + commit: fp32 crossings %.3f ms, bf16 crossings %.3f ms (%.3f ms saved)\n",
              median(ab[0]), median(ab[1]), median(ab[0]) - median(ab[1]));
  return 0;
}
// `stress <n>`: n speculative runs (K = 7, the long prompt) with the tap verify lists' crossings in fp32 and n in
// bf16, alternating; every failed crossing caught, counted by status and reset() - how often each form loses one.
int stress(l0::Context& d0, l0::Context& d1, uint32_t n) {
  Tp tp(d0, d1, 16384);
  runtime::TpEngine& e = *tp.e;
  const std::vector<uint32_t> p = golden::read_ids("tests/golden/prompts/long.ids");
  uint32_t fails[2] = {0, 0}, runs[2] = {0, 0};
  for (uint32_t i = 0; i < 2 * n; ++i) {
    const int f = int(i % 2);
    std::array<runtime::CaptureOptions, 2> o;
    o[0].tp_fp32_crossing = o[1].tp_fp32_crossing = f == 0;
    e.reset();
    e.recapture_verify(o);
    ++runs[f];
    try {
      e.prefill(p);
      speculative(e, 7);
    } catch (const std::exception& ex) {
      ++fails[f];
      std::printf("STRESS %s run %u: %s\n", f == 0 ? "fp32" : "bf16", i / 2, ex.what());
    }
  }
  std::printf("STRESS fp32 crossings: %u of %u runs lost a crossing; bf16: %u of %u (64 ids a run, ~%u crossings each)\n",
              fails[0], runs[0], fails[1], runs[1], 64u / 3u * 138u);
  return fails[0] + fails[1] ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
  g_ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  g_drafter = argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2";
  const std::string mode = argc > 3 ? argv[3] : "gate";
  const uint32_t gpus = l0::Context::gpu_count();
  if (gpus < 2) {
    std::printf("SKIP: --parallel tp needs two GPUs, Level Zero shows %u\n", gpus);
    return 77;
  }
  l0::Context d0(0u);
  l0::Context d1(d0, 1u);
  if (!d0.can_access_peer(d1) || !d1.can_access_peer(d0)) {
    std::printf("SKIP: the cards cannot access each other's memory (zeDeviceCanAccessPeer)\n");
    return 77;
  }
  if (mode == "stress") return stress(d0, d1, argc > 4 ? uint32_t(std::atoi(argv[4])) : 20);
  if (mode == "cost") {
    if (argc < 5) {
      std::fprintf(stderr, "usage: tp_dflash_test <ckpt> <drafter> cost <depth>\n");
      return 2;
    }
    return cost(d0, d1, uint32_t(std::atoi(argv[4])));
  }
  return gate(d0, d1, mode == "gate" ? std::string() : mode);   // any other mode: G1 on that case only
}
