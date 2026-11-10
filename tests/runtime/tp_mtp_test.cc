// Spec 17d: MTP speculative decoding, the snapshots' restore and the server's calls under tensor parallelism over two
// B70s (runtime::TpEngine with the MTP head whole on each card, plan 17d).
//
//   one card   runtime::Engine on card 0 with the head (freed before the TP cards load): long32k[:6000] prefilled and
//              64 greedy ids (the cold run, each deciding row kept), and its snapshot at 4096 (save_state / save_kv:
//              the one-card host layouts, the head's hidden row and KV layer included).
//   plain      --parallel tp without the head (the folded crossing, kTpDefaultCrossing): per case, prefill and 64
//              greedy ids one replay at a time, each deciding row kept - the reference the bars judge against.
//   mtp        --parallel tp with the head:
//                M1  plain ids in the head's engine (ingest / generate run verify(0) + commit) against `plain` by the
//                    golden tie rule (greedy_tie.h) - the M = 1 verify list is the plain walk with the slot build;
//                M2  greedy speculative decoding at K = 1, 2, 3 (draft -> verify -> the longest agreeing prefix ->
//                    commit) against `plain`: identical, or each first difference a near-tie (<= 1 bf16 ulp) - or,
//                    beyond that, the f32 judge of its context (forced_judge.h: B70_DUMP_FORCED dumps it,
//                    tools/box_validate/f32_refs.sh forced-qwen with QWEN_KEYS_RE='^tp_mtp_' makes it,
//                    B70_FORCED_JUDGE reads it) accepting MTP's id; without a judge the test ends SKIP (77)
//                    "awaiting", never PASS. The acceptance (accepted / drafted) is printed;
//                T1  after each run both Controls agree and the head's hidden row and KV layer are bitwise between
//                    the cards (both run the same head on the same residual);
//                D9  the gathers: every verify row's full row (read_logits_into, both halves) has the merged id as
//                    its argmax, and a draft picked by the gathered draft row's argmax (draft(k, pick)) is the merged
//                    draft;
//                S1  the block hook under TP: long32k[:6000] fires at 4096 (the pair stride) and at the end; the
//                    snapshot taken at 4096 restored (load_state / load_kv) and the rest prefilled gives the cold
//                    run's state, KV and 64 ids BITWISE;
//                S2  one card's snapshot at 4096 restored into TP, and TP's restored into one card (a fresh Engine,
//                    card 0, after the TP cards are freed): each run's 64 ids judged against that engine's own cold
//                    run by the golden tie rule (the snapshot carries the other engine's rounding);
//                T4  a verify step with card 1's list dropped throws within its bounds; reset() recovers and the
//                    prose case at K = 3 gives its ids again.
//
// argv: [1] the checkpoint; [2] the mode: "gate" (the above), or "cost <depth>" - the TP cost table at a depth
// (verify at M = 1..4 and drafts k = 1..3 in units of the K = 0 iteration, interleaved rounds, medians) - the AdaptiveK
// table MtpCost::tp_int8_head records. B70_KV_CACHE selects the KV form (tp_mtp_kv8_test). The int8 head. Exits 77
// (SKIP) with fewer than two GPUs or without peer access both ways.
#include <algorithm>
#include <chrono>
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
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/tp_engine.h"

namespace {
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 64;
constexpr uint32_t kSnap = 4096;   // the block hook's stride at the default 2048-row cap (tp_hook_stride)
constexpr loader::LmHeadForm kHead = loader::LmHeadForm::Int8;
constexpr uint32_t kV = model::Qwen35::kVocab, kUsed = model::Qwen35::kVocabUsed;

struct Case {
  std::string name;
  std::vector<uint32_t> ids;
};

struct Run {
  std::vector<uint32_t> ids;
  std::vector<std::vector<float>> rows;   // rows[i] decided ids[i] (rows[0]: the prefill's, not recorded)
  uint64_t drafted = 0, accepted = 0;
  std::vector<uint8_t> state, kv;
};

std::vector<uint8_t> bytes_of(size_t n, const std::function<void(void*)>& save) {
  std::vector<uint8_t> v(n);
  save(v.data());
  return v;
}

std::vector<float> full_row(runtime::TpEngine& e, float* pinned, uint32_t m = 0) {
  e.read_logits_into(pinned, m + 1);
  return std::vector<float>(pinned + size_t(m) * kV, pinned + size_t(m + 1) * kV);
}

uint32_t argmax_of(const float* row) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < kUsed; ++i)
    if (row[i] > row[b]) b = i;
  return b;
}

// Plain greedy ids one replay at a time, each deciding row kept (generate(1) runs verify(0) + commit with the head).
template <class E, class Row>
Run plain(E& e, const std::vector<uint32_t>& prompt, const Row& row) {
  Run r;
  e.reset();
  e.prefill(prompt);
  for (uint32_t i = 0; i < kGen; ++i) {
    r.rows.push_back(i == 0 ? std::vector<float>{} : row());
    r.ids.push_back(e.generate(1).at(0));
  }
  return r;
}

// Greedy speculative decoding at a fixed K (b70-decode's generate_mtp): exactly kGen ids.
Run speculative(runtime::TpEngine& e, uint32_t K, bool check_gathers, float* pinned, bool prefilled = false,
                const std::vector<uint32_t>& prompt = {}) {
  Run r;
  if (!prefilled) {
    e.reset();
    e.prefill(prompt);
  }
  while (r.ids.size() < kGen) {
    const uint32_t k = std::min(K, e.max_verify_k());
    const uint32_t x = e.pending();
    uint32_t j = 0;
    if (k == 0) {
      e.verify(0);
      e.commit(0, e.verify_ids()[0]);
    } else {
      if (check_gathers) {
        // D9 on the draft rows: the gathered row's argmax is the merged draft.
        e.draft(k, [&](uint32_t i) {
          e.read_draft_logits_into(pinned, i);
          const uint32_t a = argmax_of(pinned);
          // the merged device winner (both head Controls' out_token before the pick)
          CHECK_EQ(a, e.mtp_buffers(0)->hctl.as<runtime::Control>()->out_token[0]);
          CHECK_EQ(a, e.mtp_buffers(1)->hctl.as<runtime::Control>()->out_token[0]);
          return a;
        });
      } else {
        e.draft(k);
      }
      e.verify(k);
      if (check_gathers) {   // D9 on the verify rows
        e.read_logits_into(pinned, k + 1);
        for (uint32_t m = 0; m <= k; ++m) CHECK_EQ(argmax_of(pinned + size_t(m) * kV), e.verify_ids()[m]);
      }
      while (j < k && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
      e.commit(j, e.verify_ids()[j]);
    }
    r.ids.push_back(x);
    for (uint32_t i = 0; i < j && r.ids.size() < kGen; ++i) r.ids.push_back(e.draft_ids()[i]);
    r.drafted += k;
    r.accepted += j;
  }
  r.ids.resize(kGen);
  return r;
}

// T1 after a run: one Control, and the head's hidden row and KV layer bitwise between the cards.
void t1(runtime::TpEngine& e, uint32_t pos) {
  const runtime::Control& a = e.control(0);
  const runtime::Control& b = e.control(1);
  CHECK_EQ(a.pos, b.pos);
  CHECK_EQ(a.cur_token[0], b.cur_token[0]);
  CHECK_EQ(a.gdn_live, b.gdn_live);
  const runtime::MtpBuffers* m0 = e.mtp_buffers(0);
  const runtime::MtpBuffers* m1 = e.mtp_buffers(1);
  l0::CmdList i0 = l0::CmdList::immediate(e.context(0));
  l0::CmdList i1 = l0::CmdList::immediate(e.context(1));
  std::vector<uint8_t> h0(m0->hh.size()), h1(m1->hh.size());
  const size_t row = size_t(e.model(0).desc->hidden) * 2;
  i0.copy(h0.data(), m0->hh.ptr(), row);
  i1.copy(h1.data(), m1->hh.ptr(), row);
  CHECK(std::memcmp(h0.data(), h1.data(), row) == 0);
  const size_t kvb = size_t(pos) * m0->kv_lay.row_bytes();   // the head layer's rows [0, pos)
  std::vector<uint8_t> k0(kvb), k1(kvb);
  i0.copy(k0.data(), m0->kv_k.ptr(), kvb);
  i1.copy(k1.data(), m1->kv_k.ptr(), kvb);
  CHECK(k0 == k1);
}

struct Judged {
  bool ok = true, awaiting = false;
};

// MTP's ids against the plain reference: the tie rule, then the f32 judge of the context for a non-tie divergence.
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
  if (got.drafted) std::printf("; acceptance %.3f (%llu / %llu)", double(got.accepted) / double(got.drafted),
                               (unsigned long long)got.accepted, (unsigned long long)got.drafted);
  std::printf("\n");
  Judged j;
  if (v.v != greedy_tie::Verdict::Diverged) return j;
  if (v.at == 0) {   // the prefill's id: no recorded row, and both engines prefill alike
    j.ok = false;
    return j;
  }
  const golden::ForcedJudge fj;
  const std::string k = golden::ForcedJudge::key(tag, prompt, got.ids);
  fj.dump(k, prompt, got.ids);
  const std::unique_ptr<golden::Golden> g = fj.open(k, kGen, kV);
  if (!g) {
    std::printf("  AWAITING the f32 judge of %s (B70_DUMP_FORCED, then tools/box_validate/f32_refs.sh forced-qwen "
                "with QWEN_KEYS_RE='^tp_mtp_'; B70_FORCED_JUDGE=<the int8-head judges' dir>)\n", k.c_str());
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

runtime::TpEngine make_tp(l0::Context& d0, l0::Context& d1, const std::string& ckpt, bool mtp, runtime::KvCache kv,
                          uint32_t max_len = kMaxLen) {
  const model::TpCrossing c = runtime::kTpDefaultCrossing;
  return runtime::TpEngine(d0, d1,
                           {loader::load_tp(d0, ckpt, {0, c}, max_len, kHead, mtp),
                            loader::load_tp(d1, ckpt, {1, c}, max_len, kHead, mtp)},
                           max_len, {}, kv);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// The TP cost table at `depth`: verify(k) + commit(0) at M = k + 1 and draft(k), interleaved rounds after a warm-up,
// in units of the K = 0 iteration (verify(0) + commit(0) - what a plain step costs with the head loaded).
int cost(l0::Context& d0, l0::Context& d1, const std::string& ckpt, uint32_t depth, runtime::KvCache kv) {
  const uint32_t len = std::max(32768u, (depth + 2048 + 255) / 256 * 256);
  runtime::TpEngine e = make_tp(d0, d1, ckpt, true, kv, len);
  const std::vector<uint32_t> prose = golden::read_ids("tests/golden/prompts/prose.ids");
  std::vector<uint32_t> ids;
  while (ids.size() < depth) ids.push_back(prose[ids.size() % prose.size()]);
  e.prefill(ids);
  using Clock = std::chrono::steady_clock;
  const auto ms = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  constexpr int kReps = 16, kRounds = 5;
  std::vector<double> v[4], d[3];
  for (int round = 0; round <= kRounds; ++round) {   // round 0 warms up
    for (uint32_t k = 0; k <= 3; ++k) {
      const auto t0 = Clock::now();
      for (int i = 0; i < kReps; ++i) {
        for (uint32_t m = 0; m < k; ++m) e.set_draft_input(m, e.pending());
        e.verify(k);
        e.commit(0, e.verify_ids()[0]);
      }
      if (round) v[k].push_back(ms(t0) / kReps);
    }
    for (uint32_t k = 1; k <= 3; ++k) {
      const auto t0 = Clock::now();
      for (int i = 0; i < kReps; ++i) e.draft(k);
      if (round) d[k - 1].push_back(ms(t0) / kReps);
    }
  }
  const double base = median(v[0]);
  std::printf("tp cost at depth %u (%s KV): K = 0 iteration %.3f ms (verify M = 1 + commit)\n", depth,
              runtime::kv_cache_name(kv), base);
  std::printf("  the cost table: verify=");
  for (uint32_t k = 0; k <= 3; ++k) std::printf("%s%.3f", k ? "," : "", median(v[k]) / base);
  std::printf(";draft=");
  for (uint32_t k = 0; k < 3; ++k) std::printf("%s%.3f", k ? "," : "", median(d[k]) / base);
  std::printf("\n  ms: verify");
  for (uint32_t k = 0; k <= 3; ++k) std::printf(" %.3f", median(v[k]));
  std::printf(", draft");
  for (uint32_t k = 0; k < 3; ++k) std::printf(" %.3f", median(d[k]));
  std::printf(" (medians of %d rounds of %d)\n", kRounds, kReps);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const std::string mode = argc > 2 ? argv[2] : "gate";
  const runtime::KvCache kv = runtime::default_kv_cache();
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
  if (mode == "cost") {
    if (argc < 4) {
      std::fprintf(stderr, "usage: tp_mtp_test <ckpt> cost <depth>\n");
      return 2;
    }
    return cost(d0, d1, ckpt, uint32_t(std::atoi(argv[3])), kv);
  }
  const char* kvn = kv == runtime::KvCache::Int8 ? "kv8_" : "";
  std::vector<uint32_t> six = golden::read_ids("tests/golden/prompts/long32k.ids");
  six.resize(6000);
  const std::vector<Case> cases{{"prose", golden::read_ids("tests/golden/prompts/prose.ids")},
                                {"code", golden::read_ids("tests/golden/prompts/code.ids")},
                                {"long", golden::read_ids("tests/golden/prompts/long.ids")},
                                {"long32k[:6000]", six}};
  const std::vector<uint32_t> head4k(six.begin(), six.begin() + kSnap), rest(six.begin() + kSnap, six.end());

  // --- one card with the head, card 0: the cold run of long32k[:6000] and its snapshot at 4096 ------------------
  Run one_cold;
  std::vector<uint8_t> one_state, one_kv;
  const auto one_card = [&](const std::function<void(runtime::Engine&, const std::function<std::vector<float>()>&)>& f) {
    runtime::Engine e(d0, loader::load(d0, ckpt, kMaxLen, /*mtp=*/true, kHead), kMaxLen, false, kv);
    l0::CmdList imm = l0::CmdList::immediate(d0);
    const auto row = [&] {
      std::vector<float> r(kV);
      imm.copy(r.data(), e.buffers().logits.ptr(), r.size() * sizeof(float));
      return r;
    };
    f(e, row);
  };
  one_card([&](runtime::Engine& e, const std::function<std::vector<float>()>& row) {
    one_cold = plain(e, six, row);
    e.reset();
    e.prefill(head4k);
    one_state = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
    one_kv = bytes_of(e.kv_bytes(kSnap), [&](void* h) { e.save_kv(0, kSnap, h); });
    std::printf("one card (card 0, the head loaded): long32k[:6000] cold, the snapshot at %u (%zu B state, %zu B KV)\n",
                kSnap, one_state.size(), one_kv.size());
  });

  // --- plain TP: the reference ------------------------------------------------------------------------------
  std::vector<Run> ref;
  {
    runtime::TpEngine e = make_tp(d0, d1, ckpt, false, kv);
    e.prepare_prefill();
    float* pinned = e.host_rows(size_t(4) * kV);
    for (const Case& c : cases) {
      ref.push_back(plain(e, c.ids, [&] { return full_row(e, pinned); }));
      std::printf("plain tp (no head): %s, %zu prompt ids + %u generated\n", c.name.c_str(), c.ids.size(), kGen);
    }
  }

  // --- TP with the head --------------------------------------------------------------------------------------
  bool ok = true, awaiting = false;
  const auto take = [&](const Judged& j) {
    ok &= j.ok;
    awaiting |= j.awaiting;
  };
  std::vector<uint8_t> tp_state, tp_kv;
  Run tp_cold;
  {
    runtime::TpEngine e = make_tp(d0, d1, ckpt, true, kv);
    e.prepare_prefill();
    CHECK(e.mtp());
    CHECK_EQ(e.verify_step(0, 1).kernel_count, runtime::tp_verify_launches(*e.model(0).desc));
    std::printf("tp with the head: verify lists M = 1..4 (%u captured twice by depth), %zu launches each, draft lists "
                "%zu launches\n", e.verify_deep_lists(), e.verify_step(0, 1).kernel_count,
                e.draft_step(0, 0).kernel_count);
    for (uint32_t r = 0; r < 2; ++r) std::printf("tp: %s\n", e.memory_line(r).c_str());
    float* pinned = e.host_rows(size_t(4) * kV);
    std::vector<uint32_t> prose_k3;
    for (size_t i = 0; i < cases.size(); ++i) {
      const Case& c = cases[i];
      // M1: plain ids through the M = 1 verify list.
      const Run p = plain(e, c.ids, [&] { return full_row(e, pinned); });
      t1(e, e.pos());
      take(judge(c.name + ": M1 plain ids with the head vs plain tp", std::string("tp_mtp_") + kvn + "m1_" + c.name,
                 c.ids, ref[i], p));
      // M2: K = 1, 2, 3 (K = 3 checks the gathers too).
      for (uint32_t K = 1; K <= 3; ++K) {
        const Run s = speculative(e, K, K == 3, pinned, false, c.ids);
        t1(e, e.pos());
        if (K == 3 && i == 0) prose_k3 = s.ids;
        take(judge(c.name + ": M2 speculative K = " + std::to_string(K) + " vs plain tp",
                   std::string("tp_mtp_") + kvn + "k" + std::to_string(K) + "_" + c.name, c.ids, ref[i], s));
      }
    }

    // S1: the block hook at the pair stride, the snapshot restored bitwise.
    std::vector<uint32_t> hook_ends;
    e.reset();
    e.set_block_hook([&](uint32_t end, bool block_end) {
      hook_ends.push_back(end);
      if (end == kSnap) {
        CHECK(block_end);
        tp_state = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
        tp_kv = bytes_of(e.kv_bytes(kSnap), [&](void* h) { e.save_kv(0, kSnap, h); });
      }
    });
    e.prefill(six);
    e.set_block_hook({});
    CHECK(hook_ends == (std::vector<uint32_t>{kSnap, 6000u}));
    CHECK_EQ(tp_state.size(), one_state.size());   // the one-card host layouts
    CHECK_EQ(tp_kv.size(), one_kv.size());
    const std::vector<uint8_t> cold_state = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
    const std::vector<uint8_t> cold_kv = bytes_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
    tp_cold = speculative(e, 3, false, pinned, true);
    e.reset();
    e.load_state(tp_state.data(), kSnap);
    e.load_kv(0, kSnap, tp_kv.data());
    e.prefill(rest);
    const std::vector<uint8_t> st = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
    const std::vector<uint8_t> kvr = bytes_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
    const Run warm = speculative(e, 3, false, pinned, true);
    const bool s1 = st == cold_state && kvr == cold_kv && warm.ids == tp_cold.ids;
    std::printf("S1: the hook at %u and %u; restored at %u and the rest prefilled: state %s, KV %s, 64 ids at K = 3 "
                "%s\n", hook_ends[0], hook_ends[1], kSnap, st == cold_state ? "bitwise" : "DIFFERS",
                kvr == cold_kv ? "bitwise" : "DIFFERS", warm.ids == tp_cold.ids ? "identical" : "DIFFER");
    ok &= s1;

    // S2a: one card's snapshot into TP, judged against TP's own cold run (plain ids, rows kept).
    {
      const Run cold_rows = [&] {
        Run r;
        e.reset();
        e.prefill(six);
        for (uint32_t i = 0; i < kGen; ++i) {
          r.rows.push_back(i == 0 ? std::vector<float>{} : full_row(e, pinned));
          r.ids.push_back(e.generate(1).at(0));
        }
        return r;
      }();
      e.reset();
      e.load_state(one_state.data(), kSnap);
      e.load_kv(0, kSnap, one_kv.data());
      e.prefill(rest);
      Run w;
      for (uint32_t i = 0; i < kGen; ++i) w.ids.push_back(e.generate(1).at(0));
      take(judge("S2a: one card's snapshot restored into tp vs tp's cold run",
                 std::string("tp_mtp_") + kvn + "s2a", six, cold_rows, w));
    }

    // T4: a verify step with card 1's list dropped fails within its bounds; reset() recovers.
    {
      e.reset();
      e.prefill(cases[0].ids);
      e.generate(2);
      e.drop_next(1);
      const auto t0 = std::chrono::steady_clock::now();
      bool threw = false;
      try {
        e.verify(0);
      } catch (const std::exception& x) {
        threw = true;
        std::printf("T4: the dropped verify step threw in %.0f ms: %s\n",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), x.what());
      }
      CHECK(threw);
      e.reset();
      const Run again = speculative(e, 3, false, pinned, false, cases[0].ids);
      std::printf("T4: after reset() the prose case at K = 3 %s\n", again.ids == prose_k3 ? "gives its ids again" : "DIFFERS");
      ok &= again.ids == prose_k3;
    }
  }

  // S2b: TP's snapshot into one card (a fresh Engine on card 0), judged against one card's cold run.
  one_card([&](runtime::Engine& e, const std::function<std::vector<float>()>&) {
    e.reset();
    e.load_state(tp_state.data(), kSnap);
    e.load_kv(0, kSnap, tp_kv.data());
    e.prefill(rest);
    Run w;
    for (uint32_t i = 0; i < kGen; ++i) w.ids.push_back(e.generate(1).at(0));
    take(judge("S2b: tp's snapshot restored into one card vs one card's cold run",
               std::string("tp_mtp_") + kvn + "s2b", six, one_cold, w));
  });

  CHECK(ok);
  if (awaiting) {
    std::printf("SKIP: tp_mtp_test - every check passed but a divergence awaits its f32 judge (%s KV)\n",
                runtime::kv_cache_name(kv));
    return 77;
  }
  std::printf("tp_mtp_test: OK (%s KV)\n", runtime::kv_cache_name(kv));
  return 0;
}
