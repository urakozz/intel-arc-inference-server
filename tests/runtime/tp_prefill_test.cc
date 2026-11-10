// Spec 17c: tensor-parallel PREFILL over two B70s (runtime::TpEngine::prefill, plan 17c).
//
//   reference  one card (runtime::Engine on card 0, the whole model, Engine::prefill at the default chunk): per
//              prompt, prefill, then 64 greedy ids - each step's logits row (the row that decided the next id) - and
//              the state (save_state) and KV (save_kv) right after the prefill; for the two-call case a first
//              prefill, 8 ids, then a second prefill of more ids from there (a later turn of a session).
//   tp         --parallel tp's engine (the folded crossing, kTpDefaultCrossing): the same runs.
//                T1  the two cards' Controls agree after every prefill, and the residual taps of the decode steps
//                    after it are bitwise between the cards;
//                T2  judged against one card by the golden tie rule (tests/server/greedy_tie.h, tp_decode_test's
//                    folded rule): the ids identical, or the first divergence a near-tie (<= 1 bf16 ulp) of the
//                    reference row. The first id after a prefill has no recorded row, so a divergence there fails.
//                    The K split of the row-parallel linears is another summation order than one card's: logits,
//                    state and KV are printed (how many bytes differ, the second row's cosine), not gated - spec 2
//                    §6.3's rule, "tokens gate, tensors diagnose";
//                chunk invariance  the 2820-id prompt at chunk 1024 (two pairs) bitwise the same at chunk 2048
//                    (one pair): the 64-grid cuts are the walk's own, so the wavefront's pairing changes nothing;
//                replay  reset() and the first prompt again, bitwise the first run.
//
// argv: [1] the checkpoint. B70_KV_CACHE selects the KV form (tp_prefill_kv8_test). The int8 head (b70-decode's).
// Exits 77 (SKIP) with fewer than two GPUs or without peer access both ways.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "greedy_tie.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/tp_engine.h"
#include "runtime/tp_plan.h"

namespace {
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 64;
constexpr loader::LmHeadForm kHead = loader::LmHeadForm::Int8;

struct Case {
  std::string name;
  std::vector<uint32_t> first, second;   // second empty: one prefill call
};

struct Record {
  std::vector<uint32_t> ids;
  std::vector<std::vector<float>> rows;   // rows[i] decided ids[i] (rows[0]: none recorded, empty)
  std::vector<uint8_t> state, kv;         // right after the (last) prefill
  uint32_t pos = 0;
};

std::vector<uint8_t> bytes_of(size_t n, const std::function<void(void*)>& save) {
  std::vector<uint8_t> v(n);
  save(v.data());
  return v;
}

size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  size_t n = 0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) n += a[i] != b[i];
  return n + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

double cosine(const std::vector<float>& a, const std::vector<float>& b, uint32_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (uint32_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return ab / std::sqrt(aa * bb);
}

// One engine's case: prefill (and the second call after 8 ids), the state and KV, then kGen ids one replay at a time.
template <class E, class Logits>
Record run(E& e, const Case& c, const Logits& logits, const std::function<void()>& after_prefill = {}) {
  Record r;
  e.reset();
  e.prefill(c.first);
  if (!c.second.empty()) {
    e.generate(8);
    e.prefill(c.second);
  }
  if (after_prefill) after_prefill();
  r.pos = e.pos();
  r.state = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
  r.kv = bytes_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
  for (uint32_t i = 0; i < kGen; ++i) {
    r.rows.push_back(i == 0 ? std::vector<float>{} : logits());
    r.ids.push_back(e.generate(1).at(0));
  }
  return r;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
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
  const std::vector<uint32_t> prose = golden::read_ids("tests/golden/prompts/prose.ids");
  const std::vector<uint32_t> code = golden::read_ids("tests/golden/prompts/code.ids");
  const std::vector<uint32_t> longp = golden::read_ids("tests/golden/prompts/long.ids");
  std::vector<uint32_t> six = golden::read_ids("tests/golden/prompts/long32k.ids");
  six.resize(6000);   // three pairs: (2048, 2048), (960, 944)
  const std::vector<Case> cases{{"prose", prose, {}},
                                {"code", code, {}},
                                {"long", longp, {}},
                                {"long32k[:6000]", six, {}},
                                {"prose + 8 ids + code (two calls)", prose, code}};

  // --- the one-card reference on card 0 (freed before the TP cards load) ---------------------------------------
  std::vector<Record> ref;
  {
    runtime::Engine e(d0, loader::load(d0, ckpt, kMaxLen, false, kHead), kMaxLen, /*debug_resid=*/false, kv);
    l0::CmdList imm = l0::CmdList::immediate(d0);
    const auto logits = [&] {
      std::vector<float> row(model::Qwen35::kVocab);
      imm.copy(row.data(), e.buffers().logits.ptr(), row.size() * sizeof(float));
      return row;
    };
    for (const Case& c : cases) {
      ref.push_back(run(e, c, logits));
      std::printf("reference (one card, card 0): %s, pos %u, %s KV\n", c.name.c_str(), ref.back().pos,
                  runtime::kv_cache_name(kv));
    }
  }

  runtime::TpOptions opt;
  opt.debug_resid = true;
  runtime::TpEngine e(d0, d1,
                      {loader::load_tp(d0, ckpt, {0, runtime::kTpDefaultCrossing}, kMaxLen, kHead),
                       loader::load_tp(d1, ckpt, {1, runtime::kTpDefaultCrossing}, kMaxLen, kHead)},
                      kMaxLen, opt, kv);
  e.prepare_prefill();
  std::printf("tp: crossing %s, crossings %s\n", model::tp_crossing_name(e.crossing()),
              e.prefill_overlaps() ? "on the copy engines (the wavefront)" : "on the compute lists");
  for (uint32_t r = 0; r < 2; ++r) std::printf("tp: %s\n", e.memory_line(r).c_str());
  const auto t1_controls = [&] {
    const runtime::Control& a = e.control(0);
    const runtime::Control& b = e.control(1);
    CHECK_EQ(a.pos, b.pos);
    CHECK_EQ(a.cur_token[0], b.cur_token[0]);
  };
  const auto tp_logits = [&] { return e.read_logits(); };

  bool ok = true;
  std::vector<Record> got;
  for (size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    got.push_back(run(e, c, tp_logits, t1_controls));
    const Record& g = got.back();
    const Record& r = ref[i];
    // T1: the decode steps after the prefill computed one residual on both cards.
    const std::vector<uint16_t> t0 = e.read_debug_resid(0), t1 = e.read_debug_resid(1);
    CHECK(t0 == t1);
    CHECK_EQ(g.pos, r.pos);
    std::vector<greedy_tie::RefStep> steps;
    for (const std::vector<float>& row : r.rows)
      steps.push_back(row.empty() ? greedy_tie::RefStep{} : greedy_tie::ref_step(row.data(), model::Qwen35::kVocabUsed));
    const greedy_tie::Result v = greedy_tie::classify(r.ids, steps, g.ids);
    const char* verdict = v.v == greedy_tie::Verdict::Same  ? "identical ids"
                          : v.v == greedy_tie::Verdict::Tie ? "a near-tie divergence (<= 1 bf16 ulp)"
                                                            : "DIVERGED (not a near-tie)";
    std::printf("%s: T1 bitwise between the cards; T2 judged against one card: %s", c.name.c_str(), verdict);
    if (v.v != greedy_tie::Verdict::Same)
      std::printf(" at id %zu (ref %u, tp %u, margin %.2f ulp, candidate rank %u)", v.at, r.ids[v.at], g.ids[v.at],
                  v.margin_ulps, v.cand_rank);
    std::printf("\n  diagnostics: row 1 cosine %.9f; state %zu of %zu bytes differ; KV %zu of %zu bytes differ\n",
                cosine(r.rows[1], g.rows[1], model::Qwen35::kVocabUsed), differing(r.state, g.state), r.state.size(),
                differing(r.kv, g.kv), r.kv.size());
    ok &= v.v != greedy_tie::Verdict::Diverged;
  }

  // Chunk invariance: the long prompt at chunk 1024 (two pairs) against its chunk-2048 run above (one pair).
  {
    e.reset();
    e.prefill(longp, 1024);
    const std::vector<uint8_t> st = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
    const std::vector<uint8_t> k = bytes_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
    const uint32_t first = e.control(0).cur_token[0];
    const bool same = st == got[2].state && k == got[2].kv && first == got[2].ids[0];
    std::printf("chunk invariance (long, %zu ids: chunks of 1024 in %zu pairs vs of 2048 in %zu): %s\n", longp.size(),
                (runtime::tp_prefill_chunks(0, longp.size(), 1024).size() + 1) / 2,
                (runtime::tp_prefill_chunks(0, longp.size(), 2048).size() + 1) / 2,
                same ? "bitwise" : "DIFFERS");
    ok &= same;
  }
  // Replay: the first case again after reset(), bitwise.
  {
    const Record again = run(e, cases[0], tp_logits, t1_controls);
    const bool same = again.ids == got[0].ids && again.state == got[0].state && again.kv == got[0].kv;
    std::printf("replay (prose after reset): %s\n", same ? "bitwise the first run" : "DIFFERS");
    ok &= same;
  }
  CHECK(ok);
  std::printf("tp_prefill_test: OK (%s KV)\n", runtime::kv_cache_name(kv));
  return 0;
}
