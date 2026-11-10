// Spec 17b T1 / T2 (plan 17b's review, Tasks 2-4): decode with tensor parallelism over two B70s.
//
//   reference  one card (runtime::Engine on card 0, the whole model): the prose prompt ingested, then 64 greedy ids -
//              each step's logits row (the row that decided the id), and at the end the per-layer residual taps, the
//              logits, every layer's GDN state and conv ring (save_state), the KV of [0, pos) (save_kv), pos.
//   slices     --parallel tp with model::TpCrossing::Slices: the same run. T1: the two cards' residual taps and Controls
//              identical after the run. T2: BITWISE the reference - ids, logits, taps, state, KV (the one-card host
//              layouts, runtime/tp_gather.h). Then reset() and the whole run again, bitwise (replay).
//   folded     the folded crossing (spec 17a's form): T1 bitwise between the cards; T2 by the golden tie rule
//              (tests/server/greedy_tie.h: the first divergence, if any, must be the reference row's runner-up at a
//              top-2 margin <= 1 bf16 ulp) - another summation order than one card's, so judged, not bitwise.
//
// argv: [1] the checkpoint, [2] the crossings to run ("both", "slices", "folded"); `bench <crossing> <depth> <tg>`
// instead times one engine (the speed stage's folded arm: b70-decode has no crossing switch, decision D1's constant).
// B70_KV_CACHE selects the KV form (tp_decode_kv8_test). The int8 head (b70-decode's). Exits 77 (SKIP) with fewer than
// two GPUs or without peer access both ways.
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

namespace {
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 64;
constexpr loader::LmHeadForm kHead = loader::LmHeadForm::Int8;

struct Record {
  std::vector<uint32_t> ids;
  std::vector<std::vector<float>> rows;   // the logits row that decided ids[i]
  std::vector<uint16_t> taps;
  std::vector<float> logits;
  std::vector<uint8_t> state, kv;
  uint32_t pos = 0;
};

std::vector<uint8_t> bytes_of(size_t n, const std::function<void(void*)>& save) {
  std::vector<uint8_t> v(n);
  save(v.data());
  return v;
}

size_t first_diff(const void* a, const void* b, size_t n) {
  const auto* x = static_cast<const uint8_t*>(a);
  const auto* y = static_cast<const uint8_t*>(b);
  for (size_t i = 0; i < n; ++i)
    if (x[i] != y[i]) return i;
  return n;
}

bool same_bytes(const char* what, const void* a, const void* b, size_t n) {
  const size_t at = first_diff(a, b, n);
  if (at == n) return true;
  std::fprintf(stderr, "%s differs at byte %zu of %zu\n", what, at, n);
  return false;
}

// One engine's run: ingest, then kGen ids one replay at a time, keeping the deciding rows.
template <class E, class Logits>
Record run(E& e, const std::vector<uint32_t>& prompt, const Logits& logits) {
  Record r;
  e.ingest(prompt);
  for (uint32_t i = 0; i < kGen; ++i) {
    r.rows.push_back(logits());
    r.ids.push_back(e.generate(1).at(0));
  }
  r.logits = logits();
  r.pos = e.pos();
  r.state = bytes_of(e.state_bytes(), [&](void* h) { e.save_state(h); });
  r.kv = bytes_of(e.kv_bytes(e.pos()), [&](void* h) { e.save_kv(0, e.pos(), h); });
  return r;
}

Record run_tp(runtime::TpEngine& e, const std::vector<uint32_t>& prompt) {
  Record r = run(e, prompt, [&] { return e.read_logits(); });
  // T1: both cards hold the same residual after every layer, and one Control.
  const std::vector<uint16_t> t0 = e.read_debug_resid(0), t1 = e.read_debug_resid(1);
  CHECK(same_bytes("T1: card 0's / card 1's residual taps", t0.data(), t1.data(), t0.size() * 2));
  const runtime::Control& a = e.control(0);
  const runtime::Control& b = e.control(1);
  CHECK_EQ(a.pos, b.pos);
  CHECK_EQ(a.cur_token[0], b.cur_token[0]);
  CHECK_EQ(a.out_token[0], b.out_token[0]);
  r.taps = t0;
  return r;
}

runtime::TpEngine make_tp(l0::Context& d0, l0::Context& d1, const std::string& ckpt, model::TpCrossing c,
                          runtime::KvCache kv, bool taps) {
  runtime::TpOptions opt;
  opt.debug_resid = taps;
  return runtime::TpEngine(d0, d1,
                           {loader::load_tp(d0, ckpt, {0, c}, kMaxLen, kHead), loader::load_tp(d1, ckpt, {1, c}, kMaxLen, kHead)},
                           kMaxLen, opt, kv);
}

int bench(l0::Context& d0, l0::Context& d1, const std::string& ckpt, model::TpCrossing c, uint32_t depth,
          uint32_t tg, runtime::KvCache kv) {
  runtime::TpEngine e = make_tp(d0, d1, ckpt, c, kv, false);
  const std::vector<uint32_t> prose = golden::read_ids("tests/golden/prompts/prose.ids");
  std::vector<uint32_t> ids;
  while (ids.size() < depth) ids.push_back(prose[ids.size() % prose.size()]);
  e.ingest(ids);
  e.generate(tg);
  std::printf("| tp_decode_test bench%s tp2-%s | %u | %u | %.2f | %.2f |\n",
              kv == runtime::KvCache::Int8 ? " int8-kv" : "", model::tp_crossing_name(c), depth, tg,
              e.last_tok_per_s(), e.last_gen_ms() / tg);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string ckpt = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const std::string mode = argc > 2 ? argv[2] : "both";
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
  if (mode == "bench") {
    if (argc < 6) {
      std::fprintf(stderr, "usage: tp_decode_test <ckpt> bench slices|folded <depth> <tg>\n");
      return 2;
    }
    const model::TpCrossing c = std::string(argv[3]) == "folded" ? model::TpCrossing::Folded
                                                                   : model::TpCrossing::Slices;
    return bench(d0, d1, ckpt, c, uint32_t(std::atoi(argv[4])), uint32_t(std::atoi(argv[5])), kv);
  }
  const std::vector<uint32_t> prompt = golden::read_ids("tests/golden/prompts/prose.ids");

  // --- the one-card reference on card 0 (the whole model; freed before the TP cards load) ----------------------
  Record ref;
  {
    runtime::Engine e(d0, loader::load(d0, ckpt, kMaxLen, false, kHead), kMaxLen, /*debug_resid=*/true, kv);
    l0::CmdList imm = l0::CmdList::immediate(d0);
    ref = run(e, prompt, [&] {
      std::vector<float> row(model::Qwen35::kVocab);
      imm.copy(row.data(), e.buffers().logits.ptr(), row.size() * sizeof(float));
      return row;
    });
    ref.taps = e.read_debug_resid();
    std::printf("reference (one card, card 0): %zu prompt ids + %u generated, pos %u, %s KV\n", prompt.size(), kGen,
                ref.pos, runtime::kv_cache_name(kv));
  }
  std::vector<greedy_tie::RefStep> steps;
  for (const std::vector<float>& row : ref.rows) steps.push_back(greedy_tie::ref_step(row.data(), model::Qwen35::kVocabUsed));

  bool ok = true;
  if (mode == "both" || mode == "slices") {
    runtime::TpEngine e = make_tp(d0, d1, ckpt, model::TpCrossing::Slices, kv, true);
    CHECK_EQ(e.step(0).kernel_count, runtime::tp_launches(*e.model(0).desc));
    for (uint32_t r = 0; r < 2; ++r) std::printf("slices: %s\n", e.memory_line(r).c_str());
    for (int round = 0; round < 2; ++round) {
      e.reset();
      const Record got = run_tp(e, prompt);
      const std::string what = std::string("slices") + (round ? " (replayed after reset)" : "");
      bool same = got.ids == ref.ids;
      if (!same) {
        const greedy_tie::Result v = greedy_tie::classify(ref.ids, steps, got.ids);
        std::fprintf(stderr, "%s: ids differ at %zu\n", what.c_str(), v.at);
      }
      same &= same_bytes((what + ": logits").c_str(), ref.logits.data(), got.logits.data(), ref.logits.size() * 4);
      same &= same_bytes((what + ": residual taps").c_str(), ref.taps.data(), got.taps.data(), ref.taps.size() * 2);
      CHECK_EQ(ref.state.size(), got.state.size());
      same &= same_bytes((what + ": GDN state + conv ring").c_str(), ref.state.data(), got.state.data(),
                         ref.state.size());
      CHECK_EQ(ref.kv.size(), got.kv.size());
      same &= same_bytes((what + ": KV").c_str(), ref.kv.data(), got.kv.data(), ref.kv.size());
      CHECK_EQ(ref.pos, got.pos);
      std::printf("%s: T1 bitwise between the cards; T2 %s one card (%zu ids, logits, %zu B state, %zu B KV)\n",
                  what.c_str(), same ? "BITWISE" : "DIFFERS FROM", got.ids.size(), got.state.size(), got.kv.size());
      ok &= same;
    }
  }
  if (mode == "both" || mode == "folded") {
    runtime::TpEngine e = make_tp(d0, d1, ckpt, model::TpCrossing::Folded, kv, true);
    for (uint32_t r = 0; r < 2; ++r) std::printf("folded: %s\n", e.memory_line(r).c_str());
    Record first;
    for (int round = 0; round < 2; ++round) {
      e.reset();
      const Record got = run_tp(e, prompt);
      if (round == 0) {
        first = got;
        const greedy_tie::Result v = greedy_tie::classify(ref.ids, steps, got.ids);
        const char* verdict = v.v == greedy_tie::Verdict::Same ? "identical ids"
                              : v.v == greedy_tie::Verdict::Tie ? "a near-tie divergence (<= 1 bf16 ulp)"
                                                                : "DIVERGED (not a near-tie)";
        std::printf("folded: T1 bitwise between the cards; T2 judged against one card: %s", verdict);
        if (v.v != greedy_tie::Verdict::Same)
          std::printf(" at id %zu (margin %.2f ulp, candidate rank %u)", v.at, v.margin_ulps, v.cand_rank);
        std::printf("; logits bitwise one card: %s\n",
                    first_diff(ref.logits.data(), got.logits.data(), ref.logits.size() * 4) == ref.logits.size() * 4
                        ? "yes"
                        : "no (expected: another summation order)");
        ok &= v.v != greedy_tie::Verdict::Diverged;
      } else {
        // Replay: the folded engine is bitwise itself.
        bool same = got.ids == first.ids;
        same &= same_bytes("folded replay: logits", first.logits.data(), got.logits.data(), first.logits.size() * 4);
        same &= same_bytes("folded replay: state", first.state.data(), got.state.data(), first.state.size());
        same &= same_bytes("folded replay: KV", first.kv.data(), got.kv.data(), first.kv.size());
        std::printf("folded (replayed after reset): %s\n", same ? "bitwise the first run" : "DIFFERS");
        ok &= same;
      }
    }
  }
  CHECK(ok);
  std::printf("tp_decode_test: OK (%s KV)\n", runtime::kv_cache_name(kv));
  return 0;
}
