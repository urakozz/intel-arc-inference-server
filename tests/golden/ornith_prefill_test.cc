// ornith_prefill_test - spec 15d's prefill gates on the card that are not the golden gate
// itself (R3 on prefill is prefill_gate_test on oracle-out-ornith):
//
//   1. the walk: Engine::prefill appends step_chunk_launches(desc, backend, C) per chunk
//      and step_head's 5 - Ornith's chunk, MoE blocks included, as derived in step.cc;
//   2. determinism and replay, bitwise: two immediate prefills, and two recorded
//      (B70_PREFILL_REPLAY's path), leave the same GDN state, conv ring, KV rows, last
//      logits and every layer's prefill route rows - the expert sum is in fixed slot order,
//      no sum is atomic, and a row's GEMM result does not depend on its tile neighbours;
//   3. chunking: the prompt prefilled in chunks of 64 and in one chunk - bitwise the same
//      state, KV, logits and routes (every MoE row is independent of the chunk it rides
//      in; the GDN chunks fall on the same 64-position boundaries);
//   4. prefill routes as decode routes: on every prompt position and every layer, the
//      prefill's top-8 expert SET equals decode's (Engine::ingest, one id per step) except
//      where decode's 8th and 9th probabilities are within kConsistTieRel (the two paths'
//      hidden states differ by GEMM rounding, so the boundary expert may flip there);
//      matched experts' weights within kConsistWeightAbs;
//   5. prefill vs decode tokens (spec 2 §6.3, ruling A26's rule): prefill + 32 greedy
//      tokens against ingest + 32 greedy tokens - every row whose decode-side top-2 gap is
//      at least one bf16 ulp must agree, then both are teacher-forced on decode's token;
//   6. R2 on the prefill path (spec 15 §5): with 15a's golden dumps (router_logits.L*),
//      every layer's prefill top-8 set equals the reference's from the oracle's router
//      logits (moe_ref's formula), near-ties (kR2TieRel) excepted, weights within
//      kR2WeightRel - ornith_decode_test's R2, on the rows a prefill routed.
//
// argv: <snapshot> <prompt ids dir> <backend l0|l0-int8> [oracle dir] [lm_head form].
// Exit 77 (SKIP) when the Ornith int4 checkpoint (urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-
// GPTQ, spec 15 §13) is not on this machine; R2 is skipped with a message without 15a's dumps.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "kernels/moe_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "runtime/prefill/backend.h"
#include "runtime/prefill/step.h"
#include "runtime/prefill_backend.h"

namespace {

using model::Qwen35;
constexpr uint32_t kMaxLen = 16384;
constexpr uint32_t kGen = 32;
// PROPOSED (spec 15d, written blind): set from the first box run's printed distribution.
constexpr float kConsistTieRel = 5e-2f;    // decode's (p8 - p9) / p8 below this: a near-tie
constexpr float kConsistWeightAbs = 1.0f / 32;   // |w_prefill - w_decode| (bf16 weights < 1)
constexpr float kR2TieRel = 1e-3f;         // ornith_decode_test's R2, unchanged
constexpr float kR2WeightRel = 1e-3f;
const char* const kPrompts[] = {"prose", "code", "cjk"};

float bits(uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// Everything a prefill leaves that the next token reads, plus the routes.
struct State {
  std::vector<uint8_t> state;      // gdn_state + conv_ring
  std::vector<uint16_t> kv;        // [0, n)
  std::vector<float> logits;       // the prefill head's row
  std::vector<uint32_t> routes;    // [layers][n][32], the prefill scratch's route rows
  bool operator==(const State& o) const {
    return state == o.state && kv == o.kv && logits == o.logits && routes == o.routes;
  }
};

State capture(runtime::Engine& e, l0::CmdList& imm, uint32_t n) {
  const model::ModelDesc& d = *e.model().desc;
  State s;
  s.state.resize(e.state_bytes());
  e.save_state(s.state.data());
  s.kv.resize(e.kv_bytes(n) / 2);
  e.save_kv(0, n, s.kv.data());
  s.logits.resize(Qwen35::kVocab);
  imm.copy(s.logits.data(), e.prefill_scratch()->logits.ptr(), s.logits.size() * 4);
  const runtime::MoePrefillLayout L = runtime::moe_prefill_layout(d);
  s.routes.resize(size_t(d.layers) * n * 32);
  for (uint32_t l = 0; l < d.layers; ++l)
    imm.copy(s.routes.data() + size_t(l) * n * 32,
             static_cast<const uint8_t*>(e.prefill_scratch()->moe->ptr()) + L.route_at(l),
             size_t(n) * 32 * 4);
  return s;
}

// The route rows of the LAST prefill chunk only are in the scratch (one chunk per layer
// slice); the tests here prefill prompts of at most kC ids in one call, or read after a
// chunked prefill only the rows of its last chunk.
uint32_t one_chunk(uint32_t n) {
  CHECK(n <= runtime::PrefillScratch::kC);
  return n;
}

struct SetStats {
  uint64_t rows = 0, exact = 0, near_ties = 0, bad = 0, bad_weights = 0;
  float worst_w = 0;
};

// One (position, layer): the prefill route row `p` against the reference ids / weights.
void compare_set(const uint32_t* p, const uint32_t* ref_ids, const float* ref_w, uint32_t top_k,
                 bool near_tie, float wtol, bool wrel, SetStats& c) {
  ++c.rows;
  std::vector<uint32_t> g(p, p + top_k), r(ref_ids, ref_ids + top_k), gs = g, rs = r;
  std::sort(gs.begin(), gs.end());
  std::sort(rs.begin(), rs.end());
  if (gs != rs) {
    if (!near_tie) {
      ++c.bad;
      return;
    }
    ++c.near_ties;   // the boundary expert may be either: the other top_k - 1 must agree
    std::vector<uint32_t> g7(g.begin(), g.end() - 1), r7(r.begin(), r.end() - 1);
    std::sort(g7.begin(), g7.end());
    std::sort(r7.begin(), r7.end());
    if (g7 != r7) ++c.bad;
    return;
  }
  ++c.exact;
  for (uint32_t k = 0; k < top_k; ++k) {
    const uint32_t j = uint32_t(std::find(r.begin(), r.end(), g[k]) - r.begin());
    const float w = bits(p[moe_ref::kWeights + k]);
    const float err = std::fabs(w - ref_w[j]) / (wrel ? std::max(ref_w[j], 1e-6f) : 1.0f);
    c.worst_w = std::max(c.worst_w, err);
    if (err > wtol) ++c.bad_weights;
  }
}

uint32_t argmax_used(const float* v) {
  return uint32_t(std::max_element(v, v + Qwen35::kVocabUsed) - v);
}
// One bf16 ulp of the top logit: decode's top-2 gap below it makes a row undetermined.
bool decode_tie(const std::vector<float>& lg) {
  const uint32_t a = argmax_used(lg.data());
  float second = -INFINITY;
  for (uint32_t i = 0; i < Qwen35::kVocabUsed; ++i)
    if (i != a) second = std::max(second, lg[i]);
  return double(lg[a]) - second < std::fabs(double(lg[a])) / 256.0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <snapshot> <prompt ids dir> <l0|l0-int8> [oracle dir] [lm_head]\n",
                 argv[0]);
    return 2;
  }
  const std::string snap = argv[1], pdir = argv[2];
  runtime::PrefillBackend backend = runtime::PrefillBackend::L0Int8;
  CHECK(runtime::parse_prefill_backend(argv[3], backend));
  CHECK(runtime::is_l0(backend));
  const std::string gdir = argc > 4 ? argv[4] : "oracle-out-ornith";
  loader::LmHeadForm lm_head = loader::LmHeadForm::Checkpoint;
  if (argc > 5) CHECK(loader::parse_lm_head_form(argv[5], lm_head));
  try {
    (void)loader::resolve_snapshot(snap);
  } catch (const std::exception& e) {
    std::printf("SKIP: no Ornith int4 checkpoint at '%s' (%s). None is published; spec 15a "
                "decides its source (AutoRound g64 proposed) - see the box validation queue.\n",
                snap.c_str(), e.what());
    return 77;
  }

  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, lm_head);
  CHECK(model.desc->is_moe());
  const model::ModelDesc& d = *model.desc;
  runtime::Engine eng(ctx, std::move(model), kMaxLen);
  eng.set_prefill_backend(backend);
  eng.prepare_prefill();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const runtime::MoeScratchLayout dml = runtime::moe_scratch_layout(d);
  std::printf("ornith_prefill_test: backend %s, %s\n", runtime::prefill_backend_name(backend),
              eng.memory_line().c_str());
  const std::vector<uint32_t> prose = golden::read_ids(pdir + "/prose.ids");
  const uint32_t n = one_chunk(uint32_t(prose.size()));

  // ---- 1. the walk ----------------------------------------------------------------------
  {
    eng.reset();
    const size_t before = eng.prefill_launches();
    eng.prefill(prose);
    const size_t want = runtime::prefill::step_chunk_launches(d, backend, n) + 5;
    CHECK_EQ(eng.prefill_launches() - before, want);
    // 2100 ids: two chunks (2048 + 52)
    std::vector<uint32_t> longer(2100);
    for (size_t i = 0; i < longer.size(); ++i) longer[i] = prose[i % prose.size()];
    eng.reset();
    const size_t b2 = eng.prefill_launches();
    eng.prefill(longer);
    CHECK_EQ(eng.prefill_launches() - b2,
             runtime::prefill::step_chunk_launches(d, backend, 2048) +
                 runtime::prefill::step_chunk_launches(d, backend, 52) + 5);
    std::printf("1. walk: %zu launches per chunk (%s), + 5 for the head\n",
                runtime::prefill::step_chunk_launches(d, backend, 2048),
                runtime::prefill_backend_name(backend));
  }

  // ---- 2. determinism and replay --------------------------------------------------------
  State ref;
  {
    eng.set_prefill_replay(false);
    eng.reset();
    eng.prefill(prose);
    ref = capture(eng, imm, n);
    eng.reset();
    eng.prefill(prose);
    CHECK(capture(eng, imm, n) == ref);
    eng.set_prefill_replay(true);
    for (int rep = 0; rep < 2; ++rep) {
      eng.reset();
      eng.prefill(prose);
      if (!(capture(eng, imm, n) == ref)) {
        std::fprintf(stderr, "recorded prefill %d differs from the immediate one\n", rep);
        return 1;
      }
    }
    eng.set_prefill_replay(false);
    std::printf("2. determinism: two immediate and two recorded prefills of %u ids bitwise equal "
                "(state, KV, logits, every layer's routes)\n", n);
  }

  // ---- 3. chunking ------------------------------------------------------------------------
  {
    // chunks of 64: the route rows left in the scratch are the LAST chunk's (rows [0, 64) of
    // each layer's slice hold positions [n_last0, n)), so compare those against ref's tail.
    eng.reset();
    eng.prefill(prose, 64);
    State c = capture(eng, imm, n);
    const uint32_t last0 = (n - 1) / 64 * 64, nl = n - last0;
    bool routes_ok = true;
    for (uint32_t l = 0; l < d.layers; ++l)
      routes_ok = routes_ok && std::memcmp(&c.routes[size_t(l) * n * 32],
                                           &ref.routes[(size_t(l) * n + last0) * 32],
                                           size_t(nl) * 32 * 4) == 0;
    const bool same = c.state == ref.state && c.kv == ref.kv && c.logits == ref.logits;
    std::printf("3. chunking: chunks of 64 vs one chunk - state / KV / logits %s, last chunk's "
                "routes %s\n", same ? "bitwise equal" : "DIFFER", routes_ok ? "bitwise equal" : "DIFFER");
    CHECK(same && routes_ok);
  }

  // ---- 4 and 5. prefill vs decode: routes, then tokens -------------------------------------
  SetStats cs;
  for (const char* pname : kPrompts) {
    const std::vector<uint32_t> ids = golden::read_ids(pdir + "/" + pname + ".ids");
    const uint32_t T = one_chunk(uint32_t(ids.size()));
    eng.reset();
    eng.prefill(ids);
    const State pf = capture(eng, imm, T);
    // decode, one id per step, its route row per layer after each
    eng.reset();
    std::vector<uint32_t> row(32);
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      for (uint32_t l = 0; l < d.layers; ++l) {
        imm.copy(row.data(), static_cast<const uint8_t*>(eng.buffers().moe->ptr()) + dml.route_at(l),
                 32 * 4);
        const float p8 = bits(row[moe_ref::kProbs + d.moe.top_k - 1]), p9 = bits(row[moe_ref::kProb9]);
        float w[8];
        for (uint32_t k = 0; k < d.moe.top_k; ++k) w[k] = bits(row[moe_ref::kWeights + k]);
        compare_set(&pf.routes[(size_t(l) * T + t) * 32], row.data() + moe_ref::kIds, w,
                    d.moe.top_k, (p8 - p9) <= kConsistTieRel * p8, kConsistWeightAbs, false, cs);
      }
    }
    // 5: the decode walk first (its ids and tie flags), then the prefill walk graded on it
    std::vector<uint32_t> dec_ids;
    std::vector<bool> dec_tie;
    std::vector<float> lg(Qwen35::kVocab);
    for (uint32_t s = 0; s < kGen; ++s) {   // decode: ingest left the first row in logits
      imm.copy(lg.data(), eng.buffers().logits.ptr(), lg.size() * 4);
      dec_tie.push_back(decode_tie(lg));
      dec_ids.push_back(eng.generate(1)[0]);
    }
    eng.reset();
    eng.prefill(ids);
    uint32_t det = 0, det_ok = 0, ties = 0, first_bad = kGen;
    bool forced = false;
    for (uint32_t s = 0; s < kGen; ++s) {
      const uint32_t id = eng.buffers().control.as<runtime::Control>()->cur_token[0];
      const bool ok = id == dec_ids[s];
      if (dec_tie[s]) ++ties;
      else {
        ++det;
        det_ok += ok;
        if (!ok && first_bad == kGen) first_bad = s;
      }
      if (!ok) forced = true;
      if (forced) eng.ingest({dec_ids[s]});
      else CHECK_EQ(eng.generate(1)[0], id);
    }
    std::printf("5. %s: %u ids, %u/%u determined rows equal decode's, %u undetermined%s\n", pname,
                T, det_ok, det, ties, first_bad < kGen ? "  ** FAIL **" : "");
    CHECK_EQ(det_ok, det);
  }
  std::printf("4. routes: %llu (position, layer) prefill top-%u sets vs decode's - %llu equal, "
              "%llu near-ties (decode's 8th/9th within %.0e), %llu differing; weights worst %.3e, "
              "%llu beyond %.3e (PROPOSED bars)\n",
              (unsigned long long)cs.rows, d.moe.top_k, (unsigned long long)cs.exact,
              (unsigned long long)cs.near_ties, double(kConsistTieRel), (unsigned long long)cs.bad,
              double(cs.worst_w), (unsigned long long)cs.bad_weights, double(kConsistWeightAbs));
  CHECK_EQ(cs.bad, uint64_t(0));
  CHECK_EQ(cs.bad_weights, uint64_t(0));

  // ---- 6. R2 on the prefill path ------------------------------------------------------------
  SetStats r2;
  bool any = false;
  const moe_ref::Shape s{d.moe.experts, d.moe.top_k, d.hidden, d.moe.expert_intermediate,
                         d.moe.router_n()};
  for (const char* pname : kPrompts) {
    const std::string gpath = gdir + "/" + pname + ".golden.safetensors";
    if (!golden::exists(gpath)) {
      std::printf("6. R2 SKIPPED for %s: %s is absent (spec 15a's golden set)\n", pname, gpath.c_str());
      continue;
    }
    golden::Golden g(gpath);
    if (!g.has("router_logits.L0")) {
      std::printf("6. R2 SKIPPED for %s: the dump has no router_logits.L* (15a's oracle)\n", pname);
      continue;
    }
    any = true;
    const std::vector<uint32_t> ids = golden::read_ids(pdir + "/" + pname + ".ids");
    const uint32_t T = one_chunk(uint32_t(ids.size()));
    const uint32_t rows = uint32_t(g.dim("router_logits.L0", 2, 0));
    CHECK(rows >= T);
    eng.reset();
    eng.prefill(ids);
    const State pf = capture(eng, imm, T);
    std::vector<float> lgr(s.router_n, 0.0f);
    for (uint32_t l = 0; l < d.layers; ++l) {
      const std::string ls = std::to_string(l);
      const uint16_t* gl = g.bf16("router_logits.L" + ls, size_t(rows) * s.experts);
      const bool have_sg = g.has("shared_gate_logits.L" + ls);
      for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t e = 0; e < s.experts; ++e) lgr[e] = common::bf16_to_f32(gl[size_t(t) * s.experts + e]);
        lgr[s.experts] = have_sg ? common::bf16_to_f32(g.bf16("shared_gate_logits.L" + ls, rows)[t]) : 0.0f;
        const moe_ref::Route want = moe_ref::route(lgr.data(), s);
        const float pk = want.p[s.top_k - 1];
        compare_set(&pf.routes[(size_t(l) * T + t) * 32], want.ids, want.w, s.top_k,
                    (pk - want.p9) <= kR2TieRel * pk, kR2WeightRel, true, r2);
      }
    }
  }
  if (any) {
    std::printf("6. R2 (prefill): %llu routes - %llu exact sets, %llu near-ties, %llu differing; "
                "weights worst %.3e relative, %llu beyond %.0e\n",
                (unsigned long long)r2.rows, (unsigned long long)r2.exact,
                (unsigned long long)r2.near_ties, (unsigned long long)r2.bad, double(r2.worst_w),
                (unsigned long long)r2.bad_weights, double(kR2WeightRel));
    CHECK_EQ(r2.bad, uint64_t(0));
    CHECK_EQ(r2.bad_weights, uint64_t(0));
  }
  std::puts("ornith_prefill_test OK");
  return 0;
}
