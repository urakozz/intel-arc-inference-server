// Spec 25d Task 3 (B4 on prefill, plan 25d Review Focus 1-3): BoltEngine::prefill with a checkpoint - box only (labels
// checkpoint;bolt;prefill). 25b's synthetic N 12 (one card) carries it before SergiioB's export (two cards).
//
//   bolt_prefill_test <checkpoint> <ids> [bf16] [eager] [b16] [layers:<N>] [pp2[:<split>]]
//
// The prompt: <ids> cycled to kN = 4500 positions (chunks of 2048, 2048, 404 from position 0).
//
//   1. the walk: prefill_chunk_launches(desc, placement) per chunk (asserted by the walk itself, chunk by chunk) and the
//      head's 4; the plan's prefill term (prefill_scratch_bytes) equals the allocation, device by device
//   2. determinism and replay, bitwise: two immediate prefills (reset between) and two RECORDED ones (set_prefill_replay:
//      B70_PREFILL_REPLAY's path) leave the same KV rows, the three ring columns decode reads next, every SSM state, the
//      logits, the last chunk's route rows, the first token
//   3. Review Focus 1, bitwise: chunks of 512 / 1024 / 64 against 2048 (= one chunk per 2048: chunked = whole); the prompt
//      in two prefill calls split at 64 / 2048 / 2112 / 4096 (multiples of 64); a continuation that starts OFF the grid
//      (positions [0, 2085) first, then [2085, n) in one call against [2085, 2112) + [2112, n): every row they share); a
//      split off the grid (1, 2047, 2085) printed, not gated - its sub-chunk is cut there, so its states are another
//      evaluation (the walk itself never cuts off the grid)
//   4. prefill against decode's M = 1 fill (ingest, one id a replay) of the same prompt (PROPOSED bars, spec 25 §6.2 and
//      plan 25d Task 3): logits cosine >= 0.9995; the h taps at the last row median >= 0.9998, min >= 0.99; every Mamba
//      block's SSM state cosine >= 0.999 and its three conv columns cosine >= 0.999 (the in_proj GEMM's rounded rows, not
//      decode's GEMV's: not bitwise - spec 25 §16 records the departure from the plan's "bitwise"); every attention block's
//      K / V rows: median >= 0.9998, p01 >= 0.99
//   5. routing on the last chunk's rows: expert SETS equal except a near-tie (decode's 9th / 10th gap <= B70_BOLT_TIE_TOL,
//      one expert different); 0 gaps undetermined
//   6. tokens: prefill + 32 greedy against ingest + 32 greedy, tie-aware (every determined row agrees), teacher-forced
//   7. pp2[:<split>]: the same prefill on two cards (the copy hand-off a chunk) bitwise one card (when it fits one)
// `eager` / `b16` run B70_BOLT_ATTN=eager / B70_BOLT_SSM=bf16 in BOTH halves. Exit 77 (SKIP) when the checkpoint is absent
// or the placement does not fit with the prefill scratch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "bolt_rig.h"
#include "check.h"
#include "common/bf16.h"
#include "kernels/bolt_kernels.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/control.h"

namespace {

namespace rb = runtime::bolt;
namespace kr = kernels::bolt::route;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 4500, kGen = 32;
float tie_tol() {
  const char* e = std::getenv("B70_BOLT_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 1e-3f;   // PROPOSED (Decision 11)
}
float bits(uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}
double cos_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
double cos_f32(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

struct Snap {
  uint32_t first_token = 0, pos = 0;
  std::vector<std::vector<uint16_t>> kv;     // per attention block: K rows then V rows, positions [0, n)
  std::vector<std::vector<uint16_t>> conv;   // per Mamba block: the three raw columns decode reads next
  std::vector<std::vector<float>> states;    // per Mamba block: the SSM state
  std::vector<float> logits;
  std::vector<uint32_t> routes;              // the last chunk's (prefill) - empty for decode
  std::vector<uint16_t> taps;                // [layers][hidden]: each block's output at the last row
  bool operator==(const Snap& o) const {
    if (states.size() != o.states.size()) return false;
    for (size_t i = 0; i < states.size(); ++i)
      if (states[i].size() != o.states[i].size() ||
          std::memcmp(states[i].data(), o.states[i].data(), states[i].size() * 4) != 0)
        return false;
    return first_token == o.first_token && pos == o.pos && kv == o.kv && conv == o.conv &&
           logits.size() == o.logits.size() && std::memcmp(logits.data(), o.logits.data(), logits.size() * 4) == 0 &&
           routes == o.routes;
  }
};
Snap snap(rb::BoltEngine& e, uint32_t n, bool routes, bool taps) {
  const model::BoltDesc& d = e.model().desc;
  Snap s;
  const runtime::Control c = e.control(e.devices() - 1);
  s.first_token = c.cur_token[0];
  s.pos = c.pos;
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, 0, n, false), v = e.read_kv(a, 0, n, true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  for (uint32_t m = 0; m < d.mamba_layers(); ++m) {
    const std::vector<uint16_t> ring = e.read_conv_ring(m);
    std::vector<uint16_t> live;
    for (uint32_t p = n - 3; p < n; ++p)
      live.insert(live.end(), ring.begin() + size_t(p % 16) * d.m_conv_dim, ring.begin() + size_t(p % 16 + 1) * d.m_conv_dim);
    s.conv.push_back(std::move(live));
    s.states.push_back(e.read_ssm_state(m));
  }
  s.logits = e.read_logits();
  if (routes) s.routes = e.read_prefill_routes();
  if (taps) s.taps = e.read_debug_h();
  return s;
}
const uint32_t* pf_route(const std::vector<uint32_t>& r, uint32_t l, uint32_t row) {
  return r.data() + rb::pf_route_at(l) / 4 + size_t(row) * rb::kRouteWords;
}

int run_all(rb::BoltEngine& e, const Ids& prompt) {
  const model::BoltDesc& d = e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;

  // ---- 1. the walk and the plan --------------------------------------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  const size_t chunks = (n + rb::kPfC - 1) / rb::kPfC, per_chunk = rb::prefill_chunk_launches(d, e.model().place);
  CHECK_EQ(e.prefill_launches() - l0, chunks * per_chunk + rb::kPrefillHeadLaunches);
  CHECK_EQ(e.pos(), n);
  for (uint32_t dev = 0; dev < e.devices(); ++dev)
    CHECK_EQ(e.memory_use(dev).prefill_scratch, rb::prefill_scratch_bytes(d, e.model().place, dev));
  std::printf("1. walk: %zu chunk(s) x %zu + %zu launches; prefill scratch %.3f GB a device = the plan's\n%s\n", chunks,
              per_chunk, rb::kPrefillHeadLaunches, rb::prefill_sizes(d).total() / 1e9, e.memory_line().c_str());
  const Snap a = snap(e, n, true, true);

  // ---- 2. determinism and replay -------------------------------------------------------------------------------------
  e.reset();
  e.prefill(prompt);
  const bool same = snap(e, n, true, false) == a;
  e.set_prefill_replay(true);
  e.reset();
  e.prefill(prompt);   // records
  const bool rec1 = snap(e, n, true, false) == a;
  e.reset();
  e.prefill(prompt);   // replays the recordings
  const bool rec2 = snap(e, n, true, false) == a;
  e.set_prefill_replay(false);
  std::printf("2. B4: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS", rec1 ? "bitwise" : "DIFFERS",
              rec2 ? "bitwise" : "DIFFERS");
  ok = ok && same && rec1 && rec2;

  // ---- 3. Review Focus 1 ---------------------------------------------------------------------------------------------
  {
    Snap ref = a;
    ref.routes.clear();
    bool all = true;
    for (uint32_t c : {512u, 1024u, 64u}) {
      e.reset();
      e.prefill(prompt, c);
      const bool b = snap(e, n, false, false) == ref;
      std::printf("3. chunks of %4u vs %u: KV, ring columns, SSM states, logits, first token %s\n", c, rb::kPfC,
                  b ? "bitwise" : "DIFFER");
      all = all && b;
    }
    for (uint32_t s : {64u, 2048u, 2112u, 4096u}) {
      e.reset();
      e.prefill(Ids(prompt.begin(), prompt.begin() + s));
      e.prefill(Ids(prompt.begin() + s, prompt.end()));
      const bool b = snap(e, n, false, false) == ref;
      std::printf("   split at %4u (on the 64-grid): %s\n", s, b ? "bitwise" : "DIFFERS");
      all = all && b;
    }
    {   // a continuation starting off the grid at 2085: one call vs [2085, 2112) + [2112, n) - the rows they share
      const Ids p0(prompt.begin(), prompt.begin() + 2085);
      e.reset();
      e.prefill(p0);
      e.prefill(Ids(prompt.begin() + 2085, prompt.end()));
      const Snap one = snap(e, n, false, false);
      e.reset();
      e.prefill(p0);
      e.prefill(Ids(prompt.begin() + 2085, prompt.begin() + 2112));
      e.prefill(Ids(prompt.begin() + 2112, prompt.end()));
      const bool b = snap(e, n, false, false) == one;
      std::printf("   a chunk starting at 2048 + 37 vs one at 2048 + 64 (the same prefix): %s\n", b ? "bitwise" : "DIFFERS");
      all = all && b;
    }
    for (uint32_t s : {1u, 2047u, 2085u}) {   // off the grid: another evaluation of that sub-chunk - printed
      e.reset();
      e.prefill(Ids(prompt.begin(), prompt.begin() + s));
      e.prefill(Ids(prompt.begin() + s, prompt.end()));
      const Snap o = snap(e, n, false, false);
      double sc = 1.0;
      for (size_t m = 0; m < o.states.size(); ++m)
        sc = std::min(sc, cos_f32(o.states[m].data(), ref.states[m].data(), o.states[m].size()));
      std::printf("   split at %4u (off the grid, not gated): %s; the states' worst cosine to one chunk %.9f\n", s,
                  o == ref ? "bitwise" : "differs", sc);
    }
    ok = ok && all;
  }

  // ---- 4 / 5. against decode's fill ----------------------------------------------------------------------------------
  e.reset();
  const uint32_t row0 = ((n - 1) / rb::kPfC) * rb::kPfC;   // the last chunk's first position (chunks from 0)
  uint64_t rows = 0, exact = 0, ties = 0, undet = 0, bad = 0;
  for (uint32_t p = 0; p < n; ++p) {
    e.ingest({prompt[p]});
    if (p < row0) continue;
    const std::vector<uint32_t> dr = e.read_routes();   // [moe blocks][48]
    uint32_t mi = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (!d.is_moe(l)) continue;
      const uint32_t* drow = dr.data() + size_t(mi++) * rb::kRouteWords;
      const uint32_t* prow = pf_route(a.routes, l, p - row0);
      ++rows;
      std::vector<uint32_t> ds(drow + kr::kIds, drow + kr::kIds + 9), ps(prow + kr::kIds, prow + kr::kIds + 9);
      if (ds == ps) {   // both ascending
        ++exact;
        continue;
      }
      uint32_t common = 0;
      for (uint32_t x : ds) common += std::count(ps.begin(), ps.end(), x) ? 1 : 0;
      const float gap = bits(drow[kr::kSel + 8]) - bits(drow[kr::kSel + 9]);
      if (gap == 0.0f)
        ++undet;
      else if (gap <= tie_tol() && common == 8)
        ++ties;
      else if (++bad <= 10)
        std::printf("   L%u row %u: expert sets differ beyond a near-tie (decode gap %.3e, %u in common)\n", l, p,
                    double(gap), common);
    }
  }
  const Snap dec = snap(e, n, false, true);
  const double lcos = cos_f32(a.logits.data(), dec.logits.data(), a.logits.size());
  std::vector<double> taps;
  for (uint32_t l = 0; l < d.layers; ++l)
    taps.push_back(cos_bf16(a.taps.data() + size_t(l) * d.hidden, dec.taps.data() + size_t(l) * d.hidden, d.hidden));
  std::vector<double> ts = taps;
  std::sort(ts.begin(), ts.end());
  double sworst = 1.0, cworst = 1.0;
  for (size_t m = 0; m < a.states.size(); ++m) {
    sworst = std::min(sworst, cos_f32(a.states[m].data(), dec.states[m].data(), a.states[m].size()));
    for (uint32_t j = 0; j < 3; ++j)
      cworst = std::min(cworst, cos_bf16(a.conv[m].data() + size_t(j) * d.m_conv_dim,
                                         dec.conv[m].data() + size_t(j) * d.m_conv_dim, d.m_conv_dim));
  }
  std::vector<double> kv;
  for (size_t al = 0; al < a.kv.size(); ++al)
    for (uint32_t p = 0; p < n; ++p)
      for (uint32_t h = 0; h < 2; ++h)
        kv.push_back(cos_bf16(a.kv[al].data() + (size_t(h) * n + p) * d.kv_n(), dec.kv[al].data() + (size_t(h) * n + p) * d.kv_n(),
                              d.kv_n()));
  std::sort(kv.begin(), kv.end());
  const double kv_med = kv.empty() ? 1.0 : kv[kv.size() / 2], kv_p01 = kv.empty() ? 1.0 : kv[kv.size() / 100];
  const bool f4 = lcos >= 0.9995 && ts[ts.size() / 2] >= 0.9998 && ts.front() >= 0.99 && sworst >= 0.999 && cworst >= 0.999 &&
                  kv_med >= 0.9998 && kv_p01 >= 0.99;
  std::printf("4. prefill vs decode's fill (%u positions): logits cosine %.7f (bar 0.9995); h taps median %.7f / min %.6f "
              "(bars 0.9998 / 0.99); SSM states worst %.7f, conv columns worst %.7f (bars 0.999); K / V rows median %.7f, "
              "p01 %.6f%s\n", n, lcos, ts[ts.size() / 2], ts.front(), sworst, cworst, kv_med, kv_p01,
              f4 ? "" : "  ** FAIL **");
  std::printf("   h taps per block:");
  for (uint32_t l = 0; l < d.layers; ++l) std::printf(" %u:%.5f", l, taps[l]);
  std::printf("\n5. routing on the last chunk's %llu (row, MoE block) rows: %llu exact sets, %llu near-ties (gap <= %.1e), "
              "%llu undetermined (0 gap), %llu beyond%s\n", (unsigned long long)rows, (unsigned long long)exact,
              (unsigned long long)ties, double(tie_tol()), (unsigned long long)undet, (unsigned long long)bad,
              bad == 0 ? "" : "  ** FAIL **");
  ok = ok && f4 && bad == 0;

  // ---- 6. tokens, tie-aware --------------------------------------------------------------------------------------------
  std::vector<uint32_t> dtok(kGen);
  std::vector<bool> determined(kGen);
  for (uint32_t j = 0; j < kGen; ++j) {   // decode side: the state after ingest(prompt) is still there
    const std::vector<float> lg = e.read_logits();
    std::vector<float> s(lg.begin(), lg.begin() + d.vocab_used);
    std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<float>());
    const float ulp = std::fabs(common::bf16_to_f32(uint16_t(common::f32_to_bf16(s[0]) + 1)) - s[0]);
    determined[j] = s[0] - s[1] >= ulp;
    dtok[j] = e.pending();
    e.ingest({dtok[j]});
  }
  uint32_t agree = 0, det = 0, det_agree = 0;
  e.reset();
  e.prefill(prompt);
  for (uint32_t j = 0; j < kGen; ++j) {
    const uint32_t got = e.pending();
    agree += got == dtok[j];
    if (determined[j]) {
      ++det;
      det_agree += got == dtok[j];
    }
    e.ingest({dtok[j]});   // teacher-forced on decode's token
  }
  std::printf("6. tokens: %u / %u agree; %u / %u determined rows agree%s\n", agree, kGen, det_agree, det,
              det_agree == det ? "" : "  ** FAIL **");
  ok = ok && det_agree == det;
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = bolt_rig::bolt_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no Nemotron 3.5 Bolt checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  bolt_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  bool pp2 = false;
  uint32_t split = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "bf16") {
      o.int8_head = false;
    } else if (x == "eager") {
      o.attn = rb::BoltAttn::Eager;
    } else if (x == "b16") {
      o.state = rb::BoltState::Bf16;
    } else if (x.rfind("layers:", 0) == 0) {
      o.layers = uint32_t(std::stoul(x.substr(7)));
    } else if (x == "pp2" || x.rfind("pp2:", 0) == 0) {
      pp2 = true;
      if (x.size() > 4) split = uint32_t(std::stoul(x.substr(4)));
    } else {
      std::fprintf(stderr, "bolt_prefill_test: unknown flag %s (bf16, eager, b16, layers:<N>, pp2[:<split>])\n", x.c_str());
      return 2;
    }
  }
  const model::BoltDesc pre = loader::bolt_checkpoint_desc(snapdir, o.layers);
  o.devices = pre.layers == model::bolt().layers ? 2u : 1u;   // the whole model needs both cards (Decision 4)
  if (!bolt_rig::fits(pre, o, true)) {
    std::printf("SKIP: %s (%u blocks) does not fit %u card(s) with the prefill scratch\n", snapdir.c_str(), pre.layers,
                o.devices);
    return 77;
  }
  const Ids base = bolt_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  bolt_rig::Rig rig;
  bolt_rig::build(rig, snapdir, o);
  rb::BoltEngine& e = *rig.eng;
  std::printf("bolt_prefill_test: %zu ids, %u blocks on %u card(s), %s head, %s attention, %s SSM state (both halves); %zu "
              "decode launches, %zu prefill launches a chunk\n", ids.size(), e.model().desc.layers, e.devices(),
              o.int8_head ? "int8" : "bf16", rb::bolt_attn_name(e.attention()), rb::bolt_state_name(e.state()), e.launches(),
              rb::prefill_chunk_launches(e.model().desc, e.model().place));
  int rc = run_all(e, ids);
  if (pp2 && e.devices() == 1) {   // 7. two cards bitwise one card
    e.reset();
    e.prefill(ids);
    const Snap one = snap(e, kN, true, false);
    bolt_rig::Options o2 = o;
    o2.devices = 2;
    o2.split = split;
    if (l0::Context::gpu_count() < 2 || !bolt_rig::fits(pre, o2, true)) {
      std::printf("7. pp2: SKIPPED (one GPU, or the placement does not fit)\n");
    } else {
      rig.eng.reset();
      bolt_rig::build(rig, snapdir, o2);
      rig.eng->reset();
      rig.eng->prefill(ids);
      const bool b = snap(*rig.eng, kN, true, false) == one;
      std::printf("7. pp2 (split %u, the copy hand-off a chunk): %s one card\n", rig.eng->split(), b ? "bitwise" : "DIFFERS from");
      if (!b) rc = 1;
    }
  }
  std::printf("bolt_prefill_test %s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
