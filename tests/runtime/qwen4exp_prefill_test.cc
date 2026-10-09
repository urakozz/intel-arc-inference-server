// Spec 21d (F4 on prefill, Review Focus 1, 2, 4, 5): Qwen4ExpEngine::prefill on the card with a checkpoint - box only
// (labels checkpoint;qwen4exp;prefill). The synthetic real-width checkpoints (tools/box_validate/qwen4exp_oracle.sh
// synth-ckpt: 4 layers - 3 GDN incl. the PLE layer, 1 QSA) carry it before Intel's; the golden gate itself is
// qwen4exp_golden_test's `prefill` mode.
//
//   qwen4exp_prefill_test <checkpoint> <ids> [all | split | pp | gdn] [pp2[:split]] [int8] [layers:N]
//
// The prompt: <ids> cycled to kN = 4500 positions - three chunks at the default width: 0..2047 (all dense), 2048..4095
// (rows 2048..2050 dense, the rest sparse: Review Focus 1's straddling chunk) and 4096..4499 (all sparse, a 404-row
// tail) - any legal ids do for these gates (tests/golden/prompts/q4exp_4k.ids).
//
//   mode `all` (default; one card, or two with pp2):
//     1. the walk: every chunk's launches are prefill_chunk_launches(desc, placement, pos, C) (they depend on its dense
//        / sparse rows), the head's kPrefillHeadLaunches; the plan's prefill term (prefill_sizes + the prefill link)
//        equals the allocation, per device;
//     2. F4, determinism and replay, bitwise: two immediate prefills (reset between) and two RECORDED ones
//        (set_prefill_replay: B70_PREFILL_REPLAY's path) leave the same state of every layer (KV, the compressed keys
//        and tail rings, the GDN state and conv rings, the PLE rings), the last chunk's route rows and selections, the
//        last logits and the first token - no sum is atomic, the sort is one work-group, the combine is decode's;
//     3. chunking: chunks of 64 against the default chunk - every layer's state, logits, the first token BITWISE (the
//        linears and the MoE are row-local, the flash tiles and the GDN's 64-row WY sub-chunks sit at absolute
//        multiples of 64, the indexer's blocks and rings are keyed to absolute positions); chunks of 16 printed (the
//        GDN's sub-chunks move: not asserted bitwise - plan 21d's "chunk 16 bitwise" assumed no GDN, recorded);
//     4. Review Focus 5, prefill against decode's M = 1 fill (ingest, one id per replay) of the same prompt, per
//        layer: K and V cosine per position (QSA), the compressed keys per block, the GDN state and conv ring, the PLE
//        conv ring - gated (PROPOSED, spec 18c's family): every row of the first QSA layer >= 0.999, all rows' median
//        >= 0.9998, 1st percentile >= 0.99; the tail and PLE id rings bitwise (raw keys / ids); printed per layer;
//     5. routes and selections on the last chunk's rows, prefill against decode per position: the expert sets equal
//        except where decode's p10 - p11 is within kTieTol (or 0: undetermined); the selected block sets equal except
//        where decode's 512th - 513th score gap is within kSelTol (PROPOSED);
//     6. tokens: prefill + 32 greedy against ingest + 32 greedy, ruling A26's rule - every row whose decode-side top-2
//        gap is at least one bf16 ulp of the top logit agrees; teacher-forced on decode's;
//     7. the injected run chunked == whole, bitwise: a fixed valid selection (the last 512 complete blocks + the tail)
//        fed by set_prefill_injector, one call against chunks of 64.
//   mode `split` (Review Focus 5): the kN ids in ONE prefill against TWO calls split at 64, 2048, 2051, 4097 - every
//     layer's state, the last logits, the first token. Gated: the multiples of 64 bitwise, the others within the
//     split test's bars (logits cosine >= 0.9995, state rows: first QSA layer >= 0.999, median >= 0.9998, p01 >= 0.99)
//     - each printed as bitwise or not.
//   mode `pp` (two GPUs): --pp 2 --pipeline-split <s> (pp2:<s>, default 2) against --pp 1 on the same checkpoint,
//     BITWISE: one prefill (every layer's state, logits, the first token, both Control blocks equal after the mirror),
//     recorded + replayed, a split continuation (2051 + rest), chunks of 64; each card's prefill launches are
//     prefill_device_launches'; P4 - device 0's half of a chunk dropped (drop_next_handoff): the prefill throws within
//     its bound naming device 1, the next prefill is refused until reset(), which recovers bitwise.
//   mode `gdn` (no checkpoint; argv[1] / argv[2] ignored): gdn_chunk_q4 (the ten launches, pf_gated_head_SIG) against
//     decode's gdn_step_M1 + prep_gated_head_M1_SIG over 2048 positions on the device - the self-consistency oracle
//     of spec 6 §6.3 (gdn_chunk_test's case 2; the band printed, a 2.5e-1 max-relative tripwire) - and 2048 in one
//     call against 32 x 64: state, ring and y BITWISE.
//
// B70_Q4_ATTN=eager runs the eager attention in BOTH halves (prefill's q4_pf_sparse_attn EAGER and decode's
// q4_qsa_attn_eager; one variable, one parser) - so mode `all`'s comparisons 4-6 are between matched variants.
// Exit 77 (SKIP) when the checkpoint is absent, does not fit, or pp needs two GPUs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/qwen4exp_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "loader/small_layout.h"
#include "qwen4exp_rig.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gdn.h"
#include "runtime/prefill/kernels.h"
#include "runtime/qwen4exp/qwen4exp_prefill_gdn.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
namespace kq = kernels::qwen4exp;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 4500, kGen = 32;
constexpr float kTieTol = 1e-3f;   // PROPOSED: decode's p10 - p11 below this: a routing near-tie
constexpr float kSelTol = 1e-3f;   // PROPOSED: decode's 512th - 513th block score below this: a selection near-tie

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

// What a prefill leaves that a gate compares: every layer's persistent bytes (read_layer_state), the logits and the
// first token, pos; with `last` the last chunk's route rows and selections.
struct Snap {
  uint32_t first_token = 0, pos = 0;
  std::vector<std::vector<uint8_t>> layers;
  std::vector<float> logits;
  std::vector<uint32_t> routes;
  std::vector<std::vector<uint32_t>> sel;   // per QSA layer
  bool state_equal(const Snap& o) const {
    return first_token == o.first_token && pos == o.pos && layers == o.layers && logits == o.logits;
  }
  bool operator==(const Snap& o) const { return state_equal(o) && routes == o.routes && sel == o.sel; }
};
Snap snap(rq::Qwen4ExpEngine& e, bool last) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  Snap s;
  const runtime::Control c = e.control(e.devices() - 1);
  s.first_token = c.cur_token[0];
  s.pos = c.pos;
  for (uint32_t l = 0; l < d.layers; ++l) s.layers.push_back(e.read_layer_state(l));
  s.logits = e.read_logits();
  if (last) {
    s.routes = e.read_prefill_routes();
    for (uint32_t l = 0; l < d.layers; ++l)
      if (d.is_qsa(l)) s.sel.push_back(e.read_prefill_selection(l));
  }
  return s;
}

// Per-layer state comparison (Review Focus 5's bars): a QSA layer's K / V rows per position and its compressed keys
// per block; a GDN layer's state (per head) and conv ring (per slot); the PLE layer's conv ring (per slot) - each a
// cosine "row". The tail and id rings are compared bitwise apart.
struct StateCmp {
  double first_qsa_worst = 1, median = 1, p01 = 1, worst = 1;
  uint32_t worst_layer = 0, below99 = 0;
  bool rings_exact = true;
  std::vector<double> layer_worst, layer_median;
};
StateCmp compare_state(const model::Qwen4ExpDesc& d, const Snap& a, const Snap& b, uint32_t n, uint32_t max_len) {
  StateCmp c;
  std::vector<double> all;
  bool first_qsa = true;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const auto* A = reinterpret_cast<const uint16_t*>(a.layers[l].data());
    const auto* B = reinterpret_cast<const uint16_t*>(b.layers[l].data());
    std::vector<double> per;
    const auto add = [&](double v) {
      per.push_back(v);
      all.push_back(v);
      c.below99 += v < 0.99;
      if (v < c.worst) {
        c.worst = v;
        c.worst_layer = l;
      }
    };
    size_t off = 0;   // in bf16 elements
    if (d.is_qsa(l)) {
      const size_t row = d.kv_n();
      for (uint32_t p = 0; p < n; ++p) add(std::min(cos_bf16(A + p * row, B + p * row, row),
                                                    cos_bf16(A + (size_t(max_len) + p) * row, B + (size_t(max_len) + p) * row, row)));
      off = size_t(max_len) * row * 2;
      for (uint32_t blk = 0; blk < n / 4; ++blk) add(cos_bf16(A + off + size_t(blk) * d.idx_dim, B + off + size_t(blk) * d.idx_dim, d.idx_dim));
      off += size_t(max_len / d.idx_compress) * d.idx_dim;
      c.rings_exact = c.rings_exact && std::memcmp(A + off, B + off, size_t(rq::kIdxTail) * d.idx_dim * 2) == 0;
      off += size_t(rq::kIdxTail) * d.idx_dim;
      if (first_qsa)
        for (double v : per) c.first_qsa_worst = std::min(c.first_qsa_worst, v);
      first_qsa = false;
    } else {
      const auto* sa = reinterpret_cast<const float*>(a.layers[l].data());
      const auto* sb = reinterpret_cast<const float*>(b.layers[l].data());
      const size_t head = size_t(d.gdn_head) * d.gdn_head;
      for (uint32_t h = 0; h < d.gdn_v_heads; ++h) add(cos_f32(sa + h * head, sb + h * head, head));
      off = rq::gdn_state_bytes_per_layer(d) / 2;
      for (uint32_t s = 0; s < rq::kConvRing; ++s)
        add(cos_bf16(A + off + size_t(s) * d.conv_rows(), B + off + size_t(s) * d.conv_rows(), d.conv_rows()));
      off += rq::conv_ring_bytes_per_layer(d) / 2;
    }
    if (l == d.ple_layer) {   // the PLE state: the id ring (exact), the conv ring (per slot)
      const auto* pa = reinterpret_cast<const uint8_t*>(A + off);
      const auto* pb = reinterpret_cast<const uint8_t*>(B + off);
      c.rings_exact = c.rings_exact && std::memcmp(pa, pb, rq::kPleRing * 4) == 0;
      const auto* ca = reinterpret_cast<const uint16_t*>(pa + rq::kPleConvOff);
      const auto* cb = reinterpret_cast<const uint16_t*>(pb + rq::kPleConvOff);
      for (uint32_t s = 0; s < rq::kPleRing; ++s) add(cos_bf16(ca + size_t(s) * d.hc_n(), cb + size_t(s) * d.hc_n(), d.hc_n()));
    }
    std::sort(per.begin(), per.end());
    c.layer_worst.push_back(per.empty() ? 1.0 : per.front());
    c.layer_median.push_back(per.empty() ? 1.0 : per[per.size() / 2]);
  }
  std::sort(all.begin(), all.end());
  c.median = all[all.size() / 2];
  c.p01 = all[all.size() / 100];
  return c;
}
bool state_ok(const StateCmp& c) { return c.first_qsa_worst >= 0.999 && c.median >= 0.9998 && c.p01 >= 0.99; }
void print_layers(const StateCmp& c) {
  std::printf("   per layer (worst / median):");
  for (size_t l = 0; l < c.layer_worst.size(); ++l) std::printf(" L%zu %.6f/%.7f", l, c.layer_worst[l], c.layer_median[l]);
  std::printf("\n");
}

const uint32_t* pf_route(const std::vector<uint32_t>& r, uint32_t l, uint32_t row) {
  return r.data() + rq::pf_route_at(l) / 4 + size_t(row) * rq::kRouteWords;
}

// A fixed valid selection for the injected run: identity below the cut, else the LAST 512 complete blocks + the tail.
void inject_rows(uint32_t pos, uint32_t rows, uint32_t* lists) {
  for (uint32_t m = 0; m < rows; ++m) {
    uint32_t* row = lists + size_t(m) * rq::kListRow;
    const uint32_t p = pos + m, n = (p + 1) / 4;
    uint32_t at = 0;
    if (n <= 512) {
      for (uint32_t q = 0; q <= p; ++q) row[at++] = q;
    } else {
      for (uint32_t b = n - 512; b < n; ++b)
        for (uint32_t j = 0; j < 4; ++j) row[at++] = b * 4 + j;
      for (uint32_t q = n * 4; q <= p; ++q) row[at++] = q;
    }
    row[rq::kCountWord] = at;
  }
}

int run_all(rq::Qwen4ExpEngine& e, const Ids& prompt) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;

  // ---- 1. the walk and the plan --------------------------------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  size_t want = rq::kPrefillHeadLaunches;
  for (uint32_t p = 0; p < n; p += rq::kPfC) want += rq::prefill_chunk_launches(d, e.model().placement, p, std::min(rq::kPfC, n - p));
  CHECK_EQ(e.prefill_launches() - l0, want);
  CHECK_EQ(e.pos(), n);
  for (uint32_t i = 0; i < e.devices(); ++i)
    CHECK_EQ(e.memory_use(i).prefill_scratch, rq::prefill_state(d, e.devices(), i, e.max_len()));
  std::printf("1. walk: %zu launches for %u ids (chunks at 0 / 2048 / 4096: %zu / %zu / %zu) + %zu for the head on %u "
              "device(s); prefill scratch %.3f GB a device = the plan's\n%s\n", e.prefill_launches() - l0, n,
              rq::prefill_chunk_launches(d, e.model().placement, 0, rq::kPfC),
              rq::prefill_chunk_launches(d, e.model().placement, 2048, rq::kPfC),
              rq::prefill_chunk_launches(d, e.model().placement, 4096, n - 4096), rq::kPrefillHeadLaunches, e.devices(),
              rq::prefill_sizes(d, e.max_len()).total() / 1e9, e.memory_line().c_str());
  const Snap a = snap(e, true);

  // ---- 2. determinism and replay ---------------------------------------------------------------------------------
  e.reset();
  e.prefill(prompt);
  const bool same = snap(e, true) == a;
  e.set_prefill_replay(true);
  e.reset();
  e.prefill(prompt);   // records
  const bool rec1 = snap(e, true) == a;
  e.reset();
  e.prefill(prompt);   // replays the recordings
  const bool rec2 = snap(e, true) == a;
  e.set_prefill_replay(false);
  std::printf("2. F4: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS", rec1 ? "bitwise" : "DIFFERS",
              rec2 ? "bitwise" : "DIFFERS");
  ok = ok && same && rec1 && rec2;

  // ---- 3. chunking ------------------------------------------------------------------------------------------------
  {
    e.reset();
    e.prefill(prompt, 64);
    const bool c64 = snap(e, false).state_equal(a);
    e.reset();
    e.prefill(prompt, 16);
    const Snap s16 = snap(e, false);
    const bool c16 = s16.state_equal(a);
    const StateCmp cmp = compare_state(d, a, s16, n, e.max_len());
    std::printf("3. chunks of 64 vs %u: every layer's state, logits, first token %s; chunks of 16: %s (first QSA layer "
                "worst %.6f, median %.7f, p01 %.6f, logits cosine %.7f - the GDN's 64-row sub-chunks move with the chunk)\n",
                rq::kPfC, c64 ? "bitwise" : "DIFFER", c16 ? "bitwise" : "not bitwise", cmp.first_qsa_worst, cmp.median,
                cmp.p01, cos_f32(a.logits.data(), s16.logits.data(), a.logits.size()));
    ok = ok && c64;
  }

  // ---- 4 / 5. against decode's fill --------------------------------------------------------------------------------
  e.reset();
  const uint32_t lastd = n - ((n - 1) / rq::kPfC) * rq::kPfC;
  uint64_t rows = 0, exact = 0, near = 0, undet = 0, bad = 0, srows = 0, sexact = 0, snear = 0, sbad = 0;
  for (uint32_t p = 0; p < n; ++p) {
    e.ingest({prompt[p]});
    if (p < n - lastd) continue;
    const uint32_t r = p - (n - lastd);
    const std::vector<uint32_t> dr = e.read_routes();
    for (uint32_t l = 0; l < d.layers; ++l) {
      const uint32_t* drow = dr.data() + size_t(l) * rq::kRouteWords;
      const uint32_t* prow = pf_route(a.routes, l, r);
      std::vector<uint32_t> g(drow, drow + d.top_k), w(prow, prow + d.top_k);
      std::sort(g.begin(), g.end());
      std::sort(w.begin(), w.end());
      ++rows;
      if (g == w) {
        ++exact;
        continue;
      }
      const float gap = bits(drow[kq::route::kP10]) - bits(drow[kq::route::kP11]);
      if (gap == 0.0f) ++undet;
      else if (gap <= kTieTol) ++near;
      else if (++bad <= 10) std::printf("   L%u row %u: expert sets differ beyond a near-tie (decode gap %.3e)\n", l, p, double(gap));
    }
    if (p > rq::kPfDenseLast) {   // the selections (rows with a cut)
      const std::vector<float> diag = e.read_selection_diag();
      for (uint32_t l = 0; l < d.layers; ++l) {
        if (!d.is_qsa(l)) continue;
        const uint32_t qi = d.qsa_before(l);
        const std::vector<uint32_t> ds = e.read_selection(l);
        const uint32_t* prow = a.sel[qi].data() + size_t(r) * rq::kListRow;
        ++srows;
        const uint32_t cnt = ds.back();
        if (cnt == prow[rq::kCountWord] && std::equal(ds.begin(), ds.begin() + cnt, prow)) {
          ++sexact;
          continue;
        }
        const float gap = diag[qi * 2] - diag[qi * 2 + 1];
        if (gap <= kSelTol) ++snear;
        else if (++sbad <= 10) std::printf("   QSA L%u row %u: selection differs beyond a near-tie (gap %.3e)\n", l, p, double(gap));
      }
    }
  }
  const Snap dec = snap(e, false);
  const StateCmp sc = compare_state(d, a, dec, n, e.max_len());
  const bool sok = state_ok(sc) && sc.rings_exact;
  std::printf("4. prefill state vs decode's fill (%u positions x %u layers): first QSA layer's worst %.6f, median %.7f, p01 "
              "%.6f, %u rows < 0.99, worst %.4f (L%u); tail / id rings %s%s\n", n, d.layers, sc.first_qsa_worst, sc.median,
              sc.p01, sc.below99, sc.worst, sc.worst_layer, sc.rings_exact ? "bitwise" : "DIFFER", sok ? "" : "  ** FAIL **");
  print_layers(sc);
  std::printf("5. the last chunk's rows: routing %llu (row, layer) - %llu exact, %llu near-ties, %llu exact ties, %llu "
              "beyond; selections %llu (row, QSA layer) - %llu exact, %llu near-ties, %llu beyond%s\n",
              (unsigned long long)rows, (unsigned long long)exact, (unsigned long long)near, (unsigned long long)undet,
              (unsigned long long)bad, (unsigned long long)srows, (unsigned long long)sexact, (unsigned long long)snear,
              (unsigned long long)sbad, bad == 0 && sbad == 0 ? "" : "  ** FAIL **");
  ok = ok && sok && bad == 0 && sbad == 0;

  // ---- 6. tokens, ruling A26's rule ----------------------------------------------------------------------------------
  std::vector<uint32_t> dtok(kGen);
  std::vector<bool> determined(kGen);
  for (uint32_t j = 0; j < kGen; ++j) {   // decode side: the state after ingest(prompt) is still there
    const std::vector<float> lg = e.read_logits();
    std::vector<float> s(lg.begin(), lg.begin() + d.vocab_used);
    std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<float>());
    const float ulp = std::fabs(common::bf16_to_f32(uint16_t(common::f32_to_bf16(s[0]) + 1)) - s[0]);
    determined[j] = s[0] - s[1] >= ulp;
    dtok[j] = e.control(e.devices() - 1).cur_token[0];
    e.ingest({dtok[j]});
  }
  uint32_t agree = 0, det = 0, det_agree = 0;
  e.reset();
  e.prefill(prompt);
  for (uint32_t j = 0; j < kGen; ++j) {
    const uint32_t got = e.control(e.devices() - 1).cur_token[0];
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

  // ---- 7. the injected run: chunked == whole -------------------------------------------------------------------------
  {
    e.set_prefill_injector([](uint32_t, uint32_t pos, uint32_t rws, uint32_t* lists) { inject_rows(pos, rws, lists); });
    e.set_injected_selection(true);
    e.reset();
    e.prefill(prompt);
    const Snap w = snap(e, false);
    e.reset();
    e.prefill(prompt, 64);
    const bool inj_ok = snap(e, false).state_equal(w);
    e.set_injected_selection(false);
    std::printf("7. the injected selection (the last 512 blocks + the tail): chunks of 64 == one call %s\n",
                inj_ok ? "bitwise" : "DIFFER");
    ok = ok && inj_ok;
  }
  return ok ? 0 : 1;
}

int run_split(rq::Qwen4ExpEngine& e, const Ids& prompt) {
  const model::Qwen4ExpDesc& d = e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  e.reset();
  e.prefill(prompt);
  const Snap whole = snap(e, false);
  bool ok = true;
  for (uint32_t s : {64u, 2048u, 2051u, 4097u}) {
    e.reset();
    e.prefill(Ids(prompt.begin(), prompt.begin() + s));
    e.prefill(Ids(prompt.begin() + s, prompt.end()));
    const Snap two = snap(e, false);
    const bool bitwise = two.state_equal(whole);
    const StateCmp c = compare_state(d, whole, two, n, e.max_len());
    const double lc = cos_f32(whole.logits.data(), two.logits.data(), whole.logits.size());
    const bool pass = s % 64 == 0 ? bitwise : (lc >= 0.9995 && state_ok(c));
    std::printf("split at %u: %s (logits cosine %.7f, state first QSA layer worst %.6f, median %.7f, p01 %.6f)%s\n", s,
                bitwise ? "bitwise" : "not bitwise", lc, c.first_qsa_worst, c.median, c.p01, pass ? "" : "  ** FAIL **");
    ok = ok && pass;
  }
  return ok ? 0 : 1;
}

int run_pp(const std::string& snap_dir, qwen4exp_rig::Options o, const Ids& prompt) {
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::Options one = o;
  one.devices = 1;
  qwen4exp_rig::build(rig, snap_dir, one);
  rig.eng->reset();
  rig.eng->prefill(prompt);
  const Snap a = snap(*rig.eng, false);
  rig.eng->reset();
  rig.eng->prefill(Ids(prompt.begin(), prompt.begin() + 2051));
  rig.eng->prefill(Ids(prompt.begin() + 2051, prompt.end()));
  const Snap a_split = snap(*rig.eng, false);
  rig.eng->reset();
  rig.eng->prefill(prompt, 64);
  const Snap a64 = snap(*rig.eng, false);
  if (o.split == 0) o.split = 2;
  o.devices = 2;
  qwen4exp_rig::build(rig, snap_dir, o);
  rq::Qwen4ExpEngine& e = *rig.eng;
  const model::Qwen4ExpDesc& d = e.model().desc;
  bool ok = true;
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches(0), l1 = e.prefill_launches(1);
  e.reset();
  e.prefill(prompt);
  size_t w0 = 0, w1 = rq::kPrefillHeadLaunches;
  for (uint32_t p = 0; p < prompt.size(); p += rq::kPfC) {
    const uint32_t C = std::min<uint32_t>(rq::kPfC, uint32_t(prompt.size()) - p);
    w0 += rq::prefill_device_launches(d, e.model().placement, 0, p, C);
    w1 += rq::prefill_device_launches(d, e.model().placement, 1, p, C);
  }
  CHECK_EQ(e.prefill_launches(0) - l0, w0);
  CHECK_EQ(e.prefill_launches(1) - l1, w1);
  const Snap b = snap(e, false);
  const runtime::Control c0 = e.control(0), c1 = e.control(1);
  const bool same = b.state_equal(a) && std::memcmp(&c0, &c1, sizeof c0) == 0;
  e.set_prefill_replay(true);
  e.reset();
  e.prefill(prompt);
  e.reset();
  e.prefill(prompt);
  const bool rep = snap(e, false).state_equal(a);
  e.set_prefill_replay(false);
  e.reset();
  e.prefill(Ids(prompt.begin(), prompt.begin() + 2051));
  e.prefill(Ids(prompt.begin() + 2051, prompt.end()));
  const bool split = snap(e, false).state_equal(a_split);
  e.reset();
  e.prefill(prompt, 64);
  const bool c64 = snap(e, false).state_equal(a64);
  std::printf("pp: split %u, launches %zu + %zu = prefill_device_launches'; --pp 2 == --pp 1: one prefill %s (Controls "
              "mirrored %s), replayed %s, 2051 + rest %s, chunks of 64 %s\n", e.split(), w0, w1,
              b.state_equal(a) ? "bitwise" : "DIFFERS", std::memcmp(&c0, &c1, sizeof c0) == 0 ? "equal" : "DIFFER",
              rep ? "bitwise" : "DIFFERS", split ? "bitwise" : "DIFFERS", c64 ? "bitwise" : "DIFFERS");
  ok = ok && same && rep && split && c64;
  // P4: device 0's half of a chunk dropped
  e.reset();
  e.drop_next_handoff();
  bool threw = false, named = false;
  try {
    e.prefill(prompt);
  } catch (const std::exception& ex) {
    threw = true;
    named = std::string(ex.what()).find("device 1") != std::string::npos;
  }
  bool refused = false;
  try {
    e.prefill(prompt);
  } catch (const std::exception&) {
    refused = true;
  }
  e.reset();
  e.prefill(prompt);
  const bool recovered = snap(e, false).state_equal(a);
  std::printf("P4: a dropped hand-off %s%s; the next prefill %s; after reset() %s\n", threw ? "throws" : "DOES NOT THROW",
              named ? " naming device 1" : "", refused ? "refused" : "NOT REFUSED", recovered ? "bitwise again" : "DIFFERS");
  ok = ok && threw && named && refused && recovered;
  return ok ? 0 : 1;
}

// ---- mode gdn: gdn_chunk_q4 against decode's gdn_step + prep_gated_head _SIG ---------------------------------------------
struct Band {
  double max_rel = 0, mean_rel = 0;
};
Band band(const std::vector<float>& got, const std::vector<float>& ref) {
  double rms = 0;
  for (float r : ref) rms += double(r) * r;
  rms = std::sqrt(rms / double(ref.size()));
  Band b;
  double sum = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    CHECK(std::isfinite(got[i]));
    const double r = std::fabs(double(got[i]) - ref[i]) / std::max(std::fabs(double(ref[i])), rms);
    b.max_rel = std::max(b.max_rel, r);
    sum += r;
  }
  b.mean_rel = sum / double(ref.size());
  return b;
}
std::vector<float> widen(const std::vector<uint16_t>& v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = common::bf16_to_f32(v[i]);
  return f;
}

int run_gdn() {
  constexpr uint32_t P = 2048, QKVZ = 16384, AB = 128, H = 48, MIX = 48 * 128, CONV = 10240;
  const size_t small_bytes = loader::kQwen38Small.gdn_block_bytes, state_elems = size_t(H) * 128 * 128;
  l0::Context ctx(0);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::prefill::Context cx(ctx);
  runtime::prefill::KernelCache kc(ctx);
  std::printf("qwen4exp_prefill_test gdn on %s (scan entry %s)\n", ctx.name().c_str(), runtime::prefill::gdn_scan_entry_name());
  // the long-memory fixture of gdn_chunk_test (negA in [-4, -1], dt_bias -4)
  std::mt19937 rng(0x21d);
  std::normal_distribution<float> q(0.0f, 0.5f), a(0.0f, 1.0f), conv(0.0f, 0.5f);
  std::uniform_real_distribution<float> negA(-4.0f, -1.0f), gw(0.5f, 1.5f);
  std::vector<float> qkvz(size_t(P) * QKVZ), ab(size_t(P) * AB, 0.0f), small(small_bytes / 4, 0.0f);
  for (float& v : qkvz) v = q(rng);
  for (uint32_t m = 0; m < P; ++m)
    for (uint32_t h = 0; h < 2 * H; ++h) ab[size_t(m) * AB + h] = a(rng);
  for (size_t i = 0; i < size_t(CONV) * 4; ++i) small[i] = conv(rng);
  for (uint32_t h = 0; h < H; ++h) {
    small[loader::kQwen38Small.gdn_off_nega / 4 + h] = negA(rng);
    small[loader::kQwen38Small.gdn_off_dtbias / 4 + h] = -4.0f;
  }
  uint16_t* gwp = reinterpret_cast<uint16_t*>(small.data()) + loader::kQwen38Small.gdn_off_gated_norm / 2;
  for (uint32_t i = 0; i < 128; ++i) gwp[i] = common::f32_to_bf16(gw(rng));
  const auto up = [&](const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  };
  l0::Mem dq = up(qkvz.data(), qkvz.size() * 4), da = up(ab.data(), ab.size() * 4), ds = up(small.data(), small_bytes);
  l0::Mem st(ctx, l0::MemKind::Device, state_elems * 4), rg(ctx, l0::MemKind::Device, size_t(16) * CONV * 2),
      y(ctx, l0::MemKind::Device, size_t(P) * MIX * 2);
  const rq::GdnScratchSizes gs = rq::gdn_scratch_sizes(rq::kPfC);
  l0::Mem scratch(ctx, l0::MemKind::Device, gs.total());
  const auto chunked = [&](uint32_t width, std::vector<float>& state, std::vector<uint16_t>& ring, std::vector<uint16_t>& ys) {
    imm.fill(st.ptr(), 0u, st.size());
    imm.fill(rg.ptr(), 0u, rg.size());
    ys.assign(size_t(P) * MIX, 0);
    for (uint32_t p = 0; p < P; p += width) {
      rq::GdnChunkBuffers b;
      b.qkvz = dq.as<float>() + size_t(p) * QKVZ;
      b.ab = da.as<float>() + size_t(p) * AB;
      b.state = st.as<float>();
      b.ring = rg.as<uint16_t>();
      b.small = ds.ptr();
      b.gated_norm_off = loader::kQwen38Small.gdn_off_gated_norm;
      b.y = y.as<uint16_t>();
      uint8_t* base = static_cast<uint8_t*>(scratch.ptr());
      size_t off = 0;
      b.xb = base + off; off += gs.xb;
      b.seed = base + off; off += gs.seed;
      b.g = base + off; off += gs.g;
      b.beta = base + off; off += gs.beta;
      b.A = base + off; off += gs.A;
      b.A2 = base + off; off += gs.A2;
      b.w = base + off; off += gs.w;
      b.u = base + off; off += gs.u;
      b.o = base + off;
      cx.reset_launches();
      rq::gdn_chunk_q4(cx, kc, b, p, std::min(width, P - p));
      CHECK_EQ(cx.launches(), rq::kGdnChunkLaunches);
      cx.wait();
      imm.copy(ys.data() + size_t(p) * MIX, y.ptr(), size_t(std::min(width, P - p)) * MIX * 2);
    }
    state.resize(state_elems);
    ring.resize(size_t(16) * CONV);
    imm.copy(state.data(), st.ptr(), st.size());
    imm.copy(ring.data(), rg.ptr(), rg.size());
  };
  std::vector<float> s_one, s_64;
  std::vector<uint16_t> r_one, r_64, y_one, y_64;
  chunked(rq::kPfC, s_one, r_one, y_one);
  chunked(64, s_64, r_64, y_64);
  const bool bit = s_one == s_64 && r_one == r_64 && y_one == y_64;
  // the device oracle: decode's gdn_step_M1 + prep_gated_head_M1_SIG, 2048 times
  std::vector<float> dev_state(state_elems);
  std::vector<uint16_t> dev_y(size_t(P) * MIX);
  {
    l0::Mem ctrl(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    l0::Mem o1(ctx, l0::MemKind::Device, size_t(MIX) * 4), y1(ctx, l0::MemKind::Device, size_t(MIX) * 2);
    imm.fill(st.ptr(), 0u, st.size());
    imm.fill(rg.ptr(), 0u, rg.size());
    runtime::Control* c = ctrl.as<runtime::Control>();
    *c = runtime::Control{};
    c->n_active = 1;
    l0::Module m_gdn(ctx, kernels::path(kq::gdn_variant(1)));
    l0::Kernel k_gdn = m_gdn.kernel("gdn_step");
    l0::Module m_gh(ctx, kernels::path(kq::gated_head_sig_variant(1)));
    l0::Kernel k_gh = m_gh.kernel("prep_gated_head");
    const void* gw_dev = static_cast<const uint8_t*>(ds.ptr()) + loader::kQwen38Small.gdn_off_gated_norm;
    for (uint32_t p = 0; p < P; ++p) {
      c->pos = p;
      const void* qrow = dq.as<float>() + size_t(p) * QKVZ;
      const void* arow = da.as<float>() + size_t(p) * AB;
      cx.launch(k_gdn, H, 4, 1,
                {runtime::prefill::PtrArg(ctrl.ptr()), runtime::prefill::PtrArg(qrow), runtime::prefill::PtrArg(arow),
                 runtime::prefill::PtrArg(ds.ptr()), runtime::prefill::PtrArg(rg.ptr()), runtime::prefill::PtrArg(st.ptr()),
                 runtime::prefill::PtrArg(o1.ptr())});
      cx.launch(k_gh, H, 1, 1,
                {runtime::prefill::PtrArg(qrow), runtime::prefill::PtrArg(o1.ptr()), runtime::prefill::PtrArg(gw_dev),
                 runtime::prefill::PtrArg(y1.ptr())});
      cx.wait();
      imm.copy(dev_y.data() + size_t(p) * MIX, y1.ptr(), size_t(MIX) * 2);
    }
    imm.copy(dev_state.data(), st.ptr(), st.size());
  }
  const Band bs = band(s_one, dev_state), by = band(widen(y_one), widen(dev_y));
  std::printf("gdn: 2048 in one call vs 32 x 64: state, ring, y %s; vs decode's gdn_step + prep_gated_head _SIG x 2048 "
              "(the self-consistency oracle): state max rel %.3e mean %.3e, y max rel %.3e mean %.3e (tripwire 2.5e-1)\n",
              bit ? "bitwise" : "DIFFER", bs.max_rel, bs.mean_rel, by.max_rel, by.mean_rel);
  return bit && bs.max_rel < 2.5e-1 && by.max_rel < 2.5e-1 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <checkpoint> <ids> [all|split|pp|gdn] [pp2[:split]] [int8] [layers:N]\n", argv[0]);
    return 2;
  }
  std::string mode = "all";
  qwen4exp_rig::Options o;
  o.max_len = kMaxLen;
  o.timeout_ms = 60000;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "all" || a == "split" || a == "pp" || a == "gdn") mode = a;
    else if (a == "pp2") o.devices = 2;
    else if (a.rfind("pp2:", 0) == 0) { o.devices = 2; o.split = uint32_t(std::strtoul(a.c_str() + 4, nullptr, 10)); }
    else if (a == "int8") o.int8_head = true;
    else if (a.rfind("layers:", 0) == 0) o.layers = uint32_t(std::strtoul(a.c_str() + 7, nullptr, 10));
    else {
      std::fprintf(stderr, "qwen4exp_prefill_test: unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (mode == "gdn") return run_gdn();
  std::string why;
  const std::string snap_dir = qwen4exp_rig::qwen4exp_snapshot(argv[1], &why);
  if (snap_dir.empty()) {
    std::printf("SKIP: %s (the synthetic checkpoints: qwen4exp_oracle.sh synth-ckpt; Intel's: r31.ple_convert)\n", why.c_str());
    return 77;
  }
  const Ids src = qwen4exp_rig::read_ids(argv[2], 248320);
  if (src.empty()) {
    std::printf("SKIP: no ids in %s\n", argv[2]);
    return 77;
  }
  Ids prompt(kN);
  for (uint32_t i = 0; i < kN; ++i) prompt[i] = src[i % src.size()];
  if ((mode == "pp" || o.devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two GPUs needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  if (mode == "pp") {
    o.prefill_timeout_ms = 30000;   // P4's dropped hand-off throws within this bound
    return run_pp(snap_dir, o, prompt);
  }
  qwen4exp_rig::Rig rig;
  qwen4exp_rig::build(rig, snap_dir, o);
  std::printf("%s: %u layers, %s dense, %s attention (B70_Q4_ATTN), %u device(s), max_len %u; prompt %u ids\n",
              snap_dir.c_str(), rig.eng->model().desc.layers, model::q4_form_name(rig.eng->model().desc.forms.dense),
              rq::q4_attn_name(rig.eng->attention()), rig.eng->devices(), rig.eng->max_len(), kN);
  const int rc = mode == "split" ? run_split(*rig.eng, prompt) : run_all(*rig.eng, prompt);
  std::puts(rc == 0 ? "qwen4exp_prefill_test OK" : "qwen4exp_prefill_test FAILED");
  return rc;
}
