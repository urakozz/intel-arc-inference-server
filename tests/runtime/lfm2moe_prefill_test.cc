// Spec 24d Task 3 (F4 on prefill, plan 24d Review Focus 1, 2, 5): Lfm2MoeEngine::prefill on one card with a
// checkpoint - box only (labels checkpoint;lfm2moe;prefill). The synthetic checkpoints (24b's make_synth.py --layers
// 4: dense conv 0, 1, attention + MoE 2, conv + MoE 3) carry it before 24q's RTN export.
//
//   lfm2moe_prefill_test <checkpoint> <ids> [bf16] [eager] [--fuse conv]
//
// The prompt: <ids> cycled to kN = 4500 positions (3 chunks of 2048, 2048, 404).
//
//   1. the walk: prefill_chunk_launches(desc, placement) per chunk and the head's 4; the plan's prefill term
//      (prefill_sizes) equals the allocation
//   2. F4, determinism and replay, bitwise: two immediate prefills (reset between) and two RECORDED ones
//      (set_prefill_replay: B70_PREFILL_REPLAY's path) leave the same KV rows of every attention layer, the conv ring
//      slots decode reads next, the last logits, the last chunk's route rows and the first token - no sum is atomic,
//      the combine is rank-ordered, a grouped GEMM row does not depend on its tile
//   3. Review Focus 1, the chunk boundary, bitwise: chunks of 64 against the default 2048, and the prompt split into
//      two prefill calls at 1, 2, 3, 4, 5, 64, 2047, 2048, 2049 - KV, the live conv slots, logits, first token; and the
//      first 300 ids in chunks of 1 and of 2 against one chunk (the seed covers every tap; the ring write covers rows
//      0 and 1); a prompt whose first chunk starts at position 0 reads zeros from the seed (every run starts at 0)
//   4. Review Focus 2, prefill against decode's M = 1 fill (ingest, one id per replay) of the same prompt: per
//      attention layer, per position, K and V cosine - gated (PROPOSED, spec 20d's family): every row of the first
//      attention layer >= 0.999, all rows' median >= 0.9998, 1st percentile >= 0.99; the conv layers' live slots
//      and every layer's h at the last row (the debug tap) to the min bar 0.99; printed per layer (bitwise is not
//      expected: the GEMMs' K order and the bf16 dequant differ from the int4 GEMVs')
//   5. routing, prefill against decode per position (the last chunk's rows): the expert sets equal except where
//      decode's selection gap (sel 4th - sel 5th) is within B70_LFM_TIE_TOL (4e-3, RULED) and one expert differs
//   6. Review Focus 5, tokens: prefill + 32 greedy against ingest + 32 greedy, tie-aware (every row whose decode-side
//      top-2 gap is at least one bf16 ulp of the top logit agrees); teacher-forced on decode's
//
// `eager` runs B70_LFM_ATTN=eager in BOTH halves (prefill's lfm_pf_attn EAGER and decode's lfm_attn_eager.cl) - the
// comparisons 4-6 between matched variants. `--fuse conv`: in_proj in the conv-tile layout, the _T conv build and the
// decode arm conv. Exit 77 (SKIP) when the checkpoint is absent or does not fit one card with the prefill scratch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/lfm2moe_kernels.h"
#include "lfm2moe_rig.h"
#include "runtime/control.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace {

namespace rl = runtime::lfm2moe;
namespace kr = kernels::lfm2moe::route;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 4500, kGen = 32;
float tie_tol() {
  const char* e = std::getenv("B70_LFM_TIE_TOL");
  return e && *e ? std::strtof(e, nullptr) : 4e-3f;   // RULED (spec 24 §12)
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

struct Snap {
  uint32_t first_token = 0, pos = 0;
  std::vector<std::vector<uint16_t>> kv;     // per attention layer: K rows then V rows, positions [0, n)
  std::vector<std::vector<uint16_t>> conv;   // per conv layer: the two slots decode reads next ((n-2) % 16, (n-1) % 16)
  std::vector<float> logits;
  std::vector<uint32_t> routes;              // the last chunk's (prefill) - empty for decode
  std::vector<uint16_t> taps;                // [layers][hidden]: each layer's output at the last row
  bool operator==(const Snap& o) const {
    return first_token == o.first_token && pos == o.pos && kv == o.kv && conv == o.conv && logits == o.logits &&
           routes == o.routes;
  }
};

Snap snap(rl::Lfm2MoeEngine& e, uint32_t n, bool routes, bool taps) {
  const model::Lfm2MoeDesc& d = e.model().desc;
  Snap s;
  const runtime::Control c = e.control(e.devices() - 1);
  s.first_token = c.cur_token[0];
  s.pos = c.pos;
  for (uint32_t a = 0; a < d.attn_layers(); ++a) {
    std::vector<uint16_t> k = e.read_kv(a, 0, n, false), v = e.read_kv(a, 0, n, true);
    k.insert(k.end(), v.begin(), v.end());
    s.kv.push_back(std::move(k));
  }
  for (uint32_t cl = 0; cl < d.conv_layers(); ++cl) {
    const std::vector<uint16_t> ring = e.read_conv_ring(cl);
    std::vector<uint16_t> live;
    for (uint32_t p : {n - 2, n - 1})
      live.insert(live.end(), ring.begin() + size_t(p % 16) * d.hidden, ring.begin() + size_t(p % 16 + 1) * d.hidden);
    s.conv.push_back(std::move(live));
  }
  s.logits = e.read_logits();
  if (routes) s.routes = e.read_prefill_routes();
  if (taps) s.taps = e.read_debug_h();
  return s;
}

const uint32_t* pf_route(const std::vector<uint32_t>& r, uint32_t l, uint32_t row) {
  return r.data() + rl::pf_route_at(l) / 4 + size_t(row) * rl::kRouteWords;
}

struct Bars {
  double first_attn_worst = 1, median = 1, p01 = 1, worst = 1;
  uint32_t worst_layer = 0, worst_pos = 0;
  std::vector<double> layer_worst, layer_median;
};
// K and V cosine per (attention layer, position).
Bars compare_kv(const model::Lfm2MoeDesc& d, const Snap& a, const Snap& b, uint32_t n) {
  Bars c;
  const size_t row = d.kv_n();
  std::vector<double> all;
  for (uint32_t al = 0; al < d.attn_layers(); ++al) {
    std::vector<double> per;
    for (uint32_t p = 0; p < n; ++p) {
      const uint16_t* ka = a.kv[al].data() + size_t(p) * row;
      const uint16_t* kb = b.kv[al].data() + size_t(p) * row;
      const double v = std::min(cos_bf16(ka, kb, row), cos_bf16(ka + size_t(n) * row, kb + size_t(n) * row, row));
      per.push_back(v);
      all.push_back(v);
      if (al == 0) c.first_attn_worst = std::min(c.first_attn_worst, v);
      if (v < c.worst) {
        c.worst = v;
        c.worst_layer = al;
        c.worst_pos = p;
      }
    }
    std::sort(per.begin(), per.end());
    c.layer_worst.push_back(per.front());
    c.layer_median.push_back(per[per.size() / 2]);
  }
  if (!all.empty()) {
    std::sort(all.begin(), all.end());
    c.median = all[all.size() / 2];
    c.p01 = all[all.size() / 100];
  }
  return c;
}

int run_all(rl::Lfm2MoeEngine& e, const Ids& prompt) {
  const model::Lfm2MoeDesc& d = e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;

  // ---- 1. the walk and the plan ------------------------------------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  const size_t chunks = (n + rl::kPfC - 1) / rl::kPfC;
  const size_t per_chunk = rl::prefill_chunk_launches(d, e.model().place);
  CHECK_EQ(e.prefill_launches() - l0, chunks * per_chunk + rl::kPrefillHeadLaunches);
  CHECK_EQ(e.pos(), n);
  CHECK_EQ(e.memory_use(0).prefill_scratch, rl::prefill_sizes(d).total());
  std::printf("1. walk: %zu chunk(s) x %zu + %zu launches; prefill scratch %.3f GB = the plan's\n%s\n", chunks, per_chunk,
              rl::kPrefillHeadLaunches, rl::prefill_sizes(d).total() / 1e9, e.memory_line().c_str());
  const Snap a = snap(e, n, true, true);

  // ---- 2. determinism and replay -----------------------------------------------------------------------------------
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
  std::printf("2. F4: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS",
              rec1 ? "bitwise" : "DIFFERS", rec2 ? "bitwise" : "DIFFERS");
  ok = ok && same && rec1 && rec2;

  // ---- 3. the chunk boundary ---------------------------------------------------------------------------------------
  {
    e.reset();
    e.prefill(prompt, 64);
    const Snap c = snap(e, n, false, false);
    Snap ref = a;
    ref.routes.clear();
    bool all_same = c == ref;
    std::printf("3. chunks of 64 vs %u: KV, the live conv slots, logits, first token %s\n", rl::kPfC,
                all_same ? "bitwise" : "DIFFER");
    for (uint32_t s : {1u, 2u, 3u, 4u, 5u, 64u, 2047u, 2048u, 2049u}) {
      e.reset();
      e.prefill(Ids(prompt.begin(), prompt.begin() + s));
      e.prefill(Ids(prompt.begin() + s, prompt.end()));
      CHECK_EQ(e.pos(), n);
      const bool b = snap(e, n, false, false) == ref;
      std::printf("   split at %4u: %s\n", s, b ? "bitwise" : "DIFFERS");
      all_same = all_same && b;
    }
    const uint32_t m = 300;
    const Ids head(prompt.begin(), prompt.begin() + m);
    e.reset();
    e.prefill(head);
    const Snap h1 = snap(e, m, false, false);
    for (uint32_t c1 : {1u, 2u}) {
      e.reset();
      e.prefill(head, c1);
      const bool b = snap(e, m, false, false) == h1;
      std::printf("   the first %u ids in chunks of %u vs one chunk: %s\n", m, c1, b ? "bitwise" : "DIFFER");
      all_same = all_same && b;
    }
    ok = ok && all_same;
  }

  // ---- 4 / 5. against decode's fill ----------------------------------------------------------------------------------
  e.reset();
  const uint32_t row0 = ((n - 1) / rl::kPfC) * rl::kPfC;   // the last chunk's first position
  uint64_t rows = 0, exact = 0, ties = 0, bad = 0;
  for (uint32_t p = 0; p < n; ++p) {
    e.ingest({prompt[p]});
    if (p < row0) continue;
    const std::vector<uint32_t> dr = e.read_routes();   // [moe layers][16]
    uint32_t mi = 0;
    for (uint32_t l = 0; l < d.layers; ++l) {
      if (d.is_dense(l)) continue;
      const uint32_t* drow = dr.data() + size_t(mi++) * rl::kRouteWords;
      const uint32_t* prow = pf_route(a.routes, l, p - row0);
      ++rows;
      std::vector<uint32_t> ds(drow + kr::kIds, drow + kr::kIds + 4), ps(prow + kr::kIds, prow + kr::kIds + 4);
      std::sort(ds.begin(), ds.end());
      std::sort(ps.begin(), ps.end());
      if (ds == ps) {
        ++exact;
        continue;
      }
      uint32_t common = 0;
      for (uint32_t x : ds) common += std::count(ps.begin(), ps.end(), x) ? 1 : 0;
      const float gap = bits(drow[kr::kSel + 3]) - bits(drow[kr::kSel + 4]);
      if (gap <= tie_tol() && common == 3)
        ++ties;
      else if (++bad <= 10)
        std::printf("   L%u row %u: expert sets differ beyond a near-tie (decode gap %.3e, %u in common)\n", l, p,
                    double(gap), common);
    }
  }
  const Snap dec = snap(e, n, false, true);
  const Bars kc = compare_kv(d, a, dec, n);
  const bool kv_ok = kc.first_attn_worst >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
  std::printf("4. prefill KV vs decode's fill (%u positions x %u attention layers): the first attention layer's worst "
              "%.6f, median %.7f, p01 %.6f, worst %.4f (attention layer %u pos %u)%s\n", n, d.attn_layers(),
              kc.first_attn_worst, kc.median, kc.p01, kc.worst, kc.worst_layer, kc.worst_pos, kv_ok ? "" : "  ** FAIL **");
  std::printf("   per attention layer (worst / median):");
  for (size_t l = 0; l < kc.layer_worst.size(); ++l) std::printf(" %zu:%.6f/%.7f", l, kc.layer_worst[l], kc.layer_median[l]);
  std::printf("\n");
  double conv_worst = 1.0, tap_worst = 1.0;
  for (size_t cl = 0; cl < a.conv.size(); ++cl)
    for (uint32_t sl = 0; sl < 2; ++sl)
      conv_worst = std::min(conv_worst, cos_bf16(a.conv[cl].data() + size_t(sl) * d.hidden,
                                                 dec.conv[cl].data() + size_t(sl) * d.hidden, d.hidden));
  std::printf("   h at the last row (the taps), per layer:");
  for (uint32_t l = 0; l < d.layers; ++l) {
    const double c = cos_bf16(a.taps.data() + size_t(l) * d.hidden, dec.taps.data() + size_t(l) * d.hidden, d.hidden);
    tap_worst = std::min(tap_worst, c);
    std::printf(" %u:%.5f", l, c);
  }
  std::printf("\n   the conv layers' live slots: worst cosine %.6f; the taps' worst %.6f (bar 0.99)\n", conv_worst, tap_worst);
  std::printf("5. routing on the last chunk's %llu (row, MoE layer) rows: %llu exact sets, %llu near-ties (gap <= %.1e), "
              "%llu beyond%s\n", (unsigned long long)rows, (unsigned long long)exact, (unsigned long long)ties,
              double(tie_tol()), (unsigned long long)bad, bad == 0 ? "" : "  ** FAIL **");
  ok = ok && kv_ok && bad == 0 && conv_worst >= 0.99 && tap_worst >= 0.99;

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
  const std::string snapdir = lfm2moe_rig::lfm2moe_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no LFM2.5-24B-A2B checkpoint at '%s'\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  lfm2moe_rig::Options o;
  o.max_len = kMaxLen;
  o.debug_tap = true;
  for (int i = 3; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "bf16") {
      o.int8_head = false;
    } else if (x == "eager") {
      o.attn = rl::LfmAttn::Eager;
    } else if (x == "--fuse" && i + 1 < argc) {
      o.fuse = rl::parse_fuse(argv[++i]);
    } else {
      std::fprintf(stderr, "lfm2moe_prefill_test: unknown flag %s (bf16, eager, --fuse <arms>)\n", x.c_str());
      return 2;
    }
  }
  const model::Lfm2MoeDesc pre = loader::lfm2moe_checkpoint_desc(snapdir);
  if (!lfm2moe_rig::fits_one_card(pre, o.int8_head, kMaxLen, true)) {
    std::printf("SKIP: %s (%u layers) does not fit one card with the prefill scratch\n", snapdir.c_str(), pre.layers);
    return 77;
  }
  const Ids base = lfm2moe_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  lfm2moe_rig::Rig rig;
  lfm2moe_rig::build(rig, snapdir, o);
  rl::Lfm2MoeEngine& e = *rig.eng;
  std::printf("lfm2moe_prefill_test: %zu ids, %u layers, %s head, %s attention (B70_LFM_ATTN, both halves), fuse %s; "
              "%zu decode launches\n", ids.size(), e.model().desc.layers, o.int8_head ? "int8" : "bf16",
              rl::lfm_attn_name(e.attention()), rl::lfm_fuse_name(e.fuse()).c_str(), e.launches());
  const int rc = run_all(e, ids);
  std::printf("lfm2moe_prefill_test %s\n", rc == 0 ? "OK" : "FAILED");
  return rc;
}
