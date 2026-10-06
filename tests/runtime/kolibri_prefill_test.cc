// Spec 20d (KL3 on prefill, Review Focus 4-5): KolibriEngine::prefill on the card with a checkpoint - box
// only (labels checkpoint;kolibri;prefill). The synthetic checkpoints (tools/box_validate/kolibri_oracle.sh
// synth: 5 real-width layers - 4 sliding, 1 full) carry it before spec 20b; the golden gate itself is
// kolibri_golden_test's `prefill` mode.
//
//   kolibri_prefill_test <checkpoint> <ids> [all | split | pp] [int8] [pp2[:split]]
//
// The prompt: <ids> cycled to kN = 4500 positions (3 chunks: the third starts at 4096, where every sliding
// layer's ring wraps) - any legal ids do for these gates (tests/golden/prompts/kolibri_bench.ids).
//
//   mode `all` (default; one card, or two with pp2):
//     1. the walk: prefill_chunk_launches(desc, placement) per chunk (2 + 45 x layers) and the head's 5;
//        the plan's prefill term (prefill_sizes + the prefill link) equals the allocation, per device;
//     2. K3, determinism and replay, bitwise: two immediate prefills (reset between) and two RECORDED ones
//        (set_prefill_replay: B70_PREFILL_REPLAY's path) leave the same KV rows of every full layer, the
//        ring of every sliding layer, the last logits, the last chunk's route rows and the first token -
//        no sum is atomic, the combine is ascending-id, a grouped GEMM row does not depend on its tile;
//     3. chunking: chunks of 16 against the default chunk - KV, rings, logits and the first token bitwise
//        (every kernel is row-local or keyed to ABSOLUTE positions: the linears, the flash tiles, RoPE,
//        the ring slots), the route rows of the positions both last chunks hold bitwise;
//     4. Review Focus 4, prefill against decode's M = 1 fill (ingest, one id per replay) of the same
//        prompt: per layer, per position the ring / cache still holds, K and V cosine - gated (PROPOSED,
//        spec 18c's family): every row of layer 0 >= 0.999, all rows' median >= 0.9998, 1st percentile
//        >= 0.99; printed per layer (bitwise is not expected: the GEMMs' sum orders and the bf16 dequant
//        differ from the int4 GEMVs');
//     5. routing, prefill against decode per position (the last chunk's rows): the expert sets equal except
//        where decode's selection margin (the lowest selected sel minus the first not taken) is within
//        kTieTol (PROPOSED); matched weights within kWeightAbs;
//     6. tokens: prefill + 32 greedy against ingest + 32 greedy, ruling A26's rule - every row whose
//        decode-side top-2 gap is at least one bf16 ulp of the top logit agrees; teacher-forced on decode's.
//   mode `split` (prefill_split_kolibri_test, Review Focus 5): the kN ids in ONE prefill call against TWO
//     calls split at 64, 1000, 2048, 4097 - KV, rings, the last logits and the first token. Gated: the
//     multiples of 64 bitwise (plan 20d's bar; the flash tiles and the 8-row groups line up), the others
//     within the split test's bars (logits cosine >= 0.9995, layers 0-1 every row >= 0.999, median >=
//     0.9998, p01 >= 0.99) - each printed as bitwise or not. On two cards with pp2.
//   mode `pp` (two GPUs; Review Focus 5's last clause): --pp 2 --pipeline-split <s> (pp2:<s>, default 3)
//     against --pp 1 on the same checkpoint, BITWISE: one prefill (KV, rings, routes, logits, the first
//     token, both Control blocks equal after the mirror), recorded + replayed, a split continuation
//     (1000 + rest), chunks of 16; each card's prefill launches are prefill_device_launches'; P4 - device 0's
//     half of a chunk dropped (drop_next_handoff): the prefill throws within its 5 s bound naming device 1,
//     the next prefill is refused until reset(), which recovers bitwise.
//
// B70_KOLIBRI_ATTN=eager runs the reference-rounding attention in BOTH halves (prefill's kol_pf_attn EAGER
// and decode's kol_attn_eager.cl; one variable, one parser) - so mode `all`'s comparisons 4-6 are between
// matched variants (the `_eager` twins).
// Exit 77 (SKIP) when the checkpoint is absent, does not fit the asked placement, or pp needs two GPUs.
#include <algorithm>
#include <chrono>
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
#include "kolibri_rig.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_sizes.h"

namespace {

namespace rk = runtime::kolibri;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 8192, kN = 4500, kGen = 32;
constexpr float kTieTol = 2e-2f;          // PROPOSED: decode's selection margin below this: a near-tie
constexpr float kWeightAbs = 1.0f / 32;   // PROPOSED: |w_prefill - w_decode| on matched experts

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
  std::vector<uint32_t> lo;                // per layer: the first position read
  std::vector<std::vector<uint16_t>> kv;   // per layer: K rows then V rows, positions [lo, n)
  std::vector<float> logits;
  std::vector<uint32_t> routes;            // the last chunk's (prefill) - empty for decode
  bool operator==(const Snap& o) const {
    return first_token == o.first_token && pos == o.pos && kv == o.kv && logits == o.logits && routes == o.routes;
  }
};

// Every layer's K / V over the positions it still holds after n: a full layer [0, n), a sliding layer the
// ring's last min(n, 4096) positions (read through the ring, KolibriEngine::read_kv).
Snap snap(rk::KolibriEngine& e, uint32_t n, bool routes) {
  const model::Kolibri1Desc& d = e.model().desc;
  Snap s;
  const runtime::Control c = e.control(e.devices() - 1);
  s.first_token = c.cur_token[0];
  s.pos = c.pos;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t lo = d.is_sliding(l) && n > model::Kolibri1Desc::kRing ? n - model::Kolibri1Desc::kRing : 0;
    std::vector<uint16_t> k = e.read_kv(l, lo, n - lo, false), v = e.read_kv(l, lo, n - lo, true);
    k.insert(k.end(), v.begin(), v.end());
    s.lo.push_back(lo);
    s.kv.push_back(std::move(k));
  }
  s.logits = e.read_logits();
  if (routes) s.routes = e.read_prefill_routes();
  return s;
}

const uint32_t* pf_route(const std::vector<uint32_t>& r, uint32_t l, uint32_t row) {
  return r.data() + rk::pf_route_at(l) / 4 + size_t(row) * rk::kRouteWords;
}

struct KvCmp {
  double layer0_worst = 1, median = 1, p01 = 1, worst = 1;
  uint32_t worst_layer = 0, worst_pos = 0, below99 = 0;
  std::vector<double> layer_worst, layer_median;
};
// Positions both snapshots hold of every layer (the later lo), K and V cosine per row.
KvCmp compare_kv(const model::Kolibri1Desc& d, const Snap& a, const Snap& b, uint32_t n) {
  KvCmp c;
  const size_t row = d.kv_n();
  std::vector<double> all;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t lo = std::max(a.lo[l], b.lo[l]);
    const size_t na = n - a.lo[l], nb = n - b.lo[l];
    std::vector<double> per;
    for (uint32_t p = lo; p < n; ++p) {
      const uint16_t* ka = a.kv[l].data() + size_t(p - a.lo[l]) * row;
      const uint16_t* kb = b.kv[l].data() + size_t(p - b.lo[l]) * row;
      const double v = std::min(cos_bf16(ka, kb, row), cos_bf16(ka + na * row, kb + nb * row, row));
      per.push_back(v);
      all.push_back(v);
      if (l == 0) c.layer0_worst = std::min(c.layer0_worst, v);
      if (v < c.worst) {
        c.worst = v;
        c.worst_layer = l;
        c.worst_pos = p;
      }
      c.below99 += v < 0.99;
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

void print_layers(const KvCmp& c) {
  std::printf("   per layer (worst / median):");
  for (size_t l = 0; l < c.layer_worst.size(); ++l) std::printf(" L%zu %.6f/%.7f", l, c.layer_worst[l], c.layer_median[l]);
  std::printf("\n");
}

int run_all(rk::KolibriEngine& e, const Ids& prompt) {
  const model::Kolibri1Desc& d = e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;

  // ---- 1. the walk and the plan --------------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  const size_t chunks = (n + rk::kPfC - 1) / rk::kPfC;
  const size_t per_chunk = rk::prefill_chunk_launches(d, e.model().placement);
  CHECK_EQ(per_chunk, size_t(2 + 45 * d.layers));
  CHECK_EQ(e.prefill_launches() - l0, chunks * per_chunk + rk::kPrefillHeadLaunches);
  CHECK_EQ(e.pos(), n);
  for (uint32_t i = 0; i < e.devices(); ++i)
    CHECK_EQ(e.memory_use(i).prefill_scratch,
             rk::prefill_sizes(d).total() + (e.devices() > 1 ? rk::pf_link_bytes(d, i) : 0));
  std::printf("1. walk: %zu chunk(s) x %zu + %zu launches on %u device(s); prefill scratch %.3f GB a device = the "
              "plan's\n%s\n", chunks, per_chunk, rk::kPrefillHeadLaunches, e.devices(),
              rk::prefill_sizes(d).total() / 1e9, e.memory_line().c_str());
  const Snap a = snap(e, n, true);

  // ---- 2. determinism and replay ---------------------------------------------------------------
  e.reset();
  e.prefill(prompt);
  const bool same = snap(e, n, true) == a;
  e.set_prefill_replay(true);
  e.reset();
  e.prefill(prompt);   // records
  const bool rec1 = snap(e, n, true) == a;
  e.reset();
  e.prefill(prompt);   // replays the recordings
  const bool rec2 = snap(e, n, true) == a;
  e.set_prefill_replay(false);
  std::printf("2. K3: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS",
              rec1 ? "bitwise" : "DIFFERS", rec2 ? "bitwise" : "DIFFERS");
  ok = ok && same && rec1 && rec2;

  // ---- 3. chunking --------------------------------------------------------------------------------
  {
    e.reset();
    e.prefill(prompt, 16);
    const Snap c = snap(e, n, true);
    const uint32_t last16 = n - ((n - 1) / 16) * 16, lastd = n - ((n - 1) / rk::kPfC) * rk::kPfC;
    bool routes_same = true;
    for (uint32_t p = n - last16; p < n; ++p)
      for (uint32_t l = 0; l < d.layers; ++l)
        routes_same = routes_same && std::equal(pf_route(c.routes, l, p - (n - last16)),
                                                pf_route(c.routes, l, p - (n - last16)) + rk::kRouteWords,
                                                pf_route(a.routes, l, p - (n - lastd)));
    const bool kv_same = c.kv == a.kv && c.logits == a.logits && c.first_token == a.first_token;
    std::printf("3. chunks of 16 vs %u: KV, rings, logits, first token %s; shared route rows %s\n", rk::kPfC,
                kv_same ? "bitwise" : "DIFFER", routes_same ? "bitwise" : "DIFFER");
    ok = ok && kv_same && routes_same;
  }

  // ---- 4 / 5. against decode's fill ---------------------------------------------------------------
  e.reset();
  const uint32_t lastd = n - ((n - 1) / rk::kPfC) * rk::kPfC;
  uint64_t rows = 0, exact = 0, ties = 0, bad = 0;
  float worst_w = 0;
  for (uint32_t p = 0; p < n; ++p) {
    e.ingest({prompt[p]});
    if (p < n - lastd) continue;
    const std::vector<uint32_t> dr = e.read_routes();
    for (uint32_t l = 0; l < d.layers; ++l) {
      const uint32_t* drow = dr.data() + size_t(l) * rk::kRouteWords;
      const uint32_t* prow = pf_route(a.routes, l, p - (n - lastd));
      ++rows;
      if (std::equal(drow, drow + d.top_k, prow)) {
        ++exact;
        for (uint32_t j = 0; j < d.top_k; ++j) worst_w = std::max(worst_w, std::fabs(bits(drow[8 + j]) - bits(prow[8 + j])));
        continue;
      }
      // decode's margin at the cut: the lowest selected sel (kol_moe.cl R_SEL) minus the first not taken (R_NEXT)
      float lowest = bits(drow[16]);
      for (uint32_t j = 1; j < d.top_k; ++j) lowest = std::min(lowest, bits(drow[16 + j]));
      const float margin = lowest - bits(drow[24]);
      if (margin <= kTieTol)
        ++ties;
      else if (++bad <= 10)
        std::printf("   L%u row %u: expert sets differ beyond a near-tie (decode margin %.3e)\n", l, p, double(margin));
    }
  }
  const Snap dec = snap(e, n, false);
  const KvCmp kc = compare_kv(d, a, dec, n);
  const bool kv_ok = kc.layer0_worst >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
  std::printf("4. prefill KV / rings vs decode's fill (%u positions x %u layers): layer 0's worst %.6f, median %.7f, "
              "p01 %.6f, %u rows < 0.99, worst %.4f (L%u pos %u)%s\n", n, d.layers, kc.layer0_worst, kc.median, kc.p01,
              kc.below99, kc.worst, kc.worst_layer, kc.worst_pos, kv_ok ? "" : "  ** FAIL **");
  print_layers(kc);
  std::printf("5. routing on the last chunk's %llu (row, layer) rows: %llu exact, %llu near-ties (margin <= %.0e), "
              "%llu beyond; matched weights worst |dw| %.4f%s\n", (unsigned long long)rows, (unsigned long long)exact,
              (unsigned long long)ties, double(kTieTol), (unsigned long long)bad, double(worst_w),
              bad == 0 && worst_w <= kWeightAbs ? "" : "  ** FAIL **");
  ok = ok && kv_ok && bad == 0 && worst_w <= kWeightAbs;

  // ---- 6. tokens, ruling A26's rule ------------------------------------------------------------------
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
  return ok ? 0 : 1;
}

int run_split(rk::KolibriEngine& e, const Ids& ids) {
  const model::Kolibri1Desc& d = e.model().desc;
  const uint32_t N = uint32_t(ids.size());
  e.reset();
  e.prefill(ids);
  const Snap ref = snap(e, N, false);
  bool ok = true;
  for (uint32_t s : {64u, 1000u, 2048u, 4097u}) {
    if (s >= N) continue;
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + s));
    e.prefill(Ids(ids.begin() + s, ids.end()));
    CHECK_EQ(e.pos(), N);
    const Snap got = snap(e, N, false);
    const bool bitwise = got == ref;
    const double lg = cos_f32(got.logits.data(), ref.logits.data(), d.vocab_used);
    const KvCmp kc = compare_kv(d, ref, got, N);
    const double shallow = std::min(kc.layer_worst[0], d.layers > 1 ? kc.layer_worst[1] : 1.0);
    const bool bars = lg >= 0.9995 && shallow >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
    const bool pass = s % 64 == 0 ? bitwise : (bitwise || bars);
    std::printf("  split at %4u: %s | logits %.6f, layers 0-1 worst %.6f, median %.7f, p01 %.6f%s\n", s,
                bitwise ? "bitwise" : "not bitwise", lg, shallow, kc.median, kc.p01, pass ? "" : "  ** FAIL **");
    ok = ok && pass;
  }
  return ok ? 0 : 1;
}

// One card against two, bitwise (Review Focus 5's last clause). Both engines on the same checkpoint.
int run_pp(kolibri_rig::Rig& rig, const std::string& snapdir, kolibri_rig::Options o, const Ids& ids) {
  struct Run {
    Snap one, rep, cont, c16;
  };
  const auto runs = [&](rk::KolibriEngine& e) {
    Run r;
    const uint32_t n = uint32_t(ids.size());
    e.reset();
    e.prefill(ids);
    r.one = snap(e, n, true);
    if (e.devices() == 2) {
      const runtime::Control c0 = e.control(0), c1 = e.control(1);
      CHECK(std::memcmp(&c0, &c1, sizeof c0) == 0);   // the mirror after the head
      for (uint32_t i = 0; i < 2; ++i) std::printf("   device %u: %zu prefill launches\n", i, e.prefill_launches(i));
    }
    e.set_prefill_replay(true);
    e.reset();
    e.prefill(ids);
    e.reset();
    e.prefill(ids);
    r.rep = snap(e, n, true);
    e.set_prefill_replay(false);
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + 1000));
    e.prefill(Ids(ids.begin() + 1000, ids.end()));
    r.cont = snap(e, n, false);
    e.reset();
    e.prefill(ids, 16);
    r.c16 = snap(e, n, false);
    return r;
  };
  kolibri_rig::Options o2 = o;   // two cards at the asked split (pp2:<s>), else 3 (the synthetic's 5 layers)
  o2.devices = 2;
  o2.split = o.split ? o.split : 3;
  o2.prefill_timeout_ms = 5000;   // P4's bound below
  o.devices = 1;
  o.split = 0;
  kolibri_rig::build(rig, snapdir, o);
  const Run one = runs(*rig.eng);
  kolibri_rig::build(rig, snapdir, o2);
  const model::Kolibri1Desc& d = rig.eng->model().desc;
  for (uint32_t i = 0; i < 2; ++i) CHECK(rig.eng->prefill_launches(i) == 0);
  const Run two = runs(*rig.eng);
  const bool a = two.one == one.one, b = two.rep == one.rep && two.rep == one.one, c = two.cont == one.cont,
             e = two.c16 == one.c16;
  // P4 on the prefill hand-off: device 0's half of the first chunk dropped - the prefill throws within its
  // bound (5 s here), the next one says reset() first, reset() recovers bitwise.
  bool p4 = false;
  {
    rk::KolibriEngine& en = *rig.eng;
    en.reset();
    en.drop_next_handoff();
    const auto t0 = std::chrono::steady_clock::now();
    std::string what;
    try {
      en.prefill(ids);
    } catch (const std::exception& ex) {
      what = ex.what();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool refused = false;
    try {
      en.prefill(ids);
    } catch (const std::exception&) {
      refused = true;
    }
    en.reset();
    en.prefill(ids);
    const bool back = snap(en, uint32_t(ids.size()), true) == one.one;
    p4 = !what.empty() && what.find("device 1") != std::string::npos && s < 30.0 && refused && back;
    std::printf("P4: a dropped prefill hand-off threw after %.1f s (\"%.120s\"); the next prefill %s; reset() "
                "recovers %s\n", s, what.c_str(), refused ? "refused" : "RAN", back ? "bitwise" : "DIFFERENTLY");
  }
  std::printf("pp: --pp 2 --pipeline-split %u vs --pp 1 (%u layers): one prefill %s, recorded + replayed %s, split "
              "continuation (1000 + rest) %s, chunks of 16 %s\n", rig.eng->split(), d.layers,
              a ? "bitwise" : "DIFFERS", b ? "bitwise" : "DIFFERS", c ? "bitwise" : "DIFFERS", e ? "bitwise" : "DIFFERS");
  return a && b && c && e && p4 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = kolibri_rig::kolibri_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s' (the synthetic sets: kolibri_oracle.sh synth; the real one: "
                "spec 20b)\n", argc > 1 ? argv[1] : "");
    return 77;
  }
  const std::string mode = argc > 3 ? argv[3] : "all";
  CHECK(mode == "all" || mode == "split" || mode == "pp");
  kolibri_rig::Options o;
  o.max_len = kMaxLen;
  for (int i = 4; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "int8") {
      o.int8_head = true;
    } else if (x == "pp2") {
      o.devices = 2;
    } else if (x.rfind("pp2:", 0) == 0) {
      o.devices = 2;
      o.split = uint32_t(std::strtoul(x.c_str() + 4, nullptr, 10));
    } else {
      std::fprintf(stderr, "kolibri_prefill_test: unknown flag %s (int8, pp2[:split])\n", x.c_str());
      return 2;
    }
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snapdir);
  if ((mode == "pp" || o.devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  if ((mode == "pp" || o.devices == 1) &&
      !rk::fits_one_card(pre, o.int8_head, size_t(32530000000ull), size_t(1.5e9), true)) {
    std::printf("SKIP: %s (%u layers) does not fit one card with the prefill scratch\n", snapdir.c_str(), pre.layers);
    return 77;
  }
  const Ids base = kolibri_rig::read_ids(argc > 2 ? argv[2] : "", pre.vocab);
  CHECK(!base.empty());
  Ids ids(kN);
  for (uint32_t i = 0; i < kN; ++i) ids[i] = base[i % base.size()];
  kolibri_rig::Rig rig;
  if (mode == "pp") {
    const int rc = run_pp(rig, snapdir, o, ids);
    std::printf("kolibri_prefill_test pp %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
  }
  kolibri_rig::build(rig, snapdir, o);
  rk::KolibriEngine& e = *rig.eng;
  std::printf("kolibri_prefill_test %s: %zu ids, %u layers (%s attention arm), lm_head %s, %s attention "
              "(B70_KOLIBRI_ATTN, both halves), %u device(s)%s; %zu decode launches\n", mode.c_str(), ids.size(),
              e.model().desc.layers, model::kol_attn_form_name(e.model().desc.attn), o.int8_head ? "int8" : "bf16",
              rk::kol_attn_name(e.attention()), e.devices(),
              e.devices() == 2 ? (std::string(", split ") + std::to_string(e.split())).c_str() : "", e.launches());
  const int rc = mode == "split" ? run_split(e, ids) : run_all(e, ids);
  std::printf("kolibri_prefill_test %s %s\n", mode.c_str(), rc == 0 ? "OK" : "FAILED");
  return rc;
}
