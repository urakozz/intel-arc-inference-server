// Spec 18c: K2-Horizon's prefill on the card with the real checkpoint (box only, labels
// `checkpoint;prefill;k2`), everything but the golden gate itself (k2_golden_test's `prefill`
// mode is K2 on prefill):
//
//   mode `all` (default):
//     1. the walk: K2Engine::prefill appends prefill_chunk_launches(desc) = 2392 launches per
//        chunk and the head's 5; the plan's prefill term equals the allocation;
//     2. K3, determinism and replay, bitwise: two immediate prefills (reset between) and two
//        RECORDED ones (set_prefill_replay: B70_PREFILL_REPLAY's path) leave the same KV rows of
//        every layer, last logits and last-chunk route rows - no sum is atomic, the expert sums
//        are in fixed ascending-id order, a grouped GEMM row does not depend on its tile;
//     3. chunking: chunks of 64 against the default chunk - KV and logits bitwise, the route
//        rows of the positions both last chunks hold bitwise;
//     4. Review Focus 1, prefill KV against decode's M = 1 fill (ingest, one id per replay) of
//        the same prompt: per layer, per row, K and V cosine. Gated (PROPOSED, the box sets
//        them): every row of the dense layers 0-2 >= 0.999, all rows' median >= 0.9998 and 1st
//        percentile >= 0.99 - spec 8b's prefill-head bar family; bitwise is not expected (the
//        GEMMs' sum orders and the bf16 weight dequant differ from the int4 GEMVs'). A MoVA V
//        row whose routed expert set differs between the two runs (a near-tie flipped by that
//        rounding) is counted and printed, not gated.
//     5. routing, prefill against decode per position (the last chunk's rows): the expert sets
//        equal except where decode's selection margin (sel of the K-th minus the first not
//        taken) is within kTieTol (PROPOSED); matched weights within kWeightAbs;
//     6. tokens: prefill + 32 greedy against ingest + 32 greedy, ruling A26's rule - every row
//        whose decode-side top-2 gap is at least one bf16 ulp of the top logit agrees; both are
//        teacher-forced on decode's token.
//   mode `split` (prefill_split on K2, Review Focus 4): the prompt's first N = 2600 ids in ONE
//     prefill call against TWO calls split at 1, 7, 16, 63, 64, 1000, 2047, 2048, 2049 - KV of
//     every layer and the last logits. Every kernel of the walk is row-local or keyed to
//     ABSOLUTE positions (the linears, the flash attention's key tiles from 0, RoPE, the KV
//     writes; there is no GDN chunking on K2), so the argument predicts EVERY split bitwise;
//     gated: the multiples of 64 bitwise (plan 18c's bar), the others within the split test's
//     bars (Qwen3.8's l0 numbers, PROVISIONAL for K2), each printed as bitwise or not.
//
// B70_K2_ATTN=eager runs the reference-rounding attention in BOTH halves - prefill's
// k2_pf_attn.cl EAGER and decode's k2_attn_eager.cl (spec 18 §10.1), one variable, one parser
// (runtime::k2::k2_attn()) - so mode `all`'s comparison 4 / 5 / 6 is between matched variants
// (the run prints both). Matched is not bitwise: the two eager forms share the rounding points,
// not the softmax's exp / sum order (decode's is torch's own, prefill's its two-pass form).
//
// Spec 18e: under B70_KV_CACHE=int8 (the `_kv8` twins) the engine holds the int8 rotkv cache;
// the bitwise comparisons compare its int8 rows and fp16 scales byte for byte, the cosines its
// dequantised rows (snap()), so every gate above reads the same in either form.
//
// argv: <snapshot> <prompt ids> [all|split] [int8]. Exit 77 (SKIP) when the checkpoint is not
// here. Any legal ids (< 250624) do for these gates: tests/golden/prompts/long*.ids.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"   // spec 18e: the int8 cache dequantised for the comparisons
#include "kernels/k2_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/k2/k2_engine.h"
#include "runtime/k2/k2_prefill.h"
#include "runtime/k2/k2_sizes.h"

namespace {

using Ids = std::vector<uint32_t>;
constexpr uint32_t kMaxLen = 16384, kGen = 32;
constexpr float kTieTol = 2e-2f;          // PROPOSED: decode's selection margin below this: a near-tie
constexpr float kWeightAbs = 1.0f / 32;   // PROPOSED: |w_prefill - w_decode| on matched experts

Ids read_ids(const std::string& p) {
  std::ifstream f(p);
  CHECK(f.good());
  Ids ids;
  for (long long v; f >> v;) {
    CHECK(v >= 0 && v < 250624);
    ids.push_back(uint32_t(v));
  }
  CHECK(!ids.empty());
  return ids;
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
  std::vector<uint16_t> kv;       // [layer][K rows [0, n) | V rows [0, n)] as bf16 values
  std::vector<uint8_t> raw;       // spec 18e, int8 cache: [layer][K rows, K scales | V rows, V scales]
  std::vector<float> logits;
  std::vector<uint32_t> routes;   // the last chunk's (prefill) - empty for decode
  bool operator==(const Snap& o) const {
    return kv == o.kv && raw == o.raw && logits == o.logits && routes == o.routes;
  }
};

// The KV rows [0, n) of every layer. bf16 cache: the rows as stored. Spec 18e, int8 cache
// (B70_KV_CACHE=int8 twins): the int8 rows and fp16 scales byte for byte in `raw` (the bitwise
// comparisons), and in `kv` each value dequantised (rotated basis, the same rotation in every
// run) and rounded to bf16 - for the cosine comparisons, whose bars are far above bf16's 2^-9.
Snap snap(runtime::k2::K2Engine& e, l0::CmdList& imm, uint32_t n, bool routes) {
  Snap s;
  const runtime::k2::K2Buffers& b = e.buffers();
  const size_t row = b.desc.kv_n(), used = size_t(n) * row;
  s.kv.resize(size_t(b.desc.layers) * used * 2);
  if (b.kv_cache() == runtime::KvCache::Int8) {
    const runtime::KvLayout& lay = b.kv_lay;
    const size_t rb = size_t(n) * lay.row_bytes(), sb = size_t(n) * lay.scale_row_bytes();
    s.raw.resize(size_t(b.desc.layers) * 2 * (rb + sb));
    for (uint32_t l = 0; l < b.desc.layers; ++l) {
      const runtime::KvLayer L = b.kv_layer(l);
      for (uint32_t t = 0; t < 2; ++t) {
        uint8_t* dst = s.raw.data() + (size_t(l) * 2 + t) * (rb + sb);
        imm.copy(dst, t ? L.v : L.k, rb);
        imm.copy(dst + rb, t ? L.vs : L.ks, sb);
        std::vector<uint16_t> sc(sb / 2);
        std::memcpy(sc.data(), dst + rb, sb);
        uint16_t* out = s.kv.data() + size_t(l) * used * 2 + t * used;
        for (size_t i = 0; i < used; ++i)
          out[i] = common::f32_to_bf16(common::kv8::dequant(int8_t(dst[i]), sc[i / 128]));
      }
    }
  } else {
    for (uint32_t l = 0; l < b.desc.layers; ++l) {
      imm.copy(s.kv.data() + size_t(l) * used * 2, b.kv_k_layer(l), used * 2);
      imm.copy(s.kv.data() + size_t(l) * used * 2 + used, b.kv_v_layer(l), used * 2);
    }
  }
  s.logits = e.read_logits();
  if (routes) s.routes = e.read_prefill_routes();
  return s;
}

// The route row of (layer, which, chunk row) inside read_prefill_routes()'s buffer.
const uint32_t* pf_route(const std::vector<uint32_t>& r, uint32_t l, uint32_t which, uint32_t row) {
  return r.data() + runtime::k2::pf_route_at(l, which) / 4 + size_t(row) * 32;
}

struct KvCmp {
  double dense_worst = 1, median = 1, p01 = 1, worst = 1;
  uint32_t worst_layer = 0, worst_row = 0, below99 = 0;
};
KvCmp compare_kv(const model::K2Desc& d, const Snap& a, const Snap& b, uint32_t n) {
  KvCmp c;
  const size_t row = d.kv_n(), used = size_t(n) * row;
  std::vector<double> all;
  all.reserve(size_t(d.layers) * n);
  for (uint32_t l = 0; l < d.layers; ++l)
    for (uint32_t p = 0; p < n; ++p) {
      const size_t ok = size_t(l) * used * 2 + size_t(p) * row, ov = ok + used;
      const double v = std::min(cos_bf16(&a.kv[ok], &b.kv[ok], row), cos_bf16(&a.kv[ov], &b.kv[ov], row));
      all.push_back(v);
      if (d.is_dense(l)) c.dense_worst = std::min(c.dense_worst, v);
      if (v < c.worst) {
        c.worst = v;
        c.worst_layer = l;
        c.worst_row = p;
      }
      c.below99 += v < 0.99;
    }
  std::sort(all.begin(), all.end());
  c.median = all[all.size() / 2];
  c.p01 = all[all.size() / 100];
  return c;
}

int run_all(runtime::k2::K2Engine& e, l0::CmdList& imm, const Ids& prompt) {
  const model::K2Desc& d = *e.model().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;

  // ---- 1. the walk and the plan --------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  const size_t chunks = (n + runtime::k2::kPfC - 1) / runtime::k2::kPfC;
  CHECK_EQ(runtime::k2::prefill_chunk_launches(d), size_t(2392));
  CHECK_EQ(e.prefill_launches() - l0, chunks * runtime::k2::prefill_chunk_launches(d) +
                                          runtime::k2::kPrefillHeadLaunches);
  CHECK_EQ(e.pos(), n);
  CHECK_EQ(e.memory_use().prefill_scratch, runtime::k2::prefill_sizes(d).total());
  std::printf("1. walk: %zu chunk(s) x %zu + %zu launches; prefill scratch %.3f GB = the plan's\n%s\n", chunks,
              runtime::k2::prefill_chunk_launches(d), runtime::k2::kPrefillHeadLaunches,
              runtime::k2::prefill_sizes(d).total() / 1e9, e.memory_line().c_str());
  const Snap a = snap(e, imm, n, true);

  // ---- 2. determinism and replay -------------------------------------------------------
  e.reset();
  e.prefill(prompt);
  const bool same = snap(e, imm, n, true) == a;
  e.set_prefill_replay(true);
  e.reset();
  e.prefill(prompt);   // records
  const bool rec1 = snap(e, imm, n, true) == a;
  e.reset();
  e.prefill(prompt);   // replays the recordings
  const bool rec2 = snap(e, imm, n, true) == a;
  e.set_prefill_replay(false);
  std::printf("2. K3: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS",
              rec1 ? "bitwise" : "DIFFERS", rec2 ? "bitwise" : "DIFFERS");
  ok = ok && same && rec1 && rec2;

  // ---- 3. chunking -----------------------------------------------------------------------
  {
    e.reset();
    e.prefill(prompt, 64);
    const Snap c = snap(e, imm, n, true);
    bool routes_same = true;
    const uint32_t last64 = n - ((n - 1) / 64) * 64, lastd = n - ((n - 1) / runtime::k2::kPfC) * runtime::k2::kPfC;
    for (uint32_t p = n - std::min(last64, lastd); p < n; ++p)
      for (uint32_t l = d.dense_layers; l < d.layers; ++l)
        for (uint32_t w = 0; w < 2; ++w)
          routes_same = routes_same && std::equal(pf_route(c.routes, l, w, p - (n - last64)),
                                                  pf_route(c.routes, l, w, p - (n - last64)) + 32,
                                                  pf_route(a.routes, l, w, p - (n - lastd)));
    const bool kv_same = c.kv == a.kv && c.raw == a.raw && c.logits == a.logits;
    std::printf("3. chunks of 64 vs %u: KV + logits %s, shared route rows %s\n", runtime::k2::kPfC,
                kv_same ? "bitwise" : "DIFFER", routes_same ? "bitwise" : "DIFFER");
    ok = ok && kv_same && routes_same;
  }

  // ---- 4 / 5. against decode's fill -------------------------------------------------------
  e.reset();
  const uint32_t lastd = n - ((n - 1) / runtime::k2::kPfC) * runtime::k2::kPfC;
  uint64_t rows = 0, exact = 0, ties = 0, bad = 0, mova_flips = 0;
  float worst_w = 0;
  for (uint32_t p = 0; p < n; ++p) {
    e.ingest({prompt[p]});
    if (p < n - lastd) continue;
    const std::vector<uint32_t> dr = e.read_routes();
    for (uint32_t l = d.dense_layers; l < d.layers; ++l)
      for (uint32_t w = 0; w < 2; ++w) {
        const uint32_t K = w ? d.top_k : d.value_top_k;
        const uint32_t* drow = dr.data() + (w ? runtime::k2::moe_route_at(l) : runtime::k2::mova_route_at(l)) / 4;
        const uint32_t* prow = pf_route(a.routes, l, w, p - (n - lastd));
        ++rows;
        if (std::equal(drow, drow + K, prow)) {
          ++exact;
          for (uint32_t j = 0; j < K; ++j)
            worst_w = std::max(worst_w, std::fabs(bits(drow[8 + j]) - bits(prow[8 + j])));
          continue;
        }
        // decode's margin at the cut: the lowest selected sel (slots are in id order, so the
        // minimum over them) minus the first not taken (k2_moe.cl R_SEL / R_NEXT)
        float lowest = bits(drow[16]);
        for (uint32_t j = 1; j < K; ++j) lowest = std::min(lowest, bits(drow[16 + j]));
        const float margin = lowest - bits(drow[24]);
        if (w == 0) ++mova_flips;
        if (margin <= kTieTol)
          ++ties;
        else if (++bad <= 10)
          std::printf("   L%u %s row %u: expert sets differ beyond a near-tie (decode margin %.3e)\n", l,
                      w ? "MoE" : "MoVA", p, double(margin));
      }
  }
  const Snap dec = snap(e, imm, n, false);
  const KvCmp kc = compare_kv(d, a, dec, n);
  const bool kv_ok = kc.dense_worst >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
  std::printf("4. prefill KV vs decode's fill (%u rows x %u layers): dense layers' worst %.6f, median %.7f, "
              "p01 %.6f, %u rows < 0.99, worst %.4f (L%u row %u)%s\n", n, d.layers, kc.dense_worst, kc.median,
              kc.p01, kc.below99, kc.worst, kc.worst_layer, kc.worst_row, kv_ok ? "" : "  ** FAIL **");
  std::printf("5. routing on the last chunk's %llu (row, layer, router) rows: %llu exact, %llu near-ties "
              "(margin <= %.0e), %llu beyond (%llu of the differing rows MoVA); matched weights worst |dw| %.4f%s\n",
              (unsigned long long)rows, (unsigned long long)exact, (unsigned long long)ties,
              double(kTieTol), (unsigned long long)bad, (unsigned long long)mova_flips, double(worst_w),
              bad == 0 && worst_w <= kWeightAbs ? "" : "  ** FAIL **");
  ok = ok && kv_ok && bad == 0 && worst_w <= kWeightAbs;

  // ---- 6. tokens, ruling A26's rule ------------------------------------------------------
  std::vector<uint32_t> dtok(kGen);
  std::vector<bool> determined(kGen);
  {   // decode side: the state after ingest(prompt) is still in the engine
    runtime::Control* c = e.buffers().control.as<runtime::Control>();
    for (uint32_t j = 0; j < kGen; ++j) {
      const std::vector<float> lg = e.read_logits();
      std::vector<float> s(lg.begin(), lg.begin() + d.vocab_used);
      std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<float>());
      const float ulp = std::fabs(common::bf16_to_f32(uint16_t(common::f32_to_bf16(s[0]) + 1)) - s[0]);
      determined[j] = s[0] - s[1] >= ulp;
      dtok[j] = c->cur_token[0];
      e.ingest({dtok[j]});
    }
  }
  uint32_t agree = 0, det = 0, det_agree = 0;
  {
    e.reset();
    e.prefill(prompt);
    runtime::Control* c = e.buffers().control.as<runtime::Control>();
    for (uint32_t j = 0; j < kGen; ++j) {
      const uint32_t got = c->cur_token[0];
      agree += got == dtok[j];
      if (determined[j]) {
        ++det;
        det_agree += got == dtok[j];
      }
      e.ingest({dtok[j]});   // teacher-forced on decode's token
    }
  }
  std::printf("6. tokens: %u / %u agree; %u / %u determined rows agree%s\n", agree, kGen, det_agree, det,
              det_agree == det ? "" : "  ** FAIL **");
  ok = ok && det_agree == det;
  return ok ? 0 : 1;
}

int run_split(runtime::k2::K2Engine& e, l0::CmdList& imm, const Ids& all) {
  const model::K2Desc& d = *e.model().desc;
  const uint32_t N = std::min<uint32_t>(2600, uint32_t(all.size()));
  const Ids ids(all.begin(), all.begin() + N);
  e.reset();
  e.prefill(ids);
  const Snap ref = snap(e, imm, N, false);
  bool ok = true;
  for (uint32_t s : {1u, 7u, 16u, 63u, 64u, 1000u, 2047u, 2048u, 2049u}) {
    if (s >= N) continue;
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + s));
    e.prefill(Ids(ids.begin() + s, ids.end()));
    CHECK_EQ(e.pos(), N);
    const Snap got = snap(e, imm, N, false);
    const bool bitwise = got == ref;
    // Not bitwise: the split test's bars (Qwen3.8's l0 numbers, PROVISIONAL here).
    const double lg = cos_f32(got.logits.data(), ref.logits.data(), d.vocab_used);
    const KvCmp kc = compare_kv(d, ref, got, N);
    double shallow = 1.0;
    {
      const size_t row = d.kv_n(), used = size_t(N) * row;
      for (uint32_t l = 0; l < 4; ++l)
        for (uint32_t p = 0; p < N; ++p) {
          const size_t okk = size_t(l) * used * 2 + size_t(p) * row, ov = okk + used;
          shallow = std::min(shallow, std::min(cos_bf16(&got.kv[okk], &ref.kv[okk], row),
                                               cos_bf16(&got.kv[ov], &ref.kv[ov], row)));
        }
    }
    const bool bars = lg >= 0.9995 && shallow >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
    const bool pass = s % 64 == 0 ? bitwise : (bitwise || bars);
    std::printf("  split at %4u: %s | logits %.6f, layers 0-3 worst %.6f, median %.7f, p01 %.6f%s\n", s,
                bitwise ? "bitwise" : "not bitwise", lg, shallow, kc.median, kc.p01, pass ? "" : "  ** FAIL **");
    ok = ok && pass;
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = argc > 1 ? argv[1] : "";
  const std::string prompt = argc > 2 ? argv[2] : "tests/golden/prompts/long.ids";
  const std::string mode = argc > 3 ? argv[3] : "all";
  const bool int8 = argc > 4 && std::string(argv[4]) == "int8";
  CHECK(mode == "all" || mode == "split");
  try {
    (void)loader::resolve_snapshot(snapdir);
  } catch (const std::exception& ex) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", snapdir.c_str(), ex.what());
    return 77;
  }
  const Ids ids = read_ids(prompt);
  l0::Context ctx(0);
  runtime::k2::K2Engine e(ctx,
                          loader::load_k2(ctx, snapdir, kMaxLen,
                                          int8 ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint),
                          kMaxLen);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::printf("k2_prefill_test %s: %zu ids, lm_head %s, prefill attention %s, decode attention %s "
              "(B70_K2_ATTN; %zu decode launches), %s KV (B70_KV_CACHE)\n", mode.c_str(), ids.size(),
              int8 ? "int8" : "bf16", runtime::k2::prefill_attn_eager() ? "eager" : "flash",
              runtime::k2::k2_attn_name(runtime::k2::k2_attn()), e.step().kernel_count,
              runtime::kv_cache_name(e.kv_cache()));
  CHECK(runtime::k2::prefill_attn_eager() == (runtime::k2::k2_attn() == runtime::k2::K2Attn::Eager));
  CHECK_EQ(e.step().kernel_count, runtime::k2::decode_launches(*e.model().desc));
  const int rc = mode == "split" ? run_split(e, imm, ids) : run_all(e, imm, Ids(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 2600)));
  std::printf("k2_prefill_test %s %s\n", mode.c_str(), rc == 0 ? "OK" : "FAILED");
  return rc;
}
