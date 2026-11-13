// Spec 19b Task 3 on the card: the DFlash draft and projection lists, end to end, with the REAL drafter
// (z-lab/Qwen3.8-27B-DFlash2, loaded by loader::load_drafter in int8 or bf16) beside a SYNTHETIC target - a random
// embedding and a random int8 (and bf16) head of Qwen3.8's shapes, so no 17 GB checkpoint is needed - against the
// host twin tests/kernels/dflash_ref.h fed the same weights (the loader's own host forms) and the card's ring.
//
//   dflash_draft_test <drafter repo or dir> [int8|bf16|int4]  (all formats when none is given; SKIP 77 without it)
//
//   1. The lists capture to their formulas' launch counts (87 draft, 8 projection); 2053 positions of random taps
//      are projected 8 a replay (the ring wraps at 2048); the last replay's rows equal the twin's projection
//      (bitwise bf16 within an ulp: the GEMVs' contraction is the box compiler's).
//   2. A draft at p 2053 (rows reach the window's edge past the wrap), anchor 1234, K 7, T 0: the final-norm rows
//      (cosine >= 0.99999), the top-16 sets (tie-aware) and the scores (cosine >= 0.99999) against the twin, the walk
//      equal; ctrl.cur_token[1..7] = the drafts; the readback's k / anchor / pos.
//   3. Determinism: a second replay's readback is bitwise the first's.
//   4. K 3 by the control block alone (no recapture): rows past K masked, the twin at K 3 agrees.
//   5. A sampled walk (T 0.8, a 64-bit seed): the twin's choices, ties within 1e-5 excused.
//   6. Review Focus 3 at the real size: the same block over a 32768-id draft vocabulary (the compact int8 head
//      gathered by loader::gather_int8_tiled_rows, df_topk COMPACT) holding every candidate of the full run - the
//      candidates, unary logits, scores and walk BITWISE the full head's; with row 0's best id swapped out of V' (the
//      table and the compact head rewritten in place, same list) the first row differs.
//   7. The bf16 head (int8 drafter): the twin agrees as in 2.
//   8. Spec 28d: the int8 linears and heads bind gemm_sm_i8w (DPAS, the in-work-group split-K), whose order the twin
//      does not follow. For the int8 drafter 1, 2, 4, 5 and 7 are graded by the f32 JUDGE (judge_draft: the twin in
//      Numerics::Reference over fp32 weights) - gemm_sm passes when it is no further from f32 than the same lists
//      forced onto gemv_i8w (CaptureOptions::force_linear, the pre-28d binding), spec 28's N4 / N6 form; the twin
//      comparisons of gemm_sm stay printed. The forced lists keep the old gate against the twin (the projection's
//      bar 0, 2's bars); both lists timed as interleaved pairs. The bf16 drafter (only its int8 head moved) keeps the
//      twin's bars.
// It prints the draft list's host-timed replay (one fence; informational - r54.speed is the timed row).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/repack.h"
#include "kernels/dflash_kernels.h"
#include "kernels/dflash_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/snapshot.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kd = kernels::dflash;
constexpr uint32_t V = 248320, USED = 248077, H = 5120, M = kd::kM;

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
  float uni(float a) { return (float(next() & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f) * a; }
};

dfref::Lin host_lin(const loader::DFlashHostLinear& h) {
  dfref::Lin l;
  l.K = h.K;
  l.N = h.N;
  l.slices = h.slices;
  l.ksplit = kernels::gemv_bf16_tiling(h.N).ksplit;
  l.bf16 = h.bf16;
  l.i8 = h.i8;
  l.scales = h.scales;
  if (!h.int4.empty()) {
    l.contracted = true;
    l.q4.resize(size_t(h.K / 8) * h.N);
    l.s4.resize(size_t(h.K / 64) * h.N);
    for (uint32_t n = 0; n < h.N; ++n)
      for (uint32_t g = 0; g < h.K / 64; ++g) {
        const uint32_t* tile = h.int4.data() + (size_t(n / 16) * (h.K / 64) + g) * 136;
        for (uint32_t j = 0; j < 8; ++j) l.q4[size_t(g * 8 + j) * h.N + n] = tile[j * 16 + n % 16];
        l.s4[size_t(g) * h.N + n] = uint16_t(tile[128 + n % 16 / 2] >> (16 * (n % 2)));
      }
  }
  return l;
}

// The twin's weights: the loader's host forms of the same checkpoint (what was uploaded).
dfref::Weights twin_weights(const loader::DFlashCheckpoint& ck, loader::DraftFormat f) {
  const model::DFlashDesc& d = ck.desc();
  dfref::Weights w;
  dfref::Dims& D = w.d;
  D.version = d.version;
  D.layers = d.layers;
  D.H = d.hidden;
  D.NH = d.q_heads;
  D.NKV = d.kv_heads;
  D.HD = d.head_dim;
  D.I = d.inter;
  D.V = V;
  D.used = USED;
  D.M = M;
  D.ring = D.win = d.max_window();
  D.pb = kd::kAttnPb;
  D.taps = uint32_t(d.taps.size());
  D.th = ck.target_hidden();
  D.group = d.conv_group;
  D.ctaps = d.conv_taps;
  D.rank = d.sel_rank;
  D.topk = kd::kTopK;
  D.qs = kd::kQs;
  D.eps = d.eps;
  D.in_scale = d.in_scale;
  D.out_mult = d.out_mult;
  D.softcap = d.softcap;
  D.scale = float(1.0 / std::sqrt(double(d.head_dim)));
  D.theta = d.theta;
  D.causal = false;
  w.fc = host_lin(ck.fc(f));
  w.hidden_norm = ck.hidden_norm();
  w.final_norm = ck.final_norm();
  const std::vector<float> qk = ck.qk_norms(), cb = ck.conv_bases();
  for (uint32_t l = 0; l < d.layers; ++l) {
    using W = loader::DFlashLinear;
    dfref::Layer y;
    y.qkv = host_lin(ck.layer_linear(l, W::Qkv, f));
    y.o = host_lin(ck.layer_linear(l, W::O, f));
    y.gate_up = host_lin(ck.layer_linear(l, W::GateUp, f));
    y.down = host_lin(ck.layer_linear(l, W::Down, f));
    const std::vector<float> nr = ck.layer_norms(l);
    y.in_norm.assign(nr.begin(), nr.begin() + H);
    y.post_norm.assign(nr.begin() + H, nr.end());
    y.q_norm.assign(qk.begin() + loader::dflash_qk_norm_off(d, l, 0), qk.begin() + loader::dflash_qk_norm_off(d, l, 1));
    y.k_norm.assign(qk.begin() + loader::dflash_qk_norm_off(d, l, 1),
                    qk.begin() + loader::dflash_qk_norm_off(d, l, 1) + d.head_dim);
    if (d.v2()) {
      y.akp = host_lin(ck.layer_linear(l, W::AttnKproj, f));
      y.mkp = host_lin(ck.layer_linear(l, W::MlpKproj, f));
      y.conv_base.assign(cb.begin() + loader::dflash_conv_base_off(d, l, 0, 0),
                         cb.begin() + loader::dflash_conv_base_off(d, l, 0, 0) + size_t(2) * 2 * d.conv_taps * H);
    }
    w.layers.push_back(std::move(y));
  }
  if (d.v2()) {
    w.sel = host_lin(ck.selector_projection(f));
    w.pred.assign(ck.codebook(false), ck.codebook(false) + size_t(V) * d.sel_rank);
    w.succ.assign(ck.codebook(true), ck.codebook(true) + size_t(V) * d.sel_rank);
  }
  return w;
}

struct Env {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  void run(runtime::CapturedStep& s) {
    q.execute(s.list, &f);
    f.wait();
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, n * sizeof(T));
    return v;
  }
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  }
};

std::vector<uint16_t> taps_rows(uint32_t pos0, uint32_t n, uint32_t width) {
  std::vector<uint16_t> t(size_t(M) * width, 0);
  for (uint32_t r = 0; r < n; ++r) {
    Rng g(1000003ull * (pos0 + r) + 17);
    for (uint32_t i = 0; i < width; ++i) t[size_t(r) * width + i] = dfref::rne(g.uni(3.0f));
  }
  return t;
}

// The ring on the card as the twin's floats.
dfref::Ring read_ring(Env& e, const runtime::dflash::DFlashBuffers& b) {
  dfref::Ring r;
  const std::vector<uint16_t> k = e.read<uint16_t>(b.ring_k, b.ring_k.size() / 2),
                              v = e.read<uint16_t>(b.ring_v, b.ring_v.size() / 2);
  r.k.resize(k.size());
  r.v.resize(v.size());
  for (size_t i = 0; i < k.size(); ++i) {
    r.k[i] = dfref::bf(k[i]);
    r.v[i] = dfref::bf(v[i]);
  }
  return r;
}

// A draft's readback (o) and final-norm rows (h) against a twin's draft t.
struct Measure {
  double hc = 1, sc = 1;          // rows cosine (min over 0..K), scores cosine over the shared candidates
  int sets_bad = 0, walk_bad = 0, near = 0;
  uint32_t walk_at = 0;           // the first draft index whose id differs from the twin's (K: none)
  double walk_gap = 0;            // there: the twin's two keys' relative gap (the near-tie rule's measure, 1e-4)
};
Measure measure(const runtime::dflash::DFlashDraftOut& o, const std::vector<float>& h, const dfref::DraftOut& t,
                uint32_t p, uint32_t K, float T, uint64_t seed) {
  Measure m;
  m.walk_at = K;
  for (uint32_t r = 0; r <= K; ++r) m.hc = std::fmin(m.hc, dfref::cosine(&h[size_t(r) * H], &t.hidden[size_t(r) * H], H));
  std::vector<float> es, ts;
  for (uint32_t j = 0; j < K; ++j) {
    for (uint32_t c = 0; c < 16; ++c) {
      const uint32_t id = o.cand[j][c];
      const auto it = std::find(t.cand.begin() + j * 16, t.cand.begin() + (j + 1) * 16, id);
      if (it == t.cand.begin() + (j + 1) * 16) {
        // an engine-only candidate must be at the boundary: its value within 1e-3 of the twin's 16th
        if (std::fabs(o.unary[j][c] - t.unary[j * 16 + 15]) > 1e-3f * std::fmax(1.0f, std::fabs(t.unary[j * 16 + 15])))
          ++m.sets_bad;
      } else {
        es.push_back(o.scores[j][c]);
        ts.push_back(t.scores[size_t(it - t.cand.begin())]);
      }
    }
    if (o.ids[j] != t.ids[j]) {
      m.walk_at = j;
      // excused only at a near tie of the twin's own choice (sampled: of the Gumbel-noised scores - the device's
      // log / log1p are not correctly rounded)
      auto key = [&](uint32_t c) {
        float x = t.scores[j * 16 + c];
        if (T != 0.0f)
          x = x / T - std::log(-std::log1p(-dfref::murmur3_uniform(seed, p + j + dfref::kDraftNoiseSalt,
                                                                     t.cand[j * 16 + c])));
        return x;
      };
      const float a = key(t.chosen[j]), bb = key(o.chosen[j]);
      if (std::fabs(a - bb) <= 1e-4f * std::fmax(1.0f, std::fabs(a))) {
        ++m.near;
        break;   // the walks diverge from here
      }
      m.walk_gap = std::fabs(a - bb) / std::fmax(1.0f, std::fabs(a));
      ++m.walk_bad;
      break;
    }
  }
  m.sc = dfref::cosine(es.data(), ts.data(), es.size());
  return m;
}

// One draft on the card against the twin (Numerics::Engine: gemv_i8w's fp32 order). Returns failures.
int compare_draft(Env& e, const runtime::dflash::DFlashBuffers& b, const dfref::Weights& tw, const dfref::Ring& ring,
                  const std::vector<float>& rope, uint32_t anchor, uint32_t p, uint32_t K, float T, uint64_t seed,
                  const char* what) {
  const runtime::dflash::DFlashDraftOut& o = *b.readback();
  const dfref::DraftOut t = dfref::draft(tw, dfref::Numerics::Engine, ring, rope, anchor, p, K, T, seed);
  const Measure m = measure(o, e.read<float>(b.h, size_t(M) * H), t, p, K, T, seed);
  std::printf("%s (p %u, K %u, T %.1f): rows cos %.7f, scores cos %.7f, top-16 sets %s, walk %s%s\n", what, p, K,
              double(T), m.hc, m.sc, m.sets_bad ? "DIFF" : "=", m.walk_bad ? "DIFF" : "=", m.near ? " (a near tie)" : "");
  int bad = 0;
  bad += m.hc < 0.99999;
  bad += m.sc < 0.99999;
  bad += m.sets_bad != 0;
  bad += m.walk_bad != 0;
  bad += o.k != K || o.anchor != anchor || o.pos != p;
  return bad;
}

// Spec 28d: the twin's f32 judge - every linear as fp32 [N][K] (the int8 rows x their scales, the bf16 tiles widened)
// over fp32 activations (Numerics::Reference: dflash_ref.py's arithmetic up to summation order).
void fill_f32(dfref::Lin& l) {
  if (l.K == 0 || !l.f32.empty()) return;
  l.f32.resize(size_t(l.N) * l.K);
  dfref::parallel_cols(l.N, [&](uint32_t n0, uint32_t n1) {
    for (uint32_t n = n0; n < n1; ++n)
      for (uint32_t k = 0; k < l.K; ++k)
        l.f32[size_t(n) * l.K + k] =
            !l.i8.empty() ? float(l.i8[size_t(n / 16) * (l.K / 16) * 256 + size_t(k / 16) * 256 + (k % 16) * 16 + n % 16]) *
                                l.scales[n]
            : !l.q4.empty() ? common::f16_to_f32(l.s4[size_t(k / 64) * l.N + n]) *
                                  float(int((l.q4[size_t(k / 8) * l.N + n] >> (4 * (k % 8))) & 15u) - 8)
                          : dfref::bf(l.bf16[size_t(n / 16) * (l.K / 8) * 128 + size_t(k / 8) * 128 + (k % 8) * 16 + n % 16]);
  });
}
dfref::Weights judge_weights(const dfref::Weights& tw) {
  dfref::Weights j = tw;
  for (dfref::Lin* l : {&j.fc, &j.sel, &j.head}) fill_f32(*l);
  for (dfref::Layer& y : j.layers)
    for (dfref::Lin* l : {&y.qkv, &y.o, &y.gate_up, &y.down, &y.akp, &y.mkp}) fill_f32(*l);
  return j;
}
double cos_dist(double c) { return 1.0 - c; }
int g_sets_sm = 0, g_sets_gv = 0;   // judge_draft's top-16 candidates off the judge's sets, summed over the cases

// Spec 28d's restated gate (the int8 drafter; spec 28 N4 / N6's f32-judge form, tests/server/spec_judge.h): the
// default lists (gemm_sm_i8w) and the same lists forced onto gemv_i8w (the pre-28d binding) run on the same ring and
// control block, both graded against the f32 judge. gemm_sm passes when it is no further from f32 than gemv_i8w:
// rows / scores - cosine >= 0.99999 against the judge or a cosine distance at most golden::kCosSlack (2) x gemv's;
// top-16 sets - no more candidates off the judge's set (the boundary excused as above) than gemv's, SUMMED over every
// judged case (operator 2026-11-04: the drafter only proposes, and a set counts one boundary candidate either way -
// per case it was 3 against 2 once, 4 against 8 summed; checked at the end of main); the walk -
// identical to the judge's, or diverging no earlier than gemv's, or at a near tie of the judge's scores (1e-4). The
// readback / control checks as before. Leaves the default list's readback. Returns failures.
int judge_draft(Env& e, const runtime::dflash::DFlashBuffers& b, const dfref::Weights& tj, const dfref::Ring& ring,
                const std::vector<float>& rope, uint32_t anchor, uint32_t p, uint32_t K, float T, uint64_t seed,
                runtime::CapturedStep& sm, runtime::CapturedStep& gv, const char* what) {
  const dfref::DraftOut t = dfref::draft(tj, dfref::Numerics::Reference, ring, rope, anchor, p, K, T, seed);
  e.run(gv);
  const Measure g = measure(*b.readback(), e.read<float>(b.h, size_t(M) * H), t, p, K, T, seed);
  e.run(sm);
  const runtime::dflash::DFlashDraftOut& o = *b.readback();
  const Measure m = measure(o, e.read<float>(b.h, size_t(M) * H), t, p, K, T, seed);
  const double slack = 2.0;   // golden::kCosSlack (tests/golden/forced_judge.h)
  const bool rows = m.hc >= 0.99999 || cos_dist(m.hc) <= slack * cos_dist(g.hc);
  const bool scores = m.sc >= 0.99999 || cos_dist(m.sc) <= slack * cos_dist(g.sc);
  const bool sets = m.sets_bad <= g.sets_bad;   // recorded per case; the gate is the sum (main's end)
  g_sets_sm += m.sets_bad;
  g_sets_gv += g.sets_bad;
  const bool walk = m.walk_bad == 0 || m.walk_at >= g.walk_at;
  std::printf("%s (p %u, K %u, T %.1f) against the f32 judge: rows cos gemm_sm %.7f / gemv %.7f %s, scores cos %.7f / "
              "%.7f %s, sets off %d / %d %s (recorded), walk to %u / %u of %u%s %s%s\n", what, p, K, double(T), m.hc, g.hc,
              rows ? "ok" : "FAIL", m.sc, g.sc, scores ? "ok" : "FAIL", m.sets_bad, g.sets_bad, sets ? "ok" : "FAIL",
              m.walk_at, g.walk_at, K, m.near ? " (a near tie)" : "", walk ? "ok" : "FAIL",
              m.walk_bad ? (" (the twin's keys there " + std::to_string(m.walk_gap) + " apart, relative)").c_str() : "");
  int bad = !rows + !scores + !walk;
  bad += o.k != K || o.anchor != anchor || o.pos != p;
  return bad;
}

// The ring rows of positions [pos0, pos0 + n) of every layer, k then v, as one vector (for a cosine).
std::vector<float> ring_rows(const dfref::Ring& r, const dfref::Dims& d, uint32_t pos0, uint32_t n) {
  std::vector<float> v;
  for (uint32_t l = 0; l < d.layers; ++l)
    for (uint32_t i = 0; i < n; ++i) {
      const size_t base = (size_t(l) * d.ring + (pos0 + i) % d.ring) * d.kvn();
      v.insert(v.end(), r.k.begin() + base, r.k.begin() + base + d.kvn());
      v.insert(v.end(), r.v.begin() + base, r.v.begin() + base + d.kvn());
    }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& e) {
    std::printf("SKIP: no DFlash drafter (%s)\n", e.what());
    return 77;
  }
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int8, loader::DraftFormat::Bf16,
                                           loader::DraftFormat::Int4G64};
  if (argc > 2) {
    loader::DraftFormat f;
    if (std::string(argv[2]) == "int4") f = loader::DraftFormat::Int4G64;
    else CHECK(loader::parse_draft_format(argv[2], f));
    formats = {f};
  }
  dfref::threads = std::max(1u, std::thread::hardware_concurrency());
  Env e;
  const model::ModelDesc& q = model::qwen38();
  // the synthetic target: a random embedding and a random int8 head (tiled + row scales)
  std::vector<uint16_t> embed(size_t(V) * H);
  {
    Rng g(1);
    for (uint16_t& x : embed) x = dfref::rne(g.uni(0.05f));
  }
  std::vector<int8_t> head_q(size_t(V) * H);
  std::vector<float> head_s(V);
  {
    Rng g(2);
    for (int8_t& x : head_q) x = int8_t(int(g.next() % 255) - 127);
    for (float& s : head_s) s = 0.0008f + 0.0004f * (g.uni(1.0f) + 1.0f);
  }
  l0::Mem embed_d = e.upload(embed.data(), embed.size() * 2);
  loader::DeviceWeight head_d{e.upload(head_q.data(), head_q.size()),
                              std::make_unique<l0::Mem>(e.upload(head_s.data(), head_s.size() * 4)),
                              {H, V, 1, 0}, model::WeightKind::Int8};
  const loader::DFlashCheckpoint ck(snap);
  const model::DFlashDesc& d = ck.desc();
  const std::vector<float> rope = dfref::rope_table(d.theta, d.head_dim, 8192 + M);
  int bad = 0;

  for (loader::DraftFormat f : formats) {
    g_sets_sm = g_sets_gv = 0;
    std::printf("== drafter %s\n", loader::draft_format_name(f));
    const loader::DFlashWeights w = loader::load_drafter(e.ctx, snap, q, V, embed_d, f);
    dfref::Weights tw = twin_weights(ck, f);
    tw.embed = embed;
    tw.mask_row.resize(H);
    for (uint32_t i = 0; i < H; ++i) tw.mask_row[i] = dfref::bf(embed[size_t(d.mask_id) * H + i]);
    tw.head.K = H;
    tw.head.N = V;
    tw.head.i8 = head_q;
    tw.head.scales = head_s;

    runtime::dflash::DFlashTarget t;
    t.embed = &embed_d;
    t.head = &head_d;
    t.vocab_rows = V;
    t.vocab_used = USED;
    runtime::dflash::DFlashBuffers b(e.ctx, d, w.target_hidden, V, 8192, w.format);
    l0::Mem control(e.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    std::memset(control.ptr(), 0, sizeof(runtime::Control));
    // spec 28d: the int8 drafter's lists forced onto gemv_i8w (the pre-28d binding) and the f32 judge
    const bool quantised = f != loader::DraftFormat::Bf16;   // both quantised forms use the f32 judge
    // Spec 19 v3d Task 6: the twin (Numerics::Engine) is the scalar attention part's order, so the lists compared
    // with it - the bf16 drafter's and the forced gemv_i8w ones - bind the scalar part; the quantised drafters'
    // production lists bind the DPAS part (dflash_kernels_test holds it to the twin by cosine) and face the f32 judge.
    runtime::CaptureOptions lo;
    lo.dflash_attn_dpas = quantised;
    runtime::CapturedStep draft = runtime::dflash::build_dflash_draft(e.ctx, w, t, b, control, lo);
    runtime::CapturedStep proj = runtime::dflash::build_dflash_project(e.ctx, w, b);
    std::printf("launches: draft %zu, projection %zu\n", draft.kernel_count, proj.kernel_count);
    bad += draft.kernel_count != 87 || proj.kernel_count != 8;
    runtime::CaptureOptions gvo;
    gvo.force_linear = runtime::LinearForce::Gemv;
    gvo.dflash_attn_dpas = false;
    std::unique_ptr<runtime::CapturedStep> dg, pg;
    dfref::Weights tj;
    if (quantised) {
      dg = std::make_unique<runtime::CapturedStep>(runtime::dflash::build_dflash_draft(e.ctx, w, t, b, control, gvo));
      pg = std::make_unique<runtime::CapturedStep>(runtime::dflash::build_dflash_project(e.ctx, w, b, gvo));
      bad += dg->kernel_count != 87 || pg->kernel_count != 8;
      tj = judge_weights(tw);
    }

    // 1. the projection: 2053 positions, 8 a replay
    const uint32_t P = 2053, width = d.fc_k(w.target_hidden);
    for (uint32_t pos = 0; pos < P; pos += M) {
      const uint32_t n = std::min(M, P - pos);
      const std::vector<uint16_t> rows = taps_rows(pos, n, width);
      e.imm.copy(b.taps.ptr(), rows.data(), rows.size() * 2);
      b.ctl()->proj_pos = pos;
      b.ctl()->proj_n = n;
      e.run(proj);
    }
    const dfref::Ring ring = read_ring(e, b);
    if (f != loader::DraftFormat::Int4G64) {
      // 1w. the same 2053 positions at prefill width (project_wide: pf_gemm, three passes - 1000 rows on the small
      // tile, 1048 on pf_gemm's, 5 across the wrap) against the 8-row replays' ring: every slot of every layer,
      // per position and layer the K and the V row within 2e-3 relative L2, about a bf16 half-ulp (2^-9) of the
      // row's norm: the GEMMs' summation order flips some bf16 roundings (measured 2026-11-13: worst 1.33e-3 for
      // int8 and bf16, 99 % of the values bitwise); the 8-row ring is put back for the cases below.
      const size_t rb = b.ring_k.size(), kvn = tw.d.kvn();
      const std::vector<uint16_t> k8 = e.read<uint16_t>(b.ring_k, rb / 2), v8 = e.read<uint16_t>(b.ring_v, rb / 2);
      e.imm.fill(b.ring_k.ptr(), 0u, rb);
      e.imm.fill(b.ring_v.ptr(), 0u, rb);
      runtime::PrefillScratch ps(e.ctx, 8192, q, 2048);
      runtime::prefill::Context cx(e.ctx);
      runtime::prefill::KernelCache kc(e.ctx);
      l0::Mem staged(e.ctx, l0::MemKind::Device, size_t(2048) * width * 2);
      for (const auto [pos, n] : {std::pair<uint32_t, uint32_t>{0, 1000}, {1000, 1048}, {2048, P - 2048}}) {
        for (uint32_t r = 0; r < n; ++r)
          e.imm.copy(static_cast<uint8_t*>(staged.ptr()) + size_t(r) * width * 2, taps_rows(pos + r, 1, width).data(),
                     size_t(width) * 2);
        CHECK(runtime::dflash::project_wide(cx, kc, ps, w, b, staged.ptr(), pos, n));
        cx.wait();
      }
      const std::vector<uint16_t> kw = e.read<uint16_t>(b.ring_k, rb / 2), vw = e.read<uint16_t>(b.ring_v, rb / 2);
      size_t same = 0, over = 0;
      double worst = 0;
      for (const auto& [a8, aw] : {std::pair{&k8, &kw}, {&v8, &vw}})
        for (size_t row = 0; row < a8->size() / kvn; ++row) {
          double dd = 0, nn = 0;
          for (size_t i = row * kvn; i < (row + 1) * kvn; ++i) {
            const double x = dfref::bf((*a8)[i]), y = dfref::bf((*aw)[i]);
            same += (*a8)[i] == (*aw)[i];
            dd += (x - y) * (x - y);
            nn += x * x;
          }
          const double rel = nn > 0 ? std::sqrt(dd / nn) : (dd > 0 ? 1.0 : 0.0);
          worst = std::max(worst, rel);
          over += !(rel <= 2e-3);
        }
      std::printf("projection at prefill width vs 8 rows a replay, positions 0..%u: %zu of %zu ring values bitwise, "
                  "worst row relative L2 %.3g, %zu rows over 2e-3 (bar 0)\n", P - 1, same, rb, worst, over);
      bad += over != 0;
      e.imm.copy(b.ring_k.ptr(), k8.data(), rb);
      e.imm.copy(b.ring_v.ptr(), v8.data(), rb);
    }
    {
      dfref::Ring tr = dfref::make_ring(tw.d);
      const uint32_t pos0 = (P - 1) / M * M, n = P - pos0;
      const std::vector<uint16_t> rows = taps_rows(pos0, n, width);
      dfref::project(tw, dfref::Numerics::Engine, rows.data(), n, pos0, rope, tr);
      int off = 0, far = 0;
      for (uint32_t l = 0; l < d.layers; ++l)
        for (uint32_t r = 0; r < n; ++r) {
          const size_t base = (size_t(l) * tw.d.ring + (pos0 + r) % tw.d.ring) * tw.d.kvn();
          for (uint32_t i = 0; i < tw.d.kvn(); ++i)
            for (const auto* pr : {&ring.k, &ring.v}) {
              const auto& tv = pr == &ring.k ? tr.k : tr.v;
              const float a = (*pr)[base + i], bb = tv[base + i];
              if (a != bb) {
                ++off;
                if (std::fabs(a - bb) > std::fabs(bb) / 64.0f + 1e-6f) ++far;
              }
            }
        }
      std::printf("projection of positions %u..%u: %d ring values off by a bf16 ulp, %d further (%s)\n", pos0,
                  P - 1, off, far, quantised ? "gemm_sm: recorded, the f32 judge below decides" : "bar 0");
      if (!quantised) bad += far;
      if (quantised) {
        // the f32 judge's projection of the last block; gemm_sm's ring rows (above) and gemv_i8w's (the forced list
        // over the same taps; the rows depend only on the taps), then gemm_sm's restored
        dfref::Ring jr = dfref::make_ring(tw.d);
        dfref::project(tj, dfref::Numerics::Reference, rows.data(), n, pos0, rope, jr);
        const std::vector<float> want = ring_rows(jr, tw.d, pos0, n);
        e.imm.copy(b.taps.ptr(), rows.data(), rows.size() * 2);
        b.ctl()->proj_pos = pos0;
        b.ctl()->proj_n = n;
        e.run(*pg);
        const dfref::Ring rg = read_ring(e, b);
        int goff = 0, gfar = 0;
        for (uint32_t l = 0; l < d.layers; ++l)
          for (uint32_t r = 0; r < n; ++r) {
            const size_t base = (size_t(l) * tw.d.ring + (pos0 + r) % tw.d.ring) * tw.d.kvn();
            for (uint32_t i = 0; i < tw.d.kvn(); ++i)
              for (const auto* pr : {&rg.k, &rg.v}) {
                const auto& tv = pr == &rg.k ? tr.k : tr.v;
                const float a = (*pr)[base + i], bb = tv[base + i];
                if (a != bb) {
                  ++goff;
                  if (std::fabs(a - bb) > std::fabs(bb) / 64.0f + 1e-6f) ++gfar;
                }
              }
          }
        const std::vector<float> got_sm = ring_rows(ring, tw.d, pos0, n), got_gv = ring_rows(rg, tw.d, pos0, n);
        const double cs = dfref::cosine(got_sm.data(), want.data(), want.size()),
                     cg = dfref::cosine(got_gv.data(), want.data(), want.size());
        const bool ok = cs >= 0.99999 || cos_dist(cs) <= 2.0 * cos_dist(cg);
        std::printf("projection, forced gemv_i8w: %d ring values off by a bf16 ulp, %d further (bar 0); against the f32 "
                    "judge: ring cos gemm_sm %.7f / gemv %.7f %s\n", goff, gfar, cs, cg, ok ? "ok" : "FAIL");
        bad += gfar + !ok;
        e.run(proj);   // gemm_sm's rows back (the taps and the control block are still the last block's)
      }
    }

    // 2. a draft at p 2053
    runtime::Control* c = control.as<runtime::Control>();
    c->pos = P;
    c->n_active = 1;
    c->cur_token[0] = 1234;
    b.ctl()->k = 7;
    b.ctl()->temperature = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    e.run(draft);
    const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bad += quantised ? judge_draft(e, b, tj, ring, rope, 1234, P, 7, 0.0f, 0, draft, *dg, "greedy")
              : compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.0f, 0, "greedy");
    if (quantised) compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.0f, 0, "greedy, gemm_sm against the gemv_i8w twin (recorded)");
    for (uint32_t j = 0; j < 7; ++j) bad += c->cur_token[1 + j] != b.readback()->ids[j];
    const runtime::dflash::DFlashDraftOut first = *b.readback();

    // 3. determinism
    t0 = std::chrono::steady_clock::now();
    e.run(draft);
    const double ms2 = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool same = std::memcmp(&first, b.readback(), sizeof first) == 0;
    std::printf("replay: readback %s; the draft list %.3f / %.3f ms (host-timed, one fence)\n",
                same ? "bitwise identical" : "DIFFERS", ms, ms2);
    bad += !same;

    // 4. K 3 by the control block alone
    b.ctl()->k = 3;
    e.run(draft);
    bad += quantised ? judge_draft(e, b, tj, ring, rope, 1234, P, 3, 0.0f, 0, draft, *dg, "K 3")
              : compare_draft(e, b, tw, ring, rope, 1234, P, 3, 0.0f, 0, "K 3");
    if (f == loader::DraftFormat::Int4G64) {
      const auto reference = dfref::draft(tj, dfref::Numerics::Reference, ring, rope, 1234, P, 3, 0.0f, 0);
      for (bool body_gemv : {false, true}) {
        runtime::CaptureOptions mixed;
        mixed.force_linear = body_gemv ? runtime::LinearForce::Gemv : runtime::LinearForce::SmallM;
        mixed.force_dflash_head = body_gemv ? runtime::LinearForce::SmallM : runtime::LinearForce::Gemv;
        mixed.dflash_attn_dpas = !body_gemv;   // its base's attention part (dg binds the scalar one)
        auto step = runtime::dflash::build_dflash_draft(e.ctx, w, t, b, control, mixed);
        const auto& base = body_gemv ? *dg : draft;
        CHECK_EQ(step.labels.size(), base.labels.size());
        size_t changed = 0;
        for (size_t i = 0; i < step.labels.size(); ++i)
          if (step.labels[i] != base.labels[i]) {
            ++changed;
            CHECK(step.labels[i].find("_M7_") != std::string::npos);
          }
        CHECK_EQ(changed, size_t(1));   // the override changes only the head, never the body
        e.run(step);
        const auto m = measure(*b.readback(), e.read<float>(b.h, size_t(M) * H), reference, P, 3, 0.0f, 0);
        std::printf("DFLASH_LINEAR_DIAG body=%s head=%s K=3 rows_cos=%.9f scores_cos=%.9f sets_off=%d walk_to=%u\n",
                    body_gemv ? "gemv" : "gemm_sm", body_gemv ? "gemm_sm" : "gemv", m.hc, m.sc, m.sets_bad, m.walk_at);
      }
    }

    // 5. sampled
    b.ctl()->k = 7;
    b.ctl()->temperature = 0.8f;
    b.ctl()->seed_lo = 0x01234567u;
    b.ctl()->seed_hi = 0x89ABCDEFu;
    e.run(draft);
    {
      const uint64_t seed = (uint64_t(0x89ABCDEFu) << 32) | 0x01234567u;
      bad += quantised ? judge_draft(e, b, tj, ring, rope, 1234, P, 7, 0.8f, seed, draft, *dg, "sampled")
                : compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.8f, seed, "sampled");
    }
    b.ctl()->temperature = 0.0f;

    // 6. a 32768-id draft vocabulary holding every candidate of the greedy run (Review Focus 3)
    if (f == loader::DraftFormat::Int8) {
      std::set<uint32_t> keep;
      for (uint32_t j = 0; j < 7; ++j)
        for (uint32_t k = 0; k < 16; ++k) keep.insert(first.cand[j][k]);
      for (uint32_t id = 0; keep.size() < 32768; id += 7) keep.insert(id % USED);
      std::vector<uint32_t> ids(keep.begin(), keep.end());
      const uint32_t NV = uint32_t(ids.size());
      std::vector<int8_t> cq(size_t(NV) * H);
      std::vector<float> cs(NV);
      loader::gather_int8_tiled_rows(head_q.data(), head_s.data(), H, V, ids.data(), NV, cq.data(), cs.data());
      loader::DeviceWeight ch{e.upload(cq.data(), cq.size()),
                              std::make_unique<l0::Mem>(e.upload(cs.data(), cs.size() * 4)), {H, NV, 1, 0},
                              model::WeightKind::Int8};
      l0::Mem ids_d = e.upload(ids.data(), ids.size() * 4);
      runtime::dflash::DFlashTarget tc = t;
      tc.head = &ch;
      tc.ids = &ids_d;
      runtime::dflash::DFlashBuffers bc(e.ctx, d, w.target_hidden, NV, 8192, w.format);
      e.imm.copy(bc.ring_k.ptr(), b.ring_k.ptr(), b.ring_k.size());
      e.imm.copy(bc.ring_v.ptr(), b.ring_v.ptr(), b.ring_v.size());
      runtime::CapturedStep dc = runtime::dflash::build_dflash_draft(e.ctx, w, tc, bc, control, lo);
      bc.ctl()->k = 7;
      bc.ctl()->temperature = 0.0f;
      e.run(dc);
      const runtime::dflash::DFlashDraftOut& o = *bc.readback();
      const bool eq = std::memcmp(o.cand, first.cand, sizeof o.cand) == 0 &&
                      std::memcmp(o.unary, first.unary, sizeof o.unary) == 0 &&
                      std::memcmp(o.scores, first.scores, sizeof o.scores) == 0 &&
                      std::memcmp(o.ids, first.ids, sizeof o.ids) == 0;
      // swap row 0's best id out of V' (another id in its place, the table kept ascending), same list
      const uint32_t best = first.cand[0][0];
      uint32_t other = 1;
      while (keep.count(other)) other += 2;
      keep.erase(best);
      keep.insert(other);
      ids.assign(keep.begin(), keep.end());
      loader::gather_int8_tiled_rows(head_q.data(), head_s.data(), H, V, ids.data(), NV, cq.data(), cs.data());
      e.imm.copy(ch.mem.ptr(), cq.data(), cq.size());
      e.imm.copy(ch.scales->ptr(), cs.data(), cs.size() * 4);
      e.imm.copy(ids_d.ptr(), ids.data(), ids.size() * 4);
      e.run(dc);
      const bool differs = bc.readback()->cand[0][0] != best;
      std::printf("draft vocabulary (32768 ids, compact int8 head): every candidate in V' -> %s the full head's; "
                  "row 0's best removed -> %s\n", eq ? "BITWISE" : "NOT", differs ? "differs" : "SAME (wrong)");
      bad += !eq || !differs;
    }

    // 7. the bf16 head (int8 drafter only: one arm is enough for the head's form)
    if (f == loader::DraftFormat::Int8) {
      std::vector<uint16_t> hb(size_t(V) * H);
      {
        Rng g(3);
        std::vector<uint16_t> rm(size_t(V) * H);
        for (uint16_t& x : rm) x = dfref::rne(g.uni(0.05f));
        common::repack_bf16_tiled(rm.data(), H, V, hb.data());
      }
      loader::DeviceWeight hbd{e.upload(hb.data(), hb.size() * 2), nullptr, {H, V, 1, 0}, model::WeightKind::Bf16};
      runtime::dflash::DFlashTarget tb = t;
      tb.head = &hbd;
      runtime::CapturedStep db = runtime::dflash::build_dflash_draft(e.ctx, w, tb, b, control);
      runtime::CapturedStep dbg = runtime::dflash::build_dflash_draft(e.ctx, w, tb, b, control, gvo);
      b.ctl()->k = 7;
      dfref::Weights tjb = tj;
      tjb.head = dfref::Lin{};
      tjb.head.K = H;
      tjb.head.N = V;
      tjb.head.bf16 = hb;
      fill_f32(tjb.head);
      tjb.head.bf16.clear();
      bad += judge_draft(e, b, tjb, ring, rope, 1234, P, 7, 0.0f, 0, db, dbg, "bf16 head");
    }

    // 8. spec 28d: the int8 linears on gemm_sm_i8w (the default lists above) against the same lists forced onto
    // gemv_i8w (runtime::CaptureOptions::force_linear, the pre-28d binding): the forced draft agrees with the twin as
    // in 2 (its gate, kept), and both lists' host-timed replays as interleaved pairs after a warm-up - median of 20.
    if (f == loader::DraftFormat::Int8) {
      c->pos = P;
      c->cur_token[0] = 1234;
      b.ctl()->k = 7;
      b.ctl()->temperature = 0.0f;
      e.run(*dg);
      bad += compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.0f, 0, "greedy, forced gemv_i8w (the pre-28d gate)");
      const runtime::dflash::DFlashDraftOut forced = *b.readback();
      uint32_t same_ids = 0;
      for (uint32_t j = 0; j < 7; ++j) same_ids += forced.ids[j] == first.ids[j];
      std::printf("gemm_sm_i8w against gemv_i8w: %u / 7 greedy drafts the same\n", same_ids);
      auto timed = [&](runtime::CapturedStep& s) {
        const auto t1 = std::chrono::steady_clock::now();
        e.run(s);
        return 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      };
      b.ctl()->proj_pos = P - 8;
      b.ctl()->proj_n = 8;
      for (int i = 0; i < 3; ++i)
        for (runtime::CapturedStep* s : {&draft, dg.get(), &proj, pg.get()}) timed(*s);
      std::vector<double> td, tg, tp, tq;
      for (int i = 0; i < 20; ++i) {
        td.push_back(timed(draft));
        tg.push_back(timed(*dg));
        tp.push_back(timed(proj));
        tq.push_back(timed(*pg));
      }
      auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      std::printf("timing (host-timed replays, interleaved, median of 20): draft list gemm_sm %.3f ms / gemv_i8w %.3f ms "
                  "(%.3f ms saved); projection gemm_sm %.3f ms / gemv_i8w %.3f ms (%.3f ms saved)\n",
                  med(td), med(tg), med(tg) - med(td), med(tp), med(tq), med(tq) - med(tp));
    }
    // Recorded, not gated (operator go 2026-11-13 night): the count swings by 3-6 under attention changes at the
    // 1e-7 cosine level; the rows' cosine against the f32 judge gates.
    if (g_sets_sm + g_sets_gv > 0)
      std::printf("top-16 candidates off the f32 judge's sets, summed for %s: gemm_sm %d / gemv %d (recorded)\n",
                  loader::draft_format_name(f), g_sets_sm, g_sets_gv);
  }
  if (bad) {
    std::printf("dflash_draft_test: FAIL (%d)\n", bad);
    return 1;
  }
  std::printf("dflash_draft_test: PASS\n");
  return 0;
}
