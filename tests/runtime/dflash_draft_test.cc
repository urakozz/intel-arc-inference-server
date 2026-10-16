// Spec 19b Task 3 on the card: the DFlash draft and projection lists, end to end, with the REAL drafter
// (z-lab/Qwen3.8-27B-DFlash2, loaded by loader::load_drafter in int8 or bf16) beside a SYNTHETIC target - a random
// embedding and a random int8 (and bf16) head of Qwen3.8's shapes, so no 17 GB checkpoint is needed - against the
// host twin tests/kernels/dflash_ref.h fed the same weights (the loader's own host forms) and the card's ring.
//
//   dflash_draft_test <drafter repo or dir> [int8|bf16]       (both formats when none is given; SKIP 77 without it)
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
// It prints the draft list's host-timed replay (one fence; informational - r54.speed is the timed row).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
#include "runtime/control.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"

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
  l.ksplit = kernels::gemv_bf16_tiling(h.N).ksplit;
  l.bf16 = h.bf16;
  l.i8 = h.i8;
  l.scales = h.scales;
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

// One draft on the card against the twin. Returns failures.
int compare_draft(Env& e, const runtime::dflash::DFlashBuffers& b, const dfref::Weights& tw, const dfref::Ring& ring,
                  const std::vector<float>& rope, uint32_t anchor, uint32_t p, uint32_t K, float T, uint64_t seed,
                  const char* what) {
  const runtime::dflash::DFlashDraftOut& o = *b.readback();
  const dfref::DraftOut t = dfref::draft(tw, dfref::Numerics::Engine, ring, rope, anchor, p, K, T, seed);
  const std::vector<float> h = e.read<float>(b.h, size_t(M) * H);
  double hc = 1;
  for (uint32_t r = 0; r <= K; ++r) hc = std::fmin(hc, dfref::cosine(&h[size_t(r) * H], &t.hidden[size_t(r) * H], H));
  std::vector<float> es, ts;
  int sets_bad = 0, walk_bad = 0, near = 0;
  for (uint32_t j = 0; j < K; ++j) {
    for (uint32_t c = 0; c < 16; ++c) {
      const uint32_t id = o.cand[j][c];
      const auto it = std::find(t.cand.begin() + j * 16, t.cand.begin() + (j + 1) * 16, id);
      if (it == t.cand.begin() + (j + 1) * 16) {
        // an engine-only candidate must be at the boundary: its value within 1e-3 of the twin's 16th
        if (std::fabs(o.unary[j][c] - t.unary[j * 16 + 15]) > 1e-3f * std::fmax(1.0f, std::fabs(t.unary[j * 16 + 15])))
          ++sets_bad;
      } else {
        es.push_back(o.scores[j][c]);
        ts.push_back(t.scores[size_t(it - t.cand.begin())]);
      }
    }
    if (o.ids[j] != t.ids[j]) {
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
        ++near;
        break;   // the walks diverge from here
      }
      ++walk_bad;
      break;
    }
  }
  const double sc = dfref::cosine(es.data(), ts.data(), es.size());
  std::printf("%s (p %u, K %u, T %.1f): rows cos %.7f, scores cos %.7f, top-16 sets %s, walk %s%s\n", what, p, K,
              double(T), hc, sc, sets_bad ? "DIFF" : "=", walk_bad ? "DIFF" : "=", near ? " (a near tie)" : "");
  int bad = 0;
  bad += hc < 0.99999;
  bad += sc < 0.99999;
  bad += sets_bad != 0;
  bad += walk_bad != 0;
  bad += o.k != K || o.anchor != anchor || o.pos != p;
  return bad;
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
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int8, loader::DraftFormat::Bf16};
  if (argc > 2) {
    loader::DraftFormat f;
    CHECK(loader::parse_draft_format(argv[2], f));
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
    runtime::dflash::DFlashBuffers b(e.ctx, d, w.target_hidden, V, 8192);
    l0::Mem control(e.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    std::memset(control.ptr(), 0, sizeof(runtime::Control));
    runtime::CapturedStep draft = runtime::dflash::build_dflash_draft(e.ctx, w, t, b, control);
    runtime::CapturedStep proj = runtime::dflash::build_dflash_project(e.ctx, w, b);
    std::printf("launches: draft %zu, projection %zu\n", draft.kernel_count, proj.kernel_count);
    bad += draft.kernel_count != 87 || proj.kernel_count != 8;

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
      std::printf("projection of positions %u..%u: %d ring values off by a bf16 ulp, %d further (bar 0)\n", pos0,
                  P - 1, off, far);
      bad += far;
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
    bad += compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.0f, 0, "greedy");
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
    bad += compare_draft(e, b, tw, ring, rope, 1234, P, 3, 0.0f, 0, "K 3");

    // 5. sampled
    b.ctl()->k = 7;
    b.ctl()->temperature = 0.8f;
    b.ctl()->seed_lo = 0x01234567u;
    b.ctl()->seed_hi = 0x89ABCDEFu;
    e.run(draft);
    bad += compare_draft(e, b, tw, ring, rope, 1234, P, 7, 0.8f, (uint64_t(0x89ABCDEFu) << 32) | 0x01234567u,
                         "sampled");
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
      runtime::dflash::DFlashBuffers bc(e.ctx, d, w.target_hidden, NV, 8192);
      e.imm.copy(bc.ring_k.ptr(), b.ring_k.ptr(), b.ring_k.size());
      e.imm.copy(bc.ring_v.ptr(), b.ring_v.ptr(), b.ring_v.size());
      runtime::CapturedStep dc = runtime::dflash::build_dflash_draft(e.ctx, w, tc, bc, control);
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
      b.ctl()->k = 7;
      e.run(db);
      dfref::Weights twb = tw;
      twb.head.i8.clear();
      twb.head.scales.clear();
      twb.head.bf16 = hb;
      twb.head.ksplit = 1;
      bad += compare_draft(e, b, twb, ring, rope, 1234, P, 7, 0.0f, 0, "bf16 head");
    }
  }
  if (bad) {
    std::printf("dflash_draft_test: FAIL (%d)\n", bad);
    return 1;
  }
  std::printf("dflash_draft_test: PASS\n");
  return 0;
}
