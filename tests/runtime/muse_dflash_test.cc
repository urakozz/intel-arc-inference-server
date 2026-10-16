// Spec 23f Task 7 on the card (D1's list half): Muse-Glimmer's draft and projection lists through
// runtime::dflash::MuseDrafter with the REAL assistant (meta-models/Muse-Glimmer-30B-assistant, loaded by
// loader::load_drafter in int4 g64 / int8 / bf16) beside a SYNTHETIC target - a random embedding and a random int8
// (and bf16) head of Muse-Glimmer-30B's shapes (202048 x 6656), so no 17 GB checkpoint is needed - against the host
// twin tests/kernels/dflash_ref.h in its Muse form, fed the loader's own host forms and the card's ring.
//
//   muse_dflash_test <assistant repo or dir> [int4|int8|bf16]     (every format when none; SKIP 77 without it)
//
//   1. The lists capture to their formulas' launch counts (66 draft, 8 projection); 2200 positions of random taps are
//      projected 16 a replay (the 2112-slot ring wraps); the last replay's rows equal the twin's projection (within a
//      bf16 ulp: the GEMVs' contraction is the box compiler's).
//   2. A draft at p 2200 (row 0 reads p - 2048 .. p - 1 across the wrap), anchor 1234: the 16 final-norm rows (cosine
//      >= 0.99999), the 15 rows' top-16 sets (tie-aware) and raw logits (cosine >= 0.99999) against the twin, the 15
//      greedy ids equal (a near tie excused); the readback's k 15 / anchor / pos; vids = [anchor, d_1 .. d_15].
//   3. Determinism: a second replay's readback is bitwise the first's.
//   4. set_draft: a candidate written into the readback and vids; a non-candidate refused.
//   5. The capped q (spec 23 decision 11's other form) over the same weights and buffers: the same candidates, unary
//      = 20 tanh(raw x 0.19611613513818404 / 20).
//   6. The ring's snapshot: save_ring, reset (a cold ring drafts differently), load_ring -> the first draft bitwise.
//   7. The bf16 head (first format only): the twin agrees as in 2.
// It prints the draft list's host-timed replay (one fence; informational - r47.serve / D5 time the loop).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/dflash_kernels.h"
#include "kernels/dflash_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_buffers.h"
#include "runtime/dflash/dflash_capture.h"
#include "runtime/dflash/muse_drafter.h"

namespace {

namespace kd = kernels::dflash;
constexpr uint32_t V = 202048, H = 6656, M = kd::kMuseM, R = kd::kMuseR, MAXLEN = 8192;

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

// layout-1 tiles -> the twin's GPTQ-native int4 (qweight [K/8][N], f16 scales [K/64][N]): repack_int4_layout1 undone.
void untile_int4(const std::vector<uint32_t>& tiles, uint32_t K, uint32_t N, dfref::Lin& l) {
  const uint32_t G = K / 64, NT = N / 16;
  l.q4.assign(size_t(K / 8) * N, 0);
  l.s4.assign(size_t(G) * N, 0);
  for (uint32_t nt = 0; nt < NT; ++nt)
    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t* tile = tiles.data() + (size_t(nt) * G + g) * 136;
      for (uint32_t j = 0; j < 8; ++j)
        for (uint32_t ln = 0; ln < 16; ++ln) l.q4[size_t(g * 8 + j) * N + nt * 16 + ln] = tile[j * 16 + ln];
      for (uint32_t ln = 0; ln < 16; ln += 2) {
        l.s4[size_t(g) * N + nt * 16 + ln] = uint16_t(tile[128 + ln / 2] & 0xFFFFu);
        l.s4[size_t(g) * N + nt * 16 + ln + 1] = uint16_t(tile[128 + ln / 2] >> 16);
      }
    }
}

dfref::Lin host_lin(const loader::DFlashHostLinear& h) {
  dfref::Lin l;
  l.K = h.K;
  l.N = h.N;
  l.ksplit = kernels::gemv_bf16_tiling(h.N).ksplit;
  if (h.kind == model::WeightKind::Int4) {
    untile_int4(h.int4, h.K, h.N, l);
  } else {
    l.bf16 = h.bf16;
    l.i8 = h.i8;
    l.scales = h.scales;
  }
  return l;
}

dfref::Weights twin_weights(const loader::DFlashCheckpoint& ck, loader::DraftFormat f) {
  const model::DFlashDesc& d = ck.desc();
  dfref::Weights w;
  dfref::Dims& D = w.d;
  D.version = 1;
  D.muse = true;
  D.layers = d.layers;
  D.H = d.hidden;
  D.NH = d.q_heads;
  D.NKV = d.kv_heads;
  D.HD = d.head_dim;
  D.I = d.inter;
  D.V = V;
  D.used = V;
  D.M = M;
  D.ring = kd::kMuse.ring;
  D.win = kd::kMuse.win;
  D.pb = kd::kMuse.pb;
  D.taps = uint32_t(d.taps.size());
  D.th = ck.target_hidden();
  D.topk = kd::kTopK;
  D.eps = d.eps;
  D.in_scale = 1.0f;
  D.out_mult = 1.0f;
  D.softcap = 0.0f;
  D.scale = float(1.0 / std::sqrt(double(d.head_dim)));
  D.theta = d.theta;
  D.causal = false;
  w.fc = host_lin(ck.fc(f));
  w.hidden_norm = ck.hidden_norm();
  w.final_norm = ck.final_norm();
  const std::vector<float> qk = ck.qk_norms();
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
    w.layers.push_back(std::move(y));
  }
  return w;
}

struct Env {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  void run(l0::CmdList& list) {
    q.execute(list, &f);
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

// One draft's readback against the twin (all 15 rows). Returns failures.
int compare_draft(Env& e, runtime::dflash::MuseDrafter& dr, const dfref::Weights& tw, const dfref::Ring& ring,
                  const std::vector<float>& rope, uint32_t anchor, uint32_t p, const char* what) {
  const runtime::dflash::MuseDraftOut& o = dr.readback();
  const dfref::DraftOut t = dfref::draft(tw, dfref::Numerics::Engine, ring, rope, anchor, p, R);
  const std::vector<float> h = e.read<float>(dr.buffers().h, size_t(M) * H);
  double hc = 1;
  for (uint32_t r = 0; r < M; ++r) hc = std::fmin(hc, dfref::cosine(&h[size_t(r) * H], &t.hidden[size_t(r) * H], H));
  std::vector<float> es, ts;
  int sets_bad = 0, walk_bad = 0, near = 0;
  for (uint32_t j = 0; j < R; ++j) {
    for (uint32_t c = 0; c < 16; ++c) {
      const uint32_t id = o.cand[j][c];
      const auto it = std::find(t.cand.begin() + j * 16, t.cand.begin() + (j + 1) * 16, id);
      if (it == t.cand.begin() + (j + 1) * 16) {
        if (std::fabs(o.unary[j][c] - t.unary[j * 16 + 15]) > 1e-3f * std::fmax(1.0f, std::fabs(t.unary[j * 16 + 15])))
          ++sets_bad;
      } else {
        es.push_back(o.unary[j][c]);
        ts.push_back(t.unary[size_t(it - t.cand.begin())]);
      }
    }
    if (o.ids[j] != t.ids[j]) {
      // a greedy row's ids differ only at a near tie of the twin's own top two
      const float a = t.unary[j * 16 + 0], bb = t.unary[j * 16 + 1];
      if (std::fabs(a - bb) <= 1e-4f * std::fmax(1.0f, std::fabs(a))) {
        ++near;
      } else {
        ++walk_bad;
      }
    }
  }
  const std::vector<uint32_t> vids = e.read<uint32_t>(dr.vids(), M);
  bool vids_ok = vids[0] == anchor;
  for (uint32_t j = 0; j < R; ++j) vids_ok &= vids[1 + j] == o.ids[j];
  const double sc = dfref::cosine(es.data(), ts.data(), es.size());
  std::printf("%s (p %u): rows cos %.7f, logits cos %.7f, top-16 sets %s, 15 ids %s%s, vids %s\n", what, p, hc, sc,
              sets_bad ? "DIFF" : "=", walk_bad ? "DIFF" : "=", near ? " (near ties excused)" : "",
              vids_ok ? "=" : "DIFF");
  int bad = 0;
  bad += hc < 0.99999;
  bad += sc < 0.99999;
  bad += sets_bad != 0;
  bad += walk_bad != 0;
  bad += !vids_ok;
  bad += o.k != R || o.anchor != anchor || o.pos != p;
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "meta-models/Muse-Glimmer-30B-assistant");
  } catch (const std::exception& e) {
    std::printf("SKIP: no Muse-Glimmer assistant (%s)\n", e.what());
    return 77;
  }
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int4G64, loader::DraftFormat::Int8,
                                              loader::DraftFormat::Bf16};
  if (argc > 2) {
    loader::DraftFormat f;
    CHECK(loader::parse_draft_format(argv[2], f));
    formats = {f};
  }
  dfref::threads = std::max(1u, std::thread::hardware_concurrency());
  Env e;
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
  const model::DFlashTargetShape shape{"Muse-Glimmer-30B (synthetic embedding / head)", 52, H, V};
  const loader::DFlashCheckpoint ck(snap);
  const model::DFlashDesc& d0 = ck.desc();
  const std::vector<float> rope = dfref::rope_table(d0.theta, d0.head_dim, MAXLEN + M);
  int bad = 0;
  bool first_format = true;
  auto replay_on = [&](l0::CmdList& list) { e.run(list); };

  for (loader::DraftFormat f : formats) {
    std::printf("== drafter %s\n", loader::draft_format_name(f));
    runtime::dflash::MuseDrafter dr(e.ctx, loader::load_drafter(e.ctx, snap, shape, embed_d, f), MAXLEN, R);
    l0::Mem control(e.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
    std::memset(control.ptr(), 0, sizeof(runtime::Control));
    dr.attach(e.ctx, embed_d, head_d, control);
    dfref::Weights tw = twin_weights(ck, f);
    tw.embed = embed;
    tw.mask_row.resize(H);
    for (uint32_t i = 0; i < H; ++i) tw.mask_row[i] = dfref::bf(embed[size_t(d0.mask_id) * H + i]);
    tw.head.K = H;
    tw.head.N = V;
    tw.head.i8 = head_q;
    tw.head.scales = head_s;
    std::printf("launches: draft %zu, projection %zu; drafter %zu B (%s)\n", dr.draft_step()->kernel_count,
                dr.project_step().kernel_count, dr.bytes(), dr.weights().line.c_str());
    bad += dr.draft_step()->kernel_count != 66 || dr.project_step().kernel_count != 8;
    runtime::dflash::DFlashBuffers& b = dr.buffers();

    // 1. the projection: 2200 positions, 16 a replay
    const uint32_t P = 2200, width = d0.fc_k(dr.weights().target_hidden);
    for (uint32_t pos = 0; pos < P; pos += M) {
      const uint32_t n = std::min(M, P - pos);
      const std::vector<uint16_t> rows = taps_rows(pos, n, width);
      e.imm.copy(dr.taps().ptr(), rows.data(), rows.size() * 2);
      e.run(dr.projection(pos, n));
    }
    const dfref::Ring ring = read_ring(e, b);
    {
      dfref::Ring tr = dfref::make_ring(tw.d);
      const uint32_t pos0 = (P - 1) / M * M, n = P - pos0;
      const std::vector<uint16_t> rows = taps_rows(pos0, n, width);
      dfref::project(tw, dfref::Numerics::Engine, rows.data(), n, pos0, rope, tr);
      int off = 0, far = 0;
      for (uint32_t l = 0; l < d0.layers; ++l)
        for (uint32_t r = 0; r < n; ++r) {
          const size_t base = (size_t(l) * tw.d.ring + (pos0 + r) % tw.d.ring) * tw.d.kvn();
          for (uint32_t i = 0; i < tw.d.kvn(); ++i)
            for (int kv = 0; kv < 2; ++kv) {
              const float a = (kv ? ring.v : ring.k)[base + i], bb = (kv ? tr.v : tr.k)[base + i];
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

    // 2. a draft at p 2200
    runtime::Control* c = control.as<runtime::Control>();
    c->pos = P;
    c->n_active = 1;
    c->cur_token[0] = 1234;
    auto t0 = std::chrono::steady_clock::now();
    dr.draft(replay_on, 15);
    const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bad += compare_draft(e, dr, tw, ring, rope, 1234, P, "greedy");
    const runtime::dflash::MuseDraftOut first = dr.readback();

    // 3. determinism (and k only bounds the caller: a draft for k 3 is the same block)
    t0 = std::chrono::steady_clock::now();
    dr.draft(replay_on, 3);
    const double ms2 = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool same = std::memcmp(&first, &dr.readback(), sizeof first) == 0;
    std::printf("replay (k 3): readback %s; the draft list %.3f / %.3f ms (host-timed, one fence)\n",
                same ? "bitwise identical" : "DIFFERS", ms, ms2);
    bad += !same;

    // 4. set_draft
    {
      const uint32_t id = first.cand[0][3];
      dr.set_draft(0, id);
      const std::vector<uint32_t> vids = e.read<uint32_t>(dr.vids(), M);
      bool ok = dr.readback().ids[0] == id && dr.readback().chosen[0] == 3 && vids[1] == id;
      bool threw = false;
      try {
        uint32_t other = 0;
        while (std::find(first.cand[1], first.cand[1] + 16, other) != first.cand[1] + 16) ++other;
        dr.set_draft(1, other);
      } catch (const std::exception&) {
        threw = true;
      }
      std::printf("set_draft: a candidate %s, a non-candidate %s\n", ok ? "written" : "NOT WRITTEN",
                  threw ? "refused" : "ACCEPTED");
      bad += !ok || !threw;
    }

    // 5. the capped q over the same weights and buffers
    {
      runtime::dflash::DFlashTarget tc;
      tc.embed = &embed_d;
      tc.head = &head_d;
      tc.vocab_rows = V;
      tc.vocab_used = V;
      tc.q_mult = runtime::dflash::MuseDrafter::kCapMult;
      tc.q_softcap = runtime::dflash::MuseDrafter::kCapSoftcap;
      runtime::CapturedStep cap = runtime::dflash::build_dflash_draft(e.ctx, dr.weights(), tc, b, control);
      runtime::dflash::set_draft_params(*b.ctl(), dr.desc(), R, 0.0f, 0);
      e.run(cap.list);
      const runtime::dflash::MuseDraftOut& o = dr.readback();
      int diff = 0;
      for (uint32_t j = 0; j < R; ++j)
        for (uint32_t k = 0; k < 16; ++k) {
          const float want = std::tanh(first.unary[j][k] * tc.q_mult / tc.q_softcap) * tc.q_softcap;
          diff += o.cand[j][k] != first.cand[j][k] || std::fabs(o.unary[j][k] - want) > 1e-5f * std::fabs(want) + 1e-6f;
        }
      std::printf("capped q: candidates = the raw draft's, unary = 20 tanh(raw x 0.196 / 20): %s\n",
                  diff ? "DIFFERS" : "yes");
      bad += diff != 0;
    }

    // 6. the ring's snapshot round trip
    {
      std::vector<uint8_t> host(dr.ring_bytes());
      dr.save_ring(e.imm, host.data());
      dr.reset(e.imm);
      c->pos = P;
      c->cur_token[0] = 1234;
      dr.draft(replay_on, 15);
      const bool cold_differs = std::memcmp(dr.readback().unary, first.unary, sizeof first.unary) != 0;
      dr.load_ring(e.imm, host.data());
      c->pos = P;
      c->cur_token[0] = 1234;
      dr.draft(replay_on, 15);
      const bool restored = std::memcmp(&first, &dr.readback(), sizeof first) == 0;
      std::printf("ring snapshot (%zu B): a cold ring %s, the restored ring %s\n", host.size(),
                  cold_differs ? "drafts differently" : "DRAFTS THE SAME", restored ? "bitwise the first draft" : "DIFFERS");
      bad += !cold_differs || !restored;
    }

    // 7. the bf16 head
    if (first_format) {
      std::vector<uint16_t> hb(size_t(V) * H);
      {
        Rng g(3);
        std::vector<uint16_t> rm(size_t(V) * H);
        for (uint16_t& x : rm) x = dfref::rne(g.uni(0.05f));
        common::repack_bf16_tiled(rm.data(), H, V, hb.data());
      }
      loader::DeviceWeight hbd{e.upload(hb.data(), hb.size() * 2), nullptr, {H, V, 1, 0}, model::WeightKind::Bf16};
      runtime::dflash::DFlashTarget tb;
      tb.embed = &embed_d;
      tb.head = &hbd;
      tb.vocab_rows = V;
      tb.vocab_used = V;
      tb.q_mult = 1.0f;
      tb.q_softcap = 0.0f;
      runtime::CapturedStep db = runtime::dflash::build_dflash_draft(e.ctx, dr.weights(), tb, b, control);
      runtime::dflash::set_draft_params(*b.ctl(), dr.desc(), R, 0.0f, 0);
      c->pos = P;
      c->cur_token[0] = 1234;
      e.run(db.list);
      dfref::Weights twb = tw;
      twb.head.i8.clear();
      twb.head.scales.clear();
      twb.head.bf16 = hb;
      twb.head.ksplit = kernels::gemv_bf16_tiling(V).ksplit;
      bad += compare_draft(e, dr, twb, ring, rope, 1234, P, "bf16 head");
    }
    first_format = false;
  }
  if (bad) {
    std::printf("muse_dflash_test: FAIL (%d)\n", bad);
    return 1;
  }
  std::printf("muse_dflash_test: PASS\n");
  return 0;
}
