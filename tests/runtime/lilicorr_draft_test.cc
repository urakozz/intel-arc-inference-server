// Spec 26b Task 5 on the card: the LiLiCorr draft and projection lists, end to end, with the REAL drafter
// (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, loaded by loader::load_lilicorr in exact / int8 / int4 g64)
// beside a SYNTHETIC target of Bolt's shapes - a random 131072 x 4096 embedding replica and a random int8 head, so no 42 GB
// checkpoint is needed - against the host twin tests/kernels/lilicorr_ref.h fed the same weights (the loader's own host
// forms) and the card's ring. One card (the drafter's device, Decision 8's device 1 in Bolt's layout; any card here).
//
//   lilicorr_draft_test <drafter repo or dir> [exact|int8|int4]      (all three when none is given; SKIP 77 without it)
//
//   1. The lists capture to their formulas' launch counts (116 draft, 9 projection); 1100 positions of random taps are
//      projected 8 a replay (the 1024-slot ring wraps); the last replay's ring rows equal the twin's (bf16 within an
//      ulp: the GEMVs' contraction is the box compiler's) and the anchor buffer holds the last projected row (Review
//      Focus 5: the twin's hidden_norm output of position 1099, cosine >= 0.9999999).
//   2. A draft at p 1100 (rows past the ring's wrap, the window's edge at 1100 - 1023), anchor 1234, T 0, over the card's
//      own ring and anchor: the 7 slots' candidate sets (tie-aware) and log-probs, the lattice's cosine >= 0.99999 where the
//      candidates agree, the walk equal; ctrl.cur_token[1..7] = the drafts; the readback's k 7 / anchor / pos / validity;
//      the ring and the anchor bitwise as they were (Review Focus 5: the block's own K / V never reach the ring).
//   3. Determinism: a second replay's readback is bitwise the first's; draft(k = 3) runs the same 8-row block (dctl.k 7 -
//      spec 26 §12 ruling 1, 23f's finding) and its readback is bitwise the k = 7 one.
//   4. A sampled walk (T 0.8, a 64-bit seed with its high word set): the twin's walk over the card's lattice, near-ties
//      within 1e-5 excused.
//   5. The ring's snapshot: save_ring, reset (a cold drafter), load_ring -> the same draft bitwise.
//   6. A cold drafter (reset: anchor_valid 0, an empty ring) at p 0: the twin with the anchor invalid.
// It prints the draft list's host-timed replay (one fence; informational - 26c's rows time it).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "kernels/dflash_kernels.h"
#include "kernels/lilicorr_fixture_io.h"
#include "kernels/lilicorr_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/lilicorr_loader.h"
#include "loader/lilicorr_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/snapshot.h"
#include "runtime/control.h"
#include "runtime/dflash/lilicorr_drafter.h"

namespace {

namespace kd = kernels::dflash;
constexpr uint32_t V = 131072, H = 4096, M = kd::kM, NPOS = 1100;

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
std::vector<uint16_t> rnd_bf16(size_t n, float amp, uint64_t seed) {
  std::vector<uint16_t> v(n);
  const unsigned T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < T; ++t)
    pool.emplace_back([&, t] {
      Rng r(seed * 1000 + t);
      for (size_t i = n * t / T; i < n * (t + 1) / T; ++i) v[i] = dfref::rne(r.uni(amp));
    });
  for (std::thread& th : pool) th.join();
  return v;
}
int ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: lilicorr_draft_test <LiLiCorr repo or dir> [exact|int8|int4]\n");
    return 2;
  }
  std::string snap;
  try {
    snap = loader::resolve_snapshot(argv[1]);
  } catch (const std::exception& e) {
    std::printf("lilicorr_draft_test: SKIP - %s\n", e.what());
    return 77;
  }
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int8, loader::DraftFormat::Exact,
                                              loader::DraftFormat::Int4G64};
  if (argc >= 3) {
    loader::DraftFormat f;
    CHECK(loader::parse_lilicorr_draft_format(argv[2], f));
    formats = {f};
  }
  dfref::threads = std::max(1u, std::thread::hardware_concurrency());
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto replay = [&](l0::CmdList& list) {
    q.execute(list, &fence);
    fence.wait();
  };

  // the synthetic target: the embedding replica (row-major) and an int8 head
  const std::vector<uint16_t> embed = rnd_bf16(size_t(V) * H, 1.0f, 1), head_rows = rnd_bf16(size_t(V) * H, 0.05f, 2);
  l0::Mem bembed(ctx, l0::MemKind::Device, embed.size() * 2);
  imm.copy(bembed.ptr(), embed.data(), embed.size() * 2);
  loader::DeviceWeight head{l0::Mem(ctx, l0::MemKind::Device, size_t(V) * H), nullptr, {H, V, 1, 0},
                            model::WeightKind::Int8};
  {
    std::vector<int8_t> qh(size_t(V) * H);
    std::vector<float> sh(V);
    loader::quantise_int8_tiled(head_rows.data(), H, V, qh.data(), sh.data());
    imm.copy(head.mem.ptr(), qh.data(), qh.size());
    head.scales = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, size_t(V) * 4);
    imm.copy(head.scales->ptr(), sh.data(), sh.size() * 4);
  }
  const std::vector<uint16_t> taps = rnd_bf16(size_t(NPOS) * 6 * H, 2.0f, 3);
  l0::Mem control(ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  runtime::Control& c = *control.as<runtime::Control>();

  const loader::LiliCheckpoint ck(snap);
  for (loader::DraftFormat f : formats) {
    std::printf("--- %s\n", loader::draft_format_name(f));
    loader::LiliWeights w = loader::load_lilicorr(ctx, snap, {"bolt-synthetic", 88, H, V}, bembed, f);
    runtime::dflash::LiliDrafter dr(ctx, std::move(w), 4096);
    dr.attach(ctx, bembed, head, control);
    CHECK_EQ(dr.draft_step()->kernel_count, runtime::dflash::lili_draft_launches(dr.desc()));
    CHECK_EQ(dr.project_step().kernel_count, runtime::dflash::lili_project_launches(dr.desc()));
    const llfix::Form tf = f == loader::DraftFormat::Exact ? llfix::Form::Exact
                           : f == loader::DraftFormat::Int8 ? llfix::Form::Int8
                                                            : llfix::Form::Int4;
    llref::Weights tw = llfix::drafter_weights(ck, tf, kd::kLiLiCorr.pb, 0);
    llfix::set_target(tw, ck.desc().mask_id, embed, head_rows, llfix::HeadForm::Int8, V);
    const llref::Dims& D = tw.d;
    const std::vector<float> rope = dfref::rope_table(D.theta, D.HD, NPOS + 2 * M);
    runtime::dflash::LiliBuffers& b = dr.buffers();

    // 1. the projection
    dfref::Ring ring = dfref::make_ring(D);
    std::vector<float> twin_anchor;
    for (uint32_t pos = 0; pos < NPOS; pos += M) {
      const uint32_t n = std::min(M, NPOS - pos);
      imm.copy(b.taps.ptr(), &taps[size_t(pos) * D.fck()], size_t(n) * D.fck() * 2);
      replay(dr.projection(pos, n));
      if (pos + M >= NPOS) twin_anchor = llref::project(tw, dfref::Numerics::Engine, &taps[size_t(pos) * D.fck()], n, pos, rope, ring);
    }
    const size_t ring_elems = size_t(D.layers) * D.ring * D.kvn();
    std::vector<uint16_t> rk(ring_elems), rv(ring_elems);
    imm.copy(rk.data(), b.ring_k.ptr(), ring_elems * 2);
    imm.copy(rv.data(), b.ring_v.ptr(), ring_elems * 2);
    std::vector<float> anchor(H);
    imm.copy(anchor.data(), b.anchor.ptr(), size_t(H) * 4);
    int over = 0;
    for (uint32_t pos = NPOS - (NPOS - 1) % M - 1; pos < NPOS; ++pos)
      for (uint32_t l = 0; l < D.layers; ++l)
        for (uint32_t i = 0; i < D.kvn(); ++i) {
          const size_t at = (size_t(l) * D.ring + pos % D.ring) * D.kvn() + i;
          over += ulps(rk[at], dfref::rne(ring.k[at])) > 1 || ulps(rv[at], dfref::rne(ring.v[at])) > 1;
        }
    const double ac = dfref::cosine(anchor.data(), twin_anchor.data(), H);
    std::printf("1. projection: %u positions, the last replay's ring rows %d over 1 bf16 ulp; anchor cosine %.9f\n", NPOS,
                over, ac);
    CHECK(over == 0 && ac >= 0.9999999);
    // the twin reads the card's ring and anchor from here on
    for (size_t i = 0; i < ring_elems; ++i) {
      ring.k[i] = dfref::bf(rk[i]);
      ring.v[i] = dfref::bf(rv[i]);
    }

    // 2. a draft at p 1100
    c = runtime::Control{};
    c.pos = NPOS;
    c.cur_token[0] = 1234;
    const auto t0 = std::chrono::steady_clock::now();
    const runtime::dflash::LiliDraftOut o = dr.draft(replay, 7);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const llref::DraftOut want = llref::draft(tw, dfref::Numerics::Engine, ring, rope, anchor, true, 1234, NPOS, 7);
    int sets = 0, same = 0;
    double lc = 1, lpd = 0;
    bool walk = true;
    for (uint32_t s = 0; s < 7; ++s) {
      std::vector<uint32_t> a(o.cand[s], o.cand[s] + 8), bb(want.cand.begin() + s * 8, want.cand.begin() + (s + 1) * 8);
      same += a == bb;
      std::vector<uint32_t> sa = a, sb = bb;
      std::sort(sa.begin(), sa.end());
      std::sort(sb.begin(), sb.end());
      bool tie_ok = true;   // tie-aware: a candidate only one side holds sits at the 8th value within 1e-3 x the row's max
      const float edge = want.logit[s * 8 + 7], scale = std::fabs(want.logit[s * 8]);
      for (uint32_t i = 0; i < 8; ++i)
        if (!std::binary_search(sb.begin(), sb.end(), a[i]) && std::fabs(o.logit[s][i] - edge) > 1e-3f * scale)
          tie_ok = false;
      sets += tie_ok;
      for (uint32_t i = 0; i < 8; ++i) lpd = std::fmax(lpd, std::fabs(double(o.lp[s][i]) - want.lp[s * 8 + i]));
      walk &= o.ids[s] == want.walk.ids[s] && c.cur_token[1 + s] == o.ids[s];
    }
    if (same == 7) lc = dfref::cosine(&o.lattice[0][0][0], want.lattice.data(), want.lattice.size());
    std::printf("2. draft at p %u: candidate sets %d / 7 (in order %d), log-probs within %.2e, lattice cosine %.9f, walk "
                "%s; %.3f ms host-timed\n", NPOS, sets, same, lpd, lc, walk ? "=" : "DIFFERS", ms);
    CHECK(sets == 7 && lc >= 0.99999 && walk && lpd <= 1e-3);
    CHECK(o.k == 7 && o.anchor == 1234 && o.pos == NPOS && o.anchor_valid == 1);
    {   // Review Focus 5: the block's own K / V never reach the ring - a draft leaves the ring and the anchor as they were
      std::vector<uint16_t> rk2(ring_elems), rv2(ring_elems);
      std::vector<float> a2(H);
      imm.copy(rk2.data(), b.ring_k.ptr(), ring_elems * 2);
      imm.copy(rv2.data(), b.ring_v.ptr(), ring_elems * 2);
      imm.copy(a2.data(), b.anchor.ptr(), size_t(H) * 4);
      CHECK(rk2 == rk && rv2 == rv && a2 == anchor);
    }

    // 3. determinism; k = 3 runs the same block
    const runtime::dflash::LiliDraftOut o2 = dr.draft(replay, 7);
    const runtime::dflash::LiliDraftOut o3 = dr.draft(replay, 3);
    CHECK(std::memcmp(&o, &o2, sizeof o) == 0 && std::memcmp(&o, &o3, sizeof o) == 0);
    std::printf("3. a second replay bitwise; draft(k = 3) runs the whole block, its readback bitwise k = 7's\n");

    // 4. a sampled walk over the card's lattice
    {
      const uint64_t seed = (uint64_t(7) << 32) | 0x5EEDu;
      const runtime::dflash::LiliDraftOut so = dr.draft(replay, 7, 0.8f, seed);
      std::vector<float> lat(&so.lattice[0][0][0], &so.lattice[0][0][0] + 7 * 64);
      std::vector<uint32_t> cand(&so.cand[0][0], &so.cand[0][0] + 56);
      const llref::Walk wk = llref::walk(lat, cand.data(), 8, NPOS, 7, 0.8f, seed);
      int diff = 0, excused = 0;
      for (uint32_t s = 0; s < 7; ++s) {
        if (so.ids[s] == wk.ids[s]) continue;
        std::vector<double> xs;
        const uint32_t prev = s == 0 ? 0 : wk.chosen[s - 1];
        for (uint32_t i = 0; i < 8; ++i) {
          const float u = dfref::murmur3_uniform(seed ^ (uint64_t(s) << 32), NPOS + s + dfref::kDraftNoiseSalt, cand[s * 8 + i]);
          xs.push_back(double(lat[(size_t(s) * 8 + prev) * 8 + i]) / 0.8 - std::log(-std::log1p(-double(u))));
        }
        std::sort(xs.rbegin(), xs.rend());
        (xs[0] - xs[1] < 1e-5 ? excused : diff) += 1;
        break;
      }
      std::printf("4. sampled walk (T 0.8): %d differences from the twin over the card's lattice (%d near-ties excused)\n",
                  diff, excused);
      CHECK(diff == 0);
    }

    // 5. the ring's snapshot round trip
    {
      std::vector<uint8_t> snapb(dr.ring_bytes());
      CHECK_EQ(dr.ring_bytes(), runtime::dflash::lili_ring_snapshot_bytes(dr.desc()));
      dr.save_ring(imm, snapb.data());
      dr.reset(imm);
      dr.load_ring(imm, snapb.data());
      const runtime::dflash::LiliDraftOut o5 = dr.draft(replay, 7);
      CHECK(std::memcmp(&o, &o5, sizeof o) == 0);
      std::printf("5. save_ring / reset / load_ring (%zu B): the draft bitwise the same\n", snapb.size());
    }

    // 6. a cold drafter at p 0
    {
      dr.reset(imm);
      c = runtime::Control{};
      c.pos = 0;
      c.cur_token[0] = 77;
      const runtime::dflash::LiliDraftOut oc = dr.draft(replay, 7);
      const dfref::Ring cold = dfref::make_ring(D);
      const llref::DraftOut wc =
          llref::draft(tw, dfref::Numerics::Engine, cold, rope, std::vector<float>(H, 0.0f), false, 77, 0, 7);
      bool cand_eq = true;
      for (uint32_t i = 0; i < 56; ++i) cand_eq &= oc.cand[i / 8][i % 8] == wc.cand[i];
      const double cc = cand_eq ? dfref::cosine(&oc.lattice[0][0][0], wc.lattice.data(), wc.lattice.size()) : 1.0;
      std::printf("6. cold (anchor invalid) at p 0: candidates %s, lattice cosine %.9f, anchor_valid %u\n",
                  cand_eq ? "=" : "differ (near-ties)", cc, oc.anchor_valid);
      CHECK(oc.anchor_valid == 0 && cc >= 0.99999);
    }
  }
  std::printf("lilicorr_draft_test OK\n");
  return 0;
}
