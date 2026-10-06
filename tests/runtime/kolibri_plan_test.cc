// Spec 20c Task 5: Kolibri-1's planner, launch counts and the split by bytes as device-free
// arithmetic (runtime/kolibri/kolibri_sizes.h). Host only. Every number below is DERIVED from the
// descriptor (plan 20c Task 5 Step 1; docs: spec 20 §11).
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "loader/kolibri1_layout.h"
#include "model/kolibri1.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_sizes.h"
#include "runtime/pipeline_plan.h"

namespace {
using runtime::PpHandoff;
using runtime::kolibri::KolAttn;
template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}  // namespace

int main() {
  namespace rk = runtime::kolibri;
  const model::Kolibri1Desc& d = model::kolibri1();
  const model::KolPlacement one = model::KolPlacement::one(d), two = model::KolPlacement::two(d, 25);

  // --- launches: 15 / 17 a layer, + embed + fold, + 4 at the head; the cut adds nothing ----------------
  for (const model::KolPlacement* p : {&one, &two}) {
    CHECK_EQ(rk::decode_launches(d, *p, KolAttn::Flash, PpHandoff::Copy), size_t(756));
    CHECK_EQ(rk::decode_launches(d, *p, KolAttn::Eager, PpHandoff::Copy), size_t(856));
  }
  CHECK_EQ(rk::decode_launches(d, two, KolAttn::Flash, PpHandoff::Peer), size_t(758));
  CHECK_EQ(rk::decode_launches(d, two, KolAttn::Eager, PpHandoff::Peer), size_t(858));
  CHECK_EQ(rk::decode_launches(d, one, KolAttn::Flash, PpHandoff::Peer), size_t(756));   // no hand-off on one card
  CHECK_EQ(rk::device_launches(d, two, 0, KolAttn::Flash, PpHandoff::Copy), size_t(2 + 25 * 15));
  CHECK_EQ(rk::device_launches(d, two, 1, KolAttn::Flash, PpHandoff::Copy), size_t(25 * 15 + 4));

  // --- the per-device plan at 262144, int4 attention, int8 head, split 25 ------------------------------
  const uint32_t L = 262144;
  const std::vector<rk::DevicePlan> pl = rk::plan(d, two, L, /*int8_head=*/true, false, KolAttn::Flash);
  CHECK_EQ(pl.size(), size_t(2));
  CHECK_EQ(pl[0].weights, size_t(21425228800ull));
  CHECK_EQ(pl[1].weights, size_t(21098071040ull));
  for (const rk::DevicePlan& p : pl) {
    CHECK_EQ(p.full_kv, size_t(2684354560ull));   // 5 full layers x 262144 x 512 x 2 B x (K, V)
    CHECK_EQ(p.rings, size_t(167772160ull));      // 20 sliding layers x 4096 x 512 x 2 B x (K, V)
    CHECK_EQ(p.rope, size_t(134217728ull));
  }
  CHECK_EQ(rk::full_kv_bytes_per_pos(d), size_t(20480));
  CHECK_EQ(rk::ring_bytes_per_layer(d), size_t(8388608));
  CHECK_EQ(size_t(40) * rk::ring_bytes_per_layer(d), size_t(335544320ull));   // spec 20 §11's 335.5 MB
  CHECK(pl[0].link == rk::link_bytes(d, 0) && pl[1].link == rk::link_bytes(d, 1) && pl[1].link > pl[0].link);
  CHECK_EQ(rk::plan(d, one, 4096, true)[0].link, size_t(0));

  // --- the split: runtime::pp_balance over pp_layer_bytes, 16b's rule and no second one ----------------
  {
    const runtime::PpBalance b = rk::pp_split(d, L, true, PpHandoff::Copy, KolAttn::Flash);
    CHECK_EQ(b.split, 25u);
    const std::vector<size_t> lb = rk::pp_layer_bytes(d, L);
    CHECK_EQ(lb.size(), size_t(50));
    CHECK_EQ(lb[0], loader::kol_layer_bytes(d).total() + rk::ring_bytes_per_layer(d));
    CHECK_EQ(lb[4], loader::kol_layer_bytes(d).total() + rk::full_bytes_per_layer(d, L));
    // the same answer as pp_balance called directly on the same terms
    const size_t state = sizeof(runtime::Control) + rk::scratch_sizes(d).total();
    const runtime::PpBalance direct = runtime::pp_balance(
        lb, loader::kol_embed_bytes(d) + d.rope_table_bytes(L) + state + rk::link_bytes(d, 0),
        loader::kol_final_norm_bytes(d) + loader::kol_lm_head_bytes(d, true) + d.rope_table_bytes(L) + state +
            rk::link_bytes(d, 1));
    CHECK(direct.split == b.split && direct.dev0 == b.dev0 && direct.dev1 == b.dev1);
    // the plan's totals are the balance's sides (the same bytes)
    CHECK_EQ(pl[0].total(), b.dev0);
    CHECK_EQ(pl[1].total(), b.dev1);
    std::printf("split 25 at 262144 (int8 head): device 0 %.3f GB, device 1 %.3f GB", b.dev0 / 1e9, b.dev1 / 1e9);
    for (uint32_t s : {24u, 26u}) {
      const std::vector<rk::DevicePlan> q = rk::plan(d, model::KolPlacement::two(d, s), L, true, false, KolAttn::Flash);
      std::printf("; split %u: %.3f / %.3f GB", s, q[0].total() / 1e9, q[1].total() / 1e9);
      CHECK(std::max(q[0].total(), q[1].total()) > std::max(b.dev0, b.dev1));
    }
    std::printf(" (derived)\n");
  }

  // --- max_len and the fit ---------------------------------------------------------------------------
  const size_t card = size_t(32530000000ull), reserve = size_t(1.5e9);
  const std::array<size_t, runtime::kPpDevices> cards{card, card};
  CHECK_EQ(rk::max_len_that_fits(d, two, true, cards, reserve), 262144u);   // capped by decision 3
  {
    const runtime::PpChoice c = rk::pp_split_and_len(d, true, cards, reserve);
    CHECK(c.max_len == 262144u && c.split == 25u);
  }
  CHECK(!rk::fits_one_card(d, true, card, reserve));
  CHECK(!rk::fits_one_card(d, false, card, reserve));
  model::Kolibri1Desc s5 = d;
  s5.layers = 5;
  CHECK(rk::fits_one_card(s5, false, card, reserve));
  model::Kolibri1Desc l30 = d;
  l30.layers = 30;
  CHECK(rk::fits_one_card(l30, false, card, reserve));
  CHECK(rk::fits_one_card(l30, true, card, reserve));
  // one card: the synthetic model's whole context fits; device bytes the planner reports
  {
    const uint32_t len = rk::max_len_that_fits(s5, model::KolPlacement::one(s5), false, cards, reserve);
    CHECK_EQ(len, 262144u);
    const std::string line = rk::describe(rk::plan(s5, model::KolPlacement::one(s5), len, false),
                                          model::KolPlacement::one(s5), len, cards, reserve);
    CHECK(line.find("one card") != std::string::npos);
    std::printf("%s\n", line.c_str());
  }
  CHECK(rk::describe(pl, two, L, cards, reserve).find("split 25") != std::string::npos);
  // a placement outside [1, layers - 1] throws
  CHECK(throws([&] { (void)rk::plan(d, model::KolPlacement::two(d, 0), 4096, true); }));
  CHECK(throws([&] { (void)rk::plan(d, model::KolPlacement::two(d, 50), 4096, true); }));
  // eager's score row is planned only under eager
  CHECK_EQ(rk::attn_scores_bytes(d, 4096, KolAttn::Flash), size_t(0));
  CHECK_EQ(rk::attn_scores_bytes(d, 4096, KolAttn::Eager), size_t(48) * 4096 * 4);
  // an empty group still allocates a line (a 3-layer prefix has no full layer)
  model::Kolibri1Desc s3 = d;
  s3.layers = 3;
  CHECK_EQ(rk::persistent_sizes(s3, model::KolPlacement::one(s3), 0, 4096).full_k, size_t(64));

  // --- spec 20d: the prefill chunk -------------------------------------------------------------------
  // launches: 45 a layer + embed + fold on device 0, the same sum on one card and on two
  CHECK_EQ(rk::prefill_chunk_launches(d, one), size_t(2252));
  CHECK_EQ(rk::prefill_chunk_launches(d, two), size_t(2252));
  CHECK_EQ(rk::prefill_device_launches(d, two, 0), size_t(2 + 25 * 45));
  CHECK_EQ(rk::prefill_device_launches(d, two, 1), size_t(25 * 45));
  CHECK_EQ(rk::prefill_chunk_launches(s5, model::KolPlacement::one(s5)), size_t(2 + 5 * 45));
  CHECK_EQ(rk::prefill_chunk_launches(s5, model::KolPlacement::two(s5, 3)), size_t(2 + 5 * 45));
  CHECK_EQ(rk::kPrefillHeadLaunches, size_t(5));
  // the weight batches: 385 blocks (384 + the bf16 shared expert) in 512 MiB
  CHECK_EQ(rk::pf_block_gu_bytes(d), size_t(5242880));
  CHECK_EQ(rk::pf_block_dn_bytes(d), size_t(2621440));
  CHECK_EQ(rk::pf_batch_blocks_gu(d), 102u);
  CHECK_EQ(rk::pf_batch_blocks_dn(d), 204u);
  CHECK_EQ(rk::pf_batches_gu(d), 4u);
  CHECK_EQ(rk::pf_batches_dn(d), 2u);
  CHECK_EQ(rk::pf_tiles(d, 2048), 820u);
  CHECK_EQ(rk::pf_slabs(7168), 7u);
  CHECK_EQ(rk::pf_slabs(2560), 3u);
  CHECK_EQ(rk::pf_slab_width(2560, 2048), 512u);
  CHECK_EQ(rk::pf_ld(7168), 7168u);
  // the scratch per device (derived: the formula, term by term)
  {
    const rk::PrefillSizes ps = rk::prefill_sizes(d);
    const size_t C = 2048, R = size_t(820) * 32;
    CHECK_EQ(ps.resid, C * 2560 * 2);
    CHECK_EQ(ps.partials, C * 7168 * 4);
    CHECK_EQ(ps.slab, size_t(6144) * 1024 * 2);
    CHECK_EQ(ps.sumsq_r, size_t(20) * C * 4);
    CHECK_EQ(ps.attn_q, C * 6144 * 4);
    CHECK_EQ(ps.attn_out, C * 6144 * 2);
    CHECK_EQ(ps.logits, C * 512 * 4);
    CHECK_EQ(ps.routes, size_t(50) * C * 32 * 4);
    CHECK_EQ(ps.hdr, size_t(400) * 4);
    CHECK_EQ(ps.tiles, size_t(820) * 8);
    CHECK_EQ(ps.row_tok, R * 4);
    CHECK_EQ(ps.pair_row, C * 6 * 4);
    CHECK_EQ(ps.xg, R * 2560 * 2);
    CHECK_EQ(ps.h, R * 512 * 2);
    CHECK_EQ(ps.w, size_t(512) << 20);
    CHECK_EQ(ps.total(), size_t(904632800ull));
    std::printf("prefill scratch per device: %.3f GB (the 512 MiB weight batch its largest term; derived)\n",
                ps.total() / 1e9);
    // the two-card hand-off at chunk size: 10.5 MB of rows + 160 KB of sums (16b's layout)
    const runtime::PpLandingLayout ll = rk::pf_landing_layout(d);
    CHECK_EQ(ll.resid_bytes, C * 2560 * 2);
    CHECK_EQ(ll.sumsq_bytes, size_t(20) * C * 4);
    CHECK(rk::pf_link_bytes(d, 1) > ll.resid_bytes + ll.sumsq_bytes && rk::pf_link_bytes(d, 0) < 4096);
    // planned only when a run prefills, on both devices
    const std::vector<rk::DevicePlan> pp = rk::plan(d, two, L, true, false, KolAttn::Flash, true);
    CHECK_EQ(pp[0].prefill_scratch, ps.total() + rk::pf_link_bytes(d, 0));
    CHECK_EQ(pp[1].prefill_scratch, ps.total() + rk::pf_link_bytes(d, 1));
    CHECK_EQ(pl[0].prefill_scratch, size_t(0));
    CHECK_EQ(rk::plan(s5, model::KolPlacement::one(s5), 4096, false, false, KolAttn::Flash, true)[0].prefill_scratch,
             ps.total() - size_t(45) * C * 32 * 4);   // 5 layers' route rows, not 50; no link on one card
  }
  // max_len with the prefill scratch planned: still the trained context on two cards
  CHECK_EQ(rk::max_len_that_fits(d, two, true, cards, reserve, 0, true), 262144u);
  {
    const runtime::PpChoice c = rk::pp_split_and_len(d, true, cards, reserve, 0, true);
    CHECK(c.max_len == 262144u && c.split == 25u);
    const runtime::PpBalance b = rk::pp_split(d, L, true, PpHandoff::Copy, KolAttn::Flash, true);
    CHECK_EQ(b.split, 25u);
    std::printf("with prefill planned: max_len 262144 on two cards, split 25 (device 0 %.3f GB, device 1 %.3f GB)\n",
                b.dev0 / 1e9, b.dev1 / 1e9);
  }
  CHECK(rk::fits_one_card(s5, false, card, reserve, true));
  std::puts("kolibri_plan_test OK");
  return 0;
}
