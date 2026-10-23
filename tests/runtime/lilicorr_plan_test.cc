// Spec 26b Task 5 (host): the LiLiCorr drafter's buffers and its place in Bolt's plan - runtime/dflash/lilicorr_sizes.h's
// arithmetic from the published config (tests/model/lilicorr/config.json). No device.
//
//   1. The buffers at max_len 262144: the ring 5,120 B a position x 1024 = 5,242,880 B (spec 26 §4), the anchor 16,384 B,
//      the taps 8 x 6 x 4096 bf16, the head's scratch; every term from its own formula; the drafter's whole bytes per
//      format (weights at the published NVFP4 set + buffers) printed.
//   2. The lists' launch counts (draft 116, projection 9) and the snapshot's bytes (the ring + the anchor: 5,259,264).
//   3. draft_list_k: 7 at every K (the whole block - spec 26 §12 ruling 1); the vLLM form keeps K.
//   4. The prefix store's key: distinct per format, per norm weights, from the generic drafter_key; never 0.
//   5. Bolt on two 32 GiB cards with the drafter (int8 / exact / int4 g64) and its 1.07 GB replica on device 1
//      (lili_bolt_reserve), 25c's decode scratch, 25d's prefill scratch and 7 verify slots as 25b's test places them:
//      `--max-len auto` - the length each format allows, printed (DERIVED).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "loader/bolt_layout.h"
#include "loader/lilicorr_layout.h"
#include "model/bolt.h"
#include "model/dflash.h"
#include "runtime/bolt/bolt_sizes.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/dflash/lilicorr_sizes.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: lilicorr_plan_test <lilicorr config.json>\n");
    return 2;
  }
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  model::check_lilicorr(d);
  namespace rd = runtime::dflash;
  using F = loader::DraftFormat;
  const model::BoltDesc& bolt = model::bolt();
  const uint32_t TH = 4096, V = 131072, ML = 262144;

  // --- 1. the buffers ---------------------------------------------------------------------------------------------------
  {
    const rd::LiliSizes s = rd::lili_sizes(d, TH, V, ML);
    CHECK_EQ(s.ring_k + s.ring_v, size_t(5242880));
    CHECK_EQ(s.anchor, size_t(16384));
    CHECK_EQ(s.taps, size_t(8) * 6 * 4096 * 2);
    CHECK_EQ(s.rope, size_t(ML + 8) * 128 * 4);
    CHECK_EQ(s.dctl, sizeof(rd::LiliCtl));
    CHECK_EQ(s.out, sizeof(rd::LiliDraftOut));
    CHECK_EQ(s.xa, size_t(8) * 12288 * 2);
    CHECK_EQ(s.gout, size_t(8) * 24576 * 4);
    CHECK_EQ(s.part, size_t(32) * (1024 / 64) * 8 * 130 * 4);
    CHECK_EQ(s.logits, size_t(7) * V * 4);
    CHECK_EQ(s.tpart, size_t(7) * 32 * 18 * 4);
    const rd::LiliHeadScratch h = rd::lili_head_scratch(d);
    const size_t R = 56, Hl = 1024;
    CHECK_EQ(h.total, R * Hl * 7 + 7 * Hl + R * 5 + R * 3 * Hl + R * 2048 + 2 * Hl + R * 2 * 1024 + 1024 + 7 * 64);
    CHECK_EQ(s.head, h.total * 4);
    std::printf("1. buffers at max_len %u: %zu B (ring %zu, anchor %zu, RoPE %zu, head scratch %zu)\n", ML, s.total(),
                s.ring_k + s.ring_v, s.anchor, s.rope, s.head);
    for (F fm : {F::Exact, F::Int8, F::Int4G64})
      std::printf("   the drafter in %s: %zu B (weights %zu)\n", loader::draft_format_name(fm),
                  rd::lili_drafter_bytes(d, TH, V, fm, ML),
                  loader::lilicorr_weight_bytes(d, TH, fm, loader::lili_nvfp4_rows_published(d)).total());
  }

  // --- 2. launches, snapshot --------------------------------------------------------------------------------------------
  CHECK_EQ(rd::lili_draft_launches(d), size_t(116));
  CHECK_EQ(rd::lili_project_launches(d), size_t(9));
  CHECK_EQ(rd::lili_ring_snapshot_bytes(d), size_t(5242880 + 16384));
  std::printf("2. the draft list 116 launches, the projection 9; a snapshot adds %zu B\n", rd::lili_ring_snapshot_bytes(d));

  // --- 3. draft_list_k --------------------------------------------------------------------------------------------------
  for (uint32_t k = 1; k <= 7; ++k) CHECK_EQ(rd::draft_list_k(d, k), 7u);
  {
    model::DFlashDesc v = d;
    v.form = model::DFlashForm::Vllm;
    for (uint32_t k = 1; k <= 7; ++k) CHECK_EQ(rd::draft_list_k(v, k), k);
  }
  std::printf("3. draft_list_k: 7 at every K (the whole block); the vLLM form keeps K\n");

  // --- 4. the key -------------------------------------------------------------------------------------------------------
  {
    std::vector<float> norms(2 * 4096, 1.0f);
    const uint64_t ke = rd::lili_drafter_key(d, F::Exact, norms.data(), norms.size() * 4),
                   k8 = rd::lili_drafter_key(d, F::Int8, norms.data(), norms.size() * 4),
                   k4 = rd::lili_drafter_key(d, F::Int4G64, norms.data(), norms.size() * 4);
    norms[7] = 1.5f;
    const uint64_t k8b = rd::lili_drafter_key(d, F::Int8, norms.data(), norms.size() * 4);
    CHECK(ke != 0 && k8 != 0 && k4 != 0 && ke != k8 && k8 != k4 && ke != k4 && k8 != k8b);
    CHECK(k8 != rd::drafter_key(d, F::Int8, norms.data(), norms.size() * 4));
    std::printf("4. the key: distinct per format and per norm weights\n");
  }

  // --- 5. Bolt's plan with the drafter ----------------------------------------------------------------------------------
  {
    const size_t card32 = size_t(32) << 30;
    for (F fm : {F::Int8, F::Exact, F::Int4G64}) {
      runtime::bolt::BoltReserve base;
      base.decode_scratch = size_t(100e6);
      base.prefill_scratch = size_t(1.5e9);
      base.verify_slots = 7;
      const runtime::bolt::BoltReserve r = rd::lili_bolt_reserve(d, TH, V, fm, ML, base);
      CHECK_EQ(r.drafter_embed_replica, size_t(V) * TH * 2);
      CHECK_EQ(r.drafter, rd::lili_drafter_bytes(d, TH, V, fm, ML));
      CHECK(r.verify_slots == 7 && r.prefill_scratch == size_t(1.5e9));
      const runtime::bolt::BoltPlan p = runtime::bolt::plan(bolt, 0, {card32, card32}, r);
      CHECK_EQ(p.device[1].drafter, r.drafter);
      CHECK_EQ(p.device[0].drafter, size_t(0));
      std::printf("5. Bolt on two cards with the %s drafter (%.3f GB) + the replica: max_len auto %u; device 1 %s\n",
                  loader::draft_format_name(fm), double(r.drafter) / 1e9, p.max_len,
                  runtime::bolt::describe(p.device[1], card32).c_str());
    }
  }
  std::printf("lilicorr_plan_test OK\n");
  return 0;
}
