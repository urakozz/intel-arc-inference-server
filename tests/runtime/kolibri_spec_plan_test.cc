// Kolibri DSpark (host): the verify lists' and the drafter's place in Kolibri-1's plan - runtime/kolibri/kolibri_sizes.h's
// arithmetic on model::kolibri1(). No device.
//
//   1. The verify scratch at 8 rows: every term the decode scratch's x 8 but the attention parts (kol_attn.cl's 4 rows), the
//      second Control block, the verify link (8 rows + [20][8] sums) on device 1 only.
//   2. The verify lists' launches: the flash decode list's at M <= 4, + 2 a layer at M 5..8 (the second attention half).
//   3. The drafter's prefill tail: the rows inside its 2048 window of the call's end, and with a block hook the last 2048
//      rows of a chunk ending at a mid-prompt block end.
//   4. The plan with --spec dflash on two 32 GiB cards: the verify terms on both, the drafter (bf16 weights + buffers +
//      staging) and its 655,360,000 B replica on device 1; `--max-len auto` still fits (DERIVED, printed), never above the
//      plain plan's, and the split's auto choice moves layers toward device 0.
#include <array>
#include <cstdio>
#include <vector>

#include "check.h"
#include "model/kolibri1.h"
#include "runtime/control.h"
#include "runtime/kolibri/kolibri_sizes.h"

int main() {
  namespace rk = runtime::kolibri;
  const model::Kolibri1Desc& d = model::kolibri1();

  // --- 1. the verify scratch --------------------------------------------------------------------------------------------
  {
    const rk::ScratchSizes one = rk::scratch_sizes(d), v = rk::verify_sizes(d).scratch;
    CHECK_EQ(v.resid, one.resid * 8);
    CHECK_EQ(v.partials, one.partials * 8);
    CHECK_EQ(v.sumsq_r, one.sumsq_r * 8);
    CHECK_EQ(v.logits, one.logits * 8);
    CHECK_EQ(v.routes, one.routes * 8);
    CHECK_EQ(v.moe_h, one.moe_h * 8);
    CHECK_EQ(v.attn_part, one.attn_part * 4);
    CHECK_EQ(rk::verify_sizes(d).control_hi, sizeof(runtime::Control));
    CHECK_EQ(rk::verify_link_bytes(d, 0), size_t(0));
    CHECK(rk::verify_link_bytes(d, 1) >= size_t(8) * d.hidden * 2 + 20 * 8 * 4);
    CHECK_EQ(rk::route_at_rows(3, 8), size_t(3) * 8 * 32 * 4);
    CHECK_EQ(rk::route_at_rows(3, 1), rk::route_at(3));
    std::printf("1. the verify scratch %zu B a device (the decode scratch's %zu B), the link %zu B on device 1\n",
                rk::verify_sizes(d).total(), one.total(), rk::verify_link_bytes(d, 1));
  }

  // --- 2. launches --------------------------------------------------------------------------------------------------------
  {
    const model::KolPlacement p = model::KolPlacement::two(d, 25);
    for (uint32_t M = 1; M <= 8; ++M)
      for (uint32_t dev = 0; dev < 2; ++dev)
        CHECK_EQ(rk::verify_device_launches(d, p, dev, M),
                 rk::device_launches(d, p, dev, rk::KolAttn::Flash) + (M > 4 ? size_t(25) * 2 : 0));
    std::printf("2. verify launches: %zu (M <= 4), %zu (M 5..8) on two cards\n",
                rk::verify_device_launches(d, p, 0, 4) + rk::verify_device_launches(d, p, 1, 4),
                rk::verify_device_launches(d, p, 0, 8) + rk::verify_device_launches(d, p, 1, 8));
  }

  // --- 3. the prefill tail ------------------------------------------------------------------------------------------------
  {
    const uint32_t W = 2048, B = 2048;
    CHECK_EQ(rk::drafter_prefill_r0(0, 1000, 1000, W, false, B), 0u);          // the whole short prompt
    CHECK_EQ(rk::drafter_prefill_r0(0, 2048, 5000, W, false, B), 2048u);       // the first chunk: outside the window
    CHECK_EQ(rk::drafter_prefill_r0(2048, 2048, 5000, W, false, B), 2952u - 2048u);
    CHECK_EQ(rk::drafter_prefill_r0(4096, 904, 5000, W, false, B), 0u);
    CHECK_EQ(rk::drafter_prefill_r0(0, 2048, 5000, W, true, B), 0u);           // a block end: its window, for the snapshot
    CHECK_EQ(rk::prefill_taps_bytes(W, 5, 2560), size_t(52428800));
    std::printf("3. the prefill tail rule (window 2048, block 2048)\n");
  }

  // --- 4. the plan --------------------------------------------------------------------------------------------------------
  {
    const std::array<size_t, runtime::kPpDevices> dev{size_t(32) << 30, size_t(32) << 30};
    const size_t reserve = size_t(1) << 30;
    rk::SpecReserve s;
    s.verify = true;
    s.drafter = 826'000'000 + 160'000'000 + rk::prefill_taps_bytes(2048, 5, 2560);   // DERIVED: bf16 weights + buffers
    s.replica = size_t(d.vocab) * d.hidden * 2;
    const runtime::PpChoice plain = rk::pp_split_and_len(d, true, dev, reserve, 0, true);
    const runtime::PpChoice spec = rk::pp_split_and_len(d, true, dev, reserve, 0, true, s);
    CHECK(plain.max_len > 0 && spec.max_len > 0 && spec.max_len <= plain.max_len);
    CHECK(spec.split >= plain.split);
    const std::vector<rk::DevicePlan> pl = rk::plan(d, model::KolPlacement::two(d, spec.split), spec.max_len, true, false,
                                                    rk::KolAttn::Flash, true, s);
    CHECK_EQ(pl[0].drafter, size_t(0));
    CHECK_EQ(pl[1].drafter, s.drafter + s.replica);
    std::printf("4. two 32 GiB cards, int8 head, prefill: plain max_len %u split %u; with DSpark max_len %u split %u "
                "(DERIVED)\n", plain.max_len, plain.split, spec.max_len, spec.split);
  }
  std::printf("kolibri_spec_plan_test: OK\n");
  return 0;
}
