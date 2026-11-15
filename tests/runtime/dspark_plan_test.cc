// Kolibri DSpark (host): the DSpark drafter's buffers and lists as device-free arithmetic - runtime/dflash/dflash_sizes.h's
// formulas on the published config (tests/model/dspark/config.json). No device.
//
//   1. The buffers at max_len 262144 beside Kolibri-1's 128000-row head: the ring 5 x 2048 x 512 x (K + V) bf16 =
//      20,971,520 B (every layer slides), the taps 8 x 5 x 2560 bf16, the walk's hidden rows of ones 8 x 256 fp32, the head
//      rows 7 x 128000 fp32, the top-16 partials; the drafter's whole bytes (bf16 weights + buffers) printed (DERIVED).
//   2. The lists' launch counts: draft 66 (embed, 5 x 12, norm, head, top-k x 2, the walk), projection 8.
//   3. The snapshot's share (the ring: 20,971,520 B at any max_len) and the prefix store's key: distinct from a DFlash v1
//      drafter of the same shapes (the architecture is folded in) and per norm weights; never 0.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/json.h"
#include "loader/dflash_layout.h"
#include "model/dflash.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: dspark_plan_test <dspark config.json>\n");
    return 2;
  }
  std::ifstream f(argv[1]);
  std::stringstream ss;
  ss << f.rdbuf();
  const model::DFlashDesc d = model::dflash_desc_from_config(common::json::parse(ss.str()));
  model::check_dspark(d);
  namespace rd = runtime::dflash;
  const uint32_t TH = 2560, V = 128000, ML = 262144;

  // --- 1. the buffers ---------------------------------------------------------------------------------------------------
  const rd::DFlashSizes s = rd::sizes(d, TH, V, ML, loader::DraftFormat::Bf16);
  CHECK_EQ(s.ring_k + s.ring_v, size_t(20971520));
  CHECK_EQ(s.taps, size_t(8) * 5 * 2560 * 2);
  CHECK_EQ(s.hid, size_t(8) * 256 * 4);
  CHECK_EQ(s.co, size_t(0));
  CHECK_EQ(s.logits, size_t(7) * V * 4);
  CHECK_EQ(s.tpart, size_t(7) * ((V + 4095) / 4096) * 16 * 2 * 4);
  CHECK_EQ(s.part, size_t(20) * (2048 / 128) * 8 * 130 * 4);
  CHECK_EQ(s.rope, size_t(ML + 8) * 128 * 4);
  CHECK_EQ(s.dctl, sizeof(rd::DFlashCtl));
  CHECK_EQ(s.out, sizeof(rd::DFlashDraftOut));
  const loader::DFlashBytes w = loader::dflash_weight_bytes(d, TH, loader::DraftFormat::Bf16);
  std::printf("1. buffers %zu B (the ring %zu B), bf16 weights %zu B (the Markov tables %zu B): %zu B on Kolibri's last card "
              "+ the 655,360,000 B embedding replica (DERIVED)\n",
              s.total(), s.ring_k + s.ring_v, w.total(), w.codebooks, s.total() + w.total());

  // --- 2. the lists -----------------------------------------------------------------------------------------------------
  CHECK_EQ(rd::draft_launches(d), size_t(66));
  CHECK_EQ(rd::project_launches(d), size_t(8));
  std::printf("2. the draft list 66 launches, the projection 8\n");

  // --- 3. the snapshot and the key --------------------------------------------------------------------------------------
  CHECK_EQ(rd::ring_snapshot_bytes(d, ML), size_t(20971520));
  CHECK_EQ(rd::ring_snapshot_bytes(d, 4096), size_t(20971520));
  const std::vector<float> n1(2 * 2560, 1.0f);
  std::vector<float> n2 = n1;
  n2[7] = 1.5f;
  const uint64_t k1 = rd::drafter_key(d, loader::DraftFormat::Bf16, n1.data(), n1.size() * 4);
  const uint64_t k2 = rd::drafter_key(d, loader::DraftFormat::Bf16, n2.data(), n2.size() * 4);
  model::DFlashDesc v1 = d;
  v1.architecture = "DFlashDraftModel";
  const uint64_t k3 = rd::drafter_key(v1, loader::DraftFormat::Bf16, n1.data(), n1.size() * 4);
  CHECK(k1 != 0 && k2 != 0 && k1 != k2 && k1 != k3);
  std::printf("3. the ring snapshot 20,971,520 B at any max_len; keys distinct per norms and form\n");
  std::printf("dspark_plan_test: OK\n");
  return 0;
}
