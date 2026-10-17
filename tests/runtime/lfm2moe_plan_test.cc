// Spec 24b Task 5: runtime::lfm2moe's memory plan, max_len auto and the two-card split, host only.
#include <cstdio>
#include <string>

#include "check.h"
#include "common/json.h"
#include "loader/lfm2moe_layout.h"
#include "model/lfm2moe.h"
#include "runtime/lfm2moe/lfm2moe_sizes.h"

namespace {

template <class F>
void refused(const char* key, F fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception& e) {
    threw = std::string(e.what()).find(key) != std::string::npos;
    if (!threw) std::fprintf(stderr, "refusal of %s named something else: %s\n", key, e.what());
  }
  if (!threw) std::fprintf(stderr, "not refused: %s\n", key);
  CHECK(threw);
}

}  // namespace

int main() {
  using namespace runtime::lfm2moe;
  const model::Lfm2MoeDesc& d = model::lfm2moe();
  const model::LfmPlacement one = model::LfmPlacement::one(d);

  // --- one card, int8 head, at the trained max_len --------------------------------------------
  {
    const Plan p = plan(d, one, 0, 131072, true);
    CHECK_EQ(p.weights, loader::lfm::weight_bytes(d, true));
    CHECK(p.weights > size_t(13.3e9) && p.weights < size_t(13.5e9));
    CHECK_EQ(p.kv, size_t(d.attn_layers()) * 131072 * d.kv_heads * d.head_dim * 2 * 2);
    CHECK(p.kv > size_t(2.68e9) && p.kv < size_t(2.69e9));
    CHECK_EQ(p.conv_rings, loader::lfm::conv_ring_bytes(d));
    CHECK_EQ(p.rope, loader::lfm::rope_table_bytes(131072));
    CHECK_EQ(p.reserve, size_t(1.5e9));
    const size_t device_32gib = size_t(32) << 30;
    require_fits(p, device_32gib);   // must not throw
    std::printf("  %s\n", describe(p, device_32gib).c_str());
  }

  // --- the bf16-head plan: no int8 copy, the embedding tiles serve as the head -----------------
  {
    const Plan pi = plan(d, one, 0, 4096, true);
    const Plan pb = plan(d, one, 0, 4096, false);
    // one-device placement: the embedding tiles already serve the gather, so the bf16-head plan
    // needs no replica - its weights are exactly the int8 plan's minus the int8 head's own bytes.
    CHECK_EQ(pb.weights, pi.weights - loader::lfm::head_int8_bytes(d));
  }

  // --- max_len_auto: 131072 on a 32 GiB device (capped by the trained context) -----------------
  {
    const size_t device_32gib = size_t(32) << 30;
    const uint32_t len = max_len_auto(d, one, device_32gib, 1.5, true);
    CHECK_EQ(len, d.trained_max_len);
  }

  // --- max_len_auto: the largest fitting multiple of 256 on a crafted 15 GiB device ------------
  {
    const size_t device_15gib = size_t(15) << 30;
    const uint32_t len = max_len_auto(d, one, device_15gib, 1.5, true);
    CHECK(len > 0 && len < d.trained_max_len && len % 256 == 0);
    require_fits(plan(d, one, 0, len, true), device_15gib);
    bool next_fits = true;
    try {
      require_fits(plan(d, one, 0, len + 256, true), device_15gib);
    } catch (const std::exception&) {
      next_fits = false;
    }
    CHECK(!next_fits);   // len is the LARGEST that fits
    std::printf("  max_len_auto(15 GiB) = %u\n", len);
  }

  // --- an explicit max_len above the trained context is refused naming max_position_embeddings --
  refused("max_position_embeddings", [&] { (void)plan(d, one, 0, 262144, true); });

  // --- pp_split: balanced within one layer's bytes ----------------------------------------------
  {
    const uint32_t split = pp_split(d, true);
    CHECK(split >= 1 && split < d.layers);
    size_t dev0 = loader::lfm::embed_tiled_bytes(d), dev1 = size_t(d.hidden) * 4 + loader::lfm::head_int8_bytes(d);
    for (uint32_t l = 0; l < split; ++l) dev0 += loader::lfm::layer_weight_bytes(d, l);
    for (uint32_t l = split; l < d.layers; ++l) dev1 += loader::lfm::layer_weight_bytes(d, l);
    const size_t diff = dev0 > dev1 ? dev0 - dev1 : dev1 - dev0;
    const size_t one_layer = loader::lfm::layer_weight_bytes(d, split > 0 ? split - 1 : split);
    CHECK(diff <= one_layer * 2);   // balanced to within about one layer's bytes either way
    std::printf("  pp_split(int8 head) = %u (dev0 %zu B, dev1 %zu B, diff %zu B)\n", split, dev0, dev1, diff);
  }

  // --- synthetic N = 4 plans ----------------------------------------------------------------------
  {
    model::Lfm2MoeDesc d4 = d;
    d4.layers = 4;
    d4.kinds.assign(d.kinds.begin(), d.kinds.begin() + 4);
    const model::LfmPlacement one4 = model::LfmPlacement::one(d4);
    const Plan p4 = plan(d4, one4, 0, 4096, true);
    CHECK_EQ(p4.weights, loader::lfm::weight_bytes(d4, true));
    CHECK(p4.weights < plan(d, one, 0, 4096, true).weights);
    const uint32_t split4 = pp_split(d4, true);
    CHECK(split4 >= 1 && split4 < d4.layers);
  }

  std::printf("lfm2moe_plan_test: ok\n");
  return 0;
}
