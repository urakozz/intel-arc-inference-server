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
    CHECK_EQ(p.rope, loader::lfm::rope_table_bytes(d, 131072));
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

  // --- spec 24c: the decode list's launch counts (spec 24 §4.5, plan 24c Task 4) -------------------------------
  {
    const LfmFuse none{}, all{true, true, true};
    // 2 dense conv layers x 8 + 28 conv + MoE x 9 + 10 attention + MoE x 11 + the head's 5 (the formula, spelled out)
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, none, runtime::PpHandoff::Copy), size_t(2 * 8 + 28 * 9 + 10 * 11 + 5));
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, none, runtime::PpHandoff::Copy), size_t(383));
    CHECK_EQ(decode_launches(d, one, LfmAttn::Eager, none, runtime::PpHandoff::Copy), size_t(403));   // +2 a layer
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, LfmFuse{true, false, false}, runtime::PpHandoff::Copy), size_t(345));
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, LfmFuse{false, true, false}, runtime::PpHandoff::Copy), size_t(345));
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, LfmFuse{false, false, true}, runtime::PpHandoff::Copy), size_t(353));
    CHECK_EQ(decode_launches(d, one, LfmAttn::Flash, all, runtime::PpHandoff::Copy), size_t(277));
    // the synthetic N 4: dense conv 0, 1 (8 + 8), attention + MoE 2 (11), conv + MoE 3 (9), the head (5)
    model::Lfm2MoeDesc d4 = d;
    d4.layers = 4;
    d4.kinds.assign(d.kinds.begin(), d.kinds.begin() + 4);
    const model::LfmPlacement one4 = model::LfmPlacement::one(d4);
    CHECK_EQ(decode_launches(d4, one4, LfmAttn::Flash, none, runtime::PpHandoff::Copy), size_t(8 + 8 + 11 + 9 + 5));
    CHECK_EQ(decode_launches(d4, one4, LfmAttn::Flash, none, runtime::PpHandoff::Copy), size_t(41));
    // two cards: +0 with copy (the hand-off is copies and an event), +2 with peer (pp_send, pp_recv)
    const model::LfmPlacement two = model::LfmPlacement::two(d, 20);
    CHECK_EQ(decode_launches(d, two, LfmAttn::Flash, none, runtime::PpHandoff::Copy), size_t(383));
    CHECK_EQ(decode_launches(d, two, LfmAttn::Flash, none, runtime::PpHandoff::Peer), size_t(385));
    CHECK_EQ(device_launches(d, two, 0, LfmAttn::Flash, none, runtime::PpHandoff::Copy) +
                 device_launches(d, two, 1, LfmAttn::Flash, none, runtime::PpHandoff::Copy),
             size_t(383));
    const model::LfmPlacement two4 = model::LfmPlacement::two(d4, 2);
    CHECK_EQ(decode_launches(d4, two4, LfmAttn::Flash, all, runtime::PpHandoff::Peer), size_t(41 - 3 - 4 + 2));
    // the profile's event pool holds every list
    CHECK(decode_launches(d, one, LfmAttn::Eager, none, runtime::PpHandoff::Peer) <= size_t(1024));
    // B70_LFM_FUSE / B70_LFM_ATTN parse
    CHECK(parse_fuse("none").route == false && parse_fuse("all").conv == true);
    const LfmFuse rc = parse_fuse("route,conv");
    CHECK(rc.route && !rc.router && rc.conv);
    refused("B70_LFM_FUSE", [] { (void)parse_fuse("route,bogus"); });
    refused("B70_LFM_ATTN", [] { (void)parse_attn("fast"); });
    CHECK(parse_attn("eager") == LfmAttn::Eager && parse_attn("") == LfmAttn::Flash);
  }

  // --- spec 24c: the decode scratch, and the plan at 131072 on one card ------------------------------------------
  {
    const ScratchSizes f = scratch_sizes(d, 131072, LfmAttn::Flash), e = scratch_sizes(d, 131072, LfmAttn::Eager);
    CHECK_EQ(f.h0, size_t(2048) * 2);
    CHECK_EQ(f.h1, f.h0);
    CHECK_EQ(f.slices, size_t(23552) * 4);   // the widest S x N: dense gate||up at S 1 (24b's descriptor)
    CHECK_EQ(f.routes, size_t(40) * 16 * 4);
    CHECK_EQ(f.moe_h, size_t(4) * 1536 * 2);
    CHECK_EQ(f.s_mlp, size_t(11776) * 2);
    CHECK_EQ(f.logits, size_t(128000) * 4);
    CHECK_EQ(f.attn_scores, size_t(0));
    CHECK_EQ(e.attn_scores, size_t(32) * 131072 * 4);   // eager's score row: the term that scales (16 MiB)
    CHECK_EQ(e.total() - f.total(), e.attn_scores);
    CHECK(f.total() < size_t(2) << 20);
    const Plan p = plan(d, one, 0, 131072, true, LfmAttn::Eager);
    CHECK_EQ(p.decode_scratch, e.total() + size_t(128));   // + the Control block
    require_fits(p, size_t(32530000000ull));             // the whole model and 131072 positions on one B70
    std::printf("  decode scratch at 131072: flash %zu B, eager %zu B; %s\n", f.total(), e.total(),
                describe(p, size_t(32530000000ull)).c_str());
  }

  std::printf("lfm2moe_plan_test: ok\n");
  return 0;
}
