// Spec 23b Task 2: loader/muse_layout.h - every Muse-Glimmer allocation's bytes from the one formula, against
// spec 23 §2's derived numbers (docs/probe-muse-glimmer-2026-10-15.md, "Bytes"). Host only.
#include <cstdio>

#include "check.h"
#include "loader/int4_layout.h"
#include "loader/kolibri1_layout.h"
#include "loader/muse_layout.h"
#include "model/kolibri1.h"
#include "model/muse_glimmer.h"

int main() {
  const model::MuseGlimmerDesc d = model::muse_glimmer();
  const loader::MuseLayerBytes b = loader::muse_layer_bytes(d);

  // The formula (layout 0 = words [K/8][N] + f16 scales [K/64][N]), then its value.
  CHECK_EQ(b.qgkv, size_t(6656 / 8) * 8704 * 4 + size_t(6656 / 64) * 8704 * 2);
  CHECK_EQ(b.oproj, size_t(4096 / 8) * 6656 * 4 + size_t(4096 / 64) * 6656 * 2);
  CHECK_EQ(b.gate_up, size_t(6656 / 8) * 39936 * 4 + size_t(6656 / 64) * 39936 * 2);
  CHECK_EQ(b.down, size_t(19968 / 8) * 6656 * 4 + size_t(19968 / 64) * 6656 * 2);
  CHECK_EQ(b.norms, size_t(4) * 6656 * 4);
  std::printf("  a layer: q||gate||k||v %zu, o_proj %zu, gate||up %zu, down %zu = int4 %zu B; norms %zu; total %zu\n",
              b.qgkv, b.oproj, b.gate_up, b.down, b.int4(), b.norms, b.total());
  CHECK_EQ(b.int4(), size_t(257081344));                    // spec 23 §2: 257.08 MB, no layout padding
  // Layout 1 holds the same bytes at every one of these shapes (no padding: N % 16 == 0, K % 64 == 0).
  for (model::MuseLinearId id : d.layer_linears()) {
    const model::GemvShape s = d.linear(id).shape;
    CHECK_EQ(loader::int4_layout1_bytes(s.K, s.N), loader::int4_layout0_bytes(s.K, s.N));
  }
  CHECK_EQ(size_t(52) * b.int4(), size_t(13368229888ull)); // 13.368 GB
  CHECK_EQ(b.total(), size_t(257187840));

  CHECK_EQ(loader::muse_embed_bytes(d), size_t(2689662976ull));
  CHECK_EQ(loader::muse_lm_head_bytes(d, true), size_t(1345639680ull));
  CHECK_EQ(loader::muse_lm_head_bytes(d, false), size_t(2689662976ull));
  CHECK_EQ(loader::muse_final_norm_bytes(d), size_t(26624));
  CHECK_EQ(loader::muse_ring_slots(), size_t(4096));
  CHECK_EQ(loader::muse_ring_bytes_per_layer(d), size_t(4194304));
  CHECK_EQ(size_t(d.sliding_layers()) * loader::muse_ring_bytes_per_layer(d), size_t(163577856));
  CHECK_EQ(loader::muse_kv_bytes_per_position(d), size_t(13312));
  CHECK_EQ(loader::muse_full_layer_kv_bytes(d, 131072), size_t(131072) * 1024);
  CHECK_EQ(loader::muse_kv_bytes_per_position(d) * 131072, size_t(1744830464ull));   // 1.745 GB

  // The device split: one card holds 17.41 GB with the int8 head; two cards hold the same bytes in all.
  const model::MusePlacement one = model::MusePlacement::one(d), two = model::MusePlacement::two(d, 26);
  CHECK_EQ(loader::muse_device_weight_bytes(d, one, 0, true), size_t(17409096960ull));
  const size_t d0 = loader::muse_device_weight_bytes(d, two, 0, true), d1 = loader::muse_device_weight_bytes(d, two, 1, true);
  CHECK_EQ(d0, size_t(26) * b.total() + loader::muse_embed_bytes(d));
  CHECK_EQ(d1, size_t(26) * b.total() + 26624 + size_t(1345639680ull));
  CHECK_EQ(d0 + d1, size_t(17409096960ull));
  CHECK_EQ(loader::muse_device_weight_bytes(d, one, 0, false), size_t(17409096960ull) - 1345639680ull + 2689662976ull);
  std::printf("  one card: %zu B (int8 head), %zu B (bf16 head); split 26: %zu + %zu B\n",
              loader::muse_device_weight_bytes(d, one, 0, true), loader::muse_device_weight_bytes(d, one, 0, false), d0, d1);

  // Read per decoded token (derived): 14.72 GB; at depth 4096 / 131072 with the KV and up to 39 windows of 2048.
  const size_t rpt = loader::muse_read_per_token(d, true);
  CHECK_EQ(rpt, size_t(14719433984ull));
  CHECK_EQ(loader::muse_layer_read_per_token(d), b.total());
  const size_t win = size_t(d.sliding_layers()) * 2048 * d.kv_n() * 2 * 2;
  std::printf("  read per token: %zu B (%.3f GB); at depth 4096 %.3f GB, 131072 %.3f GB (+ the windows' %zu B)\n", rpt,
              rpt / 1e9, (rpt + 13312.0 * 4096 + win) / 1e9, (rpt + 13312.0 * 131072 + win) / 1e9, win);

  // A truncated descriptor (development mode): 4 layers, 1 full.
  {
    model::MuseGlimmerDesc d4 = d;
    d4.layers = 4;
    d4.full.resize(4);
    CHECK_EQ(loader::muse_kv_bytes_per_position(d4), size_t(1024));
    CHECK_EQ(loader::muse_device_weight_bytes(d4, model::MusePlacement::one(d4), 0, true),
             size_t(4) * b.total() + 2689662976ull + 26624 + 1345639680ull);
  }

  // Kolibri's aliases stay byte-identical to the shared formulas (its own tests pin its totals).
  CHECK_EQ(loader::kol_layout0_bytes(2560, 7168), loader::int4_layout0_bytes(2560, 7168));
  CHECK_EQ(loader::kol_layout1_bytes(2560, 1024), loader::int4_layout1_bytes(2560, 1024));
  CHECK_EQ(loader::kol_layer_bytes(model::kolibri1()).total(), size_t(830794752));

  std::printf("muse_layout_test OK\n");
  return 0;
}
