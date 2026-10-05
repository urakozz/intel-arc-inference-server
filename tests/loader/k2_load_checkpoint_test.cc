// Spec 18b Task 1: load_k2 against the real checkpoint on the card (box only; label
// `checkpoint;k2`). Every bucket equals loader::k2_weight_bytes (the planner's figure),
// nothing is unconsumed, and device bytes read back against the mmapped source at the
// edges a wrong stride or a wrong fuse would break. argv: <snapshot> [int8]. Exit 77
// (SKIP) when the checkpoint is not on this machine.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_layout.h"
#include "loader/k2_loader.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace {
std::vector<uint8_t> readback(l0::CmdList& imm, const l0::Mem& m, size_t off, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, bytes);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ";
  const bool int8 = argc > 2 && std::string(argv[2]) == "int8";
  std::string snap;
  try {
    snap = loader::resolve_snapshot(arg);
  } catch (const std::exception& e) {
    std::printf("SKIP: no K2-Horizon checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  l0::Context ctx(0);
  loader::K2LoadedModel m =
      loader::load_k2(ctx, snap, 16384, int8 ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint);
  const model::K2Desc& d = *m.desc;
  const loader::K2LoadReport& r = m.report;
  CHECK_EQ(r.unconsumed, size_t(0));
  CHECK_EQ(m.layers.size(), size_t(48));
  CHECK_EQ(m.trained_max_len, 524288u);
  const loader::K2WeightBytes want = loader::k2_weight_bytes(d, int8);
  CHECK_EQ(r.bytes.total(), want.total());
  CHECK_EQ(r.bytes.total(), int8 ? size_t(21160906496ull) : size_t(21801501440ull));
  CHECK_EQ(r.read_per_token, int8 ? size_t(3145142016ull) : size_t(3785736960ull));
  CHECK_EQ(r.rope_bytes, d.rope_table_bytes(16384));

  loader::SafetensorsSet set(snap);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto src = [&](const std::string& name) {
    const loader::TensorInfo t = set.tensors().at(name);
    return std::make_pair(set.data(t), set.bytes(t));
  };

  // A value expert block, the last block of the last layer: tile (n_tile 0, g 0) word j of
  // lane 0 is qweight row j, column 0 of v_experts.63.
  {
    const loader::K2ExpertBytes eb = loader::k2_expert_bytes(d);
    const auto [qw, qwb] = src("model.layers.47.self_attn.v_experts.63.qweight");
    (void)qwb;
    const std::vector<uint8_t> dev = readback(imm, *m.layers[47].value, 63 * eb.value_block, 136 * 4);
    for (uint32_t j = 0; j < 8; ++j) {
      uint32_t got, want_w;
      std::memcpy(&got, dev.data() + size_t(j * 16) * 4, 4);
      std::memcpy(&want_w, qw + size_t(j) * 1024 * 4, 4);
      CHECK_EQ(got, want_w);
    }
    // The shared expert is gate||up's last block: column 16 is up_proj column 0.
    const auto [up, upb] = src("model.layers.3.mlp.shared_experts.up_proj.qweight");
    (void)upb;
    const std::vector<uint8_t> g = readback(imm, *m.layers[3].gate_up, 100 * eb.gate_up_block +
                                                  size_t(1) * (2560 / 64) * 136 * 4, 136 * 4);
    uint32_t got, want_w;
    std::memcpy(&got, g.data(), 4);   // n_tile 1 (columns 16..31), g 0, word 0, lane 0
    std::memcpy(&want_w, up, 4);
    CHECK_EQ(got, want_w);
  }
  // The fused MoVA row: column 9216 is v_router column 0 (layout 0, row 0).
  {
    const loader::DeviceWeight& w = m.layers[20].linears[0];
    CHECK_EQ(w.shape.N, 9280u);
    const auto [vr, vrb] = src("model.layers.20.self_attn.v_router.qweight");
    (void)vrb;
    const std::vector<uint8_t> dev = readback(imm, w.mem, size_t(d.v_off()) * 4, 4);
    CHECK(std::memcmp(dev.data(), vr, 4) == 0);
  }
  // Plain w, widened; the F16 MoVA bias widened.
  {
    const auto [nw, nwb] = src("model.layers.5.post_attention_layernorm.weight");
    (void)nwb;
    const std::vector<uint8_t> dev = readback(imm, *m.layers[5].norms, d.norms_off_post(), 4 * 2560);
    for (uint32_t i : {0u, 1279u, 1280u, 2559u}) {
      uint16_t w16;
      float f;
      std::memcpy(&w16, nw + size_t(i) * 2, 2);
      std::memcpy(&f, dev.data() + size_t(i) * 4, 4);
      CHECK(f == common::bf16_to_f32(w16));
    }
    const auto [vb, vbb] = src("model.layers.30.self_attn.v_router.bias");
    (void)vbb;
    const std::vector<uint8_t> rb = readback(imm, *m.layers[30].route, d.route_off_mova(), 4 * 64);
    for (uint32_t i : {0u, 63u}) {
      uint16_t h;
      float f;
      std::memcpy(&h, vb + size_t(i) * 2, 2);
      std::memcpy(&f, rb.data() + size_t(i) * 4, 4);
      CHECK(f == common::f16_to_f32(h));
    }
  }
  // The router's pad rows 100..127 are zero on the card (Review Focus 4).
  {
    const loader::DeviceWeight& w = *m.layers[3].router;
    CHECK_EQ(w.mem.size(), size_t(128) * 2560 * 2);
    const std::vector<uint8_t> dev = readback(imm, w.mem, 0, w.mem.size());
    const uint16_t* t = reinterpret_cast<const uint16_t*>(dev.data());
    for (uint32_t n = 100; n < 128; ++n)
      for (uint32_t k = 0; k < 2560; k += 37)
        CHECK_EQ(t[((size_t(n / 16) * 320 + k / 8) * 8 + k % 8) * 16 + n % 16], uint16_t(0));
  }
  std::printf("k2_load_checkpoint_test OK: %.3f GB resident, %.3f GB read per token, %.1f s\n",
              r.total() / 1e9, r.read_per_token / 1e9, r.seconds);
  return 0;
}
