// Spec 20c Task 3: load_kolibri1 on the card (box only; label `checkpoint;kolibri`), one device.
//   kolibri1_load_checkpoint_test <snapshot> [int8|bf16] [layers]
// Loads the checkpoint (a synthetic one from tools/box_validate/kolibri_oracle.sh synth, or the real
// one truncated to its first `layers` - development mode), prints the report, and asserts: zero
// unconsumed tensors, the part's bytes equal to kol_device_weight_bytes (the planner's figure), the
// RoPE table's size, and device bytes read back against the mmapped source at the edges a wrong
// stride breaks (the last expert's gate||up block, the router's padded rows, the bias tail).
// Exit 77 (SKIP) when the snapshot is not on this machine.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/kolibri1_layout.h"
#include "loader/kolibri1_loader.h"
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
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ";
  const bool int8 = argc > 2 && std::string(argv[2]) == "int8";
  const uint32_t layers = argc > 3 ? uint32_t(std::stoul(argv[3])) : 0;
  std::string snap;
  try {
    snap = loader::resolve_snapshot(arg);
    if (!loader::is_kolibri1_checkpoint(snap)) throw std::runtime_error("not a kolibri1 checkpoint");
  } catch (const std::exception& e) {
    std::printf("SKIP: no Kolibri-1 checkpoint at '%s' (%s)\n", arg.c_str(), e.what());
    return 77;
  }
  const model::Kolibri1Desc pre = loader::kolibri1_checkpoint_desc(snap, layers);
  l0::Context ctx(0);
  std::vector<l0::Context*> devs = {&ctx};
  const uint32_t max_len = 8192;
  loader::KolLoadedModel m = loader::load_kolibri1(devs, snap, max_len, model::KolPlacement::one(pre),
                                                   int8 ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint,
                                                   layers);
  const model::Kolibri1Desc& d = m.desc;
  CHECK(d.layers == pre.layers && d.attn == pre.attn);
  CHECK_EQ(m.unconsumed, size_t(0));
  CHECK_EQ(m.parts.size(), size_t(1));
  const loader::KolDevicePart& P = m.parts[0];
  CHECK_EQ(P.layers.size(), size_t(d.layers));
  CHECK_EQ(P.bytes, loader::kol_device_weight_bytes(d, m.placement, 0, int8));
  CHECK_EQ(P.rope_bytes, d.rope_table_bytes(max_len));
  CHECK(P.embed && P.lm_head && P.final_norm && P.rope);

  loader::SafetensorsSet set(snap);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const uint32_t last = d.layers - 1;
  const std::string lp = model::Kolibri1Desc::layer_prefix(last);
  const loader::KolLayer& L = P.layer(last);
  // The last expert's gate||up block: tile (n_tile 0, g 0) word j of lane l is gate_proj qweight
  // row j column l (the interleave's first 16 columns are gate's).
  {
    const loader::TensorInfo t = set.tensors().at(lp + "mlp.experts." + std::to_string(d.experts - 1) + ".gate_proj.qweight");
    const uint32_t* q = reinterpret_cast<const uint32_t*>(set.data(t));
    const size_t blk = loader::kol_gate_up_block_bytes(d);
    const std::vector<uint8_t> b = readback(imm, *L.gate_up, (d.experts - 1) * blk, 136 * 4);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(b.data());
    for (uint32_t j = 0; j < 8; ++j)
      for (uint32_t l = 0; l < 16; ++l) CHECK(w[j * 16 + l] == q[size_t(j) * d.moe_inter + l]);
  }
  // The router's padded rows are zero; the bias tail is zero.
  {
    const size_t row_tile = size_t(d.hidden / 8) * 128 * 2;   // one 16-row tile of gemv_bf16
    const std::vector<uint8_t> b = readback(imm, L.router->mem, (d.experts / 16) * row_tile, row_tile);
    for (uint8_t x : b) CHECK(x == 0);
    const std::vector<uint8_t> bb = readback(imm, *L.bias, d.experts * 4, (d.router_n() - d.experts) * 4);
    for (uint8_t x : bb) CHECK(x == 0);
  }
  std::printf("kolibri1_load_checkpoint_test OK: %u layers, %s attention, %s head, %zu B of weights\n", d.layers,
              model::kol_attn_form_name(d.attn), int8 ? "int8" : "bf16", P.bytes);
  return 0;
}
