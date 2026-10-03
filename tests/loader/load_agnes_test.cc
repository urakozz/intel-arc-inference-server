// Spec 14: the Agnes 3.0 Flash checkpoint through the loader - descriptor,
// name map, the parallel-FFN fold on device, the MTP head under Agnes's names,
// the max_len ceiling. Device + checkpoint (label `checkpoint-agnes`).
//
//   load_agnes_test [repo-or-snapshot]   (default urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ)
//
// Byte totals are the checkpoint's own (summed from its safetensors headers,
// 2026-10-03): 14,816,378,880 B of qweight and 926,023,680 B of scales over the
// 72 layers, the parallel FFN included - the fold moves bytes, it adds none.
#include <cstdio>
#include <stdexcept>
#include <string>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/model_desc.h"

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ";
  l0::Context ctx(0);

  // The ceiling first: it throws after config.json, before a byte is uploaded.
  bool refused = false;
  try {
    loader::load(ctx, arg, 131072);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("65536") != std::string::npos;
  }
  CHECK(refused);

  loader::LoadedModel m = loader::load(ctx, arg, 16384, /*mtp=*/true);
  CHECK(m.desc == &model::agnes());
  CHECK_EQ(m.layer_small.size(), size_t(72));
  CHECK_EQ(m.linears.size(), size_t(54 * 5 + 18 * 4 + 1));
  CHECK_EQ(m.report.unconsumed, size_t(0));
  CHECK_EQ(m.report.int4_bytes, size_t(14816378880));
  CHECK_EQ(m.report.scale_bytes, size_t(926023680));
  CHECK_EQ(m.report.lm_head_bytes, size_t(2542796800));   // bf16, as shipped
  // The MTP head: Agnes's `mtp.layers.0.global_attn.*` bound as `self_attn.*`.
  CHECK(m.mtp != nullptr);
  CHECK_EQ(m.report.mtp_tensors, loader::kMtpTensors);
  CHECK_EQ(m.report.mtp_checkpoint_bytes, loader::kMtpCheckpointBytes);

  // The folded shapes, on every layer.
  for (uint32_t l = 0; l < 72; ++l) {
    const loader::DeviceWeight& gu = m.linears.at({l, model::LinearId::GateUp});
    const loader::DeviceWeight& dn = m.linears.at({l, model::LinearId::Down});
    CHECK(gu.shape.K == 5120 && gu.shape.N == 38912 && gu.shape.layout == 0 && gu.scales);
    CHECK(dn.shape.K == 19456 && dn.shape.N == 5120 && dn.shape.layout == 0 && dn.scales);
    CHECK_EQ(gu.mem.size(), size_t(5120 / 8) * 38912 * 4);
    CHECK_EQ(dn.mem.size(), size_t(19456 / 8) * 5120 * 4);
  }

  // Device readback at the joins, against the mmapped checkpoint (Agnes names).
  const std::string snap = loader::resolve_snapshot(arg);
  loader::SafetensorsSet set(snap);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  size_t checked = 0;
  for (uint32_t l : {0u, 3u, 71u}) {
    const std::string p = "model.language_model.layers." + std::to_string(l) + ".mlp.";
    const loader::LinearSrc gate = loader::LinearSrc::classify(set, p + "gate_proj");
    const loader::LinearSrc up_p = loader::LinearSrc::classify(set, p + "parallel_ffn.up_proj");
    const loader::LinearSrc down = loader::LinearSrc::classify(set, p + "down_proj");
    const loader::LinearSrc down_p = loader::LinearSrc::classify(set, p + "parallel_ffn.down_proj");
    CHECK(gate.N == 17408 && up_p.N == 2048 && down.K == 17408 && down_p.K == 2048);
    const loader::DeviceWeight& gu = m.linears.at({l, model::LinearId::GateUp});
    const loader::DeviceWeight& dn = m.linears.at({l, model::LinearId::Down});
    // gate'||up' interleaved in 16-column blocks: gate' column j at (j/16)*32 + j%16,
    // up' column j at that + 16. Columns j >= 17408 are the parallel branch's.
    auto gu_word = [&](uint32_t r, uint32_t col) {
      uint32_t w = 0;
      imm.copy(&w, gu.mem.as<uint32_t>() + size_t(r) * 38912 + col, 4);
      return w;
    };
    auto gu_scale = [&](uint32_t g, uint32_t col) {
      uint16_t s = 0;
      imm.copy(&s, gu.scales->as<uint16_t>() + size_t(g) * 38912 + col, 2);
      return s;
    };
    for (uint32_t j : {0u, 17407u, 17408u, 17408u + 5, 19455u})
      for (uint32_t r : {0u, 639u}) {
        const uint32_t gcol = (j / 16) * 32 + j % 16, ucol = gcol + 16;
        const bool par = j >= 17408;
        if (!par) CHECK_EQ(gu_word(r, gcol), gate.qweight[size_t(r) * 17408 + j]);
        if (par) CHECK_EQ(gu_word(r, ucol), up_p.qweight[size_t(r) * 2048 + (j - 17408)]);
        if (par) CHECK_EQ(gu_scale(r / 8, ucol), up_p.scales[size_t(r / 8) * 2048 + (j - 17408)]);
        checked += 1;
      }
    // down' = [down ; down_p]: qweight row 2176 + r is down_p's row r, scales row 272 + g.
    auto dn_word = [&](uint32_t r, uint32_t n) {
      uint32_t w = 0;
      imm.copy(&w, dn.mem.as<uint32_t>() + size_t(r) * 5120 + n, 4);
      return w;
    };
    auto dn_scale = [&](uint32_t g, uint32_t n) {
      uint16_t s = 0;
      imm.copy(&s, dn.scales->as<uint16_t>() + size_t(g) * 5120 + n, 2);
      return s;
    };
    for (uint32_t n : {0u, 4095u, 5119u}) {
      CHECK_EQ(dn_word(2175, n), down.qweight[size_t(2175) * 5120 + n]);
      CHECK_EQ(dn_word(2176, n), down_p.qweight[n]);
      CHECK_EQ(dn_word(2431, n), down_p.qweight[size_t(255) * 5120 + n]);
      CHECK_EQ(dn_scale(271, n), down.scales[size_t(271) * 5120 + n]);
      CHECK_EQ(dn_scale(272, n), down_p.scales[n]);
      CHECK_EQ(dn_scale(303, n), down_p.scales[size_t(31) * 5120 + n]);
      checked += 6;
    }
  }
  std::printf("load_agnes_test OK: 72 layers (54 GDN + 18 FA), folded gate||up 5120x38912 and "
              "down 19456x5120 on every layer, %zu join readbacks exact, MTP head bound, "
              "max_len 131072 refused; read/token %.3f GB\n",
              checked, m.report.read_per_token / 1e9);
  return 0;
}
