// Spec 19b Task 1 (device half): loader::load_drafter on the card beside the loaded target.
//
//   dflash_load_checkpoint_test <target snapshot> <drafter repo or dir> [bf16|int8]
//
// The target (Qwen3.8 int4, int8 head) is loaded at max_len 4096, then the drafter in the given format (both when
// none is given). Checks: the allocations add up to loader::dflash_weight_bytes (load_drafter asserts it; printed
// here against the formula); read back from the card - fc, layer 0's q||k||v and the last layer's down, the selector
// projection - byte for byte the host half's device form; the norms, the q / k norms and the conv bases the host
// half's fp32; the successor codebook's first and last rows the shard's; the mask row the target's embedding row
// mask_token_id (widened exactly) unless the checkpoint ships mask_embedding.safetensors. Exit 77 (SKIP) without either
// snapshot.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/dflash_loader.h"
#include "loader/dflash_repack.h"
#include "loader/loader.h"
#include "loader/snapshot.h"

namespace {

std::vector<uint8_t> readback(l0::Context& ctx, l0::CmdList& imm, const void* dev, size_t bytes) {
  l0::Mem stage(ctx, l0::MemKind::Host, bytes);
  imm.copy(stage.ptr(), dev, bytes);
  const uint8_t* p = stage.as<uint8_t>();
  return std::vector<uint8_t>(p, p + bytes);
}

void same_linear(l0::Context& ctx, l0::CmdList& imm, const loader::DeviceWeight& w,
                 const loader::DFlashHostLinear& h, const char* what) {
  CHECK(w.shape.K == h.K && w.shape.N == h.N && w.kind == h.kind);
  if (h.kind == model::WeightKind::Int8) {
    CHECK(readback(ctx, imm, w.mem.ptr(), h.i8.size()) ==
          std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(h.i8.data()),
                               reinterpret_cast<const uint8_t*>(h.i8.data()) + h.i8.size()));
    const std::vector<uint8_t> s = readback(ctx, imm, w.scales->ptr(), h.scales.size() * 4);
    CHECK(std::memcmp(s.data(), h.scales.data(), s.size()) == 0);
  } else {
    const std::vector<uint8_t> b = readback(ctx, imm, w.mem.ptr(), h.bf16.size() * 2);
    CHECK(std::memcmp(b.data(), h.bf16.data(), b.size()) == 0);
  }
  std::printf("  %-28s read back == the host form (%zu B)\n", what, h.bytes());
}

void same_f32(l0::Context& ctx, l0::CmdList& imm, const l0::Mem& m, const std::vector<float>& v, const char* what) {
  CHECK(m.size() == v.size() * 4);
  const std::vector<uint8_t> b = readback(ctx, imm, m.ptr(), m.size());
  CHECK(std::memcmp(b.data(), v.data(), b.size()) == 0);
  std::printf("  %-28s read back == the host fp32 (%zu floats)\n", what, v.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: dflash_load_checkpoint_test <target snapshot> <drafter> [bf16|int8]\n");
    return 2;
  }
  std::string target_snap, drafter_snap;
  try {
    target_snap = loader::resolve_snapshot(argv[1]);
    drafter_snap = loader::resolve_snapshot(argv[2]);
  } catch (const std::exception& e) {
    std::printf("SKIP: %s\n", e.what());
    return 77;
  }
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int8, loader::DraftFormat::Bf16};
  if (argc > 3) {
    loader::DraftFormat f;
    CHECK(loader::parse_draft_format(argv[3], f));
    formats = {f};
  }
  l0::Context ctx(0);
  const loader::LoadedModel m = loader::load(ctx, target_snap, 4096, false, loader::LmHeadForm::Int8);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const loader::DFlashCheckpoint ck(drafter_snap);
  const model::DFlashDesc& d = ck.desc();
  for (loader::DraftFormat f : formats) {
    const loader::DFlashWeights w = loader::load_drafter(ctx, drafter_snap, m, f);
    const size_t want = loader::dflash_weight_bytes(d, ck.target_hidden(), f).total();
    std::printf("%s: %zu B on the card, the formula %zu B\n", loader::draft_format_name(f), w.bytes, want);
    CHECK_EQ(w.bytes, want);
    same_linear(ctx, imm, *w.fc, ck.fc(f), "fc");
    same_linear(ctx, imm, w.layers[0].qkv, ck.layer_linear(0, loader::DFlashLinear::Qkv, f), "layer 0 q||k||v");
    same_linear(ctx, imm, w.layers.back().down, ck.layer_linear(d.layers - 1, loader::DFlashLinear::Down, f),
                "last layer down");
    same_f32(ctx, imm, *w.hidden_norm, ck.hidden_norm(), "hidden_norm");
    same_f32(ctx, imm, *w.final_norm, ck.final_norm(), "norm");
    same_f32(ctx, imm, w.layers[2].norms, ck.layer_norms(2), "layer 2 norms");
    same_f32(ctx, imm, *w.qk_norm, ck.qk_norms(), "q / k norms");
    if (d.v2()) {
      same_linear(ctx, imm, *w.layers[1].attn_kproj, ck.layer_linear(1, loader::DFlashLinear::AttnKproj, f),
                  "layer 1 attention_conv kproj");
      same_linear(ctx, imm, *w.sel_proj, ck.selector_projection(f), "selector hidden_projection");
      same_f32(ctx, imm, *w.conv_base, ck.conv_bases(), "conv bases");
      const size_t row = size_t(d.sel_rank) * 2, last = size_t(d.vocab - 1) * row;
      const std::vector<uint8_t> first = readback(ctx, imm, w.succ->ptr(), row);
      const std::vector<uint8_t> end =
          readback(ctx, imm, static_cast<const uint8_t*>(w.succ->ptr()) + last, row);
      const uint8_t* src = reinterpret_cast<const uint8_t*>(ck.codebook(true));
      CHECK(std::memcmp(first.data(), src, row) == 0 && std::memcmp(end.data(), src + last, row) == 0);
      std::printf("  successor codebook rows 0 and %u == the shard's\n", d.vocab - 1);
    }
    std::vector<float> mask = ck.mask_embedding();
    if (mask.empty()) {
      const std::vector<uint8_t> row =
          readback(ctx, imm, static_cast<const uint8_t*>(m.embed.ptr()) + size_t(d.mask_id) * d.hidden * 2,
                   size_t(d.hidden) * 2);
      for (uint32_t i = 0; i < d.hidden; ++i) {
        uint16_t h;
        std::memcpy(&h, row.data() + 2 * i, 2);
        mask.push_back(common::bf16_to_f32(h));
      }
    }
    CHECK(w.mask_from_checkpoint == ck.ships_mask_embedding());
    same_f32(ctx, imm, *w.mask_row, mask, w.mask_from_checkpoint ? "mask row (checkpoint)" : "mask row (embedding)");
  }
  std::printf("dflash_load_checkpoint_test: PASS\n");
  return 0;
}
