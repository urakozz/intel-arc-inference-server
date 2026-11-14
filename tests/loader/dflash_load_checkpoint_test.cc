// Spec 19b Task 1 (device half): loader::load_drafter on the card beside the loaded target.
//
//   dflash_load_checkpoint_test <target snapshot> <drafter repo or dir> [bf16|int8]
//
// The target (Qwen3.8 int4, int8 head) is loaded at max_len 4096, then the drafter in the given format (both when
// none is given; an AutoRound int4 drafter in int4, its only form). Checks: the allocations add up to loader::dflash_weight_bytes (load_drafter asserts it; printed
// here against the formula); read back from the card - fc, layer 0's q||k||v and the last layer's down, the selector
// projection - byte for byte the host half's device form; the norms, the q / k norms and the conv bases the host
// half's fp32; the successor codebook's first and last rows the shard's; the mask row the target's embedding row
// mask_token_id (widened exactly) unless the checkpoint ships mask_embedding.safetensors. Exit 77 (SKIP) without either
// snapshot. The AutoRound int4 drafter besides: EVERY linear read back from the card and unpacked from its layout-1
// tiles (w = f16 scale x (q - 8)) bitwise the CPU dequant of its safetensors parts - qweight / qzeros / scales read
// straight from the shard, w = scale x (q - (z + 1)) as GPTQ v1 stores the zero, q||k||v and gate||up concatenated.
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
#include "loader/safetensors.h"
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
  } else if (h.kind == model::WeightKind::Int4) {
    const std::vector<uint8_t> b = readback(ctx, imm, w.mem.ptr(), h.int4.size() * 4);
    CHECK(std::memcmp(b.data(), h.int4.data(), b.size()) == 0);
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

// An int4 linear on the card, unpacked: [N][K] fp32 from layout-1 tiles ([N/16][K/64][136] u32: 128 nibble words
// [j][lane], then 16 f16 scales two a word).
std::vector<float> unpack_card(l0::Context& ctx, l0::CmdList& imm, const loader::DeviceWeight& w) {
  const uint32_t K = w.shape.K, N = w.shape.N, G = K / 64;
  const std::vector<uint8_t> b = readback(ctx, imm, w.mem.ptr(), size_t(N / 16) * G * 136 * 4);
  const uint32_t* t = reinterpret_cast<const uint32_t*>(b.data());
  std::vector<float> out(size_t(N) * K);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t* tile = t + (size_t(n / 16) * G + g) * 136;
      const float s = common::f16_to_f32(uint16_t(tile[128 + n % 16 / 2] >> (16 * (n % 2))));
      for (uint32_t k = 0; k < 64; ++k)
        out[size_t(n) * K + g * 64 + k] = s * float(int((tile[(k / 8) * 16 + n % 16] >> (4 * (k % 8))) & 15u) - 8);
    }
  return out;
}

// The CPU dequant of the shard's GPTQ parts (each [K/8][N_i] qweight, [K/64][N_i / 8] qzeros, [K/64][N_i] f16
// scales), their rows concatenated: [N][K] fp32.
std::vector<float> dequant_shard(const loader::SafetensorsSet& set, const std::vector<std::string>& parts, uint32_t K) {
  std::vector<float> out;
  for (const std::string& p : parts) {
    const loader::TensorInfo& qi = set.tensors().at(p + ".qweight");
    const uint32_t N = uint32_t(qi.shape[1]);
    CHECK(qi.shape[0] * 8 == K);
    const auto* q = reinterpret_cast<const uint32_t*>(set.data(qi));
    const auto* z = reinterpret_cast<const uint32_t*>(set.data(set.tensors().at(p + ".qzeros")));
    const auto* s = reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(p + ".scales")));
    const size_t base = out.size();
    out.resize(base + size_t(N) * K);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) {
        const int zero = int((z[size_t(k / 64) * (N / 8) + n / 8] >> (4 * (n % 8))) & 15u) + 1;
        const int code = int((q[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 15u);
        out[base + size_t(n) * K + k] = common::f16_to_f32(s[size_t(k / 64) * N + n]) * float(code - zero);
      }
  }
  return out;
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
  const loader::DFlashCheckpoint ck(drafter_snap);
  const model::DFlashDesc& d = ck.desc();
  std::vector<loader::DraftFormat> formats = {loader::DraftFormat::Int8, loader::DraftFormat::Bf16};
  if (ck.packed()) formats = {loader::DraftFormat::Int4G64};
  if (argc > 3) {
    loader::DraftFormat f;
    CHECK(loader::parse_draft_format(argv[3], f));
    formats = {f};
  }
  l0::Context ctx(0);
  const loader::LoadedModel m = loader::load(ctx, target_snap, 4096, false, loader::LmHeadForm::Int8);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
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
    if (ck.packed()) {
      const loader::SafetensorsSet set(drafter_snap, true);
      size_t tensors = 0, weights = 0;
      auto unpacked = [&](const loader::DeviceWeight& dw, const std::vector<std::string>& parts) {
        const std::vector<float> got = unpack_card(ctx, imm, dw), want = dequant_shard(set, parts, dw.shape.K);
        CHECK_EQ(got.size(), want.size());
        CHECK(std::memcmp(got.data(), want.data(), got.size() * 4) == 0);
        tensors += parts.size();
        weights += got.size();
      };
      unpacked(*w.fc, {"fc"});
      for (uint32_t l = 0; l < d.layers; ++l) {
        const std::string L = "layers." + std::to_string(l) + ".";
        const loader::DFlashLayerWeights& y = w.layers[l];
        unpacked(y.qkv, {L + "self_attn.q_proj", L + "self_attn.k_proj", L + "self_attn.v_proj"});
        unpacked(y.o, {L + "self_attn.o_proj"});
        unpacked(y.gate_up, {L + "mlp.gate_proj", L + "mlp.up_proj"});
        unpacked(y.down, {L + "mlp.down_proj"});
        unpacked(*y.attn_kproj, {L + "attention_conv.kernel_projection"});
        unpacked(*y.mlp_kproj, {L + "mlp_conv.kernel_projection"});
      }
      unpacked(*w.sel_proj, {"candidate_selector.hidden_projection"});
      std::printf("  packed: %zu linears (%zu weights) unpacked from the card == the CPU dequant of the shard, bitwise\n",
                  tensors, weights);
    }
  }
  std::printf("dflash_load_checkpoint_test: PASS\n");
  return 0;
}
