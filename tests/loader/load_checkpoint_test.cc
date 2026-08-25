// End-to-end load of the real checkpoint, with a device round-trip proof:
// one tile of layer 0's qkv||z read back from the device must equal the CPU
// repack of the mmapped source - the canonical bytes on device mean exactly
// what the kernels were tested against. The small blocks get the same
// treatment: every bake is read back at its loader/small_layout.h offset.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/small_layout.h"
#include "model/qwen35.h"

namespace {
// Little helpers so a readback is compared as the type the kernel will read,
// not as bytes. memcpy, because the block is a byte buffer.
float f32_at(const std::vector<uint8_t>& b, size_t off) {
  float v;
  std::memcpy(&v, b.data() + off, 4);
  return v;
}
uint16_t u16_at(const std::vector<uint8_t>& b, size_t off) {
  uint16_t v;
  std::memcpy(&v, b.data() + off, 2);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";
  l0::Context ctx(0);
  loader::LoadedModel m = loader::load(ctx, arg);

  // Counts: 64 layers x linears + lm_head.
  CHECK_EQ(m.layer_small.size(), size_t(64));
  CHECK_EQ(m.linears.size(), size_t(48 * 5 + 16 * 4 + 1));
  CHECK_EQ(m.max_len, uint32_t(16384));
  // lm_head is the one linear that belongs to no layer (loader::kTopLevel).
  CHECK(m.linears.count({loader::kTopLevel, model::LinearId::LmHead}) == 1);

  // Resident total within 2% of W + declared padding/widening.
  const double gb = 1e9;
  const bool lm_int4 =
      m.linears.at({loader::kTopLevel, model::LinearId::LmHead}).kind == model::WeightKind::Int4;
  std::printf("resident: %.3f GB (int4 %.3f, scales %.3f, lm_head %.3f %s)\n",
              m.report.total() / gb, m.report.int4_bytes / gb, m.report.scale_bytes / gb,
              m.report.lm_head_bytes / gb, lm_int4 ? "int4 g64" : "bf16");
  CHECK(m.report.int4_bytes > 12.0e9 && m.report.int4_bytes < 12.4e9);
  // **The `lm_head` bucket is per checkpoint, so the bar is too** (spec 1.6
  // §5.1). bf16 is 5120 x 248320 x 2 = 2 542 796 800 B; int4 g64 is K*N/2
  // nibbles + K*N/32 scales = 635 699 200 + 39 731 200 = 675 430 400. A single
  // range spanning both would assert nothing; the exact byte count is asserted
  // instead, because both are computable and neither is a measurement.
  CHECK_EQ(m.report.lm_head_bytes, lm_int4 ? size_t(675430400) : size_t(2542796800));
  // The int4 head's shape row is the one the capture will bind: layout 1 (the
  // only repack the loader implements) and S = 1 (what lets the GEMV write
  // straight at `logits`). A checkpoint that moved either would fail at
  // capture with a missing binary; failing here says which row moved.
  {
    const loader::DeviceWeight& lh = m.linears.at({loader::kTopLevel, model::LinearId::LmHead});
    CHECK_EQ(lh.shape.K, uint32_t(5120));
    CHECK_EQ(lh.shape.N, uint32_t(248320));
    CHECK_EQ(lh.shape.S, uint32_t(1));
    if (lm_int4) CHECK_EQ(lh.shape.layout, uint32_t(1));
  }

  // Device round-trip: tile 0 of layer 0 QkvZ vs CPU repack of the source.
  std::string snap = loader::resolve_snapshot(arg);
  loader::SafetensorsSet set(snap);
  auto strip = [](const std::string& s) { return "model.language_model." + s; };
  loader::LinearSrc qkv = loader::LinearSrc::classify(set, strip("layers.0.linear_attn.in_proj_qkv"));
  loader::LinearSrc z = loader::LinearSrc::classify(set, strip("layers.0.linear_attn.in_proj_z"));
  std::vector<common::ColSource> cols = common::cols_concat(
      {{qkv.qweight, qkv.scales, qkv.N}, {z.qweight, z.scales, z.N}});
  const uint32_t K = 5120, NT_check = 4;  // first 4 n-tiles
  std::vector<uint32_t> want(size_t(NT_check) * (K / 64) * 136);
  // repack only the first NT_check tiles: build a truncated col list
  std::vector<common::ColSource> cols4(cols.begin(), cols.begin() + NT_check * 16);
  common::repack_int4_layout1_cols(K, NT_check * 16, cols4, want.data());

  const loader::DeviceWeight& dw = m.linears.at({0u, model::LinearId::QkvZ});
  std::vector<uint32_t> got(want.size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(got.data(), dw.mem.ptr(), got.size() * 4);
  size_t diff = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (want[i] != got[i] && ++diff == 1)
      std::fprintf(stderr, "first mismatch at u32 %zu: want %08X got %08X\n", i, want[i], got[i]);
  CHECK_EQ(diff, size_t(0));

  // Every checkpoint tensor is either loaded or deliberately dropped (qzeros,
  // g_idx, visual, mtp) - nothing is skipped by accident.
  CHECK_EQ(m.report.unconsumed, size_t(0));

  // The mmapped sources of the small tensors. The base of a safetensors data
  // section is 8-aligned by construction on every shard of this checkpoint (the
  // loader's own check_align asserts it before it casts anything), so these
  // casts are safe here.
  auto src16 = [&](const std::string& n) {
    return reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(strip(n))));
  };

  // --- the norms block, fp32 (1 + w) at both halves --------------------------
  // Ruling 2026-08-25: fp32 add, STORED fp32 - no cast back to bf16, so the
  // multiplier carries no rounding the HF reference does not have.
  std::vector<uint8_t> norms(loader::kNormsBlockBytes);
  imm.copy(norms.data(), m.layer_small[0].norms.ptr(), norms.size());
  CHECK_EQ(f32_at(norms, loader::kNormsOffInput),
           1.0f + common::bf16_to_f32(src16("layers.0.input_layernorm.weight")[0]));
  CHECK_EQ(f32_at(norms, loader::kNormsOffPost),
           1.0f + common::bf16_to_f32(src16("layers.0.post_attention_layernorm.weight")[0]));

  // --- layer 0's GDN block: one field per bake, at its offset -----------------
  std::vector<uint8_t> gdn(loader::kGdnBlockBytes);
  imm.copy(gdn.data(), m.layer_small[0].gdn.ptr(), gdn.size());
  CHECK_EQ(f32_at(gdn, loader::kGdnOffConv),
           common::bf16_to_f32(src16("layers.0.linear_attn.conv1d.weight")[0]));
  CHECK_EQ(f32_at(gdn, loader::kGdnOffNegA),
           -std::exp(common::bf16_to_f32(src16("layers.0.linear_attn.A_log")[0])));
  CHECK_EQ(f32_at(gdn, loader::kGdnOffDtBias),
           common::bf16_to_f32(src16("layers.0.linear_attn.dt_bias")[0]));
  // RMSNormGated is the one norm without the +1 - and the one that stays bf16,
  // so this must be the checkpoint's raw word, NOT 1+w and NOT widened.
  CHECK_EQ(u16_at(gdn, loader::kGdnOffGatedNorm), src16("layers.0.linear_attn.norm.weight")[0]);

  // --- an FA layer's k_norm, fp32 (1 + w) at its offset -----------------------
  std::vector<uint8_t> fa(loader::kFaBlockBytes);
  imm.copy(fa.data(), m.layer_small[3].gdn.ptr(), fa.size());
  CHECK_EQ(f32_at(fa, loader::kFaOffKNorm),
           1.0f + common::bf16_to_f32(src16("layers.3.self_attn.k_norm.weight")[0]));

  // --- the final norm, the one that belongs to no layer ----------------------
  std::vector<uint8_t> fnorm(loader::kFinalNormBytes);
  imm.copy(fnorm.data(), m.final_norm.ptr(), fnorm.size());
  CHECK_EQ(f32_at(fnorm, 0), 1.0f + common::bf16_to_f32(src16("norm.weight")[0]));

  // a||b's 32 pad rows really are zero on the device (rows 96..127 of the
  // [128][5120] tiled buffer, i.e. n-tiles 6 and 7).
  std::vector<uint16_t> ab(size_t(128) * 5120);
  imm.copy(ab.data(), m.linears.at({0u, model::LinearId::AB}).mem.ptr(), ab.size() * 2);
  size_t nonzero = 0;
  for (uint32_t n = 96; n < 128; ++n)
    for (uint32_t k = 0; k < 5120; ++k)
      nonzero += ab[((size_t(n / 16) * (5120 / 8) + k / 8) * 8 + k % 8) * 16 + n % 16] != 0;
  CHECK_EQ(nonzero, size_t(0));

  // RoPE: position 0 is exactly (1, 0); position 1's first frequency is
  // (cos 1, sin 1) since inv_freq[0] = theta^0 = 1.
  std::vector<float> rp(128);
  imm.copy(rp.data(), m.rope.ptr(), rp.size() * 4);
  CHECK_EQ(rp[0], 1.0f);
  CHECK_EQ(rp[32], 0.0f);
  CHECK_NEAR(rp[64], std::cos(1.0), 1e-6);
  CHECK_NEAR(rp[96], std::sin(1.0), 1e-6);

  std::printf("load_checkpoint_test OK (%.1f s load)\n", m.report.seconds);
  return 0;
}
