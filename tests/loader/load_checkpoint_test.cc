// End-to-end load of the real checkpoint, with a device round-trip proof:
// one tile of layer 0's qkv||z read back from the device must equal the CPU
// repack of the mmapped source - the canonical bytes on device mean exactly
// what the kernels were tested against.
#include <cmath>
#include <cstdio>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "model/qwen35.h"

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ";
  l0::Context ctx(0);
  loader::LoadedModel m = loader::load(ctx, arg);

  // Counts: 64 layers x linears + lm_head.
  CHECK_EQ(m.layer_small.size(), size_t(64));
  CHECK_EQ(m.linears.size(), size_t(48 * 5 + 16 * 4 + 1));

  // Resident total within 2% of W + declared padding/widening.
  const double gb = 1e9;
  std::printf("resident: %.3f GB (int4 %.3f, scales %.3f, lm_head %.3f)\n",
              m.report.total() / gb, m.report.int4_bytes / gb, m.report.scale_bytes / gb,
              m.report.lm_head_bytes / gb);
  CHECK(m.report.int4_bytes > 12.0e9 && m.report.int4_bytes < 12.4e9);
  CHECK(m.report.lm_head_bytes > 2.5e9 && m.report.lm_head_bytes < 2.6e9);

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

  // The (1+w) bake: one fp32 add, one round-to-nearest-even cast back to bf16.
  std::vector<uint16_t> norms(2 * 5120);
  imm.copy(norms.data(), m.layer_small[0].norms.ptr(), norms.size() * 2);
  const uint16_t* in_ln = reinterpret_cast<const uint16_t*>(
      set.data(set.tensors().at(strip("layers.0.input_layernorm.weight"))));
  CHECK_EQ(norms[0], common::f32_to_bf16(1.0f + common::bf16_to_f32(in_ln[0])));

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
