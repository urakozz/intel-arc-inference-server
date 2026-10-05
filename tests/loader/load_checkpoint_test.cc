// End-to-end load of the real checkpoint, with a device round-trip proof:
// one tile of layer 0's qkv||z read back from the device must equal the CPU
// repack of the mmapped source - the canonical bytes on device mean exactly
// what the kernels were tested against. The small blocks get the same
// treatment: every bake is read back at its loader/small_layout.h offset.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
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

void check_mtp(l0::Context& ctx, const std::string& arg);
void check_lm_int8(l0::Context& ctx, const std::string& arg);

int main(int argc, char** argv) {
  const std::string arg = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  l0::Context ctx(0);
  {
  loader::LoadedModel m = loader::load(ctx, arg);
  // Spec 8: without `mtp` the head is skipped, exactly as before.
  CHECK(m.mtp == nullptr);
  CHECK_EQ(m.report.mtp_bytes, size_t(0));
  CHECK_EQ(m.report.mtp_tensors, size_t(0));

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
  CHECK(!dw.scales);  // layout 1 carries scales inline in dw.mem
  std::vector<uint32_t> got(want.size());
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(got.data(), dw.mem.ptr(), got.size() * 4);
  size_t diff = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (want[i] != got[i] && ++diff == 1)
      std::fprintf(stderr, "first mismatch at u32 %zu: want %08X got %08X\n", i, want[i], got[i]);
  CHECK_EQ(diff, size_t(0));

  // Layout 0 owns two real allocations and preserves the checkpoint's
  // [K/8][N] qweight + [K/64][N] scales geometry after each fusion map. Read
  // both first/last rows at boundary columns; expectations are derived from
  // the checkpoint parts directly, not from the repacker's column helpers.
  auto check_layout0 = [&](uint32_t layer, model::LinearId id,
                           const std::function<common::ColSource(uint32_t)>& source,
                           const std::vector<uint32_t>& columns) {
    const loader::DeviceWeight& w = m.linears.at({layer, id});
    CHECK_EQ(w.shape.layout, uint32_t(0));
    CHECK(w.scales);
    CHECK(w.scales->ptr() != w.mem.ptr());
    CHECK_EQ(w.mem.size(), size_t(w.shape.K / 8) * w.shape.N * sizeof(uint32_t));
    CHECK_EQ(w.scales->size(), size_t(w.shape.K / 64) * w.shape.N * sizeof(uint16_t));
    for (uint32_t n : columns) {
      const common::ColSource c = source(n);
      for (uint32_t row : {uint32_t(0), w.shape.K / 8 - 1}) {
        uint32_t qgot = 0;
        imm.copy(&qgot, w.mem.as<uint32_t>() + size_t(row) * w.shape.N + n, sizeof(qgot));
        CHECK_EQ(qgot, c.qweight[size_t(row) * c.n_part + c.n]);
      }
      for (uint32_t g : {uint32_t(0), w.shape.K / 64 - 1}) {
        uint16_t sgot = 0;
        imm.copy(&sgot, w.scales->as<uint16_t>() + size_t(g) * w.shape.N + n, sizeof(sgot));
        CHECK_EQ(sgot, c.scales[size_t(g) * c.n_part + c.n]);
      }
    }
  };

  // Single: GDN out_proj.
  loader::LinearSrc outp =
      loader::LinearSrc::classify(set, strip("layers.0.linear_attn.out_proj"));
  check_layout0(
      0, model::LinearId::OutProj,
      [&](uint32_t n) { return common::ColSource{outp.qweight, outp.scales, n, outp.N}; },
      {0, 15, 16, outp.N - 1});

  // Concat: FA q || k || v, including both part boundaries.
  loader::LinearSrc q = loader::LinearSrc::classify(set, strip("layers.3.self_attn.q_proj"));
  loader::LinearSrc k = loader::LinearSrc::classify(set, strip("layers.3.self_attn.k_proj"));
  loader::LinearSrc v = loader::LinearSrc::classify(set, strip("layers.3.self_attn.v_proj"));
  check_layout0(
      3, model::LinearId::Qkv,
      [&](uint32_t n) {
        if (n < q.N) return common::ColSource{q.qweight, q.scales, n, q.N};
        n -= q.N;
        if (n < k.N) return common::ColSource{k.qweight, k.scales, n, k.N};
        n -= k.N;
        return common::ColSource{v.qweight, v.scales, n, v.N};
      },
      {0, q.N - 1, q.N, q.N + k.N - 1, q.N + k.N, q.N + k.N + v.N - 1});

  // Interleave16: gate/up in alternating 16-column blocks.
  loader::LinearSrc gate = loader::LinearSrc::classify(set, strip("layers.0.mlp.gate_proj"));
  loader::LinearSrc up = loader::LinearSrc::classify(set, strip("layers.0.mlp.up_proj"));
  check_layout0(
      0, model::LinearId::GateUp,
      [&](uint32_t n) {
        const uint32_t base = (n / 32) * 16;
        const bool from_up = n % 32 >= 16;
        const uint32_t src_n = base + n % 16;
        return common::ColSource{from_up ? up.qweight : gate.qweight,
                                 from_up ? up.scales : gate.scales, src_n,
                                 from_up ? up.N : gate.N};
      },
      {0, 15, 16, 31, 32, 47, 48, 63, gate.N + up.N - 1});

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
  const loader::SmallLayout SL = m.desc->small_layout();   // Qwen3.8: loader::kQwen38Small
  std::vector<uint8_t> norms(SL.norms_block_bytes);
  imm.copy(norms.data(), m.layer_small[0].norms.ptr(), norms.size());
  CHECK_EQ(f32_at(norms, SL.norms_off_input),
           1.0f + common::bf16_to_f32(src16("layers.0.input_layernorm.weight")[0]));
  CHECK_EQ(f32_at(norms, SL.norms_off_post),
           1.0f + common::bf16_to_f32(src16("layers.0.post_attention_layernorm.weight")[0]));

  // --- layer 0's GDN block: one field per bake, at its offset -----------------
  std::vector<uint8_t> gdn(SL.gdn_block_bytes);
  imm.copy(gdn.data(), m.layer_small[0].gdn.ptr(), gdn.size());
  CHECK_EQ(f32_at(gdn, SL.gdn_off_conv),
           common::bf16_to_f32(src16("layers.0.linear_attn.conv1d.weight")[0]));
  CHECK_EQ(f32_at(gdn, SL.gdn_off_nega),
           -std::exp(common::bf16_to_f32(src16("layers.0.linear_attn.A_log")[0])));
  CHECK_EQ(f32_at(gdn, SL.gdn_off_dtbias),
           common::bf16_to_f32(src16("layers.0.linear_attn.dt_bias")[0]));
  // RMSNormGated is the one norm without the +1 - and the one that stays bf16,
  // so this must be the checkpoint's raw word, NOT 1+w and NOT widened.
  CHECK_EQ(u16_at(gdn, SL.gdn_off_gated_norm), src16("layers.0.linear_attn.norm.weight")[0]);

  // --- an FA layer's k_norm, fp32 (1 + w) at its offset -----------------------
  std::vector<uint8_t> fa(loader::kFaBlockBytes);
  imm.copy(fa.data(), m.layer_small[3].gdn.ptr(), fa.size());
  CHECK_EQ(f32_at(fa, loader::kFaOffKNorm),
           1.0f + common::bf16_to_f32(src16("layers.3.self_attn.k_norm.weight")[0]));

  // --- the final norm, the one that belongs to no layer ----------------------
  std::vector<uint8_t> fnorm(SL.final_norm_bytes);
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

  // Spec 6 §10: the trained context (config.json text_config.max_position_embeddings)
  // and the RoPE table's own bytes, which the memory planner nets out of the total.
  CHECK_EQ(m.trained_max_len, uint32_t(262144));
  CHECK_EQ(m.report.rope_bytes, size_t(16384) * 256);
  CHECK_EQ(m.report.rope_bytes, model::Qwen35::rope_table_bytes(16384));
  // set_max_len (--max-len auto's re-tabling): the table at 4096 is the first 4096 rows
  // of the table at 16384, bit for bit, and the report moves with it.
  {
    std::vector<float> row_before(64), row_after(64);
    imm.copy(row_before.data(), static_cast<const float*>(m.rope.ptr()) + size_t(4095) * 64, 256);
    const size_t total_before = m.report.total();
    loader::set_max_len(ctx, m, 4096);
    CHECK_EQ(m.max_len, uint32_t(4096));
    CHECK_EQ(m.rope.size(), size_t(4096) * 256);
    CHECK_EQ(m.report.rope_bytes, size_t(4096) * 256);
    CHECK_EQ(m.report.total(), total_before - size_t(12288) * 256);
    imm.copy(row_after.data(), static_cast<const float*>(m.rope.ptr()) + size_t(4095) * 64, 256);
    CHECK(row_before == row_after);
    bool refused = false;
    try {
      loader::set_max_len(ctx, m, 262144 + 256);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    CHECK(refused);
    CHECK_EQ(m.max_len, uint32_t(4096));   // a refused call leaves the model as it was
  }

  std::printf("main model OK (%.1f s load)\n", m.report.seconds);
  }
  check_mtp(ctx, arg);
  check_lm_int8(ctx, arg);
  std::printf("load_checkpoint_test OK\n");
  return 0;
}

// Spec 8 §3.1 (plan 8b Task 1): the MTP head, loaded on request. 15 bf16 tensors,
// 0.849 GB in the checkpoint; every linear read back against a host repack of the
// mmapped source at tiles that cross the fusion boundaries.
void check_mtp(l0::Context& ctx, const std::string& arg) {
  loader::LoadedModel m = loader::load(ctx, arg, 16384, /*mtp=*/true);
  CHECK(m.mtp != nullptr);
  CHECK_EQ(m.report.mtp_tensors, loader::kMtpTensors);
  CHECK_EQ(m.report.mtp_checkpoint_bytes, size_t(849398784));
  // device: the linears' bf16 bytes plus the five norms widened to fp32 and the FA block.
  CHECK_EQ(m.report.mtp_bytes, size_t(424673280) * 2 + 5 * 5120 * 4 + loader::kFaBlockBytes);
  CHECK_EQ(m.report.unconsumed, size_t(0));
  const loader::MtpHead& h = *m.mtp;
  CHECK(h.gate_up && h.down && !h.moe);   // spec 15e: a dense head's FFN
  struct Row { const loader::DeviceWeight* w; uint32_t K, N; };
  for (const Row& r : {Row{&h.fc, 10240, 5120}, Row{&h.qkv, 5120, 14336}, Row{&h.o, 6144, 5120},
                       Row{h.gate_up.get(), 5120, 34816}, Row{h.down.get(), 17408, 5120}}) {
    CHECK(r.w->kind == model::WeightKind::Bf16);
    CHECK_EQ(r.w->shape.K, r.K);
    CHECK_EQ(r.w->shape.N, r.N);
    CHECK_EQ(r.w->shape.S, uint32_t(1));
    CHECK_EQ(r.w->mem.size(), size_t(r.K) * r.N * 2);
  }
  std::string snap = loader::resolve_snapshot(arg);
  loader::SafetensorsSet set(snap);
  auto src = [&](const std::string& n) {
    return reinterpret_cast<const uint16_t*>(set.data(set.tensors().at(n)));
  };
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  // Element (n, k) of a tiled weight (common::repack_bf16_tiled's index).
  auto at = [&](const loader::DeviceWeight& w, uint32_t n, uint32_t k) {
    const size_t K8 = w.shape.K / 8;
    uint16_t v = 0;
    imm.copy(&v, w.mem.as<uint16_t>() + ((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16,
             2);
    return v;
  };
  const std::string L = "mtp.layers.0.";
  for (uint32_t n : {0u, 17u, 5119u})
    for (uint32_t k : {0u, 5119u, 5120u, 10239u})
      CHECK_EQ(at(h.fc, n, k), src("mtp.fc.weight")[size_t(n) * 10240 + k]);
  // q || k || v: both part boundaries.
  const uint16_t* q = src(L + "self_attn.q_proj.weight");
  const uint16_t* kp = src(L + "self_attn.k_proj.weight");
  const uint16_t* vp = src(L + "self_attn.v_proj.weight");
  for (uint32_t k : {0u, 4095u, 5119u}) {
    CHECK_EQ(at(h.qkv, 12287, k), q[size_t(12287) * 5120 + k]);
    CHECK_EQ(at(h.qkv, 12288, k), kp[k]);
    CHECK_EQ(at(h.qkv, 13311, k), kp[size_t(1023) * 5120 + k]);
    CHECK_EQ(at(h.qkv, 13312, k), vp[k]);
    CHECK_EQ(at(h.qkv, 14335, k), vp[size_t(1023) * 5120 + k]);
  }
  // gate/up interleaved in 16-column blocks (fused column c: block c/32, gate if c%32 < 16).
  const uint16_t* g = src(L + "mlp.gate_proj.weight");
  const uint16_t* u = src(L + "mlp.up_proj.weight");
  for (uint32_t c : {0u, 15u, 16u, 31u, 32u, 34815u})
    for (uint32_t k : {0u, 5119u}) {
      const uint32_t row = (c / 32) * 16 + c % 16;
      CHECK_EQ(at(*h.gate_up, c, k), (c % 32 < 16 ? g : u)[size_t(row) * 5120 + k]);
    }
  CHECK_EQ(at(h.o, 5119, 6143), src(L + "self_attn.o_proj.weight")[size_t(5119) * 6144 + 6143]);
  CHECK_EQ(at(*h.down, 1, 17407), src(L + "mlp.down_proj.weight")[size_t(1) * 17408 + 17407]);
  // The norms: fp32 (1 + w), the main model's bake.
  std::vector<uint8_t> norms(loader::mtp_norms_bytes(*m.desc)), fa(loader::kFaBlockBytes);
  imm.copy(norms.data(), h.norms.ptr(), norms.size());
  imm.copy(fa.data(), h.fa.ptr(), fa.size());
  const std::pair<const char*, size_t> nrm[] = {
      {"mtp.pre_fc_norm_embedding.weight", loader::mtp_norm_off(*m.desc, loader::kMtpNormPreE)},
      {"mtp.pre_fc_norm_hidden.weight", loader::mtp_norm_off(*m.desc, loader::kMtpNormPreH)},
      {"mtp.layers.0.input_layernorm.weight", loader::mtp_norm_off(*m.desc, loader::kMtpNormInput)},
      {"mtp.layers.0.post_attention_layernorm.weight", loader::mtp_norm_off(*m.desc, loader::kMtpNormPost)},
      {"mtp.norm.weight", loader::mtp_norm_off(*m.desc, loader::kMtpNormFinal)}};
  for (const auto& [name, off] : nrm)
    for (uint32_t i : {0u, 5119u})
      CHECK_EQ(f32_at(norms, off + size_t(i) * 4), 1.0f + common::bf16_to_f32(src(name)[i]));
  CHECK_EQ(f32_at(fa, loader::kFaOffQNorm + 4),
           1.0f + common::bf16_to_f32(src(L + "self_attn.q_norm.weight")[1]));
  CHECK_EQ(f32_at(fa, loader::kFaOffKNorm + 1020),
           1.0f + common::bf16_to_f32(src(L + "self_attn.k_norm.weight")[255]));
  std::printf("mtp head OK: %zu tensors, %.3f GB checkpoint, %.3f GB device\n",
              m.report.mtp_tensors, m.report.mtp_checkpoint_bytes / 1e9, m.report.mtp_bytes / 1e9);
}

// Spec 9 §3 (plan 9b Task 1): `--lm-head int8` - the bf16 head quantised on the host at
// load. Bytes derived: 5120 x 248320 int8 = 1 271 398 400, plus 248320 fp32 scales =
// 993 280. Rows read back against a host quantisation of the mmapped source, at tiles
// that cross the first and last n-tile and a K-block edge; the quantisation time is
// printed and held to spec 9 §3's 10 s.
void check_lm_int8(l0::Context& ctx, const std::string& arg) {
  loader::LoadedModel m =
      loader::load(ctx, arg, 16384, /*mtp=*/false, loader::LmHeadForm::Int8);
  const loader::DeviceWeight& lh = m.linears.at({loader::kTopLevel, model::LinearId::LmHead});
  CHECK(lh.kind == model::WeightKind::Int8);
  CHECK_EQ(lh.shape.K, uint32_t(5120));
  CHECK_EQ(lh.shape.N, uint32_t(248320));
  CHECK_EQ(lh.shape.S, uint32_t(1));
  CHECK_EQ(m.report.lm_head_bytes, size_t(1271398400) + size_t(993280));
  CHECK_EQ(lh.mem.size(), size_t(1271398400));
  CHECK(lh.scales);
  CHECK_EQ(lh.scales->size(), size_t(993280));
  CHECK_EQ(m.report.unconsumed, size_t(0));
  std::printf("lm_head int8: %.3f GB, host quantisation %.3f s\n", m.report.lm_head_bytes / 1e9,
              m.report.lm_head_quant_seconds);
  CHECK(m.report.lm_head_quant_seconds > 0.0 && m.report.lm_head_quant_seconds < 10.0);

  const std::string snap = loader::resolve_snapshot(arg);
  loader::SafetensorsSet set(snap);
  const loader::LinearSrc src = loader::LinearSrc::classify(set, "lm_head");
  const uint32_t K = 5120, N = 248320;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  size_t diff = 0;
  for (uint32_t n : {0u, 1u, 15u, 16u, 12345u, N - 17, N - 1}) {
    std::vector<int8_t> want(K);
    const float s = loader::quantise_row_int8(src.weight + size_t(n) * K, K, want.data());
    float sgot = 0;
    imm.copy(&sgot, lh.scales->as<float>() + n, 4);
    CHECK_EQ(sgot, s);
    for (uint32_t k : {0u, 15u, 16u, 2047u, K - 1}) {
      int8_t got = 0;
      imm.copy(&got, lh.mem.as<int8_t>() + loader::int8_tiled_index(K, k, n), 1);
      diff += got != want[k];
    }
  }
  CHECK_EQ(diff, size_t(0));
}
