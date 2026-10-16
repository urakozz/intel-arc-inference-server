// Spec 23f Task 6 (host): Muse-Glimmer-30B's DFlash drafter (meta-models/Muse-Glimmer-30B-assistant) as spec 19b's
// pieces read it - the descriptor, the loader's host half, the int4 g64 RTN at load and the drafter's bytes in the plan.
// No device.
//
//   1. The published config (tests/model/muse_glimmer/assistant_config.json, @ e8192f3a) -> the Muse-form descriptor
//      (spec 23 §8.1: 5 sliding layers of 2048 INCLUSIVE, 32 / 8 x 128, FF 19968, block 16, mask 201818, taps 1 / 13 /
//      25 / 37 / 49, eps 1e-5, theta 5e5, no vocab_size - the target's), held to Muse-Glimmer-30B's shape and to the
//      kernels' (model::check_muse_assistant). Refused by name: a hidden or a vocabulary unlike the target's, a tap
//      layer >= the target's layers, a block other than 16, a window other than 2048, a full-attention layer, a
//      quantization_config, a shipped mask_embedding file.
//   2. A tiny synthetic assistant in the published names (encoder.fc / encoder.output_norm_enc / layers.N / norm, one
//      model.safetensors): the names both ways and every linear's device form in the three formats - bf16 tiled,
//      int8 per-row RTN (spec 9's quantiser), int4 g64 RTN (loader/rtn.h rtn_int4_g64, then common::
//      repack_int4_layout1) - and the int4 tiles' dequant = (q - 8) x f16 scale of the RTN; rtn_int4_g64 = the fixture
//      tools/oracle/fixtures/rtn_int4_g64.json the oracle's int4g64 arm is held to (both sides of P0's arm).
//   3. The bytes: loader::dflash_weight_bytes in each format (int4 g64: 1.358 GB - spec 23 §8.2), the drafter's
//      buffers at 16 rows (the 2112-slot ring, the 16-row readback, the verify's ids), and the plan - `--max-len auto`
//      still 131072 on one card with the int4 drafter, the verify scratch at 16 rows and the prefill tap staging.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "loader/dflash_fixture.h"
#include "loader/dflash_layout.h"
#include "loader/dflash_repack.h"
#include "loader/lm_head_int8.h"
#include "loader/rtn.h"
#include "model/dflash.h"
#include "model/muse_glimmer.h"
#include "runtime/dflash/dflash_control.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/muse/muse_sizes.h"

namespace fs = std::filesystem;
namespace fx = dflash_fixture;

namespace {

constexpr size_t kCard = 32530000000ull;   // memory_line()'s device total (spec 6 §8.4)

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) {
      std::printf("  refused: %s\n", e.what());
      return true;
    }
    std::fprintf(stderr, "  threw, but without '%s': %s\n", needle.c_str(), e.what());
    return false;
  }
  std::fprintf(stderr, "  did not throw (wanted '%s')\n", needle.c_str());
  return false;
}

std::string slurp(const std::string& p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos);
  return s.replace(at, from.size(), to);
}
model::DFlashDesc parse(const std::string& text) { return model::dflash_desc_from_config(common::json::parse(text)); }

fs::path tmpdir(const std::string& tag) {
  fs::path p = fs::temp_directory_path() / ("muse_assistant_test_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

// The tiny assistant: hidden 64, 2 layers, 4 / 2 heads x 16, FF 128, taps {1, 3} of a 4-layer hidden-64 target.
const char* kTinyConfig =
    "{\"architectures\":[\"MuseGlimmerAssistantModel\"],\"attention_dropout\":0,\"block_size\":16,"
    "\"bos_token_id\":200000,\"dtype\":\"bfloat16\",\"eos_token_id\":200001,\"head_dim\":16,\"hidden_act\":\"silu\","
    "\"hidden_size\":64,\"intermediate_size\":128,\"layer_types\":[\"sliding_attention\",\"sliding_attention\"],"
    "\"mask_token_id\":500,\"max_position_embeddings\":131072,\"model_type\":\"muse_glimmer_assistant\","
    "\"num_attention_heads\":4,\"num_hidden_layers\":2,\"num_key_value_heads\":2,\"pad_token_id\":200018,"
    "\"rms_norm_eps\":1e-05,\"rope_parameters\":{\"rope_theta\":500000.0,\"rope_type\":\"default\"},"
    "\"sliding_window\":2048,\"target_layer_ids\":[1,3]}";

fx::Tensors tiny_tensors(uint64_t seed) {
  fx::Rng r(seed);
  fx::Tensors ts;
  const uint64_t H = 64, qn = 64, kvn = 32, I = 128;
  ts["encoder.fc.weight"] = fx::bf16_tensor(r, {H, 2 * H}, 0.1f);
  ts["encoder.output_norm_enc.weight"] = fx::bf16_tensor(r, {H}, 0.2f, 1.0f);
  ts["norm.weight"] = fx::bf16_tensor(r, {H}, 0.2f, 1.0f);
  for (int l = 0; l < 2; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    ts[L + "input_layernorm.weight"] = fx::bf16_tensor(r, {H}, 0.2f, 1.0f);
    ts[L + "post_attention_layernorm.weight"] = fx::bf16_tensor(r, {H}, 0.2f, 1.0f);
    ts[L + "self_attn.q_proj.weight"] = fx::bf16_tensor(r, {qn, H}, 0.15f);
    ts[L + "self_attn.k_proj.weight"] = fx::bf16_tensor(r, {kvn, H}, 0.15f);
    ts[L + "self_attn.v_proj.weight"] = fx::bf16_tensor(r, {kvn, H}, 0.15f);
    ts[L + "self_attn.o_proj.weight"] = fx::bf16_tensor(r, {H, qn}, 0.15f);
    ts[L + "self_attn.q_norm.weight"] = fx::bf16_tensor(r, {16}, 0.2f, 1.0f);
    ts[L + "self_attn.k_norm.weight"] = fx::bf16_tensor(r, {16}, 0.2f, 1.0f);
    ts[L + "mlp.gate_proj.weight"] = fx::bf16_tensor(r, {I, H}, 0.15f);
    ts[L + "mlp.up_proj.weight"] = fx::bf16_tensor(r, {I, H}, 0.15f);
    ts[L + "mlp.down_proj.weight"] = fx::bf16_tensor(r, {H, I}, 0.15f);
  }
  return ts;
}

std::vector<uint16_t> rows_of(const fx::Tensors& ts, const std::vector<std::string>& names) {
  std::vector<uint16_t> v;
  for (const std::string& n : names) {
    const fx::Tensor& t = ts.at(n);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
    v.insert(v.end(), p, p + t.bytes.size() / 2);
  }
  return v;
}

// One fixture number list ("key": [a, b, ...]) from the JSON text.
std::vector<uint32_t> fixture_list(const common::json::Value& v, const char* key) {
  std::vector<uint32_t> out;
  for (const common::json::Value& x : v.find(key)->arr()) out.push_back(uint32_t(x.num()));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: muse_assistant_test <assistant_config.json> <rtn_int4_g64.json>\n");
    return 2;
  }
  const std::string real = slurp(argv[1]);
  const model::MuseGlimmerDesc target = model::muse_glimmer();
  const model::DFlashTargetShape tshape{"Muse-Glimmer-30B", target.layers, target.hidden, target.vocab};

  // --- 1. the published config ------------------------------------------------------------------------------
  {
    const model::DFlashDesc d = parse(real);
    CHECK(d.muse() && d.window_inclusive && d.norm_two_roundings && d.version == 1);
    CHECK(d.architecture == "MuseGlimmerAssistantModel" && model::is_dflash_architecture(d.architecture));
    CHECK(d.layers == 5 && d.hidden == 6656 && d.q_heads == 32 && d.kv_heads == 8 && d.head_dim == 128);
    CHECK(d.inter == 19968 && d.block == 16 && d.max_k() == 15 && d.mask_id == 201818 && d.vocab == 0);
    CHECK((d.taps == std::vector<uint32_t>{1, 13, 25, 37, 49}));
    CHECK(d.window == std::vector<uint32_t>(5, 2048) && d.causal == std::vector<bool>(5, false));
    CHECK(d.eps == 1e-5f && d.theta == 500000.0 && d.out_mult == 1.0f && d.softcap == 0.0f && d.in_scale == 1.0f);
    CHECK(d.context_reach(0) == 2048);   // row 0 reads p - 2048 .. p - 1 (vLLM's form would be 2047)
    model::check_dflash_target(d, tshape);
    model::check_muse_assistant(d);
    CHECK(throws_with([&] { model::check_dflash_target(d, {"x", 52, 5120, 202048}); }, "hidden_size"));
    CHECK(throws_with([&] { model::check_dflash_target(d, {"x", 40, 6656, 202048}); }, "target_layer_ids holds 49"));
    CHECK(throws_with([&] { model::check_dflash_target(parse(with(real, "\"block_size\": 16", "\"vocab_size\": 1000, \"block_size\": 16")),
                                                       tshape); }, "vocab_size 1000"));
    CHECK(throws_with([&] { model::check_muse_assistant(parse(with(real, "\"block_size\": 16", "\"block_size\": 8"))); },
                      "block_size 8"));
    CHECK(throws_with([&] { model::check_muse_assistant(parse(with(real, "\"sliding_window\": 2048", "\"sliding_window\": 1024"))); },
                      "sliding_window 1024"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"sliding_attention\",\n    \"sliding_attention\"\n  ]",
                                             "\"sliding_attention\",\n    \"full_attention\"\n  ]")); },
                      "full-attention"));
    CHECK(throws_with([&] { (void)parse(with(real, "\"block_size\": 16", "\"quantization_config\": {}, \"block_size\": 16")); },
                      "quantization_config"));
    std::printf("1. the published assistant config: the Muse form, block 16 (15 drafts), window 2048 inclusive, held "
                "to Muse-Glimmer-30B and the kernels\n");
  }

  // --- 2. the tiny checkpoint: names, the three formats -------------------------------------------------------
  {
    const fs::path dir = tmpdir("tiny");
    const fx::Tensors ts = tiny_tensors(7);
    fx::write_safetensors(dir / "model.safetensors", ts);
    std::ofstream(dir / "config.json") << kTinyConfig;
    const loader::DFlashCheckpoint ck(dir.string());
    CHECK(ck.desc().muse() && ck.fc_k() == 128 && ck.target_hidden() == 64 && ck.tensors() == ts.size());
    ck.check_target(model::DFlashTargetShape{"tiny", 4, 64, 512});
    CHECK(throws_with([&] { ck.check_target(model::DFlashTargetShape{"tiny", 3, 64, 512}); }, "target_layer_ids holds 3"));
    const std::vector<uint16_t> qkv = rows_of(ts, {"layers.1.self_attn.q_proj.weight", "layers.1.self_attn.k_proj.weight",
                                                  "layers.1.self_attn.v_proj.weight"});
    const uint32_t K = 64, N = 128;
    {
      const loader::DFlashHostLinear h = ck.layer_linear(1, loader::DFlashLinear::Qkv, loader::DraftFormat::Bf16);
      std::vector<uint16_t> want(qkv.size());
      common::repack_bf16_tiled(qkv.data(), K, N, want.data());
      CHECK(h.kind == model::WeightKind::Bf16 && h.K == K && h.N == N && h.bf16 == want);
    }
    {
      const loader::DFlashHostLinear h = ck.layer_linear(1, loader::DFlashLinear::Qkv, loader::DraftFormat::Int8);
      std::vector<int8_t> q(qkv.size());
      std::vector<float> s(N);
      loader::quantise_int8_tiled(qkv.data(), K, N, q.data(), s.data(), 1);
      CHECK(h.kind == model::WeightKind::Int8 && h.i8 == q && h.scales == s);
    }
    {
      const loader::DFlashHostLinear h = ck.layer_linear(1, loader::DFlashLinear::Qkv, loader::DraftFormat::Int4G64);
      std::vector<uint32_t> qw(size_t(K / 8) * N), tiles(size_t(N / 16) * (K / 64) * 136);
      std::vector<uint16_t> sc(size_t(K / 64) * N);
      loader::rtn_int4_g64(qkv.data(), K, N, qw.data(), sc.data());
      common::repack_int4_layout1(qw.data(), sc.data(), K, N, tiles.data());
      CHECK(h.kind == model::WeightKind::Int4 && h.int4 == tiles && h.shape().layout == 1 && h.shape().S == 1);
      CHECK_EQ(h.bytes(), loader::dflash_linear_bytes(K, N, loader::DraftFormat::Int4G64));
      // the tiles hold (q - 8) x scale of the RTN: column n's k-th weight read back through the tile layout
      double worst = 0;
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k) {
          const uint32_t nt = n / 16, l = n % 16, g = k / 64, j = (k % 64) / 8;
          const uint32_t* tile = h.int4.data() + (size_t(nt) * (K / 64) + g) * 136;
          const uint32_t q = (tile[j * 16 + l] >> (4 * (k % 8))) & 0xFu;
          const uint32_t sw = tile[128 + l / 2];
          const float s16 = common::f16_to_f32(uint16_t(l % 2 ? sw >> 16 : sw & 0xFFFFu));
          const float wq = (float(q) - 8.0f) * s16, w = common::bf16_to_f32(qkv[size_t(n) * K + k]);
          worst = std::max(worst, double(std::fabs(wq - w)) / std::max(1e-9, double(s16)));
        }
      CHECK(worst <= 0.51);   // RTN: within half a step of the stored scale (+ the f16 rounding of the scale)
    }
    // rtn_int4_g64 == the oracle's fixture (its int4g64 arm is held to the same file)
    {
      const common::json::Value fxj = common::json::parse(slurp(argv[2]));
      const uint32_t FN = uint32_t(fxj.find("N")->num()), FK = uint32_t(fxj.find("K")->num());
      const std::vector<uint32_t> w32 = fixture_list(fxj, "w_bf16"), q32 = fixture_list(fxj, "qweight"),
                                  s32 = fixture_list(fxj, "scales_f16");
      std::vector<uint16_t> w(w32.begin(), w32.end());
      std::vector<uint32_t> qw(size_t(FK / 8) * FN);
      std::vector<uint16_t> sc(size_t(FK / 64) * FN);
      loader::rtn_int4_g64(w.data(), FK, FN, qw.data(), sc.data());
      CHECK(qw == q32 && std::vector<uint32_t>(sc.begin(), sc.end()) == s32);
    }
    // a shipped mask embedding is refused rather than guessed at
    std::ofstream(dir / "mask_embedding.safetensors") << "x";
    CHECK(throws_with([&] { (void)loader::DFlashCheckpoint(dir.string()); }, "mask_embedding.safetensors"));
    fs::remove_all(dir);
    std::printf("2. the tiny assistant's names; q||k||v in bf16 / int8 / int4 g64 = the layout functions of the "
                "concatenated rows; rtn_int4_g64 = tools/oracle/fixtures/rtn_int4_g64.json\n");
  }

  // --- 3. the bytes and the plan ------------------------------------------------------------------------------
  {
    model::DFlashDesc d = parse(real);
    d.vocab = target.vocab;
    const size_t params = size_t(5) * (size_t(6656) * 6144 + size_t(4096) * 6656 + size_t(6656) * 39936 +
                                       size_t(19968) * 6656) + size_t(33280) * 6656;
    CHECK_EQ(params, size_t(2555904000ull));   // spec 23 §8.2
    const size_t small = size_t(6656) * 4 * 2 + size_t(5) * 2 * 6656 * 4 + size_t(5) * 2 * 128 * 4 + size_t(6656) * 4;
    CHECK_EQ(loader::dflash_weight_bytes(d, 6656, loader::DraftFormat::Bf16).total(), params * 2 + small);
    CHECK_EQ(loader::dflash_weight_bytes(d, 6656, loader::DraftFormat::Int4G64).total(), params / 1024 * 544 + small);
    const size_t i4 = loader::dflash_weight_bytes(d, 6656, loader::DraftFormat::Int4G64).total();
    CHECK(i4 > 1357000000ull && i4 < 1359000000ull);   // 1.358 GB
    // the buffers at 16 rows
    const runtime::dflash::DFlashSizes s = runtime::dflash::sizes(d, 6656, target.vocab, 131072);
    CHECK_EQ(runtime::dflash::list_rows(d), 16u);
    CHECK_EQ(runtime::dflash::ring_slots(d, 0, 131072), 2112u);
    CHECK_EQ(s.ring_k, size_t(5) * 2112 * 1024 * 2);              // 21.6 MB (spec 23 §8.2's 2048 + 64 slots)
    CHECK_EQ(s.taps, size_t(16) * 33280 * 2);                    // [16][5 x 6656] bf16
    CHECK_EQ(s.out, sizeof(runtime::dflash::MuseDraftOut));
    CHECK_EQ(s.vids, size_t(16) * 4);
    CHECK_EQ(s.logits, size_t(15) * 202048 * 4);
    CHECK_EQ(runtime::dflash::ring_snapshot_bytes(d, 131072), size_t(2) * 5 * 2112 * 1024 * 2);
    CHECK_EQ(runtime::dflash::draft_launches(d), size_t(66));
    // the plan: --max-len auto on one card with the int4 drafter, its 16-row verify scratch and the prefill staging
    namespace rm = runtime::muse;
    for (loader::DraftFormat f : {loader::DraftFormat::Int4G64, loader::DraftFormat::Int8}) {
      rm::MuseSpec spec;
      spec.rows = 16;
      spec.drafter = runtime::dflash::drafter_plan(d, 6656, target.vocab, f);
      const rm::MuseReserve r = rm::run_reserve(target, 0, 1, rm::MuseAttn::Flash, true, spec);
      CHECK_EQ(r.drafter, spec.drafter.bytes(131072));
      CHECK_EQ(r.decode_scratch, rm::decode_scratch_bytes(target, 131072, 16, rm::MuseAttn::Flash));
      CHECK_EQ(r.prefill_scratch, rm::prefill_reserve(target, 1) + size_t(2048) * 5 * 6656 * 2);
      const rm::MusePlan p = rm::plan_muse(target, 1, 0, true, r, {kCard});
      std::printf("%s\n", p.describe().c_str());
      CHECK(p.fits && p.max_len == 131072);
    }
    CHECK(throws_with([&] { (void)rm::decode_reserve(target, 0, 1, rm::MuseAttn::Flash, 17); }, "rows 17"));
    std::printf("3. int4 g64 %zu B (1.358 GB), bf16 %zu B; ring 2 x %zu B at 2112 slots; one card still plans 131072 "
                "with the int4 or the int8 drafter\n",
                i4, loader::dflash_weight_bytes(d, 6656, loader::DraftFormat::Bf16).total(), s.ring_k);
  }
  // --- 4. the final review's two rules (spec 23f) ---------------------------------------------------------------
  {
    const model::DFlashDesc d = parse(real);
    // the draft list always runs the whole 16-row block, whatever K the caller keeps (transformers' form); Qwen3.8's
    // drafter keeps dctl.k = K (its rows past K are masked)
    for (uint32_t k = 1; k <= 15; ++k) CHECK_EQ(runtime::dflash::draft_list_k(d, k), 15u);
    model::DFlashDesc q = d;
    q.form = model::DFlashForm::Vllm;
    q.block = 8;
    for (uint32_t k = 1; k <= 7; ++k) CHECK_EQ(runtime::dflash::draft_list_k(q, k), k);
    // the prefill's tapped rows of a chunk [pos, pos + C) of a call ending at `end`: the window before the call's end,
    // and - with a block hook - every row of a chunk that ends at a block end (its snapshot's ring holds [E - 2048, E))
    using runtime::muse::prefill_tap_r0;
    CHECK_EQ(prefill_tap_r0(0, 2048, 6000, 2048, false, 2048), 2048u);      // outside the window: nothing
    CHECK_EQ(prefill_tap_r0(2048, 2048, 6000, 2048, false, 2048), 1904u);   // from 3952
    CHECK_EQ(prefill_tap_r0(4096, 1904, 6000, 2048, false, 2048), 0u);      // [3952, 6000): all of [4096, 6000)
    CHECK_EQ(prefill_tap_r0(2048, 2048, 5000, 2048, false, 2048), 904u);    // from 2952
    CHECK_EQ(prefill_tap_r0(0, 2048, 6000, 2048, true, 2048), 0u);          // hooked block end 2048: the whole chunk
    CHECK_EQ(prefill_tap_r0(2048, 2048, 6000, 2048, true, 2048), 0u);       // hooked block end 4096
    CHECK_EQ(prefill_tap_r0(4096, 1904, 6000, 2048, true, 2048), 0u);       // the prompt's end: the window rule
    CHECK_EQ(prefill_tap_r0(0, 1000, 1000, 2048, true, 2048), 0u);
    CHECK_EQ(prefill_tap_r0(0, 1500, 9000, 2048, true, 2048), 1500u);       // a hooked chunk not at a block end
    std::printf("4. the draft list's dctl.k is 15 at every K; the prefill taps the window before the call's end and, "
                "hooked, every chunk ending at a block end\n");
  }
  std::printf("muse_assistant_test OK\n");
  return 0;
}
