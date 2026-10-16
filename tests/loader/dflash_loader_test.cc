// Spec 19b Task 1 (host): the DFlash drafter's descriptor and the loader's host half, no device.
//
//   1. The published config (tests/model/dflash/qwen38_dflash2_config.json, z-lab/Qwen3.8-27B-DFlash2's) -> the
//      descriptor spec 19 §1 tabulates; the drafter held to Qwen3.8 (model::qwen38(), 248320 head rows) passes, and
//      every target mismatch (a tap layer past the target's, another vocab, another hidden, num_target_layers, a
//      mask id the tokenizer does not define) throws naming its key.
//   2. Every config refusal by name (model/dflash.h's list), and v1 (DFlashDraftModel) parsed without the convs and
//      the selector; per-layer windows / causality as dflash_ref.py's resolve_layer_attention.
//   3. A tiny synthetic checkpoint (tests/loader/dflash_fixture.h, one model.safetensors) in both versions and both
//      formats: names both ways (a missing tensor, an unknown one, a shipped embedding / head, d2t, mask_embedding.pt
//      each refused by name), every linear's device form equal to the layout functions applied to the concatenated
//      bf16 rows (bf16: common::repack_bf16_tiled; int8: loader::quantise_int8_tiled - spec 9's quantiser), q||k||v
//      and gate||up in that order, the k||v rows of q||k||v at the projection's offset, the norms and conv bases as
//      fp32 widenings, the codebooks the shard's own rows, mask_embedding.safetensors read; and the pieces' bytes ==
//      loader::dflash_weight_bytes (the planner's formula).
//   4. The real shapes' byte totals (printed; the planner's numbers): bf16 and int8.
#include <unistd.h>

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
#include "model/dflash.h"
#include "model/model_desc.h"
#include "model/qwen35.h"

namespace fs = std::filesystem;
namespace fx = dflash_fixture;

namespace {

bool throws_with(const std::function<void()>& f, const std::string& needle) {
  try {
    f();
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) != std::string::npos) return true;
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

// Edit a JSON text by substring replacement (the configs are machine-written; tests only).
std::string with(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  CHECK(at != std::string::npos);
  return s.replace(at, from.size(), to);
}

model::DFlashDesc parse(const std::string& text) { return model::dflash_desc_from_config(common::json::parse(text)); }

fs::path tmpdir(const std::string& tag) {
  fs::path p = fs::temp_directory_path() / ("dflash_loader_test_" + tag + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  fs::create_directories(p);
  return p;
}

std::vector<uint16_t> rows_of(const fx::Tensors& ts, const std::vector<std::string>& names) {
  std::vector<uint16_t> out;
  for (const std::string& n : names) {
    const fx::Tensor& t = ts.at(n);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
    out.insert(out.end(), p, p + t.bytes.size() / 2);
  }
  return out;
}

// The device form a linear of these rows must have.
void check_linear(const loader::DFlashHostLinear& h, const std::vector<uint16_t>& rows, uint32_t K, uint32_t N,
                  loader::DraftFormat f) {
  CHECK_EQ(h.K, K);
  CHECK_EQ(h.N, N);
  CHECK_EQ(rows.size(), size_t(K) * N);
  if (f == loader::DraftFormat::Bf16) {
    CHECK(h.kind == model::WeightKind::Bf16);
    std::vector<uint16_t> want(size_t(K) * N);
    common::repack_bf16_tiled(rows.data(), K, N, want.data());
    CHECK(h.bf16 == want);
    CHECK(h.i8.empty() && h.scales.empty());
  } else {
    CHECK(h.kind == model::WeightKind::Int8);
    std::vector<int8_t> q(size_t(K) * N);
    std::vector<float> s(N);
    loader::quantise_int8_tiled(rows.data(), K, N, q.data(), s.data(), 1);
    CHECK(h.i8 == q);
    CHECK(h.scales == s);
    CHECK(h.bf16.empty());
  }
  CHECK_EQ(h.bytes(), loader::dflash_linear_bytes(K, N, f));
}

std::vector<float> widen(const fx::Tensor& t) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
  std::vector<float> v(t.bytes.size() / 2);
  for (size_t i = 0; i < v.size(); ++i) v[i] = common::bf16_to_f32(p[i]);
  return v;
}

void test_real_config(const std::string& path) {
  const model::DFlashDesc d = parse(slurp(path));
  CHECK_EQ(d.version, 2u);
  CHECK_EQ(d.architecture, std::string("DFlash2DraftModel"));
  CHECK(d.layers == 5 && d.hidden == 5120 && d.q_heads == 32 && d.kv_heads == 8 && d.head_dim == 128);
  CHECK(d.inter == 17408 && d.vocab == 248320 && d.block == 8 && d.mask_id == 248070);
  CHECK(d.num_target_layers == 64);
  CHECK((d.taps == std::vector<uint32_t>{5, 19, 33, 47, 61}));
  CHECK(d.window.size() == 5 && d.causal.size() == 5);
  for (uint32_t l = 0; l < 5; ++l) CHECK(d.window[l] == 2048 && !d.causal[l]);
  CHECK(d.all_sliding() && d.max_window() == 2048);
  CHECK(d.eps == 1e-6f && d.theta == 1e7 && d.neox);
  CHECK(d.conv_group == 16 && d.conv_taps == 2 && d.sel_rank == 256 && d.sel_top_k == 16);
  CHECK(d.in_scale == 1.0f && d.out_mult == 1.0f && d.softcap == 0.0f);
  CHECK(d.q_n() == 4096 && d.kv_n() == 1024 && d.qkv_n() == 6144 && d.gate_up_n() == 34816);
  CHECK(d.groups() == 320 && d.kproj_n() == 1280 && d.max_k() == 7 && d.fc_k(5120) == 25600);

  const model::ModelDesc& q = model::qwen38();
  model::check_dflash_target(d, q, model::Qwen35::kVocab);   // passes
  model::DFlashDesc bad = d;
  bad.taps = {5, 19, 33, 47, 64};
  CHECK(throws_with([&] { model::check_dflash_target(bad, q, model::Qwen35::kVocab); }, "target_layer_ids"));
  CHECK(throws_with([&] { model::check_dflash_target(d, q, 202048); }, "vocab_size"));
  bad = d;
  bad.hidden = 4096;
  CHECK(throws_with([&] { model::check_dflash_target(bad, q, model::Qwen35::kVocab); }, "hidden_size"));
  bad = d;
  bad.num_target_layers = 40;
  CHECK(throws_with([&] { model::check_dflash_target(bad, q, model::Qwen35::kVocab); }, "num_target_layers"));
  bad = d;
  bad.mask_id = 248320;   // past the embedding rows (248077 itself is legal: Ornith masks with it)
  CHECK(throws_with([&] { model::check_dflash_target(bad, q, model::Qwen35::kVocab); }, "mask_token_id"));
  std::printf("real config: DFlash 2, 5 x 5120, taps 5/19/33/47/61, window 2048, block 8 - held to qwen3.8\n");
}

void test_config_refusals(const std::string& path) {
  const std::string c = slurp(path);
  struct Case {
    std::string from, to, needle;
  };
  const std::vector<Case> cases = {
      {"\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", "hidden_act"},
      {"\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", "rope_type"},
      {"\"rope_type\": \"default\"", "\"rope_type\": \"default\", \"partial_rotary_factor\": 0.5",
       "partial_rotary_factor"},
      {"\"block_size\": 8,", "\"block_size\": 8, \"sample_from_anchor\": true,", "sample_from_anchor"},
      {"\"block_size\": 8,", "\"block_size\": 8, \"use_aux_hidden_state\": false,", "use_aux_hidden_state"},
      {"5,\n      19,", "19,\n      5,", "target_layer_ids"},
      {"\"vocab_size\": 248320", "\"vocab_size\": 248320, \"draft_vocab_size\": 32000", "draft_vocab_size"},
      {"\"attention_bias\": false", "\"attention_bias\": true", "attention_bias"},
      {"\"DFlash2DraftModel\"", "\"EagleDraftModel\"", "architectures"},
      {"\"conv_group_size\": 16", "\"conv_group_size\": 17", "conv_group_size"},
      {"\"block_size\": 8", "\"block_size\": 1", "block_size"},
      {"\"dtype\": \"bfloat16\"", "\"dtype\": \"bfloat16\", \"quantization_config\": {\"quant_method\": \"gptq\"}",
       "quantization_config"},
      {"\"block_size\": 8,", "\"block_size\": 8, \"attention_value_scale\": 0.5,", "attention_value_scale"},
      {"\"block_size\": 8,", "\"block_size\": 8, \"attention_sink_bias\": true,", "attention_sink_bias"},
  };
  for (const Case& k : cases) {
    std::printf("  refusal %-26s ", k.needle.c_str());
    CHECK(throws_with([&] { parse(with(c, k.from, k.to)); }, k.needle));
    std::printf("ok\n");
  }
  // v1: the architecture and a dflash_config without the convs and the selector.
  std::string v1 = with(c, "\"DFlash2DraftModel\"", "\"DFlashDraftModel\"");
  v1 = with(v1, "\"conv_group_size\": 16,", "");
  v1 = with(v1, "\"conv_kernel_size\": 2,", "");
  v1 = with(v1, "\"selector_rank\": 256,", "");
  v1 = with(v1, "\"selector_top_k\": 16,", "");
  const model::DFlashDesc d1 = parse(v1);
  CHECK(d1.version == 1 && !d1.v2() && d1.kproj_n() == 0 && d1.sel_rank == 0);
  // resolve_layer_attention: a full layer has no window; a dflash_config.causal is honoured.
  const model::DFlashDesc d2 =
      parse(with(c, "\"sliding_attention\"\n  ]", "\"full_attention\"\n  ]"));
  CHECK(d2.window[4] == 0 && d2.window[0] == 2048 && !d2.all_sliding());
  const model::DFlashDesc d3 = parse(with(with(c, "\"is_causal\": false,", ""), "\"block_size\": 8,",
                                          "\"block_size\": 8, \"causal\": true,"));
  CHECK(d3.causal[0] && d3.causal[4]);
  // No is_causal, no dflash_config.causal: layer types decide (sliding layers causal), vLLM's rule.
  const model::DFlashDesc d4 = parse(with(c, "\"is_causal\": false,", ""));
  CHECK(d4.causal[0]);
  std::printf("config refusals: %zu by name; v1 parsed; windows / causality per layer\n", cases.size());
}

void test_checkpoint(uint32_t version) {
  fx::Shape s;
  s.version = version;
  const fx::Tensors ts = fx::make(s, 7 + version);
  const fs::path dir = tmpdir("v" + std::to_string(version));
  fx::write(dir, s, ts);
  const loader::DFlashCheckpoint ck(dir.string());
  const model::DFlashDesc& d = ck.desc();
  CHECK_EQ(d.version, version);
  CHECK(ck.fc_k() == 128 && ck.target_hidden() == 64);
  CHECK_EQ(ck.tensors(), ts.size());

  // The target it drafts for: a hand-built descriptor of hidden 64, 4 layers, 512 rows of which 510 defined.
  model::ModelDesc target = model::qwen38();
  target.hidden = 64;
  target.layers = 4;
  target.vocab_used = 510;
  ck.check_target(target, 512);
  {
    model::ModelDesc t2 = target;
    t2.hidden = 128;   // fc's K 128 = 2 taps x 64: another target hidden is refused
    CHECK(throws_with([&] { ck.check_target(t2, 512); }, "hidden"));
  }

  const uint32_t H = s.hidden;
  for (loader::DraftFormat f : {loader::DraftFormat::Bf16, loader::DraftFormat::Int8}) {
    size_t linears = 0;
    loader::DFlashHostLinear fc = ck.fc(f, 2);
    check_linear(fc, rows_of(ts, {"fc.weight"}), 128, H, f);
    linears += fc.bytes();
    for (uint32_t l = 0; l < s.layers; ++l) {
      const std::string L = "layers." + std::to_string(l) + ".";
      using W = loader::DFlashLinear;
      auto qkv = ck.layer_linear(l, W::Qkv, f);
      check_linear(qkv, rows_of(ts, {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight",
                                     L + "self_attn.v_proj.weight"}),
                   H, d.qkv_n(), f);
      // The projection's k||v rows are q||k||v's rows [q_n, qkv_n): a tile-aligned slice of the same layout.
      CHECK(d.q_n() % 16 == 0);
      auto o = ck.layer_linear(l, W::O, f);
      check_linear(o, rows_of(ts, {L + "self_attn.o_proj.weight"}), d.q_n(), H, f);
      auto gu = ck.layer_linear(l, W::GateUp, f);
      check_linear(gu, rows_of(ts, {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"}), H, d.gate_up_n(), f);
      auto dn = ck.layer_linear(l, W::Down, f);
      check_linear(dn, rows_of(ts, {L + "mlp.down_proj.weight"}), d.inter, H, f);
      linears += qkv.bytes() + o.bytes() + gu.bytes() + dn.bytes();
      if (version == 2) {
        auto ak = ck.layer_linear(l, W::AttnKproj, f);
        check_linear(ak, rows_of(ts, {L + "attention_conv.kernel_projection.weight"}), H, d.kproj_n(), f);
        auto mk = ck.layer_linear(l, W::MlpKproj, f);
        check_linear(mk, rows_of(ts, {L + "mlp_conv.kernel_projection.weight"}), H, d.kproj_n(), f);
        linears += ak.bytes() + mk.bytes();
      } else {
        CHECK(throws_with([&] { ck.layer_linear(l, W::AttnKproj, f); }, "DFlash 2"));
      }
      const std::vector<float> norms = ck.layer_norms(l);
      std::vector<float> want = widen(ts.at(L + "input_layernorm.weight"));
      const std::vector<float> post = widen(ts.at(L + "post_attention_layernorm.weight"));
      want.insert(want.end(), post.begin(), post.end());
      CHECK(norms == want);
    }
    if (version == 2) {
      auto sp = ck.selector_projection(f);
      check_linear(sp, rows_of(ts, {"candidate_selector.hidden_projection.weight"}), H, d.sel_rank, f);
      linears += sp.bytes();
    } else {
      CHECK(throws_with([&] { ck.selector_projection(f); }, "DFlash 2"));
    }
    const loader::DFlashBytes b = loader::dflash_weight_bytes(d, ck.target_hidden(), f);
    CHECK_EQ(b.linears, linears);
  }
  CHECK(ck.hidden_norm() == widen(ts.at("hidden_norm.weight")));
  CHECK(ck.final_norm() == widen(ts.at("norm.weight")));
  {
    const std::vector<float> qk = ck.qk_norms();
    CHECK_EQ(qk.size(), size_t(s.layers) * 2 * s.head_dim);
    for (uint32_t l = 0; l < s.layers; ++l) {
      const std::string L = "layers." + std::to_string(l) + ".self_attn.";
      const std::vector<float> q = widen(ts.at(L + "q_norm.weight")), k = widen(ts.at(L + "k_norm.weight"));
      CHECK(std::equal(q.begin(), q.end(), qk.begin() + loader::dflash_qk_norm_off(d, l, 0)));
      CHECK(std::equal(k.begin(), k.end(), qk.begin() + loader::dflash_qk_norm_off(d, l, 1)));
    }
  }
  if (version == 2) {
    const std::vector<float> cb = ck.conv_bases();
    CHECK_EQ(cb.size(), size_t(s.layers) * 2 * 2 * s.conv_taps * H);
    for (uint32_t l = 0; l < s.layers; ++l)
      for (uint32_t c = 0; c < 2; ++c) {
        const std::vector<float> base =
            widen(ts.at("layers." + std::to_string(l) + (c ? ".mlp_conv" : ".attention_conv") + ".base_kernel"));
        for (uint32_t side = 0; side < 2; ++side)
          CHECK(std::equal(base.begin() + size_t(side) * s.conv_taps * H,
                           base.begin() + size_t(side + 1) * s.conv_taps * H,
                           cb.begin() + loader::dflash_conv_base_off(d, l, c, side)));
      }
    for (bool succ : {false, true}) {
      const fx::Tensor& t = ts.at(succ ? "candidate_selector.successor_codebook"
                                       : "candidate_selector.predecessor_codebook");
      CHECK(std::memcmp(ck.codebook(succ), t.bytes.data(), t.bytes.size()) == 0);
    }
  } else {
    CHECK(ck.conv_bases().empty());
  }
  CHECK(!ck.ships_mask_embedding() && ck.mask_embedding().empty());

  // The mask embedding: a safetensors file replaces the row; fp32 [1, hidden] with a matching id.
  {
    fx::Rng r(3);
    fx::Tensor e{"F32", {1, H}, std::vector<uint8_t>(size_t(H) * 4)};
    std::vector<float> v(H);
    for (uint32_t i = 0; i < H; ++i) v[i] = r.uni(1.0f);
    std::memcpy(e.bytes.data(), v.data(), e.bytes.size());
    fx::write_safetensors(dir / "mask_embedding.safetensors", {{"embedding", e}},
                          {{"mask_token_id", std::to_string(s.mask_id)}});
    const loader::DFlashCheckpoint ck2(dir.string());
    CHECK(ck2.ships_mask_embedding() && ck2.mask_embedding() == v);
    fx::write_safetensors(dir / "mask_embedding.safetensors", {{"embedding", e}}, {{"mask_token_id", "7"}});
    CHECK(throws_with([&] { loader::DFlashCheckpoint x(dir.string()); }, "mask_token_id"));
    fs::remove(dir / "mask_embedding.safetensors");
    std::ofstream(dir / "mask_embedding.pt") << "pickle";
    CHECK(throws_with([&] { loader::DFlashCheckpoint x(dir.string()); }, "dflash_mask_embedding.py"));
    fs::remove(dir / "mask_embedding.pt");
  }

  // Names both ways.
  auto refused = [&](fx::Tensors t2, const std::string& needle) {
    const fs::path d2 = tmpdir("names");
    fx::write(d2, s, t2);
    const bool ok = throws_with([&] { loader::DFlashCheckpoint x(d2.string()); }, needle);
    fs::remove_all(d2);
    return ok;
  };
  {
    fx::Tensors t2 = ts;
    t2.erase("layers.1.mlp.down_proj.weight");
    CHECK(refused(t2, "layers.1.mlp.down_proj.weight"));
  }
  {
    fx::Tensors t2 = ts;
    t2["layers.0.self_attn.extra.weight"] = ts.at("norm.weight");
    CHECK(refused(t2, "layers.0.self_attn.extra.weight"));
  }
  for (const char* own : {"embed_tokens.weight", "lm_head.weight", "d2t", "mask_hidden"}) {
    fx::Tensors t2 = ts;
    t2[own] = ts.at("norm.weight");
    CHECK(refused(t2, own));
  }
  {
    fx::Tensors t2 = ts;
    t2["norm.weight"].dtype = "F16";
    CHECK(refused(t2, "BF16"));
  }
  {
    fx::Tensors t2 = ts;
    t2["layers.0.self_attn.q_norm.weight"].shape = {8, 2};
    CHECK(refused(t2, "layers.0.self_attn.q_norm.weight"));
  }
  {   // the `model.` prefix is dflash_ref.py's _canon: accepted
    fx::Tensors t2;
    for (const auto& [n, t] : ts) t2["model." + n] = t;
    const fs::path d2 = tmpdir("prefix");
    fx::write(d2, s, t2);
    const loader::DFlashCheckpoint x(d2.string());
    CHECK(x.hidden_norm() == widen(ts.at("hidden_norm.weight")));
    fs::remove_all(d2);
  }
  fs::remove_all(dir);
  std::printf("checkpoint v%u: names both ways, bf16 / int8 forms == the layout functions, bytes == the formula\n",
              version);
}

void test_real_bytes(const std::string& path) {
  const model::DFlashDesc d = parse(slurp(path));
  for (loader::DraftFormat f : {loader::DraftFormat::Bf16, loader::DraftFormat::Int8}) {
    const loader::DFlashBytes b = loader::dflash_weight_bytes(d, 5120, f);
    std::printf("Qwen3.8-27B-DFlash2 %s: linears %zu B, small %zu B, codebooks %zu B, total %.3f GB\n",
                loader::draft_format_name(f), b.linears, b.small, b.codebooks, b.total() / 1e9);
  }
  // bf16 linears are the checkpoint's own bf16 bytes minus the norms, conv bases and codebooks (derived from its
  // header: 3,848,808,960 B of tensors in all).
  const loader::DFlashBytes bf = loader::dflash_weight_bytes(d, 5120, loader::DraftFormat::Bf16);
  const size_t norms_and_bases = (size_t(5120) * 2 + 5 * (2 * 5120 + 2 * 128) + 5 * 2 * 2 * 2 * 5120) * 2;
  CHECK_EQ(bf.linears + bf.codebooks + norms_and_bases, size_t(3848808960ull));
  loader::DraftFormat f;
  CHECK(loader::parse_draft_format("int8", f) && f == loader::DraftFormat::Int8);
  CHECK(loader::parse_draft_format("bf16", f) && f == loader::DraftFormat::Bf16);
  // spec 23f: int4 (g64 RTN, Muse-Glimmer's assistant; Qwen3.8's CLI refuses it) parses; anything else does not
  CHECK(loader::parse_draft_format("int4", f) && f == loader::DraftFormat::Int4G64);
  CHECK(!loader::parse_draft_format("fp8", f) && !loader::parse_draft_format("int4g128", f));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cfg = argc > 1 ? argv[1] : "tests/model/dflash/qwen38_dflash2_config.json";
  test_real_config(cfg);
  test_config_refusals(cfg);
  test_checkpoint(2);
  test_checkpoint(1);
  test_real_bytes(cfg);
  std::printf("dflash_loader_test: PASS\n");
  return 0;
}
