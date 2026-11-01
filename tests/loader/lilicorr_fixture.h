#pragma once
// Spec 26b: a tiny synthetic LiLiCorr checkpoint written by the C++ tests themselves, in the published checkpoint's names
// and dtypes (nvidia/...-24GB-LiLiCorr @ 449d2414: one model.safetensors, bare names, the NVFP4 linears as U8 + F8_E4M3 +
// F32 triplets, everything else BF16) and a config.json in the published form - random codes, random finite e4m3 scales
// (a zero and subnormal ones included), a random non-power-of-two g per linear. Widths: hidden 64, 2 layers, 4 / 1 heads
// x 16, FF 128, 6 taps of a hidden-64 target, conv groups of 16 (kernel projection 16 rows), window 16, the head 32 wide /
// 2 layers / 2 heads / MLP 64 / factor 32, top-8, block 8, vocab 1024 with mask id 990 (every GEMV width a multiple of
// 16, every K of the backbone a multiple of 64 - the int4 g64 arm).
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "loader/dflash_fixture.h"   // write_safetensors, Rng, bf16_tensor

namespace lilicorr_fixture {

namespace fs = std::filesystem;
using dflash_fixture::Rng;
using dflash_fixture::Tensor;
using dflash_fixture::Tensors;

struct Shape {
  uint32_t layers = 2, hidden = 64, q_heads = 4, kv_heads = 1, head_dim = 16, inter = 128, vocab = 1024, mask_id = 990;
  uint32_t window = 16, group = 16, ctaps = 2, target_hidden = 64;
  std::vector<uint32_t> taps = {1, 2, 4, 6, 9, 10};
  uint32_t lil_hidden = 32, lil_layers = 2, lil_heads = 2, lil_mlp = 64, factor = 32, topk = 8, block = 8;
  bool quant_config = true;
  uint32_t kproj() const { return 2 * ctaps * (hidden / group); }
};

inline std::string config_json(const Shape& s) {
  std::string taps, types;
  for (size_t i = 0; i < s.taps.size(); ++i) taps += (i ? "," : "") + std::to_string(s.taps[i]);
  for (uint32_t i = 0; i < s.layers; ++i) types += std::string(i ? "," : "") + "\"sliding_attention\"";
  const std::string qc =
      s.quant_config ? ",\"quantization_config\":{\"quant_algo\":\"W4A16_NVFP4\",\"group_size\":16,"
                       "\"quant_method\":\"modelopt_fp4\"}"
                     : "";
  return "{\"architectures\":[\"LiLiCorrDraftModel\"],\"model_type\":\"qwen3\",\"block_size\":" + std::to_string(s.block) +
         ",\"dflash_config\":{\"mask_token_id\":" + std::to_string(s.mask_id) + ",\"target_layer_ids\":[" + taps +
         "],\"use_swa\":true,\"swa_window_size\":" + std::to_string(s.window) +
         ",\"causal\":false,\"attention_sink_bias\":true,\"projector_type\":\"lilicorr\",\"lilicorr_enabled\":true,"
         "\"lilicorr_candidate_topk\":" + std::to_string(s.topk) + ",\"lilicorr_hidden_size\":" +
         std::to_string(s.lil_hidden) + ",\"lilicorr_num_layers\":" + std::to_string(s.lil_layers) +
         ",\"lilicorr_num_heads\":" + std::to_string(s.lil_heads) + ",\"lilicorr_mlp_ratio\":" +
         std::to_string(double(s.lil_mlp) / s.lil_hidden) + ",\"lilicorr_factor_dim\":" + std::to_string(s.factor) +
         ",\"lilicorr_vector_eps\":0.0001,\"lilicorr_logit_scale\":8.0,\"conv_kernel_size\":" + std::to_string(s.ctaps) +
         ",\"conv_group_size\":" + std::to_string(s.group) + "},\"hidden_size\":" + std::to_string(s.hidden) +
         ",\"num_hidden_layers\":" + std::to_string(s.layers) + ",\"num_attention_heads\":" + std::to_string(s.q_heads) +
         ",\"num_key_value_heads\":" + std::to_string(s.kv_heads) + ",\"head_dim\":" + std::to_string(s.head_dim) +
         ",\"intermediate_size\":" + std::to_string(s.inter) + ",\"hidden_act\":\"silu\",\"rms_norm_eps\":1e-06,"
         "\"vocab_size\":" + std::to_string(s.vocab) + ",\"attention_bias\":false,\"rope_theta\":10000.0,"
         "\"layer_types\":[" + types + "],\"sliding_window\":" + std::to_string(s.window) +
         ",\"attention_sink_bias\":true" + qc + "}";
}

// One NVFP4 linear [N][K]: random codes, random finite e4m3 scales (never a NaN code), [0][0] zero and [1][0..] subnormal
// codes 0x01..; g a non-power-of-two fp32.
inline void put_nvfp4(Tensors& ts, Rng& r, const std::string& base, uint64_t N, uint64_t K, float g) {
  Tensor w{"U8", {N, K / 2}, std::vector<uint8_t>(N * K / 2)};
  for (uint8_t& b : w.bytes) b = uint8_t(r.next());
  Tensor s{"F8_E4M3", {N, K / 16}, std::vector<uint8_t>(N * K / 16)};
  for (uint8_t& b : s.bytes) {
    uint8_t c = uint8_t(0x30 + r.next() % 0x18);   // ~ 2^-1 .. 2^2, positive
    b = c;
  }
  if (N >= 2 && K >= 32) {
    s.bytes[0] = 0x00;
    for (uint64_t j = 0; j < std::min<uint64_t>(K / 16, 7); ++j) s.bytes[K / 16 + j] = uint8_t(1 + j);
  }
  Tensor g2{"F32", {}, std::vector<uint8_t>(4)};
  std::memcpy(g2.bytes.data(), &g, 4);
  ts[base + ".weight"] = w;
  ts[base + ".weight_scale"] = s;
  ts[base + ".weight_scale_2"] = g2;
}

// The NVFP4 linears (the published checkpoint's 25 at these widths: fc, every layer's gate / up / down, the head's
// token / pass_hidden / context projections and every head layer's out_proj and MLP).
inline std::set<std::string> nvfp4_bases(const Shape& s) {
  std::set<std::string> b = {"fc", "lilicorr.token_proj", "lilicorr.pass_hidden_proj", "lilicorr.context_proj"};
  for (uint32_t l = 0; l < s.layers; ++l)
    for (const char* p : {"gate", "up", "down"}) b.insert("layers." + std::to_string(l) + ".mlp." + p + "_proj");
  for (uint32_t l = 0; l < s.lil_layers; ++l)
    for (const char* p : {"attn.out_proj", "mlp.up_proj", "mlp.down_proj"})
      b.insert("lilicorr.layers." + std::to_string(l) + "." + p);
  return b;
}

inline Tensors make(const Shape& s, uint64_t seed = 1, float g_scale = 1.0f) {
  Rng r(seed);
  Tensors ts;
  const std::set<std::string> q = nvfp4_bases(s);
  uint32_t gi = 0;
  auto lin = [&](const std::string& base, uint64_t N, uint64_t K, float amp) {
    if (q.count(base)) {
      const float g = g_scale * (0.0013f + 0.00037f * float(gi++)) * 1.37f;
      put_nvfp4(ts, r, base, N, K, g);
    } else {
      ts[base + ".weight"] = dflash_fixture::bf16_tensor(r, {N, K}, amp);
    }
  };
  auto bf = [&](const std::string& n, std::vector<uint64_t> shape, float amp, float off = 0.0f) {
    ts[n] = dflash_fixture::bf16_tensor(r, std::move(shape), amp, off);
  };
  const uint64_t H = s.hidden, qn = uint64_t(s.q_heads) * s.head_dim, kvn = uint64_t(s.kv_heads) * s.head_dim;
  lin("fc", H, uint64_t(s.taps.size()) * s.target_hidden, 0.1f);
  bf("hidden_norm.weight", {H}, 0.2f, 1.0f);
  bf("norm.weight", {H}, 0.2f, 1.0f);
  for (uint32_t l = 0; l < s.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    bf(L + "input_layernorm.weight", {H}, 0.2f, 1.0f);
    bf(L + "post_attention_layernorm.weight", {H}, 0.2f, 1.0f);
    lin(L + "self_attn.q_proj", qn, H, 0.15f);
    lin(L + "self_attn.k_proj", kvn, H, 0.15f);
    lin(L + "self_attn.v_proj", kvn, H, 0.15f);
    lin(L + "self_attn.o_proj", H, qn, 0.15f);
    bf(L + "self_attn.q_norm.weight", {s.head_dim}, 0.2f, 1.0f);
    bf(L + "self_attn.k_norm.weight", {s.head_dim}, 0.2f, 1.0f);
    bf(L + "self_attn.attention_sink_bias", {s.q_heads}, 1.5f);
    lin(L + "mlp.gate_proj", s.inter, H, 0.15f);
    lin(L + "mlp.up_proj", s.inter, H, 0.15f);
    lin(L + "mlp.down_proj", H, s.inter, 0.15f);
    for (const char* cv : {"attention_conv.", "mlp_conv."}) {
      bf(L + cv + "base_kernel", {2, s.ctaps, H}, 0.3f, 0.5f);
      lin(L + cv + "kernel_projection", s.kproj(), H, 0.05f);
    }
  }
  const std::string P = "lilicorr.";
  const uint64_t Hl = s.lil_hidden, F = s.factor;
  auto hl = [&](const std::string& n, uint64_t N, uint64_t K) {
    lin(P + n, N, K, 0.2f);
    bf(P + n + ".bias", {N}, 0.1f);
  };
  if (Hl != H) hl("token_proj", Hl, H);
  hl("pass_hidden_proj", Hl, H);
  bf(P + "feature_norm.weight", {5}, 0.1f, 1.0f);
  bf(P + "feature_norm.bias", {5}, 0.1f);
  hl("feature_mlp.up_proj", Hl, 5);
  hl("feature_mlp.down_proj", Hl, Hl);
  bf(P + "slot_embedding", {1, 1, uint64_t(s.block - 1), 1, Hl}, 0.5f);
  bf(P + "rank_embedding", {1, 1, 1, s.topk, Hl}, 0.5f);
  bf(P + "relative_slot_bias", {s.lil_heads, uint64_t(2 * s.block - 1)}, 1.0f);
  bf(P + "same_slot_bias", {s.lil_heads}, 1.0f);
  hl("context_proj", Hl, H);
  for (uint32_t l = 0; l < s.lil_layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    bf(P + L + "attn_norm.weight", {Hl}, 0.1f, 1.0f);
    bf(P + L + "attn.in_proj_weight", {3 * Hl, Hl}, 0.2f);
    bf(P + L + "attn.in_proj_bias", {3 * Hl}, 0.1f);
    hl(L + "attn.out_proj", Hl, Hl);
    bf(P + L + "mlp_norm.weight", {Hl}, 0.1f, 1.0f);
    hl(L + "mlp.up_proj", s.lil_mlp, Hl);
    hl(L + "mlp.down_proj", Hl, s.lil_mlp);
  }
  bf(P + "output_norm.weight", {Hl}, 0.1f, 1.0f);
  bf(P + "anchor_norm.weight", {Hl}, 0.1f, 1.0f);
  hl("factor_input_proj", Hl, 3 * Hl);
  hl("out_head", F, Hl);
  hl("in_head", F, Hl);
  hl("anchor_out_head", F, Hl);
  return ts;
}

inline void write(const fs::path& dir, const Shape& s, const Tensors& ts) {
  fs::create_directories(dir);
  dflash_fixture::write_safetensors(dir / "model.safetensors", ts);
  std::ofstream(dir / "config.json") << config_json(s);
}

}  // namespace lilicorr_fixture
