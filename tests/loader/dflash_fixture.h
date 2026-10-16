#pragma once
// Spec 19b: tiny synthetic DFlash / DFlash 2 drafter checkpoints written by the C++ tests themselves - every tensor
// named as z-lab/Qwen3.8-27B-DFlash2 names it (one model.safetensors, no index, as every published drafter ships),
// random bf16 weights, a config.json in the published form. The widths are tiny (hidden 64, 2 layers, 4 / 2 heads x
// 16, inter 96, vocab 512, block 8, taps {1, 3} of a 4-layer target of hidden 64) so the host tests run in
// milliseconds; tests/model/dflash/qwen38_dflash2_config.json is the real config.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "common/bf16.h"

namespace dflash_fixture {

namespace fs = std::filesystem;

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};
using Tensors = std::map<std::string, Tensor>;

// One safetensors file (header + data, 8-byte aligned), optional __metadata__ (string -> string).
inline void write_safetensors(const fs::path& file, const Tensors& ts,
                              const std::map<std::string, std::string>& meta = {}) {
  std::string hdr = "{";
  bool first = true;
  if (!meta.empty()) {
    hdr += "\"__metadata__\":{";
    bool f2 = true;
    for (const auto& [k, v] : meta) {
      hdr += (f2 ? "" : ",") + ("\"" + k + "\":\"" + v + "\"");
      f2 = false;
    }
    hdr += "}";
    first = false;
  }
  uint64_t off = 0;
  for (const auto& [name, t] : ts) {
    hdr += (first ? "" : ",");
    first = false;
    hdr += "\"" + name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); ++i) hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes.size()) + "]}";
    off += t.bytes.size();
    off = (off + 7) / 8 * 8;
  }
  hdr += "}";
  while ((8 + hdr.size()) % 8 != 0) hdr += ' ';
  std::ofstream f(file, std::ios::binary);
  const uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  uint64_t pos = 0;
  for (const auto& [name, t] : ts) {
    (void)name;
    f.write(reinterpret_cast<const char*>(t.bytes.data()), std::streamsize(t.bytes.size()));
    pos += t.bytes.size();
    const uint64_t pad = (pos + 7) / 8 * 8 - pos;
    const char zeros[8] = {};
    f.write(zeros, std::streamsize(pad));
    pos += pad;
  }
}

struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  explicit Rng(uint64_t seed = 1) : s(0x9E3779B97F4A7C15ull ^ (seed * 0xD1B54A32D192ED03ull)) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
  float uni(float a) { return (float(next() & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f) * a; }
};

inline Tensor bf16_tensor(Rng& r, std::vector<uint64_t> shape, float amp, float offset = 0.0f) {
  size_t n = 1;
  for (uint64_t s : shape) n *= size_t(s);
  Tensor t{"BF16", std::move(shape), std::vector<uint8_t>(n * 2)};
  uint16_t* v = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (size_t i = 0; i < n; ++i) v[i] = common::f32_to_bf16(offset + r.uni(amp));
  return t;
}

// The tiny drafter's shapes.
struct Shape {
  uint32_t version = 2, layers = 2, hidden = 64, q_heads = 4, kv_heads = 2, head_dim = 16, inter = 96;
  uint32_t vocab = 512, block = 8, mask_id = 500, conv_group = 16, conv_taps = 2, rank = 16, top_k = 16;
  uint32_t target_hidden = 64, target_layers = 4;
  std::vector<uint32_t> taps = {1, 3};
  std::vector<std::string> layer_types = {"sliding_attention", "sliding_attention"};
  uint32_t window = 16;
  double theta = 10000000.0;
  uint32_t groups() const { return hidden / conv_group; }
};

inline std::string config_json(const Shape& s) {
  std::string taps, types;
  for (size_t i = 0; i < s.taps.size(); ++i) taps += (i ? "," : "") + std::to_string(s.taps[i]);
  for (size_t i = 0; i < s.layer_types.size(); ++i) types += (i ? "," : "") + ("\"" + s.layer_types[i] + "\"");
  std::string dc = "\"block_size\":" + std::to_string(s.block) + ",\"mask_token_id\":" + std::to_string(s.mask_id) +
                   ",\"target_layer_ids\":[" + taps + "]";
  if (s.version == 2)
    dc += ",\"conv_group_size\":" + std::to_string(s.conv_group) + ",\"conv_kernel_size\":" +
          std::to_string(s.conv_taps) + ",\"selector_rank\":" + std::to_string(s.rank) +
          ",\"selector_top_k\":" + std::to_string(s.top_k);
  return std::string("{\"architectures\":[\"") + (s.version == 2 ? "DFlash2DraftModel" : "DFlashDraftModel") +
         "\"],\"attention_bias\":false,\"is_causal\":false,\"dflash_config\":{" + dc +
         "},\"head_dim\":" + std::to_string(s.head_dim) + ",\"hidden_act\":\"silu\",\"hidden_size\":" +
         std::to_string(s.hidden) + ",\"intermediate_size\":" + std::to_string(s.inter) + ",\"layer_types\":[" +
         types + "],\"model_type\":\"qwen3\",\"num_attention_heads\":" + std::to_string(s.q_heads) +
         ",\"num_hidden_layers\":" + std::to_string(s.layers) + ",\"num_key_value_heads\":" +
         std::to_string(s.kv_heads) + ",\"num_target_layers\":" + std::to_string(s.target_layers) +
         ",\"rms_norm_eps\":1e-06,\"rope_parameters\":{\"rope_theta\":" + std::to_string(int64_t(s.theta)) +
         ",\"rope_type\":\"default\"},\"sliding_window\":" + std::to_string(s.window) +
         ",\"vocab_size\":" + std::to_string(s.vocab) + ",\"dtype\":\"bfloat16\"}";
}

// Every tensor of the drafter, as z-lab names them.
inline Tensors make(const Shape& s, uint64_t seed = 1) {
  Rng r(seed);
  Tensors ts;
  const uint64_t H = s.hidden, qn = uint64_t(s.q_heads) * s.head_dim, kvn = uint64_t(s.kv_heads) * s.head_dim;
  ts["fc.weight"] = bf16_tensor(r, {H, uint64_t(s.taps.size()) * s.target_hidden}, 0.1f);
  ts["hidden_norm.weight"] = bf16_tensor(r, {H}, 0.2f, 1.0f);
  ts["norm.weight"] = bf16_tensor(r, {H}, 0.2f, 1.0f);
  for (uint32_t l = 0; l < s.layers; ++l) {
    const std::string L = "layers." + std::to_string(l) + ".";
    ts[L + "input_layernorm.weight"] = bf16_tensor(r, {H}, 0.2f, 1.0f);
    ts[L + "post_attention_layernorm.weight"] = bf16_tensor(r, {H}, 0.2f, 1.0f);
    ts[L + "self_attn.q_proj.weight"] = bf16_tensor(r, {qn, H}, 0.15f);
    ts[L + "self_attn.k_proj.weight"] = bf16_tensor(r, {kvn, H}, 0.15f);
    ts[L + "self_attn.v_proj.weight"] = bf16_tensor(r, {kvn, H}, 0.15f);
    ts[L + "self_attn.o_proj.weight"] = bf16_tensor(r, {H, qn}, 0.15f);
    ts[L + "self_attn.q_norm.weight"] = bf16_tensor(r, {s.head_dim}, 0.2f, 1.0f);
    ts[L + "self_attn.k_norm.weight"] = bf16_tensor(r, {s.head_dim}, 0.2f, 1.0f);
    ts[L + "mlp.gate_proj.weight"] = bf16_tensor(r, {s.inter, H}, 0.15f);
    ts[L + "mlp.up_proj.weight"] = bf16_tensor(r, {s.inter, H}, 0.15f);
    ts[L + "mlp.down_proj.weight"] = bf16_tensor(r, {H, s.inter}, 0.15f);
    if (s.version == 2)
      for (const char* cv : {"attention_conv.", "mlp_conv."}) {
        ts[L + cv + "base_kernel"] = bf16_tensor(r, {2, s.conv_taps, H}, 0.3f, 0.5f);
        ts[L + cv + "kernel_projection.weight"] =
            bf16_tensor(r, {uint64_t(2) * s.conv_taps * s.groups(), H}, 0.05f);
      }
  }
  if (s.version == 2) {
    ts["candidate_selector.hidden_projection.weight"] = bf16_tensor(r, {s.rank, H}, 0.15f);
    ts["candidate_selector.predecessor_codebook"] = bf16_tensor(r, {s.vocab, s.rank}, 0.5f);
    ts["candidate_selector.successor_codebook"] = bf16_tensor(r, {s.vocab, s.rank}, 0.5f);
  }
  return ts;
}

inline void write(const fs::path& dir, const Shape& s, const Tensors& ts) {
  fs::create_directories(dir);
  write_safetensors(dir / "model.safetensors", ts);
  std::ofstream(dir / "config.json") << config_json(s);
}

}  // namespace dflash_fixture
