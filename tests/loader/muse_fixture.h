#pragma once
// Spec 23b: synthetic Muse-Glimmer checkpoints written by the C++ tests themselves (qwen4exp_repack_test's writer,
// shared by muse_repack_test and muse_load_checkpoint_test): every tensor named as the export names it (either
// Vishva007 spelling), random int4 words with distinct finite f16 scales and 0x77777777 qzeros, bf16 norms /
// embedding / head, plus a few vision tensors the loader must skip by name.
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "common/bf16.h"
#include "model/muse_glimmer.h"

namespace muse_fixture {

namespace fs = std::filesystem;

struct Tensor {
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> bytes;
};
using Tensors = std::map<std::string, Tensor>;

inline void write_checkpoint(const fs::path& dir, const Tensors& ts, const std::string& config_json) {
  fs::create_directories(dir);
  std::string hdr = "{";
  uint64_t off = 0;
  bool first = true;
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
  {
    std::ofstream f(dir / "model-00001-of-00001.safetensors", std::ios::binary);
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
  std::ofstream ix(dir / "model.safetensors.index.json");
  ix << "{\"metadata\":{},\"weight_map\":{";
  first = true;
  for (const auto& [name, t] : ts) {
    (void)t;
    ix << (first ? "" : ",") << "\"" << name << "\":\"model-00001-of-00001.safetensors\"";
    first = false;
  }
  ix << "}}";
  std::ofstream(dir / "config.json") << config_json;
}

struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return uint32_t(s >> 16);
  }
  float uni(float a) { return (float(next() & 0xFFFFFF) / float(0xFFFFFF) * 2.0f - 1.0f) * a; }
};

inline void add_bf16(Rng& r, Tensors& ts, const std::string& name, std::vector<uint64_t> shape, float amp = 1.0f) {
  size_t n = 1;
  for (uint64_t s : shape) n *= size_t(s);
  Tensor t{"BF16", std::move(shape), std::vector<uint8_t>(n * 2)};
  uint16_t* v = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (size_t i = 0; i < n; ++i) v[i] = common::f32_to_bf16(r.uni(amp));
  ts[name] = std::move(t);
}

// A GPTQ int4 g64 linear [out N][in K]: random words, distinct finite scales, qzeros 0x77777777, g_idx i // 64
// when `g_idx`.
inline void add_int4(Rng& r, Tensors& ts, const std::string& prefix, uint32_t K, uint32_t N, bool g_idx) {
  Tensor qw{"I32", {K / 8, N}, std::vector<uint8_t>(size_t(K / 8) * N * 4)};
  uint32_t* w = reinterpret_cast<uint32_t*>(qw.bytes.data());
  for (size_t i = 0; i < size_t(K / 8) * N; ++i) w[i] = r.next() ^ (r.next() << 16);
  Tensor sc{"F16", {K / 64, N}, std::vector<uint8_t>(size_t(K / 64) * N * 2)};
  uint16_t* s = reinterpret_cast<uint16_t*>(sc.bytes.data());
  for (size_t i = 0; i < size_t(K / 64) * N; ++i) s[i] = common::f32_to_f16(0.01f + 0.04f * (r.uni(1.0f) + 1.0f));
  Tensor qz{"I32", {K / 64, N / 8}, std::vector<uint8_t>(size_t(K / 64) * (N / 8) * 4)};
  uint32_t* z = reinterpret_cast<uint32_t*>(qz.bytes.data());
  for (size_t i = 0; i < size_t(K / 64) * (N / 8); ++i) z[i] = 0x77777777u;
  ts[prefix + ".qweight"] = std::move(qw);
  ts[prefix + ".scales"] = std::move(sc);
  ts[prefix + ".qzeros"] = std::move(qz);
  if (g_idx) {
    Tensor gi{"I32", {K}, std::vector<uint8_t>(size_t(K) * 4)};
    int32_t* g = reinterpret_cast<int32_t*>(gi.bytes.data());
    for (uint32_t i = 0; i < K; ++i) g[i] = int32_t(i / 64);
    ts[prefix + ".g_idx"] = std::move(gi);
  }
}

// Every text tensor of `d` (d.layers layers) in the gptq (g_idx) or autoround form, plus `vision` vision tensors.
inline Tensors make(const model::MuseGlimmerDesc& d, bool g_idx, uint32_t vision = 3, uint64_t seed = 1) {
  Rng r;
  r.s ^= seed * 0x2545F4914F6CDD1Dull;
  Tensors ts;
  const std::string lm = model::MuseGlimmerDesc::text_prefix();
  add_bf16(r, ts, lm + "embed_tokens.weight", {d.vocab, d.hidden});
  add_bf16(r, ts, lm + "norm.weight", {d.hidden}, 2.0f);
  add_bf16(r, ts, "lm_head.weight", {d.vocab, d.hidden}, 0.1f);
  for (uint32_t l = 0; l < d.layers; ++l) {
    const std::string p = model::MuseGlimmerDesc::layer_prefix(l);
    for (const char* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                          "post_feedforward_layernorm"})
      add_bf16(r, ts, p + n + ".weight", {d.hidden}, 0.3f);
    for (model::MuseLinearId id : d.layer_linears()) {
      const model::MuseLinear ln = d.linear(id);
      for (size_t i = 0; i < ln.parts.size(); ++i) add_int4(r, ts, p + ln.parts[i], ln.shape.K, ln.part_n[i], g_idx);
    }
  }
  for (uint32_t v = 0; v < vision; ++v)
    add_bf16(r, ts, (v % 3 == 0 ? "model.vision_tower.blocks." + std::to_string(v) + ".weight"
                     : v % 3 == 1 ? "model.vision_adapter.fc" + std::to_string(v) + ".bias"
                                  : "model.vision_projection.weight" + std::to_string(v)),
             {64});
  return ts;
}

// A config.json holding what the hand-built-descriptor path reads: the model type and the quantization_config
// of either spelling (`extra` is spliced into the quantization_config object).
inline std::string config(bool gptq, const std::string& extra = "") {
  const std::string q = gptq ? "\"quant_method\":\"gptq\",\"bits\":4,\"group_size\":64,\"sym\":true,\"desc_act\":false"
                             : "\"quant_method\":\"auto-round\",\"packing_format\":\"auto_round:auto_gptq\",\"bits\":4,"
                               "\"group_size\":64,\"sym\":true";
  return "{\"model_type\":\"muse_glimmer\",\"architectures\":[\"MuseGlimmerForConditionalGeneration\"],"
         "\"quantization_config\":{" + q + extra + "}}";
}

inline fs::path tmpdir(const std::string& tag) {
  return fs::temp_directory_path() / ("muse_fixture_" + tag + "_" + std::to_string(::getpid()));
}

}  // namespace muse_fixture
