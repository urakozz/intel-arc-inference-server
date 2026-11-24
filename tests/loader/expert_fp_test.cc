#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "check.h"
#include "loader/expert_fp.h"
#include "loader/safetensors_writer.h"

namespace {

float e4m3(unsigned code) {
  const int exp = (code >> 3) & 15;
  const int frac = code & 7;
  return std::copysign(std::ldexp(float(exp ? 8 + frac : frac), exp ? exp - 10 : -9),
                       code & 128 ? -1.f : 1.f);
}

void compare(const loader::ExpertFpMatrix& m, const loader::SafetensorsSet& set,
             const std::string& name, unsigned expert, bool nv, bool fused, float outer) {
  const auto* codes = set.data(set.tensors().at(name));
  const auto* scales = set.data(set.tensors().at(name + "_scale"));
  const unsigned group = nv ? 16 : 32;
  const unsigned row_bytes = m.K / (nv ? 2 : 1);
  for (unsigned n : {0u, 15u, 16u, 31u, m.N - 1}) {
    const unsigned row = fused ? 2 * ((n / 32) * 16 + n % 16) + (n / 16) % 2 : n;
    const size_t src = size_t(expert) * m.N + row;
    for (unsigned k = 0; k < m.K; ++k) {
      const unsigned scale = scales[src * (m.K / group) + k / group];
      float weight;
      if (nv) {
        const unsigned code = (codes[src * row_bytes + k / 2] >> (4 * (k % 2))) & 15;
        constexpr float magnitudes[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        weight = std::copysign(magnitudes[code & 7], code & 8 ? -1.f : 1.f) * e4m3(scale);
      } else {
        weight = std::ldexp(e4m3(codes[src * row_bytes + k]), int(scale) - 127);
      }
      CHECK_EQ(std::bit_cast<uint32_t>(m.block_f32(n, k)), std::bit_cast<uint32_t>(weight));
      CHECK_EQ(m.block_bf16(n, k), common::f32_to_bf16(weight));
      CHECK_EQ(std::bit_cast<uint32_t>(m.f32(n, k)), std::bit_cast<uint32_t>(weight * outer));
      CHECK_EQ(m.bf16(n, k), common::f32_to_bf16(weight * outer));
    }
  }
}

void check_layer(const std::string& dir, unsigned layer, unsigned expert, bool nv) {
  const loader::ExpertFpCheckpoint checkpoint(dir);
  const loader::SafetensorsSet set(dir, true);
  const auto weights = checkpoint.load(layer, expert);
  const std::string prefix = "model.llm.layers." + std::to_string(layer) + ".mlp.experts.";
  for (bool fused : {true, false}) {
    const auto name = prefix + (fused ? "w13_weight" : "w2_weight");
    float outer = 1.f;
    if (nv) {
      float amax, denom;
      std::memcpy(&amax, set.data(set.tensors().at(name + "_amax")) + 4 * expert, 4);
      std::memcpy(&denom, set.data(set.tensors().at(name + "_outer_scale_denominator")), 4);
      outer = amax / denom;
    }
    const auto& matrix = fused ? weights.gate_up : weights.down;
    CHECK_EQ(matrix.format, nv ? common::ExpertFpFormat::Nvfp4 : common::ExpertFpFormat::Mxfp8);
    compare(matrix, set, name, expert, nv, fused, outer);
  }
}

void synthetic() {
  namespace fs = std::filesystem;
  const auto dir = fs::temp_directory_path() / ("b70_expert_fp." + std::to_string(getpid()));
  safetensors_writer::Tensors ts;
  for (unsigned l = 0; l < 2; ++l) {
    const bool nv = l == 0;
    for (bool fused : {true, false}) {
      const unsigned n = fused ? 128 : 64, k = 64;
      const unsigned row_bytes = nv ? k / 2 : k, groups = nv ? k / 16 : k / 32;
      const std::string name = "model.llm.layers." + std::to_string(l) + ".mlp.experts." +
                               (fused ? "w13_weight" : "w2_weight");
      std::vector<uint8_t> codes(2 * n * row_bytes), scales(2 * n * groups);
      for (size_t i = 0; i < codes.size(); ++i) {
        unsigned code = (i + i / row_bytes * 17) % 256;
        codes[i] = !nv && (code & 127) == 127 ? 126 : code;
      }
      for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = nv ? i % 127 : 120 + i % 16;
      // Include every FP4 code, E4M3 subnormals, signed zero and the largest finite E4M3.
      if (nv)
        for (unsigned i = 0; i < 8; ++i)
          codes[n * row_bytes + i] = uint8_t((2 * i) | ((2 * i + 1) << 4));
      ts[name] = {nv ? "U8" : "F8_E4M3", {2, n, row_bytes}, std::move(codes)};
      ts[name + "_scale"] = {nv ? "F8_E4M3" : "U8", {2, n, groups}, std::move(scales)};
      if (nv) {
        const float amax[] = {1.f, fused ? 2.f : 5.f}, denom = 3.f;
        const auto* a = reinterpret_cast<const uint8_t*>(amax);
        const auto* d = reinterpret_cast<const uint8_t*>(&denom);
        ts[name + "_amax"] = {"F32", {2}, {a, a + sizeof(amax)}};
        ts[name + "_outer_scale_denominator"] = {"F32", {1}, {d, d + sizeof(denom)}};
      }
    }
  }
  safetensors_writer::write_checkpoint(dir, ts);
  std::ofstream(dir / "hf_quant_config.json") << R"({"quant_method":"inkling2","model":{"parameter_groups":[
    {"weight_transform":"nvfp4_nearest","weight_block_shape":[1,16],"weight_scale_format":"amax_denominator",
     "tensors":["model.llm.layers.0.mlp.experts.w13_weight","model.llm.layers.0.mlp.experts.w2_weight"]},
    {"weight_transform":"mxfp8_nearest","weight_block_shape":[32,32],
     "tensors":["model.llm.layers.1.mlp.experts.w13_weight","model.llm.layers.1.mlp.experts.w2_weight"]}]}})";
  const auto path = dir.string() + "/";
  check_layer(path, 0, 1, true);
  check_layer(path, 1, 1, false);
  fs::remove_all(dir);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    synthetic();
    std::puts("Hand-built NVFP4 and MXFP8 blocks match in FP32 and BF16");
  } else {
    const auto dir = std::string(argv[1]) + "/";
    check_layer(dir, 0, 0, true);
    check_layer(dir, 19, 127, true);
    check_layer(dir, 20, 0, false);
    check_layer(dir, 23, 127, false);
    std::puts("Sampled release weights match in FP32 and BF16");
  }
}
