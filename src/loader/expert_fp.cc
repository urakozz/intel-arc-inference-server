#include "loader/expert_fp.h"
#include "common/expert_fp_config.h"

#include <bit>
#include <limits>
#include <stdexcept>
#include "loader/snapshot.h"

namespace loader {

using common::ExpertFpFormat;

float ExpertFpMatrix::outer_scale() const {
  return format == ExpertFpFormat::Nvfp4 ? std::bit_cast<float>(words.back()) : 1.f;
}

float ExpertFpMatrix::block_f32(unsigned n, unsigned k) const {
  const unsigned bits = common::expert_fp_code_bits(format);
  const unsigned group = common::expert_fp_group(format);
  const unsigned lane = n % 16, k_in_tile = k % 64, per_word = 32 / bits;
  const auto* tile = words.data() + (size_t(n / 16) * (K / 64) + k / 64) * common::expert_fp_tile_words(format);
  const unsigned code = (tile[(k_in_tile / per_word) * 16 + lane] >> ((k % per_word) * bits)) & ((1u << bits) - 1);
  const unsigned scale = (tile[32 * bits + (k_in_tile / group) * 4 + lane / 4] >> ((lane % 4) * 8)) & 255u;
  return common::expert_fp_block_f32(format, uint8_t(code), uint8_t(scale));
}

ExpertFpCheckpoint::ExpertFpCheckpoint(const std::string& dir)
    : tensors_(dir, true), formats_(common::expert_fp_formats(read_json(dir + "hf_quant_config.json"))) {}

ExpertFpMatrix ExpertFpCheckpoint::load_matrix(const std::string& name, unsigned expert, bool fused) const {
  const auto format = formats_.find(name);
  if (format == formats_.end()) throw std::runtime_error(name + ": missing expert quantization entry");
  const bool nv = format->second == ExpertFpFormat::Nvfp4;
  const unsigned bits = common::expert_fp_code_bits(format->second);
  const unsigned group = common::expert_fp_group(format->second);
  const unsigned tile_words = common::expert_fp_tile_words(format->second);
  const auto tensor = [&](const std::string& key, const char* dtype) -> const TensorInfo& {
    const auto it = tensors_.tensors().find(key);
    if (it == tensors_.tensors().end()) throw std::runtime_error(key + ": missing expert tensor");
    if (it->second.dtype != dtype) throw std::runtime_error(key + ": expected dtype " + dtype);
    return it->second;
  };
  const auto& codes = tensor(name, nv ? "U8" : "F8_E4M3");
  const auto& scales = tensor(name + "_scale", nv ? "F8_E4M3" : "U8");
  if (codes.shape.size() != 3 || codes.shape[0] <= expert || !codes.shape[1] || !codes.shape[2] ||
      codes.shape[1] > std::numeric_limits<unsigned>::max() ||
      codes.shape[2] > std::numeric_limits<unsigned>::max() / (nv ? 2 : 1) ||
      codes.shape[0] > tensors_.bytes(codes) / codes.shape[1] / codes.shape[2] ||
      codes.shape[0] * codes.shape[1] * codes.shape[2] != tensors_.bytes(codes))
    throw std::runtime_error(name + ": invalid expert tensor shape or expert index");
  const auto N = unsigned(codes.shape[1]), row_bytes = unsigned(codes.shape[2]);
  const unsigned K = row_bytes * (nv ? 2 : 1), groups = K / group;
  const size_t rows = codes.shape[0] * N;
  // MXFP8's [32,32] config stores a scale byte for each row's K32 block.
  if (N % (fused ? 32 : 16) || K % 64 ||
      scales.shape != std::vector<uint64_t>{codes.shape[0], N, groups} ||
      tensors_.bytes(scales) != rows * groups)
    throw std::runtime_error(name + ": invalid expert tile or scale shape");
  const auto* c = tensors_.data(codes) + size_t(expert) * N * row_bytes;
  const auto* s = tensors_.data(scales) + size_t(expert) * N * groups;
  ExpertFpMatrix out{format->second, K, N, {}};
  out.words.resize(size_t(N / 16) * (K / 64) * tile_words + unsigned(nv));
  if (nv) {
    const auto& amax = tensor(name + "_amax", "F32");
    const auto& denom = tensor(name + "_outer_scale_denominator", "F32");
    if (amax.shape != std::vector<uint64_t>{codes.shape[0]} || tensors_.bytes(amax) / 4 != codes.shape[0] ||
        tensors_.bytes(amax) % 4 || denom.shape != std::vector<uint64_t>{1} || tensors_.bytes(denom) != 4)
      throw std::runtime_error(name + ": invalid NVFP4 outer scale shape");
    float a, d;
    std::memcpy(&a, tensors_.data(amax) + size_t(expert) * 4, 4);
    std::memcpy(&d, tensors_.data(denom), 4);
    const float scale = a / d;
    if (!(a > 0 && d > 0 && std::isfinite(d) && scale > 0 && std::isfinite(scale)))
      throw std::runtime_error(name + ": invalid NVFP4 amax or outer scale denominator");
    out.words.back() = std::bit_cast<uint32_t>(scale);
  }
  for (unsigned nt = 0; nt < N / 16; ++nt)
    for (unsigned g = 0; g < K / 64; ++g) {
      auto* tile = out.words.data() + (size_t(nt) * (K / 64) + g) * tile_words;
      for (unsigned lane = 0; lane < 16; ++lane) {
        // Source w13 alternates gate/up rows; device order alternates 16-row tiles.
        const unsigned row = fused ? (nt / 2 * 16 + lane) * 2 + nt % 2 : nt * 16 + lane;
        const auto* src = c + size_t(row) * row_bytes + g * bits * 8;
        for (unsigned j = 0; j < 2 * bits; ++j) {
          uint32_t word;
          std::memcpy(&word, src + 4 * j, 4);
          if (!nv && (((word & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u))
            throw std::runtime_error(name + ": E4M3 NaN weight");
          tile[j * 16 + lane] = word;
        }
        for (unsigned b = 0; b < 64 / group; ++b) {
          const uint8_t scale = s[size_t(row) * groups + g * (64 / group) + b];
          if (nv ? scale >= 127 : scale == 255)
            throw std::runtime_error(name + (nv ? "_scale: negative or NaN E4M3 scale" : "_scale: E8M0 NaN scale"));
          tile[32 * bits + b * 4 + lane / 4] |= uint32_t(scale) << ((lane % 4) * 8);
        }
      }
    }
  return out;
}

ExpertFpWeights ExpertFpCheckpoint::load(unsigned layer, unsigned expert) const {
  const std::string prefix = "model.llm.layers." + std::to_string(layer) + ".mlp.experts.";
  auto gu = load_matrix(prefix + "w13_weight", expert, true);
  auto dn = load_matrix(prefix + "w2_weight", expert, false);
  if (gu.format != dn.format || gu.K != dn.N || gu.N / 2 != dn.K)
    throw std::runtime_error(prefix + "gate/up and down have incompatible expert formats or shapes");
  return {std::move(gu), std::move(dn)};
}

}  // namespace loader
