#pragma once
#include <cstdint>
#include <string>
#include <vector>

// The qwen3_5 model as data: what linears exist, how each fused device weight
// is assembled from checkpoint tensors, and each one's compiled GEMV shape.
// This is the table plan 3's capture code walks. It deliberately knows nothing
// of Level Zero, kernels or the loader - no dependencies, no I/O, no device.
namespace model {

// One GEMV's compiled configuration. layout/S are measured values
// (docs/probe-gemv-2026-08-24.md); the table below is the single place they
// live. Flipping a row to layout 0 is the phase-1 tuning knob recorded in
// spec §6.3 - mechanism supports it, default is all layout 1.
// `layout` is meaningful for WeightKind::Int4 only; bf16 weights have a single
// tiled layout (common::repack_bf16_tiled) and carry layout 0 as a filler.
struct GemvShape { uint32_t K, N, S, layout; };

enum class LinearId { QkvZ, AB, OutProj, GateUp, Down, Qkv, OProj, LmHead };

// How a fused device weight is assembled from checkpoint tensors.
enum class Fuse { Single, Concat, Interleave16 };
enum class WeightKind { Int4, Bf16 };

struct FusedLinear {
  LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // checkpoint prefixes, in order; layer-relative
  uint32_t pad_n = 0;               // zero-pad N to shape.N (a||b: 96 -> 128)
};

enum class LayerKind { GDN, FA };

struct LayerDesc {
  uint32_t index;
  LayerKind kind;
  std::vector<FusedLinear> linears;       // in per-token execution order
  std::vector<std::string> small_tensors; // norms, conv1d, A_log, dt_bias (layer-relative)
};

struct Qwen35 {
  static constexpr uint32_t kLayers = 64, kHidden = 5120, kIntermediate = 17408;
  static constexpr uint32_t kVocab = 248320, kVocabUsed = 248077;
  static constexpr uint32_t kGdnKHeads = 16, kGdnVHeads = 48, kGdnHeadDim = 128;
  static constexpr uint32_t kFaQHeads = 24, kFaKvHeads = 4, kFaHeadDim = 256;
  static constexpr uint32_t kRotaryDim = 64;
  static constexpr double kRopeTheta = 1e7;

  static bool is_fa(uint32_t layer) { return layer % 4 == 3; }
  static const GemvShape& shape(LinearId id);         // the per-shape table
  static std::vector<LayerDesc> layers();             // all 64, fully populated
  // Checkpoint-name helpers ("model.language_model." prefix already stripped
  // by the loader's name mapping): e.g. layer_prefix(5) == "layers.5."
  static std::string layer_prefix(uint32_t layer);
};

}  // namespace model
