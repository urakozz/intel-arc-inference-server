#pragma once
#include <cstdint>
#include <string>
#include <vector>

// The qwen3_5 model as data: what linears exist, how each fused device weight
// is assembled from checkpoint tensors, each one's compiled GEMV shape, and
// every per-layer tensor that is not a GEMV weight. This is the table plan 3's
// capture code walks and the loader executes. It knows nothing of Level Zero,
// kernels, I/O or the device; its one shared header is
// `loader/small_layout.h` (included by `qwen35.cc`, not here), a code-free set
// of block offsets so the table's destination offsets and the loader's packing
// cannot drift apart.
namespace model {

// One GEMV's compiled configuration. layout/S are measured values
// (docs/probe-gemv-2026-08-24.md); the table below is the single place they
// live. Flipping a row to layout 0 is the phase-1 tuning knob recorded in
// spec §6.3 - but it is **not** a one-line knob: the loader implements the
// layout-1 repack only, and `loader::load_linear` throws on any int4 row that
// is not layout 1. Turning the knob means writing the layout-0 loader path
// (GPTQ-native `w[K/8][N]` + a separate `scales[K/64][N]`, which `gemv.cl`'s
// LAYOUT==0 takes as two buffers) and giving `loader::DeviceWeight` a second
// allocation. The guard makes forgetting that loud instead of silent.
// `layout` is meaningful for WeightKind::Int4 only; bf16 weights have a single
// tiled layout (common::repack_bf16_tiled) and carry layout 0 as a filler.
struct GemvShape { uint32_t K, N, S, layout; };

// kCount is the table size, not a linear: it bounds every ordinal lookup.
enum class LinearId { QkvZ, AB, OutProj, GateUp, Down, Qkv, OProj, LmHead, kCount };

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

// What the loader does to a small tensor between the checkpoint and the device.
// The device element is fp32 for every kind but PlainBf16.
enum class SmallBake {
  OnePlusWFp32,   // RMSNorm (Gemma-style): store fp32 (1 + w). fp32 add, stored
                  // fp32 - no cast back, so no rounding the reference lacks.
  PlainBf16,      // store the checkpoint's bf16 word verbatim (RMSNormGated)
  NegExpFp32,     // store fp32 -exp(w): the GDN decay, hoisted out of the step
  RawFp32Widen,   // bf16 source widened to fp32 verbatim (fp32 accumulation)
  RawFp32,        // source already fp32: copied verbatim
};

// Which of the layer's two device allocations a small tensor lands in.
enum class SmallBlock { Norms, Kind };

// One per-layer tensor that is not a GEMV weight - the **single source of
// truth** for the loader (fix I3, 2026-08-25): `loader::load_small` walks this
// table and hardcodes no name, no dtype and no offset.
struct SmallTensor {
  std::string name;      // layer-relative checkpoint name
  uint32_t elems;        // exact element count (the doc-03 shape table)
  const char* dtype;     // required source dtype, as safetensors spells it
  SmallBlock block;      // which allocation
  uint32_t offset;       // byte offset within it - a loader/small_layout.h constant
  SmallBake bake;
};

struct LayerDesc {
  uint32_t index;
  LayerKind kind;
  std::vector<FusedLinear> linears;      // in per-token execution order
  std::vector<SmallTensor> small_tensors;  // norms, conv1d, A_log, dt_bias, ...
};

struct Qwen35 {
  static constexpr uint32_t kLayers = 64, kHidden = 5120, kIntermediate = 17408;
  static constexpr uint32_t kVocab = 248320, kVocabUsed = 248077;
  static constexpr uint32_t kGdnKHeads = 16, kGdnVHeads = 48, kGdnHeadDim = 128;
  static constexpr uint32_t kFaQHeads = 24, kFaKvHeads = 4, kFaHeadDim = 256;
  static constexpr uint32_t kRotaryDim = 64;
  static constexpr double kRopeTheta = 1e7;

  static bool is_fa(uint32_t layer) { return layer % 4 == 3; }
  // The whole table row (LmHead appears in no layer's list; consumers bind it
  // from here). Throws std::out_of_range on kCount or a bad cast.
  static const FusedLinear& linear(LinearId id);
  static const GemvShape& shape(LinearId id);         // the per-shape table
  static std::vector<LayerDesc> layers();             // all 64, fully populated
  // Checkpoint-name helpers ("model.language_model." prefix already stripped
  // by the loader's name mapping): e.g. layer_prefix(5) == "layers.5."
  static std::string layer_prefix(uint32_t layer);
};

}  // namespace model
