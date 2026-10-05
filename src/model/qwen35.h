#pragma once
#include <cstddef>
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

// One GEMV's compiled configuration. layout/S are measured values; the table
// below is the single place they live. Layout 0 is GPTQ-native
// `w[K/8][N]` plus a separate `scales[K/64][N]` allocation. Layout 1 is the
// subgroup-tiled repack with scales inline. `loader::load_linear` implements
// both and capture checks that the separate scale allocation exists exactly
// for layout 0.
// `layout` is meaningful for WeightKind::Int4 only; bf16 weights have a single
// tiled layout (common::repack_bf16_tiled) and carry layout 0 as a filler.
struct GemvShape { uint32_t K, N, S, layout; };

// kCount is the table size, not a linear: it bounds every ordinal lookup.
enum class LinearId { QkvZ, AB, OutProj, GateUp, Down, Qkv, OProj, LmHead, kCount };
constexpr size_t kLinearCount = static_cast<size_t>(LinearId::kCount);

// How a fused device weight is assembled from checkpoint tensors.
enum class Fuse { Single, Concat, Interleave16 };

// Spec 14 §2: a second set of checkpoint tensors joined onto `parts` at load, on
// packed int4 (loader/fold.h). `N`: each fold part is appended to the matching
// part along the output columns (GPTQ `qweight [K/8][N]`, `scales [K/64][N]`),
// before the Fuse map runs - gate' = [gate | gate_p], up' = [up | up_p]. `K`:
// the single fold part is appended along the packed input rows - down' =
// [down ; down_p] at `qweight` row K/8 and `scales` row K/64 of the first part.
enum class Fold { None, N, K };
// Int8 exists for ONE row: `lm_head` quantised at load (spec 9 §3, int8 rows with an
// fp32 scale per row, gemv_i8w). No checkpoint ships it and no per-layer row uses it.
enum class WeightKind { Int4, Bf16, Int8 };

struct FusedLinear {
  LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // checkpoint prefixes, in order; layer-relative
  uint32_t pad_n = 0;               // zero-pad N to shape.N (a||b: 96 -> 128)
  Fold fold = Fold::None;           // spec 14 §2; None on every Qwen3.8 row
  std::vector<std::string> fold_parts;  // layer-relative, parallel to `parts` (Fold::N)
                                        // or exactly one (Fold::K)
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

// What every supported Qwen3.5-family checkpoint shares (spec 14 §3.1, spec
// 15b): the head dims, the vocabulary, the rotary embedding, the per-layer
// linear order and the small tensors' names. Qwen3.8, Agnes and Ornith all read
// these values from their config.json (`head_dim` 256, `linear_key_head_dim` /
// `linear_value_head_dim` 128, `vocab_size` 248320, `partial_rotary_factor` 0.25,
// `rope_theta` 1e7). Every per-model shape - hidden, q/kv heads, GDN heads,
// layer counts, the FFN - is `model::ModelDesc` (model/model_desc.h), chosen by
// the loader from config.json.
struct Qwen35 {
  // lm_head / embed rows. kVocabUsed is Qwen3.8's tokenizer; the per-model value
  // the argmax masks from is ModelDesc::vocab_used.
  static constexpr uint32_t kVocab = 248320, kVocabUsed = 248077;
  static constexpr uint32_t kGdnHeadDim = 128;   // GDN k and v heads alike
  static constexpr uint32_t kFaHeadDim = 256;    // FA q, k and v heads alike
  static constexpr uint32_t kRotaryDim = 64;
  static constexpr double kRopeTheta = 1e7;

  // The per-token execution order of one layer kind's linears.
  static const std::vector<LinearId>& linear_order(LayerKind kind);
  // Checkpoint-name helpers ("model.language_model." prefix already stripped
  // by the loader's name mapping): e.g. layer_prefix(5) == "layers.5."
  static std::string layer_prefix(uint32_t layer);
};

}  // namespace model
