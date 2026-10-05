#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "common/json.h"
#include "loader/safetensors.h"

namespace loader {

// A mmapped tensor is reinterpret_cast to uint32_t/uint16_t/float; the
// safetensors data section is 8-aligned on every file seen so far, but a loud
// failure beats undefined behaviour if a future checkpoint pads differently.
// Every such cast in the loader goes through this first.
void check_align(const void* p, size_t a, const std::string& name);

// **Two spellings of the same quantisation, and the loader accepts both.**
// Measured against the two checkpoints this project runs (2026-08-26):
//
//   Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ  (published, bf16 lm_head)
//     quant_method "gptq", provider "auto-round", desc_act false,
//     `dynamic` = 98 regex rules, all "-:", 400 `g_idx` tensors shipped.
//   qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64      (ours, int4 lm_head)
//     quant_method "auto-round", packing_format "auto_round:auto_gptq",
//     **no `desc_act` key at all**, **no `dynamic` key** - the exclusions are
//     `extra_config`, a per-module object - and **no `g_idx` tensor anywhere**.
//
// Nothing here is trusted to decide what a tensor *is*: that is
// `LinearSrc::classify` on the shipped suffixes (docs/02's rule). What this
// parse does is refuse a checkpoint whose *arithmetic* differs from the one the
// kernels implement - a different group size, an asymmetric zero point, an
// activation-order permutation - and it has to read two vocabularies to do it.
struct QuantConfig {
  uint32_t bits = 0, group_size = 0;
  bool sym = false, desc_act = true;
  // `desc_act` absent is auto-round's spelling of false, and this records that
  // it was inferred rather than read. `loader::load` then proves it from the
  // bytes: an undeclared desc_act with any `g_idx` tensor in the checkpoint is
  // a throw, because `g_idx` is the only mechanism an activation-order
  // permutation has (docs/13, "The three data asserts").
  bool desc_act_declared = false;
  std::string quant_method, packing_format;
  size_t dynamic_rule_count = 0;   // GPTQ `dynamic`: "-:" exclusion regexes
  // auto-round `extra_config`: one object per module, `bits: 16` meaning "left
  // in fp", `bits: 4` meaning "quantised, and here are its parameters".
  size_t extra_excluded = 0, extra_quantised = 0;
  // Takes the whole config.json value and reads its quantization_config.
  // Throws unless bits==4, group_size is 64 or 128, sym, and desc_act is false
  // or absent; throws on any "+:" dynamic rule (the known-broken checkpoint
  // pattern, BENCHMARKS.md); throws on an `extra_config` module that would be
  // quantised at anything but g64 / g128 sym int4, and on an unknown
  // `quant_method` or `packing_format`.
  //
  // **g128 runs on the g64 kernels.** `LinearSrc::classify` expands a g128
  // linear's scales at load: g64 group 2j and 2j+1 both take g128 group j's
  // scale. The dequant is scale * (q - 8) per weight, so the expanded tensor
  // dequantises to exactly the g128 checkpoint's weights; it costs g64's bytes
  // (4.25 bits, not 4.125). A dedicated g128 kernel is a recorded future idea
  // (docs/13-loader.md, "g128 checkpoints").
  //
  // **The third spelling: llm-compressor's compressed-tensors** (spec 20 §9, docs/13
  // "compressed-tensors symmetric checkpoints"). `quant_method: compressed-tensors`,
  // `format: pack-quantized`, and per `config_groups.*.weights` {num_bits 4, type int,
  // symmetric true, strategy group, group_size 64 | 128, actorder}. Accepted only when
  // every group is exactly that; refused, naming the field and its value: asymmetric
  // (`symmetric: false`), any other num_bits / type / strategy / group_size, dynamic
  // weights, quantised activations (W4A8 / W8A8), any `format` but pack-quantized, a
  // `quantization_status` other than compressed, a non-empty `sparsity_config` or
  // `transform_config` (rotations change the weights' basis). `actorder` null / false /
  // "weight" / "static" keeps the groups contiguous in the stored column order (the
  // identity k / group the kernels compute); "group" / "dynamic" / true is accepted only
  // as far as the bytes prove it the identity: every linear must ship a `weight_g_idx`
  // and each must be k / group (assert_quant_invariants, check_quant_scan).
  static QuantConfig parse(const common::json::Value& config_json);

  // compressed-tensors only (all false / empty / 0 otherwise).
  bool compressed_tensors = false;
  std::string ct_actorder;          // the value as written ("" for null / false)
  bool ct_actorder_group = false;   // "group" / "dynamic" / true: g_idx must prove identity
  size_t ct_config_groups = 0;      // config_groups entries (each one checked)
  size_t ct_ignore = 0;             // `ignore` entries: modules left in bf16 (`.weight`)
  // `kv_cache_scheme` is set: the checkpoint ships `self_attn.{k,v}_scale` for an fp8 KV
  // cache. This engine's KV cache is bf16 or its own int8, so they are dropped by name
  // and counted (loader.cc), like qzeros.
  bool ct_kv_cache_scheme = false;
};

enum class WKind { Int4, Bf16 };

// Where an Int4 linear's bytes came from. Both end in the same GPTQ-layout pair below;
// the packing only says which tensors were read (`suffixes`) and whether the qweight
// is the mmap or a converted copy.
enum class Int4Packing { Gptq, CompressedTensors };

struct LinearSrc {
  WKind kind;
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;  // Int4: [K/8][N]
  const uint16_t* scales = nullptr;   // Int4: [K/64][N] f16, always g64 (see below)
  // The checkpoint's own group size: 64, or 128 when `scales` points at the
  // g64 expansion held by `expanded_scales` (QuantConfig::parse's comment).
  uint32_t group = 64;
  std::shared_ptr<const std::vector<uint16_t>> expanded_scales;
  // compressed-tensors: `qweight` points into this - weight_packed [N][K/8] transposed
  // to [K/8][N]. The words are copied unchanged: both formats hold k = 8r..8r+7 in one
  // int32, k = 8r in the low nibble, each as q + 8 (symmetric). Likewise `scales`
  // points into `expanded_scales` - weight_scale [N][K/g] transposed to [K/g][N], f16
  // (bf16 converted exactly or refused), then g128 -> g64 as above. Shared, so a copy
  // of the LinearSrc keeps the pointers valid.
  std::shared_ptr<const std::vector<uint32_t>> owned_qweight;
  Int4Packing packing = Int4Packing::Gptq;
  bool ct_g_idx = false;              // compressed-tensors: a weight_g_idx was read (identity)
  const uint16_t* weight = nullptr;   // Bf16: [N][K] row-major
  std::string name;
  // Suffix presence decides the kind; labels are never trusted (doc 02). Int4 is
  // `.qweight` + `.scales` (GPTQ v1) or `.weight_packed` + `.weight_scale` +
  // `.weight_shape` (compressed-tensors pack-quantized, symmetric; converted here -
  // the ONE conversion point every loader path goes through: dense, Agnes's fold,
  // the MoE experts, the MTP head's MoE layer, K2); Bf16 is `.weight`.
  static LinearSrc classify(const SafetensorsSet& set, const std::string& prefix);
  // The kind alone, from the same suffixes, without reading or converting a byte.
  static WKind kind_of(const SafetensorsSet& set, const std::string& prefix);
  // The checkpoint tensors (suffixes of `name`) this linear was read from - what a
  // loader marks consumed: {.qweight, .scales}, {.weight_packed, .weight_scale,
  // .weight_shape[, .weight_g_idx]} or {.weight}.
  std::vector<std::string> suffixes() const;
  // One of these after a linear's prefix means "the checkpoint has this linear" in
  // some form (the MoE bindings' `has`).
  static const std::vector<std::string>& marker_suffixes();
};

// What the scan counted but did not reject.
struct QuantScan {
  size_t subnormal_scales = 0;   // see assert_quant_invariants
  // How many `.g_idx` tensors the checkpoint ships (0 on an auto-round
  // `auto_round:auto_gptq` checkpoint, 400 on the published GPTQ one). Counted
  // here because this is the pass that already walks every tensor, and used by
  // `loader::load` to prove an undeclared `desc_act` really is false.
  size_t g_idx_tensors = 0;
  // compressed-tensors (pack-quantized) linears, counted by their `.weight_packed`;
  // their `.weight_g_idx` tensors (each proved the identity); which group sizes their
  // scales declare (bit 0: g64, bit 1: g128); whether any scale tensor is bf16 (each
  // value proved exactly representable in f16). `.qweight` tensors are counted too:
  // check_quant_scan refuses a checkpoint whose tensors disagree with its config.
  size_t ct_linears = 0, ct_g_idx_tensors = 0, gptq_linears = 0;
  uint32_t ct_group_mask = 0;
  bool ct_bf16_scales = false;
};

// Scans every .qzeros word (must be 0x77777777 - GPTQ v1 stores zero-1, i.e.
// symmetric zero point 8), every .g_idx (the identity k / group, the group
// read from the module's own .scales shape: 64, or 128; 64 when the module
// ships no .scales), and every
// .scales f16 (must be finite - the dequant is scale*(q-8) with no guard, so a
// NaN/Inf scale poisons a whole group of 64). Throws with tensor name, element
// index and value on the first violation.
//
// Subnormal f16 scales are **counted, not rejected**: this checkpoint really
// has them (measured 2026-08-25). The device reads the f16 word natively and
// common::f16_to_f32 decodes subnormals exactly, so they are reported rather
// than treated as corruption.
//
// compressed-tensors tensors are held to the same rules: a `.weight_zero_point`
// anywhere is a throw (asymmetric - spec 20 §9 refuses it); every `.weight_g_idx` must
// be the identity k / group, the group read from the module's `.weight_scale` columns;
// every `.weight_scale` value must be finite and, if bf16, exactly representable in f16
// (the device reads f16 scales); the group must be 64 or 128. Subnormal-in-f16 values
// are counted as above.
QuantScan assert_quant_invariants(const SafetensorsSet& set);

// The config against what the scan found, before a byte is repacked: a
// compressed-tensors config over `.qweight` tensors (or a GPTQ / auto-round config over
// `.weight_packed` ones) is refused - the label and the bytes disagree, and the reader
// would be guessing which to believe; and `actorder` "group" needs every
// compressed-tensors linear to ship its (identity) `weight_g_idx`, since a missing one
// is a permutation the checkpoint did not write down.
void check_quant_scan(const QuantConfig& qc, const QuantScan& scan);

// The ONE startup note a compressed-tensors load prints (stderr, once per load) and
// keeps in its report (operator requirement, 2026-10-05): what was converted, the group
// size found, how many linears - and that the recommended format is our own AutoRound
// g64. Empty for a checkpoint with no compressed-tensors linear.
std::string ct_conversion_note(const QuantScan& scan);

}  // namespace loader
