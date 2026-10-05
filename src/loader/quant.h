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
  static QuantConfig parse(const common::json::Value& config_json);
};

enum class WKind { Int4, Bf16 };

struct LinearSrc {
  WKind kind;
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;  // Int4: [K/8][N]
  const uint16_t* scales = nullptr;   // Int4: [K/64][N] f16, always g64 (see below)
  // The checkpoint's own group size: 64, or 128 when `scales` points at the
  // g64 expansion held by `expanded_scales` (QuantConfig::parse's comment).
  uint32_t group = 64;
  std::shared_ptr<const std::vector<uint16_t>> expanded_scales;
  const uint16_t* weight = nullptr;   // Bf16: [N][K] row-major
  std::string name;
  // Suffix presence decides the kind; labels are never trusted (doc 02).
  static LinearSrc classify(const SafetensorsSet& set, const std::string& prefix);
};

// What the scan counted but did not reject.
struct QuantScan {
  size_t subnormal_scales = 0;   // see assert_quant_invariants
  // How many `.g_idx` tensors the checkpoint ships (0 on an auto-round
  // `auto_round:auto_gptq` checkpoint, 400 on the published GPTQ one). Counted
  // here because this is the pass that already walks every tensor, and used by
  // `loader::load` to prove an undeclared `desc_act` really is false.
  size_t g_idx_tensors = 0;
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
QuantScan assert_quant_invariants(const SafetensorsSet& set);

}  // namespace loader
