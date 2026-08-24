#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "common/json.h"
#include "loader/safetensors.h"

namespace loader {

// A mmapped tensor is reinterpret_cast to uint32_t/uint16_t/float; the
// safetensors data section is 8-aligned on every file seen so far, but a loud
// failure beats undefined behaviour if a future checkpoint pads differently.
// Every such cast in the loader goes through this first.
void check_align(const void* p, size_t a, const std::string& name);

struct QuantConfig {
  uint32_t bits = 0, group_size = 0;
  bool sym = false, desc_act = true;
  size_t dynamic_rule_count = 0;
  // Takes the whole config.json value and reads its quantization_config.
  // Throws unless bits==4, group_size==64, sym, !desc_act; throws on any
  // "+:" dynamic rule (the known-broken checkpoint pattern, BENCHMARKS.md).
  static QuantConfig parse(const common::json::Value& config_json);
};

enum class WKind { Int4, Bf16 };

struct LinearSrc {
  WKind kind;
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;  // Int4: [K/8][N]
  const uint16_t* scales = nullptr;   // Int4: [K/64][N] f16
  const uint16_t* weight = nullptr;   // Bf16: [N][K] row-major
  std::string name;
  // Suffix presence decides the kind; labels are never trusted (doc 02).
  static LinearSrc classify(const SafetensorsSet& set, const std::string& prefix);
};

// What the scan counted but did not reject.
struct QuantScan {
  size_t subnormal_scales = 0;   // see assert_quant_invariants
};

// Scans every .qzeros word (must be 0x77777777 - GPTQ v1 stores zero-1, i.e.
// symmetric zero point 8), every .g_idx (identity under g64), and every
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
