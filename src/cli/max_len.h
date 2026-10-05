#pragma once
// `--max-len auto|N` and `--mem-reserve-gb G`, shared by b70-serve and b70-decode
// (spec 6 §10). The order is forced by what each step needs:
//
//   1. parse                 before the device: "auto" or a positive integer
//   2. check_before_load     config.json's max_position_embeddings bounds an explicit N
//                            (and the 256 grid, for a CLI that prefills); auto needs it
//   3. load_len              auto loads at a small max_len - the RoPE table is the only
//                            thing load() sizes by it - because the plan needs the
//                            loaded bytes and load() needs a max_len
//   4. settle                after load, before the Engine: auto plans the largest
//                            max_len that fits and re-tables the model
//                            (loader::set_max_len); an explicit N is held to the same
//                            plan and refused with its breakdown instead of failing in
//                            an allocation later
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/memory_plan.h"
#include "runtime/prefill/attn.h"

namespace cli {

struct MaxLenArg {
  bool is_auto = true;
  uint32_t value = 0;   // the explicit N; 0 with is_auto
};

inline MaxLenArg parse_max_len(const std::string& v) {
  if (v == "auto") return {true, 0};
  if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos || v.size() > 10)
    throw std::runtime_error("--max-len expects auto or a positive integer, got '" + v + "'");
  const unsigned long long n = std::stoull(v);
  if (n > 0xFFFFFFFFull) throw std::runtime_error("--max-len is out of range: " + v);
  if (n == 0) throw std::runtime_error("--max-len 0 is not a model length");
  return {false, static_cast<uint32_t>(n)};
}

// Decimal GB (1e9 B, the unit memory_line() prints in), 0 <= G < 1000.
inline size_t parse_reserve_gb(const std::string& v) {
  size_t used = 0;
  double g = -1.0;
  try {
    g = std::stod(v, &used);
  } catch (const std::exception&) {
    used = 0;
  }
  if (used != v.size() || v.empty() || !(g >= 0.0 && g < 1000.0))
    throw std::runtime_error("--mem-reserve-gb expects a number of GB in [0, 1000), got '" + v + "'");
  return static_cast<size_t>(std::llround(g * 1e9));
}

// Step 2. `trained` is loader::trained_context(snapshot) (0 = not declared).
// `require_quantum`: the CLI prefills, and the L0 prefill backends need max_len % 256
// (spec 2.1 §3.1) - refused here rather than at the first request.
inline void check_before_load(const MaxLenArg& a, uint32_t trained, bool require_quantum) {
  if (a.is_auto) {
    if (trained == 0)
      throw std::runtime_error("--max-len auto plans under config.json's max_position_embeddings,"
                               " and this checkpoint declares none: give --max-len N");
    return;
  }
  if (trained != 0 && a.value > trained)
    throw std::runtime_error("--max-len " + std::to_string(a.value) + " exceeds the trained context " +
                             std::to_string(trained) + " (config.json max_position_embeddings)");
  if (require_quantum && a.value % runtime::kMaxLenQuantum != 0)
    throw std::runtime_error("--max-len " + std::to_string(a.value) + " is not a multiple of " +
                             std::to_string(runtime::kMaxLenQuantum) +
                             " (the L0 prefill backends' grid, spec 2.1 §3.1)");
}

// Step 3: the max_len to hand loader::load().
inline uint32_t load_len(const MaxLenArg& a, uint32_t trained) {
  if (!a.is_auto) return a.value;
  return trained != 0 && trained < runtime::kMinAutoMaxLen ? trained : runtime::kMinAutoMaxLen;
}

// What the engine's prefill will allocate (runtime::PrefillPath): `prefill` false for a
// decode-only b70-decode run; the backend the engine will run (set_prefill_backend, else
// the build default); B70_PREFILL_ATTN=composed's max_len-sized S / P scratch.
inline runtime::PrefillPath prefill_path(bool prefill, runtime::PrefillBackend backend) {
  return {prefill, backend, runtime::prefill::attn_mode() == runtime::prefill::AttnMode::Composed};
}

// Decode attention v1 (B70_DECODE_ATTN=v1) bakes MAXLEN into its binaries and only some
// are compiled (src/kernels/CMakeLists.txt): auto then takes the largest compiled length
// at or below what fits. v2, the default, serves any max_len (spec 10).
inline uint32_t v1_compiled_at_most(uint32_t fit, const model::ModelDesc& d) {
  for (uint32_t len : {262144u, 131072u, 65536u, 32768u, 16384u, 4096u}) {
    if (len > fit) continue;
    const std::string v = kernels::attn_decode_variant(1, len, runtime::DecodeScratch::kAttnBlock,
                                                       d.fa_q_heads, d.fa_kv_heads);
    if (std::ifstream(kernels::path(v)).good()) return len;
  }
  return 0;
}

// Step 4. Returns the max_len the Engine is built with; `m` is re-tabled to it for auto.
// Prints the choice and the plan's breakdown to stderr.
inline uint32_t settle(l0::Context& ctx, loader::LoadedModel& m, const MaxLenArg& a,
                       size_t reserve_bytes, const runtime::PrefillPath& path) {
  const model::ModelDesc& d = *m.desc;
  const bool mtp = m.mtp != nullptr;
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const size_t device = ctx.memory_bytes();
  const double gb = 1e9;
  uint32_t len = a.value;
  if (a.is_auto) {
    const uint32_t fit = runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes,
                                                    m.trained_max_len, path);
    if (fit == 0)
      throw std::runtime_error(
          "--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
          " positions fit - " +
          runtime::describe(runtime::plan(d, runtime::kMinAutoMaxLen, mtp, weights, path), device,
                            reserve_bytes));
    len = fit;
    if (runtime::decode_attn() == runtime::DecodeAttn::V1) {
      len = v1_compiled_at_most(fit, d);
      if (len == 0)
        throw std::runtime_error("--max-len auto with B70_DECODE_ATTN=v1: no compiled v1 decode"
                                 " attention at or below " + std::to_string(fit));
    }
    loader::set_max_len(ctx, m, len);
    std::fprintf(stderr,
                 "max_len: auto -> %u (the largest multiple of %u that fits %.3f GB with a %.3f GB"
                 " reserve; trained context %u%s)\n",
                 len, runtime::kMaxLenQuantum, device / gb, reserve_bytes / gb, m.trained_max_len,
                 len != fit ? ", v1 decode attention's largest compiled length" : "");
  } else {
    const runtime::MemoryPlan p = runtime::plan(d, len, mtp, weights, path);
    if (p.total() + reserve_bytes > device) {
      const uint32_t cap = m.trained_max_len != 0 ? m.trained_max_len : len;
      const uint32_t fit =
          cap < runtime::kMaxLenQuantum
              ? 0
              : runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes, cap, path);
      throw std::runtime_error(
          "--max-len " + std::to_string(len) + " does not fit: " +
          runtime::describe(p, device, reserve_bytes) + ". The largest that fits is " +
          std::to_string(fit) + " (--max-len auto); --mem-reserve-gb lowers the reserve");
    }
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  std::fprintf(stderr, "%s\n",
               runtime::describe(runtime::plan(d, len, mtp, weights, path), device, reserve_bytes)
                   .c_str());
  return len;
}

}  // namespace cli
