#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/kernels.h"

// Spec 5's int8 prefill linears, the h8 path (docs/specs/2026-09-24-spec5-int8-
// prefill-linears-design.md, stages T1-T2): x' = x R_K per token in int8,
// W' = per-channel int8 of (W R_K) built per 1024-column slab, i8 x i8 GEMM,
// y = acc * xs[m] * ws[n]. The kernels are src/kernels/prefill/pf_int8.cl.
namespace runtime::prefill {

// Everything the int8 path owns beyond PrefillScratch: per-K signs, per-linear
// column scales (built on first use of a weight, one GPU pass), int8 scratch.
class Int8State {
 public:
  explicit Int8State(l0::Context& ctx);
  // ws/inv for this weight, computed on first call (pf_colmax_rot + host
  // finish), cached by the weight's device address. The first call waits on
  // cx, so it must run outside any recorded chunk (plan 5b calls it for every
  // linear before the first one).
  const std::pair<l0::Mem, l0::Mem>& scales(Context& cx, KernelCache& kc,
                                            const loader::DeviceWeight& w);
  const l0::Mem& signs_f32(uint32_t K);    // [K] +-1
  const l0::Mem& sign_bits(uint32_t K);    // [K/32]
  l0::Mem& xq();                           // int8 [kC][17408]
  l0::Mem& xs();                           // fp32 [kC]
  l0::Mem& w8_slab();                      // u32 [17408/4][1024]
  size_t bytes() const;

  static constexpr uint32_t kMaxK = 17408;

 private:
  l0::Context& ctx_;
  l0::CmdList imm_;
  l0::Mem xq_, xs_, w8_;
  std::map<uint32_t, l0::Mem> signs_, bits_;
  std::map<const void*, std::pair<l0::Mem, l0::Mem>> scales_;
};

// linear_l0's contract (fp32 partials [pad256(M)][N]) on the h8 path.
void linear_i8(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
               const loader::DeviceWeight& w, const uint16_t* x, uint32_t M);
// linear_l0_silu's contract (bf16 out [pad256(M)][N/2] at pitch ldx, partials untouched).
void linear_i8_silu(Context& cx, KernelCache& kc, PrefillScratch& s, Int8State& q,
                    const loader::DeviceWeight& w, const uint16_t* x, uint32_t M,
                    uint16_t* out, uint32_t ldx);
size_t linear_i8_launches(const model::GemvShape& sh);   // 1 + 2 * N / 1024

}  // namespace runtime::prefill
