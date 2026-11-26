#pragma once
#include <cstdint>

#include "cpu/expert_int4.h"   // Pool

// PERF-124 step 4: one Qwen3.8-Flash-Next routed expert computed on the CPU straight from the blocks the card reads
// (loader/qwen4exp_layout.h, layout 1: per 16-column n-tile and k-group 128 u32 of nibbles - word j of lane l at
// [j x 16 + l], k = 64 g + 8 j + i in bits 4 i - then the 16 lanes' f16 scales), so a host-RAM expert needs no second
// copy. Bit for bit q4_moe.cl / q4_moe_union.cl at hidden 2560, inter 640, UP_KS 4, DN_KS 2 (and at a tensor-parallel
// card's half expert: inter 320, DN_KS 1, its binaries' MOE_TP_PART partial): per word the 8 products
// summed from 0 in k order (each product exact in fp32: a 4-bit code times a bf16), the group's 8 word sums added, times
// the group's scale, added in group order within a K slice; the slices by the kernels' pairwise tree; gate and up
// rounded to bf16, h = bf16(bf16(silu(g)) x u) with the card's exp. out[r] is the fp32 down sum the card's down holds
// for that (row, slot) before its epilogue (the combine weights and rounds it). Built with -ffp-contract=off.
namespace cpu {

inline constexpr uint32_t kQ4Hidden = 2560, kQ4Inter = 640, kQ4TpInter = 320;

// gu / dn: the expert's gate||up and down blocks; x[r]: row r's bf16 [2560]; out[r]: fp32 [2560]; n rows, 1..8.
// pool null: the calling thread alone. isa: the AVX2 or the AVX-512 kernel (the same bits). inter: kQ4Inter, or
// kQ4TpInter for a tp card's half (out[r] is then that card's partial, before the two cards' sum).
void q4_expert(const uint32_t* gu, const uint32_t* dn, const uint16_t* const* x, uint32_t n, float* const* out,
               Pool* pool, Isa isa = best_isa(), uint32_t inter = kQ4Inter);

// Several experts in one pool run a phase (gate||up of all, then the down of all): the same bits as q4_expert each.
// Under tp the host thread hands both cards' half experts of a layer here at once (fewer pool runs, every core busy).
struct Q4Job {
  const uint32_t* gu = nullptr;
  const uint32_t* dn = nullptr;
  const uint16_t* x[kMaxRows] = {};
  float* out[kMaxRows] = {};
  uint32_t n = 0;
};
void q4_experts(const Q4Job* jobs, uint32_t count, Pool* pool, Isa isa = best_isa(), uint32_t inter = kQ4Inter);

}  // namespace cpu
