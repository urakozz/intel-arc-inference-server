// tp_fold_send.h - PERF-44 on Qwen3.8 tp: tp.cl's folded tp_send as the tail of the row-parallel linear that feeds it
// (gemv.cl and gemm_sm.cl under TP_FOLD_SEND; out_proj / o_proj / down at the card's S). The landing's chunks stay
// tp_send's (FOLD_G of them a row, K / FOLD_G columns each), so tp_res_fold and the call counters are unchanged and a
// list may mix this form with the plain linear + tp_send. A chunk's columns are written by CHUNK / 64 x S work-groups;
// each counts itself in on the chunk's arrival counter (ctr[J_CHUNKS + g], after the call counters, monotonic) and the
// last one in sends the chunk as tp_send's work-group (g, m) does for every row m: the slices summed from 0 in slice
// order into `own` and the peer's landing, the stamp lines, the flags. The same words in the same order: bitwise
// tp_send. The includer defines M, N (the hidden width), S, WG_COLS (a work-group's columns), TP_FOLD_G, and calls
// tp_fold_send after its own stores to `out`, with every work-item and TPF_LOCAL words of local memory.
#ifndef B70_TP_FOLD_SEND_H
#define B70_TP_FOLD_SEND_H
#define TP_MROWS 8
#if M > TP_MROWS
#error "TP_FOLD_SEND: M <= 8 (the landing's rows)"
#endif
#define TPF_CHUNK (N / TP_FOLD_G)
#if N % TP_FOLD_G != 0 || TPF_CHUNK % WG_COLS != 0
#error "TP_FOLD_SEND: whole work-groups under each chunk"
#endif
#define TPF_WGS (TPF_CHUNK / WG_COLS * S)   // the work-groups that write a chunk's columns
#define DATA_WORDS (TP_MROWS * N)            // the folded form: one row a call row
#define J_CHUNKS (TP_FOLD_G * TP_MROWS)
#define TPF_LOCAL (TP_MROWS + 1)
#include "tp_wire.h"

inline void tp_fold_send(__global float* out, __global uint* land, __global uint* ctr, __global float* own,
                         __local uint* sh) {
  const uint lid = get_local_id(0), wg = get_local_size(0);
  const uint g = get_group_id(0) / (TPF_CHUNK / WG_COLS);
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    const uint a = atomic_fetch_add_explicit((volatile __global atomic_uint*)(ctr + J_CHUNKS + g), 1u,
                                             memory_order_acq_rel, memory_scope_device);
    const uint last = a % TPF_WGS == TPF_WGS - 1u;
    sh[TP_MROWS] = last;
    if (last)
      for (uint m = 0; m < M; ++m) sh[m] = tp_next(ctr, g, m);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (sh[TP_MROWS] == 0u) return;   // uniform in the work-group
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
  const uint k0 = g * TPF_CHUNK;
  for (uint m = 0; m < M; ++m) {
    __global uint* d = land + (sh[m] & 1u) * DATA_WORDS;
    for (uint k = k0 + lid; k < k0 + TPF_CHUNK; k += wg) {
      float acc = 0.f;
      for (uint s = 0; s < S; ++s)
        acc += as_float(atomic_load_explicit((volatile __global atomic_uint*)(out + ((size_t)s * M + m) * N + k),
                                             memory_order_relaxed, memory_scope_device));
      own[(size_t)m * N + k] = acc;
      d[(size_t)m * N + k] = as_uint(acc);
    }
  }
  for (uint i = lid; i < M * TP_TRAILER_STRIDE; i += wg) {   // the stamps as whole lines, like the data's
    const uint m = i / TP_TRAILER_STRIDE;
    land[TRAILER(sh[m] & 1u, g * TP_MROWS + m) + i % TP_TRAILER_STRIDE] = sh[m];
  }
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    for (uint m = 0; m < M; ++m) *(volatile __global uint*)(land + FLAG(sh[m] & 1u, g * TP_MROWS + m)) = sh[m];
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  }
}
#endif
