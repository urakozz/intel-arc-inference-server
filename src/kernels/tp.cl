// tp.cl - spec 17b's tensor-parallel decode kernels (`--parallel tp`): the all-reduce after each row-parallel linear
// and the vocabulary half's argmax. A new source (every existing binary keeps its bytes); one family per define:
//
//   TP_CROSS   tp_send       right after out_proj / o_proj / down: this card's part of the linear's output into the
//                            PEER's landing buffer (a P2P push), with a trailer stamp and a flag a chunk;
//              tp_res_fold   in place of the next prep_res_fold: waits (bounded) for the peer's chunk in this card's
//                            OWN landing buffer, then folds both cards' parts into the residual exactly as
//                            prep_res_fold folds one card's split-K slices, and reduces the chunk's sum of squares.
//   TP_ARGMAX  tp_argmax_stage1 / tp_argmax_stage2   argmax.cl's two stages over this card's vocabulary half
//                            (VOCAB rows, the first VOCAB_USED real), the index offset by ID_BASE (the half's first
//                            id); stage 2 also writes the winner's fp32 bits to ctrl[CTRL_OUTV + m] so the host can
//                            merge the two halves (runtime::TpEngine: larger value, ties to the lower id).
//
// What crosses (model::TpCrossing, FOLDED):
//   FOLDED 0 (slices)  the card's S_LOC raw split-K slices [S_LOC][M][K] (the linear ran at the one-card S / 2 over the
//                      one-card slice boundaries); the fold sums slice 0 .. 2 S_LOC - 1 in one-card order (card 0's
//                      slices first) - prep_res_fold's chain over the one-card S, BITWISE one card.
//   FOLDED 1 (folded)  one fp32 row [M][K]: the card's S_LOC slices summed in slice order (kept in `own` too); the fold
//                      adds card 0's row then card 1's - bitwise between the cards, another order than one card's.
//
// The protocol (runtime/tp_protocol.h is the host mirror, line for line; tests/runtime/tp_protocol_test.cc runs it on
// threads). Spec 17a's route (P), the only one that works on this box: every
// crossing lands in the READER's own VRAM, written by the peer with plain stores, published by a plain flag store
// after a system-scope release fence; the reader polls its own memory with system-scope acquire loads and reads the
// data with system-scope loads. Two slots a direction (call n uses slot n & 1): a writer at call n + 2 has consumed the
// reader's call n + 1, which the reader sent after finishing call n. Layout of one landing buffer, in 32-bit words:
//
//   [slot 0 data | slot 1 data | trailers [2][J] x 16 words | pad to 4 KiB | flags [2][J] x 16 words | pad to 4 KiB]
//   data = S_MSG x TP_MROWS x K words (capacity at 8 rows; a call at M rows uses [S_MSG][M][K]), J = FOLD_G x TP_MROWS
//
// Every 64-byte line of the landing has one writing work-group, and the stamp goes out as a whole line (16 lanes, n in
// every word), as the data's lines do. Trailers packed 16 to a line, one 4-byte store each (TP_TRAILER_STRIDE 1, kept
// for tp_kernels_test stress), reached the reader after their flag: under DFlash serving the reader saw flag n with
// trailer n - 2 and read n on its next load (tp_late), once a 78-request opencode replay (2026-11-14).
//
// Chunk j = g x TP_MROWS + m: work-group (g, m) of either kernel - the K range [g CHUNK, + CHUNK) of row m, every slice.
// Call numbers come from per-chunk counters in device memory advanced by the kernels (send's and fold's own), so a
// replayed list with frozen arguments progresses; work-group (g, 0) also advances the counters of rows M .. 7, so lists
// of different M (17d's verify) keep every chunk's count equal. Bounded: lane 0 spins at most `spin_limit` loads; a
// failure (timeout 1, stale trailer 2) is recorded once in `state` (shared memory, read by the host after the fence) and
// every later wait on the card skips its spin - a lost peer costs one bound, not 128. The list finishes on garbage; the
// host throws and the engine stays failed until reset().
//
// state: [0] first failure's status, [1] flag read, [2] trailer read, [3] call expected, [4] chunk, [5] failed chunks,
// [6] a stale trailer's late arrival: 1 + the loads until it read the call after all, ~0u if never (tp_late).
#define WG 256
#define TP_SLOTS 2u
#define TP_FLAG_STRIDE 16u
#define TP_PAGE 1024u
#ifndef TP_TRAILER_STRIDE
#define TP_TRAILER_STRIDE 16u
#endif

#define SYS_LOAD(p, order) \
  atomic_load_explicit((volatile __global atomic_uint*)(p), order, memory_scope_all_svm_devices)

#ifndef TP_CROSS
#define TP_CROSS 0
#endif
#ifndef TP_ARGMAX
#define TP_ARGMAX 0
#endif
#if TP_CROSS + TP_ARGMAX != 1
#error "tp.cl: define exactly one of TP_CROSS, TP_ARGMAX (src/kernels/CMakeLists.txt)"
#endif

#if TP_CROSS
#if !defined(M) || !defined(K) || !defined(FOLD_G) || !defined(S_LOC) || !defined(FOLDED) || !defined(RANK)
#error "tp.cl TP_CROSS: M, K, FOLD_G, S_LOC, FOLDED and RANK are the variant's"
#endif
#ifndef TP_MROWS
#define TP_MROWS 8
#endif
#if M > TP_MROWS || RANK > 1
#error "tp.cl: M <= TP_MROWS, RANK 0 or 1"
#endif
#if FOLDED
#define S_MSG 1
#else
#define S_MSG S_LOC
#endif
#define CHUNK ((K + FOLD_G - 1) / FOLD_G)
#define DATA_WORDS (S_MSG * TP_MROWS * K)
#define J_CHUNKS (FOLD_G * TP_MROWS)
#define TRAILER(slot, j) (TP_SLOTS * DATA_WORDS + ((slot) * J_CHUNKS + (j)) * TP_TRAILER_STRIDE)
#define FLAG_BASE (((TP_SLOTS * DATA_WORDS + TP_SLOTS * J_CHUNKS * TP_TRAILER_STRIDE) + TP_PAGE - 1) / TP_PAGE * TP_PAGE)
#define FLAG(slot, j) (FLAG_BASE + ((slot) * J_CHUNKS + (j)) * TP_FLAG_STRIDE)

// prep.cl's two roundings, token for token (the fold's arithmetic is prep_res_fold's).
#include "bf16.h"

inline void tp_record(__global uint* state, uint status, uint got, uint stamp, uint want, uint j) {
  atomic_inc((volatile __global uint*)(state + 5));
  if (atomic_cmpxchg((volatile __global uint*)state, 0u, status) == 0u) {
    state[1] = got;
    state[2] = stamp;
    state[3] = want;
    state[4] = j;
  }
}

// After a stale trailer: the loads until it reads `want` after all (a late stamp), ~0u if never within the bound (a
// lost one), in state[6] for the first stale chunk.
inline void tp_late(__global uint* state, __global const uint* trailer, uint want, uint spin_limit) {
  uint i = 0u;
  while (SYS_LOAD(trailer, memory_order_acquire) != want && ++i < spin_limit) {
  }
  atomic_cmpxchg((volatile __global uint*)(state + 6), 0u, i < spin_limit ? i + 1u : ~0u);
}

// This chunk's next call number, and the counters of the rows this list does not run kept level with it.
inline uint tp_next(__global uint* ctr, uint g, uint m) {
  const uint j = g * TP_MROWS + m;
  const uint n = ctr[j] + 1u;
  ctr[j] = n;
  if (m == 0u)
    for (uint r = M; r < TP_MROWS; ++r) ctr[g * TP_MROWS + r] = n;
  return n;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tp_send(__global const float* restrict partials, __global uint* land, __global uint* ctr,
             __global float* restrict own) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint j = g * TP_MROWS + m;
  __local uint s_n;
  if (lid == 0) s_n = tp_next(ctr, g, m);
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint n = s_n;
  const uint slot = n & 1u;
  __global uint* d = land + slot * DATA_WORDS;
  const uint k0 = g * CHUNK;
  uint k1 = k0 + CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG) {
#if FOLDED
    float acc = 0.f;
    for (uint s = 0; s < S_LOC; ++s) acc += partials[((size_t)s * M + m) * K + k];
    own[(size_t)m * K + k] = acc;
    d[(size_t)m * K + k] = as_uint(acc);
#else
    for (uint s = 0; s < S_LOC; ++s) d[((size_t)s * M + m) * K + k] = as_uint(partials[((size_t)s * M + m) * K + k]);
#endif
  }
  if (lid < TP_TRAILER_STRIDE) land[TRAILER(slot, j) + lid] = n;   // the stamp as a whole line, like the data's
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    *(volatile __global uint*)(land + FLAG(slot, j)) = n;
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  }
}

// One part's contribution to the fold's chain: S slices of [S][M][K], from this card's memory or (remote) the landing.
#define LOCAL_PART(acc)                                                                   \
  for (uint s = 0; s < S_MSG; ++s) acc += mine[((size_t)s * M + m) * K + k];
#define REMOTE_PART(acc)                                                                  \
  for (uint s = 0; s < S_MSG; ++s) acc += as_float(SYS_LOAD(src + ((size_t)s * M + m) * K + k, memory_order_relaxed));

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void tp_res_fold(__global const float* restrict mine, __global const uint* land, __global ushort* restrict resid,
                          __global float* restrict sumsq, __global uint* ctr, __global uint* state, uint spin_limit) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint j = g * TP_MROWS + m;
  __local float red[WG];
  __local uint s_want;
  if (lid == 0) {
    const uint want = tp_next(ctr, g, m);
    s_want = want;
    const uint slot = want & 1u;
    if (SYS_LOAD(state, memory_order_relaxed) == 0u) {   // after a failure: no spin, the host throws anyway
      uint got = 0u, i = 0u;
      do {
        got = SYS_LOAD(land + FLAG(slot, j), memory_order_acquire);
      } while (got != want && ++i < spin_limit);
      const uint stamp = SYS_LOAD(land + TRAILER(slot, j), memory_order_acquire);
      const uint status = got != want ? 1u : (stamp != want ? 2u : 0u);
      if (status != 0u) tp_record(state, status, got, stamp, want, j);
      if (status == 2u) tp_late(state, land + TRAILER(slot, j), want, spin_limit);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
  __global const uint* src = land + (s_want & 1u) * DATA_WORDS;
  __global ushort* rp = resid + (size_t)m * K;
  const uint k0 = g * CHUNK;
  uint k1 = k0 + CHUNK;
  if (k1 > K) k1 = K;
  float acc2 = 0.f;
  for (uint k = k0 + lid; k < k1; k += WG) {
    // prep_res_fold's chain: acc from 0 over the slices in one-card order (card 0's part first), one RNE, the add.
    float acc = 0.f;
#if RANK == 0
    LOCAL_PART(acc)
    REMOTE_PART(acc)
#else
    REMOTE_PART(acc)
    LOCAL_PART(acc)
#endif
    const ushort r_b = rne_bf16(bf16f(rp[k]) + bf16f(rne_bf16(acc)));
    rp[k] = r_b;
    const float v = bf16f(r_b);
    acc2 = fma(v, v, acc2);
  }
  red[lid] = acc2;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (lid < stride) red[lid] += red[lid + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sumsq[(size_t)g * M + m] = red[0];
}

#ifdef TP_SUM
// Spec 19 v3c: the DFlash drafter's all-reduce - tp_res_fold's wait and its fixed order (card 0's part first), the sum
// written whole in fp32 into `out` slice 0 ([M][K]: the conv's finish side folds it into the drafter's fp32 residual)
// and zeros into slices 1 .. TP_SUM - 1 (a consumer that folds TP_SUM split-K slices then reads the sum, + 0 exactly).
__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void tp_sum(__global const float* restrict mine, __global const uint* land, __global float* restrict out,
                     __global uint* ctr, __global uint* state, uint spin_limit) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint j = g * TP_MROWS + m;
  __local uint s_want;
  if (lid == 0) {
    const uint want = tp_next(ctr, g, m);
    s_want = want;
    const uint slot = want & 1u;
    if (SYS_LOAD(state, memory_order_relaxed) == 0u) {
      uint got = 0u, i = 0u;
      do {
        got = SYS_LOAD(land + FLAG(slot, j), memory_order_acquire);
      } while (got != want && ++i < spin_limit);
      const uint stamp = SYS_LOAD(land + TRAILER(slot, j), memory_order_acquire);
      const uint status = got != want ? 1u : (stamp != want ? 2u : 0u);
      if (status != 0u) tp_record(state, status, got, stamp, want, j);
      if (status == 2u) tp_late(state, land + TRAILER(slot, j), want, spin_limit);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
  __global const uint* src = land + (s_want & 1u) * DATA_WORDS;
  const uint k0 = g * CHUNK;
  uint k1 = k0 + CHUNK;
  if (k1 > K) k1 = K;
  for (uint k = k0 + lid; k < k1; k += WG) {
    float acc = 0.f;
#if RANK == 0
    LOCAL_PART(acc)
    REMOTE_PART(acc)
#else
    REMOTE_PART(acc)
    LOCAL_PART(acc)
#endif
    out[(size_t)m * K + k] = acc;
    for (uint s = 1; s < TP_SUM; ++s) out[((size_t)s * M + m) * K + k] = 0.f;
  }
}
#endif  // TP_SUM
#endif  // TP_CROSS

#if TP_ARGMAX
// argmax.cl over the card's half: its comparator, its two fixed trees, its mask; the index is the GLOBAL id.
#if !defined(VOCAB) || !defined(VOCAB_USED) || !defined(ID_BASE) || !defined(CTRL_OUTV)
#error "tp.cl TP_ARGMAX: VOCAB, VOCAB_USED, ID_BASE and CTRL_OUTV are the variant's"
#endif
#ifndef M
#define M 1
#endif
#define CHUNK_V 1024
#define GROUPS ((VOCAB + CHUNK_V - 1) / CHUNK_V)
#define NO_IDX 0x7FFFFFFFu
#if GROUPS > WG
#error "tp.cl: stage 2 folds at most 256 groups"
#endif

inline int argmax_better(float av, uint ai, float bv, uint bi) { return (av > bv) || (av == bv && ai < bi); }

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void tp_argmax_stage1(__global const float* restrict logits, __global float* restrict part) {
  const uint g = get_group_id(0);
  const uint m = get_group_id(1);
  const uint lid = get_local_id(0);
  __local float sv[WG];
  __local uint si[WG];
  const uint k0 = g * CHUNK_V;
  uint k1 = k0 + CHUNK_V;
  if (k1 > VOCAB) k1 = VOCAB;
  __global const float* restrict row = logits + (size_t)m * VOCAB;
  float bv = -INFINITY;
  uint bi = NO_IDX;
  for (uint k = k0 + lid; k < k1; k += WG) {
    const float v = (k < VOCAB_USED) ? row[k] : -INFINITY;
    if (argmax_better(v, ID_BASE + k, bv, bi)) {
      bv = v;
      bi = ID_BASE + k;
    }
  }
  sv[lid] = bv;
  si[lid] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = WG / 2; stride > 0; stride >>= 1) {
    if (lid < stride && argmax_better(sv[lid + stride], si[lid + stride], sv[lid], si[lid])) {
      sv[lid] = sv[lid + stride];
      si[lid] = si[lid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    part[((size_t)m * GROUPS + g) * 2 + 0] = sv[0];
    part[((size_t)m * GROUPS + g) * 2 + 1] = (float)si[0];   // exact: ids < 2^24
  }
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void tp_argmax_stage2(__global uint* restrict ctrl, __global const float* restrict part) {
  const uint lid = get_local_id(0);
  __local float sv[WG];
  __local uint si[WG];
  const uint n = ctrl[CTRL_NACT];
  for (uint m = 0; m < n; ++m) {
    float bv = -INFINITY;
    uint bi = NO_IDX;
    if (lid < GROUPS) {
      bv = part[((size_t)m * GROUPS + lid) * 2 + 0];
      bi = (uint)part[((size_t)m * GROUPS + lid) * 2 + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sv[lid] = bv;
    si[lid] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG / 2; stride > 0; stride >>= 1) {
      if (lid < stride && argmax_better(sv[lid + stride], si[lid + stride], sv[lid], si[lid])) {
        sv[lid] = sv[lid + stride];
        si[lid] = si[lid + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
      ctrl[CTRL_OUT + m] = si[0];
      ctrl[CTRL_OUTV + m] = as_uint(sv[0]);
    }
  }
  // argmax_stage2's token boundary; the host replaces cur_token / out_token with the merged winner.
  if (lid == 0 && n != 0u) {
    ctrl[CTRL_CUR] = ctrl[CTRL_OUT + n - 1u];
    ctrl[CTRL_POS] += n;
  }
}
#endif  // TP_ARGMAX
