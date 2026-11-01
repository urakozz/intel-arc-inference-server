// probe_tp.cl - spec 17a's PROBE kernels for the TP all-reduce (fold-then-send, spec 17 §R2.5 / §R2.6). No runtime
// path binds them. The protocol is tools/probe/tp_protocol.h's, line for line (a change to one side is a change to
// both); tests/runtime/tp_crossing_protocol_test.cc runs that host mirror on threads.
//
//   tp_send   one work-group a chunk of the message (M x hidden fp32 words): folds the card's S split-K slices in
//             slice order, adds the call's term tp_gen(n, rank, e) (so every call's row differs and the host can
//             recompute it), keeps the folded row in `own` and - unless this is the control list (cross == 0) -
//             writes it, the trailer stamp, a system-scope release fence and the flag into `dst`: the PEER's landing
//             (p2p-push), the card's own exposed buffer (p2p-pull), or a pinned host mailbox (host-*).
//   tp_wait   one work-group a chunk: lane 0 spins (bounded) on the flag in `src` with system-scope acquire loads,
//             checks the trailer, records the first failure; every lane reads the data with system-scope loads and
//             writes out = own + peer in the fixed order (card 0's word, then card 1's) - or out = the sum itself
//             when the CPU summed (host-bar / host-read, take == 1). The control list (cross == 0) skips the wait and
//             adds own to itself, so the two lists differ only by the crossing. With `check`, every output word is
//             compared with the host formula (tp_fold of both ranks) and mismatches counted; every output word goes
//             into a per-chunk hash (tp_mix, summed: order-free) the host compares across the two cards.
//   tp_ping   (probe_tp bar) one work-group, persistent: for i = 1..iters waits for the CPU's stamp i in `land` (a
//             BAR-mapped device allocation, a pinned host buffer, ...), checks the row it carries against tp_gen(i),
//             then writes i to `ack` (pinned host or BAR-read VRAM) - the CPU times the round trip. The row's words are
//             tp_gen_bits(i % 64, 0, e) (the CPU writes a precomputed row; the trailer catches staleness).
//
// Both P2P rules of spec 16 §2: the landing buffers are allocations of their own holding nothing the reading card
// writes, the flags on a page of their own; everything another agent wrote (flag, trailer AND data) is read with
// system-scope atomics. Counters live in device memory, one per work-group, advanced by the kernels: a replayed
// list with frozen arguments progresses. Every spin is bounded; a failure is recorded, the list finishes, the host
// reads `state` after the fence.
//
// state: [0] first failure's status (0 ok, 1 timeout, 2 stale), [1] flag read, [2] trailer read, [3] call expected,
// [4] chunk, [5] failed chunk-calls (count), [6] mismatched output words (count, check only).

#define WG 256
#define SLOTS 2u
#define FLAG_STRIDE 16u
#define POISON 0x80000000u

#define SYS_LOAD(p, order) \
  atomic_load_explicit((volatile __global atomic_uint*)(p), order, memory_scope_all_svm_devices)
#define SYS_STORE(p, v, order) \
  atomic_store_explicit((volatile __global atomic_uint*)(p), (v), order, memory_scope_all_svm_devices)

inline uint tp_gen_bits(uint n, uint rank, uint e) {
  uint h = n * 0x9E3779B1u ^ (rank + 1u) * 0x85EBCA77u ^ e * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  h *= 0x297A2D39u;
  h ^= h >> 15;
  return h;
}
inline float tp_gen(uint n, uint rank, uint e) {
  return as_float((tp_gen_bits(n, rank, e) & 0x007FFFFFu) | 0x3F800000u) - 1.5f;
}
inline float tp_fold(__global const float* P, uint S, uint words, uint e, uint n, uint rank) {
  float v = P[e];
  for (uint s = 1u; s < S; ++s) v += P[s * words + e];
  return v + tp_gen(n, rank, e);
}
inline float tp_combine(float own, float peer, uint rank) { return rank == 0u ? own + peer : peer + own; }
inline uint tp_mix(uint bits, uint e) {
  uint h = bits ^ (e * 0x9E3779B1u);
  h ^= h >> 16;
  h *= 0x7FEB352Du;
  h ^= h >> 15;
  h *= 0x846CA68Bu;
  h ^= h >> 16;
  return h;
}
// tp_protocol::Layout: data(slot) = slot x words; trailer(slot, j) = 2 x words + slot x G + j;
// flag(slot, j) = flag_base + (slot x G + j) x 16 (flag_base from the host's Layout, a kernel argument).
inline uint tp_trailer(uint words, uint G, uint slot, uint j) { return SLOTS * words + slot * G + j; }
inline uint tp_flag(uint flag_base, uint G, uint slot, uint j) { return flag_base + (slot * G + j) * FLAG_STRIDE; }

inline void tp_record(__global uint* state, uint status, uint got, uint stamp, uint want, uint j) {
  atomic_inc((volatile __global uint*)(state + 5));
  if (atomic_cmpxchg((volatile __global uint*)state, 0u, status) == 0u) {
    state[1] = got;
    state[2] = stamp;
    state[3] = want;
    state[4] = j;
  }
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tp_send(__global const float* P, __global float* own, __global uint* dst, __global uint* ctr,
             uint words, uint chunk, uint S, uint flag_base, uint rank, uint cross,
             uint inject_at, uint inject_kind) {
  const uint lid = get_local_id(0);
  const uint j = get_group_id(0);
  const uint G = words / chunk;
  __local uint s_n;
  if (lid == 0) s_n = ctr[j] + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint n = s_n;
  const uint slot = n & 1u;
  const uint inj = (inject_at != 0u && n == inject_at) ? inject_kind : 0u;
  const uint e0 = j * chunk;
  __global uint* d = dst + slot * words;
  const bool write = cross != 0u && inj == 0u;
  for (uint e = e0 + lid; e < e0 + chunk; e += WG) {
    const float v = tp_fold(P, S, words, e, n, rank);
    own[e] = v;
    if (write) d[e] = as_uint(v);
  }
  if (lid == 0) {
    ctr[j] = n;
    if (write) dst[tp_trailer(words, G, slot, j)] = n;
  }
  if (cross == 0u || inj == 2u) return;   // the control list, or a dropped write (no flag): uniform per group
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) SYS_STORE(dst + tp_flag(flag_base, G, slot, j), n, memory_order_release);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tp_wait(__global const uint* src, __global const float* own, __global float* out, __global uint* ctr,
             __global uint* state, __global uint* hash, __global const float* P,
             uint words, uint chunk, uint S, uint flag_base, uint rank, uint cross, uint take,
             uint spin_limit, uint check) {
  const uint lid = get_local_id(0);
  const uint j = get_group_id(0);
  const uint G = words / chunk;
  __local uint s_want;
  __local uint s_h[WG];
  if (lid == 0) {
    const uint want = ctr[j] + 1u;
    ctr[j] = want;
    s_want = want;
    if (cross != 0u) {
      const uint slot = want & 1u;
      uint got = 0u, k = 0u;
      do {
        got = SYS_LOAD(src + tp_flag(flag_base, G, slot, j), memory_order_acquire);
      } while (got != want && ++k < spin_limit);
      const uint stamp = SYS_LOAD(src + tp_trailer(words, G, slot, j), memory_order_acquire);
      const uint status = got != want ? 1u : (stamp != want ? 2u : 0u);
      if (status != 0u) tp_record(state, status, got, stamp, want, j);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
  const uint n = s_want;
  const uint slot = n & 1u;
  const uint e0 = j * chunk;
  __global const uint* d = src + slot * words;
  uint h = 0u, bad = 0u;
  for (uint e = e0 + lid; e < e0 + chunk; e += WG) {
    const float o = own[e];
    float r;
    if (cross == 0u) {
      r = tp_combine(o, o, rank);
    } else {
      const float peer = as_float(SYS_LOAD(d + e, memory_order_relaxed));
      r = take != 0u ? peer : tp_combine(o, peer, rank);
    }
    out[e] = r;
    h += tp_mix(as_uint(r), e);
    if (check != 0u) {
      const float want = cross == 0u ? tp_combine(o, o, rank)
                                     : tp_fold(P, S, words, e, n, 0u) + tp_fold(P, S, words, e, n, 1u);
      if (as_uint(want) != as_uint(r)) ++bad;
    }
  }
  s_h[lid] = h;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint w = WG / 2u; w > 0u; w >>= 1) {
    if (lid < w) s_h[lid] += s_h[lid + w];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) hash[j] += s_h[0];
  if (bad != 0u) atomic_add((volatile __global uint*)(state + 6), bad);
}

// The BAR / host round trip. land: a single-chunk tp_protocol::Layout of `words` words (trailer at 2 x words + slot,
// flag at flag_base + slot x 16). ack: one word the CPU spins on.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tp_ping(__global const uint* land, __global uint* ack, __global uint* state,
             uint words, uint flag_base, uint iters, uint spin_limit) {
  const uint lid = get_local_id(0);
  __local uint s_ok;
  for (uint i = 1u; i <= iters; ++i) {
    const uint slot = i & 1u;
    if (lid == 0) {
      uint got = 0u, k = 0u;
      do {
        got = SYS_LOAD(land + flag_base + slot * FLAG_STRIDE, memory_order_acquire);
      } while (got != i && ++k < spin_limit);
      const uint stamp = SYS_LOAD(land + SLOTS * words + slot, memory_order_acquire);
      const uint status = got != i ? 1u : (stamp != i ? 2u : 0u);
      if (status != 0u) tp_record(state, status, got, stamp, i, 0u);
      s_ok = status;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
    const uint status = s_ok;
    uint bad = 0u;
    for (uint e = lid; e < words; e += WG)
      if (SYS_LOAD(land + slot * words + e, memory_order_relaxed) != tp_gen_bits(i & 63u, 0u, e)) ++bad;
    if (bad != 0u) atomic_add((volatile __global uint*)(state + 6), bad);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (status == 1u) return;   // the CPU is gone: stop rather than spin every remaining iteration
    atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
    if (lid == 0) SYS_STORE(ack, i, memory_order_release);
  }
}
