// tp_wire.h - tp.cl's crossing protocol as shared pieces (included by tp.cl's TP_CROSS family and by
// qwen4exp/q4_moe.cl's wired MoE, MOE_TP_WIRE): the landing buffer's layout, the system-scope load, the wait's failure
// record and the per-chunk call numbers. tp.cl's header comment states the protocol; runtime/tp_protocol.h mirrors
// it on the host. The includer defines M, TP_MROWS, DATA_WORDS (the slot's data words) and J_CHUNKS (FOLD_G x TP_MROWS).
#define TP_SLOTS 2u
#define TP_FLAG_STRIDE 16u
#define TP_PAGE 1024u
#ifndef TP_TRAILER_STRIDE
#define TP_TRAILER_STRIDE 16u
#endif

#define SYS_LOAD(p, order) \
  atomic_load_explicit((volatile __global atomic_uint*)(p), order, memory_scope_all_svm_devices)

#define TRAILER(slot, j) (TP_SLOTS * DATA_WORDS + ((slot) * J_CHUNKS + (j)) * TP_TRAILER_STRIDE)
#define FLAG_BASE (((TP_SLOTS * DATA_WORDS + TP_SLOTS * J_CHUNKS * TP_TRAILER_STRIDE) + TP_PAGE - 1) / TP_PAGE * TP_PAGE)
#define FLAG(slot, j) (FLAG_BASE + ((slot) * J_CHUNKS + (j)) * TP_FLAG_STRIDE)

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
