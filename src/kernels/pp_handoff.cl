// pp_handoff.cl - spec 16b's peer hand-off between the two pipeline stages
// (`--pipeline-handoff peer`; spec 16 §2 option (b)). Two entry points, one work-group of
// WG lanes each, grid (1):
//
//   pp_send   the LAST launch of device 0's decode list. Copies the residual row(s) and
//             the norm sums the cut's prep_res_fold just wrote into device 1's landing
//             buffer (a PEER write: `land_*` and `flag` are device-1 allocations), stamps
//             the sequence number after the sums, then publishes it in `flag` with a
//             system-scope release store. `seq` is device 0's own counter.
//   pp_recv   the FIRST launch of device 1's decode list. Spins (bounded) on `flag` with
//             system-scope acquire loads until it reads the next sequence number, checks
//             the stamp, and copies the landing buffer into device 1's own `resid` and
//             `norm_sumsq`, where layer s's prep_norm_finish reads them.
//
// The two rules from peer-to-peer work on this card pair (spec 16 §2), applied here:
//   1. every word the peer wrote is read with a SYSTEM-scope atomic load
//      (memory_scope_all_svm_devices): a plain or volatile load can be served from a
//      stale cache line on device 1 and miss the peer's PCIe write indefinitely. That is
//      the flag, the stamp AND the data - device 1 read the same landing lines one step
//      earlier, so a cached copy of them is exactly the stale buffer vLLM's PP decoded;
//   2. the landing buffer is an allocation of its own (runtime::PipelineLink), holding
//      nothing device 1 writes - pp_recv only reads it - with the flag on a page of its
//      own.
//
// Replayable with frozen arguments (spec 16a Review Focus 3): both counters live in device
// memory and are advanced by these kernels, never by the host - device 0's `seq[0]`, and
// device 1's `state[0]` (the last sequence it received). The host zeroes both, and the
// flag, at reset; step n then sends and expects n.
//
// Bounded (spec 16 §4 P4): pp_recv gives up after `spin_limit` loads of the flag, records
// why in `state` and lets the list finish (on garbage) - the host reads `state` after the
// fence and throws, and the engine must be reset before it runs again. state:
//   [0] the last sequence number this device consumed (advanced even on a failure)
//   [1] 0 = every hand-off so far was good; 1 = the flag never reached the expected
//       value within spin_limit loads; 2 = the flag did but the stamp did not (the data
//       did not arrive with it - the "silent hand-off" spec 16 §2 warns about)
//   [2] the flag value read at the failure, [3] the stamp read at the failure,
//   [4] the sequence number expected at the failure (the first failure is kept)
//
// Host mirror: tests/runtime/pp_protocol_test.cc runs this protocol on two host threads
// (runtime/pipeline_protocol.h) - sequence numbers, replay, timeout, stale stamp, reset.
// What only the box shows: that the system-scope atomics really bypass device 1's caches
// for a PCIe peer write (spec 16a Review Focus 1), and the latency.

#define WG 256

#define SYS_LOAD(p, order) \
  atomic_load_explicit((volatile __global atomic_uint*)(p), order, memory_scope_all_svm_devices)

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void pp_send(__global const uint* resid, __global const uint* sumsq,
             __global uint* land_resid, __global uint* land_sumsq,
             __global uint* flag, __global uint* seq,
             uint resid_words, uint sumsq_words) {
  const uint lid = get_local_id(0);
  __local uint s_next;
  if (lid == 0) s_next = seq[0] + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);
  const uint next = s_next;
  for (uint i = lid; i < resid_words; i += WG) land_resid[i] = resid[i];
  for (uint i = lid; i < sumsq_words; i += WG) land_sumsq[i] = sumsq[i];
  // The stamp: the receiver checks it after the flag, so data that did not travel with
  // the flag is caught rather than decoded.
  if (lid == 0) land_sumsq[sumsq_words] = next;
  // Every lane's peer stores are released at system scope before the work-group barrier,
  // so lane 0's release store of the flag is ordered after all of them.
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_all_svm_devices);
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    seq[0] = next;
    atomic_store_explicit((volatile __global atomic_uint*)flag, next, memory_order_release,
                          memory_scope_all_svm_devices);
  }
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void pp_recv(__global const uint* land_resid, __global const uint* land_sumsq,
             __global const uint* flag, __global uint* state,
             __global uint* resid, __global uint* sumsq,
             uint resid_words, uint sumsq_words, uint spin_limit) {
  const uint lid = get_local_id(0);
  if (lid == 0) {
    const uint want = state[0] + 1u;
    uint got = 0u, n = 0u;
    do {
      got = SYS_LOAD(flag, memory_order_acquire);
    } while (got != want && ++n < spin_limit);
    const uint stamp = SYS_LOAD(land_sumsq + sumsq_words, memory_order_acquire);
    const uint status = got != want ? 1u : (stamp != want ? 2u : 0u);
    state[0] = want;
    if (status != 0u && state[1] == 0u) {
      state[1] = status;
      state[2] = got;
      state[3] = stamp;
      state[4] = want;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_all_svm_devices);
  for (uint i = lid; i < resid_words; i += WG) resid[i] = SYS_LOAD(land_resid + i, memory_order_relaxed);
  for (uint i = lid; i < sumsq_words; i += WG) sumsq[i] = SYS_LOAD(land_sumsq + i, memory_order_relaxed);
}
