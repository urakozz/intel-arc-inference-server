// pf_probe_chain.cl -- the ordering and argument-binding probe for
// `runtime::prefill::Context` (plan 6b Task 2). It exists only for
// `tests/prefill/context_test.cc` and is bound by nothing else.
//
// `buf[i] += step` is a read-modify-write on every element. Launched N times in
// a row on one Context with `step` = the launch index, every element ends at
// `sum_{i<N} i` **only if** (a) each launch's arguments were resolved at its own
// append and not frozen at the first, and (b) the launches executed in order
// with no overlap on `buf`. Either failure gives a different number: a frozen
// argument gives `N * step_0 = 0`, and an out-of-order or concurrent pair loses
// or duplicates an increment. That is the empirical proof behind
// `ZE_COMMAND_QUEUE_FLAG_IN_ORDER` in src/sycl/context.cc, and the property
// `gdn_chunk`'s ten launches rest on.
//
// `n` is a runtime bound, in keeping with the prefill family's rule that the
// row count is never a `-D` (interfaces.md, "Layout conventions").

#define WG_CHAIN 256

__attribute__((reqd_work_group_size(WG_CHAIN, 1, 1)))
__kernel void pf_chain_step(__global uint* restrict buf, uint n, uint step) {
  const uint i = get_global_id(0);
  if (i < n) buf[i] += step;
}
