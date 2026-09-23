# The barrier-free delta net solve: a premise that was wrong by 18x

The delta net triangular solve costs about 0.77 ms per layer per chunk. The
design that this document reports on assumed the cost was synchronisation:
two barriers per row of a 64-row loop, 128 barriers per call, and a target of
0.35 ms once they were gone.

Both barrier-free kernels came out **bit-identical** to the existing solve and
both were **slower** than it. The barriers were never the bottleneck. The
control that proves it deleted half of them and bought 1.7 ms of a 73.5 ms row,
so the premise was wrong by roughly **eighteen times**.

Every number is labelled measured or derived. Device 0, checkpoint
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, snapshot
`84575a18f209992ef96d819b31f924b489e3d55d`.

## 1. What was built

The delta net WY kernel file gained three entries. All three are pinned
**bitwise** against the production `pf_gdn_solve` at chunk widths 256, 100 and
4096, from independently filled device buffers whose equality is itself checked
by `memcmp` first.

| entry | `T` lives in | row barriers | dispatched |
|---|---|---:|---|
| `pf_gdn_solve`, the unchanged default | in place, over `A` | 2 per row | yes, default |
| `pf_gdn_solve_register` | its own 16 KiB SLM array | **0** | yes, `B70_PREFILL_GDN_SOLVE=register` |
| `pf_gdn_solve_private` | `float t[CT]` per lane | **0** | control only |
| `pf_gdn_solve_1bar` | in place, over `A` | **1 per row** | control only |

A solve-entry resolver sits beside the scan-entry resolver, resolved once per
process, validated before the first launch, and routes one launch of the ten in
the chunk. Empty, unset or `vector` gives the default; `register` gives the
SLM variant; anything else throws before any launch or buffer mutation. The
ten-launch contract, the phase accounting, replay behaviour and every buffer
interface are unchanged.

A stale comment describing the matrix as `(I - A)^-1` is corrected in the
kernel header and at the launch site. The sign is `(I + A)^-1`. No arithmetic
moved.

## 2. Correctness

**Bitwise.** All three new entries are `memcmp`-equal to `pf_gdn_solve` at
every width:

| chunk width | fp32 entries compared | register | private | 1bar |
|---:|---:|---|---|---|
| 256 | 786,432 | equal | equal | equal |
| 100 | 393,216 | equal | equal | equal |
| 4096 | 12,582,912 | equal | equal | equal |

The existing fp64 residual check that `(I + A)T == I` is untouched and still
passes with margin: max 5.077e-08, 2.999e-08 and 3.595e-08 against a derived
bar of 7.629e-06. The exact diagonal, exact upper triangle and exact dead-tail
checks are unchanged and were not loosened.

**The dead tail is where a `memcmp` earns its keep.** The A kernel writes
`+0.0f` outside the live square, and staging negates the strictly lower half,
so `T`'s dead tail carries **`-0.0f`**. At width 100 the test counts **66,528**
such words, exactly `48 heads * sum(i = 36..63)`. A `CHECK_EQ(., 0.0f)` cannot
see that sign bit. It is why every new entry seeds its output storage from the
staging expression rather than zeroing it.

**Dispatch.** Because correct entries produce identical bytes on purpose, the
dispatch proof cannot observe the output. It observes the kernel cache around a
real chunk call instead: after a `register` call, asking the cache for
`pf_gdn_solve_register` adds nothing and asking for `pf_gdn_solve` adds exactly
one, and conversely.

**Mutation-proved, not asserted.** Temporarily hard-wiring the `register`
branch to `pf_gdn_solve` was built and run. The dispatch test failed at the
exact cache-count assertion while the output-comparison test still passed,
which is the demonstration that output comparison could not have caught it. The
mutation was reverted and the suite re-run green.

The invalid-selector case keeps the real-boundary proof: the chunk call throws,
the launch count is zero, and the nonzero state and ring sentinels are bytewise
unchanged.

**Model gates**, under `B70_PREFILL_GDN_SOLVE=register`, on the final binary:

| gate | result |
|---|---|
| `prefill_replay_test` | **Passed**, 92.6 s |
| `prefill_consistency_l0_test` | **Passed**, 66.7 s |
| `prefill_determinism_l0_test` | **Passed**, 81.5 s |
| `prefill_backend_equivalence_test` | **Passed**, 34.8 s |
| `prefill_gate_l0_test` | **not run**, see below |

The same four gates were run under the default as a control and pass.
`prefill_replay_test` is the load-bearing one: it snapshots exact persistent
state, ring, KV, control **and logits** across two full chunks plus a ragged
4097-token tail, bytewise.

**The golden token gate did not run, and that is a genuine gap in the
evidence.** Its CPU-oracle set was not reachable from the isolated tree these
measurements were taken in. What stands in its place is weaker in provenance
and stronger in kind: the three new entries are `memcmp`-equal to the default
at three widths, and the replay test passes bytewise on exact logits over 4097
tokens. Identical logits imply identical tokens, so the 93/93 token gate could
not have failed while the logits compare equal. It should still be run before
any adoption decision.

## 3. The measurement

One profiled process per variant, device 0, box proven idle immediately
beforehand (zero containers, zero DRM fd holders), 4096 ids as two 2048-token
chunks, split-BF16 scan, Level Zero backend. The `gdn_solve` row is 96 launches
(48 delta net layers by 2 chunks), so ms per layer per chunk is the row divided
by 96.

| variant | `gdn_solve` L0 GPU ms | **ms/layer/chunk** | vs control | vs the 0.35 target |
|---|---:|---:|---:|---|
| default, **fresh control** | 73.5 | **0.7656** | | 2.19x over |
| `register`, 32 KiB SLM | 115.9 | **1.2073** | **+57.7%** | 3.45x over |
| `private`, 8 KiB/thread scratch | 83.6 | **0.8708** | **+13.7%** | 2.49x over |
| `1bar`, in place, 1 barrier/row | 71.8 | **0.7479** | **-2.3%** | 2.14x over |

The fresh control at 0.7656 sits inside the historical 0.7528 to 0.7932 band,
so the comparison is like for like.

**Why this is the answer and not a tuning problem.** `1bar` changes exactly one
variable: it is the production kernel with the second per-row barrier deleted
and nothing else. Same in-place 16 KiB staging, same arithmetic, no extra
storage. Deleting 64 of the 128 row barriers bought **1.7 ms**. Extrapolating
linearly, all 128 are worth **about 3.4 ms (derived)** of a 73.5 ms row, which
is **4.6%**. Reaching the 0.35 ms target needed 62 ms. The row is dominated by
the serialized recurrence and its SLM traffic.

Both designed approaches then paid real money for that 4.6%: the SLM variant
gave up half its occupancy, 32 KiB per work-group instead of 16, and lost
42.4 ms; the private variant moved its per-lane array into 8,192 B per thread
of scratch and lost 10.1 ms. The prediction that the private variant would be
lowered to scratch was correct. The prediction that the SLM variant could not
spill was also correct, and it was nonetheless the worse of the two.

Instrumented whole-call walls from the same four processes were 2687.4, 2732.7,
2686.6 and 2695.5 ms. They are recorded only as process context. They are
profiled walls with phase waits in them, they are **not** throughput rows, and
they must never be quoted as t/s or compared against a vLLM row. No ABBA
throughput row was taken.

## 4. Compiler evidence

Compiled on the committed source for this device with the project's own
options, shader dumps enabled:

| | default | `register` | `private` | `1bar` |
|---|---|---|---|---|
| `simd_size` | 32 | **32** | 32 | 32 |
| `grf_count` | 128 | **128** | 128 | 128 |
| `slm_size` | 16,384 B | **32,768 B** | 16,384 B | 16,384 B |
| `private_size` (scratch) | 0 | **0** | **8,192 B** | 0 |
| register spill | none | **none** | none | none |
| static `sync.bar` | 3 | **2** | 3 | 3 |
| **barriers inside the row loop** | **2** | **0** | **0** | **1** |
| float `mad (32\|M0)` in the recurrence | 1 | **1** | 1 | 1 |

`-abortonspill -abortOnSpill 4` is on the link line for every entry and every
build succeeded, so "no spill" is enforced rather than observed. The private
variant's 8,192 B is addressed **private** memory, not register spill: 64
floats by 32 lanes, exactly its per-lane array lowered to scratch. It does not
trip the abort flag, which is why the measurement and not the flag is the
evidence.

The row-loop barrier counts are read off the control flow of the dumped Xe2
ISA, not inferred. In the default kernel the row loop runs from a label to its
back edge with `sync.bar` at two points inside it. In the SLM variant the
corresponding loop body contains **no** `sync.bar`: its two are the
post-staging barrier and the one before the coalesced copy.

The explicit FMA chain survives in every entry: one `mad (32|M0) ...:f` inside
the inner loop, SIMD32, fp32 destination. No vector reduction, no tree, no dot
intrinsic, nothing re-associated.

## 5. What this leaves

1. **The default is unchanged.** Nothing about the production path moved.
2. **The 40 ms this work was expected to be worth does not exist.** The
   mechanism it was built on is worth about 3.4 ms (derived), and the remaining
   cost of the row is the serialized recurrence itself. Any future attempt has
   to change the algorithm, for instance a block-recursive inversion, which is
   a different numerical project with its own gates rather than a
   resynchronisation.
3. **`pf_gdn_solve_1bar` is the one result worth keeping.** It is bitwise
   identical, costs nothing, adds no storage, deletes a barrier that provably
   never protected anything, and measured **-1.7 ms, -2.3%**. Folding it into
   the default kernel is a two-line change. It was deliberately not made here,
   because this round fixed the default, and it is offered with its
   measurement already taken.

The three extra entries cost compile time and nothing else. They are the
evidence behind point 2.
