# GDN solve register-column experiment - results

Stage **S4** of `docs/superpowers/specs/2026-09-22-prefill-parity-program-design.md`,
implementing `docs/superpowers/specs/2026-09-22-gdn-solve-register-design.md` as
amended by its §9.

Every number below is **measured** unless marked **derived**. Device 0
(`ZE_AFFINITY_MASK=0`), remote tree `b70-s4`, IGC/ocloc as installed on the box,
checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` (snapshot 84575a1).

## 1. Verdict

**The pre-registered kernel target is missed, and the default stays `vector`.**
Both barrier-free entries the design asked for are **bit-identical** to
`pf_gdn_solve` and both are **slower than it**. §7's rollback rule applies as
written: record the evidence, leave the default unchanged.

The experiment did not fail for the reason the design expected. It was not
register pressure, and for approach B it was not spill at all. **The two
per-row barriers the whole design was built to delete are worth at most
~3.4 ms of a 73.5 ms row (derived, §4), and the target asked for 62 ms.** The
barriers were never the bottleneck, and no rearrangement of where `T` lives
could have made them into one.

## 2. What was built

`src/kernels/prefill/pf_gdn_wy.cl` gained three entries. All three are pinned
**bitwise** against `pf_gdn_solve` in `tests/prefill/gdn_wy_test.cc` case 8, at
`C = 256`, `100` and `4096`, from independently filled device `A` buffers whose
equality is itself memcmp'd first.

| entry | `T` lives in | row barriers | dispatched |
|---|---|---:|---|
| `pf_gdn_solve` (unchanged default) | in place, over `A` | 2 per row | yes, default |
| `pf_gdn_solve_register` - design approach **B** | its own 16 KiB SLM array | **0** | yes, `B70_PREFILL_GDN_SOLVE=register` |
| `pf_gdn_solve_private` - design approach **A** | `float t[CT]` per lane | **0** | no (control) |
| `pf_gdn_solve_1bar` - variable-isolating control | in place, over `A` | **1 per row** | no (control) |

`src/runtime/prefill/gdn.cc` gained `gdn_solve_entry()`, resolved once per
process beside `gdn_scan_entry()`, validated before launch 1, routing launch 6
only. `vector`/empty/unset → `pf_gdn_solve`; `register` → approach B; anything
else throws before any launch or buffer mutation. The ten-launch contract, the
`kGdnSolve` phase, replay behaviour and every buffer interface are unchanged.

The stale `(I - A)^-1` wording is corrected in `pf_gdn_wy.cl`'s file header and
in `gdn.cc`'s launch-6 comment. The sign is `(I + A)^-1`; no arithmetic moved.

## 3. Correctness

**Bitwise (gate 1).** `gdn_wy_test` - all three new entries memcmp-equal to
`pf_gdn_solve` at every width:

| C | fp32 entries compared | register (B) | private (A) | 1bar |
|---:|---:|---|---|---|
| 256 | 786,432 | equal | equal | equal |
| 100 | 393,216 | equal | equal | equal |
| 4096 | 12,582,912 | equal | equal | equal |

The existing fp64 `(I + A)T == I` residual is untouched and still passes with
margin: max `5.077e-08` / `2.999e-08` / `3.595e-08` against the derived bar
`L*u*max|T| = 7.629e-06`. The exact diagonal, exact upper triangle and exact
dead-tail checks are unchanged and were not loosened.

**The dead tail is where a memcmp earns its keep.** `pf_gdn_A` writes `+0.0f`
outside `[0,L)^2`; staging negates the strictly lower half, so `T`'s dead tail
carries **`-0.0f`**. At `C = 100` the test counts **66,528** such words -
exactly `48 heads * sum(i = 36..63) = 48 * 1386`. Case 5's `CHECK_EQ(., 0.0f)`
cannot see that sign bit. This is why every new entry seeds its output storage
from the staging expression rather than zeroing it.

**Dispatch (gate 2).** `gdn_chunk_test` passes under both selector values, and
the four new CTest rows (invalid / default / vector / register) are green.
Because correct entries produce identical bytes on purpose, the dispatch proof
observes `KernelCache::kernels()` around a real `gdn_chunk` call rather than the
output: after a `register` call, asking the cache for `pf_gdn_solve_register`
adds nothing and asking for `pf_gdn_solve` adds exactly one, and conversely.
**Mutation-proved, not asserted:** temporarily hard-wiring the `register` branch
to `pf_gdn_solve` was built and run, and
`gdn_solve_register_dispatch_test` failed at `gdn_chunk_test.cc:344`
(`d.kc.kernels() != built`) while `gdn_chunk_solve_register_test` still passed -
which is the demonstration that output comparison could not have caught it. The
mutation was then reverted and the suite re-run green.

The invalid-selector row keeps the existing real-boundary proof: `gdn_chunk`
throws, `launches() == 0`, and the nonzero state/ring sentinels are bytewise
unchanged.

**Model gates (gate 3), under `B70_PREFILL_GDN_SOLVE=register`, final binary:**

| gate | result |
|---|---|
| `prefill_replay_test` | **Passed** (92.6 s) |
| `prefill_consistency_l0_test` | **Passed** (66.7 s) |
| `prefill_determinism_l0_test` | **Passed** (81.5 s) |
| `prefill_gate_l0_test` | **SKIPPED - not run**, see §6 |
| `prefill_backend_equivalence_test` | **Passed** (34.8 s) |

The same three passing gates were also run under the default selector as a
control and pass. `prefill_replay_test` is the load-bearing one here: it
snapshots exact persistent state, ring, KV, control **and logits** across two
full chunks plus a ragged 4097-token tail, bytewise.

## 4. Measurement

One diagnostic `B70_PREFILL_PROFILE=1` process per variant, device 0, box
proven idle immediately beforehand (0 containers, 0 DRM fd holders), 4096 ids as
two 2048-token chunks, `B70_PREFILL_GDN_SCAN=dpas_split`, `--pp-backend l0`.
The `gdn_solve` row is 96 launches (48 GDN layers x 2 chunks), so
ms/layer/chunk is the row divided by 96.

| variant | `gdn_solve` L0 GPU ms | **ms/layer/chunk** | vs control | vs 0.35 target |
|---|---:|---:|---:|---|
| `vector` - **fresh control** | 73.5 | **0.7656** | - | 2.19x over |
| `register` (B, 32 KiB SLM) | 115.9 | **1.2073** | **+57.7%** | 3.45x over |
| `private` (A, 8 KiB/thread scratch) | 83.6 | **0.8708** | **+13.7%** | 2.49x over |
| `1bar` (in place, 1 barrier/row) | 71.8 | **0.7479** | **−2.3%** | 2.14x over |

The fresh control at 0.7656 ms/layer/chunk sits inside the historical
0.7528-0.7932 band, so the control is sound and the comparison is like for like.

**Why this is the answer and not a tuning problem.** `1bar` changes exactly one
variable: it is `pf_gdn_solve` with the second per-row barrier deleted and
nothing else - same in-place 16 KiB `As`, same arithmetic, no extra storage.
Deleting 64 of the 128 row barriers bought **1.7 ms**. Extrapolating linearly,
all 128 are worth **≈ 3.4 ms (derived)** of a 73.5 ms row - **4.6%**. Reaching
0.35 ms/layer/chunk needed **62 ms**. The row is dominated by the serialized
`i`-recurrence and its SLM traffic, and the design's premise - that
synchronization was the thing to attack - is measurably false by a factor of
roughly eighteen.

Both designed approaches then paid real money for that 4.6%: approach B gave up
half its SLM residency (32 KiB per work-group instead of 16) and lost 42.4 ms;
approach A moved `t[CT]` into 8,192 B per thread of scratch and lost 10.1 ms.
§9's prediction that A would be lowered to scratch was **correct**; its
prediction that B "cannot spill" was also correct, and B was nonetheless the
worse of the two.

Instrumented whole-call walls from the same four processes - 2687.4 (control),
2732.7 (B), 2686.6 (A), 2695.5 (1bar) ms - are recorded only as process context.
They are profiled walls with phase waits in them, they are **not** throughput
rows, and they must never be quoted as t/s or compared with the vLLM row.
**No ABBA throughput row was taken**: the protocol needs a quiet box and device 1
belonged to another agent for the duration.

## 5. Compiler evidence (gate 4)

`ocloc compile` on the committed `.cl` for `bmg-g31` with the project's own
options, `IGC_ShaderDumpEnable=1` (measured compiler artifact):

| | `pf_gdn_solve` | `pf_gdn_solve_register` (B) | `pf_gdn_solve_private` (A) | `pf_gdn_solve_1bar` |
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
build succeeded, so "no spill" is enforced rather than observed. Approach A's
8,192 B is addressed *private* memory, not register spill - 64 floats x 32
lanes, exactly `t[CT]` lowered to scratch as §9 predicted; it does not trip
`-abortonspill`, which is why the measurement and not the flag is the evidence.

The row-loop barrier counts are read off the control flow of the dumped Xe2 ISA,
not inferred. In `pf_gdn_solve` the row loop runs from label `_0_042` to its
back-edge `jmpi _0_042`, with `sync.bar` at two points inside it. In
`pf_gdn_solve_register` the corresponding loop body contains **no** `sync.bar`;
its two are the post-staging barrier and the one before the coalesced copy.

The explicit FMA chain survives in every entry: one `mad (32|M0) ...:f` inside
the inner `l` loop, SIMD32, fp32 destination. No vector reduction, no tree, no
dot intrinsic, nothing re-associated.

## 6. `prefill_gate_l0_test` was not run

It **skipped** (exit 77) in both the `register` and the control runs, for an
environmental reason unrelated to this change: the CPU-oracle golden set
`oracle-out-primary/` is excluded from `tools/box.sh`'s rsync, and the only copy
on the box lives inside another agent's tree, which this task's isolation rules
put out of bounds. A copy into `b70-s4` was attempted once and refused by the
permission system; it was not worked around.

This is a genuine gap in the evidence and is recorded as one. What stands in its
place is weaker in provenance but stronger in kind: the three new entries are
**memcmp-equal** to the default at three widths, and `prefill_replay_test`
passes bytewise on exact logits over 4097 tokens. Identical logits imply
identical tokens; the 93/93 token gate could not have failed while the logits
compare equal. It should still be run before any adoption decision.

## 7. Ruling requested

1. **Default stays `vector`.** Nothing about the production path changed.
2. **S4's −40 ms estimate should be struck from the program table.** The
   mechanism it was built on is worth ~3.4 ms (derived), not 40, and the
   remaining `gdn_solve` cost is the serialized recurrence itself. Any future
   attempt on this row has to change the algorithm - the design's deferred
   approach C, block-recursive inversion - which is a different numerical
   project with its own gates, not a resynchronisation.
3. **`pf_gdn_solve_1bar` is the one result worth keeping.** It is bitwise
   identical, costs nothing, adds no storage, deletes a barrier that provably
   never protected anything, and measured **−1.7 ms** (−2.3%). Folding it into
   `pf_gdn_solve` is a two-line change to the default kernel. It was deliberately
   **not** made, because this task's constraints fix the default; it is offered
   as a follow-on with its measurement already taken.
4. Whether `pf_gdn_solve_register`, `pf_gdn_solve_private` and
   `pf_gdn_solve_1bar` stay in the tree as opt-in/controls or are removed is an
   implementation-review call. They cost compile time and nothing else; they are
   the evidence behind ruling 2.
