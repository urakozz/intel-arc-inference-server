# GDN solve register-column experiment - design

**Status:** approved design; implementation and measurement have not begun.

## 1. Purpose, scope, and decision rule

This is a bounded, opt-in Level Zero/OpenCL experiment for the triangular-solve
stage of prefill GDN.  It replaces neither the GDN algorithm nor the production
default.  Its purpose is to remove synchronization that the current solve needs
only because it overwrites its own input in SLM, while preserving the existing
fp32 arithmetic operation-for-operation.

The controlled current measurement is the recorded split-scan ABBA median:
**2579.468 ms / 1587.924 tokens/s**, against matched vLLM at **2544.043 ms /
1610.04 tokens/s**.  Native therefore trails by **35.425 ms / 22.116 tokens/s**.
The configuration is the AutoRound checkpoint
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, 4096 repeated committed prose
IDs, two 2048-token chunks, one uncached sequence, BF16 activations/KV and fp32
GDN state.  The measurement and its protocol are recorded in
`docs/prefill-gdn-scan-split-2026-09-20.md` §5, especially lines 207-247.

Only launch 6 of `runtime::prefill::gdn_chunk` is in scope.  `pf_gdn_wu` is the
next GDN candidate only if this experiment is accepted or rejected; it is not
part of this change.  GDN scan, attention, GEMM/dequant, pointwise fusion,
checkpoint formats, decode, the number/order of launches, and the default
selector behavior are explicitly out of scope.

The experiment is accepted only if all correctness gates below pass and the
matched AutoRound ABBA median is strictly below 2544.043 ms and strictly above
1610.04 tokens/s.  It is not a claim that the historical 1973 tokens/s vLLM
result is comparable or has been met.

## 2. Current contract and root cause

For a v-head `h` and an intra-chunk tile of live length `L <= 64`, current
prefill computes:

```
A[i,j] = beta[i] * dot(k[i], k[j]) * exp(gc[i] - gc[j])   for i > j
T      = (I + A)^-1
T[i,j] = -A[i,j] + sum_{l=j+1}^{i-1} (-A[i,l]) * T[l,j]  for j < i
T[i,i] = 1.0f
T[i,j] = 0.0f                                             for j > i
```

The sum is a single fp32 accumulator, `l` is ascending, and every add after
the initial `-A[i,j]` is an explicit `fma`.  `T` is fp32.  Rows at and beyond
the tail (`i >= L`), including their diagonal, are exactly zero.  These are
not cosmetic conventions: `pf_gdn_wu` relies on exact upper-triangle zeroes,
and `gdn_wy_test` asserts both the live diagonal and the dead tail exactly.

The sign is **`(I + A)^-1`**, not `(I - A)^-1`.  FLA stores positive `A` and
negates its strictly lower half when staging the solve.  The current kernel at
`src/kernels/prefill/pf_gdn_wy.cl:486-543` and CPU reference at
`tests/prefill/gdn_chunk_ref.h:342-375` implement that contract.  The WY file
header (`pf_gdn_wy.cl:1-3`) and the launch comment (`src/runtime/prefill/gdn.cc:126`)
still say `(I - A)^-1`; correcting those comments is a required accompanying
documentation/source-comment edit during implementation.  It does not change
arithmetic or runtime behavior.

Current `pf_gdn_solve` uses grid `(48 heads, ceil(C/64), 1)` and a 64-work-item
group: one lane per output column `j`.  It stages the negated 64x64 input into
16 KiB SLM, then runs 64 row steps.  At each row it has two local-memory
barriers:

1. every lane must finish reading the original `As[i,*]` row before a lane
   overwrites an element of that row with `T[i,*]`;
2. every write of `T[i,*]` must complete before row `i+1` reads it.

Thus it executes 128 barriers per full matrix.  The recurrence has real
row-to-row dependence, but the first barrier is an artifact of in-place SLM
storage.  Work is also imbalanced: row `i` has only `i` live lanes and the
slowest lane carries an `i-1`-long FMA chain.  A full matrix has 41,664 scalar
FMAs.  Existing profiling reported 0.7528-0.7932 ms per layer/chunk and
described the solve as 0.18 TFLOP/s, explicitly naming the two barriers per row
(`docs/prefill-gdn-wu-conv-2026-09-05.md:605-612` and
`docs/prefill-gdn-scan-dpas-2026-09-05.md:868-896`).  The latter value projects
to 76.1 ms over 96 layer/chunk instances at pp4096; a newer profile should
establish the actual control before judging this experiment.

## 3. Considered approaches

### A. Private-column register solve - selected

Stage immutable negated `A` in the current 16 KiB SLM allocation.  Make one
work-item own an entire output column `j`, retaining the column's prior `T`
values in private registers.  No work-item consumes another work-item's `T`, so
there is one initial staging barrier and no row barriers.  It preserves every
input, loop bound, FMA association, fp32 output, and tail convention.

This is selected because it attacks the synchronization root cause without
altering numerical order or increasing SLM use.  The cost is register pressure:
each work-item needs up to 64 fp32 private values in addition to loop state.

### B. Immutable SLM `A` plus a second SLM `T` buffer - not selected

Keep an immutable 16 KiB `A` array and a separate 16 KiB output `T` array.  It
removes the read-before-overwrite barrier but still needs one barrier per row.
It can preserve the scalar arithmetic order, but 32 KiB SLM risks materially
lower residency, and a 1.3-2x improvement is unlikely to reliably erase the
35.425 ms whole-request gap.  It is a possible attribution control if approach
A spills too heavily, not a fallback bundled into this experiment.

### C. Block-recursive inverse / FP32 or split-BF16 tiled products - deferred

A blocked triangular inverse could reduce synchronization to block stages and
potentially use tiled matrix products.  It necessarily changes reduction trees;
using BF16 DPAS additionally changes operand precision.  The analogous
single-BF16 scan was fast but failed a determined-token gate
(`docs/prefill-parity-2026-09-20.md:241-254`).  This is a different numerical
project requiring a fresh design, pre-registration, and full-model gates.  It
is not an implementation alternative for this patch.

## 4. Selected kernel algorithm

Add a second entry point, `pf_gdn_solve_register`, to the existing
`src/kernels/prefill/pf_gdn_wy.cl` module.  It retains `WG_SOLVE == 64`, the
same grid, arguments, and `c_count` tail calculation as `pf_gdn_solve`.

For each work-group:

1. Compute `h`, `chunk`, `base_m`, `L`, and `j` exactly as the current entry.
2. Cooperatively stage all 4096 words into `__local float As[CT*CT]` using the
   exact current conversion: `As[p] = pi > pj ? -At[p] : 0.0f`.
3. Execute exactly one `barrier(CLK_LOCAL_MEM_FENCE)` after staging.
4. Give each work-item private storage `float t[CT]`.  For `i=0..L-1`:
   - if `j < i`, initialize `acc = As[i*CT+j]`, then evaluate
     `for (l=j+1; l<i; ++l) acc = fma(As[i*CT+l], t[l], acc)` with ascending
     `l`, and set `t[i] = acc`;
   - if `j == i`, set `t[i] = 1.0f`;
   - otherwise perform no arithmetic and do not read `t[i]` later.
5. Cooperatively write the complete row-major `At`: a live lower entry is
   `t[i]`; a live diagonal is `1.0f`; upper entries and every entry whose row is
   not live are `0.0f`.

There are no barriers inside the `i` loop.  This is correct because `t[l]` is
private to the same column-owning work-item that reads it.  `As` is immutable
after staging, so no lane can race a read of `As[i,l]` with a write of `T[i,l]`.
The explicit `fma`, initial accumulator, and increasing `l` order are
mandatory: replacing them with a vector reduction, a tree, a dot intrinsic, or
an algebraically equivalent reassociation is outside this design.

The implementation must not rely on uninitialized tail entries of `t`.  It
must write the full 64x64 output with the structural conditions above.  It must
continue to use fp32 and must not use atomics, BF16 conversion, DPAS, or a
second launch.

## 5. Runtime selection and interfaces

The existing module remains `kernels::pf_gdn_wy_variant()`; no CMake target or
kernel-file variant is added.  `pf_gdn_solve_register` has the identical
signature to `pf_gdn_solve`:

```
(__global float* A, uint c_count)
```

Add a private `gdn_solve_entry()` adjacent to `gdn_scan_entry()` in
`src/runtime/prefill/gdn.cc`.

- `B70_PREFILL_GDN_SOLVE` unset, empty, or `vector` selects the existing
  `pf_gdn_solve`.
- `B70_PREFILL_GDN_SOLVE=register` selects `pf_gdn_solve_register`.
- Any other value throws `std::runtime_error` before buffers are read or any
  GDN launch is appended; the diagnostic names the variable and allowed values.
- Selection is function-local static, resolved once per process.  Changing the
  variable between engine calls is unsupported because recorded command lists
  retain kernel handles.  Selector comparisons run in separate processes.

Resolve both scan and solve selectors at the start of `gdn_chunk`, before
launch 1, and replace only the launch-6 literal in `gdn.cc:127` with the
selected entry.  The default stays the current solve.  The ten-launch contract,
profile phase `kGdnSolve`, replay behavior, and all buffer interfaces remain
unchanged.

## 6. Correctness, testing, and mutation proof

The numerical requirement is stronger than the existing end-to-end band:
`pf_gdn_solve_register` must be **bit-identical** to `pf_gdn_solve` for the
same device-produced `A`.  This follows from the preserved operation order and
is the first gate, not an aspirational outcome.

Implement tests red-first in this order.

1. Extend `tests/prefill/gdn_wy_test.cc` to run both solve entries from
   independently filled, identical device `A` buffers at `C=256`, `100`, and
   `4096`; require a full `memcmp` of `T`.  Keep the existing fp64
   `(I + A)T == I` derived residual and exact diagonal/upper/dead-tail checks at
   `gdn_wy_test.cc:241-296`.  Do not loosen its residual bound.
2. Run `gdn_chunk_test` under both selector values.  Its existing 4096-position
   CPU recurrent and device-step diagnostic bands, C=1 degeneration,
   multi-chunk equality, 64+36 tail boundary, and repeat byte determinism stay
   unchanged (`tests/prefill/gdn_chunk_test.cc:1-34` and `340-465`).  Preserve
   its current `max_rel < 2.5e-1` transcription tripwire; it is not a new
   acceptance tolerance.
3. Add CTest selector registrations parallel to the current scan registrations
   in `tests/CMakeLists.txt:198-205`: invalid solve selector, default/vector
   dispatch, and register dispatch.
4. Invalid-selector tests must retain the existing real-boundary proof:
   `gdn_chunk` throws before all launches and leaves nonzero state/ring sentinels
   bytewise unchanged (`gdn_chunk_test.cc:230-257`).

Because correct vector and register entries deliberately produce identical
bytes, output comparison cannot prove runtime dispatch.  The selector test must
be mutation-sensitive without exposing a public test-only resolver: use
`KernelCache::kernels()` (`src/runtime/prefill/kernels.h`) around a real
`gdn_chunk` call.  After a `register`-selected call, requesting
`pf_gdn_solve_register` from the same cache must not add a kernel; after a
default/vector-selected call, that request must add exactly one.  Conversely,
requesting `pf_gdn_solve` distinguishes default/vector.  Temporarily
hard-wiring the register selector to the old entry must fail this test.  The
test continues to assert ten runtime launches, so cache observation cannot
hide a launch-count change.

Checkpoint acceptance uses only the primary AutoRound snapshot.  Required
model gates are L0 `prefill_gate`, `prefill_consistency`,
`prefill_determinism`, and `prefill_replay`; the latter snapshots exact
persistent state, ring, KV, control, and logits and covers two full chunks plus
a ragged 4097-token tail (`tests/prefill/prefill_replay_test.cc:1-116`).  RTN is
retired and must not be built, selected, used as a performance reference, or
used to relax any gate.

## 7. Measurement, evidence, and rollback

Before the ABBA benchmark, take one diagnostic AutoRound pp4096 profile on the
same GPU-0 controlled setup for `vector` solve plus `dpas_split` scan, then the
same process setup with `B70_PREFILL_GDN_SOLVE=register`.  The pre-registered
kernel target is **at most 0.35 ms/layer/chunk** for `gdn_solve`.  This is a
diagnostic target, not a substitute for end-to-end timing.  It corresponds to
at most 33.6 ms across 96 instances.  Relative to the 76.1 ms historical
profile control this is a 2.27x reduction; the fresh control profile records
the exact reduction needed to overcome the observed 35.425 ms whole-request
gap.

Required compiler evidence for the selected entry is the generated kernel
assembly/disassembly or compiler resource report showing:

- selected SIMD width and work-group realization;
- GRF/register allocation per work-item;
- SLM use remains 16 KiB, not 32 KiB;
- no private-memory/scratch spill, or a quantified spill if unavoidable;
- no unexpected row-loop local-memory barriers;
- explicit FMA sequence is retained in the hot recurrence.

If register pressure causes spills or lower residency such that solve misses
0.35 ms/layer/chunk, record the evidence and leave the default unchanged.  Do
not tune work-group width, add a DPAS path, change chunk size, change arithmetic
order, or combine this with W/U work in the same task.

If any bitwise solve comparison, structural check, GDN chunk test, checkpoint
gate, determinism gate, or replay byte comparison fails, the register entry is
not adopted and the default remains vector.  If model gates pass but ABBA does
not meet the strict vLLM criterion, the entry remains opt-in/experimental or is
removed according to the implementation review; it is never made default based
on a profile-only win.  A measured token difference is a failure, not a reason
to widen a tensor band.

The final performance decision uses the existing serialized warm ABBA protocol:
one first-use request excluded, three warm requests per process/mode, aggregate
as documented for the scan experiment.  It reports both wall median and
tokens/s, records the exact environment and commit, and compares only with the
matched 2544.043 ms / 1610.04 t/s vLLM row.  Profiled timestamp rows diagnose
the solve and are not arithmetically substituted into whole-request results.

Relevant evidence is retained in:

- `docs/prefill-gdn-scan-split-2026-09-20.md` - current ABBA result, selector
  precedent, and profile caveat;
- `docs/prefill-parity-2026-09-20.md` - matched setup, vLLM trace boundaries,
  and AutoRound-only decision context;
- `docs/prefill-gdn-wu-conv-2026-09-05.md` and
  `docs/prefill-gdn-scan-dpas-2026-09-05.md` - current solve attribution and
  why DPAS/rounding changes are out of scope;
- `tests/prefill/gdn_chunk_ref.h` and `tests/prefill/gdn_wy_test.cc` - the
  authoritative operation order and direct kernel contract.

## 8. Implementation file map

| File | Required change |
|---|---|
| `src/kernels/prefill/pf_gdn_wy.cl` | Add `pf_gdn_solve_register`; correct stale `(I-A)` header wording; retain current entry as default/reference. |
| `src/runtime/prefill/gdn.cc` | Add private solve selector, validate it before launches, route launch 6 only, correct stale launch comment. |
| `tests/prefill/gdn_wy_test.cc` | Add direct old-vs-register bitwise solve comparison while retaining algebraic and tail checks. |
| `tests/prefill/gdn_chunk_test.cc` | Add real selector-boundary/cache mutation proof and run the existing end-to-end checks under both selector processes. |
| `tests/CMakeLists.txt` | Register solve selector CTests and environments. |
| `docs/prefill-gdn-solve-register-results.md` | Record pre-registration control, compiler evidence, correctness outcomes, profile, ABBA data, and decision after implementation; this file is created only when results exist. |

No other production interface is changed.  In particular, `pf_gdn_wu`, scan
selection, replay APIs, and any attention/fusion source files must remain
untouched by the implementation.

## 9. Amendment - 2026-09-22

This design is now **stage S4** of
`2026-09-22-prefill-parity-program-design.md`, which reorders the work ahead of
it: the linear path carries a **measured** 274.5 ms dequant round-trip
(`docs/prefill-parity-2026-09-20.md`, "The linear path, itemised"), which is
larger than this whole GDN stage and bitwise-safe to remove. Solve stays worth
doing; it is no longer first.

Three corrections to the text above.

**(a) §3's rejection of approach B rests on a false premise.** B is described as
"still needs one barrier per row". It does not. In `pf_gdn_wy.cl:533` lane `j`
reads `As[l*CT+j]`, which is `T[l][j]` - the value that same lane wrote at step
`l`, since `As[i*CT+j] = acc` writes only column `j`. The T dependency is
entirely intra-lane and no barrier ever protected it. Both existing barriers
exist for the **A** read: lane `j` reads `As[i*CT+l]` for `l > j` while lane `l`
overwrites that element. Give A and T separate storage and both barriers
disappear - in B exactly as in A.

Consequently B is barrier-free too, costs 16 KiB more SLM, and **cannot spill**.
A's private `float t[CT]` is indexed by a runtime `l` whose loop bounds
(`l < i`, `i < L`) are not compile-time known, which is the canonical pattern
IGC lowers to scratch; scratch would be slower than the SLM read it replaces.
**Implement B first, or both behind the same selector** - they are arithmetically
identical, and the compiler-evidence gate in §7 then decides between them on
measured spill rather than on prediction.

**(b) §4 step 5 would regress write coalescing.** Today the kernel ends with
`for (p = j; p < CT*CT; p += WG_SOLVE) At[p] = As[p]` - coalesced. Having each
lane write its own column is a stride-`CT` global write. Fix, without touching
arithmetic: after the `i` loop, stage the column back into the now-dead `As`
(one barrier), then keep the existing coalesced copy.

**(c) §1's acceptance rule is decoupled.** "Accepted only if ... the matched
AutoRound ABBA median is strictly below 2544.043 ms" would reject a genuine
component win for not clearing the whole vLLM gap alone, leaving nothing to stack
with S2 and S3. Acceptance is now: bitwise identity to `pf_gdn_solve`, the model
gates, and the pre-registered **≤ 0.35 ms/layer/chunk** kernel target against a
fresh control profile. The vLLM comparison is a program milestone, checked after
the stage lands, not this stage's gate.

Unchanged: the fp32 arithmetic, the operation order, the tail conventions, the
selector pattern, the mutation-sensitive dispatch test, and the rule that a
measured token difference is a failure rather than a reason to widen a band.
