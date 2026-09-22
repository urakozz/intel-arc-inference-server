# Prefill parity program - the ordered work to vLLM and beyond

**Status:** design, 2026-09-22, for operator review. Supersedes the *ordering* in
`2026-09-22-gdn-solve-register-design.md`; that document remains the design of
one stage (S4) and is amended, not withdrawn.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Where we start

Matched warm ABBA protocol, GPU 0, `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`,
4096 ids as two 2048-token chunks
(`docs/prefill-gdn-scan-split-2026-09-20.md`, `docs/prefill-parity-2026-09-20.md`):

| path | wall | throughput |
|---|---:|---:|
| native L0, split-BF16 scan (current best) | **2579.468 ms** | **1587.924 t/s** |
| vLLM, same checkpoint/ids/chunk | 2544.043 ms | 1610.040 t/s |

Native L0 GPU time is **2493.6 ms** against vLLM's **2514.027 ms** of kernel
time: we are already ahead on kernel execution and 35.425 ms behind on the wall.

**The attributed partition** (native, split-scan profile; vLLM trace boundaries
differ and this is not a subtraction):

| work | native | vLLM |
|---|---:|---:|
| `slab_gemm` (the DPAS matrix math) | 1344.3 | - |
| `slab_dequant` (the round-trip this program deletes) | **274.5** | - |
| int4 linears, total | 1618.8 | 2116.9 |
| GDN narrow core | 351.8 | 141.8 |
| attention QK/softmax/PV | 148.2 | 63.1 (flash) |
| SiLU/multiply | 98.7 | 51.3 |
| residual/norm | 104.2 | fused into vLLM's 125.3 pointwise family |

**The derived floor that orders this program.** Linear FLOPs per request are
2 x M x SUM(K.N) over 64 layers = **199 TFLOP**; at the 156.53 TFLOP/s measured
for `pf_gemm`, the floor is **1278 ms**. Measured `slab_gemm` is 1344.3 ms - 5%
above its own floor. The matrix math is done. What remains in the linear path is
274.5 ms of memory round-trip.

## 2. Goal, bars, stopping rule

**Goal:** prefill faster than the matched vLLM row, then as far past it as the
measured headroom allows.

**Program bars:**
1. **Parity:** ABBA median strictly below **2544.043 ms** (> 1610.040 t/s).
2. **Stretch:** ≥ **1900 t/s** (≤ 2155.8 ms), the operator's "great" band.
3. **Nothing moves numerically without a gate.** Every stage is either
   *bitwise-identical by construction* or carries the full model-gate set, and
   says which it is before implementation.
4. **Decode untouched**: 774 launches / 19 modules, `golden_gate_test` 93/93.

**Stage acceptance is per stage, not per program.** A stage is accepted on its
own gate and its own measured component saving. It is **not** rejected for
failing to clear the whole vLLM gap alone - that coupling, in the solve spec's
original §1, would have discarded every individual win and left nothing to
stack. The program bars are milestones, checked after each stage lands.

**Stopping rule:** a stage whose measured saving is below half its estimate, or
whose gate fails, is reverted and recorded. No tuning pass without a ruling.

## 3. The stages, in order

Ordered by expected ms per unit of numerical risk, not by pipeline position.

### S1 - fused int4 dequant in `pf_gemm` (probe first)

**Target: −198 to −270 ms.** Mechanism: the GEMM reads int4 weights and produces
the identical `rne_bf16((float)qm8 * scale)` word in-kernel, so the 48.7 GB bf16
slab round-trip per chunk disappears and the linear path issues one launch per
linear instead of two per slab (−7,616 launches per request).

**Numerical class: bitwise-identical by construction.** Same word, same k order,
same accumulator. Gate: `memcmp` of C against the two-pass path, then the full
`pf_gemm_test` cell matrix, then the four model gates.

Probe design: `2026-09-22-fused-dequant-gemm-probe-design.md`. Break-even is
122.9 TFLOP/s effective; the probe is a failure below it and that is recorded
rather than tuned away. **Nothing downstream in this program depends on S1**, so
a rejection costs the program only its own estimate.

### S2 - epilogue fusion: SiLU in the gate‖up GEMM, bf16 `partials` elsewhere

**Target: −80 to −95 ms** (the two overlap; see below). Two mechanisms:

- **SiLU into the gate‖up epilogue.** `pf_silu_mul` reads `partials` and writes
  bf16 `x`. Its gate and up operands are 16 columns apart (`gflat = (k/16)*32 +
  k%16`, `uflat = gflat + 16`), so both live inside any 256-aligned N tile: the
  epilogue can compute `rne_bf16(bf16f(s_b) * bf16f(u_b))` locally and never
  write the fp32 [2048][34816] intermediate at all.
- **bf16 `partials` for the remaining linears.** Every consumer's first act on
  `partials` is already `rne_bf16` - verified in `pf_res_fold` (at S_PREV = 1),
  `pf_attn_prep`, `pf_silu_mul`, `pf_gdn_conv`, `pf_attn_gate` - so storing the
  rounded word halves that traffic with no value change.

**Numerical class: bitwise-identical by construction**, conditional on S_PREV = 1,
which is the prefill configuration. Gate: `memcmp` plus `prefill_replay`.

**Two costs to plan for.** These pointwise kernels are **shared with decode**,
where the S slices are a split-K sum and early rounding would *not* be identical:
this needs prefill-only kernel variants, and `pf_prep_test` currently pins the
M = 1 output bit-identical to the decode binaries. And the two mechanisms overlap
on gate‖up, the largest N: fusing SiLU removes those partials entirely, so the
savings must be measured together, never added.

### S3 - causal row-block attention

**Target: −33 ms.** Chunk 2 computes a full 2048 x 4096 QK^T when causality needs
roughly half of it. Split the launch into row blocks, each with N limited to
`pad256(pos + block_end)`.

**Numerical class: bitwise-identical by construction** - the skipped score
columns are exactly the ones `pf_softmax_causal` writes as +0.0 and P·V then
multiplies by KV rows, contributing nothing to an fp32 sum. **Runtime only**: no
kernel source changes, which makes this the cheapest stage to try and to revert.

### S4 - GDN solve, register/immutable-A

**Target: −40 ms.** Design: `2026-09-22-gdn-solve-register-design.md`, as amended
by that file's §9. Numerical class: bitwise-identical (fp32, same FMA order).

### S5 - GDN W/U on DPAS - last, and only if the program still needs it

**Estimated −60 ms, and the only stage that moves a rounding point.** `pf_gdn_wu`
is matrix-shaped and therefore DPAS-able, but single-BF16 is exactly the class
that failed a determined-token gate in the scan experiment (92/93, state cosine
0.998499 < 0.999); a split representation of T is the minimum, and three limbs
may be needed.

**Numerical class: gated.** Full model-gate set with every printed `gdn_state`
cosine strictly above 0.999, plus `prefill_determinism` bitwise, on the AutoRound
checkpoint. If S1-S4 have already cleared the program bars, S5 is optional and
should stay opt-in.

## 4. What the program is expected to reach

| | conservative | optimistic |
|---|---:|---:|
| S1 fused dequant | −198 | −270 |
| S2 epilogue fusion | −80 | −95 |
| S3 causal attention | −33 | −33 |
| S4 GDN solve | −40 | −45 |
| S5 W/U DPAS (gated) | 0 | −60 |
| **total** | **−351** | **−503** |
| **wall** | 2228.5 ms | 2076.5 ms |
| **throughput** | **1838 t/s** | **1972 t/s** |

All derived from the measured partition. **Parity (bar 1) is reached by S1 alone**
- the gap is 35.425 ms and S1's conservative estimate is five times that. The
stretch bar needs S1 plus any two of S2-S4.

**2100 t/s (1950.5 ms) is not reachable from this list**, and saying so now is
cheaper than discovering it later: it requires going below the linears' 1278 ms
DPAS floor plus the rest of the walk, which needs either a larger chunk (out of
scope, estimated −116 ms) or a change to a rounding point. Both memos behind this
program agree on that.

## 5. Out of scope

Decode and its replayed list; checkpoint formats and quantisation; chunk size;
vendor primitive adoption (the exact-pin oneDNN int4 path fails the strict
weight-identity gate and is slower than ours); the transposed-B attention GEMM's
arithmetic; RTN, which is retired.

## 6. Measurement protocol

Unchanged from the scan experiment, because the comparisons must remain
commensurable: serialized warm ABBA, one first-use request excluded, three warm
requests per process/mode, both wall median and t/s, the harness's grade word
quoted verbatim, GPU 0, zero DRM holders and zero containers proven immediately
before the first timed row. Profiled rows diagnose a stage and are never
substituted into a whole-request result.
