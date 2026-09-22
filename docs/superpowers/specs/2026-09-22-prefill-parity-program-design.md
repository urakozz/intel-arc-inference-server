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

## 7. Amendment - 2026-09-22, after the S1 probe

**S1 is rejected on measurement.** `docs/probe-fused-dequant-2026-09-22.md`:
both fused variants are bitwise identical to the two-pass path, and both are
slower. On gate‖up at M = 2048 the control is 5.341 ms (136.72 TFLOP/s: dequant
0.826 + GEMM 4.515); registers-only is 11.117 ms (65.68) and SLM-staged is
7.274 ms (100.37). Fusion **adds 2.759 ms to the GEMM half to delete a 0.826 ms
pass**, and that pass already runs at ~546 GB/s of the 608 GB/s peak. The
dequant was never the expensive part - it was the *visible* part.

The 274.5 ms in-situ figure stands as a measurement; what was wrong was the
inference that removing the round-trip would recover it. Reading the weights is
work the GEMM must do either way, and doing it inside the mainloop costs more
than doing it once into memory the DPAS pipeline streams back efficiently.

**What this does to the program.**

| | conservative | optimistic |
|---|---:|---:|
| ~~S1 fused dequant~~ | **rejected** | **rejected** |
| S2 epilogue fusion | −80 | −95 |
| S3 causal attention | −33 | −33 |
| S4 GDN solve | −40 | −45 |
| S5 W/U DPAS (gated) | 0 | −60 |
| **total** | **−153** | **−233** |
| **wall** | 2426.5 ms | 2346.5 ms |
| **throughput** | **1688 t/s** | **1745 t/s** |

**Bar 1 (parity) still falls out of S2 alone** - 2499.5 ms, past the 2544.043 ms
vLLM row. **Bar 2 (1900 t/s) is no longer reachable from this list**: it needs
−424 ms and the list now tops out at −233. Saying so is the point of writing
estimates down before measuring.

**Where the remaining headroom actually is**, from the measured partition, after
the designed stages take their share:

| family | native | vLLM | claimed by S2-S5 | still unclaimed |
|---|---:|---:|---:|---:|
| GDN core (scan 127.2 of it, vs vendor forward 47.6) | 351.8 | 141.8 | ~100 | **~110** |
| attention | 148.2 | 63.1 | 33 | **~52** |
| pointwise + residual/norm | 202.9 | 125.3 | ~80 | **~0-40** |
| non-kernel wall (17,383 launches) | ~86 | 23.6 | 0 | **~60** |

Reaching 1800-1900 t/s therefore requires **new stages, not better versions of
these**: a second pass at the GDN scan against the vendor's 47.6 ms rate, a real
flash-attention rather than the causal row-block trim, and launch-count
reduction. Each needs its own design and its own gates; none is in this spec yet.

**And the linears are finished in bf16.** The probe's control measures
`pf_gemm` at 161.7 TFLOP/s on its best shape - the bf16 DPAS rate, not a
bandwidth limit, and the reason no memory-side trick can move it. The linear
path is 1618.8 ms of a 2493.6 ms GPU total, so the single largest remaining
lever in the engine is **lower-precision matrix math**: this driver exposes
native `i8_i8` DPAS at K = 32 and `i4_i4` at K = 64 against bf16's K = 16, and
the group-64 quantisation already aligns with the int4 K depth. That is a
**quantisation project**, not a kernel optimisation: both DPAS operands must be
integer, so it means W4A8 or W4A4 activations, a new accuracy study, and a new
CPU oracle - every current gate compares against the bf16 path by construction.
It is recorded here as the fork it is, not proposed.

## 8. Amendment - 2026-09-22, after S3 landed

**S3 delivered −14.8 ms, not −33.** Measured on device 1, diagnostic profile,
4096 ids: `attn_QK^T` 59.0 → 44.2 ms, `attn_PV` unchanged at 31.7. Bitwise
identical - `prefill_backend_equivalence_test` passes all five families with
**0 words differing and 0 sign-of-zero** against the untouched sycl-tla path,
and `prefill_gate_l0_test` is 93/93.

The estimate was wrong in a way worth recording: −33 ms was read off the whole
attention family (148.2 ms), but causal trimming only touches the QK^T half
(59.0 ms), and the 256-row granularity caps the saving near the theoretical 25%
for the second chunk. The measured 25% of QK^T is what the mechanism can give.

**Ruling S3-a (deviation, accepted).** The stage as written blocked **both**
GEMMs. Blocking P·V was measured at **+19.7 ms net loss** (31.9 → 66.3) and was
rejected: P·V's N axis is a single 256-wide tile, so splitting M leaves six
work-groups per launch and starves the machine. The implementer committed
QK^T-only and reported the deviation with its measurement instead of shipping
a regression. That is the behaviour the stopping rule is for.

**Ruling S3-b (stopping rule, kept).** §2 makes a stage delivering below half
its estimate an operator call. S3 is kept: it is **bitwise identical**, the
saving is real, and nothing downstream depends on its size. Two qualifications
travel with it:
- it costs **+1,792 launches per request** and +4.0 ms of host submission,
  against a walk that already spends ~86 ms outside kernels - so the GPU saving
  is an upper bound on the wall saving;
- **no whole-request ABBA has been taken** (device 0 was busy). Until one is, S3
  counts as −14.8 ms of GPU time, not as a throughput result, and it does not
  appear in any record as t/s.

`step_chunk_launches` now takes `C`: 8689 at C ≤ 256, **9137** at C = 2048.

**Revised program expectation.**

| | conservative | optimistic |
|---|---:|---:|
| S2 epilogue fusion | −80 | −95 |
| S3 causal QK^T (**measured, GPU**) | −14.8 | −14.8 |
| S4 GDN solve | −40 | −45 |
| S5 W/U DPAS (gated) | 0 | −60 |
| **total** | **−134.8** | **−214.8** |
| **wall** | 2444.7 ms | 2364.7 ms |
| **throughput** | **1675 t/s** | **1732 t/s** |

Parity (bar 1) still falls out of S2 alone. Bar 2 remains out of reach for this
list, as §7 already recorded.

## 9. S3's whole-request row - 2026-09-22, RECORD

Taken after the S3 commit on a provably idle box (0 DRM holder fds, 0
containers, load 0.43), device 0, `tools/bench_decode.sh --pp 4096`, median of 3,
same harness and same IGC 2.41.5 as the morning's baseline, so S3 is the only
variable.

| | wall | throughput | spread | grade |
|---|---:|---:|---:|---|
| before S3 (`docs/BENCHMARKS.md`, IGC 2.41.5 re-measure) | 2732.5 ms | 1498.97 t/s | 0.43% | RECORD |
| **with S3** (`779097a`) | **2721.3 ms** | **1505.16 t/s** | 0.09% | RECORD |

**−11.2 ms, +6.19 t/s.** The stage's own profile predicted −14.8 ms of GPU time
against +4.0 ms of added host submission, i.e. −10.8 ms; the wall moved −11.2.
Prediction and measurement agree to within a millisecond, which is the evidence
that the +1,792 launches are priced correctly rather than merely acknowledged.
Ruling S3-b stands: kept.

## 10. Amendment - 2026-09-22, after S2 landed

**S2(a) delivered −84.0 ms; S2(b) was rejected on measurement.** Device 1,
diagnostic profile, 4096 ids in two 2048 chunks, both arms of each mechanism
interleaved through a selector so one binary and one session separate them
(`B70_PREFILL_SILU_FUSED`, and a `B70_PREFILL_BF16_PARTIALS` that did not
survive). L0 GPU ms, medians:

| phase | unfused (n = 10) | **S2(a) fused** (n = 13) | delta |
|---|---:|---:|---:|
| `slab_gemm` | 1375.6 | 1408.2 | **+32.6** |
| `slab_dequant` | 272.0 | 252.4 | **−19.6** |
| `silu` | 99.1 | **0.0** | **−99.1** |
| `norm` | 105.3 | 105.7 | +0.4 |
| **L0 GPU total** | **2668.6** | **2584.6** | **−84.0** |

Inside §3's −80 to −95 band - from **one** of the two mechanisms. Bitwise:
`prefill_backend_equivalence_test` passes all five families with 0 words
differing and 0 sign-of-zero against the untouched sycl-tla path, and
`pf_gemm_test` compares the fused epilogue to the GEMM + `pf_silu_mul` pair
device-to-device at M = 2048 and M = 772, also 0. `prefill_gate_l0_test` 93/93.
Unlike S3 this stage **removes** launches - 128 per request, one per layer per
chunk - so the wall saving is not bounded below the GPU saving by host
submission. `step_chunk_launches` is 9073 at C = 2048 and 8625 at C ≤ 256.

**Ruling S2-a (the estimate was right for the wrong reason).** §3 expected the
saving to be the deleted fp32 [2048][34816] traffic. It is not: the GEMM's own
row went **up** 32.6 ms, because the epilogue now evaluates one `exp` and four
`rne_bf16` per output element - 4.56 G of them per request - while the store is
free either way (64 scalar ushort stores and eight 16b block messages were both
built and measured against the same control, 1410.2 vs 1409.3 ms). What the
stage actually banks is the whole `silu` row plus 19.6 ms of `slab_dequant`,
which speeds up once 285 MB of dirty partials per layer stop competing with it.
A second-order effect nobody predicted is 24% of the stage.

**Ruling S2-b (rejected, and it cannot be tuned).** bf16 `partials` for
out_proj / o_proj / down was built, gated bitwise (equivalence test green) and
measured over five runs an arm: **−7.3 ms `slab_dequant`, −1.7 ms `norm`,
+9.3 ms `slab_gemm`; net zero** (TOTAL +1.3 on means, +8.9 on medians, both
inside the arms' own spreads). The reason is a driver fact, not a tuning knob:
`ocloc` declares no 16-bit 2D block write wider or taller than
`..._16b_8r16x1c`, so a bf16 C tile needs exactly as **many** store messages as
the fp32 one, each carrying half the bytes - and this epilogue is bound by the
message count, not the bytes. Halving the rectangle buys nothing on the write
side, and the read side is worth 1.7 ms of a 105 ms `norm` row, which says the
pointwise kernels were never bandwidth-bound on `partials`.

Extending it to qkv‖z and q‖k‖v was **not** built: those two write 3x the
columns of the three above, so the GEMM-side cost scales to roughly +28 ms
against consumer rows (`gdn_conv` 32.7, `attn_prep` 14.0, `attn_gate` 9.1) whose
measured analogue gave back 1.6%. The code was reverted; the finding is recorded
in `pf_gemm.cl` and `pf_prep.cl` so it is not re-derived.

This is S1's shape a second time: the traffic was real, and removing it was not
what made the walk faster.

**Revised program expectation.**

| | conservative | optimistic |
|---|---:|---:|
| S2(a) SiLU epilogue (**measured, GPU**) | −84.0 | −84.0 |
| ~~S2(b) bf16 partials~~ | **rejected** | **rejected** |
| S3 causal QK^T (**measured, wall**) | −11.2 | −11.2 |
| S4 GDN solve | −40 | −45 |
| S5 W/U DPAS (gated) | 0 | −60 |
| **total** | **−135.2** | **−200.2** |
| **wall** | 2444.3 ms | 2379.3 ms |
| **throughput** | **1676 t/s** | **1721 t/s** |

**No whole-request ABBA has been taken for S2** - device 0 belonged to another
stage - so S2(a) counts as −84.0 ms of GPU time and appears in no record as
t/s. Bar 1 (parity, 2544.043 ms) still falls out of S2 alone on this arithmetic
and still needs that row to be claimed.

## 10. The split-scan verdict is VOID - device 1 was wedged (2026-09-23)

§8's revert of the split-scan default rests on evidence that cannot be trusted.
The card it was measured on had failed.

**What was found.** Every binary in the tree segfaults with zero output when run
with `ZE_AFFINITY_MASK=1`, and passes with the mask unset or `=0`:

| mask | `buffers_test` |
|---|---|
| unset (device 0) | OK |
| 0 | OK |
| **1** | **SEGFAULT** |

`gdb` puts the crash inside the driver, not our code: `l0::Context::Context` →
`zeInit()` → `libze_intel_gpu.so.1` → SIGSEGV in a `pthread_once` init path.
`clinfo` now enumerates **one** B70 where there are two, and `/dev/dri/card2`
(08:00.0 = device 1) was recreated at **23:31 on 2026-09-22**. Both cards are
still bound to `xe`; no package changed in two days; OpenCL and `sycl-ls` work on
device 0. The device is present and dead.

**Which results this invalidates.** The split-default suite ran on device 1
between ~23:20 and ~23:31 - across the moment that node was recreated. So:

- the segfault cascade in that run was **device 1 dying**, not the two concurrent
  ctest runs the earlier note blamed;
- `prefill_gate_l0_test`'s 92/93 and the `gdn_state` cosine of 0.996344994, and
  `gdn_chunk_test`'s band miss, were produced **on a failing card**. A GPU
  mid-reset does not produce trustworthy arithmetic.

**The split scan is therefore not proven guilty.** Its −157.974 ms is back in
play and needs a clean retest on a healthy device. The default stays `vector`
until that retest exists - absence of evidence is not evidence of innocence
either.

**What is still trustworthy.** The 86/86 merged-tree suite (~22:50, device 1,
before the failure) and every throughput row in `docs/BENCHMARKS.md`, which were
taken on **device 0** with the mask unset - including the standing
2625.0 ms / 1560.40 t/s.

**Open and unanswered: what killed it.** The split-default suite was the workload
running when the node was recreated, so "the split scan faults the device" is a
live hypothesis and a much more serious one than a cosine miss. The alternative
is that the card was left fragile by the processes killed earlier that evening.
The retest must watch for a second reset, and should not be run on device 0 -
the series card - until device 1 is back and has survived it.

## 11. Correction to §10 - the split scan DOES fail the gate (2026-09-23)

§10 said the verdict was void because the card had faulted. Half right: the card
had faulted, and that explains the segfaults - but not the numbers. Re-run on a
healthy device 0, the failure reproduces **to nine digits**.

| run | scan | cosines: prose / code / cjk | tokens |
|---|---|---|---|
| 2026-09-18, IGC 2.38.2 | vector | 0.999903435 / 0.999189068 / 0.999903208 | 93/93 |
| 2026-09-21, IGC 2.38.2 | "dpas_split" | **identical to the vector row** | 93/93 |
| 2026-09-22 night, device 1 | dpas_split | 0.999910833 / 0.996344994 / 0.999892216 | 92/93 |
| **2026-09-23, device 0** | **dpas_split** | **0.999910833 / 0.996344994 / 0.999892216** | **92/93** |
| 2026-09-23, device 0 | vector | 0.999903435 / 0.999189068 / 0.999903208 | 93/93 |

Three findings, in order of consequence.

**1. The split scan fails `prefill_gate_l0_test`: 92/93, code's L60 `gdn_state`
cosine 0.996344994 against a > 0.999 bar.** Deterministic - the device-1 and
device-0 runs agree exactly, and `prefill_determinism_test` passes under it. The
default stays `vector`, now on evidence rather than on a contaminated run.

**2. The 2026-09-21 record that declared its gates green was measuring the
VECTOR kernel.** Its cosines are bit-identical to the vector row from three days
earlier; approximate split-BF16 arithmetic cannot reproduce fp32 vector code to
nine digits across three prompts. Whatever that run selected, it was not
`pf_gdn_scan_dpas_split`. `docs/prefill-gdn-scan-split-2026-09-20.md`'s
"Available Vishva gates (split selector)" section is therefore **not evidence**,
and the experiment's own decision - "the opt-in split implementation ... and
available primary-model gates are green" - does not hold.

**3. Kernel numerics are stable across the IGC 2.38.2 -> 2.41.5 upgrade.** The
vector cosines are identical before and after the 139-kernel recompile. The
compiler was a plausible suspect for the change and is now excluded.

**What survives of the split experiment.** Its ABBA measured a real -157.974 ms:
the two selectors produced different wall times, so the selector did take effect
in `prefill_replay_test` run directly. So the kernel is genuinely faster **and**
genuinely fails the golden gate - the same shape as the single-BF16 D2 attempt
it was built to fix. It is a fast wrong answer, not a missed opportunity.

**The process lesson, which is the expensive part.** That experiment built a
mutation-sensitive dispatch test *precisely because* output comparison cannot
prove which kernel ran - and then ran its model gates with no such proof,
through a different launch path, and trusted the result. A selector's gate run
needs the same dispatch evidence the unit test has: print the resolved entry
name from inside the process under test, or assert it via `KernelCache::kernels()`
in the gate itself. Without it, "the gates are green" can mean "the gates never
ran your kernel".
