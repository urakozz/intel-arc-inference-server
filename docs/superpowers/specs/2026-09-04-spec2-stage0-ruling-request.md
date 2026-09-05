# Spec 2 Stage 0 - ruling request (2026-09-05)

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

This composition uses the corrected P2 AOT-256-GRF matrix. It is an
**optimistic outer bound**, not a record row: P4 is host-wall timed and lacks
its correctness gates; P5 is blocked and estimated; widened small kernels are
not implemented.

## Composed ceiling

| term | C=2048 ms | C=4096 ms | how priced | grade |
|---|---:|---:|---|---|
| GEMM | 680.062 | 1585.201 | sum corrected P2 production-shape rows across 48 GDN + 16 FA layers | derived from measured P2 |
| dequant | 210.116 | 210.116 | P3's 256 selected [K][N] matrix dequants | derived from measured P3 |
| attention | 576.544 | 775.811 | P4 VTiles=8 host wall, x16 FA layers | measured host-wall, correctness incomplete |
| GDN | 15.365 | 30.474 | P5 blocked own-design price | estimated |
| norms / SiLU / gated-head / attn_prep / embed | 0.000-2532.002 | 0.000-5064.004 | lower bound 0; upper is decode `t1 x M` using 2.012, 1.739, 9.7, 1.6, 3.2, 3.65 µs and launch counts 129/129/64/48/16/1 | derived range |
| interop | 3.290 | 3.290 | P1 8.569 µs wait path x384 | derived from measured P1 |
| lm_head | 4.379 | 4.379 | existing bf16 decode route, once/chunk | measured decode |
| **total** | **1489.756-4021.757** | **2609.270-7673.274** | terms above | mixed, as labelled |
| **throughput** | **509.2-1374.7 t/s** | **533.8-1569.8 t/s** | C / total seconds | derived |

The small-kernel interval is intentionally wide. The lower endpoint assumes
unwritten widened kernels cost nothing; the upper endpoint is the literal
decode-launch cost. L1 must measure between them. Even the optimistic endpoints
are below the external references.

## Comparison and requested ruling

The best optimistic Stage-0 outer bounds are 1374.7 t/s at C=2048 and
1569.8 t/s at C=4096 (derived, iterate inputs). They are below vLLM's **1973
t/s pp4096** (external measured, HTTP-inclusive: upload, parse, tokenisation,
scheduling, prefill, and first streamed byte), while these bounds are intended
to be device-side, loader-excluded, first-token-inclusive. vLLM's default
32-GB scheduling limit makes pp4096 at least two sequential ~2048-token
steps; that asymmetry favours a C=2048 comparison.

They are also below the third-party **2400-2500 t/s** same-model/same-command
report, whose implied **119-124 TFLOP/s** is external. C=4096's optimistic
1569.8 t/s is 79.6% of vLLM and 62.8-65.4% of that third-party reference
(derived); C=2048 is 69.7% and 55.0-57.3% respectively (derived).

At the spec's recommended 90% margin, the provisional bars are **1237.3 t/s**
for C=2048 and **1412.8 t/s** for C=4096 (derived). Because the ceiling is
already below 1973 t/s before any real small-kernel cost, the Stage-1 ladder
must not start without a controller ruling.

Please rule:

1. Is the margin 90%, or another percentage?
2. Is C=2048 or C=4096 the gate width? The corrected P2 matrix favours
   C=2048 for five of six shapes at M=4096 versus M=2048, while the composed
   optimistic total is faster at C=4096.
3. Does the gate quote device-side pp only, or must an HTTP-inclusive number
   be established before publishing a row?
4. Since the current ceiling is under 1973 t/s, does the ladder proceed after
   P4 correctness and a P5 own-kernel gate, or must a re-assessment memo come
   first?
5. Are the outstanding interface choices acceptable: P4 bf16 q output and
   direct-cache strides, and a new own P5 kernel rather than the blocked torch
   header?

No Stage 1 work begins until these are ruled.

## 2026-09-05 addendum - P4 diagnosis and tightened C=2048 ceiling

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held). This is an
addendum: the original C=2048/C=4096 tables above remain the Stage-0 record.

P4's selected hdim=256 FMHA route remains 576.194 ms across 16 FA layers at
C=2048 (measured host-wall, correctness incomplete, iterate grade). The
causal scheduler prunes future KV tiles; the generated image has DPAS and 2D
block loads; and the same-tile hdim=128 comparison reaches 18.719 TFLOP/s,
not the GEMM-like rate that would identify head dimension as the complete cause
(measured time; derived rate, iterate grade). Details:
[P4 diagnosis](../../probe-prefill-attn-2026-09-04.md).

T6 replaced the four largest terms of the former 0.000-2532.002 ms
small-kernel range with direct M=2048 device-timestamp measurements. The
remaining `attn_prep` plus `embed_gather` interval is 0.000-112.333 ms
(derived), so the residual uncertainty no longer controls the verdict. Details:
[T6 small-kernel record](../../probe-prefill-small-2026-09-05.md).

| term | C=2048 ms | how priced | grade |
|---|---:|---|---|
| GEMM | 680.062 | unchanged P2 sum | derived from measured P2, iterate |
| dequant | 210.116 | unchanged P3 sum | derived from measured P3, iterate |
| attention | 576.194 | P4 hdim=256 VTiles=8 host wall × 16 | measured host-wall, correctness incomplete, iterate |
| GDN | 15.365 | unchanged P5 own-design price | estimated, iterate |
| four direct small terms | 360.888 | `res_fold` + `norm_finish` + `silu_mul` + `gated_head` | derived sum of measured timestamps, iterate |
| `attn_prep` + `embed_gather` | 0.000-112.333 | retained lower / literal decode upper bound | derived range, iterate |
| interop | 3.290 | unchanged P1 wait path × 384 | derived from measured P1, iterate |
| lm_head | 4.379 | unchanged once-per-chunk route | measured decode, iterate |
| **total** | **1850.294-1962.627** | terms above | mixed, as labelled, iterate |
| **throughput** | **1043.5-1106.9 t/s** | 2048 / total seconds | derived, iterate |

This tightened device-side, loader-excluded ceiling does **not** clear vLLM's
1973 t/s pp4096 result (external measured, HTTP-inclusive). Even its optimistic
1106.9 t/s endpoint is 56.1% of that reference (derived). The comparison still
has the documented device-side versus HTTP-inclusive asymmetry, but the ceiling
is now below vLLM before the unmeasured residual, P4 correctness work, or P5's
estimated term can improve its grade.

## 2026-09-05 addendum 2 - the small-kernel term MEASURED at S = 1; T6 recomposed

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held). This
supersedes the small-kernel rows of the addendum above **in place**; every other
term is carried forward unchanged and every prior number is kept.

The addendum above priced the small kernels at 360.888-473.221 ms/chunk. That
was a **composition error, not a kernel defect**: those rows were measured on
the *decode* binaries, i.e. at decode's split-K (`prep_silu_mul` folding
`SILU_S` = 8, `prep_res_fold` at SP4), while the prefill path runs **S = 1**
(plan 6b ruling R1). The S = 1 runtime-`M` kernels now exist
(`src/kernels/prefill/pf_*.cl`, graded bit-for-bit against the decode binaries
at M = 1 and against the CPU references at M = 64) and have been measured at
M = 2048 in the same process as the decode control, which re-read 361.319 ms
against its earlier 360.888 (**+0.12%** drift). Details and the scored
pre-registration: [T6 small-kernel record](../../probe-prefill-small-2026-09-05.md).

**All seven kernels are direct rows** - `attn_prep` and `embed_gather` were the
two the addendum above had to bound at 0.000-112.333 ms - so the term is a
number, not an interval:

| kernel | ms/chunk | GB/s achieved |
|---|---:|---:|
| `pf_silu_mul` (S8 → S1: 281.607 → 48.453) | 48.453 | 470.9 |
| `pf_res_fold` SP1 (SP4 → SP1: 49.208 → 23.059) | 23.059 | 470.2 |
| `pf_ab_proj` | 19.320 | 58.0 |
| `pf_norm_finish` (S-independent: 17.079 → 17.617) | 17.617 | 308.5 |
| `pf_gated_head` (S already 1: 13.425 → 13.440) | 13.440 | 449.4 |
| `pf_attn_prep` | 8.347 | 435.2 |
| `pf_embed_gather` | 0.111 | 378.4 |
| **small kernels, C=2048** | **130.346** | (device measures 590) |

That removes **230.542-342.875 ms/chunk** from the composition. It is also a
**miss against both pre-registrations** - the controller's ~60-80 ms by
1.63-2.17×, and the probe's own byte-derived ~85-110 ms by 1.19-1.53× - because
two of the four kernels the correction was expected to shrink cannot be reached
by it at all (`norm_finish` never reads `partials`; `gated_head`'s `GATED_S` is
already 1 in decode), and because a launch that moves fewer bytes sustains a
lower rate (534.8 → 470.9 GB/s), so `silu_mul`'s 6.60× of traffic removed -
its reads fall 8× but its 71 MB write does not - buys 5.81× of time.

### Recomposed ceiling at C = 2048

| term | ms | how priced | grade |
|---|---:|---|---|
| GEMM | 680.062 | unchanged P2 sum | derived from measured P2, iterate |
| dequant | 210.116 | unchanged P3 sum | derived from measured P3, iterate |
| **small kernels** | **130.346** | **seven direct M=2048 device timestamps at S=1** | **measured, iterate** |
| GDN | 15.365 | unchanged P5 own-design price | estimated, iterate |
| interop | 3.290 | unchanged P1 wait path × 384 | derived from measured P1, iterate |
| lm_head | 4.379 | unchanged once-per-chunk route | measured decode, iterate |
| *subtotal, everything but attention* | *1043.558* | | mixed, iterate |
| attention **(a)** FMHA as measured | 576.194 | P4 hdim=256 VTiles=8 host wall × 16 | measured host-wall, correctness incomplete, iterate |
| attention **(b)** composed, ruling A14 | 85-100 | QK^T + softmax + PV from the measured GEMM cells | **pre-registered, NOT yet measured** |

The attention term is left as the controller has ruled it: **A14
(interfaces.md, 2026-09-05) pivots L3 to composed attention** - `S = QK^T` and
`O = PV` through the inherited sycl-tla GEMM (measured 54.80 and 40.59-73.50
TFLOP/s at the attention shapes) with one bandwidth-bound softmax of ours - at a
**pre-registered 85-100 ms/chunk**. That number is a prediction, not a
measurement, and the batched-PV cell A14 names as "L3's first pre-registered
task" is still unmeasured. Both attention terms are therefore carried:

| composition | total ms | t/s | vs vLLM 1973 |
|---|---:|---:|---:|
| (a) with FMHA as measured today | 1619.752 | **1264.4** | 64.1% |
| (b) with A14's composed attention | 1128.558-1143.558 | **1790.9-1814.7** | 90.8-92.0% |
| (b) + dequant hidden behind the GEMM (plan 6c's lever, not landed) | 918.442-933.442 | **2193.9-2229.9** | 111.2-113.0% |

### Does the ceiling clear vLLM's 1973 t/s? Plainly: not yet, and by 8-9%.

- **Today, with the attention that actually exists: no.** 1264.4 t/s, 64.1% of
  vLLM. That is nonetheless a 14.2-21.2% improvement on the 1043.5-1106.9 t/s
  this document last recorded, and the whole of it is the small-kernel
  correction.
- **With A14's composed attention landed at its pre-registered price: still
  no** - 1790.9-1814.7 t/s, **8.0-9.2% under** the bar. The controller's
  projection of 1905 t/s (progress.md, 2026-09-05) assumed a 70 ms small-kernel
  term; the measured 130.346 costs that projection ~90-115 t/s and turns "within
  3.5% of vLLM" into "8-9% under it".
- **Only with the dequant also hidden behind the GEMM does it clear**, at
  2193.9-2229.9 t/s, 11-13% above the bar and inside the third party's
  2400-2500 band's lower reach. Two unlanded levers are then load-bearing at
  once, and neither is measured.

The device-side versus HTTP-inclusive asymmetry documented above still applies
and still favours us.

**Where the time now sits.** With composed attention at its 92.5 ms midpoint the
budget is **GEMM 59.9% · dequant 18.5% · our small kernels 11.5% · attention
8.1% · GDN 1.4% · interop+lm_head 0.7%**. Two consequences the composition makes
unavoidable:

1. **The dequant is now the second-largest term** (210.116 ms for work that is
   pure format conversion), which promotes plan 6c's double-buffer lever from
   "third lever" to the one that decides whether the bar is cleared.
2. **Our small kernels are no longer 20-26% of the budget; they are 11.5%**, and
   93.3 of their 130.3 ms run at 435-471 GB/s - **74-80% of the device's
   measured 590** - which is as finished as a bandwidth-bound kernel gets. What
   is left in them is not a widening problem: it is `pf_ab_proj`'s 19.320 ms at
   58.0 GB/s (a GEMM shape that plan 6c's `gemm_bf16` should take over, not a
   kernel to tune) and `pf_norm_finish`'s 17.617 ms at 308.5 GB/s. Those two are
   the only rows with a factor left in them, and together they are 36.9 ms -
   3.2% of a composed-attention chunk.

## 2026-09-05 addendum 3 - the two load-bearing levers MEASURED; the ceiling recomposed

The controller's probe-first ruling (progress.md, 2026-09-05) sent two levers
to measurement before L1's plumbing spend. Both are now measured, and they
went in opposite directions. Commits `31013b7..31de770`; records
`docs/probe-gemm-batched-2026-09-05.md` and
`docs/probe-dequant-overlap-2026-09-05.md`; suite 46/46, decode untouched.

### Lever 1 - batched attention GEMM: pre-registration MOSTLY HIT

| pre-registered | measured | ratio | verdict |
|---|---:|---:|---|
| P-1 PV M=2048, L=24, packed (192 WGs) ≥ 60 TFLOP/s | **107.72** | 1.80× | **HIT** |
| P-2 PV M=2048, L=6, `strideB=0` (48 WGs) ≥ 90 TFLOP/s | **83.68** | 0.93× | MISS; stop bar 60 does not fire |
| P-3 QKᵀ M=2048, L=6, `strideB=0` (768 WGs) ≥ 45 TFLOP/s | **51.25** | 1.14× | **HIT** |

All measured, iterate-grade, card 1, `ZE_AFFINITY_MASK=1`. The single-head
control cells reproduce P2 at **+0.09% (QKᵀ)** and **+5.5% (PV)**, so the
batteries are comparable and the gain is not a harness artefact.
`transB` - ruling A14's claim, withdrawn by A15 as unverified - is **settled
affirmatively**: the ColumnMajor-B chain instantiates at the pin and is
bitwise identical to the packed run, so `pf_k_transpose` is not built and its
0.23 ms/chunk is not spent.

Two mechanisms, both recorded because they bind later work: **PV at L=24 is
bandwidth-bound** (525.9 GB/s = 89.1% of the card's measured 590 GB/s
streaming reference), so 107.72 TFLOP/s is near this shape's ceiling and A14's
fallback operand-role swap is not needed; and **QKᵀ's rate is capped by the
fp32 `S` write** (91.4% of its DRAM traffic at L=24), which is why it barely
moves with grid size - so QKᵀ's cost and `pf_softmax_causal`'s `S` read are
**the same DRAM stream** and plan 6d's two price rows are not independent.

**Composed attention with the two measured rows substituted:** plan 6d Task 6
Step 2's table with QKᵀ = 64 launches × 0.503 ms = **32.19 ms** and
PV = 64 × 0.308 = **19.71 ms**; softmax 27.3-32.8, `pf_attn_scale_pack` 2.0
and 256 queue handoffs 2.19 unchanged (derived). `attn_chunk` total
**83.4 - 88.9 ms/chunk (derived)**, against A14's pre-registered 85-100 -
**the band holds, at its better end.**

### Lever 2 - dequant overlap: pre-registration MISSED THE ACCEPTANCE BAR

| pre-registered | acceptance bar | measured | verdict |
|---|---|---:|---|
| recovers **0.6-0.9** of the 210.116 ms dequant | **≥ 0.3** | **0.11 - 0.13** | **REJECTED (0.42× of the bar)** |

Two 356,515,840 B scratches, `pf_dequant_tile` on the L0 immediate list
against `gemm_bf16` on the SYCL queue, 8 gate‖up linears: 1.560 ms hidden out
of 12.315 ms of dequant = **0.127** (0.121 after subtracting the 8 removed
queue handoffs; a replicate gave 0.112). **The two queues serialise on the
device, and submission order does not change it** - the same dependencies with
the GEMM submitted first measure 0.122. The dequant control reproduces P3 to
the digit (1.539 ms, 293.1 GB/s), and the overlapped battery's output is
bitwise identical to the serial one's over two deliberately different weight
copies, so this is a measurement and not a race.

**Consequence: the 210.116 ms/chunk dequant is STRUCTURAL on this execution
model.** The row "+ dequant hidden behind the GEMM" in the addendum-2 table is
**withdrawn, not deferred**; the second slot's 356.5 MB is not spent; and P3's
corollary now holds twice over - the only route past the dequant is
eliminating the materialisation (register-only int4 unpack feeding DPAS),
which spec §10 keeps out of scope.

### Recomposed ceiling at C = 2048, with both levers measured

Everything but attention is unchanged from addendum 2: GEMM 680.062 · dequant
210.116 · small kernels 130.346 (measured) · GDN 15.365 · interop 3.290 ·
lm_head 4.379 = **1043.558 ms**. `attn_prep_chunk` is already inside the
measured small-kernel term; `attn_gate_chunk` (+2.7 ms, derived) is not, and
is carried separately below.

| composition | total ms | t/s | vs vLLM 1973 |
|---|---:|---:|---:|
| (a) FMHA as measured today | 1619.752 | 1264.4 | 64.1% |
| (b) composed attention, A14 **pre-registered** 85-100 | 1128.558-1143.558 | 1790.9-1814.7 | 90.8-92.0% |
| **(c) composed attention with the two GEMM rows MEASURED, 83.4-88.9** | **1126.958-1132.458** | **1808.5-1817.3** | **91.7-92.1%** |
| (c) + `attn_gate_chunk` 2.7 (derived, not yet counted) | 1129.658-1135.158 | 1804.2-1813.0 | 91.4-91.9% |
| (c) with the REJECTED overlap lever forced anyway at its measured 0.127 | 1100.28-1105.78 | 1852.1-1861.3 | 93.9-94.3% |

### Does it clear vLLM's 1973 t/s? Plainly: NO - by 8%, and the gap no longer has a lever behind it.

- **The composed ceiling is 1808.5-1817.3 t/s, 91.7-92.1% of the bar -
  7.9-8.3% under.** That is *better* than addendum 2's (b) row, but only by
  ~2 t/s at the midpoint: lever 1 hit and moved the attention band from
  85-100 to 83.4-88.9, which is worth ~18 t/s at the pessimistic end and
  nothing at the optimistic end.
- **Addendum 2 said the bar "clears only if TWO unlanded levers hold at once".
  One of them has now failed.** The 111-113% row is gone. There is no
  composition of measured numbers in this document that reaches 1973.
- **Even forcing the rejected lever does not clear it** (1852-1861 t/s,
  93.9-94.3%), which is the cleanest statement of the situation: the two levers
  the controller ruled load-bearing were, between them, worth ~7% and the gap
  is ~8%.
- The device-side versus HTTP-inclusive asymmetry documented above still
  applies and still favours us; it is not quantified and is not claimed here.

**Where the time now sits**, with composed attention at its 86.2 ms midpoint:
**GEMM 60.2% · dequant 18.6% · our small kernels 11.5% · attention 7.6% ·
GDN 1.4% · interop+lm_head 0.7%.** After this addendum, **60% of the chunk is
an inherited GEMM already running at 89.5% of the derived XMX peak, and 19% is
a format conversion that is now measured to be irreducible by both tuning and
overlap.** Nearly four fifths of the budget has no lever left in it that this
spec's scope permits.

### One open risk, flagged and not asserted

The overlap probe's own GEMM control measured **131.47 TFLOP/s at gate‖up
M=2048 against P2's 150.19 (−12.5%)**, through the production `gemm_bf16`
wrapper rather than P2's harness. It is **not** the wrapper's per-call
`initialize` - `gemm_batched_test` reproduces P2 at +0.09% and +5.5% through
the same path - and the candidates named in the probe record (a ~1.31 GB
resident footprint, a control run immediately behind a 174,080-work-group
dequant) were deliberately not chased, per the no-tuning rule. **If that 12.5%
turned out to be real for the production path, the GEMM term would rise from
680.062 to ~777 ms and the ceiling would fall to ~1665-1673 t/s (84-85% of
vLLM, derived).** It is the single largest unresolved risk to this composition
and it belongs to the L2 stage, which is where the production GEMM path is
built and where the same shape can be measured under production conditions.
