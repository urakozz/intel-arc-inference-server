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
