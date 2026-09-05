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
