# P3 - pf_dequant_tile: int4 tiles -> the bf16 scratch (spec 2 §4)

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

All pre-registered predictions and all derived values below are iterate-grade.
No measurement has been made for this record.

## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

1. **>= 500 GB/s** (prediction, iterate-grade; spec §4's P3 row), where GB/s
   counts **bytes read plus bytes written, both once**. The probe prints that
   denominator in its own output, because a different one changes the number by
   2x.
2. Bit-exact against `common::Int4Gptq::at` -> `common::f32_to_bf16`, at every
   shape, in **both** tile layouts and **both** output orientations. Not a
   tolerance: the device and the host evaluate the same fp32 expression and
   round once.
3. Layout 1 is at least as fast as layout 0 at every shape: one contiguous
   544 B run per subgroup-step against 8 strided 64 B rows plus a scale row --
   the 10-vs-17-message contrast `probe_gemv_loads.cl` measured for decode.
4. The **`[N][K]` orientation is faster than `[K][N]`**, because the source
   tiles are N-tiled 16 columns wide and a lane owns one column: writing
   `[N][K]` makes each lane's 64 k-values one contiguous 128-byte run, while
   `[K][N]` makes them 64 scattered 2-byte stores at stride 2N. Predicted gap:
   large, not marginal. The prefill GEMM's native input remains `[K][N]`,
   `ldb = N`, per the interface ruling A5; this probe measures `[N][K]` as a
   second store-geometry comparison.

## The overhead this decides, and the two-model reconciliation

Bytes WRITTEN per chunk over the 256 int4 matrices a chunk dequantises
(48 GDN layers x {QkvZ, OutProj, GateUp, Down} + 16 FA x {Qkv, OProj, GateUp,
Down}), at 2 B per bf16 element (derived, iterate-grade):

| matrix | K x N | bytes (derived, iterate-grade) | x layers | subtotal (derived, iterate-grade) |
|---|---:|---:|---:|---:|
| QkvZ | 5120 x 16384 | 167.77 MB | 48 | 8053.1 MB |
| OutProj | 6144 x 5120 | 62.91 MB | 48 | 3019.9 MB |
| GateUp | 5120 x 34816 | 356.52 MB | 64 | 22817.0 MB |
| Down | 17408 x 5120 | 178.26 MB | 64 | 11408.5 MB |
| Qkv | 5120 x 14336 | 146.80 MB | 16 | 2348.8 MB |
| OProj | 6144 x 5120 | 62.91 MB | 16 | 1006.6 MB |
| **written** | | | | **48.65 GB** |
| read (the int4 linears themselves) | | | | **~13.0 GB** |

- **Model A, streaming writes:** 61.7 GB at 590 GB/s = **104 ms/chunk**
  (derived, iterate-grade). Against a 2.215 s C = 4096 chunk (P2's
  pre-registration, estimated, iterate-grade) that is **4.7 %** (derived,
  iterate-grade).
- **Model B, write-allocate** (the bf16 store pulls the line before writing it,
  so a write costs 2x its bytes): 110.3 GB = **187 ms** (derived,
  iterate-grade) -- **8.4 %** at C = 4096 and **16.8 %** at C = 2048 (derived,
  iterate-grade).

**Model B is what spec §3.1 quotes** ("derived: 8-16 % of chunk time at
C = 4096-2048") -- the two match to the digit, which is how the spec's figure
is identified as a write-allocate model. **This probe is the reconciliation:**
its measured GB/s decides which model is right, and the report then replaces
both with the measured number. Spec §11's trigger stands: if the measured
overhead exceeds 20 % the design revisits a second weight layout for prefill.

Cross-check on the same arithmetic from the other side: spec §3.1 prices the
gate||up matrix alone at "~1.2 ms at 590 GB/s" (estimated, iterate-grade).
Model A gives (356.52 write + 94.70 read) MB / 590 GB/s = **0.765 ms**
(derived, iterate-grade); Model B gives (2 x 356.52 + 94.70) / 590 =
**1.37 ms** (derived, iterate-grade), and 2 x 356.52 / 590 = **1.21 ms**
(derived, iterate-grade). So the spec's 1.2 ms is Model B with the int4 read
omitted. One quantity, three arithmetics, one measurement to settle it.


## Measured (2026-09-04; iterate-grade)

Every timing, bandwidth value, and bit-exact sample count in this section is
**measured, iterate-grade**. The probe ran on card 1 with
ZE_AFFINITY_MASK=1, on Intel(R) Arc(TM) Pro B70 Graphics. It used one
356,515,840-byte reusable prefill scratch, 40 launches per timed list, eight
replays with the first three discarded and the median of the last five. Source
copies used NB = max(2, floor(72 MiB / source_bytes) + 1), so a timing list
cycles past the 24 MiB L2. The lm_head int4 rows are explicitly probe-only:
they use a temporary 2.543 GB output because that tensor cannot fit in
PrefillScratch and is not a prefill linear.

| shape | K×N | L | orient | µs | read MB | write MB | GB/s (r+w) | GB/s (r+2w) | bit-exact |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| qkv‖z | 5120×16384 | 0 | [K][N] | 718.5 | 44.56 | 167.77 | 295.5 | 529.0 | yes (1%, 838861 cells) |
| qkv‖z | 5120×16384 | 0 | [N][K] | 1230.0 | 44.56 | 167.77 | 172.6 | 309.0 | yes (1%, 838861 cells) |
| qkv‖z | 5120×16384 | 1 | [K][N] | 734.3 | 44.56 | 167.77 | 289.2 | 517.7 | yes (1%, 838861 cells) |
| qkv‖z | 5120×16384 | 1 | [N][K] | 1209.9 | 44.56 | 167.77 | 175.5 | 314.2 | yes (1%, 838861 cells) |
| out/o_proj | 6144×5120 | 0 | [K][N] | 270.7 | 16.71 | 62.91 | 294.1 | 526.5 | yes (1%, 314573 cells) |
| out/o_proj | 6144×5120 | 0 | [N][K] | 422.1 | 16.71 | 62.91 | 188.6 | 337.7 | yes (1%, 314573 cells) |
| out/o_proj | 6144×5120 | 1 | [K][N] | 273.7 | 16.71 | 62.91 | 290.9 | 520.8 | yes (1%, 314573 cells) |
| out/o_proj | 6144×5120 | 1 | [N][K] | 423.8 | 16.71 | 62.91 | 187.9 | 336.3 | yes (1%, 314573 cells) |
| gate‖up | 5120×34816 | 0 | [K][N] | 1539.3 | 94.70 | 356.52 | 293.1 | 524.7 | yes (1%, 1782580 cells) |
| gate‖up | 5120×34816 | 0 | [N][K] | 2878.2 | 94.70 | 356.52 | 156.8 | 280.6 | yes (1%, 1782580 cells) |
| gate‖up | 5120×34816 | 1 | [K][N] | 1596.1 | 94.70 | 356.52 | 282.7 | 506.1 | yes (1%, 1782580 cells) |
| gate‖up | 5120×34816 | 1 | [N][K] | 2918.5 | 94.70 | 356.52 | 154.6 | 276.8 | yes (1%, 1782580 cells) |
| down | 17408×5120 | 0 | [K][N] | 764.4 | 47.35 | 178.26 | 295.2 | 528.4 | yes (1%, 891290 cells) |
| down | 17408×5120 | 0 | [N][K] | 1193.2 | 47.35 | 178.26 | 189.1 | 338.5 | yes (1%, 891290 cells) |
| down | 17408×5120 | 1 | [K][N] | 791.1 | 47.35 | 178.26 | 285.2 | 510.5 | yes (1%, 891290 cells) |
| down | 17408×5120 | 1 | [N][K] | 1205.8 | 47.35 | 178.26 | 187.1 | 334.9 | yes (1%, 891290 cells) |
| q‖k‖v | 5120×14336 | 0 | [K][N] | 631.7 | 38.99 | 146.80 | 294.1 | 526.5 | yes (1%, 734004 cells) |
| q‖k‖v | 5120×14336 | 0 | [N][K] | 1022.4 | 38.99 | 146.80 | 181.7 | 325.3 | yes (1%, 734004 cells) |
| q‖k‖v | 5120×14336 | 1 | [K][N] | 643.6 | 38.99 | 146.80 | 288.7 | 516.8 | yes (1%, 734004 cells) |
| q‖k‖v | 5120×14336 | 1 | [N][K] | 1057.8 | 38.99 | 146.80 | 175.6 | 314.4 | yes (1%, 734004 cells) |
| lm_head int4 (probe-only) | 5120×248320 | 0 | [K][N] | 11388.7 | 675.43 | 2542.80 | 282.6 | 505.9 | yes (1%, 12713984 cells) |
| lm_head int4 (probe-only) | 5120×248320 | 0 | [N][K] | 31146.3 | 675.43 | 2542.80 | 103.3 | 185.0 | yes (1%, 12713984 cells) |
| lm_head int4 (probe-only) | 5120×248320 | 1 | [K][N] | 12256.3 | 675.43 | 2542.80 | 262.6 | 470.0 | yes (1%, 12713984 cells) |
| lm_head int4 (probe-only) | 5120×248320 | 1 | [N][K] | 31974.2 | 675.43 | 2542.80 | 100.7 | 180.2 | yes (1%, 12713984 cells) |

Ramp control: the discarded qkv‖z L0 [K][N] warm-up was 295 GB/s (r+w,
measured, iterate-grade); its recorded row was 295.5 GB/s (r+w, measured,
iterate-grade). The end-of-battery drift control re-measured that cell at
295.2 GB/s, a -0.09% change (both measured, iterate-grade). The full
256×64 unit test separately passed all four cells bit-exactly.

### Prediction versus measurement

- The >= 500 GB/s prediction with the pre-registered r+w denominator was a
  **miss**: [K][N] rows measured 282.6-295.5 GB/s (measured, iterate-grade).
- Bit-exactness was a **hit**: every unit-test element and every probe 1%
  sample passed (measured, iterate-grade).
- The prediction that layout 1 would be at least as fast as layout 0 was a
  **miss** for the settled [K][N] geometry. The loader's actual L1 qkv‖z row
  measures 289.2 GB/s versus L0's 295.5 GB/s; every other prefill linear uses
  L0 (measured, iterate-grade).
- The [N][K]-faster prediction was a **miss in the opposite direction**.
  [K][N] is faster at every shape and layout (measured, iterate-grade), so it
  is both the settled sycl-tla-native layout and the better store geometry.

The r+2w column is 506.1-529.0 GB/s for the five prefill shapes (measured,
iterate-grade), close to the card's 590 GB/s streaming reference (measured,
iterate-grade in the earlier bandwidth probe), while r+w is only 282.7-295.5
GB/s (measured, iterate-grade). This supports Model B's write-allocate
accounting; the two normalizations describe the same elapsed measurement.

## Measured per-chunk overhead

The table uses the model's selected layout (qkv‖z L1; every other int4
prefill linear L0), [K][N], and the matching measured denominator. The traffic,
times, totals, and percentages are **derived, iterate-grade** from the table
above.

| matrix | count | read GB | written GB | dequant ms, Model A | dequant ms, Model B |
|---|---:|---:|---:|---:|---:|
| QkvZ | 48 | 2.139 | 8.053 | 35.243 | 35.243 |
| OutProj | 48 | 0.802 | 3.020 | 12.996 | 12.995 |
| GateUp | 64 | 6.061 | 22.817 | 98.525 | 98.523 |
| Down | 64 | 3.030 | 11.409 | 48.912 | 48.916 |
| Qkv | 16 | 0.624 | 2.349 | 10.108 | 10.107 |
| OProj | 16 | 0.267 | 1.007 | 4.332 | 4.332 |
| **total** | **256** | **12.924** | **48.654** | **210.116** | **210.116** |

P2 has not landed, so the comparison remains against its 2.215 s C=4096
pre-registration (estimated, iterate-grade), and the 1.1075 s C=2048
half-chunk estimate (derived, iterate-grade). The measured dequant total is
9.49% at C=4096 and 18.97% at C=2048 (derived, iterate-grade), whichever
normalization is used. The spec §11 20% trigger does **not** fire, by 1.03
percentage points at C=2048 (derived, iterate-grade).

## Double-buffering finding for L2

One scratch serializes dequant matrix i+1 behind GEMM matrix i. A second scratch
would consume another 356.52 MB (derived, iterate-grade) on the 32 GB card
against 13.7 GB weights, 1.07 GB KV, activations <= 0.5 GB, and a possible 128
MB P5 allocation (all estimated, iterate-grade from spec §3.6). It could hide
at most the largest measured single-matrix dequant, GateUp L0 [K][N] at 1.539
ms (measured, iterate-grade), at each eligible handoff. It is not required for
correctness and is not P3 work; carry this as an L2 overlap decision, where the
P2 GEMM schedule can establish how much of the derived 210.116 ms/chunk is
actually hideable.
