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
