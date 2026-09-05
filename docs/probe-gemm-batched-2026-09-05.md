# Probe - `gemm_bf16_batched` at the attention shapes (spec 2, L3 lever 1)

Written BEFORE any build or run of `gemm_bf16_batched`. Plan 6d-composed
Task 1 Step 1. Grade of every number below: **prediction** (from measured
inputs that are named); nothing here is a measurement.

*Deviation from plan 6d Task 1 Step 1, recorded: the plan names
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/l3-attention-prereg.md` as
this record's home, but `.superpowers/sdd/.gitignore` is `*`, so a file there
cannot be committed and a pre-registration that is not committed before the
measurement is not a pre-registration. This record therefore lives in
`docs/`, beside `probe-prefill-gemm-2026-09-04.md` and every other probe
record, which is where commit `4009100` put the last one for the same
reason.*

Basis rows, all **measured, iterate-grade**, from
`docs/probe-prefill-gemm-2026-09-04.md` (P2, corrected 256-GRF build, pin
`91e5bd735517d8e79591b41e0d0cd37a7bacdca7`):

| cell | rate | work-groups |
|---|---:|---:|
| PV shape (M=2048, K=4096, N=256), single head | 40.59 TFLOP/s | 8 |
| PV shape at M=4096 | 73.50 TFLOP/s | 16 |
| PV shape, the whole occupancy scan | 10.67 / 21.55 / 40.59 / 73.50 | 2 / 4 / 8 / 16 |
| QKᵀ shape (M=2048, K=256, N=4096), single head | 54.80 TFLOP/s | 128 |
| QKᵀ shape at M=4096 | 49.91 TFLOP/s | 256 |
| observed peak, any production shape | 164.21 TFLOP/s | down, M=2048 |

## The three predictions

**P-1 (ruling A14's own ask). Batched PV at M = 2048, K = 4096, N = 256,
L = 24, packed strides: ≥ 60 TFLOP/s.** Basis: single-head is 40.59 at 8
work-groups and the shape's rate is almost exactly linear in work-group count
over the measured scan; L = 24 is 192 work-groups on 32 Xe-cores.

**P-2 (the production form). PV at L = 6, `strideB = 0`, M = 2048
(48 work-groups): ≥ 90 TFLOP/s.** Basis: the same scan, 3× the 16-WG point
(73.50), capped well under the 164.21 observed peak.

**P-3. QKᵀ at L = 6, `strideB = 0`, M = 2048, K = 256, N = 4096
(768 work-groups): ≥ 45 TFLOP/s.** Basis: the single-head cell is 54.80 at
128 WGs and 49.91 at 256 WGs - mildly *decreasing* in grid size, so the
prediction is deliberately below the smaller-grid measurement.

Two further cells are measured because the controller's brief names them, and
they are **not** separately pre-registered beyond P-1/P-3's reasoning: QKᵀ at
L = 24 (packed, 3072 WGs) and PV at L = 24 with `strideB = 0`.

## Stop bars (written now, so they are not invented after a miss)

- **P-2 < 60 TFLOP/s** → the PV term exceeds 27.5 ms/chunk and the composed
  attention price is re-derived from the measured rate before any Task 4
  spend.
- **P-3 < 35 TFLOP/s** → the QKᵀ term exceeds 47 ms/chunk and the same
  re-derivation applies.

## The alternative, written now rather than after a miss

If batching does **not** restore the rate, swap the operand roles for PV -
`Oᵀ = Vᵀ · Pᵀ` gives M = 256, N = C, i.e. 1 M-tile × 8 N-tiles = 8 WGs per
head and 192 batched, the same grid by a different route. If *that* is also
flat, the per-head QKᵀ form (measured 54.80, 24 launches per layer, +0.05
ms/chunk of launch overhead) is the standing fallback for the QKᵀ half, PV
stays at its measured 40.59, and the attention price is re-registered at
**100-115 ms/chunk** (prediction).

## `transB` - the open question this task settles first

Ruling A14 asserted "sycl-tla's Xe GEMM takes ColumnMajor B natively";
plan 6c Task 1 recorded RowMajor and P2 measured only RowMajor; A15's last
bullet **withdrew** A14's claim as unverified. Task 1 Step 3 settles it by
instantiating the ColumnMajor-B chain against the pin. Two recorded outcomes:

- **transB available** → Task 4 reads the K cache in place and `pf_kt` is not
  allocated;
- **transB does not instantiate** → the fallback is `pf_k_transpose`, a packed
  `Kᵀ` `[4][256][Dp]` bf16 slab per layer, then a plain RowMajor B with
  `ldb = Dp`. Priced (derived): 4 kv-heads × 4096 × 256 × 2 B read + written =
  8.39 MB/layer → **0.23 ms/chunk at 590 GB/s**, plus **33,554,432 B** of
  scratch at max_len 16384. It is cheap precisely because there are **4**
  kv-heads, not 24.

## Measurement protocol

8 replays, discard the first 3, median of the last 5; one discarded warm-up
per cell; `ZE_AFFINITY_MASK=1`; the grid printed from
`Gemm::get_grid_shape(args)` beside every rate. **Iterate grade, not a bench
row** - the timed rows live in `gemm_batched_test`, which is not the P2
harness.

---

## Measured (2026-09-05, card 1, `ZE_AFFINITY_MASK=1`; iterate grade)

`tests/prefill/gemm_batched_test`, commit `1169930`, on Intel(R) Arc(TM) Pro
B70 Graphics at pin `91e5bd7...` (content-verified). 8 replays, first 3
discarded, median of the last 5, 4 enqueues per replay - P2's protocol, so
the rates below are comparable to its matrix. Every ms and TFLOP/s in this
section is **measured, iterate-grade**; every GB/s and every ms/chunk is
**derived** from those and from the byte arithmetic printed beside it.

| cell | M | K | N | L | strideB | grid (reported) | ms | TFLOP/s |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| **P-1** PV packed | 2048 | 4096 | 256 | 24 | packed | 1×8×24 = 192 | 0.957 | **107.72** |
| **P-2** PV shared B | 2048 | 4096 | 256 | 6 | 0 | 1×8×6 = 48 | 0.308 | **83.68** |
| **P-3** QKᵀ shared B | 2048 | 256 | 4096 | 6 | 0 | 8×16×6 = 768 | 0.503 | **51.25** |
| QKᵀ packed | 2048 | 256 | 4096 | 24 | packed | 8×16×24 = 3072 | 2.014 | 51.18 |
| PV shared B | 2048 | 4096 | 256 | 24 | 0 | 1×8×24 = 192 | 0.997 | 103.35 |
| PV packed | 2048 | 4096 | 256 | 6 | packed | 1×8×6 = 48 | 0.309 | 83.28 |
| QKᵀ packed | 2048 | 256 | 4096 | 6 | packed | 8×16×6 = 768 | 0.508 | 50.76 |
| PV single head (control) | 2048 | 4096 | 256 | 1 | packed | 1×8×1 = 8 | 0.100 | 42.84 |
| QKᵀ single head (control) | 2048 | 256 | 4096 | 1 | packed | 8×16×1 = 128 | 0.078 | 54.85 |

### Prediction versus measurement

| # | predicted | measured | ratio | verdict |
|---|---:|---:|---:|---|
| P-1 PV L=24 packed | ≥ 60 | **107.72** | **1.80×** | **HIT** |
| P-2 PV L=6 strideB=0 | ≥ 90 | **83.68** | **0.93×** | **MISS** (stop bar 60 does NOT fire) |
| P-3 QKᵀ L=6 strideB=0 | ≥ 45 | **51.25** | **1.14×** | **HIT** |

**Battery comparability.** The two single-head control cells reproduce P2's
own numbers on a different harness: QKᵀ **54.85 vs P2's 54.80 (+0.09%)** and
PV **42.84 vs P2's 40.59 (+5.5%)**. The batteries are comparable, and the
gains below are not a harness artefact.

### The mechanism, and which model died

**The linear-in-work-group model died at 48 work-groups, and DRAM bandwidth
is what replaced it.** The pre-registration extrapolated PV's measured
10.67 / 21.55 / 40.59 / 73.50 at 2 / 4 / 8 / 16 WGs linearly. Measured here:
42.84 at 8, 83.68 at 48, 107.72 at 192 - i.e. 6× the work-groups bought
1.95× the rate and 24× bought 2.51×. Traffic accounting says why (derived,
at the measured ms):

| cell | A read | B read | C write | total | GB/s | % of the 590 GB/s streaming reference |
|---|---:|---:|---:|---:|---:|---:|
| P-1 PV L=24 packed | 402,653,184 | 50,331,648 | 50,331,648 | 503,316,480 | **525.9** | **89.1%** |
| P-2 PV L=6 strideB=0 | 100,663,296 | 2,097,152 | 12,582,912 | 115,343,360 | 374.5 | 63.5% |
| P-3 QKᵀ L=6 strideB=0 | 6,291,456 | 2,097,152 | 201,326,592 | 209,715,200 | 416.9 | 70.7% |
| QKᵀ L=24 packed | 25,165,824 | 50,331,648 | 805,306,368 | 880,803,840 | 437.3 | 74.1% |

- **PV at L = 24 is bandwidth-bound, not occupancy-bound**: 525.9 GB/s is
  89.1% of the card's measured streaming reference. 107.72 TFLOP/s is
  therefore near this shape's ceiling on this part, and the alternative
  operand-role swap (`Oᵀ = Vᵀ·Pᵀ`) written into the pre-registration is **not
  needed and is not taken**.
- **P-2's miss is a 7% extrapolation error, not a structural finding.** At 48
  WGs the shape is at 63.5% of the wall - neither occupancy-saturated nor
  bandwidth-saturated - so the linear model over-predicted. It clears its
  stop bar (60) by 1.39×, so nothing is re-derived.
- **QKᵀ's rate is capped by the fp32 `S` write, which is why it barely moves
  with grid size** (54.85 at 128 WGs → 51.25 at 768 → 51.18 at 3072). At
  L = 24, 91.4% of that GEMM's DRAM traffic is the 805 MB `S` output. The
  consequence for plan 6d: **QKᵀ's cost and `pf_softmax_causal`'s S read are
  the same DRAM stream**, so the two rows in that plan's price table are not
  independent, and any future lever on one is a lever on the other.
- **`strideB = 0` is worth nothing on rate** (PV L=24: 103.35 shared vs
  107.72 packed; PV L=6: 83.68 vs 83.28; QKᵀ L=6: 51.25 vs 50.76 - all inside
  ±4%). It is still the right production form because it is the only one that
  needs no copy of the kv-head, but it must not be *sold* as a speed-up.

### `transB` - SETTLED, affirmatively

**The ColumnMajor-B chain instantiates at the pin and is bitwise correct.**
`XeGemmT` (identical to `XeGemm` but `LayoutB = cutlass::layout::ColumnMajor`)
compiled without a diagnostic, and test case 5 - B as `[N][K]` row-major with
`ldb = 1024`, the K/V cache's own geometry - is **bitwise identical** to the
packed `[K][N]` run of the same logical matrix.

No ColumnMajor-specific code is written: `block_2d_transform_selector`
(`cute/atom/copy_traits_xe_2d.hpp:807-816`) picks the transpose message from
the global stride, so `TagToStrideB<ColumnMajor>` = `(ldb, 1, strideB)` over
`(N, K, L)` is the whole change. Consequences, as pre-registered:

- Task 4 reads the K cache in place;
- **`pf_k_transpose` is not built**, its 0.23 ms/chunk is not spent, and its
  33,554,432 B are not allocated;
- ruling A14's original claim was right and A15's withdrawal of it was
  correctly cautious - the claim is now measured rather than asserted.

### The composed attention price, with the two measured rows substituted

Plan 6d Task 6 Step 2's table, with rows 2 and 4 replaced by measurement.
The QKᵀ and PV rows are `64 launches/chunk` (4 kv-groups × 16 FA layers) at
the measured P-3 and P-2 ms; the other rows are the plan's own derived
figures, unchanged.

| term | pre-registered ms/chunk | with measurement | grade |
|---|---:|---:|---|
| `attn_prep_chunk` | 2.7 | 2.7 | derived |
| QKᵀ (`gemm_bf16_batched`) | 30.1 - 36.6 | **32.19** | 64 × 0.503 ms measured |
| `pf_softmax_causal` | 27.3 - 32.8 | 27.3 - 32.8 | derived |
| PV (`gemm_bf16_batched`) | 18.3 - 27.5 | **19.71** | 64 × 0.308 ms measured |
| `pf_attn_scale_pack` | 2.0 | 2.0 | derived |
| queue handoffs | 2.19 | 2.19 | 256 × 8.569 µs measured (P1) |
| **`attn_chunk` total** | **85 - 100** | **83.4 - 88.9** | derived |
| `attn_gate_chunk` | 2.7 | 2.7 | derived |
| **attention family total** | 91 - 106 | **88.8 - 94.3** | derived |

The band narrows because its two widest rows are now single numbers; what
remains open is the softmax's 5.5 ms of L2 uncertainty. **The A14 band holds
at its better end**, and the composed attention term the ceiling should now
carry is **83.4 - 88.9 ms/chunk (derived)** rather than 85 - 100.

### Correctness and determinism, as recorded by the test

Max ratio to the `64 · 2^-24 · Σ|A·B|` bar, over 4096 sampled `(l, m, n)`
triples per case: **0.654 - 0.795** - an ~80× margin, and within 15% of the
random-walk model's predicted 0.93, at every shape and every L. Two runs into
two distinct allocations were **bitwise equal** on every case. Batch
independence, `strideB = 0`, non-packed `ldb`/`lda`/`ldc`, `transB` and the
M-stacking identity (`L = 6, strideB = 0` ≡ one GEMM at `M = 12288`) are all
**bitwise**, not toleranced.

Suite after this work: **46/46** (was 45/45). Decode untouched.
