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
