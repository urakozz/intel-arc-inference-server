# Probe - L2-resident slab dequant (spec 2, the last unpriced lever on the 210 ms)

## Pre-registration - written BEFORE the probe is built or run

Nothing in this section is a measurement. Every value below is a **prediction**
or is **derived** from a measurement already on the record and named at the
point of use.

## The question

`docs/probe-dequant-2026-09-04.md` (P3) measured the whole-chunk dequant at
**210.116 ms/chunk (derived from measured)** and identified the mechanism:
`r + 2w` normalisation gives 506-529 GB/s on a 590 GB/s part while `r + w`
gives only 283-296, so the cost is **DRAM write-allocate on the bf16 scratch** -
the 356,515,840 B gate‖up matrix is written to DRAM and then read back by the
GEMM. `docs/probe-dequant-overlap-2026-09-05.md` then measured that the write
cannot be *hidden* (recovery 0.11-0.13 against a 0.3 bar).

This probe asks whether it can be **avoided**. If the dequant and the GEMM are
interleaved per **N-slab** - dequant `[K][Ns]` bf16 into a slab small enough to
sit in the 24 MB L2, then run the GEMM on that slab, then the next - the write
and the re-read may never reach DRAM. The int4 read (0.5 B/weight) still
happens, but that is traffic the GEMM's B-load would have paid anyway.

## Pre-registered cells and their arithmetic

Shape: **gate‖up, K = 5120, N = 34816, layout 0, `[K][N]`, M = 2048** - the
same shape P3, P2 and the overlap probe all measured, so every control has a
prior number to tie to.

| `Ns` | slabs | slab bytes (derived) | vs 24 MB L2 | GEMM N-tiles × M-tiles = WGs (derived, 256×256 tile) |
|---:|---:|---:|---|---:|
| 1024 | 34 | 10,485,760 | fits, 2.3× spare | 4 × 8 = **32** |
| **2048** (the cell) | **17** | **20,971,520** | **fits, ~3 MB spare** | 8 × 8 = **64** |
| 4096 | 8 × 4096 + 1 × 2048 | 41,943,040 | **does not fit** | 16 × 8 = **128** |

`N = 34816 = 2048 × 17`, so 4096 does not divide it; the `Ns = 4096` cell is
8 full slabs plus one 2048-wide remainder, and it is kept deliberately as the
cell that *should* fall back to DRAM if the L2 story is real.

Full-width reference grid at M = 2048 is **136 × 8 = 1088 WGs** (P2's measured
grid table), at **150.19 TFLOP/s (measured, P2)** = **4.862 ms (derived:
730.145 GFLOP ÷ 150.19)**. The overlap probe measured the same cell at
**131.47 TFLOP/s / 5.554 ms (measured)** in a ~1.3 GB-resident harness; this
probe reports its own control and scores against both.

## The pre-registered predictions

1. **Slab GEMM rate.** P2's occupancy scan on the PV shape measured 40.59
   TFLOP/s at 8 WGs and 73.50 at 16 (`docs/probe-gemm-batched-2026-09-05.md`,
   measured). At **64 WGs** predict **≥ 110 TFLOP/s**, and the **sum of the 17
   slab GEMMs stays within +10% of the full-width GEMM time**.
2. **The lever.** gate‖up dequant + GEMM, slab-interleaved, against the current
   two-pass (full dequant, then full GEMM). Predict the dequant overhead drops
   **210 → ≤ 70 ms/chunk-equivalent**, i.e. **≥ 2/3 of the write-allocate
   traffic stays in L2**.
3. **Decision rule.** Saving = 210.116 − (chunk-equivalent slab overhead):
   - **≥ 90 ms saved → the lever is real** and plan 6c's L2 stage adopts slab
     interleaving;
   - **30 - 90 ms → priced, the operator rules**;
   - **< 30 ms → dead.**

### How "chunk-equivalent" is computed, fixed in advance

The probe measures one shape. The chunk-equivalent overhead is

```
overhead_slab_per_matrix = t_slab_interleaved − t_gemm_full            (measured − measured)
ratio                    = overhead_slab_per_matrix / t_dequant_full   (measured / measured)
chunk_equivalent         = ratio × 210.116 ms                          (derived)
```

`t_gemm_full` is this probe's own full-width GEMM control, not P2's, so any
harness-level GEMM difference cancels out of the ratio. **Any slab-GEMM
slow-down is charged to the lever**, because it lands inside
`t_slab_interleaved`; that is deliberate - the lever must pay for its own
occupancy loss.

## The named risk, and the controls that isolate it

**Xe2's L2 is not guaranteed to retain a kernel's writes for the next kernel.**
Three controls, all pre-registered:

- **C6 - slab dequant with no consumer.** 17 slab dequants into the *same*
  20.97 MB slab buffer, cycling the int4 source across all 17 slabs so the
  *input* is cold (94.70 MB, measured by P3) while the *output* is the same
  20.97 MB every time. Against **C1**, the full `[K][N]` dequant, which writes
  356.52 MB to DRAM at a measured 1.539 ms. **If L2 absorbs the writes, C6 is
  far faster than C1; if it does not, C6 ≈ C1.** This is the direct measurement
  of the named risk and it does not depend on any GEMM.
- **R1 - guaranteed-warm slab GEMM.** The same slab GEMM run twice
  back-to-back on the same slab, with a host wait between, so the second run
  reads a slab the first run has just pulled. Its time against the first run's
  bounds what an L2-resident B operand is worth at all.
- **Achieved GB/s** is printed for every dequant cell under both `r + w` and
  `r + 2w`, the two normalisations P3 already used, so the write-allocate model
  is scored the same way it was established.

## Launch overhead, priced in advance and not hidden

The interleaved battery's own wall clock already contains its launch overhead,
so the decision rule needs no correction. The whole-chunk figure is reported
separately, and the brief's estimate is corrected here because only gate‖up has
17 slabs:

| matrix | count | N | slabs at Ns = 2048 (derived) | slabs × count |
|---|---:|---:|---:|---:|
| QkvZ | 48 | 16384 | 8 | 384 |
| OutProj | 48 | 5120 | 2 full + 1 × 1024 = 3 | 144 |
| GateUp | 64 | 34816 | 17 | 1088 |
| Down | 64 | 5120 | 3 | 192 |
| Qkv | 16 | 14336 | 7 | 112 |
| OProj | 16 | 5120 | 3 | 48 |
| **total** | **256** | | | **1968** |

1968 slabs × 2 launches = **3936 launches/chunk (derived)** against today's
**512** (256 dequants + 256 GEMMs), i.e. **+3424 launches** at the measured
2.1-4.5 µs/launch = **+7.2 to +15.4 ms/chunk (derived)** - not the brief's
25-50 ms, which assumed 17 slabs for every matrix. It is subtracted from the
saving before the decision rule is applied, and if it eats the win that is
said outright.

## Grade and protocol

Iterate grade: 8 replays, first 3 discarded, median of the last 5, one
discarded warm-up; `ZE_AFFINITY_MASK=1`; card 1. Disk free and the DRM holder
count are recorded beside the measurement. One defect, one fix, one
measurement: no tuning of `Ns` beyond the three pre-registered cells, and the
verdict is the `Ns = 2048` cell's whatever the other two do.

Deliverable: `tools/probe/probe_dequant_slab`, this document's measured
section.
