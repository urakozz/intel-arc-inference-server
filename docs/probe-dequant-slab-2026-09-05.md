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

---

## Measured (2026-09-05, card 1, `ZE_AFFINITY_MASK=1`; iterate grade)

`tools/probe/probe_dequant_slab` on Intel(R) Arc(TM) Pro B70 Graphics. 8
replays, first 3 discarded, median of the last 5, one discarded warm-up.

Box state at the timed run: **disk 83 GB free** (`/dev/nvme0n1p2 915G 786G 83G
91% /`); **DRM holders = 2** - pid 285644 `baobab`, pid 285678 `ptyxis`, both
desktop GUI processes on `/dev/dri/card0` plus all three render nodes. Not a
provably idle box, so this is iterate grade, which is what the probe is graded
at.

### Controls

| control | ms | rate | reference | Δ |
|---|---:|---:|---|---:|
| **C1** full dequant, production `pf_dequant_tile` | 1.548 | 291.5 GB/s r+w, 521.8 r+2w | P3 measured 1.539 / 293.1 / 524.7 | **+0.6%** |
| **C1'** full dequant, this probe's slab kernel at `NS = N` | 1.546 | 291.8 GB/s r+w | C1 | **−0.12%** |
| **C2** `gemm_bf16` full width, 1088 WGs | 4.751 | **153.69 TFLOP/s** | P2 measured 150.19 | **+2.3%** |
| **C3** two-pass total (C1 then C2) | 6.325 | | sum model 6.299 | +0.41% |

**Correctness: bitwise identical.** The slab kernel's output was compared
element by element against the production kernel's own `[K][N]` scratch over
slabs 0, 8 and 16 at `Ns = 2048` - **31,457,280 weights, 0 mismatches.**

**A flagged risk does not reproduce, and that is reported here because it is
this probe's control that settles it.** `progress.md` carries an open risk from
the overlap probe: its `gemm_bf16` control read **131.08-131.47 TFLOP/s**, 12.5%
under P2, with the consequence that "if real for the production path, GEMM
rises 680 → ~777 ms and the ceiling falls to ~1665-1673 t/s". **This probe's C2
measures 153.69 TFLOP/s at the identical shape, M and code path - +2.3% ABOVE
P2, not 12.5% below.** The difference between the two harnesses is the one the
overlap probe named: it held ~1.31 GB resident (two 356.5 MB scratches, two
94.7 MB int4 sources) and ran its GEMM control immediately behind a
174,080-work-group dequant; this probe holds one 356.5 MB scratch and one int4
source. **The 131 TFLOP/s was the harness, not the GEMM.** The 680.1 ms/chunk
GEMM term stands and the ~1665-1673 t/s downside is withdrawn.

### The slab battery

| `Ns` | slabs | slab B | **C6** dequant-only, no consumer | C6 GB/s r+w | C6 r+2w | **C5a** slab GEMMs | TFLOP/s | **C5b** + a wait per slab | **C4** interleaved (the lever) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 34 | 10,485,760 | **0.514** | **878.2** | **1572.0** | 4.595 | 158.91 | 5.100 | 6.483 |
| **2048** | **17** | **20,971,520** | **0.710** | **635.6** | **1137.9** | 5.082 | **143.68** | 5.462 | **6.816** |
| 4096 | 9 | 41,943,040 | 1.525 | 295.8 | 529.5 | 5.367 | 136.05 | 5.568 | 7.022 |

R1 (`Ns = 2048`, one slab, the L2 evicted by a full-width dequant between
replays): first **0.3435 ms**, second (guaranteed warm) **0.3265 ms**,
**−4.95%**.

## The named risk is FALSIFIED: this L2 *does* retain the writes

The pre-registration's named risk was that Xe2's L2 need not keep a kernel's
writes for the next kernel. **C6 measures that it does, and the evidence does
not depend on any model of what a cache ought to do:**

- C6 moves **exactly C1's nominal traffic** - the same 94,699,520 B of int4
  read once and the same 356,515,840 B of bf16 written - and differs only in
  that the writes all land in one 20.97 MB buffer instead of a 356.52 MB one.
- At `Ns = 2048` C6 takes **0.710 ms against C1's 1.548 - 2.18× faster for
  identical nominal traffic.**
- **The conservative `r + w` normalisation alone breaks the DRAM ceiling:
  878.2 GB/s at `Ns = 1024` and 635.6 at `Ns = 2048`, against the card's
  measured 590 GB/s streaming reference.** A rate above the part's DRAM
  bandwidth is only possible if part of the traffic never reached DRAM.
- **The cutoff sits exactly where the pre-registration put it.** The 41.94 MB
  slab at `Ns = 4096` does not fit the 24 MB L2 and measures **295.8 / 529.5
  GB/s - C1's DRAM numbers to within 1.5%.** The 20.97 MB and 10.49 MB slabs
  do fit and do not. One knob, three cells, and the transition is where the
  cache size says it should be.

**So the physical premise of the lever is confirmed: an L2-sized dequant slab
keeps the write-allocate out of DRAM, and the dequant work itself drops
1.548 → 0.710 ms at the pre-registered width.**

## The GEMM sub-predictions: both HIT

| pre-registered | measured at `Ns = 2048` | outcome |
|---|---:|---|
| ≥ 110 TFLOP/s per slab at 64 WGs | **143.68** | **HIT, 1.31×** |
| slab-GEMM sum within +10% of full width | **+7.0%** (5.082 vs 4.751) | **HIT** |

P2's occupancy scan (8 WGs → 40.59, 16 → 73.50) does not extrapolate to 64
WGs on this shape: the slab GEMM is close to full width, and at `Ns = 1024`
(32 WGs) it is **faster** than full width (−3.3%, 158.91 TFLOP/s) because its
B operand is L2-resident. Occupancy was not the obstacle.

## Verdict: MISS. The lever is rejected - not by the cache, by the handoff.

| `Ns` | C4 − C2 = overhead | ÷ C1 = ratio | × 210.116 = chunk-equivalent | **saved** |
|---:|---:|---:|---:|---:|
| 1024 | 1.732 | 1.119 | 235.1 ms | **−25.0 ms** |
| **2048 (the pre-registered cell)** | **2.065** | **1.334** | **280.3 ms** | **−70.1 ms** |
| 4096 | 2.271 | 1.467 | 308.3 ms | **−98.2 ms** |

| | value |
|---|---|
| pre-registered | overhead **210 → ≤ 70 ms/chunk-equivalent** |
| decision rule | ≥ 90 ms saved → real; 30-90 → priced; **< 30 → dead** |
| **measured, `Ns = 2048`** | **280.3 ms/chunk-equivalent, i.e. 70.1 ms WORSE than the two-pass** |
| verdict | **DEAD.** Not merely below the bar - the wrong side of zero, at every one of the three pre-registered widths. |

### Where the saving goes, itemised (measured, `Ns = 2048`)

| term | ms | |
|---|---:|---|
| dequant work saved by L2 residency | **−0.838** | C6 0.710 vs C1 1.548 |
| slab-GEMM occupancy penalty | **+0.331** | C5a 5.082 vs C2 4.751 |
| **34 host waits** | **+0.760** | 2 per slab × 17, at a **measured 22.35 µs** each |
| interleaving residue | +0.264 | C4 − waits − (C6 + C5a) |
| **net** | **+0.517** | overhead 2.065 vs C1's 1.548 |

**The host handoff is the whole defect, and it is structural.** The lever needs
two host round trips per slab - one so the GEMM does not read a slab the
dequant has not finished writing, one so the next dequant does not overwrite a
slab the GEMM is still reading - because `pf_dequant_tile` runs on the L0
immediate list and `gemm_bf16` on the SYCL queue, and **there is no device-side
ordering primitive between them.** PROBE B measured why: the B70 exposes
**exactly one compute queue** (`numQueues = 1`, index 1 refused with
`ZE_RESULT_ERROR_INVALID_ARGUMENT`), so the two software queues are two views
of one engine ordered only through the host. At **22.35 µs measured** (not
P1's 8.569 µs - this is a full `sycl::queue::wait` plus a
`zeCommandListHostSynchronize`, and it was measured here rather than carried
over), 34 of them cost **0.760 ms - 1.8× the entire L2 saving.**

The second-order term is named too, not hidden: the GEMM's A operand is
`[2048][5120]` bf16 = **20,971,520 B**, re-read by every slab GEMM. A + one
20.97 MB slab is **41.94 MB against a 24 MB L2**, so the slab and A evict each
other. That is the candidate mechanism behind the +0.264 ms interleaving
residue and behind C4 being worse than the sum of its parts; it is stated as a
candidate, not asserted, because no cell here separates it.

### The counterfactual, derived and clearly labelled NOT the verdict

If the handoff cost were zero - i.e. if the dequant and the GEMM could be
ordered on the device instead of through the host - the same measured cells
give (derived from measured; `C4 − 2·slabs·wait`, then the same ratio):

| `Ns` | overhead if handoffs were free | chunk-equivalent | saved |
|---:|---:|---:|---:|
| 1024 | 0.722 ms | 98.0 ms | **+112.1 ms** |
| 2048 | 1.305 ms | 177.1 ms | **+33.0 ms** |
| 4096 | 1.869 ms | 253.7 ms | −43.6 ms |

These are **derived counterfactuals, not measurements of any lever that
exists.** They are conservative in one direction (C1 and C2 each still contain
one ~22 µs wait that is not removed) and optimistic in another (they assume a
free ordering primitive this device has not been shown to have). They are
recorded because they say precisely what a follow-on would have to buy: **a
device-side dependency between the L0 list and the SYCL queue.** Even the best
of them, at `Ns = 1024`, would be **112 ms** - which is the first number in
this whole spec that would clear the ≥ 90 ms the chunk needs, and it rests
entirely on an ordering primitive that is not in hand and would itself need a
probe. At the pre-registered `Ns = 2048` the counterfactual is **33.0 ms**,
which would only reach the "priced, operator rules" band.

## Consequences

- **The 210.116 ms/chunk dequant term stays.** No change to the composition.
- `PrefillScratch` gains **no slab buffer**; plan 6c's L2 stage does **not**
  adopt slab interleaving.
- **The pre-registered launch-overhead correction (+7.2 to +15.4 ms/chunk for
  3424 extra launches) is not the reason this failed** and did not need to be
  applied: the failure is 0.760 ms of host waits on ONE matrix, which scales to
  far more than the launch tax. The launch appends themselves are inside C6 and
  C5a, both of which came in fast.
- **What is now known that was not:** this part's L2 really does absorb a
  20.97 MB dequant slab (measured, 2.18× and above the DRAM ceiling), and the
  slab GEMM at 32-64 WGs is not occupancy-starved (measured, 143.68-158.91
  TFLOP/s). The obstacle to eliminating the write-allocate is **not** the cache
  and **not** the GEMM - it is that this stack cannot order two queues without
  the host. That is the same single fact PROBE B measured, arriving from the
  other side.
