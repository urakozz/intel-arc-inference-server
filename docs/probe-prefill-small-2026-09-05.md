# T6 - direct M=2048 widened-small-kernel measurements

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held).

`tools/probe/probe_prefill_small` compiles probe-only M=2048 variants from the
unchanged production `src/kernels/prep.cl` source. No runtime variant helper
names these binaries. Inputs are finite, incompressible random data; timing is
the Level Zero kernel timestamp median of eight replays after dropping the
first three (measured, iterate grade).

## One harness defect, one fix, one measurement

The first run returned `ZE_RESULT_ERROR_DEVICE_LOST` at the fence and produced
no timing. Its cause was concrete: the probe allocated `gdn_o` as bf16, while
`prep_gated_head` declares it `const float*` (source-verified, iterate grade).
The probe allocation and random fill were changed to fp32; the following table
is the one post-fix measurement, not a retry of the invalid run.

| kernel / M=2048 variant | grid work-groups | us/launch | calls/chunk | ms/chunk | grade |
|---|---:|---:|---:|---:|---|
| `prep_res_fold`, SP4 | 20 × 2048 = 40960 | 381.354 | 129 | 49.195 | measured launch time; derived chunk time, iterate |
| `prep_norm_finish` | 20 × 2048 = 40960 | 133.333 | 129 | 17.200 | measured launch time; derived chunk time, iterate |
| `prep_silu_mul` | 5 × 2048 = 10240 | 4392.396 | 64 | 281.113 | measured launch time; derived chunk time, iterate |
| `prep_gated_head` | 48 × 2048 = 98304 | 278.750 | 48 | 13.380 | measured launch time; derived chunk time, iterate |
| **four directly measured terms** | - | - | - | **360.888** | derived sum of measured terms, iterate |

`prep_res_fold` uses the measured SP4 row for all 129 calls, a conservative
derived price because only the initial no-previous-partials call is SP0. The
four rows remove 2419.669 ms from the former literal-decode upper bound
(derived from its 2532.002 ms upper endpoint and the 112.333 ms residual below).

`attn_prep` and `embed_gather` remain unmeasured at M=2048. Their retained
literal-decode upper bound is 112.333 ms (derived from 3.2 µs × 2048 × 16 plus
3.65 µs × 2048); their lower bound remains 0 ms (derived). The small-kernel
term is consequently tightened from 0.000-2532.002 ms to
360.888-473.221 ms (mixed: direct measurements plus the stated derived
residual range, iterate grade).

---

## Pre-registration: the S = 1 runtime-`M` kernels at M = 2048 (2026-09-05)

**Written before any of these kernels was measured.** The table above measured
the *decode* binaries at M = 2048, i.e. at decode's split-K: `prep_silu_mul`
folds `SILU_S` = 8 slices (`src/kernels/prep.cl:84`, `:326`) and `prep_res_fold`
was measured at SP4. The prefill path runs **S = 1** (plan 6b ruling R1), so two
of those four rows carry 8x and 4x the traffic prefill will generate. This
section predicts what the S = 1 runtime-`M` kernels
(`src/kernels/prefill/pf_*.cl`) will cost, so that the measurement that follows
is scored rather than merely reported.

### The inherited prediction, and where its arithmetic does not close

The controller's corrected term (progress.md, 2026-09-05) is `silu_mul` ~39 ms,
`res_fold` ~13, `norm_finish` ~6, `gated_head` ~3 -> **~60-80 ms/chunk**. Those
four numbers are carried here as the **graded** prediction. Two of them have no
mechanism, and saying so before the measurement is the point of writing it down:

- **`norm_finish` never reads `partials`.** Its arguments are `sumsq`, `resid`,
  `norm_w`, `x_out` (`prep.cl:281-284`); `S_PREV` does not appear in its body.
  Its traffic at M = C is decode's widened, unchanged, whatever `S` is. There is
  no route from the S = 1 correction to 17.2 -> 6 ms.
- **`gated_head`'s `GATED_S` is ALREADY 1** in decode (`prep.cl:93`) -- qkv||z
  runs unsplit in the decode step too. Same conclusion: the correction cannot
  reach it, and 13.4 -> 3 ms has no mechanism.

### Byte arithmetic (derived), at M = 2048 and 590 GB/s

590 GB/s is docs/01's **measured** in-situ figure (docs/01:46), not its 600 GB/s
rounded planning number; the achieved-rate column beside it uses each kernel's
OWN measured rate from the decode table above, which is the tighter bound where
one exists. Bytes are unique bytes moved per launch (a value re-read inside one
launch is counted once; `norm_w`, `sumsq` and the RoPE table are cache-resident
and counted once each).

| kernel | bytes/launch | calls/chunk | ms at 590 GB/s | ms at its measured rate | graded (inherited) |
|---|---:|---:|---:|---:|---:|
| `pf_silu_mul` | 285,212,672 R + 71,303,168 W = **356.5 MB** | 64 | 38.68 | 42.59 (535.7 GB/s) | ~39 |
| `pf_res_fold` SP1 | 41,943,040 + 20,971,520 R + 20,971,520 + 163,840 W = **84.05 MB** | 129 | 18.38 | 19.70 (550.3 GB/s) | ~13 |
| `pf_norm_finish` | 20,971,520 R + 20,971,520 W + 184,320 = **42.13 MB** | 129 | 9.21 | 17.20 (316.0 GB/s) | ~6 |
| `pf_gated_head` | 50,331,648 + 50,331,648 R + 25,165,824 W = **125.83 MB** | 48 | 10.24 | 13.38 (451.4 GB/s) | ~3 |
| `pf_attn_prep` | 117,440,512 R + 109,051,904 W + 524,288 = **227.0 MB** | 16 | 6.16 | -- (never measured) | -- |
| `pf_embed_gather` | 20,971,520 R + 20,971,520 W = **41.94 MB** | 1 | 0.07 | -- | -- |
| `pf_ab_proj` | 20,971,520 R + 1,310,720 + 1,048,576 W = **23.33 MB** | 48 | 1.90 | -- | -- |
| **total** | | | **84.64** | **~109** | ~60-80 |

Two rows carry a stated uncertainty rather than a point:

- **`pf_ab_proj`** is the one kernel here that is not obviously bandwidth-bound.
  Its 23.33 MB assumes the 1.31 MB weight tile and the 20.97 MB activation
  rectangle are each pulled from DRAM once, i.e. that L2 (24 MB, docs/01) serves
  the 8x activation reuse across the eight column tiles and the 256x weight
  reuse across the M tiles. If it serves neither, the launch moves 190 MB and
  the term is 15.5 ms rather than 1.90. **Pre-registered band: 1.9-15.5
  ms/chunk.** It is also the kernel whose decode tiling ({16, 16}) was bought to
  fix a thread shortage that does not exist at M = 2048 (2048 work-groups against
  decode's 8), so a miss here is a retuning question, not a defect.
- **`pf_attn_prep`** was never measured at M = 2048 in any form; T6 carried it
  inside a 112.333 ms literal-decode upper bound. 6.16 ms is its roofline.

### The prediction this section is scored on

**~85-110 ms/chunk** (derived, this section's own arithmetic) against the
inherited **~60-80** (derived, progress.md). Both are recorded; the measurement
scores both. A kernel that misses its figure by more than 2x is reported as a
miss with its achieved GB/s and is **not** tuned in this pass.

---

## MEASURED: the S = 1 runtime-`M` kernels at M = 2048 (2026-09-05)

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held). Device
kernel timestamps, 8 replays with the first 3 dropped, median; finite
incompressible random inputs. `tools/probe/probe_prefill_small` now runs **both
batteries in one process**, so the comparison between them carries no
session-to-session drift: the decode-S control re-measured at **361.319 ms**
against the 360.888 ms recorded above, **+0.12%**, which is the same drift band
the untouched launches showed in spec 1.5's lever runs.

The prefill rows are the **production** binaries from `src/kernels/prefill/` --
the objects `tests/prefill/pf_*_test.cc` grade bit-for-bit against decode at
M = 1 -- not probe-only recompiles. No `M` is in any of their names or on any of
their ocloc command lines.

| kernel | grid work-groups | MB/launch | us/launch | GB/s | calls/chunk | ms/chunk |
|---|---:|---:|---:|---:|---:|---:|
| `pf_silu_mul` | 5 x 2048 = 10240 | 356.52 | 757.083 | 470.9 | 64 | 48.453 |
| `pf_res_fold` SP1 | 20 x 2048 = 40960 | 84.05 | 178.750 | 470.2 | 129 | 23.059 |
| `pf_norm_finish` | 20 x 2048 = 40960 | 42.13 | 136.562 | 308.5 | 129 | 17.617 |
| `pf_gated_head` | 48 x 2048 = 98304 | 125.83 | 280.000 | 449.4 | 48 | 13.440 |
| `pf_attn_prep` | 28 x 2048 = 57344 | 227.02 | 521.667 | 435.2 | 16 | 8.347 |
| `pf_ab_proj` | 8 x 256 = 2048 | 23.33 | 402.500 | 58.0 | 48 | 19.320 |
| `pf_embed_gather` | 1 x 2048 = 2048 | 41.94 | 110.833 | 378.4 | 1 | 0.111 |
| **total** | | | | | | **130.346** |

Control, same process, decode's split-K: `prep_res_fold` SP4 49.208 ·
`prep_norm_finish` 17.079 · `prep_silu_mul` S8 281.607 · `prep_gated_head`
13.425 = **361.319 ms/chunk**.

**The small-kernel term is 130.346 ms/chunk, measured, with no residual range**
-- `attn_prep` and `embed_gather` were the two kernels T6 had to bound at
0.000-112.333 ms and both are now direct rows, so the interval is gone.

### Scoring the pre-registration

| kernel | inherited (graded) | own byte-derived | measured | verdict |
|---|---:|---:|---:|---|
| `pf_silu_mul` | ~39 | 38.68-42.59 | 48.453 | over by 1.14-1.25x |
| `pf_res_fold` SP1 | ~13 | 18.38-19.70 | 23.059 | inherited MISS 1.77x; own over by 1.17-1.25x |
| `pf_norm_finish` | ~6 | 9.21-17.20 | 17.617 | inherited MISS 2.94x; own **HIT** at its upper end |
| `pf_gated_head` | ~3 | 10.24-13.38 | 13.440 | inherited MISS 4.48x; own **HIT** (13.38 vs 13.44, 0.4%) |
| `pf_attn_prep` | -- | 6.16 | 8.347 | over by 1.36x |
| `pf_ab_proj` | -- | 1.90-15.50 | 19.320 | over by 1.25x on the band's top, 10.2x on its roofline |
| `pf_embed_gather` | -- | 0.07 | 0.111 | over by 1.59x (0.04 ms absolute) |
| **total** | **~60-80** | **~85-110** | **130.346** | inherited MISS 1.63-2.17x; own MISS 1.19-1.53x |

**Both pre-registrations are misses and the smaller one is mine.** Two specific
things the pre-registration got right and one it got wrong:

- **Right, and it was the load-bearing claim:** `norm_finish` and `gated_head`
  could not improve, because neither reads `partials` at a split-K stride --
  `norm_finish` never reads `partials` at all and `gated_head`'s `GATED_S` is
  already 1 in decode. Measured: 17.079 -> 17.617 (+3.1%) and 13.425 -> 13.440
  (+0.1%). The inherited ~6 and ~3 ms figures were 2.9x and 4.5x out, and they
  are 22.1 ms of the 50.3 ms by which the inherited total misses.
- **Right:** the two kernels the S = 1 correction *does* reach moved by close to
  their traffic ratio -- `silu_mul` 281.607 -> 48.453 (**5.81x** of time against
  **6.60x** of bytes: the reads fall 8x but the 71 MB write is unchanged) and
  `res_fold` 49.208 -> 23.059 (**2.13x** against **2.50x**). Both fall short of
  their traffic ratio for the same reason: a launch that moves less data
  sustains a lower rate (534.8 -> 470.9 GB/s and 550.2 -> 470.2 GB/s), so
  removing 6.60x of bytes buys 5.81x of time. That second-order effect was not
  in the pre-registered arithmetic and it is the bulk of its miss.
- **The `+3.1%` on `norm_finish` is the price of the runtime `M`** and is
  reported rather than absorbed: `sumsq[(size_t)g * m_count + m]` is one runtime
  multiply per lane where the decode binary folds a constant. It is above the
  +0.12% drift this run measured, so it is real. It buys one binary for every
  chunk width, which is what interfaces.md asks for.

### The one row that is not bandwidth-bound: `pf_ab_proj`

19.320 ms/chunk at **58.0 GB/s**, 9.8% of the device's measured 590 and 5.5x its
own 1.90 ms roofline. Not tuned, per the brief's rule, and reported with the
mechanism as a *hypothesis*, not a conclusion:

- 2.684 GFLOP per launch in 402.500 us is **6.67 TFLOP/s** of scalar fp32 FMA,
  and the inner loop issues two loads (one `intel_sub_group_block_read_us8`, one
  `vload8`) per eight FMAs, so it is plausibly issue-bound rather than
  DRAM-bound. Nothing here measures that.
- Its tiling `{COLS_PER_WG 16, KSPLIT 16}` was bought for a shortage that no
  longer exists: at M = 1 `a||b` had **8 subgroups for the whole launch**
  (docs/15 SS L2), and KSPLIT 16 took it 48.774 -> 5.340 us. At M = 2048 the
  grid is 2048 work-groups with 8 rows each; the K-split now costs an 8 KB SLM
  tree and a four-level barrier chain to solve a problem that is gone.
- **It is also the kernel most likely to be deleted rather than tuned.** `a||b`
  is a GEMM shape at M = 2048 (2048 x 5120 x 128), and plan 6c's `gemm_bf16` is
  what should run it -- the same argument that retires `pf_gemv_int4_M`. The
  19.320 ms is therefore an upper bound on a term L2 may remove entirely.

### What the composition should now carry

**130.346 ms/chunk (measured, iterate)** for the whole small-kernel family at
C = 2048, replacing 360.888-473.221. The recomposed ceiling is in
`docs/superpowers/specs/2026-09-04-spec2-stage0-ruling-request.md`.
