# Decode attention at depth - probe record (spec 10a)

Plan: `docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md`. Spec:
`docs/specs/2026-09-28-spec10-decode-attention-at-depth-design.md`. Tools:
`tools/probe/probe_decode_attn.{cc,cl}`, `tools/probe/decode_attn_probe.sh`.

Every number is **measured** unless marked **derived**. Device 0 (`ZE_AFFINITY_MASK=0`),
every GPU job under `flock ~/b70-gpu.lock` (the box is shared with plans 7c and 8b, so
"idle" means "nothing else on the GPU during the locked job"; CPU load from other
agents' builds is recorded as `uptime` beside each table).

## P0 - per-kernel profile (Task 1)

`probe_decode_attn capture`: checkpoint loaded at max_len 131072, `Engine::prefill` of
`tests/golden/prompts/long32k.ids` tiled to 130816 ids (142.8 s), then the **profiled
captured list** (`runtime::build` with `ProfileEvents` over the engine's own buffers)
replayed 8 times per depth with `Control::pos` rewound before every step. Columns are
Σ over the 16 FA layers per step (µs), mean of 8 steps. "step wall" is the unprofiled
list's fence wall at the same depth, median of 8. M = 1.

uptime at start / end: load average 12.79 / 9.51 (other agents' CPU builds; the GPU
was held by this job).

| depth | attn_prep µs | attn_decode µs | attn_reduce µs | attn total ms | decode µs/launch | decode GB/s (KV) | reduce share of attn | step Σ ms (profiled) | step wall ms (plain) | partial table MB/layer (derived) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 55.2 | 2399.1 | 179.0 | 2.633 | 149.9 | 112 | 6.8% | 33.123 | 33.674 | 1.61 |
| 32768 | 52.8 | 15314.4 | 2049.2 | 17.416 | 957.1 | 140 | 11.8% | 47.862 | 48.279 | 12.71 |
| 65536 | 52.9 | 29603.2 | 4086.3 | 33.742 | 1850.2 | 145 | 12.1% | 64.054 | 64.543 | 25.39 |
| 130816 | 52.0 | 57391.6 | 9885.0 | 67.329 | 3587.0 | 149 | 14.7% | 97.426 | 98.045 | 50.65 |

- The plain step wall reproduces spec 6's F3 ms/token (34.15 / 48.54 / 64.68 / 98.13,
  `docs/BENCHMARKS.md`) to within 1.4%, so the profile is of the step F3 measured.
- **Decode attention is 69% of the step at 130816** (67.3 of 97.4 ms), and
  `attn_decode` streams the KV at **149 GB/s**, 25% of the 590 GB/s the weight GEMVs
  reach (decode GB/s = one layer's K + V bytes, (d + 1) x 4 KiB, over µs/launch).
- `attn_reduce` is **9.9 ms/token at 130816** (618 µs per launch), 14.7% of attention:
  the partial table is 50.65 MB per layer written by `attn_decode` and read back by
  `attn_reduce` (derived: 24 x 2045 blocks x 258 x 4 B; 1.62 GB per token written plus
  read, 19% on top of the 8.57 GB of KV). Read at 618 µs it is 82 GB/s - the serial
  walk over 2045 blocks per (head, m), not the bytes, is what it costs.
- `attn_prep` is flat (~53 µs/token); it is not a depth term.
- **M = 4 in the list: not available at max_len 131072.** Production M = 2..4 decode
  lists exist only at max_len 16384 behind `-DB70_DECODE_EXTRA_M=ON` (spec 8a); M = 4
  at depth is measured below as the probe's arm 0, attn.cl compiled at M = 4.

## P1 - the lever sweep (Task 2)

**Harness.** `probe_decode_attn arms` on the dumped real inputs: FA layer 15 (model layer
63) K/V for positions [0, 130820) and, per depth d, the q / gate / production attn_out of
4 consecutive plain steps from pos = d (rows m = 0..3 of an M = 4 verify at d). Arm 0 is
`src/kernels/attn.cl` itself compiled at the probe's M and MAXLEN 131072
(`pda_v1_M<M>`). Timing: L0 kernel timestamps, 10 launches (decode + reduce) per arm per
round, one warm-up round, 5 rounds (7 in the final job) with the arm order rotated per
round; median per arm, and the median **paired** ratio arm 0 / arm within a round. All
arms of a table ran interleaved in one locked job.

**Correctness (A1), every arm, every depth (4096 / 32768 / 130816, and 65536 in the
final job), M = 1 and 4:** worst per-(q head, m) cosine against arm 0 >= 0.9999995
(the bar is 0.99999), max abs error <= 0.0156 (one bf16 ulp at the output's scale), every
arm run twice bitwise identical. Arm 0 is **bitwise identical to production's own dumped
attn_out** at every depth and both M (so M = 4 row m equals the M = 1 step at pos + m).
Every PPW-64 arm (LOAD 0/1/2, DOT 0/1) is **bitwise identical to arm 0**: the block reads
and the shuffle tree keep attn.cl's exact arithmetic order. Arms with PPW > 64 differ
from arm 0 only by the merge reassociation (fewer, longer online-softmax blocks), and at
some depths are still bitwise equal. EXP2 arms change `exp` to `exp2` and are still
>= 0.9999995 (the same worst row as the exp arms).

**Arm 0 against production (Task 2 step 1):** arm 0's decode + reduce is 1039 / 4378 µs
per launch at 32768 / 130816 against the profiled list's 1085 / 4205 µs (-4.2% / +4.1%;
the bar was 3%). The probe re-reads one layer's KV each launch where the list walks 16
layers, and the list's events add a host-scope flush per launch; every comparison below
is therefore a **paired ratio against arm 0 in the same job**, and the derived t/s scale
production's measured attention by that ratio rather than using the probe's absolute µs.

Tags: `P<ppw>_L<load>_D<dot>_F<prefetch>_R<reduce>[_Q<qblk>_G<256grf>_E<exp2>]`
(tools/probe/probe_decode_attn.cl header). P0 = ppw derived on the device from pos and
the target work-groups per kv head `tgt` (control word), `tgt` 64 unless noted.

### Lever 1 alone - positions per work-group (LOAD 0, DOT 0)

µs per launch (decode + reduce), x = paired ratio arm 0 / arm. uptime (load average)
12.41 at start, 12.30 at end (other agents' CPU builds; GPU held by this job).

| arm | 32k M1 | x | 128k M1 | x | 32k M4 | x | 128k M4 | x |
|---|---|---|---|---|---|---|---|---|
| v1 (arm 0) | 1039.1 | 1.000 | 4377.5 | 1.000 | 3309.9 | 1.000 | 12608.0 | 1.000 |
| P64 | 1040.8 | 1.004 | 4279.4 | 1.023 | 3296.1 | 1.009 | 12969.8 | 0.972 |
| P256 | 981.3 | 1.050 | 3783.8 | 1.156 | 3484.8 | 0.947 | 14033.2 | 0.899 |
| P512 | 963.1 | 1.083 | 3769.6 | 1.159 | 3497.0 | 0.946 | 14063.9 | 0.896 |
| P1024 | 960.4 | 1.082 | 3744.1 | 1.169 | 3491.6 | 0.948 | 14021.2 | 0.899 |
| P2048 | 1560.8 | 0.668 | 3551.6 | 1.231 | 6186.1 | 0.535 | 14023.2 | 0.899 |
| P0 (tgt 64) | 1008.1 | 1.021 | 3568.4 | 1.227 | 3749.5 | 0.883 | 14029.3 | 0.899 |

PPW alone removes the reduce (638 -> 18 µs at 128k M1) but not the decode: the decode's
per-position cost is unchanged, and at M = 4 the longer walk per work-group loses
(m-outer walks re-read the block 4 times with fewer work-groups to hide it). Lever 1 is
necessary for the reduce and insufficient alone.

### Lever 2 - loads, then DOT (on PPW 64 / 256 / 1024 / P0)

| arm | 32k M1 | x | 128k M1 | x | 32k M4 | x | 128k M4 | x |
|---|---|---|---|---|---|---|---|---|
| P64_L1 (K block reads) | 1035.1 | 1.008 | 4120.3 | 1.062 | 3166.1 | 1.042 | 12870.9 | 0.979 |
| P64_L2 (+ V 2D block reads) | 812.6 | 1.284 | 3346.4 | 1.307 | 2739.6 | 1.205 | 11391.8 | 1.107 |
| P64_L2_D1 | 428.7 | 2.434 | 1835.1 | 2.385 | 1785.7 | 1.855 | 7366.4 | 1.711 |
| P256_L2 | 713.1 | 1.461 | 2916.8 | 1.501 | 2763.5 | 1.199 | 11388.0 | 1.106 |
| P256_L2_D1 | 305.5 | 3.416 | 1227.9 | 3.564 | 1537.7 | 2.157 | 6346.7 | 1.987 |
| P1024_L2 | 685.9 | 1.521 | 2844.3 | 1.538 | 2810.3 | 1.178 | 11425.9 | 1.103 |
| P1024_L2_D1 | 268.6 | 3.869 | 1126.0 | 3.884 | 1506.8 | 2.197 | 6075.1 | 2.075 |
| P0_L2 | 785.3 | 1.325 | 2835.0 | 1.544 | 3115.5 | 1.064 | 11129.8 | 1.133 |
| P0_L2_D1 | 310.0 | 3.352 | 1118.3 | 3.910 | 1646.7 | 2.015 | 5974.5 | 2.110 |

The V 2D block read (one 512 B message per sub-group per wave, in place of 16 x 32 B)
is the load lever that pays; K's 1D block reads alone buy little. **DOT 1 is the large
lever**: m inner (K/V loaded once for all 6 x M rows), the tree by shuffles, one barrier
per wave, one `exp` per lane per row - 2.4x at PPW 64, 3.9x combined with PPW, at
**bitwise-identical output**.

### Lever 2 cont. - prefetch, q block reads, 256 GRF, exp2 (P0, DOT 1)

(second job; uptime load average 11.33 start, 13.91 end)

| arm | 32k M1 | x | 128k M1 | x | 32k M4 | x | 128k M4 | x |
|---|---|---|---|---|---|---|---|---|
| v1 (arm 0) | 1005.3 | 1.000 | 4379.3 | 1.000 | 3311.5 | 1.000 | 13192.9 | 1.000 |
| P0_L2_D1 | 303.6 | 3.324 | 1128.1 | 3.863 | 1615.4 | 2.043 | 6000.6 | 2.199 |
| + F1 (prefetch) | 317.9 | 3.154 | 1179.9 | 3.698 | 1531.2 | 2.154 | 5781.8 | 2.283 |
| + Q1 (q SLM block reads) | 292.6 | 3.438 | 1100.6 | 3.970 | 1441.0 | 2.294 | 5402.6 | 2.442 |
| + Q1 F1 | 298.1 | 3.370 | 1120.6 | 3.909 | 1372.9 | 2.402 | 5133.0 | 2.570 |
| + Q1 G1 (256 GRF) | 438.0 | 2.295 | 1701.6 | 2.575 | 1862.2 | 1.774 | 6785.2 | 1.946 |
| + Q1 E1 (exp2) | 276.9 | 3.631 | 1001.1 | 4.378 | 1264.8 | 2.612 | 4484.5 | 2.942 |
| P1024_L2_D1_Q1 | 254.9 | 3.942 | 1107.7 | 3.953 | 1276.5 | 2.592 | 5082.6 | 2.598 |

Prefetch (a register double buffer) costs ~4% at M = 1 and buys ~5% at M = 4; 256-GRF
mode halves occupancy and loses 35-50% everywhere (the kernel is latency-hidden by
work-groups, not registers). exp2 buys 10% at M = 1 and 17% at M = 4: the M = 4 kernel is
instruction-bound (119 GB/s at 128k), the M = 1 kernel is at the bandwidth wall (535
GB/s at 128k, 90% of 590).

### Lever 1 again - the device-derived stride (tgt, P0_L2_D1 and P0_L2_D1_Q1)

| tgt | 32k M1 (Q1) | x | 128k M1 (Q1) | x | 32k M4 (Q1) | x | 128k M4 (Q1) | x |
|---|---|---|---|---|---|---|---|---|
| 32 | 270.2 | 3.704 | 1060.3 | 3.945 | 1279.6 | 2.489 | 5216.3 | 2.506 |
| 64 | 292.6 | 3.438 | 1100.6 | 3.970 | 1441.0 | 2.294 | 5402.6 | 2.442 |
| 128 | 333.1 | 3.006 | 1067.4 | 3.904 | 1491.8 | 2.111 | 5316.8 | 2.457 |

tgt 32 (= 128 live work-groups, one per Xe-core slot of four) is best or tied at every
point, and it is one binary for every depth: the stride is read from the control word.

### Lever 3 - the fused reduce (REDUCE 1)

| arm | 32k M1 | 128k M1 | 32k M4 | 128k M4 |
|---|---|---|---|---|
| P0_L2_D1 separate reduce | 310.0 | 1118.3 | 1646.7 | 5974.5 |
| P0_L2_D1 fused (last WG) | 402.2 | 1233.0 | 2801.5 | 10921.7 |
| P1024_L2_D1 separate | 268.6 | 1126.0 | 1506.8 | 6075.1 |
| P1024_L2_D1 fused | 316.0 | 1333.1 | 1686.1 | 6889.6 |

With the stride derived from depth the partial table is 64 blocks per row (3.17 MB per
layer written + read at 128k M1, from 101.3 MB) and the separate `pda_reduce` costs
18-20 µs; the fused reduce puts all 6 x M merges of a kv head on ONE work-group and loses
by 10% (M1) to 83% (M4). **Keep the separate reduce launch.**

**Partial-table traffic, measured:** the reduce's own counter (dbg[0] = nb) reads 2045
partials per row for v1 at 130816 (101.30 MB per layer at M1 written + read, 405.2 MB at
M4) against 64 for P0 tgt 32..64 (3.17 / 12.68 MB); v1's reduce launch is 618-638 µs per
layer against 19-20 µs.

### The final job - the winner against arm 0 at four depths (tgt 32)

7 rounds x 10 launches, interleaved, one locked job; uptime load average 11.35 at start,
9.99 at end. µs per launch (decode + reduce); x = paired median arm 0 / arm; all rows
passed A1 (worst cosine 0.9999995 at 65536 M4, every arm repeatable, arm 0 = production
dump bitwise at every depth).

| arm | 4k M1 | x | 32k M1 | x | 64k M1 | x | 128k M1 | x |
|---|---|---|---|---|---|---|---|---|
| v1 (arm 0) | 121.9 | 1.000 | 1003.7 | 1.000 | 2134.9 | 1.000 | 4379.4 | 1.000 |
| P0_L2_D1 | 46.0 | 2.650 | 277.5 | 3.612 | 556.4 | 3.834 | 1121.3 | 3.918 |
| P0_L2_D1_Q1 | 44.5 | 2.738 | 270.6 | 3.711 | 538.9 | 3.957 | 1093.3 | 4.010 |
| **P0_L2_D1_Q1_E1** | **41.0** | **2.974** | **254.1** | **3.944** | **498.8** | **4.275** | **992.3** | **4.417** |
| P0_L2_D1_F1_Q1_E1 | 41.7 | 2.926 | 257.3 | 3.900 | 504.9 | 4.214 | 1001.2 | 4.380 |

| arm | 4k M4 | x | 32k M4 | x | 64k M4 | x | 128k M4 | x |
|---|---|---|---|---|---|---|---|---|
| v1 (arm 0) | 456.3 | 1.000 | 3420.6 | 1.000 | 6875.5 | 1.000 | 13656.6 | 1.000 |
| P0_L2_D1 | 261.7 | 1.742 | 1557.3 | 2.203 | 3035.2 | 2.263 | 5849.3 | 2.333 |
| P0_L2_D1_Q1 | 234.7 | 1.946 | 1380.2 | 2.486 | 2699.8 | 2.547 | 5270.3 | 2.587 |
| **P0_L2_D1_Q1_E1** | **211.4** | **2.161** | **1233.3** | **2.779** | **2423.1** | **2.836** | **4745.7** | **2.878** |
| P0_L2_D1_F1_Q1_E1 | 213.4 | 2.138 | 1244.4 | 2.745 | 2459.2 | 2.807 | 4817.1 | 2.841 |

The winner's KV stream at M1 is 409 / 528 / 538 / 540 GB/s (4k / 32k / 64k / 128k),
against v1's 122-138; its partial table at 128k is 32 blocks per row, **1.59 MB per
layer written + read (M1), from 101.30 MB**, and its reduce launch is 9 µs from 640.

### Derived decode t/s (Task 2 step 3)

Derived: production's plain step wall at the depth (P0) minus production's profiled
`attn_decode` + `attn_reduce` (P0, Σ 16 layers), plus that attention divided by the
winner's paired ratio from the final job. M = 1.

| depth | step today ms | attention today ms | winner (E1) attention ms | derived step ms | **derived t/s** | exp-only (Q1, E0) t/s | today t/s | A-F3 bar |
|---|---|---|---|---|---|---|---|---|
| 4096 | 33.674 | 2.578 | 0.867 | 31.963 | **31.29** | 31.21 | 29.70 | (A-F4: no regression) |
| 32768 | 48.279 | 17.364 | 4.403 | 35.318 | **28.31** | 28.09 | 20.71 | 23.6 |
| 65536 | 64.543 | 33.690 | 7.881 | 38.734 | **25.82** | 25.40 | 15.49 | 21.0 |
| 130816 | 98.045 | 67.277 | 15.231 | 46.000 | **21.74** | 21.03 | 10.20 | 17.3 |

**M = 4 (recorded, spec 8's verify cost):** 16 launches of the winner are 3.38 / 19.73 /
38.77 / 75.93 ms per verify step at 4k / 32k / 64k / 128k, against v1's 7.30 / 54.73 /
110.01 / 218.51 ms (probe µs x 16; no production M = 4 list exists at max_len 131072 to
calibrate against). At M = 4 the kernel is instruction-bound (113 GB/s at 128k): four
rows cost 4.8x one row, so the KV read is not what M = 4 pays for.

## The v2 design verdict (Task 3)

**Verdict: build v2 as `P0_L2_D1_Q1_E1` (spec 10b).** Every A-F3 bar is met in the
derived step with margin - **28.31 / 25.82 / 21.74 t/s against 23.6 / 21.0 / 17.3** at
32768 / 65536 / 130816 (+20% / +23% / +26% over the bars) - and the shallow step gets
faster, not slower (4096: 31.29 against 29.70 t/s, A-F4 satisfied at max_len 131072;
the kernel has no MAXLEN-dependent cost beyond its idle-grid size, so 16384 is expected
to follow and is 10b's gate to measure). No stop condition applies.

**The design.**
- **`attn_decode_v2`**: grid (4 kv heads, MAXLEN / 64), work-group 256 = 16 sub-groups
  of 16, `reqd_sub_group_size(16)`, 128 GRF (256 GRF measured 35-50% slower). Work-group
  `(j, b)` owns positions `[b * ppw, min((b + 1) * ppw, pos + n_active))`; the rest exit
  on their first instruction.
  - **The stride**: `ppw = max(64, roundup64(ceil((pos + n_active) / tgt)))` with `tgt`
    = 32 work-groups per kv head (128 live, the device's resident slots). For 10b it is
    computed by `attn_prep` (work-group 0, one lane) into a new `Control` word
    (`Control::pad[0]` -> `attn_stride`, CTRL index 19 as in the probe) so decode and
    reduce read one value; the probe computed it in both kernels from the same words,
    which is equivalent. tgt 32 was best or tied at every depth and M (tgt 64 / 128 up to
    9% / 23% slower at 32k).
  - **Loads**: K by two `intel_sub_group_block_read_us8` per row (lane l, element t = dim
    l + 16 t, attn.cl's order); V by one `intel_sub_group_2d_block_read_16b_16r16x1c`
    per sub-group per wave (16 positions x 16 dims, lane = dim = lid); masked positions
    zeroed after the load. No prefetch (it costs 1-4% at M = 1).
  - **The wave (DOT 1)**: m inner - one K/V wave feeds all 6 x M rows; q staged once in
    SLM (6 KB x M) and read by SLM block reads; the score tree by shuffles (attn.cl's
    pairwise adds); **one** barrier per wave publishing all 6 x M raw dots
    (double-buffered SLM); lane s computes position s's weight, broadcast in ascending s
    for the sum and the V FMA.
  - **exp2** with 1/16 x log2(e) folded into the score scale (spec 10 lever 4; spec 6c's
    approved form), `mx` and the merge in the log2 domain. It is worth 10% at M = 1 and
    11% at M = 4 over the exp form; the exp form (`P0_L2_D1_Q1`) also clears every bar
    (28.09 / 25.40 / 21.03 t/s) if the operator prefers attn.cl's `exp`.
- **`attn_reduce_v2`**: the separate launch stays (grid (24, M)); `nb = (pos + m) / ppw
  + 1`, at most 32-33 blocks for any depth at tgt 32, attn.cl's ascending merge in the
  log2 domain. **The fused last-work-group reduce is rejected** (10-83% slower: it puts
  every merge of a kv head on one work-group).
- **Partials**: `attn_part` keeps its `[24][MAXLEN/64][M][258]` allocation; only the
  first `nb` blocks are touched (1.59 MB per layer at 128k M1).

**Correctness evidence for 10b's A1**: at PPW 64 the v2 wave is bitwise attn.cl; with the
derived stride the only changes are the block boundaries (fewer, longer online-softmax
runs) and exp2 - worst cosine 0.9999995 over 4 depths x M = 1, 4 x 24 heads, max abs
0.0156, bitwise repeatable. A2/A3 (golden gates, passkey, spec 8's M2) are 10b's.

**Determinism**: no atomics in the chosen design; every order is fixed by the stride,
which is a function of `pos` and `n_active` only - so two replays are bitwise identical
(measured), and M = 4 row m and the M = 1 step at pos + m see the same stride only when
`ceil((pos + 4) / 32)` and `ceil((pos + m + 1) / 32)` round to the same multiple of 64 -
**10b must make the stride a function of `pos` alone** (e.g. `ceil((pos + 4) / tgt)`
for every M) if spec 8's M2 (verify rows bitwise against M = 1) is to hold.

**What M = 4 still costs (recorded, spec 8)**: 75.9 ms of attention per verify step at
128k (from 218.5), 19.7 at 32k. The M = 4 kernel is instruction-bound, not
bandwidth-bound: per wave every sub-group recomputes the same 6 x M softmax weights (16x
redundancy in `exp` and the broadcasts). The next lever for M > 1, if spec 8 needs it,
is to compute each (row, position) weight once and publish it (a second barrier) or a
DPAS score/PV path over the 6 x M rows - a 10b-or-later probe, not needed for the bars.
