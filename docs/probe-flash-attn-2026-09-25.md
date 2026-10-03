# Probe: flash attention in prefill (spec 6 P0 and P1), 2026-09-25

Spec: `docs/superpowers/specs/2026-09-25-spec6-flash-attention-128k-design.md`.
Plan: `docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md`.

Every number is **measured** unless marked **derived**.

**Winner: pfa_KT64_R16_H6_Q0, 1.219x the composed path at depth 32k, 28.0 TFLOP/s.**
(Derived TFLOP/s; 15.2 % of the 183.45 bf16 DPAS peak. At pos 2048 it only
matches the composed path, 1.004x, so **F1 is in doubt** on this tiling: see §3.4.)

## 1. P0, the baselines

Box: one B70 (device 0, `ZE_AFFINITY_MASK=0`), idle at dispatch (no render-node
holder, no container; `bench_decode.sh` graded every pp row RECORD). Tree: 60baa0e
(production unchanged; the "-dirty" rows differ only by the probe files).
Backend: the build default, `l0-int8`. Script: `tools/probe/attn_baseline.sh`.

Deviations from the plan's draft script, forced by the tools:
- `bench_decode.sh` forwards only `ZE_AFFINITY_MASK` over ssh, so the profiled
  run is a direct `tools/box.sh run` with `B70_PREFILL_PROFILE=1`.
- The phase names are `profile.cc`'s: `attn_QK^T`, `attn_softmax`, `attn_PV`.
- **pp16384 does not fit**: `--bench` refuses depth + tg > max_len, and max_len
  16384 is the only compiled decode attention. The deep rows are pp14336 (7 full
  chunks) and pp16128 (the deepest that fits beside tg 256).

| row | value | kind |
|---|---:|---|
| pp4096 attention phases, profiled (L0 GPU ms): attn_QK^T / attn_softmax / attn_PV | 43.1 / 57.2 / 31.4 | measured (diagnostic) |
| **pp4096 attention phase, qk + sm + pv** | **131.7 ms** | derived from the three above |
| pp4096 attn_prep / attn_gate (outside the kernel spec 6 replaces) | 13.5 / 8.7 ms | measured (diagnostic) |
| pp4096 profiled run total (not a bench row: phase waits) | 2123.6 ms, 1928.81 t/s | measured (diagnostic) |
| **pp4096**, median of 3 | **2109.74 t/s** (1941.5 ms), spread 0.45 % | measured |
| **pp8192**, median of 3 | **2000.14 t/s** (4095.7 ms), spread 0.01 % | measured |
| **pp14336**, median of 3 | **1848.63 t/s** (7754.9 ms), spread 0.13 % | measured |
| **pp16128**, median of 3 | **1798.76 t/s** (8966.2 ms), spread 0.12 % | measured |
| per-chunk increment, chunks at depth 4k-8k: (pp8192 - pp4096) / 2 | 1077.1 ms | derived |
| **per-chunk increment, chunks at depth 8k-14k: (pp14336 - pp8192) / 3** | **1219.7 ms** | derived |
| one 1792-row chunk at depth 14336: pp16128 - pp14336 | 1211.3 ms | derived |
| **decode at depth 4096**, max_len 16384, tg 256, median of 3 | **29.28 t/s** (34.16 ms/token) | measured |
| **decode at depth 16000**, max_len 16384, tg 256, median of 3 | **25.08 t/s** (39.87 ms/token) | measured |

Reading: the attention phase of pp4096 is 131.7 ms against F1's 80 ms bar
(vLLM: 63.1 ms). The softmax alone (57.2 ms) is the largest of the three. A
2048-row chunk costs 142.6 ms more at depth 8k-14k than at 4k-8k (derived,
1219.7 - 1077.1); attention is the only depth-dependent term in a chunk.

## 2. The composed path against fp64

`tools/probe/probe_flash_attn <pos> <C> --arms none`: random Q (bf16 N(0,1) x
qscale) and a random K/V cache (bf16 N(0,1)) in the production layouts, the
production `attn_chunk` on the L0 backend, and an fp64 CPU reference on 25
sampled rows ({0, 1, 7, 8, 63, 64, C/2, C-2, C-1} plus 16 from
`mt19937(pos + C)`) x 24 heads. Per-(row, head) cosine, worst of all pairs.

| pos | C | qscale | composed worst cos | at (h, m) | max abs err |
|---:|---:|---:|---:|---|---:|
| 16384 | 2048 | 1 | 0.999997976 | (17, 695) | 9.16e-05 |
| 777 | 300 | 1 | 0.999997720 | (21, 299) | 6.63e-04 |
| 0 | 2048 | 30 | 0.999996452 | (8, 129) | 8.31e-03 |
| 0 | 64 | 1 | 0.999996685 | (18, 63) | 5.92e-03 |

All measured. **The composed path clears 0.99999 everywhere, so the flash bar
stays at cosine >= 0.99999** (no calibration change). Its own margin is small,
worst 0.9999965. (Section 3: every flash arm lands at or above the composed
path's own worst on every case.)

## 3. The sweep

`tools/probe/probe_flash_attn.cl`, kernel `pfa`, one binary per arm
(`-DKT -DRPW -DHPW -DQREG`, 256 GRF). All arms from the plan's list; every arm
passed the correctness bar before it was timed. Timing: `tools/probe/pfa_sweep.sh`
(detached on the box), device 0, idle (no render-node holder, no container at
dispatch), L0 kernel timestamps summed over each path's launches (Context's
launch profiler; the composed path is 4 x (2 + 8) = 40 launches per call at
C = 2048, an arm is 1), 20-iteration warm-up of every path, then 11 interleaved
rounds (control, then every arm, each round). Ratio = composed / arm, so > 1 is
the arm faster; median and range of the 11 paired ratios.

### 3.1 Correctness, every arm

| case (pos, C, qscale) | composed worst cos | every arm's worst cos | pad rows [C, pad256(C)) |
|---|---:|---:|---|
| 16384, 2048, 1 | 0.999997976 | 0.999998112 (all 8 arms) | finite |
| 777, 300, 1 (depth 1077, partial last tile) | 0.999997720 | 0.999998227 (all 8) | finite (212 rows) |
| 0, 2048, 30 (scores in the hundreds) | 0.999996452 | 0.999999009 (all 8) | finite |
| 0, 64, 1 | 0.999996685 | 0.999998031 (all 8) | finite (192 rows) |
| 2048, 2048, 1 | 0.999997778 | 0.999998232 (all 8) | finite |
| 30720, 2048, 1 | 0.999997996 | 0.999998088 (KT64: 0.999998097) | finite |

All measured. The arms differ only in scheduling, so their outputs are identical
across RPW / HPW / QREG; KT changes the online-softmax tile order, which moves
the max abs error in the fourth digit (KT64 at 16384: 9.412e-05 vs 9.423e-05).
**No arm is under the bar; the stopping rule does not apply.**

### 3.2 The table

Spill: ocloc's "allocated 256 regs and spilled around N", each arm compiled
alone (`tools/probe/pfa_spill.sh`; the parallel build log interleaves them).

| arm | spill | worst cos | ratio pos 2048 | ratio pos 16384 | ratio pos 30720 | TFLOP/s at 30720 |
|---|---:|---:|---:|---:|---:|---:|
| composed (control) | - | 0.999996452 | 1 (5.001 ms) | 1 (28.173 ms) | 1 (69.617 ms) | 23.0 |
| pfa_KT32_R16_H6_Q1 | 121 | 0.999998031 | 0.889x (0.872 .. 0.898) | 0.695x (0.597 .. 0.794) | 0.870x (0.815 .. 1.043) | 19.9 |
| pfa_KT32_R32_H6_Q1 | 121 | 0.999998031 | 0.920x (0.905 .. 0.924) | 0.874x (0.865 .. 0.883) | 1.175x (1.165 .. 1.181) | 27.0 |
| pfa_KT64_R16_H6_Q1 | 148 | 0.999998031 | 0.896x (0.886 .. 0.910) | 0.828x (0.822 .. 0.893) | 1.109x (1.098 .. 1.112) | 25.4 |
| pfa_KT32_R16_H3_Q1 | 121 | 0.999998031 | 0.663x (0.654 .. 0.669) | 0.543x (0.536 .. 0.550) | 0.654x (0.649 .. 0.668) | 15.0 |
| pfa_KT32_R16_H1_Q1 | 121 | 0.999998031 | 0.571x (0.566 .. 0.575) | 0.491x (0.490 .. 0.499) | 0.599x (0.587 .. 0.606) | 13.7 |
| pfa_KT32_R16_H6_Q0 | 58 | 0.999998031 | 1.012x (0.992 .. 1.026) | 0.863x (0.859 .. 0.953) | 1.155x (1.150 .. 1.270) | 26.5 |
| **pfa_KT64_R16_H6_Q0** | 77 | 0.999998031 | **1.004x** (0.979 .. 1.016) | **0.899x** (0.895 .. 0.998) | **1.219x** (1.203 .. 1.224) | **28.0** |
| pfa_KT32_R32_H3_Q1 | 121 | 0.999998031 | 0.870x (0.843 .. 0.885) | 0.638x (0.622 .. 0.694) | 0.774x (0.733 .. 0.808) | 17.8 |

Ratios measured (median of 11 paired rounds, range in brackets); worst cos is
the worst over all six cases of §3.1 (measured); TFLOP/s derived from the causal
FLOP count 4 x 24 x 256 x sum_m (pos + m + 1) = 1597.8 GFLOP at pos 30720 and
the median ms. The composed row's worst cos is its own worst (0, 2048, qscale 30).

### 3.3 The pick

The rule: fastest at pos 30720 among the arms at or above the bar, provided it
is not slower than the composed path at pos 2048. **pfa_KT64_R16_H6_Q0**:
1.219x at pos 30720 (57.143 ms vs 69.617 ms), 28.0 TFLOP/s (derived); at pos
2048 1.004x, a median at parity with a range (0.979 .. 1.016) that straddles 1.
It qualifies on the median. The runner-up, pfa_KT32_R16_H6_Q0, is 1.155x at
32k and 1.012x at 2048.

### 3.4 What the sweep says (for 6b)

- **Heads per work-group is the largest lever, and 6 wins.** H3 and H1 lose
  35-50 % at every depth: the six q-heads of a kv group re-reading the same K/V
  tile from one work-group is worth more than a halved per-WG footprint. (No arm
  stages K/V in SLM yet; the six heads share it only through the cache.)
- **Re-reading Q per KV tile (Q0) beats holding it (Q1)**: Q1 costs 64 GRF and
  spills 121-148; Q0 spills 58-77 and is faster at 2048 and 32k. The spill is
  a timing fact here, not a correctness one.
- **F1 is in doubt on this kernel.** At pos 2048, the second chunk of pp4096,
  the best arm is at parity with the composed path (5.0 ms per FA layer call,
  measured), so this tiling would leave pp4096's 131.7 ms attention phase
  (§1) roughly where it is, against F1's 80 ms. And at pos 16384 every arm is
  slower than the composed path (best 0.899x).
- **F2 is far off:** 28.0 TFLOP/s is 15.2 % of peak against F2's 60 %. The
  composed path itself reaches 30.9-31.1 TFLOP/s at 2k and 16k (derived).
- The one arm with a wide spread, pfa_KT32_R16_H6_Q1 (0.815 .. 1.043 at 32k,
  0.597 .. 0.794 at 16k), is the most register-starved H6 shape; its median
  ranks it last among the H6 arms either way.

Plan 6b therefore starts from pfa_KT64_R16_H6_Q0 as the correct, fastest
measured tiling, with the finding that it does not yet deliver F1: what 6b adds
(SLM-staged K/V shared by the six heads, fewer spills, a longer KV tile) has
to be measured against this row, not assumed.
