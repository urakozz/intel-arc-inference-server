# Probe: flash attention in prefill (spec 6 P0 and P1), 2026-09-25

Spec: `docs/specs/2026-09-25-spec6-flash-attention-128k-design.md`.
Plan: `docs/superpowers/plans/2026-09-25-spec6a-flash-attn-baseline-and-probe.md`.

Every number is **measured** unless marked **derived**.

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
worst 0.9999965: the single bf16 rounding of the normalised P is the dominant
error, and a flash kernel that rounds the unnormalised P should sit in the same
band.
