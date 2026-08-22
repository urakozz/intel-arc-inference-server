# `probe_gemv` raw matrix - 2026-08-23

The complete output of `tools/probe/probe_gemv`, run 1, verbatim. This is the
record the layout and `S` decisions in [12-kernels.md](12-kernels.md),
[05-perf-model.md](05-perf-model.md), [02-formats.md](02-formats.md),
[08-decode-vs-prefill.md](08-decode-vs-prefill.md) and spec §6.3 were taken
from; every number quoted in those documents is a cell below or arithmetic on
one.

```bash
tools/box.sh run ./build/tools/probe/probe_gemv | tee docs/probe-gemv-2026-08-23.md
```

**Conditions.** Box `CL_DRIVER_VERSION 26.27.39122.14`, kernels AOT-compiled by
`ocloc -device bmg-g31`, host `-O2 -Wall -Wextra -Werror`. **M = 1 only.**
Weights are random (the B70 compresses uniform fills - doc 01). GB/s counts
weight bytes (int4 nibbles + f16 scales) against `us_per_launch`; the denominator
`600 GB/s` is doc 01's roofline figure, and `probe_bw` measures 590 GB/s through
the same launch path. Each configuration is 40 launches in one regular command
list replayed 8 times, median of the last 5, cycling
`NB = max(2, 72 MB / weight_bytes + 1)` weight copies so the 24 MB L2 is missed.
**No `WRONG` row: all 50 int4 configurations matched the CPU reference.**

**Verbatim note.** The version of the probe that produced this run printed the
fused shape names with `|` (`q|k|v`, `qkv|z`, `gate|up`), so the table below does
not render as markdown. It is kept exactly as emitted rather than tidied - this
file is the measurement record. The probe now prints `‖` instead, so future runs
render.

**Stability.** A second full run the same day agreed on every cell to within
2.11% (median 0.21%) and produced an identical decision block. Run 1 is the
recorded one.

---

| shape | K×N | L | S | µs | GB/s | % of 600 | max abs err | tol |
|---|---|---|---|---|---|---|---|---|
| out/o_proj | 5120×5120 | 0 | 1 | 64.6 | 215 | 36% | 1.1e-05 | 0.0037 |
| out/o_proj | 5120×5120 | 0 | 2 | 32.4 | 430 | 72% | 7.6e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 0 | 4 | 25.4 | 549 | 92% | 5.7e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 0 | 8 | 28.8 | 484 | 81% | 4.8e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 0 | 16 | 26.1 | 533 | 89% | 7.6e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 1 | 1 | 53.9 | 259 | 43% | 1.1e-05 | 0.0037 |
| out/o_proj | 5120×5120 | 1 | 2 | 31.0 | 449 | 75% | 7.6e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 1 | 4 | 28.7 | 485 | 81% | 5.7e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 1 | 8 | 30.6 | 455 | 76% | 4.8e-06 | 0.0037 |
| out/o_proj | 5120×5120 | 1 | 16 | 26.5 | 526 | 88% | 7.6e-06 | 0.0037 |
| q|k|v | 5120×14336 | 0 | 1 | 78.3 | 498 | 83% | 1.3e-05 | 0.0045 |
| q|k|v | 5120×14336 | 0 | 2 | 67.7 | 576 | 96% | 9.5e-06 | 0.0045 |
| q|k|v | 5120×14336 | 0 | 4 | 71.0 | 549 | 91% | 4.3e-06 | 0.0045 |
| q|k|v | 5120×14336 | 0 | 8 | 71.9 | 542 | 90% | 5.7e-06 | 0.0045 |
| q|k|v | 5120×14336 | 0 | 16 | 72.0 | 541 | 90% | 5.7e-06 | 0.0045 |
| q|k|v | 5120×14336 | 1 | 1 | 72.0 | 542 | 90% | 1.3e-05 | 0.0045 |
| q|k|v | 5120×14336 | 1 | 2 | 73.0 | 534 | 89% | 9.5e-06 | 0.0045 |
| q|k|v | 5120×14336 | 1 | 4 | 72.7 | 536 | 89% | 4.3e-06 | 0.0045 |
| q|k|v | 5120×14336 | 1 | 8 | 72.0 | 541 | 90% | 5.7e-06 | 0.0045 |
| q|k|v | 5120×14336 | 1 | 16 | 72.9 | 535 | 89% | 5.7e-06 | 0.0045 |
| qkv|z | 5120×16384 | 0 | 1 | 172.9 | 258 | 43% | 1.1e-05 | 0.0043 |
| qkv|z | 5120×16384 | 0 | 2 | 113.6 | 392 | 65% | 9.5e-06 | 0.0043 |
| qkv|z | 5120×16384 | 0 | 4 | 113.9 | 391 | 65% | 5.7e-06 | 0.0043 |
| qkv|z | 5120×16384 | 0 | 8 | 108.4 | 411 | 69% | 4.3e-06 | 0.0043 |
| qkv|z | 5120×16384 | 0 | 16 | 106.0 | 420 | 70% | 7.6e-06 | 0.0043 |
| qkv|z | 5120×16384 | 1 | 1 | 79.5 | 560 | 93% | 1.1e-05 | 0.0043 |
| qkv|z | 5120×16384 | 1 | 2 | 93.9 | 475 | 79% | 9.5e-06 | 0.0043 |
| qkv|z | 5120×16384 | 1 | 4 | 90.3 | 493 | 82% | 5.7e-06 | 0.0043 |
| qkv|z | 5120×16384 | 1 | 8 | 83.7 | 532 | 89% | 4.3e-06 | 0.0043 |
| qkv|z | 5120×16384 | 1 | 16 | 82.4 | 541 | 90% | 7.6e-06 | 0.0043 |
| gate|up | 5120×34816 | 0 | 1 | 215.1 | 440 | 73% | 1.3e-05 | 0.0044 |
| gate|up | 5120×34816 | 0 | 2 | 183.9 | 515 | 86% | 7.6e-06 | 0.0044 |
| gate|up | 5120×34816 | 0 | 4 | 174.7 | 542 | 90% | 5.2e-06 | 0.0044 |
| gate|up | 5120×34816 | 0 | 8 | 170.8 | 554 | 92% | 5.7e-06 | 0.0044 |
| gate|up | 5120×34816 | 0 | 16 | 172.3 | 549 | 92% | 7.6e-06 | 0.0044 |
| gate|up | 5120×34816 | 1 | 1 | 222.9 | 425 | 71% | 1.3e-05 | 0.0044 |
| gate|up | 5120×34816 | 1 | 2 | 192.7 | 491 | 82% | 7.6e-06 | 0.0044 |
| gate|up | 5120×34816 | 1 | 4 | 176.2 | 538 | 90% | 5.2e-06 | 0.0044 |
| gate|up | 5120×34816 | 1 | 8 | 172.4 | 549 | 92% | 5.7e-06 | 0.0044 |
| gate|up | 5120×34816 | 1 | 16 | 175.6 | 539 | 90% | 7.6e-06 | 0.0044 |
| down | 17408×5120 | 0 | 1 | 219.6 | 216 | 36% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 0 | 2 | 113.9 | 416 | 69% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 0 | 4 | 83.7 | 566 | 94% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 0 | 8 | 105.0 | 451 | 75% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 0 | 16 | 87.9 | 539 | 90% | 1.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 1 | 181.0 | 262 | 44% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 1 | 2 | 100.9 | 469 | 78% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 4 | 109.8 | 431 | 72% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 1 | 8 | 105.6 | 448 | 75% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 1 | 16 | 88.6 | 534 | 89% | 1.3e-05 | 0.0067 |
| lm_head bf16 | 5120×248320 | B | 1 | 4349.5 | 585 | 97% | - | - |

Decision:
- layout 0: sum of best GB/s over shapes = 2666
- layout 1: sum of best GB/s over shapes = 2712
- canonical layout: 1
- out/o_proj (5120×5120): S = 16
- q|k|v (5120×14336): S = 1
- qkv|z (5120×16384): S = 1
- gate|up (5120×34816): S = 4
- down (17408×5120): S = 16
