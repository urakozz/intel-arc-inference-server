# `probe_gemv` 51-row matrix, re-run warm - 2026-08-26

A verbatim re-run of the instrument that produced
[probe-gemv-2026-08-24.md](probe-gemv-2026-08-24.md), on a warm device, taken so
that spec 1.7's load-path battery could be read against same-session numbers
rather than a two-day-old record - and so that the layout sums quoted beside the
battery are committed rather than floating.

**It reproduces the committed record**: median **0.21%** per cell (which is the
figure docs/12 already quotes for this instrument's run-to-run repeatability),
worst cell +4.6% at `qkv‖z` layout 0 `S` = 2, and an **identical decision
block** - canonical layout 1, `S` = 16 / 1 / 1 / 4 / 16. The five layout-0
`S` = 1 cells, which are the first configuration measured for each shape and so
the ones a clock ramp would hit first, come back 211 / 496 / 256 / 443 / 214
against the record's 211 / 492 / 257 / 442 / 214. **The 2026-08-24 record is
sound and nothing in it needs restating.**

Layout sums, for anyone quoting them: this run reads **2668 (layout 0) / 2717
(layout 1)**; the committed 2026-08-24 run reads **2667 / 2713** and its second
run 2667 / 2715. Quote a pair from one file.

Same conditions as the battery - box `CL_DRIVER_VERSION 26.27.39122.14`, `ocloc
-device bmg-g31`, idle, M = 1, random weights, GB/s over weight bytes only.

---

| shape | K×N | L | S | µs | GB/s | % of 600 | max abs err | tol |
|---|---|---|---|---|---|---|---|---|
| out/o_proj | 6144×5120 | 0 | 1 | 79.3 | 211 | 35% | 1.3e-05 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 2 | 39.6 | 422 | 70% | 7.6e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 4 | 30.3 | 551 | 92% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 8 | 34.2 | 488 | 81% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 16 | 31.3 | 534 | 89% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 1 | 64.4 | 259 | 43% | 1.3e-05 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 2 | 36.9 | 453 | 76% | 7.6e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 4 | 35.8 | 466 | 78% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 8 | 36.8 | 454 | 76% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 16 | 31.3 | 534 | 89% | 5.7e-06 | 0.0048 |
| q‖k‖v | 5120×14336 | 0 | 1 | 78.6 | 496 | 83% | 1.3e-05 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 2 | 67.8 | 575 | 96% | 9.5e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 4 | 70.9 | 550 | 92% | 4.3e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 8 | 72.3 | 539 | 90% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 16 | 71.9 | 542 | 90% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 1 | 72.2 | 540 | 90% | 1.3e-05 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 2 | 79.0 | 494 | 82% | 9.5e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 4 | 72.5 | 538 | 90% | 4.3e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 8 | 72.1 | 541 | 90% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 16 | 72.7 | 537 | 89% | 5.7e-06 | 0.0045 |
| qkv‖z | 5120×16384 | 0 | 1 | 174.1 | 256 | 43% | 1.1e-05 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 2 | 108.4 | 411 | 69% | 9.5e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 4 | 108.5 | 411 | 68% | 5.7e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 8 | 106.8 | 417 | 70% | 4.3e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 16 | 106.2 | 420 | 70% | 7.6e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 1 | 79.7 | 559 | 93% | 1.1e-05 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 2 | 91.7 | 486 | 81% | 9.5e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 4 | 86.5 | 515 | 86% | 5.7e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 8 | 83.1 | 537 | 89% | 4.3e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 16 | 82.4 | 541 | 90% | 7.6e-06 | 0.0043 |
| gate‖up | 5120×34816 | 0 | 1 | 213.8 | 443 | 74% | 1.3e-05 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 2 | 184.0 | 515 | 86% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 4 | 175.1 | 541 | 90% | 5.2e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 8 | 170.7 | 555 | 92% | 5.7e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 16 | 172.8 | 548 | 91% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 1 | 224.0 | 423 | 70% | 1.3e-05 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 2 | 192.8 | 491 | 82% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 4 | 176.7 | 536 | 89% | 5.2e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 8 | 172.4 | 549 | 92% | 5.7e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 16 | 176.0 | 538 | 90% | 7.6e-06 | 0.0044 |
| down | 17408×5120 | 0 | 1 | 221.4 | 214 | 36% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 0 | 2 | 113.9 | 416 | 69% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 0 | 4 | 83.5 | 567 | 94% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 0 | 8 | 104.2 | 454 | 76% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 0 | 16 | 88.3 | 536 | 89% | 1.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 1 | 181.1 | 261 | 44% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 1 | 2 | 101.0 | 469 | 78% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 4 | 98.6 | 480 | 80% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 1 | 8 | 105.8 | 447 | 75% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 1 | 16 | 88.8 | 533 | 89% | 1.3e-05 | 0.0067 |
| lm_head bf16 | 5120×248320 | B | 1 | 4350.2 | 585 | 97% | - | - |

Decision:
- layout 0: sum of best GB/s over shapes = 2668
- layout 1: sum of best GB/s over shapes = 2717
- canonical layout: 1
- out/o_proj (6144×5120): S = 16
- q‖k‖v (5120×14336): S = 1
- qkv‖z (5120×16384): S = 1
- gate‖up (5120×34816): S = 4
- down (17408×5120): S = 16
