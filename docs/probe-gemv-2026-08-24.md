# `probe_gemv` raw matrix - 2026-08-24

The complete output of `tools/probe/probe_gemv`, run 1, verbatim. This is the
record the layout and `S` decisions in [12-kernels.md](12-kernels.md),
[05-perf-model.md](05-perf-model.md), [02-formats.md](02-formats.md),
[08-decode-vs-prefill.md](08-decode-vs-prefill.md) and spec §6.3 were taken
from; every number quoted in those documents is a cell below or arithmetic on
one. It supersedes [probe-gemv-2026-08-23.md](probe-gemv-2026-08-23.md).

**Why this rerun exists.** The 2026-08-23 run measured `out/o_proj` at
**(5120, 5120)**. That shape does not exist in the model: doc 03 gives both
output projections - GDN `out_proj` and attention `o_proj` - as
**6144 → 5120** (24 heads × 256). Spec §4.2 listed (5120, 5120) in error, and
the error propagated into the kernel variant matrix, the probe, the GEMV test
and every doc that cited an `out/o_proj` number. The shape is corrected to
(6144, 5120) - `K % 64 == 0`, `G = 96`, every `S ∈ {1,2,4,8,16}` divides 96 -
and the **whole matrix** re-measured so that every cited cell traces to one
file rather than two. The other four int4 shapes and `lm_head` are unchanged;
they are re-run here only so that no document mixes two measurement sessions.

```bash
tools/box.sh run "./build/tools/probe/probe_gemv" | tee docs/probe-gemv-2026-08-24.md
```

**Conditions.** Box `CL_DRIVER_VERSION 26.27.39122.14` (unchanged from
2026-08-23), kernels AOT-compiled by `ocloc -device bmg-g31`, host
`-O2 -Wall -Wextra -Werror`. **M = 1 only.** Weights are random (the B70
compresses uniform fills - doc 01). GB/s counts weight bytes (int4 nibbles +
f16 scales) against `us_per_launch`; the denominator `600 GB/s` is doc 01's
roofline figure, and `probe_bw` measures 590 GB/s through the same launch path.
Each configuration is 40 launches in one regular command list replayed 8 times,
median of the last 5, cycling `NB = max(2, 72 MB / weight_bytes + 1)` weight
copies so the 24 MB L2 is missed. **No `WRONG` row: all 50 int4 configurations
matched the CPU reference**, and the probe now exits non-zero if any row misses
tolerance (it exited 0).

**Stability.** A second full run the same day produced an **identical decision
block** - layout 1, `S` = 16 / 1 / 1 / 4 / 16 - and agreed with run 1 to a
median of 0.21% per cell. The worst disagreement is **11.0% at one
non-deciding cell**, `down` layout 1 `S` = 4: 482 GB/s in run 1 against 429 in
run 2 (the 2026-08-23 run read 431 there, so run 1's 482 is the outlier). That
cell decides nothing - `down` picks `S` = 16 in both runs - but it is a wider
spread than the 2.1% worst case the 2026-08-23 pair showed, and it is recorded
rather than smoothed. **Every deciding cell agreed within 0.4%** (run 1 / run 2:
`out/o_proj` L1 S=16 533 / 533, `q‖k‖v` L1 S=1 540 / 540, `qkv‖z` L1 S=1
559 / 561, `gate‖up` L1 S=4 536 / 536, `down` L1 S=16 531 / 530, `lm_head`
584 / 585), and run 2's layout sums were 2667 / 2715 against run 1's
2667 / 2713. Run 1 is the recorded one.

---

| shape | K×N | L | S | µs | GB/s | % of 600 | max abs err | tol |
|---|---|---|---|---|---|---|---|---|
| out/o_proj | 6144×5120 | 0 | 1 | 79.2 | 211 | 35% | 1.3e-05 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 2 | 39.7 | 421 | 70% | 7.6e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 4 | 30.4 | 549 | 92% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 8 | 34.4 | 486 | 81% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 0 | 16 | 31.3 | 533 | 89% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 1 | 64.5 | 259 | 43% | 1.3e-05 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 2 | 37.0 | 451 | 75% | 7.6e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 4 | 35.8 | 467 | 78% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 8 | 36.9 | 453 | 76% | 5.7e-06 | 0.0048 |
| out/o_proj | 6144×5120 | 1 | 16 | 31.3 | 533 | 89% | 5.7e-06 | 0.0048 |
| q‖k‖v | 5120×14336 | 0 | 1 | 79.3 | 492 | 82% | 1.3e-05 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 2 | 67.6 | 577 | 96% | 9.5e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 4 | 70.4 | 554 | 92% | 4.3e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 8 | 71.8 | 543 | 91% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 0 | 16 | 71.9 | 542 | 90% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 1 | 72.2 | 540 | 90% | 1.3e-05 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 2 | 79.3 | 492 | 82% | 9.5e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 4 | 72.6 | 537 | 90% | 4.3e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 8 | 72.1 | 541 | 90% | 5.7e-06 | 0.0045 |
| q‖k‖v | 5120×14336 | 1 | 16 | 72.8 | 536 | 89% | 5.7e-06 | 0.0045 |
| qkv‖z | 5120×16384 | 0 | 1 | 173.4 | 257 | 43% | 1.1e-05 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 2 | 113.4 | 393 | 66% | 9.5e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 4 | 108.9 | 409 | 68% | 5.7e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 8 | 107.9 | 413 | 69% | 4.3e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 0 | 16 | 106.2 | 419 | 70% | 7.6e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 1 | 79.7 | 559 | 93% | 1.1e-05 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 2 | 90.8 | 491 | 82% | 9.5e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 4 | 85.9 | 519 | 86% | 5.7e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 8 | 83.2 | 536 | 89% | 4.3e-06 | 0.0043 |
| qkv‖z | 5120×16384 | 1 | 16 | 82.4 | 541 | 90% | 7.6e-06 | 0.0043 |
| gate‖up | 5120×34816 | 0 | 1 | 214.1 | 442 | 74% | 1.3e-05 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 2 | 183.7 | 516 | 86% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 4 | 174.8 | 542 | 90% | 5.2e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 8 | 170.6 | 555 | 93% | 5.7e-06 | 0.0044 |
| gate‖up | 5120×34816 | 0 | 16 | 172.6 | 549 | 91% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 1 | 222.3 | 426 | 71% | 1.3e-05 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 2 | 192.4 | 492 | 82% | 7.6e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 4 | 176.7 | 536 | 89% | 5.2e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 8 | 172.6 | 549 | 91% | 5.7e-06 | 0.0044 |
| gate‖up | 5120×34816 | 1 | 16 | 175.6 | 539 | 90% | 7.6e-06 | 0.0044 |
| down | 17408×5120 | 0 | 1 | 221.2 | 214 | 36% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 0 | 2 | 114.0 | 415 | 69% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 0 | 4 | 83.6 | 567 | 94% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 0 | 8 | 104.5 | 453 | 75% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 0 | 16 | 88.0 | 538 | 90% | 1.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 1 | 181.3 | 261 | 44% | 4.2e-05 | 0.0067 |
| down | 17408×5120 | 1 | 2 | 100.7 | 470 | 78% | 2.3e-05 | 0.0067 |
| down | 17408×5120 | 1 | 4 | 98.3 | 482 | 80% | 1.4e-05 | 0.0067 |
| down | 17408×5120 | 1 | 8 | 105.8 | 447 | 75% | 9.5e-06 | 0.0067 |
| down | 17408×5120 | 1 | 16 | 89.1 | 531 | 89% | 1.3e-05 | 0.0067 |
| lm_head bf16 | 5120×248320 | B | 1 | 4350.5 | 584 | 97% | - | - |

Decision:
- layout 0: sum of best GB/s over shapes = 2667
- layout 1: sum of best GB/s over shapes = 2713
- canonical layout: 1
- out/o_proj (6144×5120): S = 16
- q‖k‖v (5120×14336): S = 1
- qkv‖z (5120×16384): S = 1
- gate‖up (5120×34816): S = 4
- down (17408×5120): S = 16
