# `probe_gemv --loads` raw matrix - 2026-08-26

The complete output of `tools/box.sh run "./build/tools/probe/probe_gemv --loads"`,
verbatim. Spec 1.7 §3's P1 battery: the production int4 GEMV's arithmetic with a
different number of load messages under it, at the six production shapes. The
interpretation lives in docs/12-kernels.md and docs/15-step-anatomy.md; this file
is the record those are taken from.

**Conditions.** Box `CL_DRIVER_VERSION 26.27.39122.14`, kernels AOT-compiled by
`ocloc -device bmg-g31`, host `-O2 -Wall -Wextra -Werror`, C++17. Idle: no
containers, 99.8% CPU idle, and the only DRM fd holders were `gnome-shell` and a
terminal on `card0`/`renderD128` (nouveau) - **zero** holders on the two `xe`
render nodes. **M = 1 only.** Weights are random (the B70 compresses uniform
fills - doc 01). GB/s counts weight bytes (int4 nibbles + f16 scales).

**Every row is held BIT-IDENTICAL to `base`**, which is `src/kernels/gemv.cl`
verbatim: none of these variants moves the accumulation order, so a differing bit
is a bug in the variant rather than a tolerance question, and the probe exits
non-zero on one. The exception is the two rows per shape at layout 0's own best
`S`, where a different split-K width reorders the sum; those are held to the
double-accumulating CPU reference's tolerance instead, and the column says so.

**The warm-up is not a formality and it is new here.** The 8-replay/drop-3
median inside `time_list` covers the clock ramp WITHIN a configuration. It does
not cover the device ramping down while the host computes a shape's CPU
reference. Measured this day, without the discarded warm-up this battery now
runs: the first recorded configuration of a shape read **355 / 475 / 524 GB/s**
where the identical binary read **534 / 541 / 538** once warm - **+50.4% /
+13.8% / +2.7%**, and it would have been charged to whichever variant happened
to be first. The `ramp control` table at the bottom is that gap, kept as
evidence. (The 51-row matrix in probe-gemv-2026-08-24.md was re-run warm the
same hour and reproduces to a **median 0.21% per cell**, worst cell +4.6%, with
an identical decision block - so that record is sound and is not affected.)

---

# probe_gemv --loads - the P1 load-path battery (spec 1.7 §3)

Same arithmetic, same bytes, different message counts. `msgs` is the LSC
messages one subgroup issues per k-group (64 K elements, 544 weight bytes):
weights + scales + activations. Every row is held byte-identical to `base`,
which is src/kernels/gemv.cl verbatim. Timing is the house 8-replay/drop-3
median over 40 launches cycling NB weight copies past the 24 MB L2.


## out/o_proj - 6144×5120, 15.94 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 16 | 10 | 31.4 | 532 | 90% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 16 | 6 | 31.3 | 535 | 91% | +0.5% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 16 | 10 | 30.8 | 543 | 92% | +2.1% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 16 | 10+1pf | 31.3 | 533 | 90% | +0.2% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 16 | 10+1pf | 31.5 | 530 | 90% | -0.4% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 16 | 10+1pf | 31.9 | 524 | 89% | -1.5% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 16 | 10 | 31.8 | 526 | 89% | -1.2% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 16 | 10 | 31.8 | 526 | 89% | -1.2% | identical | pfbuf's register-pressure control |
| l0base | 0 | 16 | 17 | 31.2 | 536 | 91% | +0.7% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 16 | 10 | 31.3 | 534 | 90% | +0.3% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 16 | 6 | 31.2 | 535 | 91% | +0.6% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 16 | 9.5 | 33.2 | 503 | 85% | -5.5% | identical | + 2D block read 16r16 (two k-groups) |
| l0base | 0 | 4 | 17 | 30.3 | 552 | 94% | +3.7% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 4 | 10 | 30.1 | 555 | 94% | +4.3% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |

## q‖k‖v - 5120×14336, 37.19 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 72.0 | 541 | 92% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 72.1 | 541 | 92% | -0.1% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 69.9 | 558 | 95% | +3.1% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 72.3 | 539 | 91% | -0.4% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 72.9 | 535 | 91% | -1.2% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 73.6 | 530 | 90% | -2.2% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 70.4 | 554 | 94% | +2.2% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 73.5 | 531 | 90% | -2.0% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 78.2 | 499 | 84% | -7.9% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 77.5 | 503 | 85% | -7.0% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 77.5 | 503 | 85% | -7.1% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 88.2 | 442 | 75% | -18.4% | identical | + 2D block read 16r16 (two k-groups) |
| l0base | 0 | 2 | 17 | 68.0 | 574 | 97% | +6.0% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 2 | 10 | 68.0 | 574 | 97% | +6.0% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |

## qkv‖z - 5120×16384, 42.50 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 79.7 | 559 | 95% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 80.3 | 555 | 94% | -0.7% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 79.4 | 561 | 95% | +0.4% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 80.0 | 557 | 94% | -0.3% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 81.2 | 549 | 93% | -1.8% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 82.3 | 542 | 92% | -3.1% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 80.5 | 553 | 94% | -1.0% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 80.0 | 557 | 94% | -0.4% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 173.7 | 257 | 43% | -54.1% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 88.0 | 507 | 86% | -9.4% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 88.4 | 504 | 85% | -9.9% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 103.3 | 431 | 73% | -22.9% | identical | + 2D block read 16r16 (two k-groups) |
| l0base | 0 | 16 | 17 | 106.9 | 417 | 71% | -25.4% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 16 | 10 | 82.2 | 542 | 92% | -3.0% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |

## gate‖up - 5120×34816, 90.31 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 4 | 10 | 176.5 | 537 | 91% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 4 | 6 | 176.7 | 536 | 91% | -0.2% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 4 | 10 | 173.9 | 545 | 92% | +1.5% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 4 | 10+1pf | 177.0 | 535 | 91% | -0.3% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 4 | 10+1pf | 176.3 | 537 | 91% | +0.1% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 4 | 10+1pf | 178.0 | 532 | 90% | -0.9% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 4 | 10 | 178.6 | 530 | 90% | -1.2% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 4 | 10 | 177.0 | 535 | 91% | -0.3% | identical | pfbuf's register-pressure control |
| l0base | 0 | 4 | 17 | 174.0 | 544 | 92% | +1.4% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 4 | 10 | 174.0 | 544 | 92% | +1.4% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 4 | 6 | 174.3 | 543 | 92% | +1.2% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 4 | 9.5 | 177.2 | 535 | 91% | -0.4% | identical | + 2D block read 16r16 (two k-groups) |
| l0base | 0 | 8 | 17 | 170.5 | 555 | 94% | +3.5% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 8 | 10 | 170.5 | 555 | 94% | +3.5% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |

## down - 17408×5120, 45.16 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 16 | 10 | 88.9 | 533 | 90% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 16 | 6 | 89.4 | 530 | 90% | -0.5% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 16 | 10 | 87.8 | 539 | 91% | +1.2% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 16 | 10+1pf | 89.3 | 530 | 90% | -0.4% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 16 | 10+1pf | 90.5 | 523 | 89% | -1.7% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 16 | 10+1pf | 91.5 | 517 | 88% | -2.9% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 16 | 10 | 91.3 | 519 | 88% | -2.6% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 16 | 10 | 90.7 | 522 | 89% | -2.0% | identical | pfbuf's register-pressure control |
| l0base | 0 | 16 | 17 | 87.8 | 539 | 91% | +1.2% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 16 | 10 | 87.8 | 539 | 91% | +1.3% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 16 | 6 | 87.8 | 539 | 91% | +1.2% | identical | + 2D block read and wide activations |
| l0base | 0 | 4 | 17 | 83.3 | 568 | 96% | +6.7% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 4 | 10 | 83.5 | 567 | 96% | +6.4% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |

## lm_head int4 - 5120×248320, 644.14 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 1183.3 | 571 | 97% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 1182.4 | 571 | 97% | +0.1% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 1173.4 | 576 | 98% | +0.8% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 1187.0 | 569 | 96% | -0.3% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 1193.5 | 566 | 96% | -0.9% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 1208.6 | 559 | 95% | -2.1% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 1184.7 | 570 | 97% | -0.1% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 1192.2 | 567 | 96% | -0.7% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 1199.1 | 563 | 95% | -1.3% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 1178.5 | 573 | 97% | +0.4% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 1179.2 | 573 | 97% | +0.3% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 1237.4 | 546 | 93% | -4.4% | identical | + 2D block read 16r16 (two k-groups) |

## ramp control - the discarded warm-up beside the recorded `base`

| shape | warm-up (discarded) GB/s | recorded `base` GB/s | ramp |
|---|---|---|---|
| out/o_proj | 378 | 532 | +40.6% |
| q‖k‖v | 458 | 541 | +18.2% |
| qkv‖z | 559 | 559 | +0.0% |
| gate‖up | 536 | 537 | +0.0% |
| down | 532 | 533 | +0.1% |
| lm_head int4 | 571 | 571 | -0.1% |

## drift control - `base` re-measured after the whole battery

| shape | GB/s at battery start | GB/s at battery end | Δ% |
|---|---|---|---|
| out/o_proj | 532 | 535 | +0.54% |
| q‖k‖v | 541 | 541 | -0.09% |
| qkv‖z | 559 | 559 | -0.08% |
| gate‖up | 537 | 536 | -0.21% |
| down | 533 | 532 | -0.18% |
| lm_head int4 | 571 | 571 | +0.05% |

worst |Δ| over the six controls: **0.54%** (the standing per-cell repeatability of this instrument is a median 0.21%)

byte-identity and CPU-reference check: every row passed
