# `probe_gemv --loads` raw matrix - 2026-08-26

The complete output of `tools/box.sh run "./build/tools/probe/probe_gemv --loads"`,
verbatim, below the horizontal rule. Spec 1.7 §3's P1 battery: the production
int4 GEMV's arithmetic with a different number of load messages under it, at the
six production shapes. Everything above the rule is the header a reader needs in
order not to misread the table.

**Conditions.** Box `CL_DRIVER_VERSION 26.27.39122.14`, kernels AOT-compiled by
`ocloc -device bmg-g31`, host `-O2 -Wall -Wextra -Werror`, C++17. Idle: no
containers, and the only DRM fd holders were `gnome-shell` and a terminal on
`card0`/`renderD128` (nouveau) - **zero** holders on the two `xe` render nodes.
**M = 1 only.** Weights are random (the B70 compresses uniform fills - doc 01).
GB/s counts weight bytes (int4 nibbles + f16 scales) only.

## The plan's names and this file's names

Spec 1.7 §3 names the variants `{baseline, vec128, blk2d, prefetch}`. They are
built here as:

| plan name | tag(s) here | what it actually does |
|---|---|---|
| `baseline` | `base` | `src/kernels/gemv.cl` verbatim, layout 1, production `S` |
| `vec128` | `xwide` | activation loads 8 × 16 B → 4 × 32 B (10 messages → 6) |
| `blk2d` | `l0b2d`, `l0b2dx`, `l0b2d16` | `intel_sub_group_2d_block_read_32b_*` on layout 0 (17 → 10, 6, 9.5) |
| `prefetch` | `pf1`, `pf2`, `pf4`, `pfbuf`, `ballast` | OpenCL `prefetch()` at three distances; the register form and its control |
| - (added) | `deq` | xor + arithmetic-shift dequant: ALU only, no message change |
| - (added) | `l0deq`, `l0b2ddeq` | the retuned layout+`S` cell **with** `deq`, i.e. the composed cell |

`vec128` is named for a 128-bit load. Note that the baseline activation load is
**already** 128-bit (`vload8` of `ushort` = 16 B); `xwide` widens it to 256-bit.
The plan's arithmetic assumed a narrower baseline than the source has.

## How each row is checked, and the distinction matters

Rows at **the production `S`** keep the accumulation order exactly and are held
**bit-identical to `base`** - a differing bit is a bug in the variant, not a
tolerance question, and the probe exits non-zero on one.

Rows at **any other `S`** cannot be: a different split-K width folds a different
number of fp32 partials, which reorders the sum. Those twelve cells are held to
the **double-accumulating CPU reference's tolerance** instead, and the `bytes vs
base` column says `ref ok (S reorders the sum)` for exactly those. **Do not
quote "every row is bit-identical" for this table** - twelve of them are not,
and a lever built on them is a golden-gate lever, not a bit-identity one.

## Caveats to carry with any cell you quote

- **Every non-`base` cell is n = 1.** One 8-replay/drop-3 median of 40 launches,
  one invocation. The controls below bound the *instrument*, not any individual
  cell, and this instrument's own recorded history contains a cell that moved
  **11%** between runs (`down` layout 1 `S` = 4, 482 vs 429). Two cells here
  have independent corroboration from a second session and are marked where they
  appear; the rest do not.
- **Do not build on `pfbuf` at `q‖k‖v`.** It is +2.2% where the other five
  shapes are ≤ +0.0%, and it beats its own register control by 4.2 pp where the
  other five are ≤ +0.6 pp. One cell out of six is not a result. If any lever
  wants a software-pipelined GEMV, re-run that cell first.
- **Do not build on `down` at layout 0 `S` = 4 without a second run.** It is the
  best cell for that shape, and the one cell in this instrument's history that
  failed to reproduce was `down` at `S` = 4 in the *other* layout. Two sessions
  now agree (567 / 568), which is better than most cells here have, but this one
  carries a lever.
- **`out/o_proj`'s `base` reads about 0.5% low.** It is the first shape measured
  and its drift control is the largest of the six (+0.36% here, +0.54% in the
  first battery). Every Δ% in that block is inflated by roughly that much.
- **`xwide`'s message count is modelled, not verified.** The eight activation
  loads are at subgroup-uniform addresses and IGC's load vectoriser may already
  merge some of them, in which case `xwide` moved fewer messages than the model
  claims. No ISA dump was taken. The row is a *bound* on what widening those
  loads can buy, which is what it is used for.

## The ramp, and what actually causes it

`probe_gemv --loads` discards one warm-up configuration per shape and prints the
gap it hid as a `ramp control` table. This is a real instrument fix and it stays
- but the mechanism is **cold start, not host-side idle**, and the distinction
matters because this is heading for docs/16 as house method.

The evidence is the ramp table itself: the effect is **+42.8%** at the first
shape measured, **+15.4%** at the second, and **−0.1% / −0.1% / +0.4% / +0.1%**
at the other four. If the cause were "the device ramps down while the host
computes a CPU reference", `lm_head` would show the largest effect - its
reference is a 1.27 × 10⁹-iteration loop, by far the longest host gap in the
run - and it shows **none**. What actually happened: the box had rebooted about
half an hour earlier and the GPU had never been under sustained load, so the
first ~20 s of GPU work is the device climbing to its sustained clock. Once
there, it stays there across the host gaps.

**The house rule that follows:** the 8-replay/drop-3 median covers the ramp
*inside* a configuration. It does not cover the *first* configurations a process
runs on a device that has been cold. Discard a warm-up configuration, or run the
battery twice and keep the second pass. `probe_attn` and `probe_bw` do not do
this today. Measured cost of not doing it: the first recorded configuration read
**355-378 GB/s where the identical binary read 532 warm**.

The 51-row matrix in [probe-gemv-2026-08-24.md](probe-gemv-2026-08-24.md) is
**not** affected - it was re-run warm the same hour
([probe-gemv-2026-08-26-warm.md](probe-gemv-2026-08-26-warm.md)) and reproduces
to a median 0.21% per cell with an identical decision block, including the
layout-0 `S` = 1 cells that are first in that probe's own order.

## Pre-registration

These predictions were written before any variant was compiled or measured. They
are committed **after** the fact - this file did not exist when they were
written; they lived in the task's workspace, which is gitignored - so **this
round's pre-registration is dated by `0c73e88` (the probe source and its
message-count model, committed before the battery ran) and by nothing tighter.**
Future pre-registrations go in a committed file before the run. Recording the
weakness rather than implying a strength this record does not have.

**H0 (primary): the 533-559 plateau is not message-issue.** Every
message-removing variant lands within noise of its own layout's control.
Predicted flat (±1.5%) for `xwide`; ±3% for `l0b2d` against `l0base`; ≤ +1% for
the prefetch family; ≤ ±2% for the cache-control family; `deq` flat ±1%.

**H1 (the alternative): `xwide` → +8-10% on the 533-class shapes; `l0b2d` →
layout 1's number at every shape.**

| bar | threshold | measured | outcome |
|---|---|---|---|
| 1. `xwide` moves a shape > +4% | +4% | max **+0.5%** | H0 holds |
| 2. `l0b2d` reaches ≥ 520 at `qkv‖z` `S` = 1 | 520 | **507** | not tripped - **but see below** |
| 3. `l0b2d` moves a non-`qkv‖z` shape > +5% | +5% | max **+1.8%** | H0 holds |
| 4. `deq` moves a shape > 2% | 2% | **+3.1%**, +2.1% | **TRIPPED - the prediction was wrong** |
| 5. `pfbuf` − `ballast` > 2 pp | 2 pp | **+4.2 pp** at one shape of six | tripped at one cell |

**Bar 2 was set above its own hypothesis's ceiling, and that is a flaw in the
pre-registration, not a pass.** The point prediction for `l0b2d` at `qkv‖z`
`S` = 1 was **300-430 GB/s**. The measurement is **507** - **missed high by
18%** - while the bar that was supposed to falsify H0 sat at 520, i.e. *above*
the predicted ceiling. A bar placed beyond one's own stated range cannot fail,
and this one did not. **Score it as: the point prediction was wrong, and the bar
was badly set.** The reason the verdict survives it is separate and is argued
from bars 1 and 3, which were tight: at the four shapes with no stride
pathology the same instruction is worth under 2%.

## What the battery found - the composed lever

`deq` and the layout+`S` retune are **not orthogonal**: `deq`'s gain shrinks as a
shape approaches the bandwidth wall, and the retuned cells sit at 94-97% of it.
So the composed cell was measured directly (`l0deq`, `l0b2ddeq`) rather than
multiplied out. `deq`'s marginal contribution **on top of** the retune is
**+0.74% mean** against **+1.20% mean** at the production configuration - the
shrinkage is real and it is about a third.

| shape | `base` | best **without** `deq` | best **with** `deq` (composed) | `deq` marginal |
|---|---|---|---|---|
| out/o_proj | 532 | 554 (`l0b2d`, L0 S4) | **561** (`l0b2ddeq`) | +1.3% |
| q‖k‖v | 540 | 577 (`l0base`, L0 S2) | **579** (`l0b2ddeq`) | +0.3% |
| qkv‖z | 562 | 562 (`base`) | **562** - nothing beat the control | - |
| gate‖up | 536 | 555 (`l0base`, L0 S8) | **561** (`l0deq`) | +1.1% |
| down | 533 | 567 (`l0b2d`, L0 S4) | **574** (`l0b2ddeq`) | +1.2% |
| lm_head int4 | 571 | 573 (`l0b2dx`) | **576** (`deq`, L1 S1) | +0.5% |

**Estimated value** - probe-side µs ratio applied to that shape's in-situ
µs/launch from the same day's `--profile --repeats 5`. Estimated, not measured
in situ:

| | ms/token |
|---|---|
| retune only (no `deq`) | 0.909 |
| `deq`'s marginal contribution at the retuned cells | +0.216 |
| **composed, gross** | **1.125** |
| `prep` partials cost of the `S` changes (below) | +0.009 |
| **composed, net - the figure to rank with** | **1.117** |

Sub-levers, for anyone who wants a cheaper rider than the full retune. They are
**not additive with the composed figure - it contains them**:

| sub-lever | ms/token (est.) | rider |
|---|---|---|
| `deq` at the production configuration, all six shapes | **0.286** | none - bit-identical |
| `gate‖up` `S` 4 → 8 inside layout 1, with `deq` | **0.240** net of `prep` | `S`-table one-liner + golden gate |
| `down` layout 0 at production `S` = 16, with `deq` | **0.222** | layout rider only; no `S` change |

## `S` changes are not free: the `prep` side

The engine's `S` policy is *smallest within 3% of best*, not argmax, and the
reason is that a larger `S` makes `gemv` write more fp32 partials for the next
kernel to fold. The GEMV's own cost of writing them is already inside the µs
this probe measures; the **consumer's read is not**. That read is
`launches × ΔS × N × 4 B`, priced at the measured 590 GB/s:

| shape | `S` now → retuned | launches/token | Δ partial bytes/token | Δ ms/token |
|---|---|---|---|---|
| out/o_proj | 16 → 4 | 64 | −15.73 MB | **−0.027** |
| q‖k‖v | 1 → 2 | 16 | +0.92 MB | +0.002 |
| qkv‖z | 1 → 1 | 48 | 0 | 0 |
| gate‖up | 4 → 8 | 64 | +35.65 MB | **+0.060** |
| down | 16 → 4 | 64 | −15.73 MB | **−0.027** |
| lm_head | 1 → 1 | 1 | 0 | 0 |
| **net** | | | **+4.9 MB** | **+0.009** |

Two shapes want a *smaller* `S` and pay `prep` back; `gate‖up` alone costs
+0.060 ms against its own +0.25 ms gain. **The four changes very nearly cancel
at the `prep` boundary**, which is why the composed net and gross differ by
0.009 ms - but a lever that moved only `gate‖up` would have to carry the full
+0.060.

## What is an `S` one-liner and what needs the layout rider

Layout and `S` were varied together in the first battery. Layout 1 at the
alternate `S` was measured in this run to separate them:

| shape | production | **layout 1 at the alternate `S`** | composed (layout 0) | verdict |
|---|---|---|---|---|
| out/o_proj | 532 (L1 S16) | 467 (L1 S4) - **−12.3%** | 561 | all of it needs layout 0 |
| q‖k‖v | 540 (L1 S1) | 515 (L1 S2) - **−4.6%** | 579 | all of it needs layout 0 |
| qkv‖z | 562 (L1 S1) | 541 (L1 S16) - −3.8% | 562 | nothing to take |
| gate‖up | 536 (L1 S4) | **549 (L1 S8) - +2.4%** | 561 | **+2.4 pp is an `S`-table one-liner**; +2.1 pp more needs layout 0 |
| down | 533 (L1 S16) | 473 (L1 S4) - −11.4% | 574 | all of it needs layout 0 |

**Only `gate‖up` has a meaningful `S`-only component.** At the other three the
layout-1 row at the alternate `S` is *worse* than production - which is exactly
why the standing `S` policy picked what it picked, and it means those gains
cannot be had without the loader/`capture`/table rider that layout 0 implies.

There is a cheaper sub-lever the first battery missed, and it is layout-only:
**`down` at layout 0 and its PRODUCTION `S` = 16, with `deq`, reads 555 (+4.0%)**
- no `S` change, so no reordered sum and no `prep` delta, only the layout rider.
`out/o_proj` has the same shape of option at +1.9%.

## How strong the null result is

Three independent knobs act on message count - `xwide` (10 → 6 on layout 1),
`l0b2d` (17 → 10 on layout 0), `l0b2dx` (17 → 6) - across six shapes, and all
three are flat. That corroboration is why the verdict leans as hard as it does.

**It is still one experiment, and one experiment cannot fully separate "message
count is not the currency" from "message count is the currency and each variant
happens to add an offsetting overhead of almost exactly the right size".** The
second reading requires three different instructions on two layouts to
coincidentally cancel at six shapes, which is why it is not the one carried
forward - but it is not excluded, and `l0b2d16` (fewer messages, materially
*slower*) is a live reminder that these instructions do carry costs of their own.

**One correction to the verbatim block below.** Its third paragraph - printed by
the program - says "Every row is held byte-identical to `base`". That was the
program's own header at the time of this run and it is **wrong for the twelve
rows at a non-production `S`**, exactly as the section above explains. The
printf has been corrected in `tools/probe/probe_gemv.cc`; re-running the battery
purely to reprint one paragraph would have cost a device run for no measurement,
so the output is left verbatim and corrected here. **The `bytes vs base` column
in the table is, and always was, right** - it says `ref ok (S reorders the sum)`
on precisely those twelve rows.

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
| xwide | 1 | 16 | 6 | 31.4 | 533 | 90% | +0.1% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 16 | 10 | 30.9 | 540 | 92% | +1.5% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 16 | 10+1pf | 31.4 | 532 | 90% | -0.0% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 16 | 10+1pf | 31.6 | 530 | 90% | -0.5% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 16 | 10+1pf | 31.9 | 523 | 89% | -1.7% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 16 | 10 | 31.9 | 524 | 89% | -1.5% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 16 | 10 | 31.8 | 525 | 89% | -1.3% | identical | pfbuf's register-pressure control |
| l0base | 0 | 16 | 17 | 31.4 | 532 | 90% | -0.1% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 16 | 10 | 31.5 | 531 | 90% | -0.3% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 16 | 6 | 31.5 | 531 | 90% | -0.3% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 16 | 9.5 | 33.5 | 499 | 85% | -6.2% | identical | + 2D block read 16r16 (two k-groups) |
| l0deq | 0 | 16 | 17 | 30.8 | 543 | 92% | +1.9% | identical | GPTQ-native + xor/shift dequant |
| base | 1 | 4 | 10 | 35.8 | 467 | 79% | -12.3% | ref ok (S reorders the sum) | layout 1 at layout 0's best S - the S-ONLY option |
| deq | 1 | 4 | 10 | 37.9 | 441 | 75% | -17.1% | ref ok (S reorders the sum) | layout 1 at that S, + xor/shift dequant |
| l0base | 0 | 4 | 17 | 30.3 | 551 | 93% | +3.5% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 4 | 10 | 30.1 | 554 | 94% | +4.1% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |
| l0deq | 0 | 4 | 17 | 29.9 | 558 | 95% | +4.9% | ref ok (S reorders the sum) | + xor/shift dequant - THE COMPOSED CELL |
| l0b2ddeq | 0 | 4 | 10 | 29.8 | 561 | 95% | +5.4% | ref ok (S reorders the sum) | + 2D block read and xor/shift dequant |

## q‖k‖v - 5120×14336, 37.19 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 72.2 | 540 | 92% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 72.2 | 540 | 92% | +0.0% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 70.1 | 556 | 94% | +3.0% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 72.5 | 538 | 91% | -0.4% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 72.9 | 535 | 91% | -1.0% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 73.4 | 531 | 90% | -1.7% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 70.4 | 554 | 94% | +2.5% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 73.6 | 530 | 90% | -1.9% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 78.3 | 498 | 84% | -7.8% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 77.5 | 503 | 85% | -6.8% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 77.5 | 503 | 85% | -6.9% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 88.6 | 440 | 75% | -18.5% | identical | + 2D block read 16r16 (two k-groups) |
| l0deq | 0 | 1 | 17 | 78.2 | 499 | 85% | -7.7% | identical | GPTQ-native + xor/shift dequant |
| base | 1 | 2 | 10 | 75.7 | 515 | 87% | -4.5% | ref ok (S reorders the sum) | layout 1 at layout 0's best S - the S-ONLY option |
| deq | 1 | 2 | 10 | 76.6 | 509 | 86% | -5.7% | ref ok (S reorders the sum) | layout 1 at that S, + xor/shift dequant |
| l0base | 0 | 2 | 17 | 67.6 | 577 | 98% | +6.9% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 2 | 10 | 68.0 | 574 | 97% | +6.3% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |
| l0deq | 0 | 2 | 17 | 67.8 | 575 | 98% | +6.6% | ref ok (S reorders the sum) | + xor/shift dequant - THE COMPOSED CELL |
| l0b2ddeq | 0 | 2 | 10 | 67.4 | 579 | 98% | +7.1% | ref ok (S reorders the sum) | + 2D block read and xor/shift dequant |

## qkv‖z - 5120×16384, 42.50 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 79.3 | 562 | 95% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 79.3 | 562 | 95% | +0.0% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 79.4 | 561 | 95% | -0.1% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 79.5 | 561 | 95% | -0.2% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 81.2 | 549 | 93% | -2.3% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 82.5 | 540 | 92% | -3.8% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 82.8 | 538 | 91% | -4.2% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 79.6 | 560 | 95% | -0.3% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 173.8 | 256 | 43% | -54.4% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 88.7 | 502 | 85% | -10.6% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 88.7 | 502 | 85% | -10.6% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 102.6 | 434 | 74% | -22.7% | identical | + 2D block read 16r16 (two k-groups) |
| l0deq | 0 | 1 | 17 | 87.1 | 511 | 87% | -9.0% | identical | GPTQ-native + xor/shift dequant |
| base | 1 | 16 | 10 | 82.4 | 541 | 92% | -3.7% | ref ok (S reorders the sum) | layout 1 at layout 0's best S - the S-ONLY option |
| deq | 1 | 16 | 10 | 82.1 | 543 | 92% | -3.3% | ref ok (S reorders the sum) | layout 1 at that S, + xor/shift dequant |
| l0base | 0 | 16 | 17 | 105.0 | 425 | 72% | -24.4% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 16 | 10 | 81.9 | 544 | 92% | -3.2% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |
| l0deq | 0 | 16 | 17 | 82.2 | 542 | 92% | -3.5% | ref ok (S reorders the sum) | + xor/shift dequant - THE COMPOSED CELL |
| l0b2ddeq | 0 | 16 | 10 | 81.4 | 548 | 93% | -2.5% | ref ok (S reorders the sum) | + 2D block read and xor/shift dequant |

## gate‖up - 5120×34816, 90.31 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 4 | 10 | 176.6 | 536 | 91% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 4 | 6 | 176.6 | 536 | 91% | -0.0% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 4 | 10 | 174.0 | 544 | 92% | +1.5% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 4 | 10+1pf | 177.3 | 534 | 91% | -0.4% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 4 | 10+1pf | 176.3 | 537 | 91% | +0.2% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 4 | 10+1pf | 178.0 | 532 | 90% | -0.8% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 4 | 10 | 178.5 | 531 | 90% | -1.0% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 4 | 10 | 177.2 | 534 | 91% | -0.4% | identical | pfbuf's register-pressure control |
| l0base | 0 | 4 | 17 | 174.5 | 543 | 92% | +1.2% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 4 | 10 | 175.0 | 541 | 92% | +0.9% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 4 | 6 | 174.5 | 543 | 92% | +1.2% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 4 | 9.5 | 178.1 | 532 | 90% | -0.8% | identical | + 2D block read 16r16 (two k-groups) |
| l0deq | 0 | 4 | 17 | 173.1 | 547 | 93% | +2.0% | identical | GPTQ-native + xor/shift dequant |
| base | 1 | 8 | 10 | 172.5 | 549 | 93% | +2.4% | ref ok (S reorders the sum) | layout 1 at layout 0's best S - the S-ONLY option |
| deq | 1 | 8 | 10 | 172.1 | 550 | 93% | +2.6% | ref ok (S reorders the sum) | layout 1 at that S, + xor/shift dequant |
| l0base | 0 | 8 | 17 | 170.7 | 555 | 94% | +3.4% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 8 | 10 | 170.7 | 555 | 94% | +3.5% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |
| l0deq | 0 | 8 | 17 | 168.9 | 561 | 95% | +4.5% | ref ok (S reorders the sum) | + xor/shift dequant - THE COMPOSED CELL |
| l0b2ddeq | 0 | 8 | 10 | 169.1 | 560 | 95% | +4.4% | ref ok (S reorders the sum) | + 2D block read and xor/shift dequant |

## down - 17408×5120, 45.16 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 16 | 10 | 88.8 | 533 | 90% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 16 | 6 | 88.9 | 532 | 90% | -0.2% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 16 | 10 | 88.2 | 537 | 91% | +0.6% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 16 | 10+1pf | 89.3 | 530 | 90% | -0.5% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 16 | 10+1pf | 90.3 | 525 | 89% | -1.6% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 16 | 10+1pf | 91.0 | 520 | 88% | -2.4% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 16 | 10 | 90.7 | 522 | 89% | -2.0% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 16 | 10 | 90.0 | 526 | 89% | -1.3% | identical | pfbuf's register-pressure control |
| l0base | 0 | 16 | 17 | 87.9 | 539 | 91% | +1.1% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 16 | 10 | 87.9 | 539 | 91% | +1.0% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 16 | 6 | 87.8 | 539 | 91% | +1.1% | identical | + 2D block read and wide activations |
| l0deq | 0 | 16 | 17 | 85.4 | 555 | 94% | +4.0% | identical | GPTQ-native + xor/shift dequant |
| base | 1 | 4 | 10 | 100.2 | 473 | 80% | -11.3% | ref ok (S reorders the sum) | layout 1 at layout 0's best S - the S-ONLY option |
| deq | 1 | 4 | 10 | 102.4 | 462 | 78% | -13.3% | ref ok (S reorders the sum) | layout 1 at that S, + xor/shift dequant |
| l0base | 0 | 4 | 17 | 83.6 | 567 | 96% | +6.3% | ref ok (S reorders the sum) | GPTQ-native at layout 0's own best S |
| l0b2d | 0 | 4 | 10 | 83.5 | 567 | 96% | +6.4% | ref ok (S reorders the sum) | + 2D block read 8r16, same S |
| l0deq | 0 | 4 | 17 | 83.1 | 570 | 97% | +6.9% | ref ok (S reorders the sum) | + xor/shift dequant - THE COMPOSED CELL |
| l0b2ddeq | 0 | 4 | 10 | 82.5 | 574 | 97% | +7.6% | ref ok (S reorders the sum) | + 2D block read and xor/shift dequant |

## lm_head int4 - 5120×248320, 644.14 MB of weights

| variant | L | S | msgs | µs | GB/s | % of 590 | Δ% vs base | bytes vs base | what |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1 | 10 | 1181.9 | 571 | 97% | +0.0% | (the control) | gemv.cl verbatim - the control |
| xwide | 1 | 1 | 6 | 1182.3 | 571 | 97% | -0.0% | identical | activations 8x16 B -> 4x32 B |
| deq | 1 | 1 | 10 | 1173.3 | 576 | 98% | +0.7% | identical | xor+shift dequant (ALU only) |
| pf1 | 1 | 1 | 10+1pf | 1186.4 | 569 | 96% | -0.4% | identical | prefetch() 1 k-group ahead |
| pf2 | 1 | 1 | 10+1pf | 1192.6 | 566 | 96% | -0.9% | identical | prefetch() 2 ahead (bestla's distance) |
| pf4 | 1 | 1 | 10+1pf | 1207.4 | 559 | 95% | -2.1% | identical | prefetch() 4 ahead |
| pfbuf | 1 | 1 | 10 | 1185.4 | 570 | 97% | -0.3% | identical | register double-buffer (+8 u32/lane) |
| ballast | 1 | 1 | 10 | 1191.8 | 567 | 96% | -0.8% | identical | pfbuf's register-pressure control |
| l0base | 0 | 1 | 17 | 1198.5 | 564 | 96% | -1.4% | identical | GPTQ-native, 8 strided 64 B loads |
| l0b2d | 0 | 1 | 10 | 1180.0 | 572 | 97% | +0.2% | identical | + 2D block read 8r16 (one message) |
| l0b2dx | 0 | 1 | 6 | 1179.6 | 573 | 97% | +0.2% | identical | + 2D block read and wide activations |
| l0b2d16 | 0 | 1 | 9.5 | 1239.4 | 545 | 92% | -4.6% | identical | + 2D block read 16r16 (two k-groups) |
| l0deq | 0 | 1 | 17 | 1216.4 | 555 | 94% | -2.8% | identical | GPTQ-native + xor/shift dequant |

## ramp control - the discarded warm-up beside the recorded `base`

| shape | warm-up (discarded) GB/s | recorded `base` GB/s | ramp |
|---|---|---|---|
| out/o_proj | 373 | 532 | +42.8% |
| q‖k‖v | 468 | 540 | +15.4% |
| qkv‖z | 562 | 562 | -0.1% |
| gate‖up | 537 | 536 | -0.1% |
| down | 531 | 533 | +0.4% |
| lm_head int4 | 571 | 571 | +0.1% |

## drift control - `base` re-measured after the whole battery

| shape | GB/s at battery start | GB/s at battery end | Δ% |
|---|---|---|---|
| out/o_proj | 532 | 534 | +0.36% |
| q‖k‖v | 540 | 540 | +0.03% |
| qkv‖z | 562 | 561 | -0.04% |
| gate‖up | 536 | 536 | -0.09% |
| down | 533 | 533 | +0.04% |
| lm_head int4 | 571 | 571 | -0.11% |

worst |Δ| over the six controls: **0.36%** (the standing per-cell repeatability of this instrument is a median 0.21%)

byte-identity and CPU-reference check: every row passed
