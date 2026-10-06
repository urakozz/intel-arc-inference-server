# The step, launch by launch

What one decode token actually spends, measured **inside the replayed command
list**, per launch, at the benchmark shape. [12-kernels.md](12-kernels.md) says
what each kernel does; this document says where the microseconds go, and it is
the reason the kernel work that followed went where it did.

Decode today is **29.45 t/s** against vLLM's 31.01 on the same checkpoint files
([BENCHMARKS.md](BENCHMARKS.md)). The anatomy below was measured on
**2026-08-25**, when the step was **42.141 ms/token**, and three kernel changes
measured on that same day took it to **36.32 ms**. Later loader and checkpoint
work moved the step again without re-running the per-kernel profile, so every
share here is quoted against the run that produced it, and says which run that
is.

## Read this before reading a number below

**Profile mode is not bench mode.** Every launch in a profiled list signals a
host-visible kernel-timestamp event, which carries a flush an unprofiled list
never pays. So:

- **A per-kernel µs here is directly comparable to a probe's per-kernel µs.**
  The duration is `kernelStart` to `kernelEnd` from
  `zeEventQueryKernelTimestamp`, and the flush the signal carries lands *after*
  `kernelEnd`. Nothing about the kernel's own execution is inflated by being
  observed. That is what makes the in-situ column comparable to `probe_gemv`'s
  transplanted floors.
- **The inter-kernel gap is the opposite: it is an upper bound.** Every µs
  between one `kernelEnd` and the next `kernelStart` in a profiled list contains
  one flush. The gap is quoted twice below, once as measured and once corrected.
- **The fence wall here is not ms/token.** The engine's step time is the
  `--bench` median in [BENCHMARKS.md](BENCHMARKS.md). Nothing in this document
  replaces it, and no row here is ever a bench row.
- **A mean is not an attribution.** Every single-run number carries this
  instrument's drift and nothing in a single run says how much. `--profile
  --repeats R` prices that directly, and the floor it measures is in "The
  attribution floor" below. Read it before believing any delta under about
  0.3 ms.

Every number says which kind it is: **measured** (this instrument or a named
earlier one), **derived** (arithmetic over two measurements), or **estimated**
(a model or an extrapolation).

## How it was measured

```bash
tools/box.sh run "./build/src/cli/b70-decode <checkpoint> \
  --profile --depth 4096 --steps 32"
```

Idle box (load average 0.40), 2026-08-25, `CL_DRIVER_VERSION 26.27.39122.14`,
Intel Arc Pro B70, 256 EUs, `--max-len 16384`, `debug_resid` off.

`--profile` ingests the 4096 synthetic ids on an **un-instrumented** list, which
is an ordinary decode, then builds a second, profiled list against the same
`DecodeBuffers` and replays 32 instrumented steps, resetting every event before
each replay and reading every duration only after the fence.

> **645 is this run's launch count, not today's.** Every table in the first half
> of this document is the step before the residual-norm split, which was 645
> launches. That change made each of the 129 `prep_res_norm` sites two launches,
> so the current walk is **774** and a run of the same command today prints 774
> everywhere 645 appears below. Each table says which it is.

Positions 4096 to 4127, so `nb`, the count of live attention blocks, is 17
throughout at the then-current `ATTN_BLOCK` of 256: the same regime the recorded
bench row measures.

**Reproducibility (measured).** Two independent runs of the identical command,
three minutes apart:

| | run A | run B | spread |
|---|---|---|---|
| Σ of 645 kernel durations | 41571.315 µs | 41568.965 µs | **0.006%** |
| fence wall (profiled) | 42422.761 µs | 42409.736 µs | 0.031% |
| `gemv` family | 24628.975 | 24617.643 | 0.05% |
| `attn_decode` | 5782.402 | 5783.561 | 0.02% |
| `prep_res_norm` | 2871.012 | 2874.294 | 0.11% |
| `gdn_step` | 733.245 | 737.467 | 0.58% |
| `attn_prep` (the smallest family quoted) | 51.257 | 49.967 | 2.5% |

Within a run, the per-step spread over the 32 replays is 41522.708 to
41716.458 µs (0.47%). **Run A is quoted throughout below**, and anything that
turns on a difference smaller than about 1% is not claimed.

## The anatomy - in situ against the floors

Per kernel family, one decode step at depth 4096. "Probe floor" is the
per-kernel measurement transplanted from `probe_gemv` 2026-08-24
([12-kernels.md](12-kernels.md), `gemv` and `gemv_bf16` → Measured); the other
families never had one.

| family | launches | **in-situ ms** | probe-floor ms | delta | kind |
|---|---|---|---|---|---|
| `gemv` (int4, five shapes) | 256 | **24.629** | 23.995 | +2.6% | measured, in situ |
| `gemv_bf16`, `lm_head` | 1 | **4.376** | 4.351 | +0.6% | measured, in situ |
| `attn_decode` | 16 | **5.782** | (2.360 estimated) | **+145%** | measured, in situ |
| `prep_res_norm` | 129 | **2.871** | | | measured, in situ |
| `gemv_bf16`, `a‖b` | 48 | **2.335** | (never probed) | | measured, in situ |
| `gdn_step` | 48 | **0.733** | | | measured, in situ |
| `prep_silu_mul` | 64 | **0.623** | | | measured, in situ |
| `prep_gated_head` | 48 | **0.079** | | | measured, in situ |
| `attn_reduce` | 16 | **0.076** | | | measured, in situ |
| `attn_prep` | 16 | **0.051** | | | measured, in situ |
| `embed_gather` | 1 | **0.010** | 0.008 (with argmax) | | measured, in situ |
| `argmax_stage1` + `stage2` | 2 | **0.007** | same | | measured, in situ |
| **Σ kernel durations** | **645** | **41.571** | | | measured, in situ |

And per compiled variant, the rows `probe_gemv` prices one for one:

| shape (variant) | `S` | launches | in-situ µs each | probe µs | delta | in-situ ms | floor ms |
|---|---|---|---|---|---|---|---|
| `gate‖up` 5120×34816 | 4 | 64 | **182.377** | 176.7 | +3.2% | 11.672 | 11.309 |
| `down` 17408×5120 | 16 | 64 | **89.513** | 89.1 | +0.5% | 5.729 | 5.702 |
| `lm_head` bf16 5120×248320 | | 1 | **4375.557** | 4350.5 | +0.6% | 4.376 | 4.351 |
| `qkv‖z` 5120×16384 | 1 | 48 | **81.401** | 79.7 | +2.1% | 3.907 | 3.826 |
| `out_proj` + `o_proj` 6144×5120 | 16 | 64 | **33.499** | 31.3 | +7.0% | 2.144 | 2.003 |
| `q‖k‖v` 5120×14336 | 1 | 16 | **73.550** | 72.2 | +1.9% | 1.177 | 1.155 |
| **the 257-launch GEMV family** | | **257** | | | **+2.3%** | **29.005** | **28.346** |

**The single most consequential line in this document: the probe transplant was
right.** 28.346 ms was a floor, and in situ the same 257 launches cost
29.005 ms, **2.3% above it**. Every shape is within 7% of its probe number and
four of six are within 2.2%, so the stall-from-the-previous-kernel effect the
transplant could not see is real but small. `out/o_proj` (+7.0%) is the only
shape where it is worth naming: it is the shape entered directly from
`attn_reduce` or `prep_gated_head`, the two kernels with the least data in
flight.

## The dispatch gap, measured in situ

| | µs/step | µs/launch | kind |
|---|---|---|---|
| profiled gap (fence wall − Σ durations) | **851.4** | **1.320** | measured, **upper bound** |
| un-instrumented gap | **473** | **0.733** | derived (below) |
| `probe_replay`'s noop floor × 645 | 335 | 0.52 | estimated |

The **derived** row is the honest in-situ number and it is arithmetic over two
instruments: the bench step is 42.141 ms/token of which 0.097 ms is host outside
the fence (both measured), so an un-instrumented fence is 42.044 ms; Σ of the
in-situ kernel durations is 41.571 ms; the difference is 0.473 ms. It sits
between the noop floor (0.52 µs) and the `ctrl_read` floor (0.63 µs) recorded in
[07-open-questions.md](07-open-questions.md) question 5, a shade above both,
which is what 645 kernels that each read the control block should cost.

Two consequences.

1. **The instrument's own cost is measured: 0.379 ms/step, 0.587 µs per launch**
   (derived: profiled wall 42.423 minus un-instrumented fence 42.044). A
   profiled step is 0.9% longer than a real one. That is the whole distortion,
   and it lands in the gap, never in a kernel duration.
2. **Kernel *count* is exonerated, on a measured basis rather than an estimated
   one.** 0.473 ms is 1.1% of the step. Fusing kernels to reduce their number is
   not a lever at 0.7 µs each.

## The step, re-partitioned

Every launch and the host appear exactly once, and the rows sum to the measured
bench step:

| part | launches | ms/token | share | kind |
|---|---|---|---|---|
| GEMV, int4 mixers and MLPs plus bf16 `lm_head` | 257 | **29.005** | 68.8% | measured, in situ |
| `attn_decode` | 16 | **5.782** | 13.7% | measured, in situ |
| `prep` (`res_norm` 129 + `silu_mul` 64 + `gated_head` 48) | 241 | **3.573** | 8.5% | measured, in situ |
| `a‖b` GEMV | 48 | **2.335** | 5.5% | measured, in situ |
| `gdn_step` | 48 | **0.733** | 1.7% | measured, in situ |
| `attn_prep` + `attn_reduce` | 32 | **0.127** | 0.3% | measured, in situ |
| `embed_gather` + `argmax` | 3 | **0.017** | 0.04% | measured, in situ |
| dispatch gap, un-instrumented | | **0.473** | 1.1% | derived |
| host, outside the fence | | **0.097** | 0.2% | measured (bench) |
| **total** | **645** | **42.141** | 100% | measured (bench, median of three) |

**98.6% of the step is kernel time inside the fence** (41.571 of 42.141), 1.1%
is dispatch and 0.2% is the host.

Three of these rows moved on the same day, and the sections that follow measure
each move: the `a‖b` GEMV fell to **0.256 ms**, `prep`'s `res_norm` share fell
from 2.871 to **0.484** (so the `prep` row reads **1.202 ms** and the launch
count reads **774**), and the attention family fell from 6.058 to **3.839** with
the launch count unchanged. The step after all three is **36.32 ms**. No other
row here has been re-measured except as drift.

## What the measurement says that nothing else did

### 1. The unit of parallelism is the subgroup, not the work-group

Four points, all measured in this run, using the per-call byte counts in
[12-kernels.md](12-kernels.md):

| kernel | work-groups | subgroups | GB/s total |
|---|---|---|---|
| `prep_res_norm` (SP16) | 1 | 16 | 17.0 |
| `a‖b` GEMV | 2 | 8 | 27.0 |
| `prep_silu_mul` | 5 | 80 | 60.8 |
| `gdn_step` | 192 | 3072 | 540 (saturated) |

The first reading of this table divided by work-groups, concluded that a single
Xe-core cannot exceed about 15 GB/s and that saturating 590 GB/s therefore needs
roughly 40 work-groups, and predicted that giving `a‖b` four times the
work-groups would give it four times the bandwidth. **That experiment was run
and it bought nothing**: 48.774 to 49.127 µs/launch at an unchanged subgroup
count, a 0.7% regression where the model said about 12 µs. Dividing by
subgroups instead gives 1.06 / 3.36 / 0.76 GB/s per subgroup, and `a‖b`'s figure
is three to four times the others because its subgroup pulls 256 B per load
(`intel_sub_group_block_read_us8`) where prep's pulls 64 B, which is the shape
of a **latency-bound** kernel with one load in flight per thread: rate = bytes
per load ÷ about 60 to 75 ns.

**The corrected invariant held every time it was used afterwards**, and it has
now been measured at 8, 16, 32, 128 and 320 subgroups across two kernels. It is
sub-linear at the top: 16× the subgroups bought 9.1× on `a‖b`, and 20× bought
10.2× on the residual-norm fold.

One caveat the four points do not cover. The table mixes two kinds of traffic:
`prep_res_norm`'s 320 KB of partials and `prep_silu_mul`'s 557 KB were written
by the GEMV immediately before them and are **L2-served**, while `a‖b`'s 1.31 MB
of weights are a cold **DRAM** read. That they land within a factor of two of
each other says the limit is the core's outstanding-load capacity rather than
where the bytes come from, but that is an inference from three low points plus
one saturated one, and nothing here locates the knee between them.

### 2. `attn_decode` is 2.45× its estimate, and four cost models died proving why

5.782 ms in situ against the 2.360 ms extrapolated from a two-point line through
whole-step `--bench` differences at two depths. Four in-situ points, all at
`ATTN_BLOCK` 256:

| run | `nb` | live WGs | positions attended | µs/launch | ms/token |
|---|---|---|---|---|---|
| depth 4096, `--max-len 16384` | 17 | 68 | 4097 | **361.400** | 5.782 |
| depth 1024, `--max-len 16384` | 5 | 20 | 1025 | **338.165** | 5.411 |
| depth 64, `--max-len 16384` | 1 | 4 | 65 | **153.608** | 2.458 |
| depth 64, `--max-len 4096` | 1 | 4 | 65 | **153.665** | 2.459 |

**The fixed grid is free, re-confirmed per kernel.** Rows 3 and 4 differ by 192
idle work-groups per layer and by **0.04%**. A whole-step comparison put the
same conclusion at 0.046 ms/token ([07-open-questions.md](07-open-questions.md),
question 12); this is the same verdict measured where it happens.
Context-bucketed lists are not needed.

Everything else about this kernel resisted modelling. Four models were written
down with their predictions **before** the run that would test them, and all
four missed:

| model | fitted to | predicted | measured | verdict |
|---|---|---|---|---|
| linear in blocks plus an intercept | `nb` 1 and 17 | 205.6 µs at `nb` = 5 | **338.165** | falsified, −39% |
| wave/occupancy, flat while 4·`nb` ≤ 32 | `nb` 1 and 17 | 153.6 µs at `nb` = 5 | **338.165** | falsified, −55% |
| `F + fill·P` (F ≈ 90.8 µs fixed, P ≈ 247.4 µs per full block) | the `nb` 1 and 5 points | 214.5 µs at a 128-position block | **296.684** | falsified, +38.3% |
| refit `F + walk` (F ≈ 223.4, walk ≈ 146.6) | the 256 and 128 block points | 260.0 µs at a 64-position block | **224.046** | falsified, −13.8% |

The first model's "2.25 ms depth-independent term" was an artefact of fitting a
line through a point that is not on the same footing as the others: at `nb` = 1
the single live block holds **65 of its 256 positions**, so that point measures
a quarter-block and not a cheap full one. The *direction* every model implied
held, a finer block is the right attack and it paid, but no number any of them
produced survived. **Where a quantity has resisted four fits, stop fitting and
remove terms**, which is what the probe battery further down does.

Two measurements from this set survive as measurements, with their conditions
attached:

- **`nb` 5 to 17 is 3.40× the live work-groups for +6.9% of time.** That is
  real, and it is real *at `ATTN_BLOCK` 256*, where 20 to 68 work-groups is
  320 to 1088 hardware threads on a device with about 2048 slots, which is a
  low-occupancy regime. At the block size that shipped the same axis is
  **linear**, 68 to 1004 live work-groups is 14.8× for 11.8× the time. "Work-group
  count is nearly free" holds only while the device is not full.
- **It is not bandwidth-bound.** The effective rate rose with depth: 10.4 GB/s
  at `nb` = 1, 74.5 at `nb` = 5, 278.6 at `nb` = 17 (47% of the measured 590)
  counting the six-pass q-head reread since removed; the *unique* KV at depth
  4096 is 46 GB/s, which cannot be DRAM-limited.

### 3. `gdn_step` is exonerated, completely

0.733 ms in situ against a **0.671 ms** traffic floor (396 MB/token at the
measured 590 GB/s). It runs at **540 GB/s effective, 92% of the device**, on 192
work-groups with no spills. The suspicion that this kernel was occupancy-bound
or register-pressured was withdrawn on this measurement: the profile puts it at
**1.09× its own bound**, the entire kernel made perfect is worth 0.06 ms, and
the 48×4 grid with its four-fold redundant conv reads was the right trade.

### 4. `lm_head` is 10.4% of the step and nothing tunes it

4.376 ms in one launch, 0.6% above its probe floor and **581 GB/s in situ, 98.5%
of the measured 590** (the 97% the probe reports is against the 600 GB/s
theoretical denominator; this document uses the measured one everywhere else).
There is nothing to tune in the kernel. The bytes are a different question:
the same `gemv` at int4 measures **1178.164 µs, 573.3 GB/s, 2.9% above the
bandwidth floor** on a checkpoint that ships a packed head
([12-kernels.md](12-kernels.md), "The sixth shape"). The published checkpoint
ships a bf16 head, vLLM cannot load an int4 one, and this project quotes only
the byte-matched comparison ([BENCHMARKS.md](BENCHMARKS.md) says why).

## The three changes of 2026-08-25, measured

All three were executed against this document's own ranking, with this
document's own instrument, on the same box on the same day. They are recorded
here in the order they were run, because each one corrected the model the next
was designed on.

### The `a‖b` GEMV K split: 2.341 to 0.256 ms/token

`src/kernels/gemv_bf16.cl`. Five `--profile --depth 4096 --steps 32` runs, same
box, same session, each building the full engine and replaying 32 instrumented
steps. The column that decides is `gemv_bf16` at 5120×128, 48 launches:

| tiling (`COLS_PER_WG`, `KSPLIT`) | work-groups | **subgroups** | µs/launch | ms/token | GB/s | GB/s per subgroup |
|---|---|---|---|---|---|---|
| `{64, 1}`, as it shipped | 2 | 8 | **48.774** | **2.341** | 26.9 | 3.36 |
| `{16, 1}`, more work-groups only | 8 | 8 | **49.127** | 2.358 | 26.7 | 3.34 |
| `{16, 1}` plus 4 block reads in flight | 8 | 8 | **49.052** | 2.354 | 26.7 | 3.34 |
| `{16, 4}`, K split four ways | 8 | 32 | **13.115** | 0.630 | 99.9 | 3.12 |
| **`{16, 16}`, shipped** | **8** | **128** | **5.340** | **0.256** | **245** | 1.92 |

All five are measured, in situ, per launch. The shipped row re-measured at the
committed binary reads 5.342 µs, 0.04% from the tuning run, which is what this
instrument's reproducibility looks like.

**Two of the three things tried were worth zero, and they were the two the
ranking called certain.** What the three flat points share is that the launch
still had eight subgroups. `COLS_PER_WG` re-spreads the same eight over more
work-groups; loads-in-flight changes what one of them does. Only splitting K
makes more of them. The loads-in-flight variant was worse than useless: it was
bit-identical and worth nothing on this kernel *and* cost `lm_head` 1.3% by
perturbing its codegen, which is why it was deleted.

**On the baseline this table starts from.** Row 0 is 48.774 µs / 2.341 ms, while
the anatomy tables above say 48.6 µs / 2.335 ms. Those are **two runs of this
same instrument at the same binary**, 0.3% apart, which is its run-to-run spread
on this row. The arithmetic here uses 48.774 / 2.341 throughout, because a
before/after has to be two rows of one comparison; the anatomy tables keep
2.335, because that is what the run they report measured. The two must not be
mixed inside one sentence.

**The whole step, before and after:**

| | before | after | delta | kind |
|---|---|---|---|---|
| `gemv_bf16_M1_K5120_N128…` | **2341.136 µs** | **256.403 µs** | **−2084.7** | measured, in situ |
| `gemv_bf16`, `lm_head` (control) | 4378.555 | 4379.089 | +0.5 (0.01%) | measured, in situ |
| Σ of 645 kernel durations | 41597.288 | 39631.634 | −1965.7 | measured, in situ |
| fence wall (profiled) | 42453.888 | 40515.713 | −1938.2 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **42.141 ms** | **40.266 ms** | **−1.875 ms** | **measured (bench)** |

**The three deltas do not agree exactly, and the difference is the honest error
bar of this comparison rather than a missing effect.** The change's own row
falls 2.085 ms; Σ falls 1.966 ms because the other 597 launches read **+119 µs
(+0.30%)** higher in the after run, spread over every family, which is
run-to-run drift of the kind measured at 0.006% for two runs three minutes apart
and an order of magnitude larger after a day of work on the box. **The bench row
is the number that counts** (−1.875 ms, median of three, 0.04% spread) and it is
0.09 ms below Σ's delta, which is the same drift seen from the other side.
Nothing in this change is claimed at better than ±0.1 ms.

**What is left in this kernel.** 1.31 MB at the measured 590 GB/s is 2.22 µs and
the shipped tiling is 5.34 µs, so the whole remaining prize is **0.15 ms/token**,
and a 32-way split, which would need its own golden-gate run because it reorders
the sum again, cannot repay it. `a‖b` is now the step's eleventh largest variant
row at 0.65% of Σ, below `prep_silu_mul`. It is finished.

### The residual-norm split: 2.893 to 0.484 ms/token

`src/kernels/prep.cl`. RMSNorm's mean is over the whole row, so one kernel's
reduction domain is one work-group and no tiling *inside* a launch can change
it. The split is into two:

```
stage A  prep_res_fold(partials, resid, sumsq)      grid (20, M), WG 256
stage B  prep_norm_finish(sumsq, resid, norm_w, x)  grid (20, M), WG 256
```

`G = 20` at K = 5120 is **one element per lane**, so the fold runs on
20 × 16 = **320 subgroups against 16**. Three profile runs, same box, same
session, before at the binary that carried the `a‖b` K split:

| shape | stage A µs/launch | stage B µs/launch | the site, µs/step | against baseline |
|---|---|---|---|---|
| `prep_res_norm`, one work-group | | | **2893.180** | |
| two-stage, stage B on **1** work-group | 1.973 | **9.194** | **1440.586** | −1452.594 |
| **two-stage, stage B on 20 (shipped)** | **2.012** | **1.739** | **483.886** | **−2409.294** |

All measured, in situ, 129 launches of each stage per step.

> **Disclosure on the middle row.** It was produced by an *uncommitted* one-line
> edit to `src/runtime/capture.cc` binding `prep_norm_finish_..._W1` on a grid
> of 1, reverted immediately after the run. **No commit reproduces it**, so it
> cannot be re-derived by checking out history: the `W=1` *binary* is still
> built and still unit-tested, but its capture binding is not. Re-measuring it
> means re-applying that one line.

**The fold is 60% of the change and the rescale is the other 40%.** Writing
stage B as a single work-group was the obvious design, because the *reduction*
is what was being split; the measurement says the rescale pass, 10 KB of `resid`
plus 20 KB of `norm_w` plus 10 KB of `x` through one Xe-core, is the same
latency-bound walk and wanted the same grid. At `W = 1` stage B runs at
**4.5 GB/s**, worse per byte than the single-work-group kernel it came from,
because its head is a 20-deep chain of `sumsq` loads that nothing overlaps when
only one work-group is live.

**The subgroup curve held, and this is its fourth point.** Stage A moves
348,240 B in 2.012 µs, **173 GB/s**, against the single-work-group kernel's
17.0. That is **20× the subgroups buying 10.2×**, the same sub-linear shape the
`a‖b` split measured (16× the subgroups bought 9.1×).

**The whole step, before and after:**

| | before | after | delta | kind |
|---|---|---|---|---|
| `prep_res_norm` (129 launches) | **2893.180 µs** | | | measured, in situ |
| `prep_res_fold` (129) | | **259.495 µs** | | measured, in situ |
| `prep_norm_finish` (129) | | **224.391 µs** | | measured, in situ |
| **the site** | **2893.180** | **483.886** | **−2409.3** | measured, in situ |
| `gemv_bf16`, `lm_head` (control) | 4378.942 | 4380.589 | +1.6 (0.04%) | measured, in situ |
| `a‖b` GEMV | 256.624 | 265.000 | +8.4 (3.3%) | measured, in situ |
| the 516 untouched launches, Σ | 36728.369 | 36910.450 | **+182.1 (+0.50%)** | measured, in situ |
| Σ of all kernel durations | 39621.549 (645) | 37394.336 (774) | −2227.2 | measured, in situ |
| fence wall (profiled) | 40496.734 | 38400.907 | −2095.8 | measured, in situ |
| dispatch gap (profiled, upper bound) | 875.185 / 1.357 µs per launch | 1006.571 / 1.300 | +131.4 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **40.266 ms** | **38.046 ms** | **−2.220 ms** | **measured (bench)** |

**The three deltas do not agree exactly, and the arithmetic that closes them is
this:** the site falls 2.409 ms; +0.182 ms comes back as run-to-run drift on the
516 untouched launches; and +0.095 ms is **derived** for the 129 extra
dispatches at the un-instrumented 0.733 µs/launch measured above. −2.409 +
0.182 + 0.095 = **−2.133 ms predicted** against **−2.220 measured**, 0.087 ms
apart, inside the ±0.1 ms nothing here is claimed better than.

**The launch count moved on purpose, and it was cheap.** 645 to 774. At the
derived 0.733 µs/launch that is 0.095 ms/token bought for 2.409 ms saved, a
**25.5:1** trade, and it is the measured answer to a long-standing worry that
kernel count is a first-class cost. It is not: the step's dispatch total is
about 0.57 ms derived (774 × 0.733), 1.5% of 38.05.

**What is left in this pair.** 50.2 MB/token of traffic is 0.085 ms at the
measured 590 GB/s and the pair costs 0.484, **5.7× its floor**, against the
single-work-group kernel's 34.9×. The remaining prize is about **0.40 ms** and
it is in two places: stage B's 20-deep `sumsq` load chain, which a lane-parallel
load plus an SLM tree would shorten, and stage B's re-read of `resid`. Neither
can repay its own golden-gate run.

### The attention block retile: the attn family 6.058 to 3.839 ms/token

One number, `ATTN_BLOCK` in `src/kernels/CMakeLists.txt`, 256 to 64. The sweep,
the footprint argument and the correctness cost are in
[12-kernels.md](12-kernels.md); the attribution is here.

| | before | after | delta | kind |
|---|---|---|---|---|
| `attn_decode` (16 launches) | **5919.814 µs** (369.988/launch) | **3584.736** (224.046/launch) | **−2335.1** | measured, in situ |
| `attn_reduce` (16) | 80.241 (5.015/launch) | 196.195 (12.262/launch) | +116.0 | measured, in situ |
| `attn_prep` (16) | 58.057 | 58.070 | +0.0 | measured, in situ |
| **the site (attn family)** | **6058.112** | **3839.001** | **−2219.1** | measured, in situ |
| `gemv_bf16`, `lm_head` (control) | 4379.899 | 4388.493 | +8.6 (0.20%) | measured, in situ |
| `gdn_step` | 748.659 | 761.706 | +13.0 (1.74%) | measured, in situ |
| the 726 untouched launches, Σ | 31329.398 | 31503.440 | **+174.0 (+0.56%)** | measured, in situ |
| Σ of all kernel durations (774 both) | 37387.510 | 35342.441 | −2045.1 | measured, in situ |
| fence wall (profiled) | 38413.428 | 36357.567 | −2055.9 | measured, in situ |
| dispatch gap (profiled, upper bound) | 1025.919 / 1.325 µs per launch | 1005.914 / 1.300 | −20.0 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **38.046 ms** | **36.32 ms** | **−1.73 ms** | **measured (bench)** |

−2219.1 + 174.0 = **−2045.1**, the Σ delta exactly. The launch count did **not**
move, 774 before and after, so unlike the norm split there is no dispatch term
to add, which makes this change's arithmetic simpler and its residual harder to
excuse.

**And there is a residual.** The in-situ attribution predicts **−2.045 ms** on
the bench; the bench measured **−1.73**. That is **0.32 ms apart**, against
0.087 for the norm split and the ±0.1 ms this instrument had been claimed at.
Neither the launch count (unchanged) nor the drift (+0.174, already counted)
explains it. The one *named* contributor is too small by an order of magnitude:
`--bench` sweeps `pos` 4096 to 4351, so `nb` runs 65 to 69 at `ATTN_BLOCK` 64
where it was a flat 17 at 256, and the after-run therefore averages about 3%
more merge work per token than the single profile point that priced it, worth
about 0.01 ms by `attn_reduce`'s own slope. **The remaining 0.31 ms is
unattributed.** It is recorded as unattributed, because this document's whole
method is that a number nobody can account for is a finding and not a rounding
error.

### The three together

Bench rows are the recorded medians ([BENCHMARKS.md](BENCHMARKS.md)); in-situ
rows are `--profile --depth 4096 --steps 32`, the same instrument throughout.
All measured.

| change | what shipped | in-situ site | in-situ Δ | bench step | **bench Δ** |
|---|---|---|---|---|---|
| `a‖b` GEMV K split | `{COLS_PER_WG, KSPLIT}` `{64,1}` to `{16,16}`, 8 to 128 subgroups | 2.341 to **0.256** ms | −2.085 | 42.141 to 40.266 | **−1.875** |
| residual-norm split | one 1-work-group kernel to `prep_res_fold` + `prep_norm_finish`, 20 work-groups each, 16 to 320 subgroups | 2.893 to **0.484** ms | −2.409 | 40.266 to 38.046 | **−2.220** |
| attention block retile | `ATTN_BLOCK` 256 to **64**, 68 to 260 live work-groups | attn family 6.058 to **3.839** ms | −2.219 | 38.046 to 36.32 | **−1.726** |
| **all three** | | | **−6.713** | **42.141 to 36.32** | **−5.821** |

The golden gate was element-exact on every determined row before and after each
one.

**The two Δ columns differ by 0.892 ms and three measured terms close it to
0.003.** Drift on the launches each change did *not* touch: +0.119 (+0.30%),
+0.182 (+0.50%), +0.174 (+0.56%). The norm split's 129 extra dispatches: +0.095,
derived at 0.733 µs/launch. The attention retile's full in-situ to bench miss:
+0.319. 0.475 + 0.095 + 0.319 = **0.889** against **0.892**. Nothing else is
needed to explain the set, and the largest single term in that reconciliation is
still the one nobody can attribute.

**Read per launch, the same numbers say something new.** The un-instrumented gap
was **0.733 µs/launch** before the three changes (0.473 ms / 645); after them it
is 36.32 − 0.101 host − 35.342 Σ = 0.877 ms / 774 = **1.133 µs/launch, +55%**
(derived, cross-run). **The cost outside the kernels grew while the kernels
shrank.** The obvious suspect is drain, since `attn_decode` retires 260
work-groups where it retired 68, but nothing here measures that, and it is why
repeated-run averaging was built before any further sub-0.5 ms design.

**What did not work, kept because a record of only the wins is not a
measurement:** `a‖b` at `{16,1}` (+0.7%, worth zero), the block-read variant
(worth zero *and* +1.3% on `lm_head`, a cost rather than a saving), the norm
split's stage B on one work-group (60% of the win instead of 100%), and
`ATTN_BLOCK` 32 (0.070 ms/token better at family level, rejected on a 101.4 MB
`attn_part`).

**The ranking's record: three for three on yield, zero for three on mechanism.**
Every change landed in or near its predicted band; every one landed for a reason
the ranking got wrong. The K split's work-group count was worth zero and the
K split was everything; the fold was priced and the rescale, treated as a
formality, was 40% of the win; the retile worked, the model that priced it was
falsified twice, and the block size that shipped was 64 rather than the 128 that
was scoped. That is an argument for the instrument, not for the ranking.

## The attribution floor, measured

This profiler reproduces to **0.006% over two runs three minutes apart**, and
yet the *untouched* launches drifted **+0.30% / +0.50% / +0.56%** across the
three changes above, 0.12 to 0.18 ms per comparison, and two of the three then
missed their in-situ to bench prediction **in opposite directions** (+0.087 and
−0.319). Errors that run both ways at 0.09 to 0.32 ms are a resolution limit,
not a missing constant.

### What was built

`b70-decode --profile` takes **`--repeats R`** (default 1). At `--repeats 1` it
prints the same numbers as before the flag existed, because every rollup is
computed from grand totals over all `R × steps` replays. R **sessions** of
`--steps` instrumented replays run over **one** ingest, and each session begins
by rewinding `Control::pos` to the post-ingest value. That rewind is the point:
**every session profiles the identical launch shape**, the same 774 launches at
the same positions with the same `nb`, so what the spread across them contains
is drift and scheduling and nothing else.

**`pos` is the only thing rewound, and that is deliberate.** The 150.99 MB of
recurrent state keeps evolving across sessions, so session *r* is numerically a
different token from session 1 while being structurally the same launch. Zeroing
it would be worse, not better: a zero-filled KV cache is losslessly compressible
on this device and `attn_decode` would read it *faster* than it reads real
values. Empirically the drift this leaves is not a trend: across fifteen
sessions the Σ series rises and falls without direction, and the per-session ids
repeat identically in all three processes.

The report gains a per-family **mean, sd, min..max** and a **2σ-resolvable Δ**
column, and the same three rows for Σ, for the fence wall and for the derived
gap. The 2σ column is `2·sd·sqrt(2/R)`: two binaries measured this way give two
means whose difference carries a standard error of `sd·sqrt(2/R)`, and that is
the bar an attribution has to clear. It is printed per family because a family
is the unit an attribution is written in.

### The floor

2026-08-25, three **independent processes** of `--profile --depth 4096 --steps
32 --repeats 5`, 480 profiled replays in all. The box's CPU was busy with an
unrelated build throughout; every number is a **device-clocked kernel
timestamp**, which is what makes it quotable under load, and no bench-grade
wall-clock absolute is recorded in this section.

| family | proc 1 | proc 2 | proc 3 | **within-process 2σ (worst of 3)** | **across-process 2σ** |
|---|---|---|---|---|---|
| `gemv` (256) | 24850.221 | 24866.465 | 24851.246 | 11.756 | **14.855** |
| `gemv_bf16` (49) | 4647.967 | 4650.310 | 4650.266 | 2.594 | **2.188** |
| `attn_decode` (16) | 3571.395 | 3571.074 | 3571.075 | 12.425 | **0.302** |
| `gdn_step` (48) | 755.142 | 753.733 | 757.863 | 2.508 | **3.428** |
| `prep_silu_mul` (64) | 639.450 | 639.130 | 638.090 | 2.096 | **1.161** |
| `prep_res_fold` (129) | 261.690 | 257.513 | 252.235 | 2.695 | **7.737** |
| `prep_norm_finish` (129) | 223.383 | 223.514 | 223.902 | 2.288 | **0.441** |
| **`attn_reduce` (16)** | **194.034** | **193.638** | **254.099** | 1.639 | **56.817** |
| `prep_gated_head` (48) | 84.107 | 82.774 | 86.936 | 1.363 | **3.471** |
| `attn_prep` (16) | 59.292 | 59.387 | 59.534 | 1.988 | **0.199** |
| **Σ of 774 kernel durations** | **35304.571** | **35315.042** | **35362.834** | **29.101** | **50.721** |
| fence wall (profiled) | 36361.637 | 36374.786 | 36424.767 | 21.547 | 54.391 |
| gap = wall − Σ (derived) | 1057.066 | 1059.743 | 1061.933 | 22.573 | 3.981 |

µs/step, measured. The within-process column is the tool's own 2σ at R = 5; the
across-process column is 2·sd·sqrt(2/3) over the three process means (derived).

> **The step-level attribution floor is 0.051 ms with `attn_reduce` and 0.011 ms
> without** (Σ, 2σ, three processes).
>
> **Both numbers have to be quoted together, because the first one IS the
> outlier.** The three Σ means differ by 58.263 µs; `attn_reduce` alone differs
> by 60.065 µs across the same three, **103% of the whole spread** (derived).
> Subtracting that one family from Σ leaves an across-process sd of 6.854 and a
> 2σ of **11.192 µs = 0.011 ms**. The honest statement is therefore: *this
> instrument resolves 0.011 ms at the step over the 773 launches that behave,
> and 0.051 ms including the one family that does not*, and nobody may quote
> either figure without the other.
>
> Against the ±0.1 ms the single-run method was claimed at, that is a 2×
> improvement on the conservative figure and 9 to 16× on the one that excludes
> the outlier. Per family the floor runs **0.2 to 15 µs**, with `attn_reduce`'s
> 56.8 the lone exception.
>
> **Three processes is n = 3, and this document will not pretend otherwise.** At
> 2 degrees of freedom a sample sd carries a 95% upper confidence bound of
> **4.42 × ŝ**, so 0.011 ms is a point estimate whose upper bound is about 0.049
> and 0.051 one whose upper bound is about 0.224. These are the best numbers
> available and they are not tight ones.

### The finding that justifies the exercise: within is not across

**`attn_reduce` reproduces to 0.3% inside every process and lands 31% apart
between them.** Processes 1 and 2 measure 194.034 and 193.638 µs/step with
within-process sd of 0.62 and 1.30; process 3 measures **254.099** with an sd of
0.72. Three tight distributions, two of which agree and one of which does not,
on **one binary, one box, one twenty-minute window.** Its across-process floor
is **56.8 µs against a within-process 1.6**, a factor of 35.

Nothing here explains it, and it is recorded unexplained. The launch shape is
identical (`nb` is 65 in all three, the merge loop is bounded by `nb`, and the
allocation sizes are fixed by `max_len`). **One thing it is not, and the
evidence is in the same transcript**: all three processes sample the *same ids
in the same order*, so the three runs walk identical data through identical
arithmetic, and a data dependence is ruled out. What is left points outside the
program, at the allocation placement of the 50.7 MB `attn_part` or at a power or
clock state.

What it *proves* is the design rule the tool now prints in its own output:
**R sessions inside one process are a lower bound on the floor, not the floor.**
An attribution must compare two binaries across processes, R ≥ 5 each, and read
the bar off the across-process spread. **And the window matters too**: these
three processes ran within twenty minutes, while the drift measured across the
three changes above spans a day. Both tiers are measured: **compare two binaries
in one sitting and the floor is 0.05 ms; compare across a day and it is about
0.15 ms.** The day-scale same-checkpoint control in
[BENCHMARKS.md](BENCHMARKS.md) puts pure day drift at or under 0.09%.

### The 0.31 ms unattributed residual does not resolve; it is confirmed

The obvious hope for a better instrument was that the residual was the
instrument. It is not. Σ is now known to **35.327 ms ± 0.031** (mean and
across-process sd of 480 replays) where it was one run's 35.342. Σ plus the
derived 0.567 ms dispatch plus the measured 0.101 ms host gives **35.995 ms
against the recorded 36.32**, a residual of **0.325 ms**. That is **6.4× the Σ
row's own 2σ**, and the bench side contributes only ±0.015 ms. The residual is a
real term at better than 6σ, and 6.4 is the *conservative* reading, because it
divides by the outlier-inflated 0.051; against the 0.011 ms the other 773
launches support it is 29×.

*One caveat that a better Σ does not repeal:* the 0.733 µs/launch dispatch rate
in that arithmetic was **defined** as this document's closing residual for the
645-launch step, so this construction cannot be quoted as an independent
witness. What it now is, is a construction whose one noisy input has been
pinned.

### One row this refines while it is here

`attn_decode` was recorded at **224.046 µs/launch** from a single profiled run.
The 480-replay mean is **223.199 µs/launch**, with an across-process 2σ of
**0.019 µs/launch**, 0.38% under the recorded figure and the tightest number in
this document. The single-run rows above are left as they were, because they are
what that change measured; 223.199 is what the row is today, and it is what the
probe below compares against.

## What `attn_decode`'s launch buys, named by subtraction

Four cost models died on this kernel before anybody asked what it was actually
doing. `tools/probe/probe_attn` fits nothing: it is a **copy** of `attn_decode`
(`src/kernels/attn.cl` is untouched by all of this) with one term removed per
compiled variant, so every number is a subtraction between two measurements.
Every variant computes a wrong answer on purpose and no runtime path binds one.

**Box conditions:** 2026-08-25, 22:17 to 23:35 CEST, load average 12.1 to 16.4,
an unrelated docker build holding the CPU throughout. The GPU was otherwise
idle. Every figure is a device-clocked kernel timestamp
(`zeEventQueryKernelTimestamp`), the same instrument `--profile` uses, so CPU
load cannot enter it. **No bench-grade wall-clock absolute is recorded here**,
and the in-situ figures compared against are quoted from the record rather than
re-measured.

### Three values pinned, so that nothing below mixes them

1. **The in-situ launch is 223.199 µs** (480-replay mean, ±0.019 at 2σ). A
   single profiled run recorded 224.046; the two differ by 0.38%.
2. **The in-situ row is 3.571 ms/token**, 16 × 223.199. The 3.585 quoted
   elsewhere is 16 × 224.046 and is the same row measured once instead of 480
   times. Every ms/token ceiling below is scaled on 3.571.
3. **The probe base is 207.396 µs**, the median of 200 timed launches in the
   canonical battery. Five whole-battery invocations put it at 206.250 to
   207.396, a **0.55% spread**, which is the reproducibility every Δ below
   inherits.

### Two methodology findings, both of which nearly voided the battery

> **Warm-up.** The first run of this probe read **295.417 µs** for the base
> variant and its block sweep came out non-monotonic. One warm-up replay of 40
> launches is not enough for this device's clock ramp: the same binary read
> **264.167 then 206.458 µs** on two consecutive invocations at that warm-up,
> and **207.083 to 207.604 µs, a 0.25% spread, on all eight invocations from
> about 100 warm-up launches upward**. The probe now replays the closed list 8
> times and drops the first 3, which is `tests/kernels/gemv_harness.h`'s and
> `probe_gemv`'s convention and is why neither of those ever saw this.
> **Everything the first run produced was discarded.**

> **The compiler. It voided three rows and the correction inverted the
> headline.** The obvious way to price a cache miss is to hold the address
> constant so every access hits L1. That address is **loop-invariant in the wave
> index over a `restrict` pointer**, so IGC may legally hoist the loads out of
> the wave loop entirely, in which case the variant measures *a quarter of the
> messages* rather than the same messages served from L1. The control is a
> variant with the same 8 KB L1-resident footprint whose row is a **function of
> the wave index**, so nothing is hoistable:
>
> | | address invariant in `w` | address a function of `w` | verdict |
> |---|---|---|---|
> | K | **180.000** | **182.083** | 1.1% apart, K was not hoisted |
> | V | **153.125** | **184.375** | **20.4% apart, V WAS hoisted** |
> | both | **141.146** | **173.750** | **23.1% apart** |
>
> The invariant-address rows are **withdrawn**. The hoist-proof control moves
> the cache/issue split from 30.9/24.4 to **16.2/39.2**, inverting which half is
> bigger. (The control pays one extra integer AND per access that the base
> kernel does not, so its 16.2% is if anything a slight under-estimate of the
> cache term and the 39.2% a slight over-estimate.)

### The two pre-registered validations

**Validation 1, the transplant reproduces the launch: PASSED.** Probe base
207.396 µs against the in-situ 223.199, so in situ is **+7.6%** above the probe
floor, the same sign and order as `probe_gemv`'s transplant. The bar was ±10%
and it clears.

**Validation 2, the block sweep reproduces: FAILED as pre-registered**, and this
section says so rather than moving the bar.

| `ATTN_BLOCK` | live WGs | probe µs | in-situ µs | probe against in situ | at the ±10% bar |
|---|---|---|---|---|---|
| 32 | 516 | 182.396 | 203.837 (one run) | **−10.5%** | **fails** |
| **64** | 260 | **207.396** | **223.199** (480 replays) | **−7.1%** | passes |
| 128 | 132 | 261.146 | 296.684 (one run) | **−12.0%** | **fails** |
| 256 | 68 | 349.375 | 369.988 (one run) | **−5.6%** | passes |

The per-halving deltas do not reproduce either: the probe gives −88.2 / −53.8 /
−25.0 µs where in situ gives −73.3 / −72.6 / −20.2. **What the data does
support**, stated at its own strength: the probe reproduces the **sign and the
shape** of all four points, monotone in block size with the last halving's gain
collapsing, and its **fidelity to the live launch varies across a 6.4-point band
(−5.6% to −12.0%) as a function of block size.** That band is a real property of
the transplant and it is carried as a caveat on every ceiling below.

### The battery - one term removed at a time (measured)

Depth 4096, `ATTN_BLOCK` 64, 4 rotated KV caches, 200 timed launches per row
after 120 of warm-up. Base **207.396 µs**.

| what the variant removes | µs/launch | Δ against base | Δ% |
|---|---|---|---|
| nothing (`base`, the control) | **207.396** | | |
| everything; return at the early-out (`earlyout`) | 1.667 | −205.729 | −99.2% |
| everything but the 6 q-head stagings (`qstage`) | 4.167 | −203.229 | −98.0% |
| **both KV global loads** (`noloads`) | **92.396** | **−115.000** | **−55.4%** |
| the V load alone (`nov`) | 115.417 | −91.979 | −44.3% |
| the K load alone (`nok`) | 165.208 | −42.188 | −20.3% |
| **both loads' cache misses, hoist-proof** (`hotkv2`) | **173.750** | **−33.646** | **−16.2%** |
| K's alone (`hotk2`) | 182.083 | −25.313 | −12.2% |
| V's alone (`hotv2`) | 184.375 | −23.021 | −11.1% |
| the whole online update (`nosoftmax`) | 153.125 | −54.271 | −26.2% |
| `exp` alone (`noexp`) | 175.312 | −32.084 | −15.5% |
| `exp`'s precision only, `native_exp` (`natexp`) | 180.938 | −26.458 | −12.8% |
| all 6 work-group barriers per wave (`nobar`) | 172.500 | −34.896 | −16.8% |
| the tree and its 4 barriers (`notree`) | 185.521 | −21.875 | −10.5% |
| 3 of the 4 tree barriers' SCOPE only (`sgtree`) | 199.271 | −8.125 | −3.9% |
| half the K messages, 64 B each (`vec2`) | 185.000 | −22.396 | −10.8% |
| three quarters of them, 128 B each (`vec4`) | 185.938 | −21.458 | −10.3% |
| the KV cache's 2048 B position stride (`kvt`) | 191.771 | −15.625 | −7.5% |
| K's stride alone (`kt`) | 199.167 | −8.229 | −4.0% |
| V's stride alone (`vt`) | 195.938 | −11.458 | −5.5% |
| the stride AND half the K messages (`kvt_vec2`) | 183.750 | −23.646 | −11.4% |
| **the loop-carried chain across waves** (`depbreak`) | **204.792** | **−2.604** | **−1.3%** |
| **the K load's latency, one wave of prefetch** (`prefetch`) | **206.354** | **−1.042** | **−0.5%** |
| **the 1.6 MB of `attn_part` stores** (`nopart`) | **206.562** | **−0.834** | **−0.4%** |
| both loads AND `exp` (`mathonly`) | 61.146 | −146.250 | −70.5% |
| the update and the tree (`loadsonly`) | 137.917 | −69.479 | −33.5% |

The lever rows were re-measured in three further invocations; expressed as Δ%
against the base of their own invocation they are stable to **±0.5 points**. One
earlier reading does not reproduce and is withdrawn: the first battery put `kt`
at −9.3%, against −3.5 to −4.1% in the four invocations since.

### The dominant term: the KV load path, 55.4% of the launch

Removing both global loads and changing nothing else takes the launch from
207.396 to **92.396 µs**. Nothing else is close: the whole online softmax is
26.2%, all six barriers per wave 16.8%, `exp` 15.5%, and the grid, the q
staging, the output stores and the loop-carried dependency are together 3.7%.

**It is a throughput term, not a latency chain.** Three independent rows say so
and they do not lean on each other:

- **`vec2` and `vec4` saturate.** Halving K's messages (32 B to 64 B) buys
  −10.8%; quartering them buys **no more** (−10.3%). Same bytes, fewer messages,
  and the win stops the moment the message count stops binding. That is a
  throughput signature and it involves no register change.
- **The launch is linear in the work-groups it is given** (the depth sweep
  below): 68 to 1004 live work-groups is 14.8× for 11.8× the time. A
  latency-bound kernel with idle machine would absorb that; a saturated one
  cannot.
- **Cutting the wave-to-wave dependency buys 1.3%.**

`prefetch` (−0.5%) is quoted but **not leaned on**: it holds `kcur[16]` and
`knext[16]`, +32 floats per lane, and a register-pressure cost could be masking
a latency win. It is consistent with the other three rows and it is the weakest
of the four.

**The term splits, and the hoist control inverted which half is bigger:**

| half | measured by | µs | share of the launch |
|---|---|---|---|
| **message issue**, what survives when every access hits L1 | `hotkv2` − `noloads` | **81.354** | **39.2%** |
| **cache service**, what an all-L1 kernel stops paying | base − `hotkv2` | **33.646** | **16.2%** |
| the two together | base − `noloads` | 115.000 | 55.4% |

The issue half is arithmetic anyone can check: the kernel issues **3.19 M global
load messages of 32 bytes** per launch (16 lanes of a subgroup read 16
consecutive `ushort` = 32 B, half a cache line; 16 K + 16 V messages per
subgroup per wave × 16 subgroups × 24 waves × 260 work-groups), plus about 3.7 M
SLM messages. At 207.396 µs that is 0.83 messages per Xe-core per cycle at about
2.4 GHz, the right order for a pipeline that is issue-bound, and consistent with
`vec2` paying and `vec4` not.

**Three arithmetic checks, including the one that does not close.**

1. **Loads and `exp` are additive to 0.6%.** `noloads` (−115.000) plus `noexp`
   (−32.084) is −147.084; `mathonly`, which removes both in one binary, measures
   **−146.250**.
2. **The update and the tree are not additive, and the gap is 8.8%.**
   `nosoftmax` (−54.271) plus `notree` (−21.875) is −76.146 against
   `loadsonly`'s **−69.479**, **6.667 µs of overlap**. Removing the tree makes
   the update cheaper and vice versa, so those two subtractions double-count. It
   is shown here beside the pair that does close, because a battery that only
   shows its clean subtractions is making the same mistake a record that only
   keeps its wins makes.
3. **The q-head axis is linear to 0.04%.** Running 1, 2, 3 and 6 of the six
   passes measures **66.771 / 96.979 / 122.396 / 207.396 µs**; least squares over
   those four points gives **39.46 µs + 27.98 µs per pass**, predicting 207.31 at
   six. That is a two-parameter fit over four points, arithmetic over
   measurements, and deliberately not extended to the depth or block axes, which
   are where the four dead models died.

### What the "fixed" term is fixed on

An earlier fit called about 90.8 µs of the launch a per-launch fixed cost and
nobody could say of what. The battery prices the candidates directly and **none
of them is it**: the whole 1024-work-group grid, early-out included, is
**1.667 µs** (1.63 ns per work-group, so a dead work-group is nine times cheaper
than this project had estimated, and context-bucketed lists get a second,
per-kernel witness); the six q-head stagings and their twelve barriers add
**2.5 µs**; the `attn_part` stores are **0.8 µs**. Total genuinely fixed cost:
**about 4.2 µs, 2.0% of the launch.**

The 39.46 µs intercept of the q-head line is therefore **not a fixed cost, it is
the first pass being colder than the other five**. Subtracting the 4.2 µs that
is genuinely fixed leaves the first q-head pass at about 62.6 µs against about
28.0 µs for each of the following five, a ratio of **2.24**: pass 1 fills the
block into cache and passes 2 to 6 largely re-read it, exactly as the
(kv-head, block) grid was designed to do. **The six-pass reread cost
5 × 27.98 = 139.9 µs, 67% of the launch, and it was not free**, because a cache
hit still issues a 32-byte message. That is the finding register-packed GQA
acted on ([12-kernels.md](12-kernels.md), `attn` → Work assignment): it does not
make the duplicate messages cheaper, it stops issuing them, and it measured
**−1.075 ms/token** in situ.

### Where the work-groups stopped being free

Sweeping the depth on the base variant at the shipped block size:

| depth | live WGs | µs/launch | µs per live WG |
|---|---|---|---|
| 512 | 36 | 50.417 | 1.400 |
| 1024 | 68 | 57.083 | 0.839 |
| 2048 | 132 | 113.646 | 0.861 |
| **4096** | **260** | **206.458** | **0.794** |
| 8192 | 516 | 389.583 | 0.755 |
| 16000 | 1004 | 674.896 | 0.672 |

From 68 to 1004 live work-groups, **14.8×**, the launch grows **11.8×**. That is
linear, with a mild and steadily shrinking discount. The earlier "work-group
count is nearly free" reading was measured at `ATTN_BLOCK` 256 over 20 to 68
work-groups, which is 320 to 1088 hardware threads on a device with about 2048
slots: a low-occupancy regime the shipped block size no longer enters. **The
claim is withdrawn for the shipped configuration**; what survives it, and is now
measured per kernel rather than per step, is that *dead* work-groups are free at
1.63 ns each.

### What of the 3.571 ms is recoverable

Its **unique-KV traffic floor** is 268 MB/token at the measured 590 GB/s =
**0.454 ms** (derived), so the excess is 3.12 ms.

> **Every ms/token figure in the next table is DERIVED and assumes the probe's
> share transfers to the live launch.** The live launch is 7.6% above the probe
> floor and this work did not place that 7.6%; it may sit on the load path, on
> the barriers, or on neither. Validation 2 makes the caveat concrete: fidelity
> moves across a 6.4-point band with block size, so a share measured on the
> probe is worth roughly ±1 share point when carried to the engine. Read every
> row as "the probe's share × the measured row", not as a promise.

| lever | what it is | ms/token | what it costs |
|---|---|---|---|
| the locality ceiling (`hotkv2`) | every KV access an L1 or SLM hit, a **same-message-count** bound | **−0.579** | a bound on locality substitutions only |
| `exp` removed (`noexp`) | | −0.553 | **not available.** `exp` is ruled by the rounding discipline and `native_exp` (−0.457) with it |
| all wave barriers (`nobar`) | | −0.600 | **not available.** The cross-subgroup ones publish `dot_red` |
| **transposed KV cache plus 64 B K messages** (`kvt_vec2`) | `[head][pos][dim]` and `ushort2` loads | **−0.407** | a cache-layout change in `attn_prep`/`attn_decode` and one reassociation of the K dot |
| 64 B K messages alone (`vec2`) | | −0.386 | one reassociation of the K dot; no layout change |
| transposed cache alone (`kvt`) | | −0.268 | **no arithmetic change at all** |
| **subgroup-scope tree** (`sgtree`) | 3 of 4 tree barriers narrow | **−0.139** | **no arithmetic change at all** |

**About 0.45 ms/token is reachable by tuning** (`kvt_vec2` and `sgtree` do not
add, so about −13% combined, 0.46 ms, derived) **and the rest is structural**,
because message count *within one pass* is set by the (position × dim)
decomposition, which is the algorithm's shape rather than a tuning knob.

**The one place that reading was wrong is the one that mattered.** Reading "the
decomposition is the algorithm's shape" as covering the *number of times* that
shape is walked led to a verdict of "no, the excess is not recoverable".
Register-packed GQA loads each work-item's 16 K and 16 V wave values into
private registers once and serves all six q-heads from them, with no layout
change, no reassociation and no barrier-scope change, and measured **−1.075
ms/token** in situ. That is more than twice what this battery called reachable
and it beats the `hotkv2` locality ceiling, because it removes messages rather
than making them hit. What survives: message count within one pass is still
structural, `hotkv2` is still a valid bound on same-message-count locality
attacks, and the levers in the table above are still unspent.

**It also removes one candidate from the unattributed residual, without closing
it.** The guess that the cost outside the kernels grew because `attn_decode` now
leaves 260 work-groups to retire where it left 68 prices out at 1.63 ns each,
so 192 extra per launch across 16 layers is **0.005 ms/token**. That is not a
refutation, and this document will not call it one: `earlyout` measures inside
the `kernelStart` to `kernelEnd` window and drain by definition lives outside it,
in the gap. What can be said is that the hypothesis now has to live *entirely*
in a term that was measured to go **down** 20 µs in the same before/after where
work-groups went 68 to 260.

### What died here, for the tally

Four models were pre-registered before the first run, with a point prediction
for every variant.

| model | its own strongest prediction | measured | verdict |
|---|---|---|---|
| **message-issue rate** (the primary) | the all-L1 variant "barely moves", about 190 µs | **173.750** | **wrong by −8.6%, right about the mechanism**: issue is 39.2% against cache service's 16.2%, the larger half, which is what it claimed. Its *number* missed |
| **memory latency** | `prefetch` ≥10% better | **−0.5%** | **falsified** |
| **the online softmax** | `nosoftmax` ≤123 µs | **153.125** | **falsified**; it is 26.2%, not dominant |
| **barriers** | `nobar` ≤157 µs | **172.500** | **falsified**; 16.8%, not dominant |

**Three models falsified outright and the fourth wrong in its number while right
in its mechanism**, and the question was answered regardless, because a battery
of subtractions does not need a surviving model. Two of this project's own
claims were also withdrawn on measurement (work-group count is nearly free; a
cache-served reread is therefore cheap), and **two of this probe's own first
answers were withdrawn on its own controls**: the 40-launch warm-up and the
hoistable rows whose replacement inverted the headline split. The probe
falsified itself twice before it falsified anything else.

## What this document does not settle

- **`attn_decode`'s decomposition.** The internal split is settled by
  subtraction, but what is left open is the decomposition itself: about
  0.45 ms/token of the 3.571 is reachable by tuning and the rest is a
  message-count problem set by the algorithm's shape. Eliminating whole passes
  is what paid; making one pass cheaper has not been tried.
- **The 0.31 ms outside the kernels.** Confirmed at better than 6σ, candidate
  list one entry shorter, cause unknown.
- **`attn_reduce`'s across-process 56.8 µs.** Three tight distributions, one of
  which disagrees with the other two on one binary in one twenty-minute window.
  Data dependence is ruled out; nothing else is.
- **M > 1.** Everything here is the M = 1 decode step. The small kernels are the
  ones whose costs move most with M, and none of that is measured.

---

## The prefill walk

Decode's anatomy above is per-kernel device timestamps off an instrumented
capture. **The prefill walk is not a capture**, so `--profile`'s `ProfileEvents`
machinery cannot see it. It needs a different instrument: `B70_PREFILL_PROFILE=1`
times host waits around each phase and inserts one extra wait to close each one.
Its own cost is measured rather than argued: 4119.1 ms profiled against 4202.8 ms
plain in the same binary, **−2.0%**, inside the run-to-run spread.

### The launch arithmetic

Derived from the walk in `src/runtime/prefill/step.cc` and asserted by
`prefill_smoke_test` through `Context::launches()` rather than restated.

On the Level Zero backend, per chunk at C = 2048:

| per | L0 launches | SYCL GEMMs | host `wait()`s |
|---|---:|---:|---:|
| GDN layer (48) | 135 | 0 | 0 |
| FA layer (16) | 130 + 4 × row blocks | 0 | 0 |
| chunk boundary | 1 (embed) | 0 | 0 |
| **one chunk at C = 2048** | **9073** | **0** | **0** |
| `step_head`, once per `prefill()` | 5 | 0 | 0 |

A GDN layer is 2 norm launches, the slab dequant and GEMM pairs of three int4
linears, the `a‖b` projection, 10 `gdn_chunk` launches and 2 more norms. An FA
layer is the same skeleton with `pf_attn_prep`, one softmax per kv group,
`pf_attn_gate`, and attention's two GEMMs as ordinary `pf_gemm` launches on the
same list. **Only one term depends on `C`**: QKᵀ is issued per row block of 256
rows so the kernel stops computing the masked half. Everything else is C-free,
which is the return on the runtime-`M` rule, and one binary set serves every
`--prefill-chunk` width.

The `sycl-tla` backend, kept selectable as a control, is 1201 L0 launches, 384
SYCL GEMMs and **656 host waits** per chunk, and its scratch is ten times
larger. Moving the linears onto Level Zero was worth **+6.76%** on prefill
throughput in a same-session controlled comparison, and the whole of it is the
linears: 1668.9 ms of `linear_l0` replaces 1816.5 ms of separate dequant plus
SYCL GEMM, and 512 host waits per run disappear with them. The worry that 8000
or 9000 launches would cost host time does not appear at all. Both backends and
the per-phase table are in [BENCHMARKS.md](BENCHMARKS.md), and
`prefill_backend_equivalence_test` holds them **bitwise** equal.

### Where the time goes

Per-phase shares and the ladder of prefill records are in
[BENCHMARKS.md](BENCHMARKS.md). Two structural facts about the walk belong here
because they are properties of the anatomy rather than of a row:

**The linears are the walk.** `linear_l0` is over 60% of a chunk. Split with one
wait per slab so the dequant round trip could be attributed rather than
estimated, a 4096-token prompt measures **`slab_dequant` 274.5 ms and
`slab_gemm` 1344.3 ms** over 7616 launches each, summing to the unsplit phase
within an instrumented run's spread
([prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md)). Closing a phase
per slab costs 15,232 extra waits, so that instrumented wall is an attribution
and not a throughput figure.

**The delta net scan is the second largest block, and it was the biggest miss in
this project's estimating record.** A first projection put `gdn_chunk` at
15.4 ms per chunk, widening decode's `gdn_step` to `M = 2048`, written before
the chunked algorithm existed. Measured, the first working version was
**984.1 ms per chunk**, a 64× miss, of which the scan alone was 723.4 ms. Two
rewrites took `gdn_chunk` to 347.0 ms, and the split-BF16 DPAS scan took the
scan row from about 297.6 ms to about 127.7 ms profiled ABBA
([BENCHMARKS.md](BENCHMARKS.md)). The lesson is the same one the decode ladder
kept teaching: **a projection that widens a kernel written for one shape to a
shape three orders of magnitude away is not a measurement of anything.**

### Chunk width, fixed and per-position cost

A single-chunk sweep at C = 256 / 512 / 1024 / 2048 measures **411.5 ms fixed
per chunk** and **0.852 ms per position** (derived, two-point fit). At C = 256
the fixed term is 70% of the chunk, which is why prefill throughput is monotone
in the chunk width across the whole available range: 2048 reads 1406.18 t/s
where 1024 reads 1221.90 and 512 reads 932.37 (2026-09-09). **2048 is the widest
width this build runs**, and past it is a smaller `PrefillScratch`, not a flag:
`--prefill-chunk 4096` is refused by the build.

### The bytes, and the card

Weights 15.540 GB (measured, `LoadReport::read_per_token`), plus persistent
1.240 GB, decode scratch 0.069 GB and a `PrefillScratch` of **35,651,584 B** on
the Level Zero backend (measured; the SYCL path needed ten times that) against
the card's 32,656 MB. `PrefillScratch` is allocated lazily on the first
`prefill()`, so a decode-only engine's residency is byte-identical to what it
was before the buffer split, and `prefill_smoke_test` asserts both halves.
</content>
</invoke>
