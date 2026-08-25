# The step, launch by launch - the in-situ anatomy

What one decode token actually spends, measured **inside the replayed command
list**, per launch, at the benchmark shape. This is the document spec 1.5 §3.6
asks for and the one that re-ranks its lever ladder: until now the step was a
partition with one 11.33 ms row nothing had split (docs/12 top table, docs/05
§"Where the 42.14 ms goes"). That row is split here.

## Read this before reading a number below

**Profile mode is not bench mode.** Every launch in a profiled list signals a
host-visible kernel-timestamp event, which carries a flush an unprofiled list
never pays. So:

- **A per-kernel µs here is directly comparable to a probe's per-kernel µs.**
  The duration is `kernelStart → kernelEnd` from `zeEventQueryKernelTimestamp`,
  and the flush the signal carries lands *after* `kernelEnd`. Nothing about the
  kernel's own execution is inflated by being observed. This is what makes the
  in-situ column below comparable to `probe_gemv`'s transplanted floors - the
  first time in this project that those two things can be put side by side.
- **The inter-kernel gap is the opposite: it is an upper bound.** Every µs
  between one `kernelEnd` and the next `kernelStart` in a profiled list contains
  one flush. The gap is quoted twice below, once as measured and once corrected.
- **The fence wall here is not ms/token.** The engine's step time is the
  `--bench` median in [BENCHMARKS.md](BENCHMARKS.md) - 42.141 ms, measured
  2026-08-25, median of three on an idle box. Nothing in this document replaces
  it, and no row here is ever a bench row.

Every number below says which kind it is: **measured** (this instrument or a
named earlier one), **derived** (arithmetic over two measurements), or
**estimated** (a model or an extrapolation).

## How it was measured

```bash
tools/box.sh run "./build/src/cli/b70-decode Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ \
  --profile --depth 4096 --steps 32"
```

Idle box (load average 0.40), 2026-08-25, `CL_DRIVER_VERSION 26.27.39122.14`,
Intel Arc Pro B70, 256 EUs, `--max-len 16384`, `debug_resid` off.

`--profile` ingests the 4096 synthetic ids on an **un-instrumented** list - that
half is an ordinary decode, 170.4 s at 41.59 ms/token - then builds a second,
profiled list against the same `DecodeBuffers` and replays 32 instrumented
steps, resetting all 645 events before each replay and reading every duration
only after the fence.

> **645 is this run's launch count, not today's.** Every table in this section
> and the next four is the step at `43bb720`, which was 645 launches. Lever L1
> (§L1) split each of the 129 `prep_res_norm` sites into two, so the current
> walk is **774** and a run of the same command today prints 774 everywhere 645
> appears below. The counts in §L2 and §L1 say which is which per table.

Positions 4096…4127, so `nb` (live 256-position attention blocks) is 17
throughout - the same regime the recorded bench row measures.

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

Within a run, the per-step spread over the 32 replays is 41522.708 …
41716.458 µs (0.47%). **Run A is quoted throughout below**; anything that turns
on a difference smaller than ~1% is not claimed.

## The anatomy - in situ against the floors

Per kernel family, one decode step at depth 4096. "Probe floor" is the
per-kernel measurement transplanted from `probe_gemv` 2026-08-24 (docs/12,
`gemv`/`gemv_bf16` → Measured); the other families never had one.

| family | launches | **in-situ ms** | probe-floor ms | delta | kind |
|---|---|---|---|---|---|
| `gemv` (int4, five shapes) | 256 | **24.629** | 23.995 | +2.6% | measured, in situ |
| `gemv_bf16` - `lm_head` | 1 | **4.376** | 4.351 | +0.6% | measured, in situ |
| `attn_decode` | 16 | **5.782** | - (2.360 estimated) | **+145%** | measured, in situ |
| `prep_res_norm` | 129 | **2.871** | - | - | measured, in situ |
| `gemv_bf16` - `a‖b` | 48 | **2.335** | - (never probed) | - | measured, in situ |
| `gdn_step` | 48 | **0.733** | - | - | measured, in situ |
| `prep_silu_mul` | 64 | **0.623** | - | - | measured, in situ |
| `prep_gated_head` | 48 | **0.079** | - | - | measured, in situ |
| `attn_reduce` | 16 | **0.076** | - | - | measured, in situ |
| `attn_prep` | 16 | **0.051** | - | - | measured, in situ |
| `embed_gather` | 1 | **0.010** | 0.008 (with argmax) | - | measured, in situ |
| `argmax_stage1` + `stage2` | 2 | **0.007** | ″ | - | measured, in situ |
| **Σ kernel durations** | **645** | **41.571** | | | measured, in situ |

And per compiled variant - the rows `probe_gemv` and docs/12 price one for one:

| shape (variant) | `S` | launches | in-situ µs each | probe µs | delta | in-situ ms | floor ms |
|---|---|---|---|---|---|---|---|
| `gate‖up` 5120×34816 | 4 | 64 | **182.377** | 176.7 | +3.2% | 11.672 | 11.309 |
| `down` 17408×5120 | 16 | 64 | **89.513** | 89.1 | +0.5% | 5.729 | 5.702 |
| `lm_head` bf16 5120×248320 | - | 1 | **4375.557** | 4350.5 | +0.6% | 4.376 | 4.351 |
| `qkv‖z` 5120×16384 | 1 | 48 | **81.401** | 79.7 | +2.1% | 3.907 | 3.826 |
| `out_proj`+`o_proj` 6144×5120 | 16 | 64 | **33.499** | 31.3 | +7.0% | 2.144 | 2.003 |
| `q‖k‖v` 5120×14336 | 1 | 16 | **73.550** | 72.2 | +1.9% | 1.177 | 1.155 |
| **the 257-launch GEMV family** | | **257** | | | **+2.3%** | **29.005** | **28.346** |

**The single most consequential line in this document: the probe transplant was
right.** 28.346 ms was a floor, and in situ the same 257 launches cost 29.005 ms
- **2.3% above it**. docs/05 said the two sides "move in opposite directions"
and asked which; the answer is that GEMV moves by 0.659 ms and the unsplit
bucket therefore shrinks by 0.659 ms. Every shape is within 7% of its probe
number and four of six are within 2.2%, so the stall-from-the-previous-kernel
effect the transplant could not see is real but small. `out/o_proj` (+7.0%) is
the only shape where it is worth naming - it is the shape entered directly from
`attn_reduce`/`prep_gated_head`, the two kernels with the least data in flight.

## The dispatch gap, measured in situ for the first time

| | µs/step | µs/launch | kind |
|---|---|---|---|
| profiled gap (fence wall − Σ durations) | **851.4** | **1.320** | measured - **upper bound** |
| un-instrumented gap | **473** | **0.733** | derived (see below) |
| `probe_replay`'s noop floor × 645 | 335 | 0.52 | estimated (doc 07 #5) |

The **derived** row is the honest in-situ number and it is arithmetic over two
instruments: the bench step is 42.141 ms/token of which 0.097 ms is host outside
the fence (both measured, docs/05), so an un-instrumented fence is 42.044 ms;
Σ of the in-situ kernel durations is 41.571 ms; the difference is 0.473 ms. It
sits exactly where doc 07 #5 predicted it would - between the noop floor
(0.52 µs) and the `ctrl_read` floor (0.63 µs), a shade above both, which is what
645 kernels that each read the control block should cost.

Two consequences.

1. **The instrument's own cost is now measured: 0.379 ms/step, 0.587 µs per
   launch** (derived: profiled wall 42.423 − un-instrumented fence 42.044). A
   profiled step is 0.9% longer than a real one. That is the whole distortion,
   and it lands in the gap, never in a kernel duration.
2. **Kernel *count* remains exonerated, now on a measured basis rather than an
   estimated one.** 0.473 ms is 1.1% of the step. Fusion for launch-count's sake
   is still not a lever (spec 1 §4.1's ≥3 µs rule; this is 0.7 µs). Task 2's
   short-depth run reported a 728.8 µs profiled gap and estimated ~262 µs
   un-instrumented; the depth-4096 figures here (851.4 measured / 473 derived)
   supersede it - same conclusion, better arithmetic.

## The 11.330 ms bucket, attributed

The bucket was defined as the step minus everything that had a number. Here is
what is actually in it, all measured in situ:

| bucket member | launches | in-situ ms | µs/launch | work-groups per launch | effective GB/s |
|---|---|---|---|---|---|
| `prep_res_norm` | 129 | **2.871** | 22.3 | **1** | **17.0** |
| `a‖b` GEMV (`gemv_bf16` 5120×128) | 48 | **2.335** | 48.6 | **2** | **27.0** |
| `gdn_step` | 48 | **0.733** | 15.3 | 192 | **540** |
| `prep_silu_mul` | 64 | **0.623** | 9.7 | 5 | 60.8 |
| `prep_gated_head` | 48 | **0.079** | 1.6 | 48 | 37.6 |
| `attn_reduce` | 16 | **0.076** | 4.7 | 24 | - |
| `attn_prep` | 16 | **0.051** | 3.2 | 28 | - |
| **the 369 launches** | **369** | **6.768** | **18.3** | | |

(GB/s = the per-call byte counts in docs/12's "Traffic per token" tables divided
by the measured µs; **derived**, and for `prep_res_norm` it is mostly L2 traffic
- the 320 KB of partials it folds were written by the GEMV immediately before
it - which makes the number worse, not better: it is not even DRAM-limited.)

The bucket is **6.768 ms, not 11.330**. The missing 4.562 ms was never the
bucket's, and it lands in four places that account for it to within 1 µs:

| where the other 4.562 ms was | ms | kind |
|---|---|---|
| GEMV above its transplanted floor (29.005 − 28.346) | +0.659 | measured |
| `attn_decode` above its estimate (5.782 − 2.314 − 0.046) | +3.422 | measured |
| `embed_gather` + `argmax` above 0.008 | +0.009 | measured |
| the dispatch gap - which the old partition had no row for at all | +0.473 | derived |
| **total** | **4.563** | |

## The step, re-partitioned

Every launch and the host appear exactly once, and the rows sum to the measured
bench step:

| part | launches | ms/token | share | kind |
|---|---|---|---|---|
| GEMV - int4 mixers/MLPs + bf16 `lm_head` | 257 | **29.005** | 68.8% | **measured, in situ** |
| `attn_decode` | 16 | **5.782** | 13.7% | **measured, in situ** |
| `prep` (`res_norm` 129 + `silu_mul` 64 + `gated_head` 48) | 241 | **3.573** | 8.5% | **measured, in situ** |
| `a‖b` GEMV | 48 | **2.335** | 5.5% | **measured, in situ** |
| `gdn_step` | 48 | **0.733** | 1.7% | **measured, in situ** |
| `attn_prep` + `attn_reduce` | 32 | **0.127** | 0.3% | **measured, in situ** |
| `embed_gather` + `argmax` | 3 | **0.017** | 0.04% | **measured, in situ** |
| dispatch gap, un-instrumented | - | **0.473** | 1.1% | derived |
| host, outside the fence | - | **0.097** | 0.2% | measured (bench) |
| **total** | **645** | **42.141** | 100% | measured (bench, median of three) |

> **Superseded as the current step - see §L2 and §L1.** This partition is the
> step at `43bb720`, before spec 1.5 cut its first lever, and it is kept whole
> because every row is one measurement of one run and because it is what the
> 42.141 ms bench row partitions. Two rows have moved since:
> lever L2 took `a‖b` to **0.256 ms/token** and lever L1 took `prep`'s
> `res_norm` share from 2.871 to **0.484**, so the `prep` row reads **1.202 ms**
> and the launch count reads **774**. The step is **38.046 ms**
> (BENCHMARKS.md, recorded median 38.05). Each lever's before/after and its
> arithmetic are §L2 and §L1 below; no other row here has been re-measured
> except as drift.

Compare with the same table in docs/05: five rows that were a floor, an
extrapolation and a remainder are now seven rows that were all measured in one
run. **98.6% of the step is kernel time inside the fence** (41.571 of 42.141);
1.1% is dispatch and 0.2% is the host.

## What the measurement says that nothing else did

### 1. One work-group is worth about 12-17 GB/s, and that is the whole story of the bucket

> **Superseded in part by the L2 measurement below - read §L2 before you use
> this section to price anything.** The three points here are all still
> measured and still true as *totals*. What is false is the per-work-group
> reading of them and the extrapolation that reading licensed: L2 gave the
> `a‖b` GEMV four times the work-groups at the identical subgroup count and it
> bought **nothing** (48.774 → 49.127 µs/launch). The invariant that priced out
> is the **subgroup**, not the work-group. Everything below stands as written -
> it is what this run measured - with that correction attached.

Three independent points, all measured in this run, all from docs/12's own byte
counts:

| kernel | work-groups | GB/s total | GB/s **per work-group** |
|---|---|---|---|
| `prep_res_norm` (SP16) | 1 | 17.0 | **17.0** |
| `a‖b` GEMV | 2 | 27.0 | **13.5** |
| `prep_silu_mul` | 5 | 60.8 | **12.2** |
| `gdn_step` | 192 | 540 | 2.8 (saturated - the device is the limit, not the core) |

A single Xe-core cannot keep enough loads in flight to exceed ~15 GB/s, so
**saturating 590 GB/s needs roughly 40 work-groups** - a shade above the
device's 32 Xe-cores (docs/01), i.e. ~1.25 work-groups per core, which is the
coherent answer for a machine that needs a little oversubscription to hide
latency. Every kernel in the bucket that is slow is slow for exactly one reason:
it was given one or two. This is the mechanism docs/12 wrote down as a risk when
`prep_res_norm` landed ("One work-group per token in `prep_res_norm` is the risk
in this design, and it is unmeasured") - measured, confirmed, and now also
priced for `a‖b`, which had never been timed at all.

**The caveat these three points do not cover, and every yield in the ladder
rests on it.** The table mixes two kinds of traffic: `prep_res_norm`'s 320 KB of
partials and `prep_silu_mul`'s 557 KB were written by the GEMV immediately
before them and are **L2-served**, while `a‖b`'s 1.31 MB of weights are a cold
**DRAM** read. That they land within 12-17 GB/s of each other says the limit is
the core's outstanding-load capacity rather than where the bytes come from -
which is the reading this document takes - but it is three points, all at 1, 2
and 5 work-groups, and **the ladder extrapolates them 4× to 8×** to price L1
(1 → 20 work-groups) and L2 (2 → 8 or 32). Linearity that far is an assumption,
not a measurement: `gdn_step` at 192 work-groups reaches 540 GB/s, so the curve
certainly bends somewhere between 5 and 192, and nothing here locates the knee.
Every ladder yield below is marked *(extrapolated)* for that reason, and Tasks
5-6 measure the real slope with this same instrument.

The hypothesis in docs/12 and docs/05 - 129 launches at "40 µs apiece, 5.2 ms" -
was **half right**: the mechanism is exactly as named, the magnitude is 22.3 µs
and 2.871 ms.

### 2. `attn_decode` is 2.45× its estimate, and what it spends is a serial critical path

5.782 ms in situ against the 2.360 ms docs/12 extrapolated (2.314 estimated
block work + 0.046 measured early-out). The estimate was a line through two
`--bench` points at different depths; its 0.1361 ms/token per block is **35%
below** the 0.208 the first two in-situ points gave.

**Four in-situ points. The third one falsified the model the first two
suggested**, which is why this section reads the way it does.

| run | `nb` | live WGs | positions attended | µs/launch | ms/token |
|---|---|---|---|---|---|
| depth 4096, `--max-len 16384` | 17 | 68 | 4097 | **361.400** | 5.782 |
| depth 1024, `--max-len 16384` | 5 | 20 | 1025 | **338.165** | 5.411 |
| depth 64, `--max-len 16384` | 1 | 4 | 65 | **153.608** | 2.458 |
| depth 64, `--max-len 4096` | 1 | 4 | 65 | **153.665** | 2.459 |

- **The fixed grid is free - re-confirmed per kernel.** Rows 3 and 4 differ by
  192 idle work-groups per layer and by **0.04%**. Doc 07 #12 concluded this
  from a whole-step difference of 0.046 ms; this is the same conclusion measured
  on the kernel itself. Context-bucketed lists remain worthless.

**The two models the `nb` = 1 and `nb` = 17 pair admitted, and their fate.**
Both were written down with their `nb` = 5 predictions *before* the depth-1024
run (the pre-registration is in this task's report):

| model | fitted to nb 1 and 17 | predicted at nb = 5 | measured | verdict |
|---|---|---|---|---|
| **A** linear in blocks + intercept | 140.621 + 12.987·`nb` | 205.556 µs | 338.165 | **falsified**, −39% |
| **B** wave/occupancy, a wave = 32 WGs, flat while 4·`nb` ≤ 32 | 153.608 · max(1, `nb`/8) | 153.608 µs | 338.165 | **falsified**, −55% |

**Neither survives, and the "2.25 ms depth-independent term" model A implied is
withdrawn.** It was an artefact of fitting a line through a point that is not on
the same footing as the others: at `nb` = 1 the single live block holds **65 of
its 256 positions** (25.4% full), so that point measures a quarter-block, not a
cheap full one.

**What fits all four points.** `attn_decode`'s work-groups run concurrently, and
the launch costs what its *slowest single work-group* costs - one 256-position
block walked for all 6 q-heads:

- **Work-group count is nearly free.** `nb` 5 → 17 is **3.40× the live
  work-groups (20 → 68) for +6.9% of time**. Whatever this kernel is limited by,
  it is not how many work-groups are resident.
- **Block *fill* is what moves it.** Writing the cost as `F + fill · P`, where
  `fill` is the valid fraction of the critical-path block, the `nb` = 1 and
  `nb` = 5 points give **F ≈ 90.8 µs of per-launch fixed cost and P ≈ 247.4 µs
  of full-block critical path** (derived, two parameters from two points). The
  `nb` = 17 point is then a *check*, not a fit: predicted 338.2, measured 361.4
  - **−6.4%**, the residual being the mild memory pressure of 3.4× the
  work-groups.
- So **~73% of a launch at depth 4096 is one work-group's serial walk of one
  block** and the rest is fixed cost. The kernel is latency-bound on a critical
  path, not throughput-bound on a grid.
- **It is not bandwidth-bound, and the effective rate rises with depth.**
  10.4 GB/s at `nb` = 1, 74.5 at `nb` = 5, **278.6 at `nb` = 17** (47% of the
  measured 590) counting the 6× q-head reread; the *unique* KV at depth 4096 is
  46 GB/s. docs/12 inferred 740 GB/s from the bench line and called the reread
  "substantially cache-served"; the direction was right (46 GB/s of unique
  traffic cannot be DRAM-limited) but the magnitude was not.

**What this re-aims L5 at.** Not "attack a depth-independent term" - there is no
such term. The target is the **247 µs full-block critical path**, and since
work-group count is measured to be nearly free, the direct attack is a **finer
block**: 128 positions halves the walk and doubles the grid (at depth 4096, 33
blocks × 4 kv-heads = **132 work-groups**, inside the range measured free). The
`F + P/2` prediction is **214.5 µs/launch = 3.43 ms/token, saving ≈2.35 ms**;
64-position blocks predict 2.44 ms but need 264 work-groups, past the point any
measurement here reaches. SLM staging (the plan's L5 design) attacks the same
critical path from the other side - it shortens the 6 q-head passes rather than
the position loop - so the two are alternatives, not complements, and the retile
is by far the cheaper experiment. **Both numbers are extrapolations from a
two-parameter fit with one validation point; neither is a promise.**

### 3. `gdn_step` is exonerated, completely

0.733 ms in situ, against a **0.671 ms** traffic floor (396 MB/token at the
measured 590 GB/s, docs/12's own arithmetic). It runs at **540 GB/s effective -
92% of the device** - on 192 work-groups with no spills. docs/12 said "if the
profile puts it near the bound, the cause is occupancy or register pressure";
the profile puts it **at 1.09× the bound**. The entire kernel, made perfect,
is worth 0.06 ms. The 48×4 grid and the four-fold redundant conv reads were the
right trade and the measurement says so.

### 4. `lm_head` is 10.4% of the step and no lever in this spec touches it

4.376 ms in one launch, at 0.6% above its probe floor and **581 GB/s in situ -
98.5% of the measured 590** (the 97% the probe reports is against the 600 GB/s
theoretical denominator; docs/15 uses the measured one everywhere else). There
is nothing to tune. Quantising it to int4 is worth ~3.3 ms
(4.376 → ~1.1, docs/12 `gemv_bf16` → Measured) and it is item 1 of docs/05's
specialisation list, deliberately outside spec 1.5. It is named here because
the gate arithmetic below cannot be read honestly without it.

## Lever L2 - `a‖b` GEMV, cut and measured

The first lever of spec 1.5, executed 2026-08-25 against this document's own
ranking, with this document's own instrument. **Accepted**: the recorded step is
**42.141 → 40.266 ms/token** (`tools/bench_decode.sh`, median of three, idle
box, sha `a1e2d3a`; BENCHMARKS.md carries the row). The lever's own row moved
**2.341 → 0.256 ms/token**.

### What was tried, in the order it was tried, and what each was worth

Five `--profile --depth 4096 --steps 32` runs, same box, same session, each
building the full engine and replaying 32 instrumented steps. The column that
decides is `gemv_bf16` at 5120×128, 48 launches:

**On the baseline this table starts from.** Row 0 is `48.774 µs / 2.341 ms`,
while the anatomy tables earlier in this document say `48.6 µs / 2.335 ms`.
Those are **two runs of this same instrument at the same sha**, 0.3% apart,
which is its run-to-run spread on this row (§"Reproducibility" measures 0.05%
for the `gemv` family and 2.5% for the smallest row quoted). The L2 arithmetic
below uses the 48.774 / 2.341 pair throughout, because a before/after has to be
two rows of one comparison; the anatomy tables keep 2.335, because that is what
the run they report measured. The two must not be mixed inside one sentence -
48.774 × 48 is 2.341, not 2.335.

| tiling (`COLS_PER_WG`, `KSPLIT`) | work-groups | **subgroups** | µs/launch | ms/token | GB/s | GB/s per subgroup |
|---|---|---|---|---|---|---|
| `{64, 1}` - as plan 3 shipped it | 2 | 8 | **48.774** | **2.341** | 26.9 | 3.36 |
| `{16, 1}` - the ladder's option (a) | 8 | 8 | **49.127** | 2.358 | 26.7 | 3.34 |
| `{16, 1}` + 4 block reads in flight | 8 | 8 | **49.052** | 2.354 | 26.7 | 3.34 |
| `{16, 4}` - the ladder's option (b) | 8 | 32 | **13.115** | 0.630 | 99.9 | 3.12 |
| **`{16, 16}` - shipped** | **8** | **128** | **5.340** | **0.256** | **245** | 1.92 |

All five are measured, in situ, per launch. The last row is re-measured at the
committed sha and reads **5.342 µs / 256.403 µs per step** - 0.04% from the
tuning run, which is what this instrument's reproducibility looks like.

### §1's mechanism was the wrong invariant, and the lever is what proved it

The ladder priced L2 off §1: two work-groups at 13.5 GB/s each, so four times
the work-groups is four times the bandwidth. **Option (a) is exactly that
experiment and it returned nothing** - 48.774 → 49.127 µs, a 0.7% *regression*,
where the model said ~12 µs. A bit-identical variant that also gave each
subgroup four outstanding block reads (a `KU` knob, since deleted) returned
nothing either, 49.052 µs, and cost `lm_head` 1.3% by perturbing its codegen.

What the three failed/flat points share is that **the launch still had eight
subgroups**. `COLS_PER_WG` re-spreads the same eight over more work-groups;
loads-in-flight changes what one of them does. Only splitting K makes more of
them, and when it does the time falls almost in proportion: 8 → 32 → 128
subgroups reads 48.774 → 13.115 → 5.340 µs, i.e. **3.36 → 3.12 → 1.92 GB/s per
subgroup**. A subgroup pulls what it pulls; the launch was slow because it only
ever had eight.

This does not overturn §1's three measurements - they are totals and they are
still what they were. It overturns the *per-work-group* reading of them. Note
that the two are not distinguishable in §1's own table: `prep_res_norm` (1
work-group) has 16 subgroups, `prep_silu_mul` (5) has 80, `a‖b` (2) had 8, and
per **subgroup** those read 1.06 / 0.76 / 3.36 GB/s - a spread §1 could not see
because it never divided by the right thing. The `a‖b` figure is 3-4× the others
because its subgroup pulls 256 B per load (`intel_sub_group_block_read_us8`)
where prep's pulls 64 B, which is the shape of a **latency-bound** kernel with
one load in flight per thread: rate = bytes per load ÷ ~60-75 ns.

**What this changes for L1, which is next.** L1's yield does not collapse - a
two-stage `prep_res_norm` multiplies work-groups *and* subgroups together (20
work-groups × 16 subgroups = 320, against 16) - but its **basis** changes, and
so does the thing to measure first. §1's ×8 arithmetic is not evidence any more;
the evidence is this section's three-point subgroup curve, and it is
sub-linear at the top (128 subgroups gave 1.92 GB/s each where 8 gave 3.36).
Whoever cuts L1 should read the curve, not §1's table, and should expect the
same surprise budget: **two of the three things tried here were worth zero, and
they were the two the ranking called certain.**

### Attribution: the whole step, before and after

Both from `--profile --depth 4096 --steps 32`, before at `43bb720` and after at
`a1e2d3a`:

| | before | after | delta | kind |
|---|---|---|---|---|
| `gemv_bf16_M1_K5120_N128…` | **2341.136 µs** | **256.403 µs** | **−2084.7** | measured, in situ |
| `gemv_bf16` - `lm_head` | 4378.555 | 4379.089 | +0.5 (0.01%) | measured, in situ |
| Σ of 645 kernel durations | 41597.288 | 39631.634 | −1965.7 | measured, in situ |
| fence wall (profiled) | 42453.888 | 40515.713 | −1938.2 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **42.141 ms** | **40.266 ms** | **−1.875 ms** | **measured (bench)** |

**The three deltas do not agree exactly, and the difference is the honest error
bar of this comparison, not a missing effect.** The lever's own row falls
2.085 ms; Σ falls 1.966 ms because the other 597 launches read **+119 µs
(+0.30%)** higher in the after run than the before run - spread over every
family (`attn_decode` +42 µs, `prep_res_norm` +21, `gate‖up` +18), which is
run-to-run drift of the same kind §"Reproducibility" measures at 0.006% for two
runs three minutes apart and is here an order of magnitude larger after a day of
work on the box. **The bench row is the number that counts** (−1.875 ms, median
of three, 0.04% spread) and it is 0.09 ms below Σ's delta, which is the same
drift seen from the other side. Nothing in the lever is claimed at better than
±0.1 ms.

`lm_head` is the control: same binding, same `{64, 1}` variant, and it did not
move (+0.01%). The launch count is 645 before and after.

### What is left in this kernel

1.31 MB at the measured 590 GB/s is **2.22 µs**, and the shipped tiling is
5.34 µs - 2.4×. So the whole remaining prize is **0.15 ms/token**, and a 32-way
split (which would need its own golden-gate run, because it reorders the sum
again) cannot repay it. `a‖b` is now the step's **eleventh** largest variant row
at 0.65% of Σ, below `prep_silu_mul`. It is finished.

## Lever L1 - `prep_res_norm`, cut and measured

The second lever of spec 1.5, executed 2026-08-25 against this document's own
ranking and - deliberately - against §L2's correction to it rather than §1's
falsified per-work-group model. **Accepted**: the recorded step is
**40.266 → 38.046 ms/token** (`tools/bench_decode.sh`, median of three, idle
box, sha `b045e11`; BENCHMARKS.md carries the row, whose recorded median is
38.05). The lever's own row moved **2.893 → 0.484 ms/token**.

### The design, and what each half of it was worth

RMSNorm's mean is over the whole row, so one kernel's reduction domain is one
work-group and no tiling *inside* a launch can change it. The split is the
fallback docs/12 wrote down when the kernel landed and never built:

```
stage A  prep_res_fold(partials, resid, sumsq)      grid (20, M), WG 256
stage B  prep_norm_finish(sumsq, resid, norm_w, x)  grid (20, M), WG 256
```

`G = 20` at K = 5120 is **one element per lane**, so the fold runs on 20 × 16 =
**320 subgroups against 16**. Three profile runs, `--profile --depth 4096
--steps 32`, same box, same session, before at `b15f70f` (L2's binary):

| shape | stage A µs/launch | stage B µs/launch | the site, µs/step | vs baseline |
|---|---|---|---|---|
| `prep_res_norm` - one work-group, as plan 3 shipped it | - | - | **2893.180** | - |
| two-stage, stage B on **1** work-group (the plan's literal design) | 1.973 | **9.194** | **1440.586** | −1452.594 |
| **two-stage, stage B on 20 (shipped)** | **2.012** | **1.739** | **483.886** | **−2409.294** |

All measured, in situ, 129 launches of each stage per step.

> **Disclosure on the middle row.** It was produced by an *uncommitted* one-line
> edit to `src/runtime/capture.cc` binding `prep_norm_finish_..._W1` on a grid
> of 1, reverted immediately after the run. **No commit in this history
> reproduces it**, so it cannot be re-derived by checking out a sha - the `W=1`
> *binary* is still built and still unit-tested, but its capture binding is not.
> Re-measuring it means re-applying that one line. Row 1 and row 3 are both at
> committed shas (`b15f70f`, `b045e11`).

**The fold is 60% of the lever and the rescale is the other 40%.** The plan
wrote stage B as a single work-group because the *reduction* is what it was
splitting; the measurement says the rescale pass - 10 KB of `resid` + 20 KB of
`norm_w` + 10 KB of `x` through one Xe-core - is the same latency-bound walk and
wanted the same grid. At `W = 1` stage B runs at **4.5 GB/s**, worse per byte
than the single-work-group kernel it came from, because its head is a 20-deep
chain of `sumsq` loads that nothing overlaps when only one work-group is live.

### §L2's subgroup curve held, and this is its fourth point

Stage A moves 348,240 B in 2.012 µs - **173 GB/s**, against the single-work-group
kernel's 17.0. That is **20× the subgroups buying 10.2×**, the same sub-linear
shape §L2 measured on `a‖b` (16× the subgroups bought 9.1×). Per subgroup:

| launch | subgroups | GB/s | GB/s per subgroup |
|---|---|---|---|
| `prep_res_norm` | 16 | 17.0 | 1.06 |
| `prep_res_fold` | 320 | 173 | 0.54 |
| `a‖b` `{16,1}` (§L2) | 8 | 26.9 | 3.36 |
| `a‖b` `{16,16}` (§L2) | 128 | 245 | 1.92 |

Both kernels halve their per-subgroup rate over a 16-20× multiplication, and
both are still far from the device. **§L2's warning that "two of the three
things tried were worth zero" did not repeat here** - the one thing the ranking
called certain was certain, and the surprise was in the *other* direction: the
half of the design the plan treated as a formality (stage B's grid) was worth
0.96 ms.

### Attribution: the whole step, before and after

Both from `--profile --depth 4096 --steps 32`, before at `b15f70f` and after at
`b045e11`:

| | before | after | delta | kind |
|---|---|---|---|---|
| `prep_res_norm` (129 launches) | **2893.180 µs** | - | | measured, in situ |
| `prep_res_fold` (129) | - | **259.495 µs** | | measured, in situ |
| `prep_norm_finish` (129) | - | **224.391 µs** | | measured, in situ |
| **the site** | **2893.180** | **483.886** | **−2409.3** | measured, in situ |
| `gemv_bf16` - `lm_head` | 4378.942 | 4380.589 | +1.6 (0.04%) | measured, in situ |
| `a‖b` GEMV | 256.624 | 265.000 | +8.4 (3.3%) | measured, in situ |
| the 516 untouched launches, Σ | 36728.369 | 36910.450 | **+182.1 (+0.50%)** | measured, in situ |
| Σ of all kernel durations | 39621.549 (645) | 37394.336 (774) | −2227.2 | measured, in situ |
| fence wall (profiled) | 40496.734 | 38400.907 | −2095.8 | measured, in situ |
| dispatch gap (profiled, upper bound) | 875.185 / 1.357 µs per launch | 1006.571 / 1.300 | +131.4 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **40.266 ms** | **38.046 ms** | **−2.220 ms** | **measured (bench)** |

**The three deltas do not agree exactly, and the arithmetic that closes them is
this:** the lever's own row falls 2.409 ms; +0.182 ms comes back as run-to-run
drift on the 516 untouched launches (the same effect §L2 measured at +0.30%,
here +0.50%, spread over every family - `attn_decode` +89 µs, `gemv` +56,
`prep_gated_head` +6); and +0.095 ms is **derived** for the 129 extra dispatches
at the un-instrumented 0.733 µs/launch this document measures. −2.409 + 0.182 +
0.095 = **−2.133 ms predicted** (summed unrounded) against **−2.220 measured**
(40.266 − 38.046), **0.087 ms apart** -
inside the ±0.1 ms nothing here is claimed better than.

`lm_head` is the control: untouched binding, untouched binary, +0.04%.

### The launch count moved, on purpose, and it was cheap

**645 → 774.** Each of the 129 sites is two launches now. At the derived
0.733 µs/launch that is 0.095 ms/token bought for 2.409 ms saved - a
**25.5:1** trade (2.409294 / 0.094557) - and it is the measured answer to doc 04's long-standing worry that
kernel count is a first-class cost. It is not: the step's dispatch total is
~0.57 ms derived (774 × 0.733), 1.5% of 38.05.

### What is left in this pair

50.2 MB/token of traffic is 0.085 ms at the measured 590 GB/s and the pair costs
0.484 - **5.7× its floor**, against the single-work-group kernel's 34.9×. The
remaining prize is **~0.40 ms** and it is in two places: stage B's 20-deep
`sumsq` load chain (a lane-parallel load plus an SLM tree would shorten it) and
stage B's re-read of `resid`. Neither can repay its own golden-gate run, and L5
is 5.9 ms.

## The ranked lever ladder

Ranked by **measured in-situ share**, which is what this document exists to
produce. Yields are estimated, and each says on what basis.

| rank | lever | measured | share of Σ | expected yield | basis |
|---|---|---|---|---|---|
| 1 | **L5** attention (`attn_decode`) | **5.782 ms** (5.922 re-measured at `b045e11`) | 13.9% | **1.5-2.5 ms** (extrapolated) | four-point in-situ fit (§2): ~247 µs of the 361 is ONE work-group's serial walk of a 256-position block, and work-group count is measured nearly free (3.4× for +6.9%). A 128-position retile predicts 3.43 ms/token. Extrapolated from two fitted parameters with one validation point |
| 2 | ~~**L1** `prep_res_norm` two-stage~~ **- CUT, §L1** | **2.871 ms** | 6.9% | **−2.409 ms measured** (predicted 1.5-2.4) | **done**. The prediction landed at the top of its band and the *mechanism* landed too: 16 → 320 subgroups took the fold from 17.0 to 173 GB/s, exactly the sub-linear shape §L2's curve implied. The surprise was elsewhere - stage B's grid, which the plan wrote as one work-group, was worth 0.96 ms of the 2.41. Row is now 0.484 ms and 5.7× from its traffic floor |
| 3 | ~~**L2** `a‖b` GEMV occupancy~~ **- CUT, §L2** | **2.335 ms** | 5.6% | **−2.085 ms measured** (predicted 1.2-1.9) | **done**. The prediction landed, the *mechanism* did not: `COLS_PER_WG` 16 was worth **zero** and what paid was a 16-way K split inside the work-group (8 → 128 subgroups). Row is now 0.256 ms and 0.15 ms from its traffic floor |
| 4 | **L3** `gdn_step` retile | **0.733 ms** | 1.8% | **≤0.06 ms** | **candidate for skip-by-ruling** - 1.09× its own 0.671 ms traffic floor, 92% of device bandwidth; the lever cannot repay a day of work at any outcome |
| 5 | **L4** GEMV `S` retune in situ | **0.659 ms** of excess | 1.6% | **≤0.3 ms** | **candidate for skip-by-ruling** - the whole in-situ excess over the probe floor is 0.659 ms across six shapes, the `S` picks are already the probe's own best, and each candidate costs a full golden-gate run because `S` reorders the split-K merge |

### The order to execute, and why it is not the share order

**L2 → L1 → L5**, then L4 only if the gate is within reach, and L3 not at all.
*(L2 and L1 are both executed and both accepted; L5 is next and is now the
largest non-GEMV row in the step by a wide margin: measured at `b045e11`,
`attn_decode` 5.922 ms against the whole `prep` family's 1.202 (**4.9×**) and
`gdn_step`'s 0.748 (**7.9×**). Only the GEMV rows are bigger, and no lever in
this spec touches them.)*

Share ranks L5 first, and it is the biggest number on the page. It is
nevertheless third to execute, for reasons the measurement itself supplies:

- **L2 is half a day and its option (a) is bit-identical** (same per-column
  arithmetic order, more work-groups). §1 makes its yield close to arithmetic.
  It is the cheapest ms on the ladder and it de-risks nothing else.
  *(Executed. It was half a day; option (a) was bit-identical and worth
  nothing; the yield landed at the top of the predicted band by a different
  mechanism, and it did NOT de-risk nothing else - it falsified the model L1's
  yield was resting on. §L2.)*
- **L1 is a day, its mechanism is measured and its design was written when the
  kernel landed.** Its only correctness exposure is the global Σx² tree, which
  the golden gate arbitrates.
  *(Executed. It was a day; the mechanism held and the yield landed at the top
  of the band at −2.409 ms in situ, −2.220 ms on the bench. The gate arbitrated
  the Σ tree and passed 96/96 - but its per-layer **diagnostics** moved in both
  directions, and re-running the gate before as well as after is what made that
  readable; docs/14 now carries both columns. §L1.)*
- **L5 is the only sketch-level design in the plan** (the plan says so) and it
  carries the ladder's highest correctness risk (online-softmax staging). Its
  target is now identified - the 247 µs full-block critical path of §2, not the
  "depth-independent term" the two-point fit suggested and the third point
  killed - but the yield is still an extrapolation, and the mechanism the plan
  wrote L5 around (SLM staging) is probably the *wrong* half of it: a
  **128-position block retile** attacks the same critical path for a fraction of
  the effort and no change to the softmax arithmetic. Whoever runs L5 should
  measure the retile first. Running it after two banked, certain wins means it
  is attempted with the gate arithmetic already known.

### The gate arithmetic, stated in advance

Spec §2's success bar is 31.746 ms/token; the step is 42.141. **The gap is
10.395 ms.** Summing the optimistic end of every lever above:

| | ms | |
|---|---|---|
| L2 `a‖b` | −1.9 | **actual −1.875 (bench), banked** |
| L1 `prep_res_norm` | −2.4 | **actual −2.220 (bench), banked** |
| L5 attention | −3.0 | |
| L4 GEMV `S` | −0.3 | |
| L3 `gdn_step` | −0.06 | |
| **the whole ladder, optimistically** | **−7.7** | |

That lands at **34.4 ms/token ≈ 29.0 t/s** - short of 31.50 by about 2.7 ms.

**Two levers in, the arithmetic is unchanged and both estimates held.** L2 paid
−1.875 ms against −1.9; L1 paid −2.220 against −2.4. The step is now
**38.046 ms** (measured, BENCHMARKS.md, recorded median 38.05) and **the gap to
the 31.746 ms bar is 6.300 ms**; the three remaining levers are priced at −3.36
optimistically, so the ladder now lands at **34.7 ms/token ≈ 28.8 t/s** - within
0.3 ms of where this section put it before a single lever was cut, which is the
best that can be said for a set of estimates.

**What the two executions changed is where the remaining risk sits.** Both
predicted yields landed; both landed for reasons the ranking got partly wrong
(L2: the work-group count was worth zero and the K-split was everything; L1: the
fold was priced and the rescale, treated as a formality, was 40% of the win). L5
is the only lever left with a yield large enough to matter and it is the one
whose design is still a sketch. **The ladder as specified is still
arithmetically unlikely to clear the gate** - §6's short path still writes the
memo - but the shortfall is now 2.9 ms rather than 2.7, on two fewer unknowns.
This document said the ladder was arithmetically unlikely to clear the gate
before the first lever was cut rather than after the last, and two levers in it
still is. The spec anticipated exactly this ("the win is *arithmetically*
reachable but tight") and provided for it: §6's short path writes the
re-assessment memo.

What the measurement adds to that memo, in advance and in order of size:

1. **`lm_head` at int4: ~3.3 ms**, no kernel work, docs/05 specialisation 1.
   Alone it converts the projected 34.7 into 31.4 ms/token - over the bar.
   It is the single largest measured item in the step that spec 1.5 does not
   touch.
2. **`attn_decode`'s 247 µs full-block critical path** - a 128-position retile
   (at depth 4096: 33 blocks × 4 kv-heads = **132 work-groups**, against 68
   today) is a grid change of exactly the kind L3 was scoped for, predicted at
   ≈2.35 ms, and this document says L3's budget went to the wrong kernel. It is
   the cheapest untried thing in the step.
3. **`a‖b` → `gdn_step` prologue fusion** and **norm → GEMV prologue fusion**
   remain redesigns, but §1 now prices what they are really buying: not the
   0.7 µs launch, the work-group count.

## What this document does not settle

- **`attn_decode`'s internal split.** 5.782 ms is measured at four points; the
  `F` ≈ 90.8 / `P` ≈ 247.4 µs decomposition is a **two-parameter fit to two of
  them with the third as a −6.4% check**, and the fourth (the `--max-len` pair)
  only rules out the grid. Two models that fitted the first two points were
  falsified by the third - the same could happen again, and the retile yield
  above is the prediction that would go with it. What is *measured* and not
  fitted: work-group count is nearly free over 20 → 68, and depth 64 → 1024
  costs +120% while 1024 → 4096 costs +6.9%.
- **Every *remaining* yield in the ladder is an estimate.** §1's per-work-group
  ceiling was measured at 1, 2 and 5 work-groups and the extrapolation off it was
  falsified by L2; the subgroup curve that replaced it has since been measured at
  8, 16, 32, 128 and 320 subgroups across two kernels (§L2, §L1) and it has held
  every time, sub-linearly. What remains unmeasured is L5, whose yield rests on a
  two-parameter fit, not on that curve at all.
- **M > 1.** Everything here is the M = 1 decode step. The bucket's kernels are
  the ones whose costs move most with M, and none of that is measured.
