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
- **A mean is not an attribution.** Every single-run number below carries this
  instrument's drift and nothing says how much. `--profile --repeats R` prices
  that directly, and the floor it measures - **0.051 ms at the step, 0.2-15 µs
  per family, across processes** - is in "Spec 1.6 §5.4" near the end of this
  document. Read it before believing any delta under ~0.3 ms, including several
  in the lever sections that follow.

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
> 42.141 ms bench row partitions. Three rows have moved since:
> lever L2 took `a‖b` to **0.256 ms/token**, lever L1 took `prep`'s
> `res_norm` share from 2.871 to **0.484** (so the `prep` row reads **1.202 ms**
> and the launch count reads **774**), and lever L5 took the attention family
> from 6.058 to **3.839** (`attn_decode` 5.920 → **3.585**) with the launch
> count unchanged. The step is **36.32 ms** (BENCHMARKS.md, recorded median
> 27.53 t/s at `c746840`, re-measured **27.54** at the spec 1.5 gate `ef6acb0` -
> the same engine twice, 0.04% apart; both print 36.32 ms/token). Each lever's
> before/after and its arithmetic are §L2, §L1 and §L5 below; no other row here
> has been re-measured except as drift.

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

> **FALSIFIED, 2026-08-25 - both predictions in this paragraph, and the second
> one badly.** The retile was executed (§L5) and is the cleanest test this fit
> could get, since it changes the block length and nothing else.
>
> | this paragraph's prediction | measured | miss |
> |---|---|---|
> | 128-position block: **214.5 µs/launch, 3.43 ms/token** | **296.684 µs, 4.747 ms/token** | **+38.3%** |
> | 64-position block: **2.44 ms/token** | **3.585 ms/token** (224.046 µs/launch) | **+46.9%** |
> | "264 work-groups, past the point any measurement here reaches" | **260 work-groups, measured, and free** | reached |
>
> The *direction* held - a finer block is the right attack and it paid - but
> every number here is wrong and the model behind them (§2's `F + fill·P`) is
> dead. So is the refit that replaced it. §L5 carries the sweep that chose 64
> by measurement instead. The sentence about SLM staging also stands
> **corrected**: staging and the retile are indeed alternatives, and the sweep
> now bounds what staging could ever be worth at a fraction of a fraction of the
> row (docs/12, `attn` → Rejected).

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

## Lever L5 - attention `ATTN_BLOCK`, cut and measured

`attn_decode` was the largest non-GEMV row in the step and §2 above says what it
spends: not occupancy - work-group count is measured nearly free - but **one
work-group's serial walk of a full KV block**. The lever is therefore one
number, `ATTN_BLOCK`, and this section is mostly about how badly §2's model
predicted what changing it would do.

### What was tried, in the order it was tried, and what each was worth

Four block sizes, all `--profile --depth 4096 --steps 32`, same idle box, same
session. The plan (and §2) scoped this lever as **256 → 128**; 128 was run
first, and its result is why the other two were run at all.

| `ATTN_BLOCK` | live WGs @ 4096 | `attn_decode` µs/launch | `attn_reduce` µs/step | **attn family ms/token** | Σ 774 µs | verdict |
|---|---|---|---|---|---|---|
| **256** - before | 68 | 369.988 | 80.241 | **6.058** | 37387.510 | the baseline |
| 128 - the planned retile | 132 | 296.684 | 127.240 | **4.931** | 36380.755 | works, and falsifies §2's fit by +38% |
| **64 - shipped** | 260 | **224.046** | 196.195 | **3.839** | **35342.441** | shipped: the last halving that pays |
| 32 | 516 | 203.837 | 450.208 | 3.769 | 35361.950 | family 0.070 ms better; not worth its footprint |

**Why the sweep stopped at 64 - stated carefully, because the first version of
this section overclaimed it.** It said "64 is the knee, measured on both sides",
resting on the step's Σ coming out 19.5 µs higher at 32. **That does not hold:**
the non-attn launches drifted **+0.283%** (+89.3 µs) between those two runs, so
the Σ sign is drift, and at family level B32 is **0.070 ms/token BETTER** than
B64 (3.769 vs 3.839), not worse. The claim is withdrawn. What actually decides
it is three things that do not depend on a sign:

1. **The marginal gain has collapsed.** The three halvings bought **−1.127,
   −1.092 and −0.070 ms/token**. The third is 6% of the second and about a third
   of one run's drift on the untouched launches - i.e. it is at the edge of what
   this instrument can even see.
2. **`attn_reduce` is on a steep ramp** - 80.2 → 127.2 → 196.2 → **450.2**
   µs/step - and it is what cancels the rest: at 32, `attn_decode` gives up
   323 µs/step and `attn_reduce` takes back 254 of them. One more halving would
   plainly cross over.
3. **`attn_part` would double again, 50.7 → 101.4 MB**, per-step scratch 77.6 →
   128.3 MB. Doubling the largest scratch buffer in the step for 0.070 ms/token
   - under the drift - is not a trade worth making.

**The B32 transcript**, `--profile --depth 4096 --steps 32` on the idle box (a
second independent run made specifically to record this row; the first run's
family total was 3772.280 µs, 0.08% away, so it reproduces):

```
  attn_decode                               16   3261.393   9.23%     203.837
  attn_reduce                               16    450.208   1.27%      28.138
  attn_prep                                 16     57.617   0.16%       3.601
  attn_decode_M1_L16384_B32                 16   3261.393   9.23%     203.837
  attn_reduce_M1_L16384_B32                 16    450.208   1.27%      28.138
  sum of 774 kernel durations    35361.950 us   [35225.417 .. 35492.292]
  fence wall (submit + wait)     36379.774 us   [36246.242 .. 36542.172]
```

### §2's model was falsified, and so was the one that replaced it

§2 fitted the four in-situ *depth* points as `F + fill · P` - F ≈ 90.8 µs of
per-launch fixed cost, P ≈ 247.4 µs for a full 256-position block - and
predicted **214.5 µs/launch** at a 128-position block, "3.43 ms/token, saving
≈2.35 ms", marked *(extrapolated)*.

The retile is the cleanest test that model could get: it changes the block
length and nothing else. It measured **296.684 µs**, **+38.3%**.

| model | fitted to | predicted | measured | verdict |
|---|---|---|---|---|
| **C** §2's `F + fill·P` (F 90.8, P 247.4), from the `nb` 1 and 5 depth points | B128: 214.5 µs | | **296.684** | **falsified, +38.3%** |
| **D** refit `F + walk` on the B256/B128 pair (F 223.4, walk 146.6 per 256 positions) | B64: 260.0 µs | | **224.046** | **falsified, −13.8%** |

Model D's B64 prediction was written down before the run, as §2's two were.
**That is four cost models this kernel has now killed** (§2's A and B, then C
and D), and the discipline that catches them is the same every time: state the
prediction, then measure the point that would break it.

**What is measured, and what this document will not do with it.** Over 256 →
128 → 64 each halving costs **≈ 73 µs less per launch** (73.304, then 72.638) -
linear in `log2(ATTN_BLOCK)`, not in the block - and the B32 point breaks even
that (−20.2, not −73). A fifth model would fit three of the four points and this
document declines to write it. What it will say:

- **Work-group count is free a great deal further than anything here had
  measured.** 68 → 260 live work-groups while the launch got **39% faster**.
  §2 put 264 work-groups "past the point any measurement here reaches"; it is
  reached.
- **The launch is now 224 µs, and model D called ~223 µs of the *original* 370
  "not the block walk".** Whatever dominates `attn_decode` today is a term no
  model in this document has named, and it is now essentially the whole kernel.
  That is the finding, and it is the one the next person needs.

### Attribution: the whole step, before and after

Before at `0bfa891`, after at `c746840`, both `--profile --depth 4096
--steps 32` on the idle box.

| | before | after | delta | kind |
|---|---|---|---|---|
| `attn_decode` (16 launches) | **5919.814 µs** (369.988/launch) | **3584.736** (224.046/launch) | **−2335.1** | measured, in situ |
| `attn_reduce` (16) | 80.241 (5.015/launch) | 196.195 (12.262/launch) | +116.0 | measured, in situ |
| `attn_prep` (16) | 58.057 | 58.070 | +0.0 | measured, in situ |
| **the site (attn family)** | **6058.112** | **3839.001** | **−2219.1** | measured, in situ |
| `gemv_bf16` - `lm_head` | 4379.899 | 4388.493 | +8.6 (0.20%) | measured, in situ |
| `gdn_step` | 748.659 | 761.706 | +13.0 (1.74%) | measured, in situ |
| the 726 untouched launches, Σ | 31329.398 | 31503.440 | **+174.0 (+0.56%)** | measured, in situ |
| Σ of all kernel durations (774 both) | 37387.510 | 35342.441 | −2045.1 | measured, in situ |
| fence wall (profiled) | 38413.428 | 36357.567 | −2055.9 | measured, in situ |
| dispatch gap (profiled, upper bound) | 1025.919 / 1.325 µs per launch | 1005.914 / 1.300 | −20.0 | measured, in situ |
| **recorded step (`--bench`, median of 3)** | **38.046 ms** | **36.32 ms** | **−1.73 ms** | **measured (bench)** |

−2219.1 + 174.0 = **−2045.1**, the Σ delta exactly. `lm_head` is the control:
untouched binding, untouched binary, +0.20%.

**The launch count did NOT move - 774 before and after.** Unlike L1 there is no
dispatch term to add, which makes this lever's arithmetic simpler and its
residual harder to excuse.

**And there is a residual.** The in-situ attribution predicts **−2.045 ms** on
the bench; the bench measured **−1.73**. That is **0.32 ms apart**, against
0.087 for L1 and the ±0.1 ms this instrument has been claimed at. Neither the
launch count (unchanged) nor the drift (+0.174, already counted) explains it.
The one *named* contributor is too small by an order of magnitude: `--bench`
sweeps `pos` 4096 → 4351, so `nb` runs 65 → 69 at `ATTN_BLOCK` 64 where it was a
flat 17 at 256, and the after-run therefore averages ~3% more merge work per
token than the single profile point that priced it - ~0.01 ms by `attn_reduce`'s
own slope. **The remaining ~0.31 ms is unattributed.** It is recorded here as
unattributed, because this document's whole method is that a number nobody can
account for is a finding and not a rounding error.

### What is left in this kernel, and what L5 did NOT do

`attn_decode` is still **3.585 ms/token, 10.1% of the step** and the largest
non-GEMV row by a wide margin - `gdn_step` is 0.762. The plan's own L5 design,
**SLM staging of the KV tiles, was never built**, and the measurement argues
against it rather than deferring it.

**The argument deliberately uses no cost model**, since this section has just
finished killing two: a chain through model D would be a chain through something
declared falsified three paragraphs ago. Take the bound straight off the sweep
instead. Halving the block halves the positions one work-group walks serially,
so a halving buys back **at most** whatever walk is still there. The **second**
halving (64 → 32) bought only **20.2 µs/launch** - which bounds everything still
walk-shaped in a launch at roughly **0.3-0.6 ms/token** across the 16 layers.
SLM staging attacks a *subset* of that: the 6× q-head reread *inside* the walk,
not the walk itself. Its entire ceiling is a fraction of a fraction of a
3.585 ms row, for six accumulators and six `(mx, sm)` pairs per work-item and
the project's highest-risk correctness change. That is a clear no, and it stays
a clear no however the ~200 µs that is *not* walk-shaped turns out to be
explained.

**The next thing to do on this kernel is a probe that names the 224 µs**, not
another design against a model. `attn_reduce`, meanwhile, has gone 0.076 →
0.196 ms and is the term that closed the sweep - it is now 5.1% of the family
and it is the reason 32 is not better than 64.

### The gate did not move, and that is a result about the gate

96/96 element-exact before and after, and **every diagnostic cosine identical to
all nine printed decimals** on all three prompts - the first lever in this spec
where they did not move at all. docs/14 records both columns and the reason:
the gate's longest prompt reaches `pos` **92**, which at `ATTN_BLOCK` 64 is two
blocks against 256's one. It covers the reassociation barely, and the 65-block
merge that runs at depth 4096 not at all. That depth is held only by
`attn_test`'s `pos = 4095` (64-block merge) and L16384 `pos = 16383` (256-block
merge) cases, against a host reference that models the same blocking - where the
worst gated distance measured is **1 bf16 ulp on one word**, against a ruled bar
of 2 (docs/12, `attn` → What the test asserts).

**Ruling, for the gate task and for spec 1.6.** For spec 1.5, `attn_test`'s ulp
bars **are** the deep-depth authority on attention arithmetic; the golden gate
is not, and no lever should claim it is. A **deep-context golden prompt** - one
long enough to put the gate past `pos` 4096, where `nb` is 65 rather than 2 - is
a **memo item for spec 1.6**, not a spec 1.5 deliverable: it needs a new oracle
run, and this spec's gate budget is spent.

## The ranked lever ladder

Ranked by **measured in-situ share**, which is what this document exists to
produce. Yields are estimated, and each says on what basis.

| rank | lever | measured | share of Σ | expected yield | basis |
|---|---|---|---|---|---|
| 1 | ~~**L5** attention (`attn_decode`)~~ **- CUT, §L5** | **5.782 ms** (5.920 re-measured at `0bfa891`) | 13.9% | **−2.219 ms measured** on the attn family, −1.73 ms on the bench (predicted 1.5-2.5) | **done**. The prediction landed inside its band; the *model* behind it did not. §2's `F + fill·P` fit predicted 214.5 µs/launch at a 128-position block and measured 296.684 (+38%), and the refit that replaced it missed B64 by −13.8% the other way. `ATTN_BLOCK` was swept 256/128/64/32 instead of fitted and **shipped at 64** - not on a knee (B32's family is 0.070 ms/token *better*) but because the marginal gain collapses (−1.127 → −1.092 → −0.070 ms) while `attn_part` would double to 101.4 MB. Row is now 3.585 ms and still the largest non-GEMV row in the step |
| 2 | ~~**L1** `prep_res_norm` two-stage~~ **- CUT, §L1** | **2.871 ms** | 6.9% | **−2.409 ms measured** (predicted 1.5-2.4) | **done**. The prediction landed at the top of its band and the *mechanism* landed too: 16 → 320 subgroups took the fold from 17.0 to 173 GB/s, exactly the sub-linear shape §L2's curve implied. The surprise was elsewhere - stage B's grid, which the plan wrote as one work-group, was worth 0.96 ms of the 2.41. Row is now 0.484 ms and 5.7× from its traffic floor |
| 3 | ~~**L2** `a‖b` GEMV occupancy~~ **- CUT, §L2** | **2.335 ms** | 5.6% | **−2.085 ms measured** (predicted 1.2-1.9) | **done**. The prediction landed, the *mechanism* did not: `COLS_PER_WG` 16 was worth **zero** and what paid was a 16-way K split inside the work-group (8 → 128 subgroups). Row is now 0.256 ms and 0.15 ms from its traffic floor |
| 4 | **L3** `gdn_step` retile | **0.733 ms** | 1.8% | **≤0.06 ms** | **candidate for skip-by-ruling** - 1.09× its own 0.671 ms traffic floor, 92% of device bandwidth; the lever cannot repay a day of work at any outcome |
| 5 | **L4** GEMV `S` retune in situ | **0.659 ms** of excess | 1.6% | **≤0.3 ms** | **candidate for skip-by-ruling** - the whole in-situ excess over the probe floor is 0.659 ms across six shapes, the `S` picks are already the probe's own best, and each candidate costs a full golden-gate run because `S` reorders the split-K merge |

### The order to execute, and why it is not the share order

**L2 → L1 → L5**, then L4 only if the gate is within reach, and L3 not at all.
*(All three are executed and all three accepted. L5 was the largest non-GEMV row
in the step when it was run - 5.920 ms at `0bfa891` against the whole `prep`
family's 1.202 (**4.9×**) and `gdn_step`'s 0.749 (**7.9×**) - and at 3.585 ms it
still is. Only the GEMV rows are bigger, and no lever in this spec touches
them.)*

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
  *(Executed. The retile was the right half - SLM staging was never built and the
  measurement now argues against it, §L5 - and "measure the retile first" was the
  single most valuable sentence in this section: the retile immediately falsified
  the fit that priced it, and the block size ended up **64, not 128**, chosen by
  a four-value sweep - on the collapse of the marginal gain and on footprint,
  **not** on a knee: B32 is measured 0.070 ms/token BETTER at family level, and
  64 wins because the third halving buys ~nothing (−1.127 → −1.092 → −0.070 ms)
  while `attn_part` would double to 101.4 MB. The correctness risk
  did not materialise, but the gate turned out unable to see the change: its
  prompts reach `pos` 92 and the reassociation lives at depth. §L5, docs/14.)*

### The gate arithmetic, stated in advance

Spec §2's success bar is 31.746 ms/token; the step is 42.141. **The gap is
10.395 ms.** Summing the optimistic end of every lever above:

| | ms | |
|---|---|---|
| L2 `a‖b` | −1.9 | **actual −1.875 (bench), banked** |
| L1 `prep_res_norm` | −2.4 | **actual −2.220 (bench), banked** |
| L5 attention | −3.0 | **actual −1.73 (bench), banked** |
| L4 GEMV `S` | −0.3 | |
| L3 `gdn_step` | −0.06 | |
| **the whole ladder, optimistically** | **−7.7** | |

That lands at **34.4 ms/token ≈ 29.0 t/s** - short of 31.50 by about 2.7 ms.

**Three levers in, the ladder is spent and the gate is not cleared.** L2 paid
−1.875 ms against −1.9; L1 paid −2.220 against −2.4; L5 paid **−1.73 against
−3.0**, the first estimate to miss low, and it missed low even though its in-situ
row (−2.219 ms on the attn family) landed inside its own 1.5-2.5 band - 0.32 ms
of the difference is unattributed and recorded as such in §L5. The step is now
**36.32 ms** (measured, BENCHMARKS.md - 27.53 t/s at `c746840`, re-measured
27.54 at the gate below, the same engine twice) and **the gap
to the 31.746 ms bar is 4.574 ms**. What remains on the ladder is L4 (≤0.3) and
L3 (≤0.06), so **the ladder's own optimistic ceiling is 35.96 ms = 27.81 t/s** -
**4.214 ms short of the 31.746 ms bar (3.69 t/s short of 31.50) with every lever
cut.**

**This document predicted that outcome before the first lever was run**, and the
prediction was the point of ranking them: "the ladder as specified is
arithmetically unlikely to clear the gate". It said 34.4 ms optimistically; three
levers in, the measured answer is 36.3 with two negligible levers left. The
spec anticipated exactly this ("the win is *arithmetically* reachable but
tight") and provided for it: **§6's short path writes the re-assessment memo,
and it should now be written.**

**What the three executions changed is where the remaining risk sits.** All
three yields landed in or near their bands; all three landed for reasons the
ranking got partly wrong (L2: the work-group count was worth zero and the
K-split was everything; L1: the fold was priced and the rescale, treated as a
formality, was 40% of the win; L5: the retile worked, the *model* that priced it
was falsified twice, and the block size that shipped was 64 rather than the 128
the plan scoped). The pattern is consistent enough to state: **on this device
the mechanism named in advance has been wrong every single time, and the
measurement has been right every single time.** That is an argument for the
instrument, not for the ladder.

What the measurement adds to that memo, in advance and in order of size:

1. **`lm_head` at int4: ~3.3 ms**, no kernel work, docs/05 specialisation 1.
   It is the single largest measured item in the step that spec 1.5 does not
   touch. **It is no longer sufficient on its own, and that is the single most
   important consequence of L5's result.** This item used to read "it converts
   the projected 34.7 into 31.4 ms/token - over the bar"; that arithmetic was
   written against a ladder projection of 34.7 ms that assumed L5 would pay
   −3.0. L5 paid −1.73, so the ladder's optimistic ceiling is **35.96 ms**, and
   35.96 − 3.3 = **32.66 ms = 30.62 t/s - still UNDER the 31.50 t/s bar**, by
   0.88 t/s / 0.91 ms. **A second item is required**, and the memo has to name
   one rather than treating `lm_head` as the closer. Items 2 and 3 below are
   what is on the shelf, and item 2 has since been spent.
2. ~~**`attn_decode`'s 247 µs full-block critical path**~~ - **taken, §L5.**
   The retile shipped at `ATTN_BLOCK` 64 (260 work-groups at depth 4096) and paid
   −2.219 ms in situ, −1.73 on the bench. It *was* the cheapest untried thing in
   the step. What replaces it on this list is the **224 µs that a retiled
   `attn_decode` launch still costs** and that no model in this document has
   named - a probe that identifies that term is now worth more than any design
   written against the models that have already died.
3. **`a‖b` → `gdn_step` prologue fusion** and **norm → GEMV prologue fusion**
   remain redesigns, but §1 now prices what they are really buying: not the
   0.7 µs launch, the work-group count.
4. **A deep-context golden prompt** - not a performance item, a *trust* item,
   and L5 is why it is on this list. The gate's prompts reach `pos` 92, so the
   engine's attention arithmetic at the depth it is actually benchmarked at
   (`nb` = 65) has **no oracle**, only `attn_test`'s ulp bars against a host
   reference. That was tolerable while attention was untouched; it is a standing
   gap now that a lever has reassociated it. Cost is one oracle run at a longer
   prompt. §L5 and docs/14.
5. **The instrument's error bar needs re-establishing.** Two levers in a row
   have now missed their in-situ→bench prediction by more than the ±0.1 ms this
   document claims: L1 by 0.087 (inside), L5 by **0.32 (outside)**. The
   attribution method is the basis of every ranking on this page, so a residual
   it cannot explain is a finding about the method, not about the lever. §L5.

## The ladder, closed - the final ledger

The §6 gate was run 2026-08-25 from a clean worktree at `ef6acb0`
(`tools/bench_decode.sh`, median of three, idle box):
**27.54 t/s / 36.32 ms/token**, min 27.54, max 27.55, spread 0.04%. The bar was
31.50 t/s / 31.746 ms - **short by 3.96 t/s (12.6%), 4.574 ms/token still to
find**. The section above predicted this before the first lever was cut. Spec
1.5 therefore closes on its short path, and the priced menu is
[the re-assessment memo](superpowers/specs/2026-08-25-spec1.5-reassessment.md).

**Every lever, final.** Bench rows are the recorded medians (BENCHMARKS.md);
in-situ rows are `--profile --depth 4096 --steps 32`, the same instrument
throughout. All measured.

| lever | what shipped | in-situ site, before → after | in-situ Δ | bench step, before → after | **bench Δ** | sha | gate |
|---|---|---|---|---|---|---|---|
| **L2** `a‖b` GEMV | `{COLS_PER_WG, KSPLIT}` `{64,1}` → `{16,16}` - 8 → 128 subgroups | 2.341 → **0.256** ms | −2.085 | 42.141 → 40.266 | **−1.875** | `a1e2d3a` | 96/96 |
| **L1** `prep_res_norm` | one 1-work-group kernel → `prep_res_fold` + `prep_norm_finish`, 20 work-groups each - 16 → 320 subgroups | 2.893 → **0.484** ms | −2.409 | 40.266 → 38.046 | **−2.220** | `b045e11` | 96/96 |
| **L5** attention | `ATTN_BLOCK` 256 → **64** - 68 → 260 live work-groups | attn family 6.058 → **3.839** ms | −2.219 | 38.046 → 36.32 | **−1.726** | `c746840` | 96/96 |
| **L3** `gdn_step` | - | not attempted | - | - | - | - | skip-by-ruling: 1.09× its floor, whole kernel worth 0.06 ms |
| **L4** GEMV `S` | - | not attempted | - | - | - | - | skip-by-ruling: ≤0.3 ms, inside this instrument's drift |
| **the ladder** | | | **−6.713** | **42.141 → 36.32** | **−5.821** | `ef6acb0` (gate) | **96/96 throughout** |

(The L5 bench Δ is quoted unrounded here - 38.046 − 36.32 = 1.726 - so the
column sums; §L5 rounds the same number to −1.73.)

**The two Δ columns differ by 0.892 ms and three measured terms close it to
0.003.** Drift on the launches each lever did *not* touch: +0.119 (L2, +0.30%),
+0.182 (L1, +0.50%), +0.174 (L5, +0.56%). L1's 129 extra dispatches: +0.095
(derived at 0.733 µs/launch). L5's **full in-situ→bench miss**: +0.319 (§L5 -
the whole of it, of which ~0.01 is the `nb` 65 → 69 merge term named there and
the remaining ~0.31 is the unattributed residual). 0.475 + 0.095 + 0.319 =
**0.889** against **0.892**. Nothing else is needed to explain the ladder, and
the largest single term in that reconciliation is still the one nobody can
attribute. (The reconciliation takes the whole miss, not the ~0.31 remainder:
the ~0.01 that *is* attributed is part of the same column difference.)

**The step no longer closes to better than ~0.31 ms - and the second arithmetic
is not a second witness.** The L5 after-run's Σ over all 774 launches is
35.342 ms measured; the derived dispatch gap at 774 launches is 0.567 ms and
the gate's own host term is 0.101 ms measured - **36.010 ms against a measured
36.32**, the same 0.31 ms reached without going through any lever's
before/after. **It is the same measurement restated**: the 0.733 µs/launch rate
was *defined* as this document's closing residual for the 645-launch step, so
carrying it to 774 launches reproduces by construction whatever the ladder did
not close. Anyone quoting it as corroboration is quoting one number twice.

**Read per launch, the same numbers do say something new.** The un-instrumented
gap was **0.733 µs/launch** at `43bb720` (0.473 ms / 645); at the gate it is
36.32 − 0.101 host − 35.342 Σ = 0.877 ms / 774 = **1.133 µs/launch, +55%**
(derived, cross-run; a ±0.5% band on Σ spans +23…+86%, so the direction holds
and the magnitude does not). **The cost outside the kernels grew while the
kernels shrank.** The obvious suspect is drain - `attn_decode` retires 260
work-groups where it retired 68 - but nothing here measures that, and it is why
the memo ranks repeated-run averaging ahead of every sub-0.5 ms design.

**What did not work, kept because a ladder that records only its wins is not a
measurement:** `a‖b` option (a) `{16,1}` (+0.7%, worth zero - §L2), the
block-read `KU` variant (worth zero *and* **+1.3% on `lm_head`** - a cost, not
a saving; §L2), L1's
stage B on one work-group (the plan's literal design, 60% of the win - §L1),
`ATTN_BLOCK` 128 (superseded) and 32 (0.070 ms/token better at family level,
rejected on a 101.4 MB `attn_part` - §L5).

**The ranking's record: three for three on yield, zero for three on
mechanism.** Every lever landed in or near its predicted band; every one landed
for a reason the ranking got wrong. That is an argument for this instrument, not
for the ladder - and it is the reason the memo's first item is a probe rather
than a design.

## Spec 1.6 §5.4 - repeated-run averaging, and the attribution floor it measures

The re-assessment memo ranked this **above every design on its menu except the
probe**: "it is the *precondition* for believing any yield under ~0.5 ms, which
is every remaining item". The measured problem it was asked to fix is that this
profiler reproduces to **0.006% over two runs three minutes apart** and yet the
*untouched* launches drifted **+0.30% / +0.50% / +0.56%** across spec 1.5's
three levers - 0.12 to 0.18 ms per comparison - and two levers in a row then
missed their in-situ→bench prediction **in opposite directions** (L1 by +0.087,
L5 by −0.319). Errors that run both ways at 0.09-0.32 ms are a resolution limit,
not a missing constant.

### What was built

`b70-decode --profile` gains **`--repeats R`** (default 1; `--repeats 1` prints
exactly the report it printed before the flag existed). R **sessions** of
`--steps` instrumented replays run over **one** ingest - at depth 4096 the
ingest is ~140 s and repeating it would dominate the run for a number nobody
reads - and each session begins by rewinding `Control::pos` to the post-ingest
value. That rewind is the point: **every session profiles the identical launch
shape**, the same 774 launches at the same positions with the same `nb`, so what
the spread across them contains is drift and scheduling and nothing else. The
report gains a per-family **mean, sd, min..max** and a **2σ-resolvable Δ**
column, and the same three rows for Σ, for the fence wall and for the derived
gap. Two CLI rejections cover the flag (`--repeats` outside `--profile`,
`--repeats 0`), so the suite is 40 tests.

The 2σ column is `2·sd·sqrt(2/R)`: two binaries measured this way give two means
whose difference carries a standard error of `sd·sqrt(2/R)`, and that is the bar
an attribution has to clear. It is printed per family because a family is the
unit an attribution is written in.

### The floor, measured

**Box conditions.** 2026-08-25 22:34-22:56 CEST, load average **14-15** - a
docker `vllm-xpu-kernels` build held the CPU throughout. The GPU was otherwise
idle and no AutoRound run was live. Every number below is a **device-clocked
kernel timestamp**, which is what makes it quotable under load; **no
bench-grade wall-clock absolute is recorded in this section**, and the 36.32 ms
step it is compared against is the idle-box `--bench` median already on record.

Three **independent processes** of
`--profile --depth 4096 --steps 32 --repeats 5` - 480 profiled replays in all.
Two floors, and the difference between them is the finding:

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

> ### **The step-level attribution floor is 0.051 ms** (Σ, 2σ, three processes)
>
> Against the **±0.1 ms** the single-run method was claimed at and the
> **0.12-0.18 ms** of day-scale drift spec 1.5 actually measured, that is a 2-3.5×
> improvement, and it is the first time this project has had the number rather
> than an adjective. Per family, and setting `attn_reduce` aside for the
> paragraph below, the floor runs **0.2 to 15 µs** - `attn_decode` resolves to
> **0.30 µs/step**, `gemv` to **14.9**.

**What this unlocks, named.** The memo parked **L4** - 0.659 ms of measured
in-situ GEMV excess - as "blocked on §5.4 … inside the instrument's own drift".
`gemv`'s floor is now **14.9 µs**, so an `S` retune worth L4's estimated
≤0.3 ms is **20× the bar**. The instrument is no longer what blocks L4; only its
golden-gate cost is.

### The finding that justifies the whole exercise: within ≠ across

**`attn_reduce` reproduces to 0.3% inside every process and lands 31% apart
between them.** Processes 1 and 2 measure 194.034 and 193.638 µs/step with
within-process sd of 0.62 and 1.30; process 3 measures **254.099** with an sd of
0.72. Three tight distributions, two of which agree and one of which does not,
on **one binary, one box, one twenty-minute window**. Its across-process floor
is **56.8 µs against a within-process 1.6** - a factor of **35**.

Nothing here explains it (the launch shape is identical: `nb` is 65 in all
three, the merge loop is bounded by `nb`, and the allocation sizes are fixed by
`max_len`), and it is recorded unexplained, as this document records everything
it cannot account for. What it *proves* is the design rule the tool now prints
in its own output: **R sessions inside one process are a lower bound on the
floor, not the floor.** An attribution must compare two binaries across
processes, R ≥ 5 each, and read the bar off the across-process spread.

**And the window matters too.** These three processes ran within twenty minutes.
Spec 1.5's ±0.12-0.18 ms was measured across a **day**. So the operating rule is
two-tier, and both tiers are measured: **compare two binaries in one sitting and
the floor is 0.05 ms; compare across a day and it is ~0.15 ms** until somebody
measures a day-scale series with this flag.

### The ~0.31 ms unattributed residual does NOT resolve - it is confirmed

§L5 and the memo both close on a step that "no longer closes to better than
~0.31 ms". The obvious hope for a better instrument was that the residual was
the instrument. It is not.

Σ is now known to **35.327 ms ± 0.031** (mean and across-process sd of 480
replays) where it was one run's 35.342. Running the memo's own arithmetic on it -
Σ + the derived 0.567 ms dispatch + the measured 0.101 ms host - gives
**35.995 ms against the recorded 36.32**, a residual of **0.325 ms**. That is
**6.4× the Σ row's own 2σ**, and the bench side contributes only ±0.015 ms (the
gate's median-of-three spread was 0.04%). **The residual is a real term at
better than 6σ.**

*(The memo's caveat still applies and is not repealed by a better Σ: the
0.733 µs/launch dispatch rate in that arithmetic was **defined** as the closing
residual of the 645-launch step at `43bb720`, so this construction cannot be
quoted as an independent witness. What it now is, is a construction whose one
noisy input has been pinned.)*

**Read per launch, with Σ pinned**, the memo's genuinely new number holds and
tightens: 36.32 − 0.101 host − 35.327 Σ = **0.892 ms over 774 launches =
1.152 µs/launch**, against **0.733 µs/launch** at `43bb720`. That is **+57%**,
where the memo derived +55% from a single Σ and could only bound it between +23%
and +86%. Σ's own ±0.031 ms is now ±0.04 µs/launch, so the remaining width is
entirely in the *other* end - one un-repeated run at `43bb720` - and repeating
that measurement is a cheap way to close it. **The cost outside the kernels grew
while the kernels shrank, and it is now measured on 480 replays instead of one.**

### One row this refines while it is here

`attn_decode` was recorded at **224.046 µs/launch** from a single profiled run
at `c746840`. The 480-replay mean at `4f16131` is **223.199 µs/launch**, with an
across-process 2σ of **0.019 µs/launch** - 0.38% under the recorded figure and
the tightest number in this document. The recorded L5 rows are left as they
were, because they are what that lever measured; 223.199 is what the row is
today, and it is what §5.2 below compares its probe against.

## Spec 1.6 §5.2 - what `attn_decode`'s 224 µs buys, named

The re-assessment memo put this measurement above every design on its menu:
`attn_decode` is **3.585 ms/token, the largest non-GEMV row**, four
pre-registered cost models have died on it (§2's A and B, §L5's C and D), and
"what dominates that kernel's current 224 µs launch is **unknown**". It is not
unknown any more, and the way it stopped being unknown is that nothing here is
fitted: `tools/probe/probe_attn` transplants the launch and then **removes one
term at a time**, so every number below is a subtraction between two
measurements.

**Box conditions for every number in this section:** 2026-08-25 22:23-22:30
CEST, load average **13-16** - a docker `vllm-xpu-kernels` build was occupying
the CPU throughout. The GPU (card 0) was otherwise idle and no AutoRound run was
live (`/home/user/autoround-qwen38.log` absent). Every figure is a
**device-clocked kernel timestamp** (`zeEventQueryKernelTimestamp`), the same
instrument `--profile` uses, so CPU load cannot enter it; the probe reproduces
its own base row to **±0.2% across four whole-battery runs**. No bench-grade
wall-clock absolute is recorded here.

### The instrument, and the two things that had to be true before it could be read

`tools/probe/probe_attn.cl` is a **copy** of `attn_decode`, not an include:
`src/kernels/attn.cl` is untouched by all of this. Every ablation computes a
wrong answer on purpose and no runtime path binds one.

> **A methodology finding, recorded because it nearly cost the whole battery.**
> The first run of this probe read **295.417 µs** for the base variant and its
> block sweep came out non-monotonic (B64 *slower* than B128). The cause was
> warm-up: one warm-up replay of 40 launches is not enough for this device's
> clock ramp. **Measured**: the same binary read 264.167 and 206.458 µs on two
> consecutive invocations at 40 launches of warm-up, and **207.2-207.6 µs on
> every one of eight invocations from ~100 warm-up launches upward** (reps 100,
> 200, 400, 800, twice each) - 0.2%. The probe now replays the closed list 8
> times and drops the first 3, which is `tests/kernels/gemv_harness.h`'s and
> `probe_gemv`'s own convention and is why those two never saw this.

**Validation 1 - the transplant reproduces the launch.** Probe base at depth
4096, `ATTN_BLOCK` 64: **206.250 µs** against the in-situ **223.199** (§5.4's
480-replay mean; the single-run 224.046 §L5 recorded is 0.38% above it). In situ
is **+8.2%** above the probe floor, the same sign and the same order as GEMV's
transplant (+2.3% over the family, +7.0% on `out/o_proj` - the anatomy's own
table). A probe floor under a live launch is what a floor is supposed to do.

**Validation 2 - it reproduces four in-situ points it did not fit.** The
`ATTN_BLOCK` sweep §L5 measured *in situ*, re-measured *in the probe*:

| `ATTN_BLOCK` | live WGs | probe µs | in-situ µs (§L5) | in situ vs probe |
|---|---|---|---|---|
| 32 | 516 | **182.604** | 203.837 | +11.6% |
| **64** | 260 | **206.250** | 224.046 | **+8.6%** |
| 128 | 132 | **258.229** | 296.684 | +14.9% |
| 256 | 68 | **349.792** | 369.988 | +5.8% |

All four the same sign, all within 6-15%, and the *shape* - monotone, with the
last halving's gain collapsing - reproduced. Everything below is therefore read
as a **share of the launch**, never as an in-situ absolute.

### The battery - 32 variants, one term each (measured)

`tools/box.sh run "./build/tools/probe/probe_attn"`, depth 4096, `ATTN_BLOCK`
64, 4 rotated KV caches, 200 timed launches per row. Base **206.250 µs**.

| what the variant removes | µs/launch | Δ vs base | Δ% |
|---|---|---|---|
| - (`base`, the control) | **206.250** | - | - |
| everything; return at the early-out (`earlyout`) | 1.667 | −204.583 | −99.2% |
| everything but the 6 q-head stagings (`qstage`) | 4.167 | −202.083 | −98.0% |
| **both KV global loads** (`noloads`) | **92.083** | **−114.167** | **−55.4%** |
| - the V load alone (`nov`) | 114.896 | −91.354 | −44.3% |
| - the K load alone (`nok`) | 165.938 | −40.312 | −19.5% |
| **both loads' cache misses** - same messages, all L1-resident (`hotkv`) | **142.500** | **−63.750** | **−30.9%** |
| - V's misses alone (`hotv`) | 153.333 | −52.917 | −25.7% |
| - K's misses alone (`hotk`) | 180.417 | −25.833 | −12.5% |
| the whole online update (`nosoftmax`) | 152.500 | −53.750 | −26.1% |
| - `exp` alone (`noexp`) | 175.000 | −31.250 | −15.2% |
| - `exp`'s precision only, `native_exp` (`natexp`) | 180.417 | −25.833 | −12.5% |
| all 6 work-group barriers per wave (`nobar`) | 173.125 | −33.125 | −16.1% |
| - the tree and its 4 barriers (`notree`) | 182.292 | −23.958 | −11.6% |
| - 3 of the 4 tree barriers' SCOPE only (`sgtree`) | 198.333 | −7.917 | −3.8% |
| half the K messages, 64 B each (`vec2`) | 185.729 | −20.521 | −9.9% |
| a quarter of them, 128 B each (`vec4`) | 186.562 | −19.688 | −9.5% |
| the KV cache's 2048 B position stride (`kvt`) | 189.583 | −16.667 | −8.1% |
| - K's stride alone (`kt`) | 186.979 | −19.271 | −9.3% |
| - V's stride alone (`vt`) | 196.042 | −10.208 | −4.9% |
| the stride AND half the K messages (`kvt_vec2`) | 181.042 | −25.208 | −12.2% |
| **the loop-carried chain across waves** (`depbreak`) | **202.917** | **−3.333** | **−1.6%** |
| **the K load's latency - one wave of prefetch** (`prefetch`) | **206.562** | **+0.312** | **+0.2%** |
| **the 1.6 MB of `attn_part` stores** (`nopart`) | **206.458** | **+0.208** | **+0.1%** |
| both loads AND `exp` (`mathonly`) | 61.146 | −145.104 | −70.4% |
| the update and the tree (`loadsonly`) | 137.396 | −68.854 | −33.4% |

### The dominant term: **the KV load path, 55.4% of the launch**

Removing both global loads and changing nothing else takes the launch from
206.250 to **92.083 µs**. Nothing else in the kernel is close: the whole online
softmax is 26.1%, all six barriers per wave are 16.1%, `exp` is 15.2%, and the
grid, the q staging, the output stores, the loop-carried dependency and the load
latency are - together - 3.6%.

**It is a throughput term, not a latency chain, and the two cheapest
falsifications in the battery say so.** Cutting the wave-to-wave dependency so
no wave waits on the one before it buys **1.6%**. Issuing the next wave's K row
before the current wave's barriers buys **nothing at all** (+0.2%, i.e. inside
the noise). Whatever the loads are waiting for, it is not each other.

**And the term splits, measured, into two halves that want different attacks:**

| half | measured by | µs | share of the launch |
|---|---|---|---|
| **cache service** - what an all-L1 kernel stops paying | base − `hotkv` | **63.750** | **30.9%** |
| **message issue** - what survives when every access hits L1 | `hotkv` − `noloads` | **50.417** | **24.4%** |
| the two together | base − `noloads` | 114.167 | 55.4% |

The issue half is arithmetic anyone can check: the kernel issues **3.19 M global
load messages of 32 bytes** per launch (16 lanes of a subgroup read 16
consecutive `ushort` = 32 B, half a cache line; 16 K + 16 V messages per subgroup
per wave × 16 subgroups × 24 waves × 260 work-groups), plus ~3.7 M SLM
messages. `vec2` - the same bytes in half as many 64 B messages - buys
**−9.9%**, and `vec4` buys no more (−9.5%), which is what a pipeline that has
stopped being message-limited looks like.

**Two arithmetic checks that the subtractions are honest.**

1. **Loads and `exp` are additive to 0.2%.** `noloads` (−114.167) + `noexp`
   (−31.250) = −145.417; `mathonly`, which removes both in one binary, measures
   **−145.104**.
2. **The q-head axis is linear to 0.14%.** Running 1, 2, 3 and 6 of the six
   q-head passes measures **66.771 / 95.833 / 121.250 / 206.250 µs**; a least
   squares line over those four points is **39.09 µs + 27.81 µs per pass**, which
   predicts 205.97 at six against 206.250 measured. *(A two-parameter fit over
   four points - stated as arithmetic over measurements, and deliberately not
   extended to the depth or block axes, which are where the four dead models
   died.)*

### What the fixed term is fixed on - the memo's other question, answered

§2's `F ≈ 90.8 µs` was called a per-launch fixed cost and nobody could say of
what. The battery prices the candidates directly and **none of them is it**:
the whole 1024-work-group grid, early-out included, is **1.667 µs** (1.63 ns per
work-group - the fixed grid is nine times cheaper than docs/12's ~15 ns estimate,
and "context-bucketed lists are dead" gets a second, per-kernel witness); the
six q-head stagings and their twelve barriers add **2.5 µs**; the `attn_part`
stores are **0.1 µs**. Total genuinely fixed cost: **≈ 4.2 µs, 2.0% of the
launch.**

The 39.09 µs intercept of the q-head line is therefore **not a fixed cost - it
is the first pass being colder than the other five**. Subtracting the 4.2 µs
that is genuinely fixed leaves the first q-head pass at ≈ 62.6 µs against
≈ 27.8 µs for each of the following five, a ratio of **2.25**: pass 1 fills the
block into cache and passes 2-6 largely re-read it, exactly as
docs/12's (kv-head, block) grid was designed to do. **The 6× reread costs
5 × 27.81 = 139 µs - 67% of the launch - and it is not free**, which is the half
of docs/12's rationale that the measurement corrects.

### Where the work-groups stopped being free (measured, and it corrects §2)

Sweeping `--pos` on the base variant at the shipped block size:

| `--pos` | live WGs | µs/launch | µs per live WG |
|---|---|---|---|
| 512 | 36 | 50.417 | 1.400 |
| 1024 | 68 | 57.083 | 0.839 |
| 2048 | 132 | 113.646 | 0.861 |
| **4096** | **260** | **206.458** | **0.794** |
| 8192 | 516 | 389.583 | 0.755 |
| 16000 | 1004 | 674.896 | 0.672 |

From 68 to 1004 live work-groups - **14.8×** - the launch grows **11.8×**. That
is linear, with a mild and steadily shrinking discount. §2 measured "`nb` 5 → 17
is **3.40× the live work-groups for +6.9% of time**" and concluded work-group
count is nearly free; **that was measured at `ATTN_BLOCK` 256, over 20 → 68
work-groups**, which is 320 → 1088 hardware threads on a device with ~2048
thread slots. It is a *low-occupancy* regime, and the block size that shipped no
longer enters it. **The claim "work-group count is nearly free" is withdrawn for
the shipped configuration**; what survives it, and is now measured per kernel
rather than per step, is that *dead* work-groups are free (1.63 ns each).

This also retires the last of the four dead models' territory: at B64/depth 4096
the launch is **linear in the work it is given**, and the only ways to make it
smaller are to give it less work per position-pass or fewer position-passes.

### The honest implication: what of the 3.585 ms is recoverable

`attn_decode` is 3.585 ms/token in situ. Its **unique-KV traffic floor** is
268 MB/token at the measured 590 GB/s = **0.454 ms** (derived), so the excess is
**3.13 ms**. Scaling each measured share to the in-situ row (derived - shares
are measured, the row is measured, the product is arithmetic):

| lever | what it is | ceiling, ms/token | what it costs |
|---|---|---|---|
| **the locality ceiling** (`hotkv`) | every KV access an L1/SLM hit | **−1.108** | this is a BOUND, not a design. SLM staging is the only mechanism and it cannot beat it |
| `exp` removed (`noexp`) | - | −0.543 | **not available.** `exp` is ruled by the rounding discipline and `native_exp` (−0.449) is ruled with it - the host reference cannot follow either |
| all wave barriers (`nobar`) | - | −0.576 | **not available.** The cross-subgroup ones are what publish `dot_red` |
| **transposed KV cache + 64 B K messages** (`kvt_vec2`) | `[head][pos][dim]` and `ushort2` loads | **−0.438** | a cache-layout change in `attn_prep`/`attn_decode` and one reassociation of the K dot |
| - 64 B K messages alone (`vec2`) | - | −0.357 | one reassociation of the K dot; no layout change |
| - transposed cache alone (`kvt`) | - | −0.290 | **no arithmetic change at all** |
| **subgroup-scope tree** (`sgtree`) | 3 of 4 tree barriers narrow | **−0.138** | **no arithmetic change at all** - docs/12 listed this as "not used, not measured"; it is measured now |

**Read together: about 0.5 ms/token is reachable by tuning** (the three
zero-or-low-risk rows do not add - `kvt_vec2` measures −12.2% where its parts
measure −8.1% and −9.9% - so ~0.5 ms is the realistic combined figure, derived),
**and the rest is structural.** Even at the `hotkv` bound - a kernel where every
KV byte is already in L1 - attention would still be ~2.48 ms/token against a
0.454 ms byte floor, because what is left is 3.19 M global and ~3.7 M SLM
messages per launch and a device that the pos sweep shows is *saturated* at this
size. Message count is set by the (position × dim) decomposition - 16 subgroups,
one position each, 16 lanes of 16 dims - which is the algorithm's shape, not a
tuning knob.

**So: `attn_decode`'s excess is a message-count problem, and message count is
structural.** The honest answer to "is the ~3.3 ms recoverable" is **no** - about
a sixth of it is, for one layout change, one reassociation and one barrier
scope, and the remainder needs a different decomposition of the work rather than
a better version of this one. That is a finding about what NOT to spend spec
1.6 on, which is what the memo asked the probe to produce.

**It also kills the standing hypothesis for §5.4's other residual.** The memo
guessed that the cost outside the kernels grew +57% because "`attn_decode` now
leaves 260 work-groups to retire where it left 68". `earlyout` prices a
work-group's whole dispatch-and-retire at **1.63 ns**; 192 extra of them per
launch across 16 layers is **0.005 ms/token**, and even at ten times that rate -
a live work-group carries state an early-outed one does not - it is 0.05 ms
against a 0.325 ms residual. **Drain is not the explanation**, and §5.4's
residual stays unattributed with one fewer candidate.

**And it does not change the memo's headline arithmetic.** −0.5 ms/token on a
36.32 ms step is 27.54 → 27.92 t/s. With `lm_head` at int4 at its own derived
ceiling (−3.26) and the unspent ladder at its ceiling (−0.36) the step reaches
**32.20 ms = 31.06 t/s - still 0.44 t/s under the bar** (derived). The probe was
run to find the second item §4 said the project could not name; **it did not
find one**, and it cost a day rather than a requant.

### What died here, for the tally

Four predictions were pre-registered before the first run
(`.superpowers/sdd/2026-08-25-spec1.6-stage0/task-A-report.md` carries them
verbatim, with a point prediction for all 32 variants).

| model | its own strongest prediction | measured | verdict |
|---|---|---|---|
| **E - message-issue rate** (the author's primary) | `hotkv` "barely moves", ~190 µs | **142.500** | **falsified.** Issue is real (24.4%) but it is the smaller half of the load term |
| **F - memory latency** | `prefetch` ≥10% better | **+0.2%** | **falsified**, and by the cheapest row in the battery |
| **G - the online softmax** | `nosoftmax` ≤123 µs | **152.500** | **falsified**; it is 26.1%, not dominant |
| **H - barriers** | `nobar` ≤157 µs | **173.125** | **falsified**; 16.1%, not dominant |

**Four models predicted, four falsified - and the question was still answered**,
because a battery of subtractions does not need a surviving model. That is the
methodological point of this section and it is the one to carry into spec 1.6:
where a quantity has resisted four fits, stop fitting and remove terms.

## What this document does not settle

- **`attn_decode`'s internal split - SETTLED 2026-08-25, by subtraction rather
  than by a fifth model** ("Spec 1.6 §5.2" above). Four fits died here (§2's A
  and B, §L5's C and D) and a fifth was never written; `tools/probe/probe_attn`
  removes one term at a time instead, and names the **KV load path at 55.4% of
  the launch** - 30.9% cache service, 24.4% the 32-byte load messages
  themselves - against 26.1% for the whole online softmax, 16.1% for the wave
  barriers, and **under 2% for everything genuinely fixed per launch**. It is a
  throughput term, not a latency chain: cutting the wave-to-wave dependency buys
  1.6% and prefetching buys nothing. Two of this bullet's own claims did not
  survive: **"work-group count is nearly free over 20 → 260" is withdrawn** -
  that was measured at `ATTN_BLOCK` 256 in a low-occupancy regime, and at the
  shipped block the launch is *linear* in live work-groups from 68 to 1004 - and
  the "~73 µs per halving" arithmetic is superseded by a probe that reproduces
  all four block points independently. What is left open is not the split but
  the *decomposition*: ~0.5 ms/token of the 3.585 is reachable by tuning and the
  rest is a message-count problem set by the algorithm's shape.
- **Every *remaining* yield in the ladder is an estimate**, and only L3 and L4
  remain, both priced at "cannot repay a golden-gate run". §1's per-work-group
  ceiling was measured at 1, 2 and 5 work-groups and the extrapolation off it was
  falsified by L2; the subgroup curve that replaced it has since been measured at
  8, 16, 32, 128 and 320 subgroups across two kernels (§L2, §L1) and it has held
  every time, sub-linearly. L5 never rested on that curve - it rested on a
  two-parameter fit, and the fit was wrong while the *direction* was right, which
  is the pattern this whole document keeps finding.
- **M > 1.** Everything here is the M = 1 decode step. The bucket's kernels are
  the ones whose costs move most with M, and none of that is measured.
