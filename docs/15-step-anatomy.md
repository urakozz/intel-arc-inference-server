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
only after the fence. Positions 4096…4127, so `nb` (live 256-position attention
blocks) is 17 throughout - the same regime the recorded bench row measures.

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

Compare with the same table in docs/05: five rows that were a floor, an
extrapolation and a remainder are now seven rows that were all measured in one
run. **98.6% of the step is kernel time inside the fence** (41.571 of 42.141);
1.1% is dispatch and 0.2% is the host.

## What the measurement says that nothing else did

### 1. One work-group is worth about 13-17 GB/s, and that is the whole story of the bucket

Three independent points, all measured in this run, all from docs/12's own byte
counts:

| kernel | work-groups | GB/s total | GB/s **per work-group** |
|---|---|---|---|
| `prep_res_norm` (SP16) | 1 | 17.0 | **17.0** |
| `a‖b` GEMV | 2 | 27.0 | **13.5** |
| `prep_silu_mul` | 5 | 60.8 | **12.2** |
| `gdn_step` | 192 | 540 | 2.8 (saturated - the device is the limit, not the core) |

A single Xe-core cannot keep enough loads in flight to exceed ~15 GB/s, so
**saturating 590 GB/s needs roughly 40 work-groups**, and every kernel in the
bucket that is slow is slow for exactly one reason: it was given one or two.
This is the mechanism docs/12 wrote down as a risk when `prep_res_norm` landed
("One work-group per token in `prep_res_norm` is the risk in this design, and it
is unmeasured") - measured, confirmed, and now also priced for `a‖b`, which had
never been timed at all. It is also what makes both levers *predictable*: the
fix is work-groups, and the yield is close to linear in them until ~40.

The hypothesis in docs/12 and docs/05 - 129 launches at "40 µs apiece, 5.2 ms" -
was **half right**: the mechanism is exactly as named, the magnitude is 22.3 µs
and 2.871 ms.

### 2. `attn_decode` is 2.45× its estimate, and it is the second-largest thing in the step

5.782 ms in situ against the 2.360 ms docs/12 extrapolated (2.314 estimated
block work + 0.046 measured early-out). The estimate was a line through two
`--bench` points at different depths; it under-predicted the slope by 53%.

Three in-situ points, all measured with this instrument:

| run | `nb` | grid | µs/launch | ms/token |
|---|---|---|---|---|
| depth 4096, `--max-len 16384` | 17 | 4 × 64 = 256 WGs (68 live) | **361.400** | 5.782 |
| depth 64, `--max-len 16384` | 1 | 4 × 64 = 256 WGs (4 live) | **153.608** | 2.458 |
| depth 64, `--max-len 4096` | 1 | 4 × 16 = 64 WGs (4 live) | **153.665** | 2.459 |

- **The fixed grid is free - re-confirmed per kernel.** Rows 2 and 3 differ by
  192 idle work-groups per layer and by **0.04%**. Doc 07 #12 concluded this
  from a whole-step difference of 0.046 ms; this is the same conclusion measured
  on the kernel itself. Context-bucketed lists remain worthless.
- **The in-situ slope is 12.987 µs per live block per launch** = **0.208
  ms/token per block** (derived from rows 1-2), against the 0.1361 ms/token the
  bench two-point line gave (estimated, docs/12). At `nb` = 17 that is 3.53 ms
  of block-scaled work and **2.25 ms that does not scale with depth at all**.
- The depth-independent 2.25 ms is **not** the early-out (rows 2 vs 3 settle
  that). It is one wave of block work with only 4 live work-groups per layer -
  the same starvation as §1, in the kernel where it is least visible.
- **It is not bandwidth-bound.** 100.7 MB of KV reads per layer (the 6× q-head
  reread) in 361.4 µs is **279 GB/s** - 47% of the measured 590 - and the
  *unique* 16.8 MB is only 46 GB/s. docs/12 inferred 740 GB/s from the bench
  line and concluded the reread was "substantially cache-served"; the direction
  was right (46 GB/s of unique traffic cannot be DRAM-limited) but the magnitude
  was not. There is real time in this kernel and it is not being spent on DRAM.

### 3. `gdn_step` is exonerated, completely

0.733 ms in situ, against a **0.671 ms** traffic floor (396 MB/token at the
measured 590 GB/s, docs/12's own arithmetic). It runs at **540 GB/s effective -
92% of the device** - on 192 work-groups with no spills. docs/12 said "if the
profile puts it near the bound, the cause is occupancy or register pressure";
the profile puts it **at 1.09× the bound**. The entire kernel, made perfect,
is worth 0.06 ms. The 48×4 grid and the four-fold redundant conv reads were the
right trade and the measurement says so.

### 4. `lm_head` is 10.4% of the step and no lever in this spec touches it

4.376 ms in one launch, at 0.6% above its probe floor and 97% of device
bandwidth. There is nothing to tune. Quantising it to int4 is worth ~3.3 ms
(4.376 → ~1.1, docs/12 `gemv_bf16` → Measured) and it is item 1 of docs/05's
specialisation list, deliberately outside spec 1.5. It is named here because
the gate arithmetic below cannot be read honestly without it.

## The ranked lever ladder

Ranked by **measured in-situ share**, which is what this document exists to
produce. Yields are estimated, and each says on what basis.

| rank | lever | measured | share of Σ | expected yield | basis |
|---|---|---|---|---|---|
| 1 | **L5** attention (`attn_decode`) | **5.782 ms** | 13.9% | **1.0-3.0 ms**, wide | not bandwidth-bound (279 GB/s on rereads, 46 on unique bytes), so SLM staging removes issue slots rather than DRAM; the 2.25 ms depth-independent term is 4-work-group starvation and may respond to a finer block instead |
| 2 | **L1** `prep_res_norm` two-stage | **2.871 ms** | 6.9% | **1.5-2.4 ms** | one work-group at 17.0 GB/s; §1's per-WG ceiling makes 20 work-groups worth ~10×, leaving ~0.3-0.5 ms plus a second launch per site (129 × 0.7 µs = 0.09 ms) |
| 3 | **L2** `a‖b` GEMV occupancy | **2.335 ms** | 5.6% | **1.2-1.9 ms** | two work-groups at 13.5 GB/s each; `COLS_PER_WG` 16 → 8 WGs is 4× the slices at bit-identical arithmetic, K-split-by-4 → 32 WGs is the ceiling (~0.15 ms) |
| 4 | **L3** `gdn_step` retile | **0.733 ms** | 1.8% | **≤0.06 ms** | **candidate for skip-by-ruling** - 1.09× its own 0.671 ms traffic floor, 92% of device bandwidth; the lever cannot repay a day of work at any outcome |
| 5 | **L4** GEMV `S` retune in situ | **0.659 ms** of excess | 1.6% | **≤0.3 ms** | **candidate for skip-by-ruling** - the whole in-situ excess over the probe floor is 0.659 ms across six shapes, the `S` picks are already the probe's own best, and each candidate costs a full golden-gate run because `S` reorders the split-K merge |

### The order to execute, and why it is not the share order

**L2 → L1 → L5**, then L4 only if the gate is within reach, and L3 not at all.

Share ranks L5 first, and it is the biggest number on the page. It is
nevertheless third to execute, for reasons the measurement itself supplies:

- **L2 is half a day and its option (a) is bit-identical** (same per-column
  arithmetic order, more work-groups). §1 makes its yield close to arithmetic.
  It is the cheapest ms on the ladder and it de-risks nothing else.
- **L1 is a day, its mechanism is measured and its design was written when the
  kernel landed.** Its only correctness exposure is the global Σx² tree, which
  the golden gate arbitrates.
- **L5 is the only sketch-level design in the plan** (the plan says so), it is
  the one lever whose expected yield the measurement does *not* pin - 279 GB/s
  on cache-served rereads means removing the rereads may buy far less than the
  5.782 ms suggests - and it carries the ladder's highest correctness risk
  (online-softmax staging). Running it after two banked, certain wins means it
  is attempted with the gate arithmetic already known.

### The gate arithmetic, stated in advance

Spec §2's success bar is 31.746 ms/token; the step is 42.141. **The gap is
10.395 ms.** Summing the optimistic end of every lever above:

| | ms |
|---|---|
| L2 `a‖b` | −1.9 |
| L1 `prep_res_norm` | −2.4 |
| L5 attention | −3.0 |
| L4 GEMV `S` | −0.3 |
| L3 `gdn_step` | −0.06 |
| **the whole ladder, optimistically** | **−7.7** |

That lands at **34.4 ms/token ≈ 29.0 t/s** - short of 31.50 by about 2.7 ms.
**The ladder as specified is arithmetically unlikely to clear the gate**, and
this document says so before the first lever is cut rather than after the last.
The spec anticipated exactly this ("the win is *arithmetically* reachable but
tight") and provided for it: §6's short path writes the re-assessment memo.

What the measurement adds to that memo, in advance and in order of size:

1. **`lm_head` at int4: ~3.3 ms**, no kernel work, docs/05 specialisation 1.
   Alone it converts the projected 34.4 into 31.1 ms/token - over the bar.
   It is the single largest measured item in the step that spec 1.5 does not
   touch.
2. **`attn_decode`'s 2.25 ms depth-independent term** - a block retile
   (128-position blocks → 34 × 4 = 136 work-groups) is a grid change of exactly
   the kind L3 was scoped for, and this document says L3's budget went to the
   wrong kernel.
3. **`a‖b` → `gdn_step` prologue fusion** and **norm → GEMV prologue fusion**
   remain redesigns, but §1 now prices what they are really buying: not the
   0.7 µs launch, the work-group count.

## What this document does not settle

- **`attn_decode`'s internal split.** 5.782 ms is measured; the 3.53 / 2.25 ms
  scaling split is derived from two depths of the same instrument, and the
  reason the depth-independent half costs what it does is a hypothesis
  (4-work-group starvation) with the early-out ruled out but nothing else.
- **Every yield in the ladder is an estimate.** The per-work-group ceiling in §1
  is measured at 1, 2 and 5 work-groups; the claim that it stays roughly linear
  to ~40 is an extrapolation, and L1's and L2's real yields are what Tasks 5-6
  measure with this same instrument.
- **M > 1.** Everything here is the M = 1 decode step. The bucket's kernels are
  the ones whose costs move most with M, and none of that is measured.
