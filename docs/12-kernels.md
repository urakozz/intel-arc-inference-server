# Kernels

One section per device kernel: what it computes, how the work is assigned and
**why**, what was rejected, and the numbers that decided it. A section that only
restates the code is not finished. Written as kernels land (spec 1 §3).

Everything measured here was measured on the box, either with the probes in
`tools/probe/` or - for the two token-boundary kernels, whose cost is a few µs
and needs no separate harness - by a timing loop inside their own test, printed
and never asserted. Weight buffers are filled with **random** nibbles, never a
uniform value: the B70 losslessly compresses device-local memory, so a buffer of
one repeated word reads back above the 608 GB/s theoretical peak and any timing
taken from it is fiction (doc 01). Real weights are incompressible within a
cache line; so is the probe's data.

---

## The step these kernels add up to - measured 2026-08-25

Every "% of a step" in this document used to be against an *estimated* ~26 ms
token (the roofline). The step is now measured, and the sections below are
against the real number.

**42.141 ms/token - 23.73 t/s - at depth 4096, tg 256, median of three runs**
(`tools/bench_decode.sh`, spread 0.08%, `debug_resid` off, `--max-len 16384`).
99.77% of it is inside `execute` + `fence.wait()`; the host spends 97 µs per
token. The roofline on the two measured constants (15.540 GB/token, 590 GB/s)
is 26.34 ms. Full rows in [BENCHMARKS.md](BENCHMARKS.md), the verdict and what
it scopes in [05-perf-model.md](05-perf-model.md).

Four kinds of number appear in the sections below, and **every one of them says
which it is**:

- **per-kernel measured** - `gemv` and `gemv_bf16` (`probe_gemv`, at exactly the
  shapes/layout/`S` the model table binds; read as a **floor**, see that
  section's caveat), `embed_gather` and `argmax` (timing loops in their own
  tests), and `attn`'s early-out cost (the `--max-len` pair in doc 07 #12).
- **estimated** - `attn_decode`'s per-block work, a **two-point extrapolation**
  from a pair of `--bench` runs rather than a timing, and the 0.52 µs/kernel
  dispatch floor from doc 07 #5. Both are now **superseded by in-situ
  measurements** and both are kept where they stand, with the correction beside
  them: this document narrates its own history.
- **derived** - arithmetic over two measurements taken with different
  instruments; the dispatch-gap row below is the one that matters.
- **per-kernel measured, in situ** - since 2026-08-25 **every** launch has one,
  from `b70-decode --profile` (Level Zero kernel timestamps on the replayed
  list, 32 steps at depth 4096). The full anatomy, the method and the caveats
  are [15-step-anatomy.md](15-step-anatomy.md); this section carries the result.

**The aggregate bucket is gone.** It read: "`prep` and `gdn_step` have no
per-kernel number … they share an **11.33 ms** bucket with
`attn_prep`/`attn_reduce` and the `a‖b` GEMV, an average of 30.7 µs per launch."
That bucket was the step minus everything that had a number, so it also carried
everyone else's error. Measured in situ it is **6.768 ms across those same 369
launches** (18.3 µs/launch); the other 4.562 ms was GEMV sitting 0.659 ms above
its transplanted floor, `attn_decode` sitting 3.422 ms above its extrapolation,
and 0.473 ms of dispatch gap the old partition had no row for (docs/15).

The rows below are a **partition** of the token: every launch and the host
appear exactly once, and the rows sum to the measured total
(29.005 + 5.782 + 3.573 + 2.335 + 0.733 + 0.127 + 0.017 + 0.473 + 0.097 =
42.142, one µs of rounding), which is why they are quoted to three decimals even
where the prose rounds.

| | launches/token | ms/token | share | kind |
|---|---|---|---|---|
| `gemv` + `gemv_bf16` (`lm_head`) | 257 | **29.005** | 68.8% | **per-kernel measured, in situ** (floor was 28.346) |
| `attn_decode` | 16 | **5.782** | 13.7% | **per-kernel measured, in situ** (estimate was 2.360) |
| `prep` - `res_norm` 129, `silu_mul` 64, `gated_head` 48 | 241 | **3.573** | 8.5% | **per-kernel measured, in situ** |
| `a‖b` GEMV (`gemv_bf16` 5120×128) | 48 | **2.335** | 5.5% | **per-kernel measured, in situ** |
| `gdn_step` | 48 | **0.733** | 1.7% | **per-kernel measured, in situ** |
| `attn_prep` + `attn_reduce` | 32 | **0.127** | 0.3% | **per-kernel measured, in situ** |
| `embed_gather` + `argmax` | 3 | **0.017** | 0.04% | **per-kernel measured, in situ** |
| dispatch gap, un-instrumented | - | **0.473** | 1.1% | **derived** (bench fence − Σ in-situ durations) |
| host, outside the fence | - | **0.097** | 0.23% | measured |
| **total** | **645** | **42.141** | 100% | measured (bench, median of three) |

**98.6% of the step is kernel time inside the fence** (41.571 ms of Σ per-kernel
durations, measured), 1.1% is dispatch and 0.2% is the host. The dispatch row is
now a real row rather than the memo line it used to be - 0.473 ms **derived**
in situ, against the 645 × 0.52 µs = 0.335 ms doc 07 #5 estimated. And
**everything that is not a GEMV** is 42.141 − 29.005 = **13.136 ms** (31.2%),
of which 13.039 is device time and 0.097 is the host; that is the figure doc 05
compares against vLLM's 3.400 ms non-GEMV budget (it was 13.795 when GEMV was
priced at its floor).

---

## `gemv` - int4 g64 × bf16, M ∈ [1,8]

`src/kernels/gemv.cl`, variants `gemv_M<M>_K<K>_N<N>_S<S>_L<LAYOUT>`.

**Computes**

```
out[s][m][n] = Σ_{k ∈ slice s} x[m][k] · (q[k][n] − 8) · scale[k/64][n]
```

`q` is the 4-bit nibble, symmetric with the constant zero point 8 (the
checkpoint's `qzeros` is dropped at load, doc 02); `scale` is one f16 per
64 consecutive `k` per column; `x` is bf16 `[M][K]`; the output is **fp32
partials `[S][M][N]`**, summed by `prep`. Nothing else is in the kernel - no
residual, no activation, no norm - so that split-K stays deterministic and the
GEMV stays one thing (spec §9.2).

### Work assignment, and why

**Lane per `n`.** Lane `l` of a subgroup owns output column `n_tile·16 + l` and
holds its accumulator in a register for the whole K-range. Two things follow.
The weight read is perfectly coalesced by construction - the 16 lanes want 16
adjacent columns, which are adjacent words in *both* layouts. And there is **no
cross-lane reduction anywhere**: no shuffle tree, no SLM, no barrier, and the
epilogue is a single store per lane. The alternative assignment - lane per `k`,
one output per subgroup, reduce at the end - costs a 4-deep shuffle reduction
per output column and buys nothing on a kernel whose entire cost is the weight
stream.

**SIMD16.** `intel_reqd_sub_group_size(16)`; the device offers 16 and 32
(doc 01, `sub_group_sizes [16, 32]`). 16 lanes × 4 bytes is 64 bytes - exactly
one cache line per lane-step, and `intel_sub_group_block_read8` becomes 512
contiguous bytes issued as one instruction. 16 is also the granularity
everything else in the design is cut to: the layout-1 tile is 16 columns wide,
the bf16 tile is 16 columns wide, and the loader interleaves `gate‖up` in
16-column blocks so the lane holding `gate[n]` also holds `up[n]` at a fixed
offset (spec §6.4). OpenVINO's GEMV `#error`s on anything but 16 (doc 08).
SIMD32 was **not measured** - see *Rejected*.

**64 `n` per work-group** (`reqd_work_group_size(64,1,1)` = 4 subgroups). The
work-group is the dispatch unit, and the grid is `(N/64) × S`. Four subgroups
is the smallest group that still hands a subslice a useful bundle of threads per
dispatch; one subgroup per group would quadruple the number of groups to
schedule, and a larger group would coarsen the grid at exactly the shapes where
the grid is already the problem (N = 5120 is 80 groups). It also keeps
`N % 64 == 0` as the only shape constraint, which every fused shape satisfies
and `a‖b` is zero-padded to meet.

**Split-K on grid dimension 1, into separate partials.** Work-group `(gn, s)`
covers k-groups `[s·G/S, (s+1)·G/S)` with `G = K/64`, and writes
`out[s][m][n]`. `prep` sums the `S` partials on the next kernel.

*Why split-K exists:* the grid is `N/16` subgroups at `S = 1`. At N = 5120 that
is **320 subgroups for 256 EUs** - the machine cannot get enough loads in flight
to saturate DRAM. The measurement below says this costs exactly half the
bandwidth, and that `S` recovers it.

*Why partials and not atomics:* a `float` atomic add is order-dependent, so two
replays of the same captured command list would produce different logits. The
runtime's whole premise is one list replayed per token, and the golden-tensor
tests compare bitwise. Determinism is not negotiable for either. The cost is
`S ×` the output bytes, and it varies by an order of magnitude across the
chosen configurations: 0.59% at `gate‖up` (S = 4, 557 KB against 94.7 MB), but
**1.96% at `out/o_proj` (S = 16, 327,680 B against 16.71 MB)** - that is the
bound at M = 1, not the `gate‖up` figure. It is also a *write*, which this
kernel otherwise does not do. At M = 8 the same worst case grows to **~16%**
(2.62 MB of partials), which is the point at which this trade would have to be
re-examined rather than assumed. Atomics would save that and forfeit
reproducibility; today the trade is cheap, at M = 8 and S = 16 it would not be.

### The two layouts

Both hold the same 4.25 bits per weight (nibbles + one f16 scale per 64), so
`weight_bytes` and therefore GB/s are directly comparable.

**Layout 0 - GPTQ-native.** `qweight[K/8][N]` u32 and `scales[K/64][N]` f16
exactly as the checkpoint ships them. A lane covering 64 `k` reads 8 words at
**stride `N·4` bytes**, plus one scale from a second array. Zero repack: the
loader could upload the mapped file verbatim.

**Layout 1 - tiled.** For each `(n_tile of 16, k_group of 64)` one **544-byte
block**: 128 u32 of nibbles ordered `[k_octet 0..7][lane 0..15]`, then the 16
f16 scales (32 B). Tiles are ordered k-group-inner, n-tile-outer. A subgroup
therefore walks **one contiguous run of memory across its entire K-range**, and
one k-group is one `intel_sub_group_block_read8` plus one
`intel_sub_group_block_read_us` - the scales travel with the nibbles instead of
being a second stream. The address stream a subgroup issues does not depend on
`N` at all. That independence is the whole reason the layout exists.

**Which won.** Layout 1, by the probe's rule (higher GB/s summed over the five
int4 shapes), **2713 vs 2667 - 1.7%.** That is inside the noise the rule was
written to tolerate, and the honest reading is this:

| shape | K×N | layout 0 best | layout 1 best | winner |
|---|---|---|---|---|
| out/o_proj | 6144×5120 | **549** | 533 | 0, by 3.0% |
| q‖k‖v | 5120×14336 | **577** | 541 | 0, by 6.7% |
| qkv‖z | 5120×16384 | 419 | **559** | **1, by 33%** |
| gate‖up | 5120×34816 | **555** | 549 | 0, by 1.1% |
| down | 17408×5120 | **567** | 531 | 0, by 6.8% |

**Layout 1 loses four shapes out of five and wins the sum on one.** The one is
`qkv‖z`, and it is the only shape whose `N` is a power of two: at N = 16384
layout 0's row stride is exactly 64 KB, and layout 0 never exceeds 419 GB/s
there at any `S`, while layout 1 hits 559 at `S = 1`. Every other shape has a
non-power-of-two stride (20480, 57344, 139264 bytes) and layout 0 is fine. The
mechanism is almost certainly DRAM channel/bank aliasing on the power-of-two
stride - **that is an inference from the shape of the data, not a measured
cause**; nothing in this project has read a memory-controller counter. What is
measured is that the penalty is real, reproducible, and confined to N = 2^k.

In wall time rather than the rule's GB/s sum, each layout at its own rule-chosen
`S`, one instance of each of the five shapes: **layout 0 465.3 µs, layout 1
449.0 µs - layout 1 is 3.6% faster.** Both metrics agree in direction.

What layout 1 costs, recorded so the next person can revisit it: a load-time
repack pass over 12.16 GB of weights (`repack()`), and the loss of the
zero-copy-from-`mmap` option that layout 0 would have allowed. A 3.6% decode
win pays for a one-off load cost; it would not pay for much else. If the model
mix ever shifts to shapes with no power-of-two `N`, this decision is worth
re-running - the probe is the arbiter, not this paragraph.

### Rejected, and what was actually measured

Measured, and only this: **layout 0 vs layout 1** and **S ∈ {1,2,4,8,16}**, at
all five production shapes, **at M = 1**, twice. Everything below was decided by
argument. Saying so is the point of this section.

**`M` is the biggest unmeasured axis.** The kernel is parameterised on
M ∈ [1,8] and only `gemv_M2_K5120_N5120_S1_L0` is even compiled today; nothing
above M = 1 has been timed. The `S` picks in the table below are therefore
**M = 1 picks**, and they will not simply carry over: the split-K partials scale
with M, so `out/o_proj` at S = 16 goes from 1.96% of weight bytes at M = 1 to
**~16% at M = 8**, and the rule would very likely choose a smaller `S` there.
Re-run `probe_gemv` over M before trusting any of this for speculative decode.

- **Shuffle-broadcast of activations (OpenVINO's pattern) vs same-line vector
  loads - NOT measured.** OpenVINO's GEMV holds `x` in registers and
  `sub_group_broadcast`es it to all 16 lanes. This kernel instead does a plain
  `vload8` from `x + m·K + g·64 + j·8`: all 16 lanes issue the *same* address,
  which the hardware serves from one cache line. The argument for not measuring
  it was that `x` is 10 KB at K = 5120, is L1-resident after the first touch,
  and is re-read against a weight stream three orders of magnitude larger. The
  measured 88-97% of peak on the shapes that fill the device is consistent with
  the activation path not being on the critical path - but it is consistency,
  not proof. If a profile ever shows otherwise, this is the first thing to try.
- **SIMD32 - not measured.** The device supports it. It would make a subgroup
  own 32 `n`, halve the subgroup count at every shape (N = 5120 → 160), and
  make the fill problem worse in exactly the place that already hurts. Rejected
  on that reasoning alone.
- **fp32 accumulation vs OpenVINO's `half` - not measured.** fp32 was chosen
  because the split-K partials are summed by another kernel and bf16
  activations already spend the precision budget. The observed `max_abs_err`
  against a double-precision CPU reference is 4.2e-05 at worst (K = 17408),
  against a tolerance of 6.7e-03 - there is room, but no measurement of what
  `half` would cost in either accuracy or speed.
- **More than one `n` per lane / K-unrolling beyond the 64-wide group - not
  measured.** The k-group is 64 because the *scale* group is 64; unrolling
  further would need a second scale in flight for no obvious gain on a
  bandwidth-bound loop.
- **Atomics instead of partials - rejected on determinism, not measured.** The
  traffic argument says it would be worth 1.96% at most at M = 1 (`out/o_proj`,
  S = 16) and ~16% at M = 8 - see above.

### Measured

`probe_gemv`, **2026-08-24**, on the box (`CL_DRIVER_VERSION 26.27.39122.14`,
kernels AOT-compiled by `ocloc -device bmg-g31`, host `-O2`), weights random.
Timing: each configuration is 40 launches recorded in one regular command list
and replayed 8 times, median of the last 5, `µs = list time / 40`. The list
cycles through `NB = max(2, 72 MB / weight_bytes + 1)` identical weight copies
at different addresses, so consecutive launches miss the 24 MB L2 and the number
is DRAM bandwidth, not cache bandwidth. GB/s counts **weight bytes only**
(nibbles + f16 scales). The activation vector is 10 KB, read once into cache;
the fp32 partials written are **0.15-2.0% of weight bytes at the chosen `S`,
worst at `out/o_proj` S = 16** (327,680 B against 16.71 MB) - not "under 1%",
and worth remembering when reading a GB/s figure that ignores them. Correctness
is checked against a double-accumulating CPU reference at **every** one of the
50 int4 configurations - all passed.

```bash
tools/box.sh run "./build/tools/probe/probe_gemv" | tee docs/probe-gemv-2026-08-24.md
```

Best `S` per shape in the canonical layout 1, and the `S = 1` row it is measured
against:

| shape | K×N | subgroups at S=1 | S=1 GB/s | chosen S | GB/s | % of 600 | µs |
|---|---|---|---|---|---|---|---|
| out/o_proj | 6144×5120 | 320 | 259 (43%) | **16** | 533 | 89% | 31.3 |
| q‖k‖v | 5120×14336 | 896 | 540 (90%) | **1** | 540 | 90% | 72.2 |
| qkv‖z | 5120×16384 | 1024 | 559 (93%) | **1** | 559 | 93% | 79.7 |
| gate‖up | 5120×34816 | 2176 | 426 (71%) | **4** | 536 | 89% | 176.7 |
| down | 17408×5120 | 320 | 261 (44%) | **16** | 531 | 89% | 89.1 |
| `lm_head` bf16 | 5120×248320 | 15520 | 584 (97%) | - | 584 | 97% | 4350.5 |

`S` is the smallest value within 3% of that shape's best (spec §4.2). The full
51-row matrix, verbatim, is committed as
[probe-gemv-2026-08-24.md](probe-gemv-2026-08-24.md).

**What that costs per token - and it is two thirds of the step.** The engine
binds exactly these rows: layout 1 everywhere, `S` = 16 / 1 / 1 / 4 / 16, the
same picks the probe made (`model::Qwen35`'s table, `qwen35.cc`). So the probe's
µs are the engine's µs, and multiplying by the layer counts
`model::Qwen35::layers()` produces is a *per-kernel measured* attribution rather
than a model:

| linear | K×N | `S` | layers | µs each | ms/token |
|---|---|---|---|---|---|
| `in_proj_qkv‖z` | 5120×16384 | 1 | 48 GDN | 79.7 | 3.826 |
| `out_proj` | 6144×5120 | 16 | 48 GDN | 31.3 | 1.502 |
| `q‖k‖v` | 5120×14336 | 1 | 16 FA | 72.2 | 1.155 |
| `o_proj` | 6144×5120 | 16 | 16 FA | 31.3 | 0.501 |
| `gate‖up` | 5120×34816 | 4 | 64 | 176.7 | 11.309 |
| `down` | 17408×5120 | 16 | 64 | 89.1 | 5.702 |
| `lm_head` (bf16, next section) | 5120×248320 | - | 1 | 4350.5 | 4.351 |
| **total, 257 launches** | | | | | **28.346** |

**28.35 ms of the measured 42.14 ms step - 67.3%** - against a 26.34 ms
roofline for the whole token. Two readings, both uncomfortable in the right
direction:

- **These kernels are close to done.** 28.35 ms is 107.6% of the roofline for
  *all* the model's bytes, which is what 89-97% of 600 GB/s comes to. There is
  ~2.0 ms of headroom in the entire GEMV family and it is the last place to
  look, not the first.
- **They are also 89% of vLLM's whole 31.75 ms token.** Everything that is not
  a GEMV therefore has a 3.4 ms budget if this engine is to beat 31.50, and it
  currently spends 13.79 (doc 05). The 48 `a‖b` bf16 GEMVs are the one member of
  this family with no measurement at all - 2 work-groups each, never in the
  probe matrix, and named as a suspect for that reason.

The one caveat on transplanting probe µs into a step: the probe replays 40
*independent* launches in an in-order list, as the engine does, and cycles ≥ 72
MB of weight copies so the 24 MB L2 is missed - which is also the engine's case,
since a token reads 15.5 GB of distinct weights exactly once. What the probe
cannot see is a stall the *previous* kernel leaves behind, so read 28.35 as a
floor for the GEMV family, not an exact charge.

**In situ, measured 2026-08-25 - the floor was right to 2.3%.** `b70-decode
--profile --depth 4096 --steps 32` times every launch inside the replayed list
(method and caveats: [15-step-anatomy.md](15-step-anatomy.md); the per-kernel
durations exclude the profiler's own flush, which is what makes them comparable
to the probe's). The same 257 launches cost **29.005 ms**, against the 28.346 ms
transplanted above:

| shape | `S` | probe µs | **in-situ µs** | delta | in-situ ms/token |
|---|---|---|---|---|---|
| `qkv‖z` 5120×16384 | 1 | 79.7 | **81.401** | +2.1% | 3.907 |
| `out_proj` / `o_proj` 6144×5120 | 16 | 31.3 | **33.499** | **+7.0%** | 2.144 |
| `q‖k‖v` 5120×14336 | 1 | 72.2 | **73.550** | +1.9% | 1.177 |
| `gate‖up` 5120×34816 | 4 | 176.7 | **182.377** | +3.2% | 11.672 |
| `down` 17408×5120 | 16 | 89.1 | **89.513** | +0.5% | 5.729 |
| `lm_head` (bf16, next section) | - | 4350.5 | **4375.557** | +0.6% | 4.376 |
| **total, 257 launches** | | **28.346** | | **+2.3%** | **29.005** |

So the stall-from-the-preceding-kernel effect the probe could not see is real
and small. `out/o_proj` is the only shape where it is worth naming: +7.0%, and
it is the shape entered directly out of `attn_reduce` / `prep_gated_head`, the
two kernels in the step with the least data in flight. Everything the "read this
as a floor" caveat implied still holds - it was just worth 0.659 ms, not the
several the unsplit bucket could have hidden. Two consequences recorded in
docs/15: the bucket shrinks by exactly that 0.659 ms, and an `S` retune has at
most 0.659 ms of in-situ excess to chase across six shapes.

**Re-measured 2026-08-24 after a shape correction.** The 2026-08-23 run had
`out/o_proj` at (5120, 5120); the model's output projections are 6144 → 5120
(doc 03), and spec §4.2 was wrong. The whole matrix was re-run at the corrected
shape so that no number here is a mix of two sessions. **The decision did not
change:** canonical layout still 1, `S` still 16 / 1 / 1 / 4 / 16, the layout
margin still 1.7% (2713 vs 2667). `out/o_proj` grew 20% in weight bytes
(13.93 → 16.71 MB) and behaves the same: 320 subgroups at `S` = 1, 259 GB/s
there, 533 at `S` = 16.

Four things worth reading off it:

1. **Split-K is worth ~2× exactly where the spec said it would be.** Both
   N = 5120 shapes double: 259 → 533 GB/s (2.06×) and 261 → 531 (2.03×). The
   prediction that N = 5120 is the worst fill case is confirmed.
2. **Split-K is not universal.** At N = 14336 and N = 16384 the grid already
   fills the device and `S = 1` is the *best* setting, not merely adequate -
   splitting there costs a few percent. A kernel that always split K would be
   leaving that on the floor.
3. **Subgroup count is necessary, not sufficient.** `gate‖up` has 2176
   subgroups at `S` = 1 - more than twice `qkv‖z`'s 1024, which is already at
   93% - and still only reaches 71% until `S` = 4. So "fill the device" is the
   right first-order story for N = 5120 but does not explain everything the
   probe sees, and this document does not pretend otherwise.
4. **The `S` curve is not monotonic, and mostly the bumps are real.**
   `out/o_proj` reads 259 / 451 / 467 / 453 / 533 across S = 1..16 and `down`
   reads 261 / 470 / 482 / 447 / 531. A second full run picks the identical `S`
   everywhere, agrees to a median of 0.21% per cell, and reproduces
   `out/o_proj`'s dip at S = 8 exactly (259 / 451 / 465 / 454 / 533). One cell
   does **not** reproduce: `down` layout 1 S = 4 reads 482 in run 1 against 429
   in run 2 - an 11.0% spread, and the 2026-08-23 run read 431 there, so run 1
   is the outlier. It decides nothing (`down` picks S = 16 in every run) and is
   recorded rather than smoothed. Something about how `(N/64) × S` groups land
   on 32 subslices is being sampled here. It is not explained, and the rule does
   not need it explained - but do not "clean up" these numbers by assuming
   bigger `S` is monotonically better, and do not treat a single cell as
   settled at better than ~10% without a second run.

---

## `gemv_bf16` - bf16 × bf16, M ∈ [1,8]

`src/kernels/gemv_bf16.cl`, variants `gemv_bf16_M<M>_K<K>_N<N>`. Two users:
`lm_head` (N = 248320, **2.54 GB read per token**, 16.4% of `W`) and the GDN
`in_proj_a ‖ in_proj_b` projection (N = 96, zero-padded to 128).

**Computes**

```
out[m][n] = Σ_k x[m][k] · w[k][n]        (w bf16, no scales, no split-K)
```

### Work assignment, and why

Identical skeleton to `gemv`: lane per `n`, SIMD16, 64-`n` work-groups, fp32
accumulator in a register, one store per lane, no cross-lane traffic. The
reasoning is the same and is not repeated.

**What differs is that there is no `S`.** N = 248320 gives 3880 work-groups and
**15520 subgroups** against 256 EUs - the grid fills the device on `N` alone,
which is what spec §4.2 predicted. The measurement settles it rather than
arguing it: **584 GB/s, 97% of the 600 GB/s roofline denominator and 99% of the
590 GB/s that `probe_bw` measures through the same launch path.** There is no
headroom left for split-K to recover, so the `S` parameter was never added, and
the partial-sum buffer it would need (S × 248320 fp32 = 1 MB per slice) never
exists. The other user, `a‖b`, is 2 work-groups (8 subgroups) over 1.3 MB - far too
small to be worth a second kernel or a second code path.

### Layout

One layout only: the canonical bf16 tile `[n_tile][k_octet][8 k][16 n]` ushort.
One `intel_sub_group_block_read_us8` per subgroup per 8 `k` streams 256
contiguous bytes; the subgroup's whole K-walk is contiguous, the same property
that decided the int4 layout.

There is deliberately **no layout-0 equivalent to compare against, and none was
measured.** The checkpoint ships `lm_head` row-major `[N][K]`; reading that with
lane-per-`n` would give each lane a stride of `K·2 = 10240` bytes, which is the
pathological access pattern layout 1 exists to avoid - on the single largest
tensor in the model. Building it to lose was not worth the ocloc variants. The
probe therefore reports one bf16 row, and it is a bandwidth number, not a
comparison.

### Rejected

- **Split-K - not built, not measured.** Argued away by the 97% above. If a
  future model puts a small-`N` bf16 matrix on the decode path, this conclusion
  does not transfer; re-measure.
- **A separate small-`N` kernel for `a‖b` - rejected.** 1.3 MB and 2
  work-groups against a 15.52 GB/token budget. Zero-padding 96 → 128 at load
  costs 33% of a rounding error.
- **Quantising `lm_head` to int4/int8 - deferred, not a kernel question.** It
  would cut this row from 2.54 GB to 0.66 GB (int4) or 1.27 GB (int8) and is
  worth ×1.14 on the roofline (doc 05 §1) - **a far bigger lever than anything
  in this kernel**, which is already at 97% of what the memory system can do.
  Gated on the oracle existing and on an accuracy check (doc 07 #6).

### Measured

Same probe, same date, same command, and **M = 1 only** - like `gemv`, this
kernel is written for M ∈ [1,8] but has never been timed above M = 1, and only
the M = 1 variants are compiled. `w.bytes() = 2.54 GB`, 10 timed launches
cycling 2 weight copies (5.09 GB device-resident):

| shape | K×N | µs | GB/s | % of 600 |
|---|---|---|---|---|
| `lm_head` bf16 | 5120×248320 | 4350.5 | **584** | **97%** |

For scale: at 584 GB/s `lm_head` alone is **4.35 ms** - 16.5% of the 26.34 ms
roofline token for 16.4% of the bytes, and **10.3% of the 42.141 ms step
measured 2026-08-25**. That is why item 1 of doc 05's specialisation list is
`lm_head`, not a kernel: at int4 it would be ~1.1 ms.

**The other user, `a‖b` (5120×128), had never been timed - now it has.** It is
still not in the probe matrix (the matrix covers the five int4 shapes and
`lm_head`); what timed it is `b70-decode --profile`, in situ, 2026-08-25
(docs/15). It runs **48 times per token** at 2 work-groups / 8 subgroups over
1.31 MB, and the paragraph this replaces guessed the direction correctly:

**48.640 µs per launch, 2.335 ms per token, 5.5% of the step - measured, in
situ** (5.6% of the sum of kernel durations, which is the denominator docs/15's
rollups use)**.** That is **27.0 GB/s**, 4.6% of the 590 GB/s the device does, and **21×**
the 0.11 ms its traffic is worth at full bandwidth. Per work-group it is
13.5 GB/s, which docs/15 shows is simply what one work-group can pull on this
device (`prep_res_norm` at one work-group reads 17.0 GB/s, `prep_silu_mul` at
five reads 12.2 GB/s each). The kernel is not slow; it has been given 2 of 32
subslices. `lm_head`, the same kernel at N = 248320, runs at 97% of bandwidth.

doc 05 named this the second suspect for the aggregate bucket and it was the
right call: it is the bucket's largest single member after `prep_res_norm`.
Widening it - `COLS_PER_WG` 16 for tiny N, giving 8 work-groups at bit-identical
per-column arithmetic - is spec 1.5's lever L2 (docs/15's ladder, rank 3, first
to execute).

---

## `prep` - the three between-GEMV kernels

`src/kernels/prep.cl`, one file with three entry points and five compiled
variants: `prep_res_norm_M1_K5120_SP{0,16}` (plus `prep_res_norm_M2_K5120_SP16`,
compile-only, spec 1 §9's M-loop rule), `prep_silu_mul_M1`,
`prep_gated_head_M1`. Everything between two GEMVs of a decode step lives here:
sum the previous GEMV's split-K partials, add the residual, normalise, activate.

### Rounding discipline - the substance of these kernels

The oracle is torch, and **torch rounds per op**: a linear's output is bf16, and
every elementwise op widens to fp32 internally and rounds its result back to
bf16. These kernels match that discipline wherever it is cheap, so that the
engine's residual stream stays comparable to the golden tensors op for op:

- a GEMV's split-K partials are summed in fp32 and rounded to bf16 **once** -
  that bf16 value *is* the linear's output in the reference;
- the residual add is `rne_bf16(f32(resid) + f32(mixer_b))` - bf16 in, bf16 out,
  exactly torch's bf16 add;
- a norm widens the bf16 input to fp32, multiplies by the **fp32 `(1 + w)`**
  weight (the loader's bake, docs/13-loader.md) and rounds the result to bf16 -
  the reference's `type_as(x)`;
- **inside-op** accumulation (the variance sum) stays fp32 and is *not* matched
  to torch term for term. That residual drift is what the golden gate's design
  absorbs: tokens exact is the gate, tensor cosines are diagnostics.

No kernel here deviates from that discipline.

The payoff is that `tests/kernels/prep_ref.h` is not an approximation but the
same op chain in the same order, so `prep_test` asserts **bit-exact** equality
rather than a tolerance - which is a far sharper instrument. Two spellings had
to be pinned to get there, and both are copied verbatim into the reference:

1. **`1.0f / sqrt(x)`, never `rsqrt(x)`.** `rsqrt` is a ~2 ulp approximation
   with no cross-implementation guarantee, so nothing bit-exact can be built
   through it. Correctly rounded `sqrt` followed by a correctly rounded divide
   is reproducible on the host (controller ruling, 2026-08-25 - the same rule
   binds the GDN and FA kernels that come later).
2. **The square-accumulate is an explicit `fma`.** `sum += v*v` may or may not
   be contracted into a fused multiply-add by either compiler; writing the
   fusion explicitly on both sides removes the question instead of relying on
   two different compilers' `-ffp-contract` defaults agreeing.

…and one **build option**, which is the finding of this task:
**`-cl-fp32-correctly-rounded-divide-sqrt` is required on any kernel that is
compared bit-exactly and divides or takes a square root.** OpenCL's default
allows **2.5 ulp** on both `/` and `sqrt`, and this is not theoretical: without
the flag the device's `1/sqrt(mean + 1e-6)` came out **1-2 ulp below** the
host's, and one `x_out` element in 5120 sat close enough to a round-to-nearest
tie (fp32 product `0x3FB88001`, one ulp above the tie) to fall on the other side
of it. 5119/5120 exact - which is exactly the kind of "almost" a tolerance would
have hidden. `add_ocloc_kernel` grew an `OPTIONS` parameter for it (and refuses
`-cl-denorms-are-zero`, which stays forbidden project-wide, docs/13-loader.md).

It started opt-in per kernel, to leave the already-measured `gemv` binaries
untouched. **It is now the project-wide default** (`cmake/ocloc.cmake`,
controller ruling 2026-08-25, plan 3 Task 4): `gdn_step` is the second kernel to
need it and "the next kernel that grows a divide must remember the flag" is not
a rule anyone can be relied on to follow. Making it the default is safe because
the kernels without a divide or a `sqrt` cannot be affected by it - every
binary was rebuilt and all 18 tests, `gemv` and `gemv_bf16` included, came back
byte-for-byte green.

**The variance tree order** is stated identically in `prep.cl` and `prep_ref.h`,
because the comparison is only exact if both walk it the same way: each of the
work-group's WG lanes accumulates its own strided slice - lane `i` takes
`k = i, i+WG, i+2·WG, …` in ascending `k` - into one fp32 register with `fma`,
writes it to SLM, and then a fixed pairwise tree collapses SLM: for
`stride = WG/2, WG/4, …, 1`, lane `i < stride` does `red[i] += red[i + stride]`,
with a barrier after every step (so 256 → 128 → 64 → … → 1 at WG = 256). No
data-dependent branch and no atomic anywhere, so a replayed command list gives
the same bits - the acceptance every kernel in this plan is held to.

### `prep_res_norm` - residual add + RMSNorm

```
mixer_b = rne_bf16(Σ_s partials[s][m][k])            (skipped when S_PREV == 0)
r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))  (S_PREV == 0: r_b = resid)
resid[m][k] = r_b                                    (the residual stream)
rstd    = 1 / sqrt(mean_k(f32(r_b)²) + 1e-6)
x_out[m][k] = rne_bf16(f32(r_b) · rstd · norm_w[k])
```

`partials` is fp32 `[S_PREV][M][K]` from the previous GEMV, `resid` is bf16
`[M][K]` read and written in place, `norm_w` is the fp32 `(1 + w)` weight, and
`x_out` is bf16 `[M][K]` - the next GEMV's activation input.

**One work-group of 256 per token; grid `(1, M)`.** RMSNorm's mean is over the
whole row, so the row cannot be split across work-groups without a second
kernel; the work-group *is* the reduction domain. Phase 1 folds the partials in
and stages the row in SLM as fp32, phase 2 reduces, phase 3 rescales - barriers
between phases, and phase 3 re-reads the row from **SLM** instead of re-widening
`resid` from DRAM.

**SLM bound: `K·4 = 20 KB` for the row plus 1 KB for the reduction = 21 KB**, at
K = 5120. That is the number that decides the shape of this kernel: it fits
comfortably inside the 64 KB a work-group may allocate, so staging the row costs
nothing anyone can feel, and one work-group per token stays possible at every
hidden size this model family uses. At M = 1 exactly one work-group runs - see
*Rejected* for what that costs.

### `prep_silu_mul` - the MLP activation

```
gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16
g_b = rne_bf16(Σ_s partials[s][m][gflat]) ;  u_b likewise
s_b = rne_bf16(silu_f32(f32(g_b)))
x_out[m][k] = rne_bf16(f32(s_b) · f32(u_b))
```

with `silu(x) = x / (1 + exp(-x))`, plain `exp` (not `native_exp`). The index
arithmetic is the loader's `cols_interleave16`: `gate‖up` is one fused linear
whose columns interleave in 16-wide blocks, precisely so that the lane holding
`gate[n]` also holds `up[n]` at a fixed offset (docs/13-loader.md). This kernel
is the consumer that pays for that layout - its two loads are 64 bytes apart.

**Grid `(5, M)`, WG 256, no reduction, no SLM, no barrier.** Five chunks of 4096
cover 17408; the last chunk covers 1024 and the `k < k1` bound is what makes
that safe. Chunking rather than one giant work-group is free here (there is
nothing to reduce) and gives the machine five work-groups per token instead of
one.

### `prep_gated_head` - `Qwen3_5RMSNormGated`, one v-head per work-group

```
o_b = rne_bf16(gdn_o[m][h][i])                    (recurrence output → bf16)
z_b = rne_bf16(Σ_s qkvz_partials[s][m][10240 + h·128 + i])
var = mean_i(f32(o_b)²)                           (128-lane tree)
n_b = rne_bf16(f32(o_b) · (1 / sqrt(var + 1e-6)))
t_b = rne_bf16(f32(gated_w[i]) · f32(n_b))
x_out[m][h·128+i] = rne_bf16(f32(t_b) · silu_f32(f32(z_b)))
```

This is doc 03's op chain exactly - norm → cast → `×w` → `×silu(z.float())` →
cast. `gated_w` is **bf16 plain `w`, no `+1`**: the one norm in the model
without the increment and the one whose weight stays bf16, because the reference
multiplies in the bf16 domain (`loader/small_layout.h`, `kGdnOffGatedNorm`).

**Grid `(48, M)`, WG 128 - one work-group per (v-head, token), one lane per
channel.** The head dimension *is* the reduction domain (128), so the mapping is
forced and pleasant: the variance tree is exactly the work-group's 128 lanes
with one term each, which is also why there is no `fma` in this kernel's
reduction - a lane's contribution is a single multiply.

**The silu factor is deliberately the last op.** Everything through `t_b` is a
rounded scalar chain the host reproduces exactly; only the final product carries
`exp`'s slack. That ordering is what lets the test hold the norm to the exact
bar while allowing the gate a tolerance.

### What the test asserts

`tests/kernels/prep_test.cc`, against `prep_ref.h`, on random inputs
(partials ~ N(0,1), residual bf16, `norm_w = 1 + U(±0.05)`):

| case | bar | result |
|---|---|---|
| `prep_res_norm` SP=0 - `x_out`, `resid` | **bit-exact** | 5120/5120, 5120/5120 |
| `prep_res_norm` SP=16 - `x_out`, `resid` | **bit-exact** | 5120/5120, 5120/5120 |
| `prep_silu_mul` random | ≤ 2 bf16 ulp | 17408/17408 exact |
| `prep_silu_mul`, silu argument 30.0 | **bit-exact** | 17408/17408 |
| `prep_gated_head` random | ≤ 2 bf16 ulp | 6144/6144 exact |
| `prep_gated_head`, silu argument 30.0 | **bit-exact** | 6144/6144 |

The 2 ulp tolerance exists because OpenCL allows **3 ulp** on fp32 `exp` where
the host's `expf` is ~0.5 - it is a contract, not an observation: today's driver
matches the host on every one of the 23552 silu values tested. It is kept
because a driver update may change `exp` and must not fail this test, while a
change in the *arithmetic* still will.

The bar on the silu-carrying kernels is held up by the two extra cases: with the
silu argument forced to **30.0f**, `exp(-30) ≈ 9.4e-14` is far below `2^-24`, so
`1 + exp(-30)` is exactly `1.0f` in fp32 on *any* conforming implementation and
`silu(30) = 30.0f` on both sides. Those cases are asserted **bit-exact**, which
pins `s_b`, and pins the gated head's `n_b`/`t_b` chain - the norm intermediates
that are otherwise not directly observable, since only `x_out` leaves the
kernel.

### Rejected, and what was not measured

**Nothing here has been timed.** There is no `probe_prep`, no wall-clock number
in this section, and the arithmetic below is arithmetic - say so rather than
implying a measurement. What *is* measured is bit-exactness (above) and the
0.52 µs/kernel launch floor (doc 07 #5) these kernels are compared against.

- **Fusing the norm into the following GEMV's prologue - rejected for this
  plan** (spec 1 §9.2 history, doc 04's fusion list item 1). It would remove
  ~128 launches per token. Two numbers say do not bother yet: the per-kernel
  cost inside a replayed list is **0.52 µs**, so the whole between-GEMV kernel
  count - 129 `prep_res_norm` + 64 `prep_silu_mul` + 48 `prep_gated_head` = 241
  launches - is **~0.125 ms**, which was 0.5% of the ~26 ms roofline step this
  argument was made against and is **0.3% of the 42.141 ms step measured
  2026-08-25**; and the traffic these kernels
  add is ~89 MB per token (below) against the token's 15.52 GB of weights,
  **0.58%**. Fusion also has a real cost: the norm's reduction is over the whole
  row, so a fused prologue would need every GEMV work-group to either redundantly
  reduce the row (N/64 = 80 work-groups each summing 5120 elements) or take a
  cross-work-group barrier the split-K design deliberately does not have. Spec
  §4.1's rule - fuse at ≥ 3 µs, do not below 1 µs - puts this below the line.
- **One work-group per token in `prep_res_norm` is the risk in this design, and
  it is unmeasured.** At M = 1 the whole kernel is a single work-group on a
  single Xe-core pulling `S_PREV·K·4` = **320 KB of partials** at S = 16. One
  core cannot approach the 600 GB/s roofline, so the honest expectation is
  latency-bound single-digit µs, not the 0.63 µs the traffic arithmetic gives.
  If a profile ever shows this on the critical path, the fix is a two-stage
  reduction (a grid-wide sum-of-squares kernel, then a small finish kernel) or
  folding the partial sum into the GEMV epilogue - both trade the launch count
  this design saves. Not built, not measured; recorded so it is not rediscovered.
- **Summing split-K partials with atomics in the GEMV instead of here -
  rejected on determinism**, same argument as `gemv`'s section: a float atomic
  add is order-dependent and two replays would differ. That decision is what
  gives these kernels their `partials` argument in the first place.
- **Subgroup reductions (`sub_group_reduce_add`) instead of the SLM tree - not
  used.** The SLM tree's order is *stateable*, and the whole bit-exactness
  contract rests on the reference reproducing it; a subgroup reduce's internal
  order is the compiler's business. Faster, probably, on a kernel whose
  reduction is not the bottleneck. Not measured.
- **`native_exp` / `native_rsqrt` - rejected outright.** Both would put the
  residual stream somewhere the host cannot follow, for an activation that is a
  rounding error's worth of the step.

### Traffic per token (arithmetic, not a measurement)

| kernel | calls/token | bytes/call | total |
|---|---|---|---|
| `prep_res_norm` | 129 (2 per layer + the final norm; layer 0's is the SP = 0 variant, 51,200 B) | 378,880 at SP = 16 (327,680 partials + 20,480 resid r+w + 20,480 `norm_w` + 10,240 out) | 48.5 MB |
| `prep_silu_mul` | 64 | 591,872 (557,056 partials + 34,816 out) | 37.9 MB |
| `prep_gated_head` | 48 | 61,696 (24,576 `z` + 24,576 `gdn_o` + 256 `w` + 12,288 out) | 3.0 MB |

**≈ 89.4 MB per token, 0.58% of the 15.52 GB the weights cost** - 0.15 ms at
the roofline, against 0.125 ms of launch overhead for the same 241 kernels. The
split-K partials are two thirds of it, which is the price recorded in `gemv`'s
section for keeping the replay deterministic.

### Measured - per kernel, in situ (2026-08-25); the suspect was half right

**The bound this section used to carry, kept because it is what the measurement
answered.** What existed after the first 2026-08-25 measurement was this: of a
measured 42.141
ms/token step, 28.346 ms is per-kernel-measured GEMV, 2.31 ms is *estimated*
attention block work, 0.008 ms is `embed_gather` + `argmax`, 0.046 ms is the
attention grid's early-out and 0.097 ms is host time outside the fence. That
leaves **11.33 ms across 369 launches** - `prep`'s 241, `gdn_step`'s 48,
`attn_prep`/`attn_reduce`'s 32 and the 48 `a‖b` GEMVs - an average of **30.7 µs
per launch**. `prep` owns 241 of those 369. (The ±0.2 ms of uncertainty in the
attention extrapolation moves in and out of this bucket; nothing below turns on
it.)

Against that bound the traffic arithmetic above is a **75× under-prediction**:
89.4 MB is 0.15 ms at the roofline, and the bucket `prep` sits in costs 11.33.
Even sharing it three ways, these kernels are nowhere near bandwidth-bound. The
"Rejected" note above predicted exactly this and named the mechanism:

> **One work-group per token in `prep_res_norm` is the risk in this design, and
> it is unmeasured.** … One core cannot approach the 600 GB/s roofline, so the
> honest expectation is latency-bound single-digit µs, not the 0.63 µs the
> traffic arithmetic gives.

129 launches per token, each one work-group on one Xe-core pulling 320 KB of
split-K partials. At 40 µs apiece that is 5.2 ms - 12% of the step and half the
gap to vLLM, from the kernel this document already flagged. **That was a
hypothesis with a named mechanism, not a measurement**, and doc 05 put "profile
these three kernels" as spec 1.5's first task for precisely that reason.

**The measurement, per kernel, in situ.** `b70-decode --profile --depth 4096
--steps 32`, 2026-08-25, on the replayed decode list - method and caveats in
[15-step-anatomy.md](15-step-anatomy.md):

| kernel | calls/token | work-groups per call | **µs/call** | **ms/token** | share of step | effective GB/s |
|---|---|---|---|---|---|---|
| `prep_res_norm` (SP16) | 128 | **1** | **22.323** | 2.857 | 6.8% | **17.0** |
| `prep_res_norm` (SP0, layer 0) | 1 | **1** | **13.620** | 0.014 | 0.03% | 3.8 |
| `prep_silu_mul` | 64 | 5 | **9.733** | 0.623 | 1.5% | 60.8 |
| `prep_gated_head` | 48 | 48 | **1.640** | 0.079 | 0.19% | 37.6 |
| **the family** | **241** | | | **3.573** | **8.5%** | |

**The mechanism was exactly right; the magnitude was 2.3× too big.** 22.3 µs,
not 40; 2.871 ms for `prep_res_norm`, not 5.2. And the traffic arithmetic above
is a **24× under-prediction** for the family (89.4 MB is 0.15 ms at the
roofline; the family costs 3.573), not the 75× the bucket bound allowed for.

What the in-situ numbers add that no bound could: **the per-work-group ceiling.**
17.0 GB/s on one work-group, 12.2 GB/s each on `prep_silu_mul`'s five, and
13.5 GB/s each on the `a‖b` GEMV's two (`gemv_bf16` → Measured). A single Xe-core
cannot keep enough loads in flight to beat ~15 GB/s, so **~40 work-groups are
needed to saturate 590 GB/s** - and `gdn_step`, at 192, reaches 540. That makes
`prep_res_norm`'s fix predictable rather than hopeful: the two-stage reduction is
not a guess about latency, it is a work-group count. It is spec 1.5's lever L1
(docs/15's ladder, rank 2), estimated at 1.5-2.4 ms.

Worse than bandwidth, in fact: the 320 KB of partials `prep_res_norm` folds were
written by the GEMV that ran immediately before it, so most of that 17.0 GB/s is
**L2** traffic. The kernel is not DRAM-limited at all; it is one core's issue
rate.

Note what this does *not* implicate: launch count. 241 `prep` launches × 0.733 µs
(the in-situ per-launch gap, derived - docs/15) is 0.177 ms, 5% of the family and
0.4% of the step. Fusing the norm into the GEMV prologue (doc 04 item 1) would
remove launches, which is not the problem; a two-stage reduction keeps the
launches and removes the serialisation, which is.

---

## `embed_gather` - the token id → the residual stream

`src/kernels/embed_gather.cl`, one compiled variant (`embed_gather_M1`). The
first kernel of a decode step:

```
row            = ctrl[CTRL_CUR + m]        (runtime::Control::cur_token[m])
resid[m][0..K) = embed[row][0..K)          K = 5120, bf16, copied verbatim
```

No arithmetic and therefore no rounding: the embedding table is bf16, the
residual stream is bf16, and the reference op is `embed_tokens(ids)` - a gather,
not a computation. The row moves as raw `ushort`.

### Why it reads the control block instead of taking the id as an argument

This is the kernel that justifies the whole capture-once/replay-per-token
design, so it is worth stating plainly. Level Zero resolves a kernel's arguments
at `zeCommandListAppendLaunchKernel` time, **not** at execute time -
`tests/l0/arg_capture_test.cc` exists to prove that, by binding two different
buffers between two appends of the same `Kernel` object and requiring both to be
written. Good for the timed GEMV harness; fatal for a token id. An id passed as
an argument would be frozen into the list when the list was recorded, and every
replay of that list would gather the same row forever. The alternatives are to
re-record ~650 launches per token, or to drive the step from an immediate list.
Neither has been measured - a captured list exists precisely so that whatever
recording costs is paid once - and neither needs to be, because the id can
simply live in memory the kernel reads when it runs.

So the id lives in the `zeMemAllocShared` control block, whose *contents* are
read when the kernel runs. `embed_gather` is the only reader of `cur_token`
inside the list and `argmax_stage2` is its only writer; the host writes it only
during prompt ingestion. `tests/kernels/embed_gather_test.cc` asserts exactly
this property: one closed list, executed twice with nothing rebound and only
`cur_token[0]` changed between replays, must produce two different rows.

### Work assignment, and why

**Grid (1, M), one work-group of 256 per token.** 5120/256 = **20 elements per
work-item** with lane `i` taking `k = i, i+256, …`, so each iteration is one
fully coalesced 512-byte read and 512-byte write per subgroup. There is nothing
to reduce and nothing to share, so a wider grid would only add launch cost to a
kernel that moves 20 KB.

### The out-of-range convention

`row >= VOCAB` is **not clamped**. An id outside the table cannot come out of
this engine - argmax masks at `kVocabUsed`, and the tokenizer's ids are smaller
still - so it means the control block is wrong, i.e. an engine bug. Clamping
would hide that behind a plausible-looking token and a plausible-looking
continuation. Instead the kernel writes nothing, leaves the residual stream as it
was, and sets `Control::debug_flag = 0xDEAD0001` (the same channel
`check_finite` uses), so the host can see *which* step went wrong. The test
covers both `kVocab` and `0xFFFFFFFF`, checks that `resid` still holds its
prefilled sentinel, and then replays a valid id to show the list still works.

### Rejected

- **`zeCommandListAppendMemoryCopy` instead of a kernel - cannot express this.**
  The copy's source address depends on a value that is not known until the list
  executes, and an appended copy captures its pointers at append time exactly as
  a kernel captures its arguments. The indirection has to happen *on the device*,
  which means a kernel.
- **Passing the id as a kernel argument and re-recording per token - rejected**,
  see above.
- **A wider grid (several work-groups per token) - not built, not measured.** At
  10 KB per token the kernel is launch- and latency-bound; splitting it adds
  launches to save nothing.

### Measured

**3.65 µs per token** - median of 5 replays (of 8, first 3 dropped) of a closed
list holding 512 back-to-back launches, printed by `embed_gather_test`. Two
caveats, both honest: the same list at **64** launches per execution measures
**6.59 µs** per launch, and doc 07 #5's 6.4 µs submit-and-fence floor accounts
for only 0.1 µs of that 2.9 µs gap - so something (device clock ramp within a
short execution is the guess, not a finding) makes short bursts dearer, and the
512-launch figure is the marginal cost on a device already working, which is the
engine's case; and the test's stand-in table is 64 rows (655 KB, L2-resident),
while the real [248320][5120] table is 2.54 GB, so in the engine the row is a
cold 10 KB read.
Either way it is one kernel per token against the **42.141 ms step measured
2026-08-25**: **0.0087%** (it was quoted as 0.014% of the ~26 ms roofline
token), and the 20 KB of traffic is 0.0001% of the token's 15.52 GB of weights.

---

## `argmax` - deterministic greedy sampling, in two fixed stages

`src/kernels/argmax.cl`, two entry points and two compiled variants
(`argmax_stage1_M1`, `argmax_stage2`; both binaries carry both entry points, as
with `prep`). Together they turn one row of `lm_head` logits into the next token
and move the control block on by one step:

```
stage 1  grid (243, M), WG 256 - group g reduces logits[m][g·1024 … +1024)
                                 to one (value, index) pair in part[m][g]
stage 2  grid (1, 1),   WG 256 - folds each active token's 243 pairs, writes
                                 ctrl.out_token[m]; then lane 0 sets
                                 ctrl.cur_token[0] = out_token[n_active-1]
                                 and ctrl.pos += n_active
```

`argmax_stage2` is the **only writer of `pos` and `cur_token` inside the list**.
That is what makes the list self-advancing: the host writes the control block
once at prompt ingestion and then only reads `out_token[0]` after each fence.

### Why two fixed stages and not atomics

A global `atomic_max` over 248320 candidates (or the float compare-exchange loop
it really has to be, since OpenCL has no float atomic max) is **order-dependent
by construction**: which of two equal logits is seen as "already the max"
depends on which work-group arrives first, and that is scheduling, not
arithmetic. Equal logits are not exotic at the top of a peaked distribution -
two spellings of the same word is precisely where greedy decoding is fragile -
so an atomic argmax can emit a different token on two replays of the same list
with the same inputs. The acceptance for every kernel in this plan is that a
replay is bit-reproducible, and the golden gate's bar is *token-exact* output,
so that is disqualifying before any performance argument.

A two-stage tree has no data-dependent order at all: every comparison is between
a fixed pair of SLM slots in a fixed sequence, so the result is a pure function
of the logits. Nor does it look like a performance sacrifice - 243 work-groups
reading 4 KB each, then one reading 2 KB, against an atomic version's serialised
update path - but that is reasoning, not a measurement: the atomic version was
never built, because determinism settles it before speed is asked.

### The comparator, and the tie rule

One function, `argmax_better`, used by both stages and repeated in the test's
host reference:

```
(a.v > b.v) || (a.v == b.v && a.i < b.i)      // strictly greater wins;
                                              // an exact tie takes the LOWER index
```

Because ties resolve to the lower index rather than to "whichever the tree saw
first", the *bracketing* of the reduction cannot change the answer - which is
what lets stage 1 and stage 2 use different tree widths (256 lanes over 1024
logits, then 256 lanes over 243 partials) and still agree with a sequential
host scan. It is also the oracle's rule - `torch.argmax` documents that "if
multiple values equal the maximum of the tensor, the first occurrence of the
maximum value is returned" (torch 2.12 docs, checked 2026-08-25) - so the golden
gate compares like with like on exactly the inputs where greedy decoding is
most fragile. Lanes with nothing to reduce seed `(-INFINITY, 0x7FFFFFFF)`:
`-INF` loses to every real logit and `0x7FFFFFFF` loses every tie, so idle lanes
cannot affect the result (stage 2 has 13 of them, 243 partials into 256 lanes).

### The 248077 mask, and where the number comes from

`lm_head` is [5120][**248320**] because the tiling wants a round row count, but
the tokenizer defines **248077** ids (`model::Qwen35::kVocabUsed`, the
checkpoint config's `vocab_size`; doc 03). Rows 248077..248319 hold whatever the
checkpoint stored there, and their logits are ordinary finite numbers that can
perfectly well be the largest in the row - emitting one would be an id the
tokenizer cannot decode. So a candidate at index `>= VOCAB_USED` contributes
`(-INFINITY, idx)` and can never win. Indices `>= VOCAB` are not read at all:
the row is only `VOCAB` wide, and the last work-group's bound is clamped
(243 × 1024 = 248832 > 248320).

The number is baked as `-D VOCAB_USED=248077` in one CMake line, not spelled in
the `.cl` file, and `argmax_test` plants the row's two largest logits inside the
masked tail - one of them exactly at 248077 - and requires the best *usable*
index to come out.

### Work assignment, and why

**Stage 1: chunk 1024, WG 256, grid (243, M).** 1024 logits per work-group is
`runtime::DecodeBuffers`' `kArgmaxChunk`, and 243 = ⌈248320/1024⌉ is the shape of
`argmax_part` (fp32 [M][243][2]); the two constants are derived from the same
arithmetic on both sides. 243 work-groups over 32 Xe-cores (doc 01) is ~7.6 each
- enough to fill the device without making the stage-2 fold wide enough to need
a third stage. Each lane scans a strided slice into a register pair, then the
same fixed pairwise SLM tree `prep.cl` uses (256 → 128 → … → 1, barrier after
every step) collapses the work-group.

**Stage 2: one work-group, m looped internally.** The fold is 243 pairs per
token - a single tree, and the kernel has to be a single work-group anyway
because it owns the control block's per-step bookkeeping, which exactly one lane
must do. The `m` loop's bound is `ctrl[CTRL_NACT]`, read by every lane, so it is
uniform and the barriers inside are reached by all 256; the barrier at the top of
the loop body is what makes the SLM reuse across `m` safe. `n_active == 0` would
be an empty step (an engine bug) and `out_token[n-1]` would index off the front
of the field, so that case writes nothing rather than something plausible.

**The index travels as a float**, which keeps `argmax_part` a plain fp32 buffer
instead of a struct the host would have to lay out by hand. That is lossless,
not "close enough": every integer below 2²⁴ = 16777216 is exactly representable
in fp32, and the largest index this kernel can emit is 248319 - 67× under the
limit, so `(uint)(float)idx` is the identity for every id in this vocabulary.
Both the `.cl` and this section say so where the conversion happens, because the
day someone reuses this kernel for a 20M-entry table is the day it stops being
true.

### What the test asserts

`tests/kernels/argmax_test.cc`, six cases, all through **one closed list
replayed once per case** (a fresh list per case could not show that `pos`
advances per *execution*):

| case | expected |
|---|---|
| unique max at 123456 | 123456 |
| exact three-way tie at 5000, 5001 (same stage-1 group, different lanes) and 90000 (different group) | 5000 - both trees' tie-breaks exercised |
| the row's two largest logits in the masked tail (248077 and 248200) | 1000, the best usable index |
| all 248320 logits equal | 0 |
| max at 248076, the last usable index | 248076 |
| a random row, against the host reference | agrees (17977) |

and after every replay: `out_token[0]` = the expected id, `cur_token[0]` = the
same id (fed straight back to `embed_gather`), `pos` advanced by exactly
`n_active`, `debug_flag` still 0.

### Rejected, and what was not measured

- **A global atomic max - rejected on determinism**, above. Not measured, and it
  would not matter if it were faster.
- **One work-group scanning all 248320 logits - rejected, not measured.** It
  removes a kernel and the `argmax_part` buffer, but puts 993 KB of streaming
  reads on a single Xe-core out of 32. The launch it saves is 0.52 µs (doc 07
  #5); the serialisation it buys is much more than that.
- **`work_group_reduce_max` / subgroup reductions - not used**, the same
  argument as `prep`'s variance tree: the SLM tree's order is *stateable*, and a
  built-in reduce cannot carry the index alongside the value anyway, so the tie
  rule would have to be rebuilt on top of it.
- **Packing (value, index) into one `ulong` (or the float's low mantissa bits)
  - rejected as unnecessary.** The fp32 pair is exact (above) and keeps
  `argmax_part` a buffer the host can print.
- **Sampling (temperature / top-p / top-k) - out of scope for this plan**, which
  is greedy-only by spec. When it lands it replaces **stage 2 only**: stage 1's
  243 pairs are an argmax-shaped summary and a sampler needs the full row, so the
  right shape is a different stage-1 (a partial softmax) rather than a patch on
  this one. Recorded so it is not discovered the hard way.

### Measured

**4.39 µs per token for both stages** - median of 5 replays (of 8, first 3
dropped) of a closed list holding 512 back-to-back (stage 1 + stage 2) pairs,
printed by `argmax_test`. The same list at 64 pairs per execution measures
**8.88 µs**, the same short-burst penalty `embed_gather` shows and with the same
unexplained cause, so treat this as the marginal cost on a device already
working and not as a constant. The logits sit in one 993 KB buffer that is
re-read every iteration and is therefore L2-resident - which is the engine's
case too, since `lm_head` has just written them. Against the **42.141 ms step
measured 2026-08-25** this is **0.010%** (0.017% of the ~26 ms roofline token it
was first quoted against), and the ~1 MB of traffic per token (993 KB of logits
read, 1944 B of partials written and read back) is 0.006% of the token's
15.52 GB of weights. For scale, `lm_head` itself - the GEMV that produces those
logits - is 4.35 ms, **991× this kernel** (4350.5 µs / 4.39 µs). Sampling is free and the two
token-boundary kernels together (8.04 µs) are the only two numbers in this
document small enough to stop thinking about.


---

## `gdn_step` - the gated delta-rule decode step

`src/kernels/gdn_step.cl`, variants `gdn_step_M1` (used) and `gdn_step_M2`
(spec 1 §9's M-loop rule). **48 of the model's 64 layers run this**, no library
implements it and `sycl-tla` has no example for it: it is the engine's
original-work kernel, and the one that decides whether the golden gate can pass
(doc 03, "GDN dominates the layer count, not the byte count").

One kernel does the whole GDN mixer between the qkv‖z GEMV and
`prep_gated_head`: the depthwise conv1d update and its SiLU, the q/k l2norm,
and the recurrent state update.

```
gdn_step(ctrl, qkvz_partials, ab_out, gdn_small, conv_ring, state, gdn_o)

a_b, b_b = rne_bf16(ab_out[m][h]), rne_bf16(ab_out[m][48+h])
g        = negA[h] · softplus(f32(a_b) + dt_bias[h])      (fp32; negA = -exp(A_log), baked)
beta     = 1 / (1 + exp(-f32(b_b)))                       (fp32)
raw_b    = rne_bf16(qkvz_partials[0][m][ch])              -> conv_ring[(pos+m)%16][ch]
conv     = Σ_{t=0..3} w[ch][t] · f32(window_t)            (window_3 = this token)
x_b      = rne_bf16(silu_f32(conv))
qn_b     = rne_bf16(f32(q_b) · 1/sqrt(Σ q² + 1e-6))  ;  qf = f32(qn_b) · 1/√128
kn_b     = rne_bf16(f32(k_b) · 1/sqrt(Σ k² + 1e-6))  ;  kf = f32(kn_b)
S       *= exp(g) ;  kv[v] = Σ_k S[k][v]·kf[k]
Δ[v]     = (f32(v_b[v]) − kv[v])·beta ;  S[k][v] += kf[k]·Δ[v]
gdn_o[m][h][v] = Σ_k qf[k]·S[k][v]                        (fp32; prep_gated_head rounds it)
```

That is doc 03's GDN block op for op, and `tests/kernels/gdn_ref.h` is the same
chain on the host - trees included, which is the point of the next two sections.

### Work assignment, and why

**Grid `(48 heads, 4 chunks)`, work-group 256 = 16 subgroups × 16 lanes
(SIMD16).** Work-group `(h, c)` owns state columns `[32c, 32c+32)` of head `h`.
With `lid = get_local_id(0)`, `sgid = lid/16` and `lane = lid%16`, work-item
`(sgid, lane)` owns an **8 k × 2 v tile**:

```
          v columns owned by the work-group  (c = 1 shown)
          32 .. 47                48 .. 63
        +------------------------------------------+
k   0.. 7|  sgid = 0 : lane owns columns 32+lane and 32+lane+16
k   8..15|  sgid = 1 : "
k  16..23|  sgid = 2 : "
   …     |  …
k 120..127| sgid = 15: "
        +------------------------------------------+
          16 subgroups × 8 k = 128 k-rows
          16 lanes × 2 v      = 32 v columns
          16 fp32 of state per work-item, in registers
```

Three things chose that shape.

**Why (head, chunk) and not one work-group per head.** 48 work-groups on a card
with **32 Xe-cores** is 1.5 per core: half the machine idle on the layer type
that runs 48 times per token, and no second work-group per core to hide the
state load's latency behind. Splitting the 128 v columns four ways gives
**192 work-groups**, six per Xe-core, at the cost of convolving each head's
channels four times (below). The state slice per work-group is
`128 × 32 × 4 B = 16 KB`, which is 16 fp32 per work-item - registers, not SLM.

**Why the lane index picks the *column* and the subgroup picks the k-band.**
`state` is k-major, so a row's 128 v are contiguous. Under this mapping a
subgroup's 16 lanes read `32c + lane` at a fixed k-row - **16 consecutive fp32,
one full 64 B cache line per access**, and the column pair's second half is the
next line. The obvious alternative (lane picks the k-row) makes every subgroup
access a 16-way gather of 8 B at a 4 KB stride. Same arithmetic, same registers,
an order of magnitude apart on the load that dominates this kernel's traffic.

**Why the tile is 8×2 and not, say, 16×1.** Columns are independent under the
rank-1 update `S[k][v] += kf[k]·Δ[v]`, so the update needs no communication at
all - no atomics, no barrier. Only the two contractions cross work-items, and
for a fixed column they cross exactly the 16 subgroups, which is a 16-wide SLM
tree. Two columns per lane amortise that tree over two results.

**The trees, stated once and obeyed twice** (`gdn_step.cl` and `gdn_ref.h` carry
this text verbatim, as `prep` does):

- `kv[v] = Σ_k S[k][v]·kf[k]` - band `sgid` contributes
  `Σ_{kk=0..7} S[8·sgid+kk][v]·kf[8·sgid+kk]`, accumulated in **ascending kk**
  with an explicit `fma`, into `kv_red[sgid][v−32c]`. The 16 band partials then
  collapse pairwise: for `stride = 8, 4, 2, 1`, work-items with `sgid < stride`
  do `kv_red[sgid][j] += kv_red[sgid+stride][j]`, barrier after each step.
- `o[v] = Σ_k qf[k]·S[k][v]` - identical shape, in `o_red`.
- the two l2norm sums are a different tree: lane `lid < 128` contributes
  `f32(q_b[lid])²` - one term per lane, so a plain multiply and no `fma` - and
  lane `128+i` the k term; each 128-wide array collapses with
  `stride = 64, 32, …, 1`.

A tree's order is a property of the **bands**, not of `c` or `lane`. That is
what lets the host reference walk whole 128-column heads and still land on the
same bits, and it is why nothing here needs an atomic: two replays of the
captured list produce identical bytes, which `gdn_step_test` asserts directly.

**SLM: 6912 B** - 768 B of conv outputs, 1 KB of l2norm sums, 1 KB of
normalised q/k, and 2 KB each for the two reduction arrays. The reduction arrays
are laid out `[16 bands][32 columns]` rather than `[32][16]` so that both the
writes and every tree step are lane-contiguous in SLM. Measured from the
compiler's own `zeinfo`: `slm_size 6912`, `grf_count 128`, `simd_size 16` (the
`intel_reqd_sub_group_size(16)` in the source, honoured), `barrier_count 1`, and
**neither a `private_size` nor a `spill_mem_size` entry** - the 16 fp32 of state
stay in registers, which was the design's one real risk.

### The conv in the prologue, and the four-fold redundancy it costs

Each work-group convolves 384 channels: the 128 **q** and 128 **k** of k-head
`h/3` (the `repeat_interleave(·, 3)` in doc 03) and the 128 **v** of head `h`.
Every one of the head's four chunks computes all 384, so each channel is
convolved 4 times, and each q/k channel 12 times (3 v-heads × 4 chunks).

That is deliberate. The alternative is a separate `conv1d + silu` kernel per
layer, which costs **one more launch per GDN layer** (48 per token, ~25 µs at
the measured 0.52 µs floor, doc 07 #5) plus a round trip of the convolved qkv
through DRAM - and it does not even save the redundant *reads*, since every
work-group still needs the full 128-wide q and k for the l2norm, whose reduction
domain is the head. The redundant work itself is four taps of `fma` on 384
channels: 1536 `fma` per work-group against the recurrence's four passes over
the 4096-cell slice, so under a tenth of the group's arithmetic. The
redundant *reads* are 192 × (1536 B of partials + 6144 B of conv weights +
2304 B of ring history) ≈ 1.9 MB per layer, against the state's 6.3 MB, and
they are L2-resident by construction - 12 work-groups reading the same 512 B of
q channels within microseconds of each other.

### Ring ownership - the argument, written out

The conv state is a **ring of 16 slots × 10240 channels of bf16** per layer
(`runtime::DecodeBuffers::conv_ring`), holding raw qkv values - the reference's
`conv_states` likewise hold the *input* sequence, not the convolved one. Since
up to 12 work-groups compute the same raw value, exactly one stores it:

- `c == 0` writes this head's **v** channels, `4096 + 128h … +128`;
- `c == 0 && h % 3 == 0` also writes the **q** and **k** channels of k-head
  `h/3`, which that triple of v-heads shares.

Together those cover all 10240 channels exactly once - 48 × 128 v plus 16 × 128
q plus 16 × 128 k. *Which* work-group owns a channel is a bandwidth decision and
not a correctness one, because the value is identical wherever it is recomputed:
it is `rne_bf16` of a buffer nothing writes during this step.

The correctness argument is about **slots**, not owners. A work-group writes
slots `(pos+m) % 16` for `m < n_active` and reads slots `(pos−1) % 16`,
`(pos−2) % 16`, `(pos−3) % 16`. With ring depth **16 ≥ M + 3** (plan 1 §9.4,
which is where the depth 16 comes from) those two sets cannot intersect, so **no
work-group ever reads a slot any work-group is writing this step** - and that,
not a barrier, is what makes 192 independent work-groups safe without any
cross-work-group synchronisation, which Level Zero would not give inside one
launch anyway. Within a work-group the argument is simpler still: each work-item
reads its own channel's three history slots *before* the loop that writes, and
the newer window entries come from its own registers rather than from the ring.

`gdn_step_test` pins this without going through the reference at all: at
`pos = 5` it blanks ring slots 2-4 and requires `gdn_o` to move, then blanks the
*other thirteen* slots and requires `gdn_o` to be **bitwise unchanged**. The
pair proves the window is exactly those three slots and that nothing reads the
slot this step writes - a shared misreading in kernel and reference could not
survive it.

### What the test asserts

`tests/kernels/gdn_step_test.cc` against `gdn_ref.h`, on seeded random inputs
(partials and state ~ N(0,·), conv weights U(±0.5), `negA = −exp(U(−4, 0.5))`,
every ring slot filled). The reference reproduces every rounding and every tree,
so what is left between host and device is only the libm functions OpenCL does
not require to be correctly rounded - `exp` (3 ulp) in the decay, the sigmoid
and the SiLU, `log1p` (2 ulp) in the softplus. `1.0f/sqrt` is *not* among them,
because every kernel now builds with `-cl-fp32-correctly-rounded-divide-sqrt`.

| case | bar | measured |
|---|---|---|
| `pos = 0` (history below zero) - `state` | rel ≤ 1e-5 | **3.26e-07** |
| `pos = 0` - `gdn_o` | rel ≤ 1e-3 | **5.38e-07** |
| `pos = 0` - `conv_ring`, all 163840 words | **bit-exact** | 163840/163840 |
| `pos = 5` (history in slots 2,3,4) - `state` | rel ≤ 1e-5 | **3.44e-07** |
| `pos = 5` - `gdn_o` | rel ≤ 1e-3 | **4.39e-07** |
| `pos = 5` - `conv_ring` | **bit-exact** | 163840/163840 |
| `pos = 1` (partial clamp: -2, -1 zero, 0 a real slot) - `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **3.35e-07 / 4.20e-07** |
| `pos = 15`, M = 2 (write wraps to slots 15 and 0) - `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **4.18e-07 / 5.57e-07** |
| `pos = 5`, M = 2, n_active = 2 - `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **4.59e-07 / 5.58e-07** |
| slot ownership (device only, no reference) | see above | holds |
| replay of every case | **bitwise** identical | holds |

"Relative error" is per element floored at the tensor's own RMS,
`|got − ref| / max(|ref|, rms(ref))`: `Δ = (v − kv)·β` is a difference of two
same-sized numbers, so an individual state cell can land arbitrarily close to
zero by cancellation and an unfloored ratio there would measure the
cancellation, not the kernel. In practice the floor barely does anything: the
worst `state` element is above the RMS in every M = 1 case, and across the whole
table the floored and unfloored numbers differ by at most 1.3× (`pos = 1`'s
`gdn_o`), which is why both are printed. The margins are ~20× on `state` and
~1800× on `gdn_o`; the bars are kept where the plan set them because they are
the bars a *driver* change must not break, not descriptions of today's driver.

The M = 2 variant is compiled for spec 1 §9's M-loop rule and run anyway,
because nothing else covers the conv window's intra-step path: at `m = 1` the
window is positions 3, 4, 5 and 6, so **two of its four entries are this step's
own raw values** - token 0's at position 5 and token 1's own at position 6 -
while positions 3 and 4 are still ring slots. Of the three *older* taps, exactly
one is intra-step.

### Rejected, and what was not measured

**Nothing here has been timed.** There is no `probe_gdn`, and the arithmetic
below is arithmetic. Task 9 measures the per-layer time; until it does, this
section states no wall-clock number.

- **SIMD32 - the shape the SYCL reference uses, not measured here.**
  `vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp` runs
  `sub_group_size = 32`, 8 subgroups of 32, 4 k-rows and 4 v-columns per lane -
  the same 16 cells per work-item and the same 32 columns per group, reached
  from the other side. SIMD16 is this project's constraint, and the mapping
  above adapts cleanly to it (16 bands of 8 instead of 32 bands of 4), so no
  blocker was hit and no switch was needed. What SIMD32 would buy is a
  contraction that reduces inside one subgroup rather than across sixteen -
  their code uses `reduce_over_group` where this one uses an SLM tree. That
  would be faster and would give up the *stateable* reduction order the
  bit-comparable reference rests on. Unmeasured on both counts.
- **State in SLM instead of registers - rejected.** 16 KB per work-group would
  fit, and it would let the state be loaded fully coalesced regardless of the
  tile mapping. It also puts every recurrence access through SLM and caps
  occupancy at four work-groups per Xe-core. The register form compiles with no
  spill and no private memory (`zeinfo`, above), which was the condition for
  keeping it. Not measured.
- **`sub_group_reduce_add` instead of the SLM trees - not used**, the same
  ruling as `prep`: the tree's order is stateable and the whole comparison rests
  on the reference reproducing it, while a subgroup reduce's internal order is
  the compiler's business. Faster, probably. Not measured.
- **A separate conv kernel - rejected**, argued above: one more launch per GDN
  layer and a DRAM round trip, to save redundant `fma`s worth under a tenth of
  the work-group's arithmetic and reads that are L2-resident.
- **One work-group per head (48) - rejected on occupancy**, argued above.
- **`native_exp` - rejected outright**, as in `prep`: it would put the decay,
  and therefore the whole recurrence, somewhere the host cannot follow.
- **Atomics for the two contractions - never considered seriously.** A float
  atomic add is order-dependent, and two replays of the captured list would
  differ. Determinism is an acceptance criterion for this plan, not a nicety.

### Traffic per token (arithmetic, not a measurement)

The state is the whole story: `48 heads × 128 × 128 × 4 B` = **3.146 MB per
layer**, read once and written once by the 192 work-groups that partition it.

| item | per layer | per token (× 48) |
|---|---|---|
| `state` read + written | 6.29 MB | 302 MB |
| redundant conv inputs (partials, weights, ring history; 4-12× by design, L2-resident) | 1.9 MB | 92 MB |
| `conv_ring` written | 20 KB | 0.98 MB |
| `gdn_o` written | 24 KB | 1.18 MB |

**≈ 396 MB per token**, of which the state's 302 MB is **1.9% of the token's
15.52 GB of weights** - the share doc 03 predicted. At the measured 600 GB/s
that is ~0.5 ms of the ~26 ms roofline step this table was written against -
1.2% of the 42.141 ms step measured 2026-08-25 - plus 48 launches × 0.52 µs =
25 µs of launch
overhead. The redundant reads are the price of 192 work-groups instead of 48 and
of not paying for a second kernel; whether that trade was right is what the
measurement below has to settle, and this table is what it should be compared
against.

### Measured - per kernel, in situ (2026-08-25): **0.733 ms, 1.09× its own floor**

**`gdn_step` used to have no per-kernel number**, only the 11.33 ms bucket it
shared with `prep`'s 241 launches, `attn_prep`/`attn_reduce`'s 32 and the 48
`a‖b` GEMVs. What that bound allowed was stated here as: upper bound 11.33 ms
(26.9% of the step), lower bound ~0.67 ms at 590 GB/s from the traffic table
above - with the ruling that "if the profile puts it near the bound, the cause is
**not** the 396 MB: it is occupancy or register pressure".

`b70-decode --profile --depth 4096 --steps 32` (method and caveats:
[15-step-anatomy.md](15-step-anatomy.md)):

| | measured, in situ | |
|---|---|---|
| per launch | **15.276 µs** | 48 launches/token |
| per token | **0.733 ms** | **1.7% of the step** |
| effective bandwidth | **540 GB/s** | 8.25 MB per launch ÷ 15.276 µs - **92% of the measured 590** |
| against its own traffic floor | **1.09×** | 396 MB/token ÷ 590 GB/s = 0.671 ms |

**It is at the lower bound, not near the upper one.** The 48×4 grid (192
work-groups, SLM 6912 B, GRF 128, no spills) reaches 92% of what the device can
stream, and the four-fold redundant conv reads this section defends as "the
price of 192 work-groups instead of 48" are, measured, not a price at all. The
whole kernel, made perfect, is worth **0.06 ms** - 0.15% of the step.

That settles the retune this document invited. Spec 1.5's lever L3 (`CHUNK_V`
32 → 16, GRF mode, SIMD width) has nothing to buy and docs/15 ranks it
**skip-by-ruling**. The `gdn_step` occupancy suspicion in doc 05 point 5 is
withdrawn on measurement; the occupancy problem it was looking for is real, but
it is in `prep_res_norm` (1 work-group), `a‖b` (2) and `attn_decode`'s
depth-independent term (4 live), not here.

The 48 launches cost 35 µs of dispatch (0.733 µs each, derived in situ -
docs/15), 4.8% of the kernel's own time and 0.08% of the step. Kernel count was
never the question here either.

## `attn` - decode attention in three kernels

`src/kernels/attn.cl`, three entry points in one file. **16 of the model's 64
layers** run all three, in order, per token: `attn_prep` → `attn_decode` →
`attn_reduce`. Variants are `attn_prep_M{M}` and
`attn_{decode,reduce}_M{M}_L{MAXLEN}` - `attn_prep` indexes the KV caches by
absolute position and so needs no `max_len`, while the other two bake it because
`attn_part` is strided `[24][MAXLEN/256][M][258]` and a stride must be a
compile-time constant. The *grid* still comes from `buffers.max_len` at capture,
so the variant bound to a layer must be the one built for that `max_len`.

The compiled set is deliberately asymmetric while M = 2 is test-only:
`attn_prep_M{1,2}`, `attn_{decode,reduce}_M1_L{4096,16384}` and
`attn_{decode,reduce}_M2_L4096` - the 4096 rows being the length `attn_test`
allocates. **There is no `attn_{decode,reduce}_M2_L16384`**, so
`kernels::attn_decode_variant(2, 16384)` names no file and the runtime cannot
bind M = 2 attention at the loader's default `max_len`. Whoever turns M > 1 on
(plan 3's MTP work) adds those two rows to `src/kernels/CMakeLists.txt` first.

```
attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v)

x_b    = rne_bf16(qkv_partials[0][m][col])                  (the linear's bf16 out)
rstd   = 1 / sqrt(mean_i(f32(x_b)²) + 1e-6)                 (fp32; never rsqrt)
nrm[i] = f32(rne_bf16(f32(x_b[i]) · rstd · w[i]))           (fp32 (1+w) weight)
RoPE over dims 0..63, pairs (i, i+32); dims 64..255 pass through
  q-head  h: attn_q[m][h][i] = roped ;  attn_gate[m][h][i] = f32(rne_bf16(gate col))
  kv-head j: kv_k[pos+m][j][i] = rne_bf16(roped)
             kv_v[pos+m][j][i] = rne_bf16(v col)            (never normed or roped)

attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)

per (kv-head j, 256-position block), per q-head qh of j's six, per token m:
  score_p = (Σ_d attn_q[m][qh][d] · f32(kv_k[p][j][d])) / 16   for p <= pos+m
  online softmax in waves of 16 positions -> (mx, sm) and acc[256]
  attn_part[qh][block][m] = {mx, sm, acc[256]}                 (258 fp32)

attn_reduce(ctrl, attn_part, attn_gate, attn_out)

nb = (pos + m)/256 + 1 ; merge blocks 0..nb-1 ascending ; out = acc/sm
attn_out[m][h·256+d] = rne_bf16(f32(rne_bf16(out)) · sigmoid_f32(attn_gate[m][h][d]))
```

That is doc 03's full-attention block op for op, and `tests/kernels/attn_ref.h`
is the same chain on the host - trees, waves and merge order included.

**The column map** is the one thing here that is easy to get wrong and silent
when wrong. `q_proj ‖ k_proj ‖ v_proj` is 12288 + 1024 + 1024 = 14336 columns
(S = 1), and inside `q_proj` the 24 heads are **interleaved per head, not two
halves**: head `h` is `[h·512, h·512+256)` and its output gate is the next 256.
k-head `j` is at `12288 + j·256`, v-head `j` at `13312 + j·256`. GQA is 6:1, so
q-head `h` reads kv-head `h/6`.

### Work assignment, and why

**`attn_prep`: grid (28, M), work-group 256.** A head is exactly 256 wide and
the work-group is 256, so work-item `i` owns dim `i` and the RMSNorm's reduction
domain *is* the work-group - the same shape `prep_gated_head` uses for its
128-wide GDN heads, and for the same reason. Work-groups 0..23 are q-heads (each
also writing its gate), 24..27 are the four kv-heads. 28 and not 32 because k
and v of a kv-head share a work-group: v is never normed and never roped, so it
is one extra `rne_bf16` on a work-item that is already resident, against a whole
extra work-group's worth of launch and scheduling.

The one piece of cross-lane traffic is RoPE, which pairs dim `i` with dim
`i ± 32`. That is why the normalised head goes through SLM (`nrm[256]`, 1 KB)
rather than staying in a register: 64 of the 256 work-items need a value another
work-item computed.

**`attn_decode`: grid (4 kv-heads, MAXLEN/256 blocks), work-group 256 = 16
subgroups × 16 lanes (SIMD16).** Work-group `(j, blk)` owns one 256-position
block of one kv-head and loops over that head's six q-heads and the tokens in
flight. Work-item `lid` owns accumulator dim `lid`; within a wave, subgroup
`lid/16` owns one KV position and its lane `lid%16` sixteen elements of that
position's 256-dim dot.

*Why the kv-head and not the q-head.* The q-head loop is the **outer** one, so a
work-group reads its block's K and V six times over. A (q-head, block) grid
reads exactly the same bytes - 24 work-groups per block instead of 4, each
reading the block once - so the choice is not about byte count. It is about
*when*: under the kv-head grid the six passes happen inside one work-group, back
to back, so passes 2..6 are L1/L2 hits by construction instead of a bet on 24
independent work-groups landing on the same cache at the same time. It also
launches 6× fewer work-groups. What it costs is in "Traffic per token" below,
and reading the block once outright is the first item in the not-measured list.

*Why the lane index picks the dim within a position, and not the position.* At a
fixed `t` the 16 lanes of a subgroup read `kv_k[p][j][lane + 16t]` - 16
**consecutive** bf16, one 32 B access. The alternative (lane `l` takes the
contiguous run `[16l, 16l+16)`) spans 512 B in 16 two-byte pieces per step. Same
arithmetic, same registers, an order of magnitude apart on the load that *is*
this kernel.

*Why 16 subgroups × 16 positions and not one position per work-item.* Giving
work-item `p` the whole 256-dim dot for position `p` needs no reduction at all -
and makes every access a 2 KB-strided gather, because the 256 work-items would
be reading 256 different KV rows at the same dim. The subgroup form pays one
16-wide SLM tree per position to keep every load coalesced.

**`attn_reduce`: grid (24, M), work-group 256.** One work-group per (q-head,
token), work-item `d` owning dim `d`. The `nb` block headers are staged into SLM
in a single pass so the merge loop has no barrier at all, and every work-item
then runs the *identical* scalar merge alongside its own `acc` - identical
inputs in an identical order give identical bits, so nothing has to be published
and no work-item diverges. That is cheaper than electing one work-item to
compute the scalars and broadcast them, which would cost a barrier per block.

**`zeinfo`** (the compiler's own report, `-device bmg-g31`): `attn_prep`
`simd_size 32, slm_size 2048, grf_count 128, barrier_count 1`; `attn_decode`
`simd_size 16` (the `intel_reqd_sub_group_size(16)` in the source, honoured),
`slm_size 2048, grf_count 128, barrier_count 1`; `attn_reduce` `simd_size 32,
slm_size 512` (at `MAXLEN = 16384`: 64 blocks × 2 headers), `grf_count 128`.
**None of the three has a `private_size` or a `spill_mem_size` entry** - the
16-float `sc[]` array each `attn_decode` work-item carries through a wave stays
in registers, which was this kernel's one real register risk.

### The early-out - the answer to a grid that cannot be re-sized

Under replay the attention grid is baked at capture and sized for `max_len`
(doc 04, "Attention under replay"), so at a context of `pos` tokens most of
`attn_decode`'s work-groups have nothing to do. Each one tests **one uniform
condition before touching anything**:

```
if (block_start >= pos + n_active) return;
```

The test is per **block**, not per token in flight. A block live for any `m`
runs for every `m`, and a position outside a given `m`'s causal bound is masked
to −INF inside the wave - so a partially valid block writes real partials for
each `m`, and the all-masked-for-this-`m` case falls out of the online update as
`(−INF, 0, 0)` with no special case anywhere. The pairing that makes this safe
is with `attn_reduce`, which reads exactly `nb(m) = (pos+m)/256 + 1` blocks:
every one of those starts at or before `pos+m` and therefore ran, so the reducer
never reads a block the early-out skipped and needs no data-dependent skip logic
of its own. `attn_test` asserts both halves at once - it canary-fills
`attn_part` with 1e30 *inside the replayed list*, requires every block at or
beyond `pos + n_active` to come back still holding the canary, and requires
`attn_out` to match the reference, which it could not if the merge had read a
1e30 header (that block would win the running max outright and drive the output
to 1).

**What the early-out costs at short context is open question #12 in doc 07** -
whether a `max_len`-sized grid of mostly-immediate-return work-groups is cheap
enough to keep one captured list for every depth, or whether context-bucketed
lists are needed. It is measured in Task 9 as the step time at depth 64 against
depth 4096 with the same list; the threshold doc 07 sets is ~2% of a step.
Nothing here assumes an answer.

### The wave: an online softmax with a stateable order

A block's 256 positions are walked in 16 waves of 16. Everything below is stated
identically in `attn.cl` and `tests/kernels/attn_ref.h`, because the reference
reproducing it bit for bit is what buys the tight bars.

**The score dot.** Subgroup `s` owns the wave's position `p_s`; its lane `l`
accumulates the 16 elements `d = l + 16t`, `t = 0..15` **ascending**, with an
explicit `fma`, into `dot_red[16s + l]`. The 16 lane partials collapse with a
fixed pairwise tree - for `stride = 8, 4, 2, 1`,
`dot_red[16s+l] += dot_red[16s+l+stride]` - and `dot_red[16s] · 1/16` is the
score (1/16 = 1/√256, doc 03). A masked position skips the dot entirely and is
overwritten with −INF, so an unwritten cache slot never reaches the arithmetic.

**The update**, computed **redundantly by all 256 work-items** from the same SLM
words in the same order:

```
nmx   = max(mx, sc[0], sc[1], …, sc[15])          (ascending s)
resc  = exp(mx − nmx)                              (0 when mx = −INF)
sc[s] = exp(sc[s] − nmx) ;  ssum = Σ_s sc[s]       (ascending s)
sm    = fma(sm, resc, ssum)
mx    = nmx
        - and, in the work-item that owns dim d,
t     = Σ_s fma(sc[s], f32(kv_v[p_s][j][d]), t)    (ascending s)
acc   = fma(acc, resc, t)
```

Two properties are load-bearing. First, **the −INF algebra works out on its
own**: `exp(−INF − finite) = 0`, so a masked position contributes a zero weight
and a first wave with `mx = −INF` gets `resc = 0`, both without a branch. The
single case that does not work out is `nmx = −INF` - every position of the wave
masked *and* no earlier valid position - where `exp(−INF − (−INF))` is a NaN;
that wave is skipped whole (`resc = 1`, all weights 0). It is not a hypothetical:
at `pos = 256` block 1 holds exactly one valid position, in wave 0 subgroup 0,
and its other fifteen waves take that path.

Second, **redundant is cheaper than elected here.** The obvious shape is to let
one subgroup own `(mx, sm)` and publish the rescale factor and the 16 weights
through SLM; that costs an extra barrier per wave and leaves 240 work-items idle
through a serial 16-iteration loop, 96 times per (q-head, token). Recomputing
the same scalars in every work-item costs 16 SLM reads and some ALU that is
free next to the KV loads, and it is *more* obviously deterministic, not less.
(This is the one deliberate departure from the plan's ruled scheme, which named
subgroup 0 as the updater; the ruled order is unchanged, only who runs it.)

### The block merge, and why `nb` needs no skip logic

`attn_reduce` merges in **ascending block order**, `b = 0 … nb−1`:

```
nmx = max(mx, bmx) ; a = exp(mx − nmx) ; bs = exp(bmx − nmx)
sm  = fma(sm, a, bsm·bs)
acc = fma(acc, a, bacc·bs)
mx  = nmx
```

which is the canonical flash-attention rescale, with `mx = −INF` on entry so the
first block's `a` is 0 and the first merge is a plain copy. `nmx` is never −INF
inside the loop, because block `b < nb` starts at `256b ≤ pos+m` and therefore
contains at least its own first position inside the causal bound - the same fact
that lets the early-out leave blocks `≥ nb` untouched. So no guard is needed
here and none is written; the −INF case exists only in `attn_decode`'s wave.

`nb` is nonetheless clamped to `NBLOCKS` before the header-staging loop. Under
the real precondition - `pos + n_active ≤ max_len`, which the engine enforces
when it advances `Control::pos` - the clamp is dead code. It is there so that a
*violated* precondition costs a wrong answer instead of an SLM overrun writing
past `hmx`/`hsm` into whatever the compiler laid out next, which is the kind of
failure that reproduces as something else entirely three kernels later.

### The RoPE rounding - the one op this trio does not match torch on

RoPE is applied to the **fp32 widened normalised value**: `attn_q` keeps the fp32
result and the k written to the cache is `rne_bf16` of it, one rounding for the
whole pair. torch reaches the same value through bf16 tensor ops and therefore
rounds once more, *inside* the cos/sin multiply-add. This is a **≤ 1-op
difference** against the reference, taken deliberately (controller ruling
2026-08-25) rather than by oversight, and it is recorded here because it is the
only place in this trio where the kernel is not a per-op transcription of the
oracle. Everything else - the linear's rounding, the norm's `type_as`, the
gate's widening, the two roundings in the final gated product - is torch's op
chain exactly. The rotation itself is one rounded product plus one `fma`,
spelled identically on both sides:

```
out_i      = fma(x_i,      cos_i, −(x_{i+32}·sin_i))
out_{i+32} = fma(x_{i+32}, cos_i,  (x_i·sin_i))
```

with `cos/sin` from the loader's table, whose angles were computed in `double`
(doc 13, "The RoPE table"). Dims 64..255 pass through untouched - the
`partial_rotary_factor 0.25` of doc 03.

### What the test asserts

`tests/kernels/attn_test.cc` against `attn_ref.h`, on seeded random inputs
(qkv partials ~ N(0,1), the FA block's fp32 `1 + w` norms ~ U(0.75, 1.25), the
loader's real RoPE table, and **the entire 4096-position KV cache** filled with
random bf16 - not just the prefix, so a read past the causal bound shows up as
noise rather than as a convenient zero). `max_len = 4096`, i.e. 16 blocks.

Because the reference reproduces every rounding, every tree and every wave, the
only thing left between host and device is the one libm function OpenCL does not
require to be correctly rounded and this trio uses: **`exp`, 3 ulp**, in the
softmax, in the merge and in the final sigmoid. `1.0f/sqrt` is not among them
(every kernel builds with `-cl-fp32-correctly-rounded-divide-sqrt`), which is
why the norm, the RoPE and the KV cache are held **bit-exact**:

| case | `attn_q` roped dims | `attn_part` | `attn_out` rel (bar 1e-3) | `attn_out` ulp: worst anywhere / worst at `|ref| ≥ rms/8` (bar 2) | blocks still canary |
|---|---|---|---|---|---|
| `pos = 0` (one valid position; 15 blocks early-out) | **0** | **0** | **0** | 0 (0 of 6144 differ) / **0** of 5282 | 360 × 1 |
| `pos = 254` (block 0 partial; 255 masked) | **0** | 4.670e-07 | 1.326e-05 | 1 (1 word) / **0** of 5278 | 360 × 1 |
| `pos = 255` (block 0 exactly full) | **0** | 5.024e-07 | **0** | 0 (0 words) / **0** of 5299 | 360 × 1 |
| `pos = 256` (block 1: one valid position, 15 empty waves) | **0** | 7.109e-07 | 8.664e-04 | 2 (1 word) / **0** of 5267 | 336 × 1 |
| `pos = 4095` (cache full to `max_len`; 16-block merge) | **0** | 7.240e-07 | 1.995e-04 | 95 (3 words) / **0** of 5305 | 0 |
| `pos = 254`, M = 2 (per-`m` mask **inside** one block) | **0** | 6.420e-07 | **0** | 0 (0 of 12288) / **0** of 10562 | 360 × 2 |
| `pos = 255`, M = 2 (per-`m` mask **across** the block edge) | **0** | 8.018e-07 | 1.270e-05 | 2 (2 words) / **0** of 10480 | 336 × 2 |

and, in **every** case:

- `kv_k` and `kv_v` - **bit-exact over all 4 194 304 words of each cache**, not
  just the slots this step writes. The untouched slots are what prove
  `attn_prep` writes positions `pos … pos+n_active−1` and nothing else.
- `attn_q`'s pass-through dims 64..255 and the whole of `attn_gate` -
  **bit-exact**. The roped dims 0..63 carry the ruled 2 ulp bf16 bar and come
  back at **0** everywhere, which is the fma spelling above doing its job.
- replay - **bitwise identical** on all six outputs (`attn_q`, `attn_gate`,
  `attn_part`, `kv_k`, `kv_v`, `attn_out`), from freshly re-uploaded inputs.

**`attn_out` carries two bars, and the second one is the arbiter.**

**(a) Relative error ≤ 1e-3, floored at the tensor's RMS.** The 1e-3 is the
plan's; the *floor* is this task's deviation from it, and it is not optional:
`acc/sm` is a weighted average of *signed* v values, so a dim can cancel to
~6.6e-10 against an RMS of ~0.07, and an unfloored ratio there would measure the
cancellation rather than the kernel (the `pos = 4095` row's "95 ulp anywhere" is
exactly one such dim - 3.5e-10 of absolute nothing).

**(b) ≤ 2 bf16 ulp on every element with `|ref| ≥ rms/8`.** Dividing by the RMS
is what makes (a) slack on exactly the elements a bf16 output can most easily
move: `attn_out` *is* bf16, so a **single** round-to-nearest boundary flip on an
element the size of the RMS is already `2^-8 = 3.9e-3` of relative error, four
times (a)'s bar. So (b) is the one to trust - **if a driver change ever trips
(a), read (b) before believing the kernel broke.** The `rms/8` gate is what
keeps (b) from being either vacuous or false: below it a bf16 ulp is not a unit
of error at all, above it it is the only unit that means anything. The **2** is
the arithmetic of the final chain rather than a fudge -
`rne_bf16(f32(rne_bf16(acc/sm)) · sigmoid_f32(gate))` rounds to bf16 **twice**,
and 3 ulp of `exp` slack can push a boundary value one ulp at each of those
roundings and no further; 3 would mean an arithmetic difference.

The measured picture is sharper than either bar. Across all seven cases the
worst gated distance is **0**: every one of the ~5280 elements per token at or
above `rms/8` (86% of the 6144) is **bit-identical** to the reference, and the
one-to-three words that differ at all lie strictly below the gate. The thin
number in the table - `pos = 256`'s 8.664e-04 against (a)'s 1e-3, a 1.15×
margin - comes entirely from a dim at `|ref| ≈ 0.006` against an RMS of 0.070,
i.e. from (a) measuring cancellation, which is the structural reason (b) exists.

The M = 2 variant is compiled for spec 1 §9's M-loop rule and **run**, on the
gdn_step precedent, because it is the only cover for per-`m` causal masking - at
M = 1 every position in flight shares one causal bound. At `pos = 254, n = 2`
position 255 is masked for `m = 0` and valid for `m = 1`, so block 0 must produce
two different partials for the same (q-head, block); at `pos = 255, n = 2`
block 1 is wholly beyond `m = 0`'s bound, writes `(−INF, 0, 0)`, and
`attn_reduce` - whose `nb(0)` is 1 - must not read it. The `−INF` headers are
compared as **exact bit patterns**, not as numbers.

### Rejected, and what was not measured

**Nothing here has been timed.** There is no `probe_attn`; the arithmetic in the
next section is arithmetic. Task 9 measures the per-layer time and doc 07 #12's
depth sweep; until it does, this section states no wall-clock number.

- **SLM-staged K/V tiles instead of direct loads - the main open lever, not
  measured.** As built, the q-head loop is outer and a work-group reads its
  block's K and V six times (next section). Moving the q-head loop *inside* the
  wave loop and staging the wave's 16 K rows and 16 V rows in SLM
  (16 × 256 × 2 B × 2 = 16 KB, plus six q-heads staged = 6 KB, plus the existing
  1 KB tree) would read them once outright, at the cost of six accumulators and
  six `(mx, sm)` pairs per work-item instead of one. The per-q-head arithmetic
  and every stated order would be unchanged, so the reference would not move.
  Whether it is worth it depends entirely on how much of the reread the 24 MB
  L2 already absorbs, which is a measurement, not an argument.
- **`sub_group_barrier` for the score tree - not used, not measured.** The
  16-wide tree is entirely inside one subgroup, so four of the six
  `barrier(CLK_LOCAL_MEM_FENCE)` per wave could be subgroup fences instead of
  work-group barriers (96 barriers per (q-head, token) would become 32). Full
  barriers were kept for the same reason `gdn_step` keeps them: they are the
  primitive whose semantics need no argument, and this kernel's cost is its
  loads. Worth trying with a probe in hand; not worth guessing at.
- **A flash-style single work-group walking the whole `max_len` - rejected on
  fill.** One work-group per (q-head, token) is 24 work-groups per layer on a
  card with 32 Xe-cores: three quarters of the machine idle, on 16 layers per
  token. Splitting the KV into 256-position blocks is what turns that into
  4 × (max_len/256) work-groups - 256 at `max_len` 16384, eight per Xe-core -
  and the price is the two-pass structure (`attn_part` plus `attn_reduce`) and
  one extra launch per layer.
- **One work-group per (q-head, block) - rejected**, argued above: identical
  byte count, 6× the work-groups, and the six passes over a block scattered
  across independent work-groups instead of back to back inside one.
- **A two-pass softmax over the whole block** (all 256 scores to SLM, then one
  max, then a single un-rescaled accumulation) - **rejected.** It is simpler and
  drops the per-wave rescale entirely, but it costs another 1 KB of SLM and,
  more to the point, it does not generalise: the online form is what lets the
  block size grow without the score array growing with it. The plan ruled the
  online scheme; this notes what was given up.
- **`sub_group_reduce_add` for the score dot - not used**, the same ruling as
  `prep` and `gdn_step`: the SLM tree's order is *stateable*, and the whole
  comparison rests on the reference reproducing it, while a subgroup reduce's
  internal order is the compiler's business. Faster, probably. Not measured.
- **A larger `ATTN_BLOCK` (512, 1024) - not measured.** 256 was chosen because
  it makes the block a whole number of 16-position waves, makes `attn_part`'s
  `[24][max_len/256][M][258]` the size `runtime::DecodeBuffers` already
  allocates, and gives one work-group per Xe-core per 8 blocks at 16k. A larger
  block means fewer work-groups and a shorter merge; a smaller one means better
  fill at short context. Both are depth-dependent and neither is guessable.
- **`attn_part` in bf16 - rejected outright.** The partials are a softmax
  numerator and denominator; rounding them is rounding the *accumulator*, not an
  op's output, which is exactly what the rounding discipline forbids. It would
  also halve nothing that matters - `attn_part` is under 1% of this kernel's
  traffic.
- **`native_exp` - rejected outright**, as everywhere in this project: it would
  put the softmax somewhere the host reference cannot follow.
- **Atomics for the block merge (one kernel instead of two) - never considered
  seriously.** A float atomic add is order-dependent and two replays of the
  captured list would differ. Determinism is an acceptance criterion for this
  plan, not a nicety.

### Traffic per token (arithmetic, not a measurement)

At context depth `D`, per FA layer, M = 1. The KV cache is the whole story and
the 6× is the q-head loop being outer:

| item | per layer at `D` | at `D` = 4096 |
|---|---|---|
| `kv_k` + `kv_v` read by `attn_decode` (6 q-head passes) | `6·D·4·1024 B` | **100.7 MB** |
| - of which *unique* (what a staged version would read) | `D·4·1024 B` | 16.8 MB |
| `attn_part` written then read | `2 · 24 · (D/256) · 258 · 4 B` | 0.79 MB |
| `attn_q` read by `attn_decode` (staged once per q-head per block) | `24 · (D/256) · 1 KB` | 0.39 MB |
| `attn_prep`: partials read, `attn_q`/`attn_gate`/KV written | ~110 KB | 0.11 MB |
| `attn_reduce`: `attn_gate` read, `attn_out` written | ~37 KB | 0.04 MB |

**≈ 102 MB per layer, 1.63 GB per token across the 16 FA layers** - against a
*unique*-KV floor of 268 MB per token, which is the 1.7% of `W` doc 03 predicted
("KV and state sizes at the benchmark shape"). The gap between those two numbers
is the reread, and how much of it reaches DRAM is a cache question: **the card
has 24 MB of L2** (doc 01), one FA layer's whole KV at `D` = 4096 is 16.8 MB, and
at that depth only 64 of the grid's work-groups survive the early-out, holding
256 KB of block each - 16 MB resident. So at the benchmark depth the reread
should be almost entirely L2-served, and at `max_len` 16384 (64 MB of live
blocks) it cannot be. Which of those regimes the real step lands in is exactly
what Task 9 has to measure before the staged variant above is worth building.

Launches: **3 per FA layer × 16 layers = 48 per token**, ~25 µs at the measured
0.52 µs floor (doc 07 #5) - the same order as `gdn_step`'s 48, and the reason
`attn_prep` folds the norm, the RoPE and the KV write into one kernel rather
than three.

### Measured - the two numbers the fixed grid was worth

Both come from `b70-decode --bench` on an idle box, tg 256, three runs each,
2026-08-25 (doc 07 #12, now resolved; every row in
[BENCHMARKS.md](BENCHMARKS.md)). Neither is a per-kernel timing - there is no
`probe_attn`. Each pair isolates `attn_decode` by changing **only** what it
does, with the other 597 kernels held identical, so the *differences* are
attributable to this trio.

**Live blocks per token, counted properly.** `attn_reduce` merges
`nb = (pos + m)/256 + 1` blocks (`attn.cl:474`), and `attn_decode`'s early-out
turns off exactly the rest, so `nb` is the count of blocks doing work. `tg` 256
from depth `D` walks `pos = D … D+255`:

| run | `pos` range | `nb` | mean `nb` | median ms/token |
|---|---|---|---|---|
| depth 4096, `--max-len 16384` (4 × 64 grid) | 4096…4351 | 17 throughout | **17.00** | 42.141 |
| depth 64, `--max-len 16384` (4 × 64 grid) | 64…319 | 1 for 192 tokens, 2 for 64 | **1.25** | 39.997 |
| depth 64, `--max-len 4096` (4 × 16 grid) | 64…319 | same 1.25 | **1.25** | 39.951 |

**1. A live 256-position block costs ≈ 0.1361 ms/token - estimated, from a
two-point line.** (42.141 − 39.997) / (17.00 − 1.25) = 2.144 / 15.75. Across
all 16 FA layers a block is 6.29 MB × 16 = 100.7 MB of KV reads (the 6× q-head
reread in the table above), so 0.1361 ms implies **740 GB/s** - *above* the
590 GB/s `probe_bw` measures. That is the L2 question this section left open,
answered in the direction it guessed: **the 6× reread is substantially
cache-served**, as predicted for the depth-4096 regime where one layer's live
blocks are ~16 MB against 24 MB of L2. The staged variant in "Rejected" would
be optimising a read that mostly is not reaching DRAM.

Extrapolating that slope to 17 blocks puts attention's per-block work at
**≈ 2.31 ms of the 42.141 ms step, 5.5%** - and that number is **estimated, not
measured**, because it is one line through two points. Two other timings from
the same day say the line is not exact:

| point | `nb` | predicted by the line | measured |
|---|---|---|---|
| CLI run, depth 42, tg 32 (`pos` 42…73) | 1 throughout | 39.96 | **38.51** |
| depth-4096 ingest, `pos` 0…4095 | mean ≈ 8.5 | 40.98 | **41.61** |

Neither point is a clean comparison - different `tg`, different thermal history,
and the ingest average sweeps `nb` rather than holding it - but both miss by
0.6-1.5 ms, so the slope is not constant across the whole range. What survives
unconditionally is the **2.144 ms difference itself**: going from `nb` 1.25 to
17 costs that much, all of it inside `attn_decode`/`attn_reduce`. Treat 2.31 ms
as the central estimate with the measured 2.144 ms as its floor, and note that
±0.2 ms of it moves in or out of the unsplit bucket at the top of this document.

**2. The early-out is free: 0.046 ms/token, 0.11% of the step - measured.**
Rows 2 and 3 hold depth, live blocks and every byte of real work identical and
change the compiled `MAXLEN`/grid. `--max-len` is not *purely* the grid - it
also changes `attn_part`'s stride, the block headers `attn_reduce` stages into
SLM, and the KV allocation's footprint - but `attn_reduce`'s merge loop is
bounded by `nb`, not `NBLOCKS`, and `nb` is 1.25 in both rows, so the dominant
term in the difference is the 48 extra idle blocks × 4 kv-heads × 16 layers =
**3072 early-outed work-groups**: **~15 ns each**. The design decision this
section defends ("The early-out - the answer to a grid that cannot be re-sized")
is vindicated at 1/18th of the 2% bar doc 07 #12 set for abandoning it.
Context-bucketed lists are not worth building.

This pair is also the one **thermally matched** comparison in the set: both runs
are preceded by the same 2.4 s ingest, so neither carries the other's heat. The
2.144 ms depth delta above does *not* have that property - its depth-4096 run is
preceded by 170 s of continuous replay against the depth-64 run's 2.4 s - which
is a second reason to call the 2.31 ms figure estimated.

### Measured - per kernel, in situ (2026-08-25): the estimate was 2.45× low

Everything above this heading was inferred from whole-step `--bench`
differences. `b70-decode --profile --depth 4096 --steps 32` times the three
kernels directly, inside the replayed list ([15-step-anatomy.md](15-step-anatomy.md)):

| kernel | launches | work-groups per launch | **µs/launch** | **ms/token** | share of step |
|---|---|---|---|---|---|
| `attn_decode` | 16 | 4 × 64 = 256 (**68 live** at `nb` 17) | **361.400** | **5.782** | **13.7%** |
| `attn_reduce` | 16 | 24 | **4.748** | 0.076 | 0.18% |
| `attn_prep` | 16 | 28 | **3.204** | 0.051 | 0.12% |

**`attn_decode` costs 5.782 ms, against the 2.360 ms this section extrapolated**
(2.314 estimated block work + 0.046 measured early-out). The two-point line was
a genuine under-prediction, by 145%, and it is the largest single error the
profiler found anywhere in the step. `attn_prep` and `attn_reduce`, the two
kernels this section left unmeasured, are together **0.127 ms** - 0.3% - and are
not worth another sentence.

Four in-situ points, all with this instrument, 2026-08-25:

| run | `nb` | live WGs | positions attended | µs/launch | ms/token |
|---|---|---|---|---|---|
| depth 4096, `--max-len 16384` | 17 | 68 | 4097 | **361.400** | 5.782 |
| depth 1024, `--max-len 16384` | 5 | 20 | 1025 | **338.165** | 5.411 |
| depth 64, `--max-len 16384` | 1 | 4 | 65 | **153.608** | 2.458 |
| depth 64, `--max-len 4096` | 1 | 4 | 65 | **153.665** | 2.459 |

1. **The early-out is free, re-confirmed on the kernel itself.** Rows 3 and 4
   differ by 192 idle work-groups per layer and by **0.04%** - inside the noise
   of a run whose whole-step spread is 0.47%. Doc 07 #12 got this from a
   0.046 ms whole-step difference; this is the same verdict measured where it
   happens. Context-bucketed lists stay dead.
2. **The depth-1024 point falsified two models the depth-64/4096 pair
   admitted**, both pre-registered with their predictions before it was run
   (docs/15 §2): a linear-in-blocks fit with a 2.25 ms depth-independent
   intercept predicted 205.6 µs and a wave/occupancy model predicted 153.6;
   measured **338.165**. The intercept was an artefact of treating `nb` = 1 as a
   full block when it is **65 of 256 positions**. Nothing in this document claims
   a depth-independent term.
3. **What fits: the launch costs what its slowest single work-group costs.**
   Work-group count is nearly free - `nb` 5 → 17 is **3.40× the live
   work-groups for +6.9% of time** - and what moves the number is the *fill* of
   the critical-path block. `F + fill · P` gives **F ≈ 90.8 µs fixed, P ≈ 247.4 µs
   for a full 256-position block** (derived from the `nb` 1 and 5 points; the
   `nb` = 17 point is a check at −6.4%). So ~73% of a launch at depth 4096 is one
   work-group's serial walk of one block. This is a latency problem on a critical
   path, not an occupancy problem on a grid.
4. **It is not bandwidth-bound, and the L2 reading above needs correcting.**
   The effective rate rises with depth: 10.4 GB/s at `nb` = 1, 74.5 at `nb` = 5,
   **278.6 at `nb` = 17** (47% of the measured 590) counting the 6× q-head
   reread; the *unique* 16.8 MB at depth 4096 is **46 GB/s**. The inference above
   ("0.1361 ms implies 740 GB/s … the 6× reread is substantially cache-served")
   had the right direction - 46 GB/s of unique traffic cannot be DRAM-limited -
   but the wrong magnitude, because the 0.1361 ms it rested on is 35% below the
   in-situ slope. What follows for the staged variant in "Rejected" changes
   shape: the target is the 247 µs full-block critical path, and the cheapest
   attack on it is a **finer block** (128 positions → 33 blocks × 4 = 132
   work-groups at depth 4096, predicted ≈3.43 ms/token), not SLM staging. Spec
   1.5's lever L5, ranked 1 by measured share in docs/15 and third to execute
   there; expected yield 1.5-2.5 ms, extrapolated.
