# Kernels

One section per device kernel: what it computes, how the work is assigned and
**why**, what was rejected and on what evidence, and the numbers that decided
it. A section that only restates the code is not finished.

Every figure says which kind it is: **measured**, **derived** (arithmetic over
measured inputs, with the inputs shown) or **estimated** (a stated guess with
its reasoning). Three instruments produce them:

- **standalone probes** in `tools/probe/`, which time one kernel in a closed
  command list with nothing else running;
- **`b70-decode --profile`**, which reads Level Zero kernel timestamps off the
  replayed decode list, so every launch of a real step has a duration. Method
  and caveats are in [15-step-anatomy.md](15-step-anatomy.md); figures taken
  this way are labelled **in situ**;
- **timing loops inside a kernel's own test**, for the two token-boundary
  kernels whose cost is a few microseconds and needs no separate harness. Those
  loops print and never assert.

Weight buffers in every probe are filled with **random** nibbles, never a
repeated value: the B70 losslessly compresses device-local memory, so a buffer
of one repeated word reads back above the 608 GB/s theoretical peak and any
timing taken from it is fiction ([01-hardware.md](01-hardware.md)). Real weights
are incompressible within a cache line, and so is the probe's data.

---

## The step these kernels add up to

Decode today is **29.45 t/s** at depth 4096 over 256 generated tokens, on the
byte-matched checkpoint, replaying **774 kernels and 19 modules per token**.
Every recorded row and its grade is in [BENCHMARKS.md](BENCHMARKS.md).

The per-launch anatomy below was measured on **2026-08-25**, when the step was
**42.141 ms/token (23.73 t/s)**, median of three on an idle box, 99.77% of it
inside `execute` + `fence.wait()` and 97 µs of host time per token. The roofline
on the two measured constants (15.540 GB/token, 590 GB/s) is 26.34 ms. Three
kernel changes that same day took the step to 36.32 ms, and the loader and
checkpoint work in September took it to where it is now without re-running the
per-kernel profile. **Every share below is quoted against the step of the run
that measured it, and says which run that is.** Nothing in this document is
re-based onto a later denominator, because two numbers for one quantity is worse
than one number with a date.

**The partition of the 42.141 ms step.** Every launch and the host appear
exactly once, and the rows sum to the measured total (29.005 + 5.782 + 3.573 +
2.335 + 0.733 + 0.127 + 0.017 + 0.473 + 0.097 = 42.142, one µs of rounding),
which is why they are quoted to three decimals where the prose rounds.

| | launches/token | ms/token | share | kind |
|---|---|---|---|---|
| `gemv` + `gemv_bf16` (`lm_head`) | 257 | **29.005** | 68.8% | measured, in situ (probe floor was 28.346) |
| `attn_decode` | 16 | **5.782** | 13.7% | measured, in situ |
| `prep` (`res_norm` 129, `silu_mul` 64, `gated_head` 48) | 241 | **3.573** | 8.5% | measured, in situ |
| `a‖b` GEMV (`gemv_bf16` 5120×128) | 48 | **2.335** | 5.5% | measured, in situ |
| `gdn_step` | 48 | **0.733** | 1.7% | measured, in situ |
| `attn_prep` + `attn_reduce` | 32 | **0.127** | 0.3% | measured, in situ |
| `embed_gather` + `argmax` | 3 | **0.017** | 0.04% | measured, in situ |
| dispatch gap, un-instrumented | | **0.473** | 1.1% | derived (bench fence − Σ in-situ durations) |
| host, outside the fence | | **0.097** | 0.23% | measured |
| **total** | **645** | **42.141** | 100% | measured (bench, median of three) |

**98.6% of the step is kernel time inside the fence** (41.571 ms of Σ per-kernel
durations), 1.1% is dispatch and 0.2% is the host. Everything that is not a GEMV
is 42.141 − 29.005 = **13.136 ms**, of which 13.039 is device time.

**The three changes of 2026-08-25, and the rows they moved.** Each is measured
in situ with the same instrument and each has its bench row in
[BENCHMARKS.md](BENCHMARKS.md). Each has its own section in
[15-step-anatomy.md](15-step-anatomy.md).

| row | before | after | what changed |
|---|---|---|---|
| `a‖b` GEMV | 2.335 ms | **0.256 ms** | `gemv_bf16.cl`'s K split 16 ways inside the work-group, 8 hardware threads per launch to 128 |
| `prep` family | 3.573 ms (2.871 of it `res_norm`) | **1.202 ms** (0.484 of it the two-stage pair) | `prep.cl`'s one-work-group residual norm becomes two launches on 20 work-groups each |
| `attn_decode` (+ `attn_prep`/`attn_reduce`) | 5.782 ms (5.909 with the pair) | **3.585 ms** (**3.839** with the pair) | `ATTN_BLOCK` 256 to 64, quartering the serial KV walk per work-group |

The "after" columns are different runs from the "before" column, so they are a
change of state and not a before/after pair. Each change's own paired
before/after was measured within one session and is in
[15-step-anatomy.md](15-step-anatomy.md). The drift between two runs of this
instrument on untouched launches is 0.3 to 0.5%, measured three times.

**The launch count moved with the norm split and 645 did not survive it.**
Splitting every `prep_res_norm` site into `prep_res_fold` + `prep_norm_finish`
makes each of the 129 sites two launches, so the captured walk is 645 + 129 =
**774**, the number `runtime::CapturedStep::kernel_count` reports and
`tests/runtime/replay_determinism_test.cc` pins. The module count goes 18 to 19.
`ProfileEvents::kProfileCapacity` is 1024 and did not have to move.

---

## `gemv` - int4 g64 × bf16, M ∈ [1,8]

`src/kernels/gemv.cl`, variants `gemv_M<M>_K<K>_N<N>_S<S>_L<LAYOUT>`.

**Computes**

```
out[s][m][n] = Σ_{k ∈ slice s} x[m][k] · (q[k][n] − 8) · scale[k/64][n]
```

`q` is the 4-bit nibble, symmetric with the constant zero point 8 (the
checkpoint's `qzeros` is dropped at load, [02-formats.md](02-formats.md));
`scale` is one f16 per 64 consecutive `k` per column; `x` is bf16 `[M][K]`; the
output is **fp32 partials `[S][M][N]`**, summed by `prep`. Nothing else is in
the kernel, no residual, no activation, no norm, so that split-K stays
deterministic and the GEMV stays one thing.

### Work assignment, and why

**Lane per `n`.** Lane `l` of a subgroup owns output column `n_tile·16 + l` and
holds its accumulator in a register for the whole K-range. Two things follow.
The weight read is perfectly coalesced by construction, because the 16 lanes
want 16 adjacent columns, which are adjacent words in *both* layouts. And there
is **no cross-lane reduction anywhere**: no shuffle tree, no SLM, no barrier,
and the epilogue is a single store per lane. The alternative assignment, lane
per `k` with one output per subgroup and a reduction at the end, costs a 4-deep
shuffle reduction per output column and buys nothing on a kernel whose entire
cost is the weight stream.

**SIMD16.** `intel_reqd_sub_group_size(16)`; the device offers 16 and 32
([01-hardware.md](01-hardware.md), `sub_group_sizes [16, 32]`). 16 lanes × 4
bytes is 64 bytes, exactly one cache line per lane-step, and
`intel_sub_group_block_read8` becomes 512 contiguous bytes issued as one
instruction. 16 is also the granularity everything else is cut to: the layout-1
tile is 16 columns wide, the bf16 tile is 16 columns wide, and the loader
interleaves `gate‖up` in 16-column blocks so the lane holding `gate[n]` also
holds `up[n]` at a fixed offset. OpenVINO's GEMV `#error`s on anything but 16.
SIMD32 was **not measured**, see *Rejected*.

**64 `n` per work-group** (`reqd_work_group_size(64,1,1)` = 4 subgroups). The
work-group is the dispatch unit and the grid is `(N/64) × S`. Four subgroups is
the smallest group that still hands a subslice a useful bundle of threads per
dispatch; one subgroup per group would quadruple the number of groups to
schedule, and a larger group would coarsen the grid at exactly the shapes where
the grid is already the problem (N = 5120 is 80 groups). It also keeps
`N % 64 == 0` as the only shape constraint, which every fused shape satisfies
and `a‖b` is zero-padded to meet.

**Split-K on grid dimension 1, into separate partials.** Work-group `(gn, s)`
covers k-groups `[s·G/S, (s+1)·G/S)` with `G = K/64`, and writes `out[s][m][n]`.
`prep` sums the `S` partials on the next kernel.

*Why split-K exists:* the grid is `N/16` subgroups at `S = 1`. At N = 5120 that
is **320 subgroups for 256 EUs**, and the machine cannot get enough loads in
flight to saturate DRAM. The measurement below says this costs exactly half the
bandwidth, and that `S` recovers it.

*Why partials and not atomics:* a `float` atomic add is order-dependent, so two
replays of the same captured command list would produce different logits. The
runtime's whole premise is one list replayed per token, and the golden-tensor
tests compare bitwise. The cost is `S ×` the output bytes and it varies by an
order of magnitude across the chosen configurations: 0.59% at `gate‖up`
(S = 4, 557 KB against 94.7 MB), but **1.96% at `out/o_proj`** (S = 16,
327,680 B against 16.71 MB), which is the bound at M = 1. It is also a *write*,
which this kernel otherwise does not do. At M = 8 the same worst case grows to
**~16%** (2.62 MB of partials), which is the point at which this trade would
have to be re-examined rather than assumed.

### The two layouts

Both hold the same 4.25 bits per weight (nibbles plus one f16 scale per 64), so
`weight_bytes` and therefore GB/s are directly comparable.

**Layout 0, GPTQ-native.** `qweight[K/8][N]` u32 and `scales[K/64][N]` f16
exactly as the checkpoint ships them. A lane covering 64 `k` reads 8 words at
**stride `N·4` bytes**, plus one scale from a second array. Zero repack: the
loader could upload the mapped file verbatim.

**Layout 1, tiled.** For each `(n_tile of 16, k_group of 64)` one **544-byte
block**: 128 u32 of nibbles ordered `[k_octet 0..7][lane 0..15]`, then the 16
f16 scales (32 B). Tiles are ordered k-group-inner, n-tile-outer. A subgroup
therefore walks **one contiguous run of memory across its entire K-range**, and
one k-group is one `intel_sub_group_block_read8` plus one
`intel_sub_group_block_read_us`; the scales travel with the nibbles instead of
being a second stream. The address stream a subgroup issues does not depend on
`N` at all. That independence is the whole reason the layout exists.

**Which won.** Layout 1, by the probe's rule (higher GB/s summed over the five
int4 shapes), **2713 against 2667, 1.7%.** That is inside the noise the rule was
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
mechanism is almost certainly DRAM channel or bank aliasing on the power-of-two
stride, and **that is an inference from the shape of the data, not a measured
cause**; nothing in this project has read a memory-controller counter. What is
measured is that the penalty is real, reproducible, and confined to N = 2^k.

In wall time rather than the rule's GB/s sum, each layout at its own rule-chosen
`S`, one instance of each of the five shapes: **layout 0 465.3 µs, layout 1
449.0 µs**, so layout 1 is 3.6% faster. Both metrics agree in direction.

What layout 1 costs, recorded so the next person can revisit it: a load-time
repack pass over 12.16 GB of weights (`repack()`), and the loss of the
zero-copy-from-`mmap` option that layout 0 would have allowed. A 3.6% decode win
pays for a one-off load cost; it would not pay for much else. If the model mix
ever shifts to shapes with no power-of-two `N`, this decision is worth
re-running, and the probe is the arbiter rather than this paragraph.

### Rejected, and what was actually measured

Measured, and only this: **layout 0 against layout 1** and **S ∈ {1,2,4,8,16}**,
at all five production shapes, **at M = 1**, twice. Everything below was decided
by argument. Saying so is the point of this section.

**`M` is the biggest unmeasured axis.** The kernel is parameterised on
M ∈ [1,8] and only `gemv_M2_K5120_N5120_S1_L0` is even compiled today; nothing
above M = 1 has been timed. The `S` picks below are therefore **M = 1 picks**,
and they will not simply carry over: the split-K partials scale with M, so
`out/o_proj` at S = 16 goes from 1.96% of weight bytes at M = 1 to ~16% at
M = 8, and the rule would very likely choose a smaller `S` there. Re-run
`probe_gemv` over M before trusting any of this for speculative decode.

- **Shuffle-broadcast of activations (OpenVINO's pattern) against same-line
  vector loads, NOT measured.** OpenVINO's GEMV holds `x` in registers and
  `sub_group_broadcast`es it to all 16 lanes. This kernel instead does a plain
  `vload8` from `x + m·K + g·64 + j·8`: all 16 lanes issue the *same* address,
  which the hardware serves from one cache line. The argument for not measuring
  it was that `x` is 10 KB at K = 5120, is L1-resident after the first touch,
  and is re-read against a weight stream three orders of magnitude larger. The
  measured 88 to 97% of peak on the shapes that fill the device is consistent
  with the activation path not being on the critical path, but that is
  consistency, not proof. If a profile ever shows otherwise, this is the first
  thing to try.
- **SIMD32, not measured.** The device supports it. It would make a subgroup own
  32 `n`, halve the subgroup count at every shape (N = 5120 goes to 160), and
  make the fill problem worse in exactly the place that already hurts. Rejected
  on that reasoning alone.
- **fp32 accumulation against OpenVINO's `half`, not measured.** fp32 was chosen
  because the split-K partials are summed by another kernel and bf16 activations
  already spend the precision budget. The observed `max_abs_err` against a
  double-precision CPU reference is 4.2e-05 at worst (K = 17408), against a
  tolerance of 6.7e-03, so there is room, but no measurement of what `half`
  would cost in either accuracy or speed.
- **More than one `n` per lane, or K-unrolling beyond the 64-wide group, not
  measured.** The k-group is 64 because the *scale* group is 64; unrolling
  further would need a second scale in flight for no obvious gain on a
  bandwidth-bound loop.
- **Atomics instead of partials, rejected on determinism, not measured.** The
  traffic argument says it would be worth 1.96% at most at M = 1 and ~16% at
  M = 8.

### Measured

`probe_gemv`, **2026-08-24**, on the box (`CL_DRIVER_VERSION 26.27.39122.14`,
kernels AOT-compiled by `ocloc -device bmg-g31`, host `-O2`), weights random.
Timing: each configuration is 40 launches recorded in one regular command list
and replayed 8 times, median of the last 5, `µs = list time / 40`. The list
cycles through `NB = max(2, 72 MB / weight_bytes + 1)` identical weight copies
at different addresses, so consecutive launches miss the 24 MB L2 and the number
is DRAM bandwidth, not cache bandwidth. GB/s counts **weight bytes only**. The
activation vector is 10 KB, read once into cache; the fp32 partials written are
**0.15 to 2.0% of weight bytes at the chosen `S`**, worst at `out/o_proj`
S = 16 (327,680 B against 16.71 MB), which is worth remembering when reading a
GB/s figure that ignores them. Correctness is checked against a
double-accumulating CPU reference at **every** one of the 50 int4
configurations, and all passed.

Best `S` per shape in the canonical layout 1, and the `S = 1` row it is measured
against:

| shape | K×N | subgroups at S=1 | S=1 GB/s | chosen S | GB/s | % of 600 | µs |
|---|---|---|---|---|---|---|---|
| out/o_proj | 6144×5120 | 320 | 259 (43%) | **16** | 533 | 89% | 31.3 |
| q‖k‖v | 5120×14336 | 896 | 540 (90%) | **1** | 540 | 90% | 72.2 |
| qkv‖z | 5120×16384 | 1024 | 559 (93%) | **1** | 559 | 93% | 79.7 |
| gate‖up | 5120×34816 | 2176 | 426 (71%) | **4** | 536 | 89% | 176.7 |
| down | 17408×5120 | 320 | 261 (44%) | **16** | 531 | 89% | 89.1 |
| `lm_head` bf16 | 5120×248320 | 15520 | 584 (97%) | | 584 | 97% | 4350.5 |

`S` is the smallest value within 3% of that shape's best. The engine binds
exactly these rows: layout 1 everywhere, `S` = 16 / 1 / 1 / 4 / 16, the same
picks the probe made (`model::Qwen35`'s table in `src/model/qwen35.cc`). So the
probe's µs are the engine's µs, and multiplying by the layer counts
`model::Qwen35::layers()` produces is a per-kernel measured attribution rather
than a model:

| linear | K×N | `S` | layers | µs each | ms/token |
|---|---|---|---|---|---|
| `in_proj_qkv‖z` | 5120×16384 | 1 | 48 GDN | 79.7 | 3.826 |
| `out_proj` | 6144×5120 | 16 | 48 GDN | 31.3 | 1.502 |
| `q‖k‖v` | 5120×14336 | 1 | 16 FA | 72.2 | 1.155 |
| `o_proj` | 6144×5120 | 16 | 16 FA | 31.3 | 0.501 |
| `gate‖up` | 5120×34816 | 4 | 64 | 176.7 | 11.309 |
| `down` | 17408×5120 | 16 | 64 | 89.1 | 5.702 |
| `lm_head` (bf16, next section) | 5120×248320 | | 1 | 4350.5 | 4.351 |
| **total, 257 launches** | | | | | **28.346** |

**28.35 ms of the 42.141 ms step measured that week, 67.3%**, against a 26.34 ms
roofline for the whole token. Two readings, both uncomfortable in the right
direction:

- **These kernels are close to done.** 28.35 ms is 107.6% of the roofline for
  *all* the model's bytes, which is what 89 to 97% of 600 GB/s comes to. There
  is about 2.0 ms of headroom in the entire GEMV family, and it is the last
  place to look rather than the first.
- **They are also 89% of vLLM's whole token.** Everything that is not a GEMV
  therefore has a small budget if this engine is to lead on decode, and at the
  time it spent 13.1 ms.

The one caveat on transplanting probe µs into a step: the probe replays 40
*independent* launches in an in-order list as the engine does, and cycles at
least 72 MB of weight copies so the 24 MB L2 is missed, which is also the
engine's case, since a token reads 15.5 GB of distinct weights exactly once.
What the probe cannot see is a stall the *previous* kernel leaves behind, so
read 28.35 as a floor for the GEMV family and not as an exact charge.

**In situ, measured 2026-08-25: the floor was right to 2.3%.** `b70-decode
--profile --depth 4096 --steps 32` times every launch inside the replayed list.
The per-kernel durations exclude the profiler's own flush, which is what makes
them comparable to the probe's. The same 257 launches cost **29.005 ms**,
against the 28.346 ms transplanted above:

| shape | `S` | probe µs | **in-situ µs** | delta | in-situ ms/token |
|---|---|---|---|---|---|
| `qkv‖z` 5120×16384 | 1 | 79.7 | **81.401** | +2.1% | 3.907 |
| `out_proj` / `o_proj` 6144×5120 | 16 | 31.3 | **33.499** | **+7.0%** | 2.144 |
| `q‖k‖v` 5120×14336 | 1 | 72.2 | **73.550** | +1.9% | 1.177 |
| `gate‖up` 5120×34816 | 4 | 176.7 | **182.377** | +3.2% | 11.672 |
| `down` 17408×5120 | 16 | 89.1 | **89.513** | +0.5% | 5.729 |
| `lm_head` (bf16) | | 4350.5 | **4375.557** | +0.6% | 4.376 |
| **total, 257 launches** | | **28.346** | | **+2.3%** | **29.005** |

So the stall-from-the-preceding-kernel effect the probe could not see is real
and small. `out/o_proj` is the only shape where it is worth naming: +7.0%, and
it is the shape entered directly out of `attn_reduce` / `prep_gated_head`, the
two kernels in the step with the least data in flight.

**Re-measured 2026-08-24 after a shape correction.** The 2026-08-23 run had
`out/o_proj` at (5120, 5120); the model's output projections are 6144 to 5120
([03-models.md](03-models.md)). The whole matrix was re-run at the corrected
shape so that no number here is a mix of two sessions. The decision did not
change: canonical layout still 1, `S` still 16 / 1 / 1 / 4 / 16, the layout
margin still 1.7%. `out/o_proj` grew 20% in weight bytes (13.93 to 16.71 MB) and
behaves the same.

Four things worth reading off the matrix:

1. **Split-K is worth about 2× exactly where it was predicted to be.** Both
   N = 5120 shapes double: 259 to 533 GB/s (2.06×) and 261 to 531 (2.03×).
   N = 5120 is the worst fill case, confirmed.
2. **Split-K is not universal.** At N = 14336 and N = 16384 the grid already
   fills the device and `S = 1` is the *best* setting, not merely adequate;
   splitting there costs a few percent. A kernel that always split K would be
   leaving that on the floor.
3. **Subgroup count is necessary, not sufficient.** `gate‖up` has 2176 subgroups
   at `S` = 1, more than twice `qkv‖z`'s 1024 which is already at 93%, and still
   only reaches 71% until `S` = 4. So "fill the device" is the right first-order
   story for N = 5120 but does not explain everything the probe sees, and this
   document does not pretend otherwise.
4. **The `S` curve is not monotonic, and mostly the bumps are real.**
   `out/o_proj` reads 259 / 451 / 467 / 453 / 533 across S = 1..16 and `down`
   reads 261 / 470 / 482 / 447 / 531. A second full run picks the identical `S`
   everywhere, agrees to a median of 0.21% per cell, and reproduces
   `out/o_proj`'s dip at S = 8 exactly. One cell does **not** reproduce: `down`
   layout 1 S = 4 reads 482 in one run against 429 in the other, an 11.0%
   spread. It decides nothing (`down` picks S = 16 in every run) and is recorded
   rather than smoothed. Do not "clean up" these numbers by assuming bigger `S`
   is monotonically better, and do not treat a single cell as settled at better
   than about 10% without a second run.

### The sixth shape: `lm_head` at int4

Everything above prices the five per-layer shapes. There is a sixth, and it is
the only one that is not per-layer: when a checkpoint ships a packed head,
`lm_head` is a `gemv` at **5120×248320, S = 1, layout 1** instead of a
`gemv_bf16`. The published checkpoint ships a bf16 head and this engine quotes
only the byte-matched configuration ([BENCHMARKS.md](BENCHMARKS.md) says why),
so the int4 head is a compiled capability rather than the default path. Its
measurement is kept because it is the cleanest bandwidth row in the document.

**Nothing in `gemv.cl` changed.** One variant was added to the build
(`src/kernels/CMakeLists.txt`) and one binding site in `src/runtime/capture.cc`
dispatches on the loaded weight's kind. Launch count unchanged (774); module
count unchanged (19), one module swapped for another.

**The output goes straight into `logits`, and `S = 1` is what makes that legal.**
`gemv.cl` writes `out[(s·M + m)·N + n]`. At `S = 1` that collapses to `[M][N]`,
which *is* the fp32 layout `argmax_stage1` reads. So the capture binds `logits`
as the `out` argument and the split-K accumulator's degenerate case becomes the
logits row itself: no `partials` slice, no rebinding of `argmax`, no direct-out
variant of this kernel. The capture compiles `kCapM = 1`, so the launch writes
1·1·248320·4 = **993,280 B** into a `logits` allocation sized for
`DecodeBuffers::kM = 8` tokens in flight, 7,946,240 B, which is 8× headroom
rather than an exact fit. `gemv()` carries a `require` that the bound output
holds what the launch actually writes.

`S = 1` was not chosen to make that work; the two reasons coincide.
`N / 64 = 3880` work-groups × 4 subgroups = **15,520 subgroups** against 32
Xe-cores, the same grid at which the bf16 kernel runs at 98.4% of device
bandwidth. Split-K exists to buy threads and there are none left to buy.

**Measured**, `b70-decode --profile --depth 4096 --steps 32`, in situ, mean of
32 profiled replays, 2026-08-26, with the bf16 row beside it from a
**back-to-back run of the same instrument on the same box in the same hour**:

| `lm_head` | bytes/token | µs/launch | GB/s | % of measured 590 | ms/token |
|---|---|---|---|---|---|
| bf16, `gemv_bf16` | 2,542,796,800 | 4381.289 | 580.4 | 98.4% | 4.381 |
| **int4 g64, `gemv`** | **675,430,400** | **1178.164** | **573.3** | **97.2%** | **1.178** |
| Δ | −73.4% | **−3203.1** | | | **−3.203** |

0.675 GB at 590 GB/s is a 1.145 ms floor no kernel can go under, so the int4
launch is **2.9% above the bandwidth floor** and there is nothing further to win
in it. Read the GB/s column carefully: int4 reads *fewer* bytes at a *slightly
lower* rate, and 97.2% against 98.4% is not a regression to chase. The launch is
3.7× shorter, so its fixed costs are a larger fraction of it, and 573 GB/s is
still above every per-layer int4 shape in the table above.

---

## `gemv_bf16` - bf16 × bf16, M ∈ [1,8]

`src/kernels/gemv_bf16.cl`, variants `gemv_bf16_M<M>_K<K>_N<N>`. Two users:
`lm_head` (N = 248320, **2.54 GB read per token**, 16.4% of the weight bytes)
and the GDN `in_proj_a ‖ in_proj_b` projection (N = 96, zero-padded to 128).

**Computes**

```
out[m][n] = Σ_k x[m][k] · w[k][n]        (w bf16, no scales, no split-K)
```

### Work assignment, and why

Identical skeleton to `gemv`: lane per `n`, SIMD16, fp32 accumulator in a
register, one store per lane. The reasoning is the same and is not repeated.

**Two compile-time knobs, `COLS_PER_WG` and `KSPLIT`, and they exist for one
shape.** `COLS_PER_WG` is the output columns per work-group (equal to the
work-group size at KSPLIT 1), so the grid is `N / COLS_PER_WG`; `KSPLIT` is how
many subgroups split one 16-column tile's K, each into its own fp32 accumulator,
merged through SLM. `kernels::gemv_bf16_tiling(N)` picks the pair and
`src/runtime/capture.cc` binds it.

**`lm_head` needs neither knob.** N = 248320 gives 3880 work-groups and
**15,520 subgroups** against 256 EUs, so the grid fills the device on `N` alone.
The measurement settles it rather than arguing it: **584 GB/s, 97% of the
600 GB/s roofline denominator and 99% of the 590 GB/s that `probe_bw` measures
through the same launch path.** There is no headroom for split-K to recover, so
`lm_head`'s variant carries neither define.

Its in-situ time is measured unmoved across the `KSPLIT` restructuring,
**4378.555 to 4379.089 µs/launch, +0.012%**. **Its binary is not
byte-identical, and that is worth stating plainly**: restructuring the K loop
for `KSPLIT` changed IGC's output at `KSPLIT == 1` as well, and an intermediate
loads-in-flight experiment proved this is a real risk rather than a theoretical
one, because it cost `lm_head` 1.3% for a knob worth nothing. So the timing
control is the *secondary* evidence. The primary evidence is the acceptance that
ran after the change: the golden gate element-exact on every determined row, and
`replay_determinism_test` green. This project's build flags pin correctly-rounded
`/` and `sqrt` but say nothing about `FP_CONTRACT`, so "same defines implies
same numerics" is not airtight for any kernel here, which is exactly why the
gate and not a hash is the arbiter.

**`a‖b` needed both knobs, and what it needed them for was measured.** At
`{64, 1}` the whole launch is **8 subgroups**, 2 work-groups of 4, reading
1.31 MB at 26.9 GB/s. What that launch is short of is neither work-groups nor
loads in flight, both of which were tried in situ and were worth nothing:

| what was changed | subgroups | µs/launch | verdict |
|---|---|---|---|
| `{64, 1}`, as it shipped | 8 | **48.774** | the baseline |
| `COLS_PER_WG` 16: 2 work-groups to 8, bit-identical | 8 | **49.127** | **nothing** |
| plus 4 block reads in flight per subgroup, bit-identical | 8 | **49.052** | **nothing**, and it made `lm_head` **1.3% slower** |
| `{16, 4}`, K split four ways | 32 | **13.115** | −1.712 ms/token |
| **`{16, 16}`, K split sixteen ways (shipped)** | **128** | **5.340** | **−2.085 ms/token** |

A subgroup pulls 3.36 / 3.12 / 1.92 GB/s at 8 / 32 / 128 of them; **the launch
was slow because it only ever had eight**. `COLS_PER_WG` 16 is still in the
shipped pair, but for the other half of the job: it spreads the 128 subgroups
over eight Xe-cores instead of packing them onto one.

**The pitfall this cost a day to learn: the unit of parallelism on this device
is the subgroup, not the work-group.** Two kernels with the same subgroup count
measure the same however the work-groups are arranged.

**The split's merge order is part of the kernel's contract**, because it is not
the unsplit build's single ascending chain. Subgroup `q` of a tile holds
k ∈ [q·K/KSPLIT, (q+1)·K/KSPLIT) and the work-group collapses the slices with
`prep.cl`'s tree: `for stride = KSPLIT/2 … 1: if (q < stride) red[q] += red[q +
stride]`, a barrier after each step, so at KSPLIT 16 the strides are 8, 4, 2, 1.
No atomic, no data-dependent branch, fixed grid, so a replayed list gives the
same bits every time. Reordering a sum is exactly what the golden gate exists to
arbitrate, and it did: element-exact, unchanged. `gemv_bf16_test` holds the split
build to the reference tolerance rather than to the unsplit build's bytes, and
it is in fact *closer* to the double-precision reference than the unsplit build
(**4.77e-07 against 2.38e-06** on the test's own fixed-seed inputs), which is
what pairwise summation does.

**No test pins the merge *order*, and none can cheaply.** `gemv_bf16_test`'s
pair bar is the reference tolerance, which at this shape is 3.11e-04 against an
observed 2.15e-06 difference, about 150× of slack, enough to swallow any
re-association of the sixteen slices. What actually holds the order is (a) the
kernel text, which contains no atomic, no data-dependent branch and no
arrival-order dependence, so *which* subgroup finishes first cannot change the
result, and (b) the golden gate. A reordering that mattered would show up there
as a flipped token, not here as a tolerance failure.

### Layout

One layout only: the canonical bf16 tile `[n_tile][k_octet][8 k][16 n]` ushort.
One `intel_sub_group_block_read_us8` per subgroup per 8 `k` streams 256
contiguous bytes; the subgroup's whole K-walk is contiguous, the same property
that decided the int4 layout.

There is deliberately **no layout-0 equivalent to compare against, and none was
measured.** The checkpoint ships `lm_head` row-major `[N][K]`; reading that with
lane-per-`n` would give each lane a stride of `K·2 = 10240` bytes, which is the
pathological access pattern layout 1 exists to avoid, on the single largest
tensor in the model. Building it to lose was not worth the ocloc variants.

### Rejected

- **Split-K for `lm_head`, not built, not measured.** Argued away by the 97%
  above. If a future model puts a small-`N` bf16 matrix on the decode path, this
  conclusion does not transfer; re-measure.
- **A separate small-`N` kernel for `a‖b`, rejected, and still rejected.** The
  original reasoning ("1.3 MB and 2 work-groups against a 15.52 GB/token
  budget") was wrong about the *size* of the prize, since those 2 work-groups
  cost 2.335 ms/token, but right about the remedy: what `a‖b` needed was two
  `#define`s in this file, not a second kernel. Zero-padding 96 to 128 at load
  costs a rounding error.

### Measured

Same probe, same date, same command, and **M = 1 only**. Like `gemv`, this
kernel is written for M ∈ [1,8] but has never been timed above M = 1.
`w.bytes() = 2.54 GB`, 10 timed launches cycling 2 weight copies (5.09 GB
device-resident):

| shape | K×N | µs | GB/s | % of 600 |
|---|---|---|---|---|
| `lm_head` bf16 | 5120×248320 | 4350.5 | **584** | **97%** |

For scale: at 584 GB/s `lm_head` alone is **4.35 ms**, 16.5% of the 26.34 ms
roofline token for 16.4% of the bytes, and 10.3% of the 42.141 ms step measured
2026-08-25.

**The other user, `a‖b` (5120×128), is not in the probe matrix** (the matrix
covers the five int4 shapes and `lm_head`). What timed it is `b70-decode
--profile`, in situ, 2026-08-25, 48 launches per token over 1.31 MB each:

| `a‖b` tiling | subgroups | µs/launch | ms/token | GB/s | kind |
|---|---|---|---|---|---|
| `{64, 1}`, as first shipped | 8 | 48.774 | **2.341** | 26.9 | measured, in situ |
| **`{16, 16}`, since the K split** | **128** | **5.340** | **0.256** | **245** | **measured, in situ** |

The `{64, 1}` row was 2.341 ms/token, 5.6% of the sum of kernel durations, and
**21× the 0.11 ms its traffic is worth** at the measured 590 GB/s. At
`{16, 16}` it is **245 GB/s, 41.6% of the device**, and 2.4× the 2.22 µs its
1.31 MB is worth at full bandwidth, so **at most 0.15 ms/token is left in this
kernel** and a wider split cannot repay the golden-gate run it would need.
`lm_head`, the same kernel at N = 248320, remains at 97% of bandwidth and is
measured unmoved across the change.

---

## `prep` - the between-GEMV kernels

`src/kernels/prep.cl`, one file with several entry points. Everything between
two GEMVs of a decode step lives here: sum the previous GEMV's split-K partials,
add the residual, normalise, activate. The compiled set is
`prep_res_fold` + `prep_norm_finish` (the shipped residual-norm pair),
`prep_res_norm_M1_K5120_SP{0,16}` (the single-launch predecessor, still compiled
and still run by `prep_test` as the pair's reference), `prep_silu_mul_M1` and
`prep_gated_head_M1`.

### Rounding discipline - the substance of these kernels

The oracle is torch, and **torch rounds per op**: a linear's output is bf16, and
every elementwise op widens to fp32 internally and rounds its result back to
bf16. These kernels match that discipline wherever it is cheap, so that the
engine's residual stream stays comparable to the golden tensors op for op:

- a GEMV's split-K partials are summed in fp32 and rounded to bf16 **once**,
  because that bf16 value *is* the linear's output in the reference;
- the residual add is `rne_bf16(f32(resid) + f32(mixer_b))`, bf16 in, bf16 out,
  exactly torch's bf16 add;
- a norm widens the bf16 input to fp32, multiplies by the **fp32 `(1 + w)`**
  weight (the loader's bake, [13-loader.md](13-loader.md)) and rounds the result
  to bf16, which is the reference's `type_as(x)`;
- **inside-op** accumulation (the variance sum) stays fp32 and is *not* matched
  to torch term for term. That residual drift is what the golden gate's design
  absorbs: tokens exact is the gate, tensor cosines are diagnostics.

No kernel here deviates from that discipline.

The payoff is that `tests/kernels/prep_ref.h` is not an approximation but the
same op chain in the same order, so `prep_test` asserts **bit-exact** equality
rather than a tolerance, which is a far sharper instrument. Two spellings had to
be pinned to get there, and both are copied verbatim into the reference:

1. **`1.0f / sqrt(x)`, never `rsqrt(x)`.** `rsqrt` is a roughly 2 ulp
   approximation with no cross-implementation guarantee, so nothing bit-exact
   can be built through it. Correctly rounded `sqrt` followed by a correctly
   rounded divide is reproducible on the host. The same rule binds the GDN and
   FA kernels.
2. **The square-accumulate is an explicit `fma`.** `sum += v*v` may or may not
   be contracted into a fused multiply-add by either compiler; writing the
   fusion explicitly on both sides removes the question instead of relying on
   two different compilers' `-ffp-contract` defaults agreeing.

…and one **build option**, which is the finding of this work:
**`-cl-fp32-correctly-rounded-divide-sqrt` is required on any kernel that is
compared bit-exactly and divides or takes a square root.** OpenCL's default
allows **2.5 ulp** on both `/` and `sqrt`, and this is not theoretical: without
the flag the device's `1/sqrt(mean + 1e-6)` came out 1 to 2 ulp below the
host's, and one `x_out` element in 5120 sat close enough to a round-to-nearest
tie (fp32 product `0x3FB88001`, one ulp above the tie) to fall on the other side
of it. 5119/5120 exact, which is exactly the kind of "almost" a tolerance would
have hidden. `add_ocloc_kernel` grew an `OPTIONS` parameter for it and refuses
`-cl-denorms-are-zero`, which stays forbidden project-wide.

It is now the **project-wide default** (`cmake/ocloc.cmake`), because `gdn_step`
was the second kernel to need it and "the next kernel that grows a divide must
remember the flag" is not a rule anyone can be relied on to follow. Making it
the default is safe because the kernels without a divide or a `sqrt` cannot be
affected by it: every binary was rebuilt and every test came back green.

**The variance tree order** is stated identically in `prep.cl` and `prep_ref.h`,
because the comparison is only exact if both walk it the same way. Each of the
work-group's WG lanes accumulates its own strided slice, lane `i` taking
`k = i, i+WG, i+2·WG, …` in ascending `k`, into one fp32 register with `fma`,
writes it to SLM, and then a fixed pairwise tree collapses SLM: for
`stride = WG/2, WG/4, …, 1`, lane `i < stride` does `red[i] += red[i + stride]`,
with a barrier after every step (so 256 to 128 to 64 to … to 1 at WG = 256). No
data-dependent branch and no atomic anywhere, so a replayed command list gives
the same bits.

### `prep_res_norm` - residual add plus RMSNorm, the single-launch form

```
mixer_b = rne_bf16(Σ_s partials[s][m][k])            (skipped when S_PREV == 0)
r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))  (S_PREV == 0: r_b = resid)
resid[m][k] = r_b                                    (the residual stream)
rstd    = 1 / sqrt(mean_k(f32(r_b)²) + 1e-6)
x_out[m][k] = rne_bf16(f32(r_b) · rstd · norm_w[k])
```

`partials` is fp32 `[S_PREV][M][K]` from the previous GEMV, `resid` is bf16
`[M][K]` read and written in place, `norm_w` is the fp32 `(1 + w)` weight, and
`x_out` is bf16 `[M][K]`, the next GEMV's activation input.

**One work-group of 256 per token; grid `(1, M)`.** RMSNorm's mean is over the
whole row, so the row cannot be split across work-groups without a second
kernel; the work-group *is* the reduction domain. Phase 1 folds the partials in
and stages the row in SLM as fp32, phase 2 reduces, phase 3 rescales, with
barriers between phases and phase 3 re-reading the row from **SLM** instead of
re-widening `resid` from DRAM. **SLM bound: `K·4 = 20 KB` for the row plus 1 KB
for the reduction**, comfortably inside the 64 KB a work-group may allocate.

At M = 1 exactly one work-group runs, and **that is what the measurement found
and what the split below fixes**. This kernel is no longer what the runtime
binds; it is kept compiled and still run by `prep_test`, as the pair's
reference.

### `prep_res_fold` + `prep_norm_finish` - the same norm, in two launches

The kernel above measured **22.3 µs and 17.0 GB/s in situ**, 129 times a token,
and the reason is in its own design paragraph: one work-group is one Xe-core,
and one Xe-core is 16 subgroups.

```
stage A  prep_res_fold(partials, resid, sumsq)        grid (G, M), WG 256
  mixer_b = rne_bf16(Σ_s partials[s][m][k])           (skipped when S_PREV == 0)
  r_b     = rne_bf16(f32(resid[m][k]) + f32(mixer_b))
  resid[m][k] = r_b
  sumsq[g][m] = tree_lanes(Σ_{k ∈ chunk g} f32(r_b)²)

stage B  prep_norm_finish(sumsq, resid, norm_w, x)    grid (W, M), WG 256
  rstd        = 1 / sqrt(Σ_{g=0}^{G-1} sumsq[g][m] / K + 1e-6)
  x_out[m][k] = rne_bf16(f32(resid[m][k]) · rstd · norm_w[k])
```

**`G = 20` at K = 5120 is one element per lane**, which is what makes the design
land where the measurement says it should: 20 work-groups × 16 subgroups =
**320 subgroups against 16**. `W` is stage B's own grid and is also 20, because
the rescale is the *other* pass over the row and spreading it is worth about as
much as spreading the fold (both halves are priced separately below). Both
numbers are `runtime::DecodeBuffers::kNormGroups`, which is also the size of the
640-byte `norm_sumsq` scratch and is in **both compiled variant names**, so a
host that pairs mismatched grids names a binary that does not exist and throws
at capture.

**Stage A has no SLM row.** At chunk = WG = 256 the value a lane squares is the
one it just wrote, so the 20 KB `row[K]` staging disappears and the kernel's SLM
falls from 21 KB to 1 KB. Stage B re-widens `resid` from memory instead, because
SLM does not survive a launch boundary, and those are the bytes stage A wrote a
launch earlier, so they are L2-warm.

**The numerics contract, which is the reason this shape and not another.** The
per-element chain in stage A is the single-work-group kernel's, verbatim: same
`s = 0 … S_PREV-1` order, same two bf16 RNE steps. That chain does not care
which lane runs it, so **`resid` comes out bit-identical**, which `prep_test`
asserts against `prep_res_norm`'s own reference at both prep modes. The **one**
thing that changes is the global Σx² tree: G chunk trees, then a fixed
ascending-`g` fold, where there used to be 256 whole-row lane accumulators and
one tree. That can move `x` in the last bf16 ulp, and the golden gate is the
arbiter ([14-golden-gate.md](14-golden-gate.md)).

Determinism is unaffected: G is fixed at compile time, the `g` loop is in index
order, there is no atomic and no data-dependent branch.

### `prep_silu_mul` - the MLP activation

```
gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16
g_b = rne_bf16(Σ_s partials[s][m][gflat]) ;  u_b likewise
s_b = rne_bf16(silu_f32(f32(g_b)))
x_out[m][k] = rne_bf16(f32(s_b) · f32(u_b))
```

with `silu(x) = x / (1 + exp(-x))`, plain `exp` and never `native_exp`. The
index arithmetic is the loader's `cols_interleave16`: `gate‖up` is one fused
linear whose columns interleave in 16-wide blocks, precisely so that the lane
holding `gate[n]` also holds `up[n]` at a fixed offset
([13-loader.md](13-loader.md)). This kernel is the consumer that pays for that
layout, and its two loads are 64 bytes apart.

**Grid `(5, M)`, WG 256, no reduction, no SLM, no barrier.** Five chunks of 4096
cover 17408; the last chunk covers 1024 and the `k < k1` bound is what makes
that safe. Chunking rather than one giant work-group is free here, since there
is nothing to reduce, and gives the machine five work-groups per token instead
of one.

### `prep_gated_head` - `Qwen3_5RMSNormGated`, one v-head per work-group

```
o_b = rne_bf16(gdn_o[m][h][i])                    (recurrence output -> bf16)
z_b = rne_bf16(Σ_s qkvz_partials[s][m][10240 + h·128 + i])
var = mean_i(f32(o_b)²)                           (128-lane tree)
n_b = rne_bf16(f32(o_b) · (1 / sqrt(var + 1e-6)))
t_b = rne_bf16(f32(gated_w[i]) · f32(n_b))
x_out[m][h·128+i] = rne_bf16(f32(t_b) · silu_f32(f32(z_b)))
```

This is the model's op chain exactly ([03-models.md](03-models.md)): norm, cast,
`×w`, `×silu(z.float())`, cast. `gated_w` is **bf16 plain `w`, with no `+1`**,
the one norm in the model without the increment and the one whose weight stays
bf16, because the reference multiplies in the bf16 domain.

**Grid `(48, M)`, WG 128, one work-group per (v-head, token), one lane per
channel.** The head dimension *is* the reduction domain (128), so the mapping is
forced and pleasant: the variance tree is exactly the work-group's 128 lanes
with one term each, which is also why there is no `fma` in this kernel's
reduction, since a lane's contribution is a single multiply.

**The silu factor is deliberately the last op.** Everything through `t_b` is a
rounded scalar chain the host reproduces exactly; only the final product carries
`exp`'s slack. That ordering is what lets the test hold the norm to the exact
bar while allowing the gate a tolerance.

### What the test asserts

`tests/kernels/prep_test.cc`, against `prep_ref.h`, on random inputs
(partials ~ N(0,1), residual bf16, `norm_w = 1 + U(±0.05)`):

| case | bar | result |
|---|---|---|
| `prep_res_norm` SP=0, `x_out` and `resid` | **bit-exact** | 5120/5120, 5120/5120 |
| `prep_res_norm` SP=16, `x_out` and `resid` | **bit-exact** | 5120/5120, 5120/5120 |
| two-stage SP=0/16, `G`=20, `resid` against the **single-stage** reference | **bit-exact** | 5120/5120 both |
| two-stage SP=0/16, `sumsq` and `x_out` against the **two-stage** reference | **bit-exact** | 20/20, 5120/5120 both |
| two-stage SP=16, `x_out` against the **single-stage** reference | ≤ 2 bf16 ulp where \|ref\| ≥ rms/8 | **0 ulp**, 5120/5120 exact |
| two-stage SP=16 at `W`=1 (stage B on one work-group) | same three bars | bit-exact, hence identical to `W`=20 transitively |
| `prep_silu_mul` random | ≤ 2 bf16 ulp | 17408/17408 exact |
| `prep_silu_mul`, silu argument 30.0 | **bit-exact** | 17408/17408 |
| `prep_gated_head` random | ≤ 2 bf16 ulp | 6144/6144 exact |
| `prep_gated_head`, silu argument 30.0 | **bit-exact** | 6144/6144 |

**A known weakness in the `resid` bar, found by mutating the kernel and recorded
so it is not rediscovered.** `resid` bit-identity is a *necessary* check of
stage A's `s = 0 … S_PREV-1` fold order and a **weak** one. Reversing that loop
in `prep_res_fold`, a deliberate mutation, rebuilt and re-run, **did not fail
`prep_test`**: the 16-slice fp32 sum is rounded to bf16 by `rne_bf16(acc)` the
instant it is formed, and bf16 keeps 8 mantissa bits, so an fp32 last-ulp
reordering has to land on a bf16 rounding boundary to survive. **Estimated**:
reordering 16 fp32 additions perturbs the sum by about √16 = 4 fp32 ulps, a
relative 4·2⁻²⁴, while a bf16 ulp is a relative 2⁻⁸, so an element crosses a
boundary with probability about 4·2⁻¹⁶ ≈ 6×10⁻⁵, which over 5120 elements is
**about 0.3 expected crossings**. Usually none, and on this seed none. Two
things pin the `s`-order in spite of it: `prep_res_norm` is still compiled and
still run by the same test against a reference whose fold loop is the same text,
and the golden gate walks 64 layers of real rows. What is *not* covered is an
edit that changes stage A's `s` loop alone, in a way the single-stage kernel
does not share. Closing it properly would mean grading the fp32 sum **before**
it is rounded, or searching seeds for an input where the order does cross a bf16
boundary. Neither is built.

**The ≤ 2 ulp tolerance is a contract, not an observation.** OpenCL allows
**3 ulp** on fp32 `exp` where the host's `expf` is about 0.5; today's driver
matches the host on every one of the 23552 silu values tested. The bar is kept
because a driver update may change `exp` and must not fail this test, while a
change in the *arithmetic* still will.

The bar on the silu-carrying kernels is held up by the two extra cases: with the
silu argument forced to **30.0f**, `exp(-30) ≈ 9.4e-14` is far below `2^-24`, so
`1 + exp(-30)` is exactly `1.0f` in fp32 on *any* conforming implementation and
`silu(30) = 30.0f` on both sides. Those cases are asserted **bit-exact**, which
pins `s_b`, and pins the gated head's `n_b`/`t_b` chain, the norm intermediates
that are otherwise not directly observable since only `x_out` leaves the kernel.

### Rejected

- **Fusing the norm into the following GEMV's prologue, rejected.** It would
  remove about 128 launches per token. Two numbers say do not bother: the
  per-kernel dispatch cost inside a replayed list is **0.73 µs** (derived in
  situ, [15-step-anatomy.md](15-step-anatomy.md)), so the whole between-GEMV
  launch count of 241 is about 0.18 ms, 0.4% of the step measured 2026-08-25;
  and the traffic these kernels add is about 89 MB per token against the token's
  15.52 GB of weights, **0.58%**. Fusion also has a real cost: the norm's
  reduction is over the whole row, so a fused prologue would need every GEMV
  work-group to either redundantly reduce the row (80 work-groups each summing
  5120 elements) or take a cross-work-group barrier the split-K design
  deliberately does not have.
- **Summing split-K partials with atomics in the GEMV instead of here, rejected
  on determinism**, the same argument as `gemv`'s section. That decision is what
  gives these kernels their `partials` argument in the first place.
- **Subgroup reductions (`sub_group_reduce_add`) instead of the SLM tree, not
  used.** The SLM tree's order is *stateable*, and the whole bit-exactness
  contract rests on the reference reproducing it; a subgroup reduce's internal
  order is the compiler's business. Faster, probably, on a kernel whose
  reduction is not the bottleneck. Not measured.
- **`native_exp` / `native_rsqrt`, rejected outright.** Both would put the
  residual stream somewhere the host cannot follow, for an activation that is a
  rounding error's worth of the step.

### Traffic per token (arithmetic, not a measurement)

| kernel | calls/token | bytes/call | total |
|---|---|---|---|
| `prep_res_norm` (the predecessor) | 129 (2 per layer plus the final norm; layer 0's is the SP = 0 variant, 51,200 B) | 378,880 at SP = 16 (327,680 partials + 20,480 resid r+w + 20,480 `norm_w` + 10,240 out) | 48.5 MB |
| `prep_res_fold` (stage A) | 129 | 348,240 at SP = 16 (327,680 partials + 10,240 resid r + 10,240 resid w + 80 `sumsq` out) | 44.9 MB |
| `prep_norm_finish` (stage B) | 129 | 41,040 unique (80 `sumsq`, re-read by every work-group but one cache line, + 10,240 resid + 20,480 `norm_w` + 10,240 out) | 5.3 MB |
| `prep_silu_mul` | 64 | 591,872 (557,056 partials + 34,816 out) | 37.9 MB |
| `prep_gated_head` | 48 | 61,696 (24,576 `z` + 24,576 `gdn_o` + 256 `w` + 12,288 out) | 3.0 MB |

The shipped family is 44.9 + 5.3 + 37.9 + 3.0 = **91.1 MB per token across 370
launches** (the pair's re-read of `resid` in stage B is the extra 1.7 MB, and it
is L2-warm). At the measured 590 GB/s that is **0.154 ms**; the family measures
1.202 ms, so it is 7.8× its own traffic floor, against the 24× the single-launch
family sat at. The split-K partials are two thirds of the traffic, which is the
price recorded in `gemv`'s section for keeping the replay deterministic.

### Measured - per kernel, in situ (2026-08-25)

`b70-decode --profile --depth 4096 --steps 32`, on the replayed decode list:

| kernel | calls/token | work-groups per call | **µs/call** | **ms/token** | share of step | effective GB/s |
|---|---|---|---|---|---|---|
| `prep_res_norm` (SP16) | 128 | **1** | **22.323** | 2.857 | 6.8% | **17.0** |
| `prep_res_norm` (SP0, layer 0) | 1 | **1** | **13.620** | 0.014 | 0.03% | 3.8 |
| `prep_silu_mul` | 64 | 5 | **9.733** | 0.623 | 1.5% | 60.8 |
| `prep_gated_head` | 48 | 48 | **1.640** | 0.079 | 0.19% | 37.6 |
| **the family** | **241** | | | **3.573** | **8.5%** | |

The traffic arithmetic above is a **24× under-prediction** for the family
(89.4 MB is 0.15 ms at the roofline; the family costs 3.573). What the in-situ
numbers add that no traffic bound could is a **parallelism ceiling whose unit is
the subgroup**. `prep_res_norm`'s one work-group is 16 subgroups,
`prep_silu_mul`'s five are 80, and `a‖b`'s two were 8, so per subgroup they read
1.06 / 0.76 / 3.36 GB/s. `a‖b`'s is three to four times the others because its
subgroup pulls 256 B per load (`intel_sub_group_block_read_us8`) where prep's
pulls 64 B, which is the signature of a kernel latency-bound with one load in
flight per thread.

Worse than bandwidth, in fact: the 320 KB of partials `prep_res_norm` folds were
written by the GEMV that ran immediately before it, so most of that 17.0 GB/s is
**L2** traffic. The kernel is not DRAM-limited at all; it is one core's issue
rate. Launch count is not implicated either: 241 `prep` launches at the in-situ
0.733 µs dispatch gap is 0.177 ms, 5% of the family and 0.4% of the step.

### Measured - the split cut it: 2.893 to 0.484 ms/token

Executed 2026-08-25 with the same instrument, idle box, both runs the whole
engine, before at the binary that carried the `a‖b` K split and after at the one
that added `prep_res_fold` + `prep_norm_finish` to `src/kernels/prep.cl`.

| row | before | after | µs/launch | kind |
|---|---|---|---|---|
| `prep_res_norm` (129 launches) | **2893.180 µs** | | 22.428 | measured, in situ |
| `prep_res_fold` (129) | | **259.495 µs** | **2.012** | measured, in situ |
| `prep_norm_finish` (129) | | **224.391 µs** | **1.739** | measured, in situ |
| **the site, both stages** | **2893.180** | **483.886** | 3.751 | **−2409.294 µs** |
| Σ of all kernel durations | 39621.549 (645) | 37394.336 (774) | | measured, in situ |
| fence wall (profiled) | 40496.734 | 38400.907 | | measured, in situ |

**Stage A alone is 10.2× faster on 20× the subgroups**: 348,240 B in 2.012 µs is
**173 GB/s** against the single-work-group kernel's 17.0. 20× the subgroups
buying 10.2× is the same sub-linear shape the `a‖b` K split measured (16× the
subgroups bought 9.1×).

**Both halves of the split were priced, and the second half is not a rounding.**
A third profile run bound stage B at `W = 1`, the literal single-work-group
finish, with everything else identical:

| stage B grid | stage A µs/launch | stage B µs/launch | family µs/step | against `prep_res_norm` |
|---|---|---|---|---|
| `W = 1` (one work-group) | 1.973 | **9.194** | **1440.586** | −1452.594 µs |
| **`W = 20` (shipped)** | 2.012 | **1.739** | **483.886** | **−2409.294 µs** |

So the fold is worth −1.45 ms and spreading the *rescale* is worth a further
**−0.96 ms**. Writing stage B as one work-group would have banked 60% of the
change; the rescale pass is the same latency-bound walk over the same row and it
wanted the same treatment. Note what `W = 1` costs: 41,040 B in 9.194 µs is
**4.5 GB/s**, worse per byte than the single-work-group kernel it came from,
because stage B's head is a 20-deep chain of `sumsq` loads that nothing overlaps
when only one work-group is live.

**What is left in this pair.** Its own traffic is 50.2 MB/token, which is
**0.085 ms** at the measured 590 GB/s. The pair costs 0.484 ms, **5.7× its
floor**, against the **34.9×** the single-work-group kernel sat at. So the whole
remaining prize is about **0.40 ms/token**, and it would have to come from stage
B's `sumsq` fold (a 20-load dependent chain on the head of every launch, which a
lane-parallel load plus an SLM tree would shorten) or from not re-reading
`resid` in stage B at all. Neither can repay its own golden-gate run at that
size.

---

## `embed_gather` - the token id to the residual stream

`src/kernels/embed_gather.cl`, one compiled variant (`embed_gather_M1`). The
first kernel of a decode step:

```
row            = ctrl[CTRL_CUR + m]        (runtime::Control::cur_token[m])
resid[m][0..K) = embed[row][0..K)          K = 5120, bf16, copied verbatim
```

No arithmetic and therefore no rounding: the embedding table is bf16, the
residual stream is bf16, and the reference op is `embed_tokens(ids)`, a gather
and not a computation. The row moves as raw `ushort`.

### Why it reads the control block instead of taking the id as an argument

This is the kernel that justifies the whole capture-once, replay-per-token
design, so it is worth stating plainly. Level Zero resolves a kernel's arguments
at `zeCommandListAppendLaunchKernel` time, **not** at execute time, and
`tests/l0/arg_capture_test.cc` exists to prove that by binding two different
buffers between two appends of the same `Kernel` object and requiring both to be
written. Good for the timed GEMV harness; fatal for a token id. An id passed as
an argument would be frozen into the list when the list was recorded, and every
replay of that list would gather the same row forever. The alternatives are to
re-record about 650 launches per token, or to drive the step from an immediate
list. Neither has been measured, because a captured list exists precisely so
that whatever recording costs is paid once, and neither needs to be, because the
id can simply live in memory the kernel reads when it runs.

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
this engine, since argmax masks at `kVocabUsed` and the tokenizer's ids are
smaller still, so it means the control block is wrong, which is an engine bug.
Clamping would hide that behind a plausible-looking token and a plausible-looking
continuation. Instead the kernel writes nothing, leaves the residual stream as it
was, and sets `Control::debug_flag = 0xDEAD0001` (the same channel `check_finite`
uses), so the host can see *which* step went wrong. The test covers both `kVocab`
and `0xFFFFFFFF`, checks that `resid` still holds its prefilled sentinel, and
then replays a valid id to show the list still works.

### Rejected

- **`zeCommandListAppendMemoryCopy` instead of a kernel, cannot express this.**
  The copy's source address depends on a value that is not known until the list
  executes, and an appended copy captures its pointers at append time exactly as
  a kernel captures its arguments. The indirection has to happen *on the device*,
  which means a kernel.
- **Passing the id as a kernel argument and re-recording per token, rejected**,
  see above.
- **A wider grid, not built, not measured.** At 10 KB per token the kernel is
  launch- and latency-bound; splitting it adds launches to save nothing.

### Measured

**3.65 µs per token**, median of 5 replays (of 8, first 3 dropped) of a closed
list holding 512 back-to-back launches, printed by `embed_gather_test`. Two
caveats, both honest: the same list at **64** launches per execution measures
**6.59 µs** per launch, and the 6.4 µs submit-and-fence floor recorded in
[07-open-questions.md](07-open-questions.md) (question 5) accounts for only
0.1 µs of that 2.9 µs gap, so something, with device clock ramp within a short
execution the guess rather than the finding, makes short bursts dearer, and the
512-launch figure is the marginal cost on a device already working, which is the
engine's case. And the test's stand-in table is 64 rows (655 KB, L2-resident)
while the real [248320][5120] table is 2.54 GB, so in the engine the row is a
cold 10 KB read. Either way it is one kernel per token against a 42.141 ms step:
**0.0087%**, and the 20 KB of traffic is 0.0001% of the token's 15.52 GB.

---

## `argmax` - deterministic greedy sampling, in two fixed stages

`src/kernels/argmax.cl`, two entry points and two compiled variants
(`argmax_stage1_M1`, `argmax_stage2`; both binaries carry both entry points).
Together they turn one row of `lm_head` logits into the next token and move the
control block on by one step:

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

A global `atomic_max` over 248320 candidates, or the float compare-exchange loop
it really has to be since OpenCL has no float atomic max, is **order-dependent
by construction**: which of two equal logits is seen as "already the max"
depends on which work-group arrives first, and that is scheduling, not
arithmetic. Equal logits are not exotic at the top of a peaked distribution, and
two spellings of the same word is precisely where greedy decoding is fragile, so
an atomic argmax can emit a different token on two replays of the same list with
the same inputs. A replay must be bit-reproducible and the golden gate's bar is
*token-exact* output, so that is disqualifying before any performance argument.

A two-stage tree has no data-dependent order at all: every comparison is between
a fixed pair of SLM slots in a fixed sequence, so the result is a pure function
of the logits. Nor does it look like a performance sacrifice, 243 work-groups
reading 4 KB each and then one reading 2 KB against an atomic version's
serialised update path, but that is reasoning and not a measurement: the atomic
version was never built, because determinism settles it before speed is asked.

### The comparator, and the tie rule

One function, `argmax_better`, used by both stages and repeated in the test's
host reference:

```
(a.v > b.v) || (a.v == b.v && a.i < b.i)      // strictly greater wins;
                                              // an exact tie takes the LOWER index
```

Because ties resolve to the lower index rather than to "whichever the tree saw
first", the *bracketing* of the reduction cannot change the answer, which is
what lets stage 1 and stage 2 use different tree widths (256 lanes over 1024
logits, then 256 lanes over 243 partials) and still agree with a sequential host
scan. It is also the oracle's rule: `torch.argmax` documents that "if multiple
values equal the maximum of the tensor, the first occurrence of the maximum
value is returned" (torch 2.12 docs, checked 2026-08-25), so the golden gate
compares like with like on exactly the inputs where greedy decoding is most
fragile. Lanes with nothing to reduce seed `(-INFINITY, 0x7FFFFFFF)`: `-INF`
loses to every real logit and `0x7FFFFFFF` loses every tie, so idle lanes cannot
affect the result (stage 2 has 13 of them, 243 partials into 256 lanes).

### The 248077 mask, and where the number comes from

`lm_head` is [5120][**248320**] because the tiling wants a round row count, but
the tokenizer defines **248077** ids (`model::Qwen35::kVocabUsed`, the
checkpoint config's `vocab_size`). Rows 248077..248319 hold whatever the
checkpoint stored there, and their logits are ordinary finite numbers that can
perfectly well be the largest in the row; emitting one would be an id the
tokenizer cannot decode. So a candidate at index `>= VOCAB_USED` contributes
`(-INFINITY, idx)` and can never win. Indices `>= VOCAB` are not read at all:
the row is only `VOCAB` wide, and the last work-group's bound is clamped
(243 × 1024 = 248832 > 248320).

The number is baked as `-D VOCAB_USED=248077` in one CMake line rather than
spelled in the `.cl` file, and `argmax_test` plants the row's two largest logits
inside the masked tail, one of them exactly at 248077, and requires the best
*usable* index to come out.

### Work assignment, and why

**Stage 1: chunk 1024, WG 256, grid (243, M).** 1024 logits per work-group is
`runtime::DecodeBuffers::kArgmaxChunk`, and 243 = ⌈248320/1024⌉ is the shape of
`argmax_part` (fp32 [M][243][2]); the two constants are derived from the same
arithmetic on both sides. 243 work-groups over 32 Xe-cores is about 7.6 each,
enough to fill the device without making the stage-2 fold wide enough to need a
third stage. Each lane scans a strided slice into a register pair, then the same
fixed pairwise SLM tree `prep.cl` uses collapses the work-group.

**Stage 2: one work-group, m looped internally.** The fold is 243 pairs per
token, a single tree, and the kernel has to be a single work-group anyway
because it owns the control block's per-step bookkeeping, which exactly one lane
must do. The `m` loop's bound is `ctrl[CTRL_NACT]`, read by every lane, so it is
uniform and the barriers inside are reached by all 256; the barrier at the top
of the loop body is what makes the SLM reuse across `m` safe. `n_active == 0`
would be an empty step (an engine bug) and `out_token[n-1]` would index off the
front of the field, so that case writes nothing rather than something plausible.

**The index travels as a float**, which keeps `argmax_part` a plain fp32 buffer
instead of a struct the host would have to lay out by hand. That is lossless,
not "close enough": every integer below 2²⁴ = 16777216 is exactly representable
in fp32, and the largest index this kernel can emit is 248319, 67× under the
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
| exact three-way tie at 5000, 5001 (same stage-1 group, different lanes) and 90000 (different group) | 5000, both trees' tie-breaks exercised |
| the row's two largest logits in the masked tail (248077 and 248200) | 1000, the best usable index |
| all 248320 logits equal | 0 |
| max at 248076, the last usable index | 248076 |
| a random row, against the host reference | agrees (17977) |

and after every replay: `out_token[0]` is the expected id, `cur_token[0]` is the
same id (fed straight back to `embed_gather`), `pos` advanced by exactly
`n_active`, `debug_flag` still 0.

### Rejected

- **A global atomic max, rejected on determinism**, above. Not measured, and it
  would not matter if it were faster.
- **One work-group scanning all 248320 logits, rejected, not measured.** It
  removes a kernel and the `argmax_part` buffer, but puts 993 KB of streaming
  reads on a single Xe-core out of 32. The launch it saves is under a
  microsecond; the serialisation it buys is much more than that.
- **`work_group_reduce_max` or subgroup reductions, not used**, the same
  argument as `prep`'s variance tree: the SLM tree's order is *stateable*, and a
  built-in reduce cannot carry the index alongside the value anyway, so the tie
  rule would have to be rebuilt on top of it.
- **Packing (value, index) into one `ulong`, rejected as unnecessary.** The fp32
  pair is exact and keeps `argmax_part` a buffer the host can print.
- **Sampling (temperature, top-p, top-k) is not this kernel.** When it runs it
  replaces **stage 2 only**: stage 1's 243 pairs are an argmax-shaped summary
  and a sampler needs the full row, so the right shape is a different stage 1 (a
  partial softmax) rather than a patch on this one. Host-side sampling costs
  0.537 ms/token measured ([BENCHMARKS.md](BENCHMARKS.md), "Host sampling") and
  greedy is unaffected by construction.

### Measured

**4.39 µs per token for both stages**, median of 5 replays (of 8, first 3
dropped) of a closed list holding 512 back-to-back (stage 1 + stage 2) pairs,
printed by `argmax_test`. The same list at 64 pairs per execution measures
**8.88 µs**, the same short-burst penalty `embed_gather` shows and with the same
unexplained cause, so treat this as the marginal cost on a device already
working and not as a constant. The logits sit in one 993 KB buffer that is
re-read every iteration and is therefore L2-resident, which is the engine's case
too since `lm_head` has just written them. Against a 42.141 ms step this is
**0.010%**, and the roughly 1 MB of traffic per token is 0.006% of the token's
15.52 GB. For scale, `lm_head` itself is 4.35 ms, **991× this kernel**. The two
token-boundary kernels together (8.04 µs) are the only two numbers in this
document small enough to stop thinking about.

---

## `gdn_step` - the gated delta-rule decode step

`src/kernels/gdn_step.cl`, variants `gdn_step_M1` (used) and `gdn_step_M2`
(compiled for the M-loop rule). **48 of the model's 64 layers run this**, no
library implements it and `sycl-tla` has no example for it: it is the engine's
original-work kernel, and the one that decides whether the golden gate can pass.

One kernel does the whole GDN mixer between the qkv‖z GEMV and
`prep_gated_head`: the depthwise conv1d update and its SiLU, the q/k l2norm, and
the recurrent state update.

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

That is the model's GDN block op for op ([03-models.md](03-models.md)), and
`tests/kernels/gdn_ref.h` is the same chain on the host, trees included, which
is the point of the next two sections.

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
state load's latency behind. Splitting the 128 v columns four ways gives **192
work-groups**, six per Xe-core, at the cost of convolving each head's channels
four times. The state slice per work-group is `128 × 32 × 4 B = 16 KB`, which is
16 fp32 per work-item: registers, not SLM.

**Why the lane index picks the *column* and the subgroup picks the k-band.**
`state` is k-major, so a row's 128 v are contiguous. Under this mapping a
subgroup's 16 lanes read `32c + lane` at a fixed k-row, **16 consecutive fp32,
one full 64 B cache line per access**, and the column pair's second half is the
next line. The obvious alternative (lane picks the k-row) makes every subgroup
access a 16-way gather of 8 B at a 4 KB stride. Same arithmetic, same registers,
an order of magnitude apart on the load that dominates this kernel's traffic.

**Why the tile is 8×2 and not, say, 16×1.** Columns are independent under the
rank-1 update `S[k][v] += kf[k]·Δ[v]`, so the update needs no communication at
all: no atomics, no barrier. Only the two contractions cross work-items, and for
a fixed column they cross exactly the 16 subgroups, which is a 16-wide SLM tree.
Two columns per lane amortise that tree over two results.

**The trees, stated once and obeyed twice** (`gdn_step.cl` and `gdn_ref.h` carry
this text verbatim, as `prep` does):

- `kv[v] = Σ_k S[k][v]·kf[k]`: band `sgid` contributes
  `Σ_{kk=0..7} S[8·sgid+kk][v]·kf[8·sgid+kk]`, accumulated in **ascending kk**
  with an explicit `fma`, into `kv_red[sgid][v−32c]`. The 16 band partials then
  collapse pairwise: for `stride = 8, 4, 2, 1`, work-items with `sgid < stride`
  do `kv_red[sgid][j] += kv_red[sgid+stride][j]`, barrier after each step.
- `o[v] = Σ_k qf[k]·S[k][v]`: identical shape, in `o_red`.
- the two l2norm sums are a different tree: lane `lid < 128` contributes
  `f32(q_b[lid])²`, one term per lane so a plain multiply and no `fma`, and lane
  `128+i` the k term; each 128-wide array collapses with
  `stride = 64, 32, …, 1`.

A tree's order is a property of the **bands**, not of `c` or `lane`. That is
what lets the host reference walk whole 128-column heads and still land on the
same bits, and it is why nothing here needs an atomic: two replays of the
captured list produce identical bytes, which `gdn_step_test` asserts directly.

**SLM: 6912 B**, 768 B of conv outputs, 1 KB of l2norm sums, 1 KB of normalised
q/k, and 2 KB each for the two reduction arrays. The reduction arrays are laid
out `[16 bands][32 columns]` rather than `[32][16]` so that both the writes and
every tree step are lane-contiguous in SLM. Measured from the compiler's own
`zeinfo`: `slm_size 6912`, `grf_count 128`, `simd_size 16`, `barrier_count 1`,
and **neither a `private_size` nor a `spill_mem_size` entry**, so the 16 fp32 of
state stay in registers, which was the design's one real risk.

### The conv in the prologue, and the four-fold redundancy it costs

Each work-group convolves 384 channels: the 128 **q** and 128 **k** of k-head
`h/3` (the `repeat_interleave(·, 3)` of the model) and the 128 **v** of head
`h`. Every one of the head's four chunks computes all 384, so each channel is
convolved 4 times, and each q/k channel 12 times (3 v-heads × 4 chunks).

That is deliberate. The alternative is a separate `conv1d + silu` kernel per
layer, which costs **one more launch per GDN layer** plus a round trip of the
convolved qkv through DRAM, and it does not even save the redundant *reads*,
since every work-group still needs the full 128-wide q and k for the l2norm
whose reduction domain is the head. The redundant work itself is four taps of
`fma` on 384 channels: 1536 `fma` per work-group against the recurrence's four
passes over the 4096-cell slice, so under a tenth of the group's arithmetic. The
redundant *reads* are 192 × (1536 B of partials + 6144 B of conv weights +
2304 B of ring history) ≈ 1.9 MB per layer against the state's 6.3 MB, and they
are L2-resident by construction, with 12 work-groups reading the same 512 B of q
channels within microseconds of each other. **The measurement below says the
redundancy is not a price at all.**

### Ring ownership - the argument, written out

The conv state is a **ring of 16 slots × 10240 channels of bf16** per layer
(`runtime::DecodeBuffers::conv_ring`), holding raw qkv values, since the
reference's `conv_states` likewise hold the *input* sequence and not the
convolved one. Since up to 12 work-groups compute the same raw value, exactly
one stores it:

- `c == 0` writes this head's **v** channels, `4096 + 128h … +128`;
- `c == 0 && h % 3 == 0` also writes the **q** and **k** channels of k-head
  `h/3`, which that triple of v-heads shares.

Together those cover all 10240 channels exactly once: 48 × 128 v plus 16 × 128 q
plus 16 × 128 k. *Which* work-group owns a channel is a bandwidth decision and
not a correctness one, because the value is identical wherever it is recomputed:
it is `rne_bf16` of a buffer nothing writes during this step.

The correctness argument is about **slots**, not owners. A work-group writes
slots `(pos+m) % 16` for `m < n_active` and reads slots `(pos−1) % 16`,
`(pos−2) % 16`, `(pos−3) % 16`. With ring depth **16 ≥ M + 3** those two sets
cannot intersect, so **no work-group ever reads a slot any work-group is writing
this step**, and that, not a barrier, is what makes 192 independent work-groups
safe without any cross-work-group synchronisation, which Level Zero would not
give inside one launch anyway. Within a work-group the argument is simpler
still: each work-item reads its own channel's three history slots *before* the
loop that writes, and the newer window entries come from its own registers
rather than from the ring.

`gdn_step_test` pins this without going through the reference at all: at
`pos = 5` it blanks ring slots 2 to 4 and requires `gdn_o` to move, then blanks
the *other thirteen* slots and requires `gdn_o` to be **bitwise unchanged**. The
pair proves the window is exactly those three slots and that nothing reads the
slot this step writes; a shared misreading in kernel and reference could not
survive it.

### What the test asserts

`tests/kernels/gdn_step_test.cc` against `gdn_ref.h`, on seeded random inputs
(partials and state ~ N(0,·), conv weights U(±0.5), `negA = −exp(U(−4, 0.5))`,
every ring slot filled). The reference reproduces every rounding and every tree,
so what is left between host and device is only the libm functions OpenCL does
not require to be correctly rounded: `exp` (3 ulp) in the decay, the sigmoid and
the SiLU, and `log1p` (2 ulp) in the softplus. `1.0f/sqrt` is *not* among them,
because every kernel builds with `-cl-fp32-correctly-rounded-divide-sqrt`.

| case | bar | measured |
|---|---|---|
| `pos = 0` (history below zero), `state` | rel ≤ 1e-5 | **3.26e-07** |
| `pos = 0`, `gdn_o` | rel ≤ 1e-3 | **5.38e-07** |
| `pos = 0`, `conv_ring`, all 163840 words | **bit-exact** | 163840/163840 |
| `pos = 5` (history in slots 2,3,4), `state` | rel ≤ 1e-5 | **3.44e-07** |
| `pos = 5`, `gdn_o` | rel ≤ 1e-3 | **4.39e-07** |
| `pos = 5`, `conv_ring` | **bit-exact** | 163840/163840 |
| `pos = 1` (partial clamp: -2, -1 zero, 0 a real slot), `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **3.35e-07 / 4.20e-07** |
| `pos = 15`, M = 2 (write wraps to slots 15 and 0), `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **4.18e-07 / 5.57e-07** |
| `pos = 5`, M = 2, n_active = 2, `state` / `gdn_o` | rel ≤ 1e-5 / 1e-3 | **4.59e-07 / 5.58e-07** |
| slot ownership (device only, no reference) | see above | holds |
| replay of every case | **bitwise** identical | holds |

"Relative error" is per element floored at the tensor's own RMS,
`|got − ref| / max(|ref|, rms(ref))`: `Δ = (v − kv)·β` is a difference of two
same-sized numbers, so an individual state cell can land arbitrarily close to
zero by cancellation and an unfloored ratio there would measure the
cancellation, not the kernel. In practice the floor barely does anything: the
worst `state` element is above the RMS in every M = 1 case, and across the whole
table the floored and unfloored numbers differ by at most 1.3×, which is why
both are printed. The margins are about 20× on `state` and about 1800× on
`gdn_o`; the bars are kept where they are because they are the bars a *driver*
change must not break, not descriptions of today's driver.

The M = 2 variant is compiled for the M-loop rule and run anyway, because
nothing else covers the conv window's intra-step path: at `m = 1` the window is
positions 3, 4, 5 and 6, so **two of its four entries are this step's own raw
values**, token 0's at position 5 and token 1's own at position 6, while
positions 3 and 4 are still ring slots.

### Rejected

- **SIMD32, the shape the SYCL reference uses, not measured here.**
  `vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp` runs
  `sub_group_size = 32`, 8 subgroups of 32, 4 k-rows and 4 v-columns per lane,
  the same 16 cells per work-item and the same 32 columns per group reached from
  the other side. SIMD16 is this project's constraint and the mapping above
  adapts cleanly to it (16 bands of 8 instead of 32 bands of 4), so no blocker
  was hit. What SIMD32 would buy is a contraction that reduces inside one
  subgroup rather than across sixteen, which would be faster and would give up
  the *stateable* reduction order the bit-comparable reference rests on.
  Unmeasured on both counts.
- **State in SLM instead of registers, rejected.** 16 KB per work-group would
  fit, and it would let the state be loaded fully coalesced regardless of the
  tile mapping. It also puts every recurrence access through SLM and caps
  occupancy at four work-groups per Xe-core. The register form compiles with no
  spill and no private memory, which was the condition for keeping it. Not
  measured.
- **`sub_group_reduce_add` instead of the SLM trees, not used**, the same
  reasoning as `prep`.
- **A separate conv kernel, rejected**, argued above.
- **One work-group per head (48), rejected on occupancy**, argued above.
- **`native_exp`, rejected outright**: it would put the decay, and therefore the
  whole recurrence, somewhere the host cannot follow.
- **Atomics for the two contractions, never considered seriously.** A float
  atomic add is order-dependent, and two replays of the captured list would
  differ.

### Traffic per token (arithmetic, not a measurement)

The state is the whole story: `48 heads × 128 × 128 × 4 B` = **3.146 MB per
layer**, read once and written once by the 192 work-groups that partition it.

| item | per layer | per token (× 48) |
|---|---|---|
| `state` read + written | 6.29 MB | 302 MB |
| redundant conv inputs (partials, weights, ring history; 4 to 12× by design, L2-resident) | 1.9 MB | 92 MB |
| `conv_ring` written | 20 KB | 0.98 MB |
| `gdn_o` written | 24 KB | 1.18 MB |

**About 396 MB per token**, of which the state's 302 MB is **1.9% of the token's
15.52 GB of weights**.

### Measured - per kernel, in situ (2026-08-25): 0.733 ms, 1.09× its own floor

`b70-decode --profile --depth 4096 --steps 32`:

| | measured, in situ | |
|---|---|---|
| per launch | **15.276 µs** | 48 launches/token |
| per token | **0.733 ms** | **1.7% of the step** |
| effective bandwidth | **540 GB/s** | 8.25 MB per launch ÷ 15.276 µs, **92% of the measured 590** |
| against its own traffic floor | **1.09×** | 396 MB/token ÷ 590 GB/s = 0.671 ms |

**It is at the lower bound.** The 48×4 grid (192 work-groups, SLM 6912 B, GRF
128, no spills) reaches 92% of what the device can stream, and the four-fold
redundant conv reads this section defends as the price of 192 work-groups
instead of 48 are, measured, not a price at all. The whole kernel, made perfect,
is worth **0.06 ms**, 0.15% of the step. A `CHUNK_V` or GRF-mode retune has
nothing to buy. The occupancy problem this kernel was suspected of is real, but
it was in `prep_res_norm` (1 work-group) and `a‖b` (2), not here.

The 48 launches cost 35 µs of dispatch (0.733 µs each, derived in situ), 4.8% of
the kernel's own time and 0.08% of the step. Kernel count was never the question
here either.

---

## `attn` - decode attention in three kernels

`src/kernels/attn.cl`, three entry points in one file. **16 of the model's 64
layers** run all three, in order, per token: `attn_prep`, `attn_decode`,
`attn_reduce`. Variants are `attn_prep_M{M}` and
`attn_{decode,reduce}_M{M}_L{MAXLEN}_B{ATTN_BLOCK}`. `attn_prep` indexes the KV
caches by absolute position and blocks nothing, so it needs neither suffix,
while the other two bake `MAXLEN` because `attn_part` is strided
`[24][MAXLEN/ATTN_BLOCK][M][258]` and a stride must be a compile-time constant.
The *grid* still comes from `buffers.max_len` at capture, so the variant bound
to a layer must be the one built for that `max_len`.

**`B` is in the name for a reason `L` is not.** `MAXLEN` is checked at capture
(`check_sizes` requires `model.max_len == buffers.max_len`); `ATTN_BLOCK` cannot
be, because its host twin `runtime::DecodeBuffers::kAttnBlock`, which sizes
`attn_part` and sets `attn_decode`'s grid, lives on the other side of a
compiler. Putting it in the name turns a disagreement into "no such binary" at
`zeModuleCreate` instead of a buffer strided one way and written another. It is
the arrangement `prep_res_fold`'s `G` uses, for the same reason.
`kernels::attn_{decode,reduce}_variant` therefore take **three** arguments (`M`,
`MAXLEN`, `BLOCK`), and `src/kernels/CMakeLists.txt` holds the one `ATTN_BLOCK`
value both sides read.

The compiled set is deliberately asymmetric while M = 2 is test-only:
`attn_prep_M{1,2}`, `attn_{decode,reduce}_M1_L{4096,16384}_B64` and
`attn_{decode,reduce}_M2_L4096_B64`, the 4096 rows being the length `attn_test`
allocates. **There is no `attn_{decode,reduce}_M2_L16384_B*`**, so
`kernels::attn_decode_variant(2, 16384, 64)` names no file and the runtime
cannot bind M = 2 attention at the loader's default `max_len`. Whoever turns
M > 1 on adds those two rows to `src/kernels/CMakeLists.txt` first.

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

per (kv-head j, ATTN_BLOCK-position block), per q-head qh of j's six, per token m:
  score_p = (Σ_d attn_q[m][qh][d] · f32(kv_k[p][j][d])) / 16   for p <= pos+m
  online softmax in waves of 16 positions -> (mx, sm) and acc[256]
  attn_part[qh][block][m] = {mx, sm, acc[256]}                 (258 fp32)

attn_reduce(ctrl, attn_part, attn_gate, attn_out)

nb = (pos + m)/ATTN_BLOCK + 1 ; merge blocks 0..nb-1 ascending ; out = acc/sm
attn_out[m][h·256+d] = rne_bf16(f32(rne_bf16(out)) · sigmoid_f32(attn_gate[m][h][d]))
```

That is the model's full-attention block op for op, and
`tests/kernels/attn_ref.h` is the same chain on the host, trees, waves and merge
order included.

**The column map** is the one thing here that is easy to get wrong and silent
when wrong. `q_proj ‖ k_proj ‖ v_proj` is 12288 + 1024 + 1024 = 14336 columns
(S = 1), and inside `q_proj` the 24 heads are **interleaved per head, not two
halves**: head `h` is `[h·512, h·512+256)` and its output gate is the next 256.
k-head `j` is at `12288 + j·256`, v-head `j` at `13312 + j·256`. GQA is 6:1, so
q-head `h` reads kv-head `h/6`.

### Work assignment, and why

**`attn_prep`: grid (28, M), work-group 256.** A head is exactly 256 wide and
the work-group is 256, so work-item `i` owns dim `i` and the RMSNorm's reduction
domain *is* the work-group, the same shape `prep_gated_head` uses for its
128-wide GDN heads. Work-groups 0..23 are q-heads (each also writing its gate),
24..27 are the four kv-heads. 28 and not 32 because k and v of a kv-head share a
work-group: v is never normed and never roped, so it is one extra `rne_bf16` on
a work-item that is already resident, against a whole extra work-group's worth
of launch and scheduling.

The one piece of cross-lane traffic is RoPE, which pairs dim `i` with dim
`i ± 32`. That is why the normalised head goes through SLM (`nrm[256]`, 1 KB)
rather than staying in a register: 64 of the 256 work-items need a value another
work-item computed.

**`attn_decode`: grid (4 kv-heads, MAXLEN/`ATTN_BLOCK` blocks), work-group
256 = 16 subgroups × 16 lanes (SIMD16).** Work-group `(j, blk)` owns one
`ATTN_BLOCK`-position block of one kv-head, 64 positions, and loops over that
head's six q-heads and the tokens in flight. Work-item `lid` owns accumulator
dim `lid`; within a wave, subgroup `lid/16` owns one KV position and its lane
`lid%16` sixteen elements of that position's 256-dim dot.

*Why the kv-head and not the q-head.* The grid stays one work-group per
(kv-head, block), with 4 × MAXLEN/ATTN_BLOCK groups and no extra launch.
**Register-packed GQA runs inside that work-group**: for one token it stages the
six q-heads in 6 KB of SLM, and for every 16-position wave each work-item loads
its own 16 K elements plus its output-dimension V value from all 16 rows into
private registers once. It then walks six independent (mx, sm, acc) states over
that shared K/V wave. The per-head score FMA chain remains t = 0..15 ascending,
the 16-way dot tree remains fixed, and the online-softmax and V FMA chains
remain s = 0..15 ascending; only the order in which independent heads consume an
already-loaded wave moves. `qpack` (6 KB) plus `dot_red` (1 KB) is the kernel's
SLM; K and V are not staged there, and no atomic, grid, reduction order, launch
or module count changes.

*Why it pays.* The earlier q-head-outer walk issued the same K/V messages six
times. The private-register inversion issues them once and reuses the retained
values for the other five heads, avoiding both six-pass message issue and an SLM
K/V round trip. The controlled probe measured the packed form at
139.792 µs/launch against 206.354 for the six-pass base, **−66.562 µs/launch**.
In production, `--profile --depth 4096 --steps 32 --repeats 5` on a fixed
checkpoint measured `attn_decode` **228.024 to 160.829 µs/launch** (−67.195),
which is **−1.075 ms/token** over the 16 launches. Probe and production agree to
0.28% of a launch, against the 2.3 to 4.1% the untouched control kernels drifted
between the same two runs.

*Why the attribution is `attn_decode` alone and not the family net.* The other
two kernels of the trio are **byte-identical and untouched**, and the change
moves neither `ATTN_BLOCK`, nor `nb`, nor the `attn_part` layout, so it cannot
causally move work into or out of them. Their before/after rows are therefore a
**drift control, not a saving**: `attn_reduce` 194.841 to 186.954 (−4.1%) and
`attn_prep` 52.839 to 54.078 (+2.3%), run-to-run noise on kernels that did not
change. Banking the favourable one would inflate the claim; the honest figure is
**1.08 ms/token, ±0.01 run-to-run**.

**One thing besides the order moves, and it is the barrier count.** Counted from
the source at M = 1 (derived, exact): the six-pass base executed
`GQA · (2 staging + WAVES · 6 per wave)` = 6 × (2 + 24) = **156** work-group
barriers per launch; register-packed GQA as first written executed
`1 staging + WAVES · GQA · 7` = **169**, because the head loop carried *two*
fences closing the same `dot_red` write-after-read. Deleting one of them leaves
`1 + WAVES · GQA · 6` = **145**, eleven below the base.

*What the redundant fence cost, and why it is a lesson about transferring a
price.* The 24 deleted instances were priced in advance at the probe's average
barrier rate, **0.242 µs per barrier**, predicting −5.82 µs/launch. In situ they
measured **−2.295 µs/launch, −0.037 ms/token**, 39% of the prediction, so the
marginal price of a *redundant* fence is **0.096 µs**. The deleted fence sat two
instructions before a surviving one, and most of the convergence cost is paid at
the survivor either way. **An ablation that removes all six per-wave fences
prices the *average* barrier, which is the wrong number for removing one of
seven.** The deletion is kept because it is free: byte-identical output, no
arithmetic change, and −0.037 ms still resolves at 3.75× this instrument's
floor.

*Why the lane index picks the dim within a position, and not the position.* At a
fixed `t` the 16 lanes of a subgroup read `kv_k[p][j][lane + 16t]`, 16
**consecutive** bf16, one 32 B access. The alternative (lane `l` takes the
contiguous run `[16l, 16l+16)`) spans 512 B in 16 two-byte pieces per step. Same
arithmetic, same registers, an order of magnitude apart on the load that *is*
this kernel.

*Why 16 subgroups × 16 positions and not one position per work-item.* Giving
work-item `p` the whole 256-dim dot for position `p` needs no reduction at all,
and makes every access a 2 KB-strided gather, because the 256 work-items would
be reading 256 different KV rows at the same dim. The subgroup form pays one
16-wide SLM tree per position to keep every load coalesced.

**`attn_reduce`: grid (24, M), work-group 256.** One work-group per (q-head,
token), work-item `d` owning dim `d`. The `nb` block headers are staged into SLM
in a single pass so the merge loop has no barrier at all, and every work-item
then runs the *identical* scalar merge alongside its own `acc`: identical inputs
in an identical order give identical bits, so nothing has to be published and no
work-item diverges. That is cheaper than electing one work-item to compute the
scalars and broadcast them, which would cost a barrier per block.

**`zeinfo`** (the compiler's own report, `-device bmg-g31`, `M=1, MAXLEN=16384,
ATTN_BLOCK=64`, which is the binary the engine binds), for the six-pass base and
for register-packed GQA:

| kernel | six-pass base | register-packed GQA |
|---|---|---|
| `attn_prep` | simd 32, slm 2048, grf 128, barrier 1 | **unchanged** |
| `attn_decode` | simd 16, slm **2048**, grf 128, barrier 1, no `private_size` | simd 16, slm **7168**, grf 128, barrier 1, **`private_size: 1152`** |
| `attn_reduce` | simd 32, slm **2048**, grf 128, barrier 1 | **unchanged** |

Three things to read off it. **(1)** `attn_decode`'s `slm_size 7168` confirms
the source arithmetic exactly: `GQA · HD · 4 + WG_DEC · 4` = 6144 + 1024 = 7168 B
(six 256-float q heads plus the 256-float dot tree), against the 64 KB a
work-group may have. **(2) There is no `spill_mem_size` entry on any of the
three, in either column.** The register-pressure worry (`kreg[16]` + `vreg[16]`
as ushorts, `sc[16]`, and `mx`/`sm`/`acc` × 6, which is 50 to 66 dwords per lane
against 64 at SIMD16 / 128 GRF) did **not** turn into spill, and `grf_count`
stayed 128. **(3) `private_size: 1152` appears where the base had no entry at
all**, so something in the wave's private arrays is memory-backed rather than
register-resident. 1152 B is per the compiler's own unit and this document does
**not** claim to know whether that is per thread or per work-item, nor how much
traffic it generates, only that "nothing is memory-backed" is no longer true,
which is why the traffic floor below carries an explicit exclusion for it.

### The early-out - the answer to a grid that cannot be re-sized

Under replay the attention grid is baked at capture and sized for `max_len`, so
at a context of `pos` tokens most of `attn_decode`'s work-groups have nothing to
do. Each one tests **one uniform condition before touching anything**:

```
if (block_start >= pos + n_active) return;
```

The test is per **block**, not per token in flight. A block live for any `m`
runs for every `m`, and a position outside a given `m`'s causal bound is masked
to −INF inside the wave, so a partially valid block writes real partials for
each `m`, and the all-masked-for-this-`m` case falls out of the online update as
`(−INF, 0, 0)` with no special case anywhere. The pairing that makes this safe
is with `attn_reduce`, which reads exactly `nb(m) = (pos+m)/ATTN_BLOCK + 1`
blocks: every one of those starts at or before `pos+m` and therefore ran, so the
reducer never reads a block the early-out skipped and needs no data-dependent
skip logic of its own. `attn_test` asserts both halves at once: it canary-fills
`attn_part` with 1e30 *inside the replayed list*, requires every block at or
beyond `pos + n_active` to come back still holding the canary, and requires
`attn_out` to match the reference, which it could not if the merge had read a
1e30 header (that block would win the running max outright and drive the output
to 1).

**The early-out is free, measured twice.** A whole-step comparison that holds
depth and live blocks identical and changes only the compiled `MAXLEN` and grid
put it at **0.046 ms/token, 0.11% of a step**
([07-open-questions.md](07-open-questions.md), question 12, resolved). Per
kernel, a 1024-work-group grid whose work-groups all early-out costs **1.667 µs,
or 1.63 ns per work-group**. Context-bucketed lists are not needed.

### The wave: an online softmax with a stateable order

A block's 64 positions are walked in **4 waves of 16**. The wave is 16 because
the work-group has 16 subgroups. Everything below is stated identically in
`attn.cl` and `tests/kernels/attn_ref.h`, because the reference reproducing it
bit for bit is what buys the tight bars.

**The score dot.** Subgroup `s` owns the wave's position `p_s`; its lane `l`
accumulates the 16 elements `d = l + 16t`, `t = 0..15` **ascending**, with an
explicit `fma`, into `dot_red[16s + l]`. The 16 lane partials collapse with a
fixed pairwise tree, for `stride = 8, 4, 2, 1`,
`dot_red[16s+l] += dot_red[16s+l+stride]`, and `dot_red[16s] · 1/16` is the
score (1/16 = 1/√256). A masked position skips the dot entirely and is
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
single case that does not work out is `nmx = −INF`, every position of the wave
masked *and* no earlier valid position, where `exp(−INF − (−INF))` is a NaN;
that wave is skipped whole (`resc = 1`, all weights 0). It is not a
hypothetical: at `pos = 256` block 1 holds exactly one valid position, in wave 0
subgroup 0, and its other waves take that path.

Second, **redundant is cheaper than elected here.** The obvious shape is to let
one subgroup own `(mx, sm)` and publish the rescale factor and the 16 weights
through SLM; that costs an extra barrier per wave and leaves 240 work-items idle
through a serial 16-iteration loop. Recomputing the same scalars in every
work-item costs 16 SLM reads and some ALU that is free next to the KV loads, and
it is *more* obviously deterministic, not less.

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
inside the loop, because block `b < nb` starts at or before `pos+m` and
therefore contains at least its own first position inside the causal bound, the
same fact that lets the early-out leave blocks `≥ nb` untouched. So no guard is
needed here and none is written; the −INF case exists only in `attn_decode`'s
wave.

`nb` is nonetheless clamped to `NBLOCKS` before the header-staging loop. Under
the real precondition, `pos + n_active ≤ max_len`, which the engine enforces
when it advances `Control::pos`, the clamp is dead code. It is there so that a
*violated* precondition costs a wrong answer instead of an SLM overrun writing
past `hmx`/`hsm` into whatever the compiler laid out next, which is the kind of
failure that reproduces as something else entirely three kernels later.

### The RoPE rounding - the one op this trio does not match torch on

RoPE is applied to the **fp32 widened normalised value**: `attn_q` keeps the
fp32 result and the k written to the cache is `rne_bf16` of it, one rounding for
the whole pair. torch reaches the same value through bf16 tensor ops and
therefore rounds once more, *inside* the cos/sin multiply-add. This is a **≤
1-op difference** against the reference, taken deliberately rather than by
oversight, and it is recorded here because it is the only place in this trio
where the kernel is not a per-op transcription of the oracle. Everything else,
the linear's rounding, the norm's `type_as`, the gate's widening, the two
roundings in the final gated product, is torch's op chain exactly. The rotation
itself is one rounded product plus one `fma`, spelled identically on both sides:

```
out_i      = fma(x_i,      cos_i, −(x_{i+32}·sin_i))
out_{i+32} = fma(x_{i+32}, cos_i,  (x_i·sin_i))
```

with `cos/sin` from the loader's table, whose angles were computed in `double`
([13-loader.md](13-loader.md)). Dims 64..255 pass through untouched, the
`partial_rotary_factor 0.25` of the model.

### What the test asserts

`tests/kernels/attn_test.cc` against `attn_ref.h`, on seeded random inputs (qkv
partials ~ N(0,1), the FA block's fp32 `1 + w` norms ~ U(0.75, 1.25), the
loader's real RoPE table, and **the entire KV cache** filled with random bf16,
not just the prefix, so a read past the causal bound shows up as noise rather
than as a convenient zero).

Because the reference reproduces every rounding, every tree and every wave, the
only thing left between host and device is the one libm function OpenCL does not
require to be correctly rounded and this trio uses: **`exp`, 3 ulp**, in the
softmax, in the merge and in the final sigmoid. `1.0f/sqrt` is not among them,
which is why the norm, the RoPE and the KV cache are held **bit-exact**.

**11 cases at 9 distinct depths.** Eight at L4096 M = 1, two at L4096 M = 2, one
on the L16384 binary.

| case | `attn_q` roped dims | `attn_part` | `attn_out` rel (bar **8e-3**) | `attn_out` ulp: worst anywhere / worst at `\|ref\| ≥ rms/8` (bar 2) | blocks still canary |
|---|---|---|---|---|---|
| `pos = 0` (one valid position; 63 blocks early-out) | **0** | **0** | **0** | 0 / **0** of 5282 | 1512 × 1 |
| `pos = 63` (block 0 exactly full) | **0** | 3.837e-07 | 2.673e-05 | 1 (1 word) / **0** of 5264 | 1512 × 1 |
| `pos = 127` (block 1 exactly full; 2-block merge) | **0** | 5.095e-07 | **0** | 0 / **0** of 5264 | 1488 × 1 |
| `pos = 129` (block 2: two valid positions, 3 empty waves) | **0** | 5.459e-07 | 6.051e-04 | 1 (2 words) / **1** of 5307 | 1464 × 1 |
| `pos = 254` (block 3 partial; 255 masked) | **0** | 6.045e-07 | **0** | 0 / **0** of 5278 | 1440 × 1 |
| `pos = 255` (block 3 exactly full) | **0** | 5.719e-07 | **0** | 0 / **0** of 5299 | 1440 × 1 |
| `pos = 256` (block 4: one valid position, 3 empty waves) | **0** | 5.252e-07 | **0** | 0 / **0** of 5267 | 1416 × 1 |
| `pos = 4095` (cache full to `max_len`; 64-block merge) | **0** | 7.599e-07 | 1.995e-04 | 90 (3 words) / **0** of 5305 | 0 |
| `pos = 254`, M = 2 (per-`m` mask **inside** one block) | **0** | 6.009e-07 | 1.285e-05 | 1 (1 of 12288) / **0** of 10562 | 1440 × 2 |
| `pos = 255`, M = 2 (per-`m` mask **across** the block edge) | **0** | 4.818e-07 | 1.270e-05 | 2 (3 words) / **0** of 10480 | 1416 × 2 |
| **L16384**, `pos = 16383` (256-block merge; `hmx`/`hsm` at full length) | **0** | 7.522e-07 | 1.615e-03 | 1 (2 words) / **1** of 5271 | 0 |

and, in **every** case:

- `kv_k` and `kv_v` are **bit-exact over all 4,194,304 words of each cache** at
  L4096 (16,777,216 at L16384), not just the slots this step writes. The
  untouched slots are what prove `attn_prep` writes positions
  `pos … pos+n_active−1` and nothing else.
- `attn_q`'s pass-through dims 64..255 and the whole of `attn_gate` are
  **bit-exact**. The roped dims 0..63 carry the 2 ulp bf16 bar and come back at
  **0** everywhere, which is the `fma` spelling above doing its job.
- replay is **bitwise identical** on all six outputs, from freshly re-uploaded
  inputs.

**`attn_out` carries two bars, and the second one is the arbiter.**

**(a) Relative error ≤ 8e-3, floored at the tensor's RMS.** The *floor* is this
trio's deviation from a plain relative bar and it is not optional: `acc/sm` is a
weighted average of *signed* v values, so a dim can cancel to about 6.6e-10
against an RMS of about 0.07, and an unfloored ratio there would measure the
cancellation rather than the kernel. The `pos = 4095` row's "90 ulp anywhere" is
exactly one such dim, 3.8e-06 of absolute nothing against a 0.07 RMS.

**(b) ≤ 2 bf16 ulp on every element with `|ref| ≥ rms/8`.** Dividing by the RMS
is what makes (a) slack on exactly the elements a bf16 output can most easily
move. bf16 has **7 explicit mantissa bits**, so inside a binade `[2^k, 2^k+1)`
the spacing is `2^(k-7)` and one ulp is between `2^-8` and `2^-7` of the
element, **3.9e-3 to 7.8e-3**, with the top of that range at the *bottom* of a
binade. Two ulp is therefore **7.8e-3 to 1.5625e-2**. So (b) is the one to
trust: **if a change ever trips (a), read (b) before believing the kernel
broke.** The `rms/8` gate is what keeps (b) from being either vacuous or false:
below it a bf16 ulp is not a unit of error at all, above it it is the only unit
that means anything. The **2** is the arithmetic of the final chain rather than
a fudge, since `rne_bf16(f32(rne_bf16(acc/sm)) · sigmoid_f32(gate))` rounds to
bf16 **twice** and 3 ulp of `exp` slack can push a boundary value one ulp at
each of those roundings and no further; 3 would mean an arithmetic difference.

**Bar (a) read 1e-3 until the block retile, and 1e-3 was never consistent with
(b).** One ulp on a gated element reaches 7.8e-3, so (a) at 1e-3 could only pass
while the device happened to reproduce the host bit for bit, which at
`ATTN_BLOCK` 256 it did. Quartering the block reassociates the softmax partials
(4 waves per block accumulate where 16 did, and `attn_reduce` merges four times
as many), and the first gated word to land the far side of a bf16 rounding
boundary took (a) with it. **(a) is now 8e-3 = 2^-7, one gated ulp at its
worst.** That removes the guaranteed contradiction and narrows the remaining one
about eightfold, but it does not remove it: a legitimate 2-ulp gated difference
lands anywhere in `(7.8e-3, 1.5625e-2]`, so one in the residual band
`(8e-3, 1.5625e-2]` would trip (a) while (b) still passes. Nothing measured is
within 5× of that band, and (b) remains the arbiter, so the bar is left at 8e-3
rather than raised to 1.5625e-2 where it would stop covering anything. (a)'s
remaining job is the elements **below** (b)'s `rms/8` gate, where it still
catches any absolute error over 2 ulp of the RMS.

**The measured picture is much sharper than either bar.** The worst gated
distance across all 11 cases is **1 ulp**, on **one word** in each of two cases
(`pos = 129` and L16384 `pos = 16383`); the other nine cases are bit-identical
on every gated element, and the 5264 to 5307 gated words per token are 86% of
the 6144. Against `ATTN_BLOCK` 256's "worst gated 0 in all seven cases", that is
the whole correctness cost of the retile: **two words, one ulp each.**

The M = 2 variant is compiled for the M-loop rule and **run**, because it is the
only cover for per-`m` causal masking: at M = 1 every position in flight shares
one causal bound. At `pos = 254, n = 2` positions 254 and 255 are both in
block 3 and 255 is masked for `m = 0` and valid for `m = 1`, so that block must
produce two different partials for the same (q-head, block); at `pos = 255,
n = 2` block 4 is wholly beyond `m = 0`'s bound, writes `(−INF, 0, 0)`, and
`attn_reduce`, whose `nb(0)` is 4, must not read it. The `−INF` headers are
compared as **exact bit patterns**, not as numbers.

### Rejected

- **SLM-staged K/V tiles instead of private registers.** The loop inversion and
  its six-accumulator cost shipped; what did not is its 16 KB SLM K/V staging
  *medium*. The probe prices the medium directly: **9.687 µs/launch against
  SLM** (149.479 µs for the SLM-staged form against 139.792 for the
  private-register form, measured, device-clocked probe). So this is a rejection
  of the storage choice and not of the inversion.
- **A flash-style single work-group walking the whole `max_len`, rejected on
  fill.** One work-group per (q-head, token) is 24 work-groups per layer on a
  card with 32 Xe-cores: three quarters of the machine idle, on 16 layers per
  token. Splitting the KV into `ATTN_BLOCK`-position blocks turns that into
  4 × (max_len/`ATTN_BLOCK`) work-groups, 1024 at `max_len` 16384 and
  `ATTN_BLOCK` 64, and the price is the two-pass structure (`attn_part` plus
  `attn_reduce`) and one extra launch per layer.
- **One work-group per (q-head, block), rejected**: identical byte count, six
  times the work-groups, and the six passes over a block scattered across
  independent work-groups instead of back to back inside one.
- **A two-pass softmax over the whole block** (all scores to SLM, then one max,
  then a single un-rescaled accumulation), **rejected.** It is simpler and drops
  the per-wave rescale entirely, but it costs another 1 KB of SLM and, more to
  the point, it does not generalise: the online form is what lets the block size
  grow without the score array growing with it.
- **`sub_group_reduce_add` for the score dot, not used**, the same reasoning as
  `prep` and `gdn_step`. A subgroup-scope tree *is* measured, though: narrowing
  three of the four tree barriers to subgroup scope changes **no arithmetic and
  no order at all** and measures **199.271 µs against the base 207.396, −3.9%**,
  a ceiling of about 0.139 ms/token. It is recorded as measured rather than
  taken.
- **`ATTN_BLOCK`: measured at 256, 128, 64 and 32, and the answer is 64.** The
  sweep is at the end of this chapter. 256 was originally chosen because it
  makes the block a whole number of 16-position waves, makes `attn_part`'s
  `[24][max_len/ATTN_BLOCK][M][258]` a size `runtime::DecodeBuffers` can
  allocate, and gives one work-group per Xe-core per 8 blocks at 16k. Only the
  first of those was a real constraint. **Larger** blocks (512, 1024) are not
  measured and now have no motive, since the sweep's slope runs the other way.
  **Smaller than 64** is measured and rejected on the marginal-gain and
  footprint arguments rather than on a sign change.
- **`attn_part` in bf16, rejected outright.** The partials are a softmax
  numerator and denominator; rounding them is rounding the *accumulator* rather
  than an op's output, which is exactly what the rounding discipline forbids. It
  would also halve nothing that matters, since `attn_part` is under 1% of this
  kernel's traffic.
- **`native_exp`, rejected outright**, as everywhere in this project.
- **Atomics for the block merge (one kernel instead of two), never considered
  seriously.** A float atomic add is order-dependent and two replays of the
  captured list would differ.

### Traffic per token (arithmetic, not a measurement)

At context depth `D`, per FA layer, M = 1. Register-packed GQA makes the global
K/V traffic equal to the unique-KV floor: the five old rereads are removed
rather than merely made cache-resident.

| item | per layer at `D` | at `D` = 4096 |
|---|---|---:|
| `kv_k` + `kv_v` read by `attn_decode` | `D·4·1024 B` | **16.8 MB** |
| the five q-head rereads the register packing removed | `5·D·4·1024 B` | **83.9 MB avoided** |
| `attn_part` written then read | `2 · 24 · (D/64) · 258 · 4 B` | **3.17 MB** |
| `attn_q` read by `attn_decode` (six heads staged once per token and block) | `24 · (D/64) · 1 KB` | **1.57 MB** |
| `attn_prep`: partials read, `attn_q`/`attn_gate`/KV written | ~110 KB | 0.11 MB |
| `attn_reduce`: `attn_gate` read, `attn_out` written | ~37 KB | 0.04 MB |

**About 21.7 MB per layer, 346.7 MB per token across the 16 FA layers.**
Arithmetic, not a timing, and **it excludes any compiler private-memory
traffic**. That exclusion is not hypothetical: `zeinfo` reports
`private_size: 1152` on register-packed `attn_decode` where the base kernel had
no such entry. There is no `spill_mem_size` on any of the three kernels, so this
is not spill, but the 21.7 MB is a floor on the *addressed* buffers only.

Launches: **3 per FA layer × 16 layers = 48 per token**, the same order as
`gdn_step`'s 48, and the reason `attn_prep` folds the norm, the RoPE and the KV
write into one kernel rather than three.

### Measured - per kernel, in situ (2026-08-25)

`b70-decode --profile --depth 4096 --steps 32`, at `ATTN_BLOCK` 256, which is
what the trio ran at before the retile:

| kernel | launches | work-groups per launch | **µs/launch** | **ms/token** | share of step |
|---|---|---|---|---|---|
| `attn_decode` | 16 | 4 × 64 = 256 (**68 live** at `nb` 17) | **361.400** | **5.782** | **13.7%** |
| `attn_reduce` | 16 | 24 | **4.748** | 0.076 | 0.18% |
| `attn_prep` | 16 | 28 | **3.204** | 0.051 | 0.12% |

`attn_prep` and `attn_reduce` together are **0.127 ms**, 0.3%, and are not worth
another sentence at that block size. `attn_decode` at 5.782 ms was the largest
non-GEMV row in the step, which is why it got the next two rounds of work.

### Measured - the block retile: the attn family 6.058 to 3.839 ms/token

Executed 2026-08-25, `b70-decode --profile --depth 4096 --steps 32` on an idle
box. **The change is one number, `ATTN_BLOCK` in `src/kernels/CMakeLists.txt`,
256 to 64**, KV positions per `attn_decode` work-group. No arithmetic was
rewritten: the wave is still 16 positions, the online-softmax update inside a
wave is character for character what it was, and `attn_reduce`'s merge is the
same fixed ascending order over more blocks. What moved is *where the
partial-accumulation boundary falls*.

| kernel | before | after | delta | kind |
|---|---|---|---|---|
| `attn_decode` (16 launches) | **5919.814 µs** (369.988/launch) | **3584.736** (224.046/launch) | **−2335.1** | measured, in situ |
| `attn_reduce` (16) | 80.241 (5.015/launch) | 196.195 (12.262/launch) | **+116.0** | measured, in situ |
| `attn_prep` (16) | 58.057 | 58.070 | +0.0 | measured, in situ |
| **the attn family** | **6058.112** | **3839.001** | **−2219.1** | measured, in situ |
| the 726 untouched launches, Σ | 31329.398 | 31503.440 | +174.0 (**+0.56%**) | measured, in situ |
| Σ of all 774 kernel durations | 37387.510 | 35342.441 | −2045.1 | measured, in situ |
| fence wall (profiled) | 38413.428 | 36357.567 | −2055.9 | measured, in situ |

−2219.1 + 174.0 = **−2045.1**, which is the Σ delta to the last digit: the
change's own row plus run-to-run drift on everything it did not touch account
for the whole difference. The launch count did not move.

`attn_reduce` got **2.4× slower** and that is the design working as intended: it
merges `nb = (pos+m)/ATTN_BLOCK + 1` blocks, which at depth 4096 goes 17 to 65.
It costs 116 µs against `attn_decode`'s 2335, a **20:1** trade, which is why the
change is judged on the family's net rather than on `attn_decode` alone.

#### The block sweep, and why it stopped at 64

Four sizes, all at depth 4096, `--steps 32`, same box, same session:

| `ATTN_BLOCK` | live WGs @ 4096 | `attn_decode` µs/launch | `attn_decode` ms/token | `attn_reduce` µs/step | **attn family ms/token** | Σ 774 µs |
|---|---|---|---|---|---|---|
| **256** (before) | 68 | 369.988 | 5.920 | 80.241 | **6.058** | 37387.510 |
| 128 | 132 | 296.684 | 4.747 | 127.240 | **4.931** | 36380.755 |
| **64** (shipped) | 260 | **224.046** | **3.585** | 196.195 | **3.839** | **35342.441** |
| 32 | 516 | 203.837 | 3.261 | 450.208 | 3.769 | 35361.950 |

**64 is not a knee and this document will not call it one.** At `ATTN_BLOCK` 32
the attn family is **3.769 ms/token against 64's 3.839, 0.070 ms better**. The
step's Σ does come out higher at 32 (+19.5 µs), but the non-attn launches
drifted +0.283% (+89.3 µs) between those two runs, so that sign is drift and
cannot carry the argument. Three things can:

1. **The marginal gain has collapsed.** The three halvings bought **−1.127,
   −1.092 and −0.070 ms/token** on the family. The third is 6% of the second and
   roughly a third of one run's drift.
2. **`attn_reduce` is on a steep ramp**, 80.2 to 127.2 to 196.2 to **450.2**
   µs/step, and it is what cancels `attn_decode`'s remaining gain: at 32,
   `attn_decode` gives up 323 µs/step and `attn_reduce` takes back 254.
3. **`attn_part` would double again, 50.7 to 101.4 MB**, taking per-step scratch
   from 77.6 to 128.3 MB. 0.070 ms/token, under the run-to-run drift, is not
   worth doubling the largest scratch buffer in the step.

`attn_part` at 64 is 50.7 MB against 12.7 before
(`tests/runtime/buffers_test.cc` carries the arithmetic).

**What is measured over 256, 128 and 64, and what this document will not do with
it.** Each halving costs about **73 µs less per launch** (73.304 then 72.638),
which is linear in `log2(ATTN_BLOCK)` and *not* in the block. No mechanism here
explains that and none is invented; a fourth point at 32 breaks it too (−20.2,
not −73), so it is a local description over three points and not a law.
**Cost models fitted to this kernel have a bad record**: four were written down
with predictions before the run and all four missed, twice by nearly 40% in
opposite directions. What replaced fitting is a probe that removes one term at a
time, below.

#### What the retile cost in correctness, and what proved it

`attn_part` and `attn_out` are reassociated, so the last bits move.
`tests/kernels/attn_test.cc` re-ran 11 cases at 9 distinct depths with
`attn_ref.h` deriving its blocking from `kBlock` so the reference follows the
kernel by construction:

- `kv_k` / `kv_v` **bit-exact** over the whole cache at every depth;
- `attn_q` roped dims **0 ulp** at every depth;
- `attn_part` worst relative error **7.6e-07**, fp32 noise;
- `attn_out` worst **1 bf16 ulp** at `|ref| ≥ rms/8`, against a bar of 2, on
  one word in each of exactly two cases;
- the early-out asserted directly by canary at the new geometry.

**The golden gate ran before and after and was element-exact both times, with
every diagnostic cosine unchanged to the nine decimals it prints.** The reason
is worth stating rather than celebrating: the gate's longest prompt reaches
`pos` 92, so it exercises **two** 64-position blocks where it exercised one
256-position block. That is real coverage of the merge and no coverage at all of
the 65-block merge that runs at depth 4096. **The deep-depth authority on
attention arithmetic is `attn_test`'s ulp bars, not the golden gate**, and no
change to this kernel should claim otherwise until a deep-context golden prompt
exists.

### Measured - what `attn_decode`'s launch actually buys

`tools/probe/probe_attn` is a **copy** of `attn_decode`, not an include, so
`src/kernels/attn.cl` is untouched by any of it. Every variant computes a wrong
answer on purpose and no runtime path binds one. Instead of fitting a model it
**removes one term at a time**, so every number is a subtraction between two
measurements. 2026-08-25, depth 4096, `ATTN_BLOCK` 64, 4 rotated KV caches,
200 timed launches per row after 120 of warm-up. Every figure is a
device-clocked kernel timestamp, so the CPU load on the box at the time cannot
enter it; no bench-grade wall-clock absolute is recorded here.

Two validations were written down before the run:

- **Base reproduces the launch: PASSED.** 207.396 µs against the in-situ
  223.199 (a 480-replay mean), so in situ is **+7.6%** above the probe floor,
  the same direction and order as `probe_gemv`'s transplant.
- **The block sweep reproduces: FAILED at the ±10% that was written down.**
  182.396 / 207.396 / 261.146 / 349.375 against in-situ 203.837 / 223.199 /
  296.684 / 369.988 is −10.5 / −7.1 / −12.0 / −5.6%. Sign and shape reproduce,
  two of four points miss the tolerance, and the per-halving deltas do not
  reproduce at all. **Transplant fidelity varies across a 6.4-point band with
  block size**, which is carried as a caveat of about ±1 share point on every
  ceiling below.

| term, removed one at a time | µs/launch | share of the 207.396 |
|---|---|---|
| **both KV global loads** | 92.396 | **55.4%** |
| of it, the load messages themselves | | **39.2%** |
| of it, cache service | 173.750 | **16.2%** |
| the whole online softmax update | 153.125 | 26.2% |
| all six work-group barriers per wave | 172.500 | 16.8% |
| `exp` | 175.312 | 15.5% |
| the grid, the q staging and the `attn_part` stores together | | **2.0%** |
| the loop-carried chain across waves | 204.792 | **1.3%** |
| the K load's latency (one wave of prefetch) | 206.354 | **0.5%** |

**The dominant term is the KV load path and it is throughput-shaped**, not a
latency chain. Three independent rows say so: halving the K load's message width
(32 B to 64 B) buys 10.8% and quartering it buys no more, which is what
saturation looks like; the launch is linear in live work-groups (68 to 1004 is
14.8× the work-groups for 11.8× the time); and cutting the wave-to-wave
dependency buys 1.3%. The issue half is arithmetic anyone can check: the kernel
issues **3.19 M global load messages of 32 bytes** per launch (16 lanes of a
subgroup read 16 consecutive `ushort` = 32 B, half a cache line; 16 K + 16 V
messages per subgroup per wave × 16 subgroups × 24 waves × 260 work-groups),
plus about 3.7 M SLM messages. At 207.396 µs that is 0.83 messages per Xe-core
per cycle at about 2.4 GHz, the right order for a pipeline that is issue-bound.

> **The methodology pitfall that nearly voided this table, recorded because it
> inverted the answer.** The obvious way to price a cache miss is to hold the
> address constant so every access hits L1. That address is loop-invariant in
> the wave index over a `restrict` pointer, and IGC hoisted the V load out of
> the wave loop entirely, so the variant measured *a quarter of the messages*
> rather than the same messages served from L1: 141.146 µs where the hoist-proof
> control (an address that is a function of the wave index, same 8 KB
> L1-resident footprint) reads **173.750**. The uncontrolled rows are withdrawn.
> The correction moved the split from 30.9 / 24.4 to **16.2 / 39.2**, inverting
> which half is bigger. **An ablation must be a function of the loop index, or
> the compiler will answer a different question.**

**What is reachable, and what it costs.** Shares are measured; the ms/token
column is **derived** and assumes the probe's share transfers to the live
launch, which sits 7.6% above the probe floor with that offset unplaced. Read
each row as "the probe's share × the measured row", not as a promise.

| change | ms/token | arithmetic risk |
|---|---|---|
| a `[head][pos][dim]` KV cache **and** 64 B K load messages | −0.407 | one reassociation of the K dot |
| the transposed cache alone | −0.268 | **none** |
| 64 B K messages alone | −0.386 | one reassociation of the K dot |
| the tree's three inner barriers at subgroup scope | −0.139 | **none** |

`exp` (−0.553) and the wave barriers (−0.600) are measured and **not
available**: `exp` is ruled by the rounding discipline at the top of this
chapter and `native_exp` with it, and the barriers that remain after the
subgroup-scope narrowing are what publish `dot_red` across subgroups. About
0.45 ms/token is reachable by tuning and the rest is structural, because message
count *within* one pass is set by the (position × dim) decomposition, which is
the algorithm's shape rather than a tuning knob. **What was *not* structural was
the number of times that shape is walked**, which is what register-packed GQA
removed for −1.075 ms/token, more than twice what this battery called reachable
and better than its own all-L1 locality ceiling.

---

## Prefill kernels

Everything below runs **only** on `Engine::prefill`'s walk. None of it is bound
by `src/runtime/capture.cc`, and the decode list is 774 launches and 19 modules
with or without it. Sources are `src/kernels/prefill/`.

Two rules hold across the whole family and are not repeated per kernel:

- **`M` is a runtime argument, never a `-D` and never in a variant name.** One
  binary set serves every `--prefill-chunk` width. What stays in a name is what the
  binary bakes as a stride or a grid: `K`, the previous linear's split-K width
  `S_PREV`, the norm's group counts, and a dequant shape.
- **`S = 1` everywhere.** Decode's split-K exists to buy hardware threads at
  `M = 1`; at `M = C` the row axis already saturates the grid. Leaving decode's
  `S` in place when these kernels were first widened cost a measured 361 to
  473 ms per chunk against 130 ms at `S = 1`.

The prefill path runs entirely on Level Zero. A chunk makes **no SYCL call and
never waits on the host**; `sycl-tla` stays selectable as a reference backend
with `--prefill-backend sycl-tla`, and the build also works with no SYCL component at
all. Derived from the walk in `src/runtime/prefill/step.cc` and asserted by
`prefill_smoke_test` through `Context::launches()`, a chunk at C = 2048 is
**9073 L0 launches, 0 SYCL GEMMs and 0 host waits**; `step_head` adds 5 once per
`prefill()`. Prefill scratch is 35,651,584 B, a tenth of what the SYCL path
needed. Where the time goes, per phase, is in
[BENCHMARKS.md](BENCHMARKS.md).

### The small kernels, transcribed from decode

`pf_res_fold`, `pf_norm_finish`, `pf_silu_mul`, `pf_embed_gather`,
`pf_gated_head` and `pf_ab_proj` are `prep.cl` and `embed_gather.cl` at runtime
`M`, with three classes of edit and nothing else: `M` becomes an `m_count`
argument in every index expression (`sumsq[g * m_count + m]`), the split-K
widths become 1, and `m = get_group_id(1)` needs no mask because the grid's y
extent **is** `M`. Every rounding, both bf16 RNE steps, the `fma`
square-accumulate, `1.0f / sqrt(x)` and both reduction trees are the decode
file's text, which is what lets `tests/prefill/pf_prep_test.cc` hold the `M = 1`
output **bit-identical** to the decode binaries. `pf_embed_gather` loses decode's
`Control::debug_flag` channel for an out-of-vocabulary id, so `Engine::prefill`
bounds every id on the host instead and throws by name.

### `pf_dequant_slab` + `pf_gemm` - one int4 linear, on one in-order list

This is the prefill path's largest phase and the whole of the Level Zero
backend's win over the SYCL one.

`pf_dequant_slab` expands one **1024-column slab** of an int4 linear into a bf16
scratch; `pf_gemm` multiplies the chunk's activations by that slab into the
output columns it covers. For `n0 = 0, 1024, … < N` the pair alternates on
`Context`'s in-order list with **no host wait** between the two, between slabs,
or before the consumer (`src/runtime/prefill/linear_l0.cc`). `pf_dequant_tile`
is the whole-matrix form the SYCL backend still uses.

The int4 contract is `gemv.cl`'s `GEMV_DEQ_SHIFT` path, `(q − 8) × scale` with
the product in fp32 and exactly one RNE cast to bf16, so
`tests/prefill/dequant_test.cc` holds it bit-exact against
`tools/oracle/dequant.py`.

**`pf_gemm` is our own DPAS GEMM**, `src/kernels/prefill/pf_gemm.cl`:

```
C[l][m][n] fp32 = Σ_k A[l][m][k] · B[l][k][n]
```

with a 32-dpas k-tile body, 2D block loads, a two-deep prefetch and a split
work-group barrier. `lda`, `ldb` and `ldc` are runtime arguments; a batch index
`l = get_group_id(2)` offsets A, B and C, so one launch serves a batched GEMM.
Three variants are compiled:

- **`pf_gemm_T0`**, B stored `[K][N]`, the ordinary weight orientation;
- **`pf_gemm_T1`**, B stored `[N][K]` with K contiguous, which is the KV cache's
  layout. The DPAS B operand for one k-step is, per lane `n`, eight 32-bit words
  holding `(B[2t][n], B[2t+1][n])` pairs, and with K contiguous those pairs are
  **adjacent 16-bit words**, so a transposed 2D block load
  (`intel_sub_group_2d_block_read_transpose_32b_16r8x1c`) delivers lane `n`'s 8
  k-pairs directly. Same DPAS instructions, same k order, so the two variants
  are expected to be bitwise equal, and are.
- **`pf_gemm_T0_SILU`**, the `gate‖up` epilogue. A subgroup's four 16-wide
  n-atoms sit at `n0 + 64·sn + 16·b` with `n0` a multiple of 256, so atom
  `b = 2p` is a gate block and atom `b = 2p+1` is its up block, both already in
  this subgroup's registers, one lane per column, with no cross-lane traffic and
  no SLM. The epilogue therefore runs `pf_silu_mul`'s whole chain itself and
  stores bf16, and the fp32 `[M][34816]` intermediate is never written and never
  read. It removes one launch per layer.

`runtime::prefill::gemm_l0` asserts every shape requirement before launching:
`M % 256 == 0` and `N % 256 == 0` (the **caller** pads, and the padding is
computed and never read), `K % 32 == 0`, every base 64-byte aligned, and the
leading-dimension byte strides multiples of 16. The `width` of every 2D block
descriptor is the logical matrix width in bytes and the `pitch` is the leading
dimension, so a read never clamps inside the matrix and never reaches past it.

**Measured.** The kernel this one generalises measured **156.53 TFLOP/s** in its
probe and was bitwise equal to `sycl-tla`
([probe-prefill-vllm-parity-2026-09-14.md](probe-prefill-vllm-parity-2026-09-14.md)).
In production it runs at **161.7 TFLOP/s, about 88% of the measured 183.45
TFLOP/s bf16 DPAS peak** ([probe-dpas-rates-2026-09-22.md](probe-dpas-rates-2026-09-22.md)).
The two halves of a linear were separated with one wait per slab so the dequant
round trip could be attributed rather than estimated
([prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md)): **`slab_dequant`
274.5 ms and `slab_gemm` 1344.3 ms** over 7616 launches each for a 4096-token
prompt, summing to the unsplit phase within an instrumented run's spread.

**What is dead, and each died to a number.** These are the alternatives to the
slab pair, kept because the negative results cost less than re-discovering them:

- **Fusing the int4 dequant into the GEMM mainloop.** Both variants came out
  **bitwise identical** to the two-pass path over all 71,303,168 outputs, and
  both were slower. The dequant pass already runs at about 546 GB/s, roughly 90%
  of hardware peak, so there is only 0.826 ms in it at this shape, and moving
  the same arithmetic into the mainloop costs 2.759 ms even when issued exactly
  once per weight. [Details](probe-fused-dequant-2026-09-22.md).
- **W4A8 on the mixed 4-bit by 8-bit DPAS.** 1.017× the bf16 two-pass control,
  0.965× once the activation quantiser it needs is counted, against a bar of
  1.53×, and **2.79% relative L2 error** driven by activation outliers.
  [Details](probe-w4a8-2026-09-23.md).
- **oneDNN's fused int4 W4A16 matmul.** 98.28 TFLOP/s on `gate‖up` at M = 2048,
  below what an own in-kernel int4 GEMM would need to beat the two-pass path.
  [Details](probe-prefill-vllm-parity-2026-09-14.md).
- **bf16 intermediate buffers.** Halving the bytes changed nothing, because
  `ocloc` exposes no 16-bit block write wider than 8 rows, so the epilogue is
  bound by store message count rather than by bytes.

### `pf_attn_prep`, `pf_softmax_causal`, `pf_attn_gate` - the composed attention

Prefill attention is two `pf_gemm` launches per GQA group and one
bandwidth-bound kernel of ours:

```
S = Q Kᵀ            pf_gemm_T1, batched over the group's six q-heads
P = softmax(S)      pf_softmax_causal
O = P V             pf_gemm_T0, batched likewise
out = O · sigmoid(gate)   pf_attn_gate
```

GQA is a kv-head loop with the six sharing q-heads folded into the batch, which
is bitwise identical to one GEMM at `M = 6C` and gives PV 48 work-groups instead
of 8. `transB` lets the `[pos][4][256]` cache be the operand in place at
`ldb = 1024`, with `strideB = 0` so all six q-heads share it. **QKᵀ is issued
per row block of 256 rows** so the kernel stops computing the masked half, which
is the one term of the launch arithmetic that depends on `C`.

`pf_attn_prep` is `attn.cl`'s `attn_prep` widened: q/k RMSNorm, partial RoPE
over dims 0..63, and the chunk's K and V written into the cache at absolute
positions `[pos, pos+C)`. Grid (24 q-heads + 4 kv-heads, C), work-group 256, so
work-item `i` owns dim `i` and the norm's reduction domain is exactly the
work-group. **Two binaries come from one source.** `pf_attn_prep` writes fp32
`attn_q` and fp32 `attn_gate` and is held **bit-identical to `attn_prep_M1`** by
`tests/prefill/pf_attn_test.cc`. `pf_attn_prep_q16` (`-D Q_BF16=1`) writes bf16
`attn_q`, because DPAS needs bf16 operands, and writes no gate at all, because
`pf_attn_gate` reads the gate columns straight out of `qkv_partials`. Everything
above the store is one piece of text, so the identity test covers the norm, the
tree, the rstd and the RoPE of both, and the bf16 variant is provably **one
`rne_bf16` at the store and nothing else**: `attn_chunk_test` asserts
`pf_q == rne_bf16(` the fp32 `attn_q)` word for word, at every C and depth.

`pf_softmax_causal`: grid (C rows, 6 head slots), work-group 256, **three passes
over `S`**, the row max, the sum of exponentials, then the normalised store,
with the scale `1/16 = 1/√256` applied before the max, one bf16 rounding at the
store, and an exact `+0.0` in the columns past the causal bound so the PV GEMM's
8-element `K` padding contributes nothing. Both reductions are `pf_prep.cl`'s
tree. Three passes cost the same traffic as two plus a rescale, 14 B per element
either way, and round once instead of twice.

`pf_attn_gate` is `attn_reduce`'s tail and nothing else of it: the composed path
has already divided by the denominator, so what is left is
`rne_bf16(f32(rne_bf16(o)) · sigmoid_f32(gate))`, with `sigmoid` spelled as
`1.0f / (1.0f + exp(-x))` exactly as `attn_ref` spells it.

**Measured correctness** (`attn_chunk_test`, C ∈ {1, 64, 256} × depth ∈
{0, 4096}, against `attn_ref` driven with fp32 `q` and again with bf16 `q`):
relative L2 **2.65 to 2.74e-03** against the bf16-q reference at every case, and
**bit-identical at C = 1, depth 1**, where the single valid column makes `P`
exactly 1.0 and the composition has nothing left to differ in. Max relative
difference reaches 2.97e-02 on individual words and the mechanism is arithmetic
rather than a defect: bf16 attention weights perturb a **cancelling** sum
`Σ pᵢ vᵢ`, so the worst word carries a `√n_eff` factor the L2 does not. A 3-ulp
bar is scored and missed for that reason and is recorded as scored, not gated.

### `gdn_chunk` - the chunked gated delta rule

Ten launches per GDN layer (`src/runtime/prefill/gdn.cc`), the
WY-representation chunked gated delta rule at intra-chunk width 64, reading and
writing the **same** `gdn_state` and `conv_ring` buffers `gdn_step` uses, in
place and in decode's layouts, so decode continues from position `pos + C` with
no translation. The chain is seed, conv, l2norm, gate, the WY pair (`pf_gdn_A`,
`pf_gdn_A2`), the triangular solve, `pf_gdn_wu`, the scan, and `pf_gated_head`
as its tenth launch, so `y` leaves the call post-gate.

The bf16 rounds happen **inside** `gdn_chunk`, at the same points `gdn_step.cl`
rounds, which is what the self-consistency bar between the two paths rests on.

**The scan is the one approximation in the default prefill path.** It runs
`pf_gdn_scan_dpas_split`, which keeps fp32 register state and an fp32 global
master state and, at each 64-position sub-chunk, forms two BF16 limbs:

```
S_hi = rne(S)
S_lo = rne(S - bf16(S_hi))
```

Both `W·S` and `Q·S` run independent fp32-accumulating DPAS chains over the two
terms and combine **once**, at the end. Each product is exact, because a BF16 by
BF16 multiply accumulating into fp32 loses nothing, so the only rounding is the
conversion into limbs, and two limbs carry about 16 bits of mantissa instead of
8. It is worth **2.33× on its own profile row**, `gdn_scan` falling from about
297.6 ms to about 127.7 ms profiled ABBA with 96 launches each, and it is gated
on tokens and state cosines rather than on `memcmp`: 93/93 determined rows with
state cosines of 0.999912369, 0.999712545 and 0.999901750 against a bar of
0.999. One honest limit: `vn` inside the A2 term is still a single BF16 operand,
measured harmless on three prompts of one checkpoint and not proven in general.
The design is in
[prefill-gdn-scan-split-2026-09-20.md](prefill-gdn-scan-split-2026-09-20.md) and
the correction that made it correct is in
[prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md).

**Two pitfalls from this family, both bought at full price.**

- **A green gate is not evidence unless the run says which kernel it
  dispatched.** An earlier attempt at this scan passed every gate while the
  engine was launching the old vector kernel. Every gate run now prints the
  entry it launched, read back from the pointer the launcher handed the kernel
  cache, and the non-band tests assert that the printed entry matches the
  selector's resolution. Run with its dispatch proven, that same kernel
  **failed** at 92 of 93 rows, and the cause was one operand of one term.
- **Barriers were not what the triangular solve spent.** The design assumed
  synchronisation dominated `pf_gdn_solve`. Deleting 64 of 128 barriers bought
  1.7 ms of a 73.5 ms row, so the premise was wrong by about 18×; both
  barrier-free variants were bit-identical and both were slower. The row is the
  serialised recurrence itself.
  [Details](prefill-gdn-solve-register-results.md).
</content>
</invoke>
