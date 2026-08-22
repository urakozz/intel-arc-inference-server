# Kernels

One section per device kernel: what it computes, how the work is assigned and
**why**, what was rejected, and the numbers that decided it. A section that only
restates the code is not finished. Written as kernels land (spec 1 §3).

Everything measured here was measured on the box with the probes in
`tools/probe/`. Weight buffers are filled with **random** nibbles, never a
uniform value: the B70 losslessly compresses device-local memory, so a buffer of
one repeated word reads back above the 608 GB/s theoretical peak and any timing
taken from it is fiction (doc 01). Real weights are incompressible within a
cache line; so is the probe's data.

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
**2.35% at `out/o_proj` (S = 16, 327,680 B against 13.93 MB)** - that is the
bound at M = 1, not the `gate‖up` figure. It is also a *write*, which this
kernel otherwise does not do. At M = 8 the same worst case grows to **~19%**
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
int4 shapes), **2712 vs 2666 - 1.7%.** That is inside the noise the rule was
written to tolerate, and the honest reading is this:

| shape | K×N | layout 0 best | layout 1 best | winner |
|---|---|---|---|---|
| out/o_proj | 5120×5120 | **549** | 526 | 0, by 4.4% |
| q‖k‖v | 5120×14336 | **576** | 542 | 0, by 6.3% |
| qkv‖z | 5120×16384 | 420 | **560** | **1, by 33%** |
| gate‖up | 5120×34816 | **554** | 549 | 0, by 0.9% |
| down | 17408×5120 | **566** | 534 | 0, by 6.0% |

**Layout 1 loses four shapes out of five and wins the sum on one.** The one is
`qkv‖z`, and it is the only shape whose `N` is a power of two: at N = 16384
layout 0's row stride is exactly 64 KB, and layout 0 never exceeds 420 GB/s
there at any `S`, while layout 1 hits 560 at `S = 1`. Every other shape has a
non-power-of-two stride (20480, 57344, 139264 bytes) and layout 0 is fine. The
mechanism is almost certainly DRAM channel/bank aliasing on the power-of-two
stride - **that is an inference from the shape of the data, not a measured
cause**; nothing in this project has read a memory-controller counter. What is
measured is that the penalty is real, reproducible, and confined to N = 2^k.

In wall time rather than the rule's GB/s sum, each layout at its own rule-chosen
`S`, one instance of each of the five shapes: **layout 0 459.9 µs, layout 1
442.8 µs - layout 1 is 3.7% faster.** Both metrics agree in direction.

What layout 1 costs, recorded so the next person can revisit it: a load-time
repack pass over 12.16 GB of weights (`repack()`), and the loss of the
zero-copy-from-`mmap` option that layout 0 would have allowed. A 3.7% decode
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
with M, so `out/o_proj` at S = 16 goes from 2.35% of weight bytes at M = 1 to
**~19% at M = 8**, and the rule would very likely choose a smaller `S` there.
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
  traffic argument says it would be worth 2.35% at most at M = 1 (`out/o_proj`,
  S = 16) and ~19% at M = 8 - see above.

### Measured

`probe_gemv`, **2026-08-23**, on the box (`CL_DRIVER_VERSION 26.27.39122.14`,
kernels AOT-compiled by `ocloc -device bmg-g31`, host `-O2`), weights random.
Timing: each configuration is 40 launches recorded in one regular command list
and replayed 8 times, median of the last 5, `µs = list time / 40`. The list
cycles through `NB = max(2, 72 MB / weight_bytes + 1)` identical weight copies
at different addresses, so consecutive launches miss the 24 MB L2 and the number
is DRAM bandwidth, not cache bandwidth. GB/s counts **weight bytes only**
(nibbles + f16 scales). The activation vector is 10 KB, read once into cache;
the fp32 partials written are **0.15-2.4% of weight bytes at the chosen `S`,
worst at `out/o_proj` S = 16** (327,680 B against 13.93 MB) - not "under 1%",
and worth remembering when reading a GB/s figure that ignores them. Correctness
is checked against a double-accumulating CPU reference at **every** one of the
50 int4 configurations - all passed.

```bash
tools/box.sh run ./build/tools/probe/probe_gemv | tee docs/probe-gemv-2026-08-23.md
```

Best `S` per shape in the canonical layout 1, and the `S = 1` row it is measured
against:

| shape | K×N | subgroups at S=1 | S=1 GB/s | chosen S | GB/s | % of 600 | µs |
|---|---|---|---|---|---|---|---|
| out/o_proj | 5120×5120 | 320 | 259 (43%) | **16** | 526 | 88% | 26.5 |
| q‖k‖v | 5120×14336 | 896 | 542 (90%) | **1** | 542 | 90% | 72.0 |
| qkv‖z | 5120×16384 | 1024 | 560 (93%) | **1** | 560 | 93% | 79.5 |
| gate‖up | 5120×34816 | 2176 | 425 (71%) | **4** | 538 | 90% | 176.2 |
| down | 17408×5120 | 320 | 262 (44%) | **16** | 534 | 89% | 88.6 |
| `lm_head` bf16 | 5120×248320 | 15520 | 585 (97%) | - | 585 | 97% | 4349.5 |

`S` is the smallest value within 3% of that shape's best (spec §4.2). The full
51-row matrix, verbatim, is committed as
[probe-gemv-2026-08-23.md](probe-gemv-2026-08-23.md).

Four things worth reading off it:

1. **Split-K is worth 2.0× exactly where the spec said it would be.** Both
   N = 5120 shapes double: 259 → 526 and 262 → 534 GB/s. The prediction that
   N = 5120 is the worst fill case is confirmed.
2. **Split-K is not universal.** At N = 14336 and N = 16384 the grid already
   fills the device and `S = 1` is the *best* setting, not merely adequate -
   splitting there costs a few percent. A kernel that always split K would be
   leaving that on the floor.
3. **Subgroup count is necessary, not sufficient.** `gate‖up` has 2176
   subgroups at `S` = 1 - more than twice `qkv‖z`'s 1024, which is already at
   93% - and still only reaches 71% until `S` = 4. So "fill the device" is the
   right first-order story for N = 5120 but does not explain everything the
   probe sees, and this document does not pretend otherwise.
4. **The `S` curve is not monotonic and the bumps are real, not noise.**
   `out/o_proj` reads 259 / 449 / 485 / 455 / 526 across S = 1..16 and `down`
   reads 262 / 469 / 431 / 448 / 534; a second full run reproduces every cell
   (median deviation 0.21%, worst 2.1%) and picks the identical `S` everywhere.
   Something about how `(N/64) × S` groups land on 32 subslices is being
   sampled here. It is not explained, and the rule does not need it explained -
   but do not "clean up" these numbers by assuming bigger `S` is monotonically
   better.

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
arguing it: **585 GB/s, 97% of the 600 GB/s roofline denominator and 99% of the
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
| `lm_head` bf16 | 5120×248320 | 4349.5 | **585** | **97%** |

For scale: at 585 GB/s `lm_head` alone is **4.35 ms**, against the 25.8 ms a
38.7 t/s token allows - 17% of the step for 16.4% of the bytes. That is why
item 1 of doc 05's specialisation list is `lm_head`, not a kernel.
