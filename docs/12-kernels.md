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

For scale: at 584 GB/s `lm_head` alone is **4.35 ms**, against the 25.8 ms a
38.7 t/s token allows - 17% of the step for 16.4% of the bytes. That is why
item 1 of doc 05's specialisation list is `lm_head`, not a kernel.

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
The flag is opt-in per kernel rather than global so that the already-measured
`gemv` binaries are untouched; no existing kernel contains an fp32 divide or
`sqrt`, so nothing else needs it **today**, and the next kernel that grows one
must add it.

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
  launches - is **~0.125 ms of a ~26 ms step**; and the traffic these kernels
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
