# Kernels

One section per device kernel (a kernel is a function that runs on the GPU): what it computes, how the
work is assigned and **why**, what was rejected and on what evidence, and the numbers that decided it. A
section that only restates the code is not finished. The kernels are OpenCL C in `src/kernels/`, compiled
ahead of time for the B70 with `ocloc`; the host code that launches them is in `src/runtime/`.

This document covers the kernels of the main model, Qwen3.8-27B, for decode and for prefill. The kernels of
the other models live in their own directories and are listed at the end ("Kernels of the other models").

Every figure says which kind it is: **measured**, **derived** (arithmetic over measured inputs) or
**estimated** (a stated guess with its reasoning). Three instruments produce them:

- **standalone probes** in `tools/probe/`, which time one kernel in a closed command list with nothing
  else running;
- **`b70-decode --profile`**, which reads Level Zero kernel timestamps off the replayed decode list, so every
  launch of a real step has a duration. Method and caveats are in [15-step-anatomy.md](15-step-anatomy.md);
  figures taken this way are labelled **in situ**;
- **timing loops inside a kernel's own test**, for the two token-boundary kernels whose cost is a few
  microseconds. Those loops print and never assert.

Weight buffers in every probe are filled with **random** nibbles, never a repeated value: the B70 losslessly
compresses device-local memory, so a buffer of one repeated word reads back above the 608 GB/s theoretical
peak and any timing taken from it is fiction ([01-hardware.md](01-hardware.md)). Real weights are
incompressible within a cache line, and so is the probe's data.

Most per-kernel numbers below were taken in August to September 2026 on the first driver stack, at one card
and depth 4096. They are kept as **dated evidence for why a kernel is shaped as it is**, not as today's
speed. Today's speeds are in [BENCHMARKS.md](BENCHMARKS.md).

**Words used throughout.** *Xe-core*: one of the card's 32 compute blocks. *Sub-group*: 16 lanes that execute
together (SIMD16); the unit of hardware parallelism. *Work-group*: a set of sub-groups scheduled on one
Xe-core (64 to 256 lanes here). *SLM*: shared local memory, a fast per-work-group scratch of at most 64 KB.
*DPAS*: the matrix-engine instruction. *Split-K*: cutting a dot product's long K axis into slices handled by
different threads. *GQA*: grouped-query attention, several query heads sharing one key/value head.
*Replay*: the decode step is recorded once as a command list and re-executed for every token.

---

## The step these kernels add up to

Qwen3.8-27B has 64 layers: 48 **GDN** layers (gated delta-rule, a linear-attention layer whose memory is a
fixed-size recurrent state) and 16 full-attention layers, hidden size 5120, MLP width 17408, attention with
24 query heads and 4 key/value heads of 256 dimensions ([03-models.md](03-models.md)). One decode token is
**774 launches in 18 modules**, recorded once and replayed (`tests/runtime/replay_determinism_test.cc` pins
both numbers). One layer is:

| step | GDN layer | full-attention layer |
|---|---|---|
| residual add + RMSNorm (2 launches) | `prep_res_fold`, `prep_norm_finish` | the same |
| mixer, 5 launches | `gemv` qkv‖z, `gemv_bf16` a‖b, `gdn_step`, `prep_gated_head`, `gemv` out_proj | `gemv` q‖k‖v, `attn_prep`, `attn_decode_v2`, `attn_reduce_v2`, `gemv` o_proj |
| residual add + RMSNorm (2 launches) | the same | the same |
| MLP, 3 launches | `gemv` gate‖up, `prep_silu_mul`, `gemv` down | the same |

That is 12 launches a layer; the other 6 are `embed_gather` first, then the final norm (2), the `lm_head`
GEMV and the two argmax stages: 12 x 64 + 6 = 774. Speculative-decoding lists (several rows per step) bind the
same kernels compiled for M rows, with `gemm_sm` in place of `gemv` for M >= 2 and `attn_verify_dpas` in
place of `attn_decode_v2` where the depth rule says so. The `lm_head` is int8 (`gemv_i8w`) by default.

Plain decode on one card is about **33.5 tokens/s at depth 4096** (int8 `lm_head`, bf16 KV cache; the
tensor-parallel table in [BENCHMARKS.md](BENCHMARKS.md) has the row, and the older decode rows there are on
the bf16 head). Decode is a stream of weight reads (the GEMV family was about 69% of the step in the 2026-08 anatomy below),
so the work in this document is mostly about reading the weights at the memory system's rate.

**The per-launch anatomy of 2026-08-25** (the only complete one; step **42.141 ms/token**, bf16 `lm_head`,
the first attention kernel, 645 launches) split like this, every row measured in situ on an idle box:

| group | ms/token | share |
|---|---|---|
| `gemv` + `gemv_bf16` (`lm_head`) | 29.005 | 68.8% |
| decode attention | 5.782 | 13.7% |
| the between-GEMV `prep` kernels | 3.573 | 8.5% |
| the a‖b GEMV | 2.335 | 5.5% |
| `gdn_step` | 0.733 | 1.7% |
| everything else | 0.7 | 1.7% |

Three kernel changes that day took the step to 36.32 ms: splitting the a‖b GEMV's K axis 16 ways
(2.335 to 0.256 ms), splitting the residual norm over 20 work-groups (2.893 to 0.484 ms), and shrinking the
attention block from 256 to 64 positions (6.058 to 3.839 ms for the attention family). Each has its section
below and its full before/after in [15-step-anatomy.md](15-step-anatomy.md). The per-kernel shares were not
re-measured on later stacks, so none of those shares is re-based onto a later step time here.

---

## `gemv` - int4 g64 × bf16, M = 1

`src/kernels/gemv.cl`, variants `gemv_M<M>_K<K>_N<N>_S<S>_L<LAYOUT>`. The decode linears of every layer.
The kernel is written for M in [1,8]; the engine binds it at **M = 1**, and `gemm_sm` takes over at M >= 2
(`runtime::gemv_kind`, see below).

**Computes**

```
out[s][m][n] = Σ_{k ∈ slice s} x[m][k] · (q[k][n] − 8) · scale[k/64][n]
```

`q` is the 4-bit nibble, symmetric with the constant zero point 8 (the checkpoint's `qzeros` is dropped at
load, [02-formats.md](02-formats.md)); `scale` is one f16 per 64 consecutive `k` per column; `x` is bf16
`[M][K]`; the output is **fp32 partials `[S][M][N]`**, summed by `prep`. Nothing else is in the kernel (no
residual, activation or norm), so split-K stays deterministic and the GEMV stays one thing.

### Work assignment, and why

**Lane per `n`.** Lane `l` of a sub-group owns output column `n_tile·16 + l` and holds its accumulator in a
register for the whole K-range. The weight read is perfectly coalesced by construction (16 lanes want 16
adjacent columns, which are adjacent words in both layouts), and there is **no cross-lane reduction
anywhere**: no shuffle tree, no SLM, no barrier, one store per lane. The alternative, lane per `k` with one
output per sub-group and a reduction at the end, costs a 4-deep shuffle reduction per output column and buys
nothing on a kernel whose entire cost is the weight stream.

**SIMD16** (`intel_reqd_sub_group_size(16)`; the device offers 16 and 32, [01-hardware.md](01-hardware.md)).
16 lanes × 4 bytes is exactly one 64-byte cache line per lane-step, and `intel_sub_group_block_read8` becomes
512 contiguous bytes in one instruction. 16 is also the granularity everything else is cut to: the tiled
layout's tile, the bf16 tile and the loader's `gate‖up` interleave are all 16 columns wide, so the lane
holding `gate[n]` also holds `up[n]` at a fixed offset. SIMD32 was not measured.

**64 `n` per work-group** (4 sub-groups, `reqd_work_group_size(64,1,1)`); the grid is `(N/64) × S`. Four
sub-groups is the smallest group that still hands an Xe-core a useful bundle per dispatch, and a larger group
would coarsen the grid exactly where the grid is already the problem (N = 5120 is 80 groups). It also keeps
`N % 64 == 0` as the only shape constraint.

**Split-K on grid dimension 1, into separate partials.** Work-group `(gn, s)` covers k-groups
`[s·G/S, (s+1)·G/S)` (`G = K/64`) and writes `out[s][m][n]`; `prep` sums the `S` slices in the next kernel.

- *Why split-K exists.* The grid is `N/16` sub-groups at `S = 1`. At N = 5120 that is 320 sub-groups for
  256 execution units, too few loads in flight to saturate DRAM. Measured (below): this costs half the
  bandwidth, and `S` recovers it. At N = 14336 or 16384 the grid already fills the device and `S = 1` is best.
- *Why partials and not atomics.* A `float` atomic add is order-dependent, so two replays of the same command
  list would give different logits. The runtime's premise is one list replayed per token, and the golden
  tests compare bitwise. The price is `S ×` the output bytes: 0.6% of the weight bytes at `gate‖up`, about 2%
  at `out_proj`, and it grows with M (about 16% at M = 8 for the worst shape), which is part of why M >= 2
  has its own kernel.

### The two layouts, and what the engine binds

Both hold the same 4.25 bits per weight (nibbles plus one f16 scale per 64).

- **Layout 0, GPTQ-native**: `qweight[K/8][N]` u32 and `scales[K/64][N]` f16 exactly as the checkpoint
  ships them. A lane covering 64 `k` reads 8 words at stride `N·4` bytes plus one scale from a second array.
  No repack. With `GEMV_BLOCK2D=8` the eight words come from **one 2D block read** (a single hardware message
  fetching 16 columns × 8 rows) instead of eight strided loads.
- **Layout 1, tiled**: per `(n_tile of 16, k_group of 64)` one **544-byte block** (128 u32 of nibbles ordered
  `[k_octet][lane]`, then the 16 f16 scales). A sub-group walks one contiguous run of memory across its whole
  K-range, and the scales travel with the nibbles. The address stream does not depend on `N` at all. It costs
  a load-time repack (`repack()`) and loses the zero-copy mapping.

**Dequantisation** has two spellings: mask-and-subtract, or (`GEMV_DEQ_SHIFT=1`) one word-wide xor with
`0x88888888` and a left/right shift pair per nibble, since for every `q` in [0,16) the sign-extended
`q ^ 8` equals `q − 8`. Both give the identical integer, so the output is bit-identical; `gemv_test` drives
all 16 nibbles through the shift form against a control build (`gemv_control_M1_K6144_N5120_S4_L0`).

**What the engine binds for Qwen3.8** (the table in `src/model/model_desc.cc`, `kQwen38Tuning`, and the
per-shape defines in `src/kernels/CMakeLists.txt`):

| linear | K×N | layers | layout | S | weight load | dequant |
|---|---|---|---|---|---|---|
| in_proj qkv‖z | 5120×16384 | 48 GDN | 1 | 1 | tiled block read | mask |
| out_proj / o_proj | 6144×5120 | 48 + 16 | 0 | 4 | 2D block read | shift |
| q‖k‖v | 5120×14336 | 16 FA | 0 | 2 | 2D block read | shift |
| gate‖up | 5120×34816 | 64 | 0 | 8 | strided loads | shift |
| down | 17408×5120 | 64 | 0 | 4 | 2D block read | shift |

Under tensor parallel (every layer split over both cards) each card's halves take the one-card cell's
settings. Layout 1 for `qkv‖z` is the one place the tiled layout still wins (next paragraph); the other four
shapes read the checkpoint's own layout, so most weights are mapped without a repack.

### Measured

The layout and `S` matrix was measured with `probe_gemv` on **2026-08-24** (first driver stack, M = 1, random
weights, each configuration 40 launches in one list cycling enough weight copies to miss the 24 MB L2, GB/s
counts weight bytes only; correctness checked against a double-precision CPU reference at every one of the 50
int4 configurations). Best GB/s per layout at each shape:

| shape | K×N | layout 0 best | layout 1 best |
|---|---|---|---|
| out/o_proj | 6144×5120 | **549** | 533 |
| q‖k‖v | 5120×14336 | **577** | 541 |
| qkv‖z | 5120×16384 | 419 | **559** |
| gate‖up | 5120×34816 | **555** | 549 |
| down | 17408×5120 | **567** | 531 |

What it showed, and what still holds:

1. **Layout 1 loses four shapes of five.** It won the sum only because of `qkv‖z`, the one shape whose `N` is
   a power of two: at N = 16384 layout 0's row stride is exactly 64 KB and layout 0 never exceeds 419 GB/s at
   any `S`, while layout 1 reaches 559. The cause is presumably DRAM channel or bank aliasing on the
   power-of-two stride, **an inference from the data, not a measured cause** (nothing in this project reads a
   memory-controller counter). What is measured is that the penalty is real, reproducible and confined to
   N = 2^k. That is why `qkv‖z` is layout 1 today.
2. **Split-K is worth about 2× exactly at N = 5120** (259 to 533 GB/s and 261 to 531 at `S = 16` in that run)
   and is not universal: at N = 14336 and 16384 splitting costs a few percent.
3. **The `S` curve is not monotonic** (`out_proj` read 259 / 451 / 467 / 453 / 533 across S = 1..16) and one
   cell of a second run differed by 11%. Do not assume bigger `S` is better, and do not trust a single cell
   to better than about 10% without a second run.
4. The probe's 5-shape sum, transplanted into the real step, came out **2.3% under** the in situ time
   (28.35 vs 29.00 ms for the same 257 launches), worst at `out_proj` (+7%, the shape entered straight out of
   the two kernels with the least data in flight). A probe floor is a floor.

**The production map above is not that run's.** That run picked layout 1 with `S` = 16 / 1 / 1 / 4 / 16.
The engine's present cells came from a later load-path probe (`tools/probe/probe_gemv_loads.cl`, which
removes one load-message source at a time); its numbers are not kept in this document, so treat the table
above as the engine's choice, not as a result reproduced here.

### Rejected or not measured

- **Shuffle-broadcast of the activations** (hold `x` in registers, broadcast to all lanes) instead of the
  plain `vload8` whose 16 lanes read one cache line: not measured. `x` is 10 KB, L1-resident, against a weight
  stream three orders larger. Consistent with 88 to 97% of peak, not proof; the first thing to try if a profile
  ever disagrees.
- **SIMD32**: not measured; it halves the sub-group count where the fill is already the problem.
- **fp32 accumulation vs half**: fp32 was chosen because the partials are summed by another kernel and bf16
  activations already spend the precision budget (worst error 4.2e-05 against a 6.7e-03 tolerance).
- **Atomics instead of partials**: rejected on determinism, not measured.

### The int4 `lm_head`, a sixth shape

When a checkpoint ships a packed head, `lm_head` is a `gemv` at **5120×248320, S = 1, layout 1** (variant
`gemv_M1_K5120_N248320_S1_L1`), instead of a `gemv_bf16`. **`S = 1` is what lets the output go straight into
`logits`**: `gemv` writes `out[(s·M + m)·N + n]`, which at `S = 1` is the fp32 `[M][N]` layout the argmax
reads, so no partial slice and no extra kernel. `N/64 = 3880` work-groups (15,520 sub-groups) already fill
the device. Measured in situ 2026-08-26: **1178 µs, 573 GB/s (97% of the 590 GB/s the device streams)**
against the bf16 head's 4381 µs for 3.8× the bytes. The published Qwen3.8 checkpoint ships a bf16 head and
the engine quantises it to int8 at load, so this int4 form is a compiled capability for checkpoints that
ship a packed head, not the default.

---

## `gemv_i8w` - the int8 `lm_head`

`src/kernels/gemv_i8w.cl`, variant `gemv_i8w_M<M>_K5120_N248320` (the default head of `b70-serve` and
`b70-decode`). The loader quantises the bf16 head at load to int8 with **one fp32 scale per output column**
(`src/loader/lm_head_int8.h`, 1.272 GB read a token against bf16's 2.543 GB). The structure is `gemv_bf16`'s
at its `lm_head` tiling (`{64, 1}`: four 16-column tiles a work-group, no K split): the weight is stored
`[n_tile][k16][16 k][16 n]`, so one `intel_sub_group_block_read_uc16` gives lane `n` its 16 `k` bytes, and the
scale is applied **once to the fp32 sum**, `(x · q) · s`, rather than per element. One ascending chain over
`k`, no atomic, no data-dependent branch, so replays are bitwise.

Measured in situ ([BENCHMARKS.md](BENCHMARKS.md), "int8 lm_head", 2026-09-28): **2205 µs, 577 GB/s**, 99.6% of
the bf16 launch's rate, which is the whole 2.0 ms a token saved. The accuracy rules for the head are in that
section.

---

## `gemv_bf16` - bf16 × bf16

`src/kernels/gemv_bf16.cl`, variants `gemv_bf16_M<M>_K<K>_N<N>[_C<cols>][_S<ksplit>]`. Users: the GDN
`in_proj_a ‖ in_proj_b` projection (**a‖b**, N = 96 zero-padded to 128), the MTP head's linears, the MoE
routers of the other models, and `lm_head` where the head is left in bf16 (tests and the byte-matched vLLM
comparison).

```
out[m][n] = Σ_k x[m][k] · w[k][n]        (w bf16, no scales, no split-K)
```

**Work assignment.** The same skeleton as `gemv`: lane per `n`, SIMD16, fp32 accumulator, one store per lane.
Two compile-time knobs shape the grid, `COLS_PER_WG` (output columns a work-group owns) and `KSPLIT` (how many
sub-groups split one 16-column tile's K, each into its own accumulator, merged through SLM). The pair is a
function of `N`: `kernels::gemv_bf16_tiling(N)` gives `{64, 1}` when the default grid already puts a work-group
on every Xe-core and `{16, 16}` otherwise.

**`lm_head` (N = 248320) needs neither knob**: 3880 work-groups, 584 GB/s in the probe (97% of the roofline
denominator, 99% of what the bandwidth probe measures through the same launch path). Nothing to recover.

**a‖b is the shape that needed both, and the measurement is the lesson of this section.** At `{64, 1}` the
whole launch is 8 sub-groups reading 1.31 MB at 26.9 GB/s. Three things that did *not* help, then the one that did
(in situ, 2026-08-25):

| change | sub-groups | µs/launch |
|---|---|---|
| `{64, 1}` as first shipped | 8 | 48.774 |
| `COLS_PER_WG` 16 (2 work-groups to 8), bit-identical | 8 | 49.127 |
| plus 4 block reads in flight per sub-group, bit-identical | 8 | 49.052 (and `lm_head` 1.3% slower) |
| `{16, 4}`, K split four ways | 32 | 13.115 |
| **`{16, 16}`, K split sixteen ways (shipped)** | **128** | **5.340** |

A sub-group pulls 3.4 / 3.1 / 1.9 GB/s at 8 / 32 / 128 of them: **the launch was slow because it only ever had
eight sub-groups.** The unit of parallelism on this device is the sub-group, not the work-group. Two kernels
with the same sub-group count run the same however the work-groups are arranged. `COLS_PER_WG` 16 stays in
the shipped pair for the other half of the job: it spreads the 128 sub-groups over eight Xe-cores instead of
packing them on one. The a‖b launch went from 2.341 to 0.256 ms/token. At 245 GB/s it is 2.4× the time its
1.31 MB is worth at full bandwidth, so at most about 0.15 ms/token is left in it.

**The split's merge order is part of the kernel's contract.** Sub-group `q` of a tile holds
`k ∈ [q·K/KSPLIT, (q+1)·K/KSPLIT)` and the work-group collapses the slices with `prep.cl`'s pairwise tree
(`for stride = KSPLIT/2 … 1: if q < stride: red[q] += red[q + stride]`, a barrier after each step; strides
8, 4, 2, 1 at KSPLIT 16). No atomic, no data-dependent branch. This is not the unsplit kernel's single
ascending chain; the golden gate arbitrated it (element-exact), and the split build is *closer* to a
double-precision reference than the unsplit one (4.77e-07 against 2.38e-06 on the test's inputs), which is
what pairwise summation does. **No test pins the merge order and none can cheaply**: the test bar is the
reference tolerance (3.11e-04 against an observed 2.15e-06, about 150× of slack). What holds the order is the
kernel text and the golden gate.

Restructuring the K loop for `KSPLIT` also changed IGC's output at `KSPLIT == 1`, so `lm_head`'s binary is not
byte-identical to before even though its defines are. The control that its speed did not move
(4378.555 to 4379.089 µs) is secondary evidence; the primary evidence is the gate. This project's build flags
pin correctly rounded `/` and `sqrt` but say nothing about `FP_CONTRACT`, so "same defines imply same
numerics" is not airtight for any kernel here, which is why the gate and not a hash is the arbiter.

**Layout.** One only: the canonical bf16 tile `[n_tile][k_octet][8 k][16 n]` ushort, so one
`intel_sub_group_block_read_us8` streams 256 contiguous bytes. There is no layout-0 equivalent: reading the
checkpoint's row-major `[N][K]` lane-per-`n` would give each lane a 10 KB stride on the largest tensor in the
model, so it was not built to lose.

**Rejected.** Split-K for `lm_head` (argued away by the 97%; a future small-`N` bf16 matrix on the decode path
needs re-measuring). A separate small-`N` kernel for a‖b: what it needed was two `#define`s here.

---

## `gemm_sm` and `gemm_sm_i8w` - the small-batch GEMMs (spec 28b)

A speculative-decoding verify list checks M = 2..8 guessed rows at once (the DFlash drafter's block adds
M = 7, 8 and 16). `gemv` is a matrix-vector kernel: at M rows it re-reads and re-dequantises each weight tile
once per row, so verifying 8 rows cost 3.17 plain steps. `gemm_sm` (`src/kernels/gemm_sm.cl`) reads each
int4 g64 weight tile **once**, dequantises it once and multiplies it against all M rows on the DPAS, with the
weights on the DPAS B operand, in bf16. It reads exactly the bytes `gemv` reads and writes exactly the output
`gemv` writes, with the same kernel arguments, so a capture site swaps only the variant name, work-group and
grid. `gemm_sm_i8w` is the same for the int8 `lm_head` (and the DFlash drafter's int8 linears).

**Dispatch.** `runtime::gemv_kind(M)` (`src/runtime/gemv_kind.h`): M = 1 binds `gemv` / `gemv_i8w`, **bit for
bit what it was**, so plain decode did not change; M >= 2 binds `gemm_sm` for the shapes that have a row in
`kernels::kSmallMTuning` and `gemv` for the rest. There is no user flag (`CaptureOptions::force_linear` is for
tests and probes). The consequence is the one the golden gate documents ([14-golden-gate.md](14-golden-gate.md)):
verify rows are no longer bitwise a plain step's rows, and speculative output is judged against an f32
reference.

**Tuning.** One row per weight shape in `kSmallMTuning` (the table has no M column on purpose, see below):
the split-K slices `S`, and three speed knobs: `wg` (the in-work-group split-K: 2 column sub-groups × 8
K-slice sub-groups reduced through SLM, writing one slice), `magic` (a magic-exponent dequant: the nibble is
OR'd into the mantissa of bf16 `0x4300` = 128 + q and 136 subtracted in fp32; the same exact integer as the
shift form) and `blk2d` (the 2D block weight read). All are speed only: the output bits do not change.

**Property R, held by `tests/kernels/gemm_sm_test.cc` at every built M:** a DPAS output row depends only on
its own A row. The per-group order (4 × k16 DPAS from a zero accumulator, then `acc += group × scale` in
fp32), the row block of 8, `S` and the work-group shape do not depend on M, so **row m's output is bitwise the
same at every M >= 2 and whatever the other rows hold**. That is why the tuning table has no M column.

**Measured** (one card, int8 head, depth 4096, busy box, 2026-11-01): the verify step costs 1.04 / 1.09 / 1.14 /
1.18 / 1.30 / 1.36 / 1.57 plain steps at M = 2..8, against `gemv`'s 1.16 up to 3.17
([BENCHMARKS.md](BENCHMARKS.md), "The verify step on gemm_sm"; the record is
[probe-small-m-gemm-2026-11-01.md](probes/probe-small-m-gemm-2026-11-01.md)). The numbers were taken on a
busy box and are to be re-measured on a quiet one (open-work.md, the quiet-box session).

---

## `prep` - the between-GEMV kernels

`src/kernels/prep.cl`, one file with several entry points. Everything between two GEMVs of a decode step lives
here: sum the previous GEMV's split-K partials, add the residual, normalise, activate. The bound set is
`prep_res_fold` + `prep_norm_finish` (the residual add and RMSNorm as two launches), `prep_silu_mul` and
`prep_gated_head`. `prep_res_norm` (the single-launch predecessor of the pair) is still compiled and still run
by `prep_test`, as the pair's reference.

### Rounding discipline - the substance of these kernels

The oracle is torch, and **torch rounds per op**: a linear's output is bf16, and every elementwise op widens
to fp32 internally and rounds its result back to bf16. These kernels match that discipline wherever it is
cheap, so the engine's residual stream stays comparable to the golden tensors op for op:

- a GEMV's split-K partials are summed in fp32 and rounded to bf16 **once**, because that bf16 value is the
  linear's output in the reference;
- the residual add is `rne_bf16(f32(resid) + f32(mixer_b))` (`rne` = round to nearest even), exactly torch's
  bf16 add;
- a norm widens the bf16 input to fp32, multiplies by the fp32 `(1 + w)` weight (the loader's bake,
  [13-loader.md](13-loader.md)) and rounds to bf16, the reference's `type_as(x)`;
- **inside-op** accumulation (the variance sum) stays fp32 and is *not* matched to torch term for term. That
  residual drift is what the golden gate absorbs: tokens exact is the gate, tensor cosines are diagnostics.

The payoff is that `tests/kernels/prep_ref.h` is the same op chain in the same order, so `prep_test` asserts
**bit-exact** equality rather than a tolerance. Two spellings are pinned to get there:

1. **`1.0f / sqrt(x)`, never `rsqrt(x)`.** `rsqrt` is a roughly 2 ulp approximation with no cross-implementation
   guarantee. Correctly rounded `sqrt` and divide are reproducible on the host. The rule binds the GDN and
   attention kernels too.
2. **The square-accumulate is an explicit `fma`**, so no two compilers' `-ffp-contract` defaults have to agree.

…and one **build option**: **`-cl-fp32-correctly-rounded-divide-sqrt` is required on any kernel compared
bit-exactly that divides or takes a square root.** OpenCL's default allows 2.5 ulp on both, and this is not
theoretical: without the flag the device's `1/sqrt(mean + 1e-6)` came out 1 to 2 ulp below the host's and one
`x_out` element in 5120 fell on the other side of a round-to-nearest tie (5119/5120 exact, exactly the
"almost" a tolerance would have hidden). It is now the project-wide default (`cmake/ocloc.cmake`), because "the
next kernel with a divide must remember the flag" is not a rule anyone can be relied on to follow.
`-cl-denorms-are-zero` is forbidden project-wide.

**The variance tree order** is stated identically in `prep.cl` and `prep_ref.h`. Each of the work-group's WG
lanes accumulates its own strided slice (lane `i` takes `k = i, i+WG, …` ascending) into one fp32 register with
`fma`, writes it to SLM, and a fixed pairwise tree collapses SLM: for `stride = WG/2 … 1`, `red[i] += red[i + stride]`
with a barrier after every step. No data-dependent branch and no atomic, so a replayed list gives the same bits.

### `prep_res_fold` + `prep_norm_finish` - residual add and RMSNorm, two launches

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

**Why two launches.** The single-launch form (`prep_res_norm`) ran RMSNorm's whole-row reduction in **one
work-group**, which is one Xe-core (16 sub-groups). Measured in situ it took **22.3 µs and 17.0 GB/s**, 129
times a token, because one core's issue rate, not DRAM, was the limit (the 320 KB of partials it folds were
just written by the GEMV before it, so most of that traffic was L2). Splitting it onto 20 work-groups each
(`G = W = 20`; 320 sub-groups instead of 16) cut the site from **2893 to 484 µs/token** (stage A 2.0 µs,
stage B 1.7 µs a launch, in situ 2026-08-25). A third run with stage B on one work-group (`W = 1`) saved only
1.45 of the 2.41 ms: the rescale pass is the same latency-bound walk and needed the same treatment.

`G = 20` at K = 5120 is one element per lane. Both counts are `runtime::DecodeBuffers::kNormGroups` and are in
**both variant names**, so a host that pairs mismatched grids names a binary that does not exist and throws at
capture. Stage A has no SLM row (the value a lane squares is the one it just wrote), so SLM falls from 21 KB
to 1 KB; stage B re-widens `resid` from memory (SLM does not survive a launch), and those bytes are L2-warm.

**Numerics contract.** Stage A's per-element chain is the single-work-group kernel's verbatim, so **`resid`
comes out bit-identical** (`prep_test` asserts it). The one thing that changes is the global Σx² tree (G chunk
trees, then a fixed ascending-`g` fold), which can move `x` in the last bf16 ulp; the golden gate arbitrates.
Determinism is unaffected (G fixed at compile time, no atomic).

Variants: `S_PREV` (the previous GEMV's split-K width, so the fold reads the right number of slices; layer 0's
first norm is `S_PREV = 0`), `X_STRIDE` and `ZERO_RESID` for the MTP head, and `M`.

### `prep_silu_mul` - the MLP activation

```
gflat = (k/16)·32 + k%16 ;  uflat = gflat + 16
g_b = rne_bf16(Σ_s partials[s][m][gflat]) ;  u_b likewise
s_b = rne_bf16(silu_f32(f32(g_b)))
x_out[m][k] = rne_bf16(f32(s_b) · f32(u_b))
```

with `silu(x) = x / (1 + exp(-x))`, plain `exp`, never `native_exp`. The index arithmetic is the loader's
`cols_interleave16`: `gate‖up` is one fused linear whose columns interleave in 16-wide blocks, so the lane
holding `gate[n]` also holds `up[n]` at a fixed offset; this kernel is the consumer that pays for that layout.
**Grid `(5, M)`, WG 256, no reduction, no SLM, no barrier**: five chunks of 4096 cover 17408 (the last covers
1024). `SILU_S` (8) is a copy of the gate‖up split-K, and `Capture::check_sizes` asserts it, so a retune of the
table throws at capture instead of summing the wrong number of slices.

### `prep_gated_head` - `Qwen3_5RMSNormGated`, one v-head per work-group

```
o_b = rne_bf16(gdn_o[m][h][i])                    (recurrence output -> bf16)
z_b = rne_bf16(Σ_s qkvz_partials[s][m][10240 + h·128 + i])
var = mean_i(f32(o_b)²)                           (128-lane tree)
n_b = rne_bf16(f32(o_b) · (1 / sqrt(var + 1e-6)))
t_b = rne_bf16(f32(gated_w[i]) · f32(n_b))
x_out[m][h·128+i] = rne_bf16(f32(t_b) · silu_f32(f32(z_b)))
```

This is the model's op chain exactly ([03-models.md](03-models.md)). `gated_w` is the bf16 plain `w` with **no
`+1`**, the one norm in the model without the increment. **Grid `(48, M)`, WG 128, one lane per channel**: the
head dimension is the reduction domain, so the variance tree is exactly the work-group's 128 lanes with one
term each (hence no `fma`). The silu factor is deliberately the last op: everything through `t_b` is a rounded
scalar chain the host reproduces exactly, and only the final product carries `exp`'s slack.

### What the test asserts

`tests/kernels/prep_test.cc` against `prep_ref.h`, on random inputs:

- the norm kernels (`prep_res_norm` at both prep modes, and the two-stage pair against both the single-stage
  and the two-stage reference, including `W = 1`): `resid`, `sumsq` and `x_out` **bit-exact**;
- `prep_silu_mul` and `prep_gated_head`: **≤ 2 bf16 ulp**; and with the silu argument forced to 30.0,
  `exp(-30)` is below `2^-24`, so `1 + exp(-30)` is exactly `1.0f` on any conforming implementation and the
  result is asserted **bit-exact**. That pins `s_b` and the gated head's `n_b`/`t_b` chain, the intermediates
  that otherwise never leave the kernel.

**The ≤ 2 ulp bar is a contract, not an observation.** OpenCL allows 3 ulp on `exp` where the host's `expf` is
about 0.5; today's driver matches the host on every value tested. The bar stays so a driver update may change
`exp` without failing the test, while a change in the arithmetic still will.

**A known weakness in the `resid` bar, found by mutating the kernel.** `resid` bit-identity is a necessary but
**weak** check of stage A's `s = 0 … S_PREV-1` fold order: reversing that loop did not fail `prep_test`,
because the fp32 sum is rounded to bf16 immediately and an fp32 last-ulp reordering has to land on a bf16
rounding boundary to survive (estimated: about 0.3 expected crossings over 5120 elements, usually none).
`prep_res_norm` is still run against a reference with the same loop text, and the golden gate walks 64 layers
of real rows, which pin the order; an edit to stage A's `s` loop alone is not covered. Closing it would mean
grading the fp32 sum before it is rounded.

### Rejected

- **Fusing the norm into the following GEMV's prologue.** It would remove about 128 launches a token, but the
  per-kernel dispatch cost inside a replayed list is 0.73 µs (in situ), so the 241 between-GEMV launches cost
  about 0.18 ms (0.4% of the step then), and the traffic these kernels add is about 0.6% of the token's weight
  bytes. Fusion also has a real cost: the reduction is over the whole row, so every GEMV work-group would
  either redundantly reduce it or need a cross-work-group barrier the split-K design deliberately lacks.
- **Atomics in the GEMV to sum the partials**: rejected on determinism, as for `gemv`.
- **Sub-group reductions (`sub_group_reduce_add`) instead of the SLM tree**: not used. The SLM tree's order is
  *stateable* and the whole bit-exactness contract rests on the reference reproducing it. Probably faster on a
  kernel whose reduction is not the bottleneck; not measured.
- **`native_exp` / `native_rsqrt`**: rejected outright, they put the residual stream somewhere the host cannot follow.

---

## `embed_gather` - the token id to the residual stream

`src/kernels/embed_gather.cl`, variants `embed_gather_M<M>` (M up to 8, the size of `Control::cur_token`).
The first kernel of a decode step:

```
row            = ctrl[CTRL_CUR + m]        (runtime::Control::cur_token[m])
resid[m][0..K) = embed[row][0..K)          K = 5120, bf16, copied verbatim
```

No arithmetic and no rounding: the embedding table is bf16, the residual stream is bf16, and the reference op
is a gather.

### Why it reads the control block instead of taking the id as an argument

This is the kernel that justifies the whole capture-once, replay-per-token design. Level Zero resolves a
kernel's arguments at `zeCommandListAppendLaunchKernel` time, **not** at execute time
(`tests/l0/arg_capture_test.cc` proves it by binding two buffers between two appends of the same kernel). An
id passed as an argument would be frozen into the list when it was recorded, and every replay would gather the
same row forever. So the id lives in the `zeMemAllocShared` control block, whose *contents* are read when the
kernel runs. `embed_gather` is the only reader of `cur_token` inside the list and `argmax_stage2` its only
writer; the host writes it only during prompt ingestion. `tests/kernels/embed_gather_test.cc` asserts the
property: one closed list, executed twice with nothing rebound and only `cur_token[0]` changed, must produce
two different rows.

**Work assignment.** Grid `(1, M)`, one work-group of 256 per token: 5120/256 = 20 elements per work-item with
lane `i` taking `k = i, i+256, …`, so each iteration is one coalesced 512-byte read and write per sub-group.
There is nothing to reduce, and a wider grid would only add launch cost to a kernel that moves 20 KB.

**Out-of-range convention.** `row >= VOCAB` is **not clamped**. An id outside the table cannot come out of
this engine (argmax masks at the used vocabulary), so it means the control block is wrong. Clamping would hide
that behind a plausible-looking continuation. Instead the kernel writes nothing, leaves the residual stream as
it was, and sets `Control::debug_flag = 0xDEAD0001`. The test covers both `kVocab` and `0xFFFFFFFF` and then
replays a valid id to show the list still works.

**Rejected.** `zeCommandListAppendMemoryCopy` (its source address is captured at append time like a kernel
argument, so it cannot express this); re-recording the list per token (defeats the capture); a wider grid.

**Measured** (2026-08-25): **3.65 µs per token**, median of 5 replays of a closed list of 512 back-to-back
launches, printed by `embed_gather_test`. At 64 launches per execution the same kernel reads 6.59 µs, so short
bursts are dearer (cause not established; clock ramp is a guess), and the 512-launch figure is the marginal
cost on a device already working. About 0.01% of the step.

Other models bake a different hidden size: the variant gets a `_D<hidden>` suffix (`kernels::hidden_suffix`);
Qwen3.8's 5120 carries none.

---

## `argmax` - deterministic greedy sampling, in two fixed stages

`src/kernels/argmax.cl`, entry points `argmax_stage1` (variants `argmax_stage1_M<M>`) and `argmax_stage2`.
Together they turn one row of `lm_head` logits into the next token and move the control block on by one step:

```
stage 1  grid (243, M), WG 256 - group g reduces logits[m][g·1024 … +1024)
                                 to one (value, index) pair in part[m][g]
stage 2  grid (1, 1),   WG 256 - folds each active token's 243 pairs, writes
                                 ctrl.out_token[m]; then lane 0 sets
                                 ctrl.cur_token[0] = out_token[n_active-1]
                                 and ctrl.pos += n_active
```

`argmax_stage2` is the **only writer of `pos` and `cur_token` inside the list**. That is what makes the list
self-advancing: the host writes the control block once at prompt ingestion and afterwards only reads
`out_token[0]` after each fence.

**Why two fixed stages and not atomics.** A global atomic max over 248320 candidates (OpenCL has no float
atomic max, so a compare-exchange loop) is **order-dependent by construction**: which of two equal logits
counts as "already the max" depends on which work-group arrives first. Equal logits are not exotic at the top of
a peaked distribution, and two spellings of the same word is exactly where greedy decoding is fragile, so an
atomic argmax can emit a different token on two replays. The golden gate's bar is token-exact output, so that
disqualifies it before any performance argument. A two-stage tree has no data-dependent order at all. (The
atomic version was never built; determinism settles it before speed is asked.)

**The comparator and the tie rule.** One function, `argmax_better`, used by both stages and the test's host
reference: `(a.v > b.v) || (a.v == b.v && a.i < b.i)`, strictly greater wins and an exact tie takes the **lower
index**. Because of that the *bracketing* of the reduction cannot change the answer, so stage 1 and stage 2 may
use different tree widths and still agree with a sequential scan. It is also the oracle's rule
(`torch.argmax` returns the first maximum). Idle lanes seed `(-INFINITY, 0x7FFFFFFF)`, which loses to every
real logit and every tie.

**The 248077 mask.** `lm_head` is `[5120][248320]` because the tiling wants a round row count, but the
tokenizer defines only **248077** ids (`Qwen35::kVocabUsed`, the checkpoint's `vocab_size`). The logits of rows
248077..248319 are ordinary finite numbers that can be the row maximum; emitting one would be an id the
tokenizer cannot decode. So a candidate at index `>= VOCAB_USED` contributes `(-INFINITY, idx)` and never wins
(baked as `-D VOCAB_USED=248077`). `argmax_test` plants the row's two largest logits inside the masked tail.

**Work assignment.** Stage 1: chunk 1024 (`DecodeBuffers::kArgmaxChunk`), 243 = ⌈248320/1024⌉ work-groups, about
7.6 per Xe-core, enough to fill the device without making stage 2's fold wide enough to need a third stage.
Stage 2: one work-group with `m` looped inside (it owns the control block's per-step bookkeeping, which exactly
one lane may do); the loop bound `n_active` is read uniformly so the barriers are reached by all lanes, and
`n_active == 0` writes nothing rather than something plausible. The index travels as a float, which is lossless
here: every integer below 2^24 is exact in fp32 and the largest index is 248319.

The other argmaxes share this design with different widths: `tp_argmax_stage1/2` (`tp.cl`, one card's half of
the vocabulary under tensor parallel; the host merges the two winners, larger value first, ties to the lower
id) and `dv_argmax_stage1/2` (`draft_vocab.cl`, the MTP draft's reduced vocabulary).

**Test** (`tests/kernels/argmax_test.cc`): a unique maximum, an exact three-way tie across two stage-1 groups, the
two largest logits in the masked tail, all logits equal, the last usable index, and a random row against the
host reference, through one closed list replayed once per case; after every replay `out_token[0]`, `cur_token[0]`
and `pos` advanced by exactly `n_active`.

**Rejected.** A global atomic max (determinism); one work-group scanning all 248320 logits (993 KB of
streaming on one Xe-core of 32); `work_group_reduce_max` (cannot carry the index, and the order is not
stateable); packing (value, index) into a `ulong` (unneeded). **Sampling** (temperature, top-p, top-k) is not
this kernel; host-side sampling costs about 0.54 ms/token measured ([BENCHMARKS.md](BENCHMARKS.md), "Host
sampling") and greedy is unaffected.

**Measured** (2026-08-25): **4.39 µs per token for both stages**, same method and same short-burst caveat as
`embed_gather`. Together the two token-boundary kernels are the only two numbers here small enough to stop
thinking about.

---

## `gdn_step` - the gated delta-rule decode step

`src/kernels/gdn_step.cl`, variants `gdn_step_M<M>` and, for speculative decoding, `gdn_step_slots_M<M>`
(below). **48 of the model's 64 layers run this.** No library implements it and `sycl-tla` has no example for
it: it is the engine's original-work kernel, and the one that decides whether the golden gate can pass.

One kernel does the whole GDN mixer between the qkv‖z GEMV and `prep_gated_head`: the depthwise conv1d update
and its SiLU, the q/k l2norm, and the recurrent state update.

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

That is the model's GDN block op for op ([03-models.md](03-models.md)); `tests/kernels/gdn_ref.h` is the same
chain on the host, trees included.

### Work assignment, and why

**Grid `(48 heads, 4 chunks)`, work-group 256 = 16 sub-groups × 16 lanes.** Work-group `(h, c)` owns state
columns `[32c, 32c+32)` of head `h`. Work-item `(sgid, lane)` owns an **8 k × 2 v tile** (k-rows
`8·sgid … 8·sgid+7`, columns `32c+lane` and `32c+lane+16`): 16 fp32 of state per work-item, **in registers**.

- *Why (head, chunk) and not one work-group per head.* 48 work-groups on 32 Xe-cores is 1.5 per core: half
  the machine idle on the layer type that runs 48 times a token, with no second work-group to hide the state
  load's latency. Four chunks give **192 work-groups**, six per core, at the cost of convolving each head's
  channels four times. The state slice per work-group is 128 × 32 × 4 B = 16 KB: registers, not SLM.
- *Why the lane picks the column and the sub-group picks the k-band.* `state` is k-major, so a row's 128 `v`
  are contiguous. Under this mapping a sub-group's 16 lanes read 16 consecutive fp32, one full 64 B cache
  line, per access. The obvious alternative (lane picks the k-row) makes every access a 16-way gather of 8 B
  at a 4 KB stride: same arithmetic, an order of magnitude apart on the load that dominates the traffic.
- *Why 8×2.* Columns are independent under the rank-1 update, so the update needs no communication at all (no
  atomics, no barrier); only the two contractions cross work-items, and for a fixed column they cross exactly
  the 16 sub-groups, a 16-wide SLM tree. Two columns per lane amortise that tree over two results.

**The trees, stated once and obeyed twice** (`gdn_step.cl` and `gdn_ref.h` carry this text verbatim):

- `kv[v] = Σ_k S[k][v]·kf[k]`: band `sgid` sums its 8 rows in **ascending kk** with explicit `fma` into
  `kv_red[sgid][v−32c]`; the 16 band partials collapse pairwise (`stride = 8, 4, 2, 1`, barrier after each step).
- `o[v] = Σ_k qf[k]·S[k][v]`: the same shape in `o_red`.
- the two l2norm sums: lane `lid < 128` contributes `f32(q_b[lid])²` (one term per lane, plain multiply) and lane
  `128+i` the k term; each 128-wide array collapses with `stride = 64 … 1`.

A tree's order is a property of the **bands**, not of `c` or `lane`, so the host reference can walk whole
128-column heads and land on the same bits, and no atomic is needed: two replays give identical bytes.

**SLM: 6912 B** (conv outputs, l2norm sums, normalised q/k, two reduction arrays laid out `[16 bands][32
columns]` so writes and every tree step are lane-contiguous). The compiler's own report: 128 GRF, SIMD16, one
barrier, and neither a `private_size` nor a `spill_mem_size` entry, so the state stays in registers, which was
the design's one real risk.

### The conv in the prologue, and the four-fold redundancy it costs

Each work-group convolves 384 channels: the 128 q and 128 k of k-head `h/3` (the model's
`repeat_interleave(·, 3)`) and the 128 v of head `h`. Every one of a head's four chunks computes all 384, so
each channel is convolved 4 times and each q/k channel 12 times. That is deliberate: a separate conv kernel
would cost a launch per GDN layer and a DRAM round trip of the convolved qkv, and would not even avoid the
redundant *reads* (every work-group still needs the full 128-wide q and k for the l2norm). The redundant reads
are about 1.9 MB a layer against the state's 6.3 MB, L2-resident. **The measurement says the redundancy is not
a price at all** (below).

### Ring ownership - the argument, written out

The conv state is a **ring of 16 slots × 10240 channels of bf16** per layer (`DecodeBuffers::conv_ring`),
holding raw qkv values (the reference's `conv_states` likewise hold the *input* sequence). Up to 12 work-groups
compute the same raw value, so exactly one stores it: `c == 0` writes this head's **v** channels, and
`c == 0 && h % 3 == 0` also writes the **q** and **k** channels of k-head `h/3`. Together those cover all 10240
channels once. *Which* work-group owns a channel is a bandwidth decision, not a correctness one.

The correctness argument is about **slots**. A work-group writes slots `(pos+m) % 16` for `m < n_active` and
reads `(pos−1)`, `(pos−2)`, `(pos−3)` `% 16`. With ring depth **16 ≥ M + 3** the sets cannot intersect, so **no
work-group ever reads a slot any work-group is writing this step**, and that, not a barrier, is what makes 192
independent work-groups safe (Level Zero gives no cross-work-group barrier inside one launch anyway). Within a
work-group each item reads its channel's history before the loop that writes. `gdn_step_test` pins it without
the reference: at `pos = 5` it blanks slots 2 to 4 and requires `gdn_o` to move, then blanks the other thirteen
slots and requires `gdn_o` **bitwise unchanged**.

### Speculative-decoding variants

`gdn_step_slots_M<M>` (built with `SPEC_SLOTS`) additionally writes the state after row `m` into slot
`(live + m) % N_SLOTS` of a separate spec buffer, so a verify step that accepts fewer rows than it checked can
keep the right state (4 slots for MTP, `_N8` builds with 8 for DFlash). The plain variants write the state in
place. `gdn_step_slots_test` covers them.

### What the test asserts

`tests/kernels/gdn_step_test.cc` against `gdn_ref.h`, on seeded random inputs. The reference reproduces every
rounding and every tree, so what is left between host and device is only the libm functions OpenCL does not
require to be correctly rounded: `exp` (3 ulp) and `log1p` (2 ulp). Bars: `state` relative error ≤ 1e-5
(observed about 3.3e-07 to 4.6e-07), `gdn_o` ≤ 1e-3 (about 4e-07 to 5.6e-07), `conv_ring` **bit-exact**, and
every case replayed **bitwise** identical. The cases are `pos = 0` (history below zero), `pos = 5`, `pos = 1`
(partial clamp), and `pos = 15` with M = 2 (the write wraps to slots 15 and 0). The M = 2 variants exist
because nothing else covers the conv window's intra-step path: at `m = 1` two of the window's four entries are
this step's own raw values. "Relative error" is floored at the tensor's own RMS, because `Δ = (v − kv)·β` is a
difference of same-sized numbers and an individual state cell can cancel to near zero. The margins are about 20×
on `state` and 1800× on `gdn_o`; the bars stay where they are because they are the bars a driver change must
not break.

### Rejected

- **SIMD32**, the shape the SYCL reference uses (8 sub-groups of 32, 4 k-rows × 4 columns a lane): not
  measured. Its contraction would reduce inside one sub-group, faster but giving up the stateable reduction
  order the bit-comparable reference rests on.
- **State in SLM instead of registers**: 16 KB would fit, but every recurrence access then goes through SLM and
  occupancy caps at four work-groups per core. The register form compiles with no spill. Not measured.
- **`sub_group_reduce_add`**, **`native_exp`**, **atomics** for the contractions: as for `prep`.
- **A separate conv kernel** and **one work-group per head (48)**: argued above.

### Measured - in situ (2026-08-25): 0.733 ms/token, 1.09× its own floor

48 launches, **15.276 µs** each, **0.733 ms/token** (1.7% of the step then). The traffic is the state, read and
written once: 48 × 128 × 128 × 4 B = 3.146 MB per layer, about 396 MB a token with the redundant conv reads.
That is **540 GB/s effective, 92% of the measured 590**, and 1.09× the floor of 0.671 ms. The kernel is at its
lower bound; the whole kernel made perfect would be worth about 0.06 ms. The occupancy problem it was first
suspected of was in the one-work-group residual norm and a‖b, not here.

---

## Decode attention

Three kernels per full-attention layer (16 of the 64 layers), in order: `attn_prep`, `attn_decode_v2`,
`attn_reduce_v2`. `attn_prep` is in `src/kernels/attn.cl`; the pair is in `src/kernels/attn_v2.cl` (spec 10).
The first-generation pair in `attn.cl` (`attn_decode` + `attn_reduce`, "v1") is now **test-only**: it is the
reference `attn_v2_test` grades v2 against and `attn_test` grades against its host twin. The int8 KV cache has
twins of all three in `src/kernels/kv8.cl` (see "The int8 KV cache" below).

```
attn_prep(ctrl, qkv_partials, fa_small, rope, attn_q, attn_gate, kv_k, kv_v)

x_b    = rne_bf16(qkv_partials[0][m][col])                  (the linear's bf16 out)
rstd   = 1 / sqrt(mean_i(f32(x_b)²) + 1e-6)                 (fp32; never rsqrt)
nrm[i] = f32(rne_bf16(f32(x_b[i]) · rstd · w[i]))           (fp32 (1+w) weight)
RoPE over dims 0..63, pairs (i, i+32); dims 64..255 pass through
  q-head  h: attn_q[m][h][i] = roped ;  attn_gate[m][h][i] = f32(rne_bf16(gate col))
  kv-head j: kv_k[pos+m][j][i] = rne_bf16(roped)
             kv_v[pos+m][j][i] = rne_bf16(v col)            (never normed or roped)

attn_decode_v2(ctrl, attn_q, kv_k, kv_v, attn_part)         grid (4 kv-heads, 32), WG 256

  per (kv-head j, block), per token m, per q-head of j's six:
    score_p = (Σ_d attn_q[m][qh][d] · f32(kv_k[p][j][d])) / 16   for p <= pos+m
    online softmax in waves of 16 positions -> (mx, sm) and acc[256]
    attn_part[qh][block][m] = {mx, sm, acc[256]}               (258 fp32)

attn_reduce_v2(ctrl, attn_part, attn_gate, attn_out)         grid (24, M), WG 256

  merge the row's blocks in ascending order ; out = acc/sm
  attn_out[m][h·256+d] = rne_bf16(f32(rne_bf16(out)) · sigmoid_f32(attn_gate[m][h][d]))
```

That is the model's full-attention block op for op; `tests/kernels/attn_ref.h` is the same chain on the host,
trees, waves and merge order included.

**The column map** is easy to get wrong and silent when wrong. `q_proj ‖ k_proj ‖ v_proj` is 12288 + 1024 +
1024 = 14336 columns, and inside `q_proj` the 24 heads are **interleaved per head, not two halves**: head `h`
is `[h·512, h·512+256)` and its output gate is the next 256. k-head `j` is at `12288 + j·256`, v-head `j` at
`13312 + j·256`. GQA is 6:1, so q-head `h` reads kv-head `h/6`. The KV caches are `[pos][4][256]` bf16.

### `attn_prep`: grid (28, M), work-group 256

A head is exactly 256 wide and so is the work-group: work-item `i` owns dim `i` and the RMSNorm's reduction
domain *is* the work-group. Work-groups 0..23 are q-heads (each also writing its gate), 24..27 are the four
kv-heads. 28 and not 32 because k and v of a kv-head share a work-group (v is never normed or roped, one extra
`rne_bf16` on a work-item already resident). The one piece of cross-lane traffic is RoPE, which pairs dim `i`
with `i ± 32`, which is why the normalised head goes through SLM (1 KB).

### `attn_decode_v2`: many positions per work-group, a stride that depends only on the row

Decode attention has to read the whole KV cache for every layer, and the cache grows with the context, so the
kernel has to stay fast from 4k to 128k+. Two decisions define v2:

1. **The stride is per row, and a function of that row's key count alone.** Row `m` attends to
   `L_m = pos + m + 1` positions and walks them in blocks of `ppw(L) = max(64, roundup64(ceil(L / 32)))`
   positions per work-group, so at most 32 blocks for any `L`. `ppw` moves only every 2048 keys, so the rows of
   one launch have at most two different strides, and each row therefore sees exactly the blocks, waves and orders of the one-row step at that
   position, and **a verify row is bitwise the plain step at the same position** (`attn_v2_test`). The stride
   is recomputed from `pos` every step: no control word, so a snapshot copied back from the prefix cache or a rewound `pos` needs
   nothing.
2. **The grid is (4 kv-heads, 32) at every `max_len`** and the partial table is `[24][32][M][258]`. No row
   needs more than 32 work-groups, so one binary per `M` (`attn_v2_M<M>_T32`) serves every context length and
   there is no idle grid (the first-generation pair had to launch for `max_len` and let most work-groups exit
   early).

**Inside a work-group.** 16 sub-groups, sub-group `s` owns position `p0 + s` of a 16-position wave; K is read
by two sub-group block reads per row (lane `l`, element `t` = dim `l + 16t`), V by one 2D block read per
sub-group per wave. **One K/V wave feeds all 6 × M (q-head, row) pairs** ("register-packed GQA"): the K and V
values are loaded into private registers once and reused by the other five heads, instead of being fetched
six times. The per-head arithmetic order does not change, only the order in which independent heads consume an
already-loaded wave. One barrier per wave publishes the raw dots.

In the first-generation kernel this one change (six passes over the same K/V to one) measured **228.0 to
160.8 µs per launch**, −1.08 ms/token, in production, and it matched the controlled probe to 0.28% of a
launch. The matching drift on the untouched sibling kernels (2 to 4%) is why a before/after on a kernel you did
not change is a noise control, not a saving. One small lesson from the same change: deleting a *redundant*
barrier that sat two instructions before a surviving one saved only 39% of what pricing it at the average
barrier cost predicted. **An ablation that removes all the barriers prices the average barrier, the wrong
number for removing one of seven.**

**v1 against v2, measured** (`probe_decode_attn ab`, one engine with both lists captured over the same buffers;
[BENCHMARKS.md](BENCHMARKS.md), "Decode attention at depth", 2026-09-28, bf16 head): whole decode step 33.72
vs 31.84 ms at depth 4096, and **98.36 vs 49.68 ms at depth 130816** (10.2 to 20.1 tokens/s). v2 matches v1
on real data to one bf16 ulp (cosine 0.999999491 at 65536 keys).

**`exp` and not `exp2`.** An `exp2` form with the scale folded in was faster in the probe and failed the
golden gate (a row flipped), so v2 uses `exp` and the 1/16 scale, as `attn.cl` does. The build option is gone.

### The wave: an online softmax with a stateable order

A block's positions are walked in waves of 16 (the work-group has 16 sub-groups). Everything below is stated
identically in the kernel and `attn_ref.h`, because the reference reproducing it bit for bit is what buys the
tight bars.

**The score dot.** Sub-group `s` owns the wave's position; lane `l` accumulates the 16 elements `d = l + 16t`,
`t = 0..15` **ascending** with an explicit `fma`, and the 16 lane partials collapse with a fixed pairwise tree
(`stride = 8, 4, 2, 1`); the sum times `1/16 = 1/√256` is the score. A masked position skips the dot and is
overwritten with −INF, so an unwritten cache slot never reaches the arithmetic.

**The update**, computed redundantly by all lanes from the same SLM words in the same order:

```
nmx   = max(mx, sc[0], …, sc[15])                 (ascending s)
resc  = exp(mx − nmx)                              (0 when mx = −INF)
sc[s] = exp(sc[s] − nmx) ;  ssum = Σ_s sc[s]       (ascending s)
sm    = fma(sm, resc, ssum) ;  mx = nmx
t     = Σ_s fma(sc[s], f32(kv_v[p_s][j][d]), t)    (ascending s, the lane owning dim d)
acc   = fma(acc, resc, t)
```

Two properties are load-bearing. **The −INF algebra works out on its own**: `exp(−INF − finite) = 0`, so a
masked position gets zero weight and a first wave with `mx = −INF` gets `resc = 0`, both without a branch. The
one case that does not work out is `nmx = −INF` (every position of the wave masked and no earlier valid one),
where `exp(−INF − −INF)` is a NaN; that wave is skipped whole. And **redundant is cheaper than elected**:
letting one sub-group own `(mx, sm)` and publish the weights costs an extra barrier per wave and idles 240
lanes through a serial loop, while recomputing the same scalars everywhere costs 16 SLM reads and ALU that is
free next to the KV loads, and is more obviously deterministic.

### `attn_reduce_v2`: the block merge

Merges in **ascending block order**: `nmx = max(mx, bmx); a = exp(mx − nmx); bs = exp(bmx − nmx);
sm = fma(sm, a, bsm·bs); acc = fma(acc, a, bacc·bs)`, the canonical flash-attention rescale, with `mx = −INF`
on entry so the first merge is a plain copy. `nmx` is never −INF inside the loop because a block read by row
`m` starts at or before `pos + m`, so it contains at least its first position. The block headers are staged in
SLM in one pass and every work-item runs the identical scalar merge alongside its own `acc` (identical inputs,
identical order, identical bits), cheaper than electing one item and broadcasting.

### The RoPE rounding - the one op this family does not match torch on

RoPE is applied to the **fp32 widened normalised value**: `attn_q` keeps the fp32 result and the k written to
the cache is `rne_bf16` of it, one rounding for the whole pair. torch reaches the same value through bf16
tensor ops and rounds once more *inside* the cos/sin multiply-add. This is a ≤ 1-op difference taken
deliberately, and the only place the family is not a per-op transcription of the oracle. The rotation itself is
one rounded product plus one `fma`, spelled identically on both sides (`out_i = fma(x_i, cos_i, −(x_{i+32}·sin_i))`,
`out_{i+32} = fma(x_{i+32}, cos_i, x_i·sin_i)`), with `cos/sin` from the loader's table computed in `double`
([13-loader.md](13-loader.md)). Dims 64..255 pass through (`partial_rotary_factor 0.25`).

### What the tests assert

`tests/kernels/attn_test.cc` (the first-generation pair and `attn_prep` against `attn_ref.h`) and
`tests/kernels/attn_v2_test.cc` (v2 against v1 and against the plain-step bitwise claim above), on seeded random
inputs with **the entire KV cache** filled with random bf16, not just the prefix, so a read past the causal
bound shows up as noise rather than a convenient zero. The one libm function left between host and device is
`exp` (3 ulp); `1.0f/sqrt` is not among them, so the norm, the RoPE and the KV cache are held **bit-exact** (and
the untouched cache slots prove `attn_prep` writes only `pos … pos+n_active−1`). `attn_out` carries two bars,
and the second is the arbiter:

- **(a)** relative error ≤ 8e-3 = 2^-7, floored at the tensor's RMS, because `acc/sm` is a weighted average of
  *signed* values and a dim can cancel to near zero;
- **(b)** ≤ 2 bf16 ulp on every element with `|ref| ≥ rms/8`. bf16 has 7 explicit mantissa bits, so one ulp is
  3.9e-3 to 7.8e-3 of the element, and 2 ulp is the arithmetic of the final chain (it rounds to bf16 twice, and
  3 ulp of `exp` slack can push a boundary value one ulp at each), not a fudge.

If a change ever trips (a), read (b) before believing the kernel broke: (a) at 1e-3 was inconsistent with (b)
from the start and only held while the device happened to reproduce the host bit for bit. The measured worst
gated distance is **1 ulp on one word** in each of two of 11 cases. M = 2 runs exist because they are the only
cover for per-row causal masking.

### Rejected

- **SLM-staged K/V tiles instead of private registers**: the staging medium cost 9.7 µs per launch (149.5 vs
  139.8) in the controlled probe.
- **One work-group walking the whole `max_len` per (q-head, token)**: 24 work-groups a layer on 32 Xe-cores,
  three quarters of the machine idle. Splitting into blocks costs the two-pass structure and one extra launch.
- **One work-group per (q-head, block)**: same bytes, six times the work-groups, and the six passes scattered
  over independent work-groups instead of back to back.
- **A two-pass softmax over the block**: simpler but does not generalise; the online form lets the block grow
  without the score array growing.
- **`sub_group_reduce_add` for the score dot**: not used (stateable order). A sub-group-scope barrier on three
  of the four tree steps changes no arithmetic and measured 3.9% faster, a ceiling of about 0.14 ms/token;
  recorded as measured rather than taken.
- **`attn_part` in bf16**: rounding the accumulator, not an op's output, which the rounding discipline forbids.
- **`native_exp`, atomics for the block merge**: as everywhere.

### Measured

These are the first-generation pair's numbers, kept because they explain the shape of the kernel family and the
method transfers. In situ, depth 4096, 2026-08-25:

- With 256 positions per work-group (the first choice) `attn_decode` was **361 to 370 µs a launch, 5.8 to 5.9
  ms/token**, the largest non-GEMV row of the step, with 68 of 256 work-groups live at depth 4096.
- **Shrinking the block to 64 positions** (one constant, `ATTN_BLOCK`) took the family from **6.058 to 3.839
  ms/token** with no arithmetic rewritten; `attn_reduce` got 2.4× slower (it merges 65 blocks instead of 17)
  for a 20:1 trade. The sweep over 256 / 128 / 64 / 32 gave the family 6.058 / 4.931 / 3.839 / 3.769 ms: **64
  is not a knee** (32 was 0.07 ms better, under a run's drift, but `attn_reduce` was on a steep ramp and the
  scratch would double). Each halving cost about 73 µs less a launch, a local description over three points
  and not a law. **Cost models fitted to this kernel have a bad record**: four were written down with
  predictions before the run and all four missed, twice by nearly 40% in opposite directions. What replaced
  fitting is a probe that removes one term at a time.
- **What the launch buys**, removing one term at a time from a copy of `attn_decode` (`tools/probe/probe_attn`,
  device-clocked, base 207 µs): both KV global loads were **55%** of the launch (39% the load messages, 16%
  cache service), the whole online-softmax update 26%, the work-group barriers 17%, `exp` 15%, the loop-carried
  chain 1.3%, one wave of K prefetch 0.5%. **The dominant term is the KV load path and it is
  throughput-shaped**, not a latency chain. The kernel issues about 3.2 M 32-byte load messages per launch.
- **The methodology pitfall that nearly voided that table**: the obvious way to price a cache miss is to hold
  the address constant so every access hits L1, but that address was loop-invariant over a `restrict` pointer
  and IGC hoisted the V load out of the loop, so the variant measured a quarter of the messages (141 µs against
  the hoist-proof control's 174). **An ablation must be a function of the loop index, or the compiler will
  answer a different question.** The correction inverted which half of the cost was bigger.

The full before/after rows and the cost-model post-mortem are in [15-step-anatomy.md](15-step-anatomy.md).

### The int8 KV cache

`--kv-cache int8` stores K and V as int8 with fp16 scales after a rotation (the "rotkv" scheme, spec 12,
[spec](superpowers/specs/2026-09-28-spec12-int8-kv-cache-design.md)), which raises the context that fits
on one card (262144 against 171008 with MTP on). `src/kernels/kv8.cl` holds every kernel that writes or reads it, one source built per family by a define:
`attn_prep_kv8` (the writer: q, K and V rotated, K and V quantised into the cache), `attn_decode_v2_kv8` and
`attn_reduce_v2_kv8` (v2's pair reading int8 K/V plus scales, the output un-rotated per head before the gate),
`pf_attn_gate_kv8` and `pf_flash_attn_kv8` (prefill), and `kv8_widen` (exact widening of the int8 cache to bf16
once per layer and chunk, which long int8 prefill uses so the raw flash kernel can run). The decode and verify
lists use the same grid and partials as the bf16 pair.

### `attn_verify_dpas` - the verify step's attention on the matrix engine (spec 28c)

A speculative-decoding verify list attends M rows at once. `attn_verify_dpas` (and `attn_verify_dpas_kv8`,
`src/kernels/attn_verify_dpas.cl`) computes the same attention on DPAS with v2's blocking exactly and writes
**v2's partials**, so the reduce is the existing `attn_reduce_v2`. The 6 × M (q-head, row) pairs of a kv-head
are the DPAS A rows, 8 to a row block; K is read through the transposed 2D block read, V through the
VNNI-transform read (int8 K/V are widened to exact bf16 per message), and the online softmax per row is in fp32
with `exp`. **q and P are carried as a high plus a low bf16 limb** (`QSPLIT`, `PSPLIT`: `hi = rne_bf16(x)`,
`lo = rne_bf16(x − hi)`, two DPAS per atom), about 16 significant bits instead of 8. That is the fix for the
failure of 2026-11-04: with P rounded to bf16 the row error grew to 1 − cos about 2e-5 against fp64 (v2: about
1e-8), which over 16 layers moved a DFlash verify row's logits by up to 9 and its argmax off the f32 top.
Random-data tests (cosine about 0.99999 either way) cannot see it; only a real-weights f32 check did.

**Which attention a list binds** is `runtime/verify_attn.h`'s table by (M, KV form, key count): v2 at M = 1
always, and at M >= 2 DPAS from a measured depth. bf16: M = 2 from 11585 keys, M = 3..5 from 4096, M = 6..8
always; int8: M = 2 from 11585, M = 3..4 from 4096. The paired v2/DPAS time ratio at 32768 keys is 1.22 (bf16
M = 2), 1.71 (M = 4) and 8.22 (M = 8) (busy-box timing, card 1; to be re-checked on a quiet box). The long-context
rework of 2026-11-10 (int8 K by 2D reads, S split by key atom, P·V by dimension atom) made the whole verify
step at 150k keys 11.4% faster in bf16 and 20.5% faster in int8, bitwise the previous kernel (record:
[probe-longctx-decode-2026-11-09.md](probes/probe-longctx-decode-2026-11-09.md)). The thresholds were re-timed
there and four rows would move; they wait for the f32 judge because each changes rounding (open-work.md). The
record of the kernel is [probe-verify-attn-2026-11-03.md](probes/probe-verify-attn-2026-11-03.md).

---

## Prefill kernels

Everything below runs **only** on `Engine::prefill`'s walk; none of it is bound by `src/runtime/capture.cc`,
and the decode list is unchanged with or without it. Sources are `src/kernels/prefill/` (plus `kv8.cl` for the
int8 KV cache).

Two rules hold across the whole family:

- **`M` is a runtime argument, never a `-D` and never in a variant name.** One binary set serves every
  `--prefill-chunk` width. What stays in a name is what the binary bakes as a stride or a grid: `K`, the
  previous linear's split-K width `S_PREV`, the norm's group counts, a dequant shape.
- **`S = 1` everywhere.** Decode's split-K exists to buy hardware threads at `M = 1`; at `M = C` the row axis
  already saturates the grid. Leaving decode's `S` in place when these kernels were first widened cost a
  measured 361 to 473 ms per chunk against 130 ms at `S = 1`.

The prefill path runs entirely on Level Zero: a chunk **never waits on the host**, and every GEMM is one of our
own OpenCL C kernels on the same in-order command list as everything else (the sycl-tla backend it replaced was
removed on 2026-10-27). With the default flash attention a full chunk is `1 + 48 × 135 + 16 × 123 = 8449`
launches at every chunk width (derived from the walk in `src/runtime/prefill/step.cc`, asserted by
`prefill_smoke_test` through `Context::launches()`); `step_head` adds 5 once per `prefill()`. Short chunks use
fewer launches (below). Where the time goes, per phase, is in [BENCHMARKS.md](BENCHMARKS.md).

### The small kernels, transcribed from decode

`pf_res_fold`, `pf_norm_finish`, `pf_silu_mul`, `pf_embed_gather`, `pf_gated_head` and `pf_ab_proj` are
`prep.cl`, `embed_gather.cl` and `gemv_bf16.cl` at runtime `M`, with three classes of edit and nothing else:
`M` becomes an `m_count` argument in every index expression, the split-K widths become 1, and
`m = get_group_id(1)` needs no mask because the grid's y extent is `M`. Every rounding, both bf16 RNE steps,
the `fma` square-accumulate, `1.0f / sqrt(x)` and both reduction trees are the decode file's text, which is
what lets `tests/prefill/pf_prep_test.cc` hold the `M = 1` output **bit-identical** to the decode binaries.
`pf_embed_gather` loses decode's `debug_flag` channel for an out-of-vocabulary id, so `Engine::prefill` bounds
every id on the host and throws by name.

### `pf_dequant_slab` + `pf_gemm` - one int4 linear, on one in-order list

This is the prefill path's largest phase. `pf_dequant_slab` expands one **1024-column slab** of an int4 linear
into a bf16 scratch; `pf_gemm` multiplies the chunk's activations by that slab into the output columns it
covers. For `n0 = 0, 1024, … < N` the pair alternates on the in-order list with **no host wait** between the
two, between slabs, or before the consumer (`src/runtime/prefill/linear_l0.cc`). The int4 contract is `gemv.cl`'s
shift path, `(q − 8) × scale` with the product in fp32 and exactly one RNE cast to bf16, held bit-exact against
`tools/oracle/dequant.py` by `tests/prefill/dequant_test.cc`.

**`pf_gemm` is our own DPAS GEMM**, `src/kernels/prefill/pf_gemm.cl`:

```
C[l][m][n] fp32 = Σ_k A[l][m][k] · B[l][k][n]
```

with a 32-dpas k-tile body, 2D block loads, a two-deep prefetch and a split work-group barrier, one 256 × 256
output tile per work-group. `lda`, `ldb`, `ldc` are runtime arguments and a batch index `l = get_group_id(2)`
offsets A, B and C, so one launch serves a batched GEMM. Variants:

- **`pf_gemm_T0`**: B stored `[K][N]`, the ordinary weight orientation;
- **`pf_gemm_T1`**: B stored `[N][K]` with K contiguous, which is the KV cache's layout. The DPAS B operand for
  one k-step is, per lane `n`, eight 32-bit words holding `(B[2t][n], B[2t+1][n])` pairs, and with K contiguous
  those pairs are adjacent 16-bit words, so a transposed 2D block load delivers lane `n`'s 8 k-pairs directly.
  Same DPAS instructions, same k order, so T0 and T1 are bitwise equal, and are;
- **`pf_gemm_T0_SILU`**: the `gate‖up` epilogue. A sub-group's four 16-wide n-atoms sit at `n0 + 64·sn + 16·b`
  with `n0` a multiple of 256, so atom `b = 2p` is a gate block and `2p+1` its up block, both already in this
  sub-group's registers one lane per column, with no cross-lane traffic. The epilogue runs `pf_silu_mul`'s whole
  chain and stores bf16; the fp32 `[M][34816]` intermediate is never written. It removes one launch a layer.

`runtime::prefill::gemm_l0` asserts every requirement before launching: `M % 256 == 0` and `N % 256 == 0` (the
**caller** pads; the padding is computed and never read), `K % 32 == 0`, every base 64-byte aligned, leading
dimensions multiples of 16 bytes. The `width` of every 2D block descriptor is the logical matrix width in bytes
and the `pitch` the leading dimension, so a read never clamps inside the matrix or reaches past it.

**Measured.** The probe this kernel generalises reached **156.53 TFLOP/s** and was bitwise equal to `sycl-tla`
([08-decode-vs-prefill.md](08-decode-vs-prefill.md), "How prefill got here"). In production it runs at **161.7
TFLOP/s, about 88% of the measured 183.45 TFLOP/s bf16 DPAS peak**
([probe-dpas-rates-2026-09-22.md](probes/probe-dpas-rates-2026-09-22.md)). Separating the two halves with one
wait per slab (a 4096-token prompt, 7616 launches each) gave `slab_dequant` 274.5 ms and `slab_gemm` 1344.3 ms
(2026-09, first stack).

**What is dead, and each died to a number.** The negative results cost less than re-discovering them:

- **Fusing the int4 dequant into the GEMM mainloop.** Both variants were bitwise identical to the two-pass path
  over all 71,303,168 outputs and both were slower: the dequant pass already runs at about 546 GB/s (about 90%
  of peak), so only 0.826 ms is in it at that shape, and doing the same arithmetic in the mainloop cost 2.759
  ms even issued once per weight ([details](probes/probe-fused-dequant-2026-09-22.md)).
- **W4A8 on the mixed 4-bit by 8-bit DPAS**: 1.017× the bf16 control (0.965× counting the activation quantiser),
  against a bar of 1.53×, and 2.79% relative L2 error from activation outliers
  ([details](probes/probe-w4a8-2026-09-23.md)).
- **oneDNN's fused int4 W4A16 matmul**: 98.28 TFLOP/s on `gate‖up` at M = 2048, below the two-pass path.
- **bf16 intermediate buffers**: halving the bytes changed nothing, because `ocloc` exposes no 16-bit block
  write wider than 8 rows, so the epilogue is bound by store message count, not bytes.

### Short chunks: `pf_gemm_small` and `pf_gemm_q4`

A full chunk is 2048 rows; a chat turn's chunk is often a few hundred. `pf_gemm` gives each 256 × 256 output
tile one work-group, so a 1024-column slab is only 4 work-groups for any chunk up to 256 rows (4 of 32 Xe-cores,
each doing a whole tile's DPAS for the few rows the chunk has), and the slab dequant (about 110 ms of a
Qwen3.8 chunk) is paid whatever the row count.

- **`pf_gemm_small`** (up to 1024 rows): a sub-group owns `NA` row blocks of 8 × `NB` 16-column atoms, a
  work-group `SN` such sub-groups side by side, and the grid covers only the rows that exist, so a short
  chunk's slab becomes many small work-groups spread over every core.
- **`pf_gemm_q4`** (an int4 linear of up to 384 rows; the `gate‖up` SiLU form up to 320): `pf_gemm_small`
  reading the **int4 words directly** and converting them to bf16 in registers, one launch for the linear's
  whole N, no slab. The conversion runs once per row tile, so it wins only for few rows.

Both are **bit-identical** to the slab walk (the same conversion statement for statement, the same DPAS chain per
accumulator from zero, and a DPAS output row depends only on its own A row), so the choice is speed only. A
chunk of 1 / 54 / 237 / 1024 ids went from 696 / 709 / 729 / 857 ms to 229 / 238 / 288 / 660 ms with the first
and a short chunk's linears are fewer launches with the second (records:
[probe-small-chunk-2026-11-08.md](probes/probe-small-chunk-2026-11-08.md),
[probe-fused-small-chunk-2026-11-08.md](probes/probe-fused-small-chunk-2026-11-08.md)).

### The int8 prefill linears

`pf_quant_had`, `pf_requant_rot`, `pf_colmax_rot` and `pf_gemm_i8` (`src/kernels/prefill/pf_int8.cl`) are the
opt-in `--prefill-backend l0-int8` path: activations and weights are rotated by a Hadamard transform, quantised
to int8, multiplied on the int8 matrix engine (twice the bf16 rate) and rescaled in the epilogue. The design,
the rejected alternatives and the accuracy evidence are in [17-int8-prefill.md](17-int8-prefill.md).

### `pf_attn_prep` and `pf_flash_attn` - prefill attention

`pf_attn_prep` is `attn_prep` widened to a chunk: q/k RMSNorm, partial RoPE over dims 0..63, and the chunk's K
and V written into the cache at absolute positions `[pos, pos+C)`. Grid `(24 q-heads + 4 kv-heads, C)`,
work-group 256, so work-item `i` owns dim `i` and the norm's reduction domain is the work-group. **Two binaries
come from one source.** `pf_attn_prep` writes fp32 `attn_q` and `attn_gate` and is held **bit-identical to
`attn_prep_M1`** by `tests/prefill/pf_attn_test.cc`; `pf_attn_prep_q16` (`-D Q_BF16=1`) writes bf16 `attn_q`
(DPAS needs bf16 operands) and no gate (the gate kernel reads the gate columns straight out of
`qkv_partials`). Everything above the store is one piece of text, so the bf16 variant is provably **one
`rne_bf16` at the store and nothing else** (`attn_chunk_test` asserts it word for word).

**`pf_flash_attn` is the attention the engine runs** (spec 6; `src/kernels/prefill/pf_flash_attn.cl`): one fused
launch per full-attention layer for all four kv-groups, with no score scratch. Grid `(ceil(C/8), 4)`,
work-group of 6 sub-groups (one per q-head of the group), **8 query rows per sub-group**, built with
`KT=64` keys per tile and 256 registers per thread. Each sub-group holds its O accumulator (8 rows × 256 dims)
in registers; S for one key tile has the row in the component and the key in the lane, which is exactly the
bf16 DPAS A-operand layout, so P goes to the P·V product with no data movement. K comes through the transposed
2D read, V through the VNNI-transform read, softmax is fp32 online with P rounded to bf16 as the P·V operand,
and the score scale is folded with `log2(e)` so the exponential is `exp2` (see the flash research below for
why that matters). `pf_attn_gate` (`pf_attn.cl`) is `attn_reduce`'s tail: `rne_bf16(f32(rne_bf16(o)) ·
sigmoid(gate))`, with the gate read from `qkv_partials`.

The **composed attention** (`pf_gemm_T1` for `S = QKᵀ` per 256-row block, `pf_softmax_causal`, `pf_gemm_T0` for
`P V`, batched over a group's six q-heads, then `pf_attn_gate`) is kept only as the correctness reference the
tests grade flash against; since 2026-11-06 nothing but a test can select it. Its `pf_softmax_causal` makes
three passes over `S` (row max, sum of exponentials, normalised store; one bf16 rounding at the store, an exact
`+0.0` past the causal bound so the P·V padding adds nothing). The composed path's bf16 attention weights
perturb a cancelling sum, so the largest single-word relative difference reaches 2.97e-02 even though the
relative L2 is 2.7e-03; the mechanism is arithmetic, not a defect.

### `gdn_chunk` - the chunked gated delta rule

Ten launches per GDN layer (`src/runtime/prefill/gdn.cc`): the chunked gated delta rule at intra-chunk width 64
(the "WY" representation), reading and writing the **same** `gdn_state` and `conv_ring` buffers `gdn_step`
uses, in place and in decode's layouts, so decode continues from position `pos + C` with no translation. The
chain is `pf_gdn_seed` (lift the ring's three older slots into a flat seed), `pf_gdn_conv`, `pf_gdn_l2norm`,
`pf_gdn_gate`, `pf_gdn_A`, `pf_gdn_solve` (the triangular solve `T = (I + A)⁻¹`, in place), `pf_gdn_wu`,
`pf_gdn_A2_hilo`, `pf_gdn_scan_dpas_split` (the sequential chunk-to-chunk state scan, the only writer of
`gdn_state`) and `pf_gated_head`, so `y` leaves the call post-gate. The bf16 rounds happen **inside**
`gdn_chunk`, at the same points `gdn_step.cl` rounds, which is what the self-consistency bar between the two
paths rests on.

**The scan is the one approximation in the default prefill path.** It keeps fp32 register state and an fp32
global master state and, at each 64-position sub-chunk, forms two BF16 limbs:

```
S_hi = rne(S)
S_lo = rne(S - bf16(S_hi))
```

Both `W·S` and `Q·S` run independent fp32-accumulating DPAS chains over the two terms and combine once, at the
end. Each product is exact (a bf16 × bf16 multiply into fp32 loses nothing), so the only rounding is the
conversion into limbs, and two limbs carry about 16 bits of mantissa instead of 8. It is worth 2.33× on its own
profile row (about 297.6 to 127.7 ms, ABBA, 96 launches each; the scan row has since fallen further, below). It
is gated on tokens and state cosines rather than on `memcmp`: 93/93 determined golden rows, state cosines
0.99991, 0.99971, 0.99990 against a bar of 0.999. One honest limit: `vn` inside the A2 term is still a single
bf16 operand, measured harmless on three prompts of one checkpoint and not proven in general.

**Two pitfalls from this family, both bought at full price.**

- **A green gate is not evidence unless the run says which kernel it dispatched.** An earlier attempt at this
  scan passed every gate while the engine was launching the old vector kernel. Every gate run now prints the
  entry it launched (`gdn_scan_launched_entry()`, read back from the pointer the launcher handed the kernel
  cache) next to what the walk resolves to (`gdn_scan_entry_name()`), and the non-band tests assert they agree.
  Run with its dispatch proven, that same first kernel *failed* 92 of 93 rows.
- **Barriers were not what the triangular solve spent.** The design assumed synchronisation dominated
  `pf_gdn_solve`. Deleting 64 of 128 barriers bought 1.7 ms of a 73.5 ms row, so the premise was wrong by about
  18×; both barrier-free variants were bit-identical and both were slower. The row is the serialised
  recurrence itself.

### How the prefill kernels got here

The kernel-level conclusions of the September 2026 prefill investigations, folded in from their standalone
write-ups (the strategy-level ones are in [08-decode-vs-prefill.md](08-decode-vs-prefill.md), "How prefill got
here").

**The split-BF16 scan: design, the failed gate, the fix (2026-09-20 to 09-23).** A first single-BF16 DPAS scan
was fast (3.4 to 0.85 ms a layer and chunk) and failed a real golden token gate (92 of 93 rows; the code
prompt's layer-60 state cosine 0.9985 against 0.9998 for the vector kernel). The second attempt keeps the DPAS
rate and buys the precision back with two BF16 limbs. The state update `D = vn * exp(gl - gc)` is split the
same way (the high chain starts from the fp32-decayed master state, the low chain from zero, combined once).
The first version left the intra-chunk term `A2*vn` at a single BF16 on the reasoning that the state path was
where precision had been lost; that was the whole failure. A throwaway scalar-fp32 2×2 (each of `A2` and `vn`
independently fp32 or bf16, nothing else changed, one full golden gate per cell) attributed it: with `A2` in
bf16 the gate is 92/93 whatever `vn` is; with `A2` in fp32 it is 93/93. It is `A2`'s rounding, not `vn`'s; the
mechanism is not established (the decay `exp(gc_i - gc_j)` spanning decades inside one row is the suspect).
The fix splits `A2` only (`vn` stays one BF16). A fixture aimed at the `A2` chain (`A2 = 22.65625`, one bf16
ulp of the high limb) was verified RED by deleting the low chain. Since 2026-11-10 `pf_gdn_A2_hilo` writes those
`A2` hi/lo words itself and the scan reads them from global memory: SLM 40,448 to 30,208 B a work-group, four
work-groups a core instead of three, the scan row 64.1 to 44.7 ms for a 2048 chunk, bitwise
([probe-gdn-prefill-2026-11-10.md](probes/probe-gdn-prefill-2026-11-10.md), section 8). This is the only
default in the prefill path that is not bit-identical to what it replaced; its claim is "approximate with
measured error, gated on tokens and state cosines". The old vector scan and the other entries stay compiled but
unbound (the environment selector that picked between them was removed in the zero-config cleanup), and
promoting any of them means re-running every gate with the dispatch proven inside each run.

**The triangular solve: the premise was wrong by 18×.** `pf_gdn_solve` costs about 0.77 ms a layer and chunk
(73.5 ms over 96 launches). Three alternative entries, all `memcmp`-equal to the default (`register`: T in its
own SLM array with zero row barriers; `private`: zero barriers with a per-lane array lowered to scratch; `1bar`:
in place with one barrier a row), measured per layer and chunk: default 0.7656 ms, `register` 1.2073 (+57.7%,
32 KiB of SLM halves occupancy), `private` 0.8708 (+13.7%), `1bar` 0.7479 (−2.3%). All 128 barriers are worth
about 3.4 ms (derived), 4.6% of the row; the row is the serialised recurrence and its SLM traffic. Any future
attempt has to change the algorithm (a block-recursive inversion, a different numerical project), not
resynchronise. Details worth keeping: the dead tail of T carries `-0.0f`, invisible to `CHECK_EQ(., 0.0f)`, so
new entries seed their output from the staging expression; the solve dispatch is proven by kernel-cache counts,
not output; the matrix is `(I + A)^-1`, not `(I - A)^-1`.

**Flash prefill research (2026-09-27), and what shipped.** The question was how to make `pf_flash_attn` (116.3
ms for the attention of a 4096-token prompt, 28 TFLOP/s = 15% of the 183.45 TFLOP/s peak, against vLLM's 63.1
ms) faster. Findings:

- **The kernel spilled, and `exp` was the cause.** The main loop was 1639 instructions per 128 `dpas.8x8`, with
  9 of 16 O accumulator vectors going to scratch on every KV tile. IGC lowers each `exp` to two `math.exp` plus
  a range-split sequence; `exp2` with the scale and `log2(e)` folded in removes the spill and cuts the loop to
  1137 instructions. `exp2` and `native_exp2` compile identically, so there is no precision trade on this
  compiler. Streaming Q changes nothing: an earlier "Q0 beats Q1" was a spill finding.
- **Two one-line changes give 1.42× at every depth** with the fp64 cosine unchanged (diagnostic grade, paired
  ratios): `exp2` alone 1.17× to 1.23×; **8 query rows per work-group instead of 16** (6 sub-groups, so an
  Xe-core holds 30 of 32 thread slots instead of 24) 1.14× to 1.27× alone.
- **The softmax-side levers are neutral once the spill is gone** (causal mask on diagonal tiles only, FA-4's
  conditional rescale, hardware bf16 convert, deferred row sums, reversed row order: −3% to +1%). The kernel is
  latency and memory bound with too few threads to hide it, not issue bound.
- **Structural ceiling: operand bandwidth.** With 8 rows per sub-group every K and V B-operand serves one
  `dpas.8x8`: 576 B of operands per DPAS against the production GEMM's 192 B. Going past vLLM needs 16 or 32
  rows per sub-group, which at head_dim 256 forces O split on the dimension across sub-groups with P exchanged
  through SLM (what oneDNN and IPEX do); whether that is needed was left to a probe. Estimated remaining
  levers: cooperative 2D prefetch of the next K and current V plus a split barrier per tile (1.1× to 1.3×),
  split-dimension (1.3× to 1.8× beyond), larger K messages (1.0× to 1.1×).
- **Shipped (spec 6c):** 8 rows per work-group first (attention 101 ms from 117), then `exp2` too (81 ms). The
  `exp2` form moves the rounding point and failed the narrow flash-against-composed check on `l0`
  (0.999951245 against 0.999959889), so it was held until the operator ruled it on on 2026-09-27; it ships
  (`EXP2=1`). The lever arms live in `tools/probe/pfa_levers_research.cl`.

---

## Kernels of the other models

The Qwen3.8 decode and prefill kernels above are shared where a model has the same layer type: other models
reuse `prep`, `gemv_bf16`, `argmax` and the attention structure with different widths, named by suffix
(`_D<hidden>`, `_GK<k>V<v>` for GDN head counts, `_Q<q>KV<kv>` for attention heads). Each model's own kernels
live in a directory of `src/kernels/` and are described in their source headers and specs:

| where | what |
|---|---|
| `moe.cl`, `prefill/pf_moe*.cl` | mixture-of-experts block of one decode step (route, gate‖up, down) and its prefill GEMMs: Ornith 1.5 (spec 15) |
| `k2/` | K2-Horizon: spec 10's attention carried to head_dim 128, routed experts, value-expert attention (spec 18) |
| `kolibri/` | Kolibri-1: sliding-window and full attention, experts (spec 20) |
| `qwen4exp/` | Qwen3.8-Flash-Next: the block-selection attention, experts, per-layer embeddings, multi-token head (spec 21) |
| `bolt/`, `bolt_kernels.h` | Nemotron 3.5 Bolt: the Mamba-2 mixer, MoE, and the full no-position-embedding attention `mg_attn*` (spec 25); also LFM2-family prep kernels |
| `dflash/` | the DFlash draft model (spec 19) and LiLiCorr (`ll_*`, spec 26) |
| `draft_vocab.cl` | the MTP draft's argmax over a reduced vocabulary (spec 8) |
| `tp.cl`, `prefill/pf_tp.cl` | tensor parallel: the all-reduce after each row-parallel linear by P2P writes, and one card's half of the argmax (spec 17) |
| `pp_handoff.cl` | the pipeline-parallel hand-off of the residual between the two cards (spec 16) |
| `noop.cl`, `ctrl_read.cl`, `bw_sum.cl` | measurement helpers: per-launch fixed cost, a control-block read, a read-only bandwidth probe |

Bolt's attention (`mg_attn_decode` + `mg_attn_reduce`, `mg_attn_eager`, `mg_pf_flash_attn`) is `kol_attn.cl`'s
v2 structure at GQA 16 over the position-embedding-free cache, with the gate fused into the reduce; its host
twin is `tests/kernels/mg_attn_ref.h`. Bolt is parked until Flash-Next is done (open-work.md).
