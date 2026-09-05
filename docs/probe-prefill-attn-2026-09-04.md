# P4 - sycl-tla FMHA forward at head_dim 256 (spec 2 §4)

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

sycl-tla: `91e5bd735517d8e79591b41e0d0cd37a7bacdca7` content-verified by
`include/cute/util/compat/launch_policy.hpp` SHA-256
`5460b1e5119308896ab65ca99127179c2b47c2ca8173db4349b938d680fea661`.

## Pin and source-line audit - first action under ruling A11

Before this pre-registration, the box source was read at the content-verified
pin. The quoted contents and line numbers remain valid, with path drift from
the old plan: the mainloop is now
`applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp`, and the
runner is `examples/06_bmg_flash_attention/xe_fmha_fwd_runner.hpp`; they are not
under the plan's `applications/flash_attention/` path. The generic generator
remains `benchmarks/flash_attention/fmha_configuration.hpp:291-311`, the shipped
example's 64/96/128/192-only chain remains at
`examples/06_bmg_flash_attention/06_xe_fmha_fwd.cpp:96-133`, and
`compat::set_default_queue` remains at `include/cute/util/compat/device.hpp:899`.

The generic generator's template parameter list, source-verified verbatim at
the pin, is:

```cpp
template<FMHAMode Mode,
         class ElementQ, class ElementK, class ElementV, class ElementO,
         class LayoutQ, class LayoutK, class LayoutV, class LayoutO,
         class ElementScale, bool Causal, bool VarLen, bool CachedKV, bool PagedKV, bool Persistent, bool BlockScale,
         int WgTileQ, int WgTileK, int WgTileV,
         int SgTileQ, int SgTileK,
         int HeadDimQK, int HeadDimV>
struct FMHAConfigGenWithTileShape;
```

## The shape, which is the whole question

head_dim **256**, 24 q-heads / 4 kv-heads (GQA 6:1), causal, bf16 K/V read from
our cache `[pos][4][256]` per layer. Intel's shipped BMG prefill example covers
head_dim 128, not 256. A larger output head dimension can increase the
accumulator footprint retained across the KV loop; the first build is therefore
the task fork, not a timing sweep.

## Pre-registered predictions (written 2026-09-05, BEFORE any P4 measurement)

1. It instantiates through `FMHAConfigGenWithTileShape` with
   `HeadDimQK = HeadDimV = 256` (medium-high confidence).
2. The first configuration uses `WgTileQ=128`, `WgTileK=32`, `WgTileV=64`,
   `SgTileQ=16`, `SgTileK=32`: `ShapeQK=<128,32,256>`,
   `ShapePV=<128,64,32>`, and `ShapeOutput=<128,256>`, hence `VTiles=4`.
   The naive doubled control uses a 256-wide Q tile and `VTiles=8` and is timed
   only if the first configuration builds and passes correctness.
3. Causal FMHA at depth 4096 and C=4096 costs under 3.2% of a chunk, or under
   4.4 ms/layer across 16 FA layers (derived from the old P2 2.215 s
   pre-registration; this is a prediction, not a corrected-P2 result).
4. The cache is readable in place with K strides `{1024, 1, 256, 0}` and the
   matching V layout, without a copy.
5. Against `tests/kernels/attn_ref.h`, max relative error is at most `1e-2` and
   mean relative error at most `1e-3` for elements with
   `|ref| >= 1e-3 * max|ref|`.

## Configuration assertions to check at the verified pin

| source | condition | configuration argument |
|---|---|---|
| `applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp:162-165` | `(VTiles * PV N-tile) % QK K-tile == 0` | `4 * 64 == 256`, divisible by QK D tile 256 |
| `benchmarks/flash_attention/fmha_configuration.hpp:304-305`; runner `:1549-1550` | output and PV Q tiles agree | both are 128 |
| mainloop `:958-959` | PV and QK subgroup-layout sizes agree | retain a D-mode subgroup factor of 1 |
| mainloop `:761-762` | bf16 packed probability fragment count is even | if it fires, first adjust `SgTileK` |
| generator `:304-305` | Wg Q/K tiles divide by Sg Q/K tiles | 128/16 and 32/32 divide exactly |
| `06_xe_fmha_fwd.cpp:1591` | persistent and causal may not both be true | use the non-persistent causal scheduler |

`kSumDivVT = (kSumSize % VTiles == 0)` at mainloop `:756` selects the fast
row-sum reduction. The result record will state the selected branch.

## Measurement protocol if the build succeeds

The probe will use the engine's in-order SYCL queue via
`compat::set_default_queue(runtime::prefill::sycl(cx))`; run it after a discarded
warm-up and use eight replays, discard the first three, and take the median of
the final five. It will rotate independent synthetic caches, prove the custom
strides against a packed copy bit-for-bit, then compare C=64 and C=256 against
the scalar reference before recording timings for C=1024, 2048, and 4096.

If the generic configuration cannot compile, its first diagnostic and a priced
own-flash fallback are the complete planned deliverable; no timing sweep occurs.


## 2026-09-05 result - generic head_dim 256 builds and launches

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held). The
source-line audit above was completed before the pre-registration, per ruling
A11. The build used the AOT 256-GRF backend default and the run exported all
four prescribed runtime IGC variables.

The generic `FMHAConfigGenWithTileShape` instantiated and its concrete
`FMHAKernel` compiled and launched on `Context`'s in-order interop queue.
The real single-launch gate used C=64, depth=4096, 24/4 GQA, bf16 K/V in the
engine layout, K stride `{1024,1,256,0}`, V stride `{1,1024,256,0}`, and
the required `compat::set_default_queue`; `can_implement` passed and
`wait_and_throw` returned (measured build/run outcome, iterate grade).

The first standalone compile exposed a missing
`cutlass/util/packed_stride.hpp` include and the CuTe host-`printf` macro.
The example runner then exposed its optional oneMKL random-fill dependency,
which is not installed on the box. The probe uses the source-verified direct
FMHA launcher instead; no third-party checkout was changed.

### Timed launch battery

All values are measured, iterate grade. They are host wall time from enqueue to
the in-order queue drain: eight launches per cell, discarded warm-up launch,
first three samples discarded, median of the final five. This is **not** a
device timestamp measurement.

| config | VTiles | C | depth | us/launch | ms x 16 FA layers | % of corrected P2 GEMM chunk |
|---|---:|---:|---:|---:|---:|---:|
| 128x32x64 | 4 | 1024 | 4096 | 22903.267 | 366.452 | 97.2% |
| 256x32x32 | 8 | 1024 | 4096 | 20658.295 | 330.533 | 87.7% |
| 128x32x64 | 4 | 2048 | 4096 | 40317.177 | 645.075 | 94.9% |
| 256x32x32 | 8 | 2048 | 4096 | 36034.011 | 576.544 | 84.8% |
| 128x32x64 | 4 | 4096 | 4096 | 54780.824 | 876.493 | 55.3% |
| 256x32x32 | 8 | 4096 | 4096 | 48488.216 | 775.811 | 48.9% |

The percentages divide the 16-layer time by P2's corrected all-layer GEMM
totals of 376.854, 680.062, and 1585.201 ms at C=1024/2048/4096 respectively
(derived by summing the corrected P2 shape rows). The VTiles=8 control is
faster in every timed cell (measured, iterate grade), but this is a comparison,
not a tuning decision.

The VTiles=8 build also emitted IGC's compiler warning that the instantiated
SIMD16 kernel was allocated 256 registers and "spilled around 516" (measured
build-artifact outcome, iterate grade). Thus its faster host-wall rows are not
a clean no-spill choice; they must not select a production configuration before
the missing correctness gates and a device-timestamp attribution are complete.

### Scope limitation

This probe has proved generic compilation, one in-place-stride launch, and the
timing battery. It has **not yet** completed the required packed-cache
bit-exact comparison or the scalar-reference error report at C=64/256. Its
timings are therefore evidence for T6's ceiling only, not an inherited
correctness decision for L1; the missing checks remain P4 work.

## 2026-09-05 - P4 attention-cost diagnosis at the A13 width

Grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held).

### A. Causal masking is active and prunes future KV work

This probe instantiates all FMHA configurations with `Causal = true` and has
compile-time assertions for that fact (source-verified, iterate grade). At the
verified pin, `XeFMHAFwdKernel::operator()` derives `seq_len_new` from each Q
tile's causal prefix and then derives `k_blocks` from that length; its mainloop
therefore ends at that Q tile's causal KV prefix rather than executing a full
C×D tile range and masking it afterward (source-verified, iterate grade). The
final partial tile additionally writes `-INFINITY` into future score lanes
before softmax (source-verified, iterate grade). Causal masking is not the
unnecessary-full-matrix defect.

### B. Same-route head-dimension comparison

The bounded `--hd-compare` path holds the `256×32×32` work-group tiles and
`16×32` subgroup tiles at C=2048 and depth=4096. It changes only head dimension
and its required `1/sqrt(head_dim)` QK scale. Each timing is host wall time,
eight replays with a warm-up launch and the first three samples dropped
(measured, iterate grade); TFLOP/s is derived from QK^T plus PV across 16 FA
layers.

| head_dim | VTiles | us/launch | ms × 16 FA layers | work × 16 layers | TFLOP/s | grade |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 8 | 36012.104 | 576.194 | 3.299 TFLOP | 5.725 | measured time; derived work/rate, iterate |
| 128 | 4 | 5506.748 | 88.108 | 1.649 TFLOP | 18.719 | measured time; derived work/rate, iterate |

Head dimension 256 is 3.27× less efficient than the otherwise fixed
head-dimension-128 route (derived from the two measured rates). This is a real
structural penalty, but the 128 route is still only 11.4% of P2's measured
164.21 TFLOP/s GEMM peak, while the model's required 256 route is 3.5%
(derived). Head dimension therefore does not explain the approximately 25×
attention-versus-GEMM gap by itself.

### C. Generated image: DPAS and block loads are present

The AOT shader-dump rebuild emitted the selected 256-wide, VTiles=8 FMHA image.
Its `.zeinfo` reports SIMD16, 256 GRFs, `has_dpas: true`, 33024 B spill storage
and a 65536 B scratch allocation (measured compiler artifact, iterate grade).
The spill value agrees with the already-recorded ~516-register IGC warning; it
is recorded here only to identify the exact image, not reopened as a cause.

Its inner vISA loop contains repeated `dpas.bf.bf.8.8 (M1, 16)` operations, and
the matching ISA contains repeated `dpas.8x8 (16|M0)` operations (measured
compiler artifact, iterate grade). The operand paths are `lsc_load_block2d.ugm`
in vISA and `load_block2d.ugm` in ISA, including the QK/PV loop loads; they are
not per-lane scattered gathers (measured compiler artifact, iterate grade).

### Verdict

No removable defect survived A-C. The 256-head path has a material structural
efficiency penalty, but causal scheduling, DPAS generation, and block-load
selection are all present. The model-required hdim=256 attention cost therefore
survives this diagnosis at 576.194 ms per C=2048 chunk across 16 FA layers
(measured host-wall, iterate grade, correctness still incomplete). No tile
shape tuning was started.
