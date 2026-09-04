# P2 - sycl-tla stock bf16 GEMM at the prefill shapes (spec 2 §4)

grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

1. **≥ 90 TFLOP/s at M = 4096 on gate‖up** (spec §4's P2 row; vLLM's implied
   98.2 TFLOP/s sustained is the yardstick, `explorer-2 §4`).
2. TFLOP/s rises monotonically in M over {512, 1024, 2048, 4096} at every
   shape: weight bytes amortise over more rows, and the work-group tile is
   `Shape<_256,_256,_32>` (`00_bmg_gemm.cpp:364`), so M = 512 is only two
   tiles deep and M < 256 would not fill one.
3. `out/o_proj` (6144×5120) is the worst cell at every M - the smallest N,
   and the shape decode also finds hardest (`docs/12`, 533 GB/s).
4. Two runs of the same cell are **bitwise identical**. Source basis, not
   hope: the configuration omits the `TileScheduler_` template argument, so
   it is the data-parallel `PersistentScheduler`, and
   `include/cutlass/gemm/kernel/xe_gemm.hpp:82-87` static_asserts
   *"Intel Xe does not support specializing the tile scheduler"* - Stream-K
   routes to a **different** kernel (`xe_gemm_cooperative.hpp`) which this
   probe does not instantiate. There is no atomic accumulation into C.
5. `Gemm::get_workspace_size(arguments) == 0`. It has to be: the queue
   variant treats a non-zero workspace as a hard error
   (`00_bmg_gemm_with_sycl_queue.cpp:263-265`).

## What prediction 1 implies for the whole spec - pre-registered, DERIVED

Per-token GEMM FLOPs, from `model::Qwen35`'s table:
GDN layer = QkvZ + OutProj + GateUp + Down = 0.76546 GFLOP/row, ×48;
FA layer = Qkv + OProj + GateUp + Down = 0.74449 GFLOP/row, ×16.
**Σ = 48.654 GFLOP/row.** (Reconciles with `explorer-2 §1`'s 48.97
GFLOP/token, which additionally counts the conv, the recurrence and the a‖b
linear - the two are different quantities, not two values for one.)

At C = 4096 that is **199.3 TFLOP of GEMM**. At exactly 90 TFLOP/s the GEMM
term alone is **2.215 s** - **6.7 % ABOVE vLLM's 2.076 s for the entire
pp4096** - before dequant, attention, GDN, norms or interop. So:

> **If P2 measures 90 TFLOP/s, the composed ceiling lands under vLLM and spec
> §2's second bullet fires.** To reach 2.076 s device-side the GEMM term must
> run at **≥ 96 TFLOP/s** *and* everything else must be free. This is written
> down before the measurement so T6's conclusion cannot be a hindsight
> rationalisation of whatever number comes out.

## The configuration measured

The `sycl-tla` content sentinel was verified on the box at
`/home/user/sycl-tla/include/cute/util/compat/launch_policy.hpp`:
SHA-256 `5460b1e5119308896ab65ca99127179c2b47c2ca8173db4349b938d680fea661`
(source verification, pin
`91e5bd735517d8e79591b41e0d0cd37a7bacdca7`; the rsynced checkout has no
`.git`, so no Git identity was consulted).

This probe copies the following stock queue-example alias chain from
`examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp`; every line below is
source-verified at that content pin, not tuned for this probe.

| alias | instantiated value | source line |
|---|---|---:|
| `ElementInputA` / `ElementInputB` | `bfloat16_t` | 347-348 |
| `ElementAccumulator` / `ElementOutput` | `float` | 345, 349 |
| `LayoutA` / `LayoutB` / `LayoutC` / `LayoutD` | `cutlass::layout::RowMajor` | 351-354 |
| `GmemTiledCopyA` / `GmemTiledCopyB` | `void`; `MainloopXeL1Staged` selects its 2D copies | 357-362 |
| `TileShape` | `Shape<_256,_256,_32>` | 365 |
| `TiledMma` | `XE_DPAS_TT<8,float,bfloat16_t>` with subgroup layout `Shape<_8,_4,_1>, Stride<_4,_1,_0>` - 32 subgroups, each 32×64×32 | 368-378 |
| `PipelineStages` | 2 | 380 |
| mainloop policy | `cutlass::gemm::MainloopXeL1Staged<2>` | 381 |
| epilogue policy | `cutlass::epilogue::IntelXeGeneric` | 382-400 |
| scheduler | `TileScheduler_` omitted from `GemmUniversal`; the adapter therefore uses the data-parallel `PersistentScheduler`, not Stream-K | 402-420 |
| wrapper | `cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>` | 414-420 |

Thus the named configuration measured here is: **stock
`MainloopXeL1Staged<2>`, tile 256×256×32, DPAS-TT bf16→fp32 TiledMMA,
8×4×1 subgroup layout (32 subgroups), two pipeline stages, and the
data-parallel PersistentScheduler.** The older `MainloopIntelXeXMX16` name in
`explorer-3 §2` and the nearby `docs/01-hardware.md` text is real but
superseded for this measurement; it was not the policy instantiated here.

## The B layout the GEMM requires

The 16×16×16 identity-A test printed `stride_B = (_1,16,0)` and matched the
`[K][N]` reference bit-for-bit while the `[N][K]` reference mismatched:
**the GEMM requires bf16 `[K][N]` row-major, `ldb = N`**. This preserves the
shared interface contract and selects P3's non-transposed scratch for T6's
future composition (decision, source/measurement-verified; no T6 work is done
here).

## Measurement conditions and method

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

- Device: Intel Arc Pro B70 (measured, iterate grade).
- The final recorded battery was launched through `tools/box.sh` with
  `ZE_AFFINITY_MASK=1` and all required settings (measurement condition,
  iterate grade): `SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file`,
  `IGC_VISAOptions=-perfmodel`, `IGC_VectorAliasBBThreshold=10000`, and
  `IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread`.
- Inputs were incompressible xorshift bf16 words with exponent constrained to
  values in `[-0.5, 0.5)`; output was fp32 (method, source-defined).
- Each shape has a discarded M=512 ramp-control configuration immediately
  before its recorded cells. Each recorded cell is 8 replays, first 3
  discarded, median of the final 5, 4 enqueues per replay, with
  `q.wait_and_throw()` after every replay (method, source-defined).
- Each cell sampled 4,096 fixed-xorshift output positions against a host-double
  recomputation; tolerance was `5e-3 × max|ref| + 1e-4` (structural check,
  source-defined).

### Discarded ramp controls

All values in this table are measured, iterate grade; they are explicitly not
record rows.

| shape | M | ms | TFLOP/s | can_implement | workspace |
|---|---:|---:|---:|---:|---:|
| qkv‖z | 512 | 18.392 | 4.67 | 0 | 0 B |
| out/o_proj | 512 | 2.879 | 11.19 | 0 | 0 B |
| gate‖up | 512 | 45.292 | 4.03 | 0 | 0 B |
| down | 512 | 8.344 | 10.94 | 0 | 0 B |
| q‖k‖v | 512 | 16.574 | 4.53 | 0 | 0 B |
| lm_head | 512 | 340.567 | 3.82 | 0 | 0 B |

## P2 results - final recorded battery

All values in this table are measured, iterate grade under the conditions
above. `can_implement = 0` is `cutlass::Status::kSuccess`.

| shape | K×N | M | can_implement | ms | TFLOP/s | % of 90 | max abs err | tol | bitwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qkv‖z | 5120×16384 | 512 | 0 | 18.592 | 4.62 | 5.1% | 0 | 0.0527694 | identical |
| qkv‖z | 5120×16384 | 1024 | 0 | 41.369 | 4.15 | 4.6% | 0 | 0.0544298 | identical |
| qkv‖z | 5120×16384 | 2048 | 0 | 88.161 | 3.90 | 4.3% | 0 | 0.047184 | identical |
| qkv‖z | 5120×16384 | 4096 | 0 | 182.996 | 3.76 | 4.2% | 0 | 0.0521108 | identical |
| out/o_proj | 6144×5120 | 512 | 0 | 2.888 | 11.15 | 12.4% | 0 | 0.05478 | identical |
| out/o_proj | 6144×5120 | 1024 | 0 | 14.341 | 4.49 | 5.0% | 0 | 0.0594207 | identical |
| out/o_proj | 6144×5120 | 2048 | 0 | 29.884 | 4.31 | 4.8% | 0 | 0.0507913 | identical |
| out/o_proj | 6144×5120 | 4096 | 0 | 64.544 | 3.99 | 4.4% | 0 | 0.0478967 | identical |
| gate‖up | 5120×34816 | 512 | 0 | 45.429 | 4.02 | 4.5% | 0 | 0.0458065 | identical |
| gate‖up | 5120×34816 | 1024 | 0 | 94.119 | 3.88 | 4.3% | 0 | 0.049535 | identical |
| gate‖up | 5120×34816 | 2048 | 0 | 196.033 | 3.72 | 4.1% | 0 | 0.0477594 | identical |
| gate‖up | 5120×34816 | 4096 | 0 | 400.331 | 3.65 | 4.1% | 0 | 0.0503374 | identical |
| down | 17408×5120 | 512 | 0 | 8.366 | 10.91 | 12.1% | 5.72205e-06 | 0.095303 | identical |
| down | 17408×5120 | 1024 | 0 | 41.024 | 4.45 | 4.9% | 1.81198e-05 | 0.0911651 | identical |
| down | 17408×5120 | 2048 | 0 | 91.506 | 3.99 | 4.4% | 6.67572e-06 | 0.084736 | identical |
| down | 17408×5120 | 4096 | 0 | 194.086 | 3.76 | 4.2% | 2.76566e-05 | 0.0981466 | identical |
| q‖k‖v | 5120×14336 | 512 | 0 | 16.660 | 4.51 | 5.0% | 0 | 0.0498701 | identical |
| q‖k‖v | 5120×14336 | 1024 | 0 | 34.910 | 4.31 | 4.8% | 0 | 0.0536327 | identical |
| q‖k‖v | 5120×14336 | 2048 | 0 | 77.013 | 3.90 | 4.3% | 0 | 0.0478585 | identical |
| q‖k‖v | 5120×14336 | 4096 | 0 | 160.508 | 3.75 | 4.2% | 0 | 0.0462463 | identical |
| lm_head | 5120×248320 | 512 | 0 | 340.645 | 3.82 | 4.2% | 0 | 0.0491151 | identical |
| lm_head | 5120×248320 | 1024 | 0 | 705.990 | 3.69 | 4.1% | 0 | 0.0484707 | identical |
| lm_head | 5120×248320 | 2048 | 0 | 1437.249 | 3.62 | 4.0% | 0 | 0.0488342 | identical |
| lm_head | 5120×248320 | 4096 | 0 | 2899.362 | 3.59 | 4.0% | 0 | 0.0519366 | identical |

### TFLOP/s matrix

All values are measured, iterate grade; this is the exact matrix emitted by
the final recorded run.

| shape | M=512 | M=1024 | M=2048 | M=4096 |
|---|---:|---:|---:|---:|
| qkv‖z | 4.62 | 4.15 | 3.90 | 3.76 |
| out/o_proj | 11.15 | 4.49 | 4.31 | 3.99 |
| gate‖up | 4.02 | 3.88 | 3.72 | 3.65 |
| down | 10.91 | 4.45 | 3.99 | 3.76 |
| q‖k‖v | 4.51 | 4.31 | 3.90 | 3.75 |
| lm_head | 3.82 | 3.69 | 3.62 | 3.59 |

### Determinism and workspace

For every cell, the probe ran the configured GEMM twice into separate fp32
device outputs and compared every output byte in chunks: **all 24 pairs were
bitwise equal** (measured, iterate grade). The selected `GemmUniversal`
configuration omits `TileScheduler_`, uses the data-parallel
`PersistentScheduler`, and instantiates no split-K atomic accumulation
(source-verified configuration). `Gemm::get_workspace_size(arguments)` was
**0 B for every cell** (measured, iterate grade).

### Measured XMX bf16 throughput line

**Measured XMX bf16 throughput = 11.15 TFLOP/s** (measured, iterate grade,
`ZE_AFFINITY_MASK=1`, sycl-tla
`91e5bd735517d8e79591b41e0d0cd37a7bacdca7-content-verified`, out/o_proj
6144×5120 at M=512). The derived 183.5 TFLOPS bf16 peak is vendor int8
367 TOPS ÷ 2 (derived, external-source basis); the achieved rate is **6.1%**
(derived, iterate-grade inputs) of that peak. These are different quantities:
183.5 is a vendor-derived peak, while 11.15 is a measured achieved rate at one
production shape and M.

## Prediction versus measurement

| pre-registered claim | outcome |
|---|---|
| Gate‖up at M=4096 is ≥90 TFLOP/s | **Miss.** Measured 3.65 TFLOP/s (measured, iterate grade), 86.35 TFLOP/s below the prediction (derived, iterate-grade inputs), or 4.1% of it (derived, iterate-grade inputs). |
| Rate rises monotonically with M for every shape | **Miss.** Each final matrix row decreases over the measured M values (measured, iterate grade). |
| Out/o_proj is the worst cell at every M | **Miss.** It is the best measured cell at every M: 11.15, 4.49, 4.31, and 3.99 TFLOP/s respectively (measured, iterate grade). |
| Two runs of a cell are bitwise identical | **Hit.** Every cell's two distinct output buffers compared identical (measured, iterate grade). |
| Workspace is 0 B | **Hit.** Every cell reported 0 B (measured, iterate grade). |

## Post-measurement analysis

This section is deliberately downstream of the measurements above; neither
reference below was used as a probe input.

The pre-registered GEMM total for C=4096 is 199.3 TFLOP (derived, from the
immutable pre-registration). At the measured gate‖up M=4096 rate of 3.65
TFLOP/s (measured, iterate grade), that GEMM term alone takes **54.60 s**
(derived, iterate-grade input) - 26.30× the 2.076 s implied by vLLM's 1973
t/s pp4096 result (external measured reference), or 2,530% above it (derived,
external/iterate-grade inputs). This reruns the pre-registered arithmetic with
the measurement; it does not alter the prediction.

The stock configuration's best observed rate is 11.15 TFLOP/s (measured,
iterate grade), so it lands **below** the 96 TFLOP/s GEMM-only threshold
(derived from the vLLM reference): it reaches 11.6% of that threshold
(derived, iterate-grade input). It also lands **below** the reported
119-124 TFLOP/s end-to-end range from the same Vishva007 model and
llama-benchy command on a B70 (external reported reference): 11.15 TFLOP/s is
9.4% of 119 and 9.0% of 124 (derived, external/iterate-grade inputs). The
unexplained remainder is the gap between this stock configuration's measured
3.65-11.15 TFLOP/s range and those reference figures; P2 did not measure its
cause, and therefore does not propose a fix.

## Build note

The first probe-source compilation error was recorded verbatim and fixed in
the probe source, without changing compiler flags:

```text
/home/user/b70-inference-server/tools/probe/probe_prefill_gemm.cc:106:17: error: expected '>'
  106 |     Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
      |                 ^
/home/user/b70-inference-server/tools/probe/probe_prefill_gemm.cc:106:11: note: to match this '<'
  106 |     Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
      |           ^
```

The local probe `Shape` table shadowed `cute::Shape`; qualifying the CuTe type
fixed the probe. The final serial `-j1` SYCL target build and final battery
both completed successfully (measured build/run outcome, iterate grade).
