# Spec 2 - Plan 6f: the weight format pivots to MXFP4A16

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land ruling **A20 + its addendum** end to end - a `compressed-tensors`
**MXFP4A16** reader in the loader, a cross-language e2m1/e8m0 dequant oracle, an
`e2m1` unpack path in decode's `gemv.cl`, and `gemm_mxfp4` on the prefill path -
so that the **210.116 ms/chunk bf16 dequant scratch is removed from the critical
path** and the engine runs the format the third party's 2513 t/s row runs. The
int4-g64 path stays fully supported: **the checkpoint decides**, at every layer,
by content and never by a config label (docs/02's rule).

**Architecture:** 4.25 bits/weight in both formats, so **decode's byte ceiling
does not move** and every decode claim is a re-measurement, not a projection.
What moves is prefill: MXFP4's block scales are what `sycl-tla`'s mixed-dtype
mainloop upconverts **in registers**, so the `[K][N]` bf16 materialisation
disappears. Two device layouts mirror the two int4 layouts one-for-one -
**layout 2 ≡ layout 0's geometry, layout 3 ≡ layout 1's 544-byte tile** - so the
per-shape `S`/layout table Task 4 tuned (−1.20 ms/token) changes only its
*layout column*, and the retune is re-measured against a preserved lane mapping
rather than against a new kernel.

**Tech Stack:** OpenCL C 3.0 + ocloc AOT (decode + the prefill relayout),
`icpx 2026.1.1` + `sycl-tla` at pin `91e5bd7` (prefill GEMM), C++17 host /
C++20 in the icpx sub-project, pure Level Zero runtime, ctest on the box
(`tools/box.sh`), Python 3 + torch on the box for the oracle.

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - read
**§3.0 (the superseding ruling)**, §3.1/§3.2 *as the superseded record*, §3.5,
§3.6, §6, §7, §8, §9 first; they govern every task.

**Interface contract:**
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md` - binding.
**A20 and its addendum are this plan's charter, and this plan finds one of A20's
premises false against the pin** (see *What this plan verified in `sycl-tla`*
below). That finding is filed as ruling request **R1** and **no task writes code
against the overturned premise**.

**Sibling plans:** **6a** (Stage 0: `cmake/prefill.cmake`, the icpx sub-project,
`runtime::prefill::Context`, `pf_dequant_tile`), **6b** (L1: `PrefillScratch`,
`Engine::prefill`, `--pp`/`--pp-chunk`, `gdn_chunk`, the three prefill tests,
`oracle-out-long`), **6c** (L2: `gemm_bf16`, `dequant_to_bf16`, the swap),
**6d-composed** (L3: `gemm_bf16_batched`, composed attention), **6e** (the §7
gate). This plan **retires `dequant_to_bf16` from the critical path** and frees
`PrefillScratch::dequant`; it does not touch attention, the GDN chunk, the
buffers' chunk-scaled rows, or the CLI.

---

## Global Constraints

**Spec §8, verbatim (binding):**

- Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain
  (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or
  `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the
  operator's and stays read-only; `sycl-tla` built once as a static
  dependency, version pinned in the build.
- Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0;
  record rows only on a provably idle box (zero DRM holders).
- Every number labelled measured vs derived vs estimated/external, with
  grade and conditions; no two values for one quantity without a
  reconciliation sentence.
- Nothing pushed; tags local; work on the branch the operator names.
- Design must not preclude prefix caching: chunk boundaries at multiples
  of 1024 align with the queued block-snapshot scheme
  (`docs/04-architecture.md`, "Follow-on"); no work on it here.

**Carried from spec §6 and this plan's own discipline (also binding):**

- **The Mac never compiles.** Every build and every test goes through
  `tools/box.sh build` / `tools/box.sh test [regex]` with `JOBS=44`. Nothing in
  this plan is verified by reading it. **This plan's author did not run one
  command on the box** - another agent held it; every number below is quoted
  from a committed record and re-measured by the task that needs it.
- **`-Wall -Wextra -Werror`, C++17** (host) / **C++20** (the icpx sub-project).
  CUTLASS/cute headers arrive with `-isystem`
  (`src/sycl/CMakeLists.txt`'s `target_include_directories(sycl_tla SYSTEM …)`),
  never `-I`.
- **No fp atomics anywhere on the prefill path.** The `sycl-tla` configuration
  stays pinned to `cutlass::gemm::PersistentScheduler`, explicitly, in the
  MXFP4 chain exactly as in `xe_gemm_config.h`'s `Chain<>`.
- **`-cl-denorms-are-zero` forbidden** (`cmake/ocloc.cmake:41` fatals). This now
  has a **second, arithmetic reason**: an e8m0 exponent byte of `0` means
  `2^-127`, which is a **subnormal** fp32 (min normal is `2^-126`), and
  flushing it would silently zero a whole block of 32 weights. Task 3 states
  this in the kernel comment.
- **Decode's *walk* is untouched.** `src/runtime/capture.cc`'s traversal, the
  **774 launches / 19 modules** invariants and `tests/runtime/replay_determinism_test.cc`
  / `profile_capture_test.cc` do not move: a `gemv_M…_L2`/`_L3` variant replaces
  a `gemv_M…_L0`/`_L1` variant one-for-one, so the module count is arithmetic,
  not luck (Task 3 Step 9 re-derives it).
- **GPU work under `ZE_AFFINITY_MASK=1`** whenever a container holds card 0.
  Never touch docker; never kill a process.
- **~15 GB free on the box root.** The MXFP4 checkpoint is ~14.6 GB and the
  `oracle-out-mxfp4` set ~1 GB. **Check `df -h ~` before Task 1 Step 1 and
  before Task 5 Step 2, and STOP and report on any ENOSPC** - do not delete
  another agent's artifacts to make room.
- **Pre-registered predictions before any timing.** The number is written into a
  **committed** file first, then the run happens (precedent `4009100`;
  `.superpowers/sdd/.gitignore` is `*`, so an uncommitted pre-registration is
  not one). This binds Tasks 3 and 4 absolutely.
- **One defect, one fix, one measurement.** A task that finds two problems
  fixes them in two commits with a measurement between.
- **Commits** end with
  `Claude-Session: `.

---

## What this plan verified in `sycl-tla` at the pin, and what it overturns

Read on the Mac checkout `~/PycharmProjects/sycl-tla` at
**`91e5bd735517d8e79591b41e0d0cd37a7bacdca7`** (ruling A11's pin; `git log -1`
confirms it), 2026-09-05. Every line reference below is at that revision and
**Task 4 Step 1 re-reads all six before a line of code is written.**

| # | fact | evidence |
|---|---|---|
| 1 | The block-scaled MXFP4 mainloop's `kSupportedElementA` is `{float, half_t, e5m2, e4m3, e2m1}` - **`bfloat16_t` is not in it** | `include/cutlass/gemm/collective/xe_mma_blockscaled_mxfp.hpp:138-143` |
| 2 | It **requires block scales on BOTH operands**: `static_assert(kScaleALeftmostUnitStride, "…requires scale A leftmost stride to be _1.")` and the same for B | same file `:176-179` |
| 3 | It multiplies with **`XE_BDPAS_TT`** (`bdpas`), and every `BDPAS` atom plus the `e2m1` `DPAS` atoms are declared inside `#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)` | `include/cute/arch/mma_xe.hpp:254`, `:264-282`; the `#else` branch at `:290` is labelled *"Skip int8 x int4 for CRI as the dpas is removed"* |
| 4 | The only examples that instantiate it are `50_xe35_block_scaled_gemm` / `51_…`, gated `if (DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)`, and `SYCL_TARGET_INTEL_GPU_CRI` is set **only** for the `intel_gpu_cri`/`cri` targets | `examples/50_xe35_block_scaled_gemm/CMakeLists.txt:29`; `CMakeLists.txt:187-193`; `cmake/FindDPCPP.cmake:99-101` |
| 5 | Our build defines `SYCL_INTEL_TARGET` **with no value**, i.e. `1` (`target_compile_definitions(sycl_tla INTERFACE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)`), and the device is `bmg-g31` | `src/sycl/CMakeLists.txt:115`, `cmake/ocloc.cmake:4` |
| 6 | The **in-register e2m1 → bf16 upconvert exists and is NOT Xe3.5-gated**: `Xe_Reorder<ReorderKind::{UU,UV,VV}, float_e2m1_t, bfloat16_t>` is plain vISA (`shl` / `asr` / `and` / `mul 0x7E800000`) under `CUTE_ARCH_REORDER_XE_ENABLED` only | `include/cute/arch/reorder_xe.hpp:1512-1625` (the `CUTE_XE_REORDER_E2M1_BF16_SEQ` macro at `:1513-1527`) |

**Consequence - A20's "inherit sycl-tla's native blockscaled MXFP4 mainloop,
zero custom transform" is FALSE for BMG at this pin, and is filed as ruling
request R1.** `xe_mma_blockscaled_mxfp.hpp` is an **Xe3.5 (CRI)** mainloop for
**block-scaled × block-scaled** operands; it cannot take a bf16 A, and its MMA
atom is not declared for our target. `docs/01-hardware.md:149` already carried
the warning ("the block-scaled MXFP4 examples in sycl-tla are `51_xe35_*`"); the
ruling was written past it.

**What IS available on BMG**, and what this plan therefore builds on:

- **`MainloopIntelXeXMX16MixedPrecision`** (`include/cutlass/gemm/collective/xe_mma_mixed_input.hpp`,
  example `02_bmg_gemm_mixed_dtype`) - A wide × B narrow with **group-wise
  scales**, `group_size` a runtime argument, B addressed as
  `make_shape(N, K, L)` with `subbyte_iterator<const ElementB>` for 4-bit
  (`:311-317, :328-330`) and the scale as
  `make_shape(N, scale_k, L)` with `NonVoidStrideScale = Stride<_1, int64_t, int64_t>`
  (`:337-343`). This is the chain `docs/06-prior-art.md:12` and
  `docs/04-architecture.md:199` already name as "the phase-1 prefill workhorse".
- **The gap, precisely.** Its 4-bit `transform_quant` computes
  `is_quantization = numeric_limits<SrcType>::is_integer ^ numeric_limits<DstType>::is_integer`
  and applies the scale only when that is true (`:446-448, :504-513`). With
  `SrcType = float_e2m1_t` and `DstType = bfloat16_t` **both are non-integer**,
  so the scale is dropped and the nibble is *numerically cast* as an integer.
  **No configuration of the stock mainloop computes MXFP4.** It is not
  reachable by choosing types; the transform is a private member, so it is not
  reachable by specialising a trait either.
- **Therefore Task 4 writes exactly one thing**: a `CollectiveMma` partial
  specialisation on our own dispatch-policy tag whose B path is
  `copy → Xe_Reorder<…, float_e2m1_t, bfloat16_t> → × ldexp(1, E−127)`, i.e.
  Intel's own upconvert sequence with the e8m0 scale applied where the stock
  mainloop applies its own. This is progress.md's controller reading, option
  (i) - *"write only the gap"* - and it is now costed rather than assumed.

**Layout facts read from the same file, which fix what the loader must produce
for prefill:**

- **B (the packed weights) must be K-contiguous per output column**, i.e.
  `[N][K/2]` uint8 - `TagToStrideB_t<ColumnMajor>` over the `(N, K, L)` mode
  order, exactly what `02_bmg_gemm_f16_u4_f16.cpp:581` selects and what
  `weight_packed[N][K/2]` **already is on disk**.
- **The scales must be N-contiguous**, i.e. `[K/32][N]` uint8 - the leftmost
  `_1` of `Stride<_1, int64_t, int64_t>` is over the `N` mode of
  `make_shape(N, scale_k, L)`. The checkpoint ships `weight_scale[N][K/32]`, so
  **the scale plane is transposed relative to disk** and somebody must transpose
  it. This plan's loader does, once, at load (Task 1) - it is 1/16 of the
  nibble bytes and the loader already repacks every tensor.

---

## The numbers this plan is built on

Every row says its kind. Nothing here is measured *by this plan at authoring* -
Tasks 3 and 4 produce this plan's own measurements.

| quantity | value | kind | source |
|---|---|---|---|
| vLLM bar | 1973 t/s pp4096, HTTP-inclusive | measured (external) | `docs/BENCHMARKS.md:122` |
| third party, MXFP4 | `Qwen3.8-27B-MXFP4-GRIMOIRE \| pp4096 \| 2513.22 ± 68.47` | measured (external, screenshot) | progress.md, "CORRECTION - the third-party datapoint is NOT the same model" |
| decode gate rows | **32.22 t/s** (RTN, int4 `lm_head`) / **29.33** (Vishva) at `2a7df0b` | measured, near-idle, median of 3 | `docs/BENCHMARKS.md` §"The spec-1.7 gate rows" |
| RTN read/token | **13,672,613,888 B = 13.673 GB** | measured | `docs/13-loader.md` §"The report, both checkpoints" |
| RTN resident total | 16,219,604,992 B (13.673 read/token + 2,542,796,800 embed + 4,194,304 rope) | measured | `docs/13-loader.md` |
| the 256 per-layer quantised matrices, resident | **12,923,699,200 B** (12,163,481,600 nibbles + 760,217,600 scales) | measured | `docs/13-loader.md` §"Memory and the `W` cross-check" |
| Σ K·N over those 256 matrices | **24,326,963,200** | derived (this plan) | 48·382,730,240 + 16·372,244,480; cross-checks the row above at K·N/2 exactly |
| device memory | 34,241,150,976 B = **31.891 GiB** | measured | `docs/01-hardware.md:14` |
| measured DRAM bandwidth | **590 GB/s** | measured | `docs/05-perf-model.md` |
| C=2048 chunk, every term measured | **1126.958-1132.458 ms → 1808.5-1817.3 t/s** | measured composition | progress.md, "RECOMPOSED CEILING" |
| ⤷ dequant term | **210.116 ms/chunk (18.6%)** | measured | `docs/probe-dequant-overlap-2026-09-05.md` |
| ⤷ GEMM term (60.2%) | **678.43-681.74 ms** | derived from the two rows above | 0.602 × the chunk |
| ⤷ everything else | **238.412-240.602 ms** | derived | chunk − GEMM − dequant |
| dequant achieved bandwidth | 61,575,823,360 B / 210.116 ms = **293.06 GB/s** (49.7% of device) | derived from measured | plan 6c Task 2 Step 1's traffic ÷ the measured ms |
| P2, gate‖up M=2048, bf16 | **150.19 TFLOP/s** | measured, iterate | `docs/probe-prefill-gemm-2026-09-04.md` (corrected 256-GRF matrix) |
| P2 peak | 164.21 TFLOP/s (down, M=2048) = 89.5% of the derived 183.5 XMX peak | measured, iterate | same |
| open risk, unchased | a production-context GEMM control read **131.47 TFLOP/s (−12.5%)** at the same cell | measured, iterate | progress.md, "Open risk, flagged not asserted" |
| production GEMV cells (int4, with `deq`) | out/o_proj **561**, q‖k‖v **579**, qkv‖z **562**, gate‖up **561**, down **574**, lm_head **576** GB/s | measured | `docs/probe-gemv-loads-2026-08-26.md` |
| the retune this risks | **−1.20 ms/token** composed (0.909 retune + 0.216 `deq` marginal − 0.009 `prep`) | estimated from measured µs ratios | same file, "Estimated value" |
| oracle golden set cost, 3 prompts × 32 tokens | **18 min 16 s** idle / **25 min 14 s** under load | measured | `docs/14-golden-gate.md` |
| oracle long set (≥2048 ids) | **hours-class** | estimated (spec §6.2 states it) | spec §6.2 |
| `PrefillScratch::bytes()` today | **2,263,990,168** | measured (pinned) | `tests/runtime/buffers_test.cc:118` |
| `PrefillScratch::dequant` | **356,515,840 B** | measured (pinned) | `src/runtime/buffers.cc:149-154` |
| box toolchain | `icpx 2026.1.1`, `ocloc 26.27.39122.14`, 44 cores, 121 GB RAM | measured (2026-09-04) | plan 6c |

### Reconciliation - 4.25 bits/weight, and why the byte table does not move

Both formats spend **exactly the same bytes**, and the arithmetic is worth
writing out because the whole decode pre-registration rests on it:

| | nibbles | scales | total | bits/weight |
|---|---|---|---|---|
| int4 g64 sym (GPTQ/AutoRound) | `K·N/2` B | one **f16** per 64 weights = `2·K·N/64` = **`K·N/32` B** | `K·N·17/32` | 4 + 16/64 = **4.25** |
| MXFP4A16 (e2m1 + e8m0 block 32) | `K·N/2` B | one **e8m0 byte** per 32 weights = `1·K·N/32` = **`K·N/32` B** | `K·N·17/32` | 4 + 8/32 = **4.25** |

`0.25 = 16/64 = 8/32`. Not "about the same" - **the same byte count, per
tensor, exactly**, which is why `docs/16-know-how.md` §5's line ("MXFP4 buys
nothing for bandwidth-bound decode: 4.25 bits/weight, identical to GPTQ g64
(4 + 16/64 = 4 + 8/32)") is quoted rather than re-derived. Consequences:

- `LoadReport::read_per_token` for an MXFP4A16 checkpoint with a quantised
  `lm_head` must print **13,672,613,888 B**, byte-identical to the RTN row.
  Task 1 asserts that number, not a band.
- The loader's `W` cross-check (doc-03's `W` + itemised pad/widen/lm_head, ±2%)
  passes **unchanged**, because both sides move by zero.
- Decode is bandwidth-bound at 74.7% MBU on 13.673 GB; the format cannot change
  the ceiling, only the ALU work under it. Hence Task 3's pre-registration is
  "within ±3% of the int4 rows", not "faster".

### Reconciliation - three ceilings, and what separates them

Spec §3.0 projects "**~2190-2230 t/s (derived)**" for the C=2048 chunk with the
dequant removed. This plan derives **2035-2234 t/s**, a range rather than a
point, because the dequant term is not replaced by *nothing* - it is replaced by
whatever gets the packed weights from the layout decode wants into the layout
the mainloop wants. Same quantity, three routes, one arithmetic:

```
chunk(r, X) = 238.412…240.602 ms  +  (678.43…681.74 ms) / r  +  X
              └ everything else ┘    └ the GEMM at ratio r ┘   └ the bridge ┘
t/s = 2048000 / chunk_ms          r = (MXFP4 TFLOP/s) / (bf16 150.19)
```

| bridge `X` | route | chunk at r=1 | **t/s at r=1** | **break-even `r` for 1973** |
|---|---|---|---|---|
| **0 ms** | **R-b** - the loader writes a second, prefill-native weight copy (+12,923,699,200 B resident) | 916.84-922.34 ms | **2220.5-2233.8** | **0.855** (≈ 128.4 TFLOP/s) |
| **41.66 ms** | **R-a** - a per-chunk relayout kernel, priced at the 590 GB/s roofline | 958.50-964.00 ms | **2124.5-2136.7** | **0.902** (≈ 135.5) |
| **83.87 ms** | **R-a** - the same, priced at the dequant's *measured* 293.06 GB/s | 1000.71-1006.21 ms | **2035.4-2046.5** | **0.955** (≈ 143.5) |

All three clear vLLM's 1973 **at r = 1**, and this plan's headline is that
range: **2035-2234 t/s derived, 103-113% of 1973**, against the int4 path's
measured 1808-1817 (92%). But the break-even column is the decision-grade part:
**A20's pre-registered acceptance of `0.85 × 150.19` is the break-even of the
*zero-bridge* route only.** Passing at exactly 0.85 with a relayout in the path
leaves the spec's bar missed. Task 4 pre-registers both numbers and Task 4
Step 2 is the ruling request that chooses the route.

**Relayout traffic, derived.** Layout-2 shapes move nibbles in and out
(`K·N` B); the one layout-3 shape (`qkv‖z`) reads a 544-B tile and writes the
packed pair (`K·N·17/16` B). Per GDN layer 387,973,120 B, per FA layer
372,244,480 B; `48 × 387,973,120 + 16 × 372,244,480 =` **24,578,621,440 B =
24.579 GB per chunk**, C-independent. ÷ 590e9 = **41.66 ms**; ÷ 293.06e9 =
**83.87 ms**. The gap between the two is not noise: the relayout is a **4-bit
transpose** (N-major in, K-major out) and the measured 293.06 GB/s came from a
kernel that transposed nothing. **Task 4 Step 7 measures it; until then the
83.87 ms figure is the one this plan composes with.**

### Reconciliation - the open GEMM risk rides on all of it

progress.md flags an unchased −12.5% (150.19 → 131.47 TFLOP/s) between P2's
harness and a production-context control. If that is real for the production
path, the GEMM term is 777 ms rather than 680 and every row above falls: at
r = 1 with the 83.87 ms bridge the ceiling is `240.602 + 777 + 83.87 = 1101.5 ms
→ 1859.4 t/s` - **below 1973**. This plan does not chase it (the no-tuning
rule), but Task 4 Step 7's measurement is taken **in the production path**, not
in a probe harness, precisely so the question is answered as a side effect
rather than deferred again.

---

## Interfaces: Consumes

**From the existing tree (verified by reading, 2026-09-05):**

- `loader::QuantConfig::parse` (`src/loader/quant.cc:34-118`) - reads
  `quantization_config` and **unconditionally** does
  `q.bits = uint32_t(qcv->at("bits").num())` at `:38`. A `compressed-tensors`
  config has no top-level `bits`, so **the format branch must precede line 38**
  or the loader throws on a valid checkpoint.
- `loader::LinearSrc::classify` (`:120-157`) - two branches, `.qweight`+`.scales`
  → `WKind::Int4`, `.weight` → `WKind::Bf16`, else throw. Task 1 adds a third.
- `loader::assert_quant_invariants` (`:159-200`) - scans `.qzeros` (must be
  `0x77777777`), `.g_idx` (identity `k/64`) and `.scales` (f16 finite;
  subnormals **counted, not rejected**). Task 1 adds the `.weight_scale` scan
  in the same shape.
- `loader::DeviceWeight` (`src/loader/loader.h:19-26`) - `l0::Mem mem`,
  `std::unique_ptr<l0::Mem> scales` ("non-null iff layout 0"),
  `model::GemvShape shape{K,N,S,layout}`, `model::WeightKind kind`.
- `loader::LoadReport` (`:37-56`) - the seven byte buckets, `total()`,
  `read_per_token`, and the ±2% `W` cross-check at `loader.cc:619-623`.
- `loader::load_linear` (`src/loader/loader.cc:254-375`) - the fusion/column-map
  machinery (`common::cols_concat`, `cols_interleave16`), the staging bound
  checks, the per-bucket accounting, and
  `if (fl.kind == …Int4 && sh.layout > 1) throw` at `:265-266`.
- `common::repack_int4_layout1_cols` / `repack_int4_layout0_cols`
  (`src/common/repack.h:67-103`) and `common::Int4Gptq` (`src/common/int4.h`) -
  the shapes Task 1's MXFP4 counterparts mirror line for line.
- `model::GemvShape{K,N,S,layout}`, `model::WeightKind{Int4,Bf16}`,
  `model::Qwen35::{linear,lm_head,shape,layers}` and the production table
  (`src/model/qwen35.cc:36-77`): QkvZ `{5120,16384,1,1}`, AB `{5120,128,1,0}`
  bf16, OutProj `{6144,5120,4,0}`, GateUp `{5120,34816,8,0}`,
  Down `{17408,5120,4,0}`, Qkv `{5120,14336,2,0}`, OProj `{6144,5120,4,0}`,
  LmHead bf16 `{5120,248320,1,0}` / int4 `{5120,248320,1,1}`.
- `src/kernels/gemv.cl` - `dot8()` (`:46-76`), the two `#if LAYOUT` blocks
  (`:97-117`), `TILE_U32 136` (`:41`), `GROUP 64`, the grid `(N/64) × S` of
  4 subgroups × 16 lanes, and the epilogue `out[(s*M + m)*N + n]`.
- `src/kernels/CMakeLists.txt:6-51` - `add_gemv_variant(M K N S L)`, its Task-4
  production `extra_defs` block (`GEMV_BLOCK2D=8` / `GEMV_DEQ_SHIFT=1` per
  cell), the L∈{0,1} × S∈{1,2,4,8,16} probe matrix, and
  `gemv_control_M1_K6144_N5120_S4_L0`.
- `kernels::gemv_variant(M,K,N,S,L)` (`src/kernels/kernels.h:13-16`) - **already
  carries `L`, so layouts 2 and 3 need no new namer.**
- `src/runtime/prefill/gemm.h` - `GemmDims{M,K,N}`, `gemm_bf16`,
  `GemmBatch`, `gemm_bf16_batched`, `gemm_bf16_supports_transb`,
  `gemm_bf16_batched_grid`. SYCL-free by construction.
- `src/sycl/xe_gemm_config.h` - `TileShape = Shape<_256,_256,_32>`,
  `TiledMma` over `MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>` with
  `Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>`, `MainloopXeL1Staged<2>`,
  `IntelXeGeneric` epilogue with **`ElementC = void`** and its
  `static_assert`, `PersistentScheduler`, and the
  `template <class LayoutB_> struct Chain` that Task 4's MXFP4 chain sits
  beside.
- `src/sycl/gemm_batched.cc` - `check_operand` (64-byte base, `ld % 8 == 0`),
  `make_args` with `KernelHardwareInfo hw_info{}`, the `can_implement` /
  `get_workspace_size == 0` requires, and
  `sycl::queue& q = runtime::prefill::sycl_queue(cx); op.initialize(args, nullptr, &q); op.run(&q);`.
- `src/sycl/CMakeLists.txt:123` - `add_library(b70_prefill SHARED context.cc gemm_batched.cc)`;
  `:97-102` the `SYSTEM INTERFACE` include roots; `:115` the two `-D`s; `:44`
  the three `-spirv-ext` entries; `:45-48` the
  `-Xsycl-target-backend=spir64_gen "-options -cl-intel-256-GRF-per-thread"`
  that P2's 256-GRF lesson added (**a runtime IGC env var does not configure an
  AOT compile** - any new SYCL TU inherits this automatically because it is
  target-wide).
- `tools/oracle/dequant.py::dequant_gptq` and its fixture writer;
  `tests/loader/dequant_fixture_test.cc`; `tools/oracle/dump.py` (builds the
  reference bf16 state dict **from `dequant.py`**, which is what makes the
  oracle and the engine start from identical weights);
  `tools/oracle/golden.sh` (`OUT_DIR` / `ORACLE_SNAP` / `ORACLE_THREADS`, one
  golden set per checkpoint, three prompts serially).
- `tests/golden/golden_gate_test.cc` - `argv[1]` golden dir, `argv[2]` prompt
  ids, `argv[3]` snapshot; `kGen = 32`; the tie-aware three clauses; the
  `CHECK_EQ(eng.step().kernel_count, size_t(774))` structural assert; `return 77`
  = ctest SKIP when a golden file is absent.
- `tools/bench_decode.sh` (`--depth`, `--tg`, `--runs`, `--model`, `--no-build`;
  sha stamping incl. `-dirty`; median of 3 printed to stderr; the row
  `| b70-decode <sha> | <depth> | <tg> | <t/s> | <ms/token> |` on stdout).
- `tools/quantize_qwen38_mxfp4.sh` - `MODEL`, `OUT` (default
  `$HOME/models/qwen38-27b-mxfp4a16`), `DEVICE`, `WORKERS`, `LM_HEAD`; scheme
  `MXFP4A16`; the ignore list `["model.embed_tokens", "re:.*visual.*",
  "re:.*mtp.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*conv1d.*"]`; and a
  verifier that already refuses `input_activations` and checks
  `sN == N and sG * 32 == Khalf * 2`.

**From plan 6b/6c/6d (already landed or landing):** `PrefillScratch` and its
pinned `bytes() == 2,263,990,168`; `Engine::prefill`'s per-layer walk and its
six int4 linear call sites; `runtime::prefill::Context` and `Context::wait()`;
`pf_dequant_tile` and `dequant_to_bf16` (**kept, retired from the critical
path**); the three prefill tests; `--pp` / `--pp-chunk`.

## Interfaces: Produces

```cpp
// src/model/qwen35.h  - one new kind, and the layout id's meaning widened.
enum class WeightKind { Int4, Bf16, Mxfp4 };
// GemvShape::layout, by kind:
//   Int4 : 0 = GPTQ-native  w[K/8][N] u32 + f16 scales[K/64][N]
//          1 = 544-B tile, f16 scales inline
//   Mxfp4: 2 = plane   w[K/8][N] u32 (e2m1 nibbles) + e8m0 scales[K/32][N]
//          3 = 544-B tile, e8m0 scales inline
//   Bf16 : 0, a documented filler
```

```cpp
// src/loader/quant.h
enum class WKind { Int4, Bf16, Mxfp4 };

struct LinearSrc {
  WKind kind;
  uint32_t K = 0, N = 0;
  const uint32_t* qweight = nullptr;  // Int4 : [K/8][N]
  const uint16_t* scales  = nullptr;  // Int4 : [K/64][N] f16
  const uint16_t* weight  = nullptr;  // Bf16 : [N][K] row-major
  const uint8_t*  packed  = nullptr;  // Mxfp4: [N][K/2], low nibble = even k
  const uint8_t*  e8m0    = nullptr;  // Mxfp4: [N][K/32]
  std::string name;
  static LinearSrc classify(const SafetensorsSet& set, const std::string& prefix);
};

struct QuantConfig {
  enum class Scheme { Int4G64, Mxfp4A16 };
  Scheme scheme = Scheme::Int4G64;
  uint32_t bits = 0, group_size = 0;
  bool sym = false, desc_act = true, desc_act_declared = false;
  std::string quant_method, packing_format, ct_format;   // ct_format: recorded, never gated on
  size_t dynamic_rule_count = 0, extra_excluded = 0, extra_quantised = 0;
  size_t ct_ignore_rules = 0, ct_targets = 0;            // compressed-tensors' two lists
  static QuantConfig parse(const common::json::Value& config_json);
};

struct QuantScan {
  size_t subnormal_scales = 0;   // int4: f16 subnormals;  mxfp4: e8m0 byte == 0 (2^-127)
  size_t g_idx_tensors = 0;
  size_t e8m0_tensors = 0;       // how many .weight_scale planes were scanned
};
```

```cpp
// src/common/mxfp4.h  - the meaning of the bits, mirroring common/int4.h.
namespace common {
// The e2m1 code point, as a value. Bit 3 is the sign; bits 0-2 the magnitude.
// The table IS the format (OCP MX v1.0 §5.3.3) and is the oracle's table too.
inline constexpr float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
// e8m0: value = 2^(E-127). E = 0 is 2^-127 (a SUBNORMAL fp32); E = 255 is NaN
// and the loader refuses a checkpoint that contains one.
inline float e8m0_to_f32(uint8_t e) { return std::ldexp(1.0f, int(e) - 127); }

struct Mxfp4Ct {                       // "compressed-tensors", the on-disk shape
  uint32_t K = 0, N = 0;
  std::vector<uint8_t> packed;         // [N][K/2]
  std::vector<uint8_t> scale;          // [N][K/32]
  static constexpr uint32_t kBlock = 32;
  static constexpr uint32_t kTileU32 = 136;      // 544 B - the SAME tile as layout 1
  float at(uint32_t k, uint32_t n) const {
    const uint8_t b = packed[size_t(n) * (K / 2) + k / 2];
    const uint32_t q = (k & 1u) ? uint32_t(b >> 4) : uint32_t(b & 0x0Fu);
    return kE2M1[q] * e8m0_to_f32(scale[size_t(n) * (K / kBlock) + k / kBlock]);
  }
  size_t bytes() const { return packed.size() + scale.size(); }
  std::vector<uint32_t> planed(std::vector<uint8_t>& sc_out) const;  // layout 2
  std::vector<uint32_t> tiled() const;                               // layout 3
  static Mxfp4Ct random(uint32_t K, uint32_t N, uint32_t seed);
};
}  // namespace common
```

```cpp
// src/common/repack.h  - two functions, the exact shape of the int4 pair.
struct MxfpColSource { const uint8_t* packed; const uint8_t* scale;
                       uint32_t n; uint32_t k_half; uint32_t k_blocks; };
std::vector<MxfpColSource> mxfp_cols_concat(const std::vector<MxfpPart>&);
std::vector<MxfpColSource> mxfp_cols_interleave16(const MxfpPart& a, const MxfpPart& b);
// layout 2: w[K/8][N] u32 of e2m1 nibbles (8 consecutive k per word, low nibble
//           = lowest k) + sc[K/32][N] uint8 e8m0.
void repack_mxfp4_layout2_cols(uint32_t K, uint32_t N_total,
                               const std::vector<MxfpColSource>& cols,
                               uint32_t* w_out, uint8_t* sc_out);
// layout 3: per (n_tile of 16, k_group of 64) one 136-u32 tile: 128 u32 of
//           nibbles [k_octet j][lane l], then 16 ushorts, one per lane, holding
//           that column's two e8m0 bytes (k-block 2g in the low half).
void repack_mxfp4_layout3_cols(uint32_t K, uint32_t N_total,
                               const std::vector<MxfpColSource>& cols, uint32_t* out);
```

```cpp
// src/runtime/prefill/gemm.h  - ruling A20's interface, with R2's amendment.
struct MxfpWeight {
  const uint8_t* packed;   // e2m1, [N][K/2] uint8, K contiguous; row pitch ldb/2 bytes
  const uint8_t* scales;   // e8m0, [K/32][N] uint8, N contiguous; row pitch lds bytes
  uint32_t K, N;
  size_t ldb;              // element pitch of `packed`, in e2m1 ELEMENTS (production: K)
  size_t lds;              // row pitch of `scales`, in BYTES (production: N)
};
// C[M][N] fp32 (ldc = N) = A[M][K] bf16 (lda = K) · dequant(W).
// Group size is fixed at 32 by the format; there is no runtime group argument.
// Deterministic: PersistentScheduler, one work-group per 256x256 output tile,
// the whole K loop inside it, no atomics. Asynchronous; the caller syncs.
void gemm_mxfp4(Context& cx, GemmDims d, const uint16_t* A, const MxfpWeight& W, float* C);
// True iff the MXFP4 chain instantiated in this build (Task 4 Step 1's answer,
// reported rather than inferred), so a test cannot silently skip.
bool gemm_mxfp4_available();
```

- `src/sycl/xe_mma_mxfp4.hpp` - the `CollectiveMma` partial specialisation on
  `runtime::prefill::MainloopXeMxfp4Bf16<Stages>` (the gap; Task 4 Step 4).
- `src/sycl/gemm_mxfp4.cc` - the instantiation and the wrapper, added to
  `add_library(b70_prefill SHARED …)`.
- `src/kernels/prefill/mxfp4_pack.cl` - `pf_mxfp4_pack`, the layout-2/3 →
  `[N][K/2]` relayout (route R-a only; not built under R-b).
- `src/kernels/gemv.cl` - a `GEMV_FMT_MXFP4` compile-time path.
- `tools/oracle/dequant.py::dequant_mxfp4` + `--write-mxfp4-fixture`;
  `tools/oracle/dump.py`'s format branch.
- `tests/loader/mxfp4_test.cc`, `tests/prefill/gemm_mxfp4_test.cc`; extensions
  to `tests/loader/dequant_fixture_test.cc`, `tests/kernels/gemv_test.cc`,
  `tests/loader/load_checkpoint_test.cc`, `tests/runtime/buffers_test.cc`.
- `docs/12-kernels.md` (the MXFP4 GEMV section + the prefill mainloop section),
  `docs/13-loader.md` (the third spelling, the layouts, the byte table),
  `docs/16-know-how.md` §5 (the format's home, now measured),
  `docs/BENCHMARKS.md` (the MXFP4 decode record row and the pp row).

## Ruling requests to `interfaces.md`

None is applied silently; the controller rules, then the plan text and
`interfaces.md` move together. **R1 and R3 block Task 4; R2 blocks Task 4's
signature; R4 and R5 block Task 1.**

- **R1 - A20's prefill mechanism is not available on BMG at the pin.**
  `xe_mma_blockscaled_mxfp.hpp` is a **block-scaled × block-scaled** mainloop
  whose `kSupportedElementA` excludes `bfloat16_t` (`:138-143`), whose
  `static_assert`s require a scale tensor on **both** operands (`:176-179`), and
  whose MMA atom `XE_BDPAS_TT` - together with every `e2m1` `DPAS` atom - is
  declared only under `#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)`
  (`mma_xe.hpp:254`), i.e. **Xe3.5 / CRI**, which `bmg-g31` is not. Its examples
  are gated on `SYCL_TARGET_INTEL_GPU_CRI`. **Request:** amend A20's first
  consequence to read *"an MXFP4 mixed-dtype instantiation over
  `MainloopIntelXeXMX16MixedPrecision`'s structure with a project-owned B-side
  transform (`Xe_Reorder<…, float_e2m1_t, bfloat16_t>` + the e8m0 scale), because
  the stock mainloop's 4-bit `transform_quant` drops the scale whenever source
  and destination are both non-integer (`xe_mma_mixed_input.hpp:446-448`)"*, and
  record that "zero custom transform" was wrong. The plan carries the work
  either way; what the ruling changes is the **price** and therefore Task 4's
  schedule.
- **R2 - `MxfpWeight` gains `ldb`/`lds` and drops `layout`.** A20 sketches
  `{packed, scales, K, N, layout}`. The prefill operand has exactly **one**
  layout (the mainloop's `[N][K/2]` + `[K/32][N]`); what the GEMM actually needs
  from the caller is the two **pitches**, because a relayout scratch and a
  loader-owned copy may pitch differently. **Request:** the struct exactly as
  given under *Interfaces: Produces*.
- **R3 - the prefill-operand route.** Decode's lane mapping wants N-adjacent
  weights (16 lanes, 16 adjacent `n`); the mainloop wants K-contiguous ones.
  One of the three must happen and they are not equivalent:
  **R-a** a per-chunk relayout (**+41.66 ms roofline / +83.87 ms measured-scaled
  per chunk**, +94,699,520 B scratch, break-even `r` 0.902-0.955);
  **R-b** the loader writes a second, prefill-native copy of the 256 per-layer
  matrices (**+12,923,699,200 B resident**, 0 ms, break-even `r` **0.855**,
  headroom at max_len 16384 falls from 13.79 GiB to **1.75 GiB**);
  **R-c** a project-owned CuTe copy atom that reads our 544-B tile directly
  (0 ms, 0 bytes, but the DPAS B-fragment order that `partition_sg_fragment_B`
  defines is undocumented - **recorded and rejected, not attempted**).
  **Request: R-b, allocated lazily on the first `prefill()` exactly as 6b's
  ruling R7 allocates `PrefillScratch`**, so a decode-only engine's residency is
  byte-identical to today's and spec §6.5 stays checkable rather than argued -
  with **R-a implemented behind `B70_MXFP4_RELAYOUT=ON`** as the route for any
  future `max_len` that does not fit. The plan is written so that Task 4 lands
  R-b and Task 4 Step 8 lands R-a's kernel as the priced alternative; if the
  controller rules R-a primary, the two swap and nothing else changes.
- **R4 - `model::WeightKind` gains `Mxfp4`; `GemvShape::layout` gains 2 and 3.**
  `interfaces.md`'s ruling A5 describes `DeviceWeight` as
  `{mem, scales non-null iff layout 0, shape, kind}`. **Request:** amend to
  *"`scales` non-null iff layout 0 (f16 `[K/64][N]`) or layout 2 (e8m0
  `[K/32][N]`)"* and add the two layout ids with the meanings tabulated above.
- **R5 - the checkpoint path.** `tools/quantize_qwen38_mxfp4.sh` defaults
  `OUT=$HOME/models/qwen38-27b-mxfp4a16` and its verifier resolves
  `glob(root, "**", "config.json")[0]`, so the snapshot may or may not be
  nested one level. **Request:** the controller confirms the exact directory the
  producing run printed on its `VERIFIED B70-READY … -> {d}` line, and that
  directory becomes `B70_TEST_SNAPSHOT_MXFP4` in `tests/CMakeLists.txt`. Task 1
  Step 1 records it; no task hardcodes a guess.

---

### Task 1: the loader - a compressed-tensors MXFP4A16 reader, and two device layouts

*The third accepted spelling. Nothing here is trusted to decide what a tensor
**is** - that stays `LinearSrc::classify` on the shipped suffixes (docs/02's
rule). What the config parse does is **refuse a checkpoint whose arithmetic
differs from the one the kernels implement**, and it now has to read a third
vocabulary to do it.*

**Files:**
- Create: `src/common/mxfp4.h`, `tests/loader/mxfp4_test.cc`
- Modify: `src/loader/quant.{h,cc}`, `src/loader/loader.{h,cc}`,
  `src/common/repack.h`, `src/model/qwen35.{h,cc}`,
  `tests/loader/load_checkpoint_test.cc`, `tests/CMakeLists.txt`
- Test: `tests/loader/mxfp4_test.cc` (new), `load_checkpoint_test` (extended)

**Interfaces:**
- Consumes: `common::json::Value`, `SafetensorsSet`, `check_align`,
  `model::FusedLinear`/`GemvShape`, `common::{cols_concat,cols_interleave16}`'s
  shape, `LoadReport`'s buckets and the ±2% `W` cross-check.
- Produces: `loader::WKind::Mxfp4`, `QuantConfig::Scheme`,
  `model::WeightKind::Mxfp4`, layouts 2 and 3, `common::Mxfp4Ct`,
  `common::repack_mxfp4_layout{2,3}_cols`, and a `LoadReport` that prints
  **13,672,613,888 B read/token** on the MXFP4A16 artifact.

- [ ] **Step 1: find the artifact and record what it actually is, before any
  code.** Disk first: `tools/box.sh run 'df -h ~ | tail -1'` - **STOP and report
  if free space is under 15 GB.** Then, read-only:
  ```
  tools/box.sh run 'ls -d $HOME/models/qwen38-27b-mxfp4a16/ && \
    find $HOME/models/qwen38-27b-mxfp4a16 -name config.json && \
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));q=d[\"quantization_config\"];print(json.dumps(q,indent=1)[:2000])" \
      $(find $HOME/models/qwen38-27b-mxfp4a16 -name config.json | head -1)'
  ```
  Paste into the task report: the resolved snapshot directory (ruling **R5**),
  the whole `quantization_config` (`quant_method`, `format`, the single
  `config_groups` entry with its `weights` object, its `targets`, the top-level
  `ignore` list, and **whether `input_activations` is present**), and the header
  of one shard showing a `…mlp.gate_proj.weight_packed` `U8 [N, K/2]` beside its
  `weight_scale` `U8 [N, K/32]`. **Every string this step reads is recorded, not
  guessed** - in particular the `format` value, which Step 2 records and does
  **not** gate on. If the artifact does not exist yet, run
  `tools/box.sh run 'bash tools/quantize_qwen38_mxfp4.sh'` and paste its
  `VERIFIED B70-READY …` line; the script is self-verifying and already refuses
  a W4A4 config.
- [ ] **Step 2: `QuantConfig::parse` - the third spelling, branched before
  anything reads `bits`.** The current first statement after the null check is
  `q.bits = uint32_t(qcv->at("bits").num())` (`quant.cc:38`), which throws on a
  compressed-tensors config; the branch therefore goes **above** it.
  ```cpp
  QuantConfig QuantConfig::parse(const common::json::Value& config_json) {
    const common::json::Value* qcv = config_json.find("quantization_config");
    if (!qcv) throw std::runtime_error("config.json has no quantization_config");
    QuantConfig q;
    if (const common::json::Value* qm = qcv->find("quant_method")) q.quant_method = qm->str();
    // **The third vocabulary** (ruling A20 addendum, verified against
    // compressed-tensors' quant_scheme.py presets 2026-09-05): llm-compressor
    // writes no top-level bits/group_size/sym at all -- they live inside the one
    // config_groups entry's `weights` object -- so this branch precedes every
    // read of those keys. The scheme string itself is NOT read: `MXFP4A16` and
    // `MXFP4` differ only by `input_activations`, and a label is not evidence
    // (docs/02). What is read is the arithmetic, and W4A4 is refused by the
    // presence of the activation object rather than by the absence of a name.
    if (q.quant_method == "compressed-tensors") {
      q.scheme = Scheme::Mxfp4A16;
      if (const common::json::Value* f = qcv->find("format"); f && !f->is_null())
        q.ct_format = f->str();            // recorded in the report; never gated on
      const common::json::Value* cg = qcv->find("config_groups");
      if (!cg || !cg->is_object() || cg->obj().size() != 1)
        throw std::runtime_error(
            "compressed-tensors: expected exactly one config_groups entry, got " +
            std::to_string(cg && cg->is_object() ? cg->obj().size() : 0u) +
            " -- this loader implements one uniform weight scheme for the whole model");
      const auto& [gname, g] = *cg->obj().begin();
      if (const common::json::Value* ia = g.find("input_activations");
          ia && !ia->is_null() && !(ia->is_object() && ia->obj().empty()))
        throw std::runtime_error(
            "config_groups['" + gname + "'].input_activations is present -- this is the W4A4 "
            "`MXFP4` preset, not `MXFP4A16`. The engine's activations are bf16 by ruling A20; "
            "re-quantise with tools/quantize_qwen38_mxfp4.sh, which hard-codes MXFP4A16.");
      const common::json::Value* w = g.find("weights");
      if (!w || !w->is_object())
        throw std::runtime_error("config_groups['" + gname + "'] has no weights object");
      q.bits = uint32_t(w->at("num_bits").num());
      q.group_size = uint32_t(w->at("group_size").num());
      q.sym = w->at("symmetric").boolean();
      const std::string type = w->at("type").str();
      q.desc_act = false;            // the format has no activation-order permutation
      q.desc_act_declared = true;    // ...and that is a property of the format, not an inference
      if (q.bits != 4 || q.group_size != 32 || !q.sym || type != "float")
        throw std::runtime_error(
            "unsupported compressed-tensors weight scheme: need num_bits=4 type=float "
            "group_size=32 symmetric=true (MXFP4A16), got num_bits=" + std::to_string(q.bits) +
            " type=" + type + " group_size=" + std::to_string(q.group_size) +
            " symmetric=" + (q.sym ? "true" : "false"));
      if (const common::json::Value* ig = qcv->find("ignore"); ig && ig->is_array())
        q.ct_ignore_rules = ig->arr().size();
      if (const common::json::Value* tg = g.find("targets"); tg && tg->is_array())
        q.ct_targets = tg->arr().size();
      return q;
    }
    // ... the existing gptq / auto-round path, unchanged from `q.bits = ...` down.
  }
  ```
  **Note the two `desc_act` lines.** They are not a copy of the auto-round
  inference: MXFP4's scale group is a property of the tensor's own layout and
  there is no `g_idx` mechanism in compressed-tensors at all, so `false` is a
  *fact about the format*, recorded as declared. `loader::load`'s existing
  `!qc.desc_act_declared && scan.g_idx_tensors != 0` guard therefore does not
  fire, and a checkpoint that somehow shipped a `g_idx` would still be caught by
  the unconsumed-tensor count.
- [ ] **Step 3: `LinearSrc::classify` - the third branch, first.** Order
  matters: probe `.weight_packed` before `.qweight` before `.weight`, so a
  hybrid checkpoint classifies by what it ships rather than by which branch runs
  first. Insert above the `.qweight` branch:
  ```cpp
  auto wp = ts.find(prefix + ".weight_packed");
  if (wp != ts.end()) {
    auto ws = ts.find(prefix + ".weight_scale");
    if (ws == ts.end()) throw std::runtime_error(prefix + ": weight_packed without weight_scale");
    if (wp->second.dtype != "U8") throw std::runtime_error(prefix + ".weight_packed dtype " + wp->second.dtype);
    if (ws->second.dtype != "U8") throw std::runtime_error(prefix + ".weight_scale dtype " + ws->second.dtype);
    check_rank2(wp->second, prefix + ".weight_packed");
    check_rank2(ws->second, prefix + ".weight_scale");
    LinearSrc s;
    s.kind = WKind::Mxfp4;
    s.N = uint32_t(wp->second.shape[0]);
    s.K = uint32_t(wp->second.shape[1]) * 2;      // two e2m1 nibbles per byte
    if (ws->second.shape[0] != s.N || ws->second.shape[1] != s.K / 32)
      throw std::runtime_error(prefix + ".weight_scale shape mismatch: expected [" +
                               std::to_string(s.N) + "][" + std::to_string(s.K / 32) + "]");
    // uint8 planes: no alignment requirement beyond 1, but the same guard runs
    // so a future dtype change cannot skip it silently.
    check_align(set.data(wp->second), alignof(uint8_t), prefix + ".weight_packed");
    check_align(set.data(ws->second), alignof(uint8_t), prefix + ".weight_scale");
    s.packed = reinterpret_cast<const uint8_t*>(set.data(wp->second));
    s.e8m0 = reinterpret_cast<const uint8_t*>(set.data(ws->second));
    s.name = prefix;
    return s;
  }
  ```
  **`K` comes from `weight_packed.shape[1] * 2`, and the `weight_scale` shape is
  the cross-check** - the same discipline the int4 branch uses
  (`scales.shape[0] != K/64` throws). `compressed-tensors` may also emit
  `<prefix>.weight_shape` (int32 `[2]`); consume it in `load_linear` beside the
  two data tensors and, if present, assert it equals `{N, K}` - an unconsumed
  `weight_shape` would otherwise make `report.unconsumed` non-zero and fail the
  load for a cosmetic reason.
- [ ] **Step 4: `assert_quant_invariants` - the e8m0 scan, in the shape of the
  f16 one.** Add a fourth `else if` beside `.qzeros` / `.g_idx` / `.scales`:
  ```cpp
  } else if (name.size() > 13 && name.compare(name.size() - 13, 13, ".weight_scale") == 0) {
    // e8m0: value = 2^(E-127). E == 255 is NaN by OCP MX v1.0 and the dequant
    // has no guard for it -- one NaN scale poisons a block of 32 weights and
    // nothing downstream would notice, exactly as an f16 NaN scale would.
    // E == 0 is 2^-127, a perfectly meaningful SUBNORMAL fp32; it is COUNTED,
    // not rejected, the same ruling the f16 subnormals got on 2026-08-25. The
    // kernels read it correctly because -cl-denorms-are-zero is forbidden
    // (cmake/ocloc.cmake:41) -- which is now an arithmetic requirement, not
    // only a numerics preference.
    ++scan.e8m0_tensors;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(set.data(t));
    const size_t n = set.bytes(t);
    for (size_t i = 0; i < n; ++i) {
      if (p[i] == 0xFFu)
        throw std::runtime_error(name + "[" + std::to_string(i) +
                                 "] = 0xFF, an e8m0 NaN -- the dequant has no guard for it");
      if (p[i] == 0u) ++scan.subnormal_scales;
    }
  }
  ```
  Cost: 0.76 GB of uint8, ~0.3 s, the same budget the f16 scan already spends.
- [ ] **Step 5: `common/mxfp4.h` and the two repacks.** Write `common::Mxfp4Ct`
  exactly as given under *Interfaces: Produces*, then the two repack functions in
  `common/repack.h`. The **layout-3 tile is byte-for-byte the same 544-byte
  object as layout 1**, and that is the point:
  ```cpp
  // MXFP4 layout 3: per (n_tile of 16, k_group of 64) one 136-u32 tile - the
  // SAME 544 bytes as int4 layout 1, because both formats are 4.25 bits/weight:
  //   128 u32 of nibbles, tile[j*16 + l] = the 8 e2m1 codes for
  //   k = g*64 + j*8 + 0..7 of column n_tile*16 + l, low nibble = lowest k;
  //   then 16 ushorts, one per lane: tile[128..135] read as ushort[16], where
  //   ushort[l] = e8m0(k-block 2g, col l) | (e8m0(k-block 2g+1, col l) << 8).
  // The kernel therefore issues the IDENTICAL pair of instructions layout 1
  // issues -- intel_sub_group_block_read8 + intel_sub_group_block_read_us --
  // and only the arithmetic under them changes. 16 f16 scales are 32 B; 32 e8m0
  // scales are 32 B. Tiles are ordered g-inner, n_tile-outer.
  inline void repack_mxfp4_layout3_cols(uint32_t K, uint32_t N_total,
                                        const std::vector<MxfpColSource>& cols, uint32_t* out) {
    const uint32_t G = K / 64, NT = N_total / 16;
    for (uint32_t nt = 0; nt < NT; ++nt)
      for (uint32_t g = 0; g < G; ++g) {
        uint32_t* tile = out + (size_t(nt) * G + g) * 136;
        for (uint32_t j = 0; j < 8; ++j)
          for (uint32_t l = 0; l < 16; ++l) {
            const MxfpColSource& c = cols[nt * 16 + l];
            // eight consecutive k -> one u32, low nibble = lowest k. The source
            // byte holds k (low) and k+1 (high): compressed-tensors packs
            // `(x[1::2] << 4) | x[0::2]` (vLLM break_fp4_bytes, verified
            // 2026-09-05), i.e. EVEN k in the low nibble.
            uint32_t word = 0;
            const uint32_t k0 = g * 64 + j * 8;
            for (uint32_t i = 0; i < 8; i += 2) {
              const uint8_t b = c.packed[size_t(c.n) * c.k_half + (k0 + i) / 2];
              word |= uint32_t(b) << (4 * i);   // two nibbles at once, order preserved
            }
            tile[j * 16 + l] = word;
          }
        uint16_t* sc = reinterpret_cast<uint16_t*>(tile + 128);
        for (uint32_t l = 0; l < 16; ++l) {
          const MxfpColSource& c = cols[nt * 16 + l];
          sc[l] = uint16_t(c.scale[size_t(c.n) * c.k_blocks + 2 * g]) |
                  uint16_t(uint16_t(c.scale[size_t(c.n) * c.k_blocks + 2 * g + 1]) << 8);
        }
      }
  }
  ```
  `repack_mxfp4_layout2_cols` writes the same nibble words into
  `w_out[(g*8 + j)*N_total + n]` - GPTQ's `[K/8][N]` geometry, so
  `gemv.cl`'s `LAYOUT == 0` addressing and its `GEMV_BLOCK2D=8`
  `intel_sub_group_2d_block_read_32b_8r16x1c` message work **verbatim** - plus
  `sc_out[b*N_total + n] = c.scale[n*k_blocks + b]` for `b < K/32`, which is
  both what the GEMV wants (16 lanes read 16 consecutive bytes) **and exactly
  the `[K/32][N]` N-contiguous plane the prefill mainloop's
  `Stride<_1,int64_t,int64_t>` requires**. One transpose, at load, for both
  consumers.
- [ ] **Step 6: the model table and `load_linear`.** `model::WeightKind` gains
  `Mxfp4`. The production `(S, layout)` map keeps **every `S` and every
  layout *choice*** and changes only the layout *id*, because layouts 2 and 3
  are the same geometries as 0 and 1:

  | linear | int4 row today | MXFP4 row | why |
  |---|---|---|---|
  | QkvZ | `{5120,16384,1,1}` | `{5120,16384,1,3}` | N = 2^14; layout 1 beat layout 0 by 33% there (docs/12), and layout 3 has layout 1's address stream |
  | OutProj | `{6144,5120,4,0}` | `{6144,5120,4,2}` | layout 0 won by 3.0%; `GEMV_BLOCK2D=8` cell |
  | GateUp | `{5120,34816,8,0}` | `{5120,34816,8,2}` | layout 0 won by 1.1% |
  | Down | `{17408,5120,4,0}` | `{17408,5120,4,2}` | layout 0 won by 6.8%; `GEMV_BLOCK2D=8` cell |
  | Qkv | `{5120,14336,2,0}` | `{5120,14336,2,2}` | layout 0 won by 6.7%; `GEMV_BLOCK2D=8` cell |
  | OProj | `{6144,5120,4,0}` | `{6144,5120,4,2}` | shares OutProj's shape |
  | LmHead | `{5120,248320,1,1}` int4 | `{5120,248320,1,3}` | S must be 1 so capture writes straight into `logits` (qwen35.h) |

  A `Qwen35::linear(id)` cannot return two rows for one ordinal, so the MXFP4
  rows live beside the int4 ones exactly as `lm_head_int4()` lives beside
  `table()`'s bf16 row: add `const FusedLinear& mxfp4_row(LinearId)` and a
  `Qwen35::linear(LinearId, WeightKind)` overload that picks. `loader::load`
  already chooses `lm_head`'s row by content
  (`LinearSrc::classify(set, "lm_head").kind`); extend that one line to choose
  **the whole table's kind** from `qc.scheme`, and keep `lm_head`'s
  quantised-or-not decision by content on top of it. In `load_linear`:
  - widen the layout guard to
    `if (fl.kind == Int4 && sh.layout > 1) throw` **and**
    `if (fl.kind == Mxfp4 && (sh.layout < 2 || sh.layout > 3)) throw`;
  - the part/kind cross-check gains its third arm, keeping the existing
    message's meaning - *"the checkpoint's exclusions moved"*;
  - staging: `mxfp4_words(sh)` = `layout 2 ? (K/8)*N : (N/16)*(K/64)*136` and
    `mxfp4_scale_bytes(sh)` = `layout 2 ? (K/32)*N : 0`, sized from the linears
    **this** load will repack exactly as `int4_words` is;
  - accounting: `rep.mxfp4_bytes += K*N/2` and `rep.scale_bytes += K*N/32`
    (`lm_head` to `rep.lm_head_bytes` as today, `K*N/2 + K*N/32`), so the
    buckets still sum to the allocation total to the byte;
  - `view.consumed.insert(prefix + ".weight_packed" / ".weight_scale" /
    ".weight_shape")`.
- [ ] **Step 7: the report and the byte table.** `LoadReport` gains
  `size_t mxfp4_bytes = 0;`, included in `total()` and in `read_per_token`; the
  printf gains one line, `mxfp4 %13zu B %7.3f GB`, and the `quant` line prints
  the third vocabulary: `mxfp4a16 e2m1 g32 sym (compressed-tensors, format=%s), %zu ignore rules + %zu targets`.
  Then add to `docs/13-loader.md`, under "Memory and the `W` cross-check", the
  arithmetic that makes this a **non-event**:

  > **The format costs nothing.** int4 g64 spends `K·N/2` on nibbles and one
  > f16 per 64 weights = `2·K·N/64 = K·N/32` on scales. MXFP4A16 spends
  > `K·N/2` on nibbles and one e8m0 byte per 32 weights = `1·K·N/32` on scales.
  > `16/64 = 8/32 = 0.25` bits per weight either way, so both are **4.25
  > bits/weight and `K·N·17/32` bytes per tensor, exactly**. The `read/token`
  > line on the MXFP4A16 artifact is therefore **13,672,613,888 B**, the same
  > integer the `qwen38-27b-w4g64-rtn` row prints, and the `W` cross-check
  > passes with both sides unmoved.

- [ ] **Step 8: `tests/loader/mxfp4_test.cc` - the unit bars, no device.**
  Registered without the `checkpoint` label so it runs everywhere.
  1. **`QuantConfig::parse`, four inline JSON strings**: the real MXFP4A16
     config from Step 1 → `Scheme::Mxfp4A16`, `bits 4`, `group_size 32`,
     `sym`, `desc_act_declared`; the same with an `input_activations` object →
     **throws, and the message contains `W4A4`**; `group_size: 16` → throws;
     two `config_groups` entries → throws. The W4A4 case is the addendum's whole
     point and gets its own named test function.
  2. **The repacks against `Mxfp4Ct::at`.** `Mxfp4Ct::random(256, 64, 4242)`
     (uniform random bytes for `packed`; `scale` bytes drawn uniformly from
     `[120, 134]`, i.e. `2^-7 … 2^7`, so products are O(1)); build both layouts
     through the same `MxfpColSource` map a fused linear would use
     (`Single`, `Concat` of two 32-column parts, `Interleave16` of two 32-column
     parts) and require, **bitwise for every element**, that reading the packed
     result back with the kernel's own index arithmetic reproduces
     `w.at(k, n)`'s float exactly. Both sides are a table lookup and one
     `ldexp`, so a differing bit is a bug, not a tolerance.
  3. **The e8m0 edge cases**: `e8m0_to_f32(0) == std::ldexp(1.0f, -127)` and
     `std::fpclassify(...) == FP_SUBNORMAL`; `e8m0_to_f32(127) == 1.0f`;
     `e8m0_to_f32(254) == std::ldexp(1.0f, 127)` and finite. `0xFF` never
     reaches the helper - the loader refuses it - and the test asserts the
     refusal through `assert_quant_invariants` on a two-tensor synthetic set.
  4. **All 16 e2m1 codes**: `kE2M1` against the branch-free construction
     Task 3 compiles, *including the sign of zero* -
     `std::signbit(kE2M1[8])` is true and `kE2M1[0]` is `+0.0`.
- [ ] **Step 9: extend `load_checkpoint_test` to the real artifact.** It takes
  the snapshot as `argv[1]` already. Register a second ctest,
  `load_checkpoint_mxfp4_test`, that runs the same binary against
  `${B70_TEST_SNAPSHOT_MXFP4}` (ruling **R5**), labelled `checkpoint`, and make
  the binary's assertions kind-aware:
  - exactly one of `report.int4_bytes` / `report.mxfp4_bytes` is non-zero, and
    the non-zero one is in `(12.0e9, 12.4e9)`;
  - **`report.read_per_token == 13672613888`** when `lm_head` is quantised -
    an exact integer, cross-checked against the RTN row, not a band;
  - `m.linears.size() == 48*5 + 16*4 + 1`, `m.layer_small.size() == 64`;
  - the device round-trip: read back one layout-3 tile of layer 0's `qkv‖z` and
    one layout-2 `(k-octet, n)` word plus its two scale bytes from layer 1's
    `gate‖up`, and compare against the CPU repack of the mmapped source -
    the same proof the int4 path already carries;
  - `report.unconsumed == 0` (this is what catches an unhandled
    `weight_shape`).
- [ ] **Step 10: run it.** `tools/box.sh test "mxfp4_test|load_checkpoint"`.
  Paste the full loader report for the MXFP4 artifact into the task report and
  put it in `docs/13-loader.md` beside the two existing ones.
- [ ] **Step 11: Commit** -
  `feat(loader): compressed-tensors MXFP4A16 reader - e2m1 layouts 2 and 3, W4A4 refused`

### Task 2: the dequant oracle for e2m1/e8m0 - what re-anchors the golden gate

*`tools/oracle/dump.py` builds its reference bf16 state dict with
`dequant.py`'s function - "the SAME function the C++ loader is bit-compared
against - so the oracle and the engine start from identical bf16 weights"
(dump.py's own docstring). Change the format and that sentence stops being true
until this task makes it true again. **Nothing downstream can be trusted before
this passes.***

**Files:**
- Modify: `tools/oracle/dequant.py`, `tools/oracle/dump.py`,
  `tests/loader/dequant_fixture_test.cc`, `tests/CMakeLists.txt`,
  `tools/oracle/README.md`
- Create: `tests/golden/mxfp4_fixture.safetensors` (committed, ~9 KB)

**Interfaces:**
- Consumes: Task 1's `common::Mxfp4Ct` and `common::kE2M1` / `e8m0_to_f32`;
  `common::f32_to_bf16`; `loader::MappedFile` / `SafetensorsSet::parse_header`;
  `dump.py`'s strict-load contract.
- Produces: `dequant_mxfp4`, the committed fixture, a two-format
  `dequant_fixture_test`, and a `dump.py` that loads either format.

- [ ] **Step 1: `dequant.py` gains the MXFP4 path.** One function, in the same
  single-rounding discipline as `dequant_gptq` (fp32 product, **one** cast to
  bf16), with the same `n_chunk` column-blocking so an `lm_head` at
  `[248320][2560]` does not materialise 25 GB:
  ```python
  # The e2m1 code point table (OCP MX v1.0 Sec 5.3.3): bit 3 is the sign, bits
  # 0-2 the magnitude. Identical to vLLM's kE2M1ToFloat with the sign applied
  # (vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py) and
  # to common::kE2M1 on the C++ side -- three implementations, one table.
  _E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

  def dequant_mxfp4(weight_packed: torch.Tensor, weight_scale: torch.Tensor,
                    n_chunk: int = 0) -> torch.Tensor:
      """[N, K/2] uint8 + [N, K/32] uint8 e8m0 -> [K, N] bf16.

      w = E2M1[q & 7] * (-1 if q & 8 else 1) * 2**(int(e) - 127), the product in
      fp32 with ONE cast to bfloat16 -- the same single-rounding path as
      common::f32_to_bf16(common::Mxfp4Ct::at(k, n)), which is what makes the
      fixture bit-exact comparable.

      Packing: compressed-tensors stores (x[..., 1::2] << 4) | x[..., 0::2], so
      the LOW nibble of byte b is k = 2b and the HIGH nibble is k = 2b+1.
      Output is transposed to [K, N] to match dequant_gptq's contract.
      """
      assert weight_packed.dtype == torch.uint8 and weight_scale.dtype == torch.uint8
      n, k_half = weight_packed.shape
      k = k_half * 2
      assert weight_scale.shape == (n, k // 32), (weight_scale.shape, (n, k // 32))
      if n_chunk and n_chunk < n:
          out = torch.empty((k, n), dtype=torch.bfloat16)
          for a in range(0, n, n_chunk):
              b = min(a + n_chunk, n)
              out[:, a:b] = dequant_mxfp4(weight_packed[a:b].contiguous(),
                                          weight_scale[a:b].contiguous())
          return out
      lo = (weight_packed & 0x0F).to(torch.long)
      hi = (weight_packed >> 4).to(torch.long)
      q = torch.stack((lo, hi), dim=2).reshape(n, k)          # k = 2*b + {0,1}
      mag = _E2M1[q & 7]
      val = torch.where((q & 8).bool(), -mag, mag)            # -0.0 for q == 8
      # e8m0: 2**(E-127). E == 0 is 2**-127, a SUBNORMAL fp32; ldexp keeps it.
      exp = torch.ldexp(torch.ones((), dtype=torch.float32),
                        weight_scale.to(torch.int32) - 127)
      w32 = val * exp.repeat_interleave(32, dim=1)
      return w32.to(torch.bfloat16).t().contiguous()
  ```
  **Two things this must not do**, and the test in Step 3 is what proves it: use
  `2.0 ** (e - 127)` in float64 and round twice, and lose the sign of zero
  (`q == 8` must produce `-0.0`, whose bf16 word is `0x8000`, not `0x0000`).
- [ ] **Step 2: the fixture.** `--write-mxfp4-fixture` beside
  `--write-fixture`, `torch.manual_seed(1234)`, `K, N = 128, 32` (the same shape
  the int4 fixture uses), `weight_packed` uniform random bytes,
  `weight_scale` uniform in `[120, 134]` **plus a deliberately planted `0`
  (2^-127) and a `254` (2^127) so the extremes are in the committed bits**, and
  the same chunked-vs-unchunked equality assertion at `n_chunk` 16 and 12 that
  the int4 fixture carries. Tensors: `weight_packed`, `weight_scale`,
  `dequant`. Size ≈ 128·32·2 + 32·64 + 32·4 = 10,496 B - committable, like the
  existing one.
- [ ] **Step 3: `dequant_fixture_test` reads both fixtures.** Give the binary a
  second optional argument and register two ctests
  (`dequant_fixture_test`, `dequant_fixture_mxfp4_test`) rather than branching on
  a filename, so a missing fixture is a failing test and not a silent skip. The
  MXFP4 side is the exact mirror of the int4 side:
  ```cpp
  common::Mxfp4Ct w;
  w.N = uint32_t(wp->shape[0]);
  w.K = uint32_t(wp->shape[1]) * 2;
  w.packed.assign(pp, pp + size_t(w.N) * (w.K / 2));
  w.scale.assign(sp, sp + size_t(w.N) * (w.K / 32));
  for (uint32_t k = 0; k < w.K; ++k)
    for (uint32_t n = 0; n < w.N; ++n) {
      const uint16_t ours = common::f32_to_bf16(w.at(k, n));
      const uint16_t theirs = golden[size_t(k) * w.N + n];
      if (ours != theirs) { /* print k, n, both words, the byte, the nibble,
                               the e8m0 exponent and its float value */ return 1; }
    }
  ```
  **Bit-exact, no tolerance**, exactly as the int4 fixture is: both sides are a
  table lookup, an exact `ldexp`, one correctly-rounded fp32 multiply and one
  RNE. A differing bit is a disagreement about what the nibbles mean, and
  "NOTHING downstream can be trusted" (that test's own header) still applies.
- [ ] **Step 4: `dump.py` picks the format by content.** It currently calls
  `dequant_gptq` over `<name>.qweight` / `.scales`. Give it the same
  suffix-decides rule the loader has: for each linear name in the shard index,
  `weight_packed` + `weight_scale` → `dequant_mxfp4`, `qweight` + `scales` →
  `dequant_gptq`, `weight` → verbatim. **Do not read `config.json`'s
  `quant_method`** - the oracle and the loader must agree on the *rule*, not
  only on the answer, or a hybrid checkpoint would diverge silently. Keep every
  hard abort the docstring lists (strict load, no meta tensors, `tokens` length
  == `--gen`, every GDN layer contributed a state, last-layer resid finite).
  Note in the file's header that peak RSS is unchanged: the dequantised state
  dict is bf16 either way, so the measured 61-62.8 GiB stands.
- [ ] **Step 5: run both fixtures and record the cross-language proof.**
  `tools/box.sh test "dequant_fixture"`. Both green, both printing their element
  counts. Then add to `tools/oracle/README.md` the sentence that makes the trust
  chain explicit: *"Three implementations of one table - `_E2M1` in
  `dequant.py`, `common::kE2M1` in C++, and the 16-entry LUT compiled into
  `gemv.cl` (Task 3) - held together by `dequant_fixture_mxfp4_test` and by
  `gemv_test`'s control-variant bit-identity."*
- [ ] **Step 6: Commit** -
  `feat(oracle): e2m1/e8m0 dequant - bit-exact across python and C++`

### Task 3: decode - a `GEMV_FMT_MXFP4` unpack path in `gemv.cl`

*The −1.20 ms/token retune is at risk and is **re-measured, not assumed**
(A20). This task is built to give it the best possible chance: layouts 2 and 3
are layouts 0 and 1's geometry, so the lane mapping, the work-group shape, the
grid, the `intel_sub_group_block_read8` / `_read_us` pair and the
`GEMV_BLOCK2D=8` message all survive verbatim, and **only the arithmetic under
them changes**. Bytes per token are identical to the byte, so this is a pure
ALU question against a bandwidth wall.*

**Files:**
- Modify: `src/kernels/gemv.cl`, `src/kernels/CMakeLists.txt`,
  `tests/kernels/gemv_test.cc`, `tests/kernels/gemv_harness.h`,
  `tools/probe/probe_gemv_loads.cl` (the probe cells), `tests/CMakeLists.txt`
- Create: `docs/preregistration-mxfp4-decode-2026-09-05.md` (committed **before**
  any timing)

**Interfaces:**
- Consumes: Task 1's layouts 2/3 and `model::WeightKind::Mxfp4`;
  `kernels::gemv_variant(M,K,N,S,L)` unchanged; `common::Mxfp4Ct`;
  `tests/kernels/gemv_ref.h`'s `random_bf16` / `max_abs_err` / `tol_for`.
- Produces: `gemv_M1_K…_N…_S…_L{2,3}` binaries, a control variant, the
  per-shape re-measurement, and the pre-registered verdict.

- [ ] **Step 1: the kernel. One orthogonal compile-time axis, not a third
  layout.** `LAYOUT` keeps its meaning - **0 = plane, 1 = tile** - and
  `GEMV_FMT_MXFP4` selects the arithmetic. The *device* layout ids 2 and 3 live
  in `GemvShape` and in the variant **name**; `add_gemv_variant` maps
  `L2 → LAYOUT=0 GEMV_FMT_MXFP4=1` and `L3 → LAYOUT=1 GEMV_FMT_MXFP4=1`. Nothing
  in `capture.cc` changes, because it already binds
  `kernels::gemv_variant(M, K, N, s.S, s.layout)`.
  ```c
  #ifndef GEMV_FMT_MXFP4
  #define GEMV_FMT_MXFP4 0
  #endif
  #if GEMV_FMT_MXFP4 && GEMV_DEQ_SHIFT
  #error "gemv: GEMV_DEQ_SHIFT is the int4 xor trick; MXFP4 has its own decode"
  #endif

  #if GEMV_FMT_MXFP4
  // e2m1 (OCP MX v1.0 Sec 5.3.3): bit 3 sign, bits 0-2 magnitude ->
  //   0, 0.5, 1, 1.5, 2, 3, 4, 6.  Note -0.0 at q == 8: the oracle produces it
  //   (0.0 * -1.0), common::kE2M1 stores it, and f32_to_bf16(-0.0) is 0x8000,
  //   so the three implementations must agree on it or the fixture fails.
  //
  // The compiled form is branch-free bit construction, NOT a table: for
  // magnitude m in [2,7] the fp32 word is 0x3F000000 + (m << 22) exactly
  // (m=2 -> 0x3F800000 = 1.0 ... m=7 -> 0x40C00000 = 6.0), and m in {0,1} is
  // patched with two selects. A __constant float[16] would be an indexed load
  // per nibble on a kernel whose whole cost is the weight stream. The LUT
  // variant is compiled as GEMV_FMT_MXFP4_LUT for the control binary and
  // gemv_test holds the two bit-identical over all 16 codes.
  inline float e2m1f(uint q) {
  #if GEMV_FMT_MXFP4_LUT
    __constant const float t[16] = {0.f,  .5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                    -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t[q & 15u];
  #else
    const uint m = q & 7u;
    float v = as_float(0x3F000000u + (m << 22));
    v = (m == 0u) ? 0.0f : ((m == 1u) ? 0.5f : v);
    return (q & 8u) ? -v : v;
  #endif
  }

  // 8 e2m1 codes of one u32 word times 8 consecutive activations, in dot8()'s
  // exact accumulation ORDER -- lowest k first, one fp32 add per nibble. The
  // order is preserved deliberately: it is the only part of the decode the
  // golden gate can see, and keeping it is what gives a tie-aware gate its best
  // chance of landing on the same ids.
  inline float dot8_mxfp4(uint word, ushort8 xv) {
    float a = 0.f;
    a += e2m1f((word      ) & 0xFu) * bf16f(xv.s0);
    a += e2m1f((word >>  4) & 0xFu) * bf16f(xv.s1);
    a += e2m1f((word >>  8) & 0xFu) * bf16f(xv.s2);
    a += e2m1f((word >> 12) & 0xFu) * bf16f(xv.s3);
    a += e2m1f((word >> 16) & 0xFu) * bf16f(xv.s4);
    a += e2m1f((word >> 20) & 0xFu) * bf16f(xv.s5);
    a += e2m1f((word >> 24) & 0xFu) * bf16f(xv.s6);
    a += e2m1f((word >> 28) & 0xFu) * bf16f(xv.s7);
    return a;
  }

  // e8m0 -> the block's shared scale. value = 2^(E-127); E == 0 is 2^-127, a
  // SUBNORMAL fp32, which is exact here only because -cl-denorms-are-zero is
  // forbidden (cmake/ocloc.cmake:41 fatals on it). E == 255 is NaN and cannot
  // reach here: loader::assert_quant_invariants refuses a checkpoint holding
  // one. ldexp rather than an exponent-field add, because the add is undefined
  // for the subnormal end and the scale is computed ONCE per (k-block, lane) --
  // 2 per 64-k group, against 64 nibble decodes. It is not the hot instruction.
  inline float e8m0f(uint e) { return ldexp(1.0f, (int)e - 127); }
  #endif
  ```
  The k-group loop keeps `GROUP 64` and `TILE_U32 136` and grows one level of
  structure, because the scale block is now 32 rather than 64:
  ```c
    for (uint g = g0; g < g1; ++g) {
      uint wv[8];
  #if GEMV_FMT_MXFP4
      float s_lo, s_hi;                 // k-blocks 2g and 2g+1
  #else
      float scale;
  #endif
  #if LAYOUT == 0
      /* ... the existing wv[] read, block2D or strided, UNCHANGED ... */
  #if GEMV_FMT_MXFP4
      // sc is [K/32][N] uint8: 16 lanes read 16 consecutive bytes, one line.
      s_lo = e8m0f(sc[(size_t)(g * 2u    ) * N + n]);
      s_hi = e8m0f(sc[(size_t)(g * 2u + 1) * N + n]);
  #else
      scale = (float)scales[(size_t)g * N + n];
  #endif
  #else
      __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
      uint8 blk = intel_sub_group_block_read8(tile);
      wv[0] = blk.s0; /* ... */ wv[7] = blk.s7;
      ushort sh = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
  #if GEMV_FMT_MXFP4
      s_lo = e8m0f((uint)(sh & 0xFFu));   // k-block 2g   in the low byte
      s_hi = e8m0f((uint)(sh >> 8));      // k-block 2g+1 in the high byte
  #else
      scale = (float)as_half(sh);
  #endif
  #endif
  #if GEMV_FMT_MXFP4
      float lo[M], hi[M];
      for (int m = 0; m < M; ++m) { lo[m] = 0.f; hi[m] = 0.f; }
      for (int j = 0; j < 8; ++j)
        for (int m = 0; m < M; ++m) {
          ushort8 xv = vload8(0, x + (size_t)m * K + g * GROUP + j * 8);
          // j 0..3 is k in [64g, 64g+32) = block 2g; j 4..7 is block 2g+1.
          if (j < 4) lo[m] += dot8_mxfp4(wv[j], xv);
          else       hi[m] += dot8_mxfp4(wv[j], xv);
        }
      for (int m = 0; m < M; ++m) acc[m] += lo[m] * s_lo + hi[m] * s_hi;
  #else
      /* ... the existing gacc/dot8 loop and `acc[m] += gacc[m] * scale` ... */
  #endif
    }
  ```
  **The one numerics change, named.** int4 folds a 64-wide group with one
  multiply (`Σ_{j<8} dot8_j · scale`); MXFP4 folds two 32-wide halves
  (`Σ_{j<4} dot8_j · s_lo + Σ_{4≤j<8} dot8_j · s_hi`). The **within-half order is
  byte-identical to today's**; what is added is one fp32 multiply and one add per
  64-k group per `m`. It is unavoidable - the scale group *is* 32 - and it is
  the only reassociation this kernel introduces. The golden gate is the arbiter
  (docs/14, "The diagnostics move with every lever"); Step 8 pre-registers it.
  `if (j < 4)` on a compile-time-unrolled `j` is a compile-time select, not a
  branch; if IGC fails to unroll it, split the loop into `j = 0..3` and
  `j = 4..7` and record that it was necessary.
- [ ] **Step 2: `src/kernels/CMakeLists.txt` - the variants, and the build-time
  budget.** `add_gemv_variant(M K N S L)` gains the mapping and the production
  `extra_defs`, mirroring Task 4's block one-for-one:
  ```cmake
  function(add_gemv_variant M K N S L)
    set(extra_defs)
    set(hw_layout ${L})
    if(L GREATER 1)                      # 2 -> plane, 3 -> tile
      math(EXPR hw_layout "${L} - 2")
      list(APPEND extra_defs GEMV_FMT_MXFP4=1)
    endif()
    if(M EQUAL 1)
      # Task 4's production cells, carried across the format pivot: the LAYOUT
      # and S choices are unchanged and so is GEMV_BLOCK2D, which is a property
      # of the [K/8][N] surface and not of the arithmetic. GEMV_DEQ_SHIFT is the
      # int4 xor trick and has no MXFP4 counterpart -- e2m1f IS the dequant.
      if(hw_layout EQUAL 0 AND K EQUAL 6144 AND N EQUAL 5120 AND S EQUAL 4)
        list(APPEND extra_defs GEMV_BLOCK2D=8)
      elseif(hw_layout EQUAL 0 AND K EQUAL 5120 AND N EQUAL 14336 AND S EQUAL 2)
        list(APPEND extra_defs GEMV_BLOCK2D=8)
      elseif(hw_layout EQUAL 0 AND K EQUAL 17408 AND N EQUAL 5120 AND S EQUAL 4)
        list(APPEND extra_defs GEMV_BLOCK2D=8)
      endif()
      if(L LESS 2)
        # ... the existing GEMV_DEQ_SHIFT=1 cells, untouched ...
      endif()
    endif()
    add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L} SOURCE ${GEMV_CL}
                     DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${hw_layout} ${extra_defs})
  endfunction()
  ```
  **The default build adds seven binaries, not fifty:** the six production cells
  (`{6144,5120,4,2}`, `{5120,14336,2,2}`, `{5120,16384,1,3}`, `{5120,34816,8,2}`,
  `{17408,5120,4,2}`, `{5120,248320,1,3}`) plus
  `gemv_control_M1_K6144_N5120_S4_L2` (`GEMV_FMT_MXFP4_LUT=1 GEMV_BLOCK2D=0`).
  The full `L∈{2,3} × S∈{1,2,4,8,16} × 5 shapes` re-sweep sits behind a new
  cache option `B70_MXFP4_PROBE_MATRIX` (default **OFF**), which Step 6 turns on
  for the probe build only - the same discipline `lm_head`'s single variant
  already documents ("A probe sweep over this shape would cost five more ocloc
  compiles of a 3880-work-group kernel to price splits the binding cannot use").
- [ ] **Step 3: `gemv_test` - correctness and control-identity, before any
  timing.** Three additions, all cheap:
  1. `run_case` gains an `Mxfp4Ct` overload driving the six production cells
     against a `double` CPU reference over `w.at(k, n)` and `bf16_to_f32(x)`,
     held to `tol_for(ref)` - the same bar the int4 cases use.
  2. **The control identity**, the exact discipline
     `gemv_control_M1_K6144_N5120_S4_L0` established: build a
     `{6144, 5120}` weight whose bytes cycle so **all 16 e2m1 codes and both
     e8m0 extremes appear**
     (`packed[i] = (uint8_t)(0x10 * (i & 15) + ((i >> 4) & 15))`, and
     `scale[j] = (uint8_t)(j % 3 == 0 ? 0 : (j % 3 == 1 ? 254 : 127))`), run the
     compiled branch-free cell and the `..._LUT` control over the same
     activations, and require a `memcmp` of the full fp32 output to be **zero**.
     That is what pins `e2m1f`'s bit construction, `-0.0` included, to the table
     the oracle uses.
  3. A **layout-2 vs layout-3 agreement** case at `{5120, 16384}`: the same
     logical weight through both repacks must produce **bitwise identical**
     output at `S = 1`. Both layouts feed `dot8_mxfp4` the same words in the same
     order, so anything else is a repack bug - and this is the cheapest place to
     catch one.
- [ ] **Step 4: acceptance for the arithmetic, before the performance question
  is opened.** `ZE_AFFINITY_MASK=1 tools/box.sh test "gemv_test|mxfp4|dequant_fixture"`,
  then the whole suite. Nothing is timed yet. **Commit here** -
  `feat(kernels): gemv e2m1/e8m0 unpack at layouts 2 and 3` - so the
  pre-registration in Step 5 lands on a tree whose correctness is already
  green and cannot be quietly adjusted afterwards.
- [ ] **Step 5: pre-register, in a committed file, before any timing.**
  `docs/preregistration-mxfp4-decode-2026-09-05.md`, committed on its own:

  > **Mechanism.** Bytes read per token are **identical** - 4.25 bits/weight in
  > both formats, `K·N·17/32` per tensor, `13,672,613,888 B` per token (Task 1
  > asserts the integer). Decode measured **74.7% MBU** at `2a7df0b`. The
  > format changes only the ALU work under a bandwidth wall, and adds, per
  > 64-k group per lane, one extra fp32 multiply-add for the second scale and
  > swaps an `xor`+2 shifts per nibble for a 3-op bit construction.
  >
  > **H0 (primary): the per-shape GB/s lands within ±3% of the int4 production
  > cells.** Targets, from `docs/probe-gemv-loads-2026-08-26.md`'s composed
  > column: out/o_proj 561 → **[544, 578]**; q‖k‖v 579 → **[562, 596]**;
  > qkv‖z 562 → **[545, 579]**; gate‖up 561 → **[544, 578]**; down 574 →
  > **[557, 591]**; lm_head 576 → **[559, 593]**.
  >
  > **H1 (the alternative): the LUT/bit-construction decode is ALU-bound at the
  > 561-class shapes and one or more cells falls 5-15% short**, because e2m1
  > costs ~3 ops per nibble against the xor trick's ~2, and the retuned cells
  > sit at 94-97% of the bandwidth wall where there is no slack to hide in.
  >
  > **The bench prediction: `32.22 ± 0.5 t/s`** on the MXFP4A16 checkpoint at
  > depth 4096 / tg 256, record grade, against the int4 `32.22` row at
  > `2a7df0b`. Same bytes, same walk, same 774 launches.
  >
  > **Bars.** (1) every shape inside its ±3% band → H0 holds. (2) any shape
  > below its band by >3% → H0 falsified for that shape and the rider below
  > fires. (3) the bench row outside `32.22 ± 0.5` → **a finding**, reported
  > with the per-shape table beside it, not absorbed.
  >
  > **The rider, priced now so it is not invented later.** The −1.20 ms retune
  > was tuned **per layout** (`docs/probe-gemv-loads-2026-08-26.md`: the `S`
  > picks are M = 1 picks at layouts 0/1). Layouts 2 and 3 have the same address
  > streams, so the `S` picks should carry - but if a shape misses its band, the
  > answer is **its own `S` sweep at layouts 2/3**, not a widened bar:
  > `-DB70_MXFP4_PROBE_MATRIX=ON` costs 50 extra ocloc compiles (~4 min at
  > JOBS 44, estimated) and one `probe_gemv_loads` battery (~6 min, estimated
  > from the 2026-08-26 run). An `S` change costs a full golden-gate run
  > (`docs/15` §L4: "each candidate costs a full golden-gate run because `S`
  > reorders the split-K merge"), i.e. ~24 s of gate plus the oracle set that
  > Task 5 has already produced. Budget the rider at **one shape**; two or more
  > shapes missing is a finding for the operator, not a sweep.

- [ ] **Step 6: measure the six cells.** `-DB70_MXFP4_PROBE_MATRIX=ON`, then
  `ZE_AFFINITY_MASK=1 tools/box.sh run './build/tools/probe/probe_gemv_loads …'`
  with the same convention the 2026-08-26 battery used (40 launches per list,
  8 replays, median of the last 5, `NB = max(2, 72 MB / weight_bytes + 1)`
  weight copies so consecutive launches miss the 24 MB L2). Report the six
  production cells beside their int4 targets and the band verdict per shape.
  **Weight bytes are identical, so GB/s is directly comparable** - say so in the
  table header, because that comparability is the whole reason the bar can be
  ±3% rather than a re-derivation.
- [ ] **Step 7: the in-situ row.**
  `ZE_AFFINITY_MASK=1 tools/box.sh run './build/src/cli/b70-decode <mxfp4-snap> --bench --depth 4096 --tg 256 --profile --repeats 5'`
  for the per-kernel µs, then the record row itself in Task 5. Report the
  `gemv` family's ms/token against the int4 checkpoint's, at the same sha, and
  attribute any delta to a named cell rather than to "the format".
- [ ] **Step 8: the correctness gates for this task.**
  - `replay_determinism_test` on the MXFP4 checkpoint: **774 / 19**, bitwise
    across the three runs. The module count is arithmetic, not luck: the six
    `gemv_M1_…_L{0,1}` modules are replaced one-for-one by `…_L{2,3}` ones and
    nothing else in the walk depends on the format - write that sentence into
    the test's comment beside the existing int4/bf16 `lm_head` one.
  - `profile_capture_test`: 774 / 19 unchanged.
  - The tie-aware golden gate against `oracle-out-mxfp4` - **Task 5 Step 2
    produces that set; this task's gate run is scheduled there**, because one
    CPU oracle run per golden set is the expensive part and it is shared.
  - The **int4 path is untouched and stays tested**: the full existing suite
    green, `golden_gate_test` on `oracle-out`, and the `2a7df0b` decode rows
    re-verifiable. A single test moving on the int4 checkpoint is a defect in
    this task, not a consequence of the pivot.
- [ ] **Step 9: Commit** -
  `perf(kernels): MXFP4 decode re-measured - <per-shape verdict>, <X> t/s`
  (the commit message carries the measurement; the pre-registration is already
  committed and is not edited).

### Task 4: prefill - `gemm_mxfp4`, and the death of the dequant scratch

*This is the task ruling A20 exists for, and the one whose premise this plan
found false (**R1**). It is written so that Step 1 either confirms the finding
and the task proceeds on the mixed-precision route, or overturns it and the task
gets **easier**. Nothing below assumes which.*

**Files:**
- Create: `src/sycl/xe_mma_mxfp4.hpp`, `src/sycl/gemm_mxfp4.cc`,
  `tests/prefill/gemm_mxfp4_test.cc`,
  `docs/preregistration-mxfp4-prefill-2026-09-05.md`,
  `src/kernels/prefill/mxfp4_pack.cl` (route R-a only)
- Modify: `src/runtime/prefill/gemm.h`, `src/sycl/CMakeLists.txt`,
  `src/loader/loader.{h,cc}` (route R-b's second copy),
  `src/runtime/engine.cc` (`pf_linear`), `src/runtime/buffers.{h,cc}`,
  `tests/runtime/buffers_test.cc`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's layouts 2/3 and `DeviceWeight`; `xe_gemm_config.h`'s
  `TileShape` / `TiledMma` / epilogue / `PersistentScheduler`;
  `gemm_batched.cc`'s `check_operand` / `make_args` / queue plumbing;
  6b's `PrefillScratch` and `Engine::prefill` walk; 6c's `pf_linear`.
- Produces: `runtime::prefill::{MxfpWeight, gemm_mxfp4, gemm_mxfp4_available}`,
  the prefill-native weight source (R-b or R-a), an `Engine::prefill` with no
  `dequant_to_bf16` on the critical path, and a `PrefillScratch` **356,515,840 B
  smaller**.

- [ ] **Step 1: re-verify the six `sycl-tla` facts against the pin, and report
  before writing code.** The plan's *What this plan verified* table is the
  claim; this step is the check, and **it is the gate on R1**. Against
  `${B70_SYCL_TLA_SRC_DIR}` at `91e5bd7…`, paste into the task report:
  1. `include/cutlass/gemm/collective/xe_mma_blockscaled_mxfp.hpp` -
     `kSupportedElementA`'s list (`:138-143`) and whether `bfloat16_t` is in it;
  2. the same file's four `static_assert`s (`:168-179`), especially
     `kScaleALeftmostUnitStride`;
  3. `include/cute/arch/mma_xe.hpp` - the guard at `:254` and whether the
     `XE_BDPAS_TT` / `e2m1 DPAS` declarations sit inside it;
  4. `examples/50_xe35_block_scaled_gemm/CMakeLists.txt:29`'s
     `SYCL_TARGET_INTEL_GPU_CRI` gate;
  5. `include/cute/arch/reorder_xe.hpp` - that
     `Xe_Reorder<ReorderKind::UU, float_e2m1_t, bfloat16_t>` (`:1529`) and its
     `UV`/`VV` siblings are guarded **only** by `CUTE_ARCH_REORDER_XE_ENABLED`;
  6. `include/cutlass/gemm/collective/xe_mma_mixed_input.hpp` - the 4-bit
     `transform_quant`'s `is_quantization` expression (`:446-448`) and the
     `ModeHasScales && is_quantization` guard on the scale multiply (`:504-513`).
  **If (1)-(4) confirm the finding, R1 stands and Steps 3-5 are the route. If
  any of them has moved** - a bump, or a target macro this plan misread - **stop
  and report**: the blockscaled mainloop is then the cheaper route and Step 4
  collapses to an alias chain. Either way, also compile-check the claim rather
  than only reading it: add to `src/sycl/tla_smoke.cc` a `static_assert` on
  `cutlass::gemm::collective::detail::…`-free ground -
  `static_assert(!cute::is_same_v<decltype(cute::XE_BDPAS_TT<8, float, cute::bfloat16_t>{}), void>)`
  guarded by `#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)` - so
  the tree records, in a built artifact, which branch this build is on.
- [ ] **Step 2: the route ruling (R3), and the memory arithmetic that decides
  it.** Both routes are priced in *The numbers this plan is built on*; this step
  puts the residency in front of the controller and records the answer.

  | group | R-b (chosen) | R-a (fallback) | source |
  |---|---|---|---|
  | weights resident (MXFP4A16, int4 `lm_head` class) | 16,219,604,992 | 16,219,604,992 | measured, Task 1's report |
  | **prefill-native weight copy** | **+12,923,699,200** | 0 | derived: `Σ K·N/2 + Σ K·N/32` over the 256 matrices |
  | `PersistentBuffers` | 1,240,465,536 | 1,240,465,536 | pinned, `buffers_test` |
  | `DecodeScratch` (kM = 8) | 68,652,864 | 68,652,864 | pinned, `buffers_test` |
  | `PrefillScratch` (kC = 2048), `dequant` **freed** | 1,907,474,328 | 1,907,474,328 | 2,263,990,168 − 356,515,840 |
  | ⤷ R-a's relayout scratch | - | +94,699,520 | `gate‖up`: 89,128,960 nibbles + 5,570,560 scales |
  | **total** | **32,359,896,920 B = 30.138 GiB** | **19,530,897,240 B = 18.190 GiB** | derived |
  | **headroom against 34,241,150,976 B** | **1,881,254,056 B = 1.752 GiB** | **14,710,253,736 B = 13.700 GiB** | derived |
  | bridge cost per chunk | **0 ms** | 41.66 ms (roofline) / 83.87 ms (measured-scaled) | derived |
  | break-even GEMM ratio for 1973 t/s | **0.855** | 0.902 / 0.955 | derived |

  **Both fit; they buy different things.** R-b spends 12.9 GB to buy ~84 ms per
  chunk and a comfortable 0.855 break-even; R-a keeps 13.7 GiB of headroom and
  pushes the break-even to 0.955, which the GEMM may not clear. **Request R-b,
  with the copy allocated lazily on the first `prefill()`** exactly as 6b's
  ruling R7 allocates `PrefillScratch`, so a decode-only engine's residency is
  **byte-identical to today's** and spec §6.5 stays checkable rather than
  argued. Record the ruling in the task report before Step 6. Do not implement
  both on the critical path: R-a's kernel lands in Step 8 behind
  `B70_MXFP4_RELAYOUT` as the documented route for a `max_len` R-b cannot hold,
  and it is tested, not bound.
- [ ] **Step 3: the loader writes the prefill-native pair (R-b).**
  `DeviceWeight` gains two optional members and one accessor:
  ```cpp
  struct DeviceWeight {
    l0::Mem mem;                              // decode's canonical bytes (layout 0/1/2/3)
    std::unique_ptr<l0::Mem> scales;          // non-null iff layout 0 (f16) or layout 2 (e8m0)
    // The prefill operand, in the ONE layout sycl-tla's mixed-dtype mainloop
    // reads: e2m1 [N][K/2] (K contiguous, ColumnMajor from the GEMM's view) and
    // e8m0 [K/32][N] (N contiguous, the leftmost _1 of Stride<_1,int64_t,int64_t>).
    // Non-null iff kind == Mxfp4 AND the engine has called prefill() at least
    // once (ruling R3 / 6b R7): a decode-only Engine allocates exactly what it
    // allocated before the pivot, byte for byte. At layout 2 `pf_scales` ALIASES
    // `scales` -- the loader already writes that plane in the mainloop's own
    // [K/32][N] order -- so only the nibble transpose costs bytes.
    std::unique_ptr<l0::Mem> pf_packed;
    std::unique_ptr<l0::Mem> pf_scales;       // owning only at layout 3
    model::GemvShape shape;
    model::WeightKind kind;
  };
  ```
  Implementation: `loader::materialise_prefill_weights(LoadedModel&, l0::Context&)`,
  called once from `Engine::prefill`'s lazy init, walks
  `m.linears`, skips `kTopLevel` (`lm_head` runs one row on its existing route,
  spec §3.5) and every non-`Mxfp4` weight, and for each remaining linear
  **re-reads the checkpoint** through the same `SafetensorsSet` the load used -
  no, it does **not**: the mmap is gone by then. Instead the loader **stages the
  pair during `load_linear`** into a host-side `std::vector` owned by
  `LoadedModel` only if `B70_PREFILL` is enabled *and* the engine asked for it,
  which is a lifetime knot. **Resolve it the simple way: `loader::load` gains a
  `bool prefill_weights` parameter (default `false`)**, threaded from
  `Engine`'s constructor, and `load_linear` uploads the second pair in the same
  pass, from the same staging buffers, with one extra `imm.copy` per linear.
  A decode-only `Engine` passes `false` and allocates nothing; `b70-decode
  --bench --pp N` passes `true`. That keeps the mmap in scope, adds no new
  ownership, and makes the residency a **construction-time** property that
  `buffers_test` and the load report can both state. Record the parameter in
  `docs/13-loader.md`.
  The staging for the pair is `K·N/2 + K·N/32` bytes (`gate‖up`: 94,699,520),
  sized like every other staging buffer from the linears **this** load repacks;
  the load-report gains a `prefill %13zu B %7.3f GB (mainloop-native second
  copy; absent unless prefill is enabled)` line **outside** `read_per_token`
  (it is not streamed per token) but **inside** `total()`, so the buckets still
  account for every allocated byte.
- [ ] **Step 4: `src/sycl/xe_mma_mxfp4.hpp` - the gap, and only the gap.** One
  dispatch-policy tag and one `CollectiveMma` partial specialisation, structured
  as `xe_mma_mixed_input.hpp`'s is and differing from it in exactly one place.
  ```cpp
  // The MXFP4A16 mainloop: A bf16 x B e2m1 with a shared e8m0 exponent per 32
  // weights, upconverted IN REGISTERS -- no [K][N] materialisation, which is the
  // whole point of ruling A20.
  //
  // WHY THIS FILE EXISTS AT ALL (ruling request R1, verified at pin 91e5bd7):
  //   * cutlass/gemm/collective/xe_mma_blockscaled_mxfp.hpp is the NATIVE MXFP4
  //     mainloop, and it is not ours to use: kSupportedElementA excludes
  //     bfloat16_t (:138-143), it static_asserts a block-scale tensor on BOTH
  //     operands (:176-179), and it multiplies with XE_BDPAS_TT, declared only
  //     under `#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)`
  //     (cute/arch/mma_xe.hpp:254) -- Xe3.5 / CRI. bmg-g31 is Xe2.
  //   * cutlass/gemm/collective/xe_mma_mixed_input.hpp DOES run A-wide x B-4bit
  //     with group-wise scales on BMG, but its 4-bit transform_quant applies the
  //     scale only when `is_integer(Src) ^ is_integer(Dst)` (:446-448, :504-513).
  //     With Src = float_e2m1_t and Dst = bfloat16_t both are non-integer, so it
  //     drops the scale and casts the nibble as an integer. No choice of types
  //     fixes that, and transform_quant is a private member.
  // So we inherit everything -- the tile shape, the DPAS atom, the 2D block
  // loads, the prefetch pipeline, the scheduler, the epilogue, and Intel's OWN
  // e2m1 -> bf16 conversion sequence (cute/arch/reorder_xe.hpp:1512-1625, plain
  // vISA, not Xe3.5-gated) -- and write only the eight lines that apply the
  // shared exponent. Spec 2 §3.2's rule holds: inherit, own the gap.
  #pragma once
  #include "cutlass/gemm/dispatch_policy.hpp"
  #include "cute/arch/reorder_xe.hpp"
  #include "cute/atom/mma_atom.hpp"
  #include "cute/algorithm/gemm.hpp"

  namespace runtime::prefill {
  // A tag of our own rather than a reuse of MainloopIntelXeXMX16MixedPrecision:
  // partial-specialising CollectiveMma on Intel's tag would be an ODR hazard the
  // day they add the same specialisation.
  template <int Stages> struct MainloopXeMxfp4Bf16
      : cutlass::gemm::MainloopIntelXeXMX16<Stages> {};
  }  // namespace runtime::prefill

  namespace cutlass::gemm::collective {
  template <int Stages, class TileShape_, class ElementA_, class StrideA_,
            class ElementPairB_, class StridePairB_, class TiledMma_,
            class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_,
            class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
  struct CollectiveMma<runtime::prefill::MainloopXeMxfp4Bf16<Stages>, TileShape_,
                       ElementA_, StrideA_, ElementPairB_, StridePairB_, TiledMma_,
                       GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
                       GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
    using ElementB     = cute::remove_cvref_t<decltype(cute::get<0>(ElementPairB_{}))>;
    using ElementScale = cute::remove_cvref_t<decltype(cute::get<1>(ElementPairB_{}))>;
    static_assert(cute::is_same_v<ElementA_, cute::bfloat16_t>, "MXFP4A16: A is bf16");
    static_assert(cute::is_same_v<ElementB, cutlass::float_e2m1_t>, "MXFP4A16: B is e2m1");
    static_assert(cute::is_same_v<ElementScale, cutlass::float_ue8m0_t>,
                  "MXFP4A16: the block scale is e8m0");
    static constexpr int GroupSize = 32;   // the format's, not a parameter
    // ... Arguments / Params / to_underlying_arguments / can_implement, copied
    // in shape from xe_mma_mixed_input.hpp: B as make_shape(N, K, L) through a
    // cute::subbyte_iterator<const ElementB>, the scale as
    // make_shape(N, ceil_div(K, 32), L) with Stride<_1, int64_t, int64_t>.
    // can_implement adds: K % 32 == 0, and the 128-bit copy-alignment checks
    // with min_aligned_elements computed from sizeof_bits<ElementB> = 4, i.e.
    // 32 e2m1 elements -- NOT the bf16 path's 8.

    CUTLASS_DEVICE void operator()(/* accum, gA, gB, src_accum, k_tile_iter,
                                      k_tile_count, blk_coord, K_start,
                                      thread_idx, Params */) {
      // The loop body, differing from xe_mma_mixed_input.hpp's in ONE block:
      //   copy(copy_a, tAgA(_,_,_,k), tArA);              // bf16, inherited
      //   copy(copy_b, tBgB(_,_,_,k), tBrB);              // e2m1, inherited
      //   copy(copy_scale, scale_iter(_,_,_,k / k_reload), frag_scale);
      //   prefetch(...) x3;                               // inherited
      //   reorder(tArA, tCrA);                            // bf16 -> bf16, a move
      //   reorder(tBrB, tCrB_raw);                        // THE INHERITED UPCONVERT:
      //         cute::Xe_Reorder<ReorderKind::*, float_e2m1_t, bfloat16_t>,
      //         5 cycles per output register (reorder_xe.hpp:1529's own comment)
      //   --- the gap, and all of it ---
      //   CUTLASS_PRAGMA_UNROLL
      //   for (int n = 0; n < GemmIterN; ++n) {
      //     // e8m0 -> a bf16 multiplier. E == 0 is 2^-127 (subnormal); E == 255
      //     // is NaN and the loader refuses it, so no guard here. The scale is
      //     // per (n, k-block of 32) and SG_K >= 32 by construction, so it is
      //     // loop-invariant across the 32 k this iteration covers.
      //     const auto s = cutlass::bfloat16_t(cutlass::float_ue8m0_t::bitcast(
      //                        frag_scale(n)).to_float());
      //     CUTLASS_PRAGMA_UNROLL
      //     for (int e = 0; e < kPerN; ++e) tCrB(e, n) = tCrB_raw(e, n) * s;
      //   }
      //   --- end of the gap ---
      //   cute::gemm(tiled_mma, tCrA, tCrB, accum);       // inherited
    }
  };
  }  // namespace cutlass::gemm::collective
  ```
  **Three implementation notes that will decide whether this compiles the first
  time**, written down now so they are checked rather than discovered:
  (a) `partition_sg_fragment_B` defines `tCrB`'s layout; allocate `tCrB_raw`
  with the same `partition_sg_fragment_B` call and never assume an element
  order. (b) the scale copy is `scale_zero_copy_traits<float_ue8m0_t, SG_N>` -
  8-bit, so `XE_2D_U8x1x16_LD_N` or `..._U8x1x32_LD_N`
  (`xe_mma_mixed_input.hpp:82-91`); the `[K/32][N]` plane the loader writes is
  exactly what that atom reads. (c) `k_reload_factor = max(GroupSize / BLK_K, 1)`
  with `BLK_K = 32` gives 1 - the scale is re-read every k-tile, which is the
  simple case; if the tile shape's `BLK_K` changes, this constant changes with
  it. **Build the TU alone first** (`b70_tla_smoke`-style, one call, no test) so
  a template error is bounded to this file.
- [ ] **Step 5: `src/sycl/gemm_mxfp4.cc` - the chain and the wrapper.** The
  chain sits beside `xe_gemm_config.h`'s `Chain<>` and shares its `TileShape`,
  `TiledMma`, epilogue and **`PersistentScheduler`** verbatim - the §6.4
  no-atomics evidence is the same evidence, and saying so in one place is what
  stops the two from drifting:
  ```cpp
  using Mxfp4Mainloop = cutlass::gemm::collective::CollectiveMma<
      runtime::prefill::MainloopXeMxfp4Bf16<xe::kPipelineStages>, xe::TileShape,
      cute::bfloat16_t, cutlass::gemm::TagToStrideA_t<xe::LayoutA>,
      cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, StrideScale>,
      cute::tuple<cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>, StrideScale>,
      xe::TiledMma,
      void, void, void, cute::identity,
      void, void, void, cute::identity>;
  using Mxfp4Kernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, Mxfp4Mainloop, xe::CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  ```
  The wrapper mirrors `gemm_bf16_batched`'s exactly, with the alignment rules
  **re-derived for a 4-bit operand** rather than copied:
  ```cpp
  void gemm_mxfp4(Context& cx, GemmDims d, const uint16_t* A, const MxfpWeight& W, float* C) {
    require(d.M && d.K && d.N, "M, K and N must all be non-zero");
    require(d.K == W.K && d.N == W.N, "GemmDims and MxfpWeight disagree on K or N");
    // 128-bit copy granularity: 8 bf16 for A and C's 4 fp32, but THIRTY-TWO
    // e2m1 elements for B. K % 32 == 0 is also the format's own group bound.
    require(d.K % 32 == 0, "K must be a multiple of 32 e2m1 elements (and of the "
                           "block-scale group), got " + std::to_string(d.K));
    require(d.N % 8 == 0, "N (= ldc) must be a multiple of 8, got " + std::to_string(d.N));
    require(W.ldb % 32 == 0, "packed row pitch must be a multiple of 32 e2m1 elements");
    require(W.lds % 4 == 0, "scale row pitch must be a multiple of 4 bytes (the 2D block "
                            "load's surface-width rule)");
    check_operand(A, d.K, "A");
    require(reinterpret_cast<uintptr_t>(W.packed) % 64 == 0, "W.packed base is not 64-byte aligned");
    require(reinterpret_cast<uintptr_t>(W.scales) % 64 == 0, "W.scales base is not 64-byte aligned");
    check_operand(C, d.N, "C");
    // ... args as in make_args, with StrideB = (ldb, _1, 0) over (N, K, L) and
    // StrideScale = (_1, lds, 0) over (N, scale_k, L); can_implement and
    // get_workspace_size == 0 required by name; op.initialize(args, nullptr, &q);
    // op.run(&q). Asynchronous by contract: the caller syncs.
  }
  bool gemm_mxfp4_available() { return true; }   // false in the #else of Step 1's guard
  ```
  Every production shape satisfies these: `K ∈ {5120, 6144, 17408}` all divide
  by 32; `N ∈ {5120, 14336, 16384, 34816}` all divide by 8; `l0::Mem`'s default
  `align = 64` supplies both bases. Add `gemm_mxfp4.cc` to
  `add_library(b70_prefill SHARED context.cc gemm_batched.cc gemm_mxfp4.cc)` -
  **that is the only CMake change**; `-isystem`, the two `-D`s, the three
  `-spirv-ext` entries and the 256-GRF backend option are all target-wide
  already (`src/sycl/CMakeLists.txt:44-48, 97-115`).
- [ ] **Step 6: `tests/prefill/gemm_mxfp4_test.cc` - correctness, tolerance,
  determinism.** Structure and harness copied from `gemm_batched_test.cc`
  (4096 sampled `(m, n)` from `std::mt19937(0x6e6d31u)`, sorted, streamed back
  through a 64 MB staging pair; `kReplays = 8`, `kDropped = 3`, `kEnqueues = 4`
  so rates are comparable to P2's matrix).
  - **Shapes:** the five production `(K, N)` at `M ∈ {64, 2048}` - `(5120,16384)`,
    `(6144,5120)`, `(5120,34816)`, `(17408,5120)`, `(5120,14336)` - plus a
    `(256, 64)` case at `M = 8` that a human can read.
  - **The reference and the bar.** Accumulate in `double` over
    `common::Mxfp4Ct::at(k, n)` (exact: a table entry times an exact power of
    two) and `common::bf16_to_f32(A)` (exact). **The device's B operand is
    bf16, not fp32** - the mainloop upconverts e2m1 to bf16 and multiplies by a
    bf16 scale - so the bar has a term the bf16 GEMM's does not:
    ```
    |C_dev - C_ref| <= 64 * 2^-24 * S  +  2 * 2^-8 * S  +  2^-24 ,
                       └ fp32 accumulation ┘ └ the bf16 operand ┘
    S = sum_k |A[m][k] * W(k,n)|   in double
    ```
    **Derivation, stated because a tolerance without one is a guess.** Every
    e2m1 value is *exactly* representable in bf16 (three significand bits
    against bf16's eight), so the upconvert is exact. The e8m0 scale is an exact
    power of two, so `bf16(scale)` is exact. Their **product** is therefore
    exact too - `value × 2^e` only moves the exponent. **The `2·2^-8·S` term is
    the bar this task must therefore try to remove**: if the mainloop's scale
    application is a bf16 multiply of two exactly-representable operands, the
    only error left is fp32 accumulation and the bar is `gemm_bf16`'s. Step 6a
    settles it empirically: run with the loose bar, **print the measured max
    ratio against the tight bar**, and if every shape sits under 64·2^-24·S,
    tighten the test to the bf16 GEMM's own bar and record that MXFP4's operand
    path is exact. That is a stronger correctness statement than the loose bar
    and it costs one run.
  - **Determinism (spec §6.4):** every shape run twice into two distinct
    allocations, full-result `memcmp`. Bitwise, justified by the same
    `PersistentScheduler` evidence `xe_gemm_config.h` records.
  - **M-invariance:** `(2048, 17408, 5120)` and `(64, 17408, 5120)` over the
    same W and the first 64 rows of A - the first 64 output rows **bitwise
    identical**, because rows 0-63 sit in M-tile 0 either way.
  - **The int4-equivalence control, and what it is not.** Build one weight
    twice - as `Mxfp4Ct` and as the `Int4Gptq` whose dequantised values are
    *closest* - and compare `gemm_mxfp4` against `dequant_to_bf16 + gemm_bf16`.
    **Not** a bit-identity bar and not a tolerance bar: the two formats
    represent different numbers. It is printed as a **relative-RMS diagnostic**
    so the quality question A20 opens ("MXFP4 vs AutoRound-tuned quality is a
    NEW question, recorded as such") has a number attached from day one.
- [ ] **Step 7: pre-register, in a committed file, then measure.**
  `docs/preregistration-mxfp4-prefill-2026-09-05.md`, committed on its own
  before any timing:

  > **Acceptance (A20's, quoted): `gate‖up` at `M = 2048` ≥ `0.85 × 150.19` =
  > ≥ 128 TFLOP/s.** Mechanism: the in-register upconvert costs ALU that the
  > bf16 mainloop does not spend - `Xe_Reorder<e2m1, bf16>` is 5 cycles per
  > output register by its own comment, plus one bf16 multiply per element for
  > the scale - against a B operand that is **half the bytes**, so the shape is
  > moving from ~50% of the bandwidth wall toward the ALU.
  >
  > **The break-even table, so a pass at 0.85 is not mistaken for clearing the
  > §7 bar.** With `chunk(r) = 240.602 + 681.74/r + X` ms and
  > `t/s = 2048000/chunk`: at **X = 0** (route R-b, chosen) 1973 t/s needs
  > **r ≥ 0.855 (128.4 TFLOP/s)**; at X = 41.66 (R-a, roofline) **r ≥ 0.902
  > (135.5)**; at X = 83.87 (R-a, measured-scaled) **r ≥ 0.955 (143.5)**.
  > On R-b, therefore, the A20 acceptance and the spec bar coincide to within
  > 0.3%: **a pass at exactly 128 is a pass by 0.4 t/s, and that is a finding to
  > report, not a margin.**
  >
  > **The composed chunk, derived, with the dequant term at zero.**
  > `210.116 → 0`; every other measured term unchanged. At r = 1 the C = 2048
  > ceiling rises from the measured **1808.5-1817.3 t/s** to
  > **2220.5-2233.8 t/s (R-b)** / **2124.5-2136.7 (R-a roofline)** /
  > **2035.4-2046.5 (R-a measured-scaled)** - **103-113% of vLLM's 1973**,
  > against 92% today. Spec §3.0's "~2190-2230" is this plan's R-b row, and the
  > reconciliation is that §3.0 assumed no bridge and this plan prices one.
  >
  > **The standing risk, restated: the unchased −12.5% GEMM control (150.19 vs
  > 131.47).** If the production path really runs at 131.47, the GEMM term is
  > 777 ms and R-b's r = 1 ceiling is **2017 t/s**, R-a's measured-scaled is
  > **1859 - below the bar**. Step 7's measurement is taken **in the production
  > path** (`--pf-profile`), not in a probe harness, so this is answered rather
  > than deferred a third time.
  >
  > **Bars.** (1) `gate‖up` M=2048 ≥ 128 TFLOP/s → A20's acceptance met.
  > (2) ≥ 135.5 → clears 1973 even on R-a's optimistic bridge. (3) < 128 → the
  > gap-filling transform is the suspect; report the per-shape table, the
  > `.zeinfo` `grf_count` / `spill_mem_size` for the MXFP4 kernel (P2's own
  > diagnostic), and **stop** rather than tune. (4) determinism bitwise at every
  > shape. (5) workspace 0 B.

  Then run: `ZE_AFFINITY_MASK=1 tools/box.sh test gemm_mxfp4_test` for the
  correctness bars and the iterate-grade rates, and record the five production
  shapes at `M ∈ {64, 512, 1024, 2048, 4096}` beside P2's corrected bf16 matrix.
  **Label every rate iterate-grade, not a bench row** - this is not the P2
  harness, and the comparison is against P2's cells at the same `M`.
- [ ] **Step 8: the relayout kernel, as the priced alternative (R-a).** Behind
  `option(B70_MXFP4_RELAYOUT "build the per-chunk MXFP4 relayout" OFF)`.
  `src/kernels/prefill/mxfp4_pack.cl`, one entry point `pf_mxfp4_pack`,
  `-D K -D N -D LAYOUT`, reading decode's layout-2 plane or layout-3 tile and
  writing the mainloop's `[N][K/2]` + `[K/32][N]` pair. Grid `(N/16, K/64)` of
  16-lane work-groups, the same unit `gemv.cl` streams. At layout 2 the scale
  plane is already `[K/32][N]` and is **passed through by pointer, not copied**.
  Tested by `tests/prefill/gemm_mxfp4_test.cc`'s sixth case - run
  `gemm_mxfp4` over the relayout's output and require it **bitwise identical**
  to the same GEMM over the loader's R-b pair. That is a real bar: both paths
  must produce the same bytes, so a permutation bug cannot hide behind a
  tolerance. **Not bound by `Engine::prefill`.** Record its measured GB/s and
  ms/chunk against this plan's derived 41.66 / 83.87 - a **transposing** kernel
  has no right to the dequant's 293.06 GB/s and the measurement is the point.
- [ ] **Step 9: the swap in `Engine::prefill`, and the scratch that disappears.**
  `pf_linear` loses its first `wait()` and its scratch entirely:
  ```cpp
  // One MXFP4 linear of a prefill chunk. Ruling A20: no dequantisation, no
  // scratch, no cross-queue handoff before the GEMM -- the mainloop reads the
  // packed weights and upconverts in registers. Where plan 6c spent two
  // Context::wait() per linear (512 per chunk), this spends one (256): the GEMM
  // is the FIRST thing in the sequence, so wait #1 (L0 dequant -> SYCL GEMM) has
  // nothing to wait for. Wait #2 stays: the next L0 kernel reads the fp32 output.
  void Engine::pf_linear(uint32_t layer, model::LinearId id, const void* a, uint32_t C) {
    const loader::DeviceWeight& w = model_.linears.at({layer, id});
    if (w.kind == model::WeightKind::Mxfp4) {
      const prefill::MxfpWeight mw{w.pf_packed->as<uint8_t>(), w.pf_scales->as<uint8_t>(),
                                   w.shape.K, w.shape.N, w.shape.K, w.shape.N};
      prefill::gemm_mxfp4(*pf_cx_, {C, w.shape.K, w.shape.N},
                          static_cast<const uint16_t*>(a), mw, pf_->partials.as<float>());
    } else {
      // The int4 path stays fully supported: the checkpoint decides. Plan 6c's
      // dequant_to_bf16 + gemm_bf16, unchanged, including both waits.
      uint16_t* scratch = pf_->dequant.as<uint16_t>();
      prefill::dequant_to_bf16(*pf_cx_, w, scratch);
      pf_cx_->wait();
      prefill::gemm_bf16(*pf_cx_, {C, w.shape.K, w.shape.N},
                         static_cast<const uint16_t*>(a), scratch, pf_->partials.as<float>());
    }
    pf_cx_->wait();
  }
  ```
  **`PrefillScratch::dequant` becomes conditional, not deleted**, because the
  int4 path still needs it: allocate it only when the loaded model holds an
  `Int4` linear. `PrefillScratch`'s constructor gains that one flag from the
  `LoadedModel`, `bytes()` reports what was actually allocated, and
  `tests/runtime/buffers_test.cc` gains a **second** pinned total:
  - int4 checkpoint: **2,263,990,168 B**, unchanged (the existing assertion
    stands, which is how "the int4 path is untouched" is checked rather than
    claimed);
  - MXFP4 checkpoint: **1,907,474,328 B** = 2,263,990,168 − 356,515,840, with
    the per-field comment table gaining the `dequant … = 0 (MXFP4: the mainloop
    upconverts in registers)` row so the two totals are derivable from the same
    table.
  Nothing else in `PrefillScratch` moves: `partials`, the GDN fields and the six
  composed-attention rows are format-agnostic (A20: "Unaffected: everything L1
  builds").
- [ ] **Step 10: acceptance.** Full suite green on both checkpoints;
  `prefill_determinism_test` bitwise; `prefill_consistency_test`'s 64 identical
  tokens; `replay_determinism_test` 774 / 19. **Registered risk:** the
  consistency control compares prefill against ingest-by-decode on the *same*
  weights, and on the MXFP4 path prefill's B operand is bf16-exact (Step 6's
  finding) where decode's is fp32 - the **opposite** of plan 6c's registered
  risk, where prefill was the rounder side. If a token flips, **stop**: record
  the flip position, the prompt and the state-diagnostic band, and take it to
  the operator as a ruling request. Do not widen the bar.
- [ ] **Step 11: Commit** - three commits, in this order, because
  pre-registration precedes measurement and one defect gets one fix:
  1. `feat(prefill): gemm_mxfp4 - A20's mainloop, with the e2m1/e8m0 gap written`
  2. `docs(prefill): pre-register the MXFP4 GEMM bars and the recomposed ceiling`
  3. `perf(prefill): swap the MXFP4 linears onto gemm_mxfp4 - dequant scratch freed, <X> TFLOP/s`

### Task 5: gates and records

*Spec §6's bars, on the new format, plus the two `docs/BENCHMARKS` rows and the
mechanism sections §9 owes. The expensive item is the oracle re-anchor, and it
is scheduled here once and shared by Tasks 3 and 4.*

**Files:**
- Modify: `docs/BENCHMARKS.md`, `docs/12-kernels.md`, `docs/13-loader.md`,
  `docs/14-golden-gate.md`, `docs/16-know-how.md`, `docs/15-step-anatomy.md`,
  `tests/CMakeLists.txt`, `tools/oracle/README.md`
- Test: `golden_gate_test` (unchanged code, a new golden set),
  `prefill_gate_test` / `prefill_consistency_test` / `prefill_determinism_test`
  (6b's, unchanged), `replay_determinism_test`, `profile_capture_test`

**Interfaces:**
- Consumes: Task 2's `dump.py`; `tools/oracle/golden.sh`'s `OUT_DIR` /
  `ORACLE_SNAP` / `ORACLE_THREADS`; `golden_gate_test`'s three-argv contract and
  its `SKIP_RETURN_CODE 77`; `tools/bench_decode.sh`; ruling **R5**'s snapshot
  path; 6b's `oracle-out-long` and `--pp` / `--pp-chunk`.
- Produces: `oracle-out-mxfp4/`, two registered gate tests, the two
  `docs/BENCHMARKS` rows, and the four mechanism sections.

- [ ] **Step 1: disk, then the oracle set.** `tools/box.sh run 'df -h ~ | tail -1'`
  - the set is ~1 GB and the checkpoint ~14.6 GB; **STOP and report on any
  ENOSPC**, and do not delete another agent's artifacts. Then, detached, on the
  box:
  ```bash
  OUT_DIR=oracle-out-mxfp4 \
  ORACLE_SNAP=$HOME/models/qwen38-27b-mxfp4a16 \
  ORACLE_THREADS=28 setsid nohup tools/oracle/golden.sh \
    > oracle-out-mxfp4/golden.log 2>&1 </dev/null &
  ```
  **Cost, labelled:** three prompts × 32 greedy tokens, serially, model reloaded
  per prompt - **18 min 16 s measured on an idle box, 25 min 14 s measured under
  a 12-core compile** (docs/14). Use the exact `ORACLE_SNAP` string ruling R5
  fixed; `golden.sh`'s own header states why a golden set belongs to exactly one
  checkpoint and must not overwrite another's. Then `tools/oracle/check.sh` in a
  fresh process, and paste its three continuations into the task report - a
  golden set nobody read is not evidence.
- [ ] **Step 2: the short golden gate, on the new format (spec §6.1).**
  ```bash
  tools/box.sh run './build/tests/golden_gate_test "$PWD/oracle-out-mxfp4" \
    "$PWD/tests/golden/prompts" $HOME/models/qwen38-27b-mxfp4a16'
  ```
  Register it as `golden_gate_mxfp4_test` beside the existing one, same
  `LABELS "checkpoint;golden"`, same `SKIP_RETURN_CODE 77`, `TIMEOUT 1800`.
  **The bar is unchanged and is not negotiable: 100% of determined rows
  element-exact**, every undetermined row inside the golden argmax set, tails
  teacher-forced. Report the census the way docs/14 reports the other three
  (`TOTAL: n/n determined rows exact, k undetermined (a agree + b other member)`)
  and add the row to docs/14's table. **A determined-row failure is a finding,
  not a tolerance question** - the oracle and the engine now share Task 2's
  table, so a mismatch means the kernel disagrees with the format, and the
  first suspects in order are: the nibble order (`(x[1::2] << 4) | x[0::2]`),
  the sign of zero at code 8, the e8m0 subnormal at `E = 0`, and the two-half
  accumulation of Task 3 Step 1.
- [ ] **Step 3: the decode record row (spec §6.5 as amended by A20 - "decode
  re-measured on the new format").** Idle box, **DRM evidence pasted before and
  after** (`grep -l drm-driver /proc/*/fdinfo/*` returning nothing), clean
  worktree so the sha is not `-dirty`:
  ```bash
  MODEL=$HOME/models/qwen38-27b-mxfp4a16 tools/bench_decode.sh --runs 3 --depth 4096 --tg 256
  ```
  Add to `docs/BENCHMARKS.md` a **new `###` block above** "The record rows - all
  three checkpoints" and below "The spec-1.7 gate rows", in the file's
  newest-first order, using the existing 8-column format:

  | engine / checkpoint | depth | tg | runs (t/s) | **t/s** | **ms/token** | MBU | grade |
  |---|---|---|---|---|---|---|---|
  | **b70-decode `<sha>`, `qwen38-27b-mxfp4a16`** (MXFP4A16, e2m1 `lm_head`) | 4096 | 256 | … | … | … | … of **13.673 GB** | **record**, median of 3, idle |

  with the sentence that makes the row interpretable: *"the MBU denominator is
  **the same 13,672,613,888 B** the `qwen38-27b-w4g64-rtn` row uses - 4.25
  bits/weight in both formats - so this row and the 32.22 t/s row are directly
  comparable and the difference between them is arithmetic, not bytes."*
  Score it against Task 3's pre-registered `32.22 ± 0.5` **explicitly**, hit or
  miss, and if it misses, the per-shape GB/s table from Task 3 Step 6 goes in
  beside it.
- [ ] **Step 4: the prefill bars and the pp row (spec §6.2, §6.3, §6.4, §7).**
  - `prefill_gate_test` (6b's, unchanged code) on the MXFP4 checkpoint against
    `oracle-out-mxfp4`;
  - the **multi-chunk** gate at `--pp-chunk 1024` against `oracle-out-long`
    (6b's ≥ 2048-id set). **Cost: the long oracle set is hours-class
    (estimated; spec §6.2 says so, and this plan does not re-estimate it).** If
    6b's long set was produced on the int4 checkpoint it does **not** transfer -
    a golden set belongs to one checkpoint - and the MXFP4 long set is a second
    hours-class run. Schedule it detached, alongside Step 1, and say in the
    report which checkpoint each long set belongs to;
  - `prefill_consistency_test` - 64 identical tokens, prefill vs
    ingest-by-decode, with the state-diagnostic band recorded;
  - `prefill_determinism_test` - bitwise;
  - the pp row, record grade, idle box:
    `./build/src/cli/b70-decode $HOME/models/qwen38-27b-mxfp4a16 --bench --pp 4096 --tg 256`,
    parsed from the second stdout row
    `| b70-decode <sha> | pp | <N> | <ms total> | <t/s> |`. Add it to
    `docs/BENCHMARKS.md` beside the vLLM `pp4096` rows **with both labels** -
    *device-side pp, loader-excluded, first-token-inclusive* - and against the
    two external numbers it is answerable to: **vLLM 1973** (same box, GPTQ) and
    the third party's **2513.22 ± 68.47** (their engine, MXFP4). Note in one
    sentence that the file's standing "This engine has no prefill kernel yet …
    so there is no `pp4096` number to put beside vLLM's 1973" paragraph is now
    superseded, and edit it rather than leaving two claims in one file.
- [ ] **Step 5: the mechanism sections (spec §9).**
  - **`docs/12-kernels.md`** - a `gemv` subsection, *"MXFP4A16 - e2m1 × bf16 at
    block 32"*: what it computes, the 16-entry table and the branch-free
    construction held bit-identical to it, the e8m0 `ldexp` and why
    `-cl-denorms-are-zero` is now an arithmetic requirement, **the two layouts
    as the geometries of 0 and 1** with the 544-byte tile's new scale field
    spelled out, the two-half accumulation and what it does and does not change
    about order, the measured per-shape GB/s beside the int4 cells with the
    "same bytes, so directly comparable" note, and the retune verdict. Plus a
    prefill section for the mainloop: the inherited configuration with its
    file:line provenance, **the R1 finding written down as a rejected route
    with its evidence** (the Xe3.5 guard, the both-operands static_asserts, the
    `is_quantization` expression), the gap that was written, the `[N][K/2]` /
    `[K/32][N]` layout contract, and the `PersistentScheduler` no-atomics
    evidence shared with `gemm_bf16`.
  - **`docs/13-loader.md`** - the third spelling in the "Classification" table
    (`<prefix>.weight_packed` + `.weight_scale` → MXFP4A16, dtypes `U8`/`U8`,
    shapes `[N][K/2]` / `[N][K/32]`), the `input_activations` refusal and why a
    scheme *name* is not evidence, layouts 2 and 3 in "Layout and split-K come
    from the table", the byte-table paragraph from Task 1 Step 7, the
    `prefill_weights` load parameter, and the third loader report.
  - **`docs/16-know-how.md` §5** - amend the MXFP4 bullet in place rather than
    appending a second claim: keep *"4.25 bits/weight effective - identical to
    GPTQ g64 (4 + 16/64 = 4 + 8/32)"*, which this plan's decode row confirms,
    and extend it with what was learned: *"Its home is fp4 tensor-core compute
    - and on hardware without one, its real value is that block scales let a
    mixed-dtype mainloop upconvert in registers, which is what removes a
    materialised dequant scratch. On Xe2 the native block-scaled path is
    Xe3.5-only; the in-register conversion sequence is not, and that asymmetry
    is the whole engineering story."*
  - **`docs/15-step-anatomy.md`** - the decode anatomy's `gemv` rows
    re-measured on the format, and the prefill anatomy's dequant row replaced
    by the bridge row (0 ms on R-b, the measured relayout on R-a), carrying the
    same three warnings the decode section does.
  - **`docs/14-golden-gate.md`** - the `oracle-out-mxfp4` verdict block and its
    tie census, in the shape of the three that are there.
- [ ] **Step 6: the int4 path is still fully supported, and it is checked, not
  claimed.** In one commit, re-run and paste: the full suite on the int4
  checkpoint; `golden_gate_test` against `oracle-out`; `replay_determinism_test`
  and `profile_capture_test` at 774 / 19; `buffers_test`'s **unchanged**
  2,263,990,168 total; and the two `2a7df0b` decode rows re-measured at this
  plan's final sha, inside day drift (≤ 0.09%) or explained. Write the sentence
  into `docs/13-loader.md` and `docs/12-kernels.md`: **the checkpoint decides,
  at every layer, by content; the engine holds both formats and neither is a
  build-time choice.**
- [ ] **Step 7: Commit** -
  `docs(mxfp4): decode and pp record rows, mechanism sections, gate verdicts`

---

## Plan self-review (2026-09-05, at authoring)

**Spec and ruling coverage.**

- **A20, consequence by consequence.** *Prefill GEMM* → Task 4 (the
  instantiation, the freed 356,515,840 B, `dequant_to_bf16` kept as a
  probe/test and off the critical path). *Loader* → Task 1 (the
  compressed-tensors reader alongside the GPTQ/AutoRound one; the checkpoint
  decides). *Decode* → Task 3 (the `gemv.cl` unpack variant, block 32 vs group
  64 spelled out as the two-half accumulation, **the retune re-measured with a
  committed pre-registration and a priced rider**). *Correctness* → Task 2 (the
  oracle re-anchor) + Task 5 (the gate on `oracle-out-mxfp4`, semantics
  unchanged; the MXFP4-vs-AutoRound quality question recorded with a number by
  Task 4 Step 6's diagnostic). *Unaffected* → nothing in Tasks 1-5 touches
  buffers' chunk-scaled rows, `Context`, `gdn_chunk`, `Engine::prefill`'s walk,
  attention or the CLI. *The artifact* → `tools/quantize_qwen38_mxfp4.sh`,
  consumed as-is, its verifier's assertions independently re-asserted by the
  loader.
  **One consequence is contradicted on evidence and filed as R1**: A20's
  `xe_mma_blockscaled_mxfp.hpp` route is Xe3.5-only and cannot take a bf16 A.
  The plan carries the work either way and no task writes code against the
  overturned premise.
- **A20 addendum.** The scheme string is never read; `MXFP4A16` vs `MXFP4` is
  decided by the **presence of `input_activations`**, which is the addendum's
  own test and the script's. The on-disk layout it fixes -
  `weight_packed` U8 `[N][K/2]`, `weight_scale` U8 e8m0 `[N][K/32]`,
  `quant_method "compressed-tensors"`, one `config_groups` entry with
  `{num_bits 4, type float, group_size 32, symmetric true}` - is exactly what
  Task 1 Steps 2-3 parse and classify, and the ignore list is handled the way
  the GPTQ path handles Vishva's: the model description says which linears are
  quantised and `load_linear` throws if a part classifies the other way.
- **§3.1 / §3.2, as superseded.** §3.1's coupling (decode's tuned layouts vs the
  mainloop's) is **not** dissolved by the pivot - it is re-priced. Task 4
  Step 2 states the three routes and their arithmetic instead of implying the
  problem went away; §3.2's "inherit the plain GEMM" survives with one honest
  amendment (R1), and `dequant_to_bf16` / `pf_dequant_tile` survive as the int4
  path's production code, not as dead weight.
- **§6.** 6.1 → Task 5 Step 2 (both formats; the int4 gate re-run in Step 6).
  6.2 → Task 5 Step 4, with the long set's **hours-class** cost labelled and
  the "a golden set belongs to one checkpoint" consequence stated rather than
  assumed away. 6.3 → Task 5 Step 4, with Task 4 Step 10's registered risk
  naming the *direction* of the numerics change. 6.4 → Task 3 Step 8 and Task 4
  Step 6 (bitwise both sides; `PersistentScheduler` named, not defaulted; no fp
  atomics). 6.5 → Task 3 Step 8 and Task 5 Steps 3 and 6, amended by A20 to
  "decode re-measured on the new format" with `32.22 ± 0.5` pre-registered.
  6.6 → Global Constraints, and 774 / 19 re-derived rather than re-asserted.
- **§7 / §9.** The pp row and the decode row with both labels, the four
  mechanism sections, and the ceiling recomposed from measured terms with its
  break-even table - Task 5 Steps 3-5.
- **§8.** Quoted verbatim at the top; the box workflow, JOBS 44, the pinned
  `sycl-tla` re-verified as Task 4's first step, `ZE_AFFINITY_MASK=1`,
  measured/derived/estimated on every number, nothing pushed, and chunk widths
  untouched (this plan changes no chunk boundary, so prefix caching stays
  reachable).
- **§10.** No task writes a DPAS kernel in OpenCL C, touches inline vISA, adds
  prefix caching, batching, fp8 KV or MTP. The one piece of device-code
  authorship is Task 4's B-side transform, and it is `cute::Xe_Reorder` -
  **Intel's own** conversion sequence - with an eight-line scale application
  around it.

**Placeholder scan.** No TBD, no TODO, no "similar to". Every created file has
its content or its exact structure here; every command is runnable as written.
The five places that depend on an answer this plan cannot have - R5's snapshot
directory (Task 1 Step 1 records it), R3's route (Task 4 Step 2 states both
totals and both break-evens), Task 4 Step 1's re-verification (both branches
written out, and the harder one is the default), Task 4 Step 6's tolerance
(loose bar with the tight ratio printed, then tightened on evidence), and
whether 6b's long oracle set belongs to the int4 checkpoint (both cases
scheduled) - each states **both branches concretely** and says which to record.
`<sha>`, `<X>` and `<per-shape verdict>` appear only in commit messages and
table cells that are measurement outputs.

**Type consistency vs `interfaces.md`.** `runtime::prefill::{Context, GemmDims,
gemm_bf16, gemm_bf16_batched, GemmBatch, dequant_to_bf16}` are used exactly as
fixed; `MxfpWeight` and `gemm_mxfp4(Context&, GemmDims, const uint16_t*, const
MxfpWeight&, float*)` follow A20's spelling with R2's amendment
(`ldb`/`lds` instead of `layout`) requested, not applied. `loader::DeviceWeight`
keeps its four fields and gains two nullable ones, with R4 requesting the
amendment to A5's "non-null iff layout 0" sentence. `model::WeightKind` gains
one enumerator and `GemvShape::layout` two values; `GemvShape{K,N,S,layout}`,
`kernels::gemv_variant(M,K,N,S,L)`, `PrefillScratch::kC`,
`Engine::prefill(const std::vector<uint32_t>&, uint32_t)` and the bf16
row-major `[M][K]` / fp32 `[M][N]` `ldc = N` conventions are untouched. Host code
stays in `runtime::prefill`, new OpenCL C keeps the `pf_` prefix and lives in
`src/kernels/prefill/`, and the one new SYCL header/TU pair lives in
`src/sycl/` - all as specified.

**The three things this plan deliberately does not do.** It does not chase the
unchased −12.5% GEMM control (the no-tuning rule; Task 4 Step 7 measures in the
production path so the answer arrives as evidence rather than as a project). It
does not attempt R-c, a project-owned CuTe copy atom over our 544-byte tile -
recorded and rejected, because `partition_sg_fragment_B`'s fragment order is
undocumented and a wrong guess there is silent. And it does not delete one line
of the int4 path: Task 5 Step 6 is the check that "the checkpoint decides" is a
property of the tree and not a sentence in a plan.
