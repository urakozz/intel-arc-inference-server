# Spec 2 - Plan 6c: Stage 1 / L2 - the GEMM swap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land stream S1 in production - `dequant_to_bf16` + `gemm_bf16` (Intel's
`sycl-tla` bf16 GEMM behind a plain interface) replacing the temporary widened
GEMV that plan 6b installs in `Engine::prefill()`, at chunk width **C = 4096**
(fallback 2048) - under the §6 correctness bars, with the per-kernel attribution
and the `--pp` before/after that price it. This is spec §5's L2: *"where the
order-of-magnitude lands."*

**Architecture:** One reusable bf16 scratch cuts the one coupling (spec §3.1):
plan 6a's OpenCL C `pf_dequant_tile` reads decode's *untouched* layout-0/1 tiles
and writes row-major bf16 `[K][N]`; Intel's Xe GEMM
(`XE_DPAS_TT<8,float,bfloat16_t>`, `Shape<_256,_256,_32>`, 8×4×1 subgroups,
`PipelineStages = 2`, no SLM) consumes it with A row-major `[M][K]` and writes
fp32 `[M][N]` at `ldc = N` - which **is** `PrefillScratch::partials`, the `S = 1`
rectangle plan 6b's ruling R1 already built for exactly this. So the swap is a
substitution, not a refactor: five tasks land it (wrapper, dequant binding, the
substitution, the memory proof, the gates) and two price it (the overlap lever,
the attribution).

**Tech Stack:** OpenCL C 3.0 + ocloc AOT (dequant), `icpx 2026.1.1` +
`sycl-tla` header-only (revision per ruling R5), C++17, pure Level Zero runtime,
ctest on the box (box, `tools/box.sh`).

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - read
§3.1, §3.2, §3.6, §5 (L2), §6, §8 first; they govern every task.

**Interface contract:** `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md`
- binding, but **plans 6a and 6b are ahead of it and win where they differ**.
Every consumed name is listed under *Interfaces: Consumes* below, with the
corrections this plan requests (R1, R2b, R3, R5) and the ones it adopts from 6a
and 6b rather than re-requesting. It renames nothing silently.

**Sibling plans:** **6a** - `docs/superpowers/plans/2026-09-04-plan6a-spec2-stage0-probes.md`,
**read Task T0 and Task P3 before starting**: it owns the `sycl-tla` pin, the
whole icpx build, `runtime::prefill::Context`, and the `pf_dequant_tile` kernel
this plan promotes. Where 6a and `interfaces.md` disagree, 6a wins and this plan
follows 6a. Then **6b** -
`docs/superpowers/plans/2026-09-04-plan6b-spec2-L1-plumbing-gdn.md`, **read its
"Rulings this plan makes" (R1-R8) before starting**: the buffer split (including
`PrefillScratch::kC = 4096` and the `dequant` field), the runtime-`M` kernels,
`Engine::prefill()`, `--pp` / `--pp-chunk`, the chunked GDN, the three prefill
tests, the `oracle-out-long` goldens, and the temporary widened GEMV this plan
retires. Then 6d (S3 L3 -
prefill attention), 6e (the §7 gate and `docs/BENCHMARKS` rows). **This plan
writes no BENCHMARKS record row.**

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
  `tools/box.sh build` / `tools/box.sh test [regex]` with `JOBS=44`. Nothing
  in this plan is verified by reading it.
- **`-Wall -Wextra -Werror`, C++17.** CUTLASS/cute headers are included with
  `-isystem`, never `-I`, so header diagnostics cannot be silenced by
  weakening our own warning set.
- **No fp atomics anywhere on the prefill path.** The `sycl-tla`
  configuration is pinned to `cutlass::gemm::PersistentScheduler` (explicit,
  not defaulted); any stream-K / split-K scheduler is disallowed by the
  §6.4 bar and by Task 1's determinism test.
- **`-cl-denorms-are-zero` forbidden** (`cmake/ocloc.cmake:41` fatals);
  `-cl-fp32-correctly-rounded-divide-sqrt` stays the ocloc default.
- **Decode is untouched.** No file under `src/kernels/{gemv,gemv_bf16,prep,
  attn,gdn_step,embed_gather,argmax}.cl`, no change to `src/runtime/capture.cc`'s
  walk, and the decode invariants **774 launches / 19 modules** stay exactly
  where `tests/runtime/replay_determinism_test.cc` and
  `tests/runtime/profile_capture_test.cc` pin them. The full suite is green
  before any commit touching `src/`.
- **GPU work under `ZE_AFFINITY_MASK=1`** whenever a container holds card 0.
  Never touch docker; never kill a process.
- **Pre-registered predictions before any timing.** The number is written into
  the task's report section first, then the run happens. This binds Tasks 4
  and 7 absolutely.
- **Commits** end with
  `Claude-Session: `.

---

## The numbers this plan is built on

Every row says its kind. Nothing here is measured *by this plan* - Tasks 4 and 7
produce this plan's own measurements.

| quantity | value | kind | source |
|---|---|---|---|
| vLLM bar | 1973 t/s pp4096, HTTP-inclusive | measured (external) | `docs/BENCHMARKS.md:122` |
| today (no prefill) | 121 s / 4096 ids = 29.5 ms/id | measured | bench log `2a7df0b`, RTN |
| device memory | 32656 MB = 34,241,150,976 B = **31.891 GiB** | measured | `docs/01-hardware.md:14` |
| measured DRAM bandwidth | 590 GB/s | measured | `docs/05-perf-model.md` |
| RTN weights resident | 13,672,613,888 (read/token) + 2,542,796,800 (embed) + 4,194,304 (rope) = **16,219,604,992 B = 16.220 GB** | measured | `docs/13-loader.md:654`, `:377-378` |
| Vishva weights resident | 18,086,971,392 B = 18.087 GB | measured | `docs/13-loader.md:379` |
| GEMM FLOP per 4096-id chunk (int4 linears only) | **199,286,482,534,400 = 199.29 TFLOP** | derived, this plan (§Task 5) | `2·C·K·N` over `model::Qwen35`'s table |
| dequant traffic per chunk | **61,575,823,360 B = 61.576 GB** | derived, this plan (Task 2) | `K·N·(17/32)` read + `K·N·2` write, summed over the layer mix |
| dequant time per chunk @ 590 GB/s | **104.4 ms** | derived | the row above ÷ 590e9 |
| `sycl-tla`, as read for this plan's API facts | `2db1b7c94cadc52decf0013bd8b6e244fcb37dd4`, tag `v0.9.2-9-g2db1b7c9`, BSD-3-Clause, header-only (494 `.h` + 361 `.hpp` + 40 `.inl`, two INTERFACE CMake targets) | measured (read 2026-09-04) | `git -C ~/PycharmProjects/sycl-tla log -1` |
| `sycl-tla`, as plan 6a pins it | `87f6850680a580654b9ea2c80dbc01aeb36ad231` | read 2026-09-04 | 6a `cmake/prefill.cmake`, `B70_SYCL_TLA_REVISION` |
| box toolchain | `icpx 2026.1.1 (2026.1.1.20260724)`, `ocloc 26.27.39122.14`, 44 cores, 121 GB RAM | measured (ssh, 2026-09-04) | this plan's authoring session |
| decode gate rows to re-verify | 32.22 t/s (RTN) / 29.33 (Vishva) at `2a7df0b` | measured | `docs/BENCHMARKS.md` §"spec-1.7 gate rows" |

**Reconciliation - the two `sycl-tla` revisions.** The table above carries two,
deliberately: the one this plan's API facts were read from and the one plan 6a's
build pins. They are **not** two values for one quantity - they are two
revisions, and exactly one of them will be the build's. **Ruling request R5
settles which, and Task 1 Step 1 re-verifies the four load-bearing facts against
the winner before any code is written.** Nothing below depends on which wins;
what depends on it is whether Task 1 Step 1 is a formality or a redesign.

**Reconciliation - the dequant overhead.** Spec §3.1 prices the scratch at
"~1.2 ms for that matrix" (gate‖up) and "8-16% of chunk time at C = 4096-2048".
That figure counts 356 MB *both ways*. The read side is int4 + inline f16
scales, `K·N·17/32` = 94,668,800 B for gate‖up, so the correct traffic is
**451,184,640 B → 0.765 ms at 590 GB/s (derived)**, and the whole-chunk share is
**4.5% at C = 4096 / 8.6% at C = 2048 (derived)**, not 8-16%. Both figures are
derived; **P3's measured GB/s supersedes both**, and Task 2 records the
comparison. Nothing in this plan's design depends on which is right - the
double-buffer lever (Task 4) is priced from P3's number either way.

**Reconciliation - 48.65 vs 48.97 GFLOP/token.** `interfaces.md` quotes 48.97
GFLOP for the whole forward pass. This plan's 199.29 TFLOP / 4096 = **48.653
GFLOP/token** covers only the six int4 linears it swaps; the 0.32 GFLOP
difference is attention, the GDN recurrence, the norms, `a‖b` and `lm_head`.
Two quantities, not two values.

---

## Interfaces: Consumes

Everything below is another plan's product. This plan states exactly what it
assumes and, where an assumption can be wrong, names the concrete fallback.

**From plan 6a (S1, Stage 0) - read `docs/superpowers/plans/2026-09-04-plan6a-spec2-stage0-probes.md`
Task T0 and Task P3 before starting. 6a is further along than `interfaces.md`
and this plan consumes 6a's shapes, not `interfaces.md`'s, wherever they differ.**

- **`runtime::prefill::Context`** (`src/runtime/prefill/context.h`, 6a T0) -
  **SYCL-free by design**: a PIMPL over an L0 immediate command list created
  `ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS` (6a T0 Step ~554-562) plus a SYCL
  in-order queue built from the same `ze_context`/`ze_device`.
  `Context(ze_context_handle_t, ze_device_handle_t)`,
  `explicit Context(l0::Context&)`, `void wait()` (drains both, via
  `zeCommandListHostSynchronize` + the SYCL queue),
  `void launch(l0::Kernel&, uint32_t gx, uint32_t gy, uint32_t gz,
  std::initializer_list<KernelArg>)` (and a `ze_kernel_handle_t` overload),
  `void* sycl_queue_raw()`, `ze_context()`, `ze_device()`, `ze_list()`.
  The typed queue is reached only from icpx TUs, through
  `src/runtime/prefill/context_sycl.h`'s free function
  `sycl::queue& runtime::prefill::sycl(Context&)`. **Two consequences this plan
  is built on:** (1) g++ TUs - `engine.cc`, `gemm.cc`'s dequant half - include
  `context.h` freely, so no `context_fwd.h` is needed and `interfaces.md`'s
  `sycl::queue& sycl()` member is superseded by 6a's requested deviation;
  (2) the L0 list is **asynchronous**, so `Context::launch` returns before the
  kernel completes and every cross-queue dependency needs an explicit
  `wait()` - Task 3 Step 4 spends exactly two per linear and Task 4 is the lever
  that removes one of them.
- **`cmake/prefill.cmake` + `src/sycl/CMakeLists.txt`** (6a T0) - the
  `B70_PREFILL` component (`AUTO`/`ON`/`OFF`), an `ExternalProject` sub-build
  whose `CMAKE_CXX_COMPILER` is `${B70_ICPX}` producing
  `${CMAKE_BINARY_DIR}/sycl-install/lib/libb70_prefill.so`, the `sycl-tla` pin
  (`B70_SYCL_TLA_SRC_DIR` + `B70_SYCL_TLA_REVISION`), and the consumer helper
  `b70_link_prefill(<target>)` (links the .so, sets `BUILD_RPATH`, orders the
  sub-build). **This plan adds files to 6a's sub-project and calls 6a's helper;
  it creates no CMake module of its own and does not attempt a g++ link of icpx
  objects - 6a's module rejects that route by name and gives the reason.**
- **`pf_dequant_tile`** (6a P3) - `src/kernels/prefill/dequant.cl`, one entry
  point, `-D K -D N -D LAYOUT`, grid `(N/16, K/64)` of 16-lane work-groups (one
  544-byte layout-1 tile per subgroup, the same unit `gemv.cl` streams), the
  xor/shift dequant and an `rne_bf16` transcribed from `common/bf16.h`. Variants
  `pf_dequant_tile_K{K}_N{N}_L{L}`, registered in
  `src/kernels/prefill/CMakeLists.txt` at **both** layouts for all five
  production shapes plus `lm_head` plus a 256×64 test shape. `tests/prefill/dequant_test.cc`
  exists and holds it **bit-exact at 256×64, both layouts**.
  **This plan does not write that kernel.** Task 2 promotes it: the host wrapper,
  the production-scale bit-exactness, the scratch bound, and one conditional
  geometry lever if P3 comes in short.
- **P2's measured figures**: TFLOP/s per production shape at
  M ∈ {512, 1024, 2048, 4096} and the confirmation that the chosen config has no
  split-K atomics. **P3's measured figures**: GB/s and ms per matrix, and P3
  Step 9's double-buffering answer. Tasks 4 and 7 substitute these into their
  pre-registered predictions *before* running.

**From plan 6b (S4, L1) - read `docs/superpowers/plans/2026-09-04-plan6b-spec2-L1-plumbing-gdn.md`
"Rulings this plan makes" (R1-R8) before starting. 6b has already made several
decisions this plan was going to make, and it wins.**

- **6b R1 - the prefill path already runs at `S = 1`.** `PrefillScratch::partials`
  is one `fp32 [kC][34816]` rectangle (570,425,344 B) and every consumer
  (`pf_res_fold`, `pf_silu_mul`, `pf_gated_head`, `pf_attn_prep`, `gdn_chunk`)
  folds **one** slice. **Consequence for this plan: there is no split-K to
  collapse and no fold-count-1 variant to add - the GEMM's fp32 `[M][N]` output
  at `ldc = N` *is* `partials`.** The swap is a substitution, nothing more.
- **6b R4 - `gdn_chunk` takes the fp32 `[C][16384]` partials directly** and does
  the single `rne_bf16` inside, where `gdn_step.cl:253` does it. There is no
  fold-and-round step and no materialised bf16 `x_qkvz`; this plan must not add
  one.
- **6b R2 - each producer's row stride equals the consuming linear's `K`**
  (`x` at 5120 or 17408, `mixer_out` at 6144). That invariant is `gemm_bf16`'s
  `lda`, which is why Task 3's `A` pointer is `x` for four linears and
  `mixer_out` for OutProj/OProj.
- **6b R5 - the L1 attention is `pf_attn_prep` + decode's unmodified
  `attn_decode`/`attn_reduce` at an `M = 64` binary, already driven in 64-query
  sub-chunks** (`PrefillScratch::kAttnC = 64`, `attn_part` = 405,798,912 B),
  behind `attn_prep_l1` / `attn_l1` in `src/runtime/prefill/attn_l1.h`.
  **Consequence: there is no sub-batch bridge for this plan to build** - a
  4096-wide chunk already executes; plan 6d retires the route.
- **6b R7 - `PrefillScratch` is allocated lazily** on the first `prefill()`
  call, so decode-only residency is byte-identical to today's.
- **6b Task 1 - `PrefillScratch::kC = 4096` already**, with `kAttnC = 64`,
  `kGdnChunk = 64`, a `dequant` field of exactly 356,515,840 B labelled "plan
  6c's; unused in L1", and `bytes() == 1,961,713,560` pinned in
  `tests/runtime/buffers_test.cc` alongside `PersistentBuffers::bytes() ==
  1,240,465,536` and `DecodeScratch::bytes() == 68,652,864`. **Consequence:
  Task 5 moves no constant and allocates no buffer** - it adds the whole-device
  total those three do not cover.
- **6b request #3 - `--pp-chunk C`** (default `PrefillScratch::kC`) is already
  requested by 6b, with the two-row stdout shape. **This plan's earlier `--chunk`
  request is withdrawn in favour of 6b's spelling.**
- `Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0)` and its
  per-layer walk, mirroring `runtime::Capture`'s (`src/runtime/capture.cc:469-590`):
  GDN layer = res_norm → **QkvZ** → `a‖b` → `gdn_chunk` (ending at
  `pf_gated_head` → `mixer_out`) → **OutProj** → mlp(res_norm → **GateUp** →
  silu·mul → **Down**); FA layer = res_norm → **Qkv** → attention (ending at
  `attn_reduce` → `mixer_out`) → **OProj** → mlp. The six bold names are the
  int4 linears this plan swaps.
- The temporary `pf_gemv_int4_M` (`src/kernels/prefill/pf_gemv.cl`, variants
  `pf_gemv_K{K}_N{N}_L{L}`, `tests/prefill/pf_gemv_test.cc`), which 6b's own
  commit message calls "L1 temporary, retired by plan 6c" - **Task 3 Step 3 is
  that retirement**. `pf_gemv_bf16.cl` / `pf_ab_proj` stay.
- `tests/prefill/prefill_gate_test.cc`, `prefill_consistency_test.cc`,
  `prefill_determinism_test.cc`, and the registered `oracle-out-long/` golden set
  (one prompt ≥ 2048 ids, §6.2).
- `b70-decode --bench --pp N [--pp-chunk C]` and its device-side `pp` row.

**From the existing tree (verified by reading, 2026-09-04):**

- `loader::DeviceWeight` (`src/loader/loader.h:19-26`) - **this is the struct
  `interfaces.md` calls `loader::Linear`; that name does not exist.** Fields:
  `l0::Mem mem` (the canonical bytes), `std::unique_ptr<l0::Mem> scales`
  (non-null **iff** layout 0), `model::GemvShape shape` = `{K, N, S, layout}`,
  `model::WeightKind kind`. Reached as `model_.linears.at({layer, id})`.
- `model::Qwen35`'s production layout/`S` map (`src/model/qwen35.cc:36-65`,
  Task 4's retune): QkvZ `{5120,16384,S=1,L=1}`, OutProj `{6144,5120,4,0}`,
  GateUp `{5120,34816,8,0}`, Down `{17408,5120,4,0}`, Qkv `{5120,14336,2,0}`,
  OProj `{6144,5120,4,0}`. **Five distinct `(K,N,layout)` triples** for six
  linear ids (OutProj and OProj share one).
- The dequant contract, quoted exactly from `src/kernels/gemv.cl:53-62`:
  ```c
  const uint u = word ^ 0x88888888u;
  a += (float)(((int)(u << 28)) >> 28) * bf16f(xv.s0);
  ```
  i.e. `signext4(q ^ 8) == q - 8` for every `q ∈ [0,16)`, nibble `j` reached by
  `((int)(u << (28 - 4*j))) >> 28`, scale one f16 per 64 consecutive `k`.
  Equivalent host reference: `common::Int4Gptq::at(k,n)` = `float(q - 8) *
  f16_to_f32(scales[(k/64)*N + n])` (`src/common/int4.h:19-23`).
- Layout 0 = GPTQ-native `qweight[K/8][N]` u32 + a **separate** `scales[K/64][N]`
  f16 allocation. Layout 1 = per `(n_tile of 16, k_group of 64)` one **544-byte
  tile**: 128 u32 of nibbles ordered `[k_octet j][lane l]`, then the 16 f16
  scales packed two per u32 (even lane in the low half); tiles ordered
  k-group-inner, n-tile-outer (`docs/12-kernels.md:214-226`,
  `src/common/repack.h:11-30`).
- `l0::CmdList::immediate` is `ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS`
  (`src/l0/cmdlist.cc:10`) - every append completes before the call returns.
  Task 4 adds the asynchronous companion it needs.
- `l0::CmdList::launch(Kernel&, gx, gy, gz, Event* signal)` already takes a
  kernel-timestamp signal event (`src/l0/cmdlist.h:29-30`,
  `src/l0/event.h:46-66`), which is the attribution instrument Task 7 uses.

**From `~/PycharmProjects/sycl-tla` (read-only, read 2026-09-04):**

- `examples/00_bmg_gemm/00_bmg_gemm.cpp:344-433` - the config this plan copies.
  `LayoutA = LayoutB = LayoutC = LayoutD = cutlass::layout::RowMajor` (`:350-353`).
  **B row-major `[K][N]` with `ldb = N` is the shipped, native path**; the VNNI
  packing DPAS needs is done by the 2D block load in hardware
  (`media/docs/cpp/xe_rearchitecture.md:5`), not in memory. **`interfaces.md`'s
  `[K][N]`/`ldb = N` statement stands unchanged - no ruling needed.**
- `examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp:248,277,293` - the
  non-default-queue variant: `gemm_op.initialize(args, nullptr, &q)` then
  `gemm_op.run(&q)`.
- `include/cutlass/gpu_generics.h:355` - `using cudaStream_t = sycl::queue *;`.
  This is how our own interop queue reaches the GEMM: every `cudaStream_t`
  parameter is literally a `sycl::queue*`, and
  `gemm_universal_adapter.h:551` reads
  `sycl::queue q = stream ? *stream : compat::get_default_queue();` - with a
  non-null stream the compat default queue is never touched.
- `include/cutlass/gemm/kernel/xe_gemm.hpp:82-87` - `static_assert(is_void_v<TileScheduler_>
  or is_same_v<TileScheduler_, PersistentScheduler>, "Intel Xe does not support
  specializing the tile scheduler.")`; `:185-193` + `static_tile_scheduler.hpp:260-268`
  - **grid = ⌈M/256⌉ × ⌈N/256⌉ × L, one work-group per output tile, the whole
  K loop inside it**. No split-K, no stream-K, no cross-work-group reduction,
  no atomic in `xe_gemm.hpp` / `xe_epilogue.hpp` / `xe_callbacks.hpp`. This is
  the §6.4 "no fp atomics" evidence, and it is why Task 1's determinism bar is
  bitwise rather than a tolerance.
- `include/cutlass/gemm/kernel/xe_gemm.hpp:174-176` - `get_workspace_size` returns
  **0** for this kernel.
- Alignment, from `xe_mma.hpp:142-169` + `include/cutlass/detail/layout.hpp:417-423`
  (128-bit copy granularity = 8 bf16 / 4 fp32 elements):
  **`K % 8 == 0`, `lda % 8 == 0`, `N % 8 == 0`, `ldb % 8 == 0`, `N % 4 == 0`,
  `ldc % 4 == 0`; `M` is unconstrained.** Every production shape and every test
  shape satisfies these (K ∈ {5120, 6144, 17408}, N ∈ {5120, 14336, 16384, 34816}).
  Base pointers want 64-byte alignment; `l0::Mem`'s default `align = 64`
  (`src/l0/memory.h:12`) supplies it.
- **`StrideB` is over `(N, K, L)`, not `(K, N, L)`** (`xe_gemm.hpp:216-219`,
  `cutlass/detail/layout.hpp:60,79`). Row-major B ⇒
  `make_stride(Int<1>{}, int64_t(ldb), int64_t(0))`. This is the single easiest
  bug in the whole task and Task 1 Step 3 asserts against it.
- Build requirements, for 6a's sub-project to satisfy:
  `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET` (both **mandatory** -
  `CUTLASS_ENABLE_SYCL` gates the `cudaStream_t = sycl::queue*` typedef,
  `SYCL_INTEL_TARGET` gates the Xe tile-scheduler specialisations and without it
  the build dies on "Could not select a tile scheduler"), C++17 minimum,
  `-fsycl -fsycl-targets=spir64_gen`, and at the device link
  `-Xsycl-target-backend=spir64_gen "-device bmg-g31" -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`
  (`cmake/FindDPCPP.cmake:41,71-74,111-122`; `CMakeLists.txt:187,214,233-237`).
  **The three SPIR-V extensions are not optional for AOT**: the mainloop uses
  split barriers (`xe_mma.hpp:262,277`), 2D block IO and the DPAS builtin. The
  full set needs IntelLLVM ≥ 2025.2 - the box has 2026.1.1. `tools/util/include`
  is **not** needed (this plan builds its own strides and owns its USM).
  `CUTLASS_SYCL_RUNNING_CTEST` does not exist in this revision; do not add it.
  `-fno-sycl-instrument-device-code` is a free win
  (`FindDPCPP.cmake:71-74`). CUTLASS/cute headers must come in with
  **`-isystem`**, never `-I`, so `-Wall -Wextra -Werror` keeps its meaning for
  our own code in these TUs.

## Interfaces: Produces

```cpp
// src/runtime/prefill/gemm.h  - plain C++. `context.h` is SYCL-free (6a T0), so
// engine.cc (g++) includes both; only gemm_bf16's DEFINITION is an icpx TU.
#pragma once
#include <cstdint>
#include "runtime/prefill/context.h"
namespace loader { struct DeviceWeight; }
namespace runtime::prefill {

struct GemmDims { uint32_t M, K, N; };

// C[M][N] fp32 (ldc = N) = A[M][K] bf16 (lda = K) · B[K][N] bf16 (ldb = N).
// Row-major throughout. Deterministic: cutlass::gemm::PersistentScheduler,
// one work-group per 256x256 output tile, whole-K loop inside it, no atomics.
// Throws std::runtime_error naming the failing condition on unsupported dims.
// M is a runtime value; there is no per-M binary.
// Asynchronous: submits on cx's SYCL queue and returns. The caller syncs
// (Context::wait(), which drains both queues).
void gemm_bf16(Context& cx, GemmDims d, const uint16_t* A, const uint16_t* B, float* C);

// Dequantise ONE linear's weights, in whatever tile layout the loader chose for
// that shape (layout 0: [K/8][N] u32 + separate f16 [K/64][N]; layout 1: 544-B
// tiles with inline scales), into the bf16 row-major [K][N] scratch, ldb = N.
// Bit-exact with common::f32_to_bf16(common::Int4Gptq::at(k, n)).
// Launched on cx's L0 list; `lin` must be model::WeightKind::Int4.
void dequant_to_bf16(Context& cx, const loader::DeviceWeight& lin, uint16_t* scratch);
}
```

```cpp
// src/kernels/kernels.h  - one new variant namer, beside gemv_variant's, for
// 6a P3's binaries (6a registers the variants; this is the host-side namer the
// production binding needs and the P3 probe hardcodes).
inline std::string pf_dequant_variant(unsigned K, unsigned N, unsigned L) {
  return "pf_dequant_tile_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_L" + std::to_string(L);
}
```

- `src/sycl/gemm.cc` - the CUTLASS instantiation, **added to 6a's `src/sycl/`
  sub-project** so it lands in `libb70_prefill.so`.
- `src/runtime/prefill/gemm.cc` - `dequant_to_bf16`'s host side, a **g++** TU in
  `b70_runtime` (`context.h` is SYCL-free).
- `tests/prefill/gemm_test.cc` (new), `tests/prefill/dequant_test.cc`
  (6a's, **extended** to the production shapes),
  `tests/prefill/prefill_chunking_test.cc` (new),
  the multi-chunk golden gate registered against `oracle-out-long`.
- `PrefillScratch::kC = 4096`, `partials` collapsed to `S = 1`, and the memory
  arithmetic that justifies both.
- docs/12 prefill mechanism section; docs/15 prefill anatomy rows.
- **No new CMake module.** Everything hangs off 6a's `cmake/prefill.cmake`,
  `src/sycl/CMakeLists.txt`, `src/kernels/prefill/CMakeLists.txt` and
  `b70_link_prefill`.

## Ruling requests to `interfaces.md`

None of these is applied silently; the controller rules, then the plan text and
`interfaces.md` move together.

- **R1 - `loader::Linear` does not exist.** The per-linear descriptor is
  `loader::DeviceWeight` (`src/loader/loader.h:19-26`). Request: change
  `dequant_to_bf16`'s parameter to `const loader::DeviceWeight& lin` and the
  parenthetical to name the real struct and its fields (`mem`, `scales`
  non-null iff layout 0, `shape{K,N,S,layout}`, `kind`).
- **R2 - `Context`'s shape: adopt plan 6a's deviations verbatim, no separate
  request from this plan.** 6a T0 already raised and justified the three that
  matter here: `Context(l0::Device&, l0::Ctx&)` → `Context(l0::Context&)` (the
  named wrapper types do not exist); `sycl::queue& sycl()` →
  `void* sycl_queue_raw()` plus the free function
  `runtime::prefill::sycl(Context&)` in `context_sycl.h`, which is what keeps
  `context.h` parseable by g++; and a `ze_kernel_handle_t` overload of `launch`.
  This plan uses 6a's spellings throughout and adds nothing to that request.
- **R2b - `Context::launch` needs an optional signal event.** 6a's signature has
  no place for one, and Task 7's per-kernel attribution needs
  `zeCommandListAppendLaunchKernel`'s `hSignalEvent` to read
  `kernelStart → kernelEnd` (`src/l0/event.h:53-56`). Request the additive
  trailing parameter `l0::Event* signal = nullptr` on both `launch` overloads -
  observe-only, exactly as `l0::CmdList::launch` already carries it
  (`src/l0/cmdlist.h:29-30`), so an unprofiled launch appends the identical
  command. Without it Task 7 must reach through `Context::ze_list()` and
  duplicate the launch, which is worse.
- **R5 - the `sycl-tla` revision, reconciled.** 6a's `cmake/prefill.cmake` pins
  `B70_SYCL_TLA_REVISION = 87f6850680a580654b9ea2c80dbc01aeb36ad231`. The
  checkout this plan read the GEMM API from is
  `2db1b7c94cadc52decf0013bd8b6e244fcb37dd4` (`v0.9.2-9-g2db1b7c9`,
  `~/PycharmProjects/sycl-tla`, 2026-09-04) - a **different** revision, and 6a's
  default `B70_SYCL_TLA_SRC_DIR` points at that same directory, so an
  unreconciled tree either fails 6a's revision check or silently builds against
  the wrong headers. Request: one revision, ruled by the controller, recorded in
  `cmake/prefill.cmake`. **Whichever wins, Task 1 Step 1 re-verifies the four
  load-bearing facts against it before any code is written** (row-major B,
  `cudaStream_t = sycl::queue*`, the `PersistentScheduler` static_assert,
  `get_workspace_size == 0`). This is a two-values-for-one-quantity item and it
  is not this plan's to decide.
- **R3 - `PrefillBuffers::kC` vs `PrefillScratch::kC`.** `interfaces.md:110`
  writes the first, `:121` the second. Request: `PrefillScratch::kC` everywhere.
- **R4 - withdrawn.** This plan was going to request a `--chunk` flag; plan 6b
  request #3 already asks for **`--pp-chunk C`** with the same semantics
  (default `PrefillScratch::kC`) and a two-row stdout shape. This plan uses 6b's
  spelling and adds nothing. Task 6's C = 1024 gate and Task 7's crossover sweep
  both depend on it being granted.

**Not requested:** the B-layout statement. `sycl-tla`'s Xe GEMM takes row-major
B `[K][N]`, `ldb = N`, natively (`00_bmg_gemm.cpp:351`); `interfaces.md`'s
layout convention is correct as written.

---

### Task 1: `gemm_bf16` - the `sycl-tla` instantiation behind a plain bf16 interface

**Files:**
- Create: `src/sycl/gemm.cc`, `src/runtime/prefill/gemm.h`
- Modify: `src/sycl/CMakeLists.txt` (6a's sub-project - add the source),
  `tests/CMakeLists.txt`
- Test: `tests/prefill/gemm_test.cc`

**Interfaces:**
- Consumes: 6a's `runtime::prefill::Context` + `context_sycl.h`'s
  `runtime::prefill::sycl(Context&)`; 6a's `cmake/prefill.cmake`,
  `src/sycl/CMakeLists.txt` and `b70_link_prefill`; the ruled `sycl-tla`
  revision (R5). Nothing from 6b.
- Produces: `void runtime::prefill::gemm_bf16(Context&, GemmDims, const uint16_t* A,
  const uint16_t* B, float* C)` exactly as declared under *Interfaces: Produces*.

- [ ] **Step 1: settle the revision and re-verify the four facts (R5).** Before
  writing a line: confirm with the controller which `sycl-tla` revision the build
  pins - 6a's `87f6850680a580654b9ea2c80dbc01aeb36ad231` or the checked-out
  `2db1b7c94cadc52decf0013bd8b6e244fcb37dd4` - and then, **against the ruled
  revision**, re-read and paste into the task report:
  1. `examples/00_bmg_gemm/00_bmg_gemm.cpp` - `LayoutB` is still
     `cutlass::layout::RowMajor` (it is `:351` at `2db1b7c9`);
  2. `include/cutlass/gpu_generics.h` - `using cudaStream_t = sycl::queue *;`
     (`:355` at `2db1b7c9`);
  3. `include/cutlass/gemm/kernel/xe_gemm.hpp` - the
     `is_void_v<TileScheduler_> or is_same_v<TileScheduler_, PersistentScheduler>`
     static_assert (`:82-87`) and the data-parallel `get_grid_shape` (`:185-193`);
  4. the same file's `get_workspace_size` returning 0 (`:174-176`).
  If any of the four has moved, **stop and report** - every one of them is
  load-bearing for a task step below, and three of them are §6.4 evidence.
  Also confirm `${B70_SYCL_TLA_SRC_DIR}/include/cute/tensor.hpp` resolves on the
  box under 6a's `B70_PREFILL=AUTO` configure (`tools/box.sh build`, look for
  `b70: prefill component ON`); the Mac's checkout is at
  `~/PycharmProjects/sycl-tla`, which is 6a's default path and **does not exist
  on the box today** (checked 2026-09-04), so either 6a's FetchContent fallback
  clones it or the path is overridden - record which.
- [ ] **Step 2: no CMake module of this plan's own.** `gemm_bf16` is a source
  file added to **6a's** `src/sycl/` sub-project, which is already an icpx
  `ExternalProject` producing `libb70_prefill.so`; consumers already call
  `b70_link_prefill(<target>)`. Do **not** build a parallel `-fsycl-link` +
  g++ route: 6a's `cmake/prefill.cmake` rejects it by name and states the reason
  (the device-code link step, and libsycl's static initialisers). This step is
  one line in `src/sycl/CMakeLists.txt`'s source list plus, if 6a's sub-project
  does not already carry them, the `-isystem <tla>/include`,
  `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET` and the three `-spirv-ext` entries
  listed under *Interfaces: Consumes*. Verify by `nm -DC` on the .so that
  `runtime::prefill::gemm_bf16` is exported.
- [ ] **Step 3: `src/sycl/gemm.cc`** - the one heavy TU. Copy the
  `00_bmg_gemm.cpp:344-433` config verbatim, force the scheduler, build the
  strides by hand (note the `(N,K,L)` mode order for B), pre-check the alignment
  rules so the failure names its cause instead of returning `kInvalid`, and
  launch on **our** queue.
  ```cpp
  // src/sycl/gemm.cc - the ONE sycl-tla instantiation, inside plan 6a's icpx
  // sub-project (cmake/prefill.cmake -> libb70_prefill.so). The CUTLASS headers
  // arrive through -isystem so this file's own code still answers to -Werror.
  //
  // The configuration is Intel's, copied from
  // examples/00_bmg_gemm/00_bmg_gemm.cpp:344-433 at the pinned
  // revision (Task 1 Step 1): work-group tile <256,256,32>, XE_DPAS_TT<8,float,bfloat16_t>
  // (8x16x16, fp32 accumulate), 8x4x1 = 32 subgroups, PipelineStages = 2 (a
  // K-block prefetch depth, not an SLM depth -- SharedStorageSize is 0), all
  // four operands row-major. Spec 2 §3.2: this is inherited, not written.
  #include <cstdint>
  #include <stdexcept>
  #include <string>

  #include "cutlass/epilogue/collective/xe_epilogue.hpp"
  #include "cutlass/epilogue/fusion/xe_callbacks.hpp"
  #include "cutlass/gemm/collective/collective_mma.hpp"
  #include "cutlass/gemm/device/gemm_universal_adapter.h"
  #include <cute/tensor.hpp>

  #include "runtime/prefill/context.h"
  #include "runtime/prefill/context_sycl.h"   // runtime::prefill::sycl(Context&)
  #include "runtime/prefill/gemm.h"

  namespace runtime::prefill {
  namespace {
  using namespace cute;

  using ElementAccumulator     = float;
  using ElementComputeEpilogue = float;
  using ElementInputA          = bfloat16_t;
  using ElementInputB          = bfloat16_t;
  using ElementOutput          = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using TileShape = Shape<_256, _256, _32>;
  using TiledMma  = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy     = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;

  // **`ElementC = void`, deliberately, where Intel's example passes
  // ElementAccumulator.** `is_source_supported = !is_void_v<ElementC>`
  // (xe_epilogue.hpp:125-126), so `void` deletes the epilogue's C load
  // entirely. This GEMM never accumulates into C -- alpha = 1, beta = 0, D is
  // written outright -- and at gate|up the source read would be a wasted
  // 570,425,344 B per launch: 0.97 ms at 590 GB/s against the GEMM's derived
  // 16.2 ms, a 6% tax on the largest shape. Task 1 Step 3a keeps Intel's
  // ElementAccumulator as the recorded fallback if this does not instantiate.
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy, TileShape, void, void,
      cutlass::gemm::TagToStrideC_t<LayoutC>, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallbacks, void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      void, void, void, cute::identity,   // A: auto copy atom -> XE_LOAD_2D
      void, void, void, cute::identity>;  // B: auto copy atom -> XE_LOAD_2D_VNNI

  // **The determinism decision, spelled out rather than defaulted** (spec §6.4).
  // PersistentScheduler is the ONLY scheduler xe_gemm.hpp accepts
  // (xe_gemm.hpp:82-87 static_asserts it), and it is pure data-parallel: the
  // grid is ceil(M/256) x ceil(N/256) x L, one work-group per output tile, the
  // whole K loop inside that work-group (xe_gemm.hpp:185-193 ->
  // static_tile_scheduler.hpp:260-268). There is no cross-work-group reduction
  // and no atomic anywhere in xe_gemm.hpp / xe_epilogue.hpp / xe_callbacks.hpp.
  // Naming it here rather than passing `void` is what stops a future sycl-tla
  // bump from changing the default under us. Stream-K (03_bmg_gemm_streamk,
  // KernelXeCooperative + StreamKScheduler) is the configuration this spec
  // forbids; it is not reachable from these types.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;  // (M, K, L)
  using StrideB = typename GemmKernel::StrideB;  // (N, K, L)  <-- N first
  using StrideC = typename GemmKernel::StrideC;  // (M, N, L)
  using StrideD = typename GemmKernel::StrideD;  // (M, N, L)

  void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("runtime::prefill::gemm_bf16: " + what);
  }
  }  // namespace

  void gemm_bf16(Context& cx, GemmDims d, const uint16_t* A, const uint16_t* B, float* C) {
    // The 128-bit copy granularity of the Xe 2D block loads: 8 bf16 elements,
    // 4 fp32 (xe_mma.hpp:142-169, cutlass/detail/layout.hpp:417-423). Checking
    // it here turns can_implement's opaque kErrorInvalidProblem into a sentence.
    require(d.M != 0 && d.K != 0 && d.N != 0, "M, K and N must all be non-zero");
    require(d.K % 8 == 0, "K (= lda) must be a multiple of 8 bf16 elements, got " +
                              std::to_string(d.K));
    require(d.N % 8 == 0, "N (= ldb = ldc) must be a multiple of 8 bf16 elements, got " +
                              std::to_string(d.N));

    const int M = static_cast<int>(d.M), K = static_cast<int>(d.K), N = static_cast<int>(d.N);
    const StrideA sA = cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0));
    const StrideB sB = cute::make_stride(cute::Int<1>{}, int64_t(N), int64_t(0));
    const StrideC sC = cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(0));
    const StrideD sD = sC;

    // sm_count is [[maybe_unused]] for the static persistent scheduler on Xe
    // (static_tile_scheduler.hpp:103), so a default-constructed hw_info avoids
    // compat::get_device() -- which would consult the compat device manager
    // rather than the L0 context this engine actually owns.
    cutlass::KernelHardwareInfo hw_info{};

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const cute::bfloat16_t*>(A), sA,
         reinterpret_cast<const cute::bfloat16_t*>(B), sB},
        // alpha = 1, beta = 0, ptr_C = nullptr: with ElementC = void the
        // epilogue never reads a source, so D = A*B outright.
        {{ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, nullptr, sC, C, sD},
        hw_info};

    require(Gemm::can_implement(args) == cutlass::Status::kSuccess,
            "sycl-tla cannot implement M=" + std::to_string(M) + " K=" + std::to_string(K) +
                " N=" + std::to_string(N) + " (rebuild with -DCUTLASS_DEBUG_TRACE_LEVEL=1"
                " for the CAN IMPLEMENT line)");
    // xe_gemm.hpp:174-176 returns 0 for this kernel; if a bump ever changes that
    // the wrapper must grow an allocation rather than pass nullptr silently.
    require(Gemm::get_workspace_size(args) == 0,
            "this sycl-tla configuration now wants a workspace; the wrapper passes none");

    // cudaStream_t IS sycl::queue* under CUTLASS_ENABLE_SYCL
    // (cutlass/gpu_generics.h:355), and gemm_universal_adapter.h:551 uses it
    // instead of compat::get_default_queue() whenever it is non-null. That one
    // line is the whole L0<->SYCL interop story for the GEMM: our queue, built
    // from the engine's own ze_context/ze_device, IS the launch queue. The typed
    // accessor is a free function (context_sycl.h) because Context itself is
    // SYCL-free so that g++ TUs can hold one (plan 6a T0).
    sycl::queue& q = runtime::prefill::sycl(cx);
    Gemm op;
    require(op.initialize(args, /*workspace=*/nullptr, &q) == cutlass::Status::kSuccess,
            "GemmUniversalAdapter::initialize failed");
    require(op.run(&q) == cutlass::Status::kSuccess, "GemmUniversalAdapter::run failed");
    // Asynchronous by contract: the caller syncs (Context::wait()).
  }
  }  // namespace runtime::prefill
  ```
- [ ] **Step 3a: settle `ElementC`.** Build the TU as written (`ElementC = void`,
  `ptr_C = nullptr`). **If it does not instantiate**, fall back to Intel's
  `ElementAccumulator` with `ptr_C = C` and record, in the commit and in
  docs/12, the wasted source read it costs (570,425,344 B per gate‖up launch =
  0.97 ms at 590 GB/s, derived, 6% of that shape's derived GEMM time) as an
  open item for plan 6e's composed ceiling. Do not leave both variants in the
  tree.
- [ ] **Step 4: wire the consumers.** `b70_link_prefill(b70-decode)` and
  `b70_link_prefill(gemm_test)`; `b70_runtime` links the `b70_prefill` INTERFACE
  target 6a defines. Every prefill target and test is registered **inside**
  `if(B70_PREFILL_ENABLED)`, the pattern 6a's `context_test` established, so the
  `-DB70_PREFILL=OFF` tree still configures, builds and passes - 6a asserts that
  and this plan must not break it. `tools/box.sh build`, then
  `tools/box.sh run "ldd build/src/cli/b70-decode | grep -iE 'sycl|b70_prefill'"`
  to confirm the .so and its rpath resolve.
- [ ] **Step 5: `tests/prefill/gemm_test.cc`.** Three shapes,
  `(M,K,N) ∈ {(64,5120,5120), (1024,5120,17408), (4096,17408,5120)}`, inputs
  `random_bf16(n, seed)` from `tests/kernels/gemv_ref.h` in `[-1, 1]`.
  - **The reference** accumulates in `double` over the exactly-widened bf16
    inputs (`common::bf16_to_f32` is exact), so the bar covers only the device's
    fp32 accumulation. Full reference at `(64,5120,5120)` - 1,677,721,600 double
    MACs, **estimated 3-6 s single-threaded**. Sampled reference at the other
    two - 4096 `(m,n)` pairs from `std::mt19937(0x6e6d31u)`, 4096 × K ≤
    71,303,168 MACs, well under a second. State the sampling in the test's header
    comment; a full reference at `(4096,17408,5120)` would be 3.65e11 double
    MACs and is not a test.
  - **The bar, and where it comes from.** For each checked element compute
    `S = Σ_k |A[m][k]·B[k][n]|` in double alongside the reference and require
    ```
    |C_dev − C_ref| ≤ 64 · 2^-24 · S + 2^-24
    ```
    `2^-24 = 5.9604644775390625e-08` is fp32's unit roundoff; the `+2^-24` term
    only covers an all-zero row. **Derivation:** a K-term fp32 dot product
    accumulated with depth `d` carries a deterministic bound of `γ_d·S` with
    `γ_d ≈ d·2^-24`, i.e. up to `K` ulp-equivalents (17408 at the largest shape) -
    a bound so loose it tests nothing. The realistic model is a random walk:
    `|err| ≈ √d · 2^-24 · RMS(partial sums)`, and with uniform `[-1,1]` inputs
    `RMS(partials) ≈ 0.7·√K/3` while `S ≈ K/4`, so the ratio
    `|err| / (2^-24·S) ≈ 0.7/0.75 = 0.93` **independently of K**. The bar of 64
    is a **69× margin over that model and 272× under the deterministic worst
    case at K = 17408** - it is a probabilistic bar, not a proof, which is why
    the test **prints the measured max ratio at every shape** so the margin is
    visible and any drift is a finding rather than a silent pass.
  - **Determinism (spec §6.4):** run each shape twice into two distinct output
    allocations and `memcmp` the full fp32 result. Bitwise, not a tolerance -
    justified by the scheduler evidence quoted in Step 3's comment.
  - **M-invariance:** run `(4096,17408,5120)` and `(64,17408,5120)` over the same
    B and the first 64 rows of A; the first 64 output rows must be **bitwise
    identical**. This follows from the grid being `⌈M/256⌉ × ⌈N/256⌉` with the
    whole K loop inside one work-group: rows 0-63 sit in M-tile 0 either way and
    their reduction chain does not depend on `M`. A failure means the tile
    scheduler is M-dependent and the chunking argument in Task 5 must be
    re-examined; stop and report rather than relaxing it.
  - Wire it up in `tests/CMakeLists.txt` beside 6a's `context_test`, inside
    `if(B70_PREFILL_ENABLED)`: `target_link_libraries(gemm_test PRIVATE b70_l0)`,
    `b70_link_prefill(gemm_test)`, `add_test(NAME gemm_test COMMAND gemm_test)`.
- [ ] **Step 6: run it.** `ZE_AFFINITY_MASK=1 tools/box.sh test gemm_test`.
  Record, in the task report: the max ratio per shape, the per-shape TFLOP/s the
  test happens to observe (labelled **iterate-grade, not a bench row** - the test
  is not the P2 harness; if P2 has landed, quote its figure beside it and
  reconcile), and the `ElementC` decision from Step 3a.
- [ ] **Step 7: Commit** -
  `feat(prefill): gemm_bf16 - sycl-tla Xe bf16 GEMM on our L0-derived queue`

### Task 2: `dequant_to_bf16` - promote 6a's P3 kernel to production

*6a P3 writes `src/kernels/prefill/dequant.cl` (`pf_dequant_tile`), registers its
variants at both layouts for every production shape, and holds it bit-exact at
256×64. **This task writes no kernel.** It adds the production binding, raises
the bit-exactness bar to the production shapes at full size, bounds the writes
inside the shared scratch, and carries one conditional geometry lever.*

**Files:**
- Create: `src/runtime/prefill/gemm.cc` (the `dequant_to_bf16` host side - a
  **g++** TU in `b70_runtime`, because 6a's `context.h` is SYCL-free)
- Modify: `src/kernels/kernels.h` (`pf_dequant_variant`),
  `tests/prefill/dequant_test.cc` (6a's - extended),
  `tests/CMakeLists.txt` (the production-shape kernel dependencies)
- Conditional (Step 5 only): `src/kernels/prefill/dequant.cl`,
  `src/kernels/prefill/CMakeLists.txt`

**Interfaces:**
- Consumes: `loader::DeviceWeight` (R1); 6a's `Context::launch(l0::Kernel&, gx,
  gy, gz, std::initializer_list<KernelArg>)`; 6a P3's `pf_dequant_tile` and its
  `pf_dequant_tile_K{K}_N{N}_L{L}` variants, grid `(N/16, K/64)` × 16 lanes;
  6a P3's measured GB/s; `PrefillScratch::dequant` (6b), 356,515,840 B.
- Produces: `void runtime::prefill::dequant_to_bf16(Context&, const
  loader::DeviceWeight&, uint16_t* scratch)`; `kernels::pf_dequant_variant`.

- [ ] **Step 1: the traffic table and the rounding cost, before any code.**
  Derived; 6a P3 measures the GB/s. Read bytes are `K·N·17/32` (nibbles
  `K·N/2` + one f16 scale per 64 weights, `K·N/32`); write bytes are `K·N·2`.

  | linear | K×N | production layout | read B | write B | total B | ms @590 GB/s |
  |---|---|---|---|---|---|---|
  | QkvZ | 5120×16384 | 1 | 44,564,480 | 167,772,160 | 212,336,640 | 0.360 |
  | OutProj / OProj | 6144×5120 | 0 | 16,711,680 | 62,914,560 | 79,626,240 | 0.135 |
  | GateUp | 5120×34816 | 0 | 94,668,800 | 356,515,840 | 451,184,640 | 0.765 |
  | Down | 17408×5120 | 0 | 47,349,760 | 178,257,920 | 225,607,680 | 0.382 |
  | Qkv | 5120×14336 | 0 | 39,004,160 | 146,800,640 | 185,804,800 | 0.315 |

  Per chunk: 48 GDN layers × 968,755,200 B + 16 FA layers × 942,223,360 B =
  **61,575,823,360 B = 61.576 GB → 104.4 ms at 590 GB/s (derived)**, over
  **256 matrices (48×4 + 16×4)** - the same count 6a's composed-ceiling table
  uses. The figure is **C-independent**, which is the entire argument for a wide
  chunk.

  **And the numerics cost, stated once here because no kernel comment of this
  plan's will carry it.** Decode multiplies fp32 `(q−8)·scale` by the
  activation; prefill multiplies `bf16(rne((q−8)·scale))`. `(q−8)` carries ≤ 4
  significant bits and an f16 scale 11, so the product can need 15 and bf16
  keeps 8: the extra rounding is real. Its RMS relative size is
  `2⁻⁸/√12 = 0.113%`, and because the per-`k` errors are independent the dot
  product inherits the same **0.113% relative RMS - below one bf16 ulp of the
  output (2⁻⁸ = 0.39%)**, i.e. inside the per-op bf16 discipline docs/14 already
  governs. It is **not** bit-identical with decode and cannot be; the golden gate
  is the arbiter (spec §6.1, §6.3), and Task 3 Step 6 pre-registers the risk.
- [ ] **Step 2: `kernels::pf_dequant_variant`.** Add it to
  `src/kernels/kernels.h` exactly as given under *Interfaces: Produces*, beside
  `gemv_variant`. It names 6a's binaries; if 6a already added an equivalent
  namer, use 6a's and skip this step - two namers for one binary set is the
  drift `kernels.h`'s own comments warn about.
- [ ] **Step 3: `src/runtime/prefill/gemm.cc` - the host binding.** Reads the
  shape and the kind **off the loaded weight**, never off the model table (the
  discipline `capture.cc:390-401` established for `lm_head`); caches
  `l0::Module`/`l0::Kernel` per variant name; binds the scales argument the way
  `capture.cc:404-405` does, so layout 1's unused parameter is still a valid
  pointer; passes the three pointers as `KernelArg`s because 6a's
  `Context::launch` sets the arguments itself.
  ```cpp
  void dequant_to_bf16(Context& cx, const loader::DeviceWeight& lin, uint16_t* scratch) {
    const model::GemvShape& s = lin.shape;
    require(lin.kind == model::WeightKind::Int4, "dequant_to_bf16 bound to a bf16 weight");
    require((s.layout == 0) == bool(lin.scales),
            s.layout == 0 ? "layout-0 weight has no independent scales allocation"
                          : "layout-1 weight unexpectedly owns a separate scales allocation");
    // 6a P3's kernel writes out[k*N + n] for every k in [0,K), n in [0,N), so the
    // scratch must hold K*N bf16. PrefillScratch::dequant is sized for the
    // largest production matrix (gate|up, 5120x34816 -> 356,515,840 B); every
    // other production shape is smaller and uses a prefix of it.
    require(size_t(s.K) * s.N * 2 <= kDequantScratchBytes,
            "the dequant scratch is smaller than this linear's [K][N] bf16 result");
    const void* w = lin.mem.ptr();
    const void* sc = s.layout == 0 ? lin.scales->ptr() : lin.mem.ptr();
    l0::Kernel& k = kernel(kernels::pf_dequant_variant(s.K, s.N, s.layout), "pf_dequant_tile");
    cx.launch(k, s.N / 16, s.K / 64, 1,
              {{&w, sizeof w}, {&sc, sizeof sc}, {&scratch, sizeof scratch}});
  }
  ```
  The grid `(N/16, K/64)` is 6a P3's, not this plan's - read `dequant.cl`'s
  header as 6a left it and take the grid from there rather than from this
  snippet if P3 retuned it; the guard is that a grid the binary was not built for
  throws by name at `l0::Module`, the same guard `ATTN_BLOCK` uses.
- [ ] **Step 4: extend `tests/prefill/dequant_test.cc` to the production
  shapes.** 6a's test proves bit-exactness at 256×64 in both layouts, which is
  the *contract*; production needs it at the *shapes*, because the layout-1 tile
  index arithmetic scales with `G = K/64` and `N/16` and a 256×64 case exercises
  4 k-groups and 4 n-tiles. Add a loop over the **five production
  `(K, N, layout)` pairs** - `(5120,16384,1)`, `(6144,5120,0)`,
  `(5120,34816,0)`, `(17408,5120,0)`, `(5120,14336,0)` - each built from
  `common::Int4Gptq::random(K, N, seed)`, repacked with
  `common::repack_int4_layout1` (layout 1) or uploaded verbatim (layout 0), run
  through `dequant_to_bf16`, read back whole, and required
  **`out[k*N + n] == common::f32_to_bf16(w.at(k, n))` for every element,
  bitwise**. No tolerance: both sides are an exact int→float, an exact f16→float,
  one correctly-rounded fp32 multiply and one RNE, so a differing bit is a bug.
  Largest readback 356,515,840 B (the box has 121 GB); total across the five
  ≈ 1.05 GB. **Also bound the writes:** fill the whole 356,515,840-B scratch
  with `0xCD` before each of the four smaller shapes and assert every byte past
  `K·N·2` is still `0xCD`. Add the five `kernel_pf_dequant_tile_K…_L…` targets to
  the test's `add_dependencies`.
- [ ] **Step 5 (conditional lever): the layout-0 store geometry.** 6a P3's grid
  gives one 16-lane subgroup a `(n_tile, k_group)`, so at **layout 0** a lane
  reads 8 u32 at stride `N·4` and each of its stores is 16 contiguous bf16 =
  **32 B**. Writes are 4× the reads on this kernel, so the store is the side that
  matters. **If P3's measured GB/s misses its own ≥ 500 GB/s prediction on the
  layout-0 shapes**, add a layout-0-only geometry to `dequant.cl` behind a `-D`:
  grid `(N/256, K/8)`, `reqd_work_group_size(256,1,1)`, lane `l` owning column
  `wg·256 + l` and one u32 row, so the reads are one 1 KB coalesced line per
  work-group per row and each of the eight stores is **512 B**. Layout 1 keeps
  P3's geometry (its 544-byte tile read is byte-identical to `gemv.cl`'s and
  QkvZ is the only production layout-1 shape). Acceptance: bit-exactness
  unchanged (Step 4 re-run) **and** ≥ 15% GB/s on the four layout-0 shapes; below
  that, revert and record. **If P3 met its prediction, skip this step entirely
  and say so** - it is a lever, not a refactor.
- [ ] **Step 6: run and reconcile.** `ZE_AFFINITY_MASK=1 tools/box.sh test dequant_test`.
  Time each production shape inside the test with the 8-replay/drop-3 convention
  of `tests/kernels/gemv_harness.h::time_list` and print measured GB/s beside
  Step 1's derived ms. Write the reconciliation sentence for spec §3.1's
  "~1.2 ms" and "8-16%": the spec counts 356 MB both ways; the read side is int4,
  so the derived figures are 0.765 ms and 4.5% (C = 4096) / 8.6% (C = 2048).
  Quote 6a P3's number as the measured authority and this plan's derivation as
  the model it is checked against.
- [ ] **Step 7: Commit** -
  `feat(prefill): dequant_to_bf16 - bind pf_dequant_tile at the production shapes`

### Task 3: the swap in `Engine::prefill()`, and the retirement of `pf_gemv_int4_M`

*6b ruling R1 already runs the prefill path at `S = 1` and already sizes
`PrefillScratch::partials` as one `fp32 [kC][34816]` rectangle, so there is no
split-K to collapse and no fold-count variant to add here: the GEMM's fp32
`[M][N]` output **is** the buffer every prefill consumer already reads. What is
left is the substitution, the retirement of the temporary kernel, and the
statement of what the substitution does and does not change about rounding.*

**Files:**
- Modify: `src/runtime/engine.{h,cc}` (the six call sites + `pf_linear`)
- Delete: `src/kernels/prefill/pf_gemv.cl`, `tests/prefill/pf_gemv_test.cc`
- Modify: `src/kernels/prefill/CMakeLists.txt` (the five `add_pf_gemv` lines),
  `src/kernels/kernels.h` (`pf_gemv_variant`), `tests/CMakeLists.txt`
- Test: existing `tests/prefill/prefill_gate_test.cc`,
  `prefill_consistency_test.cc`, `prefill_determinism_test.cc` (6b's, unchanged)

**Interfaces:**
- Consumes: 6b's `Engine::prefill()` walk and its six `pf_gemv_int4_M` call
  sites; 6b's `PrefillScratch::{partials, x, mixer_out, dequant}`; Task 1's
  `gemm_bf16`; Task 2's `dequant_to_bf16`.
- Produces: an `Engine::prefill()` whose six int4 linears run
  `dequant_to_bf16` → `gemm_bf16`, and a tree with no `pf_gemv_int4_M` in it.

- [ ] **Step 1: state the rounding points, then verify they did not move.**
  Decode rounds a linear's output to bf16 **exactly once**, in its consumer, as
  `rne_bf16(Σ_{s<S} partials[s][m][n])`. The consumers and their spellings, read
  from source, with 6b's prefill counterpart beside each:
  - `prep_res_fold` / `prep_res_norm` → `pf_res_fold` / `pf_norm_finish`:
    `mixer_b = rne_bf16(Σ_s partials[...])`, then
    `r_b = rne_bf16(bf16f(resid[k]) + bf16f(mixer_b))` (`prep.cl:145`, `:243`) -
    consumes OutProj / OProj / Down.
  - `prep_silu_mul` → `pf_silu_mul`: `g_b = rne_bf16(ga)`, `u_b = rne_bf16(ua)`
    (`prep.cl:326-331`) - consumes GateUp.
  - `prep_gated_head` → `pf_gated_head`, called by `gdn_chunk` (6b R3):
    `z_b = rne_bf16(za)` (`prep.cl:367-369`) - consumes QkvZ's z half.
  - `attn_prep` → `pf_attn_prep`: `rne_bf16(qkv_sum(...))`
    (`attn.cl:307-311`, `:352`) - consumes Qkv.
  - `gdn_step` → `gdn_chunk`, which by 6b R4 takes the **fp32** `[C][16384]`
    partials and rounds inside, at the point `gdn_step.cl:253` rounds. There is
    no intermediate bf16 copy of the `qkv‖z` linear and this task must not add
    one.

  All of them already fold **one** slice (6b R1), so replacing the producer
  changes **nothing** about where the rounding happens: still one RNE at the
  linear's output, in the same consumer, at the same place in the chain. Verify
  it rather than assume it - grep the five prefill kernels for their fold loops
  and paste the bounds into the task report.

  **What does change is the accumulation order inside the linear.**
  `pf_gemv_int4_M` sums a register-accumulated fp32 dot over `K` in `gemv.cl`'s
  order; the GEMM sums the whole `K` in one work-group through DPAS with fp32
  accumulate. **And the weights themselves are now bf16** - Task 2 Step 1's
  0.113% relative-RMS rounding, which the GEMV path did not have. Both are
  reassociations/roundings, so by the standing rule (`docs/14`, "The diagnostics
  move with every lever") the golden gate is the arbiter and must be run
  **before as well as after**. Write this paragraph into the commit and into
  docs/12 (Task 7).
- [ ] **Step 2: the swap itself.** Add one private helper to `Engine` and
  replace all six `pf_gemv_int4_M` call sites with it:
  ```cpp
  // One int4 linear of a prefill chunk: dequantise this layer's weights into the
  // shared bf16 scratch, then one bf16 GEMM into `partials` -- the fp32 [C][N]
  // rectangle 6b's ruling R1 already made an S = 1 buffer, which is exactly the
  // GEMM's C matrix at ldc = N.
  //
  // **Two waits, and both are load-bearing.** plan 6a's Context creates its L0
  // immediate list ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, so launch() returns before
  // the dequant has run, and the GEMM reads the scratch from a DIFFERENT queue:
  // wait #1 is that cross-queue dependency. Wait #2 is the mirror -- the next L0
  // kernel (a prefill prep) reads the fp32 output, and single-buffered, the next
  // dequant overwrites the scratch this GEMM is reading. That is exactly spec
  // §3.6's "or, first cut, a queue wait between the two"; Task 4 double-buffers
  // the scratch and removes wait #2's scratch half.
  void Engine::pf_linear(uint32_t layer, model::LinearId id, const void* a, uint32_t C) {
    const loader::DeviceWeight& w = model_.linears.at({layer, id});
    uint16_t* scratch = pf_->dequant.as<uint16_t>();
    prefill::dequant_to_bf16(*pf_cx_, w, scratch);
    pf_cx_->wait();                                        // #1: L0 dequant -> SYCL GEMM
    prefill::gemm_bf16(*pf_cx_, {C, w.shape.K, w.shape.N},
                       static_cast<const uint16_t*>(a), scratch,
                       pf_->partials.as<float>());
    pf_cx_->wait();                                        // #2: SYCL GEMM -> L0 consumer
  }
  ```
  Call sites, in 6b's walk order. **GDN layer:** `pf_linear(l, QkvZ, pf_->x, C)`,
  then `gdn_chunk` (which consumes `partials` as fp32 and ends at
  `pf_gated_head` writing `mixer_out`), then
  `pf_linear(l, OutProj, pf_->mixer_out, C)`, then the MLP -
  `pf_linear(l, GateUp, pf_->x, C)`, `pf_silu_mul`,
  `pf_linear(l, Down, pf_->x, C)`. **FA layer:** `pf_linear(l, Qkv, pf_->x, C)`,
  the L1 attention route (`attn_prep_l1` / `attn_l1`, which ends at
  `attn_reduce` writing `mixer_out`), then
  `pf_linear(l, OProj, pf_->mixer_out, C)`, then the same MLP.
  **`a` is `mixer_out` for OutProj and OProj and `x` for the other four** -
  6b's ruling R2 is the invariant that makes each producer's row stride equal
  the consuming linear's `K` (5120, 17408, 6144), which is `gemm_bf16`'s `lda`.
  **`a‖b` (`pf_ab_proj`, bf16) and `lm_head` (last row only, its existing S = 1
  route on either checkpoint) are untouched** - spec §3.5.
- [ ] **Step 3: retire `pf_gemv_int4_M`.** 6b's Task 4 names it "L1 temporary,
  retired by plan 6c" in its own commit message; this is that retirement. Delete
  `src/kernels/prefill/pf_gemv.cl`, `tests/prefill/pf_gemv_test.cc`, the five
  `add_pf_gemv(...)` lines and the `pf_gemv_variant` namer, and the test's
  registration in `tests/CMakeLists.txt`. **Keep** `pf_gemv_bf16.cl` and the
  `pf_ab_proj` binary - the `a‖b` projection stays on that route by spec §3.5.
  Deleting rather than leaving it compiled is deliberate: a bindable kernel that
  nothing binds is an invitation to bind it, and the plan-6b brief already
  scheduled its death.
- [ ] **Step 4: count the syncs.** 64 layers × 4 int4 linears × 2 waits =
  **512 `Context::wait()` per chunk**. At 6a P1's predicted ≤ 10 µs per
  immediate-list append-and-complete that is **≤ 5.12 ms, 0.22% of the derived
  2318 ms chunk** - inside spec §3.6's 1% interop bar, which at this chunk is
  23 ms, i.e. 45 µs per wait. Substitute P1's **measured** figure and print the
  measured total in Task 7's attribution against the bar. If the measured total
  exceeds 1%, Task 4 stops being a lever and becomes required.
- [ ] **Step 5: acceptance.** Full suite green (`tools/box.sh test`), then 6b's
  three prefill tests unchanged
  (`ZE_AFFINITY_MASK=1 tools/box.sh test "prefill_(gate|consistency|determinism)_test"`),
  then the decode invariants: `replay_determinism_test` and
  `profile_capture_test` still assert 774 / 19, and `buffers_test`'s three
  totals are unmoved (this task changes no buffer).
  **Registered risk:** `prefill_consistency_test` requires prefill and
  ingest-by-decode to produce **64 identical tokens** (spec §6.3), and this task
  adds a per-weight bf16 rounding (0.113% relative RMS on the output, Task 2
  Step 1) that the widened-GEMV path did not have. If a token flips, **stop**:
  record the flip position, the prompt, and the state diagnostics band, and take
  it to the operator as a ruling request - do not widen the bar and do not
  revert silently. The design has no fp32-scratch fallback (spec §3.1 fixes
  bf16); the priced alternative is spec §11's "a second weight layout for
  prefill".
- [ ] **Step 6: Commit** -
  `feat(prefill): swap the six int4 linears onto dequant + sycl-tla GEMM; retire pf_gemv_int4_M`

### Task 4 (lever): double-buffer the dequant scratch

*A lever, with a pre-registered prediction and a revert path. It costs
356,515,840 B of device memory and buys only what it measures.*

**Files:**
- Modify: `src/runtime/buffers.{h,cc}` (`dequant` → 2 slots),
  `src/runtime/engine.cc` (`pf_linear`'s ordering)
- Test: `tests/prefill/prefill_determinism_test.cc` (6b's, re-run),
  `tests/runtime/buffers_test.cc` (the new total)

**Interfaces:**
- Consumes: Task 3's `pf_linear`; 6a P3's measured per-matrix dequant ms **and
  P3 Step 9's own double-buffering answer** - read it first and quote it; if P3
  already ruled the question negative on measurement, this task is skipped by
  ruling and the skip is recorded, not re-litigated.
- Produces: `PrefillScratch::dequant` at 713,031,680 B with
  `uint16_t* dequant_slot(uint32_t i)`.

- [ ] **Step 1: pre-register the prediction, in the report, before any run.**
  Let `D` = the measured (P3 / Task 2 Step 6) total dequant time per chunk and
  `T` = the measured GEMM time per chunk. Every per-matrix dequant is far
  shorter than the GEMM that follows it at C = 4096 (derived: gate‖up 0.765 ms
  of dequant against 1460.3 GFLOP ≈ 16.2 ms of GEMM at 90 TFLOP/s), so a perfect
  overlap hides **all** of `D`. **Prediction: the lever recovers 0.6-0.9·D**,
  i.e. at the derived `D = 104.4 ms` a saving of **63-94 ms per 4096-chunk =
  2.7-4.1% of the derived 2318 ms chunk**. Substitute the measured `D` before
  running. **Acceptance: ≥ 0.3·D recovered; below that the lever is reverted and
  the 356 MB given back**, with the negative recorded.
- [ ] **Step 2: the honest caveat, written down first.** Overlap requires the
  driver to schedule the L0 immediate list and the SYCL queue concurrently on
  the compute engine. Two queues on one engine may serialise. **This lever
  measures whether they do**; it is not an assumption. P1's interop probe (6a)
  is the prior evidence - quote its result here.
- [ ] **Step 3: no new L0 plumbing is needed.** 6a's `Context` already creates
  its immediate list `ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS` and drains it with
  `zeCommandListHostSynchronize`, so the dequant can already run behind the
  host. `l0::CmdList::immediate` stays exactly as it is -
  `ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS` (`src/l0/cmdlist.cc:10`), which uploads,
  `DecodeBuffers`' fills and every existing test depend on. **Do not touch
  `src/l0/`.**
- [ ] **Step 4: two slots and the shifted wait.** `PrefillScratch::dequant`
  becomes `2 × 356,515,840 = 713,031,680 B`; `pf_linear` takes a monotonically
  increasing linear index `i` and uses `dequant_slot(i & 1)`. The sequence
  becomes: dequant `i` into slot `i&1` on the L0 list, then a wait that covers
  **the GEMM of linear `i−1`** (whose fp32 output the next L0 consumer reads)
  rather than the dequant of linear `i`, then submit GEMM `i`. The slot that
  GEMM `i−1` reads is `(i−1)&1`, which is not the one being written, so the
  dequant and the GEMM are free to overlap and **one of Task 3's two waits per
  linear disappears** - 512 waits per chunk become 256. The final linear of a
  chunk still ends with a full `Context::wait()`, and every consumer of a GEMM's
  fp32 output still runs after a wait; only the *scratch* dependency is relaxed.
- [ ] **Step 5: measure and rule.** `--profile`-style attribution (Task 7's
  instrument) before and after, on the same binary, idle box, medians of 3.
  Accept per Step 1's bar; on a miss revert the buffer and the ordering in one
  commit and record the priced negative.
- [ ] **Step 6: determinism re-run.** `prefill_determinism_test` must still be
  bitwise. Double-buffering changes *when* kernels run, never what any of them
  reads: the in-order L0 list and the in-order SYCL queue both preserve
  per-stream order, and the scratch dependency is explicitly drained. A failure
  here is a real race, not a tolerance question.
- [ ] **Step 7: Commit** -
  `perf(prefill): double-buffer the dequant scratch - <X> ms/chunk measured`
  (or `revert(prefill): double-buffered dequant recovered <X> of <D> - reverted`)

### Task 5: chunk width - prove `kC = 4096` fits, and that a chunk boundary works

*6b's Task 1 already sets `PrefillScratch::kC = 4096`, `kAttnC = 64` and
`kGdnChunk = 64`, already carries the `dequant` field at 356,515,840 B labelled
"plan 6c's; unused in L1", already pins the scratch total at **1,961,713,560 B**
in `buffers_test`, already caps the L1 attention at a 64-query sub-chunk (R5),
and already asks for `--pp-chunk C` (its request #3). **So there is no constant
to move and no bridge to build.** What L2 owns is the part 6b's byte table does
not cover: the **whole-device** residency that decides whether 4096 is the
chosen width or 2048 is, the confirmation of it on the box with real weights,
and a test that the chunk decomposition is what it claims.*

**Files:**
- Modify: `tests/runtime/buffers_test.cc` (the whole-device total beside 6b's
  three), `src/runtime/buffers.h` (the derivation comment beside `kC`)
- Test: `tests/prefill/prefill_chunking_test.cc`

**Interfaces:**
- Consumes: 6b's `PrefillScratch` (`kC`, `kAttnC`, `bytes() == 1961713560`),
  `PersistentBuffers::bytes() == 1240465536`,
  `DecodeScratch::bytes() == 68652864`, ruling R7 (lazy prefill allocation) and
  6b's `--pp-chunk`; `loader::LoadReport` for the measured weight bytes.
- Produces: the whole-device arithmetic recorded beside `kC`; a chunk-decomposition
  test; the priced C = 2048 fallback.

- [ ] **Step 1: the whole-device arithmetic, at max_len 16384, kC = 4096.**
  Every figure is either measured (the load report) or pinned by 6b's own
  `buffers_test`.

  | group | bytes | source |
  |---|---|---|
  | weights resident, RTN | 16,219,604,992 | measured: 13,672,613,888 read/token + 2,542,796,800 embed + 4,194,304 rope (`docs/13-loader.md:654,377,378`) |
  | `PersistentBuffers` | 1,240,465,536 | 6b `buffers_test` |
  | `DecodeScratch` (kM = 8) | 68,652,864 | 6b `buffers_test` |
  | `PrefillScratch` (kC = 4096, `dequant` included) | 1,961,713,560 | 6b `buffers_test` |
  | **total** | **19,490,436,952 B = 19.490 GB = 18.152 GiB** | derived |
  | **device** | 34,241,150,976 B = **31.891 GiB** | `docs/01-hardware.md:14` |
  | **headroom** | 14,750,714,024 B = **13.738 GiB** | derived |

  Riders, all stated: **Task 4's double-buffered dequant adds 356,515,840 B →
  19,846,952,792 B = 18.484 GiB, headroom 13.406 GiB.** On the **Vishva (bf16
  `lm_head`) checkpoint** the weights are 18,086,971,392 B (`docs/13-loader.md:379`),
  so the total is **21,357,803,352 B = 19.891 GiB, headroom 12.005 GiB**. By
  6b's ruling R7 the prefill scratch is allocated lazily, so a **decode-only**
  engine's residency is byte-identical to today's and spec §6.5 stays checkable
  rather than argued.

  **The two largest prefill rows are the ones this plan put there, and both are
  the price of the design rather than slack:** `partials` at 570,425,344 B is
  the GEMM's fp32 `[C][N]` output at the widest `N` - 6b's ruling R1 sized it
  and noted that an `S = 8` rectangle would be 4,563,402,752 B, so the `S = 1`
  the GEMM makes natural is what keeps this row under 0.6 GB; `dequant` at
  356,515,840 B is spec §3.1's one reusable scratch. Add these two sentences to
  `buffers.h` beside `kC`, in the same "one home for the number" style
  `DecodeScratch::kAttnBlock`'s comment uses.
- [ ] **Step 2: pin it in `buffers_test` and then confirm it on the box.** Add
  the whole-device total above as a fourth assertion beside 6b's three, computed
  from `LoadReport` rather than hardcoded on the weight side (the weight bytes
  are a property of the checkpoint, so the test asserts
  `report.read_per_token + report.embed_bytes + rope + p.bytes() + s.bytes() +
  pf.bytes() < device_total` and **prints** both). Then run a real load at
  `--max-len 16384` on **both** checkpoints and paste the printed
  persistent/scratch/prefill totals against the table. Nothing here is a bench
  row.
- [ ] **Step 3: `tests/prefill/prefill_chunking_test.cc`.** 6b gates the
  multi-chunk *numerics*; this test gates the *decomposition*, which is cheap and
  currently unasserted. `Engine::prefill` records the number of chunk iterations
  it ran and exposes `uint32_t last_prefill_chunks() const` (add it if 6b did
  not). Then: 4096 ids of the cycled bench prompt at the default chunk → **1**
  chunk, `control_->pos == 4096`; 5000 ids → **2** chunks (4096 + 904),
  `pos == 5000`; 5000 ids at `chunk = 1024` → **5** chunks (1024×4 + 904),
  `pos == 5000`. In every case `control_->cur_token[0]` is populated, i.e. the
  `Control` handoff of spec §3.5 holds across a boundary. If 6b already asserts
  the decomposition somewhere, extend that test instead of adding a second one.
- [ ] **Step 4: price the fallback, do not take it.** Spec §3.6 names C = 2048
  as the fallback and §11 lists memory at C = 4096 as a risk. With 13.7 GiB of
  headroom on RTN and 12.0 GiB on Vishva the risk does not fire, so **4096
  stands**; record the fallback's price anyway so a future max_len or a second
  weight copy has a number to reach for: halving `kC` halves every 4096-scaled
  row of 6b's table - `resid`, `x`, `partials`, `ab_out`, `norm_sumsq`, `gdn_o`,
  `mixer_out`, `gdn_xb`, `gdn_g`, `gdn_beta`, `gdn_A`, `gdn_A2`, `gdn_w`,
  `gdn_u`, `ids` - and leaves `attn_q/attn_gate/attn_part` (sized by `kAttnC`),
  `logits`, `argmax_part`, `gdn_seed` and `dequant` untouched. Those two halves
  of 6b's table are **1,195,196,416** and **766,517,144** and they sum to 6b's
  pinned 1,961,713,560, which is the check that the split is right; halving the
  first gives a scratch total of **1,364,115,352 B** and a device total of
  **18,892,838,744 B = 17.596 GiB**. It also **doubles the dequant's share of chunk time from the
  derived 4.5% to 8.6%** (Task 2 Step 1), which is why it is a fallback and not
  a default.
- [ ] **Step 5: Commit** -
  `test(prefill): pin the whole-device total at kC=4096 and the chunk decomposition`

### Task 6: gates

**Files:**
- Modify: `tests/CMakeLists.txt` (the multi-chunk gate registration)
- Test: 6b's `prefill_gate_test`, `prefill_consistency_test`,
  `prefill_determinism_test` (unchanged); the new multi-chunk golden gate

**Interfaces:**
- Consumes: 6b's three prefill tests and the `oracle-out-long` golden set;
  `tests/golden/golden_gate_test.cc`'s `argv[1]` oracle-directory convention
  (`:348`).
- Produces: a registered `prefill_gate_long_test` at C = 1024.

- [ ] **Step 1: re-run 6b's three, unchanged.**
  `ZE_AFFINITY_MASK=1 tools/box.sh test "prefill_(gate|consistency|determinism)_test"`
  on **both** checkpoints - RTN via ctest, Vishva via the manual invocation
  docs/14 documents. Not one line of those tests moves in this plan; if one
  needs to move, that is a finding, not an edit.
- [ ] **Step 2: the multi-chunk golden gate (spec §6.2).** A new test that
  prefills `oracle-out-long`'s ≥ 2048-id prompt at **C = 1024** - so every chunk
  boundary is crossed in attention-over-cache and in the GDN state carry - then
  generates and applies the existing tie-aware gate (determined rows exact, tie
  rows set-membership, teacher-forced tails) against the recorded oracle.
  Registered as `prefill_gate_long_test` with the oracle directory as `argv[1]`,
  defaulting to `oracle-out-long`, exactly as `golden_gate_test` takes
  `oracle-out`. The oracle set itself is 6b's (registered by its
  `OUT_DIR=oracle-out-long IDS_DIR=... tools/oracle/golden.sh` run); this plan
  consumes it and does not re-run the oracle.
- [ ] **Step 3: say what passes now, and what does not.** The C = 1024 gate
  **runs the full production path for the GEMM and the chunked GDN** and
  exercises both kinds of chunk boundary - attention over a cache that spans
  earlier chunks, and the GDN state carry. Its attention, however, is still 6b's
  L1 stand-in (`attn_prep_l1` / `attn_l1`: decode's unmodified
  `attn_decode`/`attn_reduce` at an `M = 64` binary, driven in 64-query
  sub-chunks, `attn_part` = 405,798,912 B, 6b ruling R5) - the numerics are
  decode's attention kernel, not a flash kernel, and that binary is itself a
  recorded exception to the runtime-`M` rule. **Plan 6d must re-run this gate
  after it lands L3**; until
  then a pass here is evidence about the GEMM swap and the chunked GDN, not
  about prefill attention. Write that sentence into the test's header comment
  and into the report - it is the difference between a gate and a claim.
- [ ] **Step 4: decode untouched (spec §6.5).** Re-measure the decode gate rows
  at this plan's final sha, record-grade, idle box: **32.22 t/s (RTN) / 29.33
  (Vishva)** at `2a7df0b`. Inside day drift (≤ 0.09%) or the difference is
  explained in the report. Also re-assert 774 launches / 19 modules.
- [ ] **Step 5: Commit** -
  `test(prefill): multi-chunk golden gate at C=1024 on oracle-out-long`

### Task 7: attribution and records

**Files:**
- Modify: `src/cli/b70_decode.cc` (the prefill attribution print),
  `src/runtime/engine.{h,cc}` (the per-op timing hooks),
  `docs/12-kernels.md` (prefill mechanism section),
  `docs/15-step-anatomy.md` (prefill-step anatomy rows)
- Test: none new; the attribution is an instrument

**Interfaces:**
- Consumes: `l0::EventPool` / `l0::Event` / `TimerCalib`
  (`src/l0/event.h`), `l0::CmdList::launch`'s existing `Event* signal`
  parameter; 6b's `--pp-chunk` (R4).
- Produces: a per-kernel prefill attribution table; the before/after `--pp` rows;
  the crossover sweep; the two doc sections. **No `docs/BENCHMARKS.md` row -
  that is plan 6e.**

- [ ] **Step 1: the instrument, and which half of it is which.** `capture.cc`'s
  `ProfileEvents` is tied to a *captured* list and its label vector, so it does
  not apply here. What does apply is the layer underneath it, unchanged:
  - **L0 kernels** (`pf_dequant_tile`, every prefill `prep_*`, the attention
    route, the GDN chunk) get **real device kernel timestamps**.
    `l0::Event::duration_us()` reads `kernelStart → kernelEnd` through the
    `TimerCalib` that `EventPool` queries once (`src/l0/event.h:6-24,53-56`) -
    the same instrument `--profile` uses, one layer down. 6a's
    `Context::launch` has no place for a signal event, so this step depends on
    **ruling R2b** (the additive trailing `l0::Event* signal = nullptr`). Read
    the durations after the `Context::wait()` that already follows each launch;
    an unsignalled event makes the driver return `NOT_READY`, which throws.
  - **The SYCL GEMM** is timed by **host wall clock** around its submit and the
    following `Context::wait()`. The queue is in-order and single-buffered, the
    kernels are 0.25-16 ms, and the wait overhead measured in Task 3 Step 5 is
    the correction term - state the wall-clock nature in the table header. **If**
    6a constructed the queue with `sycl::property::queue::enable_profiling`, use
    `sycl::event::get_profiling_info<command_start/command_end>` instead and say
    so in the same header. One or the other, never a mixture within one table.
  Add a `--pf-profile` flag (bench-only, like `--profile`) that turns the hooks
  on; it **never produces a bench row**, the same rule `--profile` carries
  (`docs/15`, "Profile mode is not bench mode").
- [ ] **Step 2: pre-register, in the report, before running anything.**
  - Derived GEMM FLOP per 4096-chunk: **199.29 TFLOP** (the six int4 linears;
    per-shape breakdown in the report). At P2's measured `T` TFLOP/s the linear
    time is `199.29/T` s - at the spec's 90 TFLOP/s prediction, **2214 ms**, i.e.
    **1850 t/s from the linears alone (derived)**.
  - Derived dequant per chunk: **104.4 ms**, C-independent, **4.5% of the
    derived 2318 ms chunk at C = 4096 / 8.6% at C = 2048**.
  - **The matched-C row is predicted to be a regression.** At C = 64 the GEMM's
    256-row M tile is 75% empty and the dequant's 104.4 ms is amortised over
    3.11 TFLOP instead of 199.29. Predict the C = 64 row **no better than 6b's
    widened GEMV, plausibly 2-4× worse**.
  - **Crossover, derived:** setting the widened GEMV's optimistic vector-fp32
    ceiling (45.9 TFLOPS, spec §1) against the GEMM path
    (`104.4 ms + 199.29·(C/4096)/90 s`) gives equality at **C ≈ 200**. Band, on
    the uncertainty in both ceilings: **C ∈ [128, 1024]**. The sweep measures it.
- [ ] **Step 3: the rows.** Idle box (DRM-fd evidence pasted), medians of 3, RTN
  checkpoint, device-side `pp` (loader-excluded, first-token-inclusive).
  **R0 must be taken on 6b's closing sha, before Task 3 deletes
  `pf_gemv_int4_M`** - the before/after here is across two shas, not two modes
  of one binary, and after the deletion it cannot be re-taken without a revert:
  - **R0** - 6b's closing sha, `b70-decode --bench --pp 4096 --tg 256` at
    `--pp-chunk 4096` **and** at `--pp-chunk 64`: the *before*, at both widths,
    so the widened GEMV's own C-dependence is on record.
  - **R1** - this plan's HEAD at `--pp-chunk 64`: the matched-C isolation row, so
    the GEMM swap is separated from the chunk width.
  - **R2** - this plan's HEAD at `--pp-chunk 4096`: the production row.
  - **Sweep** - `--pp-chunk ∈ {64, 256, 1024, 4096}` on this plan's HEAD, one run
    each, against R0's two points, to locate the crossover Step 2 predicts.
  Every row labelled **device-side pp**, with checkpoint, chunk width, sha and
  box conditions. The unchanged `tg` is reported beside each per spec §7.
- [ ] **Step 4: the per-kernel table.** `--pf-profile --pp 4096 --pp-chunk 4096`:
  µs per launch and ms per chunk for `pf_dequant_tile` (per shape), `gemm_bf16` (per
  shape), the prep family, the GDN chunk, the attention route, `lm_head`, and
  the `Context::wait()` total against the §3.6 1% interop bar. This is the
  composed-ceiling evidence plan 6e's memo needs; hand it over as data, not as a
  verdict.
- [ ] **Step 5: docs/12 - the prefill mechanism section.** New section, in this
  document's voice (what it computes, how the work is assigned, **why**, what
  was rejected, the numbers that decided it): the dequant scratch (both layouts'
  geometry, the traffic table, the measured GB/s, the bf16-rounding cost
  paragraph from Task 2 Step 2), the GEMM interface (Intel's config verbatim
  with its file:line provenance, the row-major-B finding, the
  `PersistentScheduler`/no-atomics evidence, the alignment rules), the S = 1
  collapse and what it does and does not change about rounding (Task 3 Step 1),
  the chunk-width arithmetic (Task 5 Step 1), and the double-buffer lever's
  outcome (Task 4) whichever way it went. Rejected-and-recorded: the layout-1
  SLM-transpose store, a second prefill weight layout (spec §11), fusing the
  dequant into the GEMM (explorer-3 §3: Intel measured that path **below** the
  dequant-then-stock-GEMM fallback on BMG for every shape in their prefill
  sweep).
- [ ] **Step 6: docs/15 - the prefill-step anatomy rows.** A prefill counterpart
  to the decode anatomy: per-family ms per 4096-chunk from Step 4, the derived
  composed figure beside it, and the same three warnings the decode section
  carries (profile mode is not bench mode; the inter-kernel gap is an upper
  bound; a mean is not an attribution). Explicitly: **no row here is a bench
  row, and `docs/BENCHMARKS.md` gains nothing in this plan** - plan 6e owns the
  §7 gate and the record rows.
- [ ] **Step 7: Commit** -
  `docs(prefill): L2 mechanism and step anatomy - <X> t/s device-side pp at C=4096`

---

## Plan self-review (2026-09-04, at authoring)

**Spec coverage.**

- **§3.1 (the coupling cut, dequant scratch)** → Task 2: 6a P3's OpenCL C kernel
  reading decode's layout-0/1 tiles with `gemv.cl`'s exact dequant contract,
  bound to one reusable 356,515,840-B scratch, with the traffic table, the
  bit-exactness bar raised to the production shapes, the scratch bound, and the
  reconciliation of the spec's own ~1.2 ms / 8-16% figures against this plan's
  0.765 ms / 4.5%. Decode's layouts, kernels and capture are untouched -
  enforced in Global Constraints and re-asserted in Task 3 Step 6 and Task 6
  Step 4.
- **§3.2 (the plain-GEMM interface, inherited)** → Task 1: `sycl-tla`'s stock
  bf16 GEMM at Intel's own config, built with the system `icpx` at the full
  path, launched on a SYCL queue constructed from our L0 handles (the
  `cudaStream_t = sycl::queue*` route, evidenced at
  `gemm_universal_adapter.h:551`), on the USM pointers the engine owns, with
  runtime `M`. The interface is testable (`gemm_test`) and swappable (`gemm.h`
  names no CUTLASS type).
- **§3.5 (`a‖b`, `lm_head`, the `Control` handoff)** → Task 3 Step 2 leaves both
  bf16 routes alone; Task 5 Step 3 asserts `pos` and `cur_token` across a chunk
  boundary.
- **§3.6 (chunk width, buffers, execution model)** → Task 5 for the whole-device
  arithmetic that makes `kC = 4096` (already 6b's constant) the *chosen* width
  against 31.891 GiB, with the C = 2048 fallback priced; Task 3 Step 2 for the
  first-cut queue-wait ordering the spec explicitly permits, Task 4 for the
  overlapped version; Task 3 Step 4 for the under-1% interop bar.
- **§5 (L2)** → the whole plan. "`dequant_tile` + `sycl-tla` GEMM behind the
  §3.2 interface at C = 4096 (or 2048 by P2/P3)" = Tasks 1-3 + 5; "correctness
  bars unchanged" = Task 6 Steps 1 and 4; "the long-prompt multi-chunk gate
  added" = Task 6 Step 2.
- **§6 (correctness bars)** → 6.1 and 6.3 via Task 6 Step 1 (6b's tests,
  unchanged, both checkpoints), with 6.3's token-identity risk pre-registered in
  Task 3 Step 5; 6.2 via Task 6 Step 2 with the honest scope statement in Step 3;
  6.4 via Task 1 Step 5's bitwise determinism plus the `PersistentScheduler`
  source evidence and the no-fp-atomics constraint; 6.5 via Task 6 Step 4;
  6.6 via Global Constraints and the 774/19 re-assertion.
- **§8 (constraints)** → quoted verbatim at the top; the box workflow, JOBS 44,
  the full-path system `icpx` and the pinned `sycl-tla` (both 6a's
  `cmake/prefill.cmake`, with the revision reconciled by R5 and re-verified in
  Task 1 Step 1), `ZE_AFFINITY_MASK=1`, the
  measured/derived/estimated labelling on every number, nothing pushed, and the
  1024-multiple chunk boundaries (kC = 4096, the gate at 1024, the sweep at
  {64, 256, 1024, 4096}) that keep prefix caching reachable.
- **§9 (records)** → Task 7 Steps 5-6 (docs/12 mechanism, docs/15 anatomy).
  docs/01's measured XMX rate, docs/04, docs/06's `gemmstone` correction and
  docs/BENCHMARKS' pp rows are **deliberately not** this plan's (6a's P2 and
  6e's gate own them) and no task touches them.
- **§10 (out of scope)** → no task writes a DPAS kernel in OpenCL C, touches
  inline vISA, adds prefix caching, batching, fp8 KV or MTP.
- **§11 (risks)** → toolchain coupling: 6a owns the pin and the build, Task 1
  Step 1 re-verifies the four load-bearing API facts against the ruled revision
  before any code, and Task 1 Step 4 keeps the `B70_PREFILL=OFF` tree green.
  Memory at C = 4096: Task 5
  Step 1's whole-device table, both checkpoints, with the C = 2048 fallback
  priced to the byte. Dequant overhead larger than derived: Task 2 Step 6
  measures it, Task 2 Step 5 is the geometry lever and Task 4 the overlap lever;
  spec's >20% escalation path (a second weight layout) is named in Task 3 Step 5
  and Task 7 Step 5 as recorded-and-rejected.

**No duplication of plans 6a and 6b.** This plan was drafted against
`interfaces.md`, then reconciled against 6a and 6b as written - both appeared
mid-authoring and both are further along than `interfaces.md`. What they own and
this plan therefore does **not** write:

| owned by | what |
|---|---|
| 6a T0 | the `sycl-tla` pin and the whole icpx build (`cmake/prefill.cmake`, the `src/sycl/` `ExternalProject`, `b70_link_prefill`), `runtime::prefill::Context`, `context_sycl.h` |
| 6a P3 | the dequant kernel `pf_dequant_tile`, its variant registration, and the 256×64 bit-exactness case |
| 6b R1 | `partials` as one `S = 1` fp32 `[kC][34816]` rectangle and the one-slice fold in every consumer |
| 6b R4 | `gdn_chunk` reading fp32 partials and rounding inside - so there is no fold-and-round step to feed |
| 6b R5 | the L1 attention stand-in already driven in 64-query sub-chunks - so there is no bridge to build for C > 64 |
| 6b T1 | `PrefillScratch::kC = 4096`, `kAttnC = 64`, the `dequant` field, and the three pinned byte totals |
| 6b #3 | the `--pp-chunk` CLI flag |

What is left, and is this plan: `gemm_bf16` (Task 1), the production binding and
production-scale bar for 6a's kernel (Task 2), the substitution plus the
retirement of `pf_gemv_int4_M` (Task 3), the overlap lever (Task 4), the
whole-device arithmetic and the chunk-decomposition test (Task 5), the
multi-chunk gate (Task 6), and the attribution and records (Task 7).

**Five things this plan was going to build were deleted on reading 6a and 6b:**
its own `cmake/sycl_tla.cmake` and `-fsycl-link` route (6a rejects that route by
name and gives the reason), its own `context_fwd.h` (6a's SYCL-free `context.h`
makes it unnecessary), its own `l0::CmdList::immediate_async` (6a's list is
already asynchronous), its fold-count-1 prep variants and `partials` S-collapse
(6b R1 did both), and its 64-query sub-batch bridge (6b R5 has it). One request
was **withdrawn** (`--chunk`, in favour of 6b's `--pp-chunk`). Task 4 may be
**skipped by ruling** if 6a P3 Step 9 has already answered the double-buffering
question negatively on measurement.

**Placeholder scan.** No TBD, no TODO, no "similar to". Every file that is
created has its content or its exact structure in the plan; every command is
runnable as written. The places that depend on a sibling plan's shape - 6a's
final `dequant.cl` grid (Task 2 Step 3), 6a's queue-profiling property (Task 7
Step 1), 6a P3 Step 9's double-buffering verdict (Task 4), whether P3 met its
GB/s prediction (Task 2 Step 5), whether 6a already added an equivalent variant
namer (Task 2 Step 2), and whether 6b already asserts the chunk decomposition
(Task 5 Step 3) - each state **both branches concretely** and say which to
record in the commit,
the pattern plan 5's Task 4 used for its repack contingency. `<X>`, `<D>`,
`<total>`, `<Y>` and `<verdict>` appear only in commit messages and are
measurement outputs.

**Type consistency vs `interfaces.md` and plan 6a.**
`runtime::prefill::{Context, GemmDims, gemm_bf16, dequant_to_bf16}`,
`PrefillScratch::kC`, `Engine::prefill(const std::vector<uint32_t>&, uint32_t)`,
`tests/prefill/{gemm_test,dequant_test}.cc`, the bf16 row-major `[M][K]`
activation and `[K][N]` scratch with `lda = K` / `ldb = N`, and the fp32
`[M][N]` output with `ldc = N` are used exactly as `interfaces.md` fixes them.
`runtime::prefill` for host code, the `pf_` kernel prefix,
`src/kernels/prefill/` and `src/sycl/` for the new sources - all as specified.
Where 6a has already requested a deviation this plan uses **6a's** spelling and
adds nothing: `Context(l0::Context&)`, `void* sycl_queue_raw()` +
`runtime::prefill::sycl(Context&)`, the `ze_kernel_handle_t` `launch` overload,
and the kernel name `pf_dequant_tile` (so `kernels::pf_dequant_variant` returns
`pf_dequant_tile_K…_N…_L…`). This plan's own requests are R1 (`loader::DeviceWeight`
for the non-existent `loader::Linear`), R2b (an additive signal-event parameter
on `Context::launch`, without which Task 7 has no device timestamps), R3
(`PrefillScratch::kC` for `interfaces.md`'s internal drift at `:110` vs `:121`),
and R5 (the two `sycl-tla` revisions -
6a's `87f6850` against the checked-out `2db1b7c9` - which is a
two-values-for-one-quantity item and must be ruled before Task 1 writes code).
None is used before it is ruled. The `[K][N]` layout statement is explicitly
**not** challenged: `sycl-tla`'s Xe GEMM takes row-major B natively
(`00_bmg_gemm.cpp:351`) and the VNNI packing DPAS needs happens in the 2D block
load, not in memory.

**Known tension, ruled here.** Task 3 Step 3 **deletes** `pf_gemv.cl`,
`pf_gemv_test.cc` and five compiled variants that plan 6b's Task 4 creates. That
is deliberate and 6b asked for it - its own commit message calls the kernel "L1
temporary, retired by plan 6c" - but it means 6b and 6c must land in that order
and that 6c's diff removes a sibling plan's test. The alternative, leaving a
bindable int4 GEMV compiled that nothing binds, is worse: it invites a rebind
and it keeps five ocloc compiles in every build. If the operator wants the L1
path kept switchable for A/B measurement beyond Task 7's `--pp-chunk` sweep, that
is a ruling to make **before** Task 3, not after - the sweep this plan runs
compares chunk widths on the GEMM path, not GEMV against GEMM at 4096, and
recovering the latter after the deletion means a revert.
