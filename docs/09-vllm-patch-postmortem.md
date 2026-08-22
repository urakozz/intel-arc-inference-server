# Postmortem: months of patching vLLM's XPU kernels

**This document is the strongest argument for why this project exists.**

Before writing a line of new code, understand what was already tried on this exact
box, at what cost, and why it did not work. The failure was not incompetence - the
patches were real optimisations, several of them upstream PRs. It was **optimising
the wrong layer**.

## What was done

A stack of **12 patches** applied to `vllm-xpu-kernels` and built from source
(`docker/patches/kernels-*.patch` in the vLLM tree):

```
01-pr457-grouped-gemm-capture-safe    07-gate-up-sparse-dispatch
02-pr476-gdn-conv1d                   08-grouped-gemv-w4a16
03-pr477-gdn-decode                   09-gate-up-gemv
04-pr494-fused-moe-gate-up            10-gemv-w8a16-fp8
05-gate-up-grid-stride-capture-fix    11-dense-gemv-w4a16
06-sparse-expert-dispatch             12-gemv-row-gate
```

Grouped-GEMM capture fixes, GDN conv1d and decode optimisations, fused MoE
gate/up, sparse expert dispatch, and a family of GEMV kernels. Each was built,
benchmarked, and attributed individually.

## The result

**The patch stack was a prefill win and a decode loss. Pristine `p0` was the
fastest decode configuration on both quantisations.**

MXFP4 (`--max-num-seqs 2`, pp4096/tg256):

| | p0 | v5 (01-05) | v6 (01-06) | v11 (all) |
|---|---|---|---|---|
| pp4096 c1 | 9250 | 9251 | 9573 | **10032 (+8.5%)** |
| tg256 c1 | **72.65** | 67.51 | 66.13 | 68.67 |
| tg256 c2 | **119.74** | 113.06 | 109.38 | 97.20 (**−18.8%**) |

GPTQ-int4:

| | p0 | v5 | v11 |
|---|---|---|---|
| pp4096 c1 | 9232 | 9834 | **10123 (+9.7%)** |
| tg256 c1 | **73.31** | 69.90 | 70.86 |
| tg256 c2 | **120.97** | 118.60 | 102.04 (**−15.6%**) |

Months of kernel work bought **+9% prefill** and cost **up to −19% decode**.

## Why it failed - the diagnosis

**The bottleneck was never the kernels.** Three independent signals, all available
before the work started:

1. Two completely unrelated kernel paths - MXFP4 via `XPUExpertsMxFp4` and
   GPTQ-int4 via `XPUExpertsWNA16` - land **within 1%** of each other (72.65 vs
   73.31). Independent implementations do not agree that closely unless something
   outside both is the limit.
2. Turning XPU graphs off costs **~3×** (same model: 10.55 vs 31.5 t/s). That
   entire factor is host-side.
3. Measured MBU is **50-63%**. The silicon is idle 40% of the time.

Optimising a kernel that is idle 40% of the time waiting for the host cannot
produce more than a few percent - and if it perturbs kernel selection, it can
easily lose more than it gains. Which is exactly what happened.

The structural error is documented in [08-decode-vs-prefill.md](08-decode-vs-prefill.md):
at `M = 1` vLLM runs an **8-row matrix tile with no split-K** on a vector problem.
No amount of tuning inside that kernel fixes the shape of the choice.

## Specific findings worth keeping

### The GEMV auto-select threshold was the concurrency killer

Patches 08-11 added GEMV kernels with an auto-select threshold of
`avg rows/expert <= 2` (`grouped_gemm_xe2_interface.hpp`, `grouped_gemv_mode()`).
Too permissive: at concurrency 2 GEMV still engaged and lost to the DPAS tiles.

Setting `VLLM_XPU_GROUPED_GEMV=0` on the v11 image recovered decode
**GPTQ 102.0 → 118.5 (+16%)** and **MXFP4 97.2 → 112.3 (+15%)** *while keeping*
the prefill gain. Patch 12's row-gate later achieved the same with no env var.

**Lesson: a kernel that is right at `M = 1` and wrong at `M = 2` needs a correct
gate, and the gate is as important as the kernel.** This directly motivates
parameterising on `M ∈ [1,8]` rather than switching between two kernels at a
threshold.

### "int4-only" optimisations were not int4-only

v11 moved MXFP4 numbers (+8.5% prefill c1, −18.8% decode c2) even though the GEMV
path should have been inert there - patches 09/10 edit shared
`fused_moe_gate_up_impl.cpp` headers that the MXFP4 dispatch compiles through.
**Shared headers make "this only affects X" claims false by default.**

### Capture-safety bugs are real and silent

`#457`: the Xe2 grouped GEMM's global atomic tile counter was reset **inside the
kernel**, guarded only by "group 0, lane 0", with no device-wide barrier. Eager
launch timing accidentally separated reset from steals; graph capture/replay did
not → counter carried the previous replay's value → out-of-bounds tiles →
`UR_RESULT_ERROR_DEVICE_LOST` at batch > 1. **This is why `--max-num-seqs` could
not exceed 2.**

`patch 06`: `_compact_experts` ran `index_select` on raw `topk_ids`, which hold
**−1 padding** in the dummy batches used for graph capture →
`ScatterGatherKernels.cpp:233 Assertion idx_dim >= 0` and a wedged engine.

**Both bugs were invisible in eager mode and only appeared under capture.**

### Attribution is harder than it looks

`v0.1.12` → `v0.1.12.1` is **0 commits / 0 files changed** - the `.1` tag is a
rebuild with different build configuration. A source build of `v0.1.12` measured
against the released `.1` wheel therefore mixes a build-config delta into what
looks like a patch effect. **Always measure a pristine source build as the
control**, never the released wheel.

Similarly, patch 13 (oneDNN 3.13) was a **perf no-op** across three models - every
decode number within noise. Take it for correctness, not speed.

### Prefill is noisy, decode is not

Decode reproduces to <0.1% between runs; prefill varies up to 4% run-to-run on
MXFP4. Do not chase a 3% prefill "gain".

## Environment traps (cost days, will recur)

- **ARK must be built from source.** The prebuilt `auto_round_kernel` wheel links
  `libsycl.so.8` (oneAPI 2025.2); the current stack is `libsycl.so.9`. A binary
  wheel cannot be relinked.
- **BuildKit discards all completed work if the docker *client* dies** - even
  finished layers. A plain `docker build` over ssh dies with the ssh session and
  can silently lose 6+ hours. Use `setsid nohup`.
- **`docker builder prune` deletes cache mounts** - the uv wheel cache, cargo, and
  ccache. Use `docker buildx prune --filter type=regular` instead. Never
  `docker volume prune` (it would eat the `open-webui` chat history volume).
- **Build job count is a phase transition, not a dial.** On this 44T/121GB box:
  8 jobs = 52GB peak, zero swap, fastest; 16 = 90-115GB swap, slower despite more
  lanes; 32 = thrash livelock. Exceeding RAM stalls compilers mid-inflation so
  fat template TUs linger and residency compounds.
- **ccache works on `-fsycl` TUs** (1292/1292 cacheable, ~99.5% hit warm), taking
  the host C++ phase from ~7.2h to ~5 min. Device codegen (ocloc, 2405 kernels,
  ~20 min) is **not** cacheable.
- **Checkpoint metadata lies.** Published AutoRound checkpoints have declared MTP
  tensors quantised while shipping them unquantised, breaking load. Verify
  `block_name_to_quantize` / `extra_config` against the tensors actually present
  before blaming the runtime.

## The conclusion this project is built on

> Kernel micro-optimisation on a host-bound workload produced a **net-negative**
> decode result over months of effort.

Fix the host first. Write kernels second - and write them for the right value of
`M`. That ordering is the entire thesis, and it was paid for.
