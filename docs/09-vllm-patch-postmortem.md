# Postmortem: months of patching vLLM's XPU kernels

Before this engine existed, months went into patching vLLM's XPU kernels on this exact hardware. This document
records what was tried, what it cost and why it did not work. That failure is the main argument for why this
project exists.

It is for anyone tempted to speed up decode (generating tokens one at a time) by tuning kernels before looking at the
host. The patches were real optimisations, several of them upstream PRs; the mistake was optimising the wrong layer.

## What was done

A stack of 12 patches was applied to `vllm-xpu-kernels` and built from source (`docker/patches/kernels-*.patch` in
the vLLM tree):

```
01-pr457-grouped-gemm-capture-safe    07-gate-up-sparse-dispatch
02-pr476-gdn-conv1d                   08-grouped-gemv-w4a16
03-pr477-gdn-decode                   09-gate-up-gemv
04-pr494-fused-moe-gate-up            10-gemv-w8a16-fp8
05-gate-up-grid-stride-capture-fix    11-dense-gemv-w4a16
06-sparse-expert-dispatch             12-gemv-row-gate
```

They covered grouped-GEMM capture fixes, GDN (Gated DeltaNet) conv1d and decode optimisations, a fused MoE
(mixture of experts) gate/up projection, sparse expert dispatch, and a family of GEMV (matrix times vector) kernels.
Each was built, benchmarked and attributed individually.

## The result

The patch stack was a prefill win and a decode loss. The pristine build, with no patches, was the fastest decode
configuration on both quantisations.

How to read the tables:

- rows: prompt processing at 4096 tokens (pp4096) and token generation of 256 tokens (tg256), in tokens per second,
  with `--max-num-seqs 2`; c1 and c2 are one and two requests at once;
- columns: p0 is the pristine source build, v5 has patches 01-05, v6 has 01-06 and v11 has 01-11.

MXFP4:

| | p0 | v5 (01-05) | v6 (01-06) | v11 (all) |
|---|---|---|---|---|
| pp4096 c1 | 9250 | 9251 | 9573 | 10032 (+8.5%) |
| tg256 c1 | 72.65 | 67.51 | 66.13 | 68.67 |
| tg256 c2 | 119.74 | 113.06 | 109.38 | 97.20 (−18.8%) |

GPTQ-int4:

| | p0 | v5 | v11 |
|---|---|---|---|
| pp4096 c1 | 9232 | 9834 | 10123 (+9.7%) |
| tg256 c1 | 73.31 | 69.90 | 70.86 |
| tg256 c2 | 120.97 | 118.60 | 102.04 (−15.6%) |

Months of kernel work bought about 9% on prefill and cost up to 19% on decode.

## Why it failed

The bottleneck was never the kernels. Three independent signals said so, and all of them were available before the
work started:

1. Two unrelated kernel paths, MXFP4 through `XPUExpertsMxFp4` and GPTQ-int4 through `XPUExpertsWNA16`, land within
   1% of each other (72.65 against 73.31). Independent implementations do not agree that closely unless something
   outside both is the limit.
2. Turning XPU graphs off costs about 3x (the same model: 10.55 against 31.5 t/s). That whole factor is host-side.
3. Measured MBU (memory bandwidth utilisation) is 50-63%. The silicon is idle 40% of the time.

Optimising a kernel that sits idle 40% of the time waiting for the host cannot gain more than a few percent, and if
the change perturbs kernel selection it can easily lose more than it gains. That is what happened.

The structural error is described in [08-decode-vs-prefill.md](08-decode-vs-prefill.md): at `M = 1` (one token at a
time) vLLM runs an 8-row matrix tile with no split-K on a vector problem. No amount of tuning inside that kernel
fixes the shape of the choice.

## Findings worth keeping

### The GEMV auto-select threshold was what hurt concurrency

Patches 08-11 added GEMV kernels with an auto-select threshold of `avg rows/expert <= 2`
(`grouped_gemm_xe2_interface.hpp`, `grouped_gemv_mode()`). That was too permissive: at two requests at once the GEMV
still engaged and lost to the DPAS tiles.

Setting `VLLM_XPU_GROUPED_GEMV=0` on the v11 image recovered decode, GPTQ from 102.0 to 118.5 (+16%) and MXFP4 from
97.2 to 112.3 (+15%), while keeping the prefill gain. Patch 12's row gate later did the same without an environment
variable.

The lesson: a kernel that is right at `M = 1` and wrong at `M = 2` needs a correct gate, and the gate matters as much
as the kernel. This is what motivated designing for `M` in 1..8 rather than switching between two kernels at a
threshold.

### "int4-only" optimisations were not int4-only

v11 moved the MXFP4 numbers (+8.5% prefill at c1, −18.8% decode at c2) even though the GEMV path should have been
inert there: patches 09 and 10 edit shared `fused_moe_gate_up_impl.cpp` headers that the MXFP4 dispatch compiles
through. Shared headers make "this only affects X" claims false by default.

### Capture-safety bugs are real and silent

In `#457`, the Xe2 grouped GEMM's global atomic tile counter was reset inside the kernel, guarded only by "group 0,
lane 0", with no device-wide barrier. Under eager launch the timing happened to keep the reset apart from the work
steals; under graph capture and replay it did not. The counter carried the previous replay's value, tiles went out
of bounds, and the result was `UR_RESULT_ERROR_DEVICE_LOST` at batch > 1. This is why `--max-num-seqs` could not
exceed 2.

In patch 06, `_compact_experts` ran `index_select` on raw `topk_ids`, which hold −1 padding in the dummy batches used
for graph capture. That raised `ScatterGatherKernels.cpp:233 Assertion idx_dim >= 0` and wedged the engine.

Both bugs were invisible in eager mode and appeared only under capture.

### Attribution is harder than it looks

`v0.1.12` to `v0.1.12.1` is 0 commits and 0 files changed: the `.1` tag is a rebuild with a different build
configuration. A source build of `v0.1.12` measured against the released `.1` wheel therefore mixes a build-config
difference into what looks like a patch effect. Always measure a pristine source build as the control, never the
released wheel.

Similarly, patch 13 (oneDNN 3.13) made no performance difference across three models: every decode number was within
noise. Take it for correctness, not speed.

### Prefill is noisy, decode is not

Decode reproduces to within 0.1% between runs; prefill varies by up to 4% from run to run on MXFP4. Do not chase a 3%
prefill "gain".

## Environment traps

These cost days and will recur.

- ARK must be built from source. The prebuilt `auto_round_kernel` wheel links `libsycl.so.8` (oneAPI 2025.2), while
  the current stack is `libsycl.so.9`, and a binary wheel cannot be relinked.
- BuildKit discards all completed work, even finished layers, if the docker client dies. A plain `docker build` over
  ssh dies with the ssh session and can silently lose six hours or more. Detach it.
- `docker builder prune` deletes cache mounts: the uv wheel cache, cargo and ccache. Use
  `docker buildx prune --filter type=regular` instead, and never `docker volume prune`, which takes unrelated named
  volumes with it.
- The build job count behaves like a phase transition, not a dial. On a 44-thread, 121 GB host building this torch
  stack, 8 jobs peak at 52 GB with no swap and are fastest; 16 jobs swap 90-115 GB and are slower despite the extra
  lanes; 32 jobs thrash to a livelock. Once RAM is exceeded, compilers stall mid-way, large template translation units
  linger, and memory use compounds. This engine's own tree has no such problem and builds at full width.
- ccache works on `-fsycl` translation units (1292 of 1292 cacheable, about 99.5% hits when warm), taking the host C++
  phase from about 7.2 hours to about 5 minutes. Device code generation (ocloc, 2405 kernels, about 20 minutes) is not
  cacheable.
- The Rust tokenizer in this engine has its own Cargo cache. `cmake/tokenizer.cmake` finds `cargo` (preferring
  `~/.cargo/bin`) and builds the pinned `tokenizers` crate into `build/tokenizer-rs`. `B70_TOKENIZER=AUTO` enables it
  when Cargo is usable; set `ON` to require it or `OFF` to leave the component out.
- Checkpoint metadata can be wrong. Published AutoRound checkpoints have declared their MTP tensors quantised while
  shipping them unquantised, which breaks loading. Check `block_name_to_quantize` and `extra_config` against the
  tensors actually present before blaming the runtime.

## The conclusion this project is built on

Kernel micro-optimisation on a host-bound workload produced a net-negative decode result over months of effort.

Fix the host first. Write kernels second, and write them for the right value of `M`. That ordering is the whole
thesis of this engine ([04-architecture.md](04-architecture.md)), and it was paid for.
