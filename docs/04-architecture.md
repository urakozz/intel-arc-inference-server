# Architecture

**Approach: SYCL kernels + a Level Zero command list replayed for the decode step.**

## Language and toolchain

- **C++17.** `sycl-tla/CMakeLists.txt:235` pins `CMAKE_CXX_STANDARD 17` with
  `STANDARD_REQUIRED ON`; its README requires "at least C++17". Match it. Bump to
  C++20 only after the thing runs.
- **`icpx`** (oneAPI DPC++) for SYCL, **Level Zero** for the runtime, **CMake**.
- **Builds on the box, not on the Mac - natively, not in Docker.** The host has
  oneAPI 2026.1 (`icpx`), `ocloc 26.27`, IGC 2.38 and the Level Zero headers
  (doc 10). CLion uses a remote toolchain - same arrangement as
  `~/CLionProjects/vllm-xpu-kernels`. The reference container is needed only
  for the Python oracle.
- **Python appears exactly once**, in `tools/`, for offline weight conversion and
  benchmark glue. Never in the serving path - Python *is* the overhead being
  removed.

## Why the decode step is the whole design

At batch 1 this box is **host-bound, not kernel-bound**. Three independent pieces
of evidence from the vLLM work:

- Two entirely different kernel paths (MXFP4 via `XPUExpertsMxFp4`, GPTQ-int4 via
  `XPUExpertsWNA16`) land within **1%** of each other - 72.65 vs 73.31 t/s. If
  kernels were the limit they would not agree that closely.
- Turning XPU graphs off costs **~3×**: the same 27B model measured **10.55 t/s**
  without graphs against **31.5 t/s** with them.
- Measured MBU sits at **50-63%**. The card is idle waiting for work.

So the product is not a faster GEMM. It is a decode step that issues **zero host
work per token**.

## Component layout

```
src/
  loader/      safetensors mmap, dedup by name, per-layer width metadata,
               repack into the canonical on-device layout, upload once
  model/       qwen3_5 graph description: layer types, shapes, weight bindings
  kernels/     SYCL: GDN (conv1d + gated delta rule), RMSNorm, RoPE, SiLU,
               sampling; sycl-tla instantiations for GEMM and flash attention
  runtime/     L0 context, allocations, the captured decode command list,
               KV cache, sequence state
  server/      HTTP, OpenAI schema, SSE streaming
  tokenizer/   byte-level BPE encode/decode, chat template, streaming
               detokeniser - see docs/11. Host code, request path only
tools/         (python) weight conversion (incl. lm_head requantisation),
               probes, benchmark drivers, tokenizer golden vectors
```

Each directory should be understandable without reading the others. The loader
knows about checkpoint formats and nothing about kernels; the kernels know
about memory layouts and nothing about files.

## Execution model

Two distinct paths, deliberately not unified:

**Prefill** - dynamic shapes, runs through an ordinary SYCL queue. Long enough
per call that launch overhead is irrelevant. This is where W8A8 becomes
interesting later (int8 XMX has ~2× bf16 throughput, and prefill *is*
compute-bound).

**Decode** - one token, fixed shapes, built **once** into a Level Zero command
list and replayed. Between replays the host updates **nothing in the list**.

The three values that change per token - KV write offset / position, sequence
length, and the current token id - live in a small **device-resident control
block** that every kernel reads and the sampler writes. The sampler stores its
argmax into `control.cur_token`; the embedding gather at the top of the next
replay reads it; the last kernel in the list increments `control.pos`. The
host's per-token work is `zeCommandQueueExecuteCommandLists`, a fence wait, and
one read of `cur_token` for the stream.

Every buffer address, kernel argument and dispatch dimension is baked in at
capture time, and the list is **byte-identical on every replay**. That is what
makes the capture-safety rule below testable: replay the same list twice from
the same state and diff the outputs.

Level Zero's mutable-command-list extension (`level-zero/include/ze_api.h:14498`,
`ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS`; implemented for Xe2 in
`compute-runtime/level_zero/core/source/mutable_cmdlist/mutable_cmdlist_hw_from_xe_hpg_to_xe3.inl`)
exists and is the **fallback** for a kernel that genuinely cannot read its state
from memory - not the design. Mutating arguments per token is host work, and
host work is what this loop exists to remove.

A side effect worth keeping in view: once the host touches nothing between
tokens, submitting `N` tokens per `zeCommandQueueExecuteCommandLists` is a loop
unroll, not a redesign. v1 does not need it (SSE wants per-token granularity),
but nothing should preclude it.

### Attention under replay

A captured kernel cannot change its grid as the context grows, and the 8
full-attention layers read `seq_len` KV entries. Two honest options:

1. **Fixed grid over `max_model_len`, device-side early-out.** Every work-group
   reads `control.seq_len` and returns if its KV block lies beyond it. One list,
   simple, and the cost of an exiting work-group is microseconds across 8
   layers.
2. **Context-bucketed lists** - capture one list per bucket (1k, 4k, 16k) and
   choose by `seq_len`. Fewer idle slots, `B` lists to keep correct.

Start with 1. Move to 2 only if doc 07 #12 shows the idle work-groups cost more
than ~2% of a step. GDN layers have no such problem - their state is fixed-size.

### Where SYCL stops and Level Zero begins

`sycl-tla` kernels are launched through the SYCL runtime. Putting one into a raw
L0 command list means extracting its `ze_kernel_handle_t` from a named kernel
bundle (`sycl::get_native<backend::ext_oneapi_level_zero>`) and marshalling the
argument struct by hand - possible, but it couples the decode list to SYCL's
launch machinery for kernels that do not need it.

**Recommended, to be settled in the phase-1 spec:** the decode list contains no
`sycl-tla` kernels. Every decode kernel (GEMV, GDN recurrent step, conv1d step,
norms, RoPE, sampler) is a plain kernel compiled offline with `ocloc` to a device
binary and loaded with `zeModuleCreate`. The decode loop then has zero SYCL
runtime in it - the "two paths" above become two toolchains, not two branches
inside one. Prefill (`sycl-tla` GEMM, flash attention, chunked GDN) runs on a
SYCL queue, where launch overhead is irrelevant.

Rejected alternative: `sycl_ext_oneapi_graph`. It is what vLLM's XPU graphs
are; it would work; it would teach nothing about the command-list layer, and
goal 2 is "learn the metal".

Consequence for goal 3 (static binary): `libsycl` is a shared library and is not
meant to be linked statically. The decode path is static-linkable (L0 loader
only); the prefill path is not until its kernels are also prebuilt with `ocloc`.
Accept that for v1.

### What makes this hard

Kernels must be **capture-safe**: no allocation, no host synchronisation, no
state that persists across replays. This is not hypothetical - the
`vllm-xpu-kernels` grouped GEMM had a global atomic tile counter reset *inside*
the kernel with no device-wide barrier. Under eager launch the timing accidentally
worked; under graph capture the counter carried the previous replay's value and
raced into out-of-bounds tile coordinates, producing `UR_RESULT_ERROR_DEVICE_LOST`
at batch > 1. Two of the local patches in the vLLM stack exist solely to fix
this class of bug.

**Rule: any kernel that cannot be replayed byte-identically from the same command
list is a bug, not a limitation.** Design for replay from the first kernel.

## Kernel inventory for phase 1

| Kernel | Source | Notes |
|--------|--------|-------|
| Mixed-dtype GEMM (int4 × bf16) | `sycl-tla` `02_bmg_gemm_mixed_dtype` | The workhorse: qkv, o_proj, gate/up/down, lm_head |
| Flash attention | `sycl-tla` `06_bmg_flash_attention` (prefill); decode attention is a split-KV GEMV-shaped kernel, **write** | 16 of 64 layers; GQA 6:1, head_dim 256, q/k RMSNorm before RoPE, `attn_output_gate` |
| **GDN: causal conv1d, decode step** | **write** - reference `vllm-xpu-kernels/csrc/xpu/gdn_attn/causal_conv1d.hpp` | 48 of 64 layers; a 4-tap depthwise FIR over the 10240-wide qkv at decode |
| **GDN: gated delta rule, recurrent** | **write** - reference `vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp` (`gated_delta_rule_kernel`, SIMD32, 256-thread groups, handles spec-decode via `num_accepted_tokens`) | 48 of 64 layers. Hardest to get right; **not** the hot one - 3 MB of state per layer (doc 03, doc 05). Fuse conv1d + q/k l2norm + recurrence + gated RMSNorm into one kernel |
| GDN: chunked (prefill) | reference `vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` (1634 lines, CuTe) | prefill only; outside the decode-core scope |
| RMSNorm (+ fused residual) | write | trivial |
| RoPE | write | trivial |
| SiLU / gated MLP | write | trivial, fuse into GEMM epilogue |
| Sampling (argmax / top-p) | write | trivial |

Roughly **two-thirds borrowed, one-third original.** The original third (GDN)
is the part that must be *correct*; the GEMV/GEMM is the part that decides the
*speed* (doc 05). Budget correctness time for the first and benchmark time for
the second.

**Kernel count is a first-class design input.** Unfused, the 27B is ~700
kernels per token (doc 03). Inside a replayed list each kernel still pays a
fixed dispatch + drain cost; at 3-5 µs that is 2-3.5 ms of a ~26 ms step -
the same magnitude as the host overhead replay removes. The phase-1 fusion
list, in order of kernels saved:

1. RMSNorm into the following GEMV's prologue (2 per layer, 128 total);
2. `gate_proj` ‖ `up_proj` ‖ SiLU·mul into one GEMV with two weight streams
   (1 per layer, 64);
3. conv1d step + q/k l2norm + recurrence + gated RMSNorm into one GDN kernel
   (3 per GDN layer, 144);
4. `in_proj_qkv` ‖ `in_proj_z` ‖ `in_proj_a` ‖ `in_proj_b` into one GEMV over a
   concatenated weight (3 per GDN layer, 144 - the bf16 `a`/`b` rows ride along
   as a second dtype stream);
5. residual add into the GEMV epilogue (free once 1 is done).

Target after fusion: **~250 kernels per token**. Doc 07 #5 measures the
per-kernel floor before any of this is built, so the fusion list is sized by a
number rather than by taste.

## Server

OpenAI-compatible, single stream for v1. No scheduler, no paged KV, no
continuous batching. A ring KV buffer sized to `max_model_len` is sufficient
when there is exactly one sequence.

Deferring batching is not a shortcut - it isolates the variable being tested
(per-token host + kernel cost) and keeps the first milestone reachable.
Concurrency is a phase-5 concern at the earliest.

The server owns the **tokenizer and chat template** - the component the first
draft of these docs omitted entirely. Both are host code on the request path:
encode once per request, detokenise once per token (off the GPU's critical path,
overlapping the next replay). Options and the recommendation are in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).
