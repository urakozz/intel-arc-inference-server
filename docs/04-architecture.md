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
- **Linux only.** Ubuntu on the box is the target; no Windows or macOS
  runtime, and no portability shims for them.
- **The server never downloads models.** `hf download <repo>` puts them in the
  standard HF cache; the loader resolves a repo id against that cache
  (`refs/main` → snapshot) or takes an absolute snapshot path. A missing
  snapshot is an error naming the path, not a fetch.

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
  tokenizer/   leaf: HF `tokenizers` 0.22.2 BPE encode/decode (T1); chat template,
               streaming detokeniser - see docs/11. Host code, request path only
third_party/   pinned header-only minja and nlohmann/json dependencies
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

### Prefill as built (spec 2, Stage 1) - dynamic dispatch on two runtimes

The sketch above said "prefill runs through an ordinary SYCL queue". What it
actually is, now that it exists:

* **one in-order, asynchronous L0 immediate command list** for our OpenCL C
  kernels (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, `MODE_ASYNCHRONOUS`), with every
  argument resolved **per launch** - no capture, because arguments change per
  chunk and per layer;
* **one in-order `sycl::queue`** built by interop from the SAME
  `ze_context`/`ze_device`, so the engine's `zeMemAllocDevice` pointers are
  valid USM inside sycl-tla's GEMM;
* `runtime::prefill::Context::wait()` as the ONLY synchronisation, draining
  both.

**The two queues are not orderable against each other on this device, and that
is a measurement, not a design choice.**
`zeDeviceGetCommandQueueGroupProperties` on the B70 reports exactly two groups:
ordinal 0 COMPUTE+COPY+COOPERATIVE with `numQueues = 1`, ordinal 1 COPY-only
with `numQueues = 1`; `(0,1)` and `(1,1)` are refused with
`ZE_RESULT_ERROR_INVALID_ARGUMENT` (ruling A24, Probe B). There is one compute
queue, the two software queues share it, and no device-side cross-runtime
dependency primitive exists. So every L0→SYCL and SYCL→L0 boundary in the walk
is a **host `wait()`**: 656 of them per chunk, at a measured 22.35 µs each.
`docs/15-step-anatomy.md` prices them.

Two host-side rules follow and are stated once here rather than at each site:

* **a host write to `Control` is always preceded by a `wait()`.** The list is
  asynchronous; a write racing a launch that reads `pos` would be a bug no test
  could reproduce. `Engine::prefill` waits before every chunk's `memcpy` of the
  ids and before both `Control` writes.
* **`Engine::prefill` is compiled into `b70_prefill_host`, not `b70_runtime`.**
  b70_runtime is linked by every decode binary and test and must acquire no
  dependency on `libb70_prefill.so`, so `PrefillEngine` is incomplete in
  `engine.h` behind a function-pointer-deleter `unique_ptr` and a translation
  unit that never calls `prefill()` emits no reference to any prefill symbol.

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

Measured, 2026-08-25: the captured decode step is **774 kernels** - 48 GDN
layers × 12 + 16 full-attention layers × 12 + 6 at the token boundary
(`embed_gather`, the final norm's two launches, `lm_head`, two argmax stages),
i.e. 576 + 192 + 6.
It was **645** (× 10, × 10, 5) until spec 1.5's lever L1 split every
`prep_res_norm` site into `prep_res_fold` + `prep_norm_finish`. Each layer holds
two of those sites and the boundary holds one, so 129 sites × 2 = 258 launches
where there were 129, and 645 + 129 = 774
([12-kernels.md](12-kernels.md), `prep_res_fold`).
`runtime::build` (`src/runtime/capture.cc`) is the one walk that binds them,
and `tests/runtime/replay_determinism_test` is the diff above, run for real:
same list, same state, twice, bitwise on the token ids, on a per-layer residual
tap and on every persistent buffer - then once more from a re-zeroed state.

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

### Device selection

`--device N` is the single authoritative knob (user decision, 2026-08-25),
and it must govern **both** execution paths - the raw-L0 decode loop and the
future SYCL prefill - so they always land on the same physical card. The
precedence contract, top wins:

1. **`--device N`** - selects GPU `N`. For the SYCL path the engine binds the
   *same* card by Level Zero handle interop (`sycl::make_device` from the
   `ze_device_handle_t`), not by trusting a second selector to agree.
2. **`ONEAPI_DEVICE_SELECTOR=level_zero:N`** - the default when the flag is
   absent. The SYCL runtime honours it natively; the raw-L0 path parses it
   itself (L0 does not read it). `level_zero:*` and unset both mean device 0
   until P/D disaggregation exists.
3. **`ZE_AFFINITY_MASK`** - a Level Zero *driver*-level filter that sits
   underneath both: it restricts which devices are enumerated at all, and it
   already works with our binaries today. We respect it and never set it;
   note that under a mask, `--device`/selector indices refer to the masked
   (visible) view - the same re-numbering the driver gives everyone.

Rationale: the box has two B70s, and even on PCIe 3.0 the second card
usefully serves a second *independent* request (two single-stream engines
side by side) long before any cross-GPU work exists. Implementation lands
with the runtime/CLI (plan 3 task 1); until then probes and tests bind
device 0 explicitly, and `ZE_AFFINITY_MASK=1` is the working stopgap for
running them on the second card.

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

**Updated, spec 2 Stage 1:** the prefill path's OWN kernels now ARE prebuilt
with `ocloc` - the `pf_*` family in `src/kernels/prefill/`, the same
`add_ocloc_kernel` rows the decode set uses. The remaining non-AOT term is
sycl-tla's GEMM, which is `libb70_prefill.so`: an icpx-linked shared library,
because `-fsycl` device code is only turned into a registered device image by
the clang driver's LINK step and a g++ link of icpx objects drops the images
silently. So the boundary moved from "the whole prefill path" to "the GEMM
alone", and the .so links no project archive.

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

### The prefill family, as built (spec 2, Stage 1 L1)

Every row is a separate `add_ocloc_kernel` in `src/kernels/prefill/`, none is
bound by `capture.cc`, and `M` is a runtime argument in all of them.

| Kernel | What it is | Retired by |
|---|---|---|
| `pf_embed_gather`, `pf_res_fold`, `pf_norm_finish`, `pf_silu_mul`, `pf_gated_head` | decode's `prep`/`embed` family at runtime `M` and `S = 1` | - |
| `pf_ab_proj` | the a‖b bf16 GEMV at the measured `{16, 16}` tiling | plan 6c, if `gemm_bf16` absorbs the shape |
| `pf_dequant_tile` | one int4 linear → the bf16 `[K][N]` scratch | - (A23/A24: this IS the GEMM path) |
| `pf_attn_prep` / `pf_attn_prep_q16` | q/k norm + RoPE + the KV write; the `_q16` build is ruling A9's bf16 `q` | - |
| `pf_softmax_causal`, `pf_attn_gate` | the composed attention's own two kernels (ruling A14/A16) | plan 6d tunes, does not retire |
| `pf_gdn_conv` (4 entries), `pf_gdn_wy` (4), `pf_gdn_scan` | `gdn_chunk`'s ten launches | - |
| `pf_probe_chain` | the execution-context probe; bound only by `context_test` | - |

Deliberately **absent**: `pf_gemv_int4_M` (plan 6b Task 4's temporary int4
GEMV, skipped by ruling - `dequant_to_bf16` + `gemm_bf16` landed instead) and
the `M = 64` `attn_decode`/`attn_reduce` variants of ruling A10 (retired by
ruling A14 before they were built).

Roughly **two-thirds borrowed, one-third original.** The original third (GDN)
is the part that must be *correct*; the GEMV/GEMM is the part that decides the
*speed* (doc 05). Budget correctness time for the first and benchmark time for
the second.

**Kernel count is a first-class design input.** Unfused, the 27B is **774**
kernels per token - the estimate here was ~700; the built list, counted by
`runtime::CapturedStep::kernel_count`, was 645 when it was first walked
(2026-08-25) and is 774 since spec 1.5's lever L1 (above). Inside a replayed
list each kernel still pays a fixed dispatch + drain cost. The 3-5 µs guessed
here turned out to be **0.52 µs** measured (`probe_replay`, doc 07 #5), so the
whole list costs ~0.34 ms rather than the 2-3.5 ms feared. The step it is a
fraction of is now measured too: **42.14 ms/token** at depth 4096, tg 256,
median of three runs 2026-08-25 (`tools/bench_decode.sh`, docs/05 and
BENCHMARKS.md) - so the launches are **~1%** of a token, and fusion stays
deferred out of phase 1 entirely (spec §4.1) instead of being its first move.
The measurement also says where the effort *should* go, which is not here: 67%
of that step is GEMV and the other 33% is time inside `prep` / `gdn_step` /
`attn`, not launch overhead. The list below survives as the phase-1 fusion order if a
later measurement makes it worth the correctness risk:

1. RMSNorm into the following GEMV's prologue (2 per layer, 128 total);
2. `gate_proj` ‖ `up_proj` ‖ SiLU·mul into one GEMV with two weight streams
   (1 per layer, 64);
3. conv1d step + q/k l2norm + recurrence + gated RMSNorm into one GDN kernel
   (3 per GDN layer, 144);
4. `in_proj_qkv` ‖ `in_proj_z` ‖ `in_proj_a` ‖ `in_proj_b` into one GEMV over a
   concatenated weight (3 per GDN layer, 144 - the bf16 `a`/`b` rows ride along
   as a second dtype stream);
5. residual add into the GEMV epilogue (free once 1 is done).

Target after fusion: the five items above remove 128 + 64 + 144 + 144 + 0 = 480
launches, so a fully fused step would be **~165** kernels per token (estimated -
arithmetic on a list where nothing is built, and off the 645-launch list it was
written against). **774** is the *unfused* count and it is measured, not a
target: `runtime::CapturedStep::kernel_count`, 2026-08-25. **Note the direction
the first two levers moved it**: L1 *added* 129 launches to buy 2.4 ms, at a
derived 0.733 µs each. Launch count is not the lever it was once feared to be,
in either direction.
Doc 07 #5 measures the per-kernel floor before any of this is built, so the
fusion list is sized by a number rather than by taste. Measured 2026-08-22:
**0.52 µs/kernel** (`noop`) and **0.63 µs/kernel** (`ctrl_read`) inside a
replayed list - the 3-5 µs estimate above is ~6× pessimistic, so the 645 kernels
of that list were priced at 0.335 ms of the 42.14 ms step measured 2026-08-25 (**0.8%**,
estimated). **Superseded 2026-08-25 by an in-situ measurement, same conclusion:
0.473 ms, 0.733 µs/launch, 1.1% of the step** - derived from the profiler's
per-kernel timestamps (doc 07 #5, [15-step-anatomy.md](15-step-anatomy.md)),
and above both probe floors because every decode kernel reads the control
block. Fusion is *not* on the phase-1 critical path and the unfused list ships
first.

What the profile *does* say about this list is that the fusion candidates above
were mispriced in kind, not in size: what items 1 and 4 would really buy is not
the 0.7 µs launch but the **parallelism** of the kernels they absorb -
`prep_res_norm` ran on one work-group and `a‖b` on two. Spec 1.5 attacks that
directly, without fusing anything, and item 4 is now **moot**: lever L2 took
`a‖b` from 2.341 to 0.256 ms/token inside the kernel (the L2 run's own
before/after pair - the anatomy run reads the same pre-lever row as 2.335, 0.3%
away), so absorbing it into
`qkv‖z` would buy 0.256 ms at best and cost a fused kernel. (The unit that
turned out to matter is the **subgroup**, not the work-group - docs/15 §L2
measured the work-group reading and it bought nothing.)

## Server

OpenAI-compatible, single stream for v1. No scheduler, no paged KV, no
continuous batching. A ring KV buffer sized to `max_model_len` is sufficient
when there is exactly one sequence.

**Status (spec 3 T3).** `src/server/` now provides `/v1/models`,
`/v1/chat/completions`, and `/v1/completions`, including OpenAI-shaped errors
and SSE responses, against the `TokIface`, `TemplateIface`, and `EngineIface`
interfaces. Requests share one engine through a bounded FIFO; requests beyond
the configured waiting depth receive 503. The intentional parsing deviation is
that an absent `temperature` means greedy sampling. The parser also accepts the
`min_tokens`, `ignore_eos`, `return_token_ids`, and
`stream_options.include_usage` extension fields.

Deferring batching is not a shortcut - it isolates the variable being tested
(per-token host + kernel cost) and keeps the first milestone reachable.
Concurrency is a phase-5 concern at the earliest.

### Follow-on: prefix caching (for Open WebUI / opencode use)

Not in specs 1-3. The benchmark runs `--no-cache`, and the KV ring and GDN
state are already persistent device memory - which is the prerequisite. Three
stages, to be planned after plan 3 is done:

1. **Nothing in specs 1-2.** Single stream; state buffers persistent.
2. **Session continuation** - small, after spec 2: keep the last request's
   final state resident; if the next prompt's token ids start with the
   previous prompt + generated ids, continue from that state and prefill only
   the new tokens. Covers the multi-turn chat case with zero snapshot
   management: a prefix check and a `pos` update.
3. **Block snapshots** - its own spec, probably alongside batching: a pool of
   `(token-hash-chain, position, ~151 MB GDN state + KV slice)` entries at
   1024-token boundaries, LRU, restored then tail-recomputed - the vLLM
   scheme. Pays off only once several concurrent conversations share system
   prompts.

The server owns the **tokenizer and chat template** - the component the first
draft of these docs omitted entirely. Both are host code on the request path:
encode once per request, detokenise once per token (off the GPU's critical path,
overlapping the next replay). Options and the recommendation are in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).
