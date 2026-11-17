# Prior art: what to borrow, what to avoid

This document lists the other projects whose code taught this one something, what each is good for, and the traps
found in them. It is for a reader about to write a kernel or chase a driver surprise who wants to know where to
look first. None of it is built into the engine: the engine is C++ on Level Zero with its own OpenCL C kernels,
and these projects are references to read.

## `sycl-tla`, the kernel templates

`sycl-tla` is a fork of NVIDIA's CUTLASS that extends CUTLASS and CuTe (CUTLASS's layout and tiling library) to
Intel GPUs through SYCL. It is header-only C++ templates. BMG stands for Battlemage, the B70's architecture, so
every `*_bmg_*` example targets this silicon directly.

| Example | Use |
|---------|-----|
| `00_bmg_gemm` | Baseline bf16 GEMM, the first thing to build to check a toolchain |
| `02_bmg_gemm_mixed_dtype` | int4 and int8 weights times bf16 activations |
| `03_bmg_gemm_streamk` | Stream-K, a relative of split-K. Read it for how `sycl-tla` fills the device when there are too few `M x N` tiles; the decode GEMV solves the same problem with split-K |
| `05_bmg_gemm_with_epilogues` | Fusing SiLU, bias and residual into the GEMM |
| `06_bmg_flash_attention` | Flash attention, a different approach from the engine's own `pf_flash_attn` |
| `04_bmg_grouped_gemm`, `10_bmg_grouped_gemm_mixed_dtype` | MoE (mixture-of-experts) grouped GEMM; the engine's MoE models use their own kernels |
| `08_bmg_gemm_f8`, `09_bmg_grouped_gemm_f8` | FP8, not used: there is no reachable FP8 matrix multiply on this device ([01-hardware.md](01-hardware.md)) |
| `50_xe35_*`, `51_xe35_*` | Block-scaled MX, for Xe3.5, not Battlemage |
| `12_xe20_moe_gemm_cute_interface` | Xe2 MoE through the CuTe interface |
| `examples/cute/tutorial/xe_gemm_quantization.cpp` | Quantised GEMM tutorial; read it before writing anything |

It has no GDN (gated delta net, the linear-attention layer of the Qwen3.8 family) example, but `vllm-xpu-kernels`
does (below).

It does declare the native int8 x int4 DPAS (the matrix-engine instruction) as CuTe MMA atoms, in
`include/cute/arch/mma_xe.hpp`: `XE_DPAS_TT(d, s8, s4, d)` and every `u8/u4/s8/s4` permutation, compiled out only
for CRI with the comment "Skip int8 x int4 for CRI as the dpas is removed". No mainloop or example uses them:
`include/cutlass/gemm/collective/xe_mma_mixed_input.hpp` still widens int4 to 16 bits before the multiply. So the
instruction is reachable at the atom level, and what is missing is a mainloop, not a driver feature.

That looked for a while like the most attractive opening in this document. It is closed. The mixed builtin's K is
32, the same as int8's, so it runs at 2.000x bf16 rather than 4x, and a W4A8 kernel built on it measured slower
than the bf16 two-pass control once its activation quantiser was counted ([01-hardware.md](01-hardware.md) and
[02-formats.md](02-formats.md)).

### `vllm-xpu-kernels/csrc/xpu/gdn_attn`, the GDN reference

About 6,000 lines of SYCL for exactly the layer `sycl-tla` lacks:

| File | What |
|---|---|
| `gated_delta_rule.hpp` | The recurrent gated delta rule, the decode step. SIMD32, 256-thread groups, 4 value dimensions per subgroup; it takes `ssm_state`, `num_accepted_tokens` (aware of speculative decoding) and `has_initial_state` |
| `causal_conv1d.hpp` | Conv1d, including the speculative-decoding variant with state rollback |
| `xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` | The chunked form for prefill, built on CuTe tiled MMA |
| `xe_2/chunk_causal_conv1d_{,tiled_}xe2.hpp` | Chunked conv1d for prefill |
| `xe_2/l2norm_kernel.hpp` | The q and k L2 norm GDN applies before the recurrence |

Read the recurrent kernel for the math and the index arithmetic; do not lift its torch and ATen plumbing. Two of
the twelve patches in the postmortem ([09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md)) touched these
files, so their failure modes are already known.

## The other checkouts worth having

| Repo | What it is good for |
|------|---------------------|
| `level-zero` | The Level Zero spec and loader: the command-list API this design is built on |
| `compute-runtime` | Intel's GPU runtime (NEO). When Level Zero behaviour surprises you, the answer is here |
| `intel-graphics-compiler` | IGC, what `ocloc` runs. Relevant the moment you read generated assembly |
| `another stack` | A working C++ LLM pipeline. Read its scheduling and KV handling and ignore its model format. It is also the reference for the tokenizer layer: `another tokenizer` for BPE, `google/minja` for chat templates, and `text_streamer.cpp` for incremental, UTF-8-safe detokenisation ([11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md)) |
| `auto-round` | The quantiser behind every checkpoint the engine serves (our own AutoRound int4 g64 re-quantisations), and `auto_round_extension/ark`, the runtime the engine replaced |
| `vllm`, `vllm-xpu-kernels` | The incumbent and the speed baseline. Their XPU kernels are the reference implementation to beat |
| `oneDNN` | The most valuable checkout after `sycl-tla`: Intel's own JIT GEMM generator (`gemmstone`), where another stack's prefill actually runs, and where hardware capability is written down as explicit gates |
| `another stack` | The GPU plugin. `fully_connected_gpu_gemv.cl` is the decode kernel that beats vLLM; read it ([08-decode-vs-prefill.md](08-decode-vs-prefill.md)) |
| `model_server` | another server, OpenAI-compatible serving on top of another stack. A reference for the HTTP and SSE layer, not for the engine |
| `oneAPI-samples` | SYCL and Level Zero idioms with working build recipes; the fastest way to sanity-check a toolchain |

### `oneDNN`: read this one properly

oneDNN is the richest single source for what this silicon actually supports, because it is Intel's own
production kernel generator and it encodes hardware capability as explicit gates rather than as templates.

| Path | Why |
|------|-----|
| `src/gpu/intel/gemm/jit/pd.cpp` | Problem setup, and the s8s4 DPAS gate |
| `src/gpu/intel/gemm/jit/include/gemmstone/` | The JIT GEMM generator: `problem.hpp`, `strategy.hpp`, `kernel_catalog.hpp` |
| `.../gemmstone/kernel_catalog.hpp` | Kernel selection tags, including `ReqXe2Block2D`, the Xe2-specific 2D block loads |
| `src/gpu/intel/matmul/grouped_micro_gemm.{hpp,cpp}` | Micro-GEMM for grouped and small problems, the closest thing to a small-`M` strategy |
| `src/gpu/intel/sdpa/micro.cpp` | Fused attention from micro-kernels, a different approach from `sycl-tla`'s flash attention |

`gemmstone`'s `problem.hpp` gates several quantised paths on `HW::Xe3p` and `ProductFamily::CRI`, features
Battlemage does not have. Reading those gates is the cheapest way to learn what this card cannot do before
discovering it in a benchmark.

vLLM's XPU W4A16 prefill also reaches this readable Apache-2.0 stack rather than an opaque proprietary GEMM:
`XPUwNa16LinearKernel.apply_weights` routes to `torch.ops._xpu_C.int4_gemm_w4a16`, then to the vLLM XPU oneDNN
matmul entry and its primitive cache keyed on M.

We measured that path directly rather than inferring from it. oneDNN's fused int4 W4A16 matmul reaches 98.28
TFLOP/s on gate‖up at M=2048, below the 115.7 TFLOP/s an in-kernel int4 GEMM would need to beat our two-pass path,
and it failed a bitwise identity extraction against our dequant oracle on 39.6 % of weights, every mismatch a
1-ULP bf16 neighbour. The details are in
[08-decode-vs-prefill.md, "How prefill got here"](08-decode-vs-prefill.md#how-prefill-got-here).

## Traps inherited from the vLLM XPU work

These were all hit in production on this hardware, and they are expensive to rediscover.

### Kernels must be capture-safe by construction

Capture means recording a sequence of kernel launches once and replaying it, the way this engine runs every decode
step. The `vllm-xpu-kernels` Xe2 grouped GEMM used a persistent work-stealing scheduler whose global atomic tile
counter was reset from inside the kernel, guarded only by "group 0, lane 0", with no device-wide barrier. Eager
launch timing happened to separate the reset from the steals; SYCL-graph capture and replay does not. The counter
carried the previous replay's value, producing out-of-bounds tile coordinates and
`UR_RESULT_ERROR_DEVICE_LOST` at batch > 1.

The fix was 14 lines: drop the atomic work-stealing for a deterministic grid-stride loop with identical tile
coverage. Design every kernel this way from the start.

### `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` under capture is a real failure

ARK's int8 `woqgemm` runs correctly in eager mode at exactly the shapes needed and fails under XPU graph capture.
A kernel that works eagerly proves nothing about whether it can be replayed. Test capture from day one, not as a
late optimisation.

### Torch's inductor had an XPU-only pessimisation

torch 2.14 defaults `pre_grad_fusion_options` to
`{'batch_linear_lhs': {'devices': ('xpu',), 'min_fuse_set_size': 2}}`, which applies to XPU only and cost about
30 % of decode on this workload. It is not relevant once torch is gone, but it is a reminder that vendor defaults
are tuned for someone else's shapes.

### Non-contiguous tensors reach kernels that assume contiguity

torch 2.14 and later return a non-contiguous `in_proj_qkvz`; the GDN SYCL kernel indexes raw pointers and asserts.
A loader written from scratch controls layout end to end and should never be in this position. Still, GDN kernels
index raw pointers, so layout invariants must be explicit and asserted.

### Facts about the incumbent MoE path

These come from reading the kernel source.

- `XpuFusedMoe` is not fused. Per MoE layer per step it issues `remap_hidden_states`, a grouped GEMM,
  `silu_and_mul`, a second grouped GEMM and `moe_gather`: 4 kernel launches and 5 allocations. At 40 layers that
  is 160 launches and 200 allocations per decode step.
- Every work-group scans all experts, reading `rows_per_expert[i]` from global memory, even when only 8 are
  non-empty.
- The path that actually runs is W4A16 (bf16 activations), not W4A4.
- With one row (M=1) the dispatcher picks a work-group tile of `Shape<_8, _64, _32>` with 64 threads and no
  split-K anywhere. At `N = 4096` that is 64 work-groups for 32 subslices and 256 EUs, which under-fills the
  device. That applies to the dense models too, independently of host overhead.
- XPU is the only MXFP4 backend that does no weight or scale repacking; every CUDA and ROCm backend pre-swizzles.

## What not to borrow

Do not fork `vllm-xpu-kernels`. It was considered and rejected: it inherits the capture-safety defects above,
contradicts the goal of owning the stack, and gives up most of the host-side win that is the point of the project.
Read it and benchmark against it; do not depend on it.
