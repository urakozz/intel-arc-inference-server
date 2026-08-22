# Prior art: what to borrow, what to avoid

## `sycl-tla` - the kernel templates

`~/PycharmProjects/sycl-tla`. A fork of NVIDIA CUTLASS extending CUTLASS and CuTe
to Intel GPUs via SYCL. Header-only C++ templates. **BMG = Battlemage = the B70**,
so every `*_bmg_*` example targets this silicon directly.

| Example | Use |
|---------|-----|
| `00_bmg_gemm` | Baseline bf16 GEMM - start here to validate the toolchain |
| `02_bmg_gemm_mixed_dtype` | **int4/int8 weights × bf16 activations** - the phase-1 prefill workhorse |
| `03_bmg_gemm_streamk` | Stream-K - the split-K relative. Read for how `sycl-tla` fills the device when `M × N` tiles are too few; the same problem the decode GEMV solves with split-K |
| `05_bmg_gemm_with_epilogues` | Fusing SiLU / bias / residual into the GEMM |
| `06_bmg_flash_attention` | The 8 full-attention layers |
| `04_bmg_grouped_gemm`, `10_bmg_grouped_gemm_mixed_dtype` | MoE, phase 3 |
| `08_bmg_gemm_f8`, `09_bmg_grouped_gemm_f8` | FP8 - not used, see doc 02 |
| `50_xe35_*`, `51_xe35_*` | Block-scaled MX - **Xe3.5, not Battlemage** |
| `12_xe20_moe_gemm_cute_interface` | Xe2 MoE via the CuTe interface - phase 3 reading |
| `examples/cute/tutorial/xe_gemm_quantization.cpp` | Quantised GEMM tutorial - read before writing anything |

It has **no GDN / linear-attention example** - but `vllm-xpu-kernels` does (below).

It **does** declare the native int8 × int4 DPAS as CuTe MMA atoms:
`include/cute/arch/mma_xe.hpp:287-297`, `XE_DPAS_TT(d, s8, s4, d)` and
permutations, compiled out only for CRI. No mainloop consumes them -
`include/cutlass/gemm/collective/xe_mma_mixed_input.hpp` widens int4 to 16-bit
before the MMA. A W4A8 mainloop on those atoms is the phase-1 prefill
opportunity (doc 07 #9).

### `vllm-xpu-kernels/csrc/xpu/gdn_attn` - the GDN reference

~6 000 lines of SYCL for exactly the layer `sycl-tla` lacks:

| File | What |
|---|---|
| `gated_delta_rule.hpp` (833) | **Recurrent** gated delta rule - the decode step. `gated_delta_rule_kernel`: SIMD32, 256-thread groups, 4 value-dims per subgroup, takes `ssm_state`, `num_accepted_tokens` (spec-decode aware), `has_initial_state` |
| `causal_conv1d.hpp` (1146) | Conv1d incl. the spec-decode variant with state rollback (comment at :497 says it mirrors Triton's `fused_recurrent` rollback) |
| `xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` (1634) | **Chunked** form for prefill, built on CuTe tiled MMA (`chunk_gemm_policy_64x64x32_*`) |
| `xe_2/chunk_causal_conv1d_{,tiled_}xe2.hpp` | Chunked conv1d for prefill |
| `xe_2/l2norm_kernel.hpp` | The q/k L2 norm GDN applies before the recurrence |

Read the recurrent kernel for the *math* and the index arithmetic; do not lift
its torch/ATen plumbing. Two of the twelve postmortem patches (doc 09:
`02-pr476-gdn-conv1d`, `03-pr477-gdn-decode`) touched these files, so their
failure modes are already known.

## The other checkouts

| Repo | What it is good for |
|------|---------------------|
| `level-zero` | The L0 spec and loader - the command-list API this design lives on |
| `compute-runtime` | Intel NEO. When L0 behaviour surprises you, the answer is here |
| `intel-graphics-compiler` | IGC, what `ocloc` runs. Relevant if you go to offline SPIR-V |
| `openvino.genai` | A working C++ LLM pipeline. Read its scheduling and KV handling; ignore its model format. Also the reference for the **tokenizer layer**: `openvino_tokenizers` for BPE, `google/minja` (fetched in `src/cpp/CMakeLists.txt:112-119`) for chat templates (`tokenizer_impl.cpp:792`), and `text_streamer.cpp:7-47` for incremental UTF-8-safe detokenisation (doc 11) |
| `nncf` | Quantisation algorithms, comparison point for AutoRound |
| `auto-round` | The quantiser we keep, and `auto_round_extension/ark` - the runtime we are replacing |
| `vllm`, `vllm-xpu-kernels` | The incumbent and the baseline. Their XPU kernels are the reference implementation to beat |
| **`oneDNN`** | **The most valuable checkout after `sycl-tla`.** Intel's own JIT GEMM generator (`gemmstone`) - where OpenVINO's prefill actually runs, and where the native mixed s8×s4 DPAS is documented |
| `openvino` | The GPU plugin: `fully_connected_gpu_gemv.cl` is the decode kernel that beats vLLM. Read it (doc 08) |
| `model_server` | OpenVINO Model Server - OpenAI-compatible serving on top of OpenVINO. Reference for the HTTP/SSE layer, not the engine. Uses genai's `Tokenizer` + minja (`src/llm/servable.hpp:78,190`) and subclasses the text streamer (`ovms_text_streamer.hpp`) |
| `oneAPI-samples` | SYCL/L0 idioms and working build recipes. Fastest way to sanity-check the toolchain |
| `oneCCL`, `ucx`, `nixl` | Multi-GPU and disaggregated serving. **Not phase 1-4.** Relevant only if TP=2 or P/D disaggregation is ever attempted across the two B70s |

### `oneDNN` - read this one properly

The single richest source for "what does this silicon actually support", because
it is Intel's own production kernel generator and it encodes hardware capability
as explicit gates rather than templates.

| Path | Why |
|------|-----|
| `src/gpu/intel/gemm/jit/pd.cpp` | Problem setup, and the s8s4 DPAS gate (line ~772) |
| `src/gpu/intel/gemm/jit/include/gemmstone/` | The JIT GEMM generator: `problem.hpp`, `strategy.hpp`, `kernel_catalog.hpp` |
| `src/gpu/intel/gemm/jit/include/gemmstone/kernel_catalog.hpp` | Kernel selection tags - including `ReqXe2Block2D`, i.e. Xe2-specific 2D block loads |
| `src/gpu/intel/matmul/grouped_micro_gemm.{hpp,cpp}` | Micro-GEMM for grouped/small problems - closest thing to a small-`M` strategy |
| `src/gpu/intel/sdpa/micro.cpp` | Fused attention via micro-kernels - a different approach to `sycl-tla`'s flash attention |

`gemmstone`'s `problem.hpp` also gates several quantised paths on `HW::Xe3p` and
`ProductFamily::CRI` - features Battlemage does **not** have. Reading those gates
is the cheapest way to learn what this card cannot do, before discovering it in a
benchmark.

## Traps inherited from the vLLM XPU work

Hard-won, expensive to rediscover. All of these were hit in production on this
exact box.

### Kernels must be capture-safe by construction

The `vllm-xpu-kernels` Xe2 grouped GEMM used a persistent work-steal scheduler
whose **global atomic tile counter was reset from inside the kernel**, guarded
only by "group 0, lane 0", with no device-wide barrier. Eager launch timing
accidentally separated the reset from the steals; **SYCL-graph capture/replay does
not**, so the counter carried the previous replay's value → out-of-bounds tile
coordinates → `UR_RESULT_ERROR_DEVICE_LOST` at batch > 1.

The fix was 14 lines: drop the atomic work-steal for a deterministic grid-stride
loop with identical tile coverage. **Design every kernel this way from the start.**

### `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` under capture is a real failure mode

ARK's int8 `woqgemm` runs correctly in eager mode at the exact shapes needed, and
fails under XPU graph capture. A kernel working eagerly proves nothing about
whether it can be replayed. **Test capture from day one**, not as a late
optimisation.

### Torch's inductor had an XPU-only pessimisation

torch 2.14 defaults `pre_grad_fusion_options` to
`{'batch_linear_lhs': {'devices': ('xpu',), 'min_fuse_set_size': 2}}` - XPU-only,
and it cost ~30% of decode on this workload. Not directly relevant once torch is
gone, but a reminder that vendor defaults are tuned for someone else's shapes.

### XPU graphs have limits worth knowing before depending on them

Single-GPU only, and MXFP8 is unsupported under them. Both bite when TP=2 arrives.
Since this project owns its own command lists, these are constraints to *not*
inherit - but they indicate where the driver's graph support is thin.

### Non-contiguous tensors reach kernels that assume contiguity

torch 2.14+ returns a non-contiguous `in_proj_qkvz`; the GDN SYCL kernel indexes
raw pointers and asserts. A from-scratch loader controls layout end to end and
should never be in this position - but it is a reminder that **GDN kernels index
raw pointers**, so layout invariants must be explicit and asserted.

### Measured facts about the incumbent MoE path (phase 3 reading)

From reading the v0.1.12 kernel source - do not re-derive:

- `XpuFusedMoe` is **not fused**: 4 kernel launches + 5 allocations per layer per
  step; 160 launches per decode step at 40 layers.
- The path that actually runs is **W4A16** (bf16 activations), not W4A4.
- Every work-group scans **all** experts even when only 8 are non-empty.
- At M=1 the dispatcher picks `w4a16_policy_m_8` - work-group tile
  `Shape<_8, _64, _32>` (`csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp:89-93`,
  selected at `grouped_gemm_xe2_interface.hpp:355`), 64 threads/WG, and there is
  **no split-K anywhere**. At `N = 4096` that is 64 work-groups for 32
  subslices / 256 EUs. Leading hypothesis for the ~63% bandwidth utilisation is
  that gemm1 under-fills the device - and this applies to the dense models
  too, independent of host overhead.
- XPU is the **only** MXFP4 backend doing zero weight/scale repacking; every
  CUDA/ROCm backend pre-swizzles.

## What not to borrow

**Do not fork `vllm-xpu-kernels`.** It was considered and rejected: it inherits
the capture-safety defects above, contradicts the goal of owning the stack, and
forfeits most of the host-side win that is the entire point. Read it, benchmark
against it, do not depend on it.
