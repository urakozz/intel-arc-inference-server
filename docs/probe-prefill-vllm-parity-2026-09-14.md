# Probe pre-registration - can Qwen3.8-27B prefill reach vLLM's 1973 t/s?

grade: source-reading (Mac only; no box measurement in this document)

Brief: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/prefill-vllm-parity-brief.md`.
This is **Phase 1** of that brief: pull the comparison repos, read the two
never-measured levers' entry points, check sycl-tla for drift since our pin,
map vLLM's own prefill code, and pre-register predictions + decision rules
before any Phase-2 measurement. Rulings A20-A30 (`interfaces.md`) were read
first; nothing below re-tries a lever already measured dead there.

---

## 0. Repo pulls - old → new HEAD

All in `~/PycharmProjects/{repo}` (Mac), never `~/PycharmProjects/vllm` (git
pull skipped there - it is the operator's patched branch; read-only source
reads of it were still used for L4, see §4).

| repo | old HEAD | new HEAD | changed? |
|---|---|---|---|
| sycl-tla | `91e5bd735517d8e79591b41e0d0cd37a7bacdca7` | `91e5bd735517d8e79591b41e0d0cd37a7bacdca7` | **no** - `git fetch --all` confirms origin/main is the same commit (2026-08-27), zero new commits upstream |
| oneDNN | `3f27bec21fdce6b550f233bf6e0afd086ee8d543` | `b3225e00f40a2702939675d2c5ebcab875193d31` | yes |
| vllm-xpu-kernels | `f8318f01101d7fab4551c6ae2f4279f8f4ae1877` | `f8318f01101d7fab4551c6ae2f4279f8f4ae1877` | no - already at latest (commit dated 2026-09-14, i.e. today) |
| level-zero | `5c863340cab6631a31234653191b904f0028b93b` | `5c863340cab6631a31234653191b904f0028b93b` | no |
| compute-runtime | `e37c60bf2ad415d6f60af8b89b849a30a67d8ac1` | `47a0f69f967e77d0359a46197ac29fd164b571ca` | yes |
| intel-graphics-compiler | `98fff71643f1d26ba6cc25811e46c780bb022158` | `a13c18f466f03a9d13eeba3c239ecb50f87f567f` | yes |
| openvino | `7c4d2733dc0521f0f1203f3664df305905a80095` | `a5b6e1a02a6a9b387cba17db56d3d191d7729f4e` | yes |
| openvino.genai | `da1db864d2bc1a265e9a036f5b57ef13f0e86c2c` | `d8b52480be6118d2f3952a4df39da79b6235cd63` | yes |

sycl-tla and vllm-xpu-kernels - the two repos L1/L2/L3 actually read code
from - needed no pull. compute-runtime / IGC / openvino / openvino.genai
moved but are not read further here (out of this brief's scope).

---

## 1. L1 - oneDNN's int4 W4A16 matmul (vLLM's own prefill GEMM path)

### 1.1 Call chain (confirmed by reading, not guessing)

```
vLLM (XPUwNa16LinearKernel.apply_weights, vllm/model_executor/kernels/linear/mixed_precision/xpu.py:88-105)
  -> torch.ops._xpu_C.int4_gemm_w4a16
       (vllm-xpu-kernels csrc/xpu/onednn/onednn_matmul.cpp:257-276)
  -> oneDNN::dnnl_matmul_w4a16_int4  (csrc/xpu/onednn/int4_gemm_w4a16.h:12-161)
  -> dnnl matmul primitive (attribute-driven weights decompression, engine/stream
     built via dnnl::sycl_interop over the existing SYCL device+context)
  -> gemmstone JIT  (oneDNN src/gpu/intel/gemm/jit/include/gemmstone/, confirmed
     present at both the old and new oneDNN HEAD; last touched 2026-09-09)
```

`XPUwNa16LinearKernel` is the kernel vLLM actually selects for GPTQ-style
int4 weights on XPU (`weight_type in {uint4, uint4b8}`, `act_type` bf16/fp16)
- this is not a side path, it is the linear layer's production dispatch.

### 1.2 Primitive attributes, exactly (from `int4_gemm_w4a16.h`)

- `joint_dtypes_t::bf16_int4` for our activation dtype (bf16 src) - the
  weight decompress-and-multiply is fused into the GEMM mainloop; **there is
  no separate bf16 dequant-scratch pass**, confirming the brief's premise.
- Weight scale: `set_scales(DNNL_ARG_WEIGHTS, mask=(1<<0)|(1<<1), {group_size,1}, dtype)`
  - grouped along K with the group size supplied at call time (not
  compiled-in). oneDNN's own docs (`doc/functional_api/primitives/matmul.md:196-197`)
  require K-grouped weight scales to be a multiple of **16**; vLLM's own
  kernel gate (`xpu.py:39-42`) is stricter - **group_size must be a multiple
  of 32**, and the per-partition K dimension must also be a multiple of 32.
  Our checkpoint is GPTQ **g64** (A28 addendum) - 64 satisfies both.
- Zero points: **symmetric** case (`zp.dim()==1`) is a scalar `s8` constant
  broadcast via `mask=0` - and both `vllm_xpu_kernels/quantization/_quantize_convert.py:216-222`
  (`transpose_onednn_woq_format`) and vLLM's own `xpu.py:83-87`
  (`XPUwNa16LinearKernel.process_weights_after_loading`) set that constant
  to **8** for the symmetric path - the identical "GPTQ v1 unsigned nibbles,
  zero = 8" convention A21/A23 already used. Asymmetric case is `u4`,
  K-grouped at the same `group_size`. Our checkpoint is GPTQ g64 **sym**
  (A28 addendum), so the scalar-8 path applies - no zero tensor at all.
- `fpmath_mode` set to `f16`/`bf16` "true" (down-conversion allowed) -
  standard W4A16 attribute, not a numerics concern by itself.

### 1.3 Weight layout, repack, and residency - the key finding

**oneDNN needs the weights K-contiguous per N ("MK-major"), not the
N-contiguous layout our loader's layout-0 uses - and this is the identical
physical transpose A23 already measured and priced.**

Two independent Python code paths agree the incoming GPTQ tensor (standard
`qweight [K/8][N]` u32, N-contiguous - the same starting shape as our
layout-0) must be relaid out before oneDNN can use it:

- `_quantize_convert.py:211-214`: `layer.qweight.transpose(0,1).contiguous().transpose(0,1)`
  - forces stride order `(1, K/8)`, i.e. for fixed `n` the `K/8` packed
  words become contiguous.
- `xpu.py:69-74` (`XPUwNa16LinearKernel.process_weights_after_loading`):
  does the same relayout (checked by shape comparison between qweight and
  scale, then `.t().contiguous()`) before the op is ever called.

The C++ side confirms the *consumed* stride: `ldb = mat2.strides()[dim-1] * 8`
in `int4_gemm_w4a16.h:57` only produces `ldb == K` if the last-dim stride is
already `K/8` - i.e. oneDNN's matmul wants exactly the K-contiguous-per-N
byte order.

**This is byte-for-byte the same relayout A23 measured for sycl-tla's
`gemm_int4_mixed` mainloop** (`docs/probe-gemm-int4-mixed-2026-09-05.md`
§1): layout 0's 4-bit address `p0(k,n) = ⌊k/8⌋·8N + 8n + (k mod 8)` is
piecewise in `k` and unusable verbatim (A23 measured 90.92% wrong words fed
raw); the fix there was "exactly a u32 transpose of `qweight`"
(`out[n·K/8+r] = qweight[r·N+n] [^ 0x88888888 for signed int4b_t]`), costing
**+12,163,481,600 B (11.33 GiB) for the 256 layer matrices, ~6.4 s one-time
host repack**. oneDNN's requirement is the same transpose direction; the
only difference is oneDNN carries the zero point as an attribute (constant
8) instead of a `^8` bit-flip on the weight, so no XOR and no separate zero
tensor for the symmetric case - the byte *count* of the repack is the same
size class as A23's number.

**Residency verdict: L1's second-copy cost is not a new unknown - it
inherits A23's already-measured price, not a fresh estimate.** Against the
last stated headroom at max_len 16384 (13.79 GiB, A20/A21's context, same
checkpoint byte class per A28's "same byte class as Vishva, 15.54 GB,
bf16 head") that would leave roughly the same ~1.6-2.5 GiB margin A23
already flagged as tight. **Open for Phase 2**: recompute the exact current
headroom for the checkpoint actually loaded today (not re-derived here,
since it depends on live allocator state this brief did not touch) before
committing to residency; if it does not fit, the brief's fallback (price
against a relayout, i.e. per-chunk repack instead of a resident second copy)
is the one A23 already priced too (~84 ms/chunk, its §1 "predicted
alternative").

### 1.4 SYCL queue / USM zero-copy - answered, split by tensor role

vllm-xpu-kernels' `onednn_runtime.h` builds its `dnnl::engine` via
`dnnl::sycl_interop::make_engine(raw_sycl_device, sycl_context)`, its
`dnnl::stream` via `dnnl::sycl_interop::make_stream(engine, sycl_queue)`, and
every tensor argument via `dnnl::sycl_interop::make_memory(md, engine,
memory_kind::usm, ptr)` - this is exactly the externally-owned-queue, USM
integration pattern oneDNN's SYCL interop exists for. Our
`runtime::prefill::Context::sycl_queue_raw()` / `sycl_context_raw()`
(`src/runtime/prefill/context.h:68-69`, A1-A3) expose precisely the SYCL
queue/context handles this interop expects.

- **Activations (src) and output (dst): yes, zero-copy is architecturally
  available** - wrap our existing USM pointers with `memory_kind::usm`, no
  new engine/stream needed beyond one `sycl_interop::make_engine/make_stream`
  call over our own queue.
- **Weights: no** - per §1.3, the weight buffer needs the K-contiguous
  repack; the existing layout-0 buffer cannot be reinterpreted with
  different strides into the shape oneDNN requires (same conclusion A23
  reached for sycl-tla's mainloop: "no rank-3 flat stride can express" the
  layout-0 addressing).
- **Scales: pass through unchanged** - same convention as A21/A23 (`S3`):
  GPTQ's `scales f16 [K/64][N]` is exactly the grouped-scale tensor shape
  oneDNN's attribute API expects.

### 1.5 Pricing prediction - pre-registered, with reasoning

Brief's arithmetic, restated: L1 replaces dequant (205 ms) + bf16 GEMM
(656 ms) = 861 ms/chunk. At rate `R` TFLOP/s over ~98 TFLOP/chunk, the int4
matmul costs `98/R` s. **Break-even: R ≈ 114 TFLOP/s. Full 418 ms gap
closure alone: R ≈ 221 TFLOP/s** (above the 183.5 TFLOP/s XMX bf16 peak -
i.e. L1 alone cannot ever close the whole gap, only the brief's own upper
bound admits this).

**Prediction: R will land below break-even (R < 114 TFLOP/s), medium-high
confidence.** Reasoning:

1. A23 already measured a structurally similar Intel-stack int4 W4A16 path
   (sycl-tla's own hand-written `gemm_int4_mixed` mainloop, same hardware,
   same checkpoint format, same group size) at a **flat 40.9-45.1 TFLOP/s
   across all six production shapes** - 23% of the bf16 XMX peak - with the
   signature of a **fixed per-k-tile decompression cost** (transposing
   4-bit block loads + in-register unpack + a scale reload every
   `group_size/BLK_K` k-tiles), not a bandwidth or shape effect. gemmstone
   targets the same XMX/DPAS ISA with the same fundamental need to unpack
   4-bit weights and apply a per-group scale every k-tile; that physical
   cost is a property of int4 W4A16 decode on this hardware generation, not
   of which team wrote the mainloop.
2. Weak corroborating signal: vllm-xpu-kernels' own performance benchmark
   suite (`benchmark/benchmark_gemm_onednn.py`) wires up `bf16`, `fp8`,
   `fp8_w8a16`, `fp8_per_channel`, `mxfp8`, `mxfp4` but has **no int4/W4A16
   benchmark at all** in `ALL_BENCHMARKS`, despite `int4_gemm_w4a16` being a
   registered, production op with its own correctness path elsewhere in the
   tree. This does not prove a rate, but it is consistent with this path not
   having been a throughput priority for its own authors.
3. Competing hypothesis, also pre-registered: gemmstone is genuinely
   different code (a separate JIT, possibly different tiling/prefetch), and
   oneDNN is Intel's own production library with first-party access to
   ISA/microarchitecture documentation neither this project nor sycl-tla's
   public example set necessarily has, so there is real chance it clears
   sycl-tla's 42 TFLOP/s ceiling - that is exactly why L1 must still be
   measured rather than declared dead by analogy alone.
4. If R lands below break-even, that also means **vLLM's real advantage is
   not the GEMM term** - pointing at L2/L4 as where the 1038 ms actually
   comes from (see §4).

### 1.6 Decision rules to pre-register for Phase 2

- **Correctness**: bit-exact against `tools/oracle/dequant.py` on the real
  layout-0 weights after the required repack (A23's method exactly -
  `A = I` identity-extraction GEMM reproducing the oracle's bf16 word per
  weight), on at least one full production matrix. Any mismatch beyond the
  IEEE `+0/-0` class A23 already characterized (D2, 6.642% of weights,
  `q=8` with a negative scale) fails the probe outright - no tuning.
- **Rate**: TFLOP/s at gate‖up M ∈ {1024, 2048, 4096} and the other
  production shapes, oneDNN primitive over the repacked real weights.
  `R < 114` → lever dead, report plainly (mirrors A23's own branch rule).
  `114 ≤ R < 221` → partial win, must combine with a GDN saving to close the
  rest of the gap. `R ≥ 221` → L1 alone closes the gap (treated as the
  unlikely branch per §1.5).
- **Residency**: if the second copy does not fit within the checkpoint's
  actual current headroom at max_len 16384 (to be recomputed against live
  numbers in Phase 2, see §1.3), the probe reports the per-chunk-repack
  fallback price (~84 ms/chunk, A23's own derived number for the same
  transpose) instead of assuming resident bytes are free.

---

## 2. L2 - Intel's chunked GDN kernel for Xe2

### 2.1 Torch dependency status - still present, but shallow

At the current HEAD (`f8318f01…`, 2026-09-14 - today, i.e. this IS the
revision Stage-0's P5 blocker was reported against was already stale by),
`csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.h` and `.cpp` still
`#include <torch/all.h>` unconditionally and take `torch::Tensor&`
throughout the public signature - **P5's blocker
(`docs/probe-gdn-chunk-2026-09-04.md`, `fatal error: 'torch/all.h' file not
found`) is not resolved upstream.**

But the dependency is now characterized, not just present: grepping
`csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` (1634 lines)
for `torch::` finds all **18 references confined to the outer wrapper**
`chunk_gated_delta_rule_impl_xe2` (lines 1506-1633), used for exactly three
things:

1. `TORCH_CHECK` dtype/shape validation (lines 1541-1553).
2. Three `torch::zeros(...)` scratch-buffer allocations - `A`
   `[num_v_heads, total_seqlen+padding, chunk_size]`, `w`
   `[num_v_heads, total_seqlen+padding, head_k_dim]`, `u`
   `[num_v_heads, total_seqlen+padding, head_v_dim]` (lines 1560-1568).
3. `.data_ptr()` extraction into `reinterpret_cast`s that feed a pure-SYCL
   `kernel_launcher<scalar_t, state_scalar_t>` template call
   (lines 1570-1629).

**Everything above line 1504 - the actual kernel bodies - is raw
pointers/SYCL only; no ATen op executes inside the compute path.** The
public entry header (`chunk_gated_delta_rule_xe2.h`) is likewise a thin
41-line signature-only wrapper.

**Verdict: a torch-free shim is plausible and small.** It needs to replace
(1) the three `torch::zeros` calls with raw USM/L0 device allocations of the
same byte sizes, (2) the `torch::Tensor&` parameters with a small POD
view (ptr + shape + stride + dtype), and (3) `TORCH_CHECK` with our own
asserts - an estimated ~130-line wrapper rewrite, with the 1500-line kernel
body untouched.

### 2.2 Confirmed: this IS vLLM's real prefill dispatch, not a fallback

`gdn_attn_interface.cpp:690-696` (`gated_delta_rule_non_spec`):

```cpp
#ifdef VLLM_XPU_ENABLE_XE2
  if (num_prefills > 0) {
    chunk_gated_delta_rule_xe2(...);
```

- guarded only by the `VLLM_XPU_ENABLE_XE2` build flag, dispatched whenever
`num_prefills > 0`. The prefill-side `causal_conv1d` stage takes the
analogous Xe2-chunked path (`chunk_causal_conv1d_xe2` /
`chunk_causal_conv1d_tiled_xe2`, same file, lines 429-508).

### 2.3 Numerics and time predictions - pre-registered

**Numerics**: do not assume A22/A27's band (state max rel 3.506e-02, mean
1.197e-03) transfers - Intel's chunking/tiling and reduction order differ
from ours, and A25/A26 already established the precedent that a
reduction-order change re-measures its own band and uses the **token gate**
(golden 94/94, RTN 93/93, determinism) as arbiter, not a carried-over
tolerance number.

**Time**: lower confidence than L1's prediction, registered as a band
rather than a point estimate - **150-400 ms/chunk against our current
~303 ms/chunk GDN total**, i.e. genuinely could go either way. Reasoning:
Intel's kernel is vendor-authored production code for exactly this hardware
generation, algorithm, and chunk size, which argues for competitive-or-better
raw compute; but it is general-purpose (must also serve decode, spec/MTP
batches, arbitrary layouts) where our kernel went through five narrowly-
targeted tuning passes against our exact production shapes (A25 scan
rewrite 4.53×, A27 wu/conv, A29 A2/A output tiles, A30 staging backport,
plus one reverted DPAS attempt that hit its time bar but failed the gate)
- generality usually costs some of that specialization back.

### 2.4 Decision rules to pre-register for Phase 2

- Build the torch-free shim; if it turns out not small once attempted
  (i.e. torch reaches deeper than the 18-reference wrapper suggests), report
  that as the finding and stop - do not vendor a torch runtime onto the box.
- Numerics: token gate is mandatory pass/fail; band is recorded, not held to
  A22's number.
- Time: measured ms/layer/chunk at our production shapes, compared to the
  ~303 ms/chunk baseline; report the delta whichever direction it goes.

---

## 3. L3 - sycl-tla since our pin

**Zero commits upstream since `91e5bd735517d8e79591b41e0d0cd37a7bacdca7`**
(2026-08-27) - confirmed both by local `git pull` ("Already up to date")
and `git fetch --all` against `origin/main` showing the same commit hash and
date. Per the brief's own rule ("re-measure a cell only if relevant code
changed"), **no diff of `include/cutlass/gemm` or `applications/` is
warranted and no cell is re-measured** - there is nothing to diff.

---

## 4. L4 - where vLLM's own 1038 ms/chunk goes (read-only)

Read from `~/PycharmProjects/vllm` (not pulled, not modified - HEAD
`ff7f9dc8bfaacf187b576974f5de5fdac212a22f`, 2026-09-14) and
`~/PycharmProjects/vllm-xpu-kernels`.

Every GDN layer's XPU forward (`QwenGatedDeltaNetAttention.forward_xpu`,
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:976-1023`):

| stage | code | maps to |
|---|---|---|
| 1. `in_proj_qkvz` + `in_proj_ba` (2 GEMMs) | `XPUwNa16LinearKernel.apply_weights` → `torch.ops._xpu_C.int4_gemm_w4a16` | **L1** - the same oneDNN int4 path §1 prices |
| 2. `torch.ops._xpu_C.gdn_attention` (1 fused op) | `gdn_attn_interface.cpp`: `causal_conv1d` (Xe2-chunked when `num_prefills>0`) then `chunk_gated_delta_rule_xe2` when `num_prefills>0` | **L2** - conv + scan/wu/wy, one op call from Python's view |
| 3. `self.norm` (`RMSNormGated`) | `vllm.model_executor.layers.layernorm.RMSNormGated` | analogous to our l2norm/gate small-kernel term |
| 4. `out_proj` (1 GEMM) | same `XPUwNa16LinearKernel` path as (1) | **L1** again |

Full-attention layers (interleaved with GDN layers in Qwen3-Next,
`Qwen3NextAttention`, `vllm/model_executor/models/qwen3_next.py:272-455`)
use `self.attn = Attention(...)` - vLLM's generic attention module, whose
XPU backend is a separate flash-attention kernel (vllm-xpu-kernels ships
`benchmark/benchmark_cutlass_flash_attn_varlen.py` for exactly this) - QKV/O
projections on these layers go through the same `XPUwNa16LinearKernel` int4
path as the GDN layers' linears.

**Correction to spec 2 §1's premise.** Spec 2 §1 cites Triton
`fla_chunk_gated_delta_rule` at `FLA_CHUNK_SIZE 64` as the GDN reference
kernel. That Triton path exists in `qwen_gdn_linear_attn.py`
(`forward_native`/`forward_cuda`, lines ~189-352,
`fla_chunk_gated_delta_rule` / `fi_chunk_gated_delta_rule` /
`chunk_gated_delta_rule_cutedsl`) but **XPU never takes it** - `forward_xpu`
unconditionally calls the native SYCL `torch.ops._xpu_C.gdn_attention` op
(§2.2's `chunk_gated_delta_rule_xe2` for prefill). So L2's target
(`chunk_gated_delta_rule_xe2`) is confirmed as the correct and only
comparison point on this hardware; spec 2 §1's Triton/`FLA_CHUNK_SIZE 64`
citation describes vLLM's CUDA/ROCm path, not the one actually running when
vLLM measured 1973 t/s on this box.

**Net read**: vLLM's 1038 ms/chunk is spent across the *same five
categories* our engine's ledger already tracks (GEMM ×3 call sites, conv,
GDN scan/recurrence, norm, attention) - nothing in vLLM's own code performs
a fundamentally different algorithm. Whether vLLM wins is therefore entirely
a question of **which of L1 (GEMM) and L2 (GDN) individually beats our
measured per-term numbers**, which is exactly what Phase 2 measures.

---

## 5. Box check - oneDNN install (read-only, the only box contact in Phase 1)

```
$ ssh user@box "ls -d /opt/intel/oneapi/dnnl*; ..."
/opt/intel/oneapi/dnnl
```

Version directory present: **`/opt/intel/oneapi/dnnl/2026.0`** (confirmed by
`include/dnnl_version.h` and `include/oneapi/dnnl/dnnl_version.h` existing
under that path). No build, no run, no write - a single `ls`/`find`
one-liner. Nothing else was touched on the box.

---

## 6. Probe designs for Phase 2 (not run here)

Following `tools/probe/probe_gemm_int4_mixed`'s pattern (A23):

- **L1 probe**: new target under `tools/probe/`, oneDNN `int4_gemm_w4a16`
  over the real repacked layout-0 weights. Correctness: `A=I`
  identity-extraction vs `tools/oracle/dequant.py`, one full production
  matrix. Rate: TFLOP/s at gate‖up M ∈ {1024, 2048, 4096} + other production
  shapes, same replay/median protocol A23 used (8 replays, drop first 3,
  median of last 5). Residency: recompute current headroom at max_len 16384
  against the checkpoint actually loaded, before assuming the second copy
  fits.
- **L2 probe**: build the torch-free shim characterized in §2.1 over
  `chunk_gated_delta_rule_xe2`; numerics against our `gdn_chunk` (fresh band,
  token gate arbiter per A25/A26); time per chunk at our production shapes.
- Both probes: pre-registration commit precedes any measurement commit, one
  defect/one fix/one measurement, build via `tools/box.sh` only
  (`ZE_AFFINITY_MASK=1`), probe builds in a separate build directory, any
  newer sycl-tla staged under `~/sycl-tla-latest` - never `~/sycl-tla`.

---

## 7. Blockers

None found that stop Phase 2 from starting once the box is free. Open items
carried forward, not blocking:

- L1's residency check needs the checkpoint's live current headroom
  (§1.3/1.6) - not computable from source reading alone.
- L2's torch-free shim is characterized as small but unbuilt - its actual
  size is a Phase-2 finding, not assumed here.
