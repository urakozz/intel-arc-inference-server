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

---

## Addendum - re-scope and Phase 2 pre-registration (2026-09-14)

grade: source-reading + pre-registration (Mac only; no box contact; no code,
no build). Everything below is labelled **measured** (cited from a prior
record), **derived** (arithmetic on measured inputs, shown), or **estimated**
(a stated guess with its reasoning). Nothing in this addendum is a new
measurement, and nothing above it is deleted or altered - §1.5's prediction is
quoted and marked superseded in place, here.

### A0. The re-scope

Operator ruling, 2026-09-14, quoted from the brief verbatim: **"if we can throw
away something to get more low level and more performant - I'm hundred percent
up for it."** This reverses spec 2's inherit-first ruling. The direction under
test is now **dropping sycl-tla from the prefill path** in favour of our own
OpenCL C DPAS GEMM on the same Level Zero immediate list every other prefill
kernel already runs on (A22's in-order list). Consequences for §1-§4 above:
L1 (oneDNN int4) and L2 (Intel's Xe2 GDN) are **measured as reference targets
only - neither is a candidate for adoption**; the candidate is P-A + P-B. §1.3's
residency work and §1.6's candidate-style decision rules for L1 are retired as
decisions (the measurements they describe are kept, as P-C's protocol). This
brief adopts nothing into the engine; a clean "no, and here is the measured
reason" is a full deliverable.

### A1. The superseded oneDNN prediction

§1.5 above reads: **"Prediction: R will land below break-even (R < 114
TFLOP/s), medium-high confidence."** - **SUPERSEDED** by this section. It was
reasoned from sycl-tla's `gemm_int4_mixed` at 42 TFLOP/s (A23). sycl-tla and
oneDNN's gemmstone are different kernel generators, and there is stronger
evidence available: vLLM runs this exact oneDNN path (§1.1, §4) at **1973 t/s
end to end on this box**. The replacement is reasoned from that measurement.

**The linear work per chunk (derived, from `model::Qwen35`'s shapes; this is
P2's own per-row sum, restated so the chunk figure has one source):**

| linear | K × N | FLOP/row = 2·K·N | count |
|---|---:|---:|---:|
| qkv‖z | 5120 × 16384 | 167,772,160 | 48 |
| out_proj | 6144 × 5120 | 62,914,560 | 48 |
| gate‖up | 5120 × 34816 | 356,515,840 | 64 |
| down | 17408 × 5120 | 178,257,920 | 64 |
| q‖k‖v | 5120 × 14336 | 146,800,640 | 16 |
| o_proj | 6144 × 5120 | 62,914,560 | 16 |
| **Σ per row** | | **48.654 GFLOP** | |
| **× 2048 rows = per chunk** | | **99.64 TFLOP** | |

The brief's "~98 TFLOP" is the same quantity reached the other way round -
656 ms × 150 TFLOP/s = 98.4 - i.e. the in-situ GEMM term times the rate it was
priced at. The in-situ blended rate is therefore 99.64 / 0.6563 s =
**151.8 TFLOP/s (derived)**, consistent with the brief's 150. This addendum
uses 99.64 TFLOP for the chunk's linear work everywhere; the brief's rounding
changes no threshold below (§A2 shows the check).

**The bound from below (derived):**

| step | value | grade |
|---|---:|---|
| vLLM pp4096, 2 × 2048, HTTP-inclusive | 1973 t/s = 2076 ms / 4096 | measured, external (docs/BENCHMARKS) |
| per 2048-token chunk | **1038 ms** | derived |
| if the GEMMs were the whole chunk: R = 99.64 / 1.038 | **≥ 96.0 TFLOP/s** | derived - a hard floor; the row is HTTP-inclusive, so the true device floor is higher |
| GEMM time if oneDNN ran at the 183.5 peak: 99.64 / 183.5 | 543 ms | derived |
| ⇒ vLLM's non-GEMM time per chunk | **≤ 495 ms** | derived upper bound |

**What vLLM spends outside the GEMMs (estimated, each term labelled):**

| term | estimate (ms/chunk) | basis |
|---|---:|---|
| GDN (Intel Xe2 chunked kernel, §2) | 150-300 | §2.3's 150-400 band, its top cut by the ≤ 495 bound above; ours is 303 measured |
| attention (vLLM's XPU flash-attention) | 60-150 | must fit under the 495 bound with the other terms; ours is 83 measured on the composed path |
| small ops (norms, SiLU·mul, gated norm, RoPE, conv, residual) | 80-150 | ours is 131 measured; vLLM's are fused-op library calls of the same shapes |
| HTTP, tokenizer, Python scheduling, per chunk | 15-40 | vLLM's row is HTTP-inclusive; the two chunks share one request |
| **sum** | **305-640 → clipped to 305-495** | the sum's top is impossible: it would need the GEMMs above peak |

At the central estimate (380 ms non-GEMM) the GEMMs have 658 ms, so
R = 99.64 / 0.658 = **151 TFLOP/s**. At the edges: 305 ms → 136; 495 ms → 183.5
(peak). Nothing measured on this part exceeds 164.21, so the band's top is
trimmed to 175.

> **New pre-registered prediction for oneDNN's `bf16_int4` matmul (P-C):
> R ≈ 150 TFLOP/s at gate‖up M = 2048, band 135-175, hard floor 96.0
> (derived).** That is *sycl-tla's bf16 rate on int4 weights with no dequant
> pass* - the opposite of §1.5's call. Falsifier, fixed now: a measured
> R < 96 cannot be the path vLLM measured 1973 on (its GEMMs alone would
> exceed the whole 1038 ms chunk); such a reading is a probe defect
> (configuration, layout, or a 128-GRF-class build) to be fixed once, not a
> finding. §1.6's `114 / 221` rules are retired with the prediction they
> served: P-C is a reference target, and its decision rule is §A3.3's.

### A2. The brief's P-A thresholds, checked (no arithmetic error found)

The brief derives: sycl-tla costs ~656 ms/chunk at 150 TFLOP/s; a GEMM at rate
R costs `656 × (150/R − 1)` ms more; removing the host handoffs recovers at most
~112 ms (A24's Ns = 1024 counterfactual, derived, +112.1). Break-even:

```
656 × (150/R − 1) = 112.1   ⇒   R = 150 / (1 + 112.1/656) = 150 / 1.1709 = 128.1   ✓
```

With the exact in-situ inputs (656.3 ms, 151.8 TFLOP/s blended) the same
equation gives 129.7. That is a rounding, not an error, and it is inside the
0.5-0.9 % run-to-run spread of every `--pp` row on the record. **Pre-registered
as the brief states: break-even 128; go at ≥ 140; stop below 128.** A reading
in [128, 130) is reported as "at break-even" rather than "above" it.

What each threshold buys, so the verdict is arithmetic (derived):

| R at gate‖up M=2048 | GEMM delta vs today | net with +112.1 (Ns 1024) | net with +33.0 (Ns 2048) |
|---:|---:|---:|---:|
| 150 (parity) | 0 | **+112** | +33 |
| 146 (this addendum's prediction, §A3.1) | −18.0 | **+94** | +15 |
| 140 (go bar) | −46.9 | **+65** | −14 |
| 128 (break-even) | −112.8 | **0** | −80 |

Two things this table makes explicit before any number exists. First, the
Ns = 2048 counterfactual does not survive any GEMM loss: **P-B's payoff lives
at Ns = 1024**, which is why P-B measures both widths but its verdict is the
Ns = 1024 row. Second, the ceiling of the whole candidate path: today's chunk
is **1456 ms (measured: 2912.9 ms / 4096 × 2048)**; at parity and the full
counterfactual it is 1456 − 112 = 1344 ms → **1524 t/s = 77 % of 1973
(derived)**; at the predicted 146 it is 1362 ms → **1504 t/s (76 %)**. The
pure-Level-Zero GEMM path **cannot by itself reach vLLM's 1038 ms** - the
remaining ~306 ms is what P-C (in-kernel int4, removing the 205 ms scratch)
and P-D (GDN at vendor time) are measured to price. Written now so that the
Phase-2 recommendation is read against this arithmetic and not fitted to the
numbers after the fact.

### A3. Pre-registrations

One protocol for every timed cell, P2's and the slab probe's: 8 replays, first
3 discarded, median of the last 5, 4 enqueues per replay, one discarded
warm-up; incompressible xorshift bf16 inputs; `ZE_AFFINITY_MASK=1`; iterate
grade unless the DRM-holder check says otherwise; the sycl-tla C2 control
(`gemm_bf16`, same shape, same harness) re-run beside every P-A cell so any
harness-level difference cancels (the slab probe's convention). One defect,
one fix, one measurement; the first correct build's number is the number.

#### A3.1 P-A - our OpenCL C bf16 DPAS GEMM on the L0 list (CANDIDATE, go/no-go)

| | pre-registered |
|---|---|
| **rate, gate‖up M = 2048** | **146 TFLOP/s**, band 135-155 (basis: §A4.9) |
| rate, gate‖up M = 1024 / 4096 | 152 / 114 (sycl-tla's 160.68 / 119.73 × the same 0.95) |
| rate, `down` M = 2048 | 156 (sycl-tla's 164.21 × 0.95) |
| **correctness** | against `gemm_bf16` on identical inputs: **predicted bitwise identical** (same `dpas.8x8` instruction, same ascending 16-wide k order, single accumulator chain per output, alpha = 1 exact, beta = 0, no split-K). Bar if not bitwise: `gemm_batched_test.cc`'s own check - 4096 sampled outputs vs a `double` reference, `err ≤ 64 × 2⁻²⁴ × Σₖ|aₖbₖ| + 2⁻²⁴`. Determinism: two runs into two buffers bitwise equal. |
| **build gate, before any timing counts** | `.zeinfo`: `grf_count 256`, `simd_size 16`, `has_dpas true`, no `spill_mem_size`, no `private_size`; Xe2 `.asm`: 32 `dpas.8x8` per k-tile body, 3 `lsc_load_block2d` + 2 prefetch-form `lsc_load_block2d` per k-tile, 16 `lsc_store_block2d` in the epilogue, no scatter/gather `send` and no register shuffles inside the k-loop (§A4.8). A build that fails this gate is a **defect fixed once** (the 256-GRF option or a spill), not a tuning pass. |
| **decision rule (binding, from the brief)** | **≥ 140 → proceed to P-B. 128-140 → run P-B, report the margin as thin. < 128 → the pure-Level-Zero direction stops here; report and do not tune.** |

#### A3.2 P-B - slab dequant + P-A's GEMM on ONE in-order L0 list (CANDIDATE payoff)

Runs only if P-A ≥ 128. Re-runs A24's Probe A battery (gate‖up, dequant a
`[K][Ns]` bf16 slab into L2, GEMM that slab, repeat) with **zero host waits**:
`pf_dequant_tile`'s slab form already runs on the L0 list; with P-A the GEMM
does too, so the in-order list is the only ordering. Baseline = today's two-pass
(C1 + C2 with sycl-tla's GEMM, ~205 + ~656 ms/chunk).

| | pre-registered |
|---|---|
| **saving, Ns = 1024** | `112.1 − 656 × (150/R_PA − 1)` ms/chunk-equivalent → **+94 at R_PA = 146**, +65 at 140, 0 at 128 (derived, §A2) |
| saving, Ns = 2048 | `33.0 − 656 × (150/R_PA − 1)` → +15 at 146 (derived) |
| **handoff gone - verified, not assumed** | L0 kernel-timestamp events on the last slab dequant and the first GEMM launch of each pair: device-side gap **≤ 5 µs** (vs the 22.35 µs host wait A24 measured); the timed region contains **zero** `zeCommandListHostSynchronize`/`sycl::queue::wait` calls by construction |
| correctness | slab dequant bitwise vs production over the sampled slabs (the slab probe already measured this for the kernel); GEMM as P-A |
| **decision rule** (the slab probe's own bands, applied to the *net* saving, which already contains P-A's GEMM delta because P-A's GEMM is what the battery runs) | **≥ 90 ms/chunk saved → recommend speccing the pure-Level-Zero prefill; 30-90 → priced, operator rules; < 30 → dead** |
| derived consequence to print | chunk = 1456 − saving → t/s; compare against §A2's 1504-1524 ceiling |

#### A3.3 P-C - oneDNN `bf16_int4` matmul (REFERENCE TARGET)

| | pre-registered |
|---|---|
| **rate, gate‖up M = 2048** | **≈ 150 TFLOP/s, band 135-175, hard floor 96.0** (§A1) |
| correctness | 0 mismatches vs `tools/oracle/dequant.py` on real layout-0 weights after the K-contiguous repack (A23's identity-extraction method, one full production matrix), modulo the IEEE `+0/−0` class A23 characterised (D2, 6.642 %); any other mismatch fails the probe |
| **what it sets** | the rate an own in-kernel int4 GEMM must reach to beat the two-pass: break-even `R_int4 = 99.64 / (0.6563 + 0.205) = 115.7 TFLOP/s` (derived); at oneDNN's predicted 150 the four int4 linears cost 664 ms/chunk → **−197 ms/chunk vs today's 861** (derived) - the single largest term any design on the table can remove |
| decision rule | none as a candidate (A0). Record R; if R < 96 → probe defect (§A1's falsifier), fix once. §1.3's residency arithmetic (+12.16 GB second copy) is recomputed against live headroom only if a future spec wants oneDNN's layout - not for this probe. |

#### A3.4 P-D - Intel's Xe2 chunked GDN kernel (REFERENCE TARGET)

| | pre-registered |
|---|---|
| **time per chunk, C = 2048, 48 GDN layers** | **150-300 ms**, central 220 (§2.3's 150-400 band, its top cut by §A1's ≤ 495 non-GEMM bound less attention and small ops); ours is **303 ms measured** (A30) |
| numerics | band re-measured and recorded against our `gdn_chunk` on identical inputs; the **token gate** (golden + RTN + determinism) is the arbiter, not A22's 3.506e-02 / 1.197e-03 (A25/A26 precedent) |
| shim | the torch-free wrapper §2.1 sized at ~130 lines; if torch reaches deeper than the 18 wrapper references, that is the finding and the probe stops |
| **what it sets** | the delta `303 − t_Intel` is the prize for an own DPAS/vector GDN rewrite; **if t_Intel ≥ 303 ms, GDN is not where vLLM wins and the residual gap is entirely GEMM + dequant** - pre-registered interpretation, written before the number |
| decision rule | none as a candidate; record and price |

#### A3.5 The Phase-2 deliverable table (template, filled by measurement)

| path | per-chunk ms | t/s | % of 1973 | source |
|---|---:|---:|---:|---|
| today | 1456 | 1406 | 71.3 | measured (RECORD) |
| P-A + P-B (candidate) | 1456 − saving_PB | | | measured P-B net |
| + own int4 in-kernel GEMM at P-C's rate (target) | − (861 − 99.64/R_PC · 1000) | | | derived from P-C |
| + own GDN at P-D's time (target) | − (303 − t_PD) | | | derived from P-D |

### A4. P-A kernel design - `pf_gemm_bf16`

Concrete enough to implement from; every choice carries its reason and, where
the record does not settle it, says so.

#### A4.1 Contract and launch

```
C[M][N] fp32  =  A[M][K] bf16  ·  B[K][N] bf16          (lda = K, ldb = N, ldc = N)
```

This is `gemm_bf16`'s exact contract (`src/runtime/prefill/gemm.h`) and the
layout the production walk already holds: `A` is the bf16 activation `x`
(`step.cc:52`), `B` is `dequant_to_bf16`'s `[K][N]` scratch (`dequant.h:13`),
i.e. the weight matrix `W[N][K]` stored transposed, so the product is
`C = A · Wᵀ`. The brief's "`Bᵀ` for weights `[N][K]`" is this same product;
the kernel consumes the scratch as it is written today. **No repack, no
second copy, no producer changes.**

```c
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pf_gemm_bf16(__global const ushort* restrict A,
                           __global const ushort* restrict B,
                           __global float* restrict C,
                           uint M, uint K, uint N);
```

Launch through the prefill list: `cx.launch(k, M/256, N/256, 1, {PtrArg(A),
PtrArg(B), PtrArg(C), arg_val(M), arg_val(K), arg_val(N)})`. `M` is a runtime
argument (the family rule: no `M` in a name). Work-group `(gx, gy)` owns the
output tile `m0 = 256·gx, n0 = 256·gy`; **x is the M axis** so consecutive
dispatch ids share a B column-block (§A4.2's raster). Requirements, all met by
every production shape (derived): `M % 256 == 0` (C = 2048, and the M cells
1024/2048/4096), `N % 256 == 0` (34816 = 136·256, 5120 = 20·256, 16384,
14336 = 56·256, 248320 = 970·256), `K % 32 == 0` (5120, 6144, 17408 = 544·32);
bases 64-byte aligned and pitches multiples of 16 bytes (the 2D block IO
rule `probe_gemv_loads.cl` records) - every pitch here is `K·2`, `N·2` or
`N·4` bytes, all multiples of 64. A tail-masked variant for
`M % 256 != 0` is a production item, not part of the probe.

#### A4.2 Tiling across the memory hierarchy, and why those sizes

**Work-group tile 256 (m) × 256 (n) × 32 (k); 32 subgroups in an 8 (m) × 4 (n)
arrangement; subgroup tile 32 (m) × 64 (n) × 32 (k); DPAS atom 8 × 16 × 16.**
These are sycl-tla's numbers (`xe_gemm_config.h`: `Shape<_256,_256,_32>`,
`Shape<_8,_4,_1>`, `XE_DPAS_TT<8,float,bf16>`), kept deliberately: they are
the only tile whose rate on this part is measured (153.69 at gate‖up M = 2048,
164.21 at `down`), and every deviation would be an unmeasured variable in a
one-build probe. The reasons they are right, argued rather than inherited:

| level | what lives there | size | why |
|---|---|---:|---|
| **GRF (one thread = one subgroup)** | C accumulators 32×64 fp32 = 16 `float8` | **128 GRF** | 16 independent `dpas` chains per k-step hide the dpas latency; 32×64 is the largest tile whose accumulators plus one k-tile of operands fit 256 GRF |
| | A fragments, 32 m × 32 k bf16 (4 m-atoms × 2 k-steps × `short8`) | 32 GRF | one 2D block load, §A4.5 |
| | B fragments, 32 k × 64 n bf16 (4 n-atoms × 2 k-steps × `int8`) | 64 GRF | two 2D block loads, §A4.4 |
| | **total live** | **224 of 256** | ~30 left for descriptors, loop, coordinates - the reason `-cl-intel-256-GRF-per-thread` is mandatory (§A4.8) and why 128 GRF gave P2 a 33× loss |
| **L1 / Xe-core (no SLM)** | the WG's current k-tile of A (256×32×2 = 16 KB) and B (32×256×2 = 16 KB), plus two prefetched k-tiles | ≤ 96 KB in flight | every B line is read by the 8 m-subgroups and every A line by the 4 n-subgroups → 6× L1 request amplification (192 KB per k-tile per WG), identical to sycl-tla's; SLM staging is the CUDA idiom - on Xe the 2D block loads deliver fragments to GRF directly and the L1 provides the intra-WG reuse, so SLM would add barriers and copies for nothing |
| **L2 (24 MB)** | per wave of 32 WGs = 8 m-blocks × 4 n-blocks: B 4 × 2.62 MB = 10.5 MB (read from DRAM once, from L2 by 7 more WGs), A 21.0 MB (all of it, every wave), C 8.4 MB written | ~40 MB touched per wave | the slab probe measured exactly this B-residency effect (Ns 1024 slab 10.5 MB → 158.91 TFLOP/s, faster than full width); A is partially resident - its re-read is the DRAM term below |
| **DRAM** | A 21.0 MB, B 356.5 MB, C 285.2 MB (gate‖up M=2048) | min 663 MB; with A re-read per wave ≤ 34 × 21 + 356.5 + 285.2 = **1.36 GB** | at 4.75 ms → **≤ 285 GB/s = 48 % of the 590 GB/s reference (derived)** - DRAM is not the roofline; the XMX issue rate and latency hiding are |

Arithmetic intensity at the subgroup: per k-tile 2·32·64·32 = 131 kFLOP on
6 KB of fragments = **21.3 FLOP/B from L1 (derived)** - that is what lets four
threads on one XVE keep one XMX pipe fed while each other's loads return.

**Raster.** sycl-tla's scheduler picks `AlongM` when `tiles_n > tiles_m`
(`tile_scheduler_params.h:337-339`), true for every production shape, so
its measured rates were taken with m fastest - the wave composition above.
`pf_gemm_bf16` makes that explicit (`gx` = M axis). Owning the raster is a
lever sycl-tla does not give us: at M = 4096 (A = 42 MB > L2) a grouped
raster that walks all N over m-blocks 0-7 first, then 8-15, halves A's
re-read traffic (2.86 → 1.43 GB) at the cost of reading B twice (+356 MB) -
**named, not built** in the probe (§A4.10 R6).

#### A4.3 Subgroup size, grid, occupancy

- **SIMD16.** `intel_sub_group_bf16_bf16_matrix_mad_k16` is defined for
  sub-group 16 (docs/prefill-gdn-scan-dpas §3.1); the device offers 16 and 32.
- **Work-group 512 work-items = 32 subgroups = 32 hardware threads.** With
  `-cl-intel-256-GRF-per-thread` an XVE holds **4** threads (the 128-GRF file
  split two ways instead of four), so an Xe-core's 8 XVEs hold **32 threads =
  exactly one work-group**. Device-wide: 32 Xe-cores × 8 XVEs × 4 threads =
  **1024 thread slots = 32 work-groups in flight**. (At 128 GRF the same
  Xe-core holds 64 threads - the count the DPAS-scan doc measured with its
  256-work-item groups.) P2 measured local size 512 at 256 GRF launching and
  running on this driver.
- **Waves (derived).** gate‖up M=2048: 8 × 136 = **1088 WGs = 34.0 waves, no
  tail**; `down`: 8 × 20 = 160 = 5.0; M=1024: 544 = 17.0; M=4096: 2176 =
  68.0. Per WG 2·256·256·5120 = 671 MFLOP; at the 183.5 peak an Xe-core does
  5.73 TFLOP/s → 117 µs per WG; at 150 TFLOP/s → 143 µs; 34 waves × 143 µs =
  4.86 ms, against C2's measured 4.751 ms at 153.69 - the model reproduces
  the record to 2 %.
- Subgroup `s` (0..31): `sm = s >> 2` owns rows `m0 + 32·sm .. +31`,
  `sn = s & 3` owns columns `n0 + 64·sn .. +63` - sycl-tla's
  `Stride<_4,_1,_0>` mapping, so the four subgroups sharing A rows are
  adjacent.

#### A4.4 B: packing and staging

**B is not repacked and not staged. The VNNI pairing DPAS needs is done by
the LSC in the load**, exactly as sycl-tla's `XE_LOAD_2D_VNNI` does (its
`block_2d_transform_selector` picks the `V` message for N-contiguous B):

```c
uint bfrag[2][32];                       // [n-half][32 dwords] = 2 × 2 KB
intel_sub_group_2d_block_read_transform_16b_32r16x2c(
    (__global void*)B, (int)(N * 2u), (int)K, (int)(N * 2u),
    (int2)((int)(n0 + 64u * sn + 32u * h), (int)k0), bfrag[h]);   // h = 0, 1
```

One message loads 32 k-rows × 32 n (2 KB, 32 full 64-byte lines) and returns,
per lane `n`, `bfrag[h][c*16 + j] = (B[k0+2j+1][n0+64sn+32h+16c+lane] << 16)
| B[k0+2j][…]` - even k in the low half, which is precisely the B fragment
the scan probe measured bit-exactly (§3.1 of that doc: "int r = (B[2r+1][n]
<< 16) | B[2r][n]"). The `int8` for (n-atom `b = 2h + c`, k-step `s`) is
`bfrag[h][c*16 + 8s .. +7]` - a compile-time register slice, no `mov`.
Memory cost: **zero** - no second copy, no per-launch pass, no dequant
change. The builtin is declared in IGC's OpenCL headers
(`opencl_cth_pre_release.h:3512`) and covered by its `ocloc` tests
(`IGC/ocloc_tests/Builtins/cl_intel_subgroup_2d_block_io/…/block_reads.cl:330`);
its 8-row `16b` siblings already compile and run in this tree
(`gemv.cl:102`, `probe_attn.cl:527-533`).

Two alternatives, considered and not taken, with the reason:

- **`[N][K]` (K-contiguous) B with the transpose message**
  `intel_sub_group_2d_block_read_transpose_32b_16r8x1c` - one 512-byte
  message per fragment (16 rows × 32 B, half lines), i.e. 4× the messages
  and 2× the line touches of the transform load for the same bytes; it is
  the right tool for the attention K-cache (`gemm_bf16`'s `transB`), not for
  a scratch we control.
- **A pre-tiled `[N/16][K/2][16]`-dword scratch read with 1D
  `intel_sub_group_block_read_ui8`** (the `gemv_bf16.cl` idiom, 512 B per
  message): equal line efficiency, but it changes `pf_dequant_tile`'s write
  pattern on a kernel P3 measured write-allocate-bound, for no gain over the
  transform load. Recorded as the **fallback** if the transform variant
  fails to compile on the box's IGC.

#### A4.5 A: loads; C: accumulation and store

```c
ushort afrag[64];                        // 32 m × 32 k, 2 KB, one message
intel_sub_group_2d_block_read_16b_32r16x2c(
    (__global void*)A, (int)(K * 2u), (int)M, (int)(K * 2u),
    (int2)((int)k0, (int)(m0 + 32u * sm)), afrag);
```

Row-major `[M][K]` as the producers write it; the message returns per lane
`afrag[c*32 + r] = A[m0+32sm+r][k0+16c+lane]` - lane = k column, register =
m row, the measured A fragment. The `short8` for (m-atom `a`, k-step `s`) is
`afrag[s*32 + 8a .. +7]`. This is sycl-tla's `XE_LOAD_2D<16,32,32,16>`
message and its `reorder` between copy and MMA fragments is the identity
(`reorder_xe.hpp`: `dst0 = src0` for same-type) - so our design pays nothing
sycl-tla does not, and saves nothing either.

**C** accumulates in 16 `float8` (fp32) zero-initialised, k ascending in
16-wide steps, one chain per output element; the k-step loop is fully unrolled
(2 steps) and the 32 `dpas` per k-tile are issued **b-outer, a-inner** -
consecutive `dpas` share the B operand (`src1`, the 8-GRF operand) and walk
four accumulators, the pattern IGC fuses into its DPAS macro. The epilogue
writes each atom with one message:

```c
intel_sub_group_2d_block_write_32b_8r16x1c(
    (__global void*)C, (int)(N * 4u), (int)M, (int)(N * 4u),
    (int2)((int)(n0 + 64u * sn + 16u * b), (int)(m0 + 32u * sm + 8u * a)),
    (uint*)&acc[a][b]);                  // 16 stores per subgroup, 8 rows × 64 B each
```

- sycl-tla's `XE_STORE_2D<32,8,16>`. No C read (beta = 0), no SLM, fp32 out
so the correctness bar against `gemm_bf16` is on identical bytes. (A bf16 C
would halve the 285 MB epilogue traffic - a rounding-point change under A6,
priced as a later item, not here.)

#### A4.6 Prefetch

Cooperative, **two k-tiles ahead** (sycl-tla's `MainloopXeL1Staged<2>`):
the WG's A and B tiles for k-tile `t+2` are split across the 32 subgroups so
each issues two small messages and the whole 32 KB lands in L1 once:

```c
// A: 256 rows × 32 k - subgroup s takes rows 8s..8s+7 (8 lines)
intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)A, K*2, M, K*2,
    (int2)(k_pf, m0 + 8u * s));
// B: 32 k-rows × 256 n - subgroup s takes rows 8·(s>>3).., columns 32·(s&7).. (8 lines)
intel_sub_group_2d_block_prefetch_16b_8r16x2c((__global void*)B, N*2, K, N*2,
    (int2)(n0 + 32u * (s & 7u), k_pf + 8u * (s >> 3)));
```

Loop body, mirroring `xe_mma.hpp` line for line: prologue prefetches k-tiles
0 and 1; then per k-tile `t`: `intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)`;
load A (1 message) and B (2 messages) for `t`; prefetch `t+2` if it exists;
32 `dpas`; `intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE)`. The split
barrier keeps the 32 subgroups within one k-tile of each other so their L1
requests for shared lines coincide and the cooperative prefetch is timed for
everyone; it never waits on data. The extension is declared in IGC's
`opencl-c-intel.h:1277-1289`; **if the box's driver does not report
`cl_intel_split_work_group_barrier`, the fallback is one plain
`barrier(CLK_LOCAL_MEM_FENCE)` at the loop top** (priced §A4.10 R3). Why two
ahead and not more: at ~143 µs per WG over 160 k-tiles a k-tile is ~0.9 µs,
so two tiles cover ~1.8 µs of DRAM latency plus contention; the loads probe
measured `prefetch()` distance as a ≤ 2 % effect (pf1/pf2/pf4 on GEMV), and
the record has no GEMM-side distance measurement - **distance is the first
post-probe knob, not a probe variable**.

#### A4.7 What the k-loop costs per thread, and why it can hit the rate

Per k-tile per thread: 3 block loads (A 2 KB, B 2 × 2 KB), 2 prefetches, 32
`dpas` (4096 FLOP each), ~2 barrier ops, a handful of coordinate adds. Four
threads share one XVE's XMX, so per k-tile the XVE must issue 128 `dpas`; at
the 183.5 peak that is one `dpas.8x8` per **5.7 ns per XVE (derived: 183.5 /
256 XVEs / 4096 FLOP)** → ~730 ns of XMX work per k-tile-round, during which
the 12 loads per XVE must return from L1 (prefetched) - hundreds of cycles,
well inside the window. The steady state is XMX-bound provided (i) the
prefetch keeps the loads L1 hits, (ii) the 16 accumulator chains keep the dpas
pipe from stalling on its own latency, and (iii) IGC keeps everything in 256
GRF. Those three are the design's load-bearing assumptions and §A4.8/§A4.10
say how each is checked.

#### A4.8 AOT requirements and the assembly gate

- `add_ocloc_kernel(pf_gemm_bf16 SOURCE … OPTIONS -cl-intel-256-GRF-per-thread)`
  - `cmake/ocloc.cmake`'s existing `OPTIONS` hook carries it into the `ocloc
  compile -options` line at **build** time. This is P2's lesson applied where
  it bit: a runtime `IGC_*` variable never reaches an AOT compile; the option
  must be on the compiler's command line, and the built artifact is the
  evidence. Every prefill kernel today builds at 128 GRF; this is the first at
  256, so the probe adds the option for its target only (probe-only CMake).
- **Dump and inspect on the box, before any timing counts**
  (`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=…` around the `ocloc` call -
  the method §4.2 of the DPAS-scan doc used):
  1. `.zeinfo`: `grf_count 256`, `simd_size 16`, `has_dpas true`,
     `spill_mem_size` and `private_size` **absent**, `barrier_count 1`,
     `slm_size 0`.
  2. Xe2 `.asm`: **32 `dpas.8x8 (16|M0)` per unrolled k-tile body** (64 if
     IGC unrolls two), in runs of four sharing `src1` and marked as a macro
     (`{Atomic}`); **3 `lsc_load_block2d`** (one 32×32 `.d16`, two transform
     `.d16 … t`) plus **2 prefetch-form `lsc_load_block2d … null`** per
     k-tile; **16 `lsc_store_block2d`** in the epilogue; **no** `send` with a
     per-lane address vector inside the loop, **no** `mov` between the loads'
     destinations and the `dpas` sources (the fragments must alias the load
     payloads directly), **no** scratch/stack access anywhere.
  3. A 128-GRF build of the same source is **not** timed; the `.zeinfo`
     check replaces P2's before/after. "Zero spill" alone is not clearance -
     P2's 128-GRF binary also had zero spill and a degenerate schedule; the
     `dpas`-per-k-tile and no-`mov` counts above are the schedule check.

#### A4.9 Predicted TFLOP/s, with its basis

Start: sycl-tla's **153.69 TFLOP/s** at gate‖up M = 2048 (C2, measured) =
83.8 % of the 183.5 peak. The design above issues the **same LSC messages
(`lsc_load_block2d` 32×32 d16 for A, two 32×32 d16-transform for B, 8×16 d32
stores for C, 2-deep cooperative prefetch), the same `dpas.8x8` in the same
count and order, the same 224-GRF budget, the same 512-thread work-group and
raster, and the same split barrier**. Each structural difference, with its
direction:

| difference vs sycl-tla's mainloop | direction | estimate |
|---|:---:|---:|
| `dpas` and the 2D loads reach IGC as **intrinsics** (OpenCL builtins) instead of sycl-tla's **inline vISA `asm` blocks** (`mma_xe.hpp`, `copy_xe_2d.hpp`). IGC can schedule and macro-fuse intrinsics it understands; inline asm is opaque to its LLVM-level passes but scheduled identically by the vISA finalizer. Could go either way. | ± | −3 … +3 % |
| the 2D descriptor payload (base, width, height, pitch, coord) is an argument of every builtin call; sycl-tla keeps one live payload per operand and increments its coordinate. If IGC rebuilds payloads per iteration that is ALU work co-issuing beside the dpas - cheap but not free; if it fails to hoist the uniform parts it is worse. | − | −2 … −6 % |
| no `reorder` stage - sycl-tla's is the identity for bf16→bf16 | 0 | 0 |
| own raster = sycl-tla's effective `AlongM` at every production shape | 0 | 0 (M ≤ 2048); potential + at M = 4096, not in this build |
| first hand-written kernel: prologue/epilogue code quality, coordinate arithmetic, loop overhead | − | −1 … −3 % |
| **net** | | **−5 % ± 5 %** |

> **Pre-registered: 146 TFLOP/s at gate‖up M = 2048, band 135-155.** That is
> above the 140 go bar with the band's lower edge above break-even; a reading
> under 135 means one of §A4.10's named risks fired and the assembly will say
> which. `down`: 156 (164.21 × 0.95); M = 1024: 152; M = 4096: 114 (both
> sycl-tla's cells × 0.95). A 256-GRF failure is not inside this band - it is
> P2's 33× class (~5 TFLOP/s) and is caught by §A4.8 before timing.

#### A4.10 Risks, ranked, each with the measurement that detects it

| # | risk | what it would cost | detector |
|---|---|---|---|
| **R1** | **Register allocation at 224 live GRF.** IGC must keep 16 `float8` accumulators, 64 + 32 fragment ushorts/uints and the descriptors in registers with no dynamic indexing; a private-memory fallback or a conservative schedule is P2's mechanism. | 33× class (P2), or a 10-30 % scheduling tax | §A4.8: `grf_count 256`, no `spill_mem_size`/`private_size`, 32 `dpas` per k-tile with no interleaved `mov`s or scratch `send`s |
| **R2** | **2D descriptor overhead in the k-loop.** Per-call payload construction or failure to prove the payload uniform → ALU instructions or serialised sends between the dpas runs. | −2 … −6 %, possibly more | non-`dpas`, non-`send` instruction count per k-tile body in the `.asm` (target: a few dozen); if high, the fix is loop-carried coordinates - a second build, recorded as such |
| **R3** | **Split barrier unavailable** (`cl_intel_split_work_group_barrier` not reported by the box's IGC/driver). | compile failure → fallback plain barrier, estimated −3 … −8 % (subgroups serialise the next tile's loads behind the slowest dpas run) | immediate at compile; in the `.asm` a single `sync.bar`/`nbarrier` pair vs arrive/wait forms |
| **R4** | **Prefetch not landing** (wrong cache hints, wrong distance, L2-only). | loads become L2/DRAM latency-exposed; every XVE's 4 threads miss in lockstep → rate falls toward 100-120 | `.asm` shows 2 prefetch-form `lsc_load_block2d … null` per k-tile with `.ca` hints; symptom: rate < 135 with R1-R3 clean → distance/hints become the first post-probe knob |
| **R5** | **Fragment order of the transform load differs from the measured dpas B layout** (or A's from the 32-row read). | correctness failure, not a rate; one compile-time index permutation | bitwise/tolerance check vs `gemm_bf16` fails; the identity-matrix probe of §3.1's method isolates which operand |
| **R6** | **L2 behaviour at M = 4096** (A 42 MB > L2; the slab probe's cutoff sits exactly at the cache size). | sycl-tla shows it already: 119.73; ours 114 predicted | the M sweep itself; the grouped raster of §A4.2 is the named lever |
| **R7** | **Occupancy assumption**: a 512-work-item WG at 256 GRF must be schedulable as one Xe-core's 32 threads. | launch failure or 2-wave serialisation | P2 measured local size 512 at 256 GRF running; a launch error is immediate; a rate near half of prediction with clean assembly points here |
| **R8** | **Alignment** (2D block IO: 64 B base, 16 B pitch). | fault or `ZE_RESULT_ERROR_*` at launch | all production pitches are multiples of 64 (derived, §A4.1); USM allocations are ≥ 64 B aligned; a probe assert on the pointers |

The top two are R1 and R2: both are properties of what IGC makes of a
hand-written 224-GRF loop, both are invisible from the source and visible in
the `.asm`, and both are the reason the assembly gate runs before the clock.

#### A4.11 What the record does not settle, and the choice made

- **Prefetch distance** (2): no GEMM-side measurement exists; 2 is sycl-tla's
  measured configuration; kept.
- **Barrier flavour** (split): the record has the DPAS scan at 3 plain
  barriers per sub-chunk hitting its time bar, and no split-barrier
  measurement; the split form is chosen because it is what sycl-tla's 153.69
  used and it is declared in IGC's headers; fallback recorded.
- **fp32 vs bf16 C**: fp32 kept so the correctness bar is on identical bytes
  and no rounding point moves (A6); bf16 C priced later.
- **Tile 256×256×32 with 32×64 subgroups**: kept because it is the only
  measured point; a 256-GRF-bound design has no larger subgroup tile
  available, and a smaller one lowers arithmetic intensity - so the choice is
  also the arithmetic's, not only precedent's.
- **Hardware dispatch order walks x fastest**: assumed for the raster (it is
  what sycl-tla's `BlockIdxX` mapping relies on too); if the wave composition
  differs, the C2 control in the same harness still gives a like-for-like
  rate, and the DRAM headroom (48 %) means the raster is a few-% effect at
  M ≤ 2048.

---

## Phase 2 results (2026-09-15)

Box: `user@box`. Before every timed cell below, `uptime` and
`pgrep -a -f "buildkitsandbox|build_wheel|ninja|cc1plus"` were checked (none
running beyond the checking shell itself) and the DRM-holder / `docker ps -q`
sweep was empty - every cell below is **RECORD grade** by that test. The
operator's docker buildkit build referenced in the brief had already finished
by dispatch; the box was otherwise idle (other logged-in shells, no GPU
users). oneDNN 2026.0 confirmed at `/opt/intel/oneapi/dnnl/2026.0`; sycl-tla
at the pinned `~/sycl-tla` (untouched); `~/vllm-xpu-kernels` present at
`f8318f0…` (Phase 1's HEAD, read-only). Checkpoint resolved as
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` in the HF hub cache (snapshot
`84575a18…`) - the brief's literal default path
(`/home/user/models/qwen38-27b-w4g64-rtn/…`) no longer exists on the box
(another checkpoint, "Agnes", now occupies `~/models`; unrelated concurrent
work, not touched). Protocol for every timed cell: 8 replays, first 3
discarded, median of the last 5, 4 enqueues/replay, one discarded warm-up,
incompressible xorshift bf16 inputs, `ZE_AFFINITY_MASK=1` - §A3's protocol,
unchanged.

### P-C - oneDNN's `bf16_int4` matmul (REFERENCE TARGET)

*Concurrent operator build: none observed* - `pgrep -a -f
"buildkitsandbox|build_wheel|ninja|cc1plus"` was empty at this cell's
preflight check (before the operator's docker buildkit build had (re)started
for the session; it was later confirmed to use only 8 of 44 CPU threads and
~80 GB of the box's 121 GB host RAM, on neither GPU - operator ruling
2026-09-15, not to be waited for once running).

Probe: `tools/probe/probe_gemm_onednn_int4.cc` (new icpx SYCL probe,
`src/sycl/CMakeLists.txt`), calling oneDNN's plain C++ primitive API directly
- `sycl_interop::make_engine/make_stream` over our own
`runtime::prefill::Context`'s queue, `matmul::primitive_desc` with
`set_scales`/`set_zero_points` exactly as `vllm-xpu-kernels`' own
`int4_gemm_w4a16.h` calls them (`bf16_int4`: bf16 src, u4 weights, bf16 dst;
group 64 scale on dim 0; scalar `s8` zero point 8; `fpmath_mode(bf16, true)`)
- no vllm-xpu-kernels machinery (no `GpuEngineManager`, no primitive cache, no
torch) links against this probe.

**Rate (measured, RECORD grade):**

| shape | K×N | M | ms | TFLOP/s |
|---|---:|---:|---:|---:|
| qkv‖z | 5120×16384 | 1024 | 1.567 | 109.66 |
| qkv‖z | 5120×16384 | 2048 | 3.327 | 103.28 |
| qkv‖z | 5120×16384 | 4096 | 6.470 | 106.22 |
| out/o_proj | 6144×5120 | 1024 | 0.571 | 112.85 |
| out/o_proj | 6144×5120 | 2048 | 1.182 | 109.02 |
| out/o_proj | 6144×5120 | 4096 | 2.416 | 106.65 |
| **gate‖up** | 5120×34816 | 1024 | 3.455 | 105.65 |
| **gate‖up** | 5120×34816 | **2048** | **7.429** | **98.28** |
| gate‖up | 5120×34816 | 4096 | 16.636 | 87.78 |
| down | 17408×5120 | 1024 | 1.292 | 141.26 |
| down | 17408×5120 | 2048 | 2.829 | 129.06 |
| down | 17408×5120 | 4096 | 5.020 | 145.46 |
| q‖k‖v | 5120×14336 | 1024 | 1.360 | 110.52 |
| q‖k‖v | 5120×14336 | 2048 | 2.840 | 105.86 |
| q‖k‖v | 5120×14336 | 4096 | 5.818 | 103.35 |

**gate‖up M=2048: measured 98.28 TFLOP/s vs pre-registered ≈150 (band
135-175, hard floor 96.0).** Below the band, but **above the 96.0 hard
floor** - §A1's falsifier ("R < 96 cannot be the path vLLM measured 1973
on… such a reading is a probe defect") does **not** fire, so this is
reported as the measurement, not chased. `down` (141-145) and `gate‖up`
(88-106) bracket the pre-registered central estimate on opposite sides,
consistent with a real per-shape effect (weight-read-to-compute ratio) rather
than a single systematic miscalibration.

**Correctness (measured): FAILED the pre-registered bar.** Identity
extraction (A23's method: `A = I`, `M = K`) against one real production
matrix (layer 0's fused `in_proj_qkv‖in_proj_z`, K=5120, N=16384, the real
checkpoint above, K-contiguous-per-N repacked, no XOR) vs
`tools/oracle/dequant.py`'s formula transcribed in C++: **33,198,358 of
83,886,080 weights (39.6 %) differ from the oracle**, far beyond the
pre-registered "IEEE +0/−0 class" allowance (measured here at 5,372,359 =
6.4 %, matching A23's 6.642 % reference almost exactly - that part of the
bar holds). Every sampled mismatch is a **1-ULP bf16 neighbour** (e.g.
`k=0,n=0`: oracle 0xbcac (−0.020996…) vs device 0xbcab (−0.020874…), adjacent
bf16 values) - not a layout or index error (a wrong-group or wrong-column
bug would produce large, structurally patterned errors, not a uniform
one-bit-of-precision spread starting at the very first element).

**One diagnostic run, to root-cause rather than to tune (not the measurement
of record):** re-ran with `fpmath_mode(strict, false)` in place of the
production `fpmath_mode(bf16, true)` - **identical mismatch count**
(33,198,358), ruling out "oneDNN was permitted to drop precision" as the
cause. The remaining, unconfirmed but consistent explanation: gemmstone's
in-kernel dequant likely computes `q·scale − zp·scale` (two independently
rounded fp32 multiplies then a subtract) rather than the oracle's
`(q − zp)·scale` (one exact integer subtract then one fp32 multiply and a
single bf16 round) - mathematically identical over the reals, a different
rounding path over bf16, and exactly the "many elements diverge by one ULP"
signature measured. This is reported as gemmstone's own numerics, a real
difference from a different kernel generator (§1.5's competing hypothesis),
not a probe defect fixed once - the diagnostic build is not adopted and the
production `fpmath_mode(bf16, true)` config (the one actually built and
shipped) is what the rate table above reports.

**Grade: RECORD** (preflight clean; 8/3/5/4 protocol). **Decision rule: none
as a candidate (A0).** What it sets: §A3.3 pre-registered the break-even an
own in-kernel int4 GEMM would need to clear to beat today's two-pass
(dequant 205 ms + bf16 GEMM 656.3 ms): **R_int4 = 115.7 TFLOP/s (derived,
99.64 / (0.6563+0.205))**. oneDNN's *measured* gate‖up M=2048 rate, 98.28
TFLOP/s, is **below that break-even** - the vendor's own fused int4 W4A16
primitive, on this box, does not clear the bar a from-scratch in-kernel int4
GEMM of our own would have to. That reverses §A3.3's prediction (built from
vLLM's whole-pipeline 1973 t/s and an estimated non-GEMM share) that oneDNN
would set a target "−197 ms/chunk vs today's 861": the measured rate implies
the four int4 linears would cost *more* through oneDNN than through the
two-pass path they would replace, not less. Recorded as the reference
number; not tuned further, and not treated as a candidate regardless (A0).

### P-A - our OpenCL C bf16 DPAS GEMM on the Level Zero list (CANDIDATE, go/no-go)

*Concurrent operator build: none observed* at this cell's preflight check
(same basis as P-C's note above).

Kernel: `tools/probe/pf_gemm_bf16.cl` (new probe-only OpenCL C, AOT
`-cl-intel-256-GRF-per-thread`, `tools/probe/CMakeLists.txt`). Host:
`tools/probe/probe_pf_gemm.cc` (plain g++ L0 probe, links `b70_prefill` only
to call `gemm_bf16` as the same-harness C2 control). Transcribed from §A4
essentially verbatim; two build-time defects found and fixed before the
first correct build (both one-line, both address-space/init pedantry, no
algorithm change):

1. `intel_sub_group_2d_block_write_32b_8r16x1c`'s last parameter is declared
   `private uint*` in IGC's headers; passing `(uint*)&acc[a][b]` (no address
   space qualifier on the cast) let clang infer `__generic`, which IGC
   segfaulted on (ocloc exit `-11`, no diagnostic beyond a frontend error one
   line above the crash). Fix: `(__private uint*)&acc[a][b]`.
2. `probe_pf_gemm.cc`: an unused `fail()` helper and an aggregate-init with
   fewer braces than members tripped `-Werror=unused-function` /
   `-Werror=missing-field-initializers` (this project's own `-Wall -Wextra
   -Werror`). Removed the helper; switched to field assignment.

**Build gate (measured, before any timing counts - `IGC_ShaderDumpEnable=1`,
the DPAS-scan doc's method, §A4.8):**

| check | required | measured |
|---|---|---|
| `grf_count` | 256 | **256** |
| `simd_size` | 16 | **16** |
| `has_dpas` | true | **true** |
| `spill_mem_size` / `private_size` | absent | **absent** (not present in the `.zeinfo` at all) |
| `barrier_count` | 1 | **1** |
| `dpas.8x8` per k-tile body | 32, b-outer a-inner | **32**, confirmed b-outer/a-inner: 4 consecutive `dpas` share the same `src1` register (e.g. `r223:bf` × 4, then `r239:bf` × 4, …) while the 4th operand walks `r26.0/r30.0/r34.0/r38.0:bf` |
| data-returning 2D loads per k-tile | 3 (1 A `.d16`, 2 B `.d16v` transform) | **3** - `load_block2d.ugm.d16.a64` (A) + 2× `load_block2d.ugm.d16v.a64` (B); **9 total in the file** = 3 (loop) + 4 (prologue, 2 tiles × A+B) + 2 (loop's own t+2 prefetch), matching the design exactly |
| prefetch-form 2D loads (`null` dest) per k-tile | 2 | **2** in the loop (+4 in the prologue) - `load_block2d.ugm.d16.a64.ca.ca … null:0`, 6 of the file's 9 total |
| 2D stores in epilogue | 16 | **16** - `store_block2d.ugm.d32.a64` |
| scatter/gather `send` inside the loop | 0 | **0** |
| `mov` between a load's destination and the `dpas` operand that reads it | 0 | **0** - every `dpas` reads the load destination register directly (`r223:bf`, `r60:bf`, `r26.0:bf`, …), no intervening `mov` |
| split barrier | `intel_work_group_barrier_arrive`/`_wait` present | **present**: `send.gtwy` (signal, arrive) at the loop top, `sync.bar 0x0` (a real blocking wait) after the dpas block - the split form, not two plain barriers |

**One naming deviation, not a substantive one:** this driver's disassembler
names the messages `load_block2d`/`store_block2d`, not
`lsc_load_block2d`/`lsc_store_block2d` as §A4.8 pre-registered the grep
target - same LSC 2D block-IO messages (`.ugm.d16`/`.d16v`/`.d32`,
`cl_intel_subgroup_2d_block_io`'s builtins lower to nothing else), a
compiler-version disassembly spelling, not a different mechanism. **R2 (2D
descriptor overhead) is present exactly where predicted**: a block of `mov`s
building the next-prefetch's 2D descriptor (`blk2d.widthM1/heightM1/pitchM1/
X/Y`) sits between the fence/barrier-wait and the next iteration's loads -
outside the 32-`dpas` block itself, which contains zero `mov` and zero
`send`. **Gate: PASS**, first build, no spill, no private memory, no
scatter/gather; this binary's numbers are the ones below.

**Correctness (measured): bitwise identical to `gemm_bf16` in every cell,
first build - the predicted outcome, not the fallback.** Determinism:
bitwise across two independent runs, every cell.

**Rate (measured, RECORD grade; C2 = sycl-tla's `gemm_bf16` re-run beside
every cell, same harness, same inputs):**

| shape | K×N | M | pf ms | pf TFLOP/s | C2 ms | C2 TFLOP/s | pf/C2 | correctness | determinism |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| gate‖up | 5120×34816 | 1024 | 2.283 | 159.88 | 2.350 | 155.36 | 1.029 | bitwise | bitwise |
| **gate‖up** | 5120×34816 | **2048** | **4.665** | **156.53** | 5.498 | 132.81 | 1.179 | bitwise | bitwise |
| gate‖up | 5120×34816 | 4096 | 11.772 | 124.05 | 12.440 | 117.39 | 1.057 | bitwise | bitwise |
| down | 17408×5120 | 2048 | 2.214 | 164.89 | 2.314 | 157.76 | 1.045 | bitwise | bitwise |

`pf_gemm_bf16` **beats its own same-harness sycl-tla control at every cell
measured** (+2.9 % to +17.9 %). The C2 control itself reads lower in this
session than its historical record (132.81 here vs 153.69 at A24/A28,
gate‖up M=2048; still the correct like-for-like comparator per §A3's
protocol) - session-to-session driver/thermal variance the pre-registration
anticipated and the reason C2 is re-run beside every cell rather than
quoted from the record.

**gate‖up M=2048: measured 156.53 TFLOP/s vs pre-registered 146 (band
135-155).** At the top of the band, essentially matching its upper edge (a
+7.2 % beat of the point prediction, +1.0 % inside the top of the band -
within the same run-to-run spread A2 already flagged, 0.5-0.9 %, times a
wider margin). `down` M=2048: 164.89 vs 156 predicted (+5.7 %). `gate‖up`
M=1024: 159.88 vs 152 predicted (+5.2 %). `gate‖up` M=4096: 124.05 vs 114
predicted (+8.8 %) - the R6 risk (A > L2 at M=4096) costs *less* than
predicted, not more.

**Decision rule (binding, from the brief): ≥ 140 → proceed to P-B.**
**156.53 ≥ 140 - PROCEED TO P-B.**

**Grade: RECORD.**

### P-B - slab dequant + P-A's GEMM on ONE in-order L0 list (CANDIDATE payoff)

*Concurrent operator build: none observed* at this cell's preflight check
(same basis as P-C's note above).

Probe: `tools/probe/probe_pf_gemm_slab.cc`, `probe_dequant_slab.cc`'s protocol
(C1/C2/C3/C4 naming, 8-replay/3-dropped-median harness) with ONE change: the
GEMM everywhere is `pf_gemm_bf16`/`pf_gemm_bf16_slab` (P-A's kernel, raw L0),
not sycl-tla's `gemm_bf16` - so the dequant and the GEMM are now both raw L0
launches on the *same* in-order immediate list, and the lever's inner loop
calls `slab_dequant(); gemm_slab();` with **no `cx.wait()` between them at
all** (only one trailing wait after the whole battery). `pf_gemm_bf16_slab`
is a second entry point added to the SAME `pf_gemm_bf16.cl` file (§A4's
`pf_gemm_bf16` is byte-for-byte unchanged - the two dumped `.asm` files
differ only in an embedded build hash, confirmed by diff - so P-A's gate and
measurement above stand as recorded).

**Correctness (measured): bitwise identical**, `pf_gemm_bf16_slab` (Ns=2048,
n0=0) vs `pf_gemm_bf16` (full width) over the same 4,194,304-element column
window.

**Handoff verified, not assumed (measured, device timestamps):** last slab
dequant's `kernelEnd` to the following GEMM's `kernelStart`, same in-order
list, **1.667 µs** - against the pre-registered pass bar of ≤ 5 µs and A24's
22.35 µs *host*-wait handoff this replaces. The device-side gap is what the
decision rule asked to be checked, not merely "no wait() appears in the
source": **verified**.

**Controls, this probe's own harness (measured, RECORD grade):**

| control | ms | note |
|---|---:|---|
| C1 full dequant (`pf_dequant_tile`) | 1.546 | |
| C2 `pf_gemm_bf16` full width | 4.729 | 154.40 TFLOP/s |
| C3 two-pass, host wait between (C1 then C2) | 6.294 | sum model 6.275, +0.29 % |

**The lever, C4 (measured, zero host waits within a pair):**

| Ns | slabs | C4 ms | overhead = C4−C2 | ratio = overhead/C1 | chunk-equiv = ratio × 210.116 | **saved = 210.116 − chunk-equiv** |
|---:|---:|---:|---:|---:|---:|---:|
| **1024** | 34 | 5.618 | 0.889 | 0.5750 | 120.82 | **+89.29** |
| 2048 | 17 | 5.974 | 1.245 | 0.8053 | 169.21 | **+40.91** |

(Same derivation `probe_dequant_slab.cc`'s own verdict table uses, applied to
this probe's C1/C2/C4 so any harness-level GEMM difference cancels out of the
ratio - the addendum's own instruction.)

**Decision rule (binding, from A3.2): ≥ 90 → recommend speccing the
pure-Level-Zero prefill; 30-90 → priced, operator rules; < 30 → dead.**
**Ns = 1024 measures +89.29 ms/chunk-equivalent - 0.71 ms (0.8 %) under the
90 ms "recommend" line, inside this project's own documented 0.5-0.9 %
run-to-run spread (§A2).** Read exactly, not rounded either way: this lands
in the **30-90, "priced, operator rules" band, at its very top edge** - a
result close enough to "recommend" that a second run could land on either
side of the line, and far enough from "dead" that the lever is real. Ns =
2048 (+40.91) sits mid-band, confirming §A2's own note that "P-B's payoff
lives at Ns = 1024."

**Derived consequence (combining P-A's measured GEMM rate with P-B's
measured interleave saving - the two effects are additive and were kept
separable by design: C3 and C4 above both already run P-A's GEMM, so the
+89.29 above is the interleave effect ALONE, on top of P-A's GEMM speed
already being P-A's own credit):**

```
GEMM effect (§A2's formula, measured R = 156.53):
  656.3 × (150 / 156.53 − 1) = −27.4 ms/chunk (a saving: P-A's GEMM beats
  the 150 TFLOP/s reference point)
Interleave effect (P-B, measured, Ns = 1024): +89.29 ms/chunk-equivalent
Combined saving: 27.4 + 89.29 = 116.7 ms
New chunk: 1456 − 116.7 = 1339.3 ms → 2048 / 1.3393 s = 1529 t/s = 77.5 % of 1973
```

Against §A2's own pre-registered ceiling at "parity and the full
counterfactual" (1456 − 112 = 1344 ms → 1524 t/s, 77 %): the measured
combination (1339.3 ms → 1529 t/s, 77.5 %) lands **at, and fractionally
above,** that ceiling - P-A's measured rate (156.53 > the 150 reference)
buys back almost exactly what the interleave lever left short of its own
112.1 ms theoretical maximum (89.29 vs 112.1). **The pure-Level-Zero GEMM +
slab path still does not reach vLLM's 1973 t/s / 1038 ms** - by design and
by the arithmetic §A2 wrote down before any of these numbers existed: the
remaining ≈ 301 ms/chunk gap is GDN (P-D prices it next) and the in-kernel
int4 dequant P-C already measured below its own break-even.

**Grade: RECORD** (preflight clean; identical protocol to every other cell
in this report).

### P-D - Intel's Xe2 chunked GDN kernel (REFERENCE TARGET)

*Concurrent operator build: buildkit vllm-xpu-kernels, 8 threads, ~80 GB
host RAM* (operator ruling 2026-09-15: it uses only 8 of 44 CPU threads and
~80 of the box's 121 GB host RAM, on neither GPU; graded from the DRM-holder
and `docker ps` checks below, not waited for). DRM-holder sweep and
`docker ps -q` both empty at this cell's preflight check; this probe's own
host allocations are ~65 MB (the A/w/u scratch) plus a few MB of fixture
data, well inside the ~40 GB the operator asked to keep free.

Probe: `tools/probe/probe_pf_gdn_xe2.cc`, a new icpx SYCL target
(`src/sycl/CMakeLists.txt`, always built when the read-only
`vllm-xpu-kernels` checkout is present, independent of the existing
`B70_P5_CUTE_GDN` option). Calls `gdn::kernel_launcher<T, StateT>` directly -
the torch-free template Phase 1 §2.1 found confined to lines 1-1504 of
`chunk_gated_delta_rule_kernels_xe2.hpp` - never executing the file's
torch-using outer wrapper (`chunk_gated_delta_rule_impl_xe2`).

**Two environment shims, both probe-side, neither touching vendor code:**

1. `src/sycl/vllm_shim/torch/all.h` - the pre-registered torch-free shim
   (§A3.4): a minimal `torch::`/`at::`/`TORCH_CHECK` stand-in, just complete
   enough that the wrapper's UNUSED body still typechecks (it is never
   called). One bug found and fixed before the first successful build: the
   wrapper calls `torch::dtype(dtype)` where `dtype` is already a
   `TensorOptions` (this stub's `Tensor::dtype()` return type) - real ATen
   has a distinct `torch::dtype(TypeMeta)` overload for exactly this call
   shape; added the missing overload (`TensorOptions dtype(const
   TensorOptions&)`).
2. **An unanticipated second shim, found only by attempting the build**:
   `vllm-xpu-kernels`' own `gemm.hpp` (included by
   `chunk_gated_delta_rule_kernels_xe2.hpp`, needed for the GEMM building
   blocks the chunked kernel bodies use) declares `constexpr SPIRVScope
   barrier_scope = ScopeWorkgroup;` and calls `barrier_arrive(barrier_scope)`
   / `barrier_wait(barrier_scope)` (unqualified, via its own `using namespace
   cute;`). On this box's icpx + the project's pinned sycl-tla, `gemm.hpp`
   **fails to compile as shipped**: `barrier_scope`'s type at those call
   sites is `int`, not `cute::SPIRVScope` (`cute::barrier_arrive`/
   `barrier_wait` take the unscoped enum `SPIRVScope`; the overload-resolution
   error names the passed argument's type as `const int`). Confirmed
   reproducible by compiling the **unmodified vendor file alone** with the
   project's exact include paths, independent of torch or this probe's own
   code (isolated repro: `#include "gemm.hpp"` with the project's `-I`s,
   `-fsycl` and `-fsycl-device-only` both, reproduces the same two errors at
   the same lines). Root cause not fully isolated (an isolated hand-retyped
   copy of the same function, includes, namespace and template shape did
   *not* reproduce it - something about the real, complete file matters that
   a byte-level reconstruction of its first 108 lines did not capture), but
   the effect is exact and reproducible. **Fix, probe-side, not vendor-side**:
   two `int`-argument overloads of `cute::barrier_arrive`/`barrier_wait`,
   declared in `namespace cute` before including the vendor header, each
   forwarding to the real enum-typed function via `static_cast<SPIRVScope>`
   - an exact-match overload for the `int` argument the vendor code actually
   passes, value-preserving (`ScopeWorkgroup == 2` either way). This is
   *not* the "if it turns out not small" stop condition the brief
   pre-registered for torch depth - it is a **separate, unrelated**
   sycl-tla/vllm-xpu-kernels version mismatch, fixed the same way the torch
   shim is: bridging an environment gap without editing either vendor tree.

**One real numerics defect found and fixed, the hard way.** The first
successful run (torch-free shim + barrier shim, everything else as first
written) produced a **finite-but-astronomical state** (max rel 5.137e+24,
mean rel `NaN`) - a real bug, not a rounding artifact. Root cause: this
probe's initial design fed Intel's kernel **raw** (post-conv-SiLU,
pre-L2-norm) Q/K, reasoning from three "l2norm for q, k" *comments* found
inside `chunk_gated_delta_rule_kernels_xe2.hpp` that the kernel normalizes
internally. **That reasoning was wrong**: grepping the entire 1634-line file
for `rsqrt`, `sum_sq`, `l2norm_eps` or any sum-of-squares computation finds
**nothing** - the comments are stale/descriptive (an input precondition
being documented, not code being described), not evidence of internal
normalization. Fix: L2-normalize Q and K per position per k-head before
upload, replicating `gdn_ref::step`'s exact formula (`inv = 1/sqrt(Σx² +
1e-6)`, `qf = rne(q·inv_q)·kQScale`, `kf = rne(k·inv_k)`, no scale) - one
defect, one fix. This took the state from non-finite to finite.

**Numerics after the fix (measured): still far outside a defensible band.**
gdn_state vs the CPU reference (`gdn_ref::step`, same fixture, one call over
all 2048 positions): **max rel 5.187e+01, mean rel 9.604e-01** - against
A22's *unrelated* reference point of 3.506e-02 / 1.197e-03 (our own kernel,
a different comparison, quoted only for scale). This is not a "reduction
order" band (A25/A26's precedent is for a few-percent shift, not a mean
relative error near 1.0): it means the two computations are still not
computing the same values under this fixture. **Root cause not found within
this probe's remaining scope.** Candidates named, none confirmed: a residual
sign/order convention in the beta (`b`) or gate (`a`) mapping between the
project's `[0,48)`/`[48,96)` `ab_out` slots and Intel's separate `[num_v_
heads][seqlen]` `a`/`b` tensors; a scale convention on `v` or on the state
update this probe assumed rather than verified against the ~1500 remaining
lines of kernel body it did not hand-trace; or a state-layout transpose
error beyond the one already applied (Intel's own comment gives `[...,
head_v_dim, head_k_dim]`, ours `[k][v]` - transposed and corrected here, but
unverified against the kernel's actual write order). **Reported as the
finding, per the brief's own rule for depth found only by attempting the
build - not tuned, not re-fit, not further chased against a probe budget
that had already spent two rounds of shim debugging.**

**Time (measured): 0.795 ms/chunk, C = 2048, 48 v-heads** - 8 replays, drop
3, median of 5. Against the pre-registered 150-300 ms band (central 220)
and our own measured 303 ms/chunk (A30), this is a **382× ratio**, which the
numerics finding above means **cannot be reported with confidence as a
valid apples-to-apples GDN time**: if the still-unexplained state divergence
reflects the kernel taking a data-dependent short-circuit this fixture
triggers (rather than running its full, intended chunked computation), the
time would be an artifact of that, not of the algorithm. The measurement is
recorded exactly as taken - finite, reproducible (0.794 ms and 0.795 ms
across the two runs, before and after the L2-norm fix, consistent with
fixed control flow regardless of input values, which argues mildly against
a data-dependent short-circuit) - but **not adopted as P-D's headline
number** given the accompanying correctness finding. The honest summary is:
**P-D's build and torch-free/environment shims succeeded and are a real,
reusable result; its numerics and therefore its timing's validity did not
clear this probe's own bar, and are reported as an open finding rather than
a trusted price.**

**Grade: iterate** (a real defect was found and fixed, but the cell's
correctness gate - implicit in "measure a valid comparison," not explicitly
pre-registered as pass/fail since P-D has no decision rule - was not met;
downgraded from RECORD on that basis, not on the preflight, which was
clean). **Decision rule: none as a candidate (§A3.4); this probe does not
establish t_Intel with the confidence needed to price an own GDN rewrite
against it.**

### Summary - the A3.5 deliverable table, filled, and the recommendation

§A3.5's template, filled with this section's measurements (the addendum
above it is unedited; this is the filled rendering, not a change to the
template):

| path | per-chunk ms | t/s | % of 1973 | source |
|---|---:|---:|---:|---|
| today | 1456 | 1406 | 71.3 | measured (RECORD, cited from the record) |
| **P-A + P-B (candidate)** | **1339.3** (1456 − 116.7) | **1529** | **77.5** | measured: P-A's GEMM delta at R=156.53 (−27.4 ms, §A2's formula) + P-B's measured interleave saving at Ns=1024 (+89.29 ms/chunk-equivalent); both RECORD grade |
| + own int4 in-kernel GEMM at oneDNN's measured rate (target) | **+152.8** (a cost, not a saving: 861 → 1013.8) | n/a | n/a | derived from P-C's measured 98.28 TFLOP/s - **below** the two-pass break-even (115.7); building an in-kernel int4 GEMM to match oneDNN's demonstrated rate would make the chunk slower, not faster |
| + own GDN at P-D's time (target) | **not applied** | - | - | P-D's 0.795 ms measurement carries an unresolved correctness finding (max rel 51.87 vs the CPU reference, far outside a defensible band) and is not adopted as a trusted price; see P-D above |

**Recommendation.** The pure-Level-Zero direction (P-A + P-B) is real,
measured, and worth speccing on its own terms: **77.5 % of vLLM's rate**,
up from today's 71.3 %, with every number in that combination at RECORD
grade, bitwise or device-timestamp-verified correctness, and a decision
rule that fired GO on the first correct build for both P-A and P-B. It does
**not** close the gap to vLLM, and this Phase 2 measured *why* not, exactly
as pre-registered: the two reference targets that would need to fire to
close the remaining ≈23 % - an in-kernel int4 GEMM removing the dequant
scratch, and an own GDN kernel at Intel's demonstrated rate - **do not
support that path**. P-C measured oneDNN's own fused int4 W4A16 primitive
*below* the rate a from-scratch design would need to clear break-even, on
this box, with a real (if unexplained) correctness divergence from the
oracle beyond the known IEEE class. P-D's kernel numerics did not clear
this probe's own bar for a trusted time, despite two real, fixed defects
along the way (a torch-free shim and a separate sycl-tla/vllm-xpu-kernels
version mismatch, both fixed probe-side) and a third, unresolved one (the
Q/K/V/A/B tensor-convention mapping into Intel's kernel) that this probe's
remaining scope did not root-cause. **Spec the P-A + P-B GEMM-and-slab
path; do not spec an in-kernel int4 GEMM against oneDNN's measured rate;
treat P-D's GDN time as unpriced, not as a target, until its numerics are
independently re-derived** (a follow-up probe, not this one, since it needs
either a line-by-line trace of `chunk_compute_A_kernel` /
`chunk_inverse_kernel` / `chunk_compute_wu_kernel` / `chunk_fwd_o_kernel`'s
~1500 remaining lines or a from-first-principles derivation of Intel's own
Q/K/A/B conventions neither this repo nor Phase 1's reading settled).

---

### P-C follow-up - which oneDNN path was measured (2026-09-15)

Trigger: P-C's 98.28 TFLOP/s (gate‖up, M=2048) implies a chunk GEMM time
(99.64/98.28 = **1013.8 ms, derived**) that leaves only **1038 − 1013.8 =
24.2 ms/chunk for everything else** (GDN, attention, small ops, HTTP) if
vLLM really ran at that rate end to end - far below the ≥305 ms this same
addendum (§A1) already estimated those terms need. The probe never recorded
*which* oneDNN implementation it dispatched, so that gap was unresolved.
This follow-up answers it, box-only, no new pre-registration needed (P-C's
§A3.3 decision rule already covers this reference target).

**1. Implementation dispatched (measured).** `tools/probe/probe_gemm_onednn_int4.cc`
was extended to print `pd.impl_info_str()` at every `matmul::primitive_desc`
creation (the only edit made to the probe; no CMake change was needed).
Rebuilt via `tools/box.sh build` (icpx, unchanged target). Re-run with
`ZE_AFFINITY_MASK=1`, preflight clean (no DRM holders, no
`buildkitsandbox|build_wheel|ninja|cc1plus`; the operator's docker container
was running but untouched, matches Phase 2's own preflight bar) - **RECORD
grade**.

Every primitive the probe creates, across all 5 shapes × 3 M values plus the
correctness cell (16 primitives total), dispatched to **`jit:gemm:any`** -
oneDNN's real gemmstone JIT generator, not a reference/fallback path (`ref`,
`ocl:ref`, or `any`-unresolved would read differently). This rules out "the
probe silently fell back to a slow reference kernel" as the cause of the low
rate.

`DNNL_VERBOSE=1` (this oneDNN's env var; `ONEAPI_VERBOSE` had no effect) exec
lines, verbatim, for the two cells named in the brief:

```
# gate‖up, M=2048 (one of 5 timed replays' 4 enqueues shown, all identical modulo timing):
onednn_verbose,v1,primitive,exec,gpu,matmul,jit:gemm:any,undef,src:bf16::blocked:ab::f0 wei:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-scratchpad:user attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x5120:5120x34816,7.04395

# down, M=2048:
onednn_verbose,v1,primitive,exec,gpu,matmul,jit:gemm:any,undef,src:bf16::blocked:ab::f0 wei:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-scratchpad:user attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x17408:17408x5120,3.31006
```

A second re-run with `DNNL_VERBOSE=2` (diagnostic only, not RECORD-repeated)
surfaced the `create` lines the outer `matmul` primitive delegates to
internally (a nested `gemm` primitive, oneDNN's usual W4A16 implementation
strategy) - also `jit:gemm:any` both times, confirming creation-time
dispatch matches exec-time dispatch:

```
# gate‖up, M=2048, create (cache warm from an earlier shape's kernel build):
onednn_verbose,v1,primitive,create:kernel_cache_hit,gpu,gemm,jit:gemm:any,undef,src_a:bf16::blocked:ab::f0 src_b:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x5120:5120x34816,0.00390625
onednn_verbose,v1,primitive,create:nested_primitive_cache_hit,gpu,matmul,jit:gemm:any,undef,src:bf16::blocked:ab::f0 wei:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-scratchpad:user attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x5120:5120x34816,0.0187988

# down, M=2048, create (cold -- 213.7 ms JIT compile, first time this shape's kernel is built):
onednn_verbose,v1,primitive,create:cache_miss,gpu,gemm,jit:gemm:any,undef,src_a:bf16::blocked:ab::f0 src_b:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x17408:17408x5120,213.723
onednn_verbose,v1,primitive,create:nested_primitive_cache_hit,gpu,matmul,jit:gemm:any,undef,src:bf16::blocked:ab::f0 wei:u4::blocked:ba::f0 dst:bf16::blocked:ab::f0,attr-scratchpad:user attr-fpmath:bf16:true attr-scales:wei:3:f16:64x1 attr-zero-points:wei:0:s8,,2048x17408:17408x5120,0.0319824
```

Re-measured rate this session (diagnostic, not RECORD-of-record - the
existing 98.28 stands as P-C's number): gate‖up M=2048 **97.00 TFLOP/s**,
within the ≈1.3 % run-to-run spread the brief already tolerates elsewhere.
**Answer to "JIT or fallback": JIT gemmstone, confirmed, both at creation
and at every exec.**

**2. Configuration parity with vLLM (read, `~/PycharmProjects/vllm-xpu-kernels`
pulled to `1c7cbeee1c…`, 2026-09-15; box's read-only `~/vllm-xpu-kernels`
unchanged at Phase 1's `f8318f0…`, no relevant file in the diff between the
two HEADs).** Field by field, probe (`probe_gemm_onednn_int4.cc:174-187`)
vs vLLM (`csrc/xpu/onednn/int4_gemm_w4a16.h:12-105`,
`csrc/xpu/onednn/onednn_runtime.h:17-97`):

| field | probe | vLLM | match? |
|---|---|---|---|
| joint dtype | `bf16_int4` (bf16 src, u4 weights, bf16 dst) | `bf16_int4` (`in_dtype == BFloat16` branch, `int4_gemm_w4a16.h:33-34`) | yes |
| src dims/strides | `{M,K}`, `{K,1}` (row-major) | `mat1` row-major, `lda = mat1_strides[leading_dim]` - same for a 2D contiguous src | yes |
| weight dims/strides | `{K,N}` u4, strides `{1,K}` (K-contiguous-per-N, `repack_k_contig`) | `ldb = mat2.strides()[dim-1]*8`; consumed only correct when the weight is already K-contiguous-per-N - Python (`_quantize_convert.py:211-214`, `xpu.py:69-74`) relayouts to exactly that stride order before the op runs | yes (same repack Phase 1 §1.3 already read) |
| dst dims/strides | `{M,N}` bf16, `{N,1}` | row-major bf16 result | yes |
| scale mask/group/dtype | `mask=(1<<0)+(1<<1)`, group `{64,1}`, `f16` | `mask=(1<<0)+(1<<1)`, group `{group_size,1}` (=64 for this g64 checkpoint), `get_onednn_dtype(scale)` = f16 for this checkpoint (Phase 1 §1.2/S3) | yes |
| zero point mask/dims/dtype | scalar, `mask=0`, `s8` | `zp.dim()==1` branch: `mask=0`, `{}`, `s8` (our checkpoint is GPTQ g64 sym, Phase 1 §1.2/A28) | yes |
| fpmath mode | `fpmath_mode(bf16, true)` | `set_fpmath_mode(f16,true)` then, for bf16 src, `set_fpmath_mode(bf16,true)` (`int4_gemm_w4a16.h:80-83`) - last call wins, net = bf16/true | yes |
| scratchpad mode | `user` | `user` (`int4_gemm_w4a16.h:61`) | yes |
| bias | none | none for this checkpoint's linear layers (no bias tensor in the GPTQ AutoRound Qwen3.8-27B linears; Phase 1 never flagged one) | yes (assumed from architecture, not independently re-verified this session) |
| engine/stream construction | `sycl_interop::make_engine/make_stream` over probe's own `runtime::prefill::Context` queue | `sycl_interop::make_engine/make_stream` over `c10::xpu`'s device/context and current XPU stream, cached in `GpuEngineManager`/`GpuStreamManager` (`onednn_runtime.h:26-97`) | architecturally identical pattern (Phase 1 §1.4); different queue object, same L0 in-order semantics - not rate-affecting |
| primitive caching | none (probe creates each shape's primitive once, reuses across the timed loop) | `matmul_primitive_create_and_cache`, keyed by shape/strides/dtypes | not rate-affecting once warm (both amortize JIT compile before the timed region) |
| **oneDNN version/build** | **`/opt/intel/oneapi/dnnl/2026.0`, v3.11.4, commit `0291f89430882d350585d276cc0d0eda623906e9`** (`DNNL_VERBOSE` self-report, matches `dnnl_version.h`/`dnnl_version_hash.h` under that path) - Intel's prebuilt oneAPI package, shared `libdnnl.so`, linked by `src/sycl/CMakeLists.txt`'s plain `CACHE PATH` (not a project-pinned checkout) | **fetched from source**, `https://github.com/uxlfoundation/oneDNN.git` at commit `0e2a5bfeef1bfbffc3137464606540233086ce9b` (tag **v3.13**, `CMakeLists.txt:53-60`), built via `FetchContent` as **`STATIC`**, `DNNL_GPU_RUNTIME=SYCL`, `DNNL_CPU_RUNTIME=NONE`, `DNNL_ENABLE_PRIMITIVE_CACHE=TRUE`, `DNNL_ENABLE_CONCURRENT_EXEC=TRUE`, `DNNL_EXPERIMENTAL=TRUE` (`cmake/Modules/FindoneDNN.cmake:21-59`) | **no - different oneDNN release** |

Every other field matches exactly; the one mismatch is which oneDNN the two
sides actually run.

**3. Which oneDNN vLLM actually links (confirmed by reading, not guessing).**
vllm-xpu-kernels does **not** use the box's `/opt/intel/oneapi/dnnl`
installation at all - `cmake/Modules/FindoneDNN.cmake` unconditionally
`FetchContent`s its own oneDNN from `uxlfoundation/oneDNN.git` at the tag
pinned in the top-level `CMakeLists.txt` (currently **v3.13**, commit
`0e2a5bfeef1bfbffc3137464606540233086ce9b`, dated 2026-07-17). The box's
oneAPI 2026.0 package ships **v3.11.4** (commit
`0291f89430882d350585d276cc0d0eda623906e9`, dated 2026-07-01) - a real,
confirmed, non-trivial version gap, not a probe misconfiguration. Between
the two tags, on the same oneDNN repository (local clone
`~/PycharmProjects/oneDNN`, `git log v3.11.4..v3.13`): **37 commits touch
`src/gpu/intel/gemm/jit/include/gemmstone` or `src/gpu/intel/jit/gemm`**,
of which **5 explicitly target the Xe int4 GEMM path** and are present in
v3.13 but absent from v3.11.4's branch: `a19ed3ed18` "xe: ggemm: upconvert
int4 to int8 for mixed dpas", `fb3c82091b` "xe: copy plan: re-use registers
in int4 downconvert", `c0ba3dc265` "xe: ukernel: use block loads instead of
vnni loads for int4", `d01e750409` "xe: gemm: jit: fixup int4 copy plan
handling", `b27fab2773` "x64: matmul: bf16 with int4 zero points fix". This
is evidence of a real, on-topic version gap in exactly the kernel family
under test, on Xe - not proof of a specific speedup magnitude (no build of
v3.13 was measured; see §4).

Searched the box (read-only) for an already-built matching oneDNN before
considering any fix:

- `~/vllm-xpu-kernels/.deps/onednn-src`: present, but at commit `80afa71049…`
  ("doc: added release notes for v3.12") - **stale**, does not match the
  current pin (v3.13, `0e2a5bfeef1…`); a leftover `FetchContent_Populate`
  from before the pin was last bumped (dated 2026-08-08 on disk).
- `~/vllm-xpu-kernels/.deps/onednn-build`: exists but was **never
  configured or built** - no `CMakeCache.txt`, no compiled object, no
  `libdnnl.*` anywhere under it.
- Box-wide (`find / -xdev -iname 'libdnnl*'`, `-iname '_xpu_C*.so'`,
  single-filesystem box so this covers `/home` too): no compiled
  vllm-xpu-kernels extension exists anywhere, and the only other oneDNN
  present is Ubuntu's apt-packaged CPU-only `libdnnl.so.3.6`/`.so.3.9`
  under `/usr/lib/x86_64-linux-gnu` - older than even the box's own oneAPI
  package, no SYCL GPU runtime, not usable for this probe regardless of
  version.
- The operator's docker buildkit container (`414825506b24`, running,
  untouched) may contain a v3.13 build inside its image layers, but per the
  brief's hard rule this session does not exec into or otherwise inspect
  containers.

**No usable v3.13 (or any GPU-enabled, non-stale) oneDNN build exists on the
box outside a container this session is not permitted to touch.** Per the
brief's instruction, this stops here rather than building oneDNN from
source: what a real fix would need is a **fresh** `git clone
https://github.com/uxlfoundation/oneDNN.git` checked out at
`0e2a5bfeef1bfbffc3137464606540233086ce9b` (tag v3.13) - the operator's
`~/vllm-xpu-kernels/.deps/onednn-src` is the wrong tag (stale v3.12) and is
the operator's tree regardless, not to be reused or touched - then `cmake
-S <fresh v3.13 checkout> -B <new build dir> -DDNNL_GPU_RUNTIME=SYCL
-DDNNL_CPU_RUNTIME=NONE -DCMAKE_CXX_COMPILER=icpx …` (an icpx/SYCL build,
several minutes), then repointing
`src/sycl/CMakeLists.txt`'s `B70_ONEDNN_ROOT` at that build's install
prefix for one probe re-run - a multi-minute icpx/SYCL build this session
was told not to perform unprompted.

**4. Fix applied: none.** §3's mismatch (oneDNN version) is real and
confirmed, but its remedy (building oneDNN v3.13 with the SYCL GPU runtime)
is exactly the case the brief pre-empts ("If nothing usable exists, stop
and report… rather than building oneDNN from source"). No second
measurement was taken; **98.28 TFLOP/s (gate‖up, M=2048, from the table
above) remains the only RECORD-grade number**, now confirmed to be a real
gemmstone JIT rate rather than a fallback artifact, but measured against
**oneAPI's packaged v3.11.4, not vLLM's pinned v3.13**.

**5. Arithmetic (derived, unchanged from the trigger above, restated with
its basis explicit).** At the RECORD rate: chunk GEMM time = 99.64 TFLOP /
98.28 TFLOP/s = **1013.8 ms**. Against vLLM's measured, HTTP-inclusive
1038 ms/chunk (§A1): implied non-GEMM time = 1038 − 1013.8 = **24.2 ms/chunk
(derived)**. That is far below every term this addendum's own §A1 estimated
non-GEMM needs (GDN alone 150-300 ms estimated, 303 ms measured on our own
composed path; attention 60-150 estimated, 83 measured; small ops 80-150
estimated, 131 measured; HTTP/scheduling 15-40 estimated) - **not
plausible** as a reading of what vLLM's own GEMMs cost. §1's finding
(genuine JIT dispatch, no fallback) rules out "the probe measured a broken
kernel"; §2-§3's finding (a real, confirmed, unmeasured oneDNN version gap)
is the remaining, unresolved explanation for why 98.28 TFLOP/s does not
reconcile with vLLM's observed 1973 t/s.

**Verdict.** "An int4 GEMM with no dequant pass cannot pay on this box" is
**not established - it remains open**. What this follow-up settled: the
98.28 TFLOP/s number is not a probe defect in the sense the pre-registration
worried about (it is a genuine `jit:gemm:any` gemmstone dispatch, identical
in every configured field - dtypes, layout, scale/zero-point attributes,
fpmath, scratchpad, engine/stream pattern - to vLLM's own call into
`int4_gemm_w4a16.h`). What it did not settle, and could not settle within
this session's constraints: whether vLLM's actual, pinned oneDNN build
(v3.13, fetched from source with the SYCL GPU runtime) reaches a
materially different rate on this same hardware and this same primitive -
a real, dated, on-topic set of Xe int4-gemm commits separates the two
versions, but no build of v3.13 was measured, so its effect size is
unknown, not zero and not confirmed. The 98.28 TFLOP/s figure should be
read as "oneAPI 2026.0's packaged gemmstone's rate for this primitive on
this box," not as "oneDNN's real rate for vLLM's configuration" - those are
not shown to be the same number.

### Spec 2.1 S0 - assembly gate of the production variants (2026-09-15)

`src/kernels/prefill/pf_gemm.cl` promotes P-A's `pf_gemm_bf16` (above) to a
production kernel: `lda`/`ldb`/`ldc` and a batch stride become runtime
arguments, and a `TRANSB` build reads B stored `[N][K]` (the KV cache) in
place. AOT, `-cl-intel-256-GRF-per-thread`, both variants
(`pf_gemm_T0`/`pf_gemm_T1`), gate procedure per §A4.8 above.

**One build-time defect, fixed once (plan 9a Task 1 brief's anticipated
contingency):** `pf_gemm_T1` failed to compile -
`intel_sub_group_2d_block_read_transpose_32b_16r16x1c` is undeclared on this
driver (`ocloc`: "did you mean
`intel_sub_group_2d_block_read_transpose_32b_16r8x1c`?", confirmed against
`opencl-c-intel.h`: only the 8-dword-wide transposed row is exposed, the
16-dword row the design assumed is not). Fix: each n-atom's 16-dword B
fragment is two 8-dword transposed loads (one per k half-tile `ks`) instead
of one 16-dword load - same coverage, same lane, same k order, doubling
TRANSB's B-load message count from 4 to 8. No other change.

**Gate (measured, `IGC_ShaderDumpEnable=1`, both variants):**

```
== T0
      barrier_count:   1
      grf_count:       256
      has_dpas:        true
      simd_size:       16
dpas 32 load2d 9 prefetch 6 store2d 16 gtwy 2 bar 1 scratch 1
== T1
      barrier_count:   1
      grf_count:       256
      has_dpas:        true
      simd_size:       16
dpas 32 load2d 15 prefetch 6 store2d 16 gtwy 2 bar 1 scratch 1
```

`spill_mem_size` / `private_size` / `slm_size`: absent from both `.zeinfo`s
(not present at all, the same "absent means zero" reading §A4.8 uses).
`scratch 1` in both is `//.declare %scratchloc (35) ... IsBuiltin` - the
universal per-kernel ABI register declaration every kernel on this IGC
build carries (present verbatim, same line number, in both `.asm` files),
not a real scratch-surface access; the authoritative spill signal
(`spill_mem_size`/`private_size`) is absent from both. **Gate: PASS** -
256 GRF, no spill, `dpas` 32 (the unroll factor is 1, no outer k-loop
unrolling here), `store2d` 16, split barrier present (`send.gtwy` signal +
`sync.bar` wait) in both.

T0's `load2d` 9 (3 data + 6 prefetch) matches P-A's own gate exactly. T1's
`load2d` 15 (9 data + 6 prefetch) is higher than the brief's a-priori
prediction of 11 (5 data + 6 prefetch) because of the defect above: T1's two
extra data loads over T0 the brief anticipated (5 vs 3, the four transposed
dword reads replacing the two transform reads) become **six** extra (9 vs
3) once each of the four transposed reads is split into two 8-dword loads.
Everything else - `dpas`, `store2d`, `gtwy`, `bar`, `grf_count`, absence of
spill - matches P-A's gate exactly in both variants.
