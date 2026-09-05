# Spec 2 - Stage 1 / L3 (rewritten): prefill attention COMPOSED from the inherited GEMM

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **This file replaces `docs/superpowers/plans/2026-09-04-plan6d-spec2-L3-attention.md`.**
> That plan's Branch A (`sycl-tla` fused FMHA) and Branch B (a hand-written
> OpenCL C flash kernel) are both retired by ruling A14 on measurement, not on
> taste. The old file stays in the tree as the record of the retired design and
> of the FMHA numbers that killed it; nothing in it is deleted. What is carried
> forward from it, deliberately and by name: its Task 1 (`attn_prep_chunk`), its
> Task 2 (`attn_gate_chunk`), its `attn_ref`-based test harness, its gates, its
> records discipline and its house style.

**Goal:** Land causal chunk attention for the 16 full-attention layers at the
production chunk width by **composing** it from the GEMM this project already
inherits - `S = QKᵀ`, `P = softmax(S)` (ours), `O = PV` - retiring plan 6b's
temporary `C ≤ 64` route (which runs the decode trio at `M = C` and pays
`attn_part` = 405,798,912 B, `buffers.h:84` scaled to `kAttnC = 64`) and lifting
`Engine::prefill`'s chunk to `PrefillScratch::kC = 2048` (ruling A13).
**Pre-registered price for `attn_chunk` at C = 2048, D = 4096: 85-100 ms/chunk
over the 16 FA layers**, against the fused FMHA's **measured 576.194 ms** at the
identical operating point - a 5.8-6.8× cut on the second-largest term of the
prefill step, with **no hand-written DPAS anywhere**.

**Architecture:** three device entry points behind `src/runtime/prefill/attn.h`,
exactly as interfaces.md declares them, plus one addition to
`src/runtime/prefill/gemm.h`:

- `attn_prep_chunk` - the widened `attn_prep`: runtime `C`, partial RoPE on 64
  of 256 dims, the KV write at `[pos+c]`, the q‖gate split of q_proj's 12288
  columns. **Bit-identical to `attn_prep` at C = 1** except for the one
  deliberate change the DPAS path forces (q is stored bf16, ruling A9).
- `attn_chunk` - **composed**, and the whole subject of this plan. Per kv-head
  group: one batched/M-stacked `gemm_bf16_batched` for `S = QKᵀ`, one OpenCL C
  `pf_softmax_causal` for `P` (bandwidth-bound, causal mask by absolute
  position, in-kernel), one `gemm_bf16_batched` for `O = PV`, one OpenCL C
  `pf_attn_scale_pack` for `out = rne_bf16(O / rowsum)`. **No `attn_part`, no
  fused kernel, no `sycl-tla` FMHA.**
- `attn_gate_chunk` - the widened tail of `attn_reduce`: the
  `attn_output_gate` sigmoid, which nothing else in the prefill path owns.

**Tech Stack:** OpenCL C 3.0 + ocloc AOT (`cmake/ocloc.cmake`, `-device
bmg-g31`) dispatched through `runtime::prefill::Context`'s Level Zero immediate
list; `sycl-tla` (pin `91e5bd735517d8e79591b41e0d0cd37a7bacdca7`, ruling A11)
through the **system** oneAPI (`/opt/intel/oneapi/compiler/2026.1/bin/icpx`,
`docs/10-the-box.md:42`) inside plan 6a's `src/sycl/` icpx sub-project. C++17,
ctest on the box (`tools/box.sh`, JOBS 44).

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - §3.3 **as
amended 2026-09-05 by ruling A14** (the composed design is now the spec's own
text; the FMHA-first paragraph is marked SUPERSEDED there), §3.6 (execution
model, chunk width 2048), §5 L3 (this stage), §6 items 1-4 (the correctness
bars), §8 (constraints, reproduced verbatim below). Read it first; it governs.

**Interface contract:**
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md` - binding,
A1-A14. **Five ruling requests are filed against it below (A15-A19).** Only
A15 blocks: without leading dimensions on `GemmBatch` the K/V cache cannot be
addressed and Task 1 cannot be written as specified. Its fallback is stated and
priced, so the plan does not stall, but the request should be ruled first.

---

## Global Constraints

Spec §8, verbatim:

> - Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain
>   (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or
>   `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the
>   operator's and stays read-only; `sycl-tla` built once as a static
>   dependency, version pinned in the build.
> - Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0;
>   record rows only on a provably idle box (zero DRM holders).
> - Every number labelled measured vs derived vs estimated/external, with
>   grade and conditions; no two values for one quantity without a
>   reconciliation sentence.
> - Nothing pushed; tags local; work on the branch the operator names.
> - Design must not preclude prefix caching: chunk boundaries at multiples
>   of 1024 align with the queued block-snapshot scheme
>   (`docs/04-architecture.md`, "Follow-on"); no work on it here.

And, carried by every task in this plan:

- **The Mac never compiles.** Every build and every test goes through
  `tools/box.sh` (`build` / `test [regex]` / `run <cmd>`); `JOBS` defaults to 44
  and stays there. **Never touch docker, never kill a process on the box.**
- GPU work runs with `ZE_AFFINITY_MASK=1` whenever a container holds card 0.
- `-Wall -Wextra -Werror` on the host; `-cl-denorms-are-zero` is **forbidden**
  and `add_ocloc_kernel` fatals on it (`cmake/ocloc.cmake:40-43`);
  `-cl-fp32-correctly-rounded-divide-sqrt` is the ocloc default and stays.
  The SYCL target keeps 6a/6c's `B70_SYCL_AOT_256_GRF=ON` - a **runtime** IGC
  environment variable does not configure an AOT device compile
  (`docs/probe-prefill-gemm-2026-09-04.md`, "the reusable AOT lesson"), and
  that lesson is worth 33× on this GEMM.
- **No fp atomics anywhere in the path** (spec §6.4). Any `sycl-tla`
  configuration that reduces with split-K or split-KV atomics is disallowed;
  Task 1 Step 4 re-proves the `PersistentScheduler` evidence for the batched
  instantiation, and the softmax's cross-lane reductions are work-group-local
  trees, never atomics.
- **Decode is untouched.** No edit to `src/kernels/attn.cl`, to
  `src/kernels/CMakeLists.txt`'s decode rows, to `src/runtime/capture.cc`'s
  `fa_layer`, or to `DecodeScratch`'s decode fields. The launch/module
  invariants **774 / 19** (`tests/runtime/replay_determinism_test.cc:136,150`)
  do not move, and the decode gate rows **32.22 t/s (RTN) / 29.33 (Vishva)** at
  `2a7df0b` are re-measured once at this plan's final sha (spec §6.5: inside day
  drift ≤ 0.09%, or the difference is explained).
- **Pre-registered predictions before any timing.** Write the number down in the
  task's report section, then run. Task 1 Step 8 and Task 6 Step 2 are the two
  places this plan is scored.
- Full suite green before any commit touching `src/`.
- Every commit message ends with
  `Claude-Session: `.

---

## Why this plan replaces the old 6d - the measured pivot, in four numbers

All four are **measured, iterate grade** (card 1, `ZE_AFFINITY_MASK=1`), at
`sycl-tla` `91e5bd7…`-content-verified.

| what | number | source |
|---|---:|---|
| fused FMHA, head_dim 256, C=2048, D=4096, ×16 FA layers | **576.194 ms** (5.725 TFLOP/s) | `docs/probe-prefill-attn-2026-09-04.md`, §B |
| same route at Intel's own tuned head_dim 128 | 88.108 ms (18.719 TFLOP/s) | same, §B |
| unfused GEMM at the **QKᵀ** shape (M=2048, K=256, N=4096) | **54.80 TFLOP/s** | `docs/probe-prefill-gemm-2026-09-04.md`, attn cells |
| unfused GEMM at the **PV** shape (M=2048, K=4096, N=256) | **40.59** single-head (8 WGs); **73.50** at 16 WGs | same |

The defect hunt came back clean on all three axes - causal trimming is active
and source-verified, the image is SIMD16/256-GRF with `has_dpas: true` and
`lsc_load_block2d.ugm` operand paths, and the grid is the expected one. **The
FMHA cost is structural, not a bug.** Meanwhile the same library's *unfused*
GEMM reaches 164.21 TFLOP/s at the production shapes (89.5% of the derived
183.5 TFLOPS bf16 peak). Attention **is** two GEMMs and a softmax; ruling A14
composes it from the part of Intel's code that works. This also resolves the
"how does a naive server reach 2450 t/s" puzzle: torch SDPA on XPU is oneDNN
GEMMs plus a softmax - unfused attention on a good GEMM. The third party was
not running a clever kernel; they were avoiding a bad one.

**The retired FMHA path is not deleted.** Task 6 Step 5 records it as a priced
negative result in docs/12 and docs/07: what was tried, what it measured, which
defect hypotheses were tested and refuted, and why the composed path wins *here*
(head_dim 256 at GQA 6:1 with a 24 MB L2) while the fused form is the right
answer at head_dim 64-128 on the same silicon.

---

## The numbers this plan is built on

Every row is labelled. `iterate` grade means card 1 with card 0 possibly held;
`record` grade means a provably idle box (zero DRM holders), medians of 3.

| quantity | value | grade |
|---|---:|---|
| GEMM at QKᵀ shape (M=2048, K=256, N=4096) | 54.80 TFLOP/s | measured, iterate |
| GEMM at QKᵀ shape, M=4096 | 49.91 TFLOP/s | measured, iterate |
| GEMM at PV shape, M=2048 (8 WGs) | 40.59 TFLOP/s | measured, iterate |
| GEMM at PV shape, M=4096 (16 WGs) | 73.50 TFLOP/s | measured, iterate |
| GEMM peak observed (down, M=2048) | 164.21 TFLOP/s | measured, iterate |
| derived XMX bf16 peak (vendor int8 ÷ 2) | 183.5 TFLOP/s | derived, external basis |
| DRAM bandwidth used for every traffic model here | 590 GB/s | measured, `docs/15` |
| L0↔SYCL handoff, wait form | 8.569 µs | measured, iterate, P1 |
| L0↔SYCL handoff, L0-event form | 14.797 µs | measured, iterate, P1 |
| L0 immediate-list per-launch, N≥64 | 2.244 µs | measured, iterate, P1 |
| fused FMHA, C=2048, D=4096, ×16 layers | 576.194 ms | measured, iterate, P4 |
| GEMM term of a C=2048 chunk (corrected P2, all layers) | 680.062 ms | derived from measured, iterate |
| vLLM bar | 1973 t/s pp4096, HTTP-inclusive | external, measured |
| per-token forward | 48.97 GFLOP | derived |
| decode gate rows at `2a7df0b` | 32.22 / 29.33 t/s | record |

**The operating point every price in this plan is quoted at**, chosen so it is
directly comparable with P4's FMHA battery: **C = 2048 new queries, `pos` = 2048
cached, so D = pos + C = 4096 total keys**, over the 16 FA layers. P4's
"C=2048, depth 4096" row is this point: its own work figure of **3.299 TFLOP**
reproduces exactly as `2 × (QKᵀ + PV) × 24 heads × 2048 × 256 × 4096 × 16
layers` over the **full** (un-trimmed) rectangle, which is the arithmetic this
plan uses too. Reconciliation with the old plan's causal-trimmed 1.67%-of-FLOPs
figure: that counted only the executed lower triangle; this plan executes the
full rectangle by construction (§"The causal rectangle", below) and prices the
difference rather than hiding it.

---

## Interfaces: what this plan consumes

**From plan 6a / 6b (stream S4) - `runtime::prefill::Context`.**
`src/runtime/prefill/context.h`, as interfaces.md declares it after rulings
A1-A3:

```cpp
namespace runtime::prefill {
struct KernelArg { const void* ptr; size_t size; };
struct PtrArg { const void* value_; explicit PtrArg(const void* p) : value_(p) {}
                operator KernelArg() const { return {&value_, sizeof(const void*)}; } };
template <class T> KernelArg arg_val(const T& v) { return {&v, sizeof(T)}; }

class Context {
 public:
  explicit Context(l0::Context& l0ctx);
  void wait();                                       // BOTH queues drained
  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg);
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<KernelArg> args, l0::Event* signal = nullptr);
  void* sycl_queue_raw();
  l0::Context& l0() const;
};
}  // namespace runtime::prefill
// src/runtime/prefill/context_sycl.h (icpx TUs only): sycl::queue& sycl_queue(Context&);
```

Assumed: `launch` takes the grid in **work-groups**, sets the work-group size
from the kernel's `reqd_work_group_size`, resolves arguments at launch time, and
orders launches in issue order on one in-order immediate list. `wait()` drains
the L0 list *and* the SYCL queue. If any of that is false the repair is local to
`attn.cc` and is named in the task that needs it.

**From plan 6b (stream S4) - the prefill scratch and the chunk walk.** Assumed,
each with its repair:

1. `PrefillScratch` exists with `static constexpr uint32_t kC = 2048` (ruling
   A13) and holds `partials` fp32 `[kC][34816]`, `mixer_out` bf16 `[kC][6144]`,
   `attn_q`/`attn_gate` fp32 `[kAttnC][24][256]` and `attn_part` fp32
   `[24][max_len/64][kAttnC][258]`. **If 6b landed `kC = 4096`**, Task 5 Step 2
   changes it to 2048 under A13 and rewrites `tests/runtime/buffers_test.cc`'s
   arithmetic - rewritten, not patched (the discipline plan 5 Task 6 Step 3
   states).
2. `Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0)`
   exists; its fa-layer branch is 6b `step_chunk`'s sub-chunk loop
   `for (sub = 0; sub < C; sub += PrefillScratch::kAttnC)` calling
   `attn_prep_l1` + `attn_l1`. **Task 5 replaces exactly that loop** and deletes
   `src/runtime/prefill/attn_l1.{h,cc}`, `src/kernels/prefill/pf_attn_prep.cl`
   and the four `add_attn_decode(64, …)` / `add_attn_reduce(64, …)` rows -
   which is the retirement ruling A10 already scheduled for this plan.
3. The chunk's fused qkv linear reaches attention as **fp32 `[C][14336]` at
   S = 1** in `PrefillScratch::partials` - 6b ruling R1 fixes S=1 on the prefill
   path and 6b's own `attn_prep_l1` already takes `const float* qkv_partials`.
   This plan takes the same form (request **A19**), so the bf16 rounding happens
   *inside* `pf_attn_prep_chunk` at exactly decode's rounding point
   (`attn.cl:350`) - ruling A6's shape, applied to attention.
4. `oracle-out-long` (a ≥ 2048-id oracle prompt, spec §6.2) is registered and
   `tests/prefill/prefill_gate_test.cc` carries the multi-chunk row at
   `C = 1024`, capped or skipped while the temporary path clamps to 64.
   Task 5 Step 7 flips it on; **its assertions do not change**.
5. Whatever `--pp-profile`-style attribution 6b built. If none, Task 6 Step 1
   builds the drain-bracketed instrument described there.

**From plan 6c (stream S1) - `gemm_bf16`.** `src/sycl/gemm.cc`'s single
`sycl-tla` instantiation: work-group tile `Shape<_256,_256,_32>`,
`XE_DPAS_TT<8,float,bfloat16_t>`, subgroup layout `Shape<_8,_4,_1>` (32
subgroups), `MainloopXeL1Staged<2>`, `IntelXeGeneric` epilogue with
**`ElementC = void`**, `PersistentScheduler` named explicitly, `StrideB`
expressed over **`(N, K, L)`** - "the easiest bug here", and the reason Task 1
Step 5's non-packed-stride test exists. `Gemm::get_workspace_size(args) == 0`
and `cudaStream_t` is `sycl::queue*`, so `op.run(&q)` launches on **our**
L0-derived queue. Task 1 reuses that chain verbatim by extracting it into a
header; it does not re-derive it.

**From decode - the arithmetic that must be reproduced.**
`src/kernels/attn.cl`'s four stated orders and its rounding discipline
(`attn.cl:126-196`), and its host twin `tests/kernels/attn_ref.h:22-92`. The
column map: q-head `h` at `[h·512, h·512+256)`, its gate at the next 256;
k-head `j` at `12288 + j·256`; v-head `j` at `13312 + j·256`; GQA 6:1, so q-head
`h` reads kv-head `h/6`. The cache layout `[pos][4][256]` bf16 per layer
(`buffers.h:74`), read and written **in place, never relayouted** (spec §3.3).
`SCALE = 0.0625f = 1/√256` applied to the score **after** the reduction and
**before** the max (`attn.cl:502`).

---

## Interfaces: what this plan produces

```cpp
// src/runtime/prefill/gemm.h  (stream S1, addition - ruling A14 + request A15)
namespace runtime::prefill {
// C[l] = A[l] · B[l] for l in [0, L).  A bf16 [M][K] with row pitch lda,
// B bf16 [K][N] with row pitch ldb (or [N][K] with row pitch ldb when transB),
// C fp32 [M][N] with row pitch ldc.  Same determinism contract as gemm_bf16:
// PersistentScheduler, no split-K, no atomics, zero workspace.  M, K, N, L and
// every pitch/stride are RUNTIME values.  Throws, naming the cause, on any
// unsupported dim or misalignment.
struct GemmBatch {
  uint32_t M, K, N, L;
  size_t lda, ldb, ldc;              // row pitches, in ELEMENTS
  size_t strideA, strideB, strideC;  // element strides between batch entries; 0 is legal
};
void gemm_bf16_batched(Context& cx, GemmBatch b, const uint16_t* A,
                       const uint16_t* B, float* C, bool transB = false);
}
```

```cpp
// src/runtime/prefill/attn.h   (stream S3) - interfaces.md's two declarations,
// plus attn_gate_chunk (request A16) and the scratch parameter (request A18).
namespace runtime::prefill {
// (1) Write the chunk's K/V (RoPE applied on the 64 rotary dims) into the cache
//     at positions [pos, pos+C) - the widened attn_prep.  `qkv` is the fused
//     linear's fp32 output at S = 1 (request A19); the bf16 round happens here,
//     at attn.cl:350's point.  `q_out` is bf16 (ruling A9).
void attn_prep_chunk(Context& cx, uint32_t layer, uint32_t pos, uint32_t C,
                     const float* qkv /* fp32 [C][14336] */,
                     const float* fa_small, const float* rope,
                     uint16_t* q_out /* bf16 [C][24][256], RoPE'd */,
                     uint16_t* kv_k, uint16_t* kv_v);
// (2) Causal attention of the C queries over cache [0, pos+C).
//     out: bf16 [C][24][256] (pre output-gate, pre o_proj) - and out[c][h][d]
//     is exactly decode's rne_bf16(acc/sm), so attn_gate_chunk below is
//     attn.cl:633 with the inner rounding already performed.
void attn_chunk(Context& cx, PrefillScratch& s, uint32_t layer, uint32_t pos,
                uint32_t C, const uint16_t* q, const uint16_t* kv_k,
                const uint16_t* kv_v, uint16_t* out);
// (3) The attention output gate - the last op of decode's attn_reduce, widened.
//     x_out[c][h*256+d] = rne_bf16(f32(out[c][h][d]) * sigmoid_f32(gate col))
//     x_out: bf16 [C][6144] - o_proj's activation input.
void attn_gate_chunk(Context& cx, uint32_t C, const float* qkv,
                     const uint16_t* attn, uint16_t* x_out);
}
```

Device kernels, all under `src/kernels/prefill/`, all `pf_`-prefixed, all **one
binary each with no compiled `M`, `MAXLEN` or block constant** (spec's
"`M` is always a runtime argument" rule; ruling A10's `M = 64` exception is
retired by Task 5):

```c
// src/kernels/prefill/pf_attn_chunk.cl  -> ocloc target `pf_attn_chunk`
__kernel void pf_attn_prep_chunk(__global const float* qkv, __global const float* fa_small,
                                 __global const float* rope, __global ushort* q_out,
                                 __global ushort* kv_k, __global ushort* kv_v,
                                 uint pos, uint C);
__kernel void pf_kv_zero_pad(__global ushort* kv_k, __global ushort* kv_v, uint from);
__kernel void pf_attn_scale_pack(__global const float* O, __global const float* rowsum,
                                 __global ushort* out, uint C, uint h0);
__kernel void pf_attn_gate_chunk(__global const float* qkv, __global const ushort* attn,
                                 __global ushort* x_out, uint C);
// src/kernels/prefill/pf_softmax.cl     -> ocloc target `pf_softmax`
__kernel void pf_softmax_causal(__global const float* S, __global ushort* P,
                                __global float* rowsum, uint pos, uint C, uint Dp);
```

`kv_k` / `kv_v` are **this layer's slices** - the caller adds
`f * max_len * 4 * 256 * 2` bytes with `f = layer / 4`, exactly as
`capture.cc:520-521` and 6b's `step_chunk` do. `layer` is carried for
attribution labelling and for the small-tensor lookup.

New `PrefillScratch` fields (Task 5 adds them; Task 4 consumes them):

| field | type / shape | bytes at kC=2048, max_len=16384 |
|---|---|---:|
| `pf_q` | bf16 `[kC][24][256]` | 25,165,824 |
| `pf_attn` | bf16 `[kC][24][256]` | 25,165,824 |
| `pf_s` | fp32 `[kSHeads=6][kC][max_len]` | 805,306,368 |
| `pf_p` | bf16 `[6][kC][max_len]` | 402,653,184 |
| `pf_o` | fp32 `[24][kC][256]` | 50,331,648 |
| `pf_rowsum` | fp32 `[24][kC]` | 196,608 |
| `pf_kt` *(only if Task 1 Step 3 rules transB unavailable)* | bf16 `[4][256][max_len]` | 33,554,432 |
| **total added** | | **1,308,819,456** |
| retired: `attn_part` + `attn_q` + `attn_gate` | | **−408,944,640** |
| **net** | | **+899,874,816 (+858.20 MiB)** |

---

## Ruling requests to `interfaces.md`

**A15 - `GemmBatch` must carry leading dimensions. (BLOCKING for Task 1.)**
A14's struct is `{M, K, N, L, strideA, strideB, strideC}`, which forces
`lda = K`, `ldb = N` (or `K`) and `ldc = N` - packed. **The operands this plan
must pass are not packed and cannot be made packed without a copy A14 itself
forbids:**

- B for QKᵀ is the K cache, `[pos][4][256]` bf16 → one kv-head's rows have a
  pitch of **1024 elements, not 256**.
- B for PV is the V cache, the same layout, same **1024**.
- A for QKᵀ is `q_out` `[C][24][256]` → a head's rows have a pitch of **6144
  elements, not 256** (unless `q_out` is relayouted head-major, which would be a
  *second* interface change and is not requested).

A14's own sentence - "sycl-tla's Xe GEMM takes ColumnMajor B natively; **expose
it rather than copying**" - is the same argument applied to transposition; the
pitch is the other half of it. Requested: the `GemmBatch` in *Interfaces: what
this plan produces* above, with defaults `lda = K`, `ldb = transB ? K : N`,
`ldc = N` so every existing packed call is spelled the same. Nothing else about
A14 changes; `attn_chunk`'s signature is untouched, as A14 requires.

*Fallback if declined,* fully priced so the plan does not stall: a
`pf_kv_pack` gather kernel producing packed `[Dp][256]` K and V slabs per
kv-head. Cost, derived at C=2048, D=4096: 4 kv-heads × 2 tensors × 4096 × 256 ×
2 B = 16.78 MB written and the same read per layer = 33.55 MB/layer → **0.91
ms/chunk at 590 GB/s**, plus **67.1 MB** of scratch at max_len 16384. Cheap, but
it is pure waste and it is 67 MB on a part already at 16.94-18.81 GB, so the
parameter form is preferred.

**A16 - `attn_gate_chunk` is missing from `attn.h`.** (Re-filed unchanged from
the retired plan's R1; never ruled.) `attn_chunk`'s output is declared
"pre output-gate" and nothing else in the contract applies the
`attn_output_gate`. In decode it is the last line of `attn_reduce`
(`attn.cl:633`); spec §3.5's list of "everything else in the chunk" does not
name it. It is attention's, so S3 owns it. Requested: the declaration in
*Interfaces: what this plan produces*. **This one is load-bearing for Task 5:**
6b's L1 route gets the gate for free from decode's `attn_reduce`, and Task 5
deletes `attn_reduce` from the prefill path.
*Fallback if declined:* identical kernel and identical call site, declared in
`src/runtime/prefill/attn_gate.h` and owned by S4; Task 2 moves file, not code.

**A17 - `attn_prep_chunk` cannot reach the two tensors it must read.**
(Re-filed unchanged from the retired plan's R2; never ruled.) The kernel needs
the layer's FA small block (`q_norm` / `k_norm`, fp32 `(1+w)`, at
`loader::kFaOffQNorm` / `kFaOffKNorm` inside `m_.layer_small[layer].gdn`) and
the RoPE table (`m_.rope`, fp32 `[max_len][2][32]`); `capture.cc:530-531` binds
both and 6b's `attn_prep_l1` already takes both as parameters. Requested: the
two `const float*` parameters shown above.
*Fallback if declined:* `Context::model(const loader::LoadedModel&)`, which
contradicts the declared `Context` API - the parameter form is the smaller
change, and 6b has already taken it.

**A18 - `attn_chunk` needs the composed path's scratch.** The composed
implementation materialises `S`, `P`, `O` and `rowsum`; the declared signature
carries no scratch and `Context` owns none. Requested: insert
`PrefillScratch& s` after `Context& cx` - **exactly the deviation plan 6b
already took for `gdn_chunk`** (6b's `void gdn_chunk(Context& cx,
PrefillScratch& s, uint32_t pos, uint32_t C, …)` against interfaces.md's
scratch-free declaration), for the same reason and with the same shape. A14's
"`attn_chunk`'s signature is UNCHANGED" is read as binding on the **tensor
contract** - no new inputs, no new outputs, no relayout - which this preserves
exactly.
*Fallback if declined, with no interface change at all:* `attn.cc` keeps a
file-local `std::map<Context*, ScratchOwner>` that lazily allocates the four
buffers from `cx.l0()` on first use and releases them with the `Context`. It
works and it is ~15 lines, but the buffers then do not appear in
`PrefillScratch::bytes()` or in `tests/runtime/buffers_test.cc`'s totals - and
a 1.31 GB allocation that the memory ledger cannot see is exactly what that
test exists to prevent. That is why the parameter form is preferred.

**A19 - `attn_prep_chunk`'s `qkv` is fp32 partials at S = 1, not bf16.**
interfaces.md declares `const uint16_t* qkv /* [C][12288+1024+1024] */`.
Requested: `const float* qkv /* fp32 [C][14336], S = 1 */`, i.e. **ruling A6's
shape applied to attention**. Three reasons, all concrete:
(a) A6 already ruled exactly this for `gdn_chunk` - "the bf16 round happens
INSIDE, at exactly decode's rounding point … which is what spec §6.3's
self-consistency bar rests on";
(b) 6b's shipped `attn_prep_l1` already takes `const float* qkv_partials`, so
the bf16 form would make L3 *diverge* from the code it replaces;
(c) the bf16 form needs a separate rounding pass over the fused qkv output:
2048 × 14336 × (4 read + 2 write) B = **176.2 MB per FA layer = 2.82 GB per
chunk = 4.78 ms at 590 GB/s (derived)**, for a value the kernel then widens
straight back to fp32. Under A19 the kernel's first op is
`bf16f(rne_bf16(qkv[c*14336 + col]))`, which at S = 1 is `attn.cl:350`
character for character - so the C = 1 bit-identity claim gets *stronger*, not
weaker.
*Fallback if declined:* keep the bf16 parameter, and Task 5 adds the rounding
pass as a named `pf_qkv_round` launch with the 4.78 ms/chunk priced into Task
6's attribution as a cost of the interface rather than of the kernel.

---

## The composed design, written out once (Tasks 3-4 implement this)

### The dataflow, per FA layer

`D = pos + C` is the number of live keys; `Dp = (D + 7) & ~7u` is `D` rounded up
to the GEMM's 8-element granularity. GQA is 6:1, so the 24 q-heads form **4
groups of 6 sharing one kv-head**. For each group `j ∈ [0,4)`:

```
   A = q_out rows of heads 6j..6j+5      bf16, lda = 6144, batch stride 256
   B = kv_k + j*256                      bf16, ldb = 1024, transB (B is [N=Dp][K=256])
1. S  = A · Bᵀ                           -> fp32 [6][C][Dp], ldc = Dp        (SYCL)
2. P  = softmax_causal(S) ; rowsum       -> bf16 [6][C][Dp] + fp32 [6][C]    (L0)
   A' = P                                bf16, lda = Dp,  batch stride C*Dp
   B' = kv_v + j*256                     bf16, ldb = 1024, no transpose
3. O  = A' · B'                          -> fp32 [6][C][256], ldc = 256      (SYCL)
4. out[c][6j+l][d] = rne_bf16(O[l][c][d] / rowsum[l][c])                     (L0)
```

**How B is addressed under GQA - the question this design has to answer
explicitly.** It is *neither* "stride 0 across the six sharing heads" *nor* a
per-q-head kv loop. It is **a kv-head loop with the six sharing q-heads folded
into the batch (equivalently, into M)**: one launch per kv-group, `B` a single
pointer `kv_k + j*256` with `strideB = 0`, and the six heads expressed as six
batch entries whose only difference is `strideA` / `strideC`. That is strictly
better than a per-head launch, because the PV shape is **occupancy-bound**
(N = 256 is one N-tile → 8 work-groups at M = 2048 per head; the measured row
is 10.67 / 21.55 / 40.59 / 73.50 TFLOP/s at 2 / 4 / 8 / 16 work-groups, a rate
almost exactly linear in work-group count). Six heads in one launch is
**48 work-groups**, 6× the occupancy of a per-head PV.

**The equivalent M-stacked form, and why it matters.** When `C` is a multiple of
the 256-row M-tile - it is, at 2048 and at the 1024 gate width - a batched GEMM
with `strideA = C·lda_pack`, `strideB = 0`, `strideC = C·ldc` decomposes into
*exactly the same output tiles in the same order* as one plain GEMM at
`M = 6C`. Task 1 Step 6 asserts that identity **bitwise**, which gives this plan
two independent spellings of the production call and a free cross-check on the
stride plumbing. Task 4 uses the batched spelling (it does not require `P` to be
head-major-contiguous, so it survives a future `Lq < 6`).

### The causal rectangle, and what it costs

The GEMMs execute the **full** `C × Dp` rectangle; the causal mask lives
entirely in the softmax, which writes `P = +0.0` (bf16 `0x0000`, exactly) for
every masked key. Three consequences, all stated rather than discovered:

1. **It is safe.** Every key index in `[0, D)` has been written - `[0, pos)` by
   earlier chunks or by decode, `[pos, D)` by `attn_prep_chunk` immediately
   before this call - so PV's `0 · v` is `0 · finite`, never `0 · NaN`. The pad
   `[D, Dp)` (at most 7 positions) is zeroed by `pf_kv_zero_pad`, so it is
   `0 · 0`. This is the same trap `attn.cl:459-475` documents for decode, closed
   here by construction instead of by a load predicate.
2. **It costs work.** At `pos = C = 2048` the mean valid-key fraction is
   `(pos + (C-1)/2) / D = 3072.5 / 4096 = 0.7501`, so **33.3% of the QKᵀ and PV
   FLOPs are multiplied by zero (derived)**. At `pos = 0` it is 2×; at
   `pos = 14336` it is 1.14×.
3. **The lever, priced and deliberately not built here.** Splitting each GEMM
   into `C/512` query sub-blocks and trimming `N` (QKᵀ) / `K` (PV) to
   `round_up(pos + 512(t+1), 8)` recovers that 25% - **≈14 ms/chunk at the
   operating point (derived)** - for 4× the launches (≈0.5 ms) and a ragged `P`
   the softmax must zero-fill beyond each sub-block's trim. It is the **first**
   lever if Task 6 scores attention over its price; it is not built now because
   the pre-registered price already clears without it.

### The head tile `Lh`, with the byte arithmetic (ruling A14's requirement)

`S` is fp32 and is the largest object in the path: `Lh × C × Dp × 4` bytes. At
`C = 2048` and `Dp = max_len = 16384` one head alone is **134,217,728 B**, so
`Lh = 24` would be **3.221 GB** - A14's "TOO LARGE", confirmed.

GQA fixes the natural tile at **one kv-group = 6 heads**, because a tile
spanning a group boundary would need two different `B` pointers in one launch.
So the buffers are sized for exactly one group at full depth:

| buffer | formula | bytes at kC=2048, max_len=16384 |
|---|---|---:|
| `pf_s` | `6 × kC × max_len × 4` | **805,306,368** |
| `pf_p` | `6 × kC × max_len × 2` | **402,653,184** |
| `pf_o` | `24 × kC × 256 × 4` (all heads; the pack runs per group into `out`) | 50,331,648 |
| `pf_rowsum` | `24 × kC × 4` | 196,608 |

and the runtime rule - which is what makes this **derived from the depth**
rather than hard-coded - is

```cpp
// Heads that share one QK^T launch and one softmax launch. Six is the GQA
// group; fewer only when the actual Dp cannot fit six in pf_s. At kC = 2048 and
// max_len = 16384 this is 6 at every depth; it degrades gracefully if either
// constant grows.
const uint32_t Lq = std::clamp<uint32_t>(
    uint32_t(s.pf_s_bytes / (size_t(C) * Dp * 4)), 1u, 6u);
require(Lq >= 1, "pf_s cannot hold one head at C=… Dp=…");
```

**Reconciliation with A14's illustration.** A14 wrote "Lh = 4 → S = 537 MB,
P = 268 MB at depth 16384; at depth 4096, Lh = 24 fits". Those numbers are
correct (`4 × 2048 × 16384 × 4 = 536,870,912`) and they are an illustration of
the *constraint*, not a ruling of the *value*: A14's operative sentence is
"Plan 6d's rewrite picks Lh from the measured depth and documents the byte
arithmetic." This plan picks **6**, because 4 is not a divisor of the GQA group
and a tile that straddles two kv-heads needs two `B` pointers; 6 costs 268 MB
more than 4 at full depth and buys the single-`B` launch. `Lh = 24` at shallow
depth is deliberately **not** taken: it would need `pf_s` sized 4× for a case
that only exists below `Dp = 4096`, and the launch count it saves is 3 per layer
(≈0.03 ms/chunk at the measured 2.244 µs/launch).

**The documented alternative, if 1.31 GB is ever unacceptable:** size `pf_s` for
**one** head (`kSHeads = 1` → 134.2 MB) and keep `pf_p` at six. Total drops to
**604 MB (−705 MB)**; the cost is 24 QKᵀ + 24 softmax launches per layer instead
of 4 + 4, i.e. **+40 queue handoffs per layer = +5.5 ms/chunk (derived from P1's
measured 8.569 µs)**. The `Lq` formula above already produces that behaviour
with no code change - only the allocation constant moves.

### Numerics: every place this differs from decode, before any measurement

Ruling A9 already named one (q in bf16). The composed path adds four more.
**All five are stated here so Task 5's golden gate judges a known change, not a
surprise**, and all five are recorded in docs/12 by Task 6.

1. **`q_out` is bf16** where decode's `attn_q` is fp32 (A9). Towards the oracle,
   not away: torch's `Qwen3_5Attention` carries q as a bf16 tensor into the
   score matmul (doc 03).
2. **The score dot is DPAS's reassociation.** Decode: lane `l` accumulates dims
   `d = l + 16t` ascending with explicit `fma`, then a fixed 16-lane pairwise
   tree (`attn.cl:135-138`). The GEMM: fp32 accumulation over K-tiles of 32
   inside the systolic array. Same 256 fp32 terms, different association.
3. **The softmax is two-pass, not online.** Decode carries `(mx, sm, acc)` and
   rescales per 16-position wave (`attn.cl:143-153`). Ours takes the exact row
   max, then `exp`, then the sum. Mathematically identical; the two-pass form
   performs **fewer** roundings (no `resc` chain), so this term is expected to
   *reduce* error, not add it.
4. **The attention weights are rounded to bf16.** This is the one genuinely new
   rounding and the largest new term: DPAS needs bf16 operands, so
   `P[c][k] = rne_bf16(exp(s − mx))` where decode keeps the weight in fp32
   forever. bf16 has 7 explicit mantissa bits → a relative perturbation of at
   most `2^-8` per weight. Because the output is a *weighted mean*, independent
   weight perturbations `ε_k` move it by `Σ (w_k/W)(v_k − ō) ε_k`, whose scale is
   `ε_rms · σ_v / √D_eff` - sub-ulp on the bf16 output in the typical case and
   at most ~1 bf16 ulp on a boundary word (derived). **This is why Task 4's ulp
   bar is 3 and not decode's 2**, and the reason is written down here, before
   the measurement, rather than after it.
   *Mitigation, taken:* `rowsum` is accumulated from **the same rounded bf16
   words the GEMM will consume**, not from the fp32 exponentials - so the
   weights the PV actually sees sum to `rowsum` in exact arithmetic and the
   normalisation carries no bias.
5. **The PV accumulation is DPAS's** (fp32 over K-tiles of 32) instead of
   decode's per-wave `acc = fma(acc, resc, t)` chain. Same operands, different
   association - the same class of change as (2).

Unchanged, and deliberately so: `exp` is the device's plain `exp` on both paths
(never `native_exp`), the `1/16` scale is applied at decode's exact point (after
the reduction, before the max), `1.0f / sqrt` is never `rsqrt`, and the final
`out = rne_bf16(acc / rowsum)` is `attn.cl:631`'s `acc / sm` under
correctly-rounded division. **The golden gate (spec §6.1/§6.2) is the arbiter**
of the composition; the unit bars below are the arbiter of each kernel.

**Considered and rejected, with the reason** (so nobody re-opens them):
- *`S` in bf16 to halve the softmax's read traffic.* Rejected: the score is the
  argument of `exp`, so its **absolute** error sets the weight's **relative**
  error. With `|s|` reaching ~20 after scaling, one bf16 ulp is ~0.0625
  absolute → `e^0.0625 − 1 = 6.4%` weight error (derived). fp32 `S` is not
  optional.
- *Aliasing `P` over `S` to save 402 MB.* Rejected: row `c` of `P` occupies
  bytes `[2cDp, 2cDp+2Dp)` while row `c'` of `S` occupies `[4c'Dp, 4c'Dp+4Dp)`,
  so `P`'s row 100 lands inside `S`'s row 50 - safe only under a strictly
  ascending single-work-group row order, which would serialise a 12,288
  work-group kernel.
- *Aliasing `O` over the head of `S` to save 50 MB.* Legal (S is dead when PV
  runs) but declined: 3.8% of the path's scratch for a real aliasing hazard in
  a buffer three kernels touch.
- *Fusing the pack into `attn_gate_chunk`.* Would save one 25.2 MB read and one
  25.2 MB write per layer = **1.4 ms/chunk (derived)**, but `attn_chunk`'s
  declared output is the **pre-gate** tensor and the contract is binding. Named
  as a fusion lever for a later plan.

### Ordering across the two queues (spec §3.6)

The sequence alternates SYCL (both GEMMs) and Level Zero (softmax, pack, prep,
gate, zero-pad). Ordering is by **`Context::wait()` at each boundary**, which
P1 measured at **8.569 µs** - against the L0-event form's 14.797 µs, so the
drain is both simpler and faster on this driver. Count, derived: per FA layer
one prep, then per kv-group `SYCL(QKᵀ) → L0(softmax) → SYCL(PV) → L0(pack)` =
4 boundaries × 4 groups = 16, then the gate is L0→L0 (free).
**16 × 16 layers = 256 handoffs/chunk = 2.19 ms/chunk (derived from measured).**
Launches: `1 + 4·(1+1+1+1) + 1 + ≤1 = 19` per layer → **≈304/chunk**, i.e.
0.68 ms of L0 launch overhead at P1's measured 2.244 µs (derived).
`Context::launch`'s optional `l0::Event* signal` (ruling A3) is used only for
Task 6's per-kernel device timestamps, never for ordering.

---

### Task 1: `gemm_bf16_batched` - the batch dimension, the pitches, and `transB`

**Files:**
- Create: `src/sycl/xe_gemm_config.h` (the alias chain, extracted from 6c),
  `src/sycl/gemm_batched.cc`
- Modify: `src/runtime/prefill/gemm.h` (the `GemmBatch` declaration),
  `src/sycl/gemm.cc` (include the extracted header; then forward to the batched
  entry - Step 7), `src/sycl/CMakeLists.txt` (one source line),
  `tests/CMakeLists.txt`
- Test: `tests/prefill/gemm_batched_test.cc` (new)

**Interfaces:**
- **Consumes:** 6c Task 1's instantiated chain and the four facts it pinned
  (`LayoutB` RowMajor; `cudaStream_t` is `sycl::queue*`; the
  `PersistentScheduler` static_assert at `xe_gemm.hpp:82-87`; `get_workspace_size`
  returning 0); `runtime::prefill::sycl_queue(Context&)`; 6a's
  `b70_link_prefill` and the `B70_PREFILL_ENABLED` guard;
  `tests/kernels/gemv_ref.h`'s `random_bf16`.
- **Produces:** `void runtime::prefill::gemm_bf16_batched(Context&, GemmBatch,
  const uint16_t*, const uint16_t*, float*, bool transB)`, and the two
  pre-registered PV rows.

- [ ] **Step 1: Pre-register, in writing, before any build.** Append to
  `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/l3-attention-prereg.md`
  (create it):
  - **(P-1, ruling A14's own ask) Batched PV at M = 2048, K = 4096, N = 256,
    L = 24, packed strides: ≥ 60 TFLOP/s.** Basis: single-head is 40.59 at 8
    work-groups and the shape's rate is almost exactly linear in work-group
    count (10.67 / 21.55 / 40.59 / 73.50 at 2 / 4 / 8 / 16 WGs, measured);
    L = 24 is 192 work-groups on 32 Xe-cores.
  - **(P-2, the production form) PV at L = 6, `strideB = 0`, M = 2048 (48
    work-groups): ≥ 90 TFLOP/s.** Basis: the same scan, 3× the 16-WG point,
    capped well under the 164.21 TFLOP/s observed peak.
  - **(P-3) QKᵀ at L = 6, `strideB = 0`, M = 2048, K = 256, N = 4096
    (768 work-groups): ≥ 45 TFLOP/s.** Basis: the single-head cell is 54.80 at
    128 WGs and 49.91 at 256 WGs - mildly *decreasing* in grid size, so the
    prediction is below the smaller-grid measurement on purpose.
  - **Stop bars:** P-2 < 60 → the PV term exceeds 27.5 ms/chunk and Step 8
    prices the alternative; P-3 < 35 → the QKᵀ term exceeds 47 ms/chunk and
    Task 4's price is re-derived before Task 4 starts.
  - **The alternative, written now so it is not invented after a miss:** if
    batching does **not** restore the rate, swap the operand roles for PV -
    `Oᵀ = Vᵀ · Pᵀ` gives M = 256, N = C, i.e. 1 M-tile × 8 N-tiles = 8 WGs
    per head and 192 batched, the same grid by a different route; and if
    *that* is also flat, the per-head QKᵀ form (measured 54.80, 24 launches per
    layer, +0.05 ms/chunk of launch overhead) is the standing fallback for the
    QKᵀ half, with PV left at its measured 40.59 and the attention price
    re-registered at **100-115 ms/chunk**.
- [ ] **Step 2: Failing test first.** Write `tests/prefill/gemm_batched_test.cc`
  (Step 5 lists its cases) and register it in `tests/CMakeLists.txt` inside
  `if(B70_PREFILL_ENABLED)`, beside 6c's `gemm_test`, with
  `target_link_libraries(gemm_batched_test PRIVATE b70_l0)` and
  `b70_link_prefill(gemm_batched_test)`. `tools/box.sh test gemm_batched` - it
  does not link (`gemm_bf16_batched` is undefined). **Paste the failure.**
- [ ] **Step 3: Settle `transB` against the pin, before writing the launcher.**
  Ruling A14 asserts "sycl-tla's Xe GEMM takes ColumnMajor B natively", while
  6c Task 1 recorded "sycl-tla's Xe GEMM takes RowMajor for A/B/C/D" and P2
  measured only RowMajor B. **These are not the same claim and one of them is
  wrong; find out which, first.** On the box:
  ```
  tools/box.sh run 'grep -rn "ColumnMajor\|XE_LOAD_2D_VNNI\|XE_LOAD_2D_TRANSPOSE" \
    /home/user/sycl-tla/include/cutlass/gemm/collective/xe_mma.hpp \
    /home/user/sycl-tla/include/cutlass/detail/layout.hpp | head -40'
  ```
  Then instantiate the ColumnMajor-B chain in `gemm_batched.cc` and build.
  Record the outcome in the task report as one of:
  - **transB available** → Task 4 reads the K cache in place, `pf_kt` is not
    allocated, and this plan's scratch total stands at 1,308,819,456 B;
  - **transB does not instantiate** → record the exact diagnostic, drop the
    ColumnMajor chain, and take the fallback: `pf_k_transpose` (Task 2 Step 6)
    writes a packed `Kᵀ` `[4][256][Dp]` bf16 slab per layer, which is then a
    plain **RowMajor** B with `ldb = Dp`. Priced, derived: 4 kv-heads × 4096 ×
    256 × 2 B read + written = 8.39 MB/layer → **0.23 ms/chunk at 590 GB/s**,
    plus **33,554,432 B** of scratch at max_len 16384. This fallback is cheap
    precisely because there are only **4** kv-heads, not 24 - state that.
- [ ] **Step 4: Extract 6c's alias chain, then write the batched launcher.**
  `src/sycl/xe_gemm_config.h` is a **pure move** of `gemm.cc`'s anonymous-
  namespace alias chain (`ElementInputA` … `Gemm`, `StrideA` … `StrideD`) into a
  header, plus a second chain `GemmT` identical except
  `using LayoutB = cutlass::layout::ColumnMajor;`. `gemm.cc` then includes it
  and loses its local copy - **behaviour-preserving, and re-proved by
  `gemm_test` staying green with bitwise-identical output** (capture the three
  shapes' outputs before the move, compare after; Step 7 does the same for the
  forward). Then:
  ```cpp
  // src/sycl/gemm_batched.cc - the batched entry. Same instantiated
  // configuration as gemm_bf16 (plan 6c Task 1): 256x256x32 work-group tile,
  // XE_DPAS_TT<8,float,bfloat16_t>, 8x4x1 subgroups, MainloopXeL1Staged<2>,
  // IntelXeGeneric epilogue with ElementC = void, PersistentScheduler named
  // explicitly. The ONLY additions are (a) L in the problem shape and the third
  // stride mode, (b) runtime row pitches, (c) a ColumnMajor-B chain for transB.
  #include "sycl/xe_gemm_config.h"
  #include "runtime/prefill/context.h"
  #include "runtime/prefill/context_sycl.h"
  #include "runtime/prefill/gemm.h"

  namespace runtime::prefill {
  namespace {
  using namespace cute;
  void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("runtime::prefill::gemm_bf16_batched: " + what);
  }
  // The Xe 2D block copies move 128 bits at a time: 8 bf16 or 4 fp32
  // (xe_mma.hpp:142-169, cutlass/detail/layout.hpp:417-423), and the base of a
  // block load must be 64-byte aligned. Checking here turns can_implement's
  // opaque kErrorInvalidProblem into a sentence that names the operand.
  void check_operand(const void* p, size_t ld, const char* who) {
    require(reinterpret_cast<uintptr_t>(p) % 64 == 0,
            std::string(who) + " base is not 64-byte aligned");
    require(ld % 8 == 0, std::string(who) + " row pitch " + std::to_string(ld) +
                             " is not a multiple of 8 elements");
  }

  template <class G>
  void launch(Context& cx, const GemmBatch& b, const uint16_t* A, const uint16_t* B,
              float* C, typename G::GemmKernel::StrideB sB) {
    using Gk = typename G::GemmKernel;
    const int M = int(b.M), K = int(b.K), N = int(b.N), L = int(b.L);
    const auto sA = cute::make_stride(int64_t(b.lda), cute::Int<1>{}, int64_t(b.strideA));
    const auto sC = cute::make_stride(int64_t(b.ldc), cute::Int<1>{}, int64_t(b.strideC));
    cutlass::KernelHardwareInfo hw_info{};   // sm_count is unused by the static
                                             // Xe scheduler (static_tile_scheduler.hpp:103)
    typename Gk::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {reinterpret_cast<const cute::bfloat16_t*>(A), sA,
         reinterpret_cast<const cute::bfloat16_t*>(B), sB},
        {{1.0f, 0.0f}, nullptr, sC, C, sC},   // ElementC = void: D = A*B outright
        hw_info};
    require(G::Gemm::can_implement(args) == cutlass::Status::kSuccess,
            "cannot implement M=" + std::to_string(M) + " K=" + std::to_string(K) +
                " N=" + std::to_string(N) + " L=" + std::to_string(L));
    require(G::Gemm::get_workspace_size(args) == 0,
            "this configuration now wants a workspace; the wrapper passes none");
    sycl::queue& q = runtime::prefill::sycl_queue(cx);
    typename G::Gemm op;
    require(op.initialize(args, nullptr, &q) == cutlass::Status::kSuccess, "initialize failed");
    require(op.run(&q) == cutlass::Status::kSuccess, "run failed");
    // Asynchronous by contract: the caller syncs (Context::wait()).
  }
  }  // namespace

  void gemm_bf16_batched(Context& cx, GemmBatch b, const uint16_t* A, const uint16_t* B,
                         float* C, bool transB) {
    require(b.M && b.K && b.N && b.L, "M, K, N and L must all be non-zero");
    require(b.K % 8 == 0, "K must be a multiple of 8 bf16 elements, got " + std::to_string(b.K));
    require(b.N % 8 == 0, "N must be a multiple of 8 bf16 elements, got " + std::to_string(b.N));
    check_operand(A, b.lda, "A");
    check_operand(B, b.ldb, "B");
    check_operand(C, b.ldc, "C");
    if (transB) {
      // B is [N][K] row-major with row pitch ldb -> ColumnMajor from the GEMM's
      // view. StrideB is over (N, K, L): (ldb, 1, strideB).
      launch<XeGemmT>(cx, b, A, B, C,
                      cute::make_stride(int64_t(b.ldb), cute::Int<1>{}, int64_t(b.strideB)));
    } else {
      // B is [K][N] row-major with row pitch ldb. StrideB over (N, K, L) is
      // (1, ldb, strideB) -- N first. This mode order is interfaces.md's
      // "easiest bug here"; Step 5's non-packed case is what catches it.
      launch<XeGemm>(cx, b, A, B, C,
                     cute::make_stride(cute::Int<1>{}, int64_t(b.ldb), int64_t(b.strideB)));
    }
  }
  }  // namespace runtime::prefill
  ```
- [ ] **Step 5: Prove there are still no atomics, for the batched shape.**
  The `L > 1` grid is `ceil(M/256) × ceil(N/256) × L` - the batch is a third
  *grid* dimension, one work-group per output tile, the whole K loop inside it
  (`xe_gemm.hpp:185-193` → `static_tile_scheduler.hpp:260-268`). Re-grep the
  instantiated path for `atomic`, `SplitK`, `split_kv` and any cross-work-group
  reduction at the pin, paste the evidence, and assert
  `get_workspace_size == 0` in the wrapper (already in Step 4's code). This is
  spec §6.4 evidence and it is re-taken here rather than inherited, because the
  batch mode is a code path 6c did not exercise.
- [ ] **Step 6: `tests/prefill/gemm_batched_test.cc` - the cases.** Inputs are
  `random_bf16` in `[-1, 1]` from `tests/kernels/gemv_ref.h`; the reference
  accumulates in `double` over exactly-widened bf16 inputs
  (`common::bf16_to_f32` is exact), so the bar covers only the device's fp32
  accumulation. Bar and its derivation are **6c Task 1 Step 5's, quoted**:
  `|C_dev − C_ref| ≤ 64 · 2^-24 · Σ_k |A·B| + 2^-24`, with the measured max
  ratio printed at every case so the margin is visible.
  1. **Correctness, packed, at the two attention shapes** - `(M,K,N) =
     (2048,256,4096)` and `(2048,4096,256)`, each at `L ∈ {1, 4, 24}`, sampling
     4096 `(l,m,n)` triples from `std::mt19937(0x6e6d31u)`.
  2. **Batch independence:** batch entry `l`'s output must be **bitwise
     identical** to the same GEMM run alone at `L = 1`. A failure means the
     batch mode changes the reduction and the determinism story needs re-doing.
  3. **`strideB = 0`:** run `L = 6` with `strideB = 0` and require every batch
     entry's output bitwise identical to each other and to the `L = 1` run.
  4. **Non-packed pitches - the stride-plumbing test.** Run `(2048,4096,256)`
     with `ldb = 1024` against a B laid out `[K][1024]` whose columns
     `[256·j, 256·j+256)` hold the packed B for `j = 1`; require the result
     **bitwise identical** to the packed `ldb = 256` run. Repeat for
     `lda = 6144` (A laid out `[M][6144]`, head 1's columns live) and for
     `ldc = 4096` on the QKᵀ shape. This is the case that catches an `(N,K,L)`
     vs `(K,N,L)` mode swap, which is silent otherwise.
  5. **`transB`:** build B as `[N][K]` row-major with `ldb = 1024` and require
     the result **bitwise identical** to the packed `[K][N]` run of the same
     logical matrix. *(Skipped, with the skip printed and recorded, if Step 3
     ruled transB unavailable.)*
  6. **The M-stacking identity:** `gemm_bf16_batched({M=2048,…,L=6,
     strideA=2048·lda, strideB=0, strideC=2048·ldc})` must be **bitwise
     identical** to `gemm_bf16_batched({M=12288,…,L=1})` over the same buffers.
     This is the identity Task 4 relies on and it holds because 2048 is a
     multiple of the 256-row M-tile.
  7. **Determinism (spec §6.4):** every case runs twice into two distinct
     output allocations and `memcmp`s the full fp32 result.
  8. **The timed rows** for P-1, P-2 and P-3: 8 replays, discard the first 3,
     median of the final 5, one discarded warm-up, `ZE_AFFINITY_MASK=1`,
     printed as TFLOP/s with the grid `Gemm::get_grid_shape(args)` beside each.
     Labelled **iterate grade, not a bench row** - the test is not the P2
     harness.
- [ ] **Step 7: Make `gemm_bf16` a forward.** Once the batched entry is green,
  `gemm_bf16(cx, d, A, B, C)` becomes
  `gemm_bf16_batched(cx, {d.M, d.K, d.N, 1, d.K, d.N, d.N, 0, 0, 0}, A, B, C, false)`.
  Re-run 6c's `gemm_test` and require its three shapes' outputs **bitwise
  identical to the pre-change capture** from Step 4. One code path, one
  determinism argument, one place a `sycl-tla` bump can change behaviour.
- [ ] **Step 8: Run it, score the pre-registration.**
  `ZE_AFFINITY_MASK=1 tools/box.sh test 'gemm_batched|gemm_test'`, then the full
  suite. In the task report write P-1, P-2, P-3 **predicted and measured side by
  side with the ratio**, say plainly which missed, and - if P-2 or P-3 missed
  its stop bar - re-derive Task 4's price from the measured rates before Task 2
  starts. Do not start a tile sweep: the configuration is Intel's and spec §3.2
  inherits it.
- [ ] **Step 9: Commit** -
  `feat(prefill): gemm_bf16_batched - batch, runtime pitches, transB on the inherited Xe GEMM`

### Task 2: `attn_prep_chunk` and `attn_gate_chunk` - the two ends, carried over

*Substance carried unchanged from the retired plan's Tasks 1 and 2. Three
things move, each for a stated reason: the entry points are renamed to
`pf_attn_prep_chunk` / `pf_attn_gate_chunk` because plan 6b already ships a
different kernel named `pf_attn_prep` and an ocloc target of that name (6b Task
5 Step 5) - a real collision, found by reading 6b; `qkv` is fp32 at S = 1
(request A19); and the gate is in this task rather than its own because Task 5
deletes `attn_reduce` from the prefill path and the gate must land with it.*

**Files:**
- Create: `src/kernels/prefill/pf_attn_chunk.cl`, `src/runtime/prefill/attn.h`,
  `src/runtime/prefill/attn.cc`
- Modify: `src/kernels/prefill/CMakeLists.txt`, `src/kernels/prefill/pf_kernels.h`,
  `src/runtime/CMakeLists.txt`, `tests/CMakeLists.txt`
- Test: `tests/prefill/attn_prep_chunk_test.cc` (new)

**Interfaces:**
- **Consumes:** `Context::kernel` / `Context::launch`; the fused qkv column map
  (`attn.cl:32-39`); the loader's RoPE table fp32 `[max_len][2][32]` and the FA
  small block (`loader/small_layout.h`, `kFaOffQNorm`/`kFaOffKNorm`, which 6b
  passes as `QNORM_OFF=0 KNORM_OFF=256`); the KV cache layout `[pos][4][256]`;
  `attn_ref::prep`; 6b's already-green `pf_attn_prep` as a second oracle.
- **Produces:** `runtime::prefill::attn_prep_chunk` and
  `runtime::prefill::attn_gate_chunk` with the signatures above, and the device
  entry points `pf_attn_prep_chunk`, `pf_attn_gate_chunk`, `pf_kv_zero_pad`.

- [ ] **Step 1: Failing test first.** `tests/prefill/attn_prep_chunk_test.cc`,
  with the cases of Steps 6-8. `tools/box.sh test attn_prep_chunk` - no such
  target. **Paste it.**
- [ ] **Step 2: Write `pf_attn_prep_chunk`.** Grid **(28, C, 1)**, work-group
  256 - one work-group per (head, chunk position), work-item `i` owning dim `i`,
  exactly `attn_prep`'s shape (`attn.cl:314-326`). Dynamic dispatch sizes the
  grid at `C` itself, so there is **no `n_active` clamp and no device-side
  mask**: the `if (m >= n_act) return;` of `attn.cl:343` has no counterpart.
  Groups 0..23 are q-heads, 24..27 are kv-heads. The chain, op for op:
  ```c
  const uint wg = get_group_id(0), c = get_group_id(1), i = get_local_id(0);
  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  __global const float* restrict nw = fa_small + (is_q ? QNORM_OFF : KNORM_OFF);
  const size_t base = is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD;
  __global const float* restrict row = qkv + (size_t)c * QKV_N;   // S = 1 (6b R1)
  const float xf = bf16f(rne_bf16(row[base + i]));   // attn.cl:350, character for character
  red[i] = xf * xf;                                  // plain multiply, no fma
  // ... the same 256 -> 1 pairwise tree, stride = 128, 64, ..., 1, barrier per step
  const float rstd = 1.0f / sqrt(red[0] / (float)HD + 1e-6f);   // never rsqrt
  nrm[i] = bf16f(rne_bf16(xf * rstd * nw[i]));
  // ... the same partial RoPE over dims 0..63, pairs (i, i+32), on nrm,
  //     cs = rope + (size_t)(pos + c) * 2 * ROT_HALF
  ```
- [ ] **Step 3: The stores.** q-head: `q_out[((size_t)c * Q_HEADS + h) * HD + i]
  = rne_bf16(outv)` - bf16, ruling A9, the one deliberate change. kv-head:
  `kv_k[((size_t)(pos + c) * KV_HEADS + h) * HD + i] = rne_bf16(outv)` and
  `kv_v[same slot] = rne_bf16(row[V_OFF + (size_t)h * HD + i])` - v is never
  normed, never roped. **No gate store:** the gate columns stay in `qkv` and
  `pf_attn_gate_chunk` reads them there, which is what lets the declared
  signature have no gate output and what deletes 6b's `attn_gate` buffer.
- [ ] **Step 4: Write `pf_kv_zero_pad`.** Grid `(Dp − D, 1, 1)`, work-group
  **1024** (4 kv-heads × 256 dims per position):
  ```c
  __attribute__((reqd_work_group_size(1024, 1, 1)))
  __kernel void pf_kv_zero_pad(__global ushort* restrict kv_k,
                               __global ushort* restrict kv_v, uint from) {
    const size_t slot = (size_t)(from + get_group_id(0)) * KV_HEADS * HD + get_local_id(0);
    kv_k[slot] = (ushort)0;
    kv_v[slot] = (ushort)0;
  }
  ```
  At most **7 work-groups**, launched only when `(pos + C) % 8 != 0`. It exists
  so PV's `0 · v` over the `[D, Dp)` pad is `0 · 0` and never `0 · NaN` - the
  same hazard `attn.cl:459-475` closes for decode with a load predicate, closed
  here by making the data real.
- [ ] **Step 5: Write `pf_attn_gate_chunk`.** Grid **(24, C, 1)**, work-group
  256, work-item `d` owning dim `d` - `attn_reduce`'s grid with `M` replaced by
  `C`:
  ```c
  const uint h = get_group_id(0), c = get_group_id(1), d = get_local_id(0);
  const float g = bf16f(rne_bf16(qkv[(size_t)c * QKV_N + (size_t)h * 512 + HD + d]));
  const float o = bf16f(attn[((size_t)c * Q_HEADS + h) * HD + d]);
  x_out[(size_t)c * OUT_N + (size_t)h * HD + d] = rne_bf16(o * sigmoid_f32(g));
  ```
  This is `attn.cl:633` with one rounding removed and nothing else: decode
  computes `rne_bf16(bf16f(rne_bf16(acc/sm)) * sigmoid_f32(gate))`, and
  `attn_chunk` has **already** performed the inner `rne_bf16(acc/sm)` when it
  wrote its bf16 output, so `bf16f(attn[…])` is decode's
  `bf16f(rne_bf16(acc/sm))` verbatim. `sigmoid_f32` is plain `exp`, never
  `native_exp`, and it is the **last** op - the discipline `attn.cl:302-305`
  and `prep.cl:110-114` state. Note `[C][24][256]` and `[C][6144]` are the same
  index, so this kernel is index-preserving; it is nonetheless **not** run in
  place, because `restrict` on both pointers would be a lie.
- [ ] **Step 6: Write `pf_k_transpose` - only if Task 1 Step 3 ruled transB
  unavailable.** Grid `(Dp/8, 4, 1)`, work-group 256; work-group `(t, j)` reads
  the 8 cache rows `[8t, 8t+8)` of kv-head `j` and writes them transposed into
  `kt[j][d][p]`, `ldkt = Dp`. Eight rows × 256 dims staged through 4 KB of SLM
  so both the read (`kv_k[p*1024 + j*256 + d]`, 512 B contiguous per row) and
  the write (`kt[(j*256 + d)*Dp + p]`, 8 contiguous ushorts per dim) are
  coalesced. Traffic 8.39 MB/layer → **0.23 ms/chunk (derived)**.
- [ ] **Step 7: Build rules.** In `src/kernels/prefill/CMakeLists.txt` (6b's,
  which is `add_subdirectory`d after `src/kernels/CMakeLists.txt:142` sets
  `CTRL_DEFINES`):
  ```cmake
  add_ocloc_kernel(pf_attn_chunk SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_attn_chunk.cl
                   DEFINES QNORM_OFF=0 KNORM_OFF=256)
  ```
  **No `M`, no `MAXLEN`, no `ATTN_BLOCK`, no `CTRL_*`** - `pos` and `C` are
  runtime arguments and this file reads no `Control`. One binary, four entry
  points (ocloc emits one binary per `add_ocloc_kernel` call and
  `l0::Kernel(mod, "…")` picks the entry, the arrangement `attn.cl` already
  uses); `kernels::pf_attn_chunk_variant()` returns the constant
  `"pf_attn_chunk"`.
- [ ] **Step 8: The host shims** in `src/runtime/prefill/attn.cc`.
  `attn_prep_chunk`: `require(pos + C <= max_len)` - the caller's rule, asserted
  here anyway with a throw, because prefill's dispatch is dynamic and a bad `C`
  would write past the cache; then
  `cx.launch(cx.kernel("pf_attn_chunk", "pf_attn_prep_chunk", 256), 28, C, 1,
  {PtrArg(qkv), PtrArg(fa_small), PtrArg(rope), PtrArg(q_out), PtrArg(kv_k),
  PtrArg(kv_v), arg_val(pos), arg_val(C)})`. `attn_gate_chunk`: one launch at
  grid `(24, C, 1)`. Both take `max_len` from the caller's `require`, not from a
  compiled constant.
- [ ] **Step 9: The bit-identity cases.** Build one `Inputs` the way
  `attn_test.cc:446-463` does (xorshift, state the seed constant), then:
  1. `attn_ref::prep(pos, 1, 1, partials, fa_small, rope, q_ref32, g_ref,
     k_ref, v_ref)` with an `S = 2` partials buffer whose slice 1 is `+0.0f`, so
     `attn_ref`'s `qkv_sum` reduces to slice 0 exactly (6b Task 5 Step 1's
     construction and its `-0.0` caveat);
  2. `pf_attn_prep_chunk` at `C = 1` on slice 0 as fp32;
  3. assert `kv_k` and `kv_v` **bit-identical** to `k_ref`/`v_ref` over the
     whole cache (`require_bits16`, the bar `attn_test.cc:603-604` holds), and
     `q_out[i] == common::f32_to_bf16(q_ref32[i])` **bit-identical** over all
     6144 words. These are bit bars, not tolerances: nothing in this chain uses
     `exp`, and `1.0f/sqrt` is correctly rounded by the build flag
     (`cmake/ocloc.cmake:12-22`);
  4. **the second oracle, free:** run 6b's `pf_attn_prep` on the same input and
     require `kv_k`, `kv_v` bit-identical and
     `q_out[i] == f32_to_bf16(attn_q_6b[i])` - a direct regression against the
     kernel this one replaces. *(Skipped with a printed note if 6b's kernel is
     already deleted at execution time.)*
- [ ] **Step 10: The multi-position cases.** Repeat at `C ∈ {2, 64, 1024, 2048}`
  and `pos ∈ {0, 4095}` against a host loop calling `attn_ref::prep` once per
  chunk position with `M = 1, n_act = 1, pos = pos + c` - proving the cache is
  written at `[pos+c]` for every `c`, that no slot outside `[pos, pos+C)` moves
  (seed the whole cache and require the complement bit-identical to the seed),
  and that the RoPE table is indexed at the **absolute** position. Add the
  `pf_kv_zero_pad` case: at `pos + C = 4093`, require rows 4093..4095 all-zero
  and row 4096 still the seed.
- [ ] **Step 11: The gate case.** Run `attn_ref::reduce` on a synthetic
  `attn_part` twice: once with the real gate to get `out_ref`, and once with the
  gate array filled with `+INFINITY` to get the **pre-gate** tensor
  (`sigmoid_f32(+INF) = 1/(1+exp(-INF)) = 1.0f` exactly, so `reduce` degenerates
  to `rne(acc/sm)` - the identity that lets the same reference serve both sides
  without editing `attn_ref.h`). Feed the pre-gate tensor to
  `pf_attn_gate_chunk` with the real gate columns and require the result
  **bit-identical** to `out_ref`. Bit, not tolerance: if it fails at 1-2 bf16
  ulp on a handful of words it is `exp`'s 3-ulp slack, and the bar drops to
  `require_ulp(…, bar = 1, …)` **with the count printed** - record which,
  because it is the same question `attn_test.cc:26-52` answers for decode.
- [ ] **Step 12: Run.** `tools/box.sh test attn_prep_chunk`; then
  `tools/box.sh test` (full suite - nothing existing may move), then
  `tools/box.sh test 'replay_determinism_test|profile_capture_test'` with
  **774 / 19** pasted.
- [ ] **Step 13: Commit** -
  `feat(prefill): attn_prep_chunk + attn_gate_chunk - runtime-C KV write, bit-identical at C=1`

### Task 3: `pf_softmax_causal` - ours, OpenCL C, bandwidth-bound

**Files:**
- Create: `src/kernels/prefill/pf_softmax.cl`
- Modify: `src/kernels/prefill/CMakeLists.txt`,
  `src/kernels/prefill/pf_kernels.h`, `tests/CMakeLists.txt`
- Test: `tests/prefill/softmax_causal_test.cc` (new)

**Interfaces:**
- **Consumes:** `attn.cl:143-153`'s softmax statement and `attn_ref.h:35-50`'s
  identical one; `SCALE = 0.0625f`; `common::bf16_to_f32` / `f32_to_bf16`.
- **Produces:** the device entry `pf_softmax_causal(const float* S, ushort* P,
  float* rowsum, uint pos, uint C, uint Dp)` and a ctest target
  `softmax_causal_test`.

- [ ] **Step 1: Failing test first.** `tests/prefill/softmax_causal_test.cc`
  with Step 5's cases. `tools/box.sh test softmax_causal` - no such target.
  **Paste it.**
- [ ] **Step 2: Fix the grid, and say why it is not the obvious one.** A row is
  `(head-in-tile l, query c)` and its softmax needs the **whole** row: the max,
  then the sum. Splitting a row across work-groups would need either an fp
  atomic (**forbidden**, spec §6.4) or a second kernel and a partials buffer.
  It is unnecessary: at `C = 2048` and `Lq = 6` there are **12,288 rows**, i.e.
  12,288 work-groups over 32 Xe-cores - 384 per core, an order of magnitude past
  saturation. **Grid `(C, Lq, 1)`, work-group 256**, one row per work-group,
  each work-item striding the row by 256.
  **The row is deliberately NOT staged in SLM.** At `Dp = 16384` a row is 64 KB
  of the 128 KB budget (`docs/01-hardware.md:16`), which would halve occupancy
  for a re-read that L2 already serves: the concurrent working set is
  ~32 cores × a few resident groups × 64 KB ≈ 8 MB against a **24 MB L2**
  (derived). Pass 2's read of `S` is therefore expected to hit L2 and cost no
  DRAM traffic. **If Task 6 measures this kernel above its traffic model, the
  SLM-staged variant is the named lever** - one `__local float row[]` sized from
  a compile-time cap, taken only for `Dp × 4 ≤ 65536`.
- [ ] **Step 3: Write `pf_softmax_causal`.** The whole kernel, and the ordering
  statement goes in its header comment beside decode's:
  ```c
  #define WG 256
  #define SCALE 0.0625f   /* 1/sqrt(256) - attn.cl:228, applied at attn.cl:502's point */

  __attribute__((reqd_work_group_size(WG, 1, 1)))
  __kernel void pf_softmax_causal(__global const float* restrict S,   /* fp32 [Lq][C][Dp] */
                                  __global ushort* restrict P,        /* bf16 [Lq][C][Dp] */
                                  __global float* restrict rowsum,    /* fp32 [Lq][C]     */
                                  uint pos, uint C, uint Dp) {
    const uint c = get_group_id(0), l = get_group_id(1), i = get_local_id(0);
    const size_t row = (size_t)l * C + c;
    __global const float* restrict s = S + row * Dp;
    __global ushort* restrict p = P + row * Dp;
    const uint bound = pos + c;          /* causal: key k contributes iff k <= pos + c */
    __local float red[WG];

    /* Pass 1 - the row max over valid keys. The scale is applied to the score
       BEFORE the max, which is attn.cl:502's order exactly. A masked key is
       -INFINITY, never a skipped max: k = 0 <= bound always, so `mx` is finite
       for every row and exp(mx - mx) is never the NaN attn.cl:516-519 guards. */
    float m = -INFINITY;
    for (uint k = i; k < Dp; k += WG)
      m = fmax(m, (k <= bound) ? s[k] * SCALE : -INFINITY);
    red[i] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG / 2; stride > 0; stride >>= 1) {   /* fixed pairwise tree */
      if (i < stride) red[i] = fmax(red[i], red[i + stride]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mx = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Pass 2 - the weights. Plain `exp`, never native_exp (attn.cl:302-305).
       A masked key writes bf16 +0.0 EXACTLY, which is what lets PV run the full
       [0, Dp) rectangle: 0 * v is 0 for every finite v, and the pad [D, Dp) is
       zeroed by pf_kv_zero_pad so it is 0 * 0.
       `rowsum` accumulates the SAME rounded words the GEMM will read, not the
       fp32 exponentials, so the weights PV sees sum to `rowsum` in exact
       arithmetic and the normalisation carries no bias. */
    float acc = 0.0f;
    for (uint k = i; k < Dp; k += WG) {
      ushort w = (ushort)0;
      if (k <= bound) w = rne_bf16(exp(s[k] * SCALE - mx));
      p[k] = w;
      acc += bf16f(w);                   /* ascending k within a lane */
    }
    red[i] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = WG / 2; stride > 0; stride >>= 1) {
      if (i < stride) red[i] += red[i + stride];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i == 0) rowsum[row] = red[0];
  }
  ```
  **The order, stated for the record (Task 6 copies it into docs/12):** lane `i`
  reduces `k = i, i+256, i+512, …` ascending; the 256 lane partials collapse
  with a fixed pairwise tree, `stride = 128 … 1`. It is deterministic and
  data-independent, and it is **not** decode's order - decode's is an online
  16-position wave with rescaling. The difference is the "same operation, two
  associations" case §"Numerics" item 3 names; **the golden gate is the
  arbiter**, and the two-pass form performs strictly fewer roundings, so this
  term is expected to reduce error rather than add it.
- [ ] **Step 4: Build rule and `zeinfo`.**
  `add_ocloc_kernel(pf_softmax SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/pf_softmax.cl)`
  - one binary, no `-D`. Then
  ```
  tools/box.sh run 'ocloc compile -file src/kernels/prefill/pf_softmax.cl \
    -device bmg-g31 -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt" \
    -output pfs -out_dir /tmp && cat /tmp/pfs.bin.zeinfo'
  ```
  and record `simd`, `slm_size`, `grf_count`, `barrier_count`, `private_size`
  and **`spill_mem_size`** - the same six columns docs/12's kernel tables carry.
  `slm_size` must be 1024 B (the one `red[256]`); anything larger means a
  variant crept in. Any `spill_mem_size` in a kernel this simple is a finding.
- [ ] **Step 5: `tests/prefill/softmax_causal_test.cc` - the cases.** A CPU
  reference in the test file (not in `attn_ref.h` - this kernel has no decode
  twin) that performs **the same two passes in the same order**, so the bar is
  the device's `exp` against the host's and nothing else.
  1. **The bar:** `rowsum` relative ≤ **1e-6** (an fp32 sum of ≤16384 terms with
     identical association on both sides; only `exp`'s 3 ulp differs), and `P`
     **bit-identical** except where the host and device `exp` land on opposite
     sides of a bf16 rounding boundary - print the differing count and require
     **≤ 1 bf16 ulp on every word**, with the count of words at 1 ulp reported.
     A word at 2 ulp is a finding, not a pass.
  2. **The mask, asserted as bits, not as a tolerance:** every `k > pos + c`
     word of `P` must be **exactly `0x0000`** (not merely small). This is the
     property PV's full-rectangle safety rests on, so it is checked with
     `memcmp` against a zero-filled tail, per row.
  3. **The diagonal:** `pos = 0`, `C = 64`, `Dp = 64` - row `c` has exactly
     `c+1` valid keys, so the count of non-zero words per row must be exactly
     `c+1` and `rowsum` must be ≥ the largest weight.
  4. **The fully-masked-prefix edge:** `pos = 0`, `c = 0` - exactly **one**
     valid key. The reference gives `mx = s[0]·SCALE`, `P[0] = rne_bf16(1.0f) =
     0x3F80`, `rowsum = 1.0f` exactly, and every other word `0x0000`. Assert
     those three literals; this is the row where an "online softmax skip the
     wave whole" bug (`attn.cl:516-519`'s case) would surface as a NaN.
  5. **Extreme scores:** fill one row's valid prefix with `+80.0f / SCALE` and
     another with `−80.0f / SCALE`, and one row with a single `+INFINITY`
     score. Require no NaN and no Inf anywhere in `P` or `rowsum` - the
     max-subtraction is what guarantees it and this is the test that proves it.
  6. **Sizes:** `(pos, C, Dp) ∈ {(0,1,8), (0,64,64), (0,2048,2048),
     (2048,2048,4096), (14336,2048,16384)}` - the last is the deepest chunk the
     engine can produce at `max_len = 16384`.
  7. **Determinism:** two runs into distinct outputs, `memcmp` on `P` and
     `rowsum`.
  8. **The timed row:** at `(2048, 2048, 4096)` with `Lq = 6`, 8 replays / drop
     3 / median, and the **derived traffic model printed beside it**: read
     `0.7501 × 6 × 2048 × 4096 × 4 = 151.0 MB` + write `6 × 2048 × 4096 × 2 =
     100.7 MB` = **251.7 MB → 0.427 ms at 590 GB/s**, ×4 groups ×16 layers =
     **27.3 ms/chunk**. Conservative bound if pass 2 misses L2: **32.8
     ms/chunk**. Pre-register **27-33 ms/chunk** here, before running it.
- [ ] **Step 6: Run.** `ZE_AFFINITY_MASK=1 tools/box.sh test softmax_causal`,
  then the full suite.
- [ ] **Step 7: Commit** -
  `feat(kernels): pf_softmax_causal - two-pass row softmax, causal by absolute position`

### Task 4: `attn_chunk` - the composition

**Files:**
- Modify: `src/kernels/prefill/pf_attn_chunk.cl` (add `pf_attn_scale_pack`),
  `src/runtime/prefill/attn.{h,cc}`, `src/runtime/CMakeLists.txt`,
  `tests/CMakeLists.txt`
- Modify: `tests/kernels/attn_ref.h` - **additive only**: `attn_ref::chunk_ref`
- Test: `tests/prefill/attn_chunk_test.cc` (new)

**Interfaces:**
- **Consumes:** Task 1's `gemm_bf16_batched`; Task 2's `attn_prep_chunk` and
  `pf_kv_zero_pad`; Task 3's `pf_softmax_causal`; `PrefillScratch`'s four new
  buffers (Task 5 allocates them; until then the test allocates its own);
  `attn_ref::prep/decode/reduce` and every ordering statement in
  `attn_ref.h:22-60`; `attn_test.cc`'s `compare_f32` / `require_ulp` /
  `require_bits16` / `bf16_key`, **copied into the new test** rather than
  exported, so `attn_test.cc` is not edited.
- **Produces:** `runtime::prefill::attn_chunk` and `pf_attn_scale_pack`.

- [ ] **Step 1: Failing test first.** `tests/prefill/attn_chunk_test.cc` with
  Steps 6-9's cases. `tools/box.sh test attn_chunk` - no such target. **Paste.**
- [ ] **Step 2: Write `pf_attn_scale_pack`.** Grid `(C, Lh, 1)`, work-group 256:
  ```c
  __attribute__((reqd_work_group_size(HD, 1, 1)))
  __kernel void pf_attn_scale_pack(__global const float* restrict O,      /* [Lh][C][256] */
                                   __global const float* restrict rowsum, /* [Lh][C]      */
                                   __global ushort* restrict out,         /* [C][24][256] */
                                   uint C, uint h0) {
    const uint c = get_group_id(0), lh = get_group_id(1), d = get_local_id(0);
    const float sm = rowsum[(size_t)lh * C + c];
    const float o  = O[((size_t)lh * C + c) * HD + d];
    /* attn.cl:631's `acc / sm` under -cl-fp32-correctly-rounded-divide-sqrt,
       then attn.cl:633's inner rne_bf16. attn_gate_chunk supplies the rest. */
    out[((size_t)c * Q_HEADS + (h0 + lh)) * HD + d] = rne_bf16(o / sm);
  }
  ```
- [ ] **Step 3: Write `attn_chunk`.** The whole composition, with every stride
  spelled out - the strides are where this task can be silently wrong:
  ```cpp
  void attn_chunk(Context& cx, PrefillScratch& s, uint32_t layer, uint32_t pos,
                  uint32_t C, const uint16_t* q, const uint16_t* kv_k,
                  const uint16_t* kv_v, uint16_t* out) {
    require(C > 0 && C <= PrefillScratch::kC, "C out of range");
    require(size_t(pos) + C <= s.max_len, "pos + C exceeds max_len");
    const uint32_t D  = pos + C;
    const uint32_t Dp = (D + 7u) & ~7u;                 // the GEMM's 8-element granularity
    require(Dp <= s.max_len, "the 8-element pad does not fit in the cache");
    // Heads per QK^T/softmax launch: the GQA group, or fewer if pf_s cannot
    // hold six at this depth. See "The head tile Lh" for the arithmetic.
    const uint32_t Lq = std::clamp<uint32_t>(
        uint32_t(s.pf_s_bytes / (size_t(C) * Dp * 4)), 1u, kGqa);
    require(kGqa % Lq == 0, "Lq must divide the GQA group");

    if (D != Dp)                                        // <= 7 work-groups, once per layer
      cx.launch(cx.kernel("pf_attn_chunk", "pf_kv_zero_pad", 1024), Dp - D, 1, 1,
                {PtrArg(kv_k), PtrArg(kv_v), arg_val(D)});

    float*    S  = s.pf_s.as<float>();
    uint16_t* P  = s.pf_p.as<uint16_t>();
    float*    O  = s.pf_o.as<float>();
    float*    RS = s.pf_rowsum.as<float>();

    for (uint32_t j = 0; j < kKvHeads; ++j) {           // one kv-head group at a time
      for (uint32_t g = 0; g < kGqa; g += Lq) {
        const uint32_t h0 = j * kGqa + g;               // first q-head of this launch
        // --- 1. S = Q * K^T -----------------------------------------------
        // A: q_out is [C][24][256], so a head's rows have pitch 6144 and
        //    consecutive heads are 256 elements apart. No relayout.
        // B: the K cache is [pos][4][256] -> for kv-head j it IS [Dp][256]
        //    row-major with pitch 1024, i.e. B given as [N][K]: transB.
        //    strideB = 0 -- all Lq q-heads read the same kv-head.
        gemm_bf16_batched(cx,
            {C, kHeadDim, Dp, Lq,
             /*lda*/ kQHeads * kHeadDim, /*ldb*/ kKvHeads * kHeadDim, /*ldc*/ Dp,
             /*strideA*/ kHeadDim, /*strideB*/ 0, /*strideC*/ size_t(C) * Dp},
            q + size_t(h0) * kHeadDim, kv_k + size_t(j) * kHeadDim, S, /*transB*/ true);
        cx.wait();                                      // SYCL -> L0 (P1: 8.569 us)
        // --- 2. P = softmax(S), rowsum ------------------------------------
        cx.launch(cx.kernel("pf_softmax", "pf_softmax_causal", 256), C, Lq, 1,
                  {PtrArg(S), PtrArg(P), PtrArg(RS + size_t(h0) * C),
                   arg_val(pos), arg_val(C), arg_val(Dp)});
        cx.wait();                                      // L0 -> SYCL
        // --- 3. O = P * V --------------------------------------------------
        // A: P is [Lq][C][Dp], packed. B: the V cache, [Dp][256] with pitch
        //    1024, K-major already -> no transpose. strideB = 0 again.
        gemm_bf16_batched(cx,
            {C, Dp, kHeadDim, Lq,
             /*lda*/ Dp, /*ldb*/ kKvHeads * kHeadDim, /*ldc*/ kHeadDim,
             /*strideA*/ size_t(C) * Dp, /*strideB*/ 0, /*strideC*/ size_t(C) * kHeadDim},
            P, kv_v + size_t(j) * kHeadDim, O + size_t(h0) * C * kHeadDim, /*transB*/ false);
        cx.wait();                                      // SYCL -> L0
        // --- 4. out = rne_bf16(O / rowsum), scattered back to [C][24][256] --
        cx.launch(cx.kernel("pf_attn_chunk", "pf_attn_scale_pack", kHeadDim), C, Lq, 1,
                  {PtrArg(O + size_t(h0) * C * kHeadDim), PtrArg(RS + size_t(h0) * C),
                   PtrArg(out), arg_val(C), arg_val(h0)});
        cx.wait();                                      // L0 -> SYCL (next group's QK^T)
      }
    }
  }
  ```
  **If Task 1 Step 3 ruled transB unavailable**, step 1's `B` becomes
  `s.pf_kt.as<uint16_t>() + size_t(j) * kHeadDim * Dp` with `ldb = Dp` and
  `transB = false`, and one `pf_k_transpose` launch is added before the `j`
  loop. That is the only difference; write whichever Task 1 ruled and delete the
  other, so there are not two live paths in the tree.
- [ ] **Step 4: State the alignment facts the launcher depends on**, in
  `attn.cc`'s header comment, because Task 1's `check_operand` will throw
  otherwise and the message should already be understood:
  `q + h0*256` is `h0·512 B` from a 4 KB-aligned USM base → 64 B aligned ✓;
  `kv_k + j*256` is `j·512 B` ✓; `S`, `P`, `O` are allocation bases ✓;
  `RS + h0*C` is `h0·C·4 B` with `C ≥ 64` ✓. Pitches: 6144, 1024, `Dp`, 256 -
  all multiples of 8 ✓ (`Dp` by construction). `K = 256` and `N = Dp` or 256,
  both multiples of 8 ✓.
- [ ] **Step 5: `attn_ref::chunk_ref` - additive to `tests/kernels/attn_ref.h`.**
  `attn_ref::decode` writes `attn_part` at `[24][max_len/64][M][258]`; at
  `C = 2048`, `max_len = 16384` that is `24 × 256 × 2048 × 258 × 4 B` =
  **13.0 GB on the host** and is not a test. Add
  ```cpp
  // The decode chain per (q-head, query) with no attn_part: the same wave/block
  // loop as `decode` over blocks 0 … (pos+c)/kBlock, accumulating ONE running
  // (mx, sm, acc[256]) - which is `reduce`'s merge with nb steps performed
  // inline - and finishing with out[(c*24+qh)*256+d] = rne(acc[d]/sm), i.e.
  // `reduce`'s final chain with a +INF gate. It is a COMPOSITION of the two
  // existing loops, not a third transcription: the file's rule ("every block
  // below has a twin in attn.cl and the two must be edited together") is
  // honoured because no new arithmetic appears here.
  inline void chunk_ref(uint32_t pos, uint32_t C, uint32_t max_len,
                        const uint16_t* q_bf16, const uint16_t* kv_k,
                        const uint16_t* kv_v, uint16_t* out);
  ```
  It takes **bf16** `q` and widens it with `f32()` before the score dot - the
  reference must round where the kernel rounds (ruling A9).
  **Host cost**, derived: the largest case is `C = 2048, pos = 4096` →
  `2048 × 24 × 6144 × 256 = 7.73e10` MAC, **estimated 150-300 s
  single-threaded**. Split the independent `(qh, c)` rows across
  `std::thread::hardware_concurrency()` `std::thread`s - legal because rows are
  independent and each row's internal order is untouched, and deterministic
  because each thread writes only its own rows. There is **no OpenMP in this
  repository** (checked) and none is added. If the threaded run still exceeds
  the test budget, check a fixed subset of **256 of the 49,152 `(qh, c)` rows**
  drawn from `std::mt19937(0x6e6d31u)` at that one case, with the subsetting
  printed; the exhaustive sweep stays at `C ≤ 256`.
- [ ] **Step 6: The fixture.** Seed with xorshift (state the constant), fill the
  **whole** `kv_k`/`kv_v` cache - not just the `[0, D)` prefix - so a read past
  the causal bound shows up as noise and not as a convenient zero
  (`attn_test.cc:457-462`), and fill `q` as bf16 in `[-2, 2]`. Upload once; run
  `attn_chunk` twice from identical inputs into distinct outputs.
- [ ] **Step 7: The cases** - `C ∈ {1, 64, 256, 2048}` × `pos ∈ {0, 4096}`,
  eight runs at `max_len = 16384`:
  - `C = 1` is the degenerate chunk and the direct comparison against decode's
    geometry (and the case where `Dp − D = 7`, exercising `pf_kv_zero_pad`);
  - `C = 64` is one M-tile short of the GEMM's 256-row tile - the case that
    proves the GEMM's own M masking, and at `pos = 0` it is the diagonal-only
    case where every masked element is in the trimmed region;
  - `C = 256` is exactly one M-tile;
  - `C = 2048` is the production width, and at `pos = 4096` it is the deepest
    case the pre-registration is quoted at (`D = 6144`).
- [ ] **Step 8: The bars, pre-registered here before the run.** `attn_chunk`'s
  output is **bf16**, so `attn_test.cc:639`'s **relative 1e-3** bar on the fp32
  `attn_part` **does not transfer and must not be used** - `attn_test.cc:54-79`
  records that 1e-3 was never consistent with a bf16 output, because one bf16
  ulp on a gated element reaches **7.8e-3** (7 explicit mantissa bits → one ulp
  is 2⁻⁸…2⁻⁷ of the element). The bars are `attn_out`'s pair, **loosened by
  exactly one ulp with the reason stated in advance**:
  - **(a) relative ≤ 1.2e-2, floored at the tensor RMS**;
  - **(b) ≤ 3 bf16 ulp on every element with `|ref| ≥ rms/8` - (b) is the
    arbiter** (`attn_test.cc:40-52`).
  **Why 3 and not decode's 2.** Decode's 2 is "one ulp per bf16 rounding × two
  roundings, and `exp`'s 3 ulp can push a boundary value one ulp at each". The
  composed path performs **one additional rounding decode does not**: the
  attention weight itself is rounded to bf16 before PV (§"Numerics" item 4).
  The derived contribution of that rounding to the output is `ε_rms · σ_v /
  √D_eff` - sub-ulp typically, at most ~1 ulp on a boundary word. 3 is that
  arithmetic, written down before the measurement. **Print, per case, the worst
  relative, the worst ulp anywhere, the worst gated ulp, and how many of the
  `C × 6144` words differ at all**, in `attn_test.cc:649-656`'s format. A case
  at 3 is a **finding to report**, not a pass to bank silently; a case at 4 is a
  stop.
- [ ] **Step 9: Determinism and the structural assertions** (spec §6.4):
  - the second run's output is **bitwise identical** to the first (`memcmp`),
    every case;
  - the `[D, Dp)` pad rows of `kv_k`/`kv_v` are zero and rows `≥ Dp` are still
    the seeded fill;
  - `rowsum` is finite and `> 0` for every row (a zero would divide by zero and
    the pack would emit NaN - this is the one input the pack does not check);
  - **the M-stacked cross-check**, run once at `C = 256, pos = 0`: recompute
    step 1 and step 3 with `L = 1, M = 6C` over the same buffers and require
    `out` **bitwise identical**. It is Task 1 Step 6's identity exercised
    through the production call site.
- [ ] **Step 10: Register and run.** `tests/CMakeLists.txt` beside `attn_test`,
  inside `if(B70_PREFILL_ENABLED)`, with
  `add_dependencies(attn_chunk_test kernel_pf_attn_chunk kernel_pf_softmax)`
  and `b70_link_prefill(attn_chunk_test)`.
  `ZE_AFFINITY_MASK=1 tools/box.sh test attn_chunk`, then the full suite.
- [ ] **Step 11: Commit** -
  `feat(prefill): attn_chunk composed - QK^T GEMM + causal softmax + PV GEMM, no FMHA`

### Task 5: The swap into `Engine::prefill()` at `kC = 2048`, and the gates

**Files:**
- Modify: `src/runtime/engine.cc` / `src/runtime/prefill/step.cc` (6b's
  `step_chunk` fa-layer branch), `src/runtime/buffers.h` / `buffers.cc`
  (`PrefillScratch`), `src/cli/b70_decode.cc` (the `--pp-chunk` default),
  `src/kernels/prefill/CMakeLists.txt` and `src/kernels/CMakeLists.txt` (the
  retirements), `tests/runtime/buffers_test.cc`,
  `tests/prefill/prefill_gate_test.cc`, `docs/14-golden-gate.md`
- Delete: `src/runtime/prefill/attn_l1.{h,cc}`,
  `src/kernels/prefill/pf_attn_prep.cl`, `tests/prefill/pf_attn_test.cc`'s
  L1-route cases (cases 3-6 of 6b Task 5 Step 1; the two `pf_gated_head` cases
  stay)

**Interfaces:**
- **Consumes:** 6b's `Engine::prefill` walk, `PrefillScratch`, `oracle-out-long`
  and the three prefill tests; the two checkpoints (RTN at
  `$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`; Vishva007 from the HF
  cache - `docs/14-golden-gate.md:787-793`).
- **Produces:** a prefill fa-layer that is exactly three calls, a default chunk
  of `PrefillScratch::kC = 2048`, and spec §6 items 1-4 green.

- [ ] **Step 1: Replace the temporary path.** In 6b's `step_chunk`, the fa-layer
  branch's `for (sub = 0; sub < C; sub += PrefillScratch::kAttnC)` loop -
  including its `cx.wait()` and its host writes to `Control::pos` /
  `n_active` - is deleted whole and replaced by:
  ```cpp
  void* kk = at(kv_k_mem, size_t(fa) * kv_stride);
  void* vv = at(kv_v_mem, size_t(fa) * kv_stride);
  prefill::attn_prep_chunk(cx, l, pos, C, s.partials.as<float>(),
                           m.layer_small[l].gdn.as<float>(), m.rope.as<float>(),
                           s.pf_q.as<uint16_t>(),
                           static_cast<uint16_t*>(kk), static_cast<uint16_t*>(vv));
  prefill::attn_chunk(cx, s, l, pos, C, s.pf_q.as<uint16_t>(),
                      static_cast<const uint16_t*>(kk), static_cast<const uint16_t*>(vv),
                      s.pf_attn.as<uint16_t>());
  prefill::attn_gate_chunk(cx, C, s.partials.as<float>(), s.pf_attn.as<uint16_t>(),
                           s.mixer_out.as<uint16_t>());
  ```
  with `fa = layer / 4` (`Qwen35::is_fa(l) == (l % 4 == 3)`, `capture.cc:110`).
  **The prefill path no longer touches `Control` at all** on FA layers - `pos`
  and `C` are runtime kernel arguments - which removes 6b's per-sub-chunk
  `cx.wait()` and the host-write race it was guarding.
- [ ] **Step 2: `PrefillScratch` - retire three fields, add six, and re-derive
  the total.** Delete `attn_part`, `attn_q`, `attn_gate` and the constant
  `kAttnC`; add `pf_q`, `pf_attn`, `pf_s`, `pf_p`, `pf_o`, `pf_rowsum` (plus
  `pf_kt` if Task 1 Step 3 ruled transB unavailable) with the sizes tabulated in
  *Interfaces: what this plan produces*, and a `size_t pf_s_bytes` member the
  `Lq` rule reads. Confirm `kC == 2048` (ruling A13); if 6b landed 4096, change
  it here. Add `require(bytes <= l0ctx.props().maxMemAllocSize)` for `pf_s` -
  805,306,368 B is the largest single allocation this project has ever made
  (`kv_k` is 536,870,912) and a driver cap must be a named throw, not a
  mysterious `ZE_RESULT_ERROR_UNSUPPORTED_SIZE`.
  Then rewrite `tests/runtime/buffers_test.cc`'s prefill total **with the
  arithmetic rewritten, not patched** (plan 5 Task 6 Step 3's discipline), and
  state the delta in the `PrefillScratch` comment:
  **+1,308,819,456 added, −408,944,640 retired, net +899,874,816 B (+858.20
  MiB)**. Check the new resident total against the 32656 MB the device reports
  (`docs/01-hardware.md:14`) and say the number.
- [ ] **Step 3: Retire ruling A10's `M = 64` exception.** Delete from
  `src/kernels/CMakeLists.txt` the four rows `add_attn_decode(64, 16384)`,
  `add_attn_reduce(64, 16384)`, `add_attn_decode(64, 4096)`,
  `add_attn_reduce(64, 4096)`; delete `src/kernels/prefill/pf_attn_prep.cl` and
  its `add_ocloc_kernel` row; delete `src/runtime/prefill/attn_l1.{h,cc}` and
  their entry in `src/runtime/CMakeLists.txt`. Grep for `kAttnC`, `attn_l1`,
  `attn_part` and the literal `64` in the prefill path; **every one of them is
  retired in this commit, not left as a stale note.** A10 says "Plan 6d retires
  the exception" - this is that step.
- [ ] **Step 4: `--pp-chunk`.** Default resolves to `PrefillScratch::kC = 2048`;
  the `C ≤ 64` clamp 6b installed anywhere (engine, CLI) is deleted; the
  `require(chunk <= kC)` throw stays. Re-check the stdout contract interfaces.md
  fixes: the tg row unchanged, the pp row
  `| b70-decode <sha> | pp | <N> | <ms total> | <t/s> |`, and the human line on
  **stderr**.
- [ ] **Step 5: §6.1 - the golden gate after prefill**, both checkpoints, all
  three prompts, tie-aware semantics unchanged:
  ```
  tools/box.sh run './build/tests/prefill/prefill_gate_test "$PWD/oracle-out" \
    "$PWD/tests/golden/prompts" Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ'
  tools/box.sh run './build/tests/prefill/prefill_gate_test "$PWD/oracle-out-rtn" \
    "$PWD/tests/golden/prompts" $HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64'
  ```
  Determined rows 100% exact; tie rows set-membership; teacher-forced tails.
  **A determined-row flip is STOPPED and surfaced, not tuned around** - and §"Numerics"
  above is the list of five named changes to check it against before any
  hypothesis is invented.
- [ ] **Step 6: §6.3 - self-consistency** (`prefill_consistency_test`,
  assertions unchanged): `prefill(P)` vs `ingest-by-decode(P)`, 64 generated
  tokens identical. Record `gdn_state`, `conv_ring`, the chunk's KV rows and the
  last hidden as max/mean relative difference - **diagnostic, not gate** (spec
  §6.3). Any growth versus 6b's recorded band **is a finding** and is reported
  with the §"Numerics" item that explains it, because L3 is the stage that
  changed the attention arithmetic. Note that the KV rows themselves should be
  **bit-identical** to 6b's: `attn_prep_chunk` performs the same chain, so a
  KV difference here is a bug in Task 2, not a numerics band.
- [ ] **Step 7: §6.2 - the multi-chunk gate at `C = 1024`** on `oracle-out-long`,
  the row this whole task exists for: at C = 1024 over a ≥ 2048-id prompt every
  chunk boundary is crossed in attention-over-cache, so it is the only test that
  exercises `attn_chunk` with a **non-zero `pos`** in anger - Task 4's
  `pos = 4096` cases prove the arithmetic, this proves the wiring. Flip it from
  capped/skipped to enforced; **its assertions do not change**.
- [ ] **Step 8: §6.4 - determinism** (`prefill_determinism_test`, unchanged):
  prefill twice from a reset state → bitwise-identical state buffers and tokens.
- [ ] **Step 9: §6.5 - decode untouched.** `tools/box.sh test` full suite;
  `replay_determinism_test` still asserts **774 / 19** with no edit (paste it).
  Re-measure the decode gate rows once at this plan's final sha, **record
  grade** (idle box, DRM-fd evidence pasted, medians of 3): **32.22 t/s (RTN) /
  29.33 (Vishva)** at `2a7df0b`. Inside day drift (≤ 0.09%) or the difference is
  explained. `-Wall -Wextra -Werror` green; `-cl-denorms-are-zero` absent (the
  CMake fatal proves it).
- [ ] **Step 10: Commit** (two commits, in this order, so a bisect can separate
  the swap from the retirement) -
  `feat(prefill): attention at C=2048 - composed attn_chunk replaces the C<=64 route`, then
  `chore(prefill): retire attn_part, attn_l1 and the M=64 attention binaries (ruling A10)`

### Task 6: Attribution and the records

**Files:**
- Modify: `docs/12-kernels.md` (a prefill-attention mechanism section),
  `docs/15-step-anatomy.md` (the prefill-step anatomy rows),
  `docs/07-open-questions.md` (what this stage answers and what it opens),
  `docs/BENCHMARKS.md` (the prefill rows gain the L3 checkpoint),
  `docs/05-perf-model.md` (the prefill verdict line),
  `docs/probe-prefill-attn-2026-09-04.md` (a closing section: the FMHA path is
  retired, and by what)
- Modify (only if 6b built no attribution): `src/cli/b70_decode.cc`,
  `src/runtime/prefill/context.cc`
- Create: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/l3-attention-memo.md`
  (short path only - a miss)

**Interfaces:**
- **Consumes:** 6b's prefill attribution facility if it exists; Task 1's and
  Task 3's pre-registered numbers; the composed budget above.
- **Produces:** the attention family's share of the prefill step, measured, and
  the doc sections spec §9 requires in the same commit as the mechanism.

- [ ] **Step 1: The instrument.** If 6b built per-launch event timing for the
  prefill path, use it - `Context::launch`'s optional `l0::Event* signal`
  (ruling A3) gives device timestamps for the four L0 kernels, and the SYCL
  queue's own profiling info (or a drain bracket) gives the two GEMMs. If not,
  build the **drain-bracketed** one: a `--pp-profile` flag that calls
  `cx.wait()` before and after each family and accumulates host wall time. It is
  legitimate here and not in decode because prefill kernels are
  millisecond-class (spec §3.6) - but it is perturbing, so **measure the
  perturbation**: time an empty drain in the same loop and quote it as the
  instrument's floor, the way `docs/15` §5.4 quotes the profiler's. Note that
  the composed path already pays 256 drains per chunk for ordering, so the
  instrument's marginal cost here is small and quantifiable.
- [ ] **Step 2: Measure the composed attention, the three parts separately, and
  score the pre-registration.** At **C = 2048, pos = 2048 (D = 4096)** - the
  point P4's 576.194 ms was measured at - on card 1, 8 replays / drop 3 /
  median, and again at `C = 1024` over `pos ∈ {0, 1024, 2048, 3072}`:

  | term | pre-registered ms/chunk (16 layers) | basis |
  |---|---:|---|
  | `attn_prep_chunk` | 2.7 | 100.7 MB/layer at 590 GB/s (derived) |
  | QKᵀ (`gemm_bf16_batched`) | 30.1 - 36.6 | 103.079 GFLOP/layer at 54.80 - 45 TFLOP/s |
  | `pf_softmax_causal` | 27.3 - 32.8 | 251.7 - 302.0 MB/group-launch at 590 GB/s |
  | PV (`gemm_bf16_batched`) | 18.3 - 27.5 | 103.079 GFLOP/layer at 90 - 60 TFLOP/s |
  | `pf_attn_scale_pack` | 2.0 | 75.7 MB/layer at 590 GB/s (derived) |
  | queue handoffs | 2.19 | 256 × 8.569 µs (measured, P1) |
  | **`attn_chunk` total (the A14 band)** | **85 - 100** | sum of rows 2-6 |
  | `attn_gate_chunk` | 2.7 | 100.7 MB/layer at 590 GB/s (derived) |
  | **attention family total** | **91 - 106** | all rows |

  Write predicted and measured side by side **with the ratio**, convert each
  GEMM row to TFLOP/s with the FLOP formula above, and **say plainly which model
  died if one did**. The two most likely to die, named now: the softmax's L2
  assumption (Task 3 Step 2's lever is the answer) and PV's occupancy at 48
  work-groups (Task 1 Step 1's alternative is the answer).
- [ ] **Step 3: Score against the retired path and against the composed
  ceiling.** State `576.194 / measured` as the headline ratio, with both numbers
  carrying their conditions. Then recompose the chunk with this plan's measured
  attention beside the other terms - GEMM 680.1, dequant 210.1, small kernels at
  S = 1 (whatever the parallel small-kernel work measured; ~70 pre-registered),
  GDN 15.4, interop/lm_head 8 - and report the **derived t/s at C = 2048**
  against vLLM's 1973 (HTTP-inclusive) with both labels attached. The
  pre-registered composition is **≈1075 ms/chunk → ≈1905 t/s**, and with plan
  6c's dequant double-buffer lever **≈865 ms → ≈2370 t/s**.
- [ ] **Step 4: docs/12 - the mechanism section, in the shape of the `attn`
  chapter.** It must contain, at minimum:
  - the dataflow diagram of §"The composed design" and the four entry points
    with their grids;
  - **why composed beats fused HERE, with the numbers**: the fused FMHA's
    measured 5.725 TFLOP/s at head_dim 256 against the same library's unfused
    54.80 (QKᵀ) and 40.59-73.50 (PV) on the same silicon and the same pin; the
    head_dim-128 control at 18.719 showing the penalty is only ~3.3× of the ~25×
    gap; and the mechanism sentence - **fusion trades DRAM traffic for register
    pressure, and at head_dim 256 the O accumulator alone is
    `rows_per_subgroup × 256 × 4 B` per subgroup, which is what put 33,024 B of
    spill in the fastest FMHA image; the composed form pays 1.2 GB/layer of S/P
    traffic instead, which at 590 GB/s is 2 ms against the fused kernel's 36
    ms/layer.** Say explicitly that this conclusion is **shape-specific**: at
    head_dim 64-128 the fused form is the right answer on this same hardware,
    and this project's model does not have that shape;
  - the five numerics differences from decode, each with its arithmetic, and the
    statement that the golden gate is the arbiter;
  - the causal-rectangle decision and its 33.3% waste, with the trimming lever
    priced;
  - the head-tile byte arithmetic and the `Lq` rule;
  - the `zeinfo` rows for `pf_softmax` and `pf_attn_chunk`;
  - the measured ms/chunk table from Step 2;
  - **and the sentence a future reader most needs:** the KV load path, **55.4%
    of the decode launch** (`docs/15`, measured), is **not** the term here -
    prefill attention is a GEMM-rate and DRAM-traffic problem, and no
    decode-style load-path micro-variant is a lever on it.
- [ ] **Step 5: Record the retired FMHA path as a priced negative result** - not
  deleted, not buried. A closing section in
  `docs/probe-prefill-attn-2026-09-04.md` and an entry in
  `docs/07-open-questions.md`: what was built (the generic
  `FMHAConfigGenWithTileShape` at `HeadDimQK = HeadDimV = 256`, two tile configs,
  in-place cache strides, no copy), what it measured (the full battery, 330.5 -
  876.5 ms × 16 layers across C and VTiles), what was tested and **refuted** as
  a cause (causal masking inactive - refuted, source-verified; DPAS absent -
  refuted, `dpas.8x8` in the ISA; scattered loads - refuted,
  `load_block2d.ugm`; grid under-launch - refuted; spill as the binding
  constraint - refuted, the spilling config is the faster one), what remained
  (a structural head_dim-256 penalty of 3.27× plus FMHA's own efficiency on this
  part), and **what it cost and bought**: it cost Stage 0's attention arm and it
  bought the measurement that made the pivot possible. Also record the one
  correctness debt it left - the packed-cache bit-exact comparison and the
  scalar-reference error report at C=64/256 were never completed - and state
  that they are now moot because the path is retired.
- [ ] **Step 6: docs/15 - prefill-step anatomy rows** for the six kernels
  (`pf_attn_prep_chunk`, QKᵀ, `pf_softmax_causal`, PV, `pf_attn_scale_pack`,
  `pf_attn_gate_chunk`) with their per-call µs, their launch counts and their
  family share; the instrument named and its floor quoted; and a drift control
  (an untouched family's before/after) so the attention numbers are readable
  against noise the way §L5's rollups are.
- [ ] **Step 7: The stage acceptance** (spec §5, "Per-stage acceptance"): full
  suite green; the §6 bars on this stage's checkpoint (Task 5); the
  `--profile`-style attribution (Steps 1-3); docs/12 mechanism in the same
  commit. **Any miss → revert + priced record, not a patch.** Then record the
  device-side pp row at the L3 sha, **record grade** (idle box, medians of 3,
  DRM-fd evidence pasted): `b70-decode <ckpt> --bench --pp 4096 --tg 256` on the
  RTN checkpoint, reported with **both labels attached** (device-side vs vLLM's
  HTTP-inclusive 1973 t/s, chunk width, checkpoint) - spec §2. This is not the
  spec gate (§7 is, and it runs after L1-L3 are all in); it is L3's row in the
  ladder.
- [ ] **Step 8: If the attention family missed its pre-registered price**, write
  `l3-attention-memo.md`: predicted vs measured per term, which model died,
  whether the miss is trim-shaped (the causal-rectangle lever, ~14 ms/chunk),
  occupancy-shaped (Task 1's operand swap), traffic-shaped (Task 3's SLM lever)
  or library-shaped (a `sycl-tla` configuration question that goes back to the
  controller), and what it costs the composed ceiling. **STOP at the memo**;
  nothing further starts without a ruling.
- [ ] **Step 9: Commit** -
  `docs(kernels): composed prefill attention - mechanism, attribution, and the retired FMHA record`

---

## Plan self-review (2026-09-05, at authoring)

**Spec coverage.**

- **§3.3 as amended by ruling A14** (prefill attention). "The chunk's K/V are
  written into the **existing cache layout** by a widened `attn_prep` … so
  decode continues from the same cache with no relayout" → Task 2, and the
  no-relayout promise is kept literally: the composed GEMMs read
  `[pos][4][256]` **in place** through `ldb = 1024`, which is precisely why
  request A15 exists. "`S = QKᵀ` (batched GEMM, `transB` against the cache's
  `[pos][4][256]` rows)" → Task 4 Step 3, launch 1, with Task 1 Step 3 settling
  `transB` against the pin and a priced transpose fallback if A14's ColumnMajor
  claim does not hold. "`P = softmax(S)` (ours, OpenCL C, bandwidth-bound,
  causal mask by absolute position in-kernel)" → Task 3, mask
  `k <= pos + c`, in-kernel, no separate mask tensor. "`O = PV` (batched
  GEMM)" → Task 4 Step 3, launch 3. "Softmax/PV tile over heads so S/P scratch
  stays bounded at long depth (A14)" → §"The head tile Lh", with the byte
  arithmetic, the runtime `Lq` rule, and an explicit reconciliation of A14's
  `Lh = 4` illustration against GQA's group of 6. "Pre-registered: 85-100
  ms/chunk" → Task 6 Step 2's table, whose rows sum to exactly that band.
  "Flash-style streaming is **mandatory from the smallest chunk** - a naive S×S
  bf16 score block at S = 256 is exactly the 128 KB SLM budget for one head" →
  **this is the one spec sentence the composed design does not satisfy in its
  original sense, and it is superseded by A14, which materialises `S` by
  construction.** The sentence's *purpose* - never let a score block blow the
  SLM budget - is honoured: **no score block ever enters SLM at all**
  (Task 3 Step 2 states and justifies the no-SLM decision, `slm_size` = 1024 B
  asserted from `zeinfo`), and `S`'s size in DRAM is bounded by the head-tile
  rule rather than by a work-group's registers.
- **§3.6** (execution model): dynamic dispatch on one in-order context; ordering
  by drains at the 256 measured boundaries; chunk width 2048 per A13 → Task 5
  Steps 1-4.
- **§5 L3** ("FMHA (P4) or our flash kernel; retires the `C ≤ 64` limitation.
  **Gate:** correctness bars + attention-family attribution"). The disjunction
  is **overtaken by A14** - neither branch is built - and the plan says so in
  its header rather than silently substituting a third option. The two
  obligations stand and are met: the retirement → Task 5 Steps 1-4; the bars →
  Task 5 Steps 5-9; the attribution → Task 6 Steps 1-3. Per-stage acceptance →
  Task 6 Step 7.
- **§6 item 1** (golden gate after prefill, both checkpoints) → Task 5 Step 5.
  **Item 2** (multi-chunk golden gate at C = 1024 on `oracle-out-long`) →
  Task 5 Step 7. **Item 3** (self-consistency control, diagnostic) → Task 5
  Step 6. **Item 4** (determinism; no fp atomics; no split-K atomics) →
  Task 1 Step 5 (the batched scheduler evidence), Task 1 Step 6 case 7,
  Task 3 Step 5 case 7, Task 4 Step 9, Task 5 Step 8, and the Global
  Constraints. Items 5 and 6 → Task 5 Step 9.
- **§8** reproduced verbatim above. **§9 records** → Task 6 (docs/12, docs/15,
  docs/07, docs/BENCHMARKS, docs/05, and the probe doc's closing section).
- **§10 out of scope**: no task touches the tokenizer, MTP, prefix caching, fp8
  KV, or a hand-written DPAS kernel of any kind. Chunk boundaries stay at
  multiples of 1024 (2048 and the 1024 gate width), so §8's prefix-caching
  constraint is preserved by construction.

**Placeholder scan.** No TBD, no TODO, no "similar to", no "as appropriate".
Every step has real code or a real command. The angle brackets that appear are
in commit messages only and are measurement outputs, as in plan 5. The three
genuinely conditional branches are each written to full concreteness with the
condition that selects them and the loser deleted from the tree, not left live:
(i) `transB` vs `pf_k_transpose` (Task 1 Step 3 decides, Task 4 Step 3 states
the one-line difference); (ii) request A18 accepted vs the file-local scratch
map (both spelled, with the memory-ledger reason for the preference);
(iii) request A19 accepted vs a `pf_qkv_round` pass (priced at 4.78 ms/chunk).
The one external fact this plan asserts but has not itself verified is A14's
"ColumnMajor B is native", which **contradicts plan 6c's recorded
"RowMajor for A/B/C/D"**; Task 1 Step 3 is placed first precisely to settle it
against the pin, and the fallback is 0.23 ms/chunk, so the plan does not depend
on the answer.

**Type consistency vs interfaces.md, including A14's `GemmBatch`.**
`attn_prep_chunk` and `attn_chunk` keep their declared names, namespace
(`runtime::prefill`), parameter **order** and the types of every tensor
parameter; `attn_chunk`'s tensor contract is byte-for-byte what A14 says is
unchanged (`q` bf16 `[C][24][256]`, `kv_k`/`kv_v` the untouched
`[pos][4][256]`, `out` bf16 `[C][24][256]` pre-gate). The four deviations are
filed as A16-A19 with fallbacks and none is silent. `GemmBatch` keeps A14's
field **names and meanings** (`M, K, N, L, strideA, strideB, strideC` are the
same quantities with the same units - elements, not bytes) and adds exactly
three (`lda, ldb, ldc`) under request A15, with defaults chosen so every packed
call is spelled identically to A14's form; `gemm_bf16_batched`'s signature,
including `bool transB = false` and the `Context&`-first order, is A14's
verbatim. Kernels carry the `pf_` prefix and live in `src/kernels/prefill/`;
SYCL lives in `src/sycl/`; host code lives in `src/runtime/prefill/` - all four
as interfaces.md fixes them. `M` is a **runtime** argument everywhere: no
`_M{C}`, `_L{MAXLEN}` or `_B{BLOCK}` suffix appears in any new variant name,
each new kernel is one binary, and Task 5 Step 3 retires ruling A10's `M = 64`
exception exactly as A10 anticipated. Test names match interfaces.md's harness
list (`tests/prefill/attn_chunk_test.cc`); `attn_prep_chunk_test.cc`,
`softmax_causal_test.cc` and `gemm_batched_test.cc` are additions in the same
directory and idiom. `PrefillScratch` - never `PrefillBuffers` (A7);
`--pp-chunk` - never `--chunk` (A8); the pin is `91e5bd7…` (A11); `kC = 2048`
(A13). The numbers interfaces.md fixes are quoted exactly: vLLM **1973 t/s
pp4096**, today's **121 s** for 4096 ids, **48.97 GFLOP** per token, the derived
**183.5 TFLOPS** peak (and, where it is the better figure, P2's **measured**
164.21 beside it with the reconciliation sentence), and the decode gate rows
**32.22 / 29.33** at `2a7df0b`.

**Known tensions, ruled here rather than left to be discovered.**
1. **Five numerics changes, not one.** A9 named the bf16 query; the composed
   path adds the DPAS score association, the two-pass softmax, the **bf16
   attention weights** and the DPAS PV association. All five are listed with
   their arithmetic before any test runs, the ulp bar is loosened from 2 to 3
   **with the derivation stated in advance**, and the golden gate is named as
   the arbiter. This is a real risk to spec §6.1's determined rows and the plan
   stops rather than tunes if one flips.
2. **The scratch grows by 858 MiB.** That is the price of materialising `S`, and
   it is the direct consequence of A14's own design. It is bounded (it does not
   scale with head count), it is tabulated, it is checked against
   `maxMemAllocSize`, and the 604 MB alternative is documented with its 5.5
   ms/chunk cost and needs no code change to select.
3. **The GEMMs execute a rectangle the mask then zeroes**, wasting 33.3% of the
   attention FLOPs at the operating point. It is priced (≈14 ms/chunk), the
   trimming lever is specified, and it is deliberately not built because the
   pre-registered band clears without it.
4. **Request A15 is blocking and contradicts a recorded 6c fact.** Filed first,
   with the fallback priced at 0.91 ms/chunk so the plan can proceed under a
   decline, and with Task 1 Step 3 placed before any launcher code so the
   contradiction is settled by source, not by assumption.

Claude-Session: 
