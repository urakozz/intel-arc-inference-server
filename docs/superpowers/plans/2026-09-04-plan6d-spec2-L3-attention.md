# Spec 2 - Stage 1 / L3: prefill attention (stream S3's landing)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land causal chunk attention for the 16 full-attention layers at the
production chunk width, retiring plan 6b's temporary `C ≤ 64` path (which runs
the decode trio at `M = C` and pays `C × 50.72 MB` of `attn_part` - **3.25 GB at
C = 64**, derived from `buffers.h:84`) and lifting `Engine::prefill`'s `chunk`
to `PrefillScratch::kC = 4096`. Two implementations are written here; **plan
6a's probe P4 selects exactly one** and the executor builds only that one.

**Architecture:** three device entry points behind `src/runtime/prefill/attn.h`.
`attn_prep_chunk` is the widened `attn_prep` - runtime `C`, partial RoPE on 64
of 256 dims, the KV write at `[pos+c]`, the q‖gate split of q_proj's 12288
columns - and it is **bit-identical to `attn_prep` at C = 1** except for the one
deliberate change the DPAS path forces (q is stored bf16, not fp32).
`attn_chunk` is either **Branch A**, Intel's `sycl-tla` FMHA forward
instantiated at head_dim 256 on `Context::sycl()`, or **Branch B**, our OpenCL C
flash kernel `pf_attn_chunk.cl` on `intel_sub_group_bf16_bf16_matrix_mad_k16`
with `intel_sub_group_2d_block_read_*` loads (both builtins verified on this
toolchain - explorer-3 §1, and the 2D block read is already in production at
`src/kernels/gemv.cl:102`). `attn_gate_chunk` is the widened tail of
`attn_reduce`: the `attn_output_gate` sigmoid, which nothing else in the prefill
path owns. No `attn_part`: every tile owns its output.

**Tech Stack:** OpenCL C 3.0 + ocloc AOT (`cmake/ocloc.cmake`, `-device
bmg-g31`), pure Level Zero dispatch through `runtime::prefill::Context`; for
Branch A additionally SYCL via the **system** oneAPI
(`/opt/intel/oneapi/compiler/2026.1/bin/icpx`, `docs/10-the-box.md:42`) with
`sycl-tla` pinned as a static dependency. C++17, ctest on the box
(box, `tools/box.sh`, JOBS 44).

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - §3.3
(prefill attention), §5 L3 (this stage), §6 items 1-4 (the correctness bars),
§8 (constraints, reproduced verbatim below). Read it first; it governs.

**Interface contract:** `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md`
- binding. Two ruling requests against it are filed in "Interface requests"
below; **neither blocks Tasks 1-9 from starting**, because both are additive and
the fallback for each is stated.

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
  and stays there. Never touch docker, never kill a process on the box.
- GPU work runs with `ZE_AFFINITY_MASK=1` whenever a container holds card 0.
- `-Wall -Wextra -Werror` on the host; `-cl-denorms-are-zero` is **forbidden**
  and `add_ocloc_kernel` fatals on it (`cmake/ocloc.cmake:40-43`);
  `-cl-fp32-correctly-rounded-divide-sqrt` is the ocloc default and stays.
- **No fp atomics anywhere in the path** (spec §6.4). Any `sycl-tla`
  configuration that reduces with split-K/split-KV atomics is disallowed -
  Branch A's Task 4A Step 2 proves the chosen config has none.
- **Decode is untouched.** No edit to `src/kernels/attn.cl`,
  `src/kernels/CMakeLists.txt`'s decode rows, `src/runtime/capture.cc`'s
  `fa_layer`, or `DecodeBuffers`' decode fields. The launch/module invariants
  **774 / 19** (`tests/runtime/replay_determinism_test.cc:136,150`) do not move,
  and the decode gate rows **32.22 t/s (RTN) / 29.33 (Vishva)** at `2a7df0b`
  are re-measured once at this plan's final sha (spec §6.5: inside day drift
  ≤ 0.09%, or the difference is explained).
- **Pre-registered predictions before any timing.** Write the number down in
  the task's report section, then run. This is Task 3's whole job for the
  attention family and is repeated per timing step.
- Full suite green before any commit touching `src/`.
- Every commit message ends with
  `Claude-Session: `.

---

## Interfaces: what this plan consumes, and the assumptions it states

**From plan 6a (stream S1, the execution context).** `src/runtime/prefill/context.h`:

```cpp
namespace runtime::prefill {
class Context {
 public:
  Context(l0::Device& dev, l0::Ctx& ctx);
  void wait();
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
              std::initializer_list<KernelArg> args);
  sycl::queue& sycl();
};
struct KernelArg { const void* ptr; size_t size; };
}
```

Assumed: `launch` sets the work-group size from the kernel's
`reqd_work_group_size` (every kernel here declares one), takes the **grid in
work-groups** on `(gx, gy, gz)`, and orders launches in issue order on one
in-order immediate list; `wait()` drains both queues. If `launch` takes a global
size in work-items instead, every grid in this plan is multiplied by the
work-group size at the call site and nothing else changes.

**From plan 6a, probe P4.** The verdict "does `sycl-tla`'s FMHA forward
instantiate at head_dim 256, and at what µs for C ∈ {1024, 2048, 4096} over
depth 4096", recorded in the docs/12 prefill probe matrix. **P4 = pass → Task
4A; P4 = fail → Task 4B.** Task 3 reads it and records the choice.

**From plan 6b (stream S4, plumbing).** Assumed, each with its repair:

1. `PrefillScratch` exists with `static constexpr uint32_t kC = 4096` and holds
   the attention q/out buffers. This plan needs two of them, bf16:
   `pf_q` `[kC][24][256]` (50.33 MB) and `pf_attn` `[kC][24][256]` (50.33 MB),
   plus the gated output `[kC][6144]` (50.33 MB) which is `OProj`'s activation
   input. If 6b named them differently, Task 6 uses 6b's names; if 6b sized
   them for the temporary path only, Task 6 grows them and updates
   `tests/runtime/buffers_test.cc`'s totals.
2. `Engine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0)`
   exists and its fa-layer branch calls the **temporary** path: the decode trio
   at `M = C` with `C` clamped to 64. Task 6 replaces exactly that branch.
3. The chunk's fused qkv linear result reaches attention as **bf16
   `[C][14336]`** - `rne_bf16` of the GEMM's fp32 output, the same single
   rounding `attn_prep` applies to its split-K partials
   (`attn.cl:307-311,350`). Interfaces.md fixes GEMM output as fp32 `[M][N]`
   with "the consumer rounds to bf16 where decode's consumer would", and
   `attn_prep_chunk`'s declared `qkv` parameter is `const uint16_t*`, so the
   rounding pass is 6b's. **This is not bit-identical to decode** - decode sums
   S = 2 split-K slices and rounds once, the GEMM accumulates in one fp32 chain
   - and that difference is a property of L2's GEMM swap, not of L3. Task 5
   feeds both sides of its unit test the *same* bf16 `qkv`, so the unit bar is
   unaffected; the golden gate is where the composed difference is judged.
4. `oracle-out-long` (a ≥ 2048-id oracle prompt, spec §6.2) is registered, and
   `tests/prefill/prefill_gate_test.cc` carries the multi-chunk row at
   `C = 1024`. Assumed to be registered-but-capped (or skipped) while the
   temporary path clamps `chunk` to 64. Task 7 flips it on.
5. A `--profile`-style attribution for the prefill step. If 6b built one, Task 8
   consumes it; if not, Task 8 builds the drain-bracketed instrument described
   there.

**What this plan produces**, exactly as interfaces.md declares (`src/runtime/prefill/attn.h`):

```cpp
namespace runtime::prefill {
// (1) Write the chunk's K/V (RoPE applied on the 64 rotary dims) into the
//     cache at positions [pos, pos+C) - the widened attn_prep.
void attn_prep_chunk(Context& cx, uint32_t layer, uint32_t pos, uint32_t C,
                     const uint16_t* qkv /* [C][12288+1024+1024] */,
                     uint16_t* q_out /* [C][24][256], RoPE'd */,
                     uint16_t* kv_k, uint16_t* kv_v);
// (2) Causal attention of the C queries over cache [0, pos+C).
//     out: bf16 [C][24][256] (pre output-gate, pre o_proj).
void attn_chunk(Context& cx, uint32_t layer, uint32_t pos, uint32_t C,
                const uint16_t* q, const uint16_t* kv_k, const uint16_t* kv_v,
                uint16_t* out);
}
```

`kv_k` / `kv_v` are **this layer's slices** - the caller adds
`f * max_len * 4 * 256 * 2` bytes with `f = layer / 4`, exactly as
`capture.cc:520-521` does. `layer` is carried for attribution labelling and for
the small-tensor lookup (request R2 below).

### Interface requests (to the controller - additive, both with fallbacks)

**R1 - `attn_gate_chunk` is missing from `attn.h`.** `attn_chunk`'s output is
declared "pre output-gate", and nothing else in the contract applies the
`attn_output_gate`. In decode it is the last line of `attn_reduce`
(`attn.cl:633`); spec §3.5's list of "everything else in the chunk" does not
name it. It is attention's, so S3 owns it. Requested addition:

```cpp
// (3) The attention output gate - the last op of decode's attn_reduce, widened.
//     x_out[c][h*256+d] = rne_bf16(f32(out[c][h][d]) * sigmoid_f32(f32(gate col)))
//     Gate columns are read in place from `qkv` at column h*512 + 256 + d.
//     x_out: bf16 [C][6144] - o_proj's activation input.
void attn_gate_chunk(Context& cx, uint32_t C, const uint16_t* qkv,
                     const uint16_t* attn, uint16_t* x_out);
```

*Fallback if declined:* the same kernel and the same call site, declared in
`src/runtime/prefill/attn_gate.h` and owned by S4; Task 2 moves file, not code.

**R2 - `attn_prep_chunk` cannot reach the two tensors it must read.** The
kernel needs the layer's FA small block (`q_norm`/`k_norm`, fp32 `(1+w)`, at
`loader::kFaOffQNorm` / `kFaOffKNorm` inside `m_.layer_small[layer].gdn`) and
the RoPE table (`m_.rope`, fp32 `[max_len][2][32]`) - `capture.cc:530-531`
binds both. Neither is in the declared signature and `Context` carries no model.
Requested: append two parameters,

```cpp
void attn_prep_chunk(Context& cx, uint32_t layer, uint32_t pos, uint32_t C,
                     const uint16_t* qkv, const float* fa_small, const float* rope,
                     uint16_t* q_out, uint16_t* kv_k, uint16_t* kv_v);
```

*Fallback if declined:* keep the declared signature and add
`Context::model(const loader::LoadedModel&)` - but that contradicts the
declared `Context` API, so the parameter form is the smaller change.

### One deliberate numerics change, recorded up front

Decode keeps `attn_q` **fp32** and feeds fp32 queries into the score dot
(`buffers.h:82`, `attn.cl:436`). Interfaces.md fixes prefill's `q_out` as
**bf16**, which both branches require: DPAS takes bf16 operands. So prefill
applies one extra rounding to the query, `q_bf16 = rne_bf16(roped fp32)`.

This is a change **towards** the oracle, not away from it: torch's
`Qwen3_5Attention` carries q as a bf16 tensor into the score matmul (doc 03),
so decode's fp32 query is the deviation and prefill's bf16 query is the
reference's own dtype. It is recorded in docs/12 (Task 8) and it is why Task 5's
CPU reference widens the device's bf16 `q_out` back to fp32 before running
`attn_ref::decode` - the reference must round where the kernel rounds.

---

### Task 1: `attn_prep_chunk` - the widened `attn_prep`

**Files:**
- Create: `src/kernels/prefill/pf_attn.cl` (kernel `pf_attn_prep`)
- Create: `src/runtime/prefill/attn.h`, `src/runtime/prefill/attn.cc`
- Modify: `src/kernels/CMakeLists.txt` (one `add_ocloc_kernel` row),
  `src/kernels/kernels.h` (`pf_attn_prep_variant()`),
  `src/runtime/CMakeLists.txt` (the new source)
- Test: `tests/prefill/attn_prep_chunk_test.cc` (new), registered in
  `tests/CMakeLists.txt`

**Interfaces:**
- **Consumes:** `runtime::prefill::Context::launch`; the fused qkv column map
  (`attn.cl:32-39`: q-head `h` at `[h·512, h·512+256)`, its gate at the next
  256; k-head `j` at `12288 + j·256`; v-head `j` at `13312 + j·256`); the
  loader's RoPE table `fp32 [max_len][2][32]` and FA small block layout
  (`loader/small_layout.h`, `kFaOffQNorm`/`kFaOffKNorm`); the KV cache layout
  `[pos][4][256]` bf16 per layer (`buffers.h:74`).
- **Produces:** `void runtime::prefill::attn_prep_chunk(Context&, uint32_t layer,
  uint32_t pos, uint32_t C, const uint16_t* qkv, const float* fa_small,
  const float* rope, uint16_t* q_out, uint16_t* kv_k, uint16_t* kv_v)` and the
  device kernel
  `pf_attn_prep(__global const ushort* qkv, __global const float* fa_small,
  __global const float* rope, __global ushort* q_out, __global ushort* kv_k,
  __global ushort* kv_v, uint pos, uint C)`.

- [ ] **Step 1: Write `pf_attn_prep`** in `src/kernels/prefill/pf_attn.cl`.
  Grid **(28, C, 1)**, work-group 256 - one work-group per (head, chunk
  position), work-item `i` owning dim `i`, exactly `attn_prep`'s shape
  (`attn.cl:314-326`). Dynamic dispatch sizes the grid at `C` itself, so there
  is **no `n_active` clamp and no device-side mask**: the `if (m >= n_act)`
  early-out of `attn.cl:343` has no counterpart here. Groups 0..23 are q-heads,
  24..27 are kv-heads. The chain, op for op the same as `attn_prep`:
  ```c
  const uint wg = get_group_id(0), c = get_group_id(1), i = get_local_id(0);
  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  __global const float* restrict nw = fa_small + (is_q ? QNORM_OFF : KNORM_OFF);
  const size_t base = is_q ? (size_t)h * 2 * HD : (size_t)K_OFF + (size_t)h * HD;
  const float xf = bf16f(qkv[(size_t)c * QKV_N + base + i]);   // already rounded once
  red[i] = xf * xf;                                            // plain multiply, no fma
  // ... the same 256 -> 1 pairwise tree, stride = 128, 64, ..., 1, barrier per step
  const float rstd = 1.0f / sqrt(red[0] / (float)HD + 1e-6f);  // never rsqrt
  nrm[i] = bf16f(rne_bf16(xf * rstd * nw[i]));
  // ... the same partial RoPE over dims 0..63, pairs (i, i+32), on nrm,
  //     cs = rope + (size_t)(pos + c) * 2 * ROT_HALF
  ```
  **The one substitution:** decode's `bf16f(rne_bf16(qkv_sum(partials, m, col)))`
  becomes `bf16f(qkv[c*QKV_N + col])` - the split-K fold and its rounding have
  already happened upstream (Consumes assumption 3), and widening a bf16 word is
  exact, so the value entering the norm is the same value.
- [ ] **Step 2: The stores.** q-head: `q_out[(c*24 + h)*256 + i] =
  rne_bf16(outv)` (bf16 - the deliberate change recorded above). kv-head:
  `kv_k[((size_t)(pos + c) * KV_HEADS + h) * HD + i] = rne_bf16(outv)` and
  `kv_v[same slot] = qkv[c*QKV_N + V_OFF + h*HD + i]` - v is never normed,
  never roped, and is **already** the linear's bf16 word, so it is a copy, not a
  re-rounding. **No gate store:** the gate columns stay in `qkv` and
  `attn_gate_chunk` (Task 2) reads them there, which is what lets the declared
  signature have no gate output.
- [ ] **Step 3: Build rule.** `src/kernels/CMakeLists.txt`, beside the attention
  block but in its own section so nothing decode reads moves:
  ```cmake
  add_ocloc_kernel(pf_attn_prep SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/pf_attn.cl
    DEFINES CTRL_POS=0 QNORM_OFF=${QNORM_OFF} KNORM_OFF=${KNORM_OFF})
  ```
  reusing whatever the decode block already passes for `QNORM_OFF`/`KNORM_OFF`;
  **no `M`, no `MAXLEN`, no `ATTN_BLOCK`** - `pos` and `C` are runtime arguments
  and this kernel blocks nothing, so it is **one binary** and
  `kernels::pf_attn_prep_variant()` returns the constant `"pf_attn_prep"`.
  (`CTRL_POS` is only needed if the shared header block is reused; drop it if
  the file includes nothing from `attn.cl`.)
- [ ] **Step 4: The host shim** in `src/runtime/prefill/attn.cc`: load the
  module once (a function-local `static` cache keyed on nothing - one variant),
  set the six pointer args plus `pos` and `C` as `KernelArg{&pos, 4}`, and
  `cx.launch(k, 28, C, 1, {...})`. `require(pos + C <= max_len)` is the
  caller's (Engine's) job and is already Engine's rule (`engine.h`'s `replay()`
  comment); assert it here anyway with a throw, because prefill's dispatch is
  dynamic and a bad `C` would write past the cache.
- [ ] **Step 5: The bit-identity test** - `tests/prefill/attn_prep_chunk_test.cc`.
  Build one `Inputs` the way `attn_test.cc:446-463` does (xorshift or
  `std::mt19937`, seeded), then:
  1. run `attn_ref::prep(pos, 1, 1, partials, fa_small, rope, q_ref32, g_ref,
     k_ref, v_ref)` - the existing decode reference;
  2. build `qkv_bf16[col] = common::f32_to_bf16(partials[0][col] +
     partials[1][col])` on the host (the ascending-slice fold `attn_ref.h:158-163`
     performs), which is exactly what assumption 3 says the GEMM rounding pass
     produces;
  3. run `pf_attn_prep` at `C = 1` on `qkv_bf16`;
  4. assert `kv_k` and `kv_v` **bit-identical** to `k_ref`/`v_ref` over the whole
     cache (`require_bits16`, the same bar `attn_test.cc:603-604` holds), and
     `q_out[i] == common::f32_to_bf16(q_ref32[i])` **bit-identical** over all
     6144 words.
  These are bit bars, not tolerances: nothing in this chain uses `exp`, and
  `1.0f/sqrt` is correctly rounded by the build flag
  (`cmake/ocloc.cmake:12-22`).
- [ ] **Step 6: The multi-position cases.** Repeat at `C ∈ {2, 64, 1024}` and
  `pos ∈ {0, 4095}` against a host loop that calls `attn_ref::prep` once per
  chunk position with `M = 1, n_act = 1, pos = pos + c` - proving the cache is
  written at `[pos+c]` for every `c`, that no slot outside `[pos, pos+C)` moves
  (canary the cache with a seeded fill and require the complement bit-identical
  to the seed), and that the RoPE table is indexed at the absolute position.
- [ ] **Step 7:** `tools/box.sh test attn_prep_chunk`; then the full suite
  (`tools/box.sh test`) - nothing existing may move.
- [ ] **Step 8: Commit** -
  `feat(prefill): attn_prep_chunk - runtime-C KV write, bit-identical to attn_prep at C=1`

### Task 2: `attn_gate_chunk` - the output gate, widened

**Files:**
- Modify: `src/kernels/prefill/pf_attn.cl` (second entry point `pf_attn_gate`),
  `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`,
  `src/runtime/prefill/attn.{h,cc}`
- Test: `tests/prefill/attn_prep_chunk_test.cc` gains the gate case (one binary,
  one fixture - the gate reads the same `qkv`)

**Interfaces:**
- **Consumes:** the q‖gate column map; `attn_reduce`'s final chain
  (`attn.cl:633`).
- **Produces:** `void runtime::prefill::attn_gate_chunk(Context&, uint32_t C,
  const uint16_t* qkv, const uint16_t* attn, uint16_t* x_out)` - see request R1.

- [ ] **Step 1: Write `pf_attn_gate`.** Grid **(24, C, 1)**, work-group 256,
  work-item `d` owning dim `d` - `attn_reduce`'s grid with `M` replaced by `C`:
  ```c
  const uint h = get_group_id(0), c = get_group_id(1), d = get_local_id(0);
  const float g = bf16f(qkv[(size_t)c * QKV_N + (size_t)h * 512 + HD + d]);
  const float o = bf16f(attn[((size_t)c * Q_HEADS + h) * HD + d]);
  x_out[(size_t)c * OUT_N + (size_t)h * HD + d] = rne_bf16(o * sigmoid_f32(g));
  ```
  This is `attn.cl:633` with one rounding removed and nothing else: decode
  computes `rne_bf16(bf16f(rne_bf16(acc/sm)) * sigmoid_f32(gate))`, and
  `attn_chunk` has **already** performed the inner `rne_bf16(acc/sm)` when it
  wrote its bf16 output, so `bf16f(attn[...])` is decode's
  `bf16f(rne_bf16(acc/sm))` verbatim. `sigmoid_f32` is plain `exp`, not
  `native_exp`, and is the **last** op - the same discipline `attn.cl:302-305`
  and `prep.cl:110-114` state.
- [ ] **Step 2:** `add_ocloc_kernel(pf_attn_gate ...)` from the same source file
  (one `.cl`, two entry points - the arrangement `attn.cl` already uses; note
  that ocloc emits one binary per `add_ocloc_kernel` call, so both entry points
  exist in both binaries and `l0::Kernel(mod, "pf_attn_gate")` picks the one it
  wants - see `attn.cl`'s three entry points in one file and the `CMakeLists`
  comment at line 244-252 explaining why unused `-D`s cannot reach codegen).
  Simplest correct form: one `add_ocloc_kernel(pf_attn SOURCE prefill/pf_attn.cl)`
  producing `pf_attn.bin`, from which both `pf_attn_prep` and `pf_attn_gate` are
  created. Use that; `kernels::pf_attn_variant()` returns `"pf_attn"`.
- [ ] **Step 3: Test** it in the Task-1 fixture: run `attn_ref::reduce` on a
  synthetic `attn_part` to get `out_ref` **with the real gate**, and separately
  with the gate array filled with `+INFINITY` to get the **pre-gate** tensor
  (`sigmoid_f32(+INF) = 1/(1+exp(-INF)) = 1/(1+0) = 1.0f` exactly, so
  `attn_ref::reduce` degenerates to `rne(f32(rne(acc/sm)) * 1.0f) = rne(acc/sm)`
  - the identity that lets the same reference serve both sides without editing
  `attn_ref.h`). Feed the pre-gate tensor to `pf_attn_gate` with the real gate
  columns and require the result **bit-identical** to `out_ref`. Bit, not
  tolerance: the host and device `exp` may differ by up to 3 ulp, so if this
  fails at 1-2 bf16 ulp on a handful of words it is `exp`, and the bar drops to
  `require_ulp(..., bar = 1, ...)` with the count printed - record which,
  because it is the same question `attn_test.cc:26-52` answers for decode.
- [ ] **Step 4:** `tools/box.sh test attn_prep_chunk`; full suite.
- [ ] **Step 5: Commit** -
  `feat(prefill): attn_gate_chunk - the attn_output_gate, widened to C`

### Task 3: Branch selection and the pre-registered price

*No code. This task exists because the Global Constraints require the number to
be written down before it is measured, and because the executor must not build
both branches.*

**Files:**
- Create: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/l3-attention-prereg.md`
  (the pre-registration; a report artefact, not repo docs)

**Interfaces:**
- **Consumes:** plan 6a's P4 row in the docs/12 prefill probe matrix; P2's
  measured XMX bf16 TFLOP/s if it exists.
- **Produces:** the branch ruling (A or B) and the pre-registered per-call
  price that Task 8 is judged against.

- [ ] **Step 1: Read P4** and record verbatim: does the FMHA forward
  instantiate at head_dim 256 (qk and vo), at what tile config, and its µs at
  C ∈ {1024, 2048, 4096} over depth 4096. **Pass → Task 4A. Fail → Task 4B.**
  If P4 has not run, this plan stops here and says so; it does not guess.
- [ ] **Step 2: Write the FLOP arithmetic** (derived, and reproducible from the
  shapes in `docs/03-models.md`). With `n = C/64` query tiles and `P = pos/64`,
  causal block-trimming executes `n·P + n(n+1)/2` key-blocks of 64, so per
  **layer-call**:
  ```
  FLOP = 4 (mul+add, QK and PV) x 24 q-heads x 256 dims x 64 x 64 x (n·P + n(n+1)/2)
  ```
  - C = 4096, pos = 0: `n = 64, P = 0` → 2080 blocks → **209.4 GFLOP/layer-call**,
    **3.35 TFLOP** over the 16 FA layers (derived).
  - C = 1024 in four chunks to 4096: `P = 0,16,32,48` → 136+392+648+904 = 2080
    blocks, **the same 3.35 TFLOP** - the arithmetic check that the chunked and
    single-shot forms count the same work.
  - Reconciliation with spec §3.3's "**~3% of prefill FLOPs at depth 4096**":
    that figure counts the **full** `C × C` score matrix; block-trimmed causal
    execution is 3.35 TFLOP against the prefill's 200.6 TFLOP
    (48.97 GFLOP/token × 4096, interfaces.md) = **1.67%**. Same quantity, two
    masking conventions; the spec's 3% is the upper bound and this plan quotes
    1.67% as the executed share.
- [ ] **Step 3: Write the DRAM floor** (derived): per layer-call at C = 4096,
  depth 0 the unavoidable traffic is q 50.33 MB + out 50.33 MB + K/V 16.78 MB
  (`4096 × 4 × 256 × 2 B × 2`, and the whole layer's KV is **16.78 MB against
  a 24 MB L2** - `docs/01-hardware.md:16` - so it is read from DRAM once and
  serviced from L2 thereafter) = **117.4 MB → 199 µs at the measured 590 GB/s**.
  Against Step 2's compute this kernel is **compute-bound by ~23×**. That is the
  inversion from decode, where the same family is **55.4% KV-load-path**
  (`docs/15` §"The dominant term", measured): at prefill, attention is an XMX
  problem, not a message-issue problem, and no load-path micro-variant is a
  lever here.
- [ ] **Step 4: Pre-register the price**, against the **derived** XMX bf16 peak
  183.5 TFLOP/s (vendor int8 ÷ 2, interfaces.md - replaced by P2's measured
  figure the moment it exists, and if P2 has landed, restate every row against
  the measured number and say so):
  | branch | pre-registered efficiency | µs/layer-call at C=4096, depth 0 | ms over 16 layers |
  |---|---|---|---|
  | A - FMHA (Intel's tuned tile) | ≥ 50% of peak (91.8 TFLOP/s) | ≤ **2280** | ≤ **36.5** |
  | B - ours (first hand-written flash kernel) | ≥ 25% of peak (45.9 TFLOP/s) | ≤ **4560** | ≤ **73.0** |
  | B - stop bar | < 9.5% of peak (17.5 TFLOP/s) | > **12000** | > 192 |
  *(45.9 TFLOP/s coincides numerically with spec §1's "optimistic vector-only
  ceiling"; they are unrelated quantities and this plan never adds them.)*
  Below the stop bar the XMX path is not paying for itself and Task 9 writes the
  memo instead of another tile sweep.
- [ ] **Step 5: Pre-register the correctness bars** (Task 5 asserts them):
  `attn_chunk` output is **bf16**, so `attn_test`'s `attn_part` bar of
  **relative 1e-3** (`tests/kernels/attn_test.cc:26`, and asserted at
  `:639`) **does not transfer and must not be used**. That bar is for an fp32
  tensor; `attn_test.cc:54-79` records that 1e-3 was *never* consistent with a
  bf16 output, because one bf16 ulp on a gated element reaches **7.8e-3**
  (7 explicit mantissa bits → one ulp is 2⁻⁸…2⁻⁷ of the element). Both branches
  **reassociate** the softmax by construction, so a one-ulp move on a boundary
  word is expected, not a defect. The bars are therefore `attn_out`'s pair:
  **(a) relative ≤ 8e-3 floored at the tensor RMS**, **(b) ≤ 2 bf16 ulp on every
  element with `|ref| ≥ rms/8`, (b) being the arbiter** (`attn_test.cc:40-52`).
  (b) stays at **2** rather than tightening to 1 - decode's "2" is one ulp per
  bf16 rounding × two roundings and `attn_chunk` has only one, but decode's
  argument also assumes an *order-preserving* kernel, which neither branch is;
  2 is the number this family's gate has lived with since lever L5 and it is
  the arbiter. Record the measured worst-gated ulp per case: any case at 2 is a
  finding to report, not a pass to bank silently.
- [ ] **Step 6: Commit** - `docs(sdd): L3 attention - branch ruling and pre-registered price`

### Task 4A: Branch A - `sycl-tla` FMHA forward at head_dim 256

*Built only if Task 3 Step 1 read **P4 = pass**. If P4 failed, skip to 4B.*

**Files:**
- Create: `src/sycl/pf_attn_fmha.cpp` (the instantiation and the launch),
  `src/sycl/pf_attn_fmha.h` (one declaration, no SYCL types in it)
- Modify: `src/runtime/prefill/attn.cc` (`attn_chunk` forwards to it),
  `src/CMakeLists.txt` / `cmake/sycl.cmake` (6a's `sycl-tla` target; add this
  source to it)
- Test: `tests/prefill/attn_chunk_test.cc` (Task 5) is the correctness bar

**Interfaces:**
- **Consumes:** `Context::sycl()` - a SYCL in-order queue built from the
  engine's `ze_context`/`ze_device` (6a's interop), so the USM device pointers
  the engine owns are directly addressable; `sycl-tla` pinned at the sha 6a
  recorded; P4's instantiated tile config.
- **Produces:** `runtime::prefill::attn_chunk` with the declared signature,
  implemented over the FMHA forward.

- [ ] **Step 1: Pin the instantiation.** Copy P4's working type chain out of
  the probe into `src/sycl/pf_attn_fmha.cpp` verbatim - the probe is what
  proved it compiles, so nothing is re-derived here. Read
  `~/PycharmProjects/sycl-tla/applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp`
  and the runner beside it for the exact `Arguments` field names and fill them
  from a single `attn_chunk` call. The reference instance is
  `examples/06_bmg_flash_attention/06_xe_fmha_fwd.cpp:115-118` - at head_dim
  128 it is `ShapeQK = Shape<_256,_32,_32>`, `ShapePV = Shape<_256,_32,_32>`,
  `ShapeOut = Shape<_256,_128>`, `SubgroupLayoutQK = Layout<Shape<_16,_1,_1>>`,
  `PipelineStages = 2` (explorer-3 §4).
  **The register arithmetic says `ShapeOut` cannot simply widen to
  `Shape<_256,_256>`** (derived): the O accumulator is `rows_per_subgroup ×
  head_dim` fp32, so 256 queries over 16 subgroups at head_dim 256 is
  `16 × 256 × 4 B / 16 lanes = 1024 B per lane` - the **entire** 256-GRF file
  at SIMD16 (256 × 64 B / 16 lanes), leaving nothing for Q, S, softmax state or
  addresses. Whatever P4 instantiated must therefore have halved the Q tile
  (`Shape<_128,_256>`) or doubled the subgroup count; **record which**, because
  it is the number Task 8's docs/12 entry has to state and it is the reason a
  head_dim-256 FMHA is not a free widening.
- [ ] **Step 2: Prove there are no atomics** (spec §6.4, Global Constraints).
  Grep the instantiated path for `atomic`, `SplitK`, `split_kv`, `reduce` across
  work-groups, and any workspace the kernel allocates; state the scheduler the
  config selects and whether it reduces across work-groups. Paste the evidence.
  If the chosen config reduces with atomics, **it is disallowed** - reconfigure
  to a non-split scheduler and re-price, or fall to Branch B.
- [ ] **Step 3: Bind our buffers with no copy.** Q is `q_out` bf16
  `[C][24][256]` from Task 1 - position-major, head-minor, `lda` per position
  = 6144 elements, per head-row 256, contiguous. K and V are **our cache**,
  bf16 `[pos][4][256]` per layer, per-position stride 1024 elements, per-kv-head
  stride 256, contiguous. Build `StrideQ/StrideK/StrideV/StrideO` from those
  numbers. **Two checks before writing a line of it:**
  (a) if the FMHA requires head-major Q (`[24][C][256]`), Task 1's store index
  changes to `q_out[(h*C + c)*256 + i]` - a free choice at the point of store -
  and interfaces.md's `[C][24][256]` comment becomes a third ruling request;
  (b) if it requires head-major **K/V**, Branch A **fails**, because the cache
  layout is decode's and this spec does not relayout it (§3.3: "so decode
  continues from the same cache with no relayout"). Record which, immediately.
- [ ] **Step 4: The causal anchoring, which is the whole correctness question.**
  Query `c` sits at absolute position `pos + c` and must attend keys
  `[0, pos + c]`. Bind the **CachedKV** form so the kernel's own geometry does
  it: `k_cache = kv_k`, `v_cache = kv_v`, `seq_len_kv_cache = pos`; `k = kv_k +
  pos*1024`, `v = kv_v + pos*1024`, `seq_len_kv = C`; `seq_len_qo = C`. Both
  pointers address the *same* buffer at different offsets - legal, no copy, and
  it is exactly the chunked-prefill-with-cache geometry the kernel was written
  for, so the causal mask lands where it must with no offset arithmetic of our
  own. If the instantiated config instead takes a single `seq_len_kv = pos + C`,
  verify from the masking source that the anchoring is **bottom-right**
  (`key ≤ pos + c`, i.e. offset `seq_len_kv − seq_len_qo`) and not top-left
  (`key ≤ c`); the KV-loop trimming is `kblocks_cache + ceil_div(seq_len_new,
  BLK_K)` and the per-element mask is applied on the diagonal tile only
  (`xe_fmha_fwd_kernel.hpp:260-263`, `xe_fmha_fwd_mainloop.hpp:699-720`,
  explorer-3 §4). **State in the commit which form was used and quote the
  offset arithmetic.** A wrong anchoring is silent: the tokens change and only
  the golden gate catches it, which is why Task 7 is not optional.
- [ ] **Step 5: The remaining arguments.** `softmax_scale = 1.0f/16.0f`
  = 1/√256 - the same `SCALE 0.0625f` literal `attn.cl:228` uses; note that the
  FMHA folds `log2(e)` into it for its `exp2`-based softmax
  (`xe_fmha_fwd_mainloop.hpp:883-926`, explorer-3 §4), which is a **numerics
  difference from decode's plain `exp`** and is recorded in docs/12. GQA is
  24 q / 4 kv - bind it the way the problem shape expresses head groups; do not
  emulate it by launching four times. `cumulative_seqlen_q = {0, C}` and
  `cumulative_seqlen_kv = {0, pos+C}` only if the varlen path is the one that
  compiles; the single-sequence (non-varlen) path is preferred because it needs
  no device-side arrays.
- [ ] **Step 6: Submit on `cx.sycl()`** and rely on the in-order queue; do not
  add a `wait()` per layer - `Context::wait()` at the end of the chunk is the
  only drain, and 6a's P1 measured the interop cost under 1% of chunk time.
- [ ] **Step 7:** `tools/box.sh build` with the system icpx
  (`CMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx` for the SYCL
  target only, per 6a's build), then Task 5's test.
- [ ] **Step 8: Commit** -
  `feat(prefill): attn_chunk via sycl-tla FMHA at head_dim 256 - <causal form>`

### Task 4B: Branch B - `pf_attn_chunk.cl`, our flash kernel

*Built only if Task 3 Step 1 read **P4 = fail**.*

**Files:**
- Create: `src/kernels/prefill/pf_attn_chunk.cl`
- Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`,
  `src/runtime/prefill/attn.cc`
- Test: `tests/prefill/attn_chunk_test.cc` (Task 5)

**Interfaces:**
- **Consumes:** `intel_sub_group_bf16_bf16_matrix_mad_k16` - M ∈ {1,2,4,8},
  K = 16, N = 16, fp32 accumulate, verified to lower to a real `dpas.8x8` with
  `has_dpas: true` on `ocloc 26.27` / `bmg-g31` (explorer-3 §1); no pragma is
  needed or accepted. `intel_sub_group_2d_block_read_*` - already in production
  at `src/kernels/gemv.cl:102` and characterised in
  `tools/probe/probe_gemv_loads.cl:115-125`: **the base address must be 64-byte
  aligned** and no pragma is required (`probe_gemv_loads.cl:157-160`).
- **Produces:** `runtime::prefill::attn_chunk` and the kernel
  `pf_attn_chunk(__global const ushort* q, __global const ushort* kv_k,
  __global const ushort* kv_v, __global ushort* out, uint pos, uint C)`.

- [ ] **Step 1: Fix the geometry, and state the register arithmetic that fixes
  it** (derived; SIMD16, 128 GRF = 64 B × 128 / 16 lanes = **512 B = 128 dwords
  per lane**). The O accumulator is fp32 and is `rows_per_subgroup × 256` per
  subgroup, i.e. **16 dwords per lane per row** - the whole GRF file holds
  **8 rows**. Everything else (the Q fragment, the score tile, `mx`/`sm`
  replicated across the lanes of a subgroup, addresses) comes out of the same
  128. Per-lane cost of a subgroup owning `r` rows:
  | term | dwords/lane | at r = 4 |
  |---|---|---|
  | O accumulator, fp32, 256 dims | `16·r` | 64 |
  | score tile, fp32, one 64-key block | `4·r` | 16 |
  | `mx`, `sm` (per row, replicated over the 16 lanes) | `2·r` | 8 |
  | Q fragment, bf16, 256 dims | `8·r` | 32 → **0, see below** |
  | temps/addresses (estimated) | - | ~16 |
  **Q is not register-resident**: at `r = 4` the four terms sum to 120 of 128
  before temps, so Q is re-read per k-step by
  `intel_sub_group_2d_block_read_16b_*` from the `[C][24][256]` surface, leaving
  **≈ 88 + ~16 = 104 of 128**. That is the budget; `zeinfo`'s `spill_mem_size`
  and `grf_count` are the arbiter (Step 6), exactly as docs/12's Task-5 table
  used them.
  **The consequence for GQA packing, stated because it is a real finding:**
  Task 5's "one K/V walk serves six q-heads" **does not transfer whole at
  head_dim 256**. Six heads × 64 queries = 384 live rows, and 384 rows of O is
  `384 × 256 × 4 B = 393 KB` - over any work-group's register file (a 1024-item
  work-group has 1024 × 512 B = 512 KB at 128 GRF, and O alone would be 96
  dwords/lane before Q, S and `mx`/`sm`, which do not fit). The packing
  transfers **as far as the register file allows**, which is the swept
  parameter in Step 7.
- [ ] **Step 2: The starting geometry** (the point Step 7 sweeps around):
  - **Grid:** `(4 kv-heads × 3 head-pairs, ceil(C/64), 1)` = **(12, ceil(C/64))**
    - at C = 4096, **768 work-groups** over 32 Xe-cores.
  - **Work-group:** 512 work-items = **32 subgroups × SIMD16**,
    `reqd_work_group_size(512,1,1)`, `intel_reqd_sub_group_size(16)`.
  - A work-group owns **2 q-heads × 64 queries = 128 rows**, head-major
    (`row = qhl*64 + qt`), so subgroup `s` owns rows `[4s, 4s+4)` - four
    consecutive queries of one head, `r = 4`, DPAS **M = 4**.
  - **SLM: 64 KB of the 128 KB budget** (`docs/01-hardware.md:16`) - the K
    block (64 positions × 256 dims × 2 B = 32 KB) and the V block (32 KB),
    staged once per block and read by both packed heads. Scores never enter
    SLM: the subgroup that computes a row's scores also does that row's PV, so
    nothing crosses subgroups and **there is no work-group barrier in the inner
    loop** - only the two that bracket the K/V staging.
  - **No `attn_part`.** The tile accumulates `mx`/`sm`/O for its own rows and
    writes `out[(c*24 + h)*256 + d] = rne_bf16(acc/sm)` once, at the end.
- [ ] **Step 3: The K/V walk and causal trimming.** Block `b` covers positions
  `[64b, 64b+64)`. A work-group whose queries are `[pos+64t, pos+64t+63]` walks
  `b = 0 … (pos + 64t + 63)/64` - the **trimmed** range, so the whole
  upper-triangle above the diagonal block is never issued. On the **diagonal
  block only**, mask per element: `sc = (p <= pos + 64t + qt) ? sc : -INFINITY`,
  the same predicate `attn.cl:502` applies per position. Load the K/V block with
  `intel_sub_group_2d_block_read_16b_*` from the cache's `[position][4][256]`
  surface - the kv-head's 256 dims are contiguous and the position stride is
  1024 elements = 2048 B, a multiple of 64, so the alignment requirement holds
  for every block start.
- [ ] **Step 4: QKᵀ and PV on DPAS.**
  - QKᵀ: `acc = intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc)` with
    `a` = 4 rows × 16 dims of Q, `b` = 16 dims × 16 keys of K, accumulating over
    **16 k-steps** to reduce the 256-dim head. Result: a 4 × 16 fp32 score tile
    per DPAS column group, 4 column groups per 64-key block.
  - Scale **after** the reduction and **before** the max, the same order
    `attn.cl:502` uses: `sc[s] = dot * SCALE` with `SCALE = 0.0625f` (1/√256).
  - PV: downcast the 4 × 64 weights to bf16 in registers, then
    `intel_sub_group_bf16_bf16_matrix_mad_k16` with `a` = 4 rows × 16 keys,
    `b` = 16 keys × 16 dims of V, over 4 key-steps × 16 dim-tiles into the O
    accumulator.
- [ ] **Step 5: The online softmax - the SAME operation order as `attn_decode`.**
  Quote it in the kernel header and implement it word for word
  (`attn.cl:499-525`, and the identical statement in `attn_ref.h:36-50`):
  ```
  nmx  = max(mx, sc[0], sc[1], …, sc[15])          (ascending s)
  resc = exp(mx − nmx)                              (0 when mx = −INF)
  sc[s] = exp(sc[s] − nmx) ;  ssum = Σ_s sc[s]      (ascending s)
  sm   = fma(sm, resc, ssum)
  mx   = nmx
  t      = Σ_s fma(sc[s], f32(kv_v[p_s][j][d]), t)  (ascending s)
  acc    = fma(acc, resc, t)
  ```
  with the wave = **16 positions**, walked ascending inside the 64-position
  block, so the reassociation versus decode is only "which partials the DPAS
  accumulates internally", not the wave order. Carry
  `attn.cl:516-519`'s all-masked case verbatim: `nmx = −INF` → `resc = 1.0f`,
  `sc[] = 0.0f`, skip the wave whole, **because `exp(−INF − (−INF))` is a NaN
  and nothing else here is**. Plain `exp`, never `native_exp`. This is what
  gives the golden gate its best chance: everything except the dot's internal
  order matches the kernel the oracle already passes against.
- [ ] **Step 6: Build and read `zeinfo`.** One binary, no `M`/`MAXLEN`/`BLOCK`
  in the name (`pos`, `C` are runtime args; nothing is strided by `max_len`):
  ```cmake
  add_ocloc_kernel(pf_attn_chunk SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/prefill/pf_attn_chunk.cl)
  ```
  Then `tools/box.sh run 'ocloc compile -file src/kernels/prefill/pf_attn_chunk.cl
  -device bmg-g31 -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt"
  -output pf -out_dir /tmp && cat /tmp/pf.bin.zeinfo'` (or the dump the decode
  table used) and record `simd`, `slm_size`, `grf_count`, `barrier_count`,
  `private_size`, **`spill_mem_size`** and `has_dpas` - the same six columns
  docs/12's Task-5 table carries. **`has_dpas: true` is a hard requirement**: if
  it is absent the builtin did not lower and the kernel is a vector kernel
  wearing a DPAS costume. **Any `spill_mem_size` entry means Step 1's budget was
  wrong** - drop to `r = 2` (2 rows/subgroup, O = 32 dwords/lane) before
  measuring anything.
- [ ] **Step 7: The tile sweep, measured, not modelled.** This kernel's own file
  history is four dead cost models (`attn.cl:244-270`), so sweep rather than
  fit. Points: **packing factor ∈ {1, 2, 3, 6}** (q-heads sharing one K/V walk)
  × **Q-tile ∈ {64, 32}**, recomputing Step 1's budget per point and dropping
  any point that spills. Measure µs/layer-call at C = 4096, depth 0 and at
  C = 1024, depth 3072, on card 1, 8 replays / drop 3 (the `gemv_harness.h`
  convention). Require **byte-identical output across every point** - the tile
  is a scheduling choice, and if two tiles disagree in bits, one of them has a
  masking bug. Pick by measurement; record the whole table.
- [ ] **Step 8: Commit** -
  `feat(prefill): pf_attn_chunk - DPAS flash attention, <X> us/layer-call measured`

### Task 5: `tests/prefill/attn_chunk_test.cc` - the unit bar

**Files:**
- Create: `tests/prefill/attn_chunk_test.cc`
- Modify: `tests/CMakeLists.txt`
- Modify: `tests/kernels/attn_ref.h` - **additive only**: a
  `attn_ref::chunk_ref()` that composes the existing `decode` + `reduce`
  arithmetic per (q-head, query) **without materialising `attn_part`**

**Interfaces:**
- **Consumes:** `attn_ref::prep/decode/reduce` and every ordering statement in
  `attn_ref.h:22-60`; `common::bf16_to_f32` / `f32_to_bf16`;
  `attn_test.cc`'s comparison machinery (`compare_f32`, `require_ulp`,
  `require_bits16`, `bf16_key`) - copy the four helpers rather than exporting
  them, so `attn_test.cc` is not edited.
- **Produces:** a ctest target `attn_chunk_test` asserting Task 3 Step 5's bars.

- [ ] **Step 1: Why a new reference function is needed.** `attn_ref::decode`
  writes `attn_part` at `[24][max_len/64][M][258]`; at `C = 1024`,
  `max_len = 16384` that is `24 × 256 × 1024 × 258 × 4 B` = **6.5 GB on the
  host**. Add `attn_ref::chunk_ref(pos, C, max_len, q_bf16, kv_k, kv_v, out)`
  which, for each `(qh, c)`, runs the identical wave/block loop over blocks
  `0 … (pos+c)/64` accumulating one `(mx, sm, acc[256])` and finishes with
  `out[(c*24+qh)*256+d] = rne(acc[d]/sm)`. **It is the same arithmetic in the
  same order** as `decode` followed by `reduce` with a `+INF` gate - the merge
  of a single running state across blocks is `reduce`'s merge with `nb` steps -
  and the file's rule ("every block below has a twin in attn.cl and the two must
  be edited together") is honoured by writing it as a composition of the two
  existing loops, not a third transcription.
- [ ] **Step 2: The fixture.** Seed with xorshift (state the constant), fill the
  **whole** `kv_k`/`kv_v` cache - not just the prefix - so a read past the
  causal bound shows up as noise and not as a convenient zero
  (`attn_test.cc:457-462`), and fill `q` as bf16 in `[-2, 2]`. Upload once;
  run `attn_chunk` twice from identical inputs.
- [ ] **Step 3: The cases** - `C ∈ {1, 64, 256, 1024} × depth ∈ {0, 4096}`, eight
  runs at `max_len = 16384`:
  - `C = 1` is the degenerate chunk and the direct comparison against decode's
    geometry;
  - `C = 64` is exactly one Q tile and exactly one K block - the diagonal-only
    case, where every masked element is in the one trimmed block;
  - `C = 256` crosses four Q tiles;
  - `C = 1024` is the multi-chunk gate's width (spec §6.2) and, at depth 4096, the
    ragged case: 5120 keys = 80 blocks, the last Q tile walking all of them.
  Host cost is dominated by `C = 1024, depth 4096`: `1024 × 24 × 5120 × 256`
  = 3.2e10 MAC (derived) - **estimated ≤ 60 s single-threaded**; if it exceeds
  the test's budget, parallelise `chunk_ref`'s `(qh, c)` loop with OpenMP, which
  is legal because the rows are independent and each row's internal order is
  untouched.
- [ ] **Step 4: The bars** (Task 3 Step 5): **(a)** relative ≤ **8e-3** floored
  at the tensor RMS, **(b)** ≤ **2 bf16 ulp** on every element with
  `|ref| ≥ rms/8` - **(b) is the arbiter**. Print, per case, the worst relative,
  the worst ulp anywhere, the worst gated ulp, and how many of the
  `C × 6144` words differ at all, in `attn_test.cc:649-656`'s format. Do **not**
  assert the 1e-3 that `attn_test.cc:639` applies to `attn_part`: that is an
  fp32 bar and this tensor is bf16 (Task 3 Step 5 carries the argument).
- [ ] **Step 5: Determinism**, spec §6.4: the second run's output must be
  **bitwise identical** to the first (`std::memcmp`), for every case. For
  Branch A this is also the standing evidence that the chosen config has no
  atomics; for Branch B it is the same claim `attn_test.cc:668-673` makes.
- [ ] **Step 6: Register** in `tests/CMakeLists.txt` beside `attn_test`, with
  `add_dependencies(attn_chunk_test kernel_pf_attn kernel_pf_attn_chunk)`
  (Branch B) or the SYCL target (Branch A). `tools/box.sh test attn_chunk`.
- [ ] **Step 7: Commit** - `test(prefill): attn_chunk vs attn_ref at C in {1,64,256,1024}`

### Task 6: The swap - `Engine::prefill()` at `PrefillScratch::kC`

**Files:**
- Modify: `src/runtime/engine.cc` (the fa-layer branch of `prefill`),
  `src/runtime/buffers.h` / `buffers.cc` (`PrefillScratch`: `pf_q`, `pf_attn`,
  and the retirement of the temporary `attn_part`), `src/cli/b70_decode.cc`
  (the `--pp` default chunk, if 6b hard-coded 64)
- Test: `tests/runtime/buffers_test.cc` (totals + derivation),
  `tests/prefill/prefill_*_test.cc` (unchanged assertions, new width)

**Interfaces:**
- **Consumes:** 6b's `Engine::prefill` walk and `PrefillScratch`.
- **Produces:** a prefill fa-layer that is exactly three calls -
  `attn_prep_chunk` → `attn_chunk` → `attn_gate_chunk` - and a default
  `chunk == PrefillScratch::kC == 4096`.

- [ ] **Step 1: Replace the temporary path.** In `Engine::prefill`'s fa-layer
  branch, delete the decode-trio launches at `M = C` and write:
  ```cpp
  void* kk = at(buffers_.kv_k, size_t(f) * kv_stride_);
  void* vv = at(buffers_.kv_v, size_t(f) * kv_stride_);
  prefill::attn_prep_chunk(cx, layer, pos, c, qkv, fa_small, rope, pf_q, kk, vv);
  prefill::attn_chunk(cx, layer, pos, c, pf_q, kk, vv, pf_attn);
  prefill::attn_gate_chunk(cx, c, qkv, pf_attn, x_out);   // x_out is o_proj's input
  ```
  with `f = layer / 4` (`Qwen35::is_fa(l) == (l % 4 == 3)`, `capture.cc:110`).
- [ ] **Step 2: Delete the `C ≤ 64` clamp** wherever 6b put it (`Engine::prefill`,
  and the `--pp` path in `b70_decode.cc`), and make `chunk = 0` resolve to
  `PrefillScratch::kC`. Grep for the literal 64 in the prefill path and for any
  comment naming the temporary limitation; every one of them is retired in this
  commit, not left as a stale note.
- [ ] **Step 3: Retire the prefill `attn_part`.** It is `[24][max_len/64][C][258]`
  fp32 - **3.25 GB at C = 64** and **208 GB at C = 4096** (derived from
  `buffers.h:84`'s 50.72 MB at M = 1), which is why the temporary path was
  capped in the first place. Remove the allocation from `PrefillScratch`,
  update `scratch_bytes()` and the derivation comment, and update
  `tests/runtime/buffers_test.cc`'s asserted totals **with the arithmetic
  rewritten, not patched** (the discipline plan 5 Task 6 Step 3 states).
- [ ] **Step 4: Size the two new scratch buffers.** `pf_q` and `pf_attn`, bf16
  `[kC][24][256]` = **50.33 MB each** at kC = 4096; the gated output
  `[kC][6144]` bf16 = **50.33 MB**. Against spec §3.6's C = 4096 budget
  (weights 13.7 GB + KV 1.07 GB + dequant scratch 0.36 GB + activations
  ≤ 0.5 GB) these three are 0.15 GB of the 0.5 GB activation line - state the
  new activation total in the `PrefillScratch` comment and check it against the
  32656 MB the device reports (`docs/01-hardware.md:14`).
- [ ] **Step 5: Decode is untouched** - re-run `tools/box.sh test` and confirm
  `replay_determinism_test` still asserts **774 / 19** with no edit. If any of
  the three prefill kernels ended up inside the captured decode list, that is a
  bug in this task, not a ripple to absorb.
- [ ] **Step 6: Commit** -
  `feat(prefill): attention at C=4096 - the C<=64 path and its 3.25 GB attn_part retired`

### Task 7: The gates

**Files:**
- Modify: `tests/prefill/prefill_gate_test.cc` (flip the `C = 1024` multi-chunk
  row on; **no assertion changes**), `docs/14-golden-gate.md` (the prefill
  section 6b opened)

**Interfaces:**
- **Consumes:** 6b's `oracle-out-long`, the three prefill tests, the two
  checkpoints (RTN at `$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`;
  Vishva007 from the HF cache - `docs/14-golden-gate.md:787-793`).
- **Produces:** spec §6 items 1-4 green at C = 4096 and at C = 1024, and item 5
  re-verified.

- [ ] **Step 1: §6.1 - the golden gate after prefill**, both checkpoints, all
  three prompts, tie-aware semantics unchanged:
  ```
  tools/box.sh run './build/tests/prefill/prefill_gate_test "$PWD/oracle-out" \
    "$PWD/tests/golden/prompts" Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ'
  tools/box.sh run './build/tests/prefill/prefill_gate_test "$PWD/oracle-out-rtn" \
    "$PWD/tests/golden/prompts" $HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64'
  ```
  Determined rows 100% exact; tie rows set-membership; teacher-forced tails.
  A determined-row flip is **STOPPED and surfaced**, not tuned around.
- [ ] **Step 2: §6.2 - the multi-chunk gate at `C = 1024`** on `oracle-out-long`,
  which is the row this whole task exists for: at C = 1024 over a ≥ 2048-id
  prompt every chunk boundary is crossed in attention-over-cache, so it is the
  only test that exercises `attn_chunk` with a **non-zero `pos`** in anger -
  Task 5's `depth = 4096` cases prove the arithmetic, this proves the wiring.
  Flip it from capped/skipped to enforced; **its assertions do not change**.
- [ ] **Step 3: §6.3 - self-consistency** (`prefill_consistency_test`, unchanged):
  `prefill(P)` vs `ingest-by-decode(P)`, 64 generated tokens identical.
  Record `gdn_state`, `conv_ring`, the chunk's KV rows and the last hidden as
  max/mean relative difference - **diagnostic, not gate** (spec §6.3), but any
  growth versus 6b's recorded band **is a finding** and is reported, because
  L3 is the stage that changed the attention arithmetic.
- [ ] **Step 4: §6.4 - determinism** (`prefill_determinism_test`, unchanged):
  prefill twice from a reset state → bitwise-identical state buffers and tokens.
- [ ] **Step 5: §6.5 - decode untouched.** Re-measure the decode gate rows once
  at this plan's final sha, record-grade (idle box, DRM-fd evidence pasted,
  medians of 3): **32.22 t/s (RTN) / 29.33 (Vishva)** at `2a7df0b`. Inside day
  drift (≤ 0.09%) or the difference is explained. `-Wall -Wextra -Werror`
  green, `-cl-denorms-are-zero` absent (the CMake fatal proves it), the
  774/19 invariants unmoved.
- [ ] **Step 6: Commit** - `test(prefill): L3 gates - golden, multi-chunk C=1024, consistency, determinism`

### Task 8: Attribution and the records

**Files:**
- Modify: `docs/12-kernels.md` (a prefill-attention mechanism section),
  `docs/15-step-anatomy.md` (the prefill-step anatomy rows),
  `docs/07-open-questions.md` (anything this stage answers or opens)
- Modify (only if 6b built no attribution): `src/cli/b70_decode.cc`,
  `src/runtime/prefill/context.h`

**Interfaces:**
- **Consumes:** 6b's prefill attribution facility if it exists; Task 3's
  pre-registered price.
- **Produces:** the attention family's share of the prefill step, measured, and
  the two doc sections spec §9 requires in the same commit as the mechanism.

- [ ] **Step 1: The instrument.** If 6b built per-launch event timing for the
  prefill path, use it. If not, build the **drain-bracketed** one: a `--pp-profile`
  flag that calls `cx.wait()` before and after each family and accumulates host
  wall time. It is legitimate here and not in decode, because prefill kernels
  are **millisecond-class** (spec §3.6: "launch overhead is irrelevant at
  millisecond kernel scale") - but it is perturbing, so **measure the
  perturbation**: time an empty drain in the same loop and quote it as the
  instrument's floor, the way `docs/15` §5.4 quotes the profiler's.
- [ ] **Step 2: Measure** `attn_prep_chunk` / `attn_chunk` / `attn_gate_chunk`
  per layer-call at C = 4096 depth 0 and at C = 1024 depths {0, 1024, 2048,
  3072}, card 1, and roll them up into the attention family's **share of the
  prefill step** beside the GEMM, dequant and GDN families.
- [ ] **Step 3: Judge against Task 3's pre-registration** - write the predicted
  and measured numbers side by side, with the ratio, and say plainly which model
  died if it did. Convert to TFLOP/s with Step 2's FLOP formula and compare to
  P2's measured XMX rate (or, if P2 has not landed, to the derived 183.5 and
  labelled as such).
- [ ] **Step 4: docs/12** - a `pf_attn` section in the shape of the `attn`
  chapter: the three entry points and their grids, the **q-in-bf16** change and
  why it moves towards the oracle, the causal trimming + diagonal masking rule,
  the online-softmax order (quoted, and stated to be `attn_decode`'s), the
  branch actually built and its config, the `zeinfo` row (Branch B) or the
  no-atomics evidence (Branch A), the SLM and register budget, and the measured
  µs/layer-call table. State explicitly that the **KV load path, 55.4% of the
  decode launch** (`docs/15`, measured), is **not** the term here - Task 3
  Step 3's 23× compute-bound ratio is the reason and it is the single sentence a
  future reader most needs.
- [ ] **Step 5: docs/15** - prefill-step anatomy rows for the three kernels with
  their per-call µs and family share, the instrument named and its floor quoted,
  and the drift control (an untouched family's before/after) so the attention
  numbers are readable against noise the way §L5 and Task 5's rollups are.
- [ ] **Step 6: Commit** -
  `docs(kernels): prefill attention mechanism and attribution - <X>% of the chunk`

### Task 9: L3 close

**Files:**
- Modify: `docs/BENCHMARKS.md` (the prefill rows gain the L3 checkpoint),
  `docs/05-perf-model.md` (the prefill verdict line)
- Create (short path only):
  `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/l3-attention-memo.md`

- [ ] **Step 1: The stage acceptance** (spec §5, "Per-stage acceptance"): full
  suite green; the §6 bars on this stage's checkpoint (Task 7); the
  `--profile`-style attribution (Task 8); docs/12 mechanism in the same commit
  (Task 8). Any miss → **revert + priced record**, not a patch.
- [ ] **Step 2: Record the device-side pp row** at the L3 sha, record grade
  (idle box, medians of 3, DRM-fd evidence): `b70-decode <ckpt> --bench
  --pp 4096 --tg 256` on the RTN checkpoint, reported with **both labels
  attached** (device-side vs vLLM's HTTP-inclusive **1973 t/s**, chunk width,
  checkpoint) - spec §2. This is not the spec gate (§7 is, and it runs after
  L1-L3 are all in); it is L3's row in the ladder.
- [ ] **Step 3: If the attention family missed Task 3's pre-registered price**,
  write the memo: predicted vs measured per term, which model died, whether the
  miss is tile-shaped (sweep more) or path-shaped (Branch A ↔ Branch B), and
  what it costs the composed ceiling. STOP at the memo; nothing further starts
  without a ruling.
- [ ] **Step 4: Commit** - `feat(bench): L3 attention - <X> ms of the prefill chunk, <verdict>`

---

## Plan self-review (2026-09-04, at authoring)

**Spec coverage.**
- **§3.3** (prefill attention) → the whole plan. "The chunk's K/V are written
  into the existing cache layout by a widened `attn_prep`" → Task 1.
  "`sycl-tla` FMHA forward first, if it instantiates at head_dim 256 - Stage 0
  checks" → Task 3 Step 1 reads P4, Task 4A builds it. "Fallback, ours, only if
  P4 fails … with a pre-registered price" → Task 4B, priced in Task 3 Step 4.
  "Task 5's register-packed GQA transfers directly" → Task 4B Step 1, where the
  register arithmetic shows it transfers **partially** at head_dim 256 (2 heads,
  not 6) and Step 7 sweeps the factor. "Flash-style streaming is mandatory from
  the smallest chunk" → Task 4B Steps 2-3 (no S×S score block ever exists; the
  score tile is 4 rows × 64 keys in registers). "~3% of prefill FLOPs" →
  Task 3 Step 2, reconciled to 1.67% executed.
- **§5 L3** ("FMHA (P4) or our flash kernel; retires the `C ≤ 64` limitation.
  **Gate:** correctness bars + attention-family attribution") → Tasks 4A/4B
  (the kernel), Task 6 (the retirement), Task 7 (the bars), Task 8 (the
  attribution). Per-stage acceptance → Task 9 Step 1.
- **§6 item 1** (golden gate after prefill, both checkpoints) → Task 7 Step 1.
  **Item 2** (multi-chunk golden gate at C = 1024 on `oracle-out-long`) →
  Task 7 Step 2. **Item 3** (self-consistency control, diagnostic) → Task 7
  Step 3. **Item 4** (determinism, no fp atomics, no split-K atomics) →
  Task 5 Step 5, Task 7 Step 4, Task 4A Step 2, Global Constraints.
  Items 5 and 6 are carried in Global Constraints and Task 7 Step 5.
- **§8** reproduced verbatim above. **§9 records** → Task 8 (docs/12, docs/15)
  and Task 9 (docs/BENCHMARKS, docs/05).
- **§10 out of scope**: no task touches the tokenizer, MTP, prefix caching, fp8
  KV, or an OpenCL C DPAS *GEMM* - Branch B is an attention kernel, which §3.3
  names explicitly as the P4-fallback work item, and it is not the §10-excluded
  general GEMM.

**Placeholder scan.** No TBD/TODO/"similar to". `<X>`, `<verdict>`,
`<causal form>` in commit messages are measurement outputs, as in plan 5.
Both branches are written to the same level of concreteness (files, grids,
register budgets, exact call arguments, exact bars); the executor builds one.
The two ruling requests are stated with their fallbacks, so neither is a
blocking unknown. The one genuinely unresolved external fact - the exact
`Arguments` field names in `sycl-tla`'s FMHA runner - is resolved by Task 4A
Step 1 reading the header the probe already compiled against, which is the
correct place for it: P4's working instantiation is the source of truth, not
this document.

**Type consistency vs interfaces.md.** `attn_prep_chunk` and `attn_chunk` keep
their declared names, namespace (`runtime::prefill`), parameter order and
types; `attn_prep_chunk` gains two `const float*` parameters under request R2
and `attn_gate_chunk` is new under R1 - both filed, neither silent. Kernels
carry the `pf_` prefix and live in `src/kernels/prefill/`; SYCL lives in
`src/sycl/`; host code lives in `src/runtime/prefill/` - all four as
interfaces.md fixes them. `M` is a **runtime** argument everywhere on this
path: no `_M{C}`, `_L{MAXLEN}` or `_B{BLOCK}` suffix appears in any new variant
name, and each new kernel is one binary. Activation layouts match the contract
(`q_out` bf16 `[C][24][256]`, `out` bf16 `[C][24][256]` pre-gate,
`x_out` bf16 `[C][6144]`); the persistent `kv_k`/`kv_v` `[pos][4][256]` layout
is read and written in place and is never relayouted. Test names match
(`tests/prefill/attn_chunk_test.cc`); `attn_prep_chunk_test.cc` is an addition
in the same directory and idiom. The vLLM bar (1973 t/s), today's 121 s, the
per-token 48.97 GFLOP and the derived 183.5 TFLOPS are quoted exactly as
interfaces.md's "Numbers every plan quotes the same way" states them.

**Known tension, ruled here.** Interfaces.md fixes `q_out` as bf16 while decode
keeps `attn_q` fp32. That is a real numerics difference between the two paths
and the self-consistency control (§6.3) will see it in the state diagnostics.
It is taken deliberately: DPAS requires bf16 operands, and torch's own
attention carries q in bf16, so the prefill path is the one that matches the
oracle's dtype. It is recorded in docs/12 (Task 8 Step 4) and named in Task 7
Step 3 as an expected, explained contributor to the diagnostic band - not as
drift to be discovered later.

Claude-Session: 
