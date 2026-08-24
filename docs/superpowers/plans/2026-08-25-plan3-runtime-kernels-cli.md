# Plan 3 of 3 - Runtime, decode kernels, CLI, golden gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish spec 1. The captured Level Zero decode list replayed per
token, the five remaining kernel families, `b70-decode` with `--bench`, the
replay-determinism proof, and the golden gate against the oracle files -
ending with a tg256 number written beside vLLM's 31.50 in
`docs/BENCHMARKS.md`.

**Architecture:** The runtime walks `model::Qwen35::layers()` +
`loader::LoadedModel` once at startup and appends ~650 kernels into ONE
regular in-order command list; per token the host does
`zeCommandQueueExecuteCommandLists` + fence wait + one 4-byte read. All
per-token variability lives in a device-resident control block and in
device-side early-outs; nothing in the list is ever mutated (spec §8).
Kernels are OpenCL C via `ocloc` (`bmg-g31`), fp32 accumulation, no atomics,
fixed grids. Small-tensor field offsets come from `loader/small_layout.h` -
the kernels' binding code includes it and re-derives nothing.

**Tech Stack:** C++17, `g++`, existing CMake + `add_ocloc_kernel`; OpenCL C
3.0 with `cl_intel_subgroups`; Level Zero via `b70_l0`; Python only inside
the container for the (already-generated) oracle files.

**Spec:** `docs/superpowers/specs/2026-08-22-phase0-decode-core-design.md`
§3 (definition of done, items 2-4), §8 (runtime), §9 (kernels - as amended:
`prep` kernel, 645-kernel sequence), §11 (tests), §12 (CLI/bench), §13
(risks). Plans 1-2 delivered everything these tasks consume: measured GEMV
variants, the loader (`LoadedModel`), the model table, and three golden
oracle files on the box.

## Global Constraints

- C++17, `-Wall -Wextra -Werror`; host links `ze_loader` only (via `b70_l0`).
- OpenCL C 3.0; every kernel `intel_reqd_sub_group_size(16)`; fp32
  accumulation; bf16 as `ushort << 16`. **Never** compile any kernel with
  `-cl-denorms-are-zero` or any FP16 denorm-flush option - the real
  checkpoint has 1 658 subnormal f16 scales (docs/13-loader.md, 2026-08-25).
- Determinism: no floating-point atomics, no work-stealing, no
  data-dependent work-group counts; reductions fixed-tree or two-stage. The
  replay-determinism test (Task 6) is the acceptance for every kernel: a
  kernel that breaks bitwise replay is rejected, whatever else it does.
- Interfaces consumed (from plans 1-2, all verified in-tree at `49d0ef7`):
  `loader::LoadedModel{linears[(layer,LinearId)], layer_small[64]{norms,gdn},
  embed, final_norm(fp32 1+w [5120]), rope(fp32 [max_len][2][32]), max_len,
  report}` with `loader::kTopLevel` keying `lm_head`;
  `loader/small_layout.h` offsets (norms 40960 B fp32; GDN 164480 B: conv
  fp32[10240][4]@0, negA fp32[48]@163840, dt_bias fp32[48]@164032,
  gated-norm bf16[128]@164224; FA 2048 B: q_norm@0, k_norm@1024 fp32);
  `model::Qwen35` (`layers()`, `linear(id)`, `shape(id)`, `LinearId::kCount`,
  constants incl. `kVocabUsed = 248077`); `kernels::gemv_variant/gemv_bf16_variant/path`;
  `l0::` wrappers (immediate list is synchronous; `Fence::wait()` syncs+resets;
  arguments are captured at append - proven by `arg_capture_test`).
- **RMSNorm-family weights are fp32 `1+w` on device; the gated GDN norm is
  bf16 plain `w`** (plan-2 ruling). Kernels reading them must use the right
  dtype per `small_layout.h`.
- Device selection (doc 04 §Device selection, 2026-08-25): `--device N` >
  `ONEAPI_DEVICE_SELECTOR=level_zero:N` > default 0; `level_zero:*` → 0;
  `ZE_AFFINITY_MASK` respected underneath (indices are the masked view).
  Implemented in Task 1; SYCL interop rule is spec-2 scope.
- Layer math is doc 03 "Layer math - verified in the modeling file"; the
  oracle files embody it. Where this plan's kernel text and doc 03 disagree,
  doc 03 wins and the plan is corrected.
- Kernel-count reality: 0.52 µs/kernel measured (doc 07 #5); the unfused
  ~650-kernel list costs ~0.4 ms of a ~26 ms step - fusion is explicitly out
  of scope for this plan (spec §4.1 decision).
- Builds/tests on the box via `tools/box.sh` (`JOBS=44`; whole-second-mtime
  recompile trap - verify compile lines). Checkpoint-dependent tests carry
  ctest label `checkpoint`; golden-file-dependent tests carry label `golden`
  (both box-only).
- Every kernel lands with its `docs/12-kernels.md` section (computes, work
  assignment + why, rejected alternatives with honesty about what was not
  measured, measured number) - the project rule; a kernel task is not done
  without it.
- Prefix caching is OUT (queued after spec 3 - memory note). MTP machinery:
  the `M` loop exists in every kernel; only `M = 1` is compiled and tested.
- Commit after every task; `feat:`/`test:`/`fix:`/`docs:`.

---

### Task 1: Runtime scaffolding - control block, buffers, device selection

**Files:**
- Modify: `src/l0/context.h`, `src/l0/context.cc` (env-var default selection)
- Create: `src/runtime/control.h`, `src/runtime/buffers.h`, `src/runtime/buffers.cc`, `src/runtime/CMakeLists.txt`
- Modify: root `CMakeLists.txt` (add `add_subdirectory(src/runtime)` after `src/model`)
- Create: `tests/l0/device_select_test.cc`, `tests/runtime/buffers_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- `l0::Context`: new `static constexpr uint32_t kFromEnv = 0xFFFFFFFFu;` and
  `explicit Context(uint32_t device_index = kFromEnv)`. `kFromEnv` resolves
  via new `static uint32_t device_index_from_env();` which parses
  `ONEAPI_DEVICE_SELECTOR`: unset or `level_zero:*` → 0; `level_zero:N` → N;
  anything else (another backend, a list, garbage) → throw
  `std::runtime_error` naming the value and the two accepted forms. Existing
  callers that pass `0` explicitly are untouched (probes/tests keep binding
  device 0). The header comment documenting "ONEAPI_DEVICE_SELECTOR is not
  consulted" is replaced by the new contract + a note that `ZE_AFFINITY_MASK`
  filters enumeration beneath us (indices = masked view).
- `runtime::Control` - the device-resident per-token state (spec §8.2 as
  corrected here: the spec's field list needs 96 B, not the 64 B its comment
  claimed; we round up to two cache lines):

```cpp
#pragma once
#include <cstdint>

namespace runtime {
// The ONLY mutable state the host and the captured list share. Lives in
// zeMemAllocShared memory; every kernel that needs position or the current
// token reads it; argmax_stage2 writes out_token/cur_token and advances pos.
// The host writes cur_token[0] before a replay only during prompt ingestion,
// and reads out_token[0] after the fence. Nothing else moves per token.
// 128 B = two cache lines (the spec's §8.2 sketch said 64 B but its own
// fields need 96 - corrected here, noted in the spec by Task 6).
struct Control {
  uint32_t pos;            // KV slot / position of token 0 of this step
  uint32_t n_active;       // tokens in this step (1 in this plan)
  uint32_t cur_token[8];   // input ids for this step
  uint32_t out_token[8];   // argmax outputs
  uint32_t debug_flag;     // check_finite writes first bad layer+1 here (debug builds)
  uint32_t pad[13];        // pad to 128 B
};
static_assert(sizeof(Control) == 128, "control block is two cache lines");
}  // namespace runtime
```

- `runtime::DecodeBuffers` - every device allocation the decode list touches
  besides weights, sized from `model::Qwen35` constants + `max_len`:

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include "l0/context.h"
#include "l0/memory.h"
#include "model/qwen35.h"

namespace runtime {
// All decode-step state and scratch. Allocated once; every pointer is baked
// into the captured list. M = 8 capacity from day one (spec §6.5) even
// though this plan compiles M = 1 kernels only.
struct DecodeBuffers {
  static constexpr uint32_t kM = 8;
  static constexpr uint32_t kConvRing = 16;     // ring depth >= M + 3 (spec §9.4)
  static constexpr uint32_t kAttnBlock = 256;   // KV positions per attn_decode work-group

  DecodeBuffers(l0::Context& ctx, uint32_t max_len);

  // --- persistent state (survives across tokens) ---
  l0::Mem control;        // shared, sizeof(Control)
  l0::Mem gdn_state;      // fp32 [48 layers][48 heads][128 k][128 v]  = 150.99 MB
  l0::Mem conv_ring;      // bf16 [48 layers][16][10240]               = 15.73 MB
  l0::Mem kv_k, kv_v;     // bf16 [16 layers][max_len][4][256] each    = 536.87 MB each @16384
  // --- per-step scratch (overwritten every token) ---
  l0::Mem resid;          // bf16 [M][5120]
  l0::Mem x;              // bf16 [M][17408]  (prep output; largest K)
  l0::Mem partials;       // fp32 [16][M][34816] (max S x max N)       = 17.83 MB
  l0::Mem ab_out;         // fp32 [M][128]    (a||b GEMV output, S=1)
  l0::Mem gdn_o;          // fp32 [M][48][128] (gdn_step output, pre gated-norm)
  l0::Mem attn_q;         // fp32 [M][24][256] (post norm+rope)
  l0::Mem attn_gate;      // fp32 [M][24][256]
  l0::Mem attn_part;      // fp32 [24][max_len/256][M][258] (m, l, acc[256]) = 12.68 MB @16384
  l0::Mem attn_out;       // bf16 [M][6144]
  l0::Mem logits;         // fp32 [M][248320] = 7.95 MB
  l0::Mem argmax_part;    // fp32+idx pairs, stage-1 output: [M][243][2]
  uint32_t max_len;

  size_t persistent_bytes() const;   // printed at startup, asserted by the test
  size_t scratch_bytes() const;
};
}  // namespace runtime
```

  `buffers.cc` computes each size from the constants (no literals except the
  table in one place), allocates device memory (control = shared), and
  provides the two byte totals. Expected @ max_len 16384, M 8: persistent ≈
  150.99 + 15.73 + 2×536.87 + 0.000128 GB ≈ **1240.5 MB**; scratch ≈ **44 MB**
  (dominated by partials 17.8 + attn_part 12.7 + logits 7.9). The test pins
  the exact numbers.

- [ ] **Step 1: Failing tests.** `tests/l0/device_select_test.cc` (host-only,
  no GPU): `setenv`/`unsetenv` around `l0::Context::device_index_from_env()` -
  unset → 0; `level_zero:*` → 0; `level_zero:1` → 1; `level_zero:01` → 1;
  `opencl:0` → throws (message contains the value); `level_zero:0;opencl:*`
  → throws; `level_zero:x` → throws. `tests/runtime/buffers_test.cc` (GPU):
  construct `DecodeBuffers(ctx, 16384)`, `CHECK_EQ` the two byte totals
  against the hand-computed constants above (exact numbers in the test),
  write/read the control block through `Mem::as<Control>()`, and re-zero.
  CMake entries mirror existing patterns (`device_select_test` links `b70_l0`
  only and needs no kernels; `buffers_test` links `b70_l0 b70_model` + a new
  `b70_runtime`).
- [ ] **Step 2: Run to verify both fail; implement; green.** `-Werror` note:
  `setenv` is POSIX - fine (Linux-only project).
- [ ] **Step 3: Full suite (15/15 expected: 13 + these 2); commit** -
  `feat(runtime): control block, decode buffers, ONEAPI_DEVICE_SELECTOR-aware device selection`

---

### Task 2: `prep.cl` - the three between-GEMV kernels

**Files:**
- Create: `src/kernels/prep.cl`; Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`
- Create: `tests/kernels/prep_ref.h`, `tests/kernels/prep_test.cc`; Modify: `tests/CMakeLists.txt`

**Rounding discipline (binding for this and every later kernel task).** The
oracle is torch, and torch rounds **per op**: every linear's output is bf16,
every elementwise op widens to fp32 internally and rounds its result back to
bf16. To keep the engine's residual stream comparable to the golden tensors,
this plan matches that discipline wherever it is cheap:

- a GEMV's summed partials are rounded to bf16 **once** (that is the linear's
  bf16 output in the reference) before any further use;
- the residual add is `rne_bf16(f32(resid) + f32(mixer_b))` - bf16 in, bf16
  out, exactly torch's bf16 add;
- norms compute in fp32 from the widened bf16 input, multiply by the fp32
  `(1+w)` weight, and round the result to bf16 (the reference's `type_as(x)`);
- inside-op accumulation (dot products, variance sums) stays fp32 and is NOT
  matched term-for-term - that residual drift is what the golden gate's
  design absorbs (tokens exact = gate; tensor cosines = diagnostics).

Where a kernel deviates from this discipline it must say so in its
`docs/12-kernels.md` section.

**Kernels (one file, three entry points; variants by `-D`):**

```
prep_res_norm<M, K=5120, S_PREV∈{0,16}>(
  const float*  partials,   // fp32 [S_PREV][M][K]; unused when S_PREV==0
  ushort*       resid,      // bf16 [M][K], read + written in place
  const float*  norm_w,     // fp32 (1+w) [K]
  ushort*       x_out)      // bf16 [M][K]
```
One work-group of 256 per `m` (grid = (1, M)); the whole row is staged in SLM
as fp32 (5120·4 = 20 KB). Algorithm per m: `mixer_b = rne_bf16(Σ_s partials)`
per element (skip when S_PREV==0); `r_b = rne_bf16(f32(resid) + f32(mixer_b))`
(S_PREV==0: `r_b = resid`); store `r_b` back to `resid`; work-group tree-sum
of `f32(r_b)²` in SLM (fixed tree - 256→128→…→1, deterministic);
`rstd = rsqrt(mean + 1e-6f)`; `x_out = rne_bf16(f32(r_b) · rstd · norm_w[k])`.
Barriers between phases. Compiled variants: `(M=1, S_PREV=0)` and
`(M=1, S_PREV=16)`.

```
prep_silu_mul<M, S_PREV=4>(
  const float*  partials,   // fp32 [S_PREV][M][34816], gate||up interleaved 16-col blocks
  ushort*       x_out)      // bf16 [M][17408]
```
Grid = (ceil(17408/4096)=5 chunks, M), WG 256, no reduction. For output `k`:
`gflat = (k/16)*32 + k%16`, `uflat = gflat + 16`;
`g_b = rne_bf16(Σ_s partials[s][m][gflat])`, `u_b` likewise;
`s_b = rne_bf16(silu_f32(f32(g_b)))` with `silu(x) = x / (1 + exp(-x))`;
`x_out[k] = rne_bf16(f32(s_b) · f32(u_b))`. (Matches torch's op chain:
linear→bf16, silu per-op, mul per-op.)

```
prep_gated_head<M, S_PREV=1>(
  const float*  qkvz_partials, // fp32 [S_PREV][M][16384]; z at columns 10240 + h*128 + i
  const float*  gdn_o,         // fp32 [M][48][128] from gdn_step
  const ushort* gated_w,       // bf16 plain w [128] (small block, kGdnOffGatedNorm)
  ushort*       x_out)         // bf16 [M][6144]
```
Grid = (48, M), WG 128 (8 subgroups). Per (m, h): `o_b[i] =
rne_bf16(gdn_o[m][h][i])` (the reference's recurrence output is cast to bf16
before the gated norm); `z_b[i] = rne_bf16(Σ_s qkvz_partials[s][m][10240+h*128+i])`;
variance = tree-sum of `f32(o_b)²` / 128; `n_b[i] = rne_bf16(f32(o_b[i]) ·
rsqrt(var + 1e-6f))`; `t_b[i] = rne_bf16(f32(gated_w[i]) · f32(n_b[i]))`;
`x_out[m][h*128+i] = rne_bf16(f32(t_b[i]) · silu_f32(f32(z_b[i])))`. This is
exactly `Qwen3_5RMSNormGated`'s op chain (norm → cast → ×w → ×silu(z.float())
→ cast), doc 03.

**Host-side names** (`kernels.h`): `prep_res_norm_variant(M, K, S_PREV)`,
`prep_silu_mul_variant(M)`, `prep_gated_head_variant(M)` →
`"prep_res_norm_M1_K5120_SP16"` etc. CMake compiles the four variants named
above (all M=1) plus `prep_res_norm` at M=2 (compile-only, per the standing
M-loop rule).

- [ ] **Step 1: Failing test.** `tests/kernels/prep_ref.h`: CPU references
  implementing exactly the op chains above (using `common::f32_to_bf16` /
  `bf16_to_f32`; double for the tree sums is NOT used - fp32, same order as
  the kernel's fixed tree, so the comparison can be exact-or-1ulp).
  `tests/kernels/prep_test.cc`: for each kernel, random inputs (partials
  ~N(0,1), resid bf16, norm_w near 1.0), run device vs reference:
  `prep_res_norm` S_PREV∈{0,16} - `x_out` and updated `resid` must match
  **bit-exactly** (every op is a rounded scalar chain; only the variance tree
  could differ, and the reference reproduces the kernel's exact tree order -
  state this in a comment); `prep_silu_mul` and `prep_gated_head` bit-exact
  likewise except `silu` (uses `exp`) - tolerance 2 ulp bf16 on the final
  value for those, exact for the norm-only outputs. CMake mirrors gemv_test
  (kernel deps: the four variants).
- [ ] **Step 2: RED (missing kernels) → implement `prep.cl` → GREEN.** If
  bit-exactness fails only in the variance tree, make the reference match
  the kernel's tree order rather than loosening (the point is a pinned
  contract, as with the dequant fixture).
- [ ] **Step 3: `docs/12-kernels.md` section** - the rounding-discipline
  rationale (why per-op bf16 matching, what stays fp32), work assignment,
  the SLM staging bound (20 KB), rejected: fusing into GEMV prologue (plan-1
  §9.2 history - SLM and partials-traffic arithmetic, measured 0.52 µs/kernel
  making the separate kernel affordable).
- [ ] **Step 4: Full suite; commit** - `feat(kernels): prep kernels (res+norm, silu-mul, gated head) with torch-matched rounding`

---

### Task 3: `embed_gather.cl` + `argmax.cl`

**Files:**
- Create: `src/kernels/embed_gather.cl`, `src/kernels/argmax.cl`; Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`
- Create: `tests/kernels/argmax_test.cc`, `tests/kernels/embed_gather_test.cc`; Modify: `tests/CMakeLists.txt`

**Kernels:**

```
embed_gather<M>(const uint* ctrl, const ushort* embed, ushort* resid)
```
Grid (1, M), WG 256. `row = ctrl[2 + m]` (Control::cur_token[m] - the kernel
indexes the struct as a uint array; `kernels.h` records the field indices as
constants shared with `runtime/control.h` via a comment cross-reference AND a
`static_assert` in `capture.cc` comparing `offsetof` values). Copies
`embed[row*5120 .. +5120)` bf16 into `resid[m]`. 5120/256 = 20 elements per
work-item, coalesced.

```
argmax_stage1<M>(const float* logits, float* part)
```
Grid (243, M), WG 256: work-group `g` scans logits[m][g*1024 .. g*1024+1024)
(tail group masks ≥ 248320); a candidate with `idx >= 248077`
(`kVocabUsed` - baked as a `-D` constant) is treated as `-INFINITY`; SLM tree
reduce with the deterministic comparator `(a.v > b.v) || (a.v == b.v && a.i < b.i)`;
writes `(max, idx_as_float)` to `part[m][g][0..1]`.

```
argmax_stage2(uint* ctrl, const float* part)   // grid (1,1); loops m internally
```
One WG of 256: for `m < ctrl.n_active`: tree-reduce the 243 stage-1 pairs
(same comparator; lanes ≥ 243 seeded with −INF), write `ctrl.out_token[m]`;
after the loop, lane 0 sets `ctrl.cur_token[0] = out_token[n_active−1]` and
`ctrl.pos += n_active`. This kernel is the ONLY writer of `pos`/`cur_token`
inside the list.

- [ ] **Step 1: Failing tests.** `argmax_test.cc`: cases - unique max mid-range;
  exact tie at two indices (expect lowest); max located in the masked tail
  ≥ 248077 (expect the best sub-248077 index); all-equal (expect 0); max at
  index 248076 (boundary). Each case uploads fp32 logits, runs both stages
  via a small regular list, reads `ctrl.out_token[0]` and checks `pos`
  advanced by `n_active`. `embed_gather_test.cc`: random embed rows, two
  different `cur_token` values across two replays of the same closed list
  (this doubles as an arg-capture-style check that the kernel *reads ctrl at
  execution time*, unlike kernel arguments which are captured at append -
  the property the whole design rests on; say so in the test comment).
- [ ] **Step 2: RED → implement → GREEN.**
- [ ] **Step 3: docs/12 sections** (argmax: why two fixed stages and not
  atomics - determinism; the 248077 mask and where the number comes from;
  embed_gather: why it reads ctrl rather than taking the id as an argument).
- [ ] **Step 4: Full suite; commit** - `feat(kernels): embedding gather + deterministic two-stage argmax with vocab mask`

---

### Task 4: `gdn_step.cl` - the gated-delta-rule decode step

**Files:**
- Create: `src/kernels/gdn_step.cl`; Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`
- Create: `tests/kernels/gdn_ref.h`, `tests/kernels/gdn_step_test.cc`; Modify: `tests/CMakeLists.txt`

This is the model's original-work kernel: 48 of 64 layers, no `sycl-tla`
example. The math is doc 03 "Layer math" (a transcription of
`torch_recurrent_gated_delta_rule` + `causal_conv1d_update`); the SYCL
reference for lane-assignment ideas is
`~/PycharmProjects/vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp`
(read for structure, lift nothing verbatim - it is torch-coupled).

**Kernel contract:**

```
gdn_step<M>(
  const uint*   ctrl,           // Control as uint[]: pos at [0], n_active at [1]
  const float*  qkvz_partials,  // fp32 [1][M][16384] (qkv||z GEMV output, S=1)
  const float*  ab_out,         // fp32 [M][128]: a at [0..48), b at [48..96)
  const float*  gdn_small,      // the layer's GDN block (small_layout.h offsets):
                                //   conv fp32 [10240][4] @0, negA fp32[48] @163840/4,
                                //   dt_bias fp32[48] @164032/4  (gated norm NOT read here)
  ushort*       conv_ring,      // bf16 [16][10240], this layer's ring
  float*        state,          // fp32 [48][128][128], this layer's S (k-major rows, v columns)
  float*        gdn_o)          // fp32 [M][48][128]
```

Grid = (48 heads, 4 chunks), WG 256 (16 subgroups × 16). Work-group `(h, c)`
owns state columns `[32c, 32c+32)` of head `h`. Per work-group:

1. **Head scalars** (each `m`): `a_b = rne_bf16(ab_out[m][h])`,
   `b_b = rne_bf16(ab_out[m][48+h])` (the linear's bf16 outputs);
   `g = negA[h] · softplus_f32(f32(a_b) + dt_bias[h])`,
   `beta = 1/(1+exp(−f32(b_b)))` - fp32, matching the reference's
   `.float()` path (softplus: `x > 20 ? x : log1p(exp(x))`).
2. **Conv + SiLU for this WG's channels, all m of this step.** Needed
   channels: q = k-head `h/3` → flat `[h/3·128, h/3·128+128)`; k → `2048 +`
   same range; v → `4096 + [h·128, h·128+128)`; 384 channels per WG. For each
   channel `ch` and token `m`: `raw_b[m] = rne_bf16(qkvz_partials[0][m][ch])`
   (S=1: the sum is one term; still round - it is the linear's bf16 output);
   window = ring slots `(pos+m−3 … pos+m−1) mod 16` for the entries older
   than this step, and this step's own `raw_b[0..m−1]` for the newer ones
   (`pos` from ctrl; slots < 0, i.e. `pos+m−j < 0`, contribute 0 - matches
   the reference's zero-initialised conv state); `conv = Σ_{t=0..3}
   w[ch][t] · f32(window_t)` with `window_3` = current token; then
   `x_b = rne_bf16(silu_f32(conv))`. Ring depth 16 ≥ M+3 guarantees this
   step's writes never land on slots any WG still reads (plan-1 §9.4 rule).
3. **Ring write (ownership):** chunk `c==0` writes `raw_b[m]` into
   `conv_ring[(pos+m)%16][ch]` for the v channels of head `h`; the WG with
   `h%3==0 && c==0` also writes the shared q and k channels of k-head `h/3`.
   Values are identical wherever recomputed; write-after-compute within the
   WG; cross-WG readers only touch pre-step slots.
4. **l2norm + scale** per m: over the 128-wide q and k (fp32 from the bf16
   conv outputs): `qn_b[i] = rne_bf16(f32(q_b[i]) · rsqrt(Σ q² + 1e-6f))`
   (subgroup+SLM fixed-tree sum), same for k; then `qf[i] = f32(qn_b[i]) ·
   rsqrt(128.0f)` - wait, the reference scales by `1/√d` BEFORE the
   recurrence in fp32: `query = query * scale` after `.to(float32)`; so
   `qf[i] = f32(qn_b[i]) · 0.08838834764831845f` (1/√128). k/v widen to fp32
   unscaled.
5. **Recurrence** (fp32, doc 03): the WG's state slice `[128 k][32 v]` loads
   into registers (16 fp32/WI); for m in 0..n_active−1:
   `S *= exp(g)`; `kv[v] = Σ_k S[k][v]·kf[k]` (per-WI over its columns);
   `Δ[v] = (vf[v] − kv[v])·beta`; `S[k][v] += kf[k]·Δ[v]`;
   `o[v] = Σ_k qf[k]·S[k][v]` → `gdn_o[m][h][32c+…]` fp32. Store S back once.
6. No barrier crosses work-groups; the gated norm happens in
   `prep_gated_head` (Task 2).

Compiled variants: M=1 (used), M=2 (compile-only).

- [ ] **Step 1: Failing test.** `tests/kernels/gdn_ref.h`: a C++ port of the
  reference chain for ONE layer step - conv update (zero-init state), silu,
  l2norm, scaled recurrence - with the same per-op bf16 roundings the kernel
  specifies (use `common::` helpers; fp32 accumulation in the same fixed
  order as the kernel's trees where reductions occur - document the tree
  order in a comment). `tests/kernels/gdn_step_test.cc`: random
  qkvz-partials/ab/state/ring (seeded), cases: `pos=0` (empty history),
  `pos=5` (ring partially filled by the test), each M=1; compare `gdn_o`
  (tol: rel ≤ 1e-3 vs the fp32 reference - exp/softplus/rsqrt differ by
  ulps between device and host libm), updated `state` (rel ≤ 1e-5), and the
  ring slots written (bit-exact - they are rounded linear outputs). Grid a
  second identical run to confirm determinism at the test level.
- [ ] **Step 2: RED → implement → GREEN on the box.** If state-in-registers
  spills (check `ocloc` with `-options "-cl-opt-disable"`? no - check the
  build log via a temporary `-internal_options` dump only if perf in Task 9
  is catastrophic; correctness first).
- [ ] **Step 3: docs/12 section** - the state-column decomposition (why
  (head, chunk) and not (head): 192 vs 48 WGs), the ring-ownership argument
  written out, the conv-in-prologue choice (vs a separate conv kernel: one
  kernel fewer per layer, and the raw values are needed here anyway), what
  was NOT measured (SIMD32 alternative - the vllm-xpu reference uses it;
  state-in-SLM alternative), the measured per-layer time once Task 9 runs.
- [ ] **Step 4: Full suite; commit** - `feat(kernels): GDN decode step (conv + l2norm + gated delta recurrence)`

---

### Task 5: `attn_prep.cl` / `attn_decode.cl` / `attn_reduce.cl`

**Files:**
- Create: `src/kernels/attn.cl` (three entry points); Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`
- Create: `tests/kernels/attn_ref.h`, `tests/kernels/attn_test.cc`; Modify: `tests/CMakeLists.txt`

**Contracts:**

```
attn_prep<M>(const uint* ctrl, const float* qkv_partials /*[1][M][14336]*/,
             const float* fa_small /*q_norm fp32(1+w)[256]@0, k_norm@1024/4*/,
             const float* rope /*[max_len][2][32]*/,
             float* attn_q /*[M][24][256]*/, float* attn_gate /*[M][24][256]*/,
             ushort* kv_k, ushort* kv_v /*[max_len][4][256] each, this layer*/)
```
Grid (28, M): WGs 0..23 = q-heads, 24..27 = kv-heads. Column map (doc 03):
q-head `h` at `[h·512, h·512+256)`, its gate at `+256`; k-head `j` at
`12288 + j·256`; v-head `j` at `13312 + j·256`. Per q-head: `q_b =
rne_bf16(partials)`; RMSNorm over 256 in fp32 with the fp32 `(1+w)` weight →
`rne_bf16` → widen; partial RoPE on dims 0..63 (pairs `(i, i+32)`, `cos/sin =
rope[pos+m][0/1][i]`, dims 64..255 pass through) → `attn_q` fp32; gate:
`attn_gate[m][h][i] = f32(rne_bf16(partials_gate))`. Per kv-head: k gets the
same norm+rope then `rne_bf16` → `kv_k[pos+m][j]`; v is just
`rne_bf16(partials)` → `kv_v[pos+m][j]`. Reads `pos` from ctrl.

```
attn_decode<M, ATTN_BLOCK=256>(const uint* ctrl, const float* attn_q,
             const ushort* kv_k, const ushort* kv_v, float* attn_part)
```
Grid (4 kv-heads, max_len/256 blocks), WG 256 (16 sg). Early-out:
`block_start >= ctrl.pos + ctrl.n_active`. For each of the kv-head's 6
q-heads (`qh = j·6 .. j·6+5`) and each m: online softmax over the block's
positions `p` with the causal bound `p <= pos + m` and scale `1/16`
(=1/√256): scores via subgroup dot (16 lanes × 16 elements each, fixed-tree
reduce); per 16-position wave update `(mx, sm)` and the 256-wide `acc` (each
WI owns dim `wi` of acc: `acc[wi] += w_p · f32(kv_v[p][j][wi])`, waves in
fixed ascending order - deterministic). Writes `attn_part[qh][block][m]` =
`{mx, sm, acc[256]}` (258 fp32). Blocks with no valid position (all `p >
pos+m` inside a partially-valid block) still write `mx=-INF, sm=0, acc=0`.

```
attn_reduce<M>(const uint* ctrl, const float* attn_part,
               const float* attn_gate, ushort* attn_out /*[M][6144]*/)
```
Grid (24, M), WG 256: `nb = (pos + m)/256 + 1` blocks, merged in fixed
ascending order with the standard `(mx, sm, acc)` rescale; `out[i] =
acc[i]/sm`; `attn_out[m][h·256+i] = rne_bf16(f32(rne_bf16(out[i])) ·
sigmoid_f32(attn_gate[m][h][i]))` - matching the reference chain (eager
attention output cast to bf16, then `attn_output * torch.sigmoid(gate)` as a
bf16×fp32→fp32 op cast back at the end; doc 03).

Compiled variants: all three at M=1 (M=2 compile-only), `MAXLEN` baked as a
`-D` for `attn_decode`'s grid-independent code (the grid itself is set at
capture from `buffers.max_len`).

- [ ] **Step 1: Failing test.** `tests/kernels/attn_ref.h`: CPU reference of
  the full three-kernel pipeline: given synthetic qkv partials, an existing
  KV prefix (random bf16), pos, and the fp32 norms/rope tables - compute
  eager attention per doc 03 with the kernels' stated rounding chain.
  `attn_test.cc` cases at depths (pos) {0, 254, 255, 256, 4095} with
  max_len=4096 buffers: run prep→decode→reduce on device, compare
  `attn_out` (tol 1e-3 rel - exp in softmax), `kv_k/kv_v` written slots
  (bit-exact), and `attn_q` (bit-exact but for rope's cos/sin products -
  tol 2 ulp). Boundary cases 254/255/256 pin the block-edge and early-out
  logic; also assert a fully-early-out block's partials stayed untouched
  garbage (write a canary) - no, simpler: assert reduce ignores blocks
  beyond `nb` by canary-filling `attn_part` before the run.
- [ ] **Step 2: RED → implement → GREEN on the box.**
- [ ] **Step 3: docs/12 section** - the (kv-head, block) grid and the
  fixed-wave online softmax (why not one WG per q-head: KV reread ×6; why
  not flash-style single-pass over max_len in one WG: fill), the early-out
  cost pointer to doc 07 #12 (measured in Task 9 via depth 64 vs 4096), the
  canonical (mx, sm) merge maths, what was NOT measured (SLM-staged K tiles
  vs direct loads).
- [ ] **Step 4: Full suite; commit** - `feat(kernels): decode attention (prep, flash-decode blocks, gated reduce)`

---

### Task 6: `runtime/capture` - the 645-kernel list + replay determinism

**Files:**
- Create: `src/runtime/capture.h`, `src/runtime/capture.cc`; Modify: `src/runtime/CMakeLists.txt`
- Create: `tests/runtime/replay_determinism_test.cc`; Modify: `tests/CMakeLists.txt`
- Modify: spec §8.2 (one line: control block corrected to 128 B, 2026-08-25, Task 1's static_assert is the authority)

**Interfaces:**
```cpp
#pragma once
#include <map>
#include <memory>
#include <vector>
#include "l0/cmdlist.h"
#include "l0/kernel.h"
#include "l0/module.h"
#include "loader/loader.h"
#include "runtime/buffers.h"

namespace runtime {
// The decode step, captured once. Owns every Module/Kernel the list
// references (they must outlive it) and the closed list itself.
struct CapturedStep {
  l0::CmdList list;                       // closed, in-order
  size_t kernel_count = 0;                // printed; expect 645 at M=1
  std::map<std::string, std::unique_ptr<l0::Module>> modules;   // by variant
  std::vector<std::unique_ptr<l0::Kernel>> kernels;             // append order
};
// debug_resid: when non-null (bf16 [64][M][5120]), append a device-to-device
// copy of `resid` after every layer - the golden gate's per-layer tap. Copies
// are commands, not kernels; determinism is unaffected.
CapturedStep build(l0::Context& ctx, const loader::LoadedModel& m,
                   DecodeBuffers& b, l0::Mem* debug_resid = nullptr);
}  // namespace runtime
```
- `build` walks `model::Qwen35::layers()`: per token -
  `embed_gather` → per layer (GDN: `prep_res_norm(SP=0|16)`, `gemv(QkvZ)`,
  `gemv_bf16(AB)`, `gdn_step`, `prep_gated_head`, `gemv(OutProj)`,
  `prep_res_norm(SP16)`, `gemv(GateUp)`, `prep_silu_mul`, `gemv(Down)`;
  FA: `prep_res_norm`, `gemv(Qkv)`, `attn_prep`, `attn_decode`,
  `attn_reduce`, `gemv(OProj)`, `prep_res_norm`, `gemv(GateUp)`,
  `prep_silu_mul`, `gemv(Down)`) → `prep_res_norm(final_norm, SP16)` →
  `gemv_bf16(LmHead)` → `argmax_stage1` → `argmax_stage2`. Layer 0's first
  prep is the `SP=0` variant; every other layer-leading prep consumes the
  previous `Down` partials (SP=16). 48·10 + 16·10 + 5 = **645**.
- Variant names come from `kernels::*_variant` + `Qwen35::shape/linear`;
  modules cached by name; group sizes/grids from each kernel's Task-2..5
  contract; per-layer state pointers are base + layer-index·stride slices of
  the `DecodeBuffers` allocations (GDN layers indexed 0..47 in model order,
  FA layers 0..15 - two small index maps built in `build`).
- `static_assert(offsetof(Control, cur_token) == 8 && offsetof(Control, out_token) == 40)`
  beside the ctrl-as-uint[] kernels' documented indices (pos=0, n_active=1,
  cur_token=2.., out_token=10.., debug_flag=18).
- Argument binding uses `Kernel::arg_ptr`/`arg` in each kernel's declared
  order; a mismatch is caught by the smoke/golden tests, and each binding
  site cites the kernel contract's task number in a comment.

- [ ] **Step 1: Failing test `replay_determinism_test.cc`** (labels
  `checkpoint`; spec §8.7): load the model, build buffers + step; helper
  `snapshot()`/`restore()` for all persistent state (readback via immediate
  list into host vectors); ingest a fixed 16-id prompt (ids hardcoded from
  the prose golden prefix); generate 8 tokens recording the residual stream
  (via a debug_resid capture) and token ids; `restore()`; repeat; assert
  token sequences identical AND every recorded resid buffer **bitwise**
  identical; then a fresh-process-equivalent check: re-zero state, re-ingest,
  compare tokens again. Engine plumbing needed for "ingest/generate" lives in
  Task 7 - to keep this task self-contained, the test drives the raw loop
  itself (write `ctrl.cur_token[0]`, execute, fence-wait), ~30 lines, no
  Engine dependency; Task 7's Engine wraps the same loop.
- [ ] **Step 2: RED → implement `capture.cc` → GREEN on the box.** Expected
  first-run failures are binding-order mistakes; the NaN debug flag
  (`check_finite` is NOT in this plan - keep the debug_flag field reserved)
  means diagnosis is by the resid tap: add a temporary print of resid[0][0..3]
  per layer when a run goes NaN, remove before commit (say so in the report).
- [ ] **Step 3: Spec §8.2 one-line correction; docs/04 execution-model
  paragraph updated with the measured kernel_count.**
- [ ] **Step 4: Full suite; commit** - `feat(runtime): captured 645-kernel decode list + bitwise replay determinism proof`

---

### Task 7: Engine + `b70-decode` CLI

**Files:**
- Create: `src/runtime/engine.h`, `src/runtime/engine.cc`; Modify: `src/runtime/CMakeLists.txt`
- Create: `src/cli/b70_decode.cc`, `src/cli/CMakeLists.txt`; Modify: root `CMakeLists.txt`
- Create: `tests/runtime/engine_smoke_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
```cpp
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"

namespace runtime {
class Engine {
 public:
  // debug_resid=true captures the per-layer tap (golden gate); costs 64
  // device copies per token - off for benchmarking.
  Engine(l0::Context& ctx, loader::LoadedModel model, uint32_t max_len,
         bool debug_resid = false);
  void reset();                              // zero state, ring, KV, control
  // Feeds prompt ids one per replay (argmax result overwritten by the next
  // write; pos advances in-kernel). After ingest, control.pos == ids.size().
  void ingest(const std::vector<uint32_t>& ids);
  // Greedy-generates n ids; on_token is called after each fence (host side,
  // overlaps nothing in v1). Returns the ids.
  std::vector<uint32_t> generate(uint32_t n,
                                 const std::function<void(uint32_t)>& on_token = {});
  double last_tok_per_s() const;             // over the last generate() call
  // Golden-gate accessors: the per-layer resid tap for the last replay is in
  // debug buffer order [layer][m][5120]; readback helper.
  std::vector<uint16_t> read_debug_resid();  // throws unless debug_resid
  const loader::LoadedModel& model() const;
  DecodeBuffers& buffers();
 private:
  /* ctx ref, model, buffers, step, queue, fence, timing */
};
}  // namespace runtime
```
CLI (spec §12):
```
b70-decode <snapshot-or-repo> --ids <file> --n <N> [--device N] [--max-len 16384]
b70-decode <snapshot-or-repo> --bench [--depth 4096] [--tg 256] [--device N]
```
`--ids` reads whitespace-separated ids; prints one generated id per line
(stdout), `t/s` + ms/token to stderr. `--device` absent → `l0::Context::kFromEnv`
(Task 1's env contract). `--bench` is Task 9's step - in this task it exists
but only prints the timing scaffold around a depth/tg run (numbers recorded
in Task 9).

- [ ] **Step 1: Failing smoke test** (`engine_smoke_test.cc`, label
  `checkpoint`): construct Engine (debug off), ingest the same 16 ids as
  Task 6's test, generate 8 with a counting callback; assert 8 ids, all
  `< 248077`, callback fired 8×, `last_tok_per_s() > 0`; reset + repeat →
  identical ids (Engine-level determinism, complementing Task 6's raw-loop
  proof).
- [ ] **Step 2: RED → implement Engine + CLI → GREEN.** CLI run by hand once
  on the box (`tools/box.sh run "./build/src/cli/b70-decode Vishva007/… --ids /tmp/ids --n 8"`)
  - paste the output in the report.
- [ ] **Step 3: Full suite; commit** - `feat(cli): b70-decode - engine loop, ingestion, greedy generation`

---

### Task 8: The golden gate

**Files:**
- Create: `tests/golden/golden_gate_test.cc`; Modify: `tests/CMakeLists.txt`
- Create (pulled from box, committed - tiny): `tests/golden/prompts/prose.ids`, `code.ids`, `cjk.ids`
- Modify: `tools/oracle/README.md` (gate section: how to run, what it proved)

**The gate (spec §11):** for each prompt, on the box (labels
`checkpoint golden`):
1. Read the prompt ids (committed `.ids` files - pulled from
   `oracle-out/*.ids` via `tools/box.sh pull`; they are the tokenize outputs
   Task 7 of plan 2 produced; ≤ 64 ints each).
2. Engine with `debug_resid=true`: `ingest(prompt)` capturing, after **each
   prompt token**, the per-layer resid tap → engine-side
   `resid_e[t][layer][5120]` (bf16, host).
3. `generate(32)` → engine tokens.
4. Open the golden file (`oracle-out/<p>.golden.safetensors` on the box) with
   `loader::MappedFile` + `SafetensorsSet::parse_header`; compare:
   - **Gate:** the 32 generated ids == golden `tokens`, element-exact. On
     first mismatch print position, engine id, golden id, and both top-5
     logit sets (engine logits from `buffers().logits` readback at that step
     - the test regenerates stepwise to capture them; golden from `logits`).
   - **Diagnostics (printed, not gating):** per layer i: min-over-t cosine of
     `resid_e[t][i]` vs golden `resid.L{i}[t]` (fp64 accumulation), plus the
     global min and its (t, i); cosine < 0.999 marks the row `**LOW**`.
   - GDN state check: engine `gdn_state` after ingest vs golden
     `gdn_state.L{i}` - cosine ≥ 0.999 per layer, printed.
5. The test takes the golden dir as argv (default `oracle-out/`), skips with
   a clear message (exit 77 → ctest SKIP) if the files are absent.

- [ ] **Step 1: Write the test (RED: engine tap API exists but gate fails to
  build / files absent path exercised on the Mac).**
- [ ] **Step 2: Pull + commit the three `.ids` files** (verify counts 42/61/38).
- [ ] **Step 3: Run on the box.** Triage guide (write results into the
  report BEFORE fixing anything): tokens exact for all three → done. Tokens
  diverge at position p → the layer diagnostics localise it: cosine drops at
  a specific layer kind → that kernel; uniform slow drift with exact early
  tokens → accumulation-order divergence (expected class, judge magnitude);
  divergence at t=0 layer 0 → binding/order bug. Fix, re-run, keep every
  intermediate observation in the report - this step is the project's
  moment of truth and its record is a deliverable (docs/14-golden-gate.md
  gets the story: what diverged, why, what fixed it, final margins).
- [ ] **Step 4: `docs/14-golden-gate.md`** - the trust chain closed: engine ==
  oracle on 3×32 tokens; the measured cosine floor per prompt; the known,
  named divergences (GEMV accumulation, softmax path) and their observed
  magnitude vs the 0.999 bar; the vLLM cross-check status (README trust
  chain: now actionable - if a future vLLM greedy run disagrees with BOTH,
  suspect dequant).
- [ ] **Step 5: Full suite; commit** - `feat(golden): engine reproduces the oracle - 3×32 greedy tokens exact + layer diagnostics`

---

### Task 9: `--bench`, the number, and phase-1 verdict

**Files:**
- Modify: `src/cli/b70_decode.cc` (bench path finalised), `docs/BENCHMARKS.md`, `docs/05-perf-model.md`, `docs/07-open-questions.md` (#12), `docs/12-kernels.md` (measured step breakdown), `README.md` (ladder rows 1 verdict)
- Create: `tools/bench_decode.sh` (the exact invocation, committed)

- [ ] **Step 1: Bench implementation.** `--bench --depth 4096 --tg 256`:
  ingest 4096 ids (the prose ids cycled), then time `generate(256)` with a
  steady clock (debug_resid OFF); print `| b70-decode <git sha> | 4096 | 256
  | <t/s> | <ms/token> |` plus a coarse breakdown (total fence-wait time vs
  host time). Run 3× on an idle box; report median and spread.
- [ ] **Step 2: The depth experiment (doc 07 #12):** same at `--depth 64`;
  the t/s delta prices the fixed-grid attention early-out. Record the answer
  in doc 07 #12 (resolved) with both numbers.
- [ ] **Step 3: Records.** Append the median row to `docs/BENCHMARKS.md`
  beside vLLM's 31.50 with the command (`tools/bench_decode.sh`); update
  README ladder row 1 with the verdict; doc 05: the honest phase-1
  paragraph - beat 31.50 or not, achieved MBU (t/s × 15.540 GB / 590), and
  the measured gap decomposition (fence-wait vs host; per-kernel-count cost
  from doc 07 #5) naming the top suspects if short (GEMV fill at chosen S vs
  probe numbers; attention early-out; prep overhead). No tuning in this
  plan: the number is recorded as measured; optimisation is spec 1.5 work
  scoped from this decomposition.
- [ ] **Step 4: docs/12 completion check** - every kernel section has its
  measured number (per-kernel timing from a one-off instrumented run or the
  aggregate breakdown; label which). Full suite; commit -
  `feat(bench): tg256 measured and recorded; phase-1 verdict` and tag
  `decode-core-done`.

---

## Plan self-review (done at authoring, 2026-08-25)

**Spec coverage.** §8.1-8.6 → Tasks 1, 6, 7 (attention grid §8.6 in Task 5's
early-out + Task 9's depth experiment); §8.7 → Task 6; §8.8 partial - ZE_CHECK
throughout, DEVICE_LOST abort is plan-1 behaviour; `check_finite` debug kernel
deliberately NOT built (reserved field; the resid tap supersedes it for
diagnosis - recorded as a spec deviation in Task 6's step 3); §9.1 sequence →
Task 6 (645 = 48·10+16·10+5); §9.2 prep → Task 2; §9.3 → Task 3; §9.4 → Task
4; §9.5 → Task 5; §9.6 argmax → Task 3; §9.7 explanation rule → every task's
docs/12 step; §11 remaining rows (prep/gdn/attn/argmax tests, replay
determinism, golden) → Tasks 2-6, 8; §12 → Tasks 7, 9; §3 items 2-4 → Tasks
6-9. Deliberately out: fusion (§4.1 decision), check_finite, M>1 testing,
prefix caching, sampling beyond argmax.

**Buffer lifetime audit (the cross-task hazard):** `partials` is written by
each int4 GEMV and fully consumed by the next prep/attn_prep/gdn_step before
the following GEMV rewrites it - verified against the Task-6 sequence
(qkvz→[gdn_step+prep_gated_head both read it]→out_proj overwrite; in-order
list serialises). `x` (bf16) is written by each prep and consumed by exactly
the next GEMV. `gdn_o`, `ab_out`, `attn_q/gate/part/out` each have one writer
and one consumer per layer, in order. `resid` is read+written only by
single-WG-per-m kernels (embed_gather, prep_res_norm). Control is written
only by argmax_stage2 (and the host during ingestion).

**Type consistency.** `Control` field indices (pos 0, n_active 1, cur_token
2-9, out_token 10-17, debug_flag 18) match Task 1's struct and Task 3/6's
uses; `DecodeBuffers` member names used in Tasks 4-8 match Task 1's header;
kernel arg orders in Task 6's binding match Tasks 2-5's contracts;
`prep_res_norm` variants (SP 0/16) cover every use in Task 6's sequence;
attn column maps match doc 03 (q‖gate per-head interleave).

**Rulings recorded in this plan:** R3-1 Control = 128 B (spec §8.2 said 64,
its own fields need 96 - Task 6 corrects the spec); R3-2 per-op bf16 rounding
discipline (Task 2 preamble) - tightens the golden gate at zero cost; R3-3
online-softmax fp32 divergence accepted and documented; R3-4 prompt `.ids`
files committed in Task 8 (the golden files stay box-only).

**Known risks, stated:** Task 6 binding-order mistakes (diagnosed via the
resid tap); gdn_step register pressure (correctness first, Task 9 measures);
the golden gate may expose rounding-chain mismatches - Task 8's step 3 is
explicitly budgeted as the debugging moment with its record as a deliverable.
