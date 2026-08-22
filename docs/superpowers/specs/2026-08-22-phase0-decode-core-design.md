# Spec 1 - Phase 0 probes + the decode core

**Status:** approved design, 2026-08-22. Supersedes nothing; first spec of the
project.
**Target:** `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` on one Arc Pro B70.
**Reads:** docs 01-11 and `docs/BENCHMARKS.md`. This spec repeats numbers from
them only where a requirement depends on the number; the docs remain the
source for mechanism and evidence.

## 1. Purpose and scope

Build the smallest thing that can produce a decode tokens-per-second number
directly comparable to vLLM's **tg256 = 31.50** on the same model and box -
and, before that, measure the three numbers the whole design rests on.

**In scope**

- Phase 0: four probes whose results land in docs 05 and 07.
- A loader from the HF snapshot to a canonical on-device layout.
- A Level Zero runtime: one captured decode command list, replayed per token,
  with a device-resident control block and zero host work in the list.
- Decode kernels in OpenCL C (`ocloc`, `bmg-g31`): GEMV (int4 g64 and bf16),
  GDN step, decode attention, embedding gather, argmax.
- A Python oracle (transformers, CPU, in the reference container) producing
  golden tensors and tokens; tests that compare against them.
- A CLI, `b70-decode`, that ingests token ids, generates greedily, and reports
  tg256 t/s at depth 4096.

**Platform and model acquisition** (project-wide, decided 2026-08-23)

- **Linux only.** Ubuntu on the box is the only target; no Windows, no macOS
  runtime. Nothing in the tree may pay for portability it will not use
  (`mmap`, `/dev/dri`, Level Zero loader paths are Linux facts).
- **The server never downloads.** Models arrive via `hf download <repo>` into
  the standard HuggingFace cache (`~/.cache/huggingface/hub`, or
  `$HF_HOME/hub`), or the user passes an absolute snapshot path. The loader
  accepts either a repo id (`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`,
  resolved to `hub/models--<org>--<name>/snapshots/<rev>` with `<rev>` read
  from `refs/main`) or a directory. No network code, no `HF_TOKEN`, no
  `.incomplete` handling - a missing or partial snapshot is an error that
  names the path.

**Out of scope** (follow-on specs)

- Prefill kernels (sycl-tla, W4A16 then W4A8), tokenizer, chat template,
  HTTP/SSE - *spec 2, phase 1 completion*.
- MTP speculative decoding - *spec 3, phase 2*. This spec builds the `M ∈ [1,8]`
  machinery it needs but compiles and tests `M = 1` only.
- Sampling other than argmax; batching; a second GPU; quantising `lm_head`
  (an offline `tools/` conversion gated on the oracle - deferred until the
  oracle exists and the bf16 path is correct).

## 2. Fixed facts this spec depends on

| Fact | Value | Source |
|---|---|---|
| Read bandwidth | 600 GB/s | doc 01, measured |
| `W`, bytes per decode token | 15.52 GB | doc 03, measured from headers |
| Roofline, bf16 `lm_head` | 38.7 t/s | doc 05 |
| vLLM tg256 / pp4096 | 31.50 / 1973 | `docs/BENCHMARKS.md` |
| Layers | 64: GDN at `i % 4 != 3`, full attention at `i % 4 == 3` | doc 03 |
| Hidden / intermediate / vocab | 5120 / 17408 / 248320 (tokenizer uses 248077) | doc 03 |
| GDN heads | 16 k-heads, 48 v-heads, head dim 128, conv 4 taps | doc 03 |
| FA heads | 24 q, 4 kv, head dim 256, partial RoPE 64 dims, θ = 1e7, gated | doc 03 |
| Quantisation | GPTQ packing, int4, g64, sym, `desc_act: false`; `lm_head`, `in_proj_a/b`, `mtp.*`, norms in bf16 | doc 02, doc 03 |
| Layer math | doc 03, "Layer math - verified in the modeling file" | transformers 5.15.0 |
| Host toolchain | g++ 15.2, cmake 4.2, `ocloc 26.27`, L0 headers; no `icpx` needed | doc 10 |

## 3. Definition of done

All of the following, on the box, recorded in the docs:

1. **Phase 0 numbers** in doc 05 and doc 07: µs per kernel in a replayed list
   at N = 1 / 250 / 700; GEMV GB/s for every production shape (Section 4.2)
   with and without split-K, M = 1; L0-measured read bandwidth.
2. `ctest` green: kernel unit tests, loader round-trip, replay determinism,
   golden tests for three prompts (token match exact over 32 greedy tokens).
3. `b70-decode --bench` reports tg256 at depth 4096 and the number is written
   into `docs/BENCHMARKS.md` beside 31.50 with the command that produced it.
4. Every kernel has a section in a new `docs/12-kernels.md`: what it computes,
   how work is assigned to lanes and work-groups, what was tried and rejected,
   and its measured bandwidth or time. No kernel merges without it.

Beating 31.50 is the phase-1 target and is reachable inside this spec (prefill
does not affect tg256). It is *not* a condition for this spec being done - a
correct, measured, explained decode core that loses to vLLM is still done,
and tells us what spec 1.5 has to fix.

## 4. Phase 0 - probes

All under `tools/probe/`. Each is a standalone executable (or script) that
prints a markdown table to stdout; the table is pasted into the doc named.

### 4.1 `probe_replay` (C++, L0) → doc 07 #5

Create one regular, in-order command list with N launches of a trivial kernel
(writes one dword). Close it. Execute 1 000 times with fence waits. Report:

- µs per replay and µs per kernel for N ∈ {1, 250, 700};
- µs for an empty list (submit + fence round trip).

Decision rule written into doc 04: if µs/kernel ≥ 3, the fusion list in doc
04 is on this spec's critical path (Section 9.6 ordering); if < 1, the
unfused ~650-kernel list ships first and fusion is deferred.

### 4.2 `probe_gemv` (C++ host + the real `gemv.cl`) → doc 05 item 6, doc 08

Runs the production GEMV kernel (Section 9.2) on random int4 g64 weights at
the shapes the model actually runs after load-time fusion (Section 6.4):
`(K, N)` ∈ {(5120, 5120) out/o_proj, (5120, 14336) q‖k‖v, (5120, 16384)
qkv‖z, (5120, 34816) gate‖up, (17408, 5120) down}, plus the bf16 `lm_head`
(5120, 248320); M = 1; S ∈ {1, 2, 4, 8, 16}; both canonical layouts
(Section 6.3). Report GB/s of weight bytes per configuration. Checks results
against a CPU reference at every configuration (this is also the GEMV unit
test). N = 5120 is the worst fill case (320 subgroups) and the one split-K
exists for.

Decision rules: the layout with the higher GB/s summed over the five int4
shapes becomes the canonical layout; `S` per shape is the smallest value
within 3% of the best for that shape.

### 4.3 `probe_bw` (C++, L0) → doc 01

Device-to-device read-only sum over 2 GB and 8 GB buffers, 20 iterations after
5 warm-ups, with a plain OpenCL C kernel. Report GB/s. Expected ≈ 600; a
deviation > 5% from the torch-measured figure is investigated before any MBU
is quoted.

### 4.4 `checkpoint_bytes.py` → doc 03

The script that produced doc 03's byte accounting, landed verbatim. Takes a
snapshot directory, prints the group table and `W`.

## 5. Repository layout and build

```
CMakeLists.txt
cmake/ocloc.cmake          add_ocloc_kernel(<name> SOURCE <.cl> DEVICE bmg-g31 DEFINES ...)
src/
  l0/          RAII wrappers only: context.h device.h queue.h cmdlist.h module.h kernel.h
               fence.h memory.h  - one object per file; no model knowledge
  loader/      safetensors.{h,cc} (mmap + header parse + index manifest)
               quant.{h,cc} (classification, asserts, dequant reference)
               repack.{h,cc} (canonical layouts), upload.{h,cc}, loader.{h,cc}
  model/       qwen35.{h,cc}  - the model as data: layer kinds, shapes, weight→kernel bindings
  kernels/     *.cl, kernels.h (KernelSpec per kernel: name, args, work-group shape, variant keys)
  runtime/     buffers.{h,cc} control.{h,cc} capture.{h,cc} engine.{h,cc}
  cli/         b70_decode.cc
tools/
  probe/       probe_replay.cc probe_gemv.cc probe_bw.cc checkpoint_bytes.py
  oracle/      dump.py dequant.py  (run inside the reference container)
tests/
  kernels/     one test per kernel vs CPU reference
  loader/      repack round-trip
  runtime/     replay determinism
  golden/      golden_test.cc; *.safetensors here are generated, never committed
docs/12-kernels.md   written as kernels land (Section 3 item 4)
```

**Build:** C++17, `g++`, `-O2`, warnings as errors. Links `ze_loader` only.
Device code: each `.cl` × variant is an `ocloc compile -device bmg-g31
-options "-cl-std=CL3.0 -D..."` custom command producing a `.bin`
(`ZE_MODULE_FORMAT_NATIVE`) under `build/kernels/`. The runtime loads
binaries by variant name at startup; the executable does not embed them
(goal-3 nicety, later). Builds run on the box through CLion's remote
toolchain; `ctest` runs there. No Docker in the build.

**Dependency rule:** `l0` depends on nothing; `loader` on `l0`; `model` on
nothing; `kernels` on nothing (headers describe, `.cl` compute); `runtime` on
all four; `cli` on `runtime`. Tests may depend on anything. A header included
against this order is a review failure.

## 6. Loader

### 6.1 Input and naming

Input is a snapshot directory, or a HF repo id resolved against the local
cache as described in §1 (never downloaded). The loader reads `config.json`
(architecture fields, `quantization_config.dynamic` as data), uses
`model.safetensors.index.json` as the manifest, mmaps the shards it names, and
deduplicates by tensor name (index wins). Names: strip `model.language_model.`;
skip `model.visual.*`; `lm_head.weight` is top-level; `mtp.*` is loaded only
when `--mtp` is given (phase 2; this spec only verifies it parses).

### 6.2 Classification and asserts

A linear is int4 if `{qweight, scales, qzeros, g_idx}` exist for its prefix,
bf16 if `.weight` does. No label is trusted. At load, assert and fail by name:

- `quantization_config.sym == true`, `desc_act == false`, `group_size == 64`;
- `g_idx[k] == k / 64` for every k (identity under g64);
- every `qzeros` word `== 0x77777777` - GPTQ **v1** stores `zero − 1`, so the
  symmetric zero point 8 is stored as 7 (read from the checkpoint 2026-08-22);
  dequant is `w = (q − 8) · scale` either way;
- shapes match doc 03's tables for every layer;
- the 98 `dynamic` exclusions match exactly the tensors found in bf16.

Then `qzeros` and `g_idx` are dropped.

### 6.3 Canonical layouts

Two layouts, selected by `probe_gemv` (Section 4.2), both implemented by
`repack()`:

- **Layout 0, GPTQ-native:** `qweight[K/8][N]` u32 as shipped, `scales[K/64][N]`
  f16 as shipped. Zero-copy from the file.
- **Layout 1, tiled:** for each `(n_tile of 16, k_group of 64)`: 512 B of
  nibbles ordered `[k_octet (8)][n (16)]` u32, then 16 × f16 scales (32 B).
  Tiles ordered k-group inner, n-tile outer. One subgroup streams one
  contiguous 544 B block per k-group.

The loser is deleted from the tree once the probe has chosen.

### 6.4 Load-time fusion and derived tensors

- `in_proj_qkv ‖ in_proj_z` → one int4 matrix, N = 16384.
- `q_proj ‖ k_proj ‖ v_proj` → one int4 matrix, N = 14336.
- `gate_proj ‖ up_proj` → one int4 matrix, N = 34816, **interleaved in
  16-column blocks** (`gate[0:16] up[0:16] gate[16:32] …`) so the lane that
  holds `gate[n]` also holds `up[n]` after a fixed offset.
- `in_proj_a ‖ in_proj_b` → one bf16 matrix, N = 96.
- RMSNorm weights (`input_layernorm`, `post_attention_layernorm`, `norm`,
  `q_norm`, `k_norm`) stored as `1 + w`. `linear_attn.norm` stored as `w`.
- `A_log` → `−exp(A_log)` fp32; `dt_bias` fp32; `conv1d.weight` fp32
  `[10240][4]`.
- RoPE table `cos/sin[max_model_len][32]` fp32 from `inv_freq_i = θ^(−2i/64)`.
- `embed_tokens`, `lm_head` bf16 as shipped.

### 6.5 Memory plan (M = 8 activations from day one; `max_model_len` default 16384)

| Buffer | Size |
|---|---|
| int4 weights + scales | 12.92 GB |
| bf16 small weights | 0.05 GB |
| `embed_tokens`, `lm_head` | 2.54 GB each |
| GDN state `S[48][128][128]` fp32 × 48 layers | 151 MB |
| GDN conv ring `[16][10240]` bf16 × 48 layers (depth 16 ≥ M + 3 so a step's writes never land on slots another work-group reads as history) | 15.7 MB |
| KV ring `[max_model_len][4][256]` bf16 × 2 × 16 layers | 1.07 GB |
| Activations: residual `[8][5120]` bf16; normalised/activated `x` `[8][17408]` bf16; partials fp32 `[S_max=16][8][34816]` (17.8 MB); qkvz, qkv, ab, gdn_out, attn_out, mlp_mid; logits `[8][248320]` fp32 (7.9 MB); attention partials `[24][64][8][258]` fp32 (12.7 MB) | < 100 MB |
| Control block | 64 B, `zeMemAllocShared` |

≈ 19.3 GB of 32 GB. Upload: one `zeCommandListAppendMemoryCopy` per buffer on
the immediate list; the loader asserts total resident bytes against `W`.

### 6.6 Loader test

`tests/loader/repack_roundtrip`: for 8 random tiles per layout, dequantise
from the file with `quant::dequant_reference()` and from a readback of the
device buffer; bit-exact equality. `dequant_reference()` is ported line-for-line
from `tools/oracle/dequant.py` and the two are cross-checked by a fixture
the oracle writes (Section 10).

## 7. Model description (`src/model`)

`qwen35.h` is data, not code: a `LayerDesc` per layer (`kind`, every weight's
name, shape, dtype, canonical buffer), the head, and the constants above. It
knows nothing about L0 or kernels; `runtime/capture` walks it to bind buffers
to `KernelSpec` arguments. The layer math it encodes is exactly doc 03's
"Layer math" section; any divergence is a bug in one of the two and the oracle
decides which.

## 8. Runtime

### 8.1 Objects

One `Context`, one `Device` (selected by `--device N`, default 0), one compute
`Queue` (ordinal 0, index 0), one immediate `CmdList` for setup and copies,
one regular `CmdList decode` created with `ZE_COMMAND_LIST_FLAG_IN_ORDER`,
one `Fence`. No events: in-order execution is the dependency model.

### 8.2 Control block

```c
struct Control {          // 64 B, zeMemAllocShared, read by kernels, written by host + argmax
  uint pos;               // KV slot / position of token 0 of this step
  uint n_active;          // tokens in this step, 1 in this spec
  uint cur_token[8];      // input ids for this step
  uint out_token[8];      // argmax outputs, written by the last kernel
  uint pad[6];
};
```

Kernels read `pos` and `n_active` and derive `seq_len = pos + n_active`.
The argmax kernel writes `out_token[m]`, sets `cur_token[0] = out_token[n_active−1]`
and does `pos += n_active`. Nothing in the list is ever mutated after close.

### 8.3 Capture

Once, after load: for each layer append its kernels (Section 9.1) with fixed
arguments bound from `qwen35.h`, then gather → … → `lm_head` → argmax; close.
The capture code is a straight walk; no conditionals on runtime state.

### 8.4 Step

```
execute(decode, fence); fence.wait(); token = control.out_token[0];
```

That is the per-token host path. Generation loop in `engine.cc`:
`for i < n: step(); emit(token)`.

### 8.5 Ingestion (prompt) through the same list

For each prompt token: `control.cur_token[0] = id; step();` - a 4-byte store
into shared memory, not a list mutation; argmax's `cur_token` write is
overwritten by the next store. `pos` advances by the argmax kernel as in
generation. A 4096-token prompt takes ~4096 × step time (≈ 2 min at the
roofline); acceptable here, replaced by prefill in spec 2.

### 8.6 Attention grid under replay

`attn_decode`'s grid is fixed at `4 × ceil(max_model_len / 256)` work-groups;
each reads `seq_len` and returns immediately if its block start ≥ `seq_len`.
Bucketed lists are not built in this spec; doc 07 #12 is measured by running
`--bench` at depth 64 vs 4096 and reported in doc 05.

### 8.7 Determinism test

`tests/runtime/replay_determinism`: load, ingest a 16-token prompt, snapshot
all state buffers and the control block; generate 8 tokens; restore the
snapshot; generate 8 tokens again; assert the two residual-stream traces and
token sequences are bitwise identical. A second case: ingest the same prompt
twice from a fresh load and compare. Any kernel that breaks this test is
rejected - doc 04's rule.

### 8.8 Errors

Every `ze*` call is wrapped; failure throws `l0::Error{call, ze_result_t}`
with the result name. `ZE_RESULT_ERROR_DEVICE_LOST` terminates the process
with the failing call named. In `-DB70_DEBUG` builds a `check_finite` kernel
runs on the residual stream after every 8th layer and writes the first
offending layer index into the control block's `pad[0]`; the engine reports
it after the fence.

## 9. Kernels

All OpenCL C 3.0, `intel_reqd_sub_group_size(16)`, fp32 accumulation,
bf16 carried as `ushort` and widened with `<< 16`. Each kernel is declared
once in `kernels.h` as a `KernelSpec{name, args[], wg_shape, variants}` that
`capture.cc` and the unit tests both consume. Every kernel takes `M` as a
compile-time define and loops over `m < n_active`; this spec builds and
tests `M = 1` only, but the loop is present and the `M = 2` variant must at
least compile.

**Determinism rule:** no floating-point atomics, no work-stealing, no
data-dependent work-group counts. Reductions are fixed-tree or two-stage.

### 9.1 Per-token kernel sequence

GDN layer (10): `prep(RESIDUAL|RMSNORM)` → `gemv(qkv‖z)` → `gemv_bf16(a‖b)` →
`gdn_step` → `prep(GATED_HEAD)` → `gemv(out_proj)` → `prep(RESIDUAL|RMSNORM)`
→ `gemv(gate‖up)` → `prep(SILU_MUL)` → `gemv(down_proj)`.
FA layer (10): `prep(RESIDUAL|RMSNORM)` → `gemv(q‖k‖v)` → `attn_prep` →
`attn_decode` → `attn_reduce` → `gemv(o_proj)` → `prep(RESIDUAL|RMSNORM)` →
`gemv(gate‖up)` → `prep(SILU_MUL)` → `gemv(down_proj)`.
Per token: `embed_gather` first; `prep(RESIDUAL|RMSNORM)` with the final norm,
`gemv_bf16(lm_head)`, `argmax_stage1`, `argmax_stage2` last. **645 kernels per
token.** Doc 04's "~250" is revised to this number. Folding `prep` into the
following GEMV's prologue was the original design and was rejected during
planning: staging `M × K` activations in SLM does not fit (8 × 17408 × 4 B =
557 KB against 128 KB), and re-reading split-K partials in every work-group
costs up to ~25% extra L2 traffic on `gate‖up`. It returns as the first
optimisation candidate once `probe_replay` prices a kernel.

### 9.2 `gemv` - the int4 linear, and `prep`

```
gemv<M, K, N, S, LAYOUT>(
  const uint*   w,          // canonical int4 (+ scales inline for layout 1)
  const half*   scales,     // layout 0 only
  const ushort* x,          // bf16 [M][K], already normalised / activated by `prep`
  float*        out)        // fp32 partials [S][M][N]  (S == 1: just [M][N])
```

- Subgroup owns 16 consecutive `n`; work-group = 4 subgroups = 64 `n`; grid =
  `(N/64) × S`; work-group `(gn, s)` handles k-groups `[s·G/S, (s+1)·G/S)`,
  `G = K/64`. Requires `N % 64 == 0`, `K % 64 == 0`, `S | G` - asserted at
  capture; `a‖b` is zero-padded to N = 128 at load.
- Per k-group: lane reads its 8 u32 words (64 nibbles for its `n`) and one
  f16 scale - layout 0 as strided coalesced loads, layout 1 as one
  `intel_sub_group_block_read8` + one ushort block read; `x[m][k]` for the
  group's 64 k come in as `ushort8` vector loads (16 lanes hitting the same
  line - a broadcast load, no shuffles); `acc[m] += (nibble − 8) · scale · x`.
- Epilogue: `out[s][m][n] = acc[m]`. Nothing else. Residual, activation and
  normalisation live in `prep` so split-K stays deterministic and the GEMV
  stays one thing.

`gemv_bf16<M, K, N>` is the same skeleton over bf16 weights in the canonical
bf16 layout (tiles `[n_tile][k_octet][8 k][16 n]` ushort, one
`intel_sub_group_block_read_us8` per lane per 8 k), `S = 1`, used for `a‖b`
(N = 128 padded) and `lm_head` (N = 248320, writes fp32 logits).

```
prep<M, K, MODE, S_PREV>(
  const float*  partials,   // fp32 [S_PREV][M][K_in]
  ushort*       resid,      // bf16 [M][K] residual stream, read + written in place (RESIDUAL)
  const ushort* norm_w,     // (1+w) bf16 [K] (RMSNORM) | gated norm w[128] (GATED_HEAD)
  const float*  aux,        // GATED_HEAD: fp32 z [M][6144]
  ushort*       x_out)      // bf16 [M][K_out]
```

One work-group of 256 per `(m, chunk of 4096 k)`; each work-item owns its
elements end-to-end, so in-place residual update has no cross-work-group
hazard. Modes: `RESIDUAL|RMSNORM` - `r = resid + Σ_s partials; resid = r;
x_out = r · rsqrt(mean(r²)+1e-6) · norm_w` (the mean is a work-group
reduction; with K = 5120 one work-group covers the row, and `K_out = K`);
`SILU_MUL` - input `[M][34816]` interleaved in 16-column blocks, `x_out[k] =
silu(g[k]) · u[k]`, `K_out = 17408`; `GATED_HEAD` - per 128-wide head:
`x_out = w · (o · rsqrt(mean(o²)+1e-6)) · silu(z)`, `K = K_out = 6144`
(plan 3).

Variants compiled: one per distinct `(K, N, S, LAYOUT)` actually used
(≈ 6) × `M`, plus `prep` per `(K, MODE)`. `S` per shape comes from
`probe_gemv`.

### 9.3 `embed_gather<M>`

Copies `embed_tokens[cur_token[m]]` (5120 bf16) into `resid[m]`. 1 work-group
per `m`.

### 9.4 `gdn_step<M>`

Grid: 48 v-heads × 4 column chunks = 192 work-groups of 256 work-items
(16 subgroups × 16 lanes); work-group `(h, c)` owns `S[h][:, 32c : 32c+32]`.

Per token `m` in order:
1. **Prologue / conv1d:** sum `S_PREV` partials of `qkv‖z` for this head's
   channels - q = k-head `h/3` (128), k = k-head `h/3` (128), v = head `h`
   (128), z = head `h` (128) - for **all** `m < n_active` of this step, so
   the work-group holds this step's raw values itself and never needs another
   work-group's write. Apply the 4-tap causal conv over the ring slots
   `(pos+m−3 .. pos−1) % 16` (pre-step history, read-only this step) and this
   step's own raw values for `pos .. pos+m`, then SiLU, for q, k, v (not z).
   **Ring ownership:** chunk `c == 0` writes the raw values of tokens
   `pos .. pos+n_active−1` into slots `(pos+m) % 16` for the v channels of
   head `h`; the work-group with `h % 3 == 0, c == 0` writes the shared q and
   k channels of k-head `h/3`. Depth 16 ≥ `M + 3` guarantees a written slot
   is never one another work-group reads as history in the same step. Written
   values are identical wherever they are recomputed.
2. `β = sigmoid(b[h])`, `g = negA[h] · softplus(a[h] + dt_bias[h])` from the
   fp32 `a‖b` output.
3. `q, k ← l2norm` over 128 (subgroup reduce + broadcast), `q *= 1/√128`.
4. For the 32 owned columns `v`: `S[:,v] *= exp(g)`; `kv[v] = Σ_k S[k,v]·k[k]`;
   `Δ[v] = (v[v] − kv[v])·β`; `S[:,v] += k ⊗ Δ[v]`; `o[v] = Σ_k q[k]·S[k,v]`.
   The 128 × 32 state slice lives in registers across the `M` loop and is
   written back once.
5. Gated norm over the head's 128 `o` values needs all 4 chunks: the kernel
   writes `o` to scratch fp32, and the **last** step - the RMS over 128, `× w
   × silu(z)` - is `prep(GATED_HEAD)` (Section 9.2). No cross-work-group
   synchronisation inside `gdn_step`.

### 9.5 `attn_prep<M>`, `attn_decode<M>`, `attn_reduce<M>`

- `attn_prep`: grid 28 work-groups (24 q-heads + 4 k-heads) × `M`. Sums the
  `q‖k‖v` partials for its head (it is the one consumer, so it plays `prep`'s
  role) (512 values for a q-head: `[q 256 | gate 256]`
  interleaved per head, 256 for k/v), RMSNorm(1+w) over 256 for q and k,
  RoPE on dims 0-63 with `cos/sin[pos+m]` (`rotate_half` over the 64 slice),
  writes q fp32 to scratch, gate fp32 to scratch, k and v bf16 into the KV
  ring at slot `pos+m`. v-heads ride with the k work-groups.
- `attn_decode`: grid `4 × ceil(max_model_len/256)`; early-out on
  `block_start ≥ seq_len`. Work-group loads its K/V block (256 × 256 bf16 ×
  2 = 256 KB, streamed in 16-row slabs through SLM), and for each of its 6
  q-heads × `M` queries runs online softmax in fp32 with scale `1/16` and the
  causal bound `position ≤ pos+m`, emitting `(max, sum, acc[256])` to the
  partials buffer.
- `attn_reduce`: grid 24 × `M`; combines the `ceil(seq_len/256)` partials with
  the standard `(max, sum)` rescale, multiplies by `sigmoid(gate)`, writes
  `attn_out[m][head·256 …]` bf16 for `o_proj`.

### 9.6 `argmax_stage1<M>`, `argmax_stage2<M>`

Stage 1: 243 work-groups × 1024 logits each → `(max, idx)` per group, fixed
tree, lowest index wins ties. Stage 2: one work-group reduces 243 pairs, masks
`idx ≥ 248077` (tokenizer size) by never selecting them, writes
`control.out_token[m]`, and on `m == n_active−1` updates `cur_token[0]` and
`pos`.

### 9.7 Build order and the explanation rule

Order of implementation: `gemv` (via `probe_gemv`) → `embed_gather` →
`gemv_bf16` → `argmax` → a **GEMV-only end-to-end** against the oracle's
per-layer MLP outputs → `gdn_step` → `attn_*`. Each kernel lands with its
`docs/12-kernels.md` section and its `tests/kernels/` test; a kernel whose
section cannot state why its lane assignment is shaped that way is not done.

## 10. Oracle (`tools/oracle/`, runs in `vllm-xpu-env-next-p314-t214-vxkp0` on CPU)

- `dequant.py`: `dequant_gptq(qweight, scales, g=64) -> bf16 tensor` - the
  reference meaning of the nibbles. Writes `tests/golden/dequant_fixture.safetensors`
  (one small tensor in packed and dequantised form) for the C++ loader test.
- `dump.py <snapshot> <prompt_ids.json> <out.safetensors>`: builds
  `Qwen3_5ForConditionalGeneration` (text part) from `config.json` with
  `quantization_config` removed and the vision config ignored; loads
  dequantised weights (bf16; ~54 GB RAM); registers hooks; runs the prompt
  (≤ 64 tokens, pure-torch GDN path, no `fla`) and 32 greedy continuation
  steps; saves:
  - `resid.L{i}` residual stream after layer `i`, all positions, bf16;
  - `mixer.L{i}`, `mlp.L{i}` outputs;
  - `gdn_state.L{i}` fp32 and `conv_state.L{i}` after the prompt;
  - `logits` at every prompt position and every generated position;
  - `tokens` (the 32 greedy ids).
- Three prompts in `tests/golden/prompts/`: English prose, Python source, a
  CJK paragraph. Generated files are ~1 GB each and are not committed
  (`.gitignore` already excludes `*.safetensors`); `make golden` regenerates.

## 11. Tests

| Test | Checks | Tolerance |
|---|---|---|
| `tests/kernels/gemv_*` | each `(K,N,S,LAYOUT)` variant vs CPU reference, random data | per output: abs err ≤ 1e-4 × max|reference row| + 1e-5 (fp32 output, summation order only) |
| `tests/kernels/prep_*` | each `(K, MODE)` vs CPU reference | ≤ 2⁻⁷ relative (bf16 output) |
| `tests/kernels/gdn_step` | one layer, random state, vs C++ port of `torch_recurrent_gated_delta_rule` | fp32 state rel err ≤ 1e-5; output ≤ 1e-3 |
| `tests/kernels/attn_*` | random KV at depths 1, 255, 256, 4096 vs CPU softmax | ≤ 1e-3 |
| `tests/kernels/argmax` | ties, masked tail, all-equal | exact |
| `tests/loader/repack_roundtrip` | Section 6.6 | exact |
| `tests/runtime/replay_determinism` | Section 8.7 | bitwise |
| `tests/golden/golden_test` | per-layer cosine of `resid.L{i}` ≥ 0.999 and max-abs reported per layer; `gdn_state` cosine ≥ 0.999; 32 tokens equal | tokens exact; tensors diagnostic |

Golden failure output: first mismatching layer (tensor) or position (token)
with both top-5 logit sets. Tensor tolerances are diagnostics - token
equality is the gate - because 64 layers of bf16 residual drift make a hard
tensor bound either useless or flaky.

## 12. CLI and the benchmark

```
b70-decode <snapshot> --ids <file> --n <tokens> [--device N] [--max-len 16384]
b70-decode <snapshot> --bench [--depth 4096] [--tg 256]
```

`--ids` reads whitespace-separated token ids (the tokenizer lives in spec 2;
ids come from the oracle's tokenizer via `tools/oracle/tokenize.py`). Output:
one id per line, then `t/s` on stderr.

`--bench`: ingests `--depth` tokens of a fixed synthetic prompt (ids from the
golden prose prompt repeated), then times `--tg` generated tokens with a
monotonic clock around the step loop and prints a markdown row
`| b70-decode <git sha> | depth | tg | t/s |`. That row is appended to
`docs/BENCHMARKS.md` by hand with the command. This is the quantity
`llama-benchy --latency-mode generation` reports as tg256.

## 13. Risks and where they are measured

| Risk | Where it shows | Response |
|---|---|---|
| Per-kernel replay cost ≥ 3 µs makes 645 kernels ≈ 7-8% of a step | `probe_replay` | pull doc 04's fusion list into this spec before `gdn_step` |
| GEMV cannot fill the device at N = 1024 even with split-K | `probe_gemv` | lane-per-(n, k-half) variant; reported in doc 08 |
| `zeMemAllocShared` reads inside kernels are slow | `probe_replay` variant reading the control block | fall back to device memory + one 64 B copy per step |
| Oracle too slow on CPU for 64-token prompts | first `dump.py` run | shorten to 32 tokens; the test is about layers, not length |
| Kernel count in one list exceeds a driver limit | capture | split into two lists executed back-to-back (still zero host work) |
| Ring-write ownership in `gdn_step` races | `replay_determinism` | designed not to (Section 9.4); the test is the proof, not the argument |
| 645 kernels × per-kernel cost is a large fraction of a step | `probe_replay` | fold `prep` into the GEMV prologue for the K = 5120 consumers (fits: 5120 × M × 2 B) - measured, not assumed |

Open questions touched: doc 07 #5 (resolved by 4.1), #12 (measured by 8.6),
#13 (untouched - no sycl-tla here), #6 (`lm_head` quantisation - deferred to
after the oracle exists).

## 14. Follow-on specs

- **Spec 2 - phase 1 completion:** prefill (sycl-tla W4A16 mixed-dtype GEMM
  and flash attention, chunked GDN from `vllm-xpu-kernels` as reference; W4A8
  as a probe), tokenizer + chat template (doc 11), `/v1/completions` and
  `/v1/chat/completions` with SSE, `llama-benchy` parity. Its design starts
  once this spec's plan is written; its kernels run on a SYCL queue and never
  enter the decode list.
- **Spec 3 - phase 2, MTP:** the `M ∈ [1,8]` variants, the MTP head, the
  draft/verify loop in the runtime, acceptance-rate measurement against
  vLLM's 42.56 / 45.23.
