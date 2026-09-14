# Spec 4 - K2-Horizon decode core (one B70)

Status: **design, 2026-09-14, for operator review.** Brainstormed with the operator the same day;
every design choice below was ruled section by section (§9 records the rulings).

The engine gains a second model: `urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ`, a
mixture-of-experts model with **routed value-expert attention (MoVA)**. This spec builds its
**decode core** - loader, kernels, the replayed decode list, a CPU oracle and golden gate, and a
recorded decode row - on **one card**. Prefill and serving are later specs. The qwen3_5 engine is
not touched.

Every number is measured unless marked **derived** or **estimated**. "Derived from the
checkpoint" means computed from the safetensors headers of snapshot `c0fd997` (refs/main), which is
identical in config, index and weight bytes to snapshot `2b6260c`.

## 1. Where we start

**The model** (`K2HorizonForCausalLM`, custom `modeling_k2_horizon.py`, read 2026-09-14):

| property | value |
|---|---|
| layers | 48: **0-2 dense** (plain attention + dense MLP), **3-47 sparse** (MoVA attention + MoE) |
| hidden | 2560; RMSNorm is **grouped** (2 groups of 1280), plain weight, eps 1e-6 |
| attention | 32 query heads, 8 KV heads, head_dim 128, full RoPE θ = 1e7, no q/k norm, no bias, softplus output gate |
| MoVA (sparse layers) | 64 value experts (2560 → 1024) + SiLU, top-4, sigmoid router with selection-only bias |
| MoE (sparse layers) | 100 SiLU-GLU experts (intermediate 768), top-8, sigmoid router with selection-only bias, + 1 always-on shared expert (768) |
| dense MLP (layers 0-2) | SiLU-GLU, intermediate 6144 |
| vocab | 250,624, **fully used** (tokenizer vocab 250,000 + 626 added = max id 250,623) - no padding mask |
| quantisation | GPTQ int4, group 64, symmetric, `desc_act: False`, sequential `g_idx`; `lm_head` bf16; MoE router bf16 (excluded by the quantiser); MoVA router int4 with f16 bias |
| tokens | BOS `<|ifm|begin_of_text|>` = 0; EOS = [1, 250019] |

**The int4 layout is the 27B's layout 0 unchanged.** Every linear ships `qweight I32 [in/8][out]`,
`scales F16 [in/64][out]`, symmetric; `g_idx[i] == i // 64` verified on a sample. `qzeros` and
`g_idx` are not loaded.

**Bytes (derived from the checkpoint):**

| part | GB |
|---|---:|
| routed experts (45 layers × 100) | 14.100 |
| value experts (45 layers × 64) | 4.011 |
| other attention linears (q, k, v, gate, o, v_router) | 0.877 |
| `lm_head` (bf16) | 1.283 |
| `embed_tokens` (bf16) | 1.283 |
| shared experts | 0.141 |
| dense MLP, layers 0-2 | 0.075 |
| MoE routers (bf16) | 0.023 |
| norms | 0.0005 |
| **loaded on device** (checkpoint 22.216 minus `qzeros`/`g_idx` 0.422) | **21.795** |
| **read per decode token** (8/100 routed, 4/64 value experts, everything else once, embed gathered) | **3.779** |

**vLLM** serves this model on this box only with **pipeline parallel 2, eager mode, 330k context,
fp8 KV** (`~/PycharmProjects/vllm/BENCHMARKS.md`). Eager because a `torch.fx` partitioner bug blocks
graph capture on this model; two cards because 330k of fp8 KV alone is ~31 GB. It needed a local
patch merging `v_experts` to load at all. **There is no like-for-like one-card baseline**, which is
why §2's bar is correctness.

**The engine** is built around one model: about 40 files across `src/runtime`, `src/loader`,
`src/kernels` and `tests` reference `model::Qwen35`. Decode is one command list captured once and
replayed with frozen arguments; **no host decision may depend on a device-produced value.**

## 2. Goals, bar, stopping rule

**Goal:** K2-Horizon decodes correctly on one B70 through a replayed Level Zero command list, with
its performance measured and recorded.

**Bar (hard):**
1. **Golden gate:** three prompts × 32 greedy tokens against a CPU oracle of this checkpoint, every
   determined row element-exact, undetermined rows inside the oracle's argmax set - the tie-aware
   rule of `tests/golden/golden_common.h`, unchanged. Tokens gate; tensors diagnose.
2. **Replay determinism:** three replays of the K2 decode list bitwise identical.
3. **Kernel tests:** every new kernel and variant green against its host reference.
4. **Load test:** every checkpoint tensor accounted for; shapes asserted.
5. **The 27B is untouched:** after the last commit its full suite, 774 kernels / 19 modules and both
   golden gates are green. This is the proof that §3.1's loader refactor preserved behaviour.

**Performance is recorded, not gated:** a `b70-decode --bench` row at depth 4096, tg 256 (median of
3, grade as the harness prints it), with t/s, ms/token, launches per token and the fence-time
share.

**Stopping rule:** all five bars met → record the row, tag `spec4-done`. Any bar short → memo to the
operator; nothing further without a ruling.

## 3. Design

### 3.1 Structure: a second model beside qwen3_5, not a generalised engine

Operator ruling: **no generalisation.** K2 is a parallel set sharing only infrastructure that is
already model-agnostic. The 27B's code paths keep their exact behaviour.

| component | role | reuses |
|---|---|---|
| `src/model/k2_horizon.{h,cc}` - `model::K2Horizon` | the model as data: constants, layer kinds (`Dense`, `Sparse`), per-layer linear table, small-tensor table, **expert groups** | `GemvShape`, `WeightKind`, `Fuse` |
| `src/loader` - `loader::load_k2()` → `K2LoadedModel` | executes the table; builds flat expert buffers | snapshot resolution, safetensors, layout-0 int4 packing - factored out of `load()` with its behaviour preserved (§2 bar 5 proves it) |
| `src/runtime/k2/` - `K2Engine`, `k2_capture.cc` | same `reset` / `ingest` / `generate` contract as `Engine`; builds and replays the K2 list | `l0`, `CapturedStep`, `Control`, the fence, argmax, and the RoPE table **builder** - K2 gets its own table, fp32 `[max_len][2][64]` for full 128-dim rotary (the 27B's is 64-dim partial rotary, `[max_len][2][32]`) |
| `src/kernels/k2/` | new kernels (§3.3) and new shape variants of existing ones | the kernel-variant build system |
| `tools/oracle/dump_k2.py` | CPU oracle for this checkpoint | `dequant.py`, `shard_paths()`, the golden file format |
| `b70-decode` | dispatches on `config.json` `model_type` (`qwen3_5` → `Engine`, `k2_horizon` → `K2Engine`) | argument parsing, `--bench` |

**Two traps the model table must encode:** K2's RMSNorm is `w · x̂` (plain), **not** the 27B's
Gemma-style `(1 + w) · x̂` - it needs its own small-tensor bake; and the norm is computed over **2
groups**, each with its own mean of squares.

### 3.2 Reference semantics (what every kernel must reproduce)

Per layer, pre-norm residual:

```
x̂ = GNorm(h; w_in)                       # 2 groups of 1280: x·rsqrt(mean(x²)+1e-6), then ·w; fp32, cast to bf16
h = h + Attn_L(x̂)
x̂ = GNorm(h; w_post)
h = h + (L ≤ 2 ? DenseMLP(x̂) : MoE(x̂))
logits = lm_head(GNorm(h; w_final))       # bf16 lm_head; no embedding or logit scaling
```

Attention, all layers:

```
q = W_q x̂  [32×128]     k = W_k x̂  [8×128]
v = W_v x̂  [8×128]                                          (L ≤ 2)
v = Σ_{e ∈ top4, ascending id} w_e · SiLU(V_e x̂)            (L ≥ 3, MoVA)
q, k ← RoPE: rotate_half convention, dim 128, θ = 1e7, cos/sin = cat(freqs, freqs)
write k, v to the KV cache at pos
o_h = softmax(q_h · K_{h//4}ᵀ / √128, causal) · V_{h//4}
o ← o ⊙ softplus(W_gate x̂ [32×128]; β = ln 2, threshold = 20)   # returns x itself where β·x > 20
a = W_o o
```

Routers (MoVA: top 4 of 64, W_r int4; MoE: top 8 of 100, W_r bf16):

```
s   = sigmoid(fp32(W_r x̂))          # logits from the weight only
sel = s + fp32(b_r)                 # bias used ONLY for selection
ids = topk(sel)                     # emitted sorted by ascending expert id
w   = bf16(s[ids] / Σ s[ids] · 2.5)
```

Feed-forward:

```
DenseMLP(x̂) = Down(SiLU(Gate x̂) ⊙ Up x̂)                       (intermediate 6144)
MoE(x̂)      = [Σ_{e ascending} w_e · Down_e(SiLU(Gate_e x̂) ⊙ Up_e x̂)] + Shared(x̂)
```

**Rounding points to mirror:** sigmoid in fp32; routing weights rounded to bf16 before use; routed
outputs summed in **ascending expert id** order, then the shared expert added - exactly the
reference's `index_add_` loop. Where our GEMVs accumulate fp32 and the reference computes bf16, small
tensor differences are expected; the token gate arbitrates (standing rule).

### 3.3 The decode list, and routing under replay (operator ruling: option A)

**The problem:** which experts run depends on the router's output, but the list is captured with
frozen arguments. **The design:** fixed expert *slots*, the expert chosen inside the kernel.

- A **top-k kernel** writes each layer's ids and bf16 weights to a small per-layer **router
  buffer** on the device.
- All experts of one projection in one layer live in **one flat buffer**, in id order. Every expert
  has the same shape, so expert `id` sits at `id × stride` (§3.4).
- The list holds **8 fixed slot launches per MoE layer and 4 per MoVA layer.** Slot `k` reads id `k`
  from the router buffer on the device and runs the int4 GEMV at that offset. Arguments stay frozen;
  only the active experts' bytes are read.
- A **combine kernel** applies the weights and sums in ascending id order.

Launches per token (**estimated**):

| block | launches |
|---|---|
Norms are two launches (fold, then finish), as in the 27B.

| block | launches |
|---|---|
| embed | 1 |
| dense layer (×3): norm (2) · fused q‖k‖v‖gate GEMV (2560→10240) · RoPE+cache write · attention (2 stages) · softplus gate · o_proj · norm (2) · gate‖up · SiLU·mul · down | 13 each |
| sparse layer (×45): norm (2) · fused q‖k‖gate‖**v_router** GEMV (2560→9280) · top-4 · 4 value slots · value combine · RoPE+cache write · attention (2) · gate · o_proj · norm (2) · MoE router (bf16, 2560→100, padded to 128) · top-8 · 8 × (gate‖up GEMV 2560→1536, SiLU·mul, down 768→2560) · combine · shared (3) | 46 each |
| head: final norm (2) · `lm_head` (bf16, 2560→250624) · argmax (2) | 5 |
| **total** | **~2,115** (27B: 774) |

**New kernels:** grouped RMSNorm (fold + finish, as the 27B's two-stage norm), softplus gate,
router top-k (k ∈ {4, 8}), int4 expert-slot GEMV (id indirection), expert combine.
**New variants of existing kernels:** int4 GEMV at the fused and per-expert shapes above; bf16 GEMV
for the MoE router and `lm_head`; GQA attention at ratio 4, head_dim 128, `max_len` 16384; RoPE and
cache write at 32/8 × 128; embed gather at hidden 2560; argmax at vocab 250,624. Every int4 shape
divides the int4 kernels' 64-column tiling and 64-weight groups (inputs K ∈ {2560, 4096, 768, 6144}
→ 40, 64, 12, 96 groups; outputs N ∈ {10240, 9280, 1536, 12288, 2560, 1024} all divisible by 64).

**One bf16 tiling constraint, handled the 27B's way:** the bf16 GEMV tiles output rows in 16s, and
the MoE router has 100 outputs. It is compiled at **N = 128 with `pad_n`** (zero weight rows), exactly
as the 27B pads its 96-row `a‖b` projection to 128, and **top-8 reads only rows 0-99.** `lm_head`
needs no padding (250,624 / 16 = 15,664).

**A deferred alternative, recorded:** one fused kernel per MoE layer (all 8 experts + combine in a
single launch) minimises launches but is one large kernel with three matmul shapes, hard to test and
tune. It is a candidate for the tuning spec, decided by the launch-overhead measurement of stage 0.

### 3.4 Device memory at `max_len` 16384 (operator ruling: 16k, bf16 KV)

| item | size |
|---|---:|
| weights (§1) | 21.80 GB |
| KV cache: 48 × 8 × 128 × 16384 × 2 (K, V) × 2 B | 3.22 GB |
| decode scratch, router buffers, RoPE table | < 0.10 GB (**estimated**) |
| **total** | **~25.1 GB of ~32** |
| reserved for spec 5's prefill scratch and driver overhead | ~6.9 GB |

**Expert buffers - flat, one allocation per (layer, projection)** for the 45 sparse layers: fused
gate‖up weights and scales, down weights and scales, value-expert weights and scales - 270
allocations. Strides (derived from the checkpoint): gate‖up **1,966,080 B** weights + 122,880 B
scales; down **983,040 B** + 61,440 B; value expert **1,310,720 B** + 81,920 B. Per-layer weight
buffers: gate‖up 196.6 MB, down 98.3 MB, value experts 83.9 MB.

- **Flat is required, not preferred:** frozen arguments cannot hand a kernel a different buffer per
  token; it can only compute an offset inside one.
- **Per layer, not global:** one gate‖up buffer across 45 layers would be 8.9 GB - past Level Zero's
  usual single-allocation limit - and would force 64-bit indexing in every slot kernel.
- **KV** is separate K and V allocations of 1.61 GB each. Stage 0 checks both sizes against the
  device's reported maximum allocation.

## 4. Correctness (how each bar is proven)

- **Prompts.** The same prose, code and CJK texts as the 27B, **re-tokenised with K2's tokenizer**
  into `tests/golden/k2/prompts/*.ids` (the Qwen `.ids` are never overwritten). Whether each prompt
  starts with BOS (id 0) is decided by what K2's tokenizer does by default, and recorded. Counts are
  re-checked against the 24-64 id window.
- **CPU oracle** (`tools/oracle/dump_k2.py`): the reference `modeling_k2_horizon.py` via
  `trust_remote_code` in `vllm-xpu-env-next-p314-t215-vxkp0`, on weights dequantised by
  `tools/oracle/dequant.py`, reading shards through `shard_paths()`. Output: the existing golden format
  (tokens, logits rows, per-layer residual taps) **plus, for every sparse layer and position, the
  MoE and MoVA expert ids (ascending) and weights.** Peak memory **estimated ~87 GB** of the box's
  121 GB: ~18.08 GB of `qweight` is ~36.2 billion int4 parameters, ~72.3 GB in bf16, plus 2.6 GB of
  bf16 embedding/`lm_head` ≈ 75 GB of state dict, plus the ~12 GB of overhead the 27B's dump showed
  (61.4 GiB peak on a 50.1 GiB state dict).
  Starting the container is the operator's; it runs only with an explicit go, on an otherwise idle box.
- **Golden gate** (`k2_golden_gate_test`): `golden_common.h`'s decision rule verbatim. **New
  diagnostic:** per sparse layer, the first position where the engine's expert ids differ from the
  oracle's - localising a routing divergence to a layer and position.
- **Replay determinism** (`k2_replay_determinism_test`): three replays bitwise identical - the
  direct proof that in-kernel routing is replay-safe.
- **Kernel tests**, one per new kernel or variant, against host references in the repo's style:
  grouped norm; softplus gate including the threshold-20 branch; top-k (selection bias, ascending id
  order, bf16 weights); expert slot bitwise against a plain GEMV on the same expert; combine;
  attention at ratio 4 / head_dim 128 against `attn_ref`.
- **Load test** (`k2_load_checkpoint_test`): unconsumed tensors = 0 with `qzeros` and `g_idx`
  explicitly marked not-loaded; shapes asserted. **Bytes read per token is redefined for MoE** as
  §1's 3.779 GB (active experts only), so utilisation is never computed against expert bytes a token
  does not read.
- **The 27B:** full suite, 774 / 19, both golden gates - re-run after the last commit.

## 5. Stages

0. **Facts before building.** (a) Device maximum single allocation vs the 196.6 MB expert buffers and
   the 1.61 GB KV buffers. (b) The t215 image imports `modeling_k2_horizon.py`; a one-token CPU
   forward measures oracle memory. (c) Re-tokenised prompts; the BOS decision. (d) The fixed cost of
   one small launch (a 2560 → 1536 int4 GEMV) - information for the tuning spec, not a bar.
1. **Model table + loader**: `model::K2Horizon`, `load_k2()`, flat expert buffers, the load test.
2. **Kernels**, each with its test (§3.3, §4).
3. **Oracle**: `dump_k2.py` and three K2 golden sets. Needs the box's CPU, not its GPU, so it can run
   beside stage 2 - but not beside a benchmark.
4. **Decode list + `K2Engine`**, then replay determinism and the golden gate.
5. **CLI + recorded row**: `model_type` dispatch; the `--bench` row at depth 4096 / tg 256.

**Derived expectations for the row, not bars:** bandwidth roofline at 590 GB/s is 590 / 3.779 =
**156 t/s** weights-only; with depth-4096 KV read (0.805 GB/token across 48 layers) 590 / 4.584 =
**~129 t/s**. At the 27B's ~75% utilisation that suggests **~95 t/s (estimated)** - before launch
overhead, which is the named risk.

## 6. Constraints

- Branch `spec1.7-codex-exp`; never pushed. Commits end with the session trailer.
- The Mac never compiles: build and test only via `tools/box.sh`. Anything longer than about a
  minute on the box launches detached (`setsid nohup …`) - its WiFi drops several times an hour.
- Never docker without the operator's explicit go; never kill a process you did not start; never
  delete anything on the box.
- Record grade requires zero containers and zero DRM holders; `tools/bench_decode.sh` reports it.
- Every number labelled measured / derived / estimated; never two values for one quantity; one
  defect, one fix, one measurement.

## 7. Out of scope

Prefill (spec 5 - prompts ingest one token per replay here); serving through `b70-serve`, K2's
tokenizer / chat template / reasoning and tool parsers (a later spec); performance tuning, including
the fused per-layer MoE kernel and launch reduction; fp8 KV; context beyond 16k; multi-GPU; training
losses (the router load-balancing term).

## 8. Risks, most likely first

| # | risk | how it shows | where caught |
|---|---|---|---|
| 1 | per-launch fixed cost dominates ~2,115 small launches; the row lands well below ~95 t/s | fence-time share; stage-0 launch probe | stage 0, stage 5 |
| 2 | `torch.topk` tie order differs from ours | per-layer router-id diagnostic | stage 4 |
| 3 | oracle runs out of memory (~87 GB estimated of 121) | dry run with `/usr/bin/time -v` | stage 0 |
| 4 | the t215 image's transformers cannot import the model's remote code | import check | stage 0 |
| 5 | Level Zero refuses a 1.61 GB KV or 196.6 MB expert allocation | allocation probe | stage 0 |
| 6 | the softplus threshold branch (`β·x > 20 → x`) is missed | kernel test with values above the threshold | stage 2 |

## 9. Operator rulings (2026-09-14)

1. Spec 4 is the **decode core only**; prefill and serving are later specs.
2. **16k context, bf16 KV.**
3. **Bar = correctness + a recorded row**; no performance gate.
4. **No generalisation:** K2 is a second model beside qwen3_5.
5. **Routing option A:** fixed slots, expert chosen in-kernel; the fused per-layer kernel deferred.
6. §3.3's decode sequence, §3.4's memory layout, §4's correctness design and §5's stages approved.
