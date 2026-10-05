# Spec 18 - K2-Horizon MoVA 36B-A4B, the second MoE model

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.

**Model:** [`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ`](https://huggingface.co/urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ)
(the operator's AutoRound int4 g64 quant of `IFM/K2-Horizon-MoVA-36B-A4B`; 22.22 GB of safetensors;
`K2HorizonForCausalLM`, remote code `modeling_k2_horizon.py`).

**Kernel references:** the B70 W4A16 MoE kernels in the vLLM XPU stack (grouped GEMV over
selected experts, sparse expert dispatch, fused gate‖up GEMV) are Apache-2.0 references for §5's
expert kernels, read for their design and measured against in 18b; ours are written for the
replayed-list model (no host round trip).

**History:** this spec carries forward **spec 4** (*K2-Horizon decode core on one B70*, 2026-09-14,
commits `45a3115` / `d307d0e`, removed from the docs tree in `c197fb0`), which was designed with the
operator section by section and parked "before we get prefill to vLLM level". That condition is met
(pp4096 2125 against vLLM's 1610 t/s). Spec 4's reference semantics (§3 below), routing design and
memory facts stand; this spec updates them for what the engine has gained since (specs 5-17).

**Order:** after spec 15 (Ornith): spec 15 builds the MoE machinery (device-side router and top-k,
expert slots addressed by id inside kernels, the grouped expert GEMM for prefill) that K2 reuses with
a different router formula.

Every number is **measured** unless marked **derived** or **estimated**; "derived from the checkpoint"
means computed from its safetensors headers (spec 4 §1, re-read 2026-10-05: config and index
unchanged).

## 1. The model

| property | value |
|---|---|
| layers | 48: **0-2 dense** (plain attention + dense MLP, intermediate 6144), **3-47 sparse** (MoVA attention + MoE) |
| hidden | 2560; RMSNorm **grouped** (2 groups of 1280), **plain** `w · x̂` (not the Qwen `(1 + w) · x̂`), eps 1e-6 |
| attention | **every layer is full attention** (no GDN, no sliding window): 32 q-heads, 8 kv-heads, head_dim 128, full RoPE (dim 128, θ 1e7), no q/k norm, softplus output gate |
| MoVA (sparse layers) | **64 routed value experts** (2560 → 1024, SiLU), top-4, sigmoid router (int4 weight, f16 selection-only bias) |
| MoE (sparse layers) | 100 SiLU-GLU experts (intermediate 768), top-8, sigmoid router (bf16, selection-only bias), × 2.5 normalised weights, + 1 shared expert (768) |
| vocab | 250,624, fully used; BOS 0, EOS [1, 250019] |
| MTP head | **none** (no `mtp.*` tensors) |
| quantisation | GPTQ int4 g64 sym, sequential `g_idx`; `lm_head` and embedding bf16 (1.283 GB each); MoE router bf16; MoVA router int4 |
| max positions | 524,288 |

**Bytes (derived from the checkpoint, spec 4 §1):** 21.80 GB loaded; **3.78 GB read per decode
token** with the bf16 head (8 of 100 experts, 4 of 64 value experts); **~3.14 GB with spec 9's int8
head** (derived).

**KV: 192 KiB per position** (48 layers x 8 kv-heads x 128 x 2 x 2 B): 3.2 GB at 16k, 6.4 GB at 32k,
25.8 GB at 128k (derived).

## 2. What changed since spec 4, and what it means

| since spec 4 | for K2 |
|---|---|
| spec 5 int8 prefill (h8) | applies to the dense and attention linears; per-expert only if spec 15 shows it pays at small M |
| spec 6 flash attention | prefill attention at **head_dim 128, GQA 4** (the kernel is written for 256 / GQA 6): new variants, K1-checked |
| spec 7 prefix cache | **simpler than for Qwen3.8:** no GDN state, so a snapshot is KV only (restore = KV copy + `pos`); the block hook and store apply unchanged |
| spec 8 MTP | **not applicable** (no head); speculative decoding for K2 would need a draft model: out of scope |
| spec 9 int8 `lm_head` | applies (1.28 GB → 0.64 GB) |
| spec 10 decode attention v2 | variants at head_dim 128 / GQA 4; 48 attention layers make decode at depth KV-heavy |
| spec 12 int8 KV (rotkv) | halves 192 KiB per position: the main lever for long context on one card |
| spec 14 / 15 `ModelDesc` | K2 stays a **second engine beside qwen3_5** (spec 4 ruling 4: its layer structure has no GDN and a different norm); it shares the descriptor's shared pieces, not its Qwen layer plan |
| spec 15 MoE machinery | reused: router + top-k on the device, expert slots by offset, fixed-order combine, the grouped prefill GEMM, device-side routing tables |
| spec 16 / 17 multi-GPU | **PP is what gives K2 long context:** on two cards, ~64k with bf16 KV or ~128k with int8 KV (derived); TP is not planned for K2. vLLM's PP run reached 390k only with fp8 KV and an uneven 25 / 23 split (the `lm_head` rank is heavier) |

**Launch budget.** Spec 4's decode list was **2067 launches per token** (8 fixed slot launches per MoE
layer, 4 per MoVA layer). Spec 15's design (one gate‖up launch over all selected experts plus the
shared one, one down + combine launch) cuts each sparse layer from 45 to ~22 launches: **~1000 per
token (estimated)**. At decode's few-millisecond step (bandwidth-derived ~5 ms at 3.14 GB) the
per-launch floor decides the rate; spec 15 P0 measures it.

## 3. Reference semantics (carried from spec 4 §3.2, unchanged)

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

Rounding points to mirror: sigmoid in fp32; routing weights rounded to bf16; routed outputs summed in
ascending expert id, then the shared expert added (the reference's `index_add_` order). The first
stage re-reads `modeling_k2_horizon.py` and confirms each line before any kernel is written.

**Re-read 2026-10-05 (18a, `docs/probe-k2-2026-10-05.md`):** every line above confirmed against the
vendored modeling file (file:line there); the operator's vLLM fork agrees on semantics and differs
only in rounding / summation order. Rounding detail the K1 references need: the router logits are
a bf16 linear before the fp32 sigmoid; RoPE's cos/sin are bf16 and `q·cos`, `rotate_half(q)·sin`
and their sum each round; the residual stream is bf16; MoVA normalises whenever top-k > 1. The
router biases are large and coarse (8-28, bf16 ulp up to 0.125), so exact selection ties are
plausible: ties go to the **lower id** (the reference's rule; `torch.topk` leaves them open). EOS:
`config.json` has 1, `generation_config.json` [1, 250019]. BOS 0 leads every prompt.

## 4. The decisions

**1. Scope: decode, prefill and serving** (spec 4 was decode only). Stages in §8.

**2. Context and cards (decide).**
- (A, proposed) **One card, 32k with bf16 KV** (~28.2 GB with the int8 head, derived), **~64k with
  spec 12's int8 KV**; long context beyond that through spec 16's PP on two cards.
- (B) One card at 16k only (spec 4's ruling), everything longer deferred.

**3. Routing under replay: spec 4's ruling A, refined by spec 15.** Expert ids and weights stay on
the device; kernels address experts by offset in flat per-layer buffers (spec 4 §3.4's strides); the
slot launches are merged into spec 15's "all selected experts in one launch" kernels. Combine order
is ascending expert id (K2's reference order; Ornith's is slot order: the combine kernel takes the
order as a parameter).

**4. Reference and quality.** Our own CPU reference: the model's `modeling_k2_horizon.py`
(`trust_remote_code`), layer-streamed with spec 14's `tools/oracle/stream.py` on dequantised weights,
plus per-layer MoE and MoVA expert ids for the routing diagnostic (spec 4 §4). Golden sets on K2's
own tokenisation (spec 4 §4).

## 5. Design

### 5.1 Structure (spec 4 §3.1, kept)

`model::K2Horizon` (constants, layer kinds, linear table, expert groups), `loader::load_k2()`,
`src/runtime/k2/` (`K2Engine`, capture), `src/kernels/k2/` for K2-only kernels; the MoE kernels come
from spec 15; `b70-decode` / `b70-serve` dispatch on `model_type`. The qwen3_5 paths are untouched.

### 5.2 Decode

Per sparse layer (target): grouped norm (2) · fused q‖k‖gate‖v_router GEMV · MoVA top-4 · value
experts (one launch over the 4 selected) + combine · RoPE + cache write · attention decode (spec 10 v2
at head_dim 128, GQA 4) · reduce + softplus gate · o_proj · grouped norm (2) · MoE router (bf16, padded
to 128 rows) + top-8 · expert gate‖up for the 8 + shared (one launch) · down + combine + residual (one
launch). Dense layers and the head as in spec 4 §3.3.

### 5.3 Prefill

Per chunk: dense and attention linears through spec 5's path; MoVA: route all tokens, group by value
expert, grouped GEMM (2560 → 1024 per expert), SiLU, weighted combine in ascending id; flash attention
at head_dim 128 (spec 6 variants); MoE through spec 15's grouped path with K2's router.

### 5.4 Serving

K2's chat template is its own (51 KB, a configurable `tool_call_format`, `reasoning_content`): the
renderer must handle it (checked against HF's `apply_chat_template`), and the tool-call parser gains
K2's format if it differs from Qwen XML. Prefix cache: KV-only snapshots. No MTP.

## 6. Correctness gates

- **K0, nothing moves:** Qwen3.8 (and Agnes, Ornith once merged) bitwise unchanged.
- **K1, kernels:** every new kernel and variant against a host reference (spec 4 §4's list: grouped
  norm, softplus threshold branch, sigmoid top-k with selection bias and ascending order, expert slots
  bitwise against a plain GEMV, combine, attention at head_dim 128 / GQA 4).
- **K2, golden:** three prompts x 32 greedy tokens against the CPU reference, the tie-aware rule;
  the per-layer routing diagnostic (first position where the engine's expert ids differ).
- **K3:** replay determinism bitwise; prefill split tests (multiples of the chunk bitwise); prefix-cache
  C1 / C2 (KV-only snapshots).
- **K4, quality:** A4 tool calls against the bf16 model (no bar until measured); passkey at the chosen
  context.

## 7. Speed

Recorded, no bars before P0: decode at 4k / 16k / 32k depth (`--lm-head int8`), launches per token,
pp4096 and pp at 32k; against the derived roofline (~190 t/s weights-only with the int8 head; spec 4
§5's ~95 t/s estimate at the 27B's utilisation, before launch reduction). **The vLLM baseline** (vLLM with K2-Horizon support from
[vllm-project/vllm#56637](https://github.com/vllm-project/vllm/pull/56637) and the project's local
B70 patches for W4A16 MoE on XPU: grouped W4A16 GEMV for experts, sparse expert dispatch, a fused
gate‖up GEMV, a dense W4A16 GEMV; measured 2026-09-16): **two B70s, PP = 2 (layer partition 25 / 23),
fp8 KV, 390,016-token context, 44.43 t/s decode** (llama-benchy). That is the row to beat: one card
for our engine at 32k-64k, and spec 16's PP for the long-context comparison at matched settings.

## 8. Stages

- **18a, facts and reference (CPU):** re-read `modeling_k2_horizon.py` against §3; the layer-streamed
  reference; golden sets and routing dumps; the tokenisation and BOS decision; the K2 chat template
  check. Runs on the box CPU, or in Docker on the Mac if the 22.2 GB checkpoint fits there
  **(decide)**.
- **18b, decode:** model table, loader with flat expert buffers, K2 kernels (grouped norm, softplus
  gate, MoVA value experts), spec 15's MoE kernels with K2's router, `K2Engine` decode; K0, K1, K2, K3
  (decode); decode speed.
- **18c, prefill:** MoVA and MoE grouped paths, flash attention at head_dim 128; K2 / K3 on prefill;
  prefill speed.
- **18d, serving:** template and tool calls, prefix cache, the server, K4, the record.
- **18e (optional), long context:** PP on two cards (spec 16) and / or int8 KV (spec 12) for K2.

## 9. Out of scope

Speculative decoding for K2 (no MTP head; a draft model would be its own spec); TP for K2; fp8 KV;
contexts beyond what §4 decision 2 sets; training-only parts (the router's load-balancing loss).

## 10. 18b as built blind (2026-10-05, branch `spec18b-k2-decode`)

Written on the Mac with the box unavailable: plan 18b Tasks 1-3; Task 4 (speed) and every run on
the card are the box's (`docs/superpowers/plans/box-validation-queue.md`, the K2 row). Rebased on
18a's reference (`912210e`) and aligned with `docs/probe-k2-2026-10-05.md`.

**Structure (§5.1).** `model::K2Desc` / `model::k2()` (`src/model/k2_horizon.{h,cc}`: every number
from the int4 checkpoint's `config.json` and safetensors headers; `check_k2_config` holds
config.json's structure to it), the loader beside `loader.cc` (`src/loader/k2_layout.h` sizes every
allocation device-free, `k2_repack.{h,cc}` is the host half - names checked both ways and refused by
name - `k2_rope.cc` the fp32-step RoPE table, `k2_loader.{h,cc}` uploads and holds every bucket to
the layout's formula), `src/runtime/k2/` (`K2Buffers`, the capture, `K2Engine`, the planner), K2's
kernels in `src/kernels/k2/` (`k2_prep.cl`, `k2_attn.cl`, `k2_moe.cl`; names in
`src/kernels/k2_kernels.h`). No existing `.cl` changed: the reused sources (`gemv.cl`,
`gemv_bf16.cl`, `gemv_i8w.cl`, `prep.cl`'s `prep_res_fold`, `embed_gather.cl`, `argmax.cl`) run at
K2's shapes from new CMake lines, so every pre-existing binary keeps its command line (Mac
`kernel_cmdlines`: additions only).

**Device layouts.** Non-expert int4 linears GPTQ layout 0; the fused attention row is
`q | k | gate | v` (dense, 10240) or `q | k | gate | v_router` (MoVA, 9280: the int4 router's 64
logits ride in v's place); dense gate||up interleave16. Expert groups are flat per-layer layout-1
block arrays addressed by id inside the kernels (decision 3): value 64 x 1,392,640 B, gate||up and
down 101 blocks each (the shared expert is block 100) of 2,088,960 / 1,044,480 B - 405.6 MB per
MoVA/MoE layer, 18.25 GB over 45. The MoE router is bf16 [128][2560], rows 100..127 zero (Review
Focus 4). Norms fp32 plain `w`; the two selection-only biases fp32 (the F16 MoVA bias widened
exactly). Totals (derived, `k2_horizon_test`): 21.802 GB of weights with the bf16 head, 21.161 GB
with the int8 head; read per token 3.786 / 3.145 GB.

**Kernels (§5.2).** The grouped norm is `prep_res_fold` (unchanged) + `k2_norm_finish` (each group
of 1280 its own Σx², plain `w`). `k2_attn_prep` mirrors the reference's bf16 RoPE chain exactly
(three roundings; the products of bf16 values are exact in fp32). `k2_route` is the sigmoid
router: `fp32(sigmoid(fp32(bf16 logit))) + fp32(bias)` for selection, ties to the lower id, the
100 real experts on 128 lanes with the padding never ranked, Σ s in rank order, `rne((s/Σ)·2.5)`,
slots in ascending id. `k2_moe_gate_up` / `k2_moe_down` are moe.cl's kernels with the shared expert
ungated and the reference's ascending-id bf16 `index_add_` chain (moe.cl's fp32 slot-order sum is
another chain, so it is a new source and Ornith's binaries are untouched). `k2_mova_value` runs the
4 value experts in one launch and writes v into the layer's V cache. `k2_attn_decode` /
`k2_attn_reduce` are attn_v2.cl's structure at head_dim 128 / GQA 4 (8 sub-groups, two positions
each) with the softplus gate (beta ln 2, threshold 20) in the reduce.

**Launches (Review Focus 5): 717 per token** = embed + 3 dense x 12 + 45 MoVA/MoE x 15 + 5,
asserted at capture - against §2's ~1000 estimate and spec 4's 2067. Per MoVA/MoE layer: fold +
norm, fused GEMV, MoVA route, value experts, attention prep, decode + reduce, o_proj, fold + norm,
router GEMV, MoE route, gate||up (8 + shared), down + combine + residual.

**Memory (decision 2).** `--max-len auto` plans with `runtime::k2::plan`: at 32768 the engine holds
28.263 GB (bf16 head) / 27.622 GB (int8 head) - decision 2 (A) fits; on a 32.53 GB card with the
1.5 GB reserve auto gives 46592 / 49920 (derived). bf16 KV only (int8 KV is 18e).

**CLIs.** `b70-decode` dispatches on config.json's `model_type`: `--ids` / `--bench` run
`K2Engine` (`--lm-head bf16` default, `int8` available); `--prefill` / `--pp` are refused naming
18c, `--kv-cache int8` naming 18e, `--profile` naming Task 4. `b70-serve` refuses K2 before the
device naming 18c / 18d.

**Gates as written.** Host (Mac): `k2_horizon_test`, `k2_rope_test`, `k2_repack_test` (a
synthetic checkpoint in K2's naming), `k2_ref_test` (Review Focus 1, 2, 4 by independent
formulas), `k2_plan_test`, `k2_variant_names_test`, and `tools/mac/clrun/k2_run` (the portable
kernels on the Mac's GPU against `k2_ref.h`: bit-exact on the UHD 630, indicative). Box:
`k2_kernels_test` (K1 at K2's real shapes), `k2_load_checkpoint_test`, `k2_decode_test` (the list,
plan = allocation, K3 replay bitwise incl. route rows and KV), `k2_golden_test` (K2: the tie-aware
token gate and the routing diagnostic against 18a's `route.{moe,mova}.{ids,w,gap}.L*`; SKIP until
`oracle-out-k2/` exists), the `cli_reject_k2_*` dispatch cases.

**Known deviations and risks, for the box.**
- Decode attention computes the softmax in fp32 flash-style and does not round the scores or the
  probabilities to bf16 as the reference's eager path does (scores bf16, x scale bf16, softmax
  fp32 -> bf16). This is the engine's existing convention (attn.cl / attn_v2.cl); if the K2 golden
  gate fails on determined rows, it is the first suspect, and an eager-mirroring reduce is the fix.
- The router's Σ s is a sequential fp32 sum in rank order; torch's 8-element reduction may differ
  by an fp32 ulp - after the bf16 rounding of `w` rarely visible (the diagnostic prints the worst
  weight difference; it does not gate on it).
- The routing diagnostic's near-tie tolerance (1e-3 on the reference's gap at the cut) is a
  proposal until 18a's real-weight run prints the gap distribution; `B70_K2_TIE_TOL` overrides.
- The int4 rows' `{S, layout}` (q||k||gate||v S2, o_proj / gate||up / down S4, layout 0) and the
  expert kernels' K splits are PROVISIONAL (copied), for Task 4's sweep.
- The int4 checkpoint's config.json says `"dtype": "float16"` (AutoRound's export); the engine and
  18a's reference compute in bf16 (§3) - a reference run in fp16 would not be the gate's.
