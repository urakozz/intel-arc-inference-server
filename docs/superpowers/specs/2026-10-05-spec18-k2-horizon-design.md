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
`K2Engine` (`--lm-head bf16` default, `int8` available); `--prefill` / `--prefill-length` are refused naming
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

### 10.1 Amendment: the eager attention variant (2026-10-05, branch `k2-attn-eager`)

The first suspect above now has its fix built beside the default, so the first box session can
A/B the two without a development cycle. **`B70_K2_ATTN=flash|eager`** (default `flash`: nothing
changes) is read at capture, as `B70_DECODE_ATTN` is (`runtime::k2::k2_attn()`, k2_sizes.h; an
unknown value throws). `eager` binds `src/kernels/k2/k2_attn_eager.cl` (a new source: `k2_attn.cl`
and its binary are untouched, `kernel_cmdlines` +1 / -0 / ~0) in place of `k2_attn_decode` +
`k2_attn_reduce`, with the same inputs (`attn_q`, the caches, `attn_gate`) and output (`attn_out`):

| launch | grid | does |
|---|---|---|
| `k2_attn_eager_score` | (kv heads, 32) | `s_p = rne(f32(rne(q·k_p)) · scale)` into a score row `attn_s` fp32 [M][q_heads][max_len] |
| `k2_attn_eager_softmax` | (q heads, M) | one work-group per row: exact max; `e = exp_torch(s - max)`; Σ e in torch's order; `p = rne(e · (1 / Σ))` in place |
| `k2_attn_eager_pv` | (kv heads, 32) | per key block, `Σ p·v` fp32 (fma chain, keys ascending) into `attn_part` |
| `k2_attn_eager_reduce` | (q heads, M) | blocks added ascending, `rne`, then the unchanged softplus gate |

**The op chain is the reference's** (`k2_ref.py attention()` = HF's `eager_attention_forward` in
bf16): the dot rounded once to bf16 (the bf16 GEMM), then `* scaling` rounded again; torch's fp32
softmax over the full row, rounded to bf16; P·V in fp32, rounded once; the gate as before.
**What is bitwise, and what is not:**
- the softmax, exactly: `SoftMaxKernel.cpp`'s `_vec_softmax_lastdim` at AVX2 (the oracle
  container's and the box's capability) is `Sleef_expf8_u10` of `s - max`, an 8-lane
  `reduce_all` sum (lane j sums e_j, e_{j+8}, ... ascending, then
  `((l0+l4)+(l2+l6))+((l1+l5)+(l3+l7))`), one reciprocal, a multiply. `exp_torch` (in the kernel
  and in `tests/kernels/k2_ref.h`) is Sleef's xexpf line for line with fma. Probed against torch
  2.14.1 in `agnes-ref-img`: 22,760 fp32 probabilities, 0 differ (`std::exp` instead: 2,101; a
  sequential sum: 20,069). A row shorter than 8 keys is summed as the prompt pass sums it
  (zero-padded); only a decode-only row under 8 keys would be torch's sequential path.
- every rounding point of the score and of P·V, exactly. The two GEMMs' fp32 accumulation
  orders are ours (one fma chain; bf16 x bf16 products are exact, so fma = mul + add and nothing
  can be contracted), not torch's (which differs between its prompt pass and its decode steps):
  the two part only where an fp32 sum sits within its order noise of a bf16 boundary.
- measured end to end (`k2_attn_eager_ref_test` against `tests/kernels/k2_attn_eager_fixture.h`,
  generated by `tools/oracle/k2_attn_eager_fixture.py` in the container): scores, probabilities,
  P·V and the whole chain bitwise - 5,456 scores, 5,456 probabilities, 8,192 outputs - on three
  cases (exact-dot prompt rows, a 300-key decode row, 4 prompt rows over 40 keys). Flash's
  arithmetic (taken in fp64: fp32 scores and probabilities, one rounding) lands on another bf16
  output than torch's for **5,742 of those 8,192 outputs** (70 %): the systematic difference the
  review flagged, now quantified.
- the kernel against `k2_ref.h`: bitwise by construction (the same functions in the same
  orders; `-cl-fp32-correctly-rounded-divide-sqrt` makes `1 / Σ` correctly rounded), shown on the
  Mac's UHD 630 (`mac_check --kernels`: scores, probabilities, outputs 0 differences at 6 / 300 /
  1024 keys; indicative) and asserted on the card by `k2_kernels_test` §9 (6 / 300 / 3000 / 4096
  keys; gates past softplus's threshold bitwise, below it within 1 ulp for OpenCL's `exp` /
  `log1p`). One assumption: below `s - max < -87.3` an e is subnormal and the device must keep
  fp32 denormals for p to match bit for bit (P·V is unaffected at that size).

**Why two passes:** the reference rounds every probability against the whole row's max and sum
before P·V; flash's online merge rescales partial sums by `exp(m_old - m_new)`, another fp32
value, and cannot round p before the row's sum exists. **Cost:** 813 launches per token for 717
(4 attention launches a layer for 2); the score row is 4 B x q_heads x max_len (4 MiB at 32k,
`runtime::k2::attn_scores_bytes`, allocated only under eager, counted by the planner); the
softmax's 8 lane chains are `L / 8` dependent adds (512 at 4k, 4096 at 32k) - expect a measurable
decode cost at depth, unmeasured. Plain OpenCL C, no sub-group operations or block reads.

**When to switch the default.** The box runs `k2_golden_test` and `k2_golden_eager_test` (and
the `_i8head` twins; both registered, SKIP 77 without the checkpoint or `oracle-out-k2/`) and
compares: determined-row token failures, and the routing diagnostic's non-tie / near-tie counts
and first differing rows per layer. Make `eager` the default (`kDefaultK2Attn`) if flash fails
determined rows that eager passes, or if both pass and eager's routing diagnostic is strictly
closer (fewer near-tie differences, later first differences); keep `flash` if the two agree or
eager does not improve the routing - then the difference is not where the gate breaks. Either
way record eager's decode cost at 4k / 16k / 32k (`B70_K2_ATTN=eager b70-decode --bench`,
interleaved with flash); if eager becomes the default and costs more than ~3 % at 32k, the next
step is a faster eager (sub-group dots, a parallel max, the lane chains split across work-items
within each lane's order), not a return to flash.

**Prefill (18c, §11) reads the same switch.** One variable, one parser: `runtime::k2::k2_attn()`
(k2_sizes.cc), which 18c's `prefill_attn_eager()` now returns; `k2_prefill_eager_test` therefore
compares eager prefill against eager decode, and `k2_prefill_test` prints both halves and checks
they agree. The two eager forms share the rounding points, not the softmax's details: prefill's
`k2_pf_flash_attn_*_EAGER` takes the max and Σ online (rescaled partial sums), OpenCL's `exp`,
and `p = rne(e / l)` (a division), where torch - and decode's eager - take the exact max,
Sleef's exp, the 8-lane sum and one reciprocal multiply. To be torch's bit for bit, prefill would
need decode's chain per query row: the scores rounded twice, the full-row softmax in torch's order
(the prompt pass pads rows with exact zeros, which leaves the 8-lane sums identical to the
unpadded row's), p rounded to bf16, P·V in fp32. A score row per chunk is T x heads x L x 4 B
(8 GB at a 2048-row chunk over 32k keys), so it would recompute the scores per pass (max; lane
sums; rounded p into a bf16 DPAS P·V - p is bf16, so the second GEMM takes it as is) rather than
store them. Not needed unless the golden prefill rows fail where the decode rows pass.

## 11. 18c as built blind (2026-10-05, branch `spec18c-k2-prefill`)

Written on the Mac with the box unavailable: plan 18c Task 2 (kernels, the prefill step, the
engine, the CLI, tests). Task 1 (P0 timings) and Task 3 (speed) and every run on the card are the
box's (`docs/superpowers/plans/box-validation-queue.md`, the K2 prefill row).

**Structure (§5.3).** K2-only kernels in `src/kernels/k2/` (`k2_pf_linear.cl`, `k2_pf_moe.cl`,
`k2_pf_attn.cl`; names in `src/kernels/k2_kernels.h`'s spec 18c block), the walk and its scratch
in `src/runtime/k2/k2_prefill.{h,cc}`, `K2Engine::prefill` in `k2_prefill_engine.cc` - a new
archive `b70_k2_prefill` beside `b70_k2_runtime`, so a decode-only target links what it did
(Engine::prefill's arrangement). No existing `.cl` changed: the reused sources run at K2's shapes
from new CMake lines (`tools/kernel_cmdlines`: additions only).

**The chunk (one Level Zero list, no host wait inside it):**

| step | kernels | rounding chain |
|---|---|---|
| embed | `pf_embed.cl` at K2's hidden / vocabulary | a copy |
| grouped norm | `pf_res_fold` (stage A, runtime M, one slice) + `k2_norm_finish` at M = kPfC (decode's stage B) | decode's, row for row |
| int4 linears | `k2_pf_dequant_slab` (layout 0, slabs of 1024 and one zero-padded tail of pad256) + `pf_gemm` (unchanged); dense gate‖up through `pf_gemm_T0_SILU` | dequant then DPAS (spec 2.1's) |
| attention prep | decode's `k2_attn_prep` built at M = kPfC, S = 1 over the partials row at pitch pad256(N) (10240 / 9472) | decode's bf16 RoPE chain, bit for bit |
| MoVA | decode's `k2_route` on grid (1, C) over the v_router columns; `k2_pf_sort`, `k2_pf_gather`, `k2_pf_dequant_v` + `pf_moe_gemm` (2560 → 1024, plain epilogue), `k2_pf_mova_combine` into V at pos + t | `k2_mova_value`'s epilogue: SiLU, ascending-id bf16 chain |
| attention | `k2_pf_flash_attn` (new: head_dim 128, GQA 4, HPW 4, RPW 8, KT 64), the softplus gate fused in the epilogue | see "precision" below |
| MoE | `pf_gemv_bf16` router at decode's {16, 16} tiling (row m bitwise decode's GEMV), decode's `k2_route` binary on grid (1, C), sort / gather, gate‖up (`pf_moe_gemm` SiLU, 2 weight batches) and down (1 batch), `k2_pf_moe_combine` | `k2_moe_down`'s epilogue: ascending-id bf16 chain, the ungated shared expert, the residual fold |
| head (last chunk) | decode's binaries over the last row: fold + norm, lm_head, argmax | decode's |

**Launches: 2392 per chunk at every C** (1 + 3 x 62 + 45 x 49; `runtime::k2::prefill_chunk_launches`,
asserted by the walk) plus 5 for the head. Per sparse layer: 2 norm, 20 attention-row slabs (9 + the
tail), 6 MoVA, prep + flash, 6 o_proj slabs, 2 norm, 11 MoE.

**Decisions taken blind.**
- **l0 only.** spec 5's h8 (`l0-int8`) rotates in 1024-k Hadamard blocks and K2's hidden is 2560;
  sycl-tla has no K2 walk. `b70-decode --prefill-backend l0-int8 | sycl-tla` on K2 is refused by name.
- **K2's own sort** (`k2_pf_sort`): pf_moe.cl's needs a multiple of 16 experts (K2's MoE has 100)
  and always appends a shared expert (MoVA has none); the tile table, the padding to tmax, the
  header and the determinism argument are pf_moe.cl's, and for the MoE shape the host reference
  is checked equal to spec 15d's walk (`k2_pf_ref_test`). The grouped GEMM is `pf_moe_gemm.cl`
  unchanged (row independence, Review Focus 2, is its argument).
- **Every expert dequantised to bf16 per chunk** (the Ornith arrangement): one weight batch of
  401 MB (half the gate‖up blocks; all down / value blocks), experts with no row skipped. It is
  the largest term of a chunk (derived ~1.5 GB of bf16 per sparse layer written and read); an
  SLM-fused dequant inside the grouped GEMM is plan 18c Task 3's first lever.
- **The prefill scratch: 0.797 GB** (kPfC 2048: partials 84 MB, the weight batch 401 MB, the
  sorted A / y 110 MB, ...; `runtime::k2::prefill_sizes`), lazy on the engine and planned only
  when a run prefills: `--max-len auto` with `--prefill` / `--prefill-length` gives 42752 positions (bf16 head,
  32.53 GB card, 1.5 GB reserve; 46592 decode-only).
- **Natural `exp` in the flash attention** (EXP2 0), as K2's decode and the reference - PROVISIONAL,
  a Task 3 speed knob (spec 6c's exp2 was a register-pressure fix at head_dim 256).

**Precision against the reference (18b review).** The reference's eager attention (bitwise HF in
bf16, `tools/oracle/k2_ref.py attention()`) rounds four times: s_b = rne(q·k), s = rne(s_b x
1/sqrt(128)), p = rne(softmax(s)) (fp32 inside), o = rne(Σ p v). The default prefill attention
keeps the scaled scores in fp32, runs the softmax online in fp32 and rounds only the UNNORMALISED
exp(s - m_running) to bf16 as P·V's DPAS operand, dividing by l in fp32 at the end; decode's v2
does not round P at all (fp32 weights against bf16 V). So the two engine paths differ from each
other and from the reference by those rounding points - small per layer, but K2's routers are
tie-sensitive (biases 8-28 in steps up to 0.125) and the difference compounds over 45 layers. The
opt-in **`B70_K2_ATTN=eager`** prefill variant (`k2_pf_flash_attn_*_EAGER`, two passes: max and
Σ, then p = rne(exp(s - m) / l) and o = Σ p v rounded once) moves the rounding points to the
reference's (not its sum orders); `k2_pf_ref::attention_eager` is its host model. The decode half
of the switch is §10.1's `k2_attn_eager.cl` (one variable, one parser: `runtime::k2::k2_attn()`),
so Review Focus 1's comparison (`k2_prefill_eager_test`) runs both halves eager.

**Determinism and continuation.** No atomic anywhere; the sort is one work-group walking the
chunk in (token, slot) order; the combines sum in ascending expert id; a grouped GEMM row is its
own A row times its expert's block whatever tile it rides in; the flash attention's key tiles
start at key 0 (a wholly masked tile adds exact zeros). Every kernel is row-local or keyed to
absolute positions and K2 has no GDN chunking, so the argument predicts a split prefill bitwise
equal to the one-call prefill at EVERY split; the gate (`prefill_split_k2_test`) holds the
multiples of 64 bitwise and the rest to the split test's bars, printing which were bitwise.

**Gates as written.** Host (Mac): `k2_pf_ref_test` (the routing scatter with exact ties at the cut,
the tile bound and its adversary, row independence, the combines == decode's chains bit for bit,
the slab tail, the eager reference), `k2_pf_variant_names_test`, `k2_plan_test` (2392 launches,
0.797 GB, 42752), `tools/mac/clrun/k2_run` (the portable prefill kernels bit-exact on the Mac's
GPU, indicative). Box: `k2_pf_kernels_test` (K1: everything above on the card, grouped == dense
bitwise, flash attention against fp64 at depths 0 / 2k / 30k / 60k, a tail chunk and single rows,
Review Focus 1-3 at kernel level), `k2_prefill_test` (the walk, K3 determinism / recorded replay /
chunking bitwise, prefill KV and routing against decode's fill, tokens by ruling A26; `_i8head`,
`_eager`), `prefill_split_k2_test`, `k2_golden_prefill_test` / `_c16` / `_i8head` (K2 on prefill
with the routing diagnostic, Review Focus 5), `cli_reject_k2_prefill_length_int8` / `cli_reject_k2_prefill_sycl`.
The PROPOSED bars (KV vs decode: dense rows >= 0.999, median >= 0.9998, p01 >= 0.99; routing
near-tie margin 2e-2, weights 1/32) are set from the box's first printed distributions.

**Speed (derived, unmeasured).** ~9.3 GFLOP per token (45 sparse layers x 98 M MACs: the
attention row, o_proj, 4 value experts, 8 + 1 MoE experts) - pp4096 ~38 TFLOP of DPAS (~0.3-0.4 s
at spec 2.1's measured rates) plus the per-chunk expert dequant pass (~0.3 s per chunk) plus the
flash attention (~0.05 s): ~1.2 s, ~3,400 t/s, the dequant pass the largest term. Task 3 measures.

## 12. 18d host side as built (2026-10-05, branch `spec18d-k2-serving-host`)

Written on the Mac without the box: plan 18d Task 1's host half - Review Focus 1 (template), 2
(tool calls) and 5 (EOS); rebased on 18c (§11). The engine half of serving (a K2 engine behind
`b70-serve`, Review Focus 3's KV-only snapshots, K4, the comparison rows, the record) is not built:
it needs 18c's prefill validated on the card first, so `b70-serve` still refuses K2 before the
device, now with K2's chat format wired behind the refusal.

**Template (Review Focus 1).** The served (int4) repo's `chat_template.jinja` (`60c364d9…`,
51,584 B, repo commit `0e38c26c`) renders **byte for byte** as transformers 5.15's
`apply_chat_template` on 18 message lists (`tests/tokenizer/k2_template_cases.json`, renders by
`tools/tokenizer/dump_k2.py` in `agnes-ref-img`): plain, thinking, `reasoning_effort` medium / low
with `think_fast` / `think_faster` history, assistant history without a thinking field and with
`reasoning_content: null`, list content (`extract_text`), tools in the default markdown
presentation and in `xml` / `json` presentation, tools without a system message, tool-call
history in each `tool_call_format` (`xml`, `xml_typed` - its `render_arg_type` over integer,
array and `anyOf` arguments - and `json`), parallel calls with tool results (string and list
content), a `"default": null` property and `$defs` / `$ref` schemas (markdown and xml).
`template_k2_test` = `template_test` over that file with prefix `k2` (the cases file may now be an object: BOS /
EOS expected, a tool table, per-case `kwargs`).

It took **renderer changes, no K2 special case**: minja was not Jinja in eight places K2's template
reaches, and each is now patched in `third_party/minja` (marked "b70 patch (spec 18d)", listed in
`third_party/VERSIONS` with the old and new header hashes) and tested against Jinja2's own render
by `minja_ext_test` (`tools/tokenizer/dump_minja_ext.py`): Undefined apart from None (`defined`
holds for a JSON null - the `default: null` case), `is [not] sameas`, `str.split()` without a
separator (Python's Unicode whitespace), the `replace` filter, the `dict` global, attribute getters
whose all-digit parts index (`rejectattr('0', ...)` over `| items` pairs - the `$ref` merge), an
empty mapping is falsy, the lower-case `none` literal. In `chat-template.hpp` the tool-call
capability probe also accepts an argument name between tags (`>argument_needle<`): without it minja
judged K2's template unable to render tool calls and polyfilled the history into JSON content. The
Qwen3.8 and Agnes template tests stay byte-identical (run on the Mac against their snapshots'
`chat_template.jinja` / `tokenizer_config.json`). Left as they were, outside every case here: a bare
None prints as "" (Jinja "None"; changing it would move Qwen's renders), `is sequence` is false for
strings and mappings, `default` also fires on None. And as for every model, the server holds a
request in `nlohmann::json`, whose objects are key-sorted: tool schemas and call arguments render
in sorted key order (the fixtures are rendered from sorted objects, as Agnes's were).

**chat_template_kwargs.** `server::Request::template_kwargs` keeps the request's
`chat_template_kwargs`; `TemplateIface::render_with_kwargs` / `chat::Template::render(..., kwargs)`
pass them as template variables beside `enable_thinking` - for models whose `server::ChatFormat`
reads them, K2 only (Qwen3.8 / Agnes keep reading `enable_thinking` alone, as before). K2's
`tool_call_format` and `reasoning_effort` are checked first (a bad value is a 400 naming it).

**Tokenizer.** The Rust `tokenizers` 0.22.2 crate loads K2's `tokenizer.json` unchanged (BPE
250,000 + 626 added, NFC, Split + ByteLevel, the `TemplateProcessing` BOS). `k2_tokenizer_test`
(`tools/tokenizer/dump_k2.py`'s `k2_tokenizer.json`): vocabulary 250,624 with the added tokens;
the 22 chat / think / tool tags both ways; 26 texts (prose, code, CJK, Cyrillic, Arabic, emoji,
NFD input, zero-width characters, digit runs, the tags inline, long runs, empty) - ids without
and with special tokens (BOS 0 prepended, as HF does), decode with and without them, the
streamer's pieces; `corpus.txt`'s 10,240 lines (145,440 ids) digest-equal in 10 chunks; three
chat renders (29 / 330 / 649 ids) equal to `apply_chat_template(tokenize=True)` - the template
writes the BOS, the server encodes without special tokens, one BOS. The IFM original's
`tokenizer.json` (differs in `truncation` only) passes too. As 15e did for Ornith, the served
repo's small files (`chat_template.jinja`, `tokenizer_config.json`, `generation_config.json`;
Apache-2.0) are vendored at `tests/tokenizer/k2/`, so `template_k2_test` runs everywhere;
`tokenizer.json` (20.6 MB) is not: the CMake cache path `B70_K2_TOKENIZER_JSON` (default the
int4 snapshot in the HF cache) or the environment variable of that name; absent, disabled on the
Mac and SKIP (77) elsewhere. Every 18d registration is one delimited block at the end of
`tests/CMakeLists.txt` (label `k2`).

**Tool calls and reasoning (Review Focus 2).** K2's format is not Qwen XML (18a), so
`server/toolcall_k2.{h,cc}` adds `K2OutputStream` beside the Qwen `OutputStream` (both now an
`OutputParser`): `xml`, `xml_typed` and `json` in K2's tags; reasoning verbatim up to the close tag
of the think tag the prompt ended in, or implicitly up to `<ifm|tool_calls>`; content verbatim,
dropped when only whitespace; each call emitted at its `</ifm|tool_call>`; values typed by the
tool schema as the Qwen path does (schema strings verbatim, otherwise the trimmed text as JSON),
`xml_typed`'s stated type where the schema has none; a call that does not parse becomes content
with its tags; a call cut by the end of generation with a complete name line keeps its complete
arguments. Read against vLLM's `k2_horizon` tool and reasoning parsers (`vllm-project/vllm` main):
the same tags, reasoning end, whitespace rule and value typing; ours streams each call at its close
(vLLM: the whole block at its end), keeps the good calls of a block holding a bad one (vLLM: the
block becomes content), does not check names against the tools (as our Qwen parser), and treats
output with no close tag as reasoning (vLLM's streaming path; its non-streaming one says content).
`toolcall_k2_test`: every split equal to the whole text (byte at a time and random 1-7 byte
pieces, spec 7a's rule) on every case; the round trip - all 9 assistant turns of the HF renders
above (7 calls, three formats) parse back to the messages the template rendered; the template's
examples and the edges; a fuzz of 3000 generated outputs (2034 parse to their generated structure,
966 cut at random points stream equal to whole). The 36 A4 outputs of Review Focus 2 do not exist
yet (18a's bf16 A4 run is pending; `tools/toolcall/score.py` reads Qwen XML only).

**Server dispatch (Review Focus 5).** `server::ChatFormat::for_model_type(config.json's
model_type)` - `k2_horizon` -> K2, anything else -> the Qwen path, unchanged - is
`server::Options::chat_format`; `make_output_parser` picks the parser and the reasoning tag from
the rendered prompt's ending (`<ifm|think>\n`, `<ifm|think_fast>\n`, `<ifm|think_faster>\n`) and the
call syntax from `tool_call_format`. `b70-serve` sets it from `model_type` before its K2 refusal.
EOS is `generation_config.json`'s `[1, 250019]`, which `b70-serve` already reads for every model.
`k2_server_test` (mock engine): kwargs reach the template, reasoning / content / typed calls and
`finish_reason: tool_calls`, both EOS ids stop, streamed frames carry the same as the body, bad
kwargs are 400s, and the default format renders without kwargs and leaves K2's tags as text.

**What remains of 18d.** A K2 engine behind `server::EngineIface` (an adapter like
`EngineAdapter`, which is `runtime::Engine`'s: 18c's prefill, step, sampling over 250,624 logits,
the snapshot calls) and lifting `b70-serve`'s refusal; Review Focus 3 (KV-only snapshots, the store
keyed by model); a short greedy chat through the server equal to `b70-decode`; K4 (A4 - with a K2
reader in `tools/toolcall/score.py` - and passkey); the comparison rows; the record.

## 13. 18e Task 1 as built blind: int8 KV on K2 (2026-10-06, branch `spec18e-k2-kv8`)

Written on the Mac with the box unavailable: plan 18e Task 1's code, tests and the accuracy probe's
tooling. Nothing has run on the card, and the probe has not run on real weights (the checkpoint is
not downloaded; Docker was busy). Defaults unchanged: `--kv-cache bf16` binds exactly the binaries
it did (`tools/kernel_cmdlines`: +11 / -0 / ~0).

**The scheme at head_dim 128 (Review Focus 1's object).** Spec 12 §8's `rotkv`, carried over:
`src/common/kv8.h` namespace `hd128` (host) and `src/kernels/k2/k2_kv8.cl` (device) are one definition.

- R = H_128 diag(s) / sqrt(128), **s = torch's `hadamard(128, 0)`** - derived as 12a derived the 256
  (`randint(0, 2, (128,), Generator().manual_seed(0)) * 2 - 1`), which is the first 128 of 12a's
  256 signs (the CPU generator draws one word per element, in order; checked with torch 2.2.2:
  58 of 128 negative). `kv8_test` and `test_k2_kv8_probe.py` pin it.
- 1 / sqrt(128) is not a power of two, so R is carried with exact scales: **K and V rows
  `rotate_kv(x) = s ⊙ FWHT(x) / 8` (= sqrt(2) x R), q `rotate_q(x) = s ⊙ FWHT(x) / 16`
  (= x R / sqrt(2)), the output `unrotate(y) = FWHT(s ⊙ y) / 16` (= y R^T / sqrt(2))**. So
  rotate_q(q).rotate_kv(k) = q.k and unrotate(Σ p rotate_kv(v)) = Σ p v exactly in exact arithmetic;
  every multiply of the scheme is exact, the FWHT stages one rounding each, ascending.
- The quantiser is 12b's at N = 128 (`quantise_n<128>`; the 256 path is the same template at 256):
  per (position, kv head) s16 = fp16_rne(amax / 127), q8 = clamp(rint(y / f16(s16)), ±127).
- What is rotated is what the bf16 cache would hold: K = the RoPE chain's bf16 output; **V = MoVA's
  routed mix after the value experts' combine (Review Focus 2)** on layers 3-47, the fused row's v
  on 0-2. Decode: `k2_mova_value` at `MOVA_STAGE 1` (a new k2_moe.cl define, default the old line)
  writes the mix to a staging row (`attn_out[m][kv_n]`, free until the attention writes it) and
  `k2_attn_prep_kv8` rotates and quantises it with K. Prefill: `k2_pf_mova_combine`'s own binary
  with pos 0 into the staging rows (`attn_out [C][kv_n]`), then the writer. No new allocation, no
  new launch.
- Readers: decode flash (`k2_attn_decode_kv8` / `k2_attn_reduce_kv8`: k2_attn.cl's orders on the
  exactly dequantised operands, the reduce un-rotating each head through SLM before the softplus
  gate); decode eager (`k2_attn_eager_{score,pv,reduce}_kv8` around k2_attn_eager.cl's unchanged
  softmax: the reference's rounding points on the int8 operands, P·V in fp32, un-rotated, rounded
  once, gated); prefill flash (`k2_pf_flash_attn_kv8`, default and EAGER: int8 as the exact bf16 DPAS
  operand, K's scale on the score, V's folded into P as rne(p s_v) - one rounding more than bf16 KV -
  and the **un-rotation fused into the epilogue**: a sub-group holds a row's 128 dims, FWHT stages
  1-8 by xor shuffles, 16-64 across registers, ascending - `k2_kv8_ref_test` shows the decomposition
  equal to `hd128::unrotate` bit for bit).

**Launches.** Unchanged: 717 per decode token (813 eager), 2392 per prefill chunk + 5 - every int8
binary replaces a bf16 one one for one.

**Memory (derived, `k2_plan_test`; `runtime::KvLayout` at 8 kv heads x 128).** 99,840 B per position
(48 x 2 x (1024 + 16)) against 196,608. `--max-len auto`, 32.53 GB card, 1.5 GB reserve:

| | bf16 head | int8 head |
|---|---:|---:|
| int8 KV, decode only | **91904** | **98304** |
| int8 KV, with the prefill scratch | **83968** | **90368** |
| int8 KV, eager decode (+ the score row) | 91648 | - |
| bf16 KV, decode only / with prefill | 46592 / 42752 | 49920 / 45824 |

Plan 18e's "~64k on one card" was conservative: the planner gives ~84-98k.

**Plumbing.** `K2Buffers` / `K2Engine` take a `KvCache` (default `runtime::default_kv_cache()`,
i.e. `B70_KV_CACHE`); the capture and the prefill walk bind by the buffers' form;
`runtime::k2::plan` / `max_len_that_fits` plan the KV term in the form; `b70-decode` accepts
`--kv-cache int8` for K2 (`cli_reject_k2_kv8` retired) and tags its rows `int8-kv`. `b70-serve`
still refuses K2 (18d). K2 has no snapshots yet (18d's engine side), so nothing there.

**Validated on the Mac (indicative).** Host: `kv8_test` (+ hd128), `k2_kv8_ref_test`,
`k2_plan_test`, `k2_kv8_variant_names_test`; every touched C++ file syntax-checked against Level
Zero; the 11 new variants clang-checked. On the Mac's GPU (UHD 630, `tools/mac/clrun/k2_run`): the
writer (decode and prefill, dense and MoVA-staged) - q, gate and scales bitwise, the int8 codes
bitwise except 1-2 of ~38k one code apart (Apple's OpenCL has no correctly rounded divide; the card
build has it); the staged MoVA row bitwise the bf16 build's V row; eager over int8 - scores,
probabilities and outputs 0 differences at 6 / 300 / 1024 keys. Synthetic K2-shaped attention
(fp64): rotkv per-head cosine >= 0.99993 at 6 / 300 / 3000 keys against per token's 0.99883.
The flash readers (sub-groups, DPAS) have not run anywhere.

**Review Focus 1 on real weights: tooling ready, not run.** `tools/oracle/k2_kv8_probe.py` (12a's
method on k2_ref.py; every variant with its own residual stream and cache, so the routing diagnostic
sees the routers' tie sensitivity), `test_k2_kv8_probe.py` (tiny weights: passes),
`tools/oracle/kv8_k2_repeat.sh` (the Mac run). The `_kv8` gate twins and K1 are box work
(box-validation-queue row 21); the tolerances (PROPOSED in `k2_kv8_kernels_test`: decode flash
within 2 ulps of fp64, eager bitwise its reference, prefill flash >= 0.9999 default / 0.999 eager)
are re-derived from the probe's numbers.
