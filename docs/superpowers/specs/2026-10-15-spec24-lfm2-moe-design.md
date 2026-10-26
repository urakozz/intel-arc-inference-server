# Spec 24 - LiquidAI LFM2.5-24B-A2B (`lfm2_moe`) in the engine

> DROPPED (operator, 2026-10-27): support removed from the engine in 4822e3e; kept for the record.
>
> Reusable later (revive from git: `git show 4c491f8:<path>` - 4c491f8, this branch's parent on main, is the last
> commit that has all of it):
> - the MoE router: `src/kernels/lfm2moe/lfm_moe.cl` (`lfm_router`; `lfm_route` - sigmoid, the expert bias for the selection only, renormalised weights; `lfm_moe_gate_up` / `lfm_moe_down`), `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h`; twins `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_pf_ref.h`; tests `tests/kernels/lfm2moe_ref_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/kernels/lfm2moe_pf_kernels_test.cc`
> - the pythonic tool-call parser: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`; `tools/toolcall/score.py --format pythonic`, `tools/toolcall/lfm2moe_rules.cc`
> - the rest: the short conv `src/kernels/lfm2moe/lfm_conv.cl` / `lfm_pf_conv.cl`, head_dim-64 attention `src/kernels/lfm2moe/lfm_attn.cl` / `lfm_pf_attn.cl`, `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `src/loader/lfm2moe_*`, `src/runtime/lfm2moe/` (`lfm_prep.cl` still live in `src/kernels/bolt/`)

**Status:** design, 2026-10-15, for operator review. Open decisions are marked **(decide)** and collected in §10.
Nothing is built.
**Plans (2026-10-15):** `docs/superpowers/plans/2026-10-15-spec24{a,q,b,c,d,e}-*.md` - 24a facts and CPU reference,
24q our AutoRound run, 24b descriptor and loader, 24c decode (the launch-count P0 first), 24d prefill, 24e serving.
**Box queue rows 48-53** (24a, 24q, 24b, 24c, 24d, 24e in that order; rows 42-47 are spec 23's, Muse-Glimmer, written in
parallel). Each plan's last task adds its row to `box-validation-queue.md` and its `row N` block to
`tools/box_validate/stages.sh` when it is built; this spec and its plans do not edit either file.

**Model:** `LiquidAI/LFM2.5-24B-A2B` (`Lfm2MoeForCausalLM`, `model_type: lfm2_moe`; bf16, 47,943,180,288 B in 10 shards,
7610 tensors; snapshot `1769516a`). Licence: **LFM Open License v1.0** (`license: other`, `license_name: lfm1.0`; §1.4).
No int4 of our kind exists (the published derivatives are FP8, GGUF, ONNX and MLX-4bit); 24q makes
`urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ`.

**Why this model.** The card (README) claims best-in-class agentic tool use at 2.45 B active parameters (LiveClaw 48.09
against Qwen3.6-35B-A3B's 38.19, published, not ours), post-trained with agentic RL inside OpenCode and similar
harnesses - the operator's target workload (long agentic coding sessions, correct tool calls, fast). At ~1.43 GB read per
token it is the cheapest model per token the engine would serve, and the whole model plus 131072 positions of KV fits
one card (§3).

**Order:** after spec 21 / 22 and beside spec 23. It reuses Ornith's MoE machinery (spec 15: expert slots addressed by
id inside the kernels, the grouped prefill GEMM), K2's sigmoid router with a selection-only bias (spec 18,
`src/kernels/k2/k2_moe.cl`), Kolibri's family pattern (spec 20: own descriptor / loader / kernels / engine, plain
`w · x̂` norms, per-head q / k RMSNorm, synthetic real-width checkpoints) and the GDN causal-conv ring discipline
(`src/kernels/gdn_step.cl`, `src/kernels/prefill/pf_gdn_conv.cl`).

Every number is **measured** unless marked **derived** (computed from the checkpoint's config and index),
**estimated**, **proposed** or **published**. Citations of the reference are to transformers **5.19.0**:
`transformers/models/lfm2_moe/modeling_lfm2_moe.py` (`M:line`), `configuration_lfm2_moe.py` (`C:line`), and by file
name for `cache_utils.py`, `integrations/moe.py`, `modeling_utils.py`, `generation/logits_process.py`,
`integrations/hub_kernels.py`, `masking_utils.py`, `conversion_mapping.py`.

## 0. The brief (operator, 2026-10-15)

- **Stages:** 24a CPU reference + facts sheet (tiny model bitwise against transformers; the bf16 original's golden / A4
  references on the box CPU), 24q our quant, 24b `ModelDesc` + loader, 24c decode (the launch-count P0 first; one card,
  then `--pp 2`), 24d prefill, 24e serving (tokenizer, template, pythonic tool-call parser, `<think>` split, sampling
  defaults incl. the repetition-penalty decision, prefix-cache snapshots carrying the conv rings).
- **Format:** the operator's one-format rule - AutoRound 0.17.0 (`tools/quantize/ar_pin.sh`) W4A16 g64 symmetric
  `auto_round:auto_gptq`, calibrated for agentic coding; the router, `expert_bias`, conv weights, norms and the
  embedding stay bf16 (fp32 for the bias). RTN g64 is the quick fallback if AutoRound cannot run the model. Gate: the
  quant against bf16 on A4 and held-out logit error, as 21q.
- **New engine parts:** the short-conv kernel family, the head_dim 64 GQA attention (decode and prefill flash), the
  K2-style router with renormalisation, MoE top-4 of 64 x 1536 on the existing expert-kernel structure, the two dense
  layers, the tied head (int8 head built from the embedding).
- **Out of scope:** int8 prefill (a later probe), the 1.2B / 8B LFM siblings.

## 1. The model

From `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, the index
and the README / LICENSE (read 2026-10-15), and the reference:

| | |
|---|---|
| layers | 40; `layer_types` (config.json, exactly): `conv` at 0, 1, 3, 4, 5, 7, 8, 9, ... 39; `full_attention` at **2, 6, 10, 14, 18, 22, 26, 30, 34, 38** (`l % 4 == 2`): 30 conv + 10 attention; layer 39 is conv |
| hidden | 2048; vocab 128000 (the tokenizer defines 125,017 ids: 124,893 BPE + 124 added, ids 0..125016 contiguous - so `vocab_used` 125017, rows 125017..127999 of the head never win); `max_position_embeddings` 131072 |
| norms | RMSNorm `w · bf16(x · rsqrt(mean(x²) + eps))`, variance in fp32, **two roundings** (x̂ cast to bf16, then the bf16 product; M:60-65) - Kolibri's form, not Qwen's `(1 + w)`; eps 1e-5 (`norm_eps`). Per layer `operator_norm` (before the mixer) and `ffn_norm` (before the FFN, M:483-484, M:495-513); the final norm is **`embedding_norm` applied after the last layer** (M:563, M:622), not after the embedding |
| short conv (30 layers) | `Lfm2MoeShortConv` (M:400-466): `in_proj` 2048 → 6144, no bias; `B, C, x = chunk(3)` in that order (M:437-438); `Bx = B · x` (M:439); depthwise causal conv1d, kernel `conv_L_cache` 3, no bias (`conv_bias` false), **no activation**; `y = C · conv(Bx)` (M:463); `out_proj` 2048 → 2048 (M:465) |
| attention (10 layers) | `Lfm2MoeAttention` (M:287-342): 32 q / 8 kv heads (GQA 4), head_dim 64 = 2048 / 32 (M:294), `q_layernorm` / `k_layernorm` RMSNorm over each head's 64 dims with weights [64] (M:302-303, M:316-317), full RoPE (`rope_type` default, θ 1e7, rotate-half, M:88-115, M:216-246), scale 64^-0.5 = 0.125 (M:296), `out_proj` 2048 → 2048, no gate, no bias |
| FFN | layers 0-1 dense SwiGLU (`layer_idx < num_dense_layers` 2, M:478-482): `w2(silu(w1 x) · w3 x)`, intermediate 11776 (M:118-128); layers 2-39 MoE: 64 experts, top-4, `moe_intermediate_size` 1536, **no shared expert** |
| router | `Lfm2MoeTopKRouter` (M:171-195): `l = F.linear(x, gate.weight)` [64][2048] bf16; `s = sigmoid(l)` in the logits' dtype (bf16); selection on `s + expert_bias` (an fp32 buffer, M:205; selection only); weights = `s` gathered at the selected ids; renormalised `s / (Σ s + 1e-6)` (`norm_topk_prob` true); `× routed_scaling_factor` 1.0 (§2.3) |
| embedding / head | `tie_word_embeddings` true: `lm_head.weight` is `model.embed_tokens.weight` (M:632); the checkpoint stores no `lm_head` tensor; logits = `lm_head(embedding_norm(h))` (M:690) |
| tokens | BOS 124894 `<|startoftext|>` (the template writes it first), EOS 124900 `<|im_end|>` (generation_config: [124900]), pad 124893; `<think>` 124901 / `</think>` 124902, `<|tool_call_start|>` 124905 / `<|tool_call_end|>` 124906 (added, **not** special), `<|tool_list_start|>` / `_end|>` 124903-4 (special, unused by the template) |
| tokenizer | BPE, `ignore_merges: true`, byte-level; pre-tokenizer split regex `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` (Llama-3-style digits in threes; no `\p{M}`); no normaliser; `tokenizer.json` sha256 `695be780...` |
| chat template | `chat_template.jinja`, sha256 `ea663864...`: ChatML, BOS first, tools as `List of tools: [<json>, ...]` appended to the system prompt, **pythonic tool calls** `<|tool_call_start|>[fn(a=1, b='x')]<|tool_call_end|>`, the generation prompt `<|im_start|>assistant\n<think>` (the model always reasons), `preserve_thinking` (default false), `{% generation %}` blocks (§4.6) |
| sampling defaults | `do_sample` true, T 0.1, top-k 50, **repetition_penalty 1.05**, no top-p (generation_config.json; the README repeats them) |

**Parameters (derived):** 23,971,587,712 including the 38 x 64 fp32 `expert_bias` values (the index's
`total_parameters` 23,971,585,280 leaves out exactly those 2,432). Routed experts 22.95 B; dense FFNs 0.145 B; conv
projections 0.503 B; attention projections 0.105 B; routers 4.98 M; the embedding 0.262 B. **Active per token (derived):
2.455 B** including the tied head (the README publishes 2.45 B): 4 of 64 experts in 38 layers (1.434 B), everything
non-expert (0.758 B), the head (0.262 B).

### 1.4 The licence (LFM Open License v1.0, `LICENSE` in the snapshot, sha256 `4d28ca14...`)

Apache-2.0-shaped with one addition. What it allows and requires:
- §2 / §3: a perpetual, worldwide, royalty-free copyright and patent licence to reproduce, prepare Derivative Works of,
  publicly display, sublicense and distribute - **subject to §5**.
- §4: redistribution of the Work or a Derivative (our quantised export is one) requires a copy of the licence, prominent
  notices on modified files, retained attribution notices, and a NOTICE file's contents if one exists (the snapshot has
  none).
- **§5 Commercial Use Limitation:** commercial use is licensed only to a Legal Entity under the **Threshold of USD 10
  million annual revenue**; above it, commercial use is not licensed (a qualified non-profit's non-commercial or research
  use is exempt from the threshold).
- §7: no trademark grant beyond describing the origin.
So 24q may publish our export with the licence file copied, the card stating "quantised from LiquidAI/LFM2.5-24B-A2B
(modified: int4 g64 AutoRound)", and `license: other` / `license_name: lfm1.0`; whether the operator's use sits under
the threshold is the operator's (decision 9). The repository stores no weights and no licence text; `tests/model/lfm2moe/
config.json` (24b) is a configuration file, which §4 also covers - 24b copies it with a header line naming the source and
the licence, or builds it in code if the operator prefers (decision 9).

## 2. Reference semantics (pinned from source)

Each paragraph below is a facts-sheet row in 24a, with its rounding points tested on the tiny model.

**2.1 The layer** (M:486-515), `h` bf16 [T][2048]:
```
r = h
h = (attention layer ? Attn(operator_norm(h)) : ShortConv(operator_norm(h))) + r        bf16 add, rounded
h = h + FFN(ffn_norm(h))                                                                bf16 add, rounded
after layer 39: logits = lm_head(embedding_norm(h))                                     bf16 linear (tied)
```

**2.2 RMSNorm** (M:50-65): `v = mean(f32(x)²)` over the row (torch's fp32 mean); `x̂ = bf16(f32(x) · rsqrt(v + eps))`;
`y = bf16(w · x̂)` with `w` the bf16 parameter. In fp32 runs (the tiny model's fp32 check) every cast is the identity.
The q / k norms are the same over each head's 64 values (M:316-317), before RoPE.

**2.3 The router** (M:171-195), per token, every op in torch's dtype:
```
l_b  = bf16(x · Wgᵀ)                       F.linear in bf16 (M:183)
s_b  = bf16(sigmoid(f32(l_b)))             .sigmoid() on a bf16 tensor: computed in fp32, rounded once (M:184)
sel  = f32(s_b) + bias                     bf16 + fp32 buffer promotes to fp32 (M:186)
ids  = topk(sel, 4)                        sorted, descending (M:187); torch's CPU tie order is NOT the lower id (spec 21 §12 finding 2)
w_b  = s_b[ids]                            gathered, still bf16 (M:188)
d_b  = bf16(bf16(Σ_k w_b) + 1e-6)          .sum() in bf16 (fp32 accumulation, one rounding), then + 1e-6 rounded again (M:193)
w_b  = bf16(w_b / d_b)                     bf16 division (M:193)
w_b  = bf16(w_b × 1.0)                     routed_scaling_factor (M:194): exact
```
The bias moves selection only. `+ 1e-6` is below half an ulp of any sum ≥ 2^-13 and so vanishes at bf16 unless the
four sigmoids are tiny - kept anyway, as the reference does. Where K2's router (`k2_moe.cl:122-128`) keeps `s` in fp32
and rounds the weight once, this one rounds `s`, `Σ`, the sum-plus-eps and the quotient: **a fourth router chain** (Ornith
softmax, K2 sigmoid-fp32, Kolibri no-renorm, LFM2 sigmoid-bf16-renorm), so a new route kernel (§4.3).

**2.4 The experts and the combine.** transformers 5.19.0 picks `grouped_mm` when `torch.nn.functional.grouped_mm` or
`torch._grouped_mm` dispatches (`modeling_utils.py:1888`, else `eager`); `integrations/moe.py:388-497`: per (token,
slot) pair `gate, up = chunk(rne(x · W_guᵀ))`, `a = rne(rne(silu(gate)) · up)` (`_default_apply_gate`), `d = rne(a ·
W_dnᵀ)`, `t_k = rne(d_k · w_k)` (`:484`), then `y = rne(Σ_k t_k)` with the four products summed in fp32 **in the
router's slot order** (descending `sel`, `:495`). The eager fallback (M:144-168) instead accumulates `index_add_` into a
bf16 tensor in ascending expert id, rounding every add - 24a records which one the reference ran (spec 21 found
`grouped_mm` on torch 2.14.1 CPU) and the engine copies that one. This is the combine moe.cl already implements for
Ornith (`moe.cl:16-37`: `t_k` rounded, fp32 sum in rank order, one rounding) **minus the shared expert**.

**2.5 The short conv** (M:400-466) on bf16 tensors:
```
B, C, x = chunk3(rne(in_proj(x̂)))         the linear's bf16 output; rows [0, 2048) B, [2048, 4096) C, [4096, 6144) x
Bx_b    = rne(f32(B) · f32(x))              M:439
c_b     = rne(Σ_{j=0..2} w_j · Bx_b[t - 2 + j])   F.conv1d over bf16 (M:370 decode, M:388-394 prefill), taps oldest first,
                                                    positions < 0 are zero (the state is zero-initialised)
y_b     = rne(f32(C) · f32(c_b))            M:463
out     = out_proj(y_b)
```
The conv weight `conv.conv.weight` is bf16 [2048][1][3]; `w_j` meets `Bx[t - 2 + j]` (F.conv1d's cross-correlation,
`weight[..., -1]` against the newest value - the order `gdn_step.cl:103-112` already pins for the GDN conv). The
accumulation inside torch's bf16 conv1d (its order and its fp32 or bf16 accumulator) is 24a's to measure: the engine
accumulates in fp32 with explicit `fma`, taps ascending, and rounds once - three exact bf16 x bf16 products, so a
different order can only move an fp32 sum that sits on a bf16 rounding boundary; 24a counts such rows on the tiny model
and on random real-width rows.

**The decode state** (`cache_utils.py:1100-1137`, `LinearAttentionLayer` for layer type `conv`, `:1292`): transformers
keeps the last `conv_L_cache` = 3 input columns (`Bx`, not the conv output), zero-padded at prefill; a decode step
convolves `[state(3), new(1)]` and keeps the last output (M:357-374), which reads only the newest two state columns. **The
engine keeps a ring of the last two `Bx` columns per conv layer** (16 slots at `pos % 16`, the GDN ring's depth, so M-row
launches never collide; §4.1). On CPU the `causal_conv1d` package is absent, so `use_kernel_func_from_hub_with_fallback`
runs the torch path (`hub_kernels.py:984-1014`); 24a prints which implementation ran.

**2.6 Attention** (M:305-342, eager M:261-283): q / k per-head norms (§2.2), RoPE `rne(rne(q · cos) + rne(rotate_half(q)
· sin))` with cos / sin computed in fp32 from `pos · inv_freq` and cast to bf16 (M:110-115, M:244-245), the cache holds
normed + roped k and raw v; eager: `rne(q · kᵀ)`, `× 0.125` (exact), `+ mask`, softmax in fp32 then bf16, `rne(p · v)`;
`out_proj`. **The reference pins `attn_implementation="eager"`** (transformers' default is `sdpa`, `_supports_sdpa`,
M:526): the eager chain is the one the engine's eager A/B mirrors (spec 18 §10.1's rule); flash is the default and is
held to fp64 (spec 6 K1).

**2.7 The model loop** (M:571-627): `h = embed_tokens(ids)` (bf16 rows; `padding_idx` affects training only);
`position_embeddings` once; 40 layers; `embedding_norm`; the tied head. `create_recurrent_attention_mask` returns `None`
for an unpadded batch (`masking_utils.py:1471-1501`), so the conv layers see no mask in every engine-shaped run.

**2.8 Repetition penalty** (`generation/logits_process.py:373-421`, `RepetitionPenaltyLogitsProcessor`): over every id
in `input_ids` - **the prompt and the ids generated so far** - `score = score < 0 ? score · 1.05 : score / 1.05`
(`:418-419`), applied to the logits before temperature / top-k (a processor, then the warpers). The engine has no
repetition penalty today: `server::Sampling` is `{greedy, temperature, top_k, top_p, seed}` (`src/server/deps.h:45-52`)
and the sampled path copies the logits row to the host and samples there (`src/cli/serve_adapters.h:335-341`,
`server::filter_probs`). Decision 1 (§10).

## 3. What it costs (derived)

| part (per decode token) | format | MB |
|---|---|---:|
| routed experts (38 x 4 x 3 x 2048 x 1536) | int4 g64 (0.53125 B / param) | 762.1 |
| conv `in_proj` + `out_proj` (30 x 16.78 M) | int4 g64 | 267.4 |
| dense FFNs (2 x 72.35 M) | int4 g64 | 76.9 |
| attention q‖k‖v + o (10 x 10.49 M) | int4 g64 | 55.7 |
| routers (38 x 64 x 2048) | bf16 | 10.0 |
| tied head, int8 per row (spec 9) | int8 + fp32 scales | 262.7 |
| **weights read per token** | | **1,434.6** |
| KV read at depth D (10 layers x 8 kv x 64 x 2 x 2 B = 20,480 B a position) | bf16 | 0.020 x D |

- **The bandwidth roofline** is 590 GB/s / 1.4346 GB = **411 t/s**; at Qwen3.8's 77.6 % of roofline (docs/05) **~319 t/s**
  (derived). With the bf16 head (1.696 GB) 348 / 270.
- **The step is not bandwidth-bound; it is many small kernels.** The list (§4.5) is **383 launches a token** (derived,
  as built) against Qwen3.8's 774 for 9.9x the bytes. What sets the step is each kernel's floor, not the bytes: in the
  27B's list a 1.31 MB GEMV takes 5.34 µs (245 GB/s, docs/12-kernels.md, measured), the between-GEMV kernels 1.6-4.8 µs
  each (`prep_gated_head`, `attn_prep`, `attn_reduce`, `argmax`; docs/15-step-anatomy.md, measured), and the dispatch
  gap 0.73 µs a launch (un-instrumented, docs/15, measured; `probe_replay`'s noop floor 0.52 µs). LFM2's GEMVs are
  2.2-25.6 MB, its expert launches 6.7-13.4 MB. **Estimated** from those floors: ~91 µs a conv + MoE layer, ~115 µs an
  attention + MoE layer at depth 4096, ~117 µs a dense layer, ~470 µs for embedding + final norm + head + argmax, 0.28 ms
  of gaps: **~4.5 ms a token, ~220 t/s at depth 4096** - about half the roofline. The gaps alone are ~6 %; the small
  kernels' own floors ~15 %; the rest is GEMVs too small to reach full bandwidth. **24c's P0 measures every term of this
  table before the engine's default list is chosen** (§4.5).
- **Depth matters more than for any model so far:** KV per token at depth 32768 is 671 MB (47 % of the weights), at
  131072 2.68 GB (1.9x the weights): ~4.5 ms → ~5.7 ms at 32k → ~9 ms at 128k (estimated, at v2's ~85 % of bandwidth on
  the attention reads). spec 12's int8 KV (rotkv) at head_dim 64 halves it - a later spec (decision 11).
- **Memory (derived), one card:** routed experts 12.193 GB, the other int4 linears 0.400 GB, routers 0.010 GB, the
  embedding bf16 0.524 GB (tiled, also the bf16 head - §5), the int8 head 0.263 GB, norms / conv taps / biases < 1 MB:
  **13.39 GB of weights**; KV 2.684 GB at 131072 positions (the trained maximum); the conv rings 1.97 MB (30 x 16 x 2048 x
  2 B); decode scratch < 16 MB; prefill scratch ~1 GB at chunk 2048 (estimated; 24d's planner states it). **~17.1 GB at
  the full 131072 context, against ~30.6 GB usable on a B70** (32 GB less the 1.5 GB reserve): the whole model and its
  whole context fit one card with ~13 GB to spare. `--pp 2` is therefore optional (decision 10); the spare memory is
  what spec 13's batching would use.
- **Per sequence state:** KV 20,480 B a position; conv state 245,760 B (30 x 2 x 2048 x bf16) - the prefix cache's
  per-snapshot state (§4.6), against Qwen3.8's ~151 MB of GDN state.
- **Prefill (estimated):** ~4.9 GFLOP a token active (2 x 2.455 B); a 2048-row chunk routes 8192 (token, expert) pairs over
  64 experts, ~128 rows an expert on average - a well-fed grouped GEMM (spec 15d's `pf_moe_gemm`, M ~128, K 2048, N 3072
  and K 1536, N 2048). At the bf16 DPAS rate (183 TFLOP/s measured peak, README) and spec 20d's utilisation the chunk is
  tens of ms; 24d records the rate (no bar before it runs).

## 4. The new parts

### 4.1 The short conv (decode)

One launch per conv layer after the `in_proj` GEMV, family kernel `lfm_conv` (grid over the 2048 channels, one work-item
per channel and token row):
```
B, C, x = rne(Σ_s slices[s][m][c]), rne(Σ_s slices[s][m][2048 + c]), rne(Σ_s slices[s][m][4096 + c])
Bx_b    = rne(f32(B) · f32(x));  ring[(pos + m) % 16][c] = Bx_b
c_b     = rne(fma(w2, Bx_b, fma(w1, f32(ring[p-1]), fma(w0, f32(ring[p-2]), 0))))    p-1 / p-2 < 0 read as 0
y[m][c] = rne(f32(C) · f32(c_b))
```
with the same ring-ownership argument `gdn_step.cl` makes: rows of one launch write `pos..pos+M-1` and read
`pos-2..pos+M-2`, disjoint mod 16 for M ≤ 14; within a launch a row's older taps come from its own work-item's registers
(`gdn_step.cl:294-330`'s pattern). The conv taps are stored fp32 (bf16 widened: exact). The ring stores `Bx`, never the
convolved value, as the reference's state does.

**The fused arm (P0's arm `c`):** `in_proj` rows permuted at load into 16-channel tiles `[B 16 | C 16 | x 16]`, so one
work-group of the in_proj GEMV holds a channel tile's B, C and x and runs the conv as its epilogue - one launch per conv
layer instead of two. Bitwise equal to the two-launch list when both use split-K S = 1 (the GEMV's fp32 order is
unchanged; only where the sum is consumed moves).

### 4.2 Attention at head_dim 64, GQA 4

No kernel in the tree has head_dim 64: `attn_v2.cl` and `pf_flash_attn.cl` hard-code HD 256 (`attn_v2.cl:78`,
`pf_flash_attn.cl:64`), K2's and Kolibri's HD 128 (`k2_attn.cl:55`, `kol_attn.cl:57`, `k2_pf_attn.cl:70`,
`kol_pf_attn.cl:73`). The family gets its own, on Kolibri's full-attention path (decode is spec 10's v2 structure,
prefill spec 6's DPAS flash):
- `lfm_attn_prep` - `kol_attn_prep` (`kol_prep.cl:116-230`) at HD 64 with eps 1e-5 (Kolibri's `KOL_EPS` is 1e-6,
  `kol_prep.cl:30`) and RoPE on every attention layer with a growing cache (Kolibri rotates only its sliding layers):
  q / k per-head plain norm, rotate-half RoPE from a [max_len][2][32] bf16-valued table (`kolibri1_rope.cc`'s method at
  θ 1e7, dim 64), the KV write.
- `lfm_attn` + `lfm_attn_reduce` - `kol_attn.cl`'s full path at HD 64: 4 sub-groups of 16 lanes a work-group (`NSG = HD
  / 16`), 4 positions of a 16-position wave per sub-group, 4 elements per lane, scale 0.125, the 8 kv heads x 4 q heads.
- `lfm_attn_eager` - the reference's bf16 chain (`kol_attn_eager.cl` / `k2_attn_eager.cl`'s four launches), selected by
  `B70_LFM_ATTN=eager` (spec 18 §10.1's switch rule).
- prefill: `lfm_pf_attn` (`kol_pf_attn.cl`'s full path at HD 64: 4 dim-atoms of O, K-dim 64 = 4 DPAS k-steps) with its
  `EAGER` twin.
Copies, not parameterisations of the Kolibri / K2 sources: no existing binary's source changes (F0; decision 12).

### 4.3 The router and the MoE (decode)

The block is four launches on moe.cl's structure (`moe.cl:1-14`), all family kernels in `lfm_moe.cl`:
1. **`lfm_router`**: the 64-row bf16 GEMV into fp32 logits (a family kernel rather than `gemv_bf16` so that P0's fused
   arm `g` can run the same dot order inside the norm kernel and stay bitwise).
2. **`lfm_route`**: §2.3's chain exactly - `l_b = rne(logit)`, `s_b = rne(sigmoid(l_b))`, `sel = s_b + bias`, the rank
   by count over SLM with **exact ties to the lower id** (`k2_route`'s idiom, `k2_moe.cl:122-128`, `:169-175`), `Σ` of the four
   `s_b` in fp32 then `rne`, `+ 1e-6` then `rne`, `w = rne(s_b / d_b)`. The route row (u32 words): ids in **rank order**
   (the combine's order, §2.4), `w`, `sel`, the 5th `sel` (the near-tie margin), `Σ`. 64 experts on 64 lanes.
3. **`lfm_moe_gate_up`**: the 4 selected experts' gate‖up (int4 layout-1, `cols_interleave16(w1, w3)`), expert e at
   `e x 3,342,336 B` inside the layer's block through **one address macro** (spec 15 decision 3; spec 22 could swap it),
   `a = rne(rne(silu(g)) · u)`.
4. **`lfm_moe_down`**: the 4 downs, `t_k = rne(d_k · w_k)`, fp32 sum in rank order, one rounding, then **the residual
   fold** `h = rne(h + y)` (moe.cl's last line, `moe.cl:35-38`). `#pragma OPENCL FP_CONTRACT OFF` where the reference
   rounds a product before an add.

moe.cl itself is not reused because every slot set there carries the shared expert (`SLOTS = TOP_K + 1`,
`moe.cl:73`): a zero shared slot would read a whole expert's bytes per layer for nothing. The limits otherwise fit
(64 a power of two in [16, 256], I 1536 whole k-groups and 32-column pairs, `moe.cl:94-112`). **P0's fused arm `r`:**
the route folded into `lfm_moe_gate_up` (every work-group recomputes the 64-lane rank from the logits; work-group 0 also
writes the route row `lfm_moe_down` reads) - one launch fewer per MoE layer, bitwise the same.

### 4.4 The dense layers, the norms, the head

- **Dense FFN (layers 0-1):** the int4 GEMV at K 2048, N 23552 (`cols_interleave16(w1, w3)`), `lfm_silu_mul` at I 11776
  (prep.cl's `prep_silu_mul` refuses intermediates other than 17408 / 19456, `prep.cl:102-112`; the family copy carries
  the define), the down GEMV at K 11776, N 2048 (S divides 184 k-groups: 1, 2, 4 or 8).
- **Norms:** `lfm_res_norm` - one launch per norm site: fold the preceding GEMV's split-K slices into the residual
  (`h = rne(h + rne(Σ_s slices))`), the row's Σx² by one fixed tree, `x̂ = rne(h · rstd)`, `x = rne(x̂ · w)` (§2.2's two
  roundings). At hidden 2048 a row is 4 KB: every work-group recomputes the row's Σ² over all 2048 values from the
  folded values (identical bits in every work-group: the same tree) and writes its own chunk, with the residual
  **ping-ponged** between two buffers so no work-group reads a chunk another one is rewriting. Where `lfm_moe_down` has
  already folded (after every MoE layer), the variant `_NF` (no fold) only normalises. Qwen's two-launch split
  (`prep_res_fold` + `prep_norm_finish`, docs/15) existed because 5120-wide split-K partials made one work-group slow;
  at 2048 P0 measures whether one launch holds (the two-launch Kolibri form `prep_res_fold` + `kol_norm_finish`-at-1e-5 is
  the fallback, 40 launches more). **P0's fused arm `g`:** the router GEMV inside the `ffn_norm` launch of every MoE
  layer.
- **The head:** int8 per row built at load from `embed_tokens` (spec 9's rule, `loader::quantise_int8_tiled`,
  `src/loader/lm_head_int8.h:44-47`) - tied, so the same bf16 tensor feeds the embedding and the head; `--lm-head bf16`
  reads the bf16 tiles. The embedding is kept once, in `gemv_bf16`'s tiled layout, and `lfm_embed_gather` reads a row
  out of the tiles (4 KB a token): one copy serves the gather and the bf16 head. The argmax at VOCAB 128000, VOCAB_USED
  125017 (`argmax.cl:47-52`'s mask; a family line because the existing binaries carry Qwen's vocabulary).

### 4.5 The decode list and its launch count

```text
conv + MoE layer (28), 9:   res_norm · in_proj GEMV · lfm_conv · out_proj GEMV · res_norm · lfm_router · lfm_route ·
                            lfm_moe_gate_up · lfm_moe_down (folds)
attention + MoE (10), 11:   res_norm _NF · q‖k‖v GEMV (N 3072) · lfm_attn_prep · lfm_attn · lfm_attn_reduce · o GEMV ·
                            res_norm · lfm_router · lfm_route · lfm_moe_gate_up · lfm_moe_down
dense conv layer (0-1), 8:  res_norm · in_proj · lfm_conv · out_proj · res_norm · gate‖up GEMV · lfm_silu_mul · down GEMV
                            (the next layer's operator norm folds down's slices)
head, 5:                    lfm_embed_gather (before layer 0) · res_norm _NF with embedding_norm · head GEMV (int8) ·
                            argmax stage 1 · stage 2
```
**383 launches per token** (derived: 2 x 8 + 28 x 9 + 10 x 11 + 5; the plan test asserts the formula). Eager attention
adds 2 per attention layer (403). The fused arms, each bitwise equal to the list above: `r` (route into gate‖up) -38,
`g` (router into `ffn_norm`) -38, `c` (conv into the in_proj epilogue) -30: **277 with all three** (derived). By §3's
estimate the arms are worth ~0.3 ms (~7 %) of a ~4.5 ms step - which is why **P0 comes first and measures it**: a
skeleton list of 383 launches built from existing binaries at LFM2's shapes (`gemv` / `gemv_bf16` / `gemv_i8w` at the
real K / N, moe.cl at E64 T4 D2048 I1536 as the expert stand-in, Kolibri's attention at Q32KV8 as the attention
stand-in, the norm pair at K 2048, noops for the conv / route / prep slots), the same skeleton at 277, and a GEMV-only
skeleton (the bytes floor), interleaved, at depth 4096 and 32768 - before any LFM2 kernel exists. P0's table decides
the default arms (decision 6) and whether the remaining time is gaps, small kernels or sub-peak GEMVs (the next lever:
per-shape `S` / layout, or merging GEMVs).

**Two cards (`--pp 2`, decision 10):** spec 16b's pieces; the hand-off is `h` (2048 bf16, 4 KB) after layer s-1's
`lfm_moe_down` fold, no norm sums (the receiving `res_norm _NF` recomputes them).

### 4.6 Serving

- **Tokenizer:** `tok::Tokenizer` is HF `tokenizers` 0.22.2 in Rust (`src/tokenizer/rs/Cargo.toml`), so `tokenizer.json`
  loads as is (`ignore_merges`, the Llama-3-style split); 24e holds it to transformers 5.19.0's tokenizer (tokenizers
  0.23.2) id for id on a corpus.
- **Template:** `chat::Template` (minja, b70-patched, `third_party/VERSIONS`) renders `chat_template.jinja`; it already
  parses `{% generation %}` and treats it as a no-op (`third_party/minja/minja.hpp:2533-2536`, `:2662`). 24e's gate: byte-identical renders against Jinja2 /
  `apply_chat_template` over a corpus (system / no system, tools / no tools, tool calls with string / number / bool /
  null / list / dict arguments incl. quotes, backslashes and newlines, tool results, multi-turn with reasoning before
  and after the last user turn, `preserve_thinking`, list-of-parts content, the generation prompt). Known: tool schemas
  and call arguments render in sorted key order (nlohmann's objects, spec 18 §12's known limitation).
- **Tool calls (pythonic):** `<|tool_call_start|>[name(k=v, ...), ...]<|tool_call_end|>`. A **safe literal parser** in
  C++ (never `eval`): a list of calls; each call a name and keyword arguments; values are Python literals - strings in
  single, double or triple quotes, ints, floats (sign, exponent, `.5`), `True` / `False` / `None`, lists, tuples, dicts,
  nested - converted to JSON (`True` → `true`, `None` → `null`, tuples → arrays, `.5` → `0.5`); JSON's `true` / `false` /
  `null` accepted too (decision 7). **Escapes are the template's inverse, per context:** `format_arg_value`
  (`chat_template.jinja:4-12`) writes a top-level string argument single-quoted with `\\`, `\'`, `\n`, `\r` escaped, and a
  list or dict argument through `tojson` (JSON strings, JSON's escapes). So a top-level string (single- or double-quoted)
  decodes exactly `\\`, its own quote's escape, `\n`, `\r`; a string inside a list / dict decodes JSON's set
  (`\" \\ \/ \b \f \n \r \t \uXXXX`); **every other backslash is kept verbatim** (`"C:\Users\test"` stays as written, `\d`
  in a regex too). So `parse(render(x)) == x` and a parsed turn re-renders to the generated bytes. The edge cases (ideas from OpenVINO Model
  Server's LFM2 parser, Apache-2.0, read for ideas only, no code copied: `src/llm/io_processing/lfm2/lfm2_tool_parser.cpp`
  and `src/test/llm/output_parsers/lfm2_output_parser_test.cpp` at `5d8bdfb1`; its commits `503ff866` and `5fe145e5`):
  - **unescaped quotes inside strings** (`'it's a test'`, code containing `"__main__"` inside a `"..."` argument -
    `lfm2_output_parser_test.cpp:808-840`, `:874-889`): a quote closes a string only when followed, after whitespace, by
    `,` `)` `]` `}` `:` or the block end; if the rest then fails to parse, the parser backtracks over that quote once
    (bounded by the block length) and keeps it in the string. Model Server's heuristic single-to-double-quote rewrite
    (`lfm2_tool_parser.cpp:52-115`) is not reproduced: a real literal parser has no `', '`-inside-a-string failure mode;
  - **delimiters inside strings** (`,` `)` `]` `{` `=` `>=`, newlines, f-string braces; `:741-774`) are string content;
  - **quoted names** (`"path"="x"`, `'read_file'(...)` - the failure `5fe145e5` fixed for another format, keys arriving
    as `"\"command\""`): quotes are stripped from argument and function names; a key never contains a quote;
  - **separators:** `,` with any whitespace or newlines between calls and arguments (Model Server requires exactly `", "`,
    `lfm2_tool_parser.cpp:39`); a trailing comma accepted; `fn()` gives `{}`; `fn(name="")` an empty string;
  - **a bare call without the list brackets** (`<|tool_call_start|>fn(a=1)<|tool_call_end|>`) is accepted as one call
    (decision 7; Model Server rejects it, `:731-739`);
  - **positional arguments:** a single positional dict is taken as the arguments object; any other positional
    argument, a name without `=`, or `=` without a value makes the call invalid (Model Server keeps such calls with an
    empty value, `:709-729`; decision 7);
  - **generation ends inside a block** (`max_tokens`, or EOS before `]` / `<|tool_call_end|>`): every call whose `)`
    closed is emitted, exactly once (the regression `503ff866`'s test pins, `lfm2_output_parser_test.cpp:515-529`); an
    unfinished call is not emitted and its text is not leaked into content (Model Server's convention "if we can emit
    correct output, do it"); `finish_reason` stays `length` / `stop`;
  - **a stray `<|tool_call_end|>`** without an open block is dropped from content (`lfm2_tool_parser.cpp:183-187`);
  - **`<|im_end|>`** (EOS 124900) ends generation and never reaches the text (Model Server strips it from content,
    `:287-291`; our server stops on the EOS id);
  - **calls inside the reasoning** are reasoning text, not calls; a block with no valid call is content, verbatim with
    its tags; depth > 64 or a block > 1 MiB is content.
  Calls are emitted when `)` closes them (streaming), the rest of the block's text is held back until the block closes;
  any split of the text gives the same deltas (spec 7a's rule). `generation_config.json`'s `eos_token_id` is read as an
  int or a list, and `stop_strings`, where a model has them, as a string or a list (the HF definition; OpenVINO GenAI's
  `d54d3768` fixed the string form being dropped).
- **Reasoning:** the prompt ends in `<think>`, so the output opens inside reasoning, up to `</think>`, kept verbatim (no
  newline stripping: the template writes `"<think>" + thinking + "</think>"` back, `chat_template.jinja:90-92`, so a
  client that returns the parsed turn re-renders the generated bytes - the prefix cache's best case). Generation that ends inside reasoning is
  reasoning. `enable_thinking: false` cannot be honoured by this template (decision 8). `chat_template_kwargs`:
  `preserve_thinking`.
- **Known prefix-cache miss:** the template renders an assistant turn as content, then the calls; the model may emit a
  call before its text (the README's example does), so such a turn re-renders in another order and the cache diverges
  there (no correctness effect; 24e measures the hit rate on the agentic transcript).
- **Sampling defaults** (decision 2) and **repetition penalty** (decision 1): the host sampler gains
  `repetition_penalty` (HF's rule, §2.8, over the session's distinct ids - prompt and output); `generation_sampling`
  reads it, and an absent `top_p` means 1.0 (HF's `GenerationConfig` default), not `Sampling{}`'s 0.95.
- **Prefix cache (spec 7):** a snapshot is the KV blocks (41.9 MB per 2048 positions) plus the conv state: per conv layer
  the ring slots of positions p-1 and p-2 (245,760 B), written back at the restored position's slots; EngineIface's
  `state_bytes` / `save_state` / `load_state` / `save_kv` / `load_kv` / block hook (`src/server/deps.h:62-77`).
  `--prefix-split-last` applies as for Qwen (the generation prompt's last id `<think>` is where the next turn diverges
  when the turn is before the new user message).

*As built (24e): §16 - the list above is the design; the readings that differ (quote recovery by lookahead, when a call is emitted, the 1 MiB rule, sorted argument keys) are listed there.*

### 4.7 What is reused, adapted or new

| piece | state |
|---|---|
| int4 GEMV (`gemv.cl`), bf16 GEMV, int8 head GEMV (`gemv_i8w.cl`) | reused at new shapes (CMake lines only) |
| router (`lfm_router`), route (`lfm_route`), MoE gate‖up / down (`lfm_moe_*`) | new family kernels on moe.cl's / k2_moe.cl's structure (`tile_dot` verbatim) |
| short conv (`lfm_conv`, `lfm_pf_conv*`) | new, on gdn_step's / pf_gdn_conv's ring discipline |
| norms (`lfm_res_norm`), `lfm_silu_mul`, `lfm_attn_prep` | new family copies (eps 1e-5, plain w, I 11776, RoPE everywhere) |
| decode / prefill attention at HD 64 (`lfm_attn*`, `lfm_pf_attn`) | new family copies of Kolibri's full path |
| embed gather from the tiled bf16 table, argmax at 128000 / 125017 | new lines / one small kernel |
| prefill MoE (`lfm_pf_moe`: sort / gather / dequant / combine) + `pf_moe_gemm` | new family file (no shared block, uchar ids, rank-order combine) + reused GEMM at K 2048 N 3072 SILU and K 1536 N 2048 |
| prefill linears (int4 dequant slabs + `pf_gemm`) | reused, spec 20d's `l0` path |
| `ModelDesc`, loader, sizes, engine, CLI | new family files on Kolibri's pattern (§6) |
| pipeline parallel | reused pieces of spec 16b (`PipelineLink`, `pp_landing_layout`) |
| tokenizer, minja, server, prefix cache | reused; new `ChatFormat::Kind::Lfm2`, `server/toolcall_lfm2.*`, the penalty in the sampler |

## 5. Formats

- **Ours (the one format):** AutoRound 0.17.0 @ `6afaecdb` (`tools/quantize/ar_pin.sh`, + `auto-round-local.patch`) W4A16,
  `group_size` 64, symmetric, `auto_round:auto_gptq` (`loader::QuantConfig::parse`'s rules, docs/13). **Quantised:**
  every `nn.Linear` - routed experts, dense `w1` / `w2` / `w3`, conv `in_proj` / `out_proj`, attention `q/k/v/out_proj`
  (decision 3). **Kept as shipped:** the router `feed_forward.gate.weight` (a Parameter used through `F.linear`, not an
  `nn.Linear`: AutoRound's `SUPPORTED_LAYER_TYPES` is `(nn.Linear, Conv1D)`, `auto_round/utils/common.py:749` at
  `6afaecdb`), `expert_bias` (fp32 buffer), `conv.conv.weight` (`nn.Conv1d`: not supported, stays bf16), every norm,
  `embed_tokens` (no `--quant_lm_head`; the head is tied). The check refuses any other quantised tensor.
- **The expert names in the export:** transformers 5.19.0 fuses the checkpoint's per-expert `experts.N.{w1,w3,w2}` into
  `gate_up_proj` / `down_proj` at load (`conversion_mapping.py:1160-1174`), and AutoRound 0.17.0's generic unfusing names
  the per-expert linears `gate_proj` / `up_proj` / `down_proj` (`moe_experts_interface.py:94-121`; `lfm2_moe` is not in
  `unfused_moe/__init__.py`'s `MODEL_CONFIG`, so no model-specific path applies). 24q Task 1 proves which names the
  export carries on the tiny model; the loader (24b) reads either spelling, all-or-nothing per checkpoint (one format,
  two name sets).
- **RTN g64 (the fallback and the development checkpoint):** `tools/quantize/lfm2moe/rtn_export.py` writes the same
  export from the bf16 original with our RTN rule (GPTQ's symmetric quantiser per output column and 64-group:
  `s16 = f16(2 amax / 15)`, `q = clamp(rint(w / s16) + 8, 0, 15)`, `qzeros` 0x77777777 - `src/loader/rtn.h:20-27`), in minutes on the box CPU. It is what 24b-24d develop against before the
  AutoRound run finishes, and the shipping fallback if AutoRound cannot run the model (§0).
- **KV:** bf16. int8 KV (spec 12) is decision 11.
- **The bf16 original** is the reference's input only (24a), never the engine's.

## 6. Structure

On Kolibri's pattern (spec 20 §11):
- **Descriptor and loader:** `model::Lfm2MoeDesc` (`src/model/lfm2moe.*`, held to config.json key by key),
  `loader::load_lfm2moe` (`src/loader/lfm2moe_{layout.h,repack.*,loader.*,rope.cc}`).
- **Engine:** `runtime::lfm2moe::Lfm2MoeEngine` (`src/runtime/lfm2moe/`): decode list, prefill walk, snapshots, one card
  (two through spec 16b's placement).
- **Kernels:** `src/kernels/lfm2moe/lfm_*.cl`, host half `src/kernels/lfm2moe_kernels.h` (namespace
  `kernels::lfm2moe`); every shape define in the binary name; reused kernels from new CMake lines only.
- **CLI:** `b70-decode` / `b70-serve` dispatch on `model_type lfm2_moe` (`src/cli/lfm2moe_{decode,serve}.h`).
- **Server:** `ChatFormat::Kind::Lfm2`, `src/server/toolcall_lfm2.{h,cc}`, the penalty in `server::Sampling` /
  `filter_probs`.
- **Tools:** `tools/oracle/lfm2moe_ref.py` (+ facts, fixture, make_tiny, prompts), `tools/quantize_lfm2moe.sh` +
  `tools/quantize/lfm2moe/`, `tools/box_validate/lfm2moe_oracle.sh`, `tools/probe/probe_lfm_step.cc` (P0).

## 7. Gates

- **F0, nothing moves.** Every existing kernel binary sha256-identical (G0), `tools/kernel_cmdlines` additions only, no
  existing `.cl` edited, every existing suite green.
- **F1, the reference** (24a). `lfm2moe_ref.py` equals transformers 5.19.0 un-streamed on the tiny model **bitwise**
  (bf16 eager and fp32; a prompt and cached decode steps; logits and every layer's residual), with one test per trap of
  §2: the router chain (bf16 sigmoid, the bias only in selection, `+ 1e-6`, the bf16 division), planted router ties,
  the combine's order, the conv's tap order and its zero history, a prompt fed as chunks equal to one forward (the conv
  state across a chunk boundary), the two-rounding norm, `embedding_norm` at the end, the tie. The real model's
  perplexity is sane on the box CPU.
- **F2, kernels.** Every new kernel against its host twin (`tests/kernels/lfm2moe_ref.h`), bitwise where the arithmetic
  is the same: `lfm_conv` (and the fused arm), `lfm_res_norm`, `lfm_router` / `lfm_route` (ties at the 4th / 5th,
  biased selection), the MoE (each slot alone), `lfm_attn_prep`, `lfm_silu_mul`, the eager attention, the gather, the
  argmax mask; flash against fp64 at cosine ≥ 0.99999 (spec 6 K1).
- **F3, golden** (greedy, tie-aware, docs/14's undetermined-row rule) against the **engine-format** reference (our
  export dequantised): on a synthetic real-width checkpoint, on the RTN export, then on the AutoRound export. Routing:
  top-4 sets equal except near-ties at the 4th / 5th `sel` (`B70_LFM_TIE_TOL`, set from 24a's measured gap
  distribution, 1e-3 PROPOSED until then; an exact 0 gap undetermined). Prefill against decode's fill is held to cosine
  bars, not bitwise (the GEMM's K order differs from the GEMV's; plan 24d). The fused arms are bitwise the as-built list.
- **F4, determinism.** Replay bitwise; prefill split at multiples of 64 bitwise; chunked = whole; prefix-cache restores at
  block ends bitwise the cold run (KV and conv rings); `--pp 2` bitwise `--pp 1`.
- **F5 is unused** (no MTP head).
- **F6, quality** (24q). Ours (AutoRound g64) and RTN g64 against the bf16 original on held-out rows: KL and top-1 per
  set (agentic transcript, code, prose), and **A4**: the 36 tool-call scenarios through LFM2's template, first call's
  name and arguments against the bf16 reference, parse failures counted; the int8 head against the bf16 head (KL).
  Bars PROPOSED (decision 5): mean KL ≤ 0.05 and top-1 ≥ 0.93 on every set, A4 agreement ≥ 33 / 36 with 0 parse failures.
- **Speed (recorded; no bars before P0):** launches per token (383 / 277), decode at depth 1024 / 4096 / 32768 / 131072,
  int8 and bf16 heads, flash and eager; prefill rate per chunk; the P0 table.

## 8. Stages

- **24a. Facts and the CPU reference** (Mac, then the box CPU). `lfm2moe_ref.py` (transformers' model layer-streamed with
  `stream.py`; experts dequantised from our int4 g64 or read bf16), the facts sheet `docs/probe-lfm2moe-<date>.md`,
  tiny-model variants (`lfm2moe_make_tiny.py`: the real layer pattern, head_dim 64, non-zero biases, planted ties, int4
  packing), the golden prompts (re-tokenised: short, 4k, 32k, agentic) and the A4 set rendered by LFM2's template. Mac:
  F1. Box CPU (row 48): perplexity, `hfcheck`, the bf16 golden sets, the A4 reference on the bf16 original, the gap
  distributions.
- **24q. Our quantisation** (the box; row 49). `tools/quantize_lfm2moe.sh` (stages `calib`, `coverage`, `rtn`, `tune`,
  `eval`, `check`, `card`), AutoRound on the box's XPU (decision 4), F6, publish after the operator's go.
- **24b. Descriptor, loader, formats** (Mac; box for loads; row 50). `Lfm2MoeDesc`, `load_lfm2moe` (both expert name
  spellings, the tied int8 head, the tiled embedding), `make_synth.py` (real widths, N layers), the planner.
- **24c. Decode** (built blind; row 51). P0 (the skeleton lists) first, then the kernels, the engine on one card, the
  fused arms, `--pp 2`.
- **24d. Prefill** (built blind; row 52). The conv over a chunk with the ring carried, the GEMMs, the grouped MoE, flash
  at HD 64; `l0` only.
- **24e. Serving** (built blind; row 53). `b70-serve`, template, tokenizer, the pythonic parser, reasoning, the sampler's
  penalty and defaults, prefix-cache snapshots, A4 on the card, the agentic transcript.

Dependencies: 24a → 24q (its `rtn` stage first, minutes) and 24b in parallel; 24b → 24c → 24d → 24e. 24q's AutoRound
export replaces RTN in every real-weight gate when it exists (each row has an `ours` stage).

## 9. What runs blind and what needs the box

| blind (the Mac) | the box |
|---|---|
| the reference on the tiny model and its variants, the facts sheet's source readings, the A4 set's rendering | every kernel's first compile (ocloc) and run; P0's skeleton lists |
| descriptor, loader, planner, synthetic checkpoints at small N, host twins | the bf16 golden sets and the A4 reference on the box CPU (47.96 GB bf16, layer-streamed) |
| Level Zero / OpenCL syntax, kernel command lines, indicative Mac-GPU runs of the plain-OpenCL kernels | AutoRound on the XPU, F6 |
| template byte-identity, tokenizer equality, the parser, snapshot layouts | loads, decode / prefill / serve gates, speed rows |

## 10. Decisions

**Ruled (operator, 2026-10-15): all decisions below as recommended.** Decision 9: the operator confirms the LFM Open License v1.0 threshold is met (well under USD 10 M annual revenue), so our quant is published under LFM 1.0 with its notices.

1. **(decide) Repetition penalty.** The engine has none. Proposed: implement HF's rule (§2.8) in the host sampler for
   **sampled** requests - over the distinct ids of the session's prompt and output (a 128000-bit set per request, applied
   to the host logits row before `filter_probs`), from generation_config's 1.05 as the default and a request's
   `repetition_penalty` (vLLM's extension field); greedy requests (temperature 0, the gates) stay pure device argmax and
   ignore it, documented as a deviation from vLLM (which applies it to greedy too). Alternative: also route greedy
   requests with a penalty ≠ 1 through the host (a 512 KB logits read-back per token, ~40 µs pinned).
2. **(decide) Sampling defaults.** Proposed: generation_config's (sampled, T 0.1, top-k 50, top-p 1.0, penalty 1.05) as
   `b70-serve`'s defaults for this model (Kolibri's precedent, spec 20e), with `generation_sampling` reading an absent
   `top_p` as 1.0. Alternative: the Qwen family's greedy-unless-asked.
3. **(decide) The quantised set.** Proposed: every `nn.Linear` int4 (one format, 1.43 GB a token). Alternative arms if F6
   fails: conv `in_proj` / `out_proj` bf16 (+0.74 GB a token, derived) and / or attention bf16 (+0.15 GB).
4. **(decide) Where 24q runs.** Proposed: the box - AutoRound `--device` the second B70 (XPU, as `quantize_qwen38_tuned.sh`
   ran; the bf16 model in host RAM, 48 GB of the box's ~121 GB), calibration answers self-generated by the bf16 model
   through vLLM's XPU image if it serves `lfm2_moe` (TP 2), else transformers on the box CPU batched (hours, estimated);
   `NSAMPLES` 512 x 2048 (PROPOSED), `ITERS` 200. Alternative: a rented CUDA machine (Kolibri's plan).
5. **(decide) F6's bars** (PROPOSED above: KL ≤ 0.05, top-1 ≥ 0.93 per set, A4 ≥ 33 / 36 with 0 parse failures), set
   after 24q's RTN row shows the spread.
6. **(decide) The default decode list.** Proposed: every fused arm (`r`, `g`, `c`) that P0 and the engine's interleaved
   A/B show faster and that is bitwise equal (all three are by construction), i.e. 277 launches; `B70_LFM_FUSE` keeps
   the others selectable.
7. **(decide) Parser leniency** (§4.6's list). Proposed: JSON `true` / `false` / `null` beside Python's literals; the
   quote-recovery rule with bounded backtracking; quoted names stripped; a bare call without brackets accepted; a lone
   positional dict taken as the arguments, any other positional argument (or `k=` with no value) an invalid call;
   completed calls emitted when generation ends inside a block, the unfinished one dropped (not leaked as content); only
   the template's escapes decoded. 24a's A4 reference outputs (the bf16 model's own calls) are the evidence: 24e prints
   how often each rule fired.
8. **(decide) `enable_thinking: false`.** The template always opens `<think>`. Proposed: ignore it and document (the
   card: "always thinks"); alternative: append `</think>` to the generation prompt when it is false (outside the
   template, so not what the model was trained on).
9. **(decide) The licence.** Publishing our export under LFM Open License v1.0 with its notices (§1.4), and whether the
   operator's use sits under §5's USD 10 M threshold; and whether `tests/model/lfm2moe/config.json` is copied (with a
   source line) or built in code.
10. **(decide) `--pp 2`.** The model and its full context fit one card. Proposed: build it last in 24c on 16b's pieces
    (the family pattern; cheap) but keep `--pp 1` the default; or skip it until batching needs the second card.
11. **(decide, later) int8 KV at head_dim 64.** At 131072 positions the KV read (2.68 GB) exceeds the weights. Proposed:
    a follow-up spec after 24c's depth rows (spec 12's rotkv; `kv8.cl` hard-codes 24 q / 4 kv heads).
12. **(decide) Family copies vs parameterised sources** for the Kolibri-derived kernels (prep, attention): proposed
    copies (`lfm_*`), so no existing source changes and F0 is the sha256 check alone.
13. **Reference attention:** eager (ruled by precedent, spec 18 §10.1); the engine's flash default is held to fp64.

## 11. Out of scope

- int8 prefill (spec 5's h8 path for this family): a later probe once 24d's bf16 `l0` rate is known.
- The LFM2 / LFM2.5 siblings (1.2B, 8B-A1B, 350M, VL, audio) - the descriptor refuses another `model_type` or another
  width by key.
- Third-party formats (FP8, GGUF, MLX, ONNX).
- Speculative decoding: the model has no MTP head; LiquidAI's DSpark drafter is "coming soon" for SGLang. Prompt lookup
  (spec 19e) needs M-row verify lists for this family - a follow-up after 24e.
- Batching (spec 13) - the spare ~13 GB a card is where it would live.
- Tensor parallel.

## 12. 24a as built (2026-10-15; the Mac, the tiny models AND the real bf16 weights)

Plan: `docs/superpowers/plans/2026-10-15-spec24a-lfm2moe-facts-and-reference.md`, branch
`spec24a-lfm2moe-reference`. Facts: `docs/probe-lfm2moe-2026-10-15.md`. Box validation: queue row 48 (box CPU
only). Every number is measured on the Mac (`agnes-ref-img`, torch 2.14.1+cpu, the 5.19.0 site; real weights in a
40 GB container, 12 threads, under the one-model-job lock) unless marked.

- **What 5.19.0 runs:** the `grouped_mm` experts implementation (§2.4's combine: `t_k` rounded, the k products
  summed in fp32 in slot order, one rounding - the engine copies this one; the eager fallback differs on every
  row tested); the torch conv functions (`causal_conv1d` absent); `expert_bias` stays fp32 under bf16; the head is
  the embedding's storage. sdpa differs from eager by 2.9e-3 in the logits (tiny, bf16) - never used.
- **The conv's accumulation (§2.5's open point):** fp32 taps ascending with one rounding is torch's bf16 depthwise
  `conv1d` **exactly** on 100,000 rows x 2048 channels (0 rows differ; descending 72, a bf16 accumulator 99,900):
  §4.1's `lfm_conv` can be held bitwise.
- **torch's tie order:** at the 4th / 5th of 64 the lower id wins 26 / 50 (batched 25 / 50) - not the lower id;
  exact ties are undetermined (`route.gap == 0`). On the real prompts **no** exact tie occurred (0 of 637,108
  (row, layer)).
- **F1 on the tiny models (all bitwise):** the streamed reference = transformers un-streamed on `pattern8` (the
  real first eight layers at head_dim 64, biased routers) and the downloaded tiny, bf16 and fp32, 40 / 600 ids +
  8 cached steps; chunks 1 / 2 / 3 / 64 / 255 and splits at 1..5 = one forward (conv states too); the port (the
  restated ops chained) = transformers at chunks 0 / 1 / 2 / 3 / 5 / 64; one test per §2 trap (`test_lfm2moe_ref.py`,
  14 tests; `test_lfm2moe_make_tiny.py` 4; the toolcall tests - `test_score.py` 35 and `test_make_set.py` 9 after
  the merge with spec 23a - and the batched reference).
- **F1 on the real weights:** `hfcheck --layers 6` (2048 ids of `lfm2moe_4k` + 4 cached steps, transformers'
  model against the port on the bf16 original): **BITWISE** over 40 tensors (logits, every h / mixer / ffn row,
  the routes, the conv states), 390 s. Golden sets on the bf16 original: `lfm2moe_short` (100 ids, 881 s: prompt
  658 s, 32 steps 204 s), `lfm2moe_4k` (4096 ids whole, 2923 s: prompt 2496 s, steps 395 s), `lfm2moe_agentic`
  (12474 ids in chunks of 2048, 12,564 s: prompt 12,277 s, steps 265 s; py-spy: the eager attention's bf16 P V
  matmul dominates the long chunks, ~1 core busy). `peak RSS` prints 39.8-39.9 GiB everywhere: `ru_maxrss`
  counts the mmap'd shard pages under the 40 GB cap. `lfm2moe_32k` and the A4 reference were NOT run on the Mac (32k
  ESTIMATED 6-8 h from these rates, A4 days - above the operator's 6 h rule): the box's (row 48).
- **The perplexities (a departure from the plan's "below 20 on prose"):** prose 53.2 (45 tokens), code 2.54 (54),
  the agentic transcript's first 4096 ids 290. Investigated, not an input bug: BOS once and the ids
  `apply_chat_template`'s; the top-5 predictions are coherent (after "The harbour at dawn was": " still",
  " mist", " quiet"); the short set's greedy continuation re-writes the `clamp` function; `lfm2moe_4k` (long.txt
  repeated) has ppl **1.19** (copying across 4096 positions works); the agentic set's greedy continuation is "The
  test passes. The user asked to "run the planner's test" — done. The comment edit is complete. Let me verify the
  final state of the". Split by the role of the predicted token (agentic golden, 12,473 rows): **assistant turns
  6.9**, tool results 88, user 217, system (incl. the tool list JSON) 4,190, the turn markup (`<|im_start|>` /
  role words) ~4e5 - prose 53 is dominated by the first tokens (" harbour" at position 1: 16.5 nats). The model
  scores prompt-side text badly (post-trained on assistant turns); the stage's bound is now code < 10, prose and
  agentic recorded. 4k's 4,096 rows still show a few copied tokens at 11-16 nats (e.g. " ferry" in the 20th
  repetition) - recorded; whether vLLM's / LiquidAI's own forward agrees on such rows is for 24e / 24q to compare
  (they run vLLM where it serves `lfm2_moe`).
- **The routing gaps (B70_LFM_TIE_TOL's evidence):** the 4th / 5th `sel` gap over every (row, layer):

  | set | (row, layer) | == 0 | < 1e-4 | < 1e-3 | < 1e-2 | min / p0.01% / p0.1% / p1% / median |
  |---|---:|---:|---:|---:|---:|---|
  | lfm2moe_short | 5,016 | 0 | 28 | 267 | 2,056 | 2.98e-7 / 5.05e-6 / 1.39e-5 / 1.78e-4 / 0.0133 |
  | lfm2moe_4k | 156,864 | 0 | 985 | 9,605 | 70,373 | 4.47e-8 / 1.63e-6 / 1.53e-5 / 1.60e-4 / 0.0118 |
  | lfm2moe_agentic | 475,228 | 0 | 3,172 | 31,044 | 216,167 | 7.45e-8 / 1.22e-6 / 1.48e-5 / 1.51e-4 / 0.0116 |

  Per layer (agentic): < 1e-3 on 4.3 % (L21) .. 8.7 % (L36) of rows, median 0.0084 (L15) .. 0.0185 (L21). The gap is
  in `sel` units: `s_b` is bf16 (one ulp 2^-9 .. 2^-8 for s in [0.25, 1)), so gaps below ~1e-3 come from the fp32
  bias, and an engine whose router logit differs from torch's by one bf16 ulp moves `s_b` by up to ~4e-3 (|l| < 4,
  derived). RULED (operator, 2026-10-16) for 24c / 24d: `B70_LFM_TIE_TOL` = 4e-3 (one bf16 ulp of the logit carried through the
  sigmoid; ~15-20 % of (row, layer) fall under it, ESTIMATED between the 1e-3 and 1e-2 columns) unless 24c's route
  kernel reproduces torch's `l_b` bitwise, in which case the 1e-3 PROPOSAL stands (6.5 % of (row, layer)).
- **Departures from the plan:** the 5.19.0 site from host-downloaded wheels (pip in the container had no network);
  fp32 cached decode vs one uncached forward is NOT bitwise in transformers itself (1.8e-7, `conv.in_proj` at M 1 vs
  M 32) - recorded, F1 compares on the same path; the int4 tiny loads through its dequantised twin (no auto-round in
  the image); the A4 set renders key-sorted (b70-serve's order) and `make_set --from` takes a 600-id floor for
  LFM2 (748..2460 ids); `test_oracle_batched test_lfm2moe` runs on the downloaded tiny (pattern8's greedy repeats);
  4k / 32k are BOS + long.txt joined by blank lines (not mk_long_ids's whole-tree text); `score.py --format
  pythonic` implements §4.6's quote recovery (one backtrack per quote) and counts the calls that closed in a block
  cut by the end of generation; `run_in_container.sh` gained `ORACLE_MEMORY` (the Mac's container cap) and
  `lfm2moe_oracle.sh` runs on the Mac under the CPU lock.
- **Pending (box CPU, row 48):** the five test files in the box's image (another torch build), `lfm2moe_32k`, the
  A4 reference (2048 new ids, reasoning first; `score.py --format pythonic`, 0 parse failures the bar).

## 13. 24b as built (2026-10-17; the Mac only - nothing has run on a card)

Plan: `docs/superpowers/plans/2026-10-15-spec24b-lfm2moe-descriptor-loader.md`, branch
`spec24b-lfm2moe-loader`. Box validation: queue row 50. Every byte figure below is derived from
`model::lfm2moe()`'s widths (`src/loader/lfm2moe_layout.h`) and cross-checked against §3's table above,
word for word, by `lfm2moe_repack_test.cc` and `lfm2moe_plan_test.cc` (review fix, M5: this used to claim
"no test types a constant" - false. `lfm2moe_repack_test.cc` does type a handful of literal byte constants,
e.g. `embed_tiled_bytes` == 524,288,000 and `head_int8_bytes` == 262,656,000, each reconciled by hand against
spec 24 §3's table; what no test does is hardcode the grand `weight_bytes` TOTAL - that figure is always the
sum of `layer_weight_bytes`'s own per-layer outputs, `CHECK_EQ`'d against an independently-accumulated `want`).

- **The ground truth for names came from 24a, not from this plan's guesses.** `tools/oracle/lfm2moe_ref.py`'s
  `expected_names` (already built, bitwise tested against transformers 5.19.0) fixed several spellings this
  plan's prose did not pin precisely: attention's out-projection is `self_attn.out_proj` (not `o_proj`), the
  q/k per-head norms are `self_attn.q_layernorm` / `k_layernorm`, the conv block's tensors are `conv.conv.weight`
  / `conv.in_proj` / `conv.out_proj`, the final norm is `model.embedding_norm.weight` (not `model.norm`), and no
  checkpoint form (bf16 or our int4) ever ships `lm_head.weight` as a REQUIRED tensor (`expected_names` never
  lists it; an optional one is the loader's own bitwise-tie check). `src/model/lfm2moe.cc` and
  `src/loader/lfm2moe_repack.cc` use these names directly.
- **Per-linear switchable layout, not "layout 1 everywhere" (review fix, I1).** The first pass of this
  section claimed every per-layer int4 linear uses the shared subgroup-tiled layout 1 - false against the
  codebase's own evidence: Muse, K2, Kolibri and qwen4exp all use layout 0 for their DENSE linears (the
  GPTQ-native `.qweight`/`.scales` pair) and layout 1 only for routed-expert blocks. `model::Lfm2MoeDesc::linears()`
  is now the one source of truth for each `LfmLinear`'s `{S, layout}` (`src/model/lfm2moe.cc`), and
  `repack_layer` / the upload loop branch on it instead of hardcoding `{..,1,1}` (`src/loader/lfm2moe_repack.{h,cc}`,
  `src/loader/lfm2moe_loader.{h,cc}`). Provisional picks, swept in 24c (plan 24c §note): layout 1 for `in_proj`,
  `out_proj` / `o_proj` / the dense down-projection (every N = hidden row-count linear) and the routed experts
  (unchanged - this part of the original claim was right); layout 0 for the fused `q‖k‖v` and the fused dense
  `gate‖up`. `S` stays 1 everywhere (PROVISIONAL, 24c's sweep). Both layouts are byte-count-identical here (every
  linear has N % 16 == 0 and K % 64 == 0, so `int4_layout0_bytes` == `int4_l1_bytes`): this correction changes
  no byte figure in this section, only which buffer shape `repack_layer` builds and which GEMV kernel (24c) must
  read it.
- **No router padding.** Kolibri's `router_n()` rounds its 384 experts up to a power of two for `gemv_bf16`'s
  16-column tiles. LFM2.5's 64 experts is already a multiple of 16, so `loader::lfm::router_bytes` and the
  repack are one `repack_bf16_tiled` call with no padded rows and no `router_n()` function at all.
- **`vocab_used` is not read from a tokenizer at load** - a judgment call, following Kolibri's own precedent
  (its `vocab_used` is a fixed constant too, never derived from `tokenizer.json`). `load_lfm2moe` always passes
  `model::lfm2moe().vocab_used` (125017, the published fact) to `lfm2moe_desc`; a checkpoint whose tokenizer
  genuinely differs (a future sibling model, out of scope by `check_lfm2moe_config`) is not handled. Cost if
  wrong: the argmax mask stage (24c) would mask the wrong rows on such a checkpoint - never the published one,
  which this descriptor is pinned to.
- **24q does not exist yet**, despite this plan's "Dependencies" section saying "Plan 24q Tasks 1-2 merged" -
  `git log --all` shows only 24q's PLANNING commit, no `rtn_export.py` or `check.py`. Judgment call:
  `tools/quantize/lfm2moe/make_synth.py` and `check.py` are self-contained (the same shape as Kolibri's and
  Muse's own `make_synth.py` / `check.py`, which also carry their own packer - not a new pattern), and the RTN
  packer itself is reused from `tools/oracle/lfm2moe_make_tiny.py` (already proven by
  `test_lfm2moe_make_tiny.py`'s `int4-g64` variant) rather than a third copy of `src/loader/rtn.h`'s formula.
  When 24q lands its own packer, `make_synth.py` can be pointed at it; nothing else here depends on the choice.
  Cost if wrong: none to correctness (the formula is `src/loader/rtn.h`'s, read correctly); some duplicated
  Python across `tools/quantize/{kolibri,muse,lfm2moe}/make_synth.py` that a shared packer module could remove
  later.
- **RTN's `[-8, 7]` asymmetry is real, not a bug.** `--plant-sign`'s `w3 = -w1` test
  (`tools/quantize/lfm2moe/test_lfm2moe_quant.py`) found that symmetric int4 at zero point 8 represents
  `[-8, 7]`: whichever tensor's element IS a 64-group's own `amax`, a quotient that the f16-rounded scale
  pushes fractionally past 7.5 clamps on the positive side while its exact negation does not, at the negative
  side's extra slot. Measured: 0.858% of a 64-expert MoE layer's elements (one group boundary element roughly
  half the time). This is RTN's property, not the engine's: the C++ repack (`repack_expert`,
  `repack_int4_layout1_cols`) moves already-packed int4 words - no re-quantisation - so it never hits this; the
  router / combine kernels (24c) read the packed bytes as shipped either way.
- **The bf16 head's replica, two cards.** Spec §4.4 says "the embedding is kept once... one copy serves the
  gather and the bf16 head" - true on one card (device 0 is also the last device). Over two cards with
  `--pp 2` and a bf16 head, the LAST device has no copy of device 0's embedding tiles: `load_lfm2moe` uploads a
  second bf16-tiled copy there (524,288,000 B), counted in `loader::lfm::device_weight_bytes`'s `else if (dev
  != 0)` branch and `runtime::lfm2moe::plan`'s matching term. Decision 10 defers whether 24c's engine ever
  serves `--pp 2` with a bf16 head; the bytes are planned regardless. The int8 head needs no replica (it is
  built fresh on whichever device is last, from the host-mmapped bf16 bytes, not from device 0's tiles).
- **Every derived figure matches spec 24 §3 exactly:** routed experts 12,192,841,728 B (38 x 320,864,256), the
  other int4 linears 399,932,928 B (conv 267,386,880 + attention 55,705,600 + dense 76,840,448 - each
  independently reconciled against §3's "x M params at 0.53125 B / param" row), routers 9,961,472 B, the
  embedding 524,288,000 B, the int8 head 262,656,000 B, the 40 layers' small blocks (norms / conv taps /
  biases) 1,448,960 B: `weight_bytes(lfm2moe(), true)` = 13,391,129,088 B (13.391 GB; `lfm2moe_repack_test`
  prints this exact figure - review fix: an earlier manual sum here read 13,391,087,616 B, 41,472 B short,
  from mis-adding the small-block term; the formula itself never changed). KV 20,480 B a position
  (2,684,354,560 B at 131072, 2.684 GB). The conv rings 1,966,080 B. The one
  figure that does not land where §3's prose says: "norms / conv taps / biases < 1 MB" measures at ~1.41 MB
  (40 layers' small blocks summed) - a rounding in the spec's ESTIMATE, not a formula mismatch (the per-layer
  byte counts above are exact and tested).
- **The Mac gate (2026-10-17):** `tools/mac_check.sh --base main` section 1 (host) 120/120 PASS, including
  `lfm2moe_test`, `lfm2moe_repack_test`, `lfm2moe_rope_test` (bitwise transformers 5.19.0's own
  `Lfm2MoeRotaryEmbedding` at 5 fixed + 1000 random positions), `lfm2moe_plan_test` and
  `lfm2moe_synth_host_{w,proj}_test` (over two synthetic real-width checkpoints, `make_synth.py --layers 4`,
  generated on the Mac itself, both expert spellings, 0 unconsumed each); section 3 (kernel_cmdlines) +0 / -0 /
  ~0 (F0 holds: no kernel exists for this family yet). `lfm2moe_load_checkpoint_test` (the card test) is
  excluded from the Mac's ctest run by `cmake/host_only.cmake` (it links Level Zero) - row 50's `r50.load_synth`
  / `r50.load_real` are the box's.
- **What the box still needs:** `r50.synth` (the Mac already generated and `check.py`-ACCEPTED both synthetic
  spellings; the box's own run is the same command, for its own record); `r50.load_synth` (the first time
  `load_lfm2moe` ever touches a card); `r50.load_real`, which needs 24q's export (`have lfm2moe_rtn`, currently
  always false) - genuinely blocked until 24q runs.

## 14. 24c as built (2026-10-20; the Mac only - nothing has run on a card, no lfm_* binary has been compiled by ocloc)

Plan: `docs/superpowers/plans/2026-10-15-spec24c-lfm2moe-decode.md`, branch `spec24c-lfm2moe-decode`. Box validation:
queue row 51 (`tools/box_validate/stages.sh` `row 51`). Every count below is derived and asserted by a host test.

- **The list (§4.5), as built** (`runtime/lfm2moe/lfm2moe_capture.{h,cc}`; `runtime::lfm2moe::device_launches`):
  dense conv 8, conv + MoE 9, attention + MoE 11 (eager 13), the head 5 (the gather first on device 0; the final
  norm, the head GEMV, two argmax stages last): **383** a token on one card, **403** eager; the arms `route` 345,
  `router` 345, `conv` 353, all three **277**; the synthetic N 4: 41 (34 with all arms); two cards +0 (`copy`) / +2
  (`peer`). `lfm2moe_plan_test` asserts every one; `ProfileEvents`' 1024 holds them all.
- **The norm sites.** One launch a site (`lfm_res_norm`, 8 work-groups of 256 - every work-group folds the whole row
  and reduces Σ r² by the same fixed tree, writes only its chunk): `_S<s>` folds the pending split-K slices from
  h[cur] into h[1 - cur] (the residual ping-pongs; the capture asserts a fold never writes the buffer it read), `_NF`
  only normalises. The plan's per-type table mislabelled two sites; the rule as built: layer 0's operator norm `_NF`
  (the gathered embedding), layers 1-2's `_S` (the dense down slices), every later operator norm `_NF` (lfm_moe_down
  folded), every ffn_norm `_S`, the final embedding_norm `_NF` (or `_S` after a dense last layer).
- **The kernels** (`src/kernels/lfm2moe/`, `kernels::lfm2moe`): `lfm_prep.cl` (`lfm_res_norm` + its `_R` router arm,
  `lfm_silu_mul` at 11776, `lfm_attn_prep` at head_dim 64 with RoPE on every attention layer, `lfm_embed_gather` from
  the tiled embedding), `lfm_conv.cl` (`lfm_conv` over the Bx ring; `lfm_conv_in`, the conv arm: gemv.cl's inner loop
  at split-K 1 on the conv-tile `in_proj` with the conv as its epilogue), `lfm_moe.cl` (`lfm_router` - 64 rows, per
  column 8 spans of 256 k then a stride-4/2/1 tree, the order the `_R` arm repeats -, `lfm_route` - §2.3's chain,
  exact ties to the lower id, the route row {ids[4] rank order, w[4], sel[5], d} -, `lfm_moe_gate_up` (+ `_R`: the
  route recomputed in every work-group) and `lfm_moe_down` - the rank-order fp32 combine from t0, one rounding, the
  fold), `lfm_expert.h` (the one expert-address macro; ocloc gets `-I` the family directory), `lfm_attn.cl` (kol_attn's
  full path at head_dim 64: 4 sub-groups, 4 positions a sub-group a wave, 4 elements a lane, scale 0.125),
  `lfm_attn_eager.cl` (the reference's bf16 chain, four launches). Binary names carry every define (the work-group count
  `W8` and the layout `L<l>` the plan's examples left out). Reused at LFM2's shapes: gemv.cl at every S x layout of the
  five int4 shapes, gemv_i8w / gemv_bf16 for the tied head, argmax.cl at 128000 / 125017.
- **The int4 cells.** The decode list binds `model::Lfm2MoeDesc::linears`' {S, layout} (24b's PROVISIONAL: S 1
  everywhere; layout 1 for in_proj / out_proj / o_proj / the dense down, 0 for q||k||v and the dense gate||up) - **no S
  pick has been measured**: P0's `--sweep-s` (r51.p0) produces them; a re-pick edits the descriptor and adds the
  matching `lfm_res_norm` / `lfm_conv` / `lfm_silu_mul` / `lfm_attn_prep` lines (the names test names any missing one).
  `GEMV_BLOCK2D` (layout 0 only) is set on no LFM2 cell.
- **P0's stand-ins** (`tools/probe/probe_lfm_step.cc`, `lfm_step_shapes.h`): moe.cl at E64 T4 D2048 I1536 (5 slots: the
  shared slot reads a 65th block - x 4/5 PROPOSED), Kolibri's attention at Q32 KV8 (head_dim 128 - x 1/2 PROPOSED),
  `kol_norm_finish` at K 2048 over 8 work-groups as the ONE launch a norm slot (keeps 383; `--pair` times stage A of
  the two-launch form separately), the router as `gemv_bf16` N 64, noops for conv / silu / embed. The lists bind 24b's
  cells, or `--sweep-s`'s picks at 24b's layouts.
- **Two cards.** The cut hands off h (4 KB) at a MoE cut - `pp_landing_layout(4096, 0)`, the receiving norm `_NF`
  recomputes Σ. At a DENSE cut (split 1 or 2, e.g. the synthetic N 4 at split 2 that the plan's test names) the down
  GEMV's slices are still pending: they cross too (S x 2048 fp32 in the landing's sums region) and device 1's first
  norm is `_S` - departure from §4.5's "h only", which holds for every cut the real model's split picks (20).
- **The host twin and the fixture.** `tests/kernels/lfm2moe_ref.h` (namespace `lfmref`) against
  `tools/oracle/lfm2moe_fixture.py`'s torch values (transformers 5.19.0's own RoPE and eager attention, 24a's restated
  ops): BITWISE on every case, including the fine norm rows, the router's fp32 order (0 of 4096 logits off torch's bf16
  linear), the whole MoE block on int4 experts whose scales make every dequantised weight a bf16 (0 of 8192 off), and
  eager outputs at 3000 keys. torch's own tie order at the 4th / 5th is not the lower id (24a): the crafted tie rows
  compare weights as multisets and assert our rule.
- **F3's bars as built** (`tests/golden/lfm2moe_golden_test.cc`): tokens tie-aware (golden_common.h); routes as SETS
  per (row, MoE layer) - one-expert differences within `B70_LFM_TIE_TOL` (default 4e-3, RULED) are near-ties, a 0 gap
  undetermined, anything else fails; h taps median >= 0.9998 / min >= 0.99 a layer (PROPOSED); the conv state after
  the prompt to the h min bar and its exact fraction printed - **not bitwise** as the plan asked: the engine's int4 GEMV
  ((q - 8) x scale in fp32) and the engine-format reference (bf16(q - 8 x scale) through torch's bf16 linear) round
  differently, so Bx differs past the first in_proj on real or synthetic weights. The kernel test and F4 hold the ring
  bitwise.
- **The CLI.** `b70-decode` dispatches `lfm2_moe` before the Qwen rules (`cli/lfm2moe_decode.h`): `--ids`, `--bench`,
  `--profile` (one ingest at `--depth`, then `--repeats` x `--steps` profiled replays: per kernel+binary in-situ us,
  Σ, the fence wall, the gap per launch), `--lm-head bf16|int8` (int8 the default), `--max-len N|auto` (16384 default),
  `--pp 1|2`; refused before the device: `--prefill*` (24d), `--mtp` / `--spec`, `--kv-cache int8` (decision 11),
  `--layers`. `b70-serve` refuses `lfm2_moe` naming 24e. The bench prompt is `lfm2moe_short.ids`'s first 46 ids (BOS +
  prose).
- **Generated data:** `tests/kernels/lfm2moe_fixture.h` (754 KB, committed; the command and versions in its header).
  The synthetics' golden sets are the box's (r51.synth_golden: `lfm2moe_oracle.sh export`); none was made on the Mac.
- **Not done here:** every card stage (P0 first), every kernel's first ocloc build, 24q's exports (the RTN / AutoRound
  gates SKIP), Task 8's speed rows and decision 6's default (`B70_LFM_FUSE` stays `none`).

## 15. 24d as built (2026-10-20; the Mac only - nothing has run on a card, no lfm_pf_* binary has been compiled by ocloc)

Plan: `docs/superpowers/plans/2026-10-15-spec24d-lfm2moe-prefill.md`, branch `spec24d-lfm2moe-prefill`. Box
validation: queue row 52 (`tools/box_validate/stages.sh` `row 52`). Every count below is derived and asserted by a host
test or by the walk itself.

- **The walk** (`runtime/lfm2moe/lfm2moe_prefill.{h,cc}`; Kolibri's `kolibri_prefill.cc` shape, one immediate list a
  device, no host wait inside a chunk): device 0 `lfm_pf_embed_gather` (1); a conv layer 20 (operator norm, in_proj
  6 slabs x (dequant + `pf_gemm_T0`), `lfm_pf_conv_seed`, `lfm_pf_conv`, out_proj 2 x 2, ffn_norm); an attention
  layer 14 (norm, q||k||v 3 x 2, `lfm_attn_prep` at M 2048, `lfm_pf_attn`, o_proj 2 x 2, norm); a MoE FFN + 11
  (`lfm_router` / `lfm_route` at M 2048, sort, gather, gate||up 2 batches x 2, down 1 x 2, the combine with the fold);
  a dense FFN + 50 (gate||up 23 slabs x (dequant + `pf_gemm_T0_SILU`), down 2 x 2). **1259 launches a chunk at 40
  layers** (1 + 2 x 70 + 28 x 31 + 10 x 25), the same at every C and on one card or two; the head 4 (decode's M 1
  norm over the last row, the head GEMV, two argmax stages). The norm sites are 24c's rule at M 2048: `_S1` wherever
  rows are pending (every ffn_norm; the operator norm after a dense layer), `_NF` otherwise - every prefill linear is
  ONE unsplit GEMM, so the folds are S 1 whatever decode's S; the residual ping-pongs between `h0` / `h1`, and where
  each device's walk takes and leaves it (`prefill_resid_in` / `_out`) is a function of the placement alone (the walk
  asserts it; a replayed recording needs no walk to know it).
- **Each linear at its own layout** (24b review I1): the slab binary is `pf_dequant_slab_K<K>_N<N>_L<layout>` at the
  weight's `model::Lfm2MoeDesc::linears` cell - layout 1 for in_proj / out_proj / o_proj / the dense down, layout 0 (the
  GPTQ-native `.qweight` / separate `.scales`) for q||k||v and the dense gate||up today; `pf_dequant_slab.cl`'s
  existing LAYOUT 0 / 1 branches, no new kernel. Every LFM2 width is a whole number of 1024-column slabs (the kernel's
  rule). `lfm2moe_pf_kernels_test` holds each slab to the GPTQ source column (in_proj via `repack_in_proj`, q||k||v and
  the dense down via `repack_fused_linear`).
- **The conv over a chunk** (`lfm_pf_conv.cl`, one binary, two entry points, plus a `_T` build for the arm conv's
  conv-tile in_proj): `lfm_pf_conv_seed` copies the ring slots of p0 - 2 / p0 - 1 into a flat `seed` [2][2048] (0 below
  position 0) in a launch BEFORE the conv; `lfm_pf_conv` (grid (8, ceil(C / 16)), 16 rows a work-item) recomputes the
  Bx of its first row's two predecessors from in_proj's rows (`rne(rne(0 + B) · rne(0 + x))`, decode's sum at S 1)
  and writes only the chunk's last min(C, 2) Bx into the ring; it never reads the ring. Measured on the Mac
  (`lfm2moe_pf_ref_test`, host): a 2100-position prompt at the real width under 13 schedules (2048 + 52, chunks of 64,
  1 and 2, splits at 1..5 / 64 / 2047 / 2048 / 2049) is BITWISE decode's `conv_step` one position at a time - y and the
  ring slots decode reads next -, from a garbage ring; torch's conv over a prompt in chunks of 64 + 6 (fixture case
  (h), `conv_taps` with the state carried, asserted equal to one call in torch) bitwise; `lfm2moe_run` (UHD 630,
  indicative) exact at (p0, C) = (0, 1) .. (4097, 300), plain and `_T`.
- **The grouped MoE** (`lfm_pf_moe.cl`: `kol_pf_moe.cl`'s design minus the shared block): ids staged as uchar, one
  expert a lane (64 lanes); **tmax = floor((4C + 64 x 31) / 32), 318 at 2048** - the plan's formula carried Kolibri's
  `+ ceil(C / 32)` shared-tile term, which this model has no use for (the bound Σ ceil(c_e / 32) <= that floor is exact;
  the host test reaches it with 63 experts at 129 rows and one at 65); header [0] tiles, [1] 0, [2] rows, [3] C,
  [4 + e] counts (80 words). The dequant reads each expert through `lfm_expert.h`'s macro and skips an expert with no
  row; the batches are Kolibri's 512 MiB (gate||up 42 blocks a batch -> 2 batches, down 85 -> 1: derived). The combine
  is `lfm_moe_down`'s epilogue over sorted rows (t_k rounded, rank order, fp32 without fma, one rounding) with the fold
  in place. Only rows [0, C) are routed (the sort reads C route rows): no padding row reaches an expert.
- **Flash at head_dim 64** (`lfm_pf_attn.cl`): `kol_pf_attn.cl`'s full path with NDA 4, scale 0.125, the window and the
  ring removed (every LFM2 attention layer is full); HPW 4, RPW 16, KT 64 PROVISIONAL. Every row's result is a function
  of its absolute position (the key tiles start at key 0, a wholly masked tile adds exact zeros), so a split ANYWHERE
  is bitwise one chunk - which is why `lfm2moe_prefill_test` holds every split bitwise, not only multiples of 64.
- **The engine** (`lfm2moe_prefill_engine.cc`, `b70_lfm2moe_prefill` beside `b70_lfm2moe_runtime`: Kolibri's
  arrangement): `prefill(ids, chunk)`, `prepare_prefill`, `set_prefill_replay` (`B70_PREFILL_REPLAY`), the launch
  counters, `read_prefill_routes`; with `debug_tap` the last chunk copies row C - 1 of each layer's output into the
  decode tap (never under a recording). Two cards: a second `PipelineLink` in copy mode, h [C][2048] a chunk and at a
  DENSE cut the down's pending fp32 rows [C][2048] in the sums region (24c's dense-cut departure carried to the
  chunk), sequentially; P4's lost hand-off throws naming device 1 and marks the engine until reset().
- **The CLI**: `--prefill`, `--bench --prefill-length N` (the pp row), `--prefill-chunk C <= 2048`,
  `--prefill-backend l0` (l0-int8 / sycl-tla refused naming spec 24 §11); a prefilling run plans the scratch
  (`runtime::lfm2moe::prefill_sizes`: 788,687,408 B a device, derived; 131072 positions + the prefill still fit one
  B70: 18.40 GB of 32.53) and the prefill link into `--max-len` and the fit; `B70_PREFILL_PROFILE=1` prints the phase
  table (the conv's two launches under the `gdn_seed` / `gdn_conv` rows).
- **Departures from the plan (rulings):** the `_PF` lines are 24c's names at M 2048 (the M in the name is the prefill
  marker); one conv binary with two entries (`lfm_pf_conv_K2048_C2048`, Kolibri's one-binary-many-entries) instead of
  two names, plus `_T`; a fourth source `lfm_pf_prep.cl` (the gather from the chunk's ids - decode's gather reads
  Control's eight cur_token slots); the dense gate||up through `pf_gemm_T0_SILU` (no fp32 [C][23552] partials); the
  launch formula as Kolibri's `prefill_device_launches(d, placement, dev)` (chunk-independent) instead of a
  (desc, chunk, B, B') signature; F3 on prefill as the golden test's `prefill[:chunk]` mode (Kolibri's / Muse's
  precedent) instead of a separate `lfm2moe_prefill_gate_test`; the fixture's "600-id prompt at chunk 64" case replaced
  by case (h) (the C++ twin has no whole-model forward to compare a prompt's logits with; the engine-level gate is
  F4's on the card).
- **Not done here:** every card stage (row 52, after row 51), every lfm_pf_* kernel's first ocloc build, 24q's
  exports (the RTN / AutoRound gates SKIP), the rate (r52.speed) and the attention tile sweep (HPW / RPW / KT: the
  variants for it are not built - a sweep adds their CMake lines and names).


## 16. 24e as built (2026-10-20; the Mac only - nothing has run on a card)

Plan: `docs/superpowers/plans/2026-10-15-spec24e-lfm2moe-serving.md`, branch `spec24e-lfm2moe-serving`. Box validation:
queue row 53 (`tools/box_validate/stages.sh` `row 53`). Every count below is measured by a host test unless marked.

- **Template and tokenizer.** `chat::Template` renders `chat_template.jinja` (sha256 `ea663864...`) byte for byte as
  transformers 5.19.0's `apply_chat_template` on 32 message lists (`tests/tokenizer/lfm2moe_template_cases.json`: string,
  int, negative, big-int, float, bool, null, list and nested-dict arguments, quotes / backslashes / newlines / tabs / CR /
  non-ASCII in a string, two calls, content then calls, a call without `function` or without `arguments`, string arguments
  `""` / `"{}"` / `"null"`, tool results, reasoning kept and dropped by `preserve_thinking`, the three reasoning field
  names, `</think>` inside content, content lists / odd parts / a mapping, `CONTINUE_FINAL_MESSAGE_TAG`) and on the 14-message
  agentic session (33 renders, `lfm2moe_template_test`; fixtures from `tools/tokenizer/dump_lfm2moe.py` in the oracle image with the
  5.19.0 site, tokenizers 0.23.2). The tokenizer (Rust crate 0.22.2) equals HF's id for id: 125017 ids, the 124 added tokens by
  name / id / `special` flag (`<|tool_call_start|>` 124905 and `<|tool_call_end|>` 124906 are NOT special), 21 texts, `corpus.txt`'s 10240
  lines (148,442 ids) and every render's ids (`lfm2moe_tokenizer_test`). **The template needed minja patches** (the plan
  allowed "only if a gap is found"): the tool-call capability probe did not know `fn(argument_needle=` and so polyfilled every call
  history into JSON content; the arguments rewrite that stays on when `arguments` must be an object threw on a call without
  `function` / `arguments`; `x in [..]` skipped falsy items (`'' in ['']` was false, so `(args | trim) not in ["", "{}", "null"]`
  raised on an empty arguments string); and a float printed through `| string` / `{{ }}` / `tojson` was `std::to_string`'s
  or nlohmann's, not Python's repr (`1.000000`, `1e+15`). Each is marked `b70 patch (spec 24e)`, tested against Jinja2 3.1.6
  (`minja_ext_cases.json`: `in_falsy`, `float_str`, `float_tojson`) or by `minja_ext_test`'s pythonic probe case, and recorded
  in `third_party/VERSIONS`; every other model's template, parser and server test is unchanged. The agentic fixture's ids differ
  from 24a's golden ids from position 281 on: that is the tool schemas' key order (the server's JSON objects are sorted; 24a rendered
  the session as written), the known limitation - not a template difference.
- **The parser** (`src/server/toolcall_lfm2.{h,cc}`, `tests/server/toolcall_lfm2_test.cc`). Every HF render of a call parses
  back to its message and every parsed turn re-renders to the same bytes (28 calls over 27 cases); the idioms of §4.6 pass as listed,
  with these as-built readings: (1) **a quote closes a string by lookahead, not by retry**: the candidate closer is accepted
  only if what follows (after whitespace) is what the grammar allows there - for a keyword argument's value `)` (then the end,
  `,` or `]`) or `,` followed by `name=` / a quoted name / `)`; in a list `, ] )`; a dict key `:`; a dict value `, }` - so the cost
  is linear (100,000 quotes in one string: 0.6 ms; 100,000 look-alike closers: 5 ms) and there is no backtracking to bound;
  `quote_recovered` counts the quotes kept. (2) A string inside a list / dict decodes JSON's set **plus its own quote's escape**
  (`['it\'s']` -> `it's`; JSON has no single-quoted strings); a top-level string decodes only `\\`, its own quote, `\n`, `\r`.
  (3) Calls are emitted when the text after their `)` has settled how the last string ended (re-read at every piece while the block
  is under 4 KiB; above it, after a read that finishes no call, only once the block has grown by an eighth - a 120 KB argument
  streamed byte by byte: 80 ms), at the latest when the
  block ends; the deltas do not depend on when. (4) A block is read up to its first 1 MiB; calls closed there are kept and the rest is
  a dropped tail, with none a complete block is content (a block over 1 MiB with a valid early call keeps it: the spec's "a block
  > 1 MiB is content" cannot hold together with emitting at `)`). (5) After a valid call, the first thing that is not a call ends the
  block and the tail is dropped (`dropped_unfinished`); an invalid FIRST call makes the block content. (6) `arguments` objects are
  nlohmann's: keys sorted, not "in the call's order". (7) The stream is split-invariant (whole = every two-way split = byte at a
  time = random pieces, outputs and counters) on the cases, 1500 generated outputs and 3000 damaged ones. `PythonicRules`
  counts only successful parses and reaches `--log-requests` as `parser.rules` (`OutputParser::stats`).
- **Chat format and sampling.** `ChatFormat::Lfm2`; `Sampling::repetition_penalty` + `const SeenIds* seen`;
  `apply_repetition_penalty` (HF's rule on gathered-then-scattered distinct ids, ids >= `vocab_used` ignored, before
  `filter_probs`); `generation_sampling` reads `repetition_penalty` and takes HF's `GenerationConfig` default for a field the file
  does not name (T 1.0, top-k 50, top-p 1.0, penalty 1.0: Kolibri's and Muse's files name all three, so theirs are unchanged);
  `generation_eos_ids` (an integer or a list) moved out of `b70_serve.cc` to be testable; a request's `repetition_penalty` is any finite
  number > 0 (vLLM's rule, no upper bound; `null` = absent; the same check holds for `generation_config.json`). The server keeps the request's `SeenIds` (the prompt's distinct ids, then every consumed id) when the penalty is not 1 and
  hands it to the engine with each step; **an engine that does not apply it ignores the field** (Qwen3.8, Agnes, Ornith, K2,
  Kolibri, Muse), so no other model's sampling or output changes. The field's validation is not model-specific, though: a request
  that sends `repetition_penalty` as 0, a negative, a string, a bool or a non-finite number now gets a 400 on every model (it was
  ignored before); `null` and any finite value > 0 are accepted.
- **The engine side.** `state_runs` / `kv_runs` (`lfm2moe_sizes.{h,cc}`): the two conv-ring rows (positions pos-1, pos-2; a position
  below 0 is a zero row written to slot `(q+16) % 16`, which `lfm_conv` never reads) per conv layer, 60 runs of 4096 B = 245,760 B,
  + the 10 attention layers' K then V (20 runs, 20,480 B a position), one host layout under `--pp 1` and `--pp 2`;
  `Lfm2MoeEngine::state_bytes` / `save_state` / `load_state` / `kv_bytes` / `save_kv` / `load_kv` / `set_block_hook`
  (Kolibri's calls on LFM's allocations); the hook in the prefill walk at every block end (`set_ctl(end, 0)` first) and at the prompt end.
  `cli::lfm2moe::Lfm2MoeEngineAdapterT` (`src/cli/lfm2moe_serve.h`): **the next id is chosen at the start of `step()`**, from the logits row of
  the previous replay or of the prefill's head - Kolibri's adapter samples at the end of `step()`, before the server has added the
  consumed id to the request's `SeenIds`; greedy requests keep the device argmax (an id the tokenizer lacks becomes the masked
  row's argmax), `B70_LFM_PENALTY_GREEDY=1` routes a greedy request with a penalty through the host (default penalty 1.05, so every greedy
  request unless it sends 1.0); **a seeded request re-seeds the RNG at its first step** - EngineAdapter's and Kolibri's seed once per
  process, so a second seeded request there continues the first's stream (r28.serve's "seeded twice, identical" probe would
  differ for them; not changed here). `b70-serve <lfm2moe>`: `serve_lfm2moe` (one card by default, `--pp 2`, `--max-len auto` with the
  prefill scratch planned in, `B70_LFM_TOKENIZER_JSON`); refusals before the device: `cli::lfm2moe::check_serve_flags`
  (`--mtp`, `--spec mtp|lookup|dflash`, the proposers' tuning flags, `--draft-vocab`, `--layers` / `--ple-dir` / `--tokenizer`,
  `--chat-template`), `check_args` (`--kv-cache int8`, the prefill backends, `--device` with `--pp 2`, the split rules) and
  `check_serve_max_len` (above 131072) - header-only, run on the Mac by `lfm2moe_serve_args_test`, and through the binary on
  the box by `cli_reject_serve_lfm2moe_*` (21 tests).
- **Tests (Mac).** `lfm2moe_snapshot_test` (a fake engine with the real ring / KV layouts, one card and two): the layout on the real
  shapes at 15 positions and 4 placements, save / load bit for bit, restores at 0 / 1 / 2 / 3 / 15 / 16 / 17 / 18 continuing as
  the cold session (KV, rings, ids), through `PrefixSession` at 2048 / 4096 / 2049 / 5000 / 300 == cold, a snapshot moving between one
  card and two, and the server over HTTP (greedy == the decode run, a cached second turn, a seeded request reproducible on a
  fresh and on the same server, the penalty against an independent HF-ordered simulation incl. the greedy route, no undefined id);
  `lfm2moe_server_test` (scripted engine over HTTP: reasoning + content + a call in the body and streamed, the cut blocks of Review
  Focus 3, `enable_thinking` ignored with one stderr line, `preserve_thinking`, the sampling a request carries and the `SeenIds`, `parser.rules`
  in the log); `lfm2moe_sampling_test`; `lfm2moe_rules_test`. Mutations checked: the quote lookahead, the penalty's rule, the
  `SeenIds` update, the state's positions, the per-request seed - each fails its test.
- **Box tools.** `engine_generate.sh` runs LFM (2048 new ids); `tools/toolcall/lfm2moe_rules` (decision 7's evidence);
  `tools/prefix/agentic_loop.py` + `tools/box_validate/agentic_run.sh` (the recorded session as a live loop: the model's own turns
  fed back, so the reordered-call misses are real); row 53's stages. The synthetic checkpoints carry no tokenizer files:
  `r53.synth_files` copies them.
- **Departures from the plan (rulings).** `template_test` is not reused (`lfm2moe_template_test` is its own, with the agentic case); the
  plan's `sampling_test` / `openai_test` additions live in `lfm2moe_sampling_test` (`sampling_test` links the device runtime and
  is disabled on the Mac); the card-side `lfm2moe_server_test <ckpt>` is `golden_server_test --chat` + `serve_probe.sh` (the model
  with random weights never writes a call; the tool-call paths are the host test's); stage ids `r53.snap` / `.server` / `.a4` /
  `.agentic` became `r53.snap` / `.snap_rtn` / `.serve` / `.chat` / `.real` / `.a4` / `.agentic` / `.opencode` plus `.k0` /
  `.synth_files`.
- **Parser limits (decision 7's notes).** Known limits of the parser (documented, not changed): (a) an unescaped Windows path ending in a backslash, `ls(path='C:\dir\')`, leaves the string open (the `\'` reads as an escaped quote) and the block becomes content - recovery is deferred; (b) a JSON-format block `[{"name": ..., "arguments": ...}]` is not a pythonic call and becomes content; (c) a malformed call after a valid one is dropped, and only `dropped_unfinished` counts it; (d) a second `<|tool_call_start|>` inside an open block is part of the tail the first non-call text started, so it is dropped; (e) an integer beyond 2^64 becomes a double; (f) an unescaped `print('a', end='')` inside a single-quoted argument is mis-split by the quote lookahead (`', end='` reads as the string's end and a next argument). The counters of `tools/toolcall/lfm2moe_rules` show (c) as `dropped_unfinished` and (a), (b) as a block left as content.
- **Not done here:** every card stage (row 53, after rows 51 and 52), 24q's exports (the RTN / AutoRound gates SKIP), A4 and the
  agentic hit rate on the engine (decision 5's bar and decision 7's counts are PROPOSED evidence for the operator after `r53.a4`),
  `--spec lookup` for this family (needs M-row verify lists), a rotated-key-order-preserving `ToolCall` (the prefix cache misses
  at a call whose arguments the model wrote unsorted)), recovery of a string ending in a backslash (a), and counting a malformed call after a valid one other than in `dropped_unfinished` (c).
