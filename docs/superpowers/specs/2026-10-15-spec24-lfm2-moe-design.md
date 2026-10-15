# Spec 24 - LiquidAI LFM2.5-24B-A2B (`lfm2_moe`) in the engine

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
  top-4 sets equal except near-ties at the 4th / 5th `sel` (a tolerance from 24a's gap distribution, decision 4 of the
  plan, as `B70_LFM_TIE_TOL`; an exact 0 gap undetermined). The fused arms are bitwise the as-built list.
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
