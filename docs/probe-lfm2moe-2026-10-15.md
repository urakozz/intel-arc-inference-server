# LFM2.5-24B-A2B (`lfm2_moe`) facts - spec 24a, 2026-10-15

What spec 24's later stages are built against: the rounding chain of transformers **5.19.0**'s
`Lfm2MoeForCausalLM` cited into the INSTALLED sources (`M:line` = `models/lfm2_moe/modeling_lfm2_moe.py`,
`C:line` = `configuration_lfm2_moe.py`, other files by name), the checkpoint from its small files and shard
headers, tokens, the chat template, torch's CPU tie order and the short conv's accumulation. Every number is
**measured** (by the command named under each section) unless marked derived. Machine: the Mac's
`agnes-ref-img` (torch 2.14.1+cpu, Python 3.12) with the 5.19.0 site beside it (`tools/oracle/qwen4exp_env.sh`).
Snapshot `LiquidAI/LFM2.5-24B-A2B@1769516a` (complete in the HF cache: 10 shards, no partial blobs).

Commands (from the worktree root):

```bash
# the citations, the checkpoint, the bytes, tokens, licence: no torch
PYTHONPATH=oracle-out-q4exp/site python3 tools/oracle/lfm2moe_facts.py <snapshot> --section cite,checkpoint,bytes,tokens,licence
# the renders, the prompts, the tie order, the conv's accumulation: in the container (5.19.0 site, torch)
docker run ... agnes-ref-img python3 tools/oracle/lfm2moe_facts.py <snapshot> --section template,prompts,ties,conv
# the tiny model end to end
docker run ... agnes-ref-img python3 tools/oracle/lfm2moe_tiny_check.py <yujiepan/lfm2-moe-tiny-random snapshot>
```

## 1. What 5.19.0 runs (the tiny model, `lfm2moe_tiny_check.py`)

| fact | measured |
|---|---|
| experts implementation picked by 5.19.0 on torch 2.14.1 CPU | **`grouped_mm`** (bf16 and fp32; `modeling_utils.py:1888`: grouped_mm dispatches) |
| the conv kernels | `causal_conv1d` is not installed: `causal_conv1d_update` and `causal_conv1d_fn` run **the torch functions** (`hub_kernels.py:984`'s fallback; `is_new_implementation` false) |
| `tie_word_embeddings` | absent from the tiny's config.json, the class default True (C:66): `lm_head.weight` shares `embed_tokens.weight`'s storage. The real config.json says `true` explicitly |
| `expert_bias` under `dtype=bf16` | stays an **fp32** buffer [E] (M:205) |
| `generate` (cached) vs one uncached forward, bf16 | **bitwise** over 8 steps |
| the same in fp32 | NOT bitwise: max \|diff\| 1.8e-7, argmax equal; the first differing module output is `model.layers.0.conv.in_proj` (2.98e-8) - an `nn.Linear` at M = 1 (a decode step) against M = 32 (the uncached forward): transformers' own two paths. F1 compares the streamed reference with transformers on the **same** path (bitwise in fp32 too, `test_streamed_equals_hf_fp32`) |
| sdpa against eager (bf16, 32 ids) | max \|logit diff\| **2.93e-3**, argmax equal. The reference never uses sdpa (decision 13) |

## 2. The rounding chain (`--section cite`, located in the installed 5.19.0)

| point | where |
|---|---|
| RMSNorm: fp32 variance | M:63 |
| RMSNorm: x^ cast back to bf16, then w * x^ (two roundings) | M:65 |
| q / k layernorm over head_dim (64) | M:302 |
| q / k norm applied before RoPE | M:316 |
| RoPE inv_freq in fp32 | M:104 |
| RoPE freqs = pos x inv_freq in fp32 | M:110 |
| RoPE cos / sin cast to the model dtype | M:115 |
| rotate_half | M:216 |
| RoPE: rne(rne(q cos) + rne(rotate_half(q) sin)) | M:244 |
| attention scale head_dim^-0.5 = 0.125 | M:296 |
| eager: bf16 Q K^T then x scaling | M:274 |
| eager: softmax in fp32, cast to bf16 | M:278 |
| eager: bf16 P V | M:280 |
| short conv: in_proj then B, C, x = chunk(3) in that order | M:438 |
| short conv: Bx = B * x (bf16) | M:439 |
| short conv: cached decode step (seq 1) | M:446 |
| decode: cat([state(3), new]) - the state is Bx | M:368 |
| decode: F.conv1d without padding, the last output | M:370 |
| prefill: F.conv1d at padding K - 1 = 2 (zeros before position 0) | M:386 |
| short conv: y = C * conv(Bx) | M:463 |
| short conv: out_proj | M:465 |
| cache: prefill pads the state to conv_L_cache with zeros | cache_utils.py:1123 |
| cache: later chunks see cat(state, new) | cache_utils.py:1127 |
| cache: keeps the last conv_L_cache columns | cache_utils.py:1132 |
| cache: layer type conv -> LinearAttentionLayer | cache_utils.py:1292,1312 |
| router: logits = F.linear (bf16) | M:183 |
| router: s = sigmoid(logits) in the logits' dtype | M:184 |
| router: sel = s + expert_bias (fp32 buffer: promotes) | M:186 |
| router: topk on sel | M:187 |
| router: weights = s gathered (not sel), bf16 | M:188 |
| router: w / (sum(w) + 1e-6), bf16 | M:193 |
| router: x routed_scaling_factor (1.0) | M:194 |
| expert_bias is an fp32 buffer | M:205 |
| experts: eager fallback, bf16 index_add_ by ascending expert id | M:166 |
| experts: grouped_mm, sort by expert | moe.py:404 |
| experts: grouped_mm, a = act(gate) * up | moe.py:539 |
| experts: grouped_mm, t_k = proj * w (bf16) | moe.py:484 |
| experts: grouped_mm, sum over the k slots (fp32 accumulation, one rounding) | moe.py:495 |
| experts implementation default grouped_mm, eager when it cannot dispatch | modeling_utils.py:1888 |
| conv kernels: hub / package / torch fallback order | hub_kernels.py:984 |
| dense MLP: w2(silu(w1 x) * w3 x) | M:128 |
| dense layers: layer_idx < num_dense_layers | M:480 |
| layer: h = mixer(operator_norm(h)) + residual | M:512 |
| layer: h = h + FFN(ffn_norm(h)) | M:513 |
| model: embedding_norm AFTER the last layer | M:622 |
| model: the conv layers' mask (None unpadded) | M:605 |
| head tied to the embedding | M:632 |
| head: logits = lm_head(h) | M:690 |
| sdpa supported (the default when not pinned) | M:526 |
| config: tie_word_embeddings default True | C:66 |
| config: norm_eps default 1e-5 | C:71 |
| checkpoint conversion: experts.*.w1 / w3 -> gate_up_proj | conversion_mapping.py:1163 |
| recurrent mask: None for an unpadded batch | masking_utils.py:1471 |
| repetition penalty (generation) | logits_process.py:419 |

Each point is restated in `tools/oracle/lfm2moe_ref.py` and held **bitwise** to the module it restates on the
tiny models (`test_lfm2moe_ref.py`, plan 24a Task 3):

- **the norm** (`rms_norm`): bitwise the module (and the 64-wide q norm); a one-rounding variant (`bf16(w x
  f32(x^))`) differs on 32,332 of 131,072 elements of 512 random rows - the two roundings are live.
- **the router** (`route`): bitwise `Lfm2MoeTopKRouter` on 512 rows of `pattern8` layer 2. The bias selects
  an expert with a LOWER sigmoid than an unselected one on 226 of 512 rows; weights taken from `sel` instead
  of `s` would differ on all 512 (the bias never reaches the weights); a zero bias changes the selection on
  443 rows and leaves the weights of an unchanged selection the same; one fp32 renormalisation instead of the
  bf16 sum / `+ 1e-6` / bf16 division differs on 212 rows - the bf16 chain is live.
- **the combine** (`expert_outputs` + `combine`): `grouped_mm`'s restated chain - `t_k = rne(proj_k w_k)`, then
  ((t_0 + t_1) + t_2) + t_3 in fp32 in SLOT order (descending sel), one rounding - is bitwise the experts module
  AND torch's bf16 `.sum(dim=1)`; the eager fallback's (ascending expert id, a bf16 accumulator rounded at every
  add) is bitwise the eager module; each order differs from the other on all 256 rows tested. **The engine
  copies `grouped_mm`'s** (the one 5.19.0 runs).
- **the short conv** (`short_conv`): bitwise `Lfm2MoeShortConv` for a 37-row prompt and a cached step; the state
  is Bx of the last two positions (transformers keeps 3 columns, decode reads the newest two: M:368-371);
  positions 0 and 1 read zeros (a non-zero history changes rows 0..1 and nothing after); swapping taps 0 and 2
  changes the output.
- **the end of the model**: dropping `embedding_norm` changes the logits; the head IS the embedding (128,000
  rows); `layer_major` + `final_logits` = the model's forward bitwise.

## 3. The conv's accumulation (`--section conv`)

torch 2.14.1 CPU's bf16 depthwise `F.conv1d` (3 taps, padding 2 - the short conv's prefill call) on **100,000
rows x 2048 channels** of `Bx`-like values (a bf16 product of two Gaussian rows) with bf16 taps:

| the engine's candidate | rows that differ from torch |
|---|---:|
| fp32 taps ascending, ((w0 x[t-2] + w1 x[t-1]) + w2 x[t]), one rounding (= `fma` chained ascending: the bf16 x bf16 products are exact in fp32) | **0** (0 elements) |
| fp32 descending ((w2 x[t] + w1 x[t-1]) + w0 x[t-2]) | 72 |
| w0 x[t-2] + (w1 x[t-1] + w2 x[t]) | 72 |
| a bf16 accumulator (rounded after each add) | 99,900 |

So spec 24 §4.1's `lfm_conv` - `fma` taps ascending in fp32, one rounding - is torch's arithmetic exactly; F2's
conv test can be bitwise against the reference. The decode call (padding 0 over `[state(3), new]`) is the same
dot product: `test_short_conv` and the chunked tests hold the two calls bitwise to each other.

## 4. torch's CPU tie order (`--section ties`, `test_route_ties`)

`topk(k=4)` over 64 fp32 values with an exact tie planted at the 4th / 5th, 50 rows: the **lower id wins 26,
the higher 24** (repeat calls identical 50 / 50); batched [50][64]: the lower id 25 / 50. On the `ties` variant
(experts 3 / 11 in layer 2 and 5 / 6 in layer 4 tied at the cut on every token) the 4th slot went to expert 3 on
25 of 64 rows and to 11 on 39 (layer 2), to 5 on 29 and 6 on 35 (layer 4). torch does NOT prefer the lower id
(spec 21 §12 finding 2 holds for this router too). The engine's rule is the lower id; a row with `route.gap ==
0` is **undetermined** (F3), and the recorder writes the gap per row (`route.gap.L<l>`).

## 5. The checkpoint (`--section checkpoint,bytes`; headers read locally from the 10 shards)

7,610 tensors in 10 shards (`model-00001..00010-of-00010.safetensors`: 622, 799 x 8, 596 tensors); index
metadata `total_parameters` 23,971,585,280, `total_size` 47,943,180,288 B (= the headers' tensor bytes).

| group | count | dtype | shape |
|---|---:|---|---|
| `model.embed_tokens.weight` | 1 | BF16 | [128000, 2048] |
| `model.embedding_norm.weight` | 1 | BF16 | [2048] |
| `model.layers.L.conv.conv.weight` | 30 | BF16 | [2048, 1, 3] |
| `model.layers.L.conv.in_proj.weight` | 30 | BF16 | [6144, 2048] |
| `model.layers.L.conv.out_proj.weight` | 30 | BF16 | [2048, 2048] |
| `model.layers.L.feed_forward.w1.weight` | 2 | BF16 | [11776, 2048] |
| `model.layers.L.feed_forward.w2.weight` | 2 | BF16 | [2048, 11776] |
| `model.layers.L.feed_forward.w3.weight` | 2 | BF16 | [11776, 2048] |
| `model.layers.L.ffn_norm.weight` | 40 | BF16 | [2048] |
| `model.layers.L.operator_norm.weight` | 40 | BF16 | [2048] |
| `model.layers.L.feed_forward.expert_bias` | 38 | F32 | [64] |
| `model.layers.L.feed_forward.experts.E.w1.weight` | 2432 | BF16 | [1536, 2048] |
| `model.layers.L.feed_forward.experts.E.w2.weight` | 2432 | BF16 | [2048, 1536] |
| `model.layers.L.feed_forward.experts.E.w3.weight` | 2432 | BF16 | [1536, 2048] |
| `model.layers.L.feed_forward.gate.weight` | 38 | BF16 | [64, 2048] |
| `model.layers.L.self_attn.k_layernorm.weight` | 10 | BF16 | [64] |
| `model.layers.L.self_attn.k_proj.weight` | 10 | BF16 | [512, 2048] |
| `model.layers.L.self_attn.out_proj.weight` | 10 | BF16 | [2048, 2048] |
| `model.layers.L.self_attn.q_layernorm.weight` | 10 | BF16 | [64] |
| `model.layers.L.self_attn.q_proj.weight` | 10 | BF16 | [2048, 2048] |
| `model.layers.L.self_attn.v_proj.weight` | 10 | BF16 | [512, 2048] |

No `lm_head.weight` (tied). `lfm2moe_ref.expected_names(config, "bf16")` is exactly these 7,610 names
(`test_expected_names`).

**Parameters (derived from the headers):** 23,971,587,712 including the 2,432 fp32 `expert_bias` values;
23,971,585,280 without them - the index's `total_parameters` leaves out exactly the biases (spec 24 §1 holds).
**Active per token (derived, the head included): 2.455 B.**

**Per decode token (derived), spec 24 §3 re-derived from the shapes:**

| part | format | MB |
|---|---|---:|
| routed experts (38 x 4 x 3 x 2048 x 1536) | int4 g64 | 762.1 |
| conv in_proj + out_proj (30 layers) | int4 g64 | 267.4 |
| dense FFNs (2) | int4 g64 | 76.9 |
| attention q, k, v, o (10 layers) | int4 g64 | 55.7 |
| routers (38 x 64 x 2048) | bf16 | 10.0 |
| tied head, int8 per row + fp32 scales | int8 | 262.7 |
| weights read per token | | 1,434.6 |

KV per position 10 layers x 8 kv x 64 x 2 (K, V) x 2 B = 20,480 B (2.684 GB at 131,072); conv state per sequence
30 x 2 x 2048 x 2 B = 245,760 B. Memory: routed experts 12.193 GB, other int4 linears 0.400 GB, routers
0.010 GB, the bf16 embedding 0.524 GB, the int8 head 0.263 GB - spec 24 §3's table holds.

## 6. Tokens (`--section tokens`)

`tokenizer.json` sha256 `695be7802a0e4b8a81048f0ff5ebb7fc811a0ba5a6be63dbb24deb5a81096f41`: BPE,
`ignore_merges` true, no byte fallback; 124,893 vocab entries, 293,320 merges, 124 added tokens; ids
0..125016 **contiguous** -> **`vocab_used` 125017** (rows 125017..127999 of the 128000-row head are never a
target). No normaliser; pre-tokenizer `Split` on `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` (isolated) then `ByteLevel` (no prefix space, no regex);
post-processor and decoder `ByteLevel`.

| id | token | special |
|---:|---|---|
| 124893 | `<\|pad\|>` | true |
| 124894 | `<\|startoftext\|>` (BOS; the template writes it first) | true |
| 124895 | `<\|endoftext\|>` | true |
| 124896-124898 | `<\|fim_pre\|>`, `<\|fim_mid\|>`, `<\|fim_suf\|>` | true |
| 124899 | `<\|im_start\|>` | true |
| 124900 | `<\|im_end\|>` (EOS, generation_config `[124900]`) | true |
| 124901 / 124902 | `<think>` / `</think>` | **false** |
| 124903 / 124904 | `<\|tool_list_start\|>` / `<\|tool_list_end\|>` (unused by the template) | true |
| 124905 / 124906 | `<\|tool_call_start\|>` / `<\|tool_call_end\|>` | **false** |
| 124907 | `<image>` | true |
| 124908-125007 | `<\|img_row_R_col_C\|>`, R, C 1..10 (100 tokens) | true |
| 125008-125016 | `<\|img_thumbnail\|>`, `<\|image_start\|>`, `<\|image_end\|>`, `<\|image_split\|>`, `<\|audio_start\|>`, `<\|text_start\|>`, `<\|text_end\|>`, `<\|mixed_start\|>`, `<\|mixed_end\|>` | true |

Exactly four added tokens are not special: `<think>`, `</think>`, `<|tool_call_start|>`, `<|tool_call_end|>` - a
decode with `skip_special_tokens` keeps them. `generation_config.json`: `do_sample` true, temperature 0.1, top_k
50, **repetition_penalty 1.05**, no top_p, eos `[124900]`, bos 124894, pad 124893. `tokenizer_config.json`:
`TokenizersBackend`, bos `<|startoftext|>`, eos `<|im_end|>`, pad `<|pad|>`.

**The golden prompts re-tokenised** (`--section prompts`): `prose.txt` is 45 LFM2 ids (Qwen's committed 42),
`code.txt` 54 (61), `long.txt` 3,579 - none equal (another tokenizer); LFM2's own sets are
`tests/golden/prompts/lfm2moe_*.ids` (`lfm2moe_prompts.py`): short 100 (BOS + prose + code), 4k 4,096, 32k
32,768 (BOS + `long.txt` joined by blank lines, cut), agentic 12,474 (`lfm2moe_agentic.json` through the
template).

## 7. The chat template (`--section template`)

`chat_template.jinja` sha256 `ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef`, 125 lines.
Features minja must render: `namespace` (lines 20, 37, 42, 55), macros (4, 14, 36), `is mapping` / `is
iterable` / `is string`, `tojson` (8, 18, 29, 66), `replace` (6), slicing (60, 105), `.endswith` (102),
`.split` (100), the `{% generation %}` block (86), `raise_exception` (48), `loop.index0` / `loop.last`,
`trim`, `length`, `join`, `default` (2), the `string` filter, dict `.get` / `.items`.

The pythonic call grammar the template writes (`format_arg_value`, lines 4-12, verbatim):

```jinja
{%- macro format_arg_value(arg_value) -%}
    {%- if arg_value is string -%}
        {{- "'" + (arg_value | replace("\\", "\\\\") | replace("'", "\\'") | replace("\n", "\\n") | replace("\r", "\\r")) + "'" -}}
    {%- elif arg_value is mapping or arg_value is iterable -%}
        {{- arg_value | tojson -}}
    {%- else -%}
        {{- arg_value | string -}}
    {%- endif -%}
{%- endmacro -%}
```

So a top-level string is single-quoted with `\\`, `\'`, `\n`, `\r` escaped; a list or dict argument is JSON
(`tojson`: JSON's escapes, double quotes); everything else is Python's `str()` (`True`, `None`, `2.5`). Calls are
`name(k=v, ...)` joined by `", "` inside `<|tool_call_start|>[...]<|tool_call_end|>`, written AFTER the turn's
content with no separator. Renders through `apply_chat_template` (24e's byte-identity cases; the id counts are
`tokenizer(text)`'s):

- system + tools + generation prompt (91 ids):
  ```text
  <|startoftext|><|im_start|>system
  You are a coding agent.
  List of tools: [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["filePath"]}}}]<|im_end|>
  <|im_start|>user
  Open the file.<|im_end|>
  <|im_start|>assistant
  <think>
  ```
- no system, tools only (85 ids): the same with `<|im_start|>system\nList of tools: [...]<|im_end|>` alone.
- a call with a quote, a backslash, a newline and a CR in a string (113 ids):
  `<|tool_call_start|>[read(filePath='/w/it\'s\\a\nb\rc')]<|tool_call_end|><|im_end|>`
- int, float, bool, null (34 ids): `[f(a=1, b=2.5, c=True, d=None)]`
- list and nested dict (49 ids): `[f(xs=[1, "two", [3]], m={"k": "v'q", "n": {"x": 1}})]`
- two calls in one turn (24 ids): `[a(x=1), b()]`
- content then calls, then a tool result (112 ids): `<|im_start|>assistant\nReading.<|tool_call_start|>[read(filePath='/w/x')]<|tool_call_end|><|im_end|>\n<|im_start|>tool\nline 1<|im_end|>\n<|im_start|>assistant\n<think>`
- reasoning after the last user turn is kept (21 ids): `<|im_start|>assistant\n<think>think a</think>Done.<|im_end|>\n`
- reasoning before the last user turn is dropped (28 ids): `<|im_start|>assistant\nDone.<|im_end|>`; with
  `preserve_thinking` true (32 ids) it is kept: `<think>think a</think>Done.`
- content as a list of text parts (11 ids): `[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]` ->
  `ab` (concatenated, no separator).

The A4 set re-rendered by this template (`tests/golden/toolcall-lfm2moe`, `make_set.py --from`, key-sorted as
b70-serve renders): 36 scenarios, 748..2,460 ids (54,768 in all); every scenario's ids are
`apply_chat_template`'s, which are `tokenizer(text)`'s (`test_make_set.py test_lfm2moe_render_is_apply_chat_template`).

## 8. The licence (`--section licence`)

`LICENSE` sha256 `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`: "LFM Open License v1.0",
sections 1 (Definitions) .. with **5. Commercial Use Limitation**; "Threshold" means annual revenue of USD
10,000,000 or more. Spec 24 §1.4's summary holds; decision 9 (ruled): the operator's use is under the threshold,
our quant is published under LFM 1.0 with its notices.
