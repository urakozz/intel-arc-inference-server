# Muse-Glimmer-30B - facts from the files and the source (2026-10-15)

> DROPPED (operator, 2026-10-27): support removed from the engine in 98c13ef; kept for the record.

Spec 23a (`docs/superpowers/plans/2026-10-15-spec23a-muse-reference.md`, Tasks 1-2), as far as the **small files,
the shard headers and the source** decide it: both checkpoints' `config.json`, `generation_config.json`,
`tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, `model.safetensors.index.json`, every shard's
safetensors **header** (HTTP range requests: 2 + 6 shards, plus the drafter's one), and three sampled `qzeros` /
`g_idx` pairs (1.6 MB of range reads); transformers 5.19.0's source (installed by `tools/oracle/qwen4exp_env.sh`).
No weight shard was downloaded; nothing ran on the box or a card.

- `tools/oracle/muse_facts.py <bf16 snapshot> <gptq snapshot> --assistant <snapshot>` prints every table below
  (`--section cites` and the render need transformers 5.19.0: the oracle image with the site; the rest runs on the
  Mac host, no torch). `--scan-quant` adds the full `qzeros` / `g_idx` scan (~210 MB of range reads: box, Task 6).
- `tools/oracle/muse_tiny_check.py <tiny snapshot>` is Task 1 (below).

Citations as spec 23 abbreviates them, transformers **5.19.0**: `M:` `models/muse_glimmer/modeling_muse_glimmer.py`,
`C:` its configuration, `A:` / `AC:` `models/muse_glimmer_assistant/{modeling,configuration}_muse_glimmer_assistant.py`,
`G:` `generation/candidate_generator.py`, `CU:` `cache_utils.py`, `MU:` `masking_utils.py`, `OC:`
`utils/output_capturing.py`, `RP:` `utils/chat_parsing/response_parser.py`, `TB:` `tokenization_utils_base.py`, `T:`
the original's `chat_template.jinja`. **`muse_facts.py --section cites` holds all 47 citations of this sheet to the
installed source (the cited range must contain the quoted code): 47 / 47.** Numbers are **measured** unless marked
derived / estimated / published.

## Sources

| | |
|---|---|
| bf16 original | `meta-models/Muse-Glimmer-30B` @ `a4e59da5`: 2 shards, **1436 tensors** (627 text + 809 vision), files 59,553,435,272 B; data text **55,709,561,856 B**, vision 3,843,691,520 B; `config.json` saved by transformers 5.15.0.dev0, `dtype` bfloat16 |
| GPTQ (served) | `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ` @ `0987fe04`: 6 shards, **2684 tensors** (1875 text + 809 vision), files 22,804,270,776 B; data text **18,960,241,664 B**, vision 3,843,691,520 B; `config.json` saved by 5.15.0 |
| int4, auto_round spelling (the Mac's) | `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound` @ `ad7b9432` (downloaded whole, 2026-10-15): 6 shards, **2268 tensors** (1459 text + 809 vision), files 22,790,907,104 B; data text 18,946,929,664 B - the SAME export (the recipe's `format="auto_round,auto_gptq"`) without `g_idx`; `quantization_config` `quant_method` "auto-round", `packing_format` "auto_round:auto_gptq", bits 4, group_size 64, sym, iters 800, `autoround_version` 0.15.0, `block_name_to_quantize` "model.language_model.layers" (no `desc_act`, no `lm_head` key); its template is the older `114f55eb...` |
| drafter | `meta-models/Muse-Glimmer-30B-assistant` @ `e8192f3a`: one `model.safetensors`, 58 tensors, data **5,111,970,304 B** (file 5,111,976,608 B) |
| tiny | `optimum-intel-internal-testing/tiny-random-muse-glimmer` @ `cb70af19`: one shard, 123 tensors (50 text + 73 vision), 27.1 MB, bf16, **no `lm_head` tensor** (tied), the GPTQ repo's older 7167-byte template |

## Task 1: the environment and the tiny model (measured, `agnes-ref-img` + the 5.19.0 site)

- **The site is spec 21a's** (`qwen4exp_env.sh oracle-out-q4exp/site`): `transformers==5.19.0`, `tokenizers==0.23.2`,
  `huggingface_hub==1.33.0`, `safetensors==0.8.0`; `import transformers, torch` -> `5.19.0 2.14.1+cpu`. The image is
  unchanged.
- **5.19.0 loads the tiny as shipped** (`MuseGlimmerForConditionalGeneration.from_pretrained(dtype=bf16 | fp32,
  attn_implementation="eager")`): `layer_types` `[sliding x 3, full]`, `layer_rope_theta` `[5e5, 5e5, 5e5, 0]`,
  `sliding_window` 2048, `tie_word_embeddings` true at both levels - `lm_head` shares `embed_tokens`' storage.
- **Cached decode vs one uncached forward** (24 random ids, greedy 8, then the 32 ids at once, rows 23..30): **bf16
  bitwise**; fp32 max |diff| **6.9e-6** (not bitwise: a 1-row and a 32-row query run the matmuls at different shapes);
  argmax equal in both. (The plan expected fp32 bitwise; MG1 compares streamed against un-streamed at the SAME shapes,
  which this does not touch.)
- **`output_hidden_states=[1, 3]`** on 40 ids returns 4 entries, non-None at **1 and 3** only (OC:112-131); entry 1 is
  decoder layer 1's output bitwise. **Entry 3 is NOT layer 3's raw output: layer 3 is this tiny's last layer, and
  `capture_outputs` replaces a non-None last entry by the final-normed `last_hidden_state`**
  (`tie_last_hidden_states`, OC:294-305). The real taps `[1, 13, 25, 37, 49]` of 52 layers are unaffected (the list's
  last entry - layer 51's - is None, so nothing is replaced), and 23a's `l8` tap test uses `[1, 5]` of 8 for the same
  reason. A tap list that ends at the model's last layer would see the normed state: 23f's driver must not assume
  otherwise.

## The rounding chain (spec 23 §1.1; bf16 eager; every point a rounding to bf16 unless named)

| # | point | what transformers computes | module | source |
|---|---|---|---|---|
| 1 | embed | the bf16 row `e`; `bf16(f32(e) * pow(mean(f32(e)^2) + 1e-5, -0.5))` - **`pow`, not `rsqrt`**, no weight | `MuseGlimmerTextNormedEmbedding` -> `MuseGlimmerRMSNorm(with_scale=False)`, eps `rms_norm_eps` | M:428-436, M:451-453, M:110-113 |
| 1a | ids 200091 / 200092 | `<|video|>` / `<|patch|>` in `input_ids` are embedded as **id 0** (`llm_input_ids[multimodal_mask] = 0`) before the vision features are scattered; with no pixels they stay row 0 | `MuseGlimmerModel.forward` | M:1010-1013 |
| 2a | input norm | `xn = bf16(f32(h) * rsqrt(mean(f32(h)^2) + 1e-5) * (1 + f32(w)))` - **`rsqrt`**, `(1 + w)`, one cast | `MuseGlimmerTextCenteredRMSNorm` (eps `rms_norm_eps`) | M:122-136, M:128-129, M:370 |
| 2b | projections | `q, k, v, g = Linear(xn)` bf16 out (no bias); q 6656 -> 4096, k / v -> 256, `gate_proj` -> 4096 | `MuseGlimmerTextAttention` | M:295-311 |
| 2c | q / k norm | per head of 128: `q̂ = bf16(f32(q) * pow(ms + 1e-5, -0.5))` (`pow`, weight-less), then **`q = bf16(q̂ * 3.87)`** (the scalar after the norm's cast: two roundings); `k = bf16(norm(k))` | `qk_norm = MuseGlimmerRMSNorm(with_scale=False)` | M:312, M:330-331 |
| 2d | RoPE (sliding layers) | `cos, sin` from ONE table (`rope_parameters.rope_theta` 5e5, NEOX halves) in fp32, **cast to bf16** (M:202); `y = bf16(bf16(y * cos) + bf16(rotate_half(y) * sin))` | `apply_rotary_pos_emb` | M:186-191, M:202, M:205-209, M:233-234 |
| 2e | NoPE (full layers) | `position_embeddings=None` where `layer_rope_theta[i]` is 0: the layer's own theta is ONLY a switch | `MuseGlimmerTextModel.forward` | M:333-336, M:512 |
| 2f | scores | `bf16(bf16(q · k) * 0.08838834764831845)` (`head_dim^-0.5`), `+ mask` (bf16, min where masked), `softmax` in fp32, probabilities **cast to bf16**, `P · V` bf16 (fp32 accumulation, one rounding); GQA by `repeat_kv` | `eager_attention_forward` | M:263-269, M:291 |
| 2g | output gate | `a = bf16(attn * bf16(sigmoid(g)))`, then `o = o_proj(a)` | `MuseGlimmerTextAttention.forward` | M:357-359 |
| 2h | post-attention | `h = bf16(h + bf16(f32(o) * rsqrt(ms(o) + 1e-8) * (1 + f32(w))))` - eps **`post_norm_eps` 1e-8** | `post_attention_layernorm` (centered) | M:371, M:397-398 |
| 3 | MLP half | `xn = centered(h, 1e-5)`; `y = down(bf16(bf16(silu(gate(xn))) * up(xn)))`; `h = bf16(h + centered(y, 1e-8))` | `pre_feedforward_layernorm`, `MuseGlimmerTextMLP`, `post_feedforward_layernorm` | M:142-155, M:372-373, M:400-404 |
| 4a | final norm | `hn = bf16(f32(h) * pow(ms + 1e-5, -0.5) * f32(w))` - **plain `w`** (not `1 + w`), `pow`, one cast | `norm = MuseGlimmerRMSNorm(hidden)` (with scale) | M:101-119, M:457 |
| 4b | logits | `z = lm_head(hn)` (bf16, untied); then **four bf16 roundings**: `bf16(z * 0.19611613513818404)`, `bf16(/ 20)`, `bf16(tanh)`, `bf16(* 20)`; `logits_to_keep` slices BEFORE the head | `MuseGlimmerForConditionalGeneration.forward` | M:1172-1181 |

**The masks:** `create_causal_mask` for `full_attention`, `create_sliding_window_causal_mask` for `sliding_attention`
(M:499-502), built once per forward and picked per layer by `layer_types[i]` (M:510). The sliding overlay is
`kv_idx > q_idx - sliding_window` AND causal (MU:93-102): **2048 keys including the query**. The rotary table is
built once from `rope_parameters` (M:186-191); every non-zero `layer_rope_theta` entry of both checkpoints is 5e5 =
the global theta (below). **The cache:** `DynamicCache(config)` gives sliding layers a `DynamicSlidingWindowLayer`
that keeps only the last `sliding_window - 1` = 2047 keys after each update (CU:259-260), full layers
everything; a chunked prefill therefore sees 2047 cached keys + the chunk on a sliding layer.

**The trap list** (spec 23 §1 / §1.1, each one a test in `tools/oracle/test_muse_ref.py`, Task 3): the weight-less
embed / q-k norms and the weighted final norm use `pow` (M:110-113), the four centered norms `rsqrt` (M:128-129); q x
3.87 after the norm (M:330-331); the centered norms' eps 1e-5 / 1e-8 (M:370-373); NoPE by the zero switch (M:512);
the output gate (M:358); the plain-`w` final norm (M:116-119); the four-rounding softcap (M:1178-1181); the untied
head.

## The configuration (both repos)

| key | value (equal in both `text_config`s) |
|---|---|
| `model_type` | `muse_glimmer` / text `muse_glimmer_text`; `MuseGlimmerForConditionalGeneration` |
| widths | hidden 6656, intermediate 19968, 52 layers, 32 q / 2 kv heads x head_dim 128, vocab 202048 |
| `layer_types` | 39 sliding + 13 full; **full at 3, 7, ..., 51 (`l % 4 == 3`)** - C:152-156's default formula `(L - 1 - i) % 4 == 0` gives the same for L = 52 |
| `layer_rope_theta` | 0 (NoPE) at exactly the 13 full layers; the other 39 are 5e5 = `rope_parameters.rope_theta` (default rope) |
| `sliding_window` | 2048 |
| eps | `rms_norm_eps` 1e-5, `post_norm_eps` 1e-8 |
| head | `qk_scale_factor` 3.87, `output_multiplier` 0.19611613513818404 (= 1/√26), `final_logit_softcapping` 20.0 |
| tying | `text_config.tie_word_embeddings` **false**; the top level has no key. `get_expanded_tied_weights_keys` returns `{}` and `lm_head` stays a separate tensor after `tie_weights()` (checked on a meta-built model from each repo's config) - M:1063's `_tied_weights_keys` is never applied |
| tokens | `bos_token_id` 200000, text `eos_token_id` 200001, `pad_token_id` null; `image_token_id` 200092, `video_token_id` 200091 |
| `max_position_embeddings` | 131072 |

`config.json` differs between the repos only in `quantization_config` and: top-level `dtype` **`float16`** in the GPTQ
repo (the original: `bfloat16`), the GPTQ's added `torch_dtype float16`, `text_config.dtype` / `vision_config.dtype`
`bfloat16` (absent in the original), `transformers_version`. `from_pretrained(dtype="auto")` on the GPTQ repo would
therefore build fp16; **the reference always builds bf16** (the original's dtype; the GPTQ's non-quantised tensors are
bf16 on disk, its scales F16).

## The checkpoints (headers)

Per layer (52 x) - the original, all bf16:

| tensor | shape | bytes (52 layers) |
|---|---|---:|
| `self_attn.q_proj` / `gate_proj` | [4096, 6656] | 2,835,349,504 each |
| `self_attn.k_proj` / `v_proj` | [256, 6656] | 177,209,344 each |
| `self_attn.o_proj` | [6656, 4096] | 2,835,349,504 |
| `mlp.gate_proj` / `up_proj` | [19968, 6656] | 13,822,328,832 each |
| `mlp.down_proj` | [6656, 19968] | 13,822,328,832 |
| 4 norms (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`) | [6656] | 692,224 each |

plus `embed_tokens` and `lm_head` [202048, 6656] bf16, **2,689,662,976 B each, both shipped** (in both repos), and
`norm` [6656]. **Eight linears a layer** (q, k, v, the attention `gate_proj`, o; gate, up, down) - spec 23 §3.1 says
"the seven linears"; its §2 parameter count already has eight.

The GPTQ repo: each linear as `qweight` I32 `[K/8][N]`, `scales` F16 `[K/64][N]`, `qzeros` I32 `[K/64][N/8]`, `g_idx`
I32 `[K]` (e.g. `q_proj` `[832, 4096]` / `[104, 4096]` / `[104, 512]` / `[6656]`; `down_proj` `[2496, 6656]` / `[312,
6656]` / `[312, 832]` / `[19968]`); norms, `embed_tokens`, `lm_head`, `norm` bf16. Text names: 3 + 52 x 36 = **1875**;
the original's 3 + 52 x 12 = **627**.

**Vision (skipped by the text path): 809 tensors in both repos** - `model.vision_tower.*` 806 (3.705 GB),
`model.vision_adapter.*` 2 (83.89 MB), `model.vision_projection.*` 1 (54.53 MB); 407 `weight` + 402 `bias`. (Spec 23
§1 says "800 tensors" for the tower: it is 806; the 809 total stands.)

**`quantization_config`** (`config.json`, equal to the repo's `quantization_config.json`): `quant_method` "gptq",
`provider` "auto-round", `autoround_version` "0.15.0", `bits` 4, `group_size` 64, `sym` true, `desc_act` false,
`true_sequential` false, `lm_head` false, `iters` 800, `data_type` "int", `damp_percent` 0.01,
`static_attention_granularity` / `static_kv_granularity` "tensor"; **no `dynamic`, no `extra_config`, no
`packing_format`**. `loader::QuantConfig::parse` (`src/loader/quant.h:21-60`) accepts it: bits 4, g64, sym,
`desc_act` declared false, `quant_method` gptq - the spelling of `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`; the
`g_idx` tensors are shipped and docs/13's asserts check them.

**Sampled values (range reads):**

| linear | `qzeros` all `0x77777777` | `g_idx == i // 64` |
|---|---|---|
| layer 0 `self_attn.q_proj` ([104, 512]) | yes | yes |
| layer 30 `self_attn.o_proj` ([64, 832]) | yes | yes |
| layer 51 `mlp.down_proj` ([312, 832]) | yes | yes |

**The two Vishva007 repos are one export** (`muse_facts.py <bf16> <AutoRound snapshot> --twin <GPTQ snapshot>`,
the -GPTQ shards read by HTTP range): layer 0 `k_proj` `qweight` / `scales` / `qzeros`, layer 30 `o_proj.scales`,
layer 51 `down_proj.qzeros`, layer 30 `post_feedforward_layernorm`, the final `norm` - **7 / 7 byte-identical**
(sha256); the -AutoRound repo ships no `g_idx` (0 / 416 tensors), the -GPTQ one 416. Its `quantization_config` is
the spelling of our own Qwen3.8 / Ornith checkpoints (`quant_method` auto-round, `packing_format`
auto_round:auto_gptq), which `loader::QuantConfig::parse` reads (`src/loader/quant.h:21-60`): either repo is
accepted; `g_idx` is optional and, when shipped, must be `i // 64`.

**The full scan** on the -AutoRound repo (all 416 linears, local reads, `muse_facts.py --scan-quant`): **every
`qzeros` word is `0x77777777` - 196,591,616 B read, CLEAN** (no `g_idx` to check). The -GPTQ repo's 416 `g_idx`
tensors (13.3 MB) are scanned where it is downloaded (`muse_oracle.sh facts`, queue row 42).

## Bytes (derived from the headers; spec 23 §2 re-derived)

| | |
|---|---|
| linear parameters a layer | 483,917,824; x 52 = **25,163,726,848** (= the bf16 headers' linear bytes / 2) |
| int4 g64 (`qweight` + `scales`) | 12,581,863,424 + 786,366,464 = **13,368,229,888 B** (4.250 bit), 257.08 MB a layer |
| not loaded by the engine | `qzeros` 196,591,616 B, `g_idx` 13,312,000 B |
| norms | 2,782,208 B bf16 -> 5,564,416 B fp32 on the device |
| `lm_head` int8 + fp32 row scales (spec 9) | 1,345,639,680 B |
| `embed_tokens` bf16 | 2,689,662,976 B (one 13,312 B row a token) |
| **weights on the card** | **17,409,096,960 B (17.41 GB)**; read per decoded token **14.719 GB** |
| KV a position (13 full layers) | 13 x 2 x 128 x 2 x 2 B = **13,312 B**; 1.745 GB at 131072; rings 39 x 4096 slots 163,577,856 B |
| read per token at depth 4096 / 32768 / 131072 | 14.856 / 15.237 / 16.546 GB -> roofline at 590 GB/s 39.7 / 38.7 / 35.7 t/s |

These equal spec 23 §2's table.

## Tokens and the template

- **`tokenizer.json` is byte-identical in both repos** (sha256 `c9dbee66967b58f3...`): BPE, vocab 200000, 439802 merges,
  no normalizer, ByteLevel decoder.
- **2048 added tokens, ids 200000..202047, all special.** Named (not `reserved_special_token_N`): 200000
  `<|begin_of_text|>` (BOS), 200001 `<|end_of_text|>` (EOS), 200007 `<|eom|>`, 200008 `<|eot|>` (EOS), 200018
  `<|finetune_right_pad|>` (pad), 200022 `<|start|>`, 200023 `<|message|>`, 200080 `<|image_start|>`, 200081
  `<|image_end|>`, 200082 `<|vid_start|>`, 200083 `<|vid_end|>`, 200087 `<|vid_frame_separator|>`, 200090 `<|image|>`,
  200091 `<|video|>`, 200092 `<|patch|>`; and 201818 `<|reserved_special_token_1818|>` = the drafter's `mask_token_id`.
- **The post-processor adds BOS** (`TemplateProcessing`, single = `[<|begin_of_text|>, A]`), and the template writes
  `bos_token` (T:104). **Rendered and tokenised as `apply_chat_template(tokenize=True)` does, there is ONE BOS**
  (transformers 5.19.0, the original's template, one user message "Hello", `current_date` 2026-10-15,
  `add_generation_prompt`): 57 ids, equal to `tok(text, add_special_tokens=False)`; `tok(text)` with special tokens on
  has **two**. Rendered text:
  `<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent
  date: 2026-10-15.\n\nReasoning strength: high.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|>Hello<|eot|><|start|>assistant`;
  ids `[200000, 200022, 15651, 200023, 4662, ..., 200008, 200022, 1556, 200023, 19873, 200008, 200022, 140680]`
  (all 57 in `muse_facts.py`'s output).
- **The two templates:** the original's `chat_template.jinja` sha256 `cfc67e5f...cae678`, 9992 B, 211 lines, normalises
  "Reasoning effort" and names "Muse Glimmer" in its error text; the GPTQ repo's `114f55eb...07965e`, 7167 B, **one
  line**, no "Reasoning effort" handling, error text names "Onyx" (the same file as the tiny's). A line diff is
  meaningless against a one-line file; the behavioural difference is spec 23 §1.2's (the system branch). Neither
  repo's `tokenizer_config.json` carries a `chat_template` key: the `.jinja` file is the template. **This project
  renders with the original's** (decision 1).
- **`response_template`** (`tokenizer_config.json`, the schema `parse_response` reads, TB:3335-3345): `start_anchor`
  `<|start|>assistant`; `reasoning_content` open `to=self<\|message\|>` close `<|eom|>`; `content` open
  `to=user<\|message\|>` close `<|eot|>` or `<|eom|>`; `tool_calls` open `<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">`
  close `</atem:invoke>`, `repeats` true, `xml-inline` with tag pattern
  `<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>`, value parser `json` with
  `allow_non_json`.
- **`generation_config.json`** (equal in substance): BOS 200000, **EOS [200001, 200008]**, pad 200018, `do_sample`
  true, temperature 1.0, **top-k 64, top-p 0.95**, `max_length` 131072.

## The drafter (spec 23 §8.1, for 23f)

- `config.json`: `MuseGlimmerAssistantModel`, `model_type` `muse_glimmer_assistant`, 5 layers all
  `sliding_attention` (window 2048), hidden 6656, 32 q / **8 kv** x 128, FF 19968 SiLU, `rms_norm_eps` 1e-5, rope
  theta 5e5 (default), **`block_size` 16, `mask_token_id` 201818, `target_layer_ids` [1, 13, 25, 37, 49]** (AC:81-89 are
  the same defaults), pad 200018, saved by 5.15.0.dev0.
- Header table (one `model.safetensors`, 58 tensors, all bf16, data 5,111,970,304 B = 2,555,985,152 parameters; the
  linears alone 2,555,904,000 - spec 23 §8.2's count): `encoder.fc.weight` [6656, 33280] (the five taps concatenated),
  `encoder.output_norm_enc.weight` [6656]; per layer `self_attn.{q,o}_proj` [4096, 6656] / [6656, 4096],
  `self_attn.{k,v}_proj` [1024, 6656], `self_attn.{q,k}_norm.weight` [128] (weighted), `input_layernorm` /
  `post_attention_layernorm` [6656], `mlp.{gate,up}_proj` [19968, 6656], `mlp.down_proj` [6656, 19968]; final
  `norm.weight`. **No embedding and no head** - it embeds with the target's table WITHOUT the embed norm (G:1563-1579)
  and reads the target's `lm_head` raw on rows 1..15 (G:1658-1660).
- Its RMSNorm rounds twice: `bf16(x̂)` then `w * x̂` in bf16 (A:50-55) - unlike every norm of the target.
- The block: the anchor + 15 `mask_token_id` (G:1551, G:1633-1638); taps through `output_hidden_states=
  target_layer_ids` (G:1548; the OUTPUT of those layers, OC:112-131); after a verify `crop(-block_size)` (G:1620-1621).
  Its sliding mask inside the drafter is the bidirectional window `|q - k| <= 2048` (MU:142-152).
- The GPTQ repo and the original share `tokenizer.json` byte for byte (above), so the drafter's ids are the served
  model's.

## What this sheet changes in spec 23

- §3 the checkpoint: EITHER Vishva007 repo (the same tensors; `-AutoRound` without `g_idx` is the one the Mac
  downloaded and the reference reads; `g_idx` optional, checked `== i // 64` when present).
- §3.1 "the seven linears" -> eight (the attention `gate_proj` is the eighth; §2's count already had it).
- §1 vision "800 tensors" for the tower -> 806 (809 in all, as stated).
- §8.1 "one `model.safetensors`, 5,111,970,304 B" is the DATA size; the file is 5,111,976,608 B.
- New: the GPTQ `config.json` says `dtype float16` (the reference and the engine use bf16 for the non-quantised
  tensors, as stored); ids 200091 / 200092 embed as row 0 in transformers (M:1010-1013); a tap list ending at the
  model's last layer sees the final-normed state (OC:294-305).
