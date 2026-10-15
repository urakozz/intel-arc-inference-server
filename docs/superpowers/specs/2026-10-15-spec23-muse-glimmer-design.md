# Spec 23 - Muse-Glimmer-30B (text only) and its DFlash drafter on the B70

**Status:** design, 2026-10-15, for operator review. Open decisions are marked **(decide)** (§11). Nothing is
built. **Plans (2026-10-15):** `docs/superpowers/plans/2026-10-15-spec23{a,b,c,d,e,f}-*.md` - 23a reference, facts
and golden sets; 23b descriptor, loader and synthetic checkpoints; 23c decode; 23d prefill; 23e serving; 23f the
DFlash drafter (P0 first, then the engine). **Box queue rows 42-47** (one per plan, 23a = 42 ... 23f = 47): rows
36-40 are reserved by spec 22's plans, row 41 exists, and spec 24 (written in parallel) takes the rows after 47.

**What the operator ruled (2026-10-15):**
- **The checkpoint is `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ`, used as-is**: it is our format exactly
  (§3). Its quality is gated against a bf16 CPU reference of `meta-models/Muse-Glimmer-30B` on our A4 tool-call set;
  we re-quantise with our own calibration **only if that gate shows a gap** (the rule, §3.3).
- **The DFlash drafter `meta-models/Muse-Glimmer-30B-assistant` is in this spec** (stage 23f), **P0 first**:
  teacher-forced acceptance on the CPU, the verify cost on the box, and a stopping rule against plain decode AND
  prompt lookup (§8). The engine part reuses spec 19's DFlash design.
- **Out of scope:** the vision encoder (image / video inputs refused), int8 prefill (recorded as a later probe,
  §10), sampling-side shortcuts.

**Why this model.** A dense 27.9 B-parameter text model (25.2 B in its linears) with sliding-window attention on
three of every four layers, NoPE full attention on the fourth, 2 KV heads, a 131072 context, Harmony-style channels
with ATEM XML tool calls, and a shipped block-diffusion drafter; Apache-2.0. It is built for agentic coding on 24-32
GB cards - our target workload (long agentic coding sessions). At int4 it reads ~14.7 GB a token, about
Qwen3.8's 15.54 GB (docs/05), and it fits one card with room for the drafter.

Every number is **measured** unless marked **derived** (from the configs and the safetensors headers, which were
range-fetched: no weight shard was downloaded), **estimated** or **published** (model cards).

**Citations.** The reference is transformers **5.19.0** (the wheel's own files):
- `M:` = `transformers/models/muse_glimmer/modeling_muse_glimmer.py`; `C:` = `.../configuration_muse_glimmer.py`;
- `A:` = `transformers/models/muse_glimmer_assistant/modeling_muse_glimmer_assistant.py`; `AC:` = its configuration;
- `G:` = `transformers/generation/candidate_generator.py`; `CU:` = `transformers/cache_utils.py`; `MU:` =
  `transformers/masking_utils.py`; `OC:` = `transformers/utils/output_capturing.py`; `RP:` =
  `transformers/utils/chat_parsing/response_parser.py`; `TB:` = `transformers/tokenization_utils_base.py`;
- `T:` = the original's `chat_template.jinja` (sha256 `cfc67e5f...cae678`, 9992 B, 211 lines);
- the second implementation is vLLM at `13e919a3d3` (2026-10-15, the local checkout
  `/Users/urakozz/PycharmProjects/vllm`, Apache-2.0): `V/models/muse_glimmer.py`,
  `V/tool_parsers/muse_glimmer_tool_parser.py`, `V/reasoning/muse_glimmer_reasoning_parser.py`.

## 1. The model

From `meta-models/Muse-Glimmer-30B` at `a4e59da5` (`config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`, the index and both shard headers) and the GPTQ repo at `0987fe04`:

| | |
|---|---|
| architecture | `MuseGlimmerForConditionalGeneration`, `model_type: muse_glimmer`; text config `muse_glimmer_text` |
| layers | 52; `layer_types` `[sliding, sliding, sliding, full]` x 13 (full at `l % 4 == 3`: 13 full, 39 sliding; C:152-156) |
| hidden | 6656; vocab 202048 (200000 BPE + 2048 added, ids 200000-202047); untied `embed_tokens` / `lm_head`, bf16, 2,689,662,976 B each |
| attention | 32 q / 2 kv heads (GQA 16) x head_dim 128; q, k, v, **`gate_proj`** (6656 -> 4096) and o projections, no bias (M:295-311) |
| q / k norm | one **weight-less** RMSNorm over each head's 128 (eps `rms_norm_eps` 1e-5), then q **x `qk_scale_factor` 3.87**; scale `head_dim^-0.5` (M:291, M:312, M:330-331) |
| positions | RoPE (default, theta 5e5, NEOX halves) on sliding layers only; full layers NoPE: `position_embeddings=None` where `layer_rope_theta[i]` is 0 (M:333-336, M:512) |
| window | sliding layers see keys `q - 2048 < k <= q` (2048 keys including the query; MU:93-102 with the causal mask) |
| output gate | `attn_out x sigmoid(gate_proj(x_normed))` before `o_proj` (M:357-359) |
| norms | sandwich: `input_layernorm` -> attention -> `post_attention_layernorm` -> + residual -> `pre_feedforward_layernorm` -> MLP -> `post_feedforward_layernorm` -> + residual (M:384-406); all four `MuseGlimmerTextCenteredRMSNorm`: `x * rsqrt(mean(x²) + eps) * (1 + w)` in fp32, one cast (M:122-136); eps 1e-5 for input / pre-ff, **`post_norm_eps` 1e-8** for the post norms (M:370-373) |
| embedding | the bf16 row, then a **weight-less** RMSNorm (`MuseGlimmerRMSNorm`, eps 1e-5; M:428-436, M:451-453) - kept separate from the table because the drafter embeds without it (M:431-432, G:1563-1579) |
| FFN | dense SwiGLU, intermediate 19968, `silu(gate) x up` (M:142-155) |
| final norm | `MuseGlimmerRMSNorm` **with** weight: `x * pow(mean(x²) + eps, -0.5) * w` in fp32, one cast - plain `w`, not `1 + w` (M:101-119, M:457) |
| logits | `lm_head` (bf16) -> `x 0.19611613513818404` (`output_multiplier` = 1/√26) -> `/ 20` -> `tanh` -> `x 20` (M:1172-1181) |
| tokens | BOS 200000 `<|begin_of_text|>`; EOS {200001 `<|end_of_text|>`, 200008 `<|eot|>`} (`generation_config.json`); pad 200018; `<|eom|>` 200007, `<|start|>` 200022, `<|message|>` 200023 (`tokenizer.json`) |
| sampling defaults | `do_sample` true, T 1.0, top-k 64, top-p 0.95 (`generation_config.json`; the model card's "Best Practices") |
| context | `max_position_embeddings` 131072 ("131,072+", published) |
| vision | a 50-layer ViT (`model.vision_tower.*`, 800 tensors) + adapter / projection: 809 tensors, 3.84 GB - skipped |

**Two traps in the configuration:**
- **`layer_rope_theta` is only a switch in 5.19.0.** The rotary table is built once from `rope_parameters.rope_theta`
  (M:186-191); a layer's own value is only tested for zero (M:512). Every non-zero entry is 5e5 = the global theta,
  so this checkpoint is unaffected; 23b refuses a config whose non-zero entries differ from the global theta, naming
  the layer.
- **`_tied_weights_keys` names `lm_head` -> `embed_tokens`** (M:1063) but `tie_word_embeddings` is false in both
  checkpoints and the shards ship both tensors: they are separate weights (23a's facts sheet proves it from the
  headers).

### 1.1 The reference arithmetic (bf16, eager), rounding points

Per token, in the order the reference computes (23a turns each line into a facts-sheet row with its test):
1. **Embed:** `e = bf16(f32(row) * pow(mean(f32(row)²) + 1e-5, -0.5))` (M:110-119, `with_scale` false). Note `pow`,
   not `rsqrt` (M:112-113): the kernel's choice is held to it on the fixture (an ulp-level twin, recorded).
2. **Layer, attention half:** `xn = bf16(f32(h) * rsqrt(ms + 1e-5) * (1 + f32(w_in)))`; `q, k, v, g` = the four
   linears on `xn` (bf16 out); `q̂ = bf16(norm(q))`, `q = bf16(q̂ * 3.87f)`; `k = bf16(norm(k))` (M:330-331); on
   sliding layers RoPE with bf16 `cos / sin` (M:202): `y = bf16(bf16(y * cos) + bf16(rot(y) * sin))` (M:233-234);
   scores `bf16(bf16(q·k) * 0.08838834764831845)` + mask, softmax in fp32, probabilities `bf16`, `P·V` bf16
   (M:263-269); `a = bf16(attn * bf16(sigmoid(g)))` (M:358); `o = o_proj(a)`;
   `h = bf16(h + bf16(f32(o) * rsqrt(ms(o) + 1e-8) * (1 + w_post)))` (M:397-398).
3. **Layer, MLP half:** `xn = centered(h, 1e-5)`; `y = down(bf16(bf16(silu(gate(xn))) * up(xn)))`;
   `h = bf16(h + centered(y, 1e-8))` (M:400-404).
4. **Head:** `hn = bf16(f32(h) * pow(ms + 1e-5, -0.5) * f32(w_norm))` (M:457, M:116-119); `z = lm_head(hn)` (bf16);
   then **four bf16 roundings**: `bf16(z * 0.1961...)`, `bf16(/ 20)`, `bf16(tanh)`, `bf16(* 20)` (M:1178-1181).

**What the softcap does and does not change.** `20 tanh(z / 102)` is strictly increasing, so in exact arithmetic
greedy argmax is unaffected and only sampling (and sampled draft acceptance) sees it. In the reference it is
evaluated in bf16, so two different raw logits can round to the same capped value and `torch.argmax` then picks the
lower id. The engine argmaxes its fp32 raw logits (the device argmax is unchanged) and applies the cap on the host,
in fp32, only for sampling (§6.4). The golden gate therefore reads the reference's **raw** `lm_head` row for its
margins and treats a row whose capped top-2 tie in bf16 as undetermined (docs/14's rule, one more reason).

### 1.2 Tokens and the chat format

**The template** (T, read in full; every tag strips whitespace, so indentation contributes nothing, T:7-9):
- `bos_token` first (T:104). Without a system message the template writes its own (T:111-129):
  `<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.` + `\nCurrent date:
  <current_date | strftime_now('%Y-%m-%d')>.` + `\n\nReasoning strength: <rs>.` + (tools: `\n\n` + the tool
  definitions) + `\n\n# Valid recipients: "self", "<ns>.*"..., "user".<|eot|>`.
- A system message (T:133-155): its text with "Reasoning effort" normalised to "Reasoning strength" (four casings),
  the reasoning line appended unless the text already carries one, then the tools and the recipients, `<|eot|>`.
- `reasoning_strength` (template kwarg, any truthy string, default `high`; T:83-86). The card names low / medium /
  high / xhigh.
- User: `<|start|>user<|message|>` + content + `<|eot|>`. Content parts: `text` verbatim, `image` -> `<|patch|>`,
  `video` -> `<|video|>` (T:11-25) - the engine refuses image / video parts before rendering.
- Tool result: `<|start|>tool <name><|message|><tool_output name="<name>">\n` + content + `\n</tool_output><|eot|>`;
  a missing `name` is looked up from `tool_call_id` (T:160-178).
- Assistant (T:179-207): `reasoning_content` -> `<|start|>assistant to=self<|message|>` + it + `<|eom|>`; each tool
  call -> `<|start|>assistant to=<fn><|message|>` + ATEM + (`<|eom|>` between calls; after the last,
  `<|eom|>` if the next message is also the assistant's, else `<|eot|>`); otherwise `<|start|>assistant to=<recipient
  or user><|message|>` + content + (`<|eot|>` if the recipient is `user` or `end_turn`, else `<|eom|>`).
- **ATEM** (T:26-50): `<atem:function_calls>\n<atem:invoke name="<fn>">\n` + per argument `<atem:parameter
  name="<k>">` + value + `</atem:parameter>\n` + `</atem:invoke>\n</atem:function_calls>`; values: booleans
  `true` / `false`, None `null`, mappings and lists `tojson`, everything else as printed. `arguments` must be a
  mapping (a JSON string raises, T:28-30): the server parses a request's string arguments into an object first
  (vLLM's rule for such templates).
- The generation prompt is `<|start|>assistant` (T:209-211): **the model writes the recipient** (` to=self`,
  ` to=user`, ` to=<tool>`), so its first output tokens are a channel header.
- Tool definitions (T:51-82): a fixed preamble, `// Tool metadata` with one `{"name": <ns>, "description": ...}` line
  per namespace (`fn.name.split('.')[0]`, descriptions from the `tool_namespace_descriptions` kwarg), `// Function
  schemas` with one JSON line per tool, and a fixed example.

**The GPTQ repo ships an OLDER template** (`chat_template.jinja` sha256 `114f55eb...`, one line, 7167 B): its
system branch neither normalises "Reasoning effort" nor skips the reasoning line when the prompt has one, and its
error text names another model. The same 7167-byte file is in the tiny model `optimum-intel-internal-testing/
tiny-random-muse-glimmer`. **This spec renders with the original's file** (decision 1); the loader prints which one it
read, and `b70-serve` takes `--chat-template FILE` (23e).

**The tokenizer** (`tokenizer.json`, sha256 `c9dbee66...` - identical in both repos): BPE, 200000 vocab, 439802
merges, no normaliser, a split regex with `\p{M}` classes (the class spec 21e implemented for Qwen3.8-Flash-Next),
ByteLevel. **Its post-processor adds BOS** (`TemplateProcessing`), and the template writes `bos_token` too: the server
tokenises a rendered prompt **without** special tokens (transformers' `apply_chat_template(tokenize=True)` does the
same) - 23e's test pins one BOS. 2048 added tokens 200000-202047, all special; `mask_token_id` 201818 is
`<|reserved_special_token_1818|>` (the drafter's mask).

**The output format** - `tokenizer_config.json` carries a `response_template` (the "response schema" transformers
5.19.0 parses with `tokenizer.parse_response`, TB:3340-3420, RP): `start_anchor` `<|start|>assistant`; fields
`reasoning_content` (open `to=self<|message|>`, close `<|eom|>`), `content` (open `to=user<|message|>`, close
`<|eot|>` or `<|eom|>`), `tool_calls` (open `<atem:invoke ... name="(?P<name>...)">`, close `</atem:invoke>`, repeats,
`xml-inline` parameters, each value JSON-decoded when it parses and the raw string otherwise). **That is the
parser's reference** on complete turns (§6.5).

## 2. What it costs on the card (derived)

| part | parameters | bytes | read per decoded token |
|---|---:|---:|---:|
| 52 layers' linears, int4 g64 (`qweight` + f16 `scales`: 4.25 bit) | 25,163,726,848 | 13.368 GB | 13.368 GB |
| norms (4 per layer + final, fp32 on device) | | 5.6 MB | 5.6 MB |
| `lm_head`, int8 + fp32 row scales (spec 9, the default) | 1,344,831,488 | 1.346 GB | 1.346 GB |
| `embed_tokens`, bf16 (gathered: one 13 KB row a token) | 1,344,831,488 | 2.690 GB | 13 KB |
| **weights on the card** | | **17.41 GB** | **14.72 GB** |

- **Per layer:** q 6656x4096, k / v 6656x256, gate 6656x4096, o 4096x6656, gate / up 6656x19968, down 19968x6656 =
  483,917,824 parameters, 257.08 MB at int4 g64.
- **KV:** only the 13 full layers grow - 13 x 2 kv heads x 128 x (K, V) x 2 B = **13,312 B a position**, 1.745 GB at
  131072. The 39 sliding layers keep a ring; at 4096 slots (Kolibri's rule: a prefill chunk of 2048 + the window
  - 1 must not collide, §5.2) **4 MiB a layer, 163.6 MB** in all.
- **Read per token at depth d:** 14.72 GB + 13,312 B x d (full layers) + up to 81.8 MB (39 windows of 2048 rows):
  14.86 GB at 4096, 15.24 GB at 32768, **16.55 GB at 131072**.
- **Decode roofline** at 590 GB/s (docs/05): **39.7 t/s at 4k, 35.7 at 128k**; at the 77.6 % of roofline Qwen3.8
  reaches (docs/05: 26.34 / 33.96 ms) ~**30.8 / 27.7 t/s** (estimated). Qwen3.8 is 15.54 GB: the two models are
  within 5 %.
- **One card holds it:** weights 17.41 GB + KV 1.745 GB + rings 0.16 GB + the int4 drafter 1.36 GB (§8) + the
  decode and prefill scratch (~1.0-1.5 GB, estimated from Kolibri's 0.905 GB at hidden 2560 scaled by width) ≈
  **21.7-22.2 GB** against 31.0 GB usable (32.53 GB with the 1.5 GB reserve). **One card is the default**; `--pp 2`
  is supported through spec 16's pieces as Kolibri's (decision 7).

## 3. The checkpoint

### 3.1 What it is (verified on the headers, 2026-10-15)

`Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ` at `0987fe04`: 6 shards, 2684 tensors, 22,803,933,184 B (text
18.96 GB, vision 3.84 GB):
- `quantization_config`: `quant_method` "gptq", `provider` "auto-round", `autoround_version` 0.15.0, bits 4,
  group_size 64, sym true, `desc_act` false, `true_sequential` false, `lm_head` false, iters 800 (the README: 512
  samples, seqlen 2048, `quant_nontext_module` false). No `dynamic`, no `extra_config`, no `packing_format`. This is
  the spelling `loader::QuantConfig::parse` already reads for `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`
  (`src/loader/quant.h:21-33`): accepted as is.
- Per layer the seven linears as `qweight` I32 `[K/8][N]`, `scales` F16 `[K/64][N]`, `qzeros` I32 `[K/64][N/8]`,
  `g_idx` I32 `[K]`; norms bf16; `embed_tokens`, `lm_head`, final `norm` bf16.
- **Sampled values (range reads of a few KB, not the shards):** `qzeros` = `0x77777777` everywhere in layer 0's
  `q_proj` and layer 51's `down_proj`; `g_idx` = `i // 64` in those two and layer 30's `o_proj`. 23b's loader checks
  every tensor (docs/13's three data asserts) and 23a's facts sheet records the full scan.
- So: **our format exactly** - AutoRound int4 g64 symmetric, `w = (q - 8) x scale`. No new kernel, no conversion.

### 3.2 The bf16 original

`meta-models/Muse-Glimmer-30B` at `a4e59da5`: 2 shards, 59,553,253,376 B (text 55.71 GB). Same names as the GPTQ
text tensors with `.weight` for the linears. It is the quality reference (§3.3) and runs layer-streamed on the box
CPU (23a); it is never served.

### 3.3 The quality rule

The checkpoint is served as is **unless** 23a's gate Q (on the box CPU, both references layer-streamed) shows a gap:
- **Q-A4:** the 36 A4 scenarios rendered by Muse's template (23a Task 5), 192 greedy ids each, through the int4
  checkpoint dequantised (the engine-format reference) and through the bf16 original. Bars **(PROPOSED, decision
  5):** the int4 reference has **no more parse failures** than bf16's; its **first-call agreement with bf16 >= 25 / 36**
  (Qwen3.8's measured int4-vs-bf16 A4 agreement, docs/BENCHMARKS.md, the bar this project already accepted).
- **Q-KL:** teacher-forced on the bf16 reference's A4 continuations and on golden code / prose: per-row KL(bf16 ||
  int4) over the capped distribution at T 1.0 and top-1 agreement. Bars (PROPOSED): **mean KL <= 0.02, top-1 >= 95
  %** on A4; recorded, not gated, on prose.
- **If Q fails:** re-quantise from the bf16 original with our own calibration (Kolibri's §3.1 recipe adapted: agentic
  coding and tool calls rendered through Muse's template, AutoRound at the pinned commit, `--group_size 64` sym,
  `lm_head` and vision out) - a stage 23q added then, not planned now. **The rule is recorded:** third-party
  checkpoints in our exact format are used as-is when they pass gate Q; otherwise we quantise.

## 4. Reference and gates

### 4.1 The CPU reference (23a)

`tools/oracle/muse_ref.py`: **transformers 5.19.0's own `MuseGlimmerForConditionalGeneration`** text path, built on
`meta`, layer-streamed through `tools/oracle/stream.py` (Ornith's / Qwen4Exp's arrangement), the int4 linears
dequantised on demand (`(q - 8) x scale`), `attn_implementation="eager"`, run with spec 21a's 5.19.0 site beside the
oracle image (`tools/oracle/qwen4exp_env.sh` - the same pins; `agnes-ref-img` keeps 5.15.0). The vision tower is never
materialised. A recorder keeps per layer the residual after the layer (`H.L<l>`), the attention and MLP outputs,
and **the five drafter taps** (the residual after layers 1, 13, 25, 37, 49, §8.1); raw and capped logits.

**The tiny model** (decision of this spec, checked 2026-10-15): `optimum-intel-internal-testing/tiny-random-muse-
glimmer` at `cb70af19` (27.1 MB, bf16, 4 layers `[s, s, s, f]`, theta 5e5 with a NoPE fourth, hidden 64, 4 q / 2 kv
heads x 16, FF 256, vocab 202048) is closest; `hf-tiny-v2/tiny-random-MuseGlimmerForConditionalGeneration` (51.9 MB,
fp32, 2 layers `[full, sliding]` with theta 1e4) is not the model's pattern. The chosen tiny has **tied** embeddings
(no `lm_head` tensor) and the old template, so 23a's `muse_make_tiny.py` writes variants: untied (an independent
`lm_head`), 8 layers (two full), GQA 16 (16 q / 1 kv), a small sliding window (16) so window crossings are cheap to
test, int4-packed linears (g64, `qzeros` 0x77777777), and a tiny assistant (§8) on top.

### 4.2 Gates

- **MG0, nothing moves.** Every existing kernel binary sha256-identical; `tools/kernel_cmdlines` additions only; every
  suite green.
- **MG1, the reference.** `muse_ref.py` equals transformers 5.19.0 un-streamed on the tiny variants **bitwise** (bf16
  eager and fp32; prompt and cached decode; prompts crossing the window), with one test per trap of §1 / §1.1 (the
  weight-less embed norm with `pow`, q x 3.87 after the norm, the centered norms' eps 1e-5 / 1e-8, NoPE by the zero
  switch, the output gate, the plain-`w` final norm, the four-rounding softcap, untied head). The bf16 original's and
  the int4 checkpoint's perplexity sane on the box CPU (a wrong port is an order of magnitude off).
- **MG2, kernels.** Every new kernel against its host twin, bitwise where the arithmetic is the same (norms, the
  sandwich, q / k prep, RoPE, the gate, the embed norm); flash attention against fp64 at cosine >= 0.99999 and its
  eager variant against the reference's bf16 chain.
- **MG3, golden** (greedy, tie-aware, docs/14's undetermined-row rule plus §1.1's capped-tie rule) against the
  engine-format reference: synthetic real-width checkpoints (23b), then the real checkpoint (it fits: no truncation is
  needed; `--layers N` exists for development only). Prompts short (< 2048, every window covers the prompt) and long
  (4k, 8k, 32k, an agentic session: the window and the ring active).
- **MG4, determinism.** Replay bitwise; prefill chunks / splits at multiples of 64 bitwise (Kolibri's absolute-block
  rule, §5.2); prefix-cache restores at block ends bitwise the cold run; `--pp 2` bitwise `--pp 1`.
- **MG5, quality.** Gate Q (§3.3, the checkpoint); the engine's A4 against the bf16 reference (0 parse failures
  beyond the reference's; agreement recorded); passkey 3 / 3 at 131072 (one card, int8 head).
- **MG6, the drafter** (§8): P0's stopping rule; then spec 19's D0-D5 at Muse's shapes.
- **Speed (recorded):** decode and prefill rows against §2's roofline; llama-benchy rows (23e).

## 5. The engine

### 5.1 Structure (Kolibri's pattern, spec 20 §11)

- **Descriptor and loader:** `model::MuseGlimmerDesc` (`src/model/muse_glimmer.*`), `loader::load_muse_glimmer`
  (`src/loader/muse_*`), held to `config.json` key by key; the vision tensors skipped by name (809, counted), the
  GPTQ linears read through `LinearSrc::classify` (the existing path), `embed_tokens` / `lm_head` bf16 with spec 9's
  int8 head built at load.
- **Engine:** `runtime::muse::MuseEngine` (`src/runtime/muse/`): decode list, prefill walk, snapshots, the drafter
  (23f), one card or two through spec 16's placement.
- **Kernels:** family-only kernels in `src/kernels/muse/mg_*.cl` (host half `src/kernels/muse_kernels.h`); shared
  kernels (`gemv.cl`, `gemv_i8w.cl`, `gemv_bf16.cl`, `argmax.cl`, the prefill slabs and GEMM) at Muse's shapes from new
  CMake lines only; **no existing `.cl` edited** (MG0) - where an existing kernel needs a new define, the variant is a
  copy under `src/kernels/muse/` (spec 21c's fallback).
- **CLI:** `b70-decode` / `b70-serve` dispatch on `model_type muse_glimmer` before the Qwen rules (Kolibri's
  `is_kolibri` pattern).

### 5.2 Decode (23c)

Per layer, on Kolibri's arrangement with Muse's chains:
1. `mg_norm` (input_layernorm, centered, eps 1e-5) over the residual's Σ² from the previous launch;
2. **one** int4 GEMV `q || gate || k || v` (6656 -> 4096 + 4096 + 256 + 256 = 8704 rows);
3. `mg_attn_prep`: weight-less q / k norms, q x 3.87 with its two roundings, RoPE on sliding layers only, K / V
   written to the ring slot `p & 4095` (sliding) or row `p` (full);
4. attention over the ring (window 2048) or the full cache (NoPE), GQA 16 (2 kv heads x 16 q): split-K partials and a
   reduce that applies `bf16(attn x bf16(sigmoid(gate)))` (§1.1);
5. o_proj GEMV, the sub-block's own row and its Σ² (`prep_res_fold _Z`'s role), `mg_post_add` (centered post norm
   eps 1e-8, residual add, the new residual's Σ²);
6. `mg_norm` (pre-ff), int4 GEMV `gate || up` (39936 rows), SiLU x up, down GEMV, its row and Σ², `mg_post_add`.

Plus the embedding (gather + the weight-less norm, `mg_embed`) and the head (`mg_final_norm` with plain `w`, the int8
or bf16 `lm_head`, argmax x 2). **~14 launches a layer, ~735 a token** (derived; 23c asserts its formula). The
**ring is 4096 slots** for the same reason as Kolibri's (spec 20 §11): a prefill chunk of 2048 plus the 2047 keys
before it never collide, and 64-key blocks never cross the ring's end. Kernels that already serve Kolibri
(`kol_attn.cl`: `Q_HEADS`, `KV_HEADS`, `WINDOW`, `RING` defines, M <= 4, no gate) are reused at Muse's defines only if
a new CMake line suffices; the gate and M up to 16 (§8) need `mg_attn.cl` (a copy carrying the v2 structure).

### 5.3 Prefill (23d)

The `l0` backend only, chunks <= 2048: the linears as dequant slabs + `pf_gemm_T0` (Qwen3.8's / Kolibri's path at K
6656 and K 19968), `mg_prep` at M 2048, **windowed flash** over the ring for sliding layers (absolute 64-key blocks
from `lo & ~63`, a wholly masked first tile leaves `(m, l, o)` unchanged - Kolibri's rule, which makes splits at
multiples of 64 bitwise) and causal NoPE flash for full layers, both at GQA 16 / head_dim 128 with the gate applied
after. `--prefill-backend l0-int8` and `sycl-tla` are refused (§10). Two cards: Kolibri's sequential hand-off per
chunk (spec 20 §12). Prefill compute: 2 x 25.16 B = **50.3 GFLOP a token** in the linears (derived), about Qwen3.8's.

### 5.4 Snapshots and the prefix cache (23e)

Spec 7's store with Kolibri's split (spec 20 §13):
- **State at position p:** the 39 sliding rings' rows `[p - 2047, p)` - what the next query's window reads - **39 x
  2047 x 1024 B = 81,747,968 B** (positions below 0 zero rows); with the drafter, its context rows (§8.4).
- **Blocks:** the 13 full layers' KV, **13,312 B a position** (27.3 MB a 2048-block).
- One host layout for `--pp 1` and `--pp 2` (16b's rule).

### 5.5 Two cards

Spec 16's placement by bytes (`runtime::pp_balance`), the `copy` / `peer` hand-off of the residual (6656 x 2 B + the
norm sums), the bounded fence, `--pipeline-split auto|N`. At batch 1 two stages run in sequence: no decode speed-up;
the use is capacity (decision 7).

## 6. Serving (23e)

### 6.1 The request path

`ChatFormat::Kind::Muse` (`server/chat_format.h`): the template kwargs reach the template (`reasoning_strength`,
`current_date`, `knowledge_cutoff`, `tool_namespace_descriptions`); content parts joined (text) and image / video parts
refused (400 naming the vision encoder); a tool call's `arguments` given as a JSON string parsed into an object before
rendering; EOS {200001, 200008} - **`<|eom|>` (200007) is not EOS**: a turn holds several messages.

### 6.2 Template rendering

`chat::Template` renders `chat_template.jinja` byte-identical to transformers 5.19.0's `apply_chat_template` on a set
of message lists (no system / system with and without a reasoning line, each "Reasoning effort" casing, tools with
dotted and bare names, namespace descriptions, call history with parallel calls, every value type in ATEM -
booleans, null, numbers, strings with newlines and quotes, lists, objects -, tool results by `name` and by
`tool_call_id`, `reasoning_content` before calls, `recipient` / `end_turn` overrides, consecutive assistant messages).
`strftime_now` is provided (the host's local date, as transformers'); tests pass `current_date` so renders are fixed.

### 6.3 Reasoning strength

`chat_template_kwargs.reasoning_strength`, and the OpenAI field `reasoning_effort` mapped onto it; values low /
medium / high / xhigh only (400 otherwise), default the template's `high` (decision 3).

### 6.4 Sampling and the softcap

Host-side, before `filter_probs` (`server/spec_accept.h`): `z' = 20 tanh(z x 0.19611613513818404 / 20)` in fp32 on the
head's fp32 row, then temperature / top-k / top-p. Defaults from `generation_config.json` when a request names none:
**sampled, T 1.0, top-k 64, top-p 0.95** (Kolibri's precedent, decision 4). Greedy: the device argmax on the raw row.
The cap in fp32 against the reference's four bf16 roundings is a documented deviation that moves probabilities by at
most a few bf16 ulps of the capped value.

### 6.5 The output parser

`server::MuseOutputStream` (`server/toolcall_muse.{h,cc}`), segmenting the output into messages (vLLM's selection
rule, `V/tool_parsers/muse_glimmer_tool_parser.py:_iter_messages`):
- a header is `[<|start|>assistant] [ to=<recipient>]<|message|>` (the first message has no `<|start|>`: the prompt
  ended with it); a body ends at `<|eom|>` / `<|eot|>`, at the next header, or at a `<|start|>` (a header still being
  generated);
- `to=self` bodies are reasoning, joined with `"\n"` (decision 2: transformers' `parse_response` keeps only the LAST
  block, RP:352-376 - no `join`, no `repeats` on that field; vLLM's reasoning parser joins with `"\n"`);
- `to=user` (or no recipient) bodies are content; `<atem:invoke>` echoed inside reasoning or content is never a call;
- every other recipient is a tool channel: each complete `<atem:invoke name="...">...</atem:invoke>` is a call;
  parameters by `<atem:parameter name="k">v</atem:parameter>`, the value JSON-decoded when it parses, else the raw
  string; then cast by the tool's schema as `parse_response(tools=...)` does (RP:378-396);
- a doubled name `ns.ns` maps to a registered bare `ns` (the template renders a bare name's recipient as `"ns.*"`
  and the model writes `ns.ns`; `V/tool_parsers/muse_glimmer_tool_parser.py:_normalize_name`); any other unknown name
  passes through;
- a truncated call (no `</atem:invoke>`) is not a call; streamed frames hold back any tail that may still be a marker
  or a ` to=` header.
- **The gate:** on every complete turn of 23e's corpus the parse equals transformers 5.19.0's `parse_response(text,
  prefix=prompt, tools=...)` - except the reasoning join (decision 2) - and every split into 1-7 byte pieces streams
  to the same result.

## 7. Correctness of the fp32 deviations (recorded, not hidden)

As Kolibri's (spec 20 §11): flash keeps fp32 probabilities against the reference's bf16 eager chain (an eager variant
`B70_MUSE_ATTN=eager` reproduces the reference's points for A/B, spec 18 §10.1's rule); sigmoid / SiLU through
`exp_torch` (spec 21c's Sleef-matching `exp`), the embed / final norms' `pow(x, -0.5)` matched on the fixture; the
softcap in fp32 (§6.4).

## 8. The DFlash drafter (23f)

### 8.1 What it is (verified)

`meta-models/Muse-Glimmer-30B-assistant` at `e8192f3a` (one `model.safetensors`, 5,111,970,304 B, bf16;
`MuseGlimmerAssistantModel`, `model_type: muse_glimmer_assistant`; no mask embedding file):
- `encoder.fc` 33280 -> 6656 (the five taps concatenated) + `encoder.output_norm_enc` (A:293-309);
- 5 layers, all sliding 2048, hidden 6656, 32 q / 8 kv x 128, `q_norm` / `k_norm` with weights, FF 19968 SiLU, pre-norm
  (`input_layernorm`, `post_attention_layernorm`, no sandwich, **no output gate, no qk scale**), RoPE theta 5e5
  (A:135-269); its RMSNorm rounds `x̂` to bf16 before `x w` - two roundings (A:50-55), unlike the target's;
- final `norm`; no embedding, no head: it embeds with the target's table **without** the weight-less norm
  (G:1563-1579) and reads the target's `lm_head` (G:1658-1660);
- `block_size` 16, `mask_token_id` 201818, `target_layer_ids` [1, 13, 25, 37, 49] (AC:81-89).

**Semantics** (transformers 5.19.0's `DFlashTokenCandidateGenerator`, G:1505-1688):
- **Taps:** with `output_hidden_states` a list, `hidden_states[i]` is the OUTPUT of decoder layer i (0-based), no
  embedding prepended (OC:112-131) - the residual after layers 1, 13, 25, 37, 49 (spec 19's "after layer i"
  convention, `dflash_ref.py`'s `resid.L{i}`).
- **Context:** for every committed position j, `c_j = norm_enc(fc(concat taps_j))` (A:405), then each layer's `k_proj`
  / `v_proj`, `k_norm`, RoPE at j (A:177-192) into that layer's cache.
- **Block:** 16 rows at positions p .. p + 15: row 0 the anchor (the newest committed token, the target's last sample,
  not yet through the target), rows 1-15 `mask_token_id` (G:1551, G:1633-1638); queries only from the block, keys from
  the context plus the block.
- **Attention:** bidirectional inside the block; window **`|q - k| <= 2048`** (MU:142-152) - row r (absolute p + r)
  sees context `[p + r - 2048, p - 1]` and all 16 block rows. vLLM's DFlash uses `<= window - 1` (dflash_ref.py's
  reading of `flash_attn.py`): one key apart at the window's edge; this spec follows transformers (decision 13).
- **Drafts:** `lm_head` on rows 1..15 (G:1658-1660) - **15 drafts per block, so a verify is at most M = 16 rows**
  (the anchor + 15). The operator's brief said "up to 17 (block 16 + 1)": the reference's block already includes the
  anchor (decision 10). Greedy: argmax per row.
- **Raw logits:** the drafter's rows get **neither the multiplier nor the softcap** (G:1658-1688): sampled drafting in
  transformers uses `softmax(raw / T)` - about 5x sharper than the target's capped distribution (decision 11).
- **Commit:** after a verify the accepted rows (anchor + accepted drafts) append their taps to the context; the block's
  K / V are dropped (`crop(-block_size)`, G:1620-1621; CU:296-323).

### 8.2 Bytes (derived)

| | bf16 | int8 per-channel | int4 g64 |
|---|---:|---:|---:|
| drafter weights (2,555,904,000 parameters) | 5.112 GB | 2.556 GB | 1.358 GB |
| ring: 5 layers x 8 kv x 128 x (K, V) x 2 B = 20,480 B a position, 2048 positions | 41.9 MB | | |
| taps of a verify row (5 x 6656 bf16) | 66,560 B | | |

One block reads the drafter once at M = 16 rows plus the int8 head once for 15 rows: **~2.7 GB with the int4
drafter, ~0.18 plain steps** at roofline (estimated: 4.6 ms against ~32 ms) - flat in K.

### 8.3 P0 first (23f Tasks 1-5)

1. **The drafter reference:** `dflash_ref.py` gains Muse's form (the semantics above; window `<= 2048`; the raw
   embedding; raw head rows; the drafter's two-rounding norm), held **bitwise** to transformers 5.19.0's
   `MuseGlimmerAssistantModel` + `DFlashTokenCandidateGenerator` on a tiny random assistant over the tiny target.
2. **Taps** from 23a's recorder on [prompt + the engine-format reference's own greedy continuation] (the int4
   checkpoint dequantised: what we serve), box CPU: the 36 Muse-rendered A4 scenarios (§3.3's continuations), golden
   code and prose, and an agentic coding session.
3. **Teacher-forced acceptance** E_K for K = 1..15 per corpus, arms: drafter bf16, int8 per-channel RTN, int4 g64 RTN
   (our format; AutoRound only if RTN loses), and - optional, decision 12 - `incoai/Muse-Glimmer-30B-DFlash2` (a
   DFlash 2 drafter for this target whose config carries the multiplier and softcap; `dflash_ref.py` already runs
   DFlash 2). Prompt lookup's E_K on the same continuations through spec 19e's matcher.
4. **Verify cost on the box:** Muse's decode GEMVs at M = 1..16 on random weights of the real shapes (52 layers'
   int4 linears allocated, 13.4 GB, plus the int8 head) - `gemv.cl` at M = 9..16 as new variants if they compile, else
   two M = 8 passes, both measured - and one drafter block at M = 16 (bf16 / int8 / int4); interleaved pairs after a warm-up,
   median of 3 (sequential arms mis-state ratios under clock drift).
5. **The projection and the stopping rule:** tokens / s at the best K per corpus. **Go only if DFlash beats plain
   decode by >= 10 % AND prompt lookup by >= 10 % on the coding and agentic sets** (Muse has no MTP head); otherwise
   stop and record, and 23f's engine half does not start.

### 8.4 The engine half (23f Tasks 6-9, after a "go")

Spec 19's design (§5 there; plan 19b / 19c as written) at Muse's shapes:
- the drafter on the card holding `lm_head`; its linears in the P0-chosen format through `gemv.cl` / `gemv_i8w.cl` at
  M = 16; `mg_attn` with a ring mode (2048 slots, the drafter's window) and a bidirectional block term; the context
  projection kernel (fc -> norm -> per-layer k / v -> k_norm -> RoPE -> ring);
- the verify list = 23c's decode list at M = K + 1 <= 16 (M = 9..16 variants), bitwise M = 1 per row (D3), writing
  the five taps of every row;
- the commit list projects rows 0..j into the ring (the count read from Control: captured once);
- host acceptance = spec 8's `spec_accept` (greedy prefix; sampled with q from the drafter's rows - decision 11);
- prefill writes the taps of the last 2048 prompt positions and fills the ring;
- the prefix cache stores the drafter's **fc-output rows** of the last 2048 positions with the state (13,312 B a
  position, 27.3 MB; the ring is re-projected on restore - spec 19 decision 5's proposal);
- `b70-serve --spec dflash --draft-model PATH`, `--spec-max K` (1..15), the adaptive-K cost row flat in K;
  `--spec lookup` beside it (19e's matcher through the same verify lists).

## 9. Stages and what needs the box

| stage | Mac (blind) | box | queue row |
|---|---|---|---|
| **23a** reference, facts, golden | the 5.19.0 site, MG1 on the tiny variants, the facts sheet from source + small files, the A4 set rendered, the golden prompts | perplexity (bf16 + int4), golden sets, gate Q (both A4 references), the tap dumps | **42** (CPU) |
| **23b** descriptor, loader, synth | descriptor, loader host half, `make_synth.py` real-width N layers, the plan | loads (synthetic, the real 18.96 GB) | **43** |
| **23c** decode | kernels' host twins, Mac-GPU indicative runs, syntax | every first compile, MG2 / MG3 / MG4 on decode, two cards, speed | **44** |
| **23d** prefill | twins, names, the walk's launch formula | MG2-MG4 on prefill, speed | **45** |
| **23e** serving | template byte-identity, tokenizer equality, parser = `parse_response`, snapshot layouts | restores, the server, MG5 (A4, passkey), llama-benchy | **46** |
| **23f** DFlash | P0 Task 1 on the tiny; the engine's twins | P0 Tasks 2-5 (CPU taps + acceptance, GPU verify cost); D1-D5 | **47** |

**Order:** 23a -> 23b -> 23c -> 23d -> 23e; 23f's P0 needs only 23a (and its GPU probe only `gemv.cl`), so it runs
as soon as 23a's tap dumps exist; 23f's engine half needs 23c-23e and P0's "go".

## 10. Out of scope and recorded for later

- **The vision encoder** (809 tensors skipped; image / video parts refused by name).
- **int8 prefill (`l0-int8`):** its h8 linears rotate in whole 1024-column Hadamard blocks (docs/17); 6656 = 6.5 x
  1024 and 19968 = 19.5 x 1024. A later probe: a 512-wide or mixed 1024 + 512 block rotation (K 6656 = 6 x 1024 +
  512), measured for quality on gate Q's sets first.
- **int8 KV** (spec 12's form at 2 kv heads; 13 layers x 13,312 B -> 6,656 B a position): later.
- Sampling-side shortcuts; tree verification; third-party quant formats (GGUF, NVFP4, MLX): one format, ours.
- Contexts beyond 131072.

## 11. Decisions

1. **(decide) The chat template:** the original's `chat_template.jinja` (`cfc67e5f...`) for every checkpoint, not
   the GPTQ repo's older one (`114f55eb...`). **Recommended:** the original's (it is the newer, it handles a system
   prompt that already states the reasoning strength); `--chat-template FILE` overrides.
2. **(decide) Several `to=self` blocks in one turn:** joined with `"\n"` (vLLM) or the last one only (transformers'
   `parse_response`). **Recommended:** join - nothing the model wrote is dropped.
3. **(decide) Reasoning strength in the API:** `chat_template_kwargs.reasoning_strength` and the top-level
   `reasoning_effort` mapped onto it; low / medium / high / xhigh only. **Recommended:** as stated, default `high`.
4. **(decide) Sampling defaults:** `generation_config.json`'s (sampled, T 1.0, top-k 64, top-p 0.95, the softcap on
   the host) or greedy-unless-asked. **Recommended:** `generation_config.json`'s (Kolibri's precedent).
5. **(decide) Gate Q's bars:** no more A4 parse failures than bf16, first-call agreement >= 25 / 36, mean KL <= 0.02
   and top-1 >= 95 % on A4 rows. **Recommended:** these, PROPOSED until 23a prints the distributions.
6. **(decide) Context cap:** 131072 (`max_position_embeddings`). **Recommended:** 131072; above it refused.
7. **(decide) Placement:** one card by default, `--pp 2` optional. **Recommended:** one card.
8. **KV:** bf16 now, int8 later. Ruled with this spec unless the operator objects.
9. **(decide) The drafter's format:** P0 measures bf16 / int8 / int4 g64. **Recommended:** int4 g64 RTN if its E_K
   is within 0.05 of bf16's at the chosen K (spec 19a measured no loss for int8 / 4-bit on Qwen3.8's drafter).
10. **(decide) The verify width:** M <= 16 (the reference's 15 drafts + the anchor), not 17. **Recommended:** sweep
    M = 1..16; K per request from the cost table.
11. **(decide) The drafter's q for sampled acceptance:** raw rows (transformers) or with the target's multiplier and
    softcap. **Recommended:** P0 measures both; default the capped form (closer to p, lossless either way).
12. **(decide) `incoai/Muse-Glimmer-30B-DFlash2` as a P0 arm.** **Recommended:** yes, as a secondary arm (cheap:
    `dflash_ref.py` runs DFlash 2 already); the engine ships one drafter.
13. **(decide) The drafter window edge:** transformers' `|q - k| <= 2048` or vLLM's `<= 2047`. **Recommended:**
    transformers' (the pinned reference); the difference is one key at the window's edge and touches acceptance only.
14. **(decide) Who runs the bf16 reference:** the box CPU (55.7 GB of text weights, layer-streamed, ~128 GB RAM) -
    **recommended** - or a rented machine.
