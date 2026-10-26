# Spec 23 - Muse-Glimmer-30B (text only) and its DFlash drafter on the B70

> DROPPED (operator, 2026-10-27): support removed from the engine in ef3f1d7; kept for the record.
>
> Reusable later (revive from git: `git show 4c491f8:<path>` - 4c491f8, this branch's parent on main, is the last
> commit that has all of it):
> - dense attention: `src/kernels/muse/mg_attn.cl` (flash decode, sliding window over a ring or full NoPE, output gate), `src/kernels/muse/mg_attn_eager.cl`, prefill attention `src/kernels/muse/mg_pf_attn.cl` (DPAS flash + eager twin) - all three still live, byte-identical, in `src/kernels/bolt/` (Bolt binds them); their twins `tests/kernels/muse_ref.h`, `tests/kernels/muse_pf_ref.h` (the full-layer half lives on as `tests/kernels/mg_attn_ref.h`)
> - the between-GEMV kernels `src/kernels/muse/mg_prep.cl` (weight-less / centered norms, sandwich post-add, q / k prep, SiLU x up), `src/kernels/muse/mg_pf_prep.cl`, `src/kernels/muse/mg_argmax.cl`; host names `src/kernels/muse_kernels.h`; the GEMV cells at its dense shapes in `src/kernels/CMakeLists.txt`'s Spec 23c / 23d blocks
> - the dense descriptor / layout / loader: `src/model/muse_glimmer.h`, `src/model/muse_glimmer.cc`, `src/loader/muse_layout.h`, `src/loader/muse_loader.h`, `src/loader/muse_loader.cc`, `src/loader/muse_repack.h`, `src/loader/muse_repack.cc`; the engine `src/runtime/muse/` (buffers, capture, prefill, sizes, RoPE)
> - tests: `tests/kernels/muse_ref_test.cc`, `tests/kernels/muse_pf_ref_test.cc`, `tests/kernels/muse_kernels_test.cc`, `tests/kernels/muse_pf_kernels_test.cc`, `tests/loader/muse_layout_test.cc`, `tests/loader/muse_repack_test.cc`, `tests/loader/muse_load_checkpoint_test.cc`, `tests/loader/muse_synth_host_test.cc`, `tests/model/muse_glimmer_test.cc` (+ `tests/model/muse_glimmer/`), the oracle `tools/oracle/muse_ref.py`; `tests/CMakeLists.txt`'s Spec 23b-23e blocks
> - the DFlash v1 drafter in its Muse form: `src/kernels/muse/mg_dflash.cl`, `src/runtime/dflash/muse_drafter.h`, `src/runtime/dflash/muse_drafter.cc`, the Muse form of `src/model/dflash.h` / `src/model/dflash.cc`, `src/runtime/dflash/dflash_capture.cc` and `src/kernels/dflash_kernels.h`; int4 g64 drafter linears (`DraftFormat::Int4G64`, `src/loader/dflash_repack.cc`)
> - tests: `tests/runtime/muse_dflash_test.cc`, `tests/loader/muse_assistant_test.cc`, `tests/kernels/muse_dflash_names_test.cc`, the Muse form of `tests/kernels/dflash_ref.h` and `tools/oracle/dflash_ref.py`

**Status:** design, 2026-10-15, for operator review. Open decisions are marked **(decide)** (§11). Nothing is
built. **Plans (2026-10-15):** `docs/superpowers/plans/2026-10-15-spec23{a,b,c,d,e,f}-*.md` - 23a reference, facts
and golden sets; 23b descriptor, loader and synthetic checkpoints; 23c decode; 23d prefill; 23e serving; 23f the
DFlash drafter (P0 first, then the engine). **Box queue rows 42-47** (one per plan, 23a = 42 ... 23f = 47): rows
36-40 are reserved by spec 22's plans, row 41 exists, and spec 24 (written in parallel) takes the rows after 47.

**What the operator ruled (2026-10-15):**
- **The checkpoint is `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ`, used as-is**: it is our format exactly
  (§3). Its quality is gated against a bf16 CPU reference of `meta-models/Muse-Glimmer-30B` on our A4 tool-call set;
  we re-quantise with our own calibration **only if that gate shows a gap** (the rule, §3.3). *Superseded the same
  day: the quant is kept and gate Q is a recorded measurement (§3.3's ruling); either Vishva007 spelling of the
  export is read (§3.1, 23a).*
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
  `V/tool_parsers/muse_glimmer_tool_parser.py`, `V/reasoning/muse_glimmer_reasoning_parser.py`; for the parser's
  edge cases also OpenVINO Model Server at `5d8bdfb19` (`/Users/urakozz/PycharmProjects/model_server`, Apache-2.0,
  ideas only) and intel/auto-round at `da52d6da` for the checkpoint's recipe.

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
| vision | a 50-layer ViT (`model.vision_tower.*`, 806 tensors - 23a's count) + adapter / projection: 809 tensors, 3.84 GB - skipped |

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

**Either Vishva007 repo** (23a, 2026-10-15: one export in two spellings, sampled tensors byte-identical -
`docs/probe-muse-glimmer-2026-10-15.md`): `Vishva007/Muse-Glimmer-30B-W4A16-AutoRound` at `ad7b9432` (`quant_method`
"auto-round", `packing_format` "auto_round:auto_gptq" - our own checkpoints' spelling -, no `g_idx`, 2268 tensors,
22,790,907,104 B; **the repo the Mac downloaded and the reference reads**) or the GPTQ spelling below (`g_idx`
shipped). `g_idx` is optional; when present it must be `i // 64`.

`Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ` at `0987fe04`: 6 shards, 2684 tensors, 22,803,933,184 B (text
18.96 GB, vision 3.84 GB):
- `quantization_config`: `quant_method` "gptq", `provider` "auto-round", `autoround_version` 0.15.0, bits 4,
  group_size 64, sym true, `desc_act` false, `true_sequential` false, `lm_head` false, iters 800 (the README: 512
  samples, seqlen 2048, `quant_nontext_module` false). No `dynamic`, no `extra_config`, no `packing_format`. This is
  the spelling `loader::QuantConfig::parse` already reads for `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`
  (`src/loader/quant.h:21-33`): accepted as is.
- Per layer the eight linears (q, k, v, the attention `gate_proj`, o; gate, up, down - 23a) as `qweight` I32 `[K/8][N]`, `scales` F16 `[K/64][N]`, `qzeros` I32 `[K/64][N/8]`,
  `g_idx` I32 `[K]`; norms bf16; `embed_tokens`, `lm_head`, final `norm` bf16.
- **Sampled values (range reads of a few KB, not the shards):** `qzeros` = `0x77777777` everywhere in layer 0's
  `q_proj` and layer 51's `down_proj`; `g_idx` = `i // 64` in those two and layer 30's `o_proj`. 23b's loader checks
  every tensor (docs/13's three data asserts) and 23a's facts sheet records the full scan.
- So: **our format exactly** - AutoRound int4 g64 symmetric, `w = (q - 8) x scale`. No new kernel, no conversion.
- **The recipe that made it** (intel/auto-round commit `da52d6da`, 2026-10-15, author Vishva R - the checkpoint's
  author; `examples/Muse-Glimmer/Muse-Glimmer_W4A16.ipynb`, read at the local checkout
  `/Users/urakozz/PycharmProjects/auto-round`): `AutoModelForImageTextToText` in bf16; `SignRoundConfig(iters=800,
  group_size=64, sym=True)`; `AutoRound(scheme="W4A16", nsamples=512, batch_size=4, seqlen=2048, processor=processor,
  quant_nontext_module=False, enable_torch_compile=True)` with **no `dataset` argument**; `quantize_and_save(format=
  "auto_round,auto_gptq")` (the GPTQ repo is the second export). With no dataset AutoRound calibrates on its default
  **`NeelNanda/pile-10k`** (`auto_round/compressors/base.py:261` at `da52d6da`; `:239` at the `v0.15.0` tag, the
  checkpoint's `autoround_version`), and with `quant_nontext_module` false and `muse_glimmer` not in
  `NOT_SUPPORT_ONLY_TEXT_MODELS` (`auto_round/special_model_handler.py:62`) the multimodal path keeps that text set
  (`auto_round/calibration/mllm.py:90-120`). **The calibration is generic web text (the Pile), 512 x 2048 tokens - no
  chat template, no reasoning channels, no ATEM tool calls.** That is why gate Q (§3.3) is the deciding gate: our
  workload is exactly the distribution the calibration did not see.

### 3.2 The bf16 original

`meta-models/Muse-Glimmer-30B` at `a4e59da5`: 2 shards, 59,553,253,376 B (text 55.71 GB). Same names as the GPTQ
text tensors with `.weight` for the linears. It is the quality reference (§3.3) and runs layer-streamed on the box
CPU (23a); it is never served.

### 3.3 The quality rule

**Ruled 2026-10-15 (operator): Vishva007's quant is kept.** The operator has re-quantised with this recipe on
coding data before and saw no reason to repeat it here. Gate Q still runs (23a), as a **recorded measurement, not a
decision**: it documents the int4 checkpoint's distance from bf16 on A4 / KL; a re-quantisation is not on the plan.

The checkpoint is served as is **unless** 23a's gate Q (on the box CPU, both references layer-streamed) shows a gap.
The operator's ruling stands (use it as-is first); gate Q exists because its calibration was the Pile (§3.1), not
agentic coding - if quantisation error concentrates anywhere, it is on the chat-template / channel / ATEM tokens our
workload is made of, and A4 measures exactly those:
- **Q-A4:** the 36 A4 scenarios rendered by Muse's template (23a Task 5), 512 greedy ids each (23a: plan 23a's count - the replies open with a
  `to=self` channel; 192 was this spec's first figure), through the int4
  checkpoint dequantised (the engine-format reference) and through the bf16 original. Bars **(PROPOSED, decision
  5):** the int4 reference has **no more parse failures** than bf16's; its **first-call agreement with bf16 >= 25 / 36**
  (Qwen3.8's measured int4-vs-bf16 A4 agreement, docs/BENCHMARKS.md, the bar this project already accepted).
- **Q-KL:** teacher-forced on the bf16 reference's A4 continuations and on golden code / prose: per-row KL(bf16 ||
  int4) over the capped distribution at T 1.0 and top-1 agreement. Bars (PROPOSED): **mean KL <= 0.02, top-1 >= 95
  %** on A4; recorded, not gated, on prose.
- **If Q fails:** re-quantise from the bf16 original with our own calibration (Kolibri's §3.1 recipe adapted: agentic
  coding sessions and tool calls rendered through Muse's template with reasoning channels and ATEM calls, preferably
  the bf16 model's own outputs, AutoRound at the pinned commit, `--group_size 64` sym, `lm_head` and vision out;
  the same 800 iterations so the only change is the calibration set) - a stage 23q added then, not planned now. **The rule is recorded:** third-party
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
or bf16 `lm_head`, argmax x 2). **~14 launches a layer, ~733 a token** (derived; 23c asserts its formula). The
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
  2047 x 1024 B = 81,748,992 B** (positions below 0 zero rows; 23e: the first edition said 81,747,968 - a mistyped product); with the drafter, its context rows (§8.4).
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
- the call's name is read from `<atem:invoke name="...">`, never from the channel's `to=` recipient (an Apache-2.0
  server's Muse parser - OpenVINO Model Server's `onyx` parser, "Onyx" being the name the GPTQ repo's older template
  still carries - observes the model writing the BARE tool name in `to=`, `src/llm/io_processing/onyx/
  onyx_tool_parser.hpp:47-51` at `5d8bdfb19`, while vLLM observes `ns.ns`: the invoke name plus the collapse covers
  both);
- **argument values are typed by the tool's schema first** (decision 15): a string-typed parameter keeps its exact
  text; integer / number / boolean / null / object / array are decoded; with no schema, JSON-decode else the raw
  text. The three references differ here: transformers JSON-decodes first (`allow_non_json` returns the STRIPPED
  text, `content_parsers.py:46-88`) and casts only values still strings (RP:42-64, RP:378-396), so a string-typed
  `"007"` becomes `7`; vLLM JSON-decodes else keeps the raw text; the `onyx` parser types by schema
  (`onyx_tool_parser.hpp:53-56`) - our Qwen XML `convert_value`'s rule (`server/toolcall.h`);
- **a call cut by `max_tokens` or a stop string** (decision 16): never an error; the unterminated ATEM block is
  removed from content (no raw markup reaches the client) and no partial call is emitted - an agent would execute
  a truncated edit - with `finish_reason` `length`. (OVMS commit `503ff8662`, `onyx_tool_parser.cpp:211-249`,
  instead synthesises the missing closing tags and emits the partial call; vLLM drops it.)
- streamed frames hold back any tail that may still be a marker (`<|` - the framing idea of OVMS's gpt-oss parser,
  `src/llm/io_processing/gptoss/content_parser.cpp:29-32, 110` at `db2f05cff`) or a growing ` to=` header; the
  first message's header has no `<|start|>assistant` (`content_parser.cpp:56-59` lists both header forms for
  gpt-oss's channels).
- **The gate:** on every complete turn of 23e's corpus the parse equals transformers 5.19.0's `parse_response(text,
  prefix=prompt, tools=...)` - except the reasoning join (decision 2) and the listed value-typing cases (decision
  15) - and every split into 1-7 byte pieces streams to the same result.
- **Two server facts checked:** `usage.prompt_tokens_details.cached_tokens` is already reported (spec 7 / plan 7c:
  `server/openai.cc:64-70`, `server/server.cc:184`), so a prefix-cache hit on Muse shows there (23e asserts it);
  `generation_config.json`'s `stop_strings` is not read today - Muse's has none, but 23e reads it as a string OR a
  list (HF allows both; openvino.genai `d54d3768` fixed the string form in its reader) and merges it with a
  request's `stop`.

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
5. **The projection** (ruled 2026-10-15: the engine half is built whatever P0 shows - the operator's "we need to
   support it"; P0 picks the drafter format, K_max and the q form, and sets the expectation). The superseded rule:
   tokens / s at the best K per corpus. **Go only if DFlash beats plain
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

### 8.5 P0 verdict (2026-10-16; the Mac, PROVISIONAL - the box's r47.accept / r47.cost decide)

Recorded, not applied (the operator's 2026-10-15 ruling: the engine half is built whatever P0 shows). Numbers:
`docs/probe-muse-dflash-2026-10-16.md`. The Mac measured teacher-forced acceptance on the only two continuations it has
(23a's `muse_short_code` / `muse_short_prose`, 32 greedy ids each, 18 anchors each) and projected with the DERIVED cost
table (`tools/spec/muse_p0_project.py`: verify M = 1..4 Qwen3.8's int8-head rows, +0.305 a row to M = 16; the block
read once - int4 g64 0.18, int8 0.26, bf16 0.43 plain steps):

| arm | code: best K, E_K, x plain | prose | lookup (n 3) code / prose |
|---|---|---|---|
| int4 g64 + int8 head | K 8, 6.67, **1.91x** | K 1, 1.56, 1.14x | 1.09x / 1.00x |
| int8 + int8 head | K 9, 7.33, 1.89x | K 1, 1.56, 1.08x | |
| bf16 (bf16 head) | K 9, 7.22, 1.78x | K 1, 1.50, 0.93x | |

- **The rule cannot be decided on the Mac**: its agentic sets (A4 + the session) are the box's (`r47.taps` -> `r47.accept`).
  On code DFlash beats plain by 1.9x and lookup by 1.7x - the rule's coding half holds by a wide margin on 18 anchors.
- **The provisional defaults** (each `b70-serve` flag overrides; P0's box run must confirm):
  - **the drafter's format: int4 g64 RTN** (decision 9: with the int8 head it serves, its E_K equals bf16's within 0.02 at
    every K on both corpora; 1.36 GB against 5.11);
  - **K_max 15** (the block) with AdaptiveK choosing K per request (best K 8-9 on code, 1 on prose at the derived costs);
  - **q raw** (decision 11 recommended the capped form; the measurement reverses it: sampled E_15 on code 5.58 raw vs
    1.35 capped - the drafter's raw rows are what tracks the target's capped p; capping them again flattens q ~5x).
- **What the box must confirm**: E_K on A4 and the agentic session per arm (decision 9's 0.05 rule at the chosen K), the
  verify cost at M = 1..16 and the two-pass form (r47.cost), the projection with the measured table, and the q form on the
  sampled estimate.

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

**Ruled (operator, 2026-10-15): all 16 decisions below as recommended.** Each one's recommendation is the ruling; 23a starts.

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
15. **(decide) Tool-call argument typing:** schema first (a string parameter keeps its text; our Qwen XML rule and the
    `onyx` parser's) or transformers' JSON-first `parse_response`. **Recommended:** schema first - `"007"`, `"true"`,
    `"null"` or `" x "` typed as strings must reach the tool unchanged; the gate lists these as the known differences.
16. **(decide) A call cut by `max_tokens` / a stop string:** dropped and stripped from content with `finish_reason`
    `length` (vLLM-like) or emitted with its tags synthesised (OVMS-like). **Recommended:** dropped - a partial edit
    or command must never reach an agent as if complete.

## 12. 23a as built (2026-10-15; the Mac - nothing on the box yet)

Branch `spec23a-muse-reference` (plan 23a); box queue row 42 holds what the Mac could not run. Facts:
`docs/probe-muse-glimmer-2026-10-15.md`.

- **The version:** transformers **5.19.0** (spec 21a's site beside `agnes-ref-img`, unchanged) loads the tiny model
  `optimum-intel-internal-testing/tiny-random-muse-glimmer` @ `cb70af19` as shipped (`muse_tiny_check.py`: tiny
  check OK). Cached decode vs one uncached forward: bf16 bitwise, fp32 max |diff| 6.9e-6 (matmul shapes; the plan
  expected fp32 bitwise - MG1's comparisons are at equal shapes and ARE bitwise).
- **MG1 on the tiny variants (`test_muse_ref.py`, 16 tests, all pass):** the streamed port = `from_pretrained`
  un-streamed BITWISE on `untied`, `l8`, `gqa16`, `win16` (bf16 and fp32; 40 / 100 ids across window 16 + 8 cached
  decode steps; raw and capped logits, every `H.L`), `l8` at 2100 ids with the real window, chunks of 32, `--layers
  4`, the int4 variant = transformers on its dequantised twin, `layer_major` = sequential, the taps = transformers'
  `output_hidden_states=[1, 5]`; one test per trap of §1.1. **MG1 on real weights (Mac, measured):**
  `hfcheck --layers 4` on 2100 ids of `muse_4k` + 4 steps, the int4 checkpoint: streamed = resident = layer-major
  **BITWISE** (447 s streamed, 2062 s resident with another agent's container beside it).
- **Departures from the plan:** (1) the int4 checkpoint is read in **either Vishva007 spelling** (§3.1): the Mac's
  is `-AutoRound` (no `g_idx`); `expected_names` has a third form `autoround` (1459 text names; `gptq` 1875, `bf16`
  627, vision 809 in each). (2) `run` chunks the prefill above 2048 ids (not 4096): a 4096-row head OOMs an 8 GB
  container; chunked = transformers chunked, bitwise. (3) Q-A4 runs **512** new ids (§3.3 said 192). (4) Gate Q is a
  **recorded measurement** (the operator's ruling): `muse_gateq.py verdict` prints WITHIN / OUTSIDE the proposed bars,
  exit 0. (5) `muse_gateq.py kl` takes `--bf16 <snapshot>` and caches the bf16 rows beside the run. (6) The
  agentic golden prompt is 13,416 ids (plan: ~12k); `muse_32k` needs `long.txt` appended (Muse's BPE encodes
  long32k's text in 29,946 ids).
- **Facts that change this spec** (edited above): eight linears a layer (not seven); the vision tower is 806
  tensors (809 in all); the GPTQ `config.json` says `dtype float16` at the top level (the reference uses bf16 -
  the stored dtype of every non-quantised tensor); ids 200091 / 200092 embed as row 0 in transformers (M:1010-1013);
  `output_hidden_states` with the model's LAST layer in the list returns the final-normed state there
  (OC:294-305 - the real taps end at 49 of 52, unaffected); on torch's CPU `pow(x, -0.5)`, `rsqrt(x)` and
  `1 / sqrt(x)` are the same fp32 value (10^7 draws) - the norms' `pow` / `rsqrt` forms are indistinguishable there,
  and the device kernels must reproduce the correctly rounded `1 / sqrt`; transformers 5.19.0 reaches the DFlash
  generator only with `generate(..., speculation_type="dflash")` (23f).
- **The full quant scan** (the -AutoRound repo, local): every `qzeros` word `0x77777777` in all 416 linears (196.6 MB)
  - CLEAN; sampled tensors of the two repos byte-identical (7 / 7). The -GPTQ repo's `g_idx` scan is the box's.
- **Golden sets made on the Mac** (the -AutoRound checkpoint dequantised, `muse_oracle.sh . golden`, `--gen 32
  --taps`, 8 threads, 40 GB cap, measured): `oracle-out-muse/muse_short_prose` - 42 ids, prompt perplexity
  **21.48**, cap_tie 0 / 74 rows, raw top-1 / top-2 gap min 0.5 / median 19, wall 2065 s, peak RSS 17.7 GiB, the
  continuation " For now, the water held its breath.\n\nI had come to the island..." (coherent);
  `muse_short_code` - 53 ids, perplexity **1.70**, cap_tie 0 / 85, gap min 1 / median 83.75, wall 1954 s, RSS 17.7
  GiB. The files are 176 / 202 MB (taps of every row, every `H.L`, raw / capped logits). A prompt forward of ~50 ids took
  ~400 s and every decode step ~51-54 s (the 52 layers re-dequantised each step) on the Mac: `muse_4k` (~3.5 h),
  `muse_8k` (~7 h), `muse_agentic` (~11 h) and `muse_32k` (~30 h) are ESTIMATED from these and left to row 42
  (`r42.golden` keeps the two finished prompts when `--push-data` carries them). The perplexity sanity of §4.2 MG1
  holds on the two short prompts (< 30); `r42.ppl` adds the 4k text and both forms.
- **Gate Q:** not run on the Mac - ESTIMATED ~90 h a form (36 scenarios x [a ~1.8k-id prompt at ~53 ms a layer-token
  + 512 decode steps at ~51 s, each re-dequantising the 52 layers]) against the 8-hour budget; prepared for the box
  (row 42: `r42.a4`, `r42.gateq`). Its numbers will be recorded here (the checkpoint is kept either way).

## 13. 23b as built (2026-10-15; the Mac - nothing on a card yet)

Branch `spec23b-muse-loader` (plan 23b); box queue row 43 holds what the Mac could not run (every upload, the card's
read-backs, two cards). No kernel added or changed (`kernel_cmdlines` +0 / -0 / ~0).

- **The descriptor** `model::MuseGlimmerDesc` (`src/model/muse_glimmer.*`) = the vendored `config.json` (meta-models @
  `a4e59da5`, `tests/model/muse_glimmer/`, byte for byte) field by field, and Vishva007's `-AutoRound` `config.json`
  reads as the same descriptor (its `dtype float16` and `quantization_config` are not structure). `check_muse_config`
  refuses 29 in-memory edits, each naming the key and its value - incl. both halves of §1's `layer_rope_theta`
  trap (a non-zero entry != the global theta; a zero at a sliding layer, a non-zero at a full one, each naming the
  layer), `num_key_value_heads` 4, `final_logit_softcapping` null, `hidden_activation` gelu, tying at either level.
- **The bytes** (`loader/muse_layout.h`, the planner's and the loader's one formula) are §2's exactly: 257,081,344 B of
  int4 a layer (no layout padding at these shapes), 257,187,840 B with the norms, 17,409,096,960 B on one card with
  the int8 head (1,345,639,680 B), read per token 14,719,433,984 B. The int4 layouts' formulas moved to
  `loader/int4_layout.h` (shared with `moe_layout.h` / `kolibri1_layout.h`, byte-identical; Kolibri's tests green).
- **The GEMVs:** all four int4 layout 0 with PROVISIONAL S (q||gate||k||v S2, o_proj S4, gate||up S8, down S4 -
  Qwen3.8's nearest measured cells; 23c sweeps). `gate || up` is CONCATENATED (up at column 19968; spec §5.2's
  separate SiLU x up), not Qwen's interleave16. The norms are the bf16 `w` widened (the kernels add 1 in fp32).
- **The host half on the real checkpoint (measured, Mac):** the -AutoRound repo through the served constructor -
  2268 tensors, 1459 text (= `muse_expected_names(autoround)`), **809 vision skipped by name**, every `qzeros` word
  0x77777777 (`assert_quant_invariants`), **957 subnormal f16 scales** (counted, decoded exactly), all 52 layers
  repacked (bytes = the layout, the part boundaries' words and scales = the shards), 0 unconsumed - 76 s, 15.5 GB RSS
  (the mmap). The -GPTQ spelling's 1875 names are tested on fixtures and the synthetic checkpoint only.
- **Synthetic checkpoints:** `tools/quantize/muse/make_synth.py --layers 2` wrote 5.90 GB in 54 s (agnes-ref-img, 8
  threads; the plan's ~5.9 GB); `check.py` ACCEPTED it (75 names = `muse_ref.expected_names(gptq)`, every linear
  within RTN's bound of its regenerated twin, worst 0.500 of a scale) and `muse_synth_host_test` read it with 0
  unconsumed. `test_muse_quant.py` 3 / 3 (six refusals by name). 23a's tiny int4 variant (73 vision tensors, one
  un-indexed `model.safetensors`) read by `muse_repack_test` through a temporary index.
- **The plan** (`runtime::muse`): one card at 131072 with the int8 head 20.818 GB incl. the 1.5 GB reserve; with the
  1.358 GB int4 drafter and 1.5 GB of scratch placeholders 23.676 GB; bf16 head + 5.112 GB bf16 drafter 27.274 GB -
  131072 in each (decision 6). Two cards by bytes split at 24 (11.243 / 11.075 GB); a 20 GB card reaches 69632; a
  16 GB card is refused naming the bytes; 262144 is refused naming decision 6.
- **Departures from the plan:** (1) `load_muse_glimmer(devices, snapshot, max_len, placement, int8_head,
  layers_override, reserve)` - Kolibri's / qwen4exp's vector of devices plus `max_len` (the plan before the first
  allocation needs it) and the scratch / drafter placeholders. (2) `MuseHostLayer` holds each GEMV as words + scales
  (layout 0's two allocations) rather than one byte vector. (3) `plan_muse` takes an optional `split` (`--pp 2` with a
  split given). (4) The drafter's reservation sits on the last device (the head it reads). (5) `muse_oracle.sh` has
  a second new mode, `synth-tests`. (6) Task 5 was built before Task 3's device half (the load calls `require_fits`).

## 14. 23c as built (2026-10-15; the Mac - no Muse binary compiled, nothing on a card)

Branch `spec23c-muse-decode` (plan 23c); box queue row 44 holds what the Mac could not run (every ocloc build, MG2 on the
card, MG3 / MG4 on decode, two cards, speed). MG0 on the Mac: kernel_cmdlines **+19 / -0 / ~0** (no existing `.cl` edited).

- **The host twin** `tests/kernels/muse_ref.h` (`museref`) is **bitwise** transformers 5.19.0's own modules at the REAL
  widths (`tools/oracle/muse_fixture.py` runs `MuseGlimmerTextNormedEmbedding`, the centered / weight-less / plain-w
  norms, the decoder layer's post norm + add, the attention's `qk_norm` x `qk_scale_factor`, the rotary module +
  `apply_rotary_pos_emb`, `eager_attention_forward`, the gate's and the MLP's lines in the oracle image; 508 KB of
  fixture): every chain 0 values differ, the fine rows included; eager attention's scores, probabilities AND outputs 0
  differ at 5 / 2047 / 2048 / 4097 sliding and 3000 full; q / k prep RoPE'd at 0, 1, 2047, 2048, 4096, 9000, 100000,
  131071. The host RoPE table (`runtime::muse::rope_table`, fp32 `pow`, double `cos` / `sin` rounded to bf16 - Kolibri's
  rule at theta 5e5) equals the rotary module's bf16 rows bit for bit at those positions. Review Focus 1 / 2 / 5 are
  tests: Kolibri's two-rounding norm and a plain-w norm differ from torch; a one-rounding q x 3.87 differs; a NoPE row
  is position-free. The fp32 softcap is within 2 bf16 ulps of the four roundings (26 / 64 values move one or two).
- **Eager against fp64 is 0.99994**, not 0.99999 (2048+ keys: the bf16 chain rounds scores and probabilities); the
  0.99999 bar is flash's. Flash's own block / wave / merge arithmetic (`museref::attention_flash_sim`, mg_attn's
  structure on the host) is within fp64 cosine 0.99999 at 5 .. 9000 sliding and 3000 / 20000 full (worst 0.9999945),
  and the query at 9000 equals the one at 4904 bitwise (the same ring rows: the plan's "808" was Kolibri's 513-key
  arithmetic; at a 2048-key window the pair is 9000 / 4904).
- **The kernels** (`src/kernels/muse/`, `kernels::muse`): `mg_prep.cl` (`mg_embed`, `mg_norm` + `mg_final_norm` in one
  binary, `mg_post_add`, `mg_attn_prep`, `mg_silu_mul`), `mg_attn.cl` (a COPY of `kol_attn.cl`: GQA 16, TGT 64 - a
  2048-key window is 34 absolute 64-key blocks, more than the project's TGT 32 -, the gate fused into the reduce, M <= 16
  as a define; `kol_attn.cl` was not reused: M <= 4 and no gate), `mg_attn_eager.cl` (the reference's chain and the gate
  in ONE launch). Σ² in 26 chunks of 256 at hidden 6656. `prep_silu_mul` was not reused (Qwen's interleave16 and plain
  `exp`; 23b's gate || up is concatenated). The gate is stored as `bf16(sigmoid(g))` by `mg_attn_prep` (the reference
  rounds the sigmoid before the product). `mg_embed` maps the image / video placeholders (200092 / 200091) to row 0, as
  M:1010-1013 do. The four GEMVs keep 23b's PROVISIONAL S2 / S4 / S8 / S4 L0 with Qwen3.8's nearest cells' `GEMV_*`
  defines (new `add_gemv_variant` rows); the 957 subnormal f16 scales need nothing (denormals are never flushed).
- **The Mac's GPU** (UHD 630, `muse_run`, indicative, emulated sub-groups): every `mg_prep` kernel and `mg_attn_eager`
  (sliding 5 / 2046 / 2047 / 2048 / 4095 / 4096 / 4097 / 4904 / 9000, full 3000; 9000 == 4904) **exact** against the
  twin - no value moved even with Apple's 1 / sqrt. `mg_attn.cl` cannot run there (shuffles, reductions, 2D reads).
- **The engine** (`runtime::muse`): 14 launches a layer (13 eager), **733 a token at 52 layers (681 eager; +2 with the
  peer hand-off)**, asserted at capture. The decode scratch is **70,374,528 B at 131072** (+16,777,216 B eager's score
  row), 67.1 MB of it the RoPE table (23b's plan carried no RoPE term: it is in `MuseReserve::decode_scratch` now). One
  card at 131072 with it reserved still fits (`muse_plan_test`). Empty KV / ring groups are not allocated, so the
  allocation equals the planner's bytes exactly. The engine plans at 131072 while `--max-len` is auto, then re-plans at
  the length chosen.
- **The CLI**: `b70-decode` dispatches `muse_glimmer` before the Qwen rules (either Vishva007 spelling resolves); the
  int8 head is Muse's default; refused before the device by name: `--prefill*` (23d), `--mtp`, `--spec` (23f; now
  parsed by b70-decode, still an unknown option elsewhere), `--kv-cache int8` (decision 8), `--max-len` above 131072
  (decision 6) or off the 256 grid, `--profile`, `--device` with `--pp 2`. `b70-serve` refuses the model naming 23e.
- **The synthetic golden mode** (`muse_oracle.sh synth`, the checkpoint mounted as `ORACLE_SNAP`) ran on the Mac's
  2-layer synthetic: `muse_short_prose` in 50 s, cap_tie 1 / 74 rows, raw top-1 - top-2 gap **==0 in 5 rows, median
  0.17** - random weights give SMALL margins: MG3 on the synthetic leans on the undetermined-row rule, and the int8
  head (spec 9) may fail determined rows there that the bf16 head passes (read r44.golden's two arms together).
  `muse_oracle.sh real-layers N` adds the truncated reference the partial forward's head is graded against.
- **Departures from the plan**: the fixture is at the real widths (not a hidden-256 tiny); eager-vs-fp64's bar is 0.9999
  and the outputs' bar 1 ulp (measured 0); no `tests/kernels/CMakeLists.txt` or `tools/mac/clrun/CMakeLists.txt` exists
  (tests in `tests/CMakeLists.txt`, `muse_run` in `mac_check.sh`'s loop); a card test `muse_kernels_test` (MG2) was
  added for r44.k1; the speed arms are stages (`r44.speed`, `r44.speed_long`), not a `bench_decode.sh` arm; there is no
  `gpu` ctest label (card tests are `muse` / `checkpoint;golden;muse`); `r44.reject` holds the GPU lock (Kolibri's
  r24.reject: a refusal that regressed would open the device); the two-card path was built with the engine (Task 4),
  Task 5 adds its tests; the bench prompt is 23a's `muse_short_prose.ids`.

## 15. 23d as built (2026-10-16; the Mac - no Muse prefill binary compiled, nothing on a card)

Branch `spec23d-muse-prefill` (plan 23d); box queue row 45 holds what the Mac could not run (every ocloc build, MG2 on
the card, MG3 / MG4 on prefill, two cards, speed). MG0 on the Mac: kernel_cmdlines **+16 / -0 / ~0** (no existing `.cl`
edited); 23c's decode list unchanged (`muse_pf_variant_names_test` pins its 17 names; its launch formula, 733 / 681,
re-asserted in `muse_plan_test`).

- **The host twins** `tests/kernels/muse_pf_ref.h` (`musepf`, on 23c's `museref`): the window over the ring - a chunk of
  2048 at 2048 / 4096 / 6143 reads keys `[c0 - 2047, c0 + 2047]`, **4095 pairwise-distinct slots of 4096** (Review Focus
  1, the first and last rows named); the fp64 tile walk (64-key tiles from `lo(p0) & ~63`, `m_safe`) equals the direct
  softmax over per-POSITION keys to **5.8e-16** at c0 0 / 1000 / 2048 / 4096 / 6143 and full depths 3000 / 20000; a row
  whose first tile holds none of its keys (row 15 of the chunk at 2096: tile [0, 64)) gives bitwise the walk from the
  next tile - and dropping the guard makes it NaN (shown by mutation); rows reversed or permuted inside their 8-row
  groups bitwise; a row's result is the same in chunks starting 2048 / 4032 / 4096 / 2112 bitwise - and at **2049 too**
  (the twin's tiles are absolute, so any split is bitwise in exact rules; the card's GEMMs are not the twin - Review
  Focus 2's "close" bar stays for the card); flash's fp32 arithmetic (bf16 P, natural exp) against fp64 worst
  **0.9999954** (bar 0.99999) through the ring and full to 30000.
- **The kernels** (`src/kernels/muse/`, `kernels::muse` "spec 23d" block): `mg_pf_attn.cl` - one entry,
  `mg_pf_flash_attn`, two builds: **flash** (EAGER 0) is `kol_pf_attn.cl`'s DPAS default path at GQA 16 (HPW 4, RPW 8,
  KT 64, 256 GRF; grid (ceil(C / 8), 2, 4)), unchanged in structure; **eager** (EAGER 1, `B70_MUSE_ATTN=eager`) is NOT
  Kolibri's DPAS-with-rounding-points but a portable plain-OpenCL kernel (grid (32, C), WG 128) that walks a row's keys
  in 128-key tiles three times (max; torch's 8 lanes; P·V in `eager_block(L)` blocks), recomputing each score - so it is
  **bitwise decode's eager row** (`museref::attention_eager_head`, = `mg_attn_eager` at M 1) with no score scratch.
  `mg_pf_prep.cl`: `mg_pf_embed` (mg_embed's chain over the chunk's ids buffer: mg_embed reads `Control::cur_token`,
  M <= 8) and **`mg_pf_attn_gate`** - the gate is a separate launch after either attention build (decode's chain
  `rne(rne(o) x bf16(sigmoid(g)))`), not folded into the flash epilogue: portable, shared by both forms. Reused at new
  CMake lines: `mg_prep.cl` at M 2048 (mg_norm / mg_final_norm, mg_post_add, mg_attn_prep at S 1 both kinds,
  mg_silu_mul at S 1), `k2_pf_dequant_slab` at 6656x8704 (8 x 1024 + 512), 4096x6656 (6 x 1024 + 512), 6656x39936 (39 x
  1024), 19968x6656, `pf_res_fold_K6656_SP1_G26_Z`, the shape-free `pf_gemm_T0`. Every partials pitch (8704, 6656,
  39936) is a whole 256, so mg_attn_prep and mg_silu_mul read the GEMM's rows at S 1 unchanged.
- **The Mac's GPU** (UHD 630, `muse_run`, indicative): the slabs at K 6656 / 19968, `mg_pf_embed` (the placeholder's row
  0, sums at the 2048 pitch), mg_norm / `_Z` + mg_post_add / mg_attn_prep (3 rows across the ring's wrap 4095, 0, 1; NoPE
  at 37) / mg_silu_mul at M 2048, the gate - **exact** against the twin. The eager build over a sliding chunk of 24 rows
  at 4090 and a full chunk of 12 at 2990: **bitwise decode's `mg_attn_eager` rows on the same device**; against the host
  twin 8 of 98,304 values move one bf16 ulp (Apple's `1 / sum`, not correctly rounded - decode's move identically).
  The flash build cannot run there (DPAS, 2D block reads, sub-group reductions): its card test is `muse_pf_kernels_test`
  (r45.k1).
- **The walk** (`runtime/muse/muse_prefill.{h,cc}`, archive `b70_muse_prefill`): **134 launches a layer** (mg_norm,
  q||gate||k||v 9 x 2, mg_attn_prep, the attention, the gate, o_proj 7 x 2, `_Z` + mg_post_add, mg_norm, gate||up 39 x 2,
  mg_silu_mul, down 7 x 2, `_Z` + mg_post_add) + mg_pf_embed on device 0: **6969 a chunk at 52 layers**, the same under
  eager and on two cards; the head 4 (mg_final_norm from the M 2048 norm binary at grid (26, 1) on the last row's
  offsets - decode's chain, no new binary -, lm_head, argmax x 2). Asserted at the walk (`prefill_device_launches`).
- **The scratch** (`runtime::muse::prefill_sizes`, one constant a device: nothing depends on max_len): **599,171,072 B**
  - partials [2048][39936] fp32 (327 MB, gate||up's rows) the largest, the slab [19968][1024] bf16 (40.9 MB), h
  [2048][19968] (81.8 MB), attn_q / gate / attn_out, resid / x / a, the sums; + the 27.3 MB landing on two cards
  (`pf_landing_layout`). One card at 131072 with it, the int8 head, the decode scratch and the int4 drafter: **22.845 GB
  of 32.53** (`muse_plan_test`). `MuseOptions::prefill` plans it (`run_reserve`); the CLI sets it for `--prefill` /
  `--prefill-length`.
- **The engine** (`muse_prefill_engine.cc`, Kolibri's arrangement): `prefill(ids, chunk)` (chunk <= 2048), lazy
  `prepare_prefill` naming a missing binary, `set_prefill_replay` (B70_PREFILL_REPLAY), the block hook with Kolibri's
  `(end_pos, is_block_end)` signature (23e), two cards by the sequential copy hand-off with the bounded wait, P4's
  `drop_next_handoff`, the Control mirror. `b70-decode <muse> --prefill`, `--bench --prefill-length N [--prefill-chunk C]`
  print the prefill line and a pp row (t/s, and TFLOP/s against 2 x 52 x 483,917,824 = 50.3 GFLOP a token); refused by
  name: `--prefill-backend l0-int8` (6656 = 6.5 x 1024, §10), `sycl-tla` (no Muse walk), `--prefill-chunk` above 2048
  (23c's `cli_reject_muse_prefill` / `_len` removed).
- **Departures from the plan**: (1) the gate is its own launch (above); (2) prefill eager is the portable bitwise build,
  not DPAS-with-rounding-points; (3) the flash's "portable path" check is the host's (`flash_window_sim`) and the card's,
  not the Mac GPU's; (4) `PrefillSizes` carries `h` and no `logits` (the head writes decode's), `prefill_sizes` takes no
  max_len; (5) `set_block_hook` is Kolibri's two-argument hook; (6) no `gpu` ctest label (23c's ruling): the card tests
  are `checkpoint;muse;prefill` (+ `pp`); (7) `muse_pf_kernels_test` (r45.k1) was written in Task 2; (8) the two-card
  hand-off landed with Task 3's `MuseEngine::prefill` (one function), Task 4 is its tests; (9) "in-tile permutations"
  became row permutations inside the 8-row groups (the twin's key order inside a tile IS its sum order); (10) r45.reject
  holds the GPU lock (23c's r44.reject rule).
- **The final review's fixes** (one Important, two minors re-graded by effect): the eager build's cost grows with depth
  (each of its three passes re-reads a row's keys), so a chunk's bounded wait is now `prefill_wait_ms` - the configured
  bound under flash, one bound per 2048 keys of the chunk's deepest row under eager (a 32k eager chunk waits 16 x 30 s,
  not 30 s; row 45's note says how to read an eager timeout); `muse_prefill_test` now gates the last logits' cosine
  (>= 0.9995, PROPOSED) and, in `split` mode, every full layer's first-part rows bitwise at every split (Review Focus 3).

## 16. 23e as built (2026-10-16; the Mac - nothing on a card)

Branch `spec23e-muse-serving` (plan 23e); box queue row 46 holds what the Mac could not run (MG4's restores on the
card, the server on the synthetic and the real checkpoint, MG5, the llama-benchy rows). No kernel added or changed
(`kernel_cmdlines` +0 / -0 / ~0).

- **Both Vishva007 repos ship the OLDER template** (`114f55eb...`, 7167 B - the -AutoRound repo too, not only the
  -GPTQ one as §1.2 said). Decision 1's rule as built (`cli::muse::check_serve_template`): `<snapshot>/chat_template.jinja`
  is used only when its sha256 is the original's `cfc67e5f...`, else `b70-serve` refuses before the device naming
  `--chat-template FILE` (and the vendored `tests/tokenizer/muse/chat_template.jinja`); with the flag any file is
  used and its sha prefix printed in the startup line. 23b's synthetic checkpoints copy the original's: no flag there.
- **Fixtures** (`tools/tokenizer/dump_muse.py tokenizer|templates|parse`, transformers 5.19.0 from 21a's site in
  `agnes-ref-img`, small files only; every request key-sorted): `tests/tokenizer/muse_tokenizer.json` (84 KB),
  `tests/tokenizer/muse_templates.json` (26 cases, 50 KB), `tests/server/muse_parse_cases.json` (22 cases, 28 KB).
- **The template** renders byte for byte on all 26 lists (`template_muse_test`) after ONE minja patch: the tool-call
  capability probe accepts `name="argument_needle"` (without it minja found no argument name it knew and polyfilled
  every call history into JSON content - 5 renders differed); `minja_ext_test`'s `tool_call_probe_attribute` holds
  it, `third_party/VERSIONS` carries the new sha. Through `chat::Template` a string `arguments` does NOT raise (minja's
  object-arguments polyfill parses it, the same render as the object - tested); the raw template does (tested); the
  server parses string arguments first anyway and answers a non-object with a 400 naming the call (name and id).
  `chat::Template` gained `(dir, file)`, `source_sha256()`, `set_clock()` (strftime_now's clock) and
  `chat::sha256_hex`. Muse's list content is joined with "" (the template concatenates text parts).
- **The tokenizer** equals transformers' (`muse_tokenizer_test`: 202048 ids, the 16 named tokens both ways, 24 texts
  incl. every tag and ATEM, corpus.txt's 10240 lines in 10 digests, the 26 renders' ids) - and Review Focus 1: every
  render encodes to exactly one 200000 at position 0; encoding with special tokens would double it.
- **The parser** (`server::MuseOutputStream`, `server/toolcall_muse.{h,cc}`): segments messages at `<|message|>`
  (the header extended back over `to=RECIPIENT`, horizontal whitespace and `<|start|>\s*assistant`), a body ends at
  `<|eom|>` / `<|eot|>` / `<|start|>` / the next header; bodies are STRIPPED (transformers' `text` parser strips) and a
  channel's bodies joined with "\n" (decision 2; content joins too); tool channels' complete invokes are the calls,
  the rest of a tool body dropped; an output with no header at all is one content body (held until a header or the
  end); text between messages dropped; a bare `<|message|>` is content (vLLM; transformers drops it). Typing
  (`muse_convert_value`, decision 15): all-string schemas keep the exact text (a nullable string's literal `null` is
  JSON null, as transformers); other types JSON-decode first, then cast as transformers' `_coerce` (int / number /
  boolean / null / object / array), a union with string falling back to the exact text. Measured: the 22 fixture
  cases equal `parse_response` but 6 documented differences (the reasoning join; typing x 4 incl. transformers
  stripping a multi-line string value; the ns.ns collapse); the 7 rendered turns parse back to their messages; the
  damaged cases; Review Focus 2 (` to=self` cut at every byte never leaks), 3, 6, 7; a 2000-output fuzz (646 cut)
  - every parse whole == byte stream == random 1-7 byte pieces (2041 parses). A mutation (no hold-back of a growing
  `to`) is caught by the fuzz.
- **The request path** (`ChatFormat::Kind::Muse`): `request_template_kwargs` maps the top-level `reasoning_effort`
  onto `reasoning_strength` (an explicit kwarg wins); `check_template_kwargs` holds reasoning_strength to the four
  strengths, current_date / knowledge_cutoff to strings, tool_namespace_descriptions to an object of strings; image /
  video / audio parts are a 400 naming the vision encoder (for every model - none loads an encoder);
  `Sampling::logit_mult` / `softcap` (model properties: b70-serve sets them with generation_config's defaults, the
  adapter writes them into what it filters with) are applied in `filter_probs` to the k candidates (monotone: the
  top-k / top-p order is the raw one), so sampling and every acceptance path see one p; unset, `filter_probs` is
  bit-for-bit what it was. `muse_server_test`: the capped probabilities to 1e-12, a 10^5-draw chi-square 14.0 against
  the capped p (bar 30.58) and ~8.6e13 against the raw one, `accept_point_mass` identical on raw-capped and pre-capped
  rows over 200 seeds. `generation_stop_strings` (a string or a list, else refused naming the key) feed
  `Options::stop_strings`, merged with each request's `stop`.
- **Snapshots** (`runtime::muse::state_runs` / `kv_runs` / `persistent_sizes`; `MuseEngine::save_state` / `load_state`
  / `save_kv` / `load_kv`, Kolibri's loops; `load_state` resets a broken session first): **81,748,992 B** of state
  (39 x 2047 x 1024 - §5.4's first figure, 81,747,968, was a mistyped product), 13,312 B of KV a position; 78 runs (156
  across a wrap or below 0), one host layout at splits 13 and 26. `muse_snapshot_test` (host, 20 s): restores at
  2048 / 4096 / 2049 / 5000 / 300 through `PrefixSession` == the cold run on one card and two; a two-card snapshot
  continues on one and the reverse; over HTTP the greedy chat == the decode run, the repeat restored at 2048
  (`cached_tokens` 2048), a seeded sampled request reproducible through the cap. `muse_snapshot_gpu_test` (the card;
  ckpt4, one card / pp2:2 / cross, and the real checkpoint) is r46.snapshot / r46.real's.
- **b70-serve <muse>** (`src/cli/muse_serve.h`: `MuseEngineAdapterT`, store key "MUSEGLIM"; `serve_muse` in
  `b70_serve.cc`): one card by default, `--pp 2` / `--pipeline-split` / `--pipeline-handoff`, `--max-len auto`
  (MuseEngine plans with the prefill scratch), int8 head default, the startup line (the template's sha, sampling
  defaults, the rings' positions). Refused before the device by name (12 `cli_reject_serve_*` tests, box-only: b70-serve
  is not built on the Mac host): `--mtp`, `--spec mtp|lookup|dflash` (`dflash` is parsed now so Muse names 23f; every
  other model refuses it as before), `--kv-cache int8`, `--prefill-backend l0-int8|sycl-tla`, `--max-len` above
  131072, `--device` with `--pp 2`, a template that is not the original's, `--chat-template` on another model. 23c's
  `cli_reject_muse_serve` is gone. `muse_adapter_compile.cc` instantiates the adapter over `MuseEngine` (Level Zero
  syntax check PASS).
- **MG5's tooling**: `tools/probe/muse_passkey.sh` (BOS + 130000 ids - passkey.py writes no special tokens and every
  Muse prompt starts with the template's BOS); `engine_generate.sh`'s Muse arm runs `b70-decode` on the committed
  set's ids (byte-identical to the reference's input, A4's rule; 512 ids) rather than `b70-serve` - the server's path
  is held to `b70-decode` by r46.chat's `golden_server_test --chat`. BENCHMARKS carries the pending rows.
- **Departures from the plan**: (1) tests registered in `tests/CMakeLists.txt` (no `tests/tokenizer/CMakeLists.txt`
  exists); (2) the state is 81,748,992 B (above); (3) the adapter and the template rule live in `cli/muse_serve.h`
  and were written in Task 4 (the host test drives the adapter); (4) the A4 arm is `b70-decode`'s (above); (5) Task 5
  Step 2 is a stage (r46.chat), not a registered test (Kolibri's precedent: without a checkpoint it would fail, not
  SKIP); (6) the vision refusal is generic.
- **The final review's fixes** (three Important, one Minor re-graded by effect): a seed now holds per request -
  `MuseEngineAdapterT` reseeds the RNG from a seeded request's seed at its first sampled step (before, only the first
  sampled request after startup was seeded, so Muse's default sampling made `seed` meaningless on a long-running
  server and r46's SEEDED checks would differ; `muse_snapshot_test`: the same seeded request twice on one server);
  the media refusal is a list of media part types (`image`, `image_url`, `input_image`, `video`, `video_url`,
  `input_video`, `audio`, `input_audio`, `file`, `input_file`) - a text-bearing `input_text` part stays text for
  every model (`openai_test`); r46.real's template refusal is an `xfail` (the pipeline under pipefail always read BAD,
  and a regressed refusal would have hung the stage); r46.benchy's deepest decode row is 130000 (131072 + 640 did not
  fit; `test_box_validate.py`'s `test_row46_refusal_and_depths` holds both).

## 17. 23f as built (2026-10-16; the Mac - nothing on a card; box queue row 47)

Plan `docs/superpowers/plans/2026-10-15-spec23f-muse-dflash.md`, both parts on ONE branch `spec23f-muse-dflash` (the
operator's 2026-10-15 ruling: Part 2 built regardless of P0 - P0 picks the defaults, §8.5). The rulings with their costs
if wrong are the plan's SDD ledger; the ones that change what §8.4 says are below.

- **P0 (Tasks 1-5)**: `tools/oracle/dflash_ref.py`'s Muse form (`MuseSession` / `draft_block`; bitwise transformers
  5.19.0 on the tiny pair, the window's edge, the crop, Review Focus 4's three mutations), `tools/oracle/muse_taps.py` +
  `muse_oracle.sh taps`, `dflash_accept.py --model muse` (K 1..15 from one 16-row block, the int4 g64 / int8 arms with
  and without the int8 head, raw vs capped q) + `lookup_accept.py --dumps`, `probe_draft_cost --shapes muse`,
  `tools/spec/muse_p0_project.py`. The Mac's verdict and the provisional defaults: §8.5 (int4 g64 + int8 head,
  `--spec-max` 15 under AdaptiveK, q **raw** - decision 11's recommendation reversed by the measurement).
- **No new descriptor / loader** (§8.4's `MuseAssistantDesc` / `load_muse_assistant`): spec 19b's `model::DFlashDesc`
  gained `DFlashForm::Muse` (`window_inclusive`, `norm_two_roundings`; `muse_desc_from_config`,
  `check_muse_assistant`) and `loader::load_drafter` an overload by the target's shape (`DFlashTargetShape`: a Muse
  drafter's vocab is the target's head rows); `DraftFormat::Int4G64` quantises every linear at load
  (`loader::rtn_int4_g64` -> layout 1). Bytes: int4 1.358 GB, int8 2.556, bf16 5.112 (+ the buffers).
- **No edited kernel source (MG0 / D0)**: §8.4's `mg_attn` ring mode and a projection kernel became spec 19b's `df_*`
  sources at Muse's defines from new CMake lines (`df_prep_M16_..._R2112_V202048_E5`, `df_attn_M16_..._R2112_W2049_P64`,
  `df_topk_R15_N202048_U202048`) plus one new source `src/kernels/muse/mg_dflash.cl` (`mg_df_norm`: the two-rounding
  norm; `mg_df_select`: the 16-row readback and the verify's ids). **The ring is 2112 slots**: df_attn's rule is
  `|q - k| <= WIN - 1` with `RING >= WIN`, so the inclusive 2048 window is WIN 2049 over 33 blocks of PB 64 (PB 64 keeps
  its local memory under 64 KB at 64 q pairs) - 1.4 MB more ring than 2048 slots. df_prep's q / k norms round once in
  fp32 where transformers' bf16 rounds twice: inside D1's bars.
- **The draft list always runs the whole 16-row block** (dctl.k = 15, `runtime::dflash::draft_list_k`; the final
  review found the first build passing K through): df_attn drops rows and keys past dctl.k, and
  transformers' generator always attends over all 16 rows - so K only bounds what is verified (the head runs 15 rows
  either way). `runtime::dflash::MuseDrafter` (`runtime/dflash/muse_drafter.h`) owns the weights, the buffers (ring,
  taps [16][5][6656], the 804-word `MuseDraftOut`), the draft list over the target's embedding / head / Control and the
  projection list (up to 16 rows); `runtime::TapSink` (split out of `TapDrafter`, `runtime/tap_drafter.h`) is the
  engine's view of it.
- **The verify** (`runtime::muse::build_verify`, `muse_capture.h`): 23c's decode walk at M = 1..16 over ONE 16-row
  scratch (`MuseBuffers` rows = `MuseSpec::rows`), captured lazily per M; the ids come from a host-written buffer
  through `mg_pf_embed` at M (Control::cur_token has 8 slots), every row's argmax into an output buffer through
  `mg_vargmax` (Control::pos is the host's commit), the taps of every row copied after layers 1 / 13 / 25 / 37 / 49 (M
  x 5 copies). `MuseEngine::verify(drafts, k)` / `verify_ids()` / `read_verify_logits` / `commit(j, t)` (the projection
  of rows 0..j replayed at once - a rejected row's taps never reach the ring) / `max_verify_k`; with a drafter, plain
  tokens (ingest, generate, K = 0) run verify(0) + commit(0). The prefill taps the prompt's last 2048 positions per
  chunk (`MusePrefillTaps`, `Context::copy_2d`) and projects them 16 rows a replay; with a block hook (the prefix
  cache) every row of a chunk ending at a mid-prompt block end as well (`runtime::muse::prefill_tap_r0`), so that block
  end's snapshot carries a full ring (the final review's fix; ~0.5 s more projection on a 32k prompt, derived).
- **Snapshots carry the projected ring** (spec 19d decision 5 as ruled; the operator's 2026-10-16 option a), not §8.4's
  fc rows: 43.3 MB after the 81.7 MB state; `drafter_key` keys the store.
- **The loop** (`cli::muse::MuseEngineAdapterT`, `cli/muse_serve.h`): `step_many` (greedy: the device's picks,
  `accept_greedy`; sampled: the drafts DRAWN ON THE HOST from q_j = softmax(scores_j / T) over the readback's 16
  candidates, `server::dflash_q`, then `accept_sparse` against the capped, filtered p - lossless by construction),
  `step_drafts` (lookup's point mass), the deferred commit, `truncate_to`, the commit flushed before a save.
  `cli/muse_spec.h`: `--spec dflash` / `--spec lookup`, `--draft-model` (default the assistant), `--draft-format int4`
  (default) / `int8` / `bf16`, `--draft-q raw|capped`, `--spec-max 1..15`, `--spec-cost`; `server::MtpCost::muse_dflash`
  / `muse_verify` (DERIVED: verify 1.00 1.18 1.56 1.79 + 0.305 a row; the block flat - int4 0.182, int8 0.263, bf16 0.435).
  **One card**: `--spec` with `--pp 2` is refused by name. `b70-decode --spec dflash` (--ids, greedy, K = --spec-max
  every iteration) is D2's tool; `--spec lookup` is b70-serve's only. Qwen3.8's CLI refuses `--draft-format int4`.
- **Tests**: host - `dflash_ref_test` (the twin `tests/kernels/dflash_ref.h` gained the Muse form: norm2, int4 g64, the
  whole block; against `tests/loader/muse_dflash_fixture.safetensors` from `dflash_fixture.py --muse`),
  `muse_dflash_names_test`, `muse_spec_test` (D2 / D4 over a fake engine), `tools/mac/clrun/muse_dflash_run` (both
  lists on the Mac's GPU; its int4 arm does not build there); card - `muse_dflash_test` (the real assistant vs the twin),
  `muse_verify_{synth,real}_test` (D3 and the commit / prefill-tail halves of Review Focus 3); D1 proper is
  `dflash_d1` + `dflash_d1.py`'s Muse form over the tap dumps (`MODEL=muse tools/box_validate/dflash_d1.sh`).
- **Open (the box's)**: every new binary's first ocloc build; `mg_attn` at M = 16 (NPR 256) may spill registers - the
  lever is chunking rows; P0's agentic acceptance and the measured verify cost (r47.accept / r47.cost) and the verdict;
  D1-D5 (r47.d1 .. r47.benchy); `--spec dflash` stays off by default until D5's rule says otherwise.

