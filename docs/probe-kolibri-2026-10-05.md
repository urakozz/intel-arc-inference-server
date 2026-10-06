# Kolibri-1 - facts from the files, the reference, the quantisation script (2026-10-05)

Spec 20a (`docs/superpowers/plans/2026-10-05-spec20a-kolibri-reference-and-quant.md`), as far as
the **small files** decide it: `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json` (the chat template), `model.safetensors.index.json`, the safetensors
**headers** of all 32 shards (HTTP range requests), the norm and `expert_bias` tensors of layers 0,
4, 25 and 49 (range-fetched, a few KB each), the model card, and Aleph Alpha's vLLM plugin. No
weight shard was downloaded; nothing ran on the box or a card. What needs the real weights is
**pending** at the end.

## Sources

| | |
|---|---|
| bf16 original | `Aleph-Alpha/Kolibri-1-BF16`, commit `8c8b3489`, Apache-2.0, not gated; 32 shards, 58353 tensors, **156.206 GB** (index `total_size` 156206149120 = the header sum) |
| semantics | `Aleph-Alpha/aleph-alpha-inference` @ `049a6a7b` (Apache-2.0): `aleph_alpha_inference/kolibri1.py` (329 lines on vLLM's Qwen3-MoE), `config.py` (`Kolibri1Config(Qwen3MoeConfig)`, `model_type` kolibri1), `__init__.py` (registers the `kolibri1` reasoning parser and **`Hermes2ProToolParser`** as the `kolibri1` tool parser), `tests/test_kolibri1.py` (`test_routing_semantics`: torchtitan's `sigmoid_logit_add` router), `tests/kolibri1_chat_template.jinja` (= the checkpoint's template) |
| inherited parts | vLLM main `96b0efe6b0` (local checkout): `qwen3_moe.py`, `layers/layernorm.py` + `ir/ops/layernorm.py`, `fused_moe/router/gate_linear.py`, `config/model.py` (`head_dtype`), `logits_processor.py` |
| transformers | none: huggingface/transformers#49281 (open "New model" request). Ours: `tools/oracle/third_party/kolibri1/` |

## The model (config + index + headers)

| | |
|---|---|
| layers | 50, every one MoE; `layer_types`: full attention at **4, 9, 14, ..., 49** (every 5th), the other 40 sliding |
| hidden / eps | 2560, RMSNorm eps 1e-6, plain `w * x_hat` (stored weights: final norm mean 42.6, `post_attention_layernorm` means 1.6 .. 10.3 - not `(1 + w)` values near 0) |
| attention | 48 q / 4 kv heads (GQA 12) x 128; `q_proj` [6144, 2560], `k_proj` / `v_proj` [512, 2560], `o_proj` [2560, 6144]; `q_norm` / `k_norm` [128] (per head); no bias |
| positions | `rope_theta` 1e4 (top-level key; transformers 5 turns it into `rope_parameters` {default, 1e4}); RoPE in the sliding layers only, full layers NoPE; `max_position_embeddings` 262144 |
| window | `use_sliding_window` true, `sliding_window` **513 = 512 preceding tokens plus the current one** (model card "Sliding window: 512 preceding tokens plus the current token"; vLLM's per-layer window W keeps keys q-W+1 .. q; transformers' sliding cache keeps W - 1). Key j is visible to query i iff i - 513 < j <= i |
| MoE | 384 experts, top 6, `moe_intermediate_size` 512: `mlp.experts.E.{gate,up}_proj` [512, 2560], `down_proj` [2560, 512]; one shared expert `mlp.shared_experts.*` (512) - SwiGLU, **ungated** (no `shared_expert_gate`) |
| router | `mlp.gate.weight` [384, 2560] bf16; `moe.router.expert_bias` [384] bf16 (the torchtitan name; vLLM loads it into `mlp.gate.e_score_correction_bias`, fp32). Stored biases: layer 0 -7.6 .. 1.8 (273 distinct values), layer 4 -1.95 .. 3.05, layer 25 -1.29 .. 1.44, layer 49 -1.20 .. 0.98; means ~1e-4 (re-centred by the balancing update) |
| `norm_topk_prob` | false |
| vocab | 128000, untied `embed_tokens` / `lm_head` [128000, 2560] bf16 |
| dtypes | **every tensor BF16** (58353 of 58353) |
| `head_dtype` | `float32` - see below |
| sampling | `generation_config.json`: T 1.0, top-p 0.97, top-k 128, `do_sample` true |

**Shard layout:** shard 1 holds `embed_tokens`, `lm_head`, `model.norm` and layers 0-1; shards 2-32
hold 1-3 layers each (layer 26 in shard 17, 27 in shard 18 - a layer never spans more than two
shards); the experts of one layer are contiguous.

**Bytes (headers, bf16):** routed experts 151.00 GB (3.02 GB per layer), attention 3.408 GB (68.2 MB
per layer), shared experts 0.393 GB, routers 0.098 GB, embed + head 1.311 GB, norms and biases
2.2 MB. Derived for int4 g64 (4.25 bits incl. f16 scales, + 0.0625 bits of `qzeros`): experts
40.1 + 0.6 GB, attention 0.91 GB if int4; spec 20 §2's table holds.

## `head_dtype: float32`

A vLLM `ModelConfig` setting (`vllm/config/model.py` `head_dtype`, `_get_head_dtype`): "the last
Linear layer(s) of an LLM"; for a generation model the default is the model dtype, and
`head_dtype: "float32"` in `config.json` makes `LogitsProcessor._apply_head` run the **lm_head in
fp32**: on CUDA/ROCm `torch.mm(hidden_bf16, W_bf16^T, out_dtype=float32)` (bf16 operands, fp32
accumulation, fp32 logits, no fp32 weight copy), elsewhere `F.linear(h.float(), W.float())` -
the same values (bf16 widens to fp32 exactly). vLLM's doc string: "required for RL
training-inference consistency (the trainer computes logits in fp32)". It is not about storage: the
weights are bf16. So the reference and the port compute **logits = fp32(final-norm output, bf16) x
fp32(lm_head)^T**; the engine's int8 head (spec 9) computes in fp32 either way, and its argmax
gate compares against these fp32 logits. A quantised lm_head would be refused by vLLM with this
setting ("only supported for an unquantized lm_head") - one more reason §3.1 keeps it bf16.

## The router, exactly

`sigmoid_logit_add_routing` (plugin) = torchtitan `TokenChoiceTopKRouter(router_score_fn=
sigmoid_logit_add, route_norm=False, route_scale=1.0)`:

- logits = `GateLinear(out_dtype=float32)`: bf16 x bf16 with an fp32 result (cuBLAS epilogue) -
  the logits are **never rounded to bf16** (unlike K2, whose logits are a bf16 F.linear);
- ids = top-6 of `logits + expert_bias` (fp32; the bias widened from bf16);
- weights = `sigmoid(logits[ids])` - **not** `sigmoid(logits) + bias` (vLLM's built-in sigmoid
  scoring, which the plugin's test shows selects differently), not renormalised (they sum to
  ~2.5-6 on random inputs);
- ties: `torch.topk` unspecified; the reference takes the **lower id** (stable sort), and records
  the 6th/7th selection gap per token (`route.moe.gap`) so the engine's gate can tell a tie.

## Chat template, tokens, tool calls

| | |
|---|---|
| tokenizer | BPE (byte-level, "UniBPE"-trained), 127900 vocab entries + 98 added = **127998 ids** (`len(tok)`), ids 0..127997 contiguous; **ids 127998 and 127999 have no token** - the head's last two rows (corrected 2026-10-06 by spec 20e's tokenizer facts, `tests/tokenizer/kolibri_tokenizer.json`: this line said 127923 and 127924, which are `<\|reserved-token-2\|>` / `-3\|>`) |
| specials | `<\|text\|>` 127900, `<\|endoftext\|>` **127901** (pad), `<\|pad\|>` 127902, `<\|chat\|>` 127903, `<\|im_start\|>` **127904**, `<\|/role\|>` 127905, `<\|im_end\|>` **127906** (EOS), PII placeholders 127913-127922, reserved 127923-127997 (`<\|reserved-token-2..76\|>`) |
| not special (plain added) | `<think>` 127907, `</think>` 127908, `<tool_call>` 127909, `</tool_call>` 127910, `<tool_response>` 127911, `</tool_response>` 127912 - single ids, decoded as text |
| BOS | none (`bos_token` null, `add_bos_token` false); `tok("Hallo")` = [38241] |
| EOS | `config.json` `eos_token_id` 127906; `generation_config.json` **[127906, 127901]** - the engine's stop set is both |
| template | ChatML. A system turn is **always** emitted: `# Reasoning effort\n\n<sentence>` (effort high by default; `reasoning_effort` none/minimal/low/medium/high/xhigh/max, or `enable_thinking=false`), then the user's system text before it and the tools block after it. Thinking off appends `<think>\n\n</think>\n\n` to the generation prompt; on, the model opens `<think>` itself. Past assistant turns after the last real user query keep `<think>\n...\n</think>\n\n` |
| tools | `# Tools` section: each tool `tojson` on its own line inside `<tools>...</tools>`, then "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags" |
| **tool calls** | **JSON, hermes-style**: `<tool_call>\n{"name": "<fn>", "arguments": <json object>}\n</tool_call>`, several calls separated by `\n`. Not Qwen-XML. The plugin registers vLLM's `Hermes2ProToolParser` for it, so spec 20e reuses the hermes JSON parser |
| tool results | role `tool` turns become one user turn: `<\|im_start\|>user` + per result `\n<tool_response>\n...\n</tool_response>` + `<\|im_end\|>\n` |
| warning | transformers 5.15 prints "incorrect regex pattern ... fix_mistral_regex=True" for this tokenizer: a heuristic aimed at Mistral tokenizers; the pre-tokenizer regex is the checkpoint's own (`\p{L}+` words with an optional leading non-letter, single digits). Not acted on; 20e checks our tokenizer against `tokenizer.json` byte for byte anyway |

Rendered (German weather query, one call, one result, the answer) - 253 ids:

```
<|im_start|>system
# Reasoning effort

Reasoning effort is set to high. Think carefully ... correctness and clarity.

# Tools
...
<tools>
{"type": "function", "function": {"name": "get_weather", ...}}
</tools>
...<|im_end|>
<|im_start|>user
Wie ist das Wetter in Heidelberg?<|im_end|>
<|im_start|>assistant
<think>
Ich brauche das Wetter.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "Heidelberg"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp": 18}
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Antwort.
</think>

Es sind 18 Grad.<|im_end|>
```

## The reference and the port (20a)

- `tools/oracle/third_party/kolibri1/{modeling,configuration}_kolibri1.py`: `Kolibri1ForCausalLM`
  for transformers 5.x (Apache-2.0 header naming the plugin), loadable with `trust_remote_code`
  through an `auto_map` (the quantisation script's shim writes it into a copy of config.json; the
  vendored `config.json` is the original, unmodified). State dict = the checkpoint's names exactly
  (258 of 258 on the tiny config; 58353 on the real one). Routed experts are per-expert
  `nn.Linear`; the router `mlp.gate` is a plain parameter module (not an `nn.Linear`); the masks
  are built inside each attention layer from the 2D padding mask and the layer's own cache length -
  AutoRound captures block 0's kwargs once and replays them into every block, and block 0 is a
  sliding layer. KV-cached `generate()` with transformers' `DynamicCache(config)` (sliding layers
  keep W - 1 = 512), eager and sdpa.
- `tools/oracle/kolibri_ref.py`: the layer-at-a-time reference (bf16 or an int4 GPTQ export by
  name, dequant.py's rule), `run` (golden file: `resid` / `mixer` / `mlp` per layer, routes ids /
  weights / gap per layer per position, logits, greedy tokens, NLL), `ppl`, `hfcheck` (the port,
  layer-streamed, against it), `facts`, and `run_batch` (layer-major over many sequences; the
  coverage and evaluation stages use it, with `--device cuda|xpu` when there is one).
- Fixed where the plugin leaves it open: ties to the lower id; the combine sums the selected
  experts in **ascending id in fp32** (`fp32(w_e) * fp32(bf16 y_e)`), then `+ fp32(bf16 shared)`,
  one rounding to bf16. Rounding points otherwise follow transformers' Qwen3-MoE in bf16. Known
  ulp-level difference from vLLM: its fused add + RMSNorm normalises the fp32 sum before rounding
  the residual; ours rounds the residual first.
- **KL0** (`test_kolibri_ref.py`, 21 tests, ~2 min in `agnes-ref-img` at 2 threads, all pass): one
  test per trap - plain-w norm, sandwich norms (zero post-norms -> identity, doubled -> doubled
  contribution), NoPE full layers (order- and offset-invariant) vs RoPE sliding layers, the 513
  edge (key i-512 seen, i-513 not, on the real window), router selection on logit + bias with
  sigmoid weights unrenormalised (384 experts, bias x5, both vacuity checks), ties to the lower id,
  shared expert ungated, q/k norm per head, the ascending fp32 combine, the ring cache - then **the
  port == the reference bitwise in bf16 eager** (prompt forward and step-by-step cached decode,
  window 5 x 16 positions and window 513 x 530), **bitwise in fp32 too** (measured max rel diff 0),
  `generate()` == the reference's greedy ids, sdpa within 1e-4 of eager (fp32), left-padded batch
  generation == unpadded; sharded bf16 / int4 GPTQ / layer-streamed readers; `run_batch` ==
  `forward`; the CLI.

## The quantisation script (20b's tool)

`tools/quantize_kolibri1.sh` (+ `tools/quantize/kolibri/`), stages calib, coverage, rtn, tune,
eval, check, card; `tools/quantize/README.md` has the command and the machine. AutoRound is
**intel/auto-round main `6afaecdb` (0.17.0)**, the operator's ruling of 2026-10-05, not the
Qwen3.8 scripts' pinned 0.14.2. Checked against its source and on a tiny Kolibri (5 layers, 16
experts, hidden 128, the real 128000 vocab and tokenizer):

- flags / API: `scheme`, `group_size`, `sym` (default true; the CLI has `--asym`, no `--sym`),
  `iters`, `nsamples`, `seqlen`, `low_gpu_mem_usage`, `device_map` (`--device` is its alias),
  `ignore_layers` (CLI `--ignore_layers` / `--fp_layers`, substring match on module names),
  `model_dtype`, `disable_opt_rtn` (iters 0 is "optimised RTN" unless this is set - the baseline
  uses plain RTN), `disable_model_free` (its model-free fast path only triggers for format
  `auto_round` and a string model; set anyway), `format="auto_round:auto_gptq"`;
- MoE: its fused-MoE unfusing (`modeling/fused_moe/replace_modules.py`) only rewrites experts held
  as 3-D parameters; ours are per-expert `nn.Linear`, untouched - the exported names are
  `mlp.experts.E.*_proj.{qweight,qzeros,scales}`. It adds `mlp.gate` to the ignored layers by
  itself (a no-op: not a linear);
- export (RTN and 2-iteration tuning, both attention arms): every expert and (int4 arm) attention
  linear has `qweight` I32 [K/8, N], `scales` F16 [K/64, N], `qzeros` I32 all 0x77777777, no
  `g_idx`; `quantization_config` quant_method auto-round, packing_format auto_round:auto_gptq,
  bits 4, group 64, sym, `extra_config` bits 16 for the shared experts (and attention in the bf16
  arm, plus regex keys for `mlp.gate`); no lm_head packing issue applies (lm_head stays bf16). The
  0.14.2 shard-writer bug the Qwen3.8 script documents (lm_head packing dropped) is therefore not
  exercised; per-expert completeness is what `check` verifies;
- **on a CPU without bf16 support AutoRound runs and writes the unquantised tensors in fp32**
  (`amp is set to FALSE`); `autoround_run.py` restores them to the source's bf16 bytes after
  proving each is bitwise the widened source, and `check` rejects any non-BF16 plain tensor.

## Pending (needs the weights)

- KL0's second half: the real model's perplexity on German and English text (`kolibri_ref.py ppl`;
  a wrong port is an order of magnitude off), and `hfcheck` on the real checkpoint.
- The plugin itself was not run (no CUDA here): the port is checked against the plugin's code as
  read and its routing test re-implemented, not against vLLM's numbers.
- 20b: the run (machine, time) - spec 20 decision 1; the KL table - decision 2.
