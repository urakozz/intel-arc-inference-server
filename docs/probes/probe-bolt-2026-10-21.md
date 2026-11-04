# Probe: Nemotron 3.5 Bolt 74B-A9B - the facts (spec 25a, 2026-10-21)

Spec 25a Task 4. Everything here is **measured** on the Mac (the `agnes-ref-img` container, torch 2.14.1 CPU,
transformers 5.19.0) from the small files of SergiioB's pinned revision (`b9f381d2`) and NVIDIA's BF16 original
(`56702103`), the shard headers of both repos and 22 MB of tensor data read by HTTP range request
(`tools/oracle/bolt_facts.py --headers --sample`, cached in the main checkout's `oracle-out-bolt-facts/`), unless
marked **derived** (from the configs and headers). No weight shard was downloaded. Citations: `R:` = the checkpoint's
`modeling_nemotron_h.py` (sha256 `e1cb5fc0...`, HF blob `de290021`, byte-identical in SergiioB's, NVIDIA's 24GB and
BF16 repos), `RP:` / `RC:` its puzzle / base configuration files, `N:` = transformers 5.19.0's native
`models/nemotron_h/modeling_nemotron_h.py`. The citations below are found by anchor strings, not typed:
`bolt_facts.py --section cite` prints the table and fails a row that moves (63 of 63 found).

```bash
# reproduce (from the worktree root; the cache dir mounted so the range reads are kept)
docker run --rm --memory 8g --cpus 4 -e OMP_NUM_THREADS=3 -e PYTHONPATH=/ws/oracle-out-q4exp/site \
  -v ~/.cache/huggingface:/hf:ro -v "$PWD":/ws -w /ws agnes-ref-img:latest python3 tools/oracle/bolt_facts.py \
  /hf/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997 \
  --bf16 /hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/snapshots/567021037ff88364143e24ea05e6abd0f72086ff \
  --headers --sample
```

## 1. The rounding chain (spec 25 §2), cited

| point | where |
|---|---|
| block: norm on the input cast to the norm weight's dtype; `residual_in_fp32` false; `h = residual + mixer` (bf16 add) | R:817, R:818, R:839 |
| RMSNorm: fp32 variance (torch's mean), `w (fp32) * x^`, ONE rounding | R:763, R:765 |
| the mixer's dispatch: the kernel path only on CUDA (patched by `bolt_remote`) | R:735 |
| `rmsnorm_fn` imported from mamba_ssm at import (else ImportError) | R:61 |
| gated norm: `rmsnorm_fn(norm_before_gate=False)`, `group_size = d_inner / n_groups` = 1024 | R:307, R:371 |
| `in_proj` split z ‖ xBC ‖ dt (d_mlp 0) | R:480 |
| kernel path, decode: `causal_conv1d_update` on the raw column; `A = -exp(A_log)` fp32; `selective_state_update(dt_softplus=True)`, no `dt_limit` | R:403, R:416, R:425 |
| kernel path, prefill: `dt_limit = time_step_limit` = (0, inf) (RC:225's default; the config's `[0.0, Infinity]`); the conv state = the RAW xBC padded to 4; `causal_conv1d_fn` with SiLU; `mamba_chunk_scan_combined`; y -> gated norm -> `out_proj` | R:449, R:488, R:496, R:511, R:530 |
| fallback, decode: the conv's products and `torch.sum` in the cache's dtype; `dt + dt_bias` in bf16; the clamp at `time_step_min` 0.001; `dB = dt B` in bf16; the state cast to bf16 before `C S`; `y + x D` in bf16 | R:562, R:594, R:595, R:607, R:626, R:636 |
| fallback, prefill: a bf16 `conv1d`, SiLU after it (two roundings); `softplus(dt + dt_bias)` in bf16; the clamp; **the state passing summed over dim 2 - the TARGET chunk** (see §6.1); y kept fp32 into the gated norm | R:573, R:642, R:643, R:695, R:720 |
| MoE: an fp32 zero buffer; experts in ascending id; `f32(bf16 out) * w`; `index_add_`; the dummy forward of an unrouted expert; one rounding to bf16; fc1 / fc2 latent; `+ shared_experts(residuals)` on the NORMED input | R:926, R:930, R:939, R:940, R:944, R:950, R:960, R:964 |
| expert MLP `down(relu2(up(x)))`, non-gated; routed experts in the latent | R:881, R:873 |
| router: fp32 logits from widened x and W; sigmoid; `sel = s + e_score_correction_bias`; top-9 `sorted=False`; `w = s` gathered; `/ (sum + 1e-20)`; `x 5.0`; the bias kept fp32 | R:1044, R:1045, R:1001, R:1016, R:1047, R:1049, R:1051, R:1063 |
| attention: scale `head_dim ** -0.5`; q / k / v / o without bias, no norm, no RoPE; eager `Q K^T x scale`, softmax fp32 then bf16 | R:1137, R:1140, R:1091, R:1096 |
| model: `hidden_states` collects block INPUTS; `norm_f` appended last; `has_previous_state` set after a forward | R:1482, R:1497, R:1503 |
| cache: the sequence length from the first attention block; generate builds it in the model's dtype | R:250, R:1666 |
| head: `lm_head(h.to(head dtype)).float()` (every row: R ignores `logits_to_keep`) | R:1731 |
| puzzle: the per-layer config from `block_configs`, merged into the global dict | RP:68, RP:86 |
| N: `time_step_limit = (time_step_min, inf)` - N clamps its chunk path (N:275) and not its recurrent one; N's state passing sums over the SOURCE chunk | N:394, N:275, N:330 |

The three readings of spec 25 §2.0, as measured on the tiny puzzle models (`tools/oracle/test_bolt_ref.py`, B1):
the training kernels (bolt_mamba_ops, the reference), R's torch fallback (what transformers runs on a CPU), N
(cross-check only: no puzzle support).

## 2. The checkpoint (headers, range-read)

**SergiioB @ b9f381d2:** 112,940 tensors in 9 files (8 shards + `model_extra_tensors.safetensors`), index metadata
`total_size` 41,915,552,784, `total_parameters` 11,725,189,122 (AutoRound's count of packed elements); tensor bytes from
the headers 43,600,273,448 (all, vision included). Language names 112,091; vision / projector 849 (393 in
`model_extra_tensors`). `bolt_ref.expected_names(cfg, "gptq-g128")` equals the index's language names exactly.

| group (`language_model.model.` prefix) | count | dtype | shapes |
|---|---:|---|---|
| `embeddings.weight`, `lm_head.weight` (no prefix `model.`), `norm_f.weight` | 1 each | BF16 | [131072, 4096], [4096] |
| `layers.L.norm.weight` | 88 | BF16 | [4096] |
| `layers.L.mixer.{A_log, D, dt_bias}` | 40 each | F32 | [128] |
| `layers.L.mixer.conv1d.{weight, bias}` | 40 each | BF16 | [10240, 1, 4], [10240] |
| `layers.L.mixer.norm.weight` | 40 | BF16 | [8192] |
| `layers.L.mixer.in_proj.{qweight, scales, qzeros, g_idx}` | 40 each | I32 / F16 / I32 / I32 | [512, 18560], [32, 18560], [32, 2320], [4096] |
| `layers.L.mixer.out_proj.*` | 40 each | | [1024, 4096], [64, 4096], [64, 512], [8192] |
| `layers.L.mixer.{q,k,v,o}_proj.*` | 8 each | | q / o [512, 4096]; k / v [512, 256] |
| `layers.L.mixer.gate.weight` / `.e_score_correction_bias` | 40 each | BF16 / F32 | [E, 4096] / [E] |
| `layers.L.mixer.{fc1,fc2}_latent_proj.*` | 40 each | | [512, 1024] / [128, 4096] |
| `layers.L.mixer.shared_experts.{up,down}_proj.*` | 40 each | | [512, 5376] / [672, 4096] |
| `layers.L.mixer.experts.E.{up,down}_proj.*` | 13,824 each | | up [128, I]; down [I/8, 1024] |

**The layer pattern** `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME`, attention
at 7, 16, 25, 36, 47, 58, 69, 78. Per MoE layer (E, I):

```text
1:(240,2048) 3:(240,1536) 5:(240,2048) 8:(304,2304) 10:(240,2304) 12:(240,2048) 14:(240,2048) 17:(304,2304)
19:(304,2048) 21:(432,2560) 23:(432,2304) 26:(240,2048) 28:(432,2304) 30:(304,2560) 32:(304,2304) 34:(432,2304)
37:(368,2304) 39:(368,2048) 41:(368,2304) 43:(432,2560) 45:(368,2560) 48:(496,2560) 50:(432,2560) 52:(240,2304)
54:(368,2304) 56:(368,2560) 59:(304,2048) 61:(240,2304) 63:(240,2304) 65:(240,2048) 67:(240,2048) 70:(368,2048)
72:(368,2304) 74:(432,2304) 76:(432,2560) 79:(496,2560) 81:(432,2560) 83:(432,2560) 85:(496,2560) 87:(368,2560)
```
12 distinct pairs: (240, 1536) x1, (240, 2048) x7, (240, 2304) x4, (304, 2048) x2, (304, 2304) x3, (304, 2560) x1,
(368, 2048) x2, (368, 2304) x4, (368, 2560) x3, (432, 2304) x4, (432, 2560) x6, (496, 2560) x3. E: 240 x12, 304 x6,
368 x9, 432 x10, 496 x3; I: 1536 x1, 2048 x11, 2304 x15, 2560 x13; 13,824 experts.

**NVIDIA's BF16 original @ 56702103:** 28,725 tensors in 63 shards, `total_size` 149,770,189,864 B; language names
28,331 under **`language_model.backbone.*`** (`expected_names(cfg, "bf16")` equals them), vision 394. Its dtypes:
every weight BF16, `e_score_correction_bias` F32, and **`A_log` / `D` / `dt_bias` BF16** (SergiioB stores them F32
holding bf16 values - §3). Parameters from its headers: **74,042,720,768** - spec 25 §3's derived total, now measured.

**Derived bytes (spec 25 §3 re-derived from the config, int4 g64 at 0.53125 B a weight):** Mamba 4,383,047,680 (2.328
GB), attention 285,212,672 (0.152 GB), MoE dense 2,097,152,000 (1.114 GB), routed experts 66,144,174,080 (35.139 GB),
routers 56,623,104 (0.113 GB bf16), small 2,769,408, embeddings 536,870,912 (1.074 GB bf16), lm_head 536,870,912 (0.537
GB int8); total 74,042,720,768; active per token 9,052,884,992 (the spec's 9,052,880,896 plus `norm_f`'s 4096). The
checkpoint's qweights hold 72,909,586,432 int4 weights = the linears' parameter count. SSM state 41,943,040 values =
167,772,160 B fp32 a sequence; KV 8,192 B a position.

## 3. Quantisation and the sampled tensors

`quantization_config`: `{"autoround_version": "0.17.0", "bits": 4, "damp_percent": 0.01, "data_type": "int",
"desc_act": false, "group_size": 128, "lm_head": false, "provider": "auto-round", "quant_method": "gptq", "sym": true,
...}`; `dynamic`: **44** `-:` exclusions (embeddings, lm_head, vision, the 40 routers - spec 25 §1.3 said 43) and one
`+:` rule, `"+:.*mixer\\.out_proj$": {"bits": 4, "group_size": 128, "sym": true}` (the restatement 25b accepts).

| sampled linear | scales | negative | subnormal (min abs) | max abs | qzeros 0x77777777 | g_idx == i // 128 |
|---|---|---:|---:|---:|---|---|
| L0 `out_proj` | F16 [64, 4096] | 132,409 / 262,144 | 0 (min 2.7e-3) | 0.053 | yes | yes |
| L0 `in_proj` | F16 [32, 18560] | 295,828 / 593,920 | 6,773 (1.0e-5) | 0.041 | yes | yes |
| L1 expert 0 `down_proj` | F16 [16, 1024] | 8,257 / 16,384 | 0 | 0.0085 | yes | yes |
| L1 expert 0 `up_proj` | F16 [8, 2048] | 8,053 / 16,384 | 64 (1.0e-5) | 0.0067 | yes | yes |
| L7 `q_proj` | F16 [32, 4096] | 65,610 / 131,072 | 0 | 0.027 | yes | yes |
| L1 `shared_experts.up_proj` | F16 [32, 5376] | 81,031 / 172,032 | 11,469 (1.0e-5) | 0.017 | yes | yes |

No zero scale in the samples; about half the scales negative everywhere (AutoRound's signed symmetric scale); every
scale F16, `out_proj` included.

**The dt-clamp census** (every Mamba layer's `dt_bias`, 512 B each): **212 of 5,120 heads** have `softplus(dt_bias)
< 0.001` - where R's fallback clamp (R:595 / R:643) changes the decay. By layer (heads below the clamp / dt_bias min ..
max / A_log min .. max / D min .. max):

```text
L0 25 -10.50..8.50 A -4.66..4.41 D -1.63..7.84 | L2 23 -10.56..3.09 | L4 5 | L6 2 | L9 1 | L11 2 | L13 3 | L15 3
L18 1 | L20 2 | L22 1 | L24 4 | L27 1 | L29 0 | L31 4 | L33 0 | L35 3 | L38 2 (dt_bias min -11.69) | L40 3 | L42 2
L44 7 | L46 16 | L49 13 | L51 0 | L53 0 | L55 13 | L57 25 | L60 14 | L62 7 | L64 5 | L66 2 | L68 7 | L71 0 | L73 2
L75 1 | L77 4 | L80 1 | L82 0 | L84 3 | L86 5
```
Over all layers `dt_bias` -11.688 .. 8.500, `A_log` -6.81 .. 8.81 (A down to -6,700), `D` -3.77 .. 7.84; every value a
bf16 value. **`e_score_correction_bias`** (all 40 routers): 4.470 .. **48.04** - each layer's biases sit in a narrow
band around its own offset (L1 4.470 .. 4.550, L3 13.78 .. 13.87, L8 17.56 .. 17.64, ...); the largest-bias 9th / 10th
gap per layer goes down to **0** (L12: two experts carry the same bias value) - selection is decided by s on top of
nearly equal biases.

## 4. Tokens and the template

`tokenizer.json` sha256 `d0432ad8...` (identical in all three repos): BPE, vocab 131,072 (every row a token), 269,443
merges, `ignore_merges` true, byte-level, the `\p{M}` split regex (spec 25 §1.1); 1,000 added tokens (ids 0-999, 994
special). Ids: `<unk>` 0, `<s>` 1, `</s>` 2, `[INST]` 3 ... `[TOOL_CALLS]` 9, `<|im_start|>` 10, `<|im_end|>` 11,
`<think>` 12 / `</think>` 13 / `<tool_call>` 14 / `</tool_call>` 15 / `<tool_response>` 16 / `</tool_response>` 17
(**not special**), `<image>` 18, `<SPECIAL_990>` 990. The post-processor is ByteLevel only: no BOS is ever added
(`AutoTokenizer(...)(text)` = the ids without specials). `tokenizer_config.json`: `fix_mistral_regex` true, `eos_token`
`<|im_end|>`, `pad_token` `<|im_end|>`, `padding_side` left. **`fix_mistral_regex` is a no-op here:** 5.19.0 returns
before patching when the config's `transformers_version` >= 5.0.0 (`tokenization_utils_tokenizers.py:1461-1462`; the
checkpoint says 5.12.1), and the shipped regex already is the "fixed" one; measured: AutoTokenizer = the raw
`tokenizers` file on a mixed probe (38 ids). Generation configs: SergiioB's `eos_token_id` 11 only, no sampling keys;
NVIDIA's BF16 `do_sample` true, T 1.0, top-p 0.95, `eos_token_id` [2, 11] (Decision 12).

`chat_template.jinja` sha256 `6521afc75afb0661e3558e7e6dee5f2d15de8be1c4dece8171935e778ea283f4`, 12,514 B, 240 lines,
identical in all three repos. Features minja must render: macros (`render_extra_keys`, `render_content`), `namespace`,
`is mapping` / `is sequence` / `is string` / `is iterable` / `is none` / `is defined`, `|items`, `| tojson | safe`,
`| string`, `| trim`, `| length`, `| default('', true)`, `.split`, slicing (`[1:]`, `[-1]`), `loop.previtem` /
`loop.nextitem`, `loop.index0` / `loop.last`, a filtered `for ... if`, `~`, `in` / `not in`. Fourteen renders through
`apply_chat_template` are committed as **`tests/server/bolt_template_cases.json`** (messages, tools, kwargs, the text,
the ids; 25e's byte-identity cases). What they show:
- the system block is ALWAYS rendered (`<|im_start|>system\n<|im_end|>\n` with no system message);
- tools render as `<tools><function><name>...` with `<description>`, `<parameters>` / `<parameter>` (`<type>`,
  `<enum>` as JSON) and unknown keys through `render_extra_keys` - **`additionalProperties: false` renders as
  `<additionalProperties>False</additionalProperties>`** (Python's `str(False)`);
- history call arguments: mappings / lists `tojson`, everything else `| string` - **`true` -> `True`, `null` ->
  `None`** in the re-rendered history (25e must reproduce Python's spelling, not JSON's);
- an assistant turn without reasoning gets `<think></think>` prepended; with `reasoning_content`
  `<think>\nREASONING</think>CONTENT`; before the last user turn (`truncate_history_thinking` default true) the
  reasoning is dropped to `<think></think>`; consecutive tool results group into one user turn of
  `<tool_response>` blocks;
- the generation prompt `<|im_start|>assistant\n<think>\n`, or `<|im_start|>assistant\n<think></think>` with
  `enable_thinking` false; `medium_effort` appends `\n\n{reasoning effort: efficient}` to the last user message.

Golden prompt texts under Bolt's tokenizer (no BOS): `prose.txt` 43 ids, `code.txt` 53, `long.txt` 3,692; each
decodes back to its text.

## 5. torch's CPU behaviours the reference depends on (torch 2.14.1, AVX2, 3 threads)

- **`topk(k=9, sorted=False)` on an exact tie at the 9th / 10th:** of 240 values the lower id wins 28 / 50, the higher
  22 / 50; of 496, 19 / 50 vs 31 / 50; repeated calls identical. Not the lower id: exact ties are undetermined (the
  engine's rule is the lower id; the recorder's `route.gap == 0` flags them). The `ties` tiny variant: 67 / 61 and
  47 / 81 on 128 rows.
- **GEMM row-count dependence:** a linear's rows recomputed in sub-batches of M rows. fp32 (R's router: the widened
  `F.linear`, R:1044) at [512][4096] x [496][4096]^T: M = 1, 2, 9, 64, 128 change every row (5,536 - 219,204 elements),
  M = 2048 none; at [240][4096] M = 2 and 2048 none, the others every row; the same pattern at the other shapes. **bf16
  at every shape and M: 0 rows.** So R's router logits depend on how many rows a forward carries (chunking, batching),
  every bf16 linear does not - why chunked = whole holds bitwise outside the Mamba mixers only with the router's outputs
  replayed (`test_chunked_equals_whole`: the route SETS never changed, `w` on up to 1,324 of 1,500 (row, layer) pairs
  at chunk 1 by an ulp), and why B3 compares `w` with an ulp tolerance.
- **The router's sum of the nine s (R:1049) has no fixed order we could name:** on [100000][9] fp32 `torch.sum`
  equals the sequential sum on 60 %, an fp64 sum rounded once on 62 %, the best of the 1,430 order-preserving binary
  trees on 60 %, the best of 3,000 sampled sequential permutations on 90 % (a 1-D 9-element `sum` equals the row of the
  2-D one: it is ATen's own vectorised cascade). The twin's sum (25c) is a fixed order of its own; its `w` differs from
  R's by an ulp on a fraction of rows - inside B3's tolerance, never in the route set.
- **`softplus(dt_bias) < 0.001`** heads: §3's census.

## 6. B1 on the tiny puzzles (`test_bolt_ref.py`, 25 tests, 96 s; `test_bolt_make_tiny.py`, 12 tests)

### 6.1 Measured bounds (fp32, max |difference| / max |value|)

| comparison | measured |
|---|---|
| the port's Mamba mixers vs R's torch fallback, 40 ids + 8 cached steps (one SSD chunk; no head near the clamp) | 2.7e-7 .. 8.5e-7 (outputs), 4.2e-8 .. 2.9e-7 (states) |
| the same, **400 ids** | **0.40 .. 0.57 (outputs), 0.11 .. 1.0 (states)** - R's fallback is wrong past one chunk |
| the same, 400 ids, R's fallback with only its state-passing index fixed (`fallback_forward(fix_state_passing=True)`, N:328-330's form) | 7.3e-7 .. 1.6e-6 |
| a 300-row prefill's state vs the recurrence (300 `ssm_update` rows): R / R fixed / the port | 2.3e-1 / 9.7e-7 / 9.0e-7 |
| chunked (1, 2, 3, 64, 127, 128, 129, 255) vs whole, every block and logit (fp32) | <= 1.9e-5 (chunk 128 = the SSD chunk: 7.2e-6) |
| chunked vs whole, the Mamba outputs (bf16; the rest bitwise with the routers replayed) | 3.5e-3 (chunk 128: bitwise) |
| `ssm_chunk` at 181 = 128 + 53 rows vs the recurrence | 8.7e-7 (y), 2.4e-7 (state); finite (vLLM 783380af39's edge) |
| `dtclamp` (half the heads below the clamp): port vs fallback / with the ops' clamp switched on | 8.1e-2 / 6.6e-7 |

**Finding (not in spec 25):** R's torch fallback mis-passes the SSD state between chunks: R:692-697 builds
`decay_chunk[z][s]` and sums over dim 2 - the TARGET chunk z - where the recurrence sums over the source s
(transformers' native class does: N:328-330, `decay_chunk.transpose(1, 3)`). With one chunk (<= 128 rows) and a zero
initial state the result is the same; past it every Mamba output is wrong. So "R as transformers runs it on a CPU" is
not just differently rounded - it is wrong for any prompt longer than 128 tokens. Decision 1's reference (the kernel
path) is unaffected (its chunk scan is ours).

### 6.2 The bf16 difference table (one Mamba layer of `pattern12`, a 40-row prompt + 6 decode steps)

The restated fallback with every rounding point kept is R's `torch_forward` bitwise (both caches); with no point kept it
is the port to 0 (fp32 cache) / 4.5e-3 (bf16 cache, outputs) - the switches are complete. Each point ALONE (the others
at the kernel path's semantics), max relative difference from the port, outputs / states:

| point | fp32 cache | bf16 cache |
|---|---|---|
| `conv_dec_cache_dtype` - decode conv in the cache's dtype (R:557-565) | 0 / 3.5e-9 | 1.4e-2 / 3.7e-3 |
| `conv_prefill_bf16` - prefill bf16 conv1d, rounded before the SiLU (R:573) | 1.3e-2 / 1.0e-4 | 1.3e-2 / 2.3e-4 |
| `dt_bf16` - dt + dt_bias and softplus in bf16 (R:594, R:642) | 7.1e-3 / 2.4e-3 | 7.1e-3 / 3.7e-3 |
| `clamp` - dt clamped at 0.001 (R:595, R:643) | 5.4e-3 / 1.1e-3 | 5.4e-3 / 1.1e-3 |
| `dBx_bf16` - dB and dB x in bf16 (R:607, R:612) | 0 / 3.5e-9 | 4.5e-3 / 0 |
| `S_bf16` - the state cast to bf16 before C S (R:626) | 3.6e-3 / 3.5e-9 | 4.5e-3 / 0 |
| `Dx_bf16` - x D and y + x D in bf16 (R:636) | 3.6e-3 / 3.5e-9 | 4.5e-3 / 0 |
| `y_prefill_fp32` - prefill y kept fp32 into the norm (R:720) | 3.6e-3 / 3.5e-9 | 4.5e-3 / 0 |
| every point (R's fallback) | 1.6e-2 / 4.2e-3 | 1.6e-2 / 7.4e-3 |

(bf16 outputs: 3.6e-3 is one bf16 ulp of the largest output.) Spec 25 §2.0 listed the decode conv products, dt +
dt_bias, D x, dB x, the state cast, the prefill conv and the clamp; measured, two more: softplus itself rounds in bf16
(part of `dt_bf16`), and the prefill keeps y in fp32 into the gated norm (the kernel path rounds y once to bf16).

### 6.3 Other tiny-model facts

- R's remote class builds and generates in 5.19.0 (`bolt_tiny_check.py`): cached vs uncached argmax equal in fp32 (both
  paths), one undetermined row in bf16 (top-2 margin 0.070 < that row's 0.097 / 0.105 |diff|); logits not bitwise
  (2.7e-6 / 3.6e-6 fp32, 0.16 / 0.11 bf16 - torch path / kernel path).
- `from_pretrained(dtype=bf16)` keeps the router's `e_score_correction_bias` fp32 (values equal) and casts `A_log` /
  `D` / `dt_bias` to bf16 (exact: they are bf16 values).
- The bias reorders: the top-9 by `sel` differs from the top-9 by `s` on 47 / 256 rows; at s ~ 4e-31 the `+ 1e-20`
  makes the weights sum to 1.8e-9 instead of 5.0.
- The combine in rank order differs from R's ascending-id combine on 1 / 64 rows (one rounding of an fp32 sum).
