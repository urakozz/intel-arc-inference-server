# Qwen3.8-Flash-Next (`qwen4_exp`) - facts from the files and the source (2026-10-09)

Spec 21a (`docs/superpowers/plans/2026-10-09-spec21a-qwen4exp-facts-and-reference.md`, Task 2), as far as
the **small files and the source** decide it: both checkpoints' `config.json`, `generation_config.json`,
`tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, `model.safetensors.index.json`, every
shard's safetensors **header** (HTTP range requests: 131 + 17 shards), the three I64 PLE tensors and one
routed expert's `qzeros` (range-fetched, a few hundred bytes each), the original's model card; transformers
5.19.0's `qwen4_exp` source and vLLM's. No weight shard was downloaded; nothing ran on the box or a card.
`tools/oracle/qwen4exp_facts.py <snapshot | hf:REPO>` prints every table below (no torch);
`qwen4exp_ref.py facts <snapshot>` adds the expected-names check and torch's tie rule.

Citations: `M:` = transformers 5.19.0 `models/qwen4_exp/modeling_qwen4_exp.py`, `C:` =
`configuration_qwen4_exp.py` (installed by `tools/oracle/qwen4exp_env.sh`, byte-identical to the scratch
venv the spec read); `V/` = vLLM v0.31.1rc0-138 `b52ae2aa7f` (`/Users/urakozz/PycharmProjects/vllm`),
`vllm/models/qwen4_exp/`. Numbers are **measured** unless marked derived / estimated / published.

## Sources

| | |
|---|---|
| bf16 original | `Qwen/Qwen3.8-Flash-Next` @ main: 131 shards, **1658 tensors, 360.000 GB** (1325 text + 333 vision); `Qwen4ExpForConditionalGeneration`, saved by transformers 5.8.0.dev0; licence **`qwen-community-1.0`** (the card's front matter: `license: other`, `license_name: qwen-community-1.0`, `license_link: LICENSE`) |
| Intel's derived | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ main: 17 shards, **224280 tensors, 181.165 GB** (223947 text + 333 vision) |
| tiny | `qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next`: one shard, 278 tensors, fp32, 124 MB; `qwen4_exp_text`, saved by 5.16.1 |
| card (published) | 125B parameters with 6B activated, plus 51B n-gram embedding and 4B MTP; 48 layers = 12 x (3 x (GDN -> MoE) -> 1 x (QSA -> MoE)); QSA budget "512 blocks or 2048 tokens", indexer MQA 4 q heads + 1 key head of 128; 512 experts, 10 routed + 1 shared, 640; gated residual 4 branches, rank 320; "MTP: 1 layer, trained with multi-steps"; context 262,144 native |

## Task 1: the environment and the tiny model (measured, `agnes-ref-img` + the 5.19.0 site)

- **transformers 5.19.0 runs beside the image**: `qwen4exp_env.sh` installs `transformers==5.19.0`,
  `tokenizers==0.23.2` (5.19.0 requires `>=0.23.1,<0.24`; the image has 0.22.2), `huggingface_hub==1.33.0`,
  `safetensors==0.8.0` with `pip --no-deps --target`; `import torch, transformers` -> `2.14.1+cpu 5.19.0`
  (METADATA asks `torch>=2.5`). The image is unchanged (5.15.0 for every other oracle).
- **5.19.0 loads the tiny model as shipped - no config rewrite.** `Qwen4ExpTextConfig.__post_init__`
  (C:165-185) ends in `super().__post_init__`, whose last step `remap_legacy_layer_types(config=self)`
  (`configuration_utils.py:398`) maps `qwen_sparse_attention -> indexed_attention`
  (`_LEGACY_LAYER_TYPE_REMAP`, `configuration_utils.py:96`) before `validate_architecture` (C:187) runs.
  Loaded `layer_types`: `[linear_attention x3, indexed_attention]`. The real checkpoint's `full_attention` is
  remapped by the class itself (C:179-183). The plan's fallback (a rewritten config) was not needed.
- **The experts implementation 5.19.0 picks: `grouped_mm`** (`@use_experts_implementation`, M:912; the
  dispatcher `integrations/moe.py:542`). The tiny's per-expert `mlp.experts.N.{gate,up,down}_proj.weight`
  are fused on load into `gate_up_proj` [E][2I][H] **gate rows first** (`conversion_mapping.py:1918`:
  qwen3_5_moe_text's mapping = qwen2_moe's per-expert -> fused converter), and the 128
  `ngram_embedding.shard_K.weight` are concatenated into `ngram_embedding.weight` (`conversion_mapping.py:1919-1928`,
  `Concatenate(dim=0)`), both checked tensor for tensor.
- **Cached decode vs one uncached forward** (24 random ids, greedy 8, then the 32 ids at once): **bf16 bitwise**;
  fp32 max |diff| **7.45e-8** (not bitwise: the GDN's cached decode is `torch_recurrent_gated_delta_rule`, the
  uncached forward `torch_chunk_gated_delta_rule` - two fp32 op orders, M:579-604); argmax equal.
- **The selection acts past 2050**: a 2100-id bf16 forward selects differently from causal on rows
  **2051..2099 (49 rows) and on none below**; row 2050 selects all 2051 visible positions, row 2051 2048.

## The rounding chain (spec 21 §2; every point is a rounding to the model dtype unless named)

| point | what transformers computes | source |
|---|---|---|
| RMSNorm (every `(1 + w)` norm: HC, q / k, indexer q / k, PLE, MTP pre-norms) | `x.float()`; `x * rsqrt(mean(x^2) + eps)` (grouped: the mean per `group_size` slice); `* (1.0 + w.float())` in fp32; **one** cast `.type_as(x)` | M:144-169 |
| grouped HC norm | the same over `[..., 4, 2560]` (`group_size` = hidden): each stream normalised alone, the weight [10240] per element | M:1001, M:153-157 |
| GDN gated norm | `x.float()` normalised, **cast back** to the dtype, `w * x` in the dtype (rounded), `* act(z.float())` in fp32, one cast; `act = output_gate_type` = **sigmoid** (M:491-493, the config's `output_gate_type: sigmoid`) - silu on Qwen3.5 / Ornith | M:179-188 |
| GDN | `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)` fp32; `beta = sigmoid(b)` in the dtype; q / k `repeat_interleave(48 / 16 = 3)` on the head axis: **v head h reads k head h // 3**; the delta rule in fp32 (L2 norms eps 1e-6, q * 128^-0.5), the output cast once; conv SiLU in the dtype | M:571-604, M:248-251 |
| gated residual (HC) | `xn = grouped_norm(H)`; `down(xn)` rounded, `/ 4` rounded, `silu` rounded; `up(.)` rounded, `sigmoid` rounded; `g * xn` rounded; `.mean(dim=-2)` over the 4 streams (fp32 accumulation, one rounding); inject: `block_inject(xn)` rounded, `/ 4` rounded, `sigmoid` rounded, `2 *` (exact) | M:1013-1023 |
| the combine | `H0 + (y.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)`: the **product rounded, then the add** (vLLM's fused combine does both in fp32 with one rounding, `V/nvidia/ops/hc.py:225-228` - the reference is transformers') | M:1294-1301 |
| the layer | `if PLE: H = H + ple(H)`; attn HC -> mixer -> combine; MLP HC -> MoE -> combine | M:1275-1302 |
| QSA projections | `q_proj` -> `view(..., -1, 512)` -> `chunk(2)`: per head **[q 256 \| gate 256]**; q / k `(1 + w)` norms per head; partial NEOX RoPE on dims [0, 64) (`rotate_half` over the 64; cos / sin in the dtype); scale `256^-0.5 = 1/16` | M:859-869, M:627-662, M:820 |
| QSA attention (eager) | `q k^T` in the dtype, `* scaling`, `+ mask` (0 / finfo.min), softmax **in fp32**, cast, `@ v`; `* sigmoid(gate)` in the dtype; `o_proj` | M:786-808, M:889-892 |
| RoPE | `inv_freq = 1 / 1e7^(arange(0, 64, 2) / 64)` fp32; mRoPE sections [11, 11, 10] interleaved - text's three equal position ids make it plain 1D RoPE; cos / sin cast to the dtype | M:95-141 |
| indexer q | `index_qk_proj` [640] -> q 4 x 128 \|\| 1 raw key 128; `q_layernorm` (`(1 + w)`), RoPE on its first 64 dims at the query's position | M:698-706 |
| indexer keys | the cache keeps **raw** keys (un-normed, un-roped, dtype); a complete block's key = `groups.float().mean(dim=1)` -> the dtype -> `k_layernorm` -> RoPE at the block's **first** position | M:708-742 |
| indexer scores | `matmul(q.float(), keys.float().T)` [4 heads][n], `relu`, **sum over the 4 heads**, `/ sqrt(128)` - all fp32 | M:744-747 |
| selection | `topk(min(512, n))` over the complete blocks, blocks expanded to their 4 positions in topk order, then **every tail position** of the open block (`local_visible[n*4:]`); a row sees at most 2048 + 3 = 2051 positions; with <= 2051 visible it selects all of them | M:749-756 |
| router | `F.linear` in the dtype; `softmax(logits, dtype=float)`; `topk(10)`; renormalised **in fp32** (`norm_topk_prob` true, C:162); cast to the dtype; no bias | M:961-970 |
| experts (`grouped_mm`, the default) | rows sorted by expert id (`torch.sort`); gate\|\|up grouped GEMM rounded; `silu(gate) * up` rounded; down rounded; `* w` (the bf16 weight) rounded; back to row order; **`.view(T, 10, H).sum(dim=1)`**: the 10 products summed in the router's **topk slot order** (descending p) with fp32 accumulation, one rounding (`eager` instead: ascending expert id, `index_add_` in the dtype - not used) | `integrations/moe.py:388-488` |
| shared expert | `down(silu(gate(x)) * up(x))` each rounded; `sigmoid(shared_expert_gate(x))` rounded; product rounded; `routed + shared` rounded | M:907-909, M:981-992 |
| PLE ids | int64 throughout: history from the cached 2 ids + the chunk; a predecessor missing or at / before the last EOS (248044) strictly before the token reads as 248044 (padding is EOS first, M:1464-1468); `mixed = (t0 m0) XOR (t1 m1) [XOR (t2 m2)]`; `id = mixed mod prime_h + offset_h`; heads 0-7 bigram, 8-15 trigram | M:1107-1166 |
| PLE block | `e` = the 16 rows (160 each) -> 2560; `key = norm_key(key_proj(e))` grouped [4][2560]; `value = value_proj(e)`; `q = norm_query(H)` grouped; `s = (key * q).sum(-1) / sqrt(2560)` in the dtype; `gate = sqrt(max(abs(s), 1e-6)) * sign(s)`; `sigmoid(gate) * value` per stream; `norm_conv` grouped; dilated (3) depthwise conv k 4 over a 9-row history, SiLU; `gated + conv`; `H + out` | M:1227-1247, M:1208-1225 |
| the model | `embed` -> `repeat(1, 1, 4)` (4 identical streams) -> 48 layers -> `hyper_connection_mixer` (no inject) -> `lm_head` on its output; **no final norm, no per-layer input / post-attention norm** | M:1470-1485, M:1656 |

**Restated ops** (`qwen4exp_ref.py`: `rms_norm`, `gated_residual`, `hc_combine`, `gdn_gated_norm`, `router`,
`ple_hash_ids`, `block_keys`): each is the same op sequence and is held to transformers' module bitwise
(`test_qwen4exp_ref.py`), so 21c's fixture can import them as the reference's arithmetic.

## The checkpoints

**The original** (bf16): routed experts **fused** under `model.language_model.layers.N.mlp.experts.gate_up_proj`
[512, 1280, 2560] (gate rows [0, 640) then up, M:943) and `down_proj` [512, 2560, 640] - 241.6 GB of the 360;
the PLE table as 128 shards `ple.ple_embedding.ngram_embedding.shard_K.weight` [2500012, 160] (102.4 GB);
no `model.language_model.norm`; the MTP head `mtp.*` (below). `qwen4exp_ref.expected_names(tc, "bf16")` equals
the index's 1325 text tensors exactly (`test_real_names`).

**Intel's** (AutoRound 0.15.0): routed experts **per expert** `...mlp.experts.E.{gate,up,down}_proj.{qweight,
qzeros,scales}` - qweight I32 [K/8][N], scales **F16** [K/128][N], qzeros I32 [K/128][N/8]; layer 0 expert 0's
qzeros are all **0x77777777** (sym, auto_gptq's zero - 1 convention: dequant `(q - 8) x scale`, the engine's
rule); everything else bf16 as in the original incl. the 128 PLE shards; its MTP head bf16 with **per-expert
`.weight`** experts. `expected_names(tc, "intel")` equals its 223947 text tensors.
`quantization_config`: `{bits 4, group_size 128, sym true, data_type int, packing_format auto_round:auto_gptq,
quant_method auto-round, autoround_version 0.15.0, block_name_to_quantize model.language_model.layers,
static_attention_granularity / static_kv_granularity tensor}`; `extra_config` 2340 entries, **all `bits: 16`**
(2110 `data_type fp`, 230 `float`; patterns `.*embed_tokens.*`, `.*hyper_connection.*`, `.*in_proj_a/b.*`,
`.*indexer.*`, `.*linear_attn.*`, `.*mlp\.gate.*` (+ one per layer), `.*mtp.*`, `.*ple.*`, `.*self_attn.*`,
`.*shared_expert.*`, `.*visual.*`, the final mixer's down / up, `mtp.fc_*`). By reading `src/loader/quant.cc`
(:330-410; not run): bits 4, g128, sym, no `desc_act`, packing `auto_round:auto_gptq`, method `auto-round`
and `bits: 16` extra rules are all accepted - `LinearSrc::classify` expands g128 to g64 at load
(`src/loader/quant.h:49-60`).

**The PLE table** (both): one PLE layer (`ple_layer_ids [2]`, one-indexed: layer_idx 1, a GDN layer); 16 heads
x 160; `seed` absent from the config -> the default 1234 (C:155). The three I64 tensors (range-fetched from both
checkpoints) **equal the formula**: `layer_multipliers` [23703573157769, 20109073645365, 8052911324071] (odd;
max product 5,886,047,582,964,040,311 < 2^63 - 1), primes and offsets:

| head | n-gram | prime (rows) | offset | first row (shard:row) | last row (shard:row) |
|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 20000003 | 0 | 0:0 | 7:2499918 |
| 1 | 2 | 20000023 | 20000003 | 7:2499919 | 15:2499845 |
| 2 | 2 | 20000033 | 40000026 | 15:2499846 | 23:2499782 |
| 3 | 2 | 20000047 | 60000059 | 23:2499783 | 31:2499733 |
| 4 | 2 | 20000059 | 80000106 | 31:2499734 | 39:2499696 |
| 5 | 2 | 20000063 | 100000165 | 39:2499697 | 47:2499663 |
| 6 | 2 | 20000069 | 120000228 | 47:2499664 | 55:2499636 |
| 7 | 2 | 20000077 | 140000297 | 55:2499637 | 63:2499617 |
| 8 | 3 | 20000081 | 160000374 | 63:2499618 | 71:2499602 |
| 9 | 3 | 20000093 | 180000455 | 71:2499603 | 79:2499599 |
| 10 | 3 | 20000107 | 200000548 | 79:2499600 | 87:2499610 |
| 11 | 3 | 20000147 | 220000655 | 87:2499611 | 95:2499661 |
| 12 | 3 | 20000153 | 240000802 | 95:2499662 | 103:2499718 |
| 13 | 3 | 20000159 | 260000955 | 103:2499719 | 111:2499781 |
| 14 | 3 | 20000161 | 280001114 | 111:2499782 | 119:2499846 |
| 15 | 3 | 20000171 | 300001275 | 119:2499847 | 127:2499921 |

Total 320,001,446 rows, padded to a multiple of 128: 320,001,536 = 128 shards x 2,500,012 (the last 90 rows are
padding, never addressed). Every head spans 8 or 9 shards and no shard boundary is a head boundary: the int8
file (21b) stores per head, so the reader maps global -> head-local rows (`PleTable` does both forms).
`split_ngram_parts` is 128 in both configs (the class default is 512, C:156).

**The MTP head** (`mtp.*`, 31 tensors in the original): `fc_embedding`, `fc_hidden` [2560, 2560];
`pre_fc_norm_embedding` [2560], `pre_fc_norm_hidden` **[10240]**; its own `hyper_connection_mixer` (norm,
down, up - **no `block_inject_weight` in either checkpoint**: vLLM's mapper drop rule
`hyper_connection_mixer.block_inject_weight -> None`, `V/nvidia/mtp.py:351 / 438`, is inert here);
`mtp.layers.0.*` = one QSA layer's tensors with its own indexer, both HCs with inject weights, router [512, 2560],
shared expert, experts fused [512, 1280 / 2560, ...] in the original (5.033 GB bf16), per-expert `.weight` in
Intel's. Derived: the head's non-expert tensors are 0.181 GB bf16; with int4 g64 experts (1.337 GB) the head is
**1.52 GB** keeping the rest bf16, ~1.44 GB with q / k / v / o and the shared expert at int4 too (spec 21 §1's
"~1.4 GB"); 5.21 GB all bf16.

## The MTP head's semantics (vLLM, decision 1) - two findings

- Wiring (`V/nvidia/mtp.py`): `e = fc_embedding(pre_fc_norm_embedding(embed(t)))` (:293-294);
  `h = fc_hidden(pre_fc_norm_hidden(R.flatten) viewed [4][2560])` - **one RMS over all 10240** (GemmaRMSNorm
  over `hidden_size * hc_count`, :227-229, :302-305) then the shared 2560 x 2560 projection per stream;
  `prev_block_output = e`, `prev_injection = None` (:307, :317): the layer's attn-side combine adds `e` to every
  stream at **unit weight**; the decoder layer is built with `layer_type="qwen_sparse_attention"` (:213, PLE
  off); the final `hyper_connection_mixer.combine_and_mix` returns the single stream (logits) **and** the
  materialised pre-mixer 4 streams (:336-340) - the next draft step's `R`.
- **Decision 5 ("steps after the first reuse step 0's selection") is vLLM's opt-in mode, not its default.**
  `set_skip_topk` (:258-261) is toggled by the proposer only when `index_share_for_mtp_iteration` is true
  (`v1/spec_decode/llm_base_proposer.py:578-613`, `:1592-1603`); that flag comes from the speculative config
  option of the same name, default `None` = the draft HF config's value (`config/speculative.py:442-444`,
  `:845-864`), and Qwen3.8-Flash-Next's `config.json` does not carry it. So vLLM's default drafts with a
  **fresh selection per step**. The port implements both (`qwen4exp_mtp.chain(share_sel=True)` the ruled form,
  `False` vLLM's default); the operator may want decision 5 re-read. In the shared mode the reused list is the
  step-0 row's positions verbatim (`indexer_qsa.py:260-261, 409-412`): a later step's query attends neither its
  own key nor the earlier draft keys.
- `pre_fc_norm_hidden` "per_stream" (decision 4's alternative) reads the same [10240] weight as 4 groups of
  2560; the two forms give different logits on random weights (`test_norm_forms_differ`).

## Tokens and the template

| | |
|---|---|
| `generation_config.json` (both) | `do_sample` true, T 1.0, top-p 0.95, top-k 20, `eos_token_id` [248046, 248044], bos / pad 248044 |
| text-config EOS | 248044 (`<|endoftext|>`): the PLE's EOS (M:1086) |
| `chat_template.jinja` (both) | sha256 **`c3cf9e34...`** = Qwen3.8's / Agnes's (`src/tokenizer/chat_template.cc:142`); the original's `tokenizer_config.json` also carries it as a `chat_template` string, Intel's does not |
| `tokenizer.json`, original | sha256 `0997f410...`, 12.8 MB: vocab 248044, merges 247587, added 33 (ids 248044-248076), 248077 ids |
| `tokenizer.json`, Intel's | sha256 `06b95093...` = **Qwen3.8's file byte for byte** (`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`'s, 20.0 MB) |
| original vs Qwen3.8's | vocab, merges, added tokens, normalizer, post_processor **equal**; the **pre-tokenizer differs**: the split regex adds `\p{M}` (combining marks) - `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` and ` ?[^\s\p{L}\p{M}\p{N}]+` where Qwen3.8 has `\p{L}+` and `[^\s\p{L}\p{N}]+` - and the ByteLevel pre-tokenizer's `trim_offsets` is false; the ByteLevel decoder's flags differ (`add_prefix_space` / `trim_offsets` / `use_regex` false vs true: no effect on ids) |

So the original tokenizes text with combining marks (decomposed accents, Indic / Thai vowel signs) differently from
Qwen3.8 (and from Intel's copy); on text without them the ids are the same. The golden prompts are re-checked, not
assumed (Task 5, `qwen4exp_prompts.py`): see "The golden prompts" below. 21e's tokenizer gate has to use the
**original's** `tokenizer.json` (and the engine's tokenizer must implement `\p{M}` in the split).

## torch's CPU `topk` on exact ties (Task 2 Step 4)

Measured (`qwen4exp_ref.topk_ties`, printed by `qwen4exp_ref.py facts`; torch 2.14.1+cpu in `agnes-ref-img`): 50
trials each, an exact tie planted between the k-th and (k+1)-th largest values -

| | the lower index wins the tie | repeat calls identical |
|---|---:|---:|
| QSA: `topk(512)` over 600 fp32 block scores (M:749) | **29 / 50** | 50 / 50 |
| router: `topk(10)` over a 512-way fp32 softmax (M:965) | **25 / 50** | 50 / 50 |

torch's CPU `topk` is deterministic but does **not** prefer the lower index: which tied entry survives depends
on the values' positions in its partition. The tiny-model test agrees (`test_route_ties`: 16 experts top-4 with
two identical router rows - the lower id won 50 of the 71 (row, layer) where the pair straddled the cut). So the
engine's ruled rule (exact ties to the lower block / expert id, spec 21 §4.2 / §4.4, decision 3) and the reference
differ **only on exact ties**; the reference records them (`route.gap == 0`, `qsa.gap == 0` in the golden files,
the `==0` count in the logs' gap lines) and the gates treat those rows as undetermined (docs/14's rule).

## Derived bytes (spec 21 §3, from the headers)

| part (per decoded token) | format | GB |
|---|---|---:|
| routed experts (48 x 10 x 3 x 2560 x 640) | int4 g64 | 1.253 |
| shared experts | int4 g64 | 0.125 |
| routers (512 x 2560) | bf16 | 0.126 |
| hyper-connections (96 + the final mixer) | bf16 | 1.281 |
| GDN projections (int4) + a / b (bf16) | int4 g64 + bf16 | 1.121 |
| QSA projections (int4) + indexer (bf16) | int4 g64 + bf16 | 0.357 |
| PLE projections (key 10240 x 2560, value 2560 x 2560) | bf16 | 0.066 |
| `lm_head` (spec 9: int8 + an f32 scale a row) | int8 | 0.637 |
| **total** | | **4.966** |

590 GB/s / 4.966 GB = 118.8 t/s roofline (derived) - spec 21 §3's ~4.97 / ~119 confirmed from the headers. The
planner's per-layer bytes with Intel's interim forms (21b Task 2): routed experts **1,336,934,400 B** a layer
(int4 g64 after the exact g128 expansion: 512 x 3 x 870,400), GDN dense bf16 115,917,344 B, QSA dense bf16
102,893,056 B, HC + router + shared 38,876,160 B: **a GDN layer 1.492 GB, a QSA layer 1.479 GB** (derived;
spec 21 §3's ~1.49).

## The golden prompts (Task 5)

`tools/oracle/qwen4exp_prompts.py oracle-out-q4exp/orig-small` (the original's tokenizer.json /
tokenizer_config.json / chat_template.jinja; run in `agnes-ref-img` with the 5.19.0 site), measured:

- Qwen3.8's committed ids **are valid** for this model on our prompts: `prose.txt` (42), `code.txt` (61) and
  `cjk.txt` (38) re-encode to the committed ids exactly; `long32k.ids` decoded and re-encoded equals itself on
  all 32768 ids (none of these texts has a combining mark). (`long.ids` is a concatenation of id lists -
  `make_long_prompt.sh` - not `long.txt`'s tokenization; both tokenizers give the same 3412 ids on `long.txt`.)
- The set (`tests/golden/prompts/`): `q4exp_short` = prose + code, **103** ids (QSA dense: the dense-attention
  gate); `q4exp_4k` / `q4exp_8k` / `q4exp_32k` = the first **4096 / 8192 / 32768** ids of `long32k.ids`;
  `q4exp_agentic` = `q4exp_agentic.json` (an original coding session on this repository: system prompt, a task
  comparing two of its headers, four assistant turns with reasoning and XML tool calls - read / read / grep /
  bash - whose results hold `src/runtime/kolibri/kolibri_sizes.h` and `src/loader/kolibri1_layout.h` verbatim)
  through `apply_chat_template(..., tools, add_generation_prompt=True, enable_thinking=True)`: **12253** ids.
  All four long prompts reach past position 2051.

## What needs the real weights (pending, box CPU: plan 21a Tasks 6-7, queue row 30)

- F1's third bullet: the reference's perplexity on Intel's checkpoint (`qwen4exp_oracle.sh <data> ppl`).
- `hfcheck --layers 4` on real weights (transformers' own indexer against the cache, bitwise).
- The golden sets and `--layers 4 / 18` sets with the gap distributions: decision 3's tau (QSA 512th / 513th)
  and R2's MoE tolerance come from those lines.
- The routing traces (spec 22 P0.6 / P0.8).
