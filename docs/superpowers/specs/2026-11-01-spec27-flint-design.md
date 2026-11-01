# Spec 27 - Recursive Flint (`flint`) on the B70: the engine

**Status:** design, 2026-11-01, for operator review (operator approval to write it: 2026-11-01); reviewed 2026-11-01 (corrections
listed at the end). **Decided (operator, 2026-11-01): D3 - Flint's context is 262,144 (its vendor limit) with int8 KV; no option
beyond it.** The remaining open decisions are collected in section 13. Nothing is built; no code, kernel or card number exists for Flint yet. Supersedes the watch note
`2026-10-23-spec27-flint-watch.md` (which now points here).

Plans: `docs/superpowers/plans/2026-11-01-spec27{a,q,b,c,d,e,f,g}-*.md` (section 10).

**Numbers.** Every number is **published** (config / model card), **read** (the HF files read on 2026-11-01, named in section 1),
**derived** (a formula is given), or **estimated** (marked, with the formula or the reason). Nothing here is measured on the card.

## 0. The brief and the standing rules it meets

The operator judges everything by: usable context >= 242k, generation speed, decent quality, correct tool calls in long agentic
coding sessions. For Flint the context is **decided at 262,144 with int8 KV** (D3, operator 2026-11-01: the vendor limit; the
"330k+" target belongs to K2-Horizon, not to Flint). Standing rules that shape this spec:

| rule | effect here |
|---|---|
| only AutoRound int4 g64 sym | the operator's `urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ` is the one format; no third-party quant support |
| no host-RAM offload on the per-token path | everything resident: 21.8 GB on one card (section 3) |
| minimal options / env vars | capture rules instead of switches: KV form, attention arm, chunk size are chosen from the shapes (section 7) |
| gates against an f32 reference | MoE: engine no further from f32 than the bf16 golden; near-miss rule (`tests/golden/forced_judge.h`: every extra miss against the bf16 twin at most 8 bf16 ulp below the f32 top, and the logits cosine distance of the min row and of the median row each at most 2x the twin's) |
| eager paths diagnostic | the eager attention / eager MoE arms exist to be compared with, never the default |
| prefill default `l0` | the l0 backend, no sycl-tla |
| G0 bitwise for other models untouched | no existing `.cl` is edited; new kernels are new files (copy-and-adapt); existing binaries stay sha256-identical |
| commit after each step; one implementing agent per plan | the plan format of 25a-25e |
| reuse ruling (2026-10-27) | patterns are copied into **new files with generic names**, never bound or included: the sink term from LiLiCorr's `ll_attn.cl` reduce (`99adbb0^:src/kernels/dflash/ll_attn.cl:170`; the operator is restoring LiLiCorr on branch `bolt-spec-restore` (2026-11-01), so the file may be back on main - copy from whichever copy exists, depend on neither); the fp32 sigmoid router with renormalisation from Bolt's `nb_route` (`src/kernels/bolt/nb_moe.cl`, on main), so LFM2.5's dropped `lfm_moe.cl` is not needed |

## 1. The model (read 2026-11-01)

Read from: `RecursiveSI/flint` `config.json` (sha256 prefix `a57c33d11f725099`), `generation_config.json`, `tokenizer_config.json`,
`chat_template.jinja`, `model.safetensors.index.json`; transformers PR #49824 at `0ea108daf169dd66a1800e48a56d2251d4ff76bd`,
`src/transformers/models/flint/modeling_flint.py` (683 lines, sha256 prefix `869d318cec68654a`, cited below as `T:line`);
`urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ` `config.json` and index. The Mac has no local transformers checkout
(`~/PycharmProjects/transformers` does not exist) and no HF cache copy: plan 27a fetches and pins them. Also read in review
(2026-11-01): transformers' `conversion_mapping.py` at the same commit, the shard headers of both repos (HTTP range reads) and
sampled `qzeros` / `softmax_offset` tensors.

| | value |
|---|---|
| class / type | `FlintForCausalLM`, `model_type: flint`; Apache 2.0; BF16 original ~80 GB |
| layers | 49: layers 0-1 dense SwiGLU (I 6144), layers 2-48 MoE (`mlp_only_layers [0,1]`, `decoder_sparse_step 1`) |
| hidden / vocab | 2048 / 150,144; **untied** `lm_head` (`tie_word_embeddings false`); `pad_token_id 150001` |
| attention | 32 q / 4 kv heads, head_dim 128 (GQA 8), no bias, scale 128^-0.5 |
| per-head q / k RMSNorm | on the head dim, **before** RoPE (T:attention forward) |
| layer types | `layer_types`: full attention at **0,1,5,9,13,17,21,25,29,33,37,41,45,48 (14)**, sliding at the other 35 (2,3,4,6,7,8,10,...,46,47). The pattern is not periodic (leading 2 full, trailing 48 full): an explicit per-layer table |
| positions | sliding layers: RoPE neox (rotate_half), theta 1e4, window 513 (the current token and 512 before it); full layers: **NoPE** (`no_rope_layer_indices` = the same 14). `max_position_embeddings` 262,144 - the engine's limit (D3) |
| sinks | learned per-head scalar on **all 49 layers**: an extra softmax column attending to nothing (T:eager_attention_forward); checkpoint tensor `self_attn.softmax_offset` [32], **bf16** in both checkpoints, renamed to `sinks` at load by a pure `WeightRenaming` (`conversion_mapping.py`, `mapping["flint"]`), so it is stored as applied, widened exactly to the module's fp32; sampled values -9.0 .. 10.25 (layers 0, 14, 26, 38); config `softmax_type: learnable` |
| scores | `q.float() @ k.float()^T * scale` in fp32, softmax in fp32 over [scores, sink], the sink column dropped, **probabilities rounded to bf16**, then bf16 matmul with V (T:eager_attention_forward) |
| norm (all sites, incl. q / k) | `w * (x * rsqrt(mean(x^2) + 1e-6))` computed in fp32, **one** rounding to bf16 - `x_hat` is not rounded first (T:FlintRMSNorm; "main diff with Llama") |
| block | plain pre-norm: `h += attn(norm(h)); h += mlp(norm(h))` (T:FlintDecoderLayer); no post-norms |
| MoE | 256 routed experts, top-6, expert I 512, SiLU; shared expert I **1024**, always added, **ungated**; the sum `routed + shared` is one bf16 add |
| router | `gate.weight` kept fp32 in transformers (`_keep_in_fp32_modules_strict = ["gate.weight","sinks"]`) but **stored bf16** in both checkpoints (headers read), so the widening is exact: logits `F.linear(x.float(), w.float())` fp32, never rounded; scores = sigmoid(logits) fp32; **top-6 on the sigmoid scores, no bias** (`torch.topk`; its order among exactly equal scores is not specified by torch - 27a records what the pinned CPU build does, not "lower id" by assumption); weights = scores / sum(selected) fp32 (`norm_topk_prob true`; the sum runs over `topk`'s output, which is in descending-score order - 27a pins torch's summation order as 25a did for Bolt), no epsilon, no scaling factor (T:FlintTopKRouter) |
| expert combine (eager loop) | per expert: `(down(act(gate x) * up x))` bf16, times the fp32 weight, rounded to bf16, `index_add_` into a bf16 accumulator **in ascending expert id**; then `+ shared` in bf16 (T:FlintExperts, FlintSparseMoeBlock). `@use_experts_implementation` may pick a grouped / batched implementation instead: the reference forces the eager loop (27a trap) |
| rope tables | `cos`, `sin` computed fp32 from `position * inv_freq`, cast to bf16; q/k rotated in bf16 as `q*cos + rotate_half(q)*sin` with each product and the sum rounded (T:apply_rotary_pos_emb) |
| tokens | BOS `<|begin_of_text|>` 150000, never added automatically (`add_bos_token false`; the template does not emit it); EOS ids **150001** `<|end_of_text|>`, **150003** `<|user|>`, **150005** `<|observation|>`, **150018** `<|developer|>`; `<|system|>` 150002, `<|assistant|>` 150004, `<think>` 150006 / `</think>` 150007, `<tools>` 150008-9, `<tool_call>` 150010 / `</tool_call>` 150011, `<arg_key>` 150012-3, `<arg_value>` 150014-5, `<tool_response>` 150016-7 |
| sampling defaults | `generation_config.json`: temperature 1.0, top-p 0.95, top-k 20 |
| MTP / draft head | **none** in the config, the modelling code or the checkpoint index (no `mtp.*` keys) |

**Chat template** (`chat_template.jinja`, read): `<|system|>`, `<|user|>`, `<|assistant|>`, `<|developer|>` role tokens followed
directly by the text, no end-of-turn token (the next role token ends the turn). Tools: one `<|system|># Tools ... <tools>` block with
one JSON function signature per line, emitted before the first non-system message. An assistant turn: `<|assistant|>` + optional
`<think>reasoning_content</think>` + content + calls `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`
(V is the raw string for string values, `tojson(ensure_ascii=False)` otherwise). A tool result: `<|observation|><tool_response>TEXT</tool_response>`.
System, developer, user and tool text (the `flint_text` macro) has NUL, `<think>` and `</think>` removed and the six role /
BOS / EOS strings escaped (`&lt;|user|&gt;`); assistant content and reasoning are emitted verbatim. The generation prompt is
`<|assistant|>` with no `<think>` prefix: the model opens its own reasoning. **A model turn ends on 150003 / 150005 / 150001 /
150018** (the `eos_token_id` list of both repos' `generation_config.json`): after a tool call the next rendered token is
`<|observation|>` (150005) - inferred from the template and the EOS list, to be confirmed on the reference's own generations in 27a -
so the server stops on it and the call is the turn's result.

**The checkpoint as shipped (urakozz quant, index read).** 5 shards, 22,381,534,784 bytes, 109,612 tensors. AutoRound 0.17.0, `bits 4,
sym true, group_size 64, packing_format auto_round:auto_gptq`, `block_name_to_quantize model.layers`; `extra_config` keeps `mlp.gate`
of layers 2-48 at 16-bit float. Per layer: `self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales}`, `q_norm.weight`, `k_norm.weight`
(both [128]), `softmax_offset`, `input_layernorm.weight`, `post_attention_layernorm.weight`; layers 0-1 `mlp.{gate,up,down}_proj.*`
(quantised); layers 2-48 `mlp.gate.weight` (router, unquantised), `mlp.shared_expert.{gate,up,down}_proj.*` (quantised) and
**per-expert unfused** `mlp.experts.E.{gate,up,down}_proj.*` (47 x 256 x 9 = 108,288 tensors). Outside the layers: `model.embed_tokens.weight`,
`model.norm.weight`, `lm_head.weight` (bf16, in the last shard). The original `RecursiveSI/flint` index (15 shards, 79.57 GB, 36,734
names) also carries per-expert unfused `gate_proj` / `up_proj` / `down_proj` weights (transformers fuses them into `gate_up_proj` at
load, the `qwen2_moe` conversion mapping), so the reference reads either form.
**Read in review (2026-11-01, shard headers by range):** every tensor of the original is bf16; in the quant, `mlp.gate.weight` [256, 2048],
`softmax_offset` [32], all norms, `embed_tokens` and `lm_head` are **bf16**, scales f16, qweight / qzeros i32; sampled `qzeros` (20 tensors
across all 5 shards, dense, attention, routed) are all 0x77777777 (the engine's rule, `src/loader/quant.cc:526-535`); the sinks are stored
as applied (a pure rename, above). 27a still scans every `qzeros` word on the box copy.

## 2. Architecture mapping, per op

Family decision: **Flint is its own engine family** (`model_type flint`, like Kolibri / K2 / Bolt), not a `ModelDesc` (Qwen-family)
entry. The Qwen-family loader cannot express dense-first layers, a non-periodic layer table, a window, sinks, head_dim 128 or vocab
150,144 (`Qwen35::kVocab` has 53 uses in 19 files of `src/`, 183 with the tests). Template: **Kolibri's** structure (`src/model/kolibri1.*`, `src/loader/kolibri1_*`,
`src/runtime/kolibri/*`, `src/kernels/kolibri/*`), generalised where Kolibri hard-codes its dims.

"Reuse" = the existing binary / function bound by name with new CMake lines only. "Adapt" = a **new file** copied from the named
source and changed (no existing `.cl` edited: G0). "New" = written here.

| op | what Flint needs | source | verdict |
|---|---|---|---|
| embedding gather | bf16 row, 150144 x 2048 | `embed_gather.cl` | reuse (new variant name `D2048_V150144`) |
| input / post-attention norm, final norm | pre-norm, `w * x_hat` **one** rounding, residual fold | Kolibri's `kol_norm_finish` rounds `x_hat` first (`kol_prep.cl:41`) and has post-norms; **K2's `k2_norm_finish` (`k2_prep.cl:63`) is already the one-rounding form** `rne(x * rstd * w)`, eps 1e-6, K a define - at `NORM_GROUPS=1` it is Flint's norm exactly; Bolt's `nb_res_norm` is the same norm fused with the fold (K 4096, eps 1e-5 baked) | **reuse** `k2_norm_finish` at `K=2048 NORM_GROUPS=1` (new variant name) after `prep_res_fold`; a fused fold+norm (`nb_res_norm`'s pattern, copied) only if 27c's P0 says launches matter; drop `kol_post_add` |
| q / k / v projection | one fused GEMV, N = 4096 + 512 + 512 = 5120, K 2048, int4 layout 1 | `gemv.cl` (`gemv_variant`) | reuse; new shape, split-K row in the tuning table |
| q / k head norm, RoPE (sliding only), cache write | per-head norm `w * x_hat` one rounding, then neox RoPE `rne(rne(y*c) + rne(rot*s))`, ring write for sliding, linear write for full | `kol_attn_prep` (same RoPE chain, ring vs linear) but its norm rounds `x_hat` first | **adapt**: `fl_attn_prep` |
| attention decode | GQA 8, hd 128, window 513 ring or linear NoPE, **sink** | `kol_attn.cl` (window, ring, hd 128, no sink, M <= 4) | **adapt**: generic `sink_attn_decode` + `sink_attn_reduce` (sink added once in the reduce, denominator only, after the block merge, also when a block is skipped by the window; exp2 build: sink x log2 e) - the pattern of LiLiCorr's reduce, `l = l + exp(sink[h] - m)` once after the blocks and keys (`99adbb0^:src/kernels/dflash/ll_attn.cl:170`, or the restored copy if `bolt-spec-restore` lands; copied, never bound) |
| attention decode, eager arm | scores fp32, softmax with sink, bf16 P, bf16 PV | `kol_attn_eager.cl` (bf16 scores) | **adapt** (diagnostic arm; matches the reference's rounding exactly) |
| attention prefill | flash with window skip, NoPE, sink in the finalise (`lsum += exp(sink - m)` before `1/lsum`) | `kol_pf_attn.cl` (window skip, hd 128) | **adapt**: `sink_pf_attn` |
| o projection | K 4096, N 2048 | `gemv.cl` | reuse |
| dense MLP layers 0-1 | gate+up GEMV (N 12288), SiLU-mul, down GEMV | `k2_silu_mul` (parametric, K2 instantiates I = 6144, `CMakeLists.txt:744`), `gemv.cl` | reuse (prefill: `pf_gemm_silu` / `pf_gemm`) |
| router logits | fp32 logits from bf16 `x` and the bf16 router matrix (header read: bf16 in both checkpoints; transformers widens it to fp32 exactly) | `gemv_bf16` emits fp32 logits, never rounded (Kolibri's and Bolt's router use) | reuse `gemv_bf16` at N 256, K 2048 (new variant name); no fp32-weight variant is needed |
| route | fp32 sigmoid, top-6 on the **scores** (ties are real: fp32 sigmoid saturates at 1.0, so distinct logits can tie - selecting on the logit, as `kol_route` does, would break such ties differently; the tie order is the pinned reference's, recorded by 27a), `w = s / sum(s_selected)` fp32, no bias, no epsilon, slots sorted by ascending id | **Bolt's `nb_route`** (`nb_moe.cl`, on main: fp32 logits never rounded, fp32 sigmoid, renormalised; but a selection bias, `+ 1e-20`, `x 5.0` at `nb_moe.cl:157`, top-9 hard); `kol_route` (fp32 logits, but selects on logit + bias, no renormalisation); `k2_route` (bf16-rounded logit `k2_moe.cl:159`, bf16-rounded w `:198`) | **adapt from `nb_route`, generic name** `route_sigmoid_renorm`: top-6, no bias argument, no epsilon, no scale, no rounding of logit, score or w |
| routed experts (decode) | 256 blocks, gate||up K 2048 N 1024, down K 512 N 2048, int4 layout 1, 6 slots | `moe.cl` family shapes (Ornith has E256 D2048 I512; TOP_K 8 hard in the binary) | **adapt**: the Kolibri / K2 `moe_gate_up` / `moe_down` pair at `E256 T6 D2048 I512` |
| shared expert | I 1024 (twice the routed width), ungated, int4 | the fused MoE kernels tie shared I to routed I (`kol_moe.cl:23-24`, `moe.cl:73-74`, `k2_moe.cl` `SHARED = MOE_E` at `INTER`, `model_desc.cc:424`); **Bolt runs its shared expert as plain GEMVs** (`bolt_capture.cc:355-388`: fused up GEMV, `nb_sh_act`, `shared_down` GEMV, folded by the next norm) | **reuse** the dense path: gate‖up `gemv.cl` (N 2048, K 2048), `k2_silu_mul` at `SILU_I=1024`, down `gemv.cl` (N 2048, K 1024) - no new kernel, 3 launches a MoE layer. Option for 27c (decided by P0): the shared expert as **two extra 512-wide slots** of the routed pair (gate / up / SiLU-mul split exactly at i = 512; down's two slots summed in fp32 before its one rounding), which removes those 3 launches |
| expert combine | per expert `rne(rne(down) * w)` (w fp32 unrounded), bf16 sequential add ascending id, then `+ shared` bf16, then residual | `k2_moe_down`'s arithmetic **is** this chain (`k2_moe.cl:317`: `rne(f32(d_b) * w)` with w read as fp32 bits, ascending-id bf16 adds, `rne(acc + sh)`, `rne(resid + o)`); the bf16 w comes from `k2_route` (`:198`), not from the down kernel - fed an unrounded fp32 w it computes Flint's chain | **adapt** `k2_moe_down` into `fl_moe_down` only for the shared expert's width (separate dense shared: the routed sum to its own buffer and a fold `rne(h + rne(r + sh))`; or the two-slot option) |
| lm head | int8 row-scaled head (spec 9) from the bf16 checkpoint head, fp32 sampler argmax | `gemv_i8w.cl`, `argmax.cl`, `lm_head_int8.cc` | reuse; new vocab variant names |
| prefill MoE | route -> sort -> gather -> dequant -> grouped GEMM -> combine | `pf_moe.cl` (Ornith's allows `EXPERTS <= 256`, `pf_moe.cl:60`; K2's `k2_pf_moe.cl` allows only `PF_E <= 255`, `:60`, ids staged as bytes); Ornith's combine is fp32 rank order + gated shared, K2's `k2_pf_moe_combine` is the ascending-id bf16 chain | **adapt**: Ornith's `pf_moe.cl` sort / gather / grouped GEMM (256 passes its `#error` checks) with `k2_pf_moe_combine`'s chain (unrounded fp32 w) and the shared expert as a dense block added after the routed sum |
| prefill linears | l0 GEMM | `pf_gemm*` | reuse |
| KV cache | full layers linear [max_len][4][128], sliding layers ring | Kolibri buffers (`kolibri_buffers.*`) | **adapt** with a per-layer table |
| int8 KV (full layers) | per (position, kv head) f16 scale, 128-dim Hadamard (K2's hd128 form) | `k2_kv8.cl`, `src/common/kv8.h` `hd128` | **adapt** for the sink reduce and the ring-less linear cache (plan 27f) |
| descriptor | per-layer table (type, dense or MoE), no hard-coded periodicity | `Kolibri1Desc` has `l % 5 == 4` and a `require` that refuses any other dims (`kolibri_capture.cc:73-76`) | **new** `FlintDesc` held to `config.json` key by key |
| loader | per-expert unfused AutoRound tensors, 109,612 names | `kolibri1_loader`, `moe.cc` (separate gate/up per expert) | **adapt** |
| capture | one list per card, replayed | `kolibri_capture.cc` | **adapt** (dense branch for layers 0-1, no post-norm, shared expert, sinks) |
| prefix cache / snapshots | rings' last 512 rows + full-layer blocks | `kolibri_engine` save / load, `prefix_cache.h` | **adapt** |
| template, tool parser | role tokens, GLM-style call tags | `toolcall_k2.cc` (state machine, tags `<ifm|...>`), `chat_format.*` | **adapt**: tag table parameter (section 8) |

What is new about the model compared with everything shipped (main at 1e13e27): **sinks** (no kernel on main has one; LiLiCorr's
did until its 2026-10-27 drop), the **renormalised fp32 sigmoid route without bias / epsilon / scale** (Kolibri and Bolt keep fp32
logits, K2 and Ornith round them; none has exactly this chain), the **2x-wide shared expert** in a fused MoE kernel (Bolt's dense-GEMV
shared path avoids the problem), **dense-first layers**, and a **non-periodic layer table**. The one-rounding norm is **not** new
(K2's and Bolt's norms already have it; only Kolibri rounds `x_hat` first). Each is a trap with its own test in 27a / 27c (section 9).

## 3. What it costs on a card (derived)

Parameters (P = parameters): attention 49 x (2048x4096 + 2 x 2048x512 + 4096x2048) = 0.925 B; dense 2 x 3 x 2048 x 6144 = 0.0755 B;
routed 47 x 256 x 3 x 2048 x 512 = 37.85 B; shared 47 x 3 x 2048 x 1024 = 0.296 B; routers 47 x 256 x 2048 = 0.0246 B; embedding and head
2 x 150144 x 2048 = 0.615 B. Total 39.79 B (matches the published 39.8 B). Int4 g64 with f16 scales = 4.25 bit = 0.53125 B / P.

| part | formula | GB |
|---|---|---:|
| routed experts int4 | 37.85 B x 0.53125 | 20.11 |
| attention int4 | 0.925 B x 0.53125 | 0.49 |
| dense layers int4 | 0.0755 B x 0.53125 | 0.04 |
| shared experts int4 | 0.296 B x 0.53125 | 0.16 |
| routers bf16 | 0.0246 B x 2 | 0.05 |
| embedding bf16 | 0.3076 B x 2 | 0.62 |
| lm head int8 (+ row scales) | 0.3076 B x 1 | 0.31 |
| norms, sinks, rope table | small | ~0.01 |
| **resident** | | **~21.8** |

**Bytes read per token** (M = 1, int4 + int8 head): active int4 = (0.925 + 0.0755 + 47 x 6 x 3 x 2048 x 512 = 0.887 B + 0.296) = 2.18 B P
x 0.53125 = 1.16 GB; routers 0.049 GB (read in full); head 0.31 GB; **~1.52 GB** (the brief's figure). Not in it: the 35 sliding
layers' window, read every token once the context passes 513 (35 x 513 x 2048 B = 0.037 GB, +2.4 %). Roofline at 638 GB/s: 638 / 1.52 =
**420 t/s** at zero depth - a ceiling; the achieved fraction is unknown until 27c's P0 (spec 25c's method).

**KV.** Only the 14 full layers grow: per position 14 x 4 x 128 x 2 (K, V) = 14,336 values: **28,672 B bf16**; int8 with a f16 scale per
(position, kv head, K or V): 14,336 + 14 x 4 x 2 x 2 = **14,560 B**. The 35 sliding layers hold a ring: ring 4096 slots (a power of two >= prefill
chunk 2048 + window, Kolibri's rule) x 35 x 2048 B = **0.29 GB** bf16 (kept bf16 under int8 KV, decide D4); the live window state a snapshot needs is
35 x 512 x 2048 B = 36.7 MB.

**One card (Level Zero reports 32.53 GB, `src/cli/kolibri_decode.h:62`; the planner measures the real figure).** Free for KV = 32.53 -
21.8 (weights) - 1.5 (the planner's default reserve) - ~1.0 (prefill scratch: dequant batch, chunk buffers; estimated, 27d measures) -
0.29 (rings) = **~7.9 GB** (estimated).

| KV form | bytes / position | positions that fit = 7.9e9 / B | at 262,144 (D3) |
|---|---:|---:|---:|
| bf16 | 28,672 | ~277k | 7.5 GB (marginal: eats the scratch margin) |
| **int8** | 14,560 | ~545k | **3.8 GB** (~4.1 GB spare) |

So **one card holds the decided 262,144 with int8 KV with room** (the spare ~4 GB is headroom for a batching slot or a lookup verify
buffer, not for a longer context), and bf16 KV at 262,144 only marginally. **The context is capped at 262,144 by decision (D3):** max-len
auto = min(262,144, what memory fits), and `--max-len` above 262,144 is refused by name.

**Two cards, `--pp 2`** (spec 16's layer split by bytes): ~10.9 GB of weights each (embedding on card 0, head and final norm on card 1), and each
card holds its own full layers' KV (7 of 14). Free per card ~ 32 - 10.9 - 1.5 - 1.0 - 0.15 = ~18 GB: bf16 KV = 18e9 / 14,336 B (half the layers) =
~1.25 M positions, int8 ~2.4 M. Memory is not binding on two cards. **Pipeline stages run in sequence, so `--pp 2` adds a hand-off and no speed**
(the bytes still add up); its value is headroom (bf16 KV, batching slots, room for a drafter), not throughput. The one-card int8 path is the default
product; `--pp 2` is supported through the existing Kolibri-family pp code (27c / 27e gates) and refused nowhere by Flint itself.
(Load note: Qwen's `runtime::place_stages` loads the whole model on device 0 first, `pipeline_place.h:11-17`; the Kolibri-family path
loads each part on its own device, `kolibri1_loader.h:23-26`. Either fits Flint's 21.8 GB.)

**Decode bytes at depth L** (int8 KV, full layers read 14,560 B x L per token; sliding rings are small and capped):

| depth L | KV bytes read (int8) | total GB / token | roofline t/s = 638 / total | bf16 KV roofline (reference) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.52 | 420 | 420 |
| 32,768 | 0.48 | 2.00 | 319 | 259 |
| 100,000 | 1.46 | 2.98 | 214 | 145 |
| 262,144 | 3.82 | 5.34 | 119 | 71 |

(Sliding-window reads, +0.037 GB at every depth past 513, are left out of the table: they lower each row by ~1-2 %.)

Reading: at the depths of a long agentic session the **KV read, not the weights, is the dominant cost** (equal to the weights at ~104k tokens,
1.52e9 / 14,560). So int8 KV and the decode-attention-at-depth kernel (spec 10's split-K, measured 540 GB/s at 128k on Qwen) matter more than another
weight-side trick. **Achieved speed: an estimate with no MoE measurement behind it.** No MoE decode (Ornith, K2, Kolibri, Bolt) has a speed number on a
B70 (`docs/BENCHMARKS.md` says "nothing measured yet" for each; spec 28m §5 confirms). The only calibration on this box is dense:
Qwen3.8-27B decodes 15.54 GB a token at 29.45 t/s = 458 GB/s = **0.72 of 638** with 774 launches (`docs/BENCHMARKS.md`, decode record),
and the un-instrumented dispatch gap is **0.73-1.13 us a launch** (`docs/15-step-anatomy.md`). Applied to Flint: weights 1.52 GB at eta
0.5-0.72 = 3.3-4.8 ms, plus ~780 launches x 0.73-1.13 us = 0.6-0.9 ms, gives **~180-260 t/s at zero depth**; with the KV read at the
attention kernel's measured 540 GB/s (Qwen, spec 10, `docs/probe-decode-attn-2026-09-28.md`), **~120-150 t/s at 100k and ~80-90 t/s at
262,144** (estimated). Flint's kernels move ~2 MB a launch on average against Qwen's ~20 MB, and small launches reach a lower fraction
of the bandwidth on this card, so the real figure may fall **below** this range; 27c's P0 replaces it. Not a promise and not a bar.

**Launch count (estimated by Kolibri's structure):** per MoE layer ~16 launches (norm, qkv GEMV, prep, attn + reduce, o GEMV, residual fold, norm, router GEMV,
route, routed up, routed down, shared up, shared down, fold), per dense layer ~11, plus ~5 at the head: 47 x 16 + 2 x 11 + 5 = **~780** (Kolibri: 756, spec 20c).
27c's plan test pins the exact formula.

**Prefill (estimated).** FLOPs per token ~ 2 x 2.5 B = 5 GFLOP (about 11x fewer than Qwen3.8-27B's ~54). But the l0 MoE path dequantises int4 -> bf16 per
expert batch every chunk: for a 2048-token chunk all 256 experts are touched (48 tokens each on average), so per MoE layer the dequant moves 0.43 GB int4 in
and ~1.6 GB bf16 out; over 47 layers (read + write + the GEMM's read) ~ (20 + 75 + 75) GB = 170 GB per chunk = 0.27 s at 638 GB/s, a **fixed cost per
chunk**: rate ~ 2048 / (0.27 + 2048 x t_flop) with t_flop ~ 44 us (Qwen's measured 2.1k t/s = 0.48 ms a token at ~54 GFLOP, x 5 / 54) -> 2048 / 0.36 = **~5,700 t/s upper estimate at short depth**; at
128k depth attention dominates: 14 full layers x 32 heads x 128 is ~0.58 of Qwen3.8's attention work per token-depth, so ~1.5k t/s (Qwen's 915 t/s at 130k / 0.58;
estimated). A 200k-token first prompt is minutes; **the prefix cache and session continuation are therefore load-bearing**, not an optimisation (section 8).
Decide in 27d P0 whether a chunk > 2048 (the sort's `KC` is a compile-time define bounded by its SLM `sid[KC * TOP_K]` bytes; every shipped
prefill path uses `PrefillScratch::kC` = 2048) or an int8 / direct-int4 grouped GEMM is worth the work: the fixed 0.27 s is the lever.

## 4. Speculative decoding

| option | verdict |
|---|---|
| MTP head | **none** exists (no `mtp.*` in the index, no config key) |
| draft model | no small model shares the 150,144-token vocabulary; spec 19's arithmetic already rejected it for K2; rejected |
| DFlash | drafters are per target with taps; none exists for Flint and 19b-d have not passed on the card; **not now**; revisit if a Flint drafter is published (the standing ruling that 19b-d are built regardless stands) |
| prompt lookup (spec 19e, generic host matcher) | **the candidate**, plan 27g, P0 first |

Lookup arithmetic (estimated): verifying M tokens reads the union of their routed experts. With independent routing the expected union per layer is
`256 x (1 - (1 - 6/256)^M)`: M = 2: 11.9, M = 3: 17.6, M = 4: 23.2, M = 8: 44.3 experts (vs 6 at M = 1; consecutive tokens' routes are correlated in practice, so
this is an upper bound; a new `tools/spec/expert_union.py` (27g, does not exist yet) over Flint's own route records measures it). Expert bytes per step = union x 47 x 1.67 MB; non-expert
bytes 1.52 - 0.47 = 1.05 GB. M = 4 (K = 3): 1.05 + 23.2 x 47 x 1.67e-3 = **2.87 GB = 1.9 plain steps** of bytes at zero depth, less at depth because the KV
read is shared by all M rows (at 100k: (2.87 + 1.46) / (1.52 + 1.46) = 1.45). Spec 19e's recorded A4 replay gave 3.10 tokens per verify at K = 3 (Qwen traffic):
net **~1.6x at zero depth, ~2.1x at 100k**, if Flint's text repeats like Qwen's coding traffic (agentic coding re-quotes files heavily). That is the biggest single speed lever
available for this model, and it needs: an M = 2..5 MoE path (no shipped MoE kernel at M > 1 is gated; spec 28 excludes the expert GEMVs and spec 28m, on main since
2026-11-01, is the MoE small-M probe for every MoE engine - its Ornith rows at k = 6 cover Flint's routed shape H 2048 / I 512 / E 256, and
its union kernel U2 is the candidate M-row MoE path), a verify attention at M rows with sinks, and `truncate_to`. Plan 27g is therefore a P0 + a decision, built after 27a-f and only if the P0 clears the bar (section 9, S1).

## 5. Reference and oracle plan (27a)

- **Reference: transformers at the pinned merge commit** `0ea108da...` (or the release that contains `flint`, v5.21 estimated, when it ships; one pin, refused
  otherwise by name, the 25a rule). Built by `tools/oracle/qwen4exp_env.sh`'s site mechanism; layer-streamed (`tools/oracle/stream.py` `attach`: one block resident at a time,
  routed experts read lazily - only the selected ones - and dequantised from int4 by `stream.dequant_t` for the engine-format reference); the Mac's `agnes-ref-img` container
  for tiny runs, the box CPU (`tools/oracle/run_in_container.sh`, lock `/tmp/b70-cpu-model.lock`) for real weights.
- **Flint restatement** `tools/oracle/flint_ref.py` (the `kolibri_ref.py` pattern): restated ops, each tested bitwise against the pinned module it restates - `rms_norm` (one
  rounding), `route` (fp32 chain, ties planted), `combine` (ascending id, bf16 sequential), `attention` eager with sinks (probabilities rounded to bf16), the RoPE chain, the
  dense block; the **eager experts loop forced** (the reference must not take `grouped_mm`).
- **Three references, named:** `f32` (the whole model in fp32 on the dequantised weights: the judge, the `f32_oracle.sh` convention), `bf16 golden` (the pinned code in bf16 on the
  dequantised int4 weights: the engine-format reference the B3 golden runs compare to), `bf16 original` (the BF16 checkpoint, for gate Q only).
- **Synthetic checkpoint** `tools/oracle/flint_make_tiny.py`: the real config's structure at tiny width (e.g. 8 layers keeping the pattern types: full-NoPE dense, sliding dense, sliding
  MoE x2, full MoE, ..., hidden 256, 8 q / 2 kv x 32, E 32 top-6 -> scaled, shared 2x routed), variants: `bf16`, `fp32`, `ties` (two experts with identical router rows,
  and an expert pair saturating the fp32 sigmoid), `sinks-large` (sink values that dominate), `int4-g64` (RTN in the `auto_round:auto_gptq` bytes, the checkpoint's key layout incl.
  per-expert unfused tensors, signed scales allowed as AutoRound writes them); plus a real-width synthetic for the loader (`make_synth`, the 25b pattern).
- **Taps** (`dump_taps.py` convention): the residual after blocks 0, 1, 2, 12, 24, 36, 47, 48; route ids / weights per MoE layer; attention outputs; logits tail.
- **A4 set** (`tools/toolcall/make_set.py --from`, Flint's template and tokenizer) and its reference run on the box CPU (hours, detached, `tools/probe/detach.sh`).

## 6. The quantisation (27q)

The operator's quant exists and is the engine format; **27q does not re-quantise**. It (1) checks the export against the engine's loader rules (`tools/quantize/check_gptq_export.py`:
qzeros 0x77777777, g64, sym, `desc_act` absent, the extra_config gate rule, no `g_idx` surprises); (2) runs **gate Q** against the BF16 original on the box CPU: KL / top-1 on the golden
prompts, A4 pass count, perplexity on a long agentic transcript - **a recorded measurement**, the spec 23 §3.3 style; (3) records the AutoRound recipe from the model card
(iterations, calibration, `lm_head` / embedding kept bf16). Our own AutoRound run (tools/quantize_kolibri1.sh pattern) is only started if gate Q shows a hole the operator wants closed
(decide D6); nothing in this spec waits on it.

## 7. The engine

### 7.1 Decode (27c)
`runtime::flint::FlintEngine` (`src/runtime/flint/`) owns `loader::FlintLoaded`, a buffer set and one captured Level Zero list per device; per-layer table from `FlintDesc`.
Layer walk (flash arm; "norm" = `prep_res_fold` + `k2_norm_finish` at `K=2048 NORM_GROUPS=1`, section 2): MoE layer: norm -> qkv GEMV -> `fl_attn_prep` ->
`sink_attn_decode` + `sink_attn_reduce` -> o GEMV -> norm -> router GEMV -> `route_sigmoid_renorm` -> routed gate‖up -> routed down (bf16 routed sum) -> shared gate‖up GEMV ->
`k2_silu_mul` (`SILU_I=1024`) -> shared down GEMV -> the fold `rne(h + rne(r + sh))` (or the two-slot option of section 2, decided by P0). Dense layer: norm -> gate‖up GEMV ->
`k2_silu_mul` -> down GEMV -> fold. Head: norm + int8 head GEMV + argmax x 2. Capture refuses any descriptor drift by name (the Kolibri rule). **P0 first** (spec 25c's method, interleaved pairs after warm-up): a
~780-launch skeleton from existing binaries at Flint's shapes, the GEMV-only list, and the attention at depth, to learn the launch floor and the achieved fraction before any new kernel.
Attention arm, KV form, chunk size: capture rules from the shapes and the free memory (no env vars; the eager arm is a test-only diagnostic selected by the test harness).

### 7.2 Prefill (27d)
l0 backend, Kolibri's chunk walk (`kolibri_prefill.cc`): per chunk the norm (the reused one-rounding chain), QKV, prep, `sink_pf_attn` (window skip on sliding layers, causal flash on full), o, and per MoE layer the route / sort /
gather / dequant / grouped GEMM / combine chain with the shared expert as a dense block and the Flint combine; dense layers through `pf_gemm_silu`. Chunks end at 2048-multiples (prefix cache block
ends). B4: splits at multiples of 64 bitwise, chunked = whole, `--pp 2` split anywhere bitwise another.

### 7.3 Long context (27f)
**Decided (D3, operator 2026-11-01): the context is 262,144 - Flint's `max_position_embeddings` - with int8 KV for the 14 full layers.** No longer
context is offered, probed or gated (330k+ is K2-Horizon's target, not Flint's). Int8 KV in K2's hd128 Hadamard form (f16 scale per position per kv head),
rings stay bf16; decode and prefill readers with the sink; max-len auto = min(262,144, the free memory in 256-quanta) (`runtime::plan`, a Flint size model
`flint_sizes.*` like `kolibri_sizes.*`); `--max-len` above 262,144 is refused by name. **Gate K** (int8 KV quality, within 262,144 only): a passkey / needle
set and a long-transcript perplexity at 32k, 128k, 200k and 262,144, int8 against bf16 KV (the bf16 arm at 262,144 needs `--pp 2`, section 3); the rule is in
plan 27f. It decides nothing about the length - only whether int8 KV ships as the default (if it failed, the operator would choose between bf16 KV with a
shorter one-card context and `--pp 2`).

## 8. Serving (27e)

- `ChatFormat::Kind::Flint` (`chat_format.*`): the template as read (section 1) rendered by minja; needs fixtures for `replace`, `tojson(ensure_ascii=False)`, `namespace`, `.items()`;
  `arguments` as a dict (the server parses the OpenAI JSON string before rendering, the K2 / Kolibri precedent); BOS never added (the template does not emit it, the tokenizer adds none).
- Stops: 150001 / 150003 / 150005 / 150018 are read from `generation_config.json`. **The quant repo carries one** (read 2026-11-01: the four EOS ids,
  `transformers_version 5.21.0.dev0`) and also `chat_template.jinja` (byte-identical to the original's) and `tokenizer.json` (identical to the original's except an
  explicit no-BOS `TemplateProcessing` post-processor). It does **not** carry the sampling defaults (T 1.0, top-p 0.95, top-k 20 are only in the original's
  `generation_config.json`): D7. A stop on 150005 right after `</tool_call>` ends the turn with `finish_reason: tool_calls`.
- **Tool-call parser:** the structure is K2's (`toolcall_k2.cc`: state machine, `arg_key` / `arg_value` pairs, schema-string arguments verbatim, otherwise trimmed JSON with verbatim fallback) with the
  tag table parameterised (`<tool_call>`, `</tool_call>`, `<arg_key>`, `<arg_value>`; K2's table is `<ifm|...>` and has an optional `<ifm|arg_type>` tag Flint does not have), no `tool_calls` wrapper (bare calls, several per turn), reasoning closes on `</think>`;
  selected by `ChatFormat` kind, **never by sniffing `<tool_call>`** (Qwen and Kolibri share that opener). An unschematised string that parses as JSON (`"123"`) is mistyped exactly as in K2: the rule is pinned
  against the template's own rendering (the template writes raw strings for string values and JSON for the rest, so a round-trip test fixes the rule). `tools/toolcall/score.py` gets a Flint format.
  Streaming = whole-text parity (the 3000-case fuzz pattern).
- Reasoning: `<think>...</think>` split into `reasoning_content`; history replay keeps it (the published default; the template renders `reasoning_content` when present).
- Prefix cache / snapshots (Kolibri's contract): state = the sliding rings' last 512 rows (35 layers, 36.7 MB), blocks = the full layers' KV per 2048 positions (int8: 14,560 B x 2048 = 29.8 MB; bf16 58.7 MB);
  no recurrent state. NoPE full layers need nothing extra. Restore at block ends is bitwise the cold run (B4).
- b70-serve dispatch on `model_type flint` beside Kolibri / K2 / Bolt; lookup, MTP and DFlash refused by name until 27g.
- Batching slots (spec 13) are cheap for the KV (14.5 KB x L per slot, rings 36 MB) but each extra row widens the expert union (section 4): not in scope.

## 9. Gates

- **F0, nothing moves.** Every existing kernel binary sha256-identical (G0); `kernel_cmdlines` additions only; no existing `.cl`, `quant.cc` rule or `chat_format` branch changes behaviour; every existing suite green.
- **F1, the reference** (27a). The restated ops equal the pinned modules bitwise on the tiny variants in fp32 and bf16 (prompt and cached decode; chunks = whole); one test per trap: the norm's single rounding (a test
  that rounds `x_hat` first must differ), the q / k norm before RoPE, RoPE only on sliding layers (adding one to NoPE changes the output), the sink in the denominator only, the probabilities' bf16 rounding,
  the fp32 router (bf16-rounding the logit must flip a planted route), top-k on scores with ties, the renormalisation without an epsilon, the combine's ascending-id bf16 order (rank order must differ on some row),
  the shared expert ungated and added after the routed sum, dense layers 0-1, the layer table, the untied head, the eager-experts-forced check.
- **F2, kernels.** Each new kernel against its host twin (`tests/kernels/flint_ref.h`), bitwise where the arithmetic is the same; flash against fp64 at cosine >= 0.99999; the sink kernels against a no-sink run
  with sink = -inf (bitwise the no-sink kernel's output) and against an exact fp64 reference.
- **F3, golden** (greedy, tie-aware, docs/14's undetermined-row rule) against the engine-format bf16 golden: synthetic real-width, `--layers N` truncations, then the whole model. **MoE judge:** the engine is no further
  from the f32 reference than the bf16 golden is (logits cosine distance and per-row KL), the near-miss rule (`tests/golden/forced_judge.h` `NearMiss`) for extra misses against the bf16 twin (each at most 8 bf16 ulp below the f32 top; the cosine distance of the min row and of the median row each at most 2x the twin's), routes as sets per
  (row, MoE layer) with near-ties inside the tolerance 27a measures from the gap distribution.
- **F4, determinism.** Replay bitwise; prefill splits at multiples of 64 bitwise; chunked = whole; prefix-cache restores at block ends bitwise the cold run; `--pp 2` split anywhere bitwise another split; int8 KV replay bitwise.
- **F5, quality.** Gate Q (27q), the A4 set, an SWE-style small set and a long-transcript perplexity: recorded against the BF16 original, then (27e) the **A4 comparison the watch note promised is no longer a trigger** (the operator approved the build on 2026-11-01) but is still recorded.
- **K, int8 KV at depth** (27f): section 7.3 (within 262,144; no gate for a longer context, D3).
- **S, speed** (recorded; no bars before P0): launches (formula), decode at depth 886 .. 262,144, prefill at the same depths, against section 3. **S1** (27g only): lookup is built if the P0 shows >= 1.25x over plain decode on the recorded agentic traces at the real verify cost.

## 10. Plans and effort

Order: 27a -> 27q -> 27b -> 27c -> 27d -> 27f -> 27e; 27g after. One implementing agent per plan, serial (the standing one-agent rule; the
operator is asked before each next plan). 27q's Mac part is small; its box CPU runs (gate Q, hours, detached) may overlap 27b's implementation -
the overlap is box wall-clock, not a second agent.

| plan | what | Mac (blind) | box | effort (implementer-days, estimated by analogy) |
|---|---|---|---|---|
| **27a** facts + reference | fetch / pin, facts sheet, `flint_ref.py`, tiny variants, F1, prompts, A4 set | the whole port on tiny | header scan, golden sets (f32, bf16), A4 reference, taps | 2 (Bolt's 25a carried Mamba; Flint has no SSM) |
| **27q** gate Q | export check, KL / top-1 / A4 / perplexity vs the BF16 original | the scripts, `check_gptq_export` | CPU runs (hours-days, detached) | 0.5 + box time |
| **27b** descriptor, loader, memory plan | `FlintDesc`, layout, repack, loader, synthetic real-width checkpoint, plan | all host side | loads (synthetic, real 21.8 GB) | 2 |
| **27c** decode | P0, kernels, capture, engine, B2-B4 decode | twins, names, Mac-GPU indicative | P0, first compiles, F2-F4, speed | 4 (new: ~6 kernel entry points - `fl_attn_prep`, `sink_attn_decode` / `_reduce` and the eager arm, `route_sigmoid_renorm`, `fl_moe_gate_up` / `_down`; the norm and the shared expert reuse binaries; Bolt's 25c was the biggest plan of 25) |
| **27d** prefill | prefill kernels, walk, F3-F4 on prefill | twins, formula | F2-F4, rate, the chunk decision | 3 |
| **27f** long context | int8 KV, readers with sinks, size model, max-len auto (cap 262,144), gate K | twins, plan | F4 int8, gate K, depth speed | 2.5 |
| **27e** serving | template, parser, output stream, snapshots, `b70-serve`, A4 | fixtures, host snapshot tests | restores, server, A4, agentic transcript, llama-benchy | 3 |
| **27g** lookup (conditional) | P0 (union measure, verify cost), decision | `expert_union.py` over recorded routes | P0 | 1 for the P0; 3 more if built |

Total for the committed set: 2 + 0.5 + 2 + 4 + 3 + 2.5 + 3 = **~17 implementer-days** (estimate; the sum of the rows), plus box time (the P0, the golden runs, the gate-Q CPU hours).
**Calibration against the Bolt record (git):** the Mac-side implementation of 25a-25e took about a day of agent wall-clock (commit spans 2026-10-21 21:32
to 10-22 19:44: 25a ~3 h, 25b ~1 h, 25c ~5 h, 25d ~1 h, 25e ~1 h), while the card validation and fix rounds ran from 10-23 past 10-29 and are not closed.
So the "implementer-days" above are relative sizes, not agent hours; the critical path is the box (27a's CPU golden runs, 27c's first compiles and
fix rounds), which the Bolt series suggests is the larger cost.

## 11. Box stages (prose; no queue rows until asked)

1. **CPU stage (27a, 27q).** On the box CPU, detached, one model job at a time (`/tmp/b70-cpu-model.lock`), RAM floor checked: the f32 and bf16 golden sets on the dequantised quant, the BF16 original's A4 reference
   and gate Q. These hold the longest wall-clock (hours to a day); they need no card and run while the card stages of other work run.
2. **Load stage (27b).** The synthetic real-width checkpoint, then the real 21.8 GB on card 0; read-time with a cold and a warm cache (the USB cold copy is the operator's: the loader reads the box-local
   copy). Confirms the memory plan against `runtime::plan`'s numbers.
3. **Decode stage (27c).** P0 first; then every first compile of the new kernels, F2 on the card, F3 greedy against the bf16 golden, F4, speed at depth.
4. **Prefill stage (27d).** F2-F4 on prefill, rate at depths, the fixed-dequant-cost decision.
5. **Long-context stage (27f).** F4 for int8 KV, gate K at the depths of section 7.3 (up to 262,144), decode and prefill at depth.
6. **Serving stage (27e).** Snapshot restores on the card, the server, A4 on the card, the agentic transcript, llama-benchy.
Every card job takes `~/b70-gpu.lock`, runs in a per-branch remote tree under `~/b70`, and checks the DRM holders first (box idle protocol); the box needs no new directory outside `~/b70`.

## 12. Risks

| risk | why | mitigation |
|---|---|---|
| decode at depth is KV-bound (KV read = weights at ~104k) | 14 full layers of GQA 8 | int8 KV (27f), spec 10's split-K attention, lookup (27g) |
| the shared expert is 2x the routed width | the fused MoE kernels tie them | the dense-GEMV shared path (Bolt's precedent, existing binaries); the two-slot fusion only if P0 says launches matter |
| fp32 router vs the bf16-rounding kernels | K2 / Ornith round the logit, Bolt adds bias / eps / scale, Kolibri selects on the logit | `route_sigmoid_renorm` copied from `nb_route`, with a planted-flip test and the reference's own tie order; near-ties recorded |
| sinks everywhere, no shipped kernel | new reduce / finalise term in 3 paths (flash decode, eager, prefill) | LiLiCorr's pattern; test against sink = -inf and fp64 |
| norm convention differs from Kolibri's | `x_hat` not rounded first | reuse `k2_norm_finish` (one rounding); the q / k head norm in the new `fl_attn_prep`; trap tests |
| reference may take `grouped_mm` | `@use_experts_implementation` | the reference forces the eager loop; checked in F1 |
| the 14-vs-35 layer pattern is not periodic | Kolibri hard-codes `l % 5 == 4` | explicit table in `FlintDesc`, held to `layer_types` |
| prefill MoE has a fixed ~0.27 s dequant cost per chunk (estimated) | all 256 experts touched at 2048 tokens | 27d P0; chunk size or a direct-int4 GEMM as a measured decision |
| int8 KV loses quality at depth | 14 full NoPE layers carry all long-range retrieval | gate K within 262,144 (D3 caps the length; no extrapolation) |
| the quant's `generation_config.json` lacks the sampling defaults | it carries the stop ids only (read 2026-11-01) | D7 |
| no Flint drafter, MoE verify union grows | no MTP | lookup only, P0-gated (27g) |
| quality of the operator's quant unknown vs the BF16 | no gate Q yet | 27q; D6 |
| sinks up to ~10 dominate low-score rows | sampled `softmax_offset` -9.0 .. 10.25 | the `sinks-large` tiny variant; F2 with the real values |
| CPU reference runtime (80 GB bf16 layer-streamed) | Agnes-class job | the box CPU, detached, hours |

## 13. Decisions

**Decided (operator, 2026-11-01):**

| # | decision |
|---|---|
| D3 | **Context 262,144** (Flint's vendor limit, `max_position_embeddings`) with **int8 KV** for the 14 full layers, one card. No 330k+ option, no extrapolation probe, no length gate (330k+ is K2-Horizon's target, not Flint's). `--max-len` above 262,144 is refused by name; max-len auto = min(262,144, memory). |
| D4 | Follows from D3: int8 KV for the full layers is the default (gate K checks its quality within 262,144); rings stay bf16 (0.29 GB); bf16 full-layer KV only as the diagnostic arm. |

**Recommended, for the operator to confirm:**

| # | decision | recommendation |
|---|---|---|
| D1 | Own family (Kolibri template) vs a `ModelDesc` entry | **own family** (section 2): the Qwen path cannot express the model |
| D2 | Generic kernel names, copy-and-adapt (no existing `.cl` edited), reuse of bound binaries where the arithmetic is already Flint's | **yes**: `sink_attn_*`, `route_sigmoid_renorm`, `fl_*`; `k2_norm_finish`, `k2_silu_mul`, `gemv.cl` reused by name; nothing depends on `bolt-spec-restore` |
| D5 | `--pp 2` | supported through the Kolibri-family pp code, **not the default** (no speed gain; headroom only) |

**Open:**

| # | decision | options |
|---|---|---|
| D6 | Our own AutoRound re-quant | no, unless gate Q (27q) shows a hole the operator wants closed |
| D7 | Sampling defaults: the quant repo has the stop ids, template and tokenizer but not T 1.0 / top-p 0.95 / top-k 20 | (a) the operator copies the original's `generation_config.json` into the quant folder (or the quant repo), (b) the server takes the defaults only from the request |
| D8 | Lookup speculation | build only if 27g's P0 clears S1 (>= 1.25x); DFlash only if a Flint drafter appears |
| D9 | Prefill chunk > 2048 / direct-int4 grouped GEMM | decided by 27d's probes, not now |
| D10 | Plan order 27f before 27e | yes: D3's int8 KV is the shipped form and fixes the snapshot sizes 27e needs |

## § Review corrections (2026-11-01)

Checked against main (1e13e27) code and git, transformers `0ea108da` (`modeling_flint.py` sha256 `869d318c...`, `conversion_mapping.py`), both HF
repos' small files, and both repos' shard headers by HTTP range (plus 20 sampled `qzeros` and 4 `softmax_offset` tensors). Confirmed as written:
the norm's single rounding (T:59), q / k norm before RoPE and RoPE only on sliding layers (T:276-283), the 14 full-NoPE layers and window 513, sinks
as an extra fp32 softmax column with P rounded to bf16 before PV (T:219-229), the fp32 sigmoid top-6 router without bias / epsilon / scale (T:90-101),
the eager combine (bf16 product of `down` and the fp32 weight, ascending-id bf16 `index_add_`, then `+ shared` ungated, T:123-157), the tokens,
the template and EOS list, the quant's 109,612 tensors in 5 shards / 22,381,534,784 B with per-expert unfused tensors and router / embedding /
head unquantised, the parameter count and every byte formula of section 3 (21.8 GB resident, 1.52 GB a token, 28,672 / 14,560 B a position),
`k2_pf_moe` <= 255 and Ornith's `pf_moe` <= 256 experts, Kolibri's `x_hat`-first norm, no sink kernel on main, spec 28m's "Flint's routed shape
is Ornith's". Corrected:

1. **Decision D3 written in (operator, 2026-11-01):** context 262,144 (vendor limit) with int8 KV; the "prefer 330k+" target, the 330k memory
   column, the extrapolation discussion and gate L are removed (sections 0, 3, 7.3, 9, 11, 12, 13; plan 27f retitled and rewritten around
   **gate K**, an int8-vs-bf16 KV quality check within 262,144 that never changes the length). `--max-len` above 262,144 is refused by name.
2. **"Every shipped router rounds the logit to bf16" was wrong.** Kolibri's `kol_route` and Bolt's `nb_router` / `nb_route` keep fp32 logits
   unrounded; only K2 (`k2_moe.cl:159`) and Ornith (`moe.cl`) round. Bolt's `nb_route` (on main) is the closest template (fp32 sigmoid,
   renormalised; minus its bias, `+ 1e-20`, `x 5.0`, top-9), so LFM2.5's dropped `lfm_moe.cl` is not needed. `torch.topk`'s order among equal
   scores is unspecified - "ties to the lower id" is now a 27a measurement, not an assumption; the renormalising sum's order is pinned in 27a too.
3. **The one-rounding norm is not new:** K2's `k2_norm_finish` at `NORM_GROUPS=1` is exactly Flint's norm (eps 1e-6, K a define) and is
   reused by name; `fl_norm` is dropped (Bolt's `nb_res_norm` is the pattern for a later fused fold + norm).
4. **The combine:** `k2_moe_down`'s arithmetic (`k2_moe.cl:317`) already is Flint's chain when fed an unrounded fp32 w; the bf16 w comes from
   `k2_route` (`:198`), not the down kernel. "K2's combine already does this" is right for the down kernel, wrong for K2's route. The prefill
   combine template is `k2_pf_moe_combine`.
5. **The 2x-wide shared expert is not the largest new item:** the fused kernels tie widths (Kolibri, K2, Ornith, `model_desc.cc:424`), but Bolt
   runs its shared expert as plain GEMVs (`bolt_capture.cc:355-388`); Flint's does the same with `gemv.cl` + `k2_silu_mul` at `SILU_I=1024` -
   no new kernel. The two-512-slot fusion is a P0-gated option. 27c's kernel count drops from 8-10 to ~6 entry points (effort 5 -> 4).
6. **Header facts now read** (were "27a will read"): router matrix, sinks, norms, embedding and head are bf16 in both checkpoints (no fp32-weight
   router variant needed); sinks are stored as applied (a pure rename `softmax_offset` -> `sinks`); sampled `qzeros` all 0x77777777; sink values
   -9.0 .. 10.25. The original's per-expert tensors include `down_proj` (not only gate / up).
7. **The quant repo carries `generation_config.json`** (stop ids), `chat_template.jinja` (identical) and `tokenizer.json` (identical bar an
   explicit no-BOS post-processor), but **not** the sampling defaults: D7 is now only about those.
8. **Numbers:** the bf16-KV roofline column was wrong at 32k (232 -> 259) and 100k (129 -> 145); the sliding-window read (+0.037 GB a token)
   is noted; one-card free memory uses Level Zero's 32.53 GB (~7.9 GB free; int8 ~545k / bf16 ~277k positions fit - moot beyond 262,144);
   the prefill `t_flop` was 2x too small (22 -> 44 us a token; short-depth upper estimate 6,400 -> ~5,700 t/s); `Qwen35::kVocab` has 53 uses
   in `src/` (not ~77); `KC` is a compile-time define of the sort, 2048 everywhere today (not a hard limit).
9. **Decode estimate re-based and labelled:** no MoE engine has a decode speed on a B70 (BENCHMARKS, spec 28m). The old "210-295 t/s" applied an
   unmeasured eta and no launch cost; it is now ~180-260 t/s at zero depth, ~120-150 at 100k, ~80-90 at 262,144, calibrated only by Qwen3.8's
   dense 0.72 of 638 GB/s and the measured 0.73-1.13 us dispatch gap, flagged as possibly optimistic (Flint's ~2 MB launches vs Qwen's ~20 MB).
10. **Reuse text:** LiLiCorr is being restored on `bolt-spec-restore` (operator, 2026-11-01); the sink pattern is cited by generic description and
   by `99adbb0^:...ll_attn.cl:170`, copied from whichever copy exists, depending on neither. 27c's P0 uses Ornith's `moe_M1_E256_T8_D2048_I512`
   (Flint's exact expert shape) - no Kolibri MoE binary exists at E256.
11. **Near-miss rule** stated as `forced_judge.h` has it (extra misses <= 8 bf16 ulp below the f32 top; min- and median-row cosine distance
   each <= 2x the twin's). **27g** points at spec 28m (Ornith's k = 6 rows, the U2 union kernel) for the expert side.
12. **Plans and effort:** the order 27a, 27q, 27b, 27c, 27d, 27f, 27e, 27g is sound (27f before 27e because D3's int8 KV fixes the snapshot sizes);
   "27q parallel with 27b" means overlapping box CPU time, not a second agent. Each plan has objective gates, commit-after-each-step, box stages
   in prose and one agent. "Bolt's 25a was ~3 days" was wrong: git shows 25a-25e's Mac implementation took about a day of agent wall-clock and
   the card fix rounds over a week; the per-plan days are relative sizes (total ~17), and the box is the critical path.
