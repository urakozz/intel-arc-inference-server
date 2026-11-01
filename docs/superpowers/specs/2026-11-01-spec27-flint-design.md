# Spec 27 - Recursive Flint (`flint`) on the B70: the engine

**Status:** design, 2026-11-01, for operator review (operator approval to write it: 2026-11-01). Open decisions are marked **(decide)**
and collected in section 13. Nothing is built; no code, kernel or card number exists for Flint yet. Supersedes the watch note
`2026-10-23-spec27-flint-watch.md` (which now points here).

Plans: `docs/superpowers/plans/2026-11-01-spec27{a,q,b,c,d,e,f,g}-*.md` (section 10).

**Numbers.** Every number is **published** (config / model card), **read** (the HF files read on 2026-11-01, named in section 1),
**derived** (a formula is given), or **estimated** (marked, with the formula or the reason). Nothing here is measured on the card.

## 0. The brief and the standing rules it meets

The operator judges everything by: usable context >= 242k (prefer 330k+), generation speed, decent quality, correct tool calls
in long agentic coding sessions. Standing rules that shape this spec:

| rule | effect here |
|---|---|
| only AutoRound int4 g64 sym | the operator's `urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ` is the one format; no third-party quant support |
| no host-RAM offload on the per-token path | everything resident: 21.8 GB on one card (section 3) |
| minimal options / env vars | capture rules instead of switches: KV form, attention arm, chunk size are chosen from the shapes (section 7) |
| gates against an f32 reference | MoE: engine no further from f32 than the bf16 golden; near-miss rule (8 bf16 ulp, logits cosine distance <= 2x the twin's) |
| eager paths diagnostic | the eager attention / eager MoE arms exist to be compared with, never the default |
| prefill default `l0` | the l0 backend, no sycl-tla |
| G0 bitwise for other models untouched | no existing `.cl` is edited; new kernels are new files (copy-and-adapt); existing binaries stay sha256-identical |
| commit after each step; one implementing agent per plan | the plan format of 25a-25e |
| reuse ruling (2026-10-27) | dropped code is revived from git under **generic** names: sinks from LiLiCorr's `ll_attn.cl` (`99adbb0^:src/kernels/dflash/ll_attn.cl`), the renormalised sigmoid router from LFM2.5's `lfm_moe.cl` (`77d07ad^:src/kernels/lfm2moe/lfm_moe.cl`); the pattern is taken, no dependency on any branch |

## 1. The model (read 2026-11-01)

Read from: `RecursiveSI/flint` `config.json` (sha256 prefix `a57c33d11f725099`), `generation_config.json`, `tokenizer_config.json`,
`chat_template.jinja`, `model.safetensors.index.json`; transformers PR #49824 at `0ea108daf169dd66a1800e48a56d2251d4ff76bd`,
`src/transformers/models/flint/modeling_flint.py` (683 lines, sha256 prefix `869d318cec68654a`, cited below as `T:line`);
`urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ` `config.json` and index. The Mac has no local transformers source with `flint`
(`~/PycharmProjects/transformers` predates the merge) and no HF cache copy: plan 27a fetches and pins them.

| | value |
|---|---|
| class / type | `FlintForCausalLM`, `model_type: flint`; Apache 2.0; BF16 original ~80 GB |
| layers | 49: layers 0-1 dense SwiGLU (I 6144), layers 2-48 MoE (`mlp_only_layers [0,1]`, `decoder_sparse_step 1`) |
| hidden / vocab | 2048 / 150,144; **untied** `lm_head` (`tie_word_embeddings false`); `pad_token_id 150001` |
| attention | 32 q / 4 kv heads, head_dim 128 (GQA 8), no bias, scale 128^-0.5 |
| per-head q / k RMSNorm | on the head dim, **before** RoPE (T:attention forward) |
| layer types | `layer_types`: full attention at **0,1,5,9,13,17,21,25,29,33,37,41,45,48 (14)**, sliding at the other 35 (2,3,4,6,7,8,10,...,46,47). The pattern is not periodic (leading 2 full, trailing 48 full): an explicit per-layer table |
| positions | sliding layers: RoPE neox (rotate_half), theta 1e4, window 513 (the current token and 512 before it); full layers: **NoPE** (`no_rope_layer_indices` = the same 14). Trained context 262,144 |
| sinks | learned per-head scalar on **all 49 layers**: an extra softmax column attending to nothing (T:eager_attention_forward); checkpoint tensor `self_attn.softmax_offset` [32]; config `softmax_type: learnable` |
| scores | `q.float() @ k.float()^T * scale` in fp32, softmax in fp32 over [scores, sink], the sink column dropped, **probabilities rounded to bf16**, then bf16 matmul with V (T:eager_attention_forward) |
| norm (all sites, incl. q / k) | `w * (x * rsqrt(mean(x^2) + 1e-6))` computed in fp32, **one** rounding to bf16 - `x_hat` is not rounded first (T:FlintRMSNorm; "main diff with Llama") |
| block | plain pre-norm: `h += attn(norm(h)); h += mlp(norm(h))` (T:FlintDecoderLayer); no post-norms |
| MoE | 256 routed experts, top-6, expert I 512, SiLU; shared expert I **1024**, always added, **ungated**; the sum `routed + shared` is one bf16 add |
| router | `gate.weight` kept fp32 in transformers (`_keep_in_fp32_modules_strict = ["gate.weight","sinks"]`): logits `F.linear(x.float(), w.float())` fp32; scores = sigmoid(logits) fp32; **top-6 on the sigmoid scores, no bias**; weights = scores / sum(selected) fp32 (`norm_topk_prob true`), no epsilon, no scaling factor (T:FlintTopKRouter) |
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
User / system text has `<think>`, `</think>` removed and the role tokens escaped (`&lt;|user|&gt;`). The generation prompt is
`<|assistant|>` with no `<think>` prefix: the model opens its own reasoning. **A model turn ends on 150003 / 150005 / 150001 /
150018**: after a tool call the model emits `<|observation|>` (150005) - the server stops on it and the call is the turn's result.

**The checkpoint as shipped (urakozz quant, index read).** 5 shards, 22,381,534,784 bytes, 109,612 tensors. AutoRound 0.17.0, `bits 4,
sym true, group_size 64, packing_format auto_round:auto_gptq`, `block_name_to_quantize model.layers`; `extra_config` keeps `mlp.gate`
of layers 2-48 at 16-bit float. Per layer: `self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales}`, `q_norm.weight`, `k_norm.weight`
(both [128]), `softmax_offset`, `input_layernorm.weight`, `post_attention_layernorm.weight`; layers 0-1 `mlp.{gate,up,down}_proj.*`
(quantised); layers 2-48 `mlp.gate.weight` (router, unquantised), `mlp.shared_expert.{gate,up,down}_proj.*` (quantised) and
**per-expert unfused** `mlp.experts.E.{gate,up,down}_proj.*` (47 x 256 x 9 = 108,288 tensors). Outside the layers: `model.embed_tokens.weight`,
`model.norm.weight`, `lm_head.weight` (bf16, in the last shard). The original `RecursiveSI/flint` index also carries per-expert unfused
`gate_proj.weight` / `up_proj.weight` (36,734 names), so the reference reads either form.
Not yet read (27a, header range reads on the Mac): the dtypes of `gate.weight`, `softmax_offset`, the norms; the qzeros value (the engine
requires 0x77777777, `src/loader/quant.cc:526-535`); whether the sinks are stored as they are applied.

## 2. Architecture mapping, per op

Family decision: **Flint is its own engine family** (`model_type flint`, like Kolibri / K2 / Bolt), not a `ModelDesc` (Qwen-family)
entry. The Qwen-family loader cannot express dense-first layers, a non-periodic layer table, a window, sinks, head_dim 128 or vocab
150,144 (`Qwen35::kVocab` is used in about 77 places). Template: **Kolibri's** structure (`src/model/kolibri1.*`, `src/loader/kolibri1_*`,
`src/runtime/kolibri/*`, `src/kernels/kolibri/*`), generalised where Kolibri hard-codes its dims.

"Reuse" = the existing binary / function bound by name with new CMake lines only. "Adapt" = a **new file** copied from the named
source and changed (no existing `.cl` edited: G0). "New" = written here.

| op | what Flint needs | source | verdict |
|---|---|---|---|
| embedding gather | bf16 row, 150144 x 2048 | `embed_gather.cl` | reuse (new variant name `D2048_V150144`) |
| input / post-attention norm, final norm | pre-norm, `w * x_hat` **one** rounding, residual fold | Kolibri's `kol_norm_finish` / `prep_res_fold` round `x_hat` first and have post-norms | **adapt**: `fl_norm` (one rounding), plain `prep_res_fold` for the residual; drop `kol_post_add` |
| q / k / v projection | one fused GEMV, N = 4096 + 512 + 512 = 5120, K 2048, int4 layout 1 | `gemv.cl` (`gemv_variant`) | reuse; new shape, split-K row in the tuning table |
| q / k head norm, RoPE (sliding only), cache write | per-head norm `w * x_hat` one rounding, then neox RoPE `rne(rne(y*c) + rne(rot*s))`, ring write for sliding, linear write for full | `kol_attn_prep` (same RoPE chain, ring vs linear) but its norm rounds `x_hat` first | **adapt**: `fl_attn_prep` |
| attention decode | GQA 8, hd 128, window 513 ring or linear NoPE, **sink** | `kol_attn.cl` (window, ring, hd 128, no sink, M <= 4) | **adapt**: generic `sink_attn_decode` + `sink_attn_reduce` (sink added once in the reduce, denominator only, after the block merge, also when a block is skipped by the window; exp2 build: sink x log2 e) - pattern from `ll_attn.cl:170` |
| attention decode, eager arm | scores fp32, softmax with sink, bf16 P, bf16 PV | `kol_attn_eager.cl` (bf16 scores) | **adapt** (diagnostic arm; matches the reference's rounding exactly) |
| attention prefill | flash with window skip, NoPE, sink in the finalise (`lsum += exp(sink - m)` before `1/lsum`) | `kol_pf_attn.cl` (window skip, hd 128) | **adapt**: `sink_pf_attn` |
| o projection | K 4096, N 2048 | `gemv.cl` | reuse |
| dense MLP layers 0-1 | gate+up GEMV (N 12288), SiLU-mul, down GEMV | `k2_silu_mul` (parametric, K2 instantiates I = 6144, `CMakeLists.txt:744`), `gemv.cl` | reuse (prefill: `pf_gemm_silu` / `pf_gemm`) |
| router logits | fp32 logits from bf16 `x` and the router matrix (bf16 or fp32 in the checkpoint, widened) | Kolibri's router GEMV already emits fp32 logits (`gemv_bf16`) | reuse; if the matrix is fp32 in the checkpoint, a fp32-weight variant (27a decides from the header) |
| route | fp32 sigmoid, top-6 on the **scores** (ties to the lower id, ties are real: fp32 sigmoid saturates at 1.0), `w = s / sum(s_selected)` fp32, no bias, no epsilon, slots sorted by ascending id | `k2_route` (sigmoid, **bf16-rounded** logit, mandatory bias, `ROUTE_SCALE`, ascending-id slots), `lfm_route` (bf16-rounded scores, +1e-6) | **adapt, generic name** `route_sigmoid_renorm`: fp32 chain, no bias argument, no rounding of logit or score |
| routed experts (decode) | 256 blocks, gate||up K 2048 N 1024, down K 512 N 2048, int4 layout 1, 6 slots | `moe.cl` family shapes (Ornith has E256 D2048 I512; TOP_K 8 hard in the binary) | **adapt**: the Kolibri / K2 `moe_gate_up` / `moe_down` pair at `E256 T6 D2048 I512` |
| shared expert | I 1024 (twice the routed width), ungated, int4 | every existing MoE kernel ties shared I to routed I (`kol_moe.cl:23-24`, `moe.cl:73-74`, `model_desc.cc:424`) | **new**: a shared-expert pair with its own `SH_I` (`fl_sh_up`, `fl_sh_down`) - the largest new item; runs as its own launches, or fused into the routed pair later (27c arm) |
| expert combine | per expert `rne(rne(down) * w)` (w fp32 unrounded), bf16 sequential add ascending id, then `+ shared` bf16, then residual | `k2_moe_down` (matches, but rounds `w` to bf16 before: to check) | **adapt** inside `fl_moe_down`; the shared add is a separate step after the routed sum |
| lm head | int8 row-scaled head (spec 9) from the bf16 checkpoint head, fp32 sampler argmax | `gemv_i8w.cl`, `argmax.cl`, `lm_head_int8.cc` | reuse; new vocab variant names |
| prefill MoE | route -> sort -> gather -> dequant -> grouped GEMM -> combine | `pf_moe.cl` (Ornith's allows 256; K2's `k2_pf_moe.cl` allows only <= 255: ids staged as bytes) | **adapt** Ornith's `pf_moe.cl` sort / gather / GEMM (256 ok, hidden % 1024, I % 64, 2I % 256 all pass) with the Flint combine and the separate shared add |
| prefill linears | l0 GEMM | `pf_gemm*` | reuse |
| KV cache | full layers linear [max_len][4][128], sliding layers ring | Kolibri buffers (`kolibri_buffers.*`) | **adapt** with a per-layer table |
| int8 KV (full layers) | per (position, kv head) f16 scale, 128-dim Hadamard (K2's hd128 form) | `k2_kv8.cl`, `src/common/kv8.h` `hd128` | **adapt** for the sink reduce and the ring-less linear cache (plan 27f) |
| descriptor | per-layer table (type, dense or MoE), no hard-coded periodicity | `Kolibri1Desc` has `l % 5 == 4` and a `require` that refuses any other dims (`kolibri_capture.cc:73-76`) | **new** `FlintDesc` held to `config.json` key by key |
| loader | per-expert unfused AutoRound tensors, 109,612 names | `kolibri1_loader`, `moe.cc` (separate gate/up per expert) | **adapt** |
| capture | one list per card, replayed | `kolibri_capture.cc` | **adapt** (dense branch for layers 0-1, no post-norm, shared expert, sinks) |
| prefix cache / snapshots | rings' last 512 rows + full-layer blocks | `kolibri_engine` save / load, `prefix_cache.h` | **adapt** |
| template, tool parser | role tokens, GLM-style call tags | `toolcall_k2.cc` (state machine, tags `<ifm|...>`), `chat_format.*` | **adapt**: tag table parameter (section 8) |

What is new about the model compared with everything shipped: **sinks** (no kernel on main has one), **fp32 router chain** (every
shipped sigmoid / softmax router rounds the logit to bf16), the **one-rounding norm** (Kolibri rounds `x_hat` first), the **2x-wide
shared expert**, **dense-first layers**, and a **non-periodic layer table**. Each is a trap with its own test in 27a / 27c (section 9).

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
x 0.53125 = 1.16 GB; routers 0.049 GB (read in full); head 0.31 GB; **~1.52 GB** (the brief's figure). Roofline at 638 GB/s: 638 / 1.52 =
**420 t/s** at zero depth - a ceiling; the achieved fraction is unknown until 27c's P0 (spec 25c's method).

**KV.** Only the 14 full layers grow: per position 14 x 4 x 128 x 2 (K, V) = 14,336 values: **28,672 B bf16**; int8 with a f16 scale per
(position, kv head, K or V): 14,336 + 14 x 4 x 2 x 2 = **14,560 B**. The 35 sliding layers hold a ring: ring 4096 slots (a power of two >= prefill
chunk 2048 + window, Kolibri's rule) x 35 x 2048 B = **0.29 GB** bf16 (kept bf16 under int8 KV, decide D4); the live window state a snapshot needs is
35 x 512 x 2048 B = 36.7 MB.

**One card (32 GB nominal; the planner measures the real figure).** Free for KV = 32 - 21.8 (weights) - 1.5 (the planner's default
reserve) - ~1.0 (prefill scratch: dequant batch, chunk buffers; estimated, 27d measures) - 0.29 (rings) = **~7.4 GB** (estimated).

| KV form | bytes / position | positions that fit = 7.4e9 / B | at 262,144 | at 330,000 |
|---|---:|---:|---:|---:|
| bf16 | 28,672 | ~258k | 7.5 GB (marginal: does not leave the reserve) | no |
| **int8** | 14,560 | **~508k** | 3.8 GB | 4.8 GB |

So **one card reaches the brief's 242k with int8 KV and not safely with bf16**; 330k fits in memory at int8. The model's trained context is
262,144, so memory is not the limit beyond it: quality is (section 7.3).

**Two cards, `--pp 2`** (spec 16's layer split by bytes): ~10.9 GB of weights each (embedding on card 0, head and final norm on card 1), and each
card holds its own full layers' KV (7 of 14). Free per card ~ 32 - 10.9 - 1.5 - 1.0 - 0.15 = ~18 GB: bf16 KV = 18e9 / 14,336 B (half the layers) =
~1.25 M positions, int8 ~2.4 M. Memory is not binding on two cards. **Pipeline stages run in sequence, so `--pp 2` adds a hand-off and no speed**
(the bytes still add up); its value is headroom (bf16 KV, batching slots, room for a drafter), not throughput. The one-card int8 path is the default
product; `--pp 2` is supported through the existing Kolibri-family pp code (27c / 27e gates) and refused nowhere by Flint itself.
(Load note: `place_stages` loads the whole model on card 0 first - 21.8 GB fits.)

**Decode bytes at depth L** (int8 KV, full layers read 14,560 B x L per token; sliding rings are small and capped):

| depth L | KV bytes read | total GB / token | roofline t/s = 638 / total | bf16 KV roofline |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.52 | 420 | 420 |
| 32,768 | 0.48 | 2.00 | 319 | 232 |
| 100,000 | 1.46 | 2.98 | 214 | 129 |
| 262,144 | 3.82 | 5.34 | 119 | 71 |

Reading: at the depths of a long agentic session the **KV read, not the weights, is the dominant cost** (equal to the weights at ~104k tokens,
1.52e9 / 14,560). So int8 KV and the decode-attention-at-depth kernel (spec 10's split-K, measured 540 GB/s at 128k on Qwen) matter more than another
weight-side trick. Achieved = roofline x eta; the attention part has a measured eta ~0.85 (540 / 638, Qwen, spec 10); the GEMV part's eta is estimated
0.5-0.7 (no Flint, Kolibri or K2 decode is measured in this repo; `docs/BENCHMARKS.md` MoE rows say "nothing measured yet"). **Estimated decode: 210-295 t/s at
zero depth, ~110-160 t/s at 100k, ~65-100 t/s at 262k** (formula: roofline x eta, eta 0.5-0.7 on weights, 0.85 on KV). These are guesses to be replaced by 27c's P0, not
promises.

**Launch count (estimated by Kolibri's structure):** per MoE layer ~16 launches (norm, qkv GEMV, prep, attn + reduce, o GEMV, residual fold, norm, router GEMV,
route, routed up, routed down, shared up, shared down, fold), per dense layer ~11, plus ~5 at the head: 47 x 16 + 2 x 11 + 5 = **~780** (Kolibri: 756, spec 20c).
27c's plan test pins the exact formula.

**Prefill (estimated).** FLOPs per token ~ 2 x 2.5 B = 5 GFLOP (about 11x fewer than Qwen3.8-27B's ~54). But the l0 MoE path dequantises int4 -> bf16 per
expert batch every chunk: for a 2048-token chunk all 256 experts are touched (48 tokens each on average), so per MoE layer the dequant moves 0.43 GB int4 in
and ~1.6 GB bf16 out; over 47 layers (read + write + the GEMM's read) ~ (20 + 75 + 75) GB = 170 GB per chunk = 0.27 s at 638 GB/s, a **fixed cost per
chunk**: rate ~ 2048 / (0.27 + 2048 x t_flop) with t_flop ~ 22 us (Qwen's measured 2.1k t/s at ~54 GFLOP) -> 2048 / 0.32 = **~6,400 t/s upper estimate at short depth**; at
128k depth attention dominates: 14 full layers x 32 heads x 128 is ~0.58 of Qwen3.8's attention work per token-depth, so ~1.5k t/s (Qwen's 915 t/s at 130k / 0.58;
estimated). A 200k-token first prompt is minutes; **the prefix cache and session continuation are therefore load-bearing**, not an optimisation (section 8).
Decide in 27d P0 whether a chunk > 2048 (`kC` = 2048 is a hard limit of the shipped sort) or an int8 / direct-int4 grouped GEMM is worth the work: the fixed 0.27 s is
the lever.

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
available for this model, and it needs: an M = 2..5 MoE path (no shipped MoE kernel at M > 1 is gated; spec 28 excludes the expert GEMVs), a verify attention at M rows with
sinks, and `truncate_to`. Plan 27g is therefore a P0 + a decision, built after 27a-f and only if the P0 clears the bar (section 9, S1).

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
Layer walk (flash arm): MoE layer: `fl_norm` -> qkv GEMV -> `fl_attn_prep` -> `sink_attn_decode` + `sink_attn_reduce` -> o GEMV -> residual fold -> `fl_norm` -> router GEMV ->
`route_sigmoid_renorm` -> routed up -> routed down -> shared up -> shared down -> fold. Dense layer: ... `fl_norm` -> gate||up GEMV -> `k2_silu_mul` -> down GEMV -> fold.
Head: `fl_norm` + int8 head GEMV + argmax x 2. Capture refuses any descriptor drift by name (the Kolibri rule). **P0 first** (spec 25c's method, interleaved pairs after warm-up): a
~780-launch skeleton from existing binaries at Flint's shapes, the GEMV-only list, and the attention at depth, to learn the launch floor and the achieved fraction before any new kernel.
Attention arm, KV form, chunk size: capture rules from the shapes and the free memory (no env vars; the eager arm is a test-only diagnostic selected by the test harness).

### 7.2 Prefill (27d)
l0 backend, Kolibri's chunk walk (`kolibri_prefill.cc`): per chunk `fl_norm`, QKV, prep, `sink_pf_attn` (window skip on sliding layers, causal flash on full), o, and per MoE layer the route / sort /
gather / dequant / grouped GEMM / combine chain with the shared expert as a dense block and the Flint combine; dense layers through `pf_gemm_silu`. Chunks end at 2048-multiples (prefix cache block
ends). B4: splits at multiples of 64 bitwise, chunked = whole, `--pp 2` split anywhere bitwise another.

### 7.3 Long context (27f)
Int8 KV for the 14 full layers (K2's hd128 Hadamard form, scale per position per kv head), rings stay bf16; decode and prefill readers with the sink; max-len auto fills the free memory in
256-quanta (`runtime::plan`, a Flint size model `flint_sizes.*` like `kolibri_sizes.*`). **Position beyond 262,144 is an extrapolation**: the 14 full layers have no positional encoding and the 35
sliding layers only ever see a 513 window with RoPE, so nothing in the architecture breaks, and Kolibri (same design) is published as validated to 1,048,576 - but Flint publishes no such
number. Gate L: a passkey / needle set at 128k, 242k, 262k, 300k, 330k, perplexity on a long transcript at the same depths, int8 vs bf16 KV; the default max-len is set to the **largest depth
that passes**, capped at what memory fits (decide D3: the operator's "prefer 330k+" is met only if gate L passes at 330k).

## 8. Serving (27e)

- `ChatFormat::Kind::Flint` (`chat_format.*`): the template as read (section 1) rendered by minja; needs fixtures for `replace`, `tojson(ensure_ascii=False)`, `namespace`, `.items()`;
  `arguments` as a dict (the server parses the OpenAI JSON string before rendering, the K2 / Kolibri precedent); BOS never added (the template does not emit it, the tokenizer adds none).
- Stops: 150001 / 150003 / 150005 / 150018 are read from `generation_config.json` (the server reads it; the quant repo must carry it: 27e checks, and copies the file into the quant folder if absent -
  the operator's call, D7). A stop on 150005 right after `</tool_call>` ends the turn with `finish_reason: tool_calls`.
- **Tool-call parser:** the structure is K2's (`toolcall_k2.cc`: state machine, `arg_key` / `arg_value` pairs, schema-string arguments verbatim, otherwise trimmed JSON with verbatim fallback) with the
  tag table parameterised (`<tool_call>`, `</tool_call>`, `<arg_key>`, `<arg_value>`; K2's table is `<ifm|...>`), no `tool_calls` wrapper (bare calls, several per turn), reasoning closes on `</think>`;
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
  from the f32 reference than the bf16 golden is (logits cosine distance and per-row KL), the near-miss rule for rows that differ from the bf16 twin (within 8 bf16 ulp and cosine distance <= 2x the twin's), routes as sets per
  (row, MoE layer) with near-ties inside the tolerance 27a measures from the gap distribution.
- **F4, determinism.** Replay bitwise; prefill splits at multiples of 64 bitwise; chunked = whole; prefix-cache restores at block ends bitwise the cold run; `--pp 2` split anywhere bitwise another split; int8 KV replay bitwise.
- **F5, quality.** Gate Q (27q), the A4 set, an SWE-style small set and a long-transcript perplexity: recorded against the BF16 original, then (27e) the **A4 comparison the watch note promised is no longer a trigger** (the operator approved the build on 2026-11-01) but is still recorded.
- **L, long context** (27f): section 7.3.
- **S, speed** (recorded; no bars before P0): launches (formula), decode at depth 886 .. the largest passing depth, prefill at the same depths, against section 3. **S1** (27g only): lookup is built if the P0 shows >= 1.25x over plain decode on the recorded agentic traces at the real verify cost.

## 10. Plans and effort

Order: 27a -> 27q (parallel on the box CPU with 27b) -> 27b -> 27c -> 27d -> 27f -> 27e; 27g after. One implementing agent per plan, serial (the standing one-agent rule).

| plan | what | Mac (blind) | box | effort (implementer-days, estimated by analogy) |
|---|---|---|---|---|
| **27a** facts + reference | fetch / pin, facts sheet, `flint_ref.py`, tiny variants, F1, prompts, A4 set | the whole port on tiny | header scan, golden sets (f32, bf16), A4 reference, taps | 2 (Bolt's 25a was ~3 with Mamba; Flint has no SSM) |
| **27q** gate Q | export check, KL / top-1 / A4 / perplexity vs the BF16 original | the scripts, `check_gptq_export` | CPU runs (hours-days, detached) | 0.5 + box time |
| **27b** descriptor, loader, memory plan | `FlintDesc`, layout, repack, loader, synthetic real-width checkpoint, plan | all host side | loads (synthetic, real 21.8 GB) | 2 |
| **27c** decode | P0, kernels, capture, engine, B2-B4 decode | twins, names, Mac-GPU indicative | P0, first compiles, F2-F4, speed | 5 (new: 8-10 kernel entry points; Bolt's 25c was the biggest plan of 25) |
| **27d** prefill | prefill kernels, walk, F3-F4 on prefill | twins, formula | F2-F4, rate, the chunk decision | 3 |
| **27f** long context | int8 KV, readers with sinks, size model, max-len auto, gate L | twins, plan | F4 int8, gate L, depth speed | 3 |
| **27e** serving | template, parser, output stream, snapshots, `b70-serve`, A4 | fixtures, host snapshot tests | restores, server, A4, agentic transcript, llama-benchy | 3 |
| **27g** lookup (conditional) | P0 (union measure, verify cost), decision | `expert_union.py` over recorded routes | P0 | 1 for the P0; 3 more if built |

Total for the committed set: 2 + 0.5 + 2 + 5 + 3 + 3 + 3 = **~18.5 implementer-days** (estimate; the sum of the rows), plus box time (the P0, the golden runs, the gate-Q CPU hours).

## 11. Box stages (prose; no queue rows until asked)

1. **CPU stage (27a, 27q).** On the box CPU, detached, one model job at a time (`/tmp/b70-cpu-model.lock`), RAM floor checked: the f32 and bf16 golden sets on the dequantised quant, the BF16 original's A4 reference
   and gate Q. These hold the longest wall-clock (hours to a day); they need no card and run while the card stages of other work run.
2. **Load stage (27b).** The synthetic real-width checkpoint, then the real 21.8 GB on card 0; read-time with a cold and a warm cache (the USB cold copy is the operator's: the loader reads the box-local
   copy). Confirms the memory plan against `runtime::plan`'s numbers.
3. **Decode stage (27c).** P0 first; then every first compile of the new kernels, F2 on the card, F3 greedy against the bf16 golden, F4, speed at depth.
4. **Prefill stage (27d).** F2-F4 on prefill, rate at depths, the fixed-dequant-cost decision.
5. **Long-context stage (27f).** F4 for int8 KV, gate L at the depths of section 7.3, decode and prefill at depth.
6. **Serving stage (27e).** Snapshot restores on the card, the server, A4 on the card, the agentic transcript, llama-benchy.
Every card job takes `~/b70-gpu.lock`, runs in a per-branch remote tree under `~/b70`, and checks the DRM holders first (box idle protocol); the box needs no new directory outside `~/b70`.

## 12. Risks

| risk | why | mitigation |
|---|---|---|
| decode at depth is KV-bound (KV read = weights at ~104k) | 14 full layers of GQA 8 | int8 KV (27f), spec 10's split-K attention, lookup (27g) |
| the shared expert is 2x the routed width and no kernel supports it | every MoE kernel ties them | a dedicated shared pair (27c), fused later if P0 says launches matter |
| fp32 router vs the bf16-rounding kernels | revived routers round the logit and score | the new `route_sigmoid_renorm`, with a planted-flip test; near-ties recorded |
| sinks everywhere, no shipped kernel | new reduce / finalise term in 3 paths (flash decode, eager, prefill) | LiLiCorr's pattern; test against sink = -inf and fp64 |
| norm convention differs from Kolibri's | `x_hat` not rounded first | new `fl_norm` / `fl_attn_prep`, trap tests |
| reference may take `grouped_mm` | `@use_experts_implementation` | the reference forces the eager loop; checked in F1 |
| the 14-vs-35 layer pattern is not periodic | Kolibri hard-codes `l % 5 == 4` | explicit table in `FlintDesc`, held to `layer_types` |
| prefill MoE has a fixed ~0.27 s dequant cost per chunk (estimated) | all 256 experts touched at 2048 tokens | 27d P0; chunk size or a direct-int4 GEMM as a measured decision |
| positions > 262,144 are untested by the vendor | NoPE extrapolation | gate L; default max-len = largest passing depth |
| the quant checkpoint may lack `generation_config.json` / template files | stops 150001/3/5/18 are read from it | 27e checks, D7 |
| no Flint drafter, MoE verify union grows | no MTP | lookup only, P0-gated (27g) |
| quality of the operator's quant unknown vs the BF16 | no gate Q yet | 27q; D6 |
| `softmax_offset` / `gate.weight` dtypes unknown | not in the index | 27a header scan, before 27b's layout is fixed |
| CPU reference runtime (80 GB bf16 layer-streamed) | Agnes-class job | the box CPU, detached, hours |

## 13. Open decisions for the operator

| # | decision | recommendation |
|---|---|---|
| D1 | Own family (Kolibri template) vs a `ModelDesc` entry | **own family** (section 2): the Qwen path cannot express the model |
| D2 | Generic kernel naming and copy-and-adapt (no existing `.cl` edited) | **yes**: `sink_attn_*`, `route_sigmoid_renorm`, `fl_*`; costs some duplication, keeps G0 trivial |
| D3 | Default max context | **262,144** at int8 KV on one card; **330k+ only if gate L passes**; never advertise beyond the passing depth |
| D4 | KV form | int8 for full layers always (after gate L's int8-vs-bf16 check), rings bf16; bf16 full-layer KV only as the diagnostic / fallback arm |
| D5 | `--pp 2` | supported through Kolibri's pp code, not the default (no speed gain; use for headroom) |
| D6 | Own AutoRound re-quant | **no**, unless gate Q shows a hole; the operator's quant stays |
| D7 | `generation_config.json`, template and tokenizer files in the quant folder | 27e checks; if absent, the operator copies them from `RecursiveSI/flint` (one `hf download --include` command is given) |
| D8 | Lookup speculation | build only if 27g's P0 clears S1; DFlash only if a Flint drafter appears |
| D9 | Prefill chunk > 2048 / direct int4 grouped GEMM | decided by 27d's P0, not now |
| D10 | Plan order: 27f before 27e | yes (the goal needs the long context before the serving gates are final); swap if serving first is wanted |
