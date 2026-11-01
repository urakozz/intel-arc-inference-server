# Spec 29 - Nanbeige4.3-33B (a plain dense Llama, all 52 layers full attention) in the engine

**Status (2026-11-01): DESIGN - nothing built.** Operator approved writing it 2026-11-01. The slot "spec 29 (Nanbeige4.3-33B)"
was reserved 2026-10-27; no file existed (checked: `docs/superpowers/specs` has 28 / 28c as its highest numbers).
Plans: `docs/superpowers/plans/2026-11-01-spec29a-...` to `29f-...` (section 12). No box queue rows are added
(operator rule, 2026-10-27): the plans describe their box stages in prose; rows are allocated when the operator asks.

**What the operator wants, and what every choice below is judged by (2026-10-27 / 11-01):** usable context **>= 242k,
preferably 330k+**; generation speed; decent quality; **correct tool calls in long agentic coding sessions**. Rules that
bind this spec: only AutoRound int4 g64 sym weights; no host-RAM offload on the per-token path; minimal flags and
environment variables; correctness against an f32 CPU reference and a bf16 twin (the near-miss rule, section 10);
eager paths are diagnostic; the default prefill backend is `l0`; G0 stays bitwise for every other model; one
implementing agent per plan, objective gates, a signed commit after every step.

**Grades.** Every number is **measured** (this repo, on this box, record cited), **published** (by someone else, cited),
**derived** (arithmetic from measured or published numbers, the formula shown) or **estimated** (a judgement, flagged).
Nothing in this spec about Nanbeige is measured: no weight has been read on this Mac, nothing has run on a card.

## 0. The short version

| question | answer |
|---|---|
| what is it | `LlamaForCausalLM`, 33.1 B parameters, 52 layers x (GQA attention 56 q / 8 kv x 128 + SwiGLU 22016), hidden 7168, vocab 166912, untied head, RoPE theta 1.2e8, trained context 524288. Stock Llama - no remote code, no q/k norm, no gate, no sinks, no sliding window |
| what does a decoded token cost | 17.52 GB of weights (int4 g64 linears 16.32 + int8 head 1.20) plus **108,160 B of int8 KV per position of context** - at 162k positions the KV read equals the weights |
| plain decode, one stream | **~25.7 t/s at depth 4k, ~10.6 t/s at 242k, ~8.7 t/s at 330k** (derived, section 6) - the engine's measured Qwen3.8 bandwidth efficiency applied to Nanbeige's bytes |
| context | one card: ~84k (int8 KV) / ~43k (bf16 KV). **`--pp 2` + int8 KV: ~345-360k.** 242k and 330k both fit; bf16 KV (~175k) does not reach 242k |
| speculative decoding | no MTP head exists (checked: 471 tensors, none). A third-party **DFlash2 drafter exists** (ProCreations, Apache 2.0, 3 layers, 0.98 B) - same family as the z-lab drafter we already run; lookup decoding needs nothing. Both gated behind spec 28b / 28c / 19f. **Speculation is where the long-context speed is: ~21-25 t/s at 242-330k instead of 9-11** (derived, estimated acceptance) |
| prefill | ~1,780 t/s linears-only; **attention dominates from ~20k up**: cold 242k prefill is ~28 min on one card, ~14 min with `--pp 2` pipelining (derived). Cold prefills of that size must be rare: session continuation and the prefix cache are first-class, and the flash-prefill kernel at depth is the biggest speed lever after speculation |
| nearest code to reuse | **K2-Horizon's kernels** (head_dim 128, all layers full attention, int8 KV, flash prefill, bf16 RoPE chain) - not Muse's; Muse's dense pieces are mostly the same shape but K2's are live in the tree. GEMV family, prefill GEMM (`l0`), loaders: reused. Bolt's / Kolibri's own two-card engine is the template for `--pp 2` |
| biggest risks | (1) the Qwen-path attention kernels are head_dim 256 / GQA 6 - Nanbeige needs head_dim 128 / **GQA 7** variants (K2's are GQA 4); (2) decode at depth is KV-bound; (3) cold-prefill time at 242k+; (4) down_proj K = 22016 is not a multiple of 1024, so `l0-int8` prefill cannot be used; (5) the SentencePiece-style tokenizer has never run through our tokenizer; (6) the all-full-attention layout is refused by the current pipeline planner |

## 1. The model (read 2026-11-01 from the Hub's `config.json`, `model.safetensors.index.json`, `tokenizer_config.json`, model card)

| | value | grade |
|---|---|---|
| class | `LlamaForCausalLM`, `model_type: llama`, transformers 4.51.1, no `auto_map` (no remote code), Apache 2.0 | published |
| layers / hidden / FFN | 52 / 7168 / 22016 (SwiGLU, `silu`, no bias anywhere) | published |
| attention | 56 q heads, 8 kv heads, head_dim 128 (**q width 7168 = hidden**, kv width 1024), GQA 7, all 52 layers full attention, no q/k norm, no output gate | published |
| RoPE | neox (`rotate_half`) over all 128 dims, theta **120,000,000**, `rope_scaling: null`, trained and configured to 524288 positions | published |
| norms | RMSNorm eps **1e-5** (not 1e-6), plain weight (not 1 + w). HF Llama's form: `weight * x_hat.to(bf16)` - x_hat is rounded to bf16 **before** the weight multiply (two roundings; our kernels do one) | published (config), read from transformers' llama code (to be pinned in 29a) |
| embeddings | `vocab_size` 166912, untied `lm_head` (2 x 166912 x 7168 = 2.39 B bf16 parameters) | published |
| tokens | `<unk>` 0, `<s>` 1, `</s>` 2, `<|im_start|>` 166100 (bos), `<|im_end|>` 166101 (eos), `<|endoftext|>` 166102, `<think>` 166103, `</think>` 166104, `<tool_call>` 166105, `</tool_call>` 166106; ids 166107.. 166911 are padding rows (the head has them; the argmax must mask them: `VOCAB_USED` = 166107, as Agnes does with 248089) | published (added_tokens), derived (padding) |
| tokenizer | `LlamaTokenizer` (SentencePiece, `add_prefix_space`, `legacy: false`), the card says "use_fast=False"; the repo also ships `tokenizer.json`. Our engine reads `tokenizer.json` only | published |
| chat template | ChatML (`<|im_start|>role\n...<|im_end|>`); the generation prompt ends `<|im_start|>assistant\n<think>\n` (or `<think>\n\n</think>\n\n` with `enable_thinking=false`); assistant history is `<think>\n{reasoning}\n</think>\n\n{content}`; **`preserve_thinking`**: only an explicit `false` strips old turns' reasoning (undefined keeps it); tool calls **Qwen3-Coder XML** (`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`), results as `<tool_response>` inside a `user` turn; `tool_call_format=json` also exists | published |
| MTP / draft head | **none**: the 471 tensors are `embed_tokens`, `lm_head`, `norm`, and 9 per layer | derived from the index |
| sampling the vendor recommends | agentic: T 1.0, top-p 0.95, 65,536 new tokens; reasoning: T 0.6, top-p 0.95, 262,144 | published |
| quality (vendor's table, thinking at highest effort, `preserve_thinking=true`) | Terminal-Bench 2.1 80.9 (Qwen3.8-27B 73.0), DeepSWE 65.5 (42.5), SWE-Bench Pro 60.3 (61.7), NL2Repo 53.0 (48.3), Toolathlon 70.4 (68.5), AA-LCR 84.0 (82.0) | published (vendor; never a bar - section 10 measures ours) |
| our checkpoint | `urakozz/Nanbeige4.3-33B-W4A16-g64-AutoRound-GPTQ`: `auto-round` 0.17.0, `auto_round:auto_gptq` packing, bits 4, **group 64, sym**; `model.layers` quantised (all 7 linears per layer); embed, head and norms stay bf16; 4 shards: 3.90 B I32 words (qweight + qzeros), 0.48 B F16 scales, 2.39 B BF16 | published (repo metadata) |
| the original | 14 bf16 shards, 66.24 GB (`metadata.total_size` 66,237,954,048 = 33.12 B x 2) | published |

The Hub also lists third-party quantisations (NVFP4, EXL3, GGUF, MLX) and Nanbeige's own FP8 / INT8 / INT4 releases. **None is used**
(one-format rule). A related small family exists: Nanbeige4.2-3B (vocab 166144, looped transformer, own `DSpark` drafter) - it
is not a drafter for this model (section 7.4).

## 2. What it costs (derived)

Per layer: attention `q` 51,380,224 + `k` 7,340,032 + `v` 7,340,032 + `o` 51,380,224 = 117.44 M parameters; MLP 3 x 157,810,688 =
473.43 M; layer 590.87 M. 52 layers = 30.72 B; embed + head 2 x 1,196,425,216 = 2.39 B; total 33.11 B (matches the index: 66.24 GB / 2).

| item | formula | bytes |
|---|---|---:|
| int4 g64 sym linears, per parameter | 0.5 B + a 2-byte scale per 64 = 0.53125 B | |
| one layer | 590.87 M x 0.53125 | 313.9 MB |
| 52 layers | 30.72 B x 0.53125 | **16.32 GB** |
| int8 head | 166,912 x 7168 x 1 B + 4 B x 166,912 scales | **1.20 GB** |
| bf16 embedding (a gather: not read per token) | 166,912 x 7168 x 2 | 2.39 GB |
| **resident weights** (one card) | 16.32 + 1.20 + 2.39 | **19.91 GB** |
| **bytes read per decoded token** | 16.32 + 1.20 | **17.52 GB** |
| KV, bf16, per position | 52 layers x 2 (K, V) x 8 kv x 128 x 2 B | **212,992 B** (208 KiB) |
| KV, int8 (rotkv, per-token-per-head scale), per position | 52 x 2 x 8 x (128 + 2) B | **108,160 B** |

The bandwidth ceiling: 17.52 GB / 638 GB/s = 27.5 ms = **36.4 t/s** at 100 % of the card's rated bandwidth. The engine does
not reach that: Qwen3.8 decode at depth 4096 reads (15.54 GB of weights + 4096 x 64 KiB) x 29.28 t/s = **462.9 GB/s = 72.5 % of
638** (measured, docs/BENCHMARKS.md "Decode at depth (F3)"), and decode attention v2 reads KV at **~466 GB/s** (derived from the
same table: +17.8 ms for 126,720 more positions x 64 KiB = 8.30 GB). So this spec's planning basis is **463 GB/s for weights
and 466 GB/s for KV**, which gives Nanbeige **37.84 ms of weights per token** (17.52 GB / 463) and **0.2321 ms of KV per 1,000
positions** (108,160 B x 1000 / 466 GB/s) at int8 KV.

## 3. What is reused, what is new (per op)

Reuse is by **copy under generic names, never by editing a shared source** (G0: every other model's binaries stay
byte-identical - the repo's precedent from K2 / Kolibri / Bolt). The new family is `dense` (descriptor `model::Dense`, kernels
`dn_*`, directories `src/{model,loader,runtime,kernels}/dense/`); it is selected by `config.json`'s `model_type: llama`, like K2
by `k2_horizon`. It validates the dimensions in section 1 and refuses any other Llama by name (kernel binaries are per shape).

| op | status | what it is / where it comes from |
|---|---|---|
| int4 g64 GEMV (`gemv`, layouts 0 / 1), int8 head (`gemv_i8w`), bf16 head (`gemv_bf16`) | **reuse binaries; add shapes** | new (K, N) cells in `src/kernels/CMakeLists.txt`: qkv 7168 x 9216, o 7168 x 7168, gate-up 7168 x 44032 (interleave-16), down 22016 x 7168, head 7168 x 166912. Split-K S must divide K/64: 112 (S in 1..16 power of two all fine) for K = 7168, **344 for K = 22016 (S = 16 is invalid; S <= 8)**. Per-shape S from a 29c probe, as `kQwen38Tuning` was |
| residual + norm (`prep_res_fold`, `prep_norm_finish`) | **new variant** | eps 1e-5 (the `1e-6f` literal is hardcoded in `prep.cl`, `attn.cl`, `kv8.cl`, `k2_prep.cl`: a new `EPS` define in the `dn_` copies), Llama's two-rounding form (a define; the only per-model difference in the norm) |
| `silu * up` | **new variant** | `prep.cl` allows INTERMEDIATE 17408 / 19456 only; K2 has its own (`k2_silu_mul`, SILU_I) - the `dn_` copy takes 22016 |
| attention prep (RoPE + KV write) | **new, from K2** | `k2_prep.cl`'s `k2_attn_prep`: head_dim 128, full-128 neox RoPE, no q/k norm, bf16 RoPE chain (HF applies cos / sin in bf16). Strip the softplus gate. The Qwen-path `attn_prep` is head_dim 256 / rotary 64 / q-k norm / sigmoid gate - unusable |
| decode attention | **new, from K2** | `k2_attn.cl` (spec 10's v2 at head_dim 128, GQA from `-DQ_HEADS -DKV_HEADS`, 16-position waves); `NPR = M x GQA` (q, row) pairs per wave: **GQA 7 is new** (K2 is 4, Qwen 6, Ornith 8; no kernel rejects it - the only check is q % kv == 0) and **K2's kernel is limited to M = 1..4** (`k2_attn.cl:68`), so the verify width M <= 8 needs the `dn_` copy widened (29c Task 4 for M <= 4 plumbing, 29f for 8) - a P0 check of the qpack SLM size and the 16-lane tree at 7 x M pairs (M up to 8 = 56 pairs). No gate in the reduce |
| int8 KV | **new, from K2** | `k2_kv8.cl` (head_dim 128, rotkv: Hadamard-rotated K and V, per-token-per-head scale). Quality was measured on Qwen3.8 only to 32k (docs/probe-int8-kv-qwen38-2026-10-06.md); 29c repeats it at depth |
| prefill attention | **new, from K2** | `k2_pf_attn.cl` (`k2_pf_flash_attn`, DPAS flash at head_dim 128; HPW heads per work-group must divide GQA: **HPW 1 or 7**, K2 used 4) - tile and HPW chosen by a 29d probe |
| prefill GEMMs | **reuse** | `l0` (the default backend; dequant slabs + bf16 DPAS GEMM). Check tile divisibility for K = 22016 (= 86 x 256, fine) and N = 44032 (= 43 x 1024). **`l0-int8` is refused**: its 1024-wide Hadamard rotation and 1024-column slabs (spec 5) do not divide K = 22016 (= 21.5 x 1024) |
| embedding gather, argmax | **new variants** | `embed_gather` HIDDEN 7168 VOCAB 166912; `argmax` VOCAB 166912, `VOCAB_USED` 166107 |
| RoPE table | **new** | host table, theta 1.2e8 (representable exactly in fp32), 128 dims. Two precedents differ: the Qwen path computes angles in double, K2's table in fp32 (`ang = float(p) * inv`, cos / sin then rounded to bf16 as HF does). At 300k+ positions the fp32 product errs by up to ~0.008 rad for the slowly rotating pairs - more than the bf16 rounding of cos / sin - so the two forms differ measurably. 29a measures both against the reference and the table follows the form the judge uses (open decision 9) |
| descriptor, loader | **new, from K2 / Bolt shapes** | `model::Dense` (plain struct, layer count, widths, vocab, tuning table); loader for the AutoRound g64 tensors under standard Llama names (the same `auto_round:auto_gptq` bytes Qwen3.8's loader reads), bf16 embed, int8 head by the existing `lm_head_int8` tiling |
| engine, capture, two-card placement | **new engine, patterned on Bolt's / Kolibri's own `--pp 2`** | the Qwen path's `PipelineEngine` cannot take an all-attention model (`require_split` refuses a stage without a GDN layer and one without an FA layer; `is_fa` is hardcoded `layer % 4 == 3`) and K2 is refused under `--pp`; the dense engine has its own 1- or 2-stage form from day one (section 5) |
| tokenizer, template, parsers | **reuse the code, new format kind** | the minja template renderer and the Qwen XML parser are used as they are; `ChatFormat` needs a `Dense` kind so `chat_template_kwargs` reach the template (the Qwen kind drops them); the `<think>\n` prompt ending is the Qwen reasoning parser's existing rule |
| speculative decoding | **new, gated** | section 7 |

**Why K2 and not Muse (operator's reuse ruling 2026-10-27 named Muse).** The ruling's intent is "revive from git under generic
names". K2's attention stack is **live in `main`** and already is what Nanbeige needs: all layers full attention, head_dim 128,
no q/k norm, full-128 RoPE, an int8-KV variant, a flash-prefill variant. Muse's attention differs (sliding window ring, NoPE, output
gate, weight-less q/k norm, GQA 16) and is mostly the same files K2 and Bolt already carry (`src/kernels/bolt/mg_attn.cl` is still
live and byte-identical). Reviving Muse's descriptor/loader pattern is optional: the revival parent is `99adbb0` (the old
spec's note names `4c491f8` and `c57e08e`; **`4c491f8` does not resolve** - the history was rewritten; `c57e08e` is an ancestor
of `99adbb0`). Muse's `src/model/muse_glimmer.{h,cc}` + `src/loader/muse_{layout,loader,repack}` are the dense
descriptor/loader shape; 29b may read them from git for structure and copies nothing it does not need (no gate, no sandwich norms,
no softcap).

## 4. Memory and context, per card (derived)

Card = 32.53 GB (`ctx.memory_bytes()`), planner reserve 1.5 GB (`kDefaultReserveGb`; an estimate the box has not confirmed) =
**31.03 GB usable**. Other per-card bytes, estimated (the planner prints the real ones): RoPE table 0.18 GB at 355k positions
(512 B per position), decode state 0.3 GB (no GDN, no conv ring), prefill scratch ~0.7-0.8 GB at the engine's chunk of 2048 (the planner's eager terms for hidden 7168 / FFN 22016 sum to ~0.78 GB; Qwen3.8's measured line is 0.700 GB; ~1.5 GB if chunk 4096 lands) = **~1.2 GB; this spec budgets 2.0 GB** to stay conservative.

KV per layer-position: bf16 4,096 B, int8 2,080 B.

| configuration | weights on the card | KV room | positions | formula |
|---|---:|---:|---:|---|
| one card, bf16 KV | 19.91 | 9.12 GB | **~43k** | 9.12e9 / 212,992 |
| one card, int8 KV | 19.91 | 9.12 GB | **~84k** | 9.12e9 / 108,160 |
| `--pp 2` split 26 / 26, bf16 KV | d0 10.55 (embed 2.39 + 26 x 0.3139), d1 9.36 (head 1.20 + 26 layers) | d0 18.48, d1 19.67 GB | **~173k** (d0 binds) | 18.48e9 / (26 x 4,096) |
| `--pp 2` split 25 / 27, bf16 KV | d0 10.24, d1 9.67 | 18.79, 19.36 | **~172k** (d1 binds) | 19.36e9 / (27 x 4,096) |
| **`--pp 2` split 26 / 26, int8 KV** | as above | 18.48, 19.67 | **~342k** (d0 binds) | 18.48e9 / (26 x 2,080) |
| **`--pp 2` split 25 / 27, int8 KV** | | 18.79, 19.36 | **~345k** (d1 binds) | 19.36e9 / (27 x 2,080) |
| `--pp 2`, a split balancing the two (~25.5 layers) | | | **~352k** | the bytes-balanced `auto` split |

- **242k needs int8 KV and two cards** (bf16 KV tops out near 175k even on two). **330k fits with ~5 % to spare** (330k x 54,080 B
  = 17.85 GB against 18.48 GB on d0). The 512k trained length never fits.
- With the DFlash drafter loaded (section 7), each card loses ~0.5-1 GB (int8 drafter 0.98 GB on one card, its windowed KV and
  tap buffers): ~10-18k positions. A **full-context** drafter KV (the vendor's drafter attends the whole context) costs 6,144 B per
  position (3 layers x 2 x 4 kv x 128 x 2 B) = 2.0 GB at 330k, + 3.2 ms a step at 242k (derived) - section 7.3 chooses.
- The planner is the one that decides: the spec's numbers are the bar the planner's printed line is checked against (29b).
- The prefix cache is pinned **host** RAM, not in the device plan (`prefix_cache_size.h`: min(32 GiB, MemTotal / 2,
  MemAvailable - 8 GiB)). A full int8 session of 242k positions is 26.2 GB: one session fits the 32 GiB default
  (34.4e9 / 108,160 = ~318k positions); restoring it is a host-to-device copy at the measured 12.2-12.9 GB/s: ~2.1 s (derived),
  against 14-28 minutes of cold prefill. This is not per-token offload: it is a per-request cache restore.

## 5. The engine shape: `DenseEngine`, 1 or 2 stages

- **Two stages are native.** Stage 0: embedding, layers [0, s); stage 1: layers [s, 52), final norm, head, argmax. The cut is
  at a layer boundary after `prep_res_fold` (the folded residual rows + norm sums cross: 14,336 B + the sums per token, ~14.5 KB;
  prefill ~29.4 MB per 2048-row chunk) - spec 16's cut, rebuilt in the dense capture. Hand-off: `copy` (the copy engine +
  cross-device event; the default) with `peer` (`pp_handoff.cl`, generic and reused) as the opt-in, as spec 16.
- **Split:** bytes-balanced `auto` over (weights + KV at the chosen `--max-len` + scratch) per card - the logic of `pp_balance`,
  but written for the dense engine because the existing planner refuses an all-FA split. Result for 330k int8: s ~ 25-26.
- **One card is the same engine with one stage** (short contexts, `--max-len` <= ~80k int8), so every kernel and test runs on
  one card first (29c's gates), then two.
- **Single-stream decode over two stages is not faster than one card** (the layers run in sequence; spec 16 §1): PP buys
  capacity, not speed. The hand-off is ~14.5 KB per token over PCIe 3.0 (a few microseconds; spec 16 has never run it on the cards - 29c measures it).
- **Prefill is pipelined by chunk** (device 0 on chunk c + 1 while device 1 runs chunk c; spec 16c's back-pressure rule, two landing
  slots): derived speed-up up to ~2x at long prompts (spec 16 derived 1.88x at 32768; not measured).
- **No `--batch`** (spec 13 §9.1: a model without a batched engine refuses `--batch > 1` by name). At 242k+ there is no memory for a
  second slot anyway.
- **TP is a follow-on, not built here** (section 8): the descriptor exposes per-layer head counts through a `tp_rank / tp_size`
  view so spec 17 can retarget it without a rewrite; nothing else.

## 6. Decode speed (derived; formulas shown)

Per token, one stream, plain decode, int8 KV, `--pp 2` (so two stages' work in sequence plus the hand-off):

`T(D) = 37.84 ms (weights) + 0.2321 ms x (D / 1000) (KV) + 0.1 ms (hand-off, estimated)`.  Check of the method against Qwen3.8's
measured v2 row at 130,816: the same formula (15.54 GB / 463 GB/s + 8.57 GB / 466 GB/s = 52.0 ms) predicts 19.2 t/s, measured
20.13 t/s (+4.8 %) - so the formula is slightly pessimistic.

| depth D | KV ms | T ms | plain t/s | **with spec (tau 3.0, agentic)** | **with spec (tau 2.0, general)** |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.95 | 38.9 | **25.7** | 61.7 | 41.1 |
| 32,768 | 7.61 | 45.6 | **21.9** | 52.9 | 35.3 |
| 65,536 | 15.21 | 53.2 | **18.8** | ~46 | ~31 |
| 131,072 | 30.42 | 68.4 | **14.6** | 35.1 | 23.4 |
| 200,000 | 46.42 | 84.4 | **11.8** | ~28 | ~19 |
| 242,000 | 56.17 | 94.1 | **10.6** | 25.5 | 17.0 |
| 330,000 | 76.60 | 114.5 | **8.7** | 21.0 | 14.0 |

Speculative column: `t/s = tau / T_step`, `T_step(D) = 37.84 x 1.10 (M = 8 weight rows, spec 28a's stand-in: 1.083-1.10 measured on Qwen shapes) +
0.2321 x D/1000 x 1.25 (M = 8 attention rows share one K/V walk; 28c's number is not in yet - estimated) + 5.7 ms (drafter, 0.15 plain steps, estimated) + 0.1`.
Examples: D = 4,096: 41.62 + 1.19 + 5.7 + 0.1 = 48.6 ms; D = 242,000: 41.62 + 70.2 + 5.7 + 0.1 = 117.6 ms; D = 330,000: 41.62 + 95.8 + 5.8 = 143.2 ms.
**tau (accepted tokens per verify step) is the unknown**: estimated 3.0 on agentic coding (the vendor's published agentic
speed-up of 2.3-2.6x on a GPU where an 8-row verify costs ~1.1-1.2 steps implies tau ~2.8-3.2 with their drafter), 2.0 on general text (their 1.76x),
lower on prose. The z-lab drafter on Qwen3.8 measured 7.21 tokens per step teacher-forced on tool calls and ~4.1 on the card
(docs/probe-dflash-2026-10-08.md; BENCHMARKS) - a different drafter; not carried over. 29f measures it first.

Reading the table: at depth the speed is **KV-bound** (at 162k positions the KV read equals the weights'), so a two-card context of 330k
decodes at 8.7 t/s plain. Three levers, in order of size: (1) **speculation** (verify rows share the KV walk: x2.4 at 330k), (2) **tensor
parallel** - each card reads half the weights and half the KV (4 kv heads each): derived `T = (37.84 / 0.9 + 76.6) / 2 + 104 x 33 us`
= 62.7 ms plain at 330k = **16 t/s**, and ~37 t/s with speculation (all-reduce at spec 17's published 33 us each; a go / no-go that spec 17's P0 owns),
(3) a faster int8-KV decode attention than 466 GB/s (not planned). The dense family is TP's easiest case (no GDN state), which is
why the descriptor keeps the hook; the spec does not depend on it.

Plain decode at short depth is the unavoidable cost of a dense 33 B at 638 GB/s: ~26 t/s is **below Qwen3.8's 31.2** and below its
MTP rate (~49 t/s on tool calls); Nanbeige pays only with its drafter or lookup, and in context. That is the trade the operator
chose it for (quality in agentic coding: the vendor's numbers above) and section 11 states it plainly.

## 7. Speculative decoding

### 7.1 What exists

| proposer | exists for Nanbeige | how it plugs in |
|---|---|---|
| MTP | **no head** | not applicable (`--mtp` refused by name) |
| lookup (prompt lookup, spec 19e) | **works with any target, no weights** | host matcher + a verify list at M <= 8; its measured value is on copy/edit-heavy agentic turns: lookup alone gave **1.44x over plain** on the A4 one-shot tool calls (derived, docs/probe-prompt-lookup-2026-10-05.md - to be re-measured here). Strong for agentic coding (file contents repeat) |
| DFlash2 (spec 19) | **a third-party drafter exists**: `ProCreations/Nanbeige4.3-33B-DFlash2`, Apache 2.0, BF16, 979,731,200 parameters, "v2 with more training data is planned", training stopped at 99.66 % of its schedule | section 7.2 |
| a small draft model | the only same-tokenizer-family model, Nanbeige4.2-3B, is a looped transformer with vocab 166144 and another post-training | **not planned**: unknown acceptance, a second engine, and a 3 B bf16 model is 6 GB read per draft token. Recorded as an option if the DFlash2 drafter disappoints |

### 7.2 The drafter against the engine's DFlash2 build (compared field by field, 2026-11-01)

The vendor's `dflash_config` and the z-lab Qwen3.8 drafter we run are the **same family and the same config schema**
(`DFlash2DraftModel`, block 8 with one verified anchor, dynamic convolution 2 taps in groups of 16, a low-rank selector, mask token,
target embedding and untied head shared, neox RoPE). Every dimension differs:

| field | z-lab Qwen3.8-27B-DFlash2 (built) | ProCreations Nanbeige4.3-33B-DFlash2 |
|---|---|---|
| layers / hidden / FFN | 5 / 5120 / 17408 | **3 / 7168 / 8192** |
| heads q / kv x dim | 32 / 8 x 128 | **28 / 4 x 128** (q width 3584 < hidden: o-proj K = 3584) |
| taps (zero-based layer outputs, before the final norm) | [5, 19, 33, 47, 61] of 64 | **[1, 25, 49] of 52** (fc 21,504 -> 7168) |
| selector rank / top-k | 256 / 16 | **128 / 32** |
| mask token | 248070 | **0** (`<unk>`) |
| attention | sliding window 2048 (every layer) | **`full_attention`, `sliding_window: null`** - the drafter attends the whole context |
| rope theta / eps | 1e7 / 1e-6 | 1.2e8 / 1e-5 |

What the build hardcodes and 29f must generalise (read from the code): `kTopK = 16` (`dflash_dims.h`); `Shape kQwen38` and `fc = 5 x H`
(`dflash_kernels.h`); `check_common` pins `layers == 5`, `taps == 5`, the Qwen3.8 dims, every layer a 2048-window non-causal layer, eps 1e-6,
group 16 / 2 taps / rank 256 / top-k 16 (`dflash_capture.cc`); `--draft-model` defaults to the z-lab id and `--kv-cache int8` is refused
with `--spec dflash` (`dflash_args.h`). Generic already: the descriptor parser (`model/dflash.cc`), the loader layout (q_n / kv_n),
the host RoPE table from theta, the mask id from config, `TapDrafter` and the host policy code (AdaptiveK, lookup matcher,
accept rules). The target side (`Engine`, verify lists with 8 GDN state slots) is Qwen-specific; for an attention-only target it is **simpler**:
verify writes KV at n..n+k, commit just sets `pos = n + j + 1`, no state rolls back.

### 7.3 Decisions for 29f (P0 first)

1. **Acceptance P0 (before any kernel):** run the drafter in a Python reference (`tools/oracle/dflash_ref.py` extended) on taps dumped by
   the engine's own teacher-forced decode (29c) over golden / A4 / an agentic session, **at depths 4k, 32k, 128k and 240k** with
   (a) the window 2048 the built drafter supports and (b) full context. tau per corpus per depth, per-position acceptance.
   Bar: **tau >= 2.5 on A4 and the agentic session at 32k with (a) or (b)**, else DFlash is not built and lookup alone ships.
2. **Window or full context:** (a) reuses the built ring (2048, a fixed small KV, no depth cost); (b) needs a new drafter attention
   over a growing KV (+6,144 B / position, +3.2 ms a step at 242k, derived) and +3-4 % prefill work for the drafter's fc + 3 layers over
   every context token. Chosen by P0: (b) only if its tau at 128k+ beats (a)'s by >= 0.4.
3. **`--kv-cache int8` with `--spec dflash`:** today refused. Nanbeige's context targets need int8 KV, so 29f lifts the refusal for the
   dense family (the drafter's KV is its own; the tap rows are bf16 residuals either way).
4. **Drafter weights:** int8 first (the built path, 0.98 GB), then AutoRound int4 g64 sym of the vendor's weights **by us** (one-format
   rule; spec 19 measured w4a16 acceptance on the z-lab drafter) - 0.52 GB, a quantisation job for the operator, not an engine
   task. Its embedding / head are the target's (nothing duplicated).
5. **Placement under `--pp 2`:** the taps at layers 1 and 25 live on stage 0 (split >= 26) and 49 on stage 1; the drafter lives on stage 1
   with the head (the argmax, the draft-vocab share) and receives layer-1/25 tap rows with the hand-off (28 KB a token extra) - the
   anchor token's embedding row is gathered on stage 0 and sent with the hand-off. Spec 19's `--pp 2` DFlash path has **never run on a card**
   (docs/19-running-models.md): this is an extra risk, named in 29f's gate.
6. **Cost table:** `MtpCost::dflash` rows measured on the dense verify list at M = 1..8 (not extrapolated); `AdaptiveK` picks K.
7. **Lookup policy:** spec 19f's lookup-in-DFlash policy (ENTER on a >= 8-token match beyond the drafter's window, STAY while each
   verify accepts >= 3) is built **for Qwen3.8 first** (plan 19f, not started); the dense family takes whatever 19f ships. Until then
   Nanbeige's `--spec lookup` and `--spec dflash` are separate modes (refusing each other, as today).
8. **Verify width:** M <= 8 today (`DecodeBuffers::kM`). Spec 28b's `gemm_sm` (M 2..8 and 16) is a build dependency for a good verify
   (without it a dense M = 8 step is ~3.1x plain on the GEMVs, spec 28 §1), and Nanbeige's shapes are **added to 28b's shape table**
   (29c Task 3 lists them; 28b's operator-approved scope is Qwen3.8 + Agnes, so this is an addition the operator confirms). Spec 28c's verify
   attention by shape (M = 8) applies to the dense decode attention as well and is its own plan.

### 7.4 What speculation changes in the budget

The drafter costs 0.5-1.0 GB and ~5.7 ms a step (derived: 0.98 GB / 463 GB/s = 2.1 ms of weights + launches, estimated 0.15 plain
steps). Everything in section 6's right-hand columns is conditional on tau; the **plain** column is the floor.

## 8. Tensor parallel and batching (what spec 13 / 17 imply)

- **Batching (spec 13):** not for this family now. Per-slot cost would be KV only (no GDN state), but one 242k session leaves no room
  for a second; `--batch > 1` refuses by name.
- **Tensor parallel (spec 17):** retargeted at Qwen3.8 first, **not started**, P0 pending (r22 rows). If its P0 passes (all-reduce <= ~4 ms
  a token at 128 per token; the dense Nanbeige has 104), the dense family is its best fit: every linear splits by heads / columns, KV halves per card
  (4 kv heads each), no recurrent state. Derived gain at 330k: 8.7 -> ~16 t/s plain. The dense engine's descriptor stays splittable
  (head counts and N / K per rank derived from a view); building it is **spec 17's plan for this family, after Qwen3.8's**.

## 9. Prefill (derived)

Linear work 2 x 30.72 B = 61.4 GFLOP a token. Qwen3.8's measured pp4096 (2,125 t/s on `l0-int8`, ~51.4 GFLOP a token) is an effective
**~109 TFLOP/s**, so Nanbeige's linears run at **~1,780 t/s** (derived; `l0` bf16 may differ - 29d measures it). Attention (all 52 layers):
causal FLOPs `2 x 7168 x 52 x N^2 = 745,472 N^2`; the measured flash kernel runs **28.3 TFLOP/s** (head_dim 256, 15 % of peak, BENCHMARKS
"Flash attention and 128k"; head_dim 128 not measured).

`T_cold(N) = N x 61.4e9 / 109e12 + 745,472 N^2 / 28.3e12`

| prompt N | linears s | attention s | cold, one card | `--pp 2` pipelined (about half) |
|---:|---:|---:|---:|---:|
| 4,096 | 2.3 | 0.4 | 2.8 s (1,490 t/s) | ~1.5 s |
| 32,768 | 18.5 | 28.3 | 47 s | ~25 s |
| 65,536 | 36.9 | 113 | 150 s | ~80 s |
| 131,072 | 73.9 | 453 | 8.8 min | ~4.5 min |
| 242,000 | 136 | 1,543 | **28 min** | **~14 min** |
| 330,000 | 186 | 2,869 | **51 min** | **~26 min** |

**A new turn in a long session** (c = 4,096 new tokens at depth D): attention `4 x (D + c/2) x c x 7168 x 52`; D = 200k: 1.23e15 FLOP = 43.6 s at
28.3 TFLOP/s + 2.3 s of linears = **46 s** on one card, ~23 s with pipelining; with the spec 6 F2 target (60 % of peak = 110 TFLOP/s) 13.5 s / ~7 s.
Consequences for the design:

1. **Session continuation (the live KV of the previous turn) and the prefix cache are required, not optional**; both are KV-only
   here (`state_bytes() == 0`, K2's zero-state snapshot precedent), so they are cheap. 29e's gate includes a 200k-token session replayed turn by turn.
2. **`preserve_thinking`: the template's default (undefined) keeps old reasoning, which keeps the history byte-stable and the prefix
   cache valid; an explicit `false` rewrites older turns and invalidates it every turn.** The server never sets it; clients that do pay a cold prefill.
3. **The flash-prefill kernel at head_dim 128 / GQA 7 with long depth is the largest prefill lever** (~4x if it reaches 60 % of peak). 29d
   builds the K2-derived kernel, measures its TFLOP/s at 32k / 128k / 240k and records the gap; a faster kernel (split-d / SLM-staged, spec 11's levers)
   is a follow-on decision, not part of this spec.
4. Chunk size: the engine's chunk is 2048 today (ruling A13); 4096 only if plan 5d has merged by then.

## 10. Reference, oracle and gates

**The judge is an f32 CPU reference of the engine's own weights** (the int4 g64 checkpoint dequantised `(q - 8) x scale`, every op in fp32,
eager attention, RoPE in the table's form) - spec 15 / 18's "f32 judge" arrangement. **The bf16 twin** is the same dequantised weights run
as a plain bf16 `transformers` `LlamaForCausalLM` (eager attention): it measures how far "ordinary bf16 execution" sits from the f32 judge,
and the engine must be no worse than that on both counts below.

**The near-miss rule (operator's standing rule, 2026-11-01; docs/14-golden-gate.md still carries the one-ulp judge rule of Ornith / K2,
and 29a amends the doc):** on the golden decision rows, (a) every row where the engine's argmax differs from the f32 judge's is a **near-tie** - the judge's
logit gap between the two ids is within **8 bf16 ulps** at the logit's magnitude (`bf16_ulp_at = 2^(e-8)`, as `golden_common.h`) - and the
engine has **no more such misses than the bf16 twin** (its misses beyond the twin's are the "extra misses" and each must be within 8 ulps); (b) the
logits' cosine distance to the judge (1 - cos), averaged over the decision rows, is **<= 2x the twin's**. Greedy token equality against the
twin is information, not a bar (a different rounding path legitimately diverges after a near-tie).

| gate | what | bar |
|---|---|---|
| **N0 / G0** | every other model's binaries checksum-identical (`g0.sha` lists only added `dn_*` / dense-shape binaries), the whole suite green, `tools/kernel_cmdlines` additions only | bitwise, never traded |
| **D1 loader** | the dequantised weights read back from the card equal the host dequant of the safetensors | bitwise, every tensor |
| **D2 kernels** | each `dn_*` kernel against its fp64 / fp32 host twin at head_dim 128 / GQA 7, M = 1 and M = 2..8 | fp32 accumulation bounds as spec 28 N2; the attention at depth to 240k within the spec 6 bound |
| **D3 golden** | prose / code / cjk (42 / 61 / 38 ids, 32 greedy ids each), engine decode and prefill, `l0` | the near-miss rule |
| **D4 depth** | `golden-at-depth`: a 4k and an 8k prompt vs the f32 judge (the CPU judge at 32k+ is day-class; opt-in only); at 32k / 128k / 240k / 330k **engine vs engine** (bf16 KV vs int8 KV, one stage vs two) and **passkey at 5 / 50 / 95 %** (240k and 330k ids, int8 KV) | passkey 3/3 at both lengths; int8-vs-bf16 KV logits cosine distance recorded per depth (the Qwen3.8 floor: 5e-5) |
| **D5 two cards** | `--pp 2` against one card (a short context): bitwise equal logits, MTP-free; the 330k run finishes | bitwise |
| **D6 tokenizer / template** | `tokenizer.json` vs the slow `LlamaTokenizer` on >= 10,000 strings (code, CJK, XML, runs of spaces and newlines, byte fallback) and the 8,003-char template rendered byte for byte against transformers for >= 40 conversations (system / tools / tool calls / `preserve_thinking` / `enable_thinking`) | id for id, byte for byte |
| **D7 tool calls** | the A4 set (36 scenarios) rendered for Nanbeige, engine vs the f32 judge's references, scored by `toolcall/score.py` | zero parser-level errors; call-level correctness not below the bf16 twin's |
| **D8 agentic** | a recorded long coding session (>= 150k tokens, written for this repo, no third-party content) replayed turn by turn through `b70-serve` with session continuation; every tool call parses; the transcript's time per turn recorded | all calls parse; no turn re-prefills the whole history |
| **S1-S4 speed** | plain decode at depth 4k / 32k / 128k / 240k / 330k (section 6's table is the **estimate**, the measured row replaces it); prefill pp4096 / 32768 / 131072 / 240000; verify step at M = 1..8; speculative speed per corpus and depth | recorded; **S-bars set after 29c / 29d's first measurements, never below 85 % of section 6 / 9's derived plain numbers** (an honest estimate miss is reported, not hidden) |

## 11. Risks and open decisions

| # | risk | mitigation / owner |
|---|---|---|
| R1 | GQA 7 never built: decode attention `NPR = 7 x M` pairs, flash HPW must be 1 or 7 | 29c Task 1 and 29d Task 1 probe it on the card before any engine work |
| R2 | decode at 242k+ is KV-bound (~10.6 / 8.7 t/s plain) | speculation (section 7), TP as a follow-on (spec 17); stated to the operator, not hidden |
| R3 | cold prefill at 242k+ is 14-28 min | session continuation + prefix cache (29e), the flash kernel's TFLOP/s measured (29d); a new long session starts slowly - the operator should know |
| R4 | int8 KV beyond 32k never measured | D4 (passkey, depth cosine); fallback is a smaller context with bf16 KV (~172k) - which misses the 242k bar |
| R5 | the third-party drafter's acceptance, window / full-context behaviour, and its v1 status ("more training data planned") | 29f P0 with a stop rule; lookup alone ships otherwise |
| R6 | tokenizer: SentencePiece + `add_prefix_space` through our Rust `tokenizers` Metaspace path is untested (no code or test in the repo mentions it) | D6 first thing in 29a; a mismatch is fixed in the tokenizer JSON handling before anything else |
| R7 | `--pp 2` has never run on a card for any model (spec 16 blind-built, rows 22 / 23 / 27 pending) | 29c brings its own two-stage engine and measures the hand-off; the two-card gates run late in 29c, after one-stage gates pass |
| R8 | 29f depends on 28b / 28c / 19f (none built) | 29a-29e do not; 29f starts when its predecessors are merged or the operator waives one |
| R9 | HF Llama's two-rounding RMSNorm is not what our kernels do | a define in the `dn_` copies (29c Task 2), selected by the reference check; a measurement, not an assumption |
| R10 | the planner's reserve (1.5 GB) is unconfirmed; the context numbers move by ~14k positions per GB | the planner's printed line is the truth; D4 runs at the planner's chosen length |
| R12 | the K / V buffers are ONE allocation per stage: int8 K at 355k = 26 x 355,000 x 1,024 B = **9.45 GB** (bf16 K 18.9 GB); `ZE_RELAXED_ALLOCATION_LIMITS_EXP` is proven on this box only to ~8 GB | 29b Task 5 allocates, fills and reads back a 10 GB buffer on the card first; fallback: one buffer per layer group (a descriptor change, not a kernel change) |
| R11 | box RAM for the f32 judge: a 33 B model streamed layer by layer needs ~10 GB working set but CPU time is days at depth | oracle runs are detached, resumable, one at a time (`/tmp/b70-cpu-model.lock`), RAM checked first (standing rule) |

**Open decisions (operator):**

1. **Spec 28b's scope:** add Nanbeige's five shapes to `gemm_sm` (29f depends on it)? Proposed **yes**.
2. **Drafter:** use the vendor's third-party DFlash2 drafter (Apache 2.0, "v2 planned") and re-quantise it ourselves, or wait for v2? Proposed: P0 on v1 now (29f), decide the quant on its acceptance.
3. **Drafter attention window (7.3 item 2):** window 2048 (built, free) or full context (new kernel)? Proposed: P0 decides by the 0.4-token rule.
4. **Cold-prefill target:** accept 14-26 min for a cold 242k-330k prompt on `--pp 2`, or fund the flash-prefill rework (a follow-on spec)? Proposed: accept for now, measure in 29d, decide with the numbers.
5. **TP:** build for this family after spec 17's Qwen3.8 P0, or not? Proposed: the descriptor keeps the hook; the decision waits for spec 17's P0.
6. **Muse revival path:** K2-derived kernels as proposed, Muse's dense descriptor/loader read from git for structure only (`99adbb0`)? Proposed **yes**; the operator's 2026-10-27 wording named Muse's attention / GEMV / loader shapes.
7. **Near-miss rule text:** the "8 ulp / bf16 twin / 2x cosine" rule is not in docs/14-golden-gate.md; 29a writes it there as stated in section 10. Confirm the twin's definition (the same dequantised weights in plain bf16 transformers).
9. **RoPE table precision:** fp32 angle (HF's form, K2's precedent) or double (Qwen's)? Proposed: whichever the f32 judge uses, measured in 29a at 1k..355k positions.
8. **`--chat-template` / `preserve_thinking`:** keep the template default (reasoning preserved) and give clients no server-side switch (proposed), or add a flag? Proposed **no flag**.

## 12. Plans (one implementing agent per plan; box stages in prose; effort in implementing-agent sessions)

| plan | what | needs | effort |
|---|---|---|---|
| **29a** facts + reference | transformers llama code pinned (norm rounding, RoPE chain, softmax), the f32 judge and bf16 twin (`tools/oracle/dense_ref.py`), tiny random Llama and tiny int4 g64 export builders, the facts sheet, golden / A4 / agentic prompt sets, tokenizer + template parity (D6), RoPE-table measurement, docs/14 amendment | nothing (Mac); box CPU for real-weight references | **M**: 8 tasks, 1 session on the Mac, box CPU runs in the background (hours to days, resumable) |
| **29b** descriptor + loader + memory plan | `model::Dense`, loader (AutoRound g64 under Llama names, bf16 embed, int8 head), synthetic checkpoints, the dense memory plan and the 1- / 2-stage placement, `b70-decode` dispatch up to load | 29a's tiny builders | **M**: 6 tasks, 1 session |
| **29c** decode | P0 on the card (GQA 7 attention; shape tuning; hand-off cost), `dn_*` kernels, `DenseEngine` 1 and 2 stages, int8 KV, `b70-decode`, D1-D3 / D5, S1 | 29a, 29b; box (GPU) | **L**: 9 tasks, 1-2 sessions + box time |
| **29d** prefill | flash prefill at head_dim 128 / GQA 7 (HPW probe), `l0` GEMM shapes, chunked prefill + int8 KV write, pipelined 2-stage prefill, D3 / D4 prefill half, S3 | 29c; box | **L**: 6 tasks, 1-2 sessions + box time |
| **29e** serving | `Dense` chat format, tokenizer wiring, sampling defaults, `b70-serve` (1- and 2-stage adapters), session continuation + prefix cache (KV-only), D6-D8, S4 | 29c, 29d; box | **M**: 6 tasks, 1 session + box |
| **29f** speculative | acceptance P0, lookup on the dense list, DFlash2 generalisation (shapes, taps, top-k, full-context option), `--kv-cache int8` with it, the cost table, speed bars | 29e, **28b, 28c, 19f** | **L**: 8 tasks, 1-2 sessions; starts only after its dependencies |

Order: 29a -> 29b -> 29c -> 29d -> 29e, then 29f; the operator is asked before each (one implementing agent at a time). The first
card contact is 29c's P0 (a few hours of box time); no plan before it needs a GPU.

**Box stages in prose.** 29a: CPU-only runs on the box - perplexity of the dequantised weights, the golden / A4 references in f32 and the bf16
twin (days, resumable, detached), the RoPE-table comparison (minutes). 29b: the real checkpoint uploaded to one card and read back,
the planner's printed lines checked against section 4. 29c: P0 (attention kernel at GQA 7, shape tuning, one hand-off timing), then
golden decode on one card, then two cards, int8 KV, depth rows. 29d: prefill rows, the flash kernel's TFLOP/s at depth, the 240k and 330k runs. 29e: the
served golden, the A4 set, the agentic replay. 29f: the acceptance P0 on dumped taps, the verify cost table, the speculative rows.
All timed stages follow the standing protocol (idle box, interleaved pairs after a warm-up, the GPU lock).

## 13. Out of scope

Batching; the Nanbeige4.2-3B family and its DSpark drafter; Nanbeige's FP8 / INT8 / INT4 / third-party quantisations; NVFP4; tensor parallel (spec 17's plan); a faster
flash-prefill kernel beyond what 29d measures; anything beyond 524288 positions; host-RAM offload of weights or KV on the per-token path; new CLI flags beyond those
the existing flags already cover (`--pp 2`, `--max-len`, `--kv-cache int8`, `--spec`, `--draft-model`).
