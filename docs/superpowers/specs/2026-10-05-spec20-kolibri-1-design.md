# Spec 20 - Kolibri-1 (Aleph Alpha), German and English, on two B70s

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.

**Why this model:** Aleph Alpha's Kolibri-1 is a German- and English-focused reasoning MoE with tool
calling, Apache-2.0, released 2026-10-03. Spec 20 serves it in this engine's own format (int4 g64
symmetric, AutoRound) with a quantisation chosen to keep its German.

**Order:** after spec 16 (pipeline parallel): at int4 the model does not fit one card (§3). Stages
20a-20b (the reference and the quantisation) need neither the card nor spec 16 and can run first.

Every number is **measured** unless marked **derived**, **estimated** or **published** (from the
model cards and configs cited).

## 1. The model

From `Aleph-Alpha/Kolibri-1-BF16`'s `config.json`, index and shard headers, the model card, and the
reference implementation in Aleph Alpha's vLLM plugin (`aleph_alpha_inference/kolibri1.py`,
Apache-2.0, 329 lines on top of vLLM's Qwen3-MoE):

| | |
|---|---|
| architecture | `Kolibri1ForCausalLM`, `model_type: kolibri1` |
| parameters | 78.1B total, 3.46B active per token (published) |
| layers | 50, every one MoE |
| hidden | 2560, RMSNorm eps 1e-6 |
| attention | 48 q / 4 kv heads (GQA 12), head_dim 128, q/k RMSNorm per head, scale head_dim^-0.5, no bias |
| attention pattern | 4:1 sliding:full (`layer_types`): 40 sliding-window layers, window 513, **with** RoPE (theta 1e4); 10 full-attention layers **without** positional encoding (NoPE) |
| norms | sandwich: `input_layernorm` → attention → `post_attn_norm` → residual add → `post_attention_layernorm` → MoE → `post_ffn_norm` → residual add |
| MoE | 384 routed experts, top-6, expert intermediate 512, SiLU; one **ungated** shared expert (intermediate 512), added to the routed sum |
| router | `mlp.gate` bf16 [384][2560], logits in **fp32**; select top-6 on `logits + expert_bias` (`moe.router.expert_bias`, bf16 [384]); weights `sigmoid(logits)` of the selected, **not** renormalised (`norm_topk_prob: false`) |
| vocab | 128000 (a tokenizer built for German word structure), untied embedding / `lm_head` (bf16); `head_dtype: float32` in config |
| tokens | ChatML (`<|im_start|>` / `<|im_end|>`), `<think>`, `<tool_call>` / `<tools>` / `<tool_response>`; EOS [127906, 127901]; no BOS |
| context | trained to 262144 natively; NoPE full layers extend it (validated to 1,048,576, published) |
| sampling defaults | T 1.0, top-p 0.97, top-k 128 |

**Reference sources.** The vLLM plugin is the semantics. **transformers has no implementation:**
only an open "New model" request (huggingface/transformers#49281, 2026-10-03, no PR, no comments).
A CPU implementation with goldens exists in `mudler/vllm.cpp`; it may be used as a cross-check only
after its licence is confirmed, never as a source.

## 2. What it costs on the card (derived)

| part | parameters | int4 g64 (4.25 bit) | bf16 |
|---|---:|---:|---:|
| routed experts | 50 x 384 x 3 x 2560 x 512 = 75.5B | 40.1 GB | - |
| shared experts | 0.20B | 0.10 GB | 0.39 GB |
| attention | 1.70B | 0.90 GB | 3.40 GB |
| embedding, `lm_head` | 0.66B | - | 1.31 GB (head int8: 0.33) |
| **total** (experts and attention int4, head int8, embedding bf16) | | **~42.3 GB** | |

- **It does not fit one 32.5 GB card. Two cards with spec 16's pipeline split hold ~21 GB each.**
- **KV is small:** only the 10 full layers grow with context: 10 x 4 x 128 x 2 x 2 B = **20 KiB
  per position** (5.4 GB at 262144, 21 GB at 1M). The 40 sliding layers keep a 513-position ring,
  ~42 MB in total.
- **Read per token:** active experts 50 x 6 x 3 x 2560 x 512 x 0.53 B ≈ 0.63 GB, shared 0.1 GB,
  attention 0.9 GB (int4) or 3.4 GB (bf16), head 0.33 GB: **~2.0 GB per token** with int4 attention,
  ~4.5 GB with bf16 attention. Pipeline stages run in sequence, so the bytes add up across the two
  cards: a roofline near **250 t/s** with int4 attention. The real figure is set by launch counts
  and the hand-off (spec 16).

## 3. Quantisation for German (stages 20a-20b)

**Why German needs care (published).** Against bf16, Aleph Alpha's own FP8 loses about six times
more KL on German than on English (KL 0.060 DE vs 0.010 EN, top-1 92.8 % vs 97.2 %), and the
community GPTQ quant (`Padakovec/Kolibri-1-W4A16-GPTQ`: compressed-tensors, experts only,
asymmetric int4 g128, calibrated on 1M tokens of English chat + German Wikipedia) is at KL DE
0.069 / top-1 91.5 %. These are the bars to beat.

**The procedure:**
1. **Source:** `Aleph-Alpha/Kolibri-1-BF16` (156 GB), never the FP8 release (no double rounding).
2. **A transformers-compatible model class first** (20a): AutoRound (like llm-compressor) needs one
   and none is published. Ported from the vLLM plugin with an Apache-2.0 header naming it.
3. **German-heavy calibration matching the use:**
   - at least 50 % German, chat-formatted through the model's own template, with reasoning traces,
     tool calls and long documents; plus English chat and code;
   - preferably **self-generated**: the bf16 model answering German and English prompts, so the
     calibration matches its real output distribution;
   - sequences of 2-4k tokens.
4. **Enough tokens for 384 experts:** top-6 over 1M tokens gives each expert ~16k on average, but
   routing is uneven, and German-specialised experts may barely appear in an English-heavy mix. Use
   **2-4M tokens**, log per-expert token counts per layer, and top up experts below a floor (e.g.
   2048 tokens) with targeted German text.
5. **Precision by role:**
   - router weights and `expert_bias`: bf16 (required);
   - shared experts: bf16 (every token uses them; 0.39 GB);
   - attention: **(decide)** int4 g64 if 20b's KL allows, else bf16 (2.5 GB more read per token);
   - routed experts: int4 g64 symmetric (`--group_size 64 --sym`);
   - embedding and `lm_head`: bf16 in the checkpoint (the engine builds the int8 head at load,
     spec 9).
6. **Arms (20b):** RTN baseline (`--iters 0`), then AutoRound; attention int4 vs bf16; calibration
   German-heavy vs balanced. **Measure** KL and top-1 vs bf16 separately for German and English on
   held-out text (German news / Wikipedia not used in calibration, German chat, English chat, code).
   **Bar:** KL DE below 0.060 (the official FP8) is the target, below 0.069 (the community GPTQ) the
   minimum.
7. **Publish** as `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` with the measured table.

### 3.1 The recipe, exactly (the operator's ruling, 2026-10-05)

This is the checklist 20b executes. Nothing here is left to the executor's taste.

| item | setting | why |
|---|---|---|
| source | `Aleph-Alpha/Kolibri-1-BF16` (bf16, 156 GB) | never the FP8 release: no double rounding |
| model class | 20a's transformers-compatible `Kolibri1ForCausalLM`, checked against the vLLM plugin (KL0) | AutoRound loads the model through transformers |
| tool | AutoRound, the version pinned in `tools/quantize_qwen38_tuned.sh` (v0.14.2), upgraded only if it cannot handle 384 per-expert linears, and then re-checked for the shard-writer bug that script documents | the export format our loader reads |
| scheme | `--scheme W4A16 --group_size 64` symmetric | the engine's int4 g64 kernels; `w = (q - 8) * scale`, `qzeros` 0x77777777 |
| format | `--format auto_round:auto_gptq` | what `loader::QuantConfig::parse` accepts (docs/13) |
| quantised | every routed expert's `gate_proj` / `up_proj` / `down_proj`; attention `q/k/v/o_proj` only if decision 2 says int4 | the bytes |
| left in bf16 | `mlp.gate` (router), `moe.router.expert_bias`, `shared_experts.*`, all norms, `embed_tokens`, `lm_head` (no `--quant_lm_head`) | every token reads them; routing must not move; the engine builds the int8 head at load (spec 9) |
| calibration mix (tokens) | ≥ 50 % German (German chat with reasoning traces and tool calls, German long documents), ~30 % English chat, ~20 % code; all chat-formatted through the model's own chat template | the use case; German loses most (§3 table) |
| calibration source | preferably self-generated: the bf16 model's answers (reasoning mode on) to German and English prompts; plus held-back-free German text (news, Wikipedia) for documents | matches the model's own output distribution |
| calibration size | `--nsamples` 1024-2048 at `--seqlen` 2048 (2-4M tokens) | 384 experts x top-6 need coverage |
| expert coverage | log per-layer per-expert routed-token counts on the calibration set before tuning; any expert under 2048 tokens gets targeted German text added until it clears the floor | German-specialised experts are the ones an English-heavy mix starves |
| tuning | first `--iters 0` (RTN) as the baseline, then the default sign-SGD tuning (`--iters 200`), `--low_gpu_mem_usage` | RTN shows how much tuning buys |
| held-out evaluation | KL and top-1 vs bf16, **German and English separately**: German news / Wikipedia not used in calibration, German chat, English chat, code; plus perplexity per set | the published tables measure exactly this |
| bars | KL DE < 0.060 (the official FP8) target; < 0.069 (the community GPTQ) minimum; English not worse than KL 0.013 | §3's published numbers |
| output | `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ`, model card with the KL / top-1 table, calibration recipe and per-expert coverage summary | reproducibility |

**(decide) Where to quantise.** It needs ~160 GB of disk for the source and enough RAM to hold a
layer's 384 experts in fp32 during tuning (~6 GB) plus activations. Options: the box (CPU, or the
B70 as an AutoRound XPU device), or a rented GPU for a day. 20a measures one layer's tuning time to
decide.

## 4. The engine (stages 20c-20e)

**Reused:** spec 16's pipeline split across two cards; spec 15c/15d's MoE blocks (routing on the
device, grouped expert GEMMs in prefill); spec 18's patterns for a non-Qwen model (own table, loader
and capture, as K2); spec 9's int8 head; spec 6 / 10 attention kernels.

**New:**
- **Router for 384 experts:** `moe.cl`'s route is one lane per expert with a power-of-two limit of
  256. Kolibri needs a 512-lane variant with 128 padded lanes that are never selected (as K2's 100
  on 128), selection on `logit + bias` in fp32, ties to the lower id, weights `sigmoid(logit)` with
  **no** renormalisation, and the combine in a fixed, documented order (FusedMoE's order is
  unspecified; the reference picks ascending slot or id and the kernel matches it).
- **Ungated shared expert,** added after the routed sum.
- **Sandwich norms:** `post_attn_norm` and `post_ffn_norm` change the fold sequence (two more norms
  per layer, before each residual add).
- **Attention:** q/k RMSNorm; RoPE only in sliding layers; NoPE in full layers; GQA 12. **A
  513-position ring KV for the 40 sliding layers** (decode attention over the ring; prefill flash
  with a window mask), the growing KV only for the 10 full layers.
- **`head_dtype: float32`:** what it means for the logits is pinned in 20a (the weights are bf16);
  the int8 head computes in fp32 either way.
- **Tokenizer and chat:** the 128k tokenizer, ChatML template, `<think>` reasoning, and `<tool_call>`
  calls. Check whether their content is the JSON form (hermes-style) or Qwen XML, and reuse the
  matching parser.

**Single-card development mode:** load the first N layers on one card and compare a partial
forward with the reference, so kernels and capture can be tested before spec 16 lands.

## 5. Gates

- **KL0 (20a):** the reference port equals the vLLM plugin's semantics on tiny random weights
  (operator-level tests, as 18a), and the real bf16 model's perplexity on German and English text
  is sane (a wrong port shows up as an order-of-magnitude perplexity).
- **KL1 (20b):** the quantisation table of §3, German and English separately.
- **KL2:** greedy decode equals the reference's argmax at every determined position on the golden
  prompts (German and English), tie-aware; the routing diagnostic with a near-tie tolerance, as
  K2's.
- **KL3:** replay and chunked prefill bitwise.
- **KL4:** a German tool-call set (an A4-style scenario set in German) and passkey at 262144.
- **Speed:** decode and prefill rows on two cards against the derived roofline.

## 6. Stages

- **20a. Reference and model class (no card):** `tools/oracle/kolibri_ref.py` (layer-streamed, the
  repo's oracle pattern) plus a transformers-compatible `modeling_kolibri1.py` for AutoRound; facts
  sheet (`head_dtype`, the chat and tool-call format, EOS behaviour); KL0. The tokenizer and template
  are checked against HF's `tokenizer.json` / `apply_chat_template`.
- **20b. Quantisation (no card, or the box CPU/XPU):** calibration set build (German-heavy,
  self-generated), per-expert coverage report, RTN then AutoRound arms, KL1 table, publish.
- **20c. Decode on two cards:** model table, loader (384 + shared experts as contiguous blocks,
  router and bias), kernels (512-lane route, sandwich norms, NoPE / RoPE attention with the sliding
  ring), capture per pipeline stage; KL2, KL3 on decode. Needs spec 16 (decode across two cards).
- **20d. Prefill:** grouped MoE (15d's path with Kolibri's router and combine), windowed flash
  attention; KL2, KL3 on prefill.
- **20e. Serving:** tokenizer, template, reasoning and tool-call parsing, prefix cache (only the full
  layers' KV is large; the ring is part of the snapshot), `--max-len auto`; KL4; speed rows.

## 7. Decisions

1. **(decide)** Where to run 20b: the box or a rented GPU (§3).
2. **(decide)** Attention int4 vs bf16: decided by 20b's KL table (2.5 GB more per token in bf16).
3. **(decide)** Context target: 262144 (KV 5.4 GB) or more.
4. **(decide)** Whether the shared experts stay bf16 (proposed) or go int8.

## 8. Out of scope

- Loading the community `Padakovec/Kolibri-1-W4A16-GPTQ` (asymmetric zero points in compressed-tensors
  format): see §9.
- One-card operation (it does not fit at 4 bits).
- The FP8 release.

## 9. Note: asymmetric g128 checkpoints from llm-compressor

**Recommendation: do not build an asymmetric kernel path; read symmetric compressed-tensors.**
- **Cost of asymmetric support:** every int4 path (decode GEMV, prefill dequant slab, h8 requant,
  the MoE kernels, the compact heads) gains a zero-point variant: a second kernel family to tune and
  gate per shape, the same ongoing cost as the native g128 kernel recorded in docs/13 ("g128
  checkpoints"), for checkpoints we would not choose to serve.
- **No exact cheap conversion exists:** asymmetric int4 g128 is exactly representable as symmetric
  **int8** g128 (q - z lies in [-15, 15]), but that doubles the bytes, and Kolibri's experts would no
  longer fit two cards. Re-rounding to symmetric int4 adds a second rounding error, the kind of
  shortcut this project rejected.
- **Not built either: a compressed-tensors symmetric loader.** It would be exact and cheap, but
  **operator ruling (2026-10-05): the engine supports relevant models in one format, re-quantised by
  us for the best quality on the B70 (AutoRound int4 g64 symmetric), not every published quant.**
  Third-party formats are refused by name; a model worth serving gets its own recipe and script
  (`tools/quantize_*.sh`), as Kolibri does here.
- **For Kolibri:** our own AutoRound g64 symmetric quantisation (§3) is better on German than either
  route, and it is the format every kernel already serves.
