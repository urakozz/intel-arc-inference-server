# Spec 20 - Kolibri-1 (Aleph Alpha), German and English, on two B70s

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**. 20a built
2026-10-06 (§10).
20c built blind 2026-10-06 (§11, branch `spec20c-kolibri-decode`; box queue row 24). 20d built blind 2026-10-06 (§12, branch `spec20d-kolibri-prefill`; box queue row 26). 20e built blind 2026-10-06 (§13, branch `spec20e-kolibri-serving`; box queue row 28). Plans 20c-20e written 2026-10-06: `docs/superpowers/plans/2026-10-06-spec20c-kolibri-decode.md`, `...-spec20d-kolibri-prefill.md`, `...-spec20e-kolibri-serving.md` (two cards on spec 16b's merged pieces, `--pp 2` Kolibri's default).

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
| tool | AutoRound, intel/auto-round main at commit `6afaecdbe4092dc803f3e91026fe81a65b64b372` (0.17.0) - the operator's ruling of 2026-10-05, replacing the v0.14.2 pin of `tools/quantize_qwen38_tuned.sh` (both Qwen3.8 scripts moved to this commit on 2026-10-06, with `auto-round-local.patch`; Kolibri passes pre-tokenised rows, so the patch's dataset fix does not apply - `tools/quantize/README.md`, "AutoRound version"); recorded in the script (`AR_COMMIT`), `kolibri_recipe.json` and the card; re-checked on a tiny Kolibri: complete per-expert `qweight` / `scales` / `qzeros` in the `auto_round:auto_gptq` form, accepted by `check` (the 0.14.2-era lm_head shard-writer bug cannot apply: lm_head stays bf16) | the export format our loader reads |
| scheme | `--scheme W4A16 --group_size 64` symmetric (AutoRound's default; 0.17.0 has `--asym`, no `--sym`) | the engine's int4 g64 kernels; `w = (q - 8) * scale`, `qzeros` 0x77777777 |
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
- **Built: a loader for compressed-tensors symmetric `pack-quantized` checkpoints** (operator
  ruling, 2026-10-05: exact, loader-only conversions are welcome; new kernel families are not).
  `weight_packed` int32 [N][K/8] holds the same nibbles as GPTQ's `qweight` (q + 8, low nibble first)
  transposed, `weight_scale` [N][K/g] is the scales transposed: a repack into the GPTQ layout plus
  the existing g128 → g64 scale expansion, no kernel change. That covers the common symmetric W4A16
  releases from llm-compressor. Asymmetric checkpoints, activation ordering (`weight_g_idx` that is
  not the identity) and any group other than 64 / 128 are refused by name.
- **For Kolibri:** our own AutoRound g64 symmetric quantisation (§3) is better on German than either
  route, and it is the format every kernel already serves.
- **As built (2026-10-05, branch `ct-sym-loader`).** One conversion point:
  `loader::LinearSrc::classify` finds `.weight_packed` and builds `qweight[r][n] =
  weight_packed[n][r]` (the int32 words verbatim) and `scales[g][n] = f16(weight_scale[n][g])`.
  A bf16 scale is converted only when f16 holds it exactly, otherwise the load is refused
  naming the tensor and index. Then the existing g128 → g64 expansion runs. Dense linears,
  Agnes's fold, the MoE experts (main layers and the MTP head's MoE layer) and K2 all go
  through it; K2's name check takes either packing, one per checkpoint. Host tests:
  `ct_loader_test` (every weight equals (u − 8) × s and q × s, byte-equal to the GPTQ form,
  g64 / g128 × f16 / bf16; the refusals; the real configs in `tests/loader/ct/`), and
  `ornith_repack_test` / `k2_repack_test` (a compressed-tensors MoE layer and K2 checkpoint
  repack to the GPTQ form's bytes). A load prints one note naming the group, the linear
  count and the recommended format (operator requirement). Refused by name: `symmetric:
  false`, `.weight_zero_point`, a non-identity `weight_g_idx`, actorder "group" without g_idx,
  group ≠ 64 / 128, bits ≠ 4, a format other than pack-quantized, quantised activations,
  transforms. Real repos: RedHatAI/Qwen3.8-27B-INT4 (this engine's architecture, sym g128,
  bf16 scales) and RedHatAI/Qwen3-32B-quantized.w4a16 are accepted;
  Padakovec/Kolibri-1-W4A16-GPTQ and RedHatAI/Qwen3-8B-quantized.w4a16 (asymmetric) and
  halt95/Qwen3.8-Flash-Next-W4A16-Merlin (an int8 group) are refused. Box validation: queue
  row 18.

## 10. 20a as built (2026-10-06)

Plan: `docs/superpowers/plans/2026-10-05-spec20a-kolibri-reference-and-quant.md`. Facts:
`docs/probe-kolibri-2026-10-05.md`. No card, no weights: small files and range-fetched headers.

- **The model class:** `tools/oracle/third_party/kolibri1/{modeling,configuration}_kolibri1.py`,
  `Kolibri1ForCausalLM` for transformers 5.x (Apache-2.0 header naming the plugin), names = the
  checkpoint's (58353), per-expert `nn.Linear`, the router not an `nn.Linear`, masks built per layer
  (AutoRound replays block 0's kwargs - a sliding layer - into every block), KV-cached `generate()`.
- **The reference:** `tools/oracle/kolibri_ref.py` (bf16 / int4 by name, `run` / `ppl` / `hfcheck` /
  `facts`, layer-major `run_batch`). **KL0 (tiny weights): passed** - `test_kolibri_ref.py`, 21
  tests, one per trap of §1, and the port == the reference **bitwise** in bf16 eager (prompt and
  cached decode, window 5 and 513) and in fp32. KL0's real-model perplexity is pending (weights).
- **Pinned by 20a** (§4's open points): `head_dtype: float32` = vLLM's fp32 lm_head (bf16 operands,
  fp32 accumulation and logits); the window is 513 keys **including** the query (i - 513 < j <= i);
  tool calls are **hermes JSON** inside `<tool_call>` (the plugin registers `Hermes2ProToolParser`) -
  20e reuses the JSON parser; EOS {127906, 127901}; no BOS; the router's logits are fp32, never
  bf16-rounded; ties to the lower id; the combine is ascending-id fp32 + shared, one rounding.
- **The quantisation (20b's tool):** `tools/quantize_kolibri1.sh` + `tools/quantize/kolibri/`, §3.1
  stage by stage (calib, coverage, rtn, tune, eval, check, card), resumable, `--dry-run`; ran end
  to end on a tiny random Kolibri on the Mac (`tools/quantize/README.md`). Additions to §3.1 found
  on the way: AutoRound's iters 0 is "optimised RTN" unless `disable_opt_rtn` - the baseline uses
  plain RTN; on a CPU without bf16 AutoRound writes the unquantised tensors in fp32 - the script
  restores the source's bf16 bytes after a bitwise check; the held-out German documents are a
  different Wikipedia shard (00019) from the calibration ones (00000), revision pinned.
- **Open:** decision 1 (where 20b runs; the script prints the first block's tuning time - x 50 is
  the estimate), decision 2 (from 20b's table).


## 11. 20c as built blind (2026-10-06)

Plan: `docs/superpowers/plans/2026-10-06-spec20c-kolibri-decode.md`, branch `spec20c-kolibri-decode`.
Written on the Mac: the box was unavailable and the real int4 checkpoint does not exist yet (decision
1, 20b), so nothing here has run on a card, and every real-weight gate SKIPs (77). A synthetic
real-width checkpoint carries the Mac gates and, on the box, KL2 / KL3 before 20b. Box validation:
queue row 24. Every number below is **derived** unless marked measured.

**Structure** (K2's pattern, spec 18 §5.1): `model::Kolibri1Desc` (`src/model/kolibri1.*`),
`loader::load_kolibri1` (`src/loader/kolibri1_*`), `runtime::kolibri::KolibriEngine`
(`src/runtime/kolibri/`), Kolibri-only kernels `src/kernels/kolibri/kol_*.cl` (host half
`src/kernels/kolibri_kernels.h`), `b70-decode`'s dispatch on `model_type kolibri1`
(`src/cli/kolibri_decode.h`). Reused at Kolibri's shapes from new CMake lines only: `gemv.cl` (the
int4 attention arm), `gemv_bf16.cl` (router, bf16 attention arm, bf16 head), `gemv_i8w.cl` (int8 head),
`prep.cl`'s `prep_res_fold` (SP0, and with `ZERO_RESID` at SP1 / SP4), `embed_gather.cl`, `argmax.cl`.
Kernel command lines +23 / -0 / ~0 (K2 on; 25 without - `gemv_M1_K6144_N2560_S4_L0` and
`prep_res_fold_M1_K2560_SP0_G20` are K2's binaries, built here only without K2). No existing `.cl` edited.

**The layer list (15 launches; 17 under eager):** `kol_norm_finish` (input_layernorm) · q||k||v GEMV
(int4 S2 L0 / bf16) · `kol_attn_prep` · `kol_attn_decode` + `kol_attn_reduce` (eager: score,
softmax, P·V, reduce) · o_proj GEMV (int4 S4 / bf16) · `prep_res_fold ..._Z` (o_proj's row `a`, Σa²)
· `kol_post_add` (post_attn_norm, resid +=, Σresid²) · `kol_norm_finish` (post_attention_layernorm)
· router `gemv_bf16` (fp32 logits) · `kol_route` · `kol_moe_gate_up` · `kol_moe_down` (into `mo`,
not the residual) · `prep_res_fold SP0` (Σmo²) · `kol_post_add` (post_ffn_norm). Plus embed + fold on
device 0 and four at the head: **756 launches at 50 layers, 856 eager**; on two cards the same sum
(the cut is 16b's: device 0 ends with layer s-1's `kol_post_add`, which already wrote `resid` and the
20 sums layer s's norm reads), + pp_send / pp_recv under `peer` (758 / 858).

**The sandwich** (Review Focus 1) is two kernels around an unchanged stage A: `prep_res_fold _Z`
writes the sub-block's own bf16 row and its Σ² (MoE: `kol_moe_down` writes `mo`, `SP0` re-reads
it), `kol_post_add` normalises it (x̂ rounded, then × the plain post-norm w), adds it to the residual
with one rounding and reduces the new residual's per-chunk Σ² in prep_res_fold's tree - which is
what the next `kol_norm_finish` (x̂ rounded, then × w: two roundings, the reference's) reads.

**The ring: 4096 slots per sliding layer, not 513** (335.5 MB over 40 layers, against §2's ~42 MB for a
513-slot ring): a power of two ≥ a prefill chunk (2048, 20d) + window - 1, so a chunk's own keys and
the window before it never collide, and key p's row is `p & 4095`. Flash attention walks a sliding
layer in ABSOLUTE 64-key blocks from `lo & ~63`: a block, and every 16-key wave in it, is contiguous
ring rows (4096 is whole 64-row blocks), so no 2D block read crosses the ring's end; a block wholly
below the row's window is skipped by the reduce. Full (NoPE) layers keep the growing cache, 20 KiB a
position for 10 layers.

**The router: §4's "512-lane variant" is 512 SLOTS on 256 lanes** (two experts a lane, the Mac's
256 work-item cap); the router GEMV's 128 padded rows are zero and their slots are never ranked
(sel = -INF). `kol_route`: sel = fp32 logit + fp32 bias, rank with ties to the lower id, ids
ascending, weights sigmoid(logit) in fp32, never rounded, not renormalised. The combine
(`kol_moe_down`): ascending id, `acc = acc + f32(rne(y_j)) × w_j` with the product rounded before
the add (`FP_CONTRACT OFF`: the reference's `index_add_` of `y.float() * w`), + the bf16 shared
expert, one rounding. The shared expert is the MoE block's seventh slot, read from bf16
`gemv_bf16` tiles (decision 4 stays open: bf16 proposed).

**Attention:** q / k RMSNorm per head (128-lane tree), RoPE only in sliding layers
(`rne(rne(y·cos) + rne(rot(y)·sin))`, the RoPE table bitwise torch's at 0, 1, 513, 100000, 262143 -
measured on the Mac, `kolibri1_rope_test`), full layers position-free. Flash (`kol_attn.cl`,
`k2_attn.cl`'s v2 structure at GQA 12, no gate) is the default; `B70_KOLIBRI_ATTN=eager`
(`kol_attn_eager.cl`, `k2_attn_eager.cl` without the gate) is the reference's bf16 chain, for the
box's A/B (spec 18 §10.1's rule).

**Two cards** (16b's pieces, reused): placement at load (`load_kolibri1` puts each layer straight onto
its device; the model is never whole on device 0), the split by bytes = `runtime::pp_balance` over
Kolibri's per-layer bytes at max_len (weights + 8 MiB ring or max_len × 4 KiB of full KV),
`PipelineLink` from the descriptor-free `pp_landing_layout(5120, 80)` (two overloads added to 16b's
files, the `ModelDesc` forms delegating byte for byte), the `copy` / `peer` hand-off, the Control
mirror and the bounded fence. Real model, int4 attention, int8 head, 262144: **split 25**, device 0
24.413 GB against device 1 24.086 GB (split 24: 23.045 / 25.454; 26: 25.252 / 23.247) - the plan
estimated ~24.11 / ~23.78. `--max-len auto` on two cards: 262144 (capped by decision 3). One card
cannot hold the model (~42.5 GB of int4 weights): `b70-decode` defaults to `--pp 2` and refuses `--pp 1`
naming the bytes unless the planner says it fits (a synthetic checkpoint, `--layers N` -
development mode, e.g. `--layers 30` ~26 GB).

**Known deviations** (ulp-level, documented, not hidden): flash keeps fp32 probabilities against the
reference's eager bf16; eager's 8-lane softmax indexes a sliding row from its first visible key (the
cached decode pass's row), which differs from the reference's PROMPT pass for prompt rows past
position 512; sigmoid's OpenCL `exp` against torch's (≤ 2 ulp in the fp32 weight; 1 ulp measured on
the Mac GPU); the reference's prompt rows are a full-sequence forward, the engine ingests one replay
per id.

**Validated on the Mac (measured):** the host tests (`kolibri1_test`, `kolibri1_repack_test` - every
expert block of two real-width layers word by word, both attention arms, refusals by name -,
`kolibri1_rope_test`, `kolibri_ref_test` - the kernels' twin BITWISE against `kolibri_ref.py`'s own
outputs (`tests/kernels/kolibri_fixture.h`): norms, q/k norm + RoPE, the route incl. a tie at the
cut, the combine, eager attention scores / probabilities / outputs at 5, 512, 513, 700 sliding and 700
full -, `kolibri_variant_names_test`, `kolibri_plan_test`, `pipeline_plan_test`'s overload);
`kolibri_run` on the Mac's GPU (indicative): every portable kernel exact against the twin, and the
eager attention exact against torch's fixture; Level Zero and OpenCL syntax of every new source and
variant; a 1-layer real-width synthetic checkpoint (2.15 GB) `ACCEPTED` by `check.py`.

**Deviations from the plan:** `KolPlacement` carries the layer count (`{devices, split, layers}`:
the two-field form cannot say where device 1 ends); `KolLayer::qkv` / `oproj` are
`unique_ptr<DeviceWeight>` (`l0::Mem` has no default constructor); the runtime, not Task 5, already
holds the two-card half (Task 6 lifted the CLI gate and added its gates); the bench prompt
(`tests/golden/prompts/kolibri_bench.ids`, `kKolibriBenchPrompt`) is PLACEHOLDER legal ids - no
Kolibri `tokenizer.json` on the Mac and nothing downloaded - until `kolibri_oracle.sh synth` prints
de_prose's first 42; the box queue row is 24 (23 is 16c's); the fixture compares coarse inputs
bitwise and generic ones within an ulp (the Σx² order is torch's own), and eager outputs against the
fp64 form at cosine ≥ 0.999 on the host (flash's 0.99999 bar is the card test's); there is no
existing b70-serve refusal test to copy, so `cli_reject_kolibri_serve` is its own `sh -c` block.

**What the box must prove (row 24):** K0 (every pre-existing binary's sha256, the suites); the 23
binaries compile under ocloc; K1 (`kolibri_kernels_test`: `kol_attn.cl`'s sub-group / 2D path over the
ring, the 224-lane `kol_moe_down`, the 256-lane route); the loader on the synthetic checkpoints; K3 and
KL2 on both synthetic arms (flash and eager); `--pp 2` bitwise `--pp 1` under both hand-offs; the CLI.
After 20b: the partial forward (30 layers, one card), KL2 / K3 on two cards, the speed rows.


## 12. 20d as built blind (2026-10-06)

Plan: `docs/superpowers/plans/2026-10-06-spec20d-kolibri-prefill.md`, branch `spec20d-kolibri-prefill`.
Written on the Mac like 20c: no card, no real checkpoint, so nothing here has run on a B70 and every
real-weight gate SKIPs (77); the synthetic checkpoints carry KL2 / KL3 on the box before 20b. Box
validation: queue row 26. Every number is **derived** unless marked measured.

**Structure** (K2's 18c arrangement): the walk `runtime/kolibri/kolibri_prefill.{h,cc}` (one device's
layers per chunk, no host wait inside), `KolibriEngine::prefill` / `prepare_prefill` / `read_prefill_routes`
/ `set_prefill_replay` / `set_block_hook` in `kolibri_prefill_engine.cc` - its own archive
`b70_kolibri_prefill`, so a decode-only target links what it did; the prefill state is lazy. Kolibri-only
kernels `kol_pf_moe.cl`, `kol_pf_attn.cl`, `kol_pf_linear.cl` (host half: `kolibri_kernels.h`'s spec 20d
block); reused at Kolibri's shapes from new CMake lines only: 20c's `kol_prep.cl` at M = 2048, `pf_embed`,
`pf_prep`'s `pf_res_fold` (SP0, `_Z`), `pf_gemv_bf16` (the router at decode's {16, 16}), `pf_moe_gemm`
(2560 x 1024 SiLU, 512 x 2560), `k2_pf_linear.cl` (the int4 arm's layout-0 slabs), `pf_gemm_T0`; the route
is 20c's decode binary on grid (1, C). Kernel command lines **+17 / -0 / ~0** with K2 on (measured,
`kernel_cmdlines`; `pf_res_fold_K2560_SP0_G20` and `k2_pf_dequant_slab_K6144_N2560` are K2's names, built
here only without K2). No existing `.cl` edited; 20c's decode list (756 / 856) unchanged.

**The chunk, per layer (45 launches):** `kol_norm_finish` (M 2048) · q||k||v as 7 slabs of 1024 (int4:
`k2_pf_dequant_slab`; bf16: `kol_pf_bf16_slab`) each + `pf_gemm_T0` · `kol_attn_prep` (M 2048, S 1: q/k head
norm, RoPE in sliding layers, every row's k / v into the ring slot `p & 4095` or the full cache row `p`) ·
`kol_pf_flash_attn` · o_proj as 3 slabs (1024, 1024, 512) x 2 · `pf_res_fold _Z` + `kol_post_add` (the
sandwich, decode's chain) · `kol_norm_finish` · the router (`pf_ab_proj`, fp32 logits) · `kol_route` ·
`kol_pf_sort` · `kol_pf_gather` · gate||up: 4 weight batches x (`kol_pf_dequant_gu` + `pf_moe_gemm` SiLU) ·
down: 2 x (`kol_pf_dequant_dn` + `pf_moe_gemm`) · `kol_pf_moe_combine` (into `mo`) · `pf_res_fold SP0` +
`kol_post_add`. Device 0 adds `pf_embed_gather` + `pf_res_fold SP0`: **2252 launches per chunk at 50
layers, on one card and on two** (the cut is 16b's: device 0 ends with layer s-1's `kol_post_add`, device 1
starts at layer s's norm), + 5 for the head (decode's binaries over the last row on the last device).

**The rounding chains** are decode's wherever a decode twin exists (norms, sandwich, prep, route, the
combine's ascending-id fp32 + shared, one rounding, `FP_CONTRACT OFF`); what differs is where the GEMM sums
form (DPAS over bf16-dequantised weights) and the softmax walk (64-key flash tiles) - so prefill's KV,
rings and routes are decode's to a cosine bar and near-ties (Review Focus 4's proposed bars: layer 0
every row >= 0.999, median >= 0.9998, p01 >= 0.99), not bitwise.

**Decisions taken blind:**
- **`l0` only.** `--prefill-backend l0-int8` refused (its h8 linears rotate in whole 1024-k Hadamard blocks;
  hidden is 2560), `sycl-tla` refused (no Kolibri walk); `--prefill-chunk` above 2048 refused.
- **The separate dequant pass in 512 MiB batches:** 385 blocks a layer (384 experts + the bf16 shared
  expert as block 384, COPIED from its gemv_bf16 tiles, not dequantised) of 5,242,880 B (gate||up, 102 a
  batch, 4 batches) and 2,621,440 B (down, 204, 2 batches); an expert with no row this chunk is skipped.
- **The 384-expert sort** stages ids as `ushort` (k2_pf_sort's bytes stop at 255) with two experts a lane on
  256 lanes; tmax(2048) = 820 tiles (`floor((C x 6 + 384 x 31) / 32) + ceil(C / 32)`), reachable exactly
  (372 experts with 33 rows, 12 with 1 - tested).
- **The window over the ring:** row t of a chunk at c0 sees keys `(c0 + t - 513, c0 + t]`; an 8-row group's
  key tiles start at the ABSOLUTE multiple of 64 below its first row's window and every tile is 64
  contiguous ring rows (4096 % 64 = 0), so no 2D read crosses the ring's end. A chunk of 2048 needs keys
  `[c0 - 512, c0 + 2047]`: 2560 < 4096, the reason for 20c's 4096 slots. A row's first tile may hold none
  of ITS keys (the group's first row reaches further): the online softmax uses `m_safe = (m == -INF ? 0 :
  m)`, so such a tile leaves (m, l, o) exactly as they were - a row's result is a function of its absolute
  position, which makes a split at a multiple of 64 bitwise.
- **Two cards, sequential:** each chunk runs layers [0, s) on device 0, crosses by spec 16b's `copy`
  hand-off at chunk size - a SECOND `PipelineLink` over `pp_landing_layout(2048 x 2560 x 2, 20 x 2048 x 4)`
  (10.5 MB + 160 KB), two copies and the cross-device event - then layers [s, L) on device 1, the host
  waiting on device 1 with `PipelineOptions::prefill_timeout_ms`. The chunk always crosses by copy:
  `pp_handoff.cl`'s peer kernels are one 256-lane work-group sized for a 5 KB row. A lost hand-off
  host-signals the event, throws naming device 1 and marks the engine until `reset()`
  (`drop_next_handoff()` drops device 0's half of the next chunk - P4). Spec 16c's overlapped chunk
  pipeline is NOT reused: it is built around `PipelineEngine`'s `PrefillScratch` / `Int8State` and
  Qwen's walk (`step_stage`), and taking it for `KolibriEngine` means a second, Kolibri-shaped slot ring
  and per-slot events - not trivial; it stays Task 5's recorded lever (up to ~2x on two cards).
- **Replay** (`B70_PREFILL_REPLAY`): each device's chunk walk is recorded once per (pos, rows); the
  hand-off's copies / event stay on the immediate lists (a recording cannot hold them).
- **The prefill scratch: 904,632,800 B (0.905 GB) per device** (the 512 MiB batch its largest term), +
  the prefill link on two cards; planned only when a run prefills: `--max-len auto` still reaches 262144
  on two cards (split 25: device 0 25.318 GB, device 1 25.001 GB with the int8 head and the 1.5 GB
  reserve).
- **Precision:** flash keeps fp32 probabilities (the engine's convention); `B70_KOLIBRI_ATTN=eager` covers
  prefill too - the `_EAGER` flash builds (k2_pf_attn's two passes: the reference's rounding points, not its
  sum order), one parser `runtime::kolibri::kolibri_attn()`, read once at engine construction.

**Validated on the Mac (measured):** host tests `kolibri_pf_ref_test` (the sort on random / tied / adversarial
routes == `pf_moe_ref::sort`; the combine == 20c's decode chain on 300 tokens x 2560 and == torch's fixture
row bitwise; reversed chunks and in-tile permutations bitwise; the flash walk in fp64 == the direct softmax
within 3.4e-16 at c0 0 / 1000 / 1001 / 4396 through the ring, two rows starting on a wholly masked tile;
`eager_window` == torch's eager rows **bitwise** - scores, probabilities and outputs - at 5, 512, 513, 700
sliding and 700 full), `kolibri_pf_variant_names_test`, `kolibri_plan_test` (2252 / 2252, batches 4 + 2,
the scratch term by term, 262144 with prefill planned); `kolibri_run` on the Mac's GPU (indicative): the
sort at C = 2048 on four route sets, gather, dequant (routed, skipped, the shared block 384), combine, both
slab kernels, `pf_res_fold` + `kol_prep.cl` at M = 2048 - exact, except 11 / 16 of 5.2M norm / post-add
values that move with Apple's 1/sqrt (each the host chain at an rstd within 4 fp32 ulps); Level Zero syntax
of every new / changed source; OpenCL syntax of every variant; `kernel_cmdlines` +17 / -0 / ~0.

**Deviations from the plan:** `kol_pf_bf16_slab` takes the slab width `ns` as an argument (k2_pf_dequant_slab's
contract); `pf_res_fold_K2560_SP1_G20` is not built for Kolibri (nothing binds it - the names test would
call it dead); `PrefillSizes` has the walk's own fields (`ids, resid, x, a, mo, partials, slab, sumsq_a,
sumsq_r, attn_q, attn_out, logits, routes, hdr, tiles, row_tok, pair_row, xg, h, w`); the head computes on
the prefill row in place (K2's) rather than copying it into the decode `resid`; `prefill_split_kolibri_test`
is `kolibri_prefill_test`'s `split` mode (K2's arrangement), and its `pp` mode is the one-card / two-card
comparison; the plan's "all to 6 experts" adversary is the sort's per-lane serial worst case, not the
tile bound's (448 tiles) - a separate adversary reaches 820; `eager_window` matches torch bitwise
(stronger than the plan asked: P·V as one ascending fp32 chain).

**What the box must prove (row 26):** K0 (G0, 20c's decode gates); the 17 binaries under ocloc
(the DPAS flash at GQA 12 with the window, `kol_pf_sort`'s 28 KB of SLM, the grouped GEMMs at Kolibri's
shapes); K1 (`kolibri_pf_kernels_test`: grouped == dense bitwise, flash >= 0.99999 against fp64 over the
ring and to depth 62048, a split at 64 bitwise); K3 on prefill on both synthetic arms (2 + 5 x 45 launches
+ 5, replay and chunks of 16 bitwise, splits at 64 / 2048 bitwise); KL2 on prefill (flash, eager, chunks of
16, the int8 head); two cards bitwise one card, P4 on the chunk hand-off; the CLI. After 20b: K3 / KL2 on the
real checkpoint across two cards, and Task 5's speed rows (derived pp4096 ~4,000 t/s; the dequant pass
~3 GB of bf16 per layer per chunk).


## 13. 20e as built blind (2026-10-06)

Plan: `docs/superpowers/plans/2026-10-06-spec20e-kolibri-serving.md`, branch `spec20e-kolibri-serving`.
Written on the Mac like 20c / 20d: no card, no real checkpoint, so nothing here has run on a B70 and every
real-weight gate SKIPs (77); the synthetic checkpoints carry the engine side on the box before 20b. The
tokenizer files (not the weights) of `Aleph-Alpha/Kolibri-1-BF16` at `8c8b3489` were in the Mac's HF cache,
so the tokenizer and template gates RAN. Box validation: queue row 28. Every number is **measured** on the
Mac where it says so, else **derived**.

**What `b70-serve <Kolibri-1>` does now.** It dispatches on `model_type kolibri1` before the Qwen family's
rules (`cli::kolibri::is_kolibri`, b70-decode's pattern) and runs `runtime::kolibri::KolibriEngine` behind
`cli::kolibri::KolibriEngineAdapterT` (`src/cli/kolibri_serve.h`): **two cards by default** (`--pp 1` only
when the planner says the model fits one card - a synthetic checkpoint), `--pipeline-split auto|N`,
`--pipeline-handoff copy|peer`, `--max-len auto|N` planned over both cards with the prefill scratch
(`cli::kolibri::settle`, b70-decode's; 262144 on two cards at split 25 with the int8 head, derived, 20d),
`--lm-head int8` by default (spec 9), prefill on `l0` (20d's walk, chunks ending at block ends when the
prefix cache hooks it), the prefix cache on by the usual flags, EOS [127906, 127901] and the sampling
defaults from `generation_config.json` (**sampled**, T 1.0, top-p 0.97, top-k 128 - verified against the
downloaded file), `ChatFormat::kolibri()`. Refused before any device, each by name: `--mtp` / `--spec mtp`
(no head), `--spec lookup` (no verify lists), `--kv-cache int8` (bf16 only), `--prefill-backend l0-int8` /
`sycl-tla`, a `--pp 1` that does not fit (naming the bytes), `--max-len` above 262144 (decision 3),
`--device` with `--pp 2`. 20c's b70-serve refusal and 16d's `--pp 2` refusal for Kolibri are lifted
(`cli/pipeline_serve.h` refuses nothing for `kolibri1`: its rules are `cli::kolibri::check_args`'s). No
kernel was added or changed (`kernel_cmdlines` +0 / -0 / ~0).

**Tokenizer and template (Task 1; measured on the Mac).** `template_kolibri_test`: 16 message lists (every
`reasoning_effort`, `enable_thinking` off, tools with and without a system turn, call history, parallel
calls and their tool results, reasoning before and after the last user query, the `reasoning` field,
`<think>` inside content, German) render **byte-identical** to transformers 5.15's `apply_chat_template` -
no minja patch was needed. `kolibri_tokenizer_test`: the Rust tokenizer equals HF's on 24 texts (German
prose, `„…“`, a 42-letter compound, code, digits, emoji, the tags inline, empty), corpus.txt's 10240 lines
(177,677 ids, 10 digests), three chat renders' ids (66 / 424 / 325), no BOS ever added, EOS
`<|im_end|>` / `<|endoftext|>`. transformers warns "incorrect regex pattern ... fix_mistral_regex" for this
tokenizer; the references are the `tokenizer.json` as written (raw `tokenizers` 0.22.2 and HF without the
fix agree, asserted per text) - what vLLM and the engine read.

**Reasoning and tool calls (Task 2).** `server::KolibriOutputStream` keys reasoning on the OUTPUT (optional
whitespace, then `<think>` - the model opens it; the prompt never ends in it), and reads hermes JSON calls in
`<tool_call>` through `server::parse_json_call` - K2's JSON body rule, moved out of `toolcall_k2.cc`'s
anonymous namespace into `toolcall.{h,cc}` (K2 unchanged: `toolcall_k2_test`, `k2_server_test` green). A
body that does not parse is content verbatim with its tags; a cut call is content; whitespace after a call
and a segment's whitespace before one are dropped, so a template render parses back to its message and a
client that sends a parsed turn back re-renders exactly the generated text (the prefix cache's best case).
`toolcall_kolibri_test` (measured): the 8 assistant turns of the HF renders (7 with reasoning, 5 calls)
parse back to their messages, every Review Focus 1 / 2 / 5 case, a 2000-output fuzz, every parse whole =
byte stream = random 1-7 byte pieces. `kolibri_server_test`: kwargs reach the template (a bad
`reasoning_effort` is a 400 naming it), list content joined, both EOS ids stop, streamed frames carry the
body's reasoning / content / call, the defaults 1.0 / 0.97 / 128 apply when a request names none and a
request's own fields win.

**The engine behind the server (Task 3).** Snapshots (Review Focus 4): Kolibri's "state" is the 40 sliding
rings' rows of `[pos - 512, pos)` - what the next query's window reads - **41,943,040 B** at every pos
(positions below 0: zero rows on the host, written back as zeros); the blocks are the 10 full layers' KV,
**20,480 B a position** (derived). `runtime::kolibri::state_runs` / `kv_runs` (device-free) give both host
layouts as `[K | V][layer, in layer order][positions][kv_n]` - addressed through the placement, so `--pp 1`
and `--pp 2` share them byte for byte (16b's rule); `KolibriEngine::save_state` / `load_state` / `save_kv`
/ `load_kv` copy them on the immediate lists (`load_state` resets a session whose hand-off failed first).
The adapter: host sampling at the tokenizer's id count; a greedy device argmax on a head row without a
token replaced by the masked row's argmax; the store keyed `KOLIBRI`. `kolibri_snapshot_test` (host,
measured): the layouts on the real shapes (80 runs, 160 across a wrap or below 0; inside the allocations;
one host layout at splits 3 and 25), save / load on host buffers, restores through `PrefixSession` at the
block ends 2048 / 4096 and the request ends 2049 / 5000 / 300 continuing exactly as a cold run, on one card
and two, a snapshot moving between them, and the server over HTTP (greedy = the decode run, a second
request restored at 2048, seeded sampling reproducible, no undefined id ever emitted). The card test is
`kolibri_snapshot_gpu_test` (row 28).

**KL4 tooling (Task 4).** `make_set.py --lang de` (the six templates over the same six files, the system
prompt and user turns in German - the English table renders the original strings, checked against main's
script), Kolibri rendered from key-sorted objects with thinking off; `tests/golden/toolcall-kolibri-de`:
36 scenarios, 886-2998 ids, made on the Mac from the release's tokenizer (20b's export copies it). `score.py
--format hermes` (KL4's hard bar - the reference's parse failures - printed); `oracle_generate.py --model
kolibri [--device]` (`KolibriRef`, bf16, greedy through its cache); `a4_ref.sh kolibri`;
`tools/probe/kolibri_passkey.sh` (262144 on two cards, the int8 head, 3 / 3).

**20c's leftover.** `tests/golden/prompts/kolibri_bench.ids` and `kKolibriBenchPrompt` are now the first 42
ids of `de_prose.txt` through Kolibri's tokenizer ("Als im Frühjahr 1871 die ersten Vermessungstrupps ...
vom Holz": 27498 2625 19919 32 49 56 55 49 ...), replacing 20c's placeholder legal ids.

**Deviations from the plan:**
- **The undefined ids are 127998 and 127999**, not 127923 / 127924: the tokenizer defines 0..127997
  contiguously (98 added tokens; 127923-127997 are `<|reserved-token-2..76|>`), the head has 128000 rows
  (measured; the probe sheet is corrected). So the sampler's existing `vocab_used` mask (the tokenizer's
  count) is the whole rule - no hole mask; the device argmax still ranks every row (no kernel change) and
  the adapter replaces an argmax on the last two rows.
- **No `chat_template.jinja` for Kolibri:** the release (and the synthetic checkpoints, which copy its
  files) keep the template as `tokenizer_config.json`'s `chat_template` string; `chat::Template` now reads
  that string when there is no `.jinja` (transformers' order otherwise). `tests/tokenizer/kolibri/` vendors
  `tokenizer_config.json` and `generation_config.json` only - 20a's `kolibri1_chat_template.jinja` carries a
  comment header, so it is not the checkpoint's bytes.
- **`list_content` is not a template case:** Kolibri's template concatenates content as a string (HF's own
  render of a list fails); the server joins a message's text parts with `"\n"` first (vLLM's rule for such
  templates, `ChatFormat::string_content`), tested in `kolibri_server_test`. Two cases were added
  (`reasoning_field`, `think_in_content`). `preserve_thinking` is checked as a bool.
- **Sampling:** a Qwen-format server keeps `Sampling{}` - greedy unless the request names a temperature (the
  plan's "1.0 / 0.95 / 20" are its fields). Kolibri's defaults come from `generation_config.json` through
  `server::generation_sampling` (`do_sample` true: sampled). Its adapter also samples the FIRST id after a
  prefill from the head's row when the request samples (the other adapters keep the device argmax there).
- **No change to `kolibri_prefill_engine.cc`:** 20d's block hook already ends chunks at block ends.
- **Test names:** the host test is `kolibri_snapshot_test` (layouts, a device-free engine, `PrefixSession`,
  the HTTP server - K2's `k2_serve_test` arrangement); the plan's card test is `kolibri_snapshot_gpu_test`.
- **A request-end restore** (2049, 300) is checked bitwise against the session that never left (the same
  chunks), not against ONE cold prefill: 20d's split rule makes a split off a multiple of 64 close, not
  bitwise. Block-end restores are bitwise the cold run.
- `a4_ref.sh kolibri` and the data key `oracle_kolibri_a4` were added (the reference runs where the bf16
  source is - decision 1 - and is pushed to the box). `serve_benchy.sh` needed nothing: 16d's
  `SERVE_AFFINITY` already serves on both cards.
- The records (BENCHMARKS, docs/03, docs/13, the speed rows) wait for the box: no number here is a
  measurement on a card.

**What the box must prove (row 28):** K0 (G0's sha256: no binary moved; the server path's host tests); the
host tests on the box; b70-serve's Kolibri refusals before the device; `kolibri_snapshot_gpu_test` on the
synthetic checkpoint - a restore at 4096 bitwise the cold run, at 2049 / 300 bitwise the uncached session,
one card / two cards and a snapshot crossing between them; b70-serve on the synthetic (`--pp 2` by default,
peer, `--pp 1`, a seeded request twice identical, the startup line); `golden_server_test --chat`: the
server's greedy chat = `b70-decode --prefill --lm-head int8` on its `prompt_token_ids`. After 20b: the
same on the real checkpoint (`--max-len auto` -> 262144), KL4 - passkey 3 / 3 at 262144 with the int8 head,
A4 against the bf16 source's reference with 0 parse failures (agreement recorded) - and the llama-benchy
rows (pp4096 tg256 copy / peer, prefix caching at 4k / 16k / 32k, decode at 4k / 32k / 128k).
