# Spec 25 - NVIDIA Nemotron 3.5 Bolt 74B-A9B (text only) in the engine

**Status:** design, 2026-10-20, approved by the operator 2026-10-20. Decisions are marked **Decision N** with their
cost if wrong and collected in §12. 25a is built on the Mac (§13: the reference, B1 on tiny models; no real
weight read yet); 25b is built on the Mac (§14: the descriptor, the loader, the plan; no card); 25q, 25c-25e are not built.
**Plans (2026-10-20):** `docs/superpowers/plans/2026-10-20-spec25{a,q,b,c,d,e}-*.md` - 25a facts and the CPU
reference, 25q our AutoRound g64 re-quantisation (the recipe; the operator runs it), 25b descriptor / loader /
memory plan, 25c decode (the launch-count and state-bandwidth P0 first), 25d prefill, 25e serving.
**Box queue rows 57-62** (25a, 25q, 25b, 25c, 25d, 25e in that order; rows 63-65 are spec 26's, LiLiCorr, written in
the same branch). Each plan's last task adds its row to `box-validation-queue.md` and its `row N` block to
`tools/box_validate/stages.sh` when it is built; this spec and its plans do not edit `stages.sh`.

**The operator's request (2026-10-20, verbatim):** "write specs to support LiLiCorr and Nemotron 3.5 Bolt 74B-A9B,
AutoRound int4 g128 sym, GPTQ version via our g64 kernel. I'll quant it later as g64 with coding dataset and more
iterations". LiLiCorr, the drafter, is spec 26.

**Model:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` (the original; 63 shards, 139.5 GiB published by
`SergiioB`'s card - the BF16 repo's listing was read 2026-10-20, its weights were not) and the int4 checkpoint the
engine serves first, **`SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym` @ `b9f381d2`** (9 files, 112,940
tensors, 41,915,552,784 B; AutoRound 0.17.0 + intel/auto-round#2512, g128 symmetric, GPTQ packing). Architecture
`NemotronH_Omni_Reasoning_V3` (`model_type: nemotron_h_omni`) wrapping a `NemotronHPuzzleForCausalLM`
(`llm_config.model_type: nemotron_h_puzzle`) and a RADIO vision tower. **Text only**: the vision tower, the projector and
the `<image>` / `<video>` paths are out of scope (§11). Licence: **OpenMDW-1.1** (`license: other`,
`license_name: openmdw-1.1`; §1.5).

**Why this model.** NVIDIA's card positions Bolt for "local agent harnesses and general agentic tasks involving tool
calling, web browsing, and terminal use" (published), post-trained with end-to-end GRPO on MCP and SWE environments
(published) - the operator's target workload. It is the largest model the engine would serve (74.04 B parameters, 9.05 B
active - derived, §3; the card says 74 B / 9 B), it ships a NVIDIA-trained block drafter (LiLiCorr, spec 26), it reads
**~5.5 GB a token** - one third of Qwen3.8-27B's 15.54 GB - and its KV cache is **8 KiB a position** (8 attention
layers x 2 KV heads), so a 262144-token context costs 2.15 GB. SergiioB's vLLM 0.31 XPU numbers on two B70 at TP 2
(published, his card, 2026-10-16): decode 60.5 t/s at 886 prompt tokens, 50.7 at 8k, 33.5 at 32k, 23.0 at 64k; prefill
2925 / 2014 / 871 / 486 t/s at the same depths. §4 derives what our lists should reach.

**Order:** after spec 24 (it reuses LFM2's family pattern, Kolibri's fp32 sigmoid router, Muse-Glimmer's GQA-16 NoPE
attention at head_dim 128, the GDN prefill's chunk-scan discipline, spec 16b-16d's pipeline across two cards, spec 19c's
per-row recurrent-state slots for verify). Spec 26 (LiLiCorr) needs 25c-25e.

Every number is **measured** unless marked **derived** (from the configs, the index and the shard headers - range-read
2026-10-20, a few MB; no weight shard was downloaded), **estimated**, **proposed** or **published**.

**Citations.**
- `R:` = the checkpoint's own modelling file `modeling_nemotron_h.py` (blob `de290021...`, byte-identical in NVIDIA's
  24GB repo and SergiioB's - the remote code `trust_remote_code` runs); `RP:` = `configuration_nemotron_h_puzzle.py`;
  `RC:` = `configuration_nemotron_h.py`; `RO:` = `modeling_nemotron_h_omni.py`.
- `N:` = transformers **5.19.0**'s native `transformers/models/nemotron_h/modeling_nemotron_h.py` (the oracle site,
  `oracle-out-q4exp/site`); it has no puzzle (per-layer) support, so it is a cross-check, not the reference (§2.0).
- `V/` = vLLM at `3901d56800` (the operator's checkout `/Users/urakozz/PycharmProjects/vllm`, branch `b70-xpu-stack`,
  Apache-2.0, **ideas only - no code is copied**): `V/models/nemotron_h.py`, `V/layers/mamba/mamba_mixer2.py`; upstream
  commits cited by hash: `3dc6def360` (varlen Mamba-2 decode for verification), `783380af39` (the chunk-scan NaN fix),
  `ee22f682ab` (shared-prefix prefill on hybrid models), `38168d5b82` (XPU fused-GDN toggle).
- `AR/` = intel/auto-round (`/Users/urakozz/PycharmProjects/auto-round`): `ec10a649` (#2512, Nemotron-H Omni in plain
  AutoRound calls - SergiioB's recipe and 25q's basis), `e31b7728` (any HF dataset as calibration).

## 0. The brief (operator, 2026-10-20) and the standing rules it meets

- **Format.** The engine's one format is AutoRound int4 g64 symmetric in GPTQ packing. SergiioB's g128 checkpoint loads
  **only** through the exact g128 -> g64 expansion `LinearSrc::classify` already performs (each g128 scale row written to
  both g64 rows it covers - `src/loader/quant.h`, docs/13 "g128 checkpoints"; spec 21b used it for Intel's interim
  Qwen3.8-Flash-Next experts) onto the existing g64 GEMV / expert kernels. No g128 kernel, no re-rounding. Two further
  **exact, loader-only** relaxations are needed for this checkpoint (§5.2) and nothing else.
- **Later checkpoint.** The operator re-quantises Bolt himself as g64 with a coding calibration set and more iterations
  (25q gives the recipe). The loader reads both - the g128-expanded checkpoint now, ours later - into **one** device
  layout; nothing downstream knows which one it read.
- **Kept as shipped:** bf16 embeddings, `lm_head`, router weights, norms, `conv1d` weight and bias; fp32
  `e_score_correction_bias`, `A_log`, `D`, `dt_bias` (stored F32 holding bf16 values - §1.3). The int8 head is built at
  load (spec 9).
- **Two cards by default** (41.9 GB of checkpoint, 40.46 GB on the cards - §3); one card only through spec 22's
  expert-offload tier, an option this spec records and does not build (Decision 4).
- **Agentic serving:** Qwen3-Coder XML tool calls (vLLM's `qwen3_coder` parser - our `server/toolcall.h` already parses
  this format) and the `ultra_v3` reasoning semantics; prefix-cache snapshots at aligned block ends carrying the SSM and
  conv state (vLLM's `--mamba-cache-mode align`).
- **Out of scope:** vision / video / audio, MTP (Bolt has none: `num_nextn_predict_layers` 0, `mtp_block_configs` []),
  tensor parallel (spec 17, unbuilt - recorded as the decode lever, §4.4), batching.

## 1. The model

### 1.1 The language model (`llm_config`, read 2026-10-20)

| | |
|---|---|
| layers | **88** blocks, each `norm -> mixer -> + residual` (R:767-840); `layers_block_type`: 40 `mamba`, 40 `moe`, 8 `attention`; pattern `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME` (M mamba, E moe, * attention); attention at **7, 16, 25, 36, 47, 58, 69, 78**; the last layer (87) is MoE |
| hidden | 4096; vocab **131072, every row a token** (BPE 131072 incl. ids 0-999 added; `vocab_used` = 131072); `max_position_embeddings` 262144; untied `embeddings` / `lm_head` (bf16) |
| norm | `NemotronHRMSNorm` (R:741-765): fp32 variance, `x · rsqrt(var + 1e-5)`, `w · x̂` in fp32, **one** rounding to bf16 (R:765) - not LFM2's two, not Qwen's `1 + w` |
| Mamba-2 mixer (40) | `mamba_num_heads` 128 x `mamba_head_dim` 64 = d_inner 8192; `n_groups` **8** (B, C shared by 16 heads); `ssm_state_size` 128; `conv_kernel` 4 with bias; `in_proj` 4096 -> 18560 = z 8192 ‖ xBC 10240 ‖ dt 128 (R:353, R:478-482; `d_mlp` 0); depthwise conv over xBC + SiLU; `A = -exp(A_log)` per head, `D` skip per head, `dt_bias` per head; gated RMSNorm over **groups of 1024** with `norm_before_gate` false (R:278-308, R:371); `out_proj` 8192 -> 4096; `chunk_size` 128 |
| attention (8) | 32 q / 2 kv heads (GQA 16) x 128; q, k, v, o projections, **no bias, no q/k norm, no RoPE** (NoPE; R:1104-1188), scale 128^-0.5 (R:1137), causal |
| MoE (40) | **LatentMoE**: router on the hidden 4096; `fc1_latent_proj` 4096 -> 1024; **routed experts in the 1024 latent**: `up_proj` 1024 -> I, `relu²`, `down_proj` I -> 1024 (non-gated, R:871-881); `fc2_latent_proj` 1024 -> 4096; one **shared expert on the hidden 4096**: up 4096 -> 5376, relu², down 5376 -> 4096, added after `fc2` (R:952-965) |
| per-layer experts | heterogeneous ("Puzzle", RP:68-100): `n_routed_experts` E ∈ {240 (12 layers), 304 (6), 368 (9), 432 (10), 496 (3)}; `moe_intermediate_size` I ∈ {1536 (1), 2048 (11), 2304 (15), 2560 (13)}; 12 distinct (E, I) pairs (25a's sheet lists them by layer); top-**9** everywhere |
| router | `NemotronHTopkRouter` (R:968-1063): **fp32** logits (x and W widened, R:1044), `sigmoid` fp32, selection on `s + e_score_correction_bias` (fp32 buffer), `n_group` **1** / `topk_group` 1 (the group mask is all ones), top-9 `sorted=False`, weights `s` at the ids, renormalised `/ (Σ + 1e-20)`, `x routed_scaling_factor` **5.0** |
| tokens | BOS `<s>` 1 (the template never writes it; the post-processor adds none), `<|im_start|>` 10, `<|im_end|>` 11, `</s>` 2, `<think>` 12 / `</think>` 13, `<tool_call>` 14 / `</tool_call>` 15, `<tool_response>` 16 / 17 (added, **not** special), `<image>` 18; mask id of the drafter 990 = `<SPECIAL_990>` |
| tokenizer | BPE, `ignore_merges` true, 269,443 merges, byte-level; split regex with `\p{M}` classes (`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|...`, the class spec 21e implemented); `fix_mistral_regex` true in `tokenizer_config.json` is a no-op here (transformers 5.19.0 returns before patching when the config's `transformers_version` >= 5.0.0, `tokenization_utils_tokenizers.py:1442-1458`, and the shipped regex already IS the "fixed" one) |
| generation config | SergiioB's: `eos_token_id` 11 only, no sampling keys; NVIDIA's (24GB repo): `do_sample` true, T 1.0, top-p 0.95, `eos_token_id` [2, 11] (Decision 12) |

**What the brief had wrong (corrected here, and in the final report):** the routed-expert counts are 240 / 304 / 368 /
432 / 496 (not "240/304"), the routed intermediate varies by layer (1536-2560), and `n_groups` 8 is the **Mamba**
B / C group count - the router has `n_group` 1 (no grouped routing in effect).

### 1.2 The SSM state and the conv state (derived)

- SSM state per Mamba layer: 128 heads x 64 x 128 = 1,048,576 values: **4 MiB fp32 / 2 MiB bf16**; 40 layers **160 MiB
  fp32** (167,772,160 B) a sequence. `mamba_ssm_cache_dtype` is `"float32"` in the config; NVIDIA's vLLM recipe overrides
  it to bfloat16 and its card says the architecture search budgeted "16-bit SSM states" (published). Decision 2.
- Conv state: the last 3 raw xBC columns (10240 bf16) a layer: 40 x 3 x 10240 x 2 = 2,457,600 B. The engine keeps a
  **16-slot ring** of raw xBC per layer (13.1 MB), the GDN ring's depth, so M-row launches never collide (spec 24 §4.1's
  argument at taps 4: rows of one launch read `pos-3..pos+M-2` and write `pos..pos+M-1`, disjoint mod 16 for M <= 13).

### 1.3 The checkpoint as shipped (headers range-read 2026-10-20; scale data sampled)

- **Names:** `language_model.model.layers.<l>.{norm.weight, mixer.*}`, `language_model.model.embeddings.weight`,
  `language_model.model.norm_f.weight`, `language_model.lm_head.weight`; 849 vision / projector tensors
  (`vision_model.*`, `vision_projector.*`, `mlp1.*`; 393 of them in `model_extra_tensors.safetensors`) skipped by name.
  NVIDIA's own repos name the trunk `language_model.backbone.*` (`hf_quant_config.json`); AutoRound #2512 renames it at
  load (`AR/auto_round/modeling/nemotron_h_omni.py`, `_NESTED_TRUNK_SOURCE`) - the loader accepts `backbone` as an alias
  only if 25b finds a checkpoint that ships it (none of ours will).
- **Int4 (GPTQ v1: `qweight` I32 [K/8][N], `scales` F16 [K/128][N], `qzeros` I32, `g_idx` I32 [K]):** every routed
  expert's `up_proj` / `down_proj` (13,824 experts, 27,648 linears), `shared_experts.{up,down}_proj`,
  `fc{1,2}_latent_proj`, Mamba `in_proj` / `out_proj`, attention `q/k/v/o_proj` - 27,920 linears (matches SergiioB's card).
- **Sampled (range reads of ~3 MB):** `qzeros` = 0x77777777 (layer 0 `out_proj`, layer 1 expert 0 `down_proj`);
  `g_idx` = `i // 128` (layer 0 `out_proj`); **every `scales` tensor F16** - including `mixer.out_proj`, although #2512
  pins its scale dtype to bf16 for in-memory quantisation (`AR/auto_round/special_model_handler.py`
  `_PRE_DEFINED_LAYER_CONFIG`); **about half the scales are negative** (layer 0 `out_proj`: 132,409 of 262,144) -
  AutoRound's signed symmetric scale, which lets a group's extreme map to -8; layer 0 `in_proj` and layer 1 expert 0
  carry f16 **subnormal** scales (6,773 and 64) down to 1.0e-5; no zero scales in the samples. 25a's facts sheet scans
  every tensor (on the box, where the weights are).
- **Kept tensors:** `gate.weight` BF16 [E][4096], `gate.e_score_correction_bias` **F32** [E] (layer 1: 4.470 .. 4.550 -
  large, nearly equal: selection is decided by small differences on top of 4.5), `A_log` / `D` / `dt_bias` **F32** [128]
  (holding bf16 values, SergiioB's card), `conv1d.weight` BF16 [10240][1][4], `conv1d.bias` BF16 [10240],
  `mixer.norm.weight` BF16 [8192], every block `norm.weight` BF16, `embeddings` / `lm_head` / `norm_f` BF16.
- **`quantization_config.dynamic`** carries 43 `-:` exclusions (embeddings, `lm_head`, vision, the 40 routers) **and one
  `+:` rule**, `"+:.*mixer\\.out_proj$": {bits 4, group_size 128, sym true}` - a restatement of the top level.
  `loader::QuantConfig::parse` refuses every `+:` rule today (`src/loader/quant.cc:359-366`, the broken-MTP-config
  guard). §5.2.

### 1.4 The original and the 24GB release

NVIDIA's `...-24GB` repo (the one the vLLM recipe serves and LiLiCorr was trained against) is a ModelOpt mixed-precision
export: routed **and shared** experts IQ2_XXS (2.0625 bit, ggml packing), Mamba / attention linears NVFP4, `lm_head`
Q8_0 (`hf_quant_config.json`), recovered with quantisation-aware distillation (published). We never load it (one format);
it matters to spec 26 because LiLiCorr's acceptance was trained and published against it. The BF16 original is 25a's
quality reference and 25q's quantisation input.

### 1.5 The licence

OpenMDW-1.1 on all three repos (the cards' `license_link` https://openmdw.ai/license/1-1/). The BF16 repo's `LICENSE`
file is listed with **0 bytes** (HF API tree, 2026-10-20). The repository stores no weights and no licence text; 25b
may vendor `config.json` and other model files, and our g64 export may be published (Decision 20, ruled 2026-10-20),
and 25q's card copies the licence by link and states the derivation.

## 2. Reference semantics (pinned from source)

### 2.0 Which reference (Decision 1)

There are three readings of this model's arithmetic, and they disagree:
1. **The training kernels** (mamba_ssm's Triton `selective_state_update` / `mamba_chunk_scan_combined` /
   `causal_conv1d` / `rmsnorm_fn`, which NVIDIA trains with and vLLM serves with): every SSM quantity in fp32 inside the
   kernel, the conv accumulated in fp32 with the bias and SiLU before one rounding, `dt_limit = (0, inf)` - no lower clamp
   (`V/layers/mamba/mamba_mixer2.py` passes `dt_limit=(0.0, float("inf"))` everywhere; R:449 + R:522 pass the config's
   `time_step_limit`, whose default is (0, inf), RC:225).
2. **The remote code's torch fallback** (R:535-726, what transformers runs on a CPU): in a bf16 model it rounds the
   decode conv's four products to bf16 before summing (R:562, when the cache is bf16), computes `dt + dt_bias`, `D · x`,
   `dB · x` in bf16, casts the SSM state to bf16 before `C · S` (R:626), runs the prefill conv as a bf16 `conv1d` with a
   rounding before the SiLU (two roundings, R:573) - and **clamps dt at `time_step_min` = 0.001** (R:595, R:643). The
   checkpoint's `dt_bias` reaches -10.5: in layer 0, **25 of 128 heads** have `softplus(dt_bias) < 0.001` (measured from
   the shard), so at small `dt` the clamp moves their decay `exp(dt · A)` by orders of magnitude (A down to -82). It
   also needs `mamba_ssm` at import (R:60-63: `rmsnorm_fn` raises without it).
3. **transformers 5.19.0's native class** (N:): no puzzle support, clamps the chunk path at `time_step_min` (N:394,
   N:275) but not the recurrent one; another rounding set.

**Decision 1 (approved, operator 2026-10-20):** the reference is **the remote code (R) run through transformers 5.19.0 with its kernel path
enabled on the CPU** - R's `cuda_kernels_forward` (R:384-532), which calls exactly the five training-kernel functions
(`causal_conv1d_update` / `causal_conv1d_fn`, `selective_state_update`, `mamba_chunk_scan_combined`, `rmsnorm_fn`) and
passes `dt_limit` (0, inf) - **with plain-torch implementations of those five** (`tools/oracle/bolt_mamba_ops.py`, fp32
internals, in the order the engine's twin uses) and **an fp32 SSM / conv cache passed in explicitly** (R:1664-1667 would
build it in the model's dtype). Everything outside the Mamba mixer (norms, attention, router, experts, combine, head, the
model loop) is R's own code and the port equals R there bitwise, fp32 and bf16. The Mamba mixer is held two ways: to the
engine's twin bitwise (the ops compute in the twin's order), and to R's torch fallback in fp32 within a measured bound -
the fallback is the same function summed in other orders (the decode conv's `torch.sum`, the `bmm`, the naive chunk scan),
except the clamp, which a variant built to trigger it exercises on its own. In bf16 the fallback's extra roundings and the
clamp are listed, measured on the tiny model and on real rows (25a), and **not** copied. The engine implements reading 1.
*Cost if wrong:* if the operator prefers the torch fallback as the reference, every SSM kernel gains bf16 rounding points
and the clamp, 25a's sheet changes and 25c's twins are re-pinned (~2 days); serving would then disagree with vLLM /
NVIDIA's training on 20 % of layer-0 heads.

### 2.1 The block (R:802-840), `h` bf16 [T][4096]

```
x = rms_norm(h)                                   one rounding (R:760-765), eps 1e-5
y = mixer(x)                                      Mamba-2 | attention | MoE, bf16 out
h = rne(h + y)                                    bf16 add (residual_in_fp32 false)
after layer 87: logits = lm_head(rms_norm_f(h))   bf16 linear, .float() (R:1731)
```

### 2.2 The Mamba-2 mixer (reading 1, per token; the decode recurrence)

```
zxbcdt = rne(in_proj(x))                       [18560] bf16: z [0, 8192), xBC [8192, 18432), dt_raw [18432, 18560)
ring[pos % 16] = xBC                           the raw projection (the reference's conv state holds inputs, R:568-572)
xc[c]  = rne(silu(fma-chain_{j=0..3}(w[c][j] · f32(xBC[pos-3+j][c])) + b[c]))   taps oldest first, fp32, bias, SiLU,
                                                ONE rounding (causal_conv1d_update); positions < 0 read 0
x, B, C = xc[0, 8192), xc[8192, 9216), xc[9216, 10240)           heads h = 0..127 of 64; groups g = h / 16 of 128
dt_h   = softplus(f32(dt_raw_h) + dt_bias_h)    fp32; softplus(v) = v > 20 ? v : log1p(exp(v)) (mamba_ssm's threshold);
                                                NO clamp
A_h    = -exp(A_log_h)                          fp32, computed once at load with torch's exp (25a pins the host function)
dA_h   = exp(dt_h · A_h)
S[h][p][n] = S[h][p][n] · dA_h + (dt_h · f32(x[h][p])) · f32(B[g][n])           fp32 state (Decision 2)
y[h][p]    = rne(Σ_n S[h][p][n] · f32(C[g][n]) + D_h · f32(x[h][p]))           the reduction order is 25a's to pin
v      = f32(y) · silu(f32(z))                  rmsnorm_fn, norm_before_gate false (R:300-308)
out    = rne(v · (1 / sqrt(mean_{1024-group}(v²) + 1e-5)) · w)            per group of 1024 (8 groups)
mixer  = out_proj(out)                          bf16
```

The reduction orders inside `Σ_n` and the group mean are not defined by any reference to the bit (Triton's are tiled);
25a fixes the engine's (a fixed pairwise tree) and holds the twin bitwise to it, and holds the remote-code port to the
twin at fp32 within a measured bound. Prefill computes the same function in the chunked (SSD) form (§6.2); it is held
to decode's fill by cosine bars, not bitwise (the GDN precedent, spec 2 / docs/14).

### 2.3 Attention (R:1104-1188, eager R:1077-1101)

`q, k, v = rne(linear(x))`, no norm, no position encoding; the cache holds raw k, v. Eager: `rne(q · kᵀ)`, `x 128^-0.5`
(rounded), `+ causal mask`, softmax in fp32 then bf16, `rne(p · v)`, `o_proj`. The engine's default is flash (fp32
probabilities, held to fp64 at cosine >= 0.99999 - spec 6 K1) with an eager build for A/B (`B70_BOLT_ATTN=eager`, spec
18 §10.1's rule). **Same shape as Muse-Glimmer's full layers** (32 / 2 x 128, NoPE, causal): §6.1.

### 2.4 The MoE block (the remote code's eager path, R:925-965 and R:1040-1052)

```
l     = f32(x) · f32(Wg)ᵀ                       [E] fp32 (F.linear in fp32, R:1044; torch's CPU sgemm order - 25a measures
                                                how often our fixed order differs by an ulp)
s     = sigmoid(l)                              fp32
sel   = s + bias                                fp32 (the group step is the identity at n_group 1, R:1000-1017)
ids   = top-9 of sel                            torch.topk(sorted=False): the ORDER is unspecified; the SET is what counts.
                                                Ties: the engine takes the lower id; exact ties are undetermined (24a's rule)
w     = s[ids] / (Σ_ids s + 1e-20) · 5.0        fp32, never rounded (R:1047-1051)
lat   = rne(fc1_latent(x))                      [1024]
u_e   = rne(up_e(lat)); a_e = rne(rne(relu(u_e))²)    relu2 = relu then torch.square, both on bf16 (activations.py:206)
d_e   = rne(down_e(a_e))                        [1024]
acc   = 0 (fp32); for e in ids ASCENDING:  acc = acc + f32(rne32(f32(d_e) · w_e))      index_add_ in ascending expert id
                                                (R:930-940): the product rounded to fp32, then the add - no fma
r     = rne(acc)                                [1024] (.type(hidden dtype), R:950)
o     = rne(fc2_latent(r))                      [4096]
sh    = rne(down_sh(rne(rne(relu(rne(up_sh(x))))²)))  the shared expert on the NORMED hidden x (R:964)
mixer = rne(o + sh)                             bf16 add
```
The combine is Kolibri's (ascending id, product then add, one rounding - `kol_moe.cl`'s header) without its shared
slot: Bolt's shared expert runs on the 4096 hidden, not in the 1024 latent, so it cannot be a tenth slot of the latent
kernel (§6.1). Decision 10 records the combine choice (vLLM's fused path sums in another order).

### 2.5 The model loop

`h = embeddings[ids]` (bf16 row); 88 blocks; `norm_f`; `lm_head` (bf16, untied). `output_hidden_states[i]` is the
**input** of block i (R:1481-1482), so "the residual after layer i" - the DFlash tap convention (spec 19 §2,
`dflash_ref.py`'s `taps_from_hf_hidden_states`) - is `hidden_states[i + 1]`; vLLM captures aux index i + 1 after block i
(`V/models/nemotron_h.py:643-653`). LiLiCorr taps layers 1, 18, 35, 51, 68, 85 (spec 26).

## 3. What it costs (derived)

| part | parameters | on the cards | read per decoded token |
|---|---:|---:|---:|
| Mamba `in_proj` + `out_proj` (40 x 109.58 M), int4 g64 (0.53125 B a parameter) | 4,383,047,680 | 2.328 GB | 2.328 GB |
| attention q, k, v, o (8 x 35.65 M), int4 g64 | 285,212,672 | 0.152 GB | 0.152 GB |
| MoE dense: `fc1` + `fc2` latent + shared up / down (40 x 52.43 M), int4 g64 | 2,097,152,000 | 1.114 GB | 1.114 GB |
| routed experts (13,824 experts), int4 g64 | 66,144,174,080 | **35.139 GB** | 0.900 GB (9 a layer) |
| routers, bf16 | 56,623,104 | 0.113 GB | 0.113 GB |
| `embeddings`, bf16 (a 8 KB row gathered) | 536,870,912 | 1.074 GB | 8 KB |
| `lm_head`, int8 per row + fp32 scales (spec 9; bf16 1.074 GB) | 536,870,912 | 0.537 GB | 0.537 GB |
| conv taps / biases, norms, A / D / dt_bias, router biases (fp32 on the cards) | 2.77 M | ~11 MB | ~11 MB |
| **total** | **74,042,720,768** | **40.46 GB** | **5.145 GB** |

- **Active per token:** 9,052,880,896 (derived; the card's "9B active"). The g128 checkpoint stores its linears at
  4.125 bit (37.6 GB); the g64 expansion costs 3 % more bytes on the card than g128 would (docs/13's known price).
- **Per-token state traffic:** the SSM state is read and written every token: 2 x 167.8 MB = **0.336 GB fp32** (0.168
  bf16) - 6.5 % of the weight bytes, more than every attention layer together. KV: **8,192 B a position**
  (8 x 2 x 128 x {K, V} x bf16): 0.54 GB at 65536, 2.15 GB at 262144.
- **Decode roofline** at 590 GB/s (docs/05): 5.48 GB a token -> **107.6 t/s** at short depth (weights + fp32 state);
  at Qwen3.8's 77.6 % of roofline **~84 t/s** (estimated); at depth 65536 +0.54 GB -> ~77 t/s; at 262144 ~64 t/s
  (estimated). **Two cards in a pipeline do not add bandwidth at batch 1** (the stages run in sequence, spec 16 §1): these
  are single-card-bandwidth figures, and they still beat the published TP-2 vLLM figures by 1.4x at 886 tokens and 3.3x at
  65536 (estimated against published - 25c measures).
- **The step is many small kernels** (spec 24 §3's lesson): §6.1's list is **653 launches a token** (derived: 40 x 6 + 8 x
  6 + 40 x 9 + 5), 533 with the three fused arms. At docs/15's 0.73 µs dispatch gap that is ~0.48 ms of gaps in a ~12 ms
  step (estimated) - smaller in proportion than LFM2's, because Bolt's GEMVs are large (in_proj 40.4 MB, shared up 11.7
  MB). **25c's P0 measures every term before the default list is chosen.**
- **Memory, two cards (`--pp 2`, split by bytes at layer 48, `runtime::pp_balance`):** device 0 holds the embedding and
  layers 0-47 (19.93 GB of weights; 22 Mamba layers, attention 7 / 16 / 25 / 36 / 47), device 1 layers 48-87, `norm_f`
  and the int8 head (20.54 GB; 18 Mamba layers, attention 58 / 69 / 78). At 262144 positions KV is 1.34 / 0.81 GB, the
  SSM state 92 / 75 MB (fp32) per slot, the conv rings 7.2 / 5.9 MB, decode scratch < 100 MB, prefill scratch ~1.5 GB a
  card (estimated; 25d's planner states it). **~22.9 / 23.0 GB against 31.0 GB usable a card** (32 GB less the 1.5 GB
  reserve): room for spec 26's drafter (+1.1 GB int8, + the 1.07 GB embedding replica on device 1) and spec 19c's
  verify slots (7 more state copies: +0.65 / +0.53 GB at fp32). The full 262144 context fits.
- **One card** does not hold it: 40.46 GB > 31 GB (Decision 4).
- **Prefill (estimated):** 18.1 GFLOP a token active (2 x 9.05 B); a 2048-row chunk routes 18,432 (token, expert) pairs
  over 240-496 experts - 37-77 rows an expert on average, a fed grouped GEMM (spec 15d's `pf_moe_gemm`, K 1024, N = I;
  K = I, N 1024). Only 8 attention layers grow with depth: at 65536 the flash term is ~4 GFLOP a token against 18 (spec 6's
  measured rates), so prefill should stay near flat with depth where vLLM's falls 6x (published) - 25d records the rate,
  no bar before it runs.

## 4. Placement, speed levers and what is not built

### 4.1 Two cards (default)

Spec 16b-16d's pieces: `runtime::pp_balance` places stages by bytes (§3: split 48), the hand-off is `h` (4096 bf16, 8 KB)
after the last block of device 0, `copy` or `peer`, the bounded fence, `--pipeline-split auto|N`. In §6.1's list no
block folds its own output - the next block's norm folds it - so every cut also sends the pending split-K slices of
device 0's last block (S x 4096 fp32, two slice sets after a MoE block: 24c's dense-cut departure, here the rule at every
cut), and device 1's first `nb_res_norm` folds them and recomputes Σ.

### 4.2 One card through spec 22 (Decision 4 - option, not built)

Non-expert weights 5.3 GB + 35.1 GB of routed experts against 31 GB: about 63 % of the expert bytes resident, the rest
over PCIe 3.0. Each expert block is 1.7-2.8 MB; a token reads 360. Spec 22a's probe (row 36) measures the link and the
hit-rate curve; Bolt would need its own routing trace (25a's golden sets record routes). Not in these plans.

### 4.3 Tensor parallel (recorded)

At batch 1 the two cards' bandwidth adds only under TP (spec 17, designed, unbuilt): ~2x decode for this bandwidth-bound
model (SergiioB's 60.5 t/s is TP 2). It is the next lever after 25e if the operator wants it; nothing here blocks it
(the descriptor carries per-linear shapes; the Mamba heads and the experts split by 2 cleanly - 128 heads, 8 groups,
E even in every layer).

### 4.4 What the speed rows compare

25c / 25e record decode at depth 886 / 8192 / 32000 / 65536 / 131072 / 262144 and prefill at the same depths, beside
SergiioB's published table and the operator's own vLLM XPU run of the same g128 checkpoint (llama-benchy, row 41's
matrix) - like for like.

## 5. Formats

### 5.1 The one format, two sources

| | SergiioB g128 (now) | ours g64 (25q, later) |
|---|---|---|
| packing | GPTQ v1, `quant_method` gptq, `provider` auto-round, `g_idx` shipped | `auto_round:auto_gptq` (or the same GPTQ spelling - 25q records which), no `g_idx` |
| groups | 128, expanded to 64 at load (exact) | 64, as stored |
| scales | F16, signed | F16 (Decision 9), signed |
| names | `language_model.model.*` | the same (25q Task 1 proves it) |
| device layout | identical | identical |

The loader decides nothing from labels (docs/02): `LinearSrc::classify` reads each linear's scale rows (K/64 or K/128)
and expands; the descriptor's per-linear `{S, layout}` (24b's I1 rule) is the same for both sources.

### 5.2 The exact loader relaxations this checkpoint needs (25b)

1. **A `+:` dynamic rule that restates the top level is accepted.** `QuantConfig::parse` keeps refusing every `+:` rule
   whose `bits`, `group_size` or `sym` differ from the top-level values, or which carries any other key; a rule whose
   three values equal them is a no-op and is counted (`dynamic_restating`), and `LinearSrc::classify` still checks every
   module against its shipped tensors. Test: SergiioB's exact `dynamic` object accepted; the broken-MTP pattern (a `+:`
   rule with `bits 8`, or `group_size 64` under a g128 top level) still refused.
2. **Signed scales.** No path may assume a positive scale: the dequant is `(q - 8) x scale` everywhere, but 25b audits and
   tests every consumer (the GEMV / expert kernels, the prefill dequant slabs, `assert_quant_invariants`, `check.py`, any
   "amax from scales" diagnostic) with negative scales, and the g128 expansion copies the sign bit unchanged.
3. **(Only if a future export ships them) BF16 `.scales`** are converted with the existing `bf16_to_f16_exact`
   (`src/loader/quant.cc:169-186`, today compressed-tensors only) - exact or refused, naming the tensor and the element.
   SergiioB's checkpoint does not need it (every scale is F16); 25q's recipe avoids it (Decision 9).

Nothing else changes in `quant.*`; `g_idx` must be `i // 128` (or absent), `qzeros` 0x77777777 (the existing asserts).

### 5.3 bf16 and fp32 tensors

Router `[E][4096]` bf16 in `gemv_bf16`'s tiles (E is a multiple of 16 in every layer: no padding rows), its fp32
logits never rounded (Kolibri's `GateLinear(out_dtype=fp32)` form); `e_score_correction_bias` fp32; `A_log` -> `A =
-exp(A_log)` fp32 at load; `D`, `dt_bias` fp32; conv taps and bias fp32 (bf16 widened: exact); gated-norm and block norm
weights fp32 (widened); embedding bf16 tiled (the gather and, with `--lm-head bf16`, nothing else - the head is untied);
`lm_head` int8 per row built at load (spec 9's `quantise_int8_tiled`) or bf16 tiles.

### 5.4 KV

bf16, 8 KiB a position. int8 KV (spec 12) is not worth its complexity here (Decision 17).

## 6. The engine

### 6.1 Decode (25c)

Family `nb_*` (Nemotron-Bolt), own descriptor / loader / kernels / engine on Kolibri's and LFM2's pattern (spec 20 §11,
spec 24 §6). The residual ping-pongs between two buffers across norm sites (24c's rule).

```text
Mamba (40), 6:      nb_res_norm _S<s> · in_proj GEMV (K 4096, N 18560) · nb_conv · nb_ssm · nb_gnorm · out_proj GEMV
                    (K 8192, N 4096; its slices folded by the next norm)
attention (8), 6:   nb_res_norm _S<s> · q‖k‖v GEMV (N 4608) · nb_attn_prep (KV write) · nb_attn · nb_attn_reduce ·
                    o GEMV (slices folded by the next norm)
MoE (40), 9:        nb_res_norm _S<s> · nb_router (bf16, fp32 out) · nb_route · fc1‖shared_up GEMV (K 4096,
                    N 1024 + 5376) · nb_sh_act (shared relu², latent rounding) · nb_moe_up (9 slots, relu²) ·
                    nb_moe_down (9 slots, the combine, r) · fc2 GEMV (K 1024) · shared_down GEMV (K 5376) -> the next
                    norm folds o + sh into h (two slice sets, §2.4's roundings)
head, 5:            nb_embed_gather (device 0, first) · nb_res_norm + norm_f · head GEMV (int8 | bf16) ·
                    argmax stage 1 · stage 2 (VOCAB 131072, VOCAB_USED 131072)
```
**653 launches a token** (derived; 25c's plan test asserts the formula and `ProfileEvents` must hold it - 24c raised the
capacity to 1024). Eager attention +2 a layer (669). The fused arms, each bitwise the as-built list: `router` (the E router
rows inside the MoE norm launch, LFM2's `_R` arm) -40, `route` (the rank recomputed in every `nb_moe_up` work-group,
LFM2's arm `r`) -40, `conv` (the conv computed inside `nb_ssm` for its own channels; B and C recomputed per head-group
work-group) -40: **533 with all three** (derived). Decision 16.

**The kernels:**
- `nb_conv` / `nb_ssm` / `nb_gnorm` (`src/kernels/bolt/nb_mamba.cl`): §2.2 exactly; `nb_ssm`'s grid over (head, 16-dim
  slice), each work-group owning 16 x 128 fp32 state values in registers, M rows walked in order (one launch serves the
  M-row verify, spec 26); with spec 19c's slot form (`SPEC_SLOTS`) the state after row m goes to slot `(live + m) % 8`.
  The reductions: Σ_n over 128 by a fixed pairwise tree across the sub-group; the gated norm's Σ v² over each 1024-group by
  one fixed tree (every work-group of a group recomputes it - LFM2's `res_norm` rule - or a two-launch split if P0 shows
  the recompute costs; Decision 16).
- `nb_route` (`nb_moe.cl`): `kol_route`'s structure (256 lanes, two experts a lane: E <= 512) with §2.4's chain - fp32
  sigmoid, `sel = s + bias`, rank by count with exact ties to the lower id, **top-9**, Σ of the nine `s` in fp32 (rank
  order: an ulp-level choice 25a pins against torch's `sum`), `w = s / (Σ + 1e-20) x 5.0`; the route row widened to
  **48 words** (ids [9] ascending, w [9], sel [9], the 10th sel - the near-tie margin -, Σ, the smallest selected sel).
  **Top-9 is not a problem for our kernels:** the rank is a count over SLM (a permutation for any E), the selected set is
  `rank < TOP_K` for any TOP_K, and the up / down kernels loop over SLOTS = TOP_K in a fixed sequential order through SLM
  - no pairwise tree over slots, so no power of two is needed anywhere (`moe.cl:120-140`, `k2_moe.cl:126-186`). The only
  hard limit is the 8-slot route row (`#error "TOP_K is 1..8"` in moe.cl / k2_moe.cl / kol_moe.cl), which the family
  copy widens. No padding to 10 (vLLM's XPU kernel needed it: SergiioB's `b70/patch_xpu_moe_topk_pad.py`, +11 % expert
  work).
- `nb_moe_up` / `nb_moe_down`: `kol_moe_gate_up` / `_down`'s layout-1 `tile_dot` and fixed-order merges, **non-gated**
  (up only: `NT_UP = I / 16`), relu² epilogue, K = 1024 latent; the combine in ascending expert id (§2.4), one rounding.
  One binary per I (1536 / 2048 / 2304 / 2560); the expert block of layer l at `base_l + id x block(I_l)` through one
  address macro (spec 15 decision 3; spec 22 could swap it). Expert blocks are multiples of 8,704 B (I x 544), so every
  block base is 64 B aligned when the layer base is (the LSC 2D-load rule, vllm-xpu-kernels 6f5935c).
- `nb_attn` / `nb_attn_reduce` and prefill flash: **Muse-Glimmer's `mg_attn` full NoPE path is the same shape** (32 / 2
  x 128, GQA 16, causal, no RoPE). Its reduce applies Muse's output gate `rne(rne(o) x g)`; with g a constant buffer of
  bf16 1.0 that is exactly `rne(o)`. Decision 15: reuse `mg_attn`'s binaries from new CMake lines with a ones buffer
  (zero new attention code; the prefill flash `mg_pf_flash_attn` is gate-free already, 23d) or a gate-free family copy.
  Either way depth 262144 is new ground for that kernel (Muse ran to 131072): 25c's twins go to 262143.
- Norms: `nb_res_norm` (LFM2's one-launch-per-site form at hidden 4096 with Bolt's one-rounding chain), the MoE site's
  two-source fold `h = rne(h + rne(rne(Σ fc2) + rne(Σ sh)))`.
- Reused at Bolt's shapes from new CMake lines: `gemv.cl` (int4, every shape x S x layout the descriptor names; M 1..8
  for spec 26's verify), `gemv_bf16.cl` (routers; the bf16 head), `gemv_i8w.cl` (int8 head), `argmax.cl` (131072).

**Never route padding rows** (vllm-xpu-kernels 8b95ea6): every M-row list runs exactly M real rows (one captured list
per M, spec 23f's lazy capture), so no row with a dummy route reaches an expert.

### 6.2 Prefill (25d)

The `l0` backend, chunks <= 2048, on the GDN prefill's discipline (`pf_gdn_conv.cl`'s seed-then-conv over the ring,
`pf_gdn_scan.cl`'s record):
- **Mamba-2 in the chunked (SSD) form** over 64- or 128-row sub-chunks (25d picks; the config's training chunk is 128):
  intra-chunk `Y_diag = (C Bᵀ ⊙ L) · (x ⊙ dt)` with `L = exp(segsum(A dt))`, inter-chunk states, the state carried in
  **fp32**. DPAS only where both operands are bf16 already (`C Bᵀ`: B and C are the conv's bf16 outputs); every product
  with the fp32 state stays on the vector path - the GDN scan's DPAS variant cost a token for exactly that reason
  (`pf_gdn_scan.cl:9-58`). The conv over a chunk seeds from the ring and writes back the chunk's last three columns.
  Edge cases from `783380af39`'s test (a partial last sub-chunk, 181 = 128 + 53 rows; ngroups 8, dstate 128, headdim 64;
  every tiling variant finite and equal; 10 repeats) and `3dc6def360`'s (query lengths 0 / 1 / ragged, accepted counts
  1..M) become 25d's / 26c's kernel tests.
- Linears: int4 dequant slabs + `pf_gemm` (spec 20d's `l0` path) at K 4096 / 8192 / 1024 / 5376.
- MoE: the grouped expert GEMM (spec 15d's `pf_moe_gemm` structure, non-gated, K 1024 latent, relu² epilogue) with a
  sort / gather / combine family file (`nb_pf_moe.cl`: 9 slots, up to 496 experts, the ascending-id combine); only the
  chunk's real rows are sorted (no padding rows routed).
- Attention: `mg_pf_flash_attn`'s full causal NoPE build at Bolt's defines (Decision 15) and its eager twin.
- Two cards: Kolibri's sequential hand-off per chunk (spec 20 §12).
- Gates: prefill against decode's fill by cosine bars (h taps, the SSM state, logits; PROPOSED, 25d), splits at
  multiples of 64 bitwise, chunked = whole bitwise (the SSD sub-chunks are absolute: a chunk boundary is a sub-chunk
  boundary - 25d's walk enforces it).

### 6.3 Snapshots and the prefix cache (25e)

Spec 7's store with the hybrid split (Qwen3.8's GDN snapshots, spec 23e / 24e's EngineIface hooks):
- **State at a block end p:** the SSM state of the 40 Mamba layers (167.8 MB fp32 / 83.9 MB bf16), the conv ring's
  three raw xBC columns before p (2.46 MB), written back at their slots on restore. **Snapshots only at aligned block
  ends** - the recurrent state exists for exactly one position, so a hit restores the largest block end <= the common
  prefix and prefills the rest (vLLM's `--mamba-cache-mode align`, the same idea).
- **Blocks:** the 8 attention layers' KV, 8,192 B a position (16.8 MB a 2048-block).
- With spec 26's drafter: its 1024-slot ring and the anchor row ride in the snapshot (spec 26 §6).
- One host layout for `--pp 1` and `--pp 2` (16b's rule) - although `--pp 1` cannot hold the model, the layout is
  defined by layer, not by device.
- A hybrid model shares recurrent state only through snapshots: two requests that start together on a shared prefix
  each prefill it (`ee22f682ab`'s observation). With batch 1 it does not arise; spec 13 would prefill the shared prefix
  once first.

## 7. Serving (25e)

- **Tokenizer:** HF `tokenizers` in Rust reads `tokenizer.json` as is (§1.1); 25e holds it to transformers 5.19.0 id for
  id on a corpus. No BOS anywhere (the template writes none, the post-processor adds none).
- **Template** (`chat_template.jinja`, sha256 `6521afc7...`, 12,514 B, byte-identical in NVIDIA's and SergiioB's repos):
  ChatML; the system block is **always** rendered (`system_message` is set to "" when absent, so `<|im_start|>system\n`
  + `<|im_end|>\n` open every prompt); tools as an XML `<tools><function><name>...` block with the Qwen3-Coder calling
  instructions; history tool calls as `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n
  </tool_call>` (V `tojson` for mappings / lists, `| string` otherwise); tool results grouped into one user turn as
  `<tool_response>\n...\n</tool_response>`; the generation prompt `<|im_start|>assistant\n<think>\n`, or
  `<|im_start|>assistant\n<think></think>` with `enable_thinking: false`; `medium_effort: true` appends
  `\n\n{reasoning effort: efficient}` to the last user message; `truncate_history_thinking` (default true) drops the
  reasoning of assistant turns before the last user message. minja features to check (25e): `|items`, `is mapping`,
  `is sequence`, `is string`, `loop.previtem` / `nextitem`, namespaces, macros with `caller`-less recursion,
  `.split`, `| trim`, `| tojson | safe`, `| default('', true)`.
- **Tool calls:** Qwen3-Coder XML through the existing `server/toolcall.h` parser (schema-first value typing,
  `convert_value`) - the format vLLM's `qwen3_coder` parser reads. 25e holds it to vLLM's parser on a fixture of the
  model's own outputs (25a's A4 reference) and adds Bolt's edge cases: multi-line values, values with `</parameter>`-like
  text, a call cut by `max_tokens` (dropped and stripped, spec 23 decision 16's rule), calls inside reasoning (reasoning).
- **Reasoning (`ultra_v3`, the reasoning-parser plugin both repos ship):** with thinking on, the prompt ends in
  `<think>\n`, so the output opens inside reasoning up to `</think>`; generation that ends inside reasoning is reasoning,
  unless `force_nonempty_content` is set (then it becomes content - ultra_v3's rule); with `enable_thinking: false` there
  is no reasoning channel and everything, including a stray `</think>`, is content. A full assistant span re-parsed (not
  our live path) drops one leading newline from the reasoning (the template owns it). `ChatFormat::Kind::Bolt`
  (`template_kwargs`: `enable_thinking`, `medium_effort`, `truncate_history_thinking`, `force_nonempty_content`;
  Decision 13 maps OpenAI's `reasoning_effort`).
- **Known prefix-cache miss:** a new user message re-renders every earlier assistant turn without its reasoning
  (`truncate_history_thinking`), so the cache diverges at the first assistant turn before the last user message;
  within one user turn (the agentic tool loop) the re-render equals the generated bytes when the client returns the
  reasoning verbatim. 25e measures the hit rate on the agentic transcript.
- **Sampling defaults** (Decision 12) and EOS {2, 11}; `stop_strings` read as a string or a list (23e's rule).
- **Context:** `--max-len auto` up to 262144 (Decision 14).

## 8. Structure

- **Descriptor and loader:** `model::BoltDesc` (`src/model/bolt.*`: the per-layer type, E and I tables, held to
  `config.json`'s `llm_config` key by key, refusing another `model_type`, another hidden, a non-NoPE attention, MTP layers,
  `residual_in_fp32` true, `n_group` != 1, a scoring function other than sigmoid), `loader::load_bolt`
  (`src/loader/bolt_{layout.h,repack.*,loader.*}`).
- **Engine:** `runtime::bolt::BoltEngine` (`src/runtime/bolt/`): decode list, prefill walk, snapshots, two cards through
  spec 16b's placement (one card for the synthetic and `--layers N` development checkpoints).
- **Kernels:** `src/kernels/bolt/nb_*.cl`, host half `src/kernels/bolt_kernels.h` (`kernels::bolt`); every shape define in
  the binary name; reused kernels from new CMake lines only.
- **CLI:** `b70-decode` / `b70-serve` dispatch on `model_type nemotron_h_omni` with `llm_config.model_type
  nemotron_h_puzzle` (`src/cli/bolt_{decode,serve}.h`).
- **Server:** `ChatFormat::Kind::Bolt`; the Qwen XML parser reused; the reasoning rules in the output stream.
- **Tools:** `tools/oracle/bolt_ref.py` (+ `bolt_mamba_ops.py`, facts, fixture, make_tiny, prompts),
  `tools/quantize_bolt.sh` + `tools/quantize/bolt/`, `tools/box_validate/bolt_oracle.sh`, `tools/probe/probe_bolt_step.cc`.

## 9. Gates

- **B0, nothing moves.** Every existing kernel binary sha256-identical (G0), `tools/kernel_cmdlines` additions only, no
  existing `.cl` edited (a reused Muse binary is bound by name), every existing suite green; the `quant.cc` relaxations
  leave every existing loader test green and every refusal test still refusing.
- **B1, the reference** (25a). Outside the Mamba mixer the port equals R bitwise (fp32 and bf16) on the tiny variants
  (prompt and cached decode; chunked = whole), the Mamba mixer equals the twin's order bitwise and R's torch fallback in
  fp32 within the bound 25a measures (the clamp aside); one test per trap of §2 (the clamp, the gated norm's groups, the
  conv's bias and SiLU before the rounding, `D` skip, the router's fp32 chain and `+ 1e-20`, the ascending-id combine,
  relu² on bf16, the shared expert on the normed hidden, the latent projections, NoPE, the untied head, the tap
  convention). bf16-mode differences against the remote fallback recorded per op. The real model's perplexity sane on the
  box CPU (g128 dequantised and the BF16 original).
- **B2, kernels.** Every new kernel against its host twin (`tests/kernels/bolt_ref.h`), bitwise where the arithmetic is
  the same; flash against fp64 at cosine >= 0.99999.
- **B3, golden** (greedy, tie-aware, docs/14's undetermined-row rule) against the engine-format reference (SergiioB's
  checkpoint dequantised; later ours): synthetic real-width checkpoints, `--layers N` truncations, then the whole model on
  two cards. Routes as sets per (row, MoE layer) with near-ties within `B70_BOLT_TIE_TOL` (set from 25a's measured gap
  distribution; Decision 11); the SSM state after the prompt to cosine bars (the int4 GEMV's fp32 order and torch's bf16
  linear round differently - 24c's finding).
- **B4, determinism.** Replay bitwise; prefill splits at multiples of 64 bitwise; chunked = whole; prefix-cache restores at
  block ends bitwise the cold run (KV, SSM state, conv rings); `--pp 2` split anywhere bitwise another split.
- **B5, quality.** Gate Q for SergiioB's checkpoint (a recorded measurement, spec 23 §3.3's ruling style): KL / top-1 and
  A4 against the BF16 original. 25q's F6 for ours: the same table plus ours vs SergiioB's - ours replaces his only if it is
  no worse anywhere and better on the agentic sets (Decision 6).
- **Speed (recorded; no bars before P0):** launches (653 / 533), decode at depth 886 .. 262144, prefill at the same
  depths, against §3's roofline and SergiioB's published / measured vLLM rows.

## 10. Stages

| stage | Mac (blind) | box | row |
|---|---|---|---|
| **25a** facts, CPU reference, golden, A4, taps | the port + B1 on the tiny variants, the facts sheet from source and the range-read headers, the A4 set rendered, the golden prompts | the full header / scale scan, perplexity (g128 + BF16), golden sets, the A4 references, gate Q, tap dumps for spec 26 | 57 (CPU) |
| **25q** our AutoRound g64 | the recipe, `--dry-run`, the tiny end to end | calibration generation, AutoRound on both cards (days), F6 | 58 |
| **25b** descriptor, loader, plan | descriptor, loader host half, `make_synth.py`, the plan, the relaxations | loads (synthetic, SergiioB's 41.9 GB on two cards) | 59 |
| **25c** decode | twins, names, Mac-GPU indicative runs | P0 first, every first compile, B2-B4, two cards, speed | 60 |
| **25d** prefill | twins, the walk's launch formula | B2-B4 on prefill, speed | 61 |
| **25e** serving | template / tokenizer / parser fixtures, snapshot layouts | restores, the server, A4 on the card, llama-benchy | 62 |

Order: 25a -> 25b -> 25c -> 25d -> 25e; 25q needs 25a (the reference, the A4 set, the held-out sets) and runs when the
operator schedules the box days; its export replaces SergiioB's in every real-weight gate when it passes (each row
carries an `ours` stage).

## 11. Out of scope

- The RADIO vision tower, the projector, video / image / audio inputs (refused by name, 23e's list).
- MTP (none), tensor parallel (spec 17; §4.3), batching (spec 13), the one-card offload tier (spec 22; Decision 4).
- Third-party formats: NVIDIA's 24GB release (IQ2_XXS / NVFP4 / Q8_0), GGUF, FP8, compressed-tensors exports of Bolt.
- int8 prefill for this family; int8 KV (Decision 17).

## 12. Decisions

The operator approved every proposed decision below as written on 2026-10-20 (Decisions 3 and 20 of spec 25 were
ruled separately the same day).

1. **Decision 1 (approved, operator 2026-10-20) - the reference.** R with its kernel path enabled through plain-torch implementations of the
   five mamba_ssm / causal_conv1d functions (fp32 internals, no dt clamp, the twin's summation order), an fp32 cache;
   bitwise R outside the Mamba mixer, R's torch fallback within a measured fp32 bound inside it; the fallback's bf16
   roundings and its 0.001 clamp recorded, not copied (§2.0). *Cost if wrong:* ~2 days to re-pin 25a / 25c's twins to the
   fallback, and the engine would then diverge from vLLM / training on 25 of 128 heads in layer 0 alone.
2. **Decision 2 (approved, operator 2026-10-20) - SSM state precision: fp32** (the config's `mamba_ssm_cache_dtype`), with bf16 as a measured
   arm (25c: B3 golden and A4 on both; the state traffic 0.336 vs 0.168 GB a token, ~3 % of the step). NVIDIA's recipe
   uses bf16 and the architecture search budgeted 16-bit states. *Cost if wrong:* fp32 wastes ~0.3 ms a token and 80 MiB a
   snapshot if bf16 is as good; bf16 could cost long-context quality if it is not - the arm decides on data.
3. **Decision 3 (operator, 2026-10-20) - the shipping checkpoint until 25q passes: SergiioB's g128 via the exact expansion**
   (the operator: "Ship SergiioB's g128 until our g64 passes 25q"), gate Q recorded, not gating. *Cost if wrong:* none to correctness; if his 128-sample / 200-iter
   calibration on generic data loses on A4, users see it until 25q lands.
4. **Decision 4 (approved, operator 2026-10-20) - placement: two cards (`--pp 2`) by default; one card only through spec 22 (not built here);
   TP (spec 17) is the decode lever after 25e.** *Cost if wrong:* if the operator needs one-card Bolt, spec 22's 22b-22d
   must be adapted (latent experts, heterogeneous E / I) - a separate spec.
5. **Decision 5 (approved, operator 2026-10-20) - the loader relaxations of §5.2** (a restating `+:` rule accepted; signed scales audited;
   BF16 scales exact-or-refused in the GPTQ path). *Cost if wrong:* refusing the `+:` rule blocks SergiioB's checkpoint
   entirely; accepting more than a restatement would re-open the broken-MTP-config hazard - the narrow form avoids both.
6. **Decision 6 (approved, operator 2026-10-20) - when ours replaces SergiioB's:** F6 no worse on every held-out set (KL, top-1) and better or
   equal on A4 agreement with the BF16 original, 0 parse failures beyond the BF16 reference's. *Cost if wrong:* a stricter
   rule keeps a worse quant; a looser one ships a regression on non-coding text.
7. **Decision 7 (approved, operator 2026-10-20) - who generates 25q's calibration answers:** SergiioB's int4 checkpoint on the box's vLLM XPU
   image (TP 2, ~60 t/s published) for the self-generated agentic sessions, plus a public agentic slice rendered through
   Bolt's template (`openbmb/UltraData-SFT-Agent-2609`'s Code-Agent / Tool-Use configs through `AR/` e31b7728's loader -
   licence checked first). The BF16 original cannot run on the cards (139.5 GiB) and is too slow on the box CPU for
   thousands of answers. *Cost if wrong:* int4-generated answers carry the g128 quant's own errors into the calibration
   targets (they are inputs only - AutoRound fits the bf16 block outputs, not the answers - so the effect is distribution
   shift, small); the alternative is days of CPU generation.
8. **Decision 8 (approved, operator 2026-10-20) - 25q's iteration budget:** `iters` 400, `nsamples` 512, `seqlen` 2048. AutoRound's step works
   on a mini-batch (`batch_size` 8) drawn from the cached samples, so the tuning time scales with iters x seqlen and
   `nsamples` adds the per-block input caching: ~2.7x SergiioB's 5 h 57 min at 200 / 128 / 1536, **~16-20 h** on two B70s
   (ESTIMATED; the first block's time x 88 is the honest figure), with 800 iterations as the arm if the 400 run's A4 gain
   over RTN is still rising (~32-40 h). *Cost if wrong:* one to two box days either way; too
   few iterations leaves quality on the table, too many burns days for nothing measurable.
9. **Decision 9 (approved, operator 2026-10-20) - our export's `out_proj` scales F16:** #2512 pins `mixer.out_proj` to bf16 scales in memory;
   the GPTQ export writes F16 (SergiioB's does); 25q's `check.py` asserts F16 everywhere and the loader's exact BF16
   conversion is the fallback. *Cost if wrong:* none (exact either way); a refusal names the element if a bf16 scale is not
   f16-representable.
10. **Decision 10 (approved, operator 2026-10-20) - the MoE combine order:** the remote code's eager path (ascending expert id, fp32 product
    then add, one rounding), as Kolibri's. *Cost if wrong:* an ulp-level difference on rows where vLLM's fused order
    differs; B3 near-ties absorb it.
11. **Decision 11 (approved, operator 2026-10-20) - `B70_BOLT_TIE_TOL`** from 25a's measured 9th / 10th `sel` gap distribution (1e-3 PROPOSED
    until then; the fp32 router chain makes ulp-level gaps rarer than LFM2's bf16 chain). *Cost if wrong:* a loose
    tolerance hides a routing bug; a tight one fails determined rows on ulp noise.
12. **Decision 12 (approved, operator 2026-10-20) - sampling defaults and EOS:** NVIDIA's `generation_config.json` (sampled, T 1.0, top-p 0.95,
    EOS [2, 11]; the card's recommendation), not SergiioB's (EOS 11 only, no sampling keys). *Cost if wrong:* greedy
    users unaffected; sampled agents get the card's recommended distribution.
13. **Decision 13 (approved, operator 2026-10-20) - reasoning in the API:** `chat_template_kwargs` `enable_thinking` (default true),
    `medium_effort`, `truncate_history_thinking`, `force_nonempty_content`; OpenAI `reasoning_effort`: `none` / `minimal`
    -> `enable_thinking: false`, `low` / `medium` -> `medium_effort: true`, `high` -> default; an explicit kwarg wins.
    *Cost if wrong:* a client's effort maps to another template switch than it expects; no correctness effect.
14. **Decision 14 (approved, operator 2026-10-20) - context cap 262144** (`max_position_embeddings`; the card's 256K), `--max-len auto`
    choosing what fits. *Cost if wrong:* none at the cap; above it refused.
15. **Decision 15 (approved, operator 2026-10-20) - attention kernels: reuse Muse's `mg_attn` (decode, with a ones gate buffer) and
    `mg_pf_flash_attn` (prefill) binaries from new CMake lines**, since the shape is identical; a gate-free family copy only
    if the ones buffer measures slower. *Cost if wrong:* 8 KB of reads a layer; or a copy later (one day).
16. **Decision 16 (approved, operator 2026-10-20) - the default decode list:** every fused arm (`router`, `route`, `conv`) that P0 and the
    interleaved A/B show faster and that is bitwise the as-built list; `B70_BOLT_FUSE` keeps the others selectable; the
    gated norm one launch (recompute) or two (P0 decides). *Cost if wrong:* ~1-3 % of the step.
17. **Decision 17 (approved, operator 2026-10-20) - KV bf16 only.** 8 KiB a position: int8 KV would save 1 GB at 262144 for a kernel family
    of its own. *Cost if wrong:* none until batching wants the memory.
18. **Decision 18 (approved, operator 2026-10-20) - int8 head by default** (spec 9), `--lm-head bf16` selectable. *Cost if wrong:* 0.54 GB a
    token less read vs a measured KL; spec 9's evidence on Qwen3.8 applies, 25q's F6 adds Bolt's row.
19. **Decision 19 (approved, operator 2026-10-20) - family copies** (`nb_*`) for every Kolibri- / LFM2-derived kernel; no existing source
    edited. *Cost if wrong:* duplicated code a later refactor can merge.
20. **Decision 20 (operator, 2026-10-20) - the licence:** the weights are published open weights under OpenMDW-1.1; our
    g64 export may be published and model files (config, template, tokenizer) may be vendored; the empty `LICENSE` file is
    expected to be fixed upstream (the operator: "it's published open weights, of course we can use it"). 25q's card links
    the licence and states the derivation.

## 13. 25a as built (2026-10-21; the Mac only - no real weight has been read)

Branch `spec25a-bolt-reference`; box queue row 57. Built: `tools/oracle/bolt_remote.py` (R imported from the pinned
snapshot under a private package, the four text files sha256-checked; the `mamba_ssm` / `causal_conv1d` stub; the
mixer's dispatch patched per instance, `kernel` or R's `torch`), `bolt_mamba_ops.py` (the five training-kernel
functions in plain torch), `bolt_ref.py` (the streamed reference, the restated ops 25b / 25c import, `run` / `ppl` /
`hfcheck` / `taps` / `facts` / `gateq`, `fallback_forward` - R's fallback with its rounding points as switches),
`bolt_make_tiny.py` + `bolt_tiny_check.py`, `bolt_facts.py` (`docs/probe-bolt-2026-10-21.md`), `bolt_prompts.py`,
`bolt_gateq.py`, the A4 set `tests/golden/toolcall-bolt`, `score.py --format qwen-think`, `oracle_generate.py --model
bolt`, `a4_ref.sh bolt`, `tools/box_validate/bolt_oracle.sh`, the template fixtures `tests/server/bolt_template_cases.json`.
Measured on the Mac (`agnes-ref-img`, torch 2.14.1 CPU, transformers 5.19.0): `test_bolt_ref.py` 26 tests,
`test_bolt_make_tiny.py` 12, the toolcall tests, `bolt_tiny_check.py` OK; the probe sheet's tables. Not run (the
weights were not downloaded): every real-weight stage of row 57 - perplexity, `hfcheck` on real weights, the golden
sets, the A4 references, the tap dumps, gate Q; so the 9th / 10th `sel` gap distributions (B70_BOLT_TIE_TOL,
Decision 11), the real SSM-state magnitudes and the per-prompt times are NOT measured yet (the DRY_RUN times are
ESTIMATED).

**The ops' orders (the twin's, 25c generates its host twin from them; each tested one scalar op at a time):** the
conv seeds the fp32 accumulator with the bias, then adds the taps oldest first (§2.2 wrote `+ b` after the chain;
causal_conv1d's and vLLM's kernels seed with the bias - bf16 x bf16 products are exact, so only the add order
matters), `silu(v) = v / (1 + exp(-v))`, one rounding; the SSM update is mamba_ssm's - `dB[n] = B[n] dt`, then
`S = S dA + dB x` (§2.2 wrote `(dt x) B`), products and adds separately rounded (no fma: the kernels' house rule is
FP_CONTRACT OFF); `y = tree_n(S C) + x D` with the halving pairwise tree over the 128 state entries (`v[:n/2] +
v[n/2:]` repeated); the gate is `z sigmoid(z)` (layernorm_gated's form), the group mean the same halving tree / 1024,
`rstd = 1 / sqrt(ms + eps)`, `(v rstd) w`, one rounding; the chunk scan is the SSD form over 128-row sub-chunks from
the call's first row, the cumsum sequential, exp(segsum) masked before the exp. Transcendentals (exp, log1p) are
torch's; the engine's differ by ulps, so B2 is bitwise only where the arithmetic is the same (as for the GDN).

**Rulings (each with its cost if wrong):**
1. *R's kernel path has no prompt continuation* (its cached branch reads one row, R:397-439): a chunk after a cached
   prefix runs `bolt_remote.continue_forward` - the same five functions, the conv seeded from the cache's raw
   columns, the scan from its state. *Cost if wrong:* none to the engine (its prefill is 25d's); the golden sets past
   4096 ids (chunks of 2048) would carry our continuation's arithmetic, which `test_chunked_equals_whole` bounds.
2. *The streamed MoE skips R's dummy forward of unrouted experts* (R:941-945 adds exactly +0; tested against R's own
   `moe`). *Cost if wrong:* none measurable (bitwise on the tiny model); without it every decode step would run up to
   487 extra expert GEMVs.
3. *The router's `e_score_correction_bias` stays fp32 in a bf16 model* (5.19.0's `from_pretrained` keeps it fp32 too).
   *Cost if wrong:* none (the checkpoint's F32 values, the remote code's own intent R:1054-1063).
4. *A4 renders Bolt with `enable_thinking` true and `medium_effort` true*, key-sorted (b70-serve's order): the
   agentic mode, the effort that keeps the reasoning inside 2048 new ids (K2's / Muse's precedent). *Cost if wrong:*
   A4 measures the medium-effort mode; a default-effort set is one `make_set.py --kwargs '{}'` away.
5. *EOS {2, 11} for both checkpoints* in the references (Decision 12; SergiioB's generation config lists 11 only).
   *Cost if wrong:* a reply ending in `</s>` would continue past it in a g128-only run - not seen on the tiny model.
6. *A4 outputs are tagged by form* (`<name>.int4.*` / `<name>.bf16.*`) in one directory; gate Q compares them.
   *Cost if wrong:* none (names only).
7. *The tiny variants use vocab 131072* (the real tokenizer's ids are valid; plan Task 1's in-code config keeps 512),
   and `int4-g128-gptq` sits on a sibling config with every K a multiple of 128 (latent 128, I 128 / 256 / 384).
   *Cost if wrong:* none (test fixtures).
8. *Tap dumps start at `n_prompt - 1024`* (LiLiCorr's window, spec 26). *Cost if wrong:* 26a re-dumps with `--window`.
9. *`bolt_mamba_ops.py` landed in Task 1's commit* (the kernel-path check needs it); Task 2 added its tests.

**Facts that differ from this spec (corrections; the body above is left as written):**
- **R's torch fallback is wrong past one SSD chunk:** R:692-697 sums the state passing over the TARGET chunk (dim 2);
  transformers' native class sums over the source (N:328-330). On a 300-row prefill R's state is 0.23 relative off
  the recurrence, 9.7e-7 with the index fixed - so "what transformers runs on a CPU" is not just differently rounded
  (§2.0); Decision 1's reference is unaffected.
- **torch's CPU fp32 GEMM is row-count dependent, its bf16 GEMM is not** (§2.4's "torch's CPU sgemm order"): R's
  router logits change with the rows a forward carries (every row at M = 1 / 2 / 9 / 64 / 128 of a 512-row batch at
  [496][4096]); route SETS did not flip on the tiny model, `w` did by an ulp - B3 compares `w` with an ulp tolerance.
- **The router's sum of the nine s has no fixed order we could name** (torch.sum matches sequential 60 %, the best
  order-preserving tree 60 %, the best sampled permutation 90 %): §6.1's "Σ ... an ulp-level choice 25a pins against
  torch's sum" cannot pin it; 25c's twin keeps a fixed order of its own.
- **`dynamic` has 44 `-:` exclusions** (§1.3: 43) and the one restating `+:` rule.
- **`e_score_correction_bias` reaches 48.0** (layer 1 is 4.47 .. 4.55, others sit near 10-48, each in a narrow band);
  the 9th / 10th largest bias of layer 12 are equal.
- **`A_log` spans -6.81 .. 8.81 and `dt_bias` -11.69 .. 8.50 over all layers** (§1.3 quoted layer 0's); the dt-clamp
  census is **212 of 5,120 heads** (layer 0: 25 of 128 as §2.0 says; layers 2 / 57: 23 / 25).
- **The BF16 original stores `A_log` / `D` / `dt_bias` as BF16** (SergiioB F32 holding bf16 values) and names its trunk
  `language_model.backbone.*` (measured from its index); its headers give 74,042,720,768 parameters - §3's derived
  total, now measured. Active per token 9,052,884,992 with `norm_f` (§3: 9,052,880,896 without).
- **The template renders Python's spelling for non-JSON scalars:** a history argument `true` -> `True`, `null` ->
  `None`, a tool schema's `additionalProperties: false` -> `<additionalProperties>False</additionalProperties>` (§7:
  25e must reproduce it).
- **torch's `topk(sorted=False)` keeps either id of an exact tie** (28 / 22 of 50 at 240, 19 / 31 at 496).
- **The bf16 difference table** (`docs/probe-bolt-2026-10-21.md` §6.2): beyond §2.0's list, softplus itself rounds in
  bf16, and the fallback's prefill keeps y fp32 into the gated norm (the kernel path rounds y once); with every point
  switched to the kernel path's semantics the fallback equals the port (fp32 cache: 0 difference on the outputs).

## 14. 25b as built (2026-10-22; the Mac only - no card, no real weight)

Branch `spec25b-bolt-loader`; box queue row 59. Built: `model::BoltDesc` (`src/model/bolt.*`: the per-block types, E and I per
layer, per-linear `{S, layout}`, `BoltPlacement`), `loader::load_bolt` (`src/loader/bolt_{layout.h,repack.*,loader.*}`, one
device layout from SergiioB's g128 or our g64), `src/common/exp_torch.h`, the two exact relaxations in `src/loader/quant.{h,cc}`
(§5.2 items 1 and 3; item 2's audit), `runtime::bolt::plan` (`src/runtime/bolt/`), the synthetic real-width checkpoints
(`tools/quantize/bolt/{make_synth,check,test_bolt_synth}.py`, `tests/model/bolt/{config.json,make_config.py}`,
`tools/box_validate/bolt_synth.sh`), the tests `bolt_test`, `quant_test` (extended), `bolt_repack_test`, `bolt_names_test`,
`bolt_synth_host_{g128,pair,g64}_test`, `bolt_plan_test` and the card test `bolt_load_checkpoint_test`; docs/13's "Nemotron 3.5
Bolt" section. **Measured on the Mac (all host tests; the card test only compiles - Level Zero syntax):**
- the descriptor's names equal the published index's 112,091 language names exactly (+ 849 vision / projector tensors skipped by
  prefix) - the typed pattern and the typed (E, I) table ARE the real ones (`bolt_names_test`); `bolt_test` holds
  `tests/model/bolt/config.json` (the published one, vendored, Decision 20) to the descriptor and refuses 46 edits by key name;
  `make_config.py --check` holds the code-typed config to the published file key by key (69 `llm_config` keys, 88 blocks, 44 `-:`
  rules + 1 `+:`).
- the synthetic checkpoints (random weights at the real widths, signed scales - 50.0 % negative): N 8 = 3.93 GB (g128), 3.99 GB
  (g64, exact expansion or native), N 12 = 5.44 / 5.55 GB (§12's ESTIMATE ~5.5 GB); 84 s and 191 s on the Mac; `check.py`
  ACCEPTED on all; `bolt_ref.build_streamed(weights="int4")` loads the g128 form and its exact g64 expansion and their logits on 16
  ids are **bitwise equal**; the host half (`bolt_synth_host_*`) repacks every block of N 8, 20,400 sampled words equal the shards',
  both spellings byte-identical, 0 unconsumed.
- the bytes of §3, re-derived from `loader/bolt_layout.h` and printed: routed experts 35,139,092,480 B (13,824 experts), Mamba
  2,328,494,080, attention 151,519,232, MoE dense 1,114,112,000, routers 113,246,208, small blocks 11,061,248, embeddings
  1,073,741,824, int8 head 537,395,200 B, total 40.469 GB; the int4 linears hold 72,909,586,432 weights (25a's header scan).
  Split 48 (pp_balance, derived): 19,928,067,776 / 20,540,610,880 B; 262144 positions fit on two 32 GiB cards with the scratch, the
  drafter, its replica and 7 verify slots reserved (device 1: 27.23 GB of 34.36 GB); one card is refused naming Decision 4; a
  24 GB pair reaches `max_len` 90368 with the scratch reserved (ESTIMATED scratch).

**Facts that differ from the plan / spec (the body above is left as written; each with its cost if wrong):**
1. *`exp_torch` is not in `common/`* (it is the kernels' and `tests/kernels/k2_ref.h`'s Sleef softmax exp, and `torch.exp` on an x86
   fp32 tensor goes to MKL's `vsExp`, which has no host reproduction): `src/common/exp_torch.h` copies the function and the loader's
   `A = -exp(A_log)` uses it. Measured against `torch.exp` over all 33,266 bf16 values in [-7, 9]: 196 differ, never by more than 1
   ulp (libm's `expf`: 192; reproduce with `tools/oracle/bolt_exp_dump.cc` + `bolt_exp_compare.py`). Consequence for 25c: `A` can
   differ by 1 ulp between the engine and `bolt_ref`, so a Mamba twin held bitwise against `bolt_ref` must take `A` from the engine's
   arithmetic - 25c adds a `bolt_ref` switch computing `A` with a float32 port of `exp_torch` (recorded in plan 25c's dependencies;
   not built here). *Cost if wrong:* a 1 ulp difference in `A` before `exp(dt x A)`; 25c's B3 absorbs it (a host function
   that matches MKL does not exist).
2. *`BoltLinear` is richer than `{K, N, S, layout}`* (id, `GemvShape`, kind, fuse, parts, part_n - 24b's `LfmLinear`), and the
   MoE list carries the Router and one expert's up / down shapes; the routed experts are ONE allocation per block (E blocks of up ||
   down, plan: `base + id x block(I)`), not 24b's two arrays. *Cost if wrong:* 25c's expert kernels bind one buffer and two offsets.
3. *`load_bolt` takes `l0::Context*` (the repo's device type, not `l0::Device*`), a `BoltLoadOptions` (int8 head, `max_len`,
   `layers`) and a `BoltPlacement`*, not a `BoltPlan placement` (the plan is computed from them before the first allocation);
   `BoltDesc::int8_head` is a load / plan option, not a descriptor field (Decision 18); `plan()` gained `int8_head`;
   `BoltReserve::verify_slots` is a COUNT of extra SSM slots (spec 19c / 26: 7), the other fields bytes. *Cost if wrong:* a signature.
4. *The `+:` rule must carry exactly {bits, group_size, sym}* - an empty `{}` rule, the pre-25b refusal's own test, stays refused
   (the plan's "compares the rule's three values" read as: all three present). *Cost if wrong:* a checkpoint whose restating rule
   omits a key would be refused naming the rule; none seen.
5. *`A_log` / `D` / `dt_bias` are accepted F32 or BF16* (SergiioB's F32-holding-bf16 and NVIDIA's BF16), `e_score_correction_bias`
   F32 only; `time_step_limit` must be [0, inf] (the decode chain has no clamp - §2.0); the first attention block need not be in
   the first N blocks (the engine has no R-cache). *Cost if wrong:* none (stricter than needed only on the bias).
6. *`cmake/host_only.cmake` is not edited* (the plan lists it): it derives the card tests (`checkpoint` label + `b70_l0`) itself.
   *`tools/box_validate/bolt_synth.sh`, stages `r59.host` / `r59.pytest` / `r59.synth_host` and data keys `bolt_synth` / `bolt_rtn`
   are beyond the plan's four stages* (the synthetics need a runner; the host tests need their own stage). `tools/quantize/bolt/
   check.py` is self-contained (25q merges its own). `model_extra_tensors.safetensors` holds 3 stand-in vision tensors, one more
   sits in a language shard. *Cost if wrong:* none.
7. *`common::f32_to_f16` flushes f16 subnormals* (a test helper): the loader's exact BF16 conversion uses `common::kv8::
   f32_to_f16_rne`; the quant tests were first written with the wrong helper and caught it. *Cost if wrong:* none now.

**Not run (box, row 59):** every card stage - the synthetics on one and two cards (plan == allocation, read-backs), SergiioB's
41.9 GB checkpoint on two cards (split 48, 262144 positions; load time, RSS, the int8 head's quantisation time), our g64 export
(25q); the plan's memory figures against the cards' real `memory_bytes` (the Mac assumed 32 GiB); every layout / `S` choice is still
PROVISIONAL until 25c's P0 sweep.
