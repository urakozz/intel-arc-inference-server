# Spec 26 - LiLiCorr: NVIDIA's DFlash-family drafter for Nemotron 3.5 Bolt

> DROPPED (operator, 2026-10-27): support removed from the engine in 0219cc2 (26c) and 99adbb0 (26a / 26b); kept for the record.
>
> Reusable later (revive from git: `git show c57e08e:<path>` - c57e08e, this branch's parent on main, is the last
> commit that has all of it):
> - sink attention: `src/kernels/dflash/ll_attn.cl` (the learned sink logit added in the softmax reduce); the top-k with the logsumexp `src/kernels/dflash/ll_topk.cl`; the bf16 GEMV with an fp32 output scale `src/kernels/dflash/ll_gemv_bf16_g.cl`; host names in `src/kernels/dflash_kernels.h`
> - the NVFP4 decoder: `src/loader/nvfp4.h`, `src/loader/nvfp4.cc`, `tests/loader/nvfp4_test.cc`; NVFP4 into our formats `src/loader/lilicorr_repack.h`, `src/loader/lilicorr_repack.cc`, `tests/loader/lilicorr_repack_test.cc`
> - twins and tests: `tests/kernels/lilicorr_ref.h`, `tests/kernels/lilicorr_ref_test.cc`, `tests/kernels/lilicorr_kernels_test.cc`; the oracle's LiLiCorr form (sinks) in `tools/oracle/dflash_ref.py`
> - 26c: the union-aware M-row MoE `src/kernels/bolt/nb_moe_union.cl` (+ `tests/kernels/bolt_moe_union_test.cc`), Bolt's verify lists and Mamba state slots in `src/runtime/bolt/`, the serving glue `src/cli/bolt_spec.h`

**Status:** design, 2026-10-20, approved by the operator 2026-10-20. Decisions are marked **Decision N** with their
cost if wrong and collected in §10. **26a built on the Mac 2026-10-23** (§12 as built; §11 the P0 verdict - PENDING the
box, its defaults PROVISIONAL); **26b built on the Mac 2026-10-23** (§13 as built: the drafter on the card - nothing
run on a card yet, row 64); **26c built on the Mac 2026-10-23** (§14 as built: Bolt's verify lists, the slots, the union
MoE, the prefill tail, snapshots and `b70-serve --spec dflash|lookup` - nothing run on a card yet, row 65; `--spec dflash` is
NOT Bolt's default until D5).
**Plans (2026-10-20):** `docs/superpowers/plans/2026-10-20-spec26{a,b,c}-*.md` - 26a P0 (the reference, teacher-forced
acceptance on our target, the verify cost, the projection), 26b the drafter on the card (D1), 26c the engine loop and
serving (D0, D2-D5). **Box queue rows 63-65** (spec 25 takes 57-62).

**What it extends.** Spec 19's DFlash design and its as-built stages - 19b (`model::DFlashDesc`, `loader::load_drafter`,
the `df_*` kernels, the draft and projection lists), 19c (`runtime::TapDrafter`, the tap verify list, the per-row
recurrent-state slots and the commit that moves the live slot, host acceptance with a sparse q), 19d (the ring in every
snapshot, `store_root(kv_form, drafter_key)`, `--spec dflash --pp 2` with the taps crossing on the copy hand-off) - and
spec 23f (a second drafter form beside vLLM's, `DFlashForm`, int4 g64 drafter weights at load, the "built regardless of
P0" ruling). Spec 25 is the target.

**The operator's standing rulings this spec follows:** DFlash is built whatever P0 shows ("in vLLM DFlash is a monster
of performance", 2026-10-15) - **P0 sets the defaults (K, format, q) and the expectation; it is not a go / no-go gate**;
our formats only (the NVFP4 checkpoint is converted at load - §3); lossless sampled acceptance as 19c; one implementing
agent per plan.

Every number is **measured** unless marked **derived** (from the config and the safetensors header, range-read
2026-10-20; the 962 MB weight file was not downloaded), **estimated**, **proposed** or **published**.

**Citations.** `V/` = vLLM at `3901d56800` (the operator's checkout, branch `b70-xpu-stack`, Apache-2.0; **ideas and
semantics only, no code copied**): `V/models/lilicorr.py` (the LiLiCorr head, "adapted from sgl-project/sglang PR
#37462"), `V/models/qwen3_dflash.py`, `V/models/qwen3_dflash2.py` (the backbone and its grouped convolution),
`V/spec_decode/lilicorr/speculator.py`, `V/spec_decode/dflash2/speculator.py` (the shared candidate walk),
`V/spec_decode/dflash/speculator.py`, `V/models/nemotron_h.py` (the tap capture). Upstream commits: `3dc6def360`
(varlen Mamba-2 decode for verification), `a608b6e6e0` (the drafter's SWA tail at a shared-prefix junction). The paper is
"LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding", arXiv 2608.20530 (cited by
the card; not read for this spec - vLLM's module is the operative definition).

## 1. The checkpoint

`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` @ `449d2414` (one `model.safetensors`, 962,485,872 B; OpenMDW-1.1;
"1.113B draft parameters, excluding shared target embeddings and the LM head; the two-layer LiLiCorr head accounts for
37M of them", published).

| | |
|---|---|
| architecture | `LiLiCorrDraftModel`, `model_type: qwen3`; a DFlash 2 backbone (vLLM's `DFlash2Qwen3DecoderLayer` because `conv_kernel_size` is set, `V/models/lilicorr.py:443-450`) plus the LiLiCorr head |
| block | `block_size` 8: row 0 the anchor, rows 1-7 `mask_token_id` **990** (`<SPECIAL_990>`, a real vocabulary row of Bolt's tokenizer); vLLM runs `num_speculative_tokens` 7 (`1 <= K < block`, `V/models/lilicorr.py:438-442`) |
| taps | `target_layer_ids` [1, 18, 35, 51, 68, 85]: the residual **after** those Bolt blocks (spec 25 §2.5) - 1 and 85 are MoE blocks, 18 / 35 / 51 / 68 Mamba blocks; at spec 25's split (48) three taps live on each card |
| projector | `fc` 24576 -> 4096 (6 taps x 4096), `hidden_norm` (RMSNorm 4096); `projector_type: "lilicorr"` names no other code path in vLLM (grep: unused) |
| layers | 5, hidden 4096, 32 q / **2** kv heads x 128 (GQA 16), `q_norm` / `k_norm`, RoPE θ 10000 (neox, default), **all sliding, window 1024** (`use_swa`, `swa_window_size`), **non-causal** (`causal: false`), **attention sinks** (`attention_sink_bias`: a learned per-head logit in the softmax denominator, `layers.N.self_attn.attention_sink_bias` BF16 [32]), MLP SwiGLU 12288, RMSNorm eps 1e-6 |
| convolutions | DFlash 2's two grouped convolutions per layer (`attention_conv`, `mlp_conv`): `conv_kernel_size` 2 taps, `conv_group_size` 16 (256 groups), `kernel_projection` [1024][4096] (2 sides x 2 taps x 256), `base_kernel` [2][2][4096] - the shapes of Qwen3.8's DFlash 2 drafter that 19b built |
| head | the **target's** `lm_head` (no `has_own_lm_head`), the **target's** embedding |
| LiLiCorr head | `lilicorr_candidate_topk` 8, hidden 1024, 2 layers, 8 heads (x 128), `mlp_ratio` 2.0 (1024 -> 2048 -> 1024, SiLU, with bias, non-gated), `factor_dim` 1024, `vector_eps` 1e-4, `logit_scale` 8.0 (§2.3) |
| quantisation | ModelOpt **NVFP4 (W4A16)**, group 16: `weight` U8 [N][K/2] (two e2m1 a byte), `weight_scale` F8_E4M3 [N][K/16], `weight_scale_2` F32 scalar (the global scale) on `fc`, every layer's `mlp.{gate,up,down}_proj`, and the head's `token_proj`, `pass_hidden_proj`, `context_proj`, `layers.N.attn.out_proj`, `layers.N.mlp.{up,down}_proj`; **BF16** everything else (attention q/k/v/o and their norms, the convolutions, the sinks, the norms, `anchor_out_head`, `factor_input_proj`, `feature_mlp`, `in_head`, `out_head`, the lattice attention's `in_proj_weight` / `in_proj_bias`, the slot / rank embeddings, the slot biases) - the header lists 179 tensors: 25 NVFP4 linears (U8 + F8_E4M3 + F32 each) and 104 BF16 (§3) |
| target it was trained for | **the 24GB release** (2-bit IQ2_XXS experts, NVFP4 linears; spec 25 §1.4), at temperature 1.0 on Bolt's own responses (published) - not the BF16 original and not our int4 (Decision 9) |
| published acceptance | SPEED-Bench, draft length 7, T 1.0, top-p 0.95, against the 24GB target: coding **5.26**, math 4.59, RAG 5.13, multilingual 5.06, reasoning 4.47, writing 3.41, overall 4.30 tokens per verify (published) |

**Parameters (derived from the header):** `fc` 100,663,296; per layer 35,651,584 (attention) + 150,994,944 (MLP) +
8,388,608 (convolution projections) + small = ~195.07 M, x 5 = 975.3 M; the LiLiCorr head ~36.7 M; **~1,112.7 M** in all
(the card's 1.113 B).

## 2. Reference semantics (vLLM's module; there is no transformers implementation)

### 2.1 The backbone: DFlash 2 (spec 19 §2) with sinks and a 1024 window

Exactly `tools/oracle/dflash_ref.py`'s vLLM DFlash 2 path, which already implements every part LiLiCorr's backbone uses
(it was written for vLLM's `qwen3_dflash*.py`): the context (`c_j = hidden_norm(fc(cat(taps_j)))`, per layer
`k = RoPE(k_norm(k_proj(c)))`, `v = v_proj(c)`), the block (`[anchor, mask x 7]` through the target's embedding,
`input_embedding_scale` 1.0), pre-norm layers with the grouped two-tap convolution around attention and the MLP (one row
of look-back inside the block), **non-causal** attention with the window rule `|q - k| <= window - 1` (1023 here; the
vLLM reading `dflash_ref.py` already uses), the final norm, the target head on rows 1..7. Sinks are already in
`dflash_ref.py` (`_attention`: `softmax(cat([scores, sink]))[..., :-1]`, the sink a per-head logit not scaled by
`head_dim^-0.5`). The output rows that feed the head and the LiLiCorr head are the final-normed rows
(`V/models/qwen3_dflash.py:726-745`, "pass_hidden").

### 2.2 Candidates

`V/models/lilicorr.py:478-486`: per mask row, the target head's logits (raw: no `output_multiplier`, no softcap), the
**top 8** ids and their **log-probabilities** - `log_softmax` over the **whole** 131072-row vocabulary, so a full
logsumexp per row. Order: by logit descending (vLLM's `topk`; the engine breaks exact ties to the lower id, df_topk's
comparator - the reference records whether torch's CPU `topk` agrees on tied rows, 24a's lesson).

### 2.3 The LiLiCorr head (`V/models/lilicorr.py:125-425`), per draft block, B = 1, S = 7 slots, k = 8

```
tok   = token_proj(E[cand])                [7][8][1024]   E = the target's embedding rows of the 56 candidates
pass  = pass_hidden_proj(row_s)            [7][1][1024]   row_s = the final-normed backbone row of mask row s + 1
f     = [lp, exp(lp), lp - max_k lp, k / 7, k == 0]  [7][8][5]  lp = the candidates' log-probs (fp32)
h     = tok + pass + feature_mlp(LayerNorm_5(f)) + slot_emb[s] + rank_emb[k]          -> 56 lattice tokens
a     = context_proj(anchor) · anchor_valid                                            [1024]
2 x:  h = h + attn(RMSNorm(h), bias);  h = h + mlp(RMSNorm(h))                         full attention over the 56
          attn: q, k, v = split(in_proj_weight · x + in_proj_bias), 8 heads x 128, SDPA (scale 128^-0.5) with the
          additive bias B[head][i][j] = relative_slot_bias[head][slot(i) - slot(j) + 7] + same_slot_bias[head] ·
          [slot(i) == slot(j)]; out_proj
          mlp: down(silu(up(x))), both with bias
h     = RMSNorm_out(h);  a = RMSNorm_anchor(a)
pre   = W_self h + W_anchor a + W_cross (h ⊙ a) + b      factor_input_proj [1024][3072] split in three
fh    = silu(pre)
out_v, in_v = normalize(out_head(fh)), normalize(in_head(fh))     L2, eps 1e-4
a_out = normalize(anchor_out_head(a))
start[j]       = 8 · (a_out · in_v[0][j])                                          anchor -> slot 0
pair[s][i][j]  = 8 · (out_v[s][i] · in_v[s + 1][j]),  s = 0..5                     slot s candidate i -> slot s+1 candidate j
lattice        = [start broadcast over i] ++ pair          [7][8 (previous)][8 (next)], fp32
```
**The anchor** (`V/spec_decode/lilicorr/speculator.py:71-87`): `hidden_norm(fc(taps))` of **the newest committed
position** (the last accepted verify row), i.e. the context row the projection just computed for position p - 1 when
the block's anchor token sits at p; `anchor_valid` is false only for a request that verified nothing this step (never in
our one-sequence loop after the first prefill). The candidates' **log-probs enter only as features**: the walk's scores
are the lattice alone.

### 2.4 The walk and q (`V/spec_decode/dflash2/speculator.py:15-75`)

`previous = 0`; for slot s = 0..K-1: `scores = lattice[s][previous]` (8 values), the pick is argmax at T = 0 (ties: the
lower candidate index) or Gumbel-argmax at T (vLLM's murmur3 keying, `dflash_ref.py` / `df_select.cl` already implement
it), `previous = index`, draft s = `cand[s][index]`. The proposal distribution of slot s is `softmax(scores / T)` over the
8 candidates and 0 elsewhere (`_cache_draft_logits_kernel`: the realized scores at the candidate ids, -inf elsewhere) -
**19c's sparse q, unchanged**: `accept_sparse` is lossless for any q with this support.

## 3. The weights: NVFP4 -> our formats (exact where it can be)

An NVFP4 weight is `W = g · s_b · e` with `e` an e2m1 code (values 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6: at most 2
significant bits), `s_b` the F8_E4M3 scale of its 16-element block (at most 4 significant bits, |s_b| in [2^-9, 448]) and
`g` the tensor's fp32 global scale.
- **`W' = s_b · e` is exact in bf16 (and in f16):** at most 6 significant bits (bf16 keeps 8), magnitude in [2^-10, 2688]
  - inside bf16's (and f16's normal) range. 26a's test enumerates all 16 x 256 (code, scale) pairs (the two e4m3 NaN codes refused).
- **`W = g · W'` is not exact in bf16** - a 24-bit g times a 6-bit W' has up to 30 significant bits - **nor, in general,
  in fp32.** What is exact is the factorisation: `y = g · (W' x)` with `W'` stored bf16 and `g` applied to the GEMV's
  fp32 output (one fp32 multiply - how an NVFP4 GEMM applies the global scale, as `alpha` on the accumulator).
- **Our drafter formats** (19b / 23f's, Decision 2):
  - **exact**: W' bf16 in `gemv_bf16`'s tiles + g per tensor, applied by a family copy of the bf16 GEMV with an fp32 output
    scale (`ll_gemv_bf16_g`; the BF16 tensors have g = 1 and use the shared kernel) - 2.23 GB;
  - **int8 per row (19b's)**: from W' (exactly), `s'_r = amax_r(W') / 127`, `q = rint(W' / s'_r)` - independent of g -
    and the stored fp32 row scale `fp32(g · s'_r)`: the only g-dependent rounding is that one fp32 product - 1.11 GB;
  - **int4 g64 RTN (23f's `rtn_int4_g64`)**: from `g · W'` in fp64, `src/loader/rtn.h`'s rule (a requantisation of a
    4-bit format into another: the group shapes differ, 16 vs 64) - 0.59 GB.
- **The LiLiCorr head stays exact** (W' bf16 + g; 37 M parameters, 74 MB): it scores the walk, so its error moves
  acceptance directly, and it is small (Decision 3).

## 4. What it costs (derived / estimated)

| | exact (bf16 W' + g) | int8 per row | int4 g64 |
|---|---:|---:|---:|
| backbone + `fc` (1.076 B parameters) | 2.152 GB | 1.077 GB | 0.572 GB |
| LiLiCorr head (exact in every arm) | 0.074 GB | 0.074 GB | 0.074 GB |
| ring: 5 layers x 2 kv x 128 x {K, V} x bf16 = 5,120 B a position, **1024 positions** | 5.2 MB | | |
| taps of a verify row (6 x 4096 bf16) | 48 KB | | |
| a draft block: drafter once at M = 8 + the int8 target head at 7 rows (0.54 GB) + 56 embedding rows | 2.77 GB | 1.69 GB | 1.19 GB |
| **one block in plain steps** (Bolt's 5.48 GB a step, spec 25 §3) | **0.50** | **0.31** | **0.22** |

**The verify** (M = K + 1 <= 8 rows through Bolt) reads the non-expert weights once (4.25 GB), the experts of the
**union** of the M rows' routes, the SSM state once, and writes the state per row under 19c's slot scheme. Without a
union-aware MoE kernel the M rows read 8 x 9 expert blocks a layer - 7.2 GB at M = 8, more than everything else (Decision
10). With it, at an average union of u experts a layer the routed bytes are ~40 x u x 2.45 MB: u = 30 gives 2.9 GB and a
verify of ~1.4-1.6 plain steps (estimated; 26a measures u from 25a's recorded routes).
**Projection (estimated, published acceptance):** coding 5.26 tokens a verify / (~1.5 + 0.31) = **~2.9x plain** with the
int8 drafter; prose-like 3.4 / 1.8 = ~1.9x. 26a replaces every term with ours.

## 5. Design (on 19b-19d's mechanisms)

### 5.1 The descriptor and the loader (26b)

`model::DFlashDesc` gains **`DFlashForm::LiLiCorr`**: the vLLM form's fields (window 1024, non-causal, conv 2 / 16, v2)
plus `sinks` (true), `cand_topk` 8, `lil_hidden` 1024, `lil_layers` 2, `lil_heads` 8, `lil_mlp` 2048, `factor_dim` 1024,
`vector_eps` 1e-4, `logit_scale` 8.0; `dflash_desc_from_config` accepts `LiLiCorrDraftModel` with
`lilicorr_enabled` true, refusing by name `has_own_lm_head`, a draft vocabulary, `candidate_topk` not a power of two <=
16, a block other than 8, a window other than 1024, causal layers, any `quantization_config` but this one's NVFP4 form
(`quant_method: modelopt_fp4`, `quant_algo: W4A16_NVFP4`, `group_size` 16 - our loader converts it, §3; nothing else
NVFP4 is ever accepted). `check_dflash_target` against Bolt: hidden 4096, vocab 131072, every tap < 88, the mask id a row.
19b's refusal of a sink bias and of a quantization_config are lifted **for this form only**.
`loader::load_drafter`: the NVFP4 decode (W', g) and the three formats (§3); the LiLiCorr head always exact; norms, sinks,
conv bases fp32; the biases of the head's projections fp32.

### 5.2 Kernels (26b) - family copies in `src/kernels/dflash/` (`ll_*`), no existing source edited

- `ll_attn` = `df_attn.cl` with the sink term in the reduce (`l += exp(sink_h - m)` after the blocks and the block keys
  merge, the reference's softmax-with-sink; the numerator unchanged) at NH 32, NKV 2, HD 128, **WIN 1024, RING 1024**, PB
  64 (16 context blocks).
- `ll_topk` = `df_topk.cl` at TOPK 8 plus the row's logsumexp: stage 1 also writes each chunk's (max, Σ exp(v - max)),
  stage 2 merges them in chunk order (fixed) and writes `lp = v - lse` for the 8 candidates. N = USED = 131072.
- The LiLiCorr head (`ll_lattice.cl`, plain OpenCL C, runs on the Mac): the 56-row projections (`token_proj` over the 56
  gathered embedding rows, `in_proj` / `out_proj` / the MLP at 56 rows, `factor_input_proj`, `out_head` / `in_head`) as a
  small-M GEMM family kernel (56 rows x K <= 4096 x N <= 3072: 0.5 GFLOP a block; weight bytes 74 MB dominate),
  `pass_hidden_proj` (7 rows) and `context_proj` / `anchor_out_head` (1 row) through it too; the 56 x 56 x 8-head attention
  with the slot bias in one work-group per head; the norms, LayerNorm(5), the feature MLP, the L2 normalisations, the
  dot products. Every reduction a fixed tree. Estimated ~0.1-0.2 ms a block (26a's probe times it).
- `ll_select` = `df_select.cl`'s walk with the edge term replaced by a lattice lookup (`scores = lattice[s][previous]`),
  greedy or Gumbel, the readback (ids, chosen indices, the 8 candidates and their realized scores per slot) and
  `ctrl.cur_token[1 + s]` for the verify.
- Reused from 19b at LiLiCorr's defines (new CMake lines): `df_prep` (embed, norms, the grouped conv prepare / finish,
  SiLU x up, q / k prep with RoPE θ 1e4, the ring store), the GEMVs (`gemv_i8w` / `gemv.cl` int4 g64 / `gemv_bf16`) at
  M = 8.

### 5.3 The lists

- **Draft list** (one capture; K and T from the drafter's control block, rows past K computed and discarded - the
  backbone is non-causal over the whole block, so the block keeps its 8 rows whatever K, 23f's finding): embed
  `[anchor, mask x 7]` -> 5 layers -> final norm -> target head on rows 1..7 -> `ll_topk` -> the LiLiCorr head ->
  `ll_select`.
- **Projection list** (19b's, `proj_n <= 8` rows at `proj_pos`): fc -> hidden_norm -> **the anchor buffer gets the last
  row's normed vector** (new: 4096 fp32) -> per layer k / v -> k_norm -> RoPE -> ring slot `pos % 1024`.
- **Verify** (26c): spec 25c's decode list at M = K + 1 <= 8 (25c's kernels take M <= 8 as a define and its twins test
  M = 1..4 in one launch; 26c compiles M = 2..8 from new CMake lines), writing the six taps of every row (19c's
  `build_verify_taps`: after block i's output is folded, i.e. inside the next block's norm, the row's residual is copied
  out).
- **Mamba state for verify (Decision 4):** 19c's mechanism - `nb_ssm`'s slot build writes the state after row m into slot
  `(live + m) % 8`, the conv ring keeps raw xBC at `(pos + m) % 16`; commit moves `Control.ssm_live = (live + j) % 8` on
  both cards; rejected rows' slots and ring columns are rewritten before anything reads them. Cost: 7 extra state copies
  (1.17 GB fp32 across the two cards) and M x 168 MB written per verify (1.34 GB at M = 8, ~2.3 ms at 590 GB/s). The lever
  (recorded, built only on P0's numbers): the verify keeps the state read-only (each `nb_ssm` work-group walks its M rows
  in registers and emits y without writing S), and the commit replays rows 0..j over the live state with the same fp32
  code - bitwise the in-line recurrence by construction, 0.34 GB of traffic instead of 1.34 GB, no slots. vLLM's
  `3dc6def360` takes the conv half of this route (a longer conv state indexed by `num_accepted_tokens`); its test cases
  (query lengths 0 / 1 / ragged, accepted counts 1..M) become 26c's.
- **MoE at M rows (Decision 10):** the union-aware form - the M x 9 (row, expert) pairs sorted by expert in SLM, each
  expert's block read once for every row that chose it, the per-row combine still in ascending id - built in 26c beside the
  plain M-row form (the bitwise reference: same products, same per-row order).
- **Two cards:** the drafter on device 1 (it holds `lm_head`) with an embedding replica (1.07 GB; 19d's rule); taps 1 / 18 /
  35 copied by device 0 straight into device 1's tap buffer before its hand-off (19d: the `copy` hand-off only, `peer`
  refused by name); the draft list on device 1 mirrors the drafts to device 0's Control; commit moves the slot index on both
  and replays the projection on device 1.

### 5.4 The loop (26c; 19c's, unchanged in shape)

Per iteration: the draft list -> `verify(K)` -> host acceptance (`accept_greedy`, or `accept_sparse` with q from the
readback's realized scores at the request's T - Decision 6) -> `commit(j, t)`: pos += j + 1, the slot index, the
projection of verify rows 0..j into the ring and the anchor. `ingest` / `generate` run `verify(0)` + `commit(0)` so the
ring and the anchor stay current across plain and speculative steps. `AdaptiveK` over a cost row flat in K (DERIVED until
26a measures: verify(M) from P0, the block from §4).

### 5.5 Prefill (26c)

Every chunk copies the six taps of its rows into a staging (6 x 4096 bf16 a row; 100 MB at 2048 rows) and projects
**the chunk's last 1024 rows** (Decision 7) - with prefill chunks ending at every block end where a snapshot can be taken
(the recurrent state exists only at chunk ends), the ring then holds exactly the 1024 positions before every snapshot's
position, so a restore at any block end drafts as the unbroken session would. The projection is a GEMM at M = 1024
(fc: 2 x 1024 x 24576 x 4096 = 206 GFLOP, ~0.6 % of a 2048-row chunk's 37 TFLOP - estimated). The anchor is the last
prompt row's. This closes 19d's recorded caveat for this drafter (its 2048 window made the same choice cost more there).

## 6. The prefix cache (26c)

19d's ruling (the ring rides in every snapshot) at LiLiCorr's sizes: **+5,242,880 B of ring and 16 KB of anchor** after
spec 25's state (167.8 MB fp32) - 3 % more a snapshot. `drafter_key` (19d's FNV over the drafter's shapes, taps, format
and its two norm weights) keys the store. `a608b6e6e0`'s lesson - a drafter's sliding tail must be retained at a
shared-prefix junction, and a sibling that diverges right after it can only use the block below - holds here by
construction: every snapshot carries the whole 1024-row tail of its own position (§5.5), and a hit is taken at the
largest block end <= the common prefix (spec 25 §6.3).

## 7. Correctness gates (spec 19's D0-D5 at LiLiCorr's shapes)

- **D0.** Without `--spec dflash` every kernel binary sha256-identical and the suites green; 19b-19d's and 23f's drafter
  tests unchanged.
- **D1.** The engine's draft from its own taps against `dflash_ref.py`'s LiLiCorr form fed the same taps: the 8 candidate
  sets equal (tie-aware), their log-probs to 1e-3, the lattice cosine >= 0.9999 (exact arm) / 0.999 (int8, int4), the walk's
  ids equal at T 0, the anchor row cosine >= 0.9999.
- **D2. Greedy is lossless:** `--spec dflash` greedy equals `--spec off` token for token on the golden prompts and A4 (two
  cards), whatever K the policy picks.
- **D3. Verify rows are bitwise M = 1:** logits, KV rows, the SSM state of every slot, conv ring columns, taps, at M = 2..8
  on synthetic and real weights; the union-aware MoE bitwise the plain M-row MoE.
- **D4. Sampling is exact:** 10^6 seeded draws through `accept_sparse` with LiLiCorr's 8-candidate q match the target's
  filtered p (spec 8 M4's bars); no accepted draft outside the candidates.
- **D5. Speed:** `b70-serve` with llama-benchy and the agentic transcript, `--spec off` / `lookup` / `dflash` (auto K):
  `dflash` becomes Bolt's default only if it beats both by >= 10 % on the agentic set and loses nowhere by more than 3 %
  (spec 19 D5's form); otherwise it ships as an option.

## 8. Stages

| stage | Mac (blind) | box | row |
|---|---|---|---|
| **26a** P0 | the LiLiCorr form of `dflash_ref.py` on a tiny random drafter + its NVFP4 decode; the cross-check against vLLM's own head module on CPU (Decision 1); the projection tool | tap dumps (25a's), teacher-forced acceptance per arm and corpus, the expert-union counts, the verify / block cost probe, the projection; optionally vLLM XPU serving Bolt with LiLiCorr for an external acceptance figure | 63 |
| **26b** the drafter | descriptor, loader host half (NVFP4 -> three formats), twins, `ll_*` kernels on the Mac GPU | first compiles, the drafter on device 1, D1 | 64 |
| **26c** engine and serving | verify-with-taps lists, slot / commit logic, the union-aware MoE twin, prefill projection, snapshot layout, the server path | D0, D2-D4 on two cards, restores, D5 | 65 |

Order: 26a needs 25a's tap dumps and routes (CPU) and nothing of the engine; 26b needs 25b (the target's embedding / head
on device 1) and can run beside 25c; 26c needs 25c-25e.

## 9. Out of scope

Training or fine-tuning LiLiCorr for our int4 target (Decision 9); tree verification; vLLM's adaptive verification
(`acceptance_estimator.py` - AdaptiveK covers K); a LiLiCorr drafter for any other target; batching (19d's hook covers the
interface).

## 10. Decisions

The operator approved every proposed decision below as written on 2026-10-20.

1. **Decision 1 (approved, operator 2026-10-20) - the reference:** `dflash_ref.py`'s new LiLiCorr form, ported from `V/models/lilicorr.py` and the
   DFlash 2 speculator (fp32 compute, as its vLLM forms), and **cross-checked once against vLLM's own `LiLiCorrHead`
   module instantiated on CPU** (one-rank process group, unquantised weights) on the tiny drafter - bitwise where both run
   fp32, else to 1e-6. If vLLM cannot be imported on the Mac, the box's vLLM image runs the check. *Cost if wrong:* a
   misread (a bias term, the lattice offset, the anchor's position) costs acceptance silently - D1 compares against our
   port, not against vLLM; the cross-check is the only external anchor (half a day).
2. **Decision 2 (approved, operator 2026-10-20) - the backbone's format: P0 picks among exact / int8 / int4 g64; the default is int8 per row
   (19b's) unless int4's E_K is within 0.05 of exact's at the chosen K** (23f's rule, which picked int4 for Muse).
   *Cost if wrong:* 0.1-0.3 of a plain step per block, or 0.05+ of acceptance.
3. **Decision 3 (approved, operator 2026-10-20) - the LiLiCorr head always exact** (W' bf16 + g; 74 MB). *Cost if wrong:* ~0.03 GB a block if
   int8 would do.
4. **Decision 4 (approved, operator 2026-10-20) - Mamba state at verify: 19c's slots first** (the operator's "reuse the mechanism"), the
   read-only-verify + replay-at-commit lever built in 26c only if P0's verify cost shows the slot writes above 10 % of a
   verify at the chosen K. *Cost if wrong:* ~1.7 ms a verify at M = 8 and 1.17 GB of memory, or a second commit path to
   prove bitwise.
5. **Decision 5 (approved, operator 2026-10-20) - K up to 7** (block 8; verify M <= 8), AdaptiveK per request over the measured cost row;
   `--spec-max 1..7`, default 7. *Cost if wrong:* none - K is a request-time choice.
6. **Decision 6 (approved, operator 2026-10-20) - sampled q = softmax(realized lattice scores / T) over the 8 candidates** (vLLM's), lossless
   through `accept_sparse`. P0 also estimates the alternative that folds the candidates' log-probs in, for the record only.
   *Cost if wrong:* acceptance at T > 0 only; never the output distribution.
7. **Decision 7 (approved, operator 2026-10-20) - prefill projects each chunk's last 1024 rows**, making every snapshot's ring exact (vs 19d's
   ruled "as the prefill left it" for Qwen3.8). *Cost if wrong:* ~0.6 % of prefill (estimated) for restores the recording
   may show never happen.
8. **Decision 8 (approved, operator 2026-10-20) - the drafter on device 1 with an embedding replica (1.07 GB)**, taps over the copy hand-off
   (19d). *Cost if wrong:* the alternative (drafter on device 0, logits shipped) moves 7 x 131072 fp32 a block across
   PCIe 3.0 - worse.
9. **Decision 9 (approved, operator 2026-10-20) - accept the target mismatch:** LiLiCorr was trained against the 24GB 2-bit release; our target
   is SergiioB's g128 (then ours, g64) - its taps are closer to the BF16 original's than the 24GB's are. P0 measures E_K on
   ours; no retraining. *Cost if wrong:* if acceptance on our int4 target is far below the published figures, the speed-up
   is smaller; the engine half is built anyway (ruled), and a retrained drafter would be a new spec.
10. **Decision 10 (approved, operator 2026-10-20) - the union-aware M-row MoE in 26c** (each expert block read once per verify), with the plain
    M-row form as its bitwise reference; P0's union counts set the expectation. *Cost if wrong:* without it a verify at
    M = 8 reads ~7.2 GB of experts (more than all other weights) and the speed-up shrinks by ~40 % (estimated).
11. **Decision 11 (ruled by the reference) - the window edge** is vLLM's `|q - k| <= 1023` (the only reference).
12. **Decision 12 (approved, operator 2026-10-20) - an external acceptance figure from vLLM XPU** serving SergiioB's checkpoint with LiLiCorr on
    the box (if the image's DFlash / LiLiCorr path runs on XPU - the card says SM120 only); recorded beside ours, never a
    gate. *Cost if wrong:* a box hour for a number that cannot be had.

## 11. P0 verdict (26a) - PENDING the box; the defaults until then are PROVISIONAL

Plan 26a, `tools/spec/lilicorr_p0_project.py` (r63.verdict). **Not a gate** (the operator's 2026-10-15 ruling): 26b / 26c are
built whatever it shows. On 2026-10-23 the real drafter's weights are read on the Mac (§12: decoded exactly, the formats'
bytes and weight error) but its acceptance needs Bolt's real taps, and the tap dumps, route records and card costs are the
box's (queue row 63), so
every default below is PROVISIONAL and every cost DERIVED from the bytes (`lilicorr_p0_project.py`'s table over spec 25
§14's layout bytes, printed by `test_lilicorr_p0_project.py`'s `test_derived`; depth 4096):

| | M = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verify, plain M-row MoE (plain steps) | 1.000 | 1.193 | 1.387 | 1.580 | 1.773 | 1.966 | 2.160 | 2.353 |
| slot writes' share of the verify (Decision 4) | - | 5.1 % | 6.6 % | 7.7 % | 8.6 % | 9.3 % | 9.8 % | 10.3 % |

(a plain step 5.524 GB: the non-expert weights 4.255 GB with the int8 head, 9 x 40 expert blocks of 2.500 MB average
0.900 GB, the SSM state read + written 0.336 GB, KV 0.034 GB; at depth 65536 the shares are 4.7 .. 9.9 %). One block:
exact 0.500, int8 0.306, int4 g64 0.214 plain steps (spec 26 §4's figures, re-derived). With the union-aware MoE at u = 30
distinct experts a MoE block at M = 8 (§4's ESTIMATE; r63.union measures u), the M = 8 verify falls from 2.353 to 1.593.

- **Drafter format (Decision 2): int8 per row** (the default) - PROVISIONAL until r63.accept's E_K (int4 g64 if within
  0.05 of exact's at the chosen K). Measured on the real weights (§12): int8's weight error ~1 %, int4 g64's ~11 % relative
  to g x W' - int4 starts behind; only E_K can rule.
- **K_max (Decision 5): 7**; AdaptiveK picks per request over the cost row above (26c's `MtpCost::bolt_lilicorr` starts from
  it until r63.cost's measured row replaces it).
- **q form (Decision 6): `lattice`** (softmax of the realized scores / T, vLLM's) - PROVISIONAL until r63.accept's sampled
  estimate compares it with `lattice+lp`.
- **The replay lever (Decision 4): not built** - the DERIVED slot-write share stays at or below 10 % up to M = 7 (10.3 % at
  M = 8, depth 4096); r63.cost's measured state rate at the chosen K decides.
- **The union MoE (Decision 10): built in 26c regardless**; its saving is r63.union's u and r63.cost's time per expert block.
- **The expectation (ESTIMATED: SPEED-Bench's published sampled acceptance at K 7 on the 24GB target, our DERIVED costs,
  the int8 drafter):** coding 5.26 tokens a verify -> 1.98x plain with the plain M-row MoE, 2.77x with the union MoE at
  u = 30; overall 4.30 -> 1.62x / 2.27x; writing 3.41 -> 1.28x / 1.80x. Decision 9's gap (our int4 target vs the 2-bit one
  the drafter was trained against) is what r63.accept measures.

## 12. 26a as built (2026-10-23; the Mac only - no card; the drafter's real weights read, Bolt's not)

Branch `spec26a-lilicorr-p0`; box queue row 63. Built: `tools/oracle/dflash_ref.py`'s LiLiCorr form (`lilicorr_config_from_dict`,
`nvfp4_decode` and the NVFP4 reader, `Drafter._lin` / `_softmax_sink`, `lilicorr_lattice` / `lilicorr_walk` /
`lilicorr_anchor`, `lilicorr_candidates`, `LiLiCorrSession`, `ContextKV.anchor_row`, `Drafter.mut` - test-only mutations),
`tools/oracle/lilicorr_make_tiny.py`, `tools/oracle/lilicorr_vllm_check.py`, `bolt_ref.py --tap-layers`,
`dflash_accept.py --model bolt-lilicorr` (+ `lilicorr_draft`, `int8_rows_g`, `rtn_int4_g64_f64`, `p_rows_lse`,
`run_source_lilicorr`), `dflash_p0.sh MODEL=lilicorr` (check / accept / lookup / union), `tools/spec/expert_union.py`,
`probe_draft_cost --shapes bolt-lilicorr` (+ `draft_cost_shapes.h`'s `bolt_lilicorr`), `tools/spec/lilicorr_p0_project.py`, the
tests `test_dflash_ref.py`'s 13 `test_lilicorr_*` (one on the real checkpoint), `test_dflash_accept.py`'s 5, `test_expert_union.py`,
`test_lilicorr_p0_project.py`, `lilicorr_cost_names_test`; queue row 63's stages and the data key `lilicorr`; the record
`docs/probe-lilicorr-2026-10-23.md`. kernel_cmdlines **+99 / -0 / ~0** (1,406 -> 1,505); no existing `.cl` edited.

**Measured on the Mac** (`agnes-ref-img`, torch 2.14.1+cpu, the 5.19.0 site; every number printed by the test named): NVFP4's
16 x 254 finite (code, scale) products exact in bf16 and f16, |W'| in [2^-10, 2688], the 2 NaN scale codes refused naming the
element, the e4m3 table = torch's `float8_e4m3fn` on every finite code; g on the output bitwise `g x (x W'^T)`, folding it into
W' moves 2125 / 2304 fc outputs; Review Focus 1-5 each a mutation that moves the lattice (20 / 20 anchors) or flips drafts
(the probe sheet's table: 5 .. 119 of 140); the window edge at 16; K = 1..7 prefixes of one walk; the batch = `draft_block`
(candidates and ids bitwise, the lattice within 9.5e-6); the int8 codes independent of g; the tiny pair end to end (the dump's
taps bitwise `run`'s h after blocks 1 / 2 / 4 / 6 / 9 / 10; taps one block early flip 94 / 126 drafts);
`lilicorr_cost_names_test` (110 names bound, 99 in the block, a stand-in expert block 2.507 MB against the 40 blocks' average
2.500 MB, the drafter's linears 1.0758 B parameters); `test_box_validate.py` 29 tests.

**The real drafter on the Mac** (its weights reached the Mac's HF cache during this work; `test_lilicorr_real_checkpoint`): 179 tensors (25 NVFP4 linears x U8 / F8_E4M3 / F32 + 104 BF16 - the header's census, the head's linears all with a bias), all 25 decoded exactly in 8 s (W' bf16-exact, no NaN scale; of 54,919,168 scale blocks none zero, none subnormal); **1,112,803,882 parameters** (the card's 1.113 B; 36,750,474 in the head); g in [0.000791, 0.00656], none a power of two; the formats' bytes exact 2.152 GB / int8 1.077 / int4 g64 0.572 + the head 0.074 (spec 26 §4's derived figures, now measured); the weight error against g x W' over the 16 NVFP4 backbone linears (relative, min / median / max): int8 0.0088 / 0.0098 / 0.0130, **int4 g64 0.1070 / 0.1085 / 0.1125** (re-quantising a 4-bit format into another with other groups costs ~11 % - Decision 2's E_K comparison decides whether it matters); one draft on random taps over a random stand-in target: finite, slot 0 a broadcast, 5.3 s on this CPU in fp32. Bolt's own 41.9 GB is not on the Mac, so acceptance on real taps is still the box's.

**Rulings where this differs from the plan's text or a precedent (each with its cost if wrong):**
1. *The block is always 8 rows and the head scores all 7 slots* (§5.3; the plan's "K = 1..7 from one block"), while vLLM runs a
   (1 + K)-row block with a K-slot lattice at K < 7 (its docs: "the corresponding prefix of the learned slot embeddings and
   attention bias"). *Cost if wrong:* drafts at K < 7 differ from vLLM's (the trained K 7 is the same).
2. *The sampled walk is keyed `seed ^ (slot << 32)`* (vLLM 3901d56800's `spec_step_seed`; 19a's DFlash 2 port at 96b0efe6b0 has
   no step seed); the noise is `-log(-log1p(-u))` in double (vLLM's fp32 path `_log1p_neg_stable`: picks can differ at
   near-ties). 26b's `ll_select` must key the same way to compare at T > 0. *Cost:* none to the output distribution.
3. *`bolt_ref.py` gained `--tap-layers`* (default 1, 18, 35, 51, 68, 85): the tiny Bolt has 12 blocks, so the tiny pair taps
   1, 2, 4, 6, 9, 10 (the real taps' kinds). *Cost:* none.
4. *The arms add the int8-head forms* (`exacth`, `int8h`, `int4g64h`: what b70-serve runs, spec 25 Decision 18) beside the
   plan's three. *Cost:* box CPU hours (ESTIMATED ~0.5 h an arm at 300 GFLOP/s).
5. *The sampled estimate conditions the walk on the text*: slot j's previous index is the teacher-forced token's among slot
   j - 1's candidates; a row whose token is not among them is censored at j (counted). The target row is the dump's top 64
   with its full-vocabulary lse at T 1 (rows whose 64 hold < top-p counted as `short`), no top-k (spec 25 §17 ruling 6).
   *Cost if wrong:* alpha_j for j >= 1 is biased by the conditioning (upward if the reachable rows are the easier ones).
6. *The union counts read the golden runs' route records* (25a's `run` - the A4 references record none) and a `--rows gen`
   form (decode rows only); `r63.union` is its own stage (the plan put it in r63.accept). *Cost:* none.
7. *The cost probe allocates all 88 blocks on one card* (~15.8 GB, ESTIMATED; the plan's "scaled by the counts" not needed), the GEMVs at
   the descriptor's {S 1, layout} (25b's PROVISIONAL cells), 25c's `nb_router` at M rows (E an argument - 25c ruling 2), the
   routed experts as layout-1 GEMV stand-ins at I 2304, the attention ESTIMATED from the KV bytes (23f's form), the state's
   slot ((1 + M) x 4 MiB) and replay (3 x 4 MiB: the verify's read + the commit's read-modify-write; §5.3's "2 x 4 MiB" is the
   commit alone) terms DERIVED at a measured `nb_state_rmw` rate, the LiLiCorr head's 56 rows as 7 launches of 8 (re-reads
   - an upper bound), argmax stage 1 as the top-8 + logsumexp stand-in, the exact arm as bf16 W' (g's multiply not modelled).
   The ~650 non-GEMV launches a step are not in it: `lilicorr_p0_project.py --plain-ms` (r60's measured step) converts.
   *Cost if wrong:* the verify's ratio to M = 1 overstated without --plain-ms.
8. *Decision 1's cross-check cannot run on the Mac*: vLLM's import needs 16 modules `agnes-ref-img` lacks (stubbing pydantic /
   msgspec breaks vLLM's own classes - measured), so `lilicorr_vllm_check.py` runs in the box's vLLM image (r63.vllmref) on
   unquantised fp32 weights (fp32(g W') folded on both sides); SKIP 77 if that image's vLLM predates `models/lilicorr.py`.
   *Cost if wrong:* the port's agreement with vLLM is unproven until then (the mutations hold it to vLLM's text).
9. *The int4 arm's f16 scale is ONE rounding from fp64* (numpy's double -> half); on bf16 inputs it equals rtn.h's fp32 rule
   (256 / 256 scales, 16384 / 16384 codes - `test_lilicorr_int4_from_g_w`). 26b's loader reproduces the fp64 rule.
   *Cost if wrong:* an int4 code differs where 2 amax / 15 rounds differently through fp32.
10. *Shared reference edits*: `read_target_embed_head` reads Nemotron-H's `language_model.{model,backbone}.embeddings.weight`;
    the vLLM path's linears go through `Drafter._lin` and its sinks through `_softmax_sink` (`linear` / the softmax unchanged
    when no g is set - every pre-existing test of `test_dflash_ref.py` / `test_dflash_accept.py` passes, the Muse form still
    bitwise transformers). *Cost:* none.
11. *Beyond the plan's row*: stages r63.k0 and r63.union; r63.accept's data keys add `bolt_g128` (the target's embedding /
    head) and `oracle_image`; the x.rest stage skips the new `lilicorr` label. *Cost:* none.

**Not run (box, row 63):** everything with Bolt's real weights or a card - the cross-check against vLLM's head (on the tiny
head and the real one: `lilicorr_vllm_check.py` runs both when the weights are cached), the acceptance
tables, lookup on the real continuations, the union counts, the cost probe (no new binary compiled by ocloc), the
projection with measured numbers, vLLM XPU with LiLiCorr. Every default in §11 is PROVISIONAL until then.

## 13. 26b as built (2026-10-23; the Mac only - no card)

Branch `spec26b-lilicorr-drafter`; box queue row 64 (Session 13 of `docs/box-day-plan.md`). kernel_cmdlines **+9 / -0 / ~0**;
no existing `.cl` edited; nothing compiled by ocloc, nothing run on a card.

**Built.** `model::DFlashForm::LiLiCorr` (`dflash_desc_from_config` reads `LiLiCorrDraftModel` through the vLLM DFlash 2
rules - the block from the top level, sinks, the `lilicorr_*` widths, NVFP4's quantization_config only; `check_lilicorr`
holds it to the kernels); `loader/nvfp4` (the decode, `f64_to_f16_rne`, `rtn_int4_g64_g`, `quantise_int8_tiled_g`);
`loader::LiliCheckpoint` (`lilicorr_repack`: the names both ways - every linear BF16 or an NVFP4 triplet -, the three
formats, the head as W'^T + g + bias), `loader/lilicorr_layout.h` (the bytes), `loader::load_lilicorr` /
`lilicorr_embed_replica`; `DraftFormat::Exact` (`parse_lilicorr_draft_format`: exact | int8 | int4; the other drafters
refuse it); the kernels `ll_attn`, `ll_topk`, `ll_lattice`, `ll_select`, `ll_gemv_bf16_g` (+ `df_prep` and the bf16 head at
M 7 from new CMake lines; 26a's block supplies the M 8 GEMVs and the int8 head at M 7); `runtime::dflash::LiliDrafter`
(`lilicorr_drafter`: `LiliBuffers`, the 116-launch draft list, the 9-launch projection with the anchor row, `LiliCtl` /
`LiliDraftOut` in `dflash_control.h`), `lilicorr_sizes` (the buffers, the launch counts, the snapshot bytes, the key,
`lili_bolt_reserve`); `draft_list_k` returns 7 for this form; the twin `tests/kernels/lilicorr_ref.h` and its fixture
(`dflash_fixture.py --lilicorr`); D1's tools - `dflash_d1 --model lilicorr`, `dflash_d1.py`'s LiLiCorr form,
`tests/kernels/lilicorr_d1_twin`, `dflash_d1.sh MODEL=lilicorr` (gpu / ref / twin). Tests: `lilicorr_desc_test`,
`nvfp4_test`, `lilicorr_repack_test`, `lilicorr_ref_test`, `lilicorr_names_test`, `lilicorr_plan_test` (host);
`lilicorr_kernels_test`, `lilicorr_draft_test` (card); `test_dflash_d1.py`'s `test_lilicorr_form`; `tools/mac/clrun/
lilicorr_run` (mac_check section 5).

**Measured on the Mac** (every number printed by the test named): NVFP4's 16 x 254 finite (code, scale) products exact in
bf16 and f16, |W'| in [2^-10, 2688], the NaN scales refused naming `weight_scale[row][block]`; f64 -> f16 in one rounding
(= the fp32 path on 200k floats; 1 + 2^-11 + 2^-40 rounds up where the two-step path ties to even); folding g into bf16
before int4 moves 325 of 2048 words of a random tensor; the int8 codes are W''s whatever g (`nvfp4_test`). The real
checkpoint through `LiliCheckpoint`: 179 tensors, 25 NVFP4 linears, held to Bolt (`lilicorr_repack_test`). The twin against
dflash_ref.py's LiLiCorr form on the fixture's 10 cases (no context; the window's edge for rows 0 and 7; the 16-slot ring's
wrap; K 7 / 3 / 1; the mask id as the anchor; two sampled walks with the slot-keyed seed): the 8 rows cosine 1.000000000,
candidates and walks equal, the lattice within 1.3e-5 (|values| <= 2.8), logits / log-probs within 2.7e-5 (|logits| <= 23), the anchor within
9.5e-7; the reference's `no_sink` and `anchor_p2` mutations reproduced by the twin within 3.7e-6 (the true lattice 1.7 /
2.7 away) (`lilicorr_ref_test`). `lilicorr_run` on the Mac's GPU (Intel UHD 630): every `ll_*` kernel and both lists agree
with the twin in the exact (bf16 head) and int8 arms - rows, anchor, candidates, lattice cosine 1.0000000, walks; `ll_attn`
alone within 1 bf16 ulp at p 1..123; `ll_topk` exact on planted rows over 3 chunks (log-probs within 8e-8); gemv.cl's int4
does not build there (as 23f). DERIVED (`lilicorr_plan_test`): the drafter at max_len 262144 is exact 2.377 GB / int8
1.302 / int4 g64 0.797 (weights + 150.5 MB of buffers, 134 MB of it the RoPE table), and Bolt plans 262144 on two 32 GiB
cards in every format beside the 1.07 GB replica (device 1: 26.9-28.5 of 34.4 GB).

**D1's Mac half on 25a's tiny pair** (`bolt_ref.py taps` over the tiny Bolt `pattern12`, 26a's `oracle-out-lilicorr-tiny`;
`lilicorr_d1_twin` + `dflash_d1.py`, 2 dumps x 16 anchors): the twin in **fp32** passes every D1 bar - sets 32 / 32, log-probs
within 4.8e-6, the lattice 32 / 32 at cosine 1.0000000, walks 32 / 32, the anchor cosine 1.0000000: the pipeline (the dump
format, the projection's window, the anchor at p - 1, the comparator) is right. The twin in the **engine's bf16 numerics**
(the card's arithmetic: 19b's bf16 GEMV inputs, bf16 ring, bf16 attention / SiLU outputs) in every arm: sets 15-16 / 32
tie-aware, walks 18-21 / 32, log-probs worst 1.7-1.9e-2 (of logits <= 5.4), the anchor cosine 1.0000000. **D1's 1e-3 log-prob
bar and its 1e-3 tie tolerance sit below what bf16 activations move** (rows cosine ~0.99997, ~0.8 % relative; the tiny
target's top-8 logits are within ~0.2 of each other): on the box the bar is expected to fail for numeric reasons unless the
real drafter's margins are much larger. `dflash_d1.py --lp-bar` (`LP_BAR`) re-runs the comparison at another bar; the
lattice, sets, walks and anchor are recorded either way. The levers, if r64.d1_ref shows the gap: relax the bar to the
measured noise (e.g. 1e-2, a cosine on log-probs), or fp32-input GEMVs for the backbone (19b's note; bytes unchanged,
speed unmeasured).

**Rulings where this differs from the plan's text (each with its cost if wrong):**
1. *LiLiCorr's checkpoint has its own host class* (`loader::LiliCheckpoint`, `lilicorr_repack.cc`) beside `DFlashCheckpoint`,
   and its own device loader / buffers / lists (`load_lilicorr`, `LiliBuffers`, `build_lili_*`) - the plan modified
   `dflash_{repack,loader,layout}`: the dtypes, names, head and readback differ wholesale; the 19b / 23f objects are
   untouched (D0 by construction). *Cost:* two parallel code paths to maintain.
2. *The tests are new executables* (`lilicorr_desc_test`, `lilicorr_repack_test`, `lilicorr_ref_test`, `lilicorr_names_test`,
   `lilicorr_plan_test`, `lilicorr_kernels_test`, `lilicorr_draft_test`) instead of extending `dflash_test` / `dflash_ref_test` /
   `dflash_names_test`; the twin is `lilicorr_ref.h` over `dflash_ref.h`. *Cost:* none.
3. *The exact arm's output scale is per column* (`gs[N]`, 4 N bytes: 589,824 B over the backbone): gate||up is one GEMV over
   two NVFP4 tensors with two g's. Numerically the per-tensor multiply. *Cost:* none.
4. *The head's projections read fp32 inputs* (`ll_gemm`: the final-normed rows, the anchor, the head's own activations), not
   19b's bf16 GEMV inputs - the plan's "lever" taken up front; the head is plain OpenCL C, so no bf16 copy is needed.
   *Cost:* unmeasured speed (r63.cost timed bf16 GEMVs as its stand-in).
5. *`ll_attn_part` runs 8 of a kv head's 16 q heads a work-group* (HPW 8): 16 x 8 q pairs would need 96 KB of local memory.
   *Cost:* each ring block is read twice (5,120 B a position, negligible).
6. *`LiliDrafter` is a `runtime::TapSink`*, not a `TapDrafter` (whose proposer side is Qwen3.8's 16-candidate readback and
   `runtime::Engine`): its proposer is its own `draft(replay, k, T, seed)` returning `LiliDraftOut` (MuseDrafter's
   precedent). *Cost:* none; 26c binds Bolt's verify through TapSink and the loop through `draft()`.
7. *The draft list always walks all 7 slots* (dctl.k = `draft_list_k` = 7); the caller keeps the first K (§12 ruling 1). The
   walk's noise is fp32 (`log(-log1p(-u))`, df_select's and vLLM's), the reference's double - a sampled pick can differ at
   near-ties. *Cost:* none to the output distribution.
8. *The anchor's validity is a control word* (`LiliCtl.anchor_valid`, DCTL 6): set by `projection()`, cleared by `reset()`, set
   by `load_ring()` (a snapshot follows a projection). *Cost:* a restore of a snapshot taken before any projection reads a
   zero anchor as valid (none is taken there: snapshots are at block ends after a prefill).
9. *`bolt_sizes.*` is not modified*: `runtime::dflash::lili_bolt_reserve` fills 25b's `BoltReserve` (`drafter`,
   `drafter_embed_replica`); b70_dflash_plan links b70_bolt_plan. *Cost:* none.
10. *The fixture is narrower than 26a's tiny drafter* (hidden 128, 4 / 1 heads x 32, FF 256, conv groups of 8 - the kernel
    projection a whole 64-column GEMV width -, the head 64 / 4 heads / factor 64) over a synthetic 1024-row target: 1.55 MB.
    Re-running the script writes the same tensors; safetensors' header order varies. *Cost:* none.
11. *dflash_d1.py's LiLiCorr reference rounds the head's input to bf16* (the engine's head reads `df_norm`'s bf16 copy;
    `head_input` in the run line, fp32 for the twin's `--reference`); the log-prob bar is `--lp-bar`. *Cost:* none.
12. *The RoPE table is 19b's fp32 [max_len + 8][128]* - 134 MB on device 1 at 262144 (counted). *Cost:* memory only.
13. *The embedding replica* is uploaded by `loader::lilicorr_embed_replica` from the Bolt checkpoint's row-major tensor; who owns
    it beside Bolt's engine is 26c's. *Cost:* none here.

**PENDING (box, row 64):** the first ocloc build of the 9 binaries; `lilicorr_kernels_test` (ll_gemv_bf16_g bitwise
gemv_bf16 x g, `ll_attn` within 1 ulp at p 5 .. 131000, `ll_topk` at 131072, the head chain's lattice cosine >= 0.99999 on
random real-width weights, the walk greedy and sampled, `ll_anchor`); `lilicorr_draft_test` (the real drafter in three
formats beside a synthetic target: launches, the projection across the ring's wrap, the anchor, a draft at p 1100, the
replay, k 3, a sampled walk, the snapshot, a cold drafter, the host-timed draft); D1 per arm on 25a's tap dumps (opt-in
r64.d1 / r64.d1_ref). Every default (int8, K 7, q `lattice`) stays §11's PROVISIONAL one until r63.

**For 26c:** bind Bolt's verify lists through `TapSink` (taps 1 / 18 / 35 copied by device 0 into device 1's `taps()` before
its hand-off; 51 / 68 / 85 on device 1), commit through `projection(pos, j + 1)` (<= 8 rows; the anchor follows), the
prefill's last 1024 rows 8 a replay (Decision 7), snapshots with `ring_bytes()` (5,259,264 B: the ring + the anchor) and
`key()` (`lili_drafter_key`, the form folded in); the sampled q is `softmax(LiliDraftOut.scores[s] / T)` on `cand[s]`
(Decision 6) - the device's sampled walk already keys by slot; the replica's allocation (1.07 GB on device 1,
`lili_bolt_reserve` counts it) and `--draft-format exact|int8|int4` (`parse_lilicorr_draft_format`, default int8) /
`--spec-max 1..7` (LiliDrafter's `max_k`) are the CLI's.

## 14. 26c as built (2026-10-23; the Mac only - no card)

Branch `spec26c-lilicorr-engine-serving`; box queue row 65 (Session 14 of `docs/box-day-plan.md`). kernel_cmdlines additions
only (no existing `.cl` edited, no existing command line moved); nothing compiled by ocloc, nothing run on a card.

**Built.** Kernels (the CMake spec 26c block, `B70_BOLT_VERIFY`): 25c's kernels at M = 2..8 by new lines (the norm sites, the
gather of `cur_token[0..M-1]`, conv, gated norm, route, shared act, attention prep, the plain experts, the bf16 head; 26a's
GEMVs / routers / int8 heads / argmax stage 1 and 25c's M 4 lines reused by name; Muse's `mg_attn` at M rows when the Muse block
is off), `nb_ssm`'s slot build `_SL8` at every M (25c's `SPEC_SLOTS` define: 8 slots, `CTRL_LIVE` 19 = `Control::gdn_live`,
`SPEC_SLOT_STRIDE` one block's state - slot 0 the block's own, slots 1..7 its copies side by side), and
`src/kernels/bolt/nb_moe_union.cl` (`nb_moe_up_U` / `nb_moe_down_U` / `nb_moe_comb_U`: the union schedule derived per work-group
in SLM from the M route rows, each distinct expert's tiles read once for all its (row, slot) pairs, the per-pair chains and the
ascending-id combine nb_moe.cl's verbatim; the counting build `_CNT`); `kernels::bolt::verify_variants`. Runtime:
`BoltVerifyBuffers` (the verify scratch at 8 rows, the slots, the union's down rows), `build_verify` (25c's walk over a scratch
view; the six taps of every row after the fold that produces a tap block's output - device 0's into device 1's `taps()` before
the copy hand-off's barrier; the verify link, a second `PipelineLink` at 8 rows), `BoltEngine::enable_spec / attach_drafter /
verify / commit / max_verify_k / read_verify_logits / replay_aux / read_ssm_slot` (commit: the drafter's `projection(n, j + 1)`
on device 1, then pos, the pending id and `gdn_live = (live + j) % 8` on both cards), the live slot moved back to slot 0
(`canonical_ssm`) before a plain list, a prefill or a snapshot, `read_logits` after a commit reading the verify's row j, the
prefill tail (`PrefillTaps`: each tap block's rows `lili_prefill_r0 .. C` copied by 2D copies into a staging on device 1, then
26b's projection 8 rows a replay before the block hook), `state_bytes / save_state / load_state` with the ring and the anchor
after the target's state. Server / CLI: `server::accept_sparse_rows` / `accept_point_mass_rows` (the rule over the caller's
filter), `MtpCost::bolt_verify / bolt_lilicorr` (DERIVED, §11), `BoltEngineAdapterT`'s step_many / step_drafts / truncate_to /
the deferred commit / drafter_key (inert without `spec_k`), `cli/bolt_spec.h` (`SpecArgs`, `check_spec`, `spec_cost`,
`plan_terms`, `attach_drafter` - the replica from the Bolt checkpoint's own rows -, `draft_hooks`), `b70-serve <bolt> --spec
dflash|lookup` (`--draft-model`, `--draft-format exact|int8|int4`, `--draft-q lattice|lattice+lp`, `--spec-max 1..7`,
`--spec-cost`, `B70_BOLT_VERIFY_MOE=union|plain`; the startup's `spec dflash:` line names the drafter, its format and key).
Tests: host `bolt_verify_names_test`, `bolt_spec_test`, `bolt_spec_plan_test` (+ `bolt_serve_args_test`'s new refusals); card
`bolt_moe_union_test`, `bolt_verify_test`, `bolt_commit_test`, `bolt_dflash_test`, `bolt_dflash_prefill_test`,
`bolt_dflash_prefix_gpu_test` (`tests/runtime/bolt_spec_rig.h`: a recording TapSink, session snapshots); Mac drivers
`tools/mac/clrun/nb_union_run`, `nb_slots_run` (mac_check section 5).

**Measured on the Mac** (every number printed by the test named): `nb_union_run` (UHD 630, DN_KS 1 for both forms): the union's
a / down rows / r bitwise the plain M-row form and the twin in 10 cases - the same 9 experts in 8 rows, 72 disjoint, one expert
in every row (65 distinct), random M 2..8 (14 .. 24 distinct) - and the counting build reads each chosen up block I / 64 times
and down block 16 times, every other none. `nb_slots_run`: the M-row conv + the slot build bitwise the plain kernels at M 8 /
5 / 4 (bf16 state) / 1 from live 5 / 7 / 3 / 4 (the ring wraps), every accepted count j = 0..7 continues as the plain run, the
j + 1 mutant caught 9 / 9, 3dc6def360's ragged M 8 / 1 / 5 / 2 with commits 6 / 0 / 2 / 1 equal one plain run of the 13 accepted
rows. `bolt_spec_test` (a fake engine with BoltEngine's contract and a fake readback): D2 at every K 1..7, K varying, truncated
runs and lookup (600 ids each, 197 / 403 .. 295 / 2471 drafts accepted); Review Focus 3 after every commit; D4 at 10^6 iterations
a case (lattice and lattice+lp q, Bolt's filter at T 1.0 / top-p 0.95 and filter_probs at T 0.7 / top-k 12 / top-p 0.8, lookup's
point mass): p-values 0.022 .. 0.893 (n + 1) and 0.147 .. 0.786 (n + 2 | d_1 kept), every accepted draft one of its slot's
candidates; the mutant residual chi2 48,617 (p 0.0000) - 92 s on the Mac; the deferred commit; Review Focus 4's bookkeeping.
`bolt_spec_plan_test` (DERIVED from the descriptor): the slots 1,174,405,120 B (22 / 18 Mamba blocks on the cards), the verify
scratch 14,839,296 B a card, the launches 653 at M 1 and 693 at M 2..8 with the union (653 plain), `--max-len auto` 262144 on two
cards with `--spec dflash` in every format (device 1: exact 28.276 / int8 27.200 / int4 26.695 of 32.530 GB).

**Rulings where this differs from the plan's text or a precedent (each with its cost if wrong):**
1. *The verify scratch is its own allocation* (`BoltVerifyBuffers`, 8 rows) beside 25c's M = 1 scratch, made only by
   `enable_spec`; the walk binds either through a scratch view. A plain run allocates and captures exactly 25c's (D0). *Cost:*
   ~15 MB a card.
2. *The verify lists bind no fused arm* (each is bitwise the as-built list, Decision 16). *Cost:* +120 launches a verify if r60
   makes the arms the default.
3. *The union MoE is three launches* (up, down into a per-pair buffer, the combine) - the plan named two; the combine cannot sit
   in a per-expert work-group without atomics. *Cost:* +40 launches a verify at M > 1 (~0.1 ms, ESTIMATED).
4. *With the verify lists on, every plain token is verify(0) + commit* (19c's / 23f's rule): the M = 1 verify list (the slot SSM,
   the plain MoE) keeps the slots the state's home and the drafter's ring filled. *Cost:* a commit (and a projection replay
   with a drafter) a plain token on the host.
5. *The live slot goes back to slot 0 before a plain list, a prefill or a snapshot* (`canonical_ssm`: one 4 MiB copy a Mamba
   block when live != 0, 168 MB - ~0.3 ms, ESTIMATED); snapshots hold slot 0, a restore sets live 0. *Cost:* one copy at a
   request boundary.
6. *Sampled requests walk on the host* over the readback's whole lattice (q = softmax(scores / T) in double, the request's
   RNG; the device always walks greedily) - the handoff said the device's sampled walk. Exact by construction (the fp32 Gumbel
   noise of `ll_select` never decides a sampled draft); `--draft-q lattice|lattice+lp` switches Decision 6's form. *Cost:*
   none to the output; the device's sampled walk is unused (26b's tests cover it).
7. *The acceptance filters with Bolt's own `filter_probs_top_p`* (`accept_sparse_rows`: the same rule; `filter_probs` sorts all
   131072 rows, 17 ms a row). *Cost:* none (the same distribution, spec 25e's review I1).
8. *A sampled request with a repetition penalty decodes plainly* (its p depends on ids a run has not committed). *Cost:* no
   speculation for such requests (NVIDIA's defaults carry none).
9. *The prefill's projection replays 26b's 8-row list* (Task 5's recorded fallback; the GEMM path at M 1024 is NOT built) over
   the rows `lili_prefill_r0` names - the window of the call's end, and a chunk's last 1024 rows when it ends at a mid-prompt block
   end - instead of every chunk's last 1024 (the same final ring; a chunk that ends elsewhere is overwritten). *Cost:* ~128
   replays a 1024-row window (~25-35 ms a 2048-row chunk, ~6 %, ESTIMATED; r65.prefill / benchy measure); a GEMM path if it
   matters.
10. *`read_logits` after a commit reads the verify scratch's row j* (the row the pending id came from) - the sampled server path
    draws the next id there. *Cost:* none.
11. *The adapter detects BoltEngine's speculative calls* (`HasVerify`): 25e's host fakes compile and run unchanged; `spec_k` > 0
    over an engine without them is refused. *Cost:* none.
12. *`b70-decode` still refuses `--spec`* (b70-serve only); D2 on the card runs through the server's adapter
    (`bolt_dflash_test`). *Cost:* no command-line A/B outside the server.
13. *Two cards: a second `PipelineLink` (copy, 8 rows: 204,864 B on device 1)*; `--pipeline-handoff peer` with `--spec` is
    refused by name (19d). *Cost:* none.
14. *`--spec-min-match` / `--spec-history` are accepted with `--spec lookup` for Bolt* (25e refused every proposer flag).
    *Cost:* none.
15. *Decision 4's replay lever is not built* (§11: the derived slot-write share <= 10 % to M 7, 10.3 % at M 8). *Cost:* ~1.7 ms
    a verify at M 8 if r63.cost measures more.
16. *`MtpCost::bolt_lilicorr`'s union rows assume u(M) linear from 9 at M 1 to 30 at M 8* (ESTIMATED: §4's u = 30 at M 8 is the
    only figure). *Cost:* AdaptiveK picks K on a wrong slope until r63.union / r63.cost replace it (`--spec-cost`).
17. *New test executables and a second Mac driver* (`bolt_verify_names_test`, `bolt_spec_test`, `bolt_spec_plan_test`,
    `nb_union_run` beside the plan's `nb_slots_run`); `bolt_verify_test` holds the verify rows to 25c's DECODE list (stronger
    than M = 1 of the verify lists). *Cost:* none.
18. *The branch is `spec26c-lilicorr-engine-serving`* (the plan: `spec26c-lilicorr-engine`). *Cost:* none.

**PENDING (box, row 65):** the verify binaries' first ocloc build; `bolt_moe_union_test` (the union bitwise the plain form at
the real shapes, DN_KS 2, every I; the counting build); `bolt_verify_*` (D3: logits, KV rows, ring columns, SSM slots and taps
at M 1..8 bitwise 25c's decode steps, synthetic one card and SergiioB's two, union and plain); `bolt_commit_*` (Review Focus 1
on both cards, 3dc6def360's cases); `bolt_dflash_*` (D2 on the golden prompts and the agentic transcript, every K and AdaptiveK;
the tokens a verify per format - D5's acceptance); `bolt_dflash_prefill_test` (Decision 7; the prefill's ring vs the decode
path's, the worst cosine); `bolt_dflash_prefix_gpu_test` (Review Focus 4); `r65.serve` (A4 identical, seeded sampled twice);
opt-in `r65.benchy` (D5 - the default rule). Every default stays §11's PROVISIONAL one: `--draft-format int8`, `--spec-max 7`,
`--draft-q lattice`, `B70_BOLT_VERIFY_MOE=union`; `--spec dflash` is an option until D5.
