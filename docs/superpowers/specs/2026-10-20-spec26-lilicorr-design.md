# Spec 26 - LiLiCorr: NVIDIA's DFlash-family drafter for Nemotron 3.5 Bolt

**Status:** design, 2026-10-20, for operator review. Open decisions are marked **Decision N (proposed)** with their
cost if wrong and collected in §10. Nothing is built.
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
- **Verify** (26c): spec 25c's decode list at M = K + 1 <= 8 - every `nb_*` kernel and GEMV already built for M <= 8 in
  25c - writing the six taps of every row (19c's `build_verify_taps`: after block i's output is folded, i.e. inside the
  next block's norm, the row's residual is copied out).
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

1. **Decision 1 (proposed) - the reference:** `dflash_ref.py`'s new LiLiCorr form, ported from `V/models/lilicorr.py` and the
   DFlash 2 speculator (fp32 compute, as its vLLM forms), and **cross-checked once against vLLM's own `LiLiCorrHead`
   module instantiated on CPU** (one-rank process group, unquantised weights) on the tiny drafter - bitwise where both run
   fp32, else to 1e-6. If vLLM cannot be imported on the Mac, the box's vLLM image runs the check. *Cost if wrong:* a
   misread (a bias term, the lattice offset, the anchor's position) costs acceptance silently - D1 compares against our
   port, not against vLLM; the cross-check is the only external anchor (half a day).
2. **Decision 2 (proposed) - the backbone's format: P0 picks among exact / int8 / int4 g64; the default is int8 per row
   (19b's) unless int4's E_K is within 0.05 of exact's at the chosen K** (23f's rule, which picked int4 for Muse).
   *Cost if wrong:* 0.1-0.3 of a plain step per block, or 0.05+ of acceptance.
3. **Decision 3 (proposed) - the LiLiCorr head always exact** (W' bf16 + g; 74 MB). *Cost if wrong:* ~0.03 GB a block if
   int8 would do.
4. **Decision 4 (proposed) - Mamba state at verify: 19c's slots first** (the operator's "reuse the mechanism"), the
   read-only-verify + replay-at-commit lever built in 26c only if P0's verify cost shows the slot writes above 10 % of a
   verify at the chosen K. *Cost if wrong:* ~1.7 ms a verify at M = 8 and 1.17 GB of memory, or a second commit path to
   prove bitwise.
5. **Decision 5 (proposed) - K up to 7** (block 8; verify M <= 8), AdaptiveK per request over the measured cost row;
   `--spec-max 1..7`, default 7. *Cost if wrong:* none - K is a request-time choice.
6. **Decision 6 (proposed) - sampled q = softmax(realized lattice scores / T) over the 8 candidates** (vLLM's), lossless
   through `accept_sparse`. P0 also estimates the alternative that folds the candidates' log-probs in, for the record only.
   *Cost if wrong:* acceptance at T > 0 only; never the output distribution.
7. **Decision 7 (proposed) - prefill projects each chunk's last 1024 rows**, making every snapshot's ring exact (vs 19d's
   ruled "as the prefill left it" for Qwen3.8). *Cost if wrong:* ~0.6 % of prefill (estimated) for restores the recording
   may show never happen.
8. **Decision 8 (proposed) - the drafter on device 1 with an embedding replica (1.07 GB)**, taps over the copy hand-off
   (19d). *Cost if wrong:* the alternative (drafter on device 0, logits shipped) moves 7 x 131072 fp32 a block across
   PCIe 3.0 - worse.
9. **Decision 9 (proposed) - accept the target mismatch:** LiLiCorr was trained against the 24GB 2-bit release; our target
   is SergiioB's g128 (then ours, g64) - its taps are closer to the BF16 original's than the 24GB's are. P0 measures E_K on
   ours; no retraining. *Cost if wrong:* if acceptance on our int4 target is far below the published figures, the speed-up
   is smaller; the engine half is built anyway (ruled), and a retrained drafter would be a new spec.
10. **Decision 10 (proposed) - the union-aware M-row MoE in 26c** (each expert block read once per verify), with the plain
    M-row form as its bitwise reference; P0's union counts set the expectation. *Cost if wrong:* without it a verify at
    M = 8 reads ~7.2 GB of experts (more than all other weights) and the speed-up shrinks by ~40 % (estimated).
11. **Decision 11 (ruled by the reference) - the window edge** is vLLM's `|q - k| <= 1023` (the only reference).
12. **Decision 12 (proposed) - an external acceptance figure from vLLM XPU** serving SergiioB's checkpoint with LiLiCorr on
    the box (if the image's DFlash / LiLiCorr path runs on XPU - the card says SM120 only); recorded beside ours, never a
    gate. *Cost if wrong:* a box hour for a number that cannot be had.
