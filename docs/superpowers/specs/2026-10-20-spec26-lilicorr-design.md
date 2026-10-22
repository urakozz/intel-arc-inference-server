# Spec 26 - LiLiCorr: NVIDIA's DFlash-family drafter for Nemotron 3.5 Bolt

**Status:** design, 2026-10-20, approved by the operator 2026-10-20. Decisions are marked **Decision N** with their
cost if wrong and collected in §10. **26a built on the Mac 2026-10-23** (§12 as built; §11 the P0 verdict - PENDING the
box, its defaults PROVISIONAL); 26b / 26c not built.
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
built whatever it shows. On 2026-10-23 nothing of the real drafter has been read (its 962 MB of weights were not
downloaded: PENDING-WEIGHTS) and the real Bolt tap dumps, route records and card costs are the box's (queue row 63), so
every default below is PROVISIONAL and every cost DERIVED from the bytes (`lilicorr_p0_project.py`'s table, spec 25 §14's
layout bytes; depth 4096):

| | M = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verify, plain M-row MoE (plain steps) | 1.000 | 1.193 | 1.387 | 1.580 | 1.773 | 1.966 | 2.160 | 2.353 |
| slot writes' share of the verify (Decision 4) | - | 5.1 % | 6.6 % | 7.7 % | 8.6 % | 9.3 % | 9.8 % | 10.3 % |

(a plain step 5.524 GB: the non-expert weights 4.255 GB with the int8 head, 9 x 40 expert blocks of 2.500 MB average
0.900 GB, the SSM state read + written 0.336 GB, KV 0.034 GB; at depth 65536 the shares are 4.7 .. 9.9 %). One block:
exact 0.500, int8 0.306, int4 g64 0.214 plain steps (spec 26 §4's figures, re-derived). With the union-aware MoE at u = 30
distinct experts a MoE block at M = 8 (§4's ESTIMATE; r63.union measures u), the M = 8 verify falls from 2.353 to 1.593.

- **Drafter format (Decision 2): int8 per row** (the default) - PROVISIONAL until r63.accept's E_K (int4 g64 if within
  0.05 of exact's at the chosen K).
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

## 12. 26a as built (2026-10-23; the Mac only - no card, no real weight)

Branch `spec26a-lilicorr-p0`; box queue row 63. Built: `tools/oracle/dflash_ref.py`'s LiLiCorr form (`lilicorr_config_from_dict`,
`nvfp4_decode` and the NVFP4 reader, `Drafter._lin` / `_softmax_sink`, `lilicorr_lattice` / `lilicorr_walk` /
`lilicorr_anchor`, `lilicorr_candidates`, `LiLiCorrSession`, `ContextKV.anchor_row`, `Drafter.mut` - test-only mutations),
`tools/oracle/lilicorr_make_tiny.py`, `tools/oracle/lilicorr_vllm_check.py`, `bolt_ref.py --tap-layers`,
`dflash_accept.py --model bolt-lilicorr` (+ `lilicorr_draft`, `int8_rows_g`, `rtn_int4_g64_f64`, `p_rows_lse`,
`run_source_lilicorr`), `dflash_p0.sh MODEL=lilicorr` (check / accept / lookup / union), `tools/spec/expert_union.py`,
`probe_draft_cost --shapes bolt-lilicorr` (+ `draft_cost_shapes.h`'s `bolt_lilicorr`), `tools/spec/lilicorr_p0_project.py`, the
tests `test_dflash_ref.py`'s 12 `test_lilicorr_*`, `test_dflash_accept.py`'s 5, `test_expert_union.py`,
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

**Not run (box, row 63):** everything with real weights or a card - the cross-check against vLLM's head, the acceptance
tables, lookup on the real continuations, the union counts, the cost probe (no new binary compiled by ocloc), the
projection with measured numbers, vLLM XPU with LiLiCorr. Every default in §11 is PROVISIONAL until then.
