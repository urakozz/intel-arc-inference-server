# Spec 21 - the `qwen4_exp` family (Qwen3.8-Flash-Next) in the engine

**Status:** design, 2026-10-09; the operator approved the design on 2026-10-08 (§0 records what was ruled).
Open decisions are marked **(decide)**. Nothing is built.

**Model:** `Qwen/Qwen3.8-Flash-Next` (`Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp`; bf16,
359,999,963,128 B in 1658 tensors). Intel's derived `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (181.17 GB: routed
experts int4 g128 sym, everything else bf16) is the interim checkpoint (§5). Licence per Intel's derived card:
`qwen-community-1.0` (21a reads the original's card).

**Order:** after spec 16 (pipeline parallel) and alongside spec 18 / 20; it reuses spec 15's GDN + MoE machinery
(Ornith is a GDN + MoE hybrid on the same Qwen3.5 lineage) and spec 20's patterns for a model built blind (own
descriptor, loader and engine; synthetic checkpoints; `--layers N`). **The full 512-expert model runs end to end
only once spec 22 (the expert-offload tier) lands** (§3): spec 21 builds and gates the whole family on what fits.

Every number is **measured** unless marked **derived** (computed from the checkpoint's config and safetensors
headers), **estimated** or **published**. Citations of the reference are to transformers 5.19.0,
`transformers/models/qwen4_exp/modeling_qwen4_exp.py` (abbreviated `M:line`) and `configuration_qwen4_exp.py`
(`C:line`); of the second implementation to vLLM v0.31.1rc0-138 (`b52ae2aa7f`), `vllm/models/qwen4_exp/` (`V/...`).

## 0. What the operator ruled (2026-10-08)

- **Two specs.** Spec 21 = this family, built blind, developed and gated on the tiny random model
  (`qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next`), synthetic real-width checkpoints and a layer-truncated real model
  (`--layers N`, as Kolibri's) that fit in VRAM. Spec 22 = the expert-offload tier, probe-first on the box.
- **The new parts** (§4): hyper-connections bf16 first; QSA sparse attention following transformers and vLLM;
  PLE as a host-side table read zero-copy by a device kernel (built once here, reused by spec 22); MoE widened to
  512 experts / top-10; the GDN sigmoid output gate; the MTP head's new form.
- **Formats** (§5): our AutoRound int4 g64 sym from the bf16 original; until then Intel's g128 experts through the
  exact g64 expansion with its bf16 dense layers; the PLE table int8 per row in host RAM; KV bf16 first.
- **Box-only** (§9): every kernel's first compile and run, the real-weight golden gates, spec 22's probe.

## 1. The model

From `config.json` (`text_config`), `generation_config.json`, `tokenizer_config.json`, the index and every shard
header (read 2026-10-08), and the reference:

| | |
|---|---|
| layers | 48: GDN (`linear_attention`) at `l % 4 != 3` (36), **QSA** sparse attention at `l % 4 == 3` (12). The checkpoint says `full_attention`; the config maps it to `indexed_attention` (C:179-183; the tiny model's `qwen_sparse_attention` is mapped in `configuration_utils.py:96`) |
| hidden | 2560; vocab 248320; untied `embed_tokens` / `lm_head` (bf16, 1.271 GB each); max positions 262144 |
| norms | RMSNorm `(1 + w) · x̂` computed in fp32, eps 1e-6 (M:144-169); the GDN gated norm plain `w · x̂` (M:172-188). **No input / post-attention norms in a layer and no final norm:** the hyper-connection norm replaces them, and `lm_head` reads the final mixer's output (M:1485, M:1656) |
| residual | **hyper-connections**: 4 streams x 2560 bf16; the embedding repeated into 4 streams (M:1472); per layer an attention-side and an MLP-side gated residual (§4.1) |
| GDN | 16 k / 48 v heads of 128, conv 4 over 10240 channels, SiLU conv; `g = -exp(A_log) · softplus(a + dt_bias)`, `beta = sigmoid(b)`, L2-normed q / k, q / √128; v head h reads k head `h // 3` (`repeat_interleave`, M:575-576); **output gate `sigmoid(z)`** (`output_gate_type: sigmoid`, M:492), the one difference from Qwen3.5 / Ornith |
| QSA | 24 q heads, 2 kv heads (GQA 12), head_dim 256; `q_proj` gives `[q 256 | gate 256]` per head; `(1 + w)` q / k norms; partial NEOX RoPE on the first 64 dims, θ 1e7; scale 1/16 (M:820); output x `sigmoid(gate)` (M:890); a selection mask shared by all heads, from the indexer (§4.2) |
| RoPE | mRoPE sections [11, 11, 10] interleaved; text gives three equal position ids, which is exactly 1D RoPE (M:120-142) |
| MoE (every layer) | 512 routed SwiGLU experts, intermediate 640, top-10; router bf16 GEMV, softmax over 512 in fp32, top-10, renormalised (`norm_topk_prob` default true, C:162), cast to bf16, no bias (M:961-970); + `sigmoid(shared_expert_gate · x) · shared_expert(x)` (intermediate 640, M:988); the sum is the block output |
| PLE | one per-layer embedding at `ple_layer_ids [2]`, **one-indexed: layer_idx 1** (a GDN layer, M:1260); a 320,001,536-row x 160 n-gram table (§4.3) |
| MTP | one head (`mtp.*`): its own QSA layer with its own indexer, 512 own experts + shared, both HCs, a final mixer, `fc_embedding` / `fc_hidden`, `pre_fc_norm_embedding [2560]`, `pre_fc_norm_hidden [10240]`; **transformers ignores it** (`_keys_to_ignore_on_load_unexpected`, M:1314) - vLLM is its reference (§4.5) |
| tokens | generation EOS [248046 `<|im_end|>`, 248044 `<|endoftext|>`]; text-config EOS 248044 (the PLE's EOS, M:1086); added tokens 248044-248076; chat template sha256 `c3cf9e34...` = Qwen3.8's / Agnes's (docs/11, `src/tokenizer/chat_template.cc:142`): XML tool calls, `enable_thinking`, `reasoning_effort` |
| sampling defaults | `do_sample` true, T 1.0, top-p 0.95, top-k 20 (`generation_config.json`) |
| vision | a 27-block tower (0.90 GB); skipped for text |

**Parameters (derived):** routed experts 120.80 B, the PLE table 51.20 B, everything else in the language model
~4.9 B, MTP ~2.6 B, vision 0.45 B: ~180 B in the file. Read per token (derived): ~6.7 B parameters plus 16 PLE rows.

## 2. Reference semantics (pinned from source)

The sections below are the reference's arithmetic; 21a turns each into a facts-sheet row with its rounding points.

**2.1 The layer** (M:1265-1303). `if PLE: H = H + ple(H)`; `x, H0, inj = attn_hc(H)`; `y = mixer(x)` (GDN or
QSA); `H = H0 + y ⊗ inj`; `x, H0, inj = mlp_hc(H)`; `y = moe(x)`; `H = H0 + y ⊗ inj`. After 48 layers:
`out = final_mixer(H)` (no inject) → `lm_head`.

**2.2 A gated residual** (M:995-1030), H [4 x 2560]: `xn = grouped_rmsnorm(H)` (group 2560, `(1 + w)`, weight
[10240]); `g = sigmoid(up(silu(down(xn) / 4)))` with `down` 10240 → 320 and `up` 320 → 10240; block input `x =
mean_s(g ⊙ xn)` (2560); `inj = 2 · sigmoid(block_inject(xn) / 4)` (4 scalars); the caller forms `H0 + y ⊗ inj`.
vLLM fuses the pending combine with the next HC's norm (`V/nvidia/hyperconnection.py:177-203`); a missing
injection is unit weight on every stream (the MTP head uses this, §4.5).

**2.3 The indexer and QSA** (M:665-771, M:839-893), per QSA layer and query row p:
- `index_qk_proj` 2560 → (4 + 1) x 128 (bf16): 4 query heads, 1 raw key.
- q: `q_layernorm` (`(1 + w)`), then RoPE on its first 64 dims at p. The cache holds the **raw** per-token keys
  (not normed, not roped, bf16).
- Blocks of 4 consecutive positions aligned at 0. A complete block's key = mean of its 4 raw keys in fp32 → bf16 →
  `k_layernorm` → RoPE at the block's first position (M:735-741).
- `score_b = Σ_heads relu(q_h · k_b) / √128` in fp32 (M:747).
- Select the top `min(512, complete_blocks)` blocks (M:749) plus every tail position of the incomplete block (M:754):
  at most 2048 + 3 = 2051 positions. Visible positions include p itself, so when `(p + 1) % 4 == 0` p's block is a
  candidate and p is attended only if its block wins. Prefill applies the same rule per row.
- **Consequence:** with ≤ 2051 visible positions everything is selected - QSA is exactly causal attention. The
  selection only acts past position 2050, so gates need prompts longer than that (§7).
- vLLM agrees: 512 blocks + the causal tail of the open block (`V/nvidia/ops/qsa_indexer.py:242-262`), relu-sum
  **without** the 1/√128 (`:92-99`; order-preserving, rounding differs), a compressed-key cache plus a
  **per-request ring of raw keys** (`V/common/qsa_cache.py:5-11`), bf16 or fp8.
- A community engine's port selects `min(n_kv, 2051)` positions by cells; transformers and vLLM define the selection
  by blocks plus the tail, and this spec follows them.

**2.4 PLE** (M:1072-1247). Table: 16 heads (8 bigram + 8 trigram) x 160 dims; each head's vocabulary is a distinct
prime just above `ngram_vocab_size_base` 20,000,000, offset into one table padded to a multiple of 128: 320,001,536
rows (stored as 128 shards of [2500012, 160]), 102.4 GB bf16. Per token:
- **History:** the last 3 ids; a missing predecessor, or one at or before an earlier EOS 248044 in the sequence, is
  replaced by 248044 (M:1107-1121; padding is also mapped to EOS, M:1464-1468).
- **Hash:** `mixed = (t0·m0) XOR (t1·m1) [XOR (t2·m2)]` in int64, `id = mixed mod prime_head + offset_head`
  (M:1148-1164). The multipliers are odd and at most `(2^63 - 1) / 248320`, so no product overflows and `mixed` is
  non-negative (M:1040-1049). The checkpoint stores `layer_multipliers` [3], `ngram_heads_vocab_sizes` [16] and
  `ngram_heads_offsets` [16] as I64 tensors; the engine reads them and checks them against the formula (seed 1234,
  C:155).
- **Block:** `e` = the 16 rows concatenated (2560); `key = grouped_norm(key_proj(e))` (2560 → 10240); `value =
  value_proj(e)` (2560 → 2560); per stream `s = <key_s, grouped_norm(H)_s> / √2560`; `gate = sigmoid(sign(s) ·
  √max(|s|, 1e-6))`; `gated = value · gate` (10240); `out = gated + silu(dwconv_k4_d3(grouped_norm(gated)))` over a
  9-row history (M:1208-1247); `H += out`.
- **State per sequence:** 2 ids and 9 x 10240 conv rows.

**2.5 MTP** (vLLM, `V/nvidia/mtp.py:164-357`): for draft position i (rope position i, inputs `R_i` and `t_{i+1}`):
`e = fc_embedding(gemma_rms_2560(embed(t_{i+1})))`; `h_s = fc_hidden(gemma_rms_10240(R_i))_s` per stream (one RMS
over all 10240, `:227-229`, then the shared 2560 → 2560 projection on each stream); the layer input is `h` with `e`
added to every stream (unit injection); one full QSA layer with its own KV and indexer, 512 own experts + shared;
its own final mixer → the shared `lm_head`. `R_i` is the main model's **pre-mixer** 4-stream hidden; the head
returns both the single stream (logits) and its own pre-mixer multi-stream (the next draft step's `R`). Draft step 0
selects its QSA top-k; later steps reuse step 0's selection (`skip_topk`, `:258-261`).

## 3. What it costs (derived)

| part (per token, decode) | format | GB |
|---|---|---:|
| routed experts (48 x 10 x 3 x 2560 x 640) | int4 g64 | 1.253 |
| shared experts | int4 g64 | 0.125 |
| routers | bf16 | 0.126 |
| hyper-connections (96 + the final mixer) | bf16 | 1.281 |
| GDN projections + a / b | int4 g64 + bf16 | 1.121 |
| QSA projections + indexer | int4 g64 + bf16 | 0.357 |
| PLE projections | bf16 | 0.066 |
| `lm_head` | int8 (spec 9) | 0.636 |
| **total** | | **~4.97** |

- **Decode roofline:** 590 GB/s / 4.97 GB ≈ **119 t/s**; at the 77.6 % of roofline Qwen3.8 reaches (docs/05:
  33.96 ms against 26.34) ≈ **93 t/s**. HC in int8 → 4.32 GB ≈ 136 t/s (a later lever, decision 2). Spec 22's tier
  lowers this by its hit rate (spec 22 §2).
- **Attention per step** (12 layers): ≤ 2051 selected positions x 24,576 B ≈ 50.4 MB of KV, plus an indexer scan of
  768 B x context (100 MB at 131072).
- **Per position:** KV 24,576 B bf16 (12,288 int8) + 768 B of compressed indexer keys; raw keys only in a ≤ 3-entry
  tail ring per layer (vLLM's choice, §2.3). 262144 positions: ~6.6 GB.
- **Per sequence:** GDN state 113.2 MB fp32 + 2.2 MB conv; PLE state 0.18 MB.
- **Weights:** experts 64.17 GB at int4 g64 + ~5 GB of everything else + the MTP head (~1.4 GB with int4 experts) +
  KV, against **~62.2 GB usable on two B70s**. It does not fit: the full model needs spec 22.
- **What fits (derived):** with Intel's bf16 dense layers ~1.49 GB per layer + ~2.0 GB fixed (embedding, int8 head,
  final mixer, PLE projections): **~18 layers on one card, ~38 on two**; the planner decides the exact N.
- **Host RAM:** the PLE table int8 is 51.2 GB + one scale per row (§5). The box has about 128 GB.

## 4. The new parts

### 4.1 Hyper-connections (bf16)

The engine's residual becomes `H` [4][2560] bf16; `prep_res_norm` / `prep_res_fold` are replaced on this family by
three launches per gated residual:
1. `q4_hc_combine_norm`: `H = H0 + y ⊗ inj` of the previous block (vLLM's fusion), then the grouped norm → `xn`.
2. `q4_hc_down`: one bf16 GEMV over 324 rows (`down`'s 320 and `block_inject`'s 4) reading `xn`; `silu(/4)` on
   the 320, `2 · sigmoid(/4)` on the 4.
3. `q4_hc_up_mix`: the 320 → 10240 bf16 GEMV, `sigmoid`, `⊙ xn`, the mean over streams → the block input (2560).

The rounding points are the reference's (21a pins them: bf16 after each linear, after `/4`, after each
activation, the mean, the product and the add). The final mixer is launches 1-3 without the inject rows. HC int8 or
int4 (~0.6-0.9 GB per token) is a later lever, after its quality is measured; Intel kept HC bf16.

### 4.2 QSA: indexer, selection, sparse attention

**Decode, per QSA layer:**
- **Projections:** q‖gate‖k‖v GEMV (int4, 12288 + 512 + 512 rows); the indexer GEMV (bf16, 640 rows).
- **`q4_qsa_prep`:** q / k norms, partial RoPE (first 64 dims), the KV write; the indexer q norm + RoPE; the raw key
  into the layer's 4-slot tail ring; when `(p + 1) % 4 == 0` the block's compressed key (fp32 mean → bf16 → norm →
  RoPE at the block's first position) into the compressed cache.
- **`q4_qsa_score`:** fp32 scores over every complete block (`relu`, the head sum, `/ √128`), 256 B read per block.
- **`q4_qsa_select`:** top `min(512, n)` on the device: exact ties to the lower block index; written as an ascending
  position list (blocks expanded, then the tail), a count, and the 512th / 513th scores for the near-tie diagnostic.
- **`q4_qsa_attn`:** attention over the list for the 2 kv heads. The 12 q heads of a kv head share the list; scores
  x 1/16, softmax in fp32, positions summed in ascending order (the reference's masked-row order); `× sigmoid(gate)`;
  then o_proj.
- One kernel serves every depth: below 2052 visible positions the list is the identity (decision 10).
- Spec 10's v2 is the structure to borrow (stride, block reads, waves); its contiguous-range addressing does not
  apply.

**Prefill:** rows with ≤ 2051 visible positions are plain causal attention, so the first 2048 positions of every
prompt run spec 6's flash at head_dim 256, GQA 12 (Kolibri's `kol_pf_attn` is GQA 12 at head_dim 128). Beyond that,
per chunk:
- **Scores:** one GEMM, (C x 4 heads) x 128 against the compressed keys, relu-summed per head group.
- **Selection:** a row-wise top-512 over up to 65,536 blocks.
- **Sparse flash:** one row's 12 q heads are the M of the DPAS tile against its gathered positions.
- **Compressed keys:** for the chunk's complete blocks, formed from the prefill raw keys before any row reads them.

### 4.3 PLE: the host table read zero-copy

The table lives in host USM (`zeMemAllocHost`), int8 per 160-wide row with one scale per row (§5). Two launches
replace the embedding-side work:
- **`q4_ple_gather`** (one work-group per token): reads the current id and the 2-id history from device memory
  (Control's token, no host value); applies the EOS rule, the int64 hash and the 16 indices; reads the 16 rows over
  PCIe straight from host USM (2,560 B + scales per token); dequantises to bf16 `e`; advances the history.
- **`q4_ple_block`:** `key_proj` / `value_proj` bf16 GEMVs (one launch, 12800 rows), the per-stream gate, the grouped
  norms, the dilated depthwise conv over its 9-row ring, `H += out`.

There is no host doorbell and no copy per token: the list stays fixed under replay. This is the mechanism spec 22
needs (a device kernel dereferencing host USM by an index computed on the device), so it is built and measured here
once:
- its rate goes into spec 22's P0;
- spec 22 reuses its allocation, pointer passing and checks.

### 4.4 MoE at 512 / top-10, and the GDN sigmoid gate

`moe.cl` is out of range on three counts:
- `TOP_K` is 1..8 (the route row holds 8 slots, `moe.cl:94-95`);
- `EXPERTS` is a power of two in [16, 256] (`:97-98`);
- `ROUTER_N >= EXPERTS + 1` (`:100-101`).

`pf_moe.cl` stages ids as `uchar` (`:61`, `:117-124`). FF 640 fits both: 10 k-groups, 40 gate‖up pairs. The family
gets its own route and slot kernels on Kolibri's pattern (`kol_route`: 512 slots on 256 lanes, two experts a lane;
`kol_pf_sort`: `ushort` ids):
- **`q4_route`:** fp32 softmax over 512, top-10 with ties to the lower id, renormalised, bf16 weights, the shared
  gate's sigmoid. It writes a 16-slot route row plus the 10th / 11th probabilities for the near-tie diagnostic.
- **`q4_moe_gate_up`:** 11 slots (10 routed + the shared expert); experts addressed by id inside the kernel (spec 15
  decision 3), so spec 22 can swap the address source for an indirection table without touching the list.
- **`q4_moe_down`:** writes the block output `y` (not a residual fold: the HC combine consumes it). The combine order
  is the oracle's experts implementation, pinned in 21a as Ornith's was (`grouped_mm` slot order vs `eager`
  ascending id, spec 15 §10).

The router GEMV is `gemv_bf16` over 512 + 1 rows padded to 528. GDN reuses `gdn_step.cl` unchanged (its defaults
are 16 / 48 heads and 10240 conv channels, Qwen3.8's shape); `prep_gated_head` hard-codes `silu(z)`
(`prep.cl:383-428`) and gains a `-DGDN_GATE_SIGMOID` variant. The existing binaries stay byte-identical (F0).

### 4.5 MTP

Spec 8's draft / verify / commit machinery, with the new head:
- **Inputs:** the main model's pre-mixer 4-stream hidden (`R_i`, kept by the last layer's MLP-side combine) and the
  token embedding.
- **Fusion:** `fc_embedding` + `fc_hidden` per stream, then the head's QSA layer with its own KV, indexer, tail ring
  and 512 experts.
- **Outputs:** the final mixer → `lm_head` (spec 8's draft-vocab option applies); the head's own pre-mixer hidden
  feeds the next draft step.
- **`pre_fc_norm_hidden`:** vLLM's single RMS over 10240 is built first. The per-stream variant (llama.cpp's
  reading of the same [10240] weight) is measured by acceptance, not assumed (decision 4).
- **Draft steps:** steps after the first reuse step 0's QSA list (vLLM).
- **Verify at M = K + 1:** M rows, each with its own selection (prefill's rule) and its own top-10 experts. Its cost
  is measured: the union of experts over the rows is the term that matters, and under spec 22 it is the miss term.

### 4.6 What is reused, adapted or new (engine map)

| piece | state |
|---|---|
| GDN step / conv (`gdn_step.cl`) | reused as is |
| gated norm (`prep_gated_head`) | adapt: sigmoid variant |
| full-attention prep (`attn_prep`) | adapt: QSA prep with the indexer |
| decode attention (spec 10 v2, GQA 12 untested) | adapt into the sparse kernel |
| selection, indexer score, compressed keys | new |
| `kv8.cl` (hard-codes 24 q / 4 kv heads, `kv8.cl:79-84`) | adapt (2 kv heads) when int8 KV follows |
| hyper-connections | new (replace `prep_res_norm` / `prep_res_fold` on this family) |
| PLE | new |
| `moe.cl` / `pf_moe.cl` | adapt (limits above; the `l0-int8` h8 path n/a: K = 2560 is not whole 1024 blocks) |
| router GEMV, renormalisation, shared gate | exists (spec 15) |
| MTP | adapt: the new head form |
| pipeline parallel (spec 16) | adapt: the hand-off is the materialised 4-stream residual, 10240 values (20 KB bf16), as vLLM's PP carries it |
| prefix cache (spec 7) | adapt: snapshots add the PLE history (2 ids + conv ring) and the indexer tail rings; the compressed keys are per-position blocks beside the KV |
| `ModelDesc`, loader | adapt / new descriptor (§6, 21b) |
| tokenizer, template | exists (template `c3cf9e34...`; `tokenizer.json` checked against Qwen3.8's in 21a) |

## 5. Formats

- **Ours (the target):** AutoRound int4 g64 sym, re-quantised from the 360 GB bf16 original, a big-machine run like
  Kolibri's 20b. Quantised: routed and shared experts, QSA `q/k/v/o_proj`, GDN `in_proj_qkv` / `in_proj_z` /
  `out_proj`. Kept bf16: routers, `shared_expert_gate`, `in_proj_a/b`, the indexer, hyper-connections, every norm,
  the embedding, `lm_head` (the engine builds the int8 head at load, spec 9), vision. **`ple` must be in the ignore
  list** (its projections stay bf16; the table is converted separately). The MTP head is decision 6. The export
  is `auto_round:auto_gptq`, as `loader::QuantConfig::parse` reads (docs/13).
- **Interim (development is not blocked):** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (AutoRound 0.15.0, sym,
  g128; `ignore_layers` = linear_attn, self_attn, hyper_connection, mlp.gate, shared_expert, ple, mtp, indexer,
  embed_tokens, lm_head, visual). Its experts (per-expert `qweight` / `scales` F16 / `qzeros`) load through the
  engine's exact g128 → g64 scale expansion (docs/13, the operator's "exact conversion" exception). Its dense layers
  run on the engine's bf16 arms (`gemv_bf16` as Kolibri's bf16 attention arm). Its MTP experts are bf16 (5.03 GB).
  Decode bytes on it are ~9.3 GB per token (derived; the GDN projections alone are 4.15 GB in bf16): a correctness
  vehicle, not a speed one.
- **The PLE table:** int8, symmetric per 160-wide row, in host RAM: 51.2 GB plus the scales (0.64 GB at 2 B, 1.28
  at 4 B; decision 7). It is a lookup table, not a linear: no AutoRound, quality measured by KL against the bf16
  table (F6). A bf16 table (102.4 GB) cannot sit in the box's ~128 GB beside anything else.
- **KV:** bf16 first; int8 later (spec 12's form, `kv8.cl` adapted).

## 6. Structure

On K2's / Kolibri's pattern (spec 18 §5.1, spec 20 §11):
- **Descriptor and loader:** `model::Qwen4ExpDesc` (`src/model/qwen4exp.*`), `loader::load_qwen4exp`
  (`src/loader/qwen4exp_*`). It reads per-expert int4 blocks for both checkpoints, the bf16 or int4 dense arms by
  name, the PLE table into host USM, and the I64 hash tensors (checked).
- **Engine:** `runtime::qwen4exp::Qwen4ExpEngine` (`src/runtime/qwen4exp/`): decode list, prefill walk, MTP, both
  cards through spec 16's placement.
- **Kernels:** family-only kernels in `src/kernels/qwen4exp/q4_*.cl` (host half `qwen4exp_kernels.h`). Shared
  kernels are reused at new shapes from new CMake lines only.
- **CLI:** `b70-decode` / `b70-serve` dispatch on `model_type qwen4_exp`; `--layers N` is the development mode (load
  layers [0, N), then the final mixer and the head; the reference truncates identically). A full-model load without
  spec 22 is refused, naming the bytes.
- **Tools:** `tools/oracle/qwen4exp_ref.py`; `tools/quantize/qwen4exp/` (the recipe, `make_synth.py`, the PLE
  int8 converter).

## 7. Gates

- **F0, nothing moves.** Every existing binary sha256-identical, `kernel_cmdlines` additions only, all suites green.
- **F1, the reference** (21a).
  - `qwen4exp_ref.py` equals transformers 5.19.0 on the tiny model bitwise (bf16 eager and fp32; prompt and cached
    decode), with one operator test per trap of §2: the HC rounding chain, the EOS history rule, the hash against
    the I64 tensors, the indexer at positions 2050 / 2051 / 2052 and at `(p + 1) % 4 == 0`, the tail, the sigmoid
    gate, the `h // 3` map.
  - The MTP port equals vLLM's semantics on tiny weights, checked against an independent build as EAGLE3's was.
  - The real model's perplexity is sane on the box CPU.
- **F2, kernels.** Every new kernel against its host twin, bitwise where the arithmetic is the same: the HC chain,
  PLE ids (exact integers), the gather, the indexer scores, the selection including exact ties and the
  512 / 513 boundary, the route at 512 / top-10 including a tie at the cut, the combine, the sigmoid gate. The
  sparse attention against fp64 at cosine ≥ 0.99999, and its eager variant against the reference's bf16 chain.
- **F3, golden** (greedy, tie-aware, docs/14's undetermined-row rule), against the engine-format reference (int4
  experts and int8 PLE rows dequantised, the rest as loaded). On synthetic real-width checkpoints and on the real
  model truncated with `--layers N`.
  - **Prompts:** short (< 2052 positions: QSA dense) and long (4k, 8k, 32k: the selection active).
  - **Selection gate S:** per QSA layer and row, the engine's block set equals the reference's, except where the
    512th and 513th scores are within τ (decision 3, from the measured gap distribution as `B70_KOL_TIE_TOL`).
  - **Routing:** top-10 sets equal except near-ties at the 10th / 11th (as R2).
  - **Injected run:** long prompts also run with the reference's selections fed in (a debug input), which isolates
    the attention arithmetic. It must pass the determined-row gate on every row. The free run passes S and the
    greedy gate on rows with no near-tie in any layer, and reports the others.
- **F4, determinism.** Replay bitwise; prefill splits at multiples of 64 bitwise (a row's selection depends only on
  the cache up to it); chunked = whole on the injected selection; prefix-cache restores at block ends bitwise the
  cold run; `--pp 2` bitwise `--pp 1`.
- **F5, MTP.** Verify rows at M = K + 1 bitwise M = 1 (spec 8 M2); greedy lossless (M3); acceptance recorded on
  the golden set, A4 and code (both `pre_fc_norm_hidden` forms).
- **F6, quality.** The 21q table: our g64 vs the bf16 original and Intel's g128 vs the bf16 original (KL, top-1,
  per set); int8 PLE vs bf16 PLE KL. After spec 22: A4 against the bf16 reference, passkey at the context chosen.
- **Speed (recorded, no bars before P0):** launches per token (estimated ~770 at 48 layers, Qwen3.8's 774; HC
  launch fusion is the first lever if the floor binds), decode and prefill on the truncated model; the full-model
  rows are spec 22's.

## 8. Stages

- **21a. Facts and CPU reference** (Mac, then the box CPU).
  - **Built:** `qwen4exp_ref.py` (transformers' text model, layer-streamed with `stream.py`, dequantising Intel's
    g128 / our g64 experts and int8 PLE rows; the PLE shards read by mmap, 16 rows a token; the MTP head ported
    from vLLM); a facts sheet `docs/probe-qwen4exp-<date>.md` (every rounding point of §2, the oracle's experts
    implementation, torch's CPU `topk` order on tied block scores, `tokenizer.json` vs Qwen3.8's, the original's
    card); golden sets (short and long prompts: ids, logits, per-layer routes, selections, gaps); teacher-forced
    routing traces on agentic transcripts for spec 22's P0.
  - **Mac gates:** F1 on the tiny model, the MTP port test.
  - **Box:** F1's perplexity, the golden dumps, the traces (Intel's checkpoint, 181 GB, layer-streamed; the bf16
    original when needed).
- **21b. Descriptor, loader, formats** (Mac; box for loads).
  - **Built:** `Qwen4ExpDesc`; `load_qwen4exp` (Intel's g128 experts through the expansion, bf16 dense arms, our g64
    when it exists, the PLE int8 converter and the host-USM table, the I64 checks, placement over two cards,
    `--layers N`); `make_synth.py` (N layers at real widths, our format, a reduced-prime PLE table - the gather is
    the same kernel); memory planning.
  - **Mac gates:** host tests (repack word for word, refusals by name, the plan at N = 4 / 18 / 38).
  - **Box:** the loads, a 51.8 GB pinned table, the planner's N.
- **21c. Decode** (built blind; box for every first compile).
  - **Built:** the kernels of §4.1-4.4, the decode list, two cards.
  - **Mac gates:** F0, F2 on the host twins and the Mac GPU (indicative), Level Zero / OpenCL syntax.
  - **Box:** F2 on the card, F3 and F4 on decode (synthetic, then truncated real).
- **21d. Prefill** (built blind).
  - **Built:** dense flash below 2052, the indexer GEMM and row-wise selection, sparse flash, grouped MoE at 512
    with `ushort` ids, the HC GEMMs, PLE over a chunk; `l0` only.
  - **Mac gates:** host twins, variant names.
  - **Box:** F3 and F4 on prefill.
- **21e. Serving and MTP** (built blind).
  - **Built:** the server path (template, XML tool calls, reasoning; Qwen3.8's parser), prefix-cache snapshots (KV,
    compressed keys, GDN state, PLE history and conv ring, the indexer tail rings), the MTP head and draft / verify,
    the sampling defaults (decision 11).
  - **Mac gates:** template byte-identity (the hash), tokenizer equality, snapshot layouts.
  - **Box:** F4 restores, F5.
- **21q. Our quantisation** (a big machine; parallel to 21b-21e). The recipe of §5 as a script on Kolibri's 20b
  pattern: calibration (agentic coding and tool calls, chat, code), per-expert coverage over 512 x 48, RTN then
  AutoRound, F6's table, publish.
- **Box queue rows** are assigned as each stage is built (the queue's next free row is 30).

## 9. What runs blind and what needs the box

| blind (the Mac) | the box |
|---|---|
| the reference on the tiny model, the MTP port, the facts sheet's source readings | every kernel's first compile (ocloc) and run |
| descriptor, loader, planner, synthetic checkpoints (small N), host twins of every kernel | real-weight golden gates: the reference needs the ~360 GB bf16 original or Intel's 181 GB (incl. the 102 GB PLE), on the box CPU, layer-streamed |
| Level Zero / OpenCL syntax, kernel command lines, indicative Mac-GPU runs | loads of 18-38 real layers, the 51.8 GB pinned PLE table, two cards |
| template and tokenizer equality | routing traces on real weights (spec 22 P0) |

## 10. Decisions

1. **Reference:** transformers 5.19.0 for the main model, vLLM for the MTP head (transformers drops `mtp.*`).
   Ruled.
2. **HC precision:** bf16 (ruled). int8 / int4 HC (~0.6-0.9 GB a token) **(decide)** after its quality is measured.
3. **QSA ties:** exact ties to the lower block index (ruled with the design). The near-tie tolerance τ for gate S
   **(decide)** from 21a's measured gap distribution.
4. **MTP `pre_fc_norm_hidden`:** vLLM's single RMS over 10240 first (ruled); the per-stream form **(decide)** by
   F5's measured acceptance.
5. **MTP draft attention:** steps after the first reuse step 0's selection (vLLM). Ruled.
6. **(decide) Our quantisation run:** where the 360 GB run happens (a rented big machine, as 20b's decision 1), and
   whether the MTP head's experts are quantised (RTN or a second AutoRound pass: 1.34 GB instead of 5.03 GB) or stay
   bf16.
7. **(decide) The PLE table:** the scale format (f32 1.28 GB or bf16 0.64 GB per 320M rows), and whether the
   conversion is a one-time file beside the checkpoint (proposed; our export can carry it) or happens at every load
   (102 GB read and quantised).
8. **KV:** bf16 first, int8 later. Ruled.
9. **(decide) Context target:** shared with spec 22's expert cache - every GB of KV is a GB of cache (spec 22
   decision 5).
10. **Decode attention:** one sparse kernel at every depth (proposed; the identity list below 2052), or v2 below
    and the sparse kernel above **(decide)** if P0 shows v2 is faster at short depth.
11. **(decide) Sampling defaults:** `generation_config.json`'s (sampled, T 1.0, top-p 0.95, top-k 20, as Kolibri's
    server applies them) or the Qwen family's greedy-unless-asked.

## 11. Out of scope

- The vision tower and video; the fp8 KV / fp8 indexer options.
- Third-party quant formats beyond Intel's interim checkpoint (one format, our AutoRound g64).
- Running the full 512-expert model before spec 22.
- CPU expert compute (spec 22 rules it out too).
- Tensor parallel.
- A GGUF-derived layout of any tensor: a GGUF port's v-head map `h % 16` reflects llama.cpp's reordered weights,
  not the safetensors order (`h // 3`).
