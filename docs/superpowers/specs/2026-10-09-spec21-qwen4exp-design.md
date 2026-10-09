# Spec 21 - the `qwen4_exp` family (Qwen3.8-Flash-Next) in the engine

**Status:** design, 2026-10-09; the operator approved the design on 2026-10-08 (§0 records what was ruled).
Open decisions are marked **(decide)**. Nothing is built.
**Plans (2026-10-09, after the operator's approval that day; 21a built on the Mac, §12; 21b built on the Mac, §13; 21c built on the Mac, §14; 21d built on the Mac, §15):** `docs/superpowers/plans/2026-10-09-spec21{a,b,c,d,e,q}-*.md` - 21a reference, 21b descriptor / loader / formats, 21c decode, 21d prefill, 21e serving and MTP, 21q our AutoRound run; box queue rows 30-34.

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

## 12. 21a as built (2026-10-09; the Mac - nothing on the box yet)

Plan: `docs/superpowers/plans/2026-10-09-spec21a-qwen4exp-facts-and-reference.md`, branch
`spec21a-qwen4exp-reference`. Facts: `docs/probe-qwen4exp-2026-10-09.md`. Box validation: queue row 30
(box CPU only). Every number is measured on the Mac unless marked.

- **The version.** transformers **5.19.0** runs beside `agnes-ref-img` (`tools/oracle/qwen4exp_env.sh`:
  `pip --no-deps --target`, tokenizers 0.23.2; the image keeps 5.15.0). 5.19.0 loads the tiny model **as
  shipped**: `qwen_sparse_attention` is remapped by `PreTrainedConfig.__post_init__`
  (`configuration_utils.py:96`, `:398`) before `validate_architecture` - the plan's rewritten-config fallback
  was not needed. Default experts implementation: `grouped_mm` (its combine sums the 10 weighted expert rows in
  the router's topk slot order, fp32 accumulation, one rounding - §4.4's "combine order" for the engine).
- **Built:** `tools/oracle/qwen4exp_ref.py` (transformers' own model layer-streamed; the QSA block keys cached
  beside the cache; the PLE table by mmap from the bf16 shards or 21b's int8 file; routed experts lazy in the
  fused bf16 / int4 g128 / int4 g64 / per-expert forms; `run`, `ppl`, `hfcheck`, `trace`, `facts`;
  `expected_names`, `PleTable`, the restated ops), `qwen4exp_mtp.py` (the head, vLLM's semantics),
  `qwen4exp_facts.py`, `qwen4exp_make_tiny.py`, `qwen4exp_prompts.py`, `qwen4exp_tiny_check.py`, the golden
  prompts `tests/golden/prompts/q4exp_{short,4k,8k,32k,agentic}.ids` (+ `q4exp_agentic.json`),
  `tools/box_validate/qwen4exp_oracle.sh`, `data.sh`'s q4exp keys, queue row 30.
- **F1 on the Mac (all bitwise):** the streamed port = transformers 5.19.0 un-streamed on the tiny model in
  **bf16 and fp32**, prompts of **40 and 2100** ids + 8 cached decode steps (logits and every layer's 4-stream
  residual), 2100 ids in chunks of 512; the indexer cache = transformers' per-query recomputation at rows
  2047..2060 (selection masks, attention outputs, the recorded 512th / 513th gaps) in prefill and in decode;
  int4 g128 (Intel's form) and g64 (ours, dense linears too) = transformers on the dequantised twin; the
  original's fused experts; the int8 PLE file = transformers on the dequantised table; `--layers 4` of an
  8-layer tiny; the HC chain, PLE ids, sigmoid gate, `h // 3` traps; `expected_names` = the real indexes
  (1325 / 223947 text tensors). The MTP head = an independent build bitwise (both `pre_fc_norm_hidden`
  forms). Its points each have a test: unit injection, `fc_hidden` per stream, the reused list, the pre-mixer hand-off, the norm forms differ.
- **Departures from the plan, and findings:**
  1. **Decision 5 is vLLM's opt-in, not its default** (`index_share_for_mtp_iteration`, absent from the
     checkpoint config; `config/speculative.py:442-444, 845-864`, `llm_base_proposer.py:578-613`). The port
     builds both (`chain(share_sel=True)` the ruled form); the operator may want to re-read decision 5. In the
     shared form a later step attends exactly step 1's list - neither its own key nor earlier draft keys.
  2. **torch's CPU `topk` does not prefer the lower index on exact ties** (the lower index won 29 / 50 planted ties at the QSA cut and 25 / 50 at the router's; deterministic across calls): the engine's
     rule (ties to the lower id / block, §4.2, §4.4) and the reference differ only on exact ties; the
     recorder flags them (`route.gap == 0`, `qsa.gap == 0`) so the gates treat those rows as undetermined. The
     QSA scores are relu sums, so a block with all four head products negative scores exactly 0: on the tiny
     model every 512th / 513th gap at rows 2051..2060 is an exact 0 (the cut inside a tie of zeros) - how often
     that happens on the real model is row 30's `==0` count, and those rows are undetermined for gate S.
  3. **The tokenizer is not Qwen3.8's**: same vocab / merges / added tokens, but the pre-tokenizer's split
     regex adds `\p{M}`; Intel's checkpoint ships Qwen3.8's `tokenizer.json`. The committed prompts are
     unaffected (prose / code / cjk and long32k re-encode identically); 21e must use the original's file and
     implement `\p{M}`.
  4. **`--layers N` needs N >= 4 for a cached run**: transformers 5.19.0's `DynamicCache` takes the sequence
     length from an attention layer (`cache_utils.py:1566`); `run` / `ppl` / `hfcheck` / `trace` refuse
     N < 4 by name, and the truncation test runs `layers=2` uncached (bitwise) and `layers=4` cached on an
     8-layer tiny (the downloaded tiny has 4 layers: N = 2 cannot run cached in transformers itself).
  5. **The golden layout keeps activations and logits for a tail window** (`--act-tail 256`,
     `--logits-tail 1024`; every row's routes, QSA gaps, selections past 2050, PLE ids and NLL): the full set
     at 32k would be ~50 GB of `H.L*` alone (derived). Prompts > 4096 ids are prefilled in chunks of 2048 -
     transformers' own chunked forward (`test_chunked_prefill_equals_hf`).
  6. **The trace format adds what spec 22 P0.8 (REAP) needs**: `p` is the router's pre-cast fp32
     renormalised value and `onorm` each routed expert's output norm (read from `grouped_mm`'s own down
     projection), plus the MTP head's `mtp_p` / `mtp_onorm`.
  7. Mac containers capped at 8 GB / 4 CPUs (the operator's rule, not the plan's 28 GB); the tests keep the
     last 48 logits rows of a prompt chunk (a 2100-row fp32 logits block is 2 GB) and compare every row's
     residual. The tiny model was copied into git-ignored `oracle-out-q4exp/tiny` (already downloaded).
  8. The box site lives in `<tree>/oracle-out-q4exp-site` (the container mounts only the tree), not
     `$DATA/q4exp-site`; `qwen4exp_oracle.sh` adds `tests` and `hfcheck` modes.
- **Pending (box CPU, row 30):** the perplexity, `hfcheck` on real weights, the golden and `--layers 4 / 18`
  sets, the gap distributions (decision 3's tau, R2's MoE tolerance), the per-prompt times and RSS, the
  traces. Times are ESTIMATED in `qwen4exp_oracle.sh`'s DRY_RUN table (~10 min for `q4exp_short` to
  ~1.5-2 h for `q4exp_32k`, ~3-4 h for the `intel` set).

## 13. 21b as built (2026-10-09; the Mac - nothing on a card yet)

Plan: `docs/superpowers/plans/2026-10-09-spec21b-qwen4exp-descriptor-loader-formats.md`, branch
`spec21b-qwen4exp-loader`. Box validation: queue row 31. Every byte count below is **derived** (the
descriptor and the checkpoint headers' shapes); the tests assert the formulas and print the values.

- **Built:** `model::Qwen4ExpDesc` (`src/model/qwen4exp.*`: the published model, `check_qwen4exp_config` /
  `qwen4exp_desc` holding config.json key by key, `Q4Placement`); `loader/qwen4exp_layout.h` (header-only:
  every allocation's size and every block offset - the planner's and the loader's one formula, and 21c's
  kernels' host half); `loader/qwen4exp_repack.*` (the host half: `q4_forms`, `q4_expected_names`,
  `Q4Checkpoint`, `q4_rope_table`); `loader/qwen4exp_ple_hash.h` (the PLE hash in exact integers),
  `loader/qwen4exp_ple.*` (the int8 file, its checks, the host-memory rule) and `qwen4exp_ple_usm.cc` (the
  pinning); `loader/qwen4exp_loader.*` (`load_qwen4exp`); `runtime/qwen4exp/qwen4exp_sizes.*` (the plan);
  `tools/quantize/qwen4exp/{make_synth,ple_int8,check}.py` + `test_qwen4exp_quant.py`;
  `tools/oracle/qwen4exp_ple_fixture.py` -> `tests/loader/qwen4exp_ple_fixture.h`; host tests
  `qwen4exp_test`, `qwen4exp_repack_test`, `qwen4exp_ple_test`, `qwen4exp_plan_test`; `qwen4exp_synth_host_test`
  (the host half over make_synth's files - an addition to the plan); the card test
  `qwen4exp_load_checkpoint_test` (`qwen4exp_load_synth_{ours,intel}_test`, `qwen4exp_load_intel_layers_test`);
  `qwen4exp_oracle.sh` modes `synth-ckpt` / `ple-int8`; queue row 31. No kernel (kernel_cmdlines +0 / -0 / ~0).
- **The layouts and their bytes** (`qwen4exp_layout.h`; `qwen4exp_repack_test` prints them): routed experts
  int4 layout-1, block e at `e x 1,740,800` (gate||up `cols_interleave16` {2560, 1280}) and `e x 870,400`
  (down) - **1,336,934,400 B a layer**, each expert one contiguous range per allocation (spec 22's host mirror
  and indirection table address exactly `q4_gate_up_offset` / `q4_down_offset`); a gated residual's block
  (down||inject bf16 tiles {10240, 336} - 324 rows used - , up {320, 10240}, the `(1 + w)` norm fp32):
  13,475,840 B, **26,951,680 B a layer**; the final mixer 13,148,160 B; the router {2560, 528} (row 512 =
  `shared_expert_gate`, 513..527 zero) 2,703,360 B; the shared expert one allocation (ours: two layout-1
  blocks, 2,611,200 B; Intel's: bf16 tiles, 9,830,400 B); dense projections int4 GPTQ layout 0 (qkv||z S 1,
  q||gate||k||v with k at column 12288 and v at 12800, out / o S 4 / S 2 / S 4 PROVISIONAL) or bf16 tiles; the
  GDN small block `make_small_layout(2560, 10240, 48)` 164,480 B; the QSA small block (q / k norms, the
  indexer's q / k norms, fp32 `(1 + w)`) 3,072 B; the PLE layer's block (key||value tiles {2560, 12800},
  three `(1 + w)` norms, the conv taps fp32) 65,822,720 B; the int8 head **636,692,480 B**. A layer in Intel's
  forms: GDN **1,492,583,040 B (1.493 GB)**, QSA 1,479,314,432 B; ours: GDN 1,400,658,560 B. The MTP head
  (its QSA layer with bf16 dense / shared arms and int4 g64 experts, the fc block, its own mixer):
  1,518,728,192 B.
- **The forms detection** (`q4_forms`): three groups read from the names all-or-nothing - dense (layer 0's
  `linear_attn.in_proj_qkv`), shared (layer 0's `mlp.shared_expert.gate_proj`), the MTP head's experts - and
  the routed experts' group from their scales' rows (64 / 128, one size for the checkpoint, = the config's
  `group_size`). A mixed group, a g64 expert among g128, a bf16 or the original's fused routed expert: each
  refused naming the tensor. g128 goes through `LinearSrc::classify`'s exact expansion, expert by expert
  (Review Focus 1: every block word for word, every weight of sampled columns dequantised equal).
- **The interim MTP head:** both checkpoints ship `mtp.*` bf16 (per-expert `.weight` experts); `repack_mtp`
  RTN-quantises them with `rtn_int4_g64` (spec 15e's rule, block for block equal in the test) until
  decision 6.
- **The PLE file (decision 7's proposal as built):** `ple_int8.py <snapshot> <out> [--scale bf16|f32]`
  (default **bf16**: 51.84 GB on the real table, f32 52.48 GB, derived): per head `ple.h<h>.q` I8
  `[prime_h][160]` and `ple.h<h>.s` `[prime_h]`, spec 9's row rule (`s = max|w| / 127` fp32, bf16 rounded
  once; a zero row s = 0), the three I64 constants checked against the formula and copied, one file per head +
  an index, metadata `{source, rule: row-int8-spec9, scale}`; byte-equal to 21a's `write_ple_int8`. The engine
  reads `<snapshot>-ple-int8/` or `$B70_Q4_PLE`, refusing a missing file naming the converter's command.
  `load_q4_ple`: the file against the descriptor (heads, width, the I64 tensors = `qwen4exp_ple_hash.h` =
  transformers 5.19.0's own builders and forward on 64 sequences x 2 tables, `qwen4exp_ple_test`), the
  host-memory rule (table + 16 GiB <= MemAvailable, refused naming both), then **16 + 16 host-USM ranges**
  (q and scales per head, 2 MiB aligned), a tag written at every 2 MiB page of every range and read back
  before the data (the alias check), the rows copied, sampled rows (every 4096th, the last) compared with the
  file, each page's first word recorded (`page_words`, for 21c's device read), the device pointer table
  `ptrs` u64 [32] on the device holding the PLE layer.
- **The memory plan** (`runtime::qwen4exp`): weights + persistent state (KV + compressed indexer keys
  25,344 B a position over 12 QSA layers - 6.64 GB at 262144; GDN state 113.2 MB for 36 layers; conv rings;
  the PLE state on the PLE layer's device); `pp_split` = `runtime::pp_balance` over `pp_layer_bytes`;
  `layers_that_fit` = the largest N whose truncated model fits under some placement. On 32.53 GB cards with a
  1.5 GB reserve and the int8 head: **Intel's forms N = 19 on one card and 39 on two at 32768** (18 / 38 with
  the MTP head; 18 / 38 at 131072); ours 20 / 41 (19 / 40 at 131072). 21c's decode scratch will lower these a
  little. The full model is refused on two cards by `require_fits` naming its bytes, each card's capacity and
  spec 22; `load_qwen4exp` runs the plan before its first allocation.
- **Mac gates (measured):** the four host tests PASS (`qwen4exp_repack_test` ~40 s: two 4-layer real-width
  checkpoints written and read back); `test_qwen4exp_quant.py` 6 / 6 in `agnes-ref-img` (8 GB / 4 CPUs) with the
  5.19.0 site; both synthetic checkpoints made on the Mac (`make_synth.py --layers 4 --mtp`, PLE base 1000:
  ours 13.52 GB / 20,277 tensors in 1650 s; Intel's form 13.68 GB / 20,227 tensors in 748 s) and `ACCEPTED` by
  `check.py --ple`; the C++ host half over those very files (`qwen4exp_synth_host_test`, label `checkpoint` -
  run by hand on the Mac, 425 s: both forms' names both ways, every layer and the head repacked to the layout's
  bytes, the 128 shards skipped, the PLE files = the descriptor, 0 unconsumed); Level Zero syntax PASS on every
  new source; `tools/mac_check.sh --base main --quick` exit 0 (host 77 pass / 0 fail, kernel_cmdlines 630
  variants +0 / -0 / ~0).
- **Departures from the plan:**
  1. `tests/model/qwen4exp/config.json` and `synth4.json` are not vendored (the plan's fallback): the
     checkpoint's licence is `qwen-community-1.0` and its text was not read, so `qwen4exp_test` builds both
     configs in code (every structural key with its value).
  2. The hash functions live in a header-only `loader/qwen4exp_ple_hash.h` (included by `qwen4exp_ple.h`) so
     the host-only planner shares them without linking the loader; `load_q4_ple` is its own object
     (`qwen4exp_ple_usm.cc`) so a host test that uses the file links no device call.
  3. The tag check writes its tags BEFORE the data (tags and rows cannot share the pages); the data's page
     words are then recorded in `Q4PleTable::page_words` for 21c's device read. `Q4PleTable` also records
     `rows_checked`, `mem_before` / `mem_after`, `seconds`.
  4. `Q4Checkpoint::final_mixer()` / `mtp_fc()` / `mtp_mixer()` return byte blocks; `ple_constants()` and
     `skip_ple_shards()` added; `Q4HostWeight` carries a GEMV weight's device form; `Q4DevicePart` gains
     `mtp_mixer`; `Q4Layer` gains `shared_down_offset`; `DevicePlan` uses `MemoryComponents::kv` (KV + indexer
     keys) and adds `first` / `end` / `state` / `mtp` / `whole`; the plan adds `fits`, `truncated`,
     `placement_for`, the per-layer state helpers.
  5. `q4_expected_names` excludes the PLE table's bf16 shards (skipped by design: their count is the export's
     split, not the model's).
  6. `is_qwen4exp_model_type` also accepts `qwen4_exp_text` (a text-only export's top level).
  7. The synthetic checkpoints follow Kolibri's layout, `oracle-out-q4exp-synth/{ours,intel}/ckpt` with the
     PLE file beside as `ckpt-ple-int8` (the loader's default name), so 21c's golden sets can live in
     `{ours,intel}/`. `make_synth.py` reads the original's config.json from `--tokenizer` (or `--config`); its
     Intel-form `extra_config` is the export's bits-16 patterns, not Intel's 2340 per-layer entries.
  8. `r31.ple_convert` converts Intel's 128 bf16 PLE shards (the original's rows as shipped) - no 360 GB
     download; `r31.synth` needs only the original's small files (new `have` key `q4exp_orig_small`).
  9. `load_qwen4exp` also refuses a `max_len` that is not a multiple of 256 (kMaxLenQuantum).
  10. `test_qwen4exp_quant.py`'s fixtures shrink the experts (16) and the vocabulary (1024) through test-only
      `make()` arguments; containers ran at 8 GB / 4 CPUs (the operator's cap), not the plan's 28 GB.
  11. `r31.ple_kl` runs 21q Task 3's `evaluate.py --ple-only` and SKIPs until that file exists.
  12. The indexer tail is planned at 4 slots (`kIdxTail`, this plan's form); 21c records 8.
  13. `tools/box_validate/qwen4exp_oracle.sh` (21a's) gains the modes `synth-ckpt` and `ple-int8` (21c adds its
      golden `synth` mode); row x's `x.rest` leaves the `qwen4exp` label to row 31.
  14. Added: `qwen4exp_synth_host_test` (the C++ host half over make_synth's own files, label `checkpoint`, run in
      `r31.load` before the uploads) - the Python writer and the C++ reader held to each other on real files.
- **What row 31 must prove:** the uploads (both synthetic forms with the MTP head, Intel's 18 layers) with 0
  unconsumed and every part's bytes = the plan; the device read-backs = the host repack at the edges; the
  51.8 GB table pinned as 32 ranges with no alias, its time and MemAvailable before / after (spec 22 P0.5's
  start); the pointer table; the planner's N lines.

## 14. 21c as built (2026-10-09; the Mac - nothing on a card yet)

Plan: `docs/superpowers/plans/2026-10-09-spec21c-qwen4exp-decode.md`, branch `spec21c-qwen4exp-decode`. Box
validation: queue row 32. Every launch count and byte count below is **derived** (the tests assert the formulas);
the Mac results are **measured**.

- **Built:** `tests/kernels/qwen4exp_ref.h` (namespace `q4ref`: the host twin of every chain below) with
  `tools/oracle/qwen4exp_fixture.py` -> `tests/kernels/qwen4exp_fixture.h` (21a's restated ops and transformers
  5.19.0's modules; the PLE layer's restatement held bitwise to `Qwen4ExpTextPLELayer` first) and
  `qwen4exp_ref_test`; the kernels `src/kernels/qwen4exp/q4_{hc,ple,qsa,qsa_attn,qsa_attn_eager,moe}.cl` and their
  host half `src/kernels/qwen4exp_kernels.h` (`kernels::qwen4exp`), `qwen4exp_variant_names_test`,
  `qwen4exp_kernels_test` (F2 on the card; `--bench-attn`, `--ple-rate`), the Mac driver
  `tools/mac/clrun/qwen4exp_run.cc`; `runtime::qwen4exp` - the decode scratch, the tap, the hand-off and the launch
  counts in `qwen4exp_sizes`, `Qwen4ExpBuffers`, the capture, `Qwen4ExpEngine` (one card and two); `b70-decode`'s
  `model_type qwen4_exp` dispatch (`src/cli/qwen4exp_decode.h`), `b70-serve`'s refusal; the card tests
  `qwen4exp_decode_test`, `qwen4exp_golden_test`, `qwen4exp_partial_test`, `qwen4exp_pp_test`; `cli_reject_qwen4exp_*`;
  `qwen4exp_oracle.sh synth` (the synthetic golden sets); queue row 32.
- **The list per layer** (`runtime/qwen4exp/qwen4exp_capture.h`). One gated residual = `q4_hc_combine_norm`
  (folds the PENDING block output of the block before - `_E` the embedding at layer 0, `_S<S>` a mixer GEMV's
  split-K slices, `_Y` the MoE's `y`, `_X` nothing - then the grouped `(1 + w)` norm), `gemv_bf16` {10240, 336}
  (down's 320 rows and `block_inject`'s 4), `q4_hc_up_mix` (silu(/4), the up linear, sigmoid, the mean of the 4
  streams; `inj`). A GDN layer is **15** launches (hc, qkv||z GEMV, a||b, Qwen3.8's `gdn_step_M1`,
  `prep_gated_head_M1_SIG`, out_proj, hc, router `gemv_bf16` 528, `q4_route`, `q4_moe_gate_up`, `q4_moe_down`); a QSA
  layer **19** (hc, q||gate||k||v GEMV, the indexer `gemv_bf16` 640, `attn_prep_M1_Q24KV2`, `q4_qsa_prep` /
  `_score` / `_select`, `q4_qsa_attn` + `q4_qsa_reduce`, o_proj, the MLP side's 8) - **18** under
  `B70_Q4_ATTN=eager` (one `q4_qsa_attn_eager` for attn + reduce); the PLE layer **+4** (a combine-only `_Y_NN`, the
  gather, the key||value `gemv_bf16` 12800, the block; its attn side's combine is `_X`); the head **1 + 6** (the
  embedding; the final mixer's combine `_Y` / down {10240, 320} / up_mix, lm_head, argmax x 2). **779 launches a
  token at 48 layers on one card** (1 + 36 x 15 + 12 x 19 + 4 + 6), 767 eager, 755 with injected selections, **75 at
  `--layers 4`**; two cards **+1** (device 0's `_Y_NN`: the materialised H crosses, 20,480 B; device 1 starts with
  `_X`) and peer +2 more - +0 when the cut is the PLE layer (device 1's PLE prologue then folds nothing).
- **The 8-slot tail ring** (plan Review Focus 3; §4.2 said 4): the open block's raw keys at slot `p % 8`; a
  completing row forms its block's compressed key ONCE, from this launch's rows straight from the indexer GEMV and
  older rows from the ring - with 4 rows a launch the writes `pos .. pos + 3` and the reads `pos - 3 .. pos - 1` are
  seven consecutive positions, distinct mod 8. Measured on the Mac at M = 1 and at M = 4 (not a box binary yet).
- **Reused** at this family's shapes from new CMake lines: `attn_prep` (`_Q24KV2`, QKV_S 2 int4 / `_S1` bf16 - the
  q / k `(1 + w)` norms, the partial RoPE, the KV write, the gate), Qwen3.8's `gdn_step_M1` and argmax binaries by
  name, `gemv` / `gemv_bf16` / `gemv_i8w` at 33 new shapes in all (kernel_cmdlines **+33 / -0 / ~0** with K2 and
  Kolibri on), `embed_gather_M1_D2560`, `attn_v2_M1_T32_Q24KV2` (decision 10's bench only). **One existing source
  changed:** `prep.cl`'s `-DGDN_GATE_SIGMOID` (`prep_gated_head_M1_SIG`: `GATED_ACT(z)` is `silu_f32(z)` otherwise);
  all 98 existing `prep.cl` variants preprocess token for token as main's (`clang -E -P`, measured) - G0's sha256 is
  the box's check (r32.k0); if any moved, the variant moves to a copy in `src/kernels/qwen4exp/` and prep.cl is
  restored.
- **exp:** every sigmoid / SiLU in the new kernels (and `_SIG`) is `1 / (1 + exp_torch(-x))` / `x / (1 + exp_torch(-x))`
  - Sleef's expf u10 step for step, k2_attn_eager.cl's - so the kernels, `qwen4exp_ref.h` and torch agree bit for
  bit at those points.
- **The selection** (`q4_qsa_select`, decision 3): an MSB-first radix select over the scores' bits (every score >=
  +0) with a local integer histogram, the ties at T to the lowest blocks through an exclusive scan over contiguous
  per-lane chunks; the list ascending (blocks expanded, then the tail), the count at word 2052 of a 2064-word row,
  the 512th / 513th scores as the diagnostic. Below 513 complete blocks the identity list (one kernel at every
  depth, decision 10's proposal). The device's tie rule is tested against `q4ref::qsa_select`'s `(score desc, block
  asc)` sort, not torch's topk (which broke the fixture's planted ties its own way on 15 blocks).
- **The expert address** spec 22 replaces: `q4_moe.cl`'s `Q4_EXPERT_GU(base, id)` / `Q4_EXPERT_DN(base, id)` - the
  only lines that turn a route row's id into weights.
- **Known deviations (recorded):** flash keeps fp32 probabilities (the eager twin rounds them, the reference's
  chain); the mean over the 4 streams is `(((p0 + p1) + p2) + p3) / 4` (torch's order for 4 terms, exact in practice
  - the fixture agrees bitwise); `attn_prep`'s q keeps its fp32 RoPE output (attn.cl's documented <= 1-op
  difference; the eager kernel rounds it to bf16 first); the router's softmax sum is a pairwise tree (torch's is its
  own: p differs by an ulp, the ids and weights did not on the fixture).
- **Departures from the plan:**
  1. qkv||z int4 is `gemv_M1_K2560_N16384_S1_L0` - 21b's loader writes layout 0 - not the plan's `_L1`.
  2. The bf16 arm's prep is `attn_prep_M1_Q24KV2_S1` (`kernels::attn_prep_s1_variant`'s spelling).
  3. `q4_qsa.cl`'s prep / score / select are ONE binary `q4_qsa_M1_T512_W1024`; eager attention is one launch.
  4. No `q4_expert.h`: no kernel in this tree `#include`s (ocloc's include resolution unproven blind) - the macro
     pair lives in `q4_moe.cl`.
  5. The combine is 21a's pinned `grouped_mm` order (rank order, fp32, one rounding; the shared expert's gated term
     added after), not the plan's "eager ascending id" text.
  6. The PLE state is a 16-slot id ring + a 16-slot conv ring (21b planned 2 ids + 9 rows); the tail ring 8 slots;
     21b's plan test numbers updated.
  7. `data.sh` gains `oracle_q4exp_synth_golden` instead of widening `oracle_q4exp_synth` (r31 needs only the
     checkpoints); `golden_common.h` reads I64 (`ple.ids`).
  8. `--layers N|auto` is required for `qwen4_exp` (the 4-layer synthetic too); the CLI refusal tests run over a
     self-written `model_type` stub (`tests/model/qwen4exp/config.json`: §13 departure 1 still holds).
  9. The fixture checks every step bitwise from torch's own previous output (the plan's "every case bit for bit"):
     the GEMVs' fp32 order is torch's, held to one ulp - measured 0 ulps apart on the fixture.
  10. The injected selection rides a second capture per device (built on first use), as the plan's interface says.
  11. Mac GPU caps (256 lanes): `qwen4exp_run` builds `q4_qsa` at `SEL_WG=256` and `q4_moe` at `DN_KS=1` (the B70
      binaries: 1024, 2).
- **Mac gates (measured):** `qwen4exp_ref_test` PASS (HC, PLE ids and block over 10 positions, indexer q and keys,
  scores, combine, the sigmoid gated norm all bitwise against torch; the route = the ruled order with torch's topk
  apart on 7 tied slots; Review Focus 1 - three layers in the reference's and the fused order bitwise, inj 0 / 2 -,
  2 and 3); `qwen4exp_variant_names_test`, `qwen4exp_plan_test` PASS; `qwen4exp_run` on the Mac's GPU (Intel UHD
  630): 0 disagreements - every portable kernel exact but the scores (1 fp32 ulp, Apple's divide), route p10 / p11
  (1 ulp), MoE h (1 bf16 ulp on 2 of 7040) and the random gated head (1 of 6144); Level Zero syntax of every new
  source, OpenCL syntax of every new variant.
- **What row 32 must prove:** every binary's first compile; F2 on the card (bitwise; flash at cosine 0.99999); G0
  with `prep_gated_head_M1` unchanged; F4 (75 launches, two runs bitwise, the injected run = the free run); F3 on
  both synthetics (tokens, routing, gate S, PLE ids; the injected run on every determined row); the partial forward
  on Intel's 18 layers within the proposed bars; two cards bitwise one; the PLE gather's host-USM rate (spec 22
  P0); decision 10's bench and the speed rows (Task 7).

## 15. 21d as built (2026-10-09; the Mac - nothing on a card yet)

Plan: `docs/superpowers/plans/2026-10-09-spec21d-qwen4exp-prefill.md`, branch `spec21d-qwen4exp-prefill`. Box
validation: queue row 33. Every launch count and byte count below is **derived** (the tests assert the formulas); the
Mac results are **measured**.

- **Built:** `tests/kernels/qwen4exp_pf_ref.h` (namespace `q4pf`: the 512-expert sort, gather, dequant, the combine as
  21c's `q4ref::combine` over read-back down sums, `dense_rows`, the indexer over a chunk with its ring, the sparse
  attention's tiled fp64 walk and the eager chain, the PLE over a chunk in three steps) and `qwen4exp_pf_ref_test`; the
  kernels `src/kernels/qwen4exp/q4_pf_{moe,attn,ple}.cl`, the spec 21d block of `qwen4exp_kernels.h` and of
  `src/kernels/CMakeLists.txt` (+30 binaries with K2 and Kolibri on, +31 without them),
  `qwen4exp_pf_variant_names_test`, `qwen4exp_pf_kernels_test` (F2 for the prefill on the card), `qwen4exp_run`'s prefill
  checks; `runtime::qwen4exp` - `qwen4exp_prefill.{h,cc}` (the walk, `Qwen4ExpPrefillScratch`, the head),
  `qwen4exp_prefill_gdn.{h,cc}` (`gdn_chunk_q4`), `qwen4exp_prefill_engine.cc` (`Qwen4ExpEngine::prefill`, replay, the
  injector, the block hook, two cards), the prefill sizes / launches / planner flag in `qwen4exp_sizes`, archive
  `b70_qwen4exp_prefill`; `qwen4exp_prefill_test` (modes `all`, `split`, `pp`, `gdn`), `qwen4exp_golden_test`'s `prefill`
  mode; `b70-decode <qwen4_exp> --prefill` / `--bench --prefill-length N [--prefill-chunk C]`; `cli_reject_qwen4exp_prefill_{int8,sycl,chunk}`
  (21c's `cli_reject_qwen4exp_prefill` removed); queue row 33.
- **The chunk** (`runtime/qwen4exp/qwen4exp_prefill.h`), C <= 2048 rows on one device's `l0` list:

  | piece | kernels | rounding chain | launches |
  |---|---|---|---:|
  | device 0 | `pf_embed_gather_D2560` (the rows into x; layer 0's combine `_E` repeats them) | copy | 1 |
  | a gated residual | `q4_hc_combine_norm_M2048_{E,S1,Y,X}` · the down‖inject bf16 slab {10240, 336} (one 512-column slab, 176 zero columns) + `pf_gemm_T0` (pitch 512) · `q4_hc_up_mix_M2048_I_D512` | 21c's HC chain; the down sums are DPAS's | 4 |
  | GDN mixer | qkv‖z 16 slabs (`pf_dequant_slab_K2560_N16384_L0` or `q4_pf_bf16_slab`) + GEMM · a‖b `pf_ab_proj_D2560` · `gdn_chunk_q4` (10) · out_proj 3 slabs (`k2_pf_dequant_slab_K6144_N2560` or bf16) + GEMM | Qwen3.8's chunked WY recurrence; the gated head `pf_gated_head_SIG` = `prep_gated_head _SIG`'s chain | 49 |
  | QSA mixer | q‖gate‖k‖v 13 slabs + GEMM · the indexer bf16 slab {2560, 640} + GEMM (pitch 768) · `pf_attn_prep_q16_Q24KV2` · `q4_qsa_prep` `_PF` · `q4_qsa_ring` · [sparse rows: `q4_qsa_score` + `q4_qsa_select` at M = C] · [rows <= 2050: `pf_flash_attn_Q24KV2`] · [rows >= 2051: `q4_pf_sparse_attn_Q24KV2`] · `pf_attn_gate` · o_proj 3 slabs + GEMM | the indexer's keys, scores and selection are decode's code; attention fp32 (flash) or the reference's points (eager) | 38 + 2 + 1 + 1 |
  | the MoE | router `pf_moe_router_K2560_N528` (decode's {16, 16}: rows bitwise decode's) · `q4_route_M1` on grid (1, C) · `q4_pf_sort` · `q4_pf_gather` · gate‖up 7 batches x (`q4_pf_dequant_gu[_shb]` + `pf_moe_gemm_K2560_N1280_SILU`) · down 4 x (`q4_pf_dequant_dn[_shb]` + `pf_moe_gemm_K640_N2560`) · `q4_pf_moe_combine` | the combine is `q4_moe_down`'s epilogue over sorted rows (rank order, fp32, one rounding) | 27 |
  | the PLE layer | `_Y_NN` (not when H landed) · `q4_ple_gather_M2048_*_PF` · key‖value 13 bf16 slabs + GEMM · `q4_pf_ple_gate` · `q4_pf_ple_conv` · `q4_pf_ple_ring` | 21c's block, statement for statement, cut in three | 31 (30) |
  | two cards | device 0 ends with `_Y_NN` (the chunk's materialised H, 42 MB, crosses by `copy`) | | + 1 |
  | the head (last chunk) | decode's final mixer `_Y` M1 over the last row, its down / up-mix, lm_head, argmax x 2 | decode's | 6 |

  A GDN layer is 84 launches, a QSA layer 73 + its attention's 2-4; **358 / 361 / 360** a chunk at `--layers 4` (all
  dense / straddling 2050 / all sparse), **3944 / 3968** at 48 layers, + 1 on two cards; the injected run -2 a QSA
  layer of a chunk with sparse rows. Asserted per chunk by the walk (`prefill_device_launches`).
- **The scores' departure from §4.2** (plan 21d's one deliberate one): the prefill runs decode's `q4_qsa_score` at M =
  C (fp32 FMA chains) instead of a DPAS GEMM, so a row's scores are bitwise decode's for the same query and keys and the
  selection is exactly decode's rule; cost ~8.6 GFLOP a QSA layer a chunk at 32k context, ~34 at 128k (ESTIMATED). The
  GEMM form is Task 5's recorded lever. Score rows `[2048][max_len / 4]` fp32 are the scratch term that grows (268 MB at
  131072).
- **`gdn_chunk_q4`:** `gdn.cc`'s ten launches over explicit buffers (no `PrefillScratch`, no `ModelDesc`), the tenth
  `pf_gated_head_SIG`; the scan / solve entries follow `gdn.cc`'s process-wide selectors. The GDN shape (16 / 48 heads,
  10240 conv channels, the small block's offsets) is Qwen3.8's, so Qwen3.8's binaries apply as built.
- **The prefill scratch** (`prefill_sizes`, every device the whole set): **1.489 GB at `--layers 18` / 32768**, 1.690 GB
  at 131072 - the 512 MiB weight batch, xg 197 MB (also the PLE gate's gated / gn rows), the GDN chunk scratch 194 MB,
  the score rows, the selection rows 16.9 MB a QSA layer (all QSA layers' last chunk: the gate S read-back), partials 134
  MB; + the prefill link on two cards (42 MB of H). Planned into `--layers auto` / `--max-len auto` when a run prefills:
  Intel's forms fit **18 layers on one card (max_len auto 70144) and 37 on two** at 32768 - `--layers 38` on two cards
  does not fit beside the prefill scratch (decode-only: 19 / 39).
- **Precision against the reference:** the dense rows run spec 6's flash (fp32 probabilities, `exp2`, P rounded to
  bf16 as the P·V operand), the sparse rows `q4_pf_sparse_attn` (fp32 online softmax at 1/16, natural `exp`); both
  against fp64 at the K1 bar. `B70_Q4_ATTN=eager` binds the `_EAGER` sparse build (the reference's points: s rounded twice,
  p once, o once) in prefill as well as decode's eager kernel; the dense rows keep the flash (no eager dense form).
  `pf_attn_gate`'s tail is `rne(rne(o) x sigmoid(gate))` with plain `exp` and an unrounded sigmoid, decode's
  `q4_qsa_reduce` `rne(rne(o) x rne(sigmoid_torch(gate)))`: <= 1 bf16 ulp apart (the plan's binary, kept).
- **Departures from the plan:**
  1. **Three existing sources gain a define, not one:** besides `pf_gated_head.cl`'s `GDN_GATE_SIGMOID`, `q4_qsa.cl`'s
     `QSA_PF` (the prep at M = 2048 must not write the 8-slot tail ring in its own launch - rows past pos + 4 overwrite
     slots the first rows read for the block straddling pos - so a new `q4_qsa_ring` launch writes the last min(8, C) raw
     keys after the prep; and the indexer rows arrive at the GEMM's pitch 768) and `q4_ple.cl`'s `PLE_PF` (the gather's
     tokens from the chunk's id buffer - Control holds 8 - and no id-ring write: `q4_pf_ple_ring` writes it). +1 launch
     a QSA layer a chunk. Every existing binary of the three: preprocessed source checked on the Mac (`pf_gated_head`,
     `_GK16V32`, `q4_qsa_M1`: byte for byte; `q4_ple_gather_M1_*`: token for token, whitespace moved).
  2. `q4_hc_up_mix` at M = 2048 is built at `UP_DOWN = 512` (the HC down GEMM's pitch) and named `..._I_D512`.
  3. The int4 qkv‖z slab is layout 0 (21c's as built), `pf_dequant_slab_K2560_N16384_L0` (the plan said L1).
  4. `q4_pf_moe` is one binary with the shared form as entry points (`q4_pf_dequant_{gu,dn}` int4, `_shb` bf16).
  5. **Chunks of 16 are not gated bitwise against 2048** (the plan's Review Focus 5 / Step 3, written from Kolibri's
     GDN-free walk): the GDN's 64-row WY sub-chunks sit at multiples of 64 from the chunk start, so a 16-row chunk
     re-partitions them. Chunks of 64 (and splits at multiples of 64) ARE gated bitwise; 16 is printed.
  6. `gdn_chunk_q4` against decode's `gdn_step` lives in `qwen4exp_prefill_test`'s `gdn` mode (it needs the runtime),
     not in `qwen4exp_pf_kernels_test`.
  7. The injected run on prefill takes its lists from a per-chunk callback (`set_prefill_injector`) into host-USM rows
     `[qsa layers][2048][2064]`; `read_prefill_selection` returns those rows when injected.
  8. Intel's two-card real runs use `--layers 37` (above); the prefill scratch is on every device (the PLE gather's
     ids too), not sized per device's layers.
  9. The PLE ids of the prompt rows are not read back by the golden test's prefill mode (its decode rows still check them).
- **Mac gates (measured):** `qwen4exp_pf_ref_test` PASS (the sort at C 1 / 37 / 300 / 2048 incl. 2386 rows with an exact
  tie at the 10th, all-to-10 over the lane boundaries, the 1200-tile bound, a full tile and an empty expert, == spec
  15d's sort; the combine == 21c's down chain over real-width blocks, both shared forms, bitwise; the sparse walk ==
  the direct softmax within 3.4e-16; the indexer over 22 chunks and the PLE over 13 == decode's M = 1 chains with their
  rings, bitwise); `qwen4exp_pf_variant_names_test`, `qwen4exp_plan_test` PASS; `qwen4exp_run` on the Mac's GPU (Intel
  UHD 630): 0 disagreements - the sort / gather / dequant / combine / rings / PLE gather / selection exact, the HC xn,
  PLE gn, indexer q / keys within 1 bf16 ulp, the scores within 1 fp32 ulp (Apple's divide and sqrt); Level Zero syntax
  of every new / changed source, OpenCL syntax of every variant; kernel_cmdlines +30 / -0 / ~0; `tools/mac_check.sh
  --base main --quick --kernels` exit 0 (host 81 pass / 0 fail, l0 21 / 0, opencl 675 / 0, kernels 9 agree).
- **What row 33 must prove:** every binary's first compile (30), G0 with the three changed sources' binaries unchanged;
  F2 on the card (grouped == dense bitwise, the sparse flash at cosine 0.99999 of fp64, eager within its bars); F4 on
  prefill (two runs and replay bitwise, chunks of 64 and splits at multiples of 64 bitwise, the injected run chunked ==
  whole); prefill against decode's fill within the PROPOSED bars per layer, routes and selections equal except near-ties;
  F3 on prefill for both synthetics (Intel's `--layers 18` when 21a's data exists); two cards bitwise one; the speed rows
  (Task 5, opt-in).
