# Spec 15 - Ornith 1.5 35B-A3B, the first mixture-of-experts model

**Status:** design, 2026-10-04, for operator review. Open decisions are marked **(decide)**.

**Model:** [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
(MIT licence; built on Qwen3.5 by continued pre-, mid- and post-training). Published in bf16
only (71.9 GB, 16 shards); GGUF builds exist.

**Order:** after spec 14 merges: it reuses and extends spec 14's `ModelDesc`.

Every number is **measured** unless marked **derived** or **estimated**. The shapes below were
read from the checkpoint's `config.json`, `model.safetensors.index.json` and shard headers
(2026-10-04).

## 1. What Ornith is, against what the engine runs

| | Qwen3.8-27B (today) | Ornith 1.5 35B-A3B |
|---|---|---|
| architecture | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`) |
| layers | 64: 48 GDN + 16 FA, FA at `l % 4 == 3` | **40: 30 GDN + 10 FA**, the same pattern |
| hidden | 5120 | **2048** |
| FA | 24 q-heads (q_proj 2x for the output gate), 4 kv-heads, head_dim 256 | **16 q-heads, 2 kv-heads** (GQA 8), head_dim 256, the same gate |
| GDN | 16 k-heads, **48** v-heads, dim 128, conv 4 | 16 k-heads, **32** v-heads, dim 128, conv 4 (conv dim 8192) |
| FFN | dense SwiGLU, intermediate 17408 | **MoE: 256 routed experts, top-8, expert intermediate 512, plus one shared expert (intermediate 512) with a sigmoid gate** |
| expert storage | - | fused per layer: `mlp.experts.gate_up_proj [256, 1024, 2048]`, `mlp.experts.down_proj [256, 2048, 512]`; router `mlp.gate.weight [256, 2048]`; `mlp.shared_expert.{gate,up,down}_proj`, `mlp.shared_expert_gate.weight [1, 2048]` |
| MTP head | one dense layer | one **MoE** layer (per-expert tensors `mtp.layers.0.mlp.experts.N.*`), `mtp.fc [2048, 4096]` |
| vocab, embeddings | 248320, untied | 248320, untied (embed and `lm_head` 1.02 GB each in bf16) |
| max positions | 262144 | 262144 |
| quantisation | AutoRound W4A16 g64 (the operator's) | **none published: bf16 only** |

**Parameters (derived from the shapes):** routed experts 32.21 B, GDN projections 1.01 B, FA
projections 0.27 B, shared experts + routers 0.15 B, embeddings 1.02 B: **34.7 B total**.
**Active per token:** 8 experts x 3.15 M + the shared expert 3.15 M + router 0.52 M + the layer's
attention (GDN 33.7 M, FA 27.3 M) per layer, 40 layers, plus `lm_head`: **2.95 B**.

## 2. Why it matters, and what it costs

**Decode reads only the active weights.** At int4 g64 (~0.53 B per parameter) the quantised active
weights are ~1.30 GB per token, plus `lm_head` (1.02 GB bf16, 0.51 GB int8): **~1.8 GB per token**
with the int8 head, against Qwen3.8's ~14.3 GB (derived). The bandwidth ceiling is then ~270
t/s; **the real limit will be launches and latency, not bytes**: Qwen3.8's step is 774 kernels in
~32 ms, so a 40-layer step must stay near ~10 us of overhead per kernel to beat ~100 t/s
(estimated). The design minimises launches per layer (§4.3) and P0 measures the floor.

**Memory is easy.** ~19.9 GB of weights with int4 experts and the bf16 head (derived); KV is
**20 KiB per position** (10 FA layers x 2 kv-heads), so 128k is 2.7 GB and the full 262144 is
5.4 GB: **262k context fits** (~27 GB total, derived), which spec 12 is not needed for.

**Prefill is a grouped-GEMM problem.** A 2048-token chunk routes 16384 token-expert pairs over
256 experts, ~64 rows per expert on average (derived), each a GEMM of M ~64, K 2048, N 1024
(gate‖up) and K 512, N 2048 (down): small, many, and uneven. FLOPs per token are ~5.9 G against
Qwen3.8's ~54 G, so prefill could reach several thousand t/s if the grouped GEMM is efficient
(estimated 3-6k t/s; P0 sizes it).

## 3. The decisions

**(decide) 1. Where the int4 weights come from.**
- **(A, proposed) The operator's AutoRound W4A16 g64 quant**, as for Qwen3.8 and Agnes: the
  engine's whole path is gated on AutoRound g64, and AutoRound beat RTN clearly on Qwen3.8
  (docs/probe-w4a8 §15). Its fused-expert export format is to be checked (AutoRound / GPTQ may
  emit per-expert `qweight` tensors; the loader repacks either way).
- (B) Load-time RTN int4 g64 from the bf16 file (the loader streams 72 GB and quantises): no
  separate checkpoint, but RTN's error is higher; measured in 15a against (A) to decide.

**2. Build on spec 14's `ModelDesc`**, extended from layer counts to every per-model shape:
hidden, q/kv heads, GDN v-heads, FFN kind (dense / MoE with expert count, top-k, expert and
shared intermediate). Qwen3.8 and Agnes stay bitwise unchanged.

**3. Expert weights on the card: per-expert contiguous int4 blocks** (`[256][...]`), so a kernel
reads expert e's slice by offset `e x stride` - the router's choice is consumed on the device,
no host round trip, and the captured decode list stays fixed.

## 4. Design

### 4.1 The router (decode and prefill)

`gate.weight [256, 2048]` GEMV (bf16), softmax over 256 (fp32), top-8, renormalisation of the 8
weights if the model's `norm_topk_prob` says so (15a reads the transformers implementation and
pins the exact formula), expert ids and weights written to a small device buffer. The shared
expert's sigmoid gate `shared_expert_gate [1, 2048]` in the same launch.

### 4.2 Decode: the MoE block in three launches (target)

1. **Router + shared gate** (§4.1).
2. **Expert gate‖up for the 8 selected experts and the shared expert**: one launch, grid over
   (9 experts x 1024 rows); each work-group reads its expert id from the router buffer and its
   weights at that expert's offset; SiLU x up fused.
3. **Expert down + weighted sum + residual**: one launch, grid over 2048 output rows, summing the
   9 down projections weighted by the router weights (shared by its sigmoid gate) in fp32 and
   folding into the residual stream (as `prep_res_fold` does today).

Bitwise determinism: the sum over experts is in fixed order (expert-slot order 0..7, then the
shared expert), not atomic.

### 4.3 The rest of the step

GDN (30 layers) and FA (10 layers) use today's kernels at Ornith's shapes: `gdn_step` at 32
v-heads, decode attention v2 at GQA 8 (it is written for 6 q-heads per kv-head today), `prep`,
norms and GEMVs at K / N 2048. Every one is a variant, not a new kernel, except where P0 shows a
kernel's work-group geometry cannot reach occupancy at hidden 2048 (then a re-tiled variant).

### 4.4 Prefill: grouped experts

Per chunk: route all tokens (one GEMM [C x 2048] x [2048 x 256] + top-8); a device histogram of
tokens per expert, a prefix sum, and a scatter of token rows into expert-sorted order; **one
grouped GEMM launch for all experts' gate‖up** (each work-group owns one expert's row block,
reading the expert's offset and row count from the prefix-sum table), SiLU; one grouped GEMM for
down; a gather that multiplies by the router weight and sums each token's 8 results in fixed
order, plus the shared expert (a dense GEMM, M = C). Spec 5's int8 path (h8: rotated int8
activations, int8 weights rebuilt per slab) applies per expert if 15a/P0 show it pays at M ~64;
bf16 DPAS otherwise.

### 4.5 The MTP head

Ornith's MTP head contains one MoE layer: the decode MoE kernels at M = 1 serve it; spec 8's
draft / verify / commit machinery applies unchanged. Verify at M = K + 1 multiplies the routed
experts touched (up to 8 x (K + 1) distinct experts): its cost model differs from the dense
model's and is measured, not assumed.

## 5. Correctness gates

- **R0, nothing moves:** Qwen3.8 and Agnes bitwise unchanged (spec 14's G0, extended).
- **R1, the CPU reference:** transformers' `qwen3_5_moe` model, layer-streamed from the bf16
  checkpoint (spec 14's `tools/oracle/stream.py`, in Docker on the Mac or on the box), golden
  sets for prose / code / cjk + 32 greedy tokens.
- **R2, routing:** on the golden positions, the engine's top-8 expert sets equal the reference's,
  except where the reference's 8th and 9th router logits are within a near-tie tolerance (P0
  measures the distribution); router weights within 1e-3 relative.
- **R3, golden:** the golden gates on Ornith (`l0` and, if built, `l0-int8`), the golden tie rule.
- **R4:** A4 on Ornith against bf16 Ornith (no bar until the first measurement), spec 7's prefix
  cache, determinism and replay, passkey at 120k and 250k.
- **R5, quantisation (15a):** the chosen int4 source's logit error against bf16 on the golden
  prompts, recorded beside Qwen3.8's (7.77 % relative L2, docs/probe-w4a8 §15.2).

## 6. Speed bars

Idle box, device 0, interleaved pairs, median of 3. No bars before P0; the record is:
- **P0 (first):** the per-kernel floor of a replayed list (a list of N empty kernels and of N
  small GEMVs at hidden 2048), the router and expert-GEMV kernels in isolation, the grouped GEMM
  at the real expert-row distribution (from 15a's router statistics). From these, a derived decode
  and prefill estimate and the bars proposed for the operator.
- Recorded: decode at 4k / 32k / 128k / 250k depth, pp4096, pp at 128k, with `--lm-head int8`;
  llama-benchy rows with the operator's flags against vLLM (if it serves Ornith on XPU) and
  llama.cpp SYCL with an Ornith GGUF, same card.

## 7. Stages

- **15a, reference and quantisation (CPU; the box's oracle container, layer-streamed):**
  R1's reference and golden sets; router statistics (experts' load per layer on the golden prompts
  and two A4 prompts); the exact router formula; R5 for RTN g64 and, once the operator's quant
  exists, AutoRound g64. The bf16 checkpoint (71.9 GB) does not fit on the Mac (operator,
  2026-10-04), so the model-dependent steps run on the box's CPU; the code, its unit tests on
  synthetic tensors and the tensor-index checks are written on the Mac.
- **15b, `ModelDesc` for every per-model shape** (writable without the box, validated on it): R0.
- **15c, decode:** the MoE kernels (§4.1-4.2), the variants (§4.3), the loader (fused-expert
  repack, the quant format chosen in decision 1); P0; R2, R3 (decode path); decode speed.
- **15d, prefill:** routing, sort and grouped GEMMs (§4.4); R3 on prefill; prefill speed.
- **15e, serving and MTP:** chat template and tool calls (to be compared with Qwen3.8's), the MoE
  MTP head, A4, prefix caching, the comparison rows, the record.

## 8. Out of scope

- The vision tower; the DFlash checkpoints (`Ornith-1.5-35B-A3B-DFlash`, a separate drafter
  family, to be looked at after MTP works on Ornith).
- Expert parallelism, offloading experts to host memory (everything fits on the card).
- Ornith 1.5 9B (dense, a different shape set) and 397B (does not fit).

## 9. Amendment - 2026-10-04: refinements and P0 arms

- **Quantisation (decision 1) stands:** AutoRound g64 is the quality bar. A round-to-nearest
  4-bit model is fast to produce but is not a quality reference; any such model is judged by R3
  and R5 like everything else.
- **§4.2 refined: the shared expert is the ninth slot** of the expert gate‖up launch, its sigmoid
  gate computed by one extra work-group of that launch; down + the weighted sum run in fixed slot
  order through SLM. Three to four launches per MoE block in total.
- **Added to P0 (§6), each an arm measured against today's kernels at Ornith's shapes:**
  - a decode GEMV with the activation slice held in registers and split into even / odd K, so
    each half pairs with one nibble of a packed byte, 64-byte row loads and table-free dequant,
    beside our int4 g64 GEMV at hidden 2048;
  - router top-k in one sub-group with all 256 logits in registers;
  - for the grouped GEMM, dequant staged into SLM once per work-group per K step **against** a
    separate dequant pass: the "separate pass wins" rule (spec 2.1) was measured on dense shapes
    and may not hold at ~64 rows per expert;
  - a decode-attention arm that issues all K/V loads of a step before use, against spec 10's v2
    at GQA 8 (v2 is bandwidth-bound at depth; the arm targets short contexts).
- **Reference kernels:** the B70 W4A16 MoE kernels in the vLLM XPU stack (grouped W4A16 GEMV over
  the selected experts, sparse expert dispatch, a fused gate‖up GEMV; Apache-2.0) are a measured
  comparison point for the expert-kernel arms above, not a design to copy (they launch through
  PyTorch, without our replayed list).
- **§4.4 refined: prefill routing entirely on the device,** with a tile table padded with -1 so
  the host never reads per-expert counts. Rows land in an expert's tile in arbitrary order, which
  is harmless only if each row's GEMM result is independent of its neighbours: R3's bitwise replay
  checks this for our GEMM instead of assuming it.
- **lm_head:** int8 (spec 9) by default; a 4-bit head is evaluated under spec 9's gates (on
  Ornith the head is ~28 % of a token's bytes at int8, derived), not adopted.
- **Not adopted:** 4-bit round-to-nearest on attention and GDN projections; an unscaled FP8 KV
  cache.


## 10. Amendment - 2026-10-05: 15c as built blind (Mac; box pending)

Branch `spec15c-ornith-decode` (plan 15c Tasks 2-3; Task 1 P0 and Task 4 speed are box work).
Nothing here has run on the card. Validation: `box-validation-queue.md` entry 10.

**Checkpoint naming (the assumption 15a confirms).** No int4 Ornith exists. The loader reads
what an AutoRound W4A16 g64 sym export with GPTQ packing writes for this architecture:
AutoRound quantises `nn.Linear` modules, so it unfuses transformers 5's 3D
`mlp.experts.gate_up_proj [256, 1024, 2048]` / `down_proj [256, 2048, 512]` into one MLP per
expert (`auto_round/modeling/fused_moe/qwen3_5_moe.py`: gate = rows [0, 512), up = rows
[512, 1024) - **not interleaved**, which settles §1's open layout question for the export) and
writes `layers.L.mlp.experts.E.{gate,up,down}_proj.{qweight,scales,qzeros}` - the per-expert
names vLLM's `build_expert_params_mapping` loads. vLLM's per-expert fused form
`experts.E.gate_up_proj` (gate the first half of the columns) is read too; one form per
checkpoint. The router `mlp.gate.weight [256, 2048]` (a `Qwen3_5MoeTopKRouter` parameter, not
an `nn.Linear`) and `mlp.shared_expert_gate.weight [1, 2048]` must be bf16; the shared expert's
three linears are int4 under the table's GateUp / Down rows. Everything else is refused by name.
A bf16 checkpoint has no `quantization_config` and is refused before this (decision 1's option
B, RTN at load, is not built). `loader/moe_layout.h` carries the citations.

**Device form (spec decision 3).** Per layer: `router` bf16 tiled [272][2048] (256 router rows,
the shared gate's row at 256, zero rows to 272); `gate_up` 257 contiguous int4 layout-1 blocks
of 2048 x 1024 (gate||up interleaved in 16-column blocks, as the dense gate||up), the shared
expert as block 256; `down` 257 blocks of 512 x 2048. 430,604,288 B per layer, 17.224 GB per
model (derived; `loader::moe_bytes`, which the loader asserts its allocations against). The
table's GateUp / Down rows (the shared expert, 15b) are not bound by the decode list.

**The formula, read from transformers 5.18.0** (`modeling_qwen3_5_moe.py`; 15a pins the oracle's
version): `Qwen3_5MoeTopKRouter` takes softmax over the bf16 logits in fp32, `topk`, and
renormalises **always** (the class has no `norm_topk_prob` switch), then rounds the weights to
bf16. The default experts implementation is `grouped_mm` (`integrations/moe.py`): each expert
term `proj_out * weight` is rounded to bf16 and the k terms are summed in fp32 in slot (topk)
order, rounded once - the order moe.cl uses. The `eager` loop instead adds terms into a bf16
accumulator in ascending expert-id order with a rounding per add; **15a must record which
`experts_implementation` the oracle ran** (`grouped_mm` whenever torch can dispatch it). The
shared expert: `rne(sigmoid(rne(gate . x))) x shared_out`, added to the routed sum in bf16,
then the residual add. Top-k ties go to the lower expert id (rank = number of experts with a
larger p, or an equal p and a lower id); torch's CPU `topk` order on exact ties is not
specified, which R2's near-tie rule absorbs.

**Launches: four per MoE block, 13 per layer, 526 per token** (Qwen3.8's 774): the post norm
pair, then `gemv_bf16` over the 272 router || gate rows ({16, 16} tiling), `moe_route` (one
work-group of 256, one expert per lane; max, a pairwise Σ tree, rank-based top-k in one barrier,
renormalisation and the gate's sigmoid; writes ids, bf16 weights, the gate, the 8 p and the 9th
p), `moe_gate_up` (9 slots x 16 work-groups of 4 n-tiles x 4 K slices, the expert id read once
per work-group from the route row, SiLU x up fused; the shared expert is slot 8), `moe_down`
(128 n-tiles, a work-group of 9 slots x 2 K slices; the weighted sum in fixed slot order, the
shared expert by its gate, **the residual fold**). Because `moe_down` folds into `resid`
itself, the next prep_res_fold folds nothing (`ModelDesc::ffn_fold_s()` = 0, its SP0 variant)
and the per-layer tap is the layer's output (`golden_gate_test` builds its comparator
accordingly). Not built: the 3-launch arm (every expert work-group recomputing the top-k from
the logits instead of a route launch) - a P0 arm. The route row and the router logits are
written per layer (0.47 MB of scratch with a slot for the MTP head's MoE layer), so R2 reads
every layer's routing after a step with no debug list.

**Shape variants.** gdn_step / prep_gated_head take `-DGDN_K_HEADS/-DGDN_V_HEADS`, attn_prep and
attn_v2 `-DFA_Q_HEADS/-DFA_KV_HEADS`; unset, the defines are Qwen3.8's token for token (the
preprocessed source of all 85 existing binaries of those files is identical, and
`tools/kernel_cmdlines` shows additions only). Ornith's GEMV cells (q||k||v, qkv||z, out_proj =
o_proj, the a||b / router / lm_head bf16 GEMVs, the int8 head) carry no tuning defines:
PROVISIONAL until P0. Decode attention v1 is not built at Ornith's heads (capture refuses it).

**What Ornith refuses.** Prefill (`model::require_prefill`, from `Engine::prepare_prefill`,
`b70-decode --pp/--prefill`, `b70-serve` at startup): spec 15d. MTP (`--mtp`): the loader's
head is dense-only and capture refuses the verify / draft lists on a MoE model: spec 15e.

**Bytes (derived).** Weights with the int8 head 19.45 GB (MoE 17.22); read per token ~1.86 GB
(router + 8 experts + the shared expert per layer = 16.15 MB x 40, the attention / GDN linears,
the int8 head; `LoadReport::read_per_token` now counts only the active experts); the decode-only
plan at the full 262144 context is 24.97 GB of 32.53 (`memory_plan_test`). The loader's W
cross-check is skipped (doc_w 0) until a load on the card measures one.
