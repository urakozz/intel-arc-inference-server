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
| quantisation | AutoRound W4A16 g64 (the operator's) | the base: bf16 only; **the operator's AutoRound W4A16 g64, `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` (§13)** |

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

**1. Where the int4 weights come from - decided (§13): (A). The operator's AutoRound checkpoint
already existed: `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` (2026-08-30).**
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


## 11. Amendment - 2026-10-05: 15d as built blind (Mac; box pending)

Branch `spec15d-ornith-prefill` (plan 15d Tasks 2-3; Task 1 P0 and Task 4 speed are box work).
Nothing here has run on the card. Validation: `box-validation-queue.md` entry 13.

**The chunk's MoE block, one layer** (`runtime/prefill/moe.cc`; kernels
`src/kernels/prefill/pf_moe.cl`, `pf_moe_gemm.cl`). No count is ever read back: every grid is a
function of C alone.

1. **Routing, decode's formula op for op.** The router || shared-gate GEMV over the chunk is
   `pf_gemv_bf16.cl` at N 272 with decode's {16, 16} tiling (`pf_moe_router_K2048_N272`), so each
   row is bit-identical to decode's `gemv_bf16` of the same x (pf_gemv_bf16.cl's tree argument);
   then **decode's own `moe_route` binary** at grid (1, C) - it reads logits row m and writes route
   row m and nothing else - so prefill and decode route by one kernel. The route rows are kept per
   layer (`[layers][kC][32]`, 10.5 MB) for R2 and the prefill-vs-decode check.
2. **The sort** (`pf_moe_sort`, one work-group of 256 lanes, lane = expert): the chunk's ids staged
   into SLM as bytes, each lane counts its pairs, lane 0's prefix over the experts' tile counts,
   then each lane walks (token, slot) ascending and gives each of its pairs the next row. Every
   expert's rows are padded to whole TM = 32-row tiles (so a tile belongs to one expert and no 2D
   block access needs out-of-surface semantics); the shared expert is block 256 with the C tokens
   in order after the routed experts - a dense M = C GEMM inside the same grouped launch. The tile
   table is padded to `tmax(C) = floor((8C + 256 x 31) / 32) + ceil(C / 32)` with NONE (824 at
   C = 2048; the adversarial distribution - as many one-row experts as the pairs allow - uses 817).
   No atomics: positions are a function of the route rows alone.
3. **The A operand**: `pf_moe_gather` copies x rows into sorted order (padding rows zero); on
   l0-int8 the chunk is quantised ONCE per token by spec 5's `pf_quant_had` and
   `pf_moe_gather_i8` gathers the int8 rows and their scales.
4. **The B operand, rebuilt per chunk (the "separate pass" arm):** l0-int8 - one
   `pf_requant_rot_L1` launch over the layer's whole gate||up array (257 contiguous layout-1 blocks
   read as one weight of K 2048, N 263,168; `Int8State::scales_layout1` caches its rotated column
   scales, filled in `prepare_prefill`); l0 - `pf_moe_dequant_gu` in two batches (129 + 128
   blocks); down - `pf_moe_dequant_dn`, all 257 blocks, bf16 on **both** backends: its K = 512 is
   not whole 1024-k rotation blocks, so spec 5's h8 does not apply (a 512-block rotation is an
   arm, not built). Experts with no row this chunk are skipped by the dequant (not by the requant,
   which is spec 5's kernel unchanged).
5. **The grouped GEMM** (`pf_moe_gemm`): one launch per (form, batch) over every tile; work-group
   (tile, 256-column block) reads (block, first row) from the table and returns at once for NONE or
   a block outside the batch. The mainloops are `pf_gemm.cl`'s bf16 (32 x 64 per sub-group, 4
   sub-groups) and `pf_int8.cl`'s i8 (32 x 32, 8 sub-groups) with only the addressing changed;
   epilogues: gate||up's SiLU chain (verbatim), down's `rne(acc)` into bf16 y. No prefetch and no
   split barrier yet (P0 adds them if they pay).
6. **The combine** (`pf_moe_combine`): moe_down's epilogue over the sorted rows - TOP_K terms
   `rne(y . w_k)` summed in fp32 in **fixed slot order**, the shared expert by its gate, the bf16
   sum, **the residual fold** - so the next norm folds nothing (S_PREV 0), as on decode.

**Determinism and row independence.** The sort is a pure function of the route rows; gather and
dequant are copies; DPAS reduces along K only, so a row's accumulator never meets another row's
and the epilogue is per element (its scale is the row's own xs): a row's output does not depend
on its tile neighbours, its position in the tile, or the chunk it rides in. The combine's order is
fixed. Hence replay and chunking (at multiples of 64, where the GDN chunks agree) are bitwise. The
only atomic on the path is spec 5's `pf_colmax_rot` (`atomic_max` on non-negative float bits, at
load): a max, order-free. `pf_moe_ref_test` pins row independence on the host reference;
`pf_moe_test` on the card (the reversed chunk, and grouped == dense bitwise).

**Launches per chunk (derived):** MoE block 10 on l0-int8 (router, route, sort, quant, gather,
requant, GEMM, dequant + GEMM for down, combine) and 11 on l0; with the mixer (GDN 2 + 24 + 1 + 10
+ 4, FA 2 + 18 + 3 + 4 on l0, +2 quantisers on l0-int8) and the post norm, **2061 per chunk on
l0-int8, 2021 on l0** (`step_chunk_launches`), every C.

**Memory (derived):** the MoE prefill scratch is 0.689 GB at kC 2048 (0.541 of it the weight
batch, sized for the l0 gate||up half-array); the h8 scales of the expert arrays 84 MB; Ornith's
262144 context still fits with the l0-int8 prefill (26.1 GB + the 1.5 GB reserve,
`memory_plan_test`).

**Bandwidth of the separate pass (derived, the number P0 must check):** per layer per chunk the
int8 gate||up reads 0.29 GB of int4, writes and re-reads 0.54 GB of int8; the bf16 down reads
0.14, writes and re-reads 0.54 - ~2.6 GB, ~104 GB per 2048-token chunk, ~0.19 s at ~550 GB/s: a
ceiling near 10k t/s before any compute, so the fused-dequant arm (spec 15 §9) is the first lever
if P0 shows the pass dominating.

**Shape variants** (additive; `tools/kernel_cmdlines` 312 -> 334, no line moved):
`GDN_K_HEADS / GDN_V_HEADS` in `pf_gdn_conv`, `pf_gdn_wy`, `pf_gdn_scan`, `pf_gated_head`;
`FA_Q_HEADS / FA_KV_HEADS` in `pf_attn_prep`, `pf_attn`, `pf_flash_attn` (Ornith's flash at HPW 8 =
GQA 8, grid.z 1). Unset, the clang -E output of all 8 existing variants of these files is identical
to main's. Built for Ornith: the norm pair at K 2048 (SP0 / SP1), embed, a||b, the slab dequant at
the descriptor's layouts (qkv||z L1, the rest L0), the quantiser at K 2048 / 4096, the GDN chain and
the gated head, `pf_attn_prep_q16`, `pf_attn`, `pf_flash_attn`. On a MoE model the L0 slab and
Int8State's K are the widest mixer linear's (4096); dense models' sizes are unchanged.

**What changed in the refusals.** `model::require_prefill` passes Ornith (it refuses only a MoE
shape the prefill kernels are not written for); `Engine::prepare_prefill` refuses sycl-tla for a
MoE model by name. `b70-serve` still refuses a MoE model (serving is 15e); MTP stays refused.

**Not built:** the SLM-fused dequant arm (plan Task 1), h8 for the experts' down, prefetch / split
barriers in the grouped GEMM, the int8 KV cache's prefill at Ornith's shapes, the composed
attention's fp32-q identity prep at Ornith's shapes.


## 12. Amendment - 2026-10-05: 15e as built blind (Mac; box pending)

Branch `spec15e-ornith-serving` (plan 15e Tasks 1-2; Task 3 - A4, prefix caching on the card,
passkey, the comparison rows, the record - is box work). Nothing here has run on the card.
Validation: `box-validation-queue.md` entry 16.

**The chat template (Review Focus 1).** Ornith's `chat_template.jinja` is **Qwen3.5's, not
Qwen3.8's** (sha256 `182e77dd...`; Qwen3.8's / Agnes's is `c3cf9e34...`): no reasoning-effort
block, every assistant turn renders its `<think>` block (Qwen3.8 drops it before the last user
turn unless `preserve_thinking`), and `tool_call.arguments is defined` without the `!= ''` test.
The tool-call format it advertises is the same Qwen XML (`<tool_call>\n<function=...>\n
<parameter=...>`) that `src/server/toolcall.cc` parses: no parser change. Its
`tokenizer_config.json` embeds an older template (with `preserve_thinking`); transformers loads
`chat_template.jinja` over it, and so does the engine. The one `is undefined` test is spelled
`is not defined` for minja (keyed by the source's sha256, like Qwen3.8's fallback). Rendered by
the engine on six message lists (plain, thinking, tools, a tool call in history, tool responses,
two calls in one turn) **byte-identical to transformers 5.18.0** (`template_ornith_test`; the two
template files vendored at `tests/tokenizer/ornith/`, MIT, revision `10fbf86f`); the server over
that template (`ornith_server_test`, mock engine): the engine's prompt is transformers' render,
an Ornith-format tool call comes back as an OpenAI tool call buffered and streamed, reasoning
splits at `</think>`. EOS: `generation_config.json`'s `[248046, 248044]`.

**The tokenizer.** `tokenizer_config.json`'s added tokens are Qwen3.8's (248044..248076,
identical), but Ornith's `tokenizer.json` (12,807,982 B against Qwen3.8's 12,809,320) stops its
added tokens at `</think>` (248069): the seven audio specials 248070..248076 are missing, so the
Rust tokenizer counts 248070 ids. The greedy argmax masks from the descriptor's 248077 (15c),
sampling from the tokenizer's 248070; `b70-serve` says so at startup. Whether the merges / vocab
differ beyond that was not checked (the file is 12.8 MB; a box-side diff is in the queue row).

**`b70-serve` serves Ornith.** The MoE refusal is gone. `--max-len auto` and the memory plan carry
the MoE terms since 15c / 15d and, with `--mtp`, the head's weights and buffers (below);
`--prefix-cache-gb auto` is host-RAM only. Prefix-cache entries are in-process, sized from the
engine (`state_bytes` 70,778,880 B + 4,096 with the head: 30 GDN states, conv rings, h; KV 20 KiB
per position, 22 KiB with the head's layer), keyed by the id chain under the KV form - one model
per process, so no model key is needed. `--kv-cache int8` is not built at Ornith's heads (12b's
binaries are Qwen3.8's): capture names the missing binary.

**The MTP head, as read** (the published checkpoint's shard 16 header, 2026-10-05): 785 bf16
tensors, 1,689,281,536 B - `mtp.fc` [2048][4096], `pre_fc_norm_{embedding,hidden}`, one
full-attention layer (q_proj [8192][2048] = 16 heads x (q || gate), k / v [512][2048], o
[2048][4096], q_norm / k_norm, the two layer norms), `mtp.norm`, and as its FFN **one MoE layer
of the main model's shape**: `mlp.gate` [256][2048], **per-expert** `mlp.experts.E.{gate,up}_proj`
[512][2048] / `down_proj` [2048][512] (not the main layers' fused 3D tensors), the shared expert
and `shared_expert_gate` [1][2048] - vLLM's `Qwen3_5MoeMTP` (a `Qwen3_5DecoderLayer` with the
MoE block, `mtp_num_hidden_layers` 1). `ModelDesc::mtp_head_moe()`, `mtp_checkpoint_bytes()` /
`_tensors()`.

**Its device form.** fc, q||k||v and o bf16 as the dense head's; the MoE layer in exactly the
main layers' form (`loader::MoeLayer`: router || gate rows, 257 int4 g64 layout-1 blocks of
gate||up and down) so `moe.cl` runs it. **Decision: the head's bf16 experts are quantised at load,
round to nearest, int4 g64 symmetric** (`loader/rtn.h`: GPTQ's formula, scale 2 amax / 15 stored
as f16, q = clamp(rint(w / s) + 8, 0, 15) on the stored scale) - an AutoRound export of the main
model leaves `mtp.*` as it found it, the kernels read int4, and the head only drafts: its
quantisation can move acceptance, never output (verify decides every token, M3). An expert the
checkpoint ships int4 is repacked as shipped, linear by linear; the main layers keep refusing bf16
experts. 501,990,464 B on the card (`loader::mtp_head_bytes`); the 771 RTN linears are timed in
the load report (derived: a few seconds of host time, unmeasured).

**The lists.** The draft list runs the head's MoE layer through `moe_block()` on the MoE
scratch's last layer slot (the slot 15c reserved): **24 launches** (the dense head's 23 with the
4-launch MoE block for gate||up, SiLU, down; `draft_launches`). The verify lists are Ornith's
decode list at M rows plus the head's KV fill: **536 launches at every M** (`verify_launches`);
both counts are asserted by the walks. **`moe.cl` needed no change for M > 1**: every buffer is
indexed by the row's own group id and nothing depends on M but the grid, so an M-row launch
routes each row exactly as M = 1 does (Review Focus 3) - shown on the Mac's GPU at M = 4, in
order and with the rows reversed, bitwise (`moe_run`, indicative), and gated on the card by
`moe_m_test` and `ornith_mtp_test` (M2: logits rows, GDN slots and all 40 layers' route rows
bitwise). New binaries only (kernels' 15e block, `tools/kernel_cmdlines` 378 -> 458 on this base,
none moved): Ornith's decode list at M = 2..4 (`moe_M{2,3,4}`, the GEMVs, norms, attention, the
lm_head forms), `gdn_step_slots_M{1..4}_G30_GK16V32` (SPEC_SLOT_STRIDE 15,728,640 floats), the
head front at Ornith's shapes, the draft vocabulary's compact heads at K 2048; prefill's head KV
fill (`step_mtp_kv` on a MoE model: its final norm folds nothing, SP0) with `pf_bf16_slab` at
K 4096 / 2048. `ornith_mtp_names_test` holds every bound name against the built ones (host).

**Memory (derived).** With `--mtp`, at 262144 and the int8 head: model 20.020 GB (the head
+0.502), decode state 0.824 (the head's buffers 0.729: three 62.9 MB GDN slots, one 2-kv-head KV
layer 0.537 GB, rows), total 27.36 GB + the 1.5 GB reserve on the l0-int8 path - **the full
trained context still fits with MTP** (`ornith_mtp_head_test`).

**Not decided blind (Review Focus 2).** The verify cost at K = 1..3 on Ornith - up to 8 x (K + 1)
distinct experts, on a launch-bound step - is unmeasured; `--mtp auto` uses Qwen3.8's cost table
and says so at startup until the box records Ornith's (`ornith_mtp_test` prints acceptance and
ms/id per K; `probe_mtp_steps` prices the table). No default K is chosen for Ornith.

**Not built:** the int8 KV cache at Ornith's heads; an int4-checkpoint lm_head at M > 1 (as for
Qwen3.8); `--spec lookup` is untouched (it works on Ornith wherever the verify lists do).

**Task 3's Mac side (2026-10-06, branch `serving-18d-15e`).** Everything plan 15e Task 3 needs
before the card, written and host-tested; nothing run on weights (the Mac's one oracle container
slot was taken). **A4 on Ornith**: its calls are Qwen XML, so `tools/toolcall/score.py` reads them
on its unchanged path. The set is Ornith's own: `make_set.py --from tests/golden/toolcall` with
Ornith's template and tokenizer (the 36 conversations of Qwen3.8's set, `enable_thinking` false,
no template kwargs). The reference is the **int4 checkpoint's** (15a's rule: the engine's own
weights, so a mismatch is the engine's and not the quantisation's): `oracle_generate.py` on a
`qwen3_5_moe` checkpoint runs `tools/oracle/ornith_ref.py`'s layer-streamed
`Qwen3_5MoeForCausalLM` (experts dequantised on demand), greedy through its KV cache, 192 new ids,
stopping after an EOS id (`[248046, 248044]`). `tools/toolcall/a4_ref.sh ornith set` /
`ornith ref` run both on the Mac (agnes-ref-img, 28 GB cap, `ref` detached; refused while another
container of the image runs) into `oracle-out-ornith-a4/`, which `box_validate.sh --push-data`
copies to the box; there r16.a4 (opt-in) runs `engine_generate.sh` on the set (l0-int8 with the
bf16 and the int8 head, l0), scores it and sends the set through `b70-serve`. **Prefix caching C2**
on Ornith is r16.prefix (`prefix_gpu_test` on l0-int8, l0 and with `--mtp 3`); **passkey** at 120k
and 250k r16.passkey (opt-in); determinism and replay are rows 10 / 13's gates; the comparison rows
r16.benchy (opt-in). **The record's skeleton**: `docs/BENCHMARKS.md` "Ornith 1.5 MoE (spec 15)",
every cell pending its stage. No bar is set blind (A4: recorded beside Qwen3.8's 25/36).


## 13. Amendment - 2026-10-06: the real int4 checkpoint

Branch `ornith-real-checkpoint`. The repo was written for a checkpoint assumed not to exist
(placeholder `urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ`). The operator's export
already existed: **`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`** (revision `cb1aa7d5`,
published 2026-08-30; AutoRound 0.15.0, `quant_method: gptq`, provider auto-round, bits 4,
group_size 64, sym, desc_act false, iters 400, calibrated on `opencode-instruct`; 22.99 GB).
Decision 1 is (A). Read 2026-10-06 from its config, index, all six shard headers (HTTP range
requests, nothing downloaded) and range-fetched tensor data; not run on the card. Validation:
`box-validation-queue.md` row 19.

**What 15c / 15d / 15e assumed, checked.**

| assumption | the checkpoint | |
|---|---|---|
| per-expert GPTQ `layers.L.mlp.experts.E.{gate,up,down}_proj.{qweight,qzeros,scales,g_idx}` (§10) | yes: 30,720 linears, gate / up qweight [256][512], down [64][2048] | holds |
| shared expert `mlp.shared_expert.*` int4, router `mlp.gate.weight` [256][2048] and `mlp.shared_expert_gate.weight` [1][2048] bf16 | yes (`dynamic`: 80 `-:` rules, exactly the 40 routers and 40 shared gates) | holds |
| attention q / k / v / o int4 (q_proj N 8192 = 16 x (q \|\| gate)) | yes | holds |
| GDN in_proj_qkv / z / out_proj int4 | yes (qkv N 8192, z 4096, out_proj K 4096) | holds |
| **GDN in_proj_a / in_proj_b bf16** (the descriptor's AB row) | **int4 g64**: qweight [256][32], scales [32][32] each | **did not hold - fixed below** |
| scales f16 | F16 everywhere; signed (AutoRound's sym: a/b layer 0 [-0.0139, 0.0154]); subnormals present | holds (loader counts subnormals) |
| qzeros 0x77777777, g_idx identity k / 64 | all 60 a/b linears and 160 random other linears range-fetched: all hold | holds (the loader asserts all) |
| non-quantised tensors bf16 | yes: embed, lm_head, norms, conv1d, A_log, dt_bias, routers, MTP head, vision | holds; config.json's top-level `dtype: float16` is the export's label (text_config bf16) |
| MTP head (15e, read from the bf16 base's shard 16) | `model_extra_tensors.safetensors`: the same 785 bf16 tensors, names and 1,689,281,536 B | holds (RTN at load as built) |
| tokenizer.json defines 248070 ids (66a8923) | **248077**: re-serialised by transformers 5.14.1 (19,989,325 B), added tokens 248044..248076 incl. the seven audio specials; vocab and merges identical to the base's | **did not hold - vocab_used 248077** |
| the committed golden prompt ids | prose / code / cjk tokenise to the committed ids (trailing newline stripped, as tokenize.py) with this tokenizer.json | holds (`B70_ORNITH_PROMPTS_DIR` unchanged) |
| chat template (vendored, sha256 `182e77dd`) | `chat_template.jinja` byte-identical | holds; its tokenizer_config.json is a new 1,166 B one (no embedded template, same bos null / eos `<|im_end|>`) |
| EOS from generation_config.json | `[248046, 248044]` | holds |
| `doc_w` 0 (no W cross-check) | W = 2,344,862,976 B from the headers (below) | set |

The pre-tokenizer of this tokenizer.json is Qwen's newer regex (`[\p{L}\p{M}]+`, as its
tokenizer_config's `pretokenize_regex`), where the base's had `\p{L}+`: text with combining
marks tokenises differently from the base checkpoint's file. The engine and the reference both
read this file, so they agree.

**The a||b design (operator rule: no lossy re-rounding, no new kernel family).** a||b's kind
is the checkpoint's, as lm_head's already was: `ModelDesc::ab_int4` / `ab(kind)` beside the
table's bf16 AB row (one `LinearId::AB` ordinal; every table walk - buffer sizes, the h8 scale
pass, the variant names - keeps seeing the bf16 row). Argued against the alternative "Ornith's
AB row is int4": the form is a property of the export, not of the model - Qwen3.8's and Agnes's
AutoRound exports exclude in_proj_a/b, this one does not, and the operator's own
re-quantisation (spec 20) may do either - so the loader classifies by content (the first GDN
layer's in_proj_a, `loader/ab.h`) and the capture binds by the loaded weight's kind, exactly
the lm_head mechanism. A compressed-tensors int4 a||b goes through the same
`LinearSrc::classify` / `suffixes()` (spec 20 §9).
- **Decode:** `{K 2048, N 128, S 1, layout 1}` through gemv.cl - the existing int4 GEMV,
  new variants `gemv_M{1..4}_K2048_N128_S1_L1` only. The 64 real columns (a at [0, 32), b at
  [32, 64)) are zero-padded to 128 with the int4 zero (nibble 8 under a +0 f16 scale: each
  padded output exactly 0), so the pad-to-128 contract holds unchanged: gemv.cl at S = 1
  writes `out[m x 128 + n]`, i.e. ab_out [M][128], where gemv_bf16 did, and gdn_step /
  prefill's gate read it at AB_STRIDE as before. One launch either way: 526 / 536 / 24 launch
  counts unchanged, the MTP verify lists at M = 2..4 included. The weights the GEMV multiplies
  are the checkpoint's exactly (fp32 `(q - 8) x scale`). PROVISIONAL tuning: 128 columns at S 1
  are 8 sub-groups (gemv_bf16's L2 lesson); an S > 1 form would need gdn_step to sum slices,
  a P0 arm.
- **Prefill:** `pf_ab_proj_D2048` (pf_gemv_bf16.cl) unchanged, over a bf16 copy the loader
  makes once at load with the prefill dequant's own arithmetic (pf_dequant_slab:
  `rne_bf16((q - 8) x scale)`, the rounding every int4 linear already takes on prefill). The
  slab dequant itself does not apply (1024-column slabs, [K][NS] row-major, not the GEMV's
  tiles); a per-chunk dequant launch would redo 0.13 MB per layer per chunk for nothing.
  15.7 MB resident (`LoadedModel::ab_prefill`, in `LoadReport::total()`, not per-token); the
  launches per chunk (2061 / 2021) are unchanged; the h8 scale pass skips AB.
- **Planner:** the model bytes are the loaded total, so the plan follows: +139,264 B per GDN
  layer (the int4 a||b with its padding) and the 524,288 B prefill copy in place of the bf16
  a||b (memory_plan_test: 19,454,989,312 B with the int8 head; 262144 still fits).
- **Qwen3.8 / Agnes:** their checkpoints ship a||b bf16 -> the table's row object itself; no
  binary, command line or byte of theirs moves (`ab_int4_test` checks the object identity).

**doc_w.** 2,344,862,976 B summed from the headers over the categories a decode step reads
with a bf16 head: int4 qweight + scales of the mixer linears and shared experts 748,544,000
(a||b 2,088,960 of it), 8 routed experts per layer x 40 = 534,773,760, bf16 routers + shared
gates 42,106,880, bf16 small tensors 2,319,616, lm_head 1,017,118,720. The device side differs
by itemised terms only: a||b's zero columns (`pad`), the fp32 widening, and - new on the
expected side, MoE only - the routers' 15 zero rows per layer (2,457,600 B): the predicted delta
is 0.000 %.

**vocab_used 248077** (`Qwen35::kVocabUsed`): Ornith's lists bind Qwen3.8's
`argmax_stage1_M{1..4}`; the `_V248070` binaries stay built (no command line disappears),
unbound; `b70-serve`'s vocab note does not fire.

**Comparison point:** the checkpoint's model card reports vLLM XPU on a B70: pp4096 7090 t/s
(c1), tg256 99.8 t/s (c1) - the bar §6's record compares against.

**15a's reference from this checkpoint** (plan 15a Task 0): `tools/oracle/ornith_ref.py` on the
Mac (transformers 5.15's Qwen3_5MoeForCausalLM, layer-streamed, experts dequantised on demand),
`tools/oracle/ornith_golden.sh`; the bf16-base reference stays box work (R5, A4).
