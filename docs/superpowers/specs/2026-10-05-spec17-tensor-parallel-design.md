# Spec 17 - tensor parallel over two B70s

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.

**Order:** after spec 16 (pipeline parallel). Spec 17 reuses spec 16's multi-device foundation (one
Level Zero context holding both cards, per-device buffers and captured lists, the peer-write rules)
and spec 16 P0's "remote partial" arm, which measures this spec's key number. PP and TP are
alternatives over the same two cards: `--pp 2` or `--tp 2`, not both.

**Prior art:** the two-rank peer-to-peer all-reduce over Level Zero IPC proposed for vLLM's XPU
kernels ([vllm-project/vllm-xpu-kernels#570](https://github.com/vllm-project/vllm-xpu-kernels/pull/570),
RFC [vllm-project/vllm#54766](https://github.com/vllm-project/vllm/issues/54766)), measured on two
B70s: ~6.6-10.8 us per all-reduce of 2-64 KiB inside an XPU graph, and Qwen3.8-27B GPTQ int4 at
**54.90 t/s** decode with TP = 2 at one request (vLLM, `--max-model-len 8192`).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

Single-stream decode is bandwidth-bound: one Qwen3.8 token reads ~14.3 GB (int8 head) at ~32 ms
(31.21 t/s, spec 9). Spec 16 (PP) does not make it faster. **Tensor parallel does:** each card reads
half of every weight matrix, so a step is ~16 ms plus the collectives. Two collectives per layer
(after the attention / GDN output projection and after the MLP down projection), 128 per token, at
~7 us each are ~0.9 ms: **~55-57 t/s (derived), ~1.8x one card**, in line with vLLM's 54.90 t/s on
the same cards. For long agentic sessions this is the one multi-GPU mode that shortens every turn.

TP also halves per-card KV (2 of 4 kv-heads each) and GDN state (24 of 48 v-heads each).

## 2. The decisions

**1. The collective is folded into the residual fold, not a separate kernel.** Our decode GEMVs
already write fp32 split-K `partials`, and `prep_res_fold` (stage A of the norm pair) sums them into
the residual stream before the next layer (`src/runtime/capture.cc`). With TP the other card's
partial is **one more partial**: each row-parallel GEMV (o_proj / out_proj, down, and the MTP head's)
writes its partials locally **and** into the peer's buffer with a peer write and a system-scope flag;
each card's `prep_res_fold` waits for the peer's flag and folds **card 0's partials then card 1's**
in that fixed order on both cards. Both cards therefore compute a **bitwise-identical residual**, and
no launch is added. Spec 16 P0's remote-partial arm measures its cost against a separate all-reduce
kernel; if the fold variant loses, the separate kernel (as in the prior art) is the fallback.

**2. The split (Megatron-style), per layer:**

| block | column-parallel (split N, no collective) | row-parallel (split K, partials folded) |
|---|---|---|
| FA attention | q (12 of 24 heads + their gates), k / v (2 of 4 kv-heads) per card | o_proj (K split by the card's heads) |
| GDN | `in_proj_qkv` / `in_proj_z` by head: 8 of 16 k-heads, 24 of 48 v-heads per card; `in_proj_a/b` by v-head | out_proj (K split by v-heads) |
| MLP | gate‖up: 8704 of 17408 rows per card | down (K split) |
| embedding | replicated (gather, no traffic) | - |
| `lm_head` | vocab split: 124160 rows per card | argmax: each card's top-1 and value exchanged, the larger wins (ties to the lower id) |

Norms run replicated on the identical residual. Attention, the GDN recurrence and the conv ring are
per-head and need no communication.

**3. Sampling (decide):** with the vocabulary split, host-side sampling reads both halves of the
logits (two readbacks, ~0.5 MB each); or each card applies top-k to its half and the host merges.
Proposed: the merge (fewer bytes, same distribution after the shared filter).

**4. Prefill:** the same split, but the collective is large: a 2048-row chunk's partials are
2048 x 5120 fp32 = ~42 MB per fold (~21 MB if sent as bf16, at an accuracy cost), 128 folds per
chunk: ~2.7-5.4 GB per chunk over PCIe 3.0, ~0.2-0.45 s at ~12 GB/s (derived), against compute that
halves from ~1 s to ~0.5 s. **Estimated 1.0-1.4x**, so prefill under `--tp 2` likely runs as spec
16's PP chunk pipeline instead (a hybrid for prefill only: layers split by depth for prefill, by
width for decode, which needs each card to hold both layouts' weights or a reshuffle; ~1.7x+ for
long prompts per spec 16). **(decide)** after P0 measures the transfer and the overlap.

## 3. Design

- **Weights:** the loader slices each linear by the table above at load (packed int4 g64 slices on
  group boundaries: 17408 / 2 = 8704 = 136 groups; head splits are whole heads); `ModelDesc` gains
  `tp_rank`, `tp_size` and derives per-card shapes. Kernels are the existing variants at the per-card
  shapes (e.g. 12 q-heads, 2 kv-heads, 24 GDN v-heads, gate‖up N 17408 per card).
- **Capture:** one decode list per card, both replayed for each token; the only cross-card points are
  the folds and the argmax exchange. The host submits both and waits on both fences.
- **Spec 7:** snapshots are per card (each holds its heads' KV and GDN state); the host store's
  entries become `{card 0 part, card 1 part}`.
- **Spec 8 MTP:** the head runs TP too (it is a dense layer at the main model's shapes); verify rows
  stay row-independent, so M2 (verify rows bitwise equal to M = 1 under TP) still holds.
- **Spec 10:** decode attention v2 at 12 q-heads per card (6 per kv-head, unchanged ratio).
- **Spec 12 / 14 / 15:** int8 KV per card unchanged; Agnes and Ornith split the same way (Ornith's
  2 kv-heads give one per card; its MoE experts are split by expert, 128 per card, a different
  scheme: out of scope here).

## 4. Correctness gates

- **T1, both cards agree bitwise:** after every layer the two cards' residual streams are
  byte-identical (the fixed fold order), checked for the golden prompts and 64 greedy tokens.
- **T2, against one card:** TP sums partial products in a different order than one card, so the
  result is **not** bitwise equal to one card: logits cosine and greedy tokens under the golden tie
  rule, as the chunked-prefill and split-prefill gates do; the golden gates on `l0` / `l0-int8` with
  `--tp 2`.
- **T3:** determinism and replay bitwise with `--tp 2`; spec 7's C1/C2 per card; spec 8's M2.
- **T4, failure modes:** a missing card, no peer access, or a fold timeout refused with a clear
  error, never a hang.

## 5. Speed bars

Idle box (both cards free of other DRM holders), interleaved pairs against one card, median of 3.

- **P0:** spec 16 P0's remote-partial arm (the fold's added latency) plus, here: per-layer launch
  timing with TP shapes, the lm_head split and argmax exchange, the prefill collective's bandwidth.
- **D1, decode:** `--tp 2` **>= 1.6x** one card at 4k depth (derived ~1.8x), recorded at 32k / 128k.
- **D2, prefill:** recorded for both options (TP prefill with overlapped transfers, PP-hybrid prefill); the bar is set after 17a.
- **D3:** `--tp 1` unchanged within 1 % of today.

## 6. Stages

- **17a, probe:** P0 beyond spec 16's arm; the fold-vs-kernel choice; the prefill choice.
- **17b, decode TP:** loader slicing, per-card variants, the folded collective, the lm_head split,
  `--tp 2` decode; T1, T2 (decode), T4; D1, D3.
- **17c, prefill TP:** the prefill path under the split (or the PP hybrid, per 17a); T2 (prefill);
  D2.
- **17d, integration:** spec 7 per card, MTP under TP, sampling merge, the server flag, T3, the
  record.

## 7. Out of scope

- TP combined with PP on the same cards; more than two cards (the layout extends to four; the fold
  generalises to more partials, but this spec builds and gates two).
- Expert parallel for MoE (spec 15's Ornith), multi-tile devices.
