# Spec 19b - the DFlash drafter on the card

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** load a DFlash / DFlash 2 drafter beside the target and run its draft pass as a captured Level Zero list: context projection into a ring, the block forward, the head's top-16, the selector walk; D1 against 19a's reference.

**Architecture:** spec 19 §2, §5 "Kernels" and "Memory". The drafter is a separate weight set sharing the target's `embed` and `lm_head` (and spec 8 §11's compact draft head when `--draft-vocab` is on). Its linears run through spec 9's int8 GEMV (quantised at load; decision 2) or bf16 GEMV at M = 1 + K rows. Its attention is spec 10's v2 with a ring-addressed context (window 2048) plus a non-causal block term. New kernels: the grouped two-tap convolution, top-16 over the head rows, the selector walk.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** spec 19 (§2, §4 decisions 2-3, §5, §6 D0-D1, §7 19b). Needs 19a's "go".

## Global Constraints

- Branch `spec19b-dflash-drafter` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- **D0:** without a drafter every merged model's kernel binaries are checksum-identical to main's and the full suite is green (`tools/kernel_cmdlines` against main: additions only; `tools/mac_check.sh` once merged).
- Every buffer through `runtime/buffer_sizes.h` and the planner (`--max-len auto` must count the drafter's weights, ring and taps).
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **Ring addressing:** context position p lives in slot p % 2048; the attention reads exactly positions max(0, p - 2048) .. p - 1 plus the block; a test crosses the wrap.
2. **The block is non-causal** and its K/V are not written into the ring (only committed positions are, by 19c's projection).
3. **Shared weights:** the drafter never owns a copy of the embedding or the head; with `--draft-vocab` its candidate head is the compact one, ids mapped through the table.
4. **The mask embedding:** `mask_token_id`'s row of the target embedding, or the checkpoint's `mask_embedding.pt` when shipped (loaded as the row's replacement).
5. **DFlash v1 vs 2 by config:** the convolutions and the selector exist only when `dflash_config` names them; Ornith's v1 drafter (one full-attention layer) loads on the same path once spec 15c is merged.

---

### Task 1: loader

- [ ] `loader::load_drafter(ctx, path, target)`: config (`dflash_config`, `layer_types`, `sliding_window`, `is_causal`), weights (fc, hidden_norm, layers incl. conv and selector parts, final norm), int8 quantisation at load (decision 2) or bf16; refuse a drafter whose `target_layer_ids` exceed the target's layers or whose vocab differs. Host tests on a tiny synthetic checkpoint. **Commit** `loader: DFlash drafters (spec 19b)`.

### Task 2: kernels

- [ ] Context projection (fc + hidden_norm + per-layer k/v + k_norm + RoPE → ring), the grouped convolution (prepare / finish), attention v2 with ring + block, top-16 over head rows (plain and compact-vocab), the selector walk (greedy, and Gumbel at a seed). Host references for each; kernel tests registered for the box. **Commit** `kernels: DFlash drafter kernels (spec 19b)`.

### Task 3: the draft list and D1

- [ ] `build_dflash_draft` (captured once; K up to 7 as a parameter in Control), the drafter buffers through `buffer_sizes` and the planner; D1 on the box against 19a's reference fed the engine's taps (top-16 sets tie-aware equal; scores cosine ≥ 0.9999 bf16 / ≥ 0.999 int8; walk ids equal at T = 0). **Commit** `runtime: the DFlash draft list (spec 19b, D1)`.

**Gate for the plan:** D0 and D1 green on the box.
