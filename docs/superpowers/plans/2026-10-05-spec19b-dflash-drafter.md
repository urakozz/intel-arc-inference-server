# Spec 19b - the DFlash drafter on the card

**Status (2026-10-16): Tasks 1-3 built on the Mac (branch `spec19b-dflash-drafter`); the plan's gate - D0 and D1 on
the box - is box queue row 54.** As built: spec 19 §10 "19b as built". Built regardless of P0 (spec 19 §7, the
operator's ruling of 2026-10-15).

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
3. **Shared weights:** the drafter never owns a copy of the embedding or the head. With `--draft-vocab` its candidate head is spec 8 §11's compact one, and the top-16 **indices are mapped to real token ids through the table before anything else reads them**: the selector's codebooks (`pred` / `succ`, indexed by token id), the walk's predecessor, the anchor of the next block and the q readback all see real ids, never compact indices. A test drafts the same block with and without the draft vocabulary and checks that, whenever all 16 full-head candidates are inside V′, the candidates, scores and walk are identical.
4. **The mask embedding:** `mask_token_id`'s row of the target embedding, or the checkpoint's `mask_embedding.pt` when shipped (loaded as the row's replacement).
5. **DFlash v1 vs 2 by config:** the convolutions and the selector exist only when `dflash_config` names them; Ornith's v1 drafter (one full-attention layer) loads on the same path once spec 15c is merged.

---

### Task 1: loader

- [x] `loader::load_drafter(ctx, path, target)`: config (`dflash_config`, `layer_types`, `sliding_window`, `is_causal`), weights (fc, hidden_norm, layers incl. conv and selector parts, final norm), int8 quantisation at load (decision 2) or bf16; refuse a drafter whose `target_layer_ids` exceed the target's layers or whose vocab differs. Host tests on a tiny synthetic checkpoint. **Commit** `loader: DFlash drafters (spec 19b)`.

### Task 2: kernels

- [x] Context projection (fc + hidden_norm + per-layer k/v + k_norm + RoPE → ring), the grouped convolution (prepare / finish), attention v2 with ring + block, top-16 over head rows (plain and compact-vocab), the selector walk (greedy, and Gumbel at a seed). Host references for each; kernel tests registered for the box. **Commit** `kernels: DFlash drafter kernels (spec 19b)`.

### Task 3: the draft list and D1

- [x] (Mac half; D1 itself is box row 54: r54.d1 / r54.d1_ref) `build_dflash_draft` (captured once; K up to 7 as a parameter in Control), the drafter buffers through `buffer_sizes` and the planner; D1 on the box against 19a's reference fed the engine's taps (top-16 sets tie-aware equal; scores cosine ≥ 0.9999 bf16 / ≥ 0.999 int8; walk ids equal at T = 0). **Commit** `runtime: the DFlash draft list (spec 19b, D1)`.

**Gate for the plan:** D0 and D1 green on the box.

## As built (2026-10-16) - and where the tree's current patterns replaced the plan's

Each deviation from the 2026-10-05 text, with what it costs if it is wrong:

- **Files** follow the per-model module pattern main grew since (K2, Kolibri, Muse): `model/dflash`, `loader/dflash_repack`
  (host half) + `loader/dflash_loader` (upload), `kernels/dflash/df_*.cl` + `kernels/dflash_kernels.h` (+ `dflash_dims.h`,
  plain constants for host code), `runtime/dflash/{dflash_sizes,dflash_buffers,dflash_capture,dflash_control}`. The
  plan's `buffer_sizes` became `runtime/dflash/dflash_sizes` beside `runtime::DrafterPlan` in `memory_plan.h` (the
  drafter's sizes depend on its config, not on `ModelDesc`). Cost if wrong: file moves.
- **`load_drafter(ctx, path, target)`** takes a `LoadedModel`, plus an overload by parts (the target's descriptor, head
  rows and embedding) for the device tests.
- **Review Focus 1's window** is the D1 reference's (`dflash_ref.py`, vLLM's symmetric non-causal window): block row r
  at p + r sees context >= p + r - 2047, not "max(0, p - 2048) .. p - 1". Cost if wrong: one bound in df_attn.
- **"K up to 7 as a parameter in Control"**: K (with the walk's temperature / seed and the projection's position / row
  count) lives in the drafter's own control block `DFlashCtl` (spec 8's hctl precedent), so `runtime::Control` keeps its
  layout; the anchor is read from and the drafts written to `runtime::Control` (`cur_token[1..K]`, what the verify list
  reads). Cost if wrong: moving six words.
- **Attention** is a new pair (`df_attn_part` / `df_attn_reduce`), not attn_v2 with a ring mode: attn_v2 is
  Qwen3.8's 256-dim head with partial RoPE and gating; the drafter's heads are 128-dim with a ring and a block term.
  attn_v2.cl and every existing binary are untouched (D0). Cost if wrong: a merge into attn_v2 later.
- **Review Focus 4's `mask_embedding.pt`** is refused (a torch pickle) naming `tools/oracle/dflash_mask_embedding.py`,
  which writes `mask_embedding.safetensors`; that file replaces the row. No published drafter ships one. Cost: a pickle
  reader if one appears.
- **Review Focus 5:** v1 configs parse, load and draft greedily (top-1 a row); v1 sampling (full-vocabulary Gumbel) is
  refused by `runtime::dflash::set_draft_params` (and df_select's v1 build drafts nothing - readback k 0 - at T != 0, the
  final review's fix); causal and full-attention layers (Ornith's v1 drafter) are refused at capture - their binaries are
  Ornith's stage.
- **Formats:** bf16 and int8 at load (decision 2). int4 g64 drafter weights are not built (no checkpoint of ours; 19a:
  equal acceptance). Cost if wrong: an RTN int4 path (loader/rtn.h) and int4 GEMV variants at M 8.
- **The draft vocabulary** reuses `LoadedModel::draft_vocab`, which `loader::load` still builds only with the MTP head;
  D1's V' arm loads the head. 19c relaxes that rule with `--spec dflash`. Cost: one condition in loader.cc.
- **D1's taps** are 19a's CPU-reference dumps (oracle-out-19a), fed to both sides - "the engine's taps" arrive with
  19c's verify list; `dflash_d1` reads any dump in that format. Cost: a re-run on engine dumps.
- **Mac-side tests beyond the plan:** the twin is held to `dflash_ref.py` on a python-made fixture
  (tests/loader/dflash_ref_fixture.safetensors, 544 KB); `tools/mac/clrun/dflash_run` runs both lists on the Mac's GPU
  (mac_check section 5); `dflash_draft_test` (box) runs both lists with the real drafter beside a synthetic target.
- **Precision note for D1:** the engine's GEMV inputs are bf16 (the reused kernels' contract); a Mac preview on the
  real drafter put the final rows at cosine 0.99991 of the fp32 reference. If r54.d1's scores bar (0.9999) fails while
  the sets and walks hold, fp32-input GEMV variants (gemv_i8w / gemv_bf16 with an fp32 x) are the lever.
