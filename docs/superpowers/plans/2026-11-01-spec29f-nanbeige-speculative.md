# Spec 29f - Nanbeige4.3-33B speculative decoding: acceptance P0, lookup on the dense verify list, the DFlash2 drafter generalised, int8 KV with it, the measured cost table

**Status (2026-11-01): PLANNED - after 29e AND spec 28b (being built), spec 28c (design only, no plan) and plan 19f (being built).** Nothing built. No box queue row. This plan starts only when those are merged (or the operator waives one by name). Task 2 (the acceptance P0) is CPU work on 29c / 29d's tap dumps and needs none of the three: the operator may run it early.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed; `WIP:` if needed); handoff file when you stop.

**Goal:** the speed at depth. Nanbeige has no MTP head, so: (1) **lookup decoding** (no weights) on the dense verify list; (2) the third-party **DFlash2 drafter** (`ProCreations/Nanbeige4.3-33B-DFlash2`) if its measured acceptance clears the stop rule, with the engine's DFlash2 build generalised from Qwen3.8's shapes; (3) both under `--kv-cache int8` and `--pp 2`; (4) the measured verify-step cost table and the speculative speed rows that replace spec 29 §6's estimated tau.

**Architecture:** spec 29 §6, §7. Precedents: spec 19 (draft-model speculative decoding; plans 19a P0 / 19b-d, 19e lookup, 19f lookup-in-DFlash), spec 8 (verify lists, commit, near-tie rule), spec 28 / 28b / 28c (small-M GEMM, verify attention by shape), `server/adaptive_k.h` (`AdaptiveK`, `MtpCost`), `server/prompt_lookup.h`. The dense target has **no recurrent state**: verify writes KV at n..n+k, commit sets `pos = n + j + 1`, nothing rolls back (spec 8 design, KV-only).

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md` (§7 is the design; its decisions 1-8 are this plan's tasks).

## Dependencies and branch points

- 29e merged (served engine); 29c's tap dump (Task 8) and 29a's judge taps; **spec 28b merged with Nanbeige's five shapes added to its table (operator's confirmation, spec 29 open decision 1)** and 28c merged; **plan 19f merged** (lookup-in-DFlash policy; the dense family adopts whatever it ships). If 28b is waived, M = 8 verify costs ~3.1-3.2 plain steps (Qwen3.8 measured: GEMM sum 3.10x, whole verify(8) 3.17, busy box) and the drafter's value shrinks accordingly - the plan then measures first and says so.
- **Third-party weights:** the drafter repo is the operator's download (`hf download ProCreations/Nanbeige4.3-33B-DFlash2 --revision 22a73c4d099e61a34b6a26251b0a3a98a7b41746`; Apache 2.0 with `SPECFORGE-LICENSE.txt` / `NOTICE.md`, 979,731,200 bf16 parameters, no embedding / head, config schema as spec 29 §7.2). **Format is an operator decision (spec 29 open decision 2):** the shipped BF16 linears quantised to int8 at load is the z-lab precedent; AutoRound int4 g64 by us needs a loader path main does not have.
- Branch `spec29f-nanbeige-speculative`.

## Global Constraints

- Signed commits; no merge, no push; no `rm -rf`.
- **G0:** every other model's binaries checksum-identical; the Qwen3.8 DFlash path's binaries and results unchanged (the generalisation adds shape-parameterised variants; `kQwen38`'s names and bytes stay).
- **No new CLI flag and no environment variable:** `--spec off|lookup|dflash`, `--spec-max`, `--draft-model`, `--draft-format`, `--draft-vocab` exist and are reused; `--mtp` is refused by name for this family.
- **Greedy is lossless under the near-miss rule** (spec 8 M3 / spec 19 D2): zero non-tie divergences against `--spec off`.
- The drafter is a **proposer**: its quality moves acceptance, never the output (a wrong draft is rejected by the verify).

## Review Focus

1. **The stop rule is honoured.** If P0's tau on A4 and the agentic session at 32k is < 2.5 with the better of window / full context, the DFlash tasks (4-6) are **not built**; lookup ships alone and the plan records why.
2. **The target-side taps.** Layers 1, 25, 49 (zero-based layer *outputs*, the residual before the final norm; HF `hidden_states[2, 26, 50]`) - a test pins the convention on the tiny model (an off-by-one here silently halves acceptance).
3. **`kTopK` 16 -> 32, fc 5 x H -> 3 x H, rank 256 -> 128, mask id 0, eps 1e-5, theta 1.2e8** are descriptor-driven in the new variants; `check_common`'s Qwen pins become descriptor checks with the same strictness (a descriptor that is neither the Qwen3.8 shape nor the Nanbeige shape is refused by name).
4. **Window vs full-context drafter attention** is a measured choice (spec 29 §7.3 item 2); the full-context form's KV (6,144 B per position) is in the memory plan and refused when the context would not fit.
5. **`--pp 2` with the drafter:** taps from both stages reach the drafter; the anchor's embedding row crosses with the hand-off (no embedding replica on stage 1, unlike Qwen3.8's 2.54 GB one). Spec 19's `--pp 2` DFlash has run on the cards for Qwen3.8 (r56.d5_pp, ids identical); the dense binding is new code and is validated on its own.
6. **Verify at M <= 8 on the dense list:** `dn_attn` accepts M up to 8 (K2's limit of 4 is lifted in the copy, with spec 28c's by-shape variants), `gemm_sm` binds at M >= 2 (28b's `gemv_kind(M)`), rows bitwise across M >= 2 (28's property R) and equal to M = 1 within the cosine bars of spec 28 N4.

---

### Task 1: lookup on the dense verify list

**Files:** Create `src/runtime/dense/dense_verify.{h,cc}` (`build_verify` at M = 2..8 for the dense capture, KV-only commit), `tests/runtime/dense_verify_test.cc`; modify `src/cli/dense_serve.h` (the external-draft hook), tests `tests/server/dense_lookup_test.cc`.

- [ ] **Step 1: failing tests:** verify(M) rows equal M single decodes' logits within spec 28's N4 bars; commit(j) then 64 greedy tokens equal a run that never saw the rejected drafts; the server loop at `--spec lookup` against `--spec off` (zero non-tie divergences on golden + A4). **Step 2: implement. Step 3: card, one stage then two:** correctness, then **the lookup speed** on A4, the agentic session (29a's, a long file-edit turn set) and golden: ids per step and wall clock against plain, at depths 4k / 32k / 128k. **Step 4: commit** with the measured step costs M = 1..8 (the dense `MtpCost` verify row, measured at depth 4k / 32k / 131k).

### Task 2: acceptance P0 for the drafter (box + CPU, no kernel)

**Files:** Create `tools/oracle/dense_dflash_ref.py` (the vendor's `DFlash2DraftModel` forward in PyTorch from its config and weights, window / full-context switch; `dflash_ref.py`'s arrangement), `tools/spec/dense_accept.py`, `docs/probe-nanbeige-dflash-2026-11-<dd>.md`.

- [ ] **Step 1:** load the drafter and the target's embedding / head from the snapshots; run it on 29c's tap dumps (teacher-forced decode over golden / A4 / the agentic session) at depths **4k, 32k, 128k, 240k** (taps at depth come from the engine's long-context runs, one pass over the session prefix), window 2048 and full context; per anchor the accepted-run length at K = 7: tau per corpus, per depth, per position (E_1 / E_3 / E_7 as the Qwen probe's table). **Step 2: the stop rule and the window decision** (tau >= 2.5 on A4 and the session at 32k; full context only if its tau at 128k+ is >= 0.4 above the window's). The drafter was trained full-context and published only to 8k: the window-2048 arm is outside its training, so a low (a) is expected and is not by itself a verdict on the drafter. **Step 3:** the verdict in the doc and spec 29 §7.3; **commit.** **If the stop rule fails, skip to Task 7.**

### Task 3: the drafter descriptor and loader generalisation

**Files:** modify `src/model/dflash.{h,cc}` (descriptor already generic; add the checks), `src/loader/dflash_{layout,repack}.cc` (verify q_n / kv_n / fc_k at 3 x 7168 -> 7168), `src/cli/dflash_args.h` (`--draft-model` default stays z-lab's for Qwen; for the dense family it is **required**); tests `tests/model/dflash_dense_test.cc`, `tests/loader/dflash_dense_layout_test.cc`.

- [ ] **Step 1: failing tests** on a tiny drafter built from the ProCreations drafter's config shape (3 layers, hidden 7168 scaled down, 28 / 4 heads x 128 -> a tiny variant, taps, selector rank 128 / top-k 32); **Step 2: implement; Step 3: commit.** Format: int8 at load first (`--draft-format int8`, 0.98 GB, the z-lab precedent - subject to the operator's open decision 2). An int4 g64 sym drafter (the operator's AutoRound job on the ProCreations weights, ~0.52 GB) has **no path on main**: `DraftFormat::Int4G64` was dropped with Muse (98c13ef) and was RTN at load (`git show 99adbb0:src/loader/dflash_layout.h`, `dflash_repack.cc`); if the operator chooses int4, this task revives it under the same name and makes it read the `auto_round:auto_gptq` bytes - a scope addition, stated in the handoff.

### Task 4: the drafter's kernels at the Nanbeige shape

**Files:** modify `src/kernels/dflash_dims.h` (`kTopK` becomes a per-shape constant), `src/kernels/dflash_kernels.h` (`Shape kNanbeige`: H 7168, I 8192, NH 28, NKV 4, HD 128, rank 128, top-k 32, vocab 166912 used 166107, fc 3 x H; `draft_variants` takes a Shape), `src/kernels/CMakeLists.txt` (Spec 19b block variants for the new shape; the Qwen3.8 names unchanged), `tests/kernels/dflash_variant_names_test.cc`, `tests/kernels/dflash_dense_kernels_test.cc`.

- [ ] **Step 1: failing tests:** each drafter kernel against its host twin at the Nanbeige shape (prep with theta 1.2e8 and eps 1e-5, the conv, attention over the window / full-context ring, the selector at top-k 32, the drafter's int8 GEMVs). **Step 2: implement.** If Task 2 chose **full context**: the drafter attention over a growing per-layer KV (6,144 B per position, written by the commit-side projection of every accepted / prefilled position; prefill also runs the drafter's fc + 3 layers over context chunks: +3-4 % prefill work, measured). **Step 3: commit.**

### Task 5: the dense engine's drafter binding and `--kv-cache int8`

**Files:** create `src/runtime/dense/dense_dflash.{h,cc}` (the tap list at layers 1 / 25 / 49: `build_verify_taps` for the dense capture; `TapDrafter` implementation; the anchor embedding gather), modify `src/cli/dflash_args.h` (the int8-KV refusal lifted for the dense family only, with a test that Qwen3.8 still refuses), `src/cli/dense_serve.h`.

- [ ] **Step 1: failing tests:** tap convention pinned (Review Focus 2); verify-with-taps rows equal verify rows; commit-side projection; one-stage and two-stage taps equal; DFlash at `--spec dflash` vs `--spec off` greedy on golden + A4 (zero non-tie divergences), bf16 and int8 KV, one stage. **Step 2: implement. Step 3: commit.**

### Task 6: the cards: DFlash, two stages, depth

- [ ] **Step 1 (one stage):** D2 for the drafter kernels, the golden-with-drafter equality (N6), ids per step and wall clock on A4 / golden / the agentic session at 4k / 32k. **Step 2 (two stages):** the same at `--pp 2`, int8 KV, depths 32k / 131k / 240k / 330k - the numbers that matter: spec 29 §6's derived 40.4 / 28.9 / 21.9 / 18.3 t/s at tau 2.9 against the measured. **Step 3:** `AdaptiveK` on the measured cost table (K by corpus and depth); the memory plan's line with the drafter (the 10-18k positions lost; the planner's max-len with `--spec dflash`). **Step 4: commit** BENCHMARKS rows. A two-card DFlash failure (spec 19's `--pp 2` path was never run) is debugged here and recorded.

### Task 7: the combined policy and the verdict

- [ ] **Step 1:** lookup-in-DFlash per plan 19f's policy (ENTER on a match >= 8 beyond the window, STAY while accepts >= 3) if 19f shipped it for the dense family's list; else lookup and DFlash as separate modes. **Step 2:** the decision table: per corpus and depth, plain / lookup / DFlash / combined t/s and the default `--spec` recommendation for the dense family (the operator decides the default; the server's default stays `off` until then). **Step 3: commit** the verdict in `docs/probe-nanbeige-dflash-...md` and spec 29 §6 with the measured rows beside the estimates.

### Task 8: docs

- [ ] **Step 1:** `docs/19-running-models.md` (Nanbeige: `--spec lookup`, `--spec dflash`, `--kv-cache int8` with it, the measured rows), spec 19 §13 note (the dense family and the int8-KV refusal lifted for it), spec 29 "29f as built". **Step 2: commit.**

**Gate for the plan:** Mac - host tests, `mac_check.sh`, `kernel_cmdlines` additions only. Box - lookup lossless under the near-tie rule; **P0's verdict recorded either way**; if DFlash was built: D2 for its kernels, N6 (zero non-tie divergences), the speed rows at depth beside spec 29 §6's estimates with the memory line, the two-stage run completing at 240k; G0 bitwise for every other model including the Qwen3.8 DFlash gates.

**Box stages (prose):** (1) lookup on the card: correctness then speed, short. (2) the acceptance P0 - taps from the engine at depth (long jobs: 240k prefill to dump taps is ~14-26 min), the drafter on CPU / small, hours. (3) only if the stop rule passes: DFlash kernels' unit gates, the drafter in the engine on one stage, then two stages with int8 KV and depth rows. All under the GPU lock, interleaved pairs; the quantised drafter (the operator's job) is an optional second pass of (3).
