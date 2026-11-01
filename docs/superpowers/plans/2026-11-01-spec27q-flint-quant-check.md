# Spec 27q - Flint: the operator's AutoRound quant checked against the engine's rules, and gate Q

**Status (2026-11-01): plan written, nothing built.** Needs 27a's reference, golden prompts and A4 set. Box CPU only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** prove that `urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ` is loadable by the engine's rules and measure how far it is from the BF16 original (**gate Q**, a recorded measurement in the spec 23 section 3.3 style). **No re-quantisation** (spec 27 section 6, D6): our own AutoRound run is only started if the operator asks after reading gate Q.

**Architecture:** spec 27 sections 1, 6, 9 (F5). Precedents: `tools/quantize/check_gptq_export.py`, `src/loader/quant.cc:331-430` (accepted export rules), `tools/oracle/dequant.py`, plan 25a's gate Q task, `tools/box_validate/f32_oracle.sh`.

## Global Constraints
- Branch `spec27q-flint-quant-check`; signed commits; no merge, no push; CPU model lock `/tmp/b70-cpu-model.lock` around every real-weight run; RAM floor checked; detached runs; no GPU, no GPU lock.
- The operator downloads the weights (the box has both repos; this plan never starts a download).
- Nothing about the quant is edited; findings are recorded, not fixed.

---

### Task 1: the export check (Mac for the index, box for the data)
**Files:** modify `tools/quantize/check_gptq_export.py` only by a new `--flint` expectation table (existing modes untouched, a failing test first); create `tools/quantize/flint/test_flint_export.py`.
- [ ] **Step 1: failing tests** on the tiny `int4-g64` variant of 27a: the key set per layer type (dense layers 0-1, MoE 2-48, per-expert unfused, shared expert, gate unquantised), qzeros 0x77777777 everywhere, scales finite and f16, `quantization_config` fields (bits 4, sym, g64, `auto_round:auto_gptq`, no `desc_act`), the extra_config gate rule for layers 2-48, the tensor count 109,612 on the real index (range-read on the Mac).
- [ ] **Step 2: implement; PASS; run on the real index and headers** (box copies). Record every deviation, any tensor the loader would not consume (27b's list), the dtypes. **Commit** `quant: Flint export check against the engine's loader rules (spec 27q)`.

### Task 2: gate Q (box CPU)
- [ ] **Step 1:** with 27a's `flint_ref.py` and `flint_oracle.sh gateq`: the golden prompts through (a) the BF16 original, (b) the quant dequantised (the engine-format reference). Per row: KL(a || b), top-1 agreement, cosine distance of the logits; perplexity of both on a long agentic transcript at 4k, 32k and 64k (layer-streamed, detached; hours).
- [ ] **Step 2:** the A4 set through both (27a's references; the BF16 one is the A4 reference): pass counts and per-scenario differences; a small SWE-style prompt set if 27a made one.
- [ ] **Step 3:** record in `docs/probe-flint-quant-2026-11-<dd>.md`: the table, the verdict as a **recorded measurement with no bar** (spec 23 section 3.3 ruling style), and the hole list (per-tensor RMSE of the dequantised weights against the original, top 20: cheap, read-only). **Commit** `docs: Flint gate Q - the operator's int4 g64 against the BF16 original (spec 27q)`.

## Gates
- **F5 (part):** the export passes the loader rules or every deviation is a named item for 27b; gate Q is recorded with the table above; the per-tensor error scan is recorded. No threshold decides anything automatically: the operator reads gate Q and chooses D6.
