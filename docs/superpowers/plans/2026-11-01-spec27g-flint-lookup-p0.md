# Spec 27g - Flint speculative decoding: the expert-union measurement and the verify-cost P0 for prompt lookup (conditional)

**Status (2026-11-01): plan written, conditional - started only after 27a-27f and the operator's go (decide D8).** Flint has no MTP head; DFlash has no Flint drafter; a draft model is rejected (spec 27 section 4). Prompt lookup (spec 19e) is the one candidate.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** decide with measurements whether lookup is worth building: (1) the real expert-union growth at M = 2..8 from Flint's own route records, (2) the real verify cost on the card for the cheapest honest implementation, (3) the acceptance on recorded agentic transcripts. Gate S1: build only if lookup is >= 1.25x over plain decode on the recorded agentic traces at the measured verify cost.

**Architecture:** spec 27 section 4 (union formula `256 x (1 - (1 - 6/256)^M)`: 11.9 / 17.6 / 23.2 / 44.3 experts at M = 2 / 3 / 4 / 8; ~1.9 plain steps at M = 4 by bytes at zero depth; ~1.45 at 100k depth; estimates, upper bounds if routes correlate). Precedents: spec 19e and `docs/probe-prompt-lookup-2026-10-05.md` (the host matcher `src/server/prompt_lookup.{h,cc}`, 1.44x over plain at n = 3, K <= 3 on Qwen's A4 replay, 3.10 tokens per verify at K = 3), spec 28 and 28c (small-M verify; the dense GEMVs are covered, **the expert GEMVs are explicitly excluded**, the attention rule is Qwen-shaped), a new `tools/spec/expert_union.py`, `docs/probe-lilicorr-2026-10-23.md` section 5 (plain-M-row MoE verify 1.19 .. 2.35 plain steps at M = 2..8 for Bolt).

## Global Constraints
- Branch `spec27g-flint-lookup-p0`; signed commits; no merge, no push; GPU jobs under `~/b70-gpu.lock`; interleaved pairs after warm-up for every timing.
- A probe, not an engine change: nothing in `src/runtime/flint` changes in this plan except under an `S1`-gated task the operator releases.

---

### Task 1: the expert union from real routes (Mac / box CPU)
- [ ] **Step 1:** write and run `tools/spec/expert_union.py` (new; only `lookup_accept.py` exists in `tools/spec`) over the route records 27a's golden runs wrote (recorded agentic prompts, per MoE layer, per row): the empirical union size for windows of M = 2..8 consecutive rows against the independent-routing formula. Record the table and per-layer spread (do some layers correlate strongly?). **Commit** the doc.

### Task 2: the acceptance on recorded traces (Mac)
- [ ] **Step 1:** replay recorded agentic transcripts (the A4 set and any recorded coding session; never invent traffic) through `prompt_lookup` offline: tokens per verify at K = 1..5, n = 2..4 (the matcher is generic host code). The opencode recording that decides is not in the repo: record the gap and ask the operator for one if the A4 replay is the only trace. **Commit.**

### Task 3: the verify cost on the card (box)
- [ ] **Step 1:** a probe `tools/probe/probe_flint_verify.cc`: a verify step at M = 2..6 built from existing pieces (dense GEMVs at M rows via spec 28's `gemm_sm` if it exists by then, else the M x single-row GEMV loop; the MoE as M sequential routed blocks, i.e. the honest upper bound; the attention at M rows per full layer reading the KV once), against the plain step, at depth 886 / 32k / 100k, interleaved. **Step 2:** the union-aware MoE variant as a second arm only if Task 1 shows the union is much below `M x 6`. **Step 3:** compute the lookup speed-up = tokens per verify / (verify ms / plain ms) per (K, depth). **Commit** `docs/probe-flint-lookup-2026-11-<dd>.md`.

### Task 4: the decision (operator)
- [ ] Report S1's number and the effort: build (a verify list at M <= 5 with the sink attention at M rows, an M-row MoE path, `truncate_to`, the matcher wiring, the F3 / F4 gates for verify: tokens equal plain greedy bitwise-or-tie-aware) or stop. Estimated build: 3 implementer-days (spec 27 section 10).

## Gates
- The recorded union table, the acceptance table, the verify-cost table and S1's verdict; no engine change.
