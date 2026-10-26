# Spec 17d - tensor parallel: Bolt's snapshots, LiLiCorr, sampling, the server, the record

**Conditions (operator, 2026-10-23; for Bolt - superseded for Qwen3.8 by the 2026-10-27 status below):** this plan starts only after **plans 17b and 17c are merged and their rows passed**
(which needed spec 17 §9's GO: P0 after `r22.devices` / `r22.p1`; Bolt's rows 59-62 passed; the crossing cost <= ~1 ms
a token), and - for Task 2 - after **spec 26c (LiLiCorr in the engine) has passed its row** on `--pp 2`; and the
operator says go.

**Status (2026-10-27): PARKED -> after 17b and 17c (rewritten for Qwen3.8) have passed their rows.** Retargeted to
Qwen3.8 first (spec 17 §R2): snapshots in the one-card host layout, MTP and DFlash under `--tp 2` (the MTP head TP'd or
replicated by 17b's measurement; the DFlash drafter replicated - spec 17 §R2.8), sampling over the split vocabulary,
`b70-serve --tp 2`, T3, D1-dense / D4 against one card's `--mtp auto` and `--spec dflash`, the record. Box queue row
**74**. The 2026-10-23 Bolt-first text below is kept for Bolt's later TP plan; its LiLiCorr task (Task 2) is void -
spec 26 was dropped on 2026-10-27.

**Status (2026-10-23, superseded 2026-10-27): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8: spec 7 per card, MTP, `{card 0 part, card 1 part}` entries) is in git history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** everything else on Bolt under `--tp 2`: snapshots in the **one-card host layout** (spec 17 §4.7), spec 26's
LiLiCorr drafter under TP (spec 17 §4.8), sampling over the split vocabulary (§4.4), `b70-serve <bolt> --tp 2`; T3;
D4; the record. Kolibri is a later plan (17e, not written) on these pieces, after spec 20b's checkpoint and its own P0
line (spec 17 §3).

**Architecture:** spec 17 §4.4, §4.7-§4.9. `BoltEngine::{save,load}_{state,kv}` gather each card's heads (SSM state,
conv-ring channels, its kv head) into `runtime::bolt::{state_runs, kv_runs}`'s one host layout and scatter on restore;
the drafter's placement by spec 17 §4.8's decision (replicated / TP'd / one card) from 17a's numbers and 26a's
`r63.cost`; the taps read locally (both cards hold the residual); verify at M = 2..8 with M-row crossings; the
per-card top-k + partial log-sum-exp merge on the host; `cli::bolt` with `--tp`.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest, uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.4, §4.7-§4.9, §5 T3, §6 D4, §7).
Precedents: spec 25e (snapshots, `BoltEngineAdapterT`, `b70-serve <bolt>`), spec 26c (the tap verify, the slots, the
commit), spec 16d (snapshots byte-identical across modes).

## Dependencies and branch points

- Main with 17b and 17c merged (rows passed); 26c merged and passed for Task 2. Branch **`spec17d-tp-integration`**.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` never committed;
  `oracle-out*` symlinked. No `rm -rf`.
- `--tp 1` unchanged (G0). `--tp 2` exclusive with `--pp 2` and `--replica` (spec 13 §10), refused by name.
- **Every two-card stage is `gpu`**; `r71.benchy` is `gpu-self` like `r62.benchy`.
- A RECORD-grade row (idle box, both cards) before any README headline changes.

## Review Focus

1. **One host layout:** an entry saved under `--tp 2` is byte-identical to the same position's entry under `--pp 2`;
   restores in every direction (`--tp 2` <-> `--pp 2`, every split) give the cold run's tokens - nothing refused.
2. **Sampling is exact:** the merged per-card top-k with the two partial log-sum-exps equals filtering the full logits
   (host test on synthetic split logits, every filter combination); seeded runs reproducible; LiLiCorr's full-vocabulary
   top-8 and log-probs equal under the merge.
3. **LiLiCorr under TP:** verify rows bitwise M = 1 per card (spec 26 D3) at M = 2..8; the taps read locally equal
   `--pp 2`'s; greedy speculative output identical to `--tp 2` plain decode; acceptance within noise of `--pp 2`'s.
4. **The server:** `b70-serve <bolt> --tp 2` with the prefix cache and the drafter on; `golden_server_test --chat`
   against `b70-decode <bolt> --tp 2`.

---

### Task 1: snapshots and sampling (Mac)

- [ ] Failing tests (`bolt_tp_snapshot_test` host, Review Focus 1-2), implement, PASS. Commit
  `git commit -S -m "runtime: Bolt snapshots under --tp 2 in the one-card host layout, the split-vocabulary sampling merge (spec 17d)"`.

### Task 2: LiLiCorr under TP (Mac)

- [ ] The drafter placement per spec 17 §4.8's recorded decision; the M-row verify across the crossings; Review
  Focus 3's tests. Commit `git commit -S -m "runtime: LiLiCorr under --tp 2 (spec 17d)"`.

### Task 3: the server (Mac)

- [ ] `b70-serve <bolt> --tp 2`, the adapters, `bolt_serve_args_test` with the TP refusals. Commit
  `git commit -S -m "server: b70-serve <bolt> --tp 2 (spec 17d)"`.

### Task 4: the box queue row (proposed 71, or the next free row) and the record

```text
row 71 "spec 17d - Bolt under tensor parallel: snapshots, LiLiCorr, the server, the record (plan 17d)"
stage r71.k0      71 default cpu      -                     g0.sha,g0.bitwise,g0.suite "--tp 1 unchanged"
stage r71.host    71 default cpu      -                     -          "bolt_tp_snapshot_test, the sampling merge, bolt_serve_args_test"
stage r71.snap    71 default gpu      bolt_synth            r71.k0     "bolt_snapshot_gpu_test under --tp 2: block ends bitwise cold; --tp 2 <-> --pp 2 restores"
stage r71.spec    71 default gpu      bolt_g128,lilicorr    r71.snap   "LiLiCorr under --tp 2: D3 per card at M 2..8, greedy identity, acceptance vs --pp 2"
stage r71.chat    71 default gpu      bolt_g128             r71.snap   "golden_server_test --chat on b70-serve <bolt> --tp 2 == b70-decode <bolt> --tp 2"
stage r71.speed   71 optin   gpu      bolt_g128,lilicorr    r71.spec   "D4: --tp 2 + LiLiCorr vs --pp 2 + LiLiCorr, A4 + golden, idle, rotated rounds"
stage r71.benchy  71 optin   gpu-self bolt_g128,uvx         r71.chat   "llama-benchy at 886 / 8192 / 32000 / 65536: --tp 2 vs --pp 2, SergiioB's published TP 2 beside"
```
- [ ] docs/BENCHMARKS.md "Tensor parallel (spec 17)" complete; spec 17's as-built section; README only with a
  RECORD-grade row; docs/10-the-box.md (TP's lock: the whole box). Commit
  `git commit -S -m "spec 17: tensor parallel on Bolt - the record"`.

**Gate for the plan:** T3 green, Review Focus 1-4 on the box, D4 and the benchy rows recorded, `--tp 1` unchanged.
