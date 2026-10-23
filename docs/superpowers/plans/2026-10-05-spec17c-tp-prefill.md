# Spec 17c - tensor parallel: Bolt prefill across two cards

**Conditions (operator, 2026-10-23):** this plan starts only after **plan 17b is merged and its row passed on the box**,
which itself needed spec 17 §9's GO (P0 after `r22.devices` / `r22.p1`; Bolt's rows 59-62 passed; the crossing cost
<= ~1 ms a token), **and after row 61's `r61.speed` has measured Bolt's two-card (`--pp 2`) prefill** - the choice
below is made against that number (spec 17 §4.5) - and the operator says go.

**Status (2026-10-23): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8, the PP-hybrid against spec 16c) is in git history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** prefill under `b70-decode <bolt> --tp 2 --prefill` by the option Task 1 picks: **(T)** TP prefill - the
chunk's crossings (32 MiB fp32 a crossing at C = 2048, derived) on a copy engine overlapped with the next projection,
fp32 or bf16 by the accuracy gate; or **(H)** the hybrid - prefill as Bolt's `--pp 2` walk, decode as TP (each card
holds both layouts, or the layers are reshuffled after prefill; the TP planner decides by memory). T2 on prefill; D2.

**Architecture:** spec 17 §4.5. Bolt's `--pp 2` prefill is sequential (spec 25 §6.2), so TP prefill competes with one
card's compute rate; the decision uses `r61.speed` (measured) and P0's `r68.prefill` (the measured crossing rate,
overlapped and not).

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.5, §5 T2, §6 D2). Precedents: spec
25d (`BoltEngine::prefill`, the `nb_pf_*` walk, the 64-grid chunk rule), spec 16c (per-chunk resources doubled, the
back-pressure rule, bounded host waits).

## Dependencies and branch points

- Main with 17b merged; rows 61 (incl. `r61.speed`) and 69 passed. Branch **`spec17c-tp-prefill`** from main.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` never committed;
  `oracle-out*` symlinked. No `rm -rf`.
- `--tp 1` unchanged (G0). The chunk rule unchanged (a multiple of 64 on the absolute 64-grid).
- **Every two-card stage is `gpu`**; idle box, interleaved pairs, median of 3.

## Review Focus

1. **After prefill each card holds exactly its decode-layout state** (its Mamba heads' SSM state and conv-ring channels,
   its kv head's KV), whichever option ran; T1 holds for the first decoded token.
2. **T2 on prefill:** against `--pp 2 --prefill` under spec 25's prefill bars (h taps, SSM state, logits cosine; routes
   with near-ties); chunks of 512 / 1024 / 2048 and splits at multiples of 64 bitwise each other under `--tp 2`.
3. **(T) the overlap is real:** per-chunk crossing time hidden behind compute, shown per block from timestamps.
4. **(H) memory:** both layouts (or the reshuffle's peak) fit beside KV at the target max_len; the plan shows it.
5. **The snapshot hook** (25e's) under `--tp 2` fires when both cards have finished the block.

---

### Task 1: the choice (Mac + the box's numbers)

- [ ] From `r61.speed` (Bolt `--pp 2` prefill, measured) and `r68.prefill` (the crossing rate), the derived (T) and (H)
  rates at 886 / 8192 / 32000 / 65536; the choice and a proposed D2 bar written into spec 17 §4.5. Commit
  `git commit -S -m "spec 17: Bolt's TP prefill option, from rows 61 and 68 (spec 17c)"`.

### Task 2: the chosen path (Mac)

**Files:** `src/runtime/bolt/bolt_prefill*.{h,cc}`, `src/kernels/bolt/nb_pf_*.cl` (per-card variants only),
`src/runtime/bolt/bolt_sizes.{h,cc}`, new `tests/prefill/bolt_tp_prefill_plan_test.cc` (host) and
`tests/prefill/bolt_tp_prefill_test.cc` (cards).

- [ ] Failing tests (Review Focus 1, 2, 4, 5), implement, PASS on the Mac's host half. Commit
  `git commit -S -m "prefill: Bolt prefill under --tp 2 (<option>) (spec 17c)"`.

### Task 3: the box queue row (proposed 70, or the next free row)

```text
row 70 "spec 17c - Bolt prefill under tensor parallel (plan 17c)"
stage r70.k0     70 default cpu -                     g0.sha,g0.bitwise,g0.suite "--tp 1 unchanged"
stage r70.host   70 default cpu -                     -          "bolt_tp_prefill_plan_test"
stage r70.synth  70 default gpu bolt_synth            r70.k0     "bolt_tp_prefill_test on the N 12 synthetics: Review Focus 1, 2, 5"
stage r70.golden 70 default gpu bolt_g128,oracle_bolt r70.synth  "SergiioB's g128: T2 on prefill against --pp 2"
stage r70.speed  70 optin   gpu bolt_g128             r70.golden "D2: --tp 2 vs --pp 2 prefill at 886 / 8192 / 32000 / 65536, idle, interleaved, median of 3; per-card busy time"
```
- [ ] Commit `git commit -S -m "docs: box queue row 70 - Bolt TP prefill (spec 17c)"`.

**Gate for the plan:** T2 on prefill green against `--pp 2`, the chunk / split bitwise checks, D2 measured, `--tp 1`
unchanged.
