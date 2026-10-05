# Spec 19c - DFlash in the decode loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** speculative decoding with the drafter end to end in the engine: taps written by the verify list, the commit-side context projection, the prefill tail, verify at M = K + 1 ≤ 8 bitwise to M = 1, host acceptance with the drafter's sparse q; D2, D3, D4.

**Architecture:** spec 19 §5 "Per iteration": draft list (19b) → spec 8's verify list at M = K + 1 (writes the taps of every row) → host acceptance (spec 8's `spec_accept`, q sparse over 16 candidates) → commit list (spec 8's GDN slot index and KV length, then the projection of rows 0..j into the ring, the row count read from Control). Prefill writes the taps of the positions inside the window and fills the ring.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** spec 19 (§5, §6 D2-D4, §7 19c); spec 8 §3-§4, §8 (A4, the Control index), §10 (AdaptiveK), §11 (draft vocabulary). Needs 19b merged.

## Global Constraints

- Branch `spec19c-dflash-loop` from main; box tree automatic; `tools/box.env` never committed.
- D0 holds (kernel binaries of every path without `--spec dflash` unchanged). Lock, detached, polled.
- Spec 8's GDN slots grow to K + 1 = 8 with `--spec dflash`: memory through the planner (7 x 151 MB on Qwen3.8).
- No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **Taps come from the verify rows that get committed** (rows 0..j); rejected rows' taps are never projected.
2. **The newest token** (the target's sample after acceptance) has no tap yet; it is the next block's row 0, as in vLLM.
3. **M = 5..8 bitwise** to M = 1 for logits and GDN slots (D3) on Qwen3.8; the verify list's attention at M rows reads the KV once.
4. **Sparse q:** the acceptance rule handles q = 0 outside the candidates (as spec 8 §11 does for V′) and the residual distribution stays finite.
5. **Prefix cache interplay:** without decision 5's plane (19d) a restored request starts with a cold ring; it must still be correct (lower acceptance only).

---

### Task 1: taps and the commit projection

- [ ] The verify list writes taps (`tap_rows[M][n_taps][hidden]` bf16); the commit list projects rows 0..j into the ring. Test: projected ring rows equal 19b's context projection of the same taps. **Commit** `runtime: taps and the commit-side projection (spec 19c)`.

### Task 2: verify at M ≤ 8 and the loop

- [ ] Verify variants at M = 5..8 (from 19a's probe build), GDN slots to 8, `Engine::step_many` with the drafter as the proposer, `--spec dflash` / `--draft-model PATH` / `--spec-max` in `b70-decode` (ids mode) and `b70-serve`, the AdaptiveK cost row flat in K (`--spec-cost`). D3 and D2 on the box (golden set and A4, every K). **Commit** `runtime: DFlash speculative decoding (spec 19c, D2-D3)`.

### Task 3: sampling and the prefill tail

- [ ] Sparse-q acceptance host tests (D4: 10^6 seeded samples within spec 8 M4's bars; no draft outside the candidates); prefill writes the window's taps and fills the ring; `prefill_split_*` with the drafter on. **Commit** `server: DFlash sampled acceptance and prefill taps (spec 19c, D4)`.

**Gate for the plan:** D2, D3, D4 green; D0 green.
