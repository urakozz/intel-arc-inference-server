# Spec 19d - DFlash in the server: prefix cache, batching, the record

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve --spec dflash` in long agentic sessions: the drafter's ring survives a prefix-cache restore (decision 5), the batching hook, D5's speed rows against `--mtp auto` on the opencode recording, the default decided, the record.

**Architecture:** spec 19 §4 decision 5 (proposed: store each 2048-position block's `fc`-output plane, 10 KB per position, and re-project on restore), §5 "Batching", §6 D5.

**Tech Stack:** C++17, Level Zero, Python (A4), uvx llama-benchy.

**Spec:** spec 19 (§4 decision 5, §5, §6 D5, §7 19d); spec 7 (store, snapshots), spec 13 (scheduler). Needs 19c merged.

## Global Constraints

- Branch `spec19d-dflash-serving` from main; box tree automatic; `tools/box.env` never committed.
- D0 holds. Lock, detached, polled; interleaved pairs, median of 3.
- The prefix store's extra plane lives in system RAM; `--prefix-cache-gb auto`'s size covers it.
- No recursive force deletes. Commit on the branch; no merge, no push.

## Review Focus

1. **A restore at block b** re-projects block b - 1's plane into the ring; the ring after restore equals the ring of an uninterrupted run (bitwise, same kernels).
2. **Store keys carry the drafter** (a store written with one drafter is not restored into another).
3. **Batching:** one ring per slot; slots never read each other's ring.
4. **D5 comparability:** the same flags, the same recording, `--mtp auto` and `--spec dflash` interleaved.
5. **The default** changes only on D5's bar (≥ 10 % over `--mtp auto` on the recording, no loss > 3 % elsewhere).

---

### Task 1: the prefix-cache plane

- [ ] Decision 5 as ruled; `snapshot_test` / `prefix_gpu_*` with the drafter; the cold-ring fallback kept as the option. **Commit** `server: the DFlash plane in the prefix store (spec 19d)`.

### Task 2: batching hook

- [ ] Per-slot rings and the draft list over `BatchEngineIface` (spec 13c); scheduler tests with a mock drafter. **Commit** `server: DFlash under the batching scheduler (spec 19d)`.

### Task 3: speed and the record

- [ ] D5 rows (llama-benchy at the operator's flags; the opencode recording) for `--spec off`, `--mtp auto`, `--spec dflash` with and without `--draft-vocab`; BENCHMARKS section "DFlash (spec 19)"; the default decided; README flag rows; spec 19 amendment. Ornith's DFlash (v1) drafter after spec 15c: the same rows on Ornith. **Commit** `spec 19: DFlash served - the record`.

**Gate for the plan:** D5 recorded; D0-D4 still green.
