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

- [x] Decision 5 as ruled; `snapshot_test` / `prefix_gpu_*` with the drafter; the cold-ring fallback kept as the option. **Commit** `server: the DFlash plane in the prefix store (spec 19d)`.

### Task 2: batching hook

- [x] Per-slot rings and the draft list over `BatchEngineIface` (spec 13c); scheduler tests with a mock drafter. **Commit** `server: DFlash under the batching scheduler (spec 19d)`.

### Task 3: speed and the record

- [x] (Mac half; the rows are the box's) D5 rows (llama-benchy at the operator's flags; the opencode recording) for `--spec off`, `--mtp auto`, `--spec dflash` with and without `--draft-vocab`; BENCHMARKS section "DFlash (spec 19)"; the default decided; README flag rows; spec 19 amendment. Ornith's DFlash (v1) drafter after spec 15c: the same rows on Ornith. **Commit** `spec 19: DFlash served - the record`.

**Gate for the plan:** D5 recorded; D0-D4 still green.

## As built (2026-10-16) - and where the tree's current patterns replaced the plan's

Branch `spec19d-dflash-serving`, written on the Mac after 19b / 19c (both unvalidated); box queue row 56.
**Status: the Mac half is done; D5 and every card gate are the box's** (no new binary: D0 is G0). Spec 19 §12
"19d as built" is the account; each deviation from this plan, with its cost if wrong:

- **Decision 5 (never ruled by the operator) - the whole ring in every snapshot, not the per-block fc plane.** A plane
  per block cannot be produced for blocks finished in decode or mid-call outside the call's last window (the device keeps
  only the last window's rows); a snapshot is always taken at its own position. Bitwise by construction, no kernel, a
  restore is a copy. Cost if wrong: +41.9 MB a snapshot (~+14 % of the store a block) against the plane's ~7 %; the
  per-snapshot plane (20 MB + a re-projection list) is the lever. The cold-ring fallback is what a snapshot without a
  drafter does.
- **Mid-call block-end snapshots keep the ring as the call left it** (not every position projected under the hook,
  ~10-13 % of prefill, derived). Cost if wrong: lower acceptance after such a restore with a short tail (counted by the
  R2 caveat lines), and the block ends at or before the call's end - 2048 store 41.9 MB of ring that buys nothing (~30
  of them for a 64k first prompt); r56.d5_opencode records each request's prefix plan beside its acceptance for the
  operator's ruling (project under the hook / such restores cold / keep).
- **"snapshot_test / prefix_gpu_* with the drafter"** became one dedicated `dflash_prefix_gpu_test` (PrefixSession over
  EngineAdapter: restore vs Continue, bitwise ring, ids and per-iteration acceptance; a block-end restore vs a cold run)
  plus `dflash_engine_test` SR. Cost if wrong: spec 7's C2 sequences a-g are not re-run with the drafter on.
- **Batching:** no batched runtime engine exists on main (13b was never built), so Task 2 is the interface
  (`BatchEngineIface::spec_k / step_spec / truncate_to / drafter_key`), the scheduler's speculative round with a
  per-request AdaptiveK, and the mock with a ring per slot. Cost if wrong: the contract may move when 13b implements it.
- **`--spec dflash --pp 2`** (in the operator's list for 19d, not in this plan): built as a step between Tasks 2 and 3 -
  the drafter on device 1, tap verify stage lists, the copy hand-off only (peer refused by name), the pipeline prefill's
  taps projected from the host once both devices finish a chunk; b70-decode stays one card. Cost if wrong: peer cannot
  serve DFlash; 2.54 GB of device 1 for the embedding replica; the host projection runs on device 1's decode queue
  beside its next prefill chunk (pp_dflash_test's hooked snapshots would show a problem).
- **D5 / the record:** the opencode recording does not exist yet, so D5 is prepared, not run: the A4 / golden rows
  (r56.d5_a4, r56.d5_dv), llama-benchy at row 41's flags (r56.d5_benchy) and the recording as an opt-in stage that SKIPs
  without it (r56.d5_opencode); BENCHMARKS "DFlash (spec 19)" holds them pending. **The default did not change** (the
  bar is unmeasured); the defaults within --spec dflash are P0's (int8, K <= 7, no draft vocabulary). Ornith's v1 drafter
  is not built (a full-attention drafter layer and a 16-row block: its stage). Cost if wrong: none until the box runs.
- **Added beyond the plan:** `pp_dflash_serve_test`, `pp_serve_plan_test` 7, `pp_dflash_test`, the `oracle_dflash_rank`
  data key, README / docs/19 rows for `--spec dflash`.

