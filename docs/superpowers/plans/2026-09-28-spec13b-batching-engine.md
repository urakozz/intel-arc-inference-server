# Spec 13b - batching in the engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** per-sequence state slots and a batched decode list over B sequences, with the operator's chosen B and slot layout (from 13a); gate B1.

**Architecture:** slots own `gdn_state`, `conv_ring`, KV region and a `Control` row (pos, pending token); a row descriptor table in device memory maps each batch row to its slot; the batched kernels (13a's probe mode, promoted) read it; one captured list per active batch size (1..B), selected per step; the single-sequence list is unchanged for B = 1.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, ctest.

**Spec:** `docs/specs/2026-09-28-spec13-batching-design.md` (§3, §4 B1). Needs `docs/probe-batching-2026-09-28.md` and the operator's decision on B, layout and chunk size.

## Global Constraints

- Branch `spec13b-batching-engine` from main; box tree `~/b70-inference-server-spec13b`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. `uptime` with timings.
- With batching off (`--batch 1`, the default), the engine is bitwise today's. MTP and batching are exclusive in this version (refused with a clear error). No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **A slot's prefill while other slots decode:** prefill writes only its slot's state and KV; a decode step concurrently in flight is impossible (one queue), so the scheduler interleaves; test that a prefill between two batched steps leaves the other slots' state byte-identical.
2. **Slot reuse:** a finished sequence's slot reset (`reset()` per slot) before a new request uses it.
3. **Spec 7 snapshots per slot:** save/restore target one slot; restoring slot 2 leaves slot 1 untouched.
4. **Rows at different depths** (v2's per-row stride) and a row at its slot's max_len (refused, not overrun).
5. **Replay determinism** with a fixed arrival order.

---

### Task 1: slots

- [ ] `src/runtime/buffers.{h,cc}` slot allocation per the chosen layout; `Engine` slot API (`reset(slot)`, `prefill(slot, ids)`, `pos(slot)`, snapshot calls with a slot); tests: `buffers_test`, a slot-isolation test (Review Focus 1-3). **Commit** `runtime: per-sequence state slots (spec 13)`.

### Task 2: batched kernels and lists

- [ ] Promote 13a's probe-mode kernels (per-row descriptors) into production variants; capture one list per batch size; `Engine::step_batch(rows)` returns one id per active row; B1 test: each sequence in a batch bitwise (or near-tie, per the rule) equal to it alone, at B = 2..max and mixed depths; replay determinism. **Commit** `runtime: the batched decode step (spec 13, B1)`.

### Task 3: speed and suite

- [ ] Aggregate t/s at B = 2 / max, 4k each (bars >= 1.6x / 2.0x); B = 1 within 2 % of today; full suite. Section "Engine" in the probe doc. **Commit** `batching: engine gates and speed (spec 13)`.

**Gate for the plan:** B1, slot isolation, determinism, speed measured, full suite green.
