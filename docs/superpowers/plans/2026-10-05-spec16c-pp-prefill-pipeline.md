# Spec 16c - pipeline parallel: the prefill chunk pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** prefill overlaps across the two cards: chunk `c` runs layers `[s, L)` on device 1 while chunk `c + 1` runs `[0, s)` on device 0, with double-buffered hand-off buffers; P1 (bitwise) on prefill; S2 (>= 1.7x on pp32768 / pp65536).

**Architecture:** spec 16 §3.3. Two prefill contexts (one per device, each with its own immediate list and scratch); hand-off buffers `[2][C][5120]` bf16 on device 1; device 0 never waits for device 1 except when both buffers are full. The last chunk's last row goes through the head on device 1. Spec 7's per-block hook fires on device 1 after a block's last layer.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec16-pipeline-parallel-design.md` (§3.3, §4 P1, P2, §5 S2). Needs 16b merged.

## Global Constraints

- Both cards; branch `spec16c-pp-prefill` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--pp 1` unchanged. Lock once for both cards; detached, polled. Interleaved pairs against one card, median of 3.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **P1 bitwise on prefill:** the per-chunk math is the same as on one card (same chunk boundaries, same kernels), so KV, GDN state and the first token equal the single-card prefill byte for byte, for 2-chunk, 16-chunk and non-multiple-of-2048 prompts.
2. **Back-pressure:** device 0 waits when both hand-off buffers are in use; no buffer is overwritten before device 1 has consumed it (a test that slows device 1 artificially).
3. **Continuation and spec 7:** `prefill_split_*` with `--pp 2`; the block hook's snapshot is taken only when both devices hold the block's state.
4. **Both backends** (`l0`, `l0-int8`) and flash attention per device; the composed path refused under `--pp 2` with a clear message.
5. **The second card is ~3 % slower on prefill:** report each device's busy time per chunk, to confirm the split from 16a balances them.

---

### Task 1: two prefill contexts and the pipeline

- [ ] Per-device prefill contexts; the double-buffered hand-off; the chunk scheduler; `tests/prefill/pp_prefill_test.cc` (Review Focus 1-4). **Commit** `prefill: the two-card chunk pipeline (spec 16, P1 prefill)`.

### Task 2: speed

- [ ] pp4096, pp32768, pp65536 (and pp at 128k with max_len 262144 if 16d's memory work is in), `--pp 2` against one card (S2); per-device busy time (Review Focus 5). BENCHMARKS rows. **Commit** `spec 16c: PP prefill speed`.

**Gate for the plan:** P1 bitwise on prefill, the back-pressure test, split-prefill tests, S2 measured, `--pp 1` unchanged.
