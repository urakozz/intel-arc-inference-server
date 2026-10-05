# Spec 17d - tensor parallel: prefix cache, MTP, sampling, the server, the record

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** everything else works under `--tp 2`: spec 7's snapshots per card, spec 8's MTP head under TP, sampling with the split vocabulary (spec 17 decision 3), `b70-serve --tp 2`; T3; the record.

**Architecture:** spec 17 §3. Snapshot entries `{card 0 part, card 1 part}`; the MTP head sliced like a main layer; host sampling over the merged per-card top-k (or both vocab halves, per the operator's choice).

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest, uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§2 decision 3, §3, §4 T3, §6 17d). Needs 17b and 17c merged.

## Global Constraints

- Both cards; branch `spec17d-tp-integration` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--tp 1` unchanged. Lock once for both cards; detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Sampling is exact:** the merged per-card top-k after the shared filter equals filtering the full logits (a host test on synthetic split logits, every filter combination); seeded runs reproducible.
2. **Spec 8 under TP:** M2 (verify rows bitwise equal to M = 1 under `--tp 2`), M3 (greedy MTP output identical to `--mtp 0` under `--tp 2`), acceptance unchanged within noise.
3. **Spec 7 under TP:** `snapshot_test`, `prefix_gpu_*` with `--tp 2`; entries saved under one TP / PP layout refused under another.
4. **The server** with `--tp 2`, the prefix cache and `--mtp` on; llama-benchy rows with the operator's flags against one card and against `--pp 2`, and vLLM TP = 2 on the same cards for comparison (the prior art's configuration).
5. **The README's headline** only changes with a RECORD-grade row (idle box, both cards).

---

### Task 1: sampling and MTP under TP

- [ ] Review Focus 1, 2. **Commit** `server: sampling and MTP under tensor parallel (spec 17)`.

### Task 2: snapshots and the server

- [ ] Review Focus 3, 4; T3 (determinism and replay bitwise). **Commit** `server: --tp 2 with the prefix cache (spec 17, T3)`.

### Task 3: the record

- [ ] `docs/BENCHMARKS.md` "Tensor parallel (spec 17)" complete; spec 17 amendment §8; README (decode headline with `--tp 2` beside one card, if RECORD grade); `docs/10-the-box.md`. **Commit** `spec 17: tensor parallel - the record`.

**Gate for the plan:** T3 green, MTP M2 / M3 under TP, the comparison rows recorded, the full suite green at `--tp 1`, the record committed.
