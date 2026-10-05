# Spec 16d - pipeline parallel: prefix cache, MTP, 262k, the server, the record

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** everything else works under `--pp 2`: spec 7's snapshots per device, spec 8's MTP across the split, Qwen3.8 at max_len 262144 (P3, S3), `b70-serve --pp 2`, the record.

**Architecture:** spec 16 §3.1 (MTP head and its embedding on device 1), §3.4. Snapshot entries become `{device 0 part, device 1 part}`; MTP drafts on device 1, verify across both devices with M = K + 1 rows handed off together.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest, uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec16-pipeline-parallel-design.md` (§3.1, §3.4, §4 P2, P3, §5 S3, §6 16d). Needs 16b and 16c merged.

## Global Constraints

- Both cards; branch `spec16d-pp-integration` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--pp 1` unchanged. Lock once for both cards; detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The embedding for MTP drafts on device 1** (spec 16 decision): implement the operator's choice (replicate, 2.54 GB, or gather rows over P2P); the memory report shows it.
2. **Spec 7 under PP:** `snapshot_test` and `prefix_gpu_*` with `--pp 2`; a host-store entry saved under `--pp 2` refused when restored under `--pp 1` (and vice versa), with a clear message.
3. **Spec 8 M2 under PP:** verify rows bitwise equal to M = 1 decode across the split; commit moves `pos` on both devices.
4. **262144:** the RoPE table, attention variants (spec 10's v2 needs no max_len binary; v1 does) and the memory per device at max_len 262144; passkey at 5 / 50 / 95 % of ~250k.
5. **The server:** `--pp 2` with the prefix cache and `--mtp` both on; one request end to end through `b70-serve`.

---

### Task 1: snapshots and MTP across the split

- [ ] Review Focus 1-3; tests extended with `--pp 2` registrations. **Commit** `runtime: prefix-cache snapshots and MTP under pipeline parallel (spec 16)`.

### Task 2: 262k and the server

- [ ] Review Focus 4-5; P3 passkey; S3 memory rows; `b70-serve --pp 2`; llama-benchy rows (the operator's flags) one card vs `--pp 2` at pp4096 and at depth 32k / 128k. **Commit** `server: --pp 2, and Qwen3.8 at 262k (spec 16, P3, S3)`.

### Task 3: the record

- [ ] `docs/BENCHMARKS.md` "Pipeline parallel (spec 16)" complete; spec 16 amendment §8 with every gate and number; README (scope: two cards, PP); `docs/10-the-box.md` (both cards in use, the lock). **Commit** `spec 16: pipeline parallel - the record`.

**Gate for the plan:** P2 and P3 green with `--pp 2`, S3 recorded, the full suite green with `--pp 1`, the record committed.
