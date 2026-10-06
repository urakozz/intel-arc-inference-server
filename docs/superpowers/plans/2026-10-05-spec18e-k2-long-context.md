# Spec 18e (optional) - K2-Horizon long context: int8 KV and two cards

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** extend K2's context beyond one card's bf16 limit: spec 12's int8 KV (rotkv) on one card (~64k, derived), and spec 16's pipeline parallel on two cards (~64k bf16 / ~128k int8, derived); passkey at each new ceiling; a like-for-like row against the vLLM PP = 2 baseline.

**Architecture:** spec 18 §2 (spec 12 and spec 16 rows), decision 2. Int8 KV: rotkv per spec 12 §8 at head_dim 128 (the 128-point Hadamard per kv head replaces the 256-point one; q rotated the same way; the attention output un-rotated before the softplus gate). PP: spec 16's placement with the split chosen on bytes (the `lm_head` card is heavier; vLLM needed 25 / 23 for K2).

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest, uvx llama-benchy.

**Spec:** `docs/superpowers/specs/2026-10-05-spec18-k2-horizon-design.md` (§2, §4 decision 2, §8 18e); `2026-09-28-spec12-int8-kv-cache-design.md` §8; `2026-10-05-spec16-pipeline-parallel-design.md`. Needs 18d, spec 12b and (for the two-card part) spec 16d merged.

## Global Constraints

- Branch `spec18e-k2-long-context` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- K0 holds; defaults unchanged (`--kv-cache bf16`, `--pp 1`) unless every gate passes. Lock once for both cards in PP runs; detached, polled.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **rotkv at head_dim 128** re-checked on K2 with 18a's reference (spec 12a's method: the golden decision rows, the attention-only replay at depth) before it ships for K2; the tolerances re-derived for K2.
2. **MoVA writes the KV:** quantisation happens on the routed mix, after the value experts' combine, in both prefill and decode.
3. **PP split on bytes:** weights + KV per card at the target context equalised; the memory report per card.
4. **Passkey** at 5 / 50 / 95 % of each new ceiling.
5. **The vLLM comparison** at matched settings where possible (two cards, the same context, fp8 vs our int8 KV stated).

---

### Task 1: int8 KV for K2

- [ ] Review Focus 1, 2; the K2 golden gates and A4 with int8 KV; passkey at ~60k on one card. **Commit** `kv: rotkv int8 KV on K2-Horizon (spec 18e)`.

**Status 2026-10-06: built blind** (branch `spec18e-k2-kv8`; spec 18 §13). Code, host tests, the
kernels' Mac checks, the `_kv8` gate twins and K1 registered, Review Focus 2 built in (MoVA's mix
quantised after the combine); the planner gives 83968 / 90368 positions with prefill (bf16 / int8
head), not ~64k. Open: Review Focus 1 on real weights (`tools/oracle/kv8_k2_repeat.sh` after the
checkpoint download), every card gate (box-validation-queue row 21), passkey near the new ceiling
(`tools/probe/k2_passkey.sh`), A4 (needs 18d's K2 A4 tooling).

### Task 2: two cards

- [ ] Review Focus 3, 4; spec 16's P1 (bitwise against one card at a context both fit) on K2; passkey at the two-card ceiling. **Commit** `pp: K2-Horizon across two cards (spec 18e)`.

### Task 3: the record

- [ ] Review Focus 5; BENCHMARKS rows; spec 18 amendment. **Commit** `spec 18e: K2 long context - the record`.

**Gate for the plan:** the K2 gates green with int8 KV and with PP, passkey at each ceiling, the comparison rows recorded.
