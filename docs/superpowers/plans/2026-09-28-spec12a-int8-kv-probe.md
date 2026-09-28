# Spec 12a - int8 KV cache: the accuracy probe (CPU)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure on the CPU which int8 KV scheme keeps attention and logits closest to bf16 KV, at depth, and choose one (or stop).

**Architecture:** a Python probe in `tools/oracle/` that runs the model teacher-forced on long prompts in the oracle container, captures per-FA-layer q, K, V, and replays attention with K and V quantised by each scheme; then an end-to-end run with the schemes simulated inside every FA layer's attention (a patched attention in the CPU reference) for logits.

**Tech Stack:** Python 3 + torch (CPU, the oracle container).

**Spec:** `docs/specs/2026-09-28-spec12-int8-kv-cache-design.md` (§2 the three schemes, §4 Q1, §6 12a).

## Global Constraints

- Branch `spec12a-int8-kv-probe` from main; box tree `~/b70-inference-server-spec12a`. Copy `tools/box.env` if missing; never commit it.
- CPU oracle runs via `tools/oracle/run_in_container.sh` (mounts its own tree); `free -g` >= 70 GB available first, one at a time, detached, polled with a background until-loop. Size the CPU cost before launching (plan 9a's 21-scenario dump took 4.3 h; plan 7a's chunked oracle reached 8k ids in ~3 h): prefer 2k-8k-id captures and extrapolate depth with the attention-only replay.
- No GPU needed. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Where the rounding happens:** K is quantised after RoPE and the k-norm (what the cache stores), V as stored; q is never quantised.
2. **The rotated scheme** rotates K per kv head with a Hadamard of size 256 (random signs) and applies the same rotation to q: check q·k is unchanged in fp64 before measuring int8.
3. **KIVI-style K** (per channel over groups of 64 positions) keeps the last partial group in bf16 until it fills: simulate that, not a padded group.
4. **Depth matters:** measure attention error at positions 2k, 8k and, from the attention-only replay on real K/V tiled to depth, 32k; softmax at depth amplifies small K errors.
5. Compare with a **bf16-output control**, as plan 9a did: an int8 scheme is judged against how far bf16 itself is from fp32 on the same path.

---

### Task 1: capture and attention-only replay

**Files:** `tools/oracle/kv_int8_probe.py`, `tools/oracle/test_kv_int8_probe.py`.

- [ ] **Step 1:** tests on synthetic K/V: per-token int8 round-trip bound; the Hadamard rotation preserves q·k to 1e-6; KIVI grouping with a partial last group.
- [ ] **Step 2:** capture q, K, V for all 16 FA layers on `tests/golden/prompts/long32k.ids[:8192]` and on two A4 tool-call prompts; replay attention per scheme: per-(head, position) output cosine and max abs against bf16 KV, at positions up to 8k and (tiled) 32k.
- [ ] **Step 3: Commit** `oracle: int8 KV schemes, attention-only replay (spec 12 P0)`.

### Task 2: end to end

- [ ] **Step 1:** patch the CPU reference's FA attention to quantise K/V by scheme; last-row and per-position logits cosine, argmax, and unfiltered KL against bf16 KV on the golden prompts + 32 greedy tokens and on the two tool-call prompts; the bf16-output control.
- [ ] **Step 2:** `docs/probe-int8-kv-2026-09-28.md`: the tables, the chosen scheme (smallest error; ties to the simplest), or a stop (if every scheme's logits cosine at 8k is below the l0-int8 path's own distance to the oracle, record for the operator); proposed tolerances for spec 12's Q3.
- [ ] **Step 3:** `docs/README.md` line; **Commit** `probe: int8 KV verdict (spec 12 P0)`.

**Gate for the plan:** both tasks' tables, the scheme choice or a stop. Hand back with the tables.
