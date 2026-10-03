# Spec 9a - int8 `lm_head`: the accuracy probe (P0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure, on the CPU, whether an int8 per-row `lm_head` made from the bf16 checkpoint tensor passes spec 9's L1 and L2 against the bf16 head, before any kernel work.

**Architecture:** a Python probe in `tools/oracle/` that loads `lm_head.weight` (bf16) and final hidden states (after the final norm) dumped from the CPU oracle, quantises the head to int8 with one fp32 scale per row (symmetric, round to nearest, scale = max|row| / 127), and compares logits and sampling distributions.

**Tech Stack:** Python 3 + torch (CPU, the oracle container).

**Spec:** `docs/superpowers/specs/2026-09-28-spec9-lm-head-quantised-design.md` (§2 ruling: int8 only; §4 L1, L2; §5 P0; §6 9a stopping rule).

## Global Constraints

- Branch `spec9a-lm-head-probe` from main; `tools/box.sh` syncs to `~/b70-inference-server-spec9a` (`tools/box.sh dir`). If `tools/box.env` is missing, copy it from `/Users/urakozz/CLionProjects/b70-inference-server/tools/box.env`; never commit it or its contents.
- CPU oracle runs in `tools/oracle/run_in_container.sh`; check `free -g` (>= 70 GB available) first, one at a time, detached (`tools/probe/detach.sh`), poll for `ALLDONE rc=` with a background until-loop.
- This plan needs no GPU. If a step does touch the GPU, it runs under `flock ~/b70-gpu.lock`.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Which hidden state:** the input to `lm_head` is the hidden **after** the final norm (`model.norm`); dumping `resid` (before the norm) would make every comparison meaningless. Check: bf16 head on the dumped hidden reproduces the oracle's own logits to cosine >= 0.99999.
2. **Rows beyond the tokenizer vocabulary** (248077 used of 248320): exclude ids >= `vocab_used` from argmax, top-20 and KL, as the engine does.
3. **Scale per row, fp32**, and the int8 dot product accumulated in fp32 against the bf16 hidden (W8A16): do not quantise the hidden.
4. The sampling filter must match the engine's `sample()` (`src/cli/serve_adapters.h`): temperature, top-k, top-p in that order, over `vocab_used`.
5. Positions: every generated position of the golden prompts (32 each) and of the A4 tool-call outputs, not only prompt positions.

---

### Task 1: hidden states and the probe

**Files:** Create `tools/oracle/lm_head_probe.py`, `tools/oracle/test_lm_head_probe.py`. Modify `tools/oracle/dump.py` only to add `--final-hidden` (the post-norm hidden at every position, bf16 `[T, 5120]`) if no existing tensor holds it.

- [ ] **Step 1:** `test_lm_head_probe.py`: on a synthetic `[1000, 64]` head and random hiddens, int8 round trip error bound (|w - w_q| <= scale / 2 per element), argmax equality for well-separated logits, KL of identical distributions == 0.
- [ ] **Step 2:** implement; dump final hiddens for the three golden prompts (prompt + 32 greedy tokens) and for the A4 tool-call set's prompts + bf16 oracle outputs (`tools/toolcall/`); Review Focus 1 check.
- [ ] **Step 3:** per position: logits cosine, argmax equal (and whether a mismatch is a bf16 near-tie under the golden rule's tolerance), top-20 set equal, KL(p_bf16 || p_int8) after temperature 1.0 / top-k 20 / top-p 0.95 (`generation_config.json`: read the values from it) and at temperature 0.6.
- [ ] **Step 4: Commit** `oracle: int8 lm_head accuracy probe (spec 9 P0)`.

### Task 2: the verdict

- [ ] **Step 1:** `docs/probe-lm-head-2026-09-28.md`: the table per source (golden, toolcall): positions, cosine min / mean, argmax mismatches (near-tie or not), top-20 equality %, KL mean / p99 at both temperatures; the host quantisation time for the full head (time the quantisation of the real `[248320, 5120]` tensor).
- [ ] **Step 2:** verdict against L1 (cosine >= 0.9999, argmax equal except near-ties, top-20 equal in >= 99 %) and L2 (KL mean <= 1e-3, p99 <= 1e-2). If it fails, stop: record, and say by how much.
- [ ] **Step 3:** `docs/README.md` line; **Commit** `probe: int8 lm_head verdict (spec 9 P0)`.

**Gate for the plan:** the doc committed with every number and the verdict. Hand back with the table.
