# Spec 17a - tensor parallel: the probe beyond spec 16's arm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** with spec 16a's remote-partial fold measured, measure the rest of what TP decode and prefill cost on two B70s: per-layer step time at TP shapes, the `lm_head` vocab split with the argmax exchange, the prefill collective with and without overlap; decide fold vs separate kernel and the prefill option.

**Architecture:** probe binaries in `tools/probe/`, building on 16a's `probe_p2p` code; the engine's kernels compiled at TP per-card shapes (12 q-heads, 2 kv-heads, 24 GDN v-heads, gate‖up 8704 x 2, down K 8704) through `ModelDesc` (`tp_size` 2).

**Tech Stack:** C++17, OpenCL C, Level Zero.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§2 decisions 1-4, §5 P0, §6 17a). Needs spec 16a's doc (`docs/probe-multi-gpu-2026-10-05.md`, the remote-partial verdict) and spec 16b merged (multi-device engine scaffolding).

## Global Constraints

- Both cards; branch `spec17a-tp-probe` from main; box tree automatic; `tools/box.env` copied if missing, never committed.
- Lock once for both cards; detached, polled; interleaved arms, median of 5; `uptime` recorded; both cards idle for timing rows.
- Probe code only. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **A full layer at TP shapes on each card** (not kernels in isolation): the captured per-card list for one GDN and one FA layer with the two folds, so launch gaps and the folds' waits are inside the number.
2. **Both cards' residuals bitwise identical** after each fold in every arm (T1's property).
3. **The argmax exchange** for the split `lm_head`: ties broken by the lower id on both cards; the chosen id identical on both.
4. **The prefill collective** at 42 MB (fp32) and 21 MB (bf16) per fold, and whether it overlaps with the next column-parallel projection on the same card (two queues / copy engine).
5. **Derived decode estimate** from the per-layer numbers, against vLLM's 54.90 t/s TP = 2 on the same cards (spec 17 prior art) as a sanity check, not a bar.

---

### Task 1: one layer at TP shapes

- [ ] Per-card lists for one GDN layer and one FA layer with folds (16a's winner); per-layer time vs one card's layer; Review Focus 1, 2. `docs/probe-tp-2026-10-05.md`. **Commit** `probe: a layer under tensor parallel (spec 17 P0)`.

### Task 2: head and prefill

- [ ] The vocab-split `lm_head` + argmax exchange (Review Focus 3); the prefill collective at both precisions, alone and overlapped (Review Focus 4); the PP-hybrid prefill estimate from spec 16c's numbers if available. **Commit** `probe: TP head and prefill collectives (spec 17 P0)`.

### Task 3: verdicts

- [ ] Fold vs separate kernel; the derived D1 estimate (Review Focus 5); the prefill choice (TP with overlap, bf16 partials, or PP-hybrid) with numbers; proposed D2 bar; `docs/README.md` line. **Commit** `probe: TP verdicts (spec 17)`.

**Gate for the plan:** the per-layer, head and prefill numbers recorded with both cards idle; the three verdicts written. Hand back with the tables.
