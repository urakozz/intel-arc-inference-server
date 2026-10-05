# Spec 17c - tensor parallel: prefill

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** prefill under `--tp 2` by the option 17a chose: either TP prefill with the per-fold collective overlapped (and bf16 partials if 17a showed the accuracy cost acceptable), or the PP-hybrid (prefill as spec 16's chunk pipeline, decode as TP); T2 on prefill; D2.

**Architecture:** spec 17 §2 decision 4. **TP option:** the prefill step at per-card shapes, the row-parallel GEMMs' partials exchanged per fold on a copy engine overlapping the next column-parallel projection, folded in fixed order. **PP-hybrid option:** each card also holds the weights of its PP half in the full (unsliced) layout, or the layers are reshuffled after prefill; 17a's memory numbers decide which.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§2 decision 4, §4 T2, §5 D2). Needs 17b merged; for the PP-hybrid, spec 16c merged.

## Global Constraints

- Both cards; branch `spec17c-tp-prefill` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- `--tp 1` unchanged. Lock once for both cards; detached, polled. Interleaved pairs, median of 3.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **After prefill both cards hold exactly their decode-layout state** (their heads' KV and GDN state), whichever prefill option ran; T1's bitwise equality of the residual holds for the first decoded token.
2. **T2 on prefill:** the golden gates' prefill path with `--tp 2` under the tie rule; `prefill_split_*` with `--tp 2`.
3. **The overlap is real** (TP option): per-fold transfer time hidden behind compute, measured per layer.
4. **Memory** (PP-hybrid option): the extra full-layout weights per card fit beside KV at the target max_len; the load report shows it.
5. **Spec 7's block hook** under `--tp 2`: snapshots per card, taken when both cards hold the block.

---

### Task 1: the chosen prefill path

- [ ] Implement 17a's option; `tests/prefill/tp_prefill_test.cc` (Review Focus 1, 2, 5). **Commit** `prefill: tensor-parallel prefill (<option>) (spec 17)`.

### Task 2: speed

- [ ] pp4096, pp32768, pp65536 with `--tp 2` vs one card and vs `--pp 2` (spec 16c); D2 per 17a's proposed bar; Review Focus 3 or 4. BENCHMARKS rows. **Commit** `spec 17c: TP prefill speed`.

**Gate for the plan:** T2 on prefill, the split-prefill tests, D2 measured, `--tp 1` unchanged.
