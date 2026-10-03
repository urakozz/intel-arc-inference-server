# Spec 13a - batching: the probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure a decode step whose rows are **different sequences** (not consecutive positions) at B = 1..4, find which kernels need per-row state, and propose the slot layout and B for the operator's decision.

**Architecture:** a probe binary that builds a decode list at M = B rows (8a's `capture.cc` M parameter) with a per-row descriptor path added only in probe builds: per-row `pos`, KV base and GDN state slot. Kernels are compiled in a probe mode (`ROWS_SEQS=1`) beside production.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero.

**Spec:** `docs/superpowers/specs/2026-09-28-spec13-batching-design.md` (§1 derived gains, §2 the B / context decision, §3 kernel changes, §6 13a).

## Global Constraints

- Branch `spec13a-batching-probe` from main; box tree `~/b70-inference-server-spec13a`. Copy `tools/box.env` if missing; never commit it.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved arms, median of 3, `uptime` recorded.
- Probe builds only (`B70_BATCH_PROBE=ON`, default OFF): production binaries byte-identical. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Which kernels assume one sequence:** `gdn_step` (state in registers across the M loop), `attn_prep` / decode attention (one `pos`, one KV base), the conv ring (slot `(pos+m)%16`), `embed_gather` / argmax (row-agnostic?). List each with file:line and what it needs.
2. **Correctness of the probe rows:** each row of a B-row step equals the same sequence stepped alone, bitwise where the kernel is row-local.
3. **Attention does not share across rows:** measure its time per row at mixed depths (e.g. 4k + 32k in one step), since v2's stride is per row.
4. **Memory** of each layout option (B x max_len, mixed long + short slots) from the real buffer sizes, not only the spec's derivation.
5. The step-time table must use the real production kernels at M = B (int4 GEMVs, int8 lm_head, v2 attention), as 8b's per-M table did.

---

### Task 1: the audit

- [ ] Per decode-list kernel: row-agnostic or per-sequence; the change each needs; `docs/probe-batching-2026-09-28.md` section 1. **Commit** `probe: batching audit of the decode list (spec 13)`.

### Task 2: the batched step

**Files:** `tools/probe/probe_batch_step.cc`, probe-mode defines in `src/kernels/gdn_step.cl` and `src/kernels/attn*.cl` (compiled only with `B70_BATCH_PROBE`), `tools/probe/CMakeLists.txt`.

- [ ] **Step 1:** B sequences prefilled separately into B state slots (probe allocation), one B-row step; Review Focus 2 check against B single-sequence steps.
- [ ] **Step 2:** step time at B = 1..4, same depth (4k) and mixed depths (4k + 32k); aggregate tokens/s; compare with the derived 1.71x / 1.97x / 2.30x.
- [ ] **Step 3:** the doc's tables; **Commit** `probe: batched decode step at B = 1..4 (spec 13)`.

### Task 3: the proposal

- [ ] In the doc: the recommended B and slot layout (with memory numbers at bf16 and int8 KV), the prefill interleave chunk size (measured: prefill chunk time at 256 / 512 / 1024 / 2048 rows at depth), and the per-row descriptor design for 13b. `docs/README.md` line. **Commit** `probe: batching proposal (spec 13)`.

**Gate for the plan:** the audit, the measured step table, the proposal for the operator's decision (B, layout, chunk). Hand back with the tables.
