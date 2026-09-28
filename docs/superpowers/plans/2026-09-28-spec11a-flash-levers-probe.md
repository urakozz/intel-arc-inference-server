# Spec 11a - flash prefill levers 3-5: the probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** measure the operand-bandwidth ceiling (P0), then levers 3 (cooperative prefetch + per-tile barrier) and 5 (32-key K loads), and lever 4 (split-d) only if P0 says so, as arms of the flash probe harness; pick the winner.

**Architecture:** arms of `tools/probe/probe_flash_attn.{cl,cc}` (the spec 6a harness; production kernel `src/kernels/prefill/pf_flash_attn.cl` at KT64 RPW8 HPW6 QREG0 EXP2=1 is the control) plus two small unit probes in `tools/probe/`: an L1 operand-bandwidth DPAS loop and a 32-row transposed K load layout pin.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero.

**Spec:** `docs/specs/2026-09-28-spec11-flash-prefill-levers-design.md`. Evidence: `docs/research-flash-prefill-2026-09-27.md` (levers table, P0-P4), `tools/probe/pfa_levers_research.cl` (the research arms, including the failed 32-row K load).

## Global Constraints

- Branch `spec11a-flash-levers-probe` from main; tools/box.sh syncs to `~/b70-inference-server-spec11a` automatically (`tools/box.sh dir`). Copy `tools/box.env` from the main checkout if missing; never commit it.
- Every GPU command under `flock ~/b70-gpu.lock`, detached (`tools/probe/detach.sh`), background until-loop polling for `ALLDONE rc=`. Record `uptime` with every table; arms compared as interleaved pairs within one locked job, median of 3.
- Probe code only: `src/` unchanged. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The control is today's production kernel** (EXP2=1, RPW 8), not the 6a winner: rebuild the harness control from `pf_flash_attn.cl` itself.
2. **The barrier (lever 3) needs every sub-group of the work-group at the same trip count**: true at RPW 8 for full tiles; check the diagonal tile and the tail chunk (C % 8 != 0) do not deadlock (run the 9-row and C = 1 cases).
3. **The 32-row K load layout** is pinned by a unit probe that writes known values and reads back the register layout, before any arm uses it (the research's guess gave cosine 0.15-0.24).
4. **P0's ceiling must be DPAS fed from L1**, not from registers: loads inside the loop, same message shapes the flash kernel uses, reuse 1 / 2 / 4.
5. Depths 0, 2048, 30720 and 65536, chunk 2048; K1 per arm (cosine >= 0.99999 vs fp64, the five spec 6 cases plus 6c's single-row, 9-row and tail cases).

---

### Task 1: P0, the operand-bandwidth ceiling

**Files:** `tools/probe/probe_dpas_l1.{cl,cc}`, registration in `tools/probe/CMakeLists.txt`.

- [ ] **Step 1:** a kernel that loops DPAS (bf16, the flash kernel's 8x16x16 shape) with operands loaded each iteration by 2D block reads from a small (L1-resident) buffer, at operand reuse 1, 2, 4; TFLOP/s and the implied load GB/s per Xe-core; plus the register-fed peak as reference (183.45 TFLOP/s, docs/probe-dpas-rates).
- [ ] **Step 2:** compare with the production kernel's rate at 30720 (TFLOP/s from its FLOP count). Verdict for lever 4: build it only if the reuse-1 ceiling is within 1.3x of the kernel's current rate.
- [ ] **Step 3:** `docs/probe-flash-levers-2026-09-28.md` with the table; **Commit** `probe: DPAS operand-bandwidth ceiling from L1 (spec 11 P0)`.

### Task 2: lever 5's load layout

- [ ] **Step 1:** `tools/probe/probe_k32_layout.{cl,cc}`: the 32-row transposed 32-bit block read of a bf16 K tile; write a tile where each element encodes (row, col), read the sub-group registers back, print the mapping; derive the DPAS operand arrangement it needs.
- [ ] **Step 2:** record the mapping in the doc; **Commit** `probe: the 32-row transposed K load layout (spec 11 lever 5)`.

### Task 3: the arms

**Files:** `tools/probe/probe_flash_attn.cl` (defines `PF` prefetch 0/1 with `BAR` 0/1, `K32` 0/1, `SPLITD` 0/1 only if Task 1 says build), `tools/probe/probe_flash_attn.cc` (arm list), `tools/probe/CMakeLists.txt`.

- [ ] **Step 1:** control arm from production text; reproduce production's time within 3 %.
- [ ] **Step 2:** lever 3 alone; lever 5 alone; 3 + 5; lever 4 (+3, +5) only if built; K1 per arm; interleaved against the control at depths 0 / 2048 / 30720 / 65536.
- [ ] **Step 3:** table in the doc; the derived pp4096 attention-phase ms (scale the engine's measured phase by the paired ratio, as 10a did) against F1' (<= 70 ms) and the 64k share of peak against F2' (>= 25 %).
- [ ] **Step 4: Commit** `probe: flash prefill levers 3-5 (spec 11)`.

### Task 4: verdict

- [ ] The winner (defines), the derived F1'/F2' numbers, and a stop if no arm beats the control by >= 5 % at 30720 (record for the operator). `docs/README.md` line. **Commit** `probe: flash levers verdict (spec 11)`.

**Gate for the plan:** P0, the layout pin, every arm K1-checked and timed, the verdict. Hand back with the tables.
