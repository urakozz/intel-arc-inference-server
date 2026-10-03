# Spec 11b - flash prefill levers in production

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** promote plan 11a's winning arm into `pf_flash_attn`, pass the production gates, record pp4096 / pp65536 / pp130816.

**Architecture:** the winner's defines added to `src/kernels/prefill/pf_flash_attn.cl` (today's form stays buildable behind the defines), the CMake variant in `src/kernels/prefill/CMakeLists.txt`, launch geometry in `src/runtime/prefill/attn.cc` only if the winner changes it.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, ctest.

**Spec:** `docs/superpowers/specs/2026-09-28-spec11-flash-prefill-levers-design.md` (§4 gates, §5 F1', F2'). Needs `docs/probe-flash-levers-2026-09-28.md` (11a's verdict).

## Global Constraints

- Branch `spec11b-flash-levers` from main; box tree `~/b70-inference-server-spec11b` (automatic). Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs into the box tree.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved pairs against main's build, median of 3, `uptime` recorded.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. The tail chunk (C % 8 != 0) and single-row prefill with the barrier (no deadlock, pad rows finite).
2. `prefill_split_*` (continuation at non-64 offsets) and `prefill_replay_test` bitwise.
3. K3a (`flash_vs_oracle_test`) at the ruled tolerances (spec 6 §9: 1e-5 on `l0`, 1e-6 on `l0-int8`).
4. The launch count (`prefill_smoke_test`) unchanged unless the winner adds a launch.
5. MTP's head KV fill in prefill (spec 8 §3.2) uses the same flash kernel: `mtp_prefill_test` passes.

---

### Task 1: the kernel

- [ ] **Step 1:** port the winner's text into `pf_flash_attn.cl` behind its defines; set them in CMake; `pf_flash_attn_test` (K1, all cases) passes.
- [ ] **Step 2:** golden gates, `flash_vs_oracle_test`, `flash_long_test`, `prefill_split_*`, `prefill_replay_test`, `prefill_smoke_test`, `mtp_prefill_test`; then the full suite.
- [ ] **Step 3: Commit** `kernels: pf_flash_attn with <levers> (spec 11)`.

### Task 2: speed and record

- [ ] **Step 1:** interleaved pairs, main vs branch: pp4096 (t/s and `attn_flash` ms from the prefill profile), pp65536 and pp130816 at max_len 131072, decode at 4k (no regression); passkey 3/3 at 120k (`tools/probe/k3b_passkey.sh`).
- [ ] **Step 2:** `docs/BENCHMARKS.md` section "Flash prefill levers (spec 11)"; spec 11 amendment §8; README prefill line only if a RECORD-grade pp4096 row (idle box) beats the current one.
- [ ] **Step 3: Commit** `spec 11: flash levers in production (the record)`.

**Gate for the plan:** every gate green, F1'/F2' measured (a miss recorded, not tuned around). Hand back with the pair table.
