# Spec 6c - flash attention: exp2 and 8 rows per work-group, in production

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** put the research's two measured levers into `pf_flash_attn` (1.42x in the probe harness at every depth, fp64 cosine unchanged): `exp` → `exp2` with the scale folded, and RPW 16 → 8 (6 sub-groups per work-group); prove the gates and record the in-engine speed.

**Architecture:** kernel text change in `src/kernels/prefill/pf_flash_attn.cl`, the build define in `src/kernels/prefill/CMakeLists.txt`, the grid in `src/runtime/prefill/attn.cc` (`(C + 15) / 16` → `(C + 7) / 8`). Nothing else moves.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, ctest.

**Spec:** `docs/specs/2026-09-25-spec6-flash-attention-128k-design.md` (§3.1, §4 K1-K4, §5 F1/F2, §8). Evidence: `docs/research-flash-prefill-2026-09-27.md` (§0, the ranked table rows 1-2, P1). **Operator ruling (2026-09-27): `exp2` approved** in place of spec 6 §3.1's "`exp`, not `native_exp`".

## Global Constraints

- Branch `spec6c-flash-exp2-rows8` from main (231f41b or later); `tools/box.sh` then uses `~/b70-inference-server-spec6c` on the box automatically (`tools/box.sh dir` to check). If `tools/box.env` is missing in the worktree, copy it from `/Users/urakozz/CLionProjects/b70-inference-server/tools/box.env`; never commit it or its contents.
- Other agents share the one-GPU box: **every GPU command runs under `flock ~/b70-gpu.lock`**, detached (`tools/probe/detach.sh ~/LOG flock ~/b70-gpu.lock <cmd>`), poll for `ALLDONE rc=`. Record `uptime` load with every timing table; if the load average is above 4, say so next to the numbers (a CPU oracle run may be active).
- Timing: interleaved control/candidate pairs after a warm-up, median of 3 (memory: probe-timing-clock-state). The control is main's build in the base tree (`~/b70-inference-server`, `tools/box.sh` on main) or a second build of the merge base in your tree: say which.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. `exp2(s * (ATTN_SCALE * M_LOG2E_F) - m)` means the running max `m` is kept **in the log2 domain** (scaled scores): the correction `exp2(m - mnew)` and the masked `-INFINITY` must use the same domain, and the first tile's `m = -INF` must still give 0. K1 at depth 0 with 1 row catches a domain slip.
2. **RPW 8 with a chunk of C rows where C % 8 != 0** (C = 300, and the tail chunk of a 4097-id prompt): pad rows stay finite and unwritten beyond C. K1's (777, 300) case plus a C = 1 case.
3. **The launch count and replay:** the grid change must not change `attn_chunk_launches` (one launch per FA layer) and `prefill_replay_test` must stay bitwise.
4. **K3a (flash no further from the oracle than composed)** is rerun: the rounding point moved.

---

### Task 1: the kernel and the grid

**Files:** `src/kernels/prefill/pf_flash_attn.cl` (comment header: KT=64 RPW=8 HPW=6 QREG=0; the `exp` lines 113 and 118 and the scale in MASK per the research doc's §0/§ "promotion" form; update the header comment "with `exp`, never `native_exp`" to state the ruling), `src/kernels/prefill/CMakeLists.txt` (RPW=8), `src/runtime/prefill/attn.cc:90` (grid `(C + 7u) / 8u`).

- [ ] **Step 1:** copy the arm from `tools/probe/pfa_levers_research.cl` that measured 1.42x (KT64_R8_Q2) as the reference text; apply only the `exp2` + scale-fold part to the production kernel (no other research arm).
- [ ] **Step 2:** build on the box; `pf_flash_attn_test` (K1) passes, cosine >= 0.99999; add a C = 1 case and a depth-0 single-row case to it (Review Focus 1, 2).
- [ ] **Step 3:** golden gates in flash mode (16/16), `prefill_smoke_test`, `prefill_replay_test`, `flash_vs_oracle_test` (K3a), `flash_long_test`; then the full suite. All under the lock, detached.
- [ ] **Step 4: Commit** `kernels: pf_flash_attn with exp2 (scale folded) and 8 rows per work-group (spec 6c, research levers 1-2)`.

### Task 2: the speed, in the engine

- [ ] **Step 1:** interleaved pairs, control main vs this branch: pp4096 (t/s and the `attn_flash` phase ms from the prefill profile, as spec 6 §8 F1 measured it), pp65536 and pp130816 at max_len 131072 (F2 rows), decode at 4k depth (no-regression check; decode does not use this kernel).
- [ ] **Step 2:** the record: `docs/BENCHMARKS.md` (new rows in the flash/128k section; a new pp4096 RECORD row only if the load average was <= 4 and it beats 2125.12); spec 6 amendment §9 "Spec 6c" (the exp2 ruling, the numbers, F1 against 80 ms); `README.md` headline only if a RECORD row was written; `docs/research-flash-prefill-2026-09-27.md` P1 marked done with the in-engine numbers.
- [ ] **Step 3: Commit** `spec 6c: exp2 + RPW 8 in the engine, pp4096/pp65536/pp130816 (the record)`.

**Gate for the plan:** K1, K2 (16/16), K3a, K4, full suite green; the in-engine numbers recorded. Hand back with the pair table and the gate results.
