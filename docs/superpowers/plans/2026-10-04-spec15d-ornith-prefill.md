# Spec 15d - Ornith prefill: device routing and grouped expert GEMMs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ornith prefills in chunks: the router for all tokens, a device histogram / prefix sum / scatter into expert-sorted order, one grouped GEMM launch for every expert's gate‖up and one for down, the weighted gather back per token in fixed order, the shared expert as a dense GEMM; R3 on the prefill path; prefill speed.

**Architecture:** spec 15 §4.4 and §9. The routing produces a per-chunk tile table (expert, row offset, row count) padded with -1 so the grouped GEMM's grid is fixed and the host never reads counts. The grouped GEMM's work-groups each own one expert's row tile; the two dequant strategies (dequant into SLM once per work-group per K step, or a separate dequant pass as today's dense prefill does) are both probed and the winner is built. Attention (flash, spec 6) and GDN prefill run as Ornith-shape variants.

**Tech Stack:** OpenCL C (ocloc), C++17, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§2 prefill, §4.4, §5 R3, §6, §9). Needs 15c merged (loader repack, router kernel, MoE decode, Ornith variants) and 15a's router statistics (rows per expert per chunk).

## Global Constraints

- Branch `spec15d-ornith-prefill` from main; box tree `~/b70-inference-server-spec15d`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved pairs, median of 3, `uptime` recorded.
- R0 holds (Qwen3.8 and Agnes unchanged). **Determinism:** rows may land in an expert's tile in any order only if each row's GEMM output is independent of the other rows in the tile; prefill replay must stay bitwise (`prefill_replay_test` pattern on Ornith).
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Row independence** of the grouped GEMM (Global Constraints): a test that permutes the rows inside each expert's tile and expects bitwise-identical per-row outputs.
2. **Uneven experts:** tiles for experts with 0 rows (skipped), 1 row, and the maximum from 15a's statistics; the -1 padding never reads or writes.
3. **The gather** sums each token's 8 routed results plus the shared expert in fixed slot order (the same order as decode), so prefill and decode agree on the combination rule.
4. **Chunk boundaries and continuation:** spec 7's split-prefill test (`prefill_split_*`) on Ornith, multiples of 64 bitwise.
5. **The grouped GEMM's efficiency** is measured as TFLOP/s on the real row distribution (15a), not at a uniform 64 rows.

---

### Task 1: P0 for prefill

**Files:** `tools/probe/probe_moe_prefill.{cl,cc}`, `tools/probe/CMakeLists.txt`.

- [ ] Routing kernels (histogram, prefix sum, scatter) timed at C = 2048; the grouped GEMM in both dequant strategies at the real distribution (gate‖up K 2048 N 1024; down K 512 N 2048), TFLOP/s and ms per layer; the derived pp4096 estimate and the proposed prefill bar; `docs/probe-ornith-prefill-2026-10-04.md`. **Commit** `probe: Ornith prefill routing and grouped GEMM arms (spec 15 P0)`.

### Task 2: routing and grouped GEMM kernels

**Files:** `src/kernels/prefill/pf_moe_route.cl`, `src/kernels/prefill/pf_moe_gemm.cl` (the winning strategy), `src/kernels/prefill/pf_moe_gather.cl`, CMake, `tests/kernels/pf_moe_*_test.cc` (host references; Review Focus 1-3).

- [ ] Failing tests → implement → pass. **Commit** `kernels: Ornith prefill - device routing, grouped expert GEMM, gather (spec 15 §4.4)`.

### Task 3: the prefill step on Ornith

**Files:** `src/runtime/prefill/{step,engine_prefill}.*` (the MoE block in the chunk step), registrations (Ornith prefill golden gates, `prefill_split_*` and `prefill_replay_test` on Ornith).

- [ ] R3 on prefill (`l0`; `l0-int8` only if the h8 path is built for experts — not in this plan unless P0 shows it pays), split and replay tests, R0 full suite. **Commit** `prefill: Ornith chunks through the grouped experts (spec 15, R3 prefill)`.

### Task 4: prefill speed

- [ ] pp4096, pp32768 and pp at 128k; the BENCHMARKS section's prefill rows. **Commit** `spec 15d: Ornith prefill speed`.

**Gate for the plan:** P0 recorded, R3 (prefill), split and replay tests green, R0 green, speed recorded.
