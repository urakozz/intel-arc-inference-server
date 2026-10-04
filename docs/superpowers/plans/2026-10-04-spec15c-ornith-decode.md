# Spec 15c - Ornith decode: the MoE block, the shape variants, P0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ornith decodes on the card: the loader repacks the experts, the router and the MoE block run in 3-4 launches per layer with expert ids consumed on the device, the GDN / attention / GEMV kernels run as Ornith-shape variants, P0 measures the floor and the arms of §9, and R2 / R3 (decode path) pass.

**Architecture:** spec 15 §4.1-4.3 and §9. Expert weights per layer stored as `[256]` contiguous int4 g64 blocks (gate‖up interleaved as the dense path's gate‖up is; down separately), so a kernel addresses expert `e` by offset. Per MoE layer: (1) router GEMV + softmax/top-k + the shared-expert sigmoid gate, writing 8 ids and weights to a device buffer; (2) gate‖up for the 8 routed experts and the shared expert as a ninth slot, SiLU x up fused; (3) down for the 9 slots + the weighted sum in fixed slot order + the residual fold. The captured decode list stays fixed; nothing returns to the host.

**Tech Stack:** OpenCL C (ocloc), C++17, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-04-spec15-ornith-moe-design.md` (§2, §3 decisions 1 and 3, §4.1-4.3, §5 R2, R3, §6 P0, §9). Needs 15a (golden sets in `oracle-out-ornith/`, the router formula, the quant source decided) and 15b (`ModelDesc` with the MoE fields) merged.

## Global Constraints

- Branch `spec15c-ornith-decode` from main; box tree `~/b70-inference-server-spec15c` (automatic). Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs (including `oracle-out-ornith`) into the box tree.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled with a background until-loop. Interleaved pairs, median of 3, `uptime` recorded.
- **R0 holds:** Qwen3.8 and Agnes unchanged (full suite; their kernel binaries checksum-identical to main's).
- Bitwise determinism: no atomics in any sum that feeds a result (the expert sum is in fixed slot order).
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Router exactness:** the device formula equals 15a's pinned one (softmax / top-k order, renormalisation, the shared gate); top-k ties broken by the lower expert index, as the reference does (check).
2. **Expert addressing:** a work-group reads `ids[slot]` once and its weights at `e x stride`; a wrong stride shows as R2 passing (right experts chosen) but R3 failing — test expert slices individually (Task 2's kernel test).
3. **The shared expert as the ninth slot** with its sigmoid gate applied to its down output before the sum; its intermediate (512) equals the routed experts', so one kernel serves all nine.
4. **Launch count per token** recorded (`capture.cc`'s count) and compared with P0's floor; GDN at 32 v-heads and decode attention at 8 q-heads per kv-head as variants (15b's names).
5. **The MTP head** has a MoE layer: the same kernels at M = 1; its decode is part of 15e, but the kernels must not assume layer < 40.

---

### Task 1: P0, the floor and the arms

**Files:** `tools/probe/probe_moe_decode.{cl,cc}`, `tools/probe/CMakeLists.txt`.

- [ ] **Step 1:** the per-kernel floor: replayed lists of N empty kernels and N small GEMVs at hidden 2048 (N = 200, 400, 800); µs per kernel.
- [ ] **Step 2:** the arms of spec §9 at Ornith shapes, each against our current kernel: the register-resident even/odd-K GEMV against our int4 g64 GEMV (K 2048, N 1024 and K 512, N 2048 per expert, and the dense attention / GDN projections); router top-k in one sub-group; the decode-attention "all K/V loads issued first" arm against v2 at GQA 8 and 4k depth.
- [ ] **Step 3:** the derived decode estimate (kernels x floor + bytes / measured bandwidth) and the proposed decode bar for the operator, in `docs/probe-ornith-decode-2026-10-04.md`. **Commit** `probe: Ornith decode floor and kernel arms (spec 15 P0)`.

### Task 2: the loader and the MoE kernels

**Files:** `src/loader/*` (repack fused or per-expert int4 into per-expert contiguous blocks; the router, shared gate, `in_proj_a/b` bf16), `src/kernels/moe_router.cl`, `src/kernels/moe_expert.cl` (gate‖up and down variants), `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`, `tests/kernels/moe_router_test.cc`, `tests/kernels/moe_expert_test.cc` (host references, Review Focus 1-3), `tests/loader/ornith_repack_test.cc` (host-only).

- [ ] Failing tests → implement (P0's winning GEMV structure if it won) → pass. **Commit** `kernels: MoE router and expert kernels, loader repack (spec 15 §4.1-4.2)`.

### Task 3: the decode list on Ornith

**Files:** `src/runtime/capture.cc` (MoE layers: the three launches in place of the dense MLP), `src/runtime/buffers.*` (router buffer, per-slot scratch), the GDN / attention / prep variants at Ornith shapes; `tests/CMakeLists.txt` registrations (Ornith decode golden gate on `oracle-out-ornith`, R2 router test, replay determinism on Ornith).

- [ ] **Step 1:** R2 (the engine's top-8 sets and weights against 15a's reference on the golden positions, with 15a's tolerance); R3 on the decode path (the golden gate with M = 1 decode of the prompt, as the original decode golden gate does); replay determinism bitwise; R0 (full suite).
- [ ] **Step 2: Commit** `runtime: Ornith decodes - MoE layers in the captured list (spec 15, R2-R3 decode)`.

### Task 4: decode speed

- [ ] Decode at 4k / 32k / 128k depth with `--lm-head int8`, launches per token, against Task 1's derived estimate. `docs/BENCHMARKS.md` section "Ornith 1.5 MoE (spec 15)" (decode rows). **Commit** `spec 15c: Ornith decode speed`.

**Gate for the plan:** P0 recorded with the proposed bars, R2 and R3 (decode) green, replay bitwise, R0 green, decode speed recorded.
