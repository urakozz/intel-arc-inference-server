# Spec 18b - K2-Horizon decode on one card

**Status (2026-10-05, branch `spec18b-k2-decode`):** Tasks 1-3 written blind on the Mac (the box
unavailable) and rebased on 18a's reference; Mac checks green (`tools/mac_check.sh`: host tests,
Level Zero syntax, kernel command lines additions only, OpenCL syntax; `--kernels`: the portable K2
kernels bit-exact on the Mac's GPU, indicative). As built: spec 18 §10. Everything on the card -
K1 on ocloc's build, the checkpoint load, K2 (after 18a's `oracle-out-k2/`), K3, and Task 4 - is
the box validation queue's K2 row.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** K2 decodes through a replayed Level Zero list on one B70: model table, loader with flat per-layer expert buffers, K2's own kernels, spec 15's MoE kernels with K2's router, `K2Engine`; K0, K1, K2 (decode), K3 (replay); the decode speed rows.

**Architecture:** spec 18 §5.1-5.2. The old plans carry most of the detail — **read them first**: `git show d307d0e:docs/superpowers/plans/2026-09-14-plan8b-spec4-model-loader.md` (model table, RoPE table, `load_k2`), `...plan8c-spec4-kernels.md` (grouped norm, SiLU·mul, router top-k, expert slots and combines, the attention trio), `...plan8e-spec4-engine-gate.md` (`K2Buffers`, capture, `K2Engine`, golden gate, CLI dispatch). **What changes against them:**
- the 8 + 4 fixed **slot launches** become spec 15's single launches over all selected experts (MoE gate‖up for 8 + shared; down + combine + residual; MoVA's 4 value experts + combine), with the combine order a parameter (ascending id for K2);
- the router top-k is spec 15's kernel parameterised for K2's formula (sigmoid, selection-only bias, × 2.5 after normalising, ascending id);
- decode attention is spec 10's v2 at head_dim 128 / GQA 4 (old plan 8c Task 5's trio is replaced);
- the `lm_head` is spec 9's int8 by default in `b70-serve`, bf16 in `b70-decode`;
- `max_len` per spec 18 decision 2 (proposed 32k bf16 KV).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec18-k2-horizon-design.md` (§2, §3, §4 decisions 2-3, §5.1-5.2, §6 K0-K3, §7). Needs 18a (golden sets, routing dumps, facts) and spec 15c (the MoE decode kernels) merged.

## Global Constraints

- Branch `spec18b-k2-decode` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` (including `oracle-out-k2`) symlinked.
- **K0:** Qwen3.8 (and every other merged model) unchanged — the full suite green, their kernel binaries checksum-identical to main's.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled; interleaved pairs, median of 3, `uptime` recorded.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The two norm traps** (spec 4 §3.1): plain `w · x̂`, and two groups of 1280 each with its own mean of squares.
2. **The softplus threshold branch** (`β·x > 20 → x`) has a kernel test with values above it.
3. **Routing diagnostic:** the per-layer first position where the engine's MoE / MoVA ids differ from 18a's dumps, with 18a's tie tolerance; any non-tie difference fails.
4. **The MoE router's 100 rows** padded to 128 (zero rows) for the bf16 GEMV, top-8 reading rows 0-99 only.
5. **Launch count per token** asserted at capture and recorded against spec 18 §2's ~1000 estimate.

---

### Task 1: model table, RoPE table, loader (old plan 8b)

- [x] (blind; `k2_load_checkpoint_test` is the box's) As old plan 8b Tasks 1-3, with flat per-layer expert buffers sized for spec 15's kernels; host tests on the Mac where they build; `k2_load_checkpoint_test` on the box. **Commit** `model: K2-Horizon table and loader (spec 18b)`.

### Task 2: kernels (old plan 8c, revised)

- [x] (blind; `k2_kernels_test` is the box's) Grouped norm, softplus gate in the attention reduce, MoVA value-expert launch + combine, spec 15's MoE kernels with K2's router and ascending-id combine, attention v2 / flash variants at head_dim 128 GQA 4; a host-reference test each (Review Focus 1, 2, 4). **Commit** `kernels: K2-Horizon decode kernels (spec 18b, K1)`.

### Task 3: `K2Engine`, capture, gates (old plan 8e, revised)

- [x] (blind; the gates run on the box) `K2Buffers`, the capture (Review Focus 5), `K2Engine` (`reset` / `ingest` / `generate`), replay determinism (K3), the golden gate with the routing diagnostic (K2, Review Focus 3), `b70-decode` dispatch on `model_type`; K0 (full suite). **Commit** `runtime: K2Engine decodes (spec 18b, K2-K3)`.

### Task 4: speed

- [ ] Decode at 4k / 16k / 32k depth with bf16 and int8 heads, launches per token, against the derived roofline (~190 t/s weights-only with the int8 head). `docs/BENCHMARKS.md` section "K2-Horizon (spec 18)". **Commit** `spec 18b: K2 decode speed`.

**Gate for the plan:** K0-K3 (decode) green, the routing diagnostic clean, the speed rows recorded.
