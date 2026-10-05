# Spec 18c - K2-Horizon prefill

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** K2 prefills in chunks: dense and attention linears through spec 5's path, MoVA and MoE through grouped expert GEMMs (spec 15d's routing and grouped kernels, MoVA added), flash attention at head_dim 128 / GQA 4; K2 and K3 on prefill; prefill speed.

**Architecture:** spec 18 §5.3. Per chunk: route all tokens for MoVA (top-4 of 64) and MoE (top-8 of 100) on the device; histogram, prefix sum, scatter; grouped GEMMs per expert (MoVA 2560 → 1024 + SiLU; MoE gate‖up 2560 → 1536, down 768 → 2560); weighted combines in ascending id; the shared expert dense. Flash attention variants at head_dim 128, GQA 4, K1-checked against fp64 as spec 6.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec18-k2-horizon-design.md` (§5.3, §6 K2, K3, §7). Needs 18b and spec 15d (grouped MoE prefill) merged.

## Global Constraints

- Branch `spec18c-k2-prefill` from main; box tree automatic; `tools/box.env` copied if missing, never committed; `oracle-out*` symlinked.
- K0 holds (other models unchanged). Lock, detached, polled; interleaved pairs, median of 3.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **MoVA in prefill:** the value written to the KV cache for every position is the routed mix, identical in rule to decode's; a test compares prefill's KV rows with decode's M = 1 fill for the same tokens (cosine bar as spec 8b's prefill-head gate, bitwise not expected).
2. **Grouped GEMM row independence** (spec 15d's rule) for both MoVA and MoE: permuting rows inside an expert's tile leaves per-row outputs bitwise unchanged.
3. **Flash attention at head_dim 128 / GQA 4:** K1 (cosine >= 0.99999 against fp64) at depths 0, 2k, 30k, 60k; the tail chunk and single-row cases.
4. **Continuation:** `prefill_split_*` on K2 (multiples of 64 bitwise; others within the split test's bars).
5. **The routing diagnostic on prefill:** the engine's prefill routing against 18a's dumps for the prompt positions.

---

### Task 1: P0 for K2 prefill

- [ ] Routing and grouped GEMM timings at K2's shapes and expert counts (100 / 64 experts, real row distribution from 18a's dumps); flash at head_dim 128; the derived pp4096 estimate. `docs/probe-k2-prefill-2026-10-05.md`. **Commit** `probe: K2 prefill shapes (spec 18c)`.

### Task 2: kernels and the prefill step

- [ ] MoVA routing + grouped value-expert GEMM + combine; MoE through spec 15d's path with K2's router; flash variants; the K2 prefill step; tests (Review Focus 1-5); golden gates on prefill (K2), split and replay tests (K3); K0 full suite. **Commit** `prefill: K2-Horizon chunks through grouped MoVA and MoE (spec 18c)`.

### Task 3: speed

- [ ] pp4096, pp16384, pp at 32k; BENCHMARKS rows. **Commit** `spec 18c: K2 prefill speed`.

**Gate for the plan:** K2 / K3 on prefill green, the flash K1 cases green, speed recorded.
