# Spec 14c - Agnes in the engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** load `urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` and run it through decode, prefill (`l0`, `l0-int8`), MTP, the prefix cache and the int8 head: Agnes's `ModelDesc`, the loader's name map and the parallel-FFN fold, the new kernel variants and tuning rows; gates G1 (card), G2, G4, G5; speed rows.

**Architecture:** 14b's `ModelDesc` gains Agnes `{layers 72, gdn 54, fa 18, intermediate 19456 (folded), parallel_ffn true}`. The loader maps `delta_attn.` / `global_attn.` to the engine's names and folds `mlp.parallel_ffn.{gate,up,down}_proj` into `mlp.{gate,up,down}_proj` at load (packed int4, spec §2). Kernels with the intermediate baked in get `INTERMEDIATE=19456` variants; decode GEMV shapes gate‖up `N=38912 K=5120` and down `N=5120 K=19456`; prefill GEMM/h8 at 38 / 19 slabs.

**Tech Stack:** C++20, OpenCL C (ocloc), Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-10-02-spec14-agnes-3-flash-design.md` (§2, §3.1-3.3, §4 G1, G2, G4, G5, §5). Needs 14a (CPU reference, `oracle-out-agnes/` on the box, fold proof) and 14b (`ModelDesc`) merged.

## Global Constraints

- Branch `spec14c-agnes-engine` from main; box tree `~/b70-inference-server-spec14c`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs (including `oracle-out-agnes`) into the box tree.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. Interleaved pairs, median of 3, `uptime` recorded.
- **Qwen3.8 unchanged:** the full suite including every Qwen3.8 gate passes; Qwen3.8 kernel binaries byte-identical to main's.
- Agnes `max_len` ceiling **65536** (spec §3.3); the loader refuses more for Agnes with a message pointing at spec 12. MTP lists for Agnes at 16384, as Qwen3.8's.
- No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The fold's packed indexing** matches 14a's proven CPU fold exactly (gate/up joined along N, the columns of GPTQ's `[K/8, N]` `qweight`; down joined along K at `qweight` row 2176 and `scales` row 272); `g_idx` asserted identity.
2. **The GEMV tuning rows** for the two new shapes are measured, not copied: each picked config's GB/s against bytes is reported (the bf16 / int4 GEMVs run at ~590 GB/s; a new row much below that is a finding).
3. **h8 (spec 5) column scales** (`pf_colmax_rot`, `Engine::prepare_prefill()`) cover 38 / 19 slabs for Agnes; `prepare_prefill` time recorded.
4. **Memory at 65536** fits with MTP lists and the int8 head (load report line); the prefix cache's host budget unaffected.
5. **Agnes's 18 FA layers in spec 10's v2 decode attention and spec 6's flash prefill:** layer count is the only change; `attn_v2_test` / `pf_flash_attn_test` are shape-agnostic but the engine-level gates run on Agnes.

---

### Task 1: descriptor, loader, fold

**Files:** `src/model/model_desc.{h,cc}` (Agnes row; `AgnesForConditionalGeneration` → it), `src/loader/*` (name map; the fold; the classification table rows), `tests/loader/agnes_fold_test.cc` (host-only, synthetic GPTQ tensors: the fold's byte layout equals the CPU proof's), `tests/loader/load_checkpoint_test.cc` (an Agnes case: tensor counts, folded shapes, bytes; checkpoint label).

- [ ] Failing tests → implement → pass on the box. **Commit** `loader: Agnes 3.0 Flash - names and the parallel-FFN fold (spec 14 §2)`.

### Task 2: kernels and tuning

**Files:** `src/kernels/CMakeLists.txt`, `src/kernels/prefill/CMakeLists.txt`, `src/kernels/kernels.h`, the GEMV tuning table (where `capture.cc` reads `{C, S}` per shape), `tests/kernels/*` cases at the new shapes (host reference, `gemv_test` / `pf_gemm_test` / `pf_int8_test` patterns).

- [ ] **Step 1:** variants at `INTERMEDIATE=19456`; GEMV shapes `N=38912 K=5120`, `N=5120 K=19456`; prefill slab counts 38 / 19.
- [ ] **Step 2:** tune the two GEMV shapes with the method of `docs/probe-gemv-2026-08-24.md` (sweep `C`/`S`/layout, interleaved); record the table and GB/s (Review Focus 2).
- [ ] **Step 3:** kernel tests at the new shapes pass; Qwen3.8 binaries byte-identical. **Commit** `kernels: Agnes shapes and GEMV tuning rows (spec 14)`.

### Task 3: the engine on Agnes

**Files:** `src/runtime/*` only where 14b left a Qwen3.8 assumption; `tests/CMakeLists.txt` registrations of the Agnes gates (pattern: the existing golden / prefill gate registrations with `oracle-out-agnes` and the Agnes repo id).

- [ ] **Step 1: G1 on the card:** one layer's folded MLP output (decode GEMVs, and prefill `l0` / `l0-int8`) against a host reference of the two-branch MLP on the same input.
- [ ] **Step 2: G2:** golden gates on Agnes, `l0` and `l0-int8`, the golden tie rule; `prefill_split_*` on Agnes; replay determinism on Agnes.
- [ ] **Step 3: G4:** spec 7 `snapshot_test` + `prefix_gpu_l0-int8` on Agnes; spec 8 `mtp_head_test` (M1, against 14a's reference head dump) and `mtp_verify_test` (M2); spec 9's int8 head golden gate; spec 10 v2 on Agnes (golden).
- [ ] **Step 4: G5:** passkey 3/3 at ~60k (`tools/probe/passkey.py` with a 60k filler).
- [ ] **Step 5:** full suite (Qwen3.8 and Agnes registrations). **Commit** `tests: Agnes gates G1, G2, G4, G5 (spec 14)`.

### Task 4: speed

- [ ] Decode at 4k / 32k / 60k depth (max_len 65536), pp4096, load time, memory report, `--mtp 1..3` greedy on the golden prompts (as spec 8 D1); against the spec's derived expectations (~26 t/s, ~1850-1900 t/s). `docs/BENCHMARKS.md` section "Agnes 3.0 Flash (spec 14)". **Commit** `spec 14c: Agnes speed rows`.

**Gate for the plan:** G1, G2, G4, G5 green; the full suite green; Qwen3.8 binaries byte-identical; speed rows recorded. Hand back with the gate results and the speed table.
