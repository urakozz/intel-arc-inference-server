# Spec 14b - the model descriptor (Qwen3.8 only, behaviour-neutral)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** replace the hardcoded layer counts and intermediate size with a `ModelDesc` chosen at load, with Qwen3.8 as its only instance, and prove nothing moved (G0: golden gates and replay bitwise, Qwen3.8 kernel binaries byte-identical).

**Architecture:** `model::Qwen35`'s `static constexpr kLayers` (64), `kIntermediate` (17408) and `is_fa` (`l % 4 == 3`) move into `struct ModelDesc { uint32_t layers, gdn_layers, fa_layers, intermediate; bool parallel_ffn; ... }`, held by `loader::LoadedModel` and passed where the constants are used today. Shapes shared by every Qwen3.5-family model (hidden 5120, heads, head dims, vocab, rotary) stay `constexpr`. Kernels whose source bakes the intermediate size keep getting it as a define; the capture picks the variant name from the descriptor (only Qwen3.8's variants exist after this plan).

**Tech Stack:** C++20, CMake, OpenCL C (defines only), ctest.

**Spec:** `docs/superpowers/specs/2026-10-02-spec14-agnes-3-flash-design.md` (§3.1, §4 G0, §6 14b).

## Global Constraints

- Branch `spec14b-model-descriptor` from main; box tree `~/b70-inference-server-spec14b`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs into the box tree one by one (and `oracle-out-spec8b` from `~/b70-inference-server-spec8b` for `mtp_head_test`).
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled with a background until-loop.
- **Behaviour-neutral:** every existing registration passes; golden gates and `replay_determinism_test` (+ `_i8head`) bitwise; every Qwen3.8 kernel `.bin` byte-identical to main's build (compare checksums of `build/kernels/*.bin` between a main build and this branch).
- box.sh's rsync keeps Mac mtimes: touch changed sources before syncing. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **Every use is routed**, none left: `grep -rn "kLayers\|kIntermediate\|is_fa\|17408\|34816\| 48 \| 16 layers\|\[48\|\[16" src tests` before and after; each remaining literal is justified in a comment (e.g. a per-head 16).
2. **Constants inside kernels** (`src/kernels/prep.cl`, `src/kernels/prefill/pf_gemm.cl`, `pf_prep.cl`): they come from CMake defines per variant; no new runtime branches in kernels.
3. **Buffer sizes** from the descriptor (`gdn_state`, `conv_ring`, `kv_k`/`kv_v`, `attn_part`, MTP slots, spec 7's `state_bytes`/`kv_bytes`): `buffers_test` asserts Qwen3.8's exact byte counts unchanged.
4. **The loader rejects an unknown architecture** with a clear error naming `config.json`'s `architectures`.
5. **Tests that build `Qwen35` shapes directly** (kernel tests, `kernel_table_test`) keep compiling against Qwen3.8's descriptor without behaviour change.

---

### Task 1: `ModelDesc`

**Files:** `src/model/model_desc.{h,cc}` (new), `src/model/qwen35.{h,cc}`, `src/loader/loader.{h,cc}` (pick the descriptor from `config.json`'s `architectures[0]`: `Qwen3_5ForConditionalGeneration` → Qwen3.8's), `tests/model/model_desc_test.cc`.

- [ ] Failing test (descriptor values for Qwen3.8; unknown architecture throws) → implement → pass. **Commit** `model: ModelDesc, Qwen3.8's shape as data (spec 14b)`.

### Task 2: route the runtime

**Files:** `src/runtime/{buffers,capture,engine}.{h,cc}`, `src/runtime/prefill/{step,int8,linear_l0,gemm_l0}.*`, `src/runtime/capture.cc`'s `kFaLayers`.

- [ ] Replace each use (Review Focus 1, 3); build; `buffers_test`, `kernel_table_test`, `replay_determinism_test` (+ `_i8head`) pass. **Commit** `runtime: layer counts and intermediate size from ModelDesc (spec 14b)`.

### Task 3: kernels and CMake

**Files:** `src/kernels/CMakeLists.txt`, `src/kernels/prefill/CMakeLists.txt`, `src/kernels/kernels.h` (variant names take the intermediate as a parameter).

- [ ] Route the variant names; Qwen3.8's binaries byte-identical (Global Constraints check). **Commit** `kernels: variant names from ModelDesc (spec 14b)`.

### Task 4: G0

- [ ] The full suite on the box; the kernel-binary checksum comparison against a main build; golden gates and replay bitwise. Record in `docs/probe-agnes-2026-10-02.md` (section "Descriptor refactor") or a short new section in the spec 14 amendment if 14a's doc is not on main yet. **Commit** `spec 14b: descriptor refactor is behaviour-neutral (G0)`.

**Gate for the plan:** G0 (full suite green, binaries byte-identical, bitwise gates). Hand back with the checksum comparison and the suite count.
