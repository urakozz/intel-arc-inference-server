# Spec 9b - int8 `lm_head` in the engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the loader quantises the bf16 `lm_head` to int8 per row at load, a `gemv_i8w` kernel reads it, `--lm-head bf16|int8` selects it (`b70-serve` default int8, `b70-decode` default bf16); gates L1-L4 and bars H1-H3; the record.

**Architecture:** host-side quantisation in the loader into a new `WeightKind` (int8 rows + fp32 row scales), a new decode kernel variant built like `gemv_bf16` (`src/kernels/gemv_bf16.cl`, variant `add_gemv_bf16_variant(1 5120 248320 64 1)` in `src/kernels/CMakeLists.txt`), routed by `Qwen35::lm_head(kind)` and the capture's head launch (`src/runtime/capture.cc`, "`lm_head` is the one launch ... that depends on the checkpoint"). Prefill's `step_head` reuses the decode binary, so the first token uses the same head.

**Tech Stack:** OpenCL C (ocloc), C++20, Level Zero, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-09-28-spec9-lm-head-quantised-design.md` (§2 ruling int8, §3, §4 L1-L4, §5 H1-H3). Needs plan 9a's verdict (`docs/probe-lm-head-2026-09-28.md`) to be a pass.

## Global Constraints

- Branch `spec9b-lm-head-int8` from main; box tree `~/b70-inference-server-spec9b` automatically (`tools/box.sh dir`). Copy `tools/box.env` from the main checkout if missing; never commit it. Symlink the `oracle-out*` dirs from `~/b70-inference-server/` into the box tree one by one.
- Other agents share the one GPU: every GPU command under `flock ~/b70-gpu.lock`, detached (`tools/probe/detach.sh`), background until-loop polling for `ALLDONE rc=`. Record `uptime` with every timing table.
- `--lm-head bf16` is exactly today's engine (bitwise: golden gates, replay determinism). BENCHMARKS rows name their head form; the vLLM comparison rows stay bf16.
- Interleaved pairs after a warm-up, median of 3. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **The kernel's efficiency:** the bf16 head runs at 98.4 % of device bandwidth; an int8 kernel that loses 10 % of that throws away a third of the saving. Measure the launch's GB/s against its bytes (1.272 GB + scales) and report it.
2. **Rows past `vocab_used`** still get computed (the logits buffer is `[kVocab]`), and argmax never selects them (unchanged masking).
3. **The scale layout** (fp32 per row, 248320 x 4 B = 0.99 MB) read once per work-group, not per element.
4. **MTP draft path (spec 8):** if spec 8's engine is merged when this runs, the draft list's head launch uses the same kind; its acceptance is re-measured (H2).
5. **Load time:** quantising 1.27 G values on the host must stay under ~10 s at `-O2`, or move to a one-off device kernel.

---

### Task 1: loader and weight kind

**Files:** `src/loader/*` (a `LmHeadForm` parameter, host quantisation: symmetric, scale = max|row| / 127, round to nearest even, clamp to [-127, 127]), `src/model/qwen35.h` (the int8 row of `lm_head(kind)`), the load report line (form, bytes, quantisation ms). Test: `tests/loader/lm_head_int8_test.cc` (host-only on a synthetic row set: round trip bound, scale layout, clamp) and an extension of `tests/loader/load_checkpoint_test.cc` (the real head: bytes 1,271,398,400 + 993,280 scale bytes, derived; quantisation time printed).

- [ ] Failing tests → implement → pass on the box → **Commit** `loader: lm_head quantised to int8 per row at load (spec 9 §3)`.

### Task 2: `gemv_i8w` and the capture

**Files:** create `src/kernels/gemv_i8w.cl` (from `gemv_bf16.cl`: int8 weight loads, per-row fp32 scale applied once after the row's dot product, fp32 accumulation, same `{C, S}` tiling), `src/kernels/CMakeLists.txt` (variant `M=1 K=5120 N=248320`, plus M = 2..4 if `B70_DECODE_EXTRA_M` or spec 8's verify lists exist on main), `src/kernels/kernels.h` (variant name), `src/runtime/capture.cc` (route the int8 kind). Test: `tests/kernels/gemv_i8w_test.cc` (host reference in C++ on random int8 weights and scales at the real shape: max rel error <= 1e-5 against an fp64 reference), and the capture's kernel-table test.

- [ ] Failing tests → implement → pass → **Commit** `kernels: gemv_i8w, the int8 lm_head GEMV (spec 9 §3)`.

### Task 3: flags, gates, speed

**Files:** `src/cli/b70_serve.cc` (`--lm-head bf16|int8`, default int8), `src/cli/b70_decode.cc` (same flag, default bf16), `tools/bench_decode.sh` / `tools/probe/serve_benchy.sh` pass-through. Tests: register the golden gates once more with the int8 head (`prefill_gate_int8_head_test` or an argument to the existing ones — follow `tests/CMakeLists.txt`'s pattern for backend variants).

- [ ] **Step 1: L1 on the card:** the engine's int8-head logits against its bf16-head logits on the golden prompts (same hidden, M = 1 decode): cosine, argmax, top-20, as plan 9a measured on the CPU; they should agree with 9a's table.
- [ ] **Step 2: L3:** golden gates with the int8 head on `l0` and `l0-int8`; gate A4 (`tools/toolcall/`) with `l0-int8` + int8 head: >= 25 / 36.
- [ ] **Step 3: L4:** full suite with `--lm-head bf16` default paths unchanged; replay determinism with the int8 head.
- [ ] **Step 4: H1-H3:** interleaved pairs, decode at 4k depth, int8 head against bf16 head (bar >= 1.06x); the head launch's GB/s (Review Focus 1); pp4096 within 1 %; H2 (MTP draft step) if spec 8's engine is on main.
- [ ] **Step 5: the record:** `docs/BENCHMARKS.md` section "int8 lm_head (spec 9)" with the head form named on every row; spec 9 amendment §8; `README.md`: the decode line gains the int8-head serving number beside the byte-matched bf16 one; `docs/03-models.md` / `docs/13-loader.md` lines on the head's forms.
- [ ] **Step 6: Commit** `spec 9: int8 lm_head gates and speed (the record)`.

**Gate for the plan:** L1-L4, H1, H3 met (H2 recorded if applicable), full suite green. Hand back with the pair table and the gate results.
