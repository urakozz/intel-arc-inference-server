# Spec 14 - phase 1: write Agnes support without the box

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** all code for Agnes 3.0 Flash on branch `spec14-agnes`, checked as far as a Mac allows, so that one box session (`2026-10-03-spec14-validation-checklist.md`) validates and merges it.

**Architecture:** spec 14 §2-3: a `ModelDesc` replaces the hardcoded layer counts and intermediate size; the loader maps Agnes's names and folds the parallel FFN into a wider MLP (intermediate 19456) on packed int4; kernel variants at the new shapes; a CPU reference in `tools/oracle/`. Task detail for each part is in the superseded plans `2026-10-02-spec14a..14d` (read them; this plan says what changes because there is no box).

**Tech Stack:** C++20 (Apple clang on the Mac for host code), OpenCL C (written, not compiled here), Python 3 + torch + transformers via `uvx` / a local venv, CMake.

**Spec:** `docs/superpowers/specs/2026-10-02-spec14-agnes-3-flash-design.md` (all; §6 the re-scope).

## Global Constraints

- **No box.** Do not call `tools/box.sh` or ssh. Everything runs on the Mac.
- Branch `spec14-agnes` from main; commit there; no merge, no push.
- **Qwen3.8's behaviour must not change.** Every edit to a shared path is behaviour-neutral for Qwen3.8 by construction (same shapes, same launches, same kernel source text after preprocessing for Qwen3.8's defines). Where you cannot prove it on the Mac, add the check to the validation checklist.
- **No claim without a measurement.** Anything that needs the card (tuning, gates, speed) is written as provisional and listed in the checklist, never reported as passing.
- Real checkpoint files for reference: download the small ones now (`config.json`, `model.safetensors.index.json`, `chat_template.jinja`, `quantization_config.json`, `modeling_agnes.py`, `configuration_agnes.py`, `generation_config.json`, `tokenizer_config.json`) with `uvx --from huggingface_hub hf download urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ <files> --local-dir <scratch>`; the operator is downloading the full checkpoint into `~/.cache/huggingface` — use it for Task 2's real-layer proof if present, otherwise mark that step pending.
- No `rm -rf`. The Mac has 68.7 GB RAM: never load the whole model in bf16 (~65 GB); work layer by layer.

## Review Focus

1. **The refactor's blast radius:** `grep -rn "kLayers\|kIntermediate\|is_fa\|17408\|34816\|kFaLayers"` over `src tests tools` before and after; every remaining literal justified.
2. **The fold's packing** (spec §2): `qweight [K/8, N]`, `scales`/`qzeros [K/64, N]`; gate/up joined along N (columns), down along K at `qweight` row 2176 / `scales` row 272; `g_idx` identity asserted; the C++ fold and the Python fold produce byte-identical tensors on the same input (a shared synthetic fixture).
3. **Host code compiles on the Mac:** model, loader, server, tokenizer, and their host-only tests build and run with Apple clang (a small local CMake or a script, not the box build). Runtime / capture / prefill files are syntax-checked against the Level Zero headers (clone `https://github.com/oneapi-src/level-zero` headers into the scratchpad; `-fsyntax-only`).
4. **Provisional GEMV tuning rows** are marked as such in code and docs, and the checklist names each one.
5. **The Agnes chat template:** rendered locally with `transformers`' `apply_chat_template` on 5 message lists (plain, thinking, tools, a tool call in history, a tool response); the engine's renderer compared where it can run locally (`src/tokenizer` host tests), else listed.

---

### Task 1: `ModelDesc` (Qwen3.8 only)

Detail: `2026-10-02-spec14b-model-descriptor.md` Tasks 1-3.

- [ ] **Step 1:** `src/model/model_desc.{h,cc}`, `tests/model/model_desc_test.cc`; compile and run the test on the Mac.
- [ ] **Step 2:** route `src/runtime/{buffers,capture,engine}`, `src/runtime/prefill/*`, `src/loader/*` through it; kernel variant names from the descriptor (`src/kernels/kernels.h`, the CMake files); Review Focus 1, 3.
- [ ] **Step 3: Commit** `model: ModelDesc replaces the hardcoded layer counts (spec 14, unvalidated on the card)`.

### Task 2: the CPU reference and the fold proof

Detail: `2026-10-02-spec14a-agnes-oracle-and-fold.md` Tasks 1-2.

- [ ] **Step 1:** `tools/oracle/agnes.py`, `tools/oracle/agnes_fold.py` and their tests; run the tests locally (synthetic tensors; the name map against the real `model.safetensors.index.json`).
- [ ] **Step 2:** if the checkpoint is in the Mac's HF cache: the fold proof on layers 0, 3, 35, 71 with real packed weights and inputs from a layer-by-layer forward of a short prompt (embedding + the layers up to the one tested; keep RAM well under 60 GB), cosine >= 0.999999; `modeling_agnes.py` vs our reference on a 64-id prompt if it fits layer by layer, else pending.
- [ ] **Step 3:** `docs/probe-agnes-2026-10-03.md` with what ran and what is pending. **Commit** `oracle: Agnes CPU reference and the fold proof (spec 14)`.

### Task 3: Agnes in the loader and the engine

Detail: `2026-10-02-spec14c-agnes-engine.md` Tasks 1-2 (and Task 3's registrations).

- [ ] **Step 1:** the Agnes `ModelDesc`; the loader's name map and the packed fold; `tests/loader/agnes_fold_test.cc` (host-only, the shared fixture of Review Focus 2) compiled and run on the Mac.
- [ ] **Step 2:** kernel variants at `INTERMEDIATE=19456`, GEMV shapes `N=38912 K=5120` and `N=5120 K=19456` with **provisional** tuning rows, prefill slab counts 38 / 19; buffers sized from the descriptor; the MTP lists; the 65536 max_len ceiling for Agnes; test registrations for the Agnes gates (pointing at `oracle-out-agnes`, to be dumped on the box).
- [ ] **Step 3: Commit** `engine: Agnes 3.0 Flash - descriptor, loader fold, kernel variants (spec 14, unvalidated on the card)`.

### Task 4: serving

Detail: `2026-10-02-spec14d-agnes-serving.md` Task 1.

- [ ] Template check (Review Focus 5); Agnes's tool-call format against `src/server/toolcall` (extend the parser only if the format differs, with the spec 7a streaming tests, run locally if the server tests build on the Mac). **Commit** `server: Agnes chat template and tool calls (spec 14, unvalidated on the card)`.

### Task 5: the validation checklist

- [ ] Write `docs/superpowers/plans/2026-10-03-spec14-validation-checklist.md` concrete for this branch: the exact commands in order (build; G0 with the binary checksum comparison against main's build; kernel tests; GEMV tuning sweep replacing each provisional row by name; oracle dumps for `oracle-out-agnes` and the A4 references; G1 card, G2, G4, G5; A4; speed rows; the vLLM row with PR #57003; the record), each with its pass condition, and every item from this plan that ran only partly on the Mac. **Commit** `plans: spec 14 validation checklist for the box`.

**Gate for the plan:** everything written and committed on `spec14-agnes`; host code compiled and host tests passing on the Mac; Python tests passing; the fold proof run on real layers or listed as pending; the checklist complete. Hand back with: what compiled and ran locally, what is pending for the box, the list of provisional values.
