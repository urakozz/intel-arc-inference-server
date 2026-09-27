# Spec 8b - MTP in the engine: the head, the verify list, rollback

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `runtime::Engine` runs MTP iterations: load the head, fill its KV during prefill, draft k tokens, verify k + 1 rows in one main-model step with a GDN state per row, and commit j accepted drafts. Acceptance itself is plan 8c's (host); this plan exposes the engine calls and proves them (M1, M2, M5 engine part, D3).

**Architecture:** three captured lists beside today's decode list: **draft** (the head at M = 1), **verify** (the main model at M = K + 1, K from plan 8a's verdict), **commit** (pos, GDN slot select, head hidden/KV bookkeeping). The M-variants come from plan 8a's `capture.cc` refactor, now compiled for M = K + 1 in production. `gdn_step` gains a per-row state write into `gdn_state_spec`.

**Tech Stack:** C++20, OpenCL C kernels compiled with ocloc, Level Zero, CMake/ctest.

**Spec:** `docs/specs/2026-09-27-spec8-mtp-speculative-decoding-design.md` (§3.1-3.5, §4 M1, M2, M5, §5 D3). Needs: plan 8a's doc (`docs/probe-mtp-2026-09-27.md`: the head's wiring, K, the commit mechanism), plan 8a's `capture.cc` M parameter and `tools/oracle/mtp_ref.py`; spec 7 plan 7b merged (`Engine::save_state/load_state/save_kv/load_kv`, `set_block_hook`).

## Global Constraints

- `tools/box.env` is untracked: never commit it or its contents. No `rm -rf`.
- Box: `tools/box.sh sync|build|test|run`, `-j44`, device 0; long runs detached, poll for `ALLDONE`.
- **MTP off is today's engine:** without `--mtp`/`load_mtp`, no new allocation, no new kernel launch, decode and prefill bitwise unchanged (golden gates, `replay_determinism_test`) and within 1 % in speed (D3, interleaved pairs).
- Every kernel change lands with its own unit test in the style of `tests/kernels/*` (host reference in C++, checked on the card).
- Commit on branch `spec8b-mtp-engine`; no merge, no push.

## Review Focus

1. **Stale draft rows in the conv ring and KV** after a partial accept: the next verify step must read only accepted positions' history. M2's "64 further tokens equal a run that never saw the rejected drafts" covers it; run it for every j in 0..K-1 at K = plan 8a's K, and at a position where `pos % 16` wraps inside the verify rows.
2. **The head's KV and hidden at a prefill chunk's last position:** it has no next id during prefill (spec §3.2); the first verify step must fill it before any draft reads it. Test: prefill, then one MTP iteration, then compare the head's KV row at `pos - 1` with an M = 1 reference fill.
3. **max_len edge:** a verify step at `pos + K >= max_len` must not write past the KV or RoPE table; the engine falls back to M = 1 steps for the last K positions. Test at max_len 16384 with pos = 16380.
4. **Spec 7 snapshots with MTP on:** `save_kv`/`load_kv` and `save_state`/`load_state` include the head's KV (17th layer) and the head's hidden; `state_bytes()`/`kv_bytes()` report the larger sizes. Test: spec 7b's C1 case A with MTP on.
5. **Replay determinism of three lists** interleaved in varying acceptance patterns: two runs with the same token decisions are bitwise identical (M5).

---

### Task 1: load the head

**Files:** Modify `src/loader/*` (keep `mtp.*` when asked; today `build_view` skips them, `docs/13-loader.md:154`), `src/model/qwen35.h` (the head's tensor table), `src/runtime/buffers.{h,cc}` (the head's KV `[max_len][4][256]` bf16 K and V, `gdn_state_spec [K+1][...]`, head hidden `[K+1][5120]`; allocated only with MTP); the load report lines. Test: extend `tests/loader/load_checkpoint_test.cc` (15 `mtp.*` tensors, shapes, 0.849 GB) and `tests/runtime/buffers_test.cc` (sizes, and none allocated without MTP).

- [ ] Steps: failing tests → implement → PASS on the box → commit `loader: the MTP head, loaded on request (spec 8 §3.1)`.

### Task 2: the head's kernels and the draft list

**Files:** Create `src/kernels/decode/mtp_fc.cl` if `fc` over the concatenation is not expressible with the existing bf16 GEMV plus a gather (prefer reuse: the `lm_head` bf16 GEMV kernel handles `[N, K]` bf16 weights at M rows; the two pre-fc RMS norms are `prep` variants); the head's transformer layer reuses the FA-layer kernels at bf16 weights (if the decode FA path only has int4 GEMVs, add the bf16 GEMV route for these seven linears). Modify `src/runtime/capture.cc` (a `capture_draft()` building the draft list), `src/runtime/engine.{h,cc}`.

**Interfaces (produces):**
```cpp
// runtime::Engine, MTP on only
void draft(uint32_t k);                       // replays the draft list k times; draft ids in
                                              // Control::draft[0..k), logits rows in mtp_logits[k]
const float* mtp_logits_device() const;       // [k][kVocab] fp32, for the host sampler (8c)
```
M1: `tests/runtime/mtp_head_test.cc` (checkpoint label): on the golden prompts, after prefill, the head's logits for position t against `tools/oracle/mtp_ref.py`'s dump (add `--dump` to it): cosine >= 0.999, the reference argmax in the engine's top 2.

- [ ] Steps: dump the reference → failing test → implement → PASS → commit `kernels: the MTP head and the draft list (spec 8 §3.3, M1)`.

### Task 3: prefill fills the head's KV

**Files:** Modify `src/runtime/prefill/step.{h,cc}` / `engine_prefill.cc`: after the main chunk, run the head over the chunk's rows `t` with `embed(id[t+1])` ‖ `h_t` (rows with a next id inside the chunk; the chunk's last row waits for the next chunk or the first verify, spec §3.2). Use the prefill GEMM path (bf16 weights) at M = chunk rows and the flash-attention kernel for the head's layer. Test: `tests/prefill/mtp_prefill_test.cc`: the head's KV rows after prefill of 4096 ids against M = 1 decode fills of the same positions: cosine >= 0.9999 per row.

- [ ] Steps: failing test → implement → PASS; golden gates unchanged with MTP off → commit `prefill: the MTP head's KV, filled chunk by chunk (spec 8 §3.2)`.

### Task 4: the verify list, per-row GDN slots, commit

**Files:** Modify `src/kernels/decode/gdn_step.cl` (with `SPEC_SLOTS` defined: after row m, write the register state to `gdn_state_spec[m]`; unchanged when undefined), `src/kernels/CMakeLists.txt` (production variants at M = K + 1 with `SPEC_SLOTS`), `src/runtime/capture.cc` (`capture_verify()`, `capture_commit()`), `src/runtime/engine.{h,cc}`.

**Interfaces (produces):**
```cpp
// runtime::Engine, MTP on only. Before verify(): Control::draft[0..k) hold the drafts
// (from draft(), or written by the host sampler).
void verify(uint32_t k);                      // M = k + 1 rows at pos .. pos + k; argmax ids in
                                              // Control::verify_ids[0..k], logits in
                                              // verify_logits [k+1][kVocab] fp32
const float* verify_logits_device() const;
void commit(uint32_t j, uint32_t next_token); // j accepted drafts: pos += j + 1, GDN slot j live,
                                              // head hidden of row j; next_token becomes the
                                              // pending id
```
Tests: `tests/kernels/gdn_step_slots_test` (each slot equals the M = 1 state after that row, cosine >= 0.99999, max abs recorded); `tests/runtime/mtp_verify_test.cc` (checkpoint): M2 in full (row logits vs M = 1, commit for every j then 64 M = 1 greedy tokens equal the never-drafted run, Review Focus 1-3), Review Focus 4 (spec 7b C1 with MTP on), Review Focus 5 (two runs bitwise).

- [ ] Steps: failing tests → implement → PASS → commit `runtime: verify at M = K + 1 with per-row GDN slots, and commit (spec 8 §3.3-3.4, M2)`.

### Task 5: D3 and the suite

- [ ] **Step 1:** interleaved pairs, MTP off, this branch against main at the merge base: pp4096 and decode at 4k depth; bar within 1 %.
- [ ] **Step 2:** full suite green on the box.
- [ ] **Step 3:** section "Engine" appended to `docs/probe-mtp-2026-09-27.md` with the M1/M2 numbers and D3; commit `mtp: engine gates M1, M2, D3 (spec 8)`.

**Gate for the plan:** M1, M2, M5 (engine part), D3 passing, full suite green. Hand back with the numbers.
