# Spec 29c - Nanbeige4.3-33B decode: the head_dim-128 / GQA-7 probe first, then the `dn_*` kernels, `DenseEngine` on one and two stages, int8 KV

**Status (2026-11-01): PLANNED - after 29b.** Nothing built. No box queue row.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed; `WIP:` if needed); handoff file when you stop.

**Goal:** `b70-decode <Nanbeige>` produces tokens: plain decode on one card (short contexts) and on two stages (to 330k), bf16 and int8 KV, graded by spec 29's D1-D3, D5 and the first speed rows (S1). Before any engine work, P0 on the card: do K2's head_dim-128 attention kernels work at **GQA 7**, what split-K does each Nanbeige GEMV shape want, and what does the stage hand-off cost.

**Architecture:** spec 29 §3, §5, §6, §10. Precedents: spec 18b (K2's decode), `src/kernels/k2/{k2_attn,k2_kv8,k2_prep}.cl` (copied, not edited), plan 25c (Bolt's P0-first decode, its own two-card engine), plan 28a (the interleaved-pair probe method), spec 16's cut (`prep_res_fold` then the hand-off).

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md`.

## Dependencies and branch points

- 29a (judge, twin, golden sets) and 29b (descriptor, loader, plan) merged. Branch `spec29c-nanbeige-decode`.
- **Box:** idle, both cards, the GPU lock for every job; the real quantised checkpoint; the judge's golden files from 29a.
- Spec 28b (`gemm_sm`) is **not** a dependency: M = 1 decode uses today's `gemv`. Nanbeige's shapes are only *listed* for 28b here (Task 3).

## Global Constraints

- Signed commits after every step; no merge, no push; no `rm -rf`. Box tree automatic.
- **G0 first and never traded:** `tools/kernel_cmdlines` shows additions only (`dn_*` and the dense-shape `gemv` / `gemv_i8w` / `prep_*` / `embed` / `argmax` variants); every existing binary checksum-identical; the whole suite green. No shared kernel source is edited - the `dn_*` kernels are copies under `src/kernels/dense/`.
- No new CLI flag and no environment variable; test-only options live in the capture options struct.
- Timed stages: idle box, interleaved pairs after a warm-up, medians of 5 (kernels) / 3 (steps), depths 4096 / 32768 / 131072 / 240000 / 330000 where the memory allows.
- Every binary built with the GRF mode its source needs on its ocloc line, checked by the probe before a timed arm.

## Review Focus

1. **One kernel per op, bound by name; a missing binary throws at capture** (the repo's rule).
2. **GQA 7 in every attention kernel:** `NPR = 7 x M` (q, row) pairs per wave, the SLM size, the 16-lane tree, the reduce; tests at GQA 7 on a tiny model and on the real shape (56 / 8 x 128), M = 1 and M = 2..4.
3. **The norm form.** The Llama two-rounding form (a define) against the engine's one-rounding form, chosen by measurement against 29a's twin on real rows (spec 29 R9), not by assumption.
4. **int8 KV's quality at depth** (rotkv at head_dim 128 through `k2_kv8.cl`'s rotation): the golden decision cosines, the depth-to-depth KV rel-L2 flatness (the Qwen3.8 floor ~5e-3), then passkey.
5. **Stage cut = one card's order.** `--pp 2` logits are bitwise one card's at short context (D5): the cut introduces no arithmetic.
6. **Verify-width plumbing only.** The attention kernel accepts M = 1..4 (K2's limit) in this plan; M <= 8 is 29f's. Nothing in the capture assumes M = 1 beyond what 29f will widen.

---

### Task 1: P0 on the card - GQA 7, shapes, hand-off (box, before any engine)

**Files:** Create `tools/probe/probe_dense.cc` (+ CMake option), `src/kernels/dense/dn_attn.cl` and `dn_kv8.cl` as first copies of K2's with the gate stripped and `GQA = Q_HEADS / KV_HEADS` at 7.

- [ ] **Step 1: the attention at GQA 7 (Mac build, box run):** decode attention (bf16 and int8 KV) at 56 / 8 x 128, M = 1 and M = 2..4, depths 4k / 32k / 131k / 330k; against an fp64 host twin (cosine, max abs; the spec 6 / 10 bounds); bitwise repeatable; **rows at M = 2..4 equal the M = 1 step at pos + m** (spec 10's property). Time it: the KV read rate (GB/s of KV bytes) against the measured 466 GB/s of spec 10's v2 - **bar: >= 400 GB/s at 131k**, else the work-group shape is reworked (this is the number spec 29 §6's table rests on).
- [ ] **Step 2: the GEMV shapes:** qkv 7168 x 9216, o 7168 x 7168, gate-up 7168 x 44032 (layout 1, interleave-16), down 22016 x 7168 (**S in {1, 2, 4, 8}: 344 = 8 x 43; S = 16 invalid**), head 7168 x 166912 int8 (and bf16): each at S in {1, 2, 4, 8, 16 where valid} x layout {0, 1}, M = 1; the per-token sum with the layer counts (52 x ...) against the 463 GB/s basis of spec 29 §2. The winner per shape becomes the `kDenseTuning` table.
- [ ] **Step 3: the hand-off:** device 0 -> device 1 copy of 14.5 KB and 29.4 MB (a chunk), `copy` against `peer`, with the cross-device event, 1000 repeats: latency and GB/s. **Bar: <= 100 us per token-sized copy**, else the per-token cost in spec 29 §6 is re-derived. Record.
- [ ] **Step 4: the 52-launch-per-token skeleton floor:** a synthetic list of the per-layer launch count (norm-fold, qkv, prep, attn pair, o, fold-norm, gate-up, silu, down = ~9 per layer, ~470 per token + head) built from existing binaries at Nanbeige's shapes: the launch-bound floor of a step (Bolt's P0 lesson). Record.
- [ ] **Step 5: verdict** `docs/probe-nanbeige-decode-2026-11-<dd>.md`: GO for the engine if Step 1's bar and numerics pass and Step 2's token sum is within 10 % of 37.84 ms (otherwise spec 29 §6 is re-derived with the measured numbers and the operator is told before the engine is built). **Commit** the probe, the kernels' first form and the record.

### Task 2: the `dn_*` kernels (Mac + box)

**Files:** Create `src/kernels/dense/{dn_prep.cl,dn_attn.cl,dn_attn_eager.cl,dn_kv8.cl,dn_silu.cl}`, `src/kernels/dense_kernels.h`, the CMake block (option `B70_DENSE`, default on), `tests/kernels/dense_ref.h` (host twins), `tests/kernels/dense_kernels_test.cc`.

- [ ] **Step 1: failing tests** (host twins first, then the card): `dn_prep` (res-fold + norm-finish with `EPS 1e-5` and the Llama / single-rounding define), attention prep (full-128 neox RoPE through the bf16 chain, KV write, no q/k norm, no gate), decode attention + reduce (no gate), the int8 prep and attention, `dn_silu_mul` at I = 22016, embed gather at HIDDEN 7168, argmax at `VOCAB_USED` 166107 with the padded rows masked. **Step 2: implement**, copies of K2's / the Qwen path's with the Qwen-specific parts removed; `EXP2 0` (spec 10 §8). **Step 3: commit.**

### Task 3: the capture and the shape table

**Files:** Create `src/runtime/dense/{dense_capture.{h,cc},dense_buffers.{h,cc},dense_sizes.{h,cc}}`; modify `src/kernels/CMakeLists.txt` (the `kDenseTuning` cells and the `gemv` / `gemv_i8w` / `gemv_bf16` variants at the five shapes, M = 1; the **M = 2..8 / 16 shapes are not built here**).

- [ ] **Step 1: failing tests** (`dense_capture_test`: the decode list equals `decode_launches` derived from the descriptor; every named binary exists; sizes vs the plan; `check_sizes`). **Step 2: implement** the per-layer list (`res_norm` -> qkv `gemv` -> attn prep -> attn pair -> o `gemv` -> `res_norm` -> gate-up `gemv` -> `silu` -> down `gemv`; the residual add folded into the next fold as the repo does), per stage; the cut after a layer's `prep_res_fold`. **Step 3:** write Nanbeige's five shapes as a table in `docs/probe-nanbeige-decode-...md` for spec 28b's operator-confirmed scope (K, N, layer count, S) - docs only. **Step 4: commit.**

### Task 4: `DenseEngine`, one stage

**Files:** Create `src/runtime/dense/dense_engine.{h,cc}`, `src/cli/dense_decode.h`, tests `tests/runtime/dense_decode_test.cc`.

- [ ] **Step 1: failing tests** on the tiny GQA-7 model (host-reference twin, bitwise where the op list is exact): prefill-by-replay (`ingest` one id at a time) and `generate`, `pos` handling, the control block, the `--max-len` / `--kv-cache int8` options. **Step 2: implement** (ingest, generate, `--profile`, `--bench`, `--depth`). **Step 3: the card, one stage, real weights, short context:** D2 per-kernel bounds; **D3 golden** (prose / code / cjk) against 29a's judge and twin under the near-miss rule; the norm-form decision (Review Focus 3) recorded. **Step 4: commit.**

### Task 5: int8 KV on one stage

- [ ] **Step 1:** `--kv-cache int8` through `dn_kv8`: golden decision cosines and KL against the judge (bars: 1 - cos mean <= 1e-4, KL mean <= 1e-3 - the Qwen3.8 numbers' bars), the depth flatness of KV rel-L2 at 8k / 16k / 32k, int8 vs bf16 KV greedy equality outside near-ties. **Step 2: commit** the numbers in the probe doc.

### Task 6: two stages

**Files:** Create `src/runtime/dense/dense_pipeline.{h,cc}`; tests `tests/runtime/dense_pp_test.cc`.

- [ ] **Step 1: failing tests** (host fakes: a two-device fake as spec 16's tests): the cut, the hand-off protocol (copy and peer), the token's return path, the failure mode (a fence timeout names the stage). **Step 2: implement** `--pp 2`, `--pipeline-split auto|N`, `--pipeline-handoff copy|peer` (the **existing** flag names; no new ones). **Step 3 (box, both cards):** **D5** - logits bitwise equal to one stage at a short context, bf16 and int8 KV; the hand-off cost in a real step (against Task 1's number); decode speed at 4k / 32k within **2 %** of one card (spec 16's S1). **Step 4: commit.**

### Task 7: long context decode (box)

- [ ] **Step 1:** `--pp 2 --kv-cache int8 --max-len auto`: the planner's chosen length (expect ~345-355k); prefill by replay to depth is too slow at 240k+, so the depth rows use `--bench --prefill-length N` once 29d exists - **in this plan the depth rows run to 32k / 65k by replay and the deeper rows are filled by 29d**. **Step 2:** S1 rows: plain decode t/s at 4k / 32k / 65k (two stages, int8 KV) against spec 29 §6's derived 25.7 / 21.9 / 18.8; a miss > 15 % is investigated (the attention rate, the cut) before moving on. **Step 3: commit** the BENCHMARKS rows.

### Task 8: tap dumps for the drafter (box)

- [ ] **Step 1:** `b70-decode --dump-taps` is **not** a flag: the capture's test option `tap_layers = {1, 25, 49}` (as `build_verify_taps` does for Qwen) behind the dense engine's `generate` in a test binary `tools/probe/dense_taps.cc` that writes a safetensors of the three residual rows per decoded position, teacher-forced on golden / A4 / the agentic session prefix. **Step 2:** equals 29a's judge taps within the near-miss statistics. **Step 3: commit.** (Input to 29f's acceptance P0.)

### Task 9: docs

- [ ] **Step 1:** BENCHMARKS rows, `docs/19-running-models.md` (Nanbeige column: decode yes, the measured rows), spec 29 "29c as built" (P0 verdict, norm-form decision, kernels' rates, hand-off cost, the S1 rows, departures from spec 29 §6). **Step 2: commit.**

**Gate for the plan:** Mac - every host test, `tools/mac_check.sh` green, `tools/kernel_cmdlines` additions only. Box - P0's verdict GO (or the operator told); D1 / D2 / D3 pass under the near-miss rule on one stage; D5 bitwise on two stages; int8 KV bars; passkey **3/3 at 32k and 65k** on int8 KV (the 240k / 330k passkeys are 29d's, which has the prefill to reach them); S1 rows recorded; G0 bitwise for every other model (`g0.sha`, `g0.bitwise` on the golden / replay / prefill gates).

**Box stages (prose):** (1) P0 first, on one card then both: a few hours, writes the verdict that decides whether the rest starts. (2) one-stage golden on the real weights (the engine's first tokens), then int8 KV. (3) two stages: D5, the hand-off and S1. (4) long-context rows wait for 29d. Everything under the GPU lock, interleaved pairs for every comparison.
