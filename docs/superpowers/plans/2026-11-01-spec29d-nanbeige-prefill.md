# Spec 29d - Nanbeige4.3-33B prefill: flash attention at head_dim 128 / GQA 7, the `l0` GEMM shapes, chunked prefill with the int8 KV write, pipelined two-stage prefill

**Status (2026-11-01): PLANNED - after 29c.** Nothing built. No box queue row.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed; `WIP:` if needed); handoff file when you stop.

**Goal:** `DenseEngine::prefill(ids, chunk)` on the `l0` backend (the default), one stage and two (pipelined by chunk), with `pf_flash`-style attention at head_dim 128 and GQA 7, writing bf16 or int8 KV; the long-context rows that 29c could not reach (decode at 131k / 240k / 330k, passkey at 240k and 330k); the measured prefill table that replaces spec 29 §9's derived one.

**Architecture:** spec 29 §3 (prefill rows), §5, §9. Precedents: plan 18c (K2's prefill, `k2_pf_attn.cl`, `k2_pf_linear.cl`), spec 6 / 11 (flash prefill and its levers), spec 2.1 (the `l0` backend: dequant slabs + bf16 DPAS GEMM), spec 16c (`pipeline_prefill_plan.h`: the chunk order, two landing slots, the back-pressure rule - rewritten for the dense stages), spec 12 (the int8 KV write in prefill), plan 25d.

**Spec:** `docs/superpowers/specs/2026-11-01-spec29-nanbeige-design.md`.

## Dependencies and branch points

- 29c merged (decode, `dn_*` kernels, the engine, the hand-off). Branch `spec29d-nanbeige-prefill`.
- **Box:** both cards, idle; the GPU lock.
- Plan 5d (chunk 4096, being built) is **not** a dependency: the engine's chunk is 2048 (`PrefillScratch::kC`, ruling A13); the dense prefill takes the same constant and is retried at 4096 only if 5d has merged.

## Global Constraints

- Signed commits after every step; no merge, no push; no `rm -rf`.
- **G0:** `tools/kernel_cmdlines` additions only (`dn_pf_*` and the dense-shape prefill GEMM cells); every existing binary checksum-identical; **`l0-int8` is refused by name for the dense family** (spec 29 §3: K = 22016 is not a multiple of 1024) - a test names the reason.
- Prefill backend default `l0`; the dense family builds **no composed / eager prefill attention** (flash only; the reference attention is the judge's). The eager *decode* kernel `dn_attn_eager` (29c) is a diagnostic twin, never a default.
- No new flag / env var. Timed stages: idle box, interleaved pairs after warm-up.

## Review Focus

1. **HPW 1 or 7.** The flash work-group's heads-per-group divides GQA 7; the probe picks the faster of the two (and the RPW / KT tile) with the tile's register use checked (GRF mode on the ocloc line, probe checks the built binary).
2. **Attention at depth is the cost** (spec 29 §9): 745,472 N^2 FLOP causal over 52 layers; the kernel's TFLOP/s at 32k / 131k / 240k is the number to record, with the 28.3 TFLOP/s of the head_dim-256 kernel as the baseline to compare, not to assume.
3. **Chunk order and KV write.** Chunk c reads the KV of chunks < c, written in bf16 or int8 (rotkv) by the prefill; bitwise: prefill then decode equals replay-by-decode within the near-miss rule, and int8 prefill KV equals what decode's int8 prep would have written for the same rows (the quantiser is one function).
4. **Pipelining is exact:** two-stage prefill logits bitwise equal one-stage's at a context both fit, whatever the chunk-to-stage interleaving; the back-pressure rule's host tests (spec 16c's) are reproduced on the dense stages.
5. **Int32 hygiene at 330k:** no index product narrows to uint32; the `kv_stride` of the dense layout is `size_t`; a test at 355k positions on a synthetic layout checks every offset.

---

### Task 1: the flash kernel probe (Mac build, box run)

**Files:** Create `src/kernels/dense/dn_pf_attn.cl` (copy of `k2_pf_attn.cl`, gate removed, GQA 7), `tools/probe/probe_dense_pf.cc`, `docs/probe-nanbeige-prefill-2026-11-<dd>.md`.

- [ ] **Step 1:** the kernel at HPW 1 and 7, RPW 8 / 16, KT 32 / 64, GQA 7, hd 128, chunk 2048 over depths 0 / 32k / 131k / 240k / 330k; against an fp64 host twin (K1's cosine >= 0.999998); TFLOP/s on the causal count; the best tile per depth band. **Step 2: record** `attn_flash` TFLOP/s and the derived cold-prefill table (spec 29 §9's formula with the measured rate), **before** the engine. **Step 3: commit.**

### Task 2: prefill GEMMs and the per-layer walk

**Files:** Create `src/kernels/dense/dn_pf_prep.cl` (norm / RoPE / KV write for a chunk, bf16 and int8, the decode prep at M = kPfC as K2 does), `src/runtime/dense/dense_prefill.{h,cc}`, `dense_prefill_engine.cc`, `tests/kernels/dense_pf_kernels_test.cc`, `tests/runtime/dense_prefill_test.cc`.

- [ ] **Step 1: failing tests:** the `l0` slab GEMMs at the five shapes (M = 2048): fp64 twin, tile divisibility (N 44032 = 43 x 1024, K 22016 = 86 x 256), the slab sizes in the plan's scratch; chunk prep vs decode prep per row. **Step 2: implement** the walk (embed -> 52 x [norm, qkv GEMM, prep + KV write, flash, o GEMM, fold-norm, gate-up GEMM + silu epilogue, down GEMM] -> head on the last row). **Step 3: the card, one stage:** D3 prefill-side (prefill then decode) against the judge under the near-miss rule; prefill = replay equality; int8 KV equality of the quantiser. **Step 4: commit.**

### Task 3: speed on one stage

- [ ] **Step 1:** pp4096 / pp32768 (one stage, int8 and bf16 KV, `--max-len` at the single-card limit): t/s against spec 29 §9's derived 1,210 / ~630; the linear rate (`attn_flash` off in a profiled run) against ~1,400 t/s (the `l0` basis: Qwen3.8's 1,670.72 t/s). **Step 2:** the per-phase profile (L0 GPU ms per kernel family) at pp4096: where the linears' ~86 TFLOP/s effective goes; **no tuning in this plan** except a fix for a kernel that is clearly broken (> 2x off its twin's roofline), recorded for the operator. **Step 3: commit.**

### Task 4: two-stage pipelined prefill

**Files:** Create `src/runtime/dense/dense_pipeline_prefill_plan.{h,cc}`; tests `tests/runtime/dense_pp_prefill_test.cc` (the order checker's host tests, the single-step drops and named reorderings of spec 16c).

- [ ] **Step 1: failing tests; Step 2: implement** (chunk j appended to a device only after the host has seen both finish j - 2; two landing slots and events; the chunk's hand-off = the folded residual rows + norm sums, 29.4 MB at 2048 x 7168 bf16). **Step 3 (both cards):** bitwise equality with one stage at a context both fit (D5, prefill side), then the speed-up at pp32768 / 65536 (spec 16's derived 1.88x; the bar **>= 1.6x at 65536**, else the stage balance (the second card is ~3.4 % slower on prefill, box fact) and the split are re-examined). **Step 4: commit.**

### Task 5: long contexts (box)

- [ ] **Step 1:** prefill to 131k, 240k and 330k (int8 KV, `--pp 2`, planner-chosen max-len; detached, resumable is not possible mid-prefill, so each run is one job, ~14-26 min predicted at 240-330k): times against spec 29 §9's derived table, per-chunk times (to see the attention's O(N) growth per chunk), the memory line, no DEVICE_LOST. **Step 2: D4:** passkey at 5 / 50 / 95 % for 240k and 330k ids (int8 KV), 3/3 each (`tools/probe/passkey.sh`'s dense mode, ids from the dense tokenizer); engine vs engine bf16-vs-int8 KV and one-stage-vs-two logits at 32k / 128k (cosine distance recorded; the bf16 KV run at 128k needs the ~175k two-stage plan). **Step 3: decode at depth** (`--bench --prefill-length N`, tg 256, median of 3): 131k, 240k, 330k against spec 29 §6's derived 14.6 / 10.6 / 8.7 t/s (S1). A decode row more than 15 % under the derived number is investigated at the attention (the KV read rate) before 29e. **Step 4: commit** BENCHMARKS rows.

### Task 6: docs

- [ ] **Step 1:** spec 29 "29d as built" (the kernel's TFLOP/s, the measured prefill and decode tables beside the derived ones, the pipelining factor, the cold-prefill times and the operator decision 4 restated with numbers); `docs/19-running-models.md`; BENCHMARKS. **Step 2: commit.**

**Gate for the plan:** Mac - host tests, `tools/box_check.sh`, `kernel_cmdlines` additions only. Box - D3 prefill-side under the near-miss rule; D5 prefill bitwise across stage counts; passkey 3/3 at 240k and 330k on int8 KV; the 330k run completes under the planner's line without a device fault; the pipelined speed-up bar; the measured tables recorded; G0 bitwise for other models.

**Box stages (prose):** (1) the flash probe, minutes to hours, first; (2) one-stage prefill golden and speed; (3) two-stage bitwise and speed; (4) the long runs - 131k first, then 240k, then 330k, each a single job of minutes to half an hour, sequential, the GPU lock held, idle-box check at dispatch; (5) the passkeys and the depth decode rows. A DEVICE_LOST during a long run is investigated before any retry (fix round A's lesson: big uploads and host-memory pressure).
