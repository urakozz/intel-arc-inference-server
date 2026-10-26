# Spec 24d - LFM2.5-24B-A2B prefill: the short conv over a chunk with the ring carried, the grouped 64-expert MoE, flash at head_dim 64, on `l0`

> DROPPED (operator, 2026-10-27): support removed from the engine in 77d07ad; kept for the record.
>
> Reusable later (revive from git: `git show c57e08e:<path>` - c57e08e, this branch's parent on main, is the last
> commit that has all of it):
> - the MoE router: `src/kernels/lfm2moe/lfm_moe.cl` (`lfm_router`; `lfm_route` - sigmoid, the expert bias for the selection only, renormalised weights; `lfm_moe_gate_up` / `lfm_moe_down`), `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_expert.h`; twins `tests/kernels/lfm2moe_ref.h`, `tests/kernels/lfm2moe_pf_ref.h`; tests `tests/kernels/lfm2moe_ref_test.cc`, `tests/kernels/lfm2moe_kernels_test.cc`, `tests/kernels/lfm2moe_pf_kernels_test.cc`
> - the pythonic tool-call parser: `src/server/toolcall_lfm2.h`, `src/server/toolcall_lfm2.cc`, `tests/server/toolcall_lfm2_test.cc`; `tools/toolcall/score.py --format pythonic`, `tools/toolcall/lfm2moe_rules.cc`
> - the rest: the short conv `src/kernels/lfm2moe/lfm_conv.cl` / `lfm_pf_conv.cl`, head_dim-64 attention `src/kernels/lfm2moe/lfm_attn.cl` / `lfm_pf_attn.cl`, `src/model/lfm2moe.h`, `src/model/lfm2moe.cc`, `src/loader/lfm2moe_*`, `src/runtime/lfm2moe/` (`lfm_prep.cl` still live in `src/kernels/bolt/`)

**Status (2026-10-20): built on the Mac (branch `spec24d-lfm2moe-prefill`; spec 24 §15 "24d as built"); nothing has run on a card.** Box queue row **52** (rows 42-47 are spec 23's). Built blind on the
Mac after 24c; every kernel's first compile and run is the box's.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `Lfm2MoeEngine::prefill(ids)` - the prompt in chunks of up to 2048 rows on the `l0` backend (bf16 DPAS GEMMs over dequantised int4 slabs, spec 20d's path): the conv over a chunk with its 2-column `Bx` history seeded from the ring and the chunk's last two `Bx` written back, the 64-expert grouped MoE (sort, gather, batched dequant, `pf_moe_gemm` SiLU, down, the rank-order combine and the residual fold), flash attention at head_dim 64 / GQA 4 with its eager twin, the dense layers, the head on the last row; `b70-decode <lfm2moe> --prefill` / `--bench --prefill-length N`; one card, then two (`--pp 2`, one hand-off per chunk); F3 and F4 on prefill; the prefill rate.

**Architecture:** spec 24 §3 (prefill estimate), §4.1, §4.2, §4.7, §8 24d; spec 20d's walk (`src/runtime/kolibri/kolibri_prefill.cc`) is the template. Per chunk of C rows, per layer:

```text
conv layer:       lfm_res_norm (M C) · in_proj: dequant slab(s) + pf_gemm · lfm_pf_conv_seed · lfm_pf_conv · out_proj: slab + pf_gemm ·
                  lfm_res_norm (M C)
attention layer:  lfm_res_norm _NF (M C) · q‖k‖v: slab + pf_gemm · lfm_attn_prep (M C: norms, RoPE, KV rows pos..pos+C-1) ·
                  lfm_pf_attn (flash, HD 64, causal over the cache) · o: slab + pf_gemm · lfm_res_norm (M C)
MoE (layers 2-39): lfm_router (M C) · lfm_route (grid (1, C)) · lfm_pf_sort · lfm_pf_gather · B x (lfm_pf_dequant_gu + pf_moe_gemm SILU) ·
                  B' x (lfm_pf_dequant_dn + pf_moe_gemm) · lfm_pf_combine (rank order, one rounding, + the residual fold)
dense (0-1):      gate‖up: slabs + pf_gemm with the SiLU epilogue (I 11776) · down: slabs + pf_gemm · (the fold in the next norm)
last row only:    lfm_res_norm _NF + embedding_norm · head (int8 | bf16) · argmax
```

**The conv over a chunk (spec 24 §4.1, pf_gdn_conv.cl's discipline):** a causal conv has no recurrence - output row t needs `Bx[t-2..t]` only, and `Bx` is a function of in_proj's row alone - so `lfm_pf_conv` is a grid over (channel tiles, row tiles) where each work-item recomputes `Bx` for its rows' two predecessors from in_proj's output (`rne(B · x)`, the same rounding, so bitwise) instead of carrying it. Only rows 0 and 1 of the chunk need the previous chunk: **`lfm_pf_conv_seed`**, a prior launch, copies the ring slots of positions p0-2 and p0-1 into a flat `seed` [2][2048] (0 below position 0) - `pf_gdn_conv.cl:12-27`'s argument: the read happened in a previous launch of an in-order list, so no work-group of `lfm_pf_conv` reads a slot another one writes, whatever C is (at C = 2048 the slots `(p0 + C - 2) % 16` and `(p0 - 2) % 16` coincide - writing the ring in the same launch that reads it would race). `lfm_pf_conv` writes `Bx` of the chunk's last min(C, 2) rows into their ring slots; decode then continues from the ring exactly as after decode steps.

**The MoE dequant is the chunk's dominant cost (estimated):** at C = 2048 almost every expert is hit (8192 pairs over 64 experts), so each MoE layer dequantises ~1.21 GB of bf16 blocks (64 x 18.9 MB) and reads them back: ~2.7 GB of traffic a layer, ~5.5 ms at ~500 GB/s, ~0.21 s a chunk over 38 layers, against ~0.1 s of DPAS work (2048 x 4.9 GFLOP at ~100 TFLOP/s): **~6,000-7,000 t/s at pp 4096, ESTIMATED**. An expert with no rows in the chunk is skipped (Kolibri's rule). Fusing the dequant into the grouped GEMM (spec 2.1's own-GEMM direction) or int8 prefill (spec 24 §11, out of scope) are the levers after 24d's rate is measured.

**Tech Stack:** C++17, OpenCL C (ocloc, DPAS through `pf_gemm` / `pf_moe_gemm`), Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` (§2.5, §3, §4.1, §4.2, §4.7, §7 F3, F4, §8 24d, §10 decisions 10, 12, 13). Precedents: plan 20d (`2026-10-06-spec20d-kolibri-prefill.md`) task for task, plan 21d, `src/kernels/kolibri/{kol_pf_moe,kol_pf_attn,kol_pf_linear}.cl`, `src/kernels/prefill/{pf_moe_gemm,pf_gemm,pf_dequant_slab,pf_gdn_conv}.cl`, `src/runtime/kolibri/{kolibri_prefill,kolibri_prefill_engine}.cc`, `tests/kernels/kolibri_pf_ref.h`, `tests/runtime/kolibri_prefill_test.cc`, `tests/golden/kolibri_golden_test.cc` (its prefill mode), `src/runtime/prefill_chunks.h`.

## Dependencies and branch points

- **Plan 24c merged** (its kernels at M = 1, the engine, the decode gates; 24c's `lfm_res_norm`, `lfm_router`, `lfm_route`, `lfm_attn_prep`, `lfm_silu_mul` are compiled here again at M = C with a `_PF` suffix - their code is M-generic: one work-group set per row). **24q's RTN export and its golden sets** for the real-weight gates.
- **`l0` only:** `--prefill-backend l0-int8` / `sycl-tla` are refused for this model naming spec 24 §11 (int8 prefill is a later probe; `sycl-tla` has no hd64 path here).
- **Decision 10:** `--pp 2` prefill hands `h` (C x 2048 bf16, 8 MB at C 2048) across once per chunk (spec 16c's sequential walk, Kolibri's).
- **Decision 12:** family copies (`lfm_pf_*`); `pf_moe_gemm.cl`, `pf_gemm.cl`, `pf_dequant_slab.cl` reused unchanged from new CMake lines (K 2048 N 3072 SILU; K 1536 N 2048; the dense and linear slabs).
- **Prefill must follow each linear's layout (24b review fix, I1).** 24b's per-`LfmLinear` `{S, layout}` is PROVISIONAL (24c sweeps it) and not necessarily layout 1 for every linear this plan's slabs dequantise: `in_proj` / `out_proj` / the dense down-projection / the routed experts are layout 1, the fused `q||k||v` and dense `gate||up` are layout 0. `pf_dequant_slab.cl`'s existing layout-1 branch (Kolibri's / Muse's convention) covers the former group unchanged; the `q||k||v` and dense `gate||up` slabs need `pf_dequant_slab`'s layout-0 branch (the GPTQ-native `.qweight`/`.scales` pair), not a second kernel - confirm against `model::Lfm2MoeDesc::linears(l)[...].shape.layout` per linear, the same source 24c's decode GEMVs read, rather than assuming layout 1 because every OTHER model this prefill code was copied from happens to use it everywhere.

## Global Constraints

- Branch `spec24d-lfm2moe-prefill` from main (after 24c); box tree automatic; `tools/box.env` copied if missing, never committed, never printed.
- **F0, nothing moves:** existing binaries byte-identical (G0), `kernel_cmdlines` additions only, no existing `.cl` edited.
- **Semantics are 24a's:** the conv's taps oldest first with the zero history, the router chain, the rank-order combine, the norm's two roundings - the prefill kernels and `tests/kernels/lfm2moe_pf_ref.h` edited together; prefill rows the decode chain applied per row wherever the arithmetic allows.
- **Determinism:** no atomic; the sort and the combine in fixed order (Kolibri's sorted layout: expert-major, ascending token, padded to whole TM tiles; tmax a function of C alone, so the host reads no count).
- Every number measured, or marked derived / estimated / proposed. Box: GPU commands under `flock ~/b70-gpu.lock`, detached, polled, interleaved pairs after a warm-up, median of 3, `uptime` and idle grade; `-j44`.
- Mac checks: `tools/mac_check.sh --base main --kernels`. No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **The conv's chunk boundary.** A prompt split at every offset 1..5 and at 64, 2047, 2048, 2049 gives the conv state, the KV and the next token's logits bitwise one chunk's (the conv has no chunk-dependent order); a chunk of C = 1 and C = 2 (the seed covers all taps; the ring write covers rows 0 and 1); a prompt whose first chunk starts at position 0 reads zeros from the seed. Task 1's reference and Task 2's kernel test cover each; Task 3's gate F4 runs them through the engine.
2. **Prefill equals decode's fill where the arithmetic is the same, and is held to bars where it is not.** The per-row chains (the conv, the norm, the route, the combine) are the decode kernels' functions - tested bitwise at kernel level on identical inputs (Task 2). End to end the inputs differ: every linear's output comes from the DPAS GEMM's K order instead of the GEMV's, so the conv rings, the KV, the routes and the `h` taps after a prefilled prompt are held to cosine bars against decode's fill (spec 20d's: every row of the first attention layer ≥ 0.999, all rows' median ≥ 0.9998, p01 ≥ 0.99 - PROPOSED, printed), the routes to set equality except near-ties (`B70_LFM_TIE_TOL`), and the tokens tie-aware.
3. **The sort at 64 experts, top-4, uchar ids.** Every (token, slot) pair placed exactly once, expert-major, ascending token, padded to TM 32; tmax(C) = floor((4C + 64 x 31) / 32) + ceil(C / 32) (Kolibri's formula at this model's numbers, no shared block); an expert with no rows skipped by the dequant; a chunk where one expert takes every row (crafted router) does not overflow.
4. **Flash at head_dim 64 against the cache, causally.** Rows attend to `[0, pos + r]` across chunk boundaries (keys from earlier chunks in the cache, this chunk's written by `lfm_attn_prep` before `lfm_pf_attn` reads them); against fp64 at cosine ≥ 0.99999 (spec 6 K1) at C = 1, 63, 64, 2048 and with 4096 / 131071 earlier keys; the eager twin against the reference's bf16 chain.
5. **Only the last row reaches the head.** The head and argmax run on row C - 1 of the last chunk only; intermediate chunks end after layer 39's fold; the logits read back after `prefill` are the last prompt row's, equal (tie-aware) to decode's at the same position.

---

### Task 1: the prefill host reference and fixture (no card)

**Files:**
- Create: `tests/kernels/lfm2moe_pf_ref.h`, `tests/kernels/lfm2moe_pf_ref_test.cc`
- Modify: `tools/oracle/lfm2moe_fixture.py` (prefill cases), `tests/kernels/lfm2moe_fixture.h` (regenerated), `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `lfmpfref`, reusing `lfmref`'s per-row chains): `conv_seed(ring, p0) -> seed[2][2048]`, `conv_chunk(inproj_bf16 [C][6144], seed, taps, p0) -> {y [C][2048], ring writes}`, `pf_sort(route rows [C], C) -> {hdr, tiles, row_tok, pair_row}` (Review Focus 3's layout), `pf_combine(route, pair_row, y_rows, h) -> h'`, `flash_fp64(q, k, v, pos0, C)`.
- [ ] **Step 1: the failing test** `lfm2moe_pf_ref_test.cc`: `conv_chunk` over a split prompt equals `lfmref::conv_step` row by row (Review Focus 1, every offset 1..5, 64, 2047-2049, C 1 / 2); `pf_sort` places every pair once with the padding rule, the tmax formula at C 1, 31, 32, 2048, the one-expert-takes-all case; `pf_combine` equals `lfmref::moe`'s combine per row; the fixture's prefill rows (the 600-id prompt's conv state and last-row logits from 24a's reference at chunk 64) bitwise where per-row.
- [ ] **Step 2: register, FAIL; implement; PASS. Commit** `git commit -S -m "tests: LFM2.5-24B-A2B prefill host reference - the conv over a chunk with its seed, the 64-expert sort and combine, flash in fp64 (spec 24d)"`.

### Task 2: the prefill kernels

**Files:**
- Create: `src/kernels/lfm2moe/lfm_pf_conv.cl`, `src/kernels/lfm2moe/lfm_pf_moe.cl`, `src/kernels/lfm2moe/lfm_pf_attn.cl`, `tests/kernels/lfm2moe_pf_kernels_test.cc`, `tests/kernels/lfm2moe_pf_variant_names_test.cc`
- Modify: `src/kernels/lfm2moe_kernels.h` (prefill names), `src/kernels/CMakeLists.txt` (a block `# ==== Spec 24d: LFM2.5-24B-A2B prefill ==== (begin) / (end)`), `tools/mac/clrun/lfm2moe_run.cc` (the plain-OpenCL prefill kernels), `tests/CMakeLists.txt`

**Interfaces:**
- Produces (`kernels::lfm2moe`): `pf_conv_seed_variant()` (`"lfm_pf_conv_seed"`), `pf_conv_variant(unsigned C)` (`"lfm_pf_conv_C2048"`, C the compiled maximum; the live count from Control), `pf_moe_variant()` (`"lfm_pf_moe_E64_T4_D2048_I1536_TM32"`: `lfm_pf_sort`, `lfm_pf_gather`, `lfm_pf_dequant_gu`, `lfm_pf_dequant_dn`, `lfm_pf_combine`), `pf_attn_variant(bool eager)` (`"lfm_pf_attn_Q32KV8_HPW4_RPW16_KT64"` | `"..._EAGER"`; HPW / RPW / KT PROVISIONAL, Task 5 sweeps), the `_PF` lines of 24c's kernels at M 2048, the reused `pf_moe_gemm_K2048_N3072_SILU`, `pf_moe_gemm_K1536_N2048`, the linear slabs (`pf_dequant_slab` at in_proj / out_proj / q‖k‖v / o / dense shapes) and `pf_gemm_T0`, `pf_gemm` with the SiLU epilogue at N 23552.

```text
lfm_pf_conv.cl (plain OpenCL C)
  lfm_pf_conv_seed(ctrl, ring, seed)                 grid (2048 / 256), WG 256: seed[j][c] = p0 - 2 + j < 0 ? 0 : ring[(p0 - 2 + j) % 16][c]
  lfm_pf_conv(ctrl, inproj, seed, taps, ring, y)     grid (2048 / 256, ceil(C / 16)), WG 256: rows r0..r0+15 of channel c:
      bx(t) = t < 0 ? seed[t + 2][c] : rne(B(t) · x(t));  y[t][c] = rne(C(t) · rne(fma chain over bx(t-2), bx(t-1), bx(t)))
      rows C-2, C-1 (those >= 0): ring[(p0 + t) % 16][c] = bx(t)
lfm_pf_moe.cl (kol_pf_moe.cl's structure, minus the shared block, ids as uchar, the combine in RANK order with one rounding,
               FP_CONTRACT OFF, + the residual fold h = rne(h + y))
lfm_pf_attn.cl (kol_pf_attn.cl's full path at HD 64: 4 dim-atoms of O, K-dim 64 = 4 DPAS k-steps; scale 0.125; EAGER twin)
```
- [ ] **Step 1: the failing host test** `lfm2moe_pf_variant_names_test` (every name the walk binds is a target of the two blocks). FAIL.
- [ ] **Step 2: write the sources, the names, the CMake block;** PASS; `kernel_cmdlines` additions only; `lfm2moe_run` exact on the plain-OpenCL ones (seed, conv, sort, gather, combine) on the Mac GPU.
- [ ] **Step 3: the card test** `lfm2moe_pf_kernels_test` (label `lfm2moe`): every binary at the real shapes against `lfm2moe_pf_ref.h` - bitwise for the seed, the conv (Review Focus 1), the sort / gather / combine (Review Focus 3), the `_PF` norm / route / prep lines; `pf_moe_gemm` at both shapes against the row-wise reference (its row independence, spec 15d); flash against fp64 (Review Focus 4), eager against the bf16 chain.
- [ ] **Step 4: Mac gate; commit** `git commit -S -m "kernels: LFM2.5-24B-A2B prefill - the conv over a chunk seeded from the ring, the 64-expert sort / dequant / rank-order combine, flash at head_dim 64 (spec 24d)"`.

### Task 3: the prefill walk on one card, the CLI, F3 / F4 on prefill

**Files:**
- Create: `src/runtime/lfm2moe/lfm2moe_prefill.{h,cc}`, `src/runtime/lfm2moe/lfm2moe_prefill_engine.cc`, `tests/golden/lfm2moe_prefill_gate_test.cc`, `tests/runtime/lfm2moe_prefill_test.cc`
- Modify: `src/runtime/lfm2moe/lfm2moe_sizes.{h,cc}` (prefill scratch, `prefill_chunk_launches`), `src/runtime/lfm2moe/lfm2moe_engine.h` (`prefill`), `tests/runtime/lfm2moe_plan_test.cc`, `src/cli/lfm2moe_decode.h` (`--prefill`, `--prefill-length`, `--prefill-chunk`; `--prefill-backend l0` only), `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `void Lfm2MoeEngine::prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 2048)` (the prompt in chunks; afterwards `pending()` is the argmax of the last row and decode continues); `runtime::lfm2moe::PrefillSizes prefill_sizes(desc, chunk)` (the slabs, the sorted rows, the expert batches - the planner's term, 24b's `Plan::prefill_scratch`); `size_t prefill_chunk_launches(desc, chunk, batches_gu, batches_dn)` (the formula, asserted by the plan test and by the walk).
- [ ] **Step 1: failing host tests:** `lfm2moe_plan_test` gains the prefill scratch at C 2048 (the dequant batch size PROPOSED 512 MiB as Kolibri's, so B = 2 gate‖up batches and 1 down batch a layer at 64 experts; derived) and the launch formula; the plan at 131072 with the prefill scratch still fits one card.
- [ ] **Step 2: implement the walk** (Kolibri's `kolibri_prefill.cc` shape: one regular command list per chunk, recorded per chunk on `l0`; the expert batches; the last-row head).
- [ ] **Step 3: the card gates** (label `checkpoint;golden;lfm2moe`, SKIP 77 without data):
  - `lfm2moe_prefill_test <ckpt> <ids>` (F4): prefill twice bitwise; chunk 2048 vs chunks of 64 vs splits at 1..5 / 2047 / 2049 bitwise (Review Focus 1); prefill then 32 decode steps against the same prompt ingested by decode then 32 steps: the tokens tie-aware, the conv rings / KV / `h` taps within Review Focus 2's bars, the routes equal except near-ties.
  - `lfm2moe_prefill_gate_test <ckpt> <oracle dir> [eager]` (F3): tokens tie-aware against the golden sets after `prefill`; the per-layer `h` taps (bars of Review Focus 2); routes against `route.*.L*` with `B70_LFM_TIE_TOL`. Registered on the synthetics, the RTN export (`short`, `4k`, `agentic`, opt-in `32k`), `ours` when it exists; `_eager` twins.
- [ ] **Step 4: Mac gate; commit** `git commit -S -m "runtime: LFM2.5-24B-A2B prefill on one card - chunked walk, the conv ring carried, F3 / F4 on prefill (spec 24d)"`.

### Task 4: two cards (`--pp 2` prefill; decision 10)

**Files:**
- Modify: `src/runtime/lfm2moe/lfm2moe_prefill.cc`, `src/runtime/lfm2moe/lfm2moe_engine.cc`, `tests/runtime/lfm2moe_pp_test.cc` (24c's: prefill cases)

- [ ] **Step 1: the failing test:** `lfm2moe_pp_test` gains prefill on two cards (synthetic split 2, RTN split 20; `copy` and `peer`) bitwise one card's (KV, conv rings, last-row logits) at chunks 2048 and 64.
- [ ] **Step 2: implement** (device 0 runs every chunk's first part and hands `h` [C][2048] across once per chunk; device 1 the rest and the head; the sequential walk of spec 16c / 20d). **Commit** `git commit -S -m "runtime: LFM2.5-24B-A2B prefill across two cards - one hand-off per chunk, bitwise one card (spec 24d)"`.

### Task 5: docs, the queue row, the prefill rate

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-spec24-lfm2-moe-design.md` ("24d as built"), this plan's status line, `docs/superpowers/plans/box-validation-queue.md` (row 52), `tools/box_validate/stages.sh` (`row 52`), `docs/19-running-models.md` (the LFM2 section's prefill commands), `docs/BENCHMARKS.md` (Task 5 Step 3's rows)

- [ ] **Step 1:** "24d as built" (the walk, the launch formula, the batches, every departure).
- [ ] **Step 2: the queue row 52** (`test_box_validate.py` passes; `--dry-run --only r52` prints it):

```text
row 52 "spec 24d - LFM2.5-24B-A2B prefill: the conv over a chunk with the ring carried, the grouped 64-expert MoE, flash at head_dim 64 (plan 24d)"
stage r52.k0     52 default cpu -                            g0.sha,g0.bitwise,g0.suite "K0: pre-existing binaries identical; kernel_cmdlines +N / -0 / ~0"
stage r52.host   52 default cpu -                            -        "host: lfm2moe_pf_ref_test, lfm2moe_pf_variant_names_test, lfm2moe_plan_test (prefill terms)"
stage r52.k1     52 default gpu -                            r52.k0   "lfm2moe_pf_kernels_test: seed / conv / sort / combine bitwise, pf_moe_gemm row-wise, flash vs fp64 0.99999"
stage r52.f4     52 default gpu lfm2moe_synth,lfm2moe_rtn    r52.k1   "lfm2moe_prefill_test: twice bitwise, chunk splits bitwise, prefill vs decode-only fill within the bars"
stage r52.golden 52 default gpu lfm2moe_rtn,oracle_lfm2moe_rtn r52.f4 "lfm2moe_prefill_gate_test on the synthetics and the RTN export (_eager twins); ours opt-in"
stage r52.pp     52 default gpu lfm2moe_synth,lfm2moe_rtn    r52.f4   "two cards: prefill bitwise one card (copy, peer)"
stage r52.speed  52 optin   gpu lfm2moe_rtn                  r52.f4   "--bench --prefill-length 4096 / 32768: the rate, B70_PREFILL_PROFILE phases, the attention HPW / RPW / KT sweep"
rownote 52 "Bars PROPOSED (spec 20d's family; the tests print the distributions): prefill vs decode's fill - every row of the first attention layer >= 0.999, all rows' median >= 0.9998, p01 >= 0.99; flash vs fp64 >= 0.99999. The conv, the routes and the combine are bitwise per row."
```
- [ ] **Step 3 (box): the rate** - `b70-decode <lfm2moe> --bench --prefill-length 4096` and `32768`, `B70_PREFILL_PROFILE=1` phase table (the dequant share against spec 24 §3 / this plan's estimate), the flash tile sweep; `docs/BENCHMARKS.md` prefill rows. **Commit** `git commit -S -m "docs: spec 24 (24d as built), box queue row 52 - LFM2.5-24B-A2B prefill, the rate"`.

**Gate for the plan:** Mac - host tests, `lfm2moe_run` exact on the plain kernels, cmdlines additions only. Box - F0; every prefill kernel against its reference; F4 (twice bitwise, chunk splits bitwise; prefill against decode's fill within the bars); F3 on the synthetics and the RTN export; two cards bitwise one; the rate recorded with its phase table.
