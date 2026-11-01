# Spec 27c - Flint decode: the launch-floor P0 first, then the sink attention, the fp32 router, the 2x-wide shared expert and the dense-first walk on one card

**Status (2026-11-01): plan written, nothing built.** Needs 27a (reference, taps) and 27b (descriptor, loader, plan).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** measure first where a Flint decode step's time goes (P0, spec 25c's method), then decode the whole model through a replayed Level Zero list: `fl_*` kernels and the generic `sink_attn_*` / `route_sigmoid_renorm` kernels of spec 27 section 2, the capture with the dense-first and the non-periodic layer table, `FlintEngine`, `b70-decode flint`, `--profile`, a fused-arm follow-up only if P0 says launches matter; F2 on every kernel, F3 / F4 on decode; the speed rows at depth (bf16 KV here; int8 is 27f).

**Architecture:** spec 27 sections 2, 3, 7.1, 9. Template: Kolibri (`src/runtime/kolibri/kolibri_{capture,engine,buffers,sizes}.*`, `src/kernels/kolibri/kol_*.cl`, `src/kernels/kolibri_kernels.h`) and plan 25c (`2026-10-20-spec25c-bolt-decode.md`). Patterns copied under generic names (never bound or included): the sink term from LiLiCorr's reduce, `git show 99adbb0^:src/kernels/dflash/ll_attn.cl` line 170 (`l = l + exp(sink[h] - m)` once after the block / key merge) - the operator is restoring LiLiCorr on branch `bolt-spec-restore` (2026-11-01), so copy from the restored file if it is on main by then, from git otherwise, and depend on neither; the route from Bolt's `nb_route` (`src/kernels/bolt/nb_moe.cl`, on main: fp32 logits never rounded, fp32 sigmoid, renormalised) **without** its selection bias, `+ 1e-20` and `x 5.0` (`nb_moe.cl:157`), top-6 instead of its hard top-9. LFM2.5's dropped `lfm_moe.cl` is not needed.

**Kernel list (all new files; no existing `.cl` edited - G0):**

| kernel | from | what changes |
|---|---|---|
| (reused) norm | `prep_res_fold` + `k2_norm_finish` at `K=2048 NORM_GROUPS=1` | already `rne(x * rstd * w)`, eps 1e-6 - Flint's norm; new variant names only. A fused fold + norm (`nb_res_norm`'s pattern) is a P0-gated follow-up, not a default |
| `fl_attn_prep` | `kol_attn_prep` | q / k head norm one rounding, then neox RoPE `rne(rne(y*c) + rne(rot*s))` on sliding layers only, ring write (sliding) / linear write (full); `qkv` split 4096 / 512 / 512 |
| `sink_attn_decode`, `sink_attn_reduce` | `kol_attn.cl`, `ll_attn.cl` | GQA 8, hd 128, WINDOW 513 ring or linear NoPE; sink per head per layer added once in the reduce, also when the window skips a block; exp2 build: sink x log2 e |
| `sink_attn_eager_*` (diagnostic) | `kol_attn_eager.cl` | fp32 scores, softmax over [scores, sink], bf16 P, bf16 PV: the reference's chain |
| `route_sigmoid_renorm` | `nb_route` (Bolt) | fp32 logits in, fp32 sigmoid, top-6 on scores (the tie order 27a records from the reference), `w = s / sum` in 27a's pinned order, slots sorted by ascending id, no bias argument, no epsilon, no scale, no rounding |
| `fl_moe_gate_up`, `fl_moe_down` | `k2_moe.cl` (its down's combine, `k2_moe.cl:317`, is already Flint's chain when w is unrounded fp32) | `E256 T6 D2048 I512`; combine `rne(rne(down) * w)` with w fp32 unrounded, bf16 sequential adds ascending id; no shared slot (shared I differs): the routed sum to its own buffer, then the fold `rne(h + rne(r + sh))` (a `prep_res_fold` two-source variant or a few lines in `fl_moe_down`'s successor; twin-tested) |
| shared expert (reused) | `gemv.cl` gate‖up (N 2048, K 2048), `k2_silu_mul` (`SILU_I=1024`), `gemv.cl` down (N 2048, K 1024) | Bolt's dense-GEMV shared path (`bolt_capture.cc:355-388`); no new kernel. Option, decided by P0: the shared expert as two extra 512-wide slots of `fl_moe_*` (exact split at i = 512, down's two slots summed in fp32 before one rounding), saving 3 launches a MoE layer |
| reused as-is | `gemv.cl` (qkv N 5120, o, dense gate‖up N 12288, down), `k2_silu_mul` (I 6144 and 1024), `gemv_bf16` (router N 256), `gemv_i8w`, `embed_gather`, `argmax` | new variant names / shapes / tuning rows only |

## Global Constraints
- Branch `spec27c-flint-decode`; signed commits; no merge, no push; box tree automatic; every GPU job under `~/b70-gpu.lock` with the DRM-holder check (box idle protocol); series rows on device 0.
- **F0:** every existing kernel binary sha256-identical (G0), `tools/kernel_cmdlines` additions only, existing suites green. Names carry every shape define (the contract of `kolibri_kernels.h`); a `flint_variant_names_test` holds host names to the CMake list.
- **Capture rules, no env vars:** the attention arm (flash vs the eager diagnostic) is chosen by the test harness only; KV form bf16 in this plan.
- The capture refuses any descriptor drift by name (Kolibri's `require` pattern, from `FlintDesc`, not from constants).
- Numbers recorded as measured; estimates labelled.

---

### Task 1: P0 - where the step's time goes, before any new kernel (box first)
**Files:** create `tools/probe/probe_flint_step.cc`, `tools/probe/flint_p0.sh`.
- [ ] **Step 1:** skeletons from **existing binaries** at Flint's shapes, interleaved pairs after warm-up (the probe-timing rule): (A) the ~780-launch skeleton (47 x 16 + 2 x 11 + 5; qkv / o / dense / shared GEMVs at Flint's N / K - new compiles of existing sources, router GEMV, for the routed experts **Ornith's `moe_M1_E256_T8_D2048_I512`** (Flint's exact expert shape; 8 routed + 1 shared block of 512 against Flint's 6 + a 1024-wide shared = 8 blocks' worth: within one block of the bytes; no Kolibri MoE binary exists at E256 - Kolibri's are E384 D2560), norms, the attention at depth with the Kolibri decode kernels), (B) the GEMV-only list (the 1.52 GB of bytes), (C) an empty-kernel list of the same launch count (the launch floor), (D) attention-only at depths 886 / 32k / 128k / 262k with 14 full + 35 sliding. Output: ms per step, achieved GB/s, floor, share of attention.
- [ ] **Step 2: record** `docs/probe-flint-step-2026-11-<dd>.md` and the **decision table**: if the launch floor exceeds 25% of (B), the fused arms (fold + norm in one launch, `nb_res_norm`'s pattern; the shared expert as two slots of the routed pair) get a follow-up task here; else they do not. **Commit.**

### Task 2: the fixture and the host references (no card)
**Files:** create `tests/kernels/flint_ref.h`, `tests/kernels/flint_fixture.h`, `tools/oracle/flint_fixture.py`.
- [ ] **Step 1:** host twins in the kernels' order (FP_CONTRACT off): the norm (`k2_norm_finish`'s chain at one group), `fl_attn_prep` (RoPE chain), sink attention (fp32 scores, sink in the denominator once, bf16 P), `route_sigmoid_renorm` (ties), `fl_moe` combine, the shared add. Each twin **bitwise equals 27a's `flint_ref.py` restatement** on fixtures generated from it (`flint_fixture.py` writes the binary fixtures; the C++ reads them). Failing tests first; the named mutations from 27a Review Focus fail them. **Commit.**

### Task 3: the kernels (Mac-GPU indicative runs, card for the truth)
**Files:** create `src/kernels/flint/fl_{prep,moe}.cl`, `src/kernels/sink_attn.cl`, `src/kernels/route_sigmoid.cl`, `src/kernels/flint_kernels.h`; modify `src/kernels/CMakeLists.txt` (a `B70_FLINT` block like Kolibri's, **new lines only**), `tools/kernel_cmdlines`.
- [ ] **Step 1:** `flint_kernels.h` (constants from the descriptor, variant-name builders, `decode_variants()`), `flint_variant_names_test`. **Step 2:** the reused norm variant (`k2_norm_finish` at `NORM_GROUPS=1`) and `fl_attn_prep` against twins (F2). **Step 3:** `route_sigmoid_renorm` (F2 incl. planted ties, a saturated-sigmoid tie, a bf16-rounding mutation that must flip a planted route). **Step 4:** `fl_moe_gate_up` / `fl_moe_down`, the dense shared path and the fold (F2; the combine order test: rank order must differ on some row; shared added after the routed sum; w unrounded: a bf16-rounded w must differ on some row). **Step 5:** `sink_attn_decode` / `_reduce` (F2: sink = -inf bitwise the no-sink output; dominant sink collapses the output; a window-skipped block still gets the sink; ring wrap; depths 1, 512, 513, 514, 4096, 4097, 65536; fp64 reference at cosine >= 0.99999) and the eager arm. **Each step commits.**

### Task 4: `FlintEngine`, the capture, the CLI, B3 / B4 on synthetic checkpoints
**Files:** create `src/runtime/flint/flint_capture.{h,cc}`, `flint_engine.{h,cc}`, `src/cli/flint_decode.h`; modify `src/cli/b70_decode.cc`; CMake.
- [ ] **Step 1: the walk** (spec 27 section 7.1): embed; per layer by `FlintDesc::layer(l)`: dense layers 0-1 (no router, no shared); MoE layers (routed + shared); head; `device_launches` formula asserted in `build()` (the plan test pins it). **Step 2: `b70-decode flint`** with `--profile`; the greedy golden test against 27a's bf16 golden on the synthetic real-width checkpoint and `--layers N` truncations; replay bitwise (F4). **Step 3: host plan test** (launch formula, sizes). **Commit each.**

### Task 5: the real checkpoint on one card (box)
- [ ] **Step 1:** every first compile (the new kernels; record the binaries' sha256 and the ocloc cmdlines); F2 on the card. **Step 2:** F3 greedy against the bf16 golden on the real prompts (short / 2k / 16k / 64k), tie-aware, routes as sets with the near-tie tolerance 27a measured; **the MoE judge:** engine vs the f32 reference no further than the bf16 golden is, near-miss rule (8 bf16 ulp, cosine distance <= 2x the twin's) for rows differing from the bf16 twin; taps compared per layer to find the first divergence if any row fails. **Step 3:** F4: replay bitwise; decode of a state restored from 27b-style snapshots is a 27e item. **Step 4:** the eager arm vs the flash arm on a prompt set: the flash arm within the near-miss rule of the eager one (flash rounds unnormalised P, the reference the normalised: a recorded difference, not a defect).

### Task 6: speed (box)
- [ ] Decode tok/s at depth 886, 32k, 100k (bf16 KV; the largest depth that fits), against spec 27 section 3's roofline table; the achieved GB/s per phase (weights vs attention); `--pp 2` rows (hand-off cost). If P0's decision table called for the fused arms: build them behind a capture rule (bitwise the as-built list, F4) and re-measure. **Commit** the numbers into the spec's "27c as built" section.

### Task 7: docs; a box queue row only if asked. **Commit.**

## Gates
- F0, F2 (all kernels), F3 (the MoE judge + near-miss), F4 (replay), the launch formula test, P0 recorded **before** Task 3's first kernel is accepted as final (the P0 may change Task 4's fusion decision, never the arithmetic).
