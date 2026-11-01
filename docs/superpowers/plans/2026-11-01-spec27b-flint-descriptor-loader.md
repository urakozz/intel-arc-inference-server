# Spec 27b - Flint: the descriptor, the loader (per-expert AutoRound tensors), the memory plan, the synthetic real-width checkpoint

**Status (2026-11-01): plan written, nothing built.** Needs 27a (its Task 4 header facts fix the tensor dtypes).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `model::FlintDesc` (a per-layer table held to `config.json` key by key), `loader::load_flint` (host repack of 109,612 tensors into the engine's layout-1 int4 blocks, bf16 / fp32 smalls, the int8 head built at load), the memory plan and the max-len size model, `b70-decode` / `b70-serve` dispatch on `model_type flint` (a refusal path only until 27c / 27e), a synthetic real-width checkpoint generator, and the load on the card.

**Architecture:** spec 27 sections 2, 3, 7.1. Template: plan 25b (`2026-10-20-spec25b-bolt-descriptor-loader.md`) and Kolibri's `src/model/kolibri1.*`, `src/loader/kolibri1_{layout.h,repack.*,loader.*,rope.cc}`, `src/loader/moe.cc` (separate gate / up per expert), `src/loader/lm_head_int8.*`, `src/runtime/kolibri/kolibri_sizes.*`.

## Global Constraints
- Branch `spec27b-flint-loader`; signed commits; no merge, no push; box tree automatic, `tools/box.env` never printed or committed; every box job under `~/b70-gpu.lock`; everything under `~/b70`.
- **No existing loader behaviour changes:** every existing loader / model test stays green and every refusal test still refuses; new code beside the old (`flint_*` files). `quant.cc` relaxations, if any are needed (dense-layer shapes, a fp32 router), are exact and each gets a test that the previous refusal still fires for the old shapes.
- The descriptor refuses, each by name: another `model_type`, hidden, head dims; a layer table that disagrees between `layer_types`, `mlp_only_layers` and `no_rope_layer_indices`; `softmax_type` other than `learnable`; `router_score_function` other than sigmoid; `norm_topk_prob` false; `use_sliding_window` false; `tie_word_embeddings` true; `max_position_embeddings` other than 262,144 (the max-len cap, D3, is read from it).

---

### Task 1: the synthetic checkpoints (Python, no card)
**Files:** create `tools/quantize/flint/make_synth.py`, `tools/quantize/flint/test_make_synth.py`.
- [ ] **Step 1:** a real-width synthetic Flint (49 layers, the real config, random int4 g64 sym in the shipped key layout, bf16 / fp32 smalls per 27a's header facts, `--layers N` truncations, `--experts E` for speed): the bytes the loader will read; tests for the key set (equal to the real index's normalised keys) and the size (21.8 GB at full width: a `--dry-run` computes it without writing). **Commit.**

### Task 2: the descriptor
**Files:** create `src/model/flint.h`, `src/model/flint.cc`, `tests/model/flint_test.cc`; CMake lines.
- [ ] **Step 1: failing test:** the table the config gives (14 full / 35 sliding; 2 dense / 47 MoE; full-before / sliding-before counts per layer; RoPE only on sliding), each refusal above, the byte formulas of spec 27 section 3 (weights 21.8 GB +/- 0.1, per-token bytes 1.52 GB, KV per position 28,672 / 14,560).
- [ ] **Step 2: implement** `FlintDesc` (hidden 2048, 32 / 4 / 128, window 513, experts 256, top-6, expert I 512, shared I 1024, dense I 6144, vocab 150144, eps 1e-6, theta 1e4; `layer(l)` -> {full|sliding, dense|moe}); **commit.**

### Task 3: layouts and the host repack
**Files:** create `src/loader/flint_layout.h`, `src/loader/flint_repack.{h,cc}`, tests.
- [ ] **Step 1:** the layout-1 int4 byte formulas for qkv (N 5120), o, dense gate||up (N 12288) and down, routed gate||up / down blocks x 256, shared gate||up (N 2048) / down (K 1024). The shared pair is **its own allocation** (its I differs from the routed I, so it is not block 256 of the routed allocation as in Kolibri / Ornith). The router bf16 tiled for `gemv_bf16` (bf16 in the checkpoint, read in the spec 27 review), norms and q / k norms widened from bf16 to the fp32 `norm_w` the norm kernels take, sinks widened from bf16 to fp32 [32] per layer. The shared expert's three linears are laid out for `gemv.cl` (gate‖up N 2048, down K 1024) - spec 27 section 2's dense path - with the two-slot routed form kept possible (27c decides; the repack is a function of the form).
- [ ] **Step 2: failing tests** (repack of a tiny int4 -> the bytes the kernels expect, bitwise against an independent Python repack; per-expert tensors read by name from the shard index with no assumption of expert-to-shard binding; the qzeros check re-used). **Step 3: implement; commit.**

### Task 4: the loader (device half) and the memory plan
**Files:** create `src/loader/flint_loader.{h,cc}`, `src/runtime/flint/flint_sizes.{h,cc}`, `src/runtime/flint/flint_buffers.{h,cc}`; modify `src/cli/b70_decode.cc`, `src/cli/b70_serve.cc` (a dispatch branch on `model_type == "flint"` beside Kolibri, a refusal message until the engine lands).
- [ ] **Step 1:** `is_flint_checkpoint`, `load_flint`: read order by layer, one reused host scratch per layer (the `load_moe_layer` pattern), the int8 head quantised at load (`lm_head_int8`), the embedding bf16 verbatim; the unconsumed-tensor report must be **empty** on the synthetic and on the real index.
- [ ] **Step 2:** `flint_sizes`: weights, KV (bf16 28,672 / int8 14,560 per position, rings 4096 x 35 x 2048 B), decode and prefill scratch, snapshot sizes; `runtime::plan`-compatible max-len arithmetic in 256-quanta with the default 1.5 GB reserve; the `--pp 2` split by bytes through Kolibri's pp path (the Qwen `require_split` GDN rule does not apply to this family: a test).
- [ ] **Step 3: tests** (host): sizes against spec 27 section 3; plan on 32 GB and on 2 x 32 GB; max-len at int8 and bf16 matches the section 3 table within the stated reserve and is capped at 262,144 (D3; a request above it refused by name); **commit.**

### Task 5: the load on the card (box)
- [ ] **Step 1:** the synthetic `--layers 4`, then full width, then the real quant: loads, VRAM after load vs the plan (+/- 2%), host peak RSS, cold and warm load seconds (box-local copy; USB copies are the operator's). **Step 2:** `--pp 2` load. **Step 3:** record in the spec's "27b as built" section; a box queue row only if asked. **Commit.**

## Gates
- F0 (nothing moves) and: descriptor, repack and plan tests green on the Mac; the unconsumed-tensor report empty on the real index; the load on one card within the plan; `--pp 2` loads; every old refusal still refuses.
