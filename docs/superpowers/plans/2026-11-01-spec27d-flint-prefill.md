# Spec 27d - Flint prefill: the chunk walk on the l0 backend, the sink flash attention with window skip, the MoE chain with the shared block and Flint's combine, and the fixed dequant cost

**Status (2026-11-01): plan written, nothing built.** Needs 27c (kernels, capture, golden harness).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** prefill the model in chunks of 2048 through the l0 backend (default; no other backend), bitwise reproducible across splits, with the prefill kernels of spec 27 section 7.2, B3 / B4 on prefill, and a measured decision on the per-chunk dequant cost.

**Architecture:** spec 27 sections 2, 3 (prefill estimate), 7.2, 9. Template: Kolibri's `src/runtime/kolibri/kolibri_prefill*.{h,cc}`, `src/kernels/kolibri/kol_pf_{attn,moe,linear}.cl`, `src/runtime/prefill/moe.{h,cc}`, Ornith's `pf_moe.cl` (256 experts allowed; K2's `k2_pf_moe.cl` is capped at 255), plan 25d (`2026-10-20-spec25d-bolt-prefill.md`).

**Kernel list (new files):** `sink_pf_attn` (from `kol_pf_attn.cl`: window skip by tile on sliding layers, causal flash on full NoPE layers, `lsum += exp(sink - m_row)` before `1/lsum`, P rounding as the reference needs - see Task 3), `fl_pf_prep` (norm, q / k norm, RoPE, cache write, chunk-sized), `fl_pf_moe_*` (Ornith's sort / gather / dequant / grouped GEMM at `E256 T6 D2048 I512`; combine = Flint's ascending-id bf16 chain, the shared expert (I 1024, ungated) as a dense block and added after the routed sum), dense layers through `pf_gemm_silu` / `pf_gemm` (reused), `fl_norm` (reused from 27c).

## Global Constraints
- Branch `spec27d-flint-prefill`; signed commits; no merge, no push; GPU jobs under `~/b70-gpu.lock`.
- F0 as in 27c. The chunk is 2048 (`kC`; a larger chunk is Task 5's decision, not assumed). The ring is 4096 slots (>= chunk + window). Prefill chunks end at multiples of 2048 where a prefix-cache block ends.
- **Prefill default l0**; the composed (non-flash) attention is the diagnostic arm (test-harness selection only).
- No int8 prefill linears in this plan (decide D9 after Task 5).

---

### Task 1: the host reference and fixture (no card)
- [ ] **Step 1:** `tests/kernels/flint_pf_ref.h` + `tools/oracle/flint_pf_fixture.py`: the chunked chain on tiny and mid shapes equals 27a's `flint_ref.py` over the same rows (chunks = whole where the op is row-wise; attention across the window edge within the measured bound); failing tests first; **commit.**

### Task 2: the kernels
- [ ] **Step 1:** `fl_pf_prep` and the pf norm (F2 bitwise vs twins). **Step 2:** the MoE chain: sort / gather with 256 experts (`TM` 32, tile table `tmax(C)`, ids staged as bytes - fine at 256 only in Ornith's form), grouped GEMM equals the dense per-expert GEMM bitwise, the combine twin; shared block; **the expert-count and chunk-limit tests** (C <= 2048 asserted). **Step 3:** `sink_pf_attn`: F2 against fp64 at cosine >= 0.99999; sink = -inf bitwise the no-sink kernel; the window skip equals a full-mask run on sliding layers; depth > 513 chunks; ring wrap across chunk boundaries. **Each commits.**

### Task 3: the chunk walk, the CLI, B3 / B4 on prefill
- [ ] **Step 1:** `flint_prefill.{h,cc}` (chunk walk, launch formula per chunk asserted in a plan test), `FlintEngine::prefill`, `b70-decode flint --prefill`. **Step 2:** F3 on synthetic real-width and `--layers N`: prefill + decode greedy against 27a's bf16 golden. **Step 3:** F4: splits at multiples of 64 bitwise; chunked = whole (bitwise on the engine's own state: KV and rings); replay bitwise. **Step 4:** the P rounding question: the flash kernel rounds unnormalised P, the reference rounds normalised P (Qwen's `pf_flash_attn` finding): measure the flash arm vs the composed arm vs the reference on real rows and record the difference against the near-miss rule; if the rule fails, the flash kernel gets the normalised-P variant (a recorded decision, not a default). **Commit each.**

### Task 4: real checkpoint prefill (box)
- [ ] F2 on the card; F3 on the golden prompts (2k, 16k, 64k) incl. the MoE judge (engine no further from f32 than the bf16 golden; near-miss rule); F4 incl. `--pp 2` split anywhere bitwise another split (Kolibri's pp prefill path).

### Task 5: the rate and the fixed-cost decision (box)
- [ ] **Step 1:** prefill tok/s at depth 0, 16k, 64k, 128k, 200k, with the time split per phase (route / sort / dequant / grouped GEMM / attention / dense). **Step 2:** compare to spec 27 section 3's estimate (~6,400 t/s short, ~1.5k t/s at 128k; the dequant ~0.27 s per 2048-token chunk): if the dequant share exceeds 40% of the chunk, run the two probes that decide D9: (a) a chunk of 4096 (the sort's SLM `sid[KC*TOP_K]` bytes and the tile table scale: a probe, not an edit of the shipped kernel), (b) a direct-int4 grouped GEMM probe (small-M DPAS pattern from spec 28 / `docs/probe-small-m-*`, here large-M per expert at M ~ 48). Record both with interleaved pairs; the build decision goes to the operator (D9). **Commit** the numbers into the spec's "27d as built" section.

## Gates
- F0, F2, F3 (MoE judge + near-miss), F4; the launch formula per chunk; the rate table recorded; D9's probe numbers recorded.
