# Spec 23d - Muse-Glimmer-30B prefill: the `l0` chunk walk, windowed flash over the ring, NoPE flash

**Status (2026-10-15): plan, not started. Box queue row 45.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit after each step** (signed, `git commit -S`; a step that does not build or pass yet goes in as `WIP: ...`) - never leave a step's files uncommitted. When you stop for any reason, leave `HANDOFF-spec23d.md` (untracked, worktree root): the last step done, what is half-done, the next command.

**Goal:** `MuseEngine::prefill(ids, chunk)` on the `l0` backend, one card or two: chunks of up to 2048 rows, the linears as dequant slabs + DPAS GEMM, Muse's prep at M = 2048, flash attention with the 2048-key window over the 4096-slot ring on sliding layers and causal NoPE flash on full layers (GQA 16, head_dim 128, the output gate after), the sandwich; MG2-MG4 on prefill; `b70-decode <muse> --prefill` and `--bench --prefill-length N`.

**Architecture:** spec 23 §5.3. Kolibri's prefill (spec 20 §12) without the MoE: the walk `runtime/muse/muse_prefill.{h,cc}` records one device's layers per chunk with no host wait inside, its own archive `b70_muse_prefill` so a decode-only target links what it did; decode's chains wherever a decode twin exists, DPAS sums and 64-key flash tiles where they do not (so prefill's KV, rings and logits are decode's to a cosine bar and near-ties, not bitwise).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md` (§5.3, §5.5, §4.2 MG2-MG4, §10 int8 prefill out of scope). Precedents: plan 20d (`2026-10-06-spec20d-kolibri-prefill.md`: `kol_pf_attn.cl`'s window over the ring in absolute 64-key blocks, `m_safe`, the sequential two-card chunk hand-off, the scratch plan), plan 21d (the slab arrangement at K 2560 / 6144, replay rules), `src/kernels/prefill/{pf_dequant_slab,pf_gemm,pf_flash_attn,pf_prep,pf_embed}.cl`.

## Dependencies and branch points

- **23c merged** (the decode list, the twin `museref`, `mg_prep.cl`, the synthetic golden sets).
- The prefill slab kernels exist for K 5120 (Qwen3.8) and K 2560 / 6144 (Kolibri / K2); Muse needs K 6656, 4096 and 19968 at new CMake lines.

## Global Constraints

- Branch `spec23d-muse-prefill` from main; signed commits; no merge, no push; box rules as 23c (lock, detached, interleaved pairs).
- MG0: no existing `.cl` edited; `kernel_cmdlines` additions only; 23c's decode list unchanged (its launch formula re-asserted).
- **`l0` only.** `--prefill-backend l0-int8` refused naming spec 23 §10 (1024-wide Hadamard blocks; 6656 = 6.5 x 1024); `sycl-tla` refused (no Muse walk); `--prefill-chunk` above 2048 refused (the ring is sized for chunk 2048 + window 2047 = 4095 <= 4096).
- Precision: flash keeps fp32 probabilities (the engine's convention); `B70_MUSE_ATTN=eager` binds `_EAGER` builds in prefill too (the reference's rounding points).
- No `rm -rf`. No third-party engine named.

## Review Focus

1. **The window over the ring in prefill:** row t of a chunk at c0 sees keys `(c0 + t - 2048, c0 + t]`; a chunk of 2048 needs keys `[c0 - 2047, c0 + 2047]` - 4095 positions in 4096 slots: no row's live key is overwritten by a later row's write within the chunk. A test at c0 = 2048, 4096, 6143 checks every row's visible set against the twin, including the first and last row of the chunk.
2. **Splits at multiples of 64 are bitwise; other splits are close.** Absolute 64-key blocks from `lo & ~63`; a tile holding none of a row's keys leaves `(m, l, o)` exactly as they were (`m_safe`). Chunks of 64 and splits at 64 / 2048 bitwise; a split at 2049 within the cosine bar only.
3. **Full (NoPE) layers in prefill read the growing cache** from position 0 to the row: causal, no window, no rotary; at c0 > 0 the earlier chunks' rows come from the cache, not recomputed.
4. **The gate in prefill** multiplies after attention with decode's chain (`rne(rne(o) x rne(sigmoid_torch(g)))`), the gate rows from the fused slab GEMM's columns 4096..8191 - a test against decode's `mg_attn_reduce` output on the same row.
5. **Prefill hands decode a consistent state:** after `prefill(ids)` the next `generate` reads the rings (`p & 4095`), the full cache and Control's pos exactly as if decode had ingested the ids (logits within the bar; rings and KV within the cosine bar row by row; pos exact).

---

### Task 1: the host twins of the walk

**Files:**
- Create: `tests/kernels/muse_pf_ref.h` (namespace `musepf`), `tests/kernels/muse_pf_ref_test.cc`
- Modify: `tests/kernels/CMakeLists.txt`

**Interfaces:**
- Consumes: `museref` (23c).
- Produces:

```cpp
namespace musepf {
// the window walk in fp64, tile by tile (64-key absolute blocks), against a direct softmax
void flash_window_fp64(const uint16_t* q, const uint16_t* ring_k, const uint16_t* ring_v, uint32_t c0,
                       uint32_t rows, bool sliding, double* out);
void eager_window(/* same, bf16 chain */ uint16_t* out);
std::pair<uint32_t, uint32_t> visible(uint32_t pos, bool sliding);   // [lo, hi] absolute key range
}
```
- [ ] **Step 1: failing test** `muse_pf_ref_test`: the fp64 tiled walk == the direct softmax within 1e-12 at c0 0 / 1000 / 2048 / 4096 / 6143 through the ring; a row whose first tile holds none of its keys; Review Focus 1's visible sets; `eager_window` == 23c's `museref::attention_eager` bitwise on the same rows; reversed chunk order and in-tile permutations of the keys bitwise (the sum order is the tile order).
- [ ] **Step 2: implement, PASS.** Commit `tests: the Muse-Glimmer prefill twins - the window over the ring, tile by tile (spec 23d)`.

### Task 2: the prefill kernels

**Files:**
- Create: `src/kernels/muse/mg_pf_attn.cl`, `src/kernels/muse/mg_pf_prep.cl` (only what `mg_prep.cl` at `-DM=2048` cannot serve), `tests/kernels/muse_pf_variant_names_test.cc`
- Modify: `src/kernels/muse_kernels.h` (a "spec 23d" block), `src/kernels/CMakeLists.txt`, `tools/mac/clrun/muse_run.cc`

**Interfaces:**
- `mg_pf_flash_attn_Q32KV2_{W2048_R4096,F}[_EAGER]`: `kol_pf_attn.cl`'s DPAS flash structure at GQA 16 (an 8-row group's 16 q heads of one kv head as the DPAS M), the window from the absolute block below the group's first row, `m_safe`; F = causal NoPE over the full cache.
- `mg_pf_attn_gate` (or the gate folded into the flash epilogue, if `kol_pf_attn.cl`'s epilogue shape allows - recorded): decode's gate chain per row.
- Reused at new CMake lines: `pf_dequant_slab` / `k2_pf_dequant_slab` at {K 6656, N 8704 as 8 x 1024 + 512}, {4096, 6656 as 6 x 1024 + 512}, {6656, 39936 as 39 x 1024}, {19968, 6656}; `pf_gemm_T0`; `pf_embed` (or `mg_embed` at M 2048); `pf_res_fold` `_Z` at K 6656; `mg_prep.cl` at `-DM=2048`.
- [ ] **Step 1: failing tests:** names test; `muse_run`'s prefill checks on the Mac GPU (indicative): the slab dequant at K 6656 / 19968 exact, `mg_prep` at M 2048 against `museref` (bounded ulps as 23c), the flash (portable path) against `musepf::flash_window_fp64` at cosine >= 0.99999 and eager bitwise.
- [ ] **Step 2: implement, PASS;** OpenCL syntax of every variant, L0 syntax, `kernel_cmdlines` additions only. Commit `kernels: mg_pf_attn - DPAS flash at GQA 16 over the 4096-slot ring and the NoPE cache (spec 23d)`.

### Task 3: the walk and `MuseEngine::prefill`

**Files:**
- Create: `src/runtime/muse/muse_prefill.{h,cc}`, `src/runtime/muse/muse_prefill_engine.cc`, `tests/runtime/muse_prefill_test.cc`
- Modify: `src/runtime/muse/muse_sizes.{h,cc}` (`PrefillSizes`, the prefill scratch, `prefill_launches`), `src/runtime/muse/muse_engine.h`, `src/runtime/muse/CMakeLists.txt` (archive `b70_muse_prefill`), `src/cli/muse_decode.h` (`--prefill`, `--bench --prefill-length N [--prefill-chunk C]`)

**Interfaces:**
- Produces:

```cpp
namespace runtime::muse {
inline constexpr uint32_t kPfC = 2048;
struct PrefillSizes { size_t ids, resid, x, sub, partials, slab, sumsq_a, sumsq_r, attn_q, gate, attn_out,
                      logits; size_t total() const; };
PrefillSizes prefill_sizes(const model::MuseGlimmerDesc& d, uint32_t max_len);
size_t prefill_device_launches(const model::MuseGlimmerDesc& d, const model::MusePlacement& p, uint32_t dev);
// MuseEngine:
//   void prefill(const std::vector<uint32_t>& ids, uint32_t chunk = 0);   // 0 = kPfC
//   void prepare_prefill();  void set_prefill_replay(bool);  size_t prefill_launches() const;
//   void set_block_hook(std::function<void(uint32_t end)>);              // 23e's prefix cache
}
```
The chunk per layer: `mg_norm` (M 2048) · q||gate||k||v 9 slabs (+ `pf_gemm_T0` each) · `mg_attn_prep` (M 2048: q / k norms, 3.87, RoPE on sliding, every row's K / V into `p & 4095` or `p`) · `mg_pf_flash_attn` · gate · o_proj 7 slabs x 2 · `pf_res_fold _Z` + `mg_post_add` · `mg_norm` · gate||up 39 slabs x 2 + silu_mul · down 7 slabs x 2 · `pf_res_fold _Z` + `mg_post_add`; device 0 adds the embed; the last chunk adds decode's head over the last row. `prefill_device_launches` asserts the count (derived; printed).

- [ ] **Step 1: failing tests** `muse_prefill_test` (label `gpu`; modes `all`, `split`, `pp`): on the 4-layer synthetic: prefill of `short_prose`, `4k`, `8k` then 32 greedy ids == the decode-ingested run's ids (tie-aware) with KV / rings / logits within the PROPOSED bars (layer 0 every row >= 0.999, median >= 0.9998, p01 >= 0.99 - Kolibri's, printed); replay bitwise; chunks of 64 and splits at 64 / 2048 bitwise; a split at 2049 within the bar (Review Focus 2); Review Focus 5.
- [ ] **Step 2: implement**; the prefill scratch planned into `--max-len auto` when a run prefills (one card still reaches 131072 with the int8 head and the drafter reservation - asserted in `muse_plan_test`). Refusals: `cli_reject_muse_prefill_{int8,sycl,chunk}` (23c's `cli_reject_muse_prefill` removed). Commit `runtime: Muse-Glimmer prefill on l0 - the chunk walk, replay, the block hook (spec 23d)`.

### Task 4: two cards

**Files:**
- Modify: `src/runtime/muse/muse_prefill_engine.cc`, `tests/runtime/muse_prefill_test.cc` (`pp` mode)

- [ ] **Step 1:** Kolibri's sequential chunk hand-off (spec 20 §12): layers `[0, s)` on device 0, a second `PipelineLink` over `pp_landing_layout(2048 x 6656 x 2, the Σ² chunk)` (27.3 MB + the sums), copies and the cross-device event, then `[s, L)` on device 1, the host waiting with `PipelineOptions::prefill_timeout_ms`; a lost hand-off throws naming device 1 and marks the engine until `reset()`. Test: two cards bitwise one card on the synthetic (split 2), P4 on the chunk hand-off. Commit `runtime: Muse-Glimmer prefill across two cards - the chunk hand-off (spec 23d)`.

### Task 5: golden on prefill, speed, docs, the queue row

**Files:**
- Modify: `tests/golden/muse_golden_test.cc` (`prefill` mode), `docs/BENCHMARKS.md` (Muse prefill rows, PENDING), spec 23 ("23d as built"), this plan's status, `docs/superpowers/plans/box-validation-queue.md` (row 45), `tools/box_validate/stages.sh` (row 45 block)

- [ ] **Step 1:** MG3 on prefill (synthetic, then real): every determined prompt row's argmax equals the reference's (`logits.raw` margins, `cap_tie` undetermined), the 32-id continuation tie-aware; flash and eager; chunks of 64; int8 head.
- [ ] **Step 2: speed rows (opt-in):** pp 4096 / 16384 / 32768 at chunk 2048, one card and two; tokens/s against 50.3 GFLOP a token (derived) and Qwen3.8's l0 rows at the same length. Commit `golden: Muse-Glimmer prefill gates and speed rows (spec 23d)`.
- [ ] **Step 3: queue row 45. Stage list (sketch):**
  - `r45.k0` default cpu `g0.*`: MG0 and 23c's decode list unchanged;
  - `r45.host` default cpu: `muse_pf_ref_test`, `muse_pf_variant_names_test`, `muse_plan_test`;
  - `r45.k1` default gpu: every new binary compiled; `muse_pf_kernels_test` (flash >= 0.99999 vs fp64 over the ring to depth 30000; a split at 64 bitwise);
  - `r45.prefill` default gpu `oracle_muse_synth`: `muse_prefill_test all` / `split`;
  - `r45.golden` default gpu: MG3 on prefill (flash, eager, chunks of 64, int8 head);
  - `r45.pp` default gpu, after `r23.*`: `muse_prefill_test pp`;
  - `r45.reject` default cpu: the three refusals;
  - `r45.real` default gpu `muse_gptq,oracle_muse`: MG3 on the real checkpoint;
  - `r45.speed` optin gpu `muse_gptq`.
- [ ] **Step 4: commit** `docs: spec 23 (23d as built), box queue row 45 - Muse-Glimmer prefill`.

**Gate for the plan:** Mac - the twins (fp64 walk, eager bitwise), every variant's syntax, `kernel_cmdlines` additions only, `tools/mac_check.sh --base main --quick --kernels` exit 0. Box - MG2-MG4 on prefill (synthetic, then real), two cards bitwise one, the speed rows recorded.
