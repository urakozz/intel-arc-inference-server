# Spec 25d - Nemotron 3.5 Bolt prefill: Mamba-2 in the chunked (SSD) form with an fp32 state, the conv over a chunk with the ring carried, the grouped latent MoE, NoPE flash, on `l0`

**Status (2026-10-22): built on the Mac (branch `spec25d-bolt-prefill`; as built: spec 25 §16); every card stage and the rate are box queue row 61.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `BoltEngine::prefill(ids, chunk)` (chunks <= 2048) on the `l0` backend, one card for synthetics and truncations,
two cards for the model: the Mamba-2 layers in the chunked SSD form (absolute 64-row sub-chunks, the state carried in fp32
across sub-chunks and chunks, DPAS only where both operands are bf16), the conv over a chunk seeded from the ring with the last
three columns written back, the linears as int4 dequant slabs + `pf_gemm`, the MoE as a sort / grouped-GEMM / ascending-id
combine over the chunk's real rows only, the attention by Muse's NoPE flash; `b70-decode <bolt> --prefill`, `--bench
--prefill-length N`; B2-B4 on prefill; the prefill rate at depth beside SergiioB's published table.

**Architecture:** spec 25 §2.2, §3 ("Prefill"), §6.2, §9. The GDN prefill's discipline (`src/kernels/prefill/pf_gdn_conv.cl`'s
explicit seed launch and writeback, `pf_gdn_scan.cl`'s record: the DPAS scan that rounded the fp32 state to bf16 cost a token
and was reverted - so the state never meets DPAS here), Kolibri's / LFM2's prefill walk (24d), spec 15d's grouped expert GEMM
(`pf_moe_gemm`), Muse's `mg_pf_flash_attn` (23d; gate-free) at Bolt's identical shape (Decision 15). The SSD form, per head h
and sub-chunk c of L = 64 rows (positions `[64c, 64c + 64)` absolute):
```text
a_t     = dt_t · A_h                                    (dt from the same softplus, no clamp)
cum_t   = Σ_{u <= t} a_u   (fp32, sequential within the sub-chunk)
Y_diag  = ((C Bᵀ) ⊙ exp(cum_t - cum_s) ⊙ [s <= t]) · (dt ⊙ x)      C Bᵀ: [64 x 64] per group by DPAS (both bf16)
S_c     = exp(cum_63) · S_{c-1} + Σ_s exp(cum_63 - cum_s) dt_s · x_s Bᵀ_s      fp32, vector path
Y_off   = exp(cum_t) · (C_t · S_{c-1})                  fp32, vector path
y       = rne(Y_diag + Y_off + D · x)
```
The sub-chunk is absolute (a chunk starts at a multiple of 64 or its first sub-chunk is partial and its boundaries still sit on
the absolute grid), which is what makes splits at multiples of 64 bitwise (Kolibri's absolute-block rule, spec 20 §11).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest; Python 3 + torch (the fixture).

**Spec:** `docs/superpowers/specs/2026-10-20-spec25-nemotron-bolt-design.md` (§2, §3, §6.2, §9 B2-B4, §12 decisions 1, 2, 15).
Precedents: plan 24d task for task; plan 23d (`mg_pf_flash_attn`, the eager prefill build that is bitwise decode's eager);
`src/kernels/prefill/{pf_gdn_conv,pf_gdn_scan,pf_moe,pf_moe_gemm,pf_dequant_slab,pf_gemm}.cl`; vLLM `783380af39`'s test
(ideas only): a partial last sub-chunk (181 = 128 + 53 rows) at nheads 64, headdim 64, ngroups 8, dstate 128, chunk 128, every
tiling configuration finite and equal, ten repeats.

## Dependencies and branch points

- **Plan 25c merged:** the decode list, `bolt_ref.h`, the engine, the buffers, the CLI.
- **Decision 2:** prefill writes the state in the decode list's precision (fp32 default; the bf16 arm rounds once at the end
  of each chunk - not inside the scan).
- **Not here:** serving (25e), the drafter's taps in prefill (spec 26c adds the staging and the projection to this walk).

## Global Constraints

- Branch `spec25d-bolt-prefill` from main; B0 (cmdlines additions only; 25c's decode names unchanged - a names test pins them).
- **No padding rows routed:** the MoE sort sees exactly the chunk's `C` rows (`C` from the host, a capture per C bucket or a
  Control word), never a dummy row (vllm-xpu-kernels 8b95ea6).
- **The fp32 state never meets DPAS** (the GDN record); every DPAS input is a bf16 value the reference also holds as bf16.
- Every number measured or marked; box rules as 25c's. No `rm -rf`. Signed commits; no merge, no push.

## Review Focus

1. **Absolute sub-chunks.** A chunk starting at 2048 + 37 and one starting at 2048 + 64 give the same bits for every row they
   share; the seed of a chunk is the ring's three columns before its first row and the state at its first row; the writeback
   is the last three raw columns and the state after the last row.
2. **The SSD algebra is the recurrence.** On the host (fp64) the SSD twin equals the sequential recurrence of spec 25 §2.2 to
   1e-12 at every row for sub-chunk lengths 1, 2, 63, 64 and a partial 53; in fp32 it is within the bars below; the
   `dt_bias` -10.5 head and an `A_log` of +4.41 (decay e^-82 per unit dt) included - no NaN, no Inf, no underflow to a wrong
   value (`exp` of a large negative cumulative sum is 0, not NaN: the mask is applied before the `exp` where the reference does).
3. **The MoE over a chunk** routes every row with decode's `nb_route` arithmetic (bitwise the decode route of the same row's
   normed input), sorts (row, slot) pairs by expert (stable by row), runs the grouped up (relu²) and down GEMMs, and combines
   per row in ascending expert id with decode's arithmetic - so a row's MoE output equals decode's for the same input up to the
   GEMM's K order (cosine bar), and two rows never interact.
4. **Every tiling is finite and equal:** each sub-chunk / work-group shape the kernels can be built with gives bitwise the same
   output ten times over (`783380af39`'s lesson: a racy configuration wrote NaN rows on another GPU).

---

### Task 1: the prefill host reference and fixture (no card)

**Files:**
- Create: `tests/kernels/bolt_pf_ref.h` (`boltpf`, on 25c's `boltref`), `tests/kernels/bolt_pf_ref_test.cc`,
  `tools/oracle/bolt_pf_fixture.py`, `tests/kernels/bolt_pf_fixture.h`

- [ ] **Step 1: the failing tests:** Review Focus 1, 2 and 3 on the host (fp64 recurrence vs SSD; fp32 SSD vs fp32 recurrence
  within a measured bound - printed, the bar set from it: PROPOSED max relative 1e-4 on y, cosine >= 0.99999 on the state after
  2048 rows); the conv over a chunk = decode's conv row by row bitwise (same fp32 chain); the grouped MoE = decode's MoE per row
  bitwise in the twin (same dot order chosen for the twin); 181-row chunks; positions 0..3 at a sequence start.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "tests: Bolt prefill host reference - absolute 64-row SSD sub-chunks against the recurrence, the conv over a chunk, the grouped latent MoE (spec 25d)"`.

### Task 2: the prefill kernels

**Files:**
- Create: `src/kernels/bolt/nb_pf_mamba.cl` (`nb_pf_seed`, `nb_pf_conv`, `nb_pf_dt` (softplus, cumulative sums per sub-chunk),
  `nb_pf_cb` (C Bᵀ per group and sub-chunk by DPAS; bf16 x bf16 -> fp32), `nb_pf_ssd` (Y_diag, the sub-chunk state, the scan
  across sub-chunks in order, Y_off, + D x, rne), `nb_pf_gnorm`), `src/kernels/bolt/nb_pf_moe.cl` (route over the chunk via
  25c's `nb_route` at M = C, sort / gather (only real rows), the combine), `src/kernels/bolt/nb_pf_moe_gemm.cl` (a family copy
  of `pf_moe_gemm` with a relu² epilogue and no gate), `src/kernels/bolt/nb_pf_prep.cl` (the norm folds at M = C, `nb_sh_act`
  at M = C, the KV write at M = C), `tests/kernels/bolt_pf_kernels_test.cc` (card)
- Modify: `src/kernels/CMakeLists.txt` (the block; `pf_dequant_slab` / `k2_pf_dequant_slab` at K 4096 N 18560 / 4608 / 6400,
  K 8192 / 1024 / 5376 N 4096; `pf_gemm_T0`; Muse's `mg_pf_flash_attn` full build and its eager twin), `src/kernels/bolt_kernels.h`

- [ ] **Step 1: the failing card test** (B2 on prefill): every kernel against `bolt_pf_ref.h`; Review Focus 4 (every build
  variant, ten repeats, finite and equal); the flash against fp64 at cosine >= 0.99999 at depths 3000 / 65536 / 200000; the
  eager build bitwise decode's eager rows (23d's rule). Mac: the plain-OpenCL kernels on the Mac GPU against the twin
  (`bolt_run`), indicative.
- [ ] **Step 2: implement; Mac gates. Commit** `git commit -S -m "kernels: Bolt prefill - SSD sub-chunks with the fp32 state off DPAS, the conv seed / writeback, the grouped relu² latent experts, NoPE flash reused (spec 25d)"`.

### Task 3: the walk, the CLI, B3 / B4 on prefill

**Files:**
- Create: `src/runtime/bolt/bolt_prefill.{h,cc}`, `tests/runtime/bolt_prefill_test.cc` (card)
- Modify: `src/runtime/bolt/bolt_engine.{h,cc}`, `src/runtime/bolt/bolt_sizes.{h,cc}` (`prefill_sizes`: one constant a
  device, the largest term the MoE's sorted rows x I), `src/cli/bolt_decode.h`

- [ ] **Step 1: the failing tests:** `bolt_prefill_test` - the walk's launches per chunk (asserted, `prefill_device_launches`),
  twice bitwise, splits at multiples of 64 bitwise (Review Focus 1), chunked (512 / 1024 / 2048) = whole bitwise, prefill
  against decode's fill: logits cosine >= 0.9995, h taps median >= 0.9998, the SSM state >= 0.999 (PROPOSED - set from Task 1's
  printed bound), the conv columns bitwise; B3 golden after a prefilled prompt on the synthetic and SergiioB's checkpoint (two
  cards; Kolibri's sequential hand-off per chunk, the bounded wait).
- [ ] **Step 2: implement** (`prefill(ids, chunk)`, `prepare_prefill` naming a missing binary, the block hook
  `(end_pos, is_block_end)` for 25e, `--prefill`, `--bench --prefill-length N [--prefill-chunk C]` printing t/s and TFLOP/s
  against 2 x 9.05 B a token); refused by name: `--prefill-backend l0-int8` / `sycl-tla`, `--prefill-chunk` above 2048.
- [ ] **Step 3: commit** `git commit -S -m "runtime: Bolt prefill on one or two cards - the chunk walk, B3 / B4 on prefill (spec 25d)"`.

### Task 4: docs, the queue row, the rate

**Files:**
- Modify: spec 25 ("25d as built"), this plan's status, `box-validation-queue.md` (row 61), `tools/box_validate/stages.sh`
  (`row 61`)

- [ ] **Step 1:** "25d as built" (the sub-chunk length as built - 64, or 128 if Task 2 measured it faster and the walk
  enforces 128-alignment instead -, the launch formula, the scratch, the bars as set, every departure). The queue row:

```text
row 61 "spec 25d - Nemotron 3.5 Bolt prefill on l0: absolute SSD sub-chunks with the fp32 state, the conv seed / writeback, the grouped latent MoE over real rows, NoPE flash (plan 25d)"
stage r61.k1      61 default gpu  -                      -          "bolt_pf_kernels_test: B2 on prefill, every tiling finite and equal x10"
stage r61.synth   61 default gpu  bolt_synth             -          "bolt_prefill_test on the synthetics: splits / chunks bitwise, prefill vs decode's fill"
stage r61.golden  61 default gpu  bolt_g128,oracle_bolt  -          "B3 after a prefilled prompt on SergiioB's checkpoint, two cards"
stage r61.speed   61 optin   gpu  bolt_g128              r61.golden "prefill rate at depth 886 / 8192 / 32000 / 65536 / 131072 beside SergiioB's published table"
```
- [ ] **Step 2 (box, `r61.speed`):** the rate rows and the phase table (Mamba / MoE / linears / attention per chunk). Commit
  `git commit -S -m "docs: spec 25 (25d as built), box queue row 61 - Bolt prefill"`.

**Gate for the plan:** Mac - host tests (SSD vs recurrence, the conv, the MoE twin), `bolt_run` on the plain kernels,
cmdlines additions only. Box - B0; every prefill kernel against its twin, every tiling finite and equal; B4 (twice bitwise,
splits / chunks bitwise; prefill against decode's fill within the bars); B3 on the synthetics and SergiioB's checkpoint; the
rate recorded with its phase table.
