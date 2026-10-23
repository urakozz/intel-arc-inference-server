# Spec 26b - LiLiCorr on the card: the descriptor form, NVFP4 -> our formats at load, the sink attention, top-8 with log-probs, the lattice head and its walk, the draft and projection lists (D1)

**Status (2026-10-23): built on the Mac** (branch `spec26b-lilicorr-drafter`; spec 26 §13 as built): Tasks 1-5 and 7, Task 6's
Mac half (D1's pipeline on 25a's tiny pair) - the card's halves (`lilicorr_kernels_test`, `lilicorr_draft_test`, D1 per arm)
are box queue row **64** (Session 13), PENDING. Defaults are spec 26 §11's PROVISIONAL ones (`int8`, K 7, q `lattice`).
(Planned 2026-10-20: built whatever 26a shows; 26a's verdict picks the default format and K.)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the LiLiCorr drafter on Bolt's last card: `model::DFlashDesc`'s third form (`DFlashForm::LiLiCorr`), the NVFP4
checkpoint decoded at load into one of our three drafter formats (exact W' bf16 + per-tensor g / int8 per row / int4 g64; the
LiLiCorr head always exact), the `ll_*` kernels (sink attention over a 1024-slot ring, top-8 with a full-vocabulary logsumexp,
the lattice head, the lattice walk, a bf16 GEMV with an output scale), the draft list and the projection list (with the anchor
row) as `runtime::TapDrafter`s, `--draft-format`; D1 against 26a's reference on the target's own tap dumps.

**Architecture:** spec 26 §3, §5.1-§5.3, §7 D0 / D1, §10 decisions 2, 3, 8. Spec 19b's structure (`model/dflash.*`,
`loader/dflash_{repack,loader,layout}.*`, `src/kernels/dflash/`, `runtime/dflash/`) and 23f's extension pattern (a form, not a
new descriptor; `loader::load_drafter`'s overload by target shape; `DraftFormat`; no edited kernel source - new sources and new
CMake lines). Kernels: family copies `ll_*` in `src/kernels/dflash/` (Decision: no existing `df_*` source is edited, 23f's
MG0 / D0 rule).

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest; Python 3 + torch (the fixture, 26a's reference).

**Spec:** `docs/superpowers/specs/2026-10-20-spec26-lilicorr-design.md` (§3, §4, §5, §7, §10). Precedents: plans 19b and 23f
Part 2 (Tasks 6-7), `src/model/dflash.{h,cc}`, `src/loader/dflash_*`, `src/loader/rtn.h` (`rtn_int4_g64`), spec 9's int8 row
quantiser, `src/kernels/dflash/df_{prep,attn,topk,select}.cl`, `src/kernels/gemv_bf16.cl`, `runtime/dflash/{dflash_drafter,
muse_drafter}.h`, `runtime/tap_drafter.h`, `tests/kernels/dflash_ref.h`, `tools/oracle/dflash_fixture.py`, `tools/probe/dflash_d1`
+ `tools/oracle/dflash_d1.py`.

## Dependencies and branch points

- **Plan 26a merged** (the reference form, its tiny drafter, the NVFP4 decoder's rules). **Plan 25b merged** (`BoltDesc`, the
  plan's drafter reserve on device 1, the embedding replica) and **25c** for the target's head / embedding on device 1 (the
  draft list reads them). 26a's verdict sets `--draft-format`'s default; until it exists the default is `int8` (Decision 2).
- **Not here:** the verify, the commit, the loop, serving (26c).

## Global Constraints

- Branch `spec26b-lilicorr-drafter` from main; signed commits; no merge, no push.
- **D0:** without `--spec dflash` nothing binds an `ll_*` binary; every existing binary sha256-identical; 19b's / 23f's drafter
  tests unchanged; no existing `.cl` edited (`df_*` reused only from new CMake lines at LiLiCorr's defines).
- **NVFP4 only for this form,** decoded by our loader: `quant_method: modelopt_fp4`, `quant_algo: W4A16_NVFP4`, group 16, the
  three tensors per linear (`weight` U8 [N][K/2], `weight_scale` F8_E4M3 [N][K/16], `weight_scale_2` F32 scalar), any other
  quantised form refused by name; no NVFP4 kernel.
- Numerics as 19b's: residual and norms fp32, GEMV inputs bf16, the ring bf16, the block's K / V fp32, FP_CONTRACT off; every
  reduction a fixed tree; the lattice in fp32.
- No `rm -rf`.

## Review Focus

1. **Exact means exact:** with `--draft-format exact`, every NVFP4 linear's device output equals `g x (W' x)` with W' the
   exactly-decoded bf16 matrix (a bf16 GEMV's fp32 output times g, then one rounding) - tested against the host decode on every
   one of the 25 NVFP4 tensors' sampled columns; the int8 arm's integer codes do not depend on g (spec 26 §3).
2. **The sink** enters the reduce as `l += exp(sink_h - m)` once per (row, head), after the context blocks and the block keys
   merged in their fixed order; the numerator untouched; the window `|q - k| <= 1023` over the 1024-slot ring (row 0 at p sees
   p - 1023 .. p - 1; row 7 sees p - 1016 .. p - 1 plus the block).
3. **Top-8 and the logsumexp:** the candidates by logit descending, exact ties to the lower id; `lp = v - lse` with the lse
   merged over 32 chunks in chunk order (fixed); a row with a planted -inf / huge value gives no NaN.
4. **The lattice head is vLLM's** (26a's port): the relative slot bias offset 7 also when K < 7; the anchor from the projection
   list's anchor buffer; `start` broadcast for slot 0; `pair` for slots 1..6; x 8.0; the walk starts at index 0.
5. **The ring, the taps, the anchor:** the projection list writes position p's K / V at slot `p % 1024` and the anchor buffer
   with the last projected row's normed vector; the block's own K / V never reach the ring.

---

### Task 1: the descriptor form

**Files:**
- Modify: `src/model/dflash.{h,cc}` (`DFlashForm::LiLiCorr`; the LiLiCorr fields; `dflash_desc_from_config` accepts
  `LiLiCorrDraftModel` with `lilicorr_enabled`; `check_dflash_target` against a `DFlashTargetShape` built from `BoltDesc`;
  `check_lilicorr` - the kernels' shapes: 5 layers, hidden 4096, 32 / 2 x 128, FF 12288, window 1024, block 8, 6 taps, conv 2 /
  16, head 1024 / 2 / 8 / 2048 / 1024, top-8), `tests/model/dflash_test.cc`

- [ ] **Step 1: the failing tests:** the published config (typed in the test) parses to the form; refused by name: `has_own_lm_head`,
  a draft vocabulary, `candidate_topk` 6 or 32, block 16, window 2048, a causal layer, `lilicorr_enabled` false, another
  `quant_method` / `quant_algo` / group; held to Bolt (hidden 4096, vocab 131072, taps < 88, mask 990 < 131072); 19b's and 23f's
  cases unchanged.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "model: DFlashForm::LiLiCorr - the LiLiCorr config held to its kernels and to Bolt (spec 26b)"`.

### Task 2: NVFP4 at load and the three formats

**Files:**
- Create: `src/loader/nvfp4.{h,cc}` (`decode_nvfp4(u8, e4m3, K, N) -> W'` bf16-exact; the e2m1 / e4m3 tables), `tests/loader/nvfp4_test.cc`
- Modify: `src/loader/dflash_{repack,loader,layout}.*` (the LiLiCorr names; the three formats; the head always exact; biases,
  norms, sinks, conv bases fp32; `DraftFormat::Exact` beside `Int8` / `Int4G64` / `Bf16`), `tests/loader/dflash_repack_test.cc`,
  `src/runtime/bolt/bolt_sizes.*` (the drafter's bytes into 25b's reserve)

- [ ] **Step 1: the failing tests:** `nvfp4_test` - all 16 x 256 (code, scale) products exactly bf16 and f16; the NaN scales
  refused naming the tensor and the element; a decoded tiny tensor equals 26a's `nvfp4_decode`; `dflash_repack_test` - the
  LiLiCorr names both ways, the three formats' bytes (exact 2.152 + 0.074 GB, int8 1.077 + 0.074, int4 g64 0.572 + 0.074,
  derived; the test prints, the plan sums), the int8 codes independent of g, the int4 arm equal to `rtn_int4_g64` on `g · W'`
  (fp64), the head exact in every arm.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "loader: NVFP4 decoded exactly (W' bf16 + g) into the drafter formats - exact / int8 / int4 g64, the LiLiCorr head always exact (spec 26b)"`.

### Task 3: the host twin and the fixture

**Files:**
- Modify: `tests/kernels/dflash_ref.h` (the LiLiCorr form: sink attention, top-8 + lse, the lattice head, the walk, the anchor),
  `tools/oracle/dflash_fixture.py` (`--lilicorr`: 26a's tiny drafter over 25a's tiny Bolt), `tests/kernels/dflash_ref_test.cc`
- Create: `tests/loader/lilicorr_dflash_fixture.safetensors` (generated; small)

- [ ] **Step 1: the failing test:** the twin equals `dflash_ref.py`'s LiLiCorr form on its fixture (rows, candidates, log-probs,
  lattice, walks greedy and seeded) to 1e-6 (fp32 order differences printed); Review Focus 2-5 on the host.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "tests: the LiLiCorr form of the DFlash twin and its fixture (spec 26b)"`.

### Task 4: the kernels

**Files:**
- Create: `src/kernels/dflash/ll_attn.cl` (df_attn's structure + the sink term), `src/kernels/dflash/ll_topk.cl` (df_topk at
  TOPK 8 + the chunk (max, Σ exp) pairs and their merge, `lp` out), `src/kernels/dflash/ll_lattice.cl` (the small-M GEMM at up to
  64 rows - 56 lattice rows - with bias, the lattice attention with the slot bias (one work-group per head), RMSNorm / LayerNorm(5)
  / the feature MLP / SiLU / the factor projection / L2 normalise / the start and pair scores), `src/kernels/dflash/ll_select.cl`
  (the walk over `lattice[s][previous]`, greedy or Gumbel with df_select's keying, the readback, `cur_token[1 + s]`),
  `src/kernels/dflash/ll_gemv_bf16_g.cl` (gemv_bf16's tiles and order with an fp32 output scale), `tools/mac/clrun/lilicorr_run.cc`
- Modify: `src/kernels/dflash_kernels.h` (names), `src/kernels/CMakeLists.txt` (the `ll_*` lines; `df_prep` at H 4096, NH 32,
  NKV 2, HD 128, conv 2 / 16, RoPE θ 1e4, RING 1024; `gemv_i8w` / `gemv.cl` / `gemv_bf16` at the drafter's shapes and M 8; the
  target head at 7 rows on device 1)

- [ ] **Step 1: the failing tests:** `dflash_names_test` (every `ll_*` / reused name a target); `lilicorr_run` on the Mac GPU
  against the twin (indicative; every kernel is plain OpenCL C or has a plain path - df_*'s rule); `lilicorr_kernels_test` (card).
- [ ] **Step 2: implement; Mac gates (`--kernels`). Commit** `git commit -S -m "kernels: LiLiCorr - sink attention over the 1024 ring, top-8 with the logsumexp, the lattice head and its walk, the scaled bf16 GEMV (spec 26b)"`.

### Task 5: the lists and the drafter object

**Files:**
- Create: `src/runtime/dflash/lilicorr_drafter.{h,cc}` (`runtime::dflash::LiliDrafter : TapDrafter` - weights, buffers (ring 1024 x
  5,120 B, taps [8][6][4096], the anchor [4096] fp32, candidates / log-probs / lattice / readback), the draft list over Bolt's
  embedding replica / head / Control on device 1, the projection list (`proj_n <= 8` rows at `proj_pos`, the anchor written from
  the last row), `ring_bytes` / `save_ring` / `load_ring` incl. the anchor, `key()`), `tests/runtime/lilicorr_draft_test.cc` (card)
- Modify: `src/runtime/dflash/dflash_control.h` (the readback words for 7 x 8 candidates and the lattice, static_asserted),
  CMake files

- [ ] **Step 1: the failing tests (card):** a draft from a fixed ring / anchor equals the twin's (ids at T 0, the lattice cosine);
  the projection of 8 rows equals the twin's ring rows bitwise and its anchor; a ring save / load round-trips; `draft_list_k`
  always runs the 8-row block (23f's finding).
- [ ] **Step 2: implement; PASS (box). Commit** `git commit -S -m "runtime: LiliDrafter - the draft list and the projection list with the anchor row on Bolt's last card (spec 26b)"`.

### Task 6: D1 on the target's own taps (box)

**Files:**
- Modify: `tools/probe/dflash_d1` (`--model lilicorr`), `tools/oracle/dflash_d1.py` (the LiLiCorr form), `tools/box_validate/dflash_d1.sh`

- [ ] **Step 1 (box, `r64.d1`):** at teacher-forced anchors of 25a's tap dumps (the engine's own taps come with 26c's verify list;
  the tool takes either): the 8 candidate sets equal (tie-aware), their log-probs to 1e-3, the lattice cosine >= 0.9999 (exact) /
  0.999 (int8, int4), the walk ids equal at T 0, the anchor cosine >= 0.9999 - per arm. If the lattice bar fails while the sets and
  walks hold, fp32-input GEMV variants for the head's projections are the lever (19b's note).

### Task 7: docs, the box queue row

- [ ] **Step 1:** spec 26 ("26b as built"), this plan's status, the queue row 64 and its `stages.sh` block:

```text
row 64 "spec 26b - LiLiCorr on the card: DFlashForm::LiLiCorr, NVFP4 decoded exactly into exact / int8 / int4 g64, the ll_* kernels (sinks, top-8 + lse, the lattice head and walk), the draft and projection lists (D1) (plan 26b)"
stage r64.host   64 default cpu  -                              -        "dflash_test, nvfp4_test, dflash_repack_test, dflash_ref_test (LiLiCorr), dflash_names_test"
stage r64.k1     64 default gpu  -                              -        "lilicorr_kernels_test: every ll_* kernel against the twin"
stage r64.draft  64 default gpu  bolt_g128,lilicorr             -        "lilicorr_draft_test on Bolt's device 1: the draft, the projection, the ring round trip"
stage r64.d1     64 optin   gpu  bolt_g128,lilicorr,oracle_bolt_taps r64.draft "D1 per arm against dflash_ref.py's LiLiCorr form on the tap dumps"
```
- [ ] **Step 2: commit** `git commit -S -m "docs: spec 26 (26b as built), box queue row 64 - LiLiCorr on the card"`.

**Gate for the plan:** Mac - every host test (the form, NVFP4's enumeration, the formats, the twin), `lilicorr_run` against the
twin, cmdlines additions only, D0 by construction. Box - every `ll_*` kernel against the twin; the drafter's lists on device 1;
D1 per arm recorded against its bars.
