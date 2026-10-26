# Spec 28b - small-M GEMM build: `gemm_sm` for every M >= 2 list, M = 1 bitwise, the gates, the measured cost table

**Conditions:** this plan starts only when **spec 28 §9 records GO** (plan 28a on the box) **and the operator says
go** (one implementing plan at a time).

**Status (2026-10-27): PLANNED - after 28a's GO.** Nothing built. Box queue row **69** (spec 28's header lists the
track's rows 68-75).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `src/kernels/gemm_sm.cl` (int4 g64, both layouts) and `gemm_sm_i8w` (the int8 head and the DFlash drafter's
int8 linears) in production at the type and S per shape 28a chose; every list the capture builds at **M >= 2** binds
them, every M = 1 list keeps today's binaries bit for bit; spec 28's N1-N8 and S2-S3 on the box; the MTP / DFlash cost
table measured instead of extrapolated; the record.

**Architecture:** spec 28 §2 and §4. `runtime::gemv_kind(M)` in the capture (`src/runtime/capture.cc`'s `gemv()` and
`gemv_i8w()` helpers - the one place the decode lists bind a linear) returns `Gemv` at M = 1 and `SmallM` at M >= 2; a
test-only capture option (`CaptureOptions::force_linear`, not a flag, not an env var) overrides it for N2-N4. Kernel
names in `src/kernels/kernels.h` beside `gemv`'s (`gemm_sm_M<M>_K<K>_N<N>[_L1][_S<S>]`, `gemm_sm_i8w_M<M>_K_N`),
built by `src/kernels/CMakeLists.txt` for Qwen3.8's and Agnes's shapes at M = 2..8 and 16. The partials keep `gemv`'s
layout, so `prep_res_fold`, `prep_silu_mul` and the argmax are untouched (unless 28a chose another S for a shape: then
that consumer's S variant is added and `Capture::check_sizes` checks it).

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md` (§2-§5, §9 for 28a's choices).
Precedents: spec 9b (`gemv_i8w` beside `gemv`: new binaries, nothing old moved), spec 19a Task 4 (`B70_VERIFY_M8`, the
M = 5..8 lists), spec 8b (M2's test, `mtp_gpu_test`).

## Dependencies and branch points

- Main with 28a merged (the probe, `tests/kernels/gemm_sm_ref.h`, spec 28 §9). Branch **`spec28b-small-m-build`**.
- **Box:** row 68 passed (the verdict); the gate checkpoint; the golden and A4 sets.

## Global Constraints

- Signed commits after every step; no merge, no push. No `rm -rf`. Box tree automatic.
- **N1 is the first gate and is never traded:** `tools/kernel_cmdlines` shows only additions (`gemm_sm*` and any S
  variant 28a required); every existing binary checksum-identical (G0).
- No new CLI flag and no environment variable (the operator, 2026-10-27).
- Timed stages: idle box, device 0, interleaved pairs after a warm-up, median of 3 (steps) / 5 (kernels).

## Review Focus

1. **M = 1 untouched:** every list at M = 1 (decode, `ingest`, `generate`, the MTP draft list, the prefill tail's head,
   the drafter's projection list at its row counts that are 1) binds the old binaries - checked by `gemv_kind` tests and
   by G0.
2. **Property R holds in production:** the same row block (8), group order and S for every M >= 2 of a shape - one
   source of truth for the per-shape S table (`kQwen38Tuning`'s neighbour, `kSmallMTuning`), asserted at capture.
3. **Every M >= 2 list found:** `build_verify` (MTP, lookup), `build_verify_taps` (DFlash), the drafter's int8 linears
   (`runtime/dflash/`), `build_rows` under `B70_BATCH_PROBE` (13a), the `--pp 2` stage verify lists
   (`build_stage_verify_taps` and the MTP stage lists) - a grep for every `gemv(` / `gemv_i8w(` call site with an M
   argument, listed in the as-built.
4. **Gates restated, not loosened:** M2's cosine bars were always the bars; D3's "bitwise" becomes N4 + N3 (spec 28
   N5), recorded in spec 19 §13; the greedy gates keep zero non-tie divergences.

---

### Task 1: the kernels (Mac + box)

**Files:**
- Create: `src/kernels/gemm_sm.cl` (from 28a's stand-in, the chosen type), `tests/kernels/gemm_sm_test.cc` (box: N2
  against `gemm_sm_ref.h` and against `gemv`; N3 property R; every built shape and M)
- Modify: `src/kernels/kernels.h`, `src/kernels/CMakeLists.txt`, `tests/CMakeLists.txt`,
  `tools/kernel_cmdlines` expectations if it lists names

- [ ] **Step 1: failing test** `gemm_sm_test` (names, shapes, the bound per output, R).
- [ ] **Step 2: implement;** `tools/mac/cl_syntax.sh` passes every new command line; `tools/mac_check.sh --base main
  --quick` green, cmdlines **+N / -0 / ~0**. Commit `git commit -S -m "kernels: gemm_sm - small-M int4 g64 and int8-head DPAS GEMMs, M 2..8 and 16 (spec 28b)"`.

### Task 2: the capture rule (Mac)

**Files:** Modify `src/runtime/capture.cc` / `capture.h` (`gemv_kind`, `CaptureOptions::force_linear`,
`kSmallMTuning`, `check_sizes`), `src/runtime/dflash/` (the drafter's int8 linears at M rows), the PP stage list
builders. Create `tests/runtime/gemv_kind_test.cc` (host: M = 1 -> Gemv for every list kind; M >= 2 -> SmallM; the
override; the S table equal across M).

- [ ] **Step 1: failing test, then implement, PASS** (host); `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "runtime: every M >= 2 decode list binds gemm_sm; M = 1 keeps gemv (spec 28b)"`.

### Task 3: the gates on the card (box, `r69.kernels`, `r69.gates`)

- [ ] **Step 1:** `gemm_sm_test` (N2, N3); `mtp_gpu_test`'s M2 at M = 2..4 and the M = 5..8 rows through the M8 lists
  (N4: logits and GDN slot cosine >= 0.99999, the 64-token continuation); `dflash_engine_test` T1 / P with the bars of
  spec 28 N5 (if rows 54-56 have passed by then; otherwise recorded as pending on plan 19f); the greedy gates N6
  (`--mtp K`, `--spec lookup`, `--spec dflash` when available, against `--spec off`, golden + A4, 256 tokens, the tie
  census); determinism and replay (N8); the full suite and G0 (N1). Commit
  `git commit -S -m "test: small-M GEMM gates on the card (spec 28b)"` with any test edits.

### Task 4: speed and the cost table (box, `r69.speed`)

- [ ] **Step 1:** S2 - `probe_mtp_steps <snap> 4096 32 3 int8 off 16384 8` and `probe_verify_m` at M = 1..8 (16 through
  the probe): verify(M) in plain steps against 28a's derived values and today's 1.17 / 1.52 / 1.74. S3 -
  `tools/probe/mtp_d1.sh greedy 4096`: the tool-call geomean at `--mtp auto` against 1.633x.
- [ ] **Step 2:** the measured verify row replaces the defaults of `MtpCost` (spec 8 §10) and `MtpCost::dflash` (spec
  19c: no more +0.22 a row); host tests of `AdaptiveK` on the new table. Commit
  `git commit -S -m "spec: the measured small-M verify costs replace the extrapolated cost table (spec 28b)"`.

### Task 5: the record

- [ ] **Step 1:** spec 28 "28b as built" (deviations, each with its cost if wrong); spec 8 §4 M2's note (verify rows no
  longer bitwise M = 1 from 28b on; the bars unchanged); BENCHMARKS rows (the verify-cost table, S3); `docs/README.md`.
  Commit `git commit -S -m "docs: spec 28b as built, the verify-cost rows"`.

### Task 6: the box queue row

- [ ] **Step 1:** row **69** and its block in `tools/box_validate/stages.sh`
  (`python3 tools/box_validate/test_box_validate.py` passes):

```text
row 69 "spec 28b - gemm_sm in production: every M >= 2 decode list (MTP / lookup verify, DFlash tap verify and drafter, --pp 2 stage verify, 13a rows) on the small-M DPAS GEMM, M = 1 bitwise; N1-N8, S2-S3, the measured cost table (plan 28b)"
stage r69.k0      69 default cpu  -           g0.sha,g0.bitwise,g0.suite "N1: kernel_cmdlines +N / -0 / ~0, every existing binary identical, the suite (G0)"
stage r69.host    69 default cpu  -           -                     "gemv_kind_test, gemm_sm_ref_test, the AdaptiveK cost-table tests"
stage r69.kernels 69 default card -           r69.k0                "gemm_sm_test: N2 against the fp64 reference and gemv, N3 property R, every shape and M"
stage r69.gates   69 default card qwen        r69.kernels           "N4 (mtp_gpu_test M2 at M 2..8), N5 (dflash_engine_test T1 / P when rows 54-56 pass), N6 greedy vs --spec off under the tie rule (golden + A4), N8 replay"
stage r69.speed   69 optin   gpu  qwen        r69.gates             "S2: probe_mtp_steps / probe_verify_m verify(M) M 1..8 (16); S3: mtp_d1.sh greedy 4096; idle, interleaved, median of 3"
rownote 69 "N1 is the gate that is never traded: M = 1 lists bind the old binaries. Hand back: the N2 / N3 lines, the M2 cosines per M, the tie census, the verify(M) table and the D1 table."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 69 - the small-M GEMM build (spec 28b)"`.

**Gate for the plan:** host tests green and `tools/mac_check.sh --base main` PASS (cmdlines additions only); on the box
N1-N8 pass and S2 holds (**verify(8) <= 1.4 plain steps**, M = 1 within noise); S3 recorded against its bar (+10 % on
the tool-call geomean - a miss is recorded and goes to the operator, it does not revert the kernel if S2 holds).
