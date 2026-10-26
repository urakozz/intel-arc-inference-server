# Spec 28a - small-M GEMM P0: Qwen3.8's GEMV shapes at M = 1 / 2 / 4 / 8 / 16, the verify step's attribution, the go / no-go line

**Status (2026-10-27): PLANNED - the first plan of the Qwen speed track; starts when the operator says go.** Nothing
built. Box queue row **68** (the next free row after 67 on 2026-10-27; spec 28's header lists the track's rows 68-75).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the numbers spec 28 §6's rule needs, and the rule applied: each of Qwen3.8's decode linears timed at
M = 1 / 2 / 4 / 8 / 16 through today's kernels and through stand-in `gemm_sm` arms (bf16 DPAS; i8 x i4 DPAS), the real
verify list's time split into GEMV and non-GEMV families at M = 1..8, the stand-in's numerics against a host reference,
and the GO / NO-GO line in spec 28 §9.

**Architecture:** spec 28 §1-§3, §6. A new probe `tools/probe/probe_small_m` (C++ / Level Zero, one card) with its
stand-in kernels in `tools/probe/probe_small_m.cl` (spec 28 §2's kernel: dequant once per 64-row group, DPAS over row
blocks of 8, the group scale in fp32, split-K partials in `gemv`'s layout; compile-time M, K, N, S, LAYOUT, TYPE =
bf16 | i8i4); today's `gemv` / `gemv_i8w` binaries as the control (at M = 16 a probe-only `gemv` build - `gemv.cl` is
generic in M; production stops at 8); a host fp64 reference shared with 28b (`tests/kernels/gemm_sm_ref.h`).
`probe_verify_m`'s profiled-list section (1b) gains M = 5..8 through the M8 lists `probe_mtp_steps` already builds
(`B70_VERIFY_M8`).

**Tech Stack:** C++17, OpenCL C (Intel sub-group matrix builtins), Level Zero, CMake.

**Spec:** `docs/superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md`. Precedents:
`tools/probe/probe_gemv.cc` + docs/probe-gemv-2026-08-24.md (the shape sweep's interleaving), `probe_verify_m` and
docs/probe-mtp-2026-09-27.md (the per-family table this plan extends), `probe_w8a8` (layer-63 captures, the i8 path,
the timing protocol of docs/probe-w4a8-2026-09-23.md §14.4), `probe_dpas_rate.cl` (the builtins that compile).

## Dependencies and branch points

- Main (2e4b205 or later). Branch **`spec28a-small-m-probe`** from main.
- **Box:** the gate checkpoint (data key `qwen`, int8 head); one idle card. Real rows for the numerics come from the
  checkpoint itself: layer 63's weights, and activation rows read back from the engine's decode scratch after a prefill
  (the probe owns an engine as `probe_verify_m` does) - no capture files needed.
- `r29.cost` (verify M = 5..8, plan 19a Task 4) is folded in: if it has not run, Task 3 runs it.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed. No `rm -rf`.
- **No engine code changes:** `tools/kernel_cmdlines` +0 / -0 / ~0 for `src/` (the stand-ins and the M = 16 `gemv`
  are `tools/probe`'s); G0's sha is the check. `probe_verify_m`'s new arms are probe code.
- Timed arms: **idle box** (both cards free of DRM holders, `uptime` and the idle grade recorded), device 0, the GPU
  lock, interleaved pairs after a warm-up, **median of 5** with the range for kernels, of 3 for whole steps. Detached
  and polled (`tools/probe/detach.sh`).
- Every number written with its grade (spec 28's legend).

## Review Focus

1. **Like for like:** the control and the stand-in read the same weight bytes (the production layout per shape, L1 for
   `qkv||z`, L0 for the rest), the same activations, and write the same partials; S is the production S unless the
   arm says otherwise.
2. **Dequant once:** the stand-in's dequant instruction count must not grow with M (check the ISA dump at M = 2 / 8 /
   16 - `ocloc -options -dump`), or the arm measures the wrong kernel.
3. **The whole step, not the kernel:** the go rule's (b) is the real list's non-GEMV time plus the GEMM sum; the
   profiled list's flush per launch inflates absolutes, so shares from it are applied to the unprofiled step time.
4. **Numerics on real weights:** layer 63's weights and real activation rows, not random data alone; property R (rows bitwise across
   M >= 2 and neighbours) checked, not assumed.

---

### Task 1: the probe, the stand-ins and the host reference (Mac)

**Files:**
- Create: `tools/probe/probe_small_m.cc`, `tools/probe/probe_small_m.cl`, `tests/kernels/gemm_sm_ref.h` (fp64
  reference of int4 g64 x bf16 and int8-per-column x bf16, both layouts), `tests/kernels/gemm_sm_ref_test.cc` (the
  reference against `gemv`'s host twin on small shapes; the error bound of spec 28 N2 computed per output)
- Modify: `tools/probe/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/probe/probe_verify_m.cc` (section 1b at
  M = 5..8), `tools/mac/opencl/intel_shim.h` only if the mixed i8 x i4 builtin is missing from its declarations (a
  declaration, not an emulation)

**Interfaces:** `probe_small_m shapes|numerics [--shape qkvz|out|qkv|o|gateup|down|head|ab|all] [--m 1,2,4,8,16]
[--type bf16|i8i4] [--s N] [--rounds 5] [--device 0]`.

- [ ] **Step 1: failing tests** - `gemm_sm_ref_test` (the reference = `gemv`'s host twin within 1 fp32 ulp of the sum;
  layouts 0 / 1; the head's per-column scale).
- [ ] **Step 2: implement, PASS;** the stand-in kernels pass `tools/mac/cl_syntax.sh` (the Mac has no DPAS: numerics
  run on the box, Task 4); `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "probe: probe_small_m - stand-in small-M DPAS GEMMs (bf16, i8xi4) and their fp64 reference (spec 28a)"`.

### Task 2: the shape table (box, `r68.shapes`)

- [ ] **Step 1:** every shape (spec 28 §6's list, with its layer count) at M = 1 / 2 / 4 / 8 / 16: control against
  each stand-in type, interleaved pairs, median of 5; ms per call, GB/s against the shape's bytes, the ratio to the
  control at M = 1. Then at M = 8 the S sweep for each shape (S dividing K / 64). Record in
  `docs/probe-small-m-gemm-<date>.md` §1. Commit
  `git commit -S -m "probe: Qwen3.8's decode linears at M = 1..16, gemv against small-M DPAS (spec 28a)"`.

### Task 3: the verify step's attribution (box, `r68.attrib`)

- [ ] **Step 1:** `probe_verify_m --depth 4096` and `--depth 32768` with the M8 lists: the unprofiled verify step at
  M = 1..8 (interleaved against M = 1) and the profiled per-family table (int4 GEMV, the int8 head, attention,
  `gdn_step`, everything else) at M = 1..8 - the 2026-09-27 table extended to 8 rows with today's attention (v2) and
  head (int8). If `r29.cost` has not run, run `probe_mtp_steps <snap> 4096 32 3 int8 off 16384 8` in the same session.
  Record §2. Commit `git commit -S -m "probe: the verify step at M = 1..8, GEMV against non-GEMV (spec 28a)"`.

### Task 4: numerics (box, `r68.numerics`)

- [ ] **Step 1:** `probe_small_m numerics`: each stand-in type against the fp64 reference and against `gemv` at the same
  M on layer 63's weights with real and random rows - spec 28 N2's bound per output (max / median recorded); property R at
  M = 2..8 / 16 (10^3 trials a shape: rows bitwise across M and with the other rows randomised). For i8 x i4 also the
  logit-level error on the four prompts by docs/probe-w4a8-2026-09-23.md §15.4's method (a CPU run in the oracle image
  on the dumped rows), so the type rule of spec 28 §2 can be applied. Record §3. Commit
  `git commit -S -m "probe: small-M DPAS numerics and row independence (spec 28a)"`.

### Task 5: the verdict

- [ ] **Step 1:** spec 28 §9: the tables; the derived verify(8) = Task 3's non-GEMV time at M = 8 (shares applied to
  the unprofiled step) + Task 2's GEMM sum at M = 8 (each shape x its layers + the head), in plain steps; the type and S
  per shape chosen; and **the line** by spec 28 §6:
  - **GO** when (a) the GEMM sum at M = 8 <= 1.25x the M = 1 sum through `gemv`, (b) derived verify(8) <= ~1.4 plain
    steps, (c) N2 and N3 pass;
  - **the operator's call** when (b) is between 1.4 and 1.6;
  - **NO-GO** when (c) fails, (b) > 1.6 or (a) > 1.5 - the record names the next lever from the attribution.
  `docs/README.md`: the probe record's line. Commit `git commit -S -m "spec 28: small-M GEMM P0 verdict - <GO|NO-GO|operator's call>"`.

### Task 6: the box queue row

- [ ] **Step 1:** row **68** in `box-validation-queue.md` and its block in `tools/box_validate/stages.sh`
  (`python3 tools/box_validate/test_box_validate.py` passes):

```text
row 68 "spec 28a - the small-M GEMM P0: Qwen3.8's decode linears at M = 1 / 2 / 4 / 8 / 16 (gemv vs bf16 / i8xi4 DPAS stand-ins), the verify step's GEMV / non-GEMV split at M = 1..8, numerics and row independence, the go / no-go line (plan 28a)"
stage r68.k0       68 default cpu  -           g0.sha,g0.bitwise,g0.suite "G0: no src/ binary changes (the stand-ins are tools/probe's)"
stage r68.host     68 default cpu  -           -                     "gemm_sm_ref_test"
stage r68.shapes   68 default gpu  qwen        r68.k0                "probe_small_m shapes: every shape at M 1/2/4/8/16, gemv vs bf16 and i8xi4 stand-ins, then the S sweep at M 8; idle, interleaved, median of 5"
stage r68.attrib   68 default gpu  qwen        r68.k0                "probe_verify_m at depth 4096 / 32768 with the M8 lists: the verify step and the per-family table at M 1..8 (and r29.cost's arm if it has not run)"
stage r68.numerics 68 default card qwen        r68.k0                "probe_small_m numerics: N2 against the fp64 reference and gemv, property R; the i8xi4 logit error (oracle image)"
stage r68.verdict  68 manual  -    -           r68.shapes,r68.attrib,r68.numerics "spec 28 §9: the GO / NO-GO line by spec 28 §6"
rownote 68 "Timed stages need the idle protocol (both cards free of DRM holders). Hand back: the shape table, the attribution table, the numerics lines; they fill docs/probe-small-m-gemm-<date>.md and spec 28 §9."
```

- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 68 - the small-M GEMM P0 (spec 28a)"`.

**Gate for the plan:** `gemm_sm_ref_test` green and `tools/mac_check.sh --base main` PASS on the Mac (cmdlines ~0 for
`src/`); on the box the shape, attribution and numerics tables recorded on an idle card, and the verdict line written in
spec 28 §9. Hand back with the tables and the line.
