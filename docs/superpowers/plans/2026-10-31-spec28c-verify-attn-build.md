# Spec 28c - the verify step's attention on DPAS, chosen by shape: the P0 line, `attn_verify_dpas`, the measured table, the rule, the gates

**Conditions:** this plan starts only when **spec 28b is merged** (the capture's `gemv_kind(M)` and every M >= 2 list
on `gemm_sm`), **Task 0 below writes GO** from 28a's attribution on 28b's step, **and the operator says go** (one
implementing plan at a time). If spec 28 was parked at 28a's NO-GO, Task 0 still runs on the measured step without 28b
and its line goes to the operator instead of deciding (Open question 2).

**Status (2026-10-31): PLANNED - after 28b.** Nothing built. **No box queue row** (operator, 2026-10-31: no new rows
now); the box steps run as named jobs, and the stages to add later are described at the end in prose. Nothing is added
to `tools/box_validate/stages.sh` or `box-validation-queue.md` by this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** a DPAS verify attention, `attn_verify_dpas` (bf16 KV and int8 KV), writing v2's partials so the existing
`attn_reduce_v2` / `attn_reduce_v2_kv8` finish it; a table of v2 against it measured on the box over M = 2..8 x depth
512 / 4096 / 32768 / 130816 x bf16 / int8 KV; the rule `verify_attn_kind(M, keys, kv)` built from that table beside
28b's `gemv_kind(M)`; N1-N3 and S1-S2 of spec 28c on the card; M = 1 and every existing binary untouched.

**Architecture:** spec 28c §2-§3. One new source `src/kernels/attn_verify_dpas.cl`: grid (4 kv heads,
`kAttnV2Blocks` = 32) as v2, v2's per-row stride `ppw(L)` and row groups (at most two strides a launch), v2's partials
`[24][32][M][258]`; inside a block the 6 x M (q head, row) pairs of the kv head's GQA group as DPAS A rows in blocks of
8, K^T / V tiles by `pf_flash_attn`'s 2D block idioms (transposed 32-bit read for K, VNNI-transform read for V), S in the
A-operand layout so P goes to PV without data movement, fp32 online softmax per row with `exp`, the causal bound
`pos + m` per row (masked -> -INF -> weight 0, neutral as in v2). Compile-time levers, one define each, swept by the
probe from the same source: `KT` (keys per tile: 16 / 32 / 64), `PACK` (`g`: (head, row) pairs packed across row
blocks; `h`: one sub-group per q head, `pf_flash_attn`'s packing), `KS` (key slices per work-group, merged in SLM in a
fixed order: 1 / 2 / 4 / 8), `DSPLIT` (output dims a sub-group owns: 256 / 128), `QSPLIT` (0: q rounded to bf16; 1: q
= hi + lo, two QK^T DPAS), `GRF` (128 / 256, the option on the ocloc line), `KV8` (0 / 1: int8 K / V as exact bf16
operands, the key scale on the score and folded into P, `pf_flash_attn_kv8`'s form), and `EAGER` (1: plain loads and a
scalar tile product in the same tile, order and partial layout - the Mac build; never a production line). The reduce is
the existing `attn_reduce_v2` entry of `attn_v2_M<M>_T32` (or `attn_reduce_v2_kv8` of `_KV8`): no new reduce binary.
The rule lives in a header-only `src/runtime/verify_attn.h` (host-testable on the Mac), used by the capture's
`attn_pair` beside `gemv_kind`; `B70_VERIFY_ATTN=v2|dpas` is read at each capture, for probes only.

**Tech Stack:** C++17, OpenCL C (Intel sub-group matrix and 2D block builtins), Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-31-spec28c-verify-attention-by-shape-design.md` (§8: what this plan corrected
in it against the code). Precedents: spec 10 (`attn_v2.cl`, the per-row stride and its bitwise row property, A1 / A3,
`probe_decode_attn` and docs/probe-decode-attn-2026-09-28.md - the capture of real q / KV and the interleaved arm
timing); spec 6 (`pf_flash_attn.cl`, the DPAS / 2D block idioms and the 256-GRF build); spec 12b (`kv8.cl`, the
rotkv format and `src/common/kv8.h` its bitwise host twin); spec 28a / 28b (the plan form, `gemv_kind`, property R).

## Dependencies and branch points

- Main with 28b merged (`gemv_kind(M)`, `gemm_sm`, spec 28 §9 and "28b as built"). Branch **`spec28c-verify-attn-build`**
  from main.
- **Records read, not re-run:** spec 28 §9 / docs/probe-small-m-gemm-<date>.md §2 (28a's per-family verify table at
  M = 1..8, depth 4096 / 32768) and 28b's measured verify(M) (spec 28 "28b as built", BENCHMARKS).
- **Box:** the gate checkpoint (data key `qwen`, int8 head); one idle card; a build with `B70_MTP`, `B70_VERIFY_M8`,
  `B70_KV8` (the defaults); the long prompt `tests/golden/prompts/long32k.ids`; the golden and A4 sets. DFlash's gates
  only if rows 54-56 have passed by then (otherwise recorded as pending, as 28b does).

## Global Constraints

- Signed commits after every step (`git -c gpg.program=$(brew --prefix)/bin/gpg commit -S` on the Mac); no merge, no
  push. Box tree automatic; `tools/box.env` copied if missing, never committed. No `rm -rf`.
- **Additive only:** `tools/kernel_cmdlines` shows **+N / -0 / ~0** - the `attn_verify_dpas_M<M>_T32[_KV8]` lines and
  nothing else in `src/`; every existing binary checksum-identical (G0). `attn_v2.cl` and `kv8.cl` are not edited.
- **M = 1 never switches:** every list at M = 1 binds today's v2 pair, whatever the table or the env says (the rule
  returns V2 at M = 1 before it reads either).
- **The rule is Qwen3.8's heads only** (24 q / 4 kv; Agnes shares them): any other `fa_q_heads` / `fa_kv_heads`
  (Ornith 16 / 2) gets V2. The 13a row builds (`Mode::Rows`, `attn_pair_rows`) are not touched.
- **No engine flag.** `B70_VERIFY_ATTN` is a probe override (unset = the rule), read at each capture as
  `B70_DECODE_ATTN` is, named on the engine line when set, absent from `--help`.
- Timed arms: **idle box** (both cards free of DRM holders, `uptime` and the idle grade recorded), device 0, the GPU
  lock, interleaved pairs after a warm-up, **median of 5** with the range for kernels, of 3 for whole steps. Detached
  and polled (`tools/probe/detach.sh`).
- **256-GRF builds carry the option on their ocloc line** (`-cl-intel-256-GRF-per-thread`, as `pf_flash_attn`), and
  the probe reads the built binary's GRF mode back before a timed arm (spec 28 §2.1's pitfall).
- Every number written with its grade (spec 28's legend).

## Review Focus

1. **v2's blocking, exactly:** the DPAS kernel's work-group (j, blk) walks, for each row, the same positions
   `[blk ppw(L), min((blk + 1) ppw(L), end))` that `attn_decode_v2` walks, with tiles anchored at the block start; its
   partial for (q head, blk, m) has v2's meaning. Otherwise `attn_reduce_v2` merges the wrong blocks silently (the nb it
   reads is `(pos + m) / ppw(pos + m + 1) + 1`).
2. **Property R by construction:** `KT`, `KS`, `DSPLIT`, `QSPLIT` and the in-work-group merge order are the same for
   every M >= 2 of a KV form (one lever setting per form, chosen at M = 8 and costed at the other M); a row's result
   must not depend on its slot in the row block. Checked bitwise, not assumed (N2a).
3. **The table is what binds:** the rule's row for (M, KV form) is the measured record's verdict, the threshold a
   measured depth, ties to v2; the in-list check (S1) confirms the kernel-level table on the real step at the switching
   shapes.
4. **The env override cannot leak:** unset in every gate except the arms that name it; `B70_VERIFY_ATTN=dpas` at M = 1
   is still v2; a value other than `v2` / `dpas` is refused at capture with the accepted values.

---

### Task 0: the P0 line (Mac, records only)

**Files:** Modify `docs/superpowers/specs/2026-10-31-spec28c-verify-attention-by-shape-design.md` (a new §9 "P0
line").

- [ ] **Step 1:** from 28a's per-family table (profiled shares) and 28b's measured verify(M) (unprofiled): for
  M = 2 / 4 / 8 at depth 4096 and 32768 - attention's share (`attn_prep` + `attn_decode_v2` + `attn_reduce_v2` over the
  16 FA layers), its time on 28b's step (share x the unprofiled step, Review Focus 3 of 28a), the M = 1 attention at the
  same depth and the **excess** (M less M = 1), each graded (derived). If 28b measured no step at 32768, derive it as
  28a Task 5 did (28a's non-GEMV time at 32768 + 28b's GEMM sum). **GO** when the excess at M = 8, depth 32768 is
  >= 15 % of verify(8) there; else **PARKED** (the record names the next lever from the same table). Write §9 and stop
  the plan on PARKED. Commit `git commit -S -m "spec 28c: the P0 line - attention excess <x> % of verify(8) at 32768 (<GO|PARKED>)"`.

### Task 1: the host reference (Mac)

**Files:**
- Create: `tests/kernels/attn_verify_ref.h` - fp64 attention of one kv head's GQA group at M rows from v2's inputs:
  fp32 q `[M][24][256]`, bf16 K / V `[keys][4][256]` or int8 + fp16 scales (dequantised by `src/common/kv8.h`, the
  writer's twin, q already rotated as `attn_prep_kv8` leaves it); row m attends to keys `[0, pos + m]`, scale 1/16; the
  pre-gate output in fp64 and the gated bf16 row exactly as `attn_reduce_v2` forms it (`rne(rne(o) x sigmoid(g))`;
  under int8 the un-rotation first, `kv8_ref.h`'s `unrotate64`); the per-row partial view
  `blocks(pos, m)` = v2's `ppw` blocking (for the kernel's partials check); `cos_q(a, b)` per (q head, row).
- Create: `tests/kernels/attn_verify_ref_test.cc` (host) - the reference against `attn_ref.h`'s v1 chain (`decode` +
  `reduce`, M = 1..4, synthetic KV at 512 / 2049 / 4096 keys) and `kv8_ref.h`'s `attend` on int8: cosine >= 0.9999999
  per (q head, row), the gated bf16 rows equal or 1 bf16 ulp apart and that count recorded; `blocks()` equal to
  `attn_v2.cl`'s `v2_ppw` / nb formulas at L = 1, 64, 2048, 2049, 4096, 32768, 130824 (the stride's steps).
- Modify: `tests/CMakeLists.txt` (the test, no device dependencies - derived as a host test by
  `cmake/host_only.cmake`).

- [ ] **Step 1: failing test** `attn_verify_ref_test` (the header declared, bodies absent): it fails to link.
- [ ] **Step 2: implement, PASS;** `tools/mac_check.sh --base main --quick` green (host section shows the new test
  passing). Commit `git commit -S -m "test: attn_verify_ref - the fp64 reference of the verify attention at M rows, bf16 and int8 KV (spec 28c)"`.

### Task 2: the kernel and its names (Mac)

**Files:**
- Create: `src/kernels/attn_verify_dpas.cl` (header in `attn_v2.cl`'s style: the grid, the partials, the blocking
  borrowed from v2, the tile orders, every lever and which combinations `#error`; the property R argument; the int8
  form; `EAGER`).
- Modify: `src/kernels/kernels.h` - beside `attn_v2_variant`:
  `attn_verify_dpas_variant(unsigned M, unsigned T, bool kv8, unsigned q = kRefFaQHeads, unsigned kv = kRefFaKvHeads)`
  -> `attn_verify_dpas_M<M>_T<T>[_KV8]` (+ `fa_suffix`, which is empty at Qwen3.8's heads).
- Modify: `src/kernels/CMakeLists.txt` - a block after `B70_VERIFY_M8` / `B70_KV8`: `attn_verify_dpas_M<M>_T32` at
  M = 2..8 and `_KV8` at M = 2..4 (the int8 decode lists that exist), behind `option(B70_VERIFY_DPAS ... ON)`, with the
  lever defines of Task 4's choice (until then the provisional `KT=32 PACK=g KS=4 DSPLIT=256 QSPLIT=1 GRF=256`, written
  as such in the comment and replaced in Task 5).
- Modify: `tests/kernels/variant_names_test.cc` (the names and their CMake lines agree), `tools/mac/opencl/intel_shim.h`
  only if a 2D block shape `pf_flash_attn` / `pf_gemm` do not already use is needed - and then only after ocloc on the
  box has built it (doc 18's rule; the commit says so).
- Create: `tools/mac/clrun/attn_verify_run.cc` and its name in section 5's loop of `tools/mac_check.sh`: the `EAGER=1`
  build (no sub-group function the emulation shim leaves undeclared - no shuffles, reductions, DPAS or 2D block I/O)
  against `attn_verify_ref.h` at M = 2 / 4 / 8, keys 512 and 2049 (two strides in one launch) and the cache end: the
  partials per block against the reference's block view, the partials merged on the host in `attn_reduce_v2`'s order
  (`attn_ref.h`'s `reduce` order; `attn_v2.cl` itself cannot build on the Mac - `attn_decode_v2` in the same program
  uses shuffles and 2D block reads) and the gated rows within N1's bar, and every row bitwise across M = 2..8 and with
  the other rows randomised (property R in the eager build - INDICATIVE, the card's DPAS is Task 6's).

- [ ] **Step 1: failing test** - `variant_names_test` asks for the names; `attn_verify_run` builds against the reference
  and fails (no kernel).
- [ ] **Step 2: implement;** `tools/mac/cl_syntax.sh` passes every new command line (the production lines, the eager
  build, and the probe's lever lines of Task 3); `tools/mac_check.sh --base main --kernels` green with cmdlines
  **+N / -0 / ~0** and `attn_verify_run` exact where the reference's order is the kernel's, within N1's bar elsewhere.
  Commit `git commit -S -m "kernels: attn_verify_dpas - the verify attention on DPAS in v2's blocking and partials, bf16 and int8 KV, M 2..8 (spec 28c)"`.

### Task 3: the probe (Mac)

**Files:**
- Modify: `tools/probe/probe_decode_attn.cc` - `capture` gains `--rows R` (default 4 as today; up to 8): it keeps the
  q / gate / production `attn_out` of R plain steps from each depth (spec 10's capture at R rows) and the KV past
  `deepest + R`. Depth list for this plan: 512, 4096, 32768, 130816 at max_len 131072.
- Create: `tools/probe/probe_verify_attn.cc` -
  `probe_verify_attn arms <dump> [--m 2,3,4,5,6,7,8,16] [--depths 512,4096,32768,130816] [--kv bf16|int8]
  [--arms v2,pfa,<tags>] [--rounds 5] [--reps 10] [--time] [--device 0]`.
  Arm `v2`: the production `attn_v2_M<M>_T32` (decode + reduce), and for int8 at M = 5..8 the probe-built control
  `pvd_v2kv8_M<M>` (`kv8.cl` `KV8_DEC=1` at that M - production stops at 4; no M = 16 control exists, the DPAS arm at
  16 is recorded alone). Arm `pfa`: `pf_flash_attn` at C = M (bf16 q, no split - a reference, recorded only). Arms
  `<tag>`: `attn_verify_dpas.cl` at `K<KT>_P<g|h>_S<KS>_D<DSPLIT>_Q<QSPLIT>_G<128|256>` (`pvd_<tag>_M<M>[_KV8]`),
  followed by the production `attn_reduce_v2` of that M. The int8 inputs are made on the host from the dump
  (`src/common/kv8.h`: K / V quantised, q rotated) and are bitwise what the writer stores. Per arm, M and depth: N1
  (cosine per (q head, row) against `attn_verify_ref.h`, max abs, v2's on the same inputs beside it), N2a (rows
  bitwise across M and with neighbours randomised, 10^2 trials a depth), N2b (against v2's M = 1 row at `d + m`),
  bitwise repeatability; with `--time`, rounds of `--reps` launches per arm, arms interleaved in a rotated order after a
  warm-up, per-kernel L0 timestamps (the attention pair: decode + reduce), median and range per arm and **the median
  paired ratio v2 / arm** with its range.
- Create: `tools/probe/verify_attn_probe.sh` (`capture | arms ...`, as `decode_attn_probe.sh`: `ZE_AFFINITY_MASK=0`,
  `uptime` before and after, the GPU lock by the caller).
- Modify: `tools/probe/CMakeLists.txt` - `probe_verify_attn`, the `pvd_*` lever builds from
  `src/kernels/attn_verify_dpas.cl` (the tag list and the harness's `--arms` are two lists nothing checks against each
  other: a missing binary is named and skipped, as `probe_decode_attn`), the `pvd_v2kv8_M5..8` controls.

- [ ] **Step 1: implement;** `tools/mac/l0_syntax.sh` (via `tools/mac_check.sh --base main --quick`) and
  `cl_syntax.sh` on every `pvd_*` line pass; cmdlines still +N / -0 / ~0 for `src/` (the probe's lines are
  `tools/probe`'s). Commit `git commit -S -m "probe: probe_verify_attn - v2 against attn_verify_dpas lever arms on real q / KV at M rows, numerics and paired timing (spec 28c)"`.

### Task 4: the levers and the measured table (box)

- [ ] **Step 1: capture.** `tools/probe/detach.sh ~/vattn-capture.log flock ~/b70-gpu.lock
  tools/probe/verify_attn_probe.sh capture --rows 8 --depths 512,4096,32768,130816` (the checkpoint at max_len 131072;
  record the dump's sha in the log).
- [ ] **Step 2: levers at M = 8, depth 32768, bf16** (then int8): `KT` 16 / 32 / 64, `PACK` g / h, `KS` 1 / 2 / 4 / 8,
  `DSPLIT` 256 / 128, `QSPLIT` 0 / 1, `GRF` 128 / 256 - one at a time from the provisional setting, each arm with N1 /
  N2a / N2b and `--time`. The setting kept per KV form is the fastest **that passes N1 and N2**; it is then costed at
  M = 2 / 4 at depth 4096 and 32768 against the per-M best (the price of one setting for every M, Review Focus 2).
  `QSPLIT=0` is kept only if it passes N1 at every depth; otherwise 1.
- [ ] **Step 3: the table** with the kept settings: M = 2..8 x depth 512 / 4096 / 32768 / 130816 x bf16 / int8, arms
  `v2,<kept>`, `--time --rounds 5`; M = 16 and `pfa` recorded at 4096 and 32768. Where a verdict changes between two
  measured depths, the geometric midpoints (1448, 11585, 65536) at that M, so the threshold is placed within a factor
  of 2. Record in **docs/probe-verify-attn-<date>.md**: §1 the levers, §2 the table, §3 numerics. The table's form:

```text
| M | KV   | depth  | v2 ms [range]     | dpas ms [range]   | v2 / dpas paired [range] | verdict |
|---|------|--------|-------------------|-------------------|--------------------------|---------|
| 8 | bf16 | 32768  | x.xxx [x.xx-x.xx] | x.xxx [x.xx-x.xx] | 1.xx [1.xx-1.xx]         | dpas    |
```

  **Verdict rule (the noise line):** `dpas` when the median paired ratio v2 / dpas is >= 1.03 **and** the low end of
  its range is > 1.00; otherwise `v2` (a tie keeps v2). `dpas_from` for (M, KV) = the smallest measured depth whose
  verdict is `dpas` with every deeper measured depth `dpas` too; none -> never; the shallowest measured depth (512)
  -> 0 (every depth).
- [ ] **Step 4: commit** `git commit -S -m "probe: verify attention v2 against DPAS at M 2..8 x depth 512..130816, bf16 and int8 KV - the levers, the table, numerics (spec 28c)"`.

### Task 5: the rule in the capture (Mac)

**Files:**
- Create: `src/runtime/verify_attn.h` (header-only, no L0) - the table and the rule, the measured cells beside each row
  as comments citing the record:

```cpp
enum class VerifyAttn : uint8_t { V2, Dpas };
inline constexpr uint32_t kDpasNever = UINT32_MAX;
struct VerifyAttnRow { uint32_t m; bool kv8; uint32_t dpas_from; };   // keys of row 0 (pos + 1)
// docs/probe-verify-attn-<date>.md §2; v2 / dpas paired at 512 | 4096 | 32768 | 130816
inline constexpr VerifyAttnRow kVerifyAttnTable[] = {
    {2, false, kDpasNever},   // 0.98 | 1.01 | 1.07 | 1.12  (example form only - the record's numbers)
    // ... M = 2..8 bf16, M = 2..4 int8
};
// V2 at M = 1, at any heads but 24 / 4, for an (M, kv) the table lacks, and under B70_VERIFY_ATTN=v2;
// Dpas under B70_VERIFY_ATTN=dpas (M >= 2, a built binary); else the table.
VerifyAttn verify_attn_kind(uint32_t M, uint32_t keys, bool kv8, uint32_t q_heads, uint32_t kv_heads,
                            VerifyAttnOverride ov);
VerifyAttnOverride verify_attn_override();   // B70_VERIFY_ATTN: unset | v2 | dpas; anything else throws
bool verify_attn_depth_free(uint32_t M, bool kv8);   // dpas_from is 0 or never: a capture-time decision
```

- Modify: `src/runtime/capture.cc` - next to 28b's `gemv_kind` use: `attn_pair` binds `attn_verify_dpas` + the v2
  binary's `attn_reduce_v2` (`_kv8`) when `kCapM >= 2`, not `Mode::Rows`, and the rule says Dpas; `check_sizes` requires
  the DPAS binary when it will be bound; the override read once per build (as `decode_attn()`); `src/kernels/CMakeLists.txt`'s
  provisional lever line replaced by Task 4's kept settings.
- **Depth (spec 28c §3), by the table:** if `verify_attn_depth_free` holds for every built (M, KV), the capture decides
  by M (one list per M, nothing else changes). If some M has a finite `dpas_from > 0`: `build_verify` /
  `build_verify_taps` / `build_stage_verify*` take the attention kind as an argument (default: the rule at `keys` = 1);
  `Engine` captures that M's list twice (`verify_steps_` and `verify_steps_deep_`) and `Engine::verify` (and the DFlash
  path through it, `PipelineEngine`'s stage verify) submits the deep list when `pos + 1 >= dpas_from`; the capture time
  and list memory of the second lists are printed at engine start and recorded in Task 7.
- Create: `tests/runtime/verify_attn_test.cc` (host, header-only rule): M = 1 -> V2 under every override and table;
  Ornith's 16 / 2 -> V2; the override parse (unset, v2, dpas, a bad value refused); each table row's threshold at
  `dpas_from - 1` / `dpas_from`; `verify_attn_depth_free`; every (M, kv) the rule can return Dpas for has a CMake line
  (the names test's list passed as arguments, as `verify_m8_names_test`).
- Modify: `tests/CMakeLists.txt`; the engine's capability line (names `B70_VERIFY_ATTN` when set, as
  `B70_DECODE_ATTN`).

- [ ] **Step 1: failing test, then implement, PASS** (host); `tools/mac_check.sh --base main --quick` green, cmdlines
  +N / -0 / ~0. Commit `git commit -S -m "runtime: verify_attn_kind - the measured M x depth table beside gemv_kind; M >= 2 verify lists bind attn_verify_dpas where it wins, M = 1 keeps v2 (spec 28c)"`.

### Task 6: the gates on the card (box)

**Files:** Create `tests/kernels/attn_verify_dpas_test.cc` (device, `attn_v2_test`'s form) and its `tests/CMakeLists.txt`
lines.

- [ ] **Step 1: `attn_verify_dpas_test`.** Part S (synthetic, always): random KV of 4096 rows, the positions of
  `attn_v2_test` (depth 0 -> 1, block edges, 2048 -> 2049, the cache end), M = 2..8 bf16 and M = 2..4 int8: **N1**
  against `attn_verify_ref.h` (cosine >= 0.99999 per (q head, row), v2's error recorded beside it), **N2a** (rows
  bitwise across M = 2..8 and with neighbours randomised, 10^3 trials a shape), **N2b** (against v2 M = 1 at `pos + m`,
  cosine >= 0.99999), two runs bitwise. Part D (with Task 4's dump): the same at the four depths.
- [ ] **Step 2: the lists.** `mtp_gpu_test`'s M2 at M = 2..4 and the M8 lists' M = 5..8 rows (logits and GDN slot
  cosine >= 0.99999 against M = 1, the 64-token continuation - spec 28 N4's bars) **with the rule and with
  `B70_VERIFY_ATTN=dpas`**; the greedy gates with `--mtp 1..3` and `--spec lookup` against `--spec off` on golden + A4,
  256 tokens, zero non-tie divergences and the tie census beside 28b's (**N3**), both settings; `--spec dflash` when rows
  54-56 have passed; spec 10 A3's form: 256 greedy ids at 32k from `long32k.ids` with `--mtp 3`, rule against
  `B70_VERIFY_ATTN=v2`, identical except near-ties (a divergence classified by `probe_decode_attn fork`'s method - the parting step's top-3
  logits under both arms - with the arm switch made by `B70_VERIFY_ATTN` instead of `B70_DECODE_ATTN`); `--kv-cache int8`
  passkey 3/3 at 5 / 50 / 95 % of 120k with `--mtp 3`; replay determinism with the DPAS kernel bound; the full suite and
  G0 (existing binaries identical, `attn_v2_test` and `kv8_kernels_test` unchanged). Commit
  `git commit -S -m "test: attn_verify_dpas gates on the card - N1, property R, against M = 1, the MTP / lookup gates (spec 28c)"`
  with the test and any test edits.

### Task 7: speed (box)

**Files:** Modify `tools/probe/probe_mtp_steps.cc` (a 9th positional `mode`: `default` | `vattn`) and
`tools/probe/mtp_d1.sh` (`VATTN=1`: each K arm runs as two processes, `B70_VERIFY_ATTN=v2` and unset, their order
alternating by round).

- [ ] **Step 1: the in-list pairs.** `probe_mtp_steps <snap> <depth> 32 3 int8 off <max_len> 8 vattn`: for each M = 2..8
  the verify list built twice over the probe's 8-slot `MtpBuffers` (`setenv("B70_VERIFY_ATTN", "v2")` then unset, each
  before its `build_verify` - the override is read at each capture), pos rewound to `depth` before each arm, interleaved
  pairs (v2 list, rule list; the pair's order alternating by round) after the warm-up, median of 3 with the range.
  Depths 4096 (max_len 16384), 32768 (65536), 120000 (131072); plus 512 at the shapes the table switches at 512.
  - **S1:** at every shape where the rule binds DPAS, the rule's list is not slower than v2's beyond the verdict rule's
    noise line; any miss is a table row set back to v2 (Task 5's table edited, the record says why).
  - **S2:** verify(8) at 32768: (v2 - rule) >= 0.75 x Task 0's attention excess at M = 8 (ms on the unprofiled step); at
    4096 not slower beyond noise.
- [ ] **Step 2: end to end.** `VATTN=1 tools/probe/mtp_d1.sh greedy 4096` and `VATTN=1 MAX_LEN=65536 tools/probe/mtp_d1.sh
  greedy 32768`: the tool-call geomean at `--mtp auto`, rule against v2 (recorded; the bar is N3's "unchanged or better").
  If the rule needed two lists per M, the engine-start capture time and list memory with and without them.
- [ ] **Step 3:** record §4 of docs/probe-verify-attn-<date>.md. Commit
  `git commit -S -m "probe: the verify step with attn_verify_dpas by the rule against v2 - S1, S2, D1 (spec 28c)"`.

### Task 8: the record

- [ ] **Step 1:** spec 28c "28c as built" (the kept lever settings, the table's rows, depth handled at capture or by two
  lists, deviations each with its cost if wrong, the gates' lines, S1 / S2); BENCHMARKS rows (the verify-attention
  table's summary and verify(8) at 4096 / 32768 with and without it, conditions as spec 28 §5); spec 10 §8 a line (the
  verify rows at the switching shapes are no longer bitwise their M = 1 step; property R holds instead);
  `docs/README.md` the probe record's line. Commit `git commit -S -m "docs: spec 28c as built, the verify-attention rows"`.

**Gate for the plan:** Task 0 GO; `attn_verify_ref_test`, `verify_attn_test` and `variant_names_test` green and
`tools/mac_check.sh --base main --kernels` PASS on the Mac (cmdlines additions only, `attn_verify_run` exact / within
N1); on the box N1, N2a, N2b, N3 pass with the rule and with `dpas` forced, G0 holds, S1 holds at every switching shape
and S2 is met (a miss of S2 with S1 held is recorded and goes to the operator; it does not revert the rule). Hand back
with the P0 line, the table, the gate lines and the S1 / S2 numbers.

## The box stages to add later (prose only - no row now)

When the operator allocates a row, its block mirrors 28b's: a **k0** stage on the CPU (G0's sha, bitwise and suite:
`kernel_cmdlines` +N / -0 / ~0, every existing binary identical); a **host** stage (`attn_verify_ref_test`,
`verify_attn_test`, `variant_names_test`); a **capture** stage on one card with the checkpoint (Task 4 Step 1, its dump
reused by the next two); a **table** stage on the idle GPU (Task 4 Steps 2-3, opt-in, long: the levers and the 7 x 4 x 2
table at median of 5); a **kernels** stage on one card (`attn_verify_dpas_test` parts S and D); a **gates** stage on one
card with the checkpoint (Task 6 Step 2, both settings); a **speed** stage, opt-in, idle GPU (Task 7: the in-list pairs
at three depths, the two D1 runs); and a manual **record** stage depending on all of them. Its row note: timed stages
need the idle protocol; hand back the table, the kept settings, N1 / N2 worst cases per M, the tie census, S1 / S2.

## Operator decisions (2026-10-31)

1. **The noise line:** "within noise" is paired v2 / dpas >= 1.03 with the range's low end > 1.00 (else v2).
   **Decided: 3 %** - the attention kernel is timed alone, in pairs.
2. **28 parked:** **decided: 28c still runs** if 28a records NO-GO - Task 0 computes the share on today's step; the
   attention is one of the levers a NO-GO names.
3. **Two lists per M:** **decided: acceptable** when the table makes an M switch at a depth; capture time and list
   memory recorded.
4. **int8 KV at M = 5..8:** **decided: recorded from probe controls only;** DPAS is bound there when
   DFlash-with-int8-KV is planned, not here.
5. **The GO line and S2 on the attention's excess over M = 1** (one row's KV walk is a floor no kernel removes), not
   the raw share: **decided: yes.**
