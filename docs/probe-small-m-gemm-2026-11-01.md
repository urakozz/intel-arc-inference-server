# Probe: small-M decode GEMM (spec 28 P0, plan 28a) - 2026-11-01

Spec: `docs/superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md` (§6 the go rule, §9 the verdict).
Plan: `docs/superpowers/plans/2026-10-27-spec28a-small-m-probe.md`. Box queue row 68.

**Grades** (spec 28's legend): measured / derived / estimated / published by others.

**Timing grade of this record: busy-box timing (one card), to be re-checked on a quiet box.** Operator ruling
2026-11-01: the 28a timing ran on one card under `tools/gpu_lock.sh card N` while the other card and the CPU
carried other jobs (the box load is printed beside every table). Every ratio is from interleaved lists after a
warm-up, so a clock or contention shift between arms moves both sides of a pair; the absolute microseconds are
not comparable with an idle-box record. The GO / NO-GO line below is **provisional** until the re-check set (§6)
is re-run on a quiet box.

## 0. What was built

- `tools/probe/probe_small_m.cl` - the stand-in `psm` kernel (one source, compile-time knobs) and `psm_quant`
  (the int8 arm's per-row quantiser, pw4a8_quant's rule). It reads the bytes gemv / gemv_i8w read and writes
  gemv's partials (`out[(s M + m) N + n]`; the head's scaled `out[m N + n]`).
  - TYPE: bf16 DPAS (k16; int4 -> bf16 `q - 8` exact, int8 -> bf16 exact) or int DPAS on int8 rows
    (i8 x i4 k32 for int4 - the GPTQ words xor 0x88888888 are the signed B operand; i8 x i8 for the head).
  - ORIENT: `wB` (weights on B, a sub-group owns 16 columns, rows in blocks of 8 on A; M = 9..16 a second row
    block against the same B registers) or `wA` (weights on A, 8 columns per DPAS row block, two blocks per
    sub-group; tokens on B, one per lane - M <= 16 in one DPAS column block).
  - SPLITK: grid (gemv's: dimension 1 the slice; 4 sub-groups a work-group) or in-work-group (NSGN column
    sub-groups x KSL K-slices, groups interleaved across the slices, SLM reduction in slice order): 2x8 and 8x4.
  - DEQ shift (gemv's xor + arithmetic shift) or magic (the nibble OR'd into bf16 0x4300 = 128 + q, then -136
    in fp32 - exact); PF_G prefetch 1 / 2 / 3 own groups ahead (2D block prefetch for layout 0, prefetch() for the
    tiled forms); 256 GRF by the ocloc option (the probe reads the zebin's `grf_count` before timing an arm).
  - Per 64-k group: the group's DPAS chain from zero (4 x k16 / 2 x k32), then `acc += group x scale` in fp32
    (the checkpoint's g64 semantics); the head chains all K and applies the column scale at the end.
- 347 binaries (`psm_*`, tools/probe/CMakeLists.txt): per shape both types x both orientations at M = 1 / 2 / 4 /
  8 / 16, bf16 also at M = 3 / 5 / 6 / 7 (property R), the levers at M = 8, the 2x8 shape at M = 2 / 4 / 16 and
  8x4 at 16, the S sweep at M = 8; plus probe-only M = 16 controls built from gemv.cl / gemv_i8w.cl /
  gemv_bf16.cl with the production cells' defines. No scratch (spill) in any of the 347 (zeinfo).
- `tests/kernels/gemm_sm_ref.h` + `gemm_sm_ref_test` (host): the fp64 reference, gemv's fp32 twin, the i8
  quantiser, N2's checker.
- `probe_verify_m --verify8 1`: the verify lists at M = 1..8 (M = 1..4 on the engine's 4-slot MtpBuffers, 5..8
  on a probe-owned 8-slot one), unprofiled pairs and the profiled per-family table; `runtime::build_verify`
  gained an optional `ProfileEvents*` (default null, no command line changes: kernel_cmdlines +0 / -0 / ~0).

### Review Focus 2 - dequant once (ISA, measured 2026-11-01)

`IGC_ShaderDumpEnable` dumps of the gate||up shape (5120 x 34816, S 8, layout 0); instruction counts of `psm`:

| variant | M | dpas | shl | asr | shr | mov | mad |
|---|---|---|---|---|---|---|---|
| bf16 wB | 1 | 4 | 55 | 64 | 33 | 123 | 1 |
| bf16 wB | 2 | 4 | 55 | 64 | 33 | 125 | 2 |
| bf16 wB | 8 | 4 | 55 | 64 | 33 | 131 | 8 |
| bf16 wB | 16 | 8 | 55 | 64 | 33 | 171 | 16 |
| bf16 wA | 2 / 8 / 16 | 8 | 73 | 64 | 66 | 205 | 16 |

The dequant (shl / asr / shr) is the same at every M; what grows with M in wB is the activation loads' moves and
the per-row scale `mad`s (M of them a group), and the second DPAS row block at M = 16. wA's code is identical at
every M (the tokens are lanes).

## 1. The shape table (r68.shapes) - busy-box timing

(filled from the run)

## 2. The verify step's attribution (r68.attrib) - busy-box timing

(filled from the run)

## 3. Numerics (r68.check, r68.numerics)

(filled from the run)

## 4. The provisional verdict

(filled from the run)

## 5. Open

## 6. The quiet-box re-check set

(filled from the run)
