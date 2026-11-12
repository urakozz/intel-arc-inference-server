# MoE-model prefill phase 3, Task 1 - the probes and the gate-span check (2026-11-13)

Plan: `docs/superpowers/plans/2026-11-12-moe-model-prefill-phase3.md` Task 1. Branch `moe-pf-phase3-probes` (over the
plan's commit a7a75e61). Ornith 1.5 35B-A3B int4, one B70. Box row 85 (`r85.probe`, `r85.span`, opt-in). Logs on the box:
`~/b70/logs/moe-pf-phase3/`.

## Gates (copied from the plan, written here before any run)

- **Task 2 gate:** router + a||b on `pf_gemm` <= 0.35x the control per layer-chunk at C 4096 (control ~0.5 + 0.2 ms).
  Control = `pf_gemv_rows` MT16 TC2 (what is bound). Printed beside the time: max |logit - control| and, for the router,
  the rows whose top-8 expert set differs from the control's.
- **Task 3 gate:** `shd` saves >= 0.15 ms a layer-chunk at C 4096 against the chosen arm (4.35 ms).
- **Task 4 gate:** `c4` <= 0.88x the chosen arm at C 4096 and at C 2048, oracle-exact.
- **Task 6 gate:** `o8` minus the gather saves >= 0.1 ms a layer-chunk at C 4096, bitwise.
- **Task 7 gate:** the best arm <= 0.7x today's `pf_flash_attn` at both positions (C 4096 at pos 0 and 12288).
- **Span:** a sub-chunk span above 80 nats on any layer is reported to the operator with the layer and the value
  (sly-radiance and saber measured their factorised scan wrong above ~160); it stops nothing here.

Method: interleaved pairs after a warm-up, median of rounds, L0 global timestamps, the same card for every arm of a
comparison. Every arm's correctness check runs before any time is reported (an arm that fails is not timed).
Each table says "quiet box" or "busy box".

## What the arms are (and where the plan's wording did not fit the harness)

- `gemm` (probe_route): the router's bf16 weight [272][2048] re-laid to `pf_gemm`'s [K][N] B and zero-padded to N 512; a||b
  [128][2048] padded to N 256; `pf_gemm` T0 at M 2048 / 4096, K 2048. Not bitwise the GEMVs (a DPAS reduction order).
- `c4` (probe_moe_prefill, `CVT 4`): B as the exact integer (q - 8) in bf16 (magic exponent, 16 + q - 24), a zeroed fp32
  tile per 64-k group, `acc = fma(tile, scale, acc)`. The tile and the group's A are live together, so it is built at
  NA 4 only (`F1.a4b2m1n8` and `F1.a4b4m1n4`), not at the chosen arm's NA 8. Checked on integer-valued A rows (small
  integers in bf16): then every DPAS partial sum is an exact integer in any summation order and a host oracle that sums
  each group in the kernel's order and applies `fmaf(sum, scale, acc)` per group is bitwise-checkable on the fp32 result
  (the EPI 2 build) for two tiles' rows (the first routed tile and the shared tile) x every column of gate||up and down.
- `shd` (Ornith only): the shared expert is block E of the same layout-1 weights, so the production `pf_dequant_slab`
  takes it as it is (gate||up K 2048 N 1024 = one slab; down K 512 N 2048 = two slabs); `pf_gemm` SiLU epilogue for gate||up,
  `pf_gemm` for down (fp32 y [C][2048], what the dense path writes). A is the chunk's x in token order. The routed items
  alone run the chosen arm (items table without the shared tiles, its bound without the shared term). Timed as the sum
  against the chosen arm's full gate||up + down walk. Check: h words and y (bf16-rounded) against the control's shared
  rows, counted. Not run for Flash-Next (its N 1280 is not a whole 1024-column slab).
- `o9` (the plan's `o8`; the tag is its OPT value 9 = bit 1 + bit 8): the chosen arm with A read through `row_tok` from the
  chunk's x. A DPAS A operand wants lane = k and one row per register, which a 2D block read gives only for consecutive
  rows, so each row is one sub-group block read (`block_read_us2`, 32 k) - 64 messages per k32 per sub-group against 2.
  Bitwise the chosen arm is checked on the real routes. Down is unchanged (its A is already in sorted order).
  The gather's own ms is timed beside it (`pf_moe_gather` over the bounded grid).
- attention (probe_flash_attn `--heads 16x2`, Ornith's GQA group of 8, KT 64, EXP2 1): the control is today's
  `pf_flash_attn_Q16KV2` binary. **RPW is the query rows a WORK-GROUP covers a head, not a sub-group**: a sub-group still
  owns 8 rows because 16 rows x 256 dims of fp32 O is 256 GRF by itself. So "RPW 16 / 32" changes the work-group shape
  (more sub-groups sharing a K / V tile through the caches); `QREG 1` holds Q in registers across key tiles. The R8 Q0
  arm is today's source text and must be bitwise the control (a sanity line). The plan's "more query rows a sub-group"
  needs a different kernel (the O dims split across sub-groups with P through SLM); not built here (outside the probe's
  scope: "do not invent a new kernel").
- gate span: `tools/probe/gdn_gate_span.py` over `B70_PF_GDN_DUMP` (Ornith layers 0 / 12 / 28). Flash-Next (layers 0 / 8 / 16
  at `--layers 18`): the dump hook lives in `runtime/prefill/gdn.cc`, which Flash-Next's own walk (`gdn_chunk_q4`) does not
  run; no hook outside `src/` and `src/` is not changed here - not run.

## Results (card 1, busy box: Task 5's jobs and the main session's runs share the box; box row 85, logs `~/b70/logs/moe-pf-phase3/`)

Card 1 rather than card 0 (card 0 was held by the validation run); every comparison below is inside one card. Medians of
interleaved rounds, L0 timestamps. Checks: all printed before any time; every arm that has a check passed it except the
router `gemm` arm, whose numerics are not shown (below).

### Task 2 - router and a||b on `pf_gemm`: speed PASS, numerics not shown

| M | linear | bound GEMV (MT16 TC2) ms | pf_gemm ms | ratio |
|---|---|---:|---:|---:|
| 2048 | router N 272 (pad 512) | 0.2529 | 0.0628 | 0.25 |
| 2048 | a\|\|b N 128 (pad 256) | 0.1462 | 0.0734 | 0.50 |
| 4096 | router N 272 | 0.4970 | 0.0709 | 0.14 |
| 4096 | a\|\|b N 128 | 0.2229 | 0.0584 | 0.26 |

At C 4096 router + a||b: 0.1293 ms against 0.7199 ms = **0.18x (gate <= 0.35x): speed PASS**. The pre-existing GEMV arms stay
bitwise the control (BITWISE PASS). **But the `GEMMCHECK` is wrong**: max |logit - control| 5.96 / 5.85 / 6.14 / 6.14 on logits
of max ~4.3, and the top-8 set differs on every row (2048 of 2048, 4096 of 4096). That is not a reduction-order difference:
the probe fed `pf_gemm` a wrong B layout (or wrong ldb/pad handling), so the values are garbage; the time stands for the
shape, the numerics are NOT shown. Verdict: **speed PASS, numerics not shown (probe B layout wrong)**. Not fixed here; the
implementation task re-lays the weight and checks it.

### Task 4 - `c4` exact-integer weights: FAIL (oracle-exact, too slow)

Oracle check: 0 of 196608 fp32 results differ from the integer-order oracle, both arms, lc 1 and 3, C 2048 and 4096
(two tiles x every column of gate||up and down). Times (ms a layer-chunk, gate||up + down):

| C | chosen `F1.a8b2m1n8.c2x0o1` | `F1.a4b2m1n8.c4x0o1` | ratio | `F1.a4b4m1n4.c4x0o1` | ratio |
|---|---:|---:|---:|---:|---:|
| 2048 | 2.609 | 3.219 | 1.234 | 2.909 | 1.115 |
| 4096 | 4.425 | 5.676 | 1.283 | 4.982 | 1.126 |

Gate <= 0.88x: **FAIL** (1.12-1.28x). Only NA 4 fits (tile + a whole group of A live together); the cheaper conversion does
not pay for the half-size row tile and the extra fma per group.

### Task 3 - shared expert as dense (`shd`): FAIL

Check: h 0 of 1048576 words differ from the control's shared rows (all lc, both C); y bf16 0 of 8388608 differ (dense y is
fp32; max |control y - dense y| 0.125 = the bf16 rounding of the control). Bitwise.

| C | chosen walk (gu+dn, all rows) ms | routed items alone ms | dense half ms | shd total ms | saves ms |
|---|---:|---:|---:|---:|---:|
| 2048 | 2.7317 | 2.6814 | 0.1352 | 2.8413 | -0.110 |
| 4096 | 4.2871 | 4.2081 | 0.2650 | 4.5261 | -0.239 |

Gate saves >= 0.15 ms at C 4096: **FAIL** (-0.24). The shared expert costs only ~0.05-0.08 ms inside the item walk; the dense
path costs 0.14 / 0.27 ms.

### Task 6 - indexed A (`o9` = the plan's `o8`): FAIL

Check: h 0 of 10.4 M words differ from the control (bitwise the chosen arm, all lc). `gu`: 20.80 ms (C 2048) and 34.78 ms (C 4096)
a layer-chunk against 1.68 / 2.88 for the gathered form (12x slower); the gather it would remove is 0.170 / 0.446 ms.
Saves -18.96 / -31.46 ms. Gate >= 0.1 ms at C 4096: **FAIL**. One block read per row (64 messages per k32 per sub-group)
is the cost; a 2D block read cannot gather.

### Task 7 - flash attention work-group shape (Ornith heads, KT 64, EXP2 1, C 4096), vs today's `pf_flash_attn`

Every arm: 0 of 16 777 216 o words differ from today's (bitwise), cosine vs fp64 0.99999827 / 0.99999809, PASS.

| arm | pos 0 ms | ratio | pos 12288 ms | ratio |
|---|---:|---:|---:|---:|
| today's | 3.463 | 1 | 23.827 | 1 |
| R8 Q0 (same text) | 3.479 | 1.005 | 23.879 | 1.002 |
| R8 Q1 | 3.257 | 0.940 | 21.953 | 0.921 |
| R16 Q0 | 2.987 | 0.863 | 20.818 | 0.874 |
| R16 Q1 | 2.887 | 0.834 | 20.633 | 0.866 |
| R32 Q0 | 2.725 | 0.787 | 20.224 | 0.849 |
| **R32 Q1** | **2.703** | **0.781** | 20.330 | 0.853 |

Best arm: 0.78x at pos 0 (R32 Q1) and 0.85x at pos 12288 (R32 Q0); gate <= 0.7x at both: **FAIL**, but a real, bitwise
gain (-22 % / -15 % of the kernel) that needs no judge. All arms stay at 22-28 % of peak: the sub-group still owns 8 rows;
the true "more query rows a sub-group" kernel is a later task.

### Gate span (Ornith, `r84.dump`, first chunk, C 2048)

| layer | max span (nats) | sub-chunk, head | pairs over 80 / over 160 (of 1024) |
|---|---:|---|---|
| 0 | **5591.9** | 0, 22 | 192 / 192 |
| 12 | 420.7 | 0, 2 | 64 / 36 |
| 28 | 98.2 | 0, 8 | 1 / 0 |

All three exceed 80 nats and are reported to the operator. Note: phase 2's `probe_gdn_intra` compared our kernels against an
fp64 walk on these same three dumps and matched (`probe-prefill-phase2-2026-11-12.md` section 2), so the large span does
not break our scan on Ornith. Flash-Next: not checked (no dump hook outside `src/`).

## Summary of the gates

| gate | verdict |
|---|---|
| Task 2 router + a\|\|b on pf_gemm | speed PASS (0.18x); numerics not shown (probe B layout wrong) |
| Task 4 c4 | FAIL (1.12-1.28x; oracle-exact) |
| Task 3 shd | FAIL (-0.24 ms; bitwise) |
| Task 6 o8 | FAIL (12x slower; bitwise) |
| Task 7 attention | FAIL at 0.7x (best 0.78x / 0.85x, bitwise) |
| span | 5592 / 421 / 98 nats, reported |
