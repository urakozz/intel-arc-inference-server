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

