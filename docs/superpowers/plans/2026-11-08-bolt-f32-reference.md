# Bolt's f32 reference and judge (operator 2026-11-08, option B)

**Goal:** judge `r60.golden` (Nemotron 3.5 Bolt, our g64 export, two cards) against an f32 CPU reference, the way
Ornith and K2 are judged (memory rulings 2026-10-23 / 10-28 / 10-31), instead of the bf16-computed golden. Until it
exists the real checkpoint runs under option A (the synthetics' first-flip cut-off, tokens gated on every row;
`tests/golden/bolt_golden_test.cc`, 2026-11-08).

**Why:** the 2026-11-07 run gave 32 / 32 exact tokens on every prompt and logits cosine >= 0.99983, yet FAILed on
expert sets beyond the 1e-3 near-tie (gaps 1.0e-3 - 7.2e-3, after a true near-tie flip at row 0 L1, gap 4e-5) and on
the SSM state (0.9987-0.9990 vs the PROPOSED 0.999). The reference (`bolt_ref.py run`, `build_streamed` in bf16) rounds
in bf16 like the engine, so it is not the truth either: on K2 the bf16 golden differed from f32 in ~20 % of expert sets.

## The rules (existing rulings, applied to Bolt)

- **Tokens:** the f32 judge's tie-aware gate (golden_common.h `judge_row`): a row whose f32 top-2 gap is within one
  bf16 ulp of the top logit is a tie; the engine's token must be in the set.
- **Expert sets (2026-10-28):** count the (row, block) pairs whose engine set differs from the f32 set, and the same
  for the bf16 golden; PASS when engine <= golden. Print both counts and the near-ties (f32 sel gap within one bf16 ulp
  of the 9th sel).
- **SSM state, conv columns, h taps (2026-10-31 Q3):** per block, the engine's cosine to f32 >= the bf16 golden's
  cosine to f32 (print both; the PROPOSED absolute bars become printed diagnostics on the real checkpoint).
- **No f32 reference present:** the real-checkpoint tests SKIP (77) "awaiting f32 reference"; the synthetics keep the
  cut-off rule.
- Rows align: the f32 run is teacher-forced on the bf16 golden's 32 tokens (`bolt_ref.py run --force`), as Ornith's.

## Tasks

1. **`bolt_ref.py run --dtype fp32`** (`tools/oracle/bolt_ref.py`): `build_streamed(..., dtype=torch.float32)`, the
   int4 export dequantised to f32 (the router bias stays fp32 as now, the SSM / conv cache fp32 as now), the same golden
   layout (route.ids / sel / gap / w, ssm, conv, h taps, logits). The box CPU has no AVX-512 bf16, so f32 is the fast
   path (memory: cpu-bf16-slow). Mac test: `test_bolt_ref.py` on the tiny puzzle - fp32 vs bf16 runs agree to bf16
   rounding, the layout keys identical.
2. **`bolt_oracle.sh golden-f32`** -> `$DATA/oracle-out-bolt-f32/<p>.golden.safetensors` for bolt_short, bolt_4k,
   bolt_agentic (bolt_32k later, if its time allows), each `--force` on `oracle-out-bolt/<p>.cont*.ids`; resumable per
   prompt; under the 64 GB CPU cap (`run_in_container.sh`); a data key `oracle_bolt_f32` in `tools/box_validate/data.sh`.
   First measure bolt_short's time and peak RAM, then queue the rest as a CPU chain line (class A, peak <= 64).
3. **`bolt_golden_test --judge <dir>`**: reads the f32 golden beside the bf16 one and applies the rules above;
   `r60.golden`'s real-checkpoint tests pass `--judge $DATA/oracle-out-bolt-f32` (CMake, like the Ornith gates). Unit
   test for the counting / near-tie helper on the Mac (`bolt_cutoff_test`'s pattern).
4. **Box:** `r57.host` (the Python tests), then `r60.golden` against the f32 reference; record both counts in
   open-work and close the Waiting row.

## Cost

One agent, after 3a (one agent slot). CPU: hours per prompt for the 74B-A9B model at f32 (ESTIMATED; Task 2 measures
bolt_short first). No engine change.
