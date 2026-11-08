# 3d: the short-prefill leftovers (gate||up above 192 rows, two-card prefill timing)

**Goal:** finish what 3a (`pf_gemm_small`, merged 212ae42f) and 3b (`pf_gemm_q4`, merged 58d687a1) left: (a) the
gate||up linear (SiLU form) of a 193-384-row chunk still takes the slab walk (dequant + `pf_gemm_small`), so such a
turn gains only ~10 % (237 ids 287 -> 259 ms); (b) two-card (`--parallel pp`) prefill runs the same `gemm_l0` but was
never timed. Bitwise identical to today's path, as 3a and 3b - no f32 judge.

## (a) gate||up at 193-384 rows

**Why it is left:** 3b's SiLU tiles cover <= 192 rows (A1_B2_S4, A4_B2_S1); the 64-row x 32-column tile that would
carry 193-384 (NA 8 x NB 2) spills registers (`docs/probes/probe-fused-small-chunk-2026-11-08.md`).

**Break-even, fixed before measuring** (3b's rule): a candidate takes a row count only where it is >= 5 % below the
slab walk for gate||up in the same interleaved run, and only if every output word is bitwise the slab walk's.

**Candidates, in order (probe `tools/probe/probe_fused_small.cc`, sweep build `-DB70_PF_Q4_SWEEP=ON`):**
1. **The same NA 8 x NB 2 tile in the large register file** (256 GRF a thread, half the threads: IGC's
   `-ze-opt-large-register-file` / `intel_reqd_sub_group_size` with the large-GRF module flag) - no spill, fewer
   threads; one build flag on one kernel variant.
2. **NA 4 x NB 2 with more sub-groups a work-group (S2 / S4)** - smaller per-thread state, more threads.
3. **Split gate and up into two plain fused linears and fold the SiLU into the second** only if it stays bitwise
   (the SiLU must see the same fp32 accumulators as today's fused epilogue - check before building).

**Build** the winner into `pf_q4_tile` (automatic, no flags); `pf_gemm_test` bitwise cases at 193-384 rows; the
Qwen3.8 prefill tests 3b ran and the l0 golden gate printout byte-identical to main; before / after at 237 / 300 / 384
ids (`b70-decode --prefill-length`) and a ~250-id turn through `b70-serve`. If no candidate clears the break-even,
record the numbers in the probe doc and close (a) as not worth it.

## (b) two-card prefill timing

`b70-decode --prefill-length` with `--parallel pp` (both cards, split auto) at 1 / 54 / 237 / 512 / 2048 / 2102 ids,
and `b70-serve --parallel pp --kv-cache int8` with a cold ~2k prompt and a ~200-id follow-up, interleaved against
one card. Recorded in `docs/probes/probe-fused-small-chunk-2026-11-08.md` and the benchmark tables. Runs in the
quiet-box session (open-work item 5) - a whole-box, timed job.

## Cost

(a): one agent, a probe of a few hours (one card), a build only if it wins. (b): ~30 min of the quiet-box session.
