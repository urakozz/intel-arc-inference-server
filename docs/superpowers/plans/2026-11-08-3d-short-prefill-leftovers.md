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

## Execution authorized 2026-11-09: finish 3d and 3f

The operator requested both fixes and no additional work-list items. Stay in
the existing `kv8-prefill-fast` worktree, preserve the uncommitted report,
use signed `3d:` commits without trailers, and do not push or merge.
Implementation of the two independent kernel families may proceed in parallel;
the coordinator owns commits, shared documentation, builds and GPU scheduling.
All builds use JOBS=16 on the box. Timed jobs hold the whole-box lock, run
detached under `~/b70`, and retain the already authorized 1–2 minute checks.

**Fixed before measuring:** 3d's rule above remains unchanged: at least 5%
below the slab walk, in the same interleaved run, with every output word
bitwise identical. No candidate is selected from the old busy-box numbers.

For existing item 3f, first probe exact native finite-fp16 scale conversion
and packed V reads in the int8 plain-decode reader. Preserve scale multiply
locations, all fp32 arithmetic and reduction order, the 32-way partition,
cache representation, bf16 kernels and automatic attention selection.
Keep the original int8 reader as a frozen test/probe reference. No new engine
flags or environment variables. Promote only a bitwise-identical candidate
that reduces uninstrumented whole-step wall time by at least 5% at both
81920 and 150000 positions in interleaved before/after pairs after warm-up;
4k/32k and M = 4 must not regress by more than 2%. If a candidate misses,
continue within 3f with the next reader candidate; do not create a new item.

- [x] Freeze a baseline build from the current engine before kernel edits.
- [x] Probe 3d candidates in the specified order, with the committed 5% rule.
- [ ] Integrate the winning gate‖up tile automatically, or record a measured no-go.
- [ ] Verify the named prefill gates and byte-identical golden printouts; measure
  237/300/384-id prefill and the server turn against the baseline.
- [ ] Measure one-card versus pp at all six lengths and the two server requests.
- [x] Freeze the original int8 decode reader; verify candidate/reference bitwise
  partials and outputs across depths, tail waves, M values and finite scales.
- [ ] Integrate the winning 3f reader and run int8 attention, batching, MTP,
  snapshot and golden regressions; measure interleaved whole-step wall gains.
- [ ] Review the combined changes, update the existing probe records, close
  3d/3f with their measured results, and append the uncommitted report.

### 3d full-engine check, 2026-11-09

The two-launch S4 split passed the fixed primitive criterion, but its actual
237-id CLI prefill and warm 200/250-id server requests were 1–1.6% slower.
That candidate is not accepted as an end-to-end fix. Keep the original 5%
gate/slab rule and require the engine confirmation to support the selection.
Before concluding no-go, test the alternative already identified in 3b's
completion notes: one launch with gate/up on paired NB1 subgroups, sharing
the rounded gate through SLM before the unchanged SiLU epilogue. This keeps
one accumulator atom per subgroup and avoids the second launch/output round
trip. Prove indexing and bitwise arithmetic before building it, then apply
the same interleaved controls and full-engine checks. It remains item 3d;
do not add an item for this candidate or the non-winning row ranges.
