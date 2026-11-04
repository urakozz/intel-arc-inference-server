# Open work

One page for everything not yet closed, grouped by spec (the design docs in `docs/superpowers/specs/`). Updated at
every merge. Labels: **spec N** = a design doc; **spec N + letter** (5d, 17a, 28c) = one plan inside it; **rN** = a
row of the box test list (tests of built specs, never new work). A new finding becomes a line under its spec, not a
new label. Rule until the table is empty: no new specs or side plans.

Last updated: 2026-11-04 (main bd90836; box: kernel 7.4.0-rc1, compute-runtime 26.40, IGC 2.42.6).

## Open

| spec | what it is | open item | state | closes when |
|---|---|---|---|---|
| **21** Flash-Next | the Qwen3.8-Flash-Next model | prefill correctness: chunked prefill vs decode differ; MTP prefill threw "384 launches, not 385" | agent working (branch `fix-qwen4exp-prefill`): the launch count was off by one; the chunk difference is stale unread cache slots in the test | fix + r32 / r33 / r34 stages + merge |
| **16** pipeline | one model split across both cards (`--pp 2`) | prefix cache + MTP give wrong results in two-card mode | agent working (branch `fix-pp-prefix-mtp`): cause found (a mid-prompt block's last cache row never rewritten), fix being finished | two-card test + merge |
| **28** faster draft checking | MTP / DFlash verify speed | 28c's DPAS attention kernel is switched off: a real error at 8 rows (t6_usage-loader, position 103: 4.0 below the f32 top) | queued; plan agreed: re-test on IGC 2.42.6, compiler A/B on that kernel, custom IGC only with evidence, our fixed-index rewrite | kernel fixed and back on |
| **5** int8 prefill | faster prompt processing | 5d: prefill in chunks of 4096 | code done (branch `spec5d-chunk-4096`, Nov 1) | box speed run + merge |
| **25** Bolt | the Nemotron Bolt model | speculative decoding back on (branch `bolt-spec-restore`); the Bolt reference set (CPU, building) | code done; references building tonight | box runs + merge |
| *(tools)* | CPU reference tool | the CPU bf16 shim + the Bolt BF16-scale fix (branch `cpu-lane-1101`) | done | one CPU test row (r25.a4_ref) + merge |
| **17** tensor parallel | every layer split across both cards | 17a: the first measurement (branch `spec17a-tp-probe`) | Mac side done | box measurement on a quiet box + merge |

## After the table is empty

1. Quiet-box re-measure: every speed number on the new driver and kernel, the vLLM baseline on the new image.
2. Spec 13 batching (13b), spec 28m MoE draft-checking probe, spec 27 Flint, spec 29 Nanbeige.
3. Box layout: everything under `~/b70` (branches `box-layout`, `plan-box-move`).

## Closed recently

| date | spec | what |
|---|---|---|
| 2026-11-04 | 28 | 28d: DFlash drafter on the small-batch GEMM (draft step 16.3 -> 8.9 ms), Flash-Next verify on it |
| 2026-11-04 | build | kernels recompile when the GPU compiler changes (`cmake/ocloc_stamp.cmake`) |
| 2026-11-04 | 21 | validation fixes: Flash-Next start-up crash (host memory visibility), disk-full guard, the PLE file's location |
| 2026-11-04 | 28 | 28c merged, then its kernel switched off (above) |
| 2026-11-03 | 28 | 28b: small-batch verify GEMM |
| 2026-11-03 | server | chat requests keep the client's JSON key order (prompt ids = the reference, 48 / 48) |
| 2026-11-03 | tools | `tools/box_check.sh` replaces the Mac check |
