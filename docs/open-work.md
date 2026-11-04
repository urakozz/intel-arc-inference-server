# Open work

One page for everything not yet closed, grouped by spec (the design docs in `docs/superpowers/specs/`). Updated at
every merge. Labels: **spec N** = a design doc; **spec N + letter** (5d, 17a, 28c) = one plan inside it; **rN** = a
row of the box test list (tests of built specs, never new work). A new finding becomes a line under its spec, not a
new label. Rule until the table is empty: no new specs or side plans. Documents are deleted when obsolete: closed rows
leave `docs/superpowers/plans/box-validation-queue.md`, finished sessions leave `docs/box-day-plan.md`, and both files
go when empty; this page stays while fix rounds still add items.

Last updated: 2026-11-04 (main 4ffc3a8; box: kernel 7.4.0-rc1, compute-runtime 26.40, IGC 2.42.6).

## Open

| spec | what it is | open item | state | closes when |
|---|---|---|---|---|
| **16** pipeline | one model split across both cards (`--pp 2`) | a mid-prompt block snapshot stores the MTP head's last KV row before it is written: after restoring a deeper snapshot the head sees one stale key per block boundary - drafts only, never the output, one card and two cards alike | operator 2026-11-04: skip the fix, verify first - in the quiet-box session, draft acceptance of a long agent session served cold vs from the prefix cache (one card and `--pp 2`) | closed if acceptance differs by < 1%; otherwise build the fix (the row stored with the next block, a snapshot-store format version bump) |
| **5** int8 prefill | faster prompt processing | 5d: prefill in chunks of 4096 | code done (branch `spec5d-chunk-4096`, Nov 1) | box speed run + merge |
| **25** Bolt | the Nemotron Bolt model | speculative decoding back on (branch `bolt-spec-restore`); the Bolt reference set (CPU, building) | code done; references building tonight | box runs + merge |
| **20** Kolibri | the Aleph Alpha German / English MoE (kept) | a real tool session from a client that drops `reasoning_content` (many OpenAI clients do): earlier tool turns then render as empty think blocks, per the template - the community stalls may come from such clients (operator 2026-11-04: do it) | queued (box session, one card or two) | one real multi-turn tool session without echoing reasoning: stalls and call validity counted, compared with the 2026-11-04 echoing session (0 stalls, 24 valid calls) |
| *(tools)* | CPU reference tool | the CPU bf16 shim + the Bolt BF16-scale fix (branch `cpu-lane-1101`) | done | one CPU test row (r25.a4_ref) + merge |
| **17** tensor parallel | every layer split across both cards | 17a: the first measurement (branch `spec17a-tp-probe`) | Mac side done | box measurement on a quiet box + merge |
| **21** Flash-Next | the Qwen3.8-Flash-Next model | r32.partial and the real-checkpoint golden sets still use the old 1e-3 gap rule (remaking them with the block scores is hours of CPU); routing on layers other than the last is judged by the gap rule as a fallback (the engine keeps only ranks 9 / 10 of the router probabilities) | not started | the sets remade on the CPU |
| **28** faster draft checking | Qwen3.8 MTP / DFlash graded by the f32 references | 5 tests FAIL on judged verdicts only (mtp_gpu_test + its two 128k twins, dflash_gpu_test, dflash_lookup_gpu_test), always a turn-boundary row after a tool call (special token vs text token). Cause found: no engine defect - these rows are chaotic in bf16 (transformers bf16 on the same weights lands as far from f32 as the engine); the mtp_gpu rows are flipped by the l0-int8 prompt pass. Rulings 2026-11-04: at a row where standard bf16 moves the pair's gap by > 8 ulp, the speculative pick is graded against the plain stream's own error (max(8, \|plain gap - f32 gap\|)), the strict bar everywhere else; the reference is int8-aware for tests that prefill with l0-int8. Also dflash_engine_test's M = k+1 vs M = 8 bitwise check now differs (M >= 6 runs the DPAS attention): judged by closeness instead, bitwise kept where the kernels match | in progress (branch `spec28-special-token-gap`): bf16 references for the 72 affected contexts building on the CPU, the rule written. Next (one agent, after it): the int8-aware prompt pass - `qwen_forced.py` and the bf16 reference through `int8_sim.py` fq_x / fq_w on prompt positions (as Ornith's `--sim-int8-prefill`), int8-aware f32 and bf16 references for the mtp_gpu contexts, those tests pointed at them | the 5 tests PASS or FAIL on the new judges; merge |

## Box test rows still open

The rows are defined in `docs/superpowers/plans/box-validation-queue.md` (one row = the box tests of one spec's
plan) and run by `tools/box_validate/stages.sh` (stage `rN.<name>`); `docs/box-day-plan.md` groups them into
sessions. Results live on the box under `~/b70-validate/<tree>/<state>/<stage>.status`; the latest result of every
stage across all runs: `~/b70/logs/stage-latest.txt` (2026-11-04: 272 PASS, 14 FAIL, 86 SKIP - the SKIPs mostly
wait on a FAIL upstream or on missing reference data). Rows 44-52 / 64 were dropped models (pivot 2026-10-27).

| spec | what | rows not yet PASS | why | closes with |
|---|---|---|---|---|
| G0 | bit-for-bit comparison against a baseline build | g0.suite (on main ad711de, before the spec 21 merge) | bitwise PASS against the new baseline; the suite FAILs only on known open items - the Qwen3.8 MTP / DFlash tests (now graded by f32 references: 5 judged FAILs, spec 28 row above) and the pre-fix Flash-Next tests (fixed since) - but a suite FAIL stops every later row | the blocked rows relaunched past G0 on main fb5535b (2026-11-04 07:40) |
| **21** Flash-Next | | 32 partial / cli, real-checkpoint rows | golden, pp and serve passed on the branch (2026-11-04, run directly); the rest not yet run | one box_validate run of r32-r34 on main |
| **25** Bolt | | 60, 61, 62 (synthetic and golden chains) | waiting on G0 and the Bolt reference set (building) | after G0 re-baseline + the references |
| **16** pipeline | | 23 (two-card prefill speed), 27 (two-card serving: MTP, lookup, prefix cache) | 27 can run (the spec 16 prefix-cache + MTP fix merged 2026-11-04); 23 is speed, quiet box | one run of 27 in the validation rerun; 23 in the quiet-box session |
| **18** K2-Horizon | | 17.k0, 21.passkey, 25.a4_ref, 25.passkey | the long-context passkey and the tool-call reference (CPU, days) | one run each; 25.a4_ref on the CPU |
| **19** DFlash | | 54.d1_ref (the CPU reference half) | CPU job | one CPU run |
| **22** expert offload probe | | 36 curve / project / reap | not run | one run (CPU + one card) |
| **13** batching | | 66 check / chunk / cost | speed, idle box | quiet-box session |
| **12** int8 KV | | 11 a4 / bf16 | reference data | one run |

## After the table is empty

1. Quiet-box re-measure: every speed number on the new driver and kernel, the vLLM baseline on the new image, spec 17a's
   first tensor-parallel measurement, and spec 16's check (MTP draft acceptance cold vs prefix-cache-restored, one card
   and `--pp 2`: < 1% closes the stale-key item); then
   refresh README.md and docs/BENCHMARKS.md (they still say "nothing measured" for Ornith, K2, Flash-Next, Bolt and show
   pre-upgrade headline rows) and the per-model table in docs/18-running-models.md.
2. Spec 22 expert offload (Flash-Next whole on two cards; its P0 rows 36 first; §9 has the Basalt lessons: CPU compute
   for part of the misses, a popularity profile, a low-bit always-resident drafter), spec 13 batching (13b), spec 28m MoE
   draft-checking probe. (Spec 27 Flint and spec 29 Nanbeige PARKED 2026-11-04 on
   the independent evidence - see the notes at the top of their design docs; their plans were deleted.)
3. Box layout: everything under `~/b70` (branches `box-layout`, `plan-box-move`).

## Closed recently

| date | spec | what |
|---|---|---|
| 2026-11-04 | 28 | Qwen3.8 f32 references for all 286 forced contexts (one trie-merged fp32 CPU pass, ~2 h); no reference = SKIP (77), not FAIL; tests wired to the judges; mtp_verify_test and dflash_engine_test PASS judged; box_check 315/0/0 |
| 2026-11-04 | 21 | Flash-Next golden gates judged by scores (the CPU reference writes the block scores and router top-16; selections differing beyond rounding 7,717 -> 0) and r34.serve: the prefix cache sized entries kv_bytes(1) x n, missing Flash-Next's compressed indexer keys, so saves overran the next entry - now sized by the engine's kv_bytes(n); r32/r33 golden, pp, r34.serve PASS; box_check 314/0/0 |
| 2026-11-04 | 28 | 28c's DPAS verify attention back on: the position-103 error was its softmax weights rounded to bf16 (not the compiler); weights now as two bf16 halves, DFlash token-identical to plain on the case, thresholds re-timed unchanged, kernel test bar 0.999997; box_check 314/0/0 |
| 2026-11-04 | 20 | Kolibri tool calls: raw newlines inside JSON string arguments repaired (shared parser), top-level reasoning_effort reaches the template, a 20-turn session test; real 19-turn session on two cards: 0 stalls, 24 valid calls; box_check 314/0/0 |
| 2026-11-04 | 21 | Flash-Next prefill: the MTP prefill launch count (1 + 25, not 26) - every MTP prefill aborted; the rest was the test (dead conv-ring slots, a cosine NaN, tail ring by cosine, selections judged by block scores); box_check 313/0/0 |
| 2026-11-04 | docs | core docs 01-18 refreshed (19-running-models renamed 18) |
| 2026-11-04 | 16 | two-card prefix cache + MTP: the head's KV row end - 1 read from a shadow copy taken before the next chunk can rewrite it; 9/9 pipeline tests, box_check 313/0/0 |
| 2026-11-04 | 28 | 28d: DFlash drafter on the small-batch GEMM (draft step 16.3 -> 8.9 ms), Flash-Next verify on it |
| 2026-11-04 | build | kernels recompile when the GPU compiler changes (`cmake/ocloc_stamp.cmake`) |
| 2026-11-04 | 21 | validation fixes: Flash-Next start-up crash (host memory visibility), disk-full guard, the PLE file's location |
| 2026-11-04 | 28 | 28c merged, then its kernel switched off (above) |
| 2026-11-03 | 28 | 28b: small-batch verify GEMM |
| 2026-11-03 | server | chat requests keep the client's JSON key order (prompt ids = the reference, 48 / 48) |
| 2026-11-03 | tools | `tools/box_check.sh` replaces the Mac check |
