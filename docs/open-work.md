# Open work

One page for everything not yet closed, grouped by spec (the design docs in `docs/superpowers/specs/`). Updated at
every merge. Labels: **spec N** = a design doc; **spec N + letter** (5d, 17a, 28c) = one plan inside it; **rN** = a
row of the box test list (tests of built specs, never new work). A new finding becomes a line under its spec, not a
new label. Rule until the table is empty: no new specs or side plans. Documents are deleted when obsolete: closed rows
leave `docs/superpowers/plans/box-validation-queue.md`, finished sessions leave `docs/box-day-plan.md`, and both files
go when empty; this page stays while fix rounds still add items.

Last updated: 2026-11-04 (main 03e7858; box: kernel 7.4.0-rc1, compute-runtime 26.40, IGC 2.42.6).

## Open

| spec | what it is | open item | state | closes when |
|---|---|---|---|---|
| **21** Flash-Next | the Qwen3.8-Flash-Next model | the golden gates judge by scores (branch `fix-qwen4exp-gates-serve`): the CPU reference now also writes every QSA row's block scores and the router's top-16 probabilities; a differing selection passes when the row's scores agree (cosine >= 0.999) and every swapped block is within 2x the row's score difference of the cut. The synthetic sets were remade; r32.golden + r33.golden 17 / 18 (selections 0 differing, was 7,717), the 18th (Intel form, injected) fixed after (it failed the same way before) | done on the branch; r33.pp (two cards) PASS with the judge | merge; the real-checkpoint sets (hours) keep the old gap rule until remade |
| **21** Flash-Next | (stage run 2026-11-04 on the fixed prefill) | r34.serve's cause found and fixed: the prefix cache sized each stored block as kv_bytes(1) x 2048, but Flash-Next's blocks also carry one compressed indexer key per 4 positions, so every block was 128 KiB short and the copy ran into the next stored entry (a warmed request restored damaged state) | fixed on the branch: r34.serve PASS (warmed = cache-off, plain and --mtp 2; the old binary reproduces the FAIL), r34.pp PASS | merge |
| **16** pipeline | one model split across both cards (`--pp 2`) | (optional) a mid-prompt block snapshot stores the head's last KV row before it is written: after restoring a deeper snapshot the MTP head sees a stale key at block boundaries - drafts only, never the output, one card and two cards alike | not started; the proper fix stores that row with the next block (a snapshot-store format version bump) | operator decides whether to build it |
| **5** int8 prefill | faster prompt processing | 5d: prefill in chunks of 4096 | code done (branch `spec5d-chunk-4096`, Nov 1) | box speed run + merge |
| **25** Bolt | the Nemotron Bolt model | speculative decoding back on (branch `bolt-spec-restore`); the Bolt reference set (CPU, building) | code done; references building tonight | box runs + merge |
| **20** Kolibri | the Aleph Alpha German / English MoE (kept) | (optional) a real session from a client that drops `reasoning_content` (many OpenAI clients do): earlier tool turns then render as empty think blocks, per the template - the community stalls may come from such clients | not started | one real run without echoing reasoning |
| *(tools)* | CPU reference tool | the CPU bf16 shim + the Bolt BF16-scale fix (branch `cpu-lane-1101`) | done | one CPU test row (r25.a4_ref) + merge |
| **17** tensor parallel | every layer split across both cards | 17a: the first measurement (branch `spec17a-tp-probe`) | Mac side done | box measurement on a quiet box + merge |

## Box test rows still open

The rows are defined in `docs/superpowers/plans/box-validation-queue.md` (one row = the box tests of one spec's
plan) and run by `tools/box_validate/stages.sh` (stage `rN.<name>`); `docs/box-day-plan.md` groups them into
sessions. Results live on the box under `~/b70-validate/<tree>/<state>/<stage>.status`; the latest result of every
stage across all runs: `~/b70/logs/stage-latest.txt` (2026-11-04: 272 PASS, 14 FAIL, 86 SKIP - the SKIPs mostly
wait on a FAIL upstream or on missing reference data). Rows 44-52 / 64 were dropped models (pivot 2026-10-27).

| spec | what | rows not yet PASS | why | closes with |
|---|---|---|---|---|
| G0 | bit-for-bit comparison against a baseline build | g0.suite (on main ad711de, before the spec 21 merge) | bitwise PASS against the new baseline; the suite FAILs only on known open items - the Qwen3.8 MTP tests waiting for f32 references (spec 28) and the pre-fix Flash-Next tests (fixed since) - but a suite FAIL stops every later row | the blocked rows relaunched past G0 on main fb5535b (2026-11-04 07:40) |
| **21** Flash-Next | | 32 golden / pp / partial / cli, 33 *, 34 * | the prefill fix (open, above); the real-checkpoint refs now exist | the spec 21 fix, then one run of r32-r34 |
| **25** Bolt | | 60, 61, 62 (synthetic and golden chains) | waiting on G0 and the Bolt reference set (building) | after G0 re-baseline + the references |
| **16** pipeline | | 23 (two-card prefill speed), 27 (two-card serving: MTP, lookup, prefix cache) | 27 waits on the spec 16 fix (open, above); 23 is speed, quiet box | the spec 16 fix, then one run; 23 in the quiet-box session |
| **18** K2-Horizon | | 17.k0, 21.passkey, 25.a4_ref, 25.passkey | the long-context passkey and the tool-call reference (CPU, days) | one run each; 25.a4_ref on the CPU |
| **20** Kolibri | the Aleph Alpha German / English MoE (kept) | (optional) a real session from a client that drops `reasoning_content` (many OpenAI clients do): earlier tool turns then render as empty think blocks, per the template - the community stalls may come from such clients | not started | one real run without echoing reasoning |
| **19** DFlash | | 54.d1_ref (the CPU reference half) | CPU job | one CPU run |
| **22** expert offload probe | | 36 curve / project / reap | not run | one run (CPU + one card) |
| **13** batching | | 66 check / chunk / cost | speed, idle box | quiet-box session |
| **12** int8 KV | | 11 a4 / bf16 | reference data | one run |

## After the table is empty

1. Quiet-box re-measure: every speed number on the new driver and kernel, the vLLM baseline on the new image; then
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
