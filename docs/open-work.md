# Open work

One page for everything not yet closed, grouped by spec (the design docs in `docs/superpowers/specs/`). Updated at
every merge. Labels: **spec N** = a design doc; **spec N + letter** (5d, 17a, 28c) = one plan inside it; **rN** = a
row of the box test list (tests of built specs, never new work). A new finding becomes a line under its spec, not a
new label. Rule until the table is empty: no new specs or side plans. Documents are deleted when obsolete: closed rows
leave `docs/superpowers/plans/box-validation-queue.md`, finished sessions leave `docs/box-day-plan.md`, and both files
go when empty; this page stays while fix rounds still add items.

Last updated: 2026-11-06 (main bd0e703; box: kernel 7.4.0-rc1, compute-runtime 26.40, IGC 2.42.6).

## Open

| spec | what it is | open item | state | closes when |
|---|---|---|---|---|
| **16** pipeline | one model split across both cards (`--pp 2`) | a mid-prompt block snapshot stores the MTP head's last KV row before it is written: after restoring a deeper snapshot the head sees one stale key per block boundary - drafts only, never the output, one card and two cards alike | operator 2026-11-04: skip the fix, verify first - in the quiet-box session, draft acceptance of a long agent session served cold vs from the prefix cache (one card and `--pp 2`) | closed if acceptance differs by < 1%; otherwise build the fix (the row stored with the next block, a snapshot-store format version bump) |
| **5** int8 prefill | faster prompt processing | 5d: prefill in chunks of 4096 - built, all gates pass at 4096; +4.2 % at pp4096 (+5.3 % 32k, +1.5 % 128k; busy box), under the +5 % bar, so 2048 stays the default and 4096 is opt-in (`--prefill-chunk 4096`) | merged 2026-11-06 (box_check 318/0/0) | operator: context-first rule on or off; `r75.speed` in the quiet-box session |
| **25** Bolt | the Nemotron Bolt model | speculative decoding back on (branch `bolt-spec-restore`); the Bolt reference set (CPU, building) | code done; references building tonight | box runs + merge |
| *(tools)* | CPU reference tool | the CPU bf16 shim, the Bolt BF16-scale fix, K2's 600-id prompt floor - merged 2026-11-06 (box_check 316/0/0); K2's tool-call prompt set rebuilt (36 / 36) | merged | one CPU test row (r25.a4_ref, the paused CPU queue) |
| *(box)* | model storage | hot / cold tiers (operator 2026-11-05): the new 8 TB SATA disk holds every model (cold, fixed mount under `~/b70`), the NVMe only the working set (hot); `tools/hf_tier.sh hot|cold|status <model>` swaps each `~/.cache/huggingface/hub/models--X` between a real NVMe folder and a symlink to the 8 TB copy (copy, verify, then swap; refuses a model a job has open); replaces the USB bind mounts that the reboot dropped. First moves: Bolt bf16 hot (140 GB > RAM: the CPU reference streamed it from the USB disk at ~10 MB/s), parked models and finished reference outputs cold; settle the shared `hub/blobs` folder (599 links) first | queued: after the operator's USB -> 8 TB copy, before the validation rerun | the script + `status` on the box, the bf16 originals reachable again, Bolt bf16 hot |
| *(engine)* | zero-config cleanup | the operator's 2026-11-06 rulings: the user chooses the model, speculative decoding (default: our auto dynamic MTP), the max context length and the parallel mode (`auto` = the faster choice by model fit and measured numbers; one card, pipeline and - once built - tensor parallel stay explicitly selectable, the pipeline split an optional override, for experiments and measurement; operator 2026-11-06) - the engine decides the rest. List A (questions already decided): remove decode attention v1, prefill composed attention, the GDN scan / solve selectors, SILU_FUSED, prefill replay, K2 eager, Flash-Next fresh MTP selection, B70_Q4_PLE / --ple-dir, the EXP2 / AGNES_SWEEP / DECODE_EXTRA_M build options, --lm-head, --draft-format, --mtp-max / --mtp-cost, --spec-cost / --spec-min-match / --spec-history, --served-name, --queue, --mem-reserve-gb, Bolt's --chat-template / --generation-config, Flash-Next's --tokenizer, the renamed-flag refusals, ONEAPI_DEVICE_SELECTOR, B70_TOKENIZER_JSON, the Mac plain-C kernel branches; card memory and Xe-core count read from the device (not kB70Bytes); a startup gate (B70 only until a B60 / B65 is in the box; 1-2 cards, P2P between them); the load-time host-staged weight mover replaced by a direct card-to-card copy; their tests and stages go too | queued after spec 5d merges (both touch prefill) | box_check + one validation run |
| **17** tensor parallel | every layer split across both cards | 17a: the first measurement (branch `spec17a-tp-probe`) | Mac side done | box measurement on a quiet box + merge |
| **21** Flash-Next | the Qwen3.8-Flash-Next model | r32.partial and the real-checkpoint golden sets still use the old 1e-3 gap rule (remaking them with the block scores is hours of CPU); routing on layers other than the last is judged by the gap rule as a fallback (the engine keeps only ranks 9 / 10 of the router probabilities) | not started | the sets remade on the CPU |

## Box test rows still open

The rows are defined in `docs/superpowers/plans/box-validation-queue.md` (one row = the box tests of one spec's
plan) and run by `tools/box_validate/stages.sh` (stage `rN.<name>`); `docs/box-day-plan.md` groups them into
sessions. Results live on the box under `~/b70-validate/<tree>/<state>/<stage>.status`; the latest result of every
stage across all runs: `~/b70/logs/stage-latest.txt` (2026-11-04: 272 PASS, 14 FAIL, 86 SKIP - the SKIPs mostly
wait on a FAIL upstream or on missing reference data). Rows 44-52 / 64 were dropped models (pivot 2026-10-27).

| spec | what | rows not yet PASS | why | closes with |
|---|---|---|---|---|
| G0 | bit-for-bit comparison against a baseline build | g0.suite | the 2026-11-05 run on main cdcb9c9: bitwise PASS; the suite FAILed on 22 Bolt synthetic golden / prefill tests (failing since at least 2026-11-04, missed in this row then) - fixed since (Closed recently); a suite FAIL stops every later row | the validation rerun on main 5c21b49 |
| **21** Flash-Next | | 32 partial / cli, real-checkpoint rows | golden, pp and serve passed on the branch (2026-11-04, run directly); the rest not yet run | one box_validate run of r32-r34 on main |
| **25** Bolt | | 60, 61, 62 (synthetic and golden chains) | waiting on G0 and the Bolt reference set (building) | after G0 re-baseline + the references |
| **16** pipeline | | 23 (two-card prefill speed), 27 (two-card serving: MTP, lookup, prefix cache) | 27 can run (the spec 16 prefix-cache + MTP fix merged 2026-11-04); 23 is speed, quiet box | one run of 27 in the validation rerun; 23 in the quiet-box session |
| **18** K2-Horizon | | 17.k0, 21.passkey, 25.a4_ref, 25.passkey | the long-context passkey and the tool-call reference (CPU, days) | one run each; 25.a4_ref on the CPU |
| **19** DFlash | | 54.d1_ref (the CPU reference half) | CPU job | one CPU run |
| **13** batching | | 66 check / chunk / cost | speed, idle box | quiet-box session |
| **12** int8 KV | | 11 a4 / bf16 | reference data | one run |

## After the table is empty

1. Quiet-box re-measure: every speed number on the new driver and kernel, the vLLM baseline on the new image, spec 17a's
   first tensor-parallel measurement, the zero-config list B (the arms still waiting on a number - card hand-off copy vs peer, Bolt's fused kernels and bf16 state, Flash-Next's MTP norm, Kolibri / Flash-Next eager attention, int8 KV, int8 prefill, draft vocabulary: measure, keep the winner as the engine's choice, delete the rest), and spec 16's check (MTP draft acceptance cold vs prefix-cache-restored, one card
   and `--pp 2`: < 1% closes the stale-key item); then
   refresh README.md and docs/BENCHMARKS.md (they still say "nothing measured" for Ornith, K2, Flash-Next, Bolt and show
   pre-upgrade headline rows) and the per-model table in docs/18-running-models.md.
2. Spec 13 batching (13b), spec 28m MoE draft-checking probe. Spec 22 expert offload DROPPED 2026-11-06 (host-RAM
   offload does not suit PCIe 3.0; Flash-Next stays a partial-layer development build). (Spec 27 Flint and spec 29 Nanbeige PARKED 2026-11-04 on
   the independent evidence - see the notes at the top of their design docs; their plans were deleted.)
3. Box layout: everything under `~/b70` (branches `box-layout`, `plan-box-move`).

## Closed recently

| date | spec | what |
|---|---|---|
| 2026-11-06 | 20 | Kolibri tool session from a client that drops `reasoning_content` (empty think blocks in earlier turns), same task / settings / two cards as the 11-04 echoing run: 20 turns, 0 stalls, 25 valid / 0 invalid calls, 0 loops (echoing: 19 turns, 0 stalls, 24 valid). Cost: every turn restores a block snapshot instead of continuing (8,698 vs 7,158 re-prefilled ids). Seen in both runs: one 20-24 s prefix-cache store per server run (spec 20 §14). Driver in the repo: `tools/toolcall/kolibri_session.py --no-echo-reasoning` |
| 2026-11-05 | 25 | Bolt's 22 synthetic golden / prefill tests: no engine or reference defect - the random-weight N12 synthetics' routers sit on near-ties, so any two correct bf16 implementations flip an expert within rows 0-2 and the Mamba state spreads it (bf16 reference vs its f32 twin misses 12 / 32 tokens too). Operator rulings (spec 25 §9.1): strict bars only before the first near-tie flip, printed after; the per-block h median binds with >= 8 gated rows; the bf16-state arm's off-2048 chunkings printed (its kernel rounds the state once a chunk by design). Real checkpoints and the fp32 state unchanged. 22 / 22 + bolt_cutoff_test PASS, box_check 316/0/0 |
| 2026-11-05 | 28 | Qwen3.8 speculative tests all PASS: the operator's ruling (option 1) accepts the 4 chaotic turn-boundary positions as REVIEWED - `kReviewedRows` in tests/server/spec_judge.h, exact key / row / ids, stale entries printed (spec 28 §10): t6_usage-openai row 0 (28.63 ulp bf16 head, 28.87 int8 head), t6_usage-rotation row 20 (9.06, bf16 head), dflash_gpu t5_test-box rows 128 / 137 / 143 (10.53 / 11.68 / 16.27); before it the int8-aware f32 judges for the l0-int8 tests (t1_define-loader passes). mtp_verify, mtp_gpu + its two 128k twins, dflash_engine, dflash_gpu, dflash_lookup PASS |
| 2026-11-04 | 28 | Qwen3.8 speculative tests: the bf16-unstable grading rule (bf16 top-k references for 72 contexts, int8-aware prompt pass for the l0-int8 tests), 41 missing f32 judges made, dflash_engine_test's M = 8 check fixed after the attention change; mtp_verify, dflash_engine, dflash_lookup PASS; box_check 315/0/0 |
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
