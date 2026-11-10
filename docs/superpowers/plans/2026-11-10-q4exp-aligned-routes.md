# Flash-Next aligned router reference

User-approved rule: within the f32 router cut's one-bf16-ulp near-tie window, replay the engine's expert set in the reference. Judge every swap before applying it. Preserve fp32 mixing weights and the 2x state/logits bar. No engine implementation or environment changes, push, or merge.

The existing Ornith judge supplies the logit-ulp rule. Replay must check the actual swapped experts (including multiway boundary ties); a near adjacent gap alone cannot excuse an unrelated expert. Separate prefill/decode captures produce separate fp32 companions, since their admissible expert sets can differ. Both use the existing unaligned f32 continuation as a fixed teacher-forced context, avoiding a circular reference-generation dependency. Invalid swaps stay unforced and fail routing; residual state comparisons cannot be attributed to prefill while routing is unaligned.

- [x] Add a tested Python route judge/replay policy: exact sets, admissible single/multiway ties, reject distant/dropped high-score experts, invalid IDs, recompute fp32 weights.
- [x] Capture all prompt and forced-generation engine routes for prefill and decode without new engine flags; capture states/logits for replay checks.
- [x] Add aligned oracle job and paired dump format/reader. Validate prompt/forced context, trace identity and accepted routing before applying the unchanged 2x bar.
- [x] Verify bounded four-layer/eight-token alignment first, then L18/L37 capped CPU companions and locked registered GPU gates; record per-kind distances and any invalid route choices.
- [x] Signed commits per step, docs/Waiting21 and uncommitted REPORT updated; branch stays unmerged.

Execution: inline, worktree q4exp-f32-ref. Existing box/lock/64g/sanitization rules remain in force. Negative controls must demonstrate that unconditional forcing cannot pass.

Implementation reviewed with the existing diagnostic agent. Seven route/packaging tests PASS on the box; both capture and paired reader targets build PASS. Two wrapper tests PASS, including inherited110g overridden to64g. Bounded replay and full route captures were detached at07:55:49UTC; CPU replay waits on the shared CPU-model lock. Measurement steps remain open.

08:29UTC: L18 engine capture complete; L37 waits for whole-box lock. User states release is at least4h away and requests no frequent checks. Detached `~/b70/logs/q4exp-f32/aligned-full.log` runner uses filesystem notifications to wait for completed captures, then the CPU lock; runs bounded validation, host regressions, both paired references and registered locked gates automatically. No status checks before12:30UTC. Measurement and final records remain pending.


## Completed measurements

Both paired references and both registered gates completed; execution checkboxes do not claim passing
numerical gates. Full build and four host CTests PASS. Oracle host checks: 35 PASS / one optional
real-index fixture SKIP. Bounded replay: 27 exact decisions, five accepted near-ties, zero rejected.

L18: FAIL (345.63 s), four conv kinds and logits above 2x; tokens decode/prefill 32/32 and 32/32.
L37: FAIL (755.44 s), 15 conv kinds, two GDN kinds, three QSA tails and logits above 2x; tokens 28/32
and 29/32. Both also fail six route-distance ratios and the routing audit. Reject counts, decode/prefill:
L18 1089/1279; L37 8413/8885. First decode rejection is already L0/position62. Thus these remain
partially aligned paths and do not establish a prefill-specific defect. No route was forced outside
its f32-judged window, and the acceptance bar was not widened.

CPU companion wall/RSS: L18 prefill578.8s/45.1GiB, decode516.7s/45.2GiB; L37
prefill1111.9s/68.2GiB, decode991.3s/68.2GiB. Configured64g/no-swap cap and16threads throughout;
process RSS is not cgroup memory.peak. Branch stays unmerged and Waiting21 remains open.

[Results and interpretation](../../probes/q4exp-f32-aligned-routes-2026-11-10.md),
[208 distances](../../probes/q4exp-f32-aligned-judgment-2026-11-10.csv),
[110 per-layer route audits](../../probes/q4exp-f32-route-judgment-2026-11-10.csv).


## User-directed count gate (supersedes the 2x gate above)

New rule: count all f32-rejected layer-position routes on each paired path; prefill may have at most
1.2 times decode's count. Prefill must accept at least as many of the 32 teacher-forced tokens as
decode. Keep f32 near-tie classification/replay to obtain the counts; individual rejections are no
longer an automatic failure. State, logits, route-set and selection distances are printed diagnostics.
Malformed data, prompt/capture identity, deterministic/chunking/injected-run invariants still fail.

Use integer comparison with no upward rounding of the 1.2x budget, including decode-zero and uint64
overflow boundaries. Count comparison requires the complete paired v2 audit; legacy v1 dumps do not
contain it. CMake will enable the new judge only for the paired dump; absent paired data keeps A'.
Explicit legacy judge inputs will explain that paired counts are required rather than use the old gate.
This is an operator-selected change of acceptance policy, not evidence that the measured state drift
has disappeared. No engine/reference regeneration is needed for this policy change.

- [x] Add count-policy boundary and negative controls; watch RED then GREEN on box.
- [x] Wire complete audit counts and token totals into the new gate; keep all distances diagnostic.
- [ ] Build, review and rerun registered L18/L37 gates on existing paired data under GPU locks.
- [ ] Record new verdicts, policy change, signed commits and uncommitted report; remain unmerged.


Count implementation verification: the helper and NaN/Inf validity controls each failed to compile
before their APIs existed, then the completed code built and all four focused host CTests passed.
Review's finite-reference finding is fixed without adding an engine-distance gate. Final GPU run was
requeued from occupied GPU0 to GPU1 at17:07UTC on user instruction; the original driver was stopped
while waiting, through the narrowly guarded `tools/box_stop.sh q4exp-count-wait`. GPU1 L18 and the
subsequent two-GPU L37 registered checks use the existing paired references. Logs are
`~/b70/logs/q4exp-f32/count-{gpu1,one-card-gpu1,r33-pp}.log`; first scheduled status check17:37UTC.


17:16:30UTC, user-requested status check: registered one-card L18 count gate **PASS** on GPU1,
346.06s, finished17:12:56UTC. Tokens accepted32/32 on both paths; determinism, chunking, capture
identity and injected-selection checks pass. The driver advanced to the L37 whole-box GPU lock;
GPU0 is held by another test. L37 verdict remains pending; no CPU reference rerun.
