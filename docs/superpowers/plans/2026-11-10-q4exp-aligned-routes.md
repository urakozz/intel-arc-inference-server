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
