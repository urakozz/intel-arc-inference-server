# lookup-wide-verify handoff (open-work 28)

Branch `lookup-wide-verify` (worktree `.claude/worktrees/lookup-wide-verify`), on main
(rebased after item 25 merged).
Record: `docs/probes/probe-lookup-wide-verify-2026-11-14.md` (steps 1-3). Box tree `b70/trees/lookup-wide-verify`,
logs `~/b70/logs/item28/` (probe/, quick/, check/).

## State
- Rebased onto main (after item 25's merge); 16-row list built; the check run (record, "The check"): edit / copy
  181 t/s (92.3 before), replay 83.8 (68.6), dflash_gpu_test and mtp_verify_test PASS, mtp_lookup_gpu_test D2 0
  failing with 7 tool-call prompts awaiting f32 references (dumped in ~/b70/logs/item28/forced/), part H's new bar
  (an operator ruling to confirm) passes.
- Morning: the f32 references (record, the commands), then mtp_lookup_gpu_test once.
- Box tree builds: `build` (pre-rebase, the check's) and `build-rebased` (current). Use build-rebased.

## Not built
- The 16-row list for DFlash's lookup (16 tap rows, a 16-row drafter projection); int8 KV at 16 rows.

## Box lesson
- A detached HEAD once sent box.sh to the base tree (fixed on main 7ed60c7d, taken here). Always check
  `tools/box.sh dir` before a build.
