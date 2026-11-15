# lookup-wide-verify handoff (open-work 28)

Branch `lookup-wide-verify` (worktree `.claude/worktrees/lookup-wide-verify`), based on item 25's `verify-cost-m8`
4abdba81 (operator GO 2026-11-15: rebase onto main once item 25 lands - its merged form drops the 16-key-block
attention lever, so `runtime::verify_attn_dpas_blocks`'s M <= 8 edit here goes with it; the wide list keeps 32 blocks).
Record: `docs/probes/probe-lookup-wide-verify-2026-11-14.md` (steps 1-3). Box tree `b70/trees/lookup-wide-verify`,
logs `~/b70/logs/item28/` (probe/, quick/, check/).

## State
- Steps 1-2 done (record). Step 3 built (record, "Step 3: what was built"); host tests pass; quick card cases pass
  (gdn_step, gdn_step_slots, argmax, embed_gather); smoke on the 8k edit / copy prompts: ids identical to item 27's,
  gen t/s 335 / 288 / 133 against 115 / 111 / 93.
- The one batched check: `tools/probe/item28_check.sh 1 ~/b70/logs/item28/check` (judged tests, the sets, the
  replay, tables against item 27's lk arms). mtp_verify_test failed there on a state-size literal (ring 16) - fixed
  after the run started (commit "state-size literals"); rerun it alone after the check ends (do not rebuild the tree
  while the check runs: it uses build/).

## Not built
- The 16-row list for DFlash's lookup (16 tap rows, a 16-row drafter projection); int8 KV at 16 rows.

## Box lesson
- A detached HEAD once sent box.sh to the base tree (fixed on main 7ed60c7d, taken here). Always check
  `tools/box.sh dir` before a build.
