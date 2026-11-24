# Working in this repo

An inference engine for Intel Arc Pro B70 cards (Level Zero, OpenCL C kernels compiled ahead of time with ocloc).
Tests that need model files or reference data skip (exit 77) when those are not present; `B70_DATA` names the
reference data root where a test or tool reads from it (docs/environment-variables.md).

## Code changes

- Every changed line should trace to the task. Match the surrounding code's style, naming and comment density.
- Mention unrelated dead code you notice; don't remove it.
- No single-use abstractions, and no error handling for cases that cannot happen.
- Text a user sees (help, errors, warnings, startup and log lines) never names internal specs, plans, items or
  decision codes ("spec 19f", "plan 13c", "D5"): say what is true and what to do instead. Code comments may cite them.

## Deciding and testing

- Decide from the evidence: once a log, error line or probe points to a cause, pick the fix and make it. Read the
  evidence (logs, error messages) yourself before running more tests.
- Verify a change with ONE targeted check sized to it: the test or run that would fail without the fix. No repeated
  rounds, sweeps or stress series unless that check fails. A one-line change gets one check.
- Stop when the work is merged or reported: stop your jobs, leave nothing queued.
- A performance change that binds on one model is carried to every supported and planned model where it applies:
  build it as a define in the shared kernel source (not a model-only copy), and in the same step either queue each
  other model's byte test and one timing, or note it in that model's board item or plan with the measured
  numbers, or say in one line why it does not apply. Closed negatives are carried the same way.

## Git

- Commits are signed and carry no trailers (no Signed-off-by, Co-authored-by or similar):
  `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S -m "..."`.
- Never push. Never `git branch -D`, `git checkout --`, `git restore`, `git reset --hard` or a bare `git stash`;
  the owner deletes branches.
- Agents work in their own worktree under `.claude/worktrees/<branch>`, never in the main checkout.
