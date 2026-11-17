# Working in this repo

An inference engine for Intel Arc Pro B70 cards. Code is edited on the Mac and built and run on the box.
`docs/open-work.md` is the one list of what is not finished.

## Specs and plans

- Every new design gets the next spec number: `docs/superpowers/specs/<date>-spec<N>-<name>-design.md`. A new
  version of an existing design keeps its number: `spec<N>v2`, `spec<N>v3`.
- A design or plan is reviewed skeptically and the review's findings are folded into the document itself - no separate
  review file or section.
- Its plans carry the number plus a letter in order: `docs/superpowers/plans/<date>-spec<N><letter>-<name>.md`.
- The next free number is the highest under `docs/superpowers/specs/` plus one (`ls docs/superpowers/specs`).
- Every design that brings up a model or a mode (one card, `pp`, `tp`, `replica`, slots) carries two things from its
  first plan, never as a later task: **MTP** (or the model's speculative path) working in that mode, and on two cards
  the **prefill overlap** (two chunks in flight, the card-to-card copies on the copy engine). Modes are compared served,
  MTP on both sides; a plain number is labelled "MTP off".

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
- Stop when the work is merged or reported: stop your box jobs, leave nothing queued.

## Git

- Commits are signed and carry no trailers (no Signed-off-by, Co-authored-by or similar):
  `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S -m "..."`.
- Never push. Never `git branch -D`, `git checkout --`, `git restore`, `git reset --hard` or a bare `git stash`;
  the owner deletes branches.
- Agents work in their own worktree under `.claude/worktrees/<branch>` (`tools/box.sh` gives each branch its own
  tree on the box), never in the main checkout.

## The box

- The Mac never compiles C++. Build and test on the box with `tools/box.sh build|run` (`tools/box.env` names it;
  it is git-ignored and never printed or committed).
- Never write IP addresses or hostnames into files; mask ssh output with
  `sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'`.
- Every GPU job goes through `tools/gpu_lock.sh card <N> -- ...` or `tools/gpu_lock.sh box -- ...`; never set
  `ZE_AFFINITY_MASK` by hand. CPU reference jobs take `/tmp/b70-cpu-model.lock`.
- Lock only while the GPU is in use: wrap one GPU command (a test, a probe, a server run), never a script that
  builds, sleeps, polls or does CPU work. Build before taking the lock; run only the GPU tests the change needs
  (`ctest -R`), not a whole suite. Two-card (`box`) jobs freeze both cards' queues, so batch them. `tools/gpu_lock.sh
  status` lists each holder and flags one with NO GPU OPEN.
- Everything on the box lives under `~/b70`. Nothing is deleted with `rm -rf`: move it to `~/b70/trash/<date>` and
  the owner removes it. Stop jobs only with `tools/box_stop.sh`.
- Long jobs run detached (`nohup setsid`, logs under `~/b70/logs/<topic>/`).
