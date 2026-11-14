# Working in this repo

An inference engine for Intel Arc Pro B70 cards. Code is edited on the Mac and built and run on the box.
`docs/open-work.md` is the one list of what is not finished.

## Code changes

- Every changed line should trace to the task. Match the surrounding code's style, naming and comment density.
- Mention unrelated dead code you notice; don't remove it.
- No single-use abstractions, and no error handling for cases that cannot happen.

## Git

- Commits are signed and carry no trailers (no Signed-off-by, Co-authored-by or similar):
  `git -c gpg.program=$(brew --prefix)/bin/gpg commit -S -m "..."`.
- Never push. Never `git branch -D`, `git checkout --`, `git restore`, `git reset --hard` or a bare `git stash`;
  the operator deletes branches.
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
  the operator removes it. Stop jobs only with `tools/box_stop.sh`.
- Long jobs run detached (`nohup setsid`, logs under `~/b70/logs/<topic>/`).
