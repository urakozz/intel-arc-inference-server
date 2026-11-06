# Sourced by tools/box.sh and tools/serve_bench.sh: which tree on the box this
# checkout syncs to. Several worktrees (one per plan) can work in parallel, and
# box.sh syncs with --delete, so each line of work needs its own remote tree.
# Paths are relative to $HOME on the box.
#
#   REMOTE_DIR set          that directory
#   BOX_SUFFIX=validate     b70-inference-server-validate (an explicit suffix keeps the
#                           old top-level name: the validation campaign's tree)
#   BOX_SUFFIX=none         b70-inference-server (force the base tree)
#   neither                 from the branch name, under ~/b70/trees: a branch starting
#                           spec<N>[.<M>][<letter>] (spec7a-toolcall-probe, spec8-mtp,
#                           spec2.1-...) -> b70/trees/spec7a / spec8 / spec2.1;
#                           research-* -> b70/trees/research; any other branch
#                           -> b70/trees/<branch> (a / becomes -); main (or a detached
#                           HEAD) -> the base tree b70-inference-server, which also
#                           holds the golden sets (oracle-out*, never synced over).
#
# A new remote tree starts empty: its first `tools/box.sh build` is a full build.
# GPU work from parallel trees must still take the shared locks (tools/gpu_lock.sh,
# run on the box from the tree):
#   tools/gpu_lock.sh box -- <cmd>        the whole box: timed, two-card, idle-graded work
#   tools/gpu_lock.sh card N -- <cmd>     card N only (ZE_AFFINITY_MASK=N), beside a job on
#                                         the other card
# `flock ~/b70-gpu.lock <cmd>` still takes the whole box, without the gate's priority over
# card jobs (docs/box-day-plan.md §1.8).
b70_remote_dir() {
  if [ -n "${REMOTE_DIR:-}" ]; then
    printf '%s\n' "$REMOTE_DIR"
    return
  fi
  local base="b70-inference-server" suffix="${BOX_SUFFIX:-}" branch
  if [ -z "$suffix" ]; then
    branch="$(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    # Every branch but main gets its own tree under ~/b70/trees: nothing new at the top of
    # the box's $HOME, and the base tree holds the golden sets, so the sync's --delete must
    # not land there.
    if [[ "$branch" =~ ^(spec[0-9]+(\.[0-9]+)?[a-z]?) ]]; then
      printf 'b70/trees/%s\n' "${BASH_REMATCH[1]}"
      return
    elif [[ "$branch" == research-* ]]; then
      printf 'b70/trees/research\n'
      return
    elif [ -n "$branch" ] && [ "$branch" != main ] && [ "$branch" != HEAD ]; then
      printf 'b70/trees/%s\n' "${branch//\//-}"
      return
    fi
  fi
  if [ -z "$suffix" ] || [ "$suffix" = "none" ]; then
    printf '%s\n' "$base"
  else
    printf '%s\n' "$base-$suffix"
  fi
}
