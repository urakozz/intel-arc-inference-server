# Sourced by tools/box.sh and tools/serve_bench.sh: which tree on the box this
# checkout syncs to. Several worktrees (one per plan) can work in parallel, and
# box.sh syncs with --delete, so each line of work needs its own remote tree.
#
#   REMOTE_DIR set          that directory, as before (relative to $HOME on the box)
#   BOX_SUFFIX=spec8a       b70-inference-server-spec8a
#   BOX_SUFFIX=none         b70-inference-server (force the base tree)
#   neither                 the suffix is read from the branch name: a branch
#                           starting spec<N>[.<M>][<letter>] (spec7a-toolcall-probe,
#                           spec8-mtp, spec2.1-...) gets -spec7a / -spec8 / -spec2.1;
#                           research-* gets -research; anything else (main, ...)
#                           the base tree b70-inference-server.
#
# A new remote tree starts empty: its first `tools/box.sh build` is a full build.
# GPU work from parallel trees must still take the shared lock:
#   flock ~/b70-gpu.lock <cmd>
b70_remote_dir() {
  if [ -n "${REMOTE_DIR:-}" ]; then
    printf '%s\n' "$REMOTE_DIR"
    return
  fi
  local base="b70-inference-server" suffix="${BOX_SUFFIX:-}" branch
  if [ -z "$suffix" ]; then
    branch="$(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ "$branch" =~ ^(spec[0-9]+(\.[0-9]+)?[a-z]?) ]]; then
      suffix="${BASH_REMATCH[1]}"
    elif [[ "$branch" == research-* ]]; then
      suffix="research"
    fi
  fi
  if [ -z "$suffix" ] || [ "$suffix" = "none" ]; then
    printf '%s\n' "$base"
  else
    printf '%s\n' "$base-$suffix"
  fi
}
