#!/usr/bin/env bash
# The box's GPU locks, ON THE BOX: the whole box, or one card beside a job on the other card.
#
#   tools/gpu_lock.sh box -- <cmd...>        the whole box: timed rows, two-card work, an idle grade
#   tools/gpu_lock.sh card <N> -- <cmd...>   card N only, the command run with ZE_AFFINITY_MASK=N
#   tools/gpu_lock.sh status                 what is held right now (non-blocking probes)
#
# The command is exec'd holding the locks; they go when it (and whatever it started that kept
# the descriptors) exits. Its exit status is the command's; 2 when a lock could not be taken.
# tools/box_validate/remote.sh sources this file for gpu_lock_box / gpu_lock_card.
#
# The protocol - flock(2) on three kinds of file in $B70_LOCK_DIR (default $HOME):
#   b70-gpu.lock    the box. Whole box: EXCLUSIVE. Card job: SHARED, held for the whole job.
#   b70-gpu<N>.lock card N, exclusive (card jobs only).
#   b70-gpu.wait    the gate. A whole-box taker holds it exclusively while it WAITS for
#                   b70-gpu.lock and drops it once it has the box; a card taker takes it and
#                   drops it at once, just before its shared b70-gpu.lock.
# Order: card <N> -> gate -> box. A whole-box taker never takes a card lock, a card job
# never waits for anything once it holds the box shared, so there is no cycle. Taking the card
# first means a card job that waits for its card holds nothing a whole-box job needs.
#
# Why the gate: Linux flock(2) has no queue and no fairness. A shared lock is granted whenever
# no EXCLUSIVE lock is HELD - an exclusive request that is only WAITING does not stop it - and
# a release wakes every waiter to race. Two card streams that overlap (one always running)
# would starve a whole-box job forever. With the gate, once a whole-box job waits, a new card
# job stops at the gate: the box drains to the card jobs already running and the whole-box job
# runs next. Whole-box jobs queue behind each other on the gate and win over card jobs (timed
# work first); card jobs only ever wait for running jobs.
#
# Backward compatible: `flock ~/b70-gpu.lock <cmd>` (exclusive, no gate) still excludes every
# card job and every other whole-box job - it only lacks the priority, so while card streams
# run, use `tools/gpu_lock.sh box -- <cmd>`. Never nest: a command holding any of these locks
# must not take another (flock on a second open of the same file waits for itself).

gpu_lock_dir() { printf '%s\n' "${B70_LOCK_DIR:-$HOME}"; }

# _gpu_take FD MODE WHAT - flock MODE (-x / -s) on FD, saying so when it has to wait
_gpu_take() {
  if flock -n "$2" "$1"; then return 0; fi
  echo "gpu_lock: waiting for $3 ($(date '+%F %T'))" >&2
  flock "$2" "$1"
}

# gpu_lock_box - the whole box: the gate (fd 6) while waiting, then b70-gpu.lock exclusive (fd 9)
gpu_lock_box() {
  local d; d=$(gpu_lock_dir)
  exec 6> "$d/b70-gpu.wait" || return 2
  _gpu_take 6 -x "the gate (another whole-box job waits first)" || return 2
  exec 9> "$d/b70-gpu.lock" || return 2
  _gpu_take 9 -x "the whole box (b70-gpu.lock: running jobs drain)" || return 2
  exec 6>&-
}

# gpu_lock_card N - card N: b70-gpu<N>.lock exclusive (fd 5), the gate taken and dropped (fd 6),
# b70-gpu.lock shared (fd 9); exports ZE_AFFINITY_MASK=N
gpu_lock_card() {
  local d n="${1:-}"; d=$(gpu_lock_dir)
  case "$n" in
    ''|*[!0-9]*) echo "gpu_lock: a card is one device index (0, 1), not '$n'" >&2; return 2 ;;
  esac
  exec 5> "$d/b70-gpu$n.lock" || return 2
  _gpu_take 5 -x "card $n (b70-gpu$n.lock)" || return 2
  exec 6> "$d/b70-gpu.wait" || return 2
  _gpu_take 6 -x "the gate (a whole-box job waits: it runs first)" || return 2
  exec 6>&-
  exec 9> "$d/b70-gpu.lock" || return 2
  _gpu_take 9 -s "the box, shared (a whole-box job runs)" || return 2
  export ZE_AFFINITY_MASK="$n"
}

# gpu_lock_status - one line per lock: free / held, from non-blocking probes (they take nothing)
gpu_lock_status() {
  local d f n; d=$(gpu_lock_dir)
  if flock -n -x "$d/b70-gpu.lock" true; then echo "box     free"
  elif flock -n -s "$d/b70-gpu.lock" true; then echo "box     shared (card jobs run)"
  else echo "box     EXCLUSIVE (a whole-box job runs)"; fi
  if flock -n -x "$d/b70-gpu.wait" true; then echo "gate    free"
  else echo "gate    held (a whole-box job waits: new card jobs queue behind it)"; fi
  for f in "$d"/b70-gpu[0-9]*.lock; do
    [ -e "$f" ] || continue
    n=${f##*/b70-gpu}; n=${n%.lock}
    if flock -n -x "$f" true; then echo "card $n  free"; else echo "card $n  held"; fi
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
  case "${1:-}" in
    box) shift ;;
    card) _gpu_card="${2:-}"; shift; [ $# -gt 0 ] && shift ;;
    status) gpu_lock_status; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  [ "${1:-}" = -- ] && shift
  if [ $# -eq 0 ]; then echo "gpu_lock: no command (tools/gpu_lock.sh --help)" >&2; exit 2; fi
  if [ -n "${_gpu_card+x}" ]; then gpu_lock_card "$_gpu_card" || exit 2; else gpu_lock_box || exit 2; fi
  exec "$@"
fi
