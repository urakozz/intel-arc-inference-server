#!/usr/bin/env bash
# Stop our own long-running jobs on the box - nothing else. From the Mac:
#   tools/box_stop.sh list              what is running (read only)
#   tools/box_stop.sh cpu-chain         the CPU reference chain loop (~/b70/scripts/cpu-refs-chain.sh);
#                                       its current job keeps running - `cpu-job` stops that
#   tools/box_stop.sh cpu-job <name>    one CPU queue job (a name in ~/b70/logs/cpu-refs/queue.txt):
#                                       its process tree and its docker container; the chain logs
#                                       `END rc=143` and moves on
#   tools/box_stop.sh hf-tier           the hf_tier copy chain (~/b70/scripts/hf-tier-chain*.sh) and
#                                       its tools/hf_tier.py; an interrupted copy resumes next time,
#                                       a swap is one atomic rename
#   tools/box_stop.sh validation        the newest box_validate run (its remote.sh process group);
#                                       re-running tools/box_validate.sh with the same state resumes
#   tools/box_stop.sh q4exp-count-wait  the Flash-Next count-gate driver, only while its GPU job is waiting
# Anything else is refused. Signals are TERM, then KILL after 10 s for what is left.
set -euo pipefail
_box_env="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/box.env"
# shellcheck source=/dev/null
[ -r "$_box_env" ] && . "$_box_env"
: "${BOX:?set BOX=user@host in the environment or in tools/box.env (see tools/box.env.example)}"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
what="${1:-}"; arg="${2:-}"
case "$what" in
  list|cpu-chain|hf-tier|validation|q4exp-count-wait) [ -z "$arg" ] || usage ;;
  cpu-job) [[ "$arg" =~ ^[A-Za-z0-9._-]+$ ]] || usage ;;
  *) usage ;;
esac

# The remote side: fixed snippets only; the one argument is checked above and again on the box.
# shellcheck disable=SC2016
ssh -o ConnectTimeout=20 "$BOX" bash -s -- "$what" "$arg" <<'REMOTE' | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'
set -u
what=$1; arg=${2:-}
CPU_TREE=$HOME/b70/trees/cpu-refs
CPU_LOG=$HOME/b70/logs/cpu-refs
VROOT=$HOME/b70-validate/b70-inference-server-validate

tree_pids() {  # $1 and every descendant, children first
  local p
  for p in $(pgrep -P "$1"); do tree_pids "$p"; done
  echo "$1"
}
stop_pids() {  # TERM, wait up to 10 s, KILL what is left
  [ $# -gt 0 ] || { echo "nothing to stop"; return 0; }
  echo "stopping: $*"
  kill -TERM "$@" 2>/dev/null
  for _ in $(seq 1 10); do
    local alive=0; for p in "$@"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && { echo "stopped"; return 0; }
    sleep 1
  done
  kill -KILL "$@" 2>/dev/null; echo "stopped (KILL after 10 s)"
}
cpu_containers() { docker ps -q --filter "volume=$CPU_TREE"; }

case "$what" in
  q4exp-count-wait)
    task_tree="$HOME/b70/trees/q4exp-f32-ref"
    driver='bash .superpowers/sdd/2026-11-08-flash-next-f32-reference/count-gates.sh'
    stopped=0
    for p in $(pgrep -f "^$driver$"); do
      [ "$(readlink "/proc/$p/cwd")" = "$task_tree" ] || continue
      pids=$(tree_pids "$p")
      waiting=0
      for child in $pids; do
        cmd=$(tr '\0' ' ' < "/proc/$child/cmdline" 2>/dev/null)
        case "$cmd" in
          "$driver "|"bash tools/gpu_lock.sh card "*) ;;
          "flock "*) waiting=1 ;;
          *) echo "refused: count driver has active or unexpected child $child: $cmd"; exit 2 ;;
        esac
      done
      [ "$waiting" = 1 ] || { echo "refused: count driver is not waiting for a lock"; exit 2; }
      # Stop the driver first, so a failed lock child cannot advance it to the next GPU test.
      stop_pids "$p" $(printf '%s\n' "$pids" | grep -vx "$p")
      stopped=1
    done
    [ "$stopped" = 1 ] || { echo "no waiting Flash-Next count driver"; exit 2; }
    ;;
  list)
    echo "== cpu chain";   pgrep -af 'b70/scripts/cpu-refs-chain.sh' | grep -v pgrep || echo "  not running"
    echo "== cpu jobs";    for f in "$CPU_LOG"/state/*.pid; do [ -e "$f" ] || continue; p=$(cat "$f"); kill -0 "$p" 2>/dev/null && echo "  $(basename "$f" .pid) pid $p"; done
    echo "== cpu containers"; c=$(cpu_containers); [ -n "$c" ] && docker ps --filter "volume=$CPU_TREE" --format '  {{.ID}} {{.RunningFor}}' || echo "  none"
    echo "== hf-tier";     pgrep -af 'hf-tier-chain|tools/hf_tier.py' | grep -v pgrep || echo "  not running"
    echo "== validation";  pgrep -af 'box_validate/remote.sh' | grep -v pgrep | cut -c1-160 || echo "  not running"
    ;;
  cpu-chain)
    stop_pids $(pgrep -f 'b70/scripts/cpu-refs-chain.sh' | grep -vx $$)
    ;;
  cpu-job)
    grep -qE "^(#[A-Z-]+ )?$arg\|" "$CPU_LOG/queue.txt" || { echo "refused: '$arg' is not a job in $CPU_LOG/queue.txt"; exit 2; }
    # Snapshot before checking the PID: stopping it releases the CPU lock, and the
    # chain can start the next container during stop_pids' ten-second wait.
    c=$(cpu_containers)
    pf=$CPU_LOG/state/$arg.pid
    p=$(cat "$pf" 2>/dev/null || true)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      stop_pids $(tree_pids "$p")
      [ -z "$c" ] || { echo "killing the selected CPU job's container(s): $c"; docker kill $c >/dev/null; }
    else
      echo "job $arg has no live process"
    fi
    ;;
  hf-tier)
    pids=""; for p in $(pgrep -f 'b70/scripts/hf-tier-chain'); do pids="$pids $(tree_pids "$p")"; done
    for p in $(pgrep -f 'tools/hf_tier.py'); do case " $pids " in *" $p "*) ;; *) pids="$pids $p";; esac; done
    stop_pids $pids
    ;;
  validation)
    v=$(ls -td "$VROOT"/*-vs-* 2>/dev/null | head -1); [ -n "$v" ] || { echo "no validation state"; exit 0; }
    pids=$(pgrep -f "box_validate/remote.sh $v" || true)
    [ -n "$pids" ] || { echo "validation $(basename "$v") is not running"; exit 0; }
    groups=$(for p in $pids; do ps -o pgid= -p "$p"; done | tr -d ' ' | sort -u)
    echo "validation $(basename "$v"): process group(s) $groups"
    for g in $groups; do kill -TERM -- "-$g" 2>/dev/null; done
    sleep 10
    for g in $groups; do pgrep -g "$g" >/dev/null && kill -KILL -- "-$g" 2>/dev/null; done
    echo "stopped; resume with: tools/box_validate.sh --state $(basename "$v") --no-wait"
    ;;
esac
REMOTE
