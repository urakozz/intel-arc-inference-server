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
#   tools/box_stop.sh dflash-quant      the DFlash drafter quantisation (tools/quantize/quantize_dflash.sh run,
#                                       ~/b70/hdd/quant/dflash-current): its process group; `start` resumes
#   tools/box_stop.sh tree <branch>      jobs whose working directory is that branch's ~/b70/trees tree, plus its oracle containers
#   tools/box_stop.sh job <branch> <pid> one process tree, verified to be working in that branch
# Anything else is refused. Signals are TERM, then KILL after 10 s for what is left.
# On the box itself (Linux with ~/b70 and no BOX set) the same snippets run locally, without ssh.
set -euo pipefail
# Server harnesses stop their own child on the box, without another ssh hop.
if [ "${1:-}" = child ]; then
  [ "$#" = 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "usage: box_stop.sh child <pid>" >&2; exit 2; }
  child_parent=$(ps -o ppid= -p "$2" 2>/dev/null | tr -d ' ') || exit 0
  [ -n "$child_parent" ] || exit 0
  [ "$child_parent" = "$PPID" ] || { echo "refused: process $2 is not the caller's child" >&2; exit 2; }
  kill -TERM "$2" 2>/dev/null || true
  exit 0
fi
_box_env="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/box.env"
# shellcheck source=/dev/null
[ -r "$_box_env" ] && . "$_box_env"
if [ -z "${BOX:-}" ] && [ "$(uname)" = Linux ] && [ -d "$HOME/b70" ]; then
  remote=(bash -s --)
else
  : "${BOX:?set BOX=user@host in the environment or in tools/box.env (see tools/box.env.example)}"
  remote=(ssh -o ConnectTimeout=20 "$BOX" bash -s --)
fi

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
what="${1:-}"; arg="${2:-}"; job_pid="${3:-}"
case "$what" in
  list|cpu-chain|hf-tier|validation|q4exp-count-wait|dflash-quant) [ -z "$arg" ] || usage ;;
  cpu-job|tree) [[ "$arg" =~ ^[A-Za-z0-9._-]+$ ]] || usage ;;
  job) [[ "$arg" =~ ^[A-Za-z0-9._-]+$ && "$job_pid" =~ ^[1-9][0-9]*$ ]] || usage ;;
  *) usage ;;
esac

# The remote side: fixed snippets only; the one argument is checked above and again on the box.
# shellcheck disable=SC2016
"${remote[@]}" "$what" "$arg" "$job_pid" <<'REMOTE' | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'
set -u
what=$1; arg=${2:-}; job_pid=${3:-}
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
  dflash-quant)
    run=$(readlink -f "$HOME/b70/hdd/quant/dflash-current" 2>/dev/null || true)
    p=$(cat "$run.work/run.pid" 2>/dev/null || true)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null || { echo "no DFlash quantisation is running"; exit 0; }
    tr '\0' ' ' < "/proc/$p/cmdline" | grep -q 'quantize_dflash.sh run' ||
      { echo "refused: pid $p is not quantize_dflash.sh run"; exit 2; }
    stop_pids $(pgrep -g "$p")
    echo "resume with: tools/quantize/quantize_dflash.sh start"
    ;;
  list)
    echo "== cpu chain";   pgrep -af 'b70/scripts/cpu-refs-chain.sh' | grep -v pgrep || echo "  not running"
    echo "== cpu jobs";    for f in "$CPU_LOG"/state/*.pid; do [ -e "$f" ] || continue; p=$(cat "$f"); kill -0 "$p" 2>/dev/null && echo "  $(basename "$f" .pid) pid $p"; done
    echo "== cpu containers"; c=$(cpu_containers); [ -n "$c" ] && docker ps --filter "volume=$CPU_TREE" --format '  {{.ID}} {{.RunningFor}}' || echo "  none"
    echo "== hf-tier";     pgrep -af 'hf-tier-chain|tools/hf_tier.py' | grep -v pgrep || echo "  not running"
    echo "== validation";  pgrep -af 'box_validate/remote.sh' | grep -v pgrep | cut -c1-160 || echo "  not running"
    echo "== dflash-quant"; pgrep -af 'quantize_dflash.sh run' | grep -v pgrep || echo "  not running"
    ;;
  job)
    tree="$HOME/b70/trees/$arg"
    wd=$(readlink "/proc/$job_pid/cwd" 2>/dev/null || true)
    case "$wd" in
      "$tree"|"$tree"/*) stop_pids $(tree_pids "$job_pid") ;;
      *) echo "refused: process $job_pid is not working in branch '$arg'"; exit 2 ;;
    esac
    ;;
  tree)
    tree="$HOME/b70/trees/$arg"
    [ -d "$tree" ] || { echo "refused: no branch tree '$arg'"; exit 2; }
    c=$(docker ps -q --filter "volume=$tree")
    pids=""
    for f in /proc/[0-9]*/cwd; do
      wd=$(readlink "$f" 2>/dev/null || true)
      case "$wd" in
        "$tree"|"$tree"/*)
          p=${f#/proc/}; p=${p%/cwd}
          [ "$p" = "$$" ] || pids="$pids $p"
          ;;
      esac
    done
    stop_pids $pids
    [ -z "$c" ] || { echo "stopping branch oracle container(s): $c"; docker kill $c >/dev/null; }
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
