#!/usr/bin/env bash
# What the box is doing, read-only, for the operator's own terminal (no model, no tokens):
#   tools/box_status.sh            one snapshot, from the Mac (ssh through tools/box.env) or on the box
#   tools/box_status.sh --watch N  a fresh snapshot every N seconds (macOS has no `watch`)
# Shows: the GPU locks, the jobs holding a card or the box (and since when), the jobs waiting for one,
# the CPU model lock, the f32 reference runner's progress with a rough time left, and the logs under
# ~/b70/logs written in the last 30 minutes with their last line. It never takes a lock or stops anything.
set -uo pipefail

if [ "${1:-}" = --watch ]; then
  n="${2:-60}"
  while :; do
    out="$("$0")"
    clear
    printf '%s\n' "$out"
    sleep "$n"
  done
fi

if [ ! -d "$HOME/b70" ]; then   # on the Mac: run this same script on the box
  here="$(cd "$(dirname "$0")" && pwd)"
  "$here/box.sh" run 'bash -s' < "$0" 2>/dev/null | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'
  exit "${PIPESTATUS[0]}"
fi

W=150
cut_w() { cut -c1-"$W"; }
main=~/b70-inference-server

echo "== $(date '+%F %H:%M:%S')  load $(cut -d' ' -f1-3 /proc/loadavg)  RAM free $(awk '/MemAvailable/{printf "%.0f GB", $2/1048576}' /proc/meminfo)"

echo "== locks"
(cd "$main" && tools/gpu_lock.sh status 2>&1) | sed 's/^/  /'

# A gpu_lock.sh process that still exists is waiting: once it has its locks it execs the command.
echo "== running (holding a card or the box)"
for f in "$HOME/b70-gpu0.lock" "$HOME/b70-gpu1.lock" "$HOME/b70-gpu.lock"; do
  [ -e "$f" ] || continue
  case "$f" in *gpu0*) what="card 0";; *gpu1*) what="card 1";; *) what="box";; esac
  pids="$(fuser "$f" 2>/dev/null)"
  for p in $pids; do
    args="$(ps -o args= -p "$p" 2>/dev/null)" || continue
    case "$args" in *gpu_lock.sh*|flock*|"") continue;; esac
    pp="$(ps -o ppid= -p "$p" | tr -d ' ')"
    case " $pids " in *" $pp "*) continue;; esac   # a child of another holder: shown by its parent
    tree="$(readlink "/proc/$p/cwd" 2>/dev/null | sed "s#^$HOME/##")"
    printf '  %-6s %9s  %-28s %s\n' "$what" "$(ps -o etime= -p "$p" | tr -d ' ')" "${tree:-?}" "$args"
  done
done | sort -u -k1,2 | cut_w

echo "== waiting for a lock"
ps -eo pid=,etime=,args= | grep -E 'gpu_lock\.sh (box|card [0-9])' | grep -v grep | while read -r p t a; do
  tree="$(readlink "/proc/$p/cwd" 2>/dev/null | sed "s#^$HOME/##")"
  printf '  %9s  %-28s %s\n' "$t" "${tree:-?}" "${a#*gpu_lock.sh }"
done | cut_w

echo "== CPU model lock"
cp="$(fuser /tmp/b70-cpu-model.lock 2>/dev/null)"
if [ -n "$cp" ]; then
  for p in $cp; do ps -o etime=,args= -p "$p"; done | sed 's/^ */  /' | cut_w
else
  echo "  free"
fi

# The f32 reference runner (tools/oracle/qwen_forced.py): its group, its layer, a rough time left.
for log in $(ls -t "$main"/oracle-out-*.partial/run.log 2>/dev/null | head -1); do
  pgrep -f qwen_forced.py > /dev/null || break
  awk -v L="$log" '
    /^memory group 1\// { cur = 0; k = 0; lay = 0; w = 0 }   # the log appends run after run: the last plan counts
    /^memory group [0-9]+\/[0-9]+/ { split($3, g, "/"); n = g[2] }
    /compute estimate ~/ { match($0, /~[0-9.]+/); h[++k] = substr($0, RSTART + 1, RLENGTH - 1) }
    / layer 0 \(/ { cur++ }
    /^  layer [0-9]+ \(/ { lay = $2; match($0, /wall [0-9]+ s/); w = substr($0, RSTART + 5, RLENGTH - 7) }
    END {
      if (!cur) { print "== f32 references: " L ": loading"; exit }
      if (lay + 1 > 0 && w > 0) per = w / (lay + 1)
      left = (63 - lay) * per / 3600
      for (i = cur + 1; i <= n; i++) left += h[i]
      printf "== f32 references: group %d/%d, layer %d/64, ~%.0f s a layer; ~%.1f h left (later groups by the planner)\n", cur, n, lay + 1, per, left
    }' "$log"
done

echo "== logs written in the last 30 min (newest first)"
find ~/b70/logs "$main"/oracle-out-*.partial -maxdepth 4 -name '*.log' -mmin -30 -size +0 -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -12 | while read -r _ f; do
    last="$(grep -v '^[[:space:]]*$' "$f" | tail -1)"
    done_mark="$(grep -oE '\b[A-Z0-9_]*DONE\b' "$f" | tail -1)"
    printf '  %s%s\n    %s\n' "${f#$HOME/}" "${done_mark:+  [$done_mark]}" "$last"
  done | cut_w
