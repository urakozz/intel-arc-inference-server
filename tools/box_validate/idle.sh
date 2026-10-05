#!/usr/bin/env bash
# The box's grade at this moment, ON THE BOX (docs/10-the-box.md, "Why an idle box matters";
# the same test as tools/bench_decode.sh): RECORD when no container runs and no process
# holds a DRM file descriptor on any card, ITERATE otherwise. Prints
#   IDLE <tag> grade=RECORD|ITERATE containers=N drm_holders=N load=<1-min load average>
# and who holds the cards. Taken before and after every timed stage, inside its GPU lock,
# so a holder that appears mid-run shows up in the "after" line. Quote the grade it printed;
# never upgrade it by hand.
tag="${1:-now}"
containers=$(docker ps -q 2>/dev/null | wc -l | tr -d ' ')
pids=""
for f in /proc/[0-9]*/fdinfo/*; do
  if grep -q drm-driver "$f" 2>/dev/null; then
    p="${f#/proc/}"; p="${p%%/*}"
    case " $pids " in *" $p "*) ;; *) pids="$pids $p" ;; esac
  fi
done
n=0
for p in $pids; do
  n=$((n + 1))
  echo "  drm holder: pid $p $(cat "/proc/$p/comm" 2>/dev/null)"
done
grade=ITERATE
[ "$containers" -eq 0 ] && [ "$n" -eq 0 ] && grade=RECORD
load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)
echo "IDLE $tag grade=$grade containers=$containers drm_holders=$n load=${load:-?}"
