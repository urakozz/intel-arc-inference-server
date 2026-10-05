#!/usr/bin/env bash
# Timed arms as interleaved rounds after a warm-up, ON THE BOX, under the caller's GPU lock
# (this script does not take it - the stage holds it for the whole set, so nothing else
# lands between two arms).
#   tools/box_validate/interleave.sh -o ROWS [-r RUNS] [--ratio A/B]... -- LABEL 'CMD' [LABEL 'CMD']...
# Each arm runs once as a warm-up, then RUNS rounds (default 3), the order reversed every
# other round (A B C, C B A, A B C) so no arm always follows the same one. Every
# `| b70-decode ...` row, `| ...` table row and `BENCH ...` line an arm prints is appended to
# ROWS as "ROW <label> <line>" (warm-ups as "WARM <label> <line>"); uptime is recorded at
# the start, between rounds and at the end. Then tools/box_validate/summary.py medians
# prints the median of the rounds per arm (tg and pp t/s of b70-decode rows, t/s and
# acceptance of BENCH lines) and each requested ratio A/B of the two arms' medians.
# Exit 1 if any arm failed.
set -u
cd "$(dirname "$0")/../.." || exit 2
out="" runs=3 ratios=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -r) runs="$2"; shift 2 ;;
    --ratio) ratios+=(--ratio "$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "usage: $0 -o ROWS [-r RUNS] [--ratio A/B]... -- LABEL 'CMD'..." >&2; exit 2 ;;
  esac
done
[ -n "$out" ] || { echo "interleave.sh: -o ROWS is required" >&2; exit 2; }
labels=(); cmds=()
while [ $# -ge 2 ]; do labels+=("$1"); cmds+=("$2"); shift 2; done
n=${#labels[@]}
[ "$n" -ge 1 ] && [ $# -eq 0 ] || { echo "interleave.sh: arms come as LABEL 'CMD' pairs" >&2; exit 2; }
: > "$out"
fail=0
arm() {   # arm TAG INDEX ROUND
  local tag="$1" i="$2" r="$3" tmp="$out.cur" line rc
  echo "=== $tag ${labels[$i]} round $r $(date +%T) $(uptime | sed 's/.*load/load/')"
  echo "+ ${cmds[$i]}"
  bash -o pipefail -c "${cmds[$i]}" > "$tmp" 2>&1
  rc=$?
  cat "$tmp"
  while IFS= read -r line; do
    case "$line" in
      "| "*|BENCH*) printf '%s %s %s\n' "$tag" "${labels[$i]}" "$line" >> "$out" ;;
    esac
  done < "$tmp"
  if [ "$rc" -ne 0 ]; then echo "ARM FAILED ${labels[$i]} (rc $rc)"; fail=1; fi
}
echo "interleave: $n arm(s), warm-up then $runs round(s); uptime $(uptime)"
for i in $(seq 0 $((n - 1))); do arm WARM "$i" 0; done
for r in $(seq 1 "$runs"); do
  echo "--- round $r: $(uptime)"
  if [ $((r % 2)) -eq 1 ]; then order=$(seq 0 $((n - 1))); else order=$(seq $((n - 1)) -1 0); fi
  for i in $order; do arm ROW "$i" "$r"; done
done
echo "interleave: done; uptime $(uptime)"
rm -f "$out.cur"
python3 tools/box_validate/summary.py medians "$out" ${ratios[@]+"${ratios[@]}"}
exit $fail
