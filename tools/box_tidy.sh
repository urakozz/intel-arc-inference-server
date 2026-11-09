#!/usr/bin/env bash
# Tidy the box (operator 2026-11-10: finished work kept piling up on the 915 GB system disk). From the Mac:
#   tools/box_tidy.sh            move to ~/b70/trash/<date>-tidy:
#                                  - ~/b70/trees/<branch> whose branch no longer exists in this checkout (merged and
#                                    deleted), except the long-lived trees below and any tree a process runs in;
#                                  - ~/b70-inference-server-g0-<sha> baseline trees except the KEEP_G0 newest (default 2)
#                                    and any a running box_validate uses;
#                                then print the trash's size (only the operator's rm frees it).
#   tools/box_tidy.sh --dry-run  print what would move, move nothing.
# Nothing is deleted. tools/box_validate.sh runs it (quietly) at every launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
_box_env="$PWD/tools/box.env"
# shellcheck source=/dev/null
[ -r "$_box_env" ] && . "$_box_env"
: "${BOX:?set BOX=user@host in the environment or in tools/box.env (see tools/box.env.example)}"
dry=0
[ "${1:-}" = --dry-run ] && dry=1
# The local branches, as the box tree names tools/box_dir.sh gives them (a branch's slashes become dashes).
branches=$(git for-each-ref --format='%(refname:short)' refs/heads | tr '/' '-' | tr '\n' ' ')
ssh -o ConnectTimeout=20 "$BOX" bash -s -- "$dry" "${KEEP_G0:-2}" "$branches" <<'REMOTE' | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'
set -u
dry=$1 keep=$2 branches=" $3 "
keepers=" cpu-refs hf-tier validate-reuse main "
T=~/b70/trash/$(date +%F)-tidy
busy() { pgrep -f "$1" > /dev/null 2>&1; }
move() {
  if [ "$dry" = 1 ]; then echo "would move $1"; return; fi
  mkdir -p "$T" && mv "$1" "$T/" && echo "moved $1"
}
for d in ~/b70/trees/*/; do
  n=$(basename "$d")
  case "$keepers" in *" $n "*) continue ;; esac
  case "$branches" in *" $n "*) continue ;; esac
  busy "b70/trees/$n" && { echo "kept $d (a process runs in it)"; continue; }
  move "${d%/}"
done
i=0
for g in $(ls -td ~/b70-inference-server-g0-* 2>/dev/null); do
  i=$((i + 1))
  [ "$i" -le "$keep" ] && continue
  busy "$(basename "$g")" && { echo "kept $g (in use)"; continue; }
  move "$g"
done
[ "$dry" = 1 ] || { [ -d "$T" ] && echo "trash: $(du -sh ~/b70/trash | cut -f1) in ~/b70/trash (the operator's rm frees it)"; }
df -h / | tail -1 | awk '{print "system disk: " $5 " used, " $4 " free"}'
REMOTE
