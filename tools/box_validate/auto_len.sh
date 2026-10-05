#!/usr/bin/env bash
# The length `--max-len auto` picks for one configuration, ON THE BOX, under the caller's
# GPU lock: prints N alone (b70-decode's "max_len: auto -> N" line).
#   tools/box_validate/auto_len.sh <snapshot> [b70-decode flags...]
# Default flags `--depth 16 --tg 1` (decode only). The plan counts the prefill scratch only
# when the run prefills (spec 6 §10, ruling R7), so pass `--pp 256 --tg 1 --pp-backend B` to
# get the length a `--prefill` run (passkey.sh) is planned at.
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 1 ] || { echo "usage: $0 <snapshot> [b70-decode flags...]" >&2; exit 2; }
snap="$1"; shift
[ $# -gt 0 ] || set -- --depth 16 --tg 1
out=$(B70_GIT_SHA=auto-len build/src/cli/b70-decode "$snap" --bench --max-len auto "$@" 2>&1)
n=$(printf '%s\n' "$out" | sed -n 's/.*max_len: auto -> \([0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$n" ]; then
  printf '%s\n' "$out" | tail -20 >&2
  echo "auto_len.sh: no 'max_len: auto -> N' line" >&2
  exit 1
fi
echo "$n"
