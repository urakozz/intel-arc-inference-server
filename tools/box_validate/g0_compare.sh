#!/usr/bin/env bash
# G0's binary half: every kernel binary built in both trees must be byte-identical.
#   tools/box_validate/g0_compare.sh <baseline build dir> <build dir under test> <out dir>
# Lists every *.bin under a kernels/ directory of each build (ocloc's output,
# cmake/ocloc.cmake: <build>/kernels/<variant>.bin), sha256 of each, and compares by path:
#   identical - present in both, same sha256          (the expectation for every one)
#   differ    - present in both, different sha256     -> FAIL
#   added     - only under test (a new variant)       listed, expected on a kernel branch
#   removed   - only in the baseline                  -> FAIL (G0_ALLOW_REMOVED=1 to accept)
# Writes baseline.sha, under-test.sha, differ.txt, added.txt, removed.txt into <out dir> and
# prints one "G0 ..." summary line. Exit 0 pass, 1 fail, 2 usage / nothing to compare.
set -u
[ $# -eq 3 ] || { echo "usage: $0 <baseline build> <build under test> <out dir>" >&2; exit 2; }
base="$1" head="$2" out="$3"
mkdir -p "$out"
if command -v sha256sum > /dev/null 2>&1; then sha() { sha256sum "$@"; }; else sha() { shasum -a 256 "$@"; }; fi

list() {   # list BUILD_DIR -> "<sha256> <path relative to BUILD_DIR>" sorted by path
  (cd "$1" && find . -path '*/kernels/*' -name '*.bin' -type f -print | LC_ALL=C sort | while IFS= read -r f; do
     sha "$f"
   done) | sed 's|  \./| |' | awk '{ print $1, $2 }'
}

[ -d "$base" ] || { echo "G0 no baseline build at $base" >&2; exit 2; }
[ -d "$head" ] || { echo "G0 no build under test at $head" >&2; exit 2; }
: > "$out/differ.txt"; : > "$out/added.txt"; : > "$out/removed.txt"
list "$base" > "$out/baseline.sha"
list "$head" > "$out/under-test.sha"

awk -v out="$out" '
  NR == FNR { b[$2] = $1; nb++; next }
  { nh++
    if ($2 in b) { common++; if (b[$2] == $1) same++; else { print $2 > (out "/differ.txt"); differ++ }; seen[$2] = 1 }
    else { print $2 > (out "/added.txt"); added++ } }
  END {
    for (k in b) if (!(k in seen)) { print k > (out "/removed.txt"); removed++ }
    printf "G0 kernels: %d in the baseline, %d under test; common %d, identical %d, differ %d, added %d, removed %d\n",
           nb, nh, common, same, differ, added, removed
  }' "$out/baseline.sha" "$out/under-test.sha" | tee "$out/summary.txt"
for f in differ added removed; do
  [ -f "$out/$f.txt" ] || : > "$out/$f.txt"
  LC_ALL=C sort -o "$out/$f.txt" "$out/$f.txt"
done
n_differ=$(wc -l < "$out/differ.txt" | tr -d ' ')
n_removed=$(wc -l < "$out/removed.txt" | tr -d ' ')
n_common=$(awk '{ print $2 }' "$out/baseline.sha" | LC_ALL=C sort | LC_ALL=C comm -12 - <(awk '{ print $2 }' "$out/under-test.sha" | LC_ALL=C sort) | wc -l | tr -d ' ')
if [ "$n_differ" -gt 0 ]; then
  echo "G0 FAIL: $n_differ binaries present in both builds differ:"; head -50 "$out/differ.txt"
fi
if [ "$n_removed" -gt 0 ]; then
  echo "G0 the baseline has $n_removed binaries the tree under test does not build:"; head -50 "$out/removed.txt"
fi
if [ -s "$out/added.txt" ]; then
  echo "G0 added (only under test): $(wc -l < "$out/added.txt" | tr -d ' ')"; sed 's/^/  + /' "$out/added.txt" | head -200
fi
[ "$n_common" -gt 0 ] || { echo "G0 FAIL: no kernel binary in common - wrong build directories?"; exit 2; }
[ "$n_differ" -eq 0 ] || exit 1
if [ "$n_removed" -gt 0 ] && [ "${G0_ALLOW_REMOVED:-0}" != 1 ]; then
  echo "G0 FAIL: removed binaries (G0_ALLOW_REMOVED=1 accepts them)"; exit 1
fi
echo "G0 PASS: every kernel binary present in both builds is byte-identical"
exit 0
