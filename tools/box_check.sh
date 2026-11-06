#!/bin/bash
# tools/box_check.sh - the merge check, run from a checkout on the Mac against the box: the
# real build, the kernel command lines against a base, and the suite that needs no model files.
# It replaces tools/mac_check.sh (retired 2026-11-03 by the operator: the box is the check).
#
#   tools/box_check.sh [--base REF] [--allow-kernel-changes] [--card N] [--no-tests]
#
#   1 build     tools/box.sh build: sync this checkout to its remote tree (tools/box_dir.sh),
#               cmake configure, full build (-j$JOBS, default 44).
#   2 cmdlines  tools/kernel_cmdlines, on the box, on the remote tree and on REF's src/kernels:
#               an existing variant's ocloc command line must not change or disappear
#               (additions are listed, and fine). G0 (tools/box_validate.sh) compares the
#               binaries themselves; this is the cheap pre-check.
#   3 tests     ctest -LE checkpoint in the remote tree under tools/gpu_lock.sh card N (the
#               first free card, else card 1, waiting for it), detached on the box and polled.
#               Passed, Failed and Skipped are counted apart: a Skipped test is not a pass.
#               Every test runs traced (tools/test_launcher.sh) and its PASS is recorded with
#               what it read (tools/box_validate/reuse.py): the validation run of the same
#               sources takes those PASSes instead of running the tests again.
#
#   --base REF              what section 2 compares against (main)
#   --allow-kernel-changes  report changed / removed command lines without failing on them
#   --card N                section 3's card
#   --no-tests              sections 1 and 2 only
#
# Logs on the box under ~/b70/logs/box-check/<tree>. BOX as for tools/box.sh (tools/box.env).
# Exit status 0 when no section FAILs.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
base=main allow=0 card="" tests=1
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base=${2:?--base needs a ref}; shift 2 ;;
    --base=*) base=${1#--base=}; shift ;;
    --allow-kernel-changes) allow=1; shift ;;
    --card) card=${2:?--card needs N}; shift 2 ;;
    --no-tests) tests=0; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "box_check: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done

# shellcheck source=/dev/null
# A worktree has no tools/box.env of its own (untracked): fall back to the main checkout's.
for f in "$root/tools/box.env" "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/../tools/box.env"; do
  [ -z "${BOX:-}" ] && [ -r "$f" ] && . "$f"
done
: "${BOX:?set BOX=user@host in the environment or in tools/box.env}"
export BOX
# shellcheck source=tools/box_dir.sh
. "$root/tools/box_dir.sh"
tree=$(b70_remote_dir "$root")
logs="b70/logs/box-check/$(basename "$tree")"
if ! git -C "$root" rev-parse --verify --quiet "$base^{commit}" > /dev/null; then
  echo "box_check: --base $base is not a commit here" >&2
  exit 2
fi
# Hide the box's address in everything printed.
mask() { sed -E 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/<ip>/g'; }
rbox() { ssh "$BOX" "$@" 2>&1 | mask; return "${PIPESTATUS[0]}"; }

summary=""
failed=0
# row <section> <PASS|FAIL|SKIP> <counts> <detail>
row() {
  summary="$summary$(printf '%-9s %-5s %-34s %s' "$1" "$2" "$3" "$4")
"
  [ "$2" = "FAIL" ] && failed=1
  printf '\n== %s: %s  %s  %s\n' "$1" "$2" "$3" "$4"
}
now() { date +%s; }
rbox "mkdir -p ~/$logs" > /dev/null

# ---- 1. build --------------------------------------------------------------------------
t0=$(now)
build_ok=0
mkdir -p "$root/build"
if "$root/tools/box.sh" build > >(mask > "$root/build/box-check.build.log" 2>&1) 2>&1; then
  build_ok=1
  row build PASS "-" "$tree ($(($(now) - t0))s)"
else
  tail -30 "$root/build/box-check.build.log"
  row build FAIL "-" "$tree: see build/box-check.build.log"
fi

# ---- 2. kernel command lines against the base ------------------------------------------
# The base's src/kernels, extracted on the box into a directory per base commit.
bsha=$(git -C "$root" rev-parse --short "$base")
bdir="$logs/base-$bsha"
mkdir -p "$root/build"
kc="$root/build/box-check"
mkdir -p "$kc"
if ! git -C "$root" archive "$base" -- src/kernels | ssh "$BOX" "mkdir -p ~/$bdir && tar -x -C ~/$bdir" 2>&1 | mask; then
  row cmdlines FAIL "-" "git archive $base src/kernels to the box failed"
elif ! rbox "cd ~/$tree && cmake -S tools/kernel_cmdlines -B ~/$logs/kc-head -DTREE=\$PWD > ~/$logs/kc-head.log 2>&1 &&
             cmake -S tools/kernel_cmdlines -B ~/$logs/kc-base -DTREE=\$HOME/$bdir > ~/$logs/kc-base.log 2>&1"; then
  row cmdlines FAIL "-" "tools/kernel_cmdlines failed on the box: ~/$logs/kc-{head,base}.log"
else
  rbox "sort ~/$logs/kc-head/kernels.txt" > "$kc/kernels-head.txt"
  rbox "sort ~/$logs/kc-base/kernels.txt" > "$kc/kernels-base.txt"
  # By variant name: only-in-head = added, only-in-base = removed, in both but a
  # different line = changed (a moved -D, a new option: a different binary).
  awk -F' [|] ' 'NR == FNR { b[$1] = $0; next }
                 { h[$1] = $0 }
                 END {
                   for (n in h) if (!(n in b)) print "added\t" h[n]
                   for (n in b) if (!(n in h)) print "removed\t" b[n]
                   for (n in h) if ((n in b) && h[n] != b[n]) print "changed\t" b[n] "\t->\t" h[n]
                 }' "$kc/kernels-base.txt" "$kc/kernels-head.txt" | sort > "$kc/kernels-diff.txt"
  na=$(grep -c '^added' "$kc/kernels-diff.txt"); nr=$(grep -c '^removed' "$kc/kernels-diff.txt")
  nc=$(grep -c '^changed' "$kc/kernels-diff.txt"); nh=$(grep -c . "$kc/kernels-head.txt")
  counts="$nh variants: +$na, -$nr, ~$nc"
  if [ "$nh" = 0 ]; then
    row cmdlines FAIL "$counts" "no kernel list from this tree"
  elif [ $((nr + nc)) != 0 ] && [ $allow = 0 ]; then
    row cmdlines FAIL "$counts" "existing command lines moved (--allow-kernel-changes if meant)"
  elif [ $((nr + nc)) != 0 ]; then
    row cmdlines PASS "$counts" "changes allowed by --allow-kernel-changes"
  else
    row cmdlines PASS "$counts" "no existing command line changed against $base"
  fi
  for k in added removed changed; do
    grep "^$k" "$kc/kernels-diff.txt" | cut -f2- | sed "s/^/    $k: /" | sed 's/\t/ /g'
  done
fi

# ---- 3. the suite without model files, on a card ----------------------------------------
if [ $tests = 0 ]; then
  row tests SKIP "-" "--no-tests"
elif [ $build_ok = 0 ]; then
  row tests SKIP "-" "the build failed"
else
  if [ -z "$card" ]; then
    st=$(rbox "cd ~/$tree && tools/gpu_lock.sh status")
    if grep -q '^card 0  free' <<< "$st"; then card=0; elif grep -q '^card 1  free' <<< "$st"; then card=1; else card=1; fi
  fi
  t0=$(now)
  log="$logs/ctest.log"
  # Detached on the box: a dropped ssh connection must not kill the run.
  rbox "cd ~/$tree && ctest --test-dir build --show-only=json-v1 > ~/$logs/tests.json && nohup bash -c 'tools/gpu_lock.sh card $card -- env B70_TRACE_DIR=\$HOME/$logs/trace ctest --test-dir build -LE checkpoint --output-on-failure --test-output-size-passed 200000000 --test-output-size-failed 200000000 --output-junit \$HOME/$logs/ctest.junit.xml; echo \"box_check rc=\$?\"' > ~/$log 2>&1 < /dev/null &" > /dev/null
  echo "box_check: ctest -LE checkpoint on card $card, log ~/$log"
  until rbox "grep -q '^box_check rc=' ~/$log" > /dev/null; do sleep 30; done
  rbox "cat ~/$log" > "$kc/ctest.log"
  rbox "cd ~/$tree && python3 tools/box_validate/reuse.py record --tests ~/$logs/tests.json --junit ~/$logs/ctest.junit.xml --trace ~/$logs/trace --since ~/$logs/tests.json --from box-check@$(git -C "$root" rev-parse --short=12 HEAD)" | grep '^REUSE recorded' 
  total=$(sed -nE 's/.* tests failed out of ([0-9]+).*/\1/p' "$kc/ctest.log" | tail -1)
  nf=$(sed -nE 's/.* ([0-9]+) tests failed out of .*/\1/p' "$kc/ctest.log" | tail -1)
  ns=$(grep -cE '^ *[0-9]+ - .*\(Skipped\)' "$kc/ctest.log")
  counts="${total:-?} tests: $((${total:-0} - ${nf:-0} - ns)) passed, ${nf:-?} failed, $ns skipped"
  if [ -z "$total" ]; then
    tail -20 "$kc/ctest.log"
    row tests FAIL "-" "no ctest summary: build/box-check/ctest.log"
  elif [ "$nf" != 0 ]; then
    grep -E '^ *[0-9]+ - .*\((Failed|Timeout|SEGFAULT|Subprocess|Exception|Not Run)' "$kc/ctest.log" | sed 's/^/    /'
    row tests FAIL "$counts" "card $card ($(($(now) - t0))s)"
  else
    row tests PASS "$counts" "card $card ($(($(now) - t0))s)"
  fi
  [ "$ns" != 0 ] && grep -E '^ *[0-9]+ - .*\(Skipped\)' "$kc/ctest.log" | sed 's/^/    skipped: /'
fi

printf '\n%s' "$summary"
echo "Not proved here: the model-file tests (label checkpoint: golden gates, served rows) - tools/box_validate.sh."
exit $failed
