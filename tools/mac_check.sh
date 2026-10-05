#!/bin/bash
# tools/mac_check.sh - everything a Mac can check about this tree, in one command, without
# the box: no Level Zero runtime, no ocloc, no icpx, no ssh. docs/18-mac-checks.md has
# the full account of what each section proves and what it does not.
#
#   tools/mac_check.sh [--base REF] [--quick] [--allow-kernel-changes] [--kernels]
#
#   1 host     the host tests: cmake --preset mac-host (B70_HOST_ONLY, cmake/host_only.cmake),
#              build, ctest. Device tests and tests whose data lives on the box are
#              DISABLED by the configuration, with reasons in build/mac-host/host_only_skipped.txt.
#   2 l0       every C++ source (src/, tests/, tools/probe/) compiled -fsyntax-only with the
#              box's warning flags against the Level Zero headers (tools/mac/l0_syntax.sh).
#              SYCL / cutlass / oneDNN sources are skipped: their headers are on the box only.
#   3 cmdlines tools/kernel_cmdlines on this tree and on REF: an existing variant's ocloc
#              command line must not change or disappear (additions are listed, and fine).
#   4 opencl   every distinct kernel variant command line through clang -x cl -cl-std=CL3.0
#              -fsyntax-only with the Intel-extension shim (tools/mac/cl_syntax.sh).
#   5 kernels  (--kernels only) gemv_i8w, argmax and pf_moe built and run on the Mac's OpenCL GPU
#              against host references (tools/mac/clrun): INDICATIVE, not a B70 result.
#
#   --base REF              what section 3 compares against, and --quick's "changed" (main)
#   --quick                 sections 2 and 4 check only what changed against REF (committed
#                           or not); sections 1 and 3 always run in full - they are cheap
#                           once built, and a partial cmdline diff would prove nothing
#   --allow-kernel-changes  report changed / removed command lines without failing on them
#   --kernels               also run section 5
#
# Environment: L0_INCLUDE (Level Zero headers - a level-zero checkout's include/; default
# ~/PycharmProjects/level-zero/include), B70_JOBS (parallelism; default all cores),
# B70_MAC_CL_DEVICE (section 5's device, a name substring).
#
# Exit status 0 when no section FAILs. A SKIP is not a pass; the table says why.
set -u

root=$(cd "$(dirname "$0")/.." && pwd)
base=main quick=0 allow=0 kernels=0
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base=${2:?--base needs a ref}; shift 2 ;;
    --base=*) base=${1#--base=}; shift ;;
    --quick) quick=1; shift ;;
    --allow-kernel-changes) allow=1; shift ;;
    --kernels) kernels=1; shift ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "mac_check: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done

export L0_INCLUDE=${L0_INCLUDE:-$HOME/PycharmProjects/level-zero/include}
jobs=${B70_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)}
export B70_JOBS=$jobs
out=$root/build/mac-check
mkdir -p "$out"
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

if ! git -C "$root" rev-parse --verify --quiet "$base^{commit}" > /dev/null; then
  echo "mac_check: --base $base is not a commit here" >&2
  exit 2
fi
have_l0=1
[ -f "$L0_INCLUDE/ze_api.h" ] || have_l0=0

# Files that differ from the base: committed on this branch, staged, unstaged, untracked.
changed_files() {
  { git -C "$root" diff --name-only "$base" -- ; git -C "$root" ls-files --others --exclude-standard; } |
    sort -u
}

echo "mac_check: tree $root, base $base ($(git -C "$root" rev-parse --short "$base")), $jobs jobs$([ $quick = 1 ] && echo ', --quick')"

# ---- 1. host tests ------------------------------------------------------------------
t0=$(now)
if [ $have_l0 = 0 ]; then
  row host FAIL "-" "no ze_api.h under L0_INCLUDE=$L0_INCLUDE (the host build compiles the loader against it)"
elif ! (cd "$root" && cmake --preset mac-host > "$out/host-configure.log" 2>&1); then
  row host FAIL "configure" "see $out/host-configure.log"
elif ! (cd "$root" && cmake --build --preset mac-host -j "$jobs" > "$out/host-build.log" 2>&1); then
  row host FAIL "build" "$(grep -m1 -E 'error:|Error ' "$out/host-build.log" | sed "s|$root/||" | cut -c1-110) - $out/host-build.log"
else
  (cd "$root" && ctest --preset mac-host -j "$jobs" > "$out/host-ctest.log" 2>&1)
  total=$(grep -oE 'out of [0-9]+' "$out/host-ctest.log" | tail -1 | grep -oE '[0-9]+')
  nfail=$(grep -oE '[0-9]+ tests? failed' "$out/host-ctest.log" | tail -1 | grep -oE '^[0-9]+')
  total=${total:-0} nfail=${nfail:-0}
  skipped=$root/build/mac-host/host_only_skipped.txt
  ndis=$(grep -c . "$skipped" 2>/dev/null || echo 0)
  ndata=$(grep -c ' | needs ' "$skipped" 2>/dev/null || echo 0)
  counts="$((total - nfail)) pass, $nfail fail, $ndis disabled"
  if [ "$nfail" != 0 ] || [ "$total" = 0 ]; then
    row host FAIL "$counts" "$(sed -n '/tests FAILED:/,$p' "$out/host-ctest.log" | grep -E '^[[:space:]]+[0-9]+ - ' | head -5 | awk '{print $3}' | tr '\n' ' ')- $out/host-ctest.log"
  else
    row host PASS "$counts" "$((ndis - ndata)) need a device, $ndata box-only data ($(now | awk -v t=$t0 '{print $1-t}')s)"
  fi
  if [ "$ndata" != 0 ]; then
    grep ' | needs ' "$skipped" | sed 's/^/    skipped (data): /'
  fi
fi

# ---- 2. Level Zero syntax check -----------------------------------------------------
t0=$(now)
all_cc=$(git -C "$root" ls-files --cached --others --exclude-standard -- '*.cc' | grep -v '^third_party/' | sort -u)
if [ $quick = 1 ]; then
  ch=$(changed_files)
  sel=$(echo "$ch" | grep '\.cc$')
  # A changed header re-checks every source that includes it (by file name).
  for h in $(echo "$ch" | grep -E '\.(h|hpp)$'); do
    hb=$(basename "$h")
    sel="$sel
$(cd "$root" && echo "$all_cc" | xargs grep -l "#include.*[\"/]$hb\"" 2>/dev/null)"
  done
  # Only files that exist (a deleted file is in the diff too).
  l0_files=$(echo "$sel" | sort -u | while read -r f; do [ -n "$f" ] && [ -f "$root/$f" ] && echo "$f"; done)
else
  l0_files=$all_cc
fi
if [ $have_l0 = 0 ]; then
  row l0 FAIL "-" "no ze_api.h under L0_INCLUDE=$L0_INCLUDE"
elif [ -z "$l0_files" ]; then
  row l0 SKIP "0 files" "--quick: no C++ source or header changed against $base"
else
  # shellcheck disable=SC2086
  (cd "$root" && "$root/tools/mac/l0_syntax.sh" "$out/l0" $l0_files) > "$out/l0.txt"
  np=$(grep -c '^PASS' "$out/l0.txt"); nf=$(grep -c '^FAIL' "$out/l0.txt"); ns=$(grep -c '^SKIP' "$out/l0.txt")
  counts="$np pass, $nf fail, $ns skip"
  if [ "$nf" != 0 ]; then
    row l0 FAIL "$counts" "logs in $out/l0/"
    grep '^FAIL' "$out/l0.txt" | cut -f2,3 | sed 's/^/    /' | cut -c1-200
  else
    row l0 PASS "$counts" "skipped = SYCL/cutlass/oneDNN, icpx only ($(now | awk -v t=$t0 '{print $1-t}')s)"
  fi
  [ "$ns" != 0 ] && grep '^SKIP' "$out/l0.txt" | cut -f2,3 | sed 's/^/    skipped: /'
fi

# ---- 3. kernel command lines against the base ----------------------------------------
t0=$(now)
# kc_list <tree> <head|base>: tools/kernel_cmdlines rewrites kernels.txt from empty on
# every configure, so one build directory per tag is reused, never deleted.
kc_list() {
  cmake -S "$root/tools/kernel_cmdlines" -B "$out/kc-$2" -DTREE="$1" > "$out/kc-$2.log" 2>&1 &&
    sort "$out/kc-$2/kernels.txt" > "$out/kernels-$2.txt"
}
# The base's src/kernels, extracted into a directory per base commit: nothing is deleted
# between runs, and a tree from another base is simply a different directory.
base_tree=$out/base-$(git -C "$root" rev-parse --short "$base")
mkdir -p "$base_tree"
if ! git -C "$root" archive "$base" -- src/kernels | tar -x -C "$base_tree"; then
  row cmdlines FAIL "-" "git archive $base src/kernels failed"
elif ! kc_list "$root" head; then
  row cmdlines FAIL "-" "tools/kernel_cmdlines failed on this tree: $out/kc-head.log"
elif ! kc_list "$base_tree" base; then
  row cmdlines FAIL "-" "tools/kernel_cmdlines failed on $base: $out/kc-base.log"
else
  # By variant name: only-in-head = added, only-in-base = removed, in both but a
  # different line = changed (a moved -D, a new option: a different binary).
  awk -F' [|] ' 'NR == FNR { b[$1] = $0; next }
                 { h[$1] = $0 }
                 END {
                   for (n in h) if (!(n in b)) print "added\t" h[n]
                   for (n in b) if (!(n in h)) print "removed\t" b[n]
                   for (n in h) if ((n in b) && h[n] != b[n]) print "changed\t" b[n] "\t->\t" h[n]
                 }' "$out/kernels-base.txt" "$out/kernels-head.txt" | sort > "$out/kernels-diff.txt"
  na=$(grep -c '^added' "$out/kernels-diff.txt"); nr=$(grep -c '^removed' "$out/kernels-diff.txt")
  nc=$(grep -c '^changed' "$out/kernels-diff.txt"); nh=$(grep -c . "$out/kernels-head.txt")
  counts="$nh variants: +$na, -$nr, ~$nc"
  if [ $((nr + nc)) != 0 ] && [ $allow = 0 ]; then
    row cmdlines FAIL "$counts" "existing command lines moved (--allow-kernel-changes if meant)"
  elif [ $((nr + nc)) != 0 ]; then
    row cmdlines PASS "$counts" "changes allowed by --allow-kernel-changes"
  else
    row cmdlines PASS "$counts" "no existing command line changed against $base"
  fi
  for k in added removed changed; do
    grep "^$k" "$out/kernels-diff.txt" | cut -f2- | sed "s/^/    $k: /" | sed 's/\t/ /g'
  done
fi

# ---- 4. OpenCL C syntax of every variant --------------------------------------------
t0=$(now)
if [ ! -s "$out/kernels-head.txt" ]; then
  row opencl FAIL "-" "no kernel list (section 3 could not run tools/kernel_cmdlines)"
else
  filter=.
  if [ $quick = 1 ]; then
    ch=$(changed_files | grep '^src/kernels/')
    cls=$(echo "$ch" | grep '\.cl$' | xargs -n1 basename 2>/dev/null | sed 's/\./\\./g' | tr '\n' '|' | sed 's/|$//')
    if echo "$ch" | grep -qvE '\.cl$|^$'; then
      filter=.   # a changed CMakeLists.txt or header under src/kernels: every variant
    elif [ -n "$cls" ]; then
      filter="^($cls)\$"
    else
      filter=""
    fi
  fi
  if [ -z "$filter" ]; then
    row opencl SKIP "0 variants" "--quick: nothing under src/kernels changed against $base"
  else
    "$root/tools/mac/cl_syntax.sh" "$out/kernels-head.txt" "$out/cl" "$filter" > "$out/cl.txt"
    np=$(grep -c '^PASS' "$out/cl.txt"); nf=$(grep -c '^FAIL' "$out/cl.txt")
    counts="$np pass, $nf fail"
    if [ "$nf" != 0 ]; then
      row opencl FAIL "$counts" "logs in $out/cl/"
      grep '^FAIL' "$out/cl.txt" | cut -f2,3 | sed 's/^/    /' | cut -c1-200 | head -20
    else
      row opencl PASS "$counts" "distinct cmdlines, clang CL3.0 spir64 + intel_shim.h ($(now | awk -v t=$t0 '{print $1-t}')s)"
    fi
  fi
fi

# ---- 5. indicative runs on the Mac's OpenCL GPU -------------------------------------
if [ $kernels = 1 ]; then
  bin=$out/clrun
  mkdir -p "$bin"
  ok=1 runs="" nfail=0 npass=0
  for d in gemv_i8w argmax pf_moe; do
    if ! clang++ -std=c++17 -O2 -Wall -Wextra -Werror -DB70_CLRUN_ROOT="\"$root\"" -I"$root/src" -I"$root/tests" \
         "$root/tools/mac/clrun/clrun.cc" "$root/tools/mac/clrun/${d}_run.cc" \
         -framework OpenCL -o "$bin/${d}_run" > "$out/clrun-$d-build.log" 2>&1; then
      ok=0; runs="$runs $d:build-failed"; continue
    fi
    if "$bin/${d}_run" > "$out/clrun-$d.log" 2>&1; then
      npass=$((npass + 1)); runs="$runs $d:agrees"
    else
      nfail=$((nfail + 1)); runs="$runs $d:DISAGREES"
    fi
    sed 's/^/    /' "$out/clrun-$d.log"
  done
  if [ $ok = 0 ] || [ $nfail != 0 ]; then
    row kernels FAIL "$npass agree, $nfail disagree" "${runs# } (logs in $out)"
  else
    row kernels PASS "$npass agree, $nfail disagree" "INDICATIVE: Mac OpenCL, emulated subgroups"
  fi
else
  row kernels SKIP "-" "not requested (--kernels)"
fi

echo
echo "== summary ($(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet HEAD -- 2>/dev/null || echo '+dirty') vs $base)"
printf '%-9s %-5s %-34s %s\n' section state counts detail
printf '%s' "$summary"
echo "Not proved here: anything on a B70 - timing, ocloc's ISA build, DPAS / 2D block I/O / cross-lane behaviour, the golden gate. docs/18-mac-checks.md."
exit $failed
