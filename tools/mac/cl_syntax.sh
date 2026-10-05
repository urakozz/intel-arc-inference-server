#!/bin/bash
# tools/mac/cl_syntax.sh - OpenCL C syntax check of every kernel variant's ocloc command
# line, with the Mac's clang instead of ocloc. Part of tools/mac_check.sh (section 4);
# usable alone:
#
#   tools/mac/cl_syntax.sh <kernels.txt> <out-dir> [source-filter-regex]
#
# <kernels.txt> is tools/kernel_cmdlines' list (`name | file.cl | defines | options`).
# Prints `PASS|FAIL <tab> name <tab> detail` per DISTINCT (source, defines) pair - 280
# variants are ~230 distinct compiles - and keeps each failure's output in <out-dir>.
#
# The command line is ocloc's from cmake/ocloc.cmake: -cl-std=CL3.0,
# -cl-fp32-correctly-rounded-divide-sqrt, the variant's -D defines. clang gets
# `-target spir64` because only the SPIR targets enable the cl_intel_subgroups builtins
# and the CL 3.0 optional features (generic address space, subgroups) ocloc has on
# bmg-g31; the rest of the Intel extensions come from intel_shim.h, force-included.
# Intel-only build options (-cl-intel-256-GRF-per-thread) are dropped: register
# allocation is not a syntax question.
#
# What a PASS proves: the variant preprocesses (every #error guard of its defines holds),
# parses and type-checks as OpenCL C 3.0 with these extensions. Not: that ocloc's
# frontend agrees in every corner (it is Intel's clang fork, a few versions apart), that
# it compiles to ISA within the register budget, or anything at run time.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

if [ "${1:-}" = "--one" ]; then
  # One tab-separated record: name, source, defines, options.
  IFS='	' read -r name src defs opts <<EOF_REC
$2
EOF_REC
  path=$(find "$root/src/kernels" -name "$src" -type f | head -1)
  if [ -z "$path" ]; then
    printf 'FAIL\t%s\tsource %s not found under src/kernels\n' "$name" "$src"; exit 0
  fi
  dflags=""
  [ "$defs" = "-" ] && defs=""
  [ "$opts" = "-" ] && opts=""
  for d in $defs; do dflags="$dflags -D$d"; done
  oflags=""
  for o in $opts; do
    case "$o" in -cl-intel-*) ;; *) oflags="$oflags $o" ;; esac
  done
  log="$B70_CLS_OUT/$name.log"
  # shellcheck disable=SC2086  # the flag lists are meant to split
  if clang -x cl -cl-std=CL3.0 -target spir64-unknown-unknown -fsyntax-only \
       -cl-fp32-correctly-rounded-divide-sqrt -Wno-ignored-pragmas \
       -include "$here/opencl/intel_shim.h" -I"$(dirname "$path")" $dflags $oflags \
       "$path" > "$log" 2>&1; then
    rm -f "$log"
    printf 'PASS\t%s\t%s\n' "$name" "$src"
  else
    first=$(grep -m1 -E 'error:' "$log" | sed "s|$root/||")
    printf 'FAIL\t%s\t%s\n' "$name" "$first"
  fi
  exit 0
fi

if [ $# -lt 2 ]; then
  echo "usage: $0 <kernels.txt> <out-dir> [source-filter-regex]" >&2
  exit 2
fi
list=$1 out=$2 filter=${3:-.}
mkdir -p "$out"
export B70_CLS_OUT=$out
jobs=${B70_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}
# One compile per distinct (source, defines, options); the first name stands for it.
awk -F'|' -v f="$filter" '{
    for (i = 1; i <= 4; ++i) { gsub(/^ +| +$/, "", $i); if ($i == "") $i = "-" }
    key = $2 "|" $3 "|" $4
    if ($2 ~ f && !(key in seen)) { seen[key] = 1; printf "%s\t%s\t%s\t%s\n", $1, $2, $3, $4 }
  }' "$list" |
  tr '\n' '\0' | xargs -0 -P "$jobs" -n 1 "$0" --one | sort -k2
