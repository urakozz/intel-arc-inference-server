#!/bin/bash
# tools/mac/l0_syntax.sh - compile C++ sources with -fsyntax-only against the Level Zero
# HEADERS on a machine that has no Level Zero runtime (the Mac). Part of
# tools/mac_check.sh (section 2); usable alone:
#
#   L0_INCLUDE=~/level-zero/include tools/mac/l0_syntax.sh <out-dir> <file.cc>...
#
# Prints one line per file, `PASS|FAIL|SKIP <tab> file <tab> detail`, and keeps each
# failure's compiler output in <out-dir>/<file with / as __>.log.
#
# The flags are the box build's (C++17, -Wall -Wextra -Werror, src/ and tests/ on the
# include path), so a blind-written unused variable fails here as it would under g++
# there. What this proves is that the code PARSES and TYPE-CHECKS against ze_api.h -
# not that it links, and nothing about what it does on a device.
#
# SYCL / cutlass / oneDNN sources are SKIPPED, not failed: their headers exist only in
# the oneAPI install on the box. A file is recognised by its place (src/sycl, or a
# tools/probe/<name>.cc that src/sycl declares with b70_sycl_probe(<name>)), by a
# direct `#include <sycl/...>`-style line, or - for one that pulls SYCL in through a
# project header - by the compiler's own "'sycl/...' file not found".
set -u

if [ "${1:-}" = "--one" ]; then
  f=$2
  log="$B70_L0S_OUT/$(echo "$f" | sed 's|/|__|g').log"
  case "$f" in
    src/sycl/*) printf 'SKIP\t%s\tSYCL component (icpx only)\n' "$f"; exit 0 ;;
    tools/probe/*.cc)
      # The icpx probes are declared by src/sycl's b70_sycl_probe(<name>), and some
      # reach SYCL only through an external header (sycl-tla, vLLM's xe2 kernels).
      n=$(basename "$f" .cc)
      if grep -qE "b70_sycl_probe\\($n[ )]" "$B70_L0S_ROOT/src/sycl/CMakeLists.txt"; then
        printf 'SKIP\t%s\tSYCL probe (src/sycl/CMakeLists.txt, icpx only)\n' "$f"; exit 0
      fi ;;
  esac
  if grep -qE '^[[:space:]]*#[[:space:]]*include[[:space:]]*<(sycl|cutlass|oneapi)/' "$B70_L0S_ROOT/$f"; then
    printf 'SKIP\t%s\tincludes SYCL/cutlass/oneDNN headers (icpx only)\n' "$f"; exit 0
  fi
  # The project's two compile-time paths: kernels/kernels.h insists on B70_KERNEL_DIR,
  # and b70_tokenizer is built with its fallback template path. The values do not
  # matter to a syntax check.
  if clang++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror \
       -I"$B70_L0S_ROOT/src" -I"$B70_L0S_ROOT/tests" -I"$B70_L0S_ROOT/tests/kernels" -I"$B70_L0S_ROOT/tests/golden" \
       -I"$B70_L0S_ROOT/tools/probe" \
       -isystem "$B70_L0S_ROOT/third_party/json" -isystem "$B70_L0S_ROOT/third_party" \
       -isystem "$B70_L0S_ROOT/third_party/minja" -isystem "$B70_L0S_ROOT/third_party/httplib" \
       -isystem "$B70_L0S_SHIM" \
       -DB70_KERNEL_DIR='"/b70-kernels"' -DB70_CHAT_TEMPLATE_FALLBACK_PATH='"/b70-fallback.jinja"' \
       "$B70_L0S_ROOT/$f" > "$log" 2>&1; then
    rm -f "$log"
    printf 'PASS\t%s\t\n' "$f"
  elif grep -qE "'(sycl|cutlass|oneapi)/[^']*' file not found" "$log"; then
    printf 'SKIP\t%s\tincludes SYCL/cutlass/oneDNN through a project header\n' "$f"
  else
    first=$(grep -m1 -E 'error:' "$log" | sed "s|$B70_L0S_ROOT/||")
    printf 'FAIL\t%s\t%s\n' "$f" "$first"
  fi
  exit 0
fi

if [ $# -lt 2 ]; then
  echo "usage: L0_INCLUDE=<level-zero/include> $0 <out-dir> <file.cc>..." >&2
  exit 2
fi
out=$1; shift
root=$(cd "$(dirname "$0")/../.." && pwd)
l0=${L0_INCLUDE:-$HOME/PycharmProjects/level-zero/include}
if [ ! -f "$l0/ze_api.h" ]; then
  echo "l0_syntax: no ze_api.h under L0_INCLUDE=$l0" >&2
  exit 2
fi
mkdir -p "$out/l0-include"
# The checkout keeps ze_api.h at its top level; the code spells <level_zero/ze_api.h>.
ln -sfn "$l0" "$out/l0-include/level_zero"
export B70_L0S_ROOT=$root B70_L0S_OUT=$out B70_L0S_SHIM=$out/l0-include
jobs=${B70_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}
printf '%s\n' "$@" | xargs -P "$jobs" -n 1 "$0" --one | sort -k2
