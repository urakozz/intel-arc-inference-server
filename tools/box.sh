#!/usr/bin/env bash
# Sync this tree to the box and build / test there. The Mac never compiles.
#   tools/box.sh sync            rsync the tree
#   tools/box.sh build           sync + cmake configure + build
#   tools/box.sh test [regex]    sync + build + ctest (optionally -R regex)
#   tools/box.sh run <cmd...>    run a shell command in the remote tree
#   tools/box.sh pull <path>     copy a generated file back from the box
# Env: BOX (ssh target), REMOTE_DIR (relative to $HOME on the box), JOBS,
#      BUILD_DIR (default build; e.g. build-nosycl), CMAKE_ARGS (extra -D flags for
#      the configure step).
set -euo pipefail
BOX="${BOX:-user@box}"
REMOTE_DIR="${REMOTE_DIR:-b70-inference-server}"
JOBS="${JOBS:-44}"
# A second configuration lives beside the first on the box (spec 2.1 §4:
# build-nosycl is `cmake -DB70_PREFILL=OFF`). Both are named here rather than
# hardcoded so one tree serves every configuration.
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_ARGS="${CMAKE_ARGS:-}"
cd "$(dirname "$0")/.."

sync_tree() {
  rsync -az --delete \
    --exclude build --exclude 'build-*' --exclude .git --exclude 'cmake-build-*' --exclude .idea \
    --exclude oracle-out --exclude 'oracle-out-*' \
    ./ "$BOX:$REMOTE_DIR/"
}
configure_and_build() {
  ssh "$BOX" "cd '$REMOTE_DIR' && cmake -S . -B '$BUILD_DIR' -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS > /dev/null && cmake --build '$BUILD_DIR' -j$JOBS"
}

case "${1:-}" in
  sync)  sync_tree ;;
  build) sync_tree; configure_and_build ;;
  test)  sync_tree; configure_and_build
         ssh "$BOX" "cd '$REMOTE_DIR' && ctest --test-dir '$BUILD_DIR' --output-on-failure ${2:+-R \"$2\"}" ;;
  run)   shift; ssh "$BOX" "cd '$REMOTE_DIR' && $*" ;;
  pull)  shift; rsync -az "$BOX:$REMOTE_DIR/$1" "$1" ;;
  *)     echo "usage: $0 sync|build|test [regex]|run <cmd...>|pull <path>" >&2; exit 2 ;;
esac
