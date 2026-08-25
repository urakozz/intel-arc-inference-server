#!/usr/bin/env bash
# Sync this tree to the box and build / test there. The Mac never compiles.
#   tools/box.sh sync            rsync the tree
#   tools/box.sh build           sync + cmake configure + build
#   tools/box.sh test [regex]    sync + build + ctest (optionally -R regex)
#   tools/box.sh run <cmd...>    run a shell command in the remote tree
#   tools/box.sh pull <path>     copy a generated file back from the box
# Env: BOX (ssh target), REMOTE_DIR (relative to $HOME on the box), JOBS.
set -euo pipefail
BOX="${BOX:-user@box}"
REMOTE_DIR="${REMOTE_DIR:-b70-inference-server}"
JOBS="${JOBS:-44}"
cd "$(dirname "$0")/.."

sync_tree() {
  rsync -az --delete \
    --exclude build --exclude .git --exclude 'cmake-build-*' --exclude .idea \
    --exclude oracle-out \
    ./ "$BOX:$REMOTE_DIR/"
}
configure_and_build() {
  ssh "$BOX" "cd '$REMOTE_DIR' && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j$JOBS"
}

case "${1:-}" in
  sync)  sync_tree ;;
  build) sync_tree; configure_and_build ;;
  test)  sync_tree; configure_and_build
         ssh "$BOX" "cd '$REMOTE_DIR' && ctest --test-dir build --output-on-failure ${2:+-R \"$2\"}" ;;
  run)   shift; ssh "$BOX" "cd '$REMOTE_DIR' && $*" ;;
  pull)  shift; rsync -az "$BOX:$REMOTE_DIR/$1" "$1" ;;
  *)     echo "usage: $0 sync|build|test [regex]|run <cmd...>|pull <path>" >&2; exit 2 ;;
esac
