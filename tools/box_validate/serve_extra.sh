#!/usr/bin/env bash
# b70-serve with extra flags, for a test that spawns the server itself with fixed arguments:
#   B70_SERVE_EXTRA='--spec lookup --spec-min-match 2' \
#     build/tests/golden_server_test tools/box_validate/serve_extra.sh build/src/cli/b70-decode ...
# golden_server_test runs `<b70-serve> <snapshot> --host H --port P`; this execs the real
# binary of this tree with those arguments and then B70_SERVE_EXTRA (word-split on purpose).
here="$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
# shellcheck disable=SC2086 # B70_SERVE_EXTRA is a list of flags
exec "$here/build/src/cli/b70-serve" "$@" ${B70_SERVE_EXTRA:-}
