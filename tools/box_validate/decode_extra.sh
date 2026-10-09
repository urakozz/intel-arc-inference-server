#!/usr/bin/env bash
# b70-decode with extra flags, for a test that spawns the CLI itself with fixed arguments (serve_extra.sh's twin):
#   B70_DECODE_EXTRA='--layers 4' B70_SERVE_ARGS='--layers 4' \
#     build/tests/golden_server_test build/src/cli/b70-serve tools/box_validate/decode_extra.sh ... --chat
# golden_server_test runs `<b70-decode> <snapshot> --ids F --n 32 --prefill --lm-head int8`; this execs the real
# binary of this tree with those arguments and then B70_DECODE_EXTRA (word-split on purpose) - spec 21e's r34.serve:
# Qwen3.8-Flash-Next's synthetic checkpoints run truncated (--layers 4) on both sides.
here="$(cd "$(dirname "$0")/../.." && pwd)" || exit 2
# shellcheck disable=SC2086 # B70_DECODE_EXTRA is a list of flags
exec "$here/build/src/cli/b70-decode" "$@" ${B70_DECODE_EXTRA:-}
