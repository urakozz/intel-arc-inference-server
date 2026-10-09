#!/bin/sh
# Spec 21a: transformers 5.19.0 for the qwen4_exp reference, in a separate site directory BESIDE the
# oracle image (agnes-ref-img's 5.15.0 stays what every other oracle uses - the image is never changed).
#   tools/oracle/qwen4exp_env.sh <dir>
# Run INSIDE the container (the wheels must match its Python): pip install --no-deps --target <dir>
# -r tools/oracle/qwen4exp_requirements.txt - no dependency resolution, so the image's torch is never
# replaced - then prints `PYTHONPATH=<dir>`; every qwen4exp command runs with that first on PYTHONPATH.
# Idempotent: skips the install when <dir>/transformers-5.19.0.dist-info exists.
#   Mac: oracle-out-q4exp/site (git-ignored)   Box: $DATA/q4exp-site
set -eu
[ $# -eq 1 ] || { echo "usage: $0 <dir>" >&2; exit 2; }
dir="$1"
here=$(cd "$(dirname "$0")" && pwd)
if [ -d "$dir/transformers-5.19.0.dist-info" ]; then
  echo "qwen4exp_env: $dir has transformers 5.19.0 (kept)" >&2
else
  mkdir -p "$dir"
  python3 -m pip install --no-deps --no-compile --disable-pip-version-check --root-user-action=ignore \
    --target "$dir" -r "$here/qwen4exp_requirements.txt" >&2
fi
echo "PYTHONPATH=$dir"
