#!/usr/bin/env bash
# Run an oracle command inside the reference container, ON THE BOX.
#   tools/oracle/run_in_container.sh '<shell command using "$SNAP">'
# $SNAP is the resolved checkpoint snapshot directory. Two ways to name one:
#   * ORACLE_MODEL (default) - a repo under the read-only HF cache mount, whose
#     single `snapshots/*/` is resolved inside the container;
#   * ORACLE_SNAP - an absolute host path to a snapshot directory, mounted
#     read-only at /snap. This is how a self-quantised checkpoint that never
#     went through `hf download` is used (tools/quantize_qwen38_rtn.sh writes
#     one), and it takes precedence when set.
# Runs as the calling uid/gid so outputs are not root-owned (a root-owned file
# breaks `tools/box.sh pull`), with HOME redirected to a writable scratch
# because a non-root container has no home. No memory limit: the dequantised
# bf16 model needs ~55 GB (docs/03).
#
# ORACLE_THREADS caps the CPU pool. It exists because the box is shared: a
# dump that grabs all 44 threads fights whatever else is running, and torch's
# default is "all of them". Leave it unset for the default.
# Env: ORACLE_IMAGE, HF_CACHE, REPO_DIR, ORACLE_MODEL, ORACLE_SNAP, ORACLE_THREADS.
set -euo pipefail
# t214 was the reference until 2026-09-09 and is no longer on the box; every
# golden set dumped before that date was produced by it and its rows still say
# so. A golden set records which image made it (doc 14) - so when this default
# moves again, the sets made under the old one are not retroactively relabelled.
IMAGE="${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="${REPO_DIR:-$HOME/b70-inference-server}"
ORACLE_MODEL="${ORACLE_MODEL:-models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}"
ORACLE_SNAP="${ORACLE_SNAP:-}"
ORACLE_THREADS="${ORACLE_THREADS:-}"
[ $# -ge 1 ] || { echo "usage: $0 '<command using \$SNAP>'" >&2; exit 2; }

mounts=(-v "$REPO_DIR:/ws" -w /ws -v "$HF_CACHE:/hf:ro" -v /tmp:/scratch)
if [ -n "$ORACLE_SNAP" ]; then
  [ -f "$ORACLE_SNAP/config.json" ] || {
    echo "ORACLE_SNAP=$ORACLE_SNAP has no config.json" >&2; exit 2; }
  mounts+=(-v "$ORACLE_SNAP:/snap:ro")
  resolve='SNAP=/snap'
else
  resolve="SNAP=\$(ls -d /hf/hub/$ORACLE_MODEL/snapshots/*/ | head -1)"
fi

env_flags=(-e HF_HUB_OFFLINE=1 -e HF_HOME=/scratch/hf -e HOME=/scratch -e PYTHONUNBUFFERED=1)
if [ -n "$ORACLE_THREADS" ]; then
  # torch takes its intra-op pool from OMP_NUM_THREADS at import; MKL's own
  # knob is separate. dump.py prints torch.get_num_threads() so the cap is
  # visible in the log rather than assumed.
  env_flags+=(-e "OMP_NUM_THREADS=$ORACLE_THREADS" -e "MKL_NUM_THREADS=$ORACLE_THREADS")
fi

exec docker run --rm --entrypoint bash -u "$(id -u):$(id -g)" \
  "${mounts[@]}" "${env_flags[@]}" "$IMAGE" -c \
  "set -euo pipefail; $resolve; $*"
