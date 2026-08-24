#!/usr/bin/env bash
# Run an oracle command inside the reference container, ON THE BOX.
#   tools/oracle/run_in_container.sh '<shell command using "$SNAP">'
# $SNAP is the resolved checkpoint snapshot directory under the read-only HF
# cache mount. Runs as the calling uid/gid so outputs are not root-owned (a
# root-owned file breaks `tools/box.sh pull`), with HOME redirected to a
# writable scratch because a non-root container has no home. No memory limit:
# the dequantised bf16 model needs ~55 GB (docs/03).
# Env: ORACLE_IMAGE, HF_CACHE, REPO_DIR, ORACLE_MODEL.
set -euo pipefail
IMAGE="${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t214-vxkp0:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="${REPO_DIR:-$HOME/b70-inference-server}"
ORACLE_MODEL="${ORACLE_MODEL:-models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ}"
[ $# -ge 1 ] || { echo "usage: $0 '<command using \$SNAP>'" >&2; exit 2; }

exec docker run --rm --entrypoint bash -u "$(id -u):$(id -g)" \
  -v "$REPO_DIR:/ws" -w /ws -v "$HF_CACHE:/hf:ro" -v /tmp:/scratch \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/scratch/hf -e HOME=/scratch \
  -e PYTHONUNBUFFERED=1 "$IMAGE" -c \
  "set -euo pipefail; SNAP=\$(ls -d /hf/hub/$ORACLE_MODEL/snapshots/*/ | head -1); $*"
