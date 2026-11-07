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
# because a non-root container has no home.
#
# Memory (2026-11-07, after the 2026-11-06 OOM): every run is capped and the
# container gets NO swap (--memory X --memory-swap X), so a CPU job can never
# push the box into swap - at its cap the kernel kills the job inside its own
# container (exit 137) and nothing else on the box notices. On the box (Linux)
# X is ORACLE_MEMORY, default 64g (operator 2026-11-08: 64 GiB beside GPU work): the box has 121 GiB; one GPU job's host side
# was measured at ~52 GiB (2026-11-06 19:40-20:30, sar: pinned prefix cache up
# to 32 GiB + weight staging in the driver), plus ~3 GiB for the OS and ~10 GiB
# of page cache for the GPU job's weight reads. A run that truly needs more says
# so (ORACLE_MEMORY=110g, at most MemTotal - 9 GiB: 4 for the OS + 5 buffer, operator 2026-11-08) and gets it only with the
# cards idle: it takes the whole-box GPU lock (~/b70-gpu.lock, exclusive,
# non-blocking) for its whole run, so no GPU job starts beside it, and it is
# REFUSED (exit 75) while any GPU job holds that lock - never queued, so a
# caller already holding the lock cannot deadlock on itself. The cap counts
# the container's anon memory, its tmpfs writes (/scratch is the box's /tmp,
# a tmpfs) and its page cache; the page cache is reclaimed at the cap, so a
# layer-streamed reference only re-reads from disk. The memory planners
# (oracle_generate.py) read the cap from the container's memory.max.
# The Mac: ORACLE_MEMORY as given (Docker Desktop's VM is shared), no default.
#
# ORACLE_THREADS caps the CPU pool. It exists because the box is shared: a
# dump that grabs all 44 threads fights whatever else is running, and torch's
# default is "all of them". Leave it unset for the default.
# Env: ORACLE_IMAGE, HF_CACHE, REPO_DIR, ORACLE_MODEL, ORACLE_SNAP, ORACLE_THREADS, ORACLE_MEMORY,
#      B70_LOCK_DIR (tools/gpu_lock.sh's lock directory, default $HOME).
set -euo pipefail
# t214 was the reference until 2026-09-09 and is no longer on the box; every
# golden set dumped before that date was produced by it and its rows still say
# so. A golden set records which image made it (doc 14) - so when this default
# moves again, the sets made under the old one are not retroactively relabelled.
IMAGE="${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The tree this script lives in (per-plan trees: tools/box_dir.sh).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
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
# ORACLE_MEMORY (<N>g): the container's memory cap, swap off (above). The box: default CPU_CAP_GB.
CPU_CAP_GB=64
if [ -r /proc/meminfo ]; then   # the box
  ORACLE_MEMORY="${ORACLE_MEMORY:-${CPU_CAP_GB}g}"
fi
if [ -n "${ORACLE_MEMORY:-}" ]; then
  [[ "$ORACLE_MEMORY" =~ ^[0-9]+g$ ]] || { echo "ORACLE_MEMORY is <N>g (got $ORACLE_MEMORY)" >&2; exit 2; }
  env_flags+=(--memory "$ORACLE_MEMORY" --memory-swap "$ORACLE_MEMORY")
fi
if [ -r /proc/meminfo ] && [ "${ORACLE_MEMORY%g}" -gt "$CPU_CAP_GB" ]; then
  want=${ORACLE_MEMORY%g}
  total=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
  if [ "$want" -gt $((total - 9)) ]; then
    echo "run_in_container: REFUSED: ORACLE_MEMORY=${want}g > MemTotal ${total} GiB - 9 GiB (the OS 4 + a 5 GiB buffer)" >&2; exit 2
  fi
  # Above the CPU cap only with the cards idle: the whole-box GPU lock, exclusive, for the whole run
  # (fd 8 stays open in docker's client, which waits for the container: --rm, foreground).
  exec 8> "${B70_LOCK_DIR:-$HOME}/b70-gpu.lock" || exit 2
  if ! flock -n -x 8; then
    echo "run_in_container: REFUSED: ORACLE_MEMORY=${want}g is above the ${CPU_CAP_GB}g CPU-job cap, which needs" \
         "the cards idle, and a GPU job holds ~/b70-gpu.lock; run it when no GPU job runs" >&2
    exit 75
  fi
  echo "run_in_container: ORACLE_MEMORY=${want}g > ${CPU_CAP_GB}g: holding the whole-box GPU lock for this run" >&2
fi
if [ -n "$ORACLE_THREADS" ]; then
  # torch takes its intra-op pool from OMP_NUM_THREADS at import; MKL's own
  # knob is separate. dump.py prints torch.get_num_threads() so the cap is
  # visible in the log rather than assumed.
  env_flags+=(-e "OMP_NUM_THREADS=$ORACLE_THREADS" -e "MKL_NUM_THREADS=$ORACLE_THREADS")
fi

exec docker run --rm --entrypoint bash -u "$(id -u):$(id -g)" \
  "${mounts[@]}" "${env_flags[@]}" "$IMAGE" -c \
  "set -euo pipefail; $resolve; $*"
