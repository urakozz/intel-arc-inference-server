#!/usr/bin/env bash
# Spec 10 P0/P1 jobs ON THE BOX, from the repo root, each one short and under the shared
# GPU lock (other plans use the same device):
#   tools/probe/detach.sh ~/x.log flock ~/b70-gpu.lock tools/probe/decode_attn_probe.sh capture
#   tools/probe/detach.sh ~/x.log flock ~/b70-gpu.lock tools/probe/decode_attn_probe.sh arms <args...>
# Plan: docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ZE_AFFINITY_MASK=0
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d}"
DUMP="${DUMP:-$HOME/spec10a-dump}"
echo "uptime: $(uptime)"
case "$1" in
  capture) shift
           mkdir -p "$DUMP"
           build/tools/probe/probe_decode_attn capture "$SNAP" tests/golden/prompts/long32k.ids \
             "$DUMP" "$@" 2>&1 | grep -v '^loader\|^  ' ;;
  arms)    shift
           build/tools/probe/probe_decode_attn arms "$DUMP" "$@" ;;
  *)       echo "usage: $0 capture|arms ..." >&2; exit 2 ;;
esac
echo "uptime: $(uptime)"
