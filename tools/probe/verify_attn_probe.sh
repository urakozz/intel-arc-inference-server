#!/usr/bin/env bash
# Spec 28c (plan 28c Tasks 3-4) jobs ON THE BOX, from the repo root, under the GPU lock taken by the caller:
#   tools/probe/detach.sh <log> tools/gpu_lock.sh card 0 -- tools/probe/verify_attn_probe.sh capture [--depths ..]
#   tools/probe/detach.sh <log> tools/gpu_lock.sh card 0 -- tools/probe/verify_attn_probe.sh arms <args...>
# capture: probe_decode_attn capture --rows 8 at max_len 131072 (the q / gate of 8 plain steps per depth and FA
#          layer 15's KV past deepest + 8); arms: probe_verify_attn arms <dump> <args...>.
# ZE_AFFINITY_MASK is the lock's (card N), 0 when unset. DUMP defaults to ~/b70/logs/spec28c/vattn-dump.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d}"
DUMP="${DUMP:-$HOME/b70/logs/spec28c/vattn-dump}"
BUILD="${BUILD:-build}"
echo "uptime: $(uptime)"
case "${1:-}" in
  capture) shift
           mkdir -p "$DUMP"
           "$BUILD"/tools/probe/probe_decode_attn capture "$SNAP" tests/golden/prompts/long32k.ids \
             "$DUMP" --rows 8 --depths 512,4096,32768,130816 "$@" 2>&1 | grep -v '^loader\|^  '
           (cd "$DUMP" && sha256sum kv_k.bin kv_v.bin q_*.bin g_*.bin) ;;
  arms)    shift
           "$BUILD"/tools/probe/probe_verify_attn arms "$DUMP" "$@" ;;
  *)       echo "usage: $0 capture|arms ..." >&2; exit 2 ;;
esac
echo "uptime: $(uptime)"
