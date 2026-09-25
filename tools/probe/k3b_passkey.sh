#!/bin/bash
# Spec 6 K3b and passkey, run ON THE BOX from the repo root (detach it: ~45 min).
#   tools/probe/k3b_passkey.sh <131000-id prompt>
# 128k determinism and replay (flash_long_test --128k), then tools/probe/passkey.sh on
# l0-int8 and l0.
cd "$(dirname "$0")/../.."
ZE_AFFINITY_MASK=0 ./build/tests/flash_long_test urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --128k "$1"
echo "K3b rc=$?"
tools/probe/passkey.sh l0-int8 l0
echo "passkey rc=$?"
