#!/usr/bin/env bash
# On the box, detached: bounded 4-layer/8-token teacher-forced decode/reference contract audit.
set -euo pipefail
cd "$(dirname "$0")/../.."
logs="$HOME/b70/logs/q4exp-f32/diagnose/${1:-initial}"
mkdir -p "$logs"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$logs/configure.log" 2>&1
cmake --build build --target qwen4exp_decode_ref_probe -j16 > "$logs/build.log" 2>&1
model="$HOME/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound"
snap="$model/snapshots/$(cat "$model/refs/main")"
ple=$(cd "$snap-ple-int8" && pwd -P)
sha256sum tools/oracle/qwen4exp_decode_diagnose.py "$snap/config.json" "$ple/model.safetensors.index.json"
stat -Lc '%d:%i %s %n' "$snap-ple-int8/model.safetensors.index.json" "$ple/model.safetensors.index.json"
out="$PWD/oracle-out-q4exp-decode-diagnose"
if [ "${2:-}" != reference-only ]; then
  tools/gpu_lock.sh card 1 -- build/tests/qwen4exp_decode_ref_probe "$snap" tests/golden/prompts/q4exp_4k.ids "$out" 4 8 > "$logs/engine.log" 2>&1
fi
exec 9>/tmp/b70-cpu-model.lock
flock -x 9
arms=("${@:3}")
if [ "${#arms[@]}" -eq 0 ]; then arms=(fp32 rounded bf16); fi
for arm in "${arms[@]}"; do
  ORACLE_SNAP="$snap" ORACLE_PLE="$ple" ORACLE_THREADS=16 ORACLE_MEMORY=64g tools/oracle/run_in_container.sh \
    "export PYTHONPATH=/ws/oracle-out-q4exp-site; python tools/oracle/qwen4exp_decode_diagnose.py \"\$SNAP\" int8:/ple /ws/oracle-out-q4exp-decode-diagnose $arm" \
    > "$logs/$arm.log" 2>&1
done
