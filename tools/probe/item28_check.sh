#!/usr/bin/env bash
# Open-work 28's one check, from the tree root after the build: the judged tests the change touches, then the
# edit / copy + tool-call + golden sets and the opencode-session-int8 replay with long lookup runs (arm "wide",
# --spec auto, one card), each GPU command under its own card lock. Item 27's logged arms (lk.r1, lk.r2) are copied
# beside them for the tables - no old-build rerun. Run detached.
#   tools/probe/item28_check.sh CARD OUT
set -u
card="$1" out="$2"
old="$HOME/b70/logs/mtp-lookup/ab1"
mkdir -p "$out/replay" "$out/sets"
echo "== judged tests $(date '+%F %T')"
tools/gpu_lock.sh card "$card" -- ctest --test-dir build --output-on-failure \
  -R '^(mtp_lookup_gpu_test|dflash_gpu_test|mtp_verify_test)$' > "$out/judged.log" 2>&1
echo "rc=$? (judged)"
echo "== sets $(date '+%F %T')"
PORT=8292 tools/gpu_lock.sh card "$card" -- tools/box_validate/serve_run.sh "$out/sets/wide.r1" wide 1 \
  urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --parallel 1 --max-len 102400 --spec auto -- \
  --set tests/golden/edit-copy --set toolcall --set golden --max-tokens 256 --stop-at-eos > "$out/sets.log" 2>&1
echo "rc=$? (sets)"
echo "== replay $(date '+%F %T')"
tools/gpu_lock.sh card "$card" -- python3 "$HOME/b70/logs/mtp-lookup/arm_replay.py" \
  "$HOME/b70/logs/opencode-session-int8" "$out/replay/wide.r1" wide 1 8293 > "$out/replay.log" 2>&1
echo "rc=$? (replay)"
cp -r "$old/sets/lk.r1" "$out/sets/"
cp -r "$old/replay/lk.r1" "$old/replay/lk.r2" "$out/replay/"
for k in replay sets; do
  echo "== table $k"
  python3 tools/box_validate/serve_client.py table --dir "$out/$k" --ref lk --ratio wide/lk
done
echo ITEM28_CHECK_DONE
