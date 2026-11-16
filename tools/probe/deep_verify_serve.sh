#!/usr/bin/env bash
# Spec 32's one serving check, from the tree root after the build: the edit / copy set with lookup inside MTP on one
# card, the 16-row arm (B70_SPEC_LOOKUP_ROWS=16) and the 32-row arm in the same session, rounds A B B A, each server
# run under its own card lock. Then the text of every request byte for byte across the arms, and the speed table.
#   tools/probe/deep_verify_serve.sh CARD OUT
set -u
card="$1" out="$2"
mkdir -p "$out"
model=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
arm() {   # label, round, rows
  echo "== $1 r$2 $(date '+%F %T')"
  B70_SPEC_LOOKUP_ROWS="$3" PORT=8294 tools/gpu_lock.sh card "$card" -- tools/box_validate/serve_run.sh "$out/$1.r$2" "$1" "$2" \
    "$model" --parallel 1 --max-len 102400 --spec auto -- --set tests/golden/edit-copy --max-tokens 256 --stop-at-eos \
    > "$out/$1.r$2.log" 2>&1
  echo "rc=$? ($1 r$2)"
}
arm r16 1 16
arm r32 1 32
arm r32 2 32
arm r16 2 16
echo "== text"
python3 - "$out" <<'EOF'
import json, sys, glob
out = sys.argv[1]
runs = {}
for d in sorted(glob.glob(out + '/r*.r*')):
    if not d.endswith(('.r1', '.r2')):
        continue
    runs[d.split('/')[-1]] = {r['name']: r['text'] for r in map(json.loads, open(d + '/requests.jsonl'))}
ref = runs['r16.r1']
for k, v in runs.items():
    diff = [n for n in ref if v.get(n) != ref[n]]
    print(f"TEXT {k}: {len(ref) - len(diff)} of {len(ref)} requests byte-equal to r16.r1" + (f"; differ: {diff}" if diff else ""))
EOF
echo "== table"
python3 tools/box_validate/serve_client.py table --dir "$out" --ref r16 --ratio r32/r16
echo DEEP_VERIFY_SERVE_DONE
