#!/usr/bin/env bash
# Spec 32's one serving check, from the tree root after the build: the edit / copy set with lookup inside MTP on one
# card, the 16-row arm (B70_SPEC_LOOKUP_ROWS=16) and a wider arm (ROWS, default 64: the ladder up to that list) in the
# same session, rounds A B B A, each server run under its own card lock. Then the ids of every request across the arms
# (the texts byte for byte once the server's random tool-call ids are masked), and the speed table.
#   [ROWS=32|64] tools/probe/deep_verify_serve.sh CARD OUT
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
w="${ROWS:-64}"
arm r16 1 16
arm "r$w" 1 "$w"
arm "r$w" 2 "$w"
arm r16 2 16
echo "== text"
python3 - "$out" <<'EOF'
import json, re, sys, glob
out = sys.argv[1]
mask = lambda t: re.sub(r'call_[0-9a-f]{24}', 'call_X', t or '')
runs = {}
for d in sorted(glob.glob(out + '/r*.r*')):
    if not d.endswith(('.r1', '.r2')):
        continue
    runs[d.split('/')[-1]] = {r['name']: mask(r['text']) for r in map(json.loads, open(d + '/requests.jsonl'))}
ref = runs['r16.r1']
for k, v in runs.items():
    diff = [n for n in ref if v.get(n) != ref[n]]
    print(f"TEXT {k}: {len(ref) - len(diff)} of {len(ref)} requests byte-equal to r16.r1 (tool-call ids masked)" +
          (f"; differ: {diff}" if diff else ""))
ids = {k: [json.load(open(f)).get('out_ids') for f in sorted(glob.glob(f'{out}/{k}/server-log/*.json'))] for k in runs}
for k, v in ids.items():
    same = sum(a == b for a, b in zip(v, ids['r16.r1']))
    print(f"IDS {k}: {same} of {len(ids['r16.r1'])} requests' output ids equal to r16.r1's")
EOF
echo "== table"
python3 tools/box_validate/serve_client.py table --dir "$out" --ref r16 --ratio "r$w/r16"
echo DEEP_VERIFY_SERVE_DONE
