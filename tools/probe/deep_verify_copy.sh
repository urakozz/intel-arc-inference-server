#!/usr/bin/env bash
# Spec 32e: the pure-copy speed of the lookup verify widths, from the tree root after the build - the four copy prompts
# of a copy-bench directory (copy-code, copy-prose, edit-rename, edit-docstring; ~2k-key prompts, 512 tokens, 2 reps
# after a warm-up) against one server per arm (B70_SPEC_LOOKUP_ROWS 16 / 64), each arm under its own card lock.
#   tools/probe/deep_verify_copy.sh CARD OUT BENCH_DIR   (BENCH_DIR holds cb.py, cbw.py and the prompts' files)
set -u
card="$1" out="$2" bench="$3"
mkdir -p "$out"
model=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
port=8296
arm() {   # label, rows
  echo "== $1 $(date '+%F %T')"
  B70_SPEC_LOOKUP_ROWS="$2" tools/gpu_lock.sh card "$card" -- bash -c '
    out="$1" label="$2" model="$3" port="$4" bench="$5"
    build/src/cli/b70-serve "$model" --port "$port" --parallel 1 --max-len 102400 --spec auto > "$out/$label.log" 2>&1 &
    srv=$!
    for _ in $(seq 1 900); do curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1 && break; sleep 1; done
    url="http://127.0.0.1:$port/v1"
    (cd "$bench" && python3 cbw.py "$url" "$model" . "$out/$label-warm.json" --reps 1 --max-tokens 256 --only copy-code \
       > "$out/$label-warm.out" 2>&1 &&
     python3 cb.py "$url" "$model" . "$out/$label-512.json" --reps 2 --max-tokens 512 \
       --only copy-code,copy-prose,edit-rename,edit-docstring > "$out/$label-512.out" 2>&1)
    rc=$?
    tools/box_stop.sh child "$srv"
    wait "$srv" 2>/dev/null
    exit $rc' _ "$out" "$1" "$model" "$port" "$bench"
  echo "rc=$? ($1)"
}
arm copy-r16 16
arm copy-r64 64
python3 - "$out" <<'PY'
import json, sys
out = sys.argv[1]
for a in ('copy-r16', 'copy-r64'):
    rows = [json.loads(l) for l in open(f'{out}/{a}-512.out') if l.startswith('{')]
    print(a, ' '.join(f"{r['category']}:{r['total_tps']:.1f}" for r in rows))
PY
echo DEEP_VERIFY_COPY_DONE
