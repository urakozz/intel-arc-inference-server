#!/usr/bin/env bash
# Spec 19 v2 helpers, on the box. GPU modes run under the caller's gpu_lock; ref takes the CPU model lock.
# dflash_v2.sh sources|gpu|ref|d5|identity OUT_DIR TARGET DRAFTER
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?sources|gpu|ref|d5}
out=${2:?out dir}
target=$(tools/box_validate/data.sh resolve "${3:?target}")
drafter=$(tools/box_validate/data.sh resolve "${4:?drafter}")
mkdir -p "$out"
case "$mode" in
  sources)
    python3 tools/box_validate/dflash_v2_sources.py oracle-out-19a/dumps \
      "$HOME/b70/logs/opencode-session-2" "$target/generation_config.json" "$out/sources.json"
    ;;
  gpu)
    build/tools/probe/probe_dflash_iter "$target" "$drafter" --format int8 --candidate-head --engine-taps "$out/sources.json" "$out/engine"
    ;;
  ref)
    exec 9> /tmp/b70-cpu-model.lock
    echo "DFlash v2 reference waiting for CPU model lock"
    flock -x 9
    hf=${HF_HOME:-$HOME/.cache/huggingface}
    out=$(cd "$out" && pwd -P)
    target=/hf${target#"$hf"}
    drafter=/hf${drafter#"$hf"}
    docker run --rm --entrypoint python3 -u "$(id -u):$(id -g)" --memory 64g --memory-swap 64g \
      -e HF_HUB_OFFLINE=1 -e HOME=/scratch -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 \
      -v "$PWD:/ws" -w /ws -v "$hf:/hf:ro" -v "$out:/out" -v /tmp:/scratch \
      "${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest}" \
      tools/oracle/dflash_engine_accept.py --engine /out/engine --target "$target" --drafter "$drafter" \
      --out /out/acceptance.json --threads 16
    ;;
  d5)
    # Operator 2026-11-12: existing MTP 73.2 t/s and plain 33.5 t/s; no baseline serving reruns.
    PORT=${PORT:-8213} tools/box_validate/slots_serve_bench.sh "$out/slots" "$target" dflash \
      --max-len 32768 --kv-cache bf16 --spec dflash --draft-model "$drafter" -- --counts 1
    python3 - "$out/slots/bench.jsonl" <<'PY'
import json, sys
rows = [json.loads(s) for s in open(sys.argv[1]) if s.strip()]
if len(rows) != 1 or rows[0]['n'] != 1:
    raise SystemExit('DFlash v2: expected one slots_bench row at count 1')
row = rows[0]
rate = row['warm']['thr']
print(f'DFLASH_EARLY tps={rate:.5f} mtp_baseline=73.2 ratio={rate / 73.2:.5f} pause={rate > 80.5}')
PY
    ;;
  identity)
    reference=$(python3 - <<'PY'
import glob, json, os
paths = glob.glob(os.path.expanduser('~/b70-validate/b70-inference-server-validate/*/r55.serve/off.r1/requests.jsonl'))
for path in sorted(paths, key=os.path.getmtime, reverse=True):
    rows = [json.loads(s) for s in open(path) if s.strip()]
    if len(rows) == 39 and {r['set'] for r in rows} == {'golden', 'toolcall'} and all(r.get('out_ids') for r in rows):
        print(path)
        break
PY
)
    [ -n "$reference" ] || { echo 'SKIP: no recorded 39-prompt plain correctness baseline'; exit 77; }
    mkdir -p "$out/identity/off.r1"
    cp "$reference" "$out/identity/off.r1/requests.jsonl"
    printf '%s\n' "$reference" > "$out/identity/reference.path"
    PORT=${PORT:-8213} tools/box_validate/serve_run.sh "$out/identity/dflash.r1" dflash 1 "$target" \
      --max-len 16384 --spec dflash --draft-model "$drafter" \
      -- --set toolcall --set golden --max-tokens 192 --stop-at-eos
    python3 tools/box_validate/serve_client.py compare --dir "$out/identity" --ref off || true
    python3 tools/box_validate/serve_judge.py --dir "$out/identity" --ref off \
      --dump "$out/identity/forced" --judge oracle-out-qwen-f32-i8head-forced --bf16 oracle-out-qwen-bf16top-forced
    ;;
  *) echo "unknown DFlash v2 mode: $mode" >&2; exit 2 ;;
esac
