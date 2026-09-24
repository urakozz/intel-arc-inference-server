#!/usr/bin/env bash
# Engine greedy runs for the tool-call set (spec 5 T0, gate A4), ON THE BOX.
#   tools/toolcall/engine_generate.sh <snapshot> <set dir> <out dir> <backend>
# For every scenario in <set dir>/manifest.json: check the .ids file's SHA-256
# against the manifest (the bf16 and engine runs must see byte-identical ids),
# run b70-decode with 192 greedy ids through the prefill path on <backend>
# (l0, l0-int8 once plan 5b lands, or sycl-tla), save its stdout as
# <out>/<name>.<backend>.ids and decode it with the snapshot's tokenizer into
# <out>/<name>.<backend>.txt (special tokens kept). Scenarios whose two outputs
# exist are skipped. Device: pass ZE_AFFINITY_MASK as usual.
# Decoding needs the `tokenizers` package. The box's host python3 has none, so
# when $PYTHON cannot import it the decode runs in the reference container
# (ORACLE_IMAGE, as tools/oracle/run_in_container.sh), with only the output dir
# and the resolved tokenizer.json mounted.
# Env: B70_DECODE (default build/src/cli/b70-decode), PYTHON (default python3),
#      ORACLE_IMAGE.
set -euo pipefail
[ $# -eq 4 ] || { echo "usage: $0 <snapshot> <set dir> <out dir> <backend>" >&2; exit 2; }
SNAP="$1" SET="$2" OUT="$3" BACKEND="$4"
DECODE="${B70_DECODE:-build/src/cli/b70-decode}"
PY="${PYTHON:-python3}"
[ -x "$DECODE" ] || { echo "no b70-decode at $DECODE" >&2; exit 2; }
[ -f "$SNAP/tokenizer.json" ] || { echo "$SNAP has no tokenizer.json" >&2; exit 2; }
mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd)
TOK_JSON=$(readlink -f "$SNAP/tokenizer.json")
IMAGE="${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}"
DECODE_PY='import sys; from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1])
ids = [int(x) for x in open(sys.argv[2]).read().split()]
open(sys.argv[3], "w", encoding="utf-8").write(t.decode(ids, skip_special_tokens=False))'
if "$PY" -c 'import tokenizers' 2> /dev/null; then
  decode() { "$PY" -c "$DECODE_PY" "$TOK_JSON" "$OUT_ABS/$1" "$OUT_ABS/$2"; }
else
  decode() {
    docker run --rm --entrypoint python3 -u "$(id -u):$(id -g)" \
      -v "$OUT_ABS:/out" -v "$TOK_JSON:/tok.json:ro" "$IMAGE" -c "$DECODE_PY" /tok.json "/out/$1" "/out/$2"
  }
fi

# name, id count and SHA-256 per line, from the manifest (plain python3 json).
entries=$("$PY" -c 'import json,sys
for e in json.load(open(sys.argv[1])): print(e["name"], e["ids"], e["sha256"])' "$SET/manifest.json")

n=0
while read -r name count sha; do
  n=$((n + 1))
  ids="$SET/$name.ids"
  got=$(sha256sum "$ids" | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { echo "FATAL: $name.ids SHA-256 $got != manifest $sha" >&2; exit 1; }
  out_ids="$OUT/$name.$BACKEND.ids" out_txt="$OUT/$name.$BACKEND.txt"
  if [ -s "$out_ids" ] && [ -f "$out_txt" ]; then
    echo "[$n] $name: done, skipped"; continue
  fi
  t0=$(date +%s)
  "$DECODE" "$SNAP" --ids "$ids" --n 192 --prefill --pp-backend "$BACKEND" --max-len 16384 < /dev/null \
    > "$out_ids.tmp" 2> "$OUT/$name.$BACKEND.log"
  mv "$out_ids.tmp" "$out_ids"
  decode "$name.$BACKEND.ids" "$name.$BACKEND.txt" < /dev/null
  echo "[$n] $name: $count prompt ids, $(( $(date +%s) - t0 ))s"
done <<< "$entries"
