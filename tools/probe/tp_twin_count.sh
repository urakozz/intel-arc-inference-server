#!/bin/bash
# The tensor-parallel twin count's arms (docs/probes/tp-twin-count-2026-11-10.md), ON THE BOX, from the tree root,
# under the caller's GPU lock (this script takes none):
#   tools/gpu_lock.sh card <N> -- tools/probe/tp_twin_count.sh <out dir> one   one card: b70-decode one-pf / one-in /
#                                                                              one-mtp, b70-serve one / one-auto
#   tools/gpu_lock.sh box -- tools/probe/tp_twin_count.sh <out dir> tp         --parallel tp: tp-pf / tp-in / tp-mtp,
#                                                                              b70-serve tp / tp-auto
# Both KV forms (KVS, default "bf16 int8"). Each b70-decode run's ids go to <out>/<kv>/ids/<arm>/<name>.ids and into
# <out>/<kv>/decode/<arm>.r1/requests.jsonl (serve_client.py record); each server arm into
# <out>/<kv>/serve/<arm>.r1 (serve_run.sh). Resumable: a b70-decode run with its N ids already there is skipped, a
# server arm with its requests.jsonl complete (39 lines) too. A file <out>/STOP ends the script before its next run (a
# clean resume point: the cards handed to another job). tools/box_validate/twin_count.py reads <out>.
set -u
cd "$(dirname "$0")/../.." || exit 2
out="${1:?out dir}" side="${2:?one | tp}"
M=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
N=${N:-256} LEN=${LEN:-16384} KVS=${KVS:-bf16 int8}
case "$side" in
  one) par="--parallel 1" ;;
  tp) par="--parallel tp" ;;
  *) echo "tp_twin_count: one | tp, not '$side'" >&2; exit 2 ;;
esac
mkdir -p "$out/prompts"
# The 41 contexts: the golden three, the 36 tool-call prompts, long, long32k[:6000] (set / name / ids file).
ctx="$out/prompts/contexts.txt"
if [ ! -s "$ctx" ]; then
  { for p in prose code cjk; do echo "golden $p tests/golden/prompts/$p.ids"; done
    for f in tests/golden/toolcall/*.ids; do echo "toolcall $(basename "$f" .ids) $f"; done
    echo "test long tests/golden/prompts/long.ids"
    tr -s ' \n' '\n' < tests/golden/prompts/long32k.ids | grep . | head -6000 | tr '\n' ' ' > "$out/prompts/long32k_6000.ids"
    echo "test long32k_6000 $out/prompts/long32k_6000.ids"; } > "$ctx"
fi
echo "tp_twin_count: $side, $(wc -l < "$ctx") contexts, KV $KVS, $N ids, max-len $LEN ($(date '+%F %T'))"
rc=0
for kv in $KVS; do
  for arm in pf in mtp; do
    case "$arm" in
      pf) flags="--prefill" ;;
      in) flags="" ;;
      mtp) flags="--prefill --spec auto" ;;
    esac
    a="$side-$arm"
    mkdir -p "$out/$kv/ids/$a" "$out/$kv/err/$a"
    while read -r set name ids; do
      f="$out/$kv/ids/$a/$name.ids"
      if [ "$(grep -c . "$f" 2>/dev/null)" = "$N" ]; then continue; fi
      [ -e "$out/STOP" ] && { echo "tp_twin_count: STOP before $kv $a $name"; exit 3; }
      t0=$(date +%s)
      # shellcheck disable=SC2086
      build/src/cli/b70-decode $M --ids "$ids" --n "$N" --max-len "$LEN" --kv-cache "$kv" $par $flags \
        > "$f.part" 2> "$out/$kv/err/$a/$name.err"
      r=$?
      if [ "$r" != 0 ] || [ "$(grep -c . "$f.part")" != "$N" ]; then
        echo "FAIL $kv $a $name rc=$r ($(tail -1 "$out/$kv/err/$a/$name.err"))"
        rc=1
        continue
      fi
      mv -f "$f.part" "$f"
      python3 tools/box_validate/serve_client.py record --out "$out/$kv/decode/$a.r1/requests.jsonl" --label "$a" \
        --set "$set" --name "$name" --prompt "$ids" --ids "$f" > /dev/null || rc=1
      echo "$kv $a $set/$name: $(( $(date +%s) - t0 )) s"
    done < "$ctx"
  done
  for spec in off auto; do
    a="$side"; [ "$spec" = auto ] && a="$side-auto"
    d="$out/$kv/serve/$a.r1"
    if [ "$(grep -c . "$d/requests.jsonl" 2>/dev/null)" = 39 ]; then continue; fi
    [ -e "$out/STOP" ] && { echo "tp_twin_count: STOP before $kv serve $a"; exit 3; }
    PORT=${PORT:-8047} tools/box_validate/serve_run.sh "$d" "$a" 1 "$M" --max-len "$LEN" $par --spec "$spec" \
      --kv-cache "$kv" -- --set golden --set toolcall --max-tokens "$N" | grep -E '^(SERVE_RUN|serve_run:|max_len|memory)' || true
    [ "$(grep -c . "$d/requests.jsonl" 2>/dev/null)" = 39 ] || { echo "FAIL $kv serve $a"; rc=1; }
  done
done
echo "tp_twin_count: $side done rc=$rc ($(date '+%F %T'))"
exit "$rc"
