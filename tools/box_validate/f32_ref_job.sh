#!/usr/bin/env bash
# One f32 teacher-forced reference run ON THE BOX CPU, for the gates f32_oracle.sh's three golden prompts do not
# cover (operator ruling 2026-10-29: the MTP head's M1, the cached-vs-cold prefix sequences, the 2600-id prefill
# prompts). Like f32_oracle.sh: one CPU model job at a time (/tmp/b70-cpu-model.lock), MemAvailable checked first,
# the output written in the tree's oracle-out-f32-job.partial/ and moved into place once complete (a killed run
# starts again).
#   tools/box_validate/f32_ref_job.sh <ornith|k2> <prompt ids> <force ids> <out file> [extra ref.py flags...]
# <out file> is the golden-format dump, any host path (the data tree's oracle-out-*); the ids are whitespace-
# separated files, copied into the tree's oracle-out-f32-job.partial/ (the container sees only the tree, at /ws).
# The forced ids set --gen. Extra flags go to the reference (ornith_ref.py run --dtype f32, k2_ref.py run --mode
# f32); a `@PART@` in them is replaced by the container path of that directory (ornith --mtp-out @PART@/mtp).
# Env: F32_REF_MIN_GB (95), ORACLE_THREADS (22), MAX_PROMPT (8192), DTYPE (f32; bf16 for a bf16 twin),
# SNAP_ORNITH_CACHE / SNAP_K2_CACHE.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -ge 4 ] || { echo "usage: $0 <ornith|k2> <prompt ids> <force ids> <out file> [extra...]" >&2; exit 2; }
m="$1" prompt="$2" force="$3" out="$4"
shift 4
case "$m" in
  ornith) model="${SNAP_ORNITH_CACHE:-models--urakozz--Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}" ;;
  k2)     model="${SNAP_K2_CACHE:-models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}" ;;
  *) echo "f32_ref_job: model is ornith or k2, not $m" >&2; exit 2 ;;
esac
[ -s "$out" ] && { echo "f32_ref_job: $out kept"; exit 0; }
[ -s "$prompt" ] || { echo "f32_ref_job: $prompt is missing"; exit 1; }
[ -e "$force" ] || { echo "f32_ref_job: $force is missing"; exit 1; }
gen=$(wc -w < "$force" | tr -d ' ')   # 0: the prompt forward alone (no --force)
dt="${DTYPE:-f32}"                     # DTYPE=bf16: the bf16 reference beside a judge (prefill_split_test's)
part="oracle-out-f32-job.partial"
mkdir -p "$part" "$(dirname "$out")"
exec 9> /tmp/b70-cpu-model.lock
echo "f32_ref_job: waiting for /tmp/b70-cpu-model.lock ($(date '+%F %T'))"
flock 9
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
need="${F32_REF_MIN_GB:-95}"
if [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON f32_ref_job: MemAvailable ${avail} GB < F32_REF_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
name=$(basename "$out")
cp -f "$prompt" "$part/prompt.ids" && cp -f "$force" "$part/force.ids" || exit 1
extra="${*//@PART@//ws/$part}"
prompt="$part/prompt.ids" force="$part/force.ids"
if [ "$m" = ornith ]; then
  cmd="python3 tools/oracle/ornith_ref.py run \"\$SNAP\" --prompt /ws/$prompt --out /ws/$part/$name --gen $gen --dtype $dt --max-prompt ${MAX_PROMPT:-8192} $extra"
else
  cmd="python3 tools/oracle/k2_ref.py run \"\$SNAP\" --prompt /ws/$prompt --out /ws/$part/$name --gen $gen --mode $dt --max-prompt ${MAX_PROMPT:-8192} $extra"
fi
[ "$gen" -gt 0 ] && cmd="$cmd --force /ws/$force"
echo "f32_ref_job: $m $prompt + $gen forced -> $out; MemAvailable ${avail} GB ($(date '+%F %T'))"
if ! ORACLE_THREADS="${ORACLE_THREADS:-22}" ORACLE_MODEL="$model" tools/oracle/run_in_container.sh \
    "$cmd > /ws/$part/$name.log 2>&1"; then
  echo "f32_ref_job: FAILED"; tail -20 "$part/$name.log"; exit 1
fi
mv -f "$part/$name.log" "$out.log"
mv -f "$part/$name" "$out"
echo "f32_ref_job: done ($(date '+%F %T'))"; tail -2 "$out.log"
