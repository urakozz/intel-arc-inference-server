#!/usr/bin/env bash
# The f32 references beyond f32_oracle.sh's golden prompts (operator ruling 2026-10-29), ON THE BOX CPU, each run
# through tools/box_validate/f32_ref_job.sh (one CPU model job at a time, RAM checked, resumable):
#   tools/box_validate/f32_refs.sh <data tree> ornith-mtp
#       r16.m1's judge: ornith_ref.py --dtype f32 --mtp-out, per golden prompt, teacher-forced on the bf16 M1
#       reference's continuation (oracle-out-ornith-mtp/<p>.cont256.ids, its first 32) ->
#       oracle-out-ornith-mtp-f32/m1/<p>.mtp.safetensors (mtp_head_test's format) + <p>.golden.safetensors
#   tools/box_validate/f32_refs.sh <data tree> forced <ornith|k2> <dir> [<suffix> [extra ref.py flags...]]
#       every <dir>/<key>.prompt.ids with its <key>.force.ids (a test's own engine run wrote them:
#       B70_DUMP_FORCED=<dir>, tests/golden/forced_judge.h) -> <data tree>/oracle-out-<model>-f32<suffix>-forced/
#       <key>.golden.safetensors, run --lean (e.g. suffix -i8head with --lm-head int8: the int8 head's judges)
#       DTYPE=bf16 ... '' --gemm f32: each judge's bf16 twin -> oracle-out-<model>-bf16-forced/ (operator ruling
#       2026-10-31: the near-miss gates' bar is the twin's own count of ids the judge rejects)
#   tools/box_validate/f32_refs.sh <data tree> split <ornith|k2> <ids> <N> <out stem>
#       prefill_split_test's references of ids[0, N): f32 and bf16, --lean --dump-kv, no generated rows ->
#       <out stem>-f32.golden.safetensors and <out stem>-bf16.golden.safetensors
set -u
cd "$(dirname "$0")/../.." || exit 2
data="${1:?data tree}" what="${2:?ornith-mtp | forced}"
rc=0
case "$what" in
  ornith-mtp)
    src="$data/oracle-out-ornith-mtp" out="$data/oracle-out-ornith-mtp-f32"
    mkdir -p "$out/m1" oracle-out-f32-job.partial
    for p in prose code cjk; do
      if [ -s "$out/m1/$p.mtp.safetensors" ]; then echo "f32_refs: $p kept"; continue; fi
      [ -s "$src/$p.cont256.ids" ] || { echo "f32_refs: $src/$p.cont256.ids is missing"; rc=1; continue; }
      tr -s ' \n' '\n' < "$src/$p.cont256.ids" | grep . | head -32 | tr '\n' ' ' > "oracle-out-f32-job.partial/$p.mtp.force"
      rm -f "$out/$p.golden.safetensors"   # the M1 file is made by the same run
      tools/box_validate/f32_ref_job.sh ornith "tests/golden/prompts/$p.ids" "oracle-out-f32-job.partial/$p.mtp.force" \
        "$out/$p.golden.safetensors" --mtp-out @PART@/mtp || { rc=1; continue; }
      mv -f oracle-out-f32-job.partial/mtp/m1/prompt.mtp.safetensors "$out/m1/$p.mtp.safetensors" || rc=1
      cp -f "$src/$p.cont256.ids" "$out/$p.cont256.ids"
    done
    ;;
  forced)
    m="${3:?ornith | k2}" dir="${4:?the dumped prompt / force ids}" suffix="${5:-}"
    shift 4; [ $# -gt 0 ] && shift
    out="$data/oracle-out-$m-${DTYPE:-f32}$suffix-forced"   # DTYPE=bf16: the judges' bf16 twins (B70_FORCED_BF16)
    for pr in "$dir"/*.prompt.ids; do
      [ -e "$pr" ] || { echo "f32_refs: no *.prompt.ids in $dir"; rc=1; break; }
      n=$(basename "$pr" .prompt.ids)
      tools/box_validate/f32_ref_job.sh "$m" "$pr" "$dir/$n.force.ids" "$out/$n.golden.safetensors" --lean "$@" || rc=1
    done
    ;;
  split)
    m="${3:?ornith | k2}" ids="${4:?ids}" n="${5:?N}" stem="${6:?out stem}"
    mkdir -p oracle-out-f32-job.partial
    tr -s ' \n' '\n' < "$ids" | grep . | head -"$n" | tr '\n' ' ' > oracle-out-f32-job.partial/split.prompt
    : > oracle-out-f32-job.partial/split.none
    tools/box_validate/f32_ref_job.sh "$m" oracle-out-f32-job.partial/split.prompt oracle-out-f32-job.partial/split.none \
      "$stem-f32.golden.safetensors" --lean --dump-kv || rc=1
    DTYPE=bf16 tools/box_validate/f32_ref_job.sh "$m" oracle-out-f32-job.partial/split.prompt \
      oracle-out-f32-job.partial/split.none "$stem-bf16.golden.safetensors" --lean --dump-kv || rc=1
    ;;
  *) echo "f32_refs: ornith-mtp | forced <model> <dir> [suffix [flags]] | split <model> <ids> <N> <stem>" >&2; exit 2 ;;
esac
exit "$rc"
