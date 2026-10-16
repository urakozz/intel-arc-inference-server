#!/bin/bash
# tools/box_validate/dflash_d1.sh - spec 19 §6 D1 for the DFlash drafter (plan 19b Task 3), ON THE BOX, from a tree root.
#
#   tools/box_validate/dflash_d1.sh gpu <out dir>   the engine's half: build/tools/probe/dflash_d1 over 19a's tap dumps,
#                                                    one JSONL per arm (the card; run it under the GPU lock)
#   tools/box_validate/dflash_d1.sh ref <out dir>   the reference's half: tools/oracle/dflash_d1.py per arm in the oracle
#                                                    image (CPU), D1's bars; the table appended to <out dir>/d1.md
#
# The arms (spec 19 decision 2 and §4 decision 6): i8 - the drafter int8 (quantised at load), the head int8; bf16 - the
# drafter and the head bf16; i8dv32k - i8 over a 32768-id draft vocabulary (19a's ranked list).
# Env: SNAP_QWEN (the target: the gate checkpoint), SNAP_DFLASH (z-lab/Qwen3.8-27B-DFlash2), DUMPS (oracle-out-19a/dumps),
#      RANKED (oracle-out-19a/draft_vocab.ranked.ids), ANCHORS (a dump, default 16), ARMS (default "i8 bf16 i8dv32k"),
#      ORACLE_IMAGE, ORACLE_THREADS, HF_HOME.
set -u
mode="${1:?gpu|ref}"
out="${2:?out dir}"
mkdir -p "$out"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
DUMPS="${DUMPS:-oracle-out-19a/dumps}"
RANKED="${RANKED:-oracle-out-19a/draft_vocab.ranked.ids}"
ARMS="${ARMS:-i8 bf16 i8dv32k}"
target=$(tools/box_validate/data.sh resolve "${SNAP_QWEN:?SNAP_QWEN (the target) is unset}") || exit 2
drafter=$(tools/box_validate/data.sh resolve "${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}") || exit 2
flags() {
  case "$1" in
    i8) echo "--format int8 --head int8" ;;
    bf16) echo "--format bf16 --head bf16" ;;
    i8dv32k) echo "--format int8 --head int8 --draft-vocab 32k --draft-vocab-ids $RANKED" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}
case "$mode" in
  gpu)
    for arm in $ARMS; do
      f=$(flags "$arm") || exit 2
      echo "=== $(date '+%T') arm $arm: dflash_d1 $f"
      # shellcheck disable=SC2086 # $f is a word list
      build/tools/probe/dflash_d1 "$target" "$drafter" "$out/d1.$arm.jsonl" $f --anchors "${ANCHORS:-16}" \
        "$DUMPS"/*.taps.safetensors || exit 1
    done
    ;;
  ref)
    image="${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}"
    dumps_host=$(cd "$DUMPS" && pwd -P) || exit 2
    out_host=$(cd "$out" && pwd -P)
    in_hf() { printf '/hf%s' "${1#"$HF"}"; }
    env_flags=(-e HF_HUB_OFFLINE=1 -e HOME=/scratch -e PYTHONUNBUFFERED=1)
    [ -n "${ORACLE_THREADS:-}" ] && env_flags+=(-e "OMP_NUM_THREADS=$ORACLE_THREADS" -e "MKL_NUM_THREADS=$ORACLE_THREADS")
    rc=0
    for arm in $ARMS; do
      echo "=== $(date '+%T') arm $arm: dflash_d1.py"
      docker run --rm --entrypoint bash -u "$(id -u):$(id -g)" -v "$PWD:/ws" -w /ws -v "$HF:/hf:ro" \
        -v "$dumps_host:/dumps:ro" -v "$out_host:/out" -v /tmp:/scratch "${env_flags[@]}" "$image" -c \
        "python3 tools/oracle/dflash_d1.py --d1 /out/d1.$arm.jsonl --dumps /dumps --drafter $(in_hf "$drafter") \
           --target $(in_hf "$target") --md /out/d1.md" || rc=1
    done
    exit $rc
    ;;
  *) echo "dflash_d1.sh: gpu|ref" >&2; exit 2 ;;
esac
