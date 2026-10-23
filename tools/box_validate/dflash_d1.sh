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
#
# MODEL=muse (spec 23f Task 7): Muse-Glimmer-30B's assistant - the target SNAP_MUSE (the int4 checkpoint: dflash_d1 loads
# its embedding and head with ONE layer), the drafter SNAP_MUSE_DFLASH (meta-models/Muse-Glimmer-30B-assistant), DUMPS
# default oracle-out-muse-taps/dumps (muse_oracle.sh taps), K 15, the arms i4 (int4 g64, the default), i8, bf16 - each
# with the int8 head (the served default) and --draft-q raw (the readback's scores are the raw head logits).
#
# MODEL=lilicorr (spec 26b Task 6): the LiLiCorr drafter beside Bolt - the target SNAP_BOLT (SergiioB's g128: dflash_d1
# --model lilicorr reads only its embedding, uploaded as the drafter's replica, and its lm_head), the drafter SNAP_LILICORR
# (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr), DUMPS default oracle-out-bolt-taps/dumps (25a's bolt_oracle.sh
# taps), every draft the whole 8-row block (K 7), the arms i8 (int8, spec 26 §11's PROVISIONAL default), exact, i4 (int4
# g64) - each with the int8 head. `ref` holds them to dflash_ref.py's LiLiCorr form (LP_BAR: the log-probs' bar, spec 26
# §7's 1e-3 by default). `twin <out dir>`: the host twin (build/tests/lilicorr_d1_twin, Engine numerics) over the same dumps
# per arm, then its D1 (CPU; hours at the real shapes - ANCHORS small).
set -u
mode="${1:?gpu|ref}"
out="${2:?out dir}"
mkdir -p "$out"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
RANKED="${RANKED:-oracle-out-19a/draft_vocab.ranked.ids}"
if [ "${MODEL:-qwen38}" = lilicorr ]; then
  DUMPS="${DUMPS:-oracle-out-bolt-taps/dumps}"
  ARMS="${ARMS:-i8 exact i4}"
  target=$(tools/box_validate/data.sh resolve "${SNAP_BOLT:-SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym}") || exit 2
  drafter=$(tools/box_validate/data.sh resolve "${SNAP_LILICORR:-nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr}") || exit 2
elif [ "${MODEL:-qwen38}" = muse ]; then
  DUMPS="${DUMPS:-oracle-out-muse-taps/dumps}"
  ARMS="${ARMS:-i4 i8 bf16}"
  muse_target="${SNAP_MUSE:-}"
  if [ -z "$muse_target" ]; then
    muse_target=Vishva007/Muse-Glimmer-30B-W4A16-AutoRound
    tools/box_validate/data.sh resolve "$muse_target" > /dev/null 2>&1 || muse_target=Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ
  fi
  target=$(tools/box_validate/data.sh resolve "$muse_target") || exit 2
  drafter=$(tools/box_validate/data.sh resolve "${SNAP_MUSE_DFLASH:-meta-models/Muse-Glimmer-30B-assistant}") || exit 2
else
  DUMPS="${DUMPS:-oracle-out-19a/dumps}"
  ARMS="${ARMS:-i8 bf16 i8dv32k}"
  target=$(tools/box_validate/data.sh resolve "${SNAP_QWEN:?SNAP_QWEN (the target) is unset}") || exit 2
  drafter=$(tools/box_validate/data.sh resolve "${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}") || exit 2
fi
flags() {
  if [ "${MODEL:-qwen38}" = lilicorr ]; then
    case "$1" in
      i8) echo "--model lilicorr --format int8 --head int8" ;;
      exact) echo "--model lilicorr --format exact --head int8" ;;
      i4) echo "--model lilicorr --format int4 --head int8" ;;
      *) echo "unknown LiLiCorr arm $1" >&2; return 1 ;;
    esac
    return
  fi
  case "$1" in
    i4) echo "--format int4 --head int8 --k 15 --draft-q raw" ;;   # MODEL=muse
    i8) [ "${MODEL:-qwen38}" = muse ] && { echo "--format int8 --head int8 --k 15 --draft-q raw"; return; }
        echo "--format int8 --head int8" ;;
    bf16) [ "${MODEL:-qwen38}" = muse ] && { echo "--format bf16 --head int8 --k 15 --draft-q raw"; return; }
          echo "--format bf16 --head bf16" ;;
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
           --target $(in_hf "$target") --md /out/d1.md --lp-bar ${LP_BAR:-1e-3}" || rc=1
    done
    exit $rc
    ;;
  twin)   # MODEL=lilicorr: the host twin's drafts over the same dumps, then their D1 (as `ref`, the records d1.twin-<arm>)
    [ "${MODEL:-qwen38}" = lilicorr ] || { echo "dflash_d1.sh twin is MODEL=lilicorr's" >&2; exit 2; }
    rc=0
    for arm in $ARMS; do
      f=$(flags "$arm") || exit 2
      f=${f#--model lilicorr }
      echo "=== $(date '+%T') arm $arm: lilicorr_d1_twin $f"
      # shellcheck disable=SC2086 # $f is a word list
      build/tests/lilicorr_d1_twin "$target" "$drafter" "$out/d1.twin-$arm.jsonl" $f --anchors "${ANCHORS:-4}" \
        "$DUMPS"/*.taps.safetensors || rc=1
    done
    [ $rc = 0 ] && ARMS=$(for a in $ARMS; do printf 'twin-%s ' "$a"; done) "$0" ref "$out"
    ;;
  *) echo "dflash_d1.sh: gpu|ref|twin" >&2; exit 2 ;;
esac
