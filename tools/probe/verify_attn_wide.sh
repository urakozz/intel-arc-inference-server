#!/usr/bin/env bash
# Plan 33b (spec 33 §6): the 16-row verify attention's PAIR line at KS 1. From the tree root on the box, after the
# build; each GPU command takes its own card lock (nothing here builds or polls under one). One log per command.
#   tools/probe/verify_attn_wide.sh compile OUT       the line through ocloc at PVR 1 / 2 (no GPU): spills, private
#                                                     memory, SLM from the zebin's .ze_info
#   tools/probe/verify_attn_wide.sh absolute CARD OUT today's 16-row line, absolute times: probe_kv8_decode (random
#                                                     data, 16 caches) beside probe_verify_attn (real q / KV, one
#                                                     layer, L0 timestamps), the cards' clocks sampled throughout
#   tools/probe/verify_attn_wide.sh decompose CARD OUT  today's 16-row line: random / real KV x 16 caches / one
#   tools/probe/verify_attn_wide.sh bitwise CARD OUT PREFIX...  16 rows bitwise against the frozen copy, + negative
#   tools/probe/verify_attn_wide.sh inplace CARD OUT TAG  the 16-row list in place, 4k / 32k keys
#   tools/probe/verify_attn_wide.sh kernel CARD OUT   today's 16-row line against the candidates at pos 4096 / 32768 / 98304 (one row group), then 32767 / 98303 (rows across a block-size change: two walks)
set -u
mode="$1"
shift
run() {   # name, command...
  local name="$1"
  shift
  echo "== $name $(date '+%F %T')"
  tools/gpu_lock.sh card "$card" -- "$@" > "$out/$name.log" 2>&1
  echo "rc=$? ($name)"
}
# the cards' actual clocks (MHz) every 0.5 s into $1 until killed
clocks() {
  while :; do
    printf '%s' "$(date +%T.%N | cut -c1-12)"
    for f in /sys/class/drm/card*/device/tile0/gt0/freq0/act_freq; do printf ' %s' "$(cat "$f")"; done
    echo
    sleep 0.5
  done > "$1"
}
LINE="-DCTRL_POS=0 -DCTRL_NACT=1 -DM=16 -DTGT=32 -DKT=32 -DPACK_H=0 -DKS=1 -DDSPLIT=128 -DQSPLIT=1 -DPSPLIT=1 -DKV8=0"
case "$mode" in
  compile)
    out="$1"
    mkdir -p "$out"
    OC=$(grep OCLOC_EXECUTABLE: build/CMakeCache.txt | cut -d= -f2)
    DEV=$(grep B70_OCLOC_DEVICE build/CMakeCache.txt | head -1 | cut -d= -f2)
    for v in "today:" "pvr1:-DPAIR=1 -DPVX=8 -DPVR=1" "pvr2:-DPAIR=1 -DPVX=8 -DPVR=2"; do
      n=wide_${v%%:*}_M16
      "$OC" compile -file src/kernels/attn_verify_dpas.cl -device "${DEV:-bmg-g31}" -output "$n" -output_no_suffix \
        -out_dir "$out" -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I$PWD/src/kernels $LINE \
        ${v#*:} -cl-intel-256-GRF-per-thread" > "$out/$n.build.log" 2>&1
      echo "ocloc rc=$? ($n)"
      grep -iE "spill|error|warning" "$out/$n.build.log" || echo "  no spill / error / warning line"
      "$OC" disasm -file "$out/$n.bin" -dump "$out/$n.dis" > /dev/null 2>&1
      grep -hE "spill_size|private_size|slm_size|grf_count|has_stack|simd_size" "$out/$n.dis"/*ze_info* 2>/dev/null |
        sed 's/^ */  /' | sort -u
    done
    ;;
  absolute)
    card="$1" out="$2"
    mkdir -p "$out"
    clocks "$out/clocks.txt" &
    cpid=$!
    r="attn_verify_dpas_M16_T32+attn_v2_M16_T32"
    run kv8probe_M16 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r" "$r@dec"
    run vattn_M16 build/tools/probe/probe_verify_attn arms "$HOME/b70/logs/spec28c/vattn-dump" --m 16 \
      --depths 32768 --arms K32_Pg_S1_D128_Q1_G256_P1 --time --rounds 5 --reps 10
    kill $cpid
    echo ABSOLUTE_DONE
    ;;
  decompose)   # the absolute gap split: random / real KV and q, 16 caches / one (probe_kv8_decode's PKD_* switches)
    card="$1" out="$2"
    mkdir -p "$out"
    r="attn_verify_dpas_M16_T32+attn_v2_M16_T32"
    dump="$HOME/b70/logs/spec28c/vattn-dump"
    run random16 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    run random1 env PKD_ONE_CACHE=1 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    run real16 env PKD_KV_DUMP="$dump" build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    run real1 env PKD_KV_DUMP="$dump" PKD_ONE_CACHE=1 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    run pad8 env PKD_CAP_PAD=8 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    run pad8real1 env PKD_CAP_PAD=8 PKD_KV_DUMP="$dump" PKD_ONE_CACHE=1 build/tools/probe/probe_kv8_decode 16 32767,98303 "$r"
    echo DECOMPOSE_DONE
    ;;
  bitwise)   # attn_verify_bitwise_test at 16 rows: the candidates (probe builds) against the frozen copy, then V zeroed
    card="$1" out="$2"
    shift 2
    mkdir -p "$out"
    for c in "$@"; do
      run "bitwise_$c" build/tests/attn_verify_bitwise_test --cand "$c" --min-m 16 --max-m 16
    done
    run bitwise_negative build/tests/attn_verify_bitwise_test --negative --min-m 16 --max-m 16
    echo BITWISE_DONE
    ;;
  inplace)   # the engine's 16-row list in place against the plain step (probe_mtp_steps wide16), 4k / 32k keys
    card="$1" out="$2" tag="$3"
    mkdir -p "$out"
    snap=$(tools/box_validate/data.sh resolve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ) || exit 2
    run "wide16_4096_$tag" build/tools/probe/probe_mtp_steps "$snap" 4096 16 5 int8 off 16384 4 wide16
    run "wide16_32768_$tag" build/tools/probe/probe_mtp_steps "$snap" 32768 16 5 int8 off 65536 4 wide16
    echo INPLACE_DONE
    ;;
  kernel)
    card="$1" out="$2"
    shift 2
    mkdir -p "$out"
    r="attn_verify_dpas_M16_T32+attn_v2_M16_T32"
    arms=("$r")
    for c in "$@"; do arms+=("$c+attn_v2_M16_T32"); done
    run kernel_M16 build/tools/probe/probe_kv8_decode 16 4096,32768,98304 "${arms[@]}"
    run kernel_M16_straddle build/tools/probe/probe_kv8_decode 16 32767,98303 "${arms[@]}"
    echo KERNEL_DONE
    ;;
  *)
    echo "usage: $0 compile OUT | absolute CARD OUT | kernel CARD OUT [candidate binary ...]" >&2
    exit 2
    ;;
esac
