#!/usr/bin/env bash
# Plan 33a (spec 33 §7): where the verify attention's time goes at 32,768 and 98,304 keys. From the tree root on the
# box, after the build; each GPU command takes its own card lock (nothing here builds or polls under one). Run
# detached (tools/probe/detach.sh); one log per command under OUT.
#   tools/probe/verify_attn_depth.sh compile OUT       plan 33b's line through ocloc (no GPU): builds or not, SLM, spills
#   tools/probe/verify_attn_depth.sh gate CARD OUT     G0: the unablated probe copies bitwise production
#   tools/probe/verify_attn_depth.sh session CARD OUT  Task 1's lines
#   tools/probe/verify_attn_depth.sh half CARD OUT     16 rows: option B's halved score chain beside the bound arm
set -u
mode="$1"
shift
B=build/tools/probe/probe_kv8_decode
D=32767,98303
run() {   # name, command...
  local name="$1"
  shift
  echo "== $name $(date '+%F %T')"
  tools/gpu_lock.sh card "$card" -- "$@" > "$out/$name.log" 2>&1
  echo "rc=$? ($name)"
}
p() { echo "vdpas_vad_$1_M$2$3+attn_v2_M$2_T32$3"; }               # tag, rows, ""|_KV8
prod() { echo "attn_verify_dpas_M$1_T32$2+attn_v2_M$1_T32$2"; }    # rows, ""|_KV8
case "$mode" in
  compile)
    out="$1"
    mkdir -p "$out"
    OC=$(grep OCLOC_EXECUTABLE: build/CMakeCache.txt | cut -d= -f2)
    DEV=$(grep B70_OCLOC_DEVICE build/CMakeCache.txt | head -1 | cut -d= -f2)
    n=vad_pair_ks1_M16
    "$OC" compile -file tools/probe/verify_dpas_ablate.cl -device "${DEV:-bmg-g31}" -output "$n" -output_no_suffix \
      -out_dir "$out" -options "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -I$PWD/src/kernels \
      -DCTRL_POS=0 -DCTRL_NACT=1 -DM=16 -DTGT=32 -DKT=32 -DPACK_H=0 -DKS=1 -DDSPLIT=128 -DQSPLIT=1 -DPSPLIT=1 -DPAIR=1 \
      -DPVX=8 -DKV8=0 -cl-intel-256-GRF-per-thread" > "$out/$n.build.log" 2>&1
    echo "ocloc rc=$? ($n)"
    grep -iE "spill|error|warning" "$out/$n.build.log" || echo "no spill / error / warning line"
    ls -l "$out/$n.bin" 2>/dev/null
    ;;
  gate)
    card="$1" out="$2"
    mkdir -p "$out"
    for m in 4 8 16; do
      run "gate_probe_M$m" $B $m $D "$(prod $m "")" "$(p full $m "")"
    done
    run gate_bitwise_M4 build/tests/attn_verify_bitwise_test --cand vdpas_vad_full_ --min-m 4 --max-m 4
    run gate_bitwise_M8 build/tests/attn_verify_bitwise_test --cand vdpas_vad_full_ --min-m 8 --max-m 8
    echo GATE_DONE
    ;;
  half)   # the 16-row arms with option B's halved score chain a sub-group (beside the rule's bound arm)
    card="$1" out="$2"
    mkdir -p "$out"
    run bf16_M16_half $B 16 $D "$(prod 16 "")" "$(p bound16 16 "")" "$(p shalf 16 "")" "$(p bhalf16 16 "")" \
      "$(p nok 16 "")" "$(p nov 16 "")"
    echo HALF_DONE
    ;;
  session)
    card="$1" out="$2"
    mkdir -p "$out"
    for m in 4 8 16; do
      arms=("$(prod $m "")")
      for t in full nok nov noqk nosoft nopv noout; do arms+=("$(p $t $m "")"); done
      [ $m = 8 ] && arms+=("$(p bound8 8 "")")
      [ $m = 16 ] && arms+=("$(p bound16 16 "")" "$(p sdup 16 "")" "$(p vone 16 "")")
      arms+=("$(prod $m "")@dec" "$(prod $m "")@red")
      run "bf16_M$m" $B $m $D "${arms[@]}"
    done
    # rows 32 / 64: the 16-row line's launch 2 / 4 times a layer, the repeats with and without their reads
    r="attn_verify_dpas_M16_T32+attn_v2_M16_T32"
    run rep_M16 $B 16 $D "$r@dec" "$r:x2@dec" "$r:x2=vdpas_vad_noloads_M16@dec" "$r:x4@dec" \
      "$r:x4=vdpas_vad_noloads_M16@dec"
    for m in 4 8; do
      arms=("$(prod $m _KV8)")
      for t in full nok nov noqk nosoft nopv noout nokcvt novcvt; do arms+=("$(p $t $m _KV8)"); done
      arms+=("$(prod $m "")")
      run "int8_M$m" $B $m $D "${arms[@]}"
    done
    run v2_M1 $B 1 $D attn_v2_M1_T32 attn_v2_M1_T32_KV8
    run flash_split build/tests/flash_split_test --time 32768,98304 --rows 16,64
    echo SESSION_DONE
    ;;
  *)
    echo "usage: $0 compile OUT | gate CARD OUT | session CARD OUT | half CARD OUT" >&2
    exit 2
    ;;
esac
