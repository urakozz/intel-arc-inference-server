#!/usr/bin/env bash
# Open-work 28 step 2: what a 16-row verify would cost on one card, from its parts (docs/probes/
# probe-lookup-wide-verify-2026-11-14.md). From the tree root, after the build; each GPU command takes its own card
# lock (nothing here builds or polls under one). Run detached; one log per command under OUT.
#   tools/probe/verify16_probe.sh CARD OUT
set -u
card="$1" out="$2"
mkdir -p "$out"
snap=$(tools/box_validate/data.sh resolve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ) || exit 2
run() {   # name, command...
  local name="$1"
  shift
  echo "== $name $(date '+%F %T')"
  tools/gpu_lock.sh card "$card" -- "$@" > "$out/$name.log" 2>&1
  echo "rc=$? ($name)"
}
# The verify lists in place today, M = 1..8 in-list, interleaved pairs against M = 1 (bf16 KV, int8 head).
run mtp_steps_4096 build/tools/probe/probe_mtp_steps "$snap" 4096 32 3 int8 off 16384 8
run mtp_steps_32768 build/tools/probe/probe_mtp_steps "$snap" 32768 32 3 int8 off 65536 8
# The linears: the production gemm_sm rows at 8 and 16 rows against gemv at 1.
run small_m build/tools/probe/probe_small_m prod --m 8,16 --rounds 7
# The GDN step at 1 / 8 / 16 rows: the chain alone and with a snapshot a row.
run gdn_slots build/tools/probe/probe_gdn_slots --calls 16 --rounds 7
# The verify attention at 8 (production) and 16 rows (what builds there), real FA-layer-15 q / KV.
run verify_attn build/tools/probe/probe_verify_attn arms "$HOME/b70/logs/spec28c/vattn-dump" --m 8,16 \
  --depths 4096,32768 --arms K32_Pg_S4_D256_Q1_G256_P1,K32_Pg_S1_D256_Q1_G256_P1,K32_Pg_S1_D128_Q1_G256_P1 \
  --time --rounds 5 --reps 10
echo VERIFY16_PROBE_DONE
