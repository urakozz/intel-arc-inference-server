#!/usr/bin/env bash
# Spec 32: the timings the deep verify's stop rules read, from the tree root after the build; each GPU command takes
# its own card lock. Run detached; one log per command under OUT.
#   tools/probe/deep_verify_probe.sh CARD OUT
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
# The built 16-row list in place against the plain step (interleaved pairs), bf16 KV, int8 head.
run wide16_4096 build/tools/probe/probe_mtp_steps "$snap" 4096 16 5 int8 off 16384 4 wide16
run wide16_32768 build/tools/probe/probe_mtp_steps "$snap" 32768 16 5 int8 off 65536 4 wide16
# The linears at 16 / 32 / 64 rows, both GRF modes at 32 / 64 (property R against 16 rows and N2 first).
run small_m build/tools/probe/probe_small_m prod --m 16,32,64 --rounds 7
echo DEEP_VERIFY_PROBE_DONE
