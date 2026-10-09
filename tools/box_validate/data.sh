#!/usr/bin/env bash
# What a validation run needs that is not in git, ON THE BOX.
#   tools/box_validate/data.sh link <data tree> <tree>...
#       symlink every oracle-out* directory of <data tree> (the tree the golden sets were
#       dumped in, normally ~/b70-inference-server) into each <tree> that lacks one of that
#       name. The golden registrations read ${CMAKE_SOURCE_DIR}/oracle-out-*; box.sh's sync
#       excludes those names, so the links survive later syncs.
#   tools/box_validate/data.sh have
#       one line per data key, HAVE_<key>=1|0 plus a comment saying what was looked for.
#       The stages' NEEDS name these keys (tools/box_validate/stages.sh).
#   tools/box_validate/data.sh resolve <repo id | snapshot dir>
#       the snapshot directory the CLIs would load (refs/main for a repo id).
# Env: SNAP_QWEN, SNAP_AGNES, SNAP_ORNITH, SNAP_K2, SNAP_KOLIBRI, SNAP_Q4EXP_{TINY,INTEL,BF16}, HF_HOME, TOK_PYTHON, OPENCODE_LOG,
#      A4_REF_DIR, ORACLE_IMAGE. Run from a tree root (the oracle-out* checks look there).
set -u
hf="${HF_HOME:-$HOME/.cache/huggingface}"

resolve() {
  if [ -d "$1" ]; then printf '%s\n' "$1"; return 0; fi
  local repo
  repo="$hf/hub/models--$(printf '%s' "$1" | sed 's|/|--|g')"
  [ -r "$repo/refs/main" ] || { printf '%s\n' "$repo/snapshots/<no refs/main>"; return 1; }
  printf '%s\n' "$repo/snapshots/$(cat "$repo/refs/main")"
}

# A snapshot is complete when config.json is there and every shard the index names exists
# (a download interrupted part-way leaves the index and some shards).
complete() {
  local d; d=$(resolve "$1") || return 1
  [ -f "$d/config.json" ] || return 1
  python3 - "$d" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx):
    files = set(json.load(open(idx))["weight_map"].values())
else:
    files = {f for f in os.listdir(d) if f.endswith(".safetensors")} or {"model.safetensors"}
missing = [f for f in sorted(files) if not os.path.exists(os.path.join(d, f))]
sys.exit(1 if missing else 0)
EOF
}

have() {   # have KEY WHAT CMD...
  local key="$1" what="$2"; shift 2
  if "$@" > /dev/null 2>&1; then echo "HAVE_$key=1  # $what"; else echo "HAVE_$key=0  # missing: $what"; fi
}

case "${1:-}" in
  link)
    shift
    data="$1"; shift
    for t in "$@"; do
      [ -d "$t" ] || { echo "data.sh: no tree $t" >&2; continue; }
      [ "$(cd "$t" && pwd -P)" = "$(cd "$data" 2>/dev/null && pwd -P)" ] && continue
      for d in "$data"/oracle-out*; do
        [ -d "$d" ] || continue
        b=$(basename "$d")
        if [ -e "$t/$b" ]; then echo "kept   $t/$b"; else ln -s "$d" "$t/$b" && echo "linked $t/$b -> $d"; fi
      done
    done
    ;;
  have)
    have qwen "Qwen3.8 gate checkpoint ${SNAP_QWEN:-?}" complete "${SNAP_QWEN:-/nonexistent}"
    have oracle_qwen "oracle-out-primary/ (Qwen3.8 golden set)" test -d oracle-out-primary
    have agnes "Agnes checkpoint ${SNAP_AGNES:-?} (spec 14 checklist step 1: rsync from the Mac)" \
      complete "${SNAP_AGNES:-/nonexistent}"
    have oracle_agnes "oracle-out-agnes/ (rsync from the Mac: spec 14 checklist step 6)" test -d oracle-out-agnes
    have oracle_agnes_mtp "oracle-out-agnes-mtp/ (rsync from the Mac)" test -d oracle-out-agnes-mtp
    have ornith "Ornith int4 checkpoint ${SNAP_ORNITH:-?} (spec 15a)" complete "${SNAP_ORNITH:-/nonexistent}"
    have oracle_ornith "oracle-out-ornith/ with router_logits.L* (spec 15a)" test -d oracle-out-ornith
    have oracle_ornith_mtp "oracle-out-ornith-mtp/ (spec 15a's MTP head dump: mtp_head_ornith_test, M1)" \
      test -d oracle-out-ornith-mtp
    have k2 "K2-Horizon int4 checkpoint ${SNAP_K2:-?} (spec 18b; hf download, 21.8 GB)" \
      complete "${SNAP_K2:-/nonexistent}"
    have oracle_k2 "oracle-out-k2/{prose,code,cjk}.{ids,golden.safetensors} (spec 18a's real-weight run: --with r14.oracle)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2/$p.ids && test -s oracle-out-k2/$p.golden.safetensors || exit 1; done'
    have kolibri "Kolibri-1 int4 checkpoint ${SNAP_KOLIBRI:-?} (spec 20b, decision 1 open; ~42.5 GB)" \
      complete "${SNAP_KOLIBRI:-/nonexistent}"
    # Spec 20c: Kolibri-1's golden sets (tools/box_validate/kolibri_oracle.sh <data tree> synth|real).
    have oracle_kolibri "oracle-out-kolibri/{prose,code,de_prose,de_chat}.{ids,golden.safetensors} (spec 20c, needs 20b's checkpoint: kolibri_oracle.sh real)" \
      bash -c 'for p in prose code de_prose de_chat; do test -s oracle-out-kolibri/$p.ids && test -s oracle-out-kolibri/$p.golden.safetensors || exit 1; done'
    have oracle_kolibri_synth "oracle-out-kolibri-synth/{int4attn,bf16attn}/{ckpt/,prose,de_prose} (spec 20c: kolibri_oracle.sh synth)" \
      bash -c 'for a in int4attn bf16attn; do test -s oracle-out-kolibri-synth/$a/ckpt/model.safetensors.index.json || exit 1; for p in prose de_prose; do test -s oracle-out-kolibri-synth/$a/$p.golden.safetensors || exit 1; done; done'
    # Spec 21a: Qwen3.8-Flash-Next (qwen4_exp) - the checkpoints and the CPU reference's golden sets
    # (tools/box_validate/qwen4exp_oracle.sh <data tree> intel | intel-layers N | trace).
    have q4exp_tiny "the tiny qwen4_exp model (hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next, 124 MB)" \
      complete "${SNAP_Q4EXP_TINY:-qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next}"
    have q4exp_intel "Intel's Qwen3.8-Flash-Next W4A16 AutoRound checkpoint ${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound} (181 GB with the 102.4 GB PLE shards; df -h first)" \
      complete "${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}"
    have q4exp_bf16 "the bf16 original ${SNAP_Q4EXP_BF16:-Qwen/Qwen3.8-Flash-Next} (360 GB; its 128 PLE shards alone are what the bf16-PLE KL needs)" \
      complete "${SNAP_Q4EXP_BF16:-Qwen/Qwen3.8-Flash-Next}"
    have oracle_q4exp "oracle-out-q4exp/q4exp_{short,4k,8k,32k,agentic}.{ids,golden.safetensors} (spec 21a: qwen4exp_oracle.sh intel)" \
      bash -c 'for p in q4exp_short q4exp_4k q4exp_8k q4exp_32k q4exp_agentic; do test -s oracle-out-q4exp/$p.ids && test -s oracle-out-q4exp/$p.golden.safetensors || exit 1; done'
    have oracle_q4exp_layers "oracle-out-q4exp-L4/ and -L18/ q4exp_{short,4k,agentic} (spec 21a: qwen4exp_oracle.sh intel-layers 4 / 18 - 21c's --layers N gate)" \
      bash -c 'for n in 4 18; do for p in q4exp_short q4exp_4k q4exp_agentic; do test -s oracle-out-q4exp-L$n/$p.golden.safetensors || exit 1; done; done'
    have oracle_q4exp_traces "oracle-out-q4exp-traces/*.routes.safetensors (spec 21a Task 7: qwen4exp_oracle.sh trace - spec 22 P0.6 / P0.8's input)" \
      bash -c 'test "$(ls oracle-out-q4exp-traces/*.routes.safetensors 2>/dev/null | wc -l)" -ge 39'
    have k2_a4_set "oracle-out-k2-a4/set/manifest.json (K2's tool-call set, spec 18d: --with r25.a4_ref, or tools/toolcall/a4_ref.sh k2 set)" \
      test -s oracle-out-k2-a4/set/manifest.json
    have oracle_k2_a4 "oracle-out-k2-a4/ with K2's set and 36 <name>.bf16.txt (18a's reference on it: --with r25.a4_ref)" \
      bash -c 'test -s oracle-out-k2-a4/set/manifest.json && test "$(ls oracle-out-k2-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_kolibri_a4 "oracle-out-kolibri-a4/ with 36 <name>.bf16.txt (spec 20e KL4: the bf16 source's reference on tests/golden/toolcall-kolibri-de, made wherever 20b runs - tools/toolcall/a4_ref.sh kolibri ref - then --push-data)" \
      bash -c 'test "$(ls oracle-out-kolibri-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_ornith_a4 "oracle-out-ornith-a4/ with Ornith's set and 36 <name>.bf16.txt (spec 15e Task 3: tools/toolcall/a4_ref.sh ornith set / ref on the Mac, then --push-data)" \
      bash -c 'test -s oracle-out-ornith-a4/set/manifest.json && test "$(ls oracle-out-ornith-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_image "the oracle container image (tools/oracle/run_in_container.sh; passkey ids)" \
      docker image inspect "${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}"
    have tok_python "a python with tokenizers (${TOK_PYTHON:-?})" "${TOK_PYTHON:-/nonexistent}" -c 'import tokenizers'
    # shellcheck disable=SC2016 # expanded by the inner bash
    have uvx "uvx (llama-benchy)" bash -c 'command -v uvx || test -x "$HOME/.local/bin/uvx"'
    have xpu_smi "xpu-smi (device memory while serving)" bash -c 'command -v xpu-smi'
    have opencode_log "OPENCODE_LOG=${OPENCODE_LOG:-<unset>} (a request log dir)" test -d "${OPENCODE_LOG:-/nonexistent}"
    have a4_ref "A4_REF_DIR=${A4_REF_DIR:-<unset>} (<name>.bf16.txt reference runs)" test -d "${A4_REF_DIR:-/nonexistent}"
    ;;
  resolve) resolve "$2" ;;
  *) echo "usage: $0 link <data tree> <tree>... | have | resolve <id|dir>" >&2; exit 2 ;;
esac
