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
# Env: SNAP_QWEN, SNAP_AGNES, SNAP_ORNITH, SNAP_K2, HF_HOME, TOK_PYTHON, OPENCODE_LOG,
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
    have k2 "K2-Horizon int4 checkpoint ${SNAP_K2:-?} (spec 18b; hf download, 21.8 GB)" \
      complete "${SNAP_K2:-/nonexistent}"
    have oracle_k2 "oracle-out-k2/{prose,code,cjk}.{ids,golden.safetensors} (spec 18a's real-weight run: --with r14.oracle)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2/$p.ids && test -s oracle-out-k2/$p.golden.safetensors || exit 1; done'
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
