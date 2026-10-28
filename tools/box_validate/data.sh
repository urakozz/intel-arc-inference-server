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
# Env: SNAP_QWEN, SNAP_QWEN_BF16, SNAP_AGNES, SNAP_ORNITH, SNAP_K2, SNAP_KOLIBRI, SNAP_Q4EXP_{TINY,INTEL,BF16}, SNAP_DFLASH, HF_HOME, TOK_PYTHON, OPENCODE_LOG,
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

# bolt_snap <override> <cache dir name> <pinned revision>: the snapshot directory a Bolt key checks - with SNAP_BOLT /
# SNAP_BOLT_BF16 set (bolt_oracle.sh's spelling: an HF cache directory name) its first snapshot, else the pinned one.
bolt_snap() {
  if [ -n "$1" ]; then ls -d "$hf/hub/$1/snapshots/"*/ 2>/dev/null | head -1; else printf '%s\n' "$hf/hub/$2/snapshots/$3"; fi
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
    have oracle_ornith_f32 "oracle-out-ornith-f32/{prose,code,cjk}.golden.safetensors (the f32 judge, operator ruling 2026-10-23: --with r10.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-ornith-f32/$p.golden.safetensors || exit 1; done'
    have oracle_ornith_f32_i8head "oracle-out-ornith-f32-i8head/{prose,code,cjk}.golden.safetensors (the int8 head's quantisation-aware f32 judge: --with r10.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-ornith-f32-i8head/$p.golden.safetensors || exit 1; done'
    have oracle_ornith_f32_pfi8 "oracle-out-ornith-f32-pfi8/{prose,code,cjk}.golden.safetensors (the l0-int8 prefill's quantisation-aware f32 judge: --with r10.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-ornith-f32-pfi8/$p.golden.safetensors || exit 1; done'
    have k2 "K2-Horizon int4 checkpoint ${SNAP_K2:-?} (spec 18b; hf download, 21.8 GB)" \
      complete "${SNAP_K2:-/nonexistent}"
    have oracle_k2 "oracle-out-k2/{prose,code,cjk}.{ids,golden.safetensors} (spec 18a's real-weight run: --with r14.oracle)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2/$p.ids && test -s oracle-out-k2/$p.golden.safetensors || exit 1; done'
    # Operator ruling 2026-10-23: the Ornith / K2 token gates' JUDGE - the f32 reference teacher-forced on
    # the bf16 golden's tokens (tools/box_validate/f32_oracle.sh <data tree> ornith|k2; tests/golden/golden_common.h).
    have oracle_k2_f32 "oracle-out-k2-f32/{prose,code,cjk}.golden.safetensors (the f32 judge: --with r14.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2-f32/$p.golden.safetensors || exit 1; done'
    have oracle_k2_f32_i8head "oracle-out-k2-f32-i8head/{prose,code,cjk}.golden.safetensors (the int8 head's quantisation-aware f32 judge: --with r14.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2-f32-i8head/$p.golden.safetensors || exit 1; done'
    # Operator ruling 2026-10-29: the kv8 gates' judges (f32_oracle.sh k2-kv8 / k2-i8head-kv8).
    have oracle_k2_f32_kv8 "oracle-out-k2-f32-kv8/{prose,code,cjk}.golden.safetensors (the int8 KV cache's quantisation-aware f32 judge: --with r14.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2-f32-kv8/$p.golden.safetensors || exit 1; done'
    have oracle_k2_f32_i8head_kv8 "oracle-out-k2-f32-i8head-kv8/{prose,code,cjk}.golden.safetensors (int8 head and KV cache: --with r14.oracle_f32)" \
      bash -c 'for p in prose code cjk; do test -s oracle-out-k2-f32-i8head-kv8/$p.golden.safetensors || exit 1; done'
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
    # Spec 21b: the original's small files (config.json + tokenizer files, a few MB - make_synth.py's input), the
    # int8 PLE file beside Intel's snapshot (ple_int8.py - r31.ple_convert writes it).
    have q4exp_orig_small "the original's small files ${SNAP_Q4EXP_BF16:-Qwen/Qwen3.8-Flash-Next}/config.json + tokenizer.json (hf download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja)" \
      bash -c 'd=$(tools/box_validate/data.sh resolve "${SNAP_Q4EXP_BF16:-Qwen/Qwen3.8-Flash-Next}") && test -s "$d/config.json" && test -s "$d/tokenizer.json"'
    have q4exp_intel_ple "the int8 PLE file <Intel snapshot>-ple-int8/ (spec 21b: r31.ple_convert - ple_int8.py, ~51.8 GB)" \
      bash -c 'd=$(tools/box_validate/data.sh resolve "${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}") && test -s "${d%/}-ple-int8/model.safetensors.index.json"'
    # Spec 21b: the synthetic real-width checkpoints (make_synth.py --layers 4 --mtp, both forms) and their PLE
    # int8 files beside them (ple_int8.py) - r31.synth writes them.
    have oracle_q4exp_synth "oracle-out-q4exp-synth/{ours,intel}/ckpt/ and ckpt-ple-int8/ (spec 21b: r31.synth - make_synth.py + ple_int8.py)" \
      bash -c 'for f in ours intel; do test -s oracle-out-q4exp-synth/$f/ckpt/model.safetensors.index.json && test -s oracle-out-q4exp-synth/$f/ckpt-ple-int8/model.safetensors.index.json || exit 1; done'
    # Spec 21c: their golden sets (qwen4exp_oracle.sh synth: qwen4exp_ref.py run on each, the engine-format PLE) -
    # a key of its own, so r31's stages (the checkpoints alone) do not wait for them.
    have oracle_q4exp_synth_golden "oracle-out-q4exp-synth/{ours,intel}/q4exp_{short,4k,agentic}.{ids,golden.safetensors} (spec 21c: r32.oracle_synth - qwen4exp_oracle.sh synth)" \
      bash -c 'for f in ours intel; do for p in q4exp_short q4exp_4k q4exp_agentic; do test -s oracle-out-q4exp-synth/$f/$p.ids && test -s oracle-out-q4exp-synth/$f/$p.golden.safetensors || exit 1; done; done'
    have k2_a4_set "oracle-out-k2-a4/set/manifest.json (K2's tool-call set, spec 18d: --with r25.a4_ref, or tools/toolcall/a4_ref.sh k2 set)" \
      test -s oracle-out-k2-a4/set/manifest.json
    have oracle_k2_a4 "oracle-out-k2-a4/ with K2's set and 36 <name>.bf16.txt (18a's reference on it: --with r25.a4_ref)" \
      bash -c 'test -s oracle-out-k2-a4/set/manifest.json && test "$(ls oracle-out-k2-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_kolibri_a4 "oracle-out-kolibri-a4/ with 36 <name>.bf16.txt (spec 20e KL4: the bf16 source's reference on tests/golden/toolcall-kolibri-de, made wherever 20b runs - tools/toolcall/a4_ref.sh kolibri ref - then --push-data)" \
      bash -c 'test "$(ls oracle-out-kolibri-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_ornith_a4 "oracle-out-ornith-a4/ with Ornith's set and 36 <name>.bf16.txt (spec 15e Task 3: tools/toolcall/a4_ref.sh ornith set / ref on the Mac, then --push-data)" \
      bash -c 'test -s oracle-out-ornith-a4/set/manifest.json && test "$(ls oracle-out-ornith-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    have oracle_q4exp_a4 "oracle-out-q4exp-a4/ with Qwen3.8-Flash-Next's set and 36 <name>.bf16.txt (spec 21e Task 5: tools/toolcall/a4_ref.sh qwen4exp set / ref on Intel's checkpoint - --with r34.a4_ref; spec 22's engine run scores against it)" \
      bash -c 'test -s oracle-out-q4exp-a4/set/manifest.json && test "$(ls oracle-out-q4exp-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36'
    # Queue row 35 (the W4A4 probe, plan 2026-10-09-w4a4-probe Task 2): the bf16 reference the grid's
    # logits are compared against (tools/box_validate/w4a4_grid.sh).
    have qwen_bf16 "the bf16 original ${SNAP_QWEN_BF16:-Qwen/Qwen3.8-27B} (54 GB; row 35's W4A4 grid reference)" \
      complete "${SNAP_QWEN_BF16:-Qwen/Qwen3.8-27B}"
    # Queue row 36 (spec 22 P0, plan 22a): the masked reference's KL / argmax files (r36.prune: qwen4exp_oracle.sh
    # prune) and the probe binary (r36.build; listed for the record - the r36 stages order on r36.build itself).
    have oracle_q4exp_prune "oracle-out-q4exp-prune/maskeval/*.json (spec 22 P0.8: r36.prune - qwen4exp_oracle.sh prune)" \
      bash -c 'ls oracle-out-q4exp-prune/maskeval/*.json'
    have probe_offload_bin "build/tools/probe/probe_offload (spec 22 P0: r36.build - the probe and its two kernels)" \
      test -x build/tools/probe/probe_offload
    # Queue row 54 (spec 19b): the DFlash drafter (3.85 GB; hf download z-lab/Qwen3.8-27B-DFlash2) and 19a's tap dumps
    # (the Mac's oracle-out-19a/dumps, --push-data) that D1 drafts from on both sides.
    have dflash "the DFlash drafter ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2} (3.85 GB)" \
      complete "${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
    have oracle_dflash_dumps "oracle-out-19a/dumps/*.taps.safetensors (spec 19a's tap dumps, 2.7 GB: --push-data from the Mac)" \
      bash -c 'ls oracle-out-19a/dumps/*.taps.safetensors'
    # Queue row 56 (spec 19d): 19a's ranked draft vocabulary (tools/draft_vocab/rank.py over A4 + golden, in-sample;
    # --push-data from the Mac) for r56.d5_dv's ranked 32k arm.
    have oracle_dflash_rank "oracle-out-19a/draft_vocab.ranked.ids (spec 19a's ranking, in-sample: --push-data from the Mac)" \
      test -s oracle-out-19a/draft_vocab.ranked.ids
    have oracle_image "the oracle container image (tools/oracle/run_in_container.sh; passkey ids)" \
      docker image inspect "${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest}"
    # Queue row 41 (the llama-benchy comparison matrix): the vLLM XPU image tools/probe/vllm_benchy.sh serves.
    have vllm_image "the vLLM XPU image ${VLLM_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest} (tools/probe/vllm_benchy.sh)" \
      docker image inspect "${VLLM_IMAGE:-vllm-xpu-env-next-p314-t216-vxkp0:latest}"
    # Queue row 57 (spec 25a, plan 25a): Nemotron 3.5 Bolt - our int4 g64 export (the served checkpoint, the only one),
    # the BF16 original (gate Q's reference), the pinned small files (the tiny tests' remote code) and the CPU reference's
    # outputs (tools/box_validate/bolt_oracle.sh). Download pinned: --revision 8932aa8d... / 56702103... (df -h ~ first);
    # a revision-pinned download writes no refs/main, so the keys name the snapshot directories themselves. The int4
    # references are ours only when their directory's SNAPSHOT file (bolt_oracle.sh writes it) names this snapshot: the
    # 2026-10-22/23 ones were made from a third-party g128 export and read as missing.
    bolt_ours="models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ/snapshots/8932aa8d1844f91056c5b9433448075a842b9156"
    have bolt_small "Bolt's pinned small files (hf download urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ --revision 8932aa8d1844f91056c5b9433448075a842b9156 --exclude '*.safetensors')" \
      test -s "$hf/hub/$bolt_ours/modeling_nemotron_h.py"
    have bolt_int4 "our Bolt int4 g64 export ${SNAP_BOLT:-models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ} @ 8932aa8d (41 GB; df -h ~ first)" \
      complete "$(bolt_snap "${SNAP_BOLT:-}" models--urakozz--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ 8932aa8d1844f91056c5b9433448075a842b9156)"
    have bolt_bf16 "NVIDIA's Bolt BF16 original ${SNAP_BOLT_BF16:-models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16} @ 56702103 (149.8 GB; gate Q's reference)" \
      complete "$(bolt_snap "${SNAP_BOLT_BF16:-}" models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 567021037ff88364143e24ea05e6abd0f72086ff)"
    have oracle_bolt "oracle-out-bolt/bolt_{short,4k,agentic,32k}.{ids,golden.safetensors} + SNAPSHOT = our export's (spec 25a: bolt_oracle.sh golden - the engine-format reference)" \
      bash -c 'test "$(cat oracle-out-bolt/SNAPSHOT 2>/dev/null)" = "$1" || exit 1; for p in bolt_short bolt_4k bolt_agentic bolt_32k; do test -s oracle-out-bolt/$p.ids && test -s oracle-out-bolt/$p.golden.safetensors || exit 1; done' _ "$bolt_ours"
    have oracle_bolt_bf16 "oracle-out-bolt-bf16/bolt_{short,agentic}.{ids,golden.safetensors} (spec 25a: bolt_oracle.sh golden-bf16)" \
      bash -c 'for p in bolt_short bolt_agentic; do test -s oracle-out-bolt-bf16/$p.ids && test -s oracle-out-bolt-bf16/$p.golden.safetensors || exit 1; done'
    have bolt_a4_int4 "oracle-out-bolt-a4/ with Bolt's set, 36 <name>.int4.txt and SNAPSHOT = our export's (spec 25a: bolt_oracle.sh a4 - the int4 reference alone, no BF16 needed)" \
      bash -c 'test "$(cat oracle-out-bolt-a4/SNAPSHOT 2>/dev/null)" = "$1" && test -s oracle-out-bolt-a4/set/manifest.json && test "$(ls oracle-out-bolt-a4/*.int4.txt 2>/dev/null | wc -l)" -eq 36' _ "$bolt_ours"
    have bolt_a4_ref "oracle-out-bolt-a4/ with Bolt's set, 36 <name>.int4.txt, 36 <name>.bf16.txt and SNAPSHOT = our export's (spec 25a: bolt_oracle.sh a4 + a4-bf16)" \
      bash -c 'test "$(cat oracle-out-bolt-a4/SNAPSHOT 2>/dev/null)" = "$1" && test -s oracle-out-bolt-a4/set/manifest.json && test "$(ls oracle-out-bolt-a4/*.int4.txt 2>/dev/null | wc -l)" -eq 36 && test "$(ls oracle-out-bolt-a4/*.bf16.txt 2>/dev/null | wc -l)" -eq 36' _ "$bolt_ours"
    # Spec 25b (queue row 59): the synthetic real-width checkpoints (bolt_synth.sh synth-ckpt: N 8 for the host tests, N 12
    # for the card tests - each in the GPTQ g128 spelling, its exact g64 expansion and a native g64; our export's small
    # files).
    have bolt_synth "oracle-out-bolt-synth/n{8,12}-{g128,g64x,g64}/ + .ACCEPTED (spec 25b: r59.synth - make_synth.py)" \
      bash -c 'for n in 8 12; do for f in g128 g64x g64; do test -s oracle-out-bolt-synth/n$n-$f/model.safetensors.index.json && test -f oracle-out-bolt-synth/n$n-$f.ACCEPTED || exit 1; done; done'
    # Spec 25c (queue row 60): the N 12 synthetics' golden sets (r60.synth_golden: bolt_oracle.sh export on each).
    have oracle_bolt_synth "oracle-out-bolt-n12-{g128,g64x}/bolt_{short,4k,agentic}.golden.safetensors (spec 25c: r60.synth_golden)" \
      bash -c 'for s in n12-g128 n12-g64x; do for p in bolt_short bolt_4k bolt_agentic; do test -s oracle-out-bolt-$s/$p.golden.safetensors || exit 1; done; done'
    have tok_python "a python with tokenizers (${TOK_PYTHON:-?})" "${TOK_PYTHON:-/nonexistent}" -c 'import tokenizers'
    # shellcheck disable=SC2016 # expanded by the inner bash
    have uvx "uvx (llama-benchy)" bash -c 'command -v uvx || test -x "$HOME/.local/bin/uvx"'
    have xpu_smi "xpu-smi (device memory while serving)" bash -c 'command -v xpu-smi'
    have opencode_log "OPENCODE_LOG=${OPENCODE_LOG:-<unset>} (a request log dir)" test -d "${OPENCODE_LOG:-/nonexistent}"
    have a4_ref "A4_REF_DIR=${A4_REF_DIR:-<unset>} (<name>.bf16.txt reference runs)" test -d "${A4_REF_DIR:-/nonexistent}"
    ;;
  resolve) resolve "$2" ;;
  complete) complete "$2" ;;      # exit 0 when the snapshot is complete (spec 23a: either int4 spelling)
  *) echo "usage: $0 link <data tree> <tree>... | have | resolve <id|dir> | complete <id|dir>" >&2; exit 2 ;;
esac
