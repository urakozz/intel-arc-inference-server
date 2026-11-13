#!/usr/bin/env bash
# On the box, from the branch tree's root: a short turn deep in a served session (3l's short-chunk path, a
# prefix cache hit) on main's b70-serve against this tree's, one card, the server's own chunk cap
# (record: docs/probes/probe-qwen-pf-gemm-2026-11-13.md, section 6). llama-benchy at --depth D with the
# prefix cache: the timed pp is the short turn after the cached context.
# Usage: qwen_chunk4096_turn.sh CARD MAIN_TREE ROUNDS [DEPTH] [PP] ["SERVER ARGS"]
# Takes `tools/gpu_lock.sh card CARD` itself, once for the whole series.
set -u
card=${1:?card}
main=${2:?main tree}
rounds=${3:-3}
depth=${4:-32768}
pp=${5:-64}
extra=${6:-}
out="$HOME/b70/logs/qwen-chunk-4096"
mkdir -p "$out"
snap=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
export PATH="$HOME/.local/bin:$PATH"
tools/gpu_lock.sh card "$card" -- bash -c "
  one() {   # tree tag
    \$1/build/src/cli/b70-serve $snap --port 8017 --parallel 1 --spec off $extra > $out/turn-serve-\$2.log 2>&1 &
    srv=\$!
    for _ in \$(seq 1 600); do
      curl -sf http://127.0.0.1:8017/v1/models > /dev/null && break
      kill -0 \$srv 2>/dev/null || { echo server exited; tail -5 $out/turn-serve-\$2.log; return; }
      sleep 1
    done
    grep -E 'prefill chunk|max_len: auto' $out/turn-serve-\$2.log
    uvx llama-benchy --base-url http://127.0.0.1:8017/v1 --model $snap --served-model-name b70 \
      --tokenizer $snap --pp $pp --tg 16 --depth $depth --enable-prefix-caching --runs 3 > $out/turn-\$2.log 2>&1
    echo \"\$(date '+%T') \$2:\"; grep -E 'pp$pp|ctx_pp|tg16' $out/turn-\$2.log
    kill \$srv; wait \$srv 2>/dev/null
  }
  for r in \$(seq 1 $rounds); do
    one $main r\$r-main
    one \$PWD r\$r-branch
  done
"
echo "done $(date '+%F %T')"
