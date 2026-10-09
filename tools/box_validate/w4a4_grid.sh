#!/usr/bin/env bash
# Queue row 35, ON THE BOX CPU: the W4A4 accuracy grid (docs/superpowers/plans/2026-10-09-w4a4-probe.md
# Task 2 Step 1) - tools/rotate/eval_quantised.py sim on the bf16 Qwen3.8 against the int4 g64 gate
# checkpoint, in the oracle container (tools/oracle/run_in_container.sh), over the four prompts h8 was
# accepted on (docs/probe-w4a8-2026-09-23.md §15.4).
#   tools/box_validate/w4a4_grid.sh <data tree> prompts     write the four prompts' ids (no RAM needed)
#   tools/box_validate/w4a4_grid.sh <data tree> grid        the stopping rule's numbers, every prompt:
#       sim <bf16> <gate> --prompt <p> --variants none,w4a16,h8,h4,h4p2 --group 256 --vocab-used 248077
#   tools/box_validate/w4a4_grid.sh <data tree> per-class   Review Focus 5, heldout and long512 only:
#       sim <bf16> <gate> --prompt <p> --variants none --per-class h8,h4,h4p2 --group 256 --vocab-used 248077
#   tools/box_validate/w4a4_grid.sh <data tree> summary     every summary table, and the plan's stopping
#       rule computed from the grid's JSON (the verdict itself is Task 2 Step 3's)
# Run from the tree under test (the container mounts it as /ws). Outputs: <data tree>/oracle-out-w4a4/
# <p>.ids, <p>.<mode>.{log,json}; each run is written to <tree>/oracle-out-w4a4.partial/ and moved there
# once complete, so a killed run resumes at the prompt it was on (box.sh sync leaves oracle-out-* alone).
# The prompts: heldout (478 ids), long512 (tests/golden/prompts/long.ids, first 512), code (61), cjk (38).
# §15.4's 478 held-out ids are NOT in the tree: W4A4_HELDOUT=<file> uses that file if the box still has
# it; otherwise heldout is the first 478 ids of tools/probe/mk_ids.py over docs/05-perf-model.md +
# src/runtime/prefill/context.cc AS IN THIS TREE - not §15's ids (both texts moved since 2026-09-24), so
# read h4 against this run's none / w4a16 / h8 (same load, same ids), not against §15's 12.0 / 12.5 %.
# Exit 77 (SKIP) when MemAvailable < W4A4_MIN_GB (70; `free -g` printed): the bf16 reference and the
# dequantised checkpoint are ~54 GB each, loaded one after the other. Hours (estimated): h8 / h4 rotate
# and re-quantise every weight on every forward; per-class adds 18 forwards a prompt.
# Env: SNAP_QWEN (the gate snapshot dir; tools/box_validate/lib.sh's pinned default), SNAP_QWEN_BF16 (repo
# id or snapshot dir, default Qwen/Qwen3.8-27B), HF_CACHE / HF_HOME, W4A4_HELDOUT, W4A4_MIN_GB,
# W4A4_PROMPTS (a subset, e.g. "cjk code"), ORACLE_IMAGE, ORACLE_THREADS.
set -u
cd "$(dirname "$0")/../.." || exit 2
[ $# -eq 2 ] || { echo "usage: $0 <data tree> prompts|grid|per-class|summary" >&2; exit 2; }
data="$1" mode="$2"
out="$data/oracle-out-w4a4" part="oracle-out-w4a4.partial"
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
export HF_CACHE
gate="${SNAP_QWEN:-$HF_CACHE/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d}"
bf16="${SNAP_QWEN_BF16:-Qwen/Qwen3.8-27B}"
prompts="${W4A4_PROMPTS:-cjk code heldout long512}"
VOCAB_USED=248077   # Qwen3.8's tokenizer (src/model/qwen35.h kVocabUsed): the KL's ids

# the container's path of a host snapshot dir under the HF cache (mounted read-only at /hf)
cpath() {
  case "$1" in
    "$HF_CACHE"/*) printf '/hf%s\n' "${1#"$HF_CACHE"}" ;;
    *) return 1 ;;
  esac
}

make_prompts() {
  mkdir -p "$out"
  local p
  for p in code cjk; do
    [ -s "$out/$p.ids" ] || cp "tests/golden/prompts/$p.ids" "$out/$p.ids"
  done
  if [ ! -s "$out/long512.ids" ]; then
    tr -s ' \n' '\n\n' < tests/golden/prompts/long.ids | grep -v '^$' | head -n 512 | paste -sd' ' - > "$out/long512.ids"
  fi
  if [ ! -s "$out/heldout.ids" ]; then
    if [ -n "${W4A4_HELDOUT:-}" ]; then
      cp "$W4A4_HELDOUT" "$out/heldout.ids" || return 1
      echo "w4a4_grid: heldout = $W4A4_HELDOUT (the operator's file)"
    else
      local g; g=$(cpath "$gate") || { echo "w4a4_grid: SNAP_QWEN=$gate is not under $HF_CACHE"; return 1; }
      mkdir -p "$part"
      echo "w4a4_grid: heldout re-made - mk_ids.py over this tree's docs/05-perf-model.md + src/runtime/prefill/context.cc, first 478 ids (NOT §15.4's file)"
      tools/oracle/run_in_container.sh "python3 tools/probe/mk_ids.py '$g' /ws/$part/mixed2048.ids docs/05-perf-model.md src/runtime/prefill/context.cc" \
        || return 1
      tr -s ' \n' '\n\n' < "$part/mixed2048.ids" | grep -v '^$' | head -n 478 | paste -sd' ' - > "$part/heldout.ids"
      mv -f "$part/heldout.ids" "$out/heldout.ids"
    fi
  fi
  for p in heldout long512 code cjk; do
    printf 'w4a4_grid: %-8s %4s ids  sha256 %s\n' "$p" "$(wc -w < "$out/$p.ids" | tr -d ' ')" \
      "$( (sha256sum "$out/$p.ids" 2>/dev/null || shasum -a 256 "$out/$p.ids") | cut -c1-16)"
  done
}

summary() {
  local f
  for f in "$out"/*.log; do
    [ -f "$f" ] || continue
    echo "=== $(basename "$f")"
    sed -n '/^## summary/,$p' "$f"
  done
  python3 - "$out" <<'EOF'
import glob, json, os, sys
rows = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.grid.json"))):
    js = json.load(open(f))
    rows[os.path.basename(f).split(".")[0]] = {p["variant"]: p["metrics"] for p in js["passes"] if p["only"] is None}
if not rows:
    print("stopping rule: no grid JSON yet"); sys.exit(0)
print("\nstopping rule (plan Task 2): best of h4 / h4p2 more than 2 points of rel L2 above h8, or more than 2 argmax")
print("positions in 100 lost against h8 -> W4A4 closed. Computed here; the verdict is Task 2 Step 3's.")
print("| prompt | h8 rel L2 | h4 | h4p2 | best - h8 (points) | argmax h8 / best | lost per 100 |")
print("|---|---:|---:|---:|---:|---:|---:|")
worst, n, lost = -1e9, 0, 0
for p, m in rows.items():
    if not all(v in m for v in ("h8", "h4", "h4p2")):
        print(f"| {p} | (h8 / h4 / h4p2 missing) |"); continue
    best = min(("h4", "h4p2"), key=lambda v: m[v]["rel"])
    d = (m[best]["rel"] - m["h8"]["rel"]) * 100
    l = m["h8"]["top1"] - m[best]["top1"]
    worst, n, lost = max(worst, d), n + m["h8"]["n"], lost + l
    print(f"| {p} | {m['h8']['rel'] * 100:.2f} % | {m['h4']['rel'] * 100:.2f} % | {m['h4p2']['rel'] * 100:.2f} % | "
          f"{d:+.2f} ({best}) | {m['h8']['top1']} / {m[best]['top1']} | {100 * l / m['h8']['n']:.1f} |")
if n:
    per100 = 100 * lost / n
    closed = worst > 2 or per100 > 2
    print(f"\nworst prompt {worst:+.2f} points; argmax lost {lost} of {n} = {per100:.2f} per 100 -> "
          f"{'CLOSED by the rule' if closed else 'within the rule: Task 2 Step 2 (AutoRound g256) and Task 3 next'}")
EOF
}

case "$mode" in
  prompts) make_prompts; exit $? ;;
  summary) summary; exit 0 ;;
  grid) args="--variants none,w4a16,h8,h4,h4p2 --group 256" set_="$prompts" ;;
  per-class) args="--variants none --per-class h8,h4,h4p2 --group 256" set_="" ;;
  *) echo "usage: $0 <data tree> prompts|grid|per-class|summary" >&2; exit 2 ;;
esac
if [ "$mode" = per-class ]; then
  for p in $prompts; do case "$p" in heldout|long512) set_="$set_ $p" ;; esac; done
fi

free -g 2>/dev/null | head -2
need="${W4A4_MIN_GB:-70}"
avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
echo "w4a4_grid: $mode on [$set_] -> $out; MemAvailable ${avail:-0} GB (need $need)"
if [ "${avail:-0}" -lt "$need" ]; then
  echo "SKIP_REASON w4a4_grid: MemAvailable ${avail:-0} GB < W4A4_MIN_GB $need (another job holds the RAM)"
  exit 77
fi
bf16_dir=$(HF_HOME="$HF_CACHE" tools/box_validate/data.sh resolve "$bf16") || { echo "w4a4_grid: no snapshot for $bf16"; exit 1; }
b=$(cpath "$bf16_dir") || { echo "w4a4_grid: $bf16_dir is not under $HF_CACHE"; exit 1; }
g=$(cpath "$gate") || { echo "w4a4_grid: SNAP_QWEN=$gate is not under $HF_CACHE"; exit 1; }
make_prompts || { echo "w4a4_grid: the prompts FAILED"; exit 1; }
mkdir -p "$part"
rc=0
for p in $set_; do
  if [ -s "$out/$p.$mode.json" ]; then echo "w4a4_grid: $p.$mode kept"; continue; fi
  cp "$out/$p.ids" "$part/$p.ids"
  echo "w4a4_grid: $p.$mode - sim $args ($(date '+%F %T'))"
  if ! tools/oracle/run_in_container.sh \
      "python3 tools/rotate/eval_quantised.py sim '$b' '$g' --prompt /ws/$part/$p.ids $args --vocab-used $VOCAB_USED \
         --out /ws/$part/$p.$mode.json > /ws/$part/$p.$mode.log 2>&1"; then
    echo "w4a4_grid: $p.$mode FAILED"
    tail -20 "$part/$p.$mode.log" 2>/dev/null
    rc=1
    continue
  fi
  mv -f "$part/$p.$mode.log" "$out/$p.$mode.log"
  mv -f "$part/$p.$mode.json" "$out/$p.$mode.json"    # the JSON last: it marks done
  echo "w4a4_grid: $p.$mode done ($(date '+%F %T'))"
done
summary
exit "$rc"
