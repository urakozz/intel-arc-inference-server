#!/usr/bin/env bash
# The box side of tools/box_validate.sh: run the selected stages in order, ON THE BOX, from
# the tree under test. The driver starts it detached (tools/probe/detach.sh), so it outlives
# the ssh session and the Mac; the driver only polls.
#   tools/box_validate/remote.sh <state dir> <stage id>...
# Resumable: a stage already PASS in <state dir> is kept unless BV_REDO names it; a stage
# left RUNNING by a dead orchestrator (reboot, kill) runs again. One orchestrator per tree
# (<state dir>/../.run.lock). GPU stages hold ~/b70-gpu.lock for their whole body.
# Stop-the-line: while any G0 stage is not PASS, every later stage is recorded SKIP
# ("blocked") unless BV_FORCE=1.
#
# Per stage, in <state dir>:
#   <id>.status   id, result (RUNNING / PASS / FAIL / SKIP), rc, start, end, seconds, reason
#                 - one tab-separated line, the machine-readable verdict
#   <id>.log      everything the stage printed (an earlier attempt is kept as <id>.log.prev)
#   <id>.junit.xml, <id>.baseline.xml, <id>.rows - ctest results and timed rows
# and progress.log (one line per event), meta.env (shas, trees, times).
# Env (from the driver): BV_TREE_LABEL, BV_BASE, BV_DATA, BV_REDO ("id ..." | all),
#   BV_FORCE, B70_GIT_SHA, BV_BASE_SHA, BV_BASE_REF, JOBS, CMAKE_ARGS, DEVICE, PORT, SNAP_*.
set -u
cd "$(dirname "$0")/../.." || exit 2
TREE="$PWD"
STATE="${1:?usage: $0 <state dir> <stage id>...}"
shift
mkdir -p "$STATE"
HOME_R="$HOME"
HF_HOME_R="${HF_HOME:-}"
# shellcheck source=tools/box_validate/lib.sh
. tools/box_validate/lib.sh
# shellcheck source=tools/box_validate/stages.sh
. "${BV_STAGES_FILE:-tools/box_validate/stages.sh}"   # BV_STAGES_FILE: test_box_validate.py's stub
bv_defaults
BASE="${BV_BASE:?BV_BASE (the baseline tree) is required}"
DATA="${BV_DATA:-$HOME/b70-inference-server}"
DRY=0

exec 8> "$(dirname "$STATE")/.run.lock"
if ! flock -n 8; then
  echo "remote.sh: another validation run is active in this tree: $(cat "$(dirname "$STATE")/.current" 2>/dev/null)"
  exit 3
fi
basename "$STATE" > "$(dirname "$STATE")/.current"

# A run validates the defaults: nothing from the operator's shell may switch a code path.
unset B70_KV_CACHE B70_DECODE_ATTN B70_PREFILL_ATTN B70_PREFILL_GDN_SCAN B70_PREFILL_GDN_SOLVE \
      B70_PREFILL_SILU_FUSED B70_PREFILL_REPLAY B70_PREFILL_PROFILE B70_TOKENIZER_JSON \
      ONEAPI_DEVICE_SELECTOR
export ZE_AFFINITY_MASK="$DEVICE"
export HOME_R BV_DATA="$DATA" OPENCODE_LOG="${OPENCODE_LOG:-}" A4_REF_DIR="${A4_REF_DIR:-}"

ts() { date '+%F %T'; }
progress() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$STATE/progress.log"; }

{
  echo "tree_sha=$B70_GIT_SHA"
  echo "baseline_sha=${BV_BASE_SHA:-?}"
  echo "baseline_ref=${BV_BASE_REF:-?}"
  echo "state=$(basename "$STATE")"
  echo "tree=$TREE"
  echo "baseline_tree=$BASE"
  echo "started=$(ts)"
  echo "selected=$*"
  echo "device=$DEVICE"
} > "$STATE/meta.env"

is_redo() { case " ${BV_REDO:-} " in *" all "*|*" $1 "*) return 0 ;; esac; return 1; }

write_status() {   # id result rc start end secs reason
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" > "$STATE/$1.status"
}

g0_block() {   # prints the G0 stages that are not PASS
  local i out=""
  for i in "${!BV_IDS[@]}"; do
    if [ "${BV_SELS[$i]}" = g0 ]; then
      local s; s=$(status_of "${BV_IDS[$i]}")
      [ "$s" = PASS ] || out="$out ${BV_IDS[$i]}=${s:-not-run}"
    fi
  done
  printf '%s' "$out"
}

run_stage() {
  local id="$1" sel kind needs after title fn prev start t0 rc result reason k missing="" d blocked="" s
  if ! sel=$(bv_field "$id" sel); then progress "$id UNKNOWN (not in stages.sh)"; return; fi
  kind=$(bv_field "$id" kind); needs=$(bv_field "$id" needs)
  after=$(bv_field "$id" after); title=$(bv_field "$id" title)
  fn=$(bv_fn "$id")
  prev=$(status_of "$id")
  start=$(ts)
  if [ "$sel" = manual ]; then progress "$id MANUAL (commands only; never run here)"; return; fi
  if [ "$prev" = PASS ] && [ "$sel" != always ] && ! is_redo "$id"; then
    progress "$id PASS (kept from an earlier launch; --redo $id runs it again)"; return
  fi
  if [ "$sel" != g0 ] && [ "$sel" != always ] && [ "${BV_FORCE:-0}" != 1 ]; then
    blocked=$(g0_block)
    if [ -n "$blocked" ]; then
      write_status "$id" SKIP 77 "$start" "$start" 0 "blocked: G0 not passed ($blocked ) - --force runs past it"
      progress "$id SKIP blocked by G0:$blocked"; return
    fi
  fi
  for k in $(printf '%s' "$needs" | tr ',' ' '); do
    [ "$k" = - ] && continue
    grep -q "^HAVE_$k=1" "$STATE/have.env" 2>/dev/null || missing="$missing $k"
  done
  if [ -n "$missing" ]; then
    write_status "$id" SKIP 77 "$start" "$start" 0 "missing data:$missing (pre.log lists what was looked for)"
    progress "$id SKIP missing data:$missing"; return
  fi
  for d in $(printf '%s' "$after" | tr ',' ' '); do
    [ "$d" = - ] && continue
    s=$(status_of "$d")
    [ "$s" = PASS ] || blocked="$blocked $d=${s:-not-run}"
  done
  if [ -n "$blocked" ]; then
    write_status "$id" SKIP 77 "$start" "$start" 0 "needs PASS first:$blocked"
    progress "$id SKIP needs:$blocked"; return
  fi

  # an earlier attempt's files (log, JUnit, rows, refusal outputs) are kept as *.prev
  for k in "$STATE/$id".*; do
    case "$k" in *.status|*.prev) continue ;; esac
    if [ -f "$k" ]; then mv -f "$k" "$k.prev"; fi
  done
  write_status "$id" RUNNING "" "$start" "" "" ""
  progress "$id START ($kind) $title"
  t0=$(date +%s)
  (
    set +e
    exec 8>&-   # the run lock stays with the orchestrator, not with what a stage starts
    STAGE="$id"; LOG="$STATE/$id.log"; BV_BAD=0; BV_OK=0; BV_SKIPS=0
    echo "=== stage $id: $title"
    echo "=== tree $TREE (sha $B70_GIT_SHA), baseline $BASE (${BV_BASE_SHA:-?}), device $ZE_AFFINITY_MASK, $(ts)"
    if [ "$kind" = gpu ]; then
      exec 9> "$HOME/b70-gpu.lock"
      echo "waiting for the GPU lock ($(ts))"
      flock 9
      echo "GPU lock held ($(ts)); $(uptime)"
    fi
    "$fn"
    frc=$?
    echo "=== end of stage $id ($(ts)): ok $BV_OK, failed $BV_BAD, skipped $BV_SKIPS, body rc $frc"
    if [ "$BV_BAD" -gt 0 ]; then exit 1; fi
    if [ "$frc" -eq 77 ]; then exit 77; fi
    if [ "$frc" -ne 0 ]; then echo "BAD the stage body returned $frc"; exit 1; fi
    if [ "$BV_OK" -eq 0 ] && [ "$BV_SKIPS" -gt 0 ]; then exit 77; fi
    exit 0
  ) > "$STATE/$id.log" 2>&1
  rc=$?
  case "$rc" in
    0) result=PASS; reason="" ;;
    77) result=SKIP; reason=$(grep -E '^(SKIP_REASON|SKIPPED) ' "$STATE/$id.log" | head -3 | sed 's/^[A-Z_]* //' | paste -sd ';' -) ;;
    *) result=FAIL; reason=$(grep -E '^(BAD|FAILED) ' "$STATE/$id.log" | head -4 | sed 's/^[A-Z]* //' | paste -sd ';' -) ;;
  esac
  write_status "$id" "$result" "$rc" "$start" "$(ts)" "$(( $(date +%s) - t0 ))" "${reason:-}"
  progress "$id $result${reason:+ - $reason}"
}

progress "orchestrator start: $(basename "$STATE"), stages: $*"
for id in "$@"; do run_stage "$id"; done
echo "finished=$(ts)" >> "$STATE/meta.env"
progress "orchestrator done"
