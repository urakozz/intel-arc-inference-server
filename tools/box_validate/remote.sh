#!/usr/bin/env bash
# The box side of tools/box_validate.sh: run the selected stages in order, ON THE BOX, from
# the tree under test. The driver starts it detached (tools/probe/detach.sh), so it outlives
# the ssh session and the Mac; the driver only polls.
#   tools/box_validate/remote.sh <state dir> <stage id>...
# Resumable: a stage already PASS in <state dir> is kept unless BV_REDO names it; a stage
# left RUNNING by a dead orchestrator (reboot, kill) runs again.
# One orchestrator per tree AND DEVICE (<state dir>/../.run-<DEVICE>.lock): a device-0 and a
# device-1 orchestrator can run side by side on one tree and one state - each stage is claimed
# (<state dir>/<id>.claim, flock -n) for as long as it runs, so the two never run the same
# stage, and a RUNNING stage whose claim is free is one a dead orchestrator left. The locks a
# stage takes (tools/gpu_lock.sh has the protocol):
#   gpu       the whole box (tools/gpu_lock.sh box) for the whole body
#   card      card $DEVICE only (tools/gpu_lock.sh card $DEVICE) - beside a stage on the other card
#   gpu-self  none here: its script takes the box itself
#   cpu       <state dir>/../.cpu.lock (cpu stages of two orchestrators run one at a time: the
#             CPU oracles' RAM, the host tests' fixed /tmp directories); not the preflight
# and a stage that builds (cmake / ocloc in its body) holds <state dir>/../.build.lock first.
# G0 runs alone: a G0 stage is DEFERRED (no status written) while another orchestrator is
# active on this tree. Stop-the-line: while any G0 stage is not PASS, every later stage is
# recorded SKIP ("blocked") unless BV_FORCE=1.
#
# Per stage, in <state dir>:
#   <id>.status   id, result (RUNNING / PASS / FAIL / SKIP), rc, start, end, seconds, reason,
#                 device, lock (box / card / self / cpu / none) - one tab-separated line, the
#                 machine-readable verdict (written whole: a temporary file renamed over it)
#   <id>.log      everything the stage printed (an earlier attempt is kept as <id>.log.prev)
#   <id>.junit.xml, <id>.baseline.xml, <id>.rows - ctest results and timed rows
# and progress.log (one line per event, tagged [d<DEVICE>]), meta-<DEVICE>.env (shas, trees,
# times of this device's launch).
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
# shellcheck source=tools/gpu_lock.sh
. tools/gpu_lock.sh
bv_defaults
BASE="${BV_BASE:?BV_BASE (the baseline tree) is required}"
DATA="${BV_DATA:-$HOME/b70-inference-server}"
DRY=0
RUNS="$(dirname "$STATE")"   # this tree's state root: the run, build and cpu locks live here

# an orchestrator from before the per-device locks holds .run.lock: it knows nothing of claims
if ! flock -n "$RUNS/.run.lock" true; then
  echo "remote.sh: an older orchestrator (one per tree) is active in this tree: $(cat "$RUNS/.current" 2>/dev/null)"
  exit 3
fi
exec 8> "$RUNS/.run-$DEVICE.lock"
if ! flock -n 8; then
  echo "remote.sh: another validation run on device $DEVICE is active in this tree: $(cat "$RUNS/.current-$DEVICE" 2>/dev/null)"
  exit 3
fi
basename "$STATE" > "$RUNS/.current-$DEVICE"

# A run validates the defaults: nothing from the operator's shell may switch a code path.
unset B70_KV_CACHE B70_DECODE_ATTN B70_PREFILL_ATTN B70_PREFILL_GDN_SCAN B70_PREFILL_GDN_SOLVE \
      B70_PREFILL_SILU_FUSED B70_PREFILL_REPLAY B70_PREFILL_PROFILE B70_TOKENIZER_JSON \
      ONEAPI_DEVICE_SELECTOR
export ZE_AFFINITY_MASK="$DEVICE"
export HOME_R BV_DATA="$DATA" OPENCODE_LOG="${OPENCODE_LOG:-}" A4_REF_DIR="${A4_REF_DIR:-}"

ts() { date '+%F %T'; }
# one line per event, appended with one write (two orchestrators share the file)
progress() {
  local line; line="$(ts) [d$DEVICE] $*"
  printf '%s\n' "$line" >> "$STATE/progress.log"
  printf '%s\n' "$line"
}

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
} > "$STATE/meta-$DEVICE.env"

is_redo() { case " ${BV_REDO:-} " in *" all "*|*" $1 "*) return 0 ;; esac; return 1; }

write_status() {   # id result rc start end secs reason [lock]
  local tmp="$STATE/.$1.status.$DEVICE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$DEVICE" "${8:-none}" > "$tmp" &&
    mv -f "$tmp" "$STATE/$1.status"
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

others_active() {   # prints the devices whose orchestrator is active on this tree (not this one)
  local f k out=""
  for f in "$RUNS"/.run-*.lock; do
    [ -e "$f" ] || continue
    k=${f##*/.run-}; k=${k%.lock}
    [ "$k" = "$DEVICE" ] && continue
    flock -n "$f" true || out="$out $k"
  done
  printf '%s' "$out"
}

# run_stage ID - claim the stage (or leave it to the orchestrator that holds it), then run it
run_stage() {
  local id="$1"
  if ! bv_index "$id" > /dev/null; then progress "$id UNKNOWN (not in stages.sh)"; return; fi
  if [ "$(bv_field "$id" sel)" = manual ]; then progress "$id MANUAL (commands only; never run here)"; return; fi
  exec 7> "$STATE/$id.claim"
  if [ "$(bv_field "$id" sel)" = always ]; then
    flock -x 7   # the preflight: wait for the other's (have.env is what later stages read), then run it again
  elif ! flock -n 7; then
    exec 7>&-
    progress "$id RUNNING under the other orchestrator on this state - left to it"
    return
  fi
  run_claimed "$id"
  exec 7>&-
}

run_claimed() {
  local id="$1" sel kind needs after title fn prev start t0 rc result reason k missing="" d blocked="" s
  local build="" lock=none others
  sel=$(bv_field "$id" sel); kind=$(bv_field "$id" kind); needs=$(bv_field "$id" needs)
  after=$(bv_field "$id" after); title=$(bv_field "$id" title)
  fn=$(bv_fn "$id")
  prev=$(status_of "$id")
  start=$(ts)
  if [ "$prev" = PASS ] && [ "$sel" != always ] && ! is_redo "$id"; then
    progress "$id PASS (kept from an earlier launch; --redo $id runs it again)"; return
  fi
  if [ "$sel" = g0 ]; then
    others=$(others_active)
    if [ -n "$others" ]; then
      progress "$id DEFERRED: G0 runs with no other orchestrator on this tree (device$others active) - launch it again when that run is done"
      return
    fi
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

  case "$kind" in
    gpu) lock=box ;;
    card) lock=card; bv_card_index "$DEVICE" || lock=box ;;   # DEVICE "0,1": no single card to take
    gpu-self) lock=self ;;
    cpu) [ "$sel" = always ] || lock=cpu ;;
  esac
  bv_builds "$id" && build=1
  # an earlier attempt's files (log, JUnit, rows, refusal outputs) are kept as *.prev
  for k in "$STATE/$id".*; do
    case "$k" in *.status|*.prev|*.claim) continue ;; esac
    if [ -f "$k" ]; then mv -f "$k" "$k.prev"; fi
  done
  write_status "$id" RUNNING "" "$start" "" "" "" "$lock"
  progress "$id START ($kind, lock $lock${build:+, build lock}) $title"
  t0=$(date +%s)
  (
    set +e
    exec 7>&- 8>&-   # the run lock and the claim stay with the orchestrator, not with what a stage starts
    STAGE="$id"; LOG="$STATE/$id.log"; BV_BAD=0; BV_OK=0; BV_SKIPS=0
    echo "=== stage $id: $title"
    echo "=== tree $TREE (sha $B70_GIT_SHA), baseline $BASE (${BV_BASE_SHA:-?}), device $ZE_AFFINITY_MASK, $(ts)"
    if [ -n "$build" ]; then
      exec 3> "$RUNS/.build.lock"
      echo "waiting for the build lock ($(ts))"
      flock -x 3 || { echo "BAD could not take $RUNS/.build.lock"; exit 1; }
      echo "build lock held ($(ts))"
    fi
    if [ "$lock" = cpu ]; then
      exec 4> "$RUNS/.cpu.lock"
      flock -n -x 4 || { echo "waiting for the cpu lock (a cpu stage of the other orchestrator) ($(ts))"; flock -x 4; }
    fi
    case "$lock" in
      box)
        echo "waiting for the GPU lock, whole box ($(ts))"
        gpu_lock_box || { echo "BAD could not take the whole-box GPU lock"; exit 1; }
        echo "GPU lock held, whole box ($(ts)); $(uptime)" ;;
      card)
        echo "waiting for the GPU lock, card $DEVICE ($(ts))"
        gpu_lock_card "$DEVICE" || { echo "BAD could not take the card $DEVICE lock"; exit 1; }
        echo "GPU lock held, card $DEVICE (the box shared) ($(ts)); $(uptime)" ;;
    esac
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
  write_status "$id" "$result" "$rc" "$start" "$(ts)" "$(( $(date +%s) - t0 ))" "${reason:-}" "$lock"
  progress "$id $result${reason:+ - $reason}"
}

progress "orchestrator start: $(basename "$STATE"), device $DEVICE, stages: $*"
for id in "$@"; do run_stage "$id"; done
echo "finished=$(ts)" >> "$STATE/meta-$DEVICE.env"
progress "orchestrator done"
