#!/usr/bin/env bash
# The box validation queue (docs/superpowers/plans/box-validation-queue.md) as one resumable,
# detached runbook. From the Mac, in the checkout to validate (normally main):
#
#   tools/box_validate.sh                 sync, launch every default stage on the box, wait,
#                                         fetch, write box-validation-<date>.md
#   tools/box_validate.sh --dry-run       print every command (local and remote) - no ssh
#   tools/box_validate.sh --list          the stages: id, row, default / opt-in / manual, needs
#   tools/box_validate.sh --status        what the box has recorded so far (ssh, read only)
#   tools/box_validate.sh --summary       fetch the results and write the summary only
#   tools/box_validate.sh --registry F    write the stage registry (TSV) to F - no ssh
#
# Selection (ids or row prefixes, comma separated: `r11`, `r1.gates,r7`):
#   --only S      run just these (plus the preflight; G0 must already be PASS in this state)
#   --with S      add opt-in stages (`--with r11.passkey120k,r7.passkey`, `--with optin`)
#   --skip S      leave these out of this launch
#   --redo S      run these even if already PASS (`--redo all`); G0's tests are reused by
#                 later stages, so `--redo g0.suite` is what re-runs a test the suite ran
#   --force       run past a G0 that has not passed (stop-the-line is the default)
#   --no-reuse    run every test, even one whose inputs are all unchanged since a recorded PASS
#                 (by default such a test keeps that PASS: tools/box_validate/reuse.py)
#   --nightly     the nightly run (docs/10-the-box.md): no run at all when nothing but documentation
#                 changed since the newest finished run of this tree (tools/box_validate/changed_since.sh);
#                 a full run (--no-reuse) when the newest finished full run is older than 7 days
#   --quick       the quick run (about an hour, for every change): the stages stages.sh's --quick block names,
#                 some with a smaller input (--list shows which and how); every other selected stage is
#                 recorded SKIP "full only". Its own state (<sha>-quick-vs-<baseline>, --state NAME must contain
#                 -quick): a quick PASS never stands for a full one. No box tidy at launch. With --status /
#                 --summary / --dry-run: the quick state
# Run control:
#   --baseline REF   G0's reference build (default ce36d2d: main before the first blind code
#                    commit, a86dd86; `--baseline main` when validating a branch before merging)
#   --state NAME     results directory on the box (default <sha>-vs-<baseline sha>); a new
#                    commit starts a new state, `--state <old>` continues an old one - with
#                    the baseline it was started against (its meta-*.env, else the name's
#                    -vs-<sha>); a --baseline naming another commit is refused
#   --no-wait        launch and return (re-run the same command later to re-attach)
#   --no-sync        do not rsync the tree (the box keeps what it has)
#   --push-data      first rsync (spec 15a, made on the Mac from the int4 checkpoint)
#                    oracle-out-ornith* from this Mac - with
#                    the A4 sets and references tools/toolcall/a4_ref.sh made here
#                    (oracle-out-ornith-a4, oracle-out-k2-a4, oracle-out-kolibri-a4; spec 15e / 18d /
#                    20e)
#   --out FILE       summary path (default box-validation-<date>.md in the repo root)
#   --poll SECONDS   poll interval (default 120)
#   --device N       the card this orchestrator runs on (default 0; ZE_AFFINITY_MASK=N, PORT
#                    8013 + 100 x N). One orchestrator per device and tree: a `--device 1
#                    --only <stages>` launch runs beside a device-0 launch on the same state
#                    (same commit: it does not sync while the other runs). Its `card` stages
#                    take card N only; `gpu` stages still take the whole box.
#
# Everything on the box runs detached (tools/probe/detach.sh) in an orchestrator
# (tools/box_validate/remote.sh) that survives the ssh session and the Mac: a WiFi drop
# costs nothing, and re-running the same command re-attaches or resumes - stages already
# PASS are kept. `gpu` stages hold the whole box (tools/gpu_lock.sh box), `card` stages one
# card (tools/gpu_lock.sh card $DEVICE) for their whole body; builds use -j$JOBS (44); timed
# stages are interleaved rounds after a warm-up, median of 3, uptime and the idle grade
# recorded. --dry-run / --list print the lock each stage takes. The stages themselves:
# tools/box_validate/stages.sh.
#
# Env: BOX (tools/box.env), REMOTE_DIR / BOX_SUFFIX (default suffix `validate`: the tree
# ~/b70-inference-server-validate), BASE_DIR (~/b70-inference-server-g0-<sha>), DATA_DIR
# (the tree holding oracle-out*, default b70-inference-server), JOBS, DEVICE (0, as --device), PORT,
# CMAKE_ARGS, SNAP_QWEN / SNAP_ORNITH / SNAP_K2 / SNAP_KOLIBRI, TOK_PYTHON, OPENCODE_LOG,
# A4_REF_DIR, ORACLE_IMAGE, G0_BITWISE_RE, G0_ALLOW_REMOVED, R10_RUNS, R2_ROUNDS (the server
# rows' rounds, 3), R8_DRAFT_VOCAB (r8.auto_rows' size, 128k), K2_ORACLE_MODEL /
# K2_REF_MIN_GB / K2_HFCHECK (r14.oracle), B70_K2_TIE_TOL (r14.golden, r15.golden), B70_KOL_TIE_TOL (r24 / r26 golden), LOCAL_DATA
# (for --push-data).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.."
ROOT="$PWD"

# shellcheck source=tools/box_validate/lib.sh
. tools/box_validate/lib.sh
# shellcheck source=tools/box_validate/stages.sh
. tools/box_validate/stages.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "box_validate.sh: $*" >&2; exit 2; }

MODE=run ONLY="" WITH="" SKIP="" REDO="" FORCE=0 NO_REUSE=0 NIGHTLY=0 QUICK=0 WAIT=1 SYNC=1 PUSH=0 OUT="" POLL=120
BASELINE_REF="${BASELINE_REF:-}" STATE_NAME=""
BASELINE_GIVEN=0; [ -z "$BASELINE_REF" ] || BASELINE_GIVEN=1
need() { [ "$#" -ge 2 ] && [ -n "$2" ] || die "$1 needs a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry; shift ;;
    --list) MODE=list; shift ;;
    --status) MODE=status; shift ;;
    --summary) MODE=summary; shift ;;
    --registry) need "$@"; MODE=registry; REGISTRY_OUT="$2"; shift 2 ;;
    --only) need "$@"; ONLY="$ONLY,$2"; shift 2 ;;
    --with) need "$@"; WITH="$WITH,$2"; shift 2 ;;
    --skip) need "$@"; SKIP="$SKIP,$2"; shift 2 ;;
    --redo) need "$@"; REDO="$REDO,$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-reuse) NO_REUSE=1; shift ;;
    --nightly) NIGHTLY=1; shift ;;
    --quick) QUICK=1; shift ;;
    --baseline) need "$@"; BASELINE_REF="$2"; BASELINE_GIVEN=1; shift 2 ;;
    --state) need "$@"; STATE_NAME="$2"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --no-sync) SYNC=0; shift ;;
    --push-data) PUSH=1; shift ;;
    --out) need "$@"; OUT="$2"; shift 2 ;;
    --poll) need "$@"; POLL="$2"; shift 2 ;;
    --device) need "$@"; bv_card_index "$2" || die "--device takes one card index (0, 1), not '$2'"; DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (--help)" ;;
  esac
done
DEVICE="${DEVICE:-0}"
export DEVICE
BASELINE_REF="${BASELINE_REF:-ce36d2d}"
[ "$NIGHTLY" = 0 ] || [ -z "$ONLY" ] || die "--nightly runs the default selection: drop --only"
[ "$NIGHTLY" = 0 ] || [ "$QUICK" = 0 ] || die "--nightly is a full run: drop --quick"
BV_QUICK=$QUICK   # the stage bodies' quick_or, for --dry-run

# ---- selection ------------------------------------------------------------------------
# expand "a,b,r11" [noopt] -> the ids those name, in registry order (manual stages never
# run). With noopt (--only), a row prefix names the row's non-opt-in stages: an opt-in stage
# runs only when named by its id, by `optin` or by `all`, or added with --with.
expand() {
  local list=",$1," noopt="${2:-}" i id sel hit
  for i in "${!BV_IDS[@]}"; do
    id="${BV_IDS[$i]}"; sel="${BV_SELS[$i]}"; hit=0
    case "$list" in *",$id,"*|*",all,"*) hit=1 ;; esac
    case "$list" in
      *",${id%%.*},"*) if [ "$sel" != optin ] || [ -z "$noopt" ]; then hit=1; fi ;;
    esac
    case "$list" in *",$sel,"*) hit=1 ;; esac
    if [ "$hit" = 1 ] && [ "$sel" != manual ]; then echo "$id"; fi
  done
}
check_tokens() {
  local t
  for t in $(printf '%s' "$1" | tr ',' ' '); do
    case "$t" in all|optin|default|g0) continue ;; esac
    [ -n "$(expand "$t")" ] || bv_index "$t" > /dev/null || die "no stage or row '$t' (--list)"
  done
}
for v in "$ONLY" "$WITH" "$SKIP" "$REDO"; do check_tokens "$v"; done
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }
SELECTED=""
pick_ids="$(expand "$ONLY" noopt | tr '\n' ' ')"
with_ids="$(expand "$WITH" | tr '\n' ' ')"
skip_ids="$(expand "$SKIP" | tr '\n' ' ')"
for i in "${!BV_IDS[@]}"; do
  id="${BV_IDS[$i]}"; sel="${BV_SELS[$i]}"
  if [ -n "$ONLY" ]; then
    want=0; { [ "$sel" = always ] || in_list "$id" "$pick_ids $with_ids"; } && want=1
  else
    want=0
    case "$sel" in always|g0|default) want=1 ;; esac
    in_list "$id" "$with_ids" && want=1
  fi
  [ "$sel" = manual ] && want=0
  in_list "$id" "$skip_ids" && [ "$sel" != always ] && want=0
  [ "$want" = 1 ] && SELECTED="$SELECTED $id"
done
SELECTED="${SELECTED# }"
REDO_IDS="$(expand "$REDO" | tr '\n' ' ')"
case ",$REDO," in *",all,"*) REDO_IDS="all" ;; esac

# ---- names ----------------------------------------------------------------------------
SHA="$(git rev-parse --short=12 HEAD)"
git diff --quiet HEAD -- 2>/dev/null || SHA="$SHA-dirty"
BASE_SHA="$(git rev-parse --verify --quiet "$BASELINE_REF^{commit}")" || die "baseline '$BASELINE_REF' is not a commit here"
BASE_SHORT="${BASE_SHA:0:12}"
# A state continues with the baseline it was started against: G0 compares this tree's build with
# the baseline tree the state built, so a resume must not re-resolve a moving ref (`main`).
state_baseline() {   # state_baseline NAME - the baseline sha in a state name (<sha>-vs-<baseline sha>), or nothing
  local b="${1##*-vs-}"
  [ "$b" != "$1" ] || return 0
  case "$b" in ""|*[!0-9a-f]*) return 0 ;; esac
  if [ "${#b}" -ge 7 ]; then echo "$b"; fi
}
use_recorded_baseline() {   # use_recorded_baseline SHA WHERE - adopt the state's baseline, or refuse a --baseline that differs
  local rec="$1" where="$2" full
  full="$(git rev-parse --verify --quiet "$rec^{commit}")" ||
    die "state $STATE_NAME was started against baseline $rec ($where), which is not a commit here - fetch it, or start a new state (drop --state)"
  [ "$full" != "$BASE_SHA" ] || return 0
  [ "$BASELINE_GIVEN" = 0 ] ||
    die "state $STATE_NAME was started against baseline ${full:0:12} ($where), but --baseline $BASELINE_REF is ${BASE_SHA:0:12} now; G0 must compare against the baseline the state built. Resume it without --baseline (it uses ${full:0:12}), or start a new state for ${BASE_SHA:0:12}: drop --state (the default name is <sha>-vs-<baseline>) or pick a new --state NAME"
  echo "baseline: state $STATE_NAME was started against ${full:0:12} ($where) - using it" >&2
  BASELINE_REF="${full:0:12}"; BASE_SHA="$full"; BASE_SHORT="${BASE_SHA:0:12}"
}
if [ -n "$STATE_NAME" ]; then
  rec="$(state_baseline "$STATE_NAME")"
  [ -z "$rec" ] || use_recorded_baseline "$rec" "its name"
fi
if [ -z "${REMOTE_DIR:-}" ]; then
  # shellcheck source=tools/box_dir.sh
  . tools/box_dir.sh
  REMOTE_DIR="$(BOX_SUFFIX="${BOX_SUFFIX:-validate}" b70_remote_dir "$ROOT")"
fi
export REMOTE_DIR
BASE_DIR_ENV="${BASE_DIR:-}"
BASE_DIR="${BASE_DIR_ENV:-b70-inference-server-g0-$BASE_SHORT}"
DATA_DIR="${DATA_DIR:-b70-inference-server}"
# a quick run keeps its own state: its PASSes are never kept as, or taken for, a full run's
if [ "$QUICK" = 1 ]; then STATE_NAME="${STATE_NAME:-${SHA}-quick-vs-$BASE_SHORT}"; fi
STATE_NAME="${STATE_NAME:-${SHA}-vs-$BASE_SHORT}"
case "$STATE_NAME" in
  *-quick|*-quick-*) [ "$QUICK" = 1 ] || die "state $STATE_NAME is a quick run's: add --quick, or name a full run's state" ;;
  *) [ "$QUICK" = 0 ] || die "--quick runs in a quick state: drop --state (<sha>-quick-vs-<baseline>), or name one containing -quick" ;;
esac
STATE_REL="b70-validate/$REMOTE_DIR/$STATE_NAME"
LOCAL_STATE="build/box-validation/$STATE_NAME"
DATE="$(date +%F)"
OUT="${OUT:-box-validation-$DATE$(quick_or '' -quick).md}"

# ---- the commands the driver itself sends ------------------------------------------------
q() { printf '%q' "$1"; }
remote_env() {
  local s
  s="B70_GIT_SHA=$(q "$SHA") BV_BASE_SHA=$BASE_SHA BV_BASE_REF=$(q "$BASELINE_REF")"
  s="$s BV_BASE=\"\$HOME/$BASE_DIR\" BV_DATA=\"\$HOME/$DATA_DIR\" BV_FORCE=$FORCE BV_NO_REUSE=$NO_REUSE BV_QUICK=$QUICK BV_REDO=$(q "$REDO_IDS")"
  local v
  for v in JOBS DEVICE PORT CMAKE_ARGS SNAP_QWEN SNAP_ORNITH SNAP_K2 SNAP_KOLIBRI TOK_PYTHON OPENCODE_LOG \
           A4_REF_DIR ORACLE_IMAGE G0_BITWISE_RE G0_ALLOW_REMOVED R10_RUNS R2_ROUNDS R8_DRAFT_VOCAB \
           K2_ORACLE_MODEL K2_REF_MIN_GB K2_HFCHECK B70_K2_TIE_TOL B70_KOL_TIE_TOL; do
    if [ -n "${!v:-}" ]; then s="$s $v=$(q "${!v}")"; fi
  done
  printf '%s' "$s"
}
# One orchestrator per tree and device: ~/b70-validate/<tree>/.run-<device>.lock (.current-<device>
# names its state); .run.lock is an orchestrator from before the per-device locks. CHECK_CMD prints
# LEGACY <state> when one of those runs, OTHER <device> <state> <sha> per other device's active
# orchestrator, then IDLE <progress lines> / RUNNING <progress lines> <state> for this device.
RUNS_REL="b70-validate/$REMOTE_DIR"
ORCH_LOG="orchestrator-$DEVICE.log"
CHECK_CMD="d=\"\$HOME/$RUNS_REL\"; mkdir -p \"\$d\"; n=\$(cat \"\$HOME/$STATE_REL/progress.log\" 2>/dev/null | wc -l); flock -n \"\$d/.run.lock\" true || echo \"LEGACY \$(cat \"\$d/.current\" 2>/dev/null)\"; for f in \"\$d\"/.run-*.lock; do [ -e \"\$f\" ] || continue; k=\${f##*/.run-}; k=\${k%.lock}; [ \"\$k\" = $DEVICE ] && continue; flock -n \"\$f\" true && continue; c=\$(cat \"\$d/.current-\$k\" 2>/dev/null); echo \"OTHER \$k \$c \$(sed -n 's/^tree_sha=//p' \"\$d/\$c/meta-\$k.env\" 2>/dev/null)\"; done; if flock -n \"\$d/.run-$DEVICE.lock\" true; then echo \"IDLE \$n\"; else echo \"RUNNING \$n \$(cat \"\$d/.current-$DEVICE\" 2>/dev/null)\"; fi"
# the baselines the state's orchestrators recorded (meta-<device>.env, one sha per line; nothing for a new state)
STATE_BASE_CMD="sed -n 's/^baseline_sha=//p' \"\$HOME/$STATE_REL\"/meta-*.env 2>/dev/null | sort -u"
base_cmds() {   # the commands that name the baseline (again after the state's recorded one is adopted)
  BASE_DIR="${BASE_DIR_ENV:-b70-inference-server-g0-$BASE_SHORT}"
  BASE_CHECK_CMD="cat \"\$HOME/$BASE_DIR/.b70-baseline-ref\" 2>/dev/null || true"
  BASE_EXTRACT_CMD="mkdir -p \"\$HOME/$BASE_DIR\" && tar -xf - -C \"\$HOME/$BASE_DIR\" && echo $BASE_SHA > \"\$HOME/$BASE_DIR/.b70-baseline-ref\""
  LAUNCH_CMD="mkdir -p \"\$HOME/$STATE_REL\" && env $(remote_env) tools/probe/detach.sh \"\$HOME/$STATE_REL/$ORCH_LOG\" tools/box_validate/remote.sh \"\$HOME/$STATE_REL\" $SELECTED"
}
base_cmds
POLL_CMD="cat \"\$HOME/$STATE_REL/progress.log\" 2>/dev/null; echo @@; tail -1 \"\$HOME/$STATE_REL/$ORCH_LOG\" 2>/dev/null; echo @@; flock -n \"\$HOME/$RUNS_REL/.run-$DEVICE.lock\" true && echo idle || echo running"
STATUS_CMD="cd \"\$HOME/$STATE_REL\" 2>/dev/null && cat *.status 2>/dev/null; echo @@; tail -8 \"\$HOME/$STATE_REL/progress.log\" 2>/dev/null; echo @@; d=\"\$HOME/$RUNS_REL\"; for f in \"\$d\"/.run-*.lock; do [ -e \"\$f\" ] || continue; k=\${f##*/.run-}; k=\${k%.lock}; if flock -n \"\$f\" true; then s=idle; else s=RUNNING; fi; echo \"\$k \$s \$(cat \"\$d/.current-\$k\" 2>/dev/null)\"; done"
FETCH_SRC="b70-validate/$REMOTE_DIR/$STATE_NAME/"
LOCAL_DATA="${LOCAL_DATA:-$(cd "$(git rev-parse --git-common-dir)/.." && pwd)}"

# ---- the registry, for the summary ----------------------------------------------------------
write_registry() {   # write_registry FILE
  local i id
  {
    for i in "${!BV_ROW_IDS[@]}"; do printf 'row\t%s\t%s\n' "${BV_ROW_IDS[$i]}" "${BV_ROW_TITLES[$i]}"; done
    for i in "${!BV_NOTES[@]}"; do printf 'note\t%s\n' "${BV_NOTES[$i]}"; done
    for i in "${!BV_IDS[@]}"; do
      printf 'stage\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${BV_IDS[$i]}" "${BV_ROWS[$i]}" "${BV_SELS[$i]}" \
        "${BV_KINDS[$i]}" "${BV_NEEDS[$i]}" "${BV_AFTERS[$i]}" "${BV_TITLES[$i]}"
    done
    for i in "${!BV_IDS[@]}"; do
      id="${BV_IDS[$i]}"
      case "${BV_SELS[$i]}" in
        optin) printf 'cmd\t%s\t%s\n' "$id" "tools/box_validate.sh --with $id      # runs these on the box:" ;;
        manual) ;;
        *) continue ;;
      esac
      dry_body "$id" | sed 's/^ *//' | while IFS= read -r line; do printf 'cmd\t%s\t%s\n' "$id" "$line"; done
    done
  } > "$1"
}

dry_setup() {
  DRY=1
  # shellcheck disable=SC2016 # the literal '$HOME': the printed paths are the box's
  HOME_R='$HOME'
  HF_HOME_R=""
  TREE="\$HOME/$REMOTE_DIR"; BASE="\$HOME/$BASE_DIR"; DATA="\$HOME/$DATA_DIR"; STATE="\$HOME/$STATE_REL"
  B70_GIT_SHA="$SHA"
  OPENCODE_LOG="${OPENCODE_LOG:-\$OPENCODE_LOG}"; A4_REF_DIR="${A4_REF_DIR:-\$A4_REF_DIR}"
  bv_defaults
}
dry_body() {   # the body of one stage, printed (DRY=1), in a subshell
  ( set +e; dry_setup; STAGE="$1"; LOG="$STATE/$1.log"; "$(bv_fn "$1")" )
}

lock_of() {   # lock_of ID - the locks the orchestrator takes for a stage, in words
  local id="$1" i lock; i=$(bv_index "$id")
  case "${BV_KINDS[$i]}" in
    gpu) lock="holds the whole box for the whole stage (tools/gpu_lock.sh box: ~/b70-gpu.lock exclusive, the ~/b70-gpu.wait gate while waiting)" ;;
    card) if bv_card_index "$DEVICE"; then
            lock="holds the card lock for device $DEVICE for the whole stage (tools/gpu_lock.sh card $DEVICE: ~/b70-gpu$DEVICE.lock exclusive + ~/b70-gpu.lock shared, ZE_AFFINITY_MASK=$DEVICE) - beside the other card's stream"
          else lock="holds the whole box (DEVICE=$DEVICE is not one card)"; fi ;;
    gpu-self) lock="its script takes ~/b70-gpu.lock itself (the whole box)" ;;
    cpu) if [ "${BV_SELS[$i]}" = always ]; then lock="no GPU lock (no device)"
         else lock="no GPU lock (no device); one cpu stage at a time on this tree (~/$RUNS_REL/.cpu.lock)"; fi ;;
    *) lock="no lock" ;;
  esac
  bv_builds "$id" && lock="$lock; builds holding ~/$RUNS_REL/.build.lock"
  [ "${BV_SELS[$i]}" = g0 ] && lock="$lock; G0 - deferred while another orchestrator is active on this tree"
  printf '%s\n' "$lock"
}

device_note() {   # a launch on device 1: what still takes the whole box, and what belongs on device 0
  [ "$DEVICE" = 0 ] && return 0
  local id g=""
  for id in $SELECTED; do
    case "$(bv_field "$id" kind)" in gpu|gpu-self) g="$g $id" ;; esac
  done
  [ -n "$g" ] || return 0
  echo "# note: these take the WHOLE box even on device $DEVICE:$g"
  echo "#       (a timed / series row belongs on device 0: device 1 is ~3.4 % slower on prefill)"
}

print_stage() {
  local id="$1" i; i=$(bv_index "$id")
  local kind="${BV_KINDS[$i]}"
  echo ""
  echo "--- $id  [row ${BV_ROWS[$i]}, ${BV_SELS[$i]}, $kind] ${BV_TITLES[$i]}"
  echo "    needs: ${BV_NEEDS[$i]}   after: ${BV_AFTERS[$i]}"
  echo "    lock: $(lock_of "$id")"
  echo "    log: ~/$STATE_REL/$id.log   status: ~/$STATE_REL/$id.status"
  if [ "$QUICK" = 1 ]; then
    case "${BV_QUICKS[$i]}" in
      "") echo "    quick: full only - recorded SKIP, not run"; return 0 ;;
      same) echo "    quick: as in the full run" ;;
      *) echo "    quick: ${BV_QUICKS[$i]}" ;;
    esac
  fi
  dry_body "$id"
}
quick_count() {   # the selected stages the quick run runs
  local id n=0
  for id in $SELECTED; do [ -z "$(bv_field "$id" quick)" ] || n=$((n + 1)); done
  echo "$n"
}

# ---- modes ----------------------------------------------------------------------------
if [ "$MODE" = registry ]; then write_registry "$REGISTRY_OUT"; exit 0; fi

if [ "$MODE" = list ]; then
  printf '%-18s %-4s %-8s %-9s %-12s %-5s %-28s %s\n' STAGE ROW SEL KIND LOCK QUICK NEEDS TITLE
  for i in "${!BV_IDS[@]}"; do
    mark=""; in_list "${BV_IDS[$i]}" "$SELECTED" && mark="*"
    case "${BV_KINDS[$i]}" in
      gpu) lk=box ;; card) lk="card$DEVICE" ;; gpu-self) lk=self ;;
      cpu) lk=cpu; [ "${BV_SELS[$i]}" = always ] && lk=- ;; *) lk=- ;;
    esac
    bv_builds "${BV_IDS[$i]}" && lk="$lk+build"
    case "${BV_QUICKS[$i]}" in "") qk=- ;; same) qk=same ;; *) qk=less ;; esac
    printf '%-18s %-4s %-8s %-9s %-12s %-5s %-28s %s\n' "${BV_IDS[$i]}$mark" "${BV_ROWS[$i]}" "${BV_SELS[$i]}" \
      "${BV_KINDS[$i]}" "$lk" "$qk" "${BV_NEEDS[$i]}" "${BV_TITLES[$i]}"
  done
  echo ""
  echo "* = in this launch's selection ($(echo "$SELECTED" | wc -w | tr -d ' ') stages). LOCK: box = the whole box"
  echo "(~/b70-gpu.lock exclusive), card$DEVICE = card $DEVICE only (--device), self = the stage's script locks,"
  echo "cpu = one cpu stage at a time per tree, +build = the tree's build lock. QUICK (--quick): same = runs as in"
  echo "the full run, less = a smaller input (below), - = full only (recorded SKIP). The smaller inputs:"
  for i in "${!BV_IDS[@]}"; do
    case "${BV_QUICKS[$i]}" in ""|same) ;; *) printf '  %-18s %s\n' "${BV_IDS[$i]}" "${BV_QUICKS[$i]}" ;; esac
  done
  echo "Rows:"
  for i in "${!BV_ROW_IDS[@]}"; do printf '  %-3s %s\n' "${BV_ROW_IDS[$i]}" "${BV_ROW_TITLES[$i]}"; done
  exit 0
fi

if [ "$MODE" = dry ]; then
  echo "# tools/box_validate.sh --dry-run: every command, nothing connects."
  echo "# \$BOX is the address in tools/box.env; ~ and \$HOME are the box's."
  echo "# tree under test: $SHA -> \$BOX:~/$REMOTE_DIR"
  echo "# baseline: $BASELINE_REF = $BASE_SHA -> \$BOX:~/$BASE_DIR"
  echo "# state: \$BOX:~/$STATE_REL  (fetched to $LOCAL_STATE)"
  echo "# stages ($(echo "$SELECTED" | wc -w | tr -d ' ')): $SELECTED"
  [ -n "$REDO_IDS" ] && echo "# redo: $REDO_IDS"
  [ "$QUICK" = 1 ] && echo "# quick run: $(quick_count) of these run, the rest are recorded SKIP (full only)"
  echo "# device $DEVICE: ZE_AFFINITY_MASK=$DEVICE, PORT ${PORT:-$((8013 + 100 * DEVICE))}, orchestrator lock ~/$RUNS_REL/.run-$DEVICE.lock, log ~/$STATE_REL/$ORCH_LOG"
  device_note
  echo ""
  echo "== 1. a run already active in this tree on device $DEVICE? (attach to it instead of launching; another"
  echo "   device's orchestrator on the same state and commit: launch beside it without syncing)"
  echo "  ssh \$BOX '$CHECK_CMD'"
  echo "== 1a. the baseline the state was started against (a resume keeps it; a --baseline naming another commit is refused)"
  echo "  ssh \$BOX '$STATE_BASE_CMD'"
  if [ "$PUSH" = 1 ]; then
    echo "== 1b. --push-data"
    echo "  rsync -a $LOCAL_DATA/oracle-out-ornith $LOCAL_DATA/oracle-out-ornith-mtp $LOCAL_DATA/oracle-out-ornith-a4 $LOCAL_DATA/oracle-out-k2-a4 $LOCAL_DATA/oracle-out-kolibri-a4 \$BOX:$DATA_DIR/   (each that exists)"
  fi
  echo "== 2. sync the tree under test"
  [ "$SYNC" = 1 ] && echo "  REMOTE_DIR=$REMOTE_DIR tools/box.sh sync   (not while another device's orchestrator runs from the tree)" || echo "  (--no-sync)"
  echo "== 3. the baseline tree (skipped when ~/$BASE_DIR/.b70-baseline-ref already names $BASE_SHA)"
  echo "  ssh \$BOX '$BASE_CHECK_CMD'"
  echo "  git archive --format=tar $BASE_SHA | ssh \$BOX '$BASE_EXTRACT_CMD'"
  echo "== 4. launch the orchestrator, detached"
  echo "  REMOTE_DIR=$REMOTE_DIR tools/box.sh run '$LAUNCH_CMD'"
  echo "== 5. what the orchestrator runs on the box, in order (from ~/$REMOTE_DIR)"
  for id in $SELECTED; do print_stage "$id"; done
  echo ""
  echo "== 6. poll every ${POLL}s until $ORCH_LOG ends in ALLDONE (ssh failures are retried)"
  echo "  ssh \$BOX '$POLL_CMD'"
  echo "== 7. fetch"
  echo "  rsync -az \$BOX:$FETCH_SRC $LOCAL_STATE/"
  echo "== 8. summary"
  echo "  python3 tools/box_validate/summary.py report --state $LOCAL_STATE --registry $LOCAL_STATE/registry.tsv --logdir $LOCAL_STATE --date $DATE --out $OUT"
  exit 0
fi

# ---- from here on the box is needed --------------------------------------------------------
_box_env="$ROOT/tools/box.env"
# shellcheck source=/dev/null
[ -r "$_box_env" ] && . "$_box_env"
: "${BOX:?set BOX=user@host in the environment or in tools/box.env (see tools/box.env.example)}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=20 -o ServerAliveCountMax=3)

fetch_and_summarise() {
  mkdir -p "$LOCAL_STATE"
  echo "fetch: \$BOX:~/$STATE_REL -> $LOCAL_STATE"
  rsync -az "$BOX:$FETCH_SRC" "$LOCAL_STATE/"
  write_registry "$LOCAL_STATE/registry.tsv"
  python3 tools/box_validate/summary.py report --state "$LOCAL_STATE" --registry "$LOCAL_STATE/registry.tsv" \
    --logdir "$LOCAL_STATE" --date "$DATE" --out "$OUT"
}

if [ "$MODE" = status ]; then
  out=$("${SSH[@]}" "$BOX" "$STATUS_CMD") || die "ssh failed"
  rest="${out#*@@}"
  echo "orchestrators in ~/$RUNS_REL (one per device):"
  printf '%s\n' "${rest#*@@}" | awk 'NF >= 2 { printf "  device %s: %s%s\n", $1, $2, (NF >= 3 ? ", state " $3 : "") }'
  echo "state ~/$STATE_REL$(quick_or '' ' - a QUICK run (--quick)') (DEV: the device that ran the stage and its lock):"
  printf '%s\n' "${out%%@@*}" | awk -F'\t' 'NF >= 2 { printf "  %-18s %-8s %-8s %s\n", $1, $2, (NF >= 9 ? $8 " " $9 : "-"), $7 }'
  echo "recent:"; printf '%s\n' "${rest%%@@*}" | sed '/^$/d; s/^/  /'
  exit 0
fi
if [ "$MODE" = summary ]; then fetch_and_summarise; exit 0; fi

poll() {   # poll SEEN - print new progress lines until the orchestrator is done
  local seen="$1" fails=0 idle=0 out prog tail state lines n
  sleep 15   # let a just-launched orchestrator take its lock before the first look
  while :; do
    if ! out=$("${SSH[@]}" "$BOX" "$POLL_CMD" 2>/dev/null); then
      fails=$((fails + 1))
      echo "$(date +%T) poll: ssh failed ($fails) - the run goes on on the box; retrying in ${POLL}s (Ctrl-C is safe: re-run to re-attach)"
      sleep "$POLL"; continue
    fi
    fails=0
    prog="${out%%@@*}"; out="${out#*@@}"; tail="${out%%@@*}"; state="${out#*@@}"
    lines=$(printf '%s' "$prog" | sed '/^$/d')
    n=$(printf '%s\n' "$lines" | sed '/^$/d' | wc -l | tr -d ' ')
    if [ "$n" -gt "$seen" ]; then printf '%s\n' "$lines" | tail -n $((n - seen)); seen=$n; fi
    case "$tail" in *ALLDONE*) echo "orchestrator: $(printf '%s' "$tail" | tr -d '\n')"; return 0 ;; esac
    # idle twice in a row (its lock free, no ALLDONE): killed or the box rebooted
    case "$state" in
      *idle*) idle=$((idle + 1))
        if [ "$idle" -ge 2 ]; then
          echo "orchestrator is not running and did not finish (box rebooted?) - re-run to resume"; return 1
        fi ;;
      *) idle=0 ;;
    esac
    sleep "$POLL"
  done
}

check=$("${SSH[@]}" "$BOX" "$CHECK_CMD") || die "cannot reach the box (ssh $BOX)"
legacy=$(printf '%s\n' "$check" | awk '$1 == "LEGACY" { print $2 }')
[ -z "$(printf '%s\n' "$check" | grep '^LEGACY')" ] ||
  die "an orchestrator from before the per-device locks is active in ~/$REMOTE_DIR (state ${legacy:-?}) - wait for it to finish"
recorded=$("${SSH[@]}" "$BOX" "$STATE_BASE_CMD") || die "cannot read the state's baseline (ssh $BOX)"
recorded=$(printf '%s\n' "$recorded" | grep -v '^?*$' || true)
if [ -n "$recorded" ]; then
  [ "$(printf '%s\n' "$recorded" | wc -l | tr -d ' ')" = 1 ] ||
    die "state $STATE_NAME records more than one baseline ($(printf '%s' "$recorded" | tr '\n' ' ')) in its meta-*.env - start a new state"
  use_recorded_baseline "$recorded" "its meta-*.env"
  base_cmds
fi
self=$(printf '%s\n' "$check" | tail -1)
seen=$(printf '%s' "$self" | awk '{ print $2 }')
case "$self" in
  RUNNING*)
    current=$(printf '%s' "$self" | awk '{ print $3 }')
    [ "$current" = "$STATE_NAME" ] || die "another run ($current) is active in ~/$REMOTE_DIR on device $DEVICE - wait, or --state $current to attach"
    echo "a run of this state is active on the box: attaching"
    [ "$WAIT" = 1 ] || exit 0
    poll 0 || true
    fetch_and_summarise
    exit 0 ;;
esac
# Another device's orchestrator runs from this tree: a second stream shares its state and its
# commit, and must not sync (box.sh sync rewrites the scripts it is running) or push data under it.
others=$(printf '%s\n' "$check" | awk '$1 == "OTHER" { print $2 " " $3 " " $4 }')
if [ -n "$others" ]; then
  while read -r odev ostate osha; do
    [ "$ostate" = "$STATE_NAME" ] ||
      die "device $odev runs state ${ostate:-?} in ~/$REMOTE_DIR - a second stream joins that state (--state $ostate, the same commit)"
    [ -z "$osha" ] || [ "$osha" = "$SHA" ] ||
      die "device $odev's orchestrator runs commit $osha, this checkout is $SHA - check out $osha for the second stream"
    echo "device $odev's orchestrator runs this state from ~/$REMOTE_DIR: launching beside it on device $DEVICE"
  done <<EOF
$others
EOF
  [ "$PUSH" = 0 ] || die "--push-data while another orchestrator runs from ~/$REMOTE_DIR: push when it is done"
  if [ "$SYNC" = 1 ]; then echo "not syncing: the tree on the box is this commit, and the other orchestrator runs from it"; SYNC=0; fi
fi

if [ "$PUSH" = 1 ]; then
  # the golden sets and A4 references made on the Mac
  for d in oracle-out-ornith oracle-out-ornith-mtp oracle-out-ornith-a4 oracle-out-k2-a4 oracle-out-kolibri-a4; do
    if [ -d "$LOCAL_DATA/$d" ]; then rsync -a "$LOCAL_DATA/$d" "$BOX:$DATA_DIR/"; else echo "push-data: no $LOCAL_DATA/$d"; fi
  done
fi
if [ "$NIGHTLY" = 1 ]; then
  # The newest FINISHED run of this tree (its progress.log says the orchestrator is done), and whether a finished
  # full run (a state marked full-run) is younger than 7 days.
  runs="~/b70-validate/$REMOTE_DIR"
  last=$("${SSH[@]}" "$BOX" "cd $runs 2>/dev/null && for s in \$(ls -td *-vs-* 2>/dev/null | grep -v -- -quick); do grep -q 'orchestrator done' \$s/progress.log 2>/dev/null && { echo \$s; break; }; done") || die "ssh failed"
  recent_full=$("${SSH[@]}" "$BOX" "cd $runs 2>/dev/null && for f in \$(find . -maxdepth 2 -name full-run -mtime -7); do grep -q 'orchestrator done' \$(dirname \$f)/progress.log 2>/dev/null && { echo \$f; break; }; done") || die "ssh failed"
  if [ -n "$last" ] && [ "${SHA%-dirty}" = "$SHA" ]; then
    lsha=${last%%-vs-*}; lsha=${lsha%-dirty}
    if changed=$(tools/box_validate/changed_since.sh "$lsha" 2> /dev/null) && [ -z "$changed" ] && [ -n "$recent_full" ]; then
      echo "nightly: nothing but documentation changed since the last finished run ($last) - no run tonight"
      exit 0
    fi
    [ -n "${changed:-}" ] && echo "nightly: $(printf '%s\n' "$changed" | wc -l | tr -d ' ') files changed since $last (e.g. $(printf '%s\n' "$changed" | head -3 | tr '\n' ' '))"
  fi
  if [ -z "$recent_full" ]; then
    echo "nightly: no finished full run in the last 7 days - this one runs every test (--no-reuse)"
    NO_REUSE=1
    LAUNCH_CMD="mkdir -p \"\$HOME/$STATE_REL\" && env $(remote_env) tools/probe/detach.sh \"\$HOME/$STATE_REL/$ORCH_LOG\" tools/box_validate/remote.sh \"\$HOME/$STATE_REL\" $SELECTED"
  fi
fi
if [ "$SYNC" = 1 ]; then tools/box.sh sync; fi
if [ "$NO_REUSE" = 1 ] && [ "$QUICK" = 0 ]; then "${SSH[@]}" "$BOX" "mkdir -p ~/$STATE_REL && touch ~/$STATE_REL/full-run" || die "ssh failed"; fi
if [ "$("${SSH[@]}" "$BOX" "$BASE_CHECK_CMD")" != "$BASE_SHA" ]; then
  echo "baseline: extracting $BASELINE_REF ($BASE_SHORT) into ~/$BASE_DIR"
  git archive --format=tar "$BASE_SHA" | "${SSH[@]}" "$BOX" "$BASE_EXTRACT_CMD"
fi
# The box's system disk (operator 2026-11-10): finished branches' trees and old baselines go to trash at every full
# launch (not a quick one: it runs for every change, often beside another branch's work).
if [ "$QUICK" = 0 ]; then tools/box_tidy.sh | grep -E '^(moved|trash|system disk)' || true; fi
echo "launch: $(echo "$SELECTED" | wc -w | tr -d ' ') stages, state ~/$STATE_REL"
tools/box.sh run "$LAUNCH_CMD"
if [ "$WAIT" = 0 ]; then
  echo "launched; re-run the same command to re-attach, --status to look, --summary to fetch"
  exit 0
fi
poll "${seen:-0}" || true
fetch_and_summarise
