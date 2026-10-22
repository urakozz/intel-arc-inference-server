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
# Run control:
#   --baseline REF   G0's reference build (default b32aaaf: main before the first blind code
#                    commit, 5ffcb7c; `--baseline main` when validating a branch before merging)
#   --state NAME     results directory on the box (default <sha>-vs-<baseline sha>); a new
#                    commit starts a new state, `--state <old>` continues an old one
#   --no-wait        launch and return (re-run the same command later to re-attach)
#   --no-sync        do not rsync the tree (the box keeps what it has)
#   --push-data      first rsync the Agnes checkpoint, oracle-out-agnes* and (spec 15a, made on
#                    the Mac from the int4 checkpoint) oracle-out-ornith* from this Mac - with
#                    the A4 sets and references tools/toolcall/a4_ref.sh made here
#                    (oracle-out-ornith-a4, oracle-out-k2-a4, oracle-out-kolibri-a4; spec 15e / 18d /
#                    20e); the Agnes checkpoint is skipped when this Mac no longer has it
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
# CMAKE_ARGS, SNAP_QWEN / SNAP_AGNES / SNAP_ORNITH / SNAP_K2 / SNAP_KOLIBRI, TOK_PYTHON, OPENCODE_LOG,
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

MODE=run ONLY="" WITH="" SKIP="" REDO="" FORCE=0 WAIT=1 SYNC=1 PUSH=0 OUT="" POLL=120
BASELINE_REF="${BASELINE_REF:-b32aaaf}" STATE_NAME=""
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
    --baseline) need "$@"; BASELINE_REF="$2"; shift 2 ;;
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
if [ -z "${REMOTE_DIR:-}" ]; then
  # shellcheck source=tools/box_dir.sh
  . tools/box_dir.sh
  REMOTE_DIR="$(BOX_SUFFIX="${BOX_SUFFIX:-validate}" b70_remote_dir "$ROOT")"
fi
export REMOTE_DIR
BASE_DIR="${BASE_DIR:-b70-inference-server-g0-$BASE_SHORT}"
DATA_DIR="${DATA_DIR:-b70-inference-server}"
STATE_NAME="${STATE_NAME:-${SHA}-vs-$BASE_SHORT}"
STATE_REL="b70-validate/$REMOTE_DIR/$STATE_NAME"
LOCAL_STATE="build/box-validation/$STATE_NAME"
DATE="$(date +%F)"
OUT="${OUT:-box-validation-$DATE.md}"

# ---- the commands the driver itself sends ------------------------------------------------
q() { printf '%q' "$1"; }
remote_env() {
  local s
  s="B70_GIT_SHA=$(q "$SHA") BV_BASE_SHA=$BASE_SHA BV_BASE_REF=$(q "$BASELINE_REF")"
  s="$s BV_BASE=\"\$HOME/$BASE_DIR\" BV_DATA=\"\$HOME/$DATA_DIR\" BV_FORCE=$FORCE BV_REDO=$(q "$REDO_IDS")"
  local v
  for v in JOBS DEVICE PORT CMAKE_ARGS SNAP_QWEN SNAP_AGNES SNAP_ORNITH SNAP_K2 SNAP_KOLIBRI TOK_PYTHON OPENCODE_LOG \
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
BASE_CHECK_CMD="cat \"\$HOME/$BASE_DIR/.b70-baseline-ref\" 2>/dev/null || true"
BASE_EXTRACT_CMD="mkdir -p \"\$HOME/$BASE_DIR\" && tar -xf - -C \"\$HOME/$BASE_DIR\" && echo $BASE_SHA > \"\$HOME/$BASE_DIR/.b70-baseline-ref\""
LAUNCH_CMD="mkdir -p \"\$HOME/$STATE_REL\" && env $(remote_env) tools/probe/detach.sh \"\$HOME/$STATE_REL/$ORCH_LOG\" tools/box_validate/remote.sh \"\$HOME/$STATE_REL\" $SELECTED"
POLL_CMD="cat \"\$HOME/$STATE_REL/progress.log\" 2>/dev/null; echo @@; tail -1 \"\$HOME/$STATE_REL/$ORCH_LOG\" 2>/dev/null; echo @@; flock -n \"\$HOME/$RUNS_REL/.run-$DEVICE.lock\" true && echo idle || echo running"
STATUS_CMD="cd \"\$HOME/$STATE_REL\" 2>/dev/null && cat *.status 2>/dev/null; echo @@; tail -8 \"\$HOME/$STATE_REL/progress.log\" 2>/dev/null; echo @@; d=\"\$HOME/$RUNS_REL\"; for f in \"\$d\"/.run-*.lock; do [ -e \"\$f\" ] || continue; k=\${f##*/.run-}; k=\${k%.lock}; if flock -n \"\$f\" true; then s=idle; else s=RUNNING; fi; echo \"\$k \$s \$(cat \"\$d/.current-\$k\" 2>/dev/null)\"; done"
FETCH_SRC="b70-validate/$REMOTE_DIR/$STATE_NAME/"
LOCAL_DATA="${LOCAL_DATA:-$(cd "$(git rev-parse --git-common-dir)/.." && pwd)}"
AGNES_HF="models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ"

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
  dry_body "$id"
}

# ---- modes ----------------------------------------------------------------------------
if [ "$MODE" = registry ]; then write_registry "$REGISTRY_OUT"; exit 0; fi

if [ "$MODE" = list ]; then
  printf '%-18s %-4s %-8s %-9s %-12s %-28s %s\n' STAGE ROW SEL KIND LOCK NEEDS TITLE
  for i in "${!BV_IDS[@]}"; do
    mark=""; in_list "${BV_IDS[$i]}" "$SELECTED" && mark="*"
    case "${BV_KINDS[$i]}" in
      gpu) lk=box ;; card) lk="card$DEVICE" ;; gpu-self) lk=self ;;
      cpu) lk=cpu; [ "${BV_SELS[$i]}" = always ] && lk=- ;; *) lk=- ;;
    esac
    bv_builds "${BV_IDS[$i]}" && lk="$lk+build"
    printf '%-18s %-4s %-8s %-9s %-12s %-28s %s\n' "${BV_IDS[$i]}$mark" "${BV_ROWS[$i]}" "${BV_SELS[$i]}" \
      "${BV_KINDS[$i]}" "$lk" "${BV_NEEDS[$i]}" "${BV_TITLES[$i]}"
  done
  echo ""
  echo "* = in this launch's selection ($(echo "$SELECTED" | wc -w | tr -d ' ') stages). LOCK: box = the whole box"
  echo "(~/b70-gpu.lock exclusive), card$DEVICE = card $DEVICE only (--device), self = the stage's script locks,"
  echo "cpu = one cpu stage at a time per tree, +build = the tree's build lock. Rows:"
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
  echo "# device $DEVICE: ZE_AFFINITY_MASK=$DEVICE, PORT ${PORT:-$((8013 + 100 * DEVICE))}, orchestrator lock ~/$RUNS_REL/.run-$DEVICE.lock, log ~/$STATE_REL/$ORCH_LOG"
  device_note
  echo ""
  echo "== 1. a run already active in this tree on device $DEVICE? (attach to it instead of launching; another"
  echo "   device's orchestrator on the same state and commit: launch beside it without syncing)"
  echo "  ssh \$BOX '$CHECK_CMD'"
  if [ "$PUSH" = 1 ]; then
    echo "== 1b. --push-data"
    echo "  rsync -a --info=progress2 ~/.cache/huggingface/hub/$AGNES_HF \$BOX:.cache/huggingface/hub/   (only if this Mac still has it)"
    echo "  rsync -a $LOCAL_DATA/oracle-out-agnes $LOCAL_DATA/oracle-out-agnes-mtp $LOCAL_DATA/oracle-out-ornith $LOCAL_DATA/oracle-out-ornith-mtp $LOCAL_DATA/oracle-out-ornith-a4 $LOCAL_DATA/oracle-out-k2-a4 $LOCAL_DATA/oracle-out-kolibri-a4 \$BOX:$DATA_DIR/   (each that exists)"
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
  echo "state ~/$STATE_REL (DEV: the device that ran the stage and its lock):"
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
  # spec 14 checklist steps 1 and 6: the Mac's complete Agnes checkpoint and golden sets
  # The Agnes checkpoint only when this Mac still has it (it may have been dropped from the HF
  # cache - then download it on the box); the golden sets are pushed either way.
  agnes_src="${HF_HOME:-$HOME/.cache/huggingface}/hub/$AGNES_HF"
  if [ -d "$agnes_src" ]; then rsync -a --info=progress2 "$agnes_src" "$BOX:.cache/huggingface/hub/"
  else echo "push-data: no $agnes_src on this Mac - download Agnes on the box (docs/box-day-plan.md)"; fi
  for d in oracle-out-agnes oracle-out-agnes-mtp oracle-out-ornith oracle-out-ornith-mtp oracle-out-ornith-a4 oracle-out-k2-a4 oracle-out-kolibri-a4; do
    if [ -d "$LOCAL_DATA/$d" ]; then rsync -a "$LOCAL_DATA/$d" "$BOX:$DATA_DIR/"; else echo "push-data: no $LOCAL_DATA/$d"; fi
  done
fi
if [ "$SYNC" = 1 ]; then tools/box.sh sync; fi
if [ "$("${SSH[@]}" "$BOX" "$BASE_CHECK_CMD")" != "$BASE_SHA" ]; then
  echo "baseline: extracting $BASELINE_REF ($BASE_SHORT) into ~/$BASE_DIR"
  git archive --format=tar "$BASE_SHA" | "${SSH[@]}" "$BOX" "$BASE_EXTRACT_CMD"
fi
echo "launch: $(echo "$SELECTED" | wc -w | tr -d ' ') stages, state ~/$STATE_REL"
tools/box.sh run "$LAUNCH_CMD"
if [ "$WAIT" = 0 ]; then
  echo "launched; re-run the same command to re-attach, --status to look, --summary to fetch"
  exit 0
fi
poll "${seen:-0}" || true
fetch_and_summarise
