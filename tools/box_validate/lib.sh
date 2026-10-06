# tools/box_validate/lib.sh - the stage registry and the helpers stage bodies use.
#
# Sourced in two places:
#   * on the Mac by tools/box_validate.sh, for --list and --dry-run (DRY=1: every helper
#     prints the command it would run on the box and runs nothing);
#   * on the box by tools/box_validate/remote.sh, which runs the stages (DRY=0).
# The same stage bodies serve both, so a dry-run prints exactly the command lines the box
# will run. Bash 3.2 compatible (the Mac's /bin/bash): no associative arrays, no mapfile.
#
# Globals a stage body can read (set by remote.sh, or by the driver for a dry-run):
#   TREE   the tree under test on the box (absolute; "$HOME/..." in a dry-run)
#   BASE   the baseline tree (G0's reference build)
#   STATE  this run's state directory (logs, status lines, JUnit files)
#   STAGE  the stage being run; LOG its log file
#   B70_GIT_SHA, BV_BASE_SHA, JOBS, CMAKE_ARGS, SNAP_QWEN, SNAP_AGNES, SNAP_ORNITH, SNAP_K2, SNAP_KOLIBRI,
#   PORT, ...

# ---- the registry ---------------------------------------------------------------------
BV_IDS=(); BV_ROWS=(); BV_SELS=(); BV_KINDS=(); BV_NEEDS=(); BV_AFTERS=(); BV_TITLES=()
BV_ROW_IDS=(); BV_ROW_TITLES=(); BV_NOTES=()

# stage ID ROW SEL KIND NEEDS AFTER TITLE
#   SEL   always  - runs on every launch, never skipped as already PASS (the preflight)
#         g0      - the neutrality gate; a non-PASS here blocks every later stage
#         default - runs unless deselected
#         optin   - runs only when named (--with / --only)
#         manual  - never runs: its body prints the commands for the operator
#   KIND  cpu      - no device; runs without the GPU lock
#         gpu      - runs holding ~/b70-gpu.lock for the whole stage
#         gpu-self - runs a script that takes ~/b70-gpu.lock itself (holding it here would
#                    deadlock: flock on a second open of the same file blocks)
#   NEEDS data keys from tools/box_validate/data.sh (comma list, or -)
#   AFTER stages that must be PASS first (comma list, or -)
stage() {
  BV_IDS+=("$1"); BV_ROWS+=("$2"); BV_SELS+=("$3"); BV_KINDS+=("$4")
  BV_NEEDS+=("$5"); BV_AFTERS+=("$6"); BV_TITLES+=("$7")
}
# row N TITLE - a row of docs/superpowers/plans/box-validation-queue.md (0: the gate itself)
row() { BV_ROW_IDS+=("$1"); BV_ROW_TITLES+=("$2"); }
# rownote N TEXT - something the row asks for that no stage runs, and why
rownote() { BV_NOTES+=("$1	$2"); }

bv_index() {
  local i
  for i in "${!BV_IDS[@]}"; do
    if [ "${BV_IDS[$i]}" = "$1" ]; then echo "$i"; return 0; fi
  done
  return 1
}
bv_fn() { echo "st_$(printf '%s' "$1" | tr '.-' '__')"; }
bv_field() {   # bv_field ID FIELD  (row sel kind needs after title)
  local i; i=$(bv_index "$1") || return 1
  case "$2" in
    row) echo "${BV_ROWS[$i]}" ;; sel) echo "${BV_SELS[$i]}" ;; kind) echo "${BV_KINDS[$i]}" ;;
    needs) echo "${BV_NEEDS[$i]}" ;; after) echo "${BV_AFTERS[$i]}" ;; title) echo "${BV_TITLES[$i]}" ;;
  esac
}

# bv_defaults - the box-side defaults every stage reads. HOME_R is the box's $HOME (the
# literal string '$HOME' in a dry-run on the Mac, so the printed paths are the box's).
bv_defaults() {
  local hf="${HF_HOME_R:-$HOME_R/.cache/huggingface}"
  # The gate checkpoint's known-good snapshot, bypassing refs/main - tests/CMakeLists.txt's
  # B70_TEST_SNAPSHOT default (ruling A31). Agnes and Ornith as their registrations name them.
  : "${SNAP_QWEN:=$hf/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d}"
  : "${SNAP_AGNES:=urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ}"
  : "${SNAP_ORNITH:=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ}"
  : "${SNAP_K2:=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ}"   # B70_K2_SNAPSHOT
  : "${SNAP_KOLIBRI:=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ}"   # B70_KOLIBRI_SNAPSHOT (spec 20b)
  : "${TOK_PYTHON:=$HOME_R/auto-round/.venv/bin/python}"
  : "${JOBS:=44}"
  : "${DEVICE:=0}"
  : "${PORT:=8013}"
  : "${CMAKE_ARGS:=}"
  : "${B70_GIT_SHA:=unknown}"
  export SNAP_QWEN SNAP_AGNES SNAP_ORNITH SNAP_K2 SNAP_KOLIBRI TOK_PYTHON JOBS DEVICE PORT CMAKE_ARGS B70_GIT_SHA
}

# status_of ID - this run's recorded result of a stage (empty if none)
status_of() {
  if [ "${DRY:-0}" = 1 ]; then echo PASS; return 0; fi
  cut -f2 "$STATE/$1.status" 2>/dev/null
}
# need_pass ID... - a recap stage: each named stage must have PASSed in this run
need_pass() {
  local s r
  for s in "$@"; do
    r=$(status_of "$s")
    if [ "${DRY:-0}" = 1 ]; then printf '    (requires %s PASS in this run)\n' "$s"; continue; fi
    if [ "$r" = PASS ]; then BV_OK=$((BV_OK + 1)); printf 'NOTE %s: PASS\n' "$s"
    else bad "$s is ${r:-not run}"; fi
  done
}
# need_ok ID... - a recap stage over stages that may lack their data: PASS counts, FAIL fails,
# a SKIP (missing data, blocked) or a stage not run in this state is noted - not evidence,
# not a failure
need_ok() {
  local s r
  for s in "$@"; do
    r=$(status_of "$s")
    if [ "${DRY:-0}" = 1 ]; then printf '    (requires %s not FAIL in this run; SKIP is noted)\n' "$s"; continue; fi
    case "$r" in
      PASS) BV_OK=$((BV_OK + 1)); printf 'NOTE %s: PASS\n' "$s" ;;
      FAIL|RUNNING) bad "$s is $r" ;;
      *) printf 'NOTE %s: %s - not evidence either way (%s)\n' "$s" "${r:-not run}" \
           "$(cut -f7 "$STATE/$s.status" 2>/dev/null)" ;;
    esac
  done
}

# ---- running commands -----------------------------------------------------------------
BV_BAD=0; BV_OK=0; BV_SKIPS=0
_bv_ts() { date '+%F %T'; }

# say "<line>" - a command for the operator (manual stages): printed, never run
say() { printf '    $ %s\n' "$*"; }

# x "<command line>" - run one shell command line in the tree under test (bash -o pipefail),
# its output into the stage log. The line is expanded by the caller, so what a dry-run
# prints is what the box runs.
x() { _bv_run "$TREE" "$1"; }
# xb "<command line>" - the same in the baseline tree.
xb() { _bv_run "$BASE" "$1"; }
_bv_run() {
  if [ "${DRY:-0}" = 1 ]; then
    printf '    [%s] %s\n' "$1" "$2"
    return 0
  fi
  printf '\n+ [%s] (%s) %s\n' "$(_bv_ts)" "$1" "$2"
  (cd "$1" && bash -o pipefail -c "$2")
  local rc=$?
  printf '+ rc=%s\n' "$rc"
  return $rc
}

note() {
  if [ "${DRY:-0}" = 1 ]; then printf '    (note: %s)\n' "$*"; return 0; fi
  printf 'NOTE %s\n' "$*"
}
bad() { BV_BAD=$((BV_BAD + 1)); printf 'BAD %s\n' "$*"; }
# step_rc RC WHAT - account one step: 0 passed, 77 skipped, anything else failed
step_rc() {
  case "$1" in
    0) BV_OK=$((BV_OK + 1)) ;;
    77) BV_SKIPS=$((BV_SKIPS + 1)); printf 'SKIP_REASON %s: skipped\n' "$2" ;;
    *) bad "$2 (rc $1)" ;;
  esac
}
# chk "<command line>" WHAT - x, then account it
chk() { x "$1"; step_rc $? "$2"; }
# finish - the stage's verdict from its steps: any failure FAIL (1), only skips SKIP (77)
finish() {
  if [ "${DRY:-0}" = 1 ]; then return 0; fi
  if [ "$BV_BAD" -gt 0 ]; then return 1; fi
  if [ "$BV_OK" -eq 0 ] && [ "$BV_SKIPS" -gt 0 ]; then return 77; fi
  return 0
}
skip() {
  if [ "${DRY:-0}" = 1 ]; then printf '    (skip: %s)\n' "$*"; return 0; fi
  printf 'SKIP_REASON %s\n' "$*"; BV_SKIPS=$((BV_SKIPS + 1)); return 77
}

# ---- recording numbers ----------------------------------------------------------------
# grab LABEL ERE - the last line of this stage's log matching ERE becomes "METRIC LABEL<TAB>line"
grab() {
  if [ "${DRY:-0}" = 1 ]; then printf '    (record %s: the last log line matching /%s/)\n' "$1" "$2"; return 0; fi
  local line
  line=$(grep -E -- "$2" "$LOG" 2>/dev/null | grep -vE '^(\+ |METRIC |NOTE |BAD )' | tail -1)
  printf 'METRIC %s\t%s\n' "$1" "${line:-(not printed)}"
}
# grab_all LABEL ERE [MAX] - every matching line (at most MAX, default 40)
grab_all() {
  if [ "${DRY:-0}" = 1 ]; then printf '    (record %s: every log line matching /%s/)\n' "$1" "$2"; return 0; fi
  local n=0 line
  while IFS= read -r line; do
    printf 'METRIC %s\t%s\n' "$1" "$line"; n=$((n + 1))
  done < <(grep -E -- "$2" "$LOG" 2>/dev/null | grep -vE '^(\+ |METRIC |NOTE |BAD )' | head -"${3:-40}")
  [ "$n" -gt 0 ] || printf 'METRIC %s\t(not printed)\n' "$1"
}
# jgrab LABEL TEST_ERE PATTERN_ERE - lines of a test's output (from this run's JUnit files)
jgrab() {
  x "python3 tools/box_validate/junit.py grep --junit \"$STATE/*.junit.xml\" --test '$2' --pattern '$3' --label '$1'"
}
idle() {   # idle TAG - the record/iterate grade at this moment (docs/10-the-box.md)
  x "tools/box_validate/idle.sh $1"
}

# ---- ctest ----------------------------------------------------------------------------
# The test output is kept in full in the JUnit file (ctest truncates passed tests' output
# to 1 KiB by default), which is what G0's bitwise comparison and the numbers read.
BV_CTEST_FLAGS="--output-on-failure --no-tests=error --test-output-size-passed 200000000 --test-output-size-failed 200000000"
ctest_cmd() { echo "ctest --test-dir build $BV_CTEST_FLAGS --output-junit $STATE/$STAGE${1:-}.junit.xml"; }
ctest_cmd_base() { echo "ctest --test-dir build $BV_CTEST_FLAGS --output-junit $STATE/$STAGE${1:-}.baseline.xml"; }

# run_tests ERE [LABELS] [NOT_LABELS] [ENV] - the registered tests whose names match ERE (and
# carry every label of LABELS, none of NOT_LABELS); ENV (`VAR=value ...`) is set for the ctest
# run (B70_LONGCTX_TESTS=1 for the `longctx` variants, which SKIP without it). Tests that
# already have a result in this run (the G0 suite ran most of the old ones) are taken from it;
# the rest run now. The verdict is over all of them: a failure fails, a test with no result
# fails, all skipped is a skip.
BV_CT_N=0
run_tests() {
  local re="$1" filt="" l sfx="" envp="${4:+env $4 }"
  for l in ${2:-}; do filt="$filt --label $l"; done
  for l in ${3:-}; do filt="$filt --no-label $l"; done
  # one JUnit file per call: a second run_tests in the same stage must not overwrite the first
  BV_CT_N=$((BV_CT_N + 1))
  [ "$BV_CT_N" -gt 1 ] && sfx=".$BV_CT_N"
  x "ctest --test-dir build --show-only=json-v1 > $STATE/tests.json"
  x "re=\$(python3 tools/box_validate/junit.py todo $STATE/tests.json --re '$re'$filt --done \"$STATE/*.junit.xml\"); if [ -z \"\$re\" ]; then echo 'every selected test already has a result in this run'; else $envp$(ctest_cmd "$sfx") -R \"\$re\"; fi"
  x "python3 tools/box_validate/junit.py pick $STATE/tests.json --re '$re'$filt --junit \"$STATE/*.junit.xml\""
  step_rc $? "tests /$re/$filt"
}

# xfail "<command line>" ERE WHAT - the command must refuse: exit non-zero, not by a signal
# or a timeout (rc < 124), and print ERE.
xfail() {
  if [ "${DRY:-0}" = 1 ]; then
    printf '    [%s] %s\n      (must exit non-zero without a crash, printing /%s/)\n' "$TREE" "$1" "$2"
    return 0
  fi
  local out="$STATE/$STAGE.refusal.$BV_OK.$BV_BAD.txt" rc
  x "$1 > $out 2>&1"; rc=$?
  tail -5 "$out"
  if [ "$rc" -eq 0 ]; then bad "$3: exited 0 - expected a refusal"
  elif [ "$rc" -ge 124 ]; then bad "$3: rc $rc (crash or timeout, not a refusal)"
  elif ! grep -qE -- "$2" "$out"; then bad "$3: refused (rc $rc) without the expected message /$2/"
  else BV_OK=$((BV_OK + 1)); printf 'METRIC refusal\t%s: %s\n' "$3" "$(grep -E -- "$2" "$out" | head -1)"
  fi
}

# kbins NAME... - these kernel binaries are in the build under test (cmake/ocloc.cmake:
# build/kernels/<name>.bin), each printed with whether G0's g0.sha listed it as added (new
# since the baseline); one missing fails the step. For a row whose list binds a binary by a
# name the host tests only check as a string (e.g. Ornith's int4 a||b gemv_M<M>_K2048_N128_S1_L1).
kbins() {
  chk "miss=0; for b in $*; do if [ -s build/kernels/\$b.bin ]; then echo \"built \$b (\$(grep -cxF kernels/\$b.bin $STATE/g0-sha/added.txt 2>/dev/null) in the G0 added list)\"; else echo \"NOT BUILT \$b\"; miss=1; fi; done; [ \$miss = 0 ]" \
    "kernel binaries built ($# names)"
}

# serve PORT_ARGS... - start b70-serve, prove it serves one request, stop it (serve_probe.sh)
serve() { x "PORT=$PORT tools/box_validate/serve_probe.sh $*"; step_rc $? "b70-serve $*"; }

# serve_arms DIR ROUNDS SNAP "CLIENT ARGS" ARM... - a server row: one b70-serve per arm and
# round (tools/box_validate/serve_run.sh: start, serve_client.py over the prompt sets, stop),
# rounds in rotated order (odd rounds as given, even rounds reversed). ARM is
# "label|b70-serve flags"; each arm-round's requests land in DIR/<label>.r<round>/.
# serve_client.py table / compare read DIR afterwards.
serve_arms() {
  local dir="$1" rounds="$2" snap="$3" cargs="$4" r i n lab flags
  shift 4
  local arms=("$@") order=()
  n=${#arms[@]}
  for r in $(seq 1 "$rounds"); do
    order=()
    for i in $(seq 0 $((n - 1))); do
      if [ $((r % 2)) = 1 ]; then order+=("${arms[$i]}"); else order+=("${arms[$((n - 1 - i))]}"); fi
    done
    for i in "${order[@]}"; do
      lab="${i%%|*}"; flags="${i#*|}"
      chk "PORT=$PORT tools/box_validate/serve_run.sh $dir/$lab.r$r $lab $r $snap $flags -- $cargs" "arm $lab, round $r"
    done
  done
}

# bench_cmd SNAP ARGS... - a b70-decode bench line carrying the Mac's sha into the row
bench_cmd() { local s="$1"; shift; echo "B70_GIT_SHA=$B70_GIT_SHA build/src/cli/b70-decode $s --bench $*"; }
