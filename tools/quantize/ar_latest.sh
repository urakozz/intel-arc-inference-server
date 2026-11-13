# Sourced by quantize_qwen4exp.sh and quantize_dflash.sh before ar_pin.sh: the LATEST AutoRound of the operator's
# clone instead of a fixed pin (operator 2026-11-13). The Qwen3.8 scripts do not use this; they keep ar_pin.sh's pin.
#
# In:  AR_REPO the clone (default ~/auto-round); AR_REF the branch to build (default local-patches-v2 = upstream
#      intel/auto-round main + our one local commit: cached OpenCodeInstruct parquet, local-only snapshot load);
#      AR_COMMIT set by the caller: build exactly that commit (a run's recorded hash, or the old pin
#      a6cf748739f2dd58e6c16b0ac849e92af7396e32 on the branch pin-a6cf7487-2026-11-13)
# Out: AR_COMMIT (the 40-hex hash AR_REF had at this moment), AR_BASE (its first parent: the upstream main commit
#      under our local commit), AR_REF, AR_PATCH=0, AR_REPO - all exported, so the detached `run` process builds
#      the hash `start` resolved even if the branch moves in between. ar_pin.sh keys the package cache on the hash.
AR_REPO="${AR_REPO:-$HOME/auto-round}"
AR_REF="${AR_REF:-local-patches-v2}"
AR_EXPLICIT="${AR_COMMIT:-}"
ar_resolve() {
  [ -d "$AR_REPO/.git" ] || { echo "$AR_REPO is not a git clone" >&2; return 2; }
  if [ -z "$AR_EXPLICIT" ]; then
    AR_COMMIT=$(git -C "$AR_REPO" rev-parse --verify -q "$AR_REF^{commit}") ||
      { echo "$AR_REPO has no branch $AR_REF" >&2; return 2; }
  fi
  AR_BASE=$(git -C "$AR_REPO" rev-parse --verify -q "$AR_COMMIT^" || echo "?")
  export AR_COMMIT AR_BASE AR_REF AR_REPO AR_PATCH=0
}

# ar_run_guard <dir>: <dir>/ar_commit holds the hash the run started with. A run that has made progress
# (a .done-* stage or a resume/ directory) must not continue on another AutoRound: refuse, with the remedy.
ar_run_guard() {
  local d=$1 f=$1/ar_commit have
  if [ -f "$f" ]; then
    have=$(cut -d' ' -f1 "$f")
    [ "$have" = "$AR_COMMIT" ] || {
      echo "this run started on AutoRound $have; this call would build $AR_COMMIT ($AR_REF)." \
           "A run never switches AutoRound: resume with AR_COMMIT=$have, or start a new run (OUT=<new dir>)." >&2
      return 2; }
  elif [ -n "$(ls -A "$d"/resume "$d"/.done-* 2>/dev/null)" ] && [ -z "$AR_EXPLICIT" ]; then
    echo "$d has progress but no ar_commit record (a run from before AR_REF): resume it with AR_COMMIT=<the hash" \
         "it started on> (the old pin: a6cf748739f2dd58e6c16b0ac849e92af7396e32) so it is recorded." >&2
    return 2
  else
    mkdir -p "$d"; echo "$AR_COMMIT base $AR_BASE ref $AR_REF" > "$f"
  fi
}
