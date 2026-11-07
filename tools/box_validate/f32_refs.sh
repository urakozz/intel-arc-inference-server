#!/usr/bin/env bash
# The f32 references beyond f32_oracle.sh's golden prompts (operator ruling 2026-10-29), ON THE BOX CPU, each run
# through tools/box_validate/f32_ref_job.sh (one CPU model job at a time, RAM checked, resumable):
#   tools/box_validate/f32_refs.sh <data tree> ornith-mtp
#       r16.m1's judge: ornith_ref.py --dtype f32 --mtp-out, per golden prompt, teacher-forced on the bf16 M1
#       reference's continuation (oracle-out-ornith-mtp/<p>.cont256.ids, its first 32) ->
#       oracle-out-ornith-mtp-f32/m1/<p>.mtp.safetensors (mtp_head_test's format) + <p>.golden.safetensors
#   tools/box_validate/f32_refs.sh <data tree> forced <ornith|k2> <dir> [<suffix> [extra ref.py flags...]]
#       every <dir>/<key>.prompt.ids with its <key>.force.ids (a test's own engine run wrote them:
#       B70_DUMP_FORCED=<dir>, tests/golden/forced_judge.h) -> <data tree>/oracle-out-<model>-f32<suffix>-forced/
#       <key>.golden.safetensors, run --lean (e.g. suffix -i8head with --lm-head int8: the int8 head's judges)
#       DTYPE=bf16 ... '' --gemm f32: each judge's bf16 twin -> oracle-out-<model>-bf16-forced/ (operator ruling
#       2026-10-31: the near-miss gates' bar is the twin's own count of ids the judge rejects)
#   tools/box_validate/f32_refs.sh <data tree> split <ornith|k2> <ids> <N> <out stem>
#       prefill_split_test's references of ids[0, N): f32 and bf16, --lean --dump-kv, no generated rows ->
#       <out stem>-f32.golden.safetensors and <out stem>-bf16.golden.safetensors
#   tools/box_validate/f32_refs.sh <data tree> forced-qwen <dir> [<dir>...]
#       Qwen3.8's spec tests (spec 28 N4 / N6, tests/server/spec_judge.h: mtp_verify_test, mtp_gpu_test and its
#       twins, the DFlash tests): every Qwen3.8 context of the dirs (keys QWEN_KEYS_RE; a key in two dirs is one
#       context) through tools/oracle/qwen_forced.py - ONE layer-major f32 forward over the contexts' token trie,
#       not a run a context (~250 contexts / ~450k ids are ~48k distinct positions) -> <data tree>/
#       oracle-out-qwen-f32-forced/ (the checkpoint's bf16 head) and oracle-out-qwen-f32-i8head-forced/ (the int8
#       head: the DFlash tests and mtp_gpu_test --lm-head int8, qwen_forced.py's I8HEAD_RE). Two passes, each under
#       /tmp/b70-cpu-model.lock: the contexts QWEN_FIRST_RE names first (mtp_verify_test's and dflash_engine_test's:
#       ~2.4k positions, minutes), then the rest; a killed pass resumes at its layer (the partial dir's checkpoint).
#   tools/box_validate/f32_refs.sh <data tree> forced-qwen-pfi8 <dir> [<dir>...]
#       the l0-int8 tests' judges (mtp_gpu_test and its twins prefill with l0-int8; operator ruling 2026-10-23):
#       every mtp_gpu_ context of the dirs through qwen_forced.py --pfi8-re (the test's own prompt rows through
#       int8_sim fq_x / fq_w on the h8 linears, decode rows plain), one pass, both heads into ONE dir (the keys carry
#       the head form) -> <data tree>/oracle-out-qwen-f32-pfi8-forced/
#   tools/box_validate/f32_refs.sh <data tree> bf16top-qwen <dir> [<dir>...]
#       the bf16-unstable rule's standard-bf16 top-k references (operator 2026-11-04, tests/server/spec_judge.h;
#       B70_FORCED_BF16): tools/oracle/qwen_bf16_top.py plan on the host - the Qwen3.8 contexts of the dirs whose
#       judge rejects a forced id by more than 8 bf16 ulp, each against the judge its test reads (mtp_gpu_ keys: the
#       l0-int8 judges of oracle-out-qwen-f32-pfi8-forced; the rest: oracle-out-qwen-f32{,-i8head}-forced) - then
#       qwen_bf16_top.py run in the oracle container under /tmp/b70-cpu-model.lock (its memory cap:
#       run_in_container.sh, ORACLE_MEMORY) -> <data tree>/oracle-out-qwen-bf16top-forced/<key>.bf16top.safetensors.
#       Resumable: keys already there are skipped. Run forced-qwen / forced-qwen-pfi8 first (no judge, no plan).
set -u
cd "$(dirname "$0")/../.." || exit 2
data="${1:?data tree}" what="${2:?ornith-mtp | forced}"
rc=0
case "$what" in
  ornith-mtp)
    src="$data/oracle-out-ornith-mtp" out="$data/oracle-out-ornith-mtp-f32"
    mkdir -p "$out/m1" oracle-out-f32-job.partial
    for p in prose code cjk; do
      if [ -s "$out/m1/$p.mtp.safetensors" ]; then echo "f32_refs: $p kept"; continue; fi
      [ -s "$src/$p.cont256.ids" ] || { echo "f32_refs: $src/$p.cont256.ids is missing"; rc=1; continue; }
      tr -s ' \n' '\n' < "$src/$p.cont256.ids" | grep . | head -32 | tr '\n' ' ' > "oracle-out-f32-job.partial/$p.mtp.force"
      rm -f "$out/$p.golden.safetensors"   # the M1 file is made by the same run
      tools/box_validate/f32_ref_job.sh ornith "tests/golden/prompts/$p.ids" "oracle-out-f32-job.partial/$p.mtp.force" \
        "$out/$p.golden.safetensors" --mtp-out @PART@/mtp || { rc=1; continue; }
      mv -f oracle-out-f32-job.partial/mtp/m1/prompt.mtp.safetensors "$out/m1/$p.mtp.safetensors" || rc=1
      cp -f "$src/$p.cont256.ids" "$out/$p.cont256.ids"
    done
    ;;
  forced)
    m="${3:?ornith | k2}" dir="${4:?the dumped prompt / force ids}" suffix="${5:-}"
    shift 4; [ $# -gt 0 ] && shift
    out="$data/oracle-out-$m-${DTYPE:-f32}$suffix-forced"   # DTYPE=bf16: the judges' bf16 twins (B70_FORCED_BF16)
    for pr in "$dir"/*.prompt.ids; do
      [ -e "$pr" ] || { echo "f32_refs: no *.prompt.ids in $dir"; rc=1; break; }
      n=$(basename "$pr" .prompt.ids)
      tools/box_validate/f32_ref_job.sh "$m" "$pr" "$dir/$n.force.ids" "$out/$n.golden.safetensors" --lean "$@" || rc=1
    done
    ;;
  split)
    m="${3:?ornith | k2}" ids="${4:?ids}" n="${5:?N}" stem="${6:?out stem}"
    mkdir -p oracle-out-f32-job.partial
    tr -s ' \n' '\n' < "$ids" | grep . | head -"$n" | tr '\n' ' ' > oracle-out-f32-job.partial/split.prompt
    : > oracle-out-f32-job.partial/split.none
    tools/box_validate/f32_ref_job.sh "$m" oracle-out-f32-job.partial/split.prompt oracle-out-f32-job.partial/split.none \
      "$stem-f32.golden.safetensors" --lean --dump-kv || rc=1
    DTYPE=bf16 tools/box_validate/f32_ref_job.sh "$m" oracle-out-f32-job.partial/split.prompt \
      oracle-out-f32-job.partial/split.none "$stem-bf16.golden.safetensors" --lean --dump-kv || rc=1
    ;;
  forced-qwen|forced-qwen-pfi8)
    shift 2
    [ $# -gt 0 ] || { echo "f32_refs: $what <dir> [<dir>...]" >&2; exit 2; }
    out="$data/oracle-out-qwen-f32-forced" out8="$data/oracle-out-qwen-f32-i8head-forced"
    part="oracle-out-qwen-forced.partial"   # in the tree: the container sees only the tree (at /ws)
    keys="${QWEN_KEYS_RE:-^(mtp_verify_qwen3\.8_|mtp_gpu_|dflash_)}" pfi8=""
    if [ "$what" = forced-qwen-pfi8 ]; then
      out="$data/oracle-out-qwen-f32-pfi8-forced" out8="$out" part="oracle-out-qwen-pfi8-forced.partial"
      keys="^mtp_gpu_" pfi8="--pfi8-re '^mtp_gpu_'" QWEN_FIRST_RE=""
    fi
    mkdir -p "$part/ctx" "$part/out" "$part/out-i8head" "$out" "$out8"
    for d in "$@"; do
      for f in "$d"/*.prompt.ids; do
        [ -e "$f" ] || { echo "f32_refs: no *.prompt.ids in $d"; rc=1; break; }
        k=$(basename "$f" .prompt.ids)
        printf '%s\n' "$k" | grep -Eq "$keys" || continue
        [ -e "$part/ctx/$k.prompt.ids" ] && continue   # qwen_forced.py checks each key's hash against its ids
        cp "$f" "$part/ctx/$k.prompt.ids" && cp "${f%.prompt.ids}.force.ids" "$part/ctx/$k.force.ids" || rc=1
      done
    done
    echo "f32_refs: $(ls "$part/ctx" | grep -c '\.prompt\.ids$') Qwen3.8 contexts in $part/ctx"
    exec 9> /tmp/b70-cpu-model.lock
    for pass in first rest; do
      re=""
      if [ "$pass" = first ]; then
        re="${QWEN_FIRST_RE-^(mtp_verify|dflash_engine)_}"
        [ -n "$re" ] || continue
      fi
      echo "f32_refs: forced-qwen pass $pass${re:+ ($re)}: waiting for /tmp/b70-cpu-model.lock ($(date '+%F %T'))"
      [ -n "${B70_CPU_MODEL_LOCK_HELD:-}" ] || flock 9   # the CPU chain holds it for its job already
      avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
      if [ "${avail:-0}" -lt "${F32_REF_MIN_GB:-95}" ]; then
        echo "SKIP_REASON f32_refs: MemAvailable ${avail} GB < F32_REF_MIN_GB ${F32_REF_MIN_GB:-95} (another job holds the RAM)"
        exit 77
      fi
      { ls "$out"; ls "$out8"; } | sort -u | sed -n 's/\.golden\.safetensors$//p' > "$part/done.txt"
      echo "f32_refs: pass $pass, MemAvailable ${avail} GB ($(date '+%F %T')); log $part/run.log"
      ORACLE_THREADS="${ORACLE_THREADS:-22}" \
        ORACLE_MODEL="${SNAP_QWEN_CACHE:-models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}" \
        tools/oracle/run_in_container.sh "python3 tools/oracle/qwen_forced.py run \"\$SNAP\" --ctx /ws/$part/ctx \
          --out /ws/$part/out --out-i8head /ws/$part/out-i8head --work /ws/$part/work --done /ws/$part/done.txt \
          ${re:+--match '$re'} $pfi8 >> /ws/$part/run.log 2>&1" ||
        { echo "f32_refs: pass $pass FAILED"; tail -20 "$part/run.log"; rc=1; }
      [ -n "${B70_CPU_MODEL_LOCK_HELD:-}" ] || flock -u 9
      for f in "$part"/out/*.golden.safetensors; do [ -e "$f" ] && mv -f "$f" "$out/"; done
      for f in "$part"/out-i8head/*.golden.safetensors; do [ -e "$f" ] && mv -f "$f" "$out8/"; done
      echo "f32_refs: pass $pass done ($(date '+%F %T')): $(ls "$out" | grep -c 'golden') bf16-head and" \
        "$(ls "$out8" | grep -c 'golden') int8-head judges"
      [ "$rc" = 0 ] || break
    done
    ;;
  bf16top-qwen)
    shift 2
    [ $# -gt 0 ] || { echo "f32_refs: bf16top-qwen <dir> [<dir>...]" >&2; exit 2; }
    out="$data/oracle-out-qwen-bf16top-forced"
    part="oracle-out-qwen-bf16top.partial"   # in the tree: the container sees only the tree (at /ws)
    mkdir -p "$part/ctx-pfi8" "$part/ctx" "$part/out" "$out"
    for d in "$@"; do
      for f in "$d"/*.prompt.ids; do
        [ -e "$f" ] || { echo "f32_refs: no *.prompt.ids in $d"; rc=1; break; }
        k=$(basename "$f" .prompt.ids)
        case "$k" in
          mtp_gpu_*) c="$part/ctx-pfi8" ;;
          mtp_verify_qwen3.8_*|dflash_*) c="$part/ctx" ;;
          *) continue ;;
        esac
        [ -e "$c/$k.prompt.ids" ] && continue
        cp "$f" "$c/$k.prompt.ids" && cp "${f%.prompt.ids}.force.ids" "$c/$k.force.ids" || rc=1
      done
    done
    [ "$rc" = 0 ] || exit "$rc"
    echo "f32_refs: bf16top-qwen: $(ls "$part/ctx-pfi8" | grep -c '\.prompt\.ids$') l0-int8 and" \
      "$(ls "$part/ctx" | grep -c '\.prompt\.ids$') other Qwen3.8 contexts"
    # the plan on the host (numpy; the judges are read where they are), one judge per key: the one its test reads
    python3 tools/oracle/qwen_bf16_top.py plan --ctx "$part/ctx-pfi8" --judge "$data/oracle-out-qwen-f32-pfi8-forced" \
      --out "$out" --plan "$part/plan-pfi8.json" &&
      python3 tools/oracle/qwen_bf16_top.py plan --ctx "$part/ctx" --judge "$data/oracle-out-qwen-f32-forced" \
        --judge "$data/oracle-out-qwen-f32-i8head-forced" --out "$out" --plan "$part/plan-rest.json" &&
      python3 -c 'import json, sys
sel = [s for p in sys.argv[2:] for s in json.load(open(p))]
json.dump(sel, open(sys.argv[1], "w"))
print(f"f32_refs: bf16top-qwen plan: {len(sel)} contexts")' "$part/plan.json" "$part/plan-pfi8.json" "$part/plan-rest.json" ||
      { echo "f32_refs: bf16top-qwen: the plan FAILED"; exit 1; }
    if [ "$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$part/plan.json")" = 0 ]; then
      echo "f32_refs: bf16top-qwen: nothing to do (every context with a miss beyond 8 ulp has its reference)"
      exit 0
    fi
    exec 9> /tmp/b70-cpu-model.lock
    echo "f32_refs: bf16top-qwen: waiting for /tmp/b70-cpu-model.lock ($(date '+%F %T'))"
    flock 9
    avail=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo 2>/dev/null || echo 0)
    if [ "${avail:-0}" -lt "${F32_REF_MIN_GB:-95}" ]; then
      echo "SKIP_REASON f32_refs: MemAvailable ${avail} GB < F32_REF_MIN_GB ${F32_REF_MIN_GB:-95} (another job holds the RAM)"
      exit 77
    fi
    echo "f32_refs: bf16top-qwen, MemAvailable ${avail} GB ($(date '+%F %T')); log $part/run.log"
    ORACLE_THREADS="${ORACLE_THREADS:-22}" \
      ORACLE_MODEL="${SNAP_QWEN_CACHE:-models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ}" \
      tools/oracle/run_in_container.sh "python3 tools/oracle/qwen_bf16_top.py run \"\$SNAP\" --plan /ws/$part/plan.json \
        --out /ws/$part/out >> /ws/$part/run.log 2>&1" ||
      { echo "f32_refs: bf16top-qwen FAILED"; tail -20 "$part/run.log"; rc=1; }
    flock -u 9
    for f in "$part"/out/*.bf16top.safetensors; do [ -e "$f" ] && mv -f "$f" "$out/"; done
    echo "f32_refs: bf16top-qwen done ($(date '+%F %T')): $(ls "$out" | grep -c 'bf16top') bf16 top-k references"
    ;;
  *) echo "f32_refs: ornith-mtp | forced <model> <dir> [suffix [flags]] | split <model> <ids> <N> <stem> | forced-qwen <dir>... | forced-qwen-pfi8 <dir>... | bf16top-qwen <dir>..." >&2; exit 2 ;;
esac
exit "$rc"
