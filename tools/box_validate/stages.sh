# tools/box_validate/stages.sh - the box validation queue as stages.
#
# docs/superpowers/plans/box-validation-queue.md is the list of work merged into main
# without running on the card; each `row` below is one of its rows, each `stage` one step
# of that row's "how to validate", in the order the queue and its checklists require.
# Registry fields and helpers: tools/box_validate/lib.sh. A stage body runs ON THE BOX from
# the tree under test ($TREE); in a dry-run on the Mac every helper prints instead of running.
#
# Adding a row (a branch that lands blind, e.g. spec 15d / 18b):
#   1. `row N "title"` and one `stage rN.<step> N <sel> <kind> <needs> <after> "title"` per
#      step, after the last row (the order here is the run order);
#   2. a body `st_rN_<step>() { ...; finish; }` - usually `run_tests '<ctest regex>' [labels]`
#      for registered tests, `x "<command>"` / `chk` for anything else, `grab` for the numbers
#      the row asks for, `interleave.sh` for timed arms;
#   3. `rownote N "..."` for what the row asks that no stage can run, and why;
#   4. `tools/box_validate.sh --dry-run --only rN` prints what it will run;
#      `python3 tools/box_validate/test_box_validate.py` checks the registry.
# Stage ids are `<row>.<step>`: --only / --redo / --with accept an id or a row prefix (`r11`).

# What G0 compares OUTPUT-for-output against the baseline build (timings dropped): the gates
# the spec 14 and 15b checklists name as "bitwise identical to main's last record".
: "${G0_BITWISE_RE:=^(golden_gate_test|golden_gate_i8head_test|prefill_gate_l0_test|prefill_gate_int8_test|prefill_gate_l0_i8head_test|prefill_gate_int8_i8head_test|replay_determinism_test|replay_determinism_i8head_test|buffers_test|prefill_smoke_test|load_checkpoint_test)$}"
AGNES_ORACLE_MODEL=models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ

# ======================================================================================
row 0 "G0 - the gate: Qwen3.8 bitwise unchanged against the last box-validated main (spec 14 G0, spec 15b R0, spec 15c R0, spec 12b step 1)"

stage pre 0 always cpu - - "preflight: host and devices, idle state, data present, oracle-out* linked into both trees"
st_pre() {
  x "uptime; nproc; free -g | head -2; df -h \$HOME | tail -1"
  x "ls /dev/dri 2>/dev/null; command -v ocloc; command -v icpx || true"
  x "command -v xpu-smi > /dev/null && xpu-smi discovery 2>/dev/null | head -30 || echo 'no xpu-smi'"
  idle pre
  chk "tools/box_validate/data.sh link $DATA $TREE $BASE" "linking the oracle-out* data directories"
  chk "tools/box_validate/data.sh have > $STATE/have.env; cat $STATE/have.env" "listing the data present"
  grab_all data '^HAVE_'
  finish
}

stage g0.build 0 g0 cpu - - "build the baseline tree and the tree under test, Release, -j\$JOBS (the device code's first real compile)"
st_g0_build() {
  chk "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS > /dev/null && cmake --build build -j$JOBS" "build of the tree under test"
  x "cat $BASE/.b70-baseline-ref"
  xb "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS > /dev/null && cmake --build build -j$JOBS"
  step_rc $? "build of the baseline tree"
  finish
}

stage g0.sha 0 g0 cpu - g0.build "kernel binaries: sha256 of every binary present in both builds identical; the added ones listed"
st_g0_sha() {
  chk "tools/box_validate/g0_compare.sh $BASE/build build $STATE/g0-sha" "kernel binaries against the baseline"
  grab g0-kernels '^G0 kernels:'
  grab g0-verdict '^G0 (PASS|FAIL)'
  finish
}

stage g0.bitwise 0 g0 gpu qwen,oracle_qwen g0.build "the golden / prefill / replay gates, buffers_test, prefill_smoke_test, load_checkpoint_test on both builds: output identical line for line (timings dropped)"
st_g0_bitwise() {
  xb "$(ctest_cmd_base) -R '$G0_BITWISE_RE'"
  x "python3 tools/box_validate/junit.py verdict $STATE/g0.bitwise.baseline.xml"
  step_rc $? "the BASELINE build's own gates (a failure here is the box or the data, not the change)"
  x "$(ctest_cmd) -R '$G0_BITWISE_RE'"
  x "python3 tools/box_validate/junit.py verdict $STATE/g0.bitwise.junit.xml"
  step_rc $? "the gates on the tree under test"
  chk "python3 tools/box_validate/junit.py compare $STATE/g0.bitwise.baseline.xml $STATE/g0.bitwise.junit.xml --strip-a $BASE --strip-b $TREE --out $STATE/g0.bitwise.diff" \
    "gate output differs from the baseline's (g0.bitwise.diff)"
  grab compare '^COMPARE common='
  jgrab golden '^golden_gate' 'golden_gate_test OK|TOTAL:'
  jgrab replay '^replay_determinism' 'kernel_count|launches|bitwise'
  jgrab prefill-smoke '^prefill_smoke_test$' '8449|8705|launch'
  jgrab load '^load_checkpoint_test$' 'W check|849,?398,?784|MTP'
  finish
}

stage g0.suite 0 g0 gpu qwen,oracle_qwen g0.build "the baseline's full suite (every test registered at the baseline, Agnes / Ornith / kv8 labels aside) on the tree under test, B70_KV_CACHE unset"
st_g0_suite() {
  xb "ctest --test-dir build --show-only=json-v1 > $STATE/tests.baseline.json"
  x "ctest --test-dir build --show-only=json-v1 > $STATE/tests.json"
  x "re=\$(python3 tools/box_validate/junit.py todo $STATE/tests.json --also-in $STATE/tests.baseline.json --no-label agnes --no-label ornith --no-label kv8 --done \"$STATE/*.junit.xml\"); if [ -z \"\$re\" ]; then echo 'every selected test already has a result in this run'; else $(ctest_cmd) -R \"\$re\"; fi"
  chk "python3 tools/box_validate/junit.py pick $STATE/tests.json --also-in $STATE/tests.baseline.json --no-label agnes --no-label ornith --no-label kv8 --junit \"$STATE/*.junit.xml\"" \
    "the baseline's suite on the tree under test"
  finish
}

# ======================================================================================
row 1 "spec 14 - Agnes 3.0 Flash, ModelDesc (2026-10-03-spec14-validation-checklist.md)"
rownote 1 "Step 4's follow-up (replace P1-P3 in make_agnes() and add_gemv_variant when the sweep picks another {S, L}, rebuild, re-run G0) and step 12's record are edits on a branch: r1.sweep records the picks."
rownote 1 "Step 6's optional box re-dump of oracle-out-agnes (golden.sh, unstreamed, 61+ GiB) is not run: the gates use the Mac's layer-streamed sets."

stage r1.host 1 default cpu - - "step 3: host and table tests (model_desc, agnes_fold, kernel_table, template x2, qwen35)"
st_r1_host() {
  run_tests '^(model_desc_test|agnes_fold_test|kernel_table_test|template_agnes_test|template_test|qwen35_test)$'
  finish
}

stage r1.load 1 default gpu agnes - "step 3: load_agnes_test (72 layers, fold, MTP head, 262400 refused; W check within 2% - P4)"
st_r1_load() {
  run_tests '^load_agnes_test$'
  jgrab load-agnes '^load_agnes_test$' 'W check|load_agnes_test OK|refus'
  finish
}

stage r1.parity 1 default gpu agnes,tok_python - "step 3 (P5): parity_test against Agnes's tokenizer.json (corpus re-dumped with the reference tokenizers)"
st_r1_parity() {
  x "mkdir -p $STATE/agnes-parity/tests/tokenizer && cp tests/tokenizer/corpus.txt $STATE/agnes-parity/tests/tokenizer/"
  chk "cd $STATE/agnes-parity && $TOK_PYTHON $TREE/tools/tokenizer/dump_parity.py \$($TREE/tools/box_validate/data.sh resolve $SNAP_AGNES)/tokenizer.json" "corpus.ids from Agnes's tokenizer (Python tokenizers)"
  chk "B70_TOKENIZER_JSON=\$(tools/box_validate/data.sh resolve $SNAP_AGNES)/tokenizer.json build/tests/parity_test $STATE/agnes-parity/tests/tokenizer" "parity_test on Agnes's tokenizer.json"
  grab parity 'cases|parity|OK'
  finish
}

stage r1.sweep_build 1 default cpu - g0.build "step 4 (P1-P3): build the GEMV sweep variants (-DB70_AGNES_SWEEP=ON) in build-agnes-sweep"
st_r1_sweep_build() {
  chk "cmake -S . -B build-agnes-sweep -DCMAKE_BUILD_TYPE=Release -DB70_AGNES_SWEEP=ON $CMAKE_ARGS > /dev/null && cmake --build build-agnes-sweep -j$JOBS" "the sweep build"
  finish
}

stage r1.sweep 1 default gpu - r1.sweep_build "step 4 (P1-P3): probe_gemv --agnes - the {S, L} picks for 5120x38912 and 19456x5120, GB/s (timed)"
st_r1_sweep() {
  idle before
  chk "build-agnes-sweep/tools/probe/probe_gemv --agnes" "probe_gemv --agnes"
  idle after
  grab_all sweep-decisions 'decision|pick|chosen|best' 20
  grab_all idle '^IDLE '
  finish
}

stage r1.kernels 1 default gpu - - "step 5 (G1, card half): gemv / prep / pf_dequant_slab / pf_int8 / argmax tests at the new shapes"
st_r1_kernels() {
  run_tests '^(gemv_test|prep_test|pf_dequant_slab_test|pf_int8_test|argmax_test)$'
  finish
}

stage r1.gates 1 default gpu agnes,oracle_agnes - "step 6 (G2): golden_gate_agnes (+i8head), prefill_gate_agnes l0 / int8 - 870 launches (P8), tie rule"
st_r1_gates() {
  run_tests 'golden_gate_agnes|prefill_gate_agnes' agnes
  jgrab agnes-golden 'golden_gate_agnes|prefill_gate_agnes' 'golden_gate_test OK|TOTAL:|kernel_count|870|PASS|FAIL'
  finish
}

stage r1.features 1 default gpu agnes - "step 7 (G4): replay bitwise, prefill split, snapshot C1, prefix C2, MTP verify M2 on Agnes"
st_r1_features() {
  run_tests 'replay_determinism_agnes|prefill_split_agnes|snapshot_agnes|prefix_gpu_agnes|mtp_verify_agnes' agnes
  finish
}

stage r1.mtp_head 1 default gpu agnes,oracle_agnes_mtp - "step 7 (G4): mtp_head_agnes_test (M1) against oracle-out-agnes-mtp"
st_r1_mtp_head() {
  run_tests '^mtp_head_agnes_test$' agnes
  finish
}

stage r1.passkey 1 default gpu agnes,oracle_image - "step 8 (G5): passkey 3/3 at ~60k on l0-int8 and l0 (max_len 65536)"
st_r1_passkey() {
  chk "MODEL=$SNAP_AGNES ORACLE_MODEL=$AGNES_ORACLE_MODEL MAX_LEN=65536 N_TARGET=60000 tools/probe/passkey.sh l0-int8 l0" "passkey on Agnes"
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  finish
}

stage r1.speed 1 default gpu agnes - "step 9 (P7): decode at 4k / 32k / 60k and pp4096 at max_len 65536, bf16 and int8 head (interleaved, median of 3; derived ~26 t/s, ~1850-1900 t/s)"
st_r1_speed() {
  idle before
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio 'pp4096-int8/pp4096' -- \
pp4096 '$(bench_cmd "$SNAP_AGNES" --pp 4096 --tg 256 --max-len 65536)' \
pp4096-int8 '$(bench_cmd "$SNAP_AGNES" --pp 4096 --tg 256 --max-len 65536 --lm-head int8)' \
pp32768 '$(bench_cmd "$SNAP_AGNES" --pp 32768 --tg 256 --max-len 65536)' \
pp60000 '$(bench_cmd "$SNAP_AGNES" --pp 60000 --tg 256 --max-len 65536)'" "the Agnes speed arms"
  idle after
  grab_all memory '^memory:' 4
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}

stage r1.a4 1 manual - agnes - "step 9 (G3): A4 on Agnes against its bf16 reference (oracle in the container, hours of CPU)"
st_r1_a4() {
  say "tools/oracle/run_in_container.sh 'python3 tools/toolcall/oracle_generate.py <Agnes-AI/Agnes-3.0-Flash snapshot> tests/golden/toolcall-agnes /scratch/agnes-toolcall-ref'"
  say "flock ~/b70-gpu.lock env ZE_AFFINITY_MASK=0 tools/toolcall/engine_generate.sh $SNAP_AGNES tests/golden/toolcall-agnes ~/agnes-toolcall-out l0-int8"
  say "python3 tools/toolcall/score.py ~/agnes-toolcall-out bf16 l0-int8      # recorded beside Qwen3.8's 25/36 (no bar)"
}

stage r1.prefix_benchy 1 optin gpu-self agnes,uvx - "step 11: llama-benchy prefix caching at depth 4k / 16k / 32k, cache on vs off, through b70-serve"
st_r1_prefix_benchy() {
  chk "MODEL=$SNAP_AGNES MAX_LEN=65536 SERVE_ARGS='--mtp 0' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, cache on"
  chk "MODEL=$SNAP_AGNES MAX_LEN=65536 SERVE_ARGS='--mtp 0 --prefix-cache-gb 0' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, cache off"
  grab_all benchy '^\| ' 60
  finish
}

stage r1.vllm 1 manual - agnes - "step 10: the vLLM row (PR #57003 overlay) against b70-serve byte-matched and at defaults"
st_r1_vllm() {
  say "# the box's vLLM XPU image with PR #57003's eight files overlaid (gh pr diff 57003), --language-model-only --max-model-len 65536"
  say "uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model <agnes> --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation"
  say "MODEL=$SNAP_AGNES MAX_LEN=65536 SERVE_ARGS='--lm-head bf16 --mtp 0' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation"
  say "MODEL=$SNAP_AGNES MAX_LEN=65536 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation"
  say "# never both servers on the card at once; read the vLLM version string the server prints"
}

# ======================================================================================
row 2 "spec 13c Task 1 (step_many signature) and spec 8 §10 (--mtp auto)"
rownote 2 "D1-style rows for --mtp auto against fixed K = 1 / 3 need a client that drives b70-serve per request (AdaptiveK lives in the server; mtp_gpu_test --bench takes a fixed K): r2.auto_rows prints the hand commands."

stage r2.suite 2 default gpu qwen,oracle_qwen - "the suite around the changed signature: mtp_gpu, mtp_server, golden_server, prefix_gpu_*, mtp_verify, scheduler, adaptive_k"
st_r2_suite() {
  run_tests '^(mtp_gpu_test|mtp_server_test|golden_server_test|prefix_gpu_.*|mtp_verify_test|scheduler_test|adaptive_k_test)$' '' 'kv8 agnes'
  finish
}

stage r2.cost 2 default gpu qwen - "the int8-head --mtp auto cost table: probe_mtp_steps (verify M = 1..4, draft k = 1..3) for MtpCost's defaults (timed)"
st_r2_cost() {
  idle before
  chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 4096 32 3 int8" "probe_mtp_steps, int8 head"
  chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 4096 32 3 bf16" "probe_mtp_steps, bf16 head (the measured table it is compared with)"
  idle after
  grab_all cost-rows '^\| ' 40
  grab_all idle '^IDLE '
  finish
}

stage r2.auto_rows 2 manual - qwen - "D1-style rows: b70-serve --mtp auto against --mtp 1 and --mtp 3 on the golden and A4 prompts"
st_r2_auto_rows() {
  say "# one server per arm, under the lock, the same prompts and max_tokens 256, temperature 0:"
  say "flock ~/b70-gpu.lock build/src/cli/b70-serve $SNAP_QWEN --max-len 16384 --port 8013 --served-name b70 --mtp auto --log-requests ~/mtp-auto-log"
  say "flock ~/b70-gpu.lock build/src/cli/b70-serve $SNAP_QWEN --max-len 16384 --port 8013 --served-name b70 --mtp 1 --log-requests ~/mtp-k1-log"
  say "flock ~/b70-gpu.lock build/src/cli/b70-serve $SNAP_QWEN --max-len 16384 --port 8013 --served-name b70 --mtp 3 --log-requests ~/mtp-k3-log"
  say "# t/s per request from each log's t_first_token / t_end and out_ids; arms interleaved, median of 3"
}

stage r2.opencode 2 optin gpu-self qwen,opencode_log - "the opencode replay with --mtp auto (OPENCODE_LOG=<request log dir>), prefix cache on"
st_r2_opencode() {
  chk "MODEL=$SNAP_QWEN ARMS=on SERVE_ARGS='--mtp auto' tools/prefix/replay_ab.sh $OPENCODE_LOG $STATE/opencode" "the opencode replay"
  grab_all replay 'requests|ttft|tok/s|diverg' 20
  finish
}

# ======================================================================================
row 3 "README rows on current main (v2 decode attention, int8 head), idle box, RECORD grade"
stage r3.readme 3 default gpu qwen - "headline pp4096 / tg256 with the bf16 and int8 heads; prefill and decode at depth 32k / 64k / 130816 (max_len 131072) - interleaved, median of 3"
st_r3_readme() {
  idle before
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio 'pp4096-int8/pp4096' -- \
pp4096 '$(bench_cmd "$SNAP_QWEN" --pp 4096 --tg 256)' \
pp4096-int8 '$(bench_cmd "$SNAP_QWEN" --pp 4096 --tg 256 --lm-head int8)' \
depth32768 '$(bench_cmd "$SNAP_QWEN" --pp 32768 --tg 256 --max-len 131072)' \
depth65536 '$(bench_cmd "$SNAP_QWEN" --pp 65536 --tg 256 --max-len 131072)' \
depth130816 '$(bench_cmd "$SNAP_QWEN" --pp 130816 --tg 256 --max-len 131072)'" "the README arms"
  idle after
  grab_all memory '^memory:' 6
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 4 "g128 symmetric loader (2757bfd)"
stage r4.g64 4 default gpu qwen,oracle_qwen - "quant_test, load_checkpoint_test and the golden gate on the g64 checkpoint (unchanged bytes)"
st_r4_g64() {
  run_tests '^(quant_test|load_checkpoint_test|golden_gate_test)$'
  finish
}
stage r4.g128 4 manual - - - "optional: a public g128 sym GPTQ checkpoint of Qwen3.8, its golden prompts against its own CPU reference"
st_r4_g128() {
  say "uvx --from huggingface_hub hf download <a g128 sym GPTQ Qwen3.8 checkpoint>"
  say "ORACLE_MODEL=<its models--... dir> OUT_DIR=oracle-out-g128 tools/oracle/golden.sh        # its own CPU reference (tools/oracle/README.md)"
  say "flock ~/b70-gpu.lock build/tests/golden_gate_test oracle-out-g128 tests/golden/prompts <its snapshot dir>"
}

# ======================================================================================
row 5 "spec 15b - ModelDesc carries every per-model width (2026-10-04-spec15b-validation-checklist.md)"
rownote 5 "Step 6 (b70-decode on a config-only Ornith directory fails with 'MoE not implemented (spec 15c)') is superseded: 15c made Ornith loadable for decode; row 10's r10.refusals checks what Ornith refuses now."
stage r5.host 5 default cpu - - "step 3: model_desc, variant_names, qwen35, kernel_table, agnes_fold, template"
st_r5_host() {
  run_tests '^(model_desc_test|variant_names_test|qwen35_test|kernel_table_test|agnes_fold_test|template_test)$'
  finish
}
stage r5.r0 5 default cpu - g0.sha,g0.bitwise,g0.suite "steps 2, 4, 5: binaries, the Qwen3.8 suite bitwise (G0) and the Agnes gates (row 1) - 15b adds no binary"
st_r5_r0() {
  need_pass g0.sha g0.bitwise g0.suite r1.gates r1.features
  jgrab buffers '^buffers_test$' 'x |mixer_out|total'
  jgrab mtp-bytes '^load_checkpoint_test$' '849|MTP'
  finish
}

# ======================================================================================
row 6 "spec 8 §12 - MTP at every context length"
rownote 6 "mtp_gpu_test and mtp_verify_test still run at max_len 16384 only (kMaxLen is a constexpr in tests/server/mtp_gpu_test.cc and tests/runtime/mtp_verify_test.cc): greedy equality and M2 at 32768 / 131072 need a --max-len argument there first."
rownote 6 "probe_mtp_steps builds its engine at max_len 16384 (tools/probe/probe_mtp_steps.cc), so the cost table at 32k / 120k depth needs a max_len argument; A12's 32k / 64k rows (tools/probe/mtp_d1.sh) wait on mtp_gpu_test's for the same reason."
rownote 6 "Passkey at 120k with --mtp auto and spec 7 C1 restore with MTP at 64k are server paths (b70-decode has no --mtp; snapshot_test runs at 16384): no harness exists for either."
stage r6.memory 6 default gpu qwen - "the memory line at 131072 with MTP (~30.5 GB derived): memory_plan_box_test mtp 131072, plan == allocation"
st_r6_memory() {
  chk "build/tests/memory_plan_box_test $SNAP_QWEN mtp 131072" "memory_plan_box_test mtp 131072"
  grab_all plan 'plan == allocation|memory:'
  finish
}

# ======================================================================================
row 7 "spec 6 §10 - --max-len auto (b70-serve's default), the memory planner"
rownote 7 "P5's 'after a long session at auto' is an operator session: r7.serve records the device memory while a fresh server is up (xpu-smi, when installed); confirm or tune --mem-reserve-gb 1.5 from a long one."
stage r7.plan 7 default gpu qwen - "memory_plan_test, buffers_test (table unchanged), load_checkpoint_test (set_max_len), trained_context_test, memory_plan_box_test (+mtp) at 16384 and auto"
st_r7_plan() {
  run_tests '^(memory_plan_test|buffers_test|load_checkpoint_test|trained_context_test|memory_plan_box_test|memory_plan_box_mtp_test)$'
  jgrab plan 'memory_plan_box' 'plan == allocation'
  finish
}
stage r7.plan_agnes 7 default gpu agnes - "load_agnes_test (trained-context refusal), memory_plan_box_agnes_test (+mtp)"
st_r7_plan_agnes() {
  run_tests '^(load_agnes_test|memory_plan_box_agnes_test|memory_plan_box_agnes_mtp_test)$'
  jgrab plan 'memory_plan_box_agnes' 'plan == allocation'
  finish
}
stage r7.lines 7 default gpu qwen - "memory_plan_box_test <snap> 16384 131072 auto, with and without mtp: every component equals the allocation; 131072 reproduces spec 6 §8.4's line"
st_r7_lines() {
  chk "build/tests/memory_plan_box_test $SNAP_QWEN 16384 131072 auto" "memory_plan_box_test 16384 131072 auto"
  chk "build/tests/memory_plan_box_test $SNAP_QWEN mtp 16384 131072 auto" "memory_plan_box_test mtp 16384 131072 auto"
  grab_all plan 'plan == allocation|memory:|max_len'
  finish
}
stage r7.lines_agnes 7 default gpu agnes - "the same for Agnes"
st_r7_lines_agnes() {
  chk "build/tests/memory_plan_box_test $SNAP_AGNES 16384 131072 auto" "memory_plan_box_test (Agnes) 16384 131072 auto"
  chk "build/tests/memory_plan_box_test $SNAP_AGNES mtp 16384 auto" "memory_plan_box_test (Agnes) mtp 16384 auto"
  grab_all plan 'plan == allocation|memory:|max_len'
  finish
}
stage r7.serve 7 default gpu qwen - "b70-serve with no --max-len starts, prints 'max_len: auto -> N' (derived 201216; 169984 with --mtp auto) and serves"
st_r7_serve() {
  serve "$SNAP_QWEN"
  serve "$SNAP_QWEN" --mtp auto
  grab_all auto 'max_len: auto ->'
  grab_all memory '^memory:'
  grab_all prefix 'prefix cache:'
  grab_all device-memory '[Mm]emory.*([Uu]sed|[Ff]ree)' 8
  finish
}
stage r7.serve_agnes 7 default gpu agnes - "the same for Agnes (derived 139520; 114176 with --mtp auto)"
st_r7_serve_agnes() {
  serve "$SNAP_AGNES"
  serve "$SNAP_AGNES" --mtp auto
  grab_all auto 'max_len: auto ->'
  grab_all memory '^memory:'
  finish
}
stage r7.passkey 7 optin gpu qwen,oracle_image - "passkey 3/3 at 95% of the auto length (bf16 head, prefill planned; ~180k)"
st_r7_passkey() {
  chk "N=\$(tools/box_validate/auto_len.sh $SNAP_QWEN --pp 256 --tg 1 --pp-backend l0) && echo \"auto length (prefill planned): \$N\" && MODEL=$SNAP_QWEN MAX_LEN=auto N_TARGET=\$((N * 95 / 100)) tools/probe/passkey.sh l0-int8 l0" "passkey at 95% of auto"
  grab 'auto' 'auto length'
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  finish
}
stage r7.depth 7 optin gpu qwen - "decode at depth N - 512 under --max-len auto (one replay per id: hours)"
st_r7_depth() {
  idle before
  # shellcheck disable=SC2016 # $((N - 512)) is evaluated on the box, after N is known
  chk "N=\$(tools/box_validate/auto_len.sh $SNAP_QWEN) && echo \"auto length (decode only): \$N\" && $(bench_cmd "$SNAP_QWEN" --max-len auto --depth '$((N - 512))' --tg 256)" "decode at auto depth"
  idle after
  grab auto 'auto length'
  grab_all row '^\| b70-decode'
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 8 "spec 8 §11 - --draft-vocab (int8 and bf16 heads), --mtp auto draft costs scaled by |V'|"
rownote 8 "The default moves off 'off' only if a size beats it by >= 2 % on A4 and loses nowhere: read r8.rows's medians (the operator's ruling, then MtpCost's draft shares from r8.cost)."
stage r8.kernels 8 default gpu - - "draft_vocab_kernels_test: compact GEMV bitwise = full columns (both forms), mapped argmax, scatter; ties / -inf / NaN"
st_r8_kernels() {
  run_tests '^draft_vocab_kernels_test$'
  finish
}
stage r8.m3 8 default gpu qwen - "M3 at 128k: mtp_gpu_dv128k_test and _bf16_test (greedy identical to off, K = 1..3, golden + A4; RF1, M5, M4; 23 launches)"
st_r8_m3() {
  run_tests '^(mtp_gpu_dv128k_test|mtp_gpu_dv128k_bf16_test)$'
  jgrab m3 'mtp_gpu_dv128k' 'mtp_gpu_test .*(PASS|FAIL)|launch|diverg'
  finish
}
stage r8.off 8 default gpu qwen - "the suite unchanged with the flag off: mtp_gpu_test, mtp_verify_test, prefix_gpu_mtp_test"
st_r8_off() {
  run_tests '^(mtp_gpu_test|mtp_verify_test|prefix_gpu_mtp_test)$'
  finish
}
stage r8.memory 8 default gpu qwen - "memory_plan_box_dv128k_test and _bf16_test: the draft_vocab term = compact head + id table, dv_logits = MtpDims; allocated == planned (16384, auto)"
st_r8_memory() {
  run_tests '^(memory_plan_box_dv128k_test|memory_plan_box_dv128k_bf16_test)$'
  jgrab plan 'memory_plan_box_dv128k' 'plan == allocation'
  finish
}
stage r8.serve 8 default gpu qwen - "b70-serve --mtp 3 --draft-vocab 128k starts (both heads), prints the draft vocab load and memory lines; --mtp auto at auto picks ~160k (int8) / ~132k (bf16)"
st_r8_serve() {
  serve "$SNAP_QWEN" --max-len 16384 --mtp 3 --draft-vocab 128k --lm-head int8
  serve "$SNAP_QWEN" --max-len 16384 --mtp 3 --draft-vocab 128k --lm-head bf16
  serve "$SNAP_QWEN" --mtp auto --draft-vocab 128k --lm-head int8
  serve "$SNAP_QWEN" --mtp auto --draft-vocab 128k --lm-head bf16
  grab_all draft-vocab 'draft vocab'
  grab_all auto 'max_len: auto ->'
  grab_all memory '^memory:'
  finish
}
stage r8.cost 8 default gpu qwen - "probe_mtp_steps at |V'| = 32k / 64k / 128k for both heads: the measured draft rows replace kInt8DraftHeadShare / kBf16DraftHeadShare (timed)"
st_r8_cost() {
  idle before
  local h s
  for h in int8 bf16; do
    for s in off 32k 64k 128k; do
      chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 4096 32 3 $h $s" "probe_mtp_steps $h $s"
    done
  done
  idle after
  grab_all cost-rows '^lm_head:|^\| ' 80
  grab_all idle '^IDLE '
  finish
}
stage r8.rows 8 default gpu qwen - "§11 rows: mtp_gpu_test --bench K = 1 / 3, int8 head, greedy, at off / 32k / 64k / 128k - acceptance and t/s on golden, A4, prose (interleaved, median of 3)"
st_r8_rows() {
  idle before
  local arms="" k s dv
  for k in 1 3; do
    for s in off 32k 64k 128k; do
      dv=""; [ "$s" = off ] || dv=" --draft-vocab $s"
      arms="$arms K$k-int8-$s 'build/tests/mtp_gpu_test $SNAP_QWEN l0-int8 --bench $k --lm-head int8$dv'"
    done
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio K1-int8-128k/K1-int8-off --ratio K3-int8-128k/K3-int8-off --ratio K3-int8-64k/K3-int8-off --ratio K3-int8-32k/K3-int8-off --$arms" "the draft-vocab arms"
  idle after
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r8.rows_full 8 optin gpu qwen - "the rest of §11's rows: bf16 head and --sampled, K = 1 / 3, off / 32k / 64k / 128k (hours)"
st_r8_rows_full() {
  idle before
  local arms="" k s h m dv sm
  for h in int8 bf16; do for m in greedy sampled; do
    [ "$h/$m" = int8/greedy ] && continue
    sm=""; [ "$m" = sampled ] && sm=" --sampled"
    for k in 1 3; do for s in off 32k 64k 128k; do
      dv=""; [ "$s" = off ] || dv=" --draft-vocab $s"
      arms="$arms K$k-$h-$m-$s 'build/tests/mtp_gpu_test $SNAP_QWEN l0-int8 --bench $k --lm-head $h$sm$dv'"
    done; done
  done; done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --$arms" "the remaining draft-vocab arms"
  idle after
  grab_all idle '^IDLE '
  finish
}
stage r8.m3_small 8 optin gpu qwen - "M3 at 32k and 64k 'by hand': mtp_gpu_test's gate with --draft-vocab 32k / 64k, both heads (~1.5 h each)"
st_r8_m3_small() {
  local h s
  for h in int8 bf16; do
    for s in 32k 64k; do
      chk "build/tests/mtp_gpu_test $SNAP_QWEN l0-int8 --lm-head $h --draft-vocab $s" "M3 at $s, $h head"
    done
  done
  grab_all m3 'mtp_gpu_test .*(PASS|FAIL)'
  finish
}
stage r8.auto_rows 8 manual - qwen - "--mtp auto rows at the chosen size against fixed K (server paths, as r2.auto_rows)"
st_r8_auto_rows() {
  say "# as r2.auto_rows, each server with --draft-vocab <chosen size> added; the default b70-serve --mtp auto applies MtpCost::with_draft_vocab"
}

# ======================================================================================
row 9 "--prefix-cache-gb auto (spec 7 amendment)"
stage r9.auto 9 default gpu qwen r7.serve "b70-serve's startup line shows the auto size and its inputs (/proc/meminfo); prefix_server_test / golden_server_test / prefix_cache_size_test green with the default"
st_r9_auto() {
  x "grep -hE 'prefix cache:' $STATE/r7.serve.log | head -4"
  x "grep -E '^(MemTotal|MemAvailable)' /proc/meminfo"
  grab_all prefix 'prefix cache:' 4
  grab_all meminfo '^(MemTotal|MemAvailable)' 2
  run_tests '^(prefix_server_test|golden_server_test|prefix_cache_size_test)$'
  finish
}

# ======================================================================================
row 10 "spec 15c - Ornith 1.5 35B-A3B decode (spec 15 §10)"
rownote 10 "P0's build-variant sweeps (UP_KS 1/2/4/8, DN_KS 1/2/4, the {S, layout} sweeps of the four int4 shapes, spec 15 §9's arms, the 3-launch arm) each need a rebuild with other defines or a probe that does not exist yet: r10.p0_sweeps lists them."
rownote 10 "Everything after the no-checkpoint tests waits on 15a: the int4 checkpoint (SNAP_ORNITH) and oracle-out-ornith with router_logits.L*; those stages SKIP with 'missing data' until both are on the box."
stage r10.r0 10 default cpu - g0.sha,g0.bitwise,g0.suite "R0: every pre-existing binary identical (the four edited sources included), the Qwen3.8 suite, Agnes's gates"
st_r10_r0() {
  need_pass g0.sha g0.bitwise g0.suite r1.gates
  finish
}
stage r10.nockpt 10 default gpu - - "no checkpoint needed: moe_test (the card's sub-group path, 288-lane moe_down), moe_ref_test, ornith_repack_test, variant_names_test, memory_plan_test, model_desc_test"
st_r10_nockpt() {
  run_tests '^(moe_test|moe_ref_test|ornith_repack_test|variant_names_test|memory_plan_test|model_desc_test)$'
  finish
}
stage r10.load 10 default gpu ornith - "the loader printout on the int4 checkpoint: per-expert form, 0 unconsumed, the MoE line, read/token (set doc_w from it)"
st_r10_load() {
  chk "$(bench_cmd "$SNAP_ORNITH" --depth 16 --tg 4 --lm-head int8)" "b70-decode on Ornith"
  grab_all loader 'MoE|moe|unconsumed|W check|per token|expert' 20
  finish
}
stage r10.gates 10 default gpu ornith,oracle_ornith - "ornith_decode_test (+_i8head: 526 launches, plan == allocations, replay bitwise, R2) and golden_gate_ornith (+_i8head: R3)"
st_r10_gates() {
  run_tests 'ornith_decode|golden_gate_ornith' ornith
  jgrab ornith 'ornith' 'launch|526|R2|R3|OK|PASS|FAIL'
  finish
}
stage r10.refusals 10 default gpu ornith - "what Ornith refuses, each with its message: --pp, --ids --prefill, b70-serve, --mtp, B70_DECODE_ATTN=v1"
st_r10_refusals() {
  xfail "timeout 900 build/src/cli/b70-decode $SNAP_ORNITH --bench --pp 512 --tg 1" 'prefill of a mixture-of-experts model' "b70-decode --pp 512"
  xfail "timeout 900 build/src/cli/b70-decode $SNAP_ORNITH --ids tests/golden/prompts/prose.ids --n 1 --prefill" 'prefill of a mixture-of-experts model' "b70-decode --ids --prefill"
  xfail "timeout 900 build/src/cli/b70-serve $SNAP_ORNITH --port $PORT --max-len 16384" 'prefill of a mixture-of-experts model' "b70-serve"
  xfail "timeout 900 build/src/cli/b70-serve $SNAP_ORNITH --port $PORT --max-len 16384 --mtp 1" 'mixture-of-experts|MoE|spec 15e' "b70-serve --mtp 1"
  xfail "B70_DECODE_ATTN=v1 timeout 900 build/src/cli/b70-decode $SNAP_ORNITH --bench --depth 16 --tg 1" 'decode attention v1' "B70_DECODE_ATTN=v1"
  finish
}
stage r10.p0 10 optin gpu ornith - "P0 (plan 15c Task 1), the runnable part: the per-kernel floor (probe_replay) and b70-decode --profile --depth 4096 for the shares of the 526 launches"
st_r10_p0() {
  idle before
  chk "build/tools/probe/probe_replay" "probe_replay"
  chk "build/src/cli/b70-decode $SNAP_ORNITH --profile --depth 4096 --steps 32 --repeats 5 --lm-head int8" "the Ornith decode profile"
  idle after
  grab_all idle '^IDLE '
  finish
}
stage r10.speed 10 optin gpu ornith - "Task 4: decode at 4k / 32k / 128k / 250k, --lm-head int8, --max-len auto (decode-only auto should give 262144; one replay per id: hours)"
st_r10_speed() {
  idle before
  chk "tools/box_validate/auto_len.sh $SNAP_ORNITH --depth 16 --tg 1 --lm-head int8" "the decode-only auto length"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r ${R10_RUNS:-3} -- \
d4096 '$(bench_cmd "$SNAP_ORNITH" --max-len auto --lm-head int8 --depth 4096 --tg 256)' \
d32768 '$(bench_cmd "$SNAP_ORNITH" --max-len auto --lm-head int8 --depth 32768 --tg 256)' \
d131072 '$(bench_cmd "$SNAP_ORNITH" --max-len auto --lm-head int8 --depth 131072 --tg 256)' \
d250000 '$(bench_cmd "$SNAP_ORNITH" --max-len auto --lm-head int8 --depth 250000 --tg 256)'" "the Ornith depth arms"
  idle after
  grab_all memory '^memory:' 4
  grab_all idle '^IDLE '
  finish
}
stage r10.p0_sweeps 10 manual - ornith - "P0's build-variant sweeps and spec 15 §9's arms"
st_r10_p0_sweeps() {
  say "# UP_KS 1 / 2 / 4 / 8 and DN_KS 1 / 2 / 4: rebuild moe.cl's variants with the define, re-run moe_test and b70-decode --bench on Ornith per value"
  say "# {S, layout} sweeps of q||k||v 2048x9216, qkv||z 2048x12288, out_proj = o_proj 4096x2048 with their GEMV_* defines (probe_gemv rows as spec 14 step 4)"
  say "# spec 15 §9 arms: register-resident GEMV vs moe.cl gate||up / down; one-sub-group top-k vs moe_route; all-loads-first attention vs v2 at GQA 8, 4k; the 3-launch arm"
  say "# record: docs/probe-ornith-decode-2026-10-04.md (derived decode estimate, proposed bar)"
}

# ======================================================================================
row 11 "spec 12b - the int8 KV cache (--kv-cache int8, rotkv; spec 12 §9, plan 12b)"
rownote 11 "Step 0, the Qwen3.8 repeat of 12a, runs on the Mac CPU (tools/oracle/kv8_qwen38_repeat.sh, plan 12b Task A); until its numbers replace the PROVISIONAL constants, r11.q2 runs against the provisional tolerances and says so."
rownote 11 "Making int8 the default (plan 12b Task 4 Step 3) is an edit after every gate passes."
stage r11.bf16 11 default cpu - g0.sha,g0.bitwise,g0.suite "step 1, bf16 unchanged: every pre-existing binary identical, the suite at the default with B70_KV_CACHE unset (G0)"
st_r11_bf16() {
  need_pass g0.sha g0.bitwise g0.suite
  x "env | grep -E '^B70_(KV_CACHE|DECODE_ATTN|PREFILL_ATTN)=' || echo 'B70_KV_CACHE, B70_DECODE_ATTN, B70_PREFILL_ATTN unset in this run'"
  finish
}
stage r11.kernels 11 default gpu qwen - "step 2 FIRST - the readers that replace DPAS / 2D-block reads with byte gathers: kv8_kernels_test (W, D, F), then flash_long_kv8_test (1 - cos <= 5e-4 at 4k-32k); a failure stops row 11"
st_r11_kernels() {
  run_tests '^kv8_kernels_test$'
  if [ "$BV_BAD" -eq 0 ]; then
    run_tests '^flash_long_kv8_test$'
    jgrab flash-long '^flash_long_kv8_test$' 'cos|PASS|FAIL'
  else
    note "flash_long_kv8_test not run: kv8_kernels_test failed (the readers first)"
  fi
  jgrab kernels '^kv8_kernels_test$' 'cos|bitwise|PASS|FAIL|gated'
  x "grep -n 'PROVISIONAL' tools/probe/kv8_vs_oracle.sh tests/prefill/flash_long_test.cc tests/kernels/kv8_kernels_test.cc | head -5 || echo 'no PROVISIONAL marker left'"
  grab provisional 'PROVISIONAL|no PROVISIONAL marker'
  finish
}
stage r11.q2 11 default gpu qwen,oracle_qwen r11.kernels "step 3, Q2: golden_gate_kv8 (+i8head), prefill_gate_{int8,l0}_kv8 (tie rule unchanged), kv8_vs_oracle_test (golden drop <= 1e-4)"
st_r11_q2() {
  run_tests 'golden_gate|prefill_gate|kv8_vs_oracle' kv8
  jgrab q2 'kv8' 'golden_gate_test OK|TOTAL:|drop|cos mean'
  finish
}
stage r11.q5 11 default gpu qwen,oracle_qwen r11.kernels "step 3, Q5: replay / prefill determinism, prefill_replay, snapshot, mtp_verify (M2), mtp_gpu, prefix_gpu_*, engine_smoke, prefill_consistency (its bar may need re-deriving)"
st_r11_q5() {
  run_tests 'replay_determinism|prefill_determinism|prefill_replay|snapshot|mtp_verify|mtp_gpu|prefix_gpu|engine_smoke|prefill_consistency' kv8
  finish
}
stage r11.memory 11 default gpu qwen r11.kernels "step 4: memory_plan_box_kv8_test / _mtp_kv8_test (plan == allocation at int8); b70-serve --kv-cache int8 picks 262144 under auto (derived) and serves"
st_r11_memory() {
  run_tests 'memory_plan_box' kv8
  serve "$SNAP_QWEN" --kv-cache int8
  grab_all auto 'max_len: auto ->'
  grab_all memory '^memory:'
  finish
}
stage r11.speed 11 default gpu qwen r11.q2 "step 5: bf16-KV against int8-KV pairs - decode at 32k / 64k / 128k (bars 1.1x / 1.15x / 1.2x), prefill at depth (<= 3 % regression), the memory line at 131072"
st_r11_speed() {
  idle before
  local d arms="" ratios=""
  for d in 32768 65536 130816; do
    arms="$arms bf16-kv@$d '$(bench_cmd "$SNAP_QWEN" --pp $d --tg 256 --max-len 131072)' int8-kv@$d '$(bench_cmd "$SNAP_QWEN" --pp $d --tg 256 --max-len 131072 --kv-cache int8)'"
    ratios="$ratios --ratio int8-kv@$d/bf16-kv@$d"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the KV-form pairs"
  idle after
  grab_all ratio '^RATIO '
  grab_all memory '^memory:' 6
  grab_all idle '^IDLE '
  finish
}
stage r11.passkey120k 11 optin gpu qwen,oracle_image r11.kernels "Q3: passkey 3/3 at 120k with the int8 cache (l0-int8)"
st_r11_passkey120k() {
  chk "B70_KV_CACHE=int8 MODEL=$SNAP_QWEN tools/probe/passkey.sh l0-int8" "passkey 120k, int8 KV"
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  finish
}
stage r11.passkey262k 11 optin gpu qwen,oracle_image r11.kernels "Q3: passkey 3/3 at 250000 ids, max_len 262144 (only int8 fits it)"
st_r11_passkey262k() {
  chk "B70_KV_CACHE=int8 MODEL=$SNAP_QWEN MAX_LEN=262144 N_TARGET=250000 tools/probe/passkey.sh l0-int8" "passkey 262k, int8 KV"
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  finish
}
stage r11.a4 11 optin gpu qwen,a4_ref r11.kernels "Q4: A4 with the int8 cache, scored against the bf16 reference run (A4_REF_DIR holds <name>.bf16.txt): >= 25/36"
st_r11_a4() {
  x "mkdir -p $STATE/a4-kv8 && cp $A4_REF_DIR/*.bf16.txt $STATE/a4-kv8/"
  chk "B70_KV_CACHE=int8 tools/toolcall/engine_generate.sh $SNAP_QWEN tests/golden/toolcall $STATE/a4-kv8 l0-int8" "A4 engine runs, int8 KV"
  chk "python3 tools/toolcall/score.py $STATE/a4-kv8 bf16 l0-int8 | tee $STATE/a4-kv8/score.md" "scoring"
  grab_all a4 'match|/36|total' 6
  finish
}
stage r11.near_ties 11 manual - qwen - "Q4: 256 greedy ids on the golden prompts with int8 KV diverging only at near-ties"
st_r11_near_ties() {
  say "for p in prose code cjk; do flock ~/b70-gpu.lock build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/\$p.ids --n 256 --prefill > ~/kv8-\$p.bf16.ids; done"
  say "for p in prose code cjk; do B70_KV_CACHE=int8 flock ~/b70-gpu.lock build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/\$p.ids --n 256 --prefill > ~/kv8-\$p.int8.ids; done"
  say "# first divergence per prompt; judge it by the tie rule against the bf16 run (build/tools/probe/probe_tie_judge, as spec 7's replay cases)"
}

# ======================================================================================
row x "the rest of the suite: every registered test no stage above ran (new host tests, the routed tests' twins)"
stage x.rest x default gpu qwen,oracle_qwen - "ctest over every registered test without a result in this run (Agnes / Ornith / kv8 labels belong to their rows)"
st_x_rest() {
  run_tests '.' '' 'agnes ornith kv8'
  finish
}
