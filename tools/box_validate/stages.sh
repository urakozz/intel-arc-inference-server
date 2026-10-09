# tools/box_validate/stages.sh - the box validation queue as stages.
#
# docs/superpowers/plans/box-validation-queue.md is the list of work merged into main
# without running on the card; each `row` below is one of its rows, each `stage` one step
# of that row's "how to validate", in the order the queue and its checklists require.
# Registry fields and helpers: tools/box_validate/lib.sh. A stage body runs ON THE BOX from
# the tree under test ($TREE); in a dry-run on the Mac every helper prints instead of running.
#
# Adding a row (a branch that lands blind - rows 15 to 17 below are the latest examples):
#   1. `row N "title"` and one `stage rN.<step> N <sel> <kind> <needs> <after> "title"` per
#      step, after the last row (the order here is the run order);
#   2. a body `st_rN_<step>() { ...; finish; }` - usually `run_tests '<ctest regex>' [labels]`
#      for registered tests, `x "<command>"` / `chk` for anything else, `grab` for the numbers
#      the row asks for, `interleave.sh` for timed arms, `serve_arms` + serve_client.py for
#      rows that need b70-serve per request, `kbins` for binaries a row's list binds by name;
#   3. `rownote N "..."` for what the row asks that no stage can run, and why;
#   4. `tools/box_validate.sh --dry-run --only rN` prints what it will run;
#      `python3 tools/box_validate/test_box_validate.py` checks the registry.
# Stage ids are `<row>.<step>`: --only / --redo / --with accept an id or a row prefix (`r11`).

# What G0 compares OUTPUT-for-output against the baseline build (timings dropped): the gates
# the spec 14 and 15b checklists name as "bitwise identical to main's last record".
: "${G0_BITWISE_RE:=^(golden_gate_test|golden_gate_i8head_test|prefill_gate_l0_test|prefill_gate_int8_test|prefill_gate_l0_i8head_test|prefill_gate_int8_i8head_test|replay_determinism_test|replay_determinism_i8head_test|buffers_test|prefill_smoke_test|load_checkpoint_test)$}"
AGNES_ORACLE_MODEL=models--urakozz--Agnes-3.0-Flash-W4A16-AutoRound-GPTQ
ORNITH_ORACLE_MODEL=models--urakozz--Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
# Ornith's M = 1 decode list (tests/CMakeLists.txt B70_ORNITH_DECODE_KERNELS, ATTN_V2_TGT 32):
# since spec 15 §13 (the int4 checkpoint read) the greedy argmax is Qwen3.8's argmax_stage1_M1
# (its tokenizer.json defines 248077 ids) and a||b is int4: gemv_M1_K2048_N128_S1_L1.
ORNITH_DECODE_BINS="embed_gather_M1_D2048 argmax_stage1_M1 argmax_stage2 prep_res_fold_M1_K2048_SP0_G20
  prep_res_fold_M1_K2048_SP4_G20 prep_norm_finish_M1_K2048_G20_W20 prep_gated_head_M1_GK16V32 gdn_step_M1_GK16V32
  attn_prep_M1_Q16KV2 attn_v2_M1_T32_Q16KV2 gemv_M1_K2048_N12288_S1_L1 gemv_M1_K4096_N2048_S4_L0
  gemv_M1_K2048_N9216_S2_L0 gemv_M1_K2048_N248320_S1_L1 gemv_M1_K2048_N128_S1_L1 gemv_bf16_M1_K2048_N128_C16_S16
  gemv_bf16_M1_K2048_N272_C16_S16 gemv_bf16_M1_K2048_N248320 gemv_i8w_M1_K2048_N248320 moe_M1_E256_T8_D2048_I512"

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
pp4096 '$(bench_cmd "$SNAP_AGNES" --prefill-length 4096 --tg 256 --max-len 65536)' \
pp4096-int8 '$(bench_cmd "$SNAP_AGNES" --prefill-length 4096 --tg 256 --max-len 65536 --lm-head int8)' \
pp32768 '$(bench_cmd "$SNAP_AGNES" --prefill-length 32768 --tg 256 --max-len 65536)' \
pp60000 '$(bench_cmd "$SNAP_AGNES" --prefill-length 60000 --tg 256 --max-len 65536)'" "the Agnes speed arms"
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
rownote 2 "r2.auto_rows reads each arm's decode t/s from the server's --log-requests records; b70-serve's usage reports no accepted-draft count, so per-request acceptance is not recorded there (mtp_gpu_test --bench K and b70-decode --ids --mtp print it)."
rownote 2 "Updating MtpCost's defaults from r2.cost's table is an edit on a branch after the run."

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

stage r2.auto_rows 2 default gpu qwen - "D1-style rows through b70-serve: --mtp auto against --mtp 1 / 3 and off on the golden and A4 prompts, 256 ids, greedy (decode t/s per request from the server's records, R2_ROUNDS = 3 rounds rotated); greedy output identical across the arms"
st_r2_auto_rows() {
  idle before
  serve_arms "$STATE/$STAGE" "${R2_ROUNDS:-3}" "$SNAP_QWEN" "--set golden --set toolcall --max-tokens 256" \
    "off|--max-len 16384 --mtp off" "k1|--max-len 16384 --mtp 1" "k3|--max-len 16384 --mtp 3" \
    "auto|--max-len 16384 --mtp auto"
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output identical to --mtp off in every arm"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref off --ratio auto/k1 --ratio auto/k3"
  grab_all ratio '^RATIO ' 20
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all idle '^IDLE '
  finish
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
pp4096 '$(bench_cmd "$SNAP_QWEN" --prefill-length 4096 --tg 256)' \
pp4096-int8 '$(bench_cmd "$SNAP_QWEN" --prefill-length 4096 --tg 256 --lm-head int8)' \
depth32768 '$(bench_cmd "$SNAP_QWEN" --prefill-length 32768 --tg 256 --max-len 131072)' \
depth65536 '$(bench_cmd "$SNAP_QWEN" --prefill-length 65536 --tg 256 --max-len 131072)' \
depth130816 '$(bench_cmd "$SNAP_QWEN" --prefill-length 130816 --tg 256 --max-len 131072)'" "the README arms"
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
rownote 6 "Spec 7 C1 restore with MTP at 64k is mtp_verify_128k_test's RF4 at 69931 (max_len / 2 + 4395); snapshot_test itself stays at 16384."
rownote 6 "The passkey at 120k with --mtp auto (r6.passkey120k) and A12's 32k / 64k rows (r6.d1) are opt-in: hours each."
stage r6.memory 6 default gpu qwen - "the memory line at 131072 with MTP (~30.5 GB derived): memory_plan_box_test mtp 131072, plan == allocation"
st_r6_memory() {
  chk "build/tests/memory_plan_box_test $SNAP_QWEN mtp 131072" "memory_plan_box_test mtp 131072"
  grab_all plan 'plan == allocation|memory:'
  finish
}
stage r6.mtp_long 6 default gpu qwen r6.memory "M2 / RF3 / RF4 (C1 with MTP at ~70k) and M3 greedy equality at max_len 32768 and 131072: mtp_verify_{32k,128k}_test, mtp_gpu_{32k,128k}_test (label longctx, B70_LONGCTX_TESTS=1)"
st_r6_mtp_long() {
  run_tests '^(mtp_verify|mtp_gpu)_(32k|128k)_test$' longctx '' 'B70_LONGCTX_TESTS=1'
  jgrab m2 'mtp_verify_(32k|128k)' '^M2:|^RF[345]|max_len'
  jgrab m3 'mtp_gpu_(32k|128k)' 'M3:|M3 pooled|mtp_gpu_test .*(PASS|FAIL)|max_len'
  finish
}
stage r6.greedy_long 6 default gpu qwen r6.memory "long-context MTP greedy equality without the server: b70-decode --ids (120000 ids) --prefill --max-len 131072 --lm-head int8, 256 ids plain / --mtp 3 / --mtp auto, the three outputs identical"
st_r6_greedy_long() {
  local arm f="$STATE/$STAGE.120k.ids"
  chk "python3 tools/box_validate/long_ids.py 120000 $f" "the 120000-id prompt"
  for arm in off 3 auto; do
    chk "build/src/cli/b70-decode $SNAP_QWEN --ids $f --n 256 --prefill --max-len 131072 --lm-head int8 --mtp $arm > $STATE/$STAGE.mtp-$arm.out" "b70-decode --mtp $arm at 120k"
  done
  chk "cmp $STATE/$STAGE.mtp-off.out $STATE/$STAGE.mtp-3.out && cmp $STATE/$STAGE.mtp-off.out $STATE/$STAGE.mtp-auto.out && echo 'GREEDY identical: plain, --mtp 3, --mtp auto at 120k'" "greedy ids identical with and without MTP at 120k"
  grab_all mtp '^mtp:|^generate:|GREEDY' 12
  finish
}
stage r6.cost 6 default gpu qwen r6.memory "the cost table at depth: probe_mtp_steps at 32k (max_len 65536) and 120k (max_len 131072), int8 and bf16 heads (timed)"
st_r6_cost() {
  idle before
  local h
  for h in int8 bf16; do
    chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 32768 32 3 $h off 65536" "probe_mtp_steps 32k, $h head"
    chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 120000 32 3 $h off 131072" "probe_mtp_steps 120k, $h head"
  done
  idle after
  grab_all cost-rows '^lm_head:|^max_len:|^probe_mtp_steps:|^\| ' 80
  grab_all idle '^IDLE '
  finish
}
stage r6.passkey120k 6 optin gpu qwen,oracle_image r6.memory "passkey 3/3 at 120k with b70-decode --mtp auto --lm-head int8 (l0-int8; passkey.sh DECODE_ARGS)"
st_r6_passkey120k() {
  chk "DECODE_ARGS='--mtp auto --lm-head int8' MODEL=$SNAP_QWEN tools/probe/passkey.sh l0-int8" "passkey 120k, --mtp auto (int8 head)"
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  grab_all mtp 'mtp: ' 6
  finish
}
stage r6.d1 6 optin gpu-self qwen r6.memory "A12's rows at depth 32k and 64k: tools/probe/mtp_d1.sh greedy (mtp_gpu_test --bench K = 0..3, 3 rounds; MAX_LEN 65536 / 131072), the table (hours)"
st_r6_d1() {
  local d m
  for d in 32768 65536; do
    m=65536; [ "$d" = 65536 ] && m=131072
    chk "SNAP=$SNAP_QWEN MAX_LEN=$m tools/probe/mtp_d1.sh greedy $d > $STATE/$STAGE.$d.log && python3 tools/probe/mtp_d1_table.py $STATE/$STAGE.$d.log" "mtp_d1.sh at depth $d"
  done
  grab_all d1 'geomean|^\| ' 40
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
stage r7.passkey 7 optin gpu qwen,oracle_image - "passkey 3/3 at 95% of the auto length (bf16 head, prefill planned: auto 181760 derived, so ~172.7k ids)"
st_r7_passkey() {
  chk "N=\$(tools/box_validate/auto_len.sh $SNAP_QWEN --prefill-length 256 --tg 1 --prefill-backend l0) && echo \"auto length (prefill planned): \$N\" && MODEL=$SNAP_QWEN MAX_LEN=auto N_TARGET=\$((N * 95 / 100)) tools/probe/passkey.sh l0-int8 l0" "passkey at 95% of auto"
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
stage r8.auto_rows 8 optin gpu qwen - "--mtp auto rows at the chosen size (R8_DRAFT_VOCAB, default 128k) against fixed K = 1 / 3 and against --mtp auto without it, through b70-serve as r2.auto_rows"
st_r8_auto_rows() {
  local dv="${R8_DRAFT_VOCAB:-128k}"
  idle before
  serve_arms "$STATE/$STAGE" "${R2_ROUNDS:-3}" "$SNAP_QWEN" "--set golden --set toolcall --max-tokens 256" \
    "auto|--max-len 16384 --mtp auto" "auto-dv|--max-len 16384 --mtp auto --draft-vocab $dv" \
    "k1-dv|--max-len 16384 --mtp 1 --draft-vocab $dv" "k3-dv|--max-len 16384 --mtp 3 --draft-vocab $dv"
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref auto" "greedy output identical with the draft vocabulary"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref auto --ratio auto-dv/k1-dv --ratio auto-dv/k3-dv"
  grab_all ratio '^RATIO ' 20
  grab_all idle '^IDLE '
  finish
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
rownote 10 "The --prefill-length / --ids --prefill refusals are gone since 15d (Ornith prefills on l0 / l0-int8; row 13's r13.cli runs them), b70-serve's and --mtp's since 15e (Ornith is served, with its MoE MTP head: row 16); r10.refusals checks what remains."
stage r10.r0 10 default cpu - g0.sha,g0.bitwise,g0.suite "R0: every pre-existing binary identical (the four edited sources included), the Qwen3.8 suite, Agnes's gates; Ornith's decode list built - since spec 15 §13 its greedy argmax is Qwen3.8's argmax_stage1_M1 (the int4 checkpoint's tokenizer.json: 248077 ids) and its a||b the int4 gemv_M1_K2048_N128_S1_L1"
st_r10_r0() {
  need_pass g0.sha g0.bitwise g0.suite r1.gates
  kbins $ORNITH_DECODE_BINS
  finish
}
stage r10.nockpt 10 default gpu - - "no checkpoint needed: moe_test (the card's sub-group path, 288-lane moe_down), moe_ref_test, ornith_repack_test, variant_names_test, memory_plan_test, model_desc_test, ab_int4_test"
st_r10_nockpt() {
  run_tests '^(moe_test|moe_ref_test|ornith_repack_test|variant_names_test|memory_plan_test|model_desc_test|ab_int4_test)$'
  finish
}
stage r10.load 10 default gpu ornith - "the loader printout on the int4 checkpoint: per-expert form, 0 unconsumed, the MoE line, the int4 a||b line (64 real columns, its 15.7 MB prefill copy), read/token and the W check against doc_w 2.345 GB (spec 15 §13: derived from the headers, expected delta 0.000%; a throw past 2% is a finding)"
st_r10_load() {
  chk "$(bench_cmd "$SNAP_ORNITH" --depth 16 --tg 4 --lm-head int8)" "b70-decode on Ornith"
  grab_all loader 'MoE|moe|unconsumed|W check|per token|expert|a‖b' 20
  finish
}
stage r10.gates 10 default gpu ornith,oracle_ornith - "ornith_decode_test (+_i8head: 526 launches, plan == allocations, replay bitwise, R2) and golden_gate_ornith (+_i8head: R3)"
st_r10_gates() {
  run_tests 'ornith_decode|golden_gate_ornith' ornith
  jgrab ornith 'ornith' 'launch|526|R2|R3|OK|PASS|FAIL'
  finish
}
stage r10.refusals 10 default gpu ornith - "what Ornith still refuses, with its message: B70_DECODE_ATTN=v1 (decode attention v1 is not built at its heads); b70-serve and --mtp serve it since 15e (row 16)"
st_r10_refusals() {
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
    arms="$arms bf16-kv@$d '$(bench_cmd "$SNAP_QWEN" --prefill-length $d --tg 256 --max-len 131072)' int8-kv@$d '$(bench_cmd "$SNAP_QWEN" --prefill-length $d --tg 256 --max-len 131072 --kv-cache int8)'"
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
row 12 "spec 19e host side - --spec lookup, prompt lookup (plan 19e)"
rownote 12 "'no kernel binary changes' is G0's g0.sha (every binary present in both builds identical)."
rownote 12 "The per-request 'lookup:' stderr line the plan names is not printed by b70-serve as merged; r12.d2 / r12.a4 record whatever 'lookup' lines the server log has."
stage r12.suite 12 default gpu qwen,oracle_qwen - "the suite unchanged with --spec unset: mtp_gpu_test, golden_server_test, prefix_gpu_*; the host tests prompt_lookup_test, lookup_server_test, spec_accept_test"
st_r12_suite() {
  run_tests '^(mtp_gpu_test|golden_server_test|prefix_gpu_.*|prompt_lookup_test|lookup_server_test|spec_accept_test)$' '' 'kv8 agnes'
  finish
}
stage r12.d2 12 default gpu qwen - "D2: golden_server_test against a --spec lookup server (min match n = 2 and 3) - greedy identical to the plain decode; a seeded sampled request bitwise reproducible; the 'spec lookup:' startup and per-request 'lookup:' lines"
st_r12_d2() {
  local n
  for n in 2 3; do
    chk "B70_SERVE_EXTRA='--spec lookup --spec-min-match $n' timeout 1800 build/tests/golden_server_test $TREE/tools/box_validate/serve_extra.sh build/src/cli/b70-decode tests/golden/prompts $SNAP_QWEN" \
      "golden_server_test through --spec lookup, n = $n"
  done
  x "PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_QWEN --max-len 16384 --spec lookup"
  step_rc $? "b70-serve --spec lookup: serves; the seeded sampled request twice"
  grab_all lookup 'spec lookup:|lookup:|SEEDED' 12
  finish
}
stage r12.a4 12 default gpu qwen - "D2 on A4: the 36 tool-call scenarios (chat, with their tools) and the golden prompts through b70-serve --spec off / lookup n = 2 / lookup n = 3, greedy, max_tokens 192 - output identical to --spec off"
st_r12_a4() {
  serve_arms "$STATE/$STAGE" 1 "$SNAP_QWEN" "--set toolcall --set golden --max-tokens 192 --stop-at-eos" \
    "off|--max-len 16384 --spec off" "lookup2|--max-len 16384 --spec lookup --spec-min-match 2" \
    "lookup3|--max-len 16384 --spec lookup --spec-min-match 3"
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output through --spec lookup identical to --spec off"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref off --metric wall_tps"
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all ratio '^RATIO ' 10
  grab_all lookup 'spec lookup:|lookup:' 12
  finish
}
stage r12.d5 12 optin gpu-self qwen,opencode_log - "D5 on the opencode recording (OPENCODE_LOG): lookup_accept.py first, then --spec off / --mtp auto / --spec lookup replayed through b70-serve, 3 rounds in rotated order"
st_r12_d5() {
  chk "python3 tools/spec/lookup_accept.py --log $OPENCODE_LOG | tee $STATE/r12.d5.accept.txt" "lookup_accept.py on the recording"
  local r arm args order
  for r in 1 2 3; do
    order="off mtp lookup"; [ "$r" = 2 ] && order="lookup mtp off"
    for arm in $order; do
      case "$arm" in off) args="--spec off" ;; mtp) args="--mtp auto" ;; lookup) args="--spec lookup" ;; esac
      chk "MODEL=$SNAP_QWEN ARMS=on SERVE_ARGS='$args' tools/prefix/replay_ab.sh $OPENCODE_LOG $STATE/r12.d5.r$r.$arm" "replay round $r, $arm"
    done
  done
  grab_all replay 'tok/s|t/s|requests|ttft' 40
  finish
}

# ======================================================================================
row 13 "spec 15d - Ornith 1.5 prefill: device routing, grouped expert GEMMs (spec 15 §11, plan 15d)"
rownote 13 "Everything after r13.kernels waits on 15a, as row 10: the int4 checkpoint (SNAP_ORNITH) and oracle-out-ornith with router_logits.L*; those stages SKIP with 'missing data' until both are on the box (ornith_prefill_* needs the checkpoint only; its R2 part reports itself SKIPPED without the golden set)."
rownote 13 "Calibrating kConsistTieRel / kConsistWeightAbs from ornith_prefill_test's printed distribution and re-deriving prefill_split_ornith's non-64 bars (Qwen3.8's, PROVISIONAL) are edits on a branch after r13.prefill / r13.split record the numbers."
rownote 13 "P0's SLM-fused dequant arm needs tools/probe/probe_moe_prefill.{cl,cc}, not written; it and the rebuild arms (TM, prefetch, a multi-work-group sort) are r13.p0_arms."

stage r13.r0 13 default cpu - g0.sha,g0.bitwise,g0.suite "R0: every pre-existing binary identical (the seven edited prefill sources included), the Qwen3.8 suite incl. prefill_gate_* / prefill_split_* / prefill_replay / prefill_smoke bitwise (G0), Agnes's prefill gates; Ornith's prefill binaries built (the grouped experts, its flash attention, the last chunk's argmax_stage1_M1 - Qwen3.8's since spec 15 §13)"
st_r13_r0() {
  need_pass g0.sha g0.bitwise g0.suite r1.gates
  kbins pf_moe_E256_T8_D2048_I512 pf_moe_router_K2048_N272 pf_moe_gemm_K2048_N1024_SILU \
    pf_moe_gemm_i8_K2048_N1024_SILU pf_moe_gemm_K512_N2048 pf_flash_attn_Q16KV2 argmax_stage1_M1
  jgrab prefill 'prefill_(gate_(l0|int8)|smoke)' 'golden_gate_test OK|TOTAL:|8449|8705|launch'
  finish
}
stage r13.host 13 default cpu - - "host: pf_moe_ref_test (the sort's invariants and tile bound, row independence, combine == decode's, the chunk against decode's chain), memory_plan_test, model_desc_test"
st_r13_host() {
  run_tests '^(pf_moe_ref_test|memory_plan_test|model_desc_test)$'
  finish
}
stage r13.kernels 13 default gpu - - "no checkpoint: pf_moe_test - router rows == decode's gemv_bf16 bitwise, sort / gathers / combine exact, grouped == dense bitwise for bf16 and int8 gate||up and bf16 down (a 1-ulp difference is a finding: read the epilogues), the reversed chunk, replay, C = 37"
st_r13_kernels() {
  run_tests '^pf_moe_test$' ornith
  jgrab pf-moe '^pf_moe_test$' 'bitwise|exact|grouped|dense|ulp|C = 37|OK|PASS|FAIL'
  finish
}
stage r13.prefill 13 default gpu ornith r13.kernels "ornith_prefill_{l0,l0-int8}_test: 2061 / 2021 launches per chunk, replay and chunking bitwise, prefill-vs-decode routes and tokens (the kConsist* distribution printed), R2 on the prefill path"
st_r13_prefill() {
  run_tests '^ornith_prefill_(l0|l0-int8)_test$' ornith
  jgrab ornith-prefill 'ornith_prefill_' 'launch|2061|2021|bitwise|kConsist|tie|R2|SKIPPED|OK|FAIL'
  finish
}
stage r13.gates 13 default gpu ornith,oracle_ornith r13.kernels "R3: prefill_gate_ornith_{l0,l0-int8}_test and _c16 (one chunk and chunks of 16) against oracle-out-ornith, the tie rule"
st_r13_gates() {
  run_tests '^prefill_gate_ornith_' ornith
  jgrab r3 'prefill_gate_ornith' 'golden_gate_test OK|TOTAL:|determined|near-tie|PASS|FAIL'
  finish
}
stage r13.split 13 default gpu ornith r13.kernels "prefill_split_ornith_{l0,l0-int8}_test: splits at multiples of 64 bitwise; the other bars are Qwen3.8's (PROVISIONAL) - the printed numbers re-derive them"
st_r13_split() {
  run_tests '^prefill_split_ornith_' ornith
  jgrab split 'prefill_split_ornith' 'bitwise|cos|split|PASS|FAIL'
  finish
}
stage r13.cli 13 default gpu ornith r13.kernels "b70-decode <ornith> --prefill-length 512 runs on l0-int8 and l0 (--lm-head int8); --prefill-backend sycl-tla stops naming the L0 backends, in b70-decode and in b70-serve (which serves Ornith since 15e: row 16)"
st_r13_cli() {
  chk "$(bench_cmd "$SNAP_ORNITH" --prefill-length 512 --tg 16 --lm-head int8)" "b70-decode --prefill-length 512 (l0-int8)"
  chk "$(bench_cmd "$SNAP_ORNITH" --prefill-length 512 --tg 16 --lm-head int8 --prefill-backend l0)" "b70-decode --prefill-length 512 --prefill-backend l0"
  xfail "timeout 900 build/src/cli/b70-decode $SNAP_ORNITH --bench --prefill-length 512 --tg 1 --prefill-backend sycl-tla" 'prefills on the L0 backends only' "--prefill-backend sycl-tla"
  xfail "timeout 900 build/src/cli/b70-serve $SNAP_ORNITH --port $PORT --max-len 16384 --prefill-backend sycl-tla" 'prefills on the L0 backends only' "b70-serve --prefill-backend sycl-tla"
  grab_all rows '^\| b70-decode' 4
  grab_all pp '^pp: ' 2
  finish
}
stage r13.p0 13 optin gpu ornith r13.kernels "P0 (plan 15d Task 1), the runnable part: B70_PREFILL_PROFILE=1 b70-decode --prefill-length 2048 on l0-int8 and l0 - the moe_* rows (router, route, sort, gather, requant / dequant, GEMM, combine) against the derived ~2.6 GB / layer / chunk of the separate pass"
st_r13_p0() {
  idle before
  chk "B70_PREFILL_PROFILE=1 $(bench_cmd "$SNAP_ORNITH" --prefill-length 2048 --tg 1 --lm-head int8)" "the l0-int8 prefill profile"
  chk "B70_PREFILL_PROFILE=1 $(bench_cmd "$SNAP_ORNITH" --prefill-length 2048 --tg 1 --lm-head int8 --prefill-backend l0)" "the l0 prefill profile"
  idle after
  grab_all moe 'moe|MoE' 60
  grab_all idle '^IDLE '
  finish
}
stage r13.speed 13 optin gpu ornith r13.kernels "Task 4: pp4096, pp32768 and pp130816 (max_len 131072) on l0-int8 and l0, --lm-head int8, interleaved pairs, median of 3 - BENCHMARKS 'Ornith 1.5 MoE (spec 15)' prefill rows"
st_r13_speed() {
  idle before
  local n arms="" ratios=""
  for n in 4096 32768 130816; do
    arms="$arms pp$n-int8 '$(bench_cmd "$SNAP_ORNITH" --prefill-length $n --tg 256 --max-len 131072 --lm-head int8)'"
    arms="$arms pp$n-l0 '$(bench_cmd "$SNAP_ORNITH" --prefill-length $n --tg 256 --max-len 131072 --lm-head int8 --prefill-backend l0)'"
    ratios="$ratios --ratio pp$n-l0/pp$n-int8"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the Ornith prefill arms"
  idle after
  grab_all memory '^memory:' 6
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r13.p0_arms 13 manual - ornith - "P0's arms that need a probe or a rebuild (plan 15d Task 1)"
st_r13_p0_arms() {
  say "# the SLM-fused dequant arm: tools/probe/probe_moe_prefill.{cl,cc} (not written) against the separate pass r13.p0 measured"
  say "# the grouped GEMMs' TFLOP/s at the real row distribution (15a's router statistics); TM 16 / 32 / 64; prefetch + split barrier in pf_moe_gemm; a multi-work-group sort if moe_sort shows in r13.p0"
  say "# each: rebuild with the define, pf_moe_test bitwise, then B70_PREFILL_PROFILE=1 build/src/cli/b70-decode $SNAP_ORNITH --bench --prefill-length 2048 --tg 1 --lm-head int8"
  say "# record: docs/probe-ornith-prefill-2026-10-04.md"
}

# ======================================================================================
row 14 "spec 18b - K2-Horizon decode on one card (spec 18 §10, plan 18b)"
rownote 14 "Load, K3, CLI and speed need the int4 checkpoint (SNAP_K2: hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ, 21.8 GB); K2 also needs oracle-out-k2, which the opt-in r14.oracle makes on the box CPU (--with r14.oracle, ~20 min). Without them those stages SKIP with 'missing data'."
rownote 14 "Setting the near-tie tolerance from the gap distribution r14.oracle prints (k2_golden_test.cc proposes 1e-3) is an edit after it; r14.golden runs at the proposal, or at B70_K2_TIE_TOL when the driver's environment sets it."
rownote 14 "B70_K2_ATTN=eager's decode speed: r15.speed's eager pairs carry tg256 at depth 4k / 16k / 32k after a prefill; by hand, B70_K2_ATTN=eager before r14.speed's command lines."
rownote 14 "If K2's determined rows fail, the first suspect is decode attention's fp32 probabilities against the reference's eager bf16 scores / probabilities (spec 18 §10): r14.golden_eager runs the same gate through the eager variant (B70_K2_ATTN=eager, spec 18 §10.1); compare the two, and §10.1's rule decides the default."

stage r14.k0 14 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (no existing .cl changed), the Qwen3.8 suite (G0); Agnes's and Ornith's gates not failed (a SKIP for missing data is noted, not counted)"
st_r14_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r1.gates r1.features r10.nockpt r10.gates r13.kernels r13.gates
  finish
}
stage r14.host 14 default cpu - - "host: k2_horizon, k2_rope, k2_repack, k2_ref, k2_attn_eager_ref (the eager chain bitwise against torch), k2_plan, k2_variant_names, model_desc (the power-of-two expert-count refusal)"
st_r14_host() {
  run_tests '^(k2_horizon_test|k2_rope_test|k2_repack_test|k2_ref_test|k2_attn_eager_ref_test|k2_plan_test|k2_variant_names_test|model_desc_test)$'
  finish
}
stage r14.k1 14 default gpu - - "K1, no checkpoint: k2_kernels_test (norm / prep bit-exact, routers incl. ties and the padded lanes, MoE and MoVA within 2 ulps, attention at 6 / 300 / 3000 keys with softplus gates on both sides of 28.85, replay; §9 the eager attention bitwise against k2_ref at 6 / 300 / 3000 / 4096 keys); cli_reject_mtp_k2 (18c retired _prefill / _pp: r15.k1 runs their successors; 18e retired cli_reject_k2_kv8 - K2 takes --kv-cache int8)"
st_r14_k1() {
  run_tests '^k2_kernels_test$' k2
  run_tests '^cli_reject_mtp_k2$'
  jgrab k1 '^k2_kernels_test$' 'ulp|bit-exact|bitwise|OK|PASS|FAIL'
  finish
}
stage r14.load 14 default gpu k2 r14.k1 "k2_load_checkpoint_test and _i8head: every bucket = loader::k2_weight_bytes (21,801,501,440 / 21,160,906,496 B), read/token 3.786 / 3.145 GB, 0 unconsumed, readbacks (~5 min each)"
st_r14_load() {
  run_tests '^k2_load_checkpoint(_i8head)?_test$' k2
  jgrab load 'k2_load_checkpoint' 'bytes|read/token|per token|unconsumed|OK|FAIL'
  finish
}
stage r14.k3 14 default gpu k2 r14.load "K3: k2_decode_test and _i8head - 717 launches, plan == allocation, replay bitwise incl. route rows and KV"
st_r14_k3() {
  run_tests '^k2_decode(_i8head)?_test$' k2
  jgrab k3 'k2_decode' '717|launch|plan|allocation|bitwise|OK|FAIL'
  finish
}
stage r14.oracle 14 optin cpu k2,oracle_image - "18a's real-weight reference on the box CPU: tools/oracle/k2_ref.py run on the int4 checkpoint, 3 prompts x 32 ids + the routing dumps -> \$DATA/oracle-out-k2 (resumable per prompt, ~20 min; needs MemAvailable >= K2_REF_MIN_GB = 32), the gap distribution; then re-links oracle-out* and refreshes the data list for r14.golden"
st_r14_oracle() {
  chk "tools/box_validate/k2_oracle.sh $DATA" "the K2-Horizon reference run"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_(k2|oracle_k2)=' $STATE/have.env"
  grab_all gap 'selection gap' 12
  finish
}
stage r14.golden 14 default gpu k2,oracle_k2 r14.k3 "K2: k2_golden_test and _i8head against oracle-out-k2 - the tie-aware token gate and the per-layer routing diagnostic (a non-tie set difference fails; near-tie tolerance 1e-3 proposed, B70_K2_TIE_TOL overrides)"
st_r14_golden() {
  x "grep -h -E 'selection gap' oracle-out-k2/*.log 2>/dev/null || echo 'no gap lines in oracle-out-k2/*.log'"
  run_tests '^k2_golden(_i8head)?_test$' k2 '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden 'k2_golden' 'tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r14.golden_eager 14 default gpu k2,oracle_k2 r14.k3 "K2 through the EAGER attention (B70_K2_ATTN=eager, spec 18 §10.1; 813 launches): k2_golden_eager_test and _i8head - compare with r14.golden (determined-row failures, non-tie / near-tie routing counts, first differing row per layer); §10.1's rule decides the default"
st_r14_golden_eager() {
  run_tests '^k2_golden_eager(_i8head)?_test$' k2 '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden-eager 'k2_golden_eager' 'attention|tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r14.cli 14 default gpu k2 r14.k1 "CLI: b70-decode <k2> --ids (oracle-out-k2/prose.ids, else the committed prose.ids) --n 32; --max-len auto -> ~46592 (bf16 head) / ~49920 (--lm-head int8), derived; b70-serve <k2> --mtp 1 refuses before the device (K2 has no MTP head; serving K2 is row 25's)"
st_r14_cli() {
  chk "ids=oracle-out-k2/prose.ids; [ -s \$ids ] || ids=tests/golden/prompts/prose.ids; echo \"ids: \$ids\"; timeout 1800 build/src/cli/b70-decode $SNAP_K2 --ids \$ids --n 32 | tr '\\n' ' '; echo" "b70-decode --ids --n 32"
  chk "n=\$(tools/box_validate/auto_len.sh $SNAP_K2) && echo \"K2 max_len auto, bf16 head: \$n\"" "--max-len auto, bf16 head"
  chk "n=\$(tools/box_validate/auto_len.sh $SNAP_K2 --depth 16 --tg 1 --lm-head int8) && echo \"K2 max_len auto, int8 head: \$n\"" "--max-len auto, int8 head"
  xfail "timeout 900 build/src/cli/b70-serve $SNAP_K2 --port $PORT --max-len 16384 --mtp 1" 'K2-Horizon has no MTP head' "b70-serve --mtp 1"
  grab_all generate '^generate:|^ids: ' 4
  grab_all auto '^K2 max_len auto' 2
  finish
}
stage r14.speed 14 optin gpu k2 r14.k3 "Task 4: decode at depth 4096 / 16384 / 32768, tg 256, bf16 and int8 heads (max_len 40960), interleaved pairs, median of 3, against the derived roofline (~190 t/s weights-only, int8 head); BENCHMARKS 'K2-Horizon (spec 18)'"
st_r14_speed() {
  idle before
  local d arms="" ratios=""
  for d in 4096 16384 32768; do
    arms="$arms d$d '$(bench_cmd "$SNAP_K2" --depth $d --tg 256 --max-len 40960)'"
    arms="$arms d$d-int8 '$(bench_cmd "$SNAP_K2" --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    ratios="$ratios --ratio d$d-int8/d$d"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the K2 decode arms"
  idle after
  grab_all memory '^memory:' 6
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r14.sweeps 14 manual - k2 - "Task 4's sweeps: the {S, layout} cells of the five int4 shapes and the expert kernels' K splits"
st_r14_sweeps() {
  say "# {S, layout} of 2560x10240, 2560x9280, 4096x2560, 2560x12288, 6144x2560 with their GEMV_* defines (probe_gemv rows, as spec 14 step 4)"
  say "# UP_KS / DN_KS / MOVA_KS: rebuild k2_moe.cl's variants per value, k2_kernels_test, then build/src/cli/b70-decode $SNAP_K2 --bench --depth 4096 --tg 256 --lm-head int8 per value"
  say "# --profile is refused on K2 until Task 4 builds it; record: docs/BENCHMARKS.md 'K2-Horizon (spec 18)'"
}

# ======================================================================================
row 15 "spec 18c - K2-Horizon prefill: grouped MoVA and MoE, flash attention at head_dim 128 (spec 18 §11, plan 18c)"
rownote 15 "K1 (r15.k1) needs no checkpoint; K3, split and the CLI need SNAP_K2, the golden prefill tests oracle-out-k2 as well (--with r14.oracle makes it on the box CPU). Without them those stages SKIP with 'missing data'."
rownote 15 "The PROPOSED bars (prefill KV against decode's fill: dense rows >= 0.999, median >= 0.9998, p01 >= 0.99; routing near-tie margin 2e-2, weights 1/32) are set from the distributions r15.prefill prints - an edit on a branch after the run."
rownote 15 "P0's rebuild arms (flash EXP2 0 vs 1, RPW 8 / 16, the sort's single work-group, an SLM-fused dequant in the grouped GEMM) are r15.p0_arms; r15.p0 is the runnable profile."

stage r15.k0 15 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (no existing .cl changed), the Qwen3.8 suite (G0); Agnes's, Ornith's and K2 decode's gates not failed (a SKIP for missing data is noted, not counted)"
st_r15_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r1.gates r10.gates r13.gates r14.k1 r14.k3 r14.golden
  finish
}
stage r15.host 15 default cpu - - "host: k2_pf_ref_test (the routing scatter with exact ties, tile bound, row independence, combines == decode's chains, slab tail, eager reference), k2_pf_variant_names_test, k2_plan_test (2392 launches, 0.797 GB scratch, auto 42752 with prefill)"
st_r15_host() {
  run_tests '^(k2_pf_ref_test|k2_pf_variant_names_test|k2_plan_test)$'
  finish
}
stage r15.k1 15 default gpu - - "K1, no checkpoint: k2_pf_kernels_test (slab exact, linear cos 0.99999; norm / prep bit-exact; routers; MoE and MoVA grouped == dense bitwise - a 1-ulp difference is a finding; reversed chunk and replay bitwise; MoVA V rows cos 0.9999; flash attention against fp64 at 8 (pos, C) cases: default >= 0.99999, EAGER >= 0.9999 / 0.999, gated within 1 ulp); cli_reject_k2_prefill_length_int8, cli_reject_k2_prefill_sycl"
st_r15_k1() {
  run_tests '^k2_pf_kernels_test$' k2
  run_tests '^cli_reject_k2_(prefill_length_int8|prefill_sycl)$'
  jgrab k1-prefill '^k2_pf_kernels_test$' 'cos|ulp|bit-exact|bitwise|grouped|dense|OK|PASS|FAIL'
  finish
}
stage r15.prefill 15 default gpu k2 r15.k1 "K3: k2_prefill_test and _i8head (2392 / chunk + 5 launches, plan == allocation, immediate / recorded / replayed bitwise, chunks of 64 bitwise; prefill KV and routes against decode's fill - the PROPOSED bars' distributions printed; tokens by A26's rule); k2_prefill_eager_test (B70_K2_ATTN=eager: prefill's EAGER flash and decode's eager attention agree - matched, not bitwise)"
st_r15_prefill() {
  run_tests '^k2_prefill(_i8head|_eager)?_test$' k2
  jgrab k3-prefill '^k2_prefill(_i8head)?_test$' '2392|launch|plan|allocation|bitwise|cos|median|p01|tie|token|OK|FAIL'
  jgrab k3-prefill-eager '^k2_prefill_eager_test$' 'eager|attention|cos|median|p01|tie|agree|OK|FAIL'
  finish
}
stage r15.split 15 default gpu k2 r15.k1 "prefill_split_k2_test: splits at multiples of 64 bitwise; the argument predicts every split bitwise - read a non-bitwise one before accepting the bars"
st_r15_split() {
  run_tests '^prefill_split_k2_test$' k2
  jgrab split '^prefill_split_k2_test$' 'bitwise|cos|split|PASS|FAIL'
  finish
}
stage r15.golden 15 default gpu k2,oracle_k2 r15.prefill "K2 on prefill: k2_golden_prefill_test, _c16 (chunks of 16), _i8head against oracle-out-k2 - the tie rule and the routing diagnostic on the prompt rows (B70_K2_TIE_TOL as r14.golden)"
st_r15_golden() {
  run_tests '^k2_golden_prefill(_c16|_i8head)?_test$' k2 '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden-prefill 'k2_golden_prefill' 'tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r15.golden_eager 15 optin gpu k2,oracle_k2 r15.prefill "the golden prefill gates under B70_K2_ATTN=eager (k2_golden_test prefill / prefill:16 / int8 prefill) - for when r15.golden fails determined rows (spec 18 §11)"
st_r15_golden_eager() {
  local m tol="${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL }"
  for m in prefill prefill:16 'int8 prefill'; do
    chk "${tol}B70_K2_ATTN=eager timeout 3600 build/tests/k2_golden_test $SNAP_K2 oracle-out-k2 $m" "k2_golden_test $m, eager attention"
  done
  grab_all k2-golden-prefill-eager 'tie|routing|determined|differ|PASS|FAIL' 30
  finish
}
stage r15.cli 15 default gpu k2 r15.prefill "CLI: b70-decode <k2> --ids (oracle-out-k2/prose.ids, else the committed prose.ids) --n 32 with and without --prefill - compared (a difference is judged by the tie rule, not failed here); --max-len auto with a prefill planned -> ~42752 (bf16 head, derived)"
st_r15_cli() {
  local p flag
  for p in decode prefill; do
    flag=""; [ "$p" = prefill ] && flag=" --prefill"
    chk "ids=oracle-out-k2/prose.ids; [ -s \$ids ] || ids=tests/golden/prompts/prose.ids; echo \"ids: \$ids\"; timeout 1800 build/src/cli/b70-decode $SNAP_K2 --ids \$ids --n 32$flag > $STATE/$STAGE.$p.out" "b70-decode --ids --n 32$flag"
  done
  x "for f in decode prefill; do printf '%s: ' \$f; tr '\\n' ' ' < $STATE/$STAGE.\$f.out; echo; done; if cmp -s $STATE/$STAGE.decode.out $STATE/$STAGE.prefill.out; then echo 'PREFILL ids identical to the decode-only run'; else echo 'PREFILL ids differ from the decode-only run: judge the first difference by the tie rule'; fi"
  chk "n=\$(tools/box_validate/auto_len.sh $SNAP_K2 --prefill-length 256 --tg 1) && echo \"K2 max_len auto, prefill planned, bf16 head: \$n\"" "--max-len auto with a prefill planned"
  grab_all prefill-ids '^PREFILL ids' 2
  grab_all auto '^K2 max_len auto' 2
  finish
}
stage r15.p0 15 optin gpu k2 r15.k1 "P0 (plan 18c Task 1), the runnable part: B70_PREFILL_PROFILE=1 b70-decode <k2> --bench --prefill-length 4096, flash and B70_K2_ATTN=eager - the moe_* / slab_* / attn_flash rows against the derived ~1.2 s (the per-chunk expert dequant the largest term)"
st_r15_p0() {
  idle before
  chk "B70_PREFILL_PROFILE=1 $(bench_cmd "$SNAP_K2" --prefill-length 4096 --tg 1)" "the K2 prefill profile"
  chk "B70_K2_ATTN=eager B70_PREFILL_PROFILE=1 $(bench_cmd "$SNAP_K2" --prefill-length 4096 --tg 1)" "the K2 prefill profile, eager attention"
  idle after
  grab_all profile 'moe|mova|slab|attn|flash|dequant|sort|gather|combine' 80
  grab_all idle '^IDLE '
  finish
}
stage r15.speed 15 optin gpu k2 r15.prefill "Task 3: pp4096 / pp16384 / pp32768 then tg256 (max_len 40960, --lm-head int8), flash against B70_K2_ATTN=eager in interleaved pairs, median of 3 - BENCHMARKS 'K2-Horizon (spec 18)' prefill rows (derived pp4096 ~3,400 t/s)"
st_r15_speed() {
  idle before
  local n arms="" ratios=""
  for n in 4096 16384 32768; do
    arms="$arms pp$n '$(bench_cmd "$SNAP_K2" --prefill-length $n --tg 256 --max-len 40960 --lm-head int8)'"
    arms="$arms pp$n-eager 'B70_K2_ATTN=eager $(bench_cmd "$SNAP_K2" --prefill-length $n --tg 256 --max-len 40960 --lm-head int8)'"
    ratios="$ratios --ratio pp$n-eager/pp$n"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the K2 prefill arms"
  idle after
  grab_all memory '^memory:' 6
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r15.p0_arms 15 manual - k2 - "P0's arms that need a rebuild (plan 18c Task 1)"
st_r15_p0_arms() {
  say "# k2_pf_flash_attn: EXP2 0 (natural exp, as decode and the reference - PROVISIONAL) vs 1; RPW 8 vs 16"
  say "# the sort's single work-group (k2_pf_sort) if it shows in r15.p0; an SLM-fused dequant inside pf_moe_gemm against the per-chunk bf16 pass"
  say "# each: rebuild with the define, k2_pf_kernels_test bitwise, then B70_PREFILL_PROFILE=1 build/src/cli/b70-decode $SNAP_K2 --bench --prefill-length 4096 --tg 1"
  say "# record: docs/probe-k2-prefill-2026-10-05.md"
}

# ======================================================================================
row 16 "spec 15e - Ornith 1.5 served: template, tool calls, the MoE MTP head (spec 15 §12, plan 15e)"
rownote 16 "Everything after r16.kernels needs the int4 checkpoint (SNAP_ORNITH, 15a) and should follow rows 10 and 13; mtp_head_ornith_test (M1) also needs 15a's oracle-out-ornith-mtp/. Without them those stages SKIP with 'missing data'."
rownote 16 "Since spec 15 §13 Ornith's greedy argmax masks from 248077, the int4 checkpoint's tokenizer.json count (Qwen3.8's argmax_stage1_M{1..4}; the _V248070 binaries stay built, unbound), so b70-serve's startup note comparing the two does not fire for Ornith: r16.serve checks it is absent. Its a||b is int4 (gemv_M{1..4}_K2048_N128_S1_L1; r16.r0 checks they are built)."
rownote 16 "An Ornith --mtp-cost default (and a default K) from r16.cost's table is an edit on a branch after the run."
rownote 16 "A4 against the Ornith reference (plan 15e Task 3 Step 1) needs oracle-out-ornith-a4/ - the Ornith tool-call set (Qwen3.8's 36 conversations rendered by Ornith's own template) and the reference's <name>.bf16.txt - made on the Mac from the int4 checkpoint (r16.a4_ref: tools/toolcall/a4_ref.sh ornith set / ref, hours) and copied by --push-data; r16.a4 (opt-in) then runs the engine and scores. No bar: recorded beside Qwen3.8's 25/36."
rownote 16 "Determinism and replay on Ornith (plan 15e Task 3 Step 1) are row 10's / 13's gates (ornith_decode_test, ornith_prefill_* replay and chunking bitwise); C2 is r16.prefix, passkey r16.passkey (opt-in), the comparison rows r16.benchy (opt-in). The record (BENCHMARKS 'Ornith 1.5 MoE (spec 15)', spec 15 §10) is filled from those numbers on a branch."

stage r16.r0 16 default gpu qwen - "R0: every pre-existing binary identical (no .cl changed), the Qwen3.8 suite (G0); 15e's binaries built (moe_M{2,3,4}, gdn_step_slots_M{1..4}_G30_GK16V32, the prefill head-KV fill, the int4 a||b gemv_M{1..4}_K2048_N128_S1_L1 and Qwen3.8's argmax_stage1_M{1..4}, spec 15 §13); Qwen3.8's MTP suite unchanged (mtp_head, mtp_verify, mtp_gpu, load_checkpoint: the draft list's dense branch and the launch asserts are new code on that path)"
st_r16_r0() {
  need_pass g0.sha g0.bitwise g0.suite
  kbins argmax_stage1_M1 argmax_stage1_M2 argmax_stage1_M3 argmax_stage1_M4 \
    gemv_M1_K2048_N128_S1_L1 gemv_M2_K2048_N128_S1_L1 gemv_M3_K2048_N128_S1_L1 gemv_M4_K2048_N128_S1_L1 \
    moe_M2_E256_T8_D2048_I512 moe_M3_E256_T8_D2048_I512 moe_M4_E256_T8_D2048_I512 \
    gdn_step_slots_M1_G30_GK16V32 gdn_step_slots_M2_G30_GK16V32 gdn_step_slots_M3_G30_GK16V32 gdn_step_slots_M4_G30_GK16V32 \
    pf_res_fold_K2048_SP1_G20_Z pf_norm_finish_K2048_G20_W20_X4096 pf_bf16_slab_K4096 pf_bf16_slab_K2048
  run_tests '^(mtp_head_test|mtp_verify_test|mtp_gpu_test|load_checkpoint_test)$'
  jgrab mtp-bytes '^load_checkpoint_test$' '849|MTP'
  finish
}
stage r16.host 16 default cpu - - "host: template_ornith_test (six lists byte-identical to transformers 5.18.0), ornith_server_test, ornith_mtp_head_test, ornith_mtp_names_test (every bound name against the built ones, both a||b forms included), model_desc_test (vocab_used 248077, doc_w, the int4 a||b row), variant_names_test, memory_plan_test, ab_int4_test"
st_r16_host() {
  run_tests '^(template_ornith_test|ornith_server_test|ornith_mtp_head_test|ornith_mtp_names_test|model_desc_test|variant_names_test|memory_plan_test|ab_int4_test)$'
  finish
}
stage r16.kernels 16 default gpu - - "no checkpoint: moe_m_test - moe.cl at M = 2..4, every row bitwise M = 1's, rows reversed (Review Focus 3)"
st_r16_kernels() {
  run_tests '^moe_m_test$' ornith
  jgrab moe-m '^moe_m_test$' 'bitwise|M = |rows|OK|PASS|FAIL'
  finish
}
stage r16.mtp 16 default gpu ornith r16.kernels "ornith_mtp_test and _bf16head: the head's load (785 tensors, 1.689 GB, the RTN line and its seconds, 0 unconsumed), lists 536 / 24; M2 - logits rows, GDN slots and all 40 layers' route rows bitwise for k = 1..3 on prose / code / cjk; M3 - greedy --mtp K == a head-less engine over 128 ids; acceptance and ms/id per K"
st_r16_mtp() {
  run_tests '^ornith_mtp(_bf16head)?_test$' ornith
  jgrab ornith-mtp 'ornith_mtp' 'MTP head|RTN|unconsumed|536|24|M2|M3|accept|ms/id|route|OK|PASS|FAIL'
  finish
}
stage r16.m1 16 default gpu ornith,oracle_ornith_mtp r16.kernels "M1: mtp_head_ornith_test - the head's draft logits against 15a's oracle-out-ornith-mtp/"
st_r16_m1() {
  run_tests '^mtp_head_ornith_test$' ornith
  jgrab m1 '^mtp_head_ornith_test$' 'cos|top|M1|OK|PASS|FAIL'
  finish
}
stage r16.serve 16 default gpu ornith r16.kernels "b70-serve <ornith> startup: --max-len auto -> 262144 with and without --mtp auto (derived), the --mtp auto cost note (Qwen3.8's table until r16.cost), the MTP head's load line; NO tokenizer / argmax vocab note (both 248077 since spec 15 §13); then the A4 set as chat requests with their tools, greedy, 192 ids (tool calls back as OpenAI tool calls)"
st_r16_serve() {
  serve "$SNAP_ORNITH"
  serve "$SNAP_ORNITH" --mtp auto
  chk "if grep -v '^+ ' $LOG | grep -E 'note: tokenizer[.]json defines'; then echo 'the vocab note fired: tokenizer.json and the descriptor disagree'; exit 1; fi; echo 'no vocab note: tokenizer.json and the descriptor agree'" \
    "b70-serve's tokenizer / argmax vocab note absent for Ornith"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE.toolcall toolcall 1 $SNAP_ORNITH --max-len 16384 -- --set toolcall --max-tokens 192 --stop-at-eos" "the A4 tool-call set through b70-serve <ornith>"
  x "grep -o '\"finish_reason\": \"[a-z_]*\"' $STATE/$STAGE.toolcall/requests.jsonl | sort | uniq -c"
  grab_all auto 'max_len: auto ->'
  grab_all memory '^memory:'
  grab_all mtp-head 'MTP head|RTN|unconsumed' 6
  grab_all mtp-cost '^mtp auto:' 4
  grab_all vocab-note 'vocab note' 2
  grab_all finish '^ *[0-9]+ "finish_reason"' 4
  finish
}
stage r16.golden_server 16 default gpu ornith r16.kernels "the server's greedy chat == b70-decode on Ornith: golden_server_test <b70-serve> <b70-decode> tests/golden/prompts <ornith>, run directly (not registered for Ornith: without a checkpoint it would fail, not SKIP)"
st_r16_golden_server() {
  chk "timeout 1800 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_ORNITH" "golden_server_test against Ornith"
  finish
}
stage r16.prefix 16 default gpu ornith r16.kernels "spec 7 C2 on Ornith: prefix_gpu_test <ornith> on l0-int8, on l0 (near-tie allowance 1) and on l0-int8 with --mtp 3 - run directly, as golden_server_test"
st_r16_prefix() {
  chk "timeout 1800 build/tests/prefix_gpu_test $SNAP_ORNITH tests/golden/prompts l0-int8" "prefix_gpu_test <ornith> l0-int8"
  chk "timeout 1800 build/tests/prefix_gpu_test $SNAP_ORNITH tests/golden/prompts l0 1" "prefix_gpu_test <ornith> l0"
  chk "timeout 1800 build/tests/prefix_gpu_test $SNAP_ORNITH tests/golden/prompts l0-int8 1 0 3" "prefix_gpu_test <ornith> l0-int8 --mtp 3"
  finish
}
stage r16.cost 16 default gpu ornith r16.mtp "Review Focus 2: the verify cost at K = 1..3 on Ornith (up to 8 x (K + 1) experts) - probe_mtp_steps (verify M = 1..4, draft k = 1..3), int8 and bf16 heads (timed)"
st_r16_cost() {
  idle before
  chk "build/tools/probe/probe_mtp_steps $SNAP_ORNITH 4096 32 3 int8" "probe_mtp_steps on Ornith, int8 head"
  chk "build/tools/probe/probe_mtp_steps $SNAP_ORNITH 4096 32 3 bf16" "probe_mtp_steps on Ornith, bf16 head"
  idle after
  grab_all cost-rows '^lm_head:|^\| ' 40
  grab_all idle '^IDLE '
  finish
}
stage r16.tokdiff 16 default cpu qwen,ornith - "Ornith's tokenizer.json against Qwen3.8's beyond the added tokens (vocabulary, merges, normalizer / pre-tokenizer / decoder): tools/box_validate/tok_diff.py - a record, not a gate"
st_r16_tokdiff() {
  chk "python3 tools/box_validate/tok_diff.py \$(tools/box_validate/data.sh resolve $SNAP_QWEN)/tokenizer.json \$(tools/box_validate/data.sh resolve $SNAP_ORNITH)/tokenizer.json" "the tokenizer.json comparison"
  grab_all tokdiff '^TOKDIFF ' 20
  finish
}
stage r16.mtp_rows 16 optin gpu ornith r16.mtp "D1-style rows on Ornith through b70-serve: --mtp off / 1 / 3 / auto on the golden and A4 prompts, 256 ids, greedy, R2_ROUNDS rotated; greedy output identical across the arms (decode t/s per request from the server's records)"
st_r16_mtp_rows() {
  idle before
  serve_arms "$STATE/$STAGE" "${R2_ROUNDS:-3}" "$SNAP_ORNITH" "--set golden --set toolcall --max-tokens 256" \
    "off|--max-len 16384 --mtp off" "k1|--max-len 16384 --mtp 1" "k3|--max-len 16384 --mtp 3" \
    "auto|--max-len 16384 --mtp auto"
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output identical to --mtp off in every arm"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref off --ratio k1/off --ratio k3/off --ratio auto/off"
  grab_all ratio '^RATIO ' 20
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all idle '^IDLE '
  finish
}
stage r16.passkey 16 optin gpu ornith,oracle_image r16.kernels "Review Focus 4: passkey 3/3 at 120k (l0-int8, l0; max_len 131072) and at 250000 ids (l0-int8, max_len 262144 - Ornith's KV is 20 KiB a position)"
st_r16_passkey() {
  chk "MODEL=$SNAP_ORNITH ORACLE_MODEL=$ORNITH_ORACLE_MODEL tools/probe/passkey.sh l0-int8 l0" "passkey 120k on Ornith"
  chk "MODEL=$SNAP_ORNITH ORACLE_MODEL=$ORNITH_ORACLE_MODEL MAX_LEN=262144 N_TARGET=250000 tools/probe/passkey.sh l0-int8" "passkey 250k on Ornith"
  grab_all passkey '^passkey [^ ]+: [0-9]/3'
  finish
}
stage r16.benchy 16 optin gpu-self ornith,uvx - "plan 15e Task 3 Step 2: llama-benchy through b70-serve <ornith> - pp4096 tg256 depth 1 (--no-cache --exact-tg --latency-mode generation), then prefix caching at depth 4k / 16k / 32k, cache on vs off"
st_r16_benchy() {
  chk "MODEL=$SNAP_ORNITH MAX_LEN=65536 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation" "llama-benchy pp4096 tg256"
  chk "MODEL=$SNAP_ORNITH MAX_LEN=65536 SERVE_ARGS='--mtp 0' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, prefix cache on"
  chk "MODEL=$SNAP_ORNITH MAX_LEN=65536 SERVE_ARGS='--mtp 0 --prefix-cache-gb 0' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, prefix cache off"
  grab_all benchy '^\| ' 60
  finish
}
stage r16.a4_ref 16 manual - ornith - "plan 15e Task 3 Step 1's data, ON THE MAC: the Ornith tool-call set and its reference from the int4 checkpoint (tools/toolcall/a4_ref.sh; one 28 GB container at a time), then --push-data"
st_r16_a4_ref() {
  say "tools/toolcall/a4_ref.sh ornith set      # oracle-out-ornith-a4/set: make_set.py --from tests/golden/toolcall with Ornith's template (minutes)"
  say "tools/toolcall/a4_ref.sh ornith ref      # oracle-out-ornith-a4/<name>.bf16.{ids,txt}: ornith_ref.py streamed, 192 new ids, detached (hours)"
  say "tools/toolcall/a4_ref.sh ornith status   # until DONE; then tools/box_validate.sh --push-data copies oracle-out-ornith-a4 to the box"
}
stage r16.a4 16 optin gpu ornith,oracle_ornith_a4 r16.kernels "plan 15e Task 3 Step 1: A4 on Ornith against its reference (oracle-out-ornith-a4) - engine_generate.sh on Ornith's own set (l0-int8 with the bf16 and the int8 head, l0), 192 ids, score.py; then the set as chat requests through b70-serve (tool calls back as OpenAI tool calls, the prompt ids = the set's)"
st_r16_a4() {
  local d=$STATE/a4-ornith
  x "mkdir -p $d && cp oracle-out-ornith-a4/*.bf16.txt $d/"
  chk "tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_ORNITH) oracle-out-ornith-a4/set $d l0-int8" "A4 engine runs, l0-int8, bf16 head"
  chk "LM_HEAD=int8 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_ORNITH) oracle-out-ornith-a4/set $d l0-int8" "A4 engine runs, l0-int8, int8 head (the served default)"
  chk "tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_ORNITH) oracle-out-ornith-a4/set $d l0" "A4 engine runs, l0"
  chk "python3 tools/toolcall/score.py $d bf16 l0-int8 l0-int8-i8head l0 | tee $d/score.md" "scoring against the reference"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE.serve ornith-set 1 $SNAP_ORNITH --max-len 16384 -- --set oracle-out-ornith-a4/set --max-tokens 192 --stop-at-eos" "Ornith's set through b70-serve"
  x "grep -o '\"finish_reason\": \"[a-z_]*\"' $STATE/$STAGE.serve/requests.jsonl | sort | uniq -c; grep -c '\"prompt_ids_match\": true' $STATE/$STAGE.serve/requests.jsonl"
  grab_all a4 'match|/ 36|/36' 8
  finish
}

# ======================================================================================
row 17 "spec 18d host side - K2-Horizon's template, tokenizer, tool calls and server dispatch (spec 18 §12, plan 18d)"
rownote 17 "No device code: G0's g0.sha is the 'no kernel touched' check. Since 18d's engine side (row 25) b70-serve serves K2; r14.cli checks its --mtp refusal instead of the old one."
rownote 17 "The engine half of 18d (K2Engine behind b70-serve, KV-only prefix snapshots, the greedy chat through the server = b70-decode, K4, the comparison rows) is row 25; it follows rows 14, 15 and 21 on the card."
stage r17.host 17 default cpu - - "the 18d host tests: template_k2_test (18 lists byte-identical, the vendored tests/tokenizer/k2/), toolcall_k2_test, k2_server_test, minja_ext_test (the minja patches against Jinja2), k2_tokenizer_test (K2's tokenizer.json from the int4 snapshot in the HF cache, or B70_K2_TOKENIZER_JSON; SKIP without)"
st_r17_host() {
  run_tests '^(template_k2_test|k2_tokenizer_test|toolcall_k2_test|k2_server_test|minja_ext_test)$'
  jgrab k2-tokenizer '^k2_tokenizer_test$' 'SKIP|vocab|corpus|digest|OK|FAIL'
  finish
}
stage r17.k0 17 default gpu qwen - "K0 for the server path on the box's snapshots (the patched minja under the Qwen3.8 / Agnes / Ornith templates; ChatFormat / make_output_parser on the Qwen path): template_test, template_agnes_test, template_ornith_test, ornith_server_test, toolcall_test, protocol_test, golden_server_test, prefix_server_test, mtp_server_test, lookup_server_test - unchanged"
st_r17_k0() {
  need_pass g0.sha
  run_tests '^(template_test|template_agnes_test|template_ornith_test|ornith_server_test|toolcall_test|protocol_test|golden_server_test|prefix_server_test|mtp_server_test|lookup_server_test)$'
  finish
}

# ======================================================================================
row 18 "compressed-tensors symmetric pack-quantized loader (branch ct-sym-loader; spec 20 §9 as built, docs/13)"
rownote 18 "A real symmetric checkpoint on the card (r18.ct) needs RedHatAI/Qwen3.8-27B-INT4 downloaded (~20 GB); its golden prompts need its own CPU reference (tools/oracle/golden.sh with ORACLE_MODEL pointing at it). Both are manual."
stage r18.host 18 default cpu - - "host: ct_loader_test (exact repack g64 / g128 x f16 / bf16, the note, the refusals, the real configs), quant_test, ornith_repack_test and k2_repack_test (compressed-tensors MoE layer and K2 checkpoint == the GPTQ bytes)"
st_r18_host() {
  run_tests '^(ct_loader_test|quant_test|ornith_repack_test|k2_repack_test)$'
  finish
}
stage r18.g64 18 default gpu qwen,oracle_qwen - "the g64 checkpoint unchanged through the reworked classify / consumed sets: load_checkpoint_test (0 unconsumed, no compressed-tensors note) and golden_gate_test"
st_r18_g64() {
  run_tests '^(load_checkpoint_test|golden_gate_test)$'
  jgrab g64 'load_checkpoint_test' 'unconsumed|compressed-tensors|W check|OK|FAIL'
  finish
}
stage r18.ct 18 manual - - - "RedHatAI/Qwen3.8-27B-INT4 (sym int4 g128, bf16 scales): the load note, 0 unconsumed, the W check, decode, its golden prompts against its own CPU reference"
st_r18_ct() {
  say "uvx --from huggingface_hub hf download RedHatAI/Qwen3.8-27B-INT4"
  say "flock ~/b70-gpu.lock build/src/cli/b70-decode RedHatAI/Qwen3.8-27B-INT4 --bench --tg 64 2>&1 | tee r18.ct.log   # the ONE 'converting compressed-tensors pack-quantized (g128, 400 linears)' note; 0 unconsumed; 32 fp8 KV scales dropped; W check within 2 % of the g64 checkpoint's"
  say "flock ~/b70-gpu.lock build/src/cli/b70-decode RedHatAI/Qwen3.8-27B-INT4 --mtp 1 --bench --tg 64   # the bf16 MTP head (15 tensors) loads beside it"
  say "ORACLE_MODEL=models--RedHatAI--Qwen3.8-27B-INT4 OUT_DIR=oracle-out-rh-int4 tools/oracle/golden.sh   # its own CPU reference (tools/oracle/README.md)"
  say "flock ~/b70-gpu.lock build/tests/golden_gate_test oracle-out-rh-int4 tests/golden/prompts <its snapshot dir>"
}

# ======================================================================================
row 19 "the real Ornith int4 checkpoint - its int4 a||b, vocab_used 248077, doc_w (branch ornith-real-checkpoint; spec 15 §13)"
rownote 19 "Rows 10, 13 and 16 run on this checkpoint (SNAP_ORNITH = urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ): their gates are where its a||b is exercised end to end (decode, prefill, the MTP verify lists). oracle-out-ornith / -mtp are made on the Mac from the int4 checkpoint (plan 15a Task 0, tools/oracle/ornith_golden.sh) and pushed with --push-data."
rownote 19 "Speed: the int4 a||b is a PROVISIONAL S1 L1 cell (8 sub-groups); r10.p0's profile shows its share - an S > 1 form needs gdn_step to sum slices (a P0 arm, not built)."
stage r19.host 19 default cpu - - "host: ab_int4_test (the int4 a||b repack exact with its zero padding, the prefill copy, the compressed-tensors form, the bf16 row untouched), model_desc_test, variant_names_test, ornith_mtp_names_test (both a||b forms bound), memory_plan_test, ornith_mtp_head_test"
st_r19_host() {
  run_tests '^(ab_int4_test|model_desc_test|variant_names_test|ornith_mtp_names_test|memory_plan_test|ornith_mtp_head_test)$'
  finish
}
stage r19.r0 19 default cpu - g0.sha,g0.bitwise,g0.suite "R0: no .cl changed and every pre-existing binary identical (G0); the new int4 a||b binaries gemv_M{1..4}_K2048_N128_S1_L1 built, and Qwen3.8's argmax_stage1_M{1..4} Ornith now binds"
st_r19_r0() {
  need_pass g0.sha g0.bitwise g0.suite
  kbins gemv_M1_K2048_N128_S1_L1 gemv_M2_K2048_N128_S1_L1 gemv_M3_K2048_N128_S1_L1 gemv_M4_K2048_N128_S1_L1 \
    argmax_stage1_M1 argmax_stage1_M2 argmax_stage1_M3 argmax_stage1_M4
  finish
}
stage r19.load 19 default gpu ornith r19.r0 "the load on the card: in_proj_a/b classified int4 (the a||b line: 64 real columns, 15.7 MB prefill copy), 0 unconsumed, read/token and the W check against doc_w 2.345 GB (predicted delta 0.000%), a prefill + decode through both a||b paths"
st_r19_load() {
  chk "$(bench_cmd "$SNAP_ORNITH" --prefill-length 512 --tg 16 --lm-head int8)" "b70-decode on Ornith: prefill (pf_ab_proj over the bf16 copy) and decode (gemv.cl int4 a||b)"
  grab_all loader 'a‖b|unconsumed|W check|read/token|per token' 12
  finish
}

# ======================================================================================
# Spec 18e's 11 int8-KV binaries (src/kernels/CMakeLists.txt's 18e block, tests/CMakeLists.txt
# B70_K2_KV8_KERNELS; ATTN_V2_TGT 32) and k2_moe.cl's binaries from before 18e (18b's four, 18c's
# MoVA route): 18e gave k2_moe.cl a defaulted define (MOVA_STAGE), so those must keep their bytes.
K2_KV8_BINS="k2_attn_prep_M1_N10240_S2_Q32KV8_V_KV8 k2_attn_prep_M1_N9280_S2_Q32KV8_KV8 k2_attn_M1_T32_Q32KV8_KV8
  k2_attn_eager_M1_T32_Q32KV8_KV8 k2_attn_prep_M2048_N10240_S1_Q32KV8_V_KV8 k2_attn_prep_M2048_N9472_S1_Q32KV8_KV8
  k2_pf_flash_attn_Q32KV8_KV8 k2_pf_flash_attn_Q32KV8_EAGER_KV8 k2_pf_flash_attn_Q32KV8_O_KV8
  k2_pf_flash_attn_Q32KV8_EAGER_O_KV8 k2_mova_M1_E64_T4_D2560_N1024_STAGE"
K2_MOE_PRE18E_BINS="k2_route_M1_E64_T4_N9280_O9216_S2 k2_route_M1_E100_T8_N128_O0_S1 k2_moe_M1_E100_T8_D2560_I768
  k2_mova_M1_E64_T4_D2560_N1024 k2_route_M1_E64_T4_N9472_O9216_S1"
row 21 "spec 18e Task 1 - K2-Horizon's int8 KV cache, rotkv at head_dim 128 (--kv-cache int8; spec 18 §13, plan 18e)"
rownote 21 "After rows 14 and 15: each GPU stage waits for its bf16 sibling's PASS in this state (r14.k1 / r15.k1, r14.k3, r14.golden, r14.golden_eager, r15.prefill, r15.split, r15.golden). The twins need SNAP_K2, the golden ones oracle-out-k2 as well (--with r14.oracle makes it on the box CPU); without them those stages SKIP with 'missing data'."
rownote 21 "K0's binary half is G0's g0.sha: a pre-existing binary that differs fails G0 and stops the line. k2_moe.cl's pre-18e binaries are compared only against a baseline that builds them: the default b32aaaf predates K2, so r21.k0 lists them as not compared - --baseline 6ee1d59 (main just before 18e) compares them."
rownote 21 "Review Focus 1 (tools/oracle/kv8_k2_repeat.sh: 12a's probe on K2's real weights, Mac or box CPU, ~4-8 h) is not a stage; the K2 int8 tolerances (PROPOSED in k2_kv8_kernels_test) are re-derived from it, so r21.k1 runs at the proposals. A4 with int8 KV needs 18d's K2 A4 tooling (plan 18e): not built."
rownote 21 "Passkey (r21.passkey) and speed (r21.speed) are opt-in; speed is a record with no bar - int8 against bf16 KV at depth 4096 / 32768 (both fit at max_len 40960), int8 alone at 65536 (max_len 69632: bf16 KV holds 46592)."
stage r21.k0 21 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (G0) - k2_moe.cl's pre-18e binaries named (MOVA_STAGE defaults to the old line); the K2 bf16 suite not failed (rows 14 / 15: k2_decode_test, k2_golden_test, k2_prefill_test; a SKIP for missing data is noted, not counted); the 11 int8 binaries built"
st_r21_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r14.k1 r14.k3 r14.golden r14.golden_eager r15.k1 r15.prefill r15.split r15.golden
  # shellcheck disable=SC2086 # the names, one word each, onto one line
  set -- $K2_MOE_PRE18E_BINS
  chk "bad=0; for b in $*; do f=kernels/\$b.bin; a=\$(awk -v f=\$f '\$2 == f { print \$1 }' $STATE/g0-sha/baseline.sha 2>/dev/null); u=\$(awk -v f=\$f '\$2 == f { print \$1 }' $STATE/g0-sha/under-test.sha 2>/dev/null); if [ -z \"\$u\" ]; then echo \"K0 k2_moe.cl \$b: NOT BUILT (not in g0.sha's list)\"; bad=1; elif [ -z \"\$a\" ]; then echo \"K0 k2_moe.cl \$b: not in the baseline build (the baseline predates K2) - not compared; --baseline 6ee1d59 compares it\"; elif [ \"\$a\" = \"\$u\" ]; then echo \"K0 k2_moe.cl \$b: identical to the baseline build\"; else echo \"K0 k2_moe.cl \$b: DIFFERS from the baseline build\"; bad=1; fi; done; [ \$bad = 0 ]" \
    "k2_moe.cl's pre-18e binaries against the baseline"
  kbins $K2_KV8_BINS
  grab_all k2-moe '^K0 k2_moe[.]cl' 6
  finish
}
stage r21.host 21 default cpu - - "host: kv8_test (hd128: torch's hadamard(128, 0) signs, the rotations against fp64), k2_kv8_ref_test (the flash epilogue's FWHT decomposition == unrotate bitwise, the writer, MoVA's mix quantised after the combine), k2_plan_test (the int8 auto lengths), k2_kv8_variant_names_test (every int8 binary the walks bind is built, nothing dead)"
st_r21_host() {
  run_tests '^(kv8_test|k2_kv8_ref_test|k2_plan_test|k2_kv8_variant_names_test)$'
  finish
}
stage r21.k1 21 default gpu - r14.k1,r15.k1 "K1, no checkpoint: k2_kv8_kernels_test - the writers (decode, prefill, dense and MoVA-staged) bitwise k2_kv8_ref, the staged MoVA row bitwise the bf16 build's V row, decode flash within 2 bf16 ulps of the fp64 int8 attention (6 / 300 / 3000 keys) + replay, eager bitwise past softplus's threshold (6 / 300 / 3000 / 4096), prefill flash default >= 0.9999 / eager >= 0.999 against fp64 at depths 1 .. 32048, gated within 1 ulp (PROPOSED bars)"
st_r21_k1() {
  run_tests '^k2_kv8_kernels_test$' 'k2 kv8'
  jgrab k1-kv8 '^k2_kv8_kernels_test$' 'bitwise|ulps|worst|bar|differ|OK|FAIL'
  finish
}
stage r21.k3 21 default gpu k2 r21.k1,r14.k3 "K3 on the int8 cache: k2_decode_kv8_test (B70_KV_CACHE=int8) - 717 launches, plan == allocation with the int8 KV term, replay bitwise incl. route rows, int8 KV rows and scales"
st_r21_k3() {
  run_tests '^k2_decode_kv8_test$' 'k2 kv8'
  jgrab k3-kv8 '^k2_decode_kv8_test$' '717|KV \(B70_KV_CACHE\)|plan|allocation|bitwise|OK|FAIL'
  finish
}
stage r21.golden 21 default gpu k2,oracle_k2 r21.k3,r14.golden "K2 on the int8 cache: k2_golden_kv8_test and k2_golden_i8head_kv8_test against oracle-out-k2 - the tie-aware gate and the routing diagnostic unchanged (B70_K2_TIE_TOL as r14.golden); the non-tie / near-tie counts recorded beside r14.golden's bf16 runs"
st_r21_golden() {
  run_tests '^k2_golden(_i8head)?_kv8_test$' 'k2 kv8' '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden-vs-bf16 '^k2_golden(_i8head)?(_kv8)?_test$' 'KV \(B70_KV_CACHE\)|tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r21.golden_eager 21 default gpu k2,oracle_k2 r21.k3,r14.golden_eager "K2 through the eager attention on the int8 cache: k2_golden_eager_kv8_test (B70_KV_CACHE=int8, B70_K2_ATTN=eager; 813 launches) - compared with r14.golden_eager's bf16 run"
st_r21_golden_eager() {
  run_tests '^k2_golden_eager_kv8_test$' 'k2 kv8' '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden-eager-vs-bf16 '^k2_golden_eager(_kv8)?_test$' 'KV \(B70_KV_CACHE\)|attention|tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r21.prefill 21 default gpu k2 r21.k1,r15.prefill "K3 on prefill over the int8 cache: k2_prefill_kv8_test (2392 / chunk + 5 launches, plan == allocation, immediate / recorded / replayed bitwise, chunks of 64 bitwise; prefill KV and routes against decode's int8 fill) and k2_prefill_eager_kv8_test (B70_K2_ATTN=eager: the EAGER flash over int8 and decode's eager kv8 attention agree)"
st_r21_prefill() {
  run_tests '^k2_prefill(_eager)?_kv8_test$' 'k2 kv8'
  jgrab k3-prefill-kv8 '^k2_prefill(_eager)?_kv8_test$' '2392|KV \(B70_KV_CACHE\)|launch|plan|allocation|bitwise|cos|median|p01|tie|agree|OK|FAIL'
  finish
}
stage r21.split 21 default gpu k2 r21.k1,r15.split "prefill_split_k2_kv8_test: splits at multiples of 64 bitwise over the int8 cache (as r15.split's bf16 run)"
st_r21_split() {
  run_tests '^prefill_split_k2_kv8_test$' 'k2 kv8'
  jgrab split-kv8 '^prefill_split_k2_kv8_test$' 'KV \(B70_KV_CACHE\)|bitwise|cos|split|PASS|FAIL'
  finish
}
stage r21.golden_prefill 21 default gpu k2,oracle_k2 r21.prefill,r15.golden "K2 on prefill over the int8 cache: k2_golden_prefill_kv8_test and _c16_kv8 (chunks of 16) against oracle-out-k2 - the tie rule and the routing diagnostic on the prompt rows, the counts beside r15.golden's bf16 runs"
st_r21_golden_prefill() {
  run_tests '^k2_golden_prefill(_c16)?_kv8_test$' 'k2 kv8' '' "${B70_K2_TIE_TOL:+B70_K2_TIE_TOL=$B70_K2_TIE_TOL}"
  jgrab k2-golden-prefill-vs-bf16 '^k2_golden_prefill(_c16)?(_kv8)?_test$' 'KV \(B70_KV_CACHE\)|tie|routing|determined|differ|OK|PASS|FAIL'
  finish
}
stage r21.cli 21 default gpu k2 r21.k1 "CLI: b70-decode <k2> --kv-cache int8 --max-len auto -> 91904 (bf16 head) / 98304 (--lm-head int8); with a prefill planned 83968 / 90368 (spec 18 §13, derived; recorded against the derived numbers)"
r21_auto() {   # r21_auto WHAT DERIVED FLAGS... - the int8-KV auto length, recorded against the derived one
  local what="$1" d="$2"; shift 2
  chk "n=\$(tools/box_validate/auto_len.sh $SNAP_K2 $* --kv-cache int8) && echo \"K2 max_len auto, int8 KV, $what: \$n (derived $d\$([ \"\$n\" = $d ] || echo ', DIFFERS'))\"" \
    "--kv-cache int8 --max-len auto, $what"
}
st_r21_cli() {
  r21_auto "bf16 head" 91904 --depth 16 --tg 1
  r21_auto "int8 head" 98304 --depth 16 --tg 1 --lm-head int8
  r21_auto "prefill planned, bf16 head" 83968 --prefill-length 256 --tg 1
  r21_auto "prefill planned, int8 head" 90368 --prefill-length 256 --tg 1 --lm-head int8
  grab_all auto '^K2 max_len auto, int8 KV' 4
  finish
}
stage r21.passkey 21 optin gpu k2,oracle_image r21.prefill "passkey near the one-card int8 ceiling: tools/probe/k2_passkey.sh int8 (--max-len auto -> 83968 with the prefill planned, N_TARGET 79000; 3/3 at placements 0.05 / 0.5 / 0.95)"
st_r21_passkey() {
  chk "MODEL=$SNAP_K2 tools/probe/k2_passkey.sh int8" "passkey near the int8 ceiling"
  grab_all passkey '^passkey k2 [^ ]+( placement [0-9.]+)?: ' 4
  finish
}
stage r21.speed 21 optin gpu k2 r21.k3 "speed, a record (no bar): decode at depth 4096 / 32768 with int8 against bf16 KV (max_len 40960), int8 alone at 65536 (max_len 69632), tg 256, bf16 head, interleaved, median of 3"
st_r21_speed() {
  idle before
  local d arms="" ratios=""
  for d in 4096 32768; do
    arms="$arms bf16-kv@$d '$(bench_cmd "$SNAP_K2" --depth $d --tg 256 --max-len 40960)'"
    arms="$arms int8-kv@$d '$(bench_cmd "$SNAP_K2" --depth $d --tg 256 --max-len 40960 --kv-cache int8)'"
    ratios="$ratios --ratio int8-kv@$d/bf16-kv@$d"
  done
  arms="$arms int8-kv@65536 '$(bench_cmd "$SNAP_K2" --depth 65536 --tg 256 --max-len 69632 --kv-cache int8)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the K2 KV-form arms"
  idle after
  grab_all memory '^memory:' 6
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 22 "spec 16b - pipeline parallel decode across two B70s (--pp 2; spec 16 §8, plan 16b)"
rownote 22 "Every GPU stage of this row needs BOTH cards: its commands set ZE_AFFINITY_MASK=0,1 themselves (the run exports DEVICE's one card); the GPU lock is the one lock for both. Both cards must be free of other DRM holders for r22.s1."
rownote 22 "16a's probe (P0: peer bandwidth both ways, the three hand-offs at 10 KB / 20 MB, the TP remote-partial arm) has not run - it is plan 16a's own probe, not a stage. 16b ships both hand-offs behind --pipeline-handoff (copy default) and the split by bytes; r22.s1 is where the two hand-offs are first compared."
rownote 22 "S1 at 32k depth is opt-in (r22.s1_32k): --pp 2 ingests through the decode lists (prefill across two cards is 16c), ~20 min per arm at depth 32768."
stage r22.host 22 default cpu - - "host: pipeline_plan_test (stages; weights by device = the measured Qwen3.8 load; the per-device plan; auto split and auto length for Qwen3.8 / Agnes / Ornith; K2's bytes per card), pp_protocol_test (the peer hand-off's protocol on two threads), pipeline_args_test"
st_r22_host() {
  run_tests '^(pipeline_plan_test|pp_protocol_test|pipeline_args_test)$'
  jgrab splits '^pipeline_plan_test$' 'split|auto|k2-horizon'
  finish
}
stage r22.r0 22 default cpu - g0.sha,g0.bitwise,g0.suite "--pp 1 is today's engine: every pre-existing binary identical, the suite bitwise (G0); the one new binary (pp_handoff) built"
st_r22_r0() {
  need_pass g0.sha g0.bitwise g0.suite
  kbins pp_handoff
  finish
}
stage r22.devices 22 default gpu - - "P4's preflight: both cards in one process and peer access 0 -> 1 (b70-decode --pp 2 prints the 'devices:' line before it fails on a missing model); the refusals before the device (cli_reject_pipeline_*, and the 2026-10-06 rename's cli_reject_pp_old_* / cli_reject_renamed_*), the one-GPU one under ZE_AFFINITY_MASK=0"
st_r22_devices() {
  chk "out=\$(ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode /nonexistent/b70-model --ids tests/cli/ok.ids --n 1 --pp 2 2>&1); echo \"\$out\"; echo \"\$out\" | grep -q '^devices: 0 '" \
    "two devices and peer access (the devices line)"
  grab devices '^devices:|needs two GPUs|cannot access device 1'
  run_tests '^cli_reject_(pipeline_|pp_old_|renamed_)' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r22.p1 22 default gpu qwen r22.devices "P1 (Review Focus 1-3, 5): pp_decode_test and _i8head - ingest + 64 ids and a spec 7 restore at 4395 under copy and peer, the auto split and the cuts at 5 and 62: ids, logits, GDN state, conv ring, KV and both Control blocks bitwise one card on device 0; a reset and the run again, bitwise"
st_r22_p1() {
  run_tests '^(pp_decode_test|pp_decode_i8head_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab p1 '^pp_decode' 'bitwise|memory, device|OK|differ'
  finish
}
stage r22.p1_kv8 22 default gpu qwen r22.p1 "P1 over the int8 KV cache: pp_decode_kv8_test (B70_KV_CACHE=int8)"
st_r22_p1_kv8() {
  run_tests '^pp_decode_kv8_test$' kv8 '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r22.p4 22 default gpu qwen r22.devices "P4 (Review Focus 4): pp_fail_test - a lost hand-off under copy (the fence bound; the event host-signalled so device 1 drains) and under peer (pp_recv's spin bound) throws within its bound, never hangs; reset() recovers"
st_r22_p4() {
  run_tests '^pp_fail_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab p4 '^pp_fail_test$' 'threw after|recovered|OK'
  finish
}
stage r22.cli 22 default gpu qwen r22.p1 "b70-decode --pp 2 on Qwen3.8: --ids prose.ids --n 64 under copy and peer equal the one-card ids; --max-len auto plans both cards (262144 with the bf16 head, derived) and runs; the plan and both memory lines"
st_r22_cli() {
  chk "build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 64 > $STATE/r22-one.ids" "one card"
  local h
  for h in copy peer; do
    chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 64 --pp 2 --pipeline-handoff $h > $STATE/r22-$h.ids && cmp $STATE/r22-one.ids $STATE/r22-$h.ids" \
      "--pp 2 --pipeline-handoff $h: the one-card ids"
  done
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 16 --pp 2 --max-len auto > /dev/null" \
    "--max-len auto across two cards"
  grab_all auto 'max_len: auto ->|split: auto ->'
  grab_all plan '^  device [01]:' 8
  grab_all memory '^memory, device' 8
  finish
}
stage r22.agnes 22 default gpu agnes r22.p1 "Review Focus 5 - the split from the descriptor: Agnes (72 layers) under --pp 2 gives the one-card ids (--ids prose.ids --n 32, --lm-head int8)"
st_r22_agnes() {
  chk "build/src/cli/b70-decode $SNAP_AGNES --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 > $STATE/r22-agnes-one.ids" "Agnes, one card"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_AGNES --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --pp 2 > $STATE/r22-agnes-pp.ids && cmp $STATE/r22-agnes-one.ids $STATE/r22-agnes-pp.ids" \
    "Agnes, --pp 2: the one-card ids"
  grab_all split 'split: auto ->|^engine: --pp 2'
  finish
}
stage r22.ornith 22 default gpu ornith r22.p1 "Review Focus 5 on a MoE model: Ornith (40 layers, the cut's fold SP0) under --pp 2 gives the one-card ids (--ids prose.ids --n 32, --lm-head int8)"
st_r22_ornith() {
  chk "build/src/cli/b70-decode $SNAP_ORNITH --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 > $STATE/r22-ornith-one.ids" "Ornith, one card"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_ORNITH --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --pp 2 > $STATE/r22-ornith-pp.ids && cmp $STATE/r22-ornith-one.ids $STATE/r22-ornith-pp.ids" \
    "Ornith, --pp 2: the one-card ids"
  grab_all split 'split: auto ->|^engine: --pp 2'
  finish
}
stage r22.s1 22 default gpu qwen r22.p1 "S1: decode at depth 4096, tg 256 - one card against --pp 2 copy and peer, interleaved after a warm-up, median of 3; the bar is within 2 % (ratio >= 0.98)"
st_r22_s1() {
  idle before
  local arms
  arms=" one@4096 '$(bench_cmd "$SNAP_QWEN" --depth 4096 --tg 256)'"
  arms="$arms copy@4096 'ZE_AFFINITY_MASK=0,1 $(bench_cmd "$SNAP_QWEN" --depth 4096 --tg 256 --pp 2)'"
  arms="$arms peer@4096 'ZE_AFFINITY_MASK=0,1 $(bench_cmd "$SNAP_QWEN" --depth 4096 --tg 256 --pp 2 --pipeline-handoff peer)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio copy@4096/one@4096 --ratio peer@4096/one@4096 --$arms" \
    "the S1 arms at 4096"
  idle after
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r22.s1_32k 22 optin gpu qwen r22.p1 "S1 at depth 32768 (max_len 40960): the same three arms (~20 min per --pp 2 arm: it ingests through the decode lists)"
st_r22_s1_32k() {
  idle before
  local arms
  arms=" one@32768 '$(bench_cmd "$SNAP_QWEN" --depth 32768 --tg 256 --max-len 40960)'"
  arms="$arms copy@32768 'ZE_AFFINITY_MASK=0,1 $(bench_cmd "$SNAP_QWEN" --depth 32768 --tg 256 --max-len 40960 --pp 2)'"
  arms="$arms peer@32768 'ZE_AFFINITY_MASK=0,1 $(bench_cmd "$SNAP_QWEN" --depth 32768 --tg 256 --max-len 40960 --pp 2 --pipeline-handoff peer)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio copy@32768/one@32768 --ratio peer@32768/one@32768 --$arms" \
    "the S1 arms at 32768"
  idle after
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 23 "spec 16c - pipeline parallel PREFILL across two B70s (--pp 2 --prefill / --prefill-length; spec 16 §9, plan 16c)"
rownote 23 "After row 22: 16c's prefill hands off at 16b's cut (layer s's fold on device 0) through 16b's two hand-offs, so r22.devices and r22.p1 must pass first. Every GPU stage needs BOTH cards (ZE_AFFINITY_MASK=0,1 in its commands, the one GPU lock); both cards free of other DRM holders for the speed stages."
rownote 23 "S2 (spec 16 §5): pp32768 and pp65536 >= 1.7x one card, pp4096 >= 1.2x (estimates). Each --pp 2 arm prints each card's busy time ('pp: device N busy ...'): the pipeline runs at the slower card's rate, so the busy line says whether the byte-balanced auto split also balances time (the second card is ~3 % slower on prefill, docs/10); r23.split_sweep (opt-in) times the neighbouring splits."
rownote 23 "pp131072 is opt-in (r23.s2_128k): one card at --max-len 131328 (the most it holds with the bf16 head at that length), --pp 2 at --max-len 262144 (the P3 / S3 configuration only two cards hold); ~2.5 min per one-card run."
stage r23.host 23 default cpu - - "host: pipeline_prefill_plan_test (the chunks = Engine::prefill's, the host's order and every rule it keeps - each single-step drop of it caught -, the landing slots, the per-device plan with a prefill path), pp_prefill_protocol_test (that order run by its one executor over two host threads: copy and peer, a slowed device 1, spec 7 hooks from the shadows, a throwing hook, a lost hand-off; TSan-clean on the Mac), pipeline_args_test (the lifted refusal; sycl-tla / composed refused)"
st_r23_host() {
  run_tests '^(pipeline_prefill_plan_test|pp_prefill_protocol_test|pipeline_args_test)$'
  jgrab plan '^pipeline_prefill_plan_test$' 'chunks:|schedule:|landing:|split'
  jgrab protocol '^pp_prefill_protocol_test$' 'chunks:|threw|hook|OK'
  finish
}
stage r23.r0 23 default cpu - g0.sha,g0.bitwise,g0.suite "--pp 1 is today's engine: 16c adds no kernel binary (pp_send / pp_recv are 16b's), every pre-existing binary identical and the suite bitwise (G0) - Engine::prefill's chunk rule now comes from runtime/prefill_chunks.h and step_chunk walks through walk_layers, both meant to be the same launches in the same order"
st_r23_r0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r23.reject 23 default gpu - r22.devices "the refusals before the device: --pp 2 with a sycl-tla prefill (--prefill and --prefill-length), cli_reject_pipeline_*_sycl"
st_r23_reject() {
  run_tests '^cli_reject_pipeline_(prefill|bench_prefill)_sycl$' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r23.p1 23 default gpu qwen r22.p1 "P1 / P2 (Review Focus 1-4): pp_prefill_test, _l0 and _i8head - 4103 ids (3 chunks), 32768 (16), 2 ids, a prefill_split continuation (2500 | 1603, chunk 1000), spec 7's hooks at 2048 / 4096 (from the shadows, the next chunk running) and 5000, a hook throwing at 2048, a one-card snapshot at 3000 continued on two cards - logits row, every layer's GDN state / conv ring / KV, both Control blocks and 16 decoded ids bitwise one card on device 0; copy and peer at the auto split, and the cuts at 5 and 62; each card's launches = step_stage_launches' arithmetic"
st_r23_p1() {
  run_tests '^(pp_prefill_test|pp_prefill_l0_test|pp_prefill_i8head_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab p1 '^pp_prefill' 'bitwise|hooks|throws|busy|differs|memory, device|OK'
  finish
}
stage r23.p1_kv8 23 default gpu qwen r23.p1 "P1 over the int8 KV cache: pp_prefill_kv8_test (B70_KV_CACHE=int8; the flash path's kv8 twins per device)"
st_r23_p1_kv8() {
  run_tests '^pp_prefill_kv8_test$' kv8 '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r23.p4 23 default gpu qwen r22.devices "Back-pressure and P4 (Review Focus 2): pp_prefill_fail_test - device 1 held back 150 ms a chunk (a host thread releases its event): bitwise the clean run, the wall >= the holds; a lost hand-off (no ready signal) throws within the 5 s bound naming the chunk, device 1 is released and drains, the next prefill says reset() first, after reset() bitwise again; copy and peer"
st_r23_p4() {
  run_tests '^pp_prefill_fail_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab p4 '^pp_prefill_fail_test$' 'held|threw after|recovered|bitwise|OK'
  finish
}
stage r23.cli 23 default gpu qwen r23.p1 "b70-decode --pp 2 --prefill on Qwen3.8: --ids prose.ids --n 64 under copy and peer, l0-int8 and l0, and --prefill-chunk 1000, equal the one-card --prefill ids; --max-len auto with --prefill plans each card's prefill scratch; the pp lines (each card's busy time) and both memory lines"
st_r23_cli() {
  local b h
  for b in l0-int8 l0; do
    chk "build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 64 --prefill --prefill-backend $b > $STATE/r23-one-$b.ids" "one card, --prefill, $b"
    for h in copy peer; do
      chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 64 --prefill --prefill-backend $b --pp 2 --pipeline-handoff $h > $STATE/r23-$h-$b.ids && cmp $STATE/r23-one-$b.ids $STATE/r23-$h-$b.ids" \
        "--pp 2 --prefill $b, $h: the one-card ids"
    done
  done
  chk "build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 32 --prefill --prefill-chunk 1000 > $STATE/r23-one-c1000.ids" "one card, --prefill-chunk 1000"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 32 --prefill --prefill-chunk 1000 --pp 2 > $STATE/r23-pp-c1000.ids && cmp $STATE/r23-one-c1000.ids $STATE/r23-pp-c1000.ids" \
    "--pp 2 --prefill-chunk 1000: the one-card ids"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 16 --prefill --pp 2 --max-len auto > /dev/null" \
    "--max-len auto with --prefill across two cards"
  grab_all auto 'max_len: auto ->|split: auto ->|pipeline plan at'
  grab_all pp '^pp: ' 12
  grab_all memory '^memory, device' 8
  finish
}
stage r23.agnes 23 default gpu agnes r23.p1 "Review Focus 1 on Agnes (72 layers): --ids prose.ids --n 32 --lm-head int8 --prefill under --pp 2 gives the one-card --prefill ids"
st_r23_agnes() {
  chk "build/src/cli/b70-decode $SNAP_AGNES --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --prefill > $STATE/r23-agnes-one.ids" "Agnes, one card, --prefill"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_AGNES --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --prefill --pp 2 > $STATE/r23-agnes-pp.ids && cmp $STATE/r23-agnes-one.ids $STATE/r23-agnes-pp.ids" \
    "Agnes, --pp 2 --prefill: the one-card ids"
  grab_all pp '^pp: |split: auto ->' 6
  finish
}
stage r23.ornith 23 default gpu ornith r23.p1 "Review Focus 1 on a MoE model: Ornith (40 layers; the cut's fold SP0, the grouped experts per device) --ids prose.ids --n 32 --lm-head int8 --prefill under --pp 2 (l0-int8 and l0) gives the one-card --prefill ids"
st_r23_ornith() {
  local b
  for b in l0-int8 l0; do
    chk "build/src/cli/b70-decode $SNAP_ORNITH --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --prefill --prefill-backend $b > $STATE/r23-ornith-one-$b.ids" "Ornith, one card, --prefill $b"
    chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_ORNITH --ids tests/golden/prompts/prose.ids --n 32 --lm-head int8 --prefill --prefill-backend $b --pp 2 > $STATE/r23-ornith-pp-$b.ids && cmp $STATE/r23-ornith-one-$b.ids $STATE/r23-ornith-pp-$b.ids" \
      "Ornith, --pp 2 --prefill $b: the one-card ids"
  done
  grab_all pp '^pp: |split: auto ->' 6
  finish
}
# S2: one prefill length per stage - one card against --pp 2 copy and peer, interleaved after a
# warm-up, median of 3 (tools/box_validate/interleave.sh), the pp t/s ratio per arm.
s2_arms() {   # s2_arms LEN MAXLEN_ONE MAXLEN_PP
  local n="$1" one="$2" pp="$3"
  printf " one@%s '%s'" "$n" "$(bench_cmd "$SNAP_QWEN" --prefill-length "$n" --tg 16 --max-len "$one")"
  printf " copy@%s 'ZE_AFFINITY_MASK=0,1 %s'" "$n" "$(bench_cmd "$SNAP_QWEN" --prefill-length "$n" --tg 16 --max-len "$pp" --pp 2)"
  printf " peer@%s 'ZE_AFFINITY_MASK=0,1 %s'" "$n" "$(bench_cmd "$SNAP_QWEN" --prefill-length "$n" --tg 16 --max-len "$pp" --pp 2 --pipeline-handoff peer)"
}
s2_stage() {   # s2_stage LEN MAXLEN_ONE MAXLEN_PP
  idle before
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio copy@$1/one@$1 --ratio peer@$1/one@$1 --$(s2_arms "$1" "$2" "$3")" \
    "the S2 arms at pp$1"
  idle after
  grab_all ratio '^RATIO '
  grab_all busy '^pp: device' 12
  grab_all idle '^IDLE '
  finish
}
stage r23.s2_4k 23 default gpu qwen r23.p1 "S2 at pp4096 (two chunks: little overlap; bar >= 1.2x): one card vs --pp 2 copy and peer, --tg 16, max_len 16384, interleaved, median of 3; each card's busy time"
st_r23_s2_4k() { s2_stage 4096 16384 16384; }
stage r23.s2_32k 23 default gpu qwen r23.p1 "S2 at pp32768 (16 chunks; bar >= 1.7x): one card vs --pp 2 copy and peer, max_len 40960, interleaved, median of 3; each card's busy time"
st_r23_s2_32k() { s2_stage 32768 40960 40960; }
stage r23.s2_64k 23 default gpu qwen r23.p1 "S2 at pp65536 (32 chunks; bar >= 1.7x): one card vs --pp 2 copy and peer, max_len 69632, interleaved, median of 3; each card's busy time"
st_r23_s2_64k() { s2_stage 65536 69632 69632; }
stage r23.s2_128k 23 optin gpu qwen r23.p1 "S2 at pp131072 (64 chunks): one card at max_len 131328 vs --pp 2 copy and peer at max_len 262144 (the length only two cards hold, spec 16 P3 / S3), interleaved, median of 3"
st_r23_s2_128k() { s2_stage 131072 131328 262144; }
stage r23.split_sweep 23 optin gpu qwen r23.p1 "Review Focus 5: pp32768 --pp 2 copy at the splits 30 / 32 / 34 (and auto), interleaved, median of 3 - each card's busy time per split, to see where the second card's ~3 % slower prefill puts the time-balanced cut"
st_r23_split_sweep() {
  idle before
  local arms="" s
  for s in 30 32 34; do
    arms="$arms s$s 'ZE_AFFINITY_MASK=0,1 $(bench_cmd "$SNAP_QWEN" --prefill-length 32768 --tg 16 --max-len 40960 --pp 2 --pipeline-split $s)'"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio s30/s32 --ratio s34/s32 --$arms" \
    "pp32768 at three splits"
  idle after
  grab_all ratio '^RATIO '
  grab_all busy '^pp: device' 18
  finish
}

# ======================================================================================
row 24 "spec 20c - Kolibri-1 decode, one card then two (spec 20 §11, plan 20c)"
rownote 24 "Written blind: no Kolibri binary was ever compiled by ocloc, nothing ran on a card. The synthetic checkpoints and their golden sets (oracle-out-kolibri-synth: two 5-layer real-width checkpoints, int4 and bf16 attention, ~6.2 GB each) are made on the box CPU by the opt-in r24.oracle_synth (--with r24.oracle_synth: make_synth.py, then kolibri_ref.py run on prose and de_prose, ~1 h; needs Aleph-Alpha/Kolibri-1-BF16's tokenizer.json in the HF cache - a few MB, not the weights). Without them r24.load / k3 / golden / pp / cli SKIP with 'missing data'."
rownote 24 "Everything on the REAL checkpoint (r24.oracle_real, r24.partial, r24.pp_real, r24.speed) needs spec 20b's urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ (SNAP_KOLIBRI) and oracle-out-kolibri (--with r24.oracle_real, ~48 GB RAM): SKIP 'missing data' until spec 20 decision 1 is made and 20b has run."
rownote 24 "Set B70_KOL_TIE_TOL from the gap distribution r24.oracle_synth / r24.oracle_real print (kolibri_golden_test.cc proposes 1e-2); the partial-forward bars (median cosine 0.9998, min 0.99) are proposals too - r24.partial prints them per layer."
rownote 24 "tests/golden/prompts/kolibri_bench.ids (and cli/kolibri_decode.h's kKolibriBenchPrompt) are the first 42 ids of de_prose through Kolibri's tokenizer since spec 20e (20c's placeholders re-baked on the Mac from the BF16 release's tokenizer.json): r24.oracle_synth writes the same 42 (kolibri_bench.ids in oracle-out-kolibri-synth) and says NOTE if they differ."
rownote 24 "If determined rows fail under flash, the first suspect is flash's fp32 probabilities against the reference's eager bf16 chain: r24.golden_eager runs the eager variant (B70_KOLIBRI_ATTN=eager, 856 launches at 50 layers); spec 18 §10.1's rule decides the default. Two cards (r24.pp) need row 22's hand-offs proven first."
stage r24.k0 24 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (no existing .cl edited; kernel_cmdlines +23 / -0 / ~0 with K2 on), the Qwen3.8 suite (G0); the two descriptor-free overloads in 16b's files leave pipeline_plan_test / pp_decode_test unchanged"
st_r24_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r24.host 24 default cpu - - "host: kolibri1_test, kolibri1_repack_test, kolibri1_rope_test (bitwise torch's cos / sin), kolibri_ref_test (the kernels' twin bitwise against kolibri_ref.py's fixture), kolibri_variant_names_test, kolibri_plan_test (756 / 856 launches, the split 25 by pp_balance), pipeline_plan_test (the overload == the ModelDesc form)"
st_r24_host() {
  run_tests '^(kolibri1_test|kolibri1_repack_test|kolibri1_rope_test|kolibri_ref_test|kolibri_variant_names_test|kolibri_plan_test|pipeline_plan_test)$'
  jgrab plan '^kolibri_plan_test$' 'split 25|plan at|OK'
  finish
}
stage r24.k1 24 default gpu - - "K1, no checkpoint: the 23 new binaries built (kbins) and kolibri_kernels_test - the norm and the sandwich (prep_res_fold _Z + kol_post_add) bitwise, kol_attn_prep bitwise (ring row 1 at 4097; NoPE position-free), the route (ids exact, a tie at the cut, all -1e30, padded slots +80), the MoE block within 2 ulps (6 routed + the bf16 shared slot, each alone), eager attention bitwise (5 / 512 / 513 / 4103 / 9000 sliding, 30000 full), flash within cosine 0.99999, flash at 9000 == 808 (the ring is addressing)"
st_r24_k1() {
  kbins kol_embed_gather_M1_D2560_V128000 kol_argmax_stage1_M1_N128000_V128000 kol_argmax_stage2_N128000 \
    prep_res_fold_M1_K2560_SP1_G20_Z prep_res_fold_M1_K2560_SP4_G20_Z kol_norm_M1_K2560_G20_W20 kol_post_add_M1_K2560_G20 \
    kol_attn_prep_M1_N7168_S2_Q48KV4_R4096 kol_attn_prep_M1_N7168_S2_Q48KV4_F kol_attn_prep_M1_N7168_S1_Q48KV4_R4096 \
    kol_attn_prep_M1_N7168_S1_Q48KV4_F kol_attn_M1_T32_Q48KV4_W513_R4096 kol_attn_M1_T32_Q48KV4_F \
    kol_attn_eager_M1_T32_Q48KV4_W513_R4096 kol_attn_eager_M1_T32_Q48KV4_F kol_route_M1_E384_T6_N512_L256 \
    kol_moe_M1_E384_T6_D2560_I512_SH gemv_M1_K2560_N7168_S2_L0 gemv_bf16_M1_K2560_N7168 gemv_bf16_M1_K6144_N2560 \
    gemv_bf16_M1_K2560_N512_C16_S16 gemv_bf16_M1_K2560_N128000 gemv_i8w_M1_K2560_N128000
  run_tests '^kolibri_kernels_test$' kolibri
  jgrab k1 '^kolibri_kernels_test$' 'bitwise|ulp|cosine|differ|OK'
  finish
}
stage r24.reject 24 default gpu - - "the refusals before the device: cli_reject_kolibri_* (--pp 1 on the real model naming --pp 2, the prefill backends and chunk (spec 20d, row 26), --mtp, --kv-cache int8, --device with --pp 2, --layers on a Qwen checkpoint, b70-serve naming spec 20e)"
st_r24_reject() {
  run_tests '^cli_reject_kolibri_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r24.oracle_synth 24 optin cpu oracle_image - "the synthetic golden sets on the box CPU: kolibri_oracle.sh synth - make_synth.py --layers 5 (int4 and bf16 attention) + check.py, kolibri_ref.py run on prose and de_prose -> \$DATA/oracle-out-kolibri-synth (resumable; MemAvailable >= 24 GB); the gap distribution and kolibri_bench.ids; then re-links oracle-out*"
st_r24_oracle_synth() {
  chk "tools/box_validate/kolibri_oracle.sh $DATA synth" "the Kolibri-1 synthetic reference runs"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_(kolibri|oracle_kolibri)' $STATE/have.env"
  grab_all gap 'selection gap' 8
  grab_all bench 'kolibri_bench.ids|re-bake' 4
  finish
}
stage r24.load 24 default gpu oracle_kolibri_synth r24.k1 "the loader on the card: kolibri1_load_synth_int4attn_test and _bf16attn_test (int8 head) - every part's bytes = kol_device_weight_bytes, 0 unconsumed, the last expert's block / the router's padded rows / the bias tail read back"
st_r24_load() {
  run_tests '^kolibri1_load_synth_(int4attn|bf16attn)_test$'
  jgrab load '^kolibri1_load_synth' 'device 0|read/token|unconsumed|OK'
  finish
}
stage r24.k3 24 default gpu oracle_kolibri_synth r24.load "K3 on one card: kolibri_decode_synth_test - plan == allocation, 2 + 5 x 15 + 4 = 81 launches, two runs bitwise (logits, routes, full KV, rings), the ring at 600 holds 88..599"
st_r24_k3() {
  run_tests '^kolibri_decode_synth_test$'
  jgrab k3 '^kolibri_decode_synth_test$' 'plan ==|K3|ring|OK'
  finish
}
stage r24.golden 24 default gpu oracle_kolibri_synth r24.k3 "KL2 on the synthetic checkpoints: kolibri_golden_synth_int4attn_test, _bf16attn_test, _i8head_test - the tie-aware token gate over prose / de_prose x 32 and the routing diagnostic (a non-tie set difference fails; B70_KOL_TIE_TOL 1e-2 proposed)"
st_r24_golden() {
  run_tests '^kolibri_golden_synth_(int4attn|bf16attn|i8head)_test$' kolibri '' "${B70_KOL_TIE_TOL:+B70_KOL_TIE_TOL=$B70_KOL_TIE_TOL}"
  jgrab golden '^kolibri_golden_synth' 'gate:|routing:|tap|OK'
  finish
}
stage r24.golden_eager 24 default gpu oracle_kolibri_synth r24.k3 "KL2 through the EAGER attention (B70_KOLIBRI_ATTN=eager, 2 + 5 x 17 + 4 = 91 launches): kolibri_golden_synth_int4attn_eager_test and _bf16attn_eager_test - compare with r24.golden"
st_r24_golden_eager() {
  run_tests '^kolibri_golden_synth_(int4attn|bf16attn)_eager_test$' kolibri '' "${B70_KOL_TIE_TOL:+B70_KOL_TIE_TOL=$B70_KOL_TIE_TOL}"
  jgrab golden '^kolibri_golden_synth' 'gate:|routing:|OK'
  finish
}
stage r24.pp 24 default gpu oracle_kolibri_synth r24.k3,r22.p1 "two cards (Review Focus 5): kolibri_pp_test - --pp 1 against --pp 2 --pipeline-split 3 on the synthetic checkpoint under copy and peer: tokens, logits, routes, full KV and rings bitwise, both Control blocks equal; P4 (a lost hand-off throws within 5 s, reset recovers); replay; the real checkpoint's split 25 vs 20 when present"
st_r24_pp() {
  run_tests '^kolibri_pp_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^kolibri_pp_test$' 'device|P4|SKIP|OK'
  finish
}
stage r24.cli 24 default gpu oracle_kolibri_synth r24.pp "CLI: b70-decode <synthetic int4attn> --pp 1 --ids prose.ids --n 32 against --pp 2 (the default; copy, peer, --pipeline-split 3): the same ids; --max-len auto plans both cards; the plan and both memory lines"
st_r24_cli() {
  local ck="oracle-out-kolibri-synth/int4attn/ckpt" ids="oracle-out-kolibri-synth/int4attn/prose.ids" h
  chk "build/src/cli/b70-decode $ck --pp 1 --ids $ids --n 32 > $STATE/r24-one.ids" "one card"
  for h in copy peer; do
    chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 32 --pipeline-handoff $h > $STATE/r24-$h.ids && cmp $STATE/r24-one.ids $STATE/r24-$h.ids" \
      "--pp 2 (the default), $h: the one-card ids"
  done
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 32 --pipeline-split 3 > $STATE/r24-s3.ids && cmp $STATE/r24-one.ids $STATE/r24-s3.ids" \
    "--pipeline-split 3: the one-card ids"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 8 --max-len auto > /dev/null" "--max-len auto on two cards"
  grab_all plan 'max_len: auto ->|split: |pipeline plan at|^engine:' 8
  grab_all memory '^memory' 6
  finish
}
stage r24.oracle_real 24 optin cpu kolibri,oracle_image - "the real golden set on the box CPU (needs spec 20b): kolibri_oracle.sh real - prose, code, de_prose, de_chat (the chat template's ids) x 32 -> \$DATA/oracle-out-kolibri (MemAvailable >= 48 GB); the gap distribution"
st_r24_oracle_real() {
  chk "tools/box_validate/kolibri_oracle.sh $DATA real" "the Kolibri-1 reference runs on the real checkpoint"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_(kolibri|oracle_kolibri)' $STATE/have.env"
  grab_all gap 'selection gap' 8
  finish
}
stage r24.partial 24 default gpu kolibri,oracle_kolibri r24.k3 "development mode on the real checkpoint, one card (needs spec 20b): kolibri1_load_partial_test (30 layers, bf16 head) and kolibri_partial_test / _eager - the residual tap of layers 0..29 against resid.L* per layer (median cosine >= 0.9998, min >= 0.99, proposed) and the routing diagnostic"
st_r24_partial() {
  run_tests '^(kolibri1_load_partial_test|kolibri_partial_test|kolibri_partial_eager_test)$'
  jgrab partial 'kolibri_partial' 'layer|BELOW|routing:|OK'
  finish
}
stage r24.pp_real 24 default gpu kolibri,oracle_kolibri r24.pp "KL2 / K3 on the real checkpoint across two cards (needs spec 20b): kolibri_golden_test (flash, copy), _i8head, _eager, _peer and kolibri_decode_test (pp2: plan == allocation per device, replay bitwise)"
st_r24_pp_real() {
  run_tests '^(kolibri_golden_test|kolibri_golden_i8head_test|kolibri_golden_eager_test|kolibri_golden_peer_test|kolibri_decode_test)$' \
    '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab golden '^kolibri_(golden|decode)' 'gate:|routing:|plan ==|K3|OK'
  finish
}
stage r24.speed 24 optin gpu kolibri r24.pp_real "Task 8 (needs spec 20b): decode on two cards at depth 4096 / 32768, tg 256, max_len 40960 - bf16 and int8 heads, copy and peer, flash and eager - interleaved, median of 3; launches per token, each card's memory; against the derived roofline (~250 t/s int4 attention + int8 head at ~600 GB/s; ~120 t/s bf16 attention)"
st_r24_speed() {
  idle before
  local d arms="" ratios="" k="ZE_AFFINITY_MASK=0,1"
  for d in 4096 32768; do
    arms="$arms d$d '$k $(bench_cmd "$SNAP_KOLIBRI" --depth $d --tg 256 --max-len 40960)'"
    arms="$arms d$d-int8 '$k $(bench_cmd "$SNAP_KOLIBRI" --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    arms="$arms d$d-int8-peer '$k $(bench_cmd "$SNAP_KOLIBRI" --depth $d --tg 256 --max-len 40960 --lm-head int8 --pipeline-handoff peer)'"
    arms="$arms d$d-int8-eager '$k B70_KOLIBRI_ATTN=eager $(bench_cmd "$SNAP_KOLIBRI" --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    ratios="$ratios --ratio d$d-int8/d$d --ratio d$d-int8-peer/d$d-int8 --ratio d$d-int8-eager/d$d-int8"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the Kolibri-1 decode arms"
  idle after
  grab_all memory '^memory, device' 8
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r24.sweeps 24 manual - kolibri - "Task 8's sweeps: the int4 attention arm's {S, layout} cells (q||k||v 2560x7168, o_proj 6144x2560: S 1 / 2 / 4, L 0 / 1) and the MoE kernels' UP_KS / DN_KS"
st_r24_sweeps() {
  say "# {S, layout} of 2560x7168 and 6144x2560 with their GEMV_* defines (probe_gemv rows, spec 14 step 4's method); model::kolibri1()'s qkv_s / oproj_s and kernels::kolibri's names move together"
  say "# UP_KS / DN_KS: rebuild kol_moe.cl's variant per value, kolibri_kernels_test, then ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_KOLIBRI --bench --depth 4096 --tg 256 --lm-head int8 per value"
  say "# record: docs/BENCHMARKS.md 'Kolibri-1 (spec 20)'"
}

# ======================================================================================
row 25 "spec 18d engine side - K2-Horizon served: K2Engine behind b70-serve, KV-only prefix snapshots, K4 (A4, passkey), the comparison rows (spec 18 §14, plan 18d)"
rownote 25 "After rows 14 / 15 (K2's decode and prefill on the card) and 21 (its int8 KV); 17 is the host half. 18d adds no kernel: G0's g0.sha is the 'nothing moved' check. Everything after r25.reject needs SNAP_K2; the A4 stages need oracle-out-k2-a4 (r25.a4_ref makes it on the box CPU: hours)."
rownote 25 "A4 has no bar (spec 18 K4): the engine against 18a's reference (k2_ref.py, bf16 mode, on the int4 checkpoint dequantised) on K2's own set (xml calls, reasoning_effort low, 512 ids) is recorded; 'reasoning' cells (the budget ran out inside the reasoning) are read, not failed."
rownote 25 "Review Focus 4 (the comparison rows) is r25.benchy, opt-in: b70-serve on ONE card at decision 2's 32k (bf16 KV) and at the int8-KV auto length, against the recorded vLLM baseline (TWO B70s, PP = 2, fp8 KV, 390,016 tokens, 44.43 t/s decode, spec 18 §7) - every row says which configuration. The record (BENCHMARKS 'K2-Horizon (spec 18)', spec 18 §10's amendment, README, docs/03, docs/13) is plan 18d Task 3, an edit on a branch after the run."
stage r25.k0 25 default cpu - g0.sha,g0.bitwise,g0.suite "K0: 18d adds no kernel (every binary identical, G0) and the Qwen-family server path is unchanged - G0's suite (prefix_gpu_*, golden_server_test, snapshot_test), r17.k0, and here prefix_cache_test / prefix_server_test (the store's zero-byte snapshots are new code there); K2's decode and prefill gates not failed (rows 14 / 15 / 21; a SKIP for missing data is noted, not counted)"
st_r25_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r17.k0 r14.k3 r15.prefill r15.split r21.k3
  run_tests '^(prefix_cache_test|prefix_server_test)$'
  finish
}
stage r25.host 25 default cpu - - "host: k2_serve_test (the snapshot runs on K2's shapes, save / load on host buffers, C2's sequences through PrefixSession bitwise a cold run's in both KV forms, the server's greedy chat = the decode run, a cached second turn, seeded sampling), k2_server_test, toolcall_k2_test"
st_r25_host() {
  run_tests '^(k2_serve_test|k2_server_test|toolcall_k2_test)$'
  jgrab host '^k2_serve_test$' 'layout|round trip|restore|continue|server|OK|FAIL'
  finish
}
stage r25.reject 25 default cpu - - "b70-serve's refusals before the device (tests/model/k2: config.json only): --mtp 1, --spec mtp, --spec lookup, --prefill-backend l0-int8 for K2 by name; --pp 2 for K2 (its own engine; spec 16d serves the Qwen family on two cards)"
st_r25_reject() {
  run_tests '^cli_reject_serve_(k2_mtp|k2_spec_mtp|k2_lookup|k2_prefill_int8|pp)$'
  finish
}
stage r25.snapshot 25 default gpu k2 r15.prefill "Review Focus 3 on the card: k2_prefix_gpu_test and _split - K2Engine's save_kv / load_kv byte-exact over 4100 positions, C1 bitwise (restored at the block end 4096 + prefill [4096, 4100) = the cold run's 32 ids and logits rows), C2's sequences a-g through PrefixSession over b70-serve's adapter under the tie-aware rule (the bitwise rows counted; c / d / f restore at block ends and should be bitwise)"
st_r25_snapshot() {
  run_tests '^k2_prefix_gpu(_split)?_test$' k2
  jgrab c1-c2 '^k2_prefix_gpu' 'byte-identical|C1|bitwise|restore|continue|near-tie|bad|OK|FAIL'
  finish
}
stage r25.snapshot_kv8 25 default gpu k2 r25.snapshot,r21.k3 "Review Focus 3 over the int8 cache: k2_prefix_gpu_kv8_test (B70_KV_CACHE=int8: the rows and their fp16 scales in the snapshots, the store keyed by K2's int8 root)"
st_r25_snapshot_kv8() {
  run_tests '^k2_prefix_gpu_kv8_test$' k2
  jgrab c1-c2-kv8 '^k2_prefix_gpu_kv8_test$' 'byte-identical|C1|bitwise|restore|near-tie|bad|OK|FAIL'
  finish
}
stage r25.serve 25 default gpu k2 r15.prefill "b70-serve <k2> startup and one request: --max-len auto -> 45824 (bf16 KV, the int8 head b70-serve defaults to, the prefill scratch planned; derived) and with --kv-cache int8 -> 90368 (derived); the startup line names K2's chat format, prefill l0, mtp none, the KV-only snapshots; a seeded sampled request twice, identical"
st_r25_serve() {
  serve "$SNAP_K2"
  x "PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_K2 --kv-cache int8"
  step_rc $? "b70-serve <k2> --kv-cache int8, seeded sampling"
  grab_all auto 'max_len: auto ->' 4
  grab_all memory '^memory:' 4
  grab_all startup '^b70-serve: .*K2-Horizon' 4
  grab_all seeded '^SEEDED ' 2
  finish
}
stage r25.chat 25 default gpu k2 r25.serve "plan 18d Task 1: a short greedy chat through the server equal to b70-decode on the same ids - golden_server_test --chat (prose / code / cjk as one user message each, 32 ids, temperature 0; b70-decode --ids <the response's prompt_token_ids> --n 32 --prefill --lm-head int8), run directly (not registered for K2: without a checkpoint it would fail, not SKIP)"
st_r25_chat() {
  chk "timeout 1800 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_K2 --chat" "golden_server_test --chat against K2-Horizon"
  grab_all chat 'chat of [0-9]+ prompt ids' 3
  finish
}
stage r25.passkey 25 default gpu k2,oracle_image r15.prefill "K4: passkey 3/3 at 5 / 50 / 95 % of the chosen context (tools/probe/k2_passkey.sh's placements) - decision 2's one-card 32k with bf16 KV (MAX_LEN 32768, 31000 ids) and the bf16-KV auto length (~42752, 40000 ids); the int8-KV ceiling is r21.passkey's (opt-in)"
st_r25_passkey() {
  chk "MODEL=$SNAP_K2 MAX_LEN=32768 N_TARGET=31000 tools/probe/k2_passkey.sh bf16" "passkey at 32768, bf16 KV"
  chk "MODEL=$SNAP_K2 N_TARGET=40000 tools/probe/k2_passkey.sh bf16" "passkey near the bf16-KV auto length"
  grab_all passkey '^passkey k2 [^ ]+( placement [0-9.]+)?: ' 8
  finish
}
stage r25.toolcall 25 default gpu k2,k2_a4_set r25.serve "K2's own A4 set (oracle-out-k2-a4/set: xml calls, reasoning_effort low) as chat requests through b70-serve with their tools and template kwargs, greedy, 512 ids: the prompt ids = the set's (the server's render = HF's, key-sorted), finish reasons (tool_calls expected), the reasoning / calls parsed"
st_r25_toolcall() {
  chk "PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE.serve k2-set 1 $SNAP_K2 --max-len 16384 -- --set oracle-out-k2-a4/set --max-tokens 512 --stop-at-eos" "the K2 set through b70-serve"
  x "grep -o '\"finish_reason\": \"[a-z_]*\"' $STATE/$STAGE.serve/requests.jsonl | sort | uniq -c; echo \"prompt ids = the set's: \$(grep -c '\"prompt_ids_match\": true' $STATE/$STAGE.serve/requests.jsonl)/36\""
  grab_all finish '^ *[0-9]+ "finish_reason"' 4
  grab_all prompt-ids "^prompt ids = the set" 1
  finish
}
stage r25.a4_ref 25 optin cpu k2,oracle_image - "K4's data on the box CPU: K2's tool-call set (make_set.py --from with K2's template: xml, reasoning_effort low, key-sorted) and 18a's reference on it (oracle_generate.py -> k2_ref.py, bf16 mode, the int4 checkpoint dequantised, 512 ids, resumable) into \$DATA/oracle-out-k2-a4 - HOURS; then re-links oracle-out*"
st_r25_a4_ref() {
  chk "OUT=$DATA/oracle-out-k2-a4 tools/toolcall/a4_ref.sh k2 set" "the K2 tool-call set"
  chk "OUT=$DATA/oracle-out-k2-a4 tools/toolcall/a4_ref.sh k2 ref" "the K2 reference run (k2_ref.py)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "OUT=$DATA/oracle-out-k2-a4 tools/toolcall/a4_ref.sh k2 status"
  grab_all ref '^k2: set' 2
  finish
}
stage r25.a4 25 optin gpu k2,oracle_k2_a4 r25.chat "K4: A4 on K2 against 18a's reference (oracle-out-k2-a4) - engine_generate.sh on K2's set, l0, 512 ids, the bf16 and the int8 head; score.py (K2's xml calls behind its reasoning) - recorded, no bar"
st_r25_a4() {
  local d=$STATE/a4-k2
  x "mkdir -p $d && cp oracle-out-k2-a4/*.bf16.txt $d/"
  chk "N_NEW=512 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_K2) oracle-out-k2-a4/set $d l0" "A4 engine runs, bf16 head"
  chk "N_NEW=512 LM_HEAD=int8 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_K2) oracle-out-k2-a4/set $d l0" "A4 engine runs, int8 head (the served default)"
  chk "python3 tools/toolcall/score.py $d bf16 l0 l0-i8head | tee $d/score.md" "scoring against the reference"
  grab_all a4 'match|/ 36' 6
  finish
}
stage r25.benchy 25 optin gpu-self k2,uvx - "Review Focus 4: llama-benchy through b70-serve <k2> on one card - pp4096 tg256 depth 1 (--no-cache --exact-tg --latency-mode generation) at 32768 (bf16 KV) and at the int8-KV auto length, then prefix caching at depth 4k / 16k / 32k cache on vs off - beside vLLM's recorded PP = 2 / fp8 KV / 390k / 44.43 t/s (spec 18 §7)"
st_r25_benchy() {
  chk "MODEL=$SNAP_K2 MAX_LEN=32768 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation" "llama-benchy pp4096 tg256, 32k bf16 KV"
  chk "MODEL=$SNAP_K2 MAX_LEN=auto SERVE_ARGS='--kv-cache int8' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation" "llama-benchy pp4096 tg256, int8 KV at auto"
  chk "MODEL=$SNAP_K2 MAX_LEN=40960 tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, prefix cache on"
  chk "MODEL=$SNAP_K2 MAX_LEN=40960 SERVE_ARGS='--prefix-cache-gb 0' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "llama-benchy, prefix cache off"
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 26 "spec 20d - Kolibri-1 prefill: grouped MoE over 384 experts, windowed flash over the ring, one hand-off per chunk (spec 20 §12, plan 20d)"
rownote 26 "Written blind on the Mac, after row 24 (20c's decode on the card) and rows 22 / 23 (16b's / 16c's hand-offs): none of the 17 new binaries of the 20d block was ever compiled by ocloc - the DPAS flash at GQA 12 with the 513-key window over the 4096-slot ring (kol_pf_flash_attn_Q48KV4_{W513_R4096,F}[_EAGER]), the 384-expert sort on 256 lanes (kol_pf_moe_E384_T6_D2560_I512_L256), the grouped GEMMs at Kolibri's shapes (pf_moe_gemm_K2560_N1024_SILU, _K512_N2560), kol_prep.cl at M 2048. The 2252-launch walk and the sequential two-card chunk hand-off never ran."
rownote 26 "The synthetic checkpoints (oracle-out-kolibri-synth, row 24's r24.oracle_synth) carry KL2 / K3 before spec 20b; everything on the real checkpoint (r26.real, r26.speed) SKIPs 'missing data' until 20b. The near-tie tolerance and the prefill-vs-decode KV bars (row >= 0.999 on layer 0, median >= 0.9998, p01 >= 0.99) are PROPOSED: the tests print the distributions."
rownote 26 "If KL2 on prefill fails determined rows under flash, compare r26.golden_eager (B70_KOLIBRI_ATTN=eager: the eager prefill flash and decode's eager attention) before reading the kernels (spec 18 §10.1's rule)."
stage r26.k0 26 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (no existing .cl edited; kernel_cmdlines +17 / -0 / ~0 with K2 on), the Qwen3.8 suite (G0); 20c's decode gates not failed (756 / 856 launches unchanged)"
st_r26_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r24.k1 r24.k3 r24.golden
  finish
}
stage r26.host 26 default cpu - - "host: kolibri_pf_ref_test (the 384-expert sort and its adversaries, == pf_moe_ref::sort; the combine == decode's chain and torch's row; the windowed flash walk == the direct softmax through the ring; eager_window == torch's eager rows bitwise; the bf16 slab tail), kolibri_pf_variant_names_test, kolibri_plan_test (2252 launches one card and two, 4 + 2 weight batches, 0.905 GB scratch, 262144 on two cards with prefill planned)"
st_r26_host() {
  run_tests '^(kolibri_pf_ref_test|kolibri_pf_variant_names_test|kolibri_plan_test)$'
  jgrab plan '^kolibri_plan_test$' 'prefill|split 25|OK'
  finish
}
stage r26.k1 26 default gpu - r26.host "K1 for the prefill, no checkpoint: the 17 binaries built (kbins) and kolibri_pf_kernels_test - slabs exact, norm / sandwich / prep at M 2048 bit-exact (the ring wrapping at 4090), the router rows bitwise decode's, sort / gather / dequant exact (block 384 the bf16 shared tiles), GROUPED == DENSE bitwise (a 1-ulp difference is a finding), reversed chunk and replay bitwise, the shared expert alone; the flash against fp64 (>= 0.99999) sliding at 0 / 1000 / 4396 / a 300-row tail / single rows and full to 62048, EAGER (0.9999 / 0.999), a split at 64 bitwise"
st_r26_k1() {
  kbins kol_pf_embed_gather_D2560_V128000 pf_res_fold_K2560_SP1_G20_Z k2_pf_dequant_slab_K2560_N7168 \
    kol_pf_bf16_slab_K2560_N7168 kol_pf_bf16_slab_K6144_N2560 kol_norm_M2048_K2560_G20_W20 kol_post_add_M2048_K2560_G20 \
    kol_attn_prep_M2048_N7168_S1_Q48KV4_R4096 kol_attn_prep_M2048_N7168_S1_Q48KV4_F kol_pf_flash_attn_Q48KV4_W513_R4096 \
    kol_pf_flash_attn_Q48KV4_W513_R4096_EAGER kol_pf_flash_attn_Q48KV4_F kol_pf_flash_attn_Q48KV4_F_EAGER \
    kol_pf_moe_E384_T6_D2560_I512_L256 pf_moe_router_K2560_N512 pf_moe_gemm_K2560_N1024_SILU pf_moe_gemm_K512_N2560
  run_tests '^kolibri_pf_kernels_test$' kolibri
  jgrab k1-prefill '^kolibri_pf_kernels_test$' 'cos|ulp|bit-exact|bitwise|grouped|dense|OK|FAIL'
  finish
}
stage r26.reject 26 default gpu - - "the refusals before the device: cli_reject_kolibri_prefill_int8 (naming the 1024-k Hadamard blocks), _sycl (no Kolibri walk), _chunk (above 2048)"
st_r26_reject() {
  run_tests '^cli_reject_kolibri_prefill_(int8|sycl|chunk)$'
  finish
}
stage r26.prefill 26 default gpu oracle_kolibri_synth r26.k1 "K3 on one card: kolibri_prefill_synth_int4attn_test, _bf16attn_test, _eager_test - 2 + 5 x 45 = 227 launches a chunk + 5, plan == allocation, immediate / recorded / replayed bitwise, chunks of 16 bitwise (4500 ids: the third chunk wraps the ring), Review Focus 4 (KV / rings / routes / tokens against decode's fill, per layer)"
st_r26_prefill() {
  run_tests '^kolibri_prefill_synth_(int4attn|bf16attn|eager)_test$' kolibri
  jgrab k3-prefill '^kolibri_prefill_synth' 'walk|K3|chunks|median|p01|routing|tokens|OK|FAIL'
  finish
}
stage r26.split 26 default gpu oracle_kolibri_synth r26.k1 "Review Focus 5: prefill_split_kolibri_test - splits at 64, 1000, 2048, 4097 against one call: the multiples of 64 bitwise, the others within the split bars (the argument predicts bitwise for 1000 too only if its 8-row groups line up: read it)"
st_r26_split() {
  run_tests '^prefill_split_kolibri_test$' kolibri
  jgrab split '^prefill_split_kolibri_test$' 'split at|bitwise|OK|FAIL'
  finish
}
stage r26.golden 26 default gpu oracle_kolibri_synth r26.prefill "KL2 on prefill, synthetic arms: kolibri_golden_prefill_synth_int4attn_test, _bf16attn_test, _c16_test (chunks of 16), _i8head_test - the tie-aware token gate and the routing diagnostic on the prompt rows (B70_KOL_TIE_TOL as r24.golden)"
st_r26_golden() {
  run_tests '^kolibri_golden_prefill_synth_(int4attn|bf16attn|c16|i8head)_test$' kolibri '' "${B70_KOL_TIE_TOL:+B70_KOL_TIE_TOL=$B70_KOL_TIE_TOL}"
  jgrab golden-prefill '^kolibri_golden_prefill_synth' 'gate:|routing:|prefill:|OK|FAIL'
  finish
}
stage r26.golden_eager 26 default gpu oracle_kolibri_synth r26.prefill "KL2 on prefill through the EAGER attention (B70_KOLIBRI_ATTN=eager, both halves): kolibri_golden_prefill_synth_eager_test - compare with r26.golden"
st_r26_golden_eager() {
  run_tests '^kolibri_golden_prefill_synth_eager_test$' kolibri '' "${B70_KOL_TIE_TOL:+B70_KOL_TIE_TOL=$B70_KOL_TIE_TOL}"
  jgrab golden-prefill-eager '^kolibri_golden_prefill_synth_eager' 'gate:|routing:|OK|FAIL'
  finish
}
stage r26.pp 26 default gpu oracle_kolibri_synth r26.prefill,r22.p1 "two cards: kolibri_prefill_pp_test (--pp 2 --pipeline-split 3 against --pp 1, BITWISE: one prefill, recorded + replayed, a 1000 + rest continuation, chunks of 16, both Control blocks; P4 - a dropped chunk hand-off throws within 5 s naming device 1, reset() recovers), prefill_split_kolibri_pp_test, kolibri_golden_prefill_synth_pp_test"
st_r26_pp() {
  run_tests '^(kolibri_prefill_pp_test|prefill_split_kolibri_pp_test|kolibri_golden_prefill_synth_pp_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^(kolibri_prefill_pp|prefill_split_kolibri_pp|kolibri_golden_prefill_synth_pp)' 'pp:|P4|device|split at|gate:|OK|FAIL'
  finish
}
stage r26.cli 26 default gpu oracle_kolibri_synth r26.pp "CLI: b70-decode <synthetic int4attn> --ids prose.ids --n 32 with and without --prefill on --pp 1 and --pp 2 (the default) - the four id lists compared (a difference judged by the tie rule); --prefill-chunk 1000; --max-len auto with --prefill on two cards; the prefill line (2 + 45 x 5 launches a chunk + 5)"
st_r26_cli() {
  local ck="oracle-out-kolibri-synth/int4attn/ckpt" ids="oracle-out-kolibri-synth/int4attn/prose.ids" a
  chk "build/src/cli/b70-decode $ck --pp 1 --ids $ids --n 32 > $STATE/r26-one.ids" "one card, decode"
  chk "build/src/cli/b70-decode $ck --pp 1 --ids $ids --n 32 --prefill > $STATE/r26-one-pf.ids" "one card, --prefill"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 32 --prefill > $STATE/r26-two-pf.ids && cmp $STATE/r26-one-pf.ids $STATE/r26-two-pf.ids" \
    "--pp 2 --prefill: the one-card --prefill ids"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 32 --prefill --prefill-chunk 1000 > $STATE/r26-c1000.ids" "--prefill-chunk 1000"
  x "for a in one one-pf two-pf c1000; do printf '%s: ' \$a; tr '\\n' ' ' < $STATE/r26-\$a.ids; echo; done; if cmp -s $STATE/r26-one.ids $STATE/r26-one-pf.ids; then echo 'PREFILL ids identical to the decode-only run'; else echo 'PREFILL ids differ from the decode-only run: judge the first difference by the tie rule'; fi"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --ids $ids --n 8 --prefill --max-len auto > /dev/null" "--max-len auto with --prefill on two cards"
  grab_all prefill '^prefill:|^PREFILL ids|max_len: auto' 8
  grab_all memory '^memory' 6
  finish
}
stage r26.real 26 default gpu kolibri,oracle_kolibri r26.pp "after spec 20b, two cards: kolibri_prefill_test (pp2: K3, Review Focus 4 on 50 layers), prefill_split_kolibri_real_test, kolibri_golden_prefill_test / _c16 / _i8head (KL2 on prefill, real golden set)"
st_r26_real() {
  run_tests '^(kolibri_prefill_test|prefill_split_kolibri_real_test|kolibri_golden_prefill(_c16|_i8head)?_test)$' '' '' \
    "ZE_AFFINITY_MASK=0,1${B70_KOL_TIE_TOL:+ B70_KOL_TIE_TOL=$B70_KOL_TIE_TOL}"
  jgrab real '^(kolibri_prefill_test|prefill_split_kolibri_real|kolibri_golden_prefill)' 'walk|K3|median|split at|gate:|routing:|OK|FAIL'
  finish
}
stage r26.speed 26 optin gpu kolibri r26.real "plan 20d Task 5 (needs spec 20b): pp4096 / pp32768 / pp131072 then tg16 on two cards (--lm-head int8, max_len 140000), flash vs B70_KOLIBRI_ATTN=eager at 32768, interleaved pairs, median of 3; against the derived ~4,000 t/s at pp4096 - BENCHMARKS 'Kolibri-1 (spec 20)' prefill rows"
st_r26_speed() {
  idle before
  local n k="ZE_AFFINITY_MASK=0,1" arms=""
  for n in 4096 32768 131072; do
    arms="$arms pp$n '$k $(bench_cmd "$SNAP_KOLIBRI" --prefill-length $n --tg 16 --max-len 140000 --lm-head int8)'"
  done
  arms="$arms pp32768-eager '$k B70_KOLIBRI_ATTN=eager $(bench_cmd "$SNAP_KOLIBRI" --prefill-length 32768 --tg 16 --max-len 140000 --lm-head int8)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio pp32768-eager/pp32768 --$arms" "the Kolibri-1 prefill arms"
  idle after
  grab_all prefill '^pp:|^prefill:' 12
  grab_all memory '^memory, device' 8
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r26.p0 26 manual - kolibri - "plan 20d Task 5's profile and levers (need a rebuild or the real checkpoint)"
st_r26_p0() {
  say "# B70_PREFILL_PROFILE=1 ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode \$SNAP_KOLIBRI --bench --prefill-length 4096 --tg 1: the moe_weights (dequant) share of a chunk, per card"
  say "# per-device busy time per chunk: the sequential hand-off leaves one card idle - spec 16c's overlapped chunk pipeline is the recorded lever (up to ~2x)"
  say "# an SLM-fused int4 grouped GEMM (spec 15d Task 1's arm) removing the per-chunk bf16 dequant pass"
  say "# record: docs/BENCHMARKS.md 'Kolibri-1 (spec 20)'"
}

# ======================================================================================
row 27 "spec 16d - pipeline parallel integration: b70-serve --pp 2, MTP across the split, the prefix cache on two cards, P3 / S3 at 262144 (spec 16 §10, plan 16d)"
rownote 27 "After rows 22 and 23 (16b's hand-offs and 16c's prefill must pass first: r22.p1, r23.p1). Every GPU stage needs BOTH cards (ZE_AFFINITY_MASK=0,1 in its commands, the one GPU lock); both free of other DRM holders for r27.benchy."
rownote 27 "16d adds no kernel binary (the stage verify / draft lists bind the one-card MTP binaries; gdn_step_slots keeps the whole model's slot stride): G0's g0.sha is the 'nothing moved' check, and G0's suite (mtp_verify_test, prefix_gpu_*, golden_server_test) is --pp 1's 'unchanged' check - capture.cc's verify walk now binds its slots through spec_mem(), the same pointer."
rownote 27 "The embedding for the drafts on device 1 is REPLICATED (2.54 GB on Qwen3.8; spec 16 §3.1's open decision, taken as 16b's placement anticipated): the memory lines show it ('MTP: head ... + embedding replica ...'); P2P row gathers are the alternative if device 1's bytes ever bind."
rownote 27 "P3 (r27.p3) needs the oracle image (passkey.py builds the prompt): 250000 ids at --max-len 262144, ~1 h per placement on two cards (derived). The llama-benchy rows (r27.benchy) are opt-in: one card vs --pp 2 at pp4096 and at depth 32k / 128k, plan 16d Task 2."
stage r27.host 27 default cpu - - "host: pp_serve_plan_test (the planner's server terms: the head + embedding replica + MtpBuffers on device 1, device 0's verify slots, the prefix cache's block shadows; the stage verify lists' launches = verify_launches at every split; spec 7's KV snapshot over two devices = the one-card layout byte for byte with the head's layer; b70-serve's derived auto lengths), pp_serve_test (b70-serve's adapter over a two-device fake: one device = two, snapshots moving between one card and two, M2 at K 1 / 2 / 3 / varying, C2's sequences through PrefixSession +- MTP, the HTTP server's greedy / MTP / lookup / split-last / seeded runs), pipeline_args_test (b70-serve's --pp refusals), pipeline_plan_test"
st_r27_host() {
  run_tests '^(pp_serve_plan_test|pp_serve_test|pipeline_args_test|pipeline_plan_test)$'
  jgrab plan '^pp_serve_plan_test$' 'extras off|verify slots|kv snapshot|split|OK'
  jgrab serve '^pp_serve_test$' 'one device|snapshots move|M2|restore|server|OK'
  finish
}
stage r27.k0 27 default cpu - g0.sha,g0.bitwise,g0.suite "K0: --pp 1 is today's engine and server - every binary identical (16d adds none), G0's suite bitwise (mtp_verify_test, prefix_gpu_*, golden_server_test, snapshot_test); rows 22 / 23's P1 not failed"
st_r27_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r22.p1 r23.p1
  finish
}
stage r27.reject 27 default cpu - - "b70-serve's --pp refusals before the device: K2-Horizon (tests/model/k2) and Kolibri-1 (tests/model/kolibri1) by name, --pp 4096, --device with --pp 2, a sycl-tla prefill, --pipeline-split alone, a bad --pipeline-handoff"
st_r27_reject() {
  run_tests '^cli_reject_serve_pp(_kolibri|_value|_device|_sycl|_split_alone|_handoff_value)?$'
  finish
}
stage r27.mtp 27 default gpu qwen r23.p1 "Review Focus 1-3: pp_mtp_test and _i8head - MTP across the split bitwise one card on device 0: the prose prefill (the head's KV filled on device 1), 64 speculative iterations (K 3, 1, 2, 0: every draft and verify logits row, the ids, the session incl. the live verify slot, the head's hidden row and KV layer), M2 on both engines (4 plain rows == one M = 4 verify), a 4103-id hooked prefill (spec 7 from the shadows), a one-card snapshot continued on two cards; copy and peer at the auto split, the cuts at 5 and 62; both Control blocks equal after every call; both memory lines (the embedding replica)"
st_r27_mtp() {
  run_tests '^(pp_mtp_test|pp_mtp_i8head_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab mtp '^pp_mtp' 'bitwise|M2|memory, device|differs|OK'
  finish
}
stage r27.mtp_kv8 27 default gpu qwen r27.mtp "Review Focus 3 over the int8 KV cache: pp_mtp_kv8_test (B70_KV_CACHE=int8: the head's KV layer in the int8 form on device 1)"
st_r27_mtp_kv8() {
  run_tests '^pp_mtp_kv8_test$' kv8 '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r27.prefix 27 default gpu qwen r23.p1 "Review Focus 2 / 5, spec 16 P2 (C2 with --pp 2): pp_prefix_gpu_test, _peer, _split (the int8 head, split_last) and _mtp (K 3) - spec 7's sequences a-f through b70-serve's adapters, two cards bitwise one card: every plan, the final turn's ids and rows, the session"
st_r27_prefix() {
  run_tests '^pp_prefix_gpu(_peer|_split|_mtp)?_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab c2 '^pp_prefix_gpu' 'final turn|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r27.prefix_kv8 27 default gpu qwen r27.prefix "C2 with --pp 2 over the int8 KV cache: pp_prefix_gpu_kv8_test (B70_KV_CACHE=int8, split_last, the int8 head)"
st_r27_prefix_kv8() {
  run_tests '^pp_prefix_gpu_kv8_test$' kv8 '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r27.cli_mtp 27 default gpu qwen r27.mtp "b70-decode --pp 2 --mtp (16b's refusal lifted): --ids prose.ids --n 64 --prefill --lm-head int8 with --mtp 3 and --mtp auto under copy and peer, equal the one-card plain ids (greedy MTP is identical, spec 8 M3); the mtp line and both memory lines"
st_r27_cli_mtp() {
  local args="--ids tests/golden/prompts/prose.ids --n 64 --prefill --lm-head int8"
  chk "build/src/cli/b70-decode $SNAP_QWEN $args > $STATE/r27-one.ids" "one card, plain"
  local h m
  for h in copy peer; do
    for m in 3 auto; do
      chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN $args --mtp $m --pp 2 --pipeline-handoff $h > $STATE/r27-$h-$m.ids && cmp $STATE/r27-one.ids $STATE/r27-$h-$m.ids" \
        "--pp 2 --mtp $m, $h: the one-card plain ids"
    done
  done
  grab_all mtp '^mtp: ' 8
  grab_all memory '^memory, device' 8
  finish
}
stage r27.serve 27 default gpu qwen r27.prefix "Review Focus 5: b70-serve --pp 2 startup and one request - --max-len auto -> 262144 with the int8 head, the prefill and the prefix cache's shadows planned (derived; one card: 201216), with --mtp auto + --kv-cache int8 (262144, derived) and with --mtp auto alone (262144 at split 36, derived); the plan and both memory lines (device 1: the head and the embedding replica); a seeded sampled request twice, identical"
st_r27_serve() {
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_QWEN --pp 2"
  step_rc $? "b70-serve --pp 2, seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_QWEN --pp 2 --mtp auto --prefix-split-last"
  step_rc $? "b70-serve --pp 2 --mtp auto --prefix-split-last, seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $SNAP_QWEN --pp 2 --mtp auto --kv-cache int8 --pipeline-handoff peer"
  step_rc $? "b70-serve --pp 2 --mtp auto --kv-cache int8, peer"
  grab_all auto 'max_len: auto ->|split: auto ->' 6
  grab_all plan '^  device [01]:' 6
  grab_all memory '^memory, device' 6
  grab_all startup '^b70-serve: .*--pp 2' 3
  grab_all seeded '^SEEDED ' 2
  finish
}
stage r27.golden 27 default gpu qwen r27.serve "Review Focus 5, one request end to end: golden_server_test with B70_SERVE_ARGS - b70-serve --pp 2 (prefix cache on, the server's default) against b70-decode on ONE card (--prefill --lm-head int8): prose / code / cjk, prompt ids and 32 generated ids identical; again with --pp 2 --mtp 3, --pp 2 --spec lookup and --pp 2 --pipeline-handoff peer"
st_r27_golden() {
  local a
  for a in "--pp 2" "--pp 2 --mtp 3" "--pp 2 --spec lookup" "--pp 2 --pipeline-handoff peer"; do
    chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_QWEN" \
      "golden_server_test, b70-serve $a"
  done
  grab_all golden 'generated ids identical' 12
  finish
}
stage r27.agnes 27 default gpu agnes r27.golden "b70-serve --pp 2 on Agnes (72 layers): golden_server_test --chat (the server's greedy chat = one-card b70-decode on its prompt_token_ids), plain and --mtp 3"
st_r27_agnes() {
  local a
  for a in "--pp 2" "--pp 2 --mtp 3"; do
    chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_AGNES --chat" \
      "Agnes, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 6
  finish
}
stage r27.ornith 27 default gpu ornith r27.golden "b70-serve --pp 2 on a MoE model: Ornith (40 layers, the MoE MTP head on device 1) golden_server_test --chat, plain and --mtp 3"
st_r27_ornith() {
  local a
  for a in "--pp 2" "--pp 2 --mtp 3"; do
    chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_ORNITH --chat" \
      "Ornith, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 6
  finish
}
stage r27.s3 27 default gpu qwen r23.p1 "S3: Qwen3.8 at max_len 262144 on two cards, recorded per device - b70-decode --pp 2 --prefill at --max-len 262144 with the checkpoint's bf16 head (17.8 GB a card + the prefill, derived) and with --mtp 3 --lm-head int8; each card's memory line and the plan"
st_r27_s3() {
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 16 --prefill --pp 2 --max-len 262144 > /dev/null" \
    "262144 on two cards, bf16 head"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $SNAP_QWEN --ids tests/golden/prompts/prose.ids --n 16 --prefill --pp 2 --max-len 262144 --mtp 3 --lm-head int8 > /dev/null" \
    "262144 on two cards, --mtp 3, int8 head"
  grab_all plan 'pipeline plan at|^  device [01]:' 12
  grab_all memory '^memory, device' 8
  finish
}
stage r27.p3 27 default gpu qwen,oracle_image r27.s3 "P3: passkey 3/3 at 5 / 50 / 95 % of ~250k on two cards - b70-decode --pp 2 --prefill at --max-len 262144 (250000 ids, l0-int8; passkey.sh DECODE_ARGS)"
st_r27_p3() {
  chk "ZE_AFFINITY_MASK=0,1 MODEL=$SNAP_QWEN MAX_LEN=262144 N_TARGET=250000 DECODE_ARGS='--pp 2' tools/probe/passkey.sh l0-int8" \
    "passkey at ~250k, --pp 2"
  grab_all passkey '^passkey [^ ]+( placement [0-9.]+)?: ' 8
  finish
}
stage r27.p3_mtp 27 optin gpu qwen,oracle_image r27.p3 "P3 with MTP: the same passkey at ~250k with --pp 2 --mtp auto --lm-head int8 (hours)"
st_r27_p3_mtp() {
  chk "ZE_AFFINITY_MASK=0,1 MODEL=$SNAP_QWEN MAX_LEN=262144 N_TARGET=250000 DECODE_ARGS='--pp 2 --mtp auto --lm-head int8' tools/probe/passkey.sh l0-int8" \
    "passkey at ~250k, --pp 2 --mtp auto"
  grab_all passkey '^passkey [^ ]+( placement [0-9.]+)?: ' 8
  finish
}
stage r27.benchy 27 optin gpu-self qwen,uvx - "plan 16d Task 2's llama-benchy rows through b70-serve: one card vs --pp 2 (the operator's flags: --no-cache --exact-tg --latency-mode generation) at pp4096 tg256, and decode at depth 32k / 128k (--pp 2 at 262144, one card at its auto length)"
st_r27_benchy() {
  local f="--concurrency 1 --no-cache --exact-tg --latency-mode generation"
  chk "MODEL=$SNAP_QWEN MAX_LEN=40960 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "one card, pp4096 tg256"
  chk "MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_AFFINITY=0,1 SERVE_ARGS='--pp 2' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "--pp 2, pp4096 tg256"
  chk "MODEL=$SNAP_QWEN MAX_LEN=auto tools/probe/serve_benchy.sh --pp 512 --tg 128 --depth 32768 131072 $f" "one card, depth 32k / 128k"
  chk "MODEL=$SNAP_QWEN MAX_LEN=262144 SERVE_AFFINITY=0,1 SERVE_ARGS='--pp 2' tools/probe/serve_benchy.sh --pp 512 --tg 128 --depth 32768 131072 $f" "--pp 2, depth 32k / 128k"
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 28 "spec 20e - Kolibri-1 served: its template and tokenizer, reasoning and hermes JSON tool calls, b70-serve on two cards with the rings in the prefix-cache snapshots, KL4 (spec 20 §13, plan 20e)"
rownote 28 "Written blind on the Mac after rows 24 / 26 (20c's decode, 20d's prefill on the card) and 27 (16d): nothing of 20e ran on a card. 20e adds NO kernel (KolibriEngine's snapshot calls are copies on the immediate lists; kernel_cmdlines +0 / -0 / ~0): G0's g0.sha is the 'nothing moved' check. Every GPU stage needs BOTH cards (ZE_AFFINITY_MASK=0,1; Kolibri serves on --pp 2 by default)."
rownote 28 "The synthetic checkpoints (oracle-out-kolibri-synth, r24.oracle_synth) carry the engine side before spec 20b: make_synth.py copies the BF16 release's tokenizer.json / tokenizer_config.json (its chat_template string - the release ships no chat_template.jinja; chat::Template reads it) / generation_config.json into each ckpt, which b70-serve needs. kolibri_tokenizer_test reads the release's tokenizer.json from the HF cache (B70_KOLIBRI_TOKENIZER_JSON) - SKIP 77 without it."
rownote 28 "After spec 20b (SNAP_KOLIBRI): r28.real, r28.passkey (KL4: 3 / 3 at 262144 with the int8 head), r28.benchy. KL4's A4 reference is the bf16 SOURCE (156 GB) run wherever 20b runs (decision 1: tools/toolcall/a4_ref.sh kolibri ref, MODEL_DIR / DEVICE there) and pushed as oracle-out-kolibri-a4; r28.a4 then scores the engine on two cards against it - the hard bar is 0 parse failures of the reference (score.py --format hermes); the first-call agreement is recorded, no bar."
rownote 28 "A request-end restore (2049, 300) is bitwise the session that never left, not ONE cold prefill of the whole prompt (20d's split rule: a split off a multiple of 64 is within the split bars, not bitwise); block-end restores (4096) are bitwise the cold run."
stage r28.k0 28 default cpu - g0.sha,g0.bitwise,g0.suite "K0: 20e adds no kernel (every binary identical, G0) and the server path is unchanged - G0's suite (golden_server_test, prefix_gpu_*) and here the host tests template_test / _agnes / _ornith / _k2, toolcall_test, toolcall_k2_test (parse_json_call moved out of toolcall_k2.cc), k2_server_test, ornith_server_test, protocol_test, prefix_server_test, mtp_server_test, lookup_server_test; Qwen's sampling defaults unchanged"
st_r28_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  run_tests '^(template_test|template_agnes_test|template_ornith_test|template_k2_test|toolcall_test|toolcall_k2_test|k2_server_test|ornith_server_test|protocol_test|prefix_server_test|mtp_server_test|lookup_server_test)$' '' 'kv8'
  finish
}
stage r28.host 28 default cpu - - "host: template_kolibri_test (16 renders byte-identical to apply_chat_template), kolibri_tokenizer_test (127998 ids, the tags, 24 texts, corpus.txt, three renders' ids), toolcall_kolibri_test (the HF renders parse back, Review Focus 1 / 2 / 5, a 2000-output fuzz split-invariant), kolibri_server_test (kwargs, list content, both EOS ids, streaming, the 1.0 / 0.97 / 128 defaults), kolibri_snapshot_test (the layouts on the real shapes, restores at 2048 / 4096 / 2049 / 5000 / 300 == cold through PrefixSession, one card <-> two, the server over HTTP), pipeline_args_test; the toolcall Python tests"
st_r28_host() {
  run_tests '^(template_kolibri_test|kolibri_tokenizer_test|toolcall_kolibri_test|kolibri_server_test|kolibri_snapshot_test|pipeline_args_test)$'
  jgrab snapshot '^kolibri_snapshot_test$' 'layout|round trip|restore at|pp:|server:|OK'
  chk "python3 tools/toolcall/test_score.py && python3 tools/toolcall/test_make_set.py && python3 tools/toolcall/test_oracle_generate.py" "the toolcall Python tests (hermes scoring, the German set, --model kolibri)"
  finish
}
stage r28.reject 28 default cpu - - "b70-serve's Kolibri refusals before the device (tests/model/kolibri1: config.json only): --mtp 1, --spec mtp, --spec lookup, --kv-cache int8, --prefill-backend l0-int8, --pp 1 (does not fit), --max-len 524288 (decision 3), --pp 2 --device 1"
st_r28_reject() {
  run_tests '^cli_reject_serve_(kolibri_mtp|kolibri_spec_mtp|kolibri_lookup|kolibri_kv8|kolibri_prefill_int8|kolibri_one_card|kolibri_max_len|pp_kolibri)$'
  finish
}
stage r28.snapshot 28 default gpu oracle_kolibri_synth r26.prefill "Review Focus 4 on the card, synthetic int4attn: kolibri_snapshot_gpu_synth_test (one card) and _synth_pp_test (--pp 2, split 3) - the sizes; a restore at the block end 4096 from the hook's own saves BITWISE the cold run (32 ids, logits, KV, the rings from 3584); restores at the request ends 2049 and 300 bitwise the session that never left; _cross_test: a snapshot taken on two cards continues on one, and the reverse"
st_r28_snapshot() {
  run_tests '^kolibri_snapshot_gpu_synth(_pp)?_test$|^kolibri_snapshot_gpu_cross_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^kolibri_snapshot_gpu' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r28.serve 28 default gpu oracle_kolibri_synth r28.snapshot "b70-serve <synthetic int4attn> startup and requests: --pp 2 by default (the startup line: Kolibri-1, kolibri1 chat format, the split, prefill l0, mtp none, lm_head int8, kv cache bf16, sampling defaults sampled T 1.00 top-p 0.97 top-k 128, the rings' last 512 positions), --max-len auto planned over both cards; --pp 1 (the synthetic fits) and --pipeline-handoff peer; a seeded sampled request twice, identical"
st_r28_serve() {
  local ck="oracle-out-kolibri-synth/int4attn/ckpt"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck"
  step_rc $? "b70-serve <synthetic>, --pp 2 (default), seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pipeline-handoff peer --max-len 16384"
  step_rc $? "b70-serve <synthetic> --pipeline-handoff peer"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck --pp 1 --max-len 16384"
  step_rc $? "b70-serve <synthetic> --pp 1, seeded sampling"
  grab_all auto 'max_len: auto ->|^split: ' 6
  grab_all memory '^memory' 6
  grab_all startup '^b70-serve: .*Kolibri-1' 3
  grab_all seeded '^SEEDED ' 4
  finish
}
stage r28.chat 28 default gpu oracle_kolibri_synth r28.serve "plan 20e Task 3 Step 4: one greedy chat through b70-serve equal to b70-decode on the same ids - golden_server_test --chat on the synthetic (prose / code / cjk as one user message, 32 ids, temperature 0; b70-decode --ids <prompt_token_ids> --n 32 --prefill --lm-head int8, both on --pp 2), with B70_SERVE_ARGS '--max-len 16384', '... --pipeline-handoff peer' and '... --pp 1' (one card vs b70-decode's two: bitwise by 20c / 20d's --pp rule); run directly (not registered: without a checkpoint it would fail, not SKIP)"
st_r28_chat() {
  local ck="oracle-out-kolibri-synth/int4attn/ckpt" a
  for a in "--max-len 16384" "--max-len 16384 --pipeline-handoff peer" "--max-len 16384 --pp 1"; do
    chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $ck --chat" \
      "Kolibri synthetic, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 9
  finish
}
stage r28.real 28 default gpu kolibri r28.chat "after spec 20b, two cards: kolibri_snapshot_gpu_test (pp2: Review Focus 4 on 50 layers); b70-serve <int4> startup (--max-len auto -> 262144 at split 25 with the int8 head and the prefill scratch, derived) with a seeded request twice; golden_server_test --chat (the server's greedy chat = b70-decode on its prompt_token_ids)"
st_r28_real() {
  run_tests '^kolibri_snapshot_gpu_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_KOLIBRI"
  step_rc $? "b70-serve <kolibri>, --pp 2, auto max_len, seeded sampling"
  chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='--max-len 32768' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $SNAP_KOLIBRI --chat" \
    "golden_server_test --chat against Kolibri-1"
  grab_all auto 'max_len: auto ->|^split: |pipeline plan at' 6
  grab_all memory '^memory' 4
  grab_all startup '^b70-serve: .*Kolibri-1' 2
  grab_all chat 'chat of [0-9]+ prompt ids' 3
  finish
}
stage r28.passkey 28 optin gpu kolibri,oracle_image r28.real "KL4: passkey 3 / 3 at 5 / 50 / 95 % of 262144 on two cards with the int8 head (tools/probe/kolibri_passkey.sh: 260000 ids, --max-len 262144, 8 greedy ids); decode speed at that depth recorded (the full layers read ~5.4 GB of KV a token there, derived)"
st_r28_passkey() {
  chk "ZE_AFFINITY_MASK=0,1 MODEL=$SNAP_KOLIBRI tools/probe/kolibri_passkey.sh int8" "passkey at 262144, two cards, int8 head"
  grab_all passkey '^passkey kolibri [^ ]+( placement [0-9.]+)?: ' 8
  finish
}
stage r28.a4 28 optin gpu kolibri,oracle_kolibri_a4 r28.real "KL4: the German A4-style set (tests/golden/toolcall-kolibri-de, 36 scenarios, thinking off) on two cards - engine_generate.sh l0 with the bf16 and the int8 head against the bf16 source's reference (oracle-out-kolibri-a4, made wherever 20b ran); score.py --format hermes: 0 parse failures of the reference (the hard bar), the first-call agreement recorded"
st_r28_a4() {
  local d=$STATE/a4-kolibri
  x "mkdir -p $d && cp oracle-out-kolibri-a4/*.bf16.txt $d/"
  chk "ZE_AFFINITY_MASK=0,1 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_KOLIBRI) tests/golden/toolcall-kolibri-de $d l0" "KL4 engine runs, bf16 head"
  chk "ZE_AFFINITY_MASK=0,1 LM_HEAD=int8 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $SNAP_KOLIBRI) tests/golden/toolcall-kolibri-de $d l0" "KL4 engine runs, int8 head (the served default)"
  chk "python3 tools/toolcall/score.py --format hermes $d bf16 l0 l0-i8head | tee $d/score.md" "scoring against the bf16 reference"
  x "grep -q '^bf16: 0 parse failure' $d/score.md"
  step_rc $? "the reference's calls all parse (KL4's hard bar)"
  grab_all a4 'match|/ 36|parse failure' 6
  finish
}
stage r28.benchy 28 optin gpu-self kolibri,uvx - "plan 20e Task 5's speed rows through b70-serve <kolibri> on two cards (SERVE_AFFINITY=0,1; b70-serve's default --pp 2 - llama-benchy's --pp is its prompt length): pp4096 tg256 depth 1 at 262144 under copy and peer (the operator's flags: --no-cache --exact-tg --latency-mode generation); prefix caching at depth 4k / 16k / 32k; decode at depth 4k / 32k / 128k - every row with the cards, the attention arm (decision 2), the int8 head and bf16 KV"
st_r28_benchy() {
  local f="--concurrency 1 --no-cache --exact-tg --latency-mode generation"
  chk "MODEL=$SNAP_KOLIBRI MAX_LEN=262144 SERVE_AFFINITY=0,1 tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "two cards (copy), pp4096 tg256"
  chk "MODEL=$SNAP_KOLIBRI MAX_LEN=262144 SERVE_AFFINITY=0,1 SERVE_ARGS='--pipeline-handoff peer' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "two cards (peer), pp4096 tg256"
  chk "MODEL=$SNAP_KOLIBRI MAX_LEN=65536 SERVE_AFFINITY=0,1 tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "prefix cache on, depth 4k / 16k / 32k"
  chk "MODEL=$SNAP_KOLIBRI MAX_LEN=262144 SERVE_AFFINITY=0,1 tools/probe/serve_benchy.sh --pp 512 --tg 128 --depth 4096 32768 131072 $f" "decode at depth 4k / 32k / 128k"
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 29 "spec 19a Task 4 - DFlash's verify cost at M = 5..8 and one block's draft cost (plan 19a Task 4; box-day plan Session 6b)"
rownote 29 "Written blind on the Mac: B70_VERIFY_M8 (default ON) adds the M = 5..8 verify variants and the drafter's GEMVs; no engine binds them (G0's g0.sha lists them as added). probe_mtp_steps' M = 5..8 lists run over a probe-owned 8-slot MtpBuffers (+1.06 GB); every existing probe_mtp_steps command line (r2.cost, r6.cost, r8.cost, r16.cost) is the 4-arm mode, unchanged. verify_m8_names_test (host) runs in x.rest."
rownote 29 "Plan 19a Task 5 (the projection and the go / no-go in spec 19) is a Mac edit from r29.cost's rows: verify(M) / plain from the 'pair M=' lines (interleaved pairs), the draft cost from probe_draft_cost's 'block' lines at the arm's precision. The M = 5..8 rows carry no bitwise check (spec 19 D3 belongs to 19c): they are costs, not outputs."
stage r29.cost 29 optin gpu qwen - "verify(M) for M = 1..8 on the int8 head at depths 4k and 32k (probe_mtp_steps max_m 8: interleaved pairs, median of 3) and one DFlash2 block's GEMVs at M = 8 in int8 / bf16 / int4 g64 plus the int8 head's 7 rows (probe_draft_cost, random weights) - idle box, device 0 (timed)"
st_r29_cost() {
  idle before
  chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 4096 32 3 int8 off 16384 8 | tee $STATE/$STAGE.4k.txt" "verify M = 1..8 at depth 4k, int8 head"
  chk "build/tools/probe/probe_mtp_steps $SNAP_QWEN 32768 32 3 int8 off 65536 8" "verify M = 1..8 at depth 32k (max_len 65536), int8 head"
  chk "pm=\$(awk '\$2 == \"verify\" && \$3 == \"M=1\" {print \$5; exit}' $STATE/$STAGE.4k.txt); build/tools/probe/probe_draft_cost --calls 20 --rounds 3 --plain-ms \${pm:-0}" \
    "one DFlash2 block's GEMVs (int8 / bf16 / int4 g64, the int8 head's 7 rows), shares of the 4k plain step"
  idle after
  grab_all cost-rows '^lm_head:|^max_len:|^verify M = |^probe_mtp_steps:|^interleaved|^probe_draft_cost:|^allocated|^one DFlash|^\| ' 160
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 30 "spec 21a - Qwen3.8-Flash-Next (qwen4_exp): the facts, the CPU reference (F1), golden sets, routing traces (plan 21a)"
rownote 30 "Box CPU only, no card: nothing here takes the GPU lock. Every stage runs tools/box_validate/qwen4exp_oracle.sh in the oracle image with transformers 5.19.0 from <tree>/oracle-out-q4exp-site (tools/oracle/qwen4exp_env.sh: pip --no-deps --target, made once; the image is never changed). r30.host needs the tiny model (hf download qikp/tiny-random-Qwen4-Exp_Qwen3.8-Flash-Next, 124 MB)."
rownote 30 "Everything on real weights (r30.ppl, r30.hfcheck, r30.golden, r30.golden_layers, r30.traces) needs Intel's checkpoint: hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound - 181.17 GB including the 128 PLE shards (102.4 GB): df -h ~ first. The bf16 original (Qwen/Qwen3.8-Flash-Next, 360 GB) is needed only for 21q's bf16-PLE KL (its 128 PLE shards), not here. Without the snapshot every real stage SKIPs 77 'missing data'."
rownote 30 "RAM: the real-weight stages SKIP 77 when MemAvailable < Q4_REF_MIN_GB (64; the 32k prompt's chunked eager attention peaks near 20 GB, ESTIMATED). The reference is layer-streamed - Intel's bf16 dense arms read from the page cache, the routed experts dequantised per forward - so times are hours (DRY_RUN=1 qwen4exp_oracle.sh <data> intel prints the ESTIMATED table). The PLE source is Intel's own bf16 shards until 21b's int8 file (<snapshot>-ple-int8/) exists; then the engine-format reference reads it (spec 21 F3) - Q4_PLE overrides."
rownote 30 "Decision 3's tau and R2's MoE tolerance are read from the gap distributions r30.golden / r30.golden_layers print ('QSA 512th/513th gap' per QSA layer, 'MoE 10th/11th gap'); the plan's 'hand back' asks for those lines and each prompt's wall / RSS line. r30.traces' files are spec 22 P0.6 / P0.8's input (tools/oracle/README.md 'The routing traces')."
stage r30.host 30 default cpu q4exp_tiny,oracle_image - "F1 on the tiny model in the oracle image: test_qwen4exp_ref.py (the streamed port = transformers 5.19.0 bitwise, bf16 / fp32, 40 / 2100 ids + cached decode, chunks; the indexer cache at 2047..2060; one test per trap) and test_qwen4exp_mtp.py (the MTP head = an independent build bitwise, both pre_fc_norm_hidden forms) - qwen4exp_oracle.sh tests"
st_r30_host() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA tests" "the qwen4_exp reference tests (tiny model)"
  grab_all tests '^test_|^  |^ok|FAIL|SKIP' 120
  finish
}
stage r30.ppl 30 optin cpu q4exp_intel,oracle_image - "F1's third bullet on Intel's checkpoint: perplexity on prose.txt, code.txt and the agentic transcript (finite, below 30 on prose - ESTIMATED ceiling) - qwen4exp_oracle.sh ppl"
st_r30_ppl() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA ppl" "the reference's perplexity on the real model"
  grab_all ppl '^ppl |^wall|not sane' 8
  finish
}
stage r30.hfcheck 30 optin cpu q4exp_intel,oracle_image - "transformers' own per-query QSA indexer against the port's cached block keys on real weights: hfcheck --layers 4, 2100 ids + 4 steps, logits and every H.L* BITWISE - qwen4exp_oracle.sh hfcheck"
st_r30_hfcheck() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA hfcheck" "hfcheck on Intel's checkpoint"
  grab_all hfcheck '^hfcheck' 6
  finish
}
stage r30.golden 30 optin cpu q4exp_intel,oracle_image - "the golden sets on Intel's checkpoint (hours, resumable): q4exp_short (QSA dense) / 4k / 8k / agentic / 32k, --gen 32 -> \$DATA/oracle-out-q4exp; the gap distributions (decision 3's tau, R2); then re-links oracle-out*"
st_r30_golden() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA intel" "the qwen4_exp golden sets on Intel's checkpoint"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all gap 'gap|wall' 120
  finish
}
stage r30.golden_layers 30 optin cpu q4exp_intel,oracle_image - "the --layers N golden sets 21c's truncated gate reads: intel-layers 4 and 18 (one card's fit with Intel's bf16 dense layers) on q4exp_short / 4k / agentic -> \$DATA/oracle-out-q4exp-L4, -L18"
st_r30_golden_layers() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA intel-layers 4" "the reference at --layers 4"
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA intel-layers 18" "the reference at --layers 18"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all gap 'gap|wall' 120
  finish
}
stage r30.traces 30 optin cpu q4exp_intel,oracle_image - "spec 22 P0.6 / P0.8's input: teacher-forced routing traces (ids, p, onorm per layer; the MTP head's step-1 routes) on the 36 A4 scenarios, q4exp_agentic, code, prose and the opencode recording when OPENCODE_LOG is set -> \$DATA/oracle-out-q4exp-traces"
st_r30_traces() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA trace" "the routing traces"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all traces '^trace ' 48
  finish
}

# ======================================================================================
row 31 "spec 21b - Qwen3.8-Flash-Next (qwen4_exp): the descriptor, the loader (both checkpoint forms), the PLE int8 table pinned per head in host USM, the memory plan (plan 21b)"
rownote 31 "No kernel and no existing file's behaviour moved (kernel_cmdlines +0 / -0 / ~0): G0's g0.sha is the 'nothing moved' check. Written on the Mac: the host stages ran there (r31.host); nothing of load_qwen4exp has touched a card or a real shard."
rownote 31 "Downloads: r31.synth needs only the ORIGINAL's small files (hf download Qwen/Qwen3.8-Flash-Next config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja - a few MB, not its 360 GB); r31.ple_convert and r31.load_real need Intel's checkpoint (hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound, 181.17 GB incl. its 102.4 GB of PLE shards - row 30's download)."
rownote 31 "Disk: r31.synth writes two ~13.5 / ~13.7 GB synthetic checkpoints (+ tiny PLE files) under \$DATA/oracle-out-q4exp-synth (via the tree's oracle-out-q4exp-synth.partial: same filesystem); r31.ple_convert writes ~51.8 GB (bf16 scales; Q4_PLE_SCALE=f32: ~52.5 GB) beside Intel's snapshot - df -h ~ first."
rownote 31 "r31.load_real pins the 51.8 GB table: load_q4_ple refuses when it exceeds MemAvailable - 16 GiB (the measured pinned cap's rule), naming both - stop other RAM users first (free -g >= 70 GB). Its report line (bytes, seconds to pin, MemAvailable before / after, page tags, rows compared) is spec 22 P0.5's starting number: hand it back."
rownote 31 "r31.ple_kl is 21q Task 3's (tools/quantize/qwen4exp/evaluate.py --ple-only, decision 7's evidence) - SKIP until that file exists."
stage r31.k0 31 default cpu - g0.sha,g0.bitwise,g0.suite "K0: 21b adds no kernel (every binary identical, G0) and leaves loader::load / load_k2 / load_kolibri1 and their files untouched"
st_r31_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r31.host 31 default cpu - - "host: qwen4exp_test (the table against config.json key by key), qwen4exp_repack_test (both forms at real widths: every expert block word for word incl. the g128 expansion, gate||up, the router's row 512, names both ways), qwen4exp_ple_test (the hash = transformers' on 64 sequences, the int8 file per head), qwen4exp_plan_test (N per card, the split by bytes, the full model refused naming spec 22)"
st_r31_host() {
  run_tests '^(qwen4exp_test|qwen4exp_repack_test|qwen4exp_ple_test|qwen4exp_plan_test)$'
  jgrab host '^qwen4exp_(test|repack_test|ple_test|plan_test)$' 'OK'
  finish
}
stage r31.plan 31 default cpu - r31.host "the planner's N for one and two cards at 32k / 128k (Intel's forms and ours, int8 head, 32.53 GB cards, 1.5 GB reserve; weights + persistent state - 21c adds the decode scratch), the split by bytes, the persistent sizes"
st_r31_plan() {
  jgrab plan '^qwen4exp_plan_test$' 'layers_that_fit|plan at|pipeline plan|device [01]:|pp_split|persistent|max_len_that_fits|OK'
  finish
}
stage r31.synth 31 optin cpu q4exp_orig_small,oracle_image - "the two synthetic real-width checkpoints on the box CPU: qwen4exp_oracle.sh synth-ckpt - make_synth.py --layers 4 --mtp (ours, Intel's form), ple_int8.py beside each, check.py --ple (ACCEPTED twice) -> \$DATA/oracle-out-q4exp-synth/{ours,intel}/{ckpt,ckpt-ple-int8}; then re-links oracle-out*"
st_r31_synth() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA synth-ckpt" "the synthetic qwen4_exp checkpoints"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_(q4exp|oracle_q4exp)' $STATE/have.env"
  grab_all synth '^ACCEPTED|^wrote|REFUSED|synthetic' 12
  finish
}
stage r31.ple_convert 31 optin cpu q4exp_intel,oracle_image - "the real PLE table as int8 rows: qwen4exp_oracle.sh ple-int8 - ple_int8.py on Intel's 128 bf16 shards (the original's rows as shipped) -> <Intel snapshot>-ple-int8/ (~51.8 GB, bf16 scales; Q4_PLE_SCALE=f32 for the other arm), check.py --ple (the I64 tensors = the formula, sampled rows re-quantised bit for bit); df -h first"
st_r31_ple_convert() {
  x "df -h \$HOME | tail -1; free -g | head -2"
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA ple-int8" "the PLE int8 file"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_q4exp_intel_ple' $STATE/have.env"
  grab_all ple '^wrote|^ACCEPTED|REFUSED|PLE int8|head 15' 12
  finish
}
stage r31.load 31 default gpu oracle_q4exp_synth r31.host "the loader on the synthetic checkpoints with the MTP head: qwen4exp_synth_host_test (the host half over make_synth's files: forms, names both ways, every layer and the head repacked, the PLE file = the descriptor) and on the card qwen4exp_load_synth_ours_test and _intel_test - 0 unconsumed, every part's bytes = q4_device_weight_bytes, the last expert's blocks / the router's row 512 / the shared expert / the head's last expert read back = the host repack, the PLE ranges' page words = the file, the pointer table"
st_r31_load() {
  run_tests '^qwen4exp_(synth_host|load_synth_(ours|intel))_test$'
  jgrab load '^qwen4exp_(synth_host|load_synth)' 'model |device 0|host PLE|read/token|load  |ours:|intel:|OK'
  finish
}
stage r31.load_real 31 default gpu q4exp_intel,q4exp_intel_ple r31.load "Intel's checkpoint at 18 layers on one card (development mode, its bf16 dense arms, g128 experts expanded): qwen4exp_load_intel_layers_test - 0 unconsumed, bytes = the plan, the edges read back; and the 51.8 GB PLE table pinned in 16 + 16 host-USM ranges: the time to pin it, MemAvailable before / after (spec 22 P0.5's starting number)"
st_r31_load_real() {
  x "free -g | head -2"
  run_tests '^qwen4exp_load_intel_layers_test$'
  x "free -g | head -2"
  jgrab load-real '^qwen4exp_load_intel_layers_test$' 'model |skipped|note|device 0|host PLE|read/token|load  |OK'
  finish
}
stage r31.ple_kl 31 optin cpu q4exp_intel,q4exp_bf16,oracle_image r31.ple_convert "21q Task 3 (decision 7's evidence): the reference through the int8 PLE file (f32 and bf16 scales) against the bf16 table - KL and top-1 per held-out set (tools/quantize/qwen4exp/evaluate.py --ple-only; SKIP until 21q writes it)"
st_r31_ple_kl() {
  if [ ! -f tools/quantize/qwen4exp/evaluate.py ]; then
    skip "tools/quantize/qwen4exp/evaluate.py (21q Task 3) is not in this tree"
    finish
    return
  fi
  chk "ORACLE_MODEL=models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound tools/oracle/run_in_container.sh 'tools/oracle/qwen4exp_env.sh /ws/oracle-out-q4exp-site > /dev/null && export PYTHONPATH=/ws/oracle-out-q4exp-site && python3 tools/quantize/qwen4exp/evaluate.py --ple-only \"\$SNAP\"'" \
    "the int8-vs-bf16 PLE KL"
  grab_all ple-kl 'KL|top-1|scale' 24
  finish
}

# ======================================================================================
row x "the rest of the suite: every registered test no stage above ran (new host tests, the routed tests' twins)"
stage x.rest x default gpu qwen,oracle_qwen - "ctest over every registered test without a result in this run (Agnes / Ornith / kv8 / k2 / longctx / kolibri / qwen4exp labels belong to their rows)"
st_x_rest() {
  run_tests '.' '' 'agnes ornith kv8 k2 longctx kolibri qwen4exp'
  finish
}
