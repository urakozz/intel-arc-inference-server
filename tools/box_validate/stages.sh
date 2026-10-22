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
row 32 "spec 21c - Qwen3.8-Flash-Next (qwen4_exp) decode: hyper-connections, QSA, the PLE gather from host USM, the 512-expert MoE, one card then two (plan 21c)"
rownote 32 "Written on the Mac: every 21c binary's first ocloc compile is here (r32.k1's kbins), and nothing of Qwen4ExpEngine has touched a card. One EXISTING source changed: prep.cl gains -DGDN_GATE_SIGMOID (prep_gated_head_M1_SIG); every other prep.cl variant's command line and preprocessed source are token for token main's (checked on the Mac for all 98) - G0's g0.sha must show prep_gated_head_M1 and every prep_* binary unchanged; if one moved, the variant moves to a copy in src/kernels/qwen4exp/ and prep.cl is restored (plan 21c F0)."
rownote 32 "Data: r32.oracle_synth needs the synthetic checkpoints (r31.synth) and writes their golden sets (~0.5-1 h each on the box CPU, ESTIMATED); r32.golden / r32.k3 need them; r32.partial needs 21a's real sets on Intel's checkpoint (row 30: qwen4exp_oracle.sh intel and intel-layers 18) and the PLE int8 file (r31.ple_convert); r32.pp and the two-card CLI need two GPUs (after r22.p1)."
rownote 32 "Tolerances PROPOSED until row 30 prints the gap distributions: B70_Q4_TIE_TOL 1e-3 (the router's 10th / 11th p) and B70_Q4_SEL_TOL 1e-3 (the 512th / 513th block score); a gap of exactly 0 is undetermined whatever the tolerance (torch's topk does not prefer the lower index: spec 21 §12 finding 2). The partial forward's bars (tap median cosine >= 0.9998, min >= 0.99) are PROPOSED too."
rownote 32 "r32.ple_rate's numbers (us per token, the implied bus rate, one card / the other / both) are spec 22 P0's first host-USM measurement: hand them back with the row (spec 22 §1 records where)."
stage r32.k0 32 default cpu - g0.sha,g0.bitwise,g0.suite "K0 (F0): every pre-existing binary identical - prep.cl's GDN_GATE_SIGMOID leaves prep_gated_head_M1 (and every prep_* binary) byte-identical; kernel_cmdlines +33 / -0 / ~0 with K2 and Kolibri on; the 21b loader untouched"
st_r32_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  x "grep -E 'prep_gated_head_M1(\.bin)?$|prep_res' $STATE/g0-sha/changed.txt 2>/dev/null && echo 'a prep.cl binary MOVED' || echo 'no prep.cl binary moved'"
  finish
}
stage r32.host 32 default cpu - - "host: qwen4exp_ref_test (the kernels' twin against qwen4exp_ref.py / transformers 5.19.0's fixture: HC, PLE ids and block, indexer, scores, selection, route, combine, the sigmoid gated norm; Review Focus 1-3), qwen4exp_variant_names_test (the bound names = the CMake block), qwen4exp_plan_test (779 / 767 / 755 launches, 75 at --layers 4, +1 two cards, +3 peer; the scratch; --layers 18 / 38 fit at 32k)"
st_r32_host() {
  run_tests '^(qwen4exp_ref_test|qwen4exp_variant_names_test|qwen4exp_plan_test)$'
  jgrab host '^qwen4exp_(ref|variant_names|plan)_test$' 'HC:|PLE:|indexer:|select:|route:|21c:|OK'
  finish
}
stage r32.k1 32 default gpu - - "F2, no checkpoint: the 33 new binaries built (kbins - their first ocloc compile) and qwen4exp_kernels_test - every q4_* kernel bitwise against qwen4exp_ref.h (HC, the PLE gather from real host-USM ranges through the pointer table and q4_ple_check, the block, the indexer prep / scores, the selection at Review Focus 2's rows incl. exact ties and n 65536, the route, the MoE with each slot alone and both shared forms, eager attention, prep_gated_head_M1_SIG); flash within cosine 0.99999 of fp64 at 1 / 2050 / 2051 / 9000 / 131072 and of attn_v2 below 2052"
st_r32_k1() {
  kbins embed_gather_M1_D2560 prep_gated_head_M1_SIG attn_prep_M1_Q24KV2 attn_prep_M1_Q24KV2_S1 attn_v2_M1_T32_Q24KV2 \
    gemv_M1_K2560_N16384_S1_L0 gemv_M1_K2560_N13312_S2_L0 gemv_bf16_M1_K2560_N16384 gemv_bf16_M1_K2560_N13312 \
    gemv_bf16_M1_K10240_N336_C16_S16 gemv_bf16_M1_K10240_N320_C16_S16 gemv_bf16_M1_K2560_N640_C16_S16 \
    gemv_bf16_M1_K2560_N528_C16_S16 gemv_bf16_M1_K2560_N12800 gemv_bf16_M1_K2560_N248320 gemv_i8w_M1_K2560_N248320 \
    q4_hc_combine_norm_M1_E q4_hc_combine_norm_M1_S1 q4_hc_combine_norm_M1_S4 q4_hc_combine_norm_M1_Y \
    q4_hc_combine_norm_M1_Y_NN q4_hc_combine_norm_M1_X q4_hc_up_mix_M1_I q4_hc_up_mix_M1 q4_ple_gather_M1_BF16 \
    q4_ple_gather_M1_F32 q4_ple_block_M1 q4_qsa_M1_T512_W1024 q4_qsa_attn_M1_T32 q4_qsa_attn_eager_M1 \
    q4_route_M1_E512_T10_N528_L256 q4_moe_M1_E512_T10_D2560_I640_SH4 q4_moe_M1_E512_T10_D2560_I640_SHB
  run_tests '^qwen4exp_kernels_test$' qwen4exp
  jgrab k1 '^qwen4exp_kernels_test$' 'exact|bitwise|ulp|cosine|differ|PASS|FAIL|OK'
  finish
}
stage r32.reject 32 default gpu - - "the refusals before the device (tests/model/qwen4exp: a model_type stub): cli_reject_qwen4exp_full (no --layers: spec 22, the bytes), _prefill (spec 21d), _mtp (spec 21e), _kv8 (decision 8), _serve (b70-serve: spec 21e); and cli_reject_kolibri_layers_qwen still names Kolibri-1"
st_r32_reject() {
  run_tests '^(cli_reject_qwen4exp_|cli_reject_kolibri_layers_qwen$)'
  finish
}
stage r32.oracle_synth 32 optin cpu oracle_q4exp_synth,oracle_image - "the synthetic golden sets on the box CPU: qwen4exp_oracle.sh synth - qwen4exp_ref.py run on both synthetic checkpoints (ours, Intel's form) for q4exp_short / 4k / agentic x 32 with their own int8 PLE files -> \$DATA/oracle-out-q4exp-synth/{ours,intel}/ (resumable; MemAvailable >= 32 GB); the gap lines (decision 3's tau, the routing tolerance); then re-links oracle-out*"
st_r32_oracle_synth() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA synth" "the qwen4_exp synthetic reference runs"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_oracle_q4exp' $STATE/have.env"
  grab_all gap 'gap' 24
  finish
}
stage r32.k3 32 default gpu oracle_q4exp_synth r31.load "F4 on decode, the synthetic checkpoints: qwen4exp_decode_synth_{ours,intel}_test over 2200 ids of q4exp_4k - plan == allocation per component, 1 + 3 x 15 + 19 + 4 + 6 = 75 launches, the PLE table read back by the device (q4_ple_check), two runs bitwise (logits, routes, selections and diagnostics, KV, compressed keys, tail rings, GDN state, conv and PLE rings, PLE ids), the injected capture (73 launches) fed the free run's lists = the free run"
st_r32_k3() {
  run_tests '^qwen4exp_decode_synth_(ours|intel)_test$'
  jgrab k3 '^qwen4exp_decode_synth' 'device [01]:|launches|F4:|injected|OK'
  finish
}
stage r32.golden 32 default gpu oracle_q4exp_synth_golden r32.k3 "F3 on the synthetic checkpoints: qwen4exp_golden_synth_{ours,intel}_test, _i8head, _eager (B70_Q4_ATTN=eager, 73 launches), _inject (the reference's selections fed in: every determined row) - the tie-aware token gate over q4exp_short / 4k / agentic x 32, the routing diagnostic, gate S (rows >= 2051), the PLE ids bitwise, the H tap"
st_r32_golden() {
  run_tests '^qwen4exp_golden_synth_(ours|intel)(_i8head|_eager|_inject)?_test$' qwen4exp '' \
    "${B70_Q4_TIE_TOL:+B70_Q4_TIE_TOL=$B70_Q4_TIE_TOL} ${B70_Q4_SEL_TOL:+B70_Q4_SEL_TOL=$B70_Q4_SEL_TOL}"
  jgrab golden '^qwen4exp_golden_synth' 'gate:|routing:|gate S:|tap|OK'
  finish
}
stage r32.partial 32 default gpu q4exp_intel,q4exp_intel_ple,oracle_q4exp,oracle_q4exp_layers r32.k3 "development mode on Intel's checkpoint, one card: qwen4exp_partial_test at --layers 18 (int8 head) - the H tap of layers 0..17 against the full reference's H.L* (median cosine >= 0.9998, min >= 0.99, PROPOSED, per layer), the routing diagnostic and gate S for layers < 18, the token gate against the truncated reference oracle-out-q4exp-L18"
st_r32_partial() {
  x "free -g | head -2"
  run_tests '^qwen4exp_partial_test$'
  jgrab partial '^qwen4exp_partial_test$' 'tokens|tap cosine|routing:|OK'
  finish
}
stage r32.pp 32 default gpu oracle_q4exp_synth r32.k3,r22.p1 "two cards (Review Focus 6): qwen4exp_pp_test - --pp 1 against --pp 2 --pipeline-split 2 and 1 (the cut at the PLE layer) on the ours synthetic under copy and peer: tokens, logits, routes, selections, every layer's persistent bytes and the PLE ids bitwise, both Control blocks equal; P4 (a lost hand-off throws within 10 s, reset recovers); Intel's --layers 38 split 19 vs 20 when present"
st_r32_pp() {
  run_tests '^qwen4exp_pp_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^qwen4exp_pp_test$' '\(a\)|\(b\)|\(c\)|\(d\)|SKIP|OK'
  finish
}
stage r32.cli 32 default gpu oracle_q4exp_synth r32.pp "CLI: b70-decode <ours synthetic> --layers 4 --ids q4exp_short.ids --n 32 on one card against --pp 2 (copy, peer): the same ids; --layers auto and --max-len auto plan; the memory lines"
st_r32_cli() {
  local ck="oracle-out-q4exp-synth/ours/ckpt" ids="tests/golden/prompts/q4exp_short.ids" h
  chk "build/src/cli/b70-decode $ck --layers 4 --ids $ids --n 32 > $STATE/r32-one.ids" "one card"
  for h in copy peer; do
    chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --layers 4 --pp 2 --ids $ids --n 32 --pipeline-handoff $h > $STATE/r32-$h.ids && cmp $STATE/r32-one.ids $STATE/r32-$h.ids" \
      "--pp 2, $h: the one-card ids"
  done
  chk "build/src/cli/b70-decode $ck --layers auto --max-len auto --ids $ids --n 8 > /dev/null" "--layers auto --max-len auto"
  grab_all plan 'layers: |max_len: |plan at|^engine:' 8
  grab_all memory '^memory' 6
  finish
}
stage r32.ple_rate 32 optin gpu - r32.k1 "Task 5 (spec 22 P0's first host-USM number): qwen4exp_kernels_test --ple-rate 16, then 51.8 (the real table's size; stop other RAM users: free -g >= 70 GB) - q4_ple_gather alone in a replayed list, 10^4 replays with random ids, device 0, device 1, both: the median us per token, the implied bus rate, every row read back equal; the 'wide' mode's 256 rows a replay"
st_r32_ple_rate() {
  x "free -g | head -2"
  chk "flock \$HOME/b70-gpu.lock build/tests/qwen4exp_kernels_test --ple-rate 16" "the PLE gather's rate at 16 GB"
  chk "flock \$HOME/b70-gpu.lock build/tests/qwen4exp_kernels_test --ple-rate 51.8" "the PLE gather's rate at the real table's 51.8 GB"
  grab_all rate 'us per token|GB/s|read back|wide|device' 24
  finish
}
stage r32.bench_attn 32 optin gpu - r32.k1 "decision 10's evidence: qwen4exp_kernels_test --bench-attn - the sparse kernel over the identity list against attn_v2 at 24 / 2 heads at depths 512 / 1024 / 2048, interleaved, median of 3 (one sparse kernel at every depth, or v2 below 2052)"
st_r32_bench_attn() {
  idle before
  chk "flock \$HOME/b70-gpu.lock build/tests/qwen4exp_kernels_test --bench-attn" "the sparse kernel against attn_v2"
  idle after
  grab_all bench 'depth|us|median' 12
  finish
}
stage r32.speed 32 optin gpu q4exp_intel,q4exp_intel_ple r32.partial "Task 7: decode on what fits - Intel's checkpoint --layers 18 on one card and 38 on two (copy, peer), --bench --depth 1024 / 4096 / 32768 --tg 256, int8 and bf16 heads, flash and eager - interleaved, median of 3; launches per token, each card's step time, against the derived roofline per layer (spec 21 §3); then the {S, layout} and UP_KS / DN_KS sweeps (r32.sweeps)"
st_r32_speed() {
  idle before
  local d arms="" ratios="" k="ZE_AFFINITY_MASK=0,1" s="${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}"
  for d in 1024 4096 32768; do
    arms="$arms d$d-L18 '$(bench_cmd "$s" --layers 18 --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    arms="$arms d$d-L18-bf16head '$(bench_cmd "$s" --layers 18 --depth $d --tg 256 --max-len 40960)'"
    arms="$arms d$d-L18-eager 'B70_Q4_ATTN=eager $(bench_cmd "$s" --layers 18 --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    arms="$arms d$d-L38 '$k $(bench_cmd "$s" --layers 38 --pp 2 --depth $d --tg 256 --max-len 40960 --lm-head int8)'"
    arms="$arms d$d-L38-peer '$k $(bench_cmd "$s" --layers 38 --pp 2 --depth $d --tg 256 --max-len 40960 --lm-head int8 --pipeline-handoff peer)'"
    ratios="$ratios --ratio d$d-L18-bf16head/d$d-L18 --ratio d$d-L18-eager/d$d-L18 --ratio d$d-L38-peer/d$d-L38"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the Qwen3.8-Flash-Next decode arms"
  idle after
  grab_all memory '^memory' 8
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r32.sweeps 32 manual - q4exp_intel - "Task 7's sweeps: the int4 rows' {S, layout} (q||gate||k||v 2560x13312, out_proj / o_proj 6144x2560: S 1 / 2 / 4, L 0 / 1; qkv||z stays S 1 for gdn_step) on our synthetic, and q4_moe's UP_KS / DN_KS"
st_r32_sweeps() {
  say "# {S, layout} of 2560x13312 and 6144x2560 (probe_gemv rows, spec 14 step 4's method); model::qwen4exp()'s qkvg_s / gdn_out_s / o_s and kernels::qwen4exp's names move together"
  say "# UP_KS / DN_KS: rebuild q4_moe.cl's variants per value, qwen4exp_kernels_test, then build/src/cli/b70-decode <synthetic> --layers 4 --bench --depth 4096 --tg 256 --lm-head int8 per value"
  say "# record: docs/BENCHMARKS.md 'Qwen3.8-Flash-Next (spec 21)'"
}

# ======================================================================================
row 33 "spec 21d - Qwen3.8-Flash-Next (qwen4_exp) prefill: dense flash to 2050, the indexer and the sparse flash past it, the 512-expert grouped MoE, the PLE over a chunk, one hand-off per chunk (plan 21d)"
rownote 33 "Written blind on the Mac, after row 32 (21c's decode on the card): none of the 30 new binaries of the 21d block was ever compiled by ocloc - the DPAS sparse flash over row lists (q4_pf_sparse_attn_Q24KV2[_EAGER]), spec 6's flash at 24 / 2 heads (pf_flash_attn_Q24KV2), the 512-expert ushort sort / dequant / combine (q4_pf_moe_E512_T10_D2560_I640_L256), the grouped GEMMs at this family's shapes (pf_moe_gemm_K2560_N1280_SILU, _K640_N2560), 21c's HC / indexer / PLE gather at M 2048 (_PF), the PLE block over a chunk (q4_pf_ple_C2048). The walk (358 / 361 / 360 launches a chunk at --layers 4) and the sequential two-card chunk hand-off never ran."
rownote 33 "THREE existing sources changed, each by a define every existing binary leaves unset: prefill/pf_gated_head.cl (GDN_GATE_SIGMOID: pf_gated_head_SIG), qwen4exp/q4_qsa.cl (QSA_PF: the prep without its in-launch ring write + q4_qsa_ring; the indexer pitch 768), qwen4exp/q4_ple.cl (PLE_PF: the gather's tokens from the chunk's ids, no ring write). Their existing binaries' preprocessed sources were checked on the Mac (pf_gated_head x 2, q4_qsa_M1: byte for byte; q4_ple_gather_M1_* token for token) - G0's g0.sha must show every pf_gated_head*, q4_qsa_M1* and q4_ple_* binary unchanged (r33.k0); if one moved, the variant moves to a copy in src/kernels/qwen4exp/ and the source is restored (plan 21d F0)."
rownote 33 "Bars PROPOSED (spec 18c's family; the tests print the distributions): prefill vs decode's fill - every row of the first QSA layer >= 0.999, all rows' median >= 0.9998, p01 >= 0.99; the sparse flash against fp64 >= 0.99999 (spec 6 K1), eager >= 0.9999 / 0.999; routing / selection near-ties 1e-3 (B70_Q4_TIE_TOL / B70_Q4_SEL_TOL, row 30 decides), a 0 gap undetermined. Chunks of 16 are printed, not gated bitwise: the GDN's 64-row WY sub-chunks move with the chunk start (chunks of 64 ARE gated bitwise)."
rownote 33 "Data: r33.prefill / r33.golden need the synthetic checkpoints and their golden sets (r31.synth, r32.oracle_synth); r33.golden_real needs 21a's --layers 18 set (r30.golden_layers) and the PLE int8 file (r31.ple_convert); r33.pp needs two GPUs (after r22.p1). With prefill planned Intel's forms fit 18 layers on one card (max_len auto 70144) and 37 on two at 32768 (derived) - 38 does not fit beside the 1.49 GB prefill scratch a device: the real two-card runs use --layers 37."
stage r33.k0 33 default cpu - g0.sha,g0.bitwise,g0.suite "K0 (F0): every pre-existing binary identical - pf_gated_head.cl's GDN_GATE_SIGMOID, q4_qsa.cl's QSA_PF and q4_ple.cl's PLE_PF leave pf_gated_head / _GK16V32, q4_qsa_M1_T512_W1024 and every q4_ple_* M1 binary byte-identical; kernel_cmdlines +30 / -0 / ~0 with K2 and Kolibri on; 21c's decode gates not failed (779 / 75 launches unchanged)"
st_r33_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r32.k1 r32.k3
  x "grep -E 'pf_gated_head(_GK16V32)?(\.bin)?$|q4_qsa_M1|q4_ple_' $STATE/g0-sha/changed.txt 2>/dev/null && echo 'a source-changed binary MOVED' || echo 'no pf_gated_head / q4_qsa / q4_ple binary moved'"
  finish
}
stage r33.host 33 default cpu - - "host: qwen4exp_pf_ref_test (the 512-expert sort and its adversaries == pf_moe_ref::sort, row independence, the combine == 21c's down chain, dense_rows, the sparse attention's tiled fp64 walk == the direct softmax and the eager chain == q4ref's, the indexer and the PLE over chunks == decode's M = 1 chains with their rings), qwen4exp_pf_variant_names_test, qwen4exp_plan_test (358 / 361 / 360 launches at --layers 4, 3944 / 3968 at 48, the 1.49 GB prefill scratch, N with prefill 18 / 37)"
st_r33_host() {
  run_tests '^(qwen4exp_pf_ref_test|qwen4exp_pf_variant_names_test|qwen4exp_plan_test)$'
  jgrab host '^qwen4exp_(pf_ref|pf_variant_names|plan)_test$' 'sort:|combine:|dense_rows:|sparse|indexer|PLE|21d:|OK'
  finish
}
stage r33.k1 33 default gpu - r33.host "F2 for the prefill, no checkpoint: the 30 binaries built (kbins - their first ocloc compile) and qwen4exp_pf_kernels_test - the slabs exact, the HC / indexer / PLE at M 2048 bitwise q4ref's, the router rows bitwise decode's and q4_route on grid (1, C), the MoE (sort / gather / dequant exact, block 512 in both shared forms, GROUPED == DENSE bitwise, reversed and replayed bitwise), the sort adversaries, the dense flash and the sparse flash against fp64 (>= 0.99999) at chunk starts 2048 / 4096 / 30000 / 131072 / a 300-row tail, EAGER (0.9999 / 0.999), pf_attn_gate; and qwen4exp_prefill_gdn_test (gdn_chunk_q4 against decode's gdn_step + _SIG: 64-chunks bitwise, the band printed)"
st_r33_k1() {
  kbins pf_gated_head_SIG pf_embed_gather_D2560 pf_ab_proj_D2560 pf_moe_router_K2560_N528 pf_dequant_slab_K2560_N16384_L0 \
    pf_dequant_slab_K2560_N13312_L0 q4_pf_bf16_slab_K2560_N16384 q4_pf_bf16_slab_K2560_N13312 q4_pf_bf16_slab_K6144_N2560 \
    q4_pf_bf16_slab_K2560_N640 q4_pf_bf16_slab_K10240_N336 q4_pf_bf16_slab_K2560_N12800 q4_hc_combine_norm_M2048_E \
    q4_hc_combine_norm_M2048_S1 q4_hc_combine_norm_M2048_Y q4_hc_combine_norm_M2048_Y_NN q4_hc_combine_norm_M2048_X \
    q4_hc_up_mix_M2048_I_D512 q4_qsa_M2048_T512_W1024_PF q4_ple_gather_M2048_BF16_PF q4_ple_gather_M2048_F32_PF \
    pf_attn_prep_q16_Q24KV2 pf_attn_Q24KV2 pf_flash_attn_Q24KV2 q4_pf_sparse_attn_Q24KV2 q4_pf_sparse_attn_Q24KV2_EAGER \
    q4_pf_moe_E512_T10_D2560_I640_L256 pf_moe_gemm_K2560_N1280_SILU pf_moe_gemm_K640_N2560 q4_pf_ple_C2048
  run_tests '^(qwen4exp_pf_kernels_test|qwen4exp_prefill_gdn_test)$' qwen4exp
  jgrab k1-prefill '^(qwen4exp_pf_kernels_test|qwen4exp_prefill_gdn_test)$' 'cos|ulp|bitwise|grouped|dense|gdn:|OK|FAIL'
  finish
}
stage r33.reject 33 default gpu - - "the refusals before the device (tests/model/qwen4exp: the model_type stub): cli_reject_qwen4exp_prefill_int8 (naming the 1024-k Hadamard blocks), _sycl (no walk), _chunk (above 2048); 21c's cli_reject_qwen4exp_prefill is gone"
st_r33_reject() {
  run_tests '^cli_reject_qwen4exp_prefill_(int8|sycl|chunk)$'
  finish
}
stage r33.prefill 33 default gpu oracle_q4exp_synth r33.k1,r32.k3 "F4 on prefill, one card: qwen4exp_prefill_synth_{ours,intel}_test, _eager_test (4500 ids of q4exp_4k: chunks at 0 (dense), 2048 (3 dense rows), 4096 (sparse)) - launches = prefill_chunk_launches per chunk + 6, plan == allocation, immediate / recorded / replayed bitwise, chunks of 64 bitwise (16 printed), Review Focus 5 (state, routes, selections and tokens against decode's fill, per layer), the injected run chunked == whole; prefill_split_qwen4exp_test (splits at 64 / 2048 bitwise, 2051 / 4097 within the bars)"
st_r33_prefill() {
  run_tests '^(qwen4exp_prefill_synth_(ours|intel|eager)_test|prefill_split_qwen4exp_test)$' qwen4exp '' \
    "${B70_Q4_TIE_TOL:+B70_Q4_TIE_TOL=$B70_Q4_TIE_TOL}"
  jgrab prefill '^(qwen4exp_prefill_synth|prefill_split_qwen4exp)' 'walk|F4|chunks|median|p01|routing|tokens|injected|split at|OK|FAIL'
  finish
}
stage r33.golden 33 default gpu oracle_q4exp_synth_golden r33.prefill "F3 on prefill, the synthetic golden sets: qwen4exp_golden_prefill_synth_{ours,intel}_test, _c16, _i8head, _inject (the reference's selections through set_prefill_injector: every determined row), _eager - the tie-aware token gate from row T - 1, the routing diagnostic and gate S on the last chunk's rows"
st_r33_golden() {
  run_tests '^qwen4exp_golden_prefill_synth_(ours|intel)(_c16|_i8head|_inject|_eager)?_test$' qwen4exp '' \
    "${B70_Q4_TIE_TOL:+B70_Q4_TIE_TOL=$B70_Q4_TIE_TOL} ${B70_Q4_SEL_TOL:+B70_Q4_SEL_TOL=$B70_Q4_SEL_TOL}"
  jgrab golden-prefill '^qwen4exp_golden_prefill_synth' 'gate:|routing:|gate S:|prefill:|OK|FAIL'
  finish
}
stage r33.golden_real 33 default gpu q4exp_intel,q4exp_intel_ple,oracle_q4exp_layers r33.golden "F3 on prefill on Intel's checkpoint at --layers 18 (int8 head, one card): qwen4exp_golden_prefill_intel_test against 21a's truncated set oracle-out-q4exp-L18"
st_r33_golden_real() {
  x "free -g | head -2"
  run_tests '^qwen4exp_golden_prefill_intel_test$' qwen4exp '' \
    "${B70_Q4_TIE_TOL:+B70_Q4_TIE_TOL=$B70_Q4_TIE_TOL} ${B70_Q4_SEL_TOL:+B70_Q4_SEL_TOL=$B70_Q4_SEL_TOL}"
  jgrab golden-real '^qwen4exp_golden_prefill_intel' 'gate:|routing:|gate S:|prefill:|OK|FAIL'
  finish
}
stage r33.pp 33 default gpu oracle_q4exp_synth r33.prefill,r22.p1 "two cards: qwen4exp_prefill_pp_test (--pp 2 --pipeline-split 2 against --pp 1, BITWISE: one prefill, recorded + replayed, a 2051 + rest continuation, chunks of 64, both Control blocks; P4 - a dropped chunk hand-off throws within 30 s naming device 1, reset() recovers), prefill_split_qwen4exp_pp_test, qwen4exp_golden_prefill_synth_pp_test; Intel's --layers 37 on two cards (qwen4exp_prefill_intel_pp_test) when present"
st_r33_pp() {
  run_tests '^(qwen4exp_prefill_pp_test|prefill_split_qwen4exp_pp_test|qwen4exp_golden_prefill_synth_pp_test|qwen4exp_prefill_intel_pp_test)$' \
    '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^(qwen4exp_prefill_pp|prefill_split_qwen4exp_pp|qwen4exp_golden_prefill_synth_pp|qwen4exp_prefill_intel_pp)' \
    'pp:|P4|split at|gate:|walk|median|OK|FAIL|SKIP'
  finish
}
stage r33.cli 33 default gpu oracle_q4exp_synth r33.pp "CLI: b70-decode <ours synthetic> --layers 4 --ids q4exp_4k.ids --n 32 --prefill on one card against --pp 2 (the same ids) and against the decode-only run (a difference judged by the tie rule); --prefill-chunk 1000; --bench --prefill-length 4096 (the pp row); --layers auto --max-len auto with --prefill (the prefill scratch planned)"
st_r33_cli() {
  local ck="oracle-out-q4exp-synth/ours/ckpt" ids="tests/golden/prompts/q4exp_4k.ids" a
  chk "build/src/cli/b70-decode $ck --layers 4 --ids $ids --n 32 > $STATE/r33-one.ids" "one card, decode"
  chk "build/src/cli/b70-decode $ck --layers 4 --ids $ids --n 32 --prefill > $STATE/r33-one-pf.ids" "one card, --prefill"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --layers 4 --pp 2 --ids $ids --n 32 --prefill > $STATE/r33-two-pf.ids && cmp $STATE/r33-one-pf.ids $STATE/r33-two-pf.ids" \
    "--pp 2 --prefill: the one-card --prefill ids"
  chk "build/src/cli/b70-decode $ck --layers 4 --ids $ids --n 32 --prefill --prefill-chunk 1000 > $STATE/r33-c1000.ids" "--prefill-chunk 1000"
  x "for a in one one-pf two-pf c1000; do printf '%s: ' \$a; tr '\\n' ' ' < $STATE/r33-\$a.ids; echo; done; if cmp -s $STATE/r33-one.ids $STATE/r33-one-pf.ids; then echo 'PREFILL ids identical to the decode-only run'; else echo 'PREFILL ids differ from the decode-only run: judge the first difference by the tie rule'; fi"
  chk "$(bench_cmd "$ck" --layers 4 --prefill-length 4096 --tg 16 --max-len 8192) > $STATE/r33-pp.row" "--bench --prefill-length 4096"
  chk "build/src/cli/b70-decode $ck --layers auto --max-len auto --ids $ids --n 8 --prefill > /dev/null" "--layers auto --max-len auto with --prefill"
  grab_all prefill '^prefill:|^pp:|^PREFILL ids|max_len: auto|layers: auto' 10
  grab_all memory '^memory' 6
  finish
}
stage r33.speed 33 optin gpu q4exp_intel,q4exp_intel_ple r33.golden_real "plan 21d Task 5: Intel's checkpoint, l0 - pp4096 / pp32768 at --layers 18 (one card) and 37 (two, --pp 2), pp131072 at the planner's N (--layers auto: 18 layers fit only to 70144 with prefill) - tg16 after, --lm-head int8; flash vs B70_Q4_ATTN=eager at 32768; interleaved pairs, median of 3, uptime - BENCHMARKS 'Qwen3.8-Flash-Next (spec 21)' prefill rows"
st_r33_speed() {
  idle before
  local n k="ZE_AFFINITY_MASK=0,1" s="${SNAP_Q4EXP_INTEL:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}" arms=""
  x "uptime"
  for n in 4096 32768; do
    arms="$arms pp$n-L18 '$(bench_cmd "$s" --layers 18 --prefill-length $n --tg 16 --max-len 33024 --lm-head int8)'"
    arms="$arms pp$n-L37 '$k $(bench_cmd "$s" --layers 37 --pp 2 --prefill-length $n --tg 16 --max-len 33024 --lm-head int8)'"
  done
  arms="$arms pp131072-auto '$(bench_cmd "$s" --layers auto --prefill-length 131072 --tg 16 --max-len 131328 --lm-head int8)'"
  arms="$arms pp32768-L18-eager 'B70_Q4_ATTN=eager $(bench_cmd "$s" --layers 18 --prefill-length 32768 --tg 16 --max-len 33024 --lm-head int8)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio pp32768-L18-eager/pp32768-L18 --$arms" "the Qwen3.8-Flash-Next prefill arms"
  x "uptime"
  idle after
  grab_all prefill '^pp:|^prefill:|layers: auto' 12
  grab_all memory '^memory' 8
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r33.p0 33 manual - q4exp_intel - "plan 21d Task 5's profile and levers (one chunk's anatomy, the recorded levers)"
st_r33_p0() {
  say "# B70_PREFILL_PROFILE=1 build/src/cli/b70-decode \$SNAP_Q4EXP_INTEL --layers 18 --bench --prefill-length 4096 --tg 1 --lm-head int8: one chunk's phases - the expert dequant pass (moe_weights), the HC kernels at M = C (the HC up ~6.7 GFLOP a chunk an HC on plain FMA, ESTIMATED ~0.5 ms), the scores (fp32 FMA; the DPAS GEMM form is the lever), the sparse flash"
  say "# per-device busy time per chunk on two cards (--layers 37 --pp 2): the sequential hand-off leaves one card idle - spec 16c's overlapped chunk pipeline is the recorded lever"
  say "# levers, recorded not built: HC up through DPAS slabs, the scores as one GEMM, an SLM-fused int4 grouped GEMM (spec 15d Task 1's arm) removing the dequant pass, spec 22's prefill streaming once the full model runs"
  say "# record: docs/BENCHMARKS.md 'Qwen3.8-Flash-Next (spec 21)' prefill rows"
}

# ======================================================================================
row 34 "spec 21e - Qwen3.8-Flash-Next (qwen4_exp) served truncated: b70-serve over Qwen4ExpEngine, prefix-cache snapshots, the MTP head (draft / verify / commit), the A4 set and its CPU reference for spec 22 (plan 21e)"
rownote 34 "Written blind on the Mac, after rows 32 / 33 (21c's decode and 21d's prefill on the card): none of the 122 binaries of the 21e block was ever compiled by ocloc - the head's norm / fuse (q4_mtp_M<1..4>_{SINGLE,STREAM}, the prefill head pass q4_mtp_M2048_*_PF), the verify list at M = 2..4 (21c's decode kernels at M rows: the int4 / bf16 / i8w GEMVs, HC, QSA, route / MoE, PLE gather), the per-layer GDN slots (gdn_step_slots_M<1..4>_G1), the PLE block at M > 1 (q4_pf_ple_C4). The verify list (779 + 2 + 10 + M launches at 48 layers), the draft step (27 / 26, + 2 when it selects), the snapshot runs and the server never ran. No existing kernel source changed (kernel_cmdlines +122 / -0 / ~0): G0's g0.sha is the 'nothing moved' check."
rownote 34 "Data: every card stage runs on the synthetic checkpoints (r31.synth; 4 real-width layers + the MTP layer) - r34.fixture writes M1's data (oracle-out-q4exp-mtp, box CPU ~0.5 h, ESTIMATED); Intel's --layers 18 rows (qwen4exp_snapshot_gpu_intel_test, qwen4exp_mtp_intel_test) need Intel's checkpoint and its PLE int8 file (rows 30 / 31). The served tokenizer is the ORIGINAL's (its split has \\p{M}); Intel's checkpoint ships Qwen3.8's tokenizer.json - b70-serve refuses it unless --tokenizer names the original's file (cli_reject_serve_qwen4exp_tokenizer)."
rownote 34 "MTP acceptance has no engine bar here: on the synthetic checkpoints the drafts are random (M3 is greedy LOSSLESS at K = 1..3, not a rate). Decision 4's numbers are r34.accept's (the reference's teacher-forced acceptance on Intel's checkpoint, both pre_fc_norm_hidden forms, the selection reused and fresh) until spec 22 runs the whole model on the card. The verify cost (M = 2..4 against M = 1) and the per-layer expert union are r34.mtp's cost lines: hand them back."
rownote 34 "Spec 22's, not this row's: A4 on the engine (the reference is r34.a4_ref's), passkey at the context spec 22 chooses, the llama-benchy rows, the full-model acceptance and speed. b70-serve refuses the whole model (no --layers N) naming spec 22, and --mtp auto naming spec 22's numbers."
stage r34.k0 34 default cpu - g0.sha,g0.bitwise,g0.suite "K0: every pre-existing binary identical (21e adds 122, changes none); the server path's suite unchanged (golden_server_test, prefix_gpu_*, mtp_verify_test, snapshot_test in G0's suite); rows 32 / 33's gates not failed"
st_r34_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  need_ok r32.k1 r32.k3 r33.k1 r33.prefill
  run_tests '^(prefix_cache_test|prefix_server_test|mtp_server_test|lookup_server_test)$'
  finish
}
stage r34.host 34 default cpu - - "host: qwen4exp_tokenizer_test (the original's tokenizer.json bitwise transformers 5.19.0 over 24 cases + a 10240-line corpus, Qwen3.8's file refused by its SHA-256, the three chat renders), qwen4exp_server_test (ChatFormat, EOS, decision 11's sampling defaults, the serve refusals), qwen4exp_snapshot_test (the snapshot runs on simulated devices: one host layout for --pp 1 / 2, block and request ends, sizes), qwen4exp_mtp_names_test (every bound name built, every 21e binary bound; the verify / draft / head-pass launch counts), qwen4exp_plan_test"
st_r34_host() {
  run_tests '^(qwen4exp_tokenizer_test|qwen4exp_server_test|qwen4exp_snapshot_test|qwen4exp_mtp_names_test|qwen4exp_plan_test)$'
  jgrab host '^qwen4exp_(tokenizer|server|snapshot|mtp_names|plan)_test$' 'tokenizer|render|snapshot|bytes|verify|draft|launch|OK'
  finish
}
stage r34.k1 34 default gpu - r34.host "the 21e block's 122 binaries built (kbins - their first ocloc compile) and qwen4exp_mtp_names_test against them"
st_r34_k1() {
  kbins \
    attn_prep_M2_Q24KV2 attn_prep_M2_Q24KV2_S1 attn_prep_M3_Q24KV2 attn_prep_M3_Q24KV2_S1 attn_prep_M4_Q24KV2 \
    attn_prep_M4_Q24KV2_S1 embed_gather_M2_D2560 embed_gather_M3_D2560 embed_gather_M4_D2560 gdn_step_slots_M1_G1 \
    gdn_step_slots_M2_G1 gdn_step_slots_M3_G1 gdn_step_slots_M4_G1 gemv_M2_K2560_N13312_S2_L0 \
    gemv_M2_K2560_N16384_S1_L0 gemv_M2_K6144_N2560_S4_L0 gemv_M3_K2560_N13312_S2_L0 gemv_M3_K2560_N16384_S1_L0 \
    gemv_M3_K6144_N2560_S4_L0 gemv_M4_K2560_N13312_S2_L0 gemv_M4_K2560_N16384_S1_L0 gemv_M4_K6144_N2560_S4_L0 \
    gemv_bf16_M1_K2560_N2560 gemv_bf16_M2_K10240_N320_C16_S16 gemv_bf16_M2_K10240_N336_C16_S16 \
    gemv_bf16_M2_K2560_N12800 gemv_bf16_M2_K2560_N128_C16_S16 gemv_bf16_M2_K2560_N13312 gemv_bf16_M2_K2560_N16384 \
    gemv_bf16_M2_K2560_N248320 gemv_bf16_M2_K2560_N2560 gemv_bf16_M2_K2560_N528_C16_S16 \
    gemv_bf16_M2_K2560_N640_C16_S16 gemv_bf16_M2_K6144_N2560 gemv_bf16_M3_K10240_N320_C16_S16 \
    gemv_bf16_M3_K10240_N336_C16_S16 gemv_bf16_M3_K2560_N12800 gemv_bf16_M3_K2560_N128_C16_S16 \
    gemv_bf16_M3_K2560_N13312 gemv_bf16_M3_K2560_N16384 gemv_bf16_M3_K2560_N248320 gemv_bf16_M3_K2560_N2560 \
    gemv_bf16_M3_K2560_N528_C16_S16 gemv_bf16_M3_K2560_N640_C16_S16 gemv_bf16_M3_K6144_N2560 \
    gemv_bf16_M4_K10240_N320_C16_S16 gemv_bf16_M4_K10240_N336_C16_S16 gemv_bf16_M4_K2560_N12800 \
    gemv_bf16_M4_K2560_N128_C16_S16 gemv_bf16_M4_K2560_N13312 gemv_bf16_M4_K2560_N16384 gemv_bf16_M4_K2560_N248320 \
    gemv_bf16_M4_K2560_N2560 gemv_bf16_M4_K2560_N528_C16_S16 gemv_bf16_M4_K2560_N640_C16_S16 \
    gemv_bf16_M4_K6144_N2560 gemv_i8w_M2_K2560_N248320 gemv_i8w_M3_K2560_N248320 gemv_i8w_M4_K2560_N248320 \
    prep_gated_head_M2_SIG prep_gated_head_M3_SIG prep_gated_head_M4_SIG q4_hc_combine_norm_M2_E \
    q4_hc_combine_norm_M2_S1 q4_hc_combine_norm_M2_S4 q4_hc_combine_norm_M2_X q4_hc_combine_norm_M2_Y \
    q4_hc_combine_norm_M2_Y_NN q4_hc_combine_norm_M3_E q4_hc_combine_norm_M3_S1 q4_hc_combine_norm_M3_S4 \
    q4_hc_combine_norm_M3_X q4_hc_combine_norm_M3_Y q4_hc_combine_norm_M3_Y_NN q4_hc_combine_norm_M4_E \
    q4_hc_combine_norm_M4_S1 q4_hc_combine_norm_M4_S4 q4_hc_combine_norm_M4_X q4_hc_combine_norm_M4_Y \
    q4_hc_combine_norm_M4_Y_NN q4_hc_up_mix_M2 q4_hc_up_mix_M2_I q4_hc_up_mix_M3 q4_hc_up_mix_M3_I q4_hc_up_mix_M4 \
    q4_hc_up_mix_M4_I q4_moe_M2_E512_T10_D2560_I640_SH4 q4_moe_M2_E512_T10_D2560_I640_SHB \
    q4_moe_M3_E512_T10_D2560_I640_SH4 q4_moe_M3_E512_T10_D2560_I640_SHB q4_moe_M4_E512_T10_D2560_I640_SH4 \
    q4_moe_M4_E512_T10_D2560_I640_SHB q4_mtp_M1_SINGLE q4_mtp_M1_STREAM q4_mtp_M2048_SINGLE_PF \
    q4_mtp_M2048_STREAM_PF q4_mtp_M2_SINGLE q4_mtp_M2_STREAM q4_mtp_M3_SINGLE q4_mtp_M3_STREAM q4_mtp_M4_SINGLE \
    q4_mtp_M4_STREAM q4_pf_bf16_slab_K2560_N2560 q4_pf_ple_C4 q4_ple_gather_M2_BF16 q4_ple_gather_M2_F32 \
    q4_ple_gather_M3_BF16 q4_ple_gather_M3_F32 q4_ple_gather_M4_BF16 q4_ple_gather_M4_F32 q4_qsa_M2_T512_W1024 \
    q4_qsa_M3_T512_W1024 q4_qsa_M4_T512_W1024 q4_qsa_attn_M2_T32 q4_qsa_attn_M3_T32 q4_qsa_attn_M4_T32 \
    q4_qsa_attn_eager_M2 q4_qsa_attn_eager_M3 q4_qsa_attn_eager_M4 q4_route_M2_E512_T10_N528_L256 \
    q4_route_M3_E512_T10_N528_L256 q4_route_M4_E512_T10_N528_L256
  run_tests '^qwen4exp_mtp_names_test$'
  finish
}
stage r34.reject 34 default gpu - - "the refusals before the device: cli_reject_serve_qwen4exp_{full (spec 22), mtp_auto, spec_mtp, draft_vocab, kv8, prefill_int8, prefill_sycl, split_alone, tokenizer (Qwen3.8's split)} and cli_reject_serve_layers_other (--layers for another model); 21c's cli_reject_qwen4exp_serve is gone"
st_r34_reject() {
  run_tests '^cli_reject_serve_(qwen4exp_(full|mtp_auto|spec_mtp|draft_vocab|kv8|prefill_int8|prefill_sycl|split_alone|tokenizer)|layers_other)$'
  finish
}
stage r34.snapshot 34 default gpu oracle_q4exp_synth r34.k1,r33.prefill "F4 on the prefix cache, one card: qwen4exp_snapshot_gpu_synth_{,intel_,mtp_}test - save at block ends 2048 / 4096 and request ends 2049..2052 / 5000 / 1, load into a fresh engine, the continuation BITWISE a cold run's (logits, GDN state, conv and PLE rings, indexer tails and compressed keys, KV; with the head: its R, KV and keys, a K = 2 speculative continuation); Intel's --layers 18 when present"
st_r34_snapshot() {
  run_tests '^qwen4exp_snapshot_gpu_(synth|synth_intel|synth_mtp|intel)_test$' qwen4exp
  jgrab snapshot '^qwen4exp_snapshot_gpu' 'snapshot|bytes|bitwise|restore|cross|OK|FAIL|SKIP'
  finish
}
stage r34.mtp 34 default gpu oracle_q4exp_synth r34.k1,r33.prefill "F5 on one card: qwen4exp_mtp_synth_{,intel_,eager_,fresh_}test - M1 the head against 21a's port (r34.fixture's data, both norm forms; SKIP without it), M2 the verify rows (M = 2..4) BITWISE the plain steps, M3 greedy lossless at K = 1..3 and the head off, the selection reused (default) and fresh; the verify's cost (M = 2..4 against M = 1, us) and the per-layer expert union; Intel's --layers 18 (qwen4exp_mtp_intel_test) when present"
st_r34_mtp() {
  run_tests '^qwen4exp_mtp_(synth|synth_intel|synth_eager|synth_fresh|intel)_test$' qwen4exp
  jgrab mtp '^qwen4exp_mtp_' 'M1|M2|M3|cost|union|accept|launch|OK|FAIL|SKIP'
  finish
}
stage r34.serve 34 default gpu oracle_q4exp_synth r34.snapshot,r34.mtp "b70-serve on the ours synthetic --layers 4, one card: golden_server_test --chat (prose / code / cjk, 32 greedy ids) = b70-decode --layers 4 --prefill on the response's prompt ids, plain and --mtp 2; the prefix-cache repeat - t4_comment-step / t3_rename-linear_l0 (2748 / 2737 ids) warmed then served (cached_tokens >= 2048), plain and --mtp 2, every output identical to a cache-off server's"
st_r34_serve() {
  local ck="oracle-out-q4exp-synth/ours/ckpt" d="$STATE/r34-serve" only="t4_comment-step,t3_rename-linear_l0"
  local cl="--set toolcall --only $only --max-tokens 32"
  chk "B70_SERVE_ARGS='--layers 4' B70_DECODE_EXTRA='--layers 4' timeout 1800 build/tests/golden_server_test build/src/cli/b70-serve tools/box_validate/decode_extra.sh tests/golden/prompts $ck --chat" \
    "golden_server_test --chat: the server's greedy chat = b70-decode (--layers 4)"
  chk "B70_SERVE_ARGS='--layers 4 --mtp 2' B70_DECODE_EXTRA='--layers 4' timeout 1800 build/tests/golden_server_test build/src/cli/b70-serve tools/box_validate/decode_extra.sh tests/golden/prompts $ck --chat" \
    "golden_server_test --chat through --mtp 2 (greedy lossless)"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $d/cold cold 1 $ck --layers 4 --prefix-cache-gb 0 -- $cl --warmup 0" \
    "the cache-off server"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $d/plain plain 1 $ck --layers 4 -- $cl --warmup 2" \
    "the prefix-cache repeat (each scenario warmed once, then served)"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $d/mtp2 mtp2 1 $ck --layers 4 --mtp 2 -- $cl --warmup 2" \
    "the prefix-cache repeat through --mtp 2"
  chk "python3 tools/box_validate/serve_client.py compare --dir $d --ref cold" "the warmed outputs identical to the cache-off server's"
  chk "python3 -c 'import json, sys; c = [json.loads(l)[\"usage\"].get(\"prompt_tokens_details\", {}).get(\"cached_tokens\", 0) for f in sys.argv[1:] for l in open(f)]; print(\"cached_tokens\", c); sys.exit(0 if c and min(c) >= 2048 else 1)' $d/plain/requests.jsonl $d/mtp2/requests.jsonl" \
    "every warmed request reused at least one 2048-id block"
  grab_all serve '^IDENTICAL|chat of|^cached_tokens|prefix cache:|mtp|max_len' 20
  finish
}
stage r34.pp 34 default gpu oracle_q4exp_synth r34.serve,r22.p1 "two cards: qwen4exp_snapshot_gpu_synth_pp_test (the snapshot runs on --pp 2, bitwise a cold run's), _cross_{,mtp_}test (one host layout: taken under --pp 2, restored under --pp 1 and back, bitwise; with the head), qwen4exp_mtp_synth_pp_test (M2 / M3 and the cost on --pp 2 --pipeline-split 2: the head on device 1 reading device 0's embedding over peer), golden_server_test --chat with b70-serve --pp 2 --mtp 2 = b70-decode on one card"
st_r34_pp() {
  run_tests '^qwen4exp_(snapshot_gpu_(synth_pp|cross|cross_mtp)|mtp_synth_pp)_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^qwen4exp_(snapshot_gpu_(synth_pp|cross)|mtp_synth_pp)' 'snapshot|bitwise|M2|M3|cost|OK|FAIL|SKIP'
  chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='--layers 4 --pp 2 --mtp 2' B70_DECODE_EXTRA='--layers 4' timeout 1800 build/tests/golden_server_test build/src/cli/b70-serve tools/box_validate/decode_extra.sh tests/golden/prompts oracle-out-q4exp-synth/ours/ckpt --chat" \
    "golden_server_test --chat: --pp 2 --mtp 2 = one card's b70-decode"
  finish
}
stage r34.fixture 34 optin cpu oracle_q4exp_synth,oracle_image - "M1's data on the box CPU: qwen4exp_oracle.sh mtp-fixture - 21a's MTP port on the engine-format head (bf16 experts RTN int4 g64), both norm forms, on each synthetic checkpoint (+ Intel's at --layers 18 when present) -> \$DATA/oracle-out-q4exp-mtp/{ours,intel,intel-L18}/mtp_{single,per_stream}.safetensors; then re-links oracle-out*"
st_r34_fixture() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA mtp-fixture" "the MTP fixtures"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  finish
}
stage r34.accept 34 optin cpu q4exp_intel,oracle_image - "decision 4's acceptance on the box CPU: qwen4exp_oracle.sh mtp-accept - qwen4exp_mtp.py accept on Intel's checkpoint (the engine's RTN head), teacher-forced greedy, K <= 3, the selection reused and fresh, over the golden continuations (q4exp_short / q4exp_agentic, after row 30's intel run) and four A4 scenarios, per norm form -> \$DATA/oracle-out-q4exp-mtp/accept_{single,per_stream}.json - HOURS; the pooled lines are the row's numbers until spec 22"
st_r34_accept() {
  x "free -g | head -2"
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA mtp-accept" "the reference's MTP acceptance, both norm forms"
  grab_all accept '^pooled' 8
  finish
}
stage r34.a4_ref 34 optin cpu q4exp_intel,q4exp_orig_small,oracle_image - "plan 21e Task 5 Step 2 on the box CPU: the Qwen3.8-Flash-Next tool-call set (make_set.py --from tests/golden/toolcall on the ORIGINAL's tokenizer and template, transformers 5.19.0: the log says how many of the 36 are Qwen3.8's ids exactly) and qwen4exp_ref.py's reference on Intel's checkpoint (192 ids, resumable) into \$DATA/oracle-out-q4exp-a4 - HOURS; score.py's parse of every reference call (0 failures: the hard bar); then re-links oracle-out*. The engine's A4 run is spec 22's"
st_r34_a4_ref() {
  chk "OUT=$DATA/oracle-out-q4exp-a4 tools/toolcall/a4_ref.sh qwen4exp set" "the Qwen3.8-Flash-Next tool-call set"
  chk "OUT=$DATA/oracle-out-q4exp-a4 tools/toolcall/a4_ref.sh qwen4exp ref" "the reference run (qwen4exp_ref.py)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "OUT=$DATA/oracle-out-q4exp-a4 tools/toolcall/a4_ref.sh qwen4exp status"
  chk "python3 tools/toolcall/score.py $DATA/oracle-out-q4exp-a4 bf16" "every reference call parses (Qwen XML)"
  finish
}

# ======================================================================================
row 35 "the W4A4 probe (docs/07 §3): int4 weights x int4 activations at group 256 after h8's runtime Hadamard - the accuracy grid on the box CPU (plan 2026-10-09-w4a4-probe Task 2)"
rownote 35 "A CPU job, opt-in, outside the timed sessions (never during sessions 6-9: a running container makes every timed row ITERATE). Task 1 (tools/rotate/eval_quantised.py sim: h4 / h4p2 / w4a16, --group, --per-class; tools/rotate/test_w4a4.py) is on the tree. r35.grid gives the stopping rule's numbers (best of h4 / h4p2 > 2 points of rel L2 above h8, or > 2 argmax in 100 lost against it, closes W4A4); r35.per_class is Review Focus 5 (each class alone: o_proj / out_proj / down apart). The verdict, Task 2 Step 2 (AutoRound g256, ~3 h, only if h4 is within ~3 points of h8) and Task 3 (the GPU microprobe, only on a pass) are not stages."
rownote 35 "Prompts: §15.4's 478 held-out ids are not in the tree - W4A4_HELDOUT=<file> if the box still has it, otherwise w4a4_grid.sh re-makes a 478-id held-out prompt from this tree's texts (NOT §15's ids: h4 is read against the same run's none / w4a16 / h8, not against §15's 12.0 / 12.5 %). long512 / code / cjk are §15.4's. Data: the bf16 Qwen3.8 (SNAP_QWEN_BF16, default Qwen/Qwen3.8-27B, 54 GB) beside the gate snapshot; MemAvailable >= 70 GB (W4A4_MIN_GB) or the stage SKIPs."
stage r35.grid 35 optin cpu qwen,qwen_bf16,oracle_image - "plan Task 2 Step 1 on the box CPU: w4a4_grid.sh grid - eval_quantised.py sim <Qwen3.8 bf16> <gate checkpoint> --prompt <p> --variants none,w4a16,h8,h4,h4p2 --group 256 --vocab-used 248077 on cjk / code / heldout (478) / long512 in the oracle container (free -g >= 70 GB first) -> \$DATA/oracle-out-w4a4/<p>.grid.{log,json}, resumable - HOURS (est.); the summary tables and the stopping rule computed from the JSON"
st_r35_grid() {
  x "free -g | head -2"
  chk "tools/box_validate/w4a4_grid.sh $DATA grid" "the W4A4 grid (none, w4a16, h8, h4, h4p2 at g256) on four prompts"
  grab_all w4a4 '^(=== .*\.grid\.log|\| (none|w4a16|h8|h4|h4p2) \| all \|)' 40
  grab_all w4a4_rule '^(\| (cjk|code|heldout|long512) \||worst prompt )' 8
  finish
}
stage r35.per_class 35 optin cpu qwen,qwen_bf16,oracle_image r35.grid "Review Focus 5 on the box CPU: w4a4_grid.sh per-class - sim --variants none --per-class h8,h4,h4p2 --group 256 on heldout and long512: each linear class alone (attn_qkv, attn_o, gdn_in, gdn_out, gate_up, down; the rest none), end-to-end rel L2 / worst cos / argmax / top-5 / KL -> \$DATA/oracle-out-w4a4/<p>.per-class.{log,json} - HOURS (est.: 19 forwards a prompt)"
st_r35_per_class() {
  x "free -g | head -2"
  chk "tools/box_validate/w4a4_grid.sh $DATA per-class" "the W4A4 per-class passes (h8, h4, h4p2 one class at a time)"
  grab_all w4a4_class '^(=== .*\.per-class\.log|\| (none|h8|h4|h4p2) \| (all|attn_qkv|attn_o|gdn_in|gdn_out|gate_up|down) \|)' 80
  finish
}

# ======================================================================================
row 36 "spec 22a - the expert-offload probe (P0): the links, copy and zero-copy rates, replayed visibility, host RAM under load, pinned capacity and the alias check, the hit-rate and pruning curves, the projection and its stopping rule (plan 22a)"
rownote 36 "Nothing in src/kernels changes (kernel_cmdlines +0 / -0 / ~0: r36.k0 is G0's); the one src/ change is the host-only runtime::qwen4exp::expert_budget. The probe's two kernels are tools/probe's - po_moe_M1_E512_T10_D2560_I640_SH4 (q4_moe.cl with its two address lines as a pointer-table read, generated by mk_offload_kernel.py and drift-checked) and po_kernels - first compiled by r36.build; arm A binds r32.k1's q4_moe_M1_E512_T10_D2560_I640_SH4."
rownote 36 "Data: r36.curve / r36.reap need row 30's traces (r30.traces: oracle-out-q4exp-traces, 39+ files); r36.prune row 30's golden sets (r30.golden) and Intel's checkpoint, its A4 half row 34's reference (r34.a4_ref); r36.prune_a4 the A4 set and reference. The arms need r32.k1 and two cards in one context (r22.p1). Predecessors to hand in beside: r32.ple_rate (the first host-USM number), r31.load_real's pin line (P0.5's start), r34.accept (the projection's MTP acceptance; else the 0.5-0.8 range, PROPOSED), row 32's per-layer rate (PROJECT_BW) and row 33's compute a chunk (PROJECT_PF_MS). Outputs: \$DATA/oracle-out-q4exp-offload (budget.json, links*.txt, po-*.txt, the curves, prune / project), oracle-out-q4exp-prune (masks, maskeval), oracle-out-q4exp-traces/profile.safetensors (decision 8)."
rownote 36 "Both cards: the run exports DEVICE's one card (ZE_AFFINITY_MASK); every probe_offload command sets ZE_AFFINITY_MASK=0,1 itself, as row 22's do, and a two-card arm that still sees one card FAILS (not a SKIP). The GPU lock is the one lock for both."
rownote 36 "RAM rule: the pinned arms never beside a CPU oracle - r36.pin SKIPs 'host RAM' below 110 GB available or with any container up (probe_offload itself refuses above MemAvailable - 16 GiB, printing the cap: spec 7's rule); r36.hostbw pins 1.34 GB and allocates a 6 GiB triad. The CPU oracle stages (r36.prune, r36.prune_a4: hours to days, ESTIMATED from row 30's walls) run one at a time with free -g >= 70 GB, outside the GPU session."
rownote 36 "Session: an idle box (box-day plan Session 9c), both cards free of other DRM holders, RECORD grade for the timed stages (r36.copy, r36.zc, r36.hostbw): probe_offload warms up ~0.5 s, then times control and candidate as interleaved pairs, median of 5 with the range. A mismatch in r36.replay or r36.alias is reported first, with its replay / page. A STOP from r36.links (a card below Gen3 x8; an UNKNOWN line means lspci -vv needed root - sudo lspci -vv -s 04:00.0 by hand) stops the row: spec 22 §3, stop and report. The record is docs/probe-offload-<date>.md (plan 22a Task 5 Steps 6-7); the operator's go / no-go follows r36.project's STOP / GO-CANDIDATE line."
stage r36.k0 36 default cpu - g0.sha,g0.bitwise,g0.suite "G0 trivially: kernel_cmdlines +0 / -0 / ~0 - plan 22a changes no device code (the probe's kernels are tools/probe's); every pre-existing binary identical"
st_r36_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r36.host 36 default cpu oracle_image - "host: qwen4exp_plan_test (the routed-expert budget = plan() byte for byte; its budget-json lines -> \$DATA/oracle-out-q4exp-offload/budget.json), probe_offload_kernel_drift_test, offload_links.sh --self-test, test_offload_curve.py in the oracle image (the 5.19.0 site for its tiny-model test)"
st_r36_host() {
  local off="$DATA/oracle-out-q4exp-offload"
  run_tests '^(qwen4exp_plan_test|probe_offload_kernel_drift_test)$'
  chk "mkdir -p $off && build/tests/qwen4exp_plan_test --budget-json $off/budget.json | grep -E '^spec 22a|OK'" "the routed-expert budget lines"
  chk "tools/probe/offload_links.sh --self-test" "offload_links.sh's parser on its fixture"
  chk "ORACLE_SNAP=\$PWD/tests/model/qwen4exp tools/oracle/run_in_container.sh 'tools/oracle/qwen4exp_env.sh /ws/oracle-out-q4exp-site > /dev/null && export PYTHONPATH=/ws/oracle-out-q4exp-site && python3 tools/oracle/test_offload_curve.py'" \
    "test_offload_curve.py (the curves, the masks, the projection on synthetic traces; the tiny model's own)"
  grab_all budget '^spec 22a' 6
  finish
}
stage r36.build 36 default gpu - r32.k1 "probe_offload and its two kernels built: po_moe_M1_E512_T10_D2560_I640_SH4 and po_kernels (their first ocloc compile), beside r32.k1's q4_moe_M1_E512_T10_D2560_I640_SH4 (arm A's control)"
st_r36_build() {
  chk "cmake --build build -j$JOBS --target probe_offload" "probe_offload and its kernels"
  kbins po_moe_M1_E512_T10_D2560_I640_SH4 po_kernels q4_moe_M1_E512_T10_D2560_I640_SH4
  finish
}
stage r36.links 36 default cpu - - "P0.1: offload_links.sh idle - both B70s' LnkCap / LnkSta and every bridge up to the root port, lscpu, dmidecode (the command printed when sudo -n is refused), the derived host RAM ceiling; a STOP line (a card below Gen3 x8) stops the row"
st_r36_links() {
  local off="$DATA/oracle-out-q4exp-offload"
  chk "mkdir -p $off && tools/probe/offload_links.sh | tee $off/links.txt" "the PCIe links (a STOP: spec 22 §3, stop and report)"
  grab_all links '^(links:|STOP:|UNKNOWN:|memory:|dmidecode:)' 12
  finish
}
stage r36.links_load 36 default gpu - r36.links,r36.build,r22.p1 "P0.1 under load: offload_links.sh --under-load 20 - LnkSta sampled while probe_offload hold copies 1 GiB in a loop on both cards"
st_r36_links_load() {
  local off="$DATA/oracle-out-q4exp-offload"
  chk "ZE_AFFINITY_MASK=0,1 PROBE_OFFLOAD=build/tools/probe/probe_offload tools/probe/offload_links.sh --under-load 20 | tee $off/links-load.txt" \
    "the links under load (a STOP: spec 22 §3, stop and report)"
  grab_all links_load '^(links:|STOP:|UNKNOWN:|  hold: po hold)' 12
  finish
}
stage r36.copy 36 default gpu - r36.build,r36.links,r22.p1 "P0.2: probe_offload copy --dev 0, 1, both - pinned host -> device at 0.87 / 1.74 / 2.61 MB, 16 / 64 / 256 experts as one copy and per expert, 1 GiB; immediate and replayed lists; the compute ordinal and the copy-only group; both cards concurrently (one thread each behind a start barrier); first-byte latency"
st_r36_copy() {
  local off="$DATA/oracle-out-q4exp-offload" d
  idle r36.copy
  x "mkdir -p $off && : > $off/po-copy.txt"
  for d in 0 1 both; do
    chk "ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload copy --dev $d | tee -a $off/po-copy.txt; r=\${PIPESTATUS[0]}; [ \$r != 77 ] || { echo 'FAIL: this arm needs both cards and saw one (ZE_AFFINITY_MASK)'; exit 1; }; exit \$r" "the copy rates, --dev $d"
  done
  grab_all copy '^po copy .*(shape=(expert|1gib|256_experts_one|latency_[a-z0-9]+) |card=all|SKIP)' 80
  finish
}
stage r36.zc 36 default gpu - r36.copy "P0.3: probe_offload zc --dev 0, 1, both - M = 1, P = 1 / 2 / 7 / 26 launch pairs; arms A (21c's binary over VRAM), B (the table, identity), C (every routed slot on host USM), D1-D3 (1-3 slots on host), S (copy-queue staging: an upper bound), K (in-list staging), the streaming ceilings; interleaved pairs, y bitwise A's every run; the mixed-launch law (sum / max / mix) and P per card"
st_r36_zc() {
  local off="$DATA/oracle-out-q4exp-offload" d
  idle r36.zc
  x ": > $off/po-zc.txt"
  for d in 0 1 both; do
    chk "ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload zc --dev $d | tee -a $off/po-zc.txt; r=\${PIPESTATUS[0]}; [ \$r != 77 ] || { echo 'FAIL: this arm needs both cards and saw one (ZE_AFFINITY_MASK)'; exit 1; }; exit \$r" "the zero-copy arms, --dev $d"
  done
  grab_all zc '^po zc .*(law_summary| arm=(A|C|K|S) |FAIL)' 120
  finish
}
stage r36.replay 36 default gpu - r36.zc "P0.3's visibility: probe_offload replay --dev both --n 10000 - arms D2 and C replayed 10^4 times, y compared every replay, a host expert rewritten every 100th replay (the next must see it), the host image's page words read back every 1000th; a mismatch names its replay and page and stops the arm"
st_r36_replay() {
  local off="$DATA/oracle-out-q4exp-offload"
  chk "ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload replay --dev both --n 10000 | tee $off/po-replay.txt; r=\${PIPESTATUS[0]}; [ \$r != 77 ] || { echo 'FAIL: this arm needs both cards and saw one (ZE_AFFINITY_MASK)'; exit 1; }; exit \$r" "the replayed visibility"
  grab_all replay '^po replay ' 12
  finish
}
stage r36.hostbw 36 default gpu - r36.zc "P0.4: probe_offload hostbw --dev both - the STREAM triad over 3 x 2 GiB (1 and all threads, GPUs idle), then arm C on both cards with the CPU idle, beside a 64 MiB copy loop (PROPOSED stand-in for the server's host work), beside the all-thread triad"
st_r36_hostbw() {
  local off="$DATA/oracle-out-q4exp-offload"
  idle r36.hostbw
  chk "ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload hostbw --dev both | tee $off/po-hostbw.txt; r=\${PIPESTATUS[0]}; [ \$r != 77 ] || { echo 'FAIL: this arm needs both cards and saw one (ZE_AFFINITY_MASK)'; exit 1; }; exit \$r" "host RAM under the zero-copy load"
  grab_all hostbw '^po hostbw ' 8
  finish
}
stage r36.pin 36 default gpu - r36.build,r36.links "P0.5: probe_offload pin --dev both - the largest single pinned range (doubling from 1 GiB, then bisection to 1 GiB) and 48 per-layer ranges + the PLE's 16 + 16 shards: where it fails, MemAvailable before / after, s per GB, every page's tag read back on both cards (SKIP 'host RAM' below 110 GB available or with a container up)"
st_r36_pin() {
  local off="$DATA/oracle-out-q4exp-offload"
  chk "a=\$(awk '/^MemAvailable:/ { print int(\$2 / 1048576) }' /proc/meminfo); c=\$(docker ps -q 2>/dev/null | wc -l); if [ \"\$a\" -lt 110 ] || [ \"\$c\" -gt 0 ]; then echo \"SKIP_REASON host RAM: MemAvailable \$a GB (< 110) or \$c container(s) up - the pinned arms never run beside a CPU oracle\"; exit 77; fi; ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload pin --dev both | tee $off/po-pin.txt" \
    "pinned capacity and the page tags"
  grab_all pin '^po pin .*(cap_gib|single_largest|ranges|FAIL|SKIP)' 8
  finish
}
stage r36.alias 36 default gpu - r36.build,r36.links "P0.5: probe_offload alias --arena-gb 22 --dev 0, then 1 - a model-sized allocation history (18 layers at 21b's sizes, the int8 head, the PLE pointer table; freed), then one 22 GiB device arena: a salted tag a 2 MiB page written and read back by the device, a second salt; an alias printed with both addresses"
st_r36_alias() {
  local off="$DATA/oracle-out-q4exp-offload" d
  x ": > $off/po-alias.txt"
  for d in 0 1; do
    chk "ZE_AFFINITY_MASK=0,1 build/tools/probe/probe_offload alias --arena-gb 22 --dev $d | tee -a $off/po-alias.txt; r=\${PIPESTATUS[0]}; [ \$r != 77 ] || { echo 'FAIL: this arm needs both cards and saw one (ZE_AFFINITY_MASK)'; exit 1; }; exit \$r" "the alias check, device $d"
  done
  grab_all alias '^po alias .*(result=|ALIAS|CORRUPT)' 12
  finish
}
stage r36.curve 36 default cpu oracle_q4exp_traces,oracle_image - "P0.6: qwen4exp_oracle.sh curve - per-layer counts, h(f) held out (sha256 halves, the A4 scenarios 18 / 18) and in sample, uniform and per layer, per class; swaps at r36.copy's measured expert copy (when it ran); MTP verify unions (K 1-4) and the head's own routes; prefill bytes (2048 / 4096); the routing profile -> \$DATA/oracle-out-q4exp-traces/profile.safetensors"
st_r36_curve() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA curve" "the hit-rate curves and the profile"
  grab_all curve '^(reach |budget max_len|\| [0-9]+ \| [0-9.]+ \| [0-9]+ \||mtp head|profile:)' 60
  finish
}
stage r36.reap 36 default cpu oracle_q4exp_traces,oracle_image - "P0.8's masks: qwen4exp_oracle.sh reap - REAP (p x onorm) and freq at 50 / 62.5 / 75 / 87.5 % kept from the A4 / agentic / opencode traces, in sample, without A4, and decision 9's 4 folds -> \$DATA/oracle-out-q4exp-prune/masks"
st_r36_reap() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA reap" "the REAP and freq masks"
  grab_all reap '^reap: kept' 40
  finish
}
stage r36.prune 36 optin cpu q4exp_intel,oracle_q4exp,oracle_image r36.reap "P0.8: qwen4exp_oracle.sh prune - maskeval (per row KL(full || masked), argmax) for every REAP mask over q4exp_short / 4k / agentic (+ 32k with PRUNE_32K=1), and the 36 A4 scenarios on their in-sample and own-fold masks over the full reference's continuation - HOURS to days (ESTIMATED), resumable"
st_r36_prune() {
  x "free -g | head -2"
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA prune" "the masked reference's KL and argmax"
  grab_all prune '^maskeval .* kept=' 120
  finish
}
stage r36.prune_a4 36 optin cpu q4exp_intel,oracle_q4exp_a4,oracle_image r36.reap "P0.8: qwen4exp_oracle.sh prune-a4 - the A4 set generated under each REAP fraction whose experts fit two cards whole at 32k (the all-traces mask on 36, each fold's mask on its 9), score.py against the full reference - DAYS (ESTIMATED), resumable per scenario"
st_r36_prune_a4() {
  x "free -g | head -2"
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA prune-a4" "the masked A4 runs and their score"
  grab_all prune_a4 '^(mask-reap-[0-9.]+(-fold[0-3])?: [0-9]+ / [0-9]+ match|qwen4exp_oracle: prune-a4: kept fractions)' 24
  finish
}
stage r36.project 36 default cpu oracle_image r36.host,r36.zc,r36.curve "P0.7: qwen4exp_oracle.sh project - prune-report when r36.prune ran, then the projection (decode at 4k / 32k / 128k per Task 1's budget, MTP at K 1-3, prefill 2048 / 4096, one card) with the measured law and P per card; the last line STOP or GO-CANDIDATE (two cards, 32k, int4 dense, 40 t/s)"
st_r36_project() {
  chk "tools/box_validate/qwen4exp_oracle.sh $DATA project" "the projection and the stopping rule"
  grab_all project '^(project:|\| ours|decision 9|STOP:|GO-CANDIDATE:)' 24
  grab verdict '^(STOP|GO-CANDIDATE):'
  finish
}

# ======================================================================================
row 41 "the llama-benchy comparison matrix (operator, 2026-10-13): b70-serve and vLLM through one client, one set of flags, one card - Qwen3.8 int4 g64, every engine served the same way so the rows compare with numbers others publish"
rownote 41 "One matrix, every arm: llama-benchy pinned (BENCHY_VERSION, default 0.4.0), A = --pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md verbatim (the flags such tables are usually published with - no --no-cache, so a repeated prompt can hit a server's prefix cache: b70-serve's and vLLM's are on by default), B = A with --no-cache (the fair cross-engine reading), D = --pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache (agentic depth). HTTP-inclusive: these rows sit beside, never replace, the engine-internal records (docs/BENCHMARKS.md 'llama-benchy comparisons')."
rownote 41 "Opt-in and timed: run on an idle box (box-day plan Session 6c, after Session 6), device 0, never with a container or the CPU lane up. b70-serve has no batched decode yet (spec 13b / 13c): its concurrency 2 / 4 / 8 rows are requests served one after another - recorded as the baseline batching has to beat, labelled so. vLLM: the recorded baseline's flags (docs/BENCHMARKS.md 'vLLM: how the baseline is served') with --max-num-seqs 8; the version string vllm_benchy.sh prints goes in every vLLM row. r41.vllm_mtp is the MTP-4 + fp8-KV recipe: if the XPU image refuses or crashes on it, record the failure (~/benchy-vllm.log) - no patches."
stage r41.ours 41 optin gpu-self qwen,uvx - "b70-serve <qwen> --max-len 40960 on device 0, --mtp 0 and --mtp auto: matrices A, B and D (int8 head, bf16 KV, defaults otherwise)"
st_r41_ours() {
  local v="${BENCHY_VERSION:-0.4.0}" a="--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md"
  local d="--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache --format md" m
  for m in 0 auto; do
    chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_ARGS='--mtp $m' tools/probe/serve_benchy.sh $a" "b70-serve --mtp $m, matrix A"
    chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_ARGS='--mtp $m' tools/probe/serve_benchy.sh $a --no-cache" "b70-serve --mtp $m, matrix B"
    chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_ARGS='--mtp $m' tools/probe/serve_benchy.sh $d" "b70-serve --mtp $m, matrix D"
  done
  grab_all benchy '^(client: |\| )' 160
  finish
}
stage r41.vllm 41 optin gpu-self qwen,uvx,vllm_image - "vLLM (the box's XPU image) on <qwen>, device 0, no speculation, bf16 KV, --max-num-seqs 8: matrices A, B and D"
st_r41_vllm() {
  local v="${BENCHY_VERSION:-0.4.0}" a="--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md"
  local d="--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache --format md"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN tools/probe/vllm_benchy.sh $a" "vLLM, matrix A"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN tools/probe/vllm_benchy.sh $a --no-cache" "vLLM, matrix B"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN tools/probe/vllm_benchy.sh $d" "vLLM, matrix D"
  grab_all benchy '^(vllm version |client: |\| )' 120
  finish
}
stage r41.vllm_mtp 41 optin gpu-self qwen,uvx,vllm_image - "vLLM's MTP recipe on <qwen>: the checkpoint's MTP head, 4 speculative tokens, fp8 KV, --max-num-seqs 8 - matrices A, B and D (a refusal or crash is the row's reading)"
st_r41_vllm_mtp() {
  local v="${BENCHY_VERSION:-0.4.0}" a="--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md"
  local d="--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache --format md"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MTP=4 KV=fp8 tools/probe/vllm_benchy.sh $a" "vLLM MTP-4 fp8 KV, matrix A"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MTP=4 KV=fp8 tools/probe/vllm_benchy.sh $a --no-cache" "vLLM MTP-4 fp8 KV, matrix B"
  chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MTP=4 KV=fp8 tools/probe/vllm_benchy.sh $d" "vLLM MTP-4 fp8 KV, matrix D"
  grab_all benchy '^(vllm version |client: |\| )' 120
  finish
}

# ======================================================================================
row 42 "spec 23a - Muse-Glimmer-30B: the facts, the CPU reference (MG1), golden sets with the drafter taps, the A4 set in ATEM, gate Q (plan 23a)"
rownote 42 "Box CPU only, no card: nothing here takes the GPU lock. Every stage runs tools/box_validate/muse_oracle.sh in the oracle image with transformers 5.19.0 from <tree>/oracle-out-q4exp-site (spec 21a's site: tools/oracle/qwen4exp_env.sh, made once; the image is never changed). r42.host needs the tiny model (hf download optimum-intel-internal-testing/tiny-random-muse-glimmer, 27 MB) and the original's / the GPTQ repo's small files (a few MB each)."
rownote 42 "Downloads (df -h ~ first; never inside a stage): the int4 checkpoint in either spelling of the same export - Vishva007/Muse-Glimmer-30B-W4A16-AutoRound (auto_round:auto_gptq, no g_idx; what the Mac downloaded) or Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ (gptq, g_idx), 22.8 GB; the bf16 original meta-models/Muse-Glimmer-30B, 59.6 GB (gate Q's reference, decision 14). Without a complete snapshot (every shard, no *.incomplete blob) a stage SKIPs 77 'missing data'."
rownote 42 "RAM floors (MUSE_REF_MIN_GB): 24 GB tests / facts, 40 GB the int4 form (resident embed + head 5.4 GB, two streamed layers, a 2048-row chunk's head and eager scores - ~20 GB at 32k, ESTIMATED), 72 GB the bf16 form. Times are hours (DRY_RUN=1 tools/box_validate/muse_oracle.sh <data> golden prints the ESTIMATED table); the CPU oracle stages run one at a time, outside the GPU sessions."
rownote 42 "Gate Q is a RECORDED MEASUREMENT (operator ruling 2026-10-15: Vishva007's quant is kept): the int4 form's distance from bf16 against decision 5's PROPOSED bars - no more ATEM parse failures than bf16's, first-call agreement >= 25 / 36, mean KL(bf16 || int4) <= 0.02 and top-1 >= 95 % on the A4 rows (prose recorded) - printed WITHIN / OUTSIDE, exit 0 either way; the numbers go into spec 23 §3.3."
rownote 42 "Hand back: each golden log's cap_tie count and raw top-1 / top-2 gap line (MG3's tie rule), every wall / peak RSS line, the perplexities (sanity: below 30 on prose and the 4k text, ESTIMATED), hfcheck's BITWISE lines, the full quant scan's line (qzeros all 0x77777777; g_idx == i // 64 where shipped), gate Q's verdict with its numbers. If the Mac already made a set (oracle-out-muse*), --push-data it instead of re-running."
stage r42.host 42 default cpu oracle_image - "MG1 on the tiny variants in the oracle image: test_muse_ref.py (the streamed port = transformers 5.19.0 bitwise, bf16 / fp32, prompts across the window + cached decode, one test per trap), test_muse_gateq.py (KL / verdict / the muse runner), the toolcall tests (the ATEM render and reader) - muse_oracle.sh tests"
st_r42_host() {
  chk "tools/box_validate/muse_oracle.sh $DATA tests" "the Muse-Glimmer reference tests (tiny model)"
  grab_all tests '^test_|^  |^ok|FAIL|SKIP|passed' 120
  finish
}
stage r42.facts 42 default cpu muse_gptq,oracle_image - "the facts on the int4 checkpoint: the FULL qzeros / g_idx scan (~210 MB), the 5.19.0 citations, the one-message render (one BOS), the names = expected_names - muse_oracle.sh facts"
st_r42_facts() {
  chk "tools/box_validate/muse_oracle.sh $DATA facts" "muse_facts.py --scan-quant and muse_ref.py facts"
  grab_all facts 'FULL SCAN|citations hold|text names|BOS count|accepted' 12
  finish
}
stage r42.ppl 42 optin cpu muse_gptq,oracle_image - "MG1's real-weight sanity: perplexity on prose.txt, code.txt, muse_4k and the agentic transcript, each form present (int4 dequantised; the bf16 original when muse_bf16) - muse_oracle.sh ppl"
st_r42_ppl() {
  chk "tools/box_validate/muse_oracle.sh $DATA ppl" "the reference's perplexity on the real model"
  grab_all ppl '^ppl |^wall|outside' 12
  finish
}
stage r42.hfcheck 42 optin cpu muse_gptq,oracle_image - "streamed = resident = layer-major on real weights: hfcheck --layers 4 on 2100 ids of muse_4k (the window crossed) + 4 steps, raw / capped logits and every H.L* BITWISE, each form present - muse_oracle.sh hfcheck"
st_r42_hfcheck() {
  chk "tools/box_validate/muse_oracle.sh $DATA hfcheck" "hfcheck on the real checkpoint(s)"
  grab_all hfcheck '^hfcheck' 8
  finish
}
stage r42.golden 42 optin cpu muse_gptq,oracle_image - "the golden sets on the int4 checkpoint (hours, resumable): muse_short_prose / short_code / 4k / 8k / 32k / agentic, --gen 32 --taps -> \$DATA/oracle-out-muse; cap_tie and the raw gap per prompt; then re-links oracle-out*"
st_r42_golden() {
  chk "tools/box_validate/muse_oracle.sh $DATA golden" "the Muse-Glimmer golden sets"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all golden 'cap_tie|gap|wall|ppl' 120
  finish
}
stage r42.golden_bf16 42 optin cpu muse_bf16,oracle_image - "the bf16 original's golden sets on muse_short_prose and muse_agentic -> \$DATA/oracle-out-muse-bf16 (the quantisation's own error beside the int4 sets)"
st_r42_golden_bf16() {
  chk "tools/box_validate/muse_oracle.sh $DATA golden-bf16" "the bf16 golden sets"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all golden 'cap_tie|gap|wall|ppl' 40
  finish
}
stage r42.a4 42 optin cpu muse_gptq,muse_bf16,oracle_image - "gate Q's A4 half: both references on the 36 Muse-rendered scenarios (512 new ids, greedy on the capped row, resumable; days, ESTIMATED), then score.py --format atem bf16 gptq -> \$DATA/oracle-out-muse-a4"
st_r42_a4() {
  chk "tools/box_validate/muse_oracle.sh $DATA a4" "the A4 references, both forms"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all a4 'match bf16|parse failure' 8
  finish
}
stage r42.gateq 42 optin cpu oracle_muse_a4,muse_gptq,muse_bf16,oracle_image r42.a4 "gate Q: teacher-forced KL(bf16 || int4) and top-1 over the bf16 run's A4 continuations (+ prose / code / 4k, recorded), then the bars of decision 5 - a recorded measurement (the checkpoint is kept): WITHIN / OUTSIDE with the numbers"
st_r42_gateq() {
  chk "tools/box_validate/muse_oracle.sh $DATA gateq" "gate Q (A4 + KL)"
  grab_all gateq '^(PASS|FAIL) |^gate Q|A4 pooled' 8      # PASS / FAIL per bar: recorded, the stage exits 0
  finish
}
stage r42.taps 42 optin cpu oracle_muse_a4,muse_gptq,oracle_image r42.a4 "23f P0's input: the five drafter taps (the residual after layers 1, 13, 25, 37, 49) over [prompt + the int4 form's own A4 continuation], teacher-forced -> \$DATA/oracle-out-muse-taps"
st_r42_taps() {
  chk "tools/box_validate/muse_oracle.sh $DATA taps" "the drafter taps over the A4 continuations"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  finish
}

# ======================================================================================
row 43 "spec 23b - Muse-Glimmer-30B: the descriptor, the loader (either Vishva007 spelling, the fused GEMVs, the vision skip by name), synthetic real-width checkpoints, the memory plan one card and two (plan 23b)"
rownote 43 "No kernel and no existing file's behaviour moved (kernel_cmdlines +0 / -0 / ~0; the int4 layout formulas moved to loader/int4_layout.h byte-identically): G0's g0.sha is the 'nothing moved' check. Written on the Mac: the host stages ran there; nothing of load_muse_glimmer has touched a card."
rownote 43 "Downloads: r43.synth and r43.pytest need only SMALL files - the original's (hf download meta-models/Muse-Glimmer-30B tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json config.json) and the -GPTQ repo's config.json (hf download Vishva007/Muse-Glimmer-30B-W4A16-AutoRound-GPTQ config.json); r43.load_real needs the int4 checkpoint in either spelling (22.8 GB, row 42's download, data key muse_gptq)."
rownote 43 "Disk: r43.synth writes ~6.4 GB (4 layers at the real widths, measured 5.9 GB at 2 on the Mac) under \$DATA/oracle-out-muse-synth via the tree's oracle-out-muse-synth.partial; RAM: ~12 GB in the oracle container (the embedding's and the head's fp32 draws)."
rownote 43 "Hand back: each load line (layers, spelling, vision skipped, head, devices, bytes) and its seconds (the int8 head's quantisation time separately), every device's weights line against the plan, the real checkpoint's host-half line (tensors, 1459 or 1875 text, 809 vision, 0 unconsumed) and the plan lines (one card at 131072; split by bytes)."
stage r43.k0 43 default cpu - g0.sha,g0.bitwise,g0.suite "MG0: 23b adds no kernel (every binary identical, G0) and leaves the other loaders' files untouched"
st_r43_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r43.host 43 default cpu - - "host: muse_glimmer_test (the descriptor against the vendored config.json key by key, 29 refusals by name, layer_rope_theta's switch rule), muse_layout_test (spec 23 §2's bytes), muse_repack_test (the fused GEMVs' columns at the real widths, the vision skip by name, the data asserts and refusals; 23a's tiny int4 variant / the synthetic when linked), muse_plan_test (131072 on one card with the drafter's reservation, the split by bytes, decision 6)"
st_r43_host() {
  run_tests '^(muse_glimmer_test|muse_layout_test|muse_repack_test|muse_plan_test)$'
  jgrab host '^muse_(glimmer|layout|repack|plan)_test$' 'OK|a layer|one card|read per token|Muse-Glimmer plan|device [01] layers|a 20 GB card|real widths|tiny|make_synth'
  finish
}
stage r43.pytest 43 default cpu oracle_image - "the Python tools' tests in the oracle image with the 5.19.0 site: test_muse_quant.py (make_synth at one real-width layer = muse_ref.expected_names, check.py ACCEPTS it and refuses six corruptions by name, the packer's rule) - muse_oracle.sh synth-tests"
st_r43_pytest() {
  chk "tools/box_validate/muse_oracle.sh $DATA synth-tests" "test_muse_quant.py"
  grab_all pytest '^(PASS|FAIL|SKIP) |passed|refused:|ACCEPTED' 20
  finish
}
stage r43.synth 43 optin cpu oracle_image - "the synthetic real-width checkpoint on the box CPU: muse_oracle.sh synth-ckpt - make_synth.py --layers 4 (the -GPTQ spelling, random weights, 23a's packer), check.py ACCEPTED -> \$DATA/oracle-out-muse-synth/ckpt4 (~6.4 GB); then re-links oracle-out*"
st_r43_synth() {
  chk "tools/box_validate/muse_oracle.sh $DATA synth-ckpt" "the synthetic Muse-Glimmer checkpoint"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_oracle_muse_synth' $STATE/have.env"
  grab_all synth '^ACCEPTED|REFUSED|^wrote|FAILED' 12
  finish
}
stage r43.synth_host 43 default cpu oracle_muse_synth r43.host "the C++ host half over make_synth's own files (Review Focus 5): muse_synth_host_test - the served constructor, names both ways, every layer repacked (bytes = muse_layer_bytes, the part boundaries read back), 0 unconsumed, the plan"
st_r43_synth_host() {
  run_tests '^muse_synth_host_test$'
  jgrab synth-host '^muse_synth_host_test$' 'tensors|OK|plan at|device 0'
  finish
}
stage r43.load 43 default gpu oracle_muse_synth r43.synth_host "load_muse_glimmer on the card over the synthetic checkpoint: one card (muse_load_synth_test), then --pp 2 split 2 over both (muse_load_synth_pp2_test; SKIP without the second card) - 0 unconsumed, every part's bytes = muse_device_weight_bytes, Review Focus 1's columns (every k-row word and scale) read back = the host repack, the norms, the embedding's first / last rows, the int8 head's bytes"
st_r43_load() {
  run_tests '^muse_load_synth(_pp2)?_test$'
  jgrab load '^muse_load_synth' 'Muse-Glimmer-30B:|device [01] |read/token|load  |OK'
  finish
}
stage r43.load_real 43 default gpu muse_gptq r43.load "the real checkpoint (either spelling): muse_real_host_test (the host half: names, the full data asserts, 52 layers repacked, 0 unconsumed) then muse_load_real_test on one card (18.96 GB read, 17.41 GB on the card with the int8 head): its load line and time, 0 unconsumed, the read-backs"
st_r43_load_real() {
  x "free -g | head -2"
  run_tests '^muse_(real_host|load_real)_test$'
  jgrab load-real '^muse_(real_host|load_real)_test$' 'tensors|Muse-Glimmer-30B:|device 0 |checks|read/token|load  |OK'
  finish
}

# ======================================================================================
row 44 "spec 23c - Muse-Glimmer-30B decode: the mg_* kernels (embed with its weight-less norm, the centered norms and the sandwich, q / k prep with the 3.87 scale and RoPE on sliding layers only, flash and eager attention over the 4096-slot ring or the NoPE cache with the output gate), MuseEngine on one card and two, b70-decode's dispatch and refusals, MG2-MG4 on decode, the speed rows (plan 23c)"
rownote 44 "Written blind on the Mac: NO Muse binary has ever been compiled (ocloc is the box's) - r44.k1's kbins is the first build of the 19 new command lines (kernel_cmdlines +19 / -0 / ~0; four of them are gemv.cl / gemv_i8w.cl / gemv_bf16.cl / prep.cl at Muse's shapes, the GEMVs with PROVISIONAL cells copied from Qwen3.8's nearest: S2 / S4 / S8 / S4 L0). mg_attn.cl (flash: shuffles, sub-group reductions, 2D block reads) has never run anywhere; its block / wave / merge arithmetic was held to fp64 on the host (museref::attention_flash_sim, muse_ref_test). On the Mac's GPU every mg_prep kernel and mg_attn_eager ran bitwise the twin (muse_run, indicative)."
rownote 44 "The twin (tests/kernels/muse_ref.h) is bitwise transformers 5.19.0's own modules at the real widths (muse_ref_test against tools/oracle/muse_fixture.py's dump): the embed / centered / final norms, the sandwich, q / k prep at 0 .. 131071, the gate, SiLU x up, eager attention's scores / probabilities / outputs, the RoPE table. MG2 on the card (r44.k1) holds every mg_* kernel bitwise to it (correctly rounded sqrt / divide on the B70) and flash to fp64 at cosine 0.99999."
rownote 44 "Bars that are PROPOSED (set from the first run, printed): the partial forward's per-layer median cosine 0.9998 / min 0.99 (r44.partial). MG3 is the tie-aware token gate (docs/14 + spec 23 §1.1's capped-tie rule: a raw or a capped tie at the decision row is undetermined). The synthetic checkpoint's margins are SMALL (random weights; measured on the Mac's 2-layer synthetic: raw top-1 - top-2 gap ==0 in 5 / 74 rows, median 0.17): a determined-row failure on the int8-head arm there (r44.golden) is read against the bf16 arm before it is called a bug."
rownote 44 "Data: r44.oracle_synth (opt-in, box CPU) writes oracle-out-muse-synth/golden/ from row 43's ckpt4 (muse_oracle.sh synth; ~15 min, ESTIMATED); r44.partial / r44.real / r44.speed need the int4 checkpoint (either Vishva007 spelling, 22.8 GB - row 42's download, data key muse_gptq) and row 42's golden sets (oracle-out-muse, data key oracle_muse; the long prompts are the box CPU's - row 42's r42.golden); r44.oracle_layers (opt-in) adds the truncated reference the partial forward's head is graded against."
rownote 44 "If determined rows fail under flash, the first suspect is flash's fp32 probabilities against the reference's eager bf16 chain: r44.golden_eager runs B70_MUSE_ATTN=eager (681 launches at 52 layers against flash's 733); spec 18 §10.1's rule decides the default. Two cards (r44.pp) need row 22's hand-offs proven first."
rownote 44 "Hand back: the kbins list, muse_kernels_test's lines (bitwise / cosine), muse_decode_test's plan / launches / memory lines, every MG3 gate line and the per-layer tap cosines, muse_pp_test's lines, the partial forward's per-layer table, the speed rows (t/s, ms/token, launches, each card's memory line) and RATIO lines; then 23c's follow-up commit: the {S, layout} sweep's winners into model::MuseGlimmerDesc and kernels::muse together (r44.sweeps)."
stage r44.k0 44 default cpu - g0.sha,g0.bitwise,g0.suite "MG0: no existing .cl edited (kernel_cmdlines +19 / -0 / ~0), every pre-existing binary identical, the suite (G0); 23c's edits to shared files are b70-decode's dispatch (before the Qwen rules), b70-serve's refusal and add_gemv_variant's four new PROVISIONAL rows"
st_r44_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r44.host 44 default cpu - - "host: muse_ref_test (the twin bitwise against transformers 5.19.0's modules; flash's arithmetic against fp64), muse_variant_names_test (every name the capture binds is built, nothing built is unbound), muse_plan_test (733 / 681 launches, the decode scratch, 131072 on one card with it reserved)"
st_r44_host() {
  run_tests '^(muse_ref_test|muse_variant_names_test|muse_plan_test)$'
  jgrab host '^muse_(ref|variant_names|plan)_test$' 'OK|differ|cosine|launches|decode scratch|bound names'
  finish
}
stage r44.k1 44 default gpu - - "MG2, no checkpoint: the 19 new binaries built (kbins: the first ocloc build of every mg_* kernel) and muse_kernels_test - mg_embed (rows 7 / 202047, the placeholders' row 0), mg_norm / mg_final_norm, prep_res_fold _Z G26 + mg_post_add, mg_attn_prep (sliding 37 / 4097, full 0 / 100000: NoPE position-free), mg_silu_mul - all bitwise museref; eager attention bitwise at 5 / 2046 / 2047 / 2048 / 4095 / 4096 / 4097 / 9000 sliding and 3000 / 30000 full; flash against fp64 at cosine 0.99999 (the same rows, full 20000); flash at 9000 == 4904 (the same ring rows); the window's edge EXACT for flash and eager (the slots of p - 2048 and p + 1 .. p + 15 poisoned: bitwise unchanged; p - 2047 poisoned: changed)"
st_r44_k1() {
  kbins mg_embed_M1_K6656_G26_V202048 mg_norm_M1_K6656_G26_W26 mg_post_add_M1_K6656_G26 \
    mg_attn_prep_M1_N8704_S2_Q32KV2_R4096 mg_attn_prep_M1_N8704_S2_Q32KV2_F mg_silu_mul_M1_I19968_S8 \
    mg_attn_M1_T64_Q32KV2_W2048_R4096 mg_attn_M1_T64_Q32KV2_F mg_attn_eager_M1_T64_Q32KV2_W2048_R4096 \
    mg_attn_eager_M1_T64_Q32KV2_F prep_res_fold_M1_K6656_SP4_G26_Z gemv_M1_K6656_N8704_S2_L0 gemv_M1_K4096_N6656_S4_L0 \
    gemv_M1_K6656_N39936_S8_L0 gemv_M1_K19968_N6656_S4_L0 gemv_i8w_M1_K6656_N202048 gemv_bf16_M1_K6656_N202048 \
    mg_argmax_stage1_M1_N202048_V202048 mg_argmax_stage2_N202048
  run_tests '^muse_kernels_test$' muse
  jgrab k1 '^muse_kernels_test$' 'bitwise|cosine|differ|OK'
  finish
}
stage r44.reject 44 default gpu - - "the refusals before the device (b70-decode on tests/model/muse_glimmer: config.json only): cli_reject_muse_* - --mtp, --spec naming 23f, --kv-cache int8 (decision 8), --max-len 262144 (decision 6), --device with --pp 2 (23e removed cli_reject_muse_serve: b70-serve serves Muse, r46.reject; 23d replaced the --prefill refusals: its cli_reject_muse_prefill_* run here too, r45.reject names them). Under the GPU lock (Kolibri's r24.reject): a refusal that regressed would open the device"
st_r44_reject() {
  run_tests '^cli_reject_muse_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r44.oracle_synth 44 optin cpu oracle_image,oracle_muse_synth - "the synthetic golden sets on the box CPU: muse_oracle.sh synth - muse_ref.py run --gen 32 on row 43's ckpt4 over muse_short_prose and muse_4k -> \$DATA/oracle-out-muse-synth/golden (resumable; MemAvailable >= 24 GB; ~15 min ESTIMATED); the cap_tie / gap lines; then re-links oracle-out*"
st_r44_oracle_synth() {
  chk "tools/box_validate/muse_oracle.sh $DATA synth" "the Muse-Glimmer synthetic reference runs"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_oracle_muse_synth' $STATE/have.env"
  grab_all gap 'cap_tie|top-1 - top-2 gap|ppl|^wall' 12
  finish
}
stage r44.k3 44 default gpu oracle_muse_synth r44.k1 "MG4 on one card: muse_decode_synth_test - plan == allocation (weights, KV, rings, the decode scratch), 1 + 4 x 14 + 4 = 61 launches (733 at 52 layers), two runs bitwise (tokens, logits, full KV, rings), the ring at 4200 holds 105..4200 at rows p & 4095, --layers 2 (development mode) bitwise the whole model's first two layers"
st_r44_k3() {
  run_tests '^muse_decode_synth_test$'
  jgrab k3 '^muse_decode_synth_test$' 'plan == allocation|launches|MG4|ring:|--layers 2|memory|OK'
  finish
}
stage r44.golden 44 default gpu oracle_muse_synth_golden r44.k3 "MG3 on the synthetic checkpoint: muse_golden_synth_test (int8 head, flash) and _bf16_test - the tie-aware token gate over muse_short_prose / muse_4k x 32 (raw or capped ties undetermined), the per-layer residual taps against H.L*, the raw logits' cosine"
st_r44_golden() {
  run_tests '^muse_golden_synth(_bf16)?_test$'
  jgrab golden '^muse_golden_synth' 'gate:|tap|step [0-9]+:|OK'
  finish
}
stage r44.golden_eager 44 default gpu oracle_muse_synth_golden r44.k3 "MG3 through the EAGER attention (B70_MUSE_ATTN=eager, 1 + 4 x 13 + 4 = 57 launches): muse_golden_synth_eager_test (bf16 head: the reference's chain end to end) - compare with r44.golden"
st_r44_golden_eager() {
  run_tests '^muse_golden_synth_eager_test$'
  jgrab golden-eager '^muse_golden_synth_eager_test$' 'gate:|tap|step [0-9]+:|OK'
  finish
}
stage r44.pp 44 default gpu oracle_muse_synth r44.k3,r22.p1 "two cards (spec 16b's pieces): muse_pp_test - --pp 1 against --pp 2 --pipeline-split 2 on the synthetic checkpoint under copy and peer: tokens, logits, full KV and rings bitwise, both Control blocks equal; P4 (a lost hand-off throws within 5 s naming device 1, reset recovers); replay; then MG3 across the split (muse_golden_synth_pp2{,_peer}_test, SKIP without the golden set)"
st_r44_pp() {
  run_tests '^(muse_pp_test|muse_golden_synth_pp2(_peer)?_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^muse_(pp|golden_synth_pp2)' 'device|P4|tokens|gate:|OK'
  finish
}
stage r44.cli 44 default gpu oracle_muse_synth r44.pp "CLI: b70-decode <synthetic ckpt4> --ids muse_short_prose.ids --n 32 on one card (the default) against --pp 2 (copy, peer, --pipeline-split 2): the same ids; --max-len auto on one card plans 131072; the plan, launches and memory lines"
st_r44_cli() {
  local ck="oracle-out-muse-synth/ckpt4" ids="tests/golden/prompts/muse_short_prose.ids" h
  chk "build/src/cli/b70-decode $ck --ids $ids --n 32 > $STATE/r44-one.ids" "one card"
  for h in copy peer; do
    chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --pp 2 --ids $ids --n 32 --pipeline-handoff $h > $STATE/r44-$h.ids && cmp $STATE/r44-one.ids $STATE/r44-$h.ids" \
      "--pp 2, $h: the one-card ids"
  done
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $ck --pp 2 --ids $ids --n 32 --pipeline-split 2 > $STATE/r44-s2.ids && cmp $STATE/r44-one.ids $STATE/r44-s2.ids" \
    "--pipeline-split 2: the one-card ids"
  chk "build/src/cli/b70-decode $ck --ids $ids --n 8 --max-len auto > /dev/null" "--max-len auto on one card"
  grab_all plan 'max_len: |split: |Muse-Glimmer plan at|^engine:' 8
  grab_all memory '^memory' 6
  finish
}
stage r44.oracle_layers 44 optin cpu oracle_image,muse_gptq - "the truncated reference for the partial forward's head: muse_oracle.sh real-layers (MUSE_LAYERS=8) - muse_ref.py run --layers 8 on the int4 checkpoint over muse_short_prose -> \$DATA/oracle-out-muse-L8 (MemAvailable >= 40 GB; minutes)"
st_r44_oracle_layers() {
  chk "MUSE_LAYERS=8 tools/box_validate/muse_oracle.sh $DATA real-layers" "the 8-layer reference"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all layers 'cap_tie|top-1 - top-2 gap|^wall' 6
  finish
}
stage r44.partial 44 default gpu muse_gptq,oracle_muse r44.k3 "development mode on the real checkpoint, one card: muse_partial_test / _eager (--layers 8) - every layer's residual tap against 23a's H.L* over the golden prompts (median cosine >= 0.9998, min >= 0.99, PROPOSED - printed per layer); the head against oracle-out-muse-L8 when r44.oracle_layers made it"
st_r44_partial() {
  run_tests '^muse_partial(_eager)?_test$'
  jgrab partial '^muse_partial' 'L[0-9]+ |head|teacher-forced|OK'
  finish
}
stage r44.real 44 default gpu muse_gptq,oracle_muse r44.partial "MG3 on the REAL checkpoint (52 layers, one card: it fits): muse_golden_real_test (int8 head), _bf16, _eager over every golden prompt row 42 made (short prose / code, 4k, 8k, 32k, agentic); then _pp2 (two cards, copy; SKIP with one)"
st_r44_real() {
  x "free -g | head -2"
  run_tests '^muse_golden_real(_bf16|_eager)?_test$'
  run_tests '^muse_golden_real_pp2_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab real '^muse_golden_real' 'gate:|tap|step [0-9]+:|launches|OK'
  finish
}
stage r44.speed 44 optin gpu muse_gptq r44.real "decode speed (Task 7), --max-len 131072, tg 256, interleaved, median of 3: depth 4096 - int8 head (default), bf16 head, eager, --pp 2 copy and peer; depth 32768 - int8, bf16, --pp 2 copy. The prompt is ingested one replay per id until 23d's prefill (~30 ms an id, ESTIMATED: ~3 h for the stage). Launches per token, each card's memory; against spec 23 §2's roofline (39.7 / 38.7 t/s at 590 GB/s; ~30.8 at Qwen3.8's 77.6 %)"
st_r44_speed() {
  idle before
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" k="ZE_AFFINITY_MASK=0,1" arms="" ratios=""
  arms="$arms d4096 '$(bench_cmd "$m" --depth 4096 --tg 256 --max-len 131072)'"
  arms="$arms d4096-bf16 '$(bench_cmd "$m" --depth 4096 --tg 256 --max-len 131072 --lm-head bf16)'"
  arms="$arms d4096-eager 'B70_MUSE_ATTN=eager $(bench_cmd "$m" --depth 4096 --tg 256 --max-len 131072)'"
  arms="$arms d4096-pp2 '$k $(bench_cmd "$m" --pp 2 --depth 4096 --tg 256 --max-len 131072)'"
  arms="$arms d4096-pp2-peer '$k $(bench_cmd "$m" --pp 2 --depth 4096 --tg 256 --max-len 131072 --pipeline-handoff peer)'"
  arms="$arms d32768 '$(bench_cmd "$m" --depth 32768 --tg 256 --max-len 131072)'"
  arms="$arms d32768-bf16 '$(bench_cmd "$m" --depth 32768 --tg 256 --max-len 131072 --lm-head bf16)'"
  arms="$arms d32768-pp2 '$k $(bench_cmd "$m" --pp 2 --depth 32768 --tg 256 --max-len 131072)'"
  ratios="--ratio d4096-bf16/d4096 --ratio d4096-eager/d4096 --ratio d4096-pp2/d4096 --ratio d4096-pp2-peer/d4096-pp2"
  ratios="$ratios --ratio d32768-bf16/d32768 --ratio d32768-pp2/d32768 --ratio d32768/d4096"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 $ratios --$arms" "the Muse-Glimmer decode arms"
  idle after
  grab_all memory '^memory' 8
  grab_all launches 'launches a token' 4
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r44.speed_long 44 optin gpu muse_gptq r44.speed "decode at the full context: one card, int8 head, depth 130816 + tg 256 = 131072 (decision 6's cap; --max-len 131072), ONE run (the ingest is ~130k replays until 23d's prefill: ~70 min ESTIMATED) - t/s against spec 23 §2's 35.7 t/s roofline at 131072"
st_r44_speed_long() {
  idle before
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}"
  chk "$(bench_cmd "$m" --depth 130816 --tg 256 --max-len 131072) | tee $STATE/$STAGE.rows" "depth 130816 + 256"
  idle after
  grab_all row '^\| b70-decode' 2
  grab_all memory '^memory' 2
  finish
}
stage r44.sweeps 44 manual - muse_gptq - "Task 7's sweep: the four int4 GEMVs' {S, layout} (q||gate||k||v 6656x8704, o_proj 4096x6656, gate||up 6656x39936, down 19968x6656: S 1 / 2 / 4 (8), L 0 / 1) - the winners replace the PROVISIONAL S2 / S4 / S8 / S4 L0 in a follow-up commit"
st_r44_sweeps() {
  say "# {S, layout} of 6656x8704, 4096x6656, 6656x39936, 19968x6656 with their GEMV_* defines (probe_gemv rows, spec 14 step 4's method: interleaved pairs after a warm-up)"
  say "# model::MuseGlimmerDesc's qgkv_s / oproj_s / gate_up_s / down_s, kernels::muse::kQgkvS.. and add_gemv_variant's Spec 23c rows move together (muse_variant_names_test holds them)"
  say "# then ZE_AFFINITY_MASK=0 build/src/cli/b70-decode \${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound} --bench --depth 4096 --tg 256 per cell set"
  say "# record: docs/BENCHMARKS.md 'Muse-Glimmer-30B (spec 23)'"
}

# ======================================================================================
row 45 "spec 23d - Muse-Glimmer-30B prefill on l0: the chunk walk (q||gate||k||v / o_proj / gate||up / down as int4 slabs + pf_gemm_T0, mg_prep at M 2048), mg_pf_attn (flash: the 2048-key window over the 4096-slot ring in absolute 64-key tiles, causal NoPE over the full cache; eager: bitwise decode's eager rows), the gate after, MuseEngine::prefill on one card and two, b70-decode --prefill / --prefill-length, MG2-MG4 on prefill, the speed rows (plan 23d)"
rownote 45 "Written blind on the Mac: NO Muse prefill binary has ever been compiled (ocloc is the box's) - r45.k1's kbins is the first build of the 16 new command lines (kernel_cmdlines +16 / -0 / ~0: mg_pf_attn.cl x 4, mg_pf_prep.cl x 2, mg_prep.cl at M 2048 x 5, k2_pf_linear.cl at Muse's four linears, pf_prep.cl's _Z at K 6656). mg_pf_attn.cl's FLASH build (kol_pf_attn.cl's DPAS walk at GQA 16) has never run anywhere; its tile walk is held to fp64 on the host (musepf::flash_window_sim at 0.9999954 worst, muse_pf_ref_test). On the Mac's GPU every portable prefill kernel ran exact against the twin, and the EAGER build equalled decode's mg_attn_eager rows bitwise on the same device (muse_run, indicative)."
rownote 45 "Bars: MG2's flash >= 0.99999 vs fp64 (r45.k1); the KV / rings against decode's fill PROPOSED (Kolibri's: layer 0 every row >= 0.999, median >= 0.9998, p01 >= 0.99 - printed per layer, r45.prefill); MG3 the tie-aware token gate from row T - 1 on (r45.golden / r45.real). Bitwise: replay, chunks of 64 vs 2048, splits at multiples of 64, --pp 2 vs --pp 1; a split at 2049 within the bars (the host twin is bitwise there too: read the printed line). The synthetic checkpoint's margins are small (23c's note): a determined-row failure on the int8-head arm is read against the bf16 arm first."
rownote 45 "Eager prefill (B70_MUSE_ATTN=eager) is the portable three-pass build - bitwise decode's eager rows, NOT fast: a full layer's row at depth d reads d keys three times. Its chunk waits are scaled (runtime::muse::prefill_wait_ms: the configured bound per 2048 keys of the chunk's deepest row); an eager 'did not finish within' on a long prompt (r45.real's 32k / agentic) is read as a bound to raise, not a numerics failure - say which prompt and the bound printed."
rownote 45 "Data: r45.prefill / r45.golden / r45.pp need row 43's ckpt4 (oracle_muse_synth) and row 44's synthetic golden set (r44.oracle_synth); r45.real / r45.speed the int4 checkpoint (muse_gptq) and row 42's golden sets (oracle_muse). The prompts are in the tree (tests/golden/prompts/muse_{short_prose,4k,8k}.ids)."
rownote 45 "Hand back: the kbins list, muse_pf_kernels_test's lines, muse_prefill_test's walk / MG4 / per-layer KV tables / token lines, the split lines (bitwise or not), the pp and P4 lines, every MG3 gate line on prefill, the speed rows (pp t/s, TFLOP/s against 50.3 GFLOP a token, each card's memory line) and RATIO lines."
stage r45.k0 45 default cpu - g0.sha,g0.bitwise,g0.suite "MG0: no existing .cl edited (kernel_cmdlines +16 / -0 / ~0), every pre-existing binary identical - 23c's decode list among them (muse_pf_variant_names_test pins its 17 names) - and the suite (G0); 23d's edits to shared files: b70-decode's Muse refusals (the prefill flags now run), cli/CMakeLists.txt's b70_muse_prefill link"
st_r45_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r45.host 45 default cpu - - "host: muse_pf_ref_test (the window over the ring: 4095 distinct slots a chunk; the fp64 tile walk == the direct softmax; the masked first tile; row independence and splits; flash's fp32 arithmetic vs fp64; the eager build's structure bitwise decode's eager row; the gate), muse_pf_variant_names_test, muse_plan_test (6969 launches a chunk + 4; the prefill scratch 599,171,072 B; 131072 on one card with it and the int4 drafter)"
st_r45_host() {
  run_tests '^(muse_pf_ref_test|muse_pf_variant_names_test|muse_plan_test)$'
  jgrab host '^muse_(pf_ref|pf_variant_names|plan)_test$' '^\(|OK|23d:|bound names'
  finish
}
stage r45.k1 45 default gpu - r45.host "MG2 for the prefill, no checkpoint: the 16 new binaries built (kbins) and muse_pf_kernels_test - the slabs exact (first and last of the four linears), q||gate||k||v through pf_gemm_T0 >= 0.99999 vs fp64, mg_pf_embed / mg_norm / pf_res_fold _Z + mg_post_add / mg_attn_prep (ring wrap at 4090, NoPE at 37) / mg_silu_mul / mg_pf_attn_gate at M 2048 bit-exact; the flash vs fp64 >= 0.99999 sliding at 0 / 1000 / 4396 / 6143 / a 300-row tail / single rows and full to depth 30048; the EAGER build bitwise eager_window; a split at 64 bitwise, at 2049 within the bar (printed), replay bitwise"
st_r45_k1() {
  kbins mg_pf_embed_M2048_K6656_G26_V202048 mg_pf_attn_gate_M2048_N4096 mg_norm_M2048_K6656_G26_W26 \
    mg_post_add_M2048_K6656_G26 mg_attn_prep_M2048_N8704_S1_Q32KV2_R4096 mg_attn_prep_M2048_N8704_S1_Q32KV2_F \
    mg_silu_mul_M2048_I19968_S1 pf_res_fold_K6656_SP1_G26_Z k2_pf_dequant_slab_K6656_N8704 \
    k2_pf_dequant_slab_K4096_N6656 k2_pf_dequant_slab_K6656_N39936 k2_pf_dequant_slab_K19968_N6656 \
    mg_pf_flash_attn_Q32KV2_W2048_R4096 mg_pf_flash_attn_Q32KV2_W2048_R4096_EAGER mg_pf_flash_attn_Q32KV2_F \
    mg_pf_flash_attn_Q32KV2_F_EAGER
  run_tests '^muse_pf_kernels_test$' muse
  jgrab k1-prefill '^muse_pf_kernels_test$' 'cosine|bit-exact|bitwise|splits|exact|OK|FAIL'
  finish
}
stage r45.reject 45 default gpu - - "the refusals before the device: cli_reject_muse_prefill_int8 (naming 6656 = 6.5 x 1024, spec 23 §10), _sycl (no Muse walk), _chunk (above 2048). Under the GPU lock (Kolibri's r24.reject / 23c's r44.reject): a refusal that regressed would open the device"
st_r45_reject() {
  run_tests '^cli_reject_muse_prefill_(int8|sycl|chunk)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r45.prefill 45 default gpu oracle_muse_synth r45.k1 "MG4 / Review Focus 5 on one card: muse_prefill_synth_test, _bf16_test, _eager_test - 1 + 4 x 134 = 537 launches a chunk + 4 (6969 at 52 layers), plan == allocation, immediate / recorded / replayed bitwise, chunks of 64 bitwise (muse_4k), KV / rings / pos / tokens against decode's fill per layer on muse_short_prose / 4k / 8k; prefill_split_muse_test (splits at 64 / 1000 / 2048 / 2049 / 4097 of the 8k prompt: multiples of 64 bitwise, the others within the bars)"
st_r45_prefill() {
  run_tests '^(muse_prefill_synth(_bf16|_eager)?_test|prefill_split_muse_test)$' muse
  jgrab prefill '^(muse_prefill_synth|prefill_split_muse)' 'walk|MG4|chunks of 64|vs decode|per layer|tokens|split at|OK|FAIL'
  finish
}
stage r45.golden 45 default gpu oracle_muse_synth_golden r45.prefill "MG3 on prefill, synthetic: muse_golden_prefill_synth_test (int8 head, flash), _bf16_test, _c64_test (chunks of 64), _eager_test (B70_MUSE_ATTN=eager, both halves) - the tie-aware token gate from row T - 1 on (the prefill head's first id), the decode steps' taps - compare with r44.golden"
st_r45_golden() {
  run_tests '^muse_golden_prefill_synth(_bf16|_c64|_eager)?_test$' muse
  jgrab golden-prefill '^muse_golden_prefill_synth' 'MG3 on prefill|gate:|tap|step [0-9]+:|OK|FAIL'
  finish
}
stage r45.pp 45 default gpu oracle_muse_synth r45.prefill,r44.pp,r23.p1 "two cards: muse_prefill_pp_test and _pp_eager_test (--pp 2 --pipeline-split 2 against --pp 1, BITWISE: one prefill, recorded + replayed, a 1000 + rest continuation, chunks of 64, both Control blocks; each card's launches; P4 - a dropped chunk hand-off throws within 5 s naming device 1, reset() recovers), prefill_split_muse_pp_test, muse_golden_prefill_synth_pp2_test"
st_r45_pp() {
  run_tests '^(muse_prefill_pp(_eager)?_test|prefill_split_muse_pp_test|muse_golden_prefill_synth_pp2_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^(muse_prefill_pp|prefill_split_muse_pp|muse_golden_prefill_synth_pp2)' 'pp:|P4|device|split at|gate:|OK|FAIL'
  finish
}
stage r45.real 45 default gpu muse_gptq,oracle_muse r45.golden,r44.real "MG3 on prefill, the REAL checkpoint (52 layers, one card): muse_golden_prefill_real_test (int8 head), _bf16, _c64, _eager over every golden prompt row 42 made (short prose / code, 4k, 8k, 32k, agentic) - compare with r44.real's decode-ingested lines"
st_r45_real() {
  x "free -g | head -2"
  run_tests '^muse_golden_prefill_real(_bf16|_c64|_eager)?_test$'
  jgrab real-prefill '^muse_golden_prefill_real' 'MG3 on prefill|gate:|tap|step [0-9]+:|OK|FAIL'
  finish
}
stage r45.speed 45 optin gpu muse_gptq r45.real "prefill speed (Task 5): pp 4096 / 16384 / 32768 at chunk 2048 then tg 16, one card (int8 head, --max-len 131072), pp 4096 / 32768 on two cards (--pp 2, the chunk crosses by copy), pp 4096 eager; interleaved, median of 3 - pp t/s and TFLOP/s against 50.3 GFLOP a token (derived), beside Qwen3.8's bf16 l0 walk (1670.72 t/s at 4096, BENCHMARKS '## Prefill')"
st_r45_speed() {
  idle before
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" n k="ZE_AFFINITY_MASK=0,1" arms=""
  for n in 4096 16384 32768; do
    arms="$arms pp$n '$(bench_cmd "$m" --prefill-length $n --tg 16 --max-len 131072)'"
  done
  arms="$arms pp4096-pp2 '$k $(bench_cmd "$m" --pp 2 --prefill-length 4096 --tg 16 --max-len 131072)'"
  arms="$arms pp32768-pp2 '$k $(bench_cmd "$m" --pp 2 --prefill-length 32768 --tg 16 --max-len 131072)'"
  arms="$arms pp4096-eager 'B70_MUSE_ATTN=eager $(bench_cmd "$m" --prefill-length 4096 --tg 16 --max-len 131072)'"
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3 --ratio pp4096-pp2/pp4096 --ratio pp32768-pp2/pp32768 --ratio pp4096-eager/pp4096 --$arms" \
    "the Muse-Glimmer prefill arms"
  idle after
  grab_all prefill '^pp:|^prefill:|  pp: ' 18
  grab_all memory '^memory' 8
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row 46 "spec 23e - Muse-Glimmer-30B served: the original's chat template byte-identical to transformers 5.19.0, the tokenizer equal, the channel / ATEM parser (parse_response on complete turns, vLLM's selection on damaged ones), reasoning_strength, sampling with the host softcap, the prefix cache's snapshots (the rings' last 2047 positions + the full layers' KV), b70-serve <muse> one card by default; MG4 restores, MG5 (A4 on the engine, passkey at 131072), the llama-benchy rows (plan 23e)"
rownote 46 "Written blind on the Mac: no kernel added or changed (kernel_cmdlines +0 / -0 / ~0); nothing of b70-serve <muse> or MuseEngine's snapshot calls has run on a card. Mac-side (measured): template_muse_test (26 renders byte-identical to apply_chat_template, after one minja patch: the tool-call capability probe accepts name=\"argument_needle\"), muse_tokenizer_test (202048 ids, 24 texts, corpus.txt, 26 renders' ids with ONE BOS), toolcall_muse_test (22 parse_response cases, 6 by a documented difference; damaged turns; a 2000-output fuzz split-invariant), muse_server_test, muse_snapshot_test (the layouts on the real shapes; restores at 2048 / 4096 / 2049 / 5000 / 300 == cold; one card <-> two; the server over HTTP)."
rownote 46 "The template: BOTH Vishva007 repos ship the OLDER chat_template.jinja (114f55eb..., 7167 B); b70-serve refuses it naming --chat-template (spec 23 decision 1). Every real-checkpoint stage passes --chat-template tests/tokenizer/muse/chat_template.jinja (the original's, cfc67e5f..., vendored). The synthetic checkpoints copy the original's (make_synth.py): no flag there."
rownote 46 "Data: r46.snapshot / r46.serve / r46.chat need row 43's ckpt4 (oracle_muse_synth) and rows 44 / 45 green on the card (decode, prefill); r46.real / r46.passkey / r46.benchy the int4 checkpoint (muse_gptq); r46.a4 also 23a's A4 references (oracle_muse_a4: row 42's r42.a4, days on the box CPU)."
rownote 46 "Hand back: muse_snapshot_gpu_test's lines (bitwise or DIFFER per check), every startup line (b70-serve: ... Muse-Glimmer-30B ...: max_len, the template's sha prefix, sampling defaults, the rings' positions), the SEEDED lines, golden_server_test --chat's lines, the A4 score table and parse-failure counts, the passkey lines (with the decode speed at depth), the llama-benchy tables."
stage r46.k0 46 default cpu - g0.sha,g0.bitwise,g0.suite "no kernel changed (kernel_cmdlines +0 / -0 / ~0: G0's sha is the check) and the server path unchanged for every other model: template_test / _agnes / _ornith / _k2 / _kolibri, minja_ext_test (one new probe case), toolcall_test / _k2 / _kolibri, k2_server_test, kolibri_server_test, ornith_server_test, qwen4exp_server_test, protocol_test, openai_test, spec_accept_test (filter_probs unchanged with softcap 0), sampling_test, prefix_server_test, lookup_server_test, mtp_server_test"
st_r46_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  run_tests '^(template_test|template_agnes_test|template_ornith_test|template_k2_test|template_kolibri_test|minja_ext_test|toolcall_test|toolcall_k2_test|toolcall_kolibri_test|k2_server_test|kolibri_server_test|ornith_server_test|qwen4exp_server_test|protocol_test|openai_test|spec_accept_test|sampling_test|prefix_server_test|lookup_server_test|mtp_server_test)$' '' 'kv8'
  finish
}
stage r46.host 46 default cpu - - "host: muse_tokenizer_test (the HF cache's tokenizer.json - meta-models/Muse-Glimmer-30B's, or SNAP_MUSE's: the file is identical - 202048 ids, the named tokens, 24 texts, corpus.txt, every render's ids with one BOS), template_muse_test (26 renders byte-identical; strftime_now on a fixed clock), toolcall_muse_test (parse_response on complete turns, damaged turns, the 2000-output fuzz), muse_server_test (kwargs, reasoning_effort, the vision refusal, string arguments, both EOS ids, the host softcap's chi-square, stop_strings, cached_tokens, the template rule), muse_snapshot_test; the toolcall Python tests"
st_r46_host() {
  run_tests '^(muse_tokenizer_test|template_muse_test|toolcall_muse_test|muse_server_test|muse_snapshot_test)$' '' '' "B70_MUSE_TOKENIZER_JSON=\$(tools/box_validate/data.sh resolve ${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound})/tokenizer.json"
  jgrab host-serve '^(muse_tokenizer|template_muse|toolcall_muse|muse_server|muse_snapshot)_test$' 'OK|SKIP|renders|fixture|fuzz|layout|restore at|server:|sampling:'
  chk "python3 tools/toolcall/test_score.py && python3 tools/toolcall/test_make_set.py" "the toolcall Python tests (atem scoring, the Muse set)"
  finish
}
stage r46.reject 46 default gpu - - "b70-serve's Muse refusals before the device (tests/model/muse_glimmer: config.json, generation_config.json, the older chat_template.jinja): cli_reject_serve_muse_{mtp,spec_mtp,spec,lookup,kv8,int8pf,sycl,maxlen,device,template}, cli_reject_serve_chat_template_other, cli_reject_serve_spec_dflash_other. Under the GPU lock (23c's r44.reject): a refusal that regressed would open the device"
st_r46_reject() {
  run_tests '^cli_reject_serve_(muse_.*|chat_template_other|spec_dflash_other)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r46.snapshot 46 default gpu oracle_muse_synth r45.prefill "MG4 on the card, the synthetic ckpt4: muse_snapshot_gpu_synth_test (one card) and _synth_pp_test (--pp 2, split 2) - the sizes; a restore at the block end 4096 from the hook's own saves BITWISE the cold run (32 ids, logits, full KV, the rings from 2049); restores at the request ends 2049 and 300 bitwise the session that never left; _cross_test: a snapshot taken on two cards continues on one, and the reverse"
st_r46_snapshot() {
  run_tests '^muse_snapshot_gpu_synth(_pp)?_test$|^muse_snapshot_gpu_cross_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^muse_snapshot_gpu' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED|SKIP'
  finish
}
stage r46.serve 46 default gpu oracle_muse_synth r46.snapshot "b70-serve <synthetic ckpt4> startup and requests: ONE card by default (the startup line: Muse-Glimmer-30B, muse chat format, the template's sha cfc67e5f - the original's, prefill l0 chunk 2048, mtp none, lm_head int8, kv cache bf16, sampling defaults sampled T 1.00 top-k 64 top-p 0.95 softcap 20, the rings' last 2047 positions), --max-len auto (the plan's line); --pp 2 (copy) and --pipeline-handoff peer; a seeded sampled request twice, identical"
st_r46_serve() {
  local ck="oracle-out-muse-synth/ckpt4"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck"
  step_rc $? "b70-serve <synthetic>, one card (default), --max-len auto, seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck --pp 2 --max-len 16384"
  step_rc $? "b70-serve <synthetic> --pp 2 (copy), seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pp 2 --pipeline-handoff peer --max-len 16384"
  step_rc $? "b70-serve <synthetic> --pp 2 --pipeline-handoff peer"
  grab_all plan 'Muse-Glimmer plan at|^memory' 8
  grab_all startup '^b70-serve: .*Muse-Glimmer-30B' 3
  grab_all seeded '^SEEDED ' 4
  finish
}
stage r46.chat 46 default gpu oracle_muse_synth r46.serve "plan 23e Task 5 Step 2: the server's greedy chat = b70-decode --prefill --lm-head int8 on its prompt_token_ids - golden_server_test --chat on the synthetic (prose / code / cjk as one user message through the original's template, 32 ids, temperature 0) with B70_SERVE_ARGS '--max-len 16384' (the prefix cache on: auto), '... --prefix-cache-gb 0' (off: the same ids) and '... --pp 2' (two cards vs b70-decode's one: bitwise by 23c / 23d's --pp rule); run directly (not registered: without a checkpoint it would fail, not SKIP)"
st_r46_chat() {
  local ck="oracle-out-muse-synth/ckpt4" a
  for a in "--max-len 16384" "--max-len 16384 --prefix-cache-gb 0" "--max-len 16384 --pp 2"; do
    chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $ck --chat" \
      "Muse synthetic, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 9
  finish
}
stage r46.real 46 default gpu muse_gptq r46.chat,r45.real "the REAL checkpoint, one card: muse_snapshot_gpu_test (MG4 on 52 layers); b70-serve <int4> --chat-template <the original's> startup (--max-len auto -> 131072 with the int8 head and the prefill scratch: the plan's line) with a seeded request twice; the template refusal without the flag (exit 1 naming --chat-template, before the device); golden_server_test --chat (the server's greedy chat = b70-decode on its prompt_token_ids)"
st_r46_real() {
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" t="tests/tokenizer/muse/chat_template.jinja"
  x "free -g | head -2"
  run_tests '^muse_snapshot_gpu_test$' '' '' 'ZE_AFFINITY_MASK=0'
  xfail "timeout 900 build/src/cli/b70-serve $m --port $PORT" 'pass --chat-template FILE' \
    "b70-serve <int4> without --chat-template: refused naming the flag (the repo's older template)"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $m --chat-template $t"
  step_rc $? "b70-serve <muse int4> --chat-template <original>, one card, auto max_len, seeded sampling"
  chk "ZE_AFFINITY_MASK=0 B70_SERVE_ARGS='--max-len 32768 --chat-template $t' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $m --chat" \
    "golden_server_test --chat against Muse-Glimmer-30B"
  grab_all plan 'Muse-Glimmer plan at|^memory' 6
  grab_all startup '^b70-serve: .*Muse-Glimmer-30B' 2
  grab_all chat 'chat of [0-9]+ prompt ids' 3
  finish
}
stage r46.a4 46 optin gpu muse_gptq,oracle_muse_a4 r46.real "MG5: A4 on the engine - the 36 Muse-rendered scenarios (tests/golden/toolcall-muse, reasoning_strength low), 512 greedy ids, one card, l0, the bf16 and the int8 head (engine_generate.sh: b70-decode on the set's ids, byte-identical to the reference's), against 23a's bf16 reference (oracle-out-muse-a4); score.py --format atem bf16 gptq l0 l0-i8head: the bar - no engine run has more parse failures than the bf16 reference's; first-call agreement recorded (gptq = gate Q's int4 CPU reference, for comparison)"
st_r46_a4() {
  local d=$STATE/a4-muse m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}"
  x "mkdir -p $d && cp oracle-out-muse-a4/*.bf16.txt oracle-out-muse-a4/*.gptq.txt $d/"
  chk "ZE_AFFINITY_MASK=0 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $m) tests/golden/toolcall-muse $d l0" "A4 engine runs, bf16 head"
  chk "ZE_AFFINITY_MASK=0 LM_HEAD=int8 tools/toolcall/engine_generate.sh \$(tools/box_validate/data.sh resolve $m) tests/golden/toolcall-muse $d l0" "A4 engine runs, int8 head (the served default)"
  chk "python3 tools/toolcall/score.py --format atem $d bf16 gptq l0 l0-i8head | tee $d/score.md" "scoring against the bf16 reference"
  x "awk '/^bf16: [0-9]+ parse failure/ {b = \$2} /^l0(-i8head)?: [0-9]+ parse failure/ {n++; if (\$2 > e) e = \$2} END {exit !(b != \"\" && n == 2 && e <= b)}' $d/score.md"
  step_rc $? "MG5: no engine run has more parse failures than the bf16 reference's"
  grab_all a4 'match|parse failure' 8
  finish
}
stage r46.passkey 46 optin gpu muse_gptq,oracle_image r46.real "MG5: passkey 3 / 3 at 5 / 50 / 95 % of 131072 on one card with the int8 head (tools/probe/muse_passkey.sh: BOS + 130000 ids, --max-len 131072, 8 greedy ids); the decode speed at that depth recorded (~1.74 GB of full-layer KV a token there, derived)"
st_r46_passkey() {
  chk "ZE_AFFINITY_MASK=0 MODEL=${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound} tools/probe/muse_passkey.sh int8" "passkey at 131072, one card, int8 head"
  grab_all passkey '^passkey muse [^ ]+( placement [0-9.]+)?: ' 8
  finish
}
stage r46.benchy 46 optin gpu-self muse_gptq,uvx - "plan 23e Task 6's speed rows through b70-serve <muse> --chat-template <the original's> on one card (SERVE_AFFINITY=0): pp4096 tg256 depth 1 at 131072 (the operator's flags: --no-cache --exact-tg --latency-mode generation); prefix caching at depth 4k / 16k / 32k; decode at depth 4k / 32k / 130000 (~128k: the request must fit 131072) - every row with the card, the int8 head and bf16 KV"
st_r46_benchy() {
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" t="--chat-template tests/tokenizer/muse/chat_template.jinja"
  local f="--concurrency 1 --no-cache --exact-tg --latency-mode generation"
  chk "MODEL=$m MAX_LEN=131072 SERVE_AFFINITY=0 SERVE_ARGS='$t' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "one card, pp4096 tg256"
  chk "MODEL=$m MAX_LEN=65536 SERVE_AFFINITY=0 SERVE_ARGS='$t' tools/probe/serve_benchy.sh --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching --exact-tg --latency-mode generation --runs 3" "prefix cache on, depth 4k / 16k / 32k"
  chk "MODEL=$m MAX_LEN=131072 SERVE_AFFINITY=0 SERVE_ARGS='$t' tools/probe/serve_benchy.sh --pp 512 --tg 128 --depth 4096 32768 130000 $f" "decode at depth 4k / 32k / ~128k (130000: depth + 512 + 128 fit 131072)"
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 47 "spec 23f - Muse-Glimmer-30B's DFlash drafter (meta-models/Muse-Glimmer-30B-assistant): P0 (teacher-forced acceptance K 1..15, the int4 g64 / int8 arms, raw vs capped q, prompt lookup, the verify cost at M = 1..16, the projection and its stopping rule) and the engine - the drafter on the card in int4 g64 / int8 / bf16, df_* at Muse's shapes + mg_dflash.cl, MuseDrafter, the verify lists at M = 1..16 with the taps, the commit's projection, the prefill tail, the ring in snapshots, --spec dflash / lookup in b70-serve and b70-decode; D0-D5 (plan 23f)"
rownote 47 "Written blind on the Mac (plan 23f, ONE branch for both parts - operator 2026-10-15: Part 2 built regardless of P0, P0 picks the defaults). New binaries: the verify lists' (B70_MUSE_VERIFY_M16: every decode kernel at M 2..16, mg_pf_embed and mg_vargmax_stage1 at M 1..16, mg_vargmax_stage2) and the drafter's (B70_MUSE_DFLASH: df_prep / df_attn / df_topk at Muse's defines, mg_dflash_M16_H6656_K16_E5, the drafter GEMVs at M 16 in int8 / bf16 / int4 layout 1) - nothing binds them without --spec dflash / lookup on Muse: D0 is G0. Mac-side (measured): dflash_ref_test's Muse form (the twin vs dflash_ref.py: rows cos 1.0000000, ids / order exact; norm2 = transformers' bf16 _m_norm on 256 / 256), tools/mac/clrun/muse_dflash_run (the draft and projection lists at the fixture's shapes: rows / logits cos 1.0000000, 15 ids / vids equal, the attention at the window's edge exact; the int4 arm NOT RUN - Apple's OpenCL has no as_half), muse_spec_test (D2 greedy == plain at K 15 / 1..15 / truncated runs, lookup; D4 chi-square >= 0.01 over 10^6 for both q forms and lookup), muse_dflash_names_test, muse_verify_names_test."
rownote 47 "P0 on the Mac (2 short prompts, docs/probe-muse-dflash-2026-10-16.md, provisional): code E_15 9.33 (bf16) = 9.33 (int4 g64 + int8 head) vs 9.78 int8; prose saturates ~1.8; sampled q raw E_15 5.58 vs capped 1.35 -> the default is RAW (reverses decision 11's recommendation); lookup 1.04-1.09x; projected int4 code 1.91x plain at K 8 with the DERIVED costs. Verdict UNDECIDED - the agentic corpora (r47.taps / r47.accept) and the measured verify cost (r47.cost) decide. Provisional defaults: --draft-format int4, --spec-max 15 under AdaptiveK, --draft-q raw."
rownote 47 "Deviations to read the results by: the drafter's ring is 2112 slots (df_attn's WIN 2049 for |q - k| <= 2048; RING >= WIN; PB 64); the draft list always runs the whole 16-row block (dctl.k 15 - transformers' form), K only bounds what is verified; df_prep's q / k norms round once in fp32 (transformers' bf16 rounds twice) - inside D1's bars; the prefix snapshot carries the projected ring (43.3 MB) not spec 23 §8.4's fc rows; --spec dflash / lookup run on ONE card (--pp 2 refused); sampled drafts are drawn on the host from the readback's q (lossless). mg_attn at M = 16 is compiled as-is (NPR 256: register spills possible - r47.cost / r47.benchy show it; the lever is chunking rows)."
rownote 47 "Data: the drafter (muse_dflash: hf download meta-models/Muse-Glimmer-30B-assistant, 5.11 GB), the int4 checkpoint (muse_gptq), 23a's golden sets with taps (oracle_muse) for r47.taps, then oracle_muse_taps for r47.accept / r47.d1; row 43's ckpt4 (oracle_muse_synth) for r47.d3. RAM: r47.accept ~20-24 GiB peak (the drafter fp32 10.2 GB + head + embedding, ESTIMATED), r47.d1_ref likewise. Hand back: r47.cost's table (verify M = 1..16 at 4096 / 32768, the drafter block per format), r47.accept's summary.md + lookup.txt, r47.verdict's project output, muse_dflash_test's lines, the D1 tables (d1.md), muse_verify_*_test's lines (every M bitwise or DIFFER, the drafter lines, the prefill tail), r47.d2's GREEDY lines and dflash: acceptance lines, r47.serve's IDENTICAL / RATIO / SEEDED lines, the llama-benchy tables."
rownote 47 "D5's rule (spec 19 D5's form): --spec dflash becomes Muse's default only if it beats both --spec off and --spec lookup by >= 10 % on the agentic set and loses nowhere by more than 3 %; otherwise it ships as an option. Recorded in spec 23 '23f as built' by the operator."
stage r47.k0 47 default cpu - g0.sha,g0.bitwise,g0.suite "D0: no existing command line moved (only additions), every pre-existing binary identical, the suite (G0) - incl. 23f's edits to shared files: dflash_capture.cc (the Muse shape; Qwen3.8's walk unchanged), tap_drafter.h (TapSink split out of TapDrafter), MuseBuffers / muse_capture (rows, build_verify), MuseEngine (a defaulted MuseSpec), muse_prefill (defaulted taps), dflash_d1 (a Muse branch), b70-serve / b70-decode (Muse's --spec), cli::dflash::check_flags (int4 refused for Qwen3.8)"
st_r47_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r47.host 47 default cpu - - "host: dflash_ref_test (the twin's Muse form vs dflash_ref.py), dflash_loader_test, muse_assistant_test (the config, int4 g64 RTN vs the oracle's fixture, the plan with the drafter), muse_dflash_names_test, muse_verify_names_test, muse_spec_test (D2 / D4 over a fake engine), muse_snapshot_test, adaptive_k_test; in the oracle image test_dflash_ref.py (Muse form bitwise transformers), test_muse_taps.py, test_dflash_accept.py, test_lookup_accept.py, test_muse_p0_project.py, test_dflash_d1.py"
st_r47_host() {
  run_tests '^(dflash_ref_test|dflash_loader_test|muse_assistant_test|muse_dflash_names_test|muse_verify_names_test|muse_spec_test|muse_snapshot_test|adaptive_k_test)$'
  jgrab host '^(dflash_ref|muse_assistant|muse_dflash_names|muse_spec)_test$' 'muse|PASS|p-value|greedy|cos'
  chk "tools/oracle/run_in_container.sh 'python3 tools/oracle/test_dflash_ref.py && python3 tools/oracle/test_muse_taps.py && python3 tools/oracle/test_dflash_accept.py && python3 tools/spec/test_lookup_accept.py && python3 tools/spec/test_muse_p0_project.py && python3 tools/oracle/test_dflash_d1.py'" \
    "the reference's, the dumps', the acceptance's, lookup's, the projection's and the D1 comparator's tests (oracle image)"
  finish
}
stage r47.reject 47 default gpu - - "the refusals before the device: b70-decode cli_reject_muse_{spec,spec_bench,spec_lookup,spec_max,draft_q}, b70-serve cli_reject_serve_muse_{spec,lookup,spec_max,draft_vocab,draft_q}, cli_reject_serve_dflash_{format,int4}, cli_reject_serve_draft_q_other. Under the GPU lock (a refusal that regressed would open the device)"
st_r47_reject() {
  run_tests '^cli_reject_(muse_(spec|spec_bench|spec_lookup|spec_max|draft_q)|serve_muse_(spec|lookup|spec_max|draft_vocab|draft_q)|serve_dflash_(format|int4)|serve_draft_q_other)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r47.taps 47 optin cpu muse_gptq,oracle_muse,oracle_image - "P0 Task 2 (box CPU, hours): muse_oracle.sh taps - the code / prose golden runs at MUSE_TAPS_GEN 256 converted to dumps, the agentic prompt at 512, A4 teacher-forced when r42.a4's references exist -> \$DATA/oracle-out-muse-taps/dumps (resumable); then re-links oracle-out*"
st_r47_taps() {
  chk "tools/box_validate/muse_oracle.sh $DATA taps" "the Muse-Glimmer tap dumps"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "ls -la oracle-out-muse-taps/dumps/ | tail -20"
  finish
}
stage r47.accept 47 optin cpu muse_dflash,muse_gptq,oracle_muse_taps,oracle_image - "P0 Task 3 (box CPU): MODEL=muse tools/oracle/dflash_p0.sh - dflash_accept.py run --model muse over every dump (arms bf16, int8, int8h, int4g64, int4g64h; K 1..15 from one 16-row block an anchor; the sampled estimate for q raw and capped; incoai's DFlash2 drafter only if already in the cache - never downloaded) and lookup_accept.py --dumps --max-k 15 -> oracle-out-muse-p0/{accept/summary.md,lookup.txt}"
st_r47_accept() {
  x "free -g | head -2"
  chk "MODEL=muse ARMS=bf16,int8,int8h,int4g64,int4g64h,dflash2 tools/oracle/dflash_p0.sh" "the Muse-Glimmer acceptance and the lookup baseline"
  x "cat oracle-out-muse-p0/accept/summary.md oracle-out-muse-p0/lookup.txt 2>/dev/null | head -120"
  grab_all accept '^\| |E_15|lookup' 80
  finish
}
stage r47.cost 47 default gpu - - "P0 Task 4: probe_draft_cost --shapes muse at depth 4096 and 32768 - the verify at M = 1..16 (52 layers x 4 int4 GEMVs, the int8 head, the attention estimate) and one drafter block per format (int4 g64 layout 1 / int8 / bf16), interleaved pairs against M = 1 after a warm-up, median of 3 (random weights of the real shapes: no data)"
st_r47_cost() {
  idle before
  chk "build/tools/probe/probe_draft_cost --shapes muse --mmax 16 --depth 4096,32768 --calls 10 --rounds 3 | tee $STATE/$STAGE.txt" "the Muse verify and draft costs"
  idle after
  grab_all cost '^\||^probe_draft_cost|verify|draft' 80
  finish
}
stage r47.verdict 47 manual - - - "P0 Task 5: the projection and the verdict from the box's numbers - tools/spec/muse_p0_project.py over r47.accept's tables, r47.cost's probe output and lookup; record it in spec 23 §8.5 (go / defaults: the drafter format by decision 9 - int4 g64 within 0.05 of bf16's acceptance -, K_max, the q form)"
st_r47_verdict() {
  say "python3 tools/spec/muse_p0_project.py --accept oracle-out-muse-p0/accept --lookup oracle-out-muse-p0/lookup.json --probe \$STATE/r47.cost.txt"
  say "# then edit docs/superpowers/specs/2026-10-15-spec23-muse-glimmer-design.md §8.5 with the verdict and the defaults"
  finish
}
stage r47.k1 47 default gpu - - "the new binaries built (kbins: the first ocloc build of the drafter's df_* at Muse's defines, mg_dflash, the drafter GEMVs at M 16 in three formats; the verify lists' at M 2..16 are kbins'd by name below) and the names tests on the box build"
st_r47_k1() {
  kbins df_prep_M16_H6656_I19968_Q32KV8_D128_R2112_V202048_E5 df_attn_M16_Q32KV8_D128_R2112_W2049_P64 \
    df_topk_R15_N202048_U202048 mg_dflash_M16_H6656_K16_E5 mg_vargmax_stage2_N202048 \
    mg_vargmax_stage1_M16_N202048 mg_pf_embed_M16_K6656_G26_V202048 mg_attn_M16_T64_Q32KV2_W2048_R4096 \
    mg_attn_M16_T64_Q32KV2_F gemv_M16_K6656_N8704_S2_L0 gemv_i8w_M15_K6656_N202048 gemv_i8w_M16_K6656_N6144 \
    gemv_M16_K6656_N6144_S1_L1 gemv_bf16_M16_K6656_N6144 gemv_M16_K33280_N6656_S1_L1 gemv_M16_K19968_N6656_S1_L1
  run_tests '^(muse_dflash_names_test|muse_verify_names_test)$'
  finish
}
stage r47.d1 47 default gpu muse_dflash,muse_gptq,oracle_muse_taps r47.k1 "D1, the engine's half: muse_dflash_test (the REAL assistant beside a synthetic target vs the twin: 66 / 8 launches, 2200 positions projected across the 2112-slot ring, a draft at p 2200 - rows / logits cosine >= 0.99999, the 15 ids, vids; the replay bitwise, set_draft, the capped q, the ring's snapshot round trip, the bf16 head; int4 / int8 / bf16), then MODEL=muse dflash_d1.sh gpu over oracle-out-muse-taps (16 anchors a dump, K 15, the arms i4 / i8 / bf16 with the int8 head) -> \$STATE/d1/d1.<arm>.jsonl"
st_r47_d1() {
  x "build/tests/muse_dflash_test ${SNAP_MUSE_DFLASH:-meta-models/Muse-Glimmer-30B-assistant} | tee $STATE/$STAGE.txt"
  step_rc $? "muse_dflash_test (int4 / int8 / bf16)"
  chk "MODEL=muse tools/box_validate/dflash_d1.sh gpu $STATE/d1" "dflash_d1 (Muse) x 3 arms"
  grab_all d1 '^== drafter|^launches|^projection|^greedy|^replay|^set_draft|^capped|^ring snapshot|^bf16 head|^muse_dflash_test|^dflash_d1' 40
  finish
}
stage r47.d1_ref 47 default cpu muse_dflash,muse_gptq,oracle_muse_taps,oracle_image r47.d1 "D1, the reference's half (box CPU, the oracle image): MODEL=muse dflash_d1.sh ref - dflash_ref.py's Muse form (bitwise transformers 5.19.0) in float32 on the same taps, anchors and head form (int8 / int4 g64 RTN simulated where the engine quantised); the bars: top-16 sets tie-aware, the raw logits' cosine >= 0.9999 (bf16) / 0.999 (int8, int4), the 15 greedy ids equal -> \$STATE/d1/d1.md"
st_r47_d1_ref() {
  x "free -g | head -2"
  chk "MODEL=muse tools/box_validate/dflash_d1.sh ref $STATE/d1" "dflash_d1.py (Muse) x 3 arms: D1"
  grab_all d1-ref '^\| |^D1 |^  FAIL' 40
  finish
}
stage r47.d3 47 default gpu oracle_muse_synth,muse_gptq r47.k1 "D3: muse_verify_synth_test (ckpt4) and muse_verify_real_test (the int4 checkpoint) - the verify lists at M = 1..16 bitwise the M = 1 decode steps (logits rows, ids, the full / sliding layers' KV rows) at P 64 and 2100 (across the window and the ring); with a recording TapSink: plain steps through verify(0) bitwise, an M = 16 verify's taps bitwise the M = 1 steps', commit(5) projects P .. P + 5 only, the prefill tail covers [P - 2048, P) 16 rows a replay (taps cosine >= 0.999 vs decode's)"
st_r47_d3() {
  run_tests '^muse_verify_(synth|real)_test$' dflash '' 'ZE_AFFINITY_MASK=0'
  jgrab d3 '^muse_verify_(synth|real)_test$' 'M +[0-9]+ at P|drafter|verify M 16|prefill tail|PASS|FAIL|SKIP' 120
  finish
}
stage r47.d2 47 default gpu muse_dflash,muse_gptq r47.d3 "D2 without the server: b70-decode <muse int4> --ids (23a's muse_short_code / muse_short_prose / muse_4k / muse_agentic prompts) --prefill --n 256 plain / --spec dflash (int4, K 15) / --spec dflash --spec-max 4 / --draft-format bf16 / --draft-format int8 - the ids identical; the acceptance lines"
st_r47_d2() {
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" p arm f a dm="--draft-model ${SNAP_MUSE_DFLASH:-meta-models/Muse-Glimmer-30B-assistant}"
  for p in short_code short_prose 4k agentic; do
    f=tests/golden/prompts/muse_$p.ids
    for arm in off k15 k4 bf16 int8; do
      case "$arm" in
        off) a="" ;; k15) a="--spec dflash $dm" ;; k4) a="--spec dflash --spec-max 4 $dm" ;;
        bf16) a="--spec dflash --draft-format bf16 $dm" ;; int8) a="--spec dflash --draft-format int8 $dm" ;;
      esac
      chk "ZE_AFFINITY_MASK=0 build/src/cli/b70-decode $m --ids $f --n 256 --prefill --max-len 16384 $a > $STATE/$STAGE.$p.$arm.out" "b70-decode $p $arm"
    done
    chk "for a in k15 k4 bf16 int8; do cmp $STATE/$STAGE.$p.off.out $STATE/$STAGE.$p.\$a.out || exit 1; done && echo 'GREEDY identical: $p plain / dflash int4 K 15 / K 4 / bf16 / int8'" "greedy ids identical with and without the drafter ($p)"
  done
  grab_all dflash '^generate|^drafter:|GREEDY' 40
  finish
}
stage r47.serve 47 default gpu muse_dflash,muse_gptq r47.d2 "D2 / D4 by b70-serve <muse int4> --chat-template <the original's>: the 36 Muse-rendered A4 tool-call scenarios (tests/golden/toolcall-muse) through --spec off / --spec dflash (AdaptiveK to 15) / --spec lookup, greedy, max_tokens 192 - identical to --spec off; a seeded sampled request twice (--spec dflash, raw q) and with --draft-q capped; the startup and per-request 'dflash:' lines"
st_r47_serve() {
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" t="--chat-template tests/tokenizer/muse/chat_template.jinja"
  local dm="--draft-model ${SNAP_MUSE_DFLASH:-meta-models/Muse-Glimmer-30B-assistant}"
  serve_arms "$STATE/$STAGE" 1 "$m" "--set tests/golden/toolcall-muse --max-tokens 192 --stop-at-eos" \
    "off|--max-len 32768 $t --spec off" "dflash|--max-len 32768 $t --spec dflash $dm" "lookup|--max-len 32768 $t --spec lookup"
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output through --spec dflash / lookup identical to --spec off"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref off --metric wall_tps"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $m --max-len 32768 $t --spec dflash $dm"
  step_rc $? "b70-serve --spec dflash (raw q): serves; the seeded sampled request twice"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $m --max-len 32768 $t --spec dflash --draft-q capped $dm"
  step_rc $? "b70-serve --spec dflash --draft-q capped: the seeded sampled request twice"
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all ratio '^RATIO ' 10
  grab_all dflash 'drafter:|dflash:|lookup:|SEEDED|memory' 20
  finish
}
stage r47.benchy 47 optin gpu-self muse_dflash,muse_gptq,uvx r47.serve "D5 (spec 19 D5's rule): llama-benchy through b70-serve <muse int4> on one card at the operator's flags (pp4096 tg256 depth 1, --no-cache --exact-tg --latency-mode generation) and on the agentic session, --spec off / --spec lookup / --spec dflash (auto K) - dflash becomes Muse's default only if it beats both by >= 10 % on the agentic set and loses nowhere by more than 3 %"
st_r47_benchy() {
  local m="${SNAP_MUSE:-Vishva007/Muse-Glimmer-30B-W4A16-AutoRound}" t="--chat-template tests/tokenizer/muse/chat_template.jinja"
  local dm="--draft-model ${SNAP_MUSE_DFLASH:-meta-models/Muse-Glimmer-30B-assistant}" f="--concurrency 1 --no-cache --exact-tg --latency-mode generation" arm a
  for arm in off lookup dflash; do
    case "$arm" in off) a="--spec off" ;; lookup) a="--spec lookup" ;; dflash) a="--spec dflash $dm" ;; esac
    chk "MODEL=$m MAX_LEN=65536 SERVE_AFFINITY=0 SERVE_ARGS='$t $a' tools/probe/serve_benchy.sh --pp 4096 --tg 256 --depth 1 $f" "pp4096 tg256, $arm"
    chk "MODEL=$m MAX_LEN=65536 SERVE_AFFINITY=0 SERVE_ARGS='$t $a' tools/probe/serve_benchy.sh --pp 1024 --tg 256 --depth 4096 16384 $f" "decode at depth 4k / 16k, $arm"
  done
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 48 "spec 24a - LFM2.5-24B-A2B (lfm2_moe): the facts, the CPU reference (F1), golden sets, the A4 reference (plan 24a)"
rownote 48 "Download first: hf download LiquidAI/LFM2.5-24B-A2B (47.96 GB; df -h ~ first; MemAvailable >= 64 GB for the real-weight stages)."
rownote 48 "Box CPU only, no card: nothing here takes the GPU lock. Every stage runs tools/box_validate/lfm2moe_oracle.sh in the oracle image with transformers 5.19.0 from <tree>/oracle-out-q4exp-site (tools/oracle/qwen4exp_env.sh; the image is never changed). r48.host needs the tiny model (hf download yujiepan/lfm2-moe-tiny-random, 10 MB); the real stages SKIP 77 'missing data' without the bf16 original, or while it is still downloading."
rownote 48 "B70_LFM_TIE_TOL (24c / 24d's routing tolerance) is read from the 'MoE 4th/5th sel gap' lines r48.golden prints (per layer and over all layers); hand them back with each prompt's wall / RSS line, the perplexities, hfcheck's verdict and the A4 reference's parse-failure count (score.py --format pythonic)."
stage r48.host     48 default cpu  oracle_image                 -  "lfm2moe_oracle.sh tests: test_lfm2moe_ref.py, test_lfm2moe_make_tiny.py, the toolcall tests (tiny models: hf download yujiepan/lfm2-moe-tiny-random)"
st_r48_host() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA tests" "the lfm2_moe reference tests (tiny models)"
  grab_all tests '^ok |^  |FAIL|SKIP|passed' 120
  finish
}
stage r48.ppl      48 optin   cpu  lfm2moe_bf16,oracle_image    -  "perplexity on prose / code / agentic (finite; code < 10 - the Mac measured 2.54; prose / agentic recorded: prompt-side text scores badly on this RL-tuned model)"
st_r48_ppl() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA ppl" "the reference's perplexity on the bf16 original"
  grab_all ppl '^ppl |^wall|not sane|not finite' 8
  finish
}
stage r48.hfcheck  48 optin   cpu  lfm2moe_bf16,oracle_image    -  "hfcheck --layers 6: transformers' own model vs the port on real weights, BITWISE"
st_r48_hfcheck() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA hfcheck" "hfcheck on the bf16 original"
  grab_all hfcheck '^hfcheck|max \|diff\|' 12
  finish
}
stage r48.golden   48 optin   cpu  lfm2moe_bf16,oracle_image    -  "golden sets on the bf16 original -> oracle-out-lfm2moe (gap lines per layer)"
st_r48_golden() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA golden" "the lfm2_moe golden sets on the bf16 original"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all gap 'sel gap|wall' 200
  finish
}
stage r48.a4_ref   48 optin   cpu  lfm2moe_bf16,oracle_image    -  "the A4 reference on the bf16 original -> oracle-out-lfm2moe-a4 (days, ESTIMATED; resumable)"
st_r48_a4_ref() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA a4" "the A4 reference (36 scenarios, 2048 new ids, batched)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "python3 tools/toolcall/score.py --format pythonic $DATA/oracle-out-lfm2moe-a4 bf16 bf16"
  grab_all a4 'scenarios|parse failure|^bf16:' 12
  finish
}

# ======================================================================================
row 50 "spec 24b - LFM2.5-24B-A2B: the descriptor, the loader (both expert spellings, the tied int8 head), the memory plan (plan 24b)"
rownote 50 "No kernel and no existing loader touched (kernel_cmdlines +0 / -0 / ~0; the int4 layout-1 formula stays loader/int4_layout.h's): G0's g0.sha is the 'nothing moved' check. Written on the Mac: lfm2moe_test, lfm2moe_repack_test, lfm2moe_rope_test (bitwise transformers 5.19.0's Lfm2MoeRotaryEmbedding), lfm2moe_plan_test and lfm2moe_synth_host_{w,proj}_test all ran there, over two synthetic real-width checkpoints (make_synth.py --layers 4) generated on the Mac itself; nothing of load_lfm2moe has touched a card."
rownote 50 "24q (spec 24 §0's stage) is not built yet: r50.load_real SKIPs 77 'missing data' until an RTN or AutoRound export exists (data key lfm2moe_rtn) - r50.host, r50.pytest and r50.synth / r50.load_synth do not wait for it."
rownote 50 "r50.synth needs only the oracle image (no checkpoint download): make_synth.py draws random weights at the real widths, via lfm2moe_oracle.sh synth-ckpt (both spellings in one call). Disk: ~1.29 GB for each of the two spellings under \$DATA/oracle-out-lfm2moe-synth/{w,proj} (the tree's oracle-out-lfm2moe-synth.partial while writing, {w,proj}.ACCEPTED written last); RAM: a few GB in the oracle container (the embedding's and the 64-expert MoE layer's fp32 draws before packing). r50.pytest runs test_lfm2moe_quant.py (make_synth / check.py's own tests) via lfm2moe_oracle.sh synth-tests and needs no checkpoint either."
rownote 50 "Hand back: each load line (layers, expert spelling, int8 / bf16 head, devices, bytes) and its seconds (the int8 head's quantisation time separately), every device's weights line against loader::lfm::device_weight_bytes, the synthetic checkpoints' host-half lines (tensors, 0 unconsumed, both spellings) and the plan lines (one card at 131072; the two-card split by layer bytes)."
stage r50.host       50 default cpu -                         -        "host: lfm2moe_test, lfm2moe_repack_test, lfm2moe_rope_test, lfm2moe_plan_test, lfm2moe_synth_host_test"
st_r50_host() {
  run_tests '^(lfm2moe_test|lfm2moe_repack_test|lfm2moe_rope_test|lfm2moe_plan_test|lfm2moe_synth_host_(w|proj)_test)$'
  jgrab host '^lfm2moe_(test|repack_test|rope_test|plan_test|synth_host_(w|proj)_test)$' \
    'OK|ok$|weight_bytes|max_len_auto|pp_split|0 unconsumed|tensors'
  finish
}
stage r50.pytest     50 default cpu oracle_image              -        "the Python tools' tests in the oracle image with the 5.19.0 site: test_lfm2moe_quant.py (make_synth / check.py / the packer's rule) - lfm2moe_oracle.sh synth-tests"
st_r50_pytest() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA synth-tests" "test_lfm2moe_quant.py"
  grab_all pytest '^(PASS|FAIL|SKIP) |passed|ACCEPTED|REJECTED' 20
  finish
}
stage r50.synth      50 optin   cpu oracle_image              -        "make_synth.py --layers 4, both spellings -> \$DATA/oracle-out-lfm2moe-synth/{w,proj}; check.py ACCEPTED - lfm2moe_oracle.sh synth-ckpt"
st_r50_synth() {
  chk "tools/box_validate/lfm2moe_oracle.sh $DATA synth-ckpt" "the synthetic LFM2.5-24B-A2B checkpoints (both expert spellings)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_lfm2moe_synth' $STATE/have.env"
  grab_all synth '^ACCEPTED|REJECTED|^wrote|FAILED' 12
  finish
}
stage r50.load_synth 50 default gpu lfm2moe_synth             r50.synth "lfm2moe_load_checkpoint_test on both synthetics: one card (int8 and bf16 heads), then --pp 2 split 2 over both (SKIP without the second card) - plan == allocation, every linear's layout read back"
st_r50_load_synth() {
  run_tests '^lfm2moe_load_synth_(w|proj|bf16|pp2|pp2_bf16)_test$'
  jgrab load-synth '^lfm2moe_load_synth' 'OK|weights|unconsumed|SKIP'
  finish
}
stage r50.load_real  50 default gpu lfm2moe_rtn               -        "lfm2moe_load_checkpoint_test on 24q's RTN export (one card; two cards with its split=20 positional argument when two GPUs)"
st_r50_load_real() {
  chk "build/tests/lfm2moe_load_checkpoint_test \$(tools/box_validate/data.sh resolve \"\${SNAP_LFM2MOE_RTN:-?}\")" \
    "load_lfm2moe on 24q's RTN / AutoRound export"
  jgrab load-real '^lfm2moe_load_checkpoint_test$' 'OK|weights|unconsumed'
  finish
}

# ======================================================================================
row 51 "spec 24c - LFM2.5-24B-A2B decode: the launch-count P0, short conv / hd64 attention / sigmoid-bf16 route / 64 x 1536 MoE, the fused arms, one card then two (plan 24c)"
rownote 51 "Written blind on the Mac: NO lfm_* binary has ever been compiled by ocloc (kernel_cmdlines +72 / -0 / ~0: 57 of them P0's lines - every gemv at LFM2's five shapes x S x layout, the head, the argmax at 128000 / 125017, the stand-ins - and 15 family kernels incl. the three fused arms and a test-only M = 4 conv). No existing .cl edited: G0's sha is F0. On the Mac's GPU (tools/mac/clrun/lfm2moe_run, indicative): the norm (_NF, _S1, _S2 against torch's rows), the conv over 18 positions (the ring wraps), the router, the route (incl. a tie at the 4th / 5th, all-equal, a reordering bias, a dominant expert), gate||up, down + the rank-order combine + the fold, the attention prep at 0 / 1 / 4095 / 131071, the gather, the eager attention at 6 / 300 / 3000 keys against torch, and the three arms against their as-built pairs - all exact (silu_mul 1 ulp: exp). lfm_attn.cl (flash: shuffles, 2D block reads) has never run anywhere."
rownote 51 "P0 first (decision 6): r51.p0 runs probe_lfm_step on the card before any engine stage - the 383-launch skeleton from EXISTING kernels at LFM2's shapes, the 277-launch fused skeleton, the GEMV-only floor and 383 noops, interleaved, at depth 4096 and 32768, with --sweep-s (every int4 shape at S 1/2/4/8/16 x layouts 0/1; the smallest S within 3 % of the best) and --pair (stage A of the two-launch norm). Stand-ins: moe.cl reads 5 slots (x 4/5 PROPOSED), Kolibri's attention is head_dim 128 (x 1/2 PROPOSED), the conv / silu / embed slots are noops. Write the table into docs/probe-lfm2moe-<date>.md (a 'P0' section): step ms per list, the gap per launch, the small-kernel floor, the S picks. The decision rule (PROPOSED): every arm whose skeleton saving is >= 1 % of the step is A/B'd in r51.speed; the S picks become model::Lfm2MoeDesc::linears' {S, layout} (24b's PROVISIONAL S 1 everywhere until then) - a re-pick adds the matching gemv / lfm_res_norm / lfm_conv / lfm_silu_mul lines (lfm2moe_variant_names_test names them)."
rownote 51 "Tolerances: B70_LFM_TIE_TOL = 4e-3 (RULED 2026-10-16, spec 24 §12) on the 4th / 5th sel; an exact 0 gap is undetermined. The h-tap bars (median >= 0.9998, min >= 0.99 a layer) are PROPOSED; the conv state after the prompt is held to the h min bar (not bitwise: the engine's int4 GEMV and the reference's dequantised bf16 linear round differently). Flash against fp64 at cosine 0.99999 (spec 6 K1)."
rownote 51 "Data: the synthetics (row 50's r50.synth: oracle-out-lfm2moe-synth/{w,proj}) and their golden sets (r51.synth_golden: lfm2moe_oracle.sh export on each -> \$DATA/oracle-out-lfm2moe-{w,proj}, 4 layers, minutes each, ESTIMATED). 24q's RTN export (data key lfm2moe_rtn) and its golden sets (oracle_lfm2moe_rtn) are not built: r51.rtn / r51.ours / r51.speed SKIP until they exist; configure -DB70_LFM2MOE_RTN=<export dir> -DB70_LFM2MOE_RTN_ORACLE_DIR=<its golden dir> (and _OURS) when they do."
stage r51.k0       51 default cpu -                              g0.sha,g0.bitwise,g0.suite "K0 (F0): every pre-existing binary identical; kernel_cmdlines +72 / -0 / ~0 (no existing .cl edited)"
st_r51_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r51.p0       51 default gpu -                              r51.k0   "P0: probe_lfm_step --depth 4096 / 32768, interleaved a / fused / gemv / noop, --sweep-s, --pair (the table decides decision 6's arms and the S picks)"
st_r51_p0() {
  idle r51-p0
  x "uptime"
  chk "build/tools/probe/probe_lfm_step --depth 4096 --rounds 9 --sweep-s --pair > $STATE/r51-p0-4096.md" "P0 at depth 4096 (+ the S sweep, the norm pair)"
  chk "build/tools/probe/probe_lfm_step --depth 32768 --rounds 9 > $STATE/r51-p0-32768.md" "P0 at depth 32768"
  x "cat $STATE/r51-p0-4096.md $STATE/r51-p0-32768.md"
  grab_all p0 '^\| (a|fused|gemv|noop) \||^- step|gap per launch|roofline|\*\*pick\*\*|--pair' 60
  finish
}
stage r51.host     51 default cpu -                              -        "host: lfm2moe_ref_test, lfm2moe_variant_names_test, lfm2moe_plan_test, probe_lfm_step_shapes_test"
st_r51_host() {
  run_tests '^(lfm2moe_ref_test|lfm2moe_variant_names_test|lfm2moe_plan_test|probe_lfm_step_shapes_test)$'
  jgrab host '^(lfm2moe_(ref|variant_names|plan)_test|probe_lfm_step_shapes_test)$' 'OK|ok|bitwise|launches|decode scratch|differ'
  finish
}
stage r51.k1       51 default gpu -                              r51.k0   "kbins of the block + lfm2moe_kernels_test (F2 bitwise; flash at cosine 0.99999 at 1 .. 131072 keys; each fused arm against its pair)"
st_r51_k1() {
  kbins lfm_res_norm_M1_K2048_W8_NF lfm_res_norm_M1_K2048_W8_S1 lfm_res_norm_M1_K2048_W8_S1_R lfm_conv_M1_K2048_S1 \
    lfm_conv_M4_K2048_S1 lfm_conv_in_M1_K2048_L1 lfm_router_M1_K2048_E64 lfm_route_M1_E64_T4 \
    lfm_moe_M1_E64_T4_D2048_I1536 lfm_moe_M1_E64_T4_D2048_I1536_R lfm_attn_prep_M1_S1_Q32KV8 lfm_attn_M1_T32_Q32KV8 \
    lfm_attn_eager_M1_T32_Q32KV8 lfm_silu_mul_M1_I11776_S1 lfm_embed_gather_M1_K2048_V128000 \
    lfm_argmax_stage1_M1_V128000_U125017 lfm_argmax_stage2_V128000 gemv_i8w_M1_K2048_N128000 gemv_bf16_M1_K2048_N128000 \
    gemv_M1_K2048_N6144_S1_L1 gemv_M1_K2048_N2048_S1_L1 gemv_M1_K2048_N3072_S1_L0 gemv_M1_K2048_N23552_S1_L0 \
    gemv_M1_K11776_N2048_S1_L1
  run_tests '^lfm2moe_kernels_test$'
  jgrab k1 '^lfm2moe_kernels_test$' 'bitwise|cosine|within|arm|OK'
  finish
}
stage r51.reject   51 default gpu -                              -        "cli_reject_lfm2moe_* (prefill, mtp, kv8, layers, serve) - refused before the device"
st_r51_reject() {
  run_tests '^cli_reject_lfm2moe_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r51.synth_golden 51 optin cpu oracle_image,lfm2moe_synth   -        "the synthetics' golden sets: lfm2moe_oracle.sh export on oracle-out-lfm2moe-synth/w and /proj (short, 4k, agentic; the engine-format reference: int4 dequantised) -> \$DATA/oracle-out-lfm2moe-{w,proj}; then re-links oracle-out*"
st_r51_synth_golden() {
  local s
  for s in w proj; do
    chk "LFM_PROMPTS='lfm2moe_short lfm2moe_4k lfm2moe_agentic' tools/box_validate/lfm2moe_oracle.sh $DATA export $DATA/oracle-out-lfm2moe-synth/$s" \
      "the synthetic ($s) golden sets"
  done
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_oracle_lfm2moe_synth' $STATE/have.env"
  grab_all gap 'sel gap|^wall' 20
  finish
}
stage r51.k3       51 default gpu lfm2moe_synth                  r51.k1   "lfm2moe_decode_test on the synthetics: plan == allocation, 41 launches, two runs bitwise (tokens, logits, routes, rings, KV, both residuals, every tap), eager / bf16 head, --fuse route / router / conv / all bitwise none"
st_r51_k3() {
  run_tests '^lfm2moe_decode_synth_(w|proj|eager|bf16head|fuse_(route|router|conv|all))_test$'
  jgrab k3 '^lfm2moe_decode_synth' 'plan == allocation|launches|F4|Review Focus 4|memory|OK'
  finish
}
stage r51.golden   51 default gpu lfm2moe_synth,oracle_lfm2moe_synth r51.k3 "lfm2moe_golden_test on the synthetics: tokens tie-aware, routes within B70_LFM_TIE_TOL, conv state, h taps - short / 4k / agentic x both spellings; _bf16head, _eager"
st_r51_golden() {
  run_tests '^lfm2moe_golden_synth_(w|proj)_lfm2moe_(short|4k|agentic)_test$|^lfm2moe_golden_synth_short_(bf16head|eager)_test$'
  jgrab golden '^lfm2moe_golden_synth' 'gate:|routing:|conv state|^  h |step [0-9]+:|OK'
  finish
}
stage r51.rtn      51 default gpu lfm2moe_rtn,oracle_lfm2moe_rtn r51.k3  "the RTN export: decode_test (383 launches, --fuse each arm) and golden_test short / 4k / agentic (+ bf16 head, eager)"
st_r51_rtn() {
  run_tests '^lfm2moe_decode_rtn(_fuse_(route|router|conv|all))?_test$|^lfm2moe_golden_rtn_(lfm2moe_(short|4k|agentic)|short_bf16head|short_eager)_test$'
  jgrab rtn '^lfm2moe_(decode|golden)_rtn' 'plan == allocation|launches|gate:|routing:|conv state|OK'
  finish
}
stage r51.ours     51 optin   gpu lfm2moe_ours,oracle_lfm2moe_ours r51.k3 "24q's AutoRound export: the same golden gates (short / 4k / agentic)"
st_r51_ours() {
  run_tests '^lfm2moe_golden_ours_lfm2moe_(short|4k|agentic)_test$'
  jgrab ours '^lfm2moe_golden_ours' 'gate:|routing:|conv state|OK'
  finish
}
stage r51.pp       51 default gpu lfm2moe_synth                  r51.k3   "lfm2moe_pp_test: two cards bitwise one (copy, peer; a dense cut at split 2 and a MoE cut at 3; P4; replay; the RTN export at 20 / 19 when present) + the golden gate across split 2 (peer)"
st_r51_pp() {
  run_tests '^(lfm2moe_pp_test|lfm2moe_golden_synth_short_pp2_test)$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^lfm2moe_(pp_test|golden_synth_short_pp2)' 'landing|device|P4|tokens|gate:|OK'
  finish
}
stage r51.speed    51 optin   gpu lfm2moe_rtn                    r51.rtn  "Task 8: decode at depth 1024 / 4096 / 32768 / 130816 (+ 256 decoded = the trained 131072; the deep rows ingest one replay per id: hours, ESTIMATED), B70_LFM_FUSE none vs each arm vs all as interleaved rounds (median of 3), int8 / bf16 head, flash / eager; --profile --repeats 5 at 4096"
st_r51_speed() {
  local snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")' r d a i
  local arms=(none route router conv all)
  idle r51-speed
  x "uptime"
  for d in 1024 4096 32768 130816; do   # 130816 + 256 = the trained 131072 (max_len auto caps there)
    for r in 1 2 3; do
      for i in 0 1 2 3 4; do   # rotated per round: interleaved, not sequential arms
        a=${arms[$(((i + r) % 5))]}
        chk "B70_LFM_FUSE=$a build/src/cli/b70-decode $snap --bench --depth $d --tg 256 --max-len auto" "depth $d fuse $a round $r"
      done
    done
    chk "build/src/cli/b70-decode $snap --bench --depth $d --tg 256 --max-len auto --lm-head bf16" "depth $d bf16 head"
    chk "B70_LFM_ATTN=eager build/src/cli/b70-decode $snap --bench --depth $d --tg 256 --max-len auto" "depth $d eager"
  done
  chk "build/src/cli/b70-decode $snap --profile --depth 4096 --steps 32 --repeats 5" "--profile at 4096"
  x "uptime"
  grab_all rows '^\| b70-decode' 80
  grab_all profile '^\| (lfm_|gemv|argmax)|^MEASURED' 40
  finish
}

# ======================================================================================
row 52 "spec 24d - LFM2.5-24B-A2B prefill: the conv over a chunk with the ring carried, the grouped 64-expert MoE, flash at head_dim 64 (plan 24d)"
rownote 52 "Written blind on the Mac after row 51 (24c's decode, itself unvalidated): NO lfm_pf_* binary has ever been compiled by ocloc - r52.k1's kbins is the first build of the 18 new command lines (kernel_cmdlines +18 / -0 / ~0: lfm_pf_embed_gather, lfm_pf_conv and its conv-tile _T build, lfm_pf_moe, lfm_pf_attn and _EAGER, 24c's norm _NF / _S1, router, route and attention prep at M 2048, pf_dequant_slab at LFM2's five {K, N, layout} - each linear at the descriptor's own layout, layout 0 for q||k||v and the dense gate||up -, pf_moe_gemm 2048 x 3072 SiLU and 1536 x 2048). No existing .cl edited: G0's sha is F0. On the Mac's GPU (tools/mac/clrun/lfm2moe_run, indicative): the embed gather, the seed + conv at (p0, C) = (0, 1) .. (4097, 300) plain and _T, the sort / gather / both dequants (two batches, skipped experts) / the rank-order combine + fold at C 64 - all exact against tests/kernels/lfm2moe_pf_ref.h. lfm_pf_attn.cl (DPAS, 2D block reads) and the reused GEMMs have never run at these shapes."
rownote 52 "Bars PROPOSED (spec 20d's family; the tests print the distributions): prefill vs decode's fill - every row of the first attention layer >= 0.999, all rows' median >= 0.9998, p01 >= 0.99; the live conv slots and the last row's h taps >= 0.99; flash vs fp64 >= 0.99999 (spec 6 K1), the eager build vs the reference's rounding points >= 0.9999. The conv, the routes and the combine are bitwise per row against the host twin; any chunk split (1..5, 64, 2047, 2048, 2049, chunks of 1 / 2 / 64) is bitwise one chunk (every kernel row-local or keyed to absolute positions). Routes against decode's within B70_LFM_TIE_TOL (4e-3, RULED)."
rownote 52 "Data: the synthetics (r50.synth) and their golden sets (r51.synth_golden) for r52.f4 / r52.golden; 24q's RTN export and its golden sets for r52.rtn / r52.speed (SKIP until they exist, as row 51's). Hand back: the kbins list, lfm2moe_pf_kernels_test's lines (slabs, conv, per-row, MoE grouped == dense, flash cosines), lfm2moe_prefill_test's six sections (walk, F4, splits, the KV / conv / tap bars per layer, routing, tokens), the golden gate's lines, lfm2moe_pp_test's (e) lines, and r52.speed's pp rows with the B70_PREFILL_PROFILE phase table (the dequant share against plan 24d's ~6,000-7,000 t/s ESTIMATE at pp 4096)."
stage r52.k0     52 default cpu -                            g0.sha,g0.bitwise,g0.suite "K0 (F0): pre-existing binaries identical; kernel_cmdlines +18 / -0 / ~0 (no existing .cl edited)"
st_r52_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r52.host   52 default cpu -                            -        "host: lfm2moe_pf_ref_test (the conv over a chunk vs decode's conv_step at every split, torch's chunked conv, the 64-expert sort / combine, the flash walk in fp64), lfm2moe_pf_variant_names_test, lfm2moe_plan_test (the prefill terms: 1259 launches a chunk, 2 + 1 weight batches, the scratch, the plan at 131072 with it)"
st_r52_host() {
  run_tests '^(lfm2moe_pf_ref_test|lfm2moe_pf_variant_names_test|lfm2moe_plan_test)$'
  jgrab host '^lfm2moe_(pf_ref|pf_variant_names|plan)_test$' 'bitwise|tiles|prefill:|OK|ok'
  finish
}
stage r52.k1     52 default gpu -                            r52.k0   "kbins of the block + lfm2moe_pf_kernels_test: slabs at each linear's layout, the embed gather / seed / conv / sort / combine bitwise, the per-row kernels at M 2048 bitwise, pf_moe_gemm grouped == dense per row, flash vs fp64 0.99999, eager vs the rounding points, a split at 64 bitwise"
st_r52_k1() {
  kbins lfm_pf_embed_gather_K2048_V128000 lfm_pf_conv_K2048_C2048 lfm_pf_conv_K2048_C2048_T \
    lfm_pf_moe_E64_T4_D2048_I1536_TM32 lfm_pf_attn_Q32KV8_HPW4_RPW16_KT64 lfm_pf_attn_Q32KV8_HPW4_RPW16_KT64_EAGER \
    lfm_res_norm_M2048_K2048_W8_NF lfm_res_norm_M2048_K2048_W8_S1 lfm_router_M2048_K2048_E64 lfm_route_M2048_E64_T4 \
    lfm_attn_prep_M2048_S1_Q32KV8 pf_dequant_slab_K2048_N6144_L1 pf_dequant_slab_K2048_N2048_L1 \
    pf_dequant_slab_K2048_N3072_L0 pf_dequant_slab_K2048_N23552_L0 pf_dequant_slab_K11776_N2048_L1 \
    pf_moe_gemm_K2048_N3072_SILU pf_moe_gemm_K1536_N2048
  run_tests '^lfm2moe_pf_kernels_test$'
  jgrab k1 '^lfm2moe_pf_kernels_test$' 'bitwise|exact|cosine|grouped|OK'
  finish
}
stage r52.reject 52 default gpu -                            -        "cli_reject_lfm2moe_prefill_{int8,sycl,chunk} - the l0-only backend and the 2048 chunk, refused before the device"
st_r52_reject() {
  run_tests '^cli_reject_lfm2moe_prefill_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r52.f4     52 default gpu lfm2moe_synth                r52.k1   "lfm2moe_prefill_test on the synthetics (w, proj, eager, bf16 head, --fuse conv): the walk's launches and plan, twice bitwise, recorded == replayed, the chunk splits bitwise, prefill vs decode's fill within the bars, routes up to near-ties, tokens tie-aware"
st_r52_f4() {
  run_tests '^lfm2moe_prefill_synth_(w|proj|eager|bf16head|fuse_conv)_test$'
  jgrab f4 '^lfm2moe_prefill_synth' '^[1-6]\.|split at|chunks of|per attention layer|h at the last row|live slots|OK|FAIL'
  finish
}
stage r52.golden 52 default gpu lfm2moe_synth,oracle_lfm2moe_synth r52.f4 "lfm2moe_golden_test's prefill mode on the synthetics: short / 4k / agentic x both spellings, chunks of 64 on 4k, eager"
st_r52_golden() {
  run_tests '^lfm2moe_golden_prefill_synth_'
  jgrab golden '^lfm2moe_golden_prefill_synth' 'prefill:|gate:|routing:|conv state|^  h |step [0-9]+:|OK'
  finish
}
stage r52.rtn    52 default gpu lfm2moe_rtn,oracle_lfm2moe_rtn r52.f4 "the RTN export: lfm2moe_prefill_test (+ eager) and the golden prefill gate short / 4k / agentic (+ eager twins)"
st_r52_rtn() {
  run_tests '^lfm2moe_prefill_rtn(_eager)?_test$|^lfm2moe_golden_prefill_rtn_lfm2moe_(short|4k|agentic)(_eager)?_test$'
  jgrab rtn '^lfm2moe_(prefill|golden_prefill)_rtn' '^[1-6]\.|split at|prefill:|gate:|routing:|conv state|OK|FAIL'
  finish
}
stage r52.ours   52 optin   gpu lfm2moe_ours,oracle_lfm2moe_ours r52.f4 "24q's AutoRound export: the golden prefill gate (short / 4k / agentic)"
st_r52_ours() {
  run_tests '^lfm2moe_golden_prefill_ours_lfm2moe_(short|4k|agentic)_test$'
  jgrab ours '^lfm2moe_golden_prefill_ours' 'prefill:|gate:|routing:|OK'
  finish
}
stage r52.pp     52 default gpu lfm2moe_synth                r52.f4   "two cards: lfm2moe_pp_test's (e) - the prefill bitwise one card at chunks 2048 / 64, a dense cut (split 2) and a MoE cut (3), each decode hand-off, the RTN export at 20 when present; each card's launches; recorded + replayed; P4 on the prefill hand-off"
st_r52_pp() {
  run_tests '^lfm2moe_pp_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab pp '^lfm2moe_pp_test$' '\(e\)|prefill, split|P4|OK'
  finish
}
stage r52.speed  52 optin   gpu lfm2moe_rtn                  r52.rtn  "--bench --prefill-length 4096 / 32768 (median of 3, interleaved with the eager build), B70_PREFILL_PROFILE=1 phase tables at 4096 (the dequant's share: plan 24d's estimate), two cards at 4096"
st_r52_speed() {
  local snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")' r d
  idle r52-speed
  x "uptime"
  for d in 4096 32768; do
    for r in 1 2 3; do   # interleaved: flash and eager alternate inside a round
      chk "build/src/cli/b70-decode $snap --bench --prefill-length $d --tg 32 --max-len auto" "pp $d flash round $r"
      chk "B70_LFM_ATTN=eager build/src/cli/b70-decode $snap --bench --prefill-length $d --tg 32 --max-len auto" "pp $d eager round $r"
    done
  done
  chk "B70_PREFILL_PROFILE=1 build/src/cli/b70-decode $snap --bench --prefill-length 4096 --tg 1 --max-len auto" "the phase table at 4096"
  chk "ZE_AFFINITY_MASK=0,1 build/src/cli/b70-decode $snap --bench --prefill-length 4096 --tg 32 --pp 2 --max-len auto" "pp 4096 on two cards"
  x "uptime"
  grab_all rows '^\| b70-decode' 40
  grab_all phases 'prefill phase attribution|^  (norm|slab_|attn_|gdn_seed|gdn_conv|moe_|head|TOTAL)' 40
  finish
}

# ======================================================================================
row 53 "spec 24e - LFM2.5-24B-A2B served: the chat template byte-identical to transformers 5.19.0, the tokenizer equal, the pythonic tool-call parser and reasoning (keyed on the prompt's <think>), sampling defaults and HF's repetition penalty on the host sampler, prefix-cache snapshots with the conv rings, b70-serve <lfm2moe> on one card by default; A4 on the engine and the agentic transcript's cache hit rate (plan 24e)"
rownote 53 "Written blind on the Mac after rows 51 / 52 (24c's decode, 24d's prefill: neither has run on a card): nothing of 24e ran on a card. 24e adds NO kernel (Lfm2MoeEngine's snapshot calls are copies on the immediate lists; kernel_cmdlines +0 / -0 / ~0): G0's g0.sha is the 'nothing moved' check. It does change shared host code - four minja patches (third_party/VERSIONS: the tool-call probe's pythonic marker, the arguments rewrite tolerating a missing function / arguments, 'in' over falsy items, Python's float repr), generation_sampling's absent-field defaults (HF's: T 1.0, top-k 50, top-p 1.0), server.cc (the SeenIds, parser stats), openai.cc, chat_format - r53.k0 re-runs the other models' template / parser / server tests."
rownote 53 "The synthetic checkpoints (oracle-out-lfm2moe-synth, r50.synth) carry no tokenizer.json / chat_template.jinja / generation_config.json (make_synth.py copies them only with --tokenizer, which lfm2moe_oracle.sh synth-ckpt does not pass): b70-serve needs them, so r53.synth_files copies the four small files from the original's HF snapshot into each (cp -n, idempotent). lfm2moe_template_test / lfm2moe_tokenizer_test read the original's snapshot at 1769516a from the HF cache (B70_LFM_SNAPSHOT_DIR / B70_LFM_TOKENIZER_JSON name another) and SKIP 77 without it."
rownote 53 "A random-weight synthetic never writes a tool call: the tool-call / reasoning / cut-block paths through HTTP are lfm2moe_server_test's (host, scripted engine); the card stages here prove the engine half (snapshots, startup, the server's greedy chat == b70-decode, seeded sampling) and, with 24q's RTN export, a real model's A4 and the agentic loop."
rownote 53 "Decision 5's A4 bar (>= 33 / 36 first-call matches, 0 parse failures) and decision 7's rule counts are PROPOSED evidence; the operator rules after r53.a4. The cache's hit rate on the agentic transcript (r53.agentic) has no bar: it is recorded - the turns the template re-renders in another order than generated (a call before its text) are the known misses (spec 24 section 4.6) and are counted."
stage r53.k0 53 default cpu - g0.sha,g0.bitwise,g0.suite "K0: 24e adds no kernel (every binary identical, G0) and the other models' request path is unchanged - G0's suite and here the host tests that read the patched minja / generation_sampling / server.cc: every template, toolcall and server test, openai_test, minja_ext_test"
st_r53_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  run_tests '^(template_test|template_agnes_test|template_ornith_test|template_k2_test|template_kolibri_test|template_muse_test|minja_ext_test|toolcall_test|toolcall_k2_test|toolcall_kolibri_test|toolcall_muse_test|openai_test|k2_server_test|kolibri_server_test|muse_server_test|ornith_server_test|qwen4exp_server_test|protocol_test|prefix_server_test|mtp_server_test|lookup_server_test|spec_accept_test|sampling_test)$' '' 'kv8'
  finish
}
stage r53.host 53 default cpu - - "host: lfm2moe_template_test (32 renders + the agentic session byte-identical to apply_chat_template), lfm2moe_tokenizer_test (125017 ids, the 124 added tokens, 21 texts, corpus.txt, every render's ids), toolcall_lfm2_test (the HF renders parse back and re-render to the same bytes, the idioms, the invalid blocks, cut blocks, the limits, a fuzz - every split), lfm2moe_sampling_test (the penalty, SeenIds, the defaults), lfm2moe_serve_args_test, lfm2moe_server_test (scripted engine over HTTP), lfm2moe_snapshot_test (the layouts on the real shapes, restores at 0 / 1 / 2 / 15 / 16 / 17 / 2048 / 4096 / 2049 / 5000 / 300 == cold, one card <-> two, the server over HTTP), lfm2moe_rules_test, minja_ext_test; the agentic loop's Python test"
st_r53_host() {
  run_tests '^(lfm2moe_template_test|lfm2moe_tokenizer_test|toolcall_lfm2_test|lfm2moe_sampling_test|lfm2moe_serve_args_test|lfm2moe_server_test|lfm2moe_snapshot_test|lfm2moe_rules_test|minja_ext_test)$'
  jgrab snapshot '^lfm2moe_snapshot_test$' 'layout|round trip|restores at|restore at|pp:|server:|OK'
  jgrab parser '^toolcall_lfm2_test$' 'round trip|fuzz|quotes|OK'
  chk "python3 tools/prefix/test_agentic_loop.py && python3 tools/box_validate/test_box_validate.py AgenticRunTest" "the agentic loop's Python tests"
  finish
}
stage r53.reject 53 default cpu - - "b70-serve's LFM2 refusals before the device (tests/model/lfm2moe: config.json only): --mtp, --spec mtp|lookup|dflash, the proposers' tuning flags, --draft-vocab, --kv-cache int8, --prefill-backend l0-int8|sycl-tla, the Qwen3.8-Flash-Next flags, --chat-template, --max-len 262144, --pp 2 --device 1, --pipeline-split / -handoff without --pp 2"
st_r53_reject() {
  run_tests '^cli_reject_serve_lfm2moe_'
  finish
}
stage r53.synth_files 53 default cpu lfm2moe_synth - "the four small files (tokenizer.json, tokenizer_config.json, generation_config.json, chat_template.jinja) from the original's HF snapshot into each synthetic checkpoint (cp -n): b70-serve needs them"
st_r53_synth_files() {
  local hf='${HF_HOME:-$HOME/.cache/huggingface}' src s f
  src="\$(ls -d $hf/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots/*/ 2> /dev/null | head -1)"
  x "test -n \"$src\" && test -s \"$src/chat_template.jinja\""
  step_rc $? "the original's small files are in the HF cache (hf download LiquidAI/LFM2.5-24B-A2B --include '*.json' '*.jinja' if not)"
  for s in w proj; do
    for f in tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja; do
      x "cp -n \"$src/$f\" oracle-out-lfm2moe-synth/$s/$f"
    done
    chk "test -s oracle-out-lfm2moe-synth/$s/tokenizer.json && test -s oracle-out-lfm2moe-synth/$s/chat_template.jinja && test -s oracle-out-lfm2moe-synth/$s/generation_config.json && test -s oracle-out-lfm2moe-synth/$s/tokenizer_config.json" "the synthetic $s carries the tokenizer files"
  done
  finish
}
stage r53.snap 53 default gpu lfm2moe_synth r52.f4 "Review Focus 5 on the card, synthetic w: lfm2moe_snapshot_gpu_synth_test (one card), _synth_pp_test (--pp 2, split 2), _cross_test (a snapshot taken on two cards continues on one, and the reverse) - the sizes (245,760 B of state, n x 20,480 B of KV at the real widths); restores at the block ends 2048 / 4096 from the hook's own saves BITWISE the cold run (32 ids, logits, KV, the conv rings' last two rows); at the request ends 2049 / 5000 / 300 / 2 / 1 / 0 bitwise the session that never left"
st_r53_snap() {
  run_tests '^lfm2moe_snapshot_gpu_synth(_pp)?_test$|^lfm2moe_snapshot_gpu_cross_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^lfm2moe_snapshot_gpu' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r53.snap_rtn 53 default gpu lfm2moe_rtn r53.snap "the same on 24q's RTN export: lfm2moe_snapshot_gpu_rtn_test (one card) and _rtn_pp_test (--pp 2, the planner's split)"
st_r53_snap_rtn() {
  run_tests '^lfm2moe_snapshot_gpu_rtn(_pp)?_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^lfm2moe_snapshot_gpu' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r53.serve 53 default gpu lfm2moe_synth r53.synth_files,r53.snap "b70-serve <synthetic w> startup and requests: one card by default (the startup line: LFM2.5-24B-A2B, lfm2_moe chat format, --pp 1, prefill l0, mtp none, lm_head int8, kv cache bf16, sampling defaults sampled T 0.10 top-p 1.00 top-k 50 repetition penalty 1.05, the conv rings + KV snapshots), --max-len auto planned with the prefill scratch; --pp 2 and --pipeline-handoff peer; a seeded sampled request twice, identical (the adapter re-seeds per request); B70_LFM_PENALTY_GREEDY=1 named in the startup line"
st_r53_serve() {
  local ck="oracle-out-lfm2moe-synth/w"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck"
  step_rc $? "b70-serve <synthetic w>, one card (the default), seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pp 2 --max-len 16384"
  step_rc $? "b70-serve <synthetic w> --pp 2"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pp 2 --pipeline-handoff peer --max-len 16384"
  step_rc $? "b70-serve <synthetic w> --pp 2 --pipeline-handoff peer"
  x "B70_LFM_PENALTY_GREEDY=1 ZE_AFFINITY_MASK=0 PORT=$PORT tools/box_validate/serve_probe.sh $ck --max-len 16384"
  step_rc $? "b70-serve <synthetic w> with B70_LFM_PENALTY_GREEDY=1 (a greedy request carrying the default penalty goes through the host)"
  grab_all auto 'max_len: auto ->|^split: ' 6
  grab_all memory '^memory' 8
  grab_all startup '^b70-serve: .*LFM2.5-24B-A2B' 4
  grab_all seeded '^SEEDED ' 4
  finish
}
stage r53.chat 53 default gpu lfm2moe_synth r53.serve "one greedy chat through b70-serve equal to b70-decode on the same ids - golden_server_test --chat on the synthetic w (prose / code / cjk as one user message, 32 ids, temperature 0; b70-decode --ids <prompt_token_ids> --n 32 --prefill --lm-head int8), one card, --pp 2 and --pipeline-handoff peer; run directly (not registered: without a checkpoint it would fail, not SKIP)"
st_r53_chat() {
  local ck="oracle-out-lfm2moe-synth/w" a z
  for a in "--max-len 16384" "--max-len 16384 --pp 2" "--max-len 16384 --pp 2 --pipeline-handoff peer"; do
    z=0; case "$a" in *"--pp 2"*) z=0,1 ;; esac
    chk "ZE_AFFINITY_MASK=$z B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $ck --chat" \
      "LFM2.5 synthetic, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 9
  finish
}
stage r53.real 53 default gpu lfm2moe_rtn r53.chat,r53.snap_rtn "24q's RTN export on one card: b70-serve startup (--max-len auto -> 131072 with the int8 head and the prefill scratch, derived 18.40 GB of 32.53) with a seeded request twice, then golden_server_test --chat (the server's greedy chat = b70-decode on its prompt_token_ids)"
st_r53_real() {
  local snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")'
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $snap"
  step_rc $? "b70-serve <RTN export>, one card, auto max_len, seeded sampling"
  chk "ZE_AFFINITY_MASK=0 B70_SERVE_ARGS='--max-len 32768' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $snap --chat" \
    "golden_server_test --chat against the RTN export"
  grab_all auto 'max_len: auto ->' 4
  grab_all memory '^memory' 4
  grab_all startup '^b70-serve: .*LFM2.5-24B-A2B' 2
  grab_all chat 'chat of [0-9]+ prompt ids' 3
  finish
}
stage r53.a4 53 optin gpu lfm2moe_rtn,lfm2moe_a4_ref r53.real "plan 24e Task 5 Step 1, A4 on the engine: the 36 scenarios of tests/golden/toolcall-lfm2moe (engine_generate.sh l0, 2048 new ids each) on the RTN export against 24a's bf16 reference (oracle-out-lfm2moe-a4): score.py --format pythonic (first-call agreement, decision 5's bar PROPOSED >= 33 / 36; the reference's parse failures), lfm2moe_rules on the engine's and on the reference's texts (decision 7's evidence: how often each lenient reading fired; the engine's blocks left as content = parse failures)"
st_r53_a4() {
  local d=$STATE/a4-lfm2moe snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")'
  x "mkdir -p $d && cp oracle-out-lfm2moe-a4/*.bf16.txt $d/"
  chk "ZE_AFFINITY_MASK=0 tools/toolcall/engine_generate.sh $snap tests/golden/toolcall-lfm2moe $d l0" "A4 engine runs, bf16 head"
  chk "ZE_AFFINITY_MASK=0 LM_HEAD=int8 tools/toolcall/engine_generate.sh $snap tests/golden/toolcall-lfm2moe $d l0" "A4 engine runs, int8 head (the served default)"
  chk "python3 tools/toolcall/score.py --format pythonic $d bf16 l0 l0-i8head | tee $d/score.md" "scoring against the bf16 reference"
  x "grep -q '^bf16: 0 parse failure' $d/score.md"
  step_rc $? "the reference's calls all parse (the parser reads every call the reference wrote)"
  chk "build/tests/lfm2moe_rules $d/*.l0.txt | tee $d/rules-l0.md" "decision 7: the engine's texts (bf16 head), 0 blocks left as content"
  chk "build/tests/lfm2moe_rules $d/*.l0-i8head.txt | tee $d/rules-l0-i8head.md" "decision 7: the engine's texts (int8 head)"
  x "build/tests/lfm2moe_rules $d/*.bf16.txt | tee $d/rules-bf16.md"
  step_rc $? "decision 7: the reference's own texts (what the bf16 model writes)"
  grab_all a4 'match|/ 36|parse failure|files,|rules fired' 12
  finish
}
stage r53.agentic 53 optin gpu lfm2moe_rtn r53.real "plan 24e Task 5 Step 2, the agentic transcript (tests/golden/prompts/lfm2moe_agentic.json: 6 tools, a coding session) as a LIVE agent loop through b70-serve on the RTN export, the model's own turns fed back: per turn the prompt length, cached_tokens, TTFT and decode t/s; the hit rate; the turns whose call the template re-renders in another order than generated (the known miss, counted); three arms - the default cache, --prefix-split-last, --prefix-cache-gb 0"
st_r53_agentic() {
  local snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")' t=tests/golden/prompts/lfm2moe_agentic.json
  idle r53-agentic
  chk "ZE_AFFINITY_MASK=0 PORT=$PORT tools/box_validate/agentic_run.sh $STATE/agentic-default $snap $t --max-len 65536" "agentic loop, prefix cache auto"
  chk "ZE_AFFINITY_MASK=0 PORT=$PORT tools/box_validate/agentic_run.sh $STATE/agentic-split $snap $t --max-len 65536 --prefix-split-last" "agentic loop, --prefix-split-last"
  chk "ZE_AFFINITY_MASK=0 PORT=$PORT tools/box_validate/agentic_run.sh $STATE/agentic-off $snap $t --max-len 65536 --prefix-cache-gb 0" "agentic loop, cache off"
  x "uptime"
  grab_all agentic '^hit rate|^turns whose call|^ideal prefix|^\| [0-9]+ \|' 60
  finish
}
stage r53.opencode 53 optin gpu-self lfm2moe_rtn,opencode_log r53.real "the recorded opencode session (OPENCODE_LOG, a b70-serve --log-requests directory) replayed through b70-serve <RTN export> with the prefix cache on, then --prefix-split-last (tools/prefix/replay_ab.sh takes the GPU lock itself): per request cached_tokens and TTFT"
st_r53_opencode() {
  local snap='$(tools/box_validate/data.sh resolve "${SNAP_LFM2MOE_RTN:-?}")' arm args
  for arm in on split; do
    args=""; [ "$arm" = split ] && args="--prefix-split-last"
    chk "MODEL=$snap ARMS=on SERVE_ARGS='$args' tools/prefix/replay_ab.sh $OPENCODE_LOG $STATE/r53.opencode.$arm" "replay, cache $arm"
  done
  grab_all replay 'prompt|cached|ttft|requests' 60
  finish
}

# ======================================================================================
row 54 "spec 19b - the DFlash drafter on the card: loader::load_drafter (z-lab/Qwen3.8-27B-DFlash2 beside Qwen3.8, bf16 or int8 at load, the target's embedding and head shared), the df_* kernels (embed, norms, the grouped two-tap convs, the block's q / k / v, ring + block attention, top-16 with the vocab mask and the draft vocabulary's id map, the selector walk), the captured draft (87 launches) and projection (8) lists, the planner's drafter term; D0 and D1 (plan 19b)"
rownote 54 "Written blind on the Mac: no df_* binary has ever been compiled (ocloc is the box's) - r54.kernels' kbins is the first build of the 14 new command lines (kernel_cmdlines +14 / -0 / ~0: 8 df_*, the bf16 head at 7 rows, the compact heads at 7 rows). Nothing binds them without a drafter (19c's --spec dflash): D0 is G0. On the Mac's GPU (tools/mac/clrun/dflash_run, indicative) the whole draft and projection lists ran at the fixture's tiny shapes against the host twin: rows / scores cos 1.0000000 in 8 of 9 cases (one at 0.99999 - Apple's 1 / sqrt is not correctly rounded), every walk equal, a 256-id draft vocabulary bitwise the full head's, v1. The twin (tests/kernels/dflash_ref.h) matches tools/oracle/dflash_ref.py to 6e-7 (dflash_ref_test). The kernels use work-group reductions only (no sub-group ops): their speed on the B70 is unmeasured."
rownote 54 "Data: the drafter (hf download z-lab/Qwen3.8-27B-DFlash2, 3.85 GB; data key dflash) for r54.load / r54.draft / r54.d1; 19a's tap dumps (the Mac's oracle-out-19a/dumps, 2.7 GB, and oracle-out-19a/draft_vocab.ranked.ids: tools/box_validate.sh --push-data; data key oracle_dflash_dumps) for r54.d1; the gate checkpoint (qwen). r54.d1_ref runs dflash_ref.py in the oracle image: ~15 GB RSS (the drafter fp32 7.2 GB, the target's head and embedding), ~1-2 s a draft on 44 threads, ESTIMATED (16 anchors x 37 dumps x 3 arms ~ 0.5-1 h)."
rownote 54 "D1's bars are spec 19 §6's: top-16 sets tie-aware, scores cosine >= 0.9999 (bf16 drafter) / 0.999 (int8), walk ids equal at T 0. The engine rounds every GEMV input to bf16 (the reused gemv kernels read bf16 activations; vLLM runs the drafter in bf16 throughout) and keeps the ring bf16; the reference is fp32. A Mac preview on the real drafter with real taps (random head: the true scores need the target's head) measured the final rows at cosine 0.99991 (bf16) - if the scores bar fails, read the per-draft table: fp32-input GEMV variants are the lever (plan 19b status)."
rownote 54 "Hand back: the kbins list, dflash_kernels_test's lines, dflash_load_checkpoint_test's lines (bytes, read-backs), dflash_draft_test's lines (launches, the projection, every draft line, the replay, the draft vocabulary line, the host-timed ms), and r54.d1's three tables (out/d1.md) with any FAIL lines."
stage r54.k0 54 default cpu - g0.sha,g0.bitwise,g0.suite "D0: no existing command line moved (kernel_cmdlines +14 / -0 / ~0), every pre-existing binary identical, the suite (G0); 19b's edits to shared files: SafetensorsSet's opt-in single-file mode, runtime::plan / max_len_that_fits' defaulted DrafterPlan, the CMake blocks"
st_r54_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r54.host 54 default cpu - - "host: dflash_loader_test (the config, every refusal, the synthetic checkpoints' device forms), dflash_ref_test (the twin vs dflash_ref.py), dflash_variant_names_test, dflash_plan_test (the buffers, the planner's term, auto); test_dflash_ref.py and test_dflash_d1.py in the oracle image"
st_r54_host() {
  run_tests '^(dflash_loader_test|dflash_ref_test|dflash_variant_names_test|dflash_plan_test)$'
  jgrab host '^dflash_(loader|ref|variant_names|plan)_test$' 'PASS|OK|refusal|bytes|cos|launches|auto'
  chk "tools/oracle/run_in_container.sh 'python3 tools/oracle/test_dflash_ref.py && python3 tools/oracle/test_dflash_d1.py'" \
    "the reference's and the D1 comparator's tests (oracle image)"
  finish
}
stage r54.kernels 54 default gpu - - "the 14 new binaries built (kbins: the first ocloc build of every df_* kernel) and dflash_kernels_test at the drafter's real shapes against the twin: embed / norm / convs / q-k-v prep / the ring store bitwise, SiLU and attention within 1 bf16 ulp (the window's edge and the ring's wrap: p 5 .. 131000, K 7 / 5 / 3 / 1), top-16 exact (ties, NaN, the mask, the draft vocabulary's map), the walk (greedy bitwise; sampled to near ties)"
st_r54_kernels() {
  kbins df_prep_M8_H5120_I17408_Q32KV8_D128_G16T2_R2048_V248320 df_attn_M8_Q32KV8_D128_R2048_W2048_P128 \
    df_topk_R7_N248320_U248077 df_topk_R7_N32768_U32768_C df_topk_R7_N65536_U65536_C df_topk_R7_N131072_U131072_C \
    df_select_M8_K16_Q16_RK256_V248320 df_select_v1_M8_K16 gemv_bf16_M7_K5120_N248320 gemv_bf16_M7_K5120_N32768 \
    gemv_bf16_M7_K5120_N65536 gemv_bf16_M7_K5120_N131072 gemv_i8w_M7_K5120_N65536 gemv_i8w_M7_K5120_N131072
  run_tests '^dflash_kernels_test$' dflash
  jgrab kernels '^dflash_kernels_test$' 'df_|differ|ulp|PASS|FAIL'
  finish
}
stage r54.load 54 default gpu qwen,dflash r54.kernels "loader::load_drafter beside the int4 gate checkpoint (int8 head), int8 and bf16: the bytes = loader::dflash_weight_bytes (2.053 / 3.849 GB, derived), fc / q||k||v / down / the kernel projection / the selector read back = the host forms, the norms, the conv bases, the codebook rows, the mask row = the embedding's row 248070"
st_r54_load() {
  run_tests '^dflash_load_checkpoint_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  jgrab load '^dflash_load_checkpoint_test$' 'DFlash drafter:|read back|formula|PASS'
  finish
}
stage r54.draft 54 default gpu dflash r54.kernels "the draft and projection lists end to end - the real drafter, a synthetic target - against the twin: 87 / 8 launches, 2053 positions projected across the ring's wrap, a draft at p 2053 (rows / scores cosine >= 0.99999, top-16 tie-aware, the walk), the replay bitwise, K 3 by the control block, a sampled walk, a 32768-id draft vocabulary BITWISE the full head (Review Focus 3), the bf16 head; the draft list's host-timed replay (informational)"
st_r54_draft() {
  x "build/tests/dflash_draft_test ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2} | tee $STATE/$STAGE.txt"
  step_rc $? "dflash_draft_test (both drafter formats)"
  grab_all draft '^== drafter|^launches|^projection|^greedy|^K 3|^sampled|^bf16 head|^replay|^draft vocabulary|^dflash_draft_test' 40
  finish
}
stage r54.d1 54 default gpu qwen,dflash,oracle_dflash_dumps r54.draft "D1, the engine's half: build/tools/probe/dflash_d1 over oracle-out-19a's 37 tap dumps (16 anchors each over the continuation, K 7, T 0) for the arms i8 (int8 drafter + int8 head), bf16 (bf16 + bf16), i8dv32k (i8 over 19a's ranked 32k draft vocabulary) -> \$STATE/d1/d1.<arm>.jsonl"
st_r54_d1() {
  chk "SNAP_QWEN=$SNAP_QWEN tools/box_validate/dflash_d1.sh gpu $STATE/d1" "dflash_d1 x 3 arms"
  grab_all d1 '^dflash_d1:|^DFlash drafter:|^=== ' 20
  finish
}
stage r54.d1_ref 54 default cpu qwen,dflash,oracle_dflash_dumps,oracle_image r54.d1 "D1, the reference's half (box CPU, the oracle image): tools/oracle/dflash_d1.py per arm - dflash_ref.py on the same taps, anchors and head form (int8 RTN simulated where the engine quantised), the vocab mask, V' masked; the bars: top-16 sets tie-aware, scores cosine >= 0.9999 (bf16) / 0.999 (int8), the walk equal. The table -> \$STATE/d1/d1.md"
st_r54_d1_ref() {
  chk "SNAP_QWEN=$SNAP_QWEN tools/box_validate/dflash_d1.sh ref $STATE/d1" "dflash_d1.py x 3 arms: D1"
  grab_all d1-ref '^\| |^D1 |^  FAIL' 40
  finish
}

# ======================================================================================
row 55 "spec 19c - DFlash in the decode loop: the taps in the verify lists (runtime::build_verify_taps, M = 1..8 over 8 GDN slots), the commit-side projection of the accepted rows, the prefill tail (the window's taps through Context::copy_2d, projected 8 rows a replay), runtime::dflash::DFlashDrafter behind runtime::TapDrafter, host acceptance with the drafter's sparse q (server::accept_sparse), --spec dflash in b70-serve (AdaptiveK over MtpCost::dflash) and b70-decode --ids; D0, D2, D3, D4 (plan 19c)"
rownote 55 "Written blind on the Mac after row 54 (19b's drafter, itself unvalidated): nothing of 19c has run on a card. 7 new binaries (kernel_cmdlines +7 / -0 / ~0): gdn_step_slots_M1..4_N8 (the 8-slot verify build at the rows spec 8's lists ran at 4 slots) and the bf16 lm_head at M = 5 / 6 / 8. Nothing binds them without --spec dflash: D0 is G0. On the Mac (indicative): tools/mac/clrun/gdn_slots_run - gdn_step's 8-slot build at M = 8 and 5 bitwise to the plain kernel stepping the same rows, and a partial acceptance (j = 0 / 3 / 6 / 7 of 7, 2 of 4) leaves the live slot exactly the state after row j, with 3 new rows from it equal to the plain run (a mutant committing j + 1 is caught)."
rownote 55 "How the GDN state ends at the last accepted token (the operator's upstream question, vllm-xpu-kernels 78799fe's per-token replay state): the verify list writes the state after EVERY row into its own slot ((live + m) % 8) and commit(j) only moves Control::gdn_live to row j's slot - no copy, no replay; the rejected rows' slots and conv-ring positions are never read before a later list rewrites them. The cost is M x 151 MB of state writes per verify (1.2 GB at M = 8, ~2 ms at 600 GB/s, derived) and 1.06 GB of slots; a replay-entry scheme (per row: corr, normalised k, decay - ~2.4 MB at M = 8 - then one rank-1 replay pass at commit, ~0.5 ms) is recorded as a lever in spec 19 §11, not built."
rownote 55 "Data: the gate checkpoint (qwen) and the drafter (dflash: hf download z-lab/Qwen3.8-27B-DFlash2, 3.85 GB - row 54's). r55.engine and r55.gpu load both (~19 GB + 2.1 GB int8 drafter + 1.06 GB slots at max_len 16384). Hand back: every stage's lines - L (launch counts, the memory line), T1 / D3 per row (cosines and the bitwise marks), P per j (the live slot, the ring rows, the 64 ids), D2 per K (acceptance, ids an iteration), RF3, PF (split / chunked / decode cosines), dflash_gpu_test's per-prompt D2 lines and its S line, r55.serve's IDENTICAL / RATIO lines and the 'spec dflash:' / 'dflash:' server lines, r55.tps's generate lines."
stage r55.k0 55 default cpu - g0.sha,g0.bitwise,g0.suite "D0: no existing command line moved (kernel_cmdlines +7 / -0 / ~0), every pre-existing binary identical, the suite (G0) - incl. 19c's edits to shared files: capture.cc (a new mode; Plain / Verify / Stage walks unchanged), runtime::Engine (a defaulted TapDrafter*), step_chunk (a defaulted PrefillTaps*), the planner (a drafter term), EngineAdapter (a defaulted dflash flag), the loader's DraftVocabSpec::for_drafter"
st_r55_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r55.host 55 default cpu - - "host: dflash_engine_host_test (the tap sites, the 8 slots, the planner with a drafter, MtpCost::dflash and AdaptiveK at K <= 7), dflash_accept_test (D4: sparse-q acceptance, 10^6 seeded draws per case, chi-square >= 0.01, no draft outside the candidates, the residual finite), dflash_plan_test (19b's, now with the engine's slots and prefill taps), spec_accept_test / adaptive_k_test unchanged; b70-serve / b70-decode --spec dflash's refusals before the device"
st_r55_host() {
  run_tests '^(dflash_engine_host_test|dflash_accept_test|dflash_plan_test|spec_accept_test|adaptive_k_test|cli_reject_(serve_)?dflash_.*|cli_reject_serve_spec_dflash_other)$'
  jgrab host '^(dflash_engine_host_test|dflash_accept_test)$' 'p-value|accepted|adaptive K|plan with|OK|FAIL'
  finish
}
stage r55.kernels 55 default gpu - - "the 7 new binaries built (kbins: the first ocloc build of gdn_step_slots_M1..4_N8 and the bf16 head at M 5 / 6 / 8)"
st_r55_kernels() {
  kbins gdn_step_slots_M1_N8 gdn_step_slots_M2_N8 gdn_step_slots_M3_N8 gdn_step_slots_M4_N8 \
    gemv_bf16_M5_K5120_N248320 gemv_bf16_M6_K5120_N248320 gemv_bf16_M8_K5120_N248320
  finish
}
stage r55.engine 55 default gpu qwen,dflash r54.draft,r55.kernels "dflash_engine_test - the engine with the real drafter (int8, int8 head, bf16 KV): L (verify M = 1..8 = 774 launches each, draft 87, projection 8, the memory line), T1 (taps of M rows vs M = 1, the commit's ring rows vs the plain steps'), D3 (M = 1..8 logits and GDN slots vs M = 1, cosine >= 0.99999, bitwise recorded), P (partial acceptance at K 7 and 4, every j: the live GDN slot = the state after row j, the rejected rows' ring positions untouched, 64 further ids = plain), D2 on the card (the greedy loop at K 7 / 3 / 1 = plain), RF3 (the last positions of max_len), PF (the prefill tail: split at 1024 bitwise, chunks of 1000 and the decode path by cosine)"
st_r55_engine() {
  run_tests '^dflash_engine_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  jgrab engine '^dflash_engine_test$' '^L:|^memory:|D3 k=7|T1 k=|  P k=|^T1 \+ D3|  D2 K=|^RF3|^PF|^dflash_engine_test' 120
  finish
}
stage r55.gpu 55 default gpu qwen,dflash r55.engine "dflash_gpu_test through EngineAdapter (what b70-serve calls): D2 at K 7 / 3 / AdaptiveK on the golden and 36 A4 prompts = the plain loop (identity), truncate_to inside a burst, a seeded sampled run reproducible; acceptance and ids an iteration per prompt"
st_r55_gpu() {
  run_tests '^dflash_gpu_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  jgrab gpu '^dflash_gpu_test$' '^D2|^RF1|^S:|^dflash_gpu_test' 140
  finish
}
stage r55.serve 55 default gpu qwen,dflash r55.gpu "D2 by b70-serve: the 36 A4 tool-call scenarios and the golden prompts through --spec off / --spec dflash (auto K) / --spec dflash --spec-max 3, greedy, max_tokens 192 - output identical to --spec off; a seeded sampled request twice; the 'spec dflash:' startup line and the per-request 'dflash:' lines"
st_r55_serve() {
  local dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  serve_arms "$STATE/$STAGE" 1 "$SNAP_QWEN" "--set toolcall --set golden --max-tokens 192 --stop-at-eos" \
    "off|--max-len 16384 --spec off" "dflash|--max-len 16384 --spec dflash $dm" \
    "dflash3|--max-len 16384 --spec dflash --spec-max 3 $dm"
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output through --spec dflash identical to --spec off"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref off --metric wall_tps"
  x "PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $SNAP_QWEN --max-len 16384 --spec dflash $dm"
  step_rc $? "b70-serve --spec dflash: serves; the seeded sampled request twice"
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all ratio '^RATIO ' 10
  grab_all dflash 'spec dflash:|dflash:|SEEDED|memory:' 16
  finish
}
stage r55.decode 55 default gpu qwen,dflash r55.engine "D2 without the server: b70-decode --ids (code, prose, the 4395-id long prompt) --prefill --n 256, plain / --spec dflash (K 7) / --spec dflash --spec-max 3 / --draft-format bf16 - the outputs identical; the 'dflash:' acceptance lines"
st_r55_decode() {
  local p arm f a dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  chk "python3 tools/box_validate/long_ids.py 4395 $STATE/$STAGE.long.ids" "the 4395-id prompt"
  for p in code prose long; do
    f=tests/golden/prompts/$p.ids; [ "$p" = long ] && f=$STATE/$STAGE.long.ids
    for arm in off k7 k3 bf16; do
      case "$arm" in
        off) a="" ;; k7) a="--spec dflash $dm" ;; k3) a="--spec dflash --spec-max 3 $dm" ;; bf16) a="--spec dflash --draft-format bf16 $dm" ;;
      esac
      chk "build/src/cli/b70-decode $SNAP_QWEN --ids $f --n 256 --prefill --lm-head int8 $a > $STATE/$STAGE.$p.$arm.out" "b70-decode $p $arm"
    done
    chk "cmp $STATE/$STAGE.$p.off.out $STATE/$STAGE.$p.k7.out && cmp $STATE/$STAGE.$p.off.out $STATE/$STAGE.$p.k3.out && cmp $STATE/$STAGE.$p.off.out $STATE/$STAGE.$p.bf16.out && echo 'GREEDY identical: $p plain / dflash K 7 / K 3 / bf16 drafter'" "greedy ids identical with and without the drafter ($p)"
  done
  grab_all dflash '^dflash:|^generate:|GREEDY' 30
  finish
}
stage r55.tps 55 optin gpu-self qwen,dflash r55.decode "the speed record, informational (D5 proper is 19d's): b70-decode --ids --prefill --n 512 plain vs --spec dflash on code and on an A4 scenario, 3 rounds in rotated order after a warm-up - the generate t/s and the acceptance lines"
st_r55_tps() {
  local r arm p a order dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  idle before
  for r in 0 1 2 3; do
    order="off dflash"; [ $((r % 2)) = 1 ] && order="dflash off"
    for p in tests/golden/prompts/code.ids tests/golden/toolcall/t3_rename-openai.ids; do
      for arm in $order; do
        a=""; [ "$arm" = dflash ] && a="--spec dflash $dm"
        x "build/src/cli/b70-decode $SNAP_QWEN --ids $p --n 512 --prefill --lm-head int8 $a > /dev/null 2> $STATE/$STAGE.r$r.$arm.$(basename $p .ids).err"
        step_rc $? "round $r $arm $(basename $p)"
      done
    done
  done
  idle after
  grab_all tps '^generate:|^dflash:' 40
  finish
}

# ======================================================================================
row 56 "spec 19d - DFlash in the server: the drafter's ring in the prefix cache's snapshots (decision 5 as ruled: the whole ring after the state, the store keyed by the drafter), the batching scheduler's speculative hook (per-slot rings, a mock engine), --spec dflash --pp 2 (the drafter on device 1, tap verify stage lists, the pipeline prefill's taps), D5 against --mtp auto (plan 19d)"
rownote 56 "Written blind on the Mac after rows 54 / 55 (themselves unvalidated): nothing of 19d has run on a card. No new binary (kernel_cmdlines +0 / -0 / ~0): the --pp 2 tap verify stage lists bind 19c's gdn_step_slots_M<M>_N8 and B70_VERIFY_M8's M = 5..8 variants per stage. Nothing changes without --spec dflash: D0 is G0 (shared edits: Engine's snapshots gain the ring only with a drafter, PipelineEngine / place_stages / pp_plan gain defaulted drafter arguments, capture.cc a new mode, step_stage a defaulted PrefillTaps*, the prefix store's root store_root(kv_form, 0) = kv_form)."
rownote 56 "Decision 5 as ruled in 19d (spec 19 §12): every snapshot carries the drafter's WHOLE ring (41,943,040 B, +25 % a snapshot) - a restore is a copy, bitwise by construction; a mid-call block end's snapshot holds the ring as that call left it (positions outside the call's last 2048 not yet projected): lower acceptance after a restore there with a short tail, never another output (r56.prefix's R2 caveat line counts it)."
rownote 56 "Data: the gate checkpoint (qwen), the drafter (dflash), both cards for r56.pp / r56.pp_serve / r56.d5_pp, uvx for r56.d5_benchy, OPENCODE_LOG for r56.d5_opencode (the recording may not exist yet - the stage then SKIPs), oracle-out-19a/draft_vocab.ranked.ids (--push-data from the Mac; IN-SAMPLE on A4 / golden) for r56.d5_dv. Hand back: every stage's lines - the K / R1 / R2 lines of dflash_prefix_gpu_test, dflash_engine_test's SR and L lines, pp_dflash_test's bitwise lines per split, the serve rows' IDENTICAL / RATIO tables and 'dflash:' lines, llama-benchy's tables, the replay summaries. D5's bar (spec 19 §6): --spec dflash becomes the Qwen3.8 default only at >= +10 % over --mtp auto on the opencode recording and no loss > 3 % elsewhere - the operator decides from r56.d5_opencode; the A4 / golden rows (r56.d5_a4) are the expectation, not the bar."
stage r56.k0 56 default cpu - g0.sha,g0.bitwise,g0.suite "D0: no command line moved (kernel_cmdlines +0 / -0 / ~0), every pre-existing binary identical, the suite (G0) - incl. 19d's shared edits (Engine / PipelineEngine snapshots, capture.cc's StageVerifyTaps mode, step_stage, place_stages, pp_plan, the prefix store's root, BatchEngineIface / Scheduler's speculative round)"
st_r56_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r56.host 56 default cpu - - "host: dflash_prefix_test (the ring across a PrefixSession restore, the store key; the ring-less mutant caught), dflash_scheduler_test (speculation under the batching scheduler: lossless, per-request records, ends inside a run, a restore onto another slot), pp_dflash_serve_test (the --pp 2 adapter's DFlash path), pp_serve_plan_test (the drafter's terms on two cards), dflash_plan_test (the ring and key rows), scheduler_test / prefix_cache_test / pp_serve_test unchanged; the --spec dflash --pp 2 refusals (peer, b70-decode)"
st_r56_host() {
  run_tests '^(dflash_prefix_test|dflash_scheduler_test|pp_dflash_serve_test|pp_serve_plan_test|dflash_plan_test|scheduler_test|prefix_cache_test|pp_serve_test|cli_reject_(serve_)?dflash_pp)$'
  jgrab host '^(dflash_prefix_test|dflash_scheduler_test|pp_dflash_serve_test)$' '^(K|R1|R2|S[1-5]|[1-4])[:.] |OK$|FAIL' 40
  finish
}
stage r56.engine 56 default gpu qwen,dflash r55.engine "dflash_engine_test again (19d's L and SR: the snapshot is the state plus the ring - 41,943,040 B - and a reset + restore brings the ring back bitwise; T1 / D3 / P / D2 / RF3 / PF as row 55)"
st_r56_engine() {
  run_tests '^dflash_engine_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  jgrab engine '^dflash_engine_test$' '^L:|^SR |^T1 \+ D3|  D2 K=|^RF3|^PF|^dflash_engine_test' 60
  finish
}
stage r56.prefix 56 default gpu qwen,dflash r56.engine "dflash_prefix_gpu_test (decision 5 on the card, 19c's deferred restore-and-continue test): K (the store keyed by the drafter), R1 (a turn restored from the previous turn's request-end snapshot after a side request: the ring after the tail BITWISE the uninterrupted session's, the next 128 ids and every iteration's acceptance identical; the cold-ring control's acceptance printed), R2 (a block-end restore with a 2500-id tail = a cold run's ring bitwise; the 300-id caveat counted)"
st_r56_prefix() {
  run_tests '^dflash_prefix_gpu_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  jgrab prefix '^dflash_prefix_gpu_test$' '^K:|^R1|^R2|turn of|^dflash_prefix_gpu_test' 60
  finish
}
stage r56.pp 56 default gpu qwen,dflash r56.engine "pp_dflash_test on both cards (ZE_AFFINITY_MASK=0,1): --spec dflash across the split BITWISE the one-card drafter at the auto split and at 20 - the prefill's ring, every verify's rows, the drafts and ids at K 7, the session with the ring, the hooked snapshots at 2048 / 4096 / 4103; a one-card snapshot continued on two cards"
st_r56_pp() {
  run_tests '^pp_dflash_test$' dflash '' "B70_DFLASH_SNAPSHOT=${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2} ZE_AFFINITY_MASK=0,1"
  jgrab pp '^pp_dflash_test$' '^reference|^--pp 2|  ids|bitwise|DIFFERS|continued|PASS|FAIL' 60
  finish
}
stage r56.pp_serve 56 default gpu qwen,dflash r56.pp "D2 by b70-serve --pp 2 --spec dflash (both cards): the A4 + golden sets greedy, max_tokens 192, against one card's --spec off - identical; the plan line (the drafter on device 1, the embedding replica), both memory lines, the 'spec dflash (--pp 2' line"
st_r56_pp_serve() {
  local dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}" c="--set toolcall --set golden --max-tokens 192 --stop-at-eos"
  chk "PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE/off.r1 off 1 $SNAP_QWEN --max-len 16384 --spec off -- $c" "one card, --spec off"
  chk "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE/pp2dflash.r1 pp2dflash 1 $SNAP_QWEN --max-len 16384 --pp 2 --spec dflash $dm -- $c" "--pp 2 --spec dflash"
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output through --pp 2 --spec dflash identical to one card's --spec off"
  grab_all identical '^(IDENTICAL|NOT IDENTICAL|DIFFER)' 20
  grab_all plan 'pipeline plan|DFlash:|memory, device|spec dflash' 12
  finish
}
stage r56.d5_a4 56 optin gpu qwen,dflash r56.prefix "D5's expectation on the A4 + golden sets (the recording is the bar): b70-serve --spec off / --mtp auto / --spec dflash (auto K) / --spec dflash --spec-max 3, the same flags otherwise (--max-len 16384, the int8 head, bf16 KV, the prefix cache's default), greedy, max_tokens 256, 3 rounds in rotated order after an idle check - per-arm decode t/s (the server's decode-only rate) and the RATIO lines dflash / mtp-auto; greedy output identical across the arms"
st_r56_d5_a4() {
  local dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  idle before
  serve_arms "$STATE/$STAGE" "${R56_ROUNDS:-3}" "$SNAP_QWEN" "--set toolcall --set golden --max-tokens 256 --stop-at-eos" \
    "off|--max-len 16384 --spec off" "mtp|--max-len 16384 --mtp auto" "dflash|--max-len 16384 --spec dflash $dm" \
    "dflash3|--max-len 16384 --spec dflash --spec-max 3 $dm"
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref off" "greedy output identical across off / mtp auto / dflash"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref mtp --ratio dflash/mtp --ratio dflash3/mtp --ratio dflash/off"
  grab_all ratio '^RATIO ' 20
  grab_all dflash 'spec dflash:|^dflash:|^mtp:' 24
  grab_all idle '^IDLE '
  finish
}
stage r56.d5_dv 56 optin gpu qwen,dflash,oracle_dflash_rank r56.d5_a4 "the draft vocabulary arm: --spec dflash against --spec dflash --draft-vocab 32k --draft-vocab-ids oracle-out-19a/draft_vocab.ranked.ids (ranked IN-SAMPLE on these sets - an upper bound) and --draft-vocab 32k by id, A4 + golden, 3 rotated rounds - decode t/s ratios and acceptance; output identical"
st_r56_d5_dv() {
  local dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  idle before
  serve_arms "$STATE/$STAGE" "${R56_ROUNDS:-3}" "$SNAP_QWEN" "--set toolcall --set golden --max-tokens 256 --stop-at-eos" \
    "dflash|--max-len 16384 --spec dflash $dm" \
    "dv32k-r|--max-len 16384 --spec dflash --draft-vocab 32k --draft-vocab-ids oracle-out-19a/draft_vocab.ranked.ids $dm" \
    "dv32k|--max-len 16384 --spec dflash --draft-vocab 32k $dm"
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref dflash" "greedy output identical with the draft vocabulary"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref dflash --ratio dv32k-r/dflash --ratio dv32k/dflash"
  grab_all ratio '^RATIO ' 20
  grab_all dflash 'spec dflash:|^dflash:|draft vocab' 24
  finish
}
stage r56.d5_benchy 56 optin gpu-self qwen,dflash,uvx r56.d5_a4 "D5 with llama-benchy at the operator's flags (row 41's matrix A at concurrency 1 and matrix D: --pp 512 4096 --tg 128 --runs 2; --pp 512 --tg 128 --depth 4096 32768 --no-cache), b70-serve --max-len 40960 --spec off / --mtp auto / --spec dflash, 3 rounds in a Latin square (every arm in every position once), idle checks around them"
st_r56_d5_benchy() {
  local v="${BENCHY_VERSION:-0.4.0}" r arm args order dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  local a="--pp 512 4096 --tg 128 --concurrency 1 --runs 2 --format md"
  local d="--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache --format md"
  idle before
  for r in 1 2 3; do   # a Latin square: every arm in every position once
    case "$r" in 1) order="off mtp dflash" ;; 2) order="mtp dflash off" ;; 3) order="dflash off mtp" ;; esac
    for arm in $order; do
      case "$arm" in off) args="--spec off" ;; mtp) args="--mtp auto" ;; dflash) args="--spec dflash $dm" ;; esac
      chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_ARGS='$args' tools/probe/serve_benchy.sh $a" "round $r $arm, matrix A (c1)"
      chk "BENCHY_VERSION=$v MODEL=$SNAP_QWEN MAX_LEN=40960 SERVE_ARGS='$args' tools/probe/serve_benchy.sh $d" "round $r $arm, matrix D"
    done
  done
  idle after
  grab_all benchy '^(client: |\| )' 200
  grab_all idle '^IDLE '
  finish
}
stage r56.d5_opencode 56 optin gpu-self qwen,dflash,opencode_log r56.d5_a4 "D5 proper (spec 19 §6): the opencode recording (OPENCODE_LOG) replayed through b70-serve with the prefix cache on (replay_ab.sh, max_len 131072), --spec off / --mtp auto / --spec dflash, 3 rounds in a Latin square, idle checks around them - the per-arm t/s and ttft summaries, and per request the prefix plan beside the dflash acceptance; the bar is dflash >= 1.10 x --mtp auto here"
st_r56_d5_opencode() {
  local r arm args order dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  idle before
  for r in 1 2 3; do   # a Latin square: every arm in every position once
    case "$r" in 1) order="off mtp dflash" ;; 2) order="mtp dflash off" ;; 3) order="dflash off mtp" ;; esac
    for arm in $order; do
      case "$arm" in off) args="--spec off" ;; mtp) args="--mtp auto" ;; dflash) args="--spec dflash $dm" ;; esac
      chk "MODEL=$SNAP_QWEN ARMS=on SERVE_ARGS='$args' tools/prefix/replay_ab.sh $OPENCODE_LOG $STATE/r56.d5.r$r.$arm" "replay round $r, $arm"
    done
  done
  idle after
  # Per request: the prefix plan (cold / continue / restore and where) beside the dflash acceptance - whether restores
  # at mid-call block ends (their ring not the session's, spec 19 §12) cost acceptance.
  x "grep -hE '^prefix: |^dflash: ' $STATE/r56.d5.r1.dflash.on.serve.log | head -400 > $STATE/r56.d5.restore_vs_accept.txt; wc -l $STATE/r56.d5.restore_vs_accept.txt"
  grab_all replay 'tok/s|t/s|requests|ttft' 60
  grab_all idle '^IDLE '
  finish
}
stage r56.d5_pp 56 optin gpu qwen,dflash r56.pp_serve "the --pp 2 rows (both cards): b70-serve --pp 2 --mtp auto against --pp 2 --spec dflash on A4 + golden, 3 rounds in rotated order - decode t/s, RATIO dflash / mtp; output identical"
st_r56_d5_pp() {
  local r arm flags order c="--set toolcall --set golden --max-tokens 256 --stop-at-eos" dm="--draft-model ${SNAP_DFLASH:-z-lab/Qwen3.8-27B-DFlash2}"
  idle before
  for r in 1 2 3; do
    order="mtp dflash"; [ $((r % 2)) = 0 ] && order="dflash mtp"
    for arm in $order; do
      flags="--max-len 16384 --pp 2 --mtp auto"; [ "$arm" = dflash ] && flags="--max-len 16384 --pp 2 --spec dflash $dm"
      chk "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_run.sh $STATE/$STAGE/$arm.r$r $arm $r $SNAP_QWEN $flags -- $c" "arm $arm, round $r"
    done
  done
  idle after
  chk "python3 tools/box_validate/serve_client.py compare --dir $STATE/$STAGE --ref mtp" "greedy output identical"
  x "python3 tools/box_validate/serve_client.py table --dir $STATE/$STAGE --ref mtp --ratio dflash/mtp"
  grab_all ratio '^RATIO ' 10
  finish
}

# ======================================================================================
row 57 "spec 25a - Nemotron 3.5 Bolt: the facts, the CPU reference (B1), golden sets, the A4 references, the tap dumps, gate Q (plan 25a)"
rownote 57 "Download first (pinned; df -h ~ first; MemAvailable >= 48 GB): hf download SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym --revision b9f381d20cbd4cede9f729c2647d2542c4a35997 (41.9 GB). r57.host needs only its pinned small files (the same command with --exclude '*.safetensors')."
rownote 57 "The BF16 original (hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 --revision 567021037ff88364143e24ea05e6abd0f72086ff, 149.8 GB; MemAvailable >= 96 GB) is downloaded on the box only, for the Bolt BF16 window (docs/box-day-plan.md, Session 12, with row 58) and removed after it: r57.golden_bf16, r57.a4_bf16 and r57.gateq run there. Every other row 57 stage needs only the g128 form."
rownote 57 "Box CPU only, no card: nothing here takes the GPU lock. Every stage runs tools/box_validate/bolt_oracle.sh in the oracle image with transformers 5.19.0 from <tree>/oracle-out-q4exp-site (tools/oracle/qwen4exp_env.sh). The real stages SKIP 77 'missing data' without the checkpoints, or while they download."
rownote 57 "Hand back: B70_BOLT_TIE_TOL's evidence (the 'MoE 9th/10th sel gap' lines r57.golden prints, per block and over all), the SSM-state magnitude lines, each prompt's wall / RSS, the perplexities per form, hfcheck's verdict and its fallback lines, the A4 parse failures (score.py --format qwen-think, bar 0 for the BF16 reference) and agreement, gate Q's table (a recorded measurement, Decision 3)."
stage r57.host       57 default cpu  bolt_small,oracle_image       -  "bolt_oracle.sh tests: test_bolt_ref.py, test_bolt_make_tiny.py, bolt_tiny_check.py, the toolcall tests (the box image's torch: the bitwise claims re-checked there)"
st_r57_host() {
  chk "tools/box_validate/bolt_oracle.sh $DATA tests" "the Bolt reference tests (tiny puzzles from the pinned small files)"
  grab_all tests '^ok |^  |FAIL|SKIP|passed|tiny check' 160
  finish
}
stage r57.ppl        57 optin   cpu  bolt_g128,oracle_image       -  "perplexity on prose / code / agentic, g128 and BF16 (finite; code < 10 ESTIMATED; recorded per text)"
st_r57_ppl() {
  chk "tools/box_validate/bolt_oracle.sh $DATA ppl" "the reference's perplexity on each form present"
  grab_all ppl '^ppl |^wall|not sane|NOT SANE' 12
  finish
}
stage r57.hfcheck    57 optin   cpu  bolt_g128,oracle_image       -  "hfcheck --layers 12: the port = its restated chain BITWISE on real weights (fp32 when RAM allows), R's torch fallback recorded"
st_r57_hfcheck() {
  chk "tools/box_validate/bolt_oracle.sh $DATA hfcheck" "hfcheck on each form present"
  grab_all hfcheck '^hfcheck' 16
  finish
}
stage r57.golden     57 optin   cpu  bolt_g128,oracle_image       -  "golden sets on SergiioB's g128 dequantised -> oracle-out-bolt (gap lines per block, --taps)"
st_r57_golden() {
  chk "tools/box_validate/bolt_oracle.sh $DATA golden" "the Bolt golden sets on the g128 form"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all gap 'sel gap|SSM state|wall' 240
  finish
}
stage r57.golden_bf16 57 optin  cpu  bolt_bf16,oracle_image       -  "golden sets on the BF16 original -> oracle-out-bolt-bf16 (hours)"
st_r57_golden_bf16() {
  chk "tools/box_validate/bolt_oracle.sh $DATA golden-bf16" "the Bolt golden sets on the BF16 original"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all gap 'sel gap|SSM state|wall' 120
  finish
}
stage r57.a4_ref     57 optin   cpu  bolt_g128,oracle_image       -  "the A4 reference on the g128 form -> oracle-out-bolt-a4 (the engine's tool-call reference; days, resumable)"
st_r57_a4_ref() {
  chk "tools/box_validate/bolt_oracle.sh $DATA a4" "the A4 reference on the g128 form (36 scenarios, 2048 new ids, batched)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all a4 'scenarios|parse failure|match|^int4:|^bf16:' 16
  finish
}
stage r57.a4_bf16    57 optin   cpu  bolt_bf16,oracle_image       r57.a4_ref  "the A4 reference on the BF16 original -> oracle-out-bolt-a4 (the BF16 window; days, resumable; scores both forms)"
st_r57_a4_bf16() {
  chk "tools/box_validate/bolt_oracle.sh $DATA a4-bf16" "the A4 reference on the BF16 original"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all a4 'scenarios|parse failure|match|^int4:|^bf16:' 16
  finish
}
stage r57.taps       57 optin   cpu  bolt_g128,oracle_image       r57.golden  "tap dumps for spec 26a -> oracle-out-bolt-taps (prose / code / agentic; a4/* after r57.a4_ref)"
st_r57_taps() {
  chk "tools/box_validate/bolt_oracle.sh $DATA taps" "the tap dumps (dump_taps' format)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  grab_all taps 'taps|sources|wall|SKIPPED' 40
  finish
}
stage r57.gateq      57 optin   cpu  bolt_a4_ref,oracle_image     r57.a4_bf16  "gate Q: g128 vs BF16 (KL / top-1 / A4), recorded"
st_r57_gateq() {
  chk "tools/box_validate/bolt_oracle.sh $DATA gateq" "gate Q (a recorded measurement, Decision 3)"
  grab_all gateq '^PASS|^FAIL|gate Q|A4 pooled|^kl ' 60
  finish
}

# ======================================================================================
row 59 "spec 25b - Nemotron 3.5 Bolt: BoltDesc, load_bolt (g128-expanded and native g64 into one layout, the '+:' / BF16-scale / signed-scale relaxations), synthetic checkpoints, the two-card plan (plan 25b)"
rownote 59 "No kernel touched (kernel_cmdlines +0 / -0 / ~0: G0's sha is the 'nothing moved' check) and no existing loader file but src/loader/quant.{h,cc} (the two exact relaxations: a restating '+:' dynamic rule accepted, BF16 GPTQ scales exact-or-refused; signed scales audited) - r59.host re-runs quant_test, ct_loader_test and dequant_fixture_test (the existing tests that read quant.cc) beside the new bolt tests - not the whole loader suite (r59 does not claim the other families' loaders). Written on the Mac: bolt_test, quant_test, bolt_repack_test, bolt_plan_test, bolt_names_test (the descriptor's 112,091 names against the published index, 849 vision tensors skipped by prefix) and bolt_synth_host_{g128,pair,g64}_test over three real-width synthetic checkpoints (make_synth.py --layers 8) all ran there; nothing of load_bolt has touched a card."
rownote 59 "r59.synth needs only the oracle image and the pinned small files (hf download SergiioB/Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym --revision b9f381d20cbd4cede9f729c2647d2542c4a35997 --exclude '*.safetensors'): make_synth.py draws random weights at the real widths, N 8 (~3.9 GB a checkpoint) and N 12 (~5.5 GB) each as SergiioB's g128 GPTQ spelling, its exact g64 expansion and a native g64 packing -> \$DATA/oracle-out-bolt-synth/n{8,12}-{g128,g64x,g64} + .ACCEPTED (~28 GB in all; the Mac: ~2 min a checkpoint at N 8, ~3 min at N 12, MEASURED). r59.load_real needs SergiioB's 41.9 GB checkpoint (data key bolt_g128, row 57's download) and BOTH cards (split 48, --max-len 262144 in the test's arguments): the plan, then every block uploaded, 0 unconsumed. r59.load_rtn needs 25q's export (data key bolt_rtn; not built)."
rownote 59 "Hand back: each load line (blocks, trunk, g128 / g64 mask, restating '+:' count, int8 / bf16 head, devices, bytes) and its seconds (the int8 head's quantisation time separately), every device's weights line against loader::bolt::device_weight_bytes and runtime::bolt::plan's, the sampled read-backs' counts, the real load's RSS and wall, and the plan lines (the two-card split by block bytes; 262144 positions)."
stage r59.host       59 default cpu -                         -        "host: quant_test (the relaxations), bolt_test, bolt_repack_test, bolt_plan_test, bolt_names_test, ct_loader_test, dequant_fixture_test"
st_r59_host() {
  run_tests '^(quant_test|ct_loader_test|dequant_fixture_test|bolt_test|bolt_repack_test|bolt_plan_test|bolt_names_test)$'
  jgrab host '^(quant_test|bolt_test|bolt_repack_test|bolt_plan_test|bolt_names_test)$' \
    'OK|exp_torch|negative|pp_split|device [01]|24 GB|one card|bytes on the cards|total|descriptor'
  finish
}
stage r59.synth      59 default cpu oracle_image              -        "make_synth.py N 8 and 12 in SergiioB's g128 spelling, its exact g64 expansion and a native g64; check.py ACCEPTED on each - bolt_synth.sh synth-ckpt"
st_r59_synth() {
  chk "tools/box_validate/bolt_synth.sh $DATA synth-ckpt" "the synthetic Bolt checkpoints (both spellings, N 8 and 12)"
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_bolt_synth' $STATE/have.env"
  grab_all synth '^ACCEPTED|REJECTED|^wrote|FAILED|^check:' 12
  finish
}
stage r59.pytest     59 default cpu oracle_image              -        "test_bolt_synth.py in the oracle image (the packer, the exact expansion, the config) - bolt_synth.sh synth-tests"
st_r59_pytest() {
  chk "tools/box_validate/bolt_synth.sh $DATA synth-tests" "test_bolt_synth.py"
  grab_all pytest '^PASS|^FAIL|^SKIP|tests OK|ok$' 20
  finish
}
stage r59.synth_host 59 default cpu bolt_synth               r59.synth "bolt_synth_host_{g128,pair,g64}_test over the N 8 synthetics: every name both ways, every block repacked, the sampled words = the shards', the two spellings byte-identical, 0 unconsumed"
st_r59_synth_host() {
  run_tests '^bolt_synth_host_(g128|pair|g64)_test$'
  jgrab synth-host '^bolt_synth_host_' 'OK|sampled|unconsumed|SKIP'
  finish
}
stage r59.load_synth 59 default gpu  bolt_synth               r59.synth "bolt_load_checkpoint_test on the N 12 synthetics: one card in all three spellings (g128, its exact g64 expansion, a native g64; int8 head), the bf16 head, --pp 2 split 6 over two cards (SKIP without the second card) - plan == allocation, sampled blocks read back equal to the host repack"
st_r59_load_synth() {
  run_tests '^bolt_load_synth_(g128|g64x|g64|bf16|pp2|pp2_g64x)_test$'
  jgrab load-synth '^bolt_load_synth' 'OK|weights|unconsumed|SKIP|plan device'
  finish
}
stage r59.load_real  59 optin   gpu  bolt_g128                -        "bolt_load_checkpoint_test on SergiioB's g128 export, two cards (split 48, --max-len 262144): the plan, 0 unconsumed, sampled read-backs, load time and RSS"
st_r59_load_real() {
  run_tests '^bolt_load_real_test$'
  jgrab load-real '^bolt_load_real_test$' 'OK|weights|unconsumed|plan device|load_bolt|load  '
  finish
}
stage r59.load_rtn   59 optin   gpu  bolt_rtn                 -        "bolt_load_checkpoint_test on 25q's native-g64 export, two cards: the same device bytes as g128-expanded where the weights agree (sampled)"
st_r59_load_rtn() {
  run_tests '^bolt_load_rtn_test$'
  jgrab load-rtn '^bolt_load_rtn_test$' 'OK|weights|unconsumed|plan device'
  finish
}

# ======================================================================================
row 60 "spec 25c - Nemotron 3.5 Bolt decode: P0 (653 / 533 skeletons, GEMV and SSM-state floors, the S sweep), the nb_* kernels, BoltEngine on two cards, the fused arms, the fp32 / bf16 state arm (plan 25c)"
rownote 60 "Written blind on the Mac (spec 25 §15): NO nb_* binary has been compiled by ocloc (kernel_cmdlines +115 / -0 / ~0: the gemv cells at Bolt's 7 int4 shapes x S x layout, the heads at 131072, P0's stand-ins, nb_state_rmw, the 38 family binaries incl. the arms, the bf16-state builds and test-only M = 4 builds). No existing .cl edited: G0's sha is K0. On the Mac's GPU (tools/mac/clrun/bolt_run, indicative) every kernel and arm agrees with tests/kernels/bolt_ref.h - exact, except Apple's not-correctly-rounded division (conv / gnorm <= 1 ulp, the route's fp32 words <= 2 ulp with ids exact); Muse's mg_attn_eager exact. The flash mg_attn path and the whole engine have never run anywhere."
rownote 60 "P0 first (Decision 16): r60.p0 runs probe_bolt_step before any engine stage - the 653-launch skeleton from EXISTING kernels at Bolt's shapes, the 533-launch fused skeleton, the GEMV-only list, 653 noops and the SSM-state floor (nb_state_rmw: 4 MiB fp32 / 2 MiB bf16 a Mamba block), interleaved, at depth 4096 and 65536, --sweep-s (every int4 shape at S 1/2/4/8/16 x L 0/1, probe_gemv's 3 % rule) and --pair (the two-launch norm). Stand-ins, PROPOSED scales: moe.cl at E 64 (a 64-block pool + the shared slot a MoE block; up x I*544/2,228,224, down x I*544/1,114,112), gdn_step at Qwen3.8's shape (x 4/3), lfm_res_norm as the one-launch norm; conv / gnorm / route / sh_act / attn_prep / embed are noops. Write the table into docs/probe-bolt-2026-10-21.md (a 'P0' section). The S picks become model::BoltDesc::linears' {S, layout} (25b's PROVISIONAL S 1 until then; the block already builds every cell); every arm whose skeleton saving is >= 1 % of the step is A/B'd in r60.fuse / r60.speed (Decision 16, PROPOSED)."
rownote 60 "Every stage on the REAL checkpoint needs BOTH cards (Decision 4; split 48 by bytes): its commands set ZE_AFFINITY_MASK=0,1 themselves. The N 12 synthetics (row 59's r59.synth, data key bolt_synth) run on one card, --pp 6,4 and pp2:6 on two. Their golden sets come from r60.synth_golden (bolt_oracle.sh export on n12-g128 / n12-g64x -> \$DATA/oracle-out-bolt-n12-{g128,g64x}, short / 4k / agentic, CPU, hours ESTIMATED; data key oracle_bolt_synth). SergiioB's golden sets are row 57's r57.golden (oracle_bolt)."
rownote 60 "Tolerances (all PROPOSED): B70_BOLT_TIE_TOL = 1e-3 on the 9th / 10th sel until row 57's gap distribution on the real weights (an exact 0 gap is undetermined); h taps median >= 0.9998 / min >= 0.99 a block; the SSM state and the conv columns after the prompt cosine >= 0.999 a block (not bitwise: the int4 GEMV's fp32 order vs the reference's bf16 linear); flash against fp64 at cosine 0.99999. Decision 2's A4 subset needs b70-serve (spec 25e, not built): r60.state_arm records B3 only."
rownote 60 "Hand back: P0's tables (step per list, gap per launch, the floors, the S picks, the PROPOSED corrected step); bolt_kernels_test's lines; every plan == allocation / launches line (653 / 645 eager / 533 all arms / 655 peer; the synthetic 92); the B3 gate / routing / SSM / h lines per prompt; the arms' bitwise lines and their A/B ratios; the speed rows and the --profile table."
stage r60.k0        60 default cpu -                         g0.sha,g0.bitwise,g0.suite "K0 (B0): every pre-existing binary identical; kernel_cmdlines +115 / -0 / ~0 (no existing .cl edited)"
st_r60_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r60.host      60 default cpu -                         -        "host: probe_bolt_step_shapes_test, bolt_ref_test (the twin against the fixture), bolt_variant_names_test, bolt_plan_test (the launch counts, the decode scratch)"
st_r60_host() {
  run_tests '^(probe_bolt_step_shapes_test|bolt_ref_test|bolt_variant_names_test|bolt_plan_test)$'
  jgrab host '^(probe_bolt_step_shapes_test|bolt_ref_test|bolt_variant_names_test|bolt_plan_test)$' \
    'OK|ok|bitwise|differ|launches|scratch|lists'
  finish
}
stage r60.p0        60 default gpu -                         r60.k0   "P0: probe_bolt_step at depth 4096 / 65536, interleaved a / fused / gemv / noop / state, --sweep-s, --pair: the table into the facts sheet"
st_r60_p0() {
  idle r60-p0
  x "uptime"
  chk "build/tools/probe/probe_bolt_step --depth 4096 --rounds 9 --sweep-s --pair > $STATE/r60-p0-4096.md" "P0 at depth 4096 (+ the S sweep, the norm pair)"
  chk "build/tools/probe/probe_bolt_step --depth 65536 --rounds 9 > $STATE/r60-p0-65536.md" "P0 at depth 65536"
  x "cat $STATE/r60-p0-4096.md $STATE/r60-p0-65536.md"
  grab_all p0 '^\| (a|fused|gemv|noop|state) \||^- step|gap per launch|roofline|PROPOSED step|\*\*pick\*\*|--pair' 80
  finish
}
stage r60.k1        60 default gpu -                         r60.k0   "kbins of the block + bolt_kernels_test: B2 on every kernel and arm (bitwise bolt_ref.h; flash at cosine 0.99999 against fp64)"
st_r60_k1() {
  kbins nb_res_norm_M1_K4096_W8_NF nb_res_norm_M1_K4096_W8_S1 nb_res_norm_M1_K4096_W8_S1_T1 nb_res_norm_M1_K4096_W8_S1_R \
    nb_sh_act_M1_L1024_SH5376_S1 nb_attn_prep_M1_KV2_S1 nb_embed_gather_M1_K4096_V131072 nb_conv_M1_C10240_S1 \
    nb_ssm_M1_H128_P64_N128_G8_S1 nb_ssm_M1_H128_P64_N128_G8_S1_C nb_ssm_M1_H128_P64_N128_G8_S1_B16 \
    nb_ssm_M1_H128_P64_N128_G8_S1_C_B16 nb_gnorm_M1_D8192_G1024_S1 nb_router_M1_K4096 nb_route_M1_T9 \
    nb_moe_up_M1_I2560_L1024_T9 nb_moe_up_M1_I2560_L1024_T9_R nb_moe_down_M1_I2560_L1024_T9 \
    nb_argmax_stage1_M1_V131072 nb_argmax_stage2_V131072 gemv_i8w_M1_K4096_N131072 gemv_bf16_M1_K4096_N131072 \
    mg_attn_M1_T64_Q32KV2_F mg_attn_eager_M1_T64_Q32KV2_F nb_state_rmw_F32 nb_state_rmw_B16
  run_tests '^bolt_kernels_test$'
  jgrab k1 '^bolt_kernels_test$' 'bitwise|cosine|within|arm|OK'
  finish
}
stage r60.reject    60 default gpu -                         -        "cli_reject_bolt_* (prefill, prefill-length, mtp, spec, kv8) - refused before the device"
st_r60_reject() {
  run_tests '^cli_reject_bolt_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r60.synth_golden 60 default cpu oracle_image,bolt_synth -       "the synthetics' golden sets: bolt_oracle.sh export on oracle-out-bolt-synth/n12-g128 and n12-g64x (short, 4k, agentic) -> \$DATA/oracle-out-bolt-n12-{g128,g64x}; then re-links oracle-out*"
st_r60_synth_golden() {
  local s
  for s in n12-g128 n12-g64x; do
    chk "BOLT_PROMPTS='bolt_short bolt_4k bolt_agentic' tools/box_validate/bolt_oracle.sh $DATA export $DATA/oracle-out-bolt-synth/$s" \
      "the synthetic ($s) golden sets"
  done
  x "tools/box_validate/data.sh link $DATA $TREE $BASE"
  x "tools/box_validate/data.sh have > $STATE/have.env; grep -E '^HAVE_oracle_bolt_synth' $STATE/have.env"
  grab_all gap 'gap|^wall' 20
  finish
}
stage r60.synth     60 default gpu  bolt_synth               r60.k1   "bolt_decode_test on the N 12 synthetics, one card: plan == allocation, 92 launches, two runs bitwise (tokens, logits, routes, rings, SSM states, KV, both residuals, every tap); eager, bf16 head, bf16 state; --fuse router / route / conv / all bitwise none"
st_r60_synth() {
  run_tests '^bolt_decode_synth_(n12-g128|n12-g64x|eager|bf16head|b16state|fuse_(router|route|conv|all))_test$'
  jgrab synth '^bolt_decode_synth' 'plan == allocation|launches|B4|Review Focus|residual|OK'
  finish
}
stage r60.synth_b3  60 default gpu  bolt_synth,oracle_bolt_synth r60.synth "bolt_golden_test on the synthetics: tokens tie-aware, routes within B70_BOLT_TIE_TOL, SSM state / conv columns by cosine, h taps - short / 4k / agentic x both spellings; bf16 head, eager"
st_r60_synth_b3() {
  run_tests '^bolt_golden_synth_n12-(g128|g64x)_bolt_(short|4k|agentic)_test$|^bolt_golden_synth_short_(bf16head|eager)_test$'
  jgrab synth-b3 '^bolt_golden_synth' 'gate:|routing:|SSM state|conv|^  h |step [0-9]+:|OK'
  finish
}
stage r60.synth_pp  60 default gpu  bolt_synth               r60.synth "two cards on the synthetic: bolt_decode_test --pp 6,4 (copy at 6, peer at 4: bitwise each other AND bitwise one card) + the golden gate across split 6 (peer)"
st_r60_synth_pp() {
  run_tests '^bolt_decode_synth_pp_test$|^bolt_golden_synth_short_pp2_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab synth-pp '^bolt_(decode_synth_pp|golden_synth_short_pp2)' 'two cards|split|Review Focus 6|gate:|OK'
  finish
}
stage r60.golden    60 default gpu  bolt_g128,oracle_bolt    r60.k1   "SergiioB's checkpoint, two cards: bolt_decode_test --pp 48,40 (653 launches, B4, split 48 copy bitwise 40 peer); B3 short / 4k / agentic, split 40, bf16 head, eager"
st_r60_golden() {
  run_tests '^bolt_decode_real_test$|^bolt_golden_real_(bolt_(short|4k|agentic)|short_split40|short_bf16head|short_eager)_test$' \
    '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab golden '^bolt_(decode|golden)_real' 'plan == allocation|launches|B4|Review Focus 6|gate:|routing:|SSM state|conv|OK'
  finish
}
stage r60.golden32k 60 optin   gpu  bolt_g128,oracle_bolt    r60.golden "B3 on bolt_32k (32768 ids ingested one replay per id: hours, ESTIMATED)"
st_r60_golden32k() {
  run_tests '^bolt_golden_real_32k_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab golden32k '^bolt_golden_real_32k_test$' 'gate:|routing:|SSM state|conv|OK'
  finish
}
stage r60.fuse      60 default gpu  bolt_g128                r60.golden "every arm bitwise none on the real checkpoint (bolt_decode_test --fuse); the arms' A/B at depth 4096 (interleaved, median of 3)"
st_r60_fuse() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local k="ZE_AFFINITY_MASK=0,1" a arms="" ratios=""
  run_tests '^bolt_decode_real_fuse_(router|route|conv|all)_test$' '' '' "$k"
  jgrab fuse '^bolt_decode_real_fuse' 'Review Focus 5|launches|OK'
  idle before
  for a in none router route conv all; do
    arms="$arms $a '$k B70_BOLT_FUSE=$a $(bench_cmd "$snap" --depth 4096 --tg 256 --max-len 8192)'"
    [ "$a" = none ] || ratios="$ratios --ratio $a/none"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the Bolt arms at depth 4096"
  idle after
  grab_all ratio '^RATIO '
  grab_all idle '^IDLE '
  finish
}
stage r60.state_arm 60 optin   gpu  bolt_g128,oracle_bolt    r60.golden "B70_BOLT_SSM=bf16 (Decision 2's arm): B3 short / agentic beside fp32's, and its speed at 4096 / 65536 against fp32 (interleaved)"
st_r60_state_arm() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local k="ZE_AFFINITY_MASK=0,1" d arms="" ratios=""
  run_tests '^bolt_golden_real_(short|agentic)_b16state_test$' '' '' "$k"
  jgrab state '^bolt_golden_real_.*_b16state_test$' 'gate:|routing:|SSM state|conv|^  h |OK'
  idle before
  for d in 4096 65536; do
    arms="$arms d$d-fp32 '$k $(bench_cmd "$snap" --depth $d --tg 256 --max-len 131072)'"
    arms="$arms d$d-bf16 '$k B70_BOLT_SSM=bf16 $(bench_cmd "$snap" --depth $d --tg 256 --max-len 131072)'"
    ratios="$ratios --ratio d$d-bf16/d$d-fp32"
  done
  chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.rows -r 3$ratios --$arms" "the SSM-state arm"
  idle after
  grab_all ratio '^RATIO '
  finish
}
stage r60.speed     60 optin   gpu  bolt_g128                r60.golden "Task 8: decode at depth 886 / 8192 / 32000 / 65536 / 131072 / 262000 (tg 128), int8 / bf16 head, flash / eager, the arms (interleaved, median of 3); --profile --repeats 5 at 4096"
st_r60_speed() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local k="ZE_AFFINITY_MASK=0,1" d arms ratios
  x "uptime"
  for d in 886 8192 32000 65536 131072 262000; do   # the deep rows ingest one replay per id: hours, ESTIMATED
    idle "d$d"
    arms="d$d '$k $(bench_cmd "$snap" --depth $d --tg 128 --max-len auto)'"
    arms="$arms d$d-bf16 '$k $(bench_cmd "$snap" --depth $d --tg 128 --max-len auto --lm-head bf16)'"
    arms="$arms d$d-eager '$k B70_BOLT_ATTN=eager $(bench_cmd "$snap" --depth $d --tg 128 --max-len auto)'"
    arms="$arms d$d-all '$k B70_BOLT_FUSE=all $(bench_cmd "$snap" --depth $d --tg 128 --max-len auto)'"
    ratios="--ratio d$d-bf16/d$d --ratio d$d-eager/d$d --ratio d$d-all/d$d"
    chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.d$d.rows -r 3 $ratios -- $arms" "decode at depth $d"
  done
  chk "$k build/src/cli/b70-decode $snap --profile --depth 4096 --steps 32 --repeats 5" "--profile at 4096"
  x "uptime"
  grab_all ratio '^RATIO ' 40
  grab_all profile '^\| (nb_|gemv|mg_attn|argmax)|^MEASURED' 60
  finish
}

# ======================================================================================
row 61 "spec 25d - Nemotron 3.5 Bolt prefill on l0: absolute SSD sub-chunks with the fp32 state, the conv seed / writeback, the grouped latent MoE over real rows, NoPE flash (plan 25d)"
rownote 61 "Written blind on the Mac (spec 25 §16) after row 60 (25c's decode: nothing of it has run on a card either): NO nb_pf_* binary has been compiled by ocloc (kernel_cmdlines +35 / -0 / ~0: nb_pf_mamba in two tilings and the bf16-state build, nb_pf_cb by DPAS, nb_pf_moe / nb_pf_moe_gemm up relu2 / down at I 1536 / 2048 / 2304 / 2560, nb_pf_embed_gather, 25c's nb_prep / nb_moe at M 2048, the N 4096 slabs in both layouts, the three k2 tail slabs; Muse's mg_pf_flash_attn pair is bound by name, built here only without the Muse block). No existing .cl edited: G0's sha is K0. On the Mac's GPU (tools/mac/clrun/bolt_pf_run, indicative) every portable prefill kernel agrees with tests/kernels/bolt_pf_ref.h - the seed, C B^T (plain), the scan in fp32 and bf16 state, the sort / gather / dequant / combine, the gather, 25c's kernels at M 2048 exact; the conv, dt, gnorm and the norm within 1-2 ulp (Apple's division); both tilings x10 equal; the prefill eager attention bitwise decode's mg_attn_eager on the same device. The DPAS kernels (nb_pf_cb_L64_DPAS, nb_pf_moe_gemm) and the whole walk have never run anywhere."
rownote 61 "Every stage on the REAL checkpoint needs BOTH cards (Decision 4): its commands set ZE_AFFINITY_MASK=0,1 themselves; the N 12 synthetics (row 59's r59.synth, data key bolt_synth) run on one card, pp2:6 on two. The synthetics' golden sets are row 60's r60.synth_golden (oracle_bolt_synth); SergiioB's row 57's r57.golden (oracle_bolt). Run row 60 first: the prefill's head, its KV / ring / state hand-over and every comparison against decode's fill use 25c's list."
rownote 61 "Bars (all PROPOSED, spec 25 §16): bolt_pf_kernels_test - every portable kernel bitwise the twin, C B^T by DPAS within 2^-16 of sum |C B| of the plain build, the grouped GEMMs bitwise pf_gemm_T0 on an expert's rows and the up rows' cosine >= 0.9999 against the twin's decode order, flash >= 0.99999 against fp64 at 3000 / 65536 / 200000, the eager build bitwise decode's eager rows. bolt_prefill_test - prefill vs decode's fill: logits >= 0.9995, h taps median >= 0.9998 / min >= 0.99, SSM states and conv columns >= 0.999 a block, K / V rows median >= 0.9998 / p01 >= 0.99; chunked / split at multiples of 64 bitwise. B3 on prefill: 25c's decode bars."
rownote 61 "Hand back: bolt_pf_kernels_test's lines (the DPAS C B^T error, the grouped == dense lines, flash cosines); every launches / plan line (4333 a chunk on the real model: 2317 + 2016 on two cards; 573 on the N 12; + 4 for the head; 1,251,240,256 B of scratch a device); bolt_prefill_test's lines 1-7 per checkpoint; the B3 prefill gate / routing / SSM / h lines per prompt; the pp rows at depth 886 / 8192 / 32000 / 65536 / 131072 beside SergiioB's published 2925 / 2014 / 871 / 486 t/s (his 131072 is unpublished) and the phase tables. The SSD's sub-chunk length (64 as built), the scan's tiling (W16_R16) and the expert batch (512 MiB) change only by a commit naming these rows."
stage r61.k0        61 default cpu -                         g0.sha,g0.bitwise,g0.suite "K0 (B0): every pre-existing binary identical; kernel_cmdlines +35 / -0 / ~0 (no existing .cl edited)"
st_r61_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r61.host      61 default cpu -                         -        "host: bolt_pf_ref_test (the SSD form vs the recurrence in double and fp32, the conv over a chunk = decode's, absolute sub-chunks, the grouped MoE = decode's per row, the fixture), bolt_pf_variant_names_test (25c's names pinned), bolt_plan_test (the prefill launches, scratch, link, chunk rule)"
st_r61_host() {
  run_tests '^(bolt_pf_ref_test|bolt_pf_variant_names_test|bolt_plan_test)$'
  jgrab host '^(bolt_pf_ref_test|bolt_pf_variant_names_test|bolt_plan_test)$' 'OK|bitwise|relative|cosine|launches|prefill|differ'
  finish
}
stage r61.k1        61 default gpu -                         r61.k0   "kbins of the block + bolt_pf_kernels_test: B2 on prefill (every kernel against bolt_pf_ref.h, both tilings x10 finite and equal, C B^T by DPAS vs plain, grouped == dense, the slabs, flash vs fp64 at 3000 / 65536 / 200000, eager bitwise decode's)"
st_r61_k1() {
  kbins nb_pf_mamba_L64_W16_R16 nb_pf_mamba_L64_W8_R4 nb_pf_mamba_L64_W16_R16_B16 nb_pf_cb_L64_DPAS \
    nb_pf_moe_I1536_L1024_T9_TM32 nb_pf_moe_I2560_L1024_T9_TM32 nb_pf_moe_gemm_K1024_N1536_RELU2 \
    nb_pf_moe_gemm_K1536_N1024 nb_pf_moe_gemm_K1024_N2560_RELU2 nb_pf_moe_gemm_K2560_N1024 \
    nb_pf_embed_gather_K4096_V131072 nb_res_norm_M2048_K4096_W8_S1_T1 nb_router_M2048_K4096 nb_route_M2048_T9 \
    k2_pf_dequant_slab_K4096_N18560 pf_dequant_slab_K8192_N4096_L1 mg_pf_flash_attn_Q32KV2_F mg_pf_flash_attn_Q32KV2_F_EAGER
  run_tests '^bolt_pf_kernels_test$'
  jgrab k1 '^bolt_pf_kernels_test$' 'bitwise|cosine|DPAS|grouped|slab|OK'
  finish
}
stage r61.reject    61 default gpu -                         -        "cli_reject_bolt_prefill_{int8,sycl,chunk,chunk64} - the l0-only backend, the 2048 chunk and its 64-grid, refused before the device"
st_r61_reject() {
  run_tests '^cli_reject_bolt_prefill_' '' '' 'ZE_AFFINITY_MASK=0,1'
  finish
}
stage r61.synth     61 default gpu  bolt_synth               r61.k1   "bolt_prefill_test on the synthetics (n12-g128, n12-g64x, eager, bf16 head, bf16 state): the walk's launches and plan, twice bitwise, recorded == replayed, chunks / splits at multiples of 64 bitwise, prefill vs decode's fill within the bars, routes up to near-ties, tokens tie-aware"
st_r61_synth() {
  run_tests '^bolt_prefill_synth_(n12-g128|n12-g64x|eager|bf16head|b16state)_test$'
  jgrab synth '^bolt_prefill_synth' '^[1-7]\.|split|chunks|bitwise|cosine|routing|tokens|OK'
  finish
}
stage r61.synth_b3  61 default gpu  bolt_synth,oracle_bolt_synth r61.synth "B3 on prefill: bolt_golden_test's prefill mode on the synthetics - short / 4k / agentic x both spellings, chunks of 512 on 4k, eager"
st_r61_synth_b3() {
  run_tests '^bolt_golden_prefill_synth_'
  jgrab synth-b3 '^bolt_golden_prefill_synth' 'prefill:|gate:|routing:|SSM state|conv|^  h |OK'
  finish
}
stage r61.synth_pp  61 default gpu  bolt_synth               r61.synth "two cards on the synthetic: bolt_prefill_test pp2:6 - the prefill on two cards (the copy hand-off a chunk) bitwise one card"
st_r61_synth_pp() {
  run_tests '^bolt_prefill_synth_pp_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab synth-pp '^bolt_prefill_synth_pp_test$' 'pp2|walk|OK'
  finish
}
stage r61.golden    61 default gpu  bolt_g128,oracle_bolt    r61.k1   "SergiioB's checkpoint, two cards: bolt_prefill_test (4333 launches a chunk, B4, prefill vs decode's fill) and B3 after a prefilled prompt - short / 4k / agentic, eager"
st_r61_golden() {
  run_tests '^bolt_prefill_real_test$|^bolt_golden_prefill_real_(bolt_(short|4k|agentic)|short_eager)_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab golden '^bolt_(prefill|golden_prefill)_real' '^[1-7]\.|prefill:|gate:|routing:|SSM state|conv|OK'
  finish
}
stage r61.golden32k 61 optin   gpu  bolt_g128,oracle_bolt    r61.golden "B3 on bolt_32k after a prefilled prompt (16 chunks)"
st_r61_golden32k() {
  run_tests '^bolt_golden_prefill_real_32k_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab golden32k '^bolt_golden_prefill_real_32k_test$' 'prefill:|gate:|routing:|SSM state|conv|OK'
  finish
}
stage r61.speed     61 optin   gpu  bolt_g128                r61.golden "prefill rate at depth 886 / 8192 / 32000 / 65536 / 131072 beside SergiioB's published table (flash and eager interleaved, median of 3), the phase tables at 4096 and 65536"
st_r61_speed() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local k="ZE_AFFINITY_MASK=0,1" d arms ratios
  x "uptime"
  for d in 886 8192 32000 65536 131072; do
    idle "d$d"
    arms="d$d '$k $(bench_cmd "$snap" --prefill-length $d --tg 32 --max-len auto)'"
    arms="$arms d$d-eager '$k B70_BOLT_ATTN=eager $(bench_cmd "$snap" --prefill-length $d --tg 32 --max-len auto)'"
    ratios="--ratio d$d-eager/d$d"
    chk "tools/box_validate/interleave.sh -o $STATE/$STAGE.d$d.rows -r 3 $ratios -- $arms" "prefill at depth $d"
  done
  for d in 4096 65536; do
    chk "$k B70_PREFILL_PROFILE=1 $(bench_cmd "$snap" --prefill-length $d --tg 1 --max-len auto)" "the phase table at $d"
  done
  x "uptime"
  grab_all rows '^\| b70-decode|^RATIO ' 60
  grab_all phases 'prefill phase attribution|^  (norm|slab_|attn_|gdn_|moe_|silu|head|TOTAL)|^pp: ' 80
  finish
}

# ======================================================================================
row 62 "spec 25e - Nemotron 3.5 Bolt served: the template / tokenizer identities, ultra_v3 reasoning over Qwen XML calls, NVIDIA's sampling defaults, hybrid snapshots at aligned block ends, A4 and llama-benchy (plan 25e)"
rownote 62 "Written blind on the Mac (spec 25 §17) after rows 60 / 61 (25c's decode, 25d's prefill: neither has run on a card): nothing of 25e ran on a card. 25e adds NO kernel (BoltEngine's snapshot calls are copies on the immediate lists; kernel_cmdlines +0 / -0 / ~0): G0's g0.sha is the 'nothing moved' check. It does change shared host code - toolcall.cc (one opt-in option on the Qwen stream: drop a call cut by the end of generation), chat_format, server.cc (the template's enable_thinking comes from the kwargs after the reasoning_effort mapping), CMake - r62.k0 re-runs the other models' template / parser / server tests. The template and tokenizer.json are read from SergiioB's pinned snapshot in the HF cache (B70_BOLT_SNAPSHOT names another) and the tests SKIP 77 without it; nothing of the model is vendored but its generation_config.json files (tests/model/bolt/, Decision 20)."
rownote 62 "Every stage on the REAL checkpoint needs BOTH cards (Decision 4): its commands set ZE_AFFINITY_MASK=0,1 themselves; the N 12 synthetics (row 59's r59.synth, data key bolt_synth) run on one card, pp2:6 on two. Run rows 60 and 61 first: the snapshot calls restore into 25d's prefill and 25c's decode, and every comparison here is against those. The synthetics carry tokenizer.json / tokenizer_config.json / chat_template.jinja / generation_config.json (make_synth.py copies them), so b70-serve starts on them; a random-weight synthetic never writes a tool call or a </think>: the reasoning / tool-call paths through HTTP are bolt_request_test's and bolt_stream_test's (host, scripted engine) - the card stages prove the engine half (snapshots, startup, the server's greedy chat == b70-decode, seeded sampling) and, on SergiioB's checkpoint, a real model's A4 and the agentic loop."
rownote 62 "Decisions this row measures or leaves open (spec 25 §17): the sampling defaults when the snapshot's generation_config.json has no sampling key (SergiioB's: EOS 11 only) are NVIDIA's published T 1.0 / top-p 0.95 / EOS [2, 11] with NO top-k (the card names none, vLLM applies none; PROVISIONAL), a reasoning_effort of xhigh maps to the default like high (PROVISIONAL extension of Decision 13's table), --chat-template FILE is accepted for Bolt (its sha256 is printed and compared with Bolt's 6521afc7 in the startup line), and r62.benchy takes the lock itself (gpu-self: serve_benchy.sh does; the plan's table said gpu). The agentic transcript's hit rate has NO bar: it is recorded - truncate_history_thinking drops every earlier turn's reasoning when a new user message arrives (spec 25 §7's known miss), r62.agentic's report prints the first token each miss diverges at."
rownote 62 "Hand back: bolt_snapshot_gpu_test's lines (the sizes - 170,229,760 B of state and 8,192 B of KV a position at the real shapes - then a / b / c per placement), the startup lines (template sha, sampling defaults, the split, the snapshot size), golden_server_test's chat lines, serve_a4_texts' and score.py's A4 tables with bolt_parser_check's counts (0 failures where the scorer finds a call), the agentic report (hit rate, per-turn cached / ideal / the divergence token) in its three arms, the llama-benchy tables beside the operator's vLLM XPU run of the same checkpoint (SergiioB's flags: TP 2, fp8 KV, --enable-prefix-caching --mamba-cache-mode align, --tool-call-parser qwen3_coder --reasoning-parser ultra_v3) and SergiioB's published table (decode 60.5 / 50.7 / 33.5 / 23.0 t/s at 886 / 8k / 32k / 64k, prefill 2925 / 2014 / 871 / 486 t/s)."
stage r62.k0        62 default cpu -                         g0.sha,g0.bitwise,g0.suite "K0 (B0): 25e adds no kernel (every binary identical, G0), no existing server / template / tokenizer test changes behaviour (golden_server_test gained two opt-in env vars) - here the host tests that read the patched toolcall.cc / chat_format / server.cc: every template, toolcall and server test, openai_test, minja_ext_test"
st_r62_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  run_tests '^(template_test|template_agnes_test|template_ornith_test|template_k2_test|template_kolibri_test|template_muse_test|minja_ext_test|toolcall_test|toolcall_k2_test|toolcall_kolibri_test|toolcall_muse_test|toolcall_lfm2_test|openai_test|k2_server_test|kolibri_server_test|muse_server_test|lfm2moe_server_test|ornith_server_test|qwen4exp_server_test|protocol_test|prefix_server_test|mtp_server_test|lookup_server_test|spec_accept_test|sampling_test)$' '' 'kv8'
  finish
}
stage r62.host      62 default cpu  -                       -           "template_bolt_test, bolt_tokenizer_test, bolt_stream_test, bolt_request_test, bolt_snapshot_test"
st_r62_host() {
  run_tests '^(template_bolt_test|bolt_tokenizer_test|bolt_stream_test|bolt_request_test|bolt_serve_args_test|bolt_snapshot_test|bolt_plan_test)$'
  jgrab template '^template_bolt_test$' 'template cases|OK'
  jgrab tokenizer '^bolt_tokenizer_test$' 'text set|single ids|corpus|renders|OK'
  jgrab stream '^bolt_stream_test$' 'parse cases|streaming|fuzz|Qwen|OK'
  jgrab request '^bolt_request_test$' 'fixtures|reasoning_effort|kwarg|media|generation|sampling|output|OK'
  jgrab snapshot '^bolt_snapshot_test$' 'layout|round trip|restores at|restore at|prefix:|pp:|server:|OK'
  chk "python3 tools/toolcall/bolt_parser_check.py --selftest build/tests/bolt_parse_dump tests/server/bolt_parse_cases.json && python3 tools/toolcall/test_serve_a4_texts.py && python3 tools/prefix/test_agentic_loop.py && python3 tools/box_validate/test_box_validate.py AgenticRunTest" "the parser checker's selftest, serve_a4_texts' and the agentic loop's Python tests"
  finish
}
stage r62.reject    62 default cpu  -                       -           "b70-serve's Bolt refusals before the device (tests/model/bolt: config.json only): --mtp, --spec mtp|lookup|dflash, the proposers' tuning flags, --draft-vocab, --kv-cache int8, --prefill-backend l0-int8|sycl-tla, the Qwen3.8-Flash-Next flags, --max-len 262145, --pp 2 --device 1, --pipeline-split / -handoff without --pp 2, --generation-config of a missing file"
st_r62_reject() {
  run_tests '^cli_reject_serve_bolt_'
  finish
}
stage r62.snap_synth 62 default gpu bolt_synth              r61.synth   "bolt_snapshot_gpu_test on the N 12 synthetics (Review Focus 4 on the card): _synth_test (n12-g128, one card), _synth_g64x_test, _synth_b16_test (the bf16 state arm), _synth_pp_test (pp2:6), _cross_test (a snapshot taken on two cards continues on one, and the reverse): the sizes (170,229,760 B of state / 86,343,680 with b16 at the real shapes, n x 8,192 B of KV), restores at the block ends 2048 / 4096 BITWISE the cold run (32 ids, logits, KV, the last three conv columns, the SSM states), at the request ends 2049 / 5000 / 300 / 3 / 2 / 1 / 0 bitwise the session that never left"
st_r62_snap_synth() {
  run_tests '^bolt_snapshot_gpu_synth(_g64x|_b16)?_test$|^bolt_snapshot_gpu_synth_pp_test$|^bolt_snapshot_gpu_cross_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^bolt_snapshot_gpu' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r62.snapshot  62 default gpu  bolt_g128               -           "bolt_snapshot_gpu_test: restores bitwise the cold run, two cards, split change"
st_r62_snapshot() {
  run_tests '^bolt_snapshot_gpu_real_test$' '' '' 'ZE_AFFINITY_MASK=0,1'
  jgrab snapshot '^bolt_snapshot_gpu_real_test$' 'state|restore|snapshot restored|bitwise|DIFFER|OK|FAILED'
  finish
}
stage r62.serve     62 default gpu  bolt_synth              r62.snap_synth "b70-serve <synthetic n12-g128> startup and requests: one card by default (the startup line: Nemotron 3.5 Bolt, bolt chat format, template sha256 6521afc7 = Bolt's, --pp 1, prefill l0, mtp none, lm_head int8, kv cache bf16, the sampling defaults - the snapshot's generation_config.json names no sampling key, so the published T 1.00 top-p 0.95 top-k none and EOS [2, 11] - and the snapshot's size), --max-len auto planned with the prefill scratch; --pp 2 (--pipeline-split 6) and --pipeline-handoff peer; a seeded sampled request twice, identical"
st_r62_serve() {
  local ck="oracle-out-bolt-synth/n12-g128"
  x "ZE_AFFINITY_MASK=0 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $ck --max-len 16384"
  step_rc $? "b70-serve <synthetic n12-g128>, one card (the default), seeded sampling"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pp 2 --pipeline-split 6 --max-len 16384"
  step_rc $? "b70-serve <synthetic n12-g128> --pp 2 --pipeline-split 6"
  x "ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_probe.sh $ck --pp 2 --pipeline-split 6 --pipeline-handoff peer --max-len 16384"
  step_rc $? "b70-serve <synthetic n12-g128> --pp 2 --pipeline-handoff peer"
  grab_all auto 'max_len: auto ->|^split: ' 6
  grab_all memory '^memory' 8
  grab_all startup '^b70-serve: .*Nemotron 3.5 Bolt' 4
  grab_all seeded '^SEEDED ' 4
  finish
}
stage r62.chat_synth 62 default gpu bolt_synth              r62.serve   "one greedy chat through b70-serve equal to b70-decode on the same ids - golden_server_test --chat on the synthetic n12-g128 (prose / code / cjk as one user message, 32 ids, temperature 0; b70-decode --ids <prompt_token_ids> --n 32 --prefill --lm-head int8), one card, --pp 2 split 6 and peer"
st_r62_chat_synth() {
  local ck="oracle-out-bolt-synth/n12-g128" a z
  for a in "--max-len 16384" "--max-len 16384 --pp 2 --pipeline-split 6" "--max-len 16384 --pp 2 --pipeline-split 6 --pipeline-handoff peer"; do
    z=0; case "$a" in *"--pp 2"*) z=0,1 ;; esac
    chk "ZE_AFFINITY_MASK=$z B70_SERVE_ARGS='$a' timeout 3600 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $ck --chat" \
      "Bolt synthetic, golden_server_test --chat, b70-serve $a"
  done
  grab_all chat 'chat of [0-9]+ prompt ids' 9
  finish
}
stage r62.chat      62 default gpu  bolt_g128               -           "golden_server_test --chat on b70-serve <bolt> == b70-decode (B70_SERVE_DECODE_AFTER=1: the chats are collected, b70-serve is stopped, THEN b70-decode runs on each saved prompt - two copies of 41.9 GB do not fit two cards)"
st_r62_chat() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  x "WAIT=3600 ZE_AFFINITY_MASK=0,1 PORT=$PORT SEEDED=1 tools/box_validate/serve_probe.sh $snap"
  step_rc $? "b70-serve <SergiioB's g128>, two cards (the default), --max-len auto, seeded sampling"
  chk "ZE_AFFINITY_MASK=0,1 B70_SERVE_WAIT_S=3600 B70_SERVE_DECODE_AFTER=1 B70_SERVE_ARGS='--max-len 32768' timeout 7200 build/tests/golden_server_test build/src/cli/b70-serve build/src/cli/b70-decode tests/golden/prompts $snap --chat" \
    "golden_server_test --chat against SergiioB's g128, two cards"
  grab_all auto 'max_len: auto ->|^split: ' 6
  grab_all memory '^memory' 8
  grab_all startup '^b70-serve: .*Nemotron 3.5 Bolt' 4
  grab_all chat 'chat of [0-9]+ prompt ids' 4
  finish
}
stage r62.a4        62 optin   gpu  bolt_g128,bolt_a4_int4  r62.chat    "A4 through b70-serve vs the g128 reference, and vs the BF16 one when the BF16 window has made it (B5's engine row)"
st_r62_a4() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local d=$STATE/a4-bolt set=tests/golden/toolcall-bolt
  x "mkdir -p $d && cp oracle-out-bolt-a4/*.int4.txt $d/ && { cp oracle-out-bolt-a4/*.bf16.txt $d/ 2> /dev/null || true; }"
  chk "WAIT=3600 ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/serve_run.sh $d/serve serve 1 $snap --max-len 32768 -- --set $set --max-tokens 2048 --stop-at-eos" "A4 through b70-serve (greedy, up to 2048 new ids, the set's chat_template_kwargs)"
  chk "python3 tools/toolcall/serve_a4_texts.py $d/serve/server-log $set $d serve" "every scenario's prompt ids as the server rendered them equal the reference's (the template and tokenizer on the real tools), the raw texts written"
  chk "python3 tools/toolcall/score.py --format qwen-think $d int4 serve | tee $d/score-int4.md" "scoring b70-serve against SergiioB's g128 reference (25a's int4 run)"
  x "if ls $d/*.bf16.txt > /dev/null 2>&1; then python3 tools/toolcall/score.py --format qwen-think $d bf16 serve | tee $d/score-bf16.md; else echo 'no BF16 reference texts (the BF16 window has not run): skipped'; fi"
  step_rc $? "scoring against the BF16 reference (skipped when the BF16 window has not run: no <name>.bf16.txt)"
  chk "python3 tools/toolcall/bolt_parser_check.py build/tests/bolt_parse_dump $d serve int4 | tee $d/parser-check.md" "the server's parser reads a call wherever the scorer does, on b70-serve's texts and the g128 reference's: 0 failures"
  grab_all a4 'match|/ 36|parse failure|files,|failures|scenarios written' 14
  finish
}
stage r62.agentic   62 optin   gpu  bolt_g128               r62.chat    "the agentic transcript (tests/golden/prompts/bolt_agentic.json: a coding session) as a LIVE agent loop through b70-serve on SergiioB's g128 with the prefix cache on: per request the prompt length, cached_tokens, the hit rate, the turns whose call the template re-renders in another order than generated, and (Review Focus 5) the first token each miss diverges at - three arms: the default cache (truncate_history_thinking true: the known miss at each new user message), truncate_history_thinking false (what the miss costs), --prefix-cache-gb 0"
st_r62_agentic() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997' t=tests/golden/prompts/bolt_agentic.json
  idle r62-agentic
  chk "WAIT=3600 ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/agentic_run.sh $STATE/agentic-default $snap $t --max-len 65536" "agentic loop, prefix cache auto (Bolt's defaults: thinking on, truncate_history_thinking true)"
  chk "WAIT=3600 ZE_AFFINITY_MASK=0,1 PORT=$PORT EXTRA='{\"chat_template_kwargs\": {\"truncate_history_thinking\": false}}' tools/box_validate/agentic_run.sh $STATE/agentic-keep $snap $t --max-len 65536" "agentic loop, truncate_history_thinking false"
  chk "WAIT=3600 ZE_AFFINITY_MASK=0,1 PORT=$PORT tools/box_validate/agentic_run.sh $STATE/agentic-off $snap $t --max-len 65536 --prefix-cache-gb 0" "agentic loop, cache off"
  x "uptime"
  grab_all agentic '^hit rate|^turns whose call|^ideal prefix|^turn [0-9]+ diverges|^\| [0-9]+ \|' 90
  finish
}
stage r62.benchy    62 optin   gpu-self bolt_g128,uvx       r62.chat    "llama-benchy at 886 / 8192 / 32000 / 65536 beside vLLM XPU (row 41's matrix)"
st_r62_benchy() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local f="--concurrency 1 --no-cache --exact-tg --latency-mode generation --runs 2"
  say "the operator's vLLM XPU run of the SAME checkpoint first (SergiioB's README: vllm/vllm-openai-xpu, --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-model-len 262144 --block-size 64 --enable-prefix-caching --mamba-cache-mode align --tool-call-parser qwen3_coder --reasoning-parser-plugin ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3, its patches from the card's b70/), then the same llama-benchy rows against it: uvx llama-benchy@\${BENCHY_VERSION:-0.4.0} --base-url http://127.0.0.1:<port>/v1 --model <served name> $f --pp 886 8192 32000 65536 --tg 128 --depth 0 (the prompt-length rows of SergiioB's table) and $f --pp 512 --tg 128 --depth 886 8192 32000 65536 (decode at depth, row 41's D)"
  chk "WAIT=3600 MODEL=$snap MAX_LEN=262144 SERVE_AFFINITY=0,1 BENCHY_VERSION=0.4.0 tools/probe/serve_benchy.sh --pp 886 8192 32000 65536 --tg 128 --depth 0 $f" "b70-serve <bolt> on two cards: pp = prompt length 886 / 8192 / 32000 / 65536, tg 128"
  chk "WAIT=3600 MODEL=$snap MAX_LEN=262144 SERVE_AFFINITY=0,1 BENCHY_VERSION=0.4.0 tools/probe/serve_benchy.sh --pp 512 --tg 128 --depth 886 8192 32000 65536 $f" "b70-serve <bolt> on two cards: decode at depth 886 / 8192 / 32000 / 65536"
  grab_all benchy '^\| ' 80
  finish
}

# ======================================================================================
row 63 "spec 26a - LiLiCorr P0: the reference (dflash_ref.py's LiLiCorr form, NVFP4 decoded exactly, cross-checked against vLLM's head), acceptance on our Bolt, the expert union, the verify / block costs, the projection (plan 26a)"
rownote 63 "Download the drafter (962 MB): hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr --revision 449d24140160b8e51871bfc0f0a7338abf640aa1"
rownote 63 "Written on the Mac (spec 26 §12, docs/probe-lilicorr-2026-10-23.md): the reference's LiLiCorr form is tested on a tiny random drafter (lilicorr_make_tiny.py) - NVFP4's 16 x 254 finite products exact in bf16 and f16, every Review Focus item a mutation that moves the lattice or the drafts - and end to end on 25a's tiny Bolt (bolt_ref.py taps --tap-layers 1,2,4,6,9,10). the real drafter's weights were read on the Mac (decoded exactly, 1,112,803,882 parameters) but never beside Bolt's taps; vLLM is not importable in the Mac's oracle image, so Decision 1's cross-check is r63.vllmref's. kernel_cmdlines +99 / -0 / ~0 (Bolt's 7 GEMVs, nb_router, the int8 head and argmax stage 1 at M = 2..8; the expert stand-in; the drafter at M 8 in three formats; the LiLiCorr head's bf16 shapes) - no existing .cl edited, none of them has been compiled by ocloc."
rownote 63 "P0 gates nothing (the operator's 2026-10-15 ruling: 26b / 26c are built whatever it shows); it sets the defaults (spec 26 §11): the drafter format (Decision 2: int8 unless int4 g64 is within 0.05 of exact's E_K at the chosen K), K_max, the q form (Decision 6), the replay lever (Decision 4: built only if the slot writes exceed 10 % of a verify at the chosen K), the union MoE's expectation (Decision 10) and 26c's AdaptiveK cost row. Order: r57.golden + r57.taps (+ r57.a4_ref for the a4/* dumps) first; r63.accept and r63.union on the CPU; r63.cost on one card; then r63.verdict by hand."
rownote 63 "Hand back: r63.accept's oracle-out-lilicorr-p0/accept/summary.md (E_K per corpus and arm, alpha per depth, the sampled table with the censored / short counts) + lookup.txt; r63.union's union.txt / union.gen.txt (u per M and E bucket, the routed bytes); r63.cost's table (the verify per M at 4096 / 65536, by part, the time per expert block, the state rate, the three blocks and the LiLiCorr head alone); r63.vllmref's lines (the lattice BITWISE or the max |diff| with its reason); r63.verdict's project.md; r63.vllm_xpu's mean acceptance length (or its error)."
stage r63.k0        63 default cpu -                         g0.sha,g0.bitwise,g0.suite "K0 (B0): every pre-existing binary identical; kernel_cmdlines +99 / -0 / ~0 (no existing .cl edited)"
st_r63_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r63.host      63 default cpu oracle_image              -        "test_dflash_ref.py (LiLiCorr), test_dflash_accept.py (the tiny pair), test_lilicorr_p0_project.py, test_expert_union.py; host: lilicorr_cost_names_test"
st_r63_host() {
  run_tests '^lilicorr_cost_names_test$'
  chk "IMAGE=\${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest} SITE=oracle-out-q4exp-site MODEL=lilicorr STEPS=check tools/oracle/dflash_p0.sh" \
    "the reference's and the acceptance's tests in the oracle image (the tiny pair: 25a's tiny Bolt + lilicorr_make_tiny)"
  chk "python3 tools/spec/test_lilicorr_p0_project.py && python3 tools/spec/test_expert_union.py" "the projection's and the union's tests"
  jgrab host '^lilicorr_cost_names_test$' 'OK|names'
  x "grep -hE 'mutation|nvfp4:|tiny pair|lilicorr batch|int8 arm|int4 arm|^ok' oracle-out-lilicorr-p0/test_dflash_ref.log oracle-out-lilicorr-p0/test_dflash_accept.log | head -60"
  finish
}
stage r63.vllmref   63 optin   cpu vllm_image                -        "lilicorr_vllm_check.py in the vLLM image: the head bitwise in fp32"
st_r63_vllmref() {
  chk "ORACLE_IMAGE=\${VLLM_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest} tools/oracle/run_in_container.sh 'python3 tools/oracle/lilicorr_vllm_check.py'" \
    "dflash_ref's LiLiCorr head against vLLM's own LiLiCorrHead on CPU tensors (Decision 1; SKIP 77 when the image's vLLM has no models/lilicorr.py - then by hand with --vllm-src <a 3901d56800 checkout inside the tree>)"
  grab_all vllmref '^S [0-9]|^OK|^FAIL|^vLLM|^note|SKIP_REASON' 12
  finish
}
stage r63.accept    63 optin   cpu oracle_bolt_taps,lilicorr,bolt_g128,oracle_image - "E_K per corpus and arm, lookup (hours): MODEL=lilicorr dflash_p0.sh accept + lookup over oracle-out-bolt-taps/dumps -> oracle-out-lilicorr-p0/{accept/summary.md,lookup.txt}"
st_r63_accept() {
  x "free -g | head -2"
  chk "IMAGE=\${ORACLE_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest} SITE=oracle-out-q4exp-site MODEL=lilicorr STEPS='accept lookup' tools/oracle/dflash_p0.sh" \
    "the LiLiCorr acceptance (arms exact / int8 / int4g64 and their int8-head forms, K 1..7 from one 8-row block, sampled q lattice / lattice+lp) and the lookup baseline"
  x "cat oracle-out-lilicorr-p0/accept/summary.md oracle-out-lilicorr-p0/lookup.txt 2>/dev/null | head -160"
  grab_all accept '^\| |E_7|lookup|censored' 120
  finish
}
stage r63.union     63 default cpu oracle_bolt               -        "the expert union of M = 2..8 consecutive rows from the golden runs' route records (expert_union.py: u per M and E bucket, the routed bytes union vs plain) -> oracle-out-lilicorr-p0/union{,.gen}.{txt,json}"
st_r63_union() {
  chk "MODEL=lilicorr STEPS=union tools/oracle/dflash_p0.sh" "the expert union over oracle-out-bolt (and oracle-out-bolt-taps.partial/sources when r57.taps left its golden runs)"
  grab_all union '^\| [0-9]|^block\(I\)|rows \(from' 30
  finish
}
stage r63.cost      63 default gpu -                         r63.k0   "probe_draft_cost --shapes bolt-lilicorr at 4096 / 65536"
st_r63_cost() {
  kbins gemv_M8_K4096_N18560_S1_L0 nb_router_M8_K4096 gemv_i8w_M8_K4096_N131072 nb_argmax_stage1_M8_V131072 \
    gemv_M1_K1024_N2304_S1_L1 gemv_M8_K24576_N4096_S1_L1 gemv_i8w_M8_K24576_N4096 gemv_bf16_M8_K24576_N4096 \
    gemv_bf16_M8_K1024_N3072 gemv_bf16_M1_K1024_N1024_C16_S16
  idle r63-cost
  x "uptime"
  chk "build/tools/probe/probe_draft_cost --shapes bolt-lilicorr --mmax 8 --depth 4096,65536 --calls 10 --rounds 3 | tee $STATE/r63.cost.txt" \
    "Bolt's verify at M = 1..8 (by part; plain experts; the state's slot / replay terms) and one LiLiCorr block per format, interleaved pairs, medians of 3"
  grab_all cost '^\||^probe_draft_cost|^the (M = 1|state|experts)' 80
  finish
}
stage r63.verdict   63 manual  -   -                         r63.accept "lilicorr_p0_project.py: the defaults into spec 26 (not a gate)"
st_r63_verdict() {
  say "python3 tools/spec/lilicorr_p0_project.py --accept oracle-out-lilicorr-p0/accept --dumps oracle-out-bolt-taps/dumps --probe \$STATE/r63.cost.txt --depth 4096 --plain-ms <r60.speed's plain step at depth 4096, ms> --union oracle-out-lilicorr-p0/union.json --md oracle-out-lilicorr-p0/project.md --json oracle-out-lilicorr-p0/project.json"
  say "python3 tools/spec/lilicorr_p0_project.py --accept oracle-out-lilicorr-p0/accept --sampled lattice --probe \$STATE/r63.cost.txt --depth 4096 --plain-ms <the same> --union oracle-out-lilicorr-p0/union.json   # the sampled form (T 1.0 / top-p 0.95)"
  say "# then write spec 26 §11 (P0 verdict) with the table and the defaults, and docs/probe-lilicorr-2026-10-23.md §3-§5 with the box's tables"
  finish
}
stage r63.vllm_xpu  63 manual  gpu bolt_g128,lilicorr        -        "optional: vLLM XPU serving Bolt with LiLiCorr - external acceptance figure"
st_r63_vllm_xpu() {
  local snap='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--SergiioB--Nemotron-3.5-Bolt-74B-A9B-AutoRound-W4A16-G128-sym/snapshots/b9f381d20cbd4cede9f729c2647d2542c4a35997'
  local lil='${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr/snapshots/449d24140160b8e51871bfc0f0a7338abf640aa1'
  say "spec 26 Decision 12 (optional, by hand, both cards, the GPU lock held): in the vLLM XPU image (\${VLLM_IMAGE:-vllm-xpu-env-next-p314-t215-vxkp0:latest}) serve SergiioB's checkpoint as its README does (TP 2, --dtype bfloat16, the card's b70/ patches) plus --speculative-config '{\"method\":\"dflash\",\"model\":\"$lil\",\"num_speculative_tokens\":7,\"rejection_sample_method\":\"standard\"}' (add \"draft_sample_method\":\"probabilistic\" for the T 1.0 arm); model $snap"
  say "if it starts: send the 36 A4 prompts (tests/golden/toolcall-bolt, greedy and T 1.0 / top-p 0.95) and the agentic transcript (tests/golden/prompts/bolt_agentic.json) and record vLLM's 'SpecDecoding metrics: Mean acceptance length' lines beside r63.accept's E_7 (the card says SM120 only: if it refuses or crashes, record the error and close the step - no patches)"
  finish
}

# ======================================================================================
row 66 "spec 13a - the batching probe: rows of different sequences in one decode step (the BATCH_ROWS row builds, runtime::build_rows, probe_batch_step under B70_BATCH_PROBE); the check, the step and chunk tables (plan 13a)"
rownote 66 "Written blind on the Mac (docs/probe-batching-2026-10-23.md): B70_BATCH_PROBE is OFF by default, so G0's build is main's (capture.cc's default token stream and all 105 existing command lines of gdn_step.cl / attn.cl / attn_v2.cl / kv8.cl preprocess to main's tokens; G0's g0.sha holds the binaries). r66.build is the first ocloc compile of the 80 row builds (tools/kernel_cmdlines lists them with the option on: +80 / -0 / ~0) and of build_rows; the Mac ran gdn_step's and attn_prep's row builds bitwise against the production kernels (tools/mac/clrun/batch_rows_run), never attention v2's."
rownote 66 "Timed stages (r66.cost, r66.chunk): an idle box (box-day plan Session 6d, the idle protocol of section 1.7), BOTH cards free of DRM holders, RECORD grade, device 0, the GPU lock for the whole stage; every arm is an interleaved pair against 1x1@4k after a warm-up, median of 3 (probe_batch_step's module comment). Qwen3.8 at int8 KV is the decided form, bf16 the comparison; Agnes runs its two layouts (2 slots with MTP; 4 with --mtp off)."
rownote 66 "Hand back: every 'CHECK' line and the 'check:' verdict of r66.check (a FAIL stops 13b); r66.cost's tables ('| arm |' rows, the production verify(0) pair) per model and KV form; r66.chunk's chunk tables; the 'memory:' lines. They fill docs/probe-batching-2026-10-23.md sections 2-4 and the operator's section 7 decisions (the slots per model, the row budget's allocation, the interleave chunk)."
stage r66.k0    66 default cpu -           g0.sha,g0.bitwise,g0.suite "B0: the default build is main's (G0) - B70_BATCH_PROBE is off there"
st_r66_k0() {
  need_pass g0.sha g0.bitwise g0.suite
  finish
}
stage r66.build 66 default cpu -           g0.build "build-batch-probe: -DB70_BATCH_PROBE=ON (the 80 row builds' first ocloc compile, build_rows, probe_batch_step), -j\$JOBS"
st_r66_build() {
  chk "cmake -S . -B build-batch-probe -DCMAKE_BUILD_TYPE=Release -DB70_BATCH_PROBE=ON $CMAKE_ARGS > /dev/null && cmake --build build-batch-probe -j$JOBS --target probe_batch_step" \
    "the probe build"
  chk "miss=0; for b in gdn_step_rows_M8 gdn_step_rows_M8_G54 attn_prep_M8_ROWS attn_prep_M8_S1_KV8_ROWS attn_v2_M8_T32_ROWS attn_v2_M8_T32_KV8_ROWS gemv_M8_K5120_N38912_S8_L0 argmax_stage1_M8_V248089; do if [ -s build-batch-probe/kernels/\$b.bin ]; then echo \"built \$b\"; else echo \"NOT BUILT \$b\"; miss=1; fi; done; [ \$miss = 0 ]" \
    "the row builds and Agnes's M = 8 rows are in build-batch-probe/kernels"
  finish
}
stage r66.check 66 default gpu qwen,agnes  r66.build "the check: every arm's rows bitwise their slot's run alone (logits, ids, GDN states, KV and head KV rows), isolation, the production verify / plain tie - Qwen3.8 and Agnes, bf16 and int8 KV, Agnes also --mtp off (4 slots)"
st_r66_check() {
  x "uptime"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_QWEN --kv int8 --no-time --no-chunks" "Qwen3.8, int8 KV, MTP"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_QWEN --kv bf16 --no-time --no-chunks" "Qwen3.8, bf16 KV, MTP"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv int8 --no-time --no-chunks" "Agnes, int8 KV, MTP (2 slots)"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv bf16 --no-time --no-chunks" "Agnes, bf16 KV, MTP (2 slots)"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv int8 --mtp off --no-time --no-chunks" "Agnes, int8 KV, --mtp off (4 slots)"
  grab_all check '^CHECK |^check: |^probe_batch_step: |^memory: ' 200
  finish
}
stage r66.cost  66 optin   gpu qwen,agnes  r66.check "the step table: every arm against 1x1@4k as interleaved pairs, median of 3 - Qwen3.8 int8 and bf16 KV, Agnes int8 (2 slots, MTP) and --mtp off (4 slots) - idle box, both cards free of DRM holders (timed)"
st_r66_cost() {
  idle before
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_QWEN --kv int8 --no-check --no-chunks --calls 32 --rounds 3" "Qwen3.8, int8 KV (the decided form)"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_QWEN --kv bf16 --no-check --no-chunks --calls 32 --rounds 3" "Qwen3.8, bf16 KV"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv int8 --no-check --no-chunks --calls 32 --rounds 3" "Agnes, int8 KV, MTP (2 slots)"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv int8 --mtp off --no-check --no-chunks --calls 32 --rounds 3" "Agnes, int8 KV, --mtp off (4 slots)"
  idle after
  grab_all cost '^probe_batch_step: |^memory: |^timing: |^\| ' 200
  grab_all idle '^IDLE '
  finish
}
stage r66.chunk 66 optin   gpu qwen,agnes  r66.check "the interleave chunk: one Engine::prefill chunk of 512 / 1024 / 2048 at depth 4096 and 32768 beside the 1-row steps, rotated rounds, median of 3 - Qwen3.8 and Agnes, int8 KV - idle box (timed)"
st_r66_chunk() {
  idle before
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_QWEN --kv int8 --no-check --arms 1x1@4k,1x1@32k --chunks 512,1024,2048 --chunk-depths 4096,32768" "Qwen3.8, int8 KV"
  chk "build-batch-probe/tools/probe/probe_batch_step $SNAP_AGNES --kv int8 --no-check --arms 1x1@4k,1x1@32k --chunks 512,1024,2048 --chunk-depths 4096,32768" "Agnes, int8 KV"
  idle after
  grab_all chunk '^probe_batch_step: |^chunks: |^timing: |^\| ' 80
  grab_all idle '^IDLE '
  finish
}

# ======================================================================================
row x "the rest of the suite: every registered test no stage above ran (new host tests, the routed tests' twins)"
stage x.rest x default gpu qwen,oracle_qwen - "ctest over every registered test without a result in this run (Agnes / Ornith / kv8 / k2 / longctx / kolibri / qwen4exp / muse / dflash / bolt / lilicorr labels belong to their rows)"
st_x_rest() {
  run_tests '.' '' 'agnes ornith kv8 k2 longctx kolibri qwen4exp muse dflash bolt lilicorr'
  finish
}
