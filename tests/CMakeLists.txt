# Each test is add_executable + b70_target_kernel_dir + add_test. No framework:
# tests/check.h exits non-zero on the first failed CHECK. The exception is the
# `b70_cli_reject` block at the bottom, which has no source file of its own: it
# drives the shipped `b70-decode` binary and asserts its exit code.

add_executable(copy_test l0/copy_test.cc)
target_include_directories(copy_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(copy_test PRIVATE b70_l0)
add_test(NAME copy_test COMMAND copy_test)

add_executable(launch_test l0/launch_test.cc)
target_include_directories(launch_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(launch_test PRIVATE b70_l0)
b70_target_kernel_dir(launch_test)
add_dependencies(launch_test kernel_noop)
add_test(NAME launch_test COMMAND launch_test)

add_executable(arg_capture_test l0/arg_capture_test.cc)
target_include_directories(arg_capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(arg_capture_test PRIVATE b70_l0)
b70_target_kernel_dir(arg_capture_test)
add_dependencies(arg_capture_test kernel_noop)
add_test(NAME arg_capture_test COMMAND arg_capture_test)

# Kernel-timestamp events (plan 4's profiler foundation): the device timer
# calibration and the signal parameter on launch, on a two-launch regular list.
# Same shape as launch_test - a GPU and the noop binary, no checkpoint - so it
# carries no label, like every other test in this l0 group.
add_executable(event_test l0/event_test.cc)
target_include_directories(event_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(event_test PRIVATE b70_l0)
b70_target_kernel_dir(event_test)
add_dependencies(event_test kernel_noop)
add_test(NAME event_test COMMAND event_test)

# The only test here that needs neither a GPU nor a kernel: device_index_from_env
# is a pure parse of ONEAPI_DEVICE_SELECTOR (doc 04, "Device selection").
add_executable(device_select_test l0/device_select_test.cc)
target_include_directories(device_select_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(device_select_test PRIVATE b70_l0)
add_test(NAME device_select_test COMMAND device_select_test)

add_executable(int4_test kernels/int4_test.cc)
target_include_directories(int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME int4_test COMMAND int4_test)

add_executable(gemv_test kernels/gemv_test.cc)
target_include_directories(gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_test)
add_dependencies(gemv_test kernel_gemv_M1_K6144_N5120_S1_L0 kernel_gemv_M1_K6144_N5120_S4_L1
  kernel_gemv_M1_K17408_N5120_S8_L0 kernel_gemv_M1_K5120_N34816_S2_L1 kernel_gemv_M2_K6144_N5120_S1_L0)
add_test(NAME gemv_test COMMAND gemv_test)

add_executable(gemv_bf16_test kernels/gemv_bf16_test.cc)
target_include_directories(gemv_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_bf16_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_bf16_test)
# Both a||b tilings: the test runs the `{64, 1}` build and the retiled one the
# runtime binds over one set of inputs, and holds each - and their difference -
# to the reference tolerance (spec 1.5 lever L2 reorders the sum).
add_dependencies(gemv_bf16_test kernel_gemv_bf16_M1_K5120_N128
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320)
add_test(NAME gemv_bf16_test COMMAND gemv_bf16_test)

add_executable(prep_test kernels/prep_test.cc)
target_include_directories(prep_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(prep_test PRIVATE b70_l0)
b70_target_kernel_dir(prep_test)
# Both shapes of the residual norm: the single-work-group `prep_res_norm` and
# the `prep_res_fold` + `prep_norm_finish` pair the runtime binds since spec
# 1.5's lever L1. The test runs the pair against the single kernel's own
# reference (`resid` bit-identical, `x` to the ruled ulp bar for the reordered
# Sigma), so both must exist - the same arrangement lever L2 left for
# gemv_bf16's two tilings. `_W1` is stage B's single-work-group finish, run to
# prove the grid changes no bits.
add_dependencies(prep_test kernel_prep_res_norm_M1_K5120_SP0 kernel_prep_res_norm_M1_K5120_SP16
  kernel_prep_res_fold_M1_K5120_SP0_G20 kernel_prep_res_fold_M1_K5120_SP16_G20
  kernel_prep_norm_finish_M1_K5120_G20_W20 kernel_prep_norm_finish_M1_K5120_G20_W1
  kernel_prep_silu_mul_M1 kernel_prep_gated_head_M1)
add_test(NAME prep_test COMMAND prep_test)

# The GDN decode step. Its reference (gdn_ref.h) reproduces the kernel's tree
# orders exactly, so the bars are ulp-tight and the conv ring is bit-exact.
add_executable(gdn_step_test kernels/gdn_step_test.cc)
target_include_directories(gdn_step_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gdn_step_test PRIVATE b70_l0)
b70_target_kernel_dir(gdn_step_test)
add_dependencies(gdn_step_test kernel_gdn_step_M1 kernel_gdn_step_M2)
add_test(NAME gdn_step_test COMMAND gdn_step_test)

# The decode-attention trio. Its reference (attn_ref.h) reproduces the kernels'
# norm tree, score tree, wave order and block-merge order exactly, so the KV
# cache and the norm/RoPE outputs are held bit-exact and only the `exp`-carrying
# tensors have a tolerance. M = 2 is run, not just compiled: it is the only
# cover for per-m causal masking.
add_executable(attn_test kernels/attn_test.cc)
target_include_directories(attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(attn_test PRIVATE b70_l0)
b70_target_kernel_dir(attn_test)
add_dependencies(attn_test kernel_attn_prep_M1 kernel_attn_decode_M1_L4096
  kernel_attn_reduce_M1_L4096 kernel_attn_prep_M2 kernel_attn_decode_M2_L4096
  kernel_attn_reduce_M2_L4096 kernel_attn_decode_M1_L16384 kernel_attn_reduce_M1_L16384)
add_test(NAME attn_test COMMAND attn_test)

# The two kernels at a token's boundary. Both read the shared control block at
# *execution* time, which is what the replayed list needs and what a kernel
# argument (captured at append - tests/l0/arg_capture_test.cc) could not give.
add_executable(embed_gather_test kernels/embed_gather_test.cc)
target_include_directories(embed_gather_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(embed_gather_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(embed_gather_test)
add_dependencies(embed_gather_test kernel_embed_gather_M1)
add_test(NAME embed_gather_test COMMAND embed_gather_test)

add_executable(argmax_test kernels/argmax_test.cc)
target_include_directories(argmax_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(argmax_test PRIVATE b70_l0 b70_model)
b70_target_kernel_dir(argmax_test)
add_dependencies(argmax_test kernel_argmax_stage1_M1 kernel_argmax_stage2)
add_test(NAME argmax_test COMMAND argmax_test)

add_executable(json_test common/json_test.cc)
target_include_directories(json_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME json_test COMMAND json_test)

add_executable(safetensors_test loader/safetensors_test.cc)
target_include_directories(safetensors_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(safetensors_test PRIVATE b70_loader)
add_test(NAME safetensors_test COMMAND safetensors_test)

add_executable(quant_test loader/quant_test.cc)
target_include_directories(quant_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(quant_test PRIVATE b70_loader)
add_test(NAME quant_test COMMAND quant_test)

add_executable(dequant_fixture_test loader/dequant_fixture_test.cc)
target_include_directories(dequant_fixture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(dequant_fixture_test PRIVATE b70_loader)
add_test(NAME dequant_fixture_test COMMAND dequant_fixture_test ${CMAKE_SOURCE_DIR}/tests/golden/dequant_fixture.safetensors)

add_executable(qwen35_test model/qwen35_test.cc)
target_include_directories(qwen35_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(qwen35_test PRIVATE b70_model)
add_test(NAME qwen35_test COMMAND qwen35_test)

# The model table and src/kernels/CMakeLists.txt are two lists that have to
# agree: every row's compiled variant must exist before plan 3 binds it. No
# device, no checkpoint - just the filenames.
add_executable(kernel_table_test kernels/kernel_table_test.cc)
target_include_directories(kernel_table_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(kernel_table_test PRIVATE b70_model)
b70_target_kernel_dir(kernel_table_test)
add_dependencies(kernel_table_test
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K6144_N5120_S16_L1
  kernel_gemv_M1_K5120_N34816_S4_L1 kernel_gemv_M1_K17408_N5120_S16_L1
  kernel_gemv_M1_K5120_N14336_S1_L1
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320)
add_test(NAME kernel_table_test COMMAND kernel_table_test)

# Every decode allocation, sized from the model constants: the two byte totals
# are pinned so a size change has to be argued for in the test's arithmetic.
add_executable(buffers_test runtime/buffers_test.cc)
target_include_directories(buffers_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(buffers_test PRIVATE b70_runtime b70_l0 b70_model)
add_test(NAME buffers_test COMMAND buffers_test)

# Needs the real 19 GB checkpoint and a B70: label `checkpoint` so it can be
# excluded (`ctest -LE checkpoint`) where the model or the card is absent.
add_executable(load_checkpoint_test loader/load_checkpoint_test.cc)
target_include_directories(load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(load_checkpoint_test PRIVATE b70_loader b70_model)
add_test(NAME load_checkpoint_test COMMAND load_checkpoint_test)
set_tests_properties(load_checkpoint_test PROPERTIES LABELS checkpoint TIMEOUT 600)

# Every variant runtime::build binds, by name: a missing .bin is a
# zeModuleCreate failure minutes into a load otherwise (kernel_table_test makes
# the same point for the model table's GEMV rows). One list, because all three
# tests below capture the same decode list. (src/cli/b70-decode deliberately
# depends on ALL kernels instead - see src/cli/CMakeLists.txt.)
#
# None of the three calls `b70_target_kernel_dir`, and that is not an omission:
# they *build* the variants (add_dependencies) but never name a path, because
# `runtime::build` is what opens the modules and B70_KERNEL_DIR is PRIVATE to
# b70_runtime's own compile of capture.cc. Adding the call back would define an
# unused macro in three translation units that do not include kernels/kernels.h.
set(B70_DECODE_LIST_KERNELS
  kernel_embed_gather_M1 kernel_argmax_stage1_M1 kernel_argmax_stage2
  kernel_prep_res_fold_M1_K5120_SP0_G20 kernel_prep_res_fold_M1_K5120_SP16_G20
  kernel_prep_norm_finish_M1_K5120_G20_W20
  kernel_prep_silu_mul_M1 kernel_prep_gated_head_M1 kernel_gdn_step_M1
  kernel_attn_prep_M1 kernel_attn_decode_M1_L16384 kernel_attn_reduce_M1_L16384
  kernel_gemv_M1_K5120_N16384_S1_L1 kernel_gemv_M1_K5120_N14336_S1_L1
  kernel_gemv_M1_K6144_N5120_S16_L1 kernel_gemv_M1_K5120_N34816_S4_L1
  kernel_gemv_M1_K17408_N5120_S16_L1
  kernel_gemv_bf16_M1_K5120_N128_C16_S16 kernel_gemv_bf16_M1_K5120_N248320)

# The captured decode list, replayed: spec §8.7's determinism proof and the
# acceptance for every kernel in plan 3. Needs the real checkpoint and a B70
# (label `checkpoint`), and it is the one test that runs all 774 kernels.
add_executable(replay_determinism_test runtime/replay_determinism_test.cc)
target_include_directories(replay_determinism_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(replay_determinism_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(replay_determinism_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME replay_determinism_test COMMAND replay_determinism_test)
set_tests_properties(replay_determinism_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The same list driven through runtime::Engine - ingestion, greedy generation,
# reset, the max_len wall and the timing the CLI prints. Same checkpoint, same
# card, so the same label; two runs of 16 + 8 tokens rather than three of 16 + 8
# with full state snapshots, so it is the cheaper of the two.
add_executable(engine_smoke_test runtime/engine_smoke_test.cc)
target_include_directories(engine_smoke_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(engine_smoke_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(engine_smoke_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME engine_smoke_test COMMAND engine_smoke_test)
set_tests_properties(engine_smoke_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The same list captured twice - plain and profiled (plan 4 spec §3.5). It is
# the profiler's calibration: the instrumented list must generate the same 8
# tokens as the plain one, its 774 timestamps must tile the step in walk order,
# and their sum must fit inside the fence wall. Same checkpoint and card as the
# two above, so the same label; ~49 replays of the real step, no oracle dumps.
add_executable(profile_capture_test runtime/profile_capture_test.cc)
target_include_directories(profile_capture_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(profile_capture_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(profile_capture_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME profile_capture_test COMMAND profile_capture_test)
set_tests_properties(profile_capture_test PROPERTIES LABELS checkpoint TIMEOUT 900)

# The golden gate (spec §11): the engine against the CPU oracle, 3 prompts x 32
# greedy tokens, element-exact, plus the per-layer / GDN-state / logit-row
# diagnostics. Needs what engine_smoke_test needs AND the ~1 GB of oracle dumps
# that live only on the box - hence the second label and the SKIP_RETURN_CODE:
# absent goldens make the test exit 77, which ctest reports as SKIP rather than
# a failure. Both directories are passed absolute so the test does not depend on
# ctest's working directory; the prompt ids are committed, so a missing .ids
# file is a real failure and not a skip.
add_executable(golden_gate_test golden/golden_gate_test.cc)
target_include_directories(golden_gate_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(golden_gate_test PRIVATE b70_runtime b70_loader b70_l0 b70_model)
add_dependencies(golden_gate_test ${B70_DECODE_LIST_KERNELS})
add_test(NAME golden_gate_test COMMAND golden_gate_test
  ${CMAKE_SOURCE_DIR}/oracle-out ${CMAKE_SOURCE_DIR}/tests/golden/prompts)
set_tests_properties(golden_gate_test PROPERTIES
  LABELS "checkpoint;golden" SKIP_RETURN_CODE 77 TIMEOUT 1800)

# --- b70-decode's host-side rejections ---------------------------------------
# The product binary had no automated test at all. These are the argument
# rejections that short-circuit BEFORE `l0::Context` is constructed
# (src/cli/b70_decode.cc: the parse loop, the mode checks, then `read_ids`) - so
# they need neither a GPU nor the 19 GB checkpoint, and they carry no label:
# they belong in the plain suite that runs everywhere.
#
# Each case runs the real binary and requires **exit 1**, the code `main`'s
# catch-all returns. `WILL_FAIL TRUE` would have been shorter and wrong: it
# accepts *any* non-zero status, and a segfault is non-zero, so a CLI that
# crashed on a bad flag would pass it. The `sh -c` wrapper pins the exact
# number, so exit 0, exit 2 and a signal all fail.
#
# It also greps the output for the `device:` line the CLI prints immediately
# after `l0::Context` is built. That is the guard on this block's own premise:
# if one of these rejections ever moved to after the device is opened it would
# still exit 1 (a bogus model path throws in the loader), the test would still
# pass, and it would silently have stopped testing what it says it tests.
# `EXPECT <substring>` pins WHICH rejection fired. Without it a case only
# asserts "exit 1 before the device", which every unknown flag satisfies too -
# so a test for a flag the parser does not know yet passes for the wrong reason
# and keeps passing if the feature is deleted. The default expectation is
# `b70-decode:`, the prefix `main`'s catch-all prints on every rejection (and
# which `usage()`'s own "b70-decode <snapshot-or-repo>" lines do not match), so
# the nine cases written before this argument existed gain a free check and
# none of them changes.
function(b70_cli_reject name)
  cmake_parse_arguments(R "" "EXPECT" "" ${ARGN})
  if(NOT R_EXPECT)
    set(R_EXPECT "b70-decode:")
  endif()
  add_test(NAME ${name} COMMAND sh -c [==[
expect=$1
shift
out=$("$@" 2>&1)
rc=$?
printf '%s\n' "$out" >&2
if [ "$rc" -ne 1 ]
then
  echo "b70_cli_reject: expected exit 1, got $rc" >&2
  exit 1
fi
if printf '%s\n' "$out" | grep -q '^device:'
then
  echo "b70_cli_reject: the rejection happened after l0::Context was built" >&2
  exit 1
fi
if ! printf '%s\n' "$out" | grep -qF -- "$expect"
then
  echo "b70_cli_reject: exit 1, but the message does not contain '$expect'" >&2
  exit 1
fi
]==] b70_cli_reject "${R_EXPECT}" $<TARGET_FILE:b70-decode> ${R_UNPARSED_ARGUMENTS})
  set_tests_properties(${name} PROPERTIES TIMEOUT 60)
endfunction()

# The model argument is a path that does not exist, on purpose: every case below
# must reject before anything tries to open it, and if one stops doing so the
# `device:` grep above turns that into a failure rather than a false pass.
set(B70_NO_MODEL /nonexistent/b70-model)
# tests/cli/ok.ids is three ids from the bench prompt, all in vocabulary. Cases
# that are not about the file's contents use it deliberately, so that a pass
# means the flag combination was rejected and not that the file was bad.
set(B70_OK_IDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/ok.ids)
# tests/cli/oov.ids is one legal id followed by 248320 - `Qwen35::kVocab`
# exactly, the first row `read_ids` rejects (rows [kVocabUsed, kVocab) are
# legitimate; beyond kVocab embed_gather would read past the allocation).
set(B70_OOV_IDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/oov.ids)

# "exactly one of --ids and --bench" - the two modes are exclusive.
b70_cli_reject(cli_reject_ids_and_bench ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --bench)
# "--n belongs to --ids" - --bench sizes its run with --depth and --tg.
b70_cli_reject(cli_reject_n_with_bench ${B70_NO_MODEL} --bench --n 4)
# "--depth and --tg belong to --bench" - the mirror of the case above.
b70_cli_reject(cli_reject_depth_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --depth 64)
b70_cli_reject(cli_reject_tg_with_ids ${B70_NO_MODEL} --ids ${B70_OK_IDS} --n 4 --tg 8)
# "unknown option" - anything starting with '-' the parse loop does not know.
b70_cli_reject(cli_reject_unknown_flag ${B70_NO_MODEL} --bench --frobnicate)
# "a snapshot directory or HF repo id is required" - no positional argument.
b70_cli_reject(cli_reject_no_model --bench)
# "--ids needs --n" - the mode is chosen but not sized.
b70_cli_reject(cli_reject_ids_without_n ${B70_NO_MODEL} --ids ${B70_OK_IDS})
# "cannot open --ids file" - a typo'd prompt path must not cost a 13-second load.
b70_cli_reject(cli_reject_missing_ids_file ${B70_NO_MODEL} --ids /nonexistent/prompt.ids --n 4)
# "id 248320 is outside the vocabulary" - read_ids' bound, from the fixture.
b70_cli_reject(cli_reject_oov_id ${B70_NO_MODEL} --ids ${B70_OOV_IDS} --n 4)
# --profile is a third mode, exclusive with the other two: it replays an
# INSTRUMENTED list (774 host-scope event signals per step), so it can never
# produce a bench row, and it sizes its own run with --depth and --steps rather
# than with --ids/--n. Both rejections are the mode check, host-side - and both
# name the message, because "exit 1 before the device" is also what an unknown
# flag does: without EXPECT these two would pass on a binary that had never
# heard of --profile.
b70_cli_reject(cli_reject_profile_and_bench ${B70_NO_MODEL} --profile --bench
  EXPECT "exactly one of --ids, --bench and --profile")
b70_cli_reject(cli_reject_profile_and_ids ${B70_NO_MODEL} --profile --ids ${B70_OK_IDS} --n 4
  EXPECT "exactly one of --ids, --bench and --profile")
# "--depth 0 would ingest nothing" - both synthetic modes measure a step at a
# context depth, and depth 0 is not one.
b70_cli_reject(cli_reject_depth_zero_bench ${B70_NO_MODEL} --bench --depth 0
  EXPECT "--depth 0 would ingest nothing")
b70_cli_reject(cli_reject_depth_zero_profile ${B70_NO_MODEL} --profile --depth 0
  EXPECT "--depth 0 would ingest nothing")
