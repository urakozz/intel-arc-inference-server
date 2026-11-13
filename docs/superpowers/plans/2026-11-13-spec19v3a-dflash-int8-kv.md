# Spec 19 v3a - `--spec dflash` with `--kv-cache int8` (open-work item 12)

**Status (2026-11-13): implemented on `spec19v3a-dflash-int8-kv`; focused host/kernel check passed; long gates pending.** Spec: `docs/superpowers/specs/2026-11-13-spec19v3-dflash-modes-design.md` §2
(mode A). Short by design: the kernel sources exist; this plan builds three binaries at 5..8 rows, adds four table
rows and removes one refusal.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min in one blocking wait.

**Operator override (2026-11-13):** implement immediately and run only a quick test after implementation. The
per-step builds, RED/GREEN runs, paired-cost stop decision, full checkpoint/PP/quick-validation, CPU acceptance,
f32 serving and speed runs below remain unrun gates. Their commands and bars are retained for later; implementation
is not a claim that this mode has passed them. Validation stages are registered as row 86. `--spec auto` is unchanged.

**Focused check:** all twelve new kernels compile; kernel command lines show exactly twelve additions, no
existing changes/removals. Five host tests pass; planner **246,528**. Replay harness 12/12 and validation
regressions 2/2 pass. Three GPU kernel tests pass, no skipped tests (33.51 s on card 1 while card 0 was occupied): new int8 attention
at 512/4096/32768 keys, writer/reduce and unchanged-attention control. Property R is bitwise in 100/100 trials
at each depth. Evidence: `~/b70/logs/dflash-int8-v3a/{quick,quick-final}.log`; initial wrapper
ended rc128 on a driver dry-run requiring git metadata, resumed kernel check ended `QUICK_PASS`, rc0.
The driver dry-run/registry checks passed on the Mac checkout. Full CLI build, G0 binary comparison, checkpoint twins, PP,
acceptance, f32 serving and speed remain pending.

**Goal:** `b70-serve --spec dflash --kv-cache int8` serves on one card and on `--parallel pp`, greedy output judged
equal to `--spec off --kv-cache int8`, with `--max-len auto` at or above 200,000 on one card and a derived
~1.15-1.18x against `--spec mtp --kv-cache int8` on the opencode replay.

**Design (spec §2.2):** the verify lists at 5..8 rows over the int8 cache bind the DPAS verify attention at every
depth (`dpas_from = 0`), the int8 writer at 5..8 rows and the int8 reduce from a lifted `kv8.cl` bound. The drafter,
its ring and the draft list do not change.

**Order and why:** Task 1 builds the binaries and measures the one unknown (the int8 verify at 5..8 rows with today's
kernel) before anything is wired; Tasks 2-3 wire and gate it; Task 4 is the speed record.

**Stop rule:** Task 1 stops the plan when the int8 verify at 8 rows makes the derived replay ratio fall below 1.05x
MTP at depth 4096 (its Step 3 has the formula). The result is recorded and the operator decides.

## Dependencies

- Branch `spec19v3a-dflash-int8-kv` from `spec19v2-dflash-speed` (or from main once that is merged), own worktree
  under `.claude/worktrees/`. `tools/box.sh` gives the branch its own tree on the box.
- Box data: the gate checkpoint, the drafter `z-lab/Qwen3.8-27B-DFlash2`, the opencode recordings under
  `~/b70/logs/`, the f32 judge's reference sets (`oracle-out-qwen-f32-i8head-forced`).
- One card is enough for every task. Use card 0: card 1 is the Flash-Next quantisation's from 2026-11-14.

## Global constraints

- No new user flag and no environment variable. The int8 cache in tests is the existing seam `B70_KV_CACHE=int8`
  (`docs/environment-variables.md`).
- **Nothing else moves:** `tools/kernel_cmdlines` shows additions only (the new `_KV8` binaries at M = 5..8); a
  bf16-KV engine and an int8-KV MTP engine bind what they bound before.
- The Mac never compiles C++. Build and test with `tools/box.sh build` / `tools/box.sh run "..."`. Mask any ssh
  output you paste: `sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'`.
- GPU command form, used below:
  `tools/box.sh run "tools/gpu_lock.sh card 0 -- <command>"`; a job over ~10 minutes goes detached:
  `tools/box.sh run "mkdir -p ~/b70/logs/dflash-int8 && nohup setsid tools/gpu_lock.sh card 0 -- <command> > ~/b70/logs/dflash-int8/<name>.log 2>&1 &"`.
- Timed numbers: interleaved pairs after a warm-up, median of rounds, labelled "idle card" or "busy box".

---

### Task 1: the binaries, their numerics and the cost (box)

**Files:**
- Modify `src/kernels/kv8.cl`: the single-sequence decode bound `#error "kv8 dec: M is 1..4"` becomes 1..8 (the
  `BATCH_ROWS` bound beside it already is). No other line.
- Modify `src/kernels/CMakeLists.txt`:
  - in the `B70_KV8` block, under `B70_VERIFY_M8`: `attn_prep_M<M>_KV8` and `attn_v2_M<M>_T32_KV8` at M = 5..8;
  - in the `B70_VERIFY_DPAS` block: `attn_verify_dpas_M<M>_T32_KV8` at M = 5..8 with `VDPAS_LEVERS` (the bf16
    5..8 rows' levers, not the pair split) and `KV8=1`.
- Modify `src/kernels/kernels.h`: the comment on `attn_verify_dpas_variant` ("2..4 int8" -> "2..8").
- Modify `tests/kernels/attn_verify_dpas_test.cc`, `tests/kernels/kv8_kernels_test.cc`: the int8 cases at M = 5..8.
- Modify `tools/box_validate/stages.sh`: a new row (86, or the next free one) with stages `.k0` (host checks and
  unchanged binaries), `.kernels` (this task's tests), `.cost` (Step 3), and later tasks' stages.

- [x] **Step 1: build.** The bound, the CMake lines, `tools/box.sh build`.
  - **Gate:** all twelve new binaries compile; `tools/kernel_cmdlines` against the branch point shows only additions.
  - If `attn_v2_M<M>_T32_KV8` does not compile at 5..8 rows (register pressure in the decode kernel), record the
    compiler's message and build the reduce from the row-table form's source path instead; do not change the
    decode kernel's arithmetic.
  - Commit `kv8: the int8 KV writer, reduce and DPAS verify attention at M = 5..8 (spec 19 v3a Task 1)`.
- [ ] **Step 2: numerics.** Extend the two kernel tests to M = 5..8 over int8 KV:
  - the writer's rows bitwise the host twin (`src/common/kv8.h`), as at M = 1..4;
  - `attn_verify_dpas_kv8` against the fp64 reference at 512 / 4096 / 32768 keys: cosine >= 0.99999 on every
    (query head, row), except the one row the operator ruled on 2026-11-03 (4096 keys, query head 19, row 0), whose
    bar is "no further from the reference than the prefill's int8 flash attention on that row";
  - a row's output does not depend on its neighbours (the test's property R), 100 trials.
  - Run: `tools/box.sh run "tools/gpu_lock.sh card 0 -- ctest --test-dir build --output-on-failure -R '^(attn_verify_dpas_test|kv8_kernels_test|attn_verify_bitwise_test)$'"`.
  - **Gate:** the three tests pass with no skip. Commit.
- [ ] **Step 3: the cost.** Add the table rows `{5, true, 0} .. {8, true, 0}` to `kVerifyAttnTable`
  (`src/runtime/verify_attn.h`) and their cases to `tests/runtime/verify_attn_test.cc`. Then time the real lists:
  - `tools/box.sh run "B70_KV_CACHE=int8 tools/gpu_lock.sh card 0 -- build/tools/probe/probe_dflash_iter <target> z-lab/Qwen3.8-27B-DFlash2 --format int8 --candidate-head --depth 4096"`,
    the same at `--depth 32768`, and the bf16 arm (no `B70_KV_CACHE`) in the same session for the pair.
  - If `probe_dflash_iter` builds its engine with a fixed KV form, give it the form through the engine constructor's
    `kv` argument read from `runtime::default_kv_cache()`; no new probe flag.
  - Record in BENCHMARKS "DFlash (spec 19)", a new block "int8 KV (spec 19 v3a)": verify at 1..8 rows, the draft and
    the K = 7 round, in ms and in paired plain steps, both KV forms, both depths.
  - **Gate (objective):** every row has a value with its grade.
  - **Stop condition:** derived ratio = 1.19 x (bf16 round / int8 round) x (72.3 / 71.7). Below **1.05** at depth
    4096: stop, record, ask the operator. (Derived expectation: 1.18.)
  - Commit `verify_attn: int8 rows at M = 5..8 bind DPAS at every depth; the int8 DFlash round measured`.

### Task 2: the engine serves it (box)

**Files:**
- Modify `src/cli/dflash_args.h`: `check_engine` loses the `KvCache::Int8` refusal (the parameter stays for the
  callers); the header comment's "What is refused here" loses its line.
- Modify `tests/cli/` (the DFlash argument test beside `pipeline_args_test.cc`, or `pipeline_args_test.cc` itself if
  it holds the refusal's case): the refusal's case becomes an acceptance case.
- Modify `tests/runtime/dflash_engine_test.cc`, `tests/server/dflash_gpu_test.cc`,
  `tests/server/dflash_prefix_gpu_test.cc`, `tests/runtime/pp_dflash_test.cc`: nothing in the sources if they take
  the KV form from `runtime::default_kv_cache()`; register an `_kv8` twin of each in the tests' CMake file the way
  `tp_mtp_kv8_test` is registered (the same binary with `B70_KV_CACHE=int8`).
- Modify `tests/runtime/dflash_plan_test.cc` or `tests/runtime/memory_plan_test.cc`: the plan with a drafter and the
  int8 cache (the drafter's term unchanged, the KV term int8's, `max_len_that_fits` at the 32.53 GB card printed).

- [x] **Step 1 (host):** the refusal, the argument test, the plan test.
  - Run: `tools/box.sh test '^(pipeline_args_test|dflash_plan_test|memory_plan_test|verify_attn_test|dflash_engine_host_test)$'`.
  - **Gate:** pass; the plan test prints `--max-len auto` for one card with the drafter, the candidate head and the
    int8 cache, and it is **>= 200,000** (derived 225-245k). Record the printed value in the spec's §2.3 table.
  - Commit `dflash: --kv-cache int8 is served (the refusal removed, the plan at int8)`.
- [ ] **Step 2 (box, one card):** the four `_kv8` twins. They run for ~30 minutes together: detached, card 0.
  - What they check, over the int8 cache: the ring after deferred projections equals the undeferred one at every
    flush point; greedy DFlash at K = 7 / 3 / auto against plain decode on the golden and tool-call sets with the
    f32 judge on first differences; a turn served as a prefix cache hit drafts as the unbroken session.
  - **Gate:** all pass, no skip, no f32 failure.
- [ ] **Step 3 (box, both cards, 2 minutes):** `pp_dflash_kv8_test` under `tools/gpu_lock.sh box --`. If card 1 is
  held by the quantisation, leave this step open, say so in the handoff, and go on; it does not gate Tasks 3-4.
  - **Gate:** pass.
- [ ] **Step 4:** `tools/box_validate.sh --quick` on the branch head (about an hour, detached by the tool).
  - **Gate:** the quick run passes; G0's comparison shows the twelve additions and no removal. Commit the summary's
    facts into the handoff, not the summary file.

### Task 3: acceptance and serving correctness (box + CPU)

- [ ] **Step 1: acceptance over the int8 cache.** The engine-taps harness of spec 19 v2 Task 1 with the int8 cache:
  `tools/box.sh run "B70_KV_CACHE=int8 tools/gpu_lock.sh card 0 -- tools/box_validate/dflash_v2.sh gpu ~/b70/logs/dflash-int8/accept <target> z-lab/Qwen3.8-27B-DFlash2"`
  (run `dflash_v2.sh sources` into the same directory first). Read the EOS-stopped lines: tokens a round on the
  tool-call set and on golden code.
  - **Gate:** within 0.15 of the bf16 run's 6.65 (tool calls) and 6.08 (code). A larger gap is a defect: find it
    (the taps are the int8 run's own hidden states, so start at the tap rows of a verify at 5..8 rows against the
    same positions' plain steps) with a failing test before going on.
- [ ] **Step 2: serving, greedy.** One golden + tool-call round through `b70-serve --spec dflash --kv-cache int8`
  against the saved `--spec off --kv-cache int8` outputs (`tools/box_validate/dflash_v2.sh identity` with
  `B70_KV_CACHE=int8`; make the plain int8 reference round first if the validation state has none), judged by
  `tools/box_validate/serve_judge.py`.
  - New f32 references, if the judge asks for any, are CPU jobs: take `/tmp/b70-cpu-model.lock`, at most eight
    threads, and check the free RAM first.
  - **Gate:** every request passes: identical, or its first difference judged a near tie. No failure, no skip.
- [x] **Step 3: docs for the mode.** `docs/18-running-models.md` (the DFlash section's "Refused" list, the feature
  table's row, a one-card int8 recipe with the planner's context), `README.md`'s flag table only if a line there says
  bf16 only. Commit.

### Task 4: the speed record (one idle card)

- [ ] **Step 1:** the opencode replay `opencode-session-int8` (78 requests), three arms rotated over three rounds,
  prefix cache on, all over the int8 cache:
  `tools/box.sh run "mkdir -p ~/b70/logs/dflash-int8 && B70_KV_CACHE=int8 nohup setsid python3 tools/prefix/replay_spec.py ~/b70/logs/opencode-session-int8 ~/b70/logs/dflash-int8/replay --card 0 > ~/b70/logs/dflash-int8/replay.log 2>&1 &"`
  (`replay_spec.py` takes the card lock itself with `--card`). Before it starts, check that the card is idle and
  more than 16 of the 44 CPU threads are free; write the conditions next to the number.
  - If `replay_spec.py` does not pass the environment to its servers, add the KV form to its recorded settings (so
    `--resume` refuses a mixed dataset) rather than a new option on the server.
- [ ] **Step 2:** the same three arms at depth: `llama-benchy` tg128 at depth 32768, and one decode bench at 120,000
  keys of context (`b70-decode --bench --depth 120000 --spec dflash --kv-cache int8 --max-len auto` against
  `--spec mtp`), each as interleaved pairs through `tools/box_validate/interleave.sh`.
- [ ] **Gate (recorded, not a default change):**
  - the replay's DFlash / MTP ratio over the int8 cache, with the f32 judge at 0 failures and 0 skipped;
  - no arm slower than plain decode;
  - the depth rows recorded. A ratio below 1.0 at 120,000 keys is recorded plainly in `docs/18-running-models.md`
    ("slower than MTP past N keys") - it does not fail the plan.
- [ ] **Step 3: close.** BENCHMARKS "int8 KV (spec 19 v3a)" with the final numbers; the spec's §2.3 and §2.4 get an
  "as built" line each; open-work item 12 gets one sentence. `--spec auto` is unchanged.

**Which tasks need the box:**

| task | host tests | one card | both cards | CPU model job |
|---|---|---|---|---|
| 1 | - | steps 1-3 | - | - |
| 2 | step 1 | steps 2, 4 | step 3 (2 minutes) | - |
| 3 | - | steps 1-2 | - | step 2, only for new references |
| 4 | - | steps 1-2 (idle card) | - | - |

Size: 4 tasks, about 1.5-2 agent-days, most of it waiting on tests. Test and benchmark runs (Task 2 Steps 2-4,
Tasks 3-4) go to a Sonnet agent; Task 1 and Task 2 Step 1 to Opus.
