# Spec 19 v3c - `--spec dflash` under tensor parallel (`--parallel tp`) (open-work item 12)

**Status (2026-11-13): planned; expected to stop at Task 1.** Spec:
`docs/superpowers/specs/2026-11-13-spec19v3-dflash-modes-design.md` §4 (mode C). The spec derives ~114 t/s against
MTP's 123.1 under tensor parallel (0.93x) with the drafter whole on one card. Task 1 measures the number that
derivation hinges on - the verify step at 8 rows under tensor parallel - and decides whether Tasks 2-7 run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min in one blocking wait.

**Goal:** `b70-serve --parallel tp --spec dflash` serves Qwen3.8-27B at the full 262,144 context with greedy output
judged equal to `--parallel tp --spec off`, and is at least 1.05x `--parallel tp --spec mtp` on the opencode replay.
If Task 1 shows that cannot be reached, the goal becomes its record: the measured verify cost and what would have
to change.

**Design (spec §4.2):**
- the drafter whole on card 1, with the int4 candidate head and card 0's half of the int8 head beside it;
- the taps copied locally on card 1 (both cards hold the same residual after every fold);
- a round in two submits: card 1 runs [pending projection, draft]; the host mirrors the drafts into card 0's
  Control; both cards run their verify lists; the host merges, accepts and commits on both.

**Order and why:** Task 1 builds only what a measurement needs (per-card binaries at 5..8 rows, additive) and
measures. Tasks 2-5 are the engine; Task 6 the server; Task 7 the record.

**Stop rule (Task 1):** with `v8` the measured verify at 8 rows in ms and the TP plain step `p` in ms, the derived
round is `v8 + 8.43 + 1.0` ms and the derived serving-bench speed `4.04 / round`. Below **129 t/s** (1.05 x 123.1;
that is `v8` above 21.9 ms, ~1.16 plain steps) the plan stops: record the table, bring it to the operator with the
two levers (the draft pass at spec 19 v2's 4.5 ms target; the drafter split over both cards). The spec's derivation
expects `v8` ~26 ms, so **the expected outcome is the stop.**

## Dependencies

- Main with spec 17d (MTP under tensor parallel) and spec 19 v2's branch merged. Plan 19v3a is not needed (bf16 KV
  holds the full context under TP); the int8 cache under TP with a drafter follows from 19v3a's rows once both are in.
- Branch `spec19v3c-dflash-tp`, own worktree under `.claude/worktrees/`.
- **Both cards, with P2P between them, for every GPU step:** `tools/gpu_lock.sh box -- ...`. Card 1 is the Flash-Next
  quantisation's from 2026-11-14: run Task 1 before it starts or after it ends, never beside it (a whole-box job
  waits behind the quantisation's card lock by itself; do not stop that job).
- Box data: the gate checkpoint, the drafter, the opencode recordings, the f32 judge's reference sets.

## Global constraints

- No new user flag and no environment variable. `--parallel auto` keeps choosing TP for one request; the proposer
  under it stays `--spec auto` = MTP. Test and probe overrides through `CaptureOptions`.
- **Nothing else moves:** `tools/kernel_cmdlines` shows additions only; a TP engine with the MTP head, or with no
  proposer, binds what it bound before (spec 17b-17d's tests unchanged).
- Every in-list wait stays bounded (TpEngine's rule): a round that loses a crossing throws within the timeout and
  `reset()` recovers. The two-submit round adds no unbounded wait.
- Build and test on the box (`tools/box.sh`). GPU command form:
  `tools/box.sh run "tools/gpu_lock.sh box -- <command>"`; long jobs
  `tools/box.sh run "mkdir -p ~/b70/logs/dflash-tp && nohup setsid tools/gpu_lock.sh box -- <command> > ~/b70/logs/dflash-tp/<name>.log 2>&1 &"`.
- Mask ssh output: `sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'`. Timed numbers: interleaved rounds after a
  warm-up, median of rounds, the grade written next to the number.

---

### Task 1: verify at 5..8 rows under tensor parallel, measured (box, both cards)

**Files:**
- `src/kernels/CMakeLists.txt`, the spec 17d block (`B70_TP AND B70_MTP`), under `B70_VERIFY_M8`, at M = 5..8:
  - `gdn_step_slots_M<M>_GK8V24_N8` (8 state slots at the card's stride; also M = 1..4 at 8 slots),
    `prep_gated_head_M<M>_GK8V24`, `prep_silu_mul_M<M>_I8704`;
  - `attn_prep_M<M>_Q12KV2`, `attn_v2_M<M>_T32_Q12KV2`, `attn_verify_dpas_M<M>_T32_Q12KV2` with `VDPAS_LEVERS`
    (the one-card 5..8 rows' levers; the pair split of the 2..4-row lines does not fit 6+ row blocks);
  - `tp_cross_M<M>_K5120_G20_S4_F_R{0,1}`, `tp_argmax_M<M>` for both vocabulary halves;
  - the per-card small-M GEMM rows: `kernels::kSmallMTuning`'s spec 17d rows from M = 2..4 to 2..8
    (`src/kernels/kernels.h`, and the CMake rows that mirror them).
- `src/runtime/verify_attn.h`: nothing if the per-card heads (12 / 2) read the one-card table at M = 5..8 as they do
  at 2..4; check `verify_attn_heads` and the test.
- `tests/kernels/tp_kernels_test.cc`, `tests/kernels/tp_shapes_test.cc`, `tests/runtime/verify_attn_test.cc`: the
  new names and the crossing at 5..8 rows (both cards bitwise each other).
- `tests/runtime/tp_mtp_test.cc`: its `cost` mode takes a row count up to 8 - the probe. It builds `MtpBuffers` at 8
  GDN slots (the form spec 19a's one-card probe used) and captures `build_tp_verify` at M = 5..8; the MTP head's
  tail stays in the list (its one-card binaries at 5..8 rows exist), so the number is an upper bound by the head's
  ten launches. No engine change.
- `tools/box_validate/stages.sh`: a new row (88, or the next free one), stages `.k0`, `.kernels`, `.cost`.

- [ ] **Step 1: build.** `tools/box.sh build`.
  - **Gate:** every new binary compiles; `tools/kernel_cmdlines`: additions only.
  - If the DPAS verify attention at 12 / 2 heads does not compile with `VDPAS_LEVERS` at some M, bind v2 at that M
    for the measurement and record it (the cost row is then an upper bound).
  - Commit `tp: the per-card verify binaries at M = 5..8 (spec 19 v3c Task 1)`.
- [ ] **Step 2: numerics.** The kernel tests, both cards:
  `tools/box.sh run "tools/gpu_lock.sh box -- ctest --test-dir build --output-on-failure -R '^(tp_kernels_test|tp_shapes_test|verify_attn_test|gemm_sm_test)$'"`.
  - **Gate:** pass, no skip: the crossing at 5..8 rows bitwise between the cards; the per-card small-M GEMM rows
    against fp64 and the per-row GEMV at 5..8 rows.
- [ ] **Step 3: the cost.** `tools/box.sh run "tools/gpu_lock.sh box -- build/tests/tp_mtp_test <target> cost 4096"`
  and `... cost 32768` with the row count extended (medians of 5 interleaved rounds of 16, the mode's own form).
  Check first that both cards are idle; write the grade.
  - Record in BENCHMARKS "Tensor parallel (spec 17)", a new block "verify at 5..8 rows (spec 19 v3c)": the K = 0
    iteration in ms and verify at 2..8 rows in its units, both depths.
- [ ] **Gate / stop condition:** apply the stop rule above to the depth-4096 row. Write the derived round, the
  derived t/s and the verdict into the spec's §4.4 table ("measured" replaces "derived" on the verify row).
  - **Stop:** commit, hand back to the operator with the table. Tasks 2-7 wait for a ruling.
  - **Go** (>= 129 t/s derived): continue.

### Task 2: the plan and the loader (host, then box)

**Files:** `src/runtime/tp_plan.{h,cc}` (`tp_card_plan` / `tp_max_len_that_fits` with a `DrafterPlan` on rank 1: the
drafter's weights and buffers, the other head half, 8 GDN slots at the card's shapes on both cards, the tap staging
on rank 1), `src/loader/` (`load_tp` keeps card 0's int8 head half on card 1 as well when a drafter is asked for;
`load_drafter` beside a TP card's model), `src/cli/dflash_args.h` (`plan_only` against the TP card),
`tests/model/tp_desc_test.cc` (it holds the TP plan's cases today), `tests/runtime/dflash_plan_test.cc`.

- [ ] **Step 1 (host tests):** test first; the plan test prints each card's plan at 262,144.
  - **Gate:** both cards fit with the 1.5 GB reserve (derived: card 1 ~24 GB, card 0 ~20 GB);
    `--max-len auto` = 262,144; a TP plan without a drafter is byte for byte what it was.
- [ ] **Step 2 (box):** the loader on both cards: the full head on card 1 equals the two halves concatenated,
  bitwise. Commit.

### Task 3: the verify lists with taps (box)

**Files:** `src/runtime/capture.{h,cc}` (`build_tp_verify_taps`: `build_tp`'s walk at M rows, the card's 8-slot
`gdn_step`, the crossings, `tp_argmax` at M rows, no MTP tail; with `tv.taps` set - rank 1 - the tap copies at
`tap_sites`), `src/runtime/tp_engine.{h,cc}`, `tp_engine_card.h` (a borrowed `TapDrafter*`; each card's 7 extra GDN
states; the lists at M = 1..max_k + 1, the deep DPAS list where the table switches by depth),
new `tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first:
  - the taps card 1 writes for a verify row equal, bitwise, card 0's residual at the same sites (read through the
    residual tap of `TpOptions::debug_resid`) - the property the design rests on;
  - verify(k) for k = 0..7 against k + 1 plain TP steps: ids equal or judged (the tie rule, then the f32 judge, as
    `tp_mtp_test` judges MTP's rows); accepting j rows leaves the GDN state of j + 1 plain steps;
  - a dropped list (`drop_next`) fails within the bounds at 8 rows.
- [ ] **Step 2 (box):** `tp_dflash_test`, `tp_mtp_test`, `tp_decode_test`, `tp_fail_test` under the box lock.
  - **Gate:** all pass, no skip. Commit.

### Task 4: the round (box)

**Files:** `src/runtime/tp_engine.{h,cc}` (`draft_tap(k, temperature, seed)`: card 1's [pending projection, draft]
submit and fence, the host mirror of `cur_token[1..k]` into card 0's Control; `verify` / `commit` with the drafter;
`flush_projection`; `tap_samples`, `drafter_key`; `read_logits_into` at up to 8 rows),
`src/runtime/dflash/dflash_drafter.{h,cc}` only if `attach_target` needs the full head passed apart from the card's
model, `tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first: a greedy session of 256 ids by rounds at K = 7 / 3 / 1 equals plain TP's ids (or is
  judged); the ring after every round equals the ring of the one-card engine fed the same committed rows' taps;
  the two Controls agree after every round; a round is exactly two queue submissions on card 1 and one on card 0.
- [ ] **Step 2 (box):** the test, then the round's parts as interleaved pairs against the TP plain step at depth 4096
  and 32768 (the draft submit, the mirror, the verify pair, the commit): BENCHMARKS, the same block.
  - **Gates:** the tests pass; the host time outside the device lists <= 0.8 ms a round; the measured round within
    5 % of Task 1's derived one. Commit.

### Task 5: prefill and snapshots (box)

**Files:** `src/runtime/prefill/tp_prefill.cc` (card 1 stages the taps of the prompt's last 2048 positions and
projects them 8 rows a replay), `src/runtime/tp_engine.cc`, `src/runtime/tp_gather.h` (`state_bytes` /
`save_state` / `load_state` with card 1's ring after the one-card layout), `tests/runtime/tp_prefill_test.cc`,
`tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first: after a TP prefill the ring equals the ring after the same prompt ingested by decode
  replays (within the prefill's judged form for the tap rows); save, reset, load, continue equals the unbroken
  session's drafts.
- [ ] **Step 2 (box):** the tests; the prefill's cost with and without the drafter at 4096 and 16384 prompt tokens
  (`b70-decode --bench --prefill-length N --parallel tp`, interleaved).
  - **Gates:** the tests pass. The prefill loss is **recorded** (derived ~8 % at 4096 tokens from the tail
    projection's ~120 ms; the fix is item 12's, not this plan's). Commit.

### Task 6: the server (box)

**Files:** `src/cli/pipeline_args.h` (`check_tp`: `dflash` is served; `dflash-lookup` and `--draft-vocab` stay
refused), `src/cli/b70_serve.cc` (the TP branch loads the drafter on card 1 and passes `dflash` to
`PipelineEngineAdapterT<runtime::TpEngine>`, which compiles its DFlash path for an engine with `draft_tap`),
`src/cli/b70_decode.cc`, `src/cli/tp_decode.h`, `src/server/adaptive_k.{h,cc}` (a TP drafter cost table from
Tasks 1 and 4's medians), `tests/cli/pipeline_args_test.cc`, `tests/server/adaptive_k_test.cc`,
`tools/box_validate/stages.sh` (the row's serving stages).

- [ ] **Step 1 (host tests):** the refusals, the cost table (the K the policy picks at acceptance 0.5 / 0.8 / 0.95).
- [ ] **Step 2 (box):** golden + tool-call sets through `b70-serve --parallel tp --spec dflash` against
  `--parallel tp --spec off`, judged by `tools/box_validate/serve_judge.py` (the form of spec 17d's serving stage);
  a prefix-cache-hit turn; a seeded sampled request twice.
  - **Gates:** no failure, no skip; the sampled request reproduces; `tools/box_validate.sh --quick` passes and G0
    shows additions only.
- [ ] **Step 3 (acceptance):** tokens a round, generation stopped at EOS, on the tool-call set and golden code.
  - **Gate:** within 0.15 of one card's 6.65 and 6.08. A larger gap is a defect in the taps or the ring.

### Task 7: the speed record (both cards idle)

- [ ] **Step 1:** the opencode replay `opencode-session-int8`, three arms (`--spec off`, `mtp`, `dflash`), all under
  `--parallel tp`, three rotated rounds, prefix cache on (`tools/prefix/replay_spec.py` without `--card`: it
  reserves the box; give it the mode through its recorded settings so `--resume` refuses a mixed dataset). The
  serving bench shape (3,425-token prompt, 512 tokens) for the README's row. Both cards idle, more than 16 CPU
  threads free, the conditions written next to the numbers.
- [ ] **Gate (the goal):** `--spec dflash` >= **1.05x** `--spec mtp` under TP on the replay, f32 judge 0 failures and
  0 skipped; prefill loss recorded.
  - Passed: propose nothing about defaults (spec 19 v2's G-speed still binds); record the row.
  - Missed: record the decomposition; the docs say "slower than MTP under tensor parallel" and the mode stays an
    option, or the operator rules the refusal back in.
- [ ] **Step 2: close.** README only on a pass and the operator's word; `docs/18-running-models.md`, BENCHMARKS, the
  spec's "as built" lines, open-work item 12.

**Which tasks need the box:**

| task | host only | both cards | CPU model job |
|---|---|---|---|
| 1 | - | steps 1-3 (step 3 idle) | - |
| 2 | step 1 | step 2 | - |
| 3 | - | yes | only for new f32 references |
| 4 | - | yes (step 2 idle) | - |
| 5 | - | yes | - |
| 6 | step 1 | steps 2-3 | step 2, only for new f32 references |
| 7 | - | yes, idle | - |

Size: Task 1 alone about 1-1.5 agent-days (mostly kernel variants and the wait for both cards). Tasks 2-7 about 5-6
agent-days more, only after a "go". Task 1's build and Tasks 2-6 go to Opus; the timed runs of Tasks 1, 4, 5 and 7 to
a Sonnet agent.
