# Spec 19 v3c - `--spec dflash` under tensor parallel (`--parallel tp`) (open-work item 12)

**Status (2026-11-13): planned (revised: the drafter is split over both cards).** Spec:
`docs/superpowers/specs/2026-11-13-spec19v3-dflash-modes-design.md` §6 (mode C), §2 (what the reference engines do).
Second in the order, after plan 19v3d (the round's cost on one card). Task 1 is a measurement and may run earlier.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min in one blocking wait.

**Goal - the operator's target:** `b70-serve --parallel tp --spec dflash` at **170-190 t/s** for one session on the
serving bench (3,425-token prompt, 512 tokens; MTP under TP is 123.1), at the full 262,144 context, greedy output
judged equal to `--parallel tp --spec off`. That is a round of ~22-24 ms at 4.0-4.6 tokens a round.

**What the derivation says (spec §6.4):** today's kernels give ~127 t/s; after plan 19v3d ~157-166; with bf16
crossing messages ~166-173 at the bench's 4.04 tokens a round. 170 is within reach; 190 needs ~4.6 tokens a round
or a ~22 ms round.

**Design (spec §6.2):**
- the drafter split over both cards like the target: column-parallel `q||k||v` (by head), `gate||up`, context
  `k||v`; row-parallel `o` and `down` with an all-reduce each (ten a draft pass); norms, conv projections, `fc`, the
  selector replicated; the ring split by KV head;
- the draft head by vocabulary half: each card its candidate half, its top 64, the exact re-rank against its own
  half of the target's head, its top 16; the two lists merged on the host first, on the device if that pays;
- taps copied locally on each card;
- a round in two phases: [verify on both cards] - host accepts and commits - [projection, draft on both cards] -
  merge, selector - host writes the drafts into both Controls;
- the context store is our deferred projection (a captured list driven by its control block), per card.

**Order and why:** Task 1 measures the one number the target hinges on. Tasks 2-4 build the split drafter and prove
it equals one card's. Tasks 5-6 are the engine's round. Task 7 the session's edges, Task 8 the server, Task 9 the
record and the crossing lever.

**Stop rule (Task 1):** with `v8` the measured verify at 8 rows under TP in ms:
- run **after** plan 19v3d's verify half: `v8` above **22.5 ms** stops the plan (with a 3.0 ms split draft and 1.0 ms
  of host the round is then above 26.5 ms: below 153 t/s at 4.04 tokens). Bring the table and the crossing's share
  to the operator.
- run **before** it: first subtract what plan 19v3d is to remove, `(1.245 - 1.05) x 18.9 = 3.7 ms`; the same
  22.5 ms bar applies to `v8 - 3.7`. (Derived expectation: `v8` ~26 ms today, ~21.7 ms after.)

## Dependencies

- Main with spec 17d (MTP under tensor parallel) and spec 19 v2's branch merged; plan 19v3d merged before Task 2
  (its drafter attention and head path are what this plan splits - splitting today's kernels and then again the new
  ones would be the work twice).
- Branch `spec19v3c-dflash-tp`, own worktree under `.claude/worktrees/`.
- **Both cards, with P2P between them, for every GPU step:** `tools/gpu_lock.sh box -- ...`. Card 1 is the Flash-Next
  quantisation's from 2026-11-14: run two-card steps before it starts or after it ends, never beside it (a
  whole-box job waits behind the quantisation's card lock by itself; do not stop that job).
- Box data: the gate checkpoint, the drafter, the opencode recordings, spec 19 v2's acceptance baseline, the f32
  judge's reference sets.

## Global constraints

- No new user flag and no environment variable. `--parallel auto` keeps choosing TP for one request; the proposer
  under it stays `--spec auto` = MTP until the operator rules otherwise. Test and probe overrides through
  `CaptureOptions`.
- **Nothing else moves:** `tools/kernel_cmdlines` shows additions only; a TP engine with the MTP head, or with no
  proposer, binds what it bound before (spec 17b-17d's tests unchanged).
- Every in-list wait stays bounded (TpEngine's rule): a list that loses a crossing throws within the timeout and
  `reset()` recovers. That now includes the draft list's ten all-reduces.
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
- `tests/kernels/tp_kernels_test.cc`, `tests/kernels/tp_shapes_test.cc`, `tests/runtime/verify_attn_test.cc`: the
  new names and the crossing at 5..8 rows (both cards bitwise each other).
- `tests/runtime/tp_mtp_test.cc`: its `cost` mode takes a row count up to 8 - the probe. It builds `MtpBuffers` at 8
  GDN slots (the form spec 19a's one-card probe used) and captures `build_tp_verify` at M = 5..8; the MTP head's
  tail stays in the list, so the number is an upper bound by the head's ten launches. It also prints, at 8 rows, the
  crossings' share: the same list with the sends and waits timed by per-launch events. No engine change.
- `tools/box_validate/stages.sh`: a new row (the next free one), stages `.k0`, `.kernels`, `.cost`.

- [ ] **Step 1: build.** `tools/box.sh build`.
  - **Gate:** every new binary compiles; `tools/kernel_cmdlines`: additions only.
  - If the DPAS verify attention at 12 / 2 heads does not compile with `VDPAS_LEVERS` at some M, bind v2 at that M
    for the measurement and record it (the cost row is then an upper bound).
  - Commit `tp: the per-card verify binaries at M = 5..8 (spec 19 v3c Task 1)`.
- [ ] **Step 2: numerics.**
  `tools/box.sh run "tools/gpu_lock.sh box -- ctest --test-dir build --output-on-failure -R '^(tp_kernels_test|tp_shapes_test|verify_attn_test|gemm_sm_test)$'"`.
  - **Gate:** pass, no skip: the crossing at 5..8 rows bitwise between the cards; the per-card small-M GEMM rows
    against fp64 and the per-row GEMV at 5..8 rows.
- [ ] **Step 3: the cost.** `tools/box.sh run "tools/gpu_lock.sh box -- build/tests/tp_mtp_test <target> cost 4096"`
  and `... cost 32768` with the row count extended (medians of 5 interleaved rounds of 16). Both cards idle; write
  the grade.
  - Record in BENCHMARKS "Tensor parallel (spec 17)", block "verify at 5..8 rows (spec 19 v3c)": the K = 0
    iteration in ms, verify at 2..8 rows in its units, the crossings' ms at 1 and 8 rows, both depths.
- [ ] **Gate / stop condition:** apply the stop rule above to the depth-4096 row. Write the measured verify, the
  derived round and t/s into the spec's §6.4 table.
  - **Stop:** commit, hand the table to the operator with the two levers left (bf16 crossing messages, Task 9; the
    send fused into the linear before it). Tasks 2-9 wait for a ruling.
  - **Go:** continue once plan 19v3d is merged.

### Task 2: the drafter's split - loader and plan (host, then box)

**Files:** `src/model/dflash.{h,cc}` (the per-card descriptor: which linears are column-parallel, row-parallel,
replicated; 16 query / 4 KV heads a card), `src/loader/dflash_loader.{h,cc}`, `dflash_repack.{h,cc}`,
`dflash_layout.h` (each card's halves; the candidate head's half made from the card's own int8 head half),
`src/runtime/dflash/dflash_sizes.{h,cc}` (a card's buffers: the ring at 4 KV heads, the scratch at half widths),
`src/runtime/tp_plan.{h,cc}` (`tp_card_plan` / `tp_max_len_that_fits` with a per-card `DrafterPlan`; 8 GDN slots at
the card's shapes; tap staging on both cards), `src/cli/dflash_args.h`, `tests/loader/` (a `tp_slice_test`-style
test for the drafter), `tests/model/tp_desc_test.cc` (it holds the TP plan's cases today),
`tests/runtime/dflash_plan_test.cc`.

- [ ] **Step 1 (host tests):** test first:
  - the two cards' halves of every split linear concatenate (columns) or sum over K (rows) bitwise to the one-card
    weight; the replicated tensors are equal;
  - each card's plan at 262,144 printed.
  - **Gate:** pass; both cards fit with the 1.5 GB reserve (derived ~21.5 GB each); `--max-len auto` = 262,144; a
    TP plan without a drafter is byte for byte what it was.
- [ ] **Step 2 (box):** the loader on both cards against the host test's bytes. Commit.

### Task 3: a card's draft list (box)

**Files:** `src/runtime/dflash/dflash_capture.{h,cc}` (`build_dflash_draft_tp`: the one-card walk at the card's
shapes; after `o` and `down` a `tp_send` and, in place of the next fold, the wait-and-fold - the target's crossing
kernels at 8 rows; the head as the card's candidate half, its top 64, the exact re-rank against the card's int8 head
half, its top 16 with exact scores into a per-card candidate buffer), `src/kernels/CMakeLists.txt` and
`src/kernels/kernels.h` (the drafter's `gemm_sm` rows at the half shapes; plan 19v3d's drafter attention at 16 / 4
heads; `tp_cross` at the drafter's fold), `src/kernels/dflash/` (only defines: the head counts and the vocabulary
half's id base), `tests/kernels/dflash_kernels_test.cc`, new `tests/runtime/tp_dflash_draft_test.cc`.

- [ ] **Step 1:** test first (`tp_dflash_draft_test`, both cards, the drafter alone - ring and taps filled from a
  one-card dump):
  - after every all-reduce both cards hold the same hidden rows, bitwise;
  - the block's final hidden rows against the one-card draft list's: within the drafter's f32-judge form (the
    folded sums round differently from one card's chain);
  - the union of the two cards' top 16 contains the one-card list's 16 candidates on the baseline's draft records;
  - a dropped list fails within the bounds.
- [ ] **Step 2 (box):** the test; the draft pass timed against the one-card pass (interleaved).
  - **Gates:** pass, no skip; the split draft pass (both cards, before the merge) <= **0.60 x** the one-card pass
    in ms on the same kernels. Commit.

### Task 4: the candidate merge and the selector (box)

**Files:** `src/runtime/tp_engine.{h,cc}` or a small drafter-side class beside `DFlashDrafter` (the merge),
`src/runtime/dflash/dflash_capture.{h,cc}` (the tail list on card 1: the selector's projection and `df_select` over
the merged 16), `src/kernels/dflash/df_select.cl` only if the merged candidates need another input layout,
`tests/runtime/tp_dflash_draft_test.cc`.

- [ ] **Step 1: the host merge.** After both cards' draft fences the host merges the two top-16 lists per row
  (exact score, ties to the lower id), writes the 16 into card 1's candidate buffer and submits the tail list.
  - **Gate:** drafts and the 16 candidates equal the one-card drafter's on the baseline's records wherever the
    candidate sets are equal; acceptance on all 35 corpus / K rows within 0.05 tokens a round of one card's.
- [ ] **Step 2: the device merge (a lever).** Card 0's list ends by pushing its 16 into card 1's memory with the
  crossing's flag protocol; card 1's list waits, merges and runs the selector - no host wait between.
  - **Gate:** the same drafts as Step 1, bitwise; the round's host time lower by >= **0.2 ms**, or the lever is
    removed and Step 1 serves.

### Task 5: the verify lists with taps (box)

**Files:** `src/runtime/capture.{h,cc}` (`build_tp_verify_taps`: `build_tp`'s walk at M rows, the card's 8-slot
`gdn_step`, the crossings, `tp_argmax` at M rows, no MTP tail, the tap copies at `tap_sites` into the card's tap
buffer), `src/runtime/tp_engine.{h,cc}`, `tp_engine_card.h` (each card's 7 extra GDN states; the lists at
M = 1..max_k + 1, the deep DPAS list where the table switches by depth), new `tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first:
  - the two cards' taps of a verify row are bitwise each other - the property the design rests on;
  - verify(k) for k = 0..7 against k + 1 plain TP steps: ids equal or judged (the tie rule, then the f32 judge, as
    `tp_mtp_test` judges MTP's rows); accepting j rows leaves the GDN state of j + 1 plain steps;
  - a dropped list (`drop_next`) fails within the bounds at 8 rows.
- [ ] **Step 2 (box):** `tp_dflash_test`, `tp_mtp_test`, `tp_decode_test`, `tp_fail_test`.
  - **Gate:** all pass, no skip. Commit.

### Task 6: the round (box)

**Files:** `src/runtime/tp_engine.{h,cc}` (`draft_tap(k, temperature, seed)`: phase 2 on both cards - [the pending
projection, the draft], the merge, the tail, the drafts into both Controls; `verify` / `commit` with the drafter;
`flush_projection`; `tap_samples`, `drafter_key`; `read_logits_into` at up to 8 rows),
`src/runtime/dflash/dflash_capture.{h,cc}` (a card's projection list: the replicated `fc`, the card's `k||v`
columns, the store into its ring half), `tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first: a greedy session of 256 ids by rounds at K = 7 / 3 / 1 equals plain TP's ids (or is
  judged); each card's ring half after every round equals the same heads of the one-card engine's ring fed the
  same committed rows (within the judged form); the two Controls agree after every round; a round is exactly two
  submits a card (three on card 1 with the host merge).
- [ ] **Step 2 (box):** the test, then the round's parts as interleaved pairs against the TP plain step at depth 4096
  and 32768 (verify, the commit, projection + draft, the merge, the tail): BENCHMARKS, the same block.
  - **Gates:** the tests pass; host time outside the device lists <= **1.0 ms** a round; the measured round within
    5 % of the sum of Task 1's verify, Task 3's draft and that host time. Commit.

### Task 7: prefill and snapshots (box)

**Files:** `src/runtime/prefill/tp_prefill.cc` (each card stages the taps of the prompt's last 2048 positions and
projects them 8 rows a replay), `src/runtime/tp_engine.cc`, `src/runtime/tp_gather.h` (`state_bytes` / `save_state` /
`load_state`: the two ring halves gathered into the one-card ring layout after the one-card state),
`tests/runtime/tp_prefill_test.cc`, `tests/runtime/tp_dflash_test.cc`.

- [ ] **Step 1:** test first: after a TP prefill the ring equals the ring after the same prompt ingested by decode
  replays (within the prefill's judged form for the tap rows); save, reset, load, continue equals the unbroken
  session's drafts.
- [ ] **Step 2 (box):** the tests; the prefill's cost with and without the drafter at 4096 and 16384 prompt tokens
  (`b70-decode --bench --prefill-length N --parallel tp`, interleaved).
  - **Gates:** the tests pass. The prefill loss is **recorded** (derived ~8 % at 4096 tokens from the tail
    projection; its fix is spec 19 v2's open item, not this plan's). Commit.

### Task 8: the server (box)

**Files:** `src/cli/pipeline_args.h` (`check_tp`: `dflash` is served; `dflash-lookup` and `--draft-vocab` stay
refused), `src/cli/b70_serve.cc` (the TP branch loads the drafter's halves and passes `dflash` to
`PipelineEngineAdapterT<runtime::TpEngine>`, which compiles its DFlash path for an engine with `draft_tap`),
`src/cli/b70_decode.cc`, `src/cli/tp_decode.h`, `src/server/adaptive_k.{h,cc}` (a TP drafter cost table from
Tasks 1, 3 and 6's medians), `tests/cli/pipeline_args_test.cc`, `tests/server/adaptive_k_test.cc`,
`tools/box_validate/stages.sh` (the row's serving stages), `tools/prefix/replay_spec.py` (its server command
hard-codes `--parallel 1 --max-num-seqs 1 --kv-cache bf16`: give the tool a mode setting that it records, so
`--resume` refuses a mixed dataset).

- [ ] **Step 1 (host tests):** the refusals, the cost table (the K the policy picks at acceptance 0.5 / 0.8 / 0.95).
- [ ] **Step 2 (box):** golden + tool-call sets through `b70-serve --parallel tp --spec dflash` against
  `--parallel tp --spec off`, judged by `tools/box_validate/serve_judge.py` (the form of spec 17d's serving stage);
  a prefix-cache-hit turn; a seeded sampled request twice.
  - **Gates:** no failure, no skip; the sampled request reproduces; `tools/box_validate.sh --quick` passes and G0
    shows additions only.
- [ ] **Step 3 (acceptance):** tokens a round, generation stopped at EOS, on the tool-call set and golden code.
  - **Gate:** within 0.15 of one card's 6.65 and 6.08. A larger gap is a defect in the taps, the ring halves or
    the candidate merge.

### Task 9: the record, and the crossing lever (both cards idle)

- [ ] **Step 1:** the serving bench (3,425-token prompt, 512 tokens) and the opencode replay
  `opencode-session-int8`, three arms (`--spec off`, `mtp`, `dflash`), all under `--parallel tp`, three rotated
  rounds, prefix cache on. Both cards idle, more than 16 CPU threads free, the conditions written next to the
  numbers.
- [ ] **Step 2 (a lever, only if Step 1 is below 170 t/s and Task 1 put >= 1.5 ms of the 8-row verify in the
  crossings):** bf16 crossing messages in the M >= 2 lists (80 KiB instead of 160 KiB at 8 rows; the 1-row list
  untouched).
  - **Gate:** the cards still bitwise each other; verify(k) against plain TP judged as in Task 5 with no new
    failure; >= 0.8 ms saved at 8 rows. It changes MTP's TP verify lists too: `tp_mtp_test` and spec 17d's serving
    stage re-run. Bring the result to the operator before it is kept.
- [ ] **Gate (the goal):**
  - **>= 170 t/s** on the serving bench: the target's lower end is met. Record tokens a round beside it.
  - **140-170 t/s:** recorded with the round's decomposition (verify, draft, host, tokens a round) against the
    spec's §6.4 table, and which assumption missed.
  - **below MTP's 123.1:** the docs say "slower than MTP under tensor parallel"; the operator rules on the refusal.
  - In every case: the f32 judge at 0 failures and 0 skipped; prefill loss recorded; nothing about defaults
    (spec 19 v2's G-speed still binds).
- [ ] **Step 3: close.** README only on the target met and the operator's word; `docs/18-running-models.md`,
  BENCHMARKS, the spec's "as built" lines, open-work item 12.

**Which tasks need the box:**

| task | host only | both cards | CPU model job |
|---|---|---|---|
| 1 | - | steps 1-3 (step 3 idle) | - |
| 2 | step 1 | step 2 | - |
| 3 | - | yes (step 2 idle for the pair) | - |
| 4 | - | yes | the acceptance reference half |
| 5 | - | yes | only for new f32 references |
| 6 | - | yes (step 2 idle) | - |
| 7 | - | yes | - |
| 8 | step 1 | steps 2-3 | step 2, only for new f32 references |
| 9 | - | yes, idle | - |

Size: 9 tasks. Task 1 about 1-1.5 agent-days (kernel variants and the wait for both cards). Tasks 2-9 about 10-13
agent-days more, only after a "go". Task 1's build and Tasks 2-8 go to Opus; the timed runs of Tasks 1, 3, 6, 7 and 9
to a Sonnet agent.
