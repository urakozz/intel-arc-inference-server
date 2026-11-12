# Spec 19 v2 - make DFlash beat `--mtp auto` on Qwen3.8 (open-work item 12)

**Status (2026-11-12): IN PROGRESS, operator resumed.** Tasks 1-2 complete: calibrated costs, G0, GPU profiling,
CPU G-accept and 39/39 serving correctness PASS. The **80.5 t/s early bar PASSED** at 88.806 t/s. Continue Tasks
3-7 and 7b. Operator update: Task 8 may use one idle card while the other is busy, provided more than 16 of
44 CPU threads are available. Label those measurements "busy box". Compare against main's per-step MTP (`f65ae50e`
or later), enabled by default for `--mtp auto`. Further GPU validation uses an available, idle card; card 0 is busy
with prefill work. Spec: `docs/superpowers/specs/2026-11-11-spec19v2-dflash-v2-design.md` (C1-C7, the
gates in §3). Evidence: `docs/probes/dflash-vs-sly-2026-11-11.md`. Box row **82** (new, Task 1 adds it).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min.
> Keep the implementation turn active while a detached job runs: use completion notifications and continue
> independent work, then resume from the result. Answer status questions in commentary; a detached job continuing
> after a final response does not mean implementation is continuing. Stop when the operator pauses or input is required.

**Goal:** an iteration at K = 7 of <= 1.35 plain steps, and `--spec dflash` >= 1.10x `--mtp auto` on the opencode
recording with no loss > 3 % elsewhere. Greedy output stays identical to `--spec off` under the f32 judge.

**Order and why:** measure first (Tasks 1-2). The first measurement of DFlash since specs 28b-28d may already change
the verdict, and an acceptance defect would make every speed task's payoff smaller. Then the cheap host-side fixes
(Tasks 3-4), then the bytes (Tasks 5-6), then the small kernels the profile names (Task 7), then the bar (Task 8).

**Stop rule:** if Task 2's re-run already meets G-speed, the operator decides whether Tasks 3-7 run. If Task 1's
acceptance check fails G-accept, Tasks 3+ wait for its root cause.

## Dependencies

- Main at or after c9c311a9. Branch `spec19v2-dflash-speed`, own worktree.
- Box data:
  - the gate checkpoint (`qwen`);
  - the drafter (`dflash`);
  - the opencode recordings `~/b70/logs/opencode-session-int8` (78 requests), `opencode-session` (21),
    `opencode-session-2` (4);
  - 19a's recorded continuations (`oracle_dflash_dumps`) for the reference half.

## Global constraints

- No new user flag and no environment variable (zero-config). Test and probe overrides go through `CaptureOptions`,
  as `force_linear` / `force_attn` do.
- **G-D0 at every merge:** no non-DFlash binary moves (`kernel_cmdlines` shows only `df_*` / drafter variants).
- Timed numbers come from interleaved pairs after a warm-up, the median of rounds. Each one is labelled "quiet box" or
  "busy box".

---

### Task 1: measure the iteration and the acceptance on today's main (Mac, then box)

**Files:**
- Modify `src/runtime/dflash/dflash_capture.{h,cc}`: per-launch profile events on the draft and projection lists,
  through `CaptureOptions` as `build_verify` takes `ProfileEvents`.
- Create `tools/probe/probe_dflash_iter.cc`:
  - at depths 4096 and 32768: draft, verify(1..8) through the production rule (`gemm_sm` + `verify_attn_kind`), the
    projection, and one host submit + fence;
  - interleaved pairs against the plain step;
  - the draft list's profiled per-family ms;
  - an `--engine-taps` mode that dumps the engine's taps over a recorded continuation (as 19a's dumps are laid out).
- Modify `tests/server/dflash_gpu_test.cc`: also print ids an iteration with generation stopped at EOS (the current
  256-id line stays).
- Modify `tools/box_validate/stages.sh`: row 82 and its stages `r82.k0` (G0), `r82.iter` (the probe), `r82.accept`
  (engine taps -> `tools/oracle/dflash_d1.py`'s reference drafting -> E_K per corpus), `r82.d5` (Task 2).

- [x] **Step 1: the probe, the profile option, the EOS line.** Built on the box; 7 selected host/oracle tests pass.
  `tools/mac_check.sh` is absent in this checkout; no C++ was compiled on the Mac.
  Commit `probe_dflash_iter: draft / verify(1..8) / projection / host round trip, the draft list per launch (spec 19 v2 Task 1)`.
- [x] **Step 2 (box): `r82.iter` at 4096 and 32768.** Calibrated plain replay passes both depths. At 4096:
  draft 8.978 ms, verify(8) 1.245386 plain, iteration 1.579196 plain; C7 targets not met. Per-launch host-gap
  residual is an upper bound, with the limitation recorded. Table and grades in BENCHMARKS's "v2 baseline".
  Record the table in BENCHMARKS "DFlash (spec 19)" as a new
  "v2 baseline" block. **Gate (objective, recorded):** every C7 row has a measured value with its grade; verify(8) shows
  whether 28c's DPAS attention brought it to ~1.1.
- [x] **Step 3 (box + CPU): `r82.accept`.** G-accept PASS: A4 E7 7.080252 on both sides (544 anchors);
  code engine/reference 5.875/5.750, gap 0.125 <= 0.15 (16 anchors). All K=1..7 and five corpora recorded.
  The engine's taps over the A4 / golden recorded continuations and one
  held-out opencode slice, drafted by the CPU reference at K = 1..7, against the engine's in-loop ids an iteration
  (EOS-stopped).
  - **Gate G-accept:** engine-taps E_7 within 0.15 of the reference on A4 and code.
  - FAIL -> root-cause (tap layer, ring position, RoPE position, anchor row) with a failing test, before Task 3.

### Task 2: DFlash against today's recorded baselines (box; operator 2026-11-12)

- [x] **Step 1: `r82.d5`.** DFlash 88.806 t/s; golden/A4 39/39 f32-judge PASS (37 identical).
  Saved MTP/DFlash prompts match exactly at 3,425 ids; the nominal 2,820 label was stale.
  Do not re-run `--spec off` or `--mtp auto` for performance. Reuse today's main:
  - one card, MTP auto, slots_bench shape (2,820-token prompt, 512 ids, prefix cache hits): **73.2 t/s** (README);
  - plain decode at depth 4096: **33.5 t/s** (BENCHMARKS).
  - Run only `--spec dflash` through `tools/box_validate/slots_serve_bench.sh ... --spec dflash -- --counts 1`,
    same prompt, output length and prefix-cache shape. One sequence slot (DFlash does not support slots).
  - One golden + A4 greedy correctness round, using the existing references; no repeated plain/MTP serving arms.
  - Skip llama-benchy and rotated rounds here. The opencode replay with all three arms belongs to Tasks 7b/8.
- [x] **Early bar PASS:** 88.806 > 80.5 t/s; all baseline gates cleared, including CPU G-accept. Operator chose to continue
  Tasks 3-7 (2026-11-12), then 7b and Task 8 against a freshly measured, updated MTP arm. If DFlash exceeds
  **80.5 t/s** (73.2 x 1.10, rounded), pause for the operator's decision about
  Tasks 3-7. This shape is an early screen, not a substitute for Task 8's final G-speed gate.
- [x] Record the result and the provenance of the reused baselines in BENCHMARKS.

### Task 3: AdaptiveK on measured costs (Mac)

**Files:** `src/server/adaptive_k.{h,cc}` (`MtpCost::dflash`: verify rows M = 5..8 and the draft row from Task 1;
the bf16-head arm as measured or marked derived), `tests/server/adaptive_k_test.cc`.

- [x] **Step 1:** RED on the old verify table, GREEN on the measured medians. K=4 / 7 / 7 at alpha 0.5 / 0.8 /
  0.95. Box host tests (AdaptiveK, DFlash host, per-step MTP) 3/3 PASS; no MTP table changes. BF16 arms marked
  derived in code and BENCHMARKS. Test first (the K the policy picks at alpha 0.5 / 0.8 / 0.95 under the new table), implement, green.
  **Gate:** the table's entries equal Task 1's recorded medians (a comment cites the BENCHMARKS block); D2 unaffected
  (host-only). Commit.

### Task 4: one submit an iteration (Mac, then box) - C4

**Files:** `src/runtime/dflash/dflash_drafter.{h,cc}` and `runtime::Engine`'s verify / commit path (the projection
moved to the front of the next submit; one `zeCommandQueueExecuteCommandLists` over [projection, draft, verify]),
`src/cli/serve_adapters.h` (`step_dflash`, `flush_commit`, `truncate_to`), `src/cli/pp_*` if `--parallel pp`'s DFlash
path shares the call (keep pp bitwise its one-card form), `tests/runtime/dflash_engine_test.cc` (the ring after a
deferred projection equals today's at every flush point: plain step, request end, snapshot, truncate).

- [x] **Step 1:** projection deferral/batching implemented. Regression first failed on the missing Engine API;
  box host/syntax checks and the real GPU ring/snapshot/truncate/request-end checks pass. Ruling: use the actual
  queue and device buffers instead of adding a single-use host queue mock to the concrete Engine.
- [ ] **Step 2 (box):** `dflash_engine_test`, `dflash_gpu_test`, `dflash_prefix_gpu_test`, `pp_dflash_test`,
  `r82.iter`.
  - **Gates:** G-D2; the ring and snapshots bitwise today's; host gap (iteration wall minus the lists' device time)
    <= 0.5 ms at K = 7.
  - Commit the numbers.
  - One-card checks PASS (four actual passes, no skips); host gap 0.422639 / 0.381236 ms at 4096 / 32768,
    below 0.5 ms. Combined projection + host still 0.804514 / 0.766757 ms; iteration 1.577007 / 1.558300 plain
    steps. Full table in BENCHMARKS. PP regression pending both cards; Step 2 remains open until it passes.

### Task 5: the int4 g64 drafter (Mac, then box) - C2

**Files:**
- `src/loader/dflash_loader*` / `dflash_layout.h`: int4 g64 symmetric RTN at load into the target's int4 layout; the
  bytes formula.
- `src/runtime/dflash/dflash_capture.cc`: the drafter's linears and the context k||v on the target's int4 `gemm_sm`
  rows at M = 8; new `kSmallMTuning` rows only if a drafter shape lacks one.
- `src/kernels/kernel_cmdlines` (drafter variants only).
- `tests/runtime/dflash_draft_test.cc` (the f32-judge form of spec 28 §11 for the int4 arm).
- `runtime::DrafterPlan` (memory: ~0.8 GB more for context).
- int8 stays reachable through `CaptureOptions` / the loader's test entry for the A/B.

- [ ] **Step 1 (Mac):** loader + capture + host twin tests, `mac_check` green. Commit.
  - Source checkpoints `70d66833` / `a8c213b7`: RTN int4 layout-1 loader/upload, distinct prefix key, all eight
    drafter shapes on M=8 `gemm_sm`, forced-GEMV twin and f32 judge. Four host checks, GPU/probe syntax and
    app/kernel build PASS on the box. Device precision/cost gates are running; serving default remains int8.
    The existing byte formula reports 843,439,104 B saved. Nine acceptance-comparator unit tests PASS; it reuses
    the original int8 baseline without loading a CPU model and requires matching anchors and all K/corpora.
- [ ] **Step 2 (box):** `dflash_draft_test`, `r82.accept` with the int4 arm, `r82.iter`.
  - **Gates:**
    - G-drafter: E_K within 0.05 of int8 on A4, code, prose and the held-out opencode slice;
    - draft pass >= 2.5 ms faster than Task 1's;
    - G-D2;
    - `--max-len auto` with the drafter rises (printed).
  - FAIL on acceptance -> record and ask the operator for the AutoRound-exported drafter (spec v2 C2) instead of RTN.

### Task 6: the draft head (Mac + CPU, then box) - C3

- [ ] **Step 1 (CPU):** rank a 32k draft vocabulary from the opencode recordings minus a held-out slice
  (`tools/draft_vocab/rank.py`). Score it out of sample with `r82.accept`'s reference: A4, golden, the held-out slice.
- [ ] **Step 2 (Mac):** the drafter defaults to that ranked vocabulary. The ranking file ships in-tree and is loaded
  by `DraftVocabSpec::for_drafter`. `--draft-vocab` keeps its MTP meaning. Commit.
- [ ] **Step 3 (box):** `r82.accept`, `r82.iter`.
  - **Gates:** E_K within 0.05 of the full head out of sample; the draft pass's head share down to <= 0.5 ms; G-D2.
  - FAIL on acceptance -> Step 4.
- [ ] **Step 4 (only on Step 3's FAIL):** the int4 g64 candidate head (made at load), top-64, exact re-rank against
  the int8 head rows, top-16 to the selector. A new small kernel `df_rerank`, with its twin test. Same gates.

### Task 7: the draft list's small kernels (Mac, then box) - C5

- [ ] **Step 1:** from Task 1's (and Tasks 5-6's) per-launch profile, the top non-GEMM families in order:
  - fuse `df_norm` into the preceding fold;
  - fuse the conv-prepare into the norm;
  - widen `df_norm`'s grid;
  - fuse top-k stage 1 into the head's epilogue if the profile names it.
  - Each one is a commit with `dflash_kernels_test` / `dflash_draft_test` green (the f32 judge for order changes).
- [ ] **Step 2 (box):** `r82.iter`. **Gate:** draft pass <= 4.5 ms (0.15 plain steps) at depth 4096 with Tasks 5-6;
  iteration at K = 7 <= 1.35 plain steps; launches per draft printed.

### Task 7b: refresh before the bar (Mac, then box) - review 2026-11-12

- [x] **Step 1 (harness):** `tools/prefix/replay_spec.py` uses three rotated speculation arms, prefix cache on,
  whole-box lock or explicit `--card N` and one-card serving, default per-step MTP, controlled child shutdown,
  server-clock metrics and f32 judging. Stand-in tests cover both lock modes; no real replay run. Implemented independently while Task 4's
  frozen C++ build waits for GPU validation. Adapt `tools/prefix/replay_ab.sh` (or a sibling script) for the bar: its arms today toggle
  the prefix cache with `--spec off` fixed; Task 8 needs `--spec off` / `--mtp auto` / `--spec dflash` with the prefix
  cache ON in every arm. Take the cards through `tools/gpu_lock.sh` (never set `ZE_AFFINITY_MASK` by hand) and stop the
  server the way `tools/box_validate/serve_run.sh` does (its own child, wait for exit), not a bare `kill`.
- [ ] **Step 2 (cost table):** re-run Task 1's probe after Tasks 4-7 and refresh AdaptiveK's DFlash cost table from those
  medians; **Gate:** the table equals the new medians, so Task 8 does not run on Task 3's pre-optimisation costs.
- [ ] **Step 3 (combined acceptance):** measure E_K of the final drafter (int4 + the chosen draft head) against Task 1's
  **original int8 / full-head baseline** on A4, code, prose and the held-out opencode slice. **Gate:** within 0.05 of that
  baseline in total - the per-task gates of Tasks 5 and 6 must not add up to an unnoticed loss.

### Task 8: the bar (one idle GPU) - G-speed

- [ ] **Step 1:** Task 7b's three-arm replay harness on the branch head, one idle GPU, both other recordings as
  checks. Operator permits the other card to be busy if more than 16 of 44 CPU threads are available; verify
  that before the run and record the resource conditions with the busy-box grade. Use `--card N` for that run;
  omitting it reserves the whole box for a quiet-box grade. The MTP arm includes main `f65ae50e` or later and uses the
  default per-step policy for `--mtp auto` (no disabling `B70_MTP_STEP_K`). Task 2's slim `r82.d5` screen is not
  this final replay bar. All final DFlash speed numbers require an idle card.
- [ ] **Gate (spec 19 D5, unchanged):**
  - `--spec dflash` >= 1.10x `--mtp auto` on `opencode-session-int8`;
  - no loss > 3 % on A4, golden, llama-benchy tg128 / tg128 @ d32768 and pp4096;
  - G-D2 on every arm.
- [ ] **Step 2:** if passed, propose to the operator `dflash` as the one-card Qwen3.8 default when the drafter is
  present (cli's `--spec` resolution; one line in the startup print). Then update:
  - README / `docs/18-running-models.md`;
  - BENCHMARKS "DFlash (spec 19)";
  - spec v2 "as built";
  - open-work item 12 -> Closed recently.
  - If not passed: record the measured decomposition and what is left, item 12 stays open with the numbers.

**Which tasks need the box:**

| task | Mac | box | CPU |
|---|---|---|---|
| 1 | step 1 | steps 2-3 | step 3's reference half |
| 2 | - | yes | - |
| 3 | yes | - | - |
| 4 | step 1 | step 2 | - |
| 5 | step 1 | step 2 | - |
| 6 | steps 2, 4 | step 3 | step 1 |
| 7 | step 1 | step 2 | - |
| 8 | - | yes (one idle card; >16 free CPU threads if the other card is busy) | - |

Test and benchmark runs (Tasks 1-2, 8 measurements) go to a Sonnet agent; implementation (Tasks 3-7) to Opus.
