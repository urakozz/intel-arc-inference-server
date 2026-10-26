# Spec 19f - DFlash on the card (the `dflash_kernels_test` fix, rows 54-56), then prompt lookup inside DFlash

**Conditions:** this plan starts after **plan 28b** (spec 28's build) is merged, when the operator says go (one
implementing plan at a time). Task 1 (the fix) does not need 28b's code and may be pulled forward by the operator. **2026-10-27: pulled forward by the
operator - Task 1 (the fix, rows 54-56) runs early, before spec 28; Tasks 2+ still wait for 28b.**

**Status (2026-10-27): PLANNED - second in the Qwen speed track.** Nothing built. The fix re-runs the existing rows
**54-56**; the lookup-in-DFlash mode is box queue row **70** (spec 28's header lists the track's rows 68-75). K up to 15
is not here: plan 19g, after spec 28 (spec 19 §13.2).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** (1) `dflash_kernels_test` passes on the card and rows 54, 55 and 56 pass in order (D1-D5 of spec 19 §6, D3 as
amended in §13); (2) `--spec dflash-lookup`: prompt lookup inside DFlash by the published policy (spec 19 §13.1),
measured against plain `--spec dflash` on an edit / copy set and on A4 / golden, and the bar applied.

**Architecture:** spec 19 §5, §9, §11-§13. The fix lives where the failing check points (`src/kernels/dflash/`,
`runtime/dflash/`, `tests/kernels/dflash_ref.h` - Task 1 finds out). The mode is host-side: a policy object
`server::LookupInDFlash` (ENTER >= 8 tokens beyond the 2048 window, STAY >= 3 accepted, EXIT) in front of
`EngineAdapter::step_dflash`; while in lookup it writes the matcher's drafts into `cur_token[1..K]` (as 19e's
`step_drafts` does) and runs the **tap** verify list (`Engine::verify(K)` over `build_verify_taps`), skips the draft
list, and keeps the commit-side projection. `server::PromptLookup` gains a "source before pos - window" filter.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest; Python 3 for the edit / copy set's builder.

**Spec:** `docs/superpowers/specs/2026-10-05-spec19-draft-model-speculative-decoding-design.md` (§6, §9, §11-§13).
Precedents: plans 19b-19e (their tests and rows), `tools/mac/clrun/dflash_run` (the Mac's indicative run of the draft
lists), `lookup_server_test` (mock responses byte-identical to plain).

## Dependencies and branch points

- Main with 28b merged (verify lists on `gemm_sm`). Branch **`spec19f-dflash-card-lookup`**.
- **Box:** the drafter (`z-lab/Qwen3.8-27B-DFlash2`, data key `dflash` - the operator downloads; give the exact `hf
  download` line if it is missing), 19a's tap dumps pushed (`tools/box_validate.sh --push-data`, data key
  `oracle_dflash_dumps`), the gate checkpoint (`qwen`), the 2026-10-23 state `4c491f876139-vs-b32aaafbc7e2` for its
  `r54.kernels` log.

## Global Constraints

- Signed commits after every step; no merge, no push; no `rm -rf`. Box tree automatic; `tools/box.env` never committed.
- **D0:** nothing changes without `--spec dflash` / `--spec dflash-lookup`; kernel_cmdlines shows only what the fix
  needs (a changed `df_*` line is allowed and listed - no non-DFlash line moves).
- No new CLI flag beyond the one `--spec` value; the policy constants (8, 3, the window) are code constants.
- Timed stages: idle box, device 0, interleaved / rotated rounds after a warm-up, median of 3.

## Review Focus

1. **Debug before fixing:** read the failing lines, reproduce on the smallest case, name the root cause in the commit -
   no fix by loosening a bar (spec 19 §6's bars and `r54.kernels`' "within 1 bf16 ulp" stay unless the operator rules).
2. **Rows in order:** 54 before 55 before 56; a FAIL stops the chain and is reported, not worked around.
3. **The mode never changes the output:** D2 (greedy equal to `--spec off` under the tie rule) and D4 on the new mode;
   the mock test proves responses byte-identical to plain across EOS / stop / max_tokens / tool calls inside a
   lookup run.
4. **The drafter stays warm:** in lookup the projection of every committed row still runs (the ring equals the one plain
   DFlash would hold at the exit) - a host test on the mock, an engine test on the card.

---

### Task 1: the `dflash_kernels_test` fix (box + Mac)

- [ ] **Step 1: the failure, today.** Read `r54.kernels`' log in the 2026-10-23 state (`jgrab kernels` lines: which
  `df_*` check failed, by how much, at which p / K); then `tools/box_validate.sh --only r54.k0,r54.host,r54.kernels`
  on today's main to confirm it still fails (the tree has moved since 4c491f8). Record the lines in the handoff.
- [ ] **Step 2: reproduce small** - the failing kernel at the smallest shape and position that fails (the box), and
  `tools/mac/clrun/dflash_run` at the fixture's shapes (the Mac, indicative) to see whether the host twin or the kernel
  is wrong. Root cause stated.
- [ ] **Step 3: fix + a regression check** that fails before and passes after (a `dflash_kernels_test` case, or a
  `dflash_ref_test` case if the twin was wrong); `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "dflash: <root cause> - dflash_kernels_test (spec 19f)"`.

### Task 2: rows 54, 55, 56 on the card (box)

- [ ] **Step 1:** `tools/box_validate.sh --only r54` (kernels, load, draft, d1, d1_ref) - D1's three tables; if D1's
  scores bar fails while sets and walks hold, the lever is plan 19b's (fp32-input GEMV variants), recorded for the
  operator, not taken silently.
- [ ] **Step 2:** `--only r55` (the engine loop: L, T1, P, D2, D4) with D3 per spec 19 §13 (spec 28 N4 + property R).
- [ ] **Step 3:** `--only r56` (serving: the ring in snapshots, `dflash_prefix_gpu_test`, D5's `r56.d5_a4` /
  `r56.d5_benchy` arms; `r56.d5_opencode` SKIPs without the recording), with the measured cost table from 28b. Update
  spec 19 §6 / BENCHMARKS "DFlash (spec 19)" with the measured rows and D5's verdict (`dflash` the Qwen3.8 default only
  on D5's bar). Commit `git commit -S -m "docs: DFlash on the card - rows 54-56 results (spec 19f)"`.

### Task 3: lookup inside DFlash (Mac)

**Files:**
- Create: `src/server/lookup_in_dflash.h` / `.cc` (the policy: state, ENTER / STAY / EXIT, the full-depth K capped by
  the match), `tests/server/lookup_in_dflash_test.cc` (transitions on scripted acceptances; the window filter; K caps)
- Modify: `src/server/prompt_lookup.h` (matches whose source is before `pos - window` only, in this mode),
  `src/cli/serve_adapters.h` (`EngineAdapter::step_dflash`: the lookup branch - drafts into `cur_token[1..K]`, the tap
  verify, no draft list, the projection after commit), the `--spec` parser (`dflash-lookup`; refusals as `dflash`'s),
  `tests/server/lookup_server_test.cc` or a sibling (the mock: responses byte-identical to plain; the mock drafter's ring
  equal to plain DFlash's after an exit)

- [ ] **Step 1: failing tests, implement, PASS;** `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "server: --spec dflash-lookup - prompt lookup inside DFlash (ENTER >= 8 beyond the window, STAY >= 3) (spec 19f)"`.

### Task 4: the edit / copy set (Mac)

**Files:** Create `tools/spec/edit_copy_set.py` (+ `test_edit_copy_set.py`) and its output under
`tests/golden/edit-copy/` (ids and chat JSON, as the A4 set is stored): 12 prompts from this repository's own files at
depths 8k / 32k / 64k / 99k (3 each: re-emit a file after a small edit, rename across a function, quote a tool output
back), each with the expected far-copy span recorded (where the copied text's source lies relative to the window).

- [ ] **Step 1: implement, test, generate;** commit `git commit -S -m "spec: the edit / copy prompt set for lookup-in-DFlash (spec 19f)"`.

### Task 5: the measurement and the bar (box, `r70.*`)

- [ ] **Step 1:** D2 and D4 on `--spec dflash-lookup` (golden + A4 + the edit / copy set, greedy against `--spec off`,
  the tie census; the seeded sampled run bitwise reproducible); the engine test of Review Focus 4.
- [ ] **Step 2:** speed through `b70-serve`, rotated rounds, median of 3, idle box: `--spec off`, `--mtp auto`,
  `--spec dflash`, `--spec dflash-lookup` on the edit / copy set (per depth) and on A4 / golden; the time in lookup and
  the acceptance per mode recorded. Apply spec 19 §13.1's bar: **>= +20 % on the edit / copy set (median) and <= 3 %
  loss on A4 / golden -> replaces plain `dflash`**; +5..+20 % -> opt-in, the operator's call; < +5 % -> dropped.
  Record in spec 19 §13.1 ("measured") and BENCHMARKS. Commit
  `git commit -S -m "docs: lookup inside DFlash measured - <replaces|opt-in|dropped> (spec 19f)"`.

### Task 6: the box queue row

- [ ] **Step 1:** row **70** and its block in `tools/box_validate/stages.sh` (`test_box_validate.py` passes); the
  re-runs of 54-56 use their existing stages:

```text
row 70 "spec 19f - prompt lookup inside DFlash (--spec dflash-lookup: ENTER >= 8 tokens beyond the drafter's 2048 window, STAY >= 3 accepted, full-depth lookup verify, the drafter's projection kept): D2 / D4 on the mode, speed against plain --spec dflash on the edit / copy set (8k-99k) and A4 / golden (plan 19f)"
stage r70.k0     70 default cpu  -                 g0.sha,g0.bitwise,g0.suite "D0: no kernel binary moved by the mode (it is host-side)"
stage r70.host   70 default cpu  -                 -                     "lookup_in_dflash_test, the mock serving tests, test_edit_copy_set.py"
stage r70.d2     70 default card qwen,dflash       r56.d5_a4,r70.k0      "D2 / D4 on --spec dflash-lookup: golden + A4 + the edit / copy set vs --spec off under the tie rule; the warm-ring engine test"
stage r70.speed  70 optin   gpu  qwen,dflash       r70.d2                "b70-serve: off / --mtp auto / dflash / dflash-lookup on the edit / copy set per depth and on A4 / golden; rotated rounds, median of 3, idle"
stage r70.verdict 70 manual -    -                 r70.speed             "spec 19 §13.1's bar: replaces / opt-in / dropped"
rownote 70 "Needs rows 54-56 passed (plan 19f Tasks 1-2) and the drafter data key. Hand back: the D2 census, the speed table per corpus and depth, the time in lookup, the verdict line."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 70 - lookup inside DFlash (spec 19f)"`.

**Gate for the plan:** `dflash_kernels_test` and rows 54-56 pass on the card (D1-D5, D3 as amended); the mode's host
tests green and `tools/mac_check.sh --base main` PASS; row 70's D2 / D4 pass and the speed table and verdict are
recorded. Hand back with the fix's root cause, the rows' results and the verdict line.
