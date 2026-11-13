# Spec 19 v3d - a cheaper DFlash round on one card: verify at 8 rows, then the draft pass (open-work item 12)

**Status (2026-11-13): planned. First in the order.** Spec:
`docs/superpowers/specs/2026-11-13-spec19v3-dflash-modes-design.md` §3 (the round's cost) and §2 (what the reference
engines do). It carries spec 19 v2's open cost targets forward (its C7: draft <= 4.5 ms, verify at 8 rows <= 1.15).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min in one blocking wait.

**Goal:** on one card at depth 4096, verify at 8 rows <= **1.10** plain steps (aim 1.05; now 1.245) and the draft
pass <= **5.0 ms** (aim 4.4; now 8.43), so a round is <= 1.30 plain steps (now 1.56) and the serving bench moves
from 88.8 to ~107-113 t/s (derived). Greedy output stays judged equal to `--spec off`; the drafter's acceptance is
not lower.

**Why first:** every mode multiplies the round's cost, tensor parallel's 170-190 t/s target cannot be met without
it (spec §6.4), and the verify half also speeds MTP's verify and the slots' row step.

**Two halves, the same shape:** measure our lists by kernel family, fill a stop / continue table, then take the
levers in the order of the measured gap. A lever is kept only if its saving clears its bar; a lever that misses is
recorded with its number and removed.

**What we already have** (spec §3.1's table, read from our code): all rows in one DPAS tile with one dequant per
weight tile (`gemm_sm`), GDN rows chained in registers with a snapshot per row and accept-by-index (`gdn_step`'s
`SPEC_SLOTS`), one attention kernel for all rows plus a combine (`attn_verify_dpas` + `attn_reduce_v2`), one captured
list per row count. **What we have only partly:** glue folded into its neighbour (774 launches a step, ~12 a
layer). **What the drafter lacks:** attention on DPAS, a cheap candidate selection, fewer head and body bytes.

## Dependencies

- Branch `spec19v3d-dflash-round-cost` from `spec19v2-dflash-speed` (or main once that is merged), own worktree
  under `.claude/worktrees/`.
- Box data: the gate checkpoint, the drafter, spec 19 v2 Task 1's acceptance baseline
  (`~/b70/logs/dflash-v2/baseline/`), the opencode recordings, the f32 judge's reference sets.
- One card for every task: card 0 (card 1 is the Flash-Next quantisation's from 2026-11-14).
- Task 8 only: the tuned AutoRound int4 drafter export (open-work, "DFlash drafter quant"). Not this plan's to make.

## Global constraints

- No new user flag and no environment variable. Test and probe overrides through `CaptureOptions`.
- **Verify levers change MTP's and the slots' lists too** (they bind the same kernels). Each such lever runs those
  lists' own gates: `mtp_verify_test`, `mtp_gpu_test`, `slots_test`, and the serving judge on the golden and
  tool-call sets with `--spec mtp`.
- **Plain decode (1 row) does not move:** the 1-row list's binaries are checksum-identical after every task.
- A lever that changes no arithmetic is gated bitwise; one that changes the order of sums is gated by the f32 judge
  (target lists) or the drafter's f32-judge form (spec 28 §11) plus acceptance.
- Build and test on the box (`tools/box.sh`). GPU command form:
  `tools/box.sh run "tools/gpu_lock.sh card 0 -- <command>"`; long jobs
  `tools/box.sh run "mkdir -p ~/b70/logs/dflash-round && nohup setsid tools/gpu_lock.sh card 0 -- <command> > ~/b70/logs/dflash-round/<name>.log 2>&1 &"`.
- Mask ssh output: `sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'`.
- Timed numbers: interleaved pairs after a warm-up, median of rounds, graded "idle card" or "busy box". Per-launch
  timestamps add overhead: use them to attribute, and uninstrumented pairs to decide.

---

## Half 1 - verify at 8 rows

### Task 1: where the 0.25 goes (box)

**Files:**
- `tools/probe/probe_dflash_iter.cc`: a `--verify-profile` mode - the tap verify lists at 1, 2, 4 and 8 rows with
  per-launch events (`CaptureOptions::profile`, as `build_verify` already takes), summed by family from the lists'
  launch labels: each linear shape (`qkv||z`, `out_proj` / `o_proj`, `q||k||v`, `gate||up`, `down`), the head,
  argmax, GDN, attention prepare, attention, attention reduce, the residual fold, the norm, SiLU, the gated head,
  the embedding, the tap copies; and the wall time minus the device sum (the gaps).
- `tools/box_validate/stages.sh`: a new row (the next free one) with stages `.k0` (host checks, unchanged 1-row
  binaries), `.vprofile` (this task), and later tasks' stages.

- [ ] **Step 1:** the mode; build; run at depth 4096 and 32768:
  `tools/box.sh run "tools/gpu_lock.sh card 0 -- build/tools/probe/probe_dflash_iter <target> z-lab/Qwen3.8-27B-DFlash2 --format int8 --candidate-head --depth 4096 --verify-profile"`.
- [ ] **Step 2:** the table in BENCHMARKS "DFlash (spec 19)", block "round cost (spec 19 v3d)": per family, ms at
  1 / 2 / 4 / 8 rows and the excess of 8 rows over 1 row; the families' excess must sum to the measured 7.3 ms
  within 10 % (else the attribution is not trusted: find the missing time first).
- [ ] **Gate / the stop-continue table (objective):** a family is a lever candidate when its excess at 8 rows is
  >= **0.5 ms**. Write for each family: excess, candidate or not, and which Task 2-4 lever addresses it.
  - **Stop:** no family reaches 0.5 ms (the excess is spread thin) - record it, skip to Half 2, tell the operator
    that verify at 8 rows stays ~1.25 without a different kernel structure.
  - Commit `probe_dflash_iter: verify at 1 / 2 / 4 / 8 rows by kernel family (spec 19 v3d Task 1)`.

### Task 2: the glue at M rows (box) - only the families Task 1 names

**Files:** `src/kernels/prep.cl`, `src/kernels/gemm_sm.cl` (an epilogue only where a fold is fused into it),
`src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`, `src/runtime/capture.cc` (the M >= 2 lists' walk;
`decode_launches` stays the 1-row count), `tests/kernels/` (the fused kernel's twin test), `tests/runtime/mtp_verify_test.cc`.

Candidates, each its own commit, in Task 1's order:
- [ ] the residual fold and the norm as one launch at M >= 2;
- [ ] SiLU folded into `gate||up`'s epilogue at M >= 2;
- [ ] the tap copies folded into the fold that produced the row (5 x M device copies a verify today);
- [ ] the gated head and the GDN's small projections where Task 1 shows them.
- **Gate (each):** bitwise the unfused rows (same values, same order - fusion moves no arithmetic); verify at 8 rows
  faster by >= **0.3 ms** in interleaved pairs, or the lever is removed; MTP's verify at 4 rows not slower.

### Task 3: the GDN step at M rows (box) - only if Task 1 names GDN

**Files:** `src/kernels/gdn_step.cl` (the `SPEC_SLOTS` build), `tests/kernels/gdn_step_slots_test.cc`.

- [ ] **Step 1:** split Task 1's GDN excess with a probe build: the row chain's compute against the snapshot writes
  (8 x 151 MB a verify). Record both.
- [ ] **Step 2:** the lever the split names - the snapshot store as one streaming write per row from the registers
  (no read-back of the slot), or the chain's per-row work that is repeated and need not be. No change to a row's
  arithmetic.
- **Gate:** bitwise the current slots after every row (`gdn_step_slots_test`), accept-by-index unchanged; >= 0.3 ms
  at 8 rows or removed.

### Task 4: the linears and the head at 8 rows (box) - only the shapes Task 1 names

**Files:** `src/kernels/kernels.h` (`kSmallMTuning`), `src/kernels/CMakeLists.txt`, `tests/kernels/gemm_sm_test.cc`.

- [ ] For every shape whose `gemm_sm` time at 8 rows exceeds `gemv`'s at 1 row by >= 0.3 ms: sweep its tuning row
  (the split-K form, 256 registers, the 2D block read) with `tools/probe/probe_small_m`, keep the best.
- [ ] The head: argmax and the logits rows at 8 rows, if named.
- **Gate:** property R (a row's output does not depend on its neighbours) and the fp64 check of `gemm_sm_test` at
  every built M; the f32 judge on the golden and tool-call sets; the saving at its bar.
- **Operator ruling (2026-11-13):** a lever that is bitwise the form it replaces stays when it is measurably faster,
  under the bar too. Applied: gate||up's 2D block read (-0.18 ms at 8 rows, -0.17 at 4; bitwise its plain-read twin
  at every M, `gemm_sm_test`).

### Task 5: the verify half's record (box)

- [ ] `probe_dflash_iter` uninstrumented at 4096 and 32768; `probe_mtp_steps` for MTP's verify at 2..4 rows;
  `probe_slots_speed` for the row step. Refresh `MtpCost::dflash` and MTP's tables (`src/server/adaptive_k.cc`,
  `tests/server/adaptive_k_test.cc`) from the medians.
- [ ] `tools/box_validate.sh --quick` on the branch head.
- **Gate:** verify at 8 rows <= **1.10** plain steps at depth 4096 (aim 1.05), recorded with what each kept lever
  gave. Missed: record the remaining excess by family; Half 2 proceeds either way.

## Half 2 - the draft pass

### Task 6: the drafter's attention on DPAS (box)

Today `df_attn_part` (`src/kernels/dflash/df_attn_rows.cl`, the H1 / B4 geometry) is a scalar fp32 walk with its own
`exp` over a bf16 ring: 2.08 ms for 5 layers. The target's `attn_verify_dpas` does 8 rows over 2048 keys in
~80-110 us a layer. The references' split attention costs 38 us a layer (spec §2.4).

**Files:** new `src/kernels/dflash/df_attn_dpas.cl` (from `attn_verify_dpas.cl`'s idioms: the query rows as the DPAS
A operand staged once, K and V by 2D block reads from the ring, the online softmax per row, the slices merged in
order; the drafter's own shape - 32 query / 8 KV heads of 128, the ring's wrap, the window mask, the block's own
keys in the reduce), `src/kernels/CMakeLists.txt`, `src/runtime/dflash/dflash_capture.cc` (bind it; the scalar form
stays reachable through `CaptureOptions` for the A/B), `tests/kernels/dflash_kernels_test.cc`,
`tests/kernels/dflash_ref.h`, `tests/runtime/dflash_draft_test.cc`.

- [ ] **Step 1:** the kernel over the **bf16 ring as it is** (no format change: snapshots and the prefix cache key
  stay), with q and P carried as two bf16 halves as the target's kernel does. Twin test against the scalar form:
  cosine >= 0.99999 per (head, row) on random and on real ring contents; a row's output independent of its
  neighbours.
- [ ] **Step 2 (box):** the profile and the acceptance: `tools/box_validate/dflash_v2.sh gpu` then `ref` against
  the saved baseline.
  - **Gates:** the 5 attention launches <= **0.9 ms** in sum (a saving >= 1.2 ms; aim 0.5 ms); all 35 corpus / K
    acceptance rows within 0.05 tokens a round of the baseline; the drafter's f32-judge form passes; greedy
    identity on golden + tool calls.
  - Acceptance lost: the rounding is the suspect (the target's kernel needed the two-halves P for the same reason,
    `attn_verify_dpas.cl`'s header); fix with a failing test, do not widen the bar.
- [ ] **Step 3 (only if Step 2 leaves > 0.6 ms):** the ring in int8 by the target's rotation scheme (`kv8`). It
  changes `ring_bytes()` and the drafter's key, so old prefix cache entries are not restored beside it (the key
  already guards that). Same gates.

### Task 7: the draft head's path (box)

Today 1.79 ms: the int4 candidate GEMM 1.21 (0.64 GB, at bandwidth), two top-64 stages 0.44, the exact re-rank 0.14.

**Files:** `src/kernels/dflash/df_candidate.cl`, `df_topk.cl`, `df_rerank.cl`, `src/kernels/gemm_sm.cl` (an epilogue
form), `src/loader/dflash_loader.cc` and `lm_head_int8.*` (a 2-bit copy of the head made at load),
`src/runtime/dflash/dflash_capture.cc`, `dflash_sizes.{h,cc}`, `tests/kernels/dflash_head_kernels_test.cc`.

- [ ] **Step 1: selection.** The candidate GEMM's epilogue keeps the best 8 of every 64-column block per row
  (per-block maxima), and one small top-k picks 64 (or 80) of those; the two full-vocabulary top-64 stages go.
  - **Gate:** the same 16 final candidates as today on the baseline's 4256 draft records, or acceptance within 0.05
    on all 35 rows; selection <= 0.2 ms (a saving >= 0.25 ms).
- [ ] **Step 2: bytes.** A 2-bit candidate head (plain round-to-nearest per 128-column group, made at load, 0.32 GB)
  with the top 64 re-ranked exactly; 80 if 64 loses acceptance.
  - **Gate:** acceptance within 0.05 on all 35 rows (the int4 form's bar was a gap of zero: record the gap); the
    candidate GEMM <= 0.7 ms (a saving >= 0.4 ms); `--max-len auto` with the drafter rises by ~0.3 GB of context
    (printed).
  - Missed on acceptance at 80: keep int4, record.

### Task 8: the body's bytes (box) - waits for the export

- [ ] When the AutoRound int4 drafter export exists and its loader has landed (the open-work follow-up), re-run the
  profile and the acceptance with it on this branch's kernels.
  - **Gate:** acceptance within 0.05 on all 35 rows; body GEMMs <= 2.1 ms (1.91 measured with the at-load int4
    form).
  - Until then this task is open and the plan's goal is judged with the int8 body (aim 5.0 ms, not 4.4).

### Task 9: the round's record (one idle card)

- [ ] **Step 1:** `probe_dflash_iter` at 4096 and 32768, uninstrumented; refresh `MtpCost::dflash` from the medians.
- [ ] **Step 2:** the serving bench (3,425-token prompt, 512 tokens) and the opencode replay, three arms rotated
  over three rounds (`tools/prefix/replay_spec.py ... --card 0`), judged. Check the card is idle and more than 16
  CPU threads are free; write the conditions.
- [ ] **Gate (the goal):** verify at 8 rows <= 1.10 plain steps; draft pass <= 5.0 ms; round <= 1.30 plain steps;
  the replay's DFlash / MTP ratio not below 1.193 x 0.97 against the **new** MTP (MTP got faster too); the f32
  judge at 0 failures and 0 skipped.
- [ ] **Step 3: close.** BENCHMARKS, spec 19 v2's C7 table ("as built"), this spec's §3.3, open-work item 12.

**Which tasks need the box:**

| task | one card | idle card | CPU model job |
|---|---|---|---|
| 1 | yes | preferred | - |
| 2-4 | yes | for the kept levers' pairs | only for new f32 references |
| 5 | yes | yes | - |
| 6-7 | yes | for the pairs | the acceptance reference half (`dflash_v2.sh ref`, `/tmp/b70-cpu-model.lock`) |
| 8 | yes | - | the same |
| 9 | - | yes | - |

Size: 9 tasks. Half 1 about 4-5 agent-days (Task 1 half a day; it may end Half 1 early). Half 2 about 4-6
agent-days without Task 8. Kernel work (Tasks 2-4, 6-7) goes to Opus; the timed runs (Tasks 1, 5, 9) to a Sonnet
agent.
