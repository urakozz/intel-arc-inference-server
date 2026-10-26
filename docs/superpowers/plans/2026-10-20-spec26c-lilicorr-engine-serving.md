# Spec 26c - LiLiCorr in Bolt's loop and server: verify at M <= 8 with taps on two cards, the Mamba state slots and the commit, the union-aware MoE verify, the prefill projection, snapshots with the ring, `--spec dflash` (D0, D2-D5)

> DROPPED (operator, 2026-10-27): support removed from the engine in e21208c; kept for the record.

**Status (2026-10-23): built on the Mac** (branch `spec26c-lilicorr-engine-serving`; as built: spec 26 §14) - Tasks 1-7
but Task 2 Step 3 (the replay lever: not built, §11) and Task 5's GEMM path (the 8-row replay fallback built); nothing run
on a card - box queue row **65** (Session 14). Built whatever 26a shows (the operator's ruling); `--spec dflash` becomes Bolt's
default only on D5's rule.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** speculative decoding for Bolt with LiLiCorr, greedy losslessly and sampled exactly: Bolt's decode list at M = K + 1
<= 8 rows writing the six taps of every row (the taps of blocks 1 / 18 / 35 crossing to device 1 on the copy hand-off), the
Mamba state per row in 19c's slots with a commit that moves the live slot on both cards, the union-aware MoE at M rows (each
expert block read once per verify), host acceptance (`accept_greedy` / `accept_sparse` with LiLiCorr's 8-candidate q), the commit
projecting rows 0..j into the 1024 ring and the anchor, the prefill tap staging and the projection of each chunk's last 1024
rows, the ring and anchor in every snapshot, `b70-serve <bolt> --spec dflash|lookup`, AdaptiveK over the measured cost row; D0,
D2-D5.

**Architecture:** spec 26 §5.3-§7, §10 decisions 4-8, 10. Spec 19c's loop as built (`runtime::TapDrafter`,
`build_verify_taps`, the per-row recurrent-state slots `gdn_step_slots` and `Control.gdn_live`, `Engine::verify(K)` /
`commit(j, t)`, `server::accept_sparse`, `AdaptiveK`), 19d's serving (the ring after the target's state in every snapshot,
`store_root(kv_form, drafter_key)`, `--spec dflash --pp 2`: the drafter on device 1, device 0's taps copied into device 1's tap
buffer before its hand-off, `peer` refused by name), 23f's lazily captured verify per M and its prefill tail. The Mamba
analogue of 19c's GDN slots: `nb_ssm`'s `SPEC_SLOTS` build writes the state after row m into slot `(live + m) % 8`; the conv
ring's raw xBC at `(pos + m) % 16` (disjoint for M <= 13); commit `Control.ssm_live = (live + j) % 8` on both cards. vLLM's
`3dc6def360` keeps a longer conv state indexed by the accepted count instead - its test cases (query lengths 0 / 1 / ragged,
accepted counts 1..M) are copied into D3.

**Tech Stack:** C++17, OpenCL C (ocloc), Level Zero, CMake / ctest; the server (C++); Python for the box client.

**Spec:** `docs/superpowers/specs/2026-10-20-spec26-lilicorr-design.md` (§4, §5.3-§7, §10). Precedents: plans 19c, 19d and 23f
Part 2 (Tasks 8-9); `runtime/{tap_drafter.h,capture.*}`, `runtime/dflash/`, `src/kernels/gdn_step.cl` (the slot build),
`tools/mac/clrun/gdn_slots_run.cc`, `server/{spec_accept,adaptive_k,prefix_cache}.*`, `cli/{muse_serve,muse_spec}.h`.

## Dependencies and branch points

- **Plans 25c, 25d, 25e and 26b merged;** 26a's verdict for the defaults (format, K_max, q form) and for Decisions 4 / 10's
  thresholds - until it exists, the slots (Decision 4) and the union MoE (Decision 10) are built, the replay lever is not.
- **Decision 4's lever** (the read-only verify and the replay at commit) is Task 3's optional half, built only if 26a's cost
  table shows the slot writes above 10 % of a verify at the chosen K.
- **Not here:** batching (19d's interface covers a later batched engine), tree verification.

## Global Constraints

- Branch `spec26c-lilicorr-engine` from main; signed commits; no merge, no push.
- **D0:** without `--spec dflash` / `--spec lookup` every kernel binary sha256-identical, every existing suite green, Bolt's
  decode list exactly 25c's (its names test re-run).
- **Every M-row launch runs exactly M real rows** - one captured verify list per M, never a padded M (vllm-xpu-kernels 8b95ea6).
- **Bitwise:** verify rows at M = 2..8 equal M = 1 (logits, KV rows, every slot's SSM state, conv ring columns, taps); the
  union-aware MoE equals the plain M-row MoE; the commit's projected ring rows equal 26b's projection of the same taps.
- Every number measured or marked; box rules as 25c's. No `rm -rf`.

## Review Focus

1. **The state at the last accepted row.** After a verify of M rows and a commit of j accepted drafts, the live SSM slot on both
   cards holds exactly the state after j + 1 plain steps; the rejected rows' slots and ring columns are rewritten before anything
   reads them; a mutant committing j + 1 is caught (19c's `gdn_slots_run` pattern, here `nb_slots_run` on the Mac GPU).
2. **The taps cross before the hand-off.** Device 0 writes rows' taps of blocks 1, 18, 35 into device 1's tap buffer, then hands
   off h; device 1 writes 51, 68, 85; the projection reads all six for rows 0..j only.
3. **The anchor and the ring after commit** equal what an unbroken plain run would have projected, at every K and j (0..K),
   including j = 0 and j = K.
4. **Snapshots carry the drafter.** A restore at any aligned block end gives the cold run's ring and anchor bitwise (the prefill
   projected each chunk's last 1024 rows, Decision 7), and the drafts after it are identical; a snapshot taken without a drafter
   restores a ring of zeros and the drafts are worse, never the output; the store keys differ with and without the drafter.
5. **Sampling is exact:** q is the softmax of the realized lattice scores at the request's T over the 8 candidates; a draft with
   q = 0 is a logic error; the residual lives on p's filtered support.

---

### Task 1: Bolt's decode list at M = 2..8 with taps

**Files:**
- Modify: `src/kernels/CMakeLists.txt` (every `nb_*` and reused GEMV / attention / argmax binary at M = 2..8 - additive),
  `src/kernels/bolt_kernels.h` (`decode_variants(..., M)`), `src/runtime/bolt/bolt_capture.{h,cc}` (`build_verify_taps(M)`: 25c's
  list at M rows, the ids from a host-written buffer through the embedding gather at M, every row's argmax into an output
  buffer, the six tap copies per row; captured lazily per M), `src/runtime/bolt/bolt_sizes.*` (the tap buffers, the verify
  scratch), `tests/kernels/bolt_variant_names_test.cc`, `tests/runtime/bolt_verify_test.cc` (card)

- [ ] **Step 1: the failing tests:** names (every M) are targets; `bolt_verify_test` - D3 at M = 2..8 on the synthetic (one card)
  and SergiioB's checkpoint (two cards): verify rows bitwise the M = 1 steps (logits, KV rows, taps); the launch count per M
  asserted.
- [ ] **Step 2: implement; PASS (box). Commit** `git commit -S -m "runtime: Bolt's verify list at M = 2..8 with the six taps of every row, the taps crossing on the copy hand-off (spec 26c)"`.

### Task 2: the Mamba state slots and the commit

**Files:**
- Modify: `src/kernels/bolt/nb_mamba.cl` is **not** edited after 25c's merge - the slot form is a define of the same source that
  25c already compiles with `SPEC_SLOTS 0`; if 25c did not provide the define, the slot build is a family copy `nb_mamba_slots.cl`
  (B0 holds either way: 25c's binaries are unchanged), `src/runtime/bolt/bolt_engine.{h,cc}` (`verify(drafts, k)`, `verify_ids()`,
  `read_verify_logits`, `commit(j, t)`, `max_verify_k`; `Control.ssm_live`; the slots allocated from 25b's `verify_slots` reserve:
  7 more state copies, 1.17 GB fp32 across the cards), `tools/mac/clrun/nb_slots_run.cc`, `tests/runtime/bolt_commit_test.cc` (card)

- [ ] **Step 1: the failing tests:** Review Focus 1 on the Mac GPU (`nb_slots_run`: the slot build bitwise the plain kernel; partial
  acceptance at j = 0 / 3 / 6 / 7 of 7 and 2 of 4 continues exactly as the plain run; the j + 1 mutant caught) and on the card
  (`bolt_commit_test`: the live slot against the state after j plain steps on both cards, then 64 greedy ids); `3dc6def360`'s
  cases: M of 1, a ragged sequence of M across iterations (8, 1, 5, 2), every accepted count 0..M-1.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "runtime: Bolt's per-row Mamba state slots and the commit that moves the live slot on both cards (spec 26c)"`.
- [ ] **Step 3 (only on 26a's numbers, Decision 4):** the read-only verify (`nb_ssm` walks its M rows in registers and writes no
  state) and `commit` replaying rows 0..j over the live state with the same fp32 code - bitwise Step 1's results; `B70_BOLT_VERIFY_STATE=slots|replay`; the A/B in Task 7. Commit `git commit -S -m "runtime: Bolt's replay commit - read-only verify, rows 0..j replayed (spec 26c, Decision 4)"`.

### Task 3: the union-aware MoE at M rows

**Files:**
- Create: `src/kernels/bolt/nb_moe_union.cl` (`nb_moe_up_U` / `nb_moe_down_U`: the M x 9 (row, expert) pairs sorted by expert in SLM
  - stable by row -, one work-group set per distinct expert reading its block once for all its rows; the per-row combine in
  ascending expert id with 25c's arithmetic), `tests/kernels/bolt_moe_union_test.cc` (card), the twin in `tests/kernels/bolt_ref.h`
- Modify: `src/kernels/CMakeLists.txt`, `src/runtime/bolt/bolt_capture.cc` (`B70_BOLT_VERIFY_MOE=union|plain`, union the default)

- [ ] **Step 1: the failing tests:** the union form bitwise the plain M-row form on random routes (all rows the same experts; all
  disjoint; one expert in every row; M = 8 with 72 distinct), every block read once (a counting build), never a padding row.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "kernels: Bolt's union-aware M-row MoE - each expert block read once per verify, bitwise the plain form (spec 26c)"`.

### Task 4: the loop, acceptance, AdaptiveK (D2, D4)

**Files:**
- Modify: `src/runtime/dflash/lilicorr_drafter.*` (attach to `BoltEngine` across two devices: `attach_target` / `draft_with`, 19d),
  `src/cli/bolt_serve.h` (`step_many` for greedy / sampled through `accept_greedy` / `accept_sparse` with `server::dflash_q` over the
  readback's realized scores; `step_drafts` for lookup; the deferred commit; `truncate_to`), `src/server/adaptive_k.*`
  (`MtpCost::bolt_lilicorr` - DERIVED from spec 26 §4 until 26a's table, then the measured row), `src/cli/bolt_spec.h` (`--spec
  dflash|lookup`, `--draft-model` (default the LiLiCorr snapshot), `--draft-format exact|int8|int4` (26a's default), `--spec-max
  1..7`, `--spec-cost`), `tests/server/bolt_spec_test.cc` (host, a fake engine), `tests/runtime/bolt_dflash_test.cc` (card)

- [ ] **Step 1: the failing tests:** D2 (`bolt_dflash_test`, two cards): `--spec dflash` greedy equals `--spec off` token for token
  on the golden prompts and A4, whatever K the policy picks, and for fixed K = 1..7; D4 (`bolt_spec_test`): 10^6 seeded draws match
  the filtered p within spec 8 M4's bars; no accepted draft outside the candidates; a mutant residual fails; Review Focus 3 (the
  ring and anchor after every commit equal an unbroken plain run's projection).
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "runtime: LiLiCorr drafting Bolt - the loop, greedy lossless, sampled exact, AdaptiveK (spec 26c, D2 / D4)"`.

### Task 5: the prefill taps and the projection of each chunk's last 1024 rows

**Files:**
- Modify: `src/runtime/bolt/bolt_prefill.{h,cc}` (the tap staging per chunk - 6 x 4096 bf16 a row, device 0's three sites copied to
  device 1's staging - and, before the block hook, the projection of the chunk's last 1024 rows: `fc` as a dequant slab +
  `pf_gemm` at M 1024 (the drafter's format dequantised to bf16 slabs; the exact arm's g applied in the epilogue), `hidden_norm`,
  the anchor, per-layer k / v at M 1024, `df_prep`'s k_norm / RoPE / ring store at M 1024 - or, if the GEMM path is not ready,
  26b's 8-row projection list replayed 128 times, recorded as the slower fallback), `tests/runtime/bolt_dflash_prefill_test.cc` (card)

- [ ] **Step 1: the failing tests:** after a prefill of 4096 / 5000 / 300 ids the ring and anchor equal the plain-run projection
  of the same taps bitwise (Review Focus 3 at the prompt's end); a chunk ending at a block end has its ring complete (Decision 7);
  splits at multiples of 64 bitwise; the GEMM path bitwise the replay path or within its stated bar (the GEMM's K order: cosine
  >= 0.99999 on the ring rows - then the snapshot bitwise rule holds against the GEMM path's own cold run).
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "runtime: Bolt's prefill taps and the projection of each chunk's last 1024 rows for LiLiCorr (spec 26c, Decision 7)"`.

### Task 6: snapshots and serving (Review Focus 4; D5 prepared)

**Files:**
- Modify: `src/runtime/bolt/bolt_engine.{h,cc}` (`state_bytes` / `save_state` / `load_state` append the ring and the anchor after
  the target's state: +5,242,880 B + 16,384 B), `src/server/prefix_cache.*` (no change expected: `store_root` already mixes the
  `drafter_key`), `src/cli/b70_serve.cc` (`--spec dflash|lookup` for Bolt; the startup line names the drafter, its format and
  key), `tests/runtime/bolt_dflash_prefix_test.cc` (host, mock), `tests/runtime/bolt_dflash_prefix_gpu_test.cc` (card)

- [ ] **Step 1: the failing tests:** Review Focus 4 (host mock and card): restores at aligned block ends give the cold ring /
  anchor and identical drafts; `a608b6e6e0`'s case - a sibling request diverging one token after a shared block end restores at
  that block end and drafts as the unbroken session; a drafter-less snapshot restores zeros (lower acceptance only); keys differ.
- [ ] **Step 2: implement; PASS. Commit** `git commit -S -m "server: Bolt + LiLiCorr snapshots carry the ring and the anchor; b70-serve --spec dflash|lookup for Bolt (spec 26c)"`.

### Task 7: speed (D5), docs, the queue row

**Files:**
- Modify: `docs/BENCHMARKS.md` (a "LiLiCorr on Bolt (spec 26)" section), `docs/19-running-models.md`, `README.md`, spec 26 ("26c as
  built"), this plan's status, `box-validation-queue.md` (row 65), `tools/box_validate/stages.sh` (`row 65`)

- [ ] **Step 1 (box, D5):** `b70-serve <bolt>` with llama-benchy at the operator's flags and on the agentic transcript, `--spec off` /
  `lookup` / `dflash` (auto K), greedy and T 1.0 / top-p 0.95, each format; the slots / replay and union / plain A/Bs when built;
  per request the mean accepted length. **`dflash` becomes Bolt's default only if it beats both by >= 10 % on the agentic set and
  loses nowhere by more than 3 %**; otherwise it ships as an option.
- [ ] **Step 2: the queue row 65:**

```text
row 65 "spec 26c - LiLiCorr in Bolt's loop and server: verify M <= 8 with taps on two cards, Mamba state slots and the commit, the union-aware MoE, the prefill projection, snapshots with the ring, --spec dflash (D0, D2-D5) (plan 26c)"
stage r65.k0       65 default gpu  -                         -           "D0: every binary sha256 = main without --spec; Bolt's decode names unchanged"
stage r65.d3       65 default gpu  bolt_g128,lilicorr        -           "bolt_verify_test, bolt_commit_test, bolt_moe_union_test: D3 at M = 2..8, the slots, the union MoE"
stage r65.d2       65 default gpu  bolt_g128,lilicorr        r65.d3      "bolt_dflash_test: greedy lossless on golden + A4, every K"
stage r65.prefill  65 default gpu  bolt_g128,lilicorr        r65.d3      "bolt_dflash_prefill_test: the ring / anchor after prefill, chunk ends complete"
stage r65.prefix   65 default gpu  bolt_g128,lilicorr        r65.prefill "bolt_dflash_prefix_gpu_test: restores bitwise, the junction sibling, drafter-less snapshots"
stage r65.benchy   65 optin   gpu  bolt_g128,lilicorr        r65.d2      "D5: off / lookup / dflash on llama-benchy and the agentic transcript, per format; the A/Bs"
```
- [ ] **Step 3: commit** `git commit -S -m "docs: spec 26 (26c as built), box queue row 65 - LiLiCorr serving Bolt"`.

**Gate for the plan:** Mac - every host test (D4, the snapshot layout, the cost row, `nb_slots_run`), cmdlines additions only,
syntax checks. Box - D0; D3 at M = 2..8 (rows, slots, the union MoE); D2 on golden and A4; the prefill ring and the restores;
D5 recorded and the default chosen by its rule.
