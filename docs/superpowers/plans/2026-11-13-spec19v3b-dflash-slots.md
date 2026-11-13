# Spec 19 v3b - `--spec dflash` with several request slots on a card (`--max-num-seqs`) (open-work item 12)

**Status (2026-11-13): planned. Last in the order** (after plan 19v3d, the round's cost, and plan 19v3c, tensor
parallel), and only if its Task 1 holds. Spec:
`docs/superpowers/specs/2026-11-13-spec19v3-dflash-modes-design.md` §5 (mode B). It needs the int8 KV cache with a
drafter (plan 19v3a, in progress in another session): four slots keep a usable slot 0 only in int8.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh`, long jobs detached, checked at most every 30 min in one blocking wait.

**Goal:** `b70-serve --spec dflash --max-num-seqs N` (N up to 4 on a card, up to 8 on two with a model copy per card)
serves. A request that runs alone is bitwise the one-slot DFlash server (1.19x MTP on the opencode replay); with
several requests active the total is not below 0.95x MTP's at two and 0.90x at four; every request's output is judged
equal to `--spec off` in the same mode.

**Design (spec §5.2):**
- per slot its own drafter session (ring, taps, control block, readback) and its own copy of the draft and
  projection lists; weights and scratch shared;
- slot 0 holds 8 delta-net (GDN) state slots (K <= 7), a short slot 4 (K <= 3);
- with several requests active, one batched draft pass whose 8 rows are shared by the active slots' blocks, the
  taps of every row of the row list, and a rows projection that stores each committed row in its own slot's ring.

**The slots' KV in large pages (operator 2026-11-13; Task 0):** each slot's full-attention KV is held as 2048-token
pages listed in a per-slot page table on the card, instead of one contiguous range at a slot base. 2048 is the
prefill chunk and the prefix cache's snapshot stride, so a page boundary is a chunk boundary. With slots the table is
filled once at startup and never changes, so behaviour is today's. The point is the next step: a shared context
pool in the style of vLLM's then becomes an allocator and scheduler change: pages handed out on demand, reference
counts for shared prefix pages, preemption by recompute. It is not a kernel rewrite. Only the 16 full-attention layers' KV is paged; the GDN
states, the conv ring and the drafter's ring stay fixed per slot. The pool itself is not in this plan.

**Order and why:** Task 0 is the KV layout. It is independent of DFlash and runs first, and it is not covered by
Task 1's stop rule. Task 1 is arithmetic on numbers already on the box and decides whether the plan runs at all. Tasks
2-4 make a lone request work in a slots server (the gain). Tasks 5-6 add the batched draft (so several requests are
not slower than MTP). Task 7 is serving and the record.

**Stop rules:**
- Task 1: a projected total below 0.95x MTP at two requests or 0.90x at four stops the plan for the operator.
- Task 4: after it the server is correct, and fast for a lone request, but drafts nothing in batched steps. If the
  operator pauses here, the mode stays refused in `b70_serve.cc` (one line) and the branch waits.

## Dependencies

- Plans 19v3d and 19v3a merged (the cheaper round; int8 KV with the drafter). Branch `dflash-replica` merged or rebased onto (a drafter per card).
- Branch `spec19v3b-dflash-slots`, own worktree under `.claude/worktrees/`.
- Box data: the gate checkpoint, the drafter, the opencode recordings, `oracle_dflash_dumps`, spec 19 v2 Task 1's
  acceptance record `~/b70/logs/dflash-v2/baseline/acceptance.json`, the f32 judge's reference sets.
- One card for every task but Task 7's two-card rows.

## Global constraints

- No new user flag and no environment variable. Test and probe overrides through `CaptureOptions`.
- **A lone request is bitwise the one-slot server:** slot 0's lists, state and drafter session are today's
  allocations and bindings (the engine's rule for slots, spec 13c). Every task re-runs the one-slot DFlash tests.
- **Nothing else moves:** `tools/kernel_cmdlines` shows additions only; an engine without a drafter, and a
  one-slot engine with one, bind what they bound before.
- Build and test on the box (`tools/box.sh`). GPU command form:
  `tools/box.sh run "tools/gpu_lock.sh card 0 -- <command>"`; long jobs
  `tools/box.sh run "mkdir -p ~/b70/logs/dflash-slots && nohup setsid tools/gpu_lock.sh card 0 -- <command> > ~/b70/logs/dflash-slots/<name>.log 2>&1 &"`.
- Mask ssh output: `sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<ip>/g'`. Timed numbers: interleaved pairs after a
  warm-up, median of rounds, the grade written next to the number.

---

### Task 0: the slots' KV in 2048-token pages (box; independent of the DFlash tasks)

**Files (the agent confirms the list with a grep for the KV base binding first):**
- `src/runtime/buffer_sizes.{h,cc}`, `src/runtime/memory_plan.{h,cc}`: every slot's KV rounded to whole pages; the
  page-table buffer (one table a slot, shared by the 16 full-attention layers: page p of layer l is at
  `layer_base[l] + table[p] * page_bytes`).
- `src/runtime/engine.{h,cc}`, `src/runtime/capture.{h,cc}`: captured lists bind the slot's page table instead of a
  KV base. `setup_slots` fills each table once (slot q's pages are today's contiguous range, so the table is the
  identity at q's offset).
- The kernels that read or write the full-attention KV, in bf16 and in the `kv8` int8 forms: the KV append, decode
  attention (`attn_v2`), the verify kernels (`attn_prep` / `attn_v2` / `attn_verify_dpas` at M 1..8), and the prefill
  flash attention. Each looks up a page base once per 2048 keys. A work-group's key range must not cross a page,
  so any key split is aligned to 2048.
- The prefix cache: store and restore walk pages. The snapshot format and cache key are unchanged while one slot's
  pages are contiguous.
- Tests: `slots_test` and `slot_plan_test` gain a scattered page table (pages of slot 1 interleaved with slot 0's)
  that must give the same ids as the contiguous one.

- [ ] **Step 1:** the plan and the table, host tests first (`slot_plan_test`, `memory_plan_test`): the layouts
  printed for 1/2/4 slots, bf16 and int8, lose at most one page a slot to rounding.
- [ ] **Step 2 (box):** the kernels and the bindings. Run the slot, MTP, DFlash, int8 KV, prefix cache, TP and
  replica GPU tests (`slots_test`, `mtp_gpu_test`, `mtp_gpu_kv8_test`, `dflash_gpu_test`, `dflash_prefix_gpu_test`,
  `dflash_prefix_gpu_kv8_test`, the TP slots tests).
  - **Gate:**
    - **bitwise ids** against main for one request, and for each slot with an identity table and with a scattered
      one. If a kernel's key split cannot be aligned without changing its summation order, its tests go to the f32
      judge and the record says which kernel.
    - decode and the 8-row check at 4k / 32k / 128k context **not slower than 1 %**, measured as interleaved pairs
      after a warm-up, on one card.
  - Commit `kv: the slots' full-attention KV in 2048-token pages behind a per-slot page table`.
- [ ] **Step 3:** merge on its own: it does not wait for the DFlash tasks.

### Task 1: is it worth it? (host arithmetic, no GPU)

**Files:** create `tools/spec/dflash_slots_projection.py` (reads two JSON records, prints one table).

- [ ] **Step 1:** from `acceptance.json` (tokens a round at K = 1..7 on five corpora, the engine's own taps) and the
  MTP acceptance in the saved server logs of spec 19 v2 Task 8's MTP arm
  (`~/b70/logs/dflash-v2/task8/main-0b5ef628/`: the line the server prints per request - drafted, accepted, the
  iterations at each K), compute per corpus the tokens a step for DFlash and MTP at K = 1 and K = 3.
  - If the saved lines cannot separate K = 1 from K = 3 (the MTP arm chooses K per step), use MTP's recorded
    depth-1 acceptance (BENCHMARKS "MTP speculative decoding": golden 0.820, tool calls 0.985) for K = 1. For
    K = 3 the only recorded figure is 0.707, the per-draft acceptance of spec 17d's one-card serving run at depth
    32768 with `--spec auto` (BENCHMARKS "Speculative decoding under tensor parallel") - one prompt set, not a
    per-corpus value: mark every row that uses it "estimated".
- [ ] **Step 2:** project the step costs in plain steps with the measured parts (spec §5.4; with plan 19v3d merged, its re-measured draft pass and row step): row step of 8 rows 1.53
  (four slots), 1.4 (two slots, **estimated** - mark it), batched draft 0.29, MTP's row draft 0.15 a draft index
  plus 0.05 for the head's KV fill. Print DFlash / MTP at two and at four active requests per corpus and their
  geometric mean.
- [ ] **Gate / stop condition:** geometric mean >= **0.95** at two requests and >= **0.90** at four. Below: stop,
  record the table in the spec's §5.4, ask the operator. Either way commit the script and the table
  (`tools/box.sh pull` the output; the table goes into BENCHMARKS "DFlash (spec 19)", block "slots (spec 19 v3b)",
  graded derived).

### Task 2: the plan and the per-slot state (host, then box)

**Files:**
- `src/runtime/buffer_sizes.{h,cc}`: `SlotDims::sizes` with a drafter - slot 0's 7 extra GDN states (today's
  `TapVerifyDims::gdn_spec_bytes`), each further slot's own state plus 3 speculative ones.
- `src/runtime/dflash/dflash_sizes.{h,cc}`, `dflash_buffers.{h,cc}`: the buffers split into the shared part
  (weights' scratch, the RoPE rows at the longest slot) and a session (ring K and V, taps, control block, readback);
  `drafter_plan` gains the per-session bytes.
- `src/runtime/memory_plan.{h,cc}`: `plan_slots` and `slot0_len_that_fits` take a `DrafterPlan`.
- `src/cli/max_len.h`: `settle_slots` passes it and prints the layout.
- `tests/runtime/slot_plan_test.cc`, `tests/runtime/dflash_plan_test.cc`, `tests/runtime/dflash_buffers_test.cc`.

- [ ] **Step 1 (host tests):** the sizes and the plan, test first. The plan test prints `--max-len auto` for 2 and 4
  slots, bf16 and int8, on the 32.53 GB card.
  - Run: `tools/box.sh test '^(slot_plan_test|dflash_plan_test|memory_plan_test)$'`.
  - **Gate:** pass; the four printed layouts recorded in the spec's §5.3 table beside the derived ones (4 slots int8
    derived ~105k + 3 x 32,768). One slot with a drafter plans byte for byte what it planned before.
- [ ] **Step 2 (box):** `dflash_buffers_test` on the card - one session's ring, taps and control equal today's
  buffers' at the same offsets; the one-slot allocation list unchanged.
  - **Gate:** pass. Commit `dflash: the drafter's buffers as shared scratch + a session per slot; plan_slots with a drafter`.

### Task 3: a slot's own lists (box)

**Files:**
- `src/runtime/capture.{h,cc}`: `build_verify_taps` takes a `SlotBind` (the slot's Control, GDN states, conv ring,
  KV base) and the slot's GDN slot count (8 or 4: `gdn_step_slots_M<M>_N8`, or the 4-slot build MTP's lists bind).
- `src/runtime/engine.{h,cc}`: the constructor's refusal goes; `Seq` gains its `TapVerify` and its GDN slot count;
  `setup_slots` gives slot q its states; `capture_slot` captures its verify lists at 1..8 (slot 0) or 1..4 rows;
  `gdn_index` by a per-slot offset; `max_draft()` per selected slot; `zero_slot`, `mtp_normalise_live`,
  `save_state` / `load_state` (the selected slot's ring).
- `src/runtime/dflash/dflash_drafter.{h,cc}`, `src/runtime/tap_drafter.h`: a session per slot - `taps()`,
  `projection(pos, n)`, `reset`, `save_ring` / `load_ring`, the draft list - selected with the engine's slot. The
  interface keeps its one-slot calls; add the slot where a call needs it, nothing more.
- `tests/runtime/slots_test.cc`, new `tests/runtime/dflash_slots_test.cc`.

- [ ] **Step 1:** the engine and the drafter per slot, test first (`dflash_slots_test`):
  - a session run in slot q is bitwise the same session in a one-slot engine: ids, taps, ring, GDN state (the
    slots' isolation, spec 13c's bitwise property);
  - two slots stepped in turn do not move each other's ring, taps or state;
  - a short slot refuses K above 3 by `max_verify_k()`;
  - reset, snapshot and restore of one slot leave the others' bytes alone.
- [ ] **Step 2 (box):** `tools/box.sh run "tools/gpu_lock.sh card 0 -- ctest --test-dir build --output-on-failure -R '^(dflash_slots_test|slots_test|dflash_engine_test|dflash_gpu_test)$'"` (detached; ~40 minutes).
  - **Gate:** all pass, no skip; the one-slot DFlash tests unchanged. Commit.

### Task 4: the slots server with a lone request drafting (box)

**Files:**
- `src/runtime/capture.{h,cc}`: `build_rows` copies each row's taps into a rows tap buffer (bf16
  `[8][5][hidden]`), at the sites `tap_sites` names.
- `src/kernels/dflash/df_prep.cl` and `src/kernels/CMakeLists.txt`: a row-table build of `df_ctx_store` (each row's
  ring and position from the table); name the variants with a suffix that is not `_ROWS` or `_B<n>` (both are
  taken by the drafter's attention geometry) - `_SEQS`.
- `src/runtime/dflash/dflash_capture.{h,cc}`, `dflash_drafter.{h,cc}`: the rows projection list.
- `src/runtime/engine.{h,cc}`: `verify_rows` with a drafter (no MTP head tail; K by each slot's state count);
  the rows projection pending after the per-slot commits, flushed where a slot's own projection is.
- `src/cli/slots_engine.h`, `src/cli/serve_adapters.h`: `SlotsEngine` with a drafter - a lone row is the slot's
  `EngineAdapter::step_dflash`; a batched step runs with K = 0 for now.
- `src/cli/b70_serve.cc`: the wiring behind the existing refusal (the refusal itself stays until Task 7).
- `tests/server/dflash_scheduler_test.cc` (host mock), new `tests/server/dflash_slots_gpu_test.cc`.

- [ ] **Step 1:** rows with taps and the rows projection, test first: after a row step of rows from two slots and
  their commits, each slot's ring equals the ring of the same session stepped alone (bitwise where the row
  kernels are the lone kernels, otherwise each stored row within the f32 judge's drafter form, spec 28 §11).
- [ ] **Step 2:** `SlotsEngine` and the server test: 4 requests on 4 slots, arriving staggered so that lone and
  batched steps mix; greedy ids against the plain slots server's with the f32 judge on first differences; a lone
  request's speculative record equal to the one-slot server's.
- [ ] **Step 3 (box):** the tests of Steps 1-2 and Task 3's, detached on card 0.
  - **Gates:** all pass, no skip, no f32 failure; **a lone request in a 4-slot int8 server decodes within 2 % of the
    one-slot int8 DFlash server** (`tools/box_validate/slots_serve_bench.sh ... --spec dflash --max-num-seqs 4 --kv-cache int8 -- --counts 1`
    against the one-slot run, three rounds; busy-box grade is enough here).
  - Commit. **This is the plan's pause point** (the stop rule above).

### Task 5: the batched draft pass - kernels (box)

**Files:** `src/kernels/dflash/df_prep.cl` (and its split form), `df_attn_rows.cl` (the bound attention geometry
H1 / B4), `df_select.cl` (and its split form), `src/kernels/dflash_dims.h`, `src/kernels/CMakeLists.txt`,
`tests/kernels/dflash_kernels_test.cc`, `tests/kernels/dflash_ref.h`.

- [ ] **Step 1: list the block-aware sites from the sources before changing any.** Expected: `df_embed` (the anchor
  row and the mask rows), the conv prepare / finish if the conv reads a neighbouring row, `df_qkv_prep` (RoPE at
  the row's position), `df_attn_part` / `df_attn_reduce` (the row's ring, and only its own block's rows as block
  keys), `df_select` (one walk a block). Write the list into the kernel file headers.
- [ ] **Step 2:** a block table (the row table's form: per row its block, position, ring index; per block its
  anchor, K, temperature, seed and readback) and the `_SEQS` builds of those kernels. Every row-agnostic launch
  (the GEMMs, norms, SiLU, the head, the top-k stages) is the one-block list's binary unchanged.
- [ ] **Step 3 (box):** the kernel tests extended: a batched pass of blocks (4, 4), (2, 2, 2, 2) and (7, 1) rows
  against each block drafted alone at the same K - draft ids and the 16 candidates equal; hidden rows bitwise where
  the kernels are the one-block kernels, else within the drafter's f32-judge form.
  - **Gate:** pass; `tools/kernel_cmdlines`: additions only. Commit.

### Task 6: the batched draft pass - engine and policy (box)

**Files:** `src/runtime/dflash/dflash_capture.{h,cc}`, `dflash_drafter.{h,cc}` (the batched draft list),
`src/runtime/engine.{h,cc}` (one submit: [pending projections, batched draft, row list]),
`src/cli/slots_engine.h` (`batched()` with a drafter: K per row from the budget and the slot's state count; the
greedy and the sampled acceptance per row as `step_dflash`'s), `src/server/adaptive_k.{h,cc}` (the drafter's cost
row for batched steps), `tests/server/adaptive_k_test.cc`, `tests/server/dflash_slots_gpu_test.cc`,
`tools/probe/probe_slots_speed.cc` (a `dflash` arm beside its `mtp` arms).

- [ ] **Step 1:** the list, the engine call and `SlotsEngine::batched`, test first: greedy ids against the plain
  slots server's (f32 judge on first differences); seeded sampled requests reproduce whatever shares the step.
- [ ] **Step 2 (box):** `probe_slots_speed <target> --kv int8 --slots <the planned layout> --depth 4096` with the new
  arm: per round the row step, the batched draft and the tokens a step at 2 and 4 slots, against its MTP arms.
  - Record in BENCHMARKS "slots (spec 19 v3b)"; put the measured batched draft and row-step costs into the cost
    row (a comment cites the block).
  - **Gates:** the batched draft pass within 10 % of one card's draft pass in ms; tokens a round per slot within
    0.15 of the same K's one-slot value on the tool-call anchors (the acceptance gate, per K); the tests pass.
  - A larger acceptance gap is a defect in the block table, the ring index or the mask: root-cause it with a
    failing kernel test. Commit.

### Task 7: serving, both cards, the record (box)

**Files:** `src/cli/b70_serve.cc` (the refusal "--spec dflash ... beside slots are not built" loses `--spec dflash`;
`--draft-vocab` and `dflash-lookup` stay refused beside slots), `tests/cli/` (its case),
`tools/box_validate/stages.sh` (a new row - 87, or the next free one), `tools/prefix/replay_spec.py` only if a
slots arm needs the server's `--max-num-seqs` recorded in its settings, the docs.

- [ ] **Step 1:** the refusal, the row's stages, `tools/box_validate.sh --quick` on the branch head.
  - **Gate:** the quick run passes; G0 shows additions only.
- [ ] **Step 2 (serving correctness):** golden + tool-call sets through `b70-serve --spec dflash --max-num-seqs 4
  --kv-cache int8` at 1, 2 and 4 clients against `--spec off` in the same mode, judged by
  `tools/box_validate/serve_judge.py`; a prefix-cache-hit turn per slot.
  - **Gate:** no failure, no skip.
- [ ] **Step 3 (two cards, a model copy per card):** `--max-num-seqs 8` on both cards, the same check at 8 clients,
  under `tools/gpu_lock.sh box --`. If card 1 is held by the quantisation, this step waits; it does not hold Step 4.
- [ ] **Step 4 (the speed record, one idle card):** `tools/box_validate/slots_serve_bench.sh` with `--spec dflash`
  against `--spec mtp`, `--max-num-seqs 4 --kv-cache int8`, counts 1 / 2 / 4, three rounds in rotated order. Check
  first that the card is idle and more than 16 CPU threads are free.
  - **Gate (the goal):** 1 request >= 1.10x MTP's on this bench; 2 requests >= 0.95x; 4 requests >= 0.90x; the
    context layout printed at startup recorded.
  - Missed at 2 or 4: record the decomposition (row step, draft, tokens a step). The operator decides between
    "an option that is slower with several requests" (the docs say so) and leaving the refusal in.
- [ ] **Step 5: close.** README (the slots rows only if the operator wants a DFlash row), `docs/18-running-models.md`
  (the recipe, the context, what stays refused), BENCHMARKS, the spec's "as built" lines, open-work item 12.

**Which tasks need the box:**

| task | host only | one card | both cards | CPU model job |
|---|---|---|---|---|
| 1 | yes (reads saved records) | - | - | - |
| 2 | step 1 | step 2 | - | - |
| 3 | - | yes | - | - |
| 4 | - | yes | - | step 2, only for new f32 references |
| 5 | - | yes | - | - |
| 6 | - | yes | - | - |
| 7 | - | steps 1-2, 4 (idle card) | step 3 | step 2, only for new f32 references |

Size: 7 tasks, about 6-8 agent-days. Task 1 is an hour. Tasks 2-6 go to Opus; the test and benchmark runs of
Task 7 to a Sonnet agent.
