# Spec 22d - Qwen3.8-Flash-Next whole, served: the context default, MTP, snapshots, A4, passkey, the record

**Status (2026-10-10): planned; begins after plan 22b (and 22c when decision 3 or 7 built it), only on the operator's go
after plan 22a's record.** Task 6 runs only when plan 22m has merged. Box queue row 39 (renumbers at build time if
taken).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve <Qwen3.8-Flash-Next>` serves the whole 48-layer model on two cards through the expert tier, as an
agentic-coding server: `--max-len auto` trades KV against the cache by decision 5's rule; the MTP head drafts and the
main model verifies with each row's own experts (the verify union is the miss term); `--mtp auto` gets this model's
measured cost table; spec 7's prefix cache restores bitwise with the tier on; F5 and F4 hold on the whole model; A4
against the CPU reference (spec 21 F6's "after spec 22" row) and passkey at the chosen context; O4 - decode, prefill and
llama-benchy rows against 22a's projection - and the record. When plan 22m has merged, the coding mask composes with
the cache (a mild mask cutting the mirror and the misses).

**Architecture:** no new kernel. spec 22 §5 O4, §6 22d, §7 decisions 4, 5, 6; spec 21 §7 F4 F5 F6, §10 decisions 4, 9.
- **The serve path:** 21e's `cli::qwen4exp::check_serve` stops refusing the whole model when the tier is on (22b's flags
  `--expert-cache / --expert-profile / --expert-mirror / --expert-budget`, 22c's `--expert-swap* / --expert-stream`,
  accepted by `b70-serve` as by `b70-decode`); `cli::pp::PipelineEngineAdapterT<Qwen4ExpEngine>` unchanged - the tier is
  below the engine's interface. The swapper's step boundaries are the adapter's `generate(1)` / `verify` / `draft`
  returns (22c), so a request never waits for a swap.
- **The context default** (decision 5 / spec 21 decision 9): `--max-len auto` with the tier = decision 5's context
  (proposed from 22a's projection table: the largest of 32768 / 65536 / 131072 / 262144 whose projected two-card decode
  is within 5 % of 32768's, PROPOSED) - the cache takes every byte the plan leaves (`expert_budget` at that max_len);
  an explicit `--max-len N` trades the other way and `describe` prints the f it leaves.
- **MTP:** the head resident (decision 6, 22b); `server::MtpCost` gains `qwen4exp_tier()` - this model's verify and
  draft costs in plain-step units **measured** here (Task 2), so `--mtp auto` is built for this family (21e refused it
  naming spec 22's numbers). Verify rows read each row's own experts through the same table: misses cost what the
  union of the M rows' experts costs.
- **Snapshots:** the tier holds weights, not state: 21e's snapshot runs are unchanged; F4 is re-run whole with the tier
  (and across a swap round when 22c built swaps).

**Tech Stack:** C++17, Level Zero, CMake/ctest; bash and Python 3 (the A4 tools, llama-benchy).

**Spec:** `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` (§5 O4, §6 22d, §7 decisions 5, 6, 9);
`docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§7 F4 F5 F6, §10 decisions 4, 9, 11, §16 21e as built);
plan 22a's record (the projection, decision 5's table), plans 22b / 22c as built. Precedents: plan 21e (the adapter, the
snapshot tests, `qwen4exp_mtp_test`'s M2 / M3, `golden_server_test --chat`), `src/server/adaptive_k.h` (`MtpCost`, `--mtp
-cost`), `tools/toolcall/{engine_generate.sh,score.py}` (A4), `tools/probe/kolibri_passkey.sh` (a family's passkey
script), `tools/probe/serve_benchy.sh` (llama-benchy against `b70-serve`, `SERVE_ARGS`, `SERVE_AFFINITY=0,1`).

## Dependencies and branch points

- **Plan 22b merged, row 37 PASS** (the whole model decodes and prefills on two cards, F3 against 21a's full sets);
  plan 22c merged and row 38 PASS when it was built.
- **Row 34's CPU data:** `r34.a4_ref` (`oracle-out-q4exp-a4`: the A4 set and the full model's reference outputs on
  Intel's checkpoint - Task 4's reference) and `r34.accept` (the reference's acceptance - Task 2's comparison).
- **Decision 5** ruled (or the proposal above, recorded as such); **decision 4** (spec 21's `pre_fc_norm_hidden`) is
  re-measured here on the engine and may be ruled from it; **decision 11** (sampling defaults) stays open unless ruled.
- **Plan 22m merged** for Task 6 only.
- **The checkpoint:** Intel's interim (bf16 dense: ~9.3 GB a token of non-expert reads, derived, spec 21 §5) until 21q's
  g64 export exists; every row names the checkpoint, and the rows on ours follow when it exists (the projection's own
  forms).

## Global Constraints

- Branch `spec22d-flash-next-served` from main; box tree automatic; `tools/box.env` copied if missing, never committed,
  never printed; `oracle-out*` symlinked.
- **Commit after each step that changes a file** (signed `git commit -S`; never bypass). No `rm -rf`. No merge, no push.
- **K0 for the server path:** 21e's list (`template_test`, `toolcall_test`, `protocol_test`, `golden_server_test`,
  `prefix_server_test`, `mtp_server_test`, `lookup_server_test`, `pp_serve_test`, `kolibri_server_test`,
  `k2_server_test`, `qwen4exp_server_test`) unchanged; Qwen3.8's `MtpCost` defaults unchanged. No kernel binary changes.
- Every number measured, or marked derived / estimated / proposed. Box: `flock ~/b70-gpu.lock` (both cards), detached,
  polled; interleaved pairs after warm-up, median of 3, `uptime`, idle grade; the operator's llama-benchy flags for any
  server row; the RAM rule (no CPU oracle beside the pinned mirror and PLE).
- Mac checks `tools/mac_check.sh --base main --quick`.

## Review Focus

1. **The whole model is refused only for a reason that still holds.** `check_serve` / `check_args` refuse the whole
   model only with `--expert-cache off`, when the non-routed part does not fit, or when the host rule fails - each by
   name with the bytes; every other 21e refusal stands.
2. **F5 on the whole model is bitwise where it was.** Verify rows at M = 2..4 equal the plain steps (logits, routes,
   selections, state) with the tier on and residency mixed; greedy MTP output equals greedy plain output (M3), with
   swaps on and off.
3. **The cost table is measured, not carried over.** `MtpCost::qwen4exp_tier()`'s every entry comes from Task 2's run
   (the commit names the run and its date); `--mtp-cost` still overrides it.
4. **Restores are bitwise across the tier's state.** A snapshot taken before a swap round restores after it with a bitwise
   continuation; 21e's cross test (`--pp 2` -> `--pp 1` and back) runs where one card can hold the model - Intel's
   `--layers 18` with the tier forcing 50 % residency.
5. **Every speed row says what it ran:** the checkpoint, f per card, h measured (from 22c's counters, or from route
   read-backs against the map), swaps on / off, streaming on / off, `--mtp` K, the depth; against the projection's row.

---

### Task 1: the serve path, the context default

**Files:**
- Modify: `src/cli/qwen4exp_serve.h` (`check_serve`: the whole model with the tier; the tier flags parsed beside
  b70-decode's), `src/cli/qwen4exp_decode.h` (`settle`: `--max-len auto` with the tier = decision 5's rule; `describe`'s
  f line), `src/cli/b70_serve.cc` (`serve_qwen4exp`: the flags, the startup memory and tier lines), `src/runtime/qwen4exp/qwen4exp_sizes.{h,cc}`
  (`max_len_auto_with_tier(...)`), `tests/server/qwen4exp_server_test.cc`, `tests/runtime/qwen4exp_plan_test.cc`,
  `tests/CMakeLists.txt` (`cli_reject_serve_qwen4exp_full` now with `--expert-cache off`; `cli_reject_serve_qwen4exp_host_ram`)
- Test: `qwen4exp_server_test`, `qwen4exp_plan_test` (host); `cli_reject_serve_qwen4exp_*` (binary)

- [ ] **Step 1: failing tests** (the whole model accepted with the tier; refused with `--expert-cache off` naming the
  bytes; the host-RAM refusal naming mirror + PLE + MemAvailable; `max_len_auto_with_tier` = decision 5's value at the
  32.53 GB cards, and the f it leaves printed). FAIL.
- [ ] **Step 2: implement**; PASS; the K0 server tests unchanged. Mac gate. Commit `git commit -S -m "cli: b70-serve serves
  Qwen3.8-Flash-Next whole through the expert tier - its flags, --max-len auto by decision 5 (spec 22d)"`.

### Task 2: MTP on the whole model; the cost table

**Files:**
- Modify: `tests/runtime/qwen4exp_mtp_test.cc` (a `whole` mode: the tier on, `--pp 2`), `src/server/adaptive_k.{h,cc}`
  (`MtpCost::qwen4exp_tier()`), `src/cli/qwen4exp_serve.h` (`--mtp auto` allowed, with that table), `tests/server/adaptive_k_test.cc`,
  `tests/CMakeLists.txt` (`qwen4exp_mtp_whole_test`, label `checkpoint;qwen4exp;pp`)
- Test: `qwen4exp_mtp_whole_test` (card), `adaptive_k_test` (host)

- [ ] **Step 1: the card test** (whole model, Intel's checkpoint, two cards, the tier on, swaps on and off): M2 at M = 2..4
  bitwise; M3 greedy lossless at K = 1..3 over `q4exp_short` and `q4exp_agentic`; the engine's acceptance by depth on
  the golden continuations and four A4 scenarios for both `B70_Q4_MTP_NORM` forms and both `B70_Q4_MTP_SELECT` modes
  (decisions 4 and 5 of spec 21, now from the engine, beside `r34.accept`'s reference numbers); the cost - a draft step
  and a verify at K = 0..3 in plain-step units, interleaved, median of 3 - and the per-layer expert union of the verify
  rows (`read_verify_routes`) with its misses against the map. FAIL until the tier binds the head (22b) - then PASS on the
  box.
- [ ] **Step 2 (box):** run it; commit the measured `qwen4exp_tier()` table (`adaptive_k_test` asserts its shape and
  that Qwen3.8's tables are unchanged) and `--mtp auto` for this family. Commit `git commit -S -m "server: Qwen3.8-Flash-
  Next's measured MTP cost with the expert tier - --mtp auto for the whole model; F5 whole (spec 22d)"`.

### Task 3: snapshots and the prefix cache on the whole model

**Files:**
- Modify: `tests/runtime/qwen4exp_snapshot_gpu_test.cc` (a `whole` mode), `tests/CMakeLists.txt`
  (`qwen4exp_snapshot_gpu_whole_test`)
- Test: `qwen4exp_snapshot_gpu_whole_test` (card)

- [ ] **Step 1:** 21e's saves (block ends 2048 / 4096, request ends 2049-2052, 5000, 1) on the whole model with the
  tier, restored into a fresh engine: continuations bitwise a cold run's; with the head (K = 2); across a forced swap
  round between save and restore (22c built: `B70_Q4_SWAP` stress for 64 steps in between); `--pp 2` -> `--pp 1` is not
  possible whole (it does not fit one card) - the cross test runs on Intel's `--layers 18` with the tier forcing 50 %
  residency instead (recorded). Commit `git commit -S -m "tests: prefix-cache restores bitwise on the whole model with
  the expert tier, across swap rounds (spec 22d)"`.

### Task 4: serving gates - the chat path, A4, passkey

**Files:**
- Create: `tools/probe/q4exp_passkey.sh` (`kolibri_passkey.sh`'s form: `passkey.py`'s prompts at the context asked,
  `b70-decode --prefill` on the whole model, `--tokenizer` the original's)
- Modify: `tools/box_validate/decode_extra.sh` (21e's `golden_server_test --chat` invocation: a whole-model arm),
  `tests/CMakeLists.txt` (`golden_server_qwen4exp_whole_test`)
- Test: `golden_server_qwen4exp_whole_test` (card); the A4 and passkey stages (opt-in)

- [ ] **Step 1:** `golden_server_test --chat` against `b70-serve <intel> --pp 2 --tokenizer <orig> [--mtp 2]` = `b70-decode
  --prefill` on the response's prompt ids (prose / code / cjk, 32 greedy ids), and the prefix-cache repeat of two
  2.7k-id A4 scenarios identical to a cache-off server (21e's r34.serve, whole). Commit `git commit -S -m "tests: the
  whole Qwen3.8-Flash-Next served = b70-decode, prefix-cache repeats identical (spec 22d)"`.
- [ ] **Step 2 (box): A4** - `tools/toolcall/engine_generate.sh <intel> oracle-out-q4exp-a4/set <out> l0` (`LM_HEAD=int8`
  and `bf16`, 192 ids, `MAX_LEN` 16384) and the set as chat requests through `b70-serve` (`serve_client.py`); `score.py`
  against `<name>.bf16.txt` - recorded, the bar is the operator's (K2's precedent: "recorded, no bar"); spec 21 F6's
  "after spec 22" A4 row.
- [ ] **Step 3: `q4exp_passkey.sh`** (`bash -n`; `DRY_RUN=1`); **(box)** passkey at 32768 and at decision 5's context
  (depths 10 / 50 / 90 %), the whole model, with and without `--mtp 2`. Commit `git commit -S -m "probe: Qwen3.8-Flash-
  Next passkey on the whole model (spec 22d)"`.

### Task 5 (optional, by 22a's numbers): the embedding in host USM

- [ ] Only if 22a's projection shows the embedding's 1.27 GB back in the cache gains >= 1 % decode t/s at decision 5's
  context: `--embed-host` allocates `Q4DevicePart::embed` as host USM (one row a token zero-copy, `embed_gather_M1_D2560`
  and `pf_embed_gather_D2560` unchanged; the two-card head reads it from host instead of over peer); bitwise the device
  embedding (ids, logits); the decode rows paired. Otherwise recorded "not built: <the projected gain>". Commit `git
  commit -S -m "runtime: the embedding from host USM, its VRAM to the expert cache (spec 22d, optional)"`.

### Task 6 (only when plan 22m has merged): the mask with the cache

**Files:**
- Modify: `src/loader/qwen4exp_experts.cc` (masked experts are `None`: no slot, no mirror bytes), `src/runtime/qwen4exp/qwen4exp_experts.cc`
  (the fill and the budget over the kept experts only), `tests/runtime/qwen4exp_tier_o1_test.cc` (a `mask` mode),
  `tests/CMakeLists.txt`
- Test: `qwen4exp_tier_o1_mask_test` (card)

- [ ] **Step 1:** with `--expert-mask FILE` and the tier: the dropped experts' entries are the poison block and never
  read (the route rows hold no dropped id - checked every step; a poisoned read would NaN the logits); the masked model
  with forced residency 100 / 50 / 0 % bitwise (Intel's 18 / 37 layers); the whole model with 22a's 87.5 % mask: the
  mirror shrinks by the dropped bytes, f and h recorded, the golden gate against 22m's masked reference sets. Commit `git
  commit -S -m "runtime: the coding mask composes with the expert cache - dropped experts neither cached nor mirrored
  (spec 22d)"`.

### Task 7: the record - O4, llama-benchy, docs, the queue row

**Files:**
- Modify: `docs/BENCHMARKS.md` ("Qwen3.8-Flash-Next (spec 21)": the whole-model rows - decode at 4k / 32k / 128k with
  and without MTP, prefill pp4096 / pp32768, the llama-benchy rows, each with Review Focus 5's columns, against 22a's
  projection), `docs/19-running-models.md` (the whole model: the command, the flags, RAM, the expected rows),
  `docs/superpowers/specs/2026-10-09-spec22-moe-expert-offload-design.md` ("22d as built", O4, decision 5 as ruled),
  `docs/superpowers/specs/2026-10-09-spec21-qwen4exp-design.md` (§7 F6's "after spec 22" rows: A4, passkey; one line
  pointing at spec 22's record), `docs/superpowers/plans/box-validation-queue.md` (row 39), `tools/box_validate/stages.sh`
  (a `row 39` block), this plan's status line

- [ ] **Step 1: the row** - `r39.k0` (G0: no kernel moved), `r39.host`, `r39.reject` (`cli_reject_serve_qwen4exp_*`),
  `r39.mtp` (Task 2), `r39.snapshot` (Task 3), `r39.serve` (Task 4 Step 1), opt-in `r39.a4`, `r39.passkey`, `r39.speed`
  (`b70-decode --bench` rows), `r39.benchy` (`serve_benchy.sh` with `MODEL=<intel>`, `SERVE_ARGS='--pp 2 --tokenizer
  <orig> [--mtp 2]'`, `SERVE_AFFINITY=0,1`), `r39.mask` (Task 6, when 22m merged); rownotes: data (rows 30 / 34's),
  RAM, both cards idle for the timed rows. `test_box_validate.py` passes; `--dry-run --only r39`. Commit `git commit -S
  -m "box: queue row 39 - spec 22d, Qwen3.8-Flash-Next whole and served"`.
- [ ] **Step 2 (box): O4** - the speed and llama-benchy rows, paired where two arms compare (MTP on / off, swaps on /
  off), each against the projection's row; the record written (BENCHMARKS, docs/19, spec 22 "22d as built", spec 21 F6).
  Commit `git commit -S -m "spec 22d: Qwen3.8-Flash-Next whole on two B70s - the record (O4, A4, passkey, llama-benchy)"`.

**Gate for the plan:** Mac - `qwen4exp_server_test`, `qwen4exp_plan_test`, `adaptive_k_test` (Qwen3.8's tables
unchanged), the K0 server tests, the refusals, `mac_check.sh --quick` exit 0. Box - G0; F5 whole (M2 bitwise, M3
lossless) with the tier, swaps on and off; F4 whole (restores bitwise, across a swap round); the server = b70-decode;
A4 and passkey recorded against the reference; O4's rows beside the projection; `--mtp auto`'s table measured; the
record written.
