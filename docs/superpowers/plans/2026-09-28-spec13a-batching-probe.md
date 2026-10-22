# Spec 13a - batching: the probe

**Status (2026-10-23): refreshed against spec 13 §9 / §9.1 / §9.2 (the operator's decisions of 2026-10-23) and main at
`09681a4`; built on the Mac on branch `spec13a-batching-probe` (Tasks 1-3's Mac halves). Nothing has run on a card:
every measured cell of `docs/probe-batching-2026-10-23.md` is PENDING (box) until queue row 66.** The plan of
2026-09-28 (B = 2..4 plain rows, MTP out of scope, `docs/probe-batching-2026-09-28.md`) is superseded: rows are now
(slot, position, speculative state index) triples, MTP runs in every slot under an 8-row budget, and the models are
Qwen3.8 (four slots) and Agnes (two with MTP; four with `--mtp off` - spec 13 §9.2).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review. Steps use checkbox (`- [ ]`) syntax for
> tracking. **Commit after every step** (signed; a step that does not build yet goes in as a `WIP:` commit) and keep a
> handoff file at the worktree root (untracked) when you stop.

**Goal:** measure a decode step whose rows belong to **different sequences** - plain rows of B slots, MTP-shaped runs
(a slot's K + 1 consecutive positions) of several slots, and their mixes up to the 8-row budget - at equal and mixed
depths, at int8 and bf16 KV, on Qwen3.8 and Agnes; prove each batched row equals its sequence stepped alone; price the
prefill chunk that interleaves with the batch; and hand the operator the numbers for 13b's B, chunk size and budget
rule.

**Architecture:** spec 13 §9's row table, built as a PROBE: every per-sequence kernel gets a `BATCH_ROWS` build that
reads a per-row descriptor (`runtime::BatchRow`: position, KV base, GDN source / destination state index, slot) from a
256-byte table in shared memory; the row-agnostic kernels (GEMVs, norms, SiLU, gated head, embed, argmax, lm_head) are
bound at M rows unchanged. `runtime::build_rows` (capture.cc, compiled only under `B70_BATCH_PROBE`) walks the decode
list with those builds and, for MTP-shaped lists, the MTP head's KV fill over the rows (each slot's h_{pos-1} gathered by
copies). The probe allocates its slots inside ONE engine: the KV cache at the sum of the slots' lengths (a slot is a
position range - its KV base), GDN states as indices into a pool (index 0 the engine's `gdn_state`), a conv ring per
slot. Production binaries, every existing kernel command line and every existing kernel's preprocessed source are
unchanged.

**Tech Stack:** OpenCL C (ocloc), C++17, Level Zero; Mac: clang `-fsyntax-only`, `tools/mac/clrun`.

**Spec:** `docs/superpowers/specs/2026-09-28-spec13-batching-design.md` §9 (the row table, the budget, the added gates),
§9.1 (the layouts, the memory method, what 13b needs), §9.2 (`--batch N`, no host-RAM offload; MTP trades against slots: Agnes
`--batch 4` only with `--mtp off`), §4 (B1-B4), §8 (admission). Precedents: plan 19a Task 4 (`B70_VERIFY_M8`,
`probe_mtp_steps`' M = 5..8 lists and interleaved pairs),
plan 8a (`probe_verify_m`).

## The layouts the probe serves (spec 13 §9.1)

| model | decided layout (int8 KV, MTP in every slot) | probe slots (Task 2) | arms' slots |
|---|---|---|---|
| Qwen3.8 | 1 x 131072 + 3 x 32768 | 36864 + 3 x 8192 (61,440 positions) | up to 4 |
| Agnes, MTP on (the default batched layout) | 1 x 131072 + 1 x 32768 | 36864 + 8192 (45,056 positions) | up to 2 |
| Agnes, `--mtp off` (§9.2: one GDN state a slot, no head) | 1 x 131072 + 3 x 32768 | 36864 + 3 x 8192, `--mtp off` | up to 4, plain rows |

The probe's slots are shorter than the served ones (the step's cost depends on the depth a row reads, not on the slot's
length): slot 0 holds the 32k depth plus a 2048-row chunk, the others 4k plus room. Both layouts' memory is a host
table (Task 3), not a probe allocation.

## Global Constraints

- Branch `spec13a-batching-probe` from main; signed commits; no merge, no push. Box tree per branch (`tools/box.sh`'s
  dir), `flock ~/b70-gpu.lock` for every GPU job, detached and polled; timed arms as **interleaved pairs after a
  warm-up, median of 3**, the idle grade recorded, both cards free of DRM holders.
- **`B70_BATCH_PROBE` (default OFF)** builds the probe-mode kernel variants, the guarded `build_rows` in capture.cc and
  `probe_batch_step`. A default build compiles exactly what main compiles. `tools/kernel_cmdlines` lists the probe's
  variants (it sets the option ON for its listing) so the Mac syntax-checks them: `+N, -0, ~0`.
- **Every existing kernel source's preprocessed text is unchanged** for every existing command line (`BATCH_ROWS`
  blocks only; checked by `clang -E -P` on main's and this tree's sources over every existing command line of the
  touched files).
- One format: AutoRound int4 g64 sym weights; the int8 lm_head (the serving form; the M = 5..8 GEMVs exist at int8
  only). No `rm -rf`.

## Review Focus

1. **Which kernels assume one sequence** (Task 1's audit, file:line): `gdn_step` (state carried in registers across
   the M loop, one live slot, the conv window from one ring at `pos`), `attn_prep` / `attn_prep_kv8` (RoPE row and KV
   row at `pos + m`), `attn_decode_v2` / `_kv8` (one K/V wave shared by the M rows, at most two stride groups from one
   `pos`), `attn_reduce_v2` / `_kv8` (`pos + m`), `argmax_stage2` (`pos += n`, `cur_token[0] = out[n - 1]`), the MTP
   head's KV fill (hh row 0 is ONE h_{pos-1}), the draft list (one `hctl`). Row-agnostic: everything else.
2. **Row-mode equals alone, bitwise:** each slot's rows in a B-row step equal that slot's rows stepped alone (the same
   list at M = its row count, the same descriptor rows) - logits bitwise, ids, the destination GDN states, the KV rows
   written, the head's KV rows written; and slot 0 alone equals the PRODUCTION lists (`build` at M = 1, `build_verify`
   at M = 2..4) bitwise, which ties row mode to what ships.
3. **Isolation (B3 in probe form):** a B-row step changes no GDN state but its rows' destinations and no KV row but the
   rows' positions (every slot's source state and a window of KV rows around every slot's position hashed before and
   after).
4. **Attention does not share across slots:** the mixed-depth arms (32k + 4k) price each slot's KV read; the dense M-row
   gains of spec 8b are an upper bound (spec 13 §9.1).
5. **The step table uses the real production kernels at M rows** (int4 GEMVs, int8 lm_head, attention v2) and the
   probe's row builds for the per-sequence kernels only; the chunk arms use `Engine::prefill` as served (MTP head on).

---

### Task 1: the audit

- [ ] **Step 1:** Every kernel of the decode list (`runtime::build`), the MTP verify list (`build_verify`) and the draft
  list (`build_draft`), for Qwen3.8 and Agnes (the same binaries but `_G54` / `_V248089` / `_I19456`): row-agnostic or
  per-sequence, what rows of different slots need, file:line. `docs/probe-batching-2026-10-23.md` §1. **Commit**
  `probe: batching audit of the decode, verify and draft lists (spec 13a Task 1)`.

### Task 2: the probe

**Files:**
- `src/runtime/batch_rows.h` (new, device-free): `BatchRow` / `BatchRowTable` (the device layout), `SlotSpec`,
  `RowRun` and the row-table rules (a slot's rows contiguous at consecutive positions, distinct destinations, at most
  `kM` rows), `BatchLayout` (KV bases), `BatchMemory` (spec 13 §9.1's method over `buffer_sizes.h`).
- `src/kernels/batch_kernels.h` (new): the probe variants' names and the row word indices (`BROW_*`, mirrored in CMake).
- `src/kernels/gdn_step.cl`, `attn.cl`, `attn_v2.cl`, `kv8.cl`: `BATCH_ROWS` blocks (no existing line's tokens change).
- `src/kernels/CMakeLists.txt`: the `B70_BATCH_PROBE` block - the row builds at M = 1..8 (gdn_step at 48 and 54 GDN
  layers, attn_prep bf16 / S1 / KV8 / S1 KV8, attention v2 bf16 / KV8) and Agnes's row-agnostic M = 5..8 rows
  (gate||up', down', SiLU at 19456, argmax at 248089).
- `src/runtime/capture.{h,cc}`: `BatchRowsBind`, `build_rows` under `#if B70_BATCH_PROBE`.
- `tools/probe/probe_batch_step.cc`, `tools/probe/CMakeLists.txt`; `tools/kernel_cmdlines/CMakeLists.txt` (the option
  ON for the listing).
- Mac: `tests/runtime/batch_rows_test.cc` (host: the rules, the bases, the memory table pinned to §9.1's derived
  totals), `tools/mac/clrun/batch_rows_run.cc` (gdn_step's and attn_prep's row builds against the production kernels
  stepping each sequence alone, bitwise on the Mac's GPU - INDICATIVE).

- [ ] **Step 1 (host):** `batch_rows.h` + `batch_rows_test` (TDD): the rules refuse a split slot, a position gap, a
  repeated destination, a destination equal to another run's source, more than 8 rows; the bases; the memory table
  reproduces §9.1 (Qwen3.8 1 x 131072 + 3 x 32768: 29.10 GB total, 1.93 GB spare; Agnes 1 x 131072 + 1 x 32768:
  29.27 GB, 1.76 GB spare; Agnes 1 x 131072 + 3 x 32768: 2.21 GB over with MTP, ~1.1 GB spare with `--mtp off`, 0.47 GB
  over with MTP in the main slot only) and the largest batch that fits with and without MTP (§9.2's refusal numbers).
  **Commit.**
- [ ] **Step 2 (kernels):** the `BATCH_ROWS` builds and the CMake block; `clang -E -P` of every existing command line
  of the four sources identical to main's; `tools/mac_check.sh --kernels` (cmdlines `+N, -0, ~0`, opencl PASS);
  `batch_rows_run` bitwise. **Commit.**
- [ ] **Step 3 (capture):** `build_rows` - the walk's per-sequence sites bound to the row builds with the pools and the
  table, the head's KV fill over the rows (per run: hh[r0] <- the slot's h_{pos-1}, hh[r0 + 1 ..] <- x[r0 ..]),
  `decode_launches` (+10 with the head) asserted. **Commit.**
- [ ] **Step 4 (the probe):** `probe_batch_step <snapshot> [--kv bf16|int8] [--slots ...] [--arms ...] ...`:
  - one Engine at max_len = the slots' sum (int8 head, MTP head loaded); a state pool (one source per slot + 8
    destinations, one per row - the probe's destinations are shared scratch; 13b gives each slot its own) and a ring
    pool; each slot's sequence (long32k.ids at a slot offset) prefilled into slot 0 and copied out (state, ring, KV
    rows, the head's KV rows, h_{pos-1}), slot 0's last;
  - **the check** (Review Focus 2-3) for every arm, then the production tie for slot 0;
  - **the arms**, each against `1x1@4k` as interleaved pairs (order alternating by round, `calls` each, median of 3
    of the ratio), after a warm-up of every arm: plain `B x 1` at 4k (B = 1..4) and mixed (32k + 4k), MTP-shaped
    `2x4`, `4x2`, `1x4+3x1` and their mixed-depth forms - Qwen3.8 up to 4 slots, Agnes up to 2 (`2x4`, `2x2`,
    `1x4+1x1`); Agnes `--mtp off` (an engine without the head) the plain `B x 1` arms up to 4 slots; the production plain step against `1x1` (a lone row must not pay for the row build, spec 13 §8);
  - **the chunk arms:** `Engine::prefill` of one chunk of 512 / 1024 / 2048 ids at depth 4096 and 32768 (slot 0),
    rotated rounds, median of 3, printed beside the 1-row step (the worst inter-token gap of a decoding slot is one
    chunk plus one step).
  **Commit** `probe: batched decode step - row builds, build_rows, probe_batch_step (spec 13a Task 2)`.

### Task 3: the proposal skeleton

- [ ] In the probe doc: §2 the check, §3 the step table (derived columns filled, measured PENDING (box)), §4 the chunk
  table, §5 the memory table from the code (§9.1's method; both models' decided layouts and the alternatives, int8 and
  bf16 KV), §6 the per-row descriptor design for 13b, §7 the decisions the box run feeds (B, chunk size, the budget
  rule) with the rule that turns each measured column into a decision. `docs/README.md` line. **Commit**
  `probe: batching proposal skeleton (spec 13a Task 3)`.

### Task 4: the box row

- [ ] Queue row 66 + `tools/box_validate/stages.sh` row 66: `r66.build` (`-DB70_BATCH_PROBE=ON` in
  `build-batch-probe`), `r66.k0` (G0 must pass: the default build is main's), `r66.check` (the bitwise check, both
  models, both KV forms, Agnes also `--mtp off` at 4 slots), `r66.cost` (the timed arms, idle box), `r66.chunk` (the chunk arms); box-day plan Session 6d.
  `python3 tools/box_validate/test_box_validate.py`. **Commit.**

**Gate for the plan (Mac):** `tools/mac_check.sh --base main --kernels`: host / l0 / opencl PASS, cmdlines `+N, -0, ~0`,
kernels agree (`batch_rows_run` among them); the host test count includes `batch_rows_test`; `clang -E -P` unchanged
for every existing command line of the edited sources; the guarded capture and the probe pass `-fsyntax-only` with
`-DB70_BATCH_PROBE=1`. **On the box (row 66):** the check PASS on both models and KV forms; the step and chunk tables
filled; then the operator's decisions (§7 of the probe doc).
