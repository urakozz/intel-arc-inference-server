# Spec 13b - batching in the engine

**Status (2026-10-23): refreshed against spec 13 §9 / §9.1 / §9.2 and plan 13a as built (branch
`spec13a-batching-probe`); NOT built. It starts after 13a's box row (66) has filled the step and chunk tables and the
operator has ruled B, the chunk size and the budget rule (probe doc §7) - one implementing agent at a time, on the
operator's go.** The plan of 2026-09-28 (B plain rows, MTP refused) is superseded: the engine implements spec 13 §9's
row table with MTP in every slot.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review. Steps use checkbox (`- [ ]`) syntax for
> tracking. **Commit after every step** (signed; `WIP:` when it does not build yet) and keep a handoff file at the
> worktree root (untracked) when you stop.

**Goal:** `runtime::Engine` carries the model's slots (Qwen3.8 1 x 131072 + 3 x 32768 with MTP; Agnes 1 x 131072 +
1 x 32768 with MTP - the default batched layout - or 1 x 131072 + 3 x 32768 with `--mtp off`; int8 KV - spec 13 §9.1,
§9.2) and runs ONE captured list per (M, shape) over a row table: plain rows of
several slots, MTP verify runs of several slots, and their mixes under the 8-row budget; `BatchEngineIface` on it
(`step_batch`, `step_spec`, per-slot `pos` / `reset` / snapshots); gates B1 (incl. mixed K) and B3 (incl. the
speculative state).

**Architecture:** 13a's row builds promoted: the `BATCH_ROWS` blocks of gdn_step.cl / attn.cl / attn_v2.cl / kv8.cl
become production variants (`*_ROWS`), `build_rows` loses its `#if B70_BATCH_PROBE`, and `runtime/batch_rows.h` is the
host side. Per slot: a KV base and a max_len (one KV allocation at the layout's sum, the slot a position range - the
probe's arrangement), a `Control` row (pos, the pending id, the live GDN index), a conv ring, and - following the
ENGINE's MTP setting (§9.2: MTP trades against slots, it is not per slot) - with MTP `MtpDims::kSlots` GDN states (the
live one and three speculative; slot 0's live state is today's `gdn_state`), the head's KV range and its `hh` row 0;
with `--mtp off` one GDN state, no head KV layer and no head weights. The row table is rewritten by the host before every replay; the lists are captured once per (M, shape),
where the shape is the run lengths (the head's hidden gather is the only shape-dependent part). B = 1 keeps today's
lists, so `--batch 1` is bitwise today's engine (spec 13 §8, §9.2).

**Tech Stack:** OpenCL C (ocloc), C++17, Level Zero, ctest.

**Spec:** spec 13 §9 (the row table, the budget, B1-B3 at mixed K), §9.1 ("What this asks of 13b"), §9.2 (`--batch N`
default 1; no host-RAM offload), §4, §8. Needs `docs/probe-batching-2026-10-23.md` (13a's audit §1, the descriptor
design §6) with the box's tables and the operator's §7 decisions.

## Out of scope

- Batching over `--pp 2` (spec 13 §9.1 follow-on 2: Agnes at concurrency 4 on two cards) - after spec 16's rows 22,
  23 and 27 pass and 13b-13c are done; its own plan.
- The two-card replica mode (spec 13 §10). The MoE follow-on (§9.1 1). DFlash per slot (spec 19
  §5's proposer per slot rides the same row table; its own plan).
- Host-RAM offload of anything (§9.2): a layout that does not fit is refused (Task 1's message), never spilled.
- MTP on in some slots and off in others: superseded by §9.2 (the engine's MTP setting holds for every slot).

## Global Constraints

- Branch `spec13b-batching-engine` from main; box tree per branch; `flock ~/b70-gpu.lock`, detached, polled;
  interleaved pairs after a warm-up, median of 3, idle grade.
- **`--batch 1` (the default) is bitwise today's engine:** the production lists and kernels unchanged (G0 on the box:
  every pre-existing binary's sha256, the golden / replay / prefill gates line for line). The `_ROWS` variants are
  additions (`kernel_cmdlines +N, -0, ~0`).
- One format (AutoRound int4 g64 sym); int8 lm_head for the M = 5..8 rows. No `rm -rf`.

## Review Focus

1. **The row table is the only per-step input:** positions, KV bases, GDN source / destination indices and slots come
   from the table; `Control::pos` is per slot (the slot table), never the list's; `argmax_stage2`'s `pos += n` and
   `cur_token[0] = out[n - 1]` are not what the batched engine reads (the host sets every row's id and position).
2. **Commit per slot:** `step_spec` keeps each slot's accepted prefix (j_s of its K_s drafts): the slot's live index
   becomes the destination of its row j_s, its pos advances by j_s + 1, its `hh` row 0 is x[r0 + j_s]; rejected rows'
   states, conv-ring positions and KV rows are never read (gdn_step's ring disjointness at depth 16 >= 8 + 3).
3. **A slot's prefill while others decode:** prefill writes only its slot's state, ring and KV range (its base) - a
   prefill between two batched steps leaves every other slot's state, ring, KV and head KV byte-identical.
4. **Slot reuse and snapshots:** `reset(slot)` zeroes that slot only; spec 7's save / load per slot (its live state, its
   ring, its KV range, its hh row 0) - restoring slot 2 leaves slot 1 untouched.
5. **Rows at a slot's max_len:** a row past its slot's length is refused by the host (never overruns into the next
   slot's KV range); the MTP K is clipped per slot near its end (`max_verify_k` per slot).
6. **The 8-row budget** as the operator rules it from 13a (e.g. AdaptiveK per request, then clipped to 8 rows a step,
   the longest-running main session first) lives in the scheduler (13c); the engine refuses more than `kM` rows.

---

### Task 1: the slot layout and the planner

- [ ] `runtime::plan` grows a slot list (`SlotSpec{max_len, mtp}` per slot - every slot's `mtp` the engine's -
  `runtime/batch_rows.h`'s `BatchMemory` promoted into `memory_plan.cc`): per-slot KV (with the head's layer when MTP
  is on), per-slot GDN states (4 with MTP, 1 without) and ring, the MTP head's weights only with MTP, the captured
  lists per (M, shape) counted; the layout per model (Qwen3.8 four slots; Agnes two with MTP, four with `--mtp off`);
  `largest_batch(desc, card, mtp)` - the largest batch whose layout fits with and without MTP; short slots never below
  32768, never offloaded. **The refusal (§9.2, its exact text):** a `--batch` that does not fit is refused at startup
  with the message built from the plan (the model, the card's bytes and reserve, what the asked layout needs, the
  largest batch that fits with MTP, the two ways out: `--pp 2`, `--mtp off`); `--mtp auto` with `--batch` asks for the
  largest batch that fits with MTP and says so. Host tests: §9.1 / §9.2's totals byte for byte (Agnes `--batch 4`:
  33.24 GB with MTP, ~29.9 GB without), the message for Agnes word for word, Qwen3.8 never refused at 4,
  `memory_plan_test` unchanged. **Commit** `runtime: the batch layout in the planner (spec 13b)`.

### Task 2: slots in the engine

- [ ] `PersistentBuffers` at the layout's KV sum and a slot table (bases, lengths, per-slot `Control` words), the GDN
  state pool (`kSlots` per slot) and the rings; `Engine` slot API: `reset(slot)`, `prefill(slot, ids)` (the prefill
  path at the slot's KV base and state - prefill's attention reads `[base, base + pos)`), `pos(slot)`, snapshots per
  slot; `buffers_test`'s table unchanged at B = 1; a slot-isolation test (Review Focus 3-4). **Commit** `runtime:
  per-sequence state slots (spec 13b)`.

### Task 3: the batched lists

- [ ] The `_ROWS` variants promoted (13a's `BATCH_ROWS` blocks, now built by default), `build_rows` unguarded;
  `Engine::step_rows(rows)` - one replay of the (M, shape) list over the table, per row its id and logits; plain rows
  and verify runs alike; the head's KV fill per run; per-slot commit (Review Focus 2). **B1:** each slot in a batch
  bitwise equal to it alone at B = 2..slots, at equal and mixed depths, **at mixed K** (a slot verifying K = 3 beside
  slots at K = 0 / 1); **B3:** a slot's rejected drafts leave every other slot's states and head KV byte-identical;
  replay determinism with a fixed arrival order. **Commit** `runtime: the batched decode step (spec 13b, B1 / B3)`.

### Task 4: `BatchEngineIface` on the engine

- [ ] An adapter (`cli::BatchEngineAdapter`) - with `--mtp off`, `spec_k() == 0` and `step_batch` only - implementing `server::BatchEngineIface`: `slots()`, `max_len()` per slot
  (the interface grows `max_len(slot)`; `src/server/batch_engine.h:27` is one value today), `step_batch`, `step_spec`
  (the MTP head's drafts per slot - the draft list at M = 1 per slot today, or one draft step over the slots' rows if
  13a's audit shows it pays), `truncate_to`, the per-slot snapshot calls; host sampling per row with the request's
  own RNG. `dflash_scheduler_test` / `scheduler_test` unchanged against the mock. **Commit** `cli: the batched engine
  behind BatchEngineIface (spec 13b)`.

### Task 5: speed and suite

- [ ] Aggregate t/s at B = 2 / slots, 4k each, plain and with `--mtp` (13a's arms through the engine API: host writes,
  submit, fence); B = 1 within 2 % of today; the full suite; section "Engine" in the probe doc. Box queue row (the next free one after 67) and
  stages. **Commit** `batching: engine gates and speed (spec 13b)`.

**Gate for the plan:** B1 (incl. mixed K), B3 (incl. speculative state), slot isolation, determinism, `--batch 1`
bitwise today's (G0), speed measured, full suite green.
