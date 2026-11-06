# Spec 13c - two requests at once: two engines behind the scheduler (Qwen3.8)

**Status (2026-11-07): REWRITTEN for spec 13 §11 (operator: "focus on Qwen3.8. Build the two-engine mode (§10) first:
small, and useful immediately. Then 13b for one-card batching").** Task 1 (the scheduler against `MockBatchEngine`) is on
main since 2026-10-05 and maintained since (the prefix-cache sizing fix 2026-11-04, DFlash per slot 2026-10-16). This
plan serves Qwen3.8 with **two of today's single-sequence engines, one per card, behind `server::BatchEngineIface`** - a
two-slot batched engine with no new kernels - through `b70-serve --parallel replica`. Plan 13b (slots on one card)
follows on the same scheduler.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review. Steps use checkbox (`- [ ]`) syntax for
> tracking. **Commit after every step** (signed, no trailers; `WIP:` when it does not build yet) and keep a handoff file
> at the worktree root (untracked) when you stop.

**Goal:** `b70-serve <Qwen3.8> --parallel replica` loads the model on both cards and serves two requests at once: the
scheduler gives each request a slot (= a card), prefills it in chunks between ticks, and every tick runs one step on
each busy card at the same time (one thread per card). Every response is bitwise the same request served alone on that
card. `--parallel 1 | pp | auto` stay today's serial server, byte for byte.

**Architecture:** `cli::ReplicaEngine` implements `server::BatchEngineIface` over two `runtime::Engine`s (device 0 and
device 1, each with today's plan: the full `--max-len auto` context per card, MTP / lookup per `--spec`): `slots() == 2`,
`max_len(slot)` per card, `prefill(chunks)` and `step_batch(rows)` / `step_spec(rows)` fan out to the cards' threads and
join (one row per slot, at most one row per card - no row table, no row budget). The prefix cache stays ONE host store
(pinned, `--prefix-cache-gb auto`), a snapshot restorable on either card (the host layout is the same). `server.cc`
routes through the scheduler when the engine has more than one slot; the serial path is untouched.

**Tech Stack:** C++17, Level Zero, cpp-httplib, nlohmann::json, ctest.

**Spec:** spec 13 §10 (the replica mode), §11 (order, surface, gates R1 / B2-B4), §8 (admission), §3 (scheduling).

## Out of scope

- Slots on one card, the row table, the 8-row budget, int8-KV layouts: plan 13b.
- Other models (they refuse `--parallel replica` by name); DFlash per slot (the scheduler's hook exists; the engine
  side is spec 19's); replica x pipeline; three or four cards.
- Whether `--parallel auto` picks replica: decided after this plan's record and spec 17b (spec 13 §11.1).

## Global Constraints

- Branch `spec13c-replica` from main; box tree per branch (`tools/box.sh`, `tools/box_dir.sh`); every GPU job under
  `tools/gpu_lock.sh` (`box` for two-card runs); long jobs detached and polled at most every 30 min; timed arms
  interleaved after a warm-up, median of 3, idle grade. No box work while a validation run holds the box.
- Zero-config (operator, 2026-11-06): no new flags or environment variables - `--parallel replica` is a value of the
  existing option; nothing else is user-facing.
- `--parallel 1 | pp | auto` byte for byte today's server (`golden_server_test`, `prefix_server_test`,
  `mtp_server_test` unchanged). Merge after `tools/box_check.sh` passes on the final head. No recursive deletes (move to
  trash).

## Review Focus

1. **Bitwise per card:** a request's ids under replica equal the same request alone on that card (greedy, seeded
   sampling, MTP and lookup) - the engines share nothing but the host prefix store.
2. **Threads:** each card's engine is driven from its own thread; the scheduler's tick waits for both; an exception on
   one card fails that slot's request only (`SlotCacheIface::fail`), never the other card's.
3. **`--prefix-split-last`** through the scheduler: a chat request prefills to len - 1 and runs the last id as one
   decode step, so the next turn restores right there - as the serial path does today (server.cc
   `session.begin(..., prefix_split_last)`).
4. **Prompt lookup per request** (`--spec lookup`): the proposer's history is the request's own (plus its slot's
   earlier requests, as `lookup_history` today); never another request's.
5. **Slot affinity:** a request whose prefix the cache holds goes to a free card; the longest-running session keeps
   its card while it is busy; a request never moves between cards mid-generation.
6. **A client disconnect mid-stream** frees its slot at the next tick; one row's stop / EOS / max_tokens ends only
   that request (inside an MTP run: `truncate_to(slot, pos)`); the slot's final state is snapshotted before reuse.

---

### Task 1: the scheduler (host only)

- [x] `src/server/scheduler.{h,cc}` against a mock batched engine (`tests/server/mock.h` gains `step_batch`); tests: B4
  fairness, cancellation, per-row stops, streaming order. **Commit** `server: the batching scheduler, host only (spec
  13)`.
  - Done 2026-10-05 on the Mac. The engine side is `server::BatchEngineIface` (`src/server/batch_engine.h`: `slots`,
    `max_len`, `pos(slot)`, `reset(slot)`, `prefill(chunks)`, `step_batch(rows)` with a per-request RNG per row, spec
    7's snapshot calls per slot; spec 19d added `spec_k` / `step_spec` / `truncate_to` and AdaptiveK per row, tested by
    `dflash_scheduler_test`); `MockBatchEngine` in `tests/server/mock.h`. Both §8 admission policies are behind
    `SchedulerOptions::admission` (default `Interleave`, chunk 512; `Batched` with `burst_hold_s` 5 ms). Spec 7 per
    slot is `PrefixSlots`. `scheduler_test` prints the fairness numbers in mock ticks: worst inter-token gap 17 at chunk
    512, 65 at chunk 2048, 1025 with batched admission. Not wired into `server.cc` (Task 4).

### Task 2: the scheduler catches up with the server (host only)

- [ ] Against the mock, `scheduler_test` / `dflash_scheduler_test` extended: `max_len(slot)` per slot (the interface
  grows it; `src/server/batch_engine.h` has one value today); `--prefix-split-last` (Review Focus 3); prompt lookup as a
  per-request proposer under `step_spec` (Review Focus 4); `--log-requests` records from the scheduler path (the same
  fields as server.cc's `log_request`, `t_start` = arrival). The interleave chunk: with one row per card a prefill chunk
  stalls only its own card, so the replica engine prefills a slot at the session's chunk cap (2048) - the 512 default
  stays for 13b. **Commit** `server: the scheduler carries split-last, lookup, per-slot lengths and the request log
  (spec 13c)`.

### Task 3: the replica engine (host side + its test)

- [ ] `cli::ReplicaEngine` over two `runtime::Engine`s (Architecture), with a host test over two fake engines: the
  fan-out and join, a failure on one card, per-slot `pos` / `reset` / snapshots, the prefix store shared. The hardware
  gate already requires two cards; replica needs no peer access (say so at startup). **Commit** `cli: two engines
  behind BatchEngineIface (spec 13c)`.

### Task 4: `--parallel replica` in b70-serve

- [ ] `cli/pipeline_args.h` gains the value (help text; the refusal for other models by name; one card visible -
  refused); `b70_serve.cc` builds the two engines (each card's own plan and startup lines: model, max_len, memory) and
  the server routes through the scheduler; `--parallel 1 | pp | auto` paths untouched. Host tests: the parse and the
  refusals (`pipeline_args_test`), `cli_reject_*` rows for replica. **Commit** `cli: b70-serve --parallel replica
  (spec 13c)`.

### Task 5: gates on the cards

- [ ] **R1:** each of two concurrent requests' ids equal to the same request alone on its card - golden prompts
  (prose / code / cjk), greedy and seeded, `--spec off`, `mtp`, `lookup`; **B2:** `golden_server_test --chat` with two
  concurrent clients; **B3:** prefix caching across cards (a snapshot saved on card 0 restored on card 1, C2 equal);
  **B4:** disconnect, per-row stop, streaming order; `prefix_server_test` / `mtp_server_test` unchanged on the serial
  path. Box queue row (the next free one) and its stages (`tools/box_validate/stages.sh`). **Commit** `server: the
  replica engine on the cards (spec 13c, R1 / B2-B4)`.

### Task 6: speed and record

- [ ] Aggregate t/s with two concurrent requests against one (`tools/serve_bench.sh` / llama-benchy at concurrency 2:
  every row says how admission was timed, spec 13 §8); a lone request's latency within 2 % of `--parallel 1`; the
  opencode-shaped replay (`tools/prefix/make_synth_log.py`, side requests sent concurrently) - total wall time against
  the serial server; how often requests overlap in a real opencode session (`--log-requests`, spec 13 §11.1 item 3 -
  it sizes 13b). BENCHMARKS section "Two requests at once (spec 13c)", spec 13 §11's record, README,
  docs/18-running-models.md (Qwen3.8's `--parallel replica` line). **Commit** `spec 13c: two engines behind the
  scheduler (the record)`.

**Gate for the plan:** R1 bitwise, B2-B4 green, the serial modes byte for byte today's, speed measured (misses
recorded), box_check green.
