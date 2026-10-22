# Spec 13c - batching in the server

**Status (2026-10-23): refreshed against spec 13 §9 / §9.1 / §9.2 and plan 13a as built; Task 1 done (2026-10-05, the
scheduler against `MockBatchEngine`; spec 19d added `step_spec` / AdaptiveK per row); Tasks 2-3 NOT built - they need
13b (the batched engine behind `BatchEngineIface`).** The plan of 2026-09-28 refused `--batch > 1` with `--mtp > 0`;
spec 13 §9 withdrew that: MTP runs in every slot under the row budget.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review. Steps use checkbox (`- [ ]`) syntax for
> tracking. **Commit after every step** (signed; `WIP:` when it does not build yet) and keep a handoff file at the
> worktree root (untracked) when you stop.

**Goal:** `b70-serve --batch N` (spec 13 §9.2: N = 1, the default, is today's server bitwise; N = 2..slots of the
model - Qwen3.8 4; Agnes 2 with MTP, 4 with `--mtp off`) serves up to N requests concurrently on one card: the scheduler's tick runs one batched
step (plain rows and MTP verify runs under the 8-row budget) and interleaves prefill chunks of waiting requests at the
chunk size the operator rules from 13a; per-request streaming, stops, sampling and prefix caching (spec 7 per slot);
gates B2-B4; the record.

**Architecture:** today `src/server/server.cc` runs one request through `generate()` under a FIFO. With `--batch N > 1`
the scheduler (`server/scheduler.{h,cc}`, Task 1) owns 13b's engine through `BatchEngineIface`: a request gets a slot
(slot 0 - the 128k one - for the longest-running session, the prefix-cache affinity of `PrefixSlots` otherwise), a
state machine (queued -> prefilling -> decoding -> done), and a tick loop; `step_spec` per tick with each request's K
from its `AdaptiveK`, clipped to the budget; each request's SSE writer is fed from the tick loop.

**Tech Stack:** C++17, cpp-httplib, nlohmann::json, ctest.

**Spec:** spec 13 §3 scheduling, §4 B2-B4, §5, §8 (admission), §9 (the budget, B2 with `--mtp`), §9.2 (the option).
Needs plan 13b and 13a's §7 decisions (chunk size, the budget rule, burst hold or interleave).

## Out of scope

- `--batch > 1` with `--pp 2` (spec 13 §9.1 follow-on 4) - refused by name until its own plan; with `--replica`
  (§10). DFlash per slot (`--spec dflash --batch N`): the scheduler's hook exists (spec 19d), the engine side is spec
  19's own plan. Host-RAM offload of anything (§9.2).

## Global Constraints

- Branch `spec13c-batching-server` from main; box tree per branch; `flock ~/b70-gpu.lock`, detached, polled; interleaved
  arms after a warm-up, median of 3, idle grade.
- `--batch 1` (default) keeps today's serial behaviour byte for byte (`golden_server_test`, `prefix_server_test`,
  `mtp_server_test` unchanged). `--batch N` beyond the model's slots, or for a model without a batched engine, is
  refused by name at startup. No `rm -rf`.

## Review Focus

1. **A client disconnect mid-stream** frees its slot at the next tick and does not stall the others.
2. **One row's stop / EOS / max_tokens** ends only that request (inside an MTP run: `truncate_to(slot, pos)`); its
   slot's final state is snapshotted (spec 7) before reuse.
3. **Prefill fairness:** a long prompt is chunked at the ruled size so running requests keep emitting tokens; the worst
   inter-token gap of a decoding request while a 32k prompt prefills is recorded (13a predicts it: one chunk + one
   step).
4. **Sampling seeds per request** stay reproducible regardless of what else is in the batch (the per-request RNG, the
   adapters' re-arm since `82b212f`).
5. **The row budget:** the K per request (AdaptiveK) clipped so a step carries at most 8 rows, by the rule 13a's §7
   fixes; a lone request keeps its full K (`--mtp auto` unchanged at B = 1).
6. **Queue full:** requests beyond N wait (bounded queue as today); a waiting request's prefix-cache plan is taken when
   it gets a slot, not when it arrives.

---

### Task 1: the scheduler (host only)

- [x] `src/server/scheduler.{h,cc}` against a mock batched engine (`tests/server/mock.h` gains `step_batch`); tests: B4
  fairness, cancellation, per-row stops, streaming order, Review Focus 1, 2, 4, 6. **Commit** `server: the batching
  scheduler, host only (spec 13)`.
  - Done 2026-10-05 on the Mac (commit `server: the batching scheduler against a mock batched engine (spec 13c Task
    1)`). The engine side 13b implements is `server::BatchEngineIface` (`src/server/batch_engine.h`: `slots`, `max_len`,
    `pos(slot)`, `reset(slot)`, `prefill(chunks)`, `step_batch(rows)` with a per-request RNG per row, spec 7's snapshot
    calls per slot; spec 19d added `spec_k` / `step_spec` / `truncate_to` and AdaptiveK per row, tested by
    `dflash_scheduler_test`); `MockBatchEngine` in `tests/server/mock.h`. Both §8 admission policies are behind
    `SchedulerOptions::admission` (default `Interleave`, chunk 512; `Batched` with `burst_hold_s` 5 ms). Spec 7 per
    slot is `PrefixSlots`. `scheduler_test` prints the fairness numbers in mock ticks: worst inter-token gap 17 at chunk
    512, 65 at chunk 2048, 1025 with batched admission. Not wired into `server.cc` (Task 2).
- [ ] **Step 2 (host, before Task 2):** the scheduler's row budget (Review Focus 5: K per request clipped to 8 rows a
  step by the ruled rule) and per-slot `max_len(slot)` (13b's interface change) against the mock; `scheduler_test` /
  `dflash_scheduler_test` extended. **Commit** `server: the row budget and per-slot lengths (spec 13c)`.

### Task 2: wiring and gates on the card

- [ ] `server.cc` routes through the scheduler when `--batch > 1` (`b70_serve.cc`: the option, the model's slot count,
  the planner's layout line at startup; a `--batch` that does not fit refused with 13b's message - §9.2's text for
  Agnes `--batch 4` with MTP - and `--mtp auto --batch` taking the largest batch that fits with MTP, said in the startup
  line); **B2** golden gates and A4 with 2 and N concurrent requests, `--mtp` off and
  on (each response equal to its serial run under the tie rule; greedy MTP output identical to plain, per slot);
  **B3** prefix caching per slot (spec 7's C2 per slot, a slot reused by a new request); full suite. **Commit**
  `server: --batch N on the card (spec 13, B2-B3)`.

### Task 3: speed and record

- [ ] Aggregate t/s at N = 2 / slots through the server, `--mtp auto` on; single-request latency within 2 % of
  `--batch 1`; the opencode-shaped synthetic replay (`tools/prefix/make_synth_log.py`) with its side requests sent
  concurrently: total wall time against the serial server; Review Focus 3's worst inter-token gap; llama-benchy at
  concurrency (spec 13 §8: every row says how admission was timed). BENCHMARKS section "Batching (spec 13)"; spec 13
  amendment; README. **Commit** `spec 13: batching in the server (the record)`.

**Gate for the plan:** B2-B4 green, `--batch 1` bitwise today's, speed measured (misses recorded), full suite green.
