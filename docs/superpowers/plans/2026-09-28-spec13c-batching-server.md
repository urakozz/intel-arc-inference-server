# Spec 13c - batching in the server

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `b70-serve --batch B` serves up to B requests concurrently: a step loop that runs one batched decode step per tick and interleaves prefill chunks of waiting requests; per-request streaming, stops, sampling and prefix caching; gates B2-B4; the record.

**Architecture:** today `src/server/server.cc` runs one request through `generate()` under a FIFO. The new scheduler owns the engine: requests get a slot, a per-request state machine (queued → prefilling → decoding → done), and a tick loop; each request's SSE writer is fed from the tick loop through a per-request queue; the prefix cache (spec 7) keys per request, its resident session per slot.

**Tech Stack:** C++20, cpp-httplib, nlohmann::json, ctest.

**Spec:** `docs/superpowers/specs/2026-09-28-spec13-batching-design.md` (§3 scheduling, §4 B2-B4, §5). Needs plan 13b (`Engine` slot API, `step_batch`).

## Global Constraints

- Branch `spec13c-batching-server` from main; box tree `~/b70-inference-server-spec13c`. Copy `tools/box.env` if missing; never commit it. Symlink `oracle-out*` dirs.
- Every GPU command under `flock ~/b70-gpu.lock`, detached, polled. `uptime` with timings.
- `--batch 1` (default) keeps today's serial behaviour byte for byte (golden_server_test, prefix_server_test unchanged). `--batch > 1` with `--mtp > 0` is refused. No `rm -rf`. Commit on the branch; no merge, no push.

## Review Focus

1. **A client disconnect mid-stream** frees its slot at the next tick and does not stall the others.
2. **One row's stop / EOS / max_tokens** ends only that request; its slot's final state is snapshotted (spec 7) before reuse.
3. **Prefill fairness:** a long prompt is chunked (the chosen interleave size) so running requests keep emitting tokens; measure the worst inter-token gap for a running request while a 32k prompt prefills.
4. **Sampling seeds per request** stay reproducible regardless of what else is in the batch.
5. **Queue full:** requests beyond B wait (bounded queue as today), and a waiting request's prefix-cache plan is taken when it gets a slot, not when it arrives.

---

### Task 1: the scheduler (host only)

- [ ] `src/server/scheduler.{h,cc}` against a mock batched engine (`tests/server/mock.h` gains `step_batch`); tests: B4 fairness, cancellation, per-row stops, streaming order, Review Focus 1, 2, 4, 5. **Commit** `server: the batching scheduler, host only (spec 13)`.

### Task 2: wiring and gates on the card

- [ ] `server.cc` routes through the scheduler when `--batch > 1`; `EngineAdapter` gains the slot API; B2 golden gates and A4 with 2 and max concurrent requests (each response equal to its serial run under the tie rule); B3 prefix caching per slot (spec 7's C2 per slot, a slot reused by a new request); full suite. **Commit** `server: --batch B on the card (spec 13, B2-B3)`.

### Task 3: speed and record

- [ ] Aggregate t/s at B = 2 / max through the server; single-request latency within 2 % of `--batch 1`; the opencode-shaped synthetic replay (`tools/prefix/make_synth_log.py`) with its side requests sent concurrently: total wall time against the serial server; Review Focus 3's worst inter-token gap. BENCHMARKS section "Batching (spec 13)"; spec 13 amendment §8; README. **Commit** `spec 13: batching in the server (the record)`.

**Gate for the plan:** B2-B4 green, speed measured (misses recorded), full suite green.
