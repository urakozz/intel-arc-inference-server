# Spec 13 - batching: several sequences in one decode step

**Status:** design, 2026-09-28, for operator review. Open decisions are marked
**(decide)**.

**Order:** last of specs 10-13: it needs spec 10's decode attention (per-sequence depth
in the stride), and it is much cheaper in memory after spec 12 (int8 KV).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

`b70-serve` runs one request at a time (a bounded FIFO, `--queue 4`). opencode runs
subagents and side requests (titles, summaries) concurrently with the main thread;
today they wait in line, and each one also evicts the resident session (spec 7 then
restores it from host, ~0.35 s at 60k).

Decode is bandwidth-bound: one token reads ~14.3 GB of weights. A step with B rows
reads the weights once. Spec 8b measured the main list at M rows (consecutive positions)
at **1.17 / 1.52 / 1.74** plain steps for M = 2 / 3 / 4, so B sequences decoding
together would give, before attention (derived): **1.71x / 1.97x / 2.30x** aggregate
tokens per second. Attention does not share: each sequence reads its own KV.

## 2. The decision (proposed)

**Continuous batching of up to B sequences (B = 2..4) in the decode step, with
per-sequence state slots, prefill interleaved chunk by chunk with the batch's decode
steps.** One engine, one captured decode list per batch size, rows indexed by slot.

**(decide) B and the context per slot.** Memory at bf16 KV (derived; model 18.1 GB,
~2 GB of other buffers, 32.5 GB card):

| B | max_len per slot | KV | GDN + conv state | fits |
|---:|---:|---:|---:|---|
| 1 | 131072 | 8.6 GB | 0.17 GB | today |
| 2 | 65536 | 8.6 GB | 0.33 GB | yes |
| 4 | 32768 | 8.6 GB | 0.67 GB | yes |
| 2 | 131072 | 17.2 GB | 0.33 GB | no (int8 KV: 8.6 GB, yes) |

A mixed layout (one long slot for the main thread, small slots for subagents, e.g.
1 x 96k + 3 x 8k) matches opencode best but makes the KV addressing per slot.

## 3. Design

- **Slots:** `gdn_state`, `conv_ring`, KV and `Control` (pos, pending token) become
  per slot; each slot has a KV base and a max_len. The server maps a request to a free
  slot; spec 7's cache keys stay per request (the resident session is per slot).
- **Decode kernels:** the M loop today means consecutive positions of one sequence
  (MTP). Batching needs rows of different sequences: the GEMVs, norms, argmax and
  `lm_head` are row-agnostic already; `gdn_step` must index state by slot instead of
  carrying one state across rows; `attn_prep` / decode attention take each row's slot,
  pos and KV base (spec 10 v2's stride per row). A second compile-time mode
  (`ROWS=seqs` against `ROWS=positions`) or a per-row descriptor table.
- **Scheduling:** a step loop in the server: each tick runs one decode step over the
  active slots, and at most one prefill chunk of a waiting request between ticks (a
  chunk of 2048 costs ~1 s, so long prompts delay the batch's tokens: **(decide)** chunk
  size for interleaving, e.g. 512).
- **Sampling:** per row, host-side as today.
- **MTP:** out of scope at first (batch x (K+1) rows); `--mtp` and `--batch` are
  exclusive in the first version.

## 4. Correctness gates

- **B1:** a sequence decoded in a batch equals the same sequence decoded alone, bitwise
  where the row kernels are row-independent (GEMV rows are: the M = 1..4 verify rows are
  bitwise equal to M = 1, spec 8b), otherwise under the golden tie rule.
- **B2:** golden gates and A4 with 2 and 4 concurrent requests; replay determinism with
  a fixed arrival order.
- **B3:** spec 7's C2 per slot; a request finishing and a new one taking its slot.
- **B4, server:** concurrency tests (mock engine): fairness, cancellation, a stop in one
  row not affecting others, streaming per request.

## 5. Speed bars

- **Aggregate decode t/s** at B = 2 / 4, 4k depth each: >= 1.6x / 2.0x of one sequence.
- **Per-request latency:** a single request alone on the server within 2 % of today.
- **opencode-shaped replay** (spec 7's synthetic log with concurrent side requests):
  total wall time recorded against today's serial server.

## 6. Stages

- **13a, probe:** the batched GEMV and `gdn_step` at B rows of different sequences;
  measured step time at B = 2..4 (the table in §1 is derived from consecutive-position
  rows); the memory layout decision.
- **13b, engine:** slots, row descriptors, the batched list(s), B1.
- **13c, server:** the scheduler, slot mapping with spec 7, B2-B4, speed, record.

## 7. Out of scope

- MTP together with batching (a later spec).
- More than one GPU; B > 4.
- Prefix sharing between concurrent sequences (vLLM-style block tables).
