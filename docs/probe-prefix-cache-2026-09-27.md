# Probe: prefix caching P0 - host copies and the tail floor (2026-09-27)

Spec 7 §5 P0, plan 7a Task 5. Every number is **measured** unless marked
**derived** or **estimated**. Box: Dell T5810, B70 device 0 (`ZE_AFFINITY_MASK=0`),
121 GB RAM, one GPU job at a time under the shared lock (`flock ~/b70-gpu.lock`).
Build: branch `spec7a-toolcall-probe`, Release. Tools:
`tools/probe/probe_host_copy.cc`, `tools/probe/probe_tail.cc`.

Conditions: an oracle container was running beside both probes (`docker ps`: one
container at the start, another started during `probe_tail`). Load average
3.3 at the start of `probe_host_copy`, 5.1 at its end and at the start of
`probe_tail`, 27.1 at its end; MemAvailable 117.2 GiB before, 69 GB after. The
three runs per row agree within 0.4% (ranges in table 3), so the host load did
not visibly move the device timings.

## 1. The largest pinned host allocation

`l0::Mem(ctx, MemKind::Host, n)` (`zeMemAllocHost`, relaxed size limit), every
4 KiB page touched by the CPU, then the device copies the last 128 MiB out and
the CPU checks one byte. Sizes above MemAvailable - 16 GiB are not attempted:
the cap was **101.2 GiB**, so every planned size was tried.

| host alloc GiB | alloc ms | touch ms | device reads its tail | result |
|---|---|---|---|---|
| 4 | 409 | 556 | yes | ok |
| 8 | 807 | 1213 | yes | ok |
| 16 | 1702 | 2500 | yes | ok |
| 24 | 2373 | 3217 | yes | ok |
| 32 | 3197 | 4521 | yes | ok |
| 48 | 4886 | 6680 | yes | ok |

**Largest pinned allocation: 48 GiB** (the largest size tried; the ceiling was
not searched). Allocation costs about 100 ms per GiB, so the store's budget is
allocated once at server start, not per entry.

## 2. Copy bandwidth on an immediate list

Median of 5 after 2 warm-ups; one `l0::CmdList::immediate` (each append
completes before it returns); pinned host buffer against a device buffer.

| shape | bytes | D2H ms | D2H GB/s | H2D ms | H2D GB/s |
|---|---|---|---|---|---|
| KV block (1 copy) | 134217728 | 9.46 | 14.19 | 10.41 | 12.89 |
| state snapshot (2 copies: 144 + 15 MiB) | 166723584 | 11.78 | 14.16 | 13.34 | 12.50 |
| KV range n=2048 (32 copies) | 134217728 | 9.94 | 13.50 | 10.97 | 12.24 |
| KV range n=60000 (32 copies) | 3932160000 | 276.69 | 14.21 | 305.49 | 12.87 |

"KV range" is the form a restore really takes: 16 FA layers x K and V, one
contiguous `[n][4][256]` bf16 slice each. The 32 separate copies cost 5% against
one copy of the same bytes at n = 2048 and nothing measurable at n = 60000.
Device to host runs at 14.2 GB/s, host to device at 12.2 to 12.9 GB/s.

## 3. The tail prefill at depth

`probe_tail`: model loaded at max_len 131072 (default backend, l0-int8), base
`prefill` of N ids of `tests/golden/prompts/long32k.ids` repeated to length
(untimed), then `prefill` of T more ids timed until it returns (`pos == N + T`,
the first generated id's argmax done). Each run: `reset()` + base + tail; per N
one warm-up, then 3 runs per T.

| N (restart point) | T (tail) | runs ms | median ms | tail t/s |
|---|---|---|---|---|
| 0 | 16 | 316.1, 316.0, 316.0 | 316.0 | 51 |
| 0 | 256 | 345.0, 345.0, 344.9 | 345.0 | 742 |
| 0 | 1024 | 534.6, 537.6, 538.7 | 537.6 | 1905 |
| 0 | 2048 | 929.7, 940.4, 940.4 | 940.4 | 2178 |
| 30000 | 16 | 412.7, 413.4, 413.0 | 413.0 | 39 |
| 30000 | 256 | 538.0, 537.7, 538.1 | 538.0 | 476 |
| 30000 | 1024 | 1017.8, 1018.0, 1020.1 | 1018.0 | 1006 |
| 30000 | 2048 | 1780.8, 1781.3, 1782.3 | 1781.3 | 1150 |
| 60000 | 16 | 504.9, 505.0, 504.4 | 504.9 | 32 |
| 60000 | 256 | 728.1, 728.3, 727.9 | 728.1 | 352 |
| 60000 | 1024 | 1491.6, 1490.9, 1489.9 | 1490.9 | 687 |
| 60000 | 2048 | 2623.7, 2625.7, 2627.2 | 2625.7 | 780 |

The short-tail floor is **316 ms at depth 0** (T = 16), rising to 413 ms at 30k
and 505 ms at 60k: about 3 ms per 1k of depth (derived) for a 16-id tail, the
attention over the cached positions. The ~0.7 s floor quoted in probe-w4a8
§15.6 is not what this build measures at depth 0.

## 4. S1 and S2 (derived)

- **S1**, continuation at 60k history with a 1k tail (bar <= 1.5 s): the resident
  session continues with nothing copied, so time to first token = the tail,
  **1491 ms** (table 3, N = 60000, T = 1024). At the bar with no margin; a
  1k tail at 60k is the case the bar was written for. A 256-id tail: 728 ms.
- **S2**, the main session back after a side request at 60k (restore bar <= 1 s,
  plus the tail): KV range 60000 H2D 305.5 ms + snapshot H2D 13.3 ms = **319 ms
  restore** (derived, table 2), then the tail (table 3): 1810 ms for a 1k tail,
  1047 ms for 256 ids, all derived as sums. Restore meets its bar with 3x margin.
  (The KV below the divergence point `d` is already on the card, so the real
  restore copies `[d, p)` only: 319 ms is the worst case, d = 0.)
- **S3**, write-through cost (bar <= 3%): per 2048-position chunk one KV block
  D2H 9.46 ms + one snapshot D2H 11.78 ms = 21.2 ms against 940 ms of prefill
  for that chunk at depth 0: **2.3%** if not overlapped (derived; deeper chunks
  are slower, so the fraction falls with depth).

## 5. The stopping rule (spec 7 §6)

Stop if host-device bandwidth < 4 GB/s either way or the largest pinned
allocation < 8 GB. Measured: 12.2 GB/s at the slowest (H2D, KV range n=2048)
and 48 GiB. **Not triggered: proceed to 7b.**

## 6. The recorded opencode session

Pending: plan 7a Task 4 (the operator records a session with
`b70-serve --log-requests`) and Task 6 (`tools/prefix/analyze_log.py` on it).

## 7. Write-through cost, S3

Plan 7b Task 3, `tools/probe/probe_writethrough.cc`, branch `spec7b-engine-snapshots`
(Release). Model at max_len 131072, default backend (l0-int8), device 0
(`ZE_AFFINITY_MASK=0`), under the GPU lock with no other GPU job. Load average 1.45
at the start, 1.11 at the end.

Each run is a `reset()` and a cold `prefill` of N ids of `long32k.ids`, timed until it
returns. **plain**: no block hook. **hooked**: `Engine::set_block_hook` with a hook that
does what 7c's store will do at every call (`save_state` into a pinned host buffer, then
`save_kv` of the positions since the previous call, into pinned host memory). From 0 the
aligned chunks are the same 2048-row chunks as the plain run, so the only difference is
the copies. One warm-up pair, then 3 interleaved pairs with the order alternating;
the cost is the median of the per-pair ratios.

| N | calls | plain ms (median) | hooked ms (median) | per-pair ratios | cost | copy time in the hook |
|---|---|---|---|---|---|---|
| 4096 | 2 | 1937.8 | 1976.7 | 1.0203, 1.0220, 1.0201 | **2.03 %** | 43.9 ms |
| 32768 | 16 | 21931.5 | 22213.3 | 1.0122, 1.0129, 1.0124 | **1.24 %** | 349.4 ms |

**S3 met: 2.03 % at pp4096 against the 3 % bar.** The time inside the hook, 22.0 ms
per call (a 128 MiB KV block and a 166.7 MB state snapshot, device to host), matches
§2's 9.46 + 11.78 = 21.2 ms (derived). It accounts for all of the difference: 39 ms of
43.9 ms at N = 4096 and 282 ms of 349 ms at N = 32768 show up in the wall time, so a
few ms per call overlap the host side of the next chunk. The fraction falls with depth
because chunks at depth take longer while the copy stays fixed. The step 3 variant
(copy only at multiples of 4096) was not needed, so it was not measured; the probe has
it as `--every 4096`.
