# Probe: tensor-parallel prefill's crossing, 2026-11-10 (spec 17c step 1, operator decision D8)

`tools/probe/probe_tp_pf.cc` on both B70s under `tools/gpu_lock.sh box` (logs `~/b70/logs/17c/probe_tp_pf-1.log`,
`-2.log` on the box). Measured, busy box (3l's matrix between our runs), median of 5 (3 in run 2) after a warm-up.

## What crosses

Under TP every row-parallel linear (out_proj, o_proj, down) leaves each card with half of the output: its K rows'
product, fp32 `[rows][5120]`. The two cards add the halves before the next norm. A Qwen3.8 prefill chunk has 128 such
crossings (64 layers x 2), of which 127 matter (the last down only for the head's row). At 2048 rows one crossing is
40 MiB of fp32 each way (20 MiB in bf16).

## The rate (run 1, `--rows 1024,2048 --reps 8`)

| engine | one way, 2048 rows fp32 | both ways at once | GB/s a direction (both ways) |
|---|---|---|---|
| copy engine (`zeCommandListAppendMemoryCopy` on the copy-only queue group, ordinal 1) | 4.20 ms | 4.35 ms | 9.64 |
| compute list copy (ordinal 0) | 4.21 ms | 4.36 ms | 9.62 |
| kernel push (a kernel on the source card storing into the peer's VRAM) | 4.21 ms | 4.33 ms | 9.69 |
| kernel pull (a kernel on the destination card loading the peer's VRAM) | 24.4 ms | 23.2 ms | 1.81 |

- Every push form runs at the PCIe 3.0 x16 link's ~10 GB/s; both directions at once lose ~4 %. Pulls are 5.5x slower.
- 1024 rows: half the time (2.18 ms both ways); bf16: half the bytes, half the time. The rate is flat from 10 MiB up.
- A cross-device event round trip (card 0 signals, card 1's list waits and signals back): **6.3 us**.
- Both cards report queue groups ordinal 0 COMPUTE+COPY (1 queue) and ordinal 1 COPY-only (1 queue).

## Overlap with compute (run 2, `--gemms 400 --reps 64`)

| | alone | together |
|---|---|---|
| 400 x pf_gemm 2048 x 5120 x 1024 on each card's compute list | 48.72 ms | 48.96 ms (+0.5 %) |
| 64 crossings of 2048 rows fp32 both ways on the copy engines | 278.99 ms | 278.90 ms |

The copy engines' crossings do not slow the GEMMs and the GEMMs do not slow the crossings: the wall is the longer of the
two (278.9 ms), not the sum (327.7 ms).

## What it means (derived)

- One card's 2048-id prefill is 1174.5 ms (3n's record). A chunk's 127 crossings at 2048 rows cost ~553 ms of link time
  a card. In order on the compute list (option T) TP prefill would be ~0.6 s of a card's half of the compute plus
  ~0.55 s of crossings: about one card's time, no gain. bf16 crossings in order: ~0.9 s, ~1.3x, with an extra rounding.
- On the copy engine beside the compute (the wavefront, option W: two chunks in flight, one's crossing while the other
  computes) the crossings hide behind compute as long as a unit's compute is longer than its crossing - at 1024-row
  chunks ~2.2 ms a crossing against ~4-5 ms of a unit's compute (derived).
- The hybrid (H: prefill as `--parallel pp`, then the state scattered into the TP layout) is bitwise one card's, but a
  2048-id prompt is ONE pp chunk - pp runs it on one card after the other, no faster than one card - and H needs both
  weight layouts resident (~9.7 GB more a card) or a reshuffle after every prefill.

**Decision D8 (built at the recommendation):** the wavefront (W) with fp32 crossings on the copy engines, judged
against one card's prefill (golden tie rule); bf16 crossings and the hybrid not built. Recorded in plan 17c's "As built".
