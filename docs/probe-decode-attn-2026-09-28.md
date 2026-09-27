# Decode attention at depth - probe record (spec 10a)

Plan: `docs/superpowers/plans/2026-09-28-spec10a-decode-attn-probe.md`. Spec:
`docs/specs/2026-09-28-spec10-decode-attention-at-depth-design.md`. Tools:
`tools/probe/probe_decode_attn.{cc,cl}`, `tools/probe/decode_attn_probe.sh`.

Every number is **measured** unless marked **derived**. Device 0 (`ZE_AFFINITY_MASK=0`),
every GPU job under `flock ~/b70-gpu.lock` (the box is shared with plans 7c and 8b, so
"idle" means "nothing else on the GPU during the locked job"; CPU load from other
agents' builds is recorded as `uptime` beside each table).

## P0 - per-kernel profile (Task 1)

`probe_decode_attn capture`: checkpoint loaded at max_len 131072, `Engine::prefill` of
`tests/golden/prompts/long32k.ids` tiled to 130816 ids (142.8 s), then the **profiled
captured list** (`runtime::build` with `ProfileEvents` over the engine's own buffers)
replayed 8 times per depth with `Control::pos` rewound before every step. Columns are
Σ over the 16 FA layers per step (µs), mean of 8 steps. "step wall" is the unprofiled
list's fence wall at the same depth, median of 8. M = 1.

uptime at start / end: load average 12.79 / 9.51 (other agents' CPU builds; the GPU
was held by this job).

| depth | attn_prep µs | attn_decode µs | attn_reduce µs | attn total ms | decode µs/launch | decode GB/s (KV) | reduce share of attn | step Σ ms (profiled) | step wall ms (plain) | partial table MB/layer (derived) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 55.2 | 2399.1 | 179.0 | 2.633 | 149.9 | 112 | 6.8% | 33.123 | 33.674 | 1.61 |
| 32768 | 52.8 | 15314.4 | 2049.2 | 17.416 | 957.1 | 140 | 11.8% | 47.862 | 48.279 | 12.71 |
| 65536 | 52.9 | 29603.2 | 4086.3 | 33.742 | 1850.2 | 145 | 12.1% | 64.054 | 64.543 | 25.39 |
| 130816 | 52.0 | 57391.6 | 9885.0 | 67.329 | 3587.0 | 149 | 14.7% | 97.426 | 98.045 | 50.65 |

- The plain step wall reproduces spec 6's F3 ms/token (34.15 / 48.54 / 64.68 / 98.13,
  `docs/BENCHMARKS.md`) to within 1.4%, so the profile is of the step F3 measured.
- **Decode attention is 69% of the step at 130816** (67.3 of 97.4 ms), and
  `attn_decode` streams the KV at **149 GB/s**, 25% of the 590 GB/s the weight GEMVs
  reach (decode GB/s = one layer's K + V bytes, (d + 1) x 4 KiB, over µs/launch).
- `attn_reduce` is **9.9 ms/token at 130816** (618 µs per launch), 14.7% of attention:
  the partial table is 50.65 MB per layer written by `attn_decode` and read back by
  `attn_reduce` (derived: 24 x 2045 blocks x 258 x 4 B; 1.62 GB per token written plus
  read, 19% on top of the 8.57 GB of KV). Read at 618 µs it is 82 GB/s - the serial
  walk over 2045 blocks per (head, m), not the bytes, is what it costs.
- `attn_prep` is flat (~53 µs/token); it is not a depth term.
- **M = 4 in the list: not available at max_len 131072.** Production M = 2..4 decode
  lists exist only at max_len 16384 behind `-DB70_DECODE_EXTRA_M=ON` (spec 8a); M = 4
  at depth is measured below as the probe's arm 0, attn.cl compiled at M = 4.
