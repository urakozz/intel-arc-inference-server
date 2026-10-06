# Probe: int8 KV on Qwen3.8 - the 12a repeat (2026-10-06)

Plan 12b Task A: spec 12a's int8 KV probe repeated on Qwen3.8 27B (bf16 base, snapshot
`1d4bf0f`) on the Mac CPU, `tools/oracle/kv8_qwen38_repeat.sh`, container `agnes-ref-img`
(torch 2.14.1+cpu, transformers 5.15.0), 2026-10-06 04:47-10:05. Raw tables:
`oracle-out-12a-qwen38/summary.md` and `long4096.replay.md` (gitignored). The Agnes run it
repeats: `docs/probe-int8-kv-2026-09-28.md`.

Every metric is a variant against the bf16-KV forward in the same batch (`pt` per-token int8,
`kivi` per-channel K / per-token V, `rot` Hadamard-rotated K, `rotkv` K and V rotated with the
output un-rotated in fp32; `f32attn` fp32 attention without quantisation and `ctl_bf16out`
bf16-rounded logits are the controls).

## Verdict

**rotkv holds on Qwen3.8; spec 12 §8's ruling stands. The tolerances are set (below).**

| check (plan 12b Task A) | Agnes | Qwen3.8 | result |
|---|---|---|---|
| 1. stop rule: rotkv golden decision cos mean vs `l0-int8`'s 0.999931742 | 0.9999467 | 0.9999675 / 0.9999502 / 0.9999671 (prose / code / cjk) | pass on all three |
| 2. Q2: rotkv golden 1 - cos mean | 5.3e-5 | 3.3e-5 / 5.0e-5 / 3.3e-5 | under 1e-4 |
| 2. Q2: rotkv golden KL mean (bar 1e-3) | 1.9e-4 | 1.5e-4 / 6.7e-5 / 2.1e-4 | pass |
| 3. Q3: long4k 2048-4095 bucket 1 - cos mean | 3.1e-4 | 2.3e-4 | lower |
| 3. Q3: long4k last row (= the engine test's 4096 row) | 1.3e-4 | 7.7e-5 | lower |
| 3. replay flat 4k → 32k: KV:rot rel L2 mean (ctl_eager's) | - | 5.3e-3 → 5.0 / 5.1 / 5.1e-3 at 8k / 16k / 32k (6.1e-3 → 6.1 / 6.1 / 6.2e-3) | flat, below bf16 eager's own |
| 4. argmax, non-near-tie (gap > 0.05), bar ≤ 0.5 % | - | 0 / 512, 7 / 1536 (0.46 %), 3 / 2048 (0.15 %) | pass; `ctl_bf16out` alone flips 6 / 7 / 9 |
| 5. scheme | rotkv | rotkv best cos mean on every golden prompt and long 0-511 / 2048-4095; kivi ahead on long 512-2047 (cos 0.99945 vs 0.99928) and on code KL | rotkv stays |

The int8 attention error (KV:rot rel L2 ~5e-3) is below the error bf16 eager attention itself
makes against fp64 (ctl_eager ~6e-3), at every depth to 32k tiled. The rotation check is
exact (3.8e-15).

## One row's 1 - cos has a heavy tail

Per-row cosines reach 0.91 (rotkv, long 512-4095) and 0.80-0.94 (t1 / t2, the shared system
prompt's first 512 rows; `pt` down to 0.70) on rows where the bf16 logits are flat. Those
rows move as much under fp32 attention without any quantisation (`f32attn` min 0.58 on t2
512-1022). The last rows of the A4 prompts: rotkv 1 - cos 1.26e-3 (t1) and 5.9e-4 (t2), with
`f32attn` 6.3e-4 and 4.1e-4 on the same rows. A per-row bar of 5e-4 would fail on such a
row without any defect, so the bar is split.

## The tolerances (12b)

| where | before (PROVISIONAL, Agnes-derived) | now |
|---|---|---|
| `kv8_vs_oracle_test` (`tools/probe/kv8_vs_oracle.sh` `KV8_TOL`) | golden mean cos drop ≤ 1e-4 | ≤ 1e-4, confirmed (2-3x the measured 1 - cos) |
| `flash_long_kv8_test` (`flash_long_test.cc`) | last row 1 - cos ≤ 5e-4 at every depth | ≤ 2e-3 at every depth (breakage), ≤ 5e-4 averaged over the four depths; oracle drop ≤ 5e-4 unchanged |
| `kv8_kernels_test` gated flash rows | ≥ 0.9999 | unchanged, still PROVISIONAL: kernel rounding against the same dequantised reference, which this CPU run does not measure; the card's first run sets it |

## Timing

Golden prompts 10-37 min each, A4 46-47 min each, long 4096 x 5 variants 110 min, replay
21 min; about 5 h 20 min in all on the shared Mac.
