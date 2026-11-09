# 3e: int8 KV prefill

Completed 2026-11-09 on `kv8-prefill-fast`, code head `a23222f7`, against
main `ba99cb9f`. On one Arc Pro B70, int8 prefill is within 1.19–4.43% of
bf16 from 15,257 to 128,000 ids, with bitwise original int8 attention outputs.
No runtime flags or environment switches were added. Local merge to main
was authorized on 2026-11-09; no push was requested.

## Before/after timings

Qwen3.8-27B GPTQ, l0 prefill, chunk 2048, card 1, max_len 131072,
16 generated ids. Loader and prefill setup are outside the measured window.
Four CLI warm-ups precede three interleaved rounds; round 2 reverses build
and KV-form order. All 48 timed runs completed. Times below are medians in
seconds; speedups and gaps are medians of ratios paired by round.

| ids | main bf16 | main int8 | final bf16 | final int8 | int8 speedup | final int8 vs bf16 |
|---:|---:|---:|---:|---:|---:|---:|
| 15257 | 9.8597 | 20.3679 | 9.8545 | 9.9721 | 2.04x | +1.19% |
| 31000 | 22.2829 | 65.1315 | 22.2861 | 22.7671 | 2.86x | +2.19% |
| 65000 | 57.3032 | 254.0997 | 57.2932 | 59.1362 | 4.30x | +3.22% |
| 128000 | 152.3338 | 927.5463 | 152.3774 | 159.1294 | 5.83x | +4.43% |

Driver: `tools/probe/kv8_prefill_3e.sh`. Logs on the box:
`~/b70/logs/3e/native-driver-v1.log` (`ALLDONE rc=0`) and
`bench-{build-before,build}-{bf16,int8}-<ids>-r<1..3>.log`.

## Profile and implementation

The initial 15,257-id profile isolated the extra time to flash attention.
Baseline capacity was 131072; each KV form was warmed first. GPU execution
and instrumented wall time are separate measurements.

| baseline measurement, ms | bf16 | int8 |
|---|---:|---:|
| unprofiled prefill | 9832.1 | 20355.6 |
| instrumented prefill wall | 10392.0 | 20880.6 |
| flash-attention GPU | 1192.6 | 11876.6 |
| attention prep GPU | 50.7 | 85.2 |
| attention gate GPU | 33.6 | 45.2 |
| other GPU work, excluding flash | 8378.1 | 8196.7 |

Transposed K and transformed V block reads first reduced int8 prefill to
10.721 s, but left an 8.56% gap. The remaining repeated integer conversion
is now done once per layer/chunk: `kv8_widen` materializes **unscaled int8
integers** as exact bf16 values into one reusable layer's temporary.
Flash reads those values with the bf16 kernel's block-read layout, still
multiplying scores by K's scale and probabilities by V's scale in their
original places. DPAS order, online softmax, probability rounding and gating
stay unchanged. This adds no scaled-dequantization rounding or judge.

The temporary costs 4096 bytes per capacity position: **512 MiB per device
at max_len 131072**, allocated for int8 prefill and included in one-card,
pipeline and slot memory accounting. The persistent KV format is unchanged.
Native fp16 scale loads are exact for the finite scales the saturating
writer stores. KT=64, RPW=8, HPW=6, QREG=0 and 256 registers remain selected.
128-register and register-resident-Q probes were slower; full Q residency
spilled 2240 bytes. No further query-cache prototype was run or integrated.

The final diagnostic profile (capacity 65536) reports flash GPU time,
including widening, 1178.2 / 1292.5 ms for bf16 / int8; instrumented wall
10608.9 / 10508.2 ms. The target above uses the unprofiled, equal-capacity
paired matrix, not these instrumented wall times.

## Correctness and memory validation

- A frozen copy of the original flash section is byte-for-byte `ba99cb9f`.
  All fp32 attention and gated bf16 output bytes, including padding, match
  on 13 shapes: causal tails, partial tiles, full 2048-row chunks ending at
  31k / 65k / 128k, and a 4096-row chunk.
- GPU widening matches every host-widened K/V word and preserves tail
  guards. Native scale loads match all **63,488 finite fp16 encodings**,
  including both signs, every subnormal and signed zero.
- A corrupted V operand fails at C=1 (49,104 fp32 words differ). Supplying
  raw bf16 rows to the original int8 reader also fails (48,864 differ).
- All 15 selected tests pass without skips: the original eight gates,
  raw-widening and exhaustive-scale tests, and the five related host
  memory tests. `prefill_gate_l0_test` and `prefill_smoke_test` are unchanged.
- Actual one-card bf16/int8 allocations and launch counts match the plans
  at 16384 and 131072. With an int8 head and int8 KV at 131072, total
  allocation is 22.637 GB, including 1.272 GB prefill scratch.
- Independent review found one missed int8 argument in pipeline runtime
  launch statistics. Its existing regression was observed failing at
  actual-versus-expected launches, then passing after the fix:
  `pp_prefill_kv8_test`, 207.09 s. This also checks bitwise one/two-card
  state, KV and tokens with copy/peer hand-offs and extreme splits.

The slot/batch memory tables add the exact temporary bytes with their
original tolerances. Slot planning uses summed capacity; the older batch
estimate uses the longest slot, and their reconciliation includes that
scratch difference. The existing four-slot int8/MTP layout still fits.

## Recorded opencode replay

Replay all four request bodies from `~/b70/logs/opencode-session-2`, before
and after, int8 KV, spec off, prefix cache disabled, with each complete
recorded response length. Both builds use identical prompt-id arrays;
**all 1076 generated ids match**, with zero cached tokens.

| replay prompt ids | generated ids | main TTFT, s | final TTFT, s |
|---:|---:|---:|---:|
| 591 | 98 | 0.569 | 0.538 |
| 15244 | 163 | 20.351 | 10.120 |
| 16944 | 172 | 24.335 | 11.433 |
| 30967 | 643 | 65.377 | 22.954 |

Request bodies are retokenized by the server, so the longer replay lengths
are 13 ids shorter than the original recorded prompt-id arrays. Logs:
`~/b70/logs/3e/opencode-replay-full-driver.log` (`ALLDONE rc=0`) and
`opencode-replay-full/{summary,before,after}.json`.

## Execution record

All builds used 16 jobs on the box. One-card work used card 1's GPU lock;
the two-card regression used the whole-box lock. The operator shortened
status checks to 1–2 minutes for the takeover. Drivers were immutable copies
under `~/b70/scripts/3e`. Earlier software-scale matrix rows are preserved
under `raw-software-matrix/`; its driver rc=143 is an intentional stop before
the final implementation's complete matrix, not a validation failure.
