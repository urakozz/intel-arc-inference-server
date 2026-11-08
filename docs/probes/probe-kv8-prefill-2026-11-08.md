# 3e: int8 KV prefill

Baseline: main ba99cb9f, compiled unchanged in this worktree's `build-before`
on the box with 16 build jobs. Card 1 through `tools/gpu_lock.sh card 1`.
Qwen3.8-27B GPTQ, l0 prefill, 2048-row chunks, 15,257 synthetic ids, one card,
16 generated ids, max_len 131072. Each KV form warmed before its instrumented run.

| measurement, ms | bf16 KV | int8 KV |
|---|---:|---:|
| unprofiled prefill | 9832.1 | 20355.6 |
| instrumented prefill wall | 10392.0 | 20880.6 |
| flash-attention GPU | 1192.6 | 11876.6 |
| attention prep GPU | 50.7 | 85.2 |
| attention gate GPU | 33.6 | 45.2 |
| all GPU kernels | 9570.7 | 20073.3 |
| GPU kernels excluding flash | 8378.1 | 8196.7 |

The flash kernel accounts for the extra time: +10,684.0 ms of GPU execution,
against +10,488.6 ms of instrumented wall time. The bf16 and int8 kernels
already share KT=64, RPW=8, HPW=6 and the DPAS/online-softmax structure.
The int8 kernel constructs its K operand with scattered vector loads and its
V operand with scalar loads, while bf16 uses 2D block reads. This is the first
load-path hypothesis to test; the scale arithmetic and accumulation order stay.

Logs: `~/b70/logs/3e/profile-{bf16,int8}-before.log` and
`profile-warmup-{bf16,int8}-before.log`. Both profiles completed. The driver later
exited 1 because its script was updated in place during execution; subsequent
jobs must run fixed script copies. This does not invalidate the completed
profile tables or benchmark rows.


## First candidate: 2D int8 block reads

K uses transposed 32-bit block reads, V uses the 8-bit transform read. Raw
int8 values widen exactly to bf16 in registers; the scales, accumulation
order, tile order and probability rounding are unchanged. No runtime switch.

The detached build and comparison completed with `ALLDONE rc=0`. Every
fp32 output word (including padding) and gated bf16 word matched the original
kernel for nine shapes, through a last position of 128000. Initial 15257-id
unprofiled times were bf16 9875.6 ms and int8 10721.0 ms (8.56% behind bf16).
Profiled flash GPU time fell from 11876.6 to 1904.8 ms (bf16 1179.8 ms).
This is a substantial improvement but does not yet meet the approximately
5% target. Next: full-chunk bitwise cases at long positions, register-resident
Q / GRF compile probes, the required engine gates and interleaved paired runs.

Log: `~/b70/logs/3e/block-read-driver.log`.


## Expanded comparison and engine gates

The comparison now also runs full 2048-row chunks ending at 31000, 65000 and
128000, and a 4096-row chunk. All 13 shapes match every output byte.
`register-driver-2155.log` ended `ALLDONE rc=0`: 8/8 required tests PASS, no
skips (kv8_kernels, kv8_flash_bitwise, prefill_gate_l0_kv8,
prefill_consistency_l0_kv8, prefill_determinism_l0_kv8, kv8_vs_oracle,
prefill_gate_l0, prefill_smoke).

QREG=1 and the default 128-register builds also passed all 13 comparisons,
but were slower. At pos 13209 / C 2048 the three-run median kernel times
were Q0/G256 26.380 ms, Q0/G128 71.506 ms, Q1/G256 39.105 ms and Q1/G128
61.911 ms. At pos 125952 / C 2048: 264.468 / 1861.903 / 366.692 / 1279.859 ms.
Keep QREG=0 and 256 registers. These are kernel-only synthetic-input times,
not the required interleaved end-to-end benchmark.

Oracle scratch honors TMPDIR so this run keeps its gate logs under
`~/b70/logs/3e/tmp/kv8_vs_oracle`.


## Exact raw widening once per layer/chunk

Repeated integer-to-bf16 conversion inside every query tile is the remaining
cost. Widening the **unscaled integers** once per attention call preserves
all bits: every int8 value is exactly representable in bf16. The flash still
multiplies scores by K's fp16 scale and probabilities by V's scale in their
original order, with the original rounding. This adds no scaled-dequantization
rounding and needs no numerical judge or runtime switch.

The first raw-reader probe matched all 13 shapes and reduced the synthetic
128k-position/full-chunk kernel from about264 to172ms. Register-resident Q
spilled2240B and did not help. Native-half-scale loads and alternative head
work-groups offered little benefit; keep the existing scale helper and
HPW6/QREG0/256GRF.

The integrated writer is checked against every host-widened K/V word with
16-word tail guards. The raw flash's complete fp32 and gated bf16 outputs
match the frozen original. A deliberately corrupted V operand failed the
comparison at C=1 (49104 fp32 words differ). Feeding widened rows to the old
reader also failed before the new reader existed (48864 words differ).

One reusable layer's K/V temporary costs4096 bytes per context position:
512MiB at max_len131072, included in both one-card and two-card plans and
allocated only for int8 prefill. The planners' new512MiB assertions failed
before implementation and passed afterward. Actual one-card allocations and
launch counts match the plans at16384 and131072 for both cache formats.
At131072 with an int8 lm_head, the int8 engine totals22.637GB including
1.272GB prefill scratch.

The integrated full build and11 selected tests passed without skips
(281.43s, `final-driver.log`): the eight original gates, the GPU raw-widening
comparison, and the one-card/two-card host memory-plan tests. Fresh review
found a missed int8 argument in the pipeline's runtime launch-statistics
calculation; its existing two-card regression is queued before the fix.

Final15k profile: flash GPU1191.8ms bf16 versus1309.6ms int8 including
widening; instrumented wall10438.4/10538.5ms. Initial unprofiled warmups
9856.9/9974.0ms. Interleaved matrix and recorded opencode replay pending.


The related host slot/batch memory tests initially failed their old totals.
The tests now add the exact temporary size to those totals with their
original tolerances. The slot planner sizes this buffer at the summed
capacity, while the older batch estimate uses the longest slot; the
byte-for-byte reconciliation includes that difference. All five related
host tests pass (`host-memory-green.log`); pipeline-plan tests also pass.
The existing four-slot int8/MTP layout still fits the memory reserve.
