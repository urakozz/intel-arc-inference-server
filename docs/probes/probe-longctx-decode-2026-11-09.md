# Long-context decode anatomy — 2026-11-09

Work item 3c complete. **GO for a dedicated decode-attention follow-up:** at
150k, attention is 41.9% of the plain bf16 step and 57.9% of the int8 step.
These are the original 3c measurements. The later 3f implementation and its
separate before/after measurements are recorded below.

## Method

One Arc Pro B70, Qwen3.8 `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`,
int8 output head, full vocabulary, one sequence, capacity 150272 for every arm.
The model snapshot is `84575a18f209992ef96d819b31f924b489e3d55d`.
Reported engine buffers including prefill scratch occupy 29.567 GB with bf16
KV and 24.992 GB with int8 KV, on a 32.530 GB card.
The timed driver holds the whole-box lock throughout and selects physical card 1
through the probe's device argument. It never sets an engine environment variable.
Smoke runs used the card-1 lock with its masked device index 0.

The existing CLI's profile mode captures only the plain bf16 list and rejects
int8 KV. The older attention capture probe fixes capacity at 131072. The new
`tools/probe/probe_longctx_decode.cc` therefore calls the existing capture and
profiling APIs directly. No engine or kernel behavior changes.

`tests/golden/prompts/long32k.ids` is tiled, then prefilled with the production
l0 backend in increments ending at 4096, 32768, 81920 and 150000. At each depth,
the probe saves GDN state, convolution ring, main/head control and MTP hidden
state outside timing. It restores that genuine prefix state before each arm
and before continuing prefill. Future KV rows overwritten by a probe replay
are outside the saved prefix. Each timed replay fixes the starting position
and supplies the tiled prompt's next IDs; recurrent state evolves within the
replay block. This is a populated-cache cost probe, not a response-quality replay.
Depth is the starting `Control::pos`: main-attention row r reads depth + r + 1
keys, including its newly written row.

Three outer rounds load bf16/int8, int8/bf16, bf16/int8, one engine at a time.
Within each depth, plain M = 1 and verify M = 4 run in alternating order by
round; their order is reversed between wall and timestamp phases. Every phase
and arm gets 128 uninstrumented warm-up calls, followed by 128 wall calls or
32 instrumented calls. Model loading, prefill and state copies are excluded.
The wall interval includes control writes, queue submission and fence waits.
The box remains idle on its other card.

Wall and profile captures have identical launch labels and counts. Profile
events signal after every kernel and perturb execution; their duration sum
is used for anatomy and shares, separately from uninstrumented wall time.
Attention includes preparation, the decode/DPAS scan, and reduction over all
16 FA layers, plus the verify head's preparation. Other time is the remainder
of the full kernel sum, including weight linears, GDN, norms and output head.
The verify list appends the MTP head's KV fill; it does not execute the draft
head's attention scan. Its head preparation is included and reported separately.

M = 4 uses the production depth/KV selection table: DPAS at all four sampled
depths, for both cache forms. M = 1 uses v2. Verify throughput is four rows per
call divided by wall time; it is not accepted speculative tokens per second.
No draft generation, acceptance decision or serving overhead is included.
Plain throughput is the uninstrumented single-row replay rate.

One arm invocation on the box (repeat with the six KV/round pairs above):

```sh
tools/gpu_lock.sh box -- build/tools/probe/probe_longctx_decode \
  urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ bf16 1 0 \
  4096,32768,81920,150000 150272 128 32
python3 tools/probe/summarize_longctx_decode.py ~/b70/logs/3c/matrix-*-r*.log
```

## Verification and artifacts

- Full box build passed. Both KV forms completed the 4096-depth smoke at full
  capacity, with 774 plain / 784 verify launches and 48 / 49 attention launches.
- Invalid capacity is rejected before GPU initialization.
- `test_longctx_decode_summary.py`: five tests passed after an observed RED
  import failure. Fixtures reject missing arms, duplicates, nonfinite/negative
  times, broken launch/category accounting and inconsistent throughput.
- The result tool requires all 48 unique depth/KV/M/round measurements, equal
  capacities, valid category sums and completed logs before printing a final table.
- Box logs: `~/b70/logs/3c/driver.log`, `matrix-{bf16,int8}-r{0,1,2}.log`,
  `summary.json`. Probe source and plan first committed as `bfa3be0d`.

## Results and decision

All 48 unique measurements completed and validated. The driver ran from
01:14:09 to 01:50:17 UTC, ending `ALLDONE rc=0`. The whole-box/card locks were
free afterward. Values below are medians of three round means. Shares are
medians of the per-round ratios; independently rounded medians need not add
exactly. Wall ranges show all three rounds, not confidence intervals.

### Plain decode, M = 1

One row is one token step; rows/s is the fixed-position replay rate.

| depth | KV | wall ms | kernel sum ms | attention ms | other ms | attention share | rows/s | wall range ms |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 4096 | bf16 | 29.584 | 29.169 | 0.916 | 28.253 | 3.1% | 33.80 | 29.547–29.595 |
| 4096 | int8 | 30.648 | 30.196 | 1.959 | 28.236 | 6.5% | 32.63 | 30.612–30.653 |
| 32768 | bf16 | 33.591 | 33.194 | 4.893 | 28.297 | 14.7% | 29.77 | 33.576–33.611 |
| 32768 | int8 | 38.018 | 37.575 | 9.456 | 28.120 | 25.2% | 26.30 | 37.999–38.026 |
| 81920 | bf16 | 40.186 | 39.704 | 11.399 | 28.305 | 28.7% | 24.88 | 40.169–40.234 |
| 81920 | int8 | 50.597 | 49.859 | 21.788 | 28.071 | 43.7% | 19.76 | 50.594–50.603 |
| 150000 | bf16 | 49.357 | 48.750 | 20.426 | 28.328 | 41.9% | 20.26 | 49.304–49.424 |
| 150000 | int8 | 67.325 | 66.477 | 38.460 | 28.017 | 57.9% | 14.85 | 67.300–67.359 |

### MTP verify, M = 4

Each call processes four rows. These rates omit drafting and acceptance;
they are not end-to-end speculative generation rates.

| depth | KV | wall ms | kernel sum ms | attention ms | other ms | attention share | rows/s | wall range ms |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 4096 | bf16 | 33.395 | 32.970 | 1.468 | 31.502 | 4.5% | 119.78 | 33.386–33.473 |
| 4096 | int8 | 33.969 | 33.487 | 1.913 | 31.573 | 5.7% | 117.76 | 33.959–33.985 |
| 32768 | bf16 | 38.279 | 37.703 | 6.396 | 31.305 | 17.0% | 104.50 | 38.264–38.294 |
| 32768 | int8 | 40.151 | 39.453 | 8.066 | 31.387 | 20.4% | 99.62 | 40.146–40.156 |
| 81920 | bf16 | 45.237 | 44.776 | 13.541 | 31.232 | 30.2% | 88.42 | 45.236–45.259 |
| 81920 | int8 | 49.659 | 48.764 | 17.398 | 31.368 | 35.7% | 80.55 | 49.655–49.664 |
| 150000 | bf16 | 55.115 | 54.485 | 23.274 | 31.207 | 42.7% | 72.58 | 55.111–55.124 |
| 150000 | int8 | 62.139 | 61.475 | 30.114 | 31.361 | 49.0% | 64.37 | 62.132–62.164 |

Head KV preparation at 150k contributes 4.079 us with bf16 and 10.238 us with
int8, included in attention above; the verify head has no attention scan.

### Paired cache-form comparison

These are medians of the three within-round int8/bf16 wall ratios. A value
above 1 means int8 took longer. KV forms alternate as whole four-depth engine
loads, separated by loading and prefill, rather than adjacent per-depth calls.
Warm-up and the whole-box lock reduce drift; the ranges below report the
observed variation and do not remove possible systematic clock bias.

| depth | M = 1 median [range] | M = 4 median [range] |
|---:|:---|:---|
| 4096 | 1.0361 [1.0344, 1.0373] | 1.0169 [1.0148, 1.0180] |
| 32768 | 1.1313 [1.1312, 1.1323] | 1.0490 [1.0484, 1.0493] |
| 81920 | 1.2590 [1.2576, 1.2598] | 1.0978 [1.0971, 1.0979] |
| 150000 | 1.3635 [1.3629, 1.3655] | 1.1275 [1.1273, 1.1277] |

The largest within-arm wall spread is 0.262% across all depths/forms/M values.
Int8's smaller cache does not currently make either measured list faster.

### Verdict

**GO:** both 150k plain attention shares exceed the operator's 25% threshold.
The bf16 plain remainder stays between 28.248 and 28.329 ms over all depths
and rounds, while attention rises from 0.916 to 20.426 ms. The measured
long-context cost increase is in attention, not the constant-size remainder.
The int8 remainder similarly stays within 27.942–28.237 ms, while attention
rises from 1.959 to 38.460 ms. The remainder is an aggregate of other kernels,
not a hardware-counter measurement of weight traffic alone.

Verify does not grow faster than its four rows: at 150k, its attention costs
1.139x the plain attention at bf16 and 0.782x at int8, well below 4x. The
existing DPAS selection already amortizes attention across verify rows.
The plain-step threshold independently justifies the follow-up.

Proposed 3f: probe packed int8 V/scale reads and exact native finite-fp16
widening first, preserving the current arithmetic; compare bitwise outputs
and quiet-box whole-step timings at these four depths before any integration.
Then decide whether changing the existing sequence partition is worthwhile.
This is a plan, not a measured speedup or a change of the default KV form.

## Kernel inspection for a possible follow-up

The plain bf16 scan already splits the sequence over at most 32 blocks per
KV head; split-K would be retuning an existing partition, not introducing one.
Its K path uses subgroup block reads and its V path uses 2D block reads
(`src/kernels/attn_v2.cl`, `load_k` / `load_v`). The plain int8 K path also
uses block reads, but `src/kernels/kv8.cl::load_v8` issues scalar V and scale
loads for each of 16 positions and converts fp16 scales with `kv8_f16f`.
This suggests packed V reads and exact native finite-scale widening as a
first probe. Keep the multiply locations and reduction order fixed; use
bitwise attention/output checks before evaluating whole-step wall gains.
Changing the sequence partition would change reduction association and needs
its own numerical judgment. M = 4 already uses the DPAS verify path at these
depths, so merely enabling that existing selector is not a new optimization.
These are source-based hypotheses; no decode kernel is modified in 3c.

## 3f: packed int8 decode reads — 2026-11-09

Production code `8a5ccb1c` replaces the int8 v2 reader's scalar V loads with
packed 16-bit 2D reads. Each subgroup lane selects its signed byte from a
loaded pair; overlapping blocks stay within a head-bounded surface. One
scale gather per lane feeds subgroup broadcasts for the sixteen positions.
K keeps its existing byte block reads. Both readers widen finite fp16 scales
with `vload_half`, preserving the original fp32 value exactly. The scale
multiply locations, FP32 dot/FMA/softmax/reduction order, 32-way sequence
partition, persistent KV representation and automatic DPAS selection remain
the same. There is no new engine flag, environment variable or scratch buffer.

### Selection and correctness

The acceptance rule was fixed in `338c74dc` before measuring: at least 5%
lower whole-step wall time at 81920 and 150000, and no regression above 2%
at 4k/32k or M = 4, with bitwise equality. The native-scale-only prototype
improved the long M = 1 attention pair about 1.074x; adding packed V reads
reached about 1.66x. These are single-layer probe ratios; whole-step results
below determine acceptance.

`kv8_decode_bitwise_test` compares actual production kernels against a frozen
copy of the reader from `800783f2`. All raw partial and output words match in
140 case executions: M = 1..4 and batched M = 8, short/tail boundaries through
150015, mixed slot offsets and inactive rows. All 63,488 finite fp16 codes,
including signed zeros and subnormals, widen bitwise; output/partial guards
remain intact. Deliberately zeroing V fails the comparison. An unchanged
reader compared against itself fails the required performance gain, as expected.

### Matched whole-step measurements

Three before/after pairs use the same model, int8 head, capacity 150272,
physical card 1 and whole-box lock. Each arm/phase has 128 warm-up calls,
128 unprofiled wall calls and 32 profiled calls. Round 1 reverses build order;
the probe also reverses M = 1 / M = 4 order by round and phase. Each executable
loads the same real populated prefixes through the existing engine snapshot
APIs, with model/prompt/capacity/form/size metadata checks.

The first setup pair mixed cold prefill with restoration. Review caught the
extra lazy slab/head allocations; both logs were archived under
`~/b70/logs/3f/cold-setup` and the pair was repeated. Every accepted run now
restores all four prefixes and reports identical 24.935 GB engine buffers:
17.700 GB model, 4.963 GB KV, 0.957 GB decode state, 1.315 GB eager prefill
scratch. The comparison rejects incomplete runs, duplicates, cold prefixes
or unequal memory reports before checking the fixed performance rule.

All 48 accepted measurements validate. Times are medians of three round
means; gains are medians of paired wall-time reductions. Ranges describe
observed paired variation, not confidence intervals. Attention is the
profiled sum including prep/scan/reduce; wall is uninstrumented.

| depth | M | old wall ms | new wall ms | paired gain % | gain range % | old attention ms | new attention ms | new rows/s |
|---:|---:|---:|---:|---:|:---|---:|---:|---:|
| 4096 | 1 | 30.643 | 29.869 | 2.53 | 2.53–2.54 | 1.962 | 1.167 | 33.48 |
| 4096 | 4 | 33.939 | 33.967 | -0.14 | -0.14–-0.02 | 1.905 | 1.909 | 117.76 |
| 32768 | 1 | 38.026 | 34.740 | 8.63 | 8.59–8.64 | 9.444 | 5.940 | 28.79 |
| 32768 | 4 | 40.132 | 40.147 | -0.06 | -0.10–-0.01 | 8.069 | 8.051 | 99.63 |
| 81920 | 1 | 50.634 | 42.830 | 15.36 | 15.30–15.42 | 21.779 | 13.883 | 23.35 |
| 81920 | 4 | 49.595 | 49.589 | -0.03 | -0.06–0.02 | 17.374 | 17.395 | 80.66 |
| 150000 | 1 | 67.320 | 53.789 | 20.12 | 20.10–20.14 | 38.463 | 24.794 | 18.59 |
| 150000 | 4 | 62.162 | 62.164 | -0.03 | -0.05–0.03 | 30.126 | 30.149 | 64.35 |

The M = 4 production path already uses DPAS at these depths and remains
effectively unchanged. Rows/s for verify still excludes drafting/acceptance.
Both long plain-decode gains clear 5%; every regression check stays within 2%.
Raw logs: `~/b70/logs/3f/{before,after}-r{0,1,2}.log`, `comparison.{md,json}`,
`production-bitwise.log` and `production-negative.log`.

The named prefill regressions pass 17/17, including both snapshot forms and
MTP prefill; all four l0 golden printouts match the baseline byte for byte
after masking only numeric elapsed-time fields. The default int8 MTP
verification registration initially skips on two contexts, identically on
the immutable baseline:
`mtp_verify_qwen3.8_int8_n1998_m2a_7015c7519decf848` and
`mtp_verify_qwen3.8_int8_n2053_m2a_24d74a876aefabe2`. Its twin registration
does not copy the normal test's judge environment. Both exact references
already exist in `oracle-out-qwen-f32-forced`; rerunning with that existing
`B70_FORCED_JUDGE` directory and the existing bf16-top references **passes**
(132.70 s), with numerical thresholds unchanged.

The ten int8 checkpoint registrations complete with eight passes and two
initial skips; the judged rerun resolves one skip. Slots, replay, engine
smoke, both decode goldens, memory planning and both prefix-cache forms
pass. The long MTP GPU gate reports zero failing runs and passes its other
checks, but **skips** on 27 divergences in nine contexts. Its required
`oracle-out-qwen-f32-pfi8-forced` references are absent. The normal f32
references cannot substitute for this gate's int8 prompt arithmetic. This
remaining skip is not counted as a pass. Logs are under `~/b70/logs/3f`:
`checkpoint-regressions.log`, `baseline-mtp-verify.log` and
`mtp-verify-judged.log`. The operator stopped additional non-checkpoint validation after integration.
The merge-gate build and command-line checks pass (1451 variants, one addition,
zero changes/removals); no complete 310-test pass is claimed. The final SLM
build also passes all 17 named prefill tests and all four byte-compared goldens.

## 3h: int8 decode attention after the reader fix - 2026-11-10

Branch `longctx-attn-3h`. The acceptance was fixed and committed (`a0e955bf`) before any
decode timing: int8 plain attention at 150k <= 0.6 x bf16's (12.26 ms against 3c's 20.43),
whole-step plain wall gain >= 5 % at 81920 / 150000, no regression above 2 % at 4096 /
32768 plain or any M = 4 row, outputs bitwise today's reader. Bandwidth estimate written
first: a plain int8 step at 150k reads 4.95 GB (16 layers x 150001 x 2064 bytes), 8.4 ms at
the card's 590 GB/s; bf16's 9.83 GB is 16.7 ms and its measured 20.43 ms is 82 % of that.

### Step 0: short prefill chunks deep in a session

`kv8_flash_bitwise_test --chunk-time 128000` times one chunk's attention over 16 FA layers
at pos 128000 (random int8 K / V and scales), interleaved after warm-up, 9 rounds each, all
three paths bitwise on every shape: the frozen original int8 flash (`kv8_flash_control.cl`),
today's in-place int8 flash (`pf_flash_attn_kv8`, reads int8 directly) and today's
production pair (`kv8_widen` over pos + C rows, then the raw flash).

| C | original ms | in-place ms | widen + raw ms | in-place / widen + raw |
|---:|---:|---:|---:|---:|
| 1 | 3514.2 | 454.6 | 426.5 | 1.066 |
| 8 | 3526.2 | 455.5 | 426.6 | 1.068 |
| 16 | 3516.5 | 454.7 | 428.5 | 1.061 |
| 32 | 3505.3 | 455.0 | 425.6 | 1.069 |
| 64 | 3500.9 | 478.7 | 423.5 | 1.130 |
| 128 | 3897.8 | 510.1 | 425.7 | 1.198 |
| 300 | 4183.2 | 618.9 | 488.1 | 1.288 |
| 512 | 7743.1 | 1181.0 | 853.9 | 1.385 |
| 1024 | 15582.7 | 2395.7 | 1616.9 | 1.474 |
| 2048 | 27982.0 | 4528.3 | 2842.1 | 1.596 |

The widening pair is fastest at every chunk size, so **no chunk-size switch was built**:
re-widening the prefix costs ~1.4 ms a layer at 128k, less than the in-place reader's
per-tile conversions even at C = 1. The real cost of a short turn is elsewhere: below
~128 rows the flash's grid is ceil(C / 8) x 4 work-groups, each walking all 128k keys, so
any 1..128-row chunk at 128k costs ~0.43 s of attention (bf16 has the same kernel shape).
A sequence split for small chunks would change the online softmax's association (an
f32-judged change) - recorded as an open lever, not built.

### Profile (probe_kv8_decode, today's reader)

`tools/probe/probe_kv8_decode` times 16 layers' launches over 16 separate random caches
(incompressible: the card compresses constant memory), arms interleaved, 11 rounds.
`kv8_decode_ablate.cl` is today's reader with one part removed each (timing only).
At 150k, M = 1 (ratios to today's reader, same run): bf16 0.815; no K loads 1.058; no V
loads 1.047; no K / V loads 1.051; no scale loads 0.997; no K / V / scale loads 0.991; no
score dots 0.980; no softmax / V accumulation 0.752; the 32-way reduce alone 0.0054
(0.13 ms for 16 layers). Achieved 205 GB/s against 590. **The reader was compute-bound**:
removing every load changed nothing, while the per-sub-group softmax - repeated by all 16
sub-groups of a work-group - and the byte unpacking were the time.

### What changed (src/kernels/kv8.cl, KV8_DEC only)

1. **Owner sub-groups.** Each (q head, row) pair's wave max, weights, exp and sum are
   computed once per work-group by its owner sub-group (pr % 16) and passed through SLM
   (one more barrier a wave); every sub-group then only accumulates its V dims.
2. **V bytes without shuffles.** Sub-group pairs read the same packed 16-bit block and take
   its low / high byte, so a work-item's dim is a fixed permutation (`v8_dim`) and the 3f
   reader's per-element index shuffle and select are gone (a row past `last` is +0 from the
   2D read's bound and a zero scale).
3. **One-fma dequant.** x s = fma(as_float(0x4B400000 + x), s, -12582912 s), exact (only a
   zero product's sign can differ, and every such product is added to a +0 or nonzero sum).
4. **SLM dot tree at M = 1.** The score's 16 lane partials go to SLM; the owner sums them in
   the shuffle tree's order (stride 8, 4, 2, 1). Slower at M = 4, so M = 1 only.
5. **Cache prefetch two waves ahead.** Sub-groups 0..7 prefetch the K / V rows and 8, 9 the
   scales of wave w + 2 (distance 2 best of 2 / 3 / 4 / 6; the scale prefetch alone ~5 %).

Rejected on measurement: two waves per barrier step (1.07x slower), deferring the
conversions behind the next loads (1.13x slower: the compiler serialised registers),
256-GRF mode and register-resident q (1.46x / 1.21x slower: occupancy).
Kernel-level at 150k, M = 1, same run: today's reader 26.51 ms, new 11.55 ms (0.436),
bf16 20.74 ms (new / bf16 0.557). The partition, the blocks, waves, value orders and the
persistent format are unchanged; no flag, environment variable or buffer was added.

### Correctness

`kv8_decode_bitwise_test`: **308 cases PASS** - every partial and output word against the
frozen reader at M = 1..4, the row builds at M = 2..8 (two slots), tails to 150015, inactive
rows, all 63,488 finite fp16 scales including -0 and subnormals. Focused tests on card 0,
13 run: 11 pass (slots_kv8, kv8_kernels, kv8_flash_bitwise, kv8_flash_raw_bitwise,
kv8_scale, kv8_decode_bitwise, replay_determinism_kv8, golden_gate_kv8, prefill_gate_l0_kv8,
prefill_determinism_l0_kv8, prefill_consistency_l0_kv8); mtp_verify_kv8 skips by default as
on 3f's baseline and **passes** with 3f's judge directories (133 s); mtp_gpu_kv8 skips (3f's
missing pfi8 references; not counted as a pass). Logs: `~/b70/logs/3h/`.

### Whole-step before / after

The 3c / 3f probe (`probe_longctx_decode`), physical card 1, whole-box lock per run, capacity
150272, int8 head, 128 warm-up / 128 wall / 32 profiled calls, prefixes restored from
`~/b70/probes/3h/prefix-cache` (int8 copied from 3f's, bf16 made by an untimed setup run).
Base = main `4b714dbe` built in its own tree; new = this branch; bf16 from the new build (its
code is unchanged). Three rounds, arm order base / new / bf16, bf16 / new / base,
new / base / bf16. Walls and attention (prep + scan + reduce, profiled) are medians of the
three rounds; gain is the median paired wall reduction with its range; new / bf16 is the
median per-round attention ratio. A CPU reference job of another agent ran throughout.

| depth | M | base int8 wall | new int8 wall | gain % [range] | base int8 attn | new int8 attn | bf16 attn | new / bf16 attn | bf16 wall |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|
| 4096 | 1 | 30.407 | 29.942 | 1.54 [1.22, 2.33] | 1.153 | 0.684 | 0.915 | 0.745 | 30.119 |
| 4096 | 4 | 34.864 | 34.736 | 0.27 [-0.16, 2.06] | 1.882 | 1.880 | 1.451 | 1.296 | 34.107 |
| 32768 | 1 | 34.814 | 31.805 | 9.17 [8.07, 9.92] | 5.917 | 2.718 | 4.838 | 0.561 | 34.088 |
| 32768 | 4 | 40.702 | 40.845 | -0.65 [-1.69, 1.59] | 8.014 | 8.006 | 6.379 | 1.255 | 38.566 |
| 81920 | 1 | 43.104 | 35.829 | 17.43 [16.54, 17.44] | 13.832 | 6.134 | 11.368 | 0.540 | 40.898 |
| 81920 | 4 | 49.999 | 49.979 | 0.01 [-0.18, 0.39] | 17.268 | 17.206 | 13.512 | 1.274 | 46.347 |
| 150000 | 1 | 54.152 | 40.943 | 24.39 [24.36, 24.54] | 24.656 | 10.918 | 20.304 | 0.538 | 49.782 |
| 150000 | 4 | 63.010 | 62.898 | -0.06 [-0.11, 0.24] | 29.842 | 29.772 | 23.191 | 1.284 | 55.697 |

**Every bar is met.** Int8 plain attention at 150k is 10.92 ms, 0.538 x bf16's (bar 0.6;
the bandwidth estimate's ~11.5 ms); plain decode gains 24.4 % at 150k and 17.4 % at 81920,
and int8 plain steps are now faster than bf16 from 32k on (40.94 vs 49.78 ms at 150k).
4096 plain gains 1.5 %; M = 4 runs the unchanged DPAS verify kernel from 4096 keys on and
stays within its noise (worst paired round -1.69 %, medians -0.65..+0.27 %).

Open, not built: (1) int8 MTP verify at M = 4 still costs ~1.28 x bf16's attention (the
DPAS int8 kernel); the same owner / prefetch ideas, or v2 at M = 2..4 (now ~0.56 x the 3f v2
at M = 4 in the kernel probe), could be re-timed against `verify_attn.h`'s table. (2) The
row builds (13c slots, M = 2..8) carry the same reader changes and are bitwise, but were
not timed. (3) Step 0's short-chunk prefill attention (~0.43 s per chunk at 128k).
Logs: `~/b70/logs/3h/matrix/{base-int8,new-int8,new-bf16}-r{0,1,2}.log`, `comparison.md`;
kernel probes `~/b70/logs/3h/{profile-M1,profile-M4,snaps-*}.log`; step 0
`~/b70/logs/3h/step0-chunk-time.log`.

## 3l (1): MTP verify attention at long context - 2026-11-10

Branch `verify-attn-3l`. With `--spec auto` most serving decode steps are MTP verify steps at
M = 4, and from 4096 keys row 0 they run `attn_verify_dpas[_kv8]` (src/kernels/attn_verify_dpas.cl),
which 3h left untouched; at 150k its int8 form costs ~1.28 x bf16's attention.

### Acceptance (fixed before any timing)

- **Gain:** the whole verify step (probe_longctx_decode's uninstrumented wall, median paired
  reduction of three rounds, base = main `865da900` built in its own tree) is **>= 5 % faster at
  M = 4, and at M = 2 and 3, at depths 81920 and 150000, for both bf16 and int8 KV**.
- **No regression:** nothing else more than 2 % slower - plain decode M = 1 at every depth,
  every M at 4096 / 32768, and the 13c row builds (`attn_v2_M<M>_T32[_KV8]_ROWS`: they never
  bind the verify kernel; checked by their unchanged command lines and source).
- **Bitwise:** every partial word (mx, sm, acc[256]) and every reduced output word of the
  changed verify kernel equals a frozen copy of today's (`tests/kernels/attn_verify_control.cl`)
  at M = 2..8 (bf16) and 2..4 (int8), tails, inactive rows, finite fp16 scales incl. -0 and
  subnormals. A switch-depth change in `src/runtime/verify_attn.h` is made only where measured,
  and only if each kernel's output stays what it computes today at that depth; a change that
  swaps v2 and DPAS at a depth changes rounding (fp32 v2 against the DPAS hi + lo products) and
  is reported for the f32 judge instead of being made.
