# The first device-side `--pp 4096`, and where its time actually goes

Measured 2026-09-05 on the box (B70, `ZE_AFFINITY_MASK=1`, RTN checkpoint
`~/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`, commit `a663f7b`).
**Iterate grade throughout**: two desktop processes (baobab, ptyxis) hold DRM
fds on every card, so the "provably idle box" a record row needs is not
available. Every number below is labelled measured / derived / estimated and no
number appears twice with two values.

Pre-registration: `docs/prefill-l1-engine-preregistration-2026-09-05.md`,
committed at `571e6d5`, before any of this was run.

---

## 1. The row

```
| b70-decode a663f7b pp | 4096 | 2048 | 4187.9 | 978.07 |
| b70-decode a663f7b    | 4096 |    8 |   32.89 |  30.40 |
```

`tools/bench_decode.sh --pp 4096 --tg 8 --runs 8`, medians over **8
independent runs**:

| row | median | min | max | spread |
|---|---:|---:|---:|---:|
| **pp 4096, C = 2048** | **978.07 t/s** (4187.9 ms) | 974.16 | 979.61 | 5.45 (**0.56%**) |
| tg 8 at depth 4096 | 32.89 t/s (30.40 ms/token) | 32.87 | 32.92 | 0.05 (0.15%) |

A note on the grade's own convention: the project's iterate grade is "8
replays, drop 3, median". The **drop-3** half does not apply to a `--pp` row
and is not silently claimed - each run is a separate process with a cold
16 GB load, so there is no warm-up sequence inside a process to drop. What is
reported is the median of all 8, with the min, max and spread beside it. The
0.56% spread says the instrument is not the uncertainty here.

**The decode row is untouched and slightly better than the record**: 32.89 t/s
against docs/BENCHMARKS.md's 32.22 for this checkpoint. Prefill changes no
decode kernel and no captured launch.

## 2. Scored against the pre-registration: a 1.84× MISS

| | pre-registered | measured | |
|---|---:|---:|---|
| `--pp 4096`, C = 2048 | 2.28-2.75 s, point estimate 2.40 s | **4.188 s** | **MISS, 1.75× the point estimate** |
| | 1490-1800 t/s, point estimate 1707 | **978.07 t/s** | **MISS, 1.84× under the pre-registered 1800** |

Nothing about the miss is ambiguous, because the walk was instrumented and
every term of the ledger's composition was checked individually. **Five of the
six terms are confirmed. One is wrong by 64×.**

## 3. The per-phase attribution

`B70_PREFILL_PROFILE=1` times every `Context::wait()` in the walk and adds one
extra wait to close each L0-only phase. The walk is already a strict
alternation of "append launches" / "drain both queues" - rulings A23/A24 put a
host wait at every L0↔SYCL boundary because this device has one compute queue
and no cross-runtime dependency - so **the wall time of a wait IS the device
time of everything queued since the previous drain**, and no event pool is
needed. The instrument's own cost is reported, not argued: the profiled total
is 4119.1 ms against 4202.8 ms plain in the same binary, i.e. **−2.0%**, which
is inside the run-to-run spread; the extra waits cost nothing measurable.

`--pp 4096`, C = 2048, so **two chunks** - depth 2048 then depth 4096
(measured, iterate):

| phase | ms (2 chunks) | share | waits | ms/wait |
|---|---:|---:|---:|---:|
| norm (`pf_res_fold` + `pf_norm_finish`) | 91.4 | 2.2% | 257 | 0.356 |
| dequant (`pf_dequant_tile`) | 410.0 | 10.0% | 512 | 0.801 |
| **gemm (`gemm_bf16`, the 4 int4 linears)** | **1312.5** | **31.9%** | 512 | 2.564 |
| `pf_ab_proj` | 44.6 | 1.1% | 96 | 0.465 |
| `pf_gdn_seed` | 0.5 | 0.0% | 96 | 0.006 |
| `pf_gdn_conv` | 121.6 | 3.0% | 96 | 1.267 |
| `pf_gdn_l2norm` | 13.5 | 0.3% | 96 | 0.140 |
| `pf_gdn_gate` | 2.0 | 0.0% | 96 | 0.021 |
| `pf_gdn_A` | 53.3 | 1.3% | 96 | 0.555 |
| `pf_gdn_solve` | 62.7 | 1.5% | 96 | 0.654 |
| `pf_gdn_wu` | 179.2 | 4.3% | 96 | 1.866 |
| `pf_gdn_A2` | 88.6 | 2.2% | 96 | 0.923 |
| **`pf_gdn_scan`** | **1446.8** | **35.1%** | 96 | **15.071** |
| `pf_gated_head` | 27.8 | 0.7% | 96 | 0.289 |
| `pf_silu_mul` | 98.3 | 2.4% | 128 | 0.768 |
| `pf_attn_prep_q16` | 12.6 | 0.3% | 32 | 0.393 |
| attention QKᵀ (batched GEMM) | 80.1 | 1.9% | 128 | 0.626 |
| `pf_softmax_causal` | 56.0 | 1.4% | 128 | 0.438 |
| attention PV (batched GEMM) | 7.7 | 0.2% | 32 | 0.242 |
| `pf_attn_gate` | 9.1 | 0.2% | 32 | 0.283 |
| `step_head` | 0.7 | 0.0% | 1 | 0.667 |
| **TOTAL** | **4119.1** | 100% | 2818 | |

## 4. Five terms confirmed, one falsified

Per chunk (the table above ÷ 2), against the ledger's composition
(progress.md, "THE TWO LEVERS, MEASURED" and "THE LAST TWO LEVERS"):

| term | ledger (standalone probes) | measured in situ | |
|---|---:|---:|---|
| GEMM | 680.1 | **656.3** | −3.5%, **confirmed** |
| dequant | 210.116 | **205.0** | −2.4%, **confirmed** |
| small kernels (norm + silu + ab_proj + gated_head) | 130.346 | **131.1** | +0.5%, **confirmed** |
| composed attention | 83.4-88.9 (tuned, plan 6d) | **82.8** (untuned, this stage, averaged over the two depths) | **confirmed** |
| `step_head` / interop | ~8 | **0.35** | better |
| **GDN (`gdn_chunk` less its gated head)** | **15.4** | **984.1** | **64× MISS** |

**The ledger's 15.4 ms GDN term was never a measurement.** It comes from the
Stage-0 T6 composition (progress.md:293), which priced GDN by projecting
decode's `gdn_step` widened to M = 2048 - written before `gdn_chunk` existed.
L1-core's brief was correctness and it measured a numerics band, not a time.
Every ceiling derived since - 1043-1107, 1874, 1905, 1790-1815, and finally
**1808.5-1817.3 t/s = "92% of vLLM, every term measured"** - carries that one
unmeasured term. It is measured now.

**Sub-finding, and it is one kernel: `pf_gdn_scan` is 1446.8 ms of the 1968.2
ms GDN total - 35.1% of the entire prefill, 15.07 ms per GDN layer per chunk.**

Mechanism, read off the kernel rather than guessed
(`src/kernels/prefill/pf_gdn_scan.cl`): it is **decode's `gdn_step` tile
mapping, unchanged and deliberately so** (the file says so, and spec §6.3's
numerics band is measured against exactly that choice). Grid (48 heads, 4
state-column chunks) = **192 work-groups**, each walking `nch = C/64 = 32`
sub-chunks serially - which the recurrence requires. But **inside** a
sub-chunk, stages 1 and 2 are *not* serial in `i`, and the kernel runs them as
64 sequential 256-lane tree reductions each, five barriers per tree, with only
`sgid == 0` (16 of 256 work-items) doing the epilogue. That is **640 barriers
per sub-chunk, 20,480 per work-group per layer**, and 15/16 of the lanes idle
in every epilogue. Arithmetically the scan does ~9.66 GFLOP per layer per chunk
(3 stages × 64 positions × 128 × 128 MACs × 48 heads × 32 sub-chunks, derived),
so 15.07 ms/layer is **0.64 TFLOP/s** - 0.35% of the card's demonstrated
183.5 TFLOP/s bf16 peak, on a kernel that carries ~2% of the forward's FLOPs.

This is the **same class of finding Stage 0 already made once** ("our small
kernels are ~40× off their own roofline… decode kernels widened naively for
prefill"), on the one kernel family that stage did not reach.

## 5. What the ceiling actually is, with every term now measured

Per chunk at C = 2048, depth 4096, from the table above:

| composition | ms/chunk | t/s | vs vLLM 1973 |
|---|---:|---:|---|
| as built (this stage, untuned attention) | 2059.6 | **994.4** | 50.4% |
| + plan 6d's tuned attention (83.4-88.9) | 2060-2066 | 991-994 | ~50% |
| + `pf_gdn_scan` at 10% of its measured cost (derived, **not built**) | ~1409 | ~1454 | 74% |
| + GDN at the ledger's assumed 15.4 ms (derived, **not built**) | ~1091 | ~1877 | 95% |

The `--pp 4096` row's 978.07 t/s is slightly under the 994.4 t/s per-chunk
figure because a 4096-token prefill also pays a one-time warm-up: single-chunk
runs measure chunk 1 (depth 2048) at 2156.1 ms and the two-chunk run puts
chunk 2 (depth 4096) at 2031.8 ms - the *deeper* chunk is the cheaper one, by
123 ms, which is module loading and sycl-tla's first-launch cost landing on
whichever chunk runs first (**~170 ms one-time, derived**, after allowing the
~47 ms the extra attention depth should have added).

**RULING REQUEST, not a decision taken here.** `pf_gdn_scan` is A22's kernel,
delivered by the L1-core stage against a correctness brief, and rewriting it
changes both reduction orders and therefore the state band
`gdn_chunk_test` records. The no-tuning rule and "one defect, one fix, one
measurement" both say this is a new task with its own pre-registration, not
something to be adjusted until the number improves. What is offered here is the
attribution and the price, and the controller rules.

## 6. Fixed and per-position costs, measured

Single-chunk runs, `--pp N --pp-chunk N` (measured, one run each, iterate):

| C | ms | 
|---:|---:|
| 256 | 589.4 |
| 512 | 766.8 |
| 1024 | 1283.8 |
| 2048 | 2156.1 |

A two-point fit on 1024/2048 gives **411.5 ms fixed per chunk** and **0.852
ms per position** (derived); the 256/512 pair gives 412 ms and 0.693 ms/pos,
the difference being that a single-chunk run's attention is quadratic in C.
The fixed 411 ms against the dequant's measured 205 ms is the rest of the
per-chunk overhead - module loads on the first chunk, and the chunk-independent
part of every launch.

The practical consequence for `--pp-chunk`: at C = 256 the fixed term is 70% of
the chunk, which is why `--pp 256 --pp-chunk 64` reads 161.76 t/s and
`--pp 2048 --pp-chunk 2048` reads 949.87. **C = 2048 (ruling A13) is the right
default and this measurement agrees with it.**

## 7. What this stage delivers, plainly

4096 ids in **4.19 s** where the decode-replay ingest takes **121 s** (bench log
`2a7df0b`, RTN, 29.5 ms/id) - a **28.9× cut**, measured. The spec's headline
53× and its 92%-of-vLLM ceiling both rest on the 15.4 ms GDN term, which this
measurement retires.
