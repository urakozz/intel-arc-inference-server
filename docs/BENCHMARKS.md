# Benchmarks

This file is the evidence behind the two numbers the README quotes: prefill
2104.50 t/s (the int8 prefill linears of spec 5; 1670.72 on the bf16 walk)
against vLLM's 1610.04, and decode 29.45 t/s against vLLM's 31.01.
It records how each row was taken, what grade it earned, and the ladder of
measurements that got there.

No row is quoted without its grade, and no number here is extrapolated. Figures
are labelled **measured**, **derived** (arithmetic on measured inputs, with the
inputs shown) or **estimated** (a stated guess with its reasoning).

## How a row is graded

| grade | what it means |
|---|---|
| **RECORD** | Median of three runs, box provably idle: zero containers and zero processes holding a DRM file descriptor on any card, verified before and after the run set. |
| **near-idle** | Median of three, no containers, but some process held a DRM fd. Those processes submitted no work, which the tight spreads corroborate, but corroboration is not proof. |
| **iterate** | Single runs, or a loaded box, or an instrumented binary. Useful for attribution, never for a headline. |
| **diagnostic** | A probe, not a series row. Its own harness, its own protocol. |

The harness measures the idle condition itself and prints the grade it earned.
A grade is quoted as printed and never upgraded by hand.

Two further rules this file holds to:

- **Median of three, with the spread printed.** Record triples on this
  instrument sit between 0.00% and 0.15% spread.
- **Day-scale drift of this instrument is at or under 0.09%**, measured on a
  same-checkpoint control (see "The instrument" below). Anything larger than
  about 0.1% between two idle medians needs a cause other than drift.

## What is being compared

Both engines serve the same files. That is what makes the comparison fair, and
it is the single most load-bearing fact in this document.

- **Checkpoint**: `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, snapshot
  `84575a18f209992ef96d819b31f924b489e3d55d`. GPTQ g64 symmetric, int4 weights,
  **bf16 `lm_head`**.
- **Shape**: 4096-token prompt, 256 generated tokens, one sequence, chunk width
  2048 on both sides, no prefix caching, no speculative decoding.
- **Bytes read per token**: 15.540 GB, reported by our own loader. At the
  measured 590 GB/s that is a roofline of 26.34 ms/token, **37.97 t/s**
  (derived). MBU figures in this file divide by that 15.540 GB.

### Why the comparison is byte-matched, and why that costs us the better number

This engine can load an int4 `lm_head`; vLLM cannot, and reads the bf16 tensor
instead. On a checkpoint with an int4 head we read 13.673 GB/token rather than
15.540, and on 2026-09-04 that engine measured **32.22 t/s against vLLM's
31.01**, the first decode row this project ever led on.

That lead is not quoted anywhere, because it is not a property of the engine.
vLLM would be faster with an int4 head too; it simply cannot load one, so the
margin measured an inability of the comparison rather than a difference in
kernels. On identical weight bytes vLLM was ahead by 5.7% in the same session.
From 2026-09-13 onward this project quotes only the byte-matched line: **bf16
`lm_head` on both sides.** The int4-head checkpoint was deleted from the box
and has not been re-quantised to restore the flattering row.

Prefill does not care either way. `lm_head` runs once per prompt, and the two
checkpoints measured **0.17% apart** on prefill where they were 9.9% apart on
decode (2026-09-05, same binary, same hour).

## vLLM: how the baseline is served and benched

```bash
IMAGE=vllm-xpu-env-next-p314-t215-vxkp0
MODEL=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

docker run --rm -it \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 16k --kv-cache-dtype auto --max-num-seqs 2 \
  --language-model-only --trust-remote-code \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'
```

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

**Read the vLLM version, never the image tag.** A tag gets rebuilt. One tag in
this project's history carried two builds 149 commits apart under the same
name. Every vLLM row below names the version string the server printed.

The current matched prefill baseline is vLLM `0.29.1rc1.dev380`, measured
directly rather than through `llama-benchy`'s `pp` figure: **2544.043 ms for
4096 tokens, 1610.04 t/s**. The decode baseline is **31.01 t/s** on vLLM
`0.28.1rc1.dev396`.

### Why `llama-benchy`'s own `pp` figure is unusable against our server

Not a defect, a structural mismatch. `llama-benchy` derives
`pp_throughput = prompt_tokens / (ttfr - latency)`, and `ttfr` is the arrival
time of the first SSE frame carrying `choices` at all. `b70-serve` writes the
role-delta frame before `generate()` starts, which is correct, spec-compliant,
anti-buffering behaviour that `protocol_test.cc` asserts on directly. So `ttfr`
measures time-to-role-frame, not time-to-first-token, and the derived `pp`
comes out in the millions of tokens per second.

`tg_throughput` is unaffected: it comes from streamed content-chunk timestamps
and a token count, not from `ttfr`. Prefill over HTTP is measured directly
instead, with a non-streaming `/v1/completions` request, `max_tokens: 1`, over
a prompt our own tokenizer counts as exactly 4096 tokens.

## Prefill

All rows: device 0, `tools/bench_decode.sh --pp 4096`, chunk width 2048, median
of three, same checkpoint.

| date | what changed | pp t/s | ms total | spread | grade |
|---|---|---:|---:|---:|---|
| 2026-09-05 | first prefill kernel: `Engine::prefill` replaces replaying the decode list per prompt id | 1377.20 | 2974.2 | 0.29% | iterate |
| 2026-09-09 | two GDN tasks land, box provably idle for the first time | 1406.18 | 2912.9 | 0.50% | **RECORD** |
| 2026-09-18 | prefill moves to the Level Zero backend: our own DPAS GEMM and slab dequant on one in-order list | **1502.83** | 2725.5 | 0.31% | **RECORD** |
| 2026-09-19 | re-measured after an IGC 2.38.2 to 2.41.5 upgrade, all 139 kernel binaries recompiled | 1498.97 | 2732.5 | 0.43% | RECORD |
| 2026-09-22 | causal QK^T: the attention kernel stops computing the masked half | 1505.16 | 2721.3 | 0.09% | RECORD |
| 2026-09-22 | SiLU folded into the gate‖up GEMM epilogue | 1560.40 | 2625.0 | 0.58% | RECORD |
| 2026-09-23 | split-BF16 GDN scan becomes the default | 1670.72 | 2451.6 | 0.10% | RECORD |
| 2026-09-24 | int8 prefill linears (spec 5 h8) become the default, `l0-int8` | 2104.50 | 1946.3 | 0.22% | RECORD, idle checked before the passes |
| **2026-09-25** | **fused flash attention (spec 6) becomes the default; interleaved with the composed path's 2105.02 (below)** | **2125.12** | **1927.4** | **0.07%** | **RECORD** |
| | vLLM `0.29.1rc1.dev380`, matched | 1610.04 | 2544.043 | | measured, external |

**2104.50 is 130.7% of the matched vLLM row** (derived; vLLM was measured
2026-09-20 and not re-run); spec 6's flash attention took it to 2125.12, 132.0%
(derived; its own section below). The bf16 walk's record, 1670.72, was 103.8% of it.
The arc from the last row before the kernel work started to here is 1498.97 to
2104.50, **+40.4%**.

### The int8 prefill linears (spec 5), the default since 2026-09-24

`--pp-backend l0-int8` (spec 5, plans 5a/5b) runs every int4 linear of the L0
walk on the rotated int8 ("h8") path: a Hadamard-rotating per-token int8
quantiser, then per 1024-column slab a rotated per-channel int8 requant and an
i8 x i8 DPAS GEMM. Attention, GDN, norms and the head are the L0 backend's.
**It is the default** after the operator's 2026-09-24 ruling on gate A3's one
bf16 near-tie (spec 5 §8); `--pp-backend l0` keeps the bf16 walk.

**The row that counts (2026-09-24, after the scale pass moved to load):**
`Engine::prepare_prefill()` builds the rotated column scales when the CLI
loads the model, so the timed first prefill no longer carries them. Device 0,
box idle at the start (no render-node holder, no container), four interleaved
passes of `tools/bench_decode.sh --pp 4096 --pp-backend <b> --runs 3`:

| pass | backend | runs, pp t/s | median | ms (median run) |
|---|---|---|---:|---:|
| 1 | l0 | 1646.74, 1643.52, 1641.66 | 1643.52 | 2492.2 |
| 2 | l0-int8 | 2105.84, 2104.50, 2101.27 | **2104.50** | 1946.3 |
| 3 | l0 | 1640.85, 1639.56, 1639.36 | 1639.56 | 2498.2 |
| 4 | l0-int8 | 2102.64, 2103.03, 2100.61 | **2102.64** | 1948.0 |

Paired ratios int8 / l0: **1.2805** and **1.2824** (derived). **B1 passes**
(bar 1.20x). l0 itself ran 1.6 % under its 1670.72 record in this session,
the first after a reboot; the paired ratio is the comparison that holds.

The earlier measurement below is kept: it is what the one-time scale pass
cost when it still ran inside the first request.

2026-09-24, sha `04ce00c`, device 0, `tools/bench_decode.sh --pp 4096
--pp-backend <b> --runs 3`, taken as four interleaved passes (l0, int8, l0,
int8). No process held a render node; a CPU-only container of another job was
running, so the script graded every pass **ITERATE**.

| pass | backend | runs, pp t/s | median | ms total | spread | tg t/s (median) |
|---|---|---|---:|---:|---:|---:|
| 1 | l0 | 1671.78, 1672.51, 1671.48 | 1671.78 | 2450.1 | 0.06% | 29.44 |
| 2 | l0-int8 | 1970.89, 1980.31, 1962.02 | **1970.89** | 2078.2 | 0.93% | 29.46 |
| 3 | l0 | 1668.74, 1667.01, 1669.68 | 1668.74 | 2454.5 | 0.16% | 29.38 |
| 4 | l0-int8 | 1952.93, 1971.09, 1972.82 | **1971.09** | 2078.0 | 1.01% | 29.47 |

Paired ratios int8 / l0: **1.1789** (pass 2 / 1) and **1.1812** (pass 4 / 3).
Against the 1670.72 record, 1971.09 is **1.1798x** (derived). Spec 5's bar B1
is 1.20x, 2005 t/s: **B1 fails as measured, by about 34 t/s.**

What the row contains. `b70-decode --bench --pp` times the **first** prefill
of a fresh process, and on l0-int8 the first prefill also builds every int4
linear's rotated column scales (256 `pf_colmax_rot` passes, each with a host
wait, run before the first chunk so no recorded list holds one). Timed on a
fresh engine, three 4096-id prefills each (`prefill_int8_test --first-cost`,
host wall around `Engine::prefill`, two processes per backend, interleaved):

| backend | call 0 ms | call 1 ms | call 2 ms | call 0 minus mean(1, 2) |
|---|---:|---:|---:|---:|
| l0 | 2455.7 / 2452.7 | 2477.2 / 2484.1 | 2491.4 / 2482.2 | -28.6 / -30.5 |
| l0-int8 | 2072.1 / 2075.9 | 1932.9 / 1936.0 | 1931.5 / 1929.1 | **+139.9 / +143.3** |

So a user's first l0-int8 request pays **about 140 ms** once (about 170 ms
against l0's own first-call behaviour, derived), and the bench row carries it.
A later prefill in the same process runs at 1930 ms, 2122 t/s, 1.27x the
record (derived from host wall, not a bench row). Moving the scale pass to
load time, where spec 5 §4 T2 put it, is what would take it out of the first
request; that is not done here.

Decode (B2): `tools/bench_decode.sh --runs 3` right after the four passes gave
**29.40 t/s** (29.39 to 29.40), and the tg rows of the pp passes 29.38 to
29.47, against the 29.45 record: within the day-scale drift this file records,
and by construction, since the decode list is byte-identical on every prefill
backend. **B2 passes.**

### The Level Zero backend, and what it was worth

Before 2026-09-18 the prefill GEMM was sycl-tla's, called from SYCL, with a
host wait per slab. After, it is our own OpenCL C DPAS GEMM launched on the
same in-order Level Zero list every other prefill kernel already ran on.
Both backends were measured in the same session on an idle box, and sycl-tla
stays selectable so the control can be re-run at any time.

| | Level Zero | sycl-tla control |
|---|---:|---:|
| pp t/s | **1502.83** | 1407.63 |
| launches per chunk | 8689 | 1201 |
| host waits per chunk | **0** | 656 |
| SYCL GEMM calls | **0** | 384 |
| prefill scratch | 35,651,584 B | 356,515,840 B |

The control reproduced the previous standing row to 0.10%, so the **+6.76%**
margin is the backend and not the session.

Phase attribution from the same pair, instrumented (iterate grade by
construction: the extra waits are the instrument), one run each:

| phase | Level Zero ms | share | sycl-tla ms | share |
|---|---:|---:|---:|---:|
| `dequant` | 0.0 | 0.0% | 413.3 | 14.3% |
| `gemm` (sycl-tla) | 0.0 | 0.0% | 1403.2 | 48.7% |
| `linear_l0` | **1668.9** | **61.2%** | 0.0 | 0.0% |
| `attn_QK^T` | 84.6 | 3.1% | 83.0 | 2.9% |
| `attn_softmax` | 60.5 | 2.2% | 60.6 | 2.1% |
| `attn_PV` | 8.3 | 0.3% | 8.3 | 0.3% |
| `gdn_scan` | 328.8 | 12.1% | 330.8 | 11.5% |
| `norm` | 98.4 | 3.6% | 98.7 | 3.4% |
| `silu` | 99.6 | 3.6% | 98.5 | 3.4% |
| **total** | **2728.1** | | **2881.1** | |

The linears are the whole of the win: `linear_l0`'s 1668.9 ms replaces
`dequant` plus `gemm`'s 1816.5 ms, **8.1% less**, and 512 host waits per run
disappear with them. Every other phase moves by under 2 ms. The worry that 8,689
launches would cost host time in situ does not appear at all: the Level Zero
walk has **fewer** host waits than the SYCL one, not more.

The IGC upgrade the next day changed nothing measurable: the Level Zero row
moved 0.26% and the sycl-tla control 0.16%, both inside the runs' own spreads,
and correctness held (the GEMM was bitwise equal to sycl-tla on all 31 test
cells, the dequant bit-exact, the whole suite green).

### The split-BF16 scan, the one approximation in the default path

The 2026-09-23 row is **the first in this file whose arithmetic is not bitwise
equal to the row above it.** The delta net scan now runs an approximate
split-BF16 kernel, gated on tokens and state cosines rather than on `memcmp`.
It is worth 2.33x on its own profile row: `gdn_scan` falls from 297.5 and
297.7 ms under the vector entry to 127.1 and 128.2 ms under the split entry,
profiled ABBA with 96 launches each, about 170 ms of GPU time, of which 173.4 ms
reached the wall.

Both golden gates report 93/93 determined rows with state cosines of
0.999912369 (prose), 0.999712545 (code) and 0.999901750 (cjk) against a bar of
0.999, and the suite is 86/86 in a single run.

**Every gate run printed its own dispatch proof** (`gdn scan entry LAUNCHED:
pf_gdn_scan_dpas_split`). An earlier attempt on 2026-09-21 did not, and that
run measured the vector kernel while believing it had measured this one. See
"The dispatch lesson" below.

One honest limit: `vn` inside the A2 term is still a single BF16 operand,
measured harmless on three prompts of one checkpoint, not proven in general.
Details in [prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md).

### Chunk width

Monotone in the chunk width across the whole available range, measured
2026-09-09, one run each, context rather than a gate:

| chunk | pp t/s | ms / 4096 | vs 2048 |
|---:|---:|---:|---|
| 2048 (the default) | 1406.18 | 2912.9 | |
| 1024 | 1221.90 | 3352.2 | -13.1% |
| 512 | 932.37 | 4393.1 | -33.7% |

That is what a fixed cost per chunk predicts: halving the width doubles the
chunk count and therefore the fixed term. 2048 is the widest this build runs.

### What prefill replaced

Before there was a prefill kernel, a prompt cost one decode replay per id:
**121 s for 4096 ids**. The same prompt now costs **2.45 s**, about **49x**.
That, and not the vLLM comparison, is what the prefill work set out to do.

### Prefill over HTTP costs nothing measurable

Measured 2026-09-14, same session, same device, both windows first-token
inclusive:

| path | ms for 4096 tokens | t/s |
|---|---:|---:|
| HTTP, `/v1/completions`, median of 3 | 2915.83 | 1404.74 |
| CLI, `tools/bench_decode.sh`, RECORD | 2919.6 | 1402.91 |

The derived HTTP cost is **-3.77 ms per 4096 tokens**, statistically
indistinguishable from zero against the 0.5 to 3 ms run-to-run spread both
sides show. Tokenizer encode plus HTTP framing cost nothing on top of the
device-side prefill.

## Flash attention and 128k (spec 6)

Spec 6 (plans 6a and 6b, 2026-09-25): prefill attention is `pf_flash_attn`, one
fused bf16 DPAS kernel per FA layer (plan 6a's winning tile, `pfa_KT64_R16_H6_Q0`,
docs/probe-flash-attn-2026-09-25.md), in place of the composed QK^T GEMM, causal
softmax and PV GEMM. The composed path stays selectable with
`B70_PREFILL_ATTN=composed`, and its score scratch (`pf_s` / `pf_p`, which would be
9.66 GB at 128k, derived) is only allocated when it runs. `--max-len 131072` works in
both CLIs. Everything below is device 0 on an idle box, `l0-int8`, measured unless
marked derived.

**pp4096, interleaved composed / flash / composed / flash, median of 3 each:**

| run | composed t/s | flash t/s | flash / composed (derived) |
|---|---:|---:|---:|
| round 1 | 2105.02 (1945.8 ms) | 2125.12 (1927.4 ms) | 1.0095 |
| round 2 | 2103.76 (1947.0 ms) | 2125.30 (1927.3 ms) | 1.0102 |

All four are RECORD grade, spreads 0.07 to 0.16%. Flash is the new pp4096 record,
**2125.12 t/s** (the first flash row), 132.0% of the matched vLLM row (derived).
The plan's gate was no regression (flash >= 0.99x); it is 1.01x.

**The attention phase (F1), one profiled pp4096 each, L0 GPU ms (diagnostic):**
`attn_flash` 116.3 ms, against the composed path's QK^T 43.4 + softmax 57.4 + PV
31.4 = 132.2 ms in the same session. Spec 6's F1 target was 80 ms (vLLM's flash
kernel: 63.1 ms); it is missed and recorded, per the operator's ruling A of
2026-09-25 (integrate now, optimise later).

**Prefill at depth, `--pp N --max-len 131072`, median of 3:**

| ids | t/s | ms total | spread |
|---:|---:|---:|---:|
| 32768 | 1499.37 | 21854.5 | 0.22% |
| 65536 | 1121.74 | 58423.5 | 0.02% |
| 130816 | 746.57 | 175223.3 | 0.02% |

At 65536 ids (one profiled run) `attn_flash` is 29852 ms of 58239 ms of L0 GPU
time, **51.3%**, which is 28.3 TFLOP/s on its causal FLOP count, 15.4% of the
183.45 TFLOP/s bf16 peak (derived). Spec 6's F2 asked for 60% of peak; missed and
recorded under ruling A. The follow-up is a split-d / SLM-staged kernel.

**Decode (F4): the 128k variants cost nothing at shallow depth.** Depth 4096, tg
256, interleaved twice: max_len 16384 29.27 and 29.27 t/s, max_len 131072 29.28 and
29.28 t/s, ratio 1.0003 (derived; the bar was within 2%). The captured list still
launches `max_len / 64` attention blocks per head group every token and the idle
ones cost nothing measurable, so the indirect grid spec 6 §3.3 held in reserve was
not built.

**Decode at depth (F3), `--pp N --max-len 131072`, tg 256, median of 3:**

| depth | t/s | ms/token | bandwidth-derived rate | share |
|---:|---:|---:|---:|---:|
| 4096 | 29.28 | 34.15 | (the basis) | |
| 32768 | 20.60 | 48.54 | 26.17 | 78.7% |
| 65536 | 15.46 | 64.68 | 23.34 | 66.2% |
| 130816 | 10.19 | 98.13 | 19.20 | 53.1% |

The derived rate is the bandwidth decode achieves at depth 4096, 29.28 t/s x
(15.540 GB of weights + 4096 x 64 KiB of KV) = 462.9 GB/s, divided by the bytes one
token reads at the row's depth. Spec 6's F3 bar was 90%; missed and recorded:
decode attention, not the weights, is what falls behind at depth. (The deepest row
is 130816 so that 256 generated tokens fit in 131072.)

**Memory at 128k**, the line both CLIs now print before the first prefill:
`memory: model 18.116 GB, kv 8.590 GB, decode state 0.590 GB, prefill scratch
0.700 GB, int8 0.085 GB, total 28.081 GB of 32.530 GB`.

**Correctness.** Kernel against fp64: worst cosine 0.999998 on every case, at or
above the composed path's own. Every golden gate passes in both modes, and the
composed path is byte-identical to before in its gate output. At 32k (K3a, as the
operator redefined it on 2026-09-26) flash is no further from the CPU oracle than
composed on either backend. Mean logit cosine against the oracle over the golden set's 96
decision rows: `l0` flash 0.999960061 vs composed 0.999959889, `l0-int8` flash 0.999934564
vs composed 0.999931742 (`flash_vs_oracle_test`). Flash and composed measured directly
against each other at 32704 ids is information now, not a bar: logits cosine 0.999662 on
`l0-int8` with greedy tokens splitting at step 10 of 64, about 0.99988 on `l0`. A CPU
oracle at 32k itself was tried and is day-class (11359 s to reach 8192 ids). At 128k: two fresh engines on a 131000-id
prompt give bitwise-equal, finite last logits, prefill replay reproduces them
bitwise twice, and passkey retrieval at 119939 ids finds the key at 5, 50 and 95%
depth, 3/3 on `l0-int8` and 3/3 on `l0`.

### Spec 6c: 8 rows per work-group (2026-09-27)

`pf_flash_attn` now runs 8 query rows per work-group (6 sub-groups, grid `(ceil(C / 8), 4,
1)`), research lever 2 of docs/research-flash-prefill-2026-09-27.md. Lever 1, `exp2` with
the scale folded, was first held (`EXP2=0`: it misses spec 6 K3a on `l0` by 7.6e-6) and
is **on** since the operator's ruling of 2026-09-27 (spec 6 §9): arm E below is what ships. Interleaved C / R / E triples after one warm-up each, device 0, `l0-int8`,
`--max-len 131072` for the depth rows; C is main's build in the base tree (its prefill
and decode sources are main's; it differs only in `src/server`), R and E are this
branch's binary with the kernel file swapped. **Diagnostic grade: the load average was
4.5 to 19.9 (other agents' CPU jobs) during every row**, so no RECORD row is written even
where R beats 2125.12.

| row | C: main (RPW 16, `exp`) | R: spec 6c as built (RPW 8, `exp`) | E: RPW 8 + `exp2` (shipped) |
|---|---:|---:|---:|
| pp4096 t/s (2048 chunks, tg 256; r1 / r2 / r3) | 2129.41 / 2128.51 / 2128.81 | 2139.94 / 2136.68 / 2135.01 | 2160.57 / 2162.25 / 2152.75 |
| pp4096 t/s, median (x C, derived) | 2128.81 | 2136.68 (1.0037) | 2160.57 (1.0149) |
| `attn_flash` ms per pp4096 (profiled, p1 / p2) | 117.7 / 117.2 | 101.4 / 100.6 | 81.9 / 80.5 |
| pp65536 t/s, median of 3 (x C) | 1123.47 | 1206.97 (1.0743) | 1301.26 (1.1582) |
| pp130816 t/s, median of 3 (x C) | 748.88 | 821.37 (1.0968) | 915.33 (1.2223) |
| decode at 4k (tg 256 after pp4096), median t/s | 29.25 | 29.24 | 29.24 |

Load averages (1 min) at the rows: pp4096 5.2-12.3, profile 5.1-11.5, pp65536 5.3-19.9,
pp130816 4.7-17.3. Spreads within an arm are <= 0.45%. F1 (80 ms): R 101 ms, missed;
E would meet it (80.5-81.9 ms). RPW 8 alone does not change the numerics: K3a's cosines
are bit-identical to main's (`l0` 0.999960061, `l0-int8` 0.999934564).

## Prefix caching (spec 7)

Spec 7 (plans 7a-7c, 2026-09-27/28): `b70-serve --prefix-cache-gb 32` (the default)
continues the resident session or restores the deepest host snapshot and prefills only the
tail. Device 0, `l0-int8`, max_len 131072. Records: docs/probe-prefix-cache-2026-09-27.md.

- **P0**: pinned host copies at 14.2 GB/s device to host, 12.2-12.9 GB/s host to device;
  48 GiB pinned allocation fine; tail floor 316 ms at depth 0, 505 ms at 60k (16 ids).
- **S3** write-through: +2.03% on cold pp4096, +1.24% on pp32768 (bar 3%).
- **S1** continuation, 60000 ids of history + a 1000-id tail, time to first token:
  **1233.4 ms** (bar 1500; cold prefill of the history alone: 45.2 s). Load 8.6-11.9.
- **S2** the same after a 300-id side request: restore **347.7 ms** (3933 MB of KV + the
  state from host, bar 1000), time to first token 1581.2 ms.
- **Replay** of a synthetic opencode-shaped session (15 requests, 2 side), cache on vs off:
  outputs bitwise equal, time to first token per request:

| request | 1 | 2 | 3 | 4s | 5 | 6 | 7 | 8 | 9 | 10s | 11 | 12 | 13 | 14 | 15 | sum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt ids | 1703 | 1792 | 3435 | 111 | 4982 | 5108 | 7076 | 7931 | 8040 | 98 | 8166 | 8251 | 8336 | 10153 | 12026 | |
| on, s | 0.95 | 0.95 | 1.85 | 0.41 | 1.66 | 0.67 | 1.72 | 1.08 | 1.13 | 0.41 | 1.19 | 1.53 | 0.49 | 1.18 | 2.27 | 17.46 |
| off, s | 0.91 | 0.91 | 1.80 | 0.38 | 2.54 | 2.56 | 3.60 | 3.99 | 4.03 | 0.38 | 4.06 | 4.42 | 4.46 | 5.16 | 6.23 | 45.44 |

  The real recorded opencode session (C3) is pending; rerun with
  `tools/prefix/replay_ab.sh tests/golden/opencode/session1 ~/c3`.

## Decode

All rows: device 0, depth 4096, 256 generated tokens, median of three on an
idle box unless the grade says otherwise.

| date | what changed | t/s | ms/token | MBU | grade |
|---|---|---:|---:|---:|---|
| 2026-08-25 | phase 1, before any lever | 23.73 | 42.14 | 62.5% | record |
| 2026-08-25 | GEMV K-split: the `a‖b` GEMV splits its K 16 ways inside the work-group, 8 hardware threads per launch to 128 | 24.83 | 40.27 | 65.4% | record |
| 2026-08-25 | residual norm split: `prep_res_norm` ran 129 times per token on one work-group; it becomes two launches on 20 work-groups each | 26.28 | 38.05 | 69.2% | record |
| 2026-08-25 | attention block width 256 to 64, quartering the serial KV walk per work-group | 27.53 | 36.32 | 72.5% | record |
| 2026-08-26 | re-measured on an idle box the next day | 27.52 | 36.34 | 72.5% | record |
| 2026-09-04 | auto-round loader, byte-matched checkpoint | 29.33 | 34.09 | 77.2% | near-idle |
| 2026-09-09 | re-gate on the current checkpoint, box provably idle | 29.32 | 34.11 | | **RECORD** |
| 2026-09-23 | unmoved by every prefill change since | **29.45** | | | RECORD |
| | vLLM `0.28.1rc1.dev396`, no speculation | **31.01** | 32.25 | 81.7% | measured, external |

**Decode trails vLLM by about 5%, on the same bytes.** That has been true since
2026-09-09 and none of the prefill work moved it, which is by construction:
prefill selects the backend and the replayed decode list is byte-identical
either way. Decode replays 774 kernels and 19 modules per token with frozen
arguments and no host decisions in the loop.

The three 2026-08-25 levers each landed within about 0.1 ms of its predicted
bench delta, with the residual drift on untouched launches priced in the
arithmetic rather than absorbed into a rounding. The attention lever is the one
that did not close: predicted -2.045 ms, measured -1.73 ms, **0.32 ms apart**,
and the difference is recorded as unattributed rather than explained away.

### The one thing on the decode path that was measured three ways and did not close

The int4 `lm_head` lever was priced in situ at **3.203 ms**, at iterate-grade
bench at **3.16 ms**, and at record-grade bench at **3.05 ms**. The in-situ
figure reconciles against the bytes to a microsecond: the head reads
2,542,796,800 B at bf16 and 675,430,400 B at int4, a difference of 1.8674 GB,
which is 3.165 ms at the measured 590 GB/s; the bf16 launch ran at 98.4% of
device and the int4 launch at 97.2%, and 3.165 + 0.071 - 0.033 = 3.203 exactly.

The record bench says 3.05. The direction of the 0.15 ms discrepancy is not
resolvable from what was measured, and the run that would settle it was never
taken. The honest statement is that this lever is known to **±0.15 ms**, and
every percentage quoted from it carries its grade.

The lever is historical: this project no longer quotes the int4-head
comparison, for the reason given at the top of this file.

### MBU moves the opposite way from throughput, and that is not a contradiction

On our own two 2026-09-04 rows, MBU is **77.2%** on the slower byte-matched row
and **74.7%** on the faster int4-head row. It was predicted before it was
measured: the bf16 `lm_head` was the most bandwidth-efficient launch in the
step, at 98.4% of device, so deleting three quarters of its bytes lowers the
average efficiency of what remains while raising throughput. Faster and less
efficient at once, on one binary.

vLLM's MBU on the same 15.540 GB is **81.7%**, still the best number any
software has posted on this silicon here.

### Host sampling

Measured 2026-09-14, server-side generation-only timing, three greedy against
three sampled 256-token requests:

| | greedy ms/token | sampled ms/token |
|---|---:|---:|
| median of 3 | 31.7766 | 32.3140 |

**Delta 0.537 ms/token**, against a bar of 0.62 ms registered before the
measurement. Cross-validated by curl's own round-trip differential over the
same requests, which cancels the identical prefill cost on both sides:
0.528 ms/token. The two independent methods agree to within 0.01 ms. Sampling
ships on by default; greedy is unaffected by construction, since the sampler
only runs when sampling is not greedy.

A first attempt at this read **0.7408 ms/token**, over the bar, and was a
measurement artifact rather than a code cost: each request ran over its own
freshly connected ssh session, and the handshake time preceding the sampled
batch landed inside the measurement window unevenly. The clean measurement
reissues the identical eight requests inside one persistent session with curl
looped on the box. Discarded, not reconciled: the 0.74 ms figure measured a
network artifact.

## Correctness

Numerics are gated, not eyeballed. The full protocol is in
[14-golden-gate.md](14-golden-gate.md); what the benchmark rows rest on:

| gate | what it proves |
|---|---|
| `golden_gate_test` | Decode ingest, one id per replay, against a CPU oracle: **93/93 determined rows exact**, 3 undetermined and all inside the oracle's own argmax set. |
| `prefill_gate_test` | One `Engine::prefill` per prompt, same oracle, same tally. |
| `prefill_backend_equivalence_test` | The Level Zero and sycl-tla prefill paths agree **bitwise**: 0 words differing, 0 sign-of-zero differences, on all five case families. |
| `replay_determinism_test` | 774 kernels, 19 modules captured; 8 tokens by 3 runs bitwise identical from reset. |
| `prefill_replay_test` | Exact bytes of GDN state, convolution ring, both KV caches, control block and logits, across two full chunks plus a ragged tail. |

Suite total: **86 tests**, all passing.

**One divergence is itself a cross-check.** On the `cjk` prompt the decode path
emits token 4960 at generated position 22 where the oracle emits 271, and the
oracle's own logits mark that row undetermined: 4960 is in its argmax set. The
prefill path takes 271 at the same row. That position is exactly where
`b70-serve` and `b70-decode --ids` diverged in September before `--prefill`
existed, a divergence that blocked a gate for five days. It sits on a row the
CPU reference cannot decide either, and the reading that it is a sub-ulp tie
between two independently rounded paths is confirmed by the oracle.

The tensor diagnostics mark a run of low tap cosines on the `code` prompt, down
to about 0.989 at layers 52 to 57, while all 32 of its tokens are exact. Tokens
gate, tensors diagnose; that is recorded, not failed.

## The dispatch lesson

On 2026-09-21 a record was claimed for a GDN scan kernel that had passed its
gates. It had not run. The selector resolved, the gates ran, everything was
green, and the kernel the run actually launched was the old one.

Every gate run now prints the entry it launched, read back from the pointer the
launcher handed the kernel cache, and the non-band tests assert that the printed
entry matches the selector's resolution:

```
gdn scan selector: B70_PREFILL_GDN_SCAN=dpas_split -> entry pf_gdn_scan_dpas_split
gdn scan entry LAUNCHED: pf_gdn_scan_dpas_split
```

Run with its dispatch proven, that same kernel **failed**: 92 of 93 rows, code
prompt L60 state cosine 0.996344994. The cause was one operand of one term, and
the fix took it to 93/93 at 0.999712545. The whole episode is in
[prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md).

**A green gate is not evidence unless the run says which kernel it dispatched.**
The probes in this repo carry the same rule: the W4A8 probe proves twice, once
by asking the driver the name of the kernel handle the launcher received and
once by having the kernel write a signature word from the device.

## Work that was measured and rejected

Negative results with a mechanism, kept because they saved more time than the
wins did.

| what | why it was rejected |
|---|---|
| **Fusing the int4 dequant into the GEMM** | Both variants came out **bitwise identical** to the two-pass path over all 71,303,168 outputs, and both were slower. The dequant pass already runs at about 546 GB/s, roughly 90% of hardware peak, so there is only 0.826 ms in it at this shape; moving the same arithmetic into the mainloop costs 2.759 ms even when issued exactly once per weight. [Details](probe-fused-dequant-2026-09-22.md). |
| **W4A8 on the mixed 4-bit by 8-bit DPAS** | 1.017x the bf16 two-pass control, 0.965x once the activation quantiser it needs is counted, against a bar of 1.53x. The per-group rescale costs 3.0 to 3.4 of the 5.440 ms; the matrix math it rescales costs 1.99 ms. Separately, **2.79% relative L2 error** against the bf16 result, with a worst-row cosine of 0.9976, driven by activation outliers. [Details](probe-w4a8-2026-09-23.md). |
| **Removing barriers from the GDN triangular solve** | The design assumed synchronisation dominated that kernel. Deleting 64 of 128 barriers bought 1.7 ms of a 73.5 ms row, so the premise was wrong by about **18x**. Both barrier-free variants were bit-identical and both were slower. The row is the serialised recurrence itself. [Details](prefill-gdn-solve-register-results.md). |
| **bf16 intermediate buffers** | Halving the bytes changed nothing: `ocloc` exposes no 16-bit block write wider than 8 rows, so the epilogue is bound by store message count rather than by bytes. |
| **oneDNN's fused int4 W4A16 matmul** | Measured 98.28 TFLOP/s on gate‖up at M=2048, below the 115.7 TFLOP/s an own in-kernel int4 GEMM would need to beat the two-pass path. Also failed a bitwise identity extraction against our dequant oracle on 39.6% of weights, every mismatch a 1-ULP bf16 neighbour. [Details](probe-prefill-vllm-parity-2026-09-14.md). |

## The instrument

### Day-scale drift is at or under 0.09%

Two rows one day apart, same checkpoint, same shape, same harness, both idle,
both medians of three:

| | 2026-08-25 | 2026-08-26 | drift |
|---|---|---|---|
| decode | 27.54 t/s / 36.32 ms | 27.52 t/s / 36.34 ms | **+0.055%** |
| ingest, 4096 ids | 34.44 ms/token | 34.47 ms/token | **+0.087%** |

That is a same-checkpoint control, so it carries nothing but the day. Earlier
drift figures of +0.30% to +0.56% quoted for individual decode levers were
measured **across** a kernel change and therefore carry the change's own noise.

### A CPU compile does not contend with decode

The byte-matched checkpoint read 36.27 ms/token under a 12-core compile against
its idle median of 36.32, **0.14% apart**, because the decode step is 99.7%
inside the GPU fence. That is evidence rather than an assumption, and it is why
a loaded box downgrades a row's grade without invalidating it.

### The two cards are not interchangeable, and only prefill sees it

Same binary, same checkpoint, same ten minutes, measured 2026-09-05:

| quantity | device 0 | device 1 | delta |
|---|---:|---:|---:|
| pp4096 t/s | 1377.20 | 1330.82 | **-3.37%** |
| tg256 t/s | 32.37 | 32.31 | -0.19% |

The separation is far outside the spreads and reproduces on two checkpoints:
prefill loses 3.2% to 3.4% on device 1, decode 0.2% to 0.4%. That is a
difference in sustained compute rather than in bandwidth, seen by the
compute-bound workload and not by the bandwidth-bound one. Unattributed:
neither card drives a display, both report identical unprivileged PCIe link
fields, and the frequency sysfs needs root.

**Every series row in this file is device 0.** Per-kernel probes may run on
device 1 and say so.

### The bench commands

```bash
tools/bench_decode.sh                        # 3 runs at depth 4096, tg 256
tools/bench_decode.sh --pp 4096              # the prefill row
tools/bench_decode.sh --pp 4096 --tg 256 --runs 3
ZE_AFFINITY_MASK=1 tools/bench_decode.sh ... # names the card explicitly
```

The prefill window spans the first prefill launch to the first generated id
present in the control block, loader excluded. The decode window is the
generation phase only. vLLM's figures are HTTP-inclusive where they come from
`llama-benchy`, and that label travels with them.

## Notes

- **Torch 2.14 enables Inductor's `batch_linear_lhs` pre-grad fusion on XPU**,
  which concatenates the GDN projection weights at runtime and cost vLLM a
  large fraction of its decode rate on every checkpoint we tried it on. Passing
  `--compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'`
  restores it. That flag is in the serve command above for exactly this reason.
- **Torch 2.14 and later can return non-contiguous GDN projections.** A local
  patch materializes only the actually-used rows before calling the SYCL kernel.
- **Prefill is consistently higher on prebuilt-kernel vLLM images than on
  source-built ones.** Observed repeatedly, not chased.
- **`--enable-prefix-caching` is silently inert when MTP speculative decoding is
  on** in the vLLM configs we tried: no KV-cache group is identified as the
  draft's, every group is treated as a draft group, and the server log says
  prefix-cache reuse will be disabled. Worth knowing before trusting a cached
  row.
- The oracle's greedy tokens for the three golden prompts, 32 ids each, are the
  gate this engine must reproduce exactly. Dumping them follows
  `model.safetensors.index.json` so the oracle reads the same shard set the
  engine loads, rather than a stale set sitting beside it.
