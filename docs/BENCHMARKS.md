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

## vLLM: the README's comparison

The README's "Against vLLM" table. vLLM numbers are the operator's (vLLM XPU image `p314-t215-vxkp0` /
`p314-t214-vxkp0`, one B70, TP 1, prefix caching on, `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`),
benched with `llama-benchy --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation`.
Ours, the same llama-benchy command through b70-serve: Ornith from the replica record (one request = one card's
engine, mean of 6 runs); Qwen3.8 on one card (`--parallel 1 --max-num-seqs 1`, 3 runs each): `--prefill-backend
l0-int8` pp 2,470 / tg with MTP 62.3; the default bf16 prompt pass pp 1,647 / tg with MTP 67.9 (decode with MTP is
the mean of those 6 runs, 65.1: the prompt pass does not change decode); `--spec off` pp 1,658 / tg 33.6 (all three
runs 33.6). The bf16 prompt pass is below vLLM's prefill; the README's row names the int8 one.

| model, one B70, 1 request | ours | vLLM |
|---|---:|---:|
| Ornith 1.5 35B-A3B, pp4096 | 8,877 | 7,723.17 ± 46.08 (MTP 2) |
| Ornith 1.5 35B-A3B, tg256, MTP | 180.5 | 115.77 ± 4.34 (MTP 2) |
| Qwen3.8-27B, pp4096 | 2,470 (l0-int8) / 1,647 (bf16) | 1,973 (no spec) / 1,918 (MTP 2) |
| Qwen3.8-27B, tg256, MTP | 65.1 | 45.23 ± 2.68 (MTP 2) |
| Qwen3.8-27B, tg256, no speculation | 33.6 | 31.50 |

## vLLM: how the baseline is served and benched

```bash
IMAGE=vllm-xpu-env-next-p314-t216-vxkp0
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

### `llama-benchy`'s own `pp` figure against our server: fixed 2026-09-27

`llama-benchy` derives `pp_throughput = prompt_tokens / (ttfr - latency)`, and `ttfr` is the
arrival time of the first SSE frame carrying `choices` at all. Until 0b59fda (2026-09-27)
`b70-serve` wrote the role-delta frame before `generate()` started, so `ttfr` measured
time-to-role-frame and the derived `pp` came out in the millions of tokens per second. Since
0b59fda the role frame goes out with the first delta, after prefill, as vLLM sends it, and
`llama-benchy`'s `pp` reads the same way for both servers. `tg_throughput` was never affected.
The matched prefill baseline above (1610.04 t/s) predates the fix and was measured directly: a
non-streaming `/v1/completions` request, `max_tokens: 1`, over a prompt our own tokenizer counts
as exactly 4096 tokens.

### llama-benchy comparisons (queue row 41)

One client, one set of flags, one card, so these rows compare with tables others publish. They are **HTTP-inclusive** and sit beside, never replace, the
engine-internal records in the sections below. Runbook `r41.ours`
(box-day Session 6c, idle box, device 0); server `tools/probe/serve_benchy.sh`; llama-benchy pinned at 0.4.0 (`BENCHY_VERSION`); Qwen3.8 int4 g64
(`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`) for every arm.

| matrix | llama-benchy flags |
|---|---|
| A | `--pp 512 4096 --tg 128 --concurrency 1 2 4 8 --runs 2 --format md` (the usual published form; a repeated prompt may hit a server's prefix cache) |
| B | A + `--no-cache` (the cache-free reading) |
| D | `--pp 512 --tg 128 --depth 4096 32768 --concurrency 1 --runs 2 --no-cache --format md` (agentic depth) |

| arm | server | A | B | D |
|---|---|---|---|---|
| b70-serve `--mtp 0` | int8 head, bf16 KV, `--max-len 40960` | pending | pending | pending |
| b70-serve `--mtp auto` | the same, MTP head on | pending | pending | pending |

b70-serve has no batched decode yet (spec 13b / 13c): its concurrency 2 / 4 / 8 cells are requests
served one after another, recorded as the baseline batching has to beat.

## Prefill

All rows: device 0, `tools/bench_decode.sh --prefill-length 4096`, chunk width 2048, median
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

`--prefill-backend l0-int8` (spec 5, plans 5a/5b) runs every int4 linear of the L0
walk on the rotated int8 ("h8") path: a Hadamard-rotating per-token int8
quantiser, then per 1024-column slab a rotated per-channel int8 requant and an
i8 x i8 DPAS GEMM. Attention, GDN, norms and the head are the L0 backend's.
**It is the default** after the operator's 2026-09-24 ruling on gate A3's one
bf16 near-tie (spec 5 §8); `--prefill-backend l0` keeps the bf16 walk.

**The row that counts (2026-09-24, after the scale pass moved to load):**
`Engine::prepare_prefill()` builds the rotated column scales when the CLI
loads the model, so the timed first prefill no longer carries them. Device 0,
box idle at the start (no render-node holder, no container), four interleaved
passes of `tools/bench_decode.sh --prefill-length 4096 --prefill-backend <b> --runs 3`:

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

2026-09-24, sha `fe20b9b`, device 0, `tools/bench_decode.sh --prefill-length 4096
--prefill-backend <b> --runs 3`, taken as four interleaved passes (l0, int8, l0,
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

What the row contains. `b70-decode --bench --prefill-length` times the **first** prefill
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

#### 2026-11-13: int8 in the MLP only (open-work item 17)

On the dense model `l0-int8` now runs only gate‖up and down int8; the four mixer linears take the bf16 slab walk
(they carried the long-prompt miss, docs/17-int8-prefill.md). `b70-decode --bench --prefill-length N --tg 32
--max-len 32768 --parallel 1`, Qwen3.8, one card (card 1; card 0 ran a test throughout: busy box), main (cf22d86a)
against branch `int8-mlp-only`, three rounds with the order changed each round, median:

| prompt | arm | t/s (3 rounds) | median | launches |
|---|---|---|---:|---:|
| 4096 | main `l0` | 1738.95, 1737.69, 1736.11 | 1737.69 | 16903 |
| 4096 | branch `l0` | 1744.73, 1739.40, 1739.89 | 1739.89 | 16903 |
| 4096 | main `l0-int8` | 2260.87, 2256.32, 2254.74 | 2256.32 | 17415 |
| 4096 | branch `l0-int8` | 2082.52, 2083.80, 2080.13 | **2082.52** | 17159 |
| 16384 | main `l0` | 1571.54, 1566.46, 1567.84 | 1567.84 | 67597 |
| 16384 | branch `l0` | 1569.51, 1568.69, 1566.28 | 1568.69 | 67597 |
| 16384 | main `l0-int8` | 1980.57, 1980.51, 1977.38 | 1980.51 | 69645 |
| 16384 | branch `l0-int8` | 1844.60, 1844.74, 1844.60 | **1844.60** | 68621 |

`l0-int8` against `l0`: 1.20x / 1.18x (main's all-int8 form 1.30x / 1.26x); against main's form 0.923x / 0.931x.
`l0` is the same build either side (its logs bitwise main's). Logs on the box: `~/b70/logs/int8-mlp-only/speed1.log`.

Tensor parallel, `--parallel tp --prefill-backend l0-int8`, the same flags, whole-box lock, main and branch
alternating, median of 3: pp4096 main 3326.71 (3321.90, 3330.55, 3326.71), branch **3216.84** (3216.84, 3211.37,
3220.44), 0.967x; pp16384 main 3051.82 (3051.36, 3053.30, 3051.82), branch **2937.33** (2940.09, 2933.36, 2937.33),
0.962x. Launches 22158 -> 21646 / 88602 -> 86554 (both cards). Log `~/b70/logs/int8-mlp-only/box1.log`.

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
Details in [12-kernels.md, "How the prefill kernels got here"](12-kernels.md#how-the-prefill-kernels-got-here).

### Chunk width

Monotone in the chunk width across the whole available range, measured
2026-09-09, one run each, context rather than a gate:

| chunk | pp t/s | ms / 4096 | vs 2048 |
|---:|---:|---:|---|
| 2048 (the default) | 1406.18 | 2912.9 | |
| 1024 | 1221.90 | 3352.2 | -13.1% |
| 512 | 932.37 | 4393.1 | -33.7% |

That is what a fixed cost per chunk predicts: halving the width doubles the
chunk count and therefore the fixed term.

**4096 (plan 5d, 2026-11-06)**: `--prefill-chunk 4096` (the scratch sized for
4096 rows, +0.70 GB), Qwen3.8-27B int4 g64, `l0-int8`, card 1, `--max-len
133120`, 4 interleaved rounds after a warm-up (order flipped each round),
median; the box was busy with I/O (a disk copy) and a test job on card 0, so
these are provisional until the quiet-box re-run (`r75.speed`):

| depth | chunk 2048 t/s | chunk 4096 t/s | 4096 vs 2048 |
|---:|---:|---:|---:|
| 4096 | 2191.8 | 2284.0 | +4.2% |
| 8192 | 2099.9 | 2191.6 | +4.4% |
| 32768 | 1674.7 | 1762.7 | +5.3% |
| 131072 | 925.4 | 939.3 | +1.5% |

Every round agreed to within 0.6 %. Below the plan's +5 % bar at pp4096, so
2048 stays the default.

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
docs/probes/probe-flash-attn-2026-09-25.md), in place of the composed QK^T GEMM, causal
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

**Prefill at depth, `--prefill-length N --max-len 131072`, median of 3:**

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

**Decode at depth (F3), `--prefill-length N --max-len 131072`, tg 256, median of 3:**

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
1)`), research lever 2 of docs/12-kernels.md ("How the prefill kernels got here"). Lever 1, `exp2` with
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
continues the resident session or takes a prefix cache hit from the deepest host snapshot and prefills only the
tail. Device 0, `l0-int8`, max_len 131072. Records: docs/probes/probe-prefix-cache-2026-09-27.md.

- **P0**: pinned host copies at 14.2 GB/s device to host, 12.2-12.9 GB/s host to device;
  48 GiB pinned allocation fine; tail floor 316 ms at depth 0, 505 ms at 60k (16 ids).
- **S3** write-through: +2.03% on cold pp4096, +1.24% on pp32768 (bar 3%).
- **S1** continuation, 60000 ids of history + a 1000-id tail, time to first token:
  **1233.4 ms** (bar 1500; cold prefill of the history alone: 45.2 s). Load 8.6-11.9.
- **S2** the same after a 300-id side request: prefix cache hit **347.7 ms** (3933 MB of KV + the
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

**llama-benchy, merged main cea0e6c (2026-09-28, 21:18-21:29, load 1.0-4.2).** `uvx
llama-benchy --pp 1024 --tg 64 --depth 0 4096 16384 32768 --enable-prefix-caching
--exact-tg --latency-mode generation --runs 3` through `tools/probe/serve_benchy.sh`,
`b70-serve --max-len 131072` (int8 `lm_head`, `--mtp 0`), `--prefix-cache-gb 0` against
32, one after the other (decode attention v1: main before spec 10). `ctx_pp @ dN` loads N tokens of context; `pp1024 @ dN` is the
follow-up turn (the context plus a new 1024-token user message). Time to first response:

| follow-up after | cache off | cache on | |
|---:|---:|---:|---:|
| 4096 | 2716.9 ms | 1703.1 ms | 1.6x |
| 16384 | 9553.1 ms | 2165.2 ms | 4.4x |
| 32768 | 21003.1 ms | 2663.4 ms | 7.9x |

Decode is unchanged by the cache (tg64 30.78 / 25.79 / 21.26 t/s off, 30.79 / 25.82 /
21.26 on), and the context load costs the write-through (ctx_pp 2112.6 → 2066.7 t/s at
4096, 1631.5 → 1609.7 at 32768). The cached follow-up at 32k is 2.66 s rather than the
~1.0 s a pure 1k-token continuation takes (P0): llama-benchy's follow-up is a new user
message, so the prompt diverges at the previous prompt's assistant header, a few ids
before its end, and the prefix cache hit falls back to the last 2048-id block snapshot, re-prefilling
up to ~2000 extra ids. A snapshot at the generation prompt's start would close that.

## Decode attention at depth (spec 10)

Spec 10 (plans 10a and 10b, 2026-09-28): decode attention is `attn_decode_v2` +
`attn_reduce_v2` (src/kernels/attn_v2.cl), the default since plan 10b; v1 (attn.cl's pair)
stays selectable with `B70_DECODE_ATTN=v1`. Many positions per work-group with the stride
derived per row from the row's key count (at most 32 partials per row, from 2045 at 128k),
K by sub-group block reads and V by 2D block reads, one K/V wave feeding all 6 x M rows,
one barrier per wave. It ships with attn.cl's `exp`: the approved exp2 form failed the
golden gate (spec 10 §8). Design record: docs/probes/probe-decode-attn-2026-09-28.md; gates:
spec 10 §8.

**The captured decode step, v1 against v2** (`probe_decode_attn ab`: one engine, both
lists captured over the same buffers after prefilling `long32k.ids` tiled to the deepest
depth; 2 warm-up steps per arm, then 3 interleaved rounds of 16 steps each with the order
alternating, `pos` rewound to the depth before every step; median of the rounds and the
median paired ratio). Device 0, `l0-int8` prefill, bf16 `lm_head` (the branch predates spec
9's int8 head), every job under the GPU lock.

max_len 131072 (uptime load average 1.24 at start, 1.01 at end):

| depth | v1 ms/step | v2 ms/step | v1 t/s | v2 t/s | paired v1/v2 | A-F3 bar |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 33.718 | 31.842 | 29.66 | **31.41** | 1.059 | (A-F4: no regression) |
| 32768 | 48.433 | 35.913 | 20.65 | **27.85** | 1.348 | 23.6 |
| 65536 | 64.834 | 40.798 | 15.42 | **24.51** | 1.588 | 21.0 |
| 130816 | 98.355 | 49.679 | 10.17 | **20.13** | 1.977 | 17.3 |

v1's column reproduces spec 6's F3 rows (29.28 / 20.60 / 15.46 / 10.19 t/s, tg 256) within
1.3%. 10a's derived prediction for the exp form was 28.09 / 25.40 / 21.03 t/s. The exp2
build, same method in an earlier session (load 3.49 / 1.74), gave 31.49 / 28.21 / 25.27 /
21.04 t/s.

max_len 16384, the decode list at M = 1 and the M = 2..4 lists spec 8's verify uses (ms
per step of M rows; uptime load average 1.01 at start, 1.03 at end):

| M | depth | v1 ms/step | v2 ms/step | paired v1/v2 |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 33.706 | 31.772 | 1.060 |
| 1 | 12288 | 37.911 | 33.029 | 1.148 |
| 2 | 4096 | 39.364 | 36.029 | 1.093 |
| 2 | 12288 | 46.619 | 37.974 | 1.225 |
| 3 | 4096 | 51.369 | 46.776 | 1.098 |
| 3 | 12288 | 61.422 | 49.326 | 1.242 |
| 4 | 4096 | 58.683 | 52.985 | 1.107 |
| 4 | 12288 | 72.142 | 56.275 | 1.278 |

The M = 2..4 lists exist at max_len 16384 only (spec 8), so verify-step times at 32k and
beyond are not measured in the list; 10a's kernel-level M = 4 figures are the estimate
there.

**Correctness.** Against v1 on plan 10a's real layer-15 q/KV: worst cosine 0.999999491
(65536), max abs 0.0156 (one bf16 ulp), bitwise repeatable; every M = 2..4 row bitwise
equal to the M = 1 step at `pos + m` (the stride is a function of the row's own key count),
so spec 8's M2 is bitwise with v2. Golden gate passes; 256 greedy ids from the 32704-id
prompt at max_len 131072 match v1 for 133 ids and part at a near-tie (v1's margin 0.081 at
a logit of 20); passkey `l0-int8` 3/3 at 119939 ids.

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

### int8 lm_head (spec 9)

**Every row in this file names its head form. Rows without a form are bf16, the
checkpoint's own head, byte-matched with vLLM; the vLLM comparison rows stay bf16.**
`--lm-head int8` quantises the bf16 head at load to int8 rows with one fp32 scale per
row (W8A16, `src/loader/lm_head_int8.h`) and reads it with `gemv_i8w`. It is
`b70-serve`'s default. `b70-decode` defaults to bf16.

Interleaved pairs after a warm-up, box 2026-09-28 17:12-17:16 (`uptime` load 0.25 at
the start, 2.22 at the end; other agents share the box but run under the same GPU
lock), device 0, `tools/probe/lm_head_pairs.sh`, sha `b7e8d52-dirty` (the plan 9b
tree before its record commit). Each run is `b70-decode --bench --prefill-length 4096 --tg 256`:

| pair | head | decode t/s at 4096 | ms/token | pp4096 t/s (l0-int8) |
|---|---|---:|---:|---:|
| 1 | bf16 | 29.38 | 34.03 | 2166.10 |
| 1 | int8 | 31.21 | 32.04 | 2164.00 |
| 2 | int8 | 31.21 | 32.04 | 2160.43 |
| 2 | bf16 | 29.38 | 34.04 | 2159.22 |
| 3 | bf16 | 29.38 | 34.04 | 2162.26 |
| 3 | int8 | 31.25 | 32.00 | 2155.51 |
| **median** | **bf16** | **29.38** | 34.04 | **2162.26** |
| **median** | **int8** | **31.21** | 32.04 | **2160.43** |

- **H1, decode, pass:** 31.21 / 29.38 = **1.062x** (bar 1.06x), -2.00 ms per token.
  vLLM's 31.01 t/s is bf16 and is not compared with the int8 row.
- **H3, prefill, pass:** pp4096 -0.08 % (bar 1 %). Prefill reads the head once.
- **The head launch in situ** (`b70-decode --profile --depth 512 --steps 32`, 32
  replays): bf16 `gemv_bf16` 4387.992 us for 2,542,796,800 B = **579.5 GB/s**;
  int8 `gemv_i8w` 2204.674 us for 1,272,391,680 B (1,271,398,400 weight + 993,280
  scale) = **577.1 GB/s**, 99.6 % of the bf16 launch's rate. The kernel keeps the
  saving whole: 2.183 ms per launch in the profile, 2.00 ms per token on the bench.
  The kernel test's isolated figure is 581.1 GB/s at M = 1 and 558.6 GB/s at M = 4
  (`gemv_i8w_test`, a loaded box).
- **H2, MTP (spec 8's engine), recorded:** `probe_mtp_steps <snap> 4096 32 3 <head>`,
  two interleaved pairs, medians agree to 0.1 %:

  | list | bf16 head | int8 head | |
  |---|---:|---:|---:|
  | draft k=1 | 6.42 ms | 4.26 ms | -34 % |
  | draft k=3 | 19.00 ms | 12.50 ms | -34 % |
  | verify M=1 | 34.73 ms | 32.65 ms | -6.0 % |
  | verify M=4 | 60.28 ms | 58.35 ms | -3.2 % |

  A K = 3 iteration (three drafts + verify at M = 4) goes from 79.28 to 70.85 ms
  (derived, -10.6 %). Acceptance is not re-measured here: the engine has no
  acceptance loop until plan 8c. What the head changes in the draft's argmax is
  what L1 measures (1 near-tie flip in 2790 rows).

**Accuracy (spec 9 §4, as amended in §8):**

- **L1/L2 on the card** (`tools/probe/probe_lm_head_l1`: the engine's int8-head logits
  against its bf16-head logits from the same replay's hidden, teacher-forced on the
  3 golden prompts + 32 golden tokens and all 36 A4 scenarios + the bf16 oracle's
  output; 2790 rows): cosine min **0.9999147**; argmax 1 mismatch, a bf16 near-tie;
  top-20 equal 91.1 % raw and **99.9 %** under the amended rule (4 rows differ across
  a gap >= 0.05, max 0.064, all on A4 rows); unfiltered KL mean / p99 **8.6e-6 /
  1.9e-4** at T 1.0 and **1.4e-5 / 4.0e-4** at T 0.6. Pass. Dropped mass under the
  k 20 / p 0.95 filter, recorded: max 0.027 (T 1.0), 0.040 (T 0.6). The golden
  prompts alone (99 rows) agree with plan 9a's CPU table (cosine min 0.9999147 here
  and 0.9999148 in 9a; unfiltered KL 9.8e-5 / 5.3e-4 at T 1.0 and 1.7e-4 / 1.0e-3
  at T 0.6, as in 9a's golden column). Alone, those 99 rows are inside the KL bars
  at T 1.0 and above them at T 0.6; the bars are graded over the whole set, which
  is how 9a's verdict and the amendment's numbers took them.
- **L3, end to end, pass:** `golden_gate_i8head_test` (decode) and the prefill gates
  on `l0` and `l0-int8` with the int8 head (`prefill_gate_{l0,int8}_i8head_test`):
  **93/93 determined rows exact** each, the same 3 undetermined rows inside the
  oracle's argmax set. **Gate A4:** `l0-int8` + int8 head **25 / 36** against the
  bf16 model, the same as with the bf16 head (`LM_HEAD=int8
  tools/toolcall/engine_generate.sh`, `score.py ... bf16 l0-int8 l0-int8-i8head`).
- **L4, pass:** `replay_determinism_i8head_test` (774 kernels, 19 modules, 8 tokens x
  3 runs bitwise); every bf16-head registration unchanged.
- **Load:** the host quantisation takes 0.35-0.75 s on 44 threads, and the load is
  13.6 s against 14.4 s for bf16 (fewer bytes to upload).

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

## MTP speculative decoding (spec 8)

`b70-serve --mtp K` drafts K tokens with the checkpoint's MTP head, verifies
them in one main-model step of M = K + 1 rows, and accepts on the host (greedy,
or lossless sampling after Leviathan et al. 2023). Spec:
`docs/superpowers/specs/2026-09-27-spec8-mtp-speculative-decoding-design.md`, with its §8
amendment. The probe and the engine are in `docs/probes/probe-mtp-2026-09-27.md`.

**Scope.** The MTP lists are compiled at max_len 16384 only (spec 8 §8 A9), so
every row below runs at 16384. The spec's 32k and 64k rows cannot be measured
yet. A 12k row stands in for depth.

**Grade: iterate.** The box was shared with other agents during every run below
(the 1-minute load is printed per round). Arms were interleaved per process,
never run back to back. The seeded sampled runs draw the same tokens in every
round, so the spread in each cell is timing only.

### P0, acceptance and step costs (plan 8a/8b)

- **Greedy depth-1 acceptance** on the CPU reference head: golden 0.820,
  tool-call outputs 0.985.
- **Verify step cost** at M = 2 / 3 / 4: 1.17 / 1.52 / 1.74 plain steps.
- **Draft step cost:** 0.185 plain steps.
- Full detail: `docs/probes/probe-mtp-2026-09-27.md` §2, §3 and §6.

### The verify step on gemm_sm (spec 28b, 2026-11-01 - busy-box timing on one card, to be re-checked on a quiet box)

Every M >= 2 decode list binds the small-M DPAS GEMM (`gemm_sm`, spec 28 §10); M = 1 keeps gemv, bit for bit.
verify(M) in plain steps, int8 head, depth 4096 (`probe_verify_m --verify8`, 5 rounds interleaved; the same lists
forced onto gemv in the same session):

| M | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm_sm (28b) | 1.039 | 1.090 | 1.135 | 1.183 | 1.297 | 1.359 | 1.570 |
| gemv (before) | 1.157 | 1.513 | 1.715 | 2.005 | 2.368 | 2.633 | 3.170 |
| depth 32768, gemm_sm | 1.099 | 1.208 | 1.309 | 1.701 | 2.257 | 2.541 | 2.909 |

`probe_mtp_steps` pairs (int8): 1.037 / 1.086 / 1.131 / 1.181 / 1.300 / 1.352 / 1.562; bf16 head 1.044 / 1.094 /
1.134 (M 2..4). These are MtpCost's verify rows now. The M = 8 step is the attention build's (spec 28c), the GEMV
part is flat. **D1 at K = 3 on the same day** (mtp_d1.sh's arms, bf16 head, 4096, main's build interleaved):
tool-call geomean **2.121x** (main 1.643x), golden 1.549x (main 1.228x); tool calls ~66 t/s against ~52.

### Acceptance in the server (M3 run, 256 greedy ids, 39 prompts)

These come from `mtp_gpu_test`, using the engine's head over its own greedy
output. There are three golden prompts and 36 A4 tool-call prompts.

| K | acceptance, l0-int8 | ids per iteration | acceptance, l0 |
|---|---:|---:|---:|
| 1 | 0.9235 (4802/5200) | 1.920 | 0.9240 |
| 2 | 0.8697 (6363/7316) | 2.729 | 0.8707 |
| 3 | 0.8227 (7133/8670) | 3.455 | 0.8164 |

Free text accepts less than the tool-call prompts:

| prompt | K = 3 acceptance |
|---|---:|
| prose | 0.444 |
| cjk | 0.489 |
| code | 0.798 |
| tool-call prompts | 0.707 to 1.000 |

### D1, greedy decode (bar: >= 1.4x plain on the golden prompts and on the opencode replay)

`tools/probe/mtp_d1.sh greedy 4096`, 2026-09-28 15:33-16:34, load 1.0-11.1.

- **Setup:** each process runs one arm, 256 ids per prompt. Generation is timed
  with prefill excluded, and the median of 3 rounds is shown.
- **Rounds:** the order rotates between 0 3 1 2 and 2 1 3 0.
- **K = 0 loads no head.** It is today's server.
- **Golden prompts** sit at depth 4096 behind a prefix of `long32k.ids`.
- **Tool-call prompts** are the four P0 scenarios, at their own depth (2.2k to 3k).

| prompt | K=0 t/s | K=1 | K=2 | K=3 |
|---|---:|---:|---:|---:|
| golden/code@4096 | 29.30 | 40.92 (1.397x) | 42.44 (1.448x) | 43.93 (1.499x) |
| golden/prose@4096 | 29.27 | 35.38 (1.209x) | 31.83 (1.087x) | 28.39 (0.970x) |
| golden/cjk@4096 | 29.27 | 37.79 (1.291x) | 35.80 (1.223x) | 33.55 (1.146x) |
| toolcall/t3_rename-openai | 29.98 | 43.24 (1.442x) | 45.54 (1.519x) | 49.94 (1.666x) |
| toolcall/t4_comment-box | 30.14 | 42.97 (1.426x) | 45.95 (1.525x) | 46.97 (1.559x) |
| toolcall/t4_comment-loader | 29.97 | 42.60 (1.422x) | 44.54 (1.486x) | 51.44 (1.716x) |
| toolcall/t5_test-step | 30.01 | 42.65 (1.421x) | 45.62 (1.520x) | 47.88 (1.595x) |
| **geomean, golden** | | **1.297x** | 1.244x | 1.186x |
| **geomean, tool-call** | | 1.428x | 1.512x | **1.633x** |

The same run at depth 12288 (`mtp_d1.sh greedy 12288`, 16:50-17:10, load
1.0-5.8) gives these golden geomeans:

| K | golden geomean @12288 |
|---|---:|
| 1 | 1.264x |
| 2 | 1.179x |
| 3 | 1.099x |

- **K = 0 at 12288:** 26.07-26.09 t/s. The tool-call rows repeat the 4k rows
  to within 0.1%.
- **Verdict on the golden prompts: MISS.** The best K is 1, at 1.297x. At K = 3
  prose falls below plain decode (0.970x).
- **Tool-call proxy for agentic output: PASS** at every K. K = 3 gives 1.633x.
- **Opencode replay: pending.** The log does not exist yet. Rerun once it does:
  `tools/oracle/run_in_container.sh 'python3 tools/oracle/mtp_accept.py "$SNAP" --opencode /ws/tests/golden/opencode/session1 --out /ws/oracle-out-spec8a/accept_opencode.json'`
  for acceptance, then `tools/prefix/replay_log.py` against
  `b70-serve --mtp K` for the replay.
- **Why golden misses:** spec 8 §1 assumed a verify step costs 1.05-1.15 plain
  steps; it measures 1.17-1.74 (§8 A7). The two levers are the int4 GEMV at
  M = 3/4 and `attn_decode`'s per-row KV walk.

### D2, sampled decode at T 1.0, top-p 0.95, top-k 20 (bar: >= 1.25x)

Setup:

- `tools/probe/mtp_d1.sh sampled 4096`, 16:34-16:50, load 5.5-11.6.
- The parameters are the checkpoint's `generation_config.json` (P0 §2). The
  opencode log's own temperature is pending.
- Seed 1234. K = 0 is today's host sampler.

| prompt | K=0 t/s | K=1 | K=2 | K=3 |
|---|---:|---:|---:|---:|
| golden/code@4096 | 28.93 | 38.12 (1.318x) | 34.39 (1.189x) | 36.76 (1.271x) |
| golden/prose@4096 | 28.89 | 41.66 (1.442x) | 33.20 (1.149x) | 48.16 (1.667x) |
| golden/cjk@4096 | 28.89 | 35.82 (1.240x) | 36.86 (1.276x) | 35.08 (1.214x) |
| toolcall/t3_rename-openai | 29.53 | 42.48 (1.438x) | 45.13 (1.528x) | 47.45 (1.607x) |
| toolcall/t4_comment-box | 29.68 | 42.19 (1.421x) | 44.10 (1.486x) | 45.30 (1.526x) |
| toolcall/t4_comment-loader | 29.52 | 41.52 (1.406x) | 42.31 (1.433x) | 44.27 (1.499x) |
| toolcall/t5_test-step | 29.57 | 40.63 (1.374x) | 43.73 (1.479x) | 45.59 (1.542x) |
| **geomean, golden** | | 1.330x | 1.203x | 1.370x |
| **geomean, tool-call** | | 1.410x | 1.481x | **1.543x** |

**Verdict: PASS**, with two caveats:

- **Each cell is one sampled trajectory.** Each K draws a different text. The
  sampled prose run at K = 1 and K = 3 accepts 1.000 and 0.975, which is the
  signature of a repetitive continuation. So the golden sampled geomean is
  noisier than the greedy one.
- **Host sampling costs about 1.3%** of plain decode (28.9 t/s against 29.3
  t/s greedy).

### Correctness in the server

| gate | result |
|---|---|
| M3, l0-int8 | greedy with `--mtp K` against `--mtp 0`, 256 ids, 39 prompts, K = 1, 2, 3: **0 divergences**. Identity, as plan 8b's bitwise M2 predicted. |
| M3, l0 | the same run on l0: **0 divergences** |
| RF1 on the card | a run truncated inside an accepted burst, then continued, equals the plain run (96 ids, 3 prompts, both backends) |
| M5 | the same greedy K = 3 run twice: ids and the final logits row bitwise, on both backends |
| M4, host | `spec_accept_test`: 1e6 draws of the full procedure, chi-square of the first id against filtered p_0 and of the second against p_1, unfiltered and at top-k 12 / top-p 0.8 / T 0.7. p-values 0.61, 0.62, 0.38, 0.10 (bar 0.01). One-candidate and q(d) = 0 guards. Seeded reproducibility. |
| M4, card | a seeded sampled K = 3 run, twice: bitwise (3 prompts, both backends) |
| server | `mtp_server_test`: EOS, stop strings and `max_tokens` inside a burst, and a streamed tool call in bursts of 1-4. Byte-identical to `--mtp 0` under 4 burst patterns, and the engine is left at the same position. |
| M5, spec 7 C2 with MTP on | **pending** plan 7c's merge. `prefix_gpu_test` lives on `spec7c-server-prefix-cache` and loads no head. The merge adds an argv[6] = K there: `loader::load(..., K > 0)` and `EngineAdapter(eng, vocab, K)`. Then one command: `build/tests/prefix_gpu_test "$SNAP" tests/golden/prompts l0-int8 1 0 3`. |

### The verify attention on DPAS (spec 28c, branch `spec28c-verify-attn`, measured 2026-11-03, **busy box**, card 1)

`attn_verify_dpas` against v2 in the verify lists at M >= 2, bound by the measured rule (`src/runtime/verify_attn.h`:
bf16 M = 2 from 11585 keys, M = 3..5 from 4096, M = 6..8 always; int8 M = 2 from 11585, M = 3 / 4 from 4096). Record:
docs/probes/probe-verify-attn-2026-11-03.md. Re-check on a quiet box.

| row | v2 | the rule | ratio |
|---|---|---|---|
| attention per FA layer, M = 8, 32768 keys (bf16 / int8) | 3975 / 3255 us | 460 / 689 us | 8.4x / 4.7x |
| verify(8) at 4096 keys, int8 head (`probe_mtp_steps ... 8 vattn`) | 94.7 ms | 85.6 ms | 1.106 |
| verify(8) at 32768 keys | 145.6 ms | 91.2 ms | 1.598 |
| verify(4) / verify(2) at 32768 keys | 62.3 / 41.2 ms | 54.8 / 38.2 ms | 1.138 / 1.080 |
| verify(8) / verify(4) / verify(2) at 120000 keys | 297.3 / 94.6 / 60.4 ms | 106.5 / 67.0 / 48.9 ms | 2.795 / 1.413 / 1.238 |
| D1 tool-call geomean, `--mtp 3`, P0 prompts (2.2-3.0k keys) | 52.63 t/s | 52.61 t/s | 1.000 |
| D1 golden at 32768 keys, `--mtp 3` (cjk / code / prose) | 31.7 / 40.8 / 31.7 t/s | 34.7 / 44.7 / 34.7 t/s | 1.095 |

## DFlash (spec 19)

**On the card since 2026-10-27 (plan 19f); rows 54-56 ran 2026-10-27 (branch `dflash-bars`); D5 says: an option, not
the default.** `b70-serve --spec dflash`
drafts with `z-lab/Qwen3.8-27B-DFlash2` (5 layers, int8 at load, beside the gate checkpoint's int8 head) from the
target's own hidden states and verifies up to 8 rows at once (specs 19b-19d; box queue rows 54-56). The acceptance in
"P0" is measured on the Mac CPU (spec 19a, `docs/probes/probe-dflash-2026-10-08.md`); "D5" below is measured on the card. Spec: `docs/superpowers/specs/2026-10-05-spec19-draft-model-speculative-decoding-design.md`.

### V2 baseline (2026-11-11, spec 19 v2 Tasks 1-2)

Branch `spec19v2-dflash-speed`, base `8d21b9f9`, int8 drafter and full int8 head; no speed changes yet.
Card 0, **busy box** (another card job ran concurrently). The operator's slim Task 2 reuses today's main baselines:
MTP auto **73.2 t/s** (README's one-card row), plain decode **33.5 t/s** at depth 4096. Neither baseline was rerun.

| slots_bench, one request | warm decode | cold / warm first token |
|---|---:|---:|
| recorded main, MTP auto, four sequence slots | 73.181 t/s | 2.316 / 1.091 s |
| DFlash, one sequence slot, context 32768, bf16 KV | **88.806 t/s** | 2.502 / 1.171 s |

Both saved requests contain **identical 3,425 prompt ids** and each generates **512 ids**, greedy with `ignore_eos`.
The earlier 2,820-token shape label was stale. Both warm requests restore the prefix at token 2048, with the
remaining prompt tail prefilling; prefix cache budget 32 GiB. MTP's recorded slot had context 32768. DFlash is
**1.213x** the operator's rounded 73.2 baseline, above the 80.5 t/s early decision point. The operator chose to
continue Tasks 3-7 after baseline gates clear. **The 80.5 t/s early bar PASSED** once CPU G-accept cleared below.
Task 8 compares against freshly measured main per-step MTP (`f65ae50e` or later), the default for `--mtp auto`.
The operator subsequently allowed one idle GPU with >16/44 CPU threads available, graded busy box. This single shape does **not** establish its replay/regression G-speed gate.

Provenance on the box: `~/b70/logs/brag-speed/20261111-1536/card-bf16/{bench.jsonl,server-log/000003.json}`
and `~/b70/logs/dflash-v2/baseline/slots/{bench.jsonl,server-log/000003.json,server.log}`. The DFlash command is
`tools/box_validate/dflash_v2.sh d5`; its driver log is `~/b70/logs/dflash-v2/d5.log`.

One golden/A4 serving round: **39/39 pass the f32 judge**, 37 bitwise identical to the saved plain stream;
golden cjk (first difference 16) and prose (74) each have zero f32-rejected ids on either suffix. No missing
references or skipped requests. Logs: `~/b70/logs/dflash-v2/{identity.log,baseline/identity/}`; saved plain
reference: validation state `e88ce65d57d8-vs-e88ce65d57d8`, `r55.serve/off.r1/requests.jsonl`.
**CPU G-accept PASS** (2026-11-12): the engine and reference use the same taps, anchors and active widths.
Existing 19a batched drafting is checked against a single real-model anchor at every K=1..7 before proceeding;
the parity check passes. The complete baseline is `~/b70/logs/dflash-v2/baseline/acceptance.json`, with all
K=1..7 counts and draft ids on five corpora; the completed driver ends `ALLDONE rc=0`.

| corpus | anchors | engine E7 | CPU reference E7 | absolute gap | G-accept |
|---|---:|---:|---:|---:|---|
| A4 | 544 | 7.080252 | 7.080252 | 0.000000 | PASS |
| golden code | 16 | 5.875000 | 5.750000 | 0.125000 | PASS (<=0.15) |
| golden prose | 16 | 2.286458 | 2.286458 | 0.000000 | recorded |
| golden cjk | 16 | 1.500000 | 1.500000 | 0.000000 | recorded |
| held-out opencode | 16 | 3.687500 | 3.687500 | 0.000000 | recorded |

Initial cost-probe ratios were invalid because its restored plain row was inactive; those ratios are excluded.
The corrected probe requires its plain replay to consume one row
and match verify(1)'s greedy id before timing. Both depths pass that calibration.

Uninstrumented production lists, interleaved pairs with a restored snapshot before every call; one warm-up
round, then medians of three rounds of 16 calls. **Busy box:** another card job and the CPU reference ran
concurrently. Plain normalization is each arm's paired plain replay (about 29.0 / 32.8 ms), not the reused
33.5 t/s decode baseline above. The iteration includes live greedy acceptance and its commit.

| arm | depth 4096, ms | / paired plain | depth 32768, ms | / paired plain |
|---|---:|---:|---:|---:|
| draft, K=7 | 8.978022 | 0.309880 | 8.963257 | 0.273906 |
| verify(1) | 28.966064 | 1.000578 | 32.852369 | 1.000564 |
| verify(2) | 30.350444 | 1.047126 | 33.445930 | 1.020956 |
| verify(3) | 31.751969 | 1.091869 | 35.302113 | 1.077380 |
| verify(4) | 32.644368 | 1.121579 | 36.335702 | 1.108764 |
| verify(5) | 33.865681 | 1.168254 | 38.812380 | 1.183860 |
| verify(6) | 35.023325 | 1.205641 | 40.052069 | 1.221439 |
| verify(7) | 35.741238 | 1.229004 | 41.244214 | 1.256987 |
| verify(8) | 36.254194 | 1.245386 | 41.750586 | 1.272373 |
| projection | 0.567761 | 0.019453 | 0.563306 | 0.017169 |
| empty submit + fence | 0.020969 | 0.000723 | 0.019653 | 0.000600 |
| K=7 iteration | 45.830047 | 1.579196 | 51.413898 | 1.562625 |

The C7 grades at depth 4096:

| quantity | measured | target | grade |
|---|---:|---:|---|
| draft | 8.978 ms / 0.309880 plain | <=4.5 ms / <=0.15 | FAIL |
| verify(8) | 1.245386 plain | <=1.15 | FAIL; faster than the older 1.57, still above target |
| projection + host gaps | instrumented upper bound 1.797466 ms; projection alone 0.567761 ms wall | <=0.5 ms | NOT PASSED; exact host gap not isolated |
| iteration | 1.579196 plain | <=1.35 | FAIL |

Separate timestamped lists have **87 draft / 8 projection launches**. At depth 4096 the draft's device sum is
8.605521 ms: GEMMs 5.390208 ms (full int8 head **2.165104 ms**, remaining GEMMs 3.225104 ms), attention part
2.711250 ms, attention reduce 0.047396 ms, norms 0.127500 ms, top-k stages 0.204375 / 0.028333 ms. The
projection's device sum is 0.436563 ms (GEMMs 0.421042 ms). Per-launch details are retained in
`~/b70/logs/dflash-v2/iter-{4096,32768}.log`. Device draft/projection sums at 32768 are 8.621875 / 0.438646 ms.
The host-gap upper bounds are 1.401320 / 1.282363 ms, and projection + host upper bounds 1.797466 / 1.664446 ms:
per-launch timestamp signals add host-scope flushes and device dispatch gaps remain in the residual. These
instrumented bounds are diagnostics, not exact uninstrumented host costs or the benchmark timings above.

Task 3 uses the measured verify ratios above and a flat int8 draft row of 0.309880 in AdaptiveK. Its head share
is 0.241156 (2.165104 ms divided by the paired 28.972533 ms plain step, then by 0.309880); the remaining draft
cost is assigned to the body. The unmeasured bf16 arms are explicitly **derived**: double the head's cost and
the body's GEMM cost, keeping attention, other kernels and host overhead unchanged. Their body/head costs are
0.346466 / 0.149459 plain steps, and their verify ratios are taken equal to int8. MTP's tables are unchanged.
At acceptance alpha 0.5 / 0.8 / 0.95, this DFlash table chooses K=4 / 7 / 7. Task 7b will refresh these costs
after the GPU optimizations.

EOS-stopped free runs (capped at 256 ids), separate from the teacher-forced acceptance anchors:

| corpus | requests | ids / iterations | ids per iteration | requests reaching EOS |
|---|---:|---:|---:|---:|
| A4 | 34 | 2622 / 394 | 6.65482 | 32 |
| golden code | 1 | 146 / 24 | 6.08333 | 1 |
| golden prose | 1 | 256 / 99 | 2.58586 | 0 |
| golden cjk | 1 | 256 / 124 | 2.06452 | 0 |
| held-out opencode | 1 | 97 / 23 | 4.21739 | 1 |

The original engine-tap baseline is `~/b70/logs/dflash-v2/baseline/engine/`: 38 tap files and 4256 draft records
(16 anchors per source at each K=1..7). The CPU comparison drafts every active K separately: allocated eight-row
lists still mask noncausal attention at K+1, so smaller K's are not prefixes of K=7. Two A4 recordings with fewer
than eight EOS-stopped continuation ids cannot supply these anchors and are excluded. The separate serving
correctness round above covers all 36 A4 requests.

### Task 4: one-submit iteration (2026-11-12, `882989b8`)

The production DFlash path submits the pending projection, draft and verify together. Four device regressions
passed with no skips: `dflash_profile_test`, `dflash_engine_test`, `dflash_gpu_test` and
`dflash_prefix_gpu_test` (1620 s total). They check one submission per iteration, bitwise draft/readback,
verify rows and ring, and projection materialization at snapshots, truncate and request end. The two-card
`pp_dflash_test` passed in 114.02 s when both cards became idle (fresh rebuild, no skip). All five Task 4
device checks and both calibrated host-gap bars pass; the full C7 cost targets below remain unmet.

**Busy box, exclusive card 1**, paired medians with calibrated active plain rows at both depths. The steady-state
probe prepares the last prompt row through verify/commit outside timing, leaving its projection pending for the
batched iteration; plain pairs use the same preparation. This is not a final serving speed measurement.

| quantity | depth 4096 | depth 32768 |
|---|---:|---:|
| draft, ms | 9.039839 | 9.080943 |
| verify(8), paired plain steps | 1.254267 | 1.274542 |
| projection wall, ms | 0.490908 | 0.506852 |
| K=7 iteration, ms | 45.939428 | 51.677500 |
| K=7 iteration, paired plain steps | 1.577007 | 1.558300 |
| instrumented host outside device lists, ms | **0.422639** | **0.381236** |
| projection + host outside device lists, ms | 0.804514 | 0.766757 |
| old kernel-sum host residual upper bound, ms | 1.371910 | 1.323840 |

Task 4's host-only gap bar (<=0.5 ms) passes at both depths. C7's combined projection + host bar (<=0.5 ms)
and full iteration bar (<=1.35 plain steps) remain unmet. Device-list intervals run from the first kernel start
to the last kernel end, including dispatch and timestamp signal flushes within each list. The host figure is
instrumented wall time minus those intervals; the older wall-minus-kernel-sums residual remains an upper bound.
Evidence: `~/b70/logs/dflash-v2/task4/{device.xml,iter-4096.log,iter-32768.log}` and `task4-device.log`.

### Task 5: RTN int4 diagnostics (2026-11-12)

**Acceptance FAIL**, source `39d54ad1`, full int8 head, all original anchors/K and exact original-baseline
recalibration. K7 A4 falls 7.08025 → 7.00551, prose 2.28646 → 2.22727, held-out opencode 3.68750 → 3.25000;
these losses exceed 0.05. C2 therefore requires an AutoRound-exported drafter before promotion; int8 stays default.
Full table and provenance: [DFlash record](probes/dflash-vs-sly-2026-11-11.md#7-spec-19v2-measurements-2026-11-12).

`8a16f5a4` splits the int4 producers across K and folds their partials inside the existing consumers (87 draft /
8 projection launches). **Exclusive card 1, busy box; cost diagnostics only.**

| quantity | depth 4096 | depth 32768 |
|---|---:|---:|
| draft, ms | 7.849702 | 8.290171 |
| projection wall, ms | 0.337303 | 0.459109 |
| verify(8), paired plain steps | 1.252052 | 1.276792 |
| K7 iteration, paired plain steps | 1.528099 | 1.527929 |
| instrumented projection + host outside lists, ms | 0.665543 | 0.940995 |

The 4k draft saves 1.128320 ms against Task 1, below Task 5's 2.5 ms gate. The 4k profile has body GEMMs
1.912396 ms, full head 2.168333 ms, attention-part 2.768438 ms, and top-k stages 0.208125 / 0.028646 ms.
New split consumers pass the bitwise kernel tests. The later contracted-CPU-twin regression passes the full
int4 numerical test: both projection paths have zero differing values, ring f32 cosine is 0.9999978, and the
aggregate f32 candidate-set gate is 5 misses versus 6 for forced GEMV. The CPU twin now mirrors the verified
GEMV fused multiply-add order; GPU arithmetic and judge bars are unchanged. This does not repair RTN corpus loss. Logs: `~/b70/logs/dflash-v2/task5-split-cost/iter-{4096,32768}.log`,
`task5-split-kernels-card1.log`. These numbers do not refresh AdaptiveK or establish the final speed bar.

### Task 6: candidate head and exact rerank (2026-11-12)

At `18d366bf`, the full-vocabulary int4 candidate head selects 64 candidates per row, then reranks their
original int8 rows exactly. The unchanged int8 body isolates head quality. All 35 corpus/K acceptance values
match the original baseline exactly; the one-load `--all-k` harness first reproduced all 4256 original drafts.
This passes isolated head quality, including the held-out opencode session, but does not pass the cost gate.

| quantity | depth 4096 | depth 32768 |
|---|---:|---:|
| candidate GEMM, ms | 1.214896 | 1.199167 |
| top-64 stages 1 / 2, ms | 0.365000 / 0.073750 | 0.367812 / 0.074375 |
| exact rerank, ms | 0.135625 | 0.134167 |
| sum of head-path launch medians, ms | 1.789271 | 1.775521 |
| draft wall, ms | 8.723402 | 8.618562 |
| K7 iteration, paired plain steps | 1.591323 | 1.556495 |

The Task 6 head-share target of 0.5 ms **fails**. These are exclusive-card-1/busy-box diagnostics with the
original H4/B8 attention geometry, not final AdaptiveK costs or the Task 8 speed bar. Capture has 88 draft /
8 projection launches. No serving default changed. Evidence: `~/b70/logs/dflash-v2/task6/candidate64-int8/`
(`acceptance.json`, `iter-{4096,32768}.log`) and `task6-candidate-{pilot,cost}.log`.

### Task 7: attention geometry (2026-11-12)

Random wrap/mask/long-position tests and real-model readback/hidden-state comparisons pass bitwise for H1/B4
(one query head and four block rows per workgroup). Interleaved timestamp medians for the five attention-part
launches are 2.084063 ms versus 2.696458 ms for H4/B8 at pos4096/K7; 1.227500 versus 1.595521 ms at K2;
0.554583 versus 1.144688 ms at pos16/K7. Evidence: `task7-attn-rows.log`.

Parallel exponentials remain a probe control: bitwise checks pass, but steady K7 slows to 2.393125 ms versus
2.074167 ms for serial H1/B4 in the paired experiment. Both 16-bit and packed-32-bit local K-cache experiments
fail the bitwise gate and are rejected before timing. H1/B4 is now selected. Its kernel/profile/engine/greedy-identity
integration checks pass 4/4 with no skips (21.42 / 29.17 / 309.71 / 1075.73 seconds). The full int8 numerical check
subsequently passes (`resume-numerical-card0.log`, DONE rc0). The later real-list profiles, exclusive card1/busy box with CPU reference work, are:

| int8 body / head mode | depth | draft ms / plain steps | verify(8), plain steps | iteration, plain steps |
|---|---:|---:|---:|---:|
| full int8 head | 4096 | 8.712270 / 0.298136 | 1.246703 | 1.557816 |
| candidate64 + exact rerank | 4096 | 8.429809 / 0.287036 | 1.242865 | 1.558631 |
| full int8 head | 32768 | 8.788698 / 0.265708 | 1.271189 | 1.548661 |
| candidate64 + exact rerank | 32768 | 8.527476 / 0.257115 | 1.271936 | 1.530302 |

C7 remains **FAIL**. The 4k candidate profile's instrumented host time outside lists is 1.860053 ms and
projection + host is 2.247761 ms; concurrent CPU reference work makes these busy-box diagnostics. The final
speed bar is still a separate rotated serving comparison. All five profiled attention launches bind H1/B4.
Evidence: `~/b70/logs/dflash-v2/task7/policy-h1b4/{device.xml,iter-{full,candidate}-{4096,32768}.log}`.

The RTN int4 H1/B4 diagnostic completed all four profiles before the operator reclaimed card1:

| RTN body / head mode | depth | draft ms / plain steps | iteration, plain steps |
|---|---:|---:|---:|
| full int8 head | 4096 | 7.784417 / 0.265500 | 1.542619 |
| candidate64 + exact rerank | 4096 | 11.264767 / 0.350649 | 1.554384 |
| full int8 head | 32768 | 7.109450 / 0.215375 | 1.546868 |
| candidate64 + exact rerank | 32768 | 6.565448 / 0.198730 | 1.498815 |

**The candidate 4k wall profile is noisy and is not a policy calibration:** its empty submit/fence costs
0.751771 ms versus 0.012662–0.014936 ms in the other three arms, projection is 3.248991 ms, and verify(3)
appears more expensive than verify(4). The saved measurements remain busy-box diagnostics. They do not repair
RTN's failed acceptance; no RTN arm is enabled for serving. AdaptiveK uses the new measured int8 full/candidate
rows; experimental RTN estimates remain explicitly unqualified pending the deferred AutoRound follow-up.
Evidence: `~/b70/logs/dflash-v2/task7b/h1b4-candidate-int8/iter-int4-{full,candidate}-{4096,32768}.log`.

At the GPU handoff, the next baseline calibration was interrupted: 2463 complete rows match the original
4256-row baseline, with one truncated line. This is **partial, not a passed calibration**; it is archived under
`paused-2026-11-12T1523/` beside `calibration-status.json`. That partial run is superseded by the complete 4256-row calibration and all 35 acceptance rows below.
The full int8 H1/B4 numerical check also passes. Main `0b5ef628` is merged at `b5bb4044`; fresh G0 and a new
Task 8 collection follow the successful merged build and six host checks. Pre-merge serving timings
remain historical. The operator restarted CPU judging during G0 (69 GiB, eight threads, saved layer 26);
overlapping timings are busy-box despite the whole-box GPU reservation.

### Cold-request reset (2026-11-13)

The pre-merge replay contains 30 one-token requests among its 78 requests. Their median DFlash TTFT is
507.9 ms versus MTP's 151.1 ms in round 1. Saved prefix logs isolate DFlash's cold reset at 324–332 ms,
versus 7–8 ms for MTP; prompt processing itself is 96 ms versus 93 ms. `DFlashBuffers::zero()` was rebuilding
and uploading the entire 131072-row RoPE table on every cold request.

Generating and uploading that immutable table once at construction reduces the focused buffer reset from
244.426 to **0.179 ms** on card0 (whole-box GPU reservation, CPU references/compiler active). The device
regression first fails when reset overwrites guards in the immutable allocation, then passes after the move;
initial RoPE matches the independent twin bitwise and the ring, control and readback still reset to zero.
Seven selected checks pass. This is a buffer measurement, not an end-to-end serving speed claim; the new
real-engine check passes (297.36 s); the main rotated replay and its f32 judging pass below. Evidence: `~/b70/logs/dflash-v2/reset-v7-{red,green}.log`
and the preserved full test output `~/b70/logs/dflash-v2/reset-v7-green-detail.log`.

### Task 8: serving validation after the reset fix (2026-11-13)

Source `a83322bb`, main `0b5ef628` merged at `b5bb4044`, server SHA256
`83d62ae7749afb3a81184fce2d0ab2de8ef601badb8172482257a9de653d6deb`.
Selected arm: int8 body / candidate64 exact int8 rerank / H1B4, physical card0. All three speculation arms
run three rotated rounds, prefix cache on, bf16 KV, int8 target head; MTP's startup confirms its default
adaptive per-step policy. Whole-box GPU reservation; CPU references overlap the main recording, so the
collection retains the **busy-box** grade. The executable and all 2747 kernel files are fingerprinted.

Ratios use raw per-request medians and their geometric mean; all requests, including the main recording's
30 one-token requests, remain in the gate.

| recording | requests | DFlash / MTP | DFlash / plain | f32 judgment |
|---|---:|---:|---:|---|
| opencode-session-int8 | 78 | 1.193081621 | 1.774903067 | PASS, no failures/skips |
| opencode-session | 21 | 1.226568536 | 2.849522724 | pending long-context references |
| opencode-session-2 | 4 | 1.319626438 | 2.844003544 | pending long-context references |

| remaining speed gate | DFlash / MTP | result |
|---|---:|---|
| golden | 1.058189106 | PASS |
| A4 / toolcall | 1.826279459 | PASS |
| llama-benchy tg128 | 1.382364965 | PASS |
| llama-benchy tg128 at depth32768 | 1.486807750 | PASS |
| llama-benchy pp4096 | 0.955104041 | **FAIL**, below 0.97 |

The main recording's >=1.10 speed bar passes, but **full G-speed fails**: pp4096 is **4.49% slower** than MTP.
Its three-round median is 1489.321 t/s versus 1559.328 for MTP (plain 1576.465). This corresponds to about
2750.25 versus 2626.77 ms for 4096 ids; about **42.23 ms** must be removed to reach the unchanged 3% limit.
The saved round-1 server logs place about 120 ms of the gap inside prompt processing (2629 versus 2510 ms),
about 8 ms in prefix snapshot storage (65 versus 57 ms), and less than 1 ms in cold reset. The existing
8-row staged tail projection is the next profiling target; attribution to it is an inference from that path
and the component profiles, not a new isolated measurement. No serving default changes.

Core correctness passes: the main recording's six speculative-arm/round comparisons and all six golden/A4
comparisons have **0 FAIL / 0 SKIPPED**. Actual candidate-head PP passes **39/39** (37 identical, two f32-judged
passes). **94 new references** are complete: 88 main, four PP (137 s), two corpus (74 s), all at eight threads
under the 69 GiB cap. The full numerical, selected seven checks and real-engine regression pass; the latter
runs 297.36 s without skipping. G0 passes 2703 common kernels: 2449 byte-identical, 254 equal section by section
after normalising only the tree path in build options, 44 additive drafter variants and zero removals.

Both control recordings still need f32 judgments. The complete missing-reference plan has **32 contexts /
26 groups, estimated 140.6 CPU hours**, excluding chunk/stream overhead. This is a planner estimate, not an
observed duration. That batch is held for the operator's budget choice; no CPU model or GPU job is left
running. C7 remains failed (4k draft 8.430 ms, iteration 1.559 plain steps); RTN acceptance failed and proper
AutoRound remains the operator-deferred follow-up. Open-work item12 stays open.

Evidence under `~/b70/logs/dflash-v2/`: `task8/main-0b5ef628/gates-after-core-refs.json`, all 27 replay arms and
nine complete corpus arms, `opencode-session-int8/judge-mid-refs.log` and `corpora/judge-after-short-refs.log`
inside that dataset, `pp-candidate-main-0b5ef628/judge-after-short-refs.log`, `g0-main-0b5ef628/comparison/`,
`reset-v7-engine.log`, the three CPU logs and `task8-final-ref-plan-v7/{plan.log,summary.json}`.

### On the card (measured 2026-10-27, card 0, branch `dflash-fix`, states `09c1cae8c97d-vs-01e9b51eca5b` / `e8f21f745c58-vs-01e9b51eca5b`; then branch `dflash-bars`, state `e89445916986-vs-0fb0cb914a37` - the operator's two bar rulings, rows 54-56 complete)

**`dflash-bars` (2026-10-27, against main 0fb0cb9):** G0 - sha the same 49 path-embedding `nb_*` only, bitwise PASS,
the suite's 9 failures = main's own 9 (memory_plan_box x 4, k2_prefill x 2, k2_prefix_gpu x 2, bolt_pf_kernels_test;
`dflash_engine_test`, `dflash_gpu_test`, `dflash_prefix_gpu_test` pass in it). r54.host / kernels / load / draft / d1
PASS, **r54.d1_ref FAIL** (D1 below: the sets now pass in 585 / 587 / 586 of 592; the walks and scores bars, unchanged,
fail in 1-4 drafts an arm). r55.host / kernels / engine (PF chunked 0.9998879 >= the ruled 0.9998) / gpu / serve /
decode PASS. r56.host / engine / prefix / pp / pp_serve PASS; D5's arms ran (below; the opencode recording does not
exist, so r56.d5_opencode SKIPs). r55.decode first failed on its own harness - `b70-decode --spec dflash` printed the
drafter's load line on stdout ahead of the ids (the ids were identical) - fixed (the line goes to stderr like the
target's) and re-run on the fixed head (`g0.build` redone in the same state); r56.pp first reused G0's one-card skip of
`pp_dflash_test` - `junit.py todo` now re-runs a skipped result - and then passed on both cards (splits 29 and 20:
ids, drafts, every verify's logits, the ring, the session and 3 hooked snapshots bitwise the one card's).

The `dflash-fix` table (before the rulings):

| stage | result | the numbers |
|---|---|---|
| r54.kernels | PASS | every `df_*` kernel at the drafter's shapes against the twin; `df_attn` now **bitwise** (its softmax exp is `df_exp`, the twin's `attn_exp` - the device's `exp` differed from glibc's by 1-2 fp32 ulps in ~45 % of the arguments, which outputs cancelling toward 0 turned into up to 45 bf16 ulps: the 2026-10-23 FAIL) |
| r54.load | PASS | int8 2,053,464,064 B / bf16 3,849,364,480 B = the formula; every read-back = the host form |
| r54.draft | PASS | 87 / 8 launches; at p 2053 rows cos 0.9999927 (int8) / 1.0000000 (bf16), sets and walks equal; the 32k draft vocabulary bitwise the full head; **the draft list 20.6 / 25.0 ms int8, 19.1 / 23.3 ms bf16** (host-timed replay, one fence - against spec 19's derived 0.17-0.2 plain steps) |
| r54.d1 (engine half) | PASS | 592 drafts an arm (37 dumps x 16 anchors, K 7, T 0); 15.5 ms (i8), 16.9 ms (bf16), 12.7 ms (i8 + 32k vocabulary) a draft list, host-timed |
| **r54.d1_ref (D1)** | **FAIL** | see D1 below |
| r55.engine | **FAIL** (PF only) | L ok (774 launches a verify at M 1..8, draft 87, projection 8); D3 (pre-§13 form) **bitwise** at k = 1..7 (logits, GDN slots, taps) at n 1998 and 2062; T1 bitwise; P (partial commit, K 7 and 4, every j) bitwise, 64 further ids equal; SR (the ring across reset + a prefix cache hit) bitwise; D2 on the card 0 of 160 ids differ at K 7 / 3 / 1; RF3 yes; memory 21.254 GB of 32.530 at max_len 16384 (drafter 2.116 GB). **PF: split at 1024 bitwise, decode path median cos 0.999824 (bar 0.999), chunks of 1000 median cos 0.9998879 against the bar 0.9999** |
| `dflash_gpu_test` (r55.gpu's test, run by G0's suite) | PASS | D2 through `EngineAdapter` on the 3 golden prompts + 36 A4 scenarios at K 7 / 3 / auto: **0 divergences**; ids an iteration K 7 4.10, K 3 3.01, auto 3.52; RF1 (a truncated burst continued) = plain; a seeded sampled K 7 run twice bitwise |
| r55.gpu (the stage), r55.decode, r55.serve, r56.engine, r56.prefix, r56.pp, r56.pp_serve, D5 | not run | blocked behind r55.engine (each needs it PASS) |
| G0 (D0) | sha FAIL (expected), bitwise PASS, suite 10 fails | sha: `df_attn` (the fix) + 49 `nb_*` that embed the tree's `-I` path (not this change); `g0.bitwise` 11 of 11 identical; the suite's other 9 failures (memory_plan_box x 4, k2_prefill x 2, k2_prefix_gpu x 2, bolt_pf_kernels_test) fail the same on main 01e9b51's build |

**D1 (spec 19 §6), measured (`dflash-fix`, the fixed 1.4e-3 tie window):**

| arm | drafts | top-16 sets (tie-aware) | scores cos >= bar | walk equal | worst cos | ties excused |
|---|---:|---:|---:|---:|---:|---:|
| drafter int8, head int8 (bar 0.999) | 592 | 359 | 591 | 591 | 0.9958614 | 30 |
| drafter bf16, head bf16 (bar 0.9999) | 592 | 369 | 590 | 588 | 0.9572936 | 23 |
| int8, 32k draft vocabulary (bar 0.999) | 592 | 353 | 591 | 591 | 0.9959722 | 25 |

The set misses are boundary swaps: over all 4144 draft rows of the bf16 arm, 295 rows have one; the swapped ids sit
0.012 (median) / 0.075 (max) logits from the reference's 16th logit (|v| ~ 14), while the engine's logits differ from
the reference's by 0.078 (median of each row's largest difference over its common candidates; p90 0.18) - 290 of the
295 swaps are inside their row's own logit difference (int8 arm: 322 rows, median 0.011 / max 0.093 from the edge,
logit difference median 0.076, 315 of 322 inside). The comparator's tie window was max(1e-3, 1e-4 |v|) ~ 1.4e-3.

**D1 with the noise-aware tie (operator, 2026-10-27; `dflash-bars`):** a set miss is a tie when the swapped ids sit
within max(1e-3, 1e-4 |v|, d_row) of the reference's 16th logit, d_row the row's own largest engine-vs-reference logit
difference over its common candidates (`tools/oracle/dflash_d1.py`). Scores and walks bars unchanged.

| arm | drafts | top-16 sets | scores cos >= bar | walk equal | worst cos | ties used (by d_row) |
|---|---:|---:|---:|---:|---:|---:|
| drafter int8, head int8 (bar 0.999) | 592 | 585 | 591 | 591 | 0.9958614 | 315 (285) |
| drafter bf16, head bf16 (bar 0.9999) | 592 | 587 | 590 | 588 | 0.9572936 | 290 (267) |
| int8, 32k draft vocabulary (bar 0.999) | 592 | 586 | 591 | 591 | 0.9959722 | 307 (282) |

The 18 set misses left are swaps just outside their row's difference (gap 0.017-0.093 against d_row 0.014-0.065; at
most 1.7 x). The walk / scores misses are the same drafts as before the ruling (selector choices 0.005-0.09 apart).
D1 stays FAIL on the bars as written: the operator's call (the rows' D2 - the losslessness - passes everywhere).
Since 87fc262 (operator, 2026-10-28) a set miss passes when its draft's walk is equal and its cosine meets the bar,
which covers all 18. That comparator has not been re-run on the box. On these numbers D1 would still FAIL on the walks
and scores (1 / 4 / 1 drafts).
The walk differences (4 bf16, 1 int8) are choices 0.005-0.09 apart in selector scores of magnitude 12-36. The engine rounds
every GEMV input to bf16 (spec 19 §10); plan 19b's lever is fp32-input GEMV variants. The bars are unchanged - the
operator's call.

**PF** (chunked bar 0.9998 since the operator's ruling of 2026-10-27; it was 0.9999): the window's worst rows are long32k positions 2091, 1564 / 1565, 987, 1051, 1212 - the positions
`prefill_split_test`'s header records as the model's own amplifiers of a rounding-sized perturbation (its own split
bar at l0 is median >= 0.9998) - and they move in every pair of paths (one call / chunks / decode) alike.

**D1 under the operator's three rulings (re-run 2026-11-01, plan 19f; engine half from campaign `checks-1101`
state `32d0f97beac0-vs-ce36d2d786ee`, reference half `tools/oracle/dflash_d1.py` at f8a7758 in tree spec19f):
PASS on all three arms.**

| arm | drafts | sets (tie / noise-aware) | scores cos | walk (or near-tie split) | worst cos | set misses passed on walk + cos | walk near-ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| int8 drafter, int8 head | 592 | 585 | 592 | 592 | 0.9999663 | 7 | 1 |
| bf16 drafter, bf16 head | 592 | 587 | 592 | 592 | 0.9999682 | 5 | 4 |
| int8, V' 32768 | 592 | 586 | 592 | 592 | 0.9999666 | 6 | 1 |

### int8 KV (spec 19v3a, 2026-11-13)

Implementation branch `spec19v3a-dflash-int8-kv` adds the M5..8 int8 writer/reduce/DPAS binaries and all-depth
verify dispatch. The operator requested only a quick final check. **No new speed measurements**: paired real-list
costs at 4096/32768, replay against default per-step MTP, tg128 at 32768 and decode at 120000 are pending (row 86).
Host planner: **246,528 tokens** at 32.530 GB, 1.5 GB reserve, target weights 16,812,371,968 B, int8 drafter
and candidate64 head; allocation at that context is untested. The policy costs and `--spec auto` are unchanged.
The spec's ~1.15–1.18x is derived, not a measured result. Focused check: five host tests, three GPU kernel tests
(no skips, 33.51 s on card 1), twelve replay-harness tests and two validation regressions passed. All twelve
new kernel commands are additive, existing commands unchanged. Full G0 binary comparison remains pending.
Evidence: `~/b70/logs/dflash-int8-v3a/{quick,quick-final}.log` (kernel check rc0); validation regression logs
are in the branch's local plan workspace.

### Round cost (spec 19 v3d)

**Task 1, verify at 1 / 2 / 4 / 8 rows by kernel family** (2026-11-13, branch `spec19v3d-dflash-round-cost`,
`probe_dflash_iter --verify-profile`, card 0, bf16 KV, int8 drafter, candidate head). **Busy box:** card 0 idle,
card 1 ran the drafter quant and CPU reference jobs ran. Per-launch timestamps (instrumented), the four widths
captured in turn inside each round after a warm-up; medians of three rounds of 16 calls. The families' excess sums to
the instrumented wall excess (7.170 / 7.171 ms) and is within 2.5 % of the uninstrumented interleaved pair's 7.354 ms,
so the attribution is trusted.

| family | 1 row | 2 rows | 4 rows | 8 rows | excess 8 - 1 | at 32768: 1 / 8 rows, excess | lever candidate (>= 0.5 ms) |
|---|---:|---:|---:|---:|---:|---:|---|
| GDN step (`gdn_step_slots`) | 0.734 | 1.045 | 1.679 | 3.091 | **2.356** | 0.746 / 3.142, 2.397 | **yes: Task 3** |
| gate\|\|up | 11.088 | 11.438 | 11.849 | 12.454 | **1.366** | 11.126 / 12.380, 1.255 | **yes: Task 4** |
| out_proj (GDN) | 1.615 | 1.777 | 2.159 | 2.814 | **1.199** | 1.597 / 2.785, 1.188 | **yes: Task 4** |
| attention (v2 at 1 row, `attn_verify_dpas` at 2+) | 0.706 | 1.156 | 1.479 | 1.779 | **1.073** | 4.578 / 7.159, 2.581 | by size yes; no Task 2-4 lever (see below) |
| a\|\|b (`gemv_bf16`) | 0.237 | 0.351 | 0.413 | 0.698 | 0.461 | 0.239 / 0.701, 0.462 | no (under the bar) |
| o_proj (FA) | 0.519 | 0.524 | 0.550 | 0.715 | 0.196 | 0.517 / 0.619, 0.101 | no |
| qkv\|\|z | 3.842 | 3.931 | 3.997 | 4.021 | 0.180 | 3.843 / 4.025, 0.182 | no |
| down | 5.423 | 5.358 | 5.419 | 5.532 | 0.109 | 5.428 / 5.543, 0.115 | no |
| residual fold | 0.241 | 0.275 | 0.334 | 0.324 | 0.083 | 0.249 / 0.330, 0.081 | no (Task 2) |
| attention reduce | 0.076 | 0.083 | 0.112 | 0.133 | 0.057 | 0.122 / 0.227, 0.105 | no |
| gated head | 0.086 | 0.092 | 0.100 | 0.121 | 0.034 | 0.088 / 0.118, 0.030 | no (Task 2) |
| q\|\|k\|\|v | 1.097 | 1.098 | 1.113 | 1.132 | 0.034 | 1.095 / 1.134, 0.040 | no |
| argmax | 0.007 | 0.010 | 0.018 | 0.034 | 0.027 | 0.007 / 0.034, 0.027 | no |
| gaps after the tap copies | 0.012 | 0.017 | 0.024 | 0.040 | 0.027 | 0.012 / 0.040, 0.028 | no (Task 2) |
| norm | 0.204 | 0.207 | 0.222 | 0.230 | 0.026 | 0.211 / 0.234, 0.023 | no (Task 2) |
| other gaps between launches | 0.801 | 0.812 | 0.824 | 0.823 | 0.021 | 0.810 / 0.828, 0.018 | no |
| attention prepare | 0.052 | 0.053 | 0.053 | 0.059 | 0.007 | 0.052 / 0.059, 0.007 | no |
| embedding | 0.009 | 0.009 | 0.009 | 0.009 | 0.000 | 0.009 / 0.009, 0.000 | no |
| head (int8, `gemv_i8w` / `gemm_sm_i8w`) | 2.211 | 2.149 | 2.161 | 2.199 | -0.011 | 2.208 / 2.199, -0.010 | no |
| SiLU (the S1 build at 2+) | 0.692 | 0.633 | 0.634 | 0.629 | -0.063 | 0.704 / 0.638, -0.066 | no (Task 2) |
| **sum (device + gaps)** | 29.661 | 31.015 | 33.155 | 36.832 | **7.171** | 33.658 / 42.200, 8.543 | |
| wall, instrumented | 29.784 | 31.103 | 33.261 | 36.955 | 7.170 | 33.981 / 42.512, 8.530 | |
| **wall, uninstrumented pair against 1 row** | 29.259 | 30.624 (1.048) | 32.816 (1.123) | 36.617 (**1.251**) | **7.354** | 33.605 / 42.461 (1.264), 8.856 | |

**Stop / continue: continue.** Four families clear 0.5 ms and make 6.0 of the 7.2 ms: the GDN step (2.36, Task 3),
gate\|\|up (1.37) and out_proj (1.20) at `gemm_sm` (Task 4), and the attention (1.07; 2.58 at 32768). **The glue is not
where the excess is:** fold, norm, SiLU, gated head and the tap copies together are +0.13 ms at 8 rows, under Task 2's
0.3 ms bar for any one of them, so Task 2 has no candidate. The head and argmax at 8 rows cost what they cost at 1 (no
Task 4 head lever). The attention's excess is 1-row v2 against 8-row DPAS over the same keys; it is no Task 2-4
lever and stays with the attention work (int8 KV, plan 19v3a). The instrumented 1-row list is 0.53 ms slower than
the uninstrumented one (the per-launch signals); the excess is not.

**Task 3, the GDN step at 8 rows** (same day, card 0, busy box). `probe_gdn_slots` (48 launches of one layer's step
over a 48-layer state, standalone, medians of 7 interleaved rounds) splits the step's excess: the row chain (the
same source without per-row slots at M = 8, against the 1-row slots build) **1.78 ms**, the 7 extra snapshot writes
(8 x 151 MB a verify) **1.35 ms**. The chain was barrier-bound: 20 work-group barriers a row (the two l2norm trees
and the kv / o band trees). The kept lever: the slots builds at M >= 2 take every row's l2norm before the recurrence
(one set of trees for all rows, spread over the work-group) and each band tree in registers (2 barriers a row), the
kv / o buffers double-buffered by row parity. Bitwise the per-row form (`gdn_step_slots_test`, 4 and 8 slots, 92
rows); only the 19 SPEC_SLOTS M >= 2 binaries change, every other binary is byte-identical. **Verify at 8 rows
36.59 -> 35.88 ms (1.250 -> 1.225 plain steps, -0.70 ms)**, four interleaved process pairs against the pre-lever
build (uninstrumented, each a median of 3 rounds of 16 calls). The snapshot stores in other forms, standalone
against the kept lever: block writes -0.02 ms, LSC L1UC_L3WB -0.02, L1S_L3WB -0.08, the uncached forms +8.7 to
+15 ms - none kept. **MTP's lists bind the same builds:** `mtp_verify_test` and `slots_test` pass, `mtp_gpu_test`
fails the same 2 near-tie prompts as main's kernels (all 60 f32-judge verdicts identical); MTP verify at 4 rows
(`probe_mtp_steps`, int8 head, depth 4096, four process pairs) **33.83 -> 33.34 ms (1.112 -> 1.100 x verify at 1
row)**, at 2 rows 31.66 -> 31.42 ms.

### Lookup inside DFlash (`--spec dflash-lookup`, spec 19 §13.1, plan 19f, measured 2026-11-01)

**Busy-box timing, one card (card 1), 8+ agents on the box (load 13-45); re-check on a quiet box.** b70-serve,
greedy, `--max-len 102400`, max_tokens 256 (stop at EOS), the server's decode-only t/s, median of 3 rotated rounds
(dflash, dlk), off / `--mtp auto` 1 round. Output: dflash-lookup identical to dflash 51 / 51 in every round.

| corpus | dlk / dflash (geomean) | median over prompts | prompts |
|---|---:|---:|---|
| edit / copy (tests/golden/edit-copy) | **1.247** | **1.33** | quote 1.27 / 1.42 / 1.30 / 1.40, reemit 1.69 / 1.40 / 1.36 / 1.42, rename 1.06 / 0.98 / 0.95 / 0.95 (8k / 32k / 64k / 99k) |
| A4 tool calls | **0.971** | - | lookup entered 0-2 times a request, ~0 accepted |
| golden | 1.000 | - | no lookup |

Lookup share (dflash-lookup, round 1): quote / reemit 231-244 of 256 ids from lookup (210 / 210 drafts accepted);
rename 78-121 of 256 (acceptance ~0.5-0.6: the edit call's newString diverges at each renamed local). Context
(not like for like on edit / copy: off and mtp ran the set before its asks were rewritten): `--mtp auto` / dflash
1.49 edit / copy, 1.22 golden, 0.92 A4; on the quote prompts (32k / 99k identical across the two set versions,
8k / 64k one grep line apart) mtp 54.1 / 44.9 / 61.4 / 38.6 t/s against
dlk 51.7 / 38.2 / 72.7 / 29.3 (32k / 64k / 8k / 99k) - `--mtp auto` stays ahead of both DFlash modes at depth.
Through EngineAdapter (dflash_lookup_gpu_test, card 0): reemit-8k (numbered-read form) 21.7 against 25.1 ms an id,
quote-8k 14.5 against 19.6, quote-32k 20.3 against 25.2.

### P0, acceptance (measured, CPU reference, teacher-forced greedy)

| corpus | E_1 | E_3 | E_7 (tokens per verify at K = 7) |
|---|---:|---:|---:|
| A4 tool calls (34 scenarios, 2409 anchors) | 1.99 | 3.88 | **7.21** |
| golden code | 1.88 | 3.60 | 6.15 |
| golden prose | 1.73 | 2.19 | 2.20 |
| golden cjk | 1.38 | 1.50 | 1.54 |

int8 and the published 4-bit drafter equal bf16 to 0.01; a ranked 32k draft vocabulary looks free (7.11 vs 7.09, ranked
in-sample), by id it loses up to 1.4 at K = 7.

### Projection (derived: verify at M = 5..8 extrapolated at +0.22 a row, the draft ~0.17-0.2 plain steps)

| corpus | DFlash best K | DFlash tokens / step-time | `--mtp auto` | DFlash vs MTP |
|---|---|---:|---:|---:|
| A4 tool calls | 7 | 2.56 | 1.66 | +54 % |
| golden code | 7 | 2.18 | 1.32 | +65 % |
| golden prose | 2 | 1.24 | 1.06 | +17 % |

### D5, speed (bar: `--spec dflash` >= 1.10 x `--mtp auto` on the opencode recording, no loss > 3 % elsewhere)

**Verdict (2026-10-27): DFlash stays an option, `--mtp auto` stays the Qwen3.8 default.** The bar's own measurement
(the opencode recording) does not exist yet, and DFlash already loses more than 3 % to `--mtp auto` elsewhere: golden
prose / cjk, every llama-benchy decode row (12-18 %) and every prefill row (3-21 %: the drafter's tap projection in the
prefill tail). It wins on the A4 tool calls by 3 % - short of 10 %. Measured on card 0 (both cards for `--pp 2`),
state `e89445916986-vs-0fb0cb914a37`, greedy, every arm's output identical to `--spec off` (39 / 39 each round).

| what | configuration | number | stage |
|---|---|---|---|
| one draft block | the draft list, host-timed replay (r54.draft) | 20.6 / 25.0 ms int8 (~0.7-0.85 of a 29.2 ms plain step, against the derived 0.17) | r54.draft |
| A4 + golden through b70-serve | `--spec off` / `--mtp auto` / `--spec dflash` (auto K) / `--spec dflash --spec-max 3`, `--max-len 16384`, int8 head, bf16 KV, greedy, max_tokens 256, 3 rotated rounds; gen t/s geomean | dflash / mtp: **A4 1.033**, golden 0.791 (code 61.8 vs 60.0 t/s, prose 32.4 vs 44.5, cjk 28.9 vs 43.7); dflash3 / mtp 0.928 / 0.758; dflash / off 1.974 / 1.127 | r56.d5_a4 |
| the draft vocabulary | `--spec dflash` vs `--draft-vocab 32k` ranked (in-sample) and by id | ranked / full: A4 1.043, golden 1.022; by id / full: A4 0.916, golden 1.022; ids identical | r56.d5_dv |
| llama-benchy | row 41's matrix A at concurrency 1 and matrix D, `--max-len 40960`, off / mtp auto / dflash, 3 rounds in a Latin square, medians | tg128: off 33.9, mtp 47.9, dflash 42.1 (0.878 x mtp); tg128 @ d4096: 33.5 / 51.6 / 42.9 (0.831); tg128 @ d32768: 29.5 / 40.7 / 33.4 (0.822); pp512 628 / 603 / 478 (0.792); pp4096 1246 / 1231 / 1021 (0.829); pp512 @ d4096 1362 / 1347 / 1124 (0.835); pp512 @ d32768 1287 / 1282 / 1248 (0.973) | r56.d5_benchy |
| **the opencode recording (the bar)** | `replay_ab.sh`, prefix cache on, max_len 131072, off / mtp auto / dflash, 3 rotated rounds | **not measured - the recording does not exist yet** (SKIP, missing data) | r56.d5_opencode |
| two cards | `--pp 2 --mtp auto` vs `--pp 2 --spec dflash`, A4 + golden, 3 rounds | dflash / mtp: A4 1.033, golden 0.793 (code 68.4 vs 65.5, prose 35.1 vs 47.7, cjk 30.3 vs 46.7); ids identical | r56.d5_pp |
| `--spec dflash` vs `--spec off` on one card (r55.serve, 1 round) | A4 + golden, max_tokens 192 | wall t/s 1.094 (A4) / 1.061 (golden); seeded sampling reproducible | r55.serve |
| Ornith's DFlash (v1) drafter | the same rows on Ornith | not built (a full-attention drafter layer and a 16-row block: its own stage) | - |

### Memory (derived)

| term | bytes |
|---|---|
| the int8 drafter's weights + buffers at max_len 16384 | 2,053,464,064 + ~62.3 MB |
| the engine's verify slots (7 x one GDN state) and the prefill tap staging | 1.06 GB, 105 MB |
| `--max-len auto`, one 32 GB card, 17 GB target, int8 drafter | 139520 (190208 without the drafter) |
| a prefix-cache snapshot with the drafter (decision 5) | the state + 41,943,040 B of ring (+25 %) |
| `--pp 2`: device 1's embedding replica, each device's 8-slot verify slots | 2.54 GB; ~0.97-0.99 GB a device |

### Correctness (host, measured on the Mac; the card's are pending)

| gate | result |
|---|---|
| D4, sampling exact (`dflash_accept_test`) | 10^6 seeded draws: chi-square p 0.59 / 0.23 unfiltered, 0.29 / 0.91 filtered; no accepted draft outside the candidates |
| the ring across a prefix cache hit (`dflash_prefix_test`) | a turn served as a prefix cache hit from its predecessor's request-end snapshot leaves the ring equal to the unbroken session's; a ring-less snapshot (19c's) is caught |
| batching (`dflash_scheduler_test`) | 4 requests on 4 slots: ids = the plain scheduler's, each request's speculative record = its record alone |
| `--pp 2` adapter (`pp_dflash_serve_test`) | greedy at K 7 / 3 / auto = plain; seeded sampling reproducible; a prefix cache hit turn drafts as the unbroken one |
| D2 / D3 / ring / --pp 2 bitwise on the card | measured 2026-10-27 - PASS (r55.engine, r55.gpu, r55.decode, r55.serve, r56.prefix, r56.pp, r56.pp_serve; "On the card" above) |

## Tensor parallel (spec 17)

Qwen3.8, `b70-decode --parallel tp` (plan 17b): every layer's linears split over the two B70s, the two halves added
after each row-parallel linear by P2P writes into the other card's VRAM (spec 17a's route). Plain greedy decode, int8
head, bf16 KV, tg 256 after the depth; one card prefills to the depth, TP ingests it by decode replays (17b has no
prefill). 2026-11-10, branch spec17b-tp, `tools/box_validate/interleave.sh`, median of 3 after a warm-up, order
reversed every other round; box grade ITERATE (no container, two idle desktop DRM holders, load 1.2-2.2; the 3l agent's
jobs waited behind the lock). Spreads were under 0.1 %.

| arm | depth 4096 t/s | ms/token | x one card | depth 16384 t/s | ms/token | x one card |
|---|---|---|---|---|---|---|
| one card (`--parallel 1`) | 33.49 | 29.86 | 1.00 | 31.65 | 31.60 | 1.00 |
| tp, slices crossing (bitwise one card) | 49.99 | 20.01 | **1.49** | 47.24 | 21.17 | **1.49** |
| tp, folded crossing (the default since D1) | 53.95 | 18.54 | **1.61** | 50.80 | 19.69 | **1.61** |

The operator's D1-dense bar (plain TP >= 1.35x one card) holds for both crossings. Spec 17a derived 1.68x (folded,
unfused 1.62x); measured 1.61x - the send is its own launch (128 a token) and the wait is fused into the fold. The
slices crossing (bitwise one card) costs 7.3 % against folded: twice the bytes a crossing and half the work-groups on
out_proj / o_proj / down. `--spec auto` under TP is 17d's bar (>= 1.25x against one card's `--spec auto`).

### Prefill under tensor parallel (spec 17c)

`b70-decode --bench --prefill-length N --tg 32 --max-len 32768`, `--parallel tp` (plan 17c: both cards' prefill walks
as a wavefront of two chunks, each card's half of every row-parallel linear crossing on its copy engine while the other
chunk computes; fp32 crossings, the folded decode after it) against `--parallel 1`. l0 backend, int8 head, bf16 KV,
chunk cap 2048. 2026-11-10, branch spec17c-tp, `tools/box_validate/interleave.sh`, median of 3 after a warm-up, order
reversed every other round; box grade ITERATE (shared with the 3l agent between our locked runs; load 1.2-10 at the
start, from a CPU reference job that had just ended). Spreads were under 0.3 %.

| prompt ids | one card pp t/s | tp pp t/s | x one card | tg 32 after it: one / tp t/s |
|---|---|---|---|---|
| 2048 | 1746.32 | 2479.27 | **1.42** | 33.79 / 54.74 |
| 16384 | 1524.28 | 2672.68 | **1.75** | 31.64 / 50.24 |

The crossing is 40 MiB of fp32 each way at 2048 rows, 4.35 ms with both directions at once (~9.6 GB/s a direction:
docs/probes/probe-tp-prefill-2026-11-10.md). Without the crossings (a timing seam, wrong numbers) the walls are 714 /
5756 ms against 814 / 6159 with them: the copy engines hide all but ~100 / ~400 ms; the rest of the gap to 2x is a
card's compute, 22 % above half of one card's at 2048 ids (two 1024-row chunks) and 7 % at 16384.

### Speculative decoding under tensor parallel (spec 17d)

`b70-serve --spec auto --prefix-cache-gb 0`, `--parallel tp` (plan 17d: the MTP head whole on each card, per-card
verify lists at M = 2..4, AdaptiveK over the TP cost table) against `--parallel 1`, one request at a time, 256 ids
after a prompt of the repo's docs (tools/box_validate/depth_prompts.py), decode-only t/s (`gen_tps`: the server's own
record, first id excluded), a warm-up request a server, 3 rounds in rotated order, medians. Int8 head, bf16 KV,
`--max-len auto`. 2026-11-10, branch spec17d-tp; box grade ITERATE (a CPU reference job ran beside some rounds).

| depth (prompt ids) | one card `--spec auto` t/s | tp `--spec auto` t/s | x one card | acceptance one / tp |
|---|---|---|---|---|
| 4096 (4063) | 62.12 | 90.39 | **1.455** | (both: K 3 throughout) |
| 32768 (31821) | 62.36 | 91.68 | **1.470** | 0.707 / 0.695 |

Rounds: one card 62.21 / 62.12 / 62.01 and 62.36 / 62.21 / 62.36, tp 90.39 / 90.39 / 90.51 and 91.68 / 91.66 / 91.78 (spread
under 0.3 %); load average 2.6 - 9 (the box otherwise idle; a first run beside a CPU reference job, prompts of 5199 /
40046 ids, gave 1.451x / 1.458x).

The TP cost table behind AdaptiveK (tp_mtp_test `cost`, medians of 5 interleaved rounds of 16): at depth 4096 the K = 0
iteration 18.93 ms, verify M = 2 / 3 / 4 x 1.09 / 1.13 / 1.20, a draft 3.01 ms (0.16); at 32768 21.65 ms, x 1.01 /
1.07 / 1.13, a draft 3.26 ms (0.15). One card's: x 1.04 / 1.09 / 1.13 and 0.13 a draft - TP's verify rows pay their
M-row crossings, its draft reads the whole head's layer on each card and half the int8 head.

## Ornith 1.5 MoE (spec 15)

**Prefill after the MoE prefill plan (2026-11-12, one card, card 0 while card 1 ran another agent's jobs and a CPU job ran, median of 3, `b70-decode --bench --max-len auto`, 4096-row chunks):** 4k **8,080** t/s (`l0`, bf16 prompt pass) / **8,518** (`l0-int8`, W4A8 experts); 16k 6,800 / 7,059. The same run's main (7cfbf058, 2048-row chunks): 7,578 / 5,460 at 4k, 6,509 / 4,869 at 16k. Record and per-phase times: [probes/probe-moe-prefill-2026-11-12.md](probes/probe-moe-prefill-2026-11-12.md), "Task 8". The prefill numbers in the next paragraph predate it.

**Measured on one card (2026-11-11, main, idle box between validation stages, median of 3):** prefill 3,362 / 3,259 t/s at 4k / 16k (bf16 prompt pass), 3,937 / 3,676 t/s (`--prefill-backend l0-int8`); plain decode 142.8 t/s at depth 4096 (`b70-decode --bench`); through `b70-serve` with MTP at the full 262,144 context, bf16 KV (`--max-len auto`, 27.3 GB planned of 32.5), one request, a 2,820-token prompt and 512 generated tokens (`tools/box_validate/slots_serve_bench.sh`): **234.9 t/s**, first token 1.28 s cold / 0.59 s as a prefix cache hit. The rows below predate these numbers.

**Ornith's MTP cost table (2026-11-12, one card, `probe_mtp_steps` at depth 4096, 64 calls x 9 rounds, `tools/gpu_lock.sh card`; two int8 runs agree within 0.01; r16.cost's commands, run directly):** in units of one plain step (verify M = 1 is 7.88 ms with the int8 head, 8.73 ms with the bf16 head), verify M = 1..4 and draft k = 1..3:

| lm_head | verify M = 2 / 3 / 4 | draft k = 1 / 2 / 3 |
|---|---|---|
| int8 | 1.15 / 1.34 / 1.52 (9.05 / 10.61 / 12.03 ms) | 0.19 / 0.38 / 0.57 (1.52 / 3.00 / 4.49 ms) |
| bf16 | 1.15 / 1.30 / 1.48 | 0.28 / 0.56 / 0.84 |

Qwen3.8's int8 table is 1.04 / 1.09 / 1.13 and 0.13 / 0.26 / 0.38: Ornith's verify rows climb about four times as fast (a MoE layer reads up to 8 M distinct experts). **The table does not win:** `b70-serve --spec auto` with it wired in (after) against the same server on Qwen3.8's table (before), interleaved over 3 rounds, golden + tool-call prompts, 256 ids, `--max-len 16384`, ids identical, median decode t/s: golden prose / code / cjk 246.0 / 252.0 / 210.8 against 269.9 / 278.3 / 237.0 (geomean 0.90), the 36 tool-call prompts geomean 0.94 (about 185-240 t/s). The step-K policy drafts less under the dearer measured rows and loses more in accepted ids than it saves. So Ornith keeps Qwen3.8's table and K up to 3 (the startup note says so); the measured rows are recorded here, not wired. Open: why the probe's rows overstate the in-server cost (the probe times verify + commit one call at a time). Ornith 1.5 35B-A3B (`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`,
the int4 checkpoint, spec 15 §13) was built without the card (specs 15c-15e); this section is the
record's skeleton (plan 15e Task 3 Step 3), each cell filled from the box-validation-queue stage
named beside it. Every row will state the card, the clocks if known, the quant and the `lm_head`
form (plan 15e Review Focus 5).

| what | configuration | number | stage |
|---|---|---|---|
| decode at depth 4k / 32k / 128k / 250k | one B70, int8 head, `--max-len auto` | pending | r10.speed |
| prefill pp4096 / pp32768 / pp130816 | one B70, l0-int8 and l0, int8 head | pending | r13.speed |
| A4 tool calls, scenarios matching the reference's first call | Ornith's own set (36), the int4 checkpoint's CPU reference (`ornith_ref.py`), 192 ids; l0-int8 bf16 / int8 head, l0 | pending (Qwen3.8: 25/36) | r16.a4 |
| prefix caching C2 | `prefix_gpu_test` l0-int8, l0, l0-int8 `--mtp 3` | pending | r16.prefix |
| passkey | 120k (l0-int8, l0), 250k (l0-int8, max_len 262144) | pending | r16.passkey |
| MTP `--mtp off / 1 / 3 / auto` | b70-serve, golden + A4 prompts, 256 ids | auto on Qwen3.8's table 269.9 / 278.3 / 237.0 t/s (golden prose / code / cjk, 2026-11-12, `--max-len 16384`); the off / 1 / 3 arms and the f32 judge not run | r16.mtp_rows (partial) |
| llama-benchy pp4096 tg256 depth 1 | b70-serve, `--no-cache --exact-tg --latency-mode generation` | pending | r16.benchy |
| llama-benchy prefix caching, follow-up after 4k / 16k / 32k | cache on vs off | pending | r16.benchy |

## K2-Horizon (spec 18)

**Nothing here is measured yet.** K2-Horizon MoVA 36B-A4B
(`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ`) runs its own engine (specs 18b-18e,
built without the card). The skeleton of plan 18d Task 3's record; the configurations differ from
the vLLM baseline's and every row says how (spec 18 Review Focus 4).

| what | configuration | number | stage |
|---|---|---|---|
| decode tg256 at depth 4096 / 16384 / 32768 | one B70, bf16 and int8 head, bf16 KV | pending (roofline ~190 t/s, derived) | r14.speed |
| prefill pp4096 / pp16384 / pp32768 | one B70, l0, flash and eager attention | pending (~3,400 t/s at 4096, derived) | r15.speed |
| int8 KV against bf16 KV | depth 4096 / 32768 / 65536 | pending | r21.speed |
| A4 tool calls | K2's own set (xml, reasoning_effort low, 512 ids) against 18a's reference; bf16 / int8 head | pending (no bar) | r25.a4 |
| passkey 5 / 50 / 95 % | 32k and the bf16-KV auto length (bf16 KV); the int8-KV ceiling | pending | r25.passkey, r21.passkey |
| llama-benchy pp4096 tg256 depth 1 | b70-serve on ONE card, 32k bf16 KV; int8 KV at auto | pending | r25.benchy |
| llama-benchy prefix caching, follow-up after 4k / 16k / 32k | cache on vs off | pending | r25.benchy |
| **vLLM baseline** | **two** B70s, PP = 2 (layers 25 / 23), fp8 KV, 390,016-token context | **44.43 t/s decode** (measured 2026-09-16, spec 18 §7) | - |

## Qwen3.8-Flash-Next (spec 21)

**Nothing here is measured yet.** Qwen3.8-Flash-Next (`qwen4_exp`) runs its own engine (specs 21b-21e, built
without the card) and only **truncated** (`--layers N`) until spec 22's expert-offload tier runs the whole
model: every row below is on Intel's interim checkpoint (`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`) cut to the
layers that fit, or on the synthetic checkpoints (4 real-width layers), and says which. The whole model's rows
(decode and prefill speed, A4, passkey, llama-benchy, the engine's MTP acceptance) are spec 22's.

| what | configuration | number | stage |
|---|---|---|---|
| prefill pp4096 / pp32768 | Intel's, `--layers 18` one card / `37` two (`--pp 2`), l0, int8 head; flash vs eager at 32768 | pending | r33.speed |
| prefill pp131072 | Intel's, `--layers auto` (the planner's N) | pending | r33.speed |
| the PLE gather from host USM | us a token, the implied bus rate, one card / the other / both | pending (spec 22 P0's first number) | r32.ple_rate |
| MTP verify cost | M = 1..4 against a plain step, ms; a draft step; synthetic `--layers 4` and Intel's `--layers 18` | pending (launches 86-91 at 4 layers, 790-795 at 48, derived) | r34.mtp |
| MTP verify: experts a layer | the union over the K + 1 = 4 verify rows (spec 22 P0.6's MTP term) | pending | r34.mtp |
| MTP acceptance by depth, K <= 3 | the CPU reference on Intel's checkpoint, teacher-forced greedy, the engine's RTN head; `pre_fc_norm_hidden` single / per stream (decision 4), selection reused / fresh (decision 5) | pending (the engine's is spec 22's) | r34.accept |
| prefix-cache snapshot | bytes and a prefix cache hit's continuation, 48 layers | 115.7 MB state + 25,344 B a position (+ 21 KB / 2,112 B with the head; derived); prefix cache hit bitwise pending | r34.snapshot |

## Nemotron 3.5 Bolt (spec 25)

**Nothing here is measured on a card yet.** Bolt (`nemotron_h_omni` over `nemotron_h_puzzle`, our int4 g64 export
`urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ`, 41 GB on disk) runs its own engine on two cards (spec 25c's decode, 25d's prefill, 25e's serving; no `nb_*` binary has been
compiled by ocloc). Weights 40.469 GB with the int8 head (derived, spec 25 §14); split 48: 19.93 / 20.54 GB; 262144 positions
fit two cards (24.17 / 24.23 GB of 32.53 with the scratches, derived). Decode reads ~5.5 GB a token (derived, spec 25 §3's descriptor sum) plus 8,192 B of KV a
position and 0.336 GB of fp32 SSM state (derived: 40 blocks x 4 MiB read and written; spec 25 Decision 2; ESTIMATED as a share of the step); 653 launches a token (533 with every arm). SergiioB's vLLM 0.31 XPU numbers on two B70
at TP 2 are PUBLISHED, his card, one self-reported run (his flags: fp8 KV, `--block-size 64`, `--enable-prefix-caching
--mamba-cache-mode align`), never reproduced here.

| what | configuration | number | stage |
|---|---|---|---|
| decode at depth 886 / 8192 / 32000 / 65536 | SergiioB's vLLM 0.31 XPU, TP 2, fp8 KV (PUBLISHED) | 60.5 / 50.7 / 33.5 / 23.0 t/s | - |
| prefill at 886 / 8192 / 32000 / 65536 prompt tokens | the same (PUBLISHED) | 2925 / 2014 / 871 / 486 t/s | - |
| A4 through b70-serve vs the g128 / BF16 references | 36 scenarios, greedy, up to 2048 new ids, `medium_effort` + thinking on (25a ruling 4); `score.py --format qwen-think` | pending (B5's engine row; the reference's parse failures counted; 0 where the scorer finds a call is the bar PROPOSED) | r62.a4 |
| the agentic transcript's prefix-cache hit rate | `tests/golden/prompts/bolt_agentic.json` as a live loop, three arms (default, `truncate_history_thinking` false, cache off) | pending; no bar - the miss at each new user message (`truncate_history_thinking`) and at a call whose arguments the model wrote unsorted is recorded with its divergence token | r62.agentic |
| snapshot sizes | state 170,229,760 B (167,772,160 B of fp32 SSM states + 2,457,600 B of conv columns; 86,343,680 B with `B70_BOLT_SSM=bf16`), KV 8,192 B a position | derived, asserted by `bolt_snapshot_test` (host, measured on the Mac) and `bolt_snapshot_gpu_test` (card) | r62.snap_synth, r62.snapshot |

## LiLiCorr on Bolt (spec 26)

**Nothing here is measured on a card yet** (queue rows 63-65). NVIDIA's LiLiCorr drafter (a DFlash 2 backbone with sinks and a
1024 window + a candidate-lattice head; 1,112,803,882 parameters, measured from the checkpoint on the Mac) drafts up to 7 ids
a block on device 1; Bolt verifies them at M = K + 1 <= 8 rows (spec 26c). The costs are DERIVED (spec 26 §11, depth 4096,
in units of one plain Bolt step of 5.524 GB): the verify at M = 1..8 with the plain M-row MoE 1.000 / 1.193 / 1.387 / 1.580 /
1.773 / 1.966 / 2.160 / 2.353, with the union-aware MoE (each expert block read once a verify; u ESTIMATED linear from 9 at
M 1 to 30 at M 8 - row 63's r63.union measures it) 1.000 / 1.085 / 1.169 / 1.254 / 1.339 / 1.423 / 1.508 / 1.593; one
drafter block 0.500 (exact) / 0.306 (int8) / 0.214 (int4 g64). The expectation (ESTIMATED: SPEED-Bench's PUBLISHED
acceptance at K 7 on the 24GB target, the int8 drafter, the union MoE): coding 5.26 tokens a verify -> ~2.8x plain,
overall 4.30 -> ~2.3x, writing 3.41 -> ~1.8x (spec 26 §11).

| what | configuration | number | stage |
|---|---|---|---|
| tokens a verify (D5's acceptance) per K 1..7 and AdaptiveK | `bolt_dflash_{int8,exact,int4}_test`: the golden short prose / short code prompts and the agentic transcript, greedy | pending | r65.d2 |
| llama-benchy pp4096 tg256 depth 1; decode at depth 4k / 16k | b70-serve two cards, `--spec off` / `lookup` / `dflash` (the engine's int8 drafter and union MoE; the formats' and the MoE forms' A/B is r63.cost x r65.d2 and `bolt_moe_union_test --time`), `--no-cache --exact-tg --latency-mode generation` | pending | r65.benchy (opt-in) |
| the agentic transcript as a live loop | the same arms, greedy and T 1.0 / top-p 0.95; per request the mean accepted length | pending - D5's rule: `dflash` becomes Bolt's `--spec auto` only if it beats `off` and `lookup` by >= 10 % here and loses nowhere by more than 3 % | r65.benchy (opt-in) |
| the verify per M, the block per format | `probe_draft_cost --shapes bolt-lilicorr` | pending (replaces the derived rows above in `MtpCost::bolt_lilicorr` / `bolt_verify`) | r63.cost |
| memory with `--spec dflash` | the verify slots 1,174,405,120 B over the 40 Mamba blocks (7 x 4 MiB each), the verify scratch ~14.8 MB a card, the drafter 1.30 GB (int8, at 262144) + the replica 1.07 GB + the prefill staging 50 MB on device 1; `--max-len auto` 262144 in every format | derived (`bolt_spec_plan_test`, measured on the Mac) | r65.serve's memory lines |

## Correctness

Numerics are gated, not eyeballed. The full protocol is in
[14-golden-gate.md](14-golden-gate.md); what the benchmark rows rest on:

| gate | what it proves |
|---|---|
| `golden_gate_test` | Decode ingest, one id per replay, against a CPU oracle: **93/93 determined rows exact**, 3 undetermined and all inside the oracle's own argmax set. |
| `prefill_gate_test` | One `Engine::prefill` per prompt, same oracle, same tally. |
| `prefill_backend_equivalence_test` | The Level Zero and sycl-tla prefill paths agree **bitwise**: 0 words differing, 0 sign-of-zero differences, on all five case families. |
| `replay_determinism_test` | 774 kernels, 19 modules captured; 8 tokens by 3 runs bitwise identical from reset. |
| `*_i8head_test` (spec 9) | The same golden, prefill and replay gates with `--lm-head int8`: 93/93, bitwise replay. |
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
[12-kernels.md, "How the prefill kernels got here"](12-kernels.md#how-the-prefill-kernels-got-here).

**A green gate is not evidence unless the run says which kernel it dispatched.**
The probes in this repo carry the same rule: the W4A8 probe proves twice, once
by asking the driver the name of the kernel handle the launcher received and
once by having the kernel write a signature word from the device.

## Work that was measured and rejected

Negative results with a mechanism, kept because they saved more time than the
wins did.

| what | why it was rejected |
|---|---|
| **Fusing the int4 dequant into the GEMM** | Both variants came out **bitwise identical** to the two-pass path over all 71,303,168 outputs, and both were slower. The dequant pass already runs at about 546 GB/s, roughly 90% of hardware peak, so there is only 0.826 ms in it at this shape; moving the same arithmetic into the mainloop costs 2.759 ms even when issued exactly once per weight. [Details](probes/probe-fused-dequant-2026-09-22.md). |
| **W4A8 on the mixed 4-bit by 8-bit DPAS** | 1.017x the bf16 two-pass control, 0.965x once the activation quantiser it needs is counted, against a bar of 1.53x. The per-group rescale costs 3.0 to 3.4 of the 5.440 ms; the matrix math it rescales costs 1.99 ms. Separately, **2.79% relative L2 error** against the bf16 result, with a worst-row cosine of 0.9976, driven by activation outliers. [Details](probes/probe-w4a8-2026-09-23.md). |
| **Removing barriers from the GDN triangular solve** | The design assumed synchronisation dominated that kernel. Deleting 64 of 128 barriers bought 1.7 ms of a 73.5 ms row, so the premise was wrong by about **18x**. Both barrier-free variants were bit-identical and both were slower. The row is the serialised recurrence itself. [Details](12-kernels.md#how-the-prefill-kernels-got-here). |
| **bf16 intermediate buffers** | Halving the bytes changed nothing: `ocloc` exposes no 16-bit block write wider than 8 rows, so the epilogue is bound by store message count rather than by bytes. |
| **oneDNN's fused int4 W4A16 matmul** | Measured 98.28 TFLOP/s on gate‖up at M=2048, below the 115.7 TFLOP/s an own in-kernel int4 GEMM would need to beat the two-pass path. Also failed a bitwise identity extraction against our dequant oracle on 39.6% of weights, every mismatch a 1-ULP bf16 neighbour. [Details](08-decode-vs-prefill.md#how-prefill-got-here). |

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
tools/bench_decode.sh --prefill-length 4096  # the prefill row
tools/bench_decode.sh --prefill-length 4096 --tg 256 --runs 3
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

### Combined int8 acceptance after H1/B4 (2026-11-12)

On card0, the int8-body H1/B4/candidate64/exact-rerank arm matches the original int8/full-head baseline on
**all 35 corpus/K acceptance rows, gap zero**. Its cached-tap mode first reproduced all **4256 original draft IDs
exactly**. The earlier interrupted 2463-row run remains archived; `calibration-status.json` now records the
completed calibration. Evidence: `~/b70/logs/dflash-v2/resume-accept-card0.log` (DONE rc0) and
`task7b/h1b4-candidate-int8/{calibration-status.json,acceptance.json}`. The int8 cost table is already refreshed;
the full int8 H1/B4 numerical check subsequently passes (`resume-numerical-card0.log`, DONE rc0). Forced-GV
projection has zero differences, ring cosine is 0.9999978, and the existing aggregate f32-set criterion passes
4 misses versus 8 for forced GV. The individual BF16-head diagnostic (3 versus 2) remains recorded; no bar changed.

Task 8 is collecting on card1 (`task8-card1-continue.log`): the complete first rotation on card0 (78 requests per
arm) was checked against the original server logs and recording before reuse. Binary SHA remains
`9afaed8f866f1e7ba824ce97485ae78353a237d4b25bc6202535cd4ddb9c2f03`; each arm records its card. The resumed
CPU guard measured 19.59–20.07 available threads out of 44. Final speed and serving correctness gates remain
pending. This does not promote RTN int4 or change the speculation default.
