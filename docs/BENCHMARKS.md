# B70 image benchmarks

Latest controlled prefill comparison: [2026-09-20 parity investigation](prefill-parity-2026-09-20.md)
(same checkpoint/IDs/chunk, current vLLM trace, native replay experiment).

Image names: `p<python>-t<torch>-vxkp<kernels patch level>`.

Serve:

```bash
IMAGE=vllm-xpu-env-next-p314-t213-vxkp0
MODEL=palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4

docker run --rm -it \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" "$MODEL" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 30k --kv-cache-dtype auto --max-num-seqs 2 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --trust-remote-code --enable-prefix-caching \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'
```

Bench:

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 8192 --tg 256 --concurrency 1 2 --depth 1 2 \
  --no-cache --exact-tg --latency-mode generation
```

## MXFP4 - olka-fi/Ornith-1.0-35B-MXFP4

| image                                         | py   | torch              | kernels                       | pp8192 d2 | tg256 d2 c1 | tg256 d2 c2 |
|-----------------------------------------------|------|--------------------|-------------------------------|-----------|-------------|-------------|
| vllm-xpu-env-next                             | 3.12 | 2.13               | wheel 0.12                    | 9080      | 67.94       | -           |
| p314-t213-vxkp0                               | 3.14 | 2.13               | src, 00                       | 8057      | 71.58       | 105.95      |
| p314-t214-vxkp0                               | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 7302      | 50.22       | 78.46       |
| p314-t214-vxkp0  + explicit Inductor override | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 7735      | 72.31       | 104.16      |
| p314-t214-vxkp12 + explicit Inductor override | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 7993      | 68.35       | 98.81       |
| p314-t215-vxkp13                              | 3.14 | 2.15.0.dev20260812 | src, 00-13                    | 7548      | 48.10       | 75.31       |

## GPTQ-Int4 - palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4

| image                                         | py   | torch              | kernels                       | pp8192 d2 | tg256 d2 c1 | tg256 d2 c2 |
|-----------------------------------------------|------|--------------------|-------------------------------|-----------|-------------|-------------|
| vllm-xpu-env-next                             | 3.12 | 2.13               | wheel 0.12                    | 9561      | 69.07       | -           |
| p314-t213-vxkp0                               | 3.14 | 2.13               | src, 00                       | 7834      | 72.42       | 104.95      |
| p314-t214-vxkp0                               | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 7517      | 51.13       | 79.43       |
| p314-t214-vxkp0  + explicit Inductor override | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 7892      | 73.24       | 105.45      |
| p314-t214-vxkp12 + explicit Inductor override | 3.14 | 2.14.0             | src, 00 (SYCL-TLA `87f68506`) | 8489      | 70.86       | 104.30      |
| p314-t215-vxkp13                              | 3.14 | 2.15.0.dev20260812 | src, 00-13                    | 7860      | 49.48       | 78.34       |

## Dense 4-bit Qwen3.8-27B - two quantizations

Own serve/bench parameters - `--max-model-len 16k`, `pp4096`, `depth 1`,
`concurrency 1` - so these numbers do not compare to the MoE tables above.

```bash

docker run --rm -it \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu-env-next-p314-t214-vxkp0 Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ \
  --served-model-name Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 128k --kv-cache-dtype auto --max-num-seqs 2 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --trust-remote-code --enable-prefix-caching \
  --gpu-memory-utilization 0.96 \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  # speculative row adds:
  # --speculative-config '{"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
```

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

| image           | py   | torch  | kernels | spec                | pp4096 d1 | tg256 d1     |
|-----------------|------|--------|---------|---------------------|-----------|--------------|
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | -                   | 1973      | 31.50        |
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | dspark, 4 draft tok | 2084      | 43.76 ± 4.18 |
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | dspark, 7 draft tok | 2025      | 50.19 ± 5.17 |
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | mtp, 1 draft tok    | 1931      | 42.56 ± 0.30 |
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | mtp, 2 draft tok    | 1918      | 45.23 ± 2.68 |

palmfuture/Qwen3.8-27B-GPTQ-Int4 - GPTQ, group_size 32, mtp correctly excluded
in its shipped config:

| image           | py   | torch  | kernels | spec             | pp4096 d1 | tg256 d1     |
|-----------------|------|--------|---------|------------------|-----------|--------------|
| p314-t214-vxkp0 | 3.14 | 2.14.0 | src, 00 | mtp, 1 draft tok | 1599      | 38.00 ± 0.32 |

Speculative decoding vs the 31.50 baseline: MTP +35% at 1 draft token and +44%
at 2; DSpark +39% at 4 and +59% at 7. Prefill is unchanged throughout.

MTP is the better deal at equal or lower draft depth: 45.23 at 2 tokens beats
DSpark's 43.76 at 4, using the checkpoint's own head rather than a separate
2.5 GB draft model. Its spread is much tighter at depth 1 (±0.30) and widens at
2 (±2.68). DSpark still takes peak throughput by drafting deeper (50.19 at 7).

Comparing the two quantizations at MTP-1, AutoRound/g64 beats GPTQ/g32 on both
axes: prefill 1931 vs 1599 (+21%) and decode 42.56 vs 38.00 (+12%). Same image,
same flags, same draft depth. The finer group size costs speed without any
published evidence of better accuracy - palmfuture's `quant_log.csv` reports
per-module loss (mean 1.59e-04, 0% RTN fallback) but neither ships downstream
eval scores, so quality is still unmeasured.

The Vishva007 checkpoint needs the `config.json` fix below before MTP loads;
palmfuture ships its mtp exclusion correctly.

MTP needs a `config.json` fix. The 15 `mtp.*` tensors ship unquantized
(`.weight`) while all 64 main layers are GPTQ-packed, but `quantization_config`
declares the opposite via AutoRound's `dynamic` rules - `"+:.*mtp.*"` marks them
as quantized, so vLLM builds the draft head quantized and dies on the missing
`qweight`. Flip both mtp rules to exclusions, matching the `linear_attn` idiom
already in that file:

```json
"-:.*mtp.*": {}, "-:.*mtp\\.fc.*": {}
```

vLLM honours this (`gptq_utils.py:get_dynamic_override`). Note `config.json` in
the HF cache is a symlink into `blobs/`, so edit the resolved path. Checkpoint
bug, not vLLM or XPU.

`tg256 c2` is aggregate over 2 streams; per-request is roughly half.

### urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ - the self-quantised upload, image `p314-t215-vxkp0`, 2026-09-04

**The checkpoint** is the tuned AutoRound artifact (`autoround_version`
0.15.0, `iters` 400) re-exported for vLLM: `quant_method: gptq` and
**`lm_head: false`** (verified from the served snapshot's `config.json`), so
vLLM reads the **15.540 GB class** per token - same bytes as `Vishva007`, not
the 13.673 GB the int4-`lm_head` local checkpoints read. Its config ships the
`mtp` exclusion rules correctly (the `config.json` fix above is not needed).
Same tuned nibbles as `qwen38-27b-w4g64-tuned`; different `lm_head` bytes.

**The image** `vllm-xpu-env-next-p314-t215-vxkp0` carries vLLM
`v0.28.1rc1.dev391+g29af8bd67.d20260904` - the third distinct build in this
document (the version-not-tag note below applies). The linear layers take the
same backend as every earlier row: `XPUwNa16LinearKernel for
AutoGPTQLinearMethod` (log-verified 2026-09-04).

Serve (the operator's command, 2026-09-04 - note `--kv-cache-dtype fp8` and
`--max-model-len 200000`, both new against the blocks above):

```bash
docker run --rm -it \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e CCL_ATL_TRANSPORT=ofi \
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR='level_zero:*' \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu-env-next-p314-t215-vxkp0 urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
  --served-model-name urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 200000 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --trust-remote-code \
  --enable-prefix-caching --mamba-block-size 128 \
  --gpu-memory-utilization 0.97 \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'
  # production row below adds:
  # --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Bench: the standing `llama-benchy` command (`pp 4096 --tg 256 --concurrency 1
--depth 1 --no-cache --exact-tg --latency-mode generation`), run three times;
each invocation is itself 3 runs and prints its own ±.

| image           | vLLM                | spec             | kv  | pp4096 d1 | tg256 d1 (3 invocations)                       | median    |
|-----------------|---------------------|------------------|-----|-----------|------------------------------------------------|-----------|
| p314-t215-vxkp0 | 0.28.1rc1.dev391    | mtp, 2 draft tok | fp8 | 1870      | 46.13 ± 0.99 / 47.12 ± 2.46 / 45.84 ± 1.23     | **46.13** |

Measured 2026-09-04, llama-benchy 0.4.0, server warm (graph capture done),
box otherwise idle of GPU work. Two caveats, both from the server's own log:

- **The MTP-2 row is not comparable to any non-speculative engine number**
  (ours included) - it decodes ~2 tokens per target-model step when drafts
  are accepted.
- **`--enable-prefix-caching` is silently inert in this config**: with MTP
  enabled, no KV-cache group is identified as the draft's, every group -
  including the Mamba groups - is treated as a draft group, and the log
  states "prefix-cache reuse across requests will be disabled".

The **non-speculative row from the same command** (the bar the b70-decode
section compares against) is a named follow-on for the operator's next server
restart; `fp8` KV rides in it and must stay in its label - b70-decode holds
KV in bf16.

## b70-decode - this project, phase 1

> **Every row below names its checkpoint, and from 2026-08-26 that is
> load-bearing rather than pedantic.** This engine now runs three, and they do
> not all read the same number of bytes per token:
>
> | checkpoint | `lm_head` | **W** read/token | roofline @ 590 GB/s |
> |---|---|---|---|
> | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` | bf16 | **15.540 GB** | **37.97 t/s** (26.34 ms) |
> | `qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64` | **int4 g64** | **13.673 GB** | **43.15 t/s** (23.17 ms) |
> | `qwen38-27b-w4g64-tuned/Qwen3.8-27B-w4g64` | **int4 g64** | **13.673 GB** | **43.15 t/s** (23.17 ms) |
>
> The two self-quantised checkpoints share a row because they share a *layout*:
> 2015 tensors each, every name, dtype and shape identical, verified from the
> safetensors headers. They differ in the nibbles, not the bytes read.
>
> All measured by the loader itself (docs/13). A t/s figure is comparable
> across the boundary - it is tokens per second either way - but **MBU and the
> roofline are not**, because W is the denominator of both. Everything in this
> section down to "The new-checkpoint rows" is the **`Vishva007`** checkpoint;
> the vLLM rows above it are too. **No number is restated across the boundary
> with a different meaning**: where `647f2d0`'s `Vishva007` row (27.52 t/s /
> 36.34 ms / 72.5%) appears on both sides it is the *same* measurement of the
> *same* checkpoint, quoted once as the ladder's current head and once inside
> the record-row table.

Same box, same checkpoint, same `--max-model-len 16k` / `pp4096` / `depth 1` /
`concurrency 1` shape as the vLLM rows above, so the `tg256` columns compare
directly. This engine has no prefill kernel yet (a prompt costs one decode
replay per id - spec 2 owns prefill), so there is no `pp4096` number to put
beside vLLM's 1973; the ingest rate is recorded instead, and it is the decode
rate, because it *is* decode.

> **Superseded 2026-09-05 - there is a `pp4096` number now.** Spec 2 built
> `Engine::prefill`; the paragraph above describes the engine as it stood until
> `a663f7b` and is kept because the ingest rows below were taken under it.
> `pp4096` at `e44c40c` is **1377.20 t/s device-side** against vLLM's 1973
> HTTP-inclusive, and 4096 ids now cost **2.974 s** where the replay ingest cost
> **121 s**. See "The spec-2 gate rows" below.

```bash
tools/bench_decode.sh                 # 3 runs at depth 4096, tg 256; median + spread
tools/bench_decode.sh --depth 64      # the doc 07 #12 depth experiment
```

**Measured on an idle box** (no container, no other GPU work), three runs each,
`debug_resid` off, default `--max-len 16384`, all with the same harness.
**Six commits over two days.** Five were measured 2026-08-25: the phase-1 rows
are `62bdd4d`, the three lever rows are `a1e2d3a` (spec 1.5 lever L2),
`b045e11` (lever L1) and `c746840` (lever L5), and `ef6acb0` is the spec 1.5
**gate**. The `647f2d0` row was measured **2026-08-26** - see "The record rows"
below for that day's conditions, and the day-drift control that the `ef6acb0`
and `647f2d0` rows form together:

| engine | depth | tg | t/s | ms/token |
|---|---|---|---|---|
| vLLM `p314-t214-vxkp0`, no speculation | 1 † | 256 | **31.50** | 31.75 |
| **b70-decode `647f2d0`** - the standing row, re-measured 2026-08-26 | **4096** † | **256** | **27.52** | **36.34** |
| b70-decode `ef6acb0` - the spec 1.5 gate, levers L2 + L1 + L5 | 4096 † | 256 | 27.54 | 36.32 |
| b70-decode `c746840` - lever L5, the row it was accepted on | 4096 † | 256 | 27.53 | 36.32 |
| b70-decode `b045e11` - levers L2 + L1 | 4096 † | 256 | 26.28 | 38.05 |
| b70-decode `a1e2d3a` - lever L2 only | 4096 † | 256 | 24.83 | 40.27 |
| b70-decode `62bdd4d` - phase 1, before any lever | 4096 † | 256 | 23.73 | 42.14 |
| b70-decode `62bdd4d` | 64 † | 256 | 25.00 | 40.00 |

† **The two `depth` columns do not mean the same thing.** `llama-benchy`'s
`--depth 1` is a *conversation* depth - one turn - and that turn is
`--pp 4096`, so vLLM decodes its 256 tokens with ~4096 KV positions resident.
`b70-decode --depth N` counts **KV positions** directly. The comparable rows are
therefore vLLM's `depth 1` (≈ 4096 positions) and b70-decode's `depth 4096`,
which is why the bolded row is the one against 31.50; the `depth 64` row is the
doc 07 #12 experiment and has no vLLM counterpart.

**Why the rows say `62bdd4d` and not the tagged commit.** The sha in a row must
name the binary that produced it, so the bench apparatus was committed first
(`62bdd4d`), measured, and the numbers and docs committed after (`1210e06`, tag
`decode-core-done`). The two trees are identical along the engine's path -
`src/`, `tests/` and the kernels are byte-for-byte the same; the later commit
touches docs and one `awk` line in `tools/bench_decode.sh`.

The `a1e2d3a`, `b045e11` and `c746840` rows follow the same rule for the same
reason: each is the commit that carries its kernel change and
`tools/bench_decode.sh` read that sha out of the worktree it measured; the rows
themselves and the docs around them are committed after, touching no `src/`
file. **`ef6acb0` runs the same binary as `c746840`, and the claim is stronger than
"docs only".** There are **three** commits between them and they are *not*
docs-only: they touch `src/kernels/attn.cl`, `src/kernels/CMakeLists.txt`,
`src/runtime/buffers.h`, `src/runtime/capture.cc` and
`tests/kernels/attn_test.cc`. **Every changed line in those five files is a
comment or blank** - verified line by line - and the L5 fix rounds proved the
consequence directly: all 8 attention binaries rebuilt **sha256-identical**.
That is why the two rows measure the same engine and read 0.04% apart.

**That argument covers `c746840` ↔ `ef6acb0` and does NOT extend to
`ef6acb0` ↔ `647f2d0`.** Those two are *not* the same binary: the range carries
**666 insertions and 113 deletions across 11 `src/` files** - the auto-round
loader branch (`loader.cc`, `quant.{h,cc}`, `qwen35.{h,cc}`), the `LmHead`
capture rebind (`capture.cc`, `buffers.h`) and the CLI's checkpoint handling
(`b70_decode.cc`). (Only the last hop, `4c81c18..647f2d0`, is comment-only in
`src/`: 15 insertions, every changed line a comment or blank, verified.) The two
rows nevertheless read **27.54 and 27.52** - **−0.02 t/s, +0.02 ms/token** - and
that difference is **inside the measured day-to-day drift of this instrument on
this exact checkpoint and shape**: the same two rows put ingest at 34.44 and
34.47 ms/token, **+0.087%**, and decode at +0.055%. The delta is attributed, not
dangling: the loader work added a code path the published checkpoint does not
take, and the engine it builds for that checkpoint measures the same to within a
drift figure this document can now quote from a same-checkpoint control.

**b70-decode is 12.6% short of vLLM** - **27.52** against 31.50 at the current
commit `647f2d0` (27.54 at `ef6acb0`; **12.63% and 12.57%, which round to the
same 12.6%**), a factor of 1.145 the other way; it was 16.6% short (26.28) after two levers, 21.2% short (24.83)
after one and 24.7% short (23.73) before any. The decomposition and what it
scopes are in
[05-perf-model.md](05-perf-model.md#phase-1-measured--the-honest-verdict); why
the ladder stopped here is
[the spec 1.5 re-assessment memo](superpowers/specs/2026-08-25-spec1.5-reassessment.md).

### The new-checkpoint rows - `qwen38-27b-w4g64-rtn`, 2026-08-26

**A different checkpoint, therefore a different W, therefore a different
roofline and a different MBU denominator.** These rows are *not* continuations
of the ladder above; they are the same engine reading 1.867 GB/token less.
Spec 1.6 §5.1 - docs/15 has the derivation, docs/12 the launch.

### The spec-1.7 gate rows - `2a7df0b`, 2026-09-04

**The gate of spec 1.7, and it clears.** The operator's stop line for this spec
was moved to **≥ 32.0 t/s** (spec §10 amendment); the engine measures
**32.22 t/s / 31.03 ms/token** on the int4-`lm_head` checkpoint.

| engine / checkpoint | depth | tg | runs (t/s) | **t/s** | **ms/token** | MBU | grade |
|---|---|---|---|---|---|---|---|
| **b70-decode `2a7df0b`, `qwen38-27b-w4g64-rtn`** (int4 `lm_head`) | 4096 | 256 | 32.24 / 32.22 / 32.22 | **32.22** | **31.03** | 74.7% of 13.673 GB | **near-idle**, median of 3 - see conditions |
| vLLM `p314-t215-vxkp0` (`0.28.1rc1.dev396`), `urakozz/...-g64-AutoRound-GPTQ`, **no speculation**, fp8 KV | 1 † | 256 | 31.01 / 30.99 / 30.98 | **31.01** | 32.25 | - (bf16 `lm_head`, 15.540 GB class) | measured, 3 invocations |

Spread 0.02 t/s (**0.06%**), which falls inside the 0.00-0.15% band the strict
record triples define. **The grade is one notch below those rows and the reason
is named:** docker was down and load average 0.74, but two desktop processes
(`baobab` pid 285644, `ptyxis` pid 285678) held DRM file descriptors on both
cards throughout, verified before and after the triple. Every prior record row
in this document was taken with **no** DRM holder on the machine. The 0.06%
spread is *evidence* those two were quiescent, not proof; the same-binary
strict-idle re-run is a named follow-on. The relevant prior control:
`Vishva007` under a 12-core compile read 36.27 against its idle median 36.32,
**0.14% apart**, because the decode step is 99.7% inside the GPU fence.

**Against vLLM, and what the comparison does and does not say.** 32.22 vs
**31.01** is **+3.9%** - the first row in this document where b70-decode leads,
on the same box and the same model weights. Two asymmetries, both in our
favour and both stated rather than absorbed:

- **`lm_head`**: ours is int4 (13.673 GB/token); vLLM cannot load a quantized
  `lm_head` (measured, docs/14) and reads the bf16 tensor, 15.540 GB/token.
  **1.867 GB/token of the difference is bytes, not kernels.**
- **KV dtype**: vLLM ran `--kv-cache-dtype fp8`, half our bf16 KV traffic -
  this one favours *vLLM*.

**The byte-matched row separates those, and it reverses the verdict.** Same
binary `2a7df0b`, same shape, same hour, `Vishva007` snapshot `2a90776` - so
`W` = 15.540 GB/token, **the same bytes vLLM reads**:

| engine / checkpoint | depth | tg | runs (t/s) | **t/s** | **ms/token** | MBU | grade |
|---|---|---|---|---|---|---|---|
| **b70-decode `2a7df0b`, `Vishva007`** (bf16 `lm_head`, byte-matched to vLLM) | 4096 | 256 | 29.33 / 29.33 / 29.34 | **29.33** | **34.09** | **77.2%** of 15.540 GB | near-idle, median of 3 |

**On equal bytes vLLM is still ahead: 31.01 against our 29.33 - 5.7%, a factor
of 1.057.** So both of these are true and neither may be quoted without the
other:

- **b70-decode is 3.9% faster end to end** (32.22 vs 31.01), because our
  `lm_head` is int4 and vLLM's cannot be.
- **vLLM's kernels are 5.7% faster on identical weight bytes** (31.01 vs
  29.33). The lead is a *checkpoint* advantage, not a kernel one.

> **OPERATOR RULING, 2026-09-13 - the int4-`lm_head` comparison is retired.**
> Only the byte-matched line is quoted from here on: **bf16 `lm_head` on both
> sides.** The reasoning is one sentence and it is decisive - *vLLM would be
> faster with an int4 `lm_head` too*; it simply cannot load one, so the 3.9%
> measured an inability of the comparison, not a property of the engine. The
> rows above stay exactly as recorded (a measurement is never deleted), but the
> **+3.9% is no longer a headline anywhere**, and the deleted RTN checkpoint is
> **not being re-quantized** to restore it. The standing decode comparison is
> the 2026-09-09 re-gate row: **29.32 against 31.01, trailing by 5.4%.**

**And MBU moves the opposite way from t/s across our own two rows** - 77.2% on
the slower `Vishva007` row, 74.7% on the faster RTN row. That is not a
contradiction and it was predicted before it was measured: bf16 `lm_head` was
the most bandwidth-efficient launch in the step (98.4% of device), so deleting
three quarters of its bytes *lowers the average efficiency of what remains*
while raising throughput. Faster and less efficient at once, on one binary.

vLLM's own MBU on its 15.540 GB is **81.7%** (482 of 590) - still the best
number any software has posted on this silicon here.

**MBU is 74.7%** (441 GB/s of the measured 590), against spec 1.7's **82%**
success bar and the operator's original **90%** fold criterion. **Beating vLLM
on t/s and clearing the efficiency bar are now different answers**, and the
spec-1.7 closing memo carries the second one.

### The spec-2 re-gate rows - `977a31c`, 2026-09-09 - the first RECORD-grade rows this project has taken

**The box was finally, provably idle.** Zero containers and **zero DRM fd
holders**, verified before, midway and after (18:32, 18:34, 18:42; load 0.07 →
1.01; the daemons named under gate row 1 were killed and the box rebooted on
2026-09-09). These rows are therefore **record grade**, and they are the first
in this document that are - every earlier prefill row says iterate or near-idle
and keeps that word.

**The checkpoint changed and that is not cosmetic.** Both gate checkpoints were
deleted from the box overnight on 2026-09-09; the operator ruled the replacement
is `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` snapshot `84575a1` (GPTQ g64
sym, **bf16 `lm_head`**). So the int4-`lm_head` decode class - the 32.22 t/s
row that beat vLLM - is **not measurable at all right now**; it needs the RTN
checkpoint re-quantized. What is measurable is the bf16-`lm_head` class, which
is byte-matched to what vLLM serves.

| engine / checkpoint | device | ids | C | **pp t/s** | ms total | runs (t/s) | spread | grade |
|---|---|---|---|---|---|---|---|---|
| **b70-decode `977a31c`, `urakozz` `84575a1`** (bf16 `lm_head`) | **0** | 4096 | 2048 | **1406.18** | **2912.9** | 1406.78 / 1406.18 / 1399.79 | 0.50% | **RECORD**, median of 3 |
| vLLM `p314-t214-vxkp0`, no speculation | - | 4096 | 2×2048 | **1973** | - | - | - | measured, external, **HTTP-inclusive** |

| engine / checkpoint | device | depth | **tg t/s** | ms/token | runs (t/s) | spread | grade |
|---|---|---|---|---|---|---|---|
| **b70-decode `977a31c`, `urakozz` `84575a1`** | **0** | 4096 | **29.32** | **34.11** | 29.33 / 29.32 / 29.32 | 0.03% | **RECORD**, median of 3 |
| vLLM `dev396`, fp8 KV, no speculation, **same checkpoint** | - | ~4096 | **31.01** | - | - | - | measured, external, HTTP-inclusive |

Chunk-width sensitivity, one run each, context and not the gate (device 0):

| C | pp t/s | ms / 4096 | vs 2048 |
|---:|---:|---:|---|
| 2048 (the default) | 1406.18 | 2912.9 | - |
| 1024 | 1221.90 | 3352.2 | −13.1% |
| 512 | 932.37 | 4393.1 | −33.7% |

**What the two numbers say.**

- **Prefill 1406.18 = 71.3% of vLLM's 1973.** Against gate row 1's 1377.20 that
  is +2.1%, which is the two GDN tasks (A29, A30) landing, not the checkpoint:
  the last iterate row before them read 1408.39 on the *old* checkpoint and this
  reads 1406.18 on the new one, **0.16% apart**. Prefill is checkpoint-blind, as
  ruling A31 predicted - `lm_head` runs once per prompt.
- **Decode 29.32 trails vLLM's 31.01 by 5.4%** on the same bytes. This
  reproduces the recorded Vishva row (29.33) to **0.03%**, inside the 0.09%
  drift rule, across a different checkpoint of the same class, a reboot and a
  Level Zero update. The engine did not change; the available checkpoint did.
  **The 32.22 t/s row stands as recorded and is NOT being re-taken** - the
  operator retired that comparison on 2026-09-13 (the ruling sits above the
  spec-1.7 rows): vLLM would be faster with an int4 `lm_head` too, so the
  margin measured the comparison's asymmetry rather than the engine. **29.32
  against 31.01, on matched bytes, is the decode number this project quotes.**
- A 4096-token prompt now prefills in **2.913 s** against the 121 s the
  decode-replay ingest took: **41.5×**.

**~~Not a gate, and no tag.~~ Superseded 2026-09-14: the golden gates ran and
are green, and spec 2 is closed (tag `spec2-done`).** The CPU oracle for
`84575a1` was dumped 2026-09-14 20:40-21:20 under `vllm-xpu-env-next-p314-t215`
(`oracle-out-primary/`: prose 311,030,392 B, code 367,258,272 B, cjk
299,192,968 B - code and cjk byte-identical in size to the Vishva set, prose 8 B
of header apart). It is the first set dumped after `tools/oracle/dump.py` was
made to follow `model.safetensors.index.json`: every prompt logged `shards: 8
from model.safetensors.index.json (12 .safetensors present)`, i.e. it read the
7-shard set the engine loads and not the stale 4-shard set beside it.
`tools/oracle/check.sh` ran clean on all three.

Both gates, device 1 (`ZE_AFFINITY_MASK=1`), build at `a6ce30e`, ctest:

| gate | path | prose | code | cjk | **determined rows** | undetermined | ctest |
|---|---|---|---|---|---|---|---|
| `golden_gate_test` | decode ingest, one id per replay | 31 exact + 1 tie-agree | 32 exact | 30 exact + 1 tie-agree + 1 tie-member | **93/93 exact** | 3, all inside the golden argmax set | Passed, 66.59 s |
| `prefill_gate_test` | one `Engine::prefill` per prompt | 31 exact + 1 tie-agree | 32 exact | 30 exact + 2 tie-agree | **93/93 exact** | 3, all inside the golden argmax set | Passed, 21.26 s |

**The one divergence is itself a cross-check.** On `cjk` the decode path emits
4960 at generated position 22 where the oracle emits 271 - and the oracle's own
logits mark that row undetermined (4960 is in its argmax set), so the gate's
tie clause holds and the teacher-forced walk matches every remaining row. The
prefill path takes 271 at the same row. **Position 22 of `cjk` is exactly where
`b70-serve` and `b70-decode --ids` diverged on 2026-09-09 before `--prefill`
existed** - so that divergence, which blocked spec 3's bar 3 for five days, sat
on a row the CPU reference cannot decide either. Ruling A26's reading of it
(sub-ulp ties between two independently rounded paths) is confirmed by the
oracle on this checkpoint.

The tensor diagnostics mark a run of `**LOW**` tap cosines on `code` (down to
~0.989 at layers 52-57, positions 41 and 57) while all 32 of its tokens are
exact. Per the standing `kBar` ruling in `golden_common.h` - tokens gate,
tensors diagnose - that is recorded, not failed.

**What is and is not met at close.** Correctness: both golden gates, the
determinism gate and the prefill-vs-decode consistency gate (18/18 under A26)
are green on the gate checkpoint. Performance: **SHORT** - `pp4096` 1406.18 t/s
is **71.3%** of vLLM's 1973. **Still owed and not claimed: §6 bar 2**, the
multi-chunk golden gate over the ≥ 2048-id `long.ids` prompt at C = 1024. Its
test is registered only while the RTN checkpoint exists (deleted 2026-09-09),
and no long oracle dump exists for `84575a1`; it needs `PROMPTS=long` dumped for
this checkpoint and the registration re-pointed.

### The spec-2.1 rows - `df5f92a`, 2026-09-18 - prefill on the Level Zero backend

**The box was provably idle, and both backends were measured in the same
session.** Checked from the Mac at **18:02:12 CEST, immediately before the
first timed row**: `docker ps -q | wc -l` = **0**, the `/proc/*/fdinfo/*`
`drm-driver` scan printed **no holder lines at all**, `uptime` = `18:02:15 up 4
days, 1:56, 10 users, load average: 0.38, 0.31, 0.33` (the three GUI processes
that had held both render nodes were closed by the operator beforehand; this
agent killed nothing). The harness re-samples the same two conditions *after*
the runs and printed **`RECORD grade` for both backends** - quoted as printed,
never upgraded. Device **0**, the series device, `ZE_AFFINITY_MASK` unset;
checkpoint `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` snapshot `84575a1`
(GPTQ g64 sym, bf16 `lm_head`); `tools/bench_decode.sh --pp 4096`, three runs
each, the control taken with `--no-build` against the identical binary.

| engine / checkpoint | backend | device | ids | C | **pp t/s** | ms total | runs (t/s) | spread | grade |
|---|---|---|---|---|---|---|---|---|---|
| **b70-decode `df5f92a`, `urakozz` `84575a1`** | **l0** | 0 | 4096 | 2048 | **1502.83** | **2725.5** | 1504.63 / 1502.83 / 1500.01 | 0.31% | **RECORD**, median of 3 |
| b70-decode `df5f92a`, `urakozz` `84575a1` | sycl-tla (control) | 0 | 4096 | 2048 | 1407.63 | 2909.8 | 1405.46 / 1408.20 / 1407.63 | 0.19% | RECORD, median of 3 |
| b70-decode `977a31c` (the standing row) | sycl-tla | 0 | 4096 | 2048 | 1406.18 | 2912.9 | 1406.78 / 1406.18 / 1399.79 | 0.50% | RECORD, median of 3 |
| vLLM `p314-t214-vxkp0` | - | - | 4096 | 2×2048 | 1973 | - | - | - | measured, external, HTTP-inclusive |

Against the pre-registration (spec 2.1 §10, committed at `07d45b8` before the
first row): expected **≈ 1498 t/s** (measured-only, 2734.3 ms) / **≈ 1529 t/s**
(with the estimated GEMM credit, 2678 ms); gate **≥ 1480 t/s at RECORD grade**.
The measured **1502.83 t/s (2725.5 ms)** lands **above the gate by +22.83 t/s
(+1.54%)**, **above the measured-only expectation by +4.83 t/s (+0.32%)** - 8.8
ms faster than the derived 2734.3 - and **below the GEMM-credit estimate by
−26.17 t/s (−1.71%)**. P-B's slab saving is therefore fully realised and a
little more; the extrapolated GEMM credit is not. Against the standing
`977a31c` row it is **+96.65 t/s (+6.87%)**, and against the control taken in
the same session **+95.20 t/s (+6.76%)** - the control reproduces the standing
row to **0.10%**, so the margin is the backend and not the session. vLLM's 1973
is now **76.2%** covered (was 71.3%).

Launches per chunk **8689 (l0) vs 1201 (sycl-tla)**, host waits **0 vs 656**,
SYCL GEMMs **0 vs 384**, prefill scratch **−320,864,256 B** (356,515,840 →
35,651,584) - derived, and live-asserted by `prefill_smoke_test` (`L0 live: one
chunk of 64 ids advanced the L0 counter by 8694 launches`). The decode tg
medians from the same two runs: **29.40 t/s** (l0 session, 34.01 ms/token,
spread 0.00%) and **29.43 t/s** (sycl-tla session, 33.98 ms/token, spread
0.03%) against the standing **29.32** - +0.27% and +0.38%, inside the drift
rule: **decode is untouched**, as it must be, since prefill selects the backend
and the replayed decode list is byte-identical.

Phase attribution (instrumented, **iterate grade by construction** - the extra
waits are the instrument; device 1, `B70_PREFILL_PROFILE=1`, one run each,
`--bench --pp 4096 --tg 4`):

| phase | l0: ms | share | waits | sycl-tla: ms | share | waits |
|---|---:|---:|---:|---:|---:|---:|
| `dequant` | 0.0 | 0.0% | 0 | 413.3 | 14.3% | 512 |
| `gemm` (sycl-tla) | 0.0 | 0.0% | 0 | 1403.2 | 48.7% | 512 |
| `linear_l0` | **1668.9** | **61.2%** | 512 | 0.0 | 0.0% | 0 |
| `attn_QK^T` | 84.6 | 3.1% | 128 | 83.0 | 2.9% | 128 |
| `attn_softmax` | 60.5 | 2.2% | 128 | 60.6 | 2.1% | 128 |
| `attn_PV` | 8.3 | 0.3% | 32 | 8.3 | 0.3% | 32 |
| `gdn_scan` | 328.8 | 12.1% | 96 | 330.8 | 11.5% | 96 |
| `norm` | 98.4 | 3.6% | 257 | 98.7 | 3.4% | 257 |
| `silu` | 99.6 | 3.6% | 128 | 98.5 | 3.4% | 128 |
| **TOTAL** | **2728.1** | 100.0% | **2306** | **2881.1** | 100.0% | **2818** |
| plain (unprofiled) run of the same shape | 2796.7 | - | - | 2966.3 | - | - |
| the instrument costs | −68.6 ms (−2.5%) | - | - | −85.2 ms (−2.9%) | - | - |

**The linears are the whole of the win, measured.** `linear_l0`'s 1668.9 ms
replaces `dequant` + `gemm`'s 1816.5 ms: **−147.6 ms, −8.1%**, and 512 host
waits per run disappear with them. Every other phase moves by less than 2 ms -
the three `attn_*` phases sum to 153.4 ms on l0 against 151.9 on sycl-tla,
**+1.5 ms**, so risk 6 (256-padding the attention depth) is worth about 1% of
attention and 0.05% of the walk at this shape, and risk 3 (8,689 launches
costing host time in situ) does not appear at all: the l0 walk has *fewer* host
waits than sycl-tla's, not more. Risk 2 is the one the table does price - the
unmeasured shapes are inside `linear_l0`'s single number, which beat sycl-tla's
pair but by less than the GEMM credit extrapolated.

#### Re-measured on IGC 2.41.5 - `71da3b2`, 2026-09-19 - no change

The operator upgraded `intel-igc-core-2` / `intel-igc-opencl-2` **2.38.2 → 2.41.5**;
`intel-ocloc`, `intel-opencl-icd`, `libze-intel-gpu1` (26.35.39758.10-0) and `libigdgmm12`
(22.10.0) were reinstalled at the same versions, so **only the kernel code generator
changed**. The `.bin` kernels on the box dated 2026-08-25 and no source had changed, so a
plain build would have reused them: all 23 `.cl` sources were touched and **139 kernel
binaries recompiled** before measuring. `libb70_prefill.so` was NOT rebuilt, which makes the
sycl-tla row a pure box-state control. Box idle at 17:20:43 CEST: **0** DRM holder fds, **0**
containers (four `xe` holders - ptyxis, gnome-control-c, chrome, Xwayland - were killed by
the operator first). Device 0, mask unset, median of 3, same session.

| engine / checkpoint | backend | IGC | **pp t/s** | ms total | runs (t/s) | spread | grade |
|---|---|---|---|---|---|---|---|
| b70-decode `71da3b2`, `urakozz` `84575a1` | **l0** | 2.41.5 | **1498.97** | 2732.5 | 1502.58 / 1498.97 / 1496.17 | 0.43% | RECORD, median of 3 |
| b70-decode `71da3b2`, `urakozz` `84575a1` | sycl-tla (control) | 2.41.5 | 1409.92 | 2905.1 | 1411.46 / 1409.92 / 1408.95 | 0.18% | RECORD, median of 3 |
| b70-decode `df5f92a` (the closing rows, above) | l0 / sycl-tla | 2.38.2 | 1502.83 / 1407.63 | 2725.5 / 2909.8 | - | 0.31% / 0.19% | RECORD, median of 3 |

**Verdict: the IGC bump changed nothing measurable.** l0 −3.86 t/s (−0.26%) and sycl-tla
+2.29 t/s (+0.16%), both inside the runs' own spreads; the control reproducing the closing
row to 0.16% is what says the box is in the same state, and makes the l0 delta readable as
noise rather than a regression. l0 over sycl-tla in this session: **+6.32%**. Decode tg
median **29.41 t/s** on both backends (vs 29.40 / 29.43 at the close) - unmoved.

Correctness under the new code generator: `pf_gemm_test` 31/31 cells bitwise equal to
sycl-tla, `pf_dequant_slab_test` bit-exact, and the full suite green - `100% tests passed, 0
tests failed out of 75`, `captured: 774 kernels, 19 modules`, `TOTAL: 93/93` three times
(decode golden, prefill sycl-tla, prefill l0). The `spec2.1-done` tag stands.

### The spec-3 gate rows - `a066c3c`, 2026-09-14 - bars 3-6 closed, tag `spec3-done`

**Box conditions.** Idle proof (plan 6e Task 3 Step 1's command) taken before
and after every row set below: `docker ps -q | wc -l` = 0 and zero DRM fd
holders on every card, every time (verified around the llama-benchy runs, the
CLI control runs, and after the sampling measurement). Series rows are device
0 (unmasked, `ZE_AFFINITY_MASK` unset). Checkpoint
`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ` snapshot
`84575a18f209992ef96d819b31f924b489e3d55d` (bf16 `lm_head`), same as every
gate row above.

**llama-benchy, three invocations** (the standing command,
`docs/BENCHMARKS.md:112-116`, `--served-name b70`, each invocation itself
3 runs - llama-benchy's own `--runs` default):

| invocation | tg256 t/s | coherence | llama-benchy's own `pp4096` figure |
|---|---:|---|---|
| 1 | 29.886 ± 0.018 | PASSED | 5,930,663 ± 1,099,651 t/s - **not usable, see below** |
| 2 | 29.910 ± 0.004 | PASSED | 7,377,061 ± 322,292 t/s - not usable |
| 3 | 29.902 ± 0.004 | PASSED | 7,610,732 ± 544,601 t/s - not usable |
| **median** | **29.902** | 3/3 PASSED | - |

**llama-benchy's printed `pp4096` figure is not usable against this server,
structurally, not as a defect.** `pp_throughput = prompt_tokens / est_ppt`,
`est_ppt = ttfr − latency`, and `ttfr` is the arrival time of the FIRST SSE
frame carrying `choices` at all. `b70-serve`'s chat stream writes the
role-delta frame *before* `generate()` starts - correct, spec-compliant,
anti-buffering behaviour `protocol_test.cc` case 5 asserts on directly (first
content frame at 51 ms, every frame ordered before the next engine step
completes) - so `ttfr` measures time-to-role-frame, not time-to-first-token,
and `est_ppt` comes out sub-millisecond. The figures above (millions of t/s,
relative std 5-19%) are that artifact, not a measurement of anything real;
they are recorded here rather than discarded so the number is not silently
missing. `tg_throughput` is unaffected (it is derived from streamed content
chunk timestamps and a token count, not from `ttfr`), and it is the number
bar 5 is scored on.

**Coherence answer** (non-streaming probe, `/v1/chat/completions`,
`{"role":"user","content":"What is the capital of France?"}`, `max_tokens: 32`):

> We need answer simple. Need final concise.
> \</think\>
>
> The capital of France is Paris.

**Direct `pp_http` probe** - because llama-benchy's own figure is unusable, pp
over HTTP is measured directly instead: a non-streaming `/v1/completions`
request with `max_tokens: 1` over a prompt this server's OWN tokenizer counts
as exactly 4096 tokens (binary-searched via `usage.prompt_tokens`, not
llama-benchy's local `gpt2`-fallback estimate - `llama-benchy` could not
resolve `b70` as an HF tokenizer id and fell back to `gpt2`, whose token
density differs from this checkpoint's; `--adapt-prompt` tries to correct for
this from one warmup delta and evidently undershot it here, which is the root
cause of the implausible `pp_throughput` figures above once the FIRST
(chat-completions, template-included, `--adapt-prompt`-sized) attempt at this
number is set aside). Wall time from request send to full response received,
3 runs:

| run | prompt tokens | wall ms | t/s |
|---|---:|---:|---:|
| 1 | 4096 | 2905.84 | 1409.57 |
| 2 | 4096 | 2915.83 | 1404.74 |
| 3 | 4096 | 2918.48 | 1403.47 |
| **median** | 4096 | **2915.83** | **1404.74** |

**CLI control, same session, same device (0), record grade both before and
after** (`tools/bench_decode.sh`):

| row | t/s | ms | runs | spread | grade |
|---|---:|---:|---|---:|---|
| tg256 @ depth 4096 (`--depth 4096 --tg 256`) | **29.31** | 34.12 ms/token | 29.30 / 29.31 / 29.32 | 0.07% | RECORD |
| pp4096, device-side, first-token-inclusive (`--pp 4096 --tg 256`) | **1402.91** | 2919.6 ms total | 1402.91 / 1403.41 / 1402.83 | 0.04% | RECORD |

**Bar 5 - `tg_http / tg_cli ≥ 0.98`:** 29.902 / 29.31 = **1.0202 (102.0%). MET**,
with margin; HTTP's tg cost is not merely under 2%, it is not measurably
present (the per-token host path - detokenise + SSE write - overlaps the next
replay exactly as spec §3.4 designed it to).

**Derived HTTP cost of prefill** (`4096/pp_http − 4096/pp_cli`, both
first-token-inclusive so the two windows match): 2915.83 ms − 2919.6 ms =
**−3.77 ms per 4096 tokens** - i.e. statistically indistinguishable from zero
against the ~0.5-3 ms run-to-run spread both sides already show. Tokenizer
encode of a raw `/v1/completions` prompt (no template) plus HTTP framing costs
nothing measurable on top of the device-side prefill.

**Task 4 - chat template smoke, `enable_thinking`.** The template test's
3-turn conversation (system + user 中文 + assistant 中文 + user), streamed,
`chat_template_kwargs.enable_thinking`:

- **`true`**: first content chunk is `"User"` - a `<think>` reasoning trace
  begins immediately (the opening `<think>` tag is part of the rendered
  prompt, not generated text, so it is not itself an SSE frame; the model's
  own `</think>` closes it before the final answer, as seen separately in the
  coherence probe above).
- **`false`**: first content chunk is `"Berlin"` - the direct answer, no
  reasoning block, `finish_reason: "stop"` two frames later.

Confirmed: `<think>` appears only when `enable_thinking` is on. Not a bar
(Open WebUI / opencode itself is the operator's to point at port 8000).

**Task 5 - host sampling.** Pre-registered 2026-09-14, before measuring
(`2026-09-09-spec3-gate-memo.md` §5): bar ≤ 0.62 ms/token (2% of the 31.0 ms
step), derived estimate 0.2-0.4 ms.

**Measured: 0.537 ms/token - MET, ships on by default.** 3 greedy vs 3 sampled
256-token requests (`min_tokens: 256, ignore_eos: true`; sampled adds
`temperature: 1.0, top_k: 20, top_p: 0.95`), server-side generation-only
timing (`b70-serve`'s own `gen: N tokens, X ms, Y ms/token` stderr line, added
in `src/cli/serve_adapters.h`, timed from after `prefill()` to the next
request's `prefill()` call):

| | greedy ms/token | sampled ms/token |
|---|---:|---:|
| run 1 | 31.7671 | 32.2866 |
| run 2 | 31.7766 | 32.3276 |
| run 3 | 31.7974 | 32.3140 |
| **median** | **31.7766** | **32.3140** |

Delta = 32.3140 − 31.7766 = **0.5374 ms/token**, under the 0.62 ms bar.
Cross-validated independently by curl's own round-trip differential over the
same 3+3 requests (which cancels the identical prefill cost on both sides):
(8616.7 ms − 8481.5 ms) / 256 = **0.528 ms/token** - the two independent
methods agree to within 0.01 ms.

**A first attempt at this measurement read 0.7408 ms/token - over the bar -
and was a measurement artifact, not a code cost.** It ran each of the 6
requests over its own freshly-connected ssh session; the SSH-handshake time
preceding the sampled batch's requests happened to be larger than the greedy
batch's, and because this measurement's window (from one request's
`prefill()` to the next's) includes whatever gap sits between them, that
connection-setup jitter landed in the reported numbers unevenly across the
two batches. The clean measurement above reissues the identical 8 requests
(warmup, 3 greedy, 3 sampled, 1 trailing flush to emit the last line) inside
ONE persistent ssh session, curl looped remotely on the box - removing the
per-request reconnect jitter - and is corroborated by the independent
curl-side timing. Discarded, not reconciled: the 0.74 ms figure measured a
network artifact, not the sampler.

Greedy is unaffected either way, by construction (`sample_into_control` runs
only when `!sampling.greedy`) and by measurement (31.78 ms/token here is
consistent across all 3 runs, spread 0.03 ms).

**Bar 6 - suite and protected directories.** Full suite (`ctest --test-dir
build --output-on-failure`, `a066c3c`): **100% tests passed, 0 tests failed
out of 67** - no skips (the two golden gates now run rather than skip; see
below). `git diff --stat 6d80193..HEAD -- src/runtime src/kernels src/l0
src/loader src/model` is empty. `replay_determinism_test`: `captured: 774
kernels, 19 modules` and `replay_determinism_test OK (8 tokens x 3 runs
bitwise identical)`. `golden_gate_test`: `TOTAL: 93/93 determined rows exact,
3 undetermined (2 agree + 1 other member)`. `prefill_gate_test`: `TOTAL:
93/93 determined rows exact, 3 undetermined (3 agree + 0 other member)`.
`golden_server_test` (this run): `prose: prompt ids identical, 32/32
generated ids identical`; `code:` and `cjk:` the same.

**Verdict: all six bars met.** Tag `spec3-done`.

### The spec-3 gate attempt - `983e852`, 2026-09-09 - BLOCKED before performance rows

**No new performance number is recorded in this section.** The box was available
for a record-grade run, but Task 2's `golden_server_test` stopped the plan first:
`prose` and `code` reproduce 32/32 greedy IDs through `b70-serve`; `cjk` has
exact prompt IDs but diverges at generated ID 23 against `b70-decode --ids`.
The server uses `Engine::prefill`; the CLI reference uses `Engine::ingest`, two
intentionally differently rounded paths. This finding invalidates the plan's
assumption that their direct ID comparison has no tie question.

Accordingly, there are **no measured HTTP-inclusive pp4096/tg256 rows**, no
measured CLI control rows, no derived HTTP cost, and no host-sampling cost for
this attempt. The historical device-0 rows above remain the comparison baseline:
**measured RECORD** pp4096 `1406.18 t/s` device-side and tg256 `29.32 t/s`; vLLM
is **measured external HTTP-inclusive** at pp4096 `1973 t/s` and tg256 `31.01 t/s`.
No historical value is superseded. See
`docs/superpowers/specs/2026-09-09-spec3-gate-memo.md` for both cjk ID lists and
the ruling request. No `spec3-done` tag was made.

### The spec-2 gate rows - `e44c40c`, 2026-09-05 - the first `pp4096` this engine has ever had

**The gate of spec 2, and it does not clear.** The bar was vLLM's **1973 t/s**;
the engine measures **1377.20 t/s device-side**, **69.8%**, and the spec's own
§3.0c amendment (2026-09-05) had already ruled that bar unreachable by this
design and restated the deliverable as 70-79% of vLLM. This row is that band's
floor. The full arithmetic and what is queued behind it:
[the spec-2 gate memo](superpowers/specs/2026-09-05-spec2-gate-memo.md).
**This is gate row 1**; a strict-idle re-gate follows two more GDN tasks.

```bash
tools/bench_decode.sh --pp 4096 --tg 256 --runs 3         # the gate command
ZE_AFFINITY_MASK=1 tools/bench_decode.sh --pp 4096 ...    # names the card, since e44c40c
```

| engine / checkpoint | device | ids | C | **pp t/s** | ms total | runs (t/s) | spread | grade |
|---|---|---|---|---|---|---|---|---|
| **b70-decode `e44c40c`, `qwen38-27b-w4g64-rtn`** (int4 `lm_head`) | **0** | 4096 | 2048 | **1377.20** | **2974.2** | 1379.72 / 1377.20 / 1375.71 | 0.29% | **iterate**, median of 3 |
| b70-decode `e44c40c`, `qwen38-27b-w4g64-rtn` | 1 | 4096 | 2048 | 1330.82 | 3077.8 | 1331.43 / 1330.82 / 1328.62 | 0.21% | iterate, median of 3 |
| **b70-decode `e44c40c`, `Vishva007`** (bf16 `lm_head`, byte-matched to vLLM) | **0** | 4096 | 2048 | **1374.87** | 2979.2 | 1376.01 / 1374.87 / 1373.14 | 0.21% | iterate, median of 3 |
| b70-decode `e44c40c`, `Vishva007` | 1 | 4096 | 2048 | 1330.63 | 3078.2 | 1333.44 / 1330.63 / 1327.44 | 0.45% | iterate, median of 3 |
| vLLM `p314-t214-vxkp0`, no speculation | - | 4096 | 2×2048 | **1973** | - | - | - | measured, external, **HTTP-inclusive** |

Chunk-width sensitivity, **one run each, context and not the gate** (device 1):

| C | pp t/s | ms / 4096 | vs 2048 |
|---:|---:|---:|---|
| 2048 (`PrefillScratch::kC`, the default) | 1330.82 | 3077.8 | - |
| 1024 | 1185.27 | 3455.7 | −10.9% |
| 512 | 899.36 | 4554.3 | −32.4% |
| 4096 | - | - | **refused: `chunk 4096 exceeds PrefillScratch::kC 2048`** |

Monotone in C across the whole available range, which is what the measured
411.5 ms fixed cost per chunk predicts: halving C doubles the chunk count and
therefore the fixed term. Ruling A13's 2048 is the widest this build runs.

**Box conditions, and why no row here is record grade.** Zero containers, load
0.45-0.88, 84 GB free before and after, verified before and after every row set.
But `baobab` (285644) and `ptyxis` (285678) held DRM fds on both cards
throughout - the same two processes the spec-1.7 gate rows named. New here, and
it is why waiting does not help: **they re-acquire those fds periodically.** The
box was verified clean at 19:55; the fds the gate ran under were opened at
**19:58:59**, and another set on the GT 710's node at 20:01:41. Every one of
those clients reports `drm-total-vram0: 0` and carries no per-engine busy-cycle
line at all, so they have submitted nothing to either B70 - but zero DRM holders
is what a record row means in this document, and this is not that. The 0.21-0.45%
spreads are *evidence* they were quiescent, not proof. (docs/BENCHMARKS calls
this condition **near-idle** on the spec-1.7 rows and the prefill docs call it
**iterate**; it is one box state under two existing names, and neither is
record.)

**The two cards are not interchangeable, and only prefill sees it.** Same
binary, same checkpoint, same ten minutes:

| quantity | device 0 (`04:00.0`) | device 1 (`08:00.0`) | Δ |
|---|---:|---:|---:|
| pp4096 t/s, RTN | 1377.20 | 1330.82 | **−3.37%** |
| pp4096 t/s, Vishva | 1374.87 | 1330.63 | −3.22% |
| tg256 t/s, RTN | 32.37 | 32.31 | −0.19% |
| tg256 t/s, Vishva | 29.42 | 29.31 | −0.37% |

The separation is far outside the spreads and reproduces on both checkpoints:
**prefill loses 3.2-3.4% on device 1, decode 0.2-0.4%** - a difference in
sustained compute rather than in bandwidth, seen by the compute-bound workload
and not by the bandwidth-bound one. Unattributed: neither B70 drives a display
(the only connected connector on the box is `card0-HDMI-A-5`, on the GT 710),
both report identical unprivileged PCIe link fields, and the `xe` frequency
sysfs needs root. **The bolded rows are device 0** because every `--pp` row this
spec has taken ran there - `tools/bench_decode.sh` silently dropped
`ZE_AFFINITY_MASK` until `e44c40c` - and device 0's 1377.20 reproduces
`42921d2`'s 1375.65 to **0.11%**, which is how we know the engine did not move.

**Against vLLM, with the three labels that must travel with the number.**

1. **Device-side vs HTTP-inclusive.** Ours spans the first prefill launch to the
   first generated id present in `cur_token`, loader excluded. vLLM's `pp` is
   `llama-benchy`'s `est_ppt = ttfr − probe latency` and contains HTTP receipt,
   tokenization, scheduling and first-token sampling. **Ours is the narrower
   quantity and is flattered by the comparison**; spec 3's server measures the
   HTTP row and no equivalent is derived here.
2. **Chunk width.** Ours is C = 2048; vLLM's 4096 is two chunked steps of 2048
   (`max_num_batched_tokens` on a 32 GB card). The same width on both sides.
3. **Checkpoint bytes.** 13.673 GB/token (RTN) against vLLM's 15.540 GB.

**Label 3 does not matter for prefill, and that is new.** The byte-matched
Vishva row measures **1374.87** against RTN's **1377.20** - **0.17% apart**,
where the same two checkpoints are **9.9% apart on decode** (32.21 / 29.31).
Prefill runs `lm_head` exactly once per prompt, for the last position
(`step_head`, 0.35 ms/chunk measured), so the 1.867 GB/token asymmetry that
decides the decode comparison is immaterial here. **The byte-matched row trails
vLLM by 30.3% and the unmatched one by 30.2%**: on prefill there is no
checkpoint advantage to subtract, and none to claim.

**What the row replaces.** `Engine::ingest` replayed the decode list once per
prompt id: **121 s for 4096 ids** (bench log `2a7df0b`, RTN, 29.5 ms/id). The
same prompt is now **2.974 s** - **40.7×** (39.3× on device 1). That, and not
the vLLM comparison, is what spec 2 set out to do.

### The spec-2 decode control rows - `e44c40c`, 2026-09-05

Spec §6 bar 5: decode re-measured once at the spec's final sha, on device 0
where every historical row in this document was taken.

| engine / checkpoint | depth | tg | runs (t/s) | **t/s** | **ms/token** | vs `2a7df0b` | grade |
|---|---|---|---|---|---|---|---|
| b70-decode `e44c40c`, `qwen38-27b-w4g64-rtn` | 4096 | 256 | 32.21 / 32.21 / 32.21 | **32.21** | **31.05** | 32.22 → **−0.031%** | iterate, median of 3 |
| b70-decode `e44c40c`, `Vishva007` | 4096 | 256 | 29.32 / 29.31 / 29.28 | **29.31** | **34.12** | 29.33 → **−0.068%** | iterate, median of 3 |

**Both inside the ≤ 0.09% day-scale drift this document measures**, so the
spec-1.7 gate rows stand unchanged and prefill cost decode nothing. These are
*not* new records - they are controls, and the recorded rows remain 32.22 and
29.33 at `2a7df0b`. The RTN triple returned the identical row three times
(spread 0.00%), the tightest this harness has read.

### The record rows - all three checkpoints, one idle box, 2026-08-26

**The record-grade rows this section owed have been taken.** `647f2d0`, clean
worktree, `tools/bench_decode.sh --runs 3 --depth 4096 --tg 256`, one after
another on a box that was **idle in the strong sense**: no container running and
**not one process on the machine holding a DRM file descriptor**, verified
before and after each triple (`grep -l drm-driver /proc/*/fdinfo/*` returned
nothing every time). The only other CPU on the box was a root telemetry daemon
at ~15% of one core.

| engine / checkpoint | depth | tg | runs (t/s) | **t/s** | **ms/token** | MBU | grade |
|---|---|---|---|---|---|---|---|
| **b70-decode `647f2d0`, `qwen38-27b-w4g64-tuned`** (int4 `lm_head`) | 4096 | 256 | 30.06 / 30.04 / 30.02 | **30.04** | **33.29** | 69.6% of 13.673 GB | **record**, median of 3, idle |
| **b70-decode `647f2d0`, `qwen38-27b-w4g64-rtn`** (int4 `lm_head`) | 4096 | 256 | 30.04 / 30.05 / 30.03 | **30.04** | **33.29** | 69.6% of 13.673 GB | **record**, median of 3, idle |
| **b70-decode `647f2d0`, `Vishva007`** (bf16 `lm_head`) | 4096 | 256 | 27.52 / 27.55 / 27.51 | **27.52** | **36.34** | 72.5% of 15.540 GB | **record**, median of 3, idle |

Spreads: 0.04 (0.13%), 0.02 (0.07%), 0.04 (0.15%). Two of those three are the
**loosest record triples in this document** - every earlier one sits at
0.00-0.11% - so they are quoted as what they are rather than as confirmation of
a band they define. All three remain far inside the ±0.1 ms/token this
instrument is claimed at. Ingest, from the same runs: **31.40**, **31.41** and
**34.47** ms/token over 4096 ids.

**The two self-quantised checkpoints record the same number to every printed
digit.** 30.04 t/s, 33.29 ms/token, 411 GB/s, 69.6% - two medians of three taken
45 minutes apart with an 18-minute 22-thread oracle run between them. They have
identical tensor manifests (2015 tensors, every name/dtype/shape equal, verified
from the safetensors headers), therefore identical `W`, therefore the same speed
class; the prediction was made from the headers before either was benched.
`tuned` and `rtn` differ in the nibbles, and the nibbles are not a cost.

**The checkpoint delta, at record grade:** **+2.52 t/s, −3.05 ms/token**
(27.52 → 30.04). **Ingest moves by −3.06 ms**, the same amount to within
0.01 ms - which is what a per-token *bytes* lever must do on an engine whose
ingest is decode, and neither figure was tuned to make it so.

**Against the bar**, which is `Vishva007`-shaped and stays where it is: **30.04
t/s is 1.46 t/s (4.6%) short of vLLM's 31.50.** The re-assessment memo's
conclusion - `lm_head` at int4 is necessary and not sufficient - is unchanged
and slightly reinforced by the properly-measured row.

#### The day-scale drift of this instrument, finally measured on a control

The `ef6acb0` and `647f2d0` rows are the **same checkpoint, same shape
(`4096`/`256`), same harness, both idle, both medians of three, one day apart**.
That is a same-checkpoint day-drift control, and it is the tightest one this
project has:

| | 2026-08-25 `ef6acb0` | 2026-08-26 `647f2d0` | drift |
|---|---|---|---|
| decode | 27.54 t/s / 36.32 ms | 27.52 t/s / 36.34 ms | **+0.055%** |
| ingest (4096 ids) | 34.44 ms/token | 34.47 ms/token | **+0.087%** |

**≤0.09%.** That replaces the +0.30 / +0.50 / +0.56% figures §L2/§L1/§L5 quote
for run-to-run drift on untouched launches - those were measured *across a
kernel change*, so they carry the change's own noise; this one carries nothing
but the day. Anything larger than ~0.1% between two idle medians of this
instrument needs a cause other than drift.

#### Why these rows are under the iterate rows, and what does NOT explain it

`--bench` sweeps `pos` from `DEPTH` to `DEPTH + TG − 1`, so mean KV depth is
4127.5 at `tg 64` and 4223.5 at `tg 256`: **+2.33% of attention work per
token**, worth **+0.083 ms** against `attn_decode`'s measured 3.585 ms plus
~0.003 ms of extra `attn_reduce` block merging - **≈ +0.086 ms/token**. It
applies to **both** checkpoints equally, because attention does not know which
head is packed:

| | iterate (tg 64, loaded, 1 run) | record (tg 256, idle, med 3) | shift | `tg` predicts | residual |
|---|---|---|---|---|---|
| `Vishva007` | 36.27 | 36.34 | +0.07 | +0.086 | **−0.016** |
| `qwen38-27b-w4g64-rtn` | 33.11 | 33.29 | +0.18 | +0.086 | **+0.094** |

**And because it applies equally, `tg` cancels out of the lever itself.** The
`lm_head` delta reads **3.16 ms** at iterate grade (36.27 − 33.11) and
**3.05 ms** at record grade (36.34 − 33.29). The `tg` term cannot explain that
0.11 ms, and neither can drift: 0.11 ms on 33.2 ms is **0.33%**, which is
**~4× the 0.087% the control above measures**. This is stated as unexplained
rather than absorbed - see the next block.

#### Three values for one quantity, and they do not close

The `lm_head`-at-int4 lever has been measured three ways, and they span
**0.15 ms**:

| grade | figure | of the memo's 3.26 ms ceiling | of the byte-exact 3.165 ms |
|---|---|---|---|
| **in situ**, `--profile`, the launch alone (4381.289 → 1178.164 µs) | **3.203 ms** | **98.3%** | 101.2% |
| **bench, iterate grade** (tg 64, loaded, 1 run each) | **3.16 ms** | 96.9% | 99.8% |
| **bench, record grade** (tg 256, idle, medians of 3) | **3.05 ms** | **93.6%** | 96.4% |

**The in-situ figure is the best-anchored of the three and it is the largest.**
It reconciles against the bytes to a microsecond: `lm_head` reads
2 542 796 800 B at bf16 and 675 430 400 B at int4, a difference of 1.8674 GB =
**3.165 ms** at the measured 590 GB/s; the bf16 launch ran at 98.4% of device
(4310 ideal vs 4381.289 measured, +0.071 ms) and the int4 launch at 97.2%
(1144.8 vs 1178.164, +0.033 ms), and `3.165 + 0.071 − 0.033 = 3.203`. Exactly.

So the profile says the lever is ~3.20 ms and the record bench says 3.05 ms.
**The direction of the discrepancy is not resolvable from what has been
measured.** The `Vishva007` end is solid - it reproduces its own record row to
0.055%. That puts the 0.15 ms on the int4 end, and there are two readings:
either the single loaded `tg 64` RTN run was ~0.1 ms fast, or **today's int4
triples are ~0.1 ms slow** - and the profile evidence points at the second,
which is the opposite of what an earlier version of this section concluded. Both
int4 triples (RTN and tuned, 45 minutes apart) agree to every printed digit, so
whatever it is, it is systematic and not noise.

**The experiment that would settle it** - a `--profile` run on the current tree,
pricing the `lm_head` launch under today's conditions - was not run: this task
took no measurement after its benches. It is a named follow-on, and until it is
run the honest statement is that this lever is known to **±0.15 ms**, and that
every percentage above must be quoted with its grade attached.

### The iterate-grade rows this replaces - kept, because they are the record of how it was found

**Grade: ITERATE, not RECORD.** Single runs, not the median of three
`tools/bench_decode.sh` produces, and the box was **not idle** - a 12-core vLLM
XPU kernel compile ran throughout, and for part of the window a 22-thread oracle
dump as well.

| engine / checkpoint                                                              | depth | tg  | t/s       | ms/token  | MBU                | grade                         |
|----------------------------------------------------------------------------------|-------|-----|-----------|-----------|--------------------|-------------------------------|
| **b70-decode, `qwen38-27b-w4g64-rtn`** (int4 `lm_head`)                          | 4096  | 64  | **30.20** | **33.11** | 70.0% of 13.673 GB | iterate, 1 run, box loaded    |
| b70-decode, `Vishva007` (bf16 `lm_head`) - **the control, same hour, same load** | 4096  | 64  | 27.57     | 36.27     | 72.6% of 15.540 GB | iterate, 1 run, box loaded    |
| b70-decode `ef6acb0`, `Vishva007` - the then-standing record row                 | 4096  | 256 | 27.54     | 36.32     | 72.5% of 15.540 GB | **record**, median of 3, idle |

**The control row is what makes the pair readable.** `Vishva007` under load read
**36.27** against its standing idle median of **36.32** - 0.14% apart. A decode
step that is 99.7% inside the fence does not contend with a CPU compile, and
that is the evidence for it rather than an assumption. The two iterate rows were
run back to back, so the difference between them is the checkpoint.

**Δ = +2.63 t/s, −3.16 ms/token** against the same-hour control; **+2.66 t/s,
−3.21 ms** against the then-standing record row. In-situ, the `lm_head` launch
alone moved **4381.289 → 1178.164 µs**, −3.203 ms/token, which is **98.3% of the
−3.26 ms ceiling** the re-assessment memo derived from the bytes.

> **Both percentages are live, and they must be quoted with their grade.**
> **98.3%** is the *in-situ launch* figure; the *record-grade bench* lever is
> **3.05 ms = 93.6%** of the same ceiling, and the iterate-grade bench lever
> quoted in this very paragraph is 3.16 ms = 96.9%. The three do not close -
> "Three values for one quantity, and they do not close", above, has the
> reconciliation attempt and names the run that would settle it.

Note `tg`: **64, not 256.** These are iterate-grade runs and 64 was chosen to
keep the turnaround short on a loaded box, so they are not the same measurement
as a `tg 256` record row even ignoring the median. **That record row has since
been taken** (2026-08-26, above); the `tg` difference is priced there and is not
free.

**MBU falls while throughput rises**, from 72.6% to 70.0%, and that is the
memo's own prediction: `lm_head` at bf16 was the most bandwidth-efficient launch
in the step (98.4% of device), so removing three quarters of its bytes lowers
the average of what remains. The bench's MBU line divided by a hardcoded
15.540 GB until 2026-08-26 and would have printed **79.5%** here (30.20 x
15.53998 / 590 = 79.54); it now reads W
from the loader's own report and prints which W it used.

**Against the bar**, which is `Vishva007`-shaped and stays where it is: 30.20
t/s is **1.30 t/s short of vLLM's 31.50**. The memo said `lm_head` at int4 was
necessary and not sufficient, and it landed at 98.3% of its ceiling and did not
clear the bar - as derived, before the work started.

### The `ef6acb0` row - the spec 1.5 gate

**The recorded gate of spec 1.5**, run 2026-08-25 on the idle box from a clean
worktree at `ef6acb0`, `tools/bench_decode.sh` with its defaults:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 27.54 / 27.54 / 27.55 | **27.54** | 27.54 | 27.55 | 0.01 (0.04%) |

**Verdict: short.** The bar was **≥ 31.50 t/s** (31.746 ms/token); the engine
measures **27.54 t/s / 36.32 ms/token**, **3.96 t/s - 12.6% - short**, with
**4.574 ms/token** still to find. MBU is 428 GB/s of 590 = **72.5%**, against
vLLM's 83.0%. The ingest half of the same runs reads 141.0 s for 4096 ids,
median **34.44 ms/token** (34.43 / 34.45 / 34.44) on an un-instrumented list.

Per spec 1.5 §6 the short path stops the spec and writes
[the re-assessment memo](superpowers/specs/2026-08-25-spec1.5-reassessment.md),
which carries the per-lever ledger, the remaining gap and the priced redesign
menu. Its headline: `lm_head` at int4 (~3.3 ms, **estimated**) is the largest
item left and it is **necessary but not sufficient** - it lands at 30.62 t/s,
0.88 t/s under the bar, so a second item is required and none has a measured
price yet. **That conclusion is bounded, not estimated:** int4 g64 still reads
0.66 GB, a **1.12 ms floor** at the measured 590 GB/s, so the saving cannot
exceed **3.26 ms** and the best step reachable with L3 and L4 also at their
ceilings is **32.70 ms = 30.58 t/s - 0.92 t/s under**. `lm_head` cannot clear
31.50 even if it lands perfectly. The three levers that produced this row are `a1e2d3a`, `b045e11` and
`c746840` below; the golden gate is **96/96 element-exact** at every one of
them and the full suite is green at `ef6acb0`.

This row measures the same binary as the `c746840` row (three commits between;
their `src/` and `tests/` hunks are comment-only and the attention binaries are
sha256-identical) and reproduces it to **0.04%**: 27.54 against 27.53, both
36.32 ms/token. **Unrounded they reconcile exactly at printed precision:** the
gate's own per-token print is **36.317 ms** (= 27.5352 t/s, prints 27.54),
while a median printing 27.53 is 36.324 ms (derived) - the two are **~7 µs per
token apart**, an order of magnitude inside the 0.04% spread, and both print
36.32. Neither number is a correction of the other; they are two medians of one
engine, and the gate row is the one spec 1.5 closes on.

### The `c746840` row - spec 1.5 lever L5

**Measured 2026-08-25**, same command (`tools/bench_decode.sh`), same idle box,
three runs, `debug_resid` off, default `--max-len 16384`:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 27.51 / 27.53 / 27.54 | **27.53** | 27.51 | 27.54 | 0.03 (0.11%) |

The recorded row is the harness's own median column, **27.53 t/s / 36.32
ms/token**, and MBU 427 GB/s of 590 = **72.5%** (it was 69.2%, and 62.5% before
any lever). The ingest half of the same runs reads 141.2 s for 4096 ids,
34.48 ms/token on an un-instrumented list.

**−1.73 ms/token, and all of it is the attention trio.** `attn_decode` walked a
256-position KV block per work-group; `ATTN_BLOCK` is now **64**, which
quarters the serial walk that docs/15 §2 measured as the launch's real cost and
takes the grid from 68 to 260 live work-groups at depth 4096 - a region the same
document called unmeasured, and which turns out to be free. In situ the family
goes **6.058 → 3.839 ms/token** (`attn_decode` 5.920 → 3.585, `attn_reduce`
0.080 → 0.196 as it merges 65 blocks instead of 17). The launch count did
**not** move: 774 before and after. The four-value block sweep that chose 64,
and the two cost models it falsified on the way, are in
[12-kernels.md](12-kernels.md) "Measured - lever L5" and
[15-step-anatomy.md](15-step-anatomy.md) §L5. The golden gate is **96/96
element-exact before and after**, with every diagnostic cosine unchanged to nine
decimals - and docs/14 says why that is a weaker result than it looks.

**The three deltas, and the 0.32 ms this one does not close.** The lever's own
profile rows fall **2.219 ms**; run-to-run drift on the 726 untouched launches
adds back **+0.174 ms** (+0.56%, the same effect §L2 measured at +0.30% and §L1
at +0.50%); there is no dispatch term because the launch count is unchanged.
Predicted bench delta **−2.045 ms** against **−1.73 measured**: **0.32 ms
apart**, against the 0.087 ms L1 landed within and the ±0.1 ms this instrument
is claimed at. It is not the launch count and not the drift, both of which are
already in the arithmetic. One *named* contributor, and it is too small: the
bench sweeps `pos` 4096 → 4351, so `nb` runs 65 → 69 here where it was a flat 17
before, and the after-run therefore averages ~3% more block-merge work per token
than the single depth-4096 profile point that priced it - worth ~0.01 ms by
`attn_reduce`'s own slope. **The rest is unattributed**, and it is recorded as
unattributed rather than absorbed into a rounding.

### The `b045e11` row - spec 1.5 lever L1

**Measured 2026-08-25**, same command (`tools/bench_decode.sh`), same idle box,
three runs, `debug_resid` off, default `--max-len 16384`:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 26.28 / 26.28 / 26.27 | **26.28** | 26.27 | 26.28 | 0.01 (0.04%) |

The median run prints `per token: 38.046 ms total = 37.957 ms fence + 0.089 ms
host` and MBU 408 GB/s of 590 = **69.2%** (it was 65.4%, and 62.5% before any
lever). The three runs' totals are 38.058 / 38.046 / 38.067 ms and the recorded
row is the harness's own median, **38.05**. The ingest half of the same runs
reads 153.7-153.9 s for 4096 ids, 37.53-37.57 ms/token on an un-instrumented
list.

**−2.220 ms/token, and all of it is one kernel family.** `prep_res_norm` ran
129× per token as ONE work-group per launch - 22.4 µs and 17.0 GB/s through a
single Xe-core - and is now two launches, `prep_res_fold` + `prep_norm_finish`,
on 20 work-groups each: **2.893 → 0.484 ms/token measured in situ**. The
before/after, the separate price of each of the two stages, and why the launch
count went **645 → 774** on purpose are in
[15-step-anatomy.md](15-step-anatomy.md) §L1. The golden gate is **96/96
element-exact** before and after, which is the bar a reordered summation has to
clear; its per-layer *diagnostics* do move, and docs/14 now records both
columns.

**The three deltas, and why they differ.** The lever's own profile row falls
2.409 ms; the bench falls 2.220 ms (40.266 − 38.046). The difference is +0.182 ms of run-to-run
drift on the 516 untouched launches (+0.50%, the same effect §L2 measured at
+0.30%) and +0.095 ms derived for the 129 extra dispatches at 0.733 µs. Predicted
bench delta −2.133 ms against −2.220 measured: **0.087 ms apart**, inside the
±0.1 ms this instrument is claimed at.

### The `a1e2d3a` row - spec 1.5 lever L2

**Measured 2026-08-25**, same command, same idle box, three runs, `debug_resid`
off, default `--max-len 16384`:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 24.83 / 24.84 / 24.83 | **24.83** | 24.83 | 24.84 | 0.01 (0.04%) |

`per token: 40.266 ms total = 40.166 ms fence + 0.100 ms host`, MBU 386 GB/s of
590 = **65.4%** (it was 62.5%). The ingest half of the same runs reads
162.8 s for 4096 ids, 39.75 ms/token.

**−1.875 ms/token, and all of it is one kernel.** The `a‖b` GEMV
(`gemv_bf16` 5120×128, 48 launches/token) went from 2.341 to 0.256 ms/token when
its K was split 16 ways inside the work-group - 8 hardware threads per launch to
128. The in-situ before/after, the two bit-identical variants that were worth
**nothing**, and why the ladder's stated mechanism was the wrong invariant are
in [15-step-anatomy.md](15-step-anatomy.md) §L2. The golden gate is unchanged at
96/96 element-exact, which is the bar a reordered summation has to clear.

The `62bdd4d` rows below stay: they are what the step was before any lever, and
every number in docs/05, docs/12's partition and docs/15's anatomy is measured
against them.

Every run's t/s at `62bdd4d`, and what three of them agree on:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 23.72 / 23.73 / 23.74 | **23.73** | 23.72 | 23.74 | 0.02 (0.08%) |
| depth 64, tg 256 | 25.00 / 25.00 / 24.97 | **25.00** | 24.97 | 25.00 | 0.03 (0.12%) |
| depth 64, tg 256, `--max-len 4096` | 25.03 / 25.03 / 25.03 | **25.03** | 25.03 | 25.03 | 0.00 (0.00%) |

A 0.08% spread over three runs is not a rounding artefact of the harness: the
per-run ingest times at depth 4096 were 170432.1 / 170477.7 / 170447.8 ms, i.e.
the same 4096 replays to within 0.03%. A replayed command list with no host work
in it is about as reproducible as a wall clock allows.

The median run's own output - an **excerpt**, not the whole transcript: the
loader's full byte report is 18 more lines and lives in
[13-loader.md](13-loader.md) (its `read/token` and `W check` lines are quoted
below because they are what `W` rests on), and the two-line parenthetical that
follows the `wall` line explains what "host" is and is dropped here for width.

```
  read/token    15539980288 B   15.540 GB
  W check     15.540 GB vs 15.540 GB expected = 15.519 doc-03 + 0.016 pad + 0.005279 widen
              widen = 1337344 B RMSNorm fp32 (1+w) + 3941376 B GDN fp32  ->  -0.000%
  load        13.7 s
engine: 645 kernels, 18 modules, max_len 16384, 1.24 GB of persistent state
ingest: 4096 ids in 170477.7 ms (41.62 ms/token), pos 4096
generate: 256 ids, 23.73 t/s, 42.14 ms/token (99.8% of it inside the fence)
  wall 10788.2 ms, fence 10763.3 ms, host 24.9 ms  (host = wall - fence: argument-free
  replay, so this is submit + the four bytes of shared memory, nothing else)
  per token: 42.141 ms total = 42.044 ms fence + 0.097 ms host
  MBU: 23.73 t/s x 15.540 GB = 369 GB/s of 590 GB/s measured = 62.5%
  dispatch floor (estimated): 645 kernels x 0.52 us = 0.335 ms/token, 0.8% of the fence
```

**`W` = 15.540 GB/token and 590 GB/s are both measured.** `W` is the loader's
own report, printed by the very run above and derived in
[13-loader.md](13-loader.md) ("read/token 15.540 GB", the byte table and the
`W check` line) - 15.519 GB of doc-03 header arithmetic + 0.016 GB tiling pad +
0.005279 GB of fp32-widened norms. The bandwidth is `tools/probe/probe_bw`
through the same Level Zero launch path ([01-hardware.md](01-hardware.md)). On
those two constants the roofline is 26.34 ms/token = **37.97 t/s**, vLLM's 31.50
is **83.0% MBU**, and b70-decode's 23.73 is **62.5%**. (That 62.5% is the
`62bdd4d` row this paragraph sits under and is now historical: the current
engine on this checkpoint is the **`647f2d0` record row at 27.52 t/s = 72.5%
MBU** - see "The record rows" - with the `ef6acb0` gate row at 27.54 t/s
beside it.)

## Notes

- **An image tag is not a version. Read the vLLM version, not the tag.** The
  `vllm-xpu-env-next-p314-t214-vxkp0:latest` that produced the 31.50 t/s bar was
  vLLM `0.27.2rc1.dev365+g5ee84d3c5.d20260821`; the tag was rebuilt 2026-08-26
  and now carries `0.27.2rc1.dev514+g0e30bd62f.d20260826` - 149 commits apart
  under one name. **Every vLLM row above the `urakozz` section was measured on
  the older contents and none has been re-measured**; the 2026-09-04 `urakozz`
  section runs a third build, `0.28.1rc1.dev391+g29af8bd67.d20260904`
  (`p314-t215-vxkp0`), and carries its version in its own rows. The rebuild was verified *correct* on the published
  checkpoint (`vllm_check.py`, 96/96 element-exact - docs/14) but correctness is
  not speed. (A first version of this bullet also claimed the *quantization
  backend* changed with the rebuild. **Withdrawn** - that comparison used two
  different checkpoints, and vLLM picks the backend from `quant_method`. On the
  new build the published checkpoint still takes
  `XPUwNa16LinearKernel for AutoGPTQLinearMethod`, exactly as before. docs/14.)
- The 3.12/2.13 row was run with `--depth 1 2` and no concurrency sweep, so only
  the `d2 c1` column is a like-for-like comparison.
- Torch 2.14 enables Inductor's `batch_linear_lhs` pre-grad fusion on XPU. It
  concatenates GDN projection weights at runtime. Passing
  `--compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'`
  restores MXFP4 decode from 50.22 to 72.46 t/s.
- Prefill is consistently higher on prebuilt-kernel images than source-built
  ones; not a regression to chase.
- Baseline: ~71-72 t/s single-stream decode on Python 3.14 / torch 2.13.
- Torch 2.14+ can return non-contiguous GDN projections. The local patch
  materializes only `num_actual_tokens` rows before calling the SYCL kernel.
- The oracle's greedy tokens for the three golden prompts (`prose`, `code`,
  `cjk`, 32 ids each) exist at `~/b70-inference-server/oracle-out/` on the box
  and are the plan-3 golden gate: the engine must reproduce them exactly.
