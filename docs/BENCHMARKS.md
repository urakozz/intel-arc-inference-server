# B70 image benchmarks

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
  --max-model-len 16k --kv-cache-dtype auto --max-num-seqs 2 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --language-model-only --trust-remote-code --enable-prefix-caching \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
  # speculative row adds:
  # --speculative-config '{"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
  # --speculative-config '{"method":"mtp","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
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

## b70-decode - this project, phase 1

Same box, same checkpoint, same `--max-model-len 16k` / `pp4096` / `depth 1` /
`concurrency 1` shape as the vLLM rows above, so the `tg256` columns compare
directly. This engine has no prefill kernel yet (a prompt costs one decode
replay per id - spec 2 owns prefill), so there is no `pp4096` number to put
beside vLLM's 1973; the ingest rate is recorded instead, and it is the decode
rate, because it *is* decode.

```bash
tools/bench_decode.sh                 # 3 runs at depth 4096, tg 256; median + spread
tools/bench_decode.sh --depth 64      # the doc 07 #12 depth experiment
```

**Measured 2026-08-25** on an idle box (no container, no other GPU work), three
runs each, `b70-decode` at commit `62bdd4d`, `debug_resid` off, default
`--max-len 16384`:

| engine | depth | tg | t/s | ms/token |
|---|---|---|---|---|
| vLLM `p314-t214-vxkp0`, no speculation | 1 | 256 | **31.50** | 31.75 |
| **b70-decode `62bdd4d`** | **4096** | **256** | **23.73** | **42.14** |
| b70-decode `62bdd4d` | 64 | 256 | 25.00 | 40.00 |

**b70-decode is 24.7% short of vLLM** - 23.73 against 31.50, a factor of 1.327
the other way. The number is recorded as measured; nothing was tuned to produce
it, and no `.cl` file was touched in the task that measured it. The
decomposition and what it scopes are in
[05-perf-model.md](05-perf-model.md#phase-1-measured--the-honest-verdict).

Every run, verbatim:

| shape | runs (t/s) | median | min | max | spread |
|---|---|---|---|---|---|
| depth 4096, tg 256 | 23.72 / 23.73 / 23.74 | **23.73** | 23.72 | 23.74 | 0.02 (0.08%) |
| depth 64, tg 256 | 25.00 / 25.00 / 24.97 | **25.00** | 24.97 | 25.00 | 0.03 (0.12%) |
| depth 64, tg 256, `--max-len 4096` | 25.03 / 25.03 / 25.03 | **25.03** | 25.03 | 25.03 | 0.00 (0.00%) |

A 0.08% spread over three runs is not a rounding artefact of the harness: the
per-run ingest times at depth 4096 were 170432.1 / 170477.7 / 170447.8 ms, i.e.
the same 4096 replays to within 0.03%. A replayed command list with no host work
in it is about as reproducible as a wall clock allows.

The median row's own breakdown (run 2, the median run):

```
ingest: 4096 ids in 170477.7 ms (41.62 ms/token), pos 4096
generate: 256 ids, 23.73 t/s, 42.14 ms/token (99.8% of it inside the fence)
  wall 10788.2 ms, fence 10763.3 ms, host 24.9 ms
  per token: 42.141 ms total = 42.044 ms fence + 0.097 ms host
  MBU: 23.73 t/s x 15.540 GB = 369 GB/s of 590 GB/s measured = 62.5%
  dispatch floor (estimated): 645 kernels x 0.52 us = 0.335 ms/token, 0.8% of the fence
```

**`W` = 15.540 GB/token and 590 GB/s are both measured** - the loader's own
byte report (docs/13) and `tools/probe/probe_bw` through the same Level Zero
launch path (docs/01). On those two constants the roofline is 26.34 ms/token =
**37.97 t/s**, vLLM's 31.50 is **83.0% MBU**, and b70-decode's 23.73 is
**62.5%**.

## Notes

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
