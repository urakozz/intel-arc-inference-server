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
runs each, `debug_resid` off, default `--max-len 16384`. Three commits appear:
the phase-1 rows are `62bdd4d`, and the two lever rows are `a1e2d3a` (spec 1.5
lever L2) and `b045e11` (lever L1), all measured the same way on the same day
and box:

| engine | depth | tg | t/s | ms/token |
|---|---|---|---|---|
| vLLM `p314-t214-vxkp0`, no speculation | 1 † | 256 | **31.50** | 31.75 |
| **b70-decode `b045e11`** - spec 1.5 levers L2 + L1 | **4096** † | **256** | **26.28** | **38.05** |
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

The `a1e2d3a` and `b045e11` rows follow the same rule for the same reason: each
is the commit that carries its kernel change and `tools/bench_decode.sh` read
that sha out of the worktree it measured; the rows themselves and the docs
around them are committed after, touching no `src/` file.

**b70-decode is 16.6% short of vLLM** - 26.28 against 31.50, a factor of 1.199
the other way; it was 21.2% short (24.83) after one lever and 24.7% short
(23.73) before any. The decomposition and what it scopes are in
[05-perf-model.md](05-perf-model.md#phase-1-measured--the-honest-verdict).

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

**−2.216 ms/token, and all of it is one kernel family.** `prep_res_norm` ran
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
2.409 ms; the bench falls 2.216 ms. The difference is +0.182 ms of run-to-run
drift on the 516 untouched launches (+0.50%, the same effect §L2 measured at
+0.30%) and +0.094 ms derived for the 129 extra dispatches at 0.733 µs. Predicted
bench delta −2.133 ms against −2.216 measured: 0.083 ms apart, inside the
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
engine is `b045e11` at **26.28 t/s = 69.2% MBU**, the top b70-decode row above.)

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
