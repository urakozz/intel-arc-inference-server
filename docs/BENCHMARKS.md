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

> **Every row below names its checkpoint, and from 2026-08-26 that is
> load-bearing rather than pedantic.** This engine now runs two, and they do not
> read the same number of bytes per token:
>
> | checkpoint | `lm_head` | **W** read/token | roofline @ 590 GB/s |
> |---|---|---|---|
> | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` | bf16 | **15.540 GB** | **37.97 t/s** (26.34 ms) |
> | `qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64` | **int4 g64** | **13.673 GB** | **43.15 t/s** (23.17 ms) |
>
> Both measured by the loader itself (docs/13). A t/s figure is comparable
> across the boundary - it is tokens per second either way - but **MBU and the
> roofline are not**, because W is the denominator of both. Everything in this
> section down to "The new-checkpoint rows" is the **`Vishva007`** checkpoint;
> the vLLM rows above it are too. No number is restated across the boundary.

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
runs each, `debug_resid` off, default `--max-len 16384`. Five commits appear:
the phase-1 rows are `62bdd4d`, the three lever rows are `a1e2d3a` (spec
1.5 lever L2), `b045e11` (lever L1) and `c746840` (lever L5), and `ef6acb0` is
the spec 1.5 **gate** - all measured the same way on the same day and box:

| engine | depth | tg | t/s | ms/token |
|---|---|---|---|---|
| vLLM `p314-t214-vxkp0`, no speculation | 1 † | 256 | **31.50** | 31.75 |
| **b70-decode `ef6acb0`** - the spec 1.5 gate, levers L2 + L1 + L5 | **4096** † | **256** | **27.54** | **36.32** |
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

**b70-decode is 12.6% short of vLLM** - 27.54 against 31.50, a factor of 1.144
the other way; it was 16.6% short (26.28) after two levers, 21.2% short (24.83)
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

**Grade: ITERATE, not RECORD.** Single runs, not the median of three
`tools/bench_decode.sh` produces, and the box was **not idle** - a 12-core vLLM
XPU kernel compile ran throughout, and for part of the window a 22-thread oracle
dump as well. A record-grade median-of-three row on a quiet box is a named
follow-on and has not been taken.

| engine / checkpoint                                                              | depth | tg  | t/s       | ms/token  | MBU                | grade                         |
|----------------------------------------------------------------------------------|-------|-----|-----------|-----------|--------------------|-------------------------------|
| **b70-decode, `qwen38-27b-w4g64-rtn`** (int4 `lm_head`)                          | 4096  | 64  | **30.20** | **33.11** | 70.0% of 13.673 GB | iterate, 1 run, box loaded    |
| b70-decode, `Vishva007` (bf16 `lm_head`) - **the control, same hour, same load** | 4096  | 64  | 27.57     | 36.27     | 72.6% of 15.540 GB | iterate, 1 run, box loaded    |
| b70-decode `ef6acb0`, `Vishva007` - the standing record row                      | 4096  | 256 | 27.54     | 36.32     | 72.5% of 15.540 GB | **record**, median of 3, idle |

**The control row is what makes the pair readable.** `Vishva007` under load read
**36.27** against its standing idle median of **36.32** - 0.14% apart. A decode
step that is 99.7% inside the fence does not contend with a CPU compile, and
that is the evidence for it rather than an assumption. The two iterate rows were
run back to back, so the difference between them is the checkpoint.

**Δ = +2.63 t/s, −3.16 ms/token** against the same-hour control; **+2.66 t/s,
−3.21 ms** against the standing record row. In-situ, the `lm_head` launch alone
moved **4381.289 → 1178.164 µs**, −3.203 ms/token, which is **98.3% of the
−3.26 ms ceiling** the re-assessment memo derived from the bytes.

Note `tg`: **64, not 256.** These are iterate-grade runs and 64 was chosen to
keep the turnaround short on a loaded box. The standing record row is `tg 256`,
so the two are not the same measurement even ignoring the median - another
reason the record row is still owed.

**MBU falls while throughput rises**, from 72.6% to 70.0%, and that is the
memo's own prediction: `lm_head` at bf16 was the most bandwidth-efficient launch
in the step (98.4% of device), so removing three quarters of its bytes lowers
the average of what remains. The bench's MBU line divided by a hardcoded
15.540 GB until 2026-08-26 and would have printed 79.4% here; it now reads W
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
engine is the `ef6acb0` gate row at **27.54 t/s = 72.5% MBU**, the top
b70-decode row above.)

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
