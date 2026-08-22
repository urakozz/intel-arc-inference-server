# The box

```bash
ssh user@box
```

Dell T5810. The hostname still says `P620` in places - stale, ignore it.

## Hardware

| | |
|---|---|
| CPU | 44 threads |
| RAM | 121 GB |
| GPU | **2 × Intel Arc Pro B70**, 32 GB each |
| Render nodes | `/dev/dri/renderD128`, `renderD129`, `renderD130` |

`card0`/`renderD128` is typically the display device; select a compute GPU with
`ONEAPI_DEVICE_SELECTOR=level_zero:0` (or `:1` for the second card). Full device
properties are in [01-hardware.md](01-hardware.md).

Only **one** card is used in every measurement so far. XPU graphs are single-GPU
only in vLLM, so TP=2 was never exercised under graphs.

### Known hardware gotcha

The NVMe (Crucial P1, PCH slot) drops out under APST. Kernel needs
`nvme_core.default_ps_max_latency_us=0`. Fixed 2026-08-12 - if the box loses its
disk after a kernel update, check this first.

## Toolchain on the host (not in Docker)

Probed 2026-08-22. The C++ side builds **natively on the host**; no container
is needed for anything except the Python oracle.

| | |
|---|---|
| OS / kernel | Ubuntu 26.04, kernel 7.2.0 |
| GPUs as seen by `lspci` | 2 × `Battlemage G31` (04:00.0, 08:00.0) + an NVIDIA GT 710 for display → `ocloc -device bmg-g31` |
| GPU driver | `libze_intel_gpu.so.1.15.39122`, IGC `2.38.x`, `intel-ocloc 26.27.39122` |
| Level Zero | headers in `/usr/include/level_zero/`, loader in `/usr/lib/x86_64-linux-gnu/` |
| oneAPI | `/opt/intel/oneapi/{2026.0,2026.1}`; `icpx` at `/opt/intel/oneapi/compiler/2026.1/bin/icpx` - **not on `PATH`** in a non-interactive shell; `source /opt/intel/oneapi/setvars.sh` or use the full path |
| Host compilers | `g++ 15.2`, `cmake 4.2.3`, `ccache`; **no `ninja`** on the host (`apt install ninja-build`, or use the one in the image's venv) |
| Python on host | 3.14, **no torch / transformers** |
| Rust | none - the tokenizer dependency (doc 11) needs `rustup` or a prebuilt static lib |

Inside `vllm-xpu-env-next-p314-t214-vxkp0`: `transformers 5.15.0` with
`transformers/models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5GatedDeltaNet`,
`Qwen3_5RMSNormGated`, `Qwen3_5ForConditionalGeneration`, …), `torch
2.14.0+xpu`, `icpx` and `ocloc` also present. `fla`
(**flash-linear-attention**, the Triton GDN kernel library; vLLM vendors its
ops under `vllm/third_party/flash_linear_attention/`) is **not** installed -
deliberately left out: without it transformers runs its pure-torch GDN loop,
which is the reference math in doc 03 and the only one we want the oracle to
speak. It is slow, which is fine for short golden prompts on CPU, and the
dequantised 27B (~54 GB bf16) does not fit a B70 anyway. The container only
sees the GPU with `--device /dev/dri`.

`sycl-tla` targets this part as `-fsycl-targets=spir64_gen` with
`-device bmg-g21,bmg-g31` (`cmake/FindDPCPP.cmake:78-95`); the B70 is G31.

## Models in the HF cache

`~/.cache/huggingface` - **117 GB total**, mounted into containers at
`/root/.cache/huggingface`.

| Size | Model | Notes |
|------|-------|-------|
| 28G | `urakozz/Ornith-1.0-35B-int4-AutoRound` | self-quantised |
| 23G | `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | **phase 4** - MoE + MTP |
| 22G | `olka-fi/Ornith-1.0-35B-MXFP4` | **phase 3** - MoE, MXFP4 |
| 19G | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` | **phase 1 and 2 target.** 64 layers (48 GDN + 16 FA), int4 g64 sym, bf16 `lm_head`, bf16 MTP head. Shipped a broken MTP config - **fixed in the cache** (see below) |
| 14G | `letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound` | broken - does not load in vLLM and more; **no longer a target** |
| 8.4G | `urakozz/Ornith-1.0-9B-MTP-int4-AutoRound` | self-quantised, g64, has MTP head - also not serviceable as of 2026-08-22 |
| 2.6G | `Doopeworld/Qwen3.8-27B-DSpark-vLLM` | draft model for DSpark speculation |
| 2.8M | `gpt2` | smoke tests |

All int4 checkpoints are GPTQ-packed and symmetric; the metadata spelling
differs (`quant_method: "gptq"` + `provider: "auto-round"` on Vishva007,
`"auto-round"` + `packing_format` on the 9Bs - doc 02). Group size varies:
128 (letechlead), 64 (Vishva007, urakozz), 32 (palmfuture 27B).

### Checkpoint quirks that will bite the loader

- **`letechlead/Ornith-1.5-9B`** declares `Qwen3_5ForConditionalGeneration` and
  ships `model.language_model.*` (2177 tensors), `model.visual.*` (549), and
  `lm_head.weight`. It **does not load in vLLM** - see
  [07-open-questions.md](07-open-questions.md) #2. It also carries a
  `model_extra_tensors.safetensors` that overlaps the sharded files, so tensor
  accounting must deduplicate by name.
- **`Vishva007/Qwen3.8-27B`** declared its 15 `mtp.*` tensors quantised via
  AutoRound `dynamic` rules (`"+:.*mtp.*"`) while shipping them **unquantised**.
  Fix is to flip both rules to exclusions (`"-:.*mtp.*"`, `"-:.*mtp\\.fc.*"`) in
  the cached `config.json` - **already applied on the box** (verified
  2026-08-22: the cached config reads `-:`). A fresh download would need it
  again. Our loader trusts the tensor suffixes, not the rules, so it is immune. `palmfuture` ships its exclusion correctly.
- **`Pilcothink/Qwen3.8-27B-MixedInt4`** (not currently cached) mixes widths:
  97 layers bf16, 17 int8, rest int4. vLLM's XPU path rejects the int8 ones.

## Docker images

| Image | Notes |
|-------|-------|
| `vllm-xpu-env-next-p314-t214-vxkp0` | **current reference.** Python 3.14, torch 2.14 RC, kernels source-built pristine (`PATCH_LEVEL=0`) |
| `vllm-xpu-env-next-p314-t214-vxkp12` | same, with the full 12-patch stack - slower at decode, see [09](09-vllm-patch-postmortem.md) |

Also on the box: the serving stack (`lmstack-router`, Grafana, VictoriaMetrics,
Open WebUI) and unrelated `gigachad-grc-*` services. **Never `docker volume
prune`** - it would delete the `open-webui` chat history.

## Running the reference stack for comparison

**The canonical serve and bench commands, and every baseline number, live in
[BENCHMARKS.md](BENCHMARKS.md).** The command below is the short form; when
they disagree, BENCHMARKS.md wins (it adds `--language-model-only`, the
reasoning/tool parsers, and `--speculative-config` for the MTP rows).

```bash
docker run --rm --name ref --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e ZE_FLAT_HIERARCHY=FLAT \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e HF_HUB_OFFLINE=1 \
  -v ~/.cache/vllm:/root/.cache/vllm -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-xpu-env-next-p314-t214-vxkp0 "$MODEL" \
  --host 0.0.0.0 --port 8000 --max-model-len 16k --max-num-seqs 2 \
  --trust-remote-code --enable-prefix-caching \
  --compilation-config '{"inductor_compile_config":{"pre_grad_fusion_options":{}}}'
```

The `pre_grad_fusion_options: {}` override is **required**. torch 2.14 defaults it
to an XPU-only `batch_linear_lhs` fusion that costs ~30% of decode on this
workload.

Benchmark against it with:

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

`uvx` lives at `~/.local/bin/uvx` and is **not** on the default non-interactive
ssh `PATH` - `export PATH="$HOME/.local/bin:$PATH"` first.

## Operational notes

- **Long builds must survive ssh.** BuildKit discards all completed work if the
  docker *client* dies. Use `setsid nohup <script> >/dev/null 2>&1 </dev/null &`
  and poll the log.
- **`docker stop` does not immediately free the container name.** A follow-up
  `docker run --name X` can fail with exit 125; `docker rm -f X` first.
- Model load for a 27B is ~2-4 minutes; graph capture and compile add another
  2-4. Budget ~8 minutes from launch to a servable endpoint.
