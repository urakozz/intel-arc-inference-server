# intel-inference-server

**A fast LLM server for one or two Intel Arc Pro B70 cards, written from scratch.** Hand-written Level Zero / OpenCL
kernels tuned for each model it runs - no PyTorch, no vLLM underneath - and an OpenAI-compatible API built for long
coding-agent sessions (opencode and similar).

## Key features

- **Fast speculative decoding, on by default:** each model's MTP head, or a DFlash 2 drafter - **282 t/s** writing
  prose with Inkling 2 Mini on one card, **395 t/s** on two; **167 t/s** writing code with Qwen3.8-27B on one card.
- **Fast prompts:** up to **11,223 t/s** prefill on one card (Ornith 1.5), **7,194 t/s** on two (Inkling 2 Mini).
- **Long context on modest hardware:** Qwen3.8-27B's full 262,144 tokens on **a single 32 GB card**; Inkling 2 Mini's
  full **1M tokens** on two.
- **Several requests at once:** up to 4 per card sharing one KV cache - **626 t/s** in total with 8 requests on two
  cards (Qwen3.8-27B).
- **One choice to make:** how many requests run at once. The engine sizes context, cards and caches from it, and
  refuses at startup, naming why, anything that does not fit.
- **Exact:** speculative decoding never changes the output, and the kernels are checked against float32 references.

## Performance

One session, greedy, served, tokens a second. All setups, more context lengths and several requests:
[docs/benchmarks.md](docs/benchmarks.md).

| model | setup | prefill | code | prose | tool calls |
|---|---|---:|---:|---:|---:|
| Inkling 2 Mini | one card, DFlash 2 | 4,932 (4k, int8) | **232** | **282** | **179** |
| Inkling 2 Mini | two cards, DFlash 2 (`tp`; prefill `pp`) | 7,194 (16k, int8) | **275** | **395** | **244** |
| Qwen3.8-27B | one card, DFlash 2 | 2,800 (4k, int8) | **167** | 67 | 118 |
| Qwen3.8-27B | two cards `tp`, DFlash 2 | 3,467 (4k, int8) | 182 | **89** | **153** |
| Ornith 1.5 35B-A3B | one card, MTP | **11,223** (4k) | **320** | **341** | **287** |
| Qwen3.8-Flash-Next | two cards, MTP (`tp`; prefill `pp`) | 3,415 (32k) | 99 | 64 | 84-126 |

Prefill is the model reading your prompt; code, prose and tool calls are it writing its answer.

## Models

The Qwen-family models run from our own int4 quantisation (AutoRound, group 64, GPTQ format) on Hugging Face;
Inkling 2 Mini runs its own release:

- **[Qwen3.8-Flash-Next](https://huggingface.co/urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ)** - MoE, 512
  experts (10 a token), linear attention beside sparse attention, MTP head. Two cards and 128 GB of RAM.
- **[Qwen3.8-27B](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ)** - dense, linear attention
  beside full attention, MTP head. One card, the full 262,144-token context; with the
  [DFlash drafter](https://huggingface.co/urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound) for faster writing.
- **[Ornith 1.5 35B-A3B](https://huggingface.co/urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)** - MoE (the
  Qwen3.5-35B-A3B architecture, tuned for code), MTP head. One card.
- **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** - looped MoE (24 layers run as 48
  visits), the release itself (its NVFP4 / MXFP8 experts run exactly), with the
  [DFlash 2 drafter](https://huggingface.co/urakozz/Inkling-2-Mini-DFlash2-W4A16-g64-AutoRound) for faster writing. One card; two cards
  `tp` for one faster session, `pp` for the 1,048,576-token context with `--kv-cache int8`.

## Install

The host needs Linux with the `xe` kernel driver (the container uses the host's kernel); the fast single session on two cards needs peer-to-peer between them (native
from kernel 7.4). The image brings the rest:

```sh
docker pull urakozz/intel-arc-inference-server:1.2.1

# the engine never downloads; fetch the model into the Hugging Face cache once
# Qwen3.8-Flash-Next, about 133 GB: skips 96 GB of the bf16 n-gram table, which only other engines read
hf download urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ \
  --exclude "model-0000[4-9]-*" --exclude "model-0001?-*" --exclude "model-0002[0-3]-*"

# Qwen3.8-27B - all-round classics
hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
hf download urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound   # the DFlash drafter

# Ornith 1.5 (Qwen3.5-35B-A3B architecture, code optimised)
hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ

# Inkling 2 Mini (looped MoE), 20.2 GB, and its drafter
hf download thinkingmachines/Inkling-2-Mini
hf download urakozz/Inkling-2-Mini-DFlash2-W4A16-g64-AutoRound   # its drafter
```

Tarball (no Docker): `intel-inference-server-<version>-linux-x86_64.tar.gz`, extracting to
`intel-inference-server-<version>/` with `bin/b70-serve`, `bin/b70-decode` and the compiled kernels. The host
additionally needs the `xe` kernel driver, Intel's compute runtime (`libze-intel-gpu1`, 26.40.40196.15 or newer,
from Intel's graphics PPA `ppa:kobuk-team/intel-graphics` or the distro) and the Level Zero loader (`libze1`,
1.34.0 or newer); glibc 2.43 or newer. See the tarball's own README.md for details.

```sh
tar -xzf intel-inference-server-1.2.1-linux-x86_64.tar.gz
./intel-inference-server-1.2.1/bin/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last --kv-cache int8
```

Or build it yourself: the [Dockerfile](Dockerfile) (`docker build -t urakozz/intel-arc-inference-server .`), or
directly with Intel's compute runtime with `ocloc` and the Level Zero loader, GCC 16 (`g++-16.2`), CMake and Rust (for
the tokenizer):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Run

```sh
# an OpenAI-compatible server on port 8000 (built from source: ./build/src/cli/b70-serve in place of $B70);
# GET /health answers 503 while the model loads, 200 once it serves
B70="docker run --rm --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 urakozz/intel-arc-inference-server:1.2.1"

# Qwen3.8-Flash-Next on two cards: every layer split over both, the experts in host RAM computed on the CPU (the
# defaults: the fastest session), MTP on
$B70 urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last
# context: 65,536 by default; --max-len 131072 or 262144 for longer, with fewer experts kept in VRAM

# Qwen3.8-27B
# Fastest: the DFlash drafter + the int8 prompt pass (opt-in)
$B70 urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last \
    --spec dflash \
    --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound \
    --prefill-backend l0-int8

# Full context (262,144) on one 32 GB card: the MTP head (the default) + the int8 KV cache
$B70 urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last \
    --kv-cache int8

# Ornith 1.5 35B-A3B: its own MTP head, the default prompt pass
$B70 urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ --prefix-split-last

# Inkling 2 Mini: one card, its drafter on by default
$B70 thinkingmachines/Inkling-2-Mini
```

You choose one thing: **how many requests run at once**, `--max-num-seqs N`. The engine sizes everything else
from it - context, cards, cache - and refuses at startup, naming why, anything that does not fit.

- **`N = 1` (the default):** one request with the largest context that fits. On two cards every layer is split over
  both (`--parallel tp`) - the fastest single session, at the model's full 262,144 tokens. On one card add
  `--kv-cache int8` to get the full context there too.
- **`N = 2` to `4` on one card, up to `8` on two:** shaped for a coding agent with subagents. On one card the
  requests share the card's KV cache: the main session can use most of it while its subagents run short, and a
  finished session's memory goes to the next request that needs it. A 123k-token prompt runs beside two short ones,
  and an idle session pushed out to make room resumes from the prefix cache in about a second. On two cards every
  card runs its own copy of the model (`--parallel replica`) and its requests share that card's KV cache the same
  way; so do they beside the DFlash draft model (`--spec dflash`).

Speculative decoding and the prefix cache are on by default. Speculative decoding and `--host-experts` switch on a
running server without a reload (`curl localhost:8000/v1/control/reconfigure -d '{"spec": "off"}'`).

All recipes with their context sizes, and every `b70-serve` flag: [docs/17-running-models.md](docs/17-running-models.md).

## Docs

[docs/README.md](docs/README.md) is the index. Start with [docs/04-architecture.md](docs/04-architecture.md) for the
engine, [docs/17-running-models.md](docs/17-running-models.md) for running it, [docs/12-kernels.md](docs/12-kernels.md)
for the kernels.

## Licence

AGPL-3.0-only (see [LICENSE](LICENSE)), with a commercial licence for companies that cannot work under the AGPL.
[LICENSING.md](LICENSING.md) explains which one applies to you.

Written by Jürgen Kozyrev.
