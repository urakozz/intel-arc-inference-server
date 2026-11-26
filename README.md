# b70-inference-server

**A fast LLM server for one or two Intel Arc Pro B70 cards, written from scratch.** Hand-crafted GPU kernels,
optimised for each model it runs - no PyTorch, no vLLM underneath - and an OpenAI-compatible API built for long coding-agent sessions (opencode and similar).

- **Qwen3.8-Flash-Next:** prefill at **3,415 t/s**, code written at **99 t/s** (two cards, experts offloaded to RAM and computed on the CPU).
- **Qwen3.8-27B:** **167 t/s** writing code for one session on one card (DFlash), **343 t/s** for 4 requests; **626 t/s** in total with 8 requests on two cards.
- Prefill at **2,800 t/s** on one card / **3,467 t/s** on two cards with `tp` (opt-in int8 prefill engine).
- The full 262,144-token context of Qwen3.8-27B fits on **a single card**.
- **Inkling 2 Mini:** prefill at **4,987 t/s** on one card and **7,194 t/s** on two (`pp`, 16k prompt); prose written at
  **248 t/s** and code at **194 t/s** on one card, **310 t/s** prose on two cards (`tp`); the full **1,048,576-token**
  context on two cards.
- **Ornith 1.5 35B-A3B:** prefill at **11,223 t/s**, answers written at **312 t/s** on one card, **581 t/s** for two requests on two cards.

One choice to make - how many requests run at once. The engine sizes everything else.

## Supported models

Each model runs from our own int4 quantisation (AutoRound, group 64, GPTQ format) on Hugging Face:

- **[Qwen3.8-Flash-Next](https://huggingface.co/urakozz/Qwen3.8-Flash-Next-W4A16-g64-AutoRound-GPTQ)** - MoE, 512
  experts (10 a token), linear attention beside sparse attention, MTP head. Two cards and 128 GB of RAM.
- **[Qwen3.8-27B](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ)** - dense, linear attention
  beside full attention, MTP head. One card, the full 262,144-token context; with the
  [DFlash drafter](https://huggingface.co/urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound) for faster writing.
- **[Ornith 1.5 35B-A3B](https://huggingface.co/urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)** - MoE (the
  Qwen3.5-35B-A3B architecture, tuned for code), MTP head. One card.
- **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** - looped MoE (24 layers run as 48
  visits), the release itself (its NVFP4 / MXFP8 experts run exactly), with the
  [DFlash 2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) for faster writing. One card; two cards
  `tp` for one faster session, `pp` for the 1,048,576-token context with `--kv-cache int8`.

## Speed

**Prefill** is the model reading your prompt, **decode** is the model writing its answer

### Qwen3.8-Flash-Next

Prefill, t/s:

| setup | 32k-token prompt | 128k-token prompt |
|---|---:|---:|
| two cards `pp` | **3,415** | **2,918** |
| two cards `tp` | 2,234 | 1,980 |

A short prompt (43 tokens) gets its first token in 0.30 s on `pp` and on `tp`

Generation, t/s:

| setup | code | prose | tool calls |
|---|---:|---:|---:|
| MTP, two cards `pp` | 79 | 63 | 98-123 |
| MTP, two cards `tp` | **99** | **64** | **84-126** |
| MTP, two cards `tp`, PCIe 5.0 x16 + DDR5 | ~125-155 | | |

Flash-Next needs at least 128 GB of system RAM: the experts that do not fit VRAM, its n-gram table and the prefix
cache take about 104 GB of it

Measured on a PCIe 3.0 workstation with DDR4-2133 and P2P. PCIe 5.0 x16, DDR5 RAM, AVX-512 CPU speed it up

### Inkling 2 Mini

| setup | prefill, 4k prompt | code | prose | tool calls |
|---|---:|---:|---:|---:|
| one card | 4,780 | 193.5 | 247.6 | 147.2 |
| two cards `pp` | 5,976 | 182.5 | 243.8 | 145.9 |
| two cards `tp` | 6,173 | 216.8 | 309.8 | 187.9 |
| 2 requests, two cards `replica` (total) | - | 351 | 452 | 286 |

| prefill (`l0-int8`, int8 cache), t/s | 4k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|
| one card | 4,932 | 4,506 | 3,939 | 3,137 |
| two cards `pp` | 5,976 | **7,194** | **6,908** | **5,842** |
| two cards `tp` | **6,173** | 6,384 | 5,794 | 4,801 |

| two cards `pp`, int8 cache, the drafter on | prompt | decode |
|---|---:|---:|
| 128k context | 3,710 | 47.2 |
| 1 Mi context (a 1,039,565-id prompt) | 762 | 15.1 |

### Qwen3.8-27B

Prefill, t/s:

| setup | 4k prompt | 16k prompt | 32k prompt | 64k prompt | 128k prompt |
|---|---:|---:|---:|---:|---:|
| one card, MTP on (the default server) | 1,908 | 1,771 | 1,634 | 1,395 | 1,072 |
| one card, MTP on, `--prefill-backend l0-int8` (opt-in) | **2,800** | **2,564** | **2,346** | **1,985** | **1,500** |
| two cards, `replica`, 2 requests (total), `--prefill-backend l0-int8` (opt-in; 32k derived) | **4,610** | - | ~4,000 | - | - |

Generation, t/s (several requests: the total; llama-benchy tg128 after 4k / 32k / 128k tokens of context, median of 3):

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| MTP, 1 request, one card | 135 | 50 | 102 | 67 / 58 / 51 |
| MTP, 1 request, two cards `tp` | 138 | 74 | 115 | 95 / 84 / **69** |
| MTP, 4 requests, one card | 237 | 140 | 122 | |
| MTP, 2 requests, two cards `replica` | 168 | 99 | 144 | |
| MTP, 4 requests, two cards `replica` | 283 | 172 | 168 | |
| MTP, 8 requests, two cards `replica` | 435 | 268 | 206 | |
| DFlash, 1 request, one card | **167** | 66 | 117 | 94 / 83 / 55 (`--kv-cache int8`) |
| DFlash, 1 request, two cards `tp` | - | - | - | **119** / **107** / 68 |
| DFlash, 2 requests, one card | 233 | 98 | 114 | |
| DFlash, 4 requests, one card | **343** | 156 | 128 | |
| DFlash, 2 requests, two cards `replica` | 304 | 126 | 151 | |
| DFlash, 4 requests, two cards `replica` | 433 | 189 | 199 | |
| DFlash, 8 requests, two cards `replica` | **626** | **281** | **211** | |

DFlash is `--spec dflash --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound`

### Ornith 1.5 35B-A3B

Prefill, t/s:

| setup | 4k prompt | 16k prompt | 32k prompt | 64k prompt | 128k prompt | 262k prompt |
|---|---:|---:|---:|---:|---:|---:|
| one card | 11,223 | 8,800 | 7,420 | 5,428 | 3,544 | 2,096 |
| two cards `replica`, 2 requests (total) | **19,117** | **13,728** | - | - | - | - |

Generation with MTP, t/s (llama-benchy as above):

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| 1 request, one card | 312 | 326 | 264 | **196** / 126 / 62 |
| 2 requests, two cards `replica` (total) | **581** | **605** | **453** | |
| 4 requests, one card | ~430 | | | |

## Install

The host needs Linux with the `xe` kernel driver (the container uses the host's kernel); the fast single session on two cards needs peer-to-peer between them (native
from kernel 7.4). The image brings the rest:

```sh
docker pull urakozz/intel-arc-inference-server:1.1.0

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
hf download incoai/Inkling-2-Mini-DFlash2   # its drafter
```

Or build it yourself: the [Dockerfile](Dockerfile) (`docker build -t urakozz/intel-arc-inference-server .`), or
directly with Intel's compute runtime with `ocloc` and the Level Zero loader, GCC 16 (`g++-16.2`), CMake and Rust (for
the tokenizer):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

[The model](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ) is Qwen3.8-27B compressed to 4-bit
weights with AutoRound, in groups of 64 rather than the usual 128 - better accuracy for about 2 % of prefill speed.

## Run

```sh
# an OpenAI-compatible server on port 8000 (built from source: ./build/src/cli/b70-serve in place of $B70);
# GET /health answers 503 while the model loads, 200 once it serves
B70="docker run --rm --device /dev/dri --group-add $(getent group render | cut -d: -f3) \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 urakozz/intel-arc-inference-server:1.1.0"

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

Speculative decoding (the model's MTP head) and the prefix cache are on by default; the DFlash draft model is
`--spec dflash --draft-model ...`. Speculative decoding and `--host-experts` switch on a running server without a reload
(`curl localhost:8000/v1/control/reconfigure -d '{"spec": "off"}'`). Ready-made recipes with their context sizes:
[docs/17-running-models.md](docs/17-running-models.md).

Other models load too (Ornith 1.5, Qwen3.8-Flash-Next,
Kolibri-1); Qwen3.8 is the one everything is tuned for.

## Flags

`b70-serve`:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | a Hugging Face repo id in the local cache, or a snapshot directory; never downloads |
| `--max-num-seqs N` | `1` | how many requests run at once. They share the card's KV cache, and any one may use all of it; two cards run a model copy each, with its own cache. Up to 4 per card |
| `--max-len auto\|L` | `auto` | context length (the most one request may use with `--max-num-seqs`). `auto`: the largest that fits, up to the model's 262,144 |
| `--parallel auto\|1\|tp\|replica\|pp` | `auto` | `tp`: every layer split over both cards (one request; needs peer-to-peer between them). `replica`: a model copy per card. `pp`: the layers split between the cards, for models that do not fit one card. `1`: one card |
| `--spec auto\|off\|mtp\|dflash` | `auto` | speculative decoding: a proposer guesses tokens, the model checks them all in one pass; the output never changes. `auto` is the model's MTP head (no extra memory). `dflash` (with `--draft-model`) uses a draft model instead: faster, but its weights take ~1.2 GB of the context's memory; it also copies repeats of earlier text |
| `--spec-max K` | the most | the proposer's cap on guesses per step |
| `--draft-model PATH` | none; required with `--spec dflash` | the draft model: `urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound` (our packed int4 drafter, 1.21 GB of the context's memory) |
| `--kv-cache bf16\|int8` | `bf16` | how the context is stored; `int8` halves it (more context, more requests) |
| `--prefill-backend l0\|l0-int8` | `l0` | the prompt pass: bf16, or int8 on the matrix engines (faster; Qwen3.8 and Ornith). On Qwen3.8 `int8` means every prompt linear, the context (it implies `--kv-cache int8`) and the attention scores in int8: 40-47 % faster prompts; measured against the bf16 pass, the most likely next token differs at 3 of 585 positions and 0 of 12 long exact-copy tool calls changed. On Ornith: its experts in int8 |
| `--prefix-cache-gb auto\|N` | `auto` | the prefix cache in pinned system RAM, locked against swap when `ulimit -l` is unlimited; `auto` = up to 32 GiB from what is free, `0` = off |
| `--prefix-split-last` | off | for chat clients that drop the reasoning (opencode): the next turn is a prefix cache hit exactly where it diverges. On for agentic sessions, off for benchmarks |
| `--host H`, `--port P` | `0.0.0.0`, `8000` | where it listens |
| `--log-requests DIR` | off | one JSON file per request: timings, body, ids, text |
| `--layers N\|auto` | - | Qwen3.8-Flash-Next only: serve its first N layers |
| `--host-experts auto\|gpu\|cpu` | `auto` | Qwen3.8-Flash-Next: how experts held in host RAM are served. `auto` picks per step from a startup measurement, `gpu` reads them over PCIe into the card, `cpu` computes them on the CPU. Same output either way |

`b70-decode` (benchmarks, profiles, decoding from token ids) and the environment variables: [docs/18-development.md](docs/18-development.md).

## Docs

[docs/README.md](docs/README.md) is the index. Start with [docs/04-architecture.md](docs/04-architecture.md) for the
engine, [docs/17-running-models.md](docs/17-running-models.md) for running it, [docs/12-kernels.md](docs/12-kernels.md)
for the kernels.

## Licence

AGPL-3.0-only (see [LICENSE](LICENSE)), with a commercial licence for companies that cannot work under the AGPL.
[LICENSING.md](LICENSING.md) explains which one applies to you.

Written by Jürgen Kozyrev.
