# b70-inference-server

**A fast LLM server for one or two Intel Arc Pro B70 cards, written from scratch.** Hand-crafted GPU kernels,
optimised for each model it runs - no PyTorch, no vLLM underneath - and an OpenAI-compatible API built for long coding-agent sessions (opencode and similar).

- **Qwen3.8-27B:** **164 t/s** writing code for one session on one card (DFlash); **342 t/s** in total with 8 requests on two cards.
- **Prompts read at 2,083 t/s on one card, 3,217 t/s on two** (Qwen3.8-27B, 4,096 tokens, the opt-in int8 prompt pass).
- The full 262,144-token context of Qwen3.8-27B fits on **a single card**.
- **Ornith 1.5 35B-A3B:** prompts read at **10,798 t/s**, answers written at **312 t/s** on one card, **581 t/s** for two requests on two cards.
- **Qwen3.8-Flash-Next:** prompts read at **3,103 t/s**, code written at **82-92 t/s** (two cards, experts offloaded to RAM).

One choice to make - how many requests run at once. The engine sizes everything else.

## Speed

**Prefill** is the model reading your prompt, **decode** is the model writing its answer, both in tokens per second
(t/s). **MTP** - the model drafts several tokens ahead and checks them in one step; the answer does not change, it
arrives faster. A **prefix cache hit** is a follow-up turn whose earlier conversation is already cached.

### Qwen3.8-27B

Prefill, t/s:

| setup | 4k prompt | 16k prompt |
|---|---:|---:|
| one card | 1,788 | 1,624 |
| one card, `--prefill-backend l0-int8` (opt-in) | 2,083 | 1,845 |
| two cards, `tp` | 2,931 | 2,683 |
| two cards, `tp`, `--prefill-backend l0-int8` (opt-in) | **3,217** | **2,937** |

Generation, t/s (several requests: the total; llama-benchy tg128 at 0 / 32k tokens of context):

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| MTP, 1 request, one card | 138 | 57 | 100 | 121 / 61 |
| MTP, 1 request, two cards `tp` | 123 | 74 | 115 | 112 / **88** |
| MTP, 2 requests, two cards `replica` | 164 | 108 | 144 | |
| MTP, 4 requests, two cards `replica` | 282 | 189 | 206 | |
| MTP, 8 requests, two cards `replica` | **342** | **300** | **233** | |
| DFlash, 1 request, one card | **164** | 65 | 111 | **139** / 83 |
| DFlash, 2 requests, two cards `replica` | 317 | 127 | 191 | |

DFlash is `--spec dflash --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound`.

### Ornith 1.5 35B-A3B

Prefill, t/s:

| setup | 4k prompt | 16k prompt |
|---|---:|---:|
| one card | 10,798 | 8,632 |
| two cards `replica`, 2 requests (total) | **18,674** | **13,444** |

Generation with MTP, t/s:

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| 1 request, one card | 312 | 326 | 264 | **210** / 131 |
| 2 requests, two cards `replica` (total) | **581** | **605** | **453** | |

### Qwen3.8-Flash-Next

Prefill, t/s:

| setup | 32k-token prompt | 128k-token prompt |
|---|---:|---:|
| two cards `pp` | **3,103** | **2,458** |
| two cards `tp` | 2,128 | 1,988 |

A short prompt (43 tokens) gets its first token in 0.38 s on `pp`, 0.50 s on `tp`.

Generation, t/s:

| setup | code | prose | tool calls |
|---|---:|---:|---:|
| plain, two cards `pp` | 58 | 49 | 49-60 |
| plain, two cards `tp` | 63 | 51 | 55-62 |
| MTP, two cards `pp` | 81 | 56 | 80-105 |
| MTP, two cards `tp` | **82-92** | **60-65** | **89-99** |

Flash-Next needs at least 128 GB of system RAM: the experts that do not fit VRAM, its n-gram table and the prefix
cache take about 104 GB of it. Run nothing else memory-hungry beside the server.

How every number was taken: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

You need Linux with the `xe` driver, Intel's compute runtime with `ocloc` and the Level Zero loader, GCC 16 (`g++-16.2`), CMake and
Rust (for the tokenizer). The fast single session on two cards needs peer-to-peer between them (native from kernel
7.4).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# the engine never downloads; fetch the model into the Hugging Face cache once
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound   # the DFlash drafter
uvx --from huggingface_hub hf download urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
```

[The model](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ) is Qwen3.8-27B compressed to 4-bit
weights with AutoRound, in groups of 64 rather than the usual 128 - better accuracy for about 2 % of prefill speed.

## Run

```sh
# an OpenAI-compatible server on port 8000

# Qwen3.8-27B
# Fastest: the DFlash drafter + the int8 prompt pass (opt-in)
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last \
    --spec dflash \
    --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound \
    --prefill-backend l0-int8

# Full context (262,144) on one 32 GB card: the MTP head (the default) + the int8 KV cache
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --prefix-split-last \
    --kv-cache int8


# Ornith 1.5 35B-A3B: its own MTP head, the default prompt pass
./build/src/cli/b70-serve urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ --prefix-split-last
```

You choose one thing: **how many requests run at once**, `--max-num-seqs N`. The engine sizes everything else
from it - context, cards, cache - and refuses at startup, naming why, anything that does not fit.

- **`N = 1` (the default):** one request with the largest context that fits. On two cards every layer is split over
  both (`--parallel tp`) - the fastest single session, at the model's full 262,144 tokens. On one card add
  `--kv-cache int8` to get the full context there too.
- **`N = 2` to `4` on one card, up to `8` on two:** shaped for a coding agent with subagents. The first slot (one
  per card on two cards) gets the largest context that fits - the main session; the other slots get a shorter
  context, enough for subagent tasks. On two cards every card runs its own copy of the model (`--parallel replica`).

Speculative decoding (the model's MTP head) and the prefix cache are on by default; the DFlash draft model is
`--spec dflash --draft-model ...`. Ready-made recipes with their context sizes:
[docs/17-running-models.md](docs/17-running-models.md).

```sh
# Qwen3.8-Flash-Next on two cards: our int4 g64 export, a local snapshot directory
./build/src/cli/b70-serve <snapshot> --parallel pp
```

Other models load too (Ornith 1.5, Qwen3.8-Flash-Next,
Kolibri-1); Qwen3.8 is the one everything is tuned for.

## Flags

`b70-serve`:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | a Hugging Face repo id in the local cache, or a snapshot directory; never downloads |
| `--max-num-seqs N` | `1` | how many requests run at once; the context follows. Slot 0 gets the largest context that fits beside N - 1 slots of 32,768 tokens. Up to 4 per card; on two cards N > 1 runs a model copy per card |
| `--max-len auto\|L` | `auto` | context length (slot 0's with `--max-num-seqs`). `auto`: the largest that fits, up to the model's 262,144 |
| `--parallel auto\|1\|tp\|replica\|pp` | `auto` | `tp`: every layer split over both cards (one request; needs peer-to-peer between them). `replica`: a model copy per card. `pp`: the layers split between the cards, for models that do not fit one card. `1`: one card |
| `--spec auto\|off\|mtp\|dflash` | `auto` | speculative decoding: a proposer guesses tokens, the model checks them all in one pass; the output never changes. `auto` is the model's MTP head (no extra memory). `dflash` (with `--draft-model`) uses a draft model instead: faster, but its weights take ~1.2 GB of the context's memory; it also copies repeats of earlier text |
| `--spec-max K` | the most | the proposer's cap on guesses per step |
| `--draft-model PATH` | none; required with `--spec dflash` | the draft model: `urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound` (our packed int4 drafter, 1.21 GB of the context's memory). |
| `--kv-cache bf16\|int8` | `bf16` | how the context is stored; `int8` halves it (more context, more requests) |
| `--prefill-backend l0\|l0-int8` | `l0` | the prompt pass: bf16, or int8 on the matrix engines (faster; Qwen3.8 and Ornith) |
| `--prefix-cache-gb auto\|N` | `auto` | the prefix cache in pinned system RAM; `auto` = up to 32 GiB from what is free, `0` = off |
| `--prefix-split-last` | off | for chat clients that drop the reasoning (opencode): the next turn is a prefix cache hit exactly where it diverges. On for agentic sessions, off for benchmarks |
| `--host H`, `--port P` | `0.0.0.0`, `8000` | where it listens |
| `--log-requests DIR` | off | one JSON file per request: timings, body, ids, text |
| `--layers N\|auto` | - | Qwen3.8-Flash-Next only: serve its first N layers |

`b70-decode` (greedy decoding from token ids, benchmarks, profiles):

| flag | default | what it does |
|---|---|---|
| `--ids FILE` | - | prompt ids, whitespace-separated; generated ids go to stdout |
| `--n N` | - | ids to generate |
| `--prefill` | off | run the prompt through the chunked prefill |
| `--spec off\|mtp\|dflash`, `--spec-max K` | `off` | as for `b70-serve` |
| `--max-len L\|auto`, `--parallel auto\|1\|pp\|tp`, `--kv-cache`, `--prefill-backend` | `16384`, ... | as for `b70-serve` |
| `--layers N\|auto` | - | load only the first N layers (Kolibri-1, Qwen3.8-Flash-Next) |

Developer flags (`b70-decode`):

| flag | default | what it does |
|---|---|---|
| `--bench` | - | time `--tg` generated ids after `--depth` ids of context (or a `--prefill-length N` prompt); prints a markdown row |
| `--depth D` | `4096` | context before decoding, for `--bench` / `--profile` |
| `--prefill-length N` | - | `--bench`: prefill N ids and print the prefill row too |
| `--tg N` | `256` | ids to generate in `--bench` |
| `--prefill-chunk C` | `2048` | positions per prefill chunk, at most 4096 |
| `--profile`, `--steps N`, `--repeats R` | -, `32`, `1` | per-launch timing of the decode step |
| `--pipeline-split auto\|N` | `auto` | with `pp`: where the layers are cut |
| `--draft-vocab off\|32k\|64k\|128k`, `--draft-vocab-ids FILE` | `off` | MTP guesses from a smaller vocabulary (`b70-serve`) |
| `--spec lookup\|dflash-lookup` (`b70-serve`) | - | prompt lookup alone (verified through the MTP lists), or inside DFlash (what `dflash` already does) |

Environment variables (test switches, tool settings, the Hugging Face cache): [docs/environment-variables.md](docs/environment-variables.md).

## Docs

[docs/README.md](docs/README.md) is the index. Start with [docs/04-architecture.md](docs/04-architecture.md) for the
engine, [docs/17-running-models.md](docs/17-running-models.md) for running it, [docs/12-kernels.md](docs/12-kernels.md)
for the kernels and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for every number.

## Licence

AGPL-3.0-only (see [LICENSE](LICENSE)), with a commercial licence for companies that cannot work under the AGPL.
[LICENSING.md](LICENSING.md) explains which one applies to you.

Written by Jürgen Kozyrev.
