# b70-inference-server

An LLM inference engine built from scratch for **one or two Intel Arc Pro B70** cards. It talks straight to Level
Zero with its own OpenCL C kernels - no PyTorch, no vLLM, no vendor kernel library on the hot path - and serves an
OpenAI-compatible API made for long agentic coding sessions (opencode and similar).

It is tuned today for **Qwen3.8-27B**: speculative decoding with the model's own MTP head, a prefix cache in system
RAM that turns every follow-up turn into a prefix cache hit, and the model's full 262,144-token context **on a single
card** (with the int8 KV cache). On two cards: tensor parallel for one fast session, or a model copy per card for up
to 8 requests at once.

## The model

[`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ):
Qwen3.8-27B quantised with AutoRound to int4 weights, one scale per 64 input channels, bf16 activations, GPTQ
packing; the output head and the MTP head stay bf16 in the file.

Why group size 64 rather than the usual 128: better accuracy, at the price of about 2 % prefill speed.

## Numbers

Qwen3.8-27B, MTP speculative decoding, each request a 2,820-token prompt and 512 generated tokens:

| setup | requests at once | **total t/s** | per request t/s | first token, cold | first token, prefix cache hit |
|---|---:|---:|---:|---:|---:|
| two cards, `--parallel tp` (one session, the two-card default) | 1 | **123.1** | 123.1 | 1.84 s | 1.34 s |
| two cards, `--max-num-seqs 8` (a model copy per card, 4 slots each) | 1 | 73.0 | 73.0 | 2.30 s | 1.08 s |
| | 2 | 140.5 | 73.8 | 2.36 s | 1.10 s |
| | 4 | 209.1 | 58.5 | 3.61 s | 1.65 s |
| | 8 | **250.0** | 36.0 | 7.27 s | 2.79 s |
| same, `--kv-cache int8` (slot 0 holds 167,936 tokens instead of 46,080) | 1 / 2 / 4 / 8 | 74.0 / 163.5 / 213.9 / **258.5** | 74.0 / 84.1 / 58.8 / 37.2 | 2.33 / 2.37 / 3.63 / 7.30 s | 1.09 / 1.11 / 1.64 / 2.81 s |
| same, `--prefill-backend l0-int8` | 1 / 2 / 4 / 8 | 82.0 / 138.5 / 202.2 / 250.1 | 82.0 / 76.3 / 60.6 / 35.0 | **1.80** / 1.88 / 2.82 / 5.76 s | **0.86** / 0.89 / 1.33 / 2.24 s |
| one card, `--max-num-seqs 4` | 1 / 2 / 4 | 73.2 / 107.5 / 126.6 | 73.2 / 59.5 / 35.6 | 2.32 / 3.52 / 5.91 s | 1.09 / 1.65 / 2.77 s |

**Prefill** (reading the prompt), t/s:

| | 4,096-token prompt | 16,384-token prompt |
|---|---:|---:|
| one card, bf16 prompt pass | 1,709 | 1,539 |
| one card, int8 prompt pass (`--prefill-backend l0-int8`) | 2,205 | 1,940 |
| two cards, `--parallel tp`, bf16 prompt pass | 2,939 | 2,686 |
| two cards, `--parallel tp`, int8 prompt pass | **3,323** | **3,048** |

**Ornith 1.5 35B-A3B** (Qwen3.5 35B A3B MoE), two cards with `--parallel replica --max-num-seqs 2` (the full 262,144
context per card, bf16 cache); two requests are the total over both cards:

| | 4,096-token prompt, 1 request | 2 requests | 16,384-token prompt, 1 request | 2 requests |
|---|---:|---:|---:|---:|
| prefill | **8,877 t/s** | 15,223 t/s | 7,135 t/s | 12,116 t/s |
| decode, without MTP | 140.7 t/s | 274.2 t/s | 126.1 t/s | 222.7 t/s |
| decode, with MTP | **180.5 t/s** | **351.9 t/s** | 157.0 t/s | 260.2 t/s |

## Install

You need Linux with the `xe` driver, Intel's compute runtime with `ocloc` and the Level Zero loader, g++, CMake and
Rust (for the tokenizer). Tensor parallel on two cards needs peer-to-peer between them (native from kernel 7.4).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# the engine never downloads; fetch the checkpoint into the Hugging Face cache once
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
```

## Run

```sh
# an OpenAI-compatible server on port 8000
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --prefix-split-last
```

You choose one thing: **how many requests run at once**, `--max-num-seqs N`. The engine sizes everything else
from it - context, cards, cache - and refuses at startup, naming why, anything that does not fit.

- **`N = 1` (the default):** one request with the largest context that fits. On two cards it runs tensor parallel,
  the fastest single session, at the model's full 262,144 tokens. On one card add `--kv-cache int8` to get the full
  context there too.
- **`N = 2` to `4` on one card, up to `8` on two:** shaped for a coding agent with subagents. The first slot (one
  per card on two cards) is guaranteed the largest context that fits - the main session; the other slots get a
  shorter context, enough for subagent tasks. On two cards every card runs its own copy of the model.

MTP speculative decoding and the prefix cache are on by default. Recipes with their contexts and speeds:
[docs/18-running-models.md](docs/18-running-models.md). Other models load too (Ornith 1.5, Qwen3.8-Flash-Next,
Kolibri-1, K2-Horizon, Nemotron 3.5 Bolt); only Qwen3.8 is tuned.

Two cards run in one of three modes (`--parallel`):

| mode | for | Qwen3.8 |
|---|---|---|
| `tp` - every layer split over both cards (needs peer-to-peer) | one session at maximum speed | the default: 123 t/s, full context |
| `replica` - a model copy per card | several sessions at once (`--max-num-seqs 2` to `8`) | up to 258 t/s total |
| `pp` - the layers split between the cards | models that do not fit one card | not needed: the full context fits one card |

## Flags

`b70-serve`:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | a Hugging Face repo id in the local cache, or a snapshot directory; never downloads |
| `--max-num-seqs N` | `1` | how many requests run at once; the context follows. Slot 0 gets the largest context that fits beside N - 1 slots of 32,768 tokens. Up to 4 per card; on two cards N > 1 runs a model copy per card |
| `--max-len auto\|L` | `auto` | context length (slot 0's with `--max-num-seqs`). `auto`: the largest that fits, up to the model's 262,144 |
| `--parallel auto\|1\|tp\|replica\|pp` | `auto` | `tp`: every layer split over both cards (one request; needs peer-to-peer between them). `replica`: a model copy per card. `pp`: the layers split between the cards, for models that do not fit one card. `1`: one card |
| `--spec auto\|off\|mtp\|lookup\|dflash\|dflash-lookup` | `auto` | speculative decoding: a proposer guesses tokens, the model checks them all in one pass; the output never changes. `auto` = the MTP head on Qwen-family models |
| `--spec-max K` | the most | the proposer's cap on guesses per step |
| `--draft-model PATH` | the model's default | the draft model for `--spec dflash` |
| `--kv-cache bf16\|int8` | `bf16` | how the context is stored; `int8` halves it (more context, more requests) |
| `--prefill-backend l0\|l0-int8` | `l0` | the prompt pass: bf16, or int8 on the matrix engines (faster; every mode) |
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
| `--layers N\|auto` | - | load only the first N layers (Kolibri-1, Nemotron 3.5 Bolt, Qwen3.8-Flash-Next) |

Developer flags (`b70-decode`):

| flag | default | what it does |
|---|---|---|
| `--bench` | - | time `--tg` generated ids after `--depth` ids of context (or a `--prefill-length N` prompt); prints a markdown row |
| `--depth D` | `4096` | context before decoding, for `--bench` / `--profile` |
| `--prefill-length N` | - | `--bench`: prefill N ids and print the prefill row too |
| `--tg N` | `256` | ids to generate in `--bench` |
| `--prefill-chunk C` | `2048` | positions per prefill chunk, at most 4096 |
| `--profile`, `--steps N`, `--repeats R` | -, `32`, `1` | per-launch timing of the decode step |
| `--pipeline-split auto\|N`, `--pipeline-handoff copy\|peer` | `auto`, `copy` | with `pp`: where the layers are cut, and how the hidden state crosses |
| `--draft-vocab off\|32k\|64k\|128k`, `--draft-vocab-ids FILE` | `off` | MTP guesses from a smaller vocabulary (also on `b70-serve`) |

Environment variables (test switches, tool settings, the Hugging Face cache): [docs/environment-variables.md](docs/environment-variables.md).

## Docs

[docs/README.md](docs/README.md) is the index. Start with [docs/04-architecture.md](docs/04-architecture.md) for the
engine, [docs/18-running-models.md](docs/18-running-models.md) for running it, [docs/12-kernels.md](docs/12-kernels.md)
for the kernels and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for every number.

## Licence

Apache 2.0. See [LICENSE](LICENSE).

Written by Jürgen Kozyrev.
