# b70-inference-server

**A fast LLM server for one or two Intel Arc Pro B70 cards, written from scratch.** Hand-crafted GPU kernels,
optimised for each model it runs - no PyTorch, no vLLM underneath - and an OpenAI-compatible API built for long coding-agent sessions (opencode and similar).

- **123 tokens per second** for one session of Qwen3.8-27B on two cards, at the model's full 262,144-token context.
- **141.7 tokens per second** for one session of Qwen3.8-27B on **one card** with the DFlash draft model
  (llama-benchy tg128); 96.2 t/s replaying a real coding-agent session.
- **258 tokens per second in total** with 8 requests running at once.
- **3,217 tokens per second** reading a prompt; the first token of a follow-up turn in as little as **0.86 s**
  (measured with the int8 prompt pass as it was before 2026-11-13; re-measurement pending, see below).
- **Ornith 1.5 35B-A3B:** prompts read at **8,877 t/s**, answers written at **180.5 t/s on one card, 351.9 t/s on two**.
- The full 262,144-token context of Qwen3.8-27B fits on **a single card**.
- **Faster than vLLM** on the same card and the same model files: up to **+56 %** writing and **+25 %** reading.

One choice to make - how many requests run at once. The engine sizes everything else.

## Speed

**Prefill** is the model reading your prompt, **decode** is the model writing its answer, both in tokens per second
(t/s). **MTP** - the model drafts several tokens ahead and checks them in one step; the answer does not change, it
arrives faster. A **prefix cache hit** is a follow-up turn whose earlier conversation is already cached.

### Qwen3.8-27B

One session on two cards (the default) with TP 2: **123.1 t/s**; first token in 1.84 s for a new conversation, 1.34 s
on a prefix cache hit.

The serving-bench requests contain 3,425 prompt tokens and generate 512 tokens.

Decode with MTP, t/s - one session, or several at once (the total, per request in brackets):

| setup | 1 request | 2 requests | 4 requests | 8 requests |
|---|---:|---:|---:|---:|
| two cards, tensor parallel | **123.1** | - | - | - |
| one card, DFlash2 | **141.7** (tg128) | n/d | n/d | n/d |
| one card, `--max-num-seqs 4` | 73.2 | 107.5 (59.5) | 126.6 (35.6) | |
| two cards, `--max-num-seqs 8` | 73.0 | 140.5 (73.8) | 209.1 (58.5) | **250.0** (36.0) |
| two cards, `--max-num-seqs 8 --kv-cache int8` | 74.0 | 163.5 (84.1) | 213.9 (58.8) | **258.5** (37.2) |
| two cards, `--max-num-seqs 8 --prefill-backend l0-int8` † | 82.0 | 138.5 (76.3) | 202.2 (60.6) | 250.1 (35.0) |

Decode with **DFlash** (`--spec dflash`): a small draft model,
[`urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound`](https://huggingface.co/urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound),
guesses up to 7 tokens and the model checks them in one step. One card, one request, t/s:

| workload | MTP | DFlash | |
|---|---:|---:|---:|
| llama-benchy tg128 | 72.5 | **141.7** | 1.95x |
| llama-benchy tg128 at 32,768 tokens of context | 57.2 | 74.7 | 1.31x |
| a recorded opencode session, replayed (78 requests, prefix cache on) | 65.9 | **96.2** | 1.46x |
| short code, prose and tool-call prompts | 74.5 | 100.8 | 1.35x |

Time to the first token, new conversation / prefix cache hit:

| setup | 1 request | 2 requests | 4 requests | 8 requests |
|---|---:|---:|---:|---:|
| two cards, tensor parallel (the default for one session) | **1.84 / 1.34 s** | | | |
| two cards, `--max-num-seqs 8` | 2.30 / 1.08 s | 2.36 / 1.10 s | 3.61 / 1.65 s | 7.27 / 2.79 s |
| two cards, `--max-num-seqs 8 --kv-cache int8` | 2.33 / 1.09 s | 2.37 / 1.11 s | 3.63 / 1.64 s | 7.30 / 2.81 s |
| two cards, `--max-num-seqs 8 --prefill-backend l0-int8` † | **1.80 / 0.86 s** | 1.88 / 0.89 s | 2.82 / 1.33 s | 5.76 / 2.24 s |
| one card, `--max-num-seqs 4` | 2.32 / 1.09 s | 3.52 / 1.65 s | 5.91 / 2.77 s | |

Prefill, t/s:

| setup | 4,096-token prompt | 16,384-token prompt |
|---|---:|---:|
| one card | 1,709 | 1,539 |
| one card, `--prefill-backend l0-int8` | 2,083 | 1,845 |
| two cards, TP | 2,939 | 2,686 |
| two cards, TP, `--prefill-backend l0-int8` | **3,217** | **2,937** |

Since 2026-11-13 `l0-int8` on Qwen3.8 runs int8 only in the MLP; the four attention / delta-net projections stay
bf16, because in int8 they changed a token on a 2,820-token prompt ([docs/17](docs/17-int8-prefill.md)). The one-card
and TP rows above are measured with it (interleaved against main: 0.92x / 0.97x the old form at 4,096 tokens). Rows
marked † were measured with the old form and are expected to come down; they wait for a quiet-box re-run.

**Against vLLM** - the same model files on one B70, one request, a 4,096-token prompt:

| t/s | this engine | vLLM | |
|---|---:|---:|---:|
| Ornith 1.5 35B-A3B, prefill | **8,877** | 7,723 | +15 % |
| Ornith 1.5 35B-A3B, decode with MTP | **180.5** | 115.8 | +56 % |
| Qwen3.8-27B, prefill (`--prefill-backend l0-int8`) † | **2,470** | 1,973 | +25 % |
| Qwen3.8-27B, decode with MTP | **65.1** | 45.2 | +44 % |
| Qwen3.8-27B, decode without MTP | **33.6** | 31.5 | +7 % |

### Ornith 1.5 35B-A3B

A mixture-of-experts model (Qwen3.5 35B A3B), with the full 262,144-token context on each card. One request runs on
one card; two requests run one per card and the number is the total.

| t/s | 4,096-token prompt, 1 request | 2 requests | 16,384-token prompt, 1 request | 2 requests |
|---|---:|---:|---:|---:|
| prefill | **8,877** | **15,223** | 7,135 | 12,116 |
| decode | 140.7 | 274.2 | 126.1 | 222.7 |
| decode with MTP | **180.5** | **351.9** | 157.0 | 260.2 |

How every number was taken: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Install

You need Linux with the `xe` driver, Intel's compute runtime with `ocloc` and the Level Zero loader, g++, CMake and
Rust (for the tokenizer). The fast single session on two cards needs peer-to-peer between them (native from kernel
7.4).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# the engine never downloads; fetch the model into the Hugging Face cache once
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
```

[The model](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ) is Qwen3.8-27B compressed to 4-bit
weights with AutoRound, in groups of 64 rather than the usual 128 - better accuracy for about 2 % of prefill speed.

## Run

```sh
# an OpenAI-compatible server on port 8000
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --prefix-split-last
```

You choose one thing: **how many requests run at once**, `--max-num-seqs N`. The engine sizes everything else
from it - context, cards, cache - and refuses at startup, naming why, anything that does not fit.

- **`N = 1` (the default):** one request with the largest context that fits. On two cards every layer is split over
  both (`--parallel tp`) - the fastest single session, at the model's full 262,144 tokens. On one card add
  `--kv-cache int8` to get the full context there too.
- **`N = 2` to `4` on one card, up to `8` on two:** shaped for a coding agent with subagents. The first slot (one
  per card on two cards) gets the largest context that fits - the main session; the other slots get a shorter
  context, enough for subagent tasks. On two cards every card runs its own copy of the model (`--parallel replica`).

MTP and the prefix cache are on by default. Ready-made recipes with their context sizes:
[docs/18-running-models.md](docs/18-running-models.md). Other models load too (Ornith 1.5, Qwen3.8-Flash-Next,
Kolibri-1, K2-Horizon, Nemotron 3.5 Bolt); Qwen3.8 is the one everything is tuned for.

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
| `--draft-model PATH` | Qwen3.8: `urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound` | the draft model for `--spec dflash` |
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
| `--pipeline-split auto\|N` | `auto` | with `pp`: where the layers are cut |
| `--draft-vocab off\|32k\|64k\|128k`, `--draft-vocab-ids FILE` | `off` | MTP guesses from a smaller vocabulary (also on `b70-serve`) |

Environment variables (test switches, tool settings, the Hugging Face cache): [docs/environment-variables.md](docs/environment-variables.md).

## Docs

[docs/README.md](docs/README.md) is the index. Start with [docs/04-architecture.md](docs/04-architecture.md) for the
engine, [docs/18-running-models.md](docs/18-running-models.md) for running it, [docs/12-kernels.md](docs/12-kernels.md)
for the kernels and [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for every number.

## Licence

Apache 2.0. See [LICENSE](LICENSE).

Written by Jürgen Kozyrev.
