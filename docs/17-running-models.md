# Running the models

This is the practical guide to running the engine. For each model it gives the commands, the context length you get
and how fast it runs. Read it if you want to serve a model; read [04-architecture.md](04-architecture.md) if you want
to know how the engine works.

Qwen3.8-27B comes first, because everything is tuned for it. Ornith 1.5, Qwen3.8-Flash-Next and Kolibri-1 follow.
Every flag is listed in the [README](../README.md#flags), and environment variables in
[environment-variables.md](environment-variables.md).

## How to read the numbers

- Prefill (reading the prompt) speeds are the prompt pass alone, on a 4,096-token prompt unless a row says otherwise.
- Decode (writing the answer) speeds use MTP speculative decoding with prompt lookup on, which are the defaults. They
  are measured on the golden code, prose and tool-call prompts, 256 generated tokens each, and given as
  `code / prose / tool calls`.
- With several requests at once, the figure is their total.
- Context lengths are what the memory planner gives per card with the MTP head loaded, at the default 1.5 GB reserve
  on the 32.53 GB card.

The headline numbers are the same as in the [README](../README.md#speed).

## Before you start

Build as in the README (`cmake`, then `ctest`). The engine never downloads anything, so fetch each checkpoint into the
Hugging Face cache once:

```sh
uvx --from huggingface_hub hf download <org>/<name>
```

There are two command-line tools:

| tool | what it does |
|---|---|
| `b70-serve` | the OpenAI-compatible server |
| `b70-decode` | greedy decoding from token ids, benchmarks and profiles |

Both take either a repo id, looked up under `$HF_HOME` (default `~/.cache/huggingface`), or a snapshot directory.

### Hardware

The engine runs on one or two Intel Arc Pro B70 cards (PCI id 0xE223). Before reading any weight, a hardware check
confirms the vendor, the device id, the Xe-core count and, with two cards, peer access. Peer access means one card can
write straight into the other's memory; tensor parallel depends on it. The Arc Pro B65 and B60 are refused as not yet
tested, and so are more than two cards.

One card is device 0, and two cards are devices 0 and 1 of what `ZE_AFFINITY_MASK` exposes. So
`ZE_AFFINITY_MASK=1` runs on the second card alone.

### One process, one model

One server process serves one model. The prefix cache belongs to that process. It keeps the state of earlier prompts
in host memory, so a conversation's next turn does not reprocess its whole history.

### Connecting a client

Point opencode, or any other OpenAI-compatible client, at `http://<host>:8000/v1`. The server lists the model under its
repo id (or the snapshot directory's name) and answers to any model name in a request.

### `--prefix-split-last` for agentic sessions

Chat clients that drop the model's reasoning from the history (opencode does) change the prompt right after the last
turn. `--prefix-split-last` makes each next turn a prefix cache hit exactly where it diverges. Use it for agentic
sessions and leave it off for benchmarks. Every agentic recipe below includes it.

## Qwen3.8-27B

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
```

### One card

| recipe | context | speed |
|---|---:|---|
| default, bf16 KV cache | 171,008 | 138 / 57 / 100 t/s, prefill 1,788 t/s |
| full context: int8 KV cache | 262,144 | not measured |
| faster prefill (opt-in): int8 prompt pass, bf16 cache | 171,008 | prefill 2,337 t/s |
| four requests at once: `--max-num-seqs 4` | 46,080 + 3 x 32,768 (bf16 cache), 167,936 + 3 x 32,768 (int8 cache) | not measured |

```sh
./build/src/cli/b70-serve $Q --parallel 1 --prefix-split-last                                 # default
./build/src/cli/b70-serve $Q --parallel 1 --kv-cache int8 --prefix-split-last                 # full context
./build/src/cli/b70-serve $Q --parallel 1 --prefill-backend l0-int8 --prefix-split-last       # faster prefill (opt-in)
./build/src/cli/b70-serve $Q --parallel 1 --max-num-seqs 4 --kv-cache int8 --prefix-split-last # 4 requests
```

On a machine with one card, `--parallel 1` can be left out.

Do not combine the int8 prompt pass with the int8 KV cache. Together they round too much, and about 3 % of tool calls
that copy exact text from a long prompt come out wrong.

### Two cards

| recipe | mode | context | speed |
|---|---|---:|---|
| one session, fastest (the default) | `tp`: every layer split over both cards; needs peer access | 262,144 (bf16 cache) | 123 / 74 / 115 t/s; prefill 2,931 t/s, 3,217 with `--prefill-backend l0-int8` |
| two sessions at full context: `--max-num-seqs 2` | `replica`: a model copy per card, one request each | 171,008 (bf16) or 262,144 (int8) each | 164 / 108 / 144 t/s total |
| most requests at once: `--max-num-seqs 8` | `replica`, 4 slots per card | 46,080 + 3 x 32,768 (bf16) or 167,936 + 3 x 32,768 (int8) per card | 342 / 300 / 233 t/s total at 8 requests, 282 / 189 / 206 at 4 |

```sh
./build/src/cli/b70-serve $Q --prefix-split-last                                    # one session, tp
./build/src/cli/b70-serve $Q --max-num-seqs 2 --kv-cache int8 --prefix-split-last   # two sessions, full context each
./build/src/cli/b70-serve $Q --max-num-seqs 8 --kv-cache int8 --prefix-split-last   # up to 8 requests
```

How the slots work with `--max-num-seqs N`:

- Slot 0 gets the largest context that fits beside N - 1 slots of 32,768 positions. Slot 0 is meant for the main
  session, the others for shorter subagent tasks.
- On two cards the slots are split between the cards, card 0 taking the odd one.
- An explicit `--max-len` that does not fit beside the other slots is refused at startup, with the numbers.
- `tp` serves one request at a time.

### Two cards, pipeline parallel

Pipeline parallel (`--parallel pp`) splits the layers between the cards: the first card runs the first half, the
second card the rest. It is meant for models that do not fit one card. Qwen3.8 fits one card even at full context, so
it gains nothing here, but it runs. Each card holds about half the model, so `--max-len auto` reaches the trained
262,144 tokens, MTP included. One request is no faster than on one card.

The hidden state crosses from the first card to the second by a device-to-device copy. The MTP head runs on the second
card with its own copy of the embedding (2.54 GB). `--parallel auto` picks `pp` by itself when the model and the
requested `--max-len` do not fit one card and `tp` does not apply.

```sh
# an agentic session over both cards as pipeline parallel: 262,144 tokens with MTP and the int8 cache
./build/src/cli/b70-serve $Q --kv-cache int8 --parallel pp --prefix-split-last
# b70-decode: the same ids as on one card, greedy, with MTP across the split
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --parallel pp --spec mtp
```

At startup it prints the split (`split: auto -> N`), the plan for each card and both cards' memory lines, plus
`parallel: auto -> ...` when `auto` made the choice.

### The DFlash draft model

DFlash is a small separate draft model. Each step it drafts a block of up to 7 tokens from Qwen3.8's own hidden states
(the outputs of layers 5, 19, 33, 47 and 61), and the main model then checks them. It runs only when asked for,
because its weights take memory the context would otherwise use.

`--spec dflash` needs an explicit `--draft-model`. Without one, `b70-serve` and `b70-decode` refuse and name the two
choices:

| `--draft-model` | size | notes |
|---|---:|---|
| `urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound` | 1.21 GB | our int4 export, loaded as shipped; the recommended one. Draft pass 4.41 ms at a 4,096-token context |
| `z-lab/Qwen3.8-27B-DFlash2` | 2.05 GB | the bf16 original, quantised to int8 at load. Draft pass 6.20 ms |

The int4 drafter accepts slightly fewer tokens per round than the int8 one (0.016 fewer, pooled over the test
prompts) and is faster overall.

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound   # once
D=urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound

# one card, with prompt lookup inside
./build/src/cli/b70-serve $Q --parallel 1 --prefix-split-last --spec dflash --draft-model $D
# one card with the int8 KV cache
./build/src/cli/b70-serve $Q --parallel 1 --spec dflash --kv-cache int8 --draft-model $D
# one card, up to four requests at once (int8 KV cache: 135,168 + 3 x 32,768 positions)
./build/src/cli/b70-serve $Q --parallel 1 --max-num-seqs 4 --kv-cache int8 --spec dflash --draft-model $D
# two cards, two sessions: a drafter on each card
./build/src/cli/b70-serve $Q --max-num-seqs 2 --spec dflash --draft-model $D
# two cards, tensor parallel: the drafter split over both cards
./build/src/cli/b70-serve $Q --parallel tp --spec dflash --draft-model $D
# two cards, the layers split: the drafter on the second card
./build/src/cli/b70-serve $Q --parallel pp --spec dflash --draft-model $D
# greedy without the server (one card)
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/code.ids --n 256 --prefill \
    --spec dflash --spec-max 7 --draft-model $D
```

| setup | code / prose / tool calls, t/s |
|---|---|
| one card, one request | 164 / 65 / 111 |
| two cards `replica`, two requests (total) | 317 / 127 / 191 |

DFlash is fastest on one card. Under `--parallel tp` it works, but one session is no faster than on one card, and the
server prints a note at startup recommending one card, `replica`, or MTP on two cards.

Context with the drafter loaded:

| drafter | bf16 KV cache | int8 KV cache |
|---|---:|---:|
| int4 (recommended) | 158,208 | 262,144 |
| int8 | 145,664 | 254,976 |

Several requests on a card (`--max-num-seqs`, up to 4 per card):

- Each request keeps its own drafter context.
- A request alone drafts up to 7 tokens, as with one slot. Several share one draft pass of 8 rows: 3 tokens each for
  two requests, 1 each for four.
- Against `--spec mtp` in the same mode (one card, int8 KV cache) it is 1.15x with one request, 1.12x with two and
  0.97x with four.
- Slot 0 gets 135,168 positions beside three of 32,768; the drafter takes the difference.
- Refused, by name: `--spec dflash-lookup` and `--draft-vocab` with several requests on a card, more than four
  requests per card, and every model other than Qwen3.8.

The prefix cache keeps the drafter's context (41.9 MB) with every snapshot. A conversation that gets a prefix cache
hit at the end of a prompt or of a request therefore drafts exactly as an unbroken one would. A hit at a snapshot
inside a long prompt drafts worse, never differently, until 2,048 new positions have refilled the drafter's window.

How the drafter's acceptance was measured, and how to repeat it: [tools/oracle/README.md](../tools/oracle/README.md),
section "The DFlash P0".

### Benchmarks and greedy runs

```sh
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256           # prefill 4096, then decode 256
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 32 --parallel tp
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --spec mtp > out.ids
./build/src/cli/b70-decode $Q --profile --depth 4096 --steps 32 --repeats 5    # per-launch decode timing
```

`b70-decode` defaults to one card, `--max-len 16384` and no speculative decoding, so its rows stay comparable over
time. [15-step-anatomy.md](15-step-anatomy.md) explains how to read `--profile`.

To benchmark a running server with llama-benchy, start it without `--prefix-split-last` (that flag is for sessions):

```sh
uvx llama-benchy --base-url http://<host>:8000/v1 --model <repo id> \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

Its `tg` and `pp` figures read the same way as against vLLM, because the server sends the role frame with the first
token, after prefill, as vLLM does. Its `pp` includes HTTP; `b70-decode --bench --prefill-length` gives the
device-side prefill time.

`b70-decode --ids` takes whitespace-separated token ids made with the checkpoint's own tokenizer. To make them, and to
turn output ids back into text (needs `transformers`; `<snapshot>` is the snapshot directory in the cache):

```sh
python3 tools/oracle/tokenize.py <snapshot> encode prompt.txt > prompt.ids
python3 tools/oracle/tokenize.py <snapshot> decode $(cat out.ids)
```

`tests/golden/prompts/{prose,code,cjk}.ids` are ready-made ids for Qwen3.8's tokenizer.

## Speculative decoding

In speculative decoding a proposer guesses the next few tokens, and the main model checks them all in one pass (a
verify step of several rows). Every correct guess is a token almost for free, and the output does not change. `--spec`
picks the proposer:

| `--spec` | what proposes | where |
|---|---|---|
| `auto` (the default) | the model's own MTP head where it has one (Qwen3.8, Ornith, Flash-Next); nothing elsewhere | every run mode |
| `mtp` | the MTP head, up to 3 tokens a step | every run mode |
| `dflash` | DFlash, a small draft model, up to 7 tokens a step (Qwen3.8; Kolibri-1 with its DSpark drafter) | one card, `replica`, `tp`, `pp`; needs `--draft-model` |
| `off` | nothing | |

MTP (multi-token prediction) is a small head built into the model that predicts the following tokens. `auto` uses it
because it costs no extra memory, so the full context stays available. The adaptive form picks, per request, how many
tokens to draft from that request's recent acceptance; `--spec-max` caps it.

### Prompt lookup

Prompt lookup copies repeats of earlier text. Where the answer repeats 8 or more tokens of earlier text, the tokens
that followed that text last time are checked instead of the proposer's guesses:

- up to 3 tokens inside MTP, up to 7 inside DFlash (where the repeat must lie outside the drafter's own 2,048-token
  window);
- it runs inside both proposers on one card with one request at a time, and inside MTP on each card of
  `--parallel replica` with one request per card; elsewhere the proposer drafts as usual.

On one card inside MTP, a long copied run is checked in lists of 16, 32 or 64 rows, chosen as the copy goes. Pure
copying runs at 636-650 t/s, against 356-364 with 16 rows only. Prompts that edit or copy code run 1.173x the 16-row
speed, with identical output ids.

Lookup is on by default. `B70_SPEC_LOOKUP=0` serves plain MTP or plain DFlash for comparisons. `--spec lookup` and
`--spec dflash-lookup` remain as developer values.

### How speculative output is judged

The verify rows run other kernels than a plain step: the small-batch GEMM `gemm_sm` instead of the one-row GEMV, and
a DPAS (the card's matrix-multiply instruction) attention kernel at depth. These round the last bf16 digit
differently, so speculative output is not compared bit for bit with plain decoding.

Instead the f32 judge grades it (`tests/server/spec_judge.h`, see [14-golden-gate.md](14-golden-gate.md)). A
speculative stream passes when it is no further from a float32 reference than plain decoding on the same context, or
when every miss is a near-miss, within 8 bf16 ulps of the reference's top choice. Plain code paths are still checked
bitwise.

## Which models, and what each supports

- Qwen3.8-27B is the model in focus, with every feature and every measured row: prefill, decode, depth, prefix
  cache, MTP, DFlash, tensor parallel.
- Ornith 1.5 35B-A3B is supported, with measured prefill and decode rows.
- Qwen3.8-Flash-Next is served whole on two cards from our own export, with its routed experts partly in host memory
  (the expert tier). Pipeline parallel is the default; tensor parallel is faster at decode and slower at prefill.
- Kolibri-1 is supported by the code but not tuned, and has no speed rows.
- Muse-Glimmer and Agnes 3.0 Flash are no longer supported. A `config.json` naming `AgnesForConditionalGeneration` is
  refused by name.

A passing box test shows a configuration runs and is correct; it is not a speed row. The full box validation
(`tools/box_validate.sh`: golden sets, serving, prefix cache, speculative decoding) runs on `main` on demand.

| | Qwen3.8-27B | Ornith 1.5 35B-A3B | Kolibri-1 | Qwen3.8-Flash-Next |
|---|---|---|---|---|
| checkpoint | published (ours, int4 g64) | published (ours) | published (ours, int4 g64) | ours (int4 g64, not published) or Intel's interim W4A16 |
| `b70-serve` | yes | yes | yes, two cards by default | the whole model on two cards (`--parallel auto` = pp), experts partly in pinned host memory |
| `--parallel tp` | yes | refused | refused | yes, when asked: decode with MTP 0.99-1.35x pp's on text; prefill 0.69-0.81x pp's (32k / 128k-token prompt) |
| `replica`, `--max-num-seqs` | yes | `replica` at `--max-num-seqs 2` (one request per card); more is refused | refused | refused |
| `--parallel pp` (two cards) | yes | yes | its default | its default for the whole model |
| `b70-decode` | yes | yes | decode and prefill, two cards by default | decode and prefill, the whole model or `--layers N` |
| MTP head (`--spec mtp`; `--spec auto` = adaptive MTP) | yes | yes (an MoE head) | none (`auto` = off) | `b70-serve`: adaptive MTP by default, `--spec-max` 1 to 3; `b70-decode` decodes plain |
| `--spec lookup` (developer value) | yes | yes | no | yes |
| `--spec dflash` | yes: one card (with prompt lookup inside), up to 4 requests per card, `replica`, `tp` or `pp`, bf16 or int8 KV | no | `b70-serve`: the DSpark drafter (below); `b70-decode` refuses | refused |
| `--kv-cache int8` | yes | not built | refused | refused |
| prefill backends | `l0` (default), `l0-int8` | `l0` (default; use it); `l0-int8` opt-in and slower | `l0` only | `l0` only |
| KV cache per position (bf16) | 64 KiB | 20 KiB | 20 KiB (+ 335.5 MB of sliding-window rings) | 24.75 KiB (12 QSA layers: K + V 2 KiB + 64 B of indexer keys each) |
| `--max-len auto`, one card, int8 head | 171,008 with MTP (the default); about 201k with `--spec off` | 262,144, with or without MTP (27.3 GB planned of 32.5, bf16 KV) | 262,144 on two cards (split at layer 25) | not served whole on one card; see below |
| with `--kv-cache int8` | 262,144, with or without MTP | - | - | - |

## Ornith 1.5 35B-A3B

Ornith is a mixture-of-experts (MoE) model: each layer has many small expert MLPs, and a router picks a few of them
per token. It has:

- 40 layers: 30 GDN (gated delta-net, a linear-attention layer with a fixed-size state) and 10 full-attention layers;
- hidden size 2048;
- 256 experts with top-8 routing plus a shared expert, so about 3 B parameters are active per token.

Decode, prefill and serving are built, with its own chat template (Qwen3.5's), Qwen XML tool calls and its MoE MTP
head.

### The checkpoint

`ornith-ai/Ornith-1.5-35B-A3B` is published in bf16 only (71.9 GB), which the engine refuses: it loads int4 GPTQ. Use
our AutoRound W4A16 export,
[`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`](https://huggingface.co/urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)
(22.99 GB). It is int4 g64 symmetric with GPTQ packing. The experts are stored one by one as
`experts.E.{gate,up,down}_proj`, and the routers and the MTP head stay in bf16. Unlike Qwen3.8's export, `in_proj_a`
and `in_proj_b` are int4 too; the engine loads them as one int4 `a‖b` (see [13-loader.md](13-loader.md)).

### Speed and memory

| | one card | two cards `replica`, 2 requests (total) |
|---|---:|---:|
| prefill, 4k prompt | 11,036 t/s | 19,117 t/s |
| prefill, 16k prompt | 8,800 t/s | 13,728 t/s |
| decode with MTP (code / prose / tool calls) | 312 / 326 / 264 t/s | 581 / 605 / 453 t/s |

Ornith reads about 1.86 GB of weights per token, far less than Qwen3.8. So on decode the 526 kernel launches per
token, not the bytes, set the rate. Weights are 19.45 GB with the int8 head, plus 0.50 GB for the MTP head.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
uvx --from huggingface_hub hf download $O                 # 22.99 GB

# agentic session: the full trained context, 262,144, fits in bf16 with or without MTP
./build/src/cli/b70-serve $O --prefix-split-last
# two sessions at once on two cards (replica: a model copy per card, one request each, full context each)
./build/src/cli/b70-serve $O --max-num-seqs 2 --prefix-split-last

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $O --ids prompt.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $O --profile --depth 4096
```

### Two cards

Two cards serve two requests at once as `replica` (`--max-num-seqs 2`, or `--parallel replica`). Each card runs its
own engine with the full context, MTP and the prefix cache. A prompt cached by one card is a prefix cache hit on the
other. Each request is bitwise what it would be alone on its card.

Several requests on one card (sequence slots) are built for Qwen3.8 only, so `--max-num-seqs` above 2, or 2 on one
card, is refused. `--parallel tp` is Qwen3.8's only.

### Options that differ from Qwen3.8

- `--kv-cache int8` is not built for Ornith's attention shape. It is not needed for context, since 262,144 fits in
  bf16. `B70_KV_CACHE` must stay `bf16`.
- Use the default prompt pass (`l0`). `l0-int8` is opt-in and slower than the bf16 pass on Ornith. It also loses
  routing accuracy against f32: on the box, 1314 differing expert sets on the prefill path against the bf16
  reference's 1140, and no 8-bit activation scheme studied won that back. Its model-level accuracy checks are
  therefore printed, not gated; its kernels' own gate (`prefill_gate_ornith_l0-int8`, judged by the
  quantisation-aware f32 reference) still applies.
- The adaptive MTP (`--spec auto`) decides with Qwen3.8's cost table, drafting up to 3 tokens. Ornith's own measured
  table (a verify of 2, 3 or 4 rows costs 1.15, 1.34 or 1.52 plain steps, a draft 0.19 each) decoded 6-10 % slower in
  a direct comparison, so it is not used; the startup line says so.
- The head's bf16 experts are quantised to int4 at load (round to nearest). That can change acceptance, never output.

### Tokenizer

The checkpoint's `tokenizer.json` defines Qwen3.8's 248,077 ids, the count the greedy argmax masks to, so `b70-serve`
prints no vocabulary note. It tokenises the committed `tests/golden/prompts` (prose, code, cjk) to the committed ids.
Its pre-tokenizer splits combining marks differently from the bf16 base checkpoint's file (`\p{M}`), which only
matters for text with combining marks. `B70_PREFILL_PROFILE` works as for Qwen3.8.

## Qwen3.8-Flash-Next

Qwen's `Qwen3.8-Flash-Next` (`model_type qwen4_exp`) is a large MoE model:

- 48 layers: 36 gated delta-net layers with a sigmoid output gate, and 12 QSA (sparse-attention) layers that attend to
  the top 512 blocks of 4 positions plus the open block;
- a 512-expert, top-10 MoE in every layer;
- 4-stream hyper-connections in place of a single residual stream;
- a per-layer n-gram embedding (PLE) whose 51.8 GB int8 table lives in host memory and is read by the card directly.

Two checkpoints load. Our own g64 export (`tools/quantize/quantize_qwen4exp.sh`, not published) is the one served
whole. Intel's interim `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (int4 g128 experts, bf16 dense layers) loads
through the exact g64 expansion ([13-loader.md](13-loader.md)) and runs truncated (`--layers N`).

### Preparing a snapshot

Two files have to sit beside the weights before the engine will load Flash-Next.

1. The PLE table, converted to int8 once. That writes about 51.8 GB next to the snapshot, so check free disk space
   first. The engine looks for `<snapshot>-ple-int8/` and, if it is missing, refuses with this command:

   ```sh
   python3 tools/quantize/qwen4exp/ple_int8.py <snapshot> <snapshot>-ple-int8/
   ```

   `tools/box_validate/qwen4exp_oracle.sh <data> ple-int8` wraps the same converter for Intel's snapshot.

2. The original model's tokenizer: `Qwen/Qwen3.8-Flash-Next`'s `tokenizer.json`, whose pre-tokenizer split adds
   `\p{M}`. It is read from the snapshot directory. Our own export already ships it. Intel's checkpoint ships
   Qwen3.8's file instead, which `b70-serve` refuses, so build a directory that holds every file of Intel's snapshot
   except `tokenizer.json`, and take that from the original:

   ```sh
   I_SNAP=<Intel's snapshot directory>
   F=~/models/qwen3.8-flash-next
   mkdir -p $F
   for f in $I_SNAP/*; do [ "$(basename $f)" = tokenizer.json ] || ln -s "$f" $F/; done
   uvx --from huggingface_hub hf download Qwen/Qwen3.8-Flash-Next tokenizer.json --local-dir $F
   ln -s $I_SNAP-ple-int8 $F-ple-int8                      # the PLE file beside the new directory
   ```

The stop ids come from the snapshot's `generation_config.json` (`<|im_end|>` and `<|endoftext|>`).

### Speed

Our export, two cards, at about 1-4k context:

| | `pp` | `tp` |
|---|---:|---:|
| prefill, 32k-token prompt | 3,103 t/s | 2,128 t/s |
| prefill, 128k-token prompt | 2,458 t/s | 1,988 t/s |
| first token, 43-token prompt | 0.38 s | 0.50 s |
| decode, MTP off (code / prose / tool calls) | 58 / 49 / 49-60 t/s | 63 / 51 / 55-62 t/s |
| decode, MTP (code / prose / tool calls) | 81 / 56 / 80-105 t/s | 82-92 / 60-65 / 89-99 t/s |

`--max-len auto` gives 65,536. Passkey retrieval is 3 of 3 at 61,955 ids, and a prefix cache hit's output is identical
to the cold run.

### Memory: the expert tier

The whole model does not fit two B70s: its routed experts alone are 64.17 GB at int4 g64. So when the model is served
whole (no `--layers`):

- Each card keeps the most-routed experts its VRAM has room for after the rest of the plan, the same count in every
  layer.
- The kernels read the rest in place from pinned host memory: one range per layer, about 12.4 GB beside the 51.8 GB
  PLE table. A miss costs its bytes at about 11.2 GB/s on top of the VRAM time.
- Which experts are most routed comes from a routing profile, `<snapshot>-expert-profile.safetensors`, made by
  `tools/oracle/offload_curve.py profile`. Without one, experts are kept by id. Our profile counts 42 routing traces:
  the 36 tool-call conversations, an agentic transcript and 7.5 % prose (repository docs, a novel, the golden prose
  prompts). It gives a derived 81 % of expert reads served from VRAM on two cards at 32,768.
- `--max-len auto` takes the longest of 32,768, 65,536, 131,072 and 262,144 whose projected decode speed stays within
  5 % of 32,768's.
- The placement is static; experts are not swapped while serving.

Loading takes about 10 minutes and pins 63.2 GB of host memory. With the prefix cache the server holds about 104 GB,
so the model needs at least 128 GB of system RAM. Run nothing else memory-hungry beside it: when RAM runs short the
kernel can move the server's host memory to swap, and the experts read from it then come back from disk.

### Truncated runs: `--layers`

`--layers N` loads the first N layers, then the final mixer and the head. It is a development mode. `--layers auto`
takes the planner's N: 18-19 of Intel's layers on one card, 38-39 on two, depending on `--max-len`.

The prefill scratch is about 1.5 GB per card (the 512 MiB expert weight batch and score rows that grow with
`--max-len`). It is planned into `--layers auto` and `--max-len auto` when a run prefills; with it, Intel's forms fit
18 layers on one card and 37 on two at 32,768.

Launch counts (derived): 779 per decode token at 48 layers, 75 at `--layers 4`. Prefill runs the prompt in chunks of
up to 2048 (`--prefill-chunk C`) on the `l0` backend: flash attention for positions that see at most 2051 others,
then the indexer's own block selection and a sparse flash attention past them. That is 358, 361 or 360 launches per
chunk at `--layers 4` (all dense, straddling position 2050, all sparse) and about 3,950 at 48 layers.

### Serving

`b70-serve <checkpoint>` serves the whole model on two cards, or with `--layers N` a truncated one on one card or two,
as OpenAI chat through the Qwen chat path. The template is Qwen3.8's (the original's newer revision adds
`{% generation %}` markers, which render nothing). Tool calls are XML and reasoning is `<think>`, as for Qwen3.8.
Requests are greedy unless they ask for sampling.

The prefix cache keeps this family's state with each snapshot: 115.7 MB for the 36 GDN states and convolution rows,
the PLE history and the QSA open block's raw keys, plus 21 KB with the head. Each position takes 25,344 B in its blocks
(plus 2,112 with the head), in one layout on one card or two.

MTP in `b70-serve`:

- `--spec auto` (the default) and `--spec mtp --spec-max K` (K from 1 to 3) draft with the MTP head and verify K + 1
  rows in one list (790-795 launches at 48 layers, 86-91 at `--layers 4`, derived).
- The head uses vLLM's form, with `pre_fc_norm_hidden` as one RMSNorm over the 4 streams
  (`B70_Q4_MTP_NORM=per_stream` selects the other form). Draft steps after the first reuse step 0's sparse-attention
  selection. Its bf16 experts are quantised to int4 g64 at load.
- On two cards it runs on the second and reads the first card's embedding over peer access.
- Each request picks its drafts one at a time. Every draft step ends with the head's probability for its token, and a
  greedy request keeps drafting, up to 3, only while that probability makes the next draft pay over the model's cost
  table (under `tp` each card's half of the vocabulary is summed on the host). Sampled requests pick their number of
  drafts, 0 to 3, from their own acceptance. `B70_MTP_ADAPTIVE=0` keeps K at `--spec-max`, for A/B runs.
- In a verify of several rows each distinct expert is read once and applied to every row that chose it.

### `pp` or `tp`

Two cards split the model one of two ways:

- `--parallel pp` (what `--parallel auto` picks for the whole model on two cards; one card when only one is visible)
  gives each card half of the layers: 24 each, with the MTP head and the lm_head on the second.
- `--parallel tp` gives each card half of every layer: half the attention and gated delta-net heads, half of every
  expert (each card keeps its own experts' halves resident and reads the rest from host memory), and half the
  lm_head's vocabulary. The MTP head is whole on the second card. The two halves are summed after each mixer and each
  MoE block by peer writes.

Decode with MTP is as fast or faster under `tp` (0.99-1.35x `pp`'s on text). Prefill is slower: 0.69x `pp`'s for a
32k-token prompt and 0.81x for 128k. Both cards still run their own hyper-connections and part of the indexer's block
scoring (the scoring itself is split across the cards). `tp` needs `--lm-head int8` in `b70-decode` (`b70-serve`
always uses it) and `--layers N` rather than `--layers auto`, and has no `--pipeline-split`.

```sh
# every command below needs the PLE int8 file beside the snapshot (<snapshot>-ple-int8/)
O=<our export>   # the whole model; experts that do not fit VRAM in pinned host memory (63.2 GB pinned with the PLE)
./build/src/cli/b70-serve $O                     # two cards: pp
./build/src/cli/b70-serve $O --parallel tp        # faster decode, slower prefill
./build/src/cli/b70-decode $O --parallel pp --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $O --parallel tp --lm-head int8 --bench --prefill-length 4096 --tg 32

I=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_short.ids --n 32
./build/src/cli/b70-decode $I --layers 38 --parallel pp --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $I --layers auto --max-len auto --ids tests/golden/prompts/q4exp_short.ids --n 8
# the prompt prefilled in chunks (l0 only), then decode; and the prefill bench row
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_4k.ids --n 32 --prefill
./build/src/cli/b70-decode $I --layers 37 --parallel pp --bench --prefill-length 32768 --tg 16 --max-len 33024

# served truncated from the prepared directory (the original's tokenizer.json), MTP K = 2, one card / two
F=~/models/qwen3.8-flash-next
./build/src/cli/b70-serve $F --layers 18 --spec mtp --spec-max 2
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $F --layers 37 --parallel pp --spec mtp --spec-max 2
```

### Prefill on two cards

- Prefill runs on `l0` only. `--prefill-backend l0-int8` is refused, because its int8 linears rotate activations in
  whole 1024-wide Hadamard blocks and this model's hidden size is 2560. `--prefill-chunk` is at most 2048.
- Under `pp` each chunk crosses between the cards by copy, with card 0 running the next chunk while card 1 finishes
  this one. Under `tp` two chunks are in flight, each one's sums crossing on the copy engine beside the other's
  compute.
- Each layer's experts that live in host memory are copied onto the card on the copy engine while the layers before
  it compute. A short chunk (under 1,280 rows under `pp`, any chunk that runs alone under `tp`) reads them in place
  instead.
- Between prefills the idle prefill scratch is lent to the expert cache, so decode keeps more experts resident.

### Limits

- One request at a time, in every mode: `--max-num-seqs` above 1 is refused at startup; further requests wait their
  turn.
- At least 128 GB of system RAM.
- MTP in `b70-serve` only: `b70-decode --spec mtp` and `--draft-vocab` are refused.
- bf16 KV cache only (`--kv-cache int8` is refused).
- `reasoning_effort` is not forwarded to the template, as for Qwen3.8.
- No `--profile`. Context is capped at the trained 262,144.

Variables for tests and timing only:

| variable | values | what it does |
|---|---|---|
| `B70_Q4_ATTN` | `flash` \| `eager` | `eager` binds the reference's attention chain: in decode one launch in place of flash's two in each QSA layer, and the eager sparse attention in prefill |
| `B70_Q4_MTP_NORM` | `single` \| `per_stream` | the MTP head's `pre_fc_norm_hidden` form |
| `B70_Q4_TIE_TOL`, `B70_Q4_SEL_TOL` | a float | the golden test's routing and selection near-tie tolerances (1e-3 proposed) |
| `B70_MTP_ADAPTIVE` | `0` | a fixed K (above) |
| `B70_Q4_RESIDENT` | `all` \| `none` \| `random:<f>:<seed>` \| `adversarial` | a test placement of the routed experts, at any layer count |
| `B70_Q4_EXPERTS` | `table` (default) \| `direct` | `direct`: the kernels without the address table, only when every expert is resident |

## Kolibri-1

Aleph Alpha's Kolibri-1 (`model_type kolibri1`) is an MoE model:

- 50 MoE layers: 384 experts, top 6, plus an ungated shared expert;
- hidden size 2560, 48 query and 4 KV heads of 128;
- 40 sliding-window layers (513 keys, with RoPE) and 10 full layers without position encoding (NoPE);
- sandwich norms, a 128,000-id vocabulary and a trained context of 262,144.

Our int4 checkpoint is `urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` (AutoRound g64 symmetric, every routed and shared
expert in int4; the engine reads the shared expert as one more int4 block). `tools/quantize/kolibri/make_synth.py`
makes a small synthetic checkpoint in the same form.

Decode, prefill and serving are built; there is no speed row. At int4 the model holds about 42.5 GB of weights, so it
runs on two cards by default (`--parallel auto`). `--parallel 1` is refused unless the model fits one card: a
synthetic checkpoint, or `--layers N` (the first N layers only; 30 layers are about 26 GB).

Expected figures (derived): 756 launches per token (856 with `B70_KOLIBRI_ATTN=eager`) and about 2.39 GB read per
token with int4 attention and the int8 head, a bandwidth ceiling near 250 t/s on two cards.

Prefill (`--prefill`, `--prefill-length N`) runs chunks of up to 2048 positions on the `l0` backend: 2252 launches per
chunk plus 5 for the head, with about 0.9 GB of prefill scratch per card. Each layer takes 45 launches: the attention
linears as bf16 slabs plus `pf_gemm`, windowed flash attention over the 4096-slot ring, and the 384 experts sorted and
dequantised to bf16 in 512 MiB weight batches for the grouped GEMMs. On two cards each chunk crosses once, by copy,
and the cards take turns; they do not overlap.

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ

# every command below needs two cards unless it says otherwise
./build/src/cli/b70-decode $K --ids k.ids --n 32                                  # split by bytes
./build/src/cli/b70-decode $K --ids k.ids --n 32 --pipeline-split 25
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --max-len 40960
./build/src/cli/b70-decode $K --parallel 1 --layers 30 --ids k.ids --n 8          # one card, 30 layers
B70_KOLIBRI_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill                        # prefilled
./build/src/cli/b70-decode $K --bench --prefill-length 32768 --tg 16 --max-len 40960   # the prefill bench row
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill --prefill-chunk 1000
```

### Serving Kolibri-1

`b70-serve` runs the same engine behind the OpenAI server, on two cards by default.

- Template: Kolibri's ChatML template, the string in the checkpoint's `tokenizer_config.json`, byte-identical to
  transformers' `apply_chat_template` on 16 test cases.
- Tokenizer: its 128k tokenizer (127,998 ids; the head's last two rows have no token and are never emitted).
- Reasoning: the model opens its reasoning with `<think>`, which comes back as `reasoning_content`. The
  `chat_template_kwargs` `reasoning_effort` (`none` turns thinking off; `minimal`, `low`, `medium`, `high` (the
  default), `xhigh`, `max`) and `enable_thinking: false` reach the template.
- Tool calls: hermes JSON inside `<tool_call>`, returned as OpenAI tool calls (`finish_reason` `tool_calls`). A call
  that does not parse comes back as content.
- Sampling: a request without `temperature`, `top_p` or `top_k` samples with the checkpoint's
  `generation_config.json` values (1.0 / 0.97 / 128). `"temperature": 0` is greedy.
- Prefix cache: each snapshot carries the 40 sliding-window rings' last 512 positions (41.9 MB) beside the full
  layers' KV cache (20 KiB per position). `--max-len auto` reaches 262,144 on two cards.

```sh
# every command below needs two cards
./build/src/cli/b70-serve $K                                    # --max-len auto (262,144), int8 head
./build/src/cli/b70-serve $K --max-len 131072
./build/src/cli/b70-serve $K --prefix-split-last                # agentic sessions
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Wie spät ist es in Berlin?"}], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

Box checks: `tools/probe/kolibri_passkey.sh` (passkey 3 of 3 at 262,144 on two cards), and the German tool-call set
`tests/golden/toolcall-kolibri-de` (run `tools/toolcall/engine_generate.sh`, then `score.py --format hermes` against
the bf16 source's reference made by `oracle_generate.py --model kolibri`).

Limits, for `b70-decode` and `b70-serve` alike:

- Prefill on `--prefill-backend l0` only: `l0-int8` is refused, because its int8 linears rotate in 1024-wide Hadamard
  blocks and the hidden size is 2560. `--prefill-chunk` is at most 2048.
- No MTP head, so no `--spec mtp` or `lookup`, and `--spec auto` means off. `--spec dflash` is the DSpark drafter
  below.
- `--kv-cache int8` is refused (bf16 only). No `--profile`. Context is capped at the trained 262,144.
- The bench prompt is the first 42 ids of `de_prose.txt` through Kolibri's tokenizer.

Variables: `B70_KOLIBRI_ATTN` (`flash` | `eager`, decode and prefill alike) and `B70_KOL_TIE_TOL` (the golden test).

### Kolibri-1 with the DSpark draft model

This is built but has not yet run on the cards. `b70-serve` serves Kolibri-1 with speculative decoding from the
community DSpark drafter `intuit/Kolibri-1-dspark` (5 layers, hidden size 2560, 826 MB in bf16). The drafter reads
Kolibri's residual stream at the outputs of layers 3, 13, 23, 33 and 43 and drafts a block of up to 7 ids in one
pass. Kolibri checks them in one step of up to 8 rows. Greedy output is plain decoding's, token for token, and a
sampled request keeps the model's own distribution.

It plays the role DFlash plays for Qwen3.8, with three differences:

- each block row predicts the next position (the first draft comes from the anchor's own row);
- the block's attention is causal;
- a small Markov head (rank 256) corrects each row's candidates using the previously chosen id.

The drafter's `config.json` selects all of this; there is no extra flag.

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ
D=intuit/Kolibri-1-dspark                                  # fetch it into the HF cache first

# both need two cards
./build/src/cli/b70-serve $K --spec dflash --draft-model $D                 # drafts per step from acceptance, up to 7
./build/src/cli/b70-serve $K --spec dflash --draft-model $D --spec-max 4
```

It costs the drafter (bf16, about 1 GB with its buffers) and a copy of Kolibri's embedding (655 MB) on the second
card, plus verify scratch on both; `--max-len auto` takes them out of the context it gives. After each request the
log's `dspark:` line gives the accepted drafts and the tokens per verify. The model card reports about 3 tokens per
verify on chat and about 4 on maths, on vLLM.

Limits: `b70-serve` only (`b70-decode` decodes Kolibri plainly); `--spec lookup`, `dflash-lookup` and `mtp` are
refused; `B70_KOLIBRI_ATTN=eager` is refused with `--spec`, because the verify lists are flash only; the drafter runs
in bf16. The speed is unmeasured: the cost table that picks the number of drafts is derived from the bytes a step
reads, and each verified row reads its own six experts per layer, so deep blocks cost more than on a dense model.

#### What to check on the first card run

The card tests are `ctest -R 'kolibri_verify|kolibri_dspark'`: the verify lists against the plain decode list
(synthetic and real model), and the real model with the drafter (greedy equals plain, tokens per verify, a seeded
sampled request). The code marks what to check with `TODO(card)`:

1. The load copies Kolibri's 655 MB embedding from card 0 to card 1 for the drafter (`cli/kolibri_spec.h`). If the
   first run stalls at load, look there first, and upload the copy from the host instead.
2. The cost table that picks the number of drafts (`MtpCost::kolibri_dspark`) is derived from bytes; measure the
   verify at M = 1 to 8 and the draft pass.
3. `kol_attn_decode` spills a few registers at M = 3 and 4 (ocloc warns); time the two halves of a deep verify's
   attention.
4. Each verify row reads its own six experts per layer; price that, then decide whether to read each distinct expert
   once.
5. The taps are the outputs of layers 3, 13, 23, 33 and 43 (the drafter's layer ids minus one, vLLM's rule); a low
   acceptance points here first.
6. The Markov walk looks at the head's top 16 per row, not the whole vocabulary; measure the acceptance gap.
7. The prompt's last 2048 positions reach the drafter 8 rows per replay (256 replays); time it.
8. `kolibri_dspark_test`'s bar (1.2 tokens per verify) is a placeholder; set it from the run.
