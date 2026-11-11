# Running the models

How to download, serve, decode and benchmark each model the engine knows, what each one supports today,
and which of those configurations have actually run on a B70. Every flag and environment variable is
explained in the project README's tables; this page says which ones matter for which model, and which
setup to pick for which job.

Numbers are **measured** (on the card, with its row in [BENCHMARKS.md](BENCHMARKS.md)) or **derived**
(arithmetic from the memory planner or byte counts), as in the specs. The speed numbers below were taken
on a shared box; the quiet-box measurement session in [open-work.md](open-work.md) re-takes them.

## Which setup for what

All of this is Qwen3.8-27B, the model everything is tuned for. The box has two cards; the engine also
runs on one. The flags are explained in the next section.

| you want | run | why |
|---|---|---|
| the fastest single session (one agent, one request at a time) | nothing extra: `--parallel auto` on two cards is **tp** (tensor parallel: every layer split over both cards) | about 1.46x one card's decode with `--spec auto`; the context is the full 262144 with the bf16 cache |
| 2 to 8 requests at once (several agents, or an agent and its helpers) | `--max-num-seqs N`: on two cards that is **replica** (one copy of the model on each card, the requests spread over both) | aggregate speed grows with N: 1 / 2 / 4 requests 72 / 137 / 206 t/s (measured, MTP, shared box) |
| a model too big for one card that has no tp | `--parallel pp` (pipeline parallel: the layers split between the cards) | about one card's speed, but each card holds half the weights, so more context fits |
| one card only | `--max-num-seqs` up to 4 slots on that card; `--kv-cache int8` for long context beside the slots | 1 / 2 / 4 requests 72 / 106 / 125 t/s (measured, MTP, shared box) |

How long the contexts are (derived, per card, with MTP):

| `--max-num-seqs` on a card | bf16 cache | int8 cache |
|---|---|---|
| 1 | 171008 | 262144 |
| 2 | 129536 + 32768 | 258048 + 32768 |
| 4 | 46080 + 3 x 32768 | 167936 + 3 x 32768 |

The first number is slot 0, the largest context that fits beside the other slots; the others are slots
of 32768. On two cards `--max-num-seqs 2` is one slot a card (each the "1" row), 4 is two slots a card,
8 is four slots a card. Tensor parallel with several slots was tried and ruled out (replica is faster at
4 requests, 206 against 121 t/s; notes in [low-priority-work.md](low-priority-work.md)).

```sh
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
./build/src/cli/b70-serve $Q --prefix-split-last                                   # one session, tp on two cards
./build/src/cli/b70-serve $Q --max-num-seqs 4 --kv-cache int8 --prefix-split-last  # 4 requests, replica (2 a card)
./build/src/cli/b70-serve $Q --parallel 1 --max-num-seqs 4 --kv-cache int8         # one card, 4 slots
```

## What the CLIs take

`b70-serve` (the OpenAI-compatible server) and `b70-decode` (ids in, ids out, plus benchmark and profile
modes) choose the model from the checkpoint's `config.json`; there is no model flag. The engine has no
tuning flags and runs on B70 only. What you choose:

- **`--parallel auto|1|pp|tp|replica`** (default `auto`). `auto` on two cards: Qwen3.8 runs **tp** when
  `--max-len` is auto and `--max-num-seqs` is 1, **replica** when `--max-num-seqs` is above 1; in every
  other case one card if the model and the asked context fit it, else **pp**. `1`: one card. `pp`: the
  layers split over both cards (`--pipeline-split auto|N` and `--pipeline-handoff copy|peer` are its
  overrides and imply pp). `tp` and `replica` are Qwen3.8 only; every other model refuses them by name.
  Kolibri-1 and Nemotron 3.5 Bolt need both cards and run on two by default. `b70-decode` takes `auto`,
  `1`, `pp` and `tp` (replica is the server's).
- **`--max-num-seqs N`** (`b70-serve`, Qwen3.8; default 1; vLLM's flag, the same meaning): how many
  requests run at once, and the context follows (the table above). One card takes up to 4, two cards
  (replica) up to 8; `--parallel replica` alone means 2 (one request a card). `--parallel tp` and `pp`
  with N above 1 are refused, and so are `--spec dflash` and `--draft-vocab` beside several slots on a
  card. A request that reaches the end of its slot finishes with `length` and the server log says so. An
  explicit `--max-len L` is slot 0's length; one that does not fit beside the other slots fails at
  startup. Startup prints each card's slot layout.
- **`--kv-cache bf16|int8`** (default `bf16`). The context (the KV cache) stored as 16-bit numbers, or as
  8-bit numbers with a scale per token and head. int8 halves the cache, which is what puts the full
  262144 tokens on one card beside MTP and leaves room for several slots; the price is a little
  accuracy, and DFlash does not run on it. Built for Qwen3.8 and K2-Horizon; the other models are bf16
  only (table below).
- **`--prefix-cache-gb auto|N`** (default `auto`). The prefix cache lives in pinned system RAM, not on
  the card: every prefill is written through to it, so a later request with the same history restores
  instead of re-prefilling. `auto` takes the smallest of 32 GiB, half the RAM and the available RAM
  minus 8 GiB, and is off below 4 GiB; `0` is off. It is allocated as 2 GiB pieces (a host allocation
  that both cards use must stay under 4 GiB, or the second card's mapping can fail).
- **`--spec auto|off|mtp|lookup|dflash|dflash-lookup`** (default `auto`) with `--spec-max K`: the
  speculative proposer, explained after this list. `auto` is the model's own MTP head where it has one
  (Qwen3.8, Ornith) and off elsewhere. `b70-decode` defaults to off so its rows stay comparable.
- **`--max-len auto|L`**: context capacity. `auto` is the largest multiple of 256 whose memory plan fits
  with the engine's 1.5 GB reserve left, capped at the trained context (262144 for the Qwen models);
  both CLIs print the length and the plan at startup. `b70-decode` keeps 16384 unless asked.
- **`--prefix-split-last`**: for agentic sessions with a thinking model, so each turn restores exactly
  where the next one diverges (README, `b70-serve` flags). Off for benchmarks.
- **`--prefill-backend l0|l0-int8`**: `l0-int8` runs the prompt pass with int8 activations (faster).
  Together with `--kv-cache int8` it is an accepted known fail (a startup warning: about 3 % of tool
  calls that copy exact text from a long prompt come out wrong); pair the int8 cache with `l0`.
- **`--draft-vocab off|32k|64k|128k`**, `--draft-model PATH`, `--layers N|auto`, `--log-requests DIR`,
  `--host`, `--port`: as in the README.

The engine decides the rest: the int8 `lm_head`, the memory it leaves free (1.5 GB), the draft model's
form (int8), the speculative cost tables, the served name (the repo id), the tokenizer, chat template and
generation defaults (the snapshot's). It refuses at startup, by name, what does not fit or is not built.
The int8 prefill and int8 cache, the hand-off between cards and the draft vocabulary are still being
measured; they go when the quiet-box session decides them (open-work.md).

## Which models, and what each supports

How a change reaches `main`: it is merged after the Mac checks, and the full box validation
(`tools/box_validate.sh`: golden sets, serving, prefix cache, speculative decoding) runs on `main` on
demand. A test that passes proves a configuration **runs and is correct**; it is not a speed row. What
is not finished is listed in [open-work.md](open-work.md).

- **Qwen3.8-27B** is the model in focus, with the full set of measured rows (prefill, decode, depth,
  prefix cache, MTP, DFlash, tensor parallel).
- **Ornith 1.5** is supported and its box tests pass; it has no speed row yet.
- **Qwen3.8-Flash-Next** is next. It runs only truncated (`--layers N`) because the whole model does not
  fit two cards; open-work.md has the plan for the full size.
- **K2-Horizon and Kolibri-1** are supported by the code and parked (out of focus); no speed rows.
- **Nemotron 3.5 Bolt** is parked until Flash-Next is done (operator, 2026-11-11). Its section stays
  because the code still serves it.
- Dropped and no longer supported: LFM2.5, Muse-Glimmer and Agnes 3.0 Flash (a `config.json` naming
  `AgnesForConditionalGeneration` is refused by name). Flint and Nanbeige are parked and not built.
  LiLiCorr is Bolt's draft model (`--spec dflash`, Bolt's section).

| | Qwen3.8-27B | Ornith 1.5 35B-A3B | K2-Horizon MoVA-36B-A4B | Kolibri-1 (spec 20) | Qwen3.8-Flash-Next (spec 21) |
|---|---|---|---|---|---|
| checkpoint | published (ours, int4 g64) | published (ours, spec 15 §13) | published (ours) | published (ours, int4 g64) | Intel's interim W4A16 (181 GB); our own is not made |
| `b70-serve` | yes | yes | yes (spec 18d) | yes (spec 20e), **two cards** by default | `--layers N` only (spec 21e; the whole model is refused, naming spec 22) |
| `--parallel tp`, `replica`, `--max-num-seqs` | yes | refused | refused | refused | refused |
| `--parallel pp` (two cards, spec 16d) | yes | yes | **refused** (one card) | its default (its own engine, spec 20e) | yes, `--layers N` (spec 21e) |
| `b70-decode` | yes | yes | yes | decode and prefill (spec 20d), **two cards** (the default) | decode (spec 21c) and prefill (spec 21d), **`--layers N` required** (the whole model needs spec 22) |
| speed rows in BENCHMARKS | yes | no | no | no | no |
| MTP head (`--spec mtp`; `--spec auto` = adaptive MTP) | yes | yes (MoE head) | none (`auto` = off) | none (`auto` = off) | `b70-serve --spec mtp`, a fixed K = `--spec-max` 1..3 (`auto` = off until spec 22's numbers); `b70-decode` refuses it |
| `--spec lookup` | yes | yes (needs the head) | no | no | yes (needs the head; spec 21e) |
| `--spec dflash` (spec 19) | yes, one card or `--parallel pp` (copy hand-off), bf16 KV | no | no drafter exists (refused) | refused | refused |
| `--kv-cache int8` | yes | not built | yes (spec 18e) | refused (bf16 only) | refused (bf16 only) |
| prefill backends | `l0` (default), `l0-int8` | `l0` (default); `l0-int8` opt-in, accuracy gates diagnostic (MoE routing) | `l0` only | `l0` only (spec 20d) | `l0` only (spec 21d; `l0-int8` refused) |
| KV per position (bf16) | 64 KiB | 20 KiB | 192 KiB | 20 KiB (+ 335.5 MB of sliding rings) | 24.75 KiB (12 QSA layers: K + V 2 KiB + 64 B of indexer keys each) |
| `--max-len auto`, one card, int8 head (derived) | 171008 with MTP (the default); ~201k with `--spec off` | 262144, with or without MTP | ~46k served; `b70-decode` ~50k decode-only | 262144 on two cards (split 25) | at `--layers 19` (Intel's forms, one card) 78336 decode-only; at `--layers 18` 70144 with the prefill planned |
| with `--kv-cache int8` (derived) | 262144, with or without MTP | - | ~90k served; `b70-decode` ~98k decode-only | - | - |

The `--max-len auto` lengths are the planner's at the default 1.5 GB reserve on the 32.53 GB card.

## Common to every model

**Build** as in the README (`cmake`, then `ctest`). The engine never downloads; fetch a checkpoint into
the Hugging Face cache once:

```sh
uvx --from huggingface_hub hf download <org>/<name>
```

Both CLIs then take the repo id (looked up under `$HF_HOME`, default `~/.cache/huggingface`) or a
snapshot directory.

**Which cards.** One or two Intel Arc Pro B70s (PCI id 0xE223), with peer access between them when there
are two (one card writing straight into the other's memory; tp needs it). The hardware gate checks the
vendor, the device id, the Xe-core count and peer access before any weight is read; the Arc Pro B65 and
B60 are refused as not yet tested, and more than two cards are refused. One card is device 0, two are
devices 0 and 1 of what `ZE_AFFINITY_MASK` shows - `ZE_AFFINITY_MASK=1` runs on the second card alone.

**One model per server process.** The prefix cache belongs to that process and that model.

**Connecting a client.** `b70-serve` is OpenAI compatible: point the client (opencode or any other) at
`http://<host>:8000/v1`; the server lists the model under its repo id (or the snapshot directory's name)
and answers to any model name in a request.

**Benchmarking.** The command behind the vLLM comparison rows, against a running server (no
`--prefix-split-last`; it is for sessions, not benchmarks):

```sh
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model <repo id> \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

Its `tg` and `pp` figures read the same way as vLLM's: the server sends the role frame with the first
token, after prefill, as vLLM does (BENCHMARKS.md, "llama-benchy's own pp figure against our server").
Its `pp` is HTTP-inclusive; `b70-decode --bench --prefill-length` gives the device-side prefill time.

**Token ids for `b70-decode --ids`** are whitespace-separated, made with the checkpoint's own tokenizer
(needs `transformers`; `<snapshot>` is the snapshot directory in the cache):

```sh
python3 tools/oracle/tokenize.py <snapshot> encode prompt.txt > prompt.ids
python3 tools/oracle/tokenize.py <snapshot> decode $(cat out.ids)
```

`tests/golden/prompts/{prose,code,cjk}.ids` are ready-made ids for the Qwen tokenizer (Qwen3.8's).

**Two cards, pipeline parallel (`--parallel pp`, spec 16).** Qwen3.8 and Ornith run with their layers
split over GPUs 0 and 1: `b70-decode --parallel pp` (decode; the two-card prefill with `--prefill`;
`--spec mtp`) and `b70-serve --parallel pp` (everything the server does on one card). `--parallel auto`
picks pp by itself when the model and the asked `--max-len` do not fit one card and tp does not apply.
One answer is not faster than on one card; each card holds about half the model, so `--max-len auto`
reaches the trained 262144 tokens, with MTP too (derived, int8 head). The MTP head runs on the second
card with its own copy of the embedding (2.54 GB). K2-Horizon is one card only; Kolibri-1 runs on two
cards by default in `b70-decode` (spec 20c / 20d) and `b70-serve` (spec 20e, its own engine).

```sh
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
# an agentic session over both cards as pipeline parallel: 262144 tokens with MTP and the int8 cache
./build/src/cli/b70-serve $Q --kv-cache int8 --host 0.0.0.0 --port 8000 --parallel pp --prefix-split-last
# the peer hand-off (the first card's last kernel writes the hidden state into the second card)
./build/src/cli/b70-serve $Q --parallel pp --kv-cache int8 --pipeline-handoff peer
# b70-decode: the same ids as one card, greedy, with MTP across the split
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --parallel pp --spec mtp
```

Startup prints the split (`split: auto -> N`), the plan per card and both cards' memory lines (and
`parallel: auto -> ...` when `auto` chose).

## Speculative decoding and how its output is judged

A *proposer* guesses the next few tokens and the main model checks them all in one pass (a "verify" step
of M rows), so every correct guess is a token for free. `--spec` picks the proposer: `mtp` (the model's
own MTP head, spec 8), `dflash` (a small draft model, spec 19), `lookup` (the continuation of the
longest earlier repeat in the request, spec 19e) and `dflash-lookup` (lookup inside DFlash, spec 19f).
The verify step of every decode list at M >= 2 runs the small-batch int4 GEMM `gemm_sm` (spec 28b; M = 1
keeps the GEMV, bitwise as before); the DFlash drafter's int8 linears and Flash-Next's verify lists run
on it too. Verify attention runs the DPAS kernel (spec 28c) from the depth its table says and the v2
decode kernel below it.

Because the verify rows no longer run the same kernel as a plain step, speculative output is not
compared bit for bit with plain decode. It is graded by the **f32 judge** (`tests/server/spec_judge.h`,
doc 14): a speculative stream passes when it is no further from a float32 reference than plain decode on
the same context, or when every miss is a near-miss (`golden::NearMiss`, within 8 bf16 ulps of the f32
top). Plain, unchanged code paths still gate bitwise.

## Qwen3.8-27B

`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: the model everything is tuned for. 64 layers (48 delta
net, 16 attention), int4 g64, bf16 `lm_head` and MTP head in the file (the engine runs the head as int8).

**On the card (measured, [BENCHMARKS.md](BENCHMARKS.md)):** decode and prefill at depth (including 128k),
the 131072 and longer contexts with passkey, the prefix cache and `--prefix-split-last`, MTP and DFlash,
tensor parallel (plain decode at depth 4096: 33.49 t/s on one card, 53.95 t/s on tp; `--spec auto`:
62.12 against 90.39 t/s), replica and the slots (the table at the top), the golden gates and the
tool-call test.

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

# agentic session on two cards, the default: tp, the largest context (262144 with the bf16 cache),
# MTP choosing 0-3 guesses per step, the prefix cache sized from the machine's RAM
./build/src/cli/b70-serve $Q --host 0.0.0.0 --port 8000 --prefix-split-last

# the same on one card: the largest context that fits beside the MTP head (171008 bf16, 262144 int8)
./build/src/cli/b70-serve $Q --parallel 1 --kv-cache int8 --prefix-split-last

# several agents at once: 4 requests, replica on two cards (see the table at the top for the contexts)
./build/src/cli/b70-serve $Q --max-num-seqs 4 --kv-cache int8 --prefix-split-last

# an explicit context and cache size, no proposer
./build/src/cli/b70-serve $Q --parallel 1 --kv-cache int8 --max-len 131072 --prefix-cache-gb 32 \
    --prefix-split-last --spec off

# prompt lookup instead of the MTP guesses (one proposer per server)
./build/src/cli/b70-serve $Q --kv-cache int8 --prefix-split-last --spec lookup
```

```sh
# the BENCHMARKS.md rows: prefill 4096, then decode 256 at that depth (b70-decode defaults to one card)
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $Q --bench --depth 4096 --tg 256 --max-len 131072  # decode at a deep cache
./build/src/cli/b70-decode $Q --bench --depth 4096 --tg 256 --parallel tp     # the same on both cards

# greedy ids in, ids out
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --spec mtp

# per-launch anatomy of the decode step
./build/src/cli/b70-decode $Q --profile --depth 4096 --steps 32 --repeats 5
```

**Limits.** The trained context is 262144; with the bf16 KV cache one card holds 171008 with the MTP head
and ~201k plain (derived), and tp reaches the full 262144 because each card holds half the cache.
`--kv-cache int8` needs the `l0` or `l0-int8` prefill. Under tp: no DFlash, no draft vocabulary, no
`l0-int8` prefill, no `--profile`, one request at a time. `b70-decode` keeps `--max-len 16384` and
`--spec off` by default so its rows stay comparable.

**Relevant variables:** `B70_KV_CACHE` (the default of `--kv-cache`), `B70_PREFILL_PROFILE` for prefill
phase timings, `B70_GIT_SHA` for bench rows.

### DFlash draft model (spec 19)

`z-lab/Qwen3.8-27B-DFlash2` drafts a block of up to 7 tokens per step from Qwen3.8's own hidden states (the
outputs of layers 5 / 19 / 33 / 47 / 61). Its acceptance on our prompts was measured on the Mac CPU
(`docs/probes/probe-dflash-2026-10-08.md`: 7.21 tokens per verify at K = 7 on agentic tool calls, 6.15 on
code, ~2.2 on prose; int8 and 4-bit drafters equal bf16). It has run on the card since 2026-10-27
(BENCHMARKS "DFlash"), and the verdict is an option, not the default: `--spec auto` stays the adaptive MTP,
and `dflash` would become the default only if it beat the adaptive MTP by 10 % on the opencode recording.
The drafter's int8 linears run on the small-batch GEMM `gemm_sm_i8w` (draft step 16.3 -> 8.9 ms).
`--spec dflash-lookup` (spec 19f) adds prompt lookup inside DFlash for edit / copy turns (one card):

```sh
# one card: the drafter beside the target (int8 at load, 2.05 GB + 1.06 GB of verify slots), K per request (auto, <= 7)
./build/src/cli/b70-serve $Q --parallel 1 --prefix-split-last \
    --spec dflash --draft-model z-lab/Qwen3.8-27B-DFlash2
# two cards, pipeline parallel: the drafter on the second card beside lm_head (the copy hand-off only)
./build/src/cli/b70-serve $Q --parallel pp --spec dflash
# greedy without the server (one card), K <= 7 per step
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/code.ids --n 256 --prefill \
    --spec dflash --spec-max 7
```

- **Defaults:** the drafter quantised to int8 at load (same acceptance as bf16 at half the bytes),
  `--spec-max 7`, `--draft-vocab off` (a ranked 32k vocabulary was ranked in-sample; by id it loses up to
  1.4 tokens at K = 7).
- **Refused, by name:** `--kv-cache int8` (the int8 attention is built for M <= 4 verify rows, DFlash needs
  8), `--parallel pp --pipeline-handoff peer`, `--parallel tp` and `replica`, several slots a card
  (`--max-num-seqs` above 1 on one card), `b70-decode --parallel pp`, every other model.
- **The prefix cache** keeps the drafter's ring (41.9 MB) with every snapshot, so a conversation restored
  from a request's prompt-end or request-end snapshot drafts exactly as an unbroken one. **Caveat:** a
  snapshot at a block end inside a long prefill holds the ring as that prefill left it (the prefill
  projects only its last 2048 positions), so a restore there drafts worse - never differently - until 2048
  new positions have refilled the ring.

Re-running the acceptance measurement (a Mac or the box CPU; ~20-25 h on a 64 GB Mac with the container at
~54 GB):

```sh
# downloads (HF cache): the target in bf16 (~54 GB), the drafter (3.85 GB); optional 4-bit drafter arm
uvx --from huggingface_hub hf download Qwen/Qwen3.8-27B
uvx --from huggingface_hub hf download z-lab/Qwen3.8-27B-DFlash2
uvx --from huggingface_hub hf download syvai/Qwen3.8-27B-DFlash2-W4A16
# plans and estimates only, no model loaded
DRY_RUN=1 tools/oracle/dflash_p0.sh
# the run: tap dumps, the draft-vocabulary ranking, acceptance per drafter arm (resumable, detached)
MEM=54g nohup tools/oracle/dflash_p0.sh > /dev/null 2>&1 &
tail -f oracle-out-19a/p0.log                       # then oracle-out-19a/accept/summary.md
# one arm again, e.g. after a new ranking
STEPS="accept" ARMS=int8 tools/oracle/dflash_p0.sh
```

## Ornith 1.5 35B-A3B

`ornith-ai/Ornith-1.5-35B-A3B` is published in bf16 only (71.9 GB), which the engine
refuses: it loads int4 GPTQ. The checkpoint is the operator's AutoRound W4A16 export,
[`urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ`](https://huggingface.co/urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ)
(22.99 GB: int4 g64 symmetric, GPTQ packing, experts per expert as
`experts.E.{gate,up,down}_proj`, the routers and the MTP head bf16, and - unlike Qwen3.8's
export - `in_proj_a` / `in_proj_b` int4 too, which the engine loads as an int4 a||b; spec 15
§13 lists every fact read from it).

The first mixture-of-experts model (spec 15): 40 layers (30 delta net, 10 attention),
hidden 2048, 256 experts with top-8 plus a shared expert, ~3 B parameters active per token.
Decode (15c), prefill (15d) and serving with its own chat template (Qwen3.5's), Qwen XML
tool calls and the MoE MTP head (15e) are built.

**Status: built; no speed row yet.** Expected (derived): ~1.86 GB
read per token, so the bandwidth ceiling is far above Qwen3.8's (~270 t/s), and launches,
not bytes, will set the decode rate; 526 launches per token. Prefill estimated at 3-6k t/s.
Weights 19.45 GB with the int8 head (derived), +0.50 GB for the MTP head. For comparison,
the checkpoint's model card reports vLLM XPU on a B70 at 99.8 t/s decode (tg256, c1) and
7090 t/s prefill (pp4096, c1) - not measured here.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
uvx --from huggingface_hub hf download $O                 # 22.99 GB

# every command below: no speed row yet
# agentic session: the full trained context, 262144, fits with or without MTP (derived)
./build/src/cli/b70-serve $O --host 0.0.0.0 --port 8000 --prefix-split-last

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256 --prefill-backend l0
./build/src/cli/b70-decode $O --ids prompt.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $O --profile --depth 4096
```

**Limits.**

- `--kv-cache int8` is not built at Ornith's attention shape (capture names the missing
  binary); it is not needed for context, since 262144 fits in bf16.
- Prefill runs on `l0` (the default) and `l0-int8` (opt-in). On this MoE model `l0-int8` loses routing
  accuracy against f32 (the box, 2026-10-30: 1314 differing expert sets on the prefill path against the
  bf16 reference's 1140), and no 8-bit activation scheme studied won it back, so its accuracy gates are
  diagnostic - printed, not gated (operator ruling 2026-10-31); use it for speed only. Its kernels'
  own gate (`prefill_gate_ornith_l0-int8`, judged by the quantisation-aware f32 reference) still binds.
- The adaptive MTP (`--spec auto`) decides with Qwen3.8's cost table until the card measures Ornith's (the
  startup line says so); no default K is chosen for Ornith. The head's bf16 experts are
  quantised to int4 at load (round to nearest), which can change acceptance, never output.
- The checkpoint's `tokenizer.json` defines Qwen3.8's 248077 ids, the count the greedy
  argmax masks from, so `b70-serve` prints no vocabulary note. It tokenises the committed
  `tests/golden/prompts` (prose, code, cjk) to the committed ids; its pre-tokenizer splits
  combining marks differently from the bf16 base checkpoint's file (`\p{M}`), which only
  text with combining marks sees.

**Relevant variables:** `B70_KV_CACHE` must stay `bf16`; `B70_PREFILL_PROFILE` applies.

## K2-Horizon MoVA-36B-A4B

`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` (spec 18; 22.22 GB): a second
engine beside the Qwen one. 48 layers, every one full attention (32 q-heads, 8 kv-heads,
head dim 128); layers 3-47 route values through 64 value experts (MoVA, top-4) and the FFN
through 100 experts (top-8) plus a shared one. Its own tokenizer (250,624 ids, BOS 0) and
chat template. No MTP head. Trained context 524288.

**Status: built, parked; no speed row yet.** Decode (18b) and prefill (18c)
run through `b70-decode`; `b70-serve` serves it (18d: its chat template, tool calls and
reasoning, K2's own engine behind the server, KV-only prefix-cache snapshots - host-tested only).
Expected
(derived): 717 launches per token, 3.15 GB read per token with the int8 head (roofline
~190 t/s from bytes alone), prefill ~3,400 t/s at 4096. For reference, vLLM on **two**
cards (pipeline parallel, fp8 KV) measured 44.43 t/s decode.

```sh
uvx --from huggingface_hub hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ
K=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ

# every command below: no speed row yet
# K2 ids start with BOS 0: encode with --bos
python3 tools/oracle/tokenize.py <k2-snapshot> encode --bos prompt.txt > k2.ids

./build/src/cli/b70-decode $K --ids k2.ids --n 32                    # decode only, one replay per id
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill          # prefill on l0
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $K --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill --max-len auto   # ~46k, derived

# serving (spec 18d): int8 head, prefix cache on, --max-len auto (~46k bf16 KV, ~90k int8 KV)
./build/src/cli/b70-serve $K --kv-cache int8 --port 8000
# K2's template variables per request: "chat_template_kwargs": {"tool_call_format": "xml" |
# "xml_typed" | "json", "reasoning_effort": "high" | "medium" | "low"}
```

**Limits.**

- Served on one card only (`--parallel pp` refused; `--parallel auto` is one card), no llama-benchy
  row yet. `b70-serve` refuses `--spec mtp` / `lookup` / `dflash` and a non-`l0`
  `--prefill-backend` by name; its `--spec auto` is off.
- Context: the bf16 KV cache costs 192 KiB per position, so `--max-len auto` gives ~49.9k
  decode-only, ~45.8k when the run prefills (the int8 head) - derived. `b70-decode`'s default is
  16384. Long context needs the int8 KV cache (spec 18e); two cards are not built for K2.
- Prefill runs on `l0` only: `--prefill-backend l0-int8` is refused by name.
  Omitting `--prefill-backend` is fine.
- Refused: `--spec mtp` (no head), `--profile` (not built). `--kv-cache int8` is built (spec 18e).
  `--spec lookup` has nothing to run on: it needs verify lists, which only an MTP head
  provides today.
- Which attention matches the reference better is decided by the f32 judge (the MoE gates
  are graded against an f32 reference, with a one-bf16-ulp near-tie rule, since routing
  flips on near-ties): `flash` is the attention (the eager diagnostic arm, `B70_K2_ATTN=eager`, went with the
  zero-config cleanup; the f32 judge had decided).

**Relevant variables:** `B70_PREFILL_PROFILE` times K2's prefill phases but `b70-decode` does not
print the table for K2 yet. `B70_KV_CACHE` does not apply to K2 (`--kv-cache`).

## Kolibri-1 (spec 20)

Aleph Alpha's Kolibri-1 (`model_type kolibri1`): 50 MoE layers (384 experts top 6 + an ungated shared
expert), hidden 2560, 48 / 4 heads x 128, 40 sliding-window layers (513 keys, RoPE) and 10 full NoPE
layers, sandwich norms, a 128000-id vocabulary, trained context 262144. Our int4 checkpoint is
`urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` (AutoRound g64 sym, every routed AND shared expert int4 - the
engine reads the shared expert as one more int4 block, operator ruling 2026-10-31); a synthetic checkpoint in
the same form comes from `tools/quantize/kolibri/make_synth.py`.

**Status: built, parked; no speed row yet.** Decode (spec 20c), prefill (spec 20d) and
serving (spec 20e, `b70-serve`), on **two cards**: at int4 the model holds ~42.5 GB of weights, so two cards are Kolibri's
default (`--parallel auto`) and `--parallel 1` is refused unless the model fits one card (a synthetic checkpoint, or `--layers N`:
development mode, the first N layers only - e.g. 30, ~26 GB). Expected (derived): 756 launches per
token (856 with `B70_KOLIBRI_ATTN=eager`), ~2.39 GB read per token with int4 attention and the int8 head -
a roofline near 250 t/s on two cards. Prefill (`--prefill`, `--prefill-length N`) runs chunks of up to
2048 positions on the `l0` backend only: 2252 launches per chunk (45 a layer: the attention linears as
bf16 slabs + `pf_gemm`, the windowed flash attention over the 4096-slot ring, the 384 experts sorted and
dequantised to bf16 in 512 MiB weight batches for the grouped GEMMs) + 5 for the head, ~0.9 GB of prefill
scratch a card; on two cards each chunk crosses once, by copy, and the cards take turns (no overlap yet).

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ          # spec 20b

# every command below: no speed row yet; needs two cards
./build/src/cli/b70-decode $K --ids k.ids --n 32                                  # two cards, split by bytes
./build/src/cli/b70-decode $K --ids k.ids --n 32 --pipeline-handoff peer --pipeline-split 25
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --max-len 40960
./build/src/cli/b70-decode $K --parallel 1 --layers 30 --ids k.ids --n 8          # one card, 30 layers
B70_KOLIBRI_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill                        # spec 20d: prefilled
./build/src/cli/b70-decode $K --bench --prefill-length 32768 --tg 16 --max-len 40960   # the pp row
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill --prefill-chunk 1000
```

**Serving (spec 20e, `b70-serve`).** The same engine behind the OpenAI server, on two cards by default:
Kolibri's ChatML template (the checkpoint's `tokenizer_config.json` string; byte-identical to transformers'
`apply_chat_template` on 16 cases, measured on the Mac), its 128k tokenizer (127998 ids; the head's last two
rows have no token and are never emitted). The model opens its reasoning with `<think>` - it comes back as
`reasoning_content`; `chat_template_kwargs` `reasoning_effort` (`none` turns thinking off; `minimal`, `low`,
`medium`, `high` - the default -, `xhigh`, `max`) or `enable_thinking: false` reach the template. Tool calls
are hermes JSON in `<tool_call>` and come back as OpenAI tool calls (`finish_reason` `tool_calls`); a call
that does not parse comes back as content. A request without `temperature` / `top_p` / `top_k` samples with
the checkpoint's `generation_config.json` (1.0 / 0.97 / 128); `"temperature": 0` is greedy. The prefix cache
works as for every model: each snapshot carries the 40 sliding rings' last 512 positions (41.9 MB, derived)
beside the full layers' KV (20 KiB a position). `--max-len auto` reaches 262144 on two cards.

```sh
# every command below: no speed row yet; needs two cards
./build/src/cli/b70-serve $K                                    # two cards, --max-len auto (262144), int8 head
./build/src/cli/b70-serve $K --pipeline-handoff peer --max-len 131072
./build/src/cli/b70-serve $K --prefix-split-last                # agentic sessions: the next turn restores at len - 1
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Wie spät ist es in Berlin?"}], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

**Box checks:** `tools/probe/kolibri_passkey.sh` (passkey 3 / 3 at 262144 on two cards) and the
German tool-call set `tests/golden/toolcall-kolibri-de` (`tools/toolcall/engine_generate.sh`, then
`score.py --format hermes` against the bf16 source's reference, `oracle_generate.py --model kolibri`).

**Limits.** Prefill on `--prefill-backend l0` only (`l0-int8` refused: its h8 linears rotate in
1024-k Hadamard blocks and the hidden is 2560), `--prefill-chunk`
at most 2048; no `--spec mtp` / `lookup` / `dflash` (no head, no verify lists; its `--spec auto` is off),
`--kv-cache int8` refused (20 KiB a position: bf16 only), no `--profile`, context capped at the trained
262144 (decision 3) - `b70-decode` and `b70-serve` alike. The
bench prompt is the first 42 ids of `de_prose.txt` through Kolibri's tokenizer (spec 20e).

**Relevant variables:** `B70_KOLIBRI_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_KOL_TIE_TOL`
(the golden test).

## Qwen3.8-Flash-Next (spec 21)

Qwen's `Qwen3.8-Flash-Next` (`model_type qwen4_exp`): 48 layers (36 GatedDeltaNet with a sigmoid output gate, 12 QSA
sparse-attention layers that attend the top 512 blocks of 4 positions plus the open block), a 512-expert / top-10
MoE in every layer, 4-stream hyper-connections instead of a residual, and a per-layer n-gram embedding (PLE) whose
51.8 GB int8 table lives in host memory and is read by the card directly. Intel's interim checkpoint
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (int4 g128 experts, bf16 dense layers) loads through spec 21b's exact
g64 expansion; its PLE table must be converted once (`tools/box_validate/qwen4exp_oracle.sh <data> ple-int8`, ~51.8 GB
beside the snapshot). Our own AutoRound g64 export is not made yet.

**Status: built; runs truncated only; no speed row yet.** Decode (spec 21c), prefill (spec 21d) and serving
(spec 21e): **the whole
model does not fit two B70s** (its routed experts alone are 64.17 GB at int4 g64) and is refused, naming spec 22
(the expert-offload tier); `--layers N` (development mode: the first N layers, then the final mixer and the head) or
`--layers auto` (the planner's N, derived: 18-19 of Intel's layers on one card, 38-39 on two, by `--max-len`) is
required. Expected (derived): 779 launches per token at 48 layers, 75 at `--layers 4`; `B70_Q4_ATTN=eager` binds the
reference's attention chain (one launch for flash's two in each QSA layer) in decode and the eager sparse attention
in prefill. `--prefill` / `--prefill-length N` run the prompt in chunks of up to 2048 (`--prefill-chunk C`) on the
`l0` backend: spec 6's flash for the positions that see at most 2051 others, the indexer's own selection and a sparse
flash past them; 358 / 361 / 360 launches a chunk at `--layers 4` (all dense / straddling 2050 / all sparse), ~3950
at 48 layers (derived). The prefill scratch is ~1.5 GB a card (the 512 MiB expert weight batch, the score rows that
grow with `--max-len`), planned into `--layers auto` / `--max-len auto` when a run prefills: with it Intel's forms fit
18 layers on one card and 37 on two at 32768 (derived).

**Serving (spec 21e).** `b70-serve <checkpoint> --layers N` serves OpenAI chat on one card or two (`--parallel pp`) through
the Qwen chat path: the template is Qwen3.8's byte for byte, XML tool calls and `<think>` reasoning as for Qwen3.8,
greedy unless the request asks (decision 11 open). **The tokenizer must be the original's** (`Qwen/Qwen3.8-Flash-Next`'s
`tokenizer.json`, whose split adds `\p{M}`), and it is the snapshot's: Intel's checkpoint ships Qwen3.8's file, which
`b70-serve` refuses - serve a snapshot directory that ships the original's (our own export, not made yet; the zero-config
cleanup removed `--tokenizer`). On the box that directory is `~/b70/models/qwen3.8-flash-next/`: links to every file
of Intel's snapshot except `tokenizer.json`, which links to the original's; `qwen3.8-flash-next-ple-int8/` beside it
links the PLE int8 file.
The prefix cache keeps this family's state with each snapshot (115.7 MB: the 36 GDN states and conv rows, the PLE
history, the QSA open block's raw keys; + 21 KB with the head) and 25,344 B a position in its blocks (+ 2,112 with the
head), one layout on one card or two. `--spec mtp --spec-max K` (K 1-3, a fixed K) drafts with the MTP head (vLLM's form;
`pre_fc_norm_hidden` as one RMS over the 4 streams, `B70_Q4_MTP_NORM=per_stream` for the other form; draft steps after
the first reuse step 0's selection) and verifies K + 1 rows in one list (790-795 launches at 48 layers, 86-91 at
`--layers 4`, derived); its `--spec auto` is off until spec 22 measures the acceptance on the whole model. The head's bf16 experts are RTN-quantised to int4 g64 at load; on two cards it runs on the second and
reads the first card's embedding over peer.

```sh
I=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound

# every command below: no speed row yet; needs the PLE int8 file beside the snapshot (<snapshot>-ple-int8/)
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_short.ids --n 32
./build/src/cli/b70-decode $I --layers 38 --parallel pp --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $I --layers auto --max-len auto --ids tests/golden/prompts/q4exp_short.ids --n 8
# spec 21d: the prompt prefilled in chunks (l0 only), then decode; the pp row
./build/src/cli/b70-decode $I --layers 18 --ids tests/golden/prompts/q4exp_4k.ids --n 32 --prefill
./build/src/cli/b70-decode $I --layers 37 --parallel pp --bench --prefill-length 32768 --tg 16 --max-len 33024
# spec 21e: served truncated from a snapshot that ships the original's tokenizer.json, MTP K = 2, one card / two
F=~/b70/models/qwen3.8-flash-next   # Intel's files, the original's tokenizer.json
./build/src/cli/b70-serve $F --layers 18 --spec mtp --spec-max 2
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $F --layers 37 --parallel pp --spec mtp --spec-max 2
```

**Limits.** Prefill on `l0` only (`--prefill-backend l0-int8`: spec 5's h8 linears rotate in whole 1024-k Hadamard
blocks and the hidden is 2560), `--prefill-chunk` at most 2048; the two cards
cross each chunk by copy, sequentially (spec 16c's overlapped chunk pipeline is not used here); MTP in `b70-serve`
only, at a fixed K (`b70-decode --spec mtp` and `--draft-vocab` refused), the whole model refused in both CLIs
(spec 22), bf16 KV only (`--kv-cache int8` refused, decision 8), `reasoning_effort` not forwarded to the template (as
for Qwen3.8),
no `--profile`, context capped at the trained 262144.

**Relevant variables:** `B70_Q4_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_Q4_MTP_NORM` (`single` |
`per_stream`),
`B70_Q4_TIE_TOL` / `B70_Q4_SEL_TOL` (the golden test's routing and selection near-tie tolerances, 1e-3 proposed).

## Nemotron 3.5 Bolt 74B-A9B (spec 25)

NVIDIA's hybrid of 40 Mamba-2 blocks, 8 NoPE attention blocks (GQA 32 / 2, head_dim 128) and 40 latent MoE blocks (top-9
of 240..496 relu² experts in a 1024 latent, a shared expert), 88 blocks, vocabulary 131072; served from our own int4 g64 export
`urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ` @ `8932aa8d` (AutoRound, GPTQ packing; 41 GB on disk). The
loader also reads the GPTQ g128 spelling through its exact g64 expansion (the synthetic checkpoints test it).

**Status: decode, prefill and serving built, parked until Flash-Next is done; no speed row yet** (spec 25c / 25d / 25e, as built in
spec 25 §15 / §16 / §17). The real model runs on **two cards** (two cards are its `--parallel auto`, split 48
by bytes; `--parallel 1` is refused naming the bytes); the N 12 synthetic checkpoints and `--layers N` fit one card. Without `--prefill`
a prompt goes through the decode list one replay per id; `--prefill` / `--bench --prefill-length N` run the `l0` prefill
(spec 25d). Expected (derived): 653 launches a token (645 with `B70_BOLT_ATTN=eager`, 533 with `B70_BOLT_FUSE=all`, +2 with
`--pipeline-handoff peer`). P0 (`probe_bolt_step`) runs first on the box and decides the GEMV `{S, layout}` picks and which
arms are worth keeping.

```sh
S=oracle-out-bolt-synth/n12-g128                   # a synthetic checkpoint (25b's make_synth.py, 12 blocks)
P=tests/golden/prompts/bolt_short.ids

# every command below: no speed row yet
./build/tools/probe/probe_bolt_step --depth 4096 --rounds 9 --sweep-s --pair      # P0: the skeleton lists (no model)
./build/src/cli/b70-decode $S --ids $P --n 32                                     # one card, int8 head
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --ids $P --n 32          # two cards (the default for 88 blocks)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256 --max-len 8192
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --profile --depth 4096 --steps 32 --repeats 5
B70_BOLT_FUSE=all ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # 533 launches
B70_BOLT_SSM=bf16 ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # the bf16 state arm
```

**Serving (spec 25e).** `b70-serve <snapshot>` runs the engine on **two cards by default** (`--parallel 1` only for a synthetic /
small checkpoint; `--pipeline-split`, `--pipeline-handoff` as `b70-decode` takes them), `--max-len auto` up to 262144 with
the prefill scratch planned in, the int8 head by default, prefill on `l0` only:

```sh
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --port 8000 --prefix-cache-gb auto --log-requests ~/bolt-log
```

- **Chat template.** The snapshot's `chat_template.jinja` (sha256 `6521afc7...`, NVIDIA's, byte-identical in our
  export; its sha256 is printed and compared in the startup line). ChatML, no BOS, and the system block is ALWAYS
  rendered (`<|im_start|>system\n<|im_end|>\n` with no system message). `chat_template_kwargs`: `enable_thinking` (default
  true: the prompt ends `<think>\n`; false: `<think></think>`), `medium_effort` (appends `{reasoning effort: efficient}` to
  the last user message), `truncate_history_thinking` (default true: the reasoning of turns before the last user message is
  dropped), each a boolean (a 400 otherwise); the OpenAI `reasoning_effort` maps onto them - `none` / `minimal` ->
  `enable_thinking` false, `low` / `medium` -> `medium_effort` true, `high` / `xhigh` -> the default, anything else a 400 -,
  and an explicit kwarg wins. Tool schemas and call arguments render with sorted keys (the server's JSON objects are
  sorted: the prefix cache misses at a call whose arguments the model wrote in another order). Python's spelling for a
  history argument (`True`, `None`) is the template's own.
- **Output.** The output opens inside reasoning when the prompt ends `<think>\n` (`reasoning_content` in the response,
  VERBATIM - the template re-adds `<think>\n` + reasoning + `</think>`, so a stripped reasoning would drift every later
  prompt), up to the first `</think>`; then content and Qwen3-Coder XML tool calls (`<tool_call>\n<function=NAME>\n<parameter=
  KEY>\nVALUE\n</parameter>...`, values typed by the tool's schema first). With thinking off everything is content, a stray
  `</think>` included. `chat_template_kwargs.force_nonempty_content: true` turns reasoning that never closed (or ended with
  no content) into content - held back from the stream until decided. A call cut by `max_tokens` or EOS is dropped, never
  leaked as content; a call inside reasoning is reasoning (the `ultra_v3` reasoning parser's rules; spec 25 §7 and §17).
- **Sampling.** A request that names no `temperature` / `top_p` / `top_k` samples with NVIDIA's published T 1.0, top-p 0.95,
  no top-k, and generation ends on EOS **2 or 11** - from the snapshot's `generation_config.json` when it carries sampling
  keys (ours does: NVIDIA's file), else the published values. A seeded
  sampled request is reproducible request by request; `temperature: 0` is the device argmax.
- **Prefix cache.** A snapshot at an aligned 2048-position block end (and at a request's end) carries, per Mamba block, the
  SSM state (fp32: 4 MiB; 2 MiB with `B70_BOLT_SSM=bf16`, which is another store key) and the three raw conv columns
  before the position - 170,229,760 B in all (derived) - and the 8 attention blocks' KV, 8,192 B a position (one state per
  block end: a 64k-token prompt writes 32 of them, ~6.0 GB with the KV, derived - size `--prefix-cache-gb` for it). A hit restores
  the largest block end <= the common prefix and prefills the rest. A new user message re-renders every earlier assistant
  turn without its reasoning (`truncate_history_thinking`), so the cache diverges at the first assistant turn before the
  last user message; within one user turn (the agentic tool loop) the re-render equals what was generated.
- **Request.** Image / video / audio content parts are refused (text only).

**Speculative decoding (spec 26c; not yet run on a card).** Bolt has no MTP head, but
its decode list also runs as a *verify* pass of M = K + 1 <= 8 rows (each row bitwise a plain decode step, so greedy output
is the plain run's), and two proposers feed it on `b70-serve`:

- `--spec lookup` - prompt lookup (the continuation of the longest earlier repeat in the request, no extra model);
- `--spec dflash` - NVIDIA's LiLiCorr draft model (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`, 918 MiB, from
  the HF cache; `--draft-model PATH` points elsewhere) on the last card beside the main model's head: it reads six of Bolt's
  hidden-state rows ("taps") and drafts a block of up to 7 ids in one pass.

`--spec-max 1..7` caps the guesses (default 7); K is picked per request from how often its guesses were right. Everything
else is the engine's: the draft model's linears int8 per row (exact bf16 and int4 g64 are test arms), its draft distribution
vLLM's (the candidate lattice), the verify's experts read once per block for all rows (the union form; the plain M-row form
is its bitwise test reference). Bolt's `--spec auto` stays **off** until row 65's D5 measures `dflash` >= 10 % faster than
both `off` and `lookup` on the agentic set (and nowhere > 3 % slower). Two cards cross by the copy hand-off only
(`--pipeline-handoff peer` is refused with a proposer). `b70-decode` decodes Bolt plainly (`--spec` there is refused).

```sh
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --spec dflash      # the LiLiCorr drafter, K <= 7 adaptive
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <snapshot> --spec lookup --spec-max 4
```

**Limits.** No `--spec mtp` (no MTP head) and no `--spec dflash-lookup` (Qwen3.8's), no `--draft-vocab`, bf16 KV only
(`--kv-cache int8` refused), prefill on `l0` only (`--prefill-backend l0-int8` refused), context at most
262144, `--layers` is `b70-decode`'s development mode only.

**Relevant variables:** `B70_BOLT_ATTN` (`flash` | `eager`), `B70_BOLT_FUSE` (`none` | `all` | a comma list of `router`,
`route`, `conv`), `B70_BOLT_SSM` (`fp32` | `bf16`), `B70_BOLT_TIE_TOL` (the golden gate's near-tie tolerance, 1e-3 proposed),
`B70_BOLT_A_EXP` (the reference's `A`: `torch` | `engine`), `B70_BOLT_SNAPSHOT` (`template_bolt_test`'s,
`bolt_tokenizer_test`'s and `bolt_request_test`'s snapshot).
