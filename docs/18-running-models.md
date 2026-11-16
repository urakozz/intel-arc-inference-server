# Running the models

Recipes first for Qwen3.8-27B, the model everything is tuned for; the other models follow. Every flag is explained
in the [README](../README.md#flags). Generation speeds are MTP speculative decoding (prompt lookup on) on the golden
code / prose / tool-call prompts, 256 generated tokens, several requests as the total; prefill speeds are the prompt pass
alone, a 4,096-token prompt (BENCHMARKS, "The quiet-box session"). Context lengths are the memory planner's, per card,
with the MTP head loaded.

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
```

`--prefix-split-last` is in every agentic recipe: with chat clients that drop the reasoning (opencode) it makes
each next turn a prefix cache hit exactly where it diverges. Leave it off for benchmarks.

## Qwen3.8-27B on one card

| recipe | context | speed |
|---|---:|---|
| default, bf16 KV cache | 171,008 | 138 / 57 / 100 t/s (code / prose / tool calls), prefill 1,788 t/s |
| full context: int8 KV cache | **262,144** | no current row |
| faster prefill, opt-in: int8 prompt pass, bf16 cache | 171,008 | prefill 2,083 t/s instead of 1,788 (4k prompt) |
| several requests: `--max-num-seqs 4` | 46,080 + 3 x 32,768 (bf16), 167,936 + 3 x 32,768 (int8) | no current row |

```sh
./build/src/cli/b70-serve $Q --parallel 1 --prefix-split-last                                 # default
./build/src/cli/b70-serve $Q --parallel 1 --kv-cache int8 --prefix-split-last                 # full context
./build/src/cli/b70-serve $Q --parallel 1 --prefill-backend l0-int8 --prefix-split-last       # faster prefill (opt-in)
./build/src/cli/b70-serve $Q --parallel 1 --max-num-seqs 4 --kv-cache int8 --prefix-split-last # 4 requests
```

On a machine with one card `--parallel 1` can be left out. The int8 prompt pass and the int8 cache are not combined:
together they round too much (about 3 % of tool calls that copy exact text from a long prompt come out wrong).

## Qwen3.8-27B on two cards

| recipe | mode | context | speed |
|---|---|---:|---|
| one session, fastest (the default) | `tp`: every layer split over both cards; needs peer-to-peer between them | **262,144** (bf16) | **123 / 74 / 115 t/s** (code / prose / tool calls), prefill 2,931 t/s, 3,217 with `--prefill-backend l0-int8` (4k prompt) |
| two sessions at full context: `--max-num-seqs 2` | `replica`: a model copy per card, one request each | 171,008 (bf16) or **262,144** (int8) each | 164 / 108 / 144 t/s total |
| most requests at once: `--max-num-seqs 8` | `replica`, 4 slots per card | 46,080 + 3 x 32,768 (bf16) or 167,936 + 3 x 32,768 (int8) per card | **342 / 300 / 233 t/s** total at 8 requests (282 / 189 / 206 at 4) |

```sh
./build/src/cli/b70-serve $Q --prefix-split-last                                    # one session, tp
./build/src/cli/b70-serve $Q --max-num-seqs 2 --kv-cache int8 --prefix-split-last   # two sessions, full context each
./build/src/cli/b70-serve $Q --max-num-seqs 8 --kv-cache int8 --prefix-split-last   # up to 8 requests
```

With `--max-num-seqs N`, slot 0 is the largest context that fits beside N - 1 slots of 32,768; on two cards the
slots are split between the cards (card 0 takes the odd one). An explicit `--max-len` that does not fit beside the
other slots is refused at startup with the numbers. `tp` runs one request at a time; add `--prefill-backend l0-int8` for the faster prompt pass (3,217 instead of 2,931 t/s at 4k).
Pipeline parallel (`--parallel pp`, the layers split between the cards) is for models that do not fit one card -
Qwen3.8 fits one card even at full context, so it gains nothing there.

## Qwen3.8-27B: benchmarks and greedy runs

```sh
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256           # prefill 4096, then decode 256
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 32 --parallel tp
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --spec mtp > out.ids
./build/src/cli/b70-decode $Q --profile --depth 4096 --steps 32 --repeats 5    # per-launch decode timing
```

`b70-decode` defaults to one card, `--max-len 16384` and no speculative decoding so its rows stay comparable.

## Which models, and what each supports

How a change reaches `main`: it is merged after the Mac checks, and the full box validation
(`tools/box_validate.sh`: golden sets, serving, prefix cache, speculative decoding) runs on `main` on
demand. A test that passes proves a configuration **runs and is correct**; it is not a speed row. What
is not finished is listed in [open-work.md](open-work.md).

- **Qwen3.8-27B** is the model in focus, with the full set of measured rows (prefill, decode, depth,
  prefix cache, MTP, DFlash, tensor parallel).
- **Ornith 1.5** is supported and its box tests pass; its speed rows (decode, prefill, MTP cost table) are in BENCHMARKS.
- **Qwen3.8-Flash-Next** is next. It runs only truncated (`--layers N`) because the whole model does not
  fit two cards; open-work.md has the plan for the full size.
- **Kolibri-1** is supported by the code and parked (out of focus); no speed rows.
- Dropped and no longer supported: LFM2.5, Muse-Glimmer and Agnes 3.0 Flash (a `config.json` naming
  `AgnesForConditionalGeneration` is refused by name). Flint and Nanbeige are parked and not built.

| | Qwen3.8-27B | Ornith 1.5 35B-A3B | Kolibri-1 | Qwen3.8-Flash-Next |
|---|---|---|---|---|
| checkpoint | published (ours, int4 g64) | published (ours) | published (ours, int4 g64) | Intel's interim W4A16 (181 GB); our own is not made |
| `b70-serve` | yes | yes | yes, **two cards** by default | `--layers N` only (the whole model is refused until the expert tier is built) |
| `--parallel tp` | yes | refused | refused | refused |
| `replica`, `--max-num-seqs` | yes | `replica` at `--max-num-seqs 2` (one request a card); more refused | refused | refused |
| `--parallel pp` (two cards) | yes | yes | its default (its own engine) | yes, `--layers N` |
| `b70-decode` | yes | yes | decode and prefill, **two cards** (the default) | decode and prefill, **`--layers N` required** |
| speed rows in BENCHMARKS | yes | yes | no | no |
| MTP head (`--spec mtp`; `--spec auto` = adaptive MTP) | yes | yes (MoE head) | none (`auto` = off) | `b70-serve --spec mtp`, a fixed number of guesses = `--spec-max` 1..3 (`auto` = off until the whole model is measured); `b70-decode` refuses it |
| `--spec lookup` (developer value) | yes | yes (needs the head) | no | yes (needs the head) |
| `--spec dflash` | yes, when asked: one card (with prompt lookup inside), up to 4 requests a card (`--max-num-seqs`), `replica`, `tp` or `pp`, bf16 or int8 KV | no | refused | refused |
| `--kv-cache int8` | yes | not built | refused (bf16 only) | refused (bf16 only) |
| prefill backends | `l0` (default), `l0-int8` | `l0` (default, use it); `l0-int8` opt-in and slower here, accuracy gates diagnostic (MoE routing) | `l0` only | `l0` only |
| KV per position (bf16) | 64 KiB | 20 KiB | 20 KiB (+ 335.5 MB of sliding rings) | 24.75 KiB (12 QSA layers: K + V 2 KiB + 64 B of indexer keys each) |
| `--max-len auto`, one card, int8 head (derived) | 171008 with MTP (the default); ~201k with `--spec off` | 262144, with or without MTP (measured by the operator, bf16 KV, 27.3 GB planned of 32.5) | 262144 on two cards (split 25) | at `--layers 19` (Intel's forms, one card) 78336 decode-only; at `--layers 18` 70144 with the prefill planned |
| with `--kv-cache int8` (derived) | 262144, with or without MTP | - | - | - |

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

**Benchmarking.** The llama-benchy command against a running server (no
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
The hidden state crosses from the first card to the second by a device-to-device copy (there is no
option for it). One answer is not faster than on one card; each card holds about half the model, so `--max-len auto`
reaches the trained 262144 tokens, with MTP too (derived, int8 head). The MTP head runs on the second
card with its own copy of the embedding (2.54 GB). Kolibri-1 runs on two
cards by default in `b70-decode` (spec 20c / 20d) and `b70-serve` (spec 20e, its own engine).

```sh
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
# an agentic session over both cards as pipeline parallel: 262144 tokens with MTP and the int8 cache
./build/src/cli/b70-serve $Q --kv-cache int8 --host 0.0.0.0 --port 8000 --parallel pp --prefix-split-last
# b70-decode: the same ids as one card, greedy, with MTP across the split
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --parallel pp --spec mtp
```

Startup prints the split (`split: auto -> N`), the plan per card and both cards' memory lines (and
`parallel: auto -> ...` when `auto` chose).

## Speculative decoding and how its output is judged

A *proposer* guesses the next few tokens and the main model checks them all in one pass (a *verify* step of
several rows), so every correct guess is a token almost for free. `--spec` picks the proposer:

| `--spec` | what proposes | where |
|---|---|---|
| `auto` (the default) | the model's own MTP head where it has one (Qwen3.8, Ornith); nothing elsewhere | every run mode |
| `mtp` | the MTP head, up to 3 tokens a step | every run mode |
| `dflash` | DFlash, a small draft model, up to 7 tokens a step (Qwen3.8) | one card, `replica`, `tp`, `pp`; needs `--draft-model` |
| `off` | nothing | |

`auto` is the MTP head because the head costs no extra memory, so the full context stays available. DFlash runs
only when asked: its weights (1.21 GB) come out of the context's memory. The engine never downloads, so a
missing drafter is an error.

**Prompt lookup** runs inside both proposers on one card with one request at a time, and inside MTP on each card
of `--parallel replica` with one request a card. Where the answer repeats 8
or more tokens of earlier text, the tokens that followed that text last time are checked instead of the
proposer's guesses: up to 3 tokens inside MTP, up to 7 inside DFlash (there the repeat must lie outside the
drafter's own 2,048-token window). Elsewhere the proposer drafts as usual. On one card inside MTP a long
copied run is checked in lists of up to 64 rows (16, 32 or 64, chosen as the copy goes): pure copy 636-650 t/s
against 356-364 at 16 rows, edit/copy prompts 1.173x the 16-row form, output ids identical. It is on by default;
`B70_SPEC_LOOKUP=0` serves plain MTP or plain DFlash for comparisons. (`--spec lookup` and
`--spec dflash-lookup` remain as developer values.)

**How the output is judged.** The verify rows run other kernels than a plain step (the small-batch GEMM
`gemm_sm` instead of the one-row GEMV, and a DPAS attention kernel at depth), and those round differently. So
speculative output is not compared bit for bit with plain decode. It is graded by the **f32 judge**
(`tests/server/spec_judge.h`, [14-golden-gate.md](14-golden-gate.md)): a speculative stream passes when it is
no further from a float32 reference than plain decode on the same context, or when every miss is a near-miss
(within 8 bf16 ulps of the reference's top choice). Plain code paths are still checked bitwise.

## Qwen3.8-27B with the DFlash draft model

DFlash drafts a block of up to 7 tokens per step from Qwen3.8's own hidden states (the outputs of layers 5, 19,
33, 47 and 61). `--spec dflash` needs an explicit `--draft-model`; without it `b70-serve` and `b70-decode`
refuse and name the two choices:

| `--draft-model` | size | notes |
|---|---:|---|
| `urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound` | 1.21 GB | our int4 export, loaded as shipped. The recommended one: draft pass 4.41 ms at a 4,096-token context |
| `z-lab/Qwen3.8-27B-DFlash2` | 2.05 GB | the bf16 original, quantised to int8 at load: draft pass 6.20 ms |

The int4 drafter accepts slightly fewer tokens a round than the int8 one (0.016 fewer, pooled over the test
prompts) and is faster overall.

```sh
# the packed drafter (once)
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound
D=urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound

# one card, with prompt lookup inside
./build/src/cli/b70-serve $Q --parallel 1 --prefix-split-last --spec dflash --draft-model $D
# one card with the int8 KV cache
./build/src/cli/b70-serve $Q --parallel 1 --spec dflash --kv-cache int8 --draft-model $D
# one card, up to four requests at once (the int8 KV cache: 135,168 + 3 x 32,768 positions)
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

- **Speed** (code / prose / tool calls, t/s): one card, one request 164 / 65 / 111; two requests on two cards
  (`replica`) 317 / 127 / 191 in total.
- **Which mode.** DFlash is fastest on one card. Under `--parallel tp` it works but is not faster than on one
  card for one session, and the server prints a note at startup recommending one card, `replica` or MTP on two
  cards.
- **Context.** With the packed drafter and the bf16 KV cache, `--max-len auto` gives 158,208 tokens on one
  card (145,664 with the int8 drafter). With `--kv-cache int8` the memory planner gives the full 262,144
  (254,976 with the int8 drafter).
- **Several requests on a card** (`--max-num-seqs` up to 4 a card): each request keeps its own drafter context;
  a request alone drafts up to 7 tokens as on one slot, several share one draft pass of 8 rows (3 tokens each for
  two, 1 for four). Against `--spec mtp` in the same mode (one card, int8 KV): 1.15x with one request, 1.12x with
  two, 0.97x with four. Slot 0 is 135,168 positions beside three of 32,768 (the drafter takes the difference).
- **Refused, by name:** `--spec dflash-lookup` and `--draft-vocab` beside several requests on a card, more than
  four requests a card, and every other model.
- **The prefix cache** keeps the drafter's context (41.9 MB) with every snapshot, so a conversation that gets a
  prefix cache hit at the end of a prompt or of a request drafts exactly as an unbroken one. A hit at a
  snapshot inside a long prompt drafts worse (never differently) until 2,048 new positions have refilled the
  drafter's window.

How the drafter's acceptance was measured, and how to repeat it: `tools/oracle/README.md`, "The DFlash P0", and
[the probe record](probes/probe-dflash-2026-10-08.md).

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

**Status: built and measured on one card** (README, BENCHMARKS): answers 312 / 326 / 264 t/s (code / prose /
tool calls) with MTP; prompts read at 9,362 t/s at 4k and 7,682 at 16k (the default bf16 prompt pass). ~1.86 GB
read per token (derived), so the bandwidth ceiling is far above Qwen3.8's (~270 t/s), and launches,
not bytes, set the decode rate; 526 launches per token.
Weights 19.45 GB with the int8 head (derived), +0.50 GB for the MTP head. For comparison,
the checkpoint's model card reports vLLM XPU on a B70 at 99.8 t/s decode (tg256, c1) and
7090 t/s prefill (pp4096, c1) - not measured here.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
uvx --from huggingface_hub hf download $O                 # 22.99 GB

# agentic session: the full trained context, 262144, fits in bf16 with or without MTP (measured: 27.3 GB planned of 32.5)
./build/src/cli/b70-serve $O --host 0.0.0.0 --port 8000 --prefix-split-last
# two sessions at once on two cards (replica: a model copy per card, one request each, full context each)
./build/src/cli/b70-serve $O --host 0.0.0.0 --port 8000 --max-num-seqs 2 --prefix-split-last

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256 --prefill-backend l0
./build/src/cli/b70-decode $O --ids prompt.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $O --profile --depth 4096
```

**Limits.**

- Two cards serve two requests at once as `replica` (`--max-num-seqs 2`, or `--parallel replica`): each card
  runs its own engine with the full context, MTP and the prefix cache, and a prompt cached by one card is a
  prefix cache hit on the other. Several requests on one card (sequence slots) are built for Qwen3.8 only, so
  `--max-num-seqs` above 2, or 2 on one card, is refused. `--parallel tp` is Qwen3.8's. Each request is
  bitwise what it is alone on its card. Speed (code / prose / tool calls, t/s): two at a time 581 / 605 / 453 in total
  against 312 / 326 / 264 one at a time on one card.
- `--kv-cache int8` is not built at Ornith's attention shape (capture names the missing
  binary); it is not needed for context, since 262144 fits in bf16.
- Prefill runs on `l0` (the default) and `l0-int8` (opt-in). **Use the default on Ornith:** `l0-int8` is now
  slower than the bf16 prompt pass (0.92x at 4k, 0.95x at 16k, measured 2026-11-16). It also loses routing
  accuracy against f32 (the box, 2026-10-30: 1314 differing expert sets on the prefill path against the
  bf16 reference's 1140), and no 8-bit activation scheme studied won it back, so its accuracy gates are
  diagnostic - printed, not gated (operator ruling 2026-10-31). Its kernels'
  own gate (`prefill_gate_ornith_l0-int8`, judged by the quantisation-aware f32 reference) still binds.
- The adaptive MTP (`--spec auto`) decides with Qwen3.8's cost table, K up to 3. Ornith's own measured table (a
  verify of 2 / 3 / 4 rows costs 1.15 / 1.34 / 1.52 plain steps, a draft 0.19 each) decoded 6-10 % slower in the
  speed A/B, so it is recorded in BENCHMARKS and not used (the startup line says so). The head's bf16 experts are
  quantised to int4 at load (round to nearest), which can change acceptance, never output.
- The checkpoint's `tokenizer.json` defines Qwen3.8's 248077 ids, the count the greedy
  argmax masks from, so `b70-serve` prints no vocabulary note. It tokenises the committed
  `tests/golden/prompts` (prose, code, cjk) to the committed ids; its pre-tokenizer splits
  combining marks differently from the bf16 base checkpoint's file (`\p{M}`), which only
  text with combining marks sees.

**Relevant variables:** `B70_KV_CACHE` must stay `bf16`; `B70_PREFILL_PROFILE` applies.

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
./build/src/cli/b70-decode $K --ids k.ids --n 32 --pipeline-split 25
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
./build/src/cli/b70-serve $K --max-len 131072
./build/src/cli/b70-serve $K --prefix-split-last                # agentic sessions: the next turn is a prefix cache hit at len - 1
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Wie spät ist es in Berlin?"}], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

**Box checks:** `tools/probe/kolibri_passkey.sh` (passkey 3 / 3 at 262144 on two cards) and the
German tool-call set `tests/golden/toolcall-kolibri-de` (`tools/toolcall/engine_generate.sh`, then
`score.py --format hermes` against the bf16 source's reference, `oracle_generate.py --model kolibri`).

**Limits.** Prefill on `--prefill-backend l0` only (`l0-int8` refused: its h8 linears rotate in
1024-k Hadamard blocks and the hidden is 2560), `--prefill-chunk`
at most 2048; no `--spec mtp` / `lookup` (no head; its `--spec auto` is off - `--spec dflash` is the DSpark drafter below),
`--kv-cache int8` refused (20 KiB a position: bf16 only), no `--profile`, context capped at the trained
262144 (decision 3) - `b70-decode` and `b70-serve` alike. The
bench prompt is the first 42 ids of `de_prose.txt` through Kolibri's tokenizer (spec 20e).

**Relevant variables:** `B70_KOLIBRI_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_KOL_TIE_TOL`
(the golden test).

## Kolibri-1 with the DSpark draft model

**Status: built, not yet run on the cards.** `b70-serve` serves Kolibri-1 with speculative decoding from the community
DSpark drafter `intuit/Kolibri-1-dspark` (5 layers, hidden 2560, 826 MB bf16): the drafter reads Kolibri's residual stream
at the outputs of layers 3 / 13 / 23 / 33 / 43, drafts a block of up to 7 ids in one pass and Kolibri checks them in one
step of up to 8 rows. Greedy output is plain decoding's, token for token; a sampled request keeps the model's own
distribution (the lossless acceptance). It is what the DFlash drafter is for Qwen3.8, with three differences: each block row
predicts the NEXT position (the first draft comes from the anchor's own row), the block's attention is causal, and a small
Markov head (rank 256) corrects each row's candidates by the previously chosen id. The drafter's `config.json` picks all of
this - there is no extra flag.

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ
D=intuit/Kolibri-1-dspark                                  # in the HF cache; b70 never downloads it

# every command below: not yet run on the cards; needs two cards
./build/src/cli/b70-serve $K --spec dflash --draft-model $D                 # K per iteration from the acceptance, up to 7
./build/src/cli/b70-serve $K --spec dflash --draft-model $D --spec-max 4
```

What it costs: the drafter (bf16, ~1 GB with its buffers) and a copy of Kolibri's embedding (655 MB) on the second card,
the verify scratch on both; `--max-len auto` takes them out of the context it gives. The log's `dspark:` line after each
request gives the accepted drafts and the tokens per verify (the model card reports about 3 tokens a verify on chat and
about 4 on maths on vLLM).

**Limits.** `b70-serve` only (`b70-decode` decodes Kolibri plainly); `--spec lookup`, `dflash-lookup` and `mtp` are refused;
`B70_KOLIBRI_ATTN=eager` is refused with `--spec` (the verify lists are flash only); the drafter runs in bf16 (an int8 or
int4 drafter is a later step). The speed is unmeasured: the cost table that picks K is derived from the bytes a step reads,
and each verified row reads its own six experts a layer, so deep blocks cost more than on a dense model.

**Card tests (waiting for the first run):** `ctest -R 'kolibri_verify|kolibri_dspark'` - the verify lists against the plain
decode list (synthetic set and the real model), and the real model with the drafter (greedy equals plain, tokens a verify,
a seeded sampled request).

**To check on the first card run** (each marked `TODO(card)` in the code):
1. The load copies Kolibri's 655 MB embedding from card 0 to card 1 for the drafter (`cli/kolibri_spec.h`). If the first run
   stalls at load, look there first, and upload the replica from the host instead.
2. The cost table that picks K (`MtpCost::kolibri_dspark`) is derived from bytes: measure the verify at M = 1..8 and the
   draft pass.
3. `kol_attn_decode` spills a few registers at M 3 / 4 (ocloc's warning): time the attention's two halves of a deep verify.
4. Each verify row reads its own six experts a layer: price it, then decide on reading each distinct expert once.
5. The taps are the outputs of layers 3 / 13 / 23 / 33 / 43 (the drafter's ids minus one, vLLM's rule): a low acceptance
   points here first.
6. The Markov walk looks at the head's top 16 a row, not the whole vocabulary: measure the acceptance gap.
7. The prompt's last 2048 positions reach the drafter 8 rows a replay (256 replays): time it.
8. `kolibri_dspark_test`'s bar (1.2 tokens a verify) is a placeholder: set it from the run.

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
