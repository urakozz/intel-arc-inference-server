# Running the models

How to download, serve, decode and benchmark each model the engine knows, what each
one supports today, and which of those configurations have actually run on a B70.
Flags and environment variables are explained in the project README's tables; this
page only says which ones matter for which model.

Every number is **measured** (on the card) or **derived** (from the memory planner or
byte counts, never run), as in the specs.

## Read this first: what "measured" means here

Most of what is on `main` was written while the box was unavailable and merged
**blind**: built and tested on the Mac (host tests, Level Zero and OpenCL syntax
checks, a few kernels on the Mac's own GPU), never run on a B70. The queue of what still
has to run is [superpowers/plans/box-validation-queue.md](superpowers/plans/box-validation-queue.md).

- **Measured** below means the configuration ran on the card at or before `b32aaaf`, the
  last commit before the blind work. `main` is expected to give the same bits for
  Qwen3.8 (the descriptor refactor and the new models only add code paths); checking that
  is the queue's first step (G0) and has not happened yet.
- Commands marked `# never run on a B70` have not run on the card at all. They are
  written to work, and host-tested, but the first run may fail.

| | Qwen3.8-27B | Agnes 3.0 Flash | Ornith 1.5 35B-A3B | K2-Horizon MoVA-36B-A4B | Kolibri-1 (spec 20) | Qwen3.8-Flash-Next (spec 21) |
|---|---|---|---|---|---|---|
| checkpoint | published | published | published (spec 15 §13) | published | **not made** (spec 20b) | Intel's interim W4A16 (181 GB); ours not made (21q) |
| `b70-serve` | yes | yes | yes | yes (spec 18d, box gates pending) | yes (spec 20e, box gates pending; needs 20b), **two cards** by default | `--layers N` only (spec 21e; the whole model refused naming spec 22), never run |
| `b70-serve --pp 2` (two cards, spec 16d) | yes, never run | yes, never run | yes, never run | **refused** (one card) | its default (its own engine, spec 20e), never run | yes, `--layers N` (spec 21e), never run |
| `b70-decode` | yes | yes | yes | yes | decode and prefill (spec 20d), **two cards** (`--pp 2`, the default) | decode (spec 21c) and prefill (spec 21d), **`--layers N` required** (the whole model needs spec 22), `--pp 1` or `2`, never run |
| ran on a B70 | yes (see above) | no | no | no | no | no |
| MTP head (`--mtp`) | yes | yes | yes (MoE head) | none | none | `b70-serve --mtp 1..3` (spec 21e; `auto` refused until spec 22's numbers), never run; `b70-decode` refuses it |
| `--spec lookup` | yes | yes (needs the head) | yes (needs the head) | no | no | yes (needs the head; spec 21e), never run |
| `--spec dflash` (spec 19c / 19d) | yes, one card or `--pp 2` (copy hand-off), bf16 KV, never run | no drafter (refused) | no (its v1 drafter is a later stage) | no drafter exists (refused) | refused | refused |
| `--kv-cache int8` | yes, gates pending | yes, gates pending | not built | yes, gates pending (spec 18e) | refused (bf16 only) | refused (decision 8: bf16 first) |
| prefill backends | `l0-int8` (default), `l0`, `sycl-tla` | `l0-int8`, `l0` | `l0-int8`, `l0` (`sycl-tla` refused) | `l0` only | `l0` only (spec 20d) | `l0` only (spec 21d; `l0-int8` and `sycl-tla` refused) |
| KV per position (bf16) | 64 KiB | 72 KiB | 20 KiB | 192 KiB | 20 KiB (+ 335.5 MB of sliding rings) | 24.75 KiB (12 QSA layers: K + V 2 KiB + 64 B of indexer keys each) |
| `--max-len auto`, int8 head (derived) | ~201k; ~170k with `--mtp` | ~140k; ~114k with `--mtp` | 262144, with or without `--mtp` | ~46k served (the prefill planned); `b70-decode` ~50k decode-only | 262144 on two cards (split 25), the prefill planned or not | at `--layers 19` (Intel's forms, one card) 78336 decode-only; at `--layers 18` 70144 with the prefill planned |
| with `--kv-cache int8` (derived) | 262144, with or without `--mtp` | 262144; ~226k with `--mtp` | - | ~90k served; `b70-decode` ~98k decode-only | - | - |

The `--max-len auto` lengths are the planner's at the default 1.5 GB reserve on the
32.53 GB card; the reserve itself is an estimate the box has not confirmed, and nothing
above 131072 has run on the card. Both CLIs print the length they chose and the memory
plan at startup.

## Common to every model

**Build** as in the README (`cmake`, then `ctest`). The engine never downloads; fetch a
checkpoint into the Hugging Face cache once:

```sh
uvx --from huggingface_hub hf download <org>/<name>
```

Both CLIs then take the repo id (looked up under `$HF_HOME`, default
`~/.cache/huggingface`) or a snapshot directory. The model is chosen from the
checkpoint's `config.json`; there is no model flag.

**Which card.** `--device N`, else `ONEAPI_DEVICE_SELECTOR=level_zero:N`, else device 0.
`ZE_AFFINITY_MASK=N` hides the other cards from the process instead.

**One model per server process.** The prefix cache (`--prefix-cache-gb`, system RAM, on
by default) belongs to that process and that model.

**Connecting a client.** `b70-serve` is OpenAI compatible: point the client (opencode or
any other) at `http://<host>:8000/v1` and use the `--served-name` as the model name. For
agentic sessions with a thinking model, add `--prefix-split-last` so each turn restores
exactly where the next one diverges (README, `b70-serve` flags).

**Benchmarking.** The command behind the vLLM comparison rows, against a running
server (no `--prefix-split-last`; it is for sessions, not benchmarks):

```sh
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model <served-name> \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

Both its `tg` and its `pp` figures read the same way as vLLM's: since 2026-09-27 the server
sends the role frame with the first token, after prefill, as vLLM does (docs/BENCHMARKS.md,
"llama-benchy's own pp figure against our server"). Its `pp` is HTTP-inclusive;
`b70-decode --bench --prefill-length` gives the device-side prefill time. The same-flags
comparison against vLLM is queue row 41 (docs/BENCHMARKS.md, "llama-benchy comparisons").

**Token ids for `b70-decode --ids`** are whitespace-separated, made with the checkpoint's
own tokenizer (needs `transformers`; `<snapshot>` is the snapshot directory in the cache):

```sh
python3 tools/oracle/tokenize.py <snapshot> encode prompt.txt > prompt.ids
python3 tools/oracle/tokenize.py <snapshot> decode $(cat out.ids)
```

`tests/golden/prompts/{prose,code,cjk}.ids` are ready-made ids for the Qwen tokenizer
(Qwen3.8 and Agnes share it).

**Two cards (`--pp 2`, spec 16).** Qwen3.8, Agnes and Ornith run with their layers split over
GPUs 0 and 1 of what `ZE_AFFINITY_MASK` shows: `b70-decode --pp 2` (decode, spec 16b; the
two-card prefill with `--prefill`, spec 16c; `--mtp`, spec 16d) and `b70-serve --pp 2` (spec
16d, everything the server does on one card). One answer is not faster on two cards; each card
holds about half the model, so `--max-len auto` reaches the trained 262144 tokens on every one
of them (derived, int8 head; with `--mtp` too), and a long prompt prefills ~1.8x faster
(derived). The MTP head runs on the second card with its own copy of the embedding (2.54 GB).
K2-Horizon is one card only; Kolibri-1 runs on two cards by default in `b70-decode` (spec 20c / 20d)
and `b70-serve` (spec 20e, its own engine). **None of this has run on the cards yet** (queue rows 22, 23,
27, 28):

```sh
# every command below: never run on a B70
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
# an agentic session over both cards: 262144 tokens with MTP (derived; one card: ~170k)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --host 0.0.0.0 --port 8000 --served-name qwen \
    --pp 2 --prefix-split-last --mtp auto
# the same with the int8 KV cache and the peer hand-off (both cards ~16 GB, derived)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --served-name qwen --pp 2 --mtp auto \
    --kv-cache int8 --pipeline-handoff peer
# Agnes over two cards: 262144 instead of ~140k (derived)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ \
    --served-name agnes --pp 2 --prefix-split-last --mtp auto
# b70-decode: the same ids as one card, greedy, with MTP across the split
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 \
    --prefill --lm-head int8 --pp 2 --mtp 3
```

Startup prints the split (`split: auto -> N`), the plan per card and both cards' memory lines.
Not with `--device`, `--prefill-backend sycl-tla` or `B70_PREFILL_ATTN=composed`.

## Qwen3.8-27B

`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: the model everything is tuned for. 64
layers (48 delta net, 16 attention), int4 g64, bf16 `lm_head` and MTP head in the file.

**On the card (measured):** decode (29.45 t/s at depth 4096 with the bf16 head, 31.21 t/s
with the int8 head; 20.1 t/s at 128k), prefill (2125.12 t/s at 4096), the 131072
context (28.1 GB) with passkey at 5 / 50 / 95 % of 120k, the prefix cache and
`--prefix-split-last` (a turn at 60k of history in 1.23 s), the int8 `lm_head` (serving
default), decode attention v2, `--mtp 1|2|3` at `--max-len 16384`, the golden gates and
the tool-call set.

**Never run on the card:** `--max-len auto` (the `b70-serve` default; row 7),
`--prefix-cache-gb auto` (the default; row 9), `--mtp auto` (row 2), MTP above 16384
(row 6), `b70-decode --mtp` (row 6), `--draft-vocab` (row 8), `--kv-cache int8` (row 11),
`--spec lookup` (row 12).

```sh
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ
Q=urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

# agentic session, the configuration that has run on the card: 131072 context, an explicit
# prefix cache size (spec 7 measured 32 GiB; pick what your RAM allows), int8 head (default)
./build/src/cli/b70-serve $Q --host 0.0.0.0 --port 8000 --served-name qwen3.8 \
    --max-len 131072 --prefix-cache-gb 32 --prefix-split-last

# the intended default for agentic use: the largest context that fits (~201k, derived),
# the cache sized from the machine's RAM, MTP choosing 0-3 guesses per step (~170k)
./build/src/cli/b70-serve $Q --served-name qwen3.8 --prefix-split-last --mtp auto    # never run on a B70

# the full trained context, 262144, with the int8 KV cache (accuracy gates pending)
./build/src/cli/b70-serve $Q --served-name qwen3.8 --prefix-split-last --mtp auto \
    --kv-cache int8                                                                     # never run on a B70

# prompt lookup instead of the MTP guesses (one proposer per server)
./build/src/cli/b70-serve $Q --served-name qwen3.8 --prefix-split-last --spec lookup  # never run on a B70
```

```sh
# the BENCHMARKS.md rows: prefill 4096, then decode 256 at that depth (bf16 head, byte-matched with vLLM)
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $Q --bench --prefill-length 4096 --tg 256 --lm-head int8        # the serving head
./build/src/cli/b70-decode $Q --bench --depth 4096 --tg 256 --max-len 131072  # decode at a deep cache

# greedy ids in, ids out
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/prose.ids --n 64 --prefill --mtp auto  # never run on a B70

# per-launch anatomy of the decode step
./build/src/cli/b70-decode $Q --profile --depth 4096 --steps 32 --repeats 5
```

**Limits.** The trained context is 262144; with the bf16 KV cache the card holds ~201k
(derived) and 131072 is the largest length measured. `--kv-cache int8` needs the `l0` or
`l0-int8` prefill and decode attention v2. MTP above 16384 and the `--mtp auto` cost table
for the int8 head are derived, not measured. `b70-decode` keeps `--max-len 16384` and the
bf16 head by default so its rows stay comparable.

**Relevant variables:** `B70_KV_CACHE`, `B70_DECODE_ATTN` (`v1` only for A/B against the old
pair), the `B70_PREFILL_*` switches for prefill experiments, `B70_GIT_SHA` for bench rows.

### DFlash draft model (spec 19) - built, not yet run on the card

`z-lab/Qwen3.8-27B-DFlash2` drafts a block of up to 7 tokens per step from Qwen3.8's own hidden
states (the outputs of layers 5 / 19 / 33 / 47 / 61). Its acceptance on our prompts is measured on the Mac CPU (spec 19a
P0, `docs/probe-dflash-2026-10-08.md`: 7.21 tokens per verify at K = 7 on agentic tool calls, 6.15 on code, ~2.2 on
prose; int8 and 4-bit drafters equal bf16). The engine runs it since spec 19b-19d - **written on the Mac, nothing of it
has run on a B70** (box queue rows 54, 55, 56):

```sh
# one card: the drafter beside the target (int8 at load, 2.05 GB + 1.06 GB of verify slots), K per request (auto, <= 7)
./build/src/cli/b70-serve $Q --served-name qwen3.8 --prefix-split-last \
    --spec dflash --draft-model z-lab/Qwen3.8-27B-DFlash2                 # never run on a B70
# two cards: the drafter on the second card beside lm_head (the copy hand-off only)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $Q --served-name qwen3.8 --pp 2 \
    --spec dflash                                                         # never run on a B70
# greedy at a fixed K without the server (one card)
./build/src/cli/b70-decode $Q --ids tests/golden/prompts/code.ids --n 256 --prefill --lm-head int8 \
    --spec dflash --spec-max 7                                            # never run on a B70
```

- **Defaults (spec 19 §12, chosen from P0):** `--draft-format int8` (P0: int8 = bf16 acceptance at half the bytes),
  `--spec-max 7` (the drafter's block; P0's acceptance still rises at K = 7 on tool calls and code, and the per-request
  policy drops K on prose), `--draft-vocab off` (P0's free-looking ranked 32k was ranked in-sample; by id it loses up to
  1.4 tokens at K = 7 - r56.d5_dv measures it). `--spec` stays unset (= `--mtp`'s meaning): `dflash` becomes Qwen3.8's
  default only if it beats `--mtp auto` by >= 10 % on the opencode recording (D5, r56.d5_opencode).
- **Refused, by name:** `--kv-cache int8` (kv8.cl is built for M <= 4), `B70_DECODE_ATTN=v1`, `--prefill-backend
  sycl-tla`, `B70_PREFILL_REPLAY=1`, `--pp 2 --pipeline-handoff peer`, `b70-decode --pp 2`, every other model.
- **The prefix cache** keeps the drafter's ring (41.9 MB) with every snapshot (spec 19 decision 5 as ruled in 19d), so a
  conversation restored from a request's prompt-end or request-end snapshot drafts exactly as an unbroken one; the store
  is keyed by the drafter. **Caveat:** a snapshot at a block end inside a long prefill holds the ring as that prefill
  left it - the prefill projects only its last 2048 positions - so a restore there drafts worse (never differently)
  until 2048 new positions have refilled the ring (spec 19 §12).

Re-running the acceptance measurement (a Mac or the box CPU; ~20-25 h on a 64 GB Mac with the container at
~54 GB - the first run's numbers are in `docs/probe-dflash-2026-10-08.md` §4):

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

## Agnes 3.0 Flash

`urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ` (spec 14): Qwen3.8's math with 72 layers
(54 delta net, 18 attention) and a parallel FFN, which the loader folds into the MLP
(17408 + 2048 = 19456). Same tokenizer and chat template as Qwen3.8, so the same tool-call
format; an MTP head of Qwen3.8's shape. 22.9 GB download.

**Status: never run on a B70** (queue row 1). On the Mac: the CPU reference, the fold
(bit-exact on the real weights) and the template were checked (docs/probe-agnes-2026-10-03.md).
GEMV tuning rows are provisional copies of Qwen3.8's. Expected (derived): decode ~26 t/s at
depth 4096 with the int8 head, prefill ~1850-1900 t/s at 4096. Weights ~21.0 GB with the
bf16 head, ~19.7 GB with int8 (derived).

```sh
uvx --from huggingface_hub hf download urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ
A=urakozz/Agnes-3.0-Flash-W4A16-AutoRound-GPTQ

# every command below: never run on a B70
# agentic session: ~140k context (derived), ~114k with --mtp auto
./build/src/cli/b70-serve $A --host 0.0.0.0 --port 8000 --served-name agnes \
    --prefix-split-last --mtp auto

# the full 262144 needs the int8 KV cache (~226k with --mtp auto, derived)
./build/src/cli/b70-serve $A --served-name agnes --prefix-split-last --kv-cache int8

./build/src/cli/b70-decode $A --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $A --bench --depth 4096 --tg 256 --lm-head int8
./build/src/cli/b70-decode $A --ids tests/golden/prompts/prose.ids --n 64 --prefill > out.ids
```

**Limits.** Without `--kv-cache int8`, ~140k is the most that fits (derived; the old fixed
65536 ceiling is gone). `--kv-cache int8` uses Qwen3.8's binaries (same attention shape),
whose accuracy gates are pending. MTP and `--spec lookup` load the head (+0.85 GB of weights
plus its buffers), which is where the ~114k comes from.

**Relevant variables:** as for Qwen3.8.

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

**Status: never run on a B70** (queue rows 10, 13, 16, 19). Expected (derived): ~1.86 GB
read per token, so the bandwidth ceiling is far above Qwen3.8's (~270 t/s), and launches,
not bytes, will set the decode rate; 526 launches per token. Prefill estimated at 3-6k t/s.
Weights 19.45 GB with the int8 head (derived), +0.50 GB for the MTP head. For comparison,
the checkpoint's model card reports vLLM XPU on a B70 at 99.8 t/s decode (tg256, c1) and
7090 t/s prefill (pp4096, c1) - not measured here.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ
uvx --from huggingface_hub hf download $O                 # 22.99 GB

# every command below: never run on a B70
# agentic session: the full trained context, 262144, fits with or without MTP (derived)
./build/src/cli/b70-serve $O --host 0.0.0.0 --port 8000 --served-name ornith \
    --prefix-split-last --mtp auto

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256 --lm-head int8
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $O --bench --prefill-length 4096 --tg 256 --prefill-backend l0
./build/src/cli/b70-decode $O --ids prompt.ids --n 64 --prefill > out.ids
./build/src/cli/b70-decode $O --profile --depth 4096
```

**Limits.**

- `--kv-cache int8` is not built at Ornith's attention shape (capture names the missing
  binary); it is not needed for context, since 262144 fits in bf16.
- Prefill runs on `l0-int8` and `l0`; `sycl-tla` is refused. Decode attention `v1`
  (`B70_DECODE_ATTN=v1`) and the composed prefill attention (`B70_PREFILL_ATTN=composed`)
  are not built for Ornith.
- `--mtp auto` decides with Qwen3.8's cost table until the card measures Ornith's (the
  startup line says so); no default K is chosen for Ornith. The head's bf16 experts are
  quantised to int4 at load (round to nearest), which can change acceptance, never output.
- The checkpoint's `tokenizer.json` defines Qwen3.8's 248077 ids, the count the greedy
  argmax masks from, so `b70-serve` prints no vocabulary note. It tokenises the committed
  `tests/golden/prompts` (prose, code, cjk) to the committed ids; its pre-tokenizer splits
  combining marks differently from the bf16 base checkpoint's file (`\p{M}`), which only
  text with combining marks sees.

**Relevant variables:** `B70_KV_CACHE` must stay `bf16`; the `B70_PREFILL_*` switches apply.

## K2-Horizon MoVA-36B-A4B

`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` (spec 18; 22.22 GB): a second
engine beside the Qwen one. 48 layers, every one full attention (32 q-heads, 8 kv-heads,
head dim 128); layers 3-47 route values through 64 value experts (MoVA, top-4) and the FFN
through 100 experts (top-8) plus a shared one. Its own tokenizer (250,624 ids, BOS 0) and
chat template. No MTP head. Trained context 524288.

**Status: never run on a B70** (queue rows 14, 15, 17, 21, 25). Decode (18b) and prefill (18c)
run through `b70-decode`; `b70-serve` serves it (18d: its chat template, tool calls and
reasoning, K2's own engine behind the server, KV-only prefix-cache snapshots - host-tested only).
Expected
(derived): 717 launches per token, 3.15 GB read per token with the int8 head (roofline
~190 t/s from bytes alone), prefill ~3,400 t/s at 4096. For reference, vLLM on **two**
cards (pipeline parallel, fp8 KV) measured 44.43 t/s decode.

```sh
uvx --from huggingface_hub hf download urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ
K=urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ

# every command below: never run on a B70
# K2 ids start with BOS 0: encode with --bos
python3 tools/oracle/tokenize.py <k2-snapshot> encode --bos prompt.txt > k2.ids

./build/src/cli/b70-decode $K --ids k2.ids --n 32                    # decode only, one replay per id
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill          # prefill on l0
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --lm-head int8
./build/src/cli/b70-decode $K --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill --max-len auto   # ~43k, derived

# the reference-rounding attention, for the A/B the box will run
B70_K2_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256

# serving (spec 18d): int8 head, prefix cache on, --max-len auto (~46k bf16 KV, ~90k int8 KV)
./build/src/cli/b70-serve $K --served-name k2 --port 8000
./build/src/cli/b70-serve $K --served-name k2 --kv-cache int8
# K2's template variables per request: "chat_template_kwargs": {"tool_call_format": "xml" |
# "xml_typed" | "json", "reasoning_effort": "high" | "medium" | "low"}
```

**Limits.**

- Served on one card only (no `--pp` in the server), no llama-benchy row yet (queue row 25).
  `b70-serve` refuses `--mtp`, `--spec mtp` / `lookup` and a non-`l0` `--prefill-backend` by name.
- Context: the bf16 KV cache costs 192 KiB per position, so `--max-len auto` gives ~46.6k
  decode-only with the bf16 head, ~49.9k with `--lm-head int8`, ~42.8k when the run
  prefills (bf16 head) - all derived. `b70-decode`'s default is 16384. Long context needs
  int8 KV or two cards (spec 18e, not built).
- Prefill runs on `l0` only: `--prefill-backend l0-int8` and `sycl-tla` are refused by name.
  Omitting `--prefill-backend` is fine.
- Refused: `--mtp` (no head), `--profile` (not built yet). `--kv-cache int8` is built (spec 18e,
  gates pending).
  `--spec lookup` has nothing to run on: it needs verify lists, which only an MTP head
  provides today.
- The routing gate against the CPU reference has not run; until it has, nobody knows
  whether the default `flash` attention or `B70_K2_ATTN=eager` matches the reference better.

**Relevant variables:** `B70_K2_ATTN` (decode and prefill), `B70_PREFILL_REPLAY`.
`B70_PREFILL_PROFILE` times K2's prefill phases but `b70-decode` does not print the table
for K2 yet. `B70_KV_CACHE`, `B70_DECODE_ATTN`, `B70_PREFILL_ATTN` and the GDN switches do
not apply to K2.

## Kolibri-1 (spec 20)

Aleph Alpha's Kolibri-1 (`model_type kolibri1`): 50 MoE layers (384 experts top 6 + an ungated shared
expert), hidden 2560, 48 / 4 heads x 128, 40 sliding-window layers (513 keys, RoPE) and 10 full NoPE
layers, sandwich norms, a 128000-id vocabulary, trained context 262144. **Our int4 checkpoint
`urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ` does not exist yet** (spec 20b, decision 1 open): every
command below needs it, except on a synthetic checkpoint (`tools/quantize/kolibri/make_synth.py`).

**Status: never run on a B70** (queue rows 24, 26 and 28). Decode (spec 20c), prefill (spec 20d) and
serving (spec 20e, `b70-serve`), on **two cards**: at int4 the model holds ~42.5 GB of weights, so `--pp 2` is Kolibri's
default and `--pp 1` is refused unless the model fits one card (a synthetic checkpoint, or `--layers N`:
development mode, the first N layers only - e.g. 30, ~26 GB). Expected (derived): 756 launches per
token (856 with `B70_KOLIBRI_ATTN=eager`), ~2.39 GB read per token with int4 attention and the int8 head -
a roofline near 250 t/s on two cards. Prefill (`--prefill`, `--prefill-length N`) runs chunks of up to
2048 positions on the `l0` backend only: 2252 launches per chunk (45 a layer: the attention linears as
bf16 slabs + `pf_gemm`, the windowed flash attention over the 4096-slot ring, the 384 experts sorted and
dequantised to bf16 in 512 MiB weight batches for the grouped GEMMs) + 5 for the head, ~0.9 GB of prefill
scratch a card; on two cards each chunk crosses once, by copy, and the cards take turns (no overlap yet).

```sh
K=urakozz/Kolibri-1-W4A16-g64-AutoRound-GPTQ          # spec 20b - not made yet

# every command below: never run on a B70; needs 20b's checkpoint and two cards
./build/src/cli/b70-decode $K --ids k.ids --n 32                                  # --pp 2, split by bytes
./build/src/cli/b70-decode $K --ids k.ids --n 32 --pipeline-handoff peer --pipeline-split 25
./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --lm-head int8 --max-len 40960
./build/src/cli/b70-decode $K --pp 1 --layers 30 --ids k.ids --n 8                # one card, 30 layers
B70_KOLIBRI_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256 --lm-head int8
./build/src/cli/b70-decode $K --ids k.ids --n 32 --prefill                        # spec 20d: prefilled
./build/src/cli/b70-decode $K --bench --prefill-length 32768 --tg 16 --lm-head int8 --max-len 40960   # the pp row
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
# every command below: never run on a B70; needs 20b's checkpoint and two cards
./build/src/cli/b70-serve $K                                    # --pp 2, --max-len auto (262144), int8 head
./build/src/cli/b70-serve $K --pipeline-handoff peer --max-len 131072
./build/src/cli/b70-serve $K --prefix-split-last                # agentic sessions: the next turn restores at len - 1
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Wie spät ist es in Berlin?"}], "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

**KL4 (box, after 20b):** `tools/probe/kolibri_passkey.sh` (passkey 3 / 3 at 262144 on two cards) and the
German A4-style tool-call set `tests/golden/toolcall-kolibri-de` (`tools/toolcall/engine_generate.sh`, then
`score.py --format hermes` against the bf16 source's reference, `oracle_generate.py --model kolibri`).

**Limits.** Prefill on `--prefill-backend l0` only (`l0-int8` refused: its h8 linears rotate in
1024-k Hadamard blocks and the hidden is 2560; `sycl-tla` refused: no Kolibri walk), `--prefill-chunk`
at most 2048; no `--mtp` / `--spec` (no head, no verify lists), `--kv-cache int8` refused (20 KiB a
position: bf16 only), no `--profile`, `--device` refused with `--pp 2` (GPUs 0 and 1, `ZE_AFFINITY_MASK`
picks them), context capped at the trained 262144 (decision 3) - `b70-decode` and `b70-serve` alike. The
bench prompt is the first 42 ids of `de_prose.txt` through Kolibri's tokenizer (spec 20e).

**Relevant variables:** `B70_KOLIBRI_ATTN` (`flash` | `eager`, decode and prefill alike),
`B70_PREFILL_REPLAY` (record each prefill chunk's lists once and replay them), `B70_KOL_TIE_TOL` (the
golden test).

## Qwen3.8-Flash-Next (spec 21)

Qwen's `Qwen3.8-Flash-Next` (`model_type qwen4_exp`): 48 layers (36 GatedDeltaNet with a sigmoid output gate, 12 QSA
sparse-attention layers that attend the top 512 blocks of 4 positions plus the open block), a 512-expert / top-10
MoE in every layer, 4-stream hyper-connections instead of a residual, and a per-layer n-gram embedding (PLE) whose
51.8 GB int8 table lives in host memory and is read by the card directly. Intel's interim checkpoint
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (int4 g128 experts, bf16 dense layers) loads through spec 21b's exact
g64 expansion; its PLE table must be converted once (`tools/box_validate/qwen4exp_oracle.sh <data> ple-int8`, ~51.8 GB
beside the snapshot). Our own AutoRound g64 export is spec 21q's.

**Status: never run on a B70** (queue rows 31, 32, 33 and 34). Decode (spec 21c), prefill (spec 21d) and serving
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

**Serving (spec 21e).** `b70-serve <checkpoint> --layers N` serves OpenAI chat on one card or two (`--pp 2`) through
the Qwen chat path: the template is Qwen3.8's byte for byte, XML tool calls and `<think>` reasoning as for Qwen3.8,
greedy unless the request asks (decision 11 open). **The tokenizer must be the original's** (`Qwen/Qwen3.8-Flash-Next`'s
`tokenizer.json`, whose split adds `\p{M}`): Intel's checkpoint ships Qwen3.8's file, which `b70-serve` refuses -
pass `--tokenizer <the original's tokenizer.json>` (a few MB: `hf download Qwen/Qwen3.8-Flash-Next tokenizer.json`).
The prefix cache keeps this family's state with each snapshot (115.7 MB: the 36 GDN states and conv rows, the PLE
history, the QSA open block's raw keys; + 21 KB with the head) and 25,344 B a position in its blocks (+ 2,112 with the
head), one layout on one card or two. `--mtp K` (1-3) drafts with the MTP head (vLLM's form; `pre_fc_norm_hidden` as
one RMS over the 4 streams, `B70_Q4_MTP_NORM=per_stream` for the other form; draft steps after the first reuse step
0's selection, `B70_Q4_MTP_SELECT=fresh` selects every step) and verifies K + 1 rows in one list (790-795 launches at
48 layers, 86-91 at `--layers 4`, derived); `--mtp auto` is refused until spec 22 measures the acceptance on the
whole model. The head's bf16 experts are RTN-quantised to int4 g64 at load; on two cards it runs on the second and
reads the first card's embedding over peer.

```sh
I=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound

# every command below: never run on a B70; needs the PLE int8 file beside the snapshot (or --ple-dir / $B70_Q4_PLE)
./build/src/cli/b70-decode $I --layers 18 --lm-head int8 --ids tests/golden/prompts/q4exp_short.ids --n 32
./build/src/cli/b70-decode $I --layers 38 --pp 2 --lm-head int8 --bench --depth 4096 --tg 256
./build/src/cli/b70-decode $I --layers auto --max-len auto --ids tests/golden/prompts/q4exp_short.ids --n 8
# spec 21d: the prompt prefilled in chunks (l0 only), then decode; the pp row
./build/src/cli/b70-decode $I --layers 18 --lm-head int8 --ids tests/golden/prompts/q4exp_4k.ids --n 32 --prefill
./build/src/cli/b70-decode $I --layers 37 --pp 2 --lm-head int8 --bench --prefill-length 32768 --tg 16 --max-len 33024
# spec 21e: served truncated, the original's tokenizer.json (Intel's ships Qwen3.8's), MTP K = 2, one card / two
T=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/)tokenizer.json
./build/src/cli/b70-serve $I --layers 18 --tokenizer $T --mtp 2
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $I --layers 37 --pp 2 --tokenizer $T --mtp 2
```

**Limits.** Prefill on `l0` only (`--prefill-backend l0-int8`: spec 5's h8 linears rotate in whole 1024-k Hadamard
blocks and the hidden is 2560; `sycl-tla`: no walk for this family), `--prefill-chunk` at most 2048; the two cards
cross each chunk by copy, sequentially (spec 16c's overlapped chunk pipeline is a later lever); MTP in `b70-serve`
only (`b70-decode --mtp` refused; `--mtp auto` and `--draft-vocab` refused), the whole model refused in both CLIs
(spec 22), bf16 KV only (`--kv-cache int8` refused, decision 8), `reasoning_effort` not forwarded to the template (as
for Qwen3.8),
no `--profile`, `--device` refused with `--pp 2`, context capped at the trained 262144.

**Relevant variables:** `B70_Q4_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_Q4_PLE` (the PLE int8
directory), `B70_Q4_MTP_NORM` (`single` | `per_stream`), `B70_Q4_MTP_SELECT` (`reuse` | `fresh`), `B70_PREFILL_REPLAY` (record each prefill chunk's lists once and replay them),
`B70_Q4_TIE_TOL` / `B70_Q4_SEL_TOL` (the golden test's routing and selection near-tie tolerances, 1e-3 proposed).

## Muse-Glimmer-30B (spec 23)

`Vishva007/Muse-Glimmer-30B-W4A16-AutoRound` (or its `-GPTQ` spelling - the same export; our format exactly,
AutoRound int4 g64 sym, used as-is: spec 23 §3), the text half of `meta-models/Muse-Glimmer-30B` (`model_type
muse_glimmer`): 52 dense layers, hidden 6656, 32 / 2 heads x 128 with an output gate, 39 sliding-window layers
(2048 keys, RoPE) and 13 full NoPE layers, sandwich norms, a 202048-id vocabulary, the logits through x 1/sqrt(26)
and a softcap of 20, context 131072. The vision tower (809 tensors) is skipped: text only.

**Status: never run on a B70** (queue rows 43-46). Decode (spec 23c), prefill (spec 23d, `l0` only, chunks of up to
2048) and serving (spec 23e, `b70-serve`) on **one card** - 17.41 GB of weights with the int8 head (the default),
131072 fits beside them (derived); `--pp 2` splits the layers over two cards for capacity, not speed. Expected
(derived): 733 launches a token (681 with `B70_MUSE_ATTN=eager`), 14.72 GB read a token plus 13,312 B of KV a
position on the 13 full layers - a roofline of 39.7 t/s at depth 4096, 35.7 at 131072.

```sh
M=Vishva007/Muse-Glimmer-30B-W4A16-AutoRound
T=tests/tokenizer/muse/chat_template.jinja        # the ORIGINAL's template (the int4 repos ship an older one)

# every command below: never run on a B70
./build/src/cli/b70-decode $M --ids m.ids --n 32 --prefill                     # one card, int8 head
./build/src/cli/b70-decode $M --bench --prefill-length 4096 --tg 256 --max-len 131072
./build/src/cli/b70-serve $M --chat-template $T                                  # one card, --max-len auto (131072)
./build/src/cli/b70-serve $M --chat-template $T --prefix-split-last              # agentic sessions
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve $M --chat-template $T --pp 2      # capacity on two cards
./build/src/cli/b70-serve $M --chat-template $T --spec dflash                    # spec 23f: the assistant, int4, K <= 15
./build/src/cli/b70-serve $M --chat-template $T --spec dflash --draft-format bf16 --spec-max 8
./build/src/cli/b70-serve $M --chat-template $T --spec lookup                    # prompt lookup, the same verify lists
./build/src/cli/b70-decode $M --ids m.ids --n 256 --prefill --spec dflash       # greedy: the ids equal --spec off's
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":
  [{"role": "user", "content": "Read src/main.cc"}], "reasoning_effort": "low"}'
```

**Serving (spec 23e).** The template is `meta-models/Muse-Glimmer-30B`'s `chat_template.jinja` (sha256
`cfc67e5f...`, byte-identical to transformers 5.19.0's `apply_chat_template` on 26 message lists - measured on the
Mac); the snapshot's file is used only when it is that one, so the Vishva007 repos need `--chat-template FILE`
(their older template neither normalises "Reasoning effort" nor skips a system prompt's own reasoning line).
The model writes each message's recipient after the prompt's `<|start|>assistant`: `to=self` comes back as
`reasoning_content` (several blocks joined with a newline), `to=user` as `content`, ATEM calls in tool channels as
OpenAI tool calls (the name from `<atem:invoke name=...>`; a doubled `ns.ns` collapses to a registered bare `ns`;
values typed by the tool's schema first - a string parameter keeps `"007"` as written). A call cut by `max_tokens`
is dropped, never half-sent (`finish_reason` `length`). `<|eom|>` ends a message, not the turn: reasoning, calls and
the answer are one generation; `<|eot|>` and `<|end_of_text|>` stop it. `chat_template_kwargs`: `reasoning_strength`
(`low`, `medium`, `high` - the template's default -, `xhigh`; the OpenAI top-level `reasoning_effort` maps onto it),
`current_date`, `knowledge_cutoff`, `tool_namespace_descriptions`. Sampling: a request without `temperature` /
`top_k` / `top_p` samples with `generation_config.json`'s T 1.0 / top-k 64 / top-p 0.95, through the model's
softcap `20 tanh(z / 20)` of the scaled logits, applied in fp32 on the host (greedy is the device argmax of the raw
row: the cap is monotone). `generation_config.json`'s `stop_strings` (none in Muse's) join every request's `stop`.
The prefix cache keeps the 39 rings' last 2047 positions (81.7 MB, derived) with each snapshot, beside 13,312 B of
KV a position.

**Speculative decoding (spec 23f, never run on a B70: queue row 47).** `--spec dflash` runs Muse-Glimmer's DFlash
assistant `meta-models/Muse-Glimmer-30B-assistant` (bf16 as published, 5.11 GB; never downloaded here) beside the
engine on the same card: it reads the target's residual after layers 1 / 13 / 25 / 37 / 49 of every committed
position, and one 16-row block (the newest token + 15 mask rows, its 2048-key window inclusive) proposes up to 15
tokens through the target's own head; the target checks them in ONE pass at up to 16 rows (23c's decode list at
M = K + 1, every row bitwise the one-row step's) and keeps the longest agreeing prefix (greedy: the output is plain
decode's) - sampled requests draw the guesses from the assistant's candidates and accept them by the lossless rule
against the capped, filtered p. Flags: `--draft-format int4` (the default: g64 RTN at load, 1.36 GB) / `int8` / `bf16`,
`--spec-max 1..15` (default 15; b70-serve picks K per request with AdaptiveK over the derived cost table, `--spec-cost`
calibrates it), `--draft-q raw` (default) / `capped`. `--spec lookup` verifies the server's prompt-lookup guesses
through the same lists. The prefix cache keeps the assistant's ring (43.3 MB) after the 81.7 MB state with each
snapshot, so a restored conversation guesses as an unbroken one. P0 on the Mac (2 short prompts, provisional):
9.33 tokens a check on code at 15 guesses (int4 g64 = bf16 within 0.02), ~1.8 on prose; the box decides the agentic
sets, the verify cost and the default (spec 19 D5's rule: on only if it beats `--spec off` and `--spec lookup` by
>= 10 % on the agentic set). Refused: `--pp 2` with `--spec` (one card), `--draft-vocab`, `--spec-max` above 15,
`B70_PREFILL_REPLAY=1` with `--spec dflash`, `b70-decode --bench --spec` and `b70-decode --spec lookup`.

**MG5 (box):** `tools/toolcall/engine_generate.sh` on `tests/golden/toolcall-muse` (512 ids, `score.py --format
atem` against 23a's bf16 reference: no more parse failures than it) and `tools/probe/muse_passkey.sh` (3 / 3 at
131072, int8 head).

**Limits.** Prefill on `l0` only (`l0-int8` refused: 6656 = 6.5 x 1024-wide Hadamard blocks; `sycl-tla`: no Muse
walk), `--prefill-chunk` at most 2048; no `--mtp` / `--spec mtp` (no MTP head; `--spec dflash` / `lookup` above, one card), bf16 KV
only (`--kv-cache int8` refused, decision 8), context at most 131072 (decision 6), `--device` refused with `--pp 2`,
no `--profile`, image / video content refused.

**Relevant variables:** `B70_MUSE_ATTN` (`flash` | `eager`, decode and prefill alike), `B70_PREFILL_REPLAY`,
`B70_MUSE_TOKENIZER_JSON` (`muse_tokenizer_test`'s tokenizer.json).

## LFM2.5-24B-A2B (spec 24)

`LiquidAI/LFM2.5-24B-A2B` (`model_type lfm2_moe`, LFM Open License v1.0) re-quantised by us to int4 g64 (spec 24 §5:
24q's RTN / AutoRound export - **neither exists yet**; `make_synth.py --layers 4` gives real-width synthetic
checkpoints): 40 layers, hidden 2048, 30 short-conv layers and 10 GQA attention layers (32 / 8 heads x 64, RoPE on
all of them), 2 dense SwiGLU layers then 38 MoE layers (64 experts, top-4 x 1536, a sigmoid router with a
selection-only bias), a tied 128000-row head (125017 real ids), context 131072.

**Status: decode, prefill and serving, never run on a B70** (queue rows 51, 52 and 53; specs 24c, 24d, 24e). The whole model and
131072 positions fit **one card** (13.39 GB of weights with the int8 head + 2.68 GB of KV, derived; the prefill scratch
adds 0.79 GB at chunk 2048); `--pp 2` splits the layers over two cards (decision 10: optional). A prompt goes through
the decode list one replay per id, or - `--prefill` / `--prefill-length` - through the prefill walk in chunks of up to
2048 rows (1259 launches a chunk at 40 layers, derived; the `l0` backend only); `b70-serve` serves it since spec 24e
(below). Expected (derived): 383 launches a token (403 with
`B70_LFM_ATTN=eager`, 277 with `B70_LFM_FUSE=all`), 1.43 GB read a token plus 20,480 B of KV a position - a roofline
of 411 t/s, ~4.5 ms a token ESTIMATED at depth 4096 (spec 24 §3: the step is many small kernels).

```sh
S=oracle-out-lfm2moe-synth/w                       # a synthetic checkpoint (24b's make_synth.py --layers 4)
P=tests/golden/prompts/lfm2moe_short.ids

# every command below: never run on a B70
./build/tools/probe/probe_lfm_step --depth 4096 --rounds 9 --sweep-s --pair      # P0: the skeleton lists (no model)
./build/src/cli/b70-decode $S --ids $P --n 32                                     # one card, int8 head
./build/src/cli/b70-decode $S --ids $P --n 32 --lm-head bf16                      # the head over the embedding tiles
./build/src/cli/b70-decode <export> --bench --depth 4096 --tg 256 --max-len auto
./build/src/cli/b70-decode <export> --profile --depth 4096 --steps 32 --repeats 5 # the per-launch anatomy
B70_LFM_FUSE=all ./build/src/cli/b70-decode <export> --bench --depth 4096 --tg 256   # 277 launches, bitwise none
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode $S --ids $P --n 32 --pp 2 --pipeline-split 2 --pipeline-handoff peer
./build/src/cli/b70-decode $S --ids $P --n 32 --prefill                          # spec 24d: the prompt prefilled
./build/src/cli/b70-decode <export> --bench --prefill-length 4096 --tg 32 --max-len auto   # the pp row (2nd line)
B70_PREFILL_PROFILE=1 ./build/src/cli/b70-decode <export> --bench --prefill-length 4096 --tg 1   # the phase table
./build/src/cli/b70-decode $S --ids $P --n 32 --prefill --prefill-chunk 64       # smaller chunks: bitwise 2048's
```

**Serving (spec 24e).** `b70-serve <export>` runs the engine on **one card by default** (`--pp 2` for capacity;
`--pipeline-split`, `--pipeline-handoff` as `b70-decode` takes them), `--max-len auto` up to 131072 with the prefill
scratch planned in, the int8 head by default, prefill on `l0` only:

```sh
ZE_AFFINITY_MASK=0 ./build/src/cli/b70-serve <export> --port 8000 --prefix-cache-gb auto --prefix-split-last --log-requests ~/lfm-log
```

- **Chat template.** The snapshot's `chat_template.jinja` (the model always reasons: the generation prompt ends in
  `<think>`; `enable_thinking: false` is accepted and ignored, one stderr line says so). `chat_template_kwargs.
  preserve_thinking` (a boolean, default false) keeps the reasoning of the turns before the last user message. A
  message's `reasoning_content` / `reasoning` / `thinking` is written back as `<think>...</think>`. Tool schemas and
  call arguments render with sorted keys (the server's JSON objects are sorted: the prefix cache misses at a call
  whose arguments the model wrote in another order).
- **Output.** The output opens inside reasoning (`reasoning_content` in the response, verbatim, no byte stripped),
  then content and pythonic tool calls `<|tool_call_start|>[fn(k=v, ...)]<|tool_call_end|>` (`tool_calls`, arguments
  a JSON string of the object). A hand-written literal parser reads them (strings in single / double / triple quotes
  with the template's escapes, numbers, `True` / `False` / `None` and JSON's `true` / `false` / `null`, lists, tuples,
  dicts, depth 64, a block read up to 1 MiB); a quote that is not followed by what the grammar allows after a string
  stays in the string (`'it's'`), quoted names are stripped, a call without the list brackets and a lone positional
  dict are accepted; a block with no valid call is returned as content; a call cut by `max_tokens` or EOS is
  emitted once if its `)` closed, otherwise dropped (never leaked as content). `--log-requests` records carry
  `parser.rules`, how often each lenient reading fired.
  Known limits (documented, not changed): (a) an unescaped Windows path ending in a backslash, `ls(path='C:\dir\')`, leaves the string open (the `\'` reads as an escaped quote) and the block becomes content - recovery is deferred; (b) a JSON-format block `[{"name": ..., "arguments": ...}]` is not a pythonic call and becomes content; (c) a malformed call after a valid one is dropped, and only `dropped_unfinished` counts it; (d) a second `<|tool_call_start|>` inside an open block is part of the tail the first non-call text started, so it is dropped; (e) an integer beyond 2^64 becomes a double; (f) an unescaped `print('a', end='')` inside a single-quoted argument is mis-split by the quote lookahead (`', end='` reads as the string's end and a next argument).
- **Sampling.** A request that names no `temperature` / `top_k` / `top_p` / `repetition_penalty` samples with
  `generation_config.json`'s: sampled, T 0.1, top-k 50, top-p 1.0, repetition penalty 1.05. The penalty is HF's
  (`l < 0 ? l * p : l / p` over the distinct ids of the prompt and the output, before temperature / top-k / top-p) on the
  host sampler; `temperature: 0` is the device argmax and ignores it (vLLM applies it to greedy requests too:
  `B70_LFM_PENALTY_GREEDY=1` routes a greedy request that carries a penalty through the host). A request's
  `repetition_penalty` is any finite number > 0 (vLLM's rule; `null` is absent), checked for every model's request, so a
  malformed value is a 400 even where the engine ignores the penalty. A seeded sampled request is reproducible request by request.
- **Prefix cache.** Each snapshot carries the two conv-ring rows of positions pos-1 and pos-2 per conv layer (245,760 B)
  and the attention layers' KV (20,480 B a position).
- `B70_LFM_TOKENIZER_JSON` names a `tokenizer.json` other than the snapshot's.

**Limits.** Prefill on the `l0` backend only (`--prefill-backend l0-int8` / `sycl-tla` refused: spec 24 §11), chunk
at most 2048; no `--mtp` / `--spec` (no MTP head; `--spec lookup` needs M-row verify lists for this family: a
follow-up), no `--draft-vocab`, bf16 KV only (`--kv-cache int8` refused, decision 11), no `--layers` (the whole model
fits), context at most 131072, `--device` refused with `--pp 2`, image / video content refused.

**Relevant variables:** `B70_LFM_ATTN` (`flash` | `eager`), `B70_LFM_FUSE` (`none` | `all` | a comma list of `route`,
`router`, `conv`; the arm `conv` also selects the prefill's conv-tile build), `B70_LFM_TIE_TOL` (the golden gate's
near-tie tolerance, 4e-3), `B70_PREFILL_REPLAY` (record each chunk's list once and replay it), `B70_PREFILL_PROFILE`,
`B70_LFM_TOKENIZER_JSON` (`b70-serve`'s tokenizer.json; `lfm2moe_tokenizer_test`'s), `B70_LFM_SNAPSHOT_DIR`
(`lfm2moe_template_test`'s and `toolcall_lfm2_test`'s snapshot), `B70_LFM_PENALTY_GREEDY` (`1`: a greedy request that
carries a repetition penalty is penalised on the host).

## Nemotron 3.5 Bolt 74B-A9B (spec 25)

NVIDIA's hybrid of 40 Mamba-2 blocks, 8 NoPE attention blocks (GQA 32 / 2, head_dim 128) and 40 latent MoE blocks (top-9
of 240..496 relu² experts in a 1024 latent, a shared expert), 88 blocks, vocabulary 131072; SergiioB's int4 g128 export
(41.9 GB on disk; 40.469 GB on the cards with the int8 head, derived - spec 25 §14).

**Status: decode, prefill and serving built, never run on a B70** (queue rows 60 / 61 / 62; spec 25c / 25d / 25e, as built in
spec 25 §15 / §16 / §17). The real model runs on **two cards** (Decision 4: `--pp 2` is the default, split 48 by bytes;
`--pp 1` is refused naming the bytes); the N 12 synthetic checkpoints and `--layers N` fit one card. Without `--prefill`
a prompt goes through the decode list one replay per id; `--prefill` / `--bench --prefill-length N` run the `l0` prefill
(spec 25d). Expected (derived): 653 launches a token (645 with `B70_BOLT_ATTN=eager`, 533 with `B70_BOLT_FUSE=all`, +2 with
`--pipeline-handoff peer`). P0 (`probe_bolt_step`) runs first on the box and decides the GEMV `{S, layout}` picks and which
arms are worth keeping.

```sh
S=oracle-out-bolt-synth/n12-g128                   # a synthetic checkpoint (25b's make_synth.py, 12 blocks)
P=tests/golden/prompts/bolt_short.ids

# every command below: never run on a B70
./build/tools/probe/probe_bolt_step --depth 4096 --rounds 9 --sweep-s --pair      # P0: the skeleton lists (no model)
./build/src/cli/b70-decode $S --ids $P --n 32                                     # one card, int8 head
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <SergiioB snapshot> --ids $P --n 32   # two cards (the default for 88 blocks)
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256 --max-len 8192
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --profile --depth 4096 --steps 32 --repeats 5
B70_BOLT_FUSE=all ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # 533 launches
B70_BOLT_SSM=bf16 ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-decode <snapshot> --bench --depth 4096 --tg 256   # Decision 2's arm
```

**Serving (spec 25e).** `b70-serve <snapshot>` runs the engine on **two cards by default** (`--pp 1` only for a synthetic /
small checkpoint; `--pipeline-split`, `--pipeline-handoff` as `b70-decode` takes them), `--max-len auto` up to 262144 with
the prefill scratch planned in (Decision 14), the int8 head by default, prefill on `l0` only:

```sh
ZE_AFFINITY_MASK=0,1 ./build/src/cli/b70-serve <SergiioB snapshot> --port 8000 --prefix-cache-gb auto --log-requests ~/bolt-log
```

- **Chat template.** The snapshot's `chat_template.jinja` (sha256 `6521afc7...`, byte-identical in NVIDIA's and SergiioB's
  repos; `--chat-template FILE` renders another, its sha256 is printed). ChatML, no BOS, and the system block is ALWAYS
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
  keys, else the published values (SergiioB's file lists EOS 11 only); `--generation-config FILE` overrides. A seeded
  sampled request is reproducible request by request; `temperature: 0` is the device argmax.
- **Prefix cache.** A snapshot at an aligned 2048-position block end (and at a request's end) carries, per Mamba block, the
  SSM state (fp32: 4 MiB; 2 MiB with `B70_BOLT_SSM=bf16`, which is another store key) and the three raw conv columns
  before the position - 170,229,760 B in all (derived) - and the 8 attention blocks' KV, 8,192 B a position (one state per
  block end: a 64k-token prompt writes 32 of them, ~6.0 GB with the KV, derived - size `--prefix-cache-gb` for it). A hit restores
  the largest block end <= the common prefix and prefills the rest. A new user message re-renders every earlier assistant
  turn without its reasoning (`truncate_history_thinking`), so the cache diverges at the first assistant turn before the
  last user message; within one user turn (the agentic tool loop) the re-render equals what was generated.
- **Request.** Image / video / audio content parts are refused (text only).

**Limits.** No `--mtp` (no MTP head), no `--spec` (spec 26), no `--draft-vocab`, bf16 KV only (`--kv-cache int8` refused,
Decision 17), prefill on `l0` only (`--prefill-backend l0-int8` / `sycl-tla` refused), context at most 262144, `--layers`
is `b70-decode`'s development mode only, `--device` refused with `--pp 2`.

**Relevant variables:** `B70_BOLT_ATTN` (`flash` | `eager`), `B70_BOLT_FUSE` (`none` | `all` | a comma list of `router`,
`route`, `conv`), `B70_BOLT_SSM` (`fp32` | `bf16`), `B70_BOLT_TIE_TOL` (the golden gate's near-tie tolerance, 1e-3 proposed),
`B70_BOLT_A_EXP` (the reference's `A`: `torch` | `engine`), `B70_BOLT_SNAPSHOT` (`template_bolt_test`'s,
`bolt_tokenizer_test`'s and `bolt_request_test`'s snapshot).
