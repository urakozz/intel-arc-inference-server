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

| | Qwen3.8-27B | Agnes 3.0 Flash | Ornith 1.5 35B-A3B | K2-Horizon MoVA-36B-A4B | Kolibri-1 (spec 20) |
|---|---|---|---|---|---|
| checkpoint | published | published | published (spec 15 §13) | published | **not made** (spec 20b) |
| `b70-serve` | yes | yes | yes | yes (spec 18d, box gates pending) | yes (spec 20e, box gates pending; needs 20b), **two cards** by default |
| `b70-serve --pp 2` (two cards, spec 16d) | yes, never run | yes, never run | yes, never run | **refused** (one card) | its default (its own engine, spec 20e), never run |
| `b70-decode` | yes | yes | yes | yes | decode and prefill (spec 20d), **two cards** (`--pp 2`, the default) |
| ran on a B70 | yes (see above) | no | no | no | no |
| MTP head (`--mtp`) | yes | yes | yes (MoE head) | none | none |
| `--spec lookup` | yes | yes (needs the head) | yes (needs the head) | no | no |
| `--kv-cache int8` | yes, gates pending | yes, gates pending | not built | yes, gates pending (spec 18e) | refused (bf16 only) |
| prefill backends | `l0-int8` (default), `l0`, `sycl-tla` | `l0-int8`, `l0` | `l0-int8`, `l0` (`sycl-tla` refused) | `l0` only | `l0` only (spec 20d) |
| KV per position (bf16) | 64 KiB | 72 KiB | 20 KiB | 192 KiB | 20 KiB (+ 335.5 MB of sliding rings) |
| `--max-len auto`, int8 head (derived) | ~201k; ~170k with `--mtp` | ~140k; ~114k with `--mtp` | 262144, with or without `--mtp` | ~46k served (the prefill planned); `b70-decode` ~50k decode-only | 262144 on two cards (split 25), the prefill planned or not |
| with `--kv-cache int8` (derived) | 262144, with or without `--mtp` | 262144; ~226k with `--mtp` | - | ~90k served; `b70-decode` ~98k decode-only | - |

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

Read its `tg` figure. Its own `pp` figure is meaningless against `b70-serve`: the server
sends the role frame before it starts generating, so llama-benchy times the role frame
and reports millions of tokens per second (docs/BENCHMARKS.md, "Why llama-benchy's own pp
figure is unusable"). Use `e2e_ttft` instead, as `tools/serve_bench.sh` does, or
`b70-decode --bench --prefill-length` for the device-side prefill time.

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
