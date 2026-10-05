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

| | Qwen3.8-27B | Agnes 3.0 Flash | Ornith 1.5 35B-A3B | K2-Horizon MoVA-36B-A4B |
|---|---|---|---|---|
| checkpoint | published | published | **not published** (spec 15a) | published |
| `b70-serve` | yes | yes | yes | **refused** (spec 18d's engine side) |
| `b70-decode` | yes | yes | yes | yes |
| ran on a B70 | yes (see above) | no | no | no |
| MTP head (`--mtp`) | yes | yes | yes (MoE head) | none |
| `--spec lookup` | yes | yes (needs the head) | yes (needs the head) | no |
| `--kv-cache int8` | yes, gates pending | yes, gates pending | not built | refused (spec 18e) |
| prefill backends | `l0-int8` (default), `l0`, `sycl-tla` | `l0-int8`, `l0` | `l0-int8`, `l0` (`sycl-tla` refused) | `l0` only |
| KV per position (bf16) | 64 KiB | 72 KiB | 20 KiB | 192 KiB |
| `--max-len auto`, int8 head (derived) | ~201k; ~170k with `--mtp` | ~140k; ~114k with `--mtp` | 262144, with or without `--mtp` | `b70-decode` only: ~47k decode-only (bf16 head), ~43k with prefill |
| with `--kv-cache int8` (derived) | 262144, with or without `--mtp` | 262144; ~226k with `--mtp` | - | - |

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
`b70-decode --bench --pp` for the device-side prefill time.

**Token ids for `b70-decode --ids`** are whitespace-separated, made with the checkpoint's
own tokenizer (needs `transformers`; `<snapshot>` is the snapshot directory in the cache):

```sh
python3 tools/oracle/tokenize.py <snapshot> encode prompt.txt > prompt.ids
python3 tools/oracle/tokenize.py <snapshot> decode $(cat out.ids)
```

`tests/golden/prompts/{prose,code,cjk}.ids` are ready-made ids for the Qwen tokenizer
(Qwen3.8 and Agnes share it).

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
./build/src/cli/b70-decode $Q --bench --pp 4096 --tg 256
./build/src/cli/b70-decode $Q --bench --pp 4096 --tg 256 --lm-head int8        # the serving head
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

./build/src/cli/b70-decode $A --bench --pp 4096 --tg 256
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
refuses: it loads int4 GPTQ (AutoRound W4A16 g64, symmetric, experts exported per expert
as `experts.E.{gate,up,down}_proj`). **No such checkpoint is published yet; spec 15a
makes it.** The commands below use the repo id the tests assume,
`urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ`, **as a placeholder**: substitute
the real one, or a local snapshot directory of your own export.

The first mixture-of-experts model (spec 15): 40 layers (30 delta net, 10 attention),
hidden 2048, 256 experts with top-8 plus a shared expert, ~3 B parameters active per token.
Decode (15c), prefill (15d) and serving with its own chat template (Qwen3.5's), Qwen XML
tool calls and the MoE MTP head (15e) are built.

**Status: never run on a B70** (queue rows 10, 13, 16), and nothing can run until the
checkpoint exists. Expected (derived): ~1.86 GB read per token, so the bandwidth ceiling is
far above Qwen3.8's (~270 t/s), and launches, not bytes, will set the decode rate; 526
launches per token. Prefill estimated at 3-6k t/s. Weights 19.45 GB with the int8 head
(derived), +0.50 GB for the MTP head.

```sh
O=urakozz/Ornith-1.5-35B-A3B-W4A16-g64-AutoRound-GPTQ   # PLACEHOLDER: not published yet (spec 15a)
uvx --from huggingface_hub hf download $O

# every command below: never run on a B70
# agentic session: the full trained context, 262144, fits with or without MTP (derived)
./build/src/cli/b70-serve $O --host 0.0.0.0 --port 8000 --served-name ornith \
    --prefix-split-last --mtp auto

./build/src/cli/b70-decode $O --bench --depth 4096 --tg 256 --lm-head int8
./build/src/cli/b70-decode $O --bench --pp 4096 --tg 256
./build/src/cli/b70-decode $O --bench --pp 4096 --tg 256 --pp-backend l0
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
- At startup `b70-serve` notes that Ornith's `tokenizer.json` defines 248070 ids while the
  model masks from 248077. That is expected: seven audio tokens are missing from the
  tokenizer, and the engine never needs them.
- Whether Ornith's tokenizer gives the committed `tests/golden/prompts` ids is still to be
  checked; make your own `--ids` file with `tokenize.py` against its snapshot.

**Relevant variables:** `B70_KV_CACHE` must stay `bf16`; the `B70_PREFILL_*` switches apply.

## K2-Horizon MoVA-36B-A4B

`urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` (spec 18; 22.22 GB): a second
engine beside the Qwen one. 48 layers, every one full attention (32 q-heads, 8 kv-heads,
head dim 128); layers 3-47 route values through 64 value experts (MoVA, top-4) and the FFN
through 100 experts (top-8) plus a shared one. Its own tokenizer (250,624 ids, BOS 0) and
chat template. No MTP head. Trained context 524288.

**Status: never run on a B70** (queue rows 14, 15, 17). Decode (18b) and prefill (18c) run
through `b70-decode`. **`b70-serve` refuses it**: the chat template, tokenizer, tool-call
parser and reasoning split are built and host-tested (18d host side), but the engine half
of serving (K2 behind the server, KV-only prefix-cache snapshots) is not. Expected
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
./build/src/cli/b70-decode $K --bench --pp 4096 --tg 256
./build/src/cli/b70-decode $K --ids k2.ids --n 32 --prefill --max-len auto   # ~43k, derived

# the reference-rounding attention, for the A/B the box will run
B70_K2_ATTN=eager ./build/src/cli/b70-decode $K --bench --depth 4096 --tg 256
```

**Limits.**

- Not served (above), so there is no llama-benchy row for K2 yet.
- Context: the bf16 KV cache costs 192 KiB per position, so `--max-len auto` gives ~46.6k
  decode-only with the bf16 head, ~49.9k with `--lm-head int8`, ~42.8k when the run
  prefills (bf16 head) - all derived. `b70-decode`'s default is 16384. Long context needs
  int8 KV or two cards (spec 18e, not built).
- Prefill runs on `l0` only: `--pp-backend l0-int8` and `sycl-tla` are refused by name.
  Omitting `--pp-backend` is fine.
- Refused: `--kv-cache int8` (spec 18e), `--mtp` (no head), `--profile` (not built yet).
  `--spec lookup` has nothing to run on: it needs verify lists, which only an MTP head
  provides today.
- The routing gate against the CPU reference has not run; until it has, nobody knows
  whether the default `flash` attention or `B70_K2_ATTN=eager` matches the reference better.

**Relevant variables:** `B70_K2_ATTN` (decode and prefill), `B70_PREFILL_REPLAY`.
`B70_PREFILL_PROFILE` times K2's prefill phases but `b70-decode` does not print the table
for K2 yet. `B70_KV_CACHE`, `B70_DECODE_ATTN`, `B70_PREFILL_ATTN` and the GDN switches do
not apply to K2.
