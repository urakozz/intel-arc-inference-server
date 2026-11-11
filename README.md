# b70-inference-server

An LLM inference engine I wrote from scratch for the Intel Arc Pro B70, talking
straight to Level Zero and OpenCL C. No PyTorch, no vLLM, no vendor kernel
library in the hot path.

I started it to learn what this silicon actually does, and to see whether a
small specialised engine could beat a general one on the same box. It can, at
least for prefill.

## Where it stands

**Serving Qwen3.8-27B on two Arc Pro B70s** (2026-11-11, `b70-serve`, MTP speculative decoding, each request a
2820-token prompt and 512 generated tokens, greedy; time to first token cold, then again as a prefix cache hit):

| setup | requests at once | **total t/s** | per request t/s | first token, cold | first token, prefix cache hit |
|---|---:|---:|---:|---:|---:|
| `--parallel tp` (both cards, one session - the two-card default) | 1 | **123.1** | 123.1 | 1.84 s | 1.34 s |
| `--max-num-seqs 8` (replica: a model copy per card, 4 slots each) | 1 | 73.0 | 73.0 | 2.30 s | 1.08 s |
| | 2 | 140.5 | 73.8 | 2.36 s | 1.10 s |
| | 4 | 209.1 | 58.5 | 3.61 s | 1.65 s |
| | 8 | **250.0** | 36.0 | 7.27 s | 2.79 s |
| same, `--kv-cache int8` (slot 0 holds 167,936 tokens instead of 46,080) | 1 / 2 / 4 / 8 | 74.0 / 163.5 / 213.9 / **258.5** | 74.0 / 84.1 / 58.8 / 37.2 | 2.33 / 2.37 / 3.63 / 7.30 s | 1.09 / 1.11 / 1.64 / 2.81 s |
| same, `--prefill-backend l0-int8` (int8 prompt pass, bf16 cache) | 1 / 2 / 4 / 8 | 82.0 / 138.5 / 202.2 / 250.1 | 82.0 / 76.3 / 60.6 / 35.0 | **1.80** / 1.88 / 2.82 / 5.76 s | **0.86** / 0.89 / 1.33 / 2.24 s |
| one card, `--max-num-seqs 4` | 1 / 2 / 4 | 73.2 / 107.5 / 126.6 | 73.2 / 59.5 / 35.6 | 2.32 / 3.52 / 5.91 s | 1.09 / 1.65 / 2.77 s |

Pick by use: one session as fast as possible is TP (the default on two cards); a main session with helper requests
is `--max-num-seqs N` (N up to 8 on two cards, 4 on one). MTP's speed depends on how predictable the text is, so
per-request rates move a few t/s between runs (`tools/box_validate/slots_bench.py`, logs on the box under
`~/b70/logs/brag-speed/`).

**Prefill (reading the prompt)**, t/s, median of three:

| | 4,096-token prompt | 16,384-token prompt |
|---|---:|---:|
| one card, bf16 prompt pass (`--prefill-backend l0`) | 1,709 | 1,539 |
| one card, int8 prompt pass (`--prefill-backend l0-int8`) | 2,205 | 1,940 |
| `--parallel tp` (both cards), bf16 prompt pass | **2,939** | **2,686** |

TP runs the bf16 prompt pass only for now; the int8 pass under TP (one card gains ~1.27x from it) is open-work
item 10. A short agentic turn deep in a session is fast too: 16 new tokens at 128k of history take 98 ms (bf16
cache) or 108 ms (int8).

Against vLLM on the same checkpoint and prompt, one card reads a 4,096-token prompt at 2,205 t/s to vLLM's 1,610.
Context reaches the model's full 262,144 tokens (TP on two cards, or one card with `--kv-cache int8`), and long
sessions keep their history in the prefix cache in system RAM: a turn at 60k tokens of history is a prefix cache hit
with its first token in about 1.2 s instead of 45 s of prefill.

The checkpoint is `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: int4 weights, group size 64 (why: see below),
bf16 activations. Every measurement and its protocol is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Scope

One model family and one math path, on purpose. **So far the engine is built and
tuned for one model only: the dense Qwen3.8-27B** (measured on the card). It is a
hybrid: 48 gated delta net layers and 16 full attention layers. Everything here
is built around that shape and around W4A16: the kernel shapes, the captured
decode list and the tuning tables are that model's. Other checkpoints of the
same architecture load, but nothing is tuned for them. The first mixture-of-experts
model, Ornith 1.5 35B-A3B (spec 15: 256 experts, top-8, ~3 B active), decodes,
prefills and is served by `b70-serve` with its own chat template, tool calls and
MTP head (spec 15c-15e) - all written without the card and **not yet run on it**; its
checkpoint is `urakozz/Ornith-1.5-35B-A3B-W4A16-AutoRound-GPTQ` (spec 15 §13). Weights load from our own
AutoRound / GPTQ g64 format (recommended), from GPTQ g128, and from llm-compressor's
**symmetric** compressed-tensors `pack-quantized` int4 (g64 / g128, e.g.
RedHatAI/Qwen3.8-27B-INT4). The last is converted exactly at load with no kernel change, and
the loader prints a note that it expects reduced quality compared to our recommended
format, AutoRound GPTQ W4A16 g64 symmetric. Asymmetric checkpoints are refused (docs/13,
spec 20 §9). Specialisation is the whole strategy, since a general engine cannot hardcode
the things this one hardcodes.

## The checkpoint, and why group size 64

The engine is developed and gated against one checkpoint I quantised myself:
[`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`](https://huggingface.co/urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ),
AutoRound, int4 weights with one scale per 64 input channels, bf16 activations,
GPTQ packing; the file keeps `lm_head` and the MTP head in bf16.

Why 64 rather than the common 128, or a per-channel scale:

- **Accuracy.** Finer groups track the weights' outliers better. Measured on
  this model, a per-channel int4 checkpoint (AutoRound on the Hadamard-rotated
  model) had about three times the logit error of g64: 22.9 % against 7.77 %
  (docs/probes/probe-w4a8-2026-09-23.md §15.2).
- **Decode is bandwidth bound**, and g64 costs only a few percent more bytes
  than g128 (one fp16 scale per 64 weights), so the finer grid is nearly free
  where the time actually goes.
- **Prefill speed does not depend on the group size any more.** Coarser groups
  would help a mixed int4 x int8 matmul (g128 1.22x, g256 1.43x, per channel
  1.75x over bf16, §13), but spec 5 does better without touching the file: it
  rebuilds per-channel int8 weights from the g64 checkpoint on the fly and runs
  int8 x int8 (docs/17-int8-prefill.md). So the checkpoint keeps g64's accuracy
  and prefill still gets the int8 matrix engines.
- **vLLM loads the same file**, which keeps the comparison above byte for byte.

## How it works

**Decode** replays a Level Zero command list that was captured once, with frozen
kernel arguments. 774 kernels and 19 modules per token, no host decisions in the
loop.

**Prefill** runs entirely on Level Zero with our own OpenCL C kernels. The int4
weights are dequantised into 1024 column slabs and multiplied by a DPAS GEMM
written for this card, interleaved on one in order command list. A prefill chunk
never waits on the host. Attention and the delta net scan are ours too.
`--prefill-backend l0-int8` runs the int4 linears on a Hadamard-rotated int8 path
instead (spec 5, opt-in).

## Some hardware facts I measured along the way

Might save you a benchmark or two if you work on Xe2.

DPAS throughput on the B70, from a loop the assembly confirms is DPAS bound:

| type | K per instruction | rate | vs bf16 |
|---|---:|---:|---:|
| bf16, fp16 | 16 | 183.45 TFLOP/s | 1.000x |
| int8 | 32 | 366.90 TIOP/s | 2.000x |
| int4 | 64 | 733.80 TIOP/s | 4.000x |

That bf16 number is 99.97% of the clock derived peak, so the ratios really are
just the K depth. Our production GEMM sits at 161.7 TFLOP/s on the largest
linear, about 88% of peak.

FP4 and FP8 matmul are a different story. The compiler exposes `e2m1` and FP8
builtins, but the backend refuses them on this device: "FP4 Dpas instruction is
not supported on this device". Every `scaled_matrix_mad` form, which is how
hardware microscaling would work, crashes the compiler. So MXFP4 buys nothing
here today, and int8 or int4 activations are the only lower precision paths
that would actually run.

Details in [docs/probes/probe-dpas-rates-2026-09-22.md](docs/probes/probe-dpas-rates-2026-09-22.md).

## Things that did not work

Kept because negative results saved more time than the wins did.

- **Fusing int4 dequant into the GEMM.** Both variants came out bitwise correct
  and slower. The dequant pass already runs near memory bandwidth, and doing the
  same work inside the mainloop costs more than the traffic it removes.
- **Removing barriers from the triangular solve.** The design assumed
  synchronisation dominated that kernel. Deleting 64 of 128 barriers bought
  1.7 ms of a 73.5 ms row, so the premise was wrong by about 18x. The row is the
  serialised recurrence itself.
- **bf16 intermediate buffers.** Halving the bytes changed nothing, because
  `ocloc` has no 16 bit block write wider than 8 rows, so the epilogue is bound
  by store message count rather than by bytes.

## Testing

Numerics are gated rather than eyeballed.

- Every kernel that replaced another is compared **bitwise**, not by tolerance.
- The golden gate replays three prompts against a CPU oracle and requires all 93
  determined token rows to match element exactly.
- Determinism is checked bitwise across runs from reset.
- Benchmarks only count as a record when the box is provably idle, and the
  harness measures that itself rather than taking your word for it.

86 tests in the suite as of now.

One honest exception: the delta net scan uses approximate split bf16 arithmetic,
so it is gated on tokens and state cosines instead of bitwise equality. It is
the only default in the prefill path that is not bit identical to what it
replaced.

## Build and run

The engine builds with g++ and CMake.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# the engine never downloads; fetch the checkpoint into the Hugging Face cache once
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-serve  urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --kv-cache int8   # OpenAI compatible endpoint
# the KV cache defaults to bf16, as in every engine; the commands here pass --kv-cache int8 (half the KV bytes:
# more context, more requests at once - spec 12b); --spec dflash needs bf16 (its 8-row verify)
```

Both CLIs take the repo id and resolve it through the local cache
(`$HF_HOME` or `~/.cache/huggingface`, revision from `refs/main`); a snapshot
directory path works too.

Commands for each model (Qwen3.8, Ornith 1.5, K2-Horizon, Kolibri-1 - decode,
prefill (`l0` only) and serving on two cards, built blind, needs spec 20b's checkpoint - and
Qwen3.8-Flash-Next - decode, prefill and serving with its MTP head, built blind, truncated to `--layers N`
until spec 22 runs the whole model - and Nemotron 3.5 Bolt - decode, prefill and serving on two cards, built blind,
spec 25), what each one supports, its limits, and which configurations have
actually run on a B70:
[docs/18-running-models.md](docs/18-running-models.md).

### Serving

```sh
# long agentic sessions (opencode and similar): the defaults - the largest context that fits
# the card beside the weights and the MTP head (--max-len auto: ~170k, derived), MTP speculative
# decoding with K chosen per request (--spec auto), the prefix cache on - plus the prompt-end
# snapshot one id early so each turn is a prefix cache hit where the next one diverges
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --kv-cache int8 \
    --host 0.0.0.0 --port 8000 --prefix-split-last

# plain decoding, no speculative proposer (auto context: ~201k, derived)
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --kv-cache int8 --spec off

# measure it the way the vLLM rows were measured (the server answers to any model name;
# /v1/models lists it as the repo id)
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

You choose the model, the speculative decoding, the context length and the parallel mode; the
engine decides everything else (the int8 `lm_head`, the memory it leaves free, the draft model's
form, the cost tables, the tokenizer, chat template and generation defaults - the snapshot's) and
refuses at startup, naming why, what does not fit or is not built. It runs on one or two Intel Arc
Pro B70s with peer access between them (checked before any weight is read; `ZE_AFFINITY_MASK`
picks which cards the driver shows).

`b70-serve` flags:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | HF repo id resolved in the local cache, or a snapshot directory; never downloads. The model (Qwen3.8, Ornith 1.5, K2-Horizon, Kolibri-1, Qwen3.8-Flash-Next, Nemotron 3.5 Bolt) comes from its `config.json`, and it is served under that name (the repo id, or the directory's). What each model supports, refuses and has run on a card: [docs/18-running-models.md](docs/18-running-models.md). |
| `--spec auto\|off\|mtp\|lookup\|dflash\|dflash-lookup` | `auto` | Speculative decoding: a proposer guesses the next few tokens, the main model checks all the guesses in one pass, and every correct guess is a token for free (output is unchanged either way). `auto` is the model's validated choice: the MTP head with K picked per request on the Qwen family (Qwen3.8, Ornith), none elsewhere. `mtp`: the model's built-in MTP head; K per step from how often that request's guesses were right recently (about 3 on tool calls and code, 1 on prose, 0 when guessing does not pay); Qwen3.8-Flash-Next guesses a fixed K = `--spec-max` (its adaptive policy waits for spec 22's numbers). Loads the head (+1.4 GB at 16k context, ~2.3 GB at 128k, derived; Ornith's MoE head +0.5 GB of weights). `lookup` needs no extra model: the guesses are the continuation of the longest earlier repeat (3 ids or more) of the last tokens in the request; it checks them with the MTP head's verify lists, so only a model with a head runs it (spec 19e, not yet run on the card). `dflash` (spec 19c / 19d, Qwen3.8 only): the draft model `z-lab/Qwen3.8-27B-DFlash2`, quantised to int8 at load (2.05 GB), reads the main model's hidden states and guesses up to 7 tokens in one pass; needs `--kv-cache bf16` and, on two cards, `--pipeline-handoff copy`; the prefix cache keeps its state (41.9 MB) with each snapshot (not yet run on the card). `dflash-lookup` (spec 19f): prompt lookup inside DFlash, one card, opt-in. `off`: none. **Nemotron 3.5 Bolt** (spec 26c, opt-in, not yet run on a card): `lookup` and `dflash` over its own verify pass of up to 8 rows - `dflash` is NVIDIA's LiLiCorr draft model (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`, int8 at load, on the second card), `--spec-max` 1-7 for both; its `auto` stays off until box row 65 measures it ([docs/18-running-models.md](docs/18-running-models.md)). |
| `--spec-max K` | the most | the proposer's cap: 1-3 guesses (`mtp`, `lookup`), 1-7 (`dflash`, its block of 8; Bolt's `lookup` too) |
| `--draft-model PATH` | `z-lab/Qwen3.8-27B-DFlash2` (Bolt: its LiLiCorr) | with `--spec dflash`: the draft model, a snapshot directory or a repo id in the HF cache (never downloaded) |
| `--max-num-seqs N` | `1` | how many requests run at once (vLLM's flag, the same meaning); the context follows. `1`: one request at the largest context that fits (on two cards, `tp`). Above 1 (Qwen3.8): each card holds up to 4 request slots - slot 0 gets the largest context that fits beside N - 1 slots of 32768 tokens (e.g. 131072 + 3 x 32768); one card takes up to 4, two cards up to 8 (`replica`, the slots split between the cards: 2 = one full-context slot per card). A request that reaches its slot's end stops with `length` and the server log says so. The layout is printed at startup. |
| `--max-len auto\|L` | `auto` | context capacity. `auto`: the largest multiple of 256 whose memory plan (weights + KV + decode state + prefill scratch, the MTP head's buffers when it is loaded, the `--draft-vocab` head) fits the card with the engine's reserve (1.5 GB) left over, capped by the checkpoint's trained context (`max_position_embeddings`, 262144); the chosen length and the plan are printed at startup (spec 6 §10). Derived: Qwen3.8 ~170k with MTP (the default), ~201k with `--spec off`. The full 262144 needs `--kv-cache int8` (spec 12b) or two cards (spec 16). `L`: any multiple of 256 up to the trained context, refused at startup with the plan's breakdown if it does not fit. With `--max-num-seqs` above 1 this is slot 0's length, beside the other slots. |
| `--parallel auto\|1\|pp\|tp\|replica` | `auto` | `auto`: Qwen3.8 on two B70s runs `tp` with `--max-len auto` (spec 17, operator 2026-11-10) and `replica` with `--max-num-seqs` above 1; otherwise one card when the model and the asked context fit it, else `pp`. `tp`: every layer split over both cards (Qwen3.8; ~1.46x one card's decode with `--spec auto`, shared-box numbers; DFlash, `--draft-vocab` and the l0-int8 prefill are refused). `replica`: a copy of the model on each card, requests spread over both (`--max-num-seqs` 2 to 8, 2 when not given; `tp` runs one request at a time and refuses more). `1`: one card. `pp`: the model's layers split over two cards (spec 16d; GPUs 0 and 1 of what `ZE_AFFINITY_MASK` shows). One answer is not faster on two cards, but each card holds about half the model, so the context can be longer - Qwen3.8 reaches its full 262144 tokens with or without MTP (derived) - and a prompt that misses the prefix cache prefills on both cards at once (spec 16c). Everything else works as on one card: the prefix cache, `--prefix-split-last`, `--spec mtp` (the head on the second card with its own copy of the embedding, 2.54 GB), `--spec lookup`, `--spec dflash` (`--pipeline-handoff copy` only), `--draft-vocab`, `--kv-cache int8`. K2-Horizon is one card only; Kolibri-1 and Nemotron 3.5 Bolt run on two cards by default. |
| `--pipeline-split auto\|N` | `auto` | an override for `pp` (it implies `pp`): device 0 runs layers 0 to N-1; `auto` picks the cut that leaves both cards holding about the same bytes (the MTP head and the prefix cache's buffers count) |
| `--host H` | `0.0.0.0` | listen address |
| `--port P` | `8000` | listen port |
| `--log-requests DIR` | off | write `DIR/NNNNNN.json` per request: timings, body, prompt and generated ids, text |
| `--prefix-cache-gb auto\|N` | `auto` | The prefix cache's size in GiB of **system RAM** (pinned host memory, not the card's VRAM): every prefill is written through to it so a later request with the same history is a prefix cache hit instead of a re-prefill (spec 7). Pinned pages cannot be swapped, so `auto` takes the smallest of 32 GiB, half the machine's RAM, and the RAM available after the model is loaded minus 8 GiB, and turns the cache off below 4 GiB; the startup line says what it chose and why. `N` pins exactly N GiB (with a warning if that is more than is available); `0` = off, every request prefills in full. |
| `--prefix-split-last` | off | Chat requests only. A thinking model's next turn repeats this prompt but diverges at its **last** id: the template ended this prompt with `<think>\n`, and the history re-renders that turn as `<think>\n\n</think>` when the client drops the reasoning (opencode does). By default the prompt-end snapshot sits one id past that point, so the next turn falls back to the 2048-id block below and re-prefills up to 2047 ids it already had (~1 s per turn). With the flag the prompt is prefilled to len - 1, snapshotted there, and the last id runs as one decode step, so the next turn (or a retry of the same prompt) is a prefix cache hit exactly where it diverges. The cost: that one id goes through decode's kernels, so a near-tie first token can differ from a run without the cache (same quality, not bitwise reproducible). **On for agentic sessions; off for benchmarks and the golden gates.** |
| `--layers N\|auto` | - | Qwen3.8-Flash-Next only, and required for it until spec 22: serve its first N layers (then its final mixer and the head), as `b70-decode --layers` (spec 21c); `auto` is the planner's N. Its PLE int8 table is read from `<snapshot>-ple-int8/`. Refused for any other model |

Still being measured, and removed or made automatic when their numbers decide (docs/open-work.md):

| flag | default | what it does |
|---|---|---|
| `--kv-cache bf16\|int8` | `bf16` | How the context (the KV cache) is stored. `int8` keeps every cached key and value as 8-bit numbers plus one small scale per token and head, after mixing each row with a fixed rotation that spreads out the few very large values (the 12a probe's `rotkv`, spec 12 §8); attention mixes it back. That halves the cache: 32 KiB + 256 B per token on Qwen3.8 instead of 64 KiB, so `--max-len auto` reaches the full trained context, 262144 tokens, with or without MTP, and the prefix cache's KV blocks halve too. Accuracy: in the 12a probe (2026-09-28, on Agnes 3.0 Flash - since dropped - as the stand-in) its error was below bf16 attention's own rounding error; the gates on Qwen3.8 and on the card are **pending** (spec 12 Q2-Q5), which is why the default stays `bf16`. Spec 12b. |
| `--prefill-backend l0\|l0-int8` | `l0` | prefill GEMMs: `l0` (bf16 on the Level Zero list) or `l0-int8` (Hadamard-rotated int8, spec 5, opt-in; on a MoE model such as Ornith it costs routing accuracy, so use it there for speed only) |
| `--pipeline-handoff copy\|peer` | `copy` | with `pp`: how the hidden state crosses, as for `b70-decode` (it implies `pp`) |
| `--draft-vocab off\|32k\|64k\|128k` | `off` | With MTP (or `--spec dflash`: the draft model's guesses, spec 19c): the head only guesses from the 32k / 64k / 128k most useful tokens (the chat and tool-call tags always, then `--draft-vocab-ids`, then the most common ones), so each guess reads a fraction of the output head. The check still uses the full vocabulary, so output never changes; a guess the subset cannot make is simply a miss. About 4-5 % faster MTP steps at 128k (derived, not yet measured). Costs 0.17 / 0.34 / 0.67 GB of memory, which comes out of the `--max-len auto` context: at 128k, Qwen3.8 with MTP gets ~160k instead of ~170k (derived). Spec 8 §11. |
| `--draft-vocab-ids FILE` | - | With `--draft-vocab`: your own ranking of the most useful tokens, one id per line, most frequent first. Make it with `tools/draft_vocab/rank.py` from the logs `--log-requests` writes. |

### Benchmarking and checking

```sh
# the BENCHMARKS.md rows: prefill 4096 and decode 256 at the prefilled depth
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench --prefill-length 4096 --tg 256

# greedy ids in, ids out (tools/oracle/tokenize.py writes the --ids file)
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --ids prompt.ids --n 64 --prefill > out.ids

# per-launch anatomy of the decode step
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --profile --depth 4096 --steps 32 --repeats 5
```

`b70-decode` flags:

| flag | default | what it does |
|---|---|---|
| `--ids FILE` | - | whitespace-separated prompt ids; generated ids go to stdout, one per line |
| `--n N` | - | `--ids`: ids to generate, greedily |
| `--prefill` | off | `--ids`: run the prompt through the chunked prefill instead of one replay per id |
| `--spec off\|mtp\|dflash` | `off` | `--ids`: greedy decoding with a proposer's guesses, as `b70-serve --spec` (the same ids as without it; stderr adds the acceptance): `mtp` (adaptive K <= `--spec-max`, the Qwen family) or `dflash` (Qwen3.8, one card, `--draft-model PATH`). Not yet run on the card |
| `--spec-max K` | the most | as for `b70-serve` |
| `--max-len L\|auto` | `16384` | as for `b70-serve`, but 16384 stays the default so bench rows stay comparable |
| `--parallel auto\|1\|pp` | `auto` | as for `b70-serve`. `pp` splits the layers over two cards: the first runs the first part with the embedding, the second the rest with the output head, and each token's hidden state crosses once per step. With `--prefill` or `--bench --prefill-length N` the prompt is prefilled on both cards at once: while the second card runs one 2048-token chunk through its layers, the first already runs the next - about 1.8-1.9x one card's prefill at 32k tokens and beyond, 1.3x at 4k (derived, not yet measured), each card's busy time on stderr, results identical to one card. `--spec mtp` runs with it (the head on the second card); `--profile` is refused with it. Specs 16b / 16c. |
| `--pipeline-split auto\|N` | `auto` | with `pp` (it implies `pp`): where the layers are cut. `auto` leaves both cards holding about the same number of bytes (weights, context, buffers) - not the same number of layers, because the output head's card is heavier; Qwen3.8 cuts at 29 of 64 (derived). `N`: the first card runs layers 0 to N-1. Each card needs at least one layer of each kind. |
| `--layers N\|auto` | - | Kolibri-1, Nemotron 3.5 Bolt and Qwen3.8-Flash-Next: load only the first N layers (development mode; Flash-Next requires it until spec 22, `auto` is its planner's N) |

Developer flags (benchmarks and profiles; `b70-decode --help` lists them under "developer"):

| flag | default | what it does |
|---|---|---|
| `--bench` | - | ingest `--depth` synthetic ids (or prefill `--prefill-length N`), time `--tg` generated ones, print a markdown row (the int8 head: rows say `int8-head`) |
| `--depth D` | `4096` | `--bench` / `--profile`: context length before decoding, ingested one replay per id |
| `--prefill-length N` | - | `--bench`: prefill N synthetic ids instead (exclusive with `--depth`) and print a second row with the prefill time |
| `--tg N` | `256` | `--bench`: ids to generate |
| `--prefill-chunk C` | `2048` | positions per prefill chunk, at most 4096 |
| `--profile` | - | replay `--steps` instrumented decode steps and print the per-launch anatomy (never a bench row) |
| `--steps N` | `32` | `--profile`: steps per session |
| `--repeats R` | `1` | `--profile`: independent sessions, for the spread (R >= 5 before claiming a delta under ~0.3 ms) |

Still being measured, as for `b70-serve`: `--kv-cache bf16|int8` (int8 rows are marked `int8-kv`; **K2-Horizon too**, spec 18e: 97.5 KiB per token instead of 192 KiB, so `--max-len auto` gives ~98k decode-only and ~90k with `--prefill`, bf16 KV ~50k / ~46k; derived), `--prefill-backend l0|l0-int8`, `--pipeline-handoff copy|peer` (`copy`: the first card copies the hidden state to the second and signals it; `peer`: the first card's last kernel writes it straight into the second card's memory and raises a flag the second card waits on, with a time limit; same results either way, which is faster is not yet measured).

`tools/box.sh` builds and tests on a remote machine over ssh, which is how I
work day to day. Set `BOX=user@host` before using it. `tools/box_check.sh --base main`
is the merge check: the build on the box, every existing kernel's ocloc command line
against the base, and the tests that need no model files on a free card.

### Environment variables

Read by `b70-serve` and `b70-decode`. A command-line flag always wins over the variable
that sets its default. The switches below are for experiments and A/B timing; leave them unset
to run what the gates ran (the zero-config cleanup, 2026-11-06, removed the ones whose numbers had
decided).

| variable | read by | default | what it does |
|---|---|---|---|
| `HF_HOME` | both CLIs | `~/.cache/huggingface` | where a repo id is looked up: `$HF_HOME/hub/models--<org>--<name>`, revision from `refs/main`. The same place `hf download` writes to. |
| `ZE_AFFINITY_MASK` | the Level Zero driver | unset: every card visible | hides cards from the process: `ZE_AFFINITY_MASK=1` leaves only the second card, which the engine then sees as device 0. `tools/box.sh` and `tools/bench_decode.sh` forward it to the box. |
| `B70_KV_CACHE` | both CLIs | `bf16` | the default of `--kv-cache` (`bf16` or `int8`); any other value is refused. This is how the tests run the same binaries over the int8 cache. |
| `B70_PREFILL_PROFILE` | prefill | off | `1` times every prefill phase; `b70-decode --bench --prefill-length` prints the table on stderr (Qwen-family models; not yet for K2). It adds a host wait per phase, so never use it for a recorded row. |
| `B70_KOLIBRI_ATTN` | `b70-decode`, Kolibri-1 | `flash` | Kolibri-1's decode and prefill attention. `eager` is the reference's bf16 chain (spec 18 §10.1's switch, spec 20 §11): 856 launches per token instead of 756, plus a score row of 4 B x 48 heads x max_len; prefill binds the `_EAGER` flash builds (the same 2252 launches per chunk, spec 20 §12). Other values are refused. |
| `B70_Q4_MTP_NORM` | `b70-serve --spec mtp`, Qwen3.8-Flash-Next | `single` | the MTP head's `pre_fc_norm_hidden` (spec 21 decision 4): `single` is vLLM's one RMS over the 4 streams (10240), `per_stream` one per stream; read at construction, other values are refused |
| `B70_Q4_ATTN` | `b70-decode`, Qwen3.8-Flash-Next | `flash` | Qwen3.8-Flash-Next's QSA decode and prefill attention (spec 21c / 21d): `eager` is the reference's bf16 chain over the selected positions (one launch for flash's two in each QSA layer: 767 launches a token at 48 layers instead of 779); prefill binds the `_EAGER` sparse attention for the rows past position 2050 (the same launches a chunk). Other values are refused. Never run on a B70. |
| `B70_BOLT_ATTN` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `flash` | Nemotron 3.5 Bolt's decode attention (spec 25c): `eager` binds the one-launch `mg_attn_eager` (the reference's bf16 chain, Decision 15): 645 launches a token instead of 653, plus a score row of 4 B x 32 heads x max_len. Other values are refused. Never run on a B70. |
| `B70_BOLT_FUSE` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `none` | the fused arms of Bolt's decode list (spec 25 §6.1, Decision 16): `router` (the router rows inside the MoE block's norm launch, -40 launches), `route` (the rank recomputed in every `nb_moe_up` work-group, -40), `conv` (the conv inside `nb_ssm`, -40), `all` (533 launches), or a comma list. Each arm is bitwise the as-built list; the default stays `none` until the box's A/B (queue row 60). Other values are refused. Never run on a B70. |
| `B70_BOLT_SSM` | `b70-decode` and `b70-serve`, Nemotron 3.5 Bolt | `fp32` | the Mamba SSM state's precision (Decision 2): `bf16` halves its bytes (rounded after every update) - a measured arm; the default changes only on the operator's ruling with row 60's numbers. Other values are refused. |
| `B70_BOLT_TIE_TOL` | `bolt_golden_test` | `1e-3` | the near-tie tolerance of Bolt's routing diagnostic (the 9th / 10th sel; spec 25 Decision 11), proposed until row 57's gap distribution on the real weights; a gap of exactly 0 is undetermined |
| `B70_BOLT_A_EXP` | `tools/oracle/bolt_ref.py`, `bolt_mamba_ops.py`, `bolt_fixture.py` | `torch` | `engine`: the reference computes `A = -exp(A_log)` with a float32 port of the engine's `exp_torch` (Sleef expf u10) instead of `torch.exp` - the kernels' bitwise fixture uses it (spec 25 §15); `bolt_ref.py run --a-exp` sets it |
| `B70_BOLT_SNAPSHOT` | `template_bolt_test`, `bolt_tokenizer_test`, `bolt_request_test` | the HF cache's urakozz/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-W4A16-g64-AutoRound-GPTQ at 8932aa8d (our export, the served checkpoint) | the snapshot whose `chat_template.jinja`, `tokenizer.json` and `tokenizer_config.json` the tests render and tokenise with (spec 25e; nothing of the model is vendored but its `config.json` and `generation_config.json`); the tests SKIP (77) without it. |
| `B70_GIT_SHA` | `b70-decode --bench` | `unknown` | the commit printed in the bench row. The box's tree has no `.git`, so `tools/bench_decode.sh` sets it. |

The scripts in `tools/`:

| variable | read by | default | what it does |
|---|---|---|---|
| `BOX` | `box.sh`, `bench_decode.sh`, `serve_bench.sh`, `box_validate.sh` | none, required | ssh target (`user@host`) of the machine with the GPU. Set it in the environment or in the untracked `tools/box.env` (copy `tools/box.env.example`); the environment wins. |
| `REMOTE_DIR` | the same | from the branch | the tree on the box, relative to its `$HOME`: `b70-inference-server` for main, `b70/trees/spec7a` for a `spec7a-*` branch, `b70/trees/research` for `research-*`, `b70/trees/<branch>` for any other (`tools/box_dir.sh`); `box_validate.sh` uses `b70-inference-server-validate`. |
| `BOX_SUFFIX` | the same | from the branch | overrides that suffix; `none` forces the base tree. |
| `JOBS` | `box.sh` (and every script that builds through it) | `44` | parallelism of the remote build |
| `BUILD_DIR`, `CMAKE_ARGS` | `box.sh` | `build`, empty | build directory on the box and extra `-D` flags for the configure step |
| `MODEL`, `DEPTH`, `TG`, `RUNS`, `PP`, `PP_CHUNK`, `PP_BACKEND`, `MAX_LEN` | `bench_decode.sh` | Qwen3.8's repo id, `4096`, `256`, `3`, the rest unset | the same as its flags (`--model`, `--depth`, ...). Unset adds nothing to the command, so it stays the one every earlier row used. It also forwards `ZE_AFFINITY_MASK`. |
| `MODEL`, `PORT`, `SERVED_NAME`, `RUNS` | `serve_bench.sh` | Qwen3.8's snapshot in the box's HF cache, `8000`, `b70`, `3` | the served checkpoint (plain: `--spec off`), its port, the model name the requests carry, and the number of llama-benchy invocations |
| `SNAP_QWEN`, `SNAP_ORNITH`, `SNAP_K2`, `SNAP_KOLIBRI`, `DEVICE`, `PORT`, ... | `box_validate.sh` | the four checkpoints, `0`, `8013` | the checkpoints, card and port the validation run uses; the full list (rounds, baseline, data directories) is in `tools/box_validate.sh --help` |
| `ORACLE_MODEL` / `ORACLE_SNAP`, `ORACLE_THREADS`, `ORACLE_IMAGE`, `HF_CACHE`, `REPO_DIR` | `tools/oracle/run_in_container.sh` and the scripts that call it (`golden.sh`, `check.sh`) | Qwen3.8 in the HF cache, all threads, the script's reference image, `~/.cache/huggingface`, this tree | the CPU reference's checkpoint (a repo directory in the cache, or an absolute snapshot path, which wins), its CPU threads, the container, and the two mounts. `golden.sh` adds `OUT_DIR` (`oracle-out`), `IDS_DIR` (`tests/golden/prompts`) and `PROMPTS` (`prose code cjk`). |

Test-only (the checkpoint paths the tests load are CMake options, e.g. `-DB70_TEST_SNAPSHOT`,
`-DB70_ORNITH_SNAPSHOT`, `-DB70_K2_SNAPSHOT`, not environment variables):

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_LONGCTX_TESTS` | the `longctx` tests | unset: SKIP | `1` runs the long-context MTP variants (`mtp_verify_{32k,128k}_test`, `mtp_gpu_{32k,128k}_test`) |
| `B70_K2_TIE_TOL` | `k2_golden_test` | `1e-3` | the near-tie tolerance of K2's routing diagnostic, proposed until the reference run measures it |
| `B70_KOL_TIE_TOL` | `kolibri_golden_test`, `kolibri_partial_test` | `1e-2` | the same for Kolibri-1 (spec 20c), proposed until `kolibri_oracle.sh` prints the gap distribution |
| `B70_Q4_TIE_TOL`, `B70_Q4_SEL_TOL` | `qwen4exp_golden_test`, `qwen4exp_partial_test` | `1e-3` | Qwen3.8-Flash-Next (spec 21c): the routing diagnostic's 10th / 11th and gate S's 512th / 513th near-tie tolerances, proposed until row 30 prints the gap distributions; a gap of exactly 0 is undetermined |
| `B70_SNAPSHOT_DIR` | `template_test` | Qwen3.8's snapshot under `$HF_HOME` | the snapshot whose chat template it renders |
| `B70_K2_TOKENIZER_JSON` | `k2_tokenizer_test` | the CMake option of that name | K2's `tokenizer.json`; absent, the test is skipped |
| `B70_SPLIT_VERBOSE` | `prefill_split_test` | unset | set: also print the worst KV row per attention layer of each non-bitwise split |

## Docs

[docs/README.md](docs/README.md) is the index and says what to read first. The
short list:

- [docs/01-hardware.md](docs/01-hardware.md), [docs/12-kernels.md](docs/12-kernels.md) for the card and the kernels
- [docs/04-architecture.md](docs/04-architecture.md) for the engine layout
- [docs/05-perf-model.md](docs/05-perf-model.md) for where the time goes
- [docs/14-golden-gate.md](docs/14-golden-gate.md) for how correctness is proven
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for every measured row and its grade
- [docs/superpowers/specs/](docs/superpowers/specs/) for the designs behind the bigger changes

## Licence

Apache 2.0. See [LICENSE](LICENSE).

Written by Jürgen Kozyrev.
