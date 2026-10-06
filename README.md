# b70-inference-server

An LLM inference engine I wrote from scratch for the Intel Arc Pro B70, talking
straight to Level Zero and OpenCL C. No PyTorch, no vLLM, no vendor kernel
library in the hot path.

I started it to learn what this silicon actually does, and to see whether a
small specialised engine could beat a general one on the same box. It can, at
least for prefill.

## Where it stands

Measured on one Arc Pro B70, same checkpoint and same prompt on both sides,
median of three runs on an idle machine:

| | this engine | vLLM | |
|---|---:|---:|---|
| prefill, 4096 tokens | **2125.12 t/s** | 1610.04 t/s | we are 32.0% ahead |
| decode, 256 tokens | 29.45 t/s | 31.01 t/s | we are 5% behind |
| decode, int8 `lm_head` (our serving default) | 31.21 t/s | - | not byte-matched |

Rows above are byte-matched: both engines read the checkpoint's bf16 `lm_head`.
`b70-serve` quantises that head to int8 per row at load (spec 9, gated on the golden
prompts and the tool-call set), which is 31.21 t/s against 29.38 t/s for bf16 in the same
interleaved run (docs/BENCHMARKS.md, "int8 lm_head"); vLLM cannot load that head, so the
two are not compared. All three decode rows predate spec 10's decode attention v2 (they
ran attn.cl's v1 pair); v2 is 5.9% faster per step at depth 4096 (docs/BENCHMARKS.md,
"Decode attention at depth"), not yet re-measured in this table.

Prefill went from 1406 t/s to 1670 t/s over a couple of weeks of kernel work,
then to 2104 t/s by moving every int4 linear onto the card's int8 matrix engines
(spec 5: Hadamard-rotated int8 activations against per-channel int8 weights
rebuilt on the fly from the int4 checkpoint, at W4A16 accuracy; see
docs/probe-w4a8-2026-09-23.md sections 14 and 15), and to 2125 t/s with the
fused flash attention of spec 6.
Decode has been parked for a while and vLLM is still ahead there, so that is
the honest picture: good prefill, decode still to do.

Context now goes to 131072 tokens (`--max-len 131072`, 28.1 GB of the card's 32.5):
prefill attention is a fused flash-attention kernel of our own, and a passkey
stated once in 120k tokens of filler is found at 5, 50 and 95% depth (spec 6).
Prefill attention is still slow at depth (747 t/s at 128k); decode attention
was rebuilt for depth (spec 10): 20.1 t/s decode at 128k, from 10.2 (captured
decode step, bf16 `lm_head`).

Long agentic sessions reuse their history (spec 7): the server keeps the last session on
the card and writes every prefill through to a pinned host store in system RAM (`--prefix-cache-gb`,
by default sized from the machine's RAM, up to 32 GiB), so a turn at 60k tokens of history reaches its first token in 1.23 s instead
of re-prefilling for 45 s, and 1.58 s when a side request evicted the card in between.

Checkpoint is `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, int4 weights with
group size 64, bf16 activations (why g64: see below). vLLM serves the exact same
files, which is what makes the comparison fair. Full protocol and every row is in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
  (docs/probe-w4a8-2026-09-23.md §15.2).
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
makes no SYCL call and never waits on the host. Attention and the delta net
scan are ours too.

sycl-tla stays selectable as a reference backend with `--prefill-backend sycl-tla`,
and the whole thing also builds with no SYCL component at all.

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

Details in [docs/probe-dpas-rates-2026-09-22.md](docs/probe-dpas-rates-2026-09-22.md).

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
- Both backends must leave identical state: KV caches, delta net state,
  convolution ring, logits and the control block, compared word for word.
- Determinism is checked bitwise across runs from reset.
- Benchmarks only count as a record when the box is provably idle, and the
  harness measures that itself rather than taking your word for it.

86 tests in the suite as of now.

One honest exception: the delta net scan uses approximate split bf16 arithmetic,
so it is gated on tokens and state cosines instead of bitwise equality. It is
the only default in the prefill path that is not bit identical to what it
replaced.

## Build and run

The engine builds with g++ and CMake. The SYCL reference backend is optional and
needs icpx.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# the engine never downloads; fetch the checkpoint into the Hugging Face cache once
uvx --from huggingface_hub hf download urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ

./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench --prefill-length 4096 --tg 256
./build/src/cli/b70-serve  urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ   # OpenAI compatible endpoint
```

Both CLIs take the repo id and resolve it through the local cache
(`$HF_HOME` or `~/.cache/huggingface`, revision from `refs/main`); a snapshot
directory path works too.

Commands for each model (Qwen3.8, Agnes 3.0 Flash, Ornith 1.5, K2-Horizon, and Kolibri-1 - decode and
prefill (`l0` only) on two cards, built blind, needs spec 20b's checkpoint), what each
one supports, its limits, and which configurations have actually run on a B70:
[docs/19-running-models.md](docs/19-running-models.md).

### Serving

```sh
# long agentic sessions (opencode and similar): the largest context that fits the card
# (--max-len auto, the default: ~201k with the int8 head, derived), prefix cache on (default),
# the prompt-end snapshot one id early so each turn restores where the next one diverges
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --host 0.0.0.0 --port 8000 --served-name qwen3.8 --prefix-split-last

# the same with MTP speculative decoding, K chosen per request (auto context: ~170k, derived)
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --served-name qwen3.8 --mtp auto

# measure it the way the vLLM rows were measured
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model qwen3.8 \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

`b70-serve` flags:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | HF repo id resolved in the local cache, or a snapshot directory; never downloads. The model (Qwen3.8, Agnes 3.0 Flash, Ornith 1.5 or K2-Horizon) comes from its `config.json`. **K2-Horizon** is served by its own engine since spec 18d - built without the card, its box gates pending (queue row 25): no MTP head (`--mtp`, `--spec mtp\|lookup` refused), prefill on `l0` only, its `chat_template_kwargs` `tool_call_format` (`xml`, `xml_typed`, `json`) and `reasoning_effort` (`high`, `medium`, `low`) reach its template, the prefix cache keeps KV-only snapshots; `--max-len auto` ~46k with bf16 KV, ~90k with `--kv-cache int8` (int8 head, derived). Qwen3.8, Agnes and Ornith can also be served on two cards (`--pp 2`, below); K2-Horizon is one card only. Kolibri-1 is refused until spec 20e (`b70-decode` runs it on two cards: decode, spec 20c, and prefill on `l0` only, spec 20d). |
| `--host H` | `0.0.0.0` | listen address |
| `--port P` | `8000` | listen port |
| `--served-name NAME` | `b70` | model name in the OpenAI API (`/v1/models`, the `model` field) |
| `--max-len auto\|L` | `auto` | context capacity. `auto`: the largest multiple of 256 whose memory plan (weights + KV + decode state + prefill scratch, the MTP head's buffers when `--mtp` is on, the `--draft-vocab` head) fits the card with `--mem-reserve-gb` left over, capped by the checkpoint's trained context (`max_position_embeddings`, 262144); the chosen length and the plan are printed at startup (spec 6 §10). Derived at the default reserve with the int8 head: Qwen3.8 ~201k, ~170k with `--mtp`; Agnes ~140k, ~114k with `--mtp`. The full 262144 needs `--kv-cache int8` (spec 12b; Qwen3.8 then reaches it with or without `--mtp`, derived) or two cards (spec 16). `L`: any multiple of 256 up to the trained context, refused at startup with the plan's breakdown if it does not fit. |
| `--mem-reserve-gb G` | `1.5` | device memory the plan leaves free for what it does not count: the driver, kernel modules and command lists, allocator slack (an estimate) |
| `--device N` | `ONEAPI_DEVICE_SELECTOR`, else 0 | which GPU (not with `--pp 2`) |
| `--pp N`, `--pipeline-parallel-size N` | `1` | Serve on one card (`1`) or split the model's layers over two (`2`, spec 16d; GPUs 0 and 1 of what `ZE_AFFINITY_MASK` shows). One answer is not faster on two cards, but each card holds about half the model, so the context can be longer: Qwen3.8 with the int8 head reaches its full 262144 tokens, with or without `--mtp` (one card: ~201k, ~170k with `--mtp`; derived), and a prompt that misses the prefix cache prefills on both cards at once (spec 16c). Everything else works as on one card: the prefix cache, `--prefix-split-last`, `--mtp` / `--spec mtp` (the MTP head runs on the second card, with its own copy of the embedding, 2.54 GB), `--spec lookup`, `--draft-vocab`, `--kv-cache int8`, `--lm-head`; `--max-len auto` fits both cards, each with `--mem-reserve-gb`, and the log prints both cards' memory. Qwen3.8, Agnes and Ornith; K2-Horizon and Kolibri-1 are refused, as are `--device`, `--prefill-backend sycl-tla` and `B70_PREFILL_ATTN=composed`. Built without the cards: not yet run on them (queue row 27). |
| `--pipeline-split auto\|N` | `auto` | With `--pp 2`: where the layers are cut, as for `b70-decode` (the MTP head and the prefix cache's buffers count: `auto` moves layers to the first card when the head is on the second) |
| `--pipeline-handoff copy\|peer` | `copy` | With `--pp 2`: how the hidden state crosses, as for `b70-decode` |
| `--queue N` | `4` | requests waiting behind the running one before new ones are refused |
| `--prefix-cache-gb auto\|N` | `auto` | The prefix cache's size in GiB of **system RAM** (pinned host memory, not the card's VRAM): every prefill is written through to it so a later request with the same history restores instead of re-prefilling (spec 7). Pinned pages cannot be swapped, so `auto` takes the smallest of 32 GiB, half the machine's RAM, and the RAM available after the model is loaded minus 8 GiB, and turns the cache off below 4 GiB; the startup line says what it chose and why. `N` pins exactly N GiB (with a warning if that is more than is available); `0` = off, every request prefills in full. |
| `--prefix-split-last` | off | `on`/`off`, chat requests only. A thinking model's next turn repeats this prompt but diverges at its **last** id: the template ended this prompt with `<think>\n`, and the history re-renders that turn as `<think>\n\n</think>` when the client drops the reasoning (opencode does). By default the prompt-end snapshot sits one id past that point, so the next turn falls back to the 2048-id block below and re-prefills up to 2047 ids it already had (~1 s per turn). With the flag the prompt is prefilled to len - 1, snapshotted there, and the last id runs as one decode step, so the next turn (or a retry of the same prompt) restores exactly where it diverges. The cost: that one id goes through decode's kernels, so a near-tie first token can differ from a run without the cache (same quality, not bitwise reproducible). **On for agentic sessions; off for benchmarks and the golden gates.** |
| `--mtp off\|1\|2\|3\|auto` | `off` | Speculative decoding with the model's built-in MTP head: the head guesses the next few tokens, the main model checks all the guesses in one pass, and every correct guess is a token for free (output is unchanged either way). A number is a fixed count of guesses per step. `auto` picks 0-3 per step from how often that request's guesses were right recently: about 3 on tool calls and code, 1 on prose, 0 when guessing does not pay. Loads the head (+1.4 GB at 16k context, ~2.3 GB at 128k, derived; Ornith's MoE head +0.5 GB of weights, its experts quantised to int4 at load, and the full 262144 context still fits with it). On Ornith `auto` uses Qwen3.8's cost table until the card measures Ornith's (spec 15e). `0` is the same as `off`. |
| `--mtp-max N` | `3` | With `--mtp auto`: the most guesses it may pick (1-3). |
| `--spec off\|mtp\|lookup` | `off` | Which speculative proposer guesses the next tokens. `mtp` is the MTP head above (`--mtp auto` unless `--mtp` says otherwise). `lookup` needs no extra model: the guesses are the continuation of the longest earlier repeat of the last few tokens in the request (agentic edits re-quote files and tool output). The main model still checks every guess, so output is unchanged. On the tool-call set it projects ~1.44x alone, below `--mtp auto` (spec 19e; the opencode recording decides). It is meant for models without an MTP head, but today it checks its guesses with the MTP head's verify lists, so it loads the head and works only on a model that has one (not K2-Horizon yet). Not yet run on the card. `--spec-min-match N` (default 3) is the shortest repeat that counts. |
| `--lm-head bf16\|int8` | `int8` | the output head: int8 per row, quantised at load (spec 9), or the checkpoint's bf16 |
| `--draft-vocab off\|32k\|64k\|128k` | `off` | With `--mtp`: the head only guesses from the 32k / 64k / 128k most useful tokens (the chat and tool-call tags always, then `--draft-vocab-ids`, then the most common ones), so each guess reads a fraction of the output head. The check still uses the full vocabulary, so output never changes; a guess the subset cannot make is simply a miss. About 4-5 % faster MTP steps at 128k (derived, not yet measured). Costs 0.17 / 0.34 / 0.67 GB of memory with the int8 head, twice that with bf16, and that memory comes out of the `--max-len auto` context: at 128k, Qwen3.8 with `--mtp` gets ~160k instead of ~170k with the int8 head and ~132k instead of ~152k with bf16 (derived). Spec 8 §11. |
| `--draft-vocab-ids FILE` | - | With `--draft-vocab`: your own ranking of the most useful tokens, one id per line, most frequent first. Make it with `tools/draft_vocab/rank.py` from the logs `--log-requests` writes. |
| `--prefill-backend B` | `l0-int8` | prefill GEMMs: `l0-int8` (Hadamard rotated int8, spec 5), `l0` (bf16 on the Level Zero list), `sycl-tla` (reference, optional component). `--pp-backend` until 2026-10-06 |
| `--kv-cache bf16\|int8` | `bf16` | How the context (the KV cache) is stored. `int8` keeps every cached key and value as 8-bit numbers plus one small scale per token and head, after mixing each row with a fixed rotation that spreads out the few very large values (the 12a probe's `rotkv`, spec 12 §8); attention mixes it back. That halves the cache: 32 KiB + 256 B per token on Qwen3.8 instead of 64 KiB, so `--max-len auto` reaches the full trained context, 262144 tokens, with or without `--mtp` (Agnes: 262144, ~226k with `--mtp`; derived), and the prefix cache's KV blocks halve too (more history fits in the same RAM). Accuracy: on Agnes (the stand-in) its error was below bf16 attention's own rounding error; the gates on Qwen3.8 and on the card are **pending** (spec 12 Q2-Q5), which is why the default stays `bf16`. Needs `--prefill-backend l0` or `l0-int8` (refused with `sycl-tla`). Spec 12b. |
| `--log-requests DIR` | off | write `DIR/NNNNNN.json` per request: timings, body, prompt and generated ids, text |

Advanced, for calibration only: `--mtp-cost "verify=1,1.18,1.56,1.79;draft=0.13,0.26,0.38"` replaces the
cost table `--mtp auto` decides with (a check of 1-4 tokens and 1-3 guesses, in units of one normal
step; the defaults depend on `--lm-head`). Leave it alone unless `probe_mtp_steps` measured different
costs on your card (spec 8 §10).

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
| `--bench` | - | ingest `--depth` synthetic ids (or prefill `--prefill-length N`), time `--tg` generated ones, print a markdown row |
| `--depth D` | `4096` | `--bench` / `--profile`: context length before decoding, ingested one replay per id |
| `--prefill-length N` | - | `--bench`: prefill N synthetic ids instead (exclusive with `--depth`) and print a second row with the prefill time. `--pp N` until 2026-10-06 (`--pp` is the pipeline-parallel size now, below) |
| `--tg N` | `256` | `--bench`: ids to generate |
| `--ids FILE` | - | whitespace-separated prompt ids; generated ids go to stdout, one per line |
| `--n N` | - | `--ids`: ids to generate, greedily |
| `--prefill` | off | `--ids`: run the prompt through the chunked prefill instead of one replay per id |
| `--mtp off\|1\|2\|3\|auto` | `off` | `--ids`: greedy decoding with the MTP head's guesses, as `b70-serve --mtp` (same ids as without it; stderr adds the acceptance). Not yet run on the card |
| `--prefill-chunk C` | `2048` | positions per prefill chunk (`--pp-chunk` until 2026-10-06) |
| `--prefill-backend B` | `l0-int8` | as for `b70-serve` (`--pp-backend` until 2026-10-06) |
| `--lm-head bf16\|int8` | `bf16` | bf16 keeps the rows byte-matched with vLLM; int8 rows are marked `int8-head` |
| `--kv-cache bf16\|int8` | `bf16` | as for `b70-serve`; int8 rows are marked `int8-kv`. **K2-Horizon too** (spec 18e, built blind, on-card gates pending): the same scheme at head_dim 128 - 97.5 KiB per token instead of 192 KiB, so `--max-len auto` gives ~92k (bf16 head) / ~98k (`--lm-head int8`) decode-only and ~84k / ~90k with `--prefill` (bf16 KV: ~47k / ~50k and ~43k / ~46k; derived) |
| `--profile` | - | replay `--steps` instrumented decode steps and print the per-launch anatomy (never a bench row) |
| `--steps N` | `32` | `--profile`: steps per session |
| `--repeats R` | `1` | `--profile`: independent sessions, for the spread (R >= 5 before claiming a delta under ~0.3 ms) |
| `--max-len L\|auto` | `16384` | as for `b70-serve`, but 16384 stays the default so bench rows stay comparable |
| `--mem-reserve-gb G` | `1.5` | as for `b70-serve` |
| `--device N` | `ONEAPI_DEVICE_SELECTOR`, else 0 | which GPU |
| `--pp N`, `--pipeline-parallel-size N` | `1` | Pipeline parallel, vLLM's flag and meaning; N is 1 or 2. `2` splits the model's layers over two cards: the first card runs the first part of the layers with the embedding, the second card the rest with the output head, and each token's hidden state crosses once per step. One answer is not faster this way (the layers still run one after the other), but each card holds about half the model, so the context can be longer: Qwen3.8 reaches its full 262144 tokens with the bf16 KV cache (derived). With `--prefill` or `--bench --prefill-length N` the prompt is prefilled on both cards at once: while the second card runs one 2048-token chunk through its layers, the first card already runs the next chunk through its own - about 1.8-1.9x one card's prefill at 32k tokens and beyond, 1.3x at 4k (derived, not yet measured), each card's busy time on stderr, results identical to one card; `--prefill-backend` `l0` or `l0-int8` only. Without them `--ids` reads the prompt one id at a time. `--mtp K\|auto` (`--ids`) runs with it since spec 16d (the MTP head on the second card); `--profile` and `--device` are refused with it. Uses GPUs 0 and 1 of what `ZE_AFFINITY_MASK` shows. Specs 16b (decode) and 16c (prefill), built without the cards: not yet run on them. Any other N is refused, so a pre-2026-10-06 `--pp 4096` (then the bench's prefill length, now `--prefill-length`) fails instead of asking for 4096 cards. |
| `--pipeline-split auto\|N` | `auto` | With `--pp 2`: where the layers are cut. `auto` picks the cut that leaves both cards holding about the same number of bytes (weights, context, buffers) - not the same number of layers, because the output head's card can be heavier; Qwen3.8 cuts at 32 of 64 with the bf16 head and at 29 with the int8 head (derived). `N`: the first card runs layers 0 to N-1. Each card needs at least one layer of each kind. |
| `--pipeline-handoff copy\|peer` | `copy` | With `--pp 2`: how the hidden state crosses. `copy`: the first card copies it to the second and signals it; `peer`: the first card's last kernel writes it straight into the second card's memory and raises a flag the second card waits on, with a time limit. Same results either way; which is faster is not yet measured. |

`tools/box.sh` builds and tests on a remote machine over ssh, which is how I
work day to day. Set `BOX=user@host` before using it. Without the box,
`tools/mac_check.sh` runs the host tests and syntax-checks the device code and
every kernel variant on a Mac ([docs/18-mac-checks.md](docs/18-mac-checks.md)).

### Environment variables

Read by `b70-serve` and `b70-decode`. A command-line flag always wins over the variable
that sets its default. The `B70_PREFILL_*` switches are for experiments and A/B timing;
leave them unset to run what the gates ran.

| variable | read by | default | what it does |
|---|---|---|---|
| `HF_HOME` | both CLIs | `~/.cache/huggingface` | where a repo id is looked up: `$HF_HOME/hub/models--<org>--<name>`, revision from `refs/main`. The same place `hf download` writes to. |
| `ONEAPI_DEVICE_SELECTOR` | both CLIs | unset: device 0 | which GPU, when `--device` is not given: `level_zero:N`, or `level_zero:*` for device 0. Anything else (lists, other backends) is refused at startup rather than silently running on another card. |
| `ZE_AFFINITY_MASK` | the Level Zero driver | unset: every card visible | hides cards from the process: `ZE_AFFINITY_MASK=1` leaves only the second card, which the engine then sees as device 0. `tools/box.sh` and `tools/bench_decode.sh` forward it to the box. |
| `B70_KV_CACHE` | both CLIs | `bf16` | the default of `--kv-cache` (`bf16` or `int8`); any other value is refused. This is how the tests run the same binaries over the int8 cache. |
| `B70_DECODE_ATTN` | both CLIs, Qwen-family models | `v2` | decode attention. `v1` is the pair from before spec 10: slower at depth, and it bakes the context length, so only the compiled lengths run (4096, 16384, 32768, 131072; `--max-len auto` takes the largest that fits), MTP only at 16384, no `--kv-cache int8`, not built for Ornith. |
| `B70_PREFILL_ATTN` | both CLIs, Qwen-family models | `flash` | prefill attention. `composed` is the reference path from before spec 6 (separate QK, softmax and PV launches): its scratch grows with the context (73,728 B per position on Qwen3.8, so `--max-len auto` drops to ~92k, derived), it refuses `--kv-cache int8`, and it is not built for Ornith. Any other value means `flash`. Read once per process. |
| `B70_PREFILL_GDN_SCAN` | prefill, Qwen-family models | `dpas_split` | the delta net scan. `dpas_split` is the split-bf16 DPAS scan, the one approximate step in the default path (gated on tokens and state cosines); `vector` is the older scan, 2.33x slower on that kernel (measured). Other values are refused. Read once per process. |
| `B70_PREFILL_GDN_SOLVE` | prefill, Qwen-family models | `vector` | `register` runs a barrier-free triangular solve: bit-identical, and measured slower (docs/prefill-gdn-solve-register-results.md). Read once per process. |
| `B70_PREFILL_SILU_FUSED` | prefill on `--prefill-backend l0` | on | `0` runs gate‖up and SiLU as two launches instead of one fused GEMM: bitwise the same, slower. For before / after timing in one binary. |
| `B70_PREFILL_REPLAY` | prefill (all models, Level Zero backends) | off | `1` records each prefill chunk's command list once and replays it from then on. Experimental; refused with `sycl-tla`. |
| `B70_PREFILL_PROFILE` | prefill | off | `1` times every prefill phase; `b70-decode --bench --prefill-length` prints the table on stderr (Qwen-family models; not yet for K2). It adds a host wait per phase, so never use it for a recorded row. |
| `B70_K2_ATTN` | `b70-decode`, K2-Horizon | `flash` | K2's attention, decode and prefill both. `eager` rounds scores and probabilities to bf16 where the reference does (spec 18 §10.1): 813 launches per token instead of 717, plus a score row of 4 B x 32 heads x max_len. The first box session decides which becomes the default. Other values are refused. |
| `B70_KOLIBRI_ATTN` | `b70-decode`, Kolibri-1 | `flash` | Kolibri-1's decode and prefill attention. `eager` is the reference's bf16 chain (spec 18 §10.1's switch, spec 20 §11): 856 launches per token instead of 756, plus a score row of 4 B x 48 heads x max_len; prefill binds the `_EAGER` flash builds (the same 2252 launches per chunk, spec 20 §12). Other values are refused. |
| `B70_GIT_SHA` | `b70-decode --bench` | `unknown` | the commit printed in the bench row. The box's tree has no `.git`, so `tools/bench_decode.sh` sets it. |

The scripts in `tools/`:

| variable | read by | default | what it does |
|---|---|---|---|
| `BOX` | `box.sh`, `bench_decode.sh`, `serve_bench.sh`, `box_validate.sh` | none, required | ssh target (`user@host`) of the machine with the GPU. Set it in the environment or in the untracked `tools/box.env` (copy `tools/box.env.example`); the environment wins. |
| `REMOTE_DIR` | the same | from the branch | the tree on the box, relative to its `$HOME`: `b70-inference-server`, `-spec7a` for a `spec7a-*` branch, `-research` for `research-*` (`tools/box_dir.sh`); `box_validate.sh` uses `-validate`. |
| `BOX_SUFFIX` | the same | from the branch | overrides that suffix; `none` forces the base tree. |
| `JOBS` | `box.sh` (and every script that builds through it) | `44` | parallelism of the remote build |
| `BUILD_DIR`, `CMAKE_ARGS` | `box.sh` | `build`, empty | build directory on the box (e.g. `build-nosycl`) and extra `-D` flags for the configure step |
| `MODEL`, `DEPTH`, `TG`, `RUNS`, `PP`, `PP_CHUNK`, `PP_BACKEND`, `MAX_LEN`, `LM_HEAD` | `bench_decode.sh` | Qwen3.8's repo id, `4096`, `256`, `3`, the rest unset | the same as its flags (`--model`, `--depth`, ...). Unset adds nothing to the command, so it stays the one every earlier row used. It also forwards `ZE_AFFINITY_MASK` and `B70_PREFILL_ATTN`. |
| `MODEL`, `PORT`, `SERVED_NAME`, `RUNS` | `serve_bench.sh` | Qwen3.8's snapshot in the box's HF cache, `8000`, `b70`, `3` | the served checkpoint, its port and name, and the number of llama-benchy invocations |
| `L0_INCLUDE` | `mac_check.sh` | a path in the script | directory holding Level Zero's `ze_api.h` (a `level-zero` checkout's `include/`); without it the host and syntax sections fail |
| `B70_JOBS` | `mac_check.sh` | all cores | parallelism of the Mac checks |
| `B70_MAC_CL_DEVICE` | `mac_check.sh --kernels` | the first Intel GPU, else the first GPU | the OpenCL device, by a substring of its name (e.g. `AMD`) |
| `SNAP_QWEN`, `SNAP_AGNES`, `SNAP_ORNITH`, `SNAP_K2`, `SNAP_KOLIBRI`, `DEVICE`, `PORT`, ... | `box_validate.sh` | the five checkpoints, `0`, `8013` | the checkpoints, card and port the validation run uses; the full list (rounds, baseline, data directories) is in `tools/box_validate.sh --help` |
| `ORACLE_MODEL` / `ORACLE_SNAP`, `ORACLE_THREADS`, `ORACLE_IMAGE`, `HF_CACHE`, `REPO_DIR` | `tools/oracle/run_in_container.sh` and the scripts that call it (`golden.sh`, `check.sh`) | Qwen3.8 in the HF cache, all threads, the script's reference image, `~/.cache/huggingface`, this tree | the CPU reference's checkpoint (a repo directory in the cache, or an absolute snapshot path, which wins), its CPU threads, the container, and the two mounts. `golden.sh` adds `OUT_DIR` (`oracle-out`), `IDS_DIR` (`tests/golden/prompts`) and `PROMPTS` (`prose code cjk`). |

Test-only (the checkpoint paths the tests load are CMake options, e.g. `-DB70_TEST_SNAPSHOT`,
`-DB70_ORNITH_SNAPSHOT`, `-DB70_K2_SNAPSHOT`, not environment variables):

| variable | read by | default | what it does |
|---|---|---|---|
| `B70_LONGCTX_TESTS` | the `longctx` tests | unset: SKIP | `1` runs the long-context MTP variants (`mtp_verify_{32k,128k}_test`, `mtp_gpu_{32k,128k}_test`) |
| `B70_K2_TIE_TOL` | `k2_golden_test` | `1e-3` | the near-tie tolerance of K2's routing diagnostic, proposed until the reference run measures it |
| `B70_KOL_TIE_TOL` | `kolibri_golden_test`, `kolibri_partial_test` | `1e-2` | the same for Kolibri-1 (spec 20c), proposed until `kolibri_oracle.sh` prints the gap distribution |
| `B70_TOKENIZER_JSON` | the Qwen3.8 tokenizer tests, `mac_check.sh`'s host section | Qwen3.8's snapshot under `$HF_HOME` | path to Qwen3.8's `tokenizer.json`; without it those tests are disabled on the Mac |
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
