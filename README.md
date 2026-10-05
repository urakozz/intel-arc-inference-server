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
the card and writes every prefill through to a pinned host store (`--prefix-cache-gb`,
default 32), so a turn at 60k tokens of history reaches its first token in 1.23 s instead
of re-prefilling for 45 s, and 1.58 s when a side request evicted the card in between.

Checkpoint is `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, int4 weights with
group size 64, bf16 activations (why g64: see below). vLLM serves the exact same
files, which is what makes the comparison fair. Full protocol and every row is in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Scope

One model family and one math path, on purpose. **So far the engine is built and
tuned for one model only: the dense Qwen3.8-27B** (no mixture of experts). It is a
hybrid: 48 gated delta net layers and 16 full attention layers. Everything here
is built around that shape and around W4A16: the kernel shapes, the captured
decode list and the tuning tables are that model's. Other checkpoints of the
same architecture load, but nothing is tuned for them, and MoE models (the next
candidate is K2-Horizon) are not supported yet. Specialisation is the whole
strategy, since a general engine cannot hardcode the things this one hardcodes.

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

sycl-tla stays selectable as a reference backend with `--pp-backend sycl-tla`,
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

./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench --pp 4096 --tg 256
./build/src/cli/b70-serve  urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ   # OpenAI compatible endpoint
```

Both CLIs take the repo id and resolve it through the local cache
(`$HF_HOME` or `~/.cache/huggingface`, revision from `refs/main`); a snapshot
directory path works too.

### Serving

```sh
# long agentic sessions (opencode and similar): the largest context that fits the card
# (--max-len auto, the default: ~192k with the int8 head, derived), prefix cache on (default),
# the prompt-end snapshot one id early so each turn restores where the next one diverges
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --host 0.0.0.0 --port 8000 --served-name qwen3.8 --prefix-split-last

# the same with MTP speculative decoding, K chosen per request (auto context: ~163k, derived)
./build/src/cli/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ \
    --served-name qwen3.8 --mtp auto

# measure it the way the vLLM rows were measured
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model qwen3.8 \
    --pp 4096 --tg 256 --concurrency 1 --depth 1 --no-cache --exact-tg --latency-mode generation
```

`b70-serve` flags:

| flag | default | what it does |
|---|---|---|
| `<snapshot-or-repo>` | required | HF repo id resolved in the local cache, or a snapshot directory; never downloads. The model (Qwen3.8 or Agnes 3.0 Flash) comes from its `config.json`. |
| `--host H` | `0.0.0.0` | listen address |
| `--port P` | `8000` | listen port |
| `--served-name NAME` | `b70` | model name in the OpenAI API (`/v1/models`, the `model` field) |
| `--max-len auto\|L` | `auto` | context capacity. `auto`: the largest multiple of 256 whose memory plan (weights + KV + decode state + prefill scratch, the MTP head's buffers when `--mtp` is on) fits the card with `--mem-reserve-gb` left over, capped by the checkpoint's trained context (`max_position_embeddings`, 262144); the chosen length and the plan are printed at startup (spec 6 §10). Derived at the default reserve with the int8 head: Qwen3.8 ~192k, ~163k with `--mtp`; Agnes ~134k, ~110k with `--mtp`. The full 262144 needs int8 KV (spec 12b) or two cards (spec 16). `L`: any multiple of 256 up to the trained context, refused at startup with the plan's breakdown if it does not fit. |
| `--mem-reserve-gb G` | `1.5` | device memory the plan leaves free for what it does not count: the driver, kernel modules and command lists, allocator slack (an estimate, not yet confirmed on the box) |
| `--device N` | `ONEAPI_DEVICE_SELECTOR`, else 0 | which GPU |
| `--queue N` | `4` | requests waiting behind the running one before new ones are refused |
| `--prefix-cache-gb N` | `32` | pinned host prefix cache in GiB; `0` = off, every request prefills in full (spec 7) |
| `--prefix-split-last` | off | On/off, chat requests only. A thinking model's next turn repeats this prompt but diverges at its **last** id: the template ended this prompt with `<think>\n`, and the history re-renders that turn as `<think>\n\n</think>` when the client drops the reasoning (opencode does). By default the prompt-end snapshot sits one id past that point, so the next turn falls back to the 2048-id block below and re-prefills up to 2047 ids it already had (~1 s per turn). With the flag the prompt is prefilled to len - 1, snapshotted there, and the last id runs as one decode step, so the next turn (or a retry of the same prompt) restores exactly where it diverges. The cost: that one id goes through decode's kernels, so a near-tie first token can differ from a run without the cache (same quality, not bitwise reproducible). **On for agentic sessions; off for benchmarks and the golden gates.** |
| `--mtp off\|1\|2\|3\|auto` | `off` | Speculative decoding with the model's built-in MTP head: the head guesses the next few tokens, the main model checks all the guesses in one pass, and every correct guess is a token for free (output is unchanged either way). A number is a fixed count of guesses per step. `auto` picks 0-3 per step from how often that request's guesses were right recently: about 3 on tool calls and code, 1 on prose, 0 when guessing does not pay. Loads the head (+1.4 GB at 16k context, ~2.3 GB at 128k, derived). `0` is the same as `off`. |
| `--mtp-max N` | `3` | With `--mtp auto`: the most guesses it may pick (1-3). |
| `--lm-head bf16\|int8` | `int8` | the output head: int8 per row, quantised at load (spec 9), or the checkpoint's bf16 |
| `--pp-backend B` | `l0-int8` | prefill GEMMs: `l0-int8` (Hadamard rotated int8, spec 5), `l0` (bf16 on the Level Zero list), `sycl-tla` (reference, optional component) |
| `--log-requests DIR` | off | write `DIR/NNNNNN.json` per request: timings, body, prompt and generated ids, text |

Advanced, for calibration only: `--mtp-cost "verify=1,1.18,1.56,1.79;draft=0.13,0.26,0.38"` replaces the
cost table `--mtp auto` decides with (a check of 1-4 tokens and 1-3 guesses, in units of one normal
step; the defaults depend on `--lm-head`). Leave it alone unless `probe_mtp_steps` measured different
costs on your card (spec 8 §10).

### Benchmarking and checking

```sh
# the BENCHMARKS.md rows: prefill 4096 and decode 256 at the prefilled depth
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --bench --pp 4096 --tg 256

# greedy ids in, ids out (tools/oracle/tokenize.py writes the --ids file)
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --ids prompt.ids --n 64 --prefill > out.ids

# per-launch anatomy of the decode step
./build/src/cli/b70-decode urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --profile --depth 4096 --steps 32 --repeats 5
```

`b70-decode` flags:

| flag | default | what it does |
|---|---|---|
| `--bench` | - | ingest `--depth` synthetic ids (or prefill `--pp N`), time `--tg` generated ones, print a markdown row |
| `--depth D` | `4096` | `--bench` / `--profile`: context length before decoding, ingested one replay per id |
| `--pp N` | - | `--bench`: prefill N synthetic ids instead (exclusive with `--depth`) and print a second row with the prefill time |
| `--tg N` | `256` | `--bench`: ids to generate |
| `--ids FILE` | - | whitespace-separated prompt ids; generated ids go to stdout, one per line |
| `--n N` | - | `--ids`: ids to generate, greedily |
| `--prefill` | off | `--ids`: run the prompt through the chunked prefill instead of one replay per id |
| `--pp-chunk C` | `2048` | positions per prefill chunk |
| `--pp-backend B` | `l0-int8` | as for `b70-serve` |
| `--lm-head bf16\|int8` | `bf16` | bf16 keeps the rows byte-matched with vLLM; int8 rows are marked `int8-head` |
| `--profile` | - | replay `--steps` instrumented decode steps and print the per-launch anatomy (never a bench row) |
| `--steps N` | `32` | `--profile`: steps per session |
| `--repeats R` | `1` | `--profile`: independent sessions, for the spread (R >= 5 before claiming a delta under ~0.3 ms) |
| `--max-len L\|auto` | `16384` | as for `b70-serve`, but 16384 stays the default so bench rows stay comparable |
| `--mem-reserve-gb G` | `1.5` | as for `b70-serve` |
| `--device N` | `ONEAPI_DEVICE_SELECTOR`, else 0 | which GPU |

`tools/box.sh` builds and tests on a remote machine over ssh, which is how I
work day to day. Set `BOX=user@host` before using it.

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
