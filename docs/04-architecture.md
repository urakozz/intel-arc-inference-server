# Architecture

**A Level Zero command list captured once and replayed for every decode token,
and a Level Zero walk of our own OpenCL C kernels for prefill.**

## Language and toolchain

- **C++17**, `g++` and CMake for everything on the decode path. `icpx` (oneAPI
  DPC++) only for the optional SYCL reference backend.
- **Level Zero** is the runtime. Device kernels are OpenCL C compiled offline
  with `ocloc` to device binaries and loaded with `zeModuleCreate`.
- **Python appears exactly once**, in `tools/`, for offline weight conversion,
  the CPU oracle and benchmark glue. Never in the serving path - Python *is* the
  overhead being removed.
- **Linux only.** No Windows or macOS runtime, and no portability shims.
- **The server never downloads models.** `hf download <repo>` puts them in the
  standard HuggingFace cache; the loader resolves a repo id against that cache
  (`refs/main` → snapshot) or takes an absolute snapshot path. A missing
  snapshot is an error naming the path, not a fetch.

## Why the decode step is the whole design

At batch 1 this card is **host-bound before it is kernel-bound**. Three
independent pieces of evidence from the vLLM work that preceded this project:

- Two entirely different kernel paths (MXFP4 and GPTQ-int4) land within **1%**
  of each other, 72.65 against 73.31 t/s. If kernels were the limit they would
  not agree that closely.
- Turning XPU graphs off costs about **3x**: the same 27B measured 10.55 t/s
  without graphs against 31.5 with them.
- Measured MBU sat at **50-63%** on those models. The card was idle waiting for
  work.

So the first product was not a faster GEMM. It was a decode step that issues
**zero host work per token**. That worked, and it is finished: 99.7% of a decode
step is now inside the GPU fence and the host spends 97 us per token. What
remains is kernel efficiency, which is a different problem (doc 05).

## Component layout

```
src/
  loader/      safetensors mmap, dedup by name, per-layer width metadata,
               repack into the canonical on-device layout, upload once
  model/       qwen3_5 graph description: layer types, shapes, weight bindings
  kernels/     OpenCL C: GEMV, GDN step, conv1d, attention, RMSNorm, RoPE,
               SiLU, sampling - plus kernels/prefill/ for the pf_* family
  l0/          the Level Zero wrappers: context, memory, modules, lists, events
  runtime/     allocations, the captured decode command list, KV cache,
               sequence state; runtime/prefill/ for the prefill walk
  server/      HTTP, OpenAI schema, SSE streaming
  sycl/        the optional sycl-tla reference backend, built only with icpx
  tokenizer/   leaf: HF `tokenizers` BPE encode/decode, chat template,
               streaming detokeniser - see doc 11. Host code, request path only
third_party/   pinned header-only minja and nlohmann/json
tools/         weight conversion, the CPU oracle, probes, benchmark drivers
```

Each directory should be understandable without reading the others. The loader
knows about checkpoint formats and nothing about kernels; the kernels know about
memory layouts and nothing about files.

## Execution model

Two distinct paths, deliberately not unified.

### Decode: one captured list, replayed

One token, fixed shapes, built **once** into a Level Zero command list and
replayed. Between replays the host updates **nothing in the list**.

The three values that change per token - KV write offset and position, sequence
length, and the current token id - live in a small **device-resident control
block** that every kernel reads and the sampler writes. The sampler stores its
argmax into `control.cur_token`; the embedding gather at the top of the next
replay reads it; the last kernel in the list increments `control.pos`. The
host's per-token work is `zeCommandQueueExecuteCommandLists`, a fence wait, and
one read of `cur_token` for the stream.

Every buffer address, kernel argument and dispatch dimension is baked in at
capture time, and the list is **byte-identical on every replay**. That is what
makes the capture-safety rule below testable: replay the same list twice from
the same state and diff the outputs.

The captured step is **774 kernels across 19 modules** - 48 GDN layers x 12 +
16 full-attention layers x 12 + 6 at the token boundary (`embed_gather`, the
final norm's two launches, `lm_head`, two argmax stages), i.e. 576 + 192 + 6.
`runtime::build` (`src/runtime/capture.cc`) is the one walk that binds them, and
`tests/runtime/replay_determinism_test` is the diff above run for real: same
list, same state, twice, bitwise on the token ids, on a per-layer residual tap
and on every persistent buffer, then once more from a re-zeroed state.

Level Zero's mutable-command-list extension exists and is the **fallback** for a
kernel that genuinely cannot read its state from memory, not the design.
Mutating arguments per token is host work, and host work is what this loop
exists to remove. A test proves arguments are resolved at append time, which is
what makes the control block sufficient.

A side effect worth keeping in view: once the host touches nothing between
tokens, submitting `N` tokens per `zeCommandQueueExecuteCommandLists` is a loop
unroll rather than a redesign. Streaming wants per-token granularity so nothing
does it today, but nothing precludes it either.

### Prefill: one in-order Level Zero list, no host waits

Dynamic shapes, so nothing is captured: every argument is resolved per launch.
A prefill chunk runs on **one in-order asynchronous Level Zero immediate command
list** (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, `MODE_ASYNCHRONOUS`) and makes **no
SYCL call and no host wait at all**. The int4 weights are dequantised into 1024
column slabs and multiplied by our own DPAS GEMM, interleaved on that one list.
Attention and the delta net scan are ours too.

That is the second design of this path, and the first one is worth recording
because the reason it was replaced is not the obvious one. The original prefill
called sycl-tla's GEMM from a SYCL queue built by interop from the same
`ze_context` and `ze_device`, which made the engine's `zeMemAllocDevice`
pointers valid USM inside it. **The two queues are not orderable against each
other on this device, and that is a measurement rather than a design
preference**: `zeDeviceGetCommandQueueGroupProperties` on the B70 reports
exactly two groups, ordinal 0 COMPUTE+COPY+COOPERATIVE with `numQueues = 1` and
ordinal 1 COPY-only with `numQueues = 1`, and `(0,1)` and `(1,1)` are both
refused with `ZE_RESULT_ERROR_INVALID_ARGUMENT`. There is one compute queue, the
two software queues share it, and no device-side cross-runtime dependency
primitive exists. So every L0-to-SYCL and SYCL-to-L0 boundary was a **host
wait**: 656 per chunk at a measured 22.35 us each.

Moving the GEMM onto the Level Zero list deleted all of them. Measured on an
idle card in one session, both backends:

| | Level Zero | sycl-tla control |
|---|---:|---:|
| pp t/s | **1502.83** | 1407.63 |
| launches per chunk | 8689 | 1201 |
| host waits per chunk | **0** | 656 |
| SYCL GEMM calls | **0** | 384 |
| prefill scratch | 35,651,584 B | 356,515,840 B |

Note the direction of the launch count: the Level Zero walk issues seven times
as many launches and has *fewer* host stalls, not more. The worry that 8,689
launches would cost host time in situ does not appear at all.

sycl-tla stays selectable with `--pp-backend sycl-tla` so the control can be
re-run at any time, and `tests/prefill/prefill_backend_equivalence_test` holds
the two paths **bitwise equal** on all five case families. The whole project
also builds with no SYCL component at all.

Two host-side rules follow from the list being asynchronous, and are stated once
here rather than at each site:

- **A host write to `Control` is always preceded by a drain.** A write racing a
  launch that reads `pos` would be a bug no test could reproduce.
  `Engine::prefill` drains before every chunk's `memcpy` of the ids and before
  both `Control` writes.
- **`Engine::prefill` lives in its own translation unit**, not in the runtime
  library every decode binary and test links, so a binary that never calls
  `prefill()` emits no reference to any prefill symbol. That is what keeps the
  decode path linkable without the SYCL component present.

### Attention under replay

A captured kernel cannot change its grid as the context grows, and the 16
full-attention layers read `seq_len` KV entries. Two honest options:

1. **Fixed grid over `max_model_len`, device-side early-out.** Every work-group
   reads `control.seq_len` and returns if its KV block lies beyond it. One list,
   simple.
2. **Context-bucketed lists** - capture one list per bucket and choose by
   `seq_len`. Fewer idle slots, several lists to keep correct.

**Option 1, and it was measured rather than assumed.** A grid sized for
`max_len` 16384 against one sized for 4096, at the same depth and the same live
work, costs **0.046 ms/token - 0.11% of a step** for 3072 extra work-groups,
about 15 ns per early-outed work-group. Quadrupling the grid again when the
attention block size was retiled made the launch 39% *faster*, so the fixed grid
is cheaper than that bound, not more expensive. Context-bucketed lists would buy
0.11% and cost a captured list per bucket, its memory, and a host-side branch on
context length in the one loop that currently has no branches at all. GDN layers
have no such problem: their state is fixed-size.

### Device selection

`--device N` is the single authoritative knob, and it governs **both** execution
paths so they always land on the same physical card. The precedence contract,
top wins:

1. **`--device N`** - selects GPU `N`. Where SYCL is involved the engine binds
   the *same* card by Level Zero handle interop (`sycl::make_device` from the
   `ze_device_handle_t`), not by trusting a second selector to agree.
2. **`ONEAPI_DEVICE_SELECTOR=level_zero:N`** - the default when the flag is
   absent. The SYCL runtime honours it natively; the raw Level Zero path parses
   it itself, because Level Zero does not read it.
3. **`ZE_AFFINITY_MASK`** - a Level Zero *driver*-level filter underneath both:
   it restricts which devices are enumerated at all. We respect it and never set
   it. Note that under a mask, `--device` and selector indices refer to the
   masked view, which is the same re-numbering the driver gives everyone.

The gotcha worth stating plainly: a machine with two cards will not necessarily
give you the same one twice unless the selection is expressed once and shared.
Two selectors that happen to agree today are not a contract.

### What makes this hard

Kernels must be **capture-safe**: no allocation, no host synchronisation, no
state that persists across replays. This is not hypothetical. The
`vllm-xpu-kernels` grouped GEMM had a global atomic tile counter reset *inside*
the kernel with no device-wide barrier. Under eager launch the timing
accidentally worked; under graph capture the counter carried the previous
replay's value and raced into out-of-bounds tile coordinates, producing
`UR_RESULT_ERROR_DEVICE_LOST` at batch > 1.

**Rule: any kernel that cannot be replayed byte-identically from the same
command list is a bug, not a limitation.** Design for replay from the first
kernel.

## The kernel families

Two families, dispatched on `M`, deliberately not one kernel pretending to serve
both.

**Decode** (`src/kernels/`), all bound by `capture.cc` at fixed `M = 1`:

| Kernel | Notes |
|--------|-------|
| `gemv` | the int4 mixers and MLPs, with per-shape split-K |
| `gemv_bf16` | the `a‖b` projections and the bf16 `lm_head` |
| `gdn_step` | the recurrent gated delta rule, fused with conv1d, q/k l2norm and the gated RMSNorm |
| `attn` | split-KV decode attention: per-block partials plus a merge |
| `prep` | residual fold, RMSNorm finish, SiLU-mul, the gated head |
| `embed_gather`, `argmax` | the token boundary |

**Prefill** (`src/kernels/prefill/`), every one a separate `ocloc` build, none
bound by `capture.cc`, `M` a runtime argument in all of them:

| Kernel | What it is |
|---|---|
| `pf_embed`, `pf_prep`, `pf_gated_head` | the decode `prep`/`embed` family at runtime `M` and no split-K |
| `pf_gemv_bf16` | the `a‖b` bf16 projection |
| `pf_dequant_slab` | one int4 linear into a bf16 column slab |
| `pf_gemm` | our own DPAS GEMM |
| `pf_attn_prep`, `pf_attn` | q/k norm, RoPE, the KV write, then the composed attention |
| `pf_gdn_conv`, `pf_gdn_wy`, `pf_gdn_scan` | the chunked delta net |

The decode kernels' split-K exists to buy hardware threads at `M = 1`. At
`M = 2048` a `[S][M][N]` partials rectangle at gate‖up's `N = 34816` would be
multi-terabyte, so prefill is a **second family** compiled without it rather
than a re-parameterisation of the first. The decode binaries are untouched by
every prefill change, byte for byte, which is why every prefill lever leaves the
decode row exactly where it was.

## Server

OpenAI-compatible, single stream. No scheduler, no paged KV, no continuous
batching. A ring KV buffer sized to `max_model_len` is sufficient when there is
exactly one sequence.

`src/server/` provides `/v1/models`, `/v1/chat/completions` and
`/v1/completions`, including OpenAI-shaped errors and SSE responses, against
`TokIface`, `TemplateIface` and `EngineIface`. `b70-serve` wires those
interfaces to the real checkpoint tokenizer and template and to one
`runtime::Engine`, using `Engine::prefill` for every request prompt. Requests
share one engine through a bounded FIFO; requests beyond the configured waiting
depth receive 503. The one intentional parsing deviation is that an absent
`temperature` means greedy sampling. The parser also accepts the `min_tokens`,
`ignore_eos`, `return_token_ids`, `seed` and `stream_options.include_usage`
extension fields.

`golden_server_test` proves the server reproduces `b70-decode --ids --prefill`'s
exact ids on the three golden prompts, both sides on the same engine path.

**HTTP costs nothing measurable.** Decode over HTTP measured 29.902 t/s against
the same session's CLI control of 29.31, i.e. 102.0%: the per-token host cost of
detokenising and writing an SSE frame overlaps the next replay and is not merely
under bar, it is not measurably present. Prefill over HTTP measured 2915.83 ms
for 4096 tokens against the CLI's 2919.6, a derived cost of -3.77 ms, which is
indistinguishable from zero against the run-to-run spread both sides show.

Host sampling (temperature, top-k, top-p) is implemented in the CLI's engine
adapter: the replay's fp32 logits row is read back once per sampled token
through a reused immediate command list, masked to `vocab_used`, top-k/softmax/
top-p sampled, and written into `cur_token[0]` - the same protocol `Control`
already supports for ingest. It measured **0.537 ms/token** against a 0.62 ms
bar registered before the measurement, so it ships on: any request with
`temperature > 0` is sampled. Greedy is never slowed, because the readback only
runs when a request asks for sampling.

Deferring batching is not a shortcut. It isolates the variable being tested,
which is per-token host and kernel cost, and keeps the first milestone
reachable.

### Prefix caching

Spec 7 ([specs/2026-09-27-spec7-prefix-caching-design.md](specs/2026-09-27-spec7-prefix-caching-design.md)):
one resident session on the card and a write-through store in pinned host RAM
(`--prefix-cache-gb`, default 32, `0` = off). The GDN state cannot be rewound, so
reuse needs snapshots of it: the store (`src/server/prefix_cache.{h,cc}`) keeps KV
blocks of 2048 positions and state snapshots (at every block end, at every prompt's
end, at every request's end) in a tree keyed by exact token ids, LRU over leaves.
A request continues the resident session when its ids extend it, otherwise restores
the deepest snapshot that is a prefix of the prompt (state plus the KV the card does
not already hold), and prefills only the tail; `usage.prompt_tokens_details.cached_tokens`
reports the restart point. opencode's side requests (titles, subagents) evict the card,
not the store. Results: [BENCHMARKS.md](BENCHMARKS.md), "Prefix caching (spec 7)".

The server owns the **tokenizer and chat template**. Both are host code on the
request path: encode once per request, detokenise once per token, off the card's
critical path and overlapping the next replay. Details in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).
