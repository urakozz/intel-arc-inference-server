# Architecture

**A Level Zero command list captured once and replayed for every decode token,
and a Level Zero walk of our own OpenCL C kernels for prefill.**

## Language and toolchain

- **C++20**, `g++` and CMake for everything. No SYCL and no `icpx` in the engine build.
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
remains is kernel efficiency, which is a different problem
([05-perf-model.md](05-perf-model.md)).

## More than one model

The layout below is Qwen3.8's, the focus model, used for agentic coding. Every other family has its own engine
and descriptor on the same foundations: Level Zero lists, our own OpenCL C kernels, the loader and the server.
The model is chosen from the checkpoint's `config.json`.

| model | what is different about it |
|---|---|
| Ornith 1.5 | a mixture of experts (MoE): a router picks a few expert networks for each token |
| Qwen3.8-Flash-Next | a larger MoE; today it loads its first layers only, until the whole model is split over two cards ([open-work.md](open-work.md), item 4) |
| K2-Horizon | MoE with routed value-expert attention; parked |
| Kolibri-1 | MoE with sliding-window attention layers; parked |
| Nemotron 3.5 Bolt | Mamba-2 state-space blocks, attention and a latent MoE; parked until Flash-Next is done |

Per-model status and commands: [18-running-models.md](18-running-models.md).

### Run modes

`b70-serve --parallel` picks how the cards are used. The default, `auto`, chooses by model and load:

- Qwen3.8 on two cards: `tp` for one request at a time, `replica` when `--max-num-seqs` is above 1.
- Any other model: one card when the model and the asked context fit, else `pp`.

The four modes:

- **One card** (`--parallel 1`): the whole engine on one card.
- **`tp`, tensor parallel** (Qwen3.8): every layer is split over both cards, half the attention heads and half
  the MLP on each. After each split layer the cards add their halves by writing straight into each other's
  memory over PCIe (peer-to-peer, P2P). The fastest way to serve one request, with the full context; one request
  at a time.
- **`replica`** (Qwen3.8 and Ornith): a complete copy of the model on each card, one engine per card, requests
  spread over the cards. The choice for 2 to 8 requests at once.
- **`pp`, pipeline parallel**: the first layers on one card and the rest on the other. About one card's speed,
  but each card holds only half the weights, so it serves models that are too big for one card or have no
  tensor-parallel path. The prefix cache and the MTP head work across the split.

Speed numbers per mode are in [BENCHMARKS.md](BENCHMARKS.md).

### Several requests at once: `--max-num-seqs`

`--max-num-seqs N` (vLLM's name) is how many requests run at once; the context follows from it. The default,
1, is one request at the largest context that fits. With N above 1 (Qwen3.8) a card holds *sequence slots*:
slot 0 gets the largest context that fits beside N - 1 further slots of 32768 positions each. One card takes up
to 4 slots; two cards run `replica` and split the slots between them, up to 8 (4 a card). `tp` and `pp` refuse
N above 1. An explicit `--max-len` that does not fit beside the other slots fails at startup, naming the plan;
`--parallel replica` alone means N = 2, one request a card. A request that runs into the end of its slot finishes
with `length` and is logged.
Tensor parallel with slots was built and ruled out (replica is faster for several requests at once;
[low-priority-work.md](low-priority-work.md)).

**Why fixed slots and not a shared KV pool.** The engine is tuned for one or two people
running local agentic sessions: a main session that needs its full context plus a few short helper requests. Each
slot's KV memory is reserved at startup, so the main session always has its context, nothing is ever preempted
(stopped and recomputed later to free memory), and no request races another for memory. vLLM's paged pool (one
pool of small KV blocks shared by all requests, `max_model_len` the most any one request may reach) uses memory
better when request sizes vary, at the cost of block tables in every attention kernel, a preemption policy and
the latency spikes it causes. That option was considered and postponed ([low-priority-work.md](low-priority-work.md)).

### How the two cards stay in step under tensor parallel

Tensor parallel has no host in the loop: each card runs its half of a layer and needs the other card's half
before it can go on. The cards hand these partial results to each other through a *landing buffer* that lives in
the reader's memory and that the other card writes into over P2P. The protocol
(`src/runtime/tp_protocol.h`, device side `src/kernels/tp.cl`) has three parts for each piece of data:

1. **The data.** The sender writes its partial result into the landing buffer.
2. **The stamp.** The sender then writes the number of this call into a trailer. The stamp fills a whole
   64-byte line (the unit in which the card's memory is written), 16 identical words.
3. **The flag.** Last, after a release fence, the sender writes the same call number into a flag, which sits on
   a separate page.

The reader spins on the flag until it shows the call number it expects, then reads the stamp. If the stamp does
not match, the flag arrived before its data and the reader reports a *stale* read instead of using old data; if
the flag never arrives within a bound, it reports a timeout. The first failure is recorded once, and later waits
skip their spin, so a lost peer costs one bound and not one per layer.

Why a whole line per stamp: every 64-byte line of the landing buffer has exactly one writer, and the data is
written as whole lines too. When sixteen 4-byte stamps shared one line, each written by a separate store, a
stamp could reach the reader after its flag: the reader saw the flag of call n with the stamp of call n - 2,
and one to three requests in a few dozen failed. With one stamp per line the failures went to zero.

## Speculative decoding

A plain decode step produces one token. Speculative decoding guesses several next tokens cheaply and checks all
of them in one batched step, the *verify*. Guesses that the model itself would have produced are kept, so the
greedy output is unchanged; the gain is that a verify of several rows costs much less than that many plain
steps. The guesses come from a *proposer*, chosen with `--spec`:

| `--spec` | proposer | notes |
|---|---|---|
| `auto` (default) | the MTP head on Qwen3.8 and Ornith, none on other models | no extra VRAM |
| `off` | none | |
| `mtp` | the checkpoint's MTP head (multi-token prediction: a small extra head that predicts up to 3 tokens ahead) | how many guesses to make is decided per step from the head's own confidence |
| `dflash` | DFlash, a separate small draft model that proposes a block of up to 7 tokens (Qwen3.8) | needs `--draft-model`; the drafter's weights take VRAM from the context, so it is never chosen by default |

**Prompt lookup** is a third source of guesses and is not a mode of its own. When the answer repeats text that is
already in the context (an edit that copies a file, a tool call that quotes its input), the text that followed
last time is a very good guess. It runs inside both proposers on one card with one request at a time (and
inside `dflash` under `tp`): when the lookup finds a long enough match, its copied tokens are verified for that
step and the proposer's own draft is skipped; otherwise the step is an ordinary MTP or DFlash one. It is on by
default; `B70_SPEC_LOOKUP=0` turns it off for comparisons.

**Where each proposer runs.**

| | one card | `replica` | `tp` | `pp` | several requests on a card |
|---|---|---|---|---|---|
| `mtp` | yes | yes | yes | yes | yes |
| `dflash` | yes, with prompt lookup | yes, a drafter on each card | yes, see below | yes | refused |

DFlash under `tp` splits the drafter over both cards. It is kept as an explicit choice but it is not faster than
DFlash on one card for one session, and the server says so at startup, recommending one card
(`--parallel 1 --spec dflash`), `replica` (a drafter per card) or `--spec mtp` on two cards.

Speculative output is checked against a float32 CPU reference where it differs from the plain run
([14-golden-gate.md](14-golden-gate.md), the f32 judge). Commands and measured speeds:
[18-running-models.md](18-running-models.md) and [BENCHMARKS.md](BENCHMARKS.md).

## The prompt pass backends

The prompt pass (prefill) runs in bf16 by default. `--prefill-backend l0-int8` is an opt-in alternative that
runs the MLP's linears with int8 activations ([17-int8-prefill.md](17-int8-prefill.md)).

## Component layout

```
src/
  loader/      safetensors mmap, dedup by name, per-layer width metadata,
               repack into the canonical on-device layout, upload once
  model/       the graph descriptions: layer types, shapes, weight bindings (Qwen3.8 and the other families)
  kernels/     OpenCL C: GEMV, GDN step, conv1d, attention, RMSNorm, RoPE,
               SiLU, sampling - plus kernels/prefill/ for the pf_* family
  l0/          the Level Zero wrappers: context, memory, modules, lists, events
  runtime/     allocations, the captured decode command list, KV cache,
               sequence state; runtime/prefill/ for the prefill walk; the tensor- and
               pipeline-parallel engines
  server/      HTTP, OpenAI schema, SSE streaming, the scheduler, the prefix cache
  cli/         b70-serve and b70-decode: arguments, the hardware check, the memory plan,
               the engine adapters (one card, slots, replica, tp, pp)
  tokenizer/   leaf: HF `tokenizers` BPE encode/decode, chat template,
               streaming detokeniser (11-tokenizer-and-chat-template.md). Host code, request path only
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
column slabs and multiplied by our own DPAS GEMM, interleaved on that one list (a short chunk, up to 384 rows, uses small tiles
that read the int4 words directly, `pf_gemm_small` / `pf_gemm_q4`). Attention and the delta net scan are ours too.

This is the second design of this path. The first is worth recording because the reason it was replaced is not
the obvious one. The original prefill called sycl-tla's GEMM from a SYCL queue built by
interop from the same `ze_context` and `ze_device`. **The two queues are not orderable against each other on
this device, and that is a measurement rather than a design preference**:
`zeDeviceGetCommandQueueGroupProperties` on the B70 reports exactly two groups, ordinal 0
COMPUTE+COPY+COOPERATIVE with `numQueues = 1` and ordinal 1 COPY-only with `numQueues = 1`, and `(0,1)` and
`(1,1)` are both refused with `ZE_RESULT_ERROR_INVALID_ARGUMENT`. There is one compute queue, the two software
queues share it, and no device-side cross-runtime dependency primitive exists. So every hand-over between
Level Zero and SYCL was a **host wait**: 656 per chunk at a measured 22.35 us each.

Moving the GEMM onto the Level Zero list deleted all of them, and the Level Zero walk, with seven times as many
launches (8,689 against 1,201 a chunk), was faster on an idle card (1502.83 against 1407.63 t/s) and used a
tenth of the scratch. The worry that thousands of launches would cost host time does not appear at all. The
sycl-tla path was removed after `prefill_backend_equivalence_test` had held the two bitwise equal.

Two host-side rules follow from the list being asynchronous, and are stated once
here rather than at each site:

- **A host write to `Control` is always preceded by a drain.** A write racing a
  launch that reads `pos` would be a bug no test could reproduce.
  `Engine::prefill` drains before every chunk's `memcpy` of the ids and before
  both `Control` writes.
- **`Engine::prefill` lives in its own translation unit**, not in the runtime
  library every decode binary and test links, so a binary that never calls
  `prefill()` emits no reference to any prefill symbol.

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

There is no device flag: the CLIs run on the cards the
driver shows - one card is device 0, two are devices 0 and 1 (`tp`, `replica`, `pp`).
**`ZE_AFFINITY_MASK`**, a Level Zero *driver*-level filter, decides which cards
are enumerated at all; we respect it and never set it, and under a mask every
index refers to the masked view (`ZE_AFFINITY_MASK=1` runs on the second card as
device 0).

Before any weight is read, the hardware gate (`cli/hardware_check.h`) checks what
the driver shows: one or two Intel Arc Pro B70s (vendor 0x8086, device id 0xE223;
the B65 and B60 are refused as not tested), the Xe-core count the kernels are
built for (32), and peer access both ways when there are two. The fit checks that
run before a device opens use the memory the gate read from the cards.

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

**Decode** (`src/kernels/`), all bound by `capture.cc` at fixed `M`: 1 for a plain step, 2 to 8 for a
speculative verify list (`M` = the rows checked at once), 16 for the drafter's blocks. The table is the `M = 1`
list. Every list at `M >= 2` replaces `gemv` / `gemv_i8w` by the small-batch GEMMs `gemm_sm` / `gemm_sm_i8w`
([12-kernels.md](12-kernels.md)), which run on DPAS, the card's matrix instruction, and read each weight once
for all rows:

| Kernel | Notes |
|--------|-------|
| `gemv` | the int4 mixers and MLPs, with per-shape split-K |
| `gemv_bf16` | the `a‖b` projections and the bf16 `lm_head` |
| `gdn_step` | the recurrent gated delta rule, fused with conv1d, q/k l2norm and the gated RMSNorm |
| `attn_decode_v2`, `attn_reduce_v2` | decode attention: partial results per block of the KV cache, then a merge |
| `attn_verify_dpas` | the verify step's attention on DPAS, for deep contexts |
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
| `pf_gemm_small`, `pf_gemm_q4` | the same on small tiles for short chunks; `pf_gemm_q4` converts the int4 words itself, no slab |
| `pf_attn_prep`, `pf_attn`, `pf_flash_attn` | q/k norm, RoPE, the KV write, then attention (the fused flash form is the default; the composed form remains) |
| `pf_gdn_conv`, `pf_gdn_wy`, `pf_gdn_scan` | the chunked delta net |

The decode kernels' split-K exists to buy hardware threads at `M = 1`. At
`M = 2048` a `[S][M][N]` partials rectangle at gate‖up's `N = 34816` would be
multi-terabyte, so prefill is a **second family** compiled without it rather
than a re-parameterisation of the first. The decode binaries are untouched by
every prefill change, byte for byte, which is why every prefill lever leaves the
decode row exactly where it was.

## Server

OpenAI-compatible. `src/server/` provides `/v1/models`, `/v1/chat/completions` and `/v1/completions`,
including OpenAI-shaped errors and SSE responses, against `TokIface`, `TemplateIface` and `EngineIface`.
`b70-serve` wires those interfaces to the real checkpoint tokenizer and template and to the engine its run mode
needs. The one intentional parsing deviation is that an absent `temperature` means greedy sampling. The parser
also accepts the `min_tokens`, `ignore_eos`, `return_token_ids`, `seed` and `stream_options.include_usage`
extension fields.

**One request at a time** (`--max-num-seqs 1`, the default; one card, `tp`, `pp`): requests share one engine
through a bounded FIFO; requests beyond the waiting depth (4) receive 503. `Engine::prefill` runs every prompt.
`golden_server_test` proves the server reproduces `b70-decode --ids --prefill`'s exact ids on the three golden
prompts, both sides on the same engine path.

**Several at once** (`--max-num-seqs` above 1, `replica`): the scheduler (`src/server/scheduler.{h,cc}`)
owns a batch engine and serves up to that many requests. A request waits in a queue (also 4 beyond the free
slots, then 503), takes a free slot, prefills, and then decodes one step shared with every other decoding slot.
At most one prefill chunk runs at a time per card and alternates with the decode steps, so a long prompt does not
stall the others for its whole length. On one card the batch engine is `cli::SlotsEngine` (the per-slot engine
adapters; a step over several slots is one batched replay of small-batch kernels); under `replica` it is
`cli::ReplicaEngine` over one such engine per card, each on its own thread: a prompt goes to the slot that
holds its prefix (else the card with the most free slots), and both cards can prefill at once. Speculative
decoding (MTP, prompt lookup) runs per slot inside the shared step, within a row budget of 8 rows per step.
A lone request is bitwise what the one-request server computes; several at once are judged against their lone
runs by the f32 judge ([14-golden-gate.md](14-golden-gate.md)), because the many-row kernels round differently.

**HTTP costs nothing measurable.** On the first server build, decode over HTTP measured 102.0% of the same
session's CLI control: the per-token host cost of detokenising and writing an SSE frame overlaps the next
replay. Prefill over HTTP measured 2915.83 ms for 4096 tokens against the CLI's 2919.6, indistinguishable from
zero against the run-to-run spread.

Host sampling (temperature, top-k, top-p) is implemented in the CLI's engine adapter: the replay's fp32 logits
row is read back once per sampled token through a reused immediate command list, masked to `vocab_used`,
sampled on the host, and written into `cur_token[0]` - the same protocol `Control` already supports for
ingest. It measured **0.537 ms/token** against a 0.62 ms bar registered before the measurement, so any request
with `temperature > 0` is sampled. Greedy is never slowed, because the readback only runs when a request asks
for sampling.

### Prefix caching

An agentic session sends the same long prefix again and again: the system prompt, the tools, the conversation so
far. The prefix cache lets a request skip the part of its prompt the engine has already read. Design:
[the prefix caching spec](superpowers/specs/2026-09-27-spec7-prefix-caching-design.md).

**Two levels.** One *resident session* stays on the card. Behind it is a store in pinned host RAM that every
finished chunk is also written to (`--prefix-cache-gb`, default `auto`: at most 32 GiB, sized from the machine's
RAM; `0` turns it off).

**What is stored.** The attention layers' KV cache can be cut at any position, but the delta-net layers keep a
running state that cannot be rewound. So reuse needs snapshots of that state. The store
(`src/server/prefix_cache.{h,cc}`) keeps KV blocks of 2048 positions and state snapshots (at the end of every
block, of every prompt and of every request) in a tree keyed by exact token ids, and evicts the least recently
used leaves.

**What a request gets.** If its token ids extend the resident session, it simply continues it. Otherwise it gets
a *prefix cache hit* from the deepest snapshot that is a prefix of its prompt: the state and the KV the card does
not already hold are copied back, and only the tail of the prompt is read.
`usage.prompt_tokens_details.cached_tokens` reports where it restarted. Side requests of an agent (titles,
subagents) replace the session on the card, not the store.

**Several requests.** With several slots, each slot is its own resident session over the one shared store. Under
`replica` a snapshot saved by one card gives a prefix cache hit on the other.

**Why the store is in 2 GiB pieces.** On two cards the second card maps the first card's host allocation. A
mapping larger than 4 GiB can fail in the kernel, which answers "busy", and the GPU driver retries a busy answer
forever: a hang, not an error. So any host allocation that both cards use stays at or below 4 GiB.

Results: [BENCHMARKS.md](BENCHMARKS.md), "Prefix caching".

The server owns the **tokenizer and chat template**. Both are host code on the
request path: encode once per request, detokenise once per token, off the card's
critical path and overlapping the next replay. Details in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).
