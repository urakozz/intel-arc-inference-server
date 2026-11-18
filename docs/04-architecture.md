# Architecture

This document describes how the engine is put together: the toolchain, how the two cards are used, how
speculative decoding fits in, and the two execution paths, decode and prefill. It is for anyone who wants to read
or change the engine's code. In one sentence: every decode token replays a Level Zero command list that was
captured once, and the prompt (prefill) runs as a Level Zero walk over our own OpenCL C kernels.

## Language and toolchain

Everything is C++26, built with GCC 16 (`g++-16.2`) and CMake. C++26's static reflection (`-freflection`) does
three jobs: it maps Kolibri's and Flash-Next's `config.json` onto their descriptors (`src/common/json_reflect.h`),
maps the server's OpenAI requests, responses and stream chunks onto structs (`src/server/json_struct.h`), and turns
each binary's flags into one struct that generates its parser and `--help` (`src/cli/flags.h`). The engine build
uses no SYCL and no `icpx`.

Level Zero, Intel's low-level GPU runtime, is the only runtime. Device kernels are OpenCL C, compiled offline with
`ocloc` to device binaries and loaded with `zeModuleCreate`.

Python appears in one place, `tools/`, for offline weight conversion, the CPU oracle (a reference forward pass on
the CPU) and benchmark glue. It is never on the serving path: Python is part of the overhead this engine exists to
remove.

The engine runs on Linux only, with no Windows or macOS runtime and no portability shims.

The server never downloads models. `hf download <repo>` puts a model in the standard Hugging Face cache; the
loader resolves a repo id against that cache (`refs/main` to a snapshot) or takes an absolute snapshot path. A
missing snapshot is an error naming the path, not a fetch.

## Why the decode step is the whole design

At batch 1 this card is host-bound before it is kernel-bound: the GPU waits for the CPU to hand it work. The vLLM
work that preceded this project gave three independent signs of that
([09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md)):

- Two entirely different kernel paths, MXFP4 and GPTQ-int4, landed within 1% of each other, 72.65 against
  73.31 t/s. If the kernels were the limit they would not agree that closely.
- Turning XPU graphs off cost about 3x: the same 27B model measured 10.55 t/s without graphs and 31.5 with them.
- Measured MBU (memory bandwidth utilisation, the share of the card's bandwidth actually used) sat at 50-63% on
  those models. The card was idle, waiting for work.

So the first goal was not a faster GEMM. It was a decode step that does no host work per token. That is done:
99.7% of a decode step now runs inside the GPU fence, and the host spends 97 us per token. What remains is kernel
efficiency, which is a different problem ([05-perf-model.md](05-perf-model.md)).

## More than one model

The layout in this document is Qwen3.8's. Qwen3.8 is the focus model, used for agentic coding. Every other family
has its own engine and descriptor built on the same foundations: Level Zero command lists, our own OpenCL C
kernels, the loader and the server. The engine picks the model from the checkpoint's `config.json`.

| model | what is different about it |
|---|---|
| Ornith 1.5 | a mixture of experts (MoE): a router picks a few expert networks for each token |
| Qwen3.8-Flash-Next | a larger MoE, served whole on two cards, with part of its routed experts kept in host memory (the expert tier) |
| Kolibri-1 | MoE with sliding-window attention layers; supported by the code but parked |

Per-model status and commands are in [17-running-models.md](17-running-models.md).

### Run modes

`b70-serve --parallel` picks how the cards are used. The default, `auto`, chooses by model and load:

- Qwen3.8 on two cards runs `tp` for one request at a time, as long as the context is left at `auto` and nothing
  asked for is something `tp` does not run (DFlash, a draft vocabulary, the `l0-int8` prefill). It runs `replica`
  when `--max-num-seqs` is above 1.
- Any other model, or Qwen3.8 outside those conditions, runs on one card when the model and the asked context fit
  it, and otherwise as `pp`.

The four modes:

- One card (`--parallel 1`): the whole engine on one card.
- `tp`, tensor parallel (Qwen3.8 only): every layer is split over both cards, with half the attention heads and
  half the MLP on each. After each split layer the cards add their halves by writing straight into each other's
  memory over PCIe (peer-to-peer, P2P). This is the fastest way to serve one request, at the full context, but it
  serves one request at a time.
- `replica` (Qwen3.8 and Ornith): a complete copy of the model on each card, one engine per card, with requests
  spread over the cards. This is the choice for 2 to 8 requests at once.
- `pp`, pipeline parallel: the first layers on one card and the rest on the other. It runs at about one card's
  speed, but each card holds only half the weights, so it serves models that are too big for one card or have no
  tensor-parallel path. The prefix cache and the MTP head work across the split.

Speed numbers per mode are in the benchmark record.

### Several requests at once: `--max-num-seqs`

`--max-num-seqs N` (vLLM's name for it) sets how many requests run at once, and the context length follows from
it. The default, 1, is one request at the largest context that fits. With N above 1 (Qwen3.8), a card holds
sequence slots: slot 0 gets the largest context that fits beside N - 1 further slots of 32768 positions each. One
card takes up to 4 slots; two cards run `replica` and split the slots between them, up to 8 (4 a card). `tp` and
`pp` refuse N above 1. An explicit `--max-len` that does not fit beside the other slots fails at startup and names
the memory plan. `--parallel replica` on its own means N = 2, one request a card. A request that runs into the end
of its slot finishes with `length` and is logged. Tensor parallel with slots was built and ruled out, because
`replica` is faster for several requests at once ([low-priority-work.md](low-priority-work.md)).

Why fixed slots and not a shared KV pool? The KV cache holds the attention keys and values of every earlier
position, and it is most of a request's memory. The engine is tuned for one or two people running local agentic
sessions: a main session that needs its full context plus a few short helper requests. Each slot's KV memory is
reserved at startup, so the main session always has its context, nothing is ever preempted (stopped and recomputed
later to free memory), and no request races another for memory. vLLM's paged pool, one pool of small KV blocks
shared by all requests with `max_model_len` the most any one request may reach, uses memory better when request
sizes vary. The price is block tables in every attention kernel, a preemption policy and the latency spikes it
causes. That option was considered and postponed ([low-priority-work.md](low-priority-work.md)).

### How the two cards stay in step under tensor parallel

Tensor parallel has no host in the loop. Each card runs its half of a layer and needs the other card's half before
it can go on. The cards hand these partial results to each other through a landing buffer that lives in the
reader's memory and that the other card writes into over P2P. The protocol (`src/runtime/tp_protocol.h`, device
side `src/kernels/tp.cl`) has three parts for each piece of data:

1. The data. The sender writes its partial result into the landing buffer.
2. The stamp. The sender then writes the number of this call into a trailer. The stamp fills a whole 64-byte line
   (the unit in which the card's memory is written) with 16 identical words.
3. The flag. Last, after a release fence, the sender writes the same call number into a flag on a separate page.

The reader spins on the flag until it shows the call number it expects, then reads the stamp. If the stamp does not
match, the flag arrived before its data, and the reader reports a stale read instead of using old data. If the flag
never arrives within a bound, it reports a timeout. The first failure is recorded once and later waits skip their
spin, so a lost peer costs one bound and not one per layer.

The stamp fills a whole line because every 64-byte line of the landing buffer must have exactly one writer, and the
data is written as whole lines too. When sixteen 4-byte stamps shared one line, each written by a separate store, a
stamp could reach the reader after its flag: the reader saw the flag of call n with the stamp of call n - 2, and one
to three requests in a few dozen failed. With one stamp per line the failures went to zero.

## Speculative decoding

A plain decode step produces one token. Speculative decoding guesses several next tokens cheaply and checks all of
them in one batched step, the verify. Guesses that the model itself would have produced are kept, so greedy output
is unchanged; the gain is that a verify of several rows costs much less than that many plain steps. The guesses come
from a proposer, chosen with `--spec`:

| `--spec` | proposer | notes |
|---|---|---|
| `auto` (default) | the MTP head on Qwen3.8, Ornith and Qwen3.8-Flash-Next; none on Kolibri-1 | no extra VRAM |
| `off` | none | |
| `mtp` | the checkpoint's MTP head (multi-token prediction: a small extra head that predicts up to 3 tokens ahead) | how many guesses to make is decided per step from the head's own acceptance; `--spec-max` caps it (1 to 3) |
| `dflash` | DFlash, a separate small draft model that proposes a block of up to 7 tokens (Qwen3.8) | needs `--draft-model`; the drafter's weights take VRAM from the context, so it is never chosen by default |

Prompt lookup is a third source of guesses, not a mode of its own. When the answer repeats text that is already in
the context (an edit that copies a file, a tool call that quotes its input), the text that followed last time is a
very good guess. Lookup runs inside both proposers on one card with one request at a time, inside `mtp` under
`replica` with one request a card, and inside `dflash` under `tp`. When it finds a long enough match, its copied
tokens are verified for that step and the proposer's own draft is skipped; otherwise the step is an ordinary MTP or
DFlash one. It is on by default; `B70_SPEC_LOOKUP=0` turns it off for comparisons.

Inside MTP on one card with one request (bf16 KV, the int8 output head), a long copied run is checked in a wide
verify: one batched step of 16, 32 or 64 rows, that is up to 15, 31 or 63 copied tokens at once instead of the
head's 3. A run earns the wider lists:

- its first wide check is 16 rows;
- a check that keeps every token lets the next one take up to 63, in the narrowest list that holds what the copied
  text still has;
- a check that keeps at least half drops one width, and one that keeps less drops back to 16;
- a request whose last two wide checks both missed (a rename, whose copied runs break every few dozen tokens) stays
  at 16 rows.

The 64-row list is used up to about 90,000 tokens of context; past that it no longer pays against 32. Every row of
the 32- and 64-row lists is bitwise the 16-row list's row at the same position, because the GEMM, the attention and
the delta-net step are built so that a row's arithmetic does not depend on how many rows run beside it. The wider
lists change speed, never output. Sampled requests and `replica` keep the 16-row list.

Where each proposer runs:

| | one card | `replica` | `tp` | `pp` | several requests on a card |
|---|---|---|---|---|---|
| `mtp` | yes | yes | yes | yes | yes |
| `dflash` | yes, with prompt lookup | yes, a drafter on each card | yes, see below | yes | refused |

DFlash under `tp` splits the drafter over both cards. It is kept as an explicit choice, but it is not faster than
DFlash on one card for one session, and the server says so at startup, recommending one card
(`--parallel 1 --spec dflash`), `replica` (a drafter per card) or `--spec mtp` on two cards.

Where speculative output differs from the plain run, it is checked against a float32 CPU reference, the f32 judge
([14-golden-gate.md](14-golden-gate.md)). Commands and measured speeds are in
[17-running-models.md](17-running-models.md) and the benchmark record.

## The prompt pass backends

The prompt pass (prefill) runs in bf16 by default. `--prefill-backend l0-int8` is an opt-in alternative that runs
the MLP's linears with int8 activations ([16-int8-prefill.md](16-int8-prefill.md)).

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

Each directory should be understandable without reading the others. The loader knows about checkpoint formats and
nothing about kernels; the kernels know about memory layouts and nothing about files.

## Execution model

Decode and prefill are two separate paths, deliberately not unified.

### Decode: one captured list, replayed

A decode step processes one token with fixed shapes. It is built once into a Level Zero command list (a recorded
sequence of kernel launches that the GPU can run again without the host re-issuing them) and replayed. Between
replays the host changes nothing in the list.

The three values that change per token, the KV write offset and position, the sequence length and the current
token id, live in a small device-resident control block that every kernel reads and the sampler writes. The sampler
stores its argmax into `control.cur_token`, the embedding gather at the top of the next replay reads it, and the
last kernel in the list increments `control.pos`. The host's per-token work is one
`zeCommandQueueExecuteCommandLists`, a fence wait and one read of `cur_token` for the stream.

Every buffer address, kernel argument and dispatch dimension is fixed at capture time, and the list is
byte-identical on every replay. That is what makes the capture-safety rule below testable: replay the same list
twice from the same state and diff the outputs.

The captured step is 774 kernels across 19 modules: 48 GDN layers (Gated DeltaNet, a linear-attention layer with a
fixed-size recurrent state) of 12 kernels each, 16 full-attention layers of 12 each, and 6 at the token boundary
(`embed_gather`, the final norm's two launches, `lm_head`, two argmax stages), so 576 + 192 + 6. `runtime::build`
(`src/runtime/capture.cc`) is the one walk that binds them. `tests/runtime/replay_determinism_test` runs the diff
described above: the same list from the same state twice, bitwise on the token ids, on a per-layer residual tap and
on every persistent buffer, then once more from a re-zeroed state.

Level Zero's mutable-command-list extension exists, and it is the fallback for a kernel that genuinely cannot read
its state from memory, not the design. Changing arguments per token is host work, and host work is what this loop
exists to remove. A test proves that arguments are resolved at append time, which is what makes the control block
sufficient.

One side effect is worth keeping in view: once the host touches nothing between tokens, submitting `N` tokens per
`zeCommandQueueExecuteCommandLists` is a loop unroll rather than a redesign. Streaming wants per-token granularity,
so nothing does it today, but nothing rules it out either.

### Prefill: one in-order Level Zero list, no host waits

Prefill has dynamic shapes, so nothing is captured and every argument is set per launch. A prefill chunk runs on
one in-order asynchronous Level Zero immediate command list (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`,
`MODE_ASYNCHRONOUS`) and makes no SYCL call and no host wait at all. The int4 weights are dequantised into
1024-column bf16 slabs and multiplied by our own DPAS GEMM, interleaved on that one list. (DPAS is the card's matrix
multiply-accumulate instruction, the one that runs on its XMX systolic arrays.) A short chunk, up to 384 rows, uses
small tiles that read the int4 words directly (`pf_gemm_small` / `pf_gemm_q4`). Attention and the delta-net scan
are ours too.

This is the second design of the path, and the reason the first was replaced is not the obvious one. The original
prefill called sycl-tla's GEMM from a SYCL queue built by interop from the same `ze_context` and `ze_device`. The two
queues cannot be ordered against each other on this device, and that is a measurement, not a preference:
`zeDeviceGetCommandQueueGroupProperties` on the B70 reports exactly two groups, ordinal 0 COMPUTE+COPY+COOPERATIVE
with `numQueues = 1` and ordinal 1 COPY-only with `numQueues = 1`, and `(0,1)` and `(1,1)` are both refused with
`ZE_RESULT_ERROR_INVALID_ARGUMENT`. There is one compute queue, the two software queues share it, and no device-side
primitive exists for a dependency between the two runtimes. So every hand-over between Level Zero and SYCL was a
host wait: 656 per chunk, at a measured 22.35 us each.

Moving the GEMM onto the Level Zero list removed all of them. The Level Zero walk, with seven times as many launches
(8,689 against 1,201 a chunk), was faster on an idle card (1502.83 against 1407.63 t/s) and used a tenth of the
scratch memory. The worry that thousands of launches would cost host time did not materialise at all. The sycl-tla
path was removed after `prefill_backend_equivalence_test` had shown the two bitwise equal.

Two host-side rules follow from the list being asynchronous; they are stated here once rather than at each site:

- A host write to `Control` is always preceded by a drain. A write racing a launch that reads `pos` would be a bug
  no test could reproduce. `Engine::prefill` drains before every chunk's `memcpy` of the ids and before both
  `Control` writes.
- `Engine::prefill` lives in its own translation unit, not in the runtime library that every decode binary and
  test links, so a binary that never calls `prefill()` contains no reference to any prefill symbol.

### Attention under replay

A captured kernel cannot change its grid as the context grows, yet the 16 full-attention layers read `seq_len` KV
entries. There are two honest options:

1. A fixed grid over `max_model_len` with a device-side early-out: every work-group reads `control.seq_len` and
   returns if its KV block lies beyond it. One list, simple.
2. Context-bucketed lists: capture one list per bucket and choose by `seq_len`. Fewer idle work-groups, but several
   lists to keep correct.

The engine uses option 1, and the choice was measured rather than assumed. A grid sized for `max_len` 16384
against one sized for 4096, at the same depth and the same live work, costs 0.046 ms per token, 0.11% of a step,
for 3072 extra work-groups: about 15 ns per work-group that exits early. When the attention block size was retiled,
quadrupling the grid again made the launch 39% faster, so the fixed grid is cheaper than that bound, not more
expensive. Bucketed lists would buy 0.11% and cost a captured list per bucket, its memory, and a host-side branch
on context length in the one loop that currently has no branches at all. GDN layers have no such problem: their
state is fixed-size.

### Device selection

There is no device flag: the CLIs run on the cards the driver shows. One card is device 0; two are devices 0 and 1
(`tp`, `replica`, `pp`). `ZE_AFFINITY_MASK`, a filter at the Level Zero driver level, decides which cards are
enumerated at all. The engine respects it and never sets it, and under a mask every index refers to the masked view
(`ZE_AFFINITY_MASK=1` runs on the second card as device 0).

Before any weight is read, the hardware gate (`cli/hardware_check.h`) checks what the driver shows: one or two Intel
Arc Pro B70s (vendor 0x8086, device id 0xE223; the B65 and B60 are refused as not tested), the Xe-core count the
kernels are built for (32), and peer access both ways when there are two cards. The fit checks that run before a
device opens use the memory sizes the gate read from the cards.

### What makes this hard

Kernels must be capture-safe: no allocation, no host synchronisation, and no state that persists across replays.
This is not hypothetical. The `vllm-xpu-kernels` grouped GEMM had a global atomic tile counter that was reset inside
the kernel with no device-wide barrier. Under eager launch the timing happened to work; under graph capture the
counter carried the previous replay's value and raced into out-of-bounds tile coordinates, producing
`UR_RESULT_ERROR_DEVICE_LOST` at batch > 1.

The rule: any kernel that cannot be replayed byte-identically from the same command list is a bug, not a
limitation. Design for replay from the first kernel.

## The kernel families

There are two kernel families, chosen by `M` (the number of rows, that is tokens, processed at once), and
deliberately not one kernel pretending to serve both ([08-decode-vs-prefill.md](08-decode-vs-prefill.md) explains
why).

The decode family (`src/kernels/`) is bound by `capture.cc` at a fixed `M`: 1 for a plain step, 2 to 8 for a
speculative verify list (`M` is the number of rows checked at once), 16 for the drafter's blocks. The table below
is the `M = 1` list. Every list at `M >= 2` replaces `gemv` / `gemv_i8w` with the small-batch GEMMs `gemm_sm` /
`gemm_sm_i8w` ([12-kernels.md](12-kernels.md)), which run on DPAS and read each weight once for all rows.

| Kernel | Notes |
|--------|-------|
| `gemv` | the int4 mixers and MLPs, with per-shape split-K |
| `gemv_bf16` | the `a‖b` projections and the bf16 `lm_head` |
| `gdn_step` | the recurrent gated delta rule, fused with conv1d, q/k l2norm and the gated RMSNorm |
| `attn_decode_v2`, `attn_reduce_v2` | decode attention: partial results per block of the KV cache, then a merge |
| `attn_verify_dpas` | the verify step's attention on DPAS, for deep contexts |
| `prep` | residual fold, RMSNorm finish, SiLU-mul, the gated head |
| `embed_gather`, `argmax` | the token boundary |

The prefill family (`src/kernels/prefill/`) is a separate `ocloc` build per kernel. None is bound by `capture.cc`,
and `M` is a runtime argument in all of them:

| Kernel | What it is |
|---|---|
| `pf_embed`, `pf_prep`, `pf_gated_head` | the decode `prep`/`embed` family at runtime `M` and without split-K |
| `pf_gemv_bf16` | the `a‖b` bf16 projection |
| `pf_dequant_slab` | one int4 linear into a bf16 column slab |
| `pf_gemm` | our own DPAS GEMM |
| `pf_gemm_small`, `pf_gemm_q4` | the same on small tiles for short chunks; `pf_gemm_q4` converts the int4 words itself, with no slab |
| `pf_attn_prep`, `pf_attn`, `pf_flash_attn` | q/k norm, RoPE, the KV write, then attention (the fused flash form is the default; the composed form remains) |
| `pf_gdn_conv`, `pf_gdn_wy`, `pf_gdn_scan` | the chunked delta net |

The decode kernels' split-K (splitting the reduction dimension K over several work-groups and summing their partial
results) exists to buy hardware threads at `M = 1`. At `M = 2048` a `[S][M][N]` rectangle of partials at gate‖up's
`N = 34816` would be multi-terabyte, so prefill is a second family compiled without split-K rather than a
re-parameterisation of the first. The decode binaries are unchanged, byte for byte, by every prefill change, which is
why no prefill improvement moves the decode speed.

## Server

The server is OpenAI-compatible. `src/server/` provides `/v1/models`, `/v1/chat/completions` and
`/v1/completions`, including OpenAI-shaped errors and SSE streaming, against the interfaces `TokIface`,
`TemplateIface` and `EngineIface`. `b70-serve` wires those interfaces to the checkpoint's real tokenizer and template
and to the engine its run mode needs. The one intentional parsing deviation is that an absent `temperature` means
greedy sampling. The parser also accepts the extension fields `min_tokens`, `ignore_eos`, `return_token_ids`, `seed`
and `stream_options.include_usage`.

With one request at a time (`--max-num-seqs 1`, the default; one card, `tp` or `pp`), requests share one engine
through a bounded FIFO queue, and requests beyond its waiting depth of 4 receive 503. `Engine::prefill` runs every
prompt. `golden_server_test` proves the server reproduces the exact ids of `b70-decode --ids --prefill` on the three
golden prompts, both sides on the same engine path.

With several at once (`--max-num-seqs` above 1, or `replica`), the scheduler (`src/server/scheduler.{h,cc}`) owns a
batch engine and serves up to that many requests. A request waits in a queue (again 4 deep beyond the free slots,
then 503), takes a free slot, prefills, and then decodes in one step shared with every other decoding slot. At most
one prefill chunk runs at a time per card, alternating with the decode steps, so a long prompt does not stall the
others for its whole length. On one card the batch engine is `cli::SlotsEngine`, the per-slot engine adapter; a step
over several slots is one batched replay of small-batch kernels. Under `replica` it is `cli::ReplicaEngine` over one
such engine per card, each on its own thread: a prompt goes to the slot that holds its prefix (else to the card with
the most free slots), and both cards can prefill at once. Speculative decoding (MTP, prompt lookup) runs per slot
inside the shared step, within a budget of 8 rows per step. A lone request is bitwise what the one-request server
computes. Several at once are judged against their lone runs by the f32 judge
([14-golden-gate.md](14-golden-gate.md)), because the many-row kernels round differently.

HTTP costs nothing measurable. On the first server build, decode over HTTP measured 102.0% of the same session's CLI
control: the per-token host cost of detokenising and writing an SSE frame overlaps the next replay. Prefill over
HTTP measured 2915.83 ms for 4096 tokens against the CLI's 2919.6, indistinguishable from zero against the
run-to-run spread.

Host sampling (temperature, top-k, top-p) lives in the CLI's engine adapter. The replay's fp32 logits row is read
back once per sampled token through a reused immediate command list, masked to `vocab_used`, sampled on the host,
and written into `cur_token[0]`, the same path `Control` already supports for prompt ingest. It measured
0.537 ms per token against a 0.62 ms bar set before the measurement, so any request with `temperature > 0` is
sampled this way. Greedy decoding is never slowed, because the readback runs only when a request asks for sampling.

### Prefix caching

An agentic session sends the same long prefix again and again: the system prompt, the tools, the conversation so
far. The prefix cache lets a request skip the part of its prompt the engine has already read.

There are two levels. One resident session stays on the card. Behind it is a store in pinned host RAM to which every
finished chunk is also written (`--prefix-cache-gb`, default `auto`: at most 32 GiB, sized from the machine's RAM;
`0` turns it off).

The attention layers' KV cache can be cut at any position, but the delta-net layers keep a running state that cannot
be rewound, so reuse needs snapshots of that state. The store (`src/server/prefix_cache.{h,cc}`) keeps KV blocks of
2048 positions and state snapshots (at the end of every block, of every prompt and of every request) in a tree keyed
by exact token ids, and evicts the least recently used leaves.

If a request's token ids extend the resident session, it simply continues it. Otherwise it gets a prefix cache hit
from the deepest snapshot that is a prefix of its prompt: the state and the KV the card does not already hold are
copied back, and only the tail of the prompt is read. `usage.prompt_tokens_details.cached_tokens` reports where it
restarted. An agent's side requests (titles, subagents) replace the session on the card, not the store.

With several slots, each slot is its own resident session over the one shared store. Under `replica` a snapshot
saved by one card gives a prefix cache hit on the other.

The store is made of 2 GiB pieces for a two-card reason. On two cards the second card maps the first card's host
allocation. A mapping larger than 4 GiB can fail in the kernel, which answers "busy", and the GPU driver retries a
busy answer forever: a hang, not an error. So any host allocation that both cards use stays at or below 4 GiB.

Results are in the benchmark record, "Prefix caching".

The server also owns the tokenizer and chat template. Both are host code on the request path: encode once per
request, detokenise once per token, off the card's critical path and overlapping the next replay. Details are in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md).
