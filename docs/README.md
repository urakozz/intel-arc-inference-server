# Docs

Reference material for an LLM inference engine written from scratch for the Intel Arc Pro B70, against Level
Zero and OpenCL C. The project [README](../README.md) has the short version; this is where the reasoning and the
evidence live.

There are three kinds of document here:

- **Guides** (the numbered files) explain how the engine works today and why.
- **Records** (`probes/`, `BENCHMARKS.md`) are dated evidence: one experiment or one measured row each. They are
  not rewritten when things change.
- **Work lists** say what is not finished.

## Start here

1. [08-decode-vs-prefill.md](08-decode-vs-prefill.md) - why writing tokens (decode) and reading a prompt
   (prefill) want opposite kernels. If you read one file, read this one.
2. [04-architecture.md](04-architecture.md) - the engine in one document: the decode step that is captured once
   and replayed, the prompt pass, the two-card modes, speculative decoding, the server and the prefix cache.
3. [01-hardware.md](01-hardware.md) - what the card actually does, measured: memory bandwidth, the rate of its
   matrix instruction by data type, and what is not available on it.
4. [18-running-models.md](18-running-models.md) - the commands to download, serve and benchmark each model, and
   what each model supports.

## The guides

**The design**

| | |
|---|---|
| [02-formats.md](02-formats.md) | why the weights are 4-bit integers with 16-bit activations, and what the alternatives cost |
| [03-models.md](03-models.md) | the focus model, Qwen3.8-27B: shapes, layer math, where the bytes are |
| [05-perf-model.md](05-perf-model.md) | where the time goes, which levers are left, what was measured and rejected |
| [13-loader.md](13-loader.md) | from a checkpoint on disk to buffers on the card, and every check on the way |
| [11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md) | text to token ids and back: the tokenizer, the chat template, streaming |

**The kernels**

| | |
|---|---|
| [12-kernels.md](12-kernels.md) | every GPU kernel, its layout decisions and its measured rate |
| [15-step-anatomy.md](15-step-anatomy.md) | one decode step and one prompt chunk, launch by launch |
| [17-int8-prefill.md](17-int8-prefill.md) | the optional int8 prompt pass: what runs in int8, what it gains, where it fails |

**Correctness**

| | |
|---|---|
| [14-golden-gate.md](14-golden-gate.md) | how output is proven right: the engine against a CPU reference, and the float32 judge for paths that round differently |

**Running and measuring**

| | |
|---|---|
| [18-running-models.md](18-running-models.md) | recipes per model and per number of cards |
| [environment-variables.md](environment-variables.md) | what the programs and scripts read from the environment (none of it needed to run the engine) |
| [10-the-box.md](10-the-box.md) | the build and test machine: how to build and test on it, the GPU lock, validation runs, why an idle box matters for a benchmark |
| [BENCHMARKS.md](BENCHMARKS.md) | every measured row, how it was taken and what grade it earned |
| [coding-benchmarks.md](coding-benchmarks.md) | published coding-agent scores of the models the engine supports or considered |

**Background**

| | |
|---|---|
| [16-know-how.md](16-know-how.md) | the transferable lessons, condensed |
| [06-prior-art.md](06-prior-art.md) | what to borrow from sycl-tla, oneDNN, another stack and vLLM, and what not to |
| [09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md) | the work that preceded this project: months of patching vLLM's kernels, and the result |
| [07-open-questions.md](07-open-questions.md) | what is not known about the hardware |

## Work lists

- [open-work.md](open-work.md) - the one list of what is not finished.
- [low-priority-work.md](low-priority-work.md) - work that is ruled out for now or not scheduled.

## Records: probes

[probes/](probes/) holds one dated record per experiment. They are the evidence behind decisions; the guides
cite them where they matter. By topic:

**The hardware and the formats**

- [probe-dpas-rates-2026-09-22](probes/probe-dpas-rates-2026-09-22.md) - the matrix instruction's rate by data type
- [probe-fused-dequant-2026-09-22](probes/probe-fused-dequant-2026-09-22.md) - fusing the int4 conversion into the GEMM: rejected
- [probe-w4a8-2026-09-23](probes/probe-w4a8-2026-09-23.md) - int8 activations on int4 weights: rejected for the dense layers
- [probe-w4a4-2026-10-28](probes/probe-w4a4-2026-10-28.md) - int4 activations: doubles the output error

**Prefill (reading the prompt)**

- [probe-flash-attn-2026-09-25](probes/probe-flash-attn-2026-09-25.md) - flash attention in prefill
- [probe-small-chunk-2026-11-08](probes/probe-small-chunk-2026-11-08.md),
  [probe-fused-small-chunk-2026-11-08](probes/probe-fused-small-chunk-2026-11-08.md) - short prompt chunks
- [probe-kv8-prefill-2026-11-08](probes/probe-kv8-prefill-2026-11-08.md) - prefill onto the int8 KV cache
- [probe-prefill-chunk-split-2026-11-10](probes/probe-prefill-chunk-split-2026-11-10.md),
  [probe-gdn-prefill-2026-11-10](probes/probe-gdn-prefill-2026-11-10.md) - where a full chunk's time goes
- [probe-qwen-pf-gemm-2026-11-13](probes/probe-qwen-pf-gemm-2026-11-13.md) - the bf16 prefill GEMM and 4096-row chunks
- [a4-int8-prefill-int8-kv-2026-11-10](probes/a4-int8-prefill-int8-kv-2026-11-10.md) - the int8 prompt pass with the int8 cache: a known fail

**Decode (writing tokens)**

- [probe-decode-attn-2026-09-28](probes/probe-decode-attn-2026-09-28.md) - decode attention deep in a context
- [probe-longctx-decode-2026-11-09](probes/probe-longctx-decode-2026-11-09.md) - decode at long context, part by part
- [probe-lm-head-2026-09-28](probes/probe-lm-head-2026-09-28.md) - an int8 output head
- [probe-int8-kv-2026-09-28](probes/probe-int8-kv-2026-09-28.md),
  [probe-int8-kv-qwen38-2026-10-06](probes/probe-int8-kv-qwen38-2026-10-06.md) - the int8 KV cache
- [probe-batching-2026-10-23](probes/probe-batching-2026-10-23.md) - several requests in one decode step

**Speculative decoding**

- [probe-mtp-2026-09-27](probes/probe-mtp-2026-09-27.md) - the MTP head
- [probe-prompt-lookup-2026-10-05](probes/probe-prompt-lookup-2026-10-05.md) - prompt lookup
- [probe-dflash-2026-10-08](probes/probe-dflash-2026-10-08.md) - the DFlash draft model's acceptance
- [dflash-vs-sly-2026-11-11](probes/dflash-vs-sly-2026-11-11.md) - our DFlash against two other engines
- [probe-small-m-gemm-2026-11-01](probes/probe-small-m-gemm-2026-11-01.md),
  [probe-verify-attn-2026-11-03](probes/probe-verify-attn-2026-11-03.md) - the verify step's GEMM and attention
- [probe-dflash-kv8-verify-m8-2026-11-13](probes/probe-dflash-kv8-verify-m8-2026-11-13.md) - the 8-row verify on the int8 cache: no defect
- [probe-eagle3-k2-2026-10-08](probes/probe-eagle3-k2-2026-10-08.md) - EAGLE-3 on K2-Horizon: no-go

**Caching and two cards**

- [probe-prefix-cache-2026-09-27](probes/probe-prefix-cache-2026-09-27.md) - the prefix cache
- [probe-tp-2026-11-01](probes/probe-tp-2026-11-01.md),
  [probe-tp-prefill-2026-11-10](probes/probe-tp-prefill-2026-11-10.md),
  [tp-twin-count-2026-11-10](probes/tp-twin-count-2026-11-10.md),
  [tp-int8-prefill-2026-11-11](probes/tp-int8-prefill-2026-11-11.md),
  [probe-tp-verify-m24-2026-11-14](probes/probe-tp-verify-m24-2026-11-14.md) - tensor parallel

**Mixture-of-experts models (Ornith, Flash-Next)**

- [ornith-prefill-profile-2026-11-12](probes/ornith-prefill-profile-2026-11-12.md),
  [probe-moe-prefill-2026-11-12](probes/probe-moe-prefill-2026-11-12.md),
  [probe-prefill-phase2-2026-11-12](probes/probe-prefill-phase2-2026-11-12.md),
  [probe-prefill-phase3-2026-11-13](probes/probe-prefill-phase3-2026-11-13.md),
  [probe-moe-int8-2026-11-13](probes/probe-moe-int8-2026-11-13.md) - Ornith's prompt pass
- [probe-qwen4exp-2026-10-09](probes/probe-qwen4exp-2026-10-09.md),
  [probe-flash-next-tier-2026-11-12](probes/probe-flash-next-tier-2026-11-12.md),
  [q4exp-f32-aligned-routes-2026-11-10](probes/q4exp-f32-aligned-routes-2026-11-10.md),
  [q4exp-f32-decode-diagnostic-2026-11-10](probes/q4exp-f32-decode-diagnostic-2026-11-10.md) - Flash-Next

**Parked and dropped models**

- [probe-kolibri-2026-10-05](probes/probe-kolibri-2026-10-05.md),
  [probe-bolt-2026-10-21](probes/probe-bolt-2026-10-21.md), [probe-lilicorr-2026-10-23](probes/probe-lilicorr-2026-10-23.md),
  [lilicorr-exact-divergence-2026-11-07](probes/lilicorr-exact-divergence-2026-11-07.md) - parked
- [probe-agnes-2026-10-03](probes/probe-agnes-2026-10-03.md),
  [probe-k2-2026-10-05](probes/probe-k2-2026-10-05.md) - dropped (K2-Horizon 2026-11-15, tag `archive/k2-horizon`);
  kept as the record

The prefill investigations of September 2026 are folded into
[08-decode-vs-prefill.md](08-decode-vs-prefill.md#how-prefill-got-here) and
[12-kernels.md](12-kernels.md#how-the-prefill-kernels-got-here).

## Designs and plans

[superpowers/specs/](superpowers/specs/) holds the design document behind each larger change, one per feature or
model. [superpowers/plans/](superpowers/plans/) holds the step-by-step plans they were built from. Both are dated
and describe intent at the time; where a guide and a spec disagree, the guide and the code are current.
