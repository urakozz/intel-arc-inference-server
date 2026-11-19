# Docs

Reference material for an LLM inference engine written from scratch for the Intel Arc Pro B70, against Level Zero
and OpenCL C. The project [README](../README.md) has the short version and the current speed numbers; this is where
the reasoning and the evidence live.

The numbered files are guides. They explain how the engine works today and why, and give the measurements that
decided each design.

## Where to start

- To run a model: [17-running-models.md](17-running-models.md).
- To understand the engine: [08-decode-vs-prefill.md](08-decode-vs-prefill.md), the central idea, then
  [04-architecture.md](04-architecture.md) for the engine as a whole.
- To know what the card can do: [01-hardware.md](01-hardware.md).

## The design

| | |
|---|---|
| [01-hardware.md](01-hardware.md) | what the card does, measured: memory bandwidth, the matrix instruction's rate by data type, and what is not available on it |
| [02-formats.md](02-formats.md) | why the weights are 4-bit integers with 16-bit activations, and what the alternatives cost |
| [03-models.md](03-models.md) | the focus model, Qwen3.8-27B: shapes, layer math, where the bytes are; the other models in brief |
| [04-architecture.md](04-architecture.md) | the engine in one document: the recorded and replayed decode step, the prompt pass, the two-card modes, speculative decoding, the server and the prefix cache |
| [05-perf-model.md](05-perf-model.md) | where the time goes in one-card decode and prefill, the ceilings, what was measured and rejected |
| [08-decode-vs-prefill.md](08-decode-vs-prefill.md) | why writing tokens (decode) and reading a prompt (prefill) want opposite kernels, and how the prefill design was chosen |
| [11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md) | text to token ids and back: the tokenizer, the chat template, tool-call parsing, streaming |
| [13-loader.md](13-loader.md) | from a checkpoint on disk to buffers on the card, every check on the way, and the checkpoint forms accepted |

## The kernels

| | |
|---|---|
| [12-kernels.md](12-kernels.md) | every GPU kernel of Qwen3.8 and Ornith's MoE prompt pass: what it computes, its layout decisions and its measured rate |
| [15-step-anatomy.md](15-step-anatomy.md) | one decode step and one prompt chunk, launch by launch, in each run mode |
| [16-int8-prefill.md](16-int8-prefill.md) | the opt-in int8 prompt pass: what runs in int8, what it gains, where it fails |

## Correctness

| | |
|---|---|
| [14-golden-gate.md](14-golden-gate.md) | how output is proven right: the engine against a CPU reference, and the float32 judge for paths that round differently |

## Running and measuring

| | |
|---|---|
| [17-running-models.md](17-running-models.md) | the commands to download, serve and benchmark each model, per number of cards, and what each model supports |
| [environment-variables.md](environment-variables.md) | what the programs, tests and scripts read from the environment (none of it needed to run the engine) |
| [10-the-box.md](10-the-box.md) | the build and test machine: remote builds, the GPU lock, validation runs, why an idle box matters for a benchmark |

## Background

| | |
|---|---|
| [06-prior-art.md](06-prior-art.md) | what to borrow from sycl-tla, oneDNN and vLLM's XPU kernels, and what not to |
| [07-open-questions.md](07-open-questions.md) | what is not yet known about the hardware and the numerics |
| [09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md) | the work that preceded this project: months of patching vLLM's kernels, and why it did not pay |
