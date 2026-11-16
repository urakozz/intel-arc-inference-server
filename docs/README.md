# Docs

Reference material for an LLM inference engine written from scratch for the Intel Arc Pro B70, against Level
Zero and OpenCL C. The project [README](../README.md) has the short version; this is where the reasoning and the
evidence live.

The numbered files are guides: they explain how the engine works today and why. [BENCHMARKS.md](BENCHMARKS.md)
is the record of measured rows, each dated with how it was taken. The work lists say what is not finished.

If you read one file, read [08-decode-vs-prefill.md](08-decode-vs-prefill.md). Then
[04-architecture.md](04-architecture.md) for the engine as a whole, [01-hardware.md](01-hardware.md) for what the
card can do, and [17-running-models.md](17-running-models.md) to run a model.

## The design

| | |
|---|---|
| [01-hardware.md](01-hardware.md) | what the card does, measured: memory bandwidth, the matrix instruction's rate by data type, and what is not available on it |
| [02-formats.md](02-formats.md) | why the weights are 4-bit integers with 16-bit activations, and what the alternatives cost |
| [03-models.md](03-models.md) | the focus model, Qwen3.8-27B: shapes, layer math, where the bytes are |
| [04-architecture.md](04-architecture.md) | the engine in one document: the recorded and replayed decode step, the prompt pass, the two-card modes, speculative decoding, the server and the prefix cache |
| [05-perf-model.md](05-perf-model.md) | where the time goes, which levers are left, what was measured and rejected |
| [08-decode-vs-prefill.md](08-decode-vs-prefill.md) | why writing tokens (decode) and reading a prompt (prefill) want opposite kernels |
| [11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md) | text to token ids and back: the tokenizer, the chat template, streaming |
| [13-loader.md](13-loader.md) | from a checkpoint on disk to buffers on the card, and every check on the way |

## The kernels

| | |
|---|---|
| [12-kernels.md](12-kernels.md) | every GPU kernel, its layout decisions and its measured rate |
| [15-step-anatomy.md](15-step-anatomy.md) | one decode step and one prompt chunk, launch by launch |
| [16-int8-prefill.md](16-int8-prefill.md) | the optional int8 prompt pass: what runs in int8, what it gains, where it fails |

## Correctness

| | |
|---|---|
| [14-golden-gate.md](14-golden-gate.md) | how output is proven right: the engine against a CPU reference, and the float32 judge for paths that round differently |

## Running and measuring

| | |
|---|---|
| [17-running-models.md](17-running-models.md) | the commands to download, serve and benchmark each model, per number of cards, and what each model supports |
| [environment-variables.md](environment-variables.md) | what the programs and scripts read from the environment (none of it needed to run the engine) |
| [10-the-box.md](10-the-box.md) | the build and test machine: how to build and test on it, the GPU lock, validation runs, why an idle box matters for a benchmark |
| [BENCHMARKS.md](BENCHMARKS.md) | every measured row, how it was taken and what grade it earned |

## Background

| | |
|---|---|
| [06-prior-art.md](06-prior-art.md) | what to borrow from sycl-tla, oneDNN, another stack and vLLM, and what not to |
| [07-open-questions.md](07-open-questions.md) | what is not known about the hardware |
| [09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md) | the work that preceded this project: months of patching vLLM's kernels, and the result |

## Work lists

| | |
|---|---|
| [open-work.md](open-work.md) | the one list of what is not finished |
| [low-priority-work.md](low-priority-work.md) | work that is ruled out for now or not scheduled |
