# Docs

Reference material for an LLM inference engine written from scratch for the
Intel Arc Pro B70, against Level Zero and OpenCL C. The project README has the
short version; this is where the reasoning and the evidence live.

## Start here

- **[08-decode-vs-prefill.md](08-decode-vs-prefill.md)** - why decode and
  prefill want opposite kernels. If you read one file, read this one; everything
  else follows from it.
- **[01-hardware.md](01-hardware.md)** - what this card actually does, measured:
  bandwidth, the DPAS rate by data type, and what is simply not available on it.
- **[BENCHMARKS.md](BENCHMARKS.md)** - every measured row, how it was taken and
  what grade it earned.

## The design

| | |
|---|---|
| [02-formats.md](02-formats.md) | why int4 W4A16, and what the alternatives cost |
| [03-models.md](03-models.md) | the target model: shapes, layer math, byte accounting |
| [04-architecture.md](04-architecture.md) | the captured decode list, the prefill walk, the server |
| [05-perf-model.md](05-perf-model.md) | where the time goes, which levers are left, what was measured and rejected |
| [13-loader.md](13-loader.md) | checkpoint to device buffers, and every assert on the way |
| [11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md) | BPE, Jinja and UTF-8-safe streaming detokenisation |

## The kernels

| | |
|---|---|
| [12-kernels.md](12-kernels.md) | every kernel, its layout decisions and its measured rate |
| [15-step-anatomy.md](15-step-anatomy.md) | one decode step, per launch, in situ |

## Correctness

| | |
|---|---|
| [14-golden-gate.md](14-golden-gate.md) | the trust chain: engine against a CPU oracle against vLLM, and why tokens gate while tensors only diagnose |

## Context and what is unresolved

| | |
|---|---|
| [06-prior-art.md](06-prior-art.md) | what to borrow from sycl-tla, oneDNN, OpenVINO and vLLM, and what not to |
| [09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md) | months of patching the wrong layer, and the result |
| [07-open-questions.md](07-open-questions.md) | what is genuinely not known |
| [16-know-how.md](16-know-how.md) | the transferable findings, condensed |
| [10-the-box.md](10-the-box.md) | driving a remote build, and why an idle box matters for a benchmark |

## Measurement records

Dated, self-contained records of one experiment each. They are evidence, not
narrative, and the numbered guides cite them where they matter.

- [probe-dpas-rates-2026-09-22.md](probe-dpas-rates-2026-09-22.md) - DPAS rate
  by data type, with the assembly that proves the loops are DPAS bound
- [probe-fused-dequant-2026-09-22.md](probe-fused-dequant-2026-09-22.md) -
  fusing int4 dequant into the GEMM: bitwise correct, slower, **rejected**
- [probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md) - W4A8 on the mixed
  4-by-8 DPAS: 0.965x the control and 2.79% relative error, **rejected**
- [probe-prefill-vllm-parity-2026-09-14.md](probe-prefill-vllm-parity-2026-09-14.md)
  - oneDNN's fused int4 matmul measured directly, and where it lands
- [prefill-gdn-solve-register-results.md](prefill-gdn-solve-register-results.md)
  - removing barriers from the GDN triangular solve, **rejected**
- [prefill-gdn-scan-split-2026-09-20.md](prefill-gdn-scan-split-2026-09-20.md)
  and
  [prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md)
  - the split-BF16 delta net scan, the one approximation in the default path
- [prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md) - the matched
  protocol behind the prefill comparison

## Designs

[superpowers/specs/](superpowers/specs/) holds the design documents behind the
larger changes: the decode core, the prefill path, the tokenizer and HTTP layer,
the Level Zero prefill GEMM, and the prefill parity programme.
