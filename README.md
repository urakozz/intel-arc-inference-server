# b70-inference-server

An LLM inference server written from scratch for the **Intel Arc Pro B70**, talking
to SYCL and Level Zero directly. No PyTorch, no vLLM, no `vllm-xpu-kernels`.

Scope is deliberately narrow: **one model family (`qwen3_5`), one math path (W4A16)**.
Specialisation is the entire strategy - a general engine cannot make the choices
this one can.

## Goals, in priority order

1. **Beat vLLM** on the same box, same model, same benchmark.
2. **Learn the metal** - Level Zero, XMX, the real cost model of this silicon.
3. **Own the dependencies** - a static binary that does not rot when a torch
   nightly moves.
4. **Share it** once it is good.

## The four levers

"Beat vLLM" rests on four specialisations a general engine cannot take. They are
independent of each other, and each is labelled by how well it is established.

| Lever | What it buys on the phase-1 model | Status |
|---|---|---|
| Fill the device at `M = 1` - GEMV, split-K, `M ∈ [1,8]` | the gap from vLLM's **81% MBU** to ~95%: ≤ ×1.2 | verified in source (doc 08); the only decode lever that is pure kernel work |
| Byte-identical Level Zero replay | zero host work per token - but ~700 kernels per token means per-kernel fixed cost is the new host overhead; fusion is part of this lever | established on MoE (~55% MBU); on the 27B dense it shares the same ≤ ×1.2 with the row above (doc 05) |
| Quantise `lm_head` (+ the MTP head) | 2.54 GB of 15.52 GB per token → ×1.14 at int4, ×1.09 at int8 | **measured** bytes; accuracy cost unmeasured (doc 07 #6) |
| MTP on a weight-stationary `M ∈ [1,8]` verify | draft traffic shared with the verify step; ~55-70 t/s ceiling vs vLLM's 45 (estimate) | phase 2 (doc 05) |
| W4A8 prefill on native s8×s4 DPAS | int8 systolic rate, zero dequantisation | atom exists in `sycl-tla`; no mainloop yet (doc 07 #9) |

On the 27B dense the roofline is **38.7 t/s** and vLLM already sits at 31.50.
Every decode lever above is worth ×1.1-1.2 on its own; they multiply to
~×1.4 at best (≈44 t/s) without speculation. Phase 2 is where the larger
numbers live. Doc 05 has the arithmetic; `W` is measured, not estimated.

## How this project is built - read this first

**The point is to understand this stack, not to produce code that happens to run.**
Speed is the scoreboard; comprehension is the product. A working kernel nobody can
explain is a failure here, not a milestone.

Concretely:

- **No code lands without an explanation of why it is shaped that way** - which
  call reaches it, what it hands to the next layer, what doing it differently
  would cost. If that cannot be written down, it is not understood well enough
  to keep.
- **Every design choice names its trade-off.** "Faster" is not a reason. *Faster
  at what, worse at what, measured how* is a reason. Alternatives get written
  down and rejected explicitly, never silently skipped.
- **Measure before claiming.** Every number in these docs is labelled measured or
  estimated. An estimate never becomes an assumption without being measured -
  see [docs/07-open-questions.md](docs/07-open-questions.md).
- **Read the source; do not guess the behaviour.** The reference checkouts below
  exist to be read. Claims drawn from source cite file and line. Claims that are
  not are marked as hypotheses.
- **Ask "who calls this, and why does it exist?"** before touching a component.
  If a layer's purpose cannot be stated in one sentence, its boundary is wrong.

Explanations of mechanism belong in these docs, not only in commit messages -
they are part of the deliverable.

Start with [docs/08-decode-vs-prefill.md](docs/08-decode-vs-prefill.md). It is the
one idea the whole design rests on.

## Phase ladder

Each phase adds exactly one capability. Do not start a phase before the previous
one beats vLLM on its own model.

| # | Model | Adds |
|---|-------|------|
| 0 | `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` | **done 2026-08-23** (GEMV re-measured 2026-08-24) - `W` ✅ 15.52 GB (doc 03), vLLM baseline ✅ 31.50 t/s (BENCHMARKS.md), replay floor ✅ 0.52 µs/kernel (doc 07 #5), GEMV ✅ 533 GB/s at N=5120 / 584 GB/s lm_head (doc 12), bandwidth ✅ 590 GB/s via L0 (doc 01) |
| 1 | same | dense + hybrid attention (48 GDN + 16 full), int4 g64. **Target: > 31.50 t/s tg256, ≥ 1973 t/s pp4096** |
| 2 | same | MTP speculative decoding on the shipped head. **Target: > 45.23 t/s** (vLLM, 2 draft tokens) |
| 3 | `olka-fi/Ornith-1.0-35B-MXFP4` | MoE (grouped GEMM) + MXFP4 |
| 4 | `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | MoE **and** MTP together |

## Platform

Linux (Ubuntu on the box) only. Models are fetched with `hf download` into the
standard HuggingFace cache or pointed to by absolute path; the server itself
never downloads anything.

## v1 definition of done

OpenAI-compatible endpoint (`/v1/completions`, `/v1/chat/completions`, SSE),
**single stream**, no continuous batching. Driven by the same `llama-benchy`
command used for every number in these docs, so comparisons are apples-to-apples.
The number to beat is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md):
`p314-t214-vxkp0`, no speculation, **pp4096 1973 / tg256 31.50**.

## Approach

**SYCL kernels + Level Zero replay for the decode step.** Kernels come from
`sycl-tla` templates where they exist and are hand-written where they do not.
The decode step is captured **once** into a Level Zero command list and replayed
per token, updating only pointers and counters - no allocation, no dispatch, no
host work in steady state.

On the phase-1 model vLLM is already at 81% of the memory roofline, so the
honest non-speculative headroom is ~×1.35 with every lever pulled, not 5×.
See [docs/05-perf-model.md](docs/05-perf-model.md) for the arithmetic and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the numbers it rests on. The
host-bound story is true of the MoE models (phases 3-4), where it is worth
~×2.

## Documents

| File | Contents |
|------|----------|
| [docs/01-hardware.md](docs/01-hardware.md) | Measured B70 / Xe2 facts and what they forbid |
| [docs/02-formats.md](docs/02-formats.md) | Why int4 W4A16, why not FP8 or OpenVINO IR |
| [docs/03-models.md](docs/03-models.md) | Architecture of each phase model |
| [docs/04-architecture.md](docs/04-architecture.md) | Component design and the decode replay loop |
| [docs/05-perf-model.md](docs/05-perf-model.md) | Roofline, headroom, what to measure |
| [docs/06-prior-art.md](docs/06-prior-art.md) | What to borrow from `sycl-tla`; traps inherited from the vLLM stack |
| [docs/07-open-questions.md](docs/07-open-questions.md) | Unverified claims, ranked by how much they'd change the design |
| [docs/08-decode-vs-prefill.md](docs/08-decode-vs-prefill.md) | **Why decode and prefill want opposite kernels** - why vLLM leaves 20-50% on the floor, why OpenVINO wins decode without XMX, and the `M = 2..8` gap neither engine serves |
| [docs/09-vllm-patch-postmortem.md](docs/09-vllm-patch-postmortem.md) | What months of patching vLLM's kernels actually bought (+9% prefill, −19% decode) and why - the evidence this project rests on |
| [docs/10-the-box.md](docs/10-the-box.md) | Access, hardware, cached models and their quirks, images, how to run the reference stack |
| [docs/11-tokenizer-and-chat-template.md](docs/11-tokenizer-and-chat-template.md) | The component the first draft forgot: BPE, chat template, streaming detokenisation, and what each option costs goal 3 |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | **The baseline numbers and the exact commands that produced them.** Every vLLM figure quoted elsewhere traces back here |

**Every number in these docs is labelled measured or estimated.** Estimates are
not load-bearing until measured - see doc 07 before trusting one.

## Reference checkouts

Local clones used as source material (all fresh as of 2026-08):

```
~/PycharmProjects/sycl-tla                  CUTLASS fork for Intel GPUs - the kernel templates
~/PycharmProjects/oneDNN                    Intel's JIT GEMM generator - what OpenVINO's prefill runs on,
                                            and where hardware capability is encoded as explicit gates
~/PycharmProjects/level-zero                L0 spec + loader
~/PycharmProjects/compute-runtime           Intel NEO driver (L0/OpenCL implementation)
~/PycharmProjects/intel-graphics-compiler   IGC - what ocloc runs
~/PycharmProjects/openvino                  GPU plugin: the GEMV kernel that beats vLLM at decode
~/PycharmProjects/openvino.genai            OpenVINO's C++ LLM pipeline
~/PycharmProjects/model_server               OpenVINO Model Server - reference for the HTTP/SSE layer
~/PycharmProjects/oneAPI-samples            SYCL/L0 idioms and working build recipes
~/PycharmProjects/nncf                      Quantisation algorithms, comparison point for AutoRound
~/PycharmProjects/auto-round                AutoRound quantiser + the ARK kernels we are replacing
~/PycharmProjects/vllm{,-xpu-kernels}       The incumbent, and the baseline to beat
~/PycharmProjects/{oneCCL,ucx,nixl}         Multi-GPU / disaggregated serving - NOT phases 1-4
```

See [docs/06-prior-art.md](docs/06-prior-art.md) for what to read in each, and
which files specifically.

## Box

`ssh user@box` - Dell T5810, 44 threads / 121 GB, 2× Arc Pro B70.
