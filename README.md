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
| Quantise `lm_head` (+ the MTP head) | 2.54 GB of 15.52 GB per token → ×1.14 at int4, ×1.09 at int8 | **measured** bytes; accuracy cost unmeasured (doc 07 #6). In situ the row is **4.376 ms of the measured 36.32 ms token** and int4 would cut ~3.3 (estimated) - the largest item left in the step, and **not sufficient on its own** to clear 31.50 (memo, doc 15) |
| MTP on a weight-stationary `M ∈ [1,8]` verify | draft traffic shared with the verify step; ~55-70 t/s ceiling vs vLLM's 45 (estimate) | phase 2 (doc 05) |
| W4A8 prefill on native s8×s4 DPAS | int8 systolic rate, zero dequantisation | atom exists in `sycl-tla`; no mainloop yet (doc 07 #9) |

On the 27B dense the roofline is **38.7 t/s** (600 GB/s ÷ 15.52 GB; 37.97 on
the loader's measured 15.540 GB and the 590 GB/s a Level Zero launch actually
gets) and vLLM already sits at 31.50. Every decode lever above is worth ×1.1-1.2
on its own; they multiply to ~×1.4 at best (≈44 t/s) without speculation. Phase
2 is where the larger numbers live. Doc 05 has the arithmetic; `W` is measured,
not estimated.

**Measured state, 2026-08-25 at the spec 1.5 gate: 27.54 t/s against vLLM's
31.50 - still below it, 12.6% short.** MBU is **72.5%** (428 GB/s of the
measured 590) against vLLM's **83.0%**, on a 37.97 t/s roofline. The first
measurement of this engine was 23.73 t/s / 24.7% short (2026-08-25, `62bdd4d`);
spec 1.5's lever ladder took it to 27.54 in three measured cuts - the `a‖b`
GEMV K-split (−1.875 ms/token), the `prep_res_norm` two-stage reduction
(−2.220) and the `ATTN_BLOCK` 256 → 64 attention retile (−1.726) - with the
golden gate **96/96 element-exact** at every one. That closed **56% of the
10.395 ms/token gap** it started with and **did not close the gate**: 4.574
ms/token remain.

Lever 2 of the four below delivered completely (0.3% of the step is host time)
and lever 1 is 89-97% of roofline where it was probed; what the ladder found is
that the rest of the token is neither. The levers are not wrong; they were not
the whole cost model. Doc 05's "Phase 1, measured" section is the honest
version, and
[the spec 1.5 re-assessment memo](docs/superpowers/specs/2026-08-25-spec1.5-reassessment.md)
is why the ladder stopped there and what a spec 1.6 would cost - its headline
being that quantising `lm_head` (~3.3 ms, estimated, lever 3 below) is
**necessary but not sufficient**: it lands at 30.62 t/s, 0.88 t/s under the
bar, so a second item is required and none has a measured price yet.

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
| 1 | same | dense + hybrid attention (48 GDN + 16 full), int4 g64. **Target: > 31.50 t/s tg256, ≥ 1973 t/s pp4096.** Decode core **done 2026-08-25** (tag `decode-core-done`); the decode optimisation pass, spec 1.5, **closed short 2026-08-25** (tag `spec1.5-done`): 774-kernel replayed list, 96/96 golden tokens vs the CPU oracle at every step, `b70-decode --bench` and `--profile`. **Target NOT met: tg256 @ depth 4096 = 27.54 t/s measured at the spec 1.5 gate, 12.6% short of 31.50** (median of 3, spread 0.04%, `tools/bench_decode.sh`, BENCHMARKS.md) - it was 23.73 t/s / 24.7% short before the lever ladder. **72.5% MBU against vLLM's 83.0%**, on a 37.97 t/s roofline. 99.7% of the step is inside the fence - the gap is kernel time, not host time: **29.01 ms of the 36.32 ms token is GEMV** (79.9%, **measured per kernel in situ** - `b70-decode --profile`, doc 15 - and untouched by every lever; the `probe_gemv` transplant that read 28.35 was a floor and was right to 2.3%), and the non-GEMV remainder is ~7.3 ms, down from 13.14. The three levers, all measured on the bench: `a‖b` GEMV K-split **−1.875 ms**, `prep_res_norm` two-stage **−2.220**, `ATTN_BLOCK` 256 → 64 **−1.726**. Why the ladder stopped there and what a spec 1.6 would cost is [the re-assessment memo](docs/superpowers/specs/2026-08-25-spec1.5-reassessment.md), whose two information items are **done, 2026-08-25, and changed no engine code**: `b70-decode --profile --repeats R` measures the attribution floor at **0.051 ms/step** (it was an unmeasured "±0.1 ms"), and `tools/probe/probe_attn` names `attn_decode`'s dominant term - **the KV load path, 55.4% of the launch**, throughput-shaped, with ~0.5 ms/token of the 3.585 reachable by tuning and the rest structural (doc 15, "Spec 1.6 §5.2" and "§5.4"). No prefill kernel yet, so there is no pp4096 number |
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
| [docs/12-kernels.md](docs/12-kernels.md) | The measured GEMV kernels: the two int4 layouts, split-K, and which won on which shape |
| [docs/13-loader.md](docs/13-loader.md) | **Checkpoint → canonical device buffers**: snapshot rules, the index as manifest, every quantisation assert and what measured it, the fusion table, the `1+w` bake, and the resident-byte cross-check against `W` |
| [docs/14-golden-gate.md](docs/14-golden-gate.md) | **The engine == the CPU oracle == vLLM**: 3 prompts × 32 greedy tokens element-exact all three ways, the named divergence classes and their measured magnitudes, and the closed trust chain |
| [docs/15-step-anatomy.md](docs/15-step-anatomy.md) | **The decode step, launch by launch**: every kernel timed in situ (645 then, 774 now), the aggregate bucket finally split, the measured dispatch gap, the ranked lever ladder spec 1.5 executed, and the closing per-lever ledger |
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
