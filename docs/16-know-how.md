# Know-how - what building this taught us

The transferable findings from writing an LLM inference engine from scratch
against one GPU. Every number is measured on the box unless marked otherwise;
the derivation for each lives in the doc cited.

**The headline.** A 15-kLOC engine - pure Level Zero and OpenCL C, no PyTorch,
no vLLM, no vendor kernel library in the hot path - runs prefill at
**1670.72 t/s against vLLM's 1610.04** on the same card and the same checkpoint
files, and decode at **29.45 t/s against 31.01**. Good prefill, decode still
behind. It reproduces a CPU-torch oracle element-exact on every determined token
row and replays bitwise deterministically.

**What that headline is not.** It is not efficient yet. Decode sits at 77.6% of
the measured bandwidth against vLLM's 81.7%, and prefill sustains about
81.8 TFLOP/s against a 183.45 TFLOP/s pipe. Beating the reference
implementation and using the silicon well turned out to be different questions,
and only one of them is answered.

## 1. The hardware (BMG-G31 / B70)

- **The DPAS array issues one `dpas.8x8` every 16 EU-cycles regardless of data
  type.** Ops per second therefore scale exactly with the builtin's K depth:
  bf16 and fp16 at k16 are 183.45 TFLOP/s, int8 and *every* mixed 4-by-8 form at
  k32 are 2.000x, int4 and int2 at k64 are 4.000x. The bf16 figure is 99.97% of
  the clock-derived peak. Depth scaling is free; width is not (docs/01).
- **Halving an operand only helps when it halves K.** The mixed s8 x s4 DPAS is
  native and real - the assembly carries `:s4` on one operand of a single
  instruction - and it runs at *exactly* the int8 rate, because its K is 32.
  A W4A8 kernel built on that assumption measured slower than the bf16 control.
- **Real bandwidth is about 590 GB/s**, and the card **compresses uniform-fill
  buffers losslessly**, so naive probes read impossible numbers (1022 GB/s above
  a 608 GB/s theoretical peak). Seed probe buffers with xorshift.
- **Per-shape GEMV ceilings, not one number**: 533 to 584 GB/s depending on
  shape, 89% to 99% of peak (docs/12). Plan against the per-shape ceiling; a
  single "peak bandwidth" figure lies by up to 11%.
- **The subgroup is the occupancy unit, not the work-group.** A bit-identical
  8-work-group rewrite of a 2-work-group kernel bought exactly zero; splitting
  the same work across 8 → 32 → 128 *subgroups* took it from 48.8 to 13.1 to
  5.3 us. That falsified our own per-work-group ceiling model.
- **Single-work-group kernels are poison**: one Xe core streams about 13 to
  17 GB/s. A 129-launch-per-token family of them cost 2.9 ms where its traffic
  was worth 0.15 ms.
- **Idle early-out work-groups cost about 15 ns each**, so fixed grids with
  device-side early-out beat context-bucketed command lists at every depth
  measured (docs/04).
- **`ocloc` has no 16-bit block write wider than 8 rows**, so a bf16 epilogue is
  bound by store message count rather than by bytes. Halving intermediate buffer
  widths changed nothing at all.
- **GPU clock ramp invalidates short measurements**: consecutive cold runs of
  the same binary read 264 and 206 us. Warm a device probe for about 100
  launches, not one replay; from there it locks to 0.25%.

## 2. Level Zero as the whole runtime

- **Capture once, replay per token.** A regular in-order command list with every
  argument baked - 774 kernels across 19 modules - replays at 0.4 to 0.7 us per
  launch, and per-token host work is 97 us, 0.3% of the step. The three values
  that change per token live in a device-resident control block the kernels read
  and the sampler writes (docs/04).
- **Arguments are resolved at append time**, proven by test, so mutable-argument
  extensions are unnecessary for decode.
- **More launches can be faster.** The Level Zero prefill walk issues 8,689
  launches per chunk against sycl-tla's 1,201 and has **zero** host waits
  against 656. Launch count was never the cost; cross-runtime boundaries were.
- **Two software queues on this device share one hardware compute queue and
  cannot be ordered against each other.** The driver reports exactly one compute
  queue and refuses any second index, so every L0-to-SYCL boundary is a host
  wait at about 22 us. If a design needs two runtimes to interleave on this
  card, it does not; pick one.
- **Kernel-timestamp events observe without perturbing**: per-launch
  device-clock durations exclude the signal's host-scope flush, which lands in
  the inter-launch gap instead, so in-situ per-kernel numbers are directly
  comparable to isolated probes. Two traps: query device properties with the
  `_1_2` stype or `timerResolution` is off by 369,231x, and mask to
  `kernelTimestampValidBits` before subtracting.
- **"Python overhead" was only a third of the story.** vLLM without graphs loses
  about 3x to host overhead; with graphs it reaches 81.7% MBU. We removed host
  overhead more thoroughly - 99.7% of the step is in-kernel - and still trail at
  batch 1. The endgame there is kernel efficiency against per-shape bandwidth
  ceilings, not the host (docs/05).

## 3. Numerics you can trust

- **Per-op bf16 round-to-nearest-even discipline makes GPU-equals-CPU
  token-exact reproduction achievable.** Mirror torch's op-by-op rounding and
  the golden gate holds across kernel rewrites; the worst gated drift across a
  whole rewrite ladder was two words at 1 ulp (docs/14).
- **Tokens are robust far beyond intuition.** A 21% relative hidden-state
  deviation at one position still produced identical greedy tokens, and cosine
  diagnostics move in *both* directions under legal reassociation. Token
  exactness is the gate; tensors only diagnose.
- **The reference may not determine the answer.** The oracle's logits sit on the
  bf16 grid, so it cannot resolve two candidates inside one ulp. A row whose
  golden top-1 is not unique is undetermined, and the gate asserts membership of
  the argmax set there rather than grading the engine on `torch.argmax`'s
  lowest-index tie-break (docs/14).
- **Real checkpoints contain subnormal f16 scales** - 1,658 of them here.
  `-cl-denorms-are-zero` corrupts real models, so our build makes it a fatal
  error. OpenCL's default divide is 2.5 ulp; pass
  `-cl-fp32-correctly-rounded-divide-sqrt`, whose measured cost is none.
- **A cross-implementation oracle is worth its weight.** Dequant fixture
  (bit-exact in both languages) → CPU-torch golden files → engine gate → vLLM
  cross-check. When all three agree, every future divergence has a clean
  three-way baseline (docs/14).
- **A green gate is not evidence unless the run says which kernel it
  dispatched.** A record was once claimed for a kernel that never ran: the
  selector resolved, the gates went green, and the old kernel was what launched.
  Every gate run now prints the entry it launched, read back from the pointer
  the launcher handed the kernel cache, and the tests assert it matches the
  selector.

## 4. Measurement method (the most transferable part)

- **Price the mechanism before building the kernel.** Three optimisations were
  built, measured and rejected - fusing int4 dequant into the GEMM, W4A8 on the
  mixed DPAS, and removing barriers from the GDN triangular solve - and in all
  three the thing that looked like the cost was not the cost. The barrier one
  was wrong by about 18x: all 128 barriers are 4.6% of a row whose cost is the
  serialised recurrence itself.
- **Pre-register predictions, then measure.** Five cost models died to single
  new data points here. A two-point fit always fits; it never explains. Write
  both models' predictions down *before* the run.
- **Measure the instrument before trusting it.** Day-scale drift on a
  same-checkpoint control is at or under 0.09%. Repeated-run averaging brought
  the attribution floor to 0.051 ms/step with one unexplained outlier family and
  0.011 ms without it. Attribute nothing below the floor. Within-*process*
  repeatability is not the floor: one family reproduced to 0.3% inside a process
  and landed 31% apart across three.
- **Record-grade against iteration-grade runs.** Absolutes come only from a
  provably idle box, medians of three; everything else is labelled relative or
  under load, and every number carries measured against estimated.
- **The no-two-values rule**: one quantity, one value, or an explicit
  reconciliation sentence. Most review findings on this project were exactly
  that rule being violated by *derived* figures a literal grep missed. Sweep
  derived figures, not just literals.
- **An ablation can measure something other than what it names.** A "hold the
  address constant" cache ablation is not a cache ablation: a loop-invariant
  address over a `restrict` pointer is legally hoistable, IGC hoisted it, and it
  turned a cache-miss probe into a message-count probe and inverted the answer.
  Make the hot address a function of the loop variable.
- **The gate brackets every change.** Golden gate before *and* after each kernel
  lever. It has caught nothing so far and costs minutes, which is cheap
  insurance on a change that reorders floating-point sums.

## 5. Quantization know-how

- **Untangle AutoRound's taxonomy**: the *algorithm* (sign-SGD or RTN), the
  *export format* (`auto_gptq` against `auto_round` against
  `auto_round:auto_gptq`, which differ in config keys and zeros conventions -
  v1 `0x77777777` against v2 `0x88888888`), and the *runtime kernels*, chosen by
  the serving stack from the config. Only the last is where "format speed"
  exists, and our engine repacks at load, so format never affects our speed.
- **MXFP4 buys nothing here twice over.** It is 4.25 bits/weight effective,
  identical to GPTQ g64, so it is not a bandwidth win; and FP4 DPAS is refused
  by name by this backend while every microscaling entry point crashes the
  compiler, so it is not a compute win either.
- **`lm_head` is the forgotten 12%**: the largest single tensor, 2.54 GB read
  per token at 98.5% of achievable bandwidth. Nothing to tune, only bytes to
  remove. Public uploads do not quantise it, AutoRound supports it only in the
  `auto_round*` export formats, and its model-free path silently skips it - the
  config records the intent and the tensor stays bf16, so **probe the artifact,
  never trust the config**.
- **A checkpoint advantage is not a kernel win.** An int4 head made this engine
  measure ahead of vLLM on decode, and that margin was retired rather than
  quoted: vLLM would be faster with one too and simply cannot load one, so the
  number measured an inability of the comparison. Byte-matched or not at all.
- **RTN is the right tool for format bring-up**: minutes instead of hours,
  byte-identical layout to tuned output, and the correctness machinery re-anchors
  to whatever checkpoint you feed it, so quantisation accuracy and format
  validation stay independent axes.
- **Pick the group size your kernels' tile geometry wants.** g64 over the g128
  default costs about 3% more bytes and matches a 64-element-K tile with inline
  f16 scales.

## 6. Process

- **Review found things testing did not.** Among them: a wrong tensor shape in a
  design document, a 557 KB SLM design that could not have compiled, an
  arithmetic error in a launch-count derivation, and a summary whose corrected
  arithmetic **inverted its own conclusion**.
- **Record negative results with the same care as wins.** They saved more time
  here than the wins did, and every one of them lives in a dated record with its
  mechanism, not just its verdict.
- **Fence the dead ends in the code.** A "do not re-derive this idea" comment at
  the kernel that would have carried the optimisation is worth more than the
  document that explains why, because it is where the next person will be
  standing.
