# Know-how - what building this taught us

The distilled, transferable findings from writing an LLM inference engine
from scratch against a single GPU (Intel Arc Pro B70), specs 1-1.5,
2026-08-22 → 2026-08-25. Every number is measured on this box unless marked
otherwise; the authoritative derivation for each lives in the doc cited.

**The headline.** A ~15-kLOC engine (pure Level Zero + OpenCL C, no PyTorch,
no vLLM) decodes a 27B W4A16 model at **27.52 t/s** measured (record grade -
median of three on an idle box, docs/BENCHMARKS) - 87% of vLLM's 31.50 on the
same box/model - while reproducing a CPU-torch
oracle **96/96 greedy tokens exactly** (docs/14) and replaying **bitwise
deterministically**. Projected ceiling with the two remaining identified
items (attention tuning ~0.5 ms + int4 lm_head ≤3.26 ms, both bounded by
measurement): ~31.1 t/s - still under vLLM's number, and the roofline is
37.97 t/s. The silicon's own per-shape bandwidth ceilings, not host
overhead, are the binding constraint (docs/15).

## 1. The hardware (BMG-G31 / B70)

- **Real bandwidth is ~590 GB/s** (probe_bw, docs/01) - and the card
  **compresses uniform-fill buffers losslessly**, so naive bandwidth probes
  read impossible numbers (1022 GB/s). Seed probe buffers with xorshift.
- **Per-shape GEMV ceilings, not one number**: 533-584 GB/s depending on
  shape (89-99% of peak, docs/12 probe matrix). Plan against the per-shape
  ceiling; a single "peak bandwidth" figure lies by up to 11%.
- **The subgroup is the occupancy unit, not the work-group.** A bit-identical
  8-WG rewrite of a 2-WG kernel bought exactly zero; splitting the same work
  8→32→128 *subgroups* took 48.8→13.1→5.3 µs (docs/15 §L2). Falsified our
  own "per-work-group ceiling" model.
- **Single-work-group kernels are poison**: one Xe core streams ~13-17 GB/s.
  A 129-launch/token family of single-WG kernels cost 2.9 ms where its
  traffic was worth ~0.15 ms (docs/15 §L1).
- **Block size, not work-group count, is the attention knob - and the reason
  changed**: halving `ATTN_BLOCK` paid 39%, the second halving only 9%
  (docs/12 attn, docs/15 §L5). The `3.4× WGs → +6.9% time` measurement is real
  but it was taken at `ATTN_BLOCK` **256**, where the device is only half
  occupied; **at the shipped 64 the launch is linear in live work-groups**
  (68 → 1004 = 14.8× for 11.8× the time), so "work-group count is nearly free"
  and "it is a serial-walk latency problem" are both **withdrawn** (spec-1.6
  stage 0). Occupancy is most of why the retile paid.
- **The KV load path is the named dominant term** of the retiled attention
  launch: 55.4% - **39.2% the 32-byte load messages themselves, 16.2% cache
  service** - throughput-shaped, not a latency chain (spec-1.6 stage 0,
  docs/15). *The first, uncontrolled measurement read 30.9/24.4 and was voided
  by a compiler-hoisting control; quote the numbers above.*
- **A "hold the address constant" ablation is not a cache ablation**: a
  loop-invariant address over a `restrict` pointer is legally hoistable and IGC
  hoisted it, turning a cache-miss probe into a message-count probe and
  inverting the answer. Make the hot address a function of the loop variable
  (spec-1.6 stage 0, docs/15 §5.2 "methodology finding 2").
- **Warm up a device probe for ~100 launches, not one replay**: at 40 the same
  binary read 264 and 206 µs on consecutive invocations; from ~100 it locks to
  0.25%. `gemv_harness.h`'s 8-replays-drop-3 is the convention (spec-1.6
  stage 0).
- **L2 absorbed the 6× KV reread** (pre-Task-5; the reread is gone -
  register-packed GQA, docs/12). ~740 GB/s effective was itself later corrected
  down (docs/12 §probe finding 4).
- **Idle (early-out) work-groups cost ~15 ns each** - fixed grids with
  device-side early-out beat context-bucketed command lists at every depth
  measured (docs/07 #12).
- **GPU clock ramp invalidates short measurements**: consecutive cold probe
  runs read 264 vs 206 µs for the same kernel. Any probe needs the
  8-replay/drop-3 warm-up convention or it measures the ramp.

## 2. Level Zero as the whole runtime

- **Capture once, replay per token.** A regular in-order command list with
  every argument baked (645→774 kernels) replays at **0.4-0.7 µs/launch**;
  per-token host work is **97 µs** (0.3% of the step). The three values that
  change per token live in a device-resident control block the kernels read
  and the sampler writes (docs/04).
- Arguments are resolved **at append time** (proven by test) - mutable-args
  extensions are unnecessary for decode.
- **Kernel-timestamp events observe without perturbing**: per-launch
  device-clock durations exclude the signal's host-scope flush (only the
  inter-launch gap pays it), so in-situ per-kernel numbers are directly
  comparable to isolated probes. Two traps: query device properties with
  the `_1_2` stype or `timerResolution` is off by **369,231×**; mask to
  `kernelTimestampValidBits` before subtracting.
- **"Python overhead" was only a third of the story.** vLLM without XPU
  graphs loses ~3× to host overhead; with graphs it reaches 83% MBU. We
  removed host overhead more thoroughly (99.7% of the step in-kernel) and
  still trail - at batch 1 the endgame is kernel efficiency against
  per-shape bandwidth ceilings, not the host (docs/05).

## 3. Numerics you can trust

- **Per-op bf16 RNE discipline makes GPU==CPU token-exact reproduction
  achievable**: mirror torch's op-by-op rounding and the golden gate holds
  96/96 across three prompts - and stayed exact through three kernel
  rewrites (the worst gated drift across all of spec 1.5: **two words at
  1 ulp**, docs/14).
- **Tokens are robust far beyond intuition**: a 21% relative hidden-state
  deviation at one position still produced identical greedy tokens. Cosine
  diagnostics move both directions under legal reassociation - token
  exactness is the gate, tensors only diagnose (spec §11 ruling).
- **Real checkpoints contain subnormal f16 scales** (1,658 of them here).
  `-cl-denorms-are-zero` corrupts real models; our build makes it a fatal
  error. OpenCL's default divide is 2.5-ulp - pass
  `-cl-fp32-correctly-rounded-divide-sqrt` (measured cost: none).
- **A cross-implementation oracle is worth its weight**: dequant fixture
  (bit-exact in both languages) → CPU-torch golden files → engine gate →
  vLLM cross-check. When all three agree 96/96, every future divergence has
  a clean three-way baseline (docs/14).

## 4. Measurement method (the most transferable part)

- **Pre-register predictions, then measure.** Five cost models died to
  single new data points here (two attention models to a third depth, the
  per-WG ceiling to a bit-identical rewrite, F+fill·P and its refit to a
  block sweep). A two-point fit always fits; it never explains. The
  discipline: write both models' predictions down *before* the run.
- **Measure the instrument before trusting it**: run-to-run drift was
  0.3-0.5% (~0.15 ms) - larger than every remaining lever. Repeated-run
  averaging brought the attribution floor to **0.051 ms/step with one
  unexplained outlier family and 0.011 ms without it** (2σ, cross-process, n=3
  so both carry a 4.42× upper bound). Quote both or neither: the larger figure
  is 103% accounted for by that one family. Attribute nothing below the floor,
  and note that within-PROCESS repeatability is not the floor - one family
  reproduced to 0.3% inside a process and landed 31% apart between three.
- **Record-grade vs iteration-grade runs**: absolutes come only from an
  idle box, medians-of-3, sha-named binaries; everything else is labelled
  relative/under-load. Every number in every doc carries measured vs
  estimated.
- **The no-two-values rule**: one quantity, one value, or an explicit
  reconciliation sentence. Most review findings across four plans were
  exactly this rule being violated by *derived* survivors that literal
  greps missed - sweep derived figures, not just literals.
- **The gate brackets every change**: golden gate before AND after each
  kernel lever (a mid-plan rule that caught nothing and cost minutes -
  cheap insurance on a numerics-order change).

## 5. Quantization know-how

- **Untangle AutoRound's taxonomy**: the *algorithm* (AutoRound sign-SGD or
  RTN), the *export format* (`auto_gptq` vs `auto_round` vs
  `auto_round:auto_gptq` - different config keys and zeros conventions:
  v1 `0x77777777` vs v2 `0x88888888`), and the *runtime kernels* (chosen by
  the serving stack from the config; the only layer where "format speed"
  exists). Our engine repacks at load, so format never affects our speed.
- **MXFP4 buys nothing for bandwidth-bound decode**: 4.25 bits/weight
  effective - identical to GPTQ g64 (4 + 16/64 = 4 + 8/32). Its home is
  fp4 tensor-core compute, not weight streaming.
- **lm_head is the forgotten 12%**: the largest single tensor (2.54 GB/token
  read at 98.5% of achievable bandwidth - nothing to tune, only bytes to
  remove; ≤3.26 ms/token available, bandwidth-bounded). Public uploads
  don't quantize it; AutoRound supports it only in `auto_round*` export
  formats, and its model-free path silently skips it (config records the
  intent, tensor stays bf16 - probe the artifact, never trust the config).
- **RTN is the right tool for format bring-up**: minutes instead of hours,
  byte-identical layout to tuned output; the correctness machinery
  (oracle + gate) re-anchors to whatever checkpoint you feed it, so quant
  accuracy and format validation are independent axes.
- g64 over the g128 default: finer scale granularity for ~3% more bytes,
  and it matches a 64-element-K tile with inline f16 scales - pick the
  group size your kernels' tile geometry wants.

## 6. Process (how 3 days stayed on rails)

- Spec → plan → fresh implementer per task → skeptical review per task →
  scoped re-review per fix round → whole-branch final review. The reviews
  caught, among others: a wrong tensor shape in the spec itself, a
  557-KB-SLM design that could not compile, an arithmetic error in the
  launch-count derivation (×11 vs ×12), and a memo headline whose corrected
  arithmetic **inverted the conclusion** (lm_head int4 alone does not clear
  the bar).
- Rulings, not stalls: every deviation from plan text is decided against
  the spec, written down with its cost-if-wrong, and surfaced at close.
- The ledger survives context loss; the review packages make every diff
  auditable; probes and negative results are recorded with the same care
  as wins ("do not re-derive this idea" fences in kernel comments).
