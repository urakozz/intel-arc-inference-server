# Open questions

This page lists what is genuinely not known about the card and the numerics, ranked by how much the answer would
change the design. It is for anyone about to design around one of these points.

Nothing here is settled, and nothing here should become an assumption in a design without being measured first.
These are questions, not a to-do list.

Questions that have been answered are not kept here; their answers live where they are used. The DPAS rate
landscape (DPAS is the card's matrix multiply-accumulate instruction) is in [01-hardware.md](01-hardware.md), the
format decisions in [02-formats.md](02-formats.md), the time decomposition and the rejected optimisations in
[05-perf-model.md](05-perf-model.md), and the execution model in [04-architecture.md](04-architecture.md).

## Recently answered

- Is W4A4 (int4 weights times int4 activations) usable? No. The DPAS rate is real, 4.000x bf16, but int4
  activations roughly double the output error ([05-perf-model.md](05-perf-model.md)).
- Is W8A8 worth anything for prefill? Yes. The int8 prefill path, which turns the int4 weights into rotated int8 and
  does the matrix math in int8, is the opt-in `l0-int8` backend, 1.24x to 1.46x the bf16 pass on one card for
  Qwen3.8 ([16-int8-prefill.md](16-int8-prefill.md)).
- Does quantising `lm_head` (the output head that turns the last hidden state into vocabulary logits) to int8 cost
  accuracy? Not measurably against the gates, and it is the serving default
  ([05-perf-model.md](05-perf-model.md#1-quantise-lm_head)). The int4 head is still open, below.
- Why did the DPAS verify-attention kernel pick the wrong token at 8 rows? It rounded the softmax weights to bf16
  before the P x V product. Splitting each weight into two bf16 halves fixed it, and the f32 judge
  ([14-golden-gate.md](14-golden-gate.md)) now covers the kernel.
- Can a profile see inside the prefill walk? Yes. `B70_PREFILL_PROFILE=1` times each phase with Level Zero kernel
  timestamps. It perturbs the walk, so it attributes time between phases but does not give the headline number.
- Can a SYCL kernel share a Level Zero command list with ours? No longer asked: the sycl-tla path was removed and
  nothing in the engine uses SYCL.

## 1. What would an int4 `lm_head` cost in accuracy?

The speed is known: an int4 head saves 3.05 ms a token at record grade against the bf16 head, and the int8 head
already ships. The accuracy of an int4 head has never been measured, and `lm_head` is known to be sensitive to
quantisation.

A clean experiment does not exist yet. Comparing a self-quantised checkpoint against a published one confounds the
head with everything else in which two quantiser runs differ: the `quant_method` spelling, `packing_format`, whether
`g_idx` is shipped, how modules are excluded, and the published checkpoint's unpublished tuning. The experiment
that would answer it is one pair of checkpoints quantised by the same script and version, differing only in whether
the head is packed, judged by perplexity or `lm_eval`

## 2. Can the 24 MB L2 cache be used deliberately?

24 MB is unusually large for this class of card. A 4096x4096 int4 tile is 8 MB, so keeping several layers' weights
resident is conceivable. Whether that is achievable, or whether the replacement policy defeats it, is unknown, so
measure before designing around it. One data point the other way: L2 did absorb a 6x KV reread in the attention
kernel before register-packed GQA removed the reread. The cache helps when you do not plan for it, but that does not
show it can be planned around.

## 3. What sets the decode attention kernel's per-launch cost?

On an early driver stack the retiled attention launch cost 224.046 us, and its largest named term was the KV load
path at 55.4% (39.2% the 32-byte load messages themselves, 16.2% cache service). Four successive cost models failed
on it: two broke on a third depth point, a per-work-group occupancy ceiling broke on a bit-identical rewrite, and a
fill-fraction model and its refit broke on a block-size sweep. What is missing is a model that predicts the launch
time from the shape, which any further change here would have to be designed against. The kernel has been reworked
since (int8 KV readers, the DPAS verify attention), so such a model would have to be built on the current kernels.

A methodology warning belongs with it: holding an address constant is not a cache ablation. A loop-invariant
address through a `restrict` pointer may legally be hoisted out of the loop, IGC (the Intel Graphics Compiler) did
hoist it, and that turned a cache-miss probe into a message-count probe and inverted the answer. Make the hot
address a function of the loop variable.

## 4. Is the second card really slower at prefill?

In one session, with the same binary, the same checkpoint and the same ten minutes, prefill lost 3.2% to 3.4% on
device 1 and decode 0.2% to 0.4%, reproducing on two checkpoints and far outside the spreads. That pattern points to
sustained compute, not bandwidth. Neither card drives a display, both report identical unprivileged PCIe link
fields, and reading the clock frequencies needs root.

The most likely explanation is that a hardware monitoring tool (`hw-smi`) was open during that measurement; a monitor
that holds forcewake keeps the cards out of their idle state ([10-the-box.md](10-the-box.md#after-a-reboot)). The
cards are treated as equal for new runs, but the gap has not been re-measured with the tool closed, so the
validation tooling still keeps timed series rows on device 0, and a series row names its card.

## 5. The split-BF16 delta-net scan's remaining single-precision operand

The delta-net scan is the one approximation in the default prefill path. It is gated on tokens and state cosines
rather than bitwise, and it holds: 93 of 93 determined rows are exact, and state cosines are 0.9997 or better against
a 0.999 bar. One limit remains: `vn` inside the A2 term is still a single BF16 operand, measured harmless on three
prompts of one checkpoint but not proven in general. Whether a longer prompt, a different checkpoint or a different
numerical regime reaches it is untested. Details are in
[12-kernels.md, "How the prefill kernels got here"](12-kernels.md#how-the-prefill-kernels-got-here).
