# Open questions

What is genuinely not known, ranked by how much the answer would change the design. These are questions about
the card and the numerics, not work items: scheduled work is in [open-work.md](open-work.md), ruled-out work
in [low-priority-work.md](low-priority-work.md). Nothing here is settled, and nothing here should become an
assumption in a design document without being measured first.

Questions that have been answered are not listed; their answers live where they are used: the DPAS rate
landscape in [01-hardware.md](01-hardware.md), the format decisions in [02-formats.md](02-formats.md), the
time decomposition and the rejected optimisations in [05-perf-model.md](05-perf-model.md), the execution
model in [04-architecture.md](04-architecture.md). Recently answered, in one line each:

- **Is W4A4 (int4 weights times int4 activations) reachable?** No: the DPAS rate is real (4.000x bf16) but the
  activations roughly double the output error ([probe](probes/probe-w4a4-2026-10-28.md), doc 05).
- **Is W8A8 worth anything for prefill?** Yes: the int8 prefill path (int4 weights turned into rotated int8, int8
  matrix math) is the `l0-int8` backend, about 30% faster than the bf16 walk (doc 17, BENCHMARKS.md).
- **Does quantising `lm_head` to int8 cost accuracy?** Not measurably against the gates; it is the serving default
  (spec 9, BENCHMARKS.md "int8 lm_head"). The int4 head is the open part, below.
- **Why did the DPAS verify-attention kernel pick the wrong token at 8 rows?** It rounded the softmax weights to
  bf16 before the P x V product. Splitting them into two bf16 halves fixed it, and the f32 judge (doc 14) now
  covers the kernel ([probe](probes/probe-verify-attn-2026-11-03.md)).
- **Can a profile see inside the prefill walk?** Yes: `B70_PREFILL_PROFILE=1` times each phase with Level Zero
  kernel timestamps (it perturbs the walk, so it attributes and does not give the headline wall).
- **Can a SYCL kernel share a Level Zero command list with ours?** No longer asked: the sycl-tla path was removed
  and nothing in the engine uses SYCL.

## 1. What sets `attn_decode`'s per-launch term?

At the 2026-08 stack the retiled attention launch cost 224.046 us, its dominant named term the **KV load path at
55.4%** (39.2% the 32-byte load messages themselves, 16.2% cache service), and **four successive cost models
died** on it: two to a third depth point, a per-work-group occupancy ceiling to a bit-identical rewrite, a
fill-fraction model and its refit to a block sweep. What is missing is a model that predicts the launch time from
the shape, which any further lever here would have to be designed against. The kernel has been reworked since
(int8 KV readers, the DPAS verify attention), so a model would be built on the current kernels.

A methodology warning that belongs with it: **a "hold the address constant" ablation is not a cache ablation.** A
loop-invariant address over a `restrict` pointer is legally hoistable, IGC hoisted it, and that turned a
cache-miss probe into a message-count probe and inverted the answer. Make the hot address a function of the loop
variable.

## 2. What would an int4 `lm_head` cost in accuracy?

The speed is known (-3.05 ms a token at record grade against the bf16 head) and the int8 head already ships. The
accuracy of the int4 head has never been run, and `lm_head` is known to be quantisation-sensitive. A clean
experiment does not exist: comparing a self-quantised checkpoint against a published one confounds the head with
everything else two quantiser runs differ in (`quant_method` spelling, `packing_format`, whether `g_idx` is
shipped, how modules are excluded, the published checkpoint's unpublished tuning). The answering experiment is
**one pair quantised by the same script and version, differing only in whether the head is packed**, judged by
perplexity or `lm_eval`.

## 3. Can 24 MB of L2 be exploited deliberately?

Unusually large for this class of card. A 4096x4096 int4 tile is 8 MB, so multi-layer weight residency is
arguable. Whether it is achievable or the replacement policy defeats it is unknown. Measure before designing
around it. One data point the other way: L2 did absorb a 6x KV reread in the attention kernel before
register-packed GQA removed the reread, so the cache helps when you do not plan for it; that does not show it can
be planned around.

## 4. Why is the second card slower, and only at prefill?

Same binary, same checkpoint, same ten minutes: prefill loses 3.2% to 3.4% on device 1 and decode 0.2% to 0.4%,
reproducing on two checkpoints and far outside the spreads (BENCHMARKS.md, "The two cards are not
interchangeable"). That is a difference in sustained compute, not bandwidth. **Unattributed**: neither card drives
a display, both report identical unprivileged PCIe link fields, and the frequency interface needs root. Until it
is explained, every series row in [BENCHMARKS.md](BENCHMARKS.md) is taken on one named card and says so.

## 5. The split-BF16 scan's remaining single-precision operand

The delta net scan is the one approximation in the default prefill path. It is gated on tokens and state cosines
rather than bitwise, and it holds: 93 of 93 determined rows exact, state cosines 0.9997 and better against a
0.999 bar. **One honest limit**: `vn` inside the A2 term is still a single BF16 operand, measured harmless on
three prompts of one checkpoint and not proven in general. Whether a longer prompt, a different checkpoint or a
different numerical regime reaches it is untested. Details in
[12-kernels.md, "How the prefill kernels got here"](12-kernels.md#how-the-prefill-kernels-got-here).
