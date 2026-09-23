# Open questions

What is genuinely not known, ranked by how much the answer would change the
design. Nothing here is settled, and nothing here should become an assumption in
a design document without being measured first.

Questions that have been answered are not listed. Their answers live where they
are used: the DPAS rate landscape in [01-hardware.md](01-hardware.md), the
format decisions in [02-formats.md](02-formats.md), the time decomposition and
the rejected optimisations in [05-perf-model.md](05-perf-model.md), the
execution model in [04-architecture.md](04-architecture.md).

## 1. Why is decode still about 5% behind vLLM, on the same bytes?

The standing rows are **29.45 t/s against 31.01**, byte-matched, on a roofline
of 37.97. This is the project's largest open question and it has resisted the
obvious answers.

What is ruled out, each by measurement:

- **Host overhead.** 99.77% of the step is inside the GPU fence and the host
  spends 97 us per token. There is nothing left to delete.
- **The linears.** They run at 89-97% of their per-shape bandwidth ceilings, and
  `lm_head` alone runs at 98.5% of the measured 590 GB/s.
- **`gdn_step`.** 1.7% of the step at 1.09x its own traffic floor, 92% of device
  bandwidth.
- **Launch count.** The in-situ dispatch gap is 0.473 ms, about 1.1% of a step.
- **The attention grid's early-out.** 0.11% of a step for 3072 idle
  work-groups.

What remains is the attention kernel's unexplained per-launch term, below, plus
the accumulation of several kernels each within a few percent of their own
ceilings. A 5% gap made of ten 0.5% terms is a different problem from a 5% gap
with one cause, and which of the two this is has not been established.

## 2. The 224 us per-launch term in `attn_decode`

**Four successive cost models have died on this kernel.** Two attention models
died to a third depth point, a per-work-group occupancy ceiling died to a
bit-identical rewrite, and a fill-fraction model and its refit died to a block
sweep.

What is measured: the retiled launch costs 224.046 us, its dominant named term
is the **KV load path at 55.4%** - 39.2% the 32-byte load messages themselves,
16.2% cache service - and the shape is throughput-bound rather than a latency
chain. What is not measured is a model that predicts the launch time from the
shape, which is what any further lever here would have to be designed against.

A methodology warning that belongs with it: **a "hold the address constant"
ablation is not a cache ablation.** A loop-invariant address over a `restrict`
pointer is legally hoistable, IGC hoisted it, and that turned a cache-miss probe
into a message-count probe and inverted the answer. Make the hot address a
function of the loop variable.

## 3. Is W4A4 reachable, and what does it cost in accuracy?

`i4_i4_matrix_mad_k64` measures **733.80 TIOP/s, 4.000x bf16**, and the assembly
shows `:s4` on both operands of a single `dpas.8x8`, so it is a genuine
single-instruction rate and not an emulation. The unsigned form measures
identically, so an asymmetric checkpoint's packing costs nothing.

**It is the only DPAS lever on this device worth more than 2x.** Everything
narrower buys nothing further, and FP4 is refused by the backend outright.

Unknown, and in this order:

- What 4-bit activations cost in output quality. The W4A8 probe already measured
  **2.79% relative L2 error** from int8 activations alone, driven by outliers,
  which is not encouraging for int4.
- Whether a W4A4 kernel would keep the production GEMM's 88.1% efficiency. If it
  did, the rate would be about 646 TIOP/s - an estimate by proportion with no
  kernel written.
- Whether the per-group rescale that sank W4A8 sinks this too. That cost was
  3.0 to 3.4 ms of a 5.440 ms kernel, and it does not obviously get cheaper.

The order matters: measure the accuracy on a probe before writing a mainloop.
That is the mistake W4A8 made and it cost a kernel.

## 4. Is W8A8 worth anything for prefill?

int8 XMX has 2x bf16 throughput and prefill is compute-bound. The rate is now
known exactly - `i8_i8_k32` at 366.90 TIOP/s - and it is **the same rate as the
mixed 4-bit form**, so W8A8 and W4A8 share one ceiling.

That makes W8A8 strictly worse than W4A8 on weight bytes for the same compute
ceiling, and W4A8 already lost. The remaining argument for W8A8 is that it needs
no per-group weight rescale, which is the term that sank W4A8 - but it would
need the checkpoint re-quantised to int8, and it still pays the activation
quantiser and the activation-outlier accuracy cost. Nobody has priced that
combination.

## 5. How much accuracy does quantising `lm_head` cost?

**The speed half is closed and the accuracy half has never been run.**

At int4 the head is 2.543 GB to 0.675 GB per token, worth a measured -3.05 ms
per token at record grade. `lm_head` is known to be quantisation-sensitive and
checkpoint authors routinely decline to quantise it, so this needs a perplexity
or `lm_eval` comparison against the unquantised baseline before anyone adopts
it.

**The clean experiment does not exist yet**, and that is the useful part of this
entry. Comparing a self-quantised checkpoint against a published one confounds
the head with everything else that differs between two quantiser runs:
`quant_method` spelling, `packing_format`, whether `g_idx` is shipped at all,
whether exclusions are regexes or per-module objects, how many `mtp` tensors
there are, and the published checkpoint's own unpublished tuning
hyper-parameters. The experiment that answers the title question is **one pair
quantised by the same script and version, differing only in whether the head is
packed**.

## 6. Can 24 MB of L2 be exploited deliberately?

Unusually large for this class of card. A 4096x4096 int4 tile is 8 MB, so
multi-layer weight residency is arguable. Whether it is achievable or whether
the replacement policy defeats it is unknown. Measure before designing around
it.

One data point in the opposite direction: L2 did absorb a 6x KV reread in the
attention kernel before register-packed GQA removed the reread. That says the
cache helps when you do not plan for it, not that it can be planned around.

## 7. The two cards are not interchangeable, and only prefill sees it

Same binary, same checkpoint, same ten minutes: prefill loses 3.2% to 3.4% on
the second card and decode loses 0.2% to 0.4%. The separation is far outside the
spreads and reproduces on two checkpoints.

That is a difference in sustained compute rather than in bandwidth, seen by the
compute-bound workload and not by the bandwidth-bound one. **Unattributed**:
neither card drives a display, both report identical unprivileged PCIe link
fields, and the frequency interface needs root. Until it is explained, every
series row in [BENCHMARKS.md](BENCHMARKS.md) is taken on one named card and says
so.

## 8. The split-BF16 scan's remaining single-precision operand

The delta net scan is the one approximation in the default prefill path. It is
gated on tokens and state cosines rather than bitwise, and it holds: 93 of 93
determined rows exact, state cosines 0.9997 and better against a 0.999 bar.

**One honest limit**: `vn` inside the A2 term is still a single BF16 operand,
measured harmless on three prompts of one checkpoint and not proven in general.
Whether a longer prompt, a different checkpoint or a different numerical regime
reaches it is untested. Details in
[prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md).

## 9. The profiler cannot see inside the prefill walk

`--profile` builds two captured steps and grades one against the other. The
prefill walk is not a capture and never will be, because its arguments are
resolved per launch, so the profile events have nothing to attach to.

**A cheap version exists and is priced.** `B70_PREFILL_PROFILE=1` times every
drain in the walk, which is a complete attribution when there is a drain at
every boundary. Its own overhead measures -2.0%, inside the run-to-run spread.
What it cannot do is separate two Level Zero kernels appended back to back
without inserting a drain between them - which it does, in profile mode only,
and which is why the profiled total is an upper bound.

**The full version is one task and is not scheduled**: an optional event pool
signalled per launch, giving device timestamps without extra drains. What is
*not* recommended is capturing the walk. Arguments change per chunk and per
layer, so there is nothing to capture.

## 10. Can a SYCL kernel be appended to a raw Level Zero command list at all?

The decode list deliberately contains no SYCL kernels, which sidesteps this, and
prefill now runs its own GEMM on the Level Zero list rather than sycl-tla's,
which sidesteps it again. So this is no longer on any critical path.

It is kept because it is cheap and it settles a real architectural question:
take a `sycl-tla` example, extract its `ze_kernel_handle_t` via
`sycl::get_native<backend::ext_oneapi_level_zero>` on a named kernel bundle,
append it to a regular command list, replay twice, diff. That would say whether
the two toolchains can ever share one list, which matters the moment a kernel
family exists that is genuinely easier to write in CuTe.
