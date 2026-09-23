# Performance model

Where the time goes, what the ceilings are, and which levers are left. Every
figure is labelled **measured**, **derived** (arithmetic on measured inputs,
with the inputs shown) or **estimated**. Every measured row and its grade lives
in [BENCHMARKS.md](BENCHMARKS.md); this document is the reasoning over them.

## The two ceilings

Decode and prefill are bound by different things, which is the whole reason
[08-decode-vs-prefill.md](08-decode-vs-prefill.md) exists.

### Decode: bandwidth

At batch 1 every weight is read once per token, so

```
tokens/sec_ceiling = read_bandwidth / bytes_read_per_token = 590 GB/s / W
```

`590 GB/s` is measured through a plain Level Zero launch (doc 01) - the launch
path this engine actually submits on. `W` is measured by the loader from what it
makes resident: **15,539,980,288 B = 15.540 GB per token** (doc 13), which is
doc 03's 15.519 GB of header arithmetic plus 0.016 GB of tiling pad and 0.005 GB
of fp32-widened norms.

```
ceiling = 590 / 15.540 = 37.97 t/s   (26.34 ms/token)   [derived]
vLLM    =               31.01 t/s -> 482 GB/s -> 81.7% MBU
this    =               29.45 t/s -> 458 GB/s -> 77.6% MBU   [derived]
```

**Use 590 GB/s and 15.540 GB, and use them on both sides of any comparison.**
Doc 01 also recommends 600 GB/s as a general hardware denominator and doc 03's
header arithmetic gives 15.519 GB; those are the same measurements on different
denominators, not different measurements, and mixing them inside one comparison
moves a figure by about 2%.

### Prefill: compute

The forward pass is **48.97 GFLOP/token** (derived from the doc 03 shapes), so
4096 tokens in 2.4516 s is **81.8 TFLOP/s sustained** (derived). The DPAS pipe
runs bf16 at **183.45 TFLOP/s measured** (doc 01) and our own GEMM reaches
**161.7 TFLOP/s measured** on the largest linear, 88.1% of it.

The gap between 81.8 and 161.7 is not kernel quality: it is everything that is
not the GEMM. The linear path alone is **199 TFLOP of per-request work**, and
the dequant pass that feeds the GEMM is a second pass over the same weights.
That is the shape of the remaining prefill headroom, and it is why the three
rejected optimisations below all attacked the dequant-plus-GEMM pair.

## Where decode time goes

The one full per-kernel in-situ profile of a decode step used Level Zero kernel
timestamps on the replayed list, with every launch timed in the step it actually
runs in. It was taken on a **42.141 ms** step, before any of the levers below
were cut, and it is kept whole because it is what those levers were chosen from.

| part | launches | ms/token | share |
|---|---|---:|---:|
| GEMV - the int4 mixers and MLPs and the bf16 `lm_head` | 257 | **29.005** | 68.8% |
| `attn_decode` | 16 | 5.782 | 13.7% |
| `prep` family | 241 | 3.573 | 8.5% |
| the `a‖b` GEMV | 48 | 2.335 | 5.5% |
| `gdn_step` | 48 | 0.733 | 1.7% |
| `attn_prep` + `attn_reduce` | 32 | 0.127 | 0.3% |
| dispatch gap (fence wall minus the sum of kernel durations) | - | 0.473 | 1.1% |
| host, outside the fence entirely | - | 0.097 | 0.23% |
| **total** | **645** | **42.141** | 100% |

It is a **partition**: every launch and the host appear once and the rows sum to
the measured step, which is why they carry three decimals where the prose rounds
to two. The launch count is 645 because this predates the `prep` split that took
it to 774.

Four things that table settled.

1. **The host-overhead thesis is finished for this engine.** 99.77% of the step
   is inside `execute` plus `fence.wait()`; the host spends 97 us per token
   writing four bytes and submitting. There is no host work left to delete. The
   thesis was always about vLLM's Python and Triton launch path (doc 09), and
   replay did exactly what doc 08 said it would - which is precisely why the
   remaining gap is *not* excusable as overhead. It is kernel time.
2. **The linears are not the problem.** They run at 89-97% of their per-shape
   bandwidth ceilings, and `lm_head` alone is 4.376 ms at **581 GB/s, 98.5% of
   the measured 590**. There is nothing to tune there; there are only bytes to
   remove.
3. **`gdn_step` is exonerated.** 48 launches moving 396 MB is 0.67 ms at the
   roofline, and it measures 0.733 - 1.09x its own traffic floor, at 92% of
   device bandwidth. The kernel the design worried most about is 1.7% of the
   step and the best-occupied thing in the engine.
4. **The occupancy problem is real, and the unit is the subgroup.** The
   expensive kernels were the ones given one or two work-groups. Giving `a‖b`
   4x the work-groups at an unchanged subgroup count bought exactly nothing;
   splitting its K 16 ways, 8 subgroups to 128, took it from 2.335 to
   **0.256 ms/token**.

### The levers that followed, all measured

| lever | before | after | bench delta |
|---|---:|---:|---:|
| `a‖b` GEMV K-split, 8 → 128 subgroups | 2.335 | 0.256 | **-1.875 ms/token** |
| `prep_res_norm` split into a two-stage reduction, 16 → 320 subgroups | 2.893 | 0.484 | **-2.220 ms/token** |
| attention block width 256 → 64 | 5.782 | 3.585 | **-1.726 ms/token** |

42.141 → **36.32 ms/token**, +16.1% throughput, 62.5% → 72.5% MBU, and the
golden gate green at every step. The remaining ladder was then ranked and
stopped: `gdn_step` at 92% of device bandwidth has 0.06 ms in the whole kernel,
and the dispatch gap is 0.473 ms, so neither is worth a golden-gate run.

Two things worth knowing about that ladder.

**The launch count went up on purpose.** Splitting `prep_res_norm` added 129
launches to buy 2.4 ms, at a derived 0.733 us each. Launch count is not the
lever it was once feared to be, in either direction.

**Attention did not close its prediction.** It was priced at -2.045 ms and
measured -1.73, 0.32 ms apart, and the difference is recorded as unattributed
rather than explained away. What is left inside that kernel is a per-launch term
of about 224 us that four successive cost models failed to explain. The named
dominant term of the retiled launch is the **KV load path at 55.4%** - 39.2% the
32-byte load messages themselves, 16.2% cache service - which is
throughput-shaped rather than a latency chain.

The step has since moved from 36.32 to **33.96 ms/token** (derived from the
standing 29.45 t/s) through the loader and per-shape retune work recorded in
BENCHMARKS.md. **Decode trails vLLM by about 5% on the same bytes**, and none of
the prefill work since moved it, by construction: prefill selects the backend
and the replayed decode list is byte-identical either way.

## Where prefill time goes

Phase attribution from the Level Zero and sycl-tla backends in one session,
instrumented (iterate grade by construction, since the extra waits are the
instrument), one run each, at the 1502.83 t/s state:

| phase | Level Zero ms | share | sycl-tla ms | share |
|---|---:|---:|---:|---:|
| `dequant` | 0.0 | 0.0% | 413.3 | 14.3% |
| `gemm` (sycl-tla) | 0.0 | 0.0% | 1403.2 | 48.7% |
| `linear_l0` | **1668.9** | **61.2%** | 0.0 | 0.0% |
| `attn_QK^T` | 84.6 | 3.1% | 83.0 | 2.9% |
| `attn_softmax` | 60.5 | 2.2% | 60.6 | 2.1% |
| `attn_PV` | 8.3 | 0.3% | 8.3 | 0.3% |
| `gdn_scan` | 328.8 | 12.1% | 330.8 | 11.5% |
| `norm` | 98.4 | 3.6% | 98.7 | 3.4% |
| `silu` | 99.6 | 3.6% | 98.5 | 3.4% |
| **total** | **2728.1** | | **2881.1** | |

**The linears are the whole of the backend win**: `linear_l0`'s 1668.9 ms
replaces `dequant` plus `gemm`'s 1816.5 ms, 8.1% less, and 512 host waits per
run disappear with them. Every other phase moves by under 2 ms.

Three levers have landed since that table and each is in BENCHMARKS.md with its
grade: causal `QK^T` (the attention kernel stops computing the masked half),
SiLU folded into the gate‖up GEMM epilogue, and a split-BF16 delta net scan that
took `gdn_scan` from about 297 ms to about 127, a 2.33x on its own profile row.
Prefill now stands at **1670.72 t/s against vLLM's 1610.04** on matched files.

The split-BF16 scan is the **one approximation in the default path**. It is
gated on tokens and state cosines rather than on `memcmp`, and it is the only
default in the prefill path that is not bit-identical to what it replaced.

## What was measured and rejected

Negative results with a mechanism, kept because they saved more time than the
wins did.

| what | why it was rejected |
|---|---|
| **Fusing the int4 dequant into the GEMM** | Both variants came out bitwise identical to the two-pass path over all 71,303,168 outputs, and both were slower. The dequant pass already runs at about 546 GB/s, roughly 90% of hardware peak, so there is only 0.826 ms in it at this shape; moving the same arithmetic into the mainloop costs 2.759 ms even when issued exactly once per weight. |
| **W4A8 on the mixed 4-bit by 8-bit DPAS** | The mixed builtin's K is 32, the same as int8's, so it runs at 2.000x bf16 and not 4x (doc 01). Measured 1.017x the bf16 two-pass control, 0.965x once the activation quantiser it needs is counted, against a 1.53x bar: the per-group rescale costs 3.0 to 3.4 of the 5.440 ms while the matrix math it rescales costs 1.99. Separately, **2.79% relative L2 error** against the bf16 result, worst-row cosine 0.9976, driven by activation outliers. |
| **Removing barriers from the GDN triangular solve** | The design assumed synchronisation dominated that kernel. Deleting 64 of 128 barriers bought 1.7 ms of a 73.5 ms row, so all 128 are about 4.6% of it and the premise was wrong by roughly 18x. Both barrier-free variants were bit-identical and both were slower. The row is the serialised recurrence itself. |
| **bf16 intermediate buffers** | Halving the bytes changed nothing: `ocloc` exposes no 16-bit block write wider than 8 rows, so the epilogue is bound by store message count rather than by bytes. |
| **oneDNN's fused int4 W4A16 matmul** | Measured 98.28 TFLOP/s on gate‖up at M=2048, below the 115.7 TFLOP/s an in-kernel int4 GEMM would need to beat the two-pass path. It also failed a bitwise identity extraction against our dequant oracle on 39.6% of weights, every mismatch a 1-ULP bf16 neighbour. |

The pattern across all five is the same: **the thing that looked like the cost
was not the cost.** Price the mechanism before building the kernel.

## Specialisations a general engine will not take

Ranked by expected value.

### 1. Quantise `lm_head`

`lm_head.weight` is BF16, 248320 x 5120 = **2.54 GB, 16.4% of `W`**, read in
full on every token at 98.5% of achievable bandwidth. There is nothing to tune
and only bytes to remove: at int4 it is 0.675 GB, at int8 1.27 GB. It costs one
offline conversion pass and no kernel work at all, and it measured **-3.05 ms
per token at record grade**.

**It is not quoted in any comparison, deliberately.** vLLM cannot load a
quantised `lm_head`, so a margin measured against it prices an inability of the
comparison rather than a difference in kernels. The standing numbers in this
project are byte-matched with bf16 `lm_head` on both sides. The lever is real
and it is available; it is simply not evidence about this engine against that
one.

Its accuracy cost is also **unmeasured**. `lm_head` is known to be
quantisation-sensitive and checkpoint authors routinely decline to quantise it.
Adopting it needs a perplexity or `lm_eval` comparison against the unquantised
baseline, and none has been run.

### 2. Pre-swizzle weights at load

Rearranging weights once at load into the exact order the mainloop wants is free
at run time. Every CUDA backend does it; vLLM's XPU MXFP4 path does not. This
engine does it, which is why the decode kernel has no layout branch in the hot
path (doc 02).

### 3. Delete per-step allocations

Static allocation makes per-step allocation zero. `vllm-xpu-kernels`' MoE path
allocates 5 tensors per layer per step. This matters most for model families
this engine does not yet serve.

### 4. Exploit 24 MB of L2

Unusually large for this class of card. A 4096x4096 int4 tile is 8 MB, so
multi-layer weight residency is at least arguable. Unmeasured; measure before
designing around it.

## The honest projection

Decode's remaining gap to the roofline is **1.29x** (33.96 ms against 26.34),
and to vLLM it is **1.05x the wrong way**. Prefill is **1.038x ahead** of vLLM
and at about half of its own DPAS ceiling.

Nothing here supports a claim of several times faster. The MBU argument sets the
bound: vLLM is already at 81.7% of the measured bandwidth on this model, so
there is at most a 1.23x of bandwidth left for *anyone*, and the levers that
reach past it are byte-removal levers (a quantised head) rather than kernel
levers. Anyone promising more is counting the graphs-off number as the baseline,
which is not the fair comparison.

Set expectations there and the project stays honest. Learning the metal,
controlling the dependencies and shipping something that works are what justify
the effort beyond the multiplier.

## Instrument discipline

Three rules the numbers above rest on, each of which cost a wrong conclusion to
learn.

- **Measure the instrument before trusting it.** Day-scale drift on a
  same-checkpoint control is at or under 0.09%. Repeated-run averaging brings
  the attribution floor to 0.051 ms/step with one unexplained outlier family and
  0.011 ms without it. Attribute nothing below the floor. Within-process
  repeatability is *not* the floor: one kernel family reproduced to 0.3% inside
  a process and landed 31% apart across three.
- **Pre-register the prediction, then measure.** Five cost models died to single
  new data points here. A two-point fit always fits; it never explains.
- **One quantity, one value.** Either a document carries one number for a thing
  or it carries an explicit reconciliation sentence. Most review findings on
  this project were that rule being violated by *derived* figures that a literal
  grep missed.
