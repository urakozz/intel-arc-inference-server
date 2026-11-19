# Performance model

This document explains where the time goes in Qwen3.8-27B's decode (writing the answer) and prefill (reading the
prompt), what the hard ceilings are, which levers have been pulled, and which were tried and rejected. It is for a
reader who wants to reason about the engine's speed rather than just read its numbers.

Figures are measured unless the text says they are derived (arithmetic on measured inputs, with the inputs shown) or
estimated. The profile tables were taken on older driver stacks: the decode profile before any lever was pulled, the
prefill phases on an earlier compute runtime and IGC. The ceilings are properties of the card and still stand. The
current speed numbers are in the [README](../README.md#speed).

This document is about plain decoding of one request on one card. Speculative decoding (MTP, DFlash, prompt lookup)
changes the decode arithmetic: one step checks M rows for the weight traffic of about 1.04 to 1.6 plain steps. So do
several requests at once, and tensor parallel (every layer split over both cards). Those are the levers that move
the served speed far past the one-card ceiling below.

## The two ceilings

Decode (generating one token at a time) and prefill (reading the whole prompt) are bound by different things,
which is the whole reason [08-decode-vs-prefill.md](08-decode-vs-prefill.md) exists.

### Decode: bandwidth

With one request, every weight is read once per token, so

```
tokens/sec_ceiling = read_bandwidth / bytes_read_per_token = 590 GB/s / W
```

590 GB/s is measured through a plain Level Zero launch ([01-hardware.md](01-hardware.md)), the launch path this
engine actually submits on. `W` is measured by the loader from what it makes resident: 15,539,980,288 B = 15.540
GB per token ([13-loader.md](13-loader.md)). That is [03-models.md](03-models.md)'s 15.519 GB of header
arithmetic plus 0.016 GB of tiling pad and 0.005 GB of norms widened to fp32.

```
ceiling = 590 / 15.540 = 37.97 t/s   (26.34 ms/token)   [derived]
vLLM    =               31.01 t/s -> 482 GB/s -> 81.7% MBU
this    =               29.45 t/s -> 458 GB/s -> 77.6% MBU   [derived]
```

MBU (memory bandwidth utilisation) is the achieved read rate as a share of the 590 GB/s.

Use 590 GB/s and 15.540 GB, and use them on both sides of any comparison. [01-hardware.md](01-hardware.md) also
gives the card's peak, 638 GB/s, and [03-models.md](03-models.md)'s header arithmetic gives 15.519 GB. 638 is what
the memory could carry and 590 is what a kernel launch reaches; the two weight totals differ only by pad and norm
widening. Mixing denominators inside one comparison moves a figure by up to 8 %.

### Prefill: compute

The forward pass is 48.97 GFLOP per token (derived from the [03-models.md](03-models.md) shapes), so 4096 tokens
in 2.4516 s is 81.8 TFLOP/s sustained (derived). The DPAS pipe, the matrix engine, runs bf16 at 183.45 TFLOP/s
([01-hardware.md](01-hardware.md)), and our own GEMM reaches 161.7 TFLOP/s on the largest linear, 88.1 % of it.

The gap between 81.8 and 161.7 is not kernel quality: it is everything that is not the GEMM. The linear path alone
is 199 TFLOP of work per request, and the dequant pass that feeds the GEMM (unpacking int4 weights to bf16) is a
second pass over the same weights. That is the shape of the remaining prefill headroom, and it is why three of the
rejected optimisations below attacked the dequant-plus-GEMM pair.

## Where decode time goes

The one full per-kernel profile of a decode step used Level Zero kernel timestamps on the replayed command list,
with every launch timed in the step it actually runs in. It was taken on a 42.141 ms step, before any of the
levers below were pulled, and it is kept whole because it is what those levers were chosen from.

| part | launches | ms/token | share |
|---|---|---:|---:|
| GEMV: the int4 mixers and MLPs and the bf16 `lm_head` | 257 | 29.005 | 68.8% |
| `attn_decode` | 16 | 5.782 | 13.7% |
| `prep` family | 241 | 3.573 | 8.5% |
| the `a‖b` GEMV | 48 | 2.335 | 5.5% |
| `gdn_step` | 48 | 0.733 | 1.7% |
| `attn_prep` + `attn_reduce` | 32 | 0.127 | 0.3% |
| dispatch gap (fence wall minus the sum of kernel durations) | - | 0.473 | 1.1% |
| host, outside the fence entirely | - | 0.097 | 0.23% |
| **total** | **645** | **42.141** | 100% |

The table is a partition: every launch and the host appear once, and the rows sum to the measured step, which is
why they carry three decimals where the prose rounds to two. The launch count is 645 because the profile predates
the `prep` split that took it to 774.

The table settled four things.

1. **Host overhead is not the problem in this engine.** 99.77 % of the step is inside `execute` plus
   `fence.wait()`; the host spends 97 us per token writing four bytes and submitting. There is no host work left to
   delete. The host-overhead thesis was always about vLLM's Python and Triton launch path
   ([09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md)), and replaying a recorded list did exactly what
   [08-decode-vs-prefill.md](08-decode-vs-prefill.md) said it would. That is precisely why the remaining gap
   cannot be excused as overhead: it is kernel time.
2. **The linears are not the problem.** They run at 89-97 % of their per-shape bandwidth ceilings, and `lm_head`
   alone is 4.376 ms at 581 GB/s, 98.5 % of the measured 590. There is nothing to tune there, only bytes to remove.
3. **`gdn_step` is cleared.** 48 launches moving 396 MB is 0.67 ms at the roofline, and it measures 0.733 ms, 1.09x
   its own traffic floor, at 92 % of device bandwidth. The kernel the design worried about most is 1.7 % of the
   step and the best-occupied thing in the engine.
4. **The occupancy problem is real, and its unit is the subgroup** (the lanes that execute one instruction
   together). The expensive kernels were the ones given one or two work-groups. Giving `a‖b` 4x the work-groups at
   an unchanged subgroup count bought nothing; splitting its K 16 ways, from 8 subgroups to 128, took it from
   2.335 to 0.256 ms a token.

### The levers that followed

| lever | before | after | bench delta |
|---|---:|---:|---:|
| `a‖b` GEMV K-split, 8 → 128 subgroups | 2.335 | 0.256 | -1.875 ms/token |
| `prep_res_norm` split into a two-stage reduction, 16 → 320 subgroups | 2.893 | 0.484 | -2.220 ms/token |
| attention block width 256 → 64 | 5.782 | 3.585 | -1.726 ms/token |

Together they took the step from 42.141 to 36.32 ms a token, +16.1 % throughput, 62.5 % to 72.5 % MBU, with the
golden gate (the check of the engine's output against a CPU reference, [14-golden-gate.md](14-golden-gate.md)) green at every
step. The rest of the list was then ranked and stopped: `gdn_step` at 92 % of device bandwidth has 0.06 ms left in
the whole kernel, and the dispatch gap is 0.473 ms, so neither was worth a golden-gate run.

Two things are worth knowing about that sequence. First, the launch count went up on purpose: splitting
`prep_res_norm` added 129 launches to buy 2.4 ms, at a derived 0.733 us each. Launch count is not the lever it was
once feared to be, in either direction.

Second, attention did not meet its prediction. It was priced at -2.045 ms and measured -1.73, 0.32 ms apart, and
the difference is recorded as unexplained rather than explained away. What is left inside that kernel is a
per-launch term of about 224 us that four successive cost models failed to explain. The largest named term of the
retiled launch is the KV load path at 55.4 % (39.2 % the 32-byte load messages themselves, 16.2 % cache service),
which is shaped like throughput rather than like a chain of latencies.

The step later moved from 36.32 to 33.96 ms a token (29.45 t/s) through loader and per-shape retuning. On the
byte-matched bf16 head that plain decode trails vLLM by about 5 %. No prefill work moves it, by construction: the
prefill backend is selectable, and the replayed decode list is byte-identical either way.

The serving default reads an int8 `lm_head` (below). With it, plain decode on one card measured 33.49 t/s at depth
4096, and tensor parallel 53.95 t/s, 1.61x. Those are not byte-matched with vLLM's 31.01.

## Where prefill time goes

The Level Zero prefill backend, the engine's own DPAS GEMM and slab dequant on one in-order command list, replaced
an earlier backend built on sycl-tla because the linears were the whole of the difference. One instrumented session
at 1502.83 t/s had `linear_l0` at 61.2 % of the 2728 ms run, replacing sycl-tla's `dequant` plus `gemm` (1816.5
ms) with 1668.9 ms, and 512 host waits a run disappeared with it. Every other phase moved by under 2 ms.

Several changes followed: the causal `QK^T` (the attention kernel stops computing the masked half), SiLU folded
into the gate‖up GEMM epilogue, a split-BF16 delta-net scan (`gdn_scan` from about 297 ms to about 127), flash
attention, and 4096-row chunks. vLLM measured 1,610 t/s on a 4,096-token prompt with the same files; the engine reads
it at 1,788 t/s on one card and 2,931 t/s under tensor parallel.

The breakdown below is from a probe with one card, real weights, a bf16 KV cache and the two backends timed in
interleaved pairs. A 2048-row chunk takes 1189.7 ms on `l0` (1,721 t/s) and 913.7 ms on `l0-int8` (2,241 t/s).
The int4 multiply is the largest share on both. On `l0` it is 61 % of the chunk, running at 77.6 % of the measured
bf16 peak, so a new bf16 kernel is not worth building. On `l0-int8` it is 44 %, running at 67.6 % of the int8 peak:
a possible lever, not scheduled. The delta net is about 20 % of a chunk.

The split-BF16 scan is the one approximation in the default path. It is gated on tokens and state cosines rather
than on `memcmp`, and it is the only default in the prefill path that is not bit-identical to what it replaced.

## What was measured and rejected

These are negative results with a known mechanism. They are kept because they saved more time than the wins did.

| what | why it was rejected |
|---|---|
| Fusing the int4 dequant into the GEMM | Both variants came out bitwise identical to the two-pass path over all 71,303,168 outputs, and both were slower. The dequant pass already runs at about 546 GB/s (93 % of the measured 590; 86 % of the 638 GB/s peak, derived), so there is only 0.826 ms in it at this shape; moving the same arithmetic into the GEMM's inner loop costs 2.759 ms even when issued exactly once per weight. |
| W4A8 on the mixed 4-bit by 8-bit DPAS | The mixed builtin's K is 32, the same as int8's, so it runs at 2.000x bf16 and not 4x ([01-hardware.md](01-hardware.md)). Measured 1.017x the bf16 two-pass control, 0.965x once the activation quantiser it needs is counted, against a 1.53x bar: the per-group rescale costs 3.0 to 3.4 of the 5.440 ms while the matrix math it rescales costs 1.99. Separately, 2.79 % relative L2 error against the bf16 result, worst-row cosine 0.9976, driven by activation outliers. |
| Removing barriers from the GDN triangular solve | The design assumed synchronisation dominated that kernel. Deleting 64 of 128 barriers bought 1.7 ms of a 73.5 ms row, so all 128 are about 4.6 % of it and the premise was wrong by roughly 18x. Both barrier-free variants were bit-identical and both were slower. The row's cost is the serialised recurrence itself. |
| bf16 intermediate buffers | Halving the bytes changed nothing: `ocloc` exposes no 16-bit block write wider than 8 rows, so the epilogue is bound by the number of store messages rather than by bytes. |
| W4A4: int4 weights times int4 activations | The DPAS rate is real (4.000x bf16, 733.80 TIOP/s), but even after the Hadamard rotation int4 activations roughly double the output error: held-out logits 23.3 % relative L2 against the int8 path's 12.5 %, and 3.4 argmax positions in 100 lost against 2 allowed. Only `o_proj` alone passes, and it is 2.1 % of the prefill GEMM work. |
| oneDNN's fused int4 W4A16 matmul | Measured 98.28 TFLOP/s on gate‖up at M=2048, below the 115.7 TFLOP/s an in-kernel int4 GEMM would need to beat the two-pass path. It also failed a bitwise identity extraction against our dequant oracle on 39.6 % of weights, every mismatch a 1-ULP bf16 neighbour. |

The pattern across all six is the same: the thing that looked like the cost was not the cost. Price the mechanism
before building the kernel.

## Specialisations a general engine will not take

Ranked by expected value.

### 1. Quantise `lm_head`

`lm_head.weight` is BF16, 248320 x 5120 = 2.54 GB, 16.4 % of `W`, read in full on every token at 98.5 % of
achievable bandwidth. There is nothing to tune, only bytes to remove. The engine's serving default is an int8 head,
with one fp32 scale per row, quantised at load. It halves the head's bytes and measured 31.21 t/s against 29.38
for the bf16 head in interleaved pairs, -2.00 ms a token. It passed its accuracy checks: logits cosine 0.9999147 at
worst, one near-tie argmax flip in 2790 rows, and the golden and prefill gates exact on the determined rows. An
int4 head would save more, -3.05 ms a token at record grade, but its accuracy
has never been measured and `lm_head` is known to be sensitive to quantisation, so it is not adopted.

The comparison with vLLM stays byte-matched on purpose. vLLM cannot load a quantised `lm_head`, so a margin
measured against it with one would price a limitation of the comparison rather than a difference in kernels. The
byte-matched rows use the bf16 head on both sides.

### 2. Pre-swizzle weights at load

Rearranging weights once at load into the exact order the GEMM's inner loop wants is free at run time. Every CUDA
backend does it; vLLM's XPU MXFP4 path does not. This engine does, which is why the decode kernel has no layout
branch in the hot path ([02-formats.md](02-formats.md)).

### 3. Delete per-step allocations

Allocating every buffer once, up front, makes per-step allocation zero. `vllm-xpu-kernels`' MoE path allocates 5
tensors per layer per step. This matters most for the MoE model families.

### 4. Exploit 24 MB of L2

The L2 is unusually large for this class of card. A 4096x4096 int4 tile is 8 MB, so keeping several layers'
weights in cache is at least arguable. It is unmeasured; measure it before designing around it.

## What to expect

On the byte-matched bf16 head, plain decode is 1.29x off the roofline (33.96 ms against 26.34) and 1.05x behind
vLLM. Prefill is ahead of vLLM (1,788 t/s on the bf16 path, 2,337 on the int8 path, against 1,610) at about half of
its own DPAS ceiling on the bf16 path. The levers that moved decode past that were byte removal and parallelism (the
int8 head, speculative decoding, several requests, both cards), not single-card kernel speed.

Plain one-card decode cannot get several times faster. The MBU argument sets the bound: vLLM is already at 81.7 % of
the measured bandwidth on this model, so at most 1.23x of bandwidth is left for anyone. The levers that reach past it
remove bytes (a quantised head) or check several tokens per weight read (speculative decoding, batching). A claim of
a large one-card multiplier usually takes the number with graph replay turned off as its baseline.

## Instrument discipline

The numbers above rest on three rules, each of which cost a wrong conclusion to learn.

**Measure the instrument before trusting it.** Drift over a day on a same-checkpoint control is at or under
0.09 %. Averaging repeated runs brings the attribution floor to 0.051 ms a step with one unexplained outlier
family, and 0.011 ms without it; attribute nothing below the floor. Repeatability within one process is not the
floor: one kernel family reproduced to 0.3 % inside a process and landed 31 % apart across three.

**Write the prediction down, then measure.** Five cost models died to single new data points here. A two-point fit
always fits; it never explains.

**One quantity, one value.** A document either carries one number for a thing or carries an explicit sentence
reconciling two. Most review findings on this project were this rule broken by derived figures that a literal
search missed.
