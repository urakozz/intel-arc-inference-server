# Why decode and prefill want opposite kernels

This is the central idea of the project, and most of the other design choices follow from it. It explains why
generating one token (decode) and reading a prompt (prefill) need different kernels, what vLLM does about it, and how
this engine's prefill path was arrived at.

Read it before writing a kernel. If you read only one document here, make it this one.

## The two regimes

A transformer layer's linear operations are `Y = X · W`, where `X` is `M x K` and `W` is `K x N`. `M` is the number
of tokens being processed at once.

| | Prefill | Decode (batch 1) |
|---|---|---|
| `M` | thousands | 1 |
| Bound by | compute | memory bandwidth |
| Weight reuse | high: `M` rows share each weight tile | none: every weight byte is used once |
| Arithmetic intensity | about `M` FLOP/byte | about 2 FLOP/byte |
| Right instruction | the XMX / DPAS systolic array | scalar or vector FMA |
| Right parallelism | tile over `M x N` | split-K, or nothing fills the device |

XMX is the card's matrix engine, and DPAS is the instruction that drives it. It is a matrix-matrix engine: fed with
`M = 1`, an 8-row systolic tile does one useful row and seven wasted ones. The array is not slow, it is starved. At
`M = 1` you do not want a matrix engine at all; you want to stream the weights at full bandwidth and
multiply-accumulate as they arrive.

That is why "use XMX harder" is the wrong instinct for decode, and why a kernel that ignores XMX can beat one that
uses it.

## What a decode kernel looks like

A decode kernel is a dedicated GEMV (matrix times vector), selected only when the batch dimension is exactly 1. The
design that works:

- It runs SIMD16 (`intel_reqd_sub_group_size(16)`).
- `sub_group_block_read` for the weights: coalesced, full-width streaming from memory.
- int4 is dequantised inline in the accumulation loop, stepping by the quantisation group size, with no separate
  dequant pass and no temporary buffer.
- The activations are held in registers and `sub_group_broadcast` to all lanes; the vector is tiny and fully reused.
- Accumulation is plain scalar FMA.
- The outer loop runs a per-work-group K-range, that is split-K: the reduction dimension K is divided among
  work-groups whose partial sums are added at the end. That is what fills 256 EUs when the output has only a few
  thousand elements.

There is no `matrix_mad` or DPAS intrinsic in such a kernel: it never touches XMX. For decode that is a correct and
deliberate choice. Prefill is the opposite regime and goes through matrix-engine GEMMs.

## What vLLM does

The following was verified in the `vllm-xpu-kernels` source. At `M = 1` its dispatcher selects an 8-row policy:

- a work-group tile of 8 x 64 x 32, with 64 threads per work-group;
- a grid of `sm_count * 8` persistent work-groups;
- no split-K anywhere.

So a vector problem runs on a matrix kernel: seven of the eight tile rows are wasted, and without split-K the output
dimension alone cannot fill the machine.

Two independent observations confirm that this, and not kernel quality, is the limit:

1. MXFP4 (72.65 t/s) and GPTQ-int4 (73.31 t/s) are two completely different kernel paths and land within 1% of
   each other. Both are throttled by the same structural problem.
2. Adding GEMV kernels to that stack recovered 16% of decode at two requests at once.

The measured memory bandwidth utilisation of 50-63% on those models is the direct consequence. The full story is in
[09-vllm-patch-postmortem.md](09-vllm-patch-postmortem.md).

## The gap a plain GEMV leaves: M = 2..8

| `M` | vLLM | What the problem actually is |
|-----|------|---------|
| 1 | 8-row tile, 7/8 wasted | bandwidth-bound |
| 2-8 | tile partly used | bandwidth-bound, weights can be shared |
| large | XMX tiles | compute-bound |

A GEMV gated strictly at `M == 1` drops onto a general path the moment speculative decoding produces two or more
tokens. vLLM's tile becomes fully used only at `M = 8`.

Weights dominate decode traffic and activations are negligible, so the right kernel for `M` in 2..8 is
weight-stationary: stream each weight tile once and multiply it into all `M` activation vectors held in registers,
so the whole weight read is shared by `M` tokens. That compounds with speculative decoding instead of competing with
it, because otherwise speculation costs a full weight read for every row it checks.

The engine has that kernel. Speculative decoding (the model's MTP head, a DFlash draft model, prompt lookup) makes
`M` 2..8 on every verify step, and those steps run a small-batch DPAS GEMM, `gemm_sm`
([12-kernels.md](12-kernels.md)), that reads each weight tile once for all `M` rows. Verifying 8 rows went from the
cost of 3.17 plain decode steps to 1.57 (measured on a busy box).
`M = 1` keeps the GEMV, bit for bit.

## Design rules that follow

1. Two kernel families, chosen by `M`, never one kernel pretending to serve both. This engine has exactly that: a
   decode family compiled with split-K at `M = 1`, and a `pf_*` prefill family compiled without split-K and with
   `M` as a runtime argument. The decode binaries are byte-identical across every prefill change.

2. The decode kernel must split K. Without it the device cannot be filled at realistic output sizes, whatever else
   is right. This is confirmed on this card: at N = 5120, the worst fill case at 320 subgroups, the int4 GEMV runs
   at 259 GB/s (44% of the measured 590) with no split (S = 1) and at 533 GB/s (90%) with S = 16, and `down` at
   K = 17408 goes from 261 to 531 GB/s. Split-K is worth about 2x at that shape.

   It is not universal, and it does not track `N` monotonically. At N = 14336 and N = 16384 the grid already fills
   the device and S = 1 is the best setting, not merely an acceptable one, while `gate‖up` at N = 34816 still needs
   S = 4 despite having 2176 subgroups. Subgroup count is necessary, not sufficient: measure per shape
   ([12-kernels.md](12-kernels.md)).

3. At decode, dequantise inline in the accumulation loop, never in a separate pass or into a temporary buffer; that
   would double the traffic that is the entire budget. At prefill the opposite holds, and it was measured: fusing
   the dequant into the GEMM produced bitwise-identical output and was slower, because the separate dequant pass
   already runs at 93% of the measured 590 GB/s (86% of the 638 GB/s peak, derived), and moving the same arithmetic
   into the GEMM's main loop costs more than the traffic it removes ([05-perf-model.md](05-perf-model.md)).

4. Weight-stationary, activation-broadcast. Weights are the traffic and activations are free, so structure every
   decode kernel around that asymmetry.

## Prefill: the path we took, and the one that did not work

vLLM runs prefill as W4A16: bf16 activations against int4 weights. On Xe2 that means the weights
are upconverted to 16 bits in the GEMM's main loop before the systolic array sees them.

Battlemage also has a native mixed int8 x int4 DPAS that skips the upconversion. Its only condition is that the int4
matrix has no zero points, which symmetric checkpoints satisfy. That looked like the clearest way to beat
vLLM rather than match it.

It was built, and it lost. The mixed instruction's K is 32, the same as int8's, so it runs at 2.000x bf16 and not
4x. The per-group rescale it needs costs more than the instruction rate buys. And int8 activations cost 2.79%
relative L2 error from outliers. The numbers are in [01-hardware.md](01-hardware.md) and
[02-formats.md](02-formats.md).

What did work was less exotic and more structural: stay at W4A16 and own the whole walk. Dequantise int4 into bf16
column slabs, multiply them with our own DPAS GEMM written for this card, and interleave both on one in-order Level
Zero command list, so a prefill chunk makes no SYCL call and never waits on the host. That was worth 6.76% over
calling sycl-tla's GEMM from a SYCL queue (that backend has since been removed), and it is what puts prefill ahead of
vLLM. The bf16 walk is the default. A second, opt-in prefill (`--prefill-backend l0-int8`) computes the linears in
int8 after a Hadamard rotation of the activations and weights, and is faster still
([16-int8-prefill.md](16-int8-prefill.md)). It is not the rejected mixed int8 x int4 instruction above.

### How prefill got here

These are the investigations that decided the prefill design. The numbers are from that time; the current speeds
are in the [README](../README.md#speed). The kernel-level side of the same
work is in [12-kernels.md](12-kernels.md), "How the prefill kernels got here".

**What vLLM runs.** Read from its source: on XPU the int4 linears go through `torch.ops._xpu_C.int4_gemm_w4a16` into
oneDNN's `bf16_int4` matmul (dequant fused into the GEMM main loop, K-grouped scales, scalar zero point 8), and the
delta net goes through a native SYCL chunked kernel, not the Triton `fla_chunk_gated_delta_rule` path (which is CUDA
and ROCm only). Nothing there is a different algorithm, so whether it wins is a question of which individual kernels
beat ours. oneDNN needs the weights K-contiguous per N, a transpose of our layout: about 11.33 GiB of one-time repack
for the 256 matrices.

**Our own GEMM, measured.** `pf_gemm_bf16` uses a 256 x 256 x 32 work-group tile with 32 sub-groups, a sub-group
tile of 32 x 64, 224 of 256 GRF (general register file) registers live, B read through the 2D transform load with no
repack, a cooperative 2-deep prefetch plus a split barrier, and no shared local memory. It reached 156.53 TFLOP/s at
gate‖up M = 2048, bitwise identical to the sycl-tla reference and 2.9% to 17.9% faster than it in the same harness
at every cell (predicted 146, band 135 to 155). Interleaving the slab dequant with it on one in-order Level Zero
list, with zero host waits, saved 89.29 ms per chunk-equivalent at 1024-column slabs (40.91 ms at 2048), because a
1024-column B slab of about 10.5 MB stays resident in L2. The device-side handoff costs 1.667 us against the
22.35 us host wait it replaces. Combined, predicted and measured, that gave 1529 t/s, 77.5% of the vLLM figure then
in use. That decided it: drop SYCL for prefill and own the whole walk. The order in which tiles are visited is
explicit and ours; a grouped order for M = 4096 (where A exceeds L2) is identified but not built.

**oneDNN's int4 matmul: not adopted.** Called directly with vLLM's exact attributes, it measured 98.28 TFLOP/s at
gate‖up M = 2048 (87.8 to 145 across shapes, `down` best). That is below the 115.7 TFLOP/s break-even that an
in-kernel int4 GEMM of ours would need to beat the two-pass dequant plus bf16 GEMM (99.64 TFLOP of chunk work over
0.8613 s, derived). It is the real JIT generator, not a fallback, and the configuration matches vLLM's field by
field. Its numerics also fail our identity gate: 33,198,358 of 83,886,080 weights differ (39.6%), every one a 1-ULP
bf16 neighbour, and strict fp32 fpmath gives the same count. A later build of the exact revision vLLM pins (v3.13,
with 37 commits touching the JIT GEMM generator since the packaged v3.11.4, 5 of them int4-specific) shows the same
failure and pins down its cause: every extracted weight equals the fp32 dequant product truncated to bf16 (zero signs
ignored), at most one bf16 ULP from our round-to-nearest oracle. Slower and not identical, so it was not adopted.

**Intel's chunked delta-net kernel: not a target.** It builds without torch, with two small shims. Once the input
contract of our test harness was repaired, the revision vLLM's build carries passes the CPU-reference bands (state
max / mean relative error 0.040 / 0.0016 at 129 tokens, 0.056 / 0.0016 at 2048), but it returns different bytes
across repetitions of one call, which fails the determinism gate outright. A newer upstream revision is
byte-identical over eight repetitions and four fixtures. It is a numerical and performance reference, not a drop-in;
no model-level gate for vendor arithmetic was ever run.

**The matched comparison.** Same box, same checkpoint, 4096 ids, chunk 2048, three warm requests: this
engine took 2748.2 ms (1490 t/s), of which the recorded Level Zero chunks were 2734.4 ms; vLLM took 2544.0 ms
(1610.04 t/s), with the same first token on every path. The warm gap was 204 ms, not the roughly 657 ms that an older,
uncontrolled 1973 t/s vLLM figure implied (its vLLM and oneDNN revisions were never recorded, so that figure explains
nothing causally).

A trace of vLLM's request showed 2514.0 ms of kernel time (GEMM 2130.6, GDN 177.3, flash attention 63.1, Triton
pointwise 125.3, other 17.9), only 23.6 ms of the span outside a kernel, and 38 host-device copies totalling 14,056
bytes, so no large tensors shuttle through the CPU. The split that steered the next round: our int4 linears were
about 497 ms faster than vLLM's, and everything else about 634 ms slower (1031.3 against 397.1 ms; GDN core 527.6
against 141.8, attention 148.2 against 63.1, SiLU and multiply 98.7 against 51.3). That is how the path with fewer
layers of software still lost, and it sent the work to the GDN scan, attention and pointwise fusion rather than the
GEMM. All three landed: the split-BF16 scan (2.33x on its row), a QK kernel that stopped computing the masked half,
and SiLU folded into the gate‖up epilogue. They took the engine from the 1490 t/s here to 1670.72 t/s, past vLLM; later kernel
work brought it to 1,788 t/s.

The linear path itself, split with one wait per slab for attribution, took 274.5 ms in `slab_dequant` and 1344.3 ms
in `slab_gemm` over 7616 launches each. The GEMM half sits about 5% above its own floor (199 TFLOP at
156.53 TFLOP/s = 1278 ms). The dequant ceiling was then tested by fusing the dequant into the GEMM, and that lost
(design rule 3 above).

**Whole-chunk Level Zero replay: tried and removed.** An experimental mode (`B70_PREFILL_REPLAY=1`) recorded a
chunk's launches in an eight-entry FIFO keyed by position and active row count and replayed them. It was numerically
exact in every case exercised (GDN state, conv history, both KV caches, control and logits, bitwise, including ragged
tails and eviction), but it saved only 13.75 ms against a 204 ms gap, about 0.5%. It was not the lever, the
complexity was real, and it was removed.

**Continuing a prefill is correct.** A prefill that starts at a non-zero `pos` gives the right answer.
Two alarms suggested otherwise. In one, the final hidden state after an `l0-int8` prefill, compared with `M = 1`
decode, had a median cosine of 0.99971 but 0.27 at row 2091, just past the first 2048-row chunk edge. In the other, a
prefill split into two calls at 3000 had a last-row logits cosine of 0.91 against one call. Both were the model
amplifying a rounding-sized perturbation at a handful of positions (987, 1050, 1212, 1565 and 2091 of the long32k
prompt, the same ones in every run).

Splits at a multiple of 64 are bitwise the one-call run (KV, GDN state, live conv slots, logits), because everything
on the walk is row-local or keyed to absolute positions, and only the chunked delta rule's 64-id cut moves with the
call start. Any other split re-rounds only GDN: layer 0's state moves 2e-6 to 3e-6 in cosine, and the full-attention
layers 0 to 3 stay at or above 0.9998 on every row. The CPU oracle chunked at 1000 instead of one forward sits at
cosine 0.7596 at row 2091, against the engine's split-at-1000 0.7555, so it is the model at that position and not
the continuation; at row 987 the one-call engine run is the outlier (0.938).

A real hand-off bug, injected for comparison (`pf_gdn_seed` with `pos = 0`), shows at exactly the chunk-start rows
from the first full-attention layer on and moves GDN layer 0's state 20x to 60x more, while the last-row logits stay
in the clean band. So the discriminating checks of `prefill_split_l0_test` and `prefill_split_l0-int8_test` are the
shallow-layer rows and GDN layer 0's state, with splits at multiples of 64 bitwise, and not the last logits.

## Summary: vLLM and this project

| | vLLM | This project |
|---|---|---|
| Decode `M=1` | 8-row tile, no split-K | GEMV with per-shape split-K |
| Decode `M=2..8` | tile partly used | small-batch DPAS GEMM `gemm_sm`, weights read once for all rows |
| Prefill | W4A16 XMX tiles | W4A16, own DPAS GEMM, zero host waits |

Prefill is ahead of vLLM. Plain decode at `M=1` was about 5% behind it in the last matched comparison
([05-perf-model.md](05-perf-model.md)). The kernel for `M` in 2..8 that motivated the original idea exists here.
