# Why decode and prefill want opposite kernels

This is the central insight of the project. Everything else follows from it.

Read this before writing a kernel. If you only understand one document here,
make it this one.

## The two regimes

A transformer layer's linear ops are `Y = X · W`, where `X` is `M x K` and `W`
is `K x N`. `M` is the number of tokens being processed at once.

| | Prefill | Decode (batch 1) |
|---|---|---|
| `M` | thousands | **1** |
| Bound by | compute | **memory bandwidth** |
| Weight reuse | high - `M` rows share each weight tile | **zero** - every weight byte used once |
| Arithmetic intensity | about `M` FLOP/byte | about 2 FLOP/byte |
| Right instruction | XMX / DPAS systolic array | scalar or vector FMA |
| Right parallelism | tile over `M x N` | **split-K**, or nothing fills the device |

**XMX is a matrix-matrix engine.** Feeding it `M = 1` means an 8-row systolic
tile does one useful row and seven wasted ones. The array is not slow, it is
starved. At `M = 1` you do not want a matrix engine at all; you want to stream
weights at full bandwidth and multiply-accumulate as they arrive.

This is why "use XMX harder" is the wrong instinct for decode, and why a kernel
that ignores XMX can beat one that uses it.

## What OpenVINO does - verified in source

**Decode.** A dedicated GEMV kernel,
`src/plugins/intel_gpu/src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl`,
selected only when the batch dimension is exactly 1.

How it works:

- `intel_reqd_sub_group_size(16)` - SIMD16, and it `#error`s on anything else.
- `sub_group_block_read` for weights - coalesced, full-width DRAM streaming.
- int4 dequantised **inline in the accumulation loop**, stepping by
  `DECOMPRESSION_GROUP_SIZE`. No separate dequant pass, no temporary buffer.
- Activations held in registers and `sub_group_broadcast`-ed to all lanes; the
  vector is tiny and fully reused.
- Accumulation is **plain scalar FMA on `half`**.
- The outer loop runs a per-work-group K-range, i.e. **split-K**. That is what
  fills 256 EUs when the output is only a few thousand elements.

**There is not a single `matrix_mad` or DPAS intrinsic in any OpenVINO `.cl`
kernel.** Its own OpenCL kernels never touch XMX. For decode that is a correct
and deliberate choice.

**Prefill.** Delegated to oneDNN, which does support compressed weights and does
use XMX.

## What vLLM does - verified from the kernel source

At `M = 1` the `vllm-xpu-kernels` dispatcher selects an 8-row policy:

- work-group tile **8 x 64 x 32**, 64 threads per work-group;
- grid is `sm_count * 8` persistent work-groups;
- **no split-K anywhere**.

So a vector problem runs on a matrix kernel: **seven of eight tile rows are
wasted**, and with no split-K the output dimension alone cannot fill the
machine.

Two independent confirmations that this, and not kernel quality, is the limit:

1. MXFP4 (72.65 t/s) and GPTQ-int4 (73.31 t/s) are two completely different
   kernel paths and land **within 1%**. Both are throttled by the same
   structural problem.
2. Adding GEMV kernels to that stack recovered **+16%** decode at concurrency 2.

Measured MBU of 50-63% on those models is the direct consequence.

## The gap neither engine serves: M = 2..8

| `M` | OpenVINO | vLLM | Reality |
|-----|----------|------|---------|
| 1 | dedicated GEMV | 8-row tile, 7/8 wasted | bandwidth-bound |
| **2-8** | **falls off to oneDNN** | tile partly used | **bandwidth-bound, weights amortisable** |
| large | oneDNN | XMX tiles | compute-bound |

OpenVINO's GEMV is hard-gated at `M == 1`, so the moment speculative decoding
produces two or more tokens it drops onto a general path. vLLM's tile only
becomes fully utilised *at* `M = 8`.

**This was an opportunity nobody had taken; since 2026-11-01 this engine has (spec 28b, `gemm_sm`,
doc 12).** Weights dominate
decode traffic and activations are negligible, so a **weight-stationary kernel
parameterised on `M ∈ [1,8]`** would amortise the entire weight read across `M`
tokens: stream each weight tile once, multiply it into `M` activation vectors
held in registers. It would compound with speculative decoding rather than
competing with it, because speculation's whole cost today is paying full weight
traffic on every draft step.

The engine now has speculative decoding (the model's MTP head, a DFlash draft model, prompt lookup),
so `M` is 2..8 on every verify step. Those lists run a small-batch DPAS GEMM that streams each weight
tile once and multiplies it into all `M` rows (`gemm_sm`, spec 28b): verify(8) went from 3.17 plain
steps to 1.57 (busy box, to be re-checked quiet; BENCHMARKS). `M = 1` keeps the GEMV, bitwise as
before. Weight traffic was the reason, and the numbers agree with it: the speculation's cost is no
longer a full weight read per guessed row.

## Design rules that follow

1. **Two kernel families, dispatched on `M`.** Never one kernel pretending to
   serve both. This engine has exactly that: a decode family compiled with
   split-K at `M = 1`, and a `pf_*` prefill family compiled without split-K with
   `M` as a runtime argument. The decode binaries are byte-identical across
   every prefill change.
2. **The decode kernel must split K.** Without it the device cannot be filled at
   realistic output sizes, whatever else is right. **Confirmed on this
   silicon**: at N = 5120, the worst fill case at 320 subgroups, the int4 GEMV
   runs at **259 GB/s (43% of 600) at S = 1 and 533 GB/s (89%) at S = 16**, and
   261 to 531 GB/s for `down` at K = 17408. Split-K is worth about **2x** at
   that shape.

   It is not universal and it does not track `N` monotonically: at N = 14336 and
   N = 16384 the grid already fills the device and S = 1 is the *best* setting
   rather than merely an acceptable one, while `gate‖up` at N = 34816 still
   needs S = 4 despite having 2176 subgroups. **Subgroup count is necessary, not
   sufficient - measure per shape** (doc 12).
3. **Dequantise inline in the accumulation loop, at decode.** Never a separate
   pass, never a temporary buffer; that would double the traffic that is the
   entire budget. **At prefill the opposite is true**, and it was measured:
   fusing the dequant into the GEMM produced bitwise-identical output and was
   slower, because the separate dequant pass already runs at about 90% of
   hardware peak and moving the same arithmetic into the mainloop costs more
   than the traffic it removes (doc 05).
4. **Weight-stationary, activation-broadcast.** Weights are the traffic;
   activations are free. Structure every decode kernel around that asymmetry.

## Prefill: the path we took, and the one that did not work

Both vLLM and OpenVINO run prefill as **W4A16**, bf16 activations against int4
weights. On Xe2 that means the weights are **upconverted to 16 bits in the
mainloop** before the systolic array sees them.

Battlemage does have a **native mixed int8 x int4 DPAS** that skips the
upconversion, gated only on the int4 matrix having no zero points, which
symmetric checkpoints satisfy. That looked like the clearest candidate for
beating both engines rather than matching one.

**It was built and it lost.** The mixed builtin's K is 32, the same as int8's,
so it runs at 2.000x bf16 and not 4x; the per-group rescale it needs costs more
than the instruction rate buys; and int8 activations cost 2.79% relative L2
error from outliers. Docs [01](01-hardware.md), [02](02-formats.md) and
[probe-w4a8-2026-09-23.md](probes/probe-w4a8-2026-09-23.md) carry the numbers.

What did work was less exotic and more structural: **stay at W4A16 and own the
whole walk.** Dequantise int4 into bf16 column slabs, multiply with our own DPAS
GEMM written for this card, and interleave both on one in-order Level Zero list
so a prefill chunk makes no SYCL call and never waits on the host. That was worth
6.76% over calling sycl-tla's GEMM from a SYCL queue (that backend was removed on 2026-10-27), and it
is what puts prefill ahead of vLLM. The bf16 walk is the default. A second, opt-in prefill
(`--prefill-backend l0-int8`) computes the linears in int8 after a Hadamard rotation of the
activations and weights and is faster still ([17-int8-prefill.md](17-int8-prefill.md)); it is not the
rejected W4A8 mixed instruction above.

### How prefill got here

The investigations of September 2026 (14 to 28) that decided the prefill design. Their
standalone write-ups were folded into this section and into
[12-kernels.md](12-kernels.md) ("How the prefill kernels got here") once the work closed.

**What vLLM runs.** Read from its source: on XPU the int4 linears go through
`torch.ops._xpu_C.int4_gemm_w4a16` into oneDNN's `bf16_int4` matmul (dequant fused in the
GEMM mainloop, K-grouped scales, scalar zero point 8), and the delta net goes through a
native SYCL chunked kernel, not the Triton `fla_chunk_gated_delta_rule` path (that is
CUDA / ROCm only). Nothing there is a different algorithm, so whether it wins is a question
of which individual kernels beat ours. oneDNN needs the weights K-contiguous per N, a
transpose of our layout (about 11.33 GiB of one-time repack for the 256 matrices).

**Our own GEMM, measured.** `pf_gemm_bf16` (256 x 256 x 32 work-group tile, 32 sub-groups,
sub-group tile 32 x 64, 224 of 256 GRF live, B read through the 2D transform load with no
repack, cooperative 2-deep prefetch plus a split barrier, no SLM) reached **156.53 TFLOP/s**
at gate||up M = 2048, bitwise identical to the sycl-tla reference and 2.9% to 17.9% faster
than it in the same harness at every cell (predicted 146, band 135 to 155). Interleaving the
slab dequant with it on one in-order Level Zero list, zero host waits, saved **89.29 ms** per
chunk-equivalent at 1024-column slabs (40.91 ms at 2048: a 1024-column B slab of about
10.5 MB stays L2-resident), with a device-side handoff of 1.667 us against the 22.35 us host
wait it replaces. Combined, predicted and measured, 1529 t/s: 77.5% of the vLLM figure then
in use. That decided it: drop SYCL for prefill, own the whole walk. The raster is explicit
and owned; a grouped raster for M = 4096 (A exceeds L2) is named, not built.

**oneDNN's int4 matmul: not adopted.** Called directly with vLLM's exact attributes it
measured **98.28 TFLOP/s** at gate||up M = 2048 (87.8 to 145 across shapes, `down` best),
below the **115.7 TFLOP/s break-even** an in-kernel int4 GEMM of ours would need against the
two-pass dequant plus bf16 GEMM (99.64 TFLOP of chunk work over 0.8613 s, derived). It is the
real JIT generator, not a fallback, and the configuration matches vLLM's field by field. Its
numerics also fail our identity gate: 33,198,358 of 83,886,080 weights differ (39.6%), every
one a 1-ULP bf16 neighbour; strict fp32 fpmath gives the identical count. The later build of
the exact revision vLLM pins (v3.13; 37 commits touching the JIT GEMM generator since the
packaged v3.11.4, 5 of them int4-specific) shows the same failure and pins it down: every
extracted weight equals the fp32 dequant product **truncated** to bf16 (zero signs ignored),
at most one bf16 ULP from our round-to-nearest oracle. Slower and not identical: do not
adopt.

**Intel's chunked delta net kernel: not a target.** It builds torch-free (two probe-side
shims: a minimal ATen stand-in, and `int` overloads for a barrier-scope constant that fails
to compile as shipped). The first probe fed raw Q and K (the file's comments describe an
input precondition, not code: it normalizes nothing), and after fixing that its state still
sat at a mean relative error near 1.0 against the CPU reference, so its 0.795 ms per chunk
(against our 303 ms then, 382x) was never a price. After the probe's input and reset
contract were repaired, the revision vLLM's build carries passes the CPU-reference bands
(state max / mean relative error 0.040 / 0.0016 at 129 tokens, 0.056 / 0.0016 at 2048) but
returns **differing bytes across repetitions** of one call, which fails the determinism gate
outright. A newer upstream revision (local-space barriers replaced by
`sycl::group_barrier`, split-barrier scopes typed) is byte-identical over eight repetitions
and four fixtures. It is a numerical and performance reference, not a drop-in: no
model-level gate for vendor arithmetic was ever run.

**The matched comparison (2026-09-20).** Same box, checkpoint, 4096 ids, chunk 2048, three
warm requests: this engine 2748.2 ms (1490 t/s), with recorded L0 chunks 2734.4 ms, vLLM
**2544.0 ms (1610.04 t/s)**, the same first token on every path. The warm gap was 204 ms, not
the roughly 657 ms an older uncontrolled 1973 t/s vLLM figure implied (its vLLM and oneDNN
revisions were never recorded; that figure explains nothing causally). A trace of vLLM's
request: 2514.0 ms of kernel time (GEMM 2130.6, GDN 177.3, flash attention 63.1, Triton
pointwise 125.3, other 17.9), only 23.6 ms of the span outside a kernel, 38 host-device
copies totalling 14,056 bytes, so no large tensors shuttle through the CPU. The partition
that steered the next round: **our int4 linears were about 497 ms faster than vLLM's, and
everything else about 634 ms slower** (1031.3 against 397.1 ms; GDN core 527.6 against
141.8, attention 148.2 against 63.1, SiLU and multiply 98.7 against 51.3). That is how the
path with fewer layers of software still lost, and it sent the work to the GDN scan,
attention and pointwise fusion rather than the GEMM. All three landed (the split-BF16 scan,
2.33x on its row; the QK kernel stopped computing the masked half; SiLU folded into the
gate||up epilogue) and are the difference between the 1490 t/s here and 1670.72 at the time.
The linear path itself, split one wait per slab for attribution: `slab_dequant` 274.5 ms and
`slab_gemm` 1344.3 ms over 7616 launches each; the GEMM half sits about 5% above its own
floor (199 TFLOP at 156.53 TFLOP/s = 1278 ms), and the dequant ceiling was then tested and
lost (see the fused-dequant probe in [probes/](probes/probe-fused-dequant-2026-09-22.md)).

**Whole-chunk Level Zero replay: tried, removed.** `B70_PREFILL_REPLAY=1` recorded the
chunk's launches (an eight-entry FIFO keyed by position and active row count) and replayed
them (the zero-config cleanup, 2026-11-06, removed it). It was numerically exact on every exercised case (GDN state, conv history, both KV
caches, control and logits, bitwise, including ragged tails and eviction) and saved
**13.75 ms** against a 204 ms gap, about 0.5%: not the lever, and the complexity is real.

**Continuation is not a bug (2026-09-28).** A prefill that starts at a non-zero `pos` is
correct. Two alarms (the main hidden after l0-int8 prefill against M = 1 decode: median
cosine 0.99971 but 0.27 at row 2091, just past the first 2048 chunk edge; and a two-call
prefill split at 3000 against one call: last-row logits cosine 0.91) were the model
amplifying a rounding-sized perturbation at a handful of positions (987, 1050, 1212, 1565,
2091 of the long32k prompt, the same ones in every run). Splits at a multiple of 64 are
**bitwise** the one-call run (KV, GDN state, live conv slots, logits), because everything on
the walk is row-local or keyed to absolute positions and only the chunked delta rule's
64-id cut moves with the call start; any other split re-rounds GDN only (layer 0's state
moves 2e-6 to 3e-6 in cosine; the full-attention layers 0 to 3 stay at or above 0.9998
on every row). The CPU oracle chunked at 1000 instead of one forward sits at cosine **0.7596**
at row 2091 against the engine's split-1000 0.7555, so it is the model at that position, not
the continuation; at row 987 the one-call engine run is the outlier (0.938). A real hand-off
bug (the injected `pf_gdn_seed` with `pos = 0`) shows at exactly the chunk-start rows from
the first full-attention layer on and moves GDN L0's state 20x to 60x more, while the
last-row logits stay in the clean band. So the discriminating bars of
`prefill_split_l0_test` and `prefill_split_l0-int8_test` are the shallow-layer rows and GDN
L0's state, with splits at multiples of 64 bitwise, and not the last logits.

## Summary: where each engine wins

| | vLLM | OpenVINO | This project |
|---|---|---|---|
| Decode `M=1` | 8-row tile, no split-K | dedicated GEMV | GEMV with per-shape split-K |
| Decode `M=2..8` | tile partly used | falls off to oneDNN | small-batch DPAS GEMM `gemm_sm`, weights read once for all rows (spec 28b) |
| Prefill | W4A16 XMX tiles | W4A16 via oneDNN | W4A16, own DPAS GEMM, zero host waits |

Prefill is ahead, decode `M=1` is about 5% behind, and the `M ∈ [1,8]` kernel that
motivated the original thesis now exists here, for `M >= 2`.

## Still open

Why OpenVINO's prefill is slower than vLLM's is **not established**. oneDNN does
use XMX and does handle int4, so "it does not use XMX" is not the explanation:
both engines upconvert int4 in a W4A16 mainloop. The remaining plausible causes,
untested, are that oneDNN's generic path is less tuned for Battlemage than
`sycl-tla`'s Battlemage-specific tiles, and that layout reorders cost an extra
pass. Profile both before assuming a design.
