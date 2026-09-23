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

**This is an opportunity nobody has taken, including us.** Weights dominate
decode traffic and activations are negligible, so a **weight-stationary kernel
parameterised on `M ∈ [1,8]`** would amortise the entire weight read across `M`
tokens: stream each weight tile once, multiply it into `M` activation vectors
held in registers. It would compound with speculative decoding rather than
competing with it, because speculation's whole cost today is paying full weight
traffic on every draft step.

This engine does not build it. There is no speculative decoding here, so `M` is
always 1 on the decode path, and the kernel family is compiled for that.

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
[probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md) carry the numbers.

What did work was less exotic and more structural: **stay at W4A16 and own the
whole walk.** Dequantise int4 into bf16 column slabs, multiply with our own DPAS
GEMM written for this card, and interleave both on one in-order Level Zero list
so a prefill chunk makes no SYCL call and never waits on the host. That is worth
6.76% over calling sycl-tla's GEMM from a SYCL queue, and it is what puts
prefill ahead of vLLM.

## Summary: where each engine wins

| | vLLM | OpenVINO | This project |
|---|---|---|---|
| Decode `M=1` | 8-row tile, no split-K | dedicated GEMV | GEMV with per-shape split-K |
| Decode `M=2..8` | tile partly used | falls off to oneDNN | **not built - the standing opportunity** |
| Prefill | W4A16 XMX tiles | W4A16 via oneDNN | W4A16, own DPAS GEMM, zero host waits |

Prefill is ahead, decode is about 5% behind, and the `M ∈ [1,8]` kernel that
motivated the original thesis remains unwritten on every engine on this
silicon.

## Still open

Why OpenVINO's prefill is slower than vLLM's is **not established**. oneDNN does
use XMX and does handle int4, so "it does not use XMX" is not the explanation:
both engines upconvert int4 in a W4A16 mainloop. The remaining plausible causes,
untested, are that oneDNN's generic path is less tuned for Battlemage than
`sycl-tla`'s Battlemage-specific tiles, and that layout reorders cost an extra
pass. Profile both before assuming a design.
