# Why decode and prefill want opposite kernels

This is the central insight of the project. Everything else follows from it.

Read this before writing a kernel. If you only understand one document here,
make it this one.

## The two regimes

A transformer layer's linear ops are `Y = X · W`, where `X` is `M × K` and `W` is
`K × N`. `M` is the number of tokens being processed at once.

| | Prefill | Decode (batch 1) |
|---|---|---|
| `M` | thousands | **1** |
| Bound by | compute | **memory bandwidth** |
| Weight reuse | high - `M` rows share each weight tile | **zero** - every weight byte used once |
| Arithmetic intensity | ~`M` FLOP/byte | ~2 FLOP/byte |
| Right instruction | XMX / DPAS systolic array | scalar or vector FMA |
| Right parallelism | tile over `M × N` | **split-K**, or nothing fills the device |

**XMX is a matrix-matrix engine.** Feeding it `M = 1` means an 8-row systolic
tile does one useful row and seven wasted ones. The array is not slow - it is
starved. At `M = 1` you do not want a matrix engine at all; you want to stream
weights at full bandwidth and multiply-accumulate as they arrive.

This is why "use XMX harder" is the wrong instinct for decode, and why a kernel
that ignores XMX can beat one that uses it.

## What OpenVINO does - verified in source

`~/PycharmProjects/openvino/src/plugins/intel_gpu`

**Decode.** A dedicated GEMV kernel,
`src/kernel_selector/cl_kernels/fully_connected_gpu_gemv.cl` (609 lines),
selected only when the batch dimension is exactly 1
(`kernels/fully_connected/fully_connected_kernel_gemv.cpp:139`:
`output_size.first == 1 && output_size.second % 16 == 0`).

How it works:

- `intel_reqd_sub_group_size(16)` - SIMD16, and it `#error`s on anything else.
- `sub_group_block_read` for weights - coalesced, full-width DRAM streaming.
- int4 dequantised **inline in the accumulation loop**, stepping by
  `DECOMPRESSION_GROUP_SIZE`. No separate dequant pass, no temporary buffer.
- Activations held in registers and `sub_group_broadcast`-ed to all lanes; the
  vector is tiny and fully reused.
- Accumulation is **plain scalar FMA on `half`**.
- The outer loop runs `for (int gk = gk0; gk < gk1; gk++)` - a per-work-group
  K-range, i.e. **split-K**. This is what fills 256 EUs when the output is only a
  few thousand elements.

**There is not a single `matrix_mad` / DPAS intrinsic in any OpenVINO `.cl`
kernel.** Its own OpenCL kernels never touch XMX. For decode that is a correct
and deliberate choice.

**Prefill.** Delegated to oneDNN - `src/graph/impls/onednn/fully_connected_onednn.cpp`,
`gemm_onednn.cpp`, `gated_mlp_onednn.cpp`. oneDNN does support compressed weights
(`"[GPU] oneDNN supports only 4bit/8bit compressed weights"`) and does use XMX.

## What vLLM does - verified from the kernel source

At `M = 1` the `vllm-xpu-kernels` dispatcher selects `w4a16_policy_m_8`:

- work-group tile **8 × 64 × 32**, 64 threads per work-group;
- grid = `sm_count × 8` persistent work-groups;
- **no split-K anywhere** - `tile_coord = make_coord(m, n, _, 0)`.

So a vector problem runs on a matrix kernel: **seven of eight tile rows are
wasted**, and with no split-K the output dimension alone cannot fill the machine.

Two independent confirmations that this - not kernel quality - is the limit:

1. MXFP4 (72.65 t/s) and GPTQ-int4 (73.31 t/s) are two completely different
   kernel paths and land **within 1%**. Both are throttled by the same structural
   problem.
2. The local patch stack added GEMV kernels and recovered **+16%** decode at
   concurrency 2 (GPTQ 102 → 118.5, MXFP4 97 → 112.3).

Measured MBU of 50-63% is the direct consequence.

## The gap neither engine serves: M = 2..8

| `M` | OpenVINO | vLLM | Reality |
|-----|----------|------|---------|
| 1 | dedicated GEMV ✅ | M=8 tile, 7/8 wasted ❌ | bandwidth-bound |
| **2-8** | **falls off to oneDNN** ❌ | tile partly used ❌ | **bandwidth-bound, weights amortisable** |
| large | oneDNN ✅ | XMX tiles ✅ | compute-bound |

OpenVINO's GEMV is hard-gated at `M == 1`, so the moment speculative decoding
produces 2+ tokens it drops onto a general path. vLLM's tile only becomes fully
utilised *at* `M = 8`.

**This is the opportunity.** Weights dominate decode traffic and activations are
negligible, so a **weight-stationary kernel parameterised on `M ∈ [1,8]`**
amortises the entire weight read across `M` tokens. Stream each weight tile once,
multiply it into `M` activation vectors held in registers.

The payoff is concrete. MTP on vLLM already gives **31.5 → 45.2 t/s** (+43%)
*while paying full weight traffic on every draft step*. Make that traffic shared
and speculation approaches free - a gain that **compounds** with closing the
GEMV gap rather than competing with it.

## Design rules that follow

1. **Two kernel families, dispatched on `M`.** Never one kernel pretending to
   serve both. `M == 1` and small-`M` share an implementation; large `M` is a
   separate sycl-tla instantiation.
2. **The decode kernel must split K.** Without it the device cannot be filled at
   realistic output sizes, whatever else is right. **Confirmed on this
   silicon** (2026-08-23, `probe_gemv`, doc 05 item 6): at N = 5120 - the worst
   fill case, 320 subgroups - the int4 GEMV runs at **259 GB/s (43% of 600) at
   S = 1 and 526 GB/s (88%) at S = 16** in the canonical layout, and 262 → 534
   GB/s for `down` (K = 17408); split-K is worth **2.0×** at that shape. It is
   not universal, and it does not track `N` monotonically: at N = 14336 and
   N = 16384 the grid already fills the device and S = 1 is the *best* setting,
   not merely an acceptable one, while `gate‖up` at N = 34816 still needs S = 4
   (S = 1 → 425 GB/s, 71%; S = 4 → 538) despite having 2176 subgroups. Subgroup
   count is necessary, not sufficient - measure per shape (doc 12).
3. **Dequantise inline in the accumulation loop.** Never a separate pass, never a
   temporary buffer - that would double the traffic that is the entire budget.
4. **Parameterise the decode kernel on `M ∈ [1,8]` from the start.** Retrofitting
   this after hard-coding `M == 1` is how OpenVINO ended up with a cliff.
5. **Weight-stationary, activation-broadcast.** Weights are the traffic;
   activations are free. Structure every decode kernel around that asymmetry.

## Prefill: a path neither engine takes

Both vLLM and OpenVINO run prefill as **W4A16** - bf16 activations against int4
weights. On Xe2 that means the weights must be **upconverted to 16-bit in the
mainloop** before the systolic array sees them (`MainloopIntelXeXMX16MixedPrecision`
in `sycl-tla`; the equivalent widening in oneDNN).

But Battlemage has a **native mixed int8 × int4 DPAS** that skips the
upconversion entirely - `oneDNN/src/gpu/intel/gemm/jit/pd.cpp:772` states it is
supported on everything pre-Xe3p, on one condition: the int4 matrix must have
**no zero points**. Every checkpoint on the box is `sym: true`.

So a **W4A8** prefill path would get:

- the int8 systolic throughput (~2× bf16), and
- **zero dequantisation work** in the mainloop.

The cost is dynamic int8 activation quantisation and whatever accuracy that
loses. Prefill is the right place to spend that: it is compute-bound, and its
output is the KV cache plus one token, not a long autoregressive chain.

`sycl-tla` already declares the instruction as CuTe atoms -
`include/cute/arch/mma_xe.hpp:287-297`, `XE_DPAS_TT(d, s8, s4, d)` - but no
mainloop feeds them; `xe_mma_mixed_input.hpp` still widens to 16-bit. The work
is a mainloop, not a driver feature.

This is unproven - see open question 9 - but it is the clearest candidate for
beating *both* engines at prefill rather than merely matching one.

## Summary: where each engine wins, and where we can beat both

| | vLLM | OpenVINO | This project |
|---|---|---|---|
| Decode `M=1` | M=8 tile, no split-K ❌ | dedicated GEMV ✅ | GEMV **+ `M∈[1,8]`** |
| Decode `M=2..8` | tile partly used ❌ | falls off to oneDNN ❌ | **the opportunity** |
| Prefill | W4A16 XMX tiles ✅ | W4A16 via oneDNN ✅ | **W4A8 native s8s4 DPAS** |

Nobody is running W4A8 prefill or an `M∈[1,8]` decode kernel on this silicon.
Those two are the whole thesis.

## Open

Why OpenVINO's prefill is slower than vLLM's is **not established**. oneDNN does
use XMX and does handle int4, so "it doesn't use XMX" is not the explanation -
both engines upconvert int4 in a W4A16 mainloop. The remaining plausible causes,
untested, are that oneDNN's generic path is less tuned for Battlemage than
`sycl-tla`'s Battlemage-specific tiles, and that layout reorders cost an extra
pass. Profile both before assuming a design.
