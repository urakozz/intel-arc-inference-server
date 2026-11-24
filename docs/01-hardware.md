# The Arc Pro B70, measured

This document is for anyone who wants to know what the card underneath the engine can actually do: how fast it
reads memory, how fast its matrix engine multiplies each data type, and which formats it cannot run at all. These
facts constrain every other decision in the project, from the weight format ([02-formats.md](02-formats.md)) to the
shape of each kernel ([12-kernels.md](12-kernels.md)).

Everything here was measured on the card or read out of the driver, not looked up in a datasheet. The probes that
produced it are in `tools/probe/`; rerun them before trusting a figure after a driver update.

## Device properties

Read once from `torch.xpu.get_device_properties(0)`. The properties are the card's and do not change with the
driver.

```
name                   Intel(R) Arc(TM) Pro B70 Graphics
device_id              0xE223
total_memory           32656 MB
last_level_cache_size  24576 KB      <- 24 MB L2
local_mem_size         128 KB        <- per work-group SLM
max_compute_units      256
gpu_eu_count           256
gpu_subslice_count     32
max_work_group_size    1024
max_num_sub_groups     64
sub_group_sizes        [16, 32]
```

SLM (shared local memory) is the fast on-chip scratch a work-group shares. An EU (execution unit) is the smallest
compute block; 8 of them make one Xe-core, which the driver calls a subslice.

The `memory_bus_width` field reports 64-bit, which is wrong: it contradicts the measured bandwidth by 4x. Do not
use it.

The maximum core clock is 2.8 GHz over 32 Xe-cores. The matrix-engine section below derives its peak from that pair.

## Memory bandwidth

Bandwidth matters more than anything else here, because generating one token means reading every weight once
(see [05-perf-model.md](05-perf-model.md)). The first measurement copied device memory to device memory at three
buffer sizes, 20 iterations after 5 warm-ups:

| Buffer | Read-only (`sum`) | Copy (r+w) |
|--------|------------------|------------|
| 0.5 GB | 578 GB/s | 538 GB/s |
| 2.0 GB | 601 GB/s | 529 GB/s |
| 8.0 GB | 601 GB/s | 529 GB/s |

The rate does not fall as the buffer grows far past the 24 MB cache, so this is DRAM bandwidth, not cache.

The card's peak is 638 GB/s, as the vendor's monitoring tool (`hw-smi`) reads it under load (615 GB/s current,
638 GB/s peak). The 601 GB/s above is 94 % of that peak.

The same read was then measured with a plain Level Zero launch (`tools/probe/probe_bw`, `bw_sum.cl`), the way the
engine itself submits work: 590 GB/s median at 2 GB (min 585, max 592) and 584 GB/s median at 8 GB (min 580,
max 588), 20 timed iterations after 5 warm-ups. That agrees with the torch probe within 3 % and is 92 % of the
peak.

**590 GB/s is the roofline denominator.** It is the rate through the launch path this engine actually uses, and
every roofline figure in [05-perf-model.md](05-perf-model.md) divides by it. So
these documents use two bases, and each figure should say which: "of the measured 590" is what a kernel can reach,
"of the 638 peak" is what the memory could carry. A figure computed against 608 GB/s (a 19 Gbps memory clock) is
about 5 % too high.

Seed a bandwidth probe's buffer with incompressible data. The B70 compresses device-local memory losslessly, so a
buffer filled by `zeCommandListAppendMemoryFill` with one repeated 32-bit word reads back at 1022 GB/s, above the
638 GB/s peak, because most of those loads never reach DRAM. Timing is not the culprit: at 1, 2 and 4 launches per
submission the elapsed time was 2.102, 4.203 and 8.402 ms, exactly linear. Any probe that reports bandwidth above
the theoretical peak is measuring compression.

## What the matrix engine actually does

The matrix engine is driven by one instruction, DPAS (dot-product accumulate systolic): it multiplies a small tile
of one matrix by a tile of another and adds the result into an accumulator. Its rate per data type is the most
consequential set of facts about the B70, and it constrains every format decision.

All of it was measured with `tools/probe/probe_dpas_rates`. The Xe2 assembly confirms the probe loops are bound by
DPAS alone: exactly 64 `dpas.8x8` in the loop body for every type, with no register spill, no conversion and no
repacking. In the table, K is the depth of one multiply, the number of elements summed per output.

| builtin family | K | rate | vs bf16 |
|---|---:|---:|---:|
| `bf16_bf16_matrix_mad_k16` | 16 | 183.45 TFLOP/s | 1.000x |
| `f16_f16_matrix_mad_k16` | 16 | 183.45 TFLOP/s | 1.000x |
| `i8_i8_matrix_mad_k32` | 32 | 366.90 TIOP/s | 2.000x |
| `i4_i8_matrix_mad_k32` | 32 | 366.89 TIOP/s | 2.000x |
| `i8_i4_matrix_mad_k32` | 32 | 366.90 TIOP/s | 2.000x |
| `i4_i4_matrix_mad_k64` | 64 | 733.80 TIOP/s | 4.000x |
| `u4_u4_matrix_mad_k64` | 64 | 733.80 TIOP/s | 4.000x |
| `i2_i2_matrix_mad_k64` | 64 | 733.79 TIOP/s | 4.000x |

One model explains the whole table. Divide each rate by its operations per multiply and every type lands on the
same issue rate, 4.479e10 `dpas.8x8` per second, within 0.01 %. That is 16.0 EU-cycles per `dpas.8x8` whatever the
data type, or 2048 bf16 FLOP per Xe-core per cycle. Deepening the multiply from K=16 to K=64 is free; widening it
is not. So the table shows no difference in rate between types, only one constant issue rate seen through three
values of K.

The bf16 peak derived from the clock is 32 Xe-cores x 2048 FLOP/clk x 2.8 GHz = 183.50 TFLOP/s. The measured
183.45 is 99.97 % of it, so the ratios in the table are ratios of the matrix pipe itself and not of some other
bottleneck.

The engine's own prefill GEMM measures 161.7 TFLOP/s on the gate‖up shape at M=2048 (2048 prompt rows), 88.1 % of
the micro-benchmark. A real kernel must sit below the micro-benchmark; the gap is the GEMM's memory, barrier and
epilogue cost.

### The mixed int8 x int4 DPAS is real, and it is worth nothing extra

oneDNN's JIT GEMM generator documents a native mixed int8 x int4 DPAS on this hardware, in
`src/gpu/intel/gemm/jit/pd.cpp`:

```c
// Mixed s8/s4 DPAS support:
// - Xe3p: Not supported, require s4->s8 upconversion
// - pre-Xe3p: supported, but only when s4 matrix doesn't have zero points
bool has_s8s4_dpas = getCore(problem.product.family) != ngen::HW::Xe3p;
if (problem.Tb_ext.isInt4() && problem.Ta_ext.isInt8()) {
    bool s8s4_dpas_ok = has_s8s4_dpas && (b_quant.zp_ndims < 0);
    if (!s8s4_dpas_ok) problem.Tb = Type::s8;   // else widen int4 -> int8
}
```

Battlemage, the B70's architecture, is pre-Xe3p, so it has the path, and our own probe confirms the instruction
issues natively: the assembly carries `:s4` on one operand of a single `dpas.8x8`, with no widening to bytes.

But the mixed builtin's K is 32, the same as int8's, so the 4-bit operand shortens no part of the pipe.
`i8_i4_k32` measures 366.90 TIOP/s against `i8_i8_k32`'s 366.90, identical within 0.01 %. The matrix-engine
ceiling of a W4A8 linear (4-bit weights, 8-bit activations) is therefore 2.000x bf16, exactly that of W8A8, and
no W4A8 win could come from the math pipe.

That prediction was tested with a real kernel on real weights and activations, and W4A8 was rejected for
Qwen3.8's dense layers. The per-group rescale costs more than the 2x instruction rate buys, and the activation
outliers cost 2.79 % relative L2 error against the bf16 result ([02-formats.md](02-formats.md) has the numbers).
The one place W4A8 runs today is the expert layers of MoE (mixture-of-experts) models under the optional int8
prompt pass, `--prefill-backend l0-int8` ([16-int8-prefill.md](16-int8-prefill.md)).

The one DPAS lever on this device worth more than 2x is 4-bit by 4-bit, at 4.000x. W4A4 does not reach acceptable
accuracy: int4 activations at group 256, even after a Hadamard rotation (a fixed orthogonal transform that spreads
outliers across a vector before quantising it), double Qwen3.8's output error against int8's
([05-perf-model.md](05-perf-model.md#what-was-measured-and-rejected)).

### FP4 and FP8 are not available

- **FP4 is refused by the backend.** `e2m1_e2m1_matrix_mad_k64` type-checks, then the backend rejects it by name:
  `FP4 Dpas instruction is not supported on this device!` So MXFP4 buys nothing on this card today. An FP4 path
  would have to decode `e2m1` in the ALU and issue the int4 or bf16 form anyway, which makes it int4 with extra
  work.
- **Every `scaled_matrix_mad` form crashes the compiler**: the bf16, fp16, hf8, bf8 and e2m1 variants alike. That
  is how hardware microscaling (a shared scale per small block of values) would be reached, so there is no
  microscaling path.
- **No unscaled FP8 entry point is declared.** The compiler's header declares only the scaled FP8 forms, and those
  are the ones that crash. There is no reachable FP8 matrix multiply.

The block-scaled MX examples in `sycl-tla` are `xe35`-prefixed (`50_xe35_*`, `51_xe35_*`), targeting Xe3.5 rather
than Battlemage. MXFP4 on the B70 is not a first-class path in Intel's own kernel library either.

### Three consequences

1. **int4 AutoRound, int4 GPTQ and MXFP4 are the same hardware path** once the weights reach the matrix engine.
   They differ only in packing and scale layout, so choosing between them is a bytes-per-weight and accuracy
   decision, never a kernel-architecture one.
2. **FP8 has no hardware advantage here.** On Hopper and Blackwell FP8 wins because those GPUs have native FP8
   tensor cores. This card has no reachable FP8 multiply at all, and FP8 costs twice the memory traffic of int4.
   For a bandwidth-bound decode it is the worst choice, not the safe one.
3. **Halving the operand width only helps when it doubles K.** int8 buys 2x over bf16 because K doubles. int4
   against int8 buys nothing, because K does not change.

## Implications for kernel design

The 24 MB L2 is large for this class of card. A 4096x4096 int4 weight tile is 8 MB, so keeping several layers'
weights in cache is plausible in a way it is not on most GPUs. It is worth measuring before assuming every weight
read hits DRAM.

The subgroup sizes are 16 and 32. A subgroup is the set of lanes that execute one instruction together, the
counterpart of a CUDA warp. Matrix tile shapes and the GDN kernel's lane assignment must be built around these
sizes, not the 32 assumed by code derived from CUDA. For reference, an existing SIMD16 GEMV works this way, and
`vllm-xpu-kernels`' GDN decode kernel is SIMD32 with 256-thread groups.

There are 256 EUs over 32 subslices. The persistent-kernel grid of the grouped GEMM in `vllm-xpu-kernels` is
`sm_count * 8`; the equivalent constant here has to be derived from `gpu_subslice_count`, not copied.

The subgroup, not the work-group, is the unit of occupancy. A bit-identical rewrite that gave a kernel 4x the
work-groups at an unchanged subgroup count bought exactly nothing; splitting the same work across 16x the
subgroups took the launch from 48.8 to 5.3 us.

`ocloc`, the offline compiler for the kernels, has no 16-bit block write wider than 8 rows. A bf16 epilogue is
therefore bound by the number of store messages rather than by bytes, which is why halving the width of
intermediate buffers changed nothing.

## Two cards, and the software stack behind these numbers

The numbers in this document were measured on a machine with two B70s on PCIe 3.0 x16 and 121 GB of host RAM.

### Peer access between the cards

Peer-to-peer access (P2P: one card writing straight into the other's memory over PCIe, without bouncing through
host RAM) needs a kernel that supports it natively - mainline Linux 7.4.0-rc2 does; earlier kernels needed a patch -
and both cards under one PCIe root complex. The hardware check refuses two visible cards that cannot both reach each
other's memory, with the advice to run on one card with `ZE_AFFINITY_MASK=0`. Both two-card modes use it:

- tensor parallel (every layer split across both cards): each card writes its partial results straight into the
  other card's memory every layer;
- pipeline parallel (the layers split between the cards): the first card copies the hidden state into the second
  card's memory once per step, ordered by a cross-device event.

The BAR is the window through which the host or the other card reaches a card's memory. A firmware resizable-BAR
setting may cap it at 16 GB per card; the `xe` kernel driver resizes both cards' BARs to the full 32 GB when it loads.

A lasting "busy" is a hang. The Level Zero runtime retries an ioctl that returns EBUSY forever, so a lasting EBUSY
never surfaces as an error. One large pinned host allocation that the second card maps through a shared import (the
prefix cache) triggered it. This is why any host allocation that both cards map stays at or below 4 GiB; the prefix
cache, for one, is pinned in 2 GiB pieces.

### Device selection

The CLIs have no device flag ([04-architecture.md](04-architecture.md)): `ZE_AFFINITY_MASK` decides which cards the
driver enumerates; one card is device 0 of that view, and two are devices 0 and 1 (`ZE_AFFINITY_MASK=1` runs on the
second card alone). Where one card also drives a display, the render nodes do not map to selector indices the way you
would guess, and the display card is not the one you want under a benchmark.

The two cards may not be interchangeable. In one session, with the same binary and checkpoint, prefill lost 3.2% to
3.4% on the second card while decode lost only 0.2% to 0.4%: a difference in sustained compute rather than bandwidth.
The likeliest cause is a hardware monitoring tool left open during that measurement, but it has not been re-measured
([07-open-questions.md](07-open-questions.md)), so a timed row names its card.

### Driver versions tested

| | version |
|---|---|
| kernel | Linux 7.4.0-rc2 (mainline) |
| IGC (Intel Graphics Compiler, which turns OpenCL C into GPU code) | 2.42.6 |
| compute runtime (`libze-intel-gpu1`, `intel-opencl-icd`, `intel-ocloc`) | 26.40.40196.15 |
| Level Zero loader | 1.34 |

Kernel binaries cannot go stale silently. `cmake/ocloc_stamp.cmake` records `ocloc --version` and the real paths of
the `libigc` / `libigdfcl` libraries the loader resolves, and rewrites a stamp file only when they change. Every
kernel target depends on that stamp, so a new compiler recompiles every kernel.

The rates in this document were measured on earlier driver stacks. The hardware rates (DPAS, bandwidth) are
properties of the silicon and are not expected to move. Kernel timings can move with the compiler; the current speed
numbers are in the [README](../README.md#speed).
