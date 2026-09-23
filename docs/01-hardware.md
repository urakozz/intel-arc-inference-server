# The Arc Pro B70, measured

Everything here was measured on the card or read out of the driver, not looked
up. Reproduce with the probes in `tools/probe/` before trusting any of it after
a driver bump.

## Device properties

Read from `torch.xpu.get_device_properties(0)`:

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
driver_version         1.17.39395+13
platform               oneAPI Unified Runtime over Level-Zero V2
```

`memory_bus_width` reports **64-bit**, which is wrong: it contradicts the
measured bandwidth by 4x. Do not use that field.

Max core clock is **2.8 GHz** over 32 Xe-cores. That pair is what the DPAS
section below derives its peak from.

## Memory bandwidth

Device-to-device, three buffer sizes, 20 iterations after 5 warm-ups:

| Buffer | Read-only (`sum`) | Copy (r+w) |
|--------|------------------|------------|
| 0.5 GB | 578 GB/s | 538 GB/s |
| 2.0 GB | **601 GB/s** | 529 GB/s |
| 8.0 GB | **601 GB/s** | 529 GB/s |

Flat from 0.5 GB upward, so this is DRAM bandwidth, not cache. That is about
99% of a 256-bit GDDR6 at 19 Gbps theoretical peak (608 GB/s), which also tells
you the hardware peak.

Re-measured with a plain Level Zero launch (`tools/probe/probe_bw`, `bw_sum.cl`):
**590 GB/s** median at 2 GB (min 585, max 592) and 584 GB/s median at 8 GB
(min 580, max 588), 20 timed iterations after 5 warm-ups. Agrees with the torch
probe within 3%.

**Use 590 GB/s as the roofline denominator.** It is the rate through the launch
path this engine actually submits on, and it is the one every roofline figure in
`docs/BENCHMARKS.md` and `05-perf-model.md` divides by.

Seed the buffer with incompressible data. The B70 compresses device-local
memory losslessly, so a `zeCommandListAppendMemoryFill` of one repeated 32-bit
word reads back at 1022 GB/s, above the 608 GB/s theoretical peak, because most
of those loads never reach DRAM. Timing is not the culprit: at 1, 2 and 4
launches per submission the elapsed time was 2.102 / 4.203 / 8.402 ms, exactly
linear. Any probe reporting bandwidth above the theoretical peak is measuring
compression.

## What the matrix engine actually does

This is the single most consequential set of facts about the B70, and it
constrains every format decision. All of it is measured by
`tools/probe/probe_dpas_rates`, whose loops the Xe2 assembly confirms are DPAS
bound: exactly 64 `dpas.8x8` in the loop body for every type, no spill, no
conversion, no repack.

| builtin family | K | rate (measured) | vs bf16 |
|---|---:|---:|---:|
| `bf16_bf16_matrix_mad_k16` | 16 | **183.45 TFLOP/s** | **1.000x** |
| `f16_f16_matrix_mad_k16` | 16 | 183.45 TFLOP/s | 1.000x |
| `i8_i8_matrix_mad_k32` | 32 | **366.90 TIOP/s** | **2.000x** |
| `i4_i8_matrix_mad_k32` | 32 | 366.89 TIOP/s | 2.000x |
| `i8_i4_matrix_mad_k32` | 32 | 366.90 TIOP/s | 2.000x |
| `i4_i4_matrix_mad_k64` | 64 | **733.80 TIOP/s** | **4.000x** |
| `u4_u4_matrix_mad_k64` | 64 | 733.80 TIOP/s | 4.000x |
| `i2_i2_matrix_mad_k64` | 64 | 733.79 TIOP/s | 4.000x |

**One model explains the whole table.** Divide each rate by its ops per mad and
every type lands on the same issue rate, 4.479e10 `dpas.8x8` per second, within
0.01%. That is **16.0 EU-cycles per `dpas.8x8`** regardless of data type, or
2048 bf16 FLOP per Xe-core per cycle. The array's depth scaling from K=16 to
K=64 is free; its width is not. Nothing in the table is a rate difference
between types, only one constant issue rate seen through three values of K.

The clock-derived bf16 peak is 32 Xe-cores x 2048 FLOP/clk x 2.8 GHz =
**183.50 TFLOP/s**. Measured 183.45 is **99.97%** of it, so the ratios are
ratios of the pipe and not of some other bottleneck.

**Our production GEMM measures 161.7 TFLOP/s** on the gate‖up shape at M=2048,
**88.1%** of the micro-benchmark. The micro-benchmark sits above the production
kernel, as it must; the gap is the GEMM's memory, barrier and epilogue cost.

Details, including the assembly, in
[probe-dpas-rates-2026-09-22.md](probe-dpas-rates-2026-09-22.md).

### The mixed s8 x s4 DPAS is real, and it is worth nothing extra

oneDNN's JIT GEMM generator documents a native mixed int8 x int4 DPAS on this
hardware, in `src/gpu/intel/gemm/jit/pd.cpp`:

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

Battlemage is pre-Xe3p, so it has the path, and our own probe confirms the
instruction issues natively: the assembly carries `:s4` on one operand of a
single `dpas.8x8`, with no widening to bytes.

**But the mixed builtin's K is 32, the same as int8's.** The 4-bit operand
shortens no part of the pipe. `i8_i4_k32` measures 366.90 TIOP/s against
`i8_i8_k32`'s 366.90 - identical to within 0.01%. So the DPAS ceiling of a W4A8
linear path is 2.000x bf16, exactly the same as a W8A8 one, and none of a W4A8
win could come from the math pipe.

That prediction was tested with a real kernel on real weights and real
activations, and **W4A8 was rejected**: the per-group rescale costs more than
the 2x instruction rate buys, and the activation outliers cost 2.79% relative
L2 error against the bf16 result. See
[probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md).

The one DPAS lever on this device worth more than 2x is **4-bit by 4-bit**, at
4.000x. Whether W4A4 is reachable at acceptable accuracy is untested here.

### FP4 and FP8 are not available

- **FP4 is refused by the backend.** `e2m1_e2m1_matrix_mad_k64` type-checks and
  then the backend rejects it by name: `FP4 Dpas instruction is not supported on
  this device!` So MXFP4 buys nothing on this card today; an FP4 path would have
  to decode `e2m1` in the ALU and issue the int4 or bf16 form anyway, which
  makes it int4 with extra work.
- **Every `scaled_matrix_mad` form crashes the compiler** - the bf16, fp16,
  hf8, bf8 and e2m1 variants alike. That is how hardware microscaling would be
  reached, so there is no microscaling path.
- **No unscaled FP8 entry point is declared.** The compiler's header declares
  only the *scaled* FP8 forms, and those are the ones that crash. There is no
  reachable FP8 matmul.

Block-scaled MX examples in `sycl-tla` are `xe35`-prefixed (`50_xe35_*`,
`51_xe35_*`), targeted at Xe3.5 rather than Battlemage. MXFP4 on the B70 is not
a first-class path in Intel's own kernel library either.

### Three consequences, each load-bearing

1. **int4-AutoRound, int4-GPTQ and MXFP4 are the same hardware path** once the
   weights reach the array. They differ only in packing and scale layout.
   Choosing between them is a bytes-per-weight and accuracy decision, never a
   kernel-architecture one.
2. **FP8 has no hardware advantage here.** On Hopper and Blackwell FP8 wins
   because there are native FP8 tensor cores. This card has no reachable FP8
   matmul at all, and FP8 costs 2x the memory traffic of int4. It is the worst
   choice for a bandwidth-bound decode, not the safe one.
3. **Halving the operand width only helps when it halves K.** int8 buys 2x over
   bf16 because K doubles. int4 against int8 buys nothing, because K does not.

## Implications for kernel design

- **24 MB of L2 is a lot.** A 4096x4096 int4 weight tile is 8 MB. Multi-layer
  weight residency is plausible in a way it is not on most GPUs, and worth
  measuring before assuming every weight read hits DRAM.
- **Subgroup sizes are 16 and 32.** XMX tile shapes and the GDN kernel's lane
  assignment must be built around these, not the 32 assumed by CUDA-derived
  code. For reference, OpenVINO's GEMV is SIMD16 and `vllm-xpu-kernels`' GDN
  decode kernel is SIMD32 with 256-thread groups.
- **256 EUs over 32 subslices.** The persistent-kernel grid used by the grouped
  GEMM in `vllm-xpu-kernels` is `sm_count * 8`; the equivalent constant here has
  to be derived from `gpu_subslice_count`, not copied.
- **The subgroup is the occupancy unit, not the work-group.** A bit-identical
  rewrite that gave a kernel 4x the work-groups at an unchanged subgroup count
  bought exactly zero; splitting the same work across 16x the subgroups took the
  launch from 48.8 to 5.3 us.
- **`ocloc` has no 16-bit block write wider than 8 rows.** A bf16 epilogue is
  therefore bound by store message count rather than by bytes, which is why
  halving intermediate buffer widths changed nothing.
