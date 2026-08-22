# The Arc Pro B70, measured

Everything here was measured on the box or read out of the driver, not looked up.
Reproduce with the probes in `tools/probe/` before trusting any of it after a
driver bump.

## Device properties

Read from `torch.xpu.get_device_properties(0)` on the running box:

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

`memory_bus_width` reports **64-bit**, which is wrong - it contradicts the
measured bandwidth by 4×. Do not use that field.

## Memory bandwidth - measured

Device-to-device, three buffer sizes, 20 iterations after 5 warm-ups:

| Buffer | Read-only (`sum`) | Copy (r+w) |
|--------|------------------|------------|
| 0.5 GB | 578 GB/s | 538 GB/s |
| 2.0 GB | **601 GB/s** | 529 GB/s |
| 8.0 GB | **601 GB/s** | 529 GB/s |

Flat from 0.5 GB upward, so this is DRAM bandwidth, not cache. **Use 600 GB/s as
the read-bandwidth denominator in every roofline calculation.** That is ~99% of a
256-bit GDDR6 @ 19 Gbps theoretical peak (608 GB/s), which also tells you the
hardware peak.

> An earlier probe reported 1150 GB/s. It was taken while a model was still
> downloading and is wrong. Discard it.

## What the matrix engine can actually do

This is the single most consequential fact about the B70, and it constrains
every format decision. From the `sycl-tla` mainloop dispatch policies
(`~/PycharmProjects/sycl-tla/examples/`):

| Policy | Math performed |
|--------|----------------|
| `MainloopIntelXeXMX16` | bf16 / fp16 - **native** |
| `MainloopIntelW8A8` | int8 × int8 - **native** |
| `MainloopIntelXeXMX16MixedPrecision` | int4/int8 weights → **upconverted** to XMX16 |
| `MainloopIntelXeXMX16FP8Scaling` | FP8 → **upconverted** to XMX16 |
| `MainloopIntelXeXMX16BlockScaled` | MX formats → **upconverted** to XMX16 |

**With 16-bit activations, Xe2 XMX executes exactly two things natively: 16-bit
float and int8×int8.** Every narrow format - int4, MXFP4, FP8 - is dequantised in
the mainloop and the multiply happens at 16-bit.

### The exception: native mixed s8 × s4 DPAS

`sycl-tla` is not the whole story. oneDNN's JIT GEMM generator documents a
**native mixed int8 × int4 DPAS** on this hardware
(`~/PycharmProjects/oneDNN/src/gpu/intel/gemm/jit/pd.cpp:772`):

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

Battlemage is **pre-Xe3p, so it has this path**: int4 weights feed the systolic
array directly, with no upconversion to 16-bit.

Two conditions:

- **int8 activations** (W4A8, not W4A16) - requires dynamic activation
  quantisation and pays its accuracy cost.
- **No zero points on the int4 matrix**, i.e. *symmetric* quantisation only.
  Every checkpoint on the box is `sym: true`, so this holds.

This matters for **prefill**, which is compute-bound: it combines the int8
systolic path (~2× bf16 throughput) with zero dequantisation work. It does
nothing for decode, which is bandwidth-bound - there the cost is int4 weight
*bytes*, not the multiply. See [08-decode-vs-prefill.md](08-decode-vs-prefill.md).

`sycl-tla` exposes the same instruction as CuTe MMA atoms:
`include/cute/arch/mma_xe.hpp:287-297` declares `XE_DPAS_TT(d, s8, s4, d)` and
every `u8/u4/s8/s4` permutation, compiled out only for CRI with the comment
"Skip int8 x int4 for CRI as the dpas is removed". **No mainloop or example uses
them** - `xe_mma_mixed_input.hpp` still widens int4 to 16-bit. So the path is
reachable at the atom level; the job is a mainloop, not a driver feature.

Unverified: whether it is actually *faster* on a B70, and what it costs in
accuracy. Open question 9 in [07-open-questions.md](07-open-questions.md).

Three consequences of the 16-bit-activation table above, each load-bearing:

1. **int4-AutoRound, int4-GPTQ and MXFP4 are the same hardware path.** They differ
   only in packing and scale layout. Choosing between them is a bytes-per-weight
   and accuracy decision, never a kernel-architecture one.
2. **FP8 has no hardware advantage here.** On Hopper/Blackwell FP8 wins because
   there are native FP8 tensor cores. Xe2 has none - FP8 upconverts like
   everything else, while costing 2× the memory traffic of int4. It is the worst
   choice for a bandwidth-bound decode, not the safe one.
3. **W8A8 is the one genuinely different path** and it is a *compute* win
   (int8 XMX has ~2× the bf16 throughput). Batch-1 decode is not compute-bound,
   so it is irrelevant to phase 1 - but it is the natural weapon for prefill.

Block-scaled MX examples in `sycl-tla` are `xe35`-prefixed (`50_xe35_*`,
`51_xe35_*`), i.e. targeted at Xe3.5, not Battlemage. MXFP4 on the B70 is not a
first-class path in Intel's own kernel library. Phase 3 should expect to write
or adapt this rather than lift it.

## Implications for kernel design

- **24 MB of L2 is a lot.** A 4096×4096 int4 weight tile is 8 MB. Multi-layer
  weight residency is plausible in a way it is not on most GPUs - worth measuring
  before assuming every weight read hits DRAM.
- **Subgroup sizes are 16 and 32.** XMX tile shapes and the GDN kernel's lane
  assignment must be built around these, not the 32 assumed by CUDA-derived code.
  For reference, OpenVINO's GEMV is SIMD16 and `vllm-xpu-kernels`' GDN decode
  kernel is SIMD32 with 256-thread groups (`gdn_attn/gated_delta_rule.hpp:5-9`).
- **256 EUs / 32 subslices.** The persistent-kernel grid used by the grouped GEMM
  in `vllm-xpu-kernels` is `sm_count × 8`; the equivalent constant here needs to
  be derived from `gpu_subslice_count`, not copied.
