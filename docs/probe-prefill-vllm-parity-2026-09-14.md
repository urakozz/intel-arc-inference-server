# Writing our own DPAS GEMM, and measuring the vendor's (2026-09-14 to 09-15)

This is the investigation that decided prefill would drop SYCL entirely and run
its own OpenCL C GEMM on the same Level Zero list as everything else. Four
things were measured:

| what | outcome |
|---|---|
| **Our own bf16 DPAS GEMM**, written from scratch for this card | **156.53 TFLOP/s** at gate‖up M=2048, bitwise identical to the reference and **17.9% faster** than it in the same harness |
| **Interleaving slab dequant with that GEMM** on one in-order list, zero host waits | **+89.29 ms** per chunk-equivalent, device-side handoff **1.667 µs** |
| **oneDNN's fused int4 W4A16 matmul**, the primitive vLLM itself calls | **98.28 TFLOP/s**, below the 115.7 break-even an in-kernel int4 GEMM would need, and it fails a bitwise identity check against our dequant oracle |
| **Intel's chunked delta net kernel for Xe2** | builds torch-free, but its numerics did not clear this probe's own bar, so its time is not a trusted price |

The first two were adopted and are the production prefill path. The second two
were not. The combination of the first two was predicted to land at 1529 t/s,
77.5% of the vLLM figure then in use, and it did.

Numbers are labelled **measured**, **derived** (arithmetic on measured inputs,
shown) or **estimated**.

---

## Part 1: what vLLM actually runs

Read from source rather than inferred, because the whole comparison depends on
knowing which kernels are on the other side.

### The linear layers go through oneDNN's int4 matmul

```
vLLM's XPU mixed-precision linear kernel
  -> torch.ops._xpu_C.int4_gemm_w4a16
  -> oneDNN's dnnl_matmul_w4a16_int4
  -> a matmul primitive with attribute-driven weight decompression
  -> the JIT GEMM generator
```

This is the production dispatch for GPTQ-style int4 weights on XPU with bf16 or
fp16 activations, not a side path.

The primitive's attributes, exactly:

- **`bf16_int4` joint dtype**, so weight decompress-and-multiply is fused into
  the GEMM mainloop and **there is no separate bf16 dequant scratch pass**.
- **Weight scale grouped along K**, with the group size supplied at call time.
  oneDNN requires K-grouped weight scales to be a multiple of 16; vLLM's own
  gate is stricter at a multiple of 32. Our checkpoint is g64, which satisfies
  both.
- **Zero points**: the symmetric case is a scalar `s8` constant broadcast, and
  both the quantisation converter and vLLM's own post-load hook set that
  constant to **8**, the same "GPTQ unsigned nibbles, zero is 8" convention our
  loader uses. Our checkpoint is symmetric, so there is no zero tensor at all.

### The weight layout it needs is a transpose of ours

**oneDNN needs the weights K-contiguous per N, not the N-contiguous layout our
loader uses**, and this is the identical physical transpose our own earlier
probe of a different vendor GEMM had already measured and priced.

Two independent code paths in vLLM's stack agree that the incoming GPTQ tensor,
packed `[K/8][N]` and N-contiguous, is relaid out before oneDNN sees it: the
quantisation converter forces the stride order explicitly, and the linear
kernel's post-load hook does the same before the op is ever called. The C++
side confirms the consumed stride: the leading dimension it computes only comes
out right if the last-dim stride is already `K/8`.

Layout 0's 4-bit address is piecewise in `k`, so it is unusable verbatim; a
previous probe measured 90.92% wrong words when fed raw, and the fix was
exactly a u32 transpose of the packed weight. oneDNN's requirement is the same
transpose direction. The only difference is that oneDNN carries the zero point
as an attribute rather than as an XOR on the weight, so there is no bit flip
and no separate zero tensor in the symmetric case. The byte cost of the repack
is the same size class: about **11.33 GiB** for the 256 layer matrices, roughly
6.4 s of one-time host work, which is a real residency question on a 32 GB
card.

Activations and outputs, by contrast, can be zero-copy: oneDNN's SYCL interop
wraps existing USM pointers directly, and our prefill context already exposes
the SYCL queue and context handles that interop expects. **Weights cannot**,
because no flat stride can express layout 0's addressing.

### The delta net goes through a native SYCL kernel, not Triton

Each delta net layer's XPU forward is four stages: two input-projection GEMMs
through the int4 path above, one fused op that runs a chunked causal
convolution and then the chunked gated delta rule, an `RMSNormGated`, and an
output projection through the int4 path again. Full-attention layers use vLLM's
generic attention module over a separate flash-attention kernel, with their QKV
and output projections on the same int4 path.

**A correction worth recording.** The Triton `fla_chunk_gated_delta_rule` path
exists in vLLM's model code, and it is what our own early design notes cited as
the reference kernel. **XPU never takes it.** The XPU forward unconditionally
calls the native SYCL op. So the correct comparison point on this hardware is
the Xe2 chunked kernel, and the Triton citation describes vLLM's CUDA and ROCm
path rather than the one actually running here.

**Net read**: vLLM's per-chunk time is spent across the *same* categories our
own ledger already tracks, and nothing in its code performs a fundamentally
different algorithm. Whether it wins is therefore entirely a question of which
individual kernels beat ours.

That framing is what the rest of this document measures.

---

## Part 2: the GEMM, designed

The kernel is `pf_gemm_bf16`, later promoted to `pf_gemm.cl`. Every choice
below carries its reason, and where the evidence did not settle a choice, it
says so.

### Contract

```
C[M][N] fp32  =  A[M][K] bf16  ·  B[K][N] bf16          (lda = K, ldb = N, ldc = N)
```

That is the existing GEMM's exact contract and the layout the production walk
already holds: `A` is the bf16 activation, `B` is the dequant scratch, that is
the weight matrix stored transposed, so the product is `C = A · Wᵀ`. **No
repack, no second copy, no producer changes.**

```c
__attribute__((reqd_work_group_size(512, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pf_gemm_bf16(__global const ushort* restrict A,
                           __global const ushort* restrict B,
                           __global float* restrict C,
                           uint M, uint K, uint N);
```

Work-group `(gx, gy)` owns the output tile at `m0 = 256*gx, n0 = 256*gy`, with
**x on the M axis** so consecutive dispatch ids share a B column block.
Requirements, all met by every production shape (derived): `M % 256 == 0`,
`N % 256 == 0`, `K % 32 == 0`, bases 64-byte aligned, pitches multiples of 16
bytes. Every pitch here is `K*2`, `N*2` or `N*4` bytes, all multiples of 64.

### Tiling, and why those sizes

**Work-group tile 256 (m) x 256 (n) x 32 (k); 32 sub-groups in an 8 by 4
arrangement; sub-group tile 32 x 64 x 32; DPAS atom 8 x 16 x 16.**

These are the reference implementation's numbers, kept deliberately: they are
the only tile whose rate on this part is measured, and any deviation would have
been an unmeasured variable in a one-build probe. The reasons they are right,
argued rather than inherited:

| level | what lives there | size | why |
|---|---|---:|---|
| **GRF, one thread is one sub-group** | C accumulators, 32x64 fp32 | **128 GRF** | 16 independent `dpas` chains per k-step hide the pipe latency; 32x64 is the largest tile whose accumulators plus one k-tile of operands fit in 256 GRF |
| | A fragments, 32 m by 32 k bf16 | 32 GRF | one 2D block load |
| | B fragments, 32 k by 64 n bf16 | 64 GRF | two 2D block loads |
| | **total live** | **224 of 256** | about 30 left for descriptors, loop and coordinates. This is why 256 GRF per thread is mandatory, and why a 128-GRF build of an earlier kernel lost 33x |
| **L1, per Xe-core, no SLM** | the group's current k-tile of A (16 KB) and B (16 KB), plus two prefetched k-tiles | 96 KB or less in flight | every B line is read by the 8 m-subgroups and every A line by the 4 n-subgroups, a 6x L1 request amplification. SLM staging is the CUDA idiom: on Xe the 2D block loads deliver fragments straight to GRF and L1 provides the intra-group reuse, so SLM would add barriers and copies for nothing |
| **L2, 24 MB** | per wave of 32 groups: B 10.5 MB read from DRAM once then from L2 by seven more groups, A 21.0 MB, C 8.4 MB written | about 40 MB touched per wave | a slab probe measured exactly this B-residency effect: a 10.5 MB slab runs faster than full width |
| **DRAM** | A 21.0 MB, B 356.5 MB, C 285.2 MB at gate‖up M=2048 | 663 MB minimum, at most **1.36 GB** with A re-read per wave | at 4.75 ms that is **285 GB/s at most, 48% of the 590 GB/s reference** (derived). DRAM is not the roofline. The XMX issue rate and latency hiding are |

Arithmetic intensity at the sub-group: per k-tile, 2 x 32 x 64 x 32 = 131
kFLOP on 6 KB of fragments, **21.3 FLOP/B from L1** (derived). That is what
lets four threads on one vector engine keep one XMX pipe fed while each other's
loads return.

**Raster.** The reference scheduler picks m-fastest when there are more n tiles
than m tiles, which is true for every production shape, so its measured rates
were taken with that wave composition. Our kernel makes it explicit. Owning the
raster is a lever the library does not give us: at M=4096, where A exceeds L2, a
grouped raster that walks all of N over m-blocks 0 to 7 first would halve A's
re-read traffic at the cost of reading B twice. Named, not built.

### Sub-group size, grid, occupancy

- **SIMD16**, because the bf16 matrix-mad builtin is defined for sub-group 16.
- **Work-group 512 work-items, 32 sub-groups, 32 hardware threads.** At 256 GRF
  a vector engine holds **4** threads rather than 8, so an Xe-core's 8 engines
  hold **32 threads, exactly one work-group**. Device-wide that is 32 Xe-cores
  x 8 engines x 4 threads = **1024 thread slots, 32 work-groups in flight**.
- **Waves** (derived): gate‖up at M=2048 is 8 x 136 = **1088 groups, 34.0
  waves, no tail**. Per group, 2 x 256 x 256 x 5120 = 671 MFLOP. At 150
  TFLOP/s an Xe-core takes about 143 µs per group, and 34 waves x 143 µs =
  4.86 ms against the reference's measured 4.751 ms. The model reproduces the
  record to 2%.
- Sub-group `s` maps as `sm = s >> 2` for rows and `sn = s & 3` for columns, so
  the four sub-groups sharing A rows are adjacent.

### B: no repack, no staging

**The VNNI pairing DPAS needs is done by the load itself**, by the hardware's
transform message:

```c
uint bfrag[2][32];                       // [n-half][32 dwords] = 2 × 2 KB
intel_sub_group_2d_block_read_transform_16b_32r16x2c(
    (__global void*)B, (int)(N * 2u), (int)K, (int)(N * 2u),
    (int2)((int)(n0 + 64u * sn + 32u * h), (int)k0), bfrag[h]);   // h = 0, 1
```

One message loads 32 k-rows by 32 n, that is 2 KB in 32 full 64-byte lines, and
returns per lane the dword `(B[k0+2j+1][n] << 16) | B[k0+2j][n]`, even k in the
low half, which is precisely the B fragment DPAS wants. Each n-atom's operand
is a compile-time register slice, with no `mov`. Memory cost: **zero**. No
second copy, no per-launch pass, no change to the dequant that produces B.

Two alternatives, considered and not taken:

- **K-contiguous B with the transpose message**: one 512-byte message per
  fragment against the transform load's 2 KB, so 4x the messages and 2x the
  line touches for the same bytes. It is the right tool for the attention KV
  cache, not for a scratch we control.
- **A pre-tiled scratch read with 1D block reads**: equal line efficiency, but
  it changes the dequant kernel's write pattern, and that kernel is already
  write-allocate bound, for no gain over the transform load. Kept as the
  fallback if the transform builtin failed to compile.

### A, C, and the prefetch

A is read row-major as the producers write it, one 2 KB message per sub-group
per k-tile. C accumulates in 16 fp32 vectors, zero-initialised, k ascending in
16-wide steps, one chain per output element, with the 32 `dpas` per k-tile
issued **b-outer, a-inner** so consecutive instructions share the B operand,
which is the pattern the compiler fuses into a macro chain. The epilogue writes
each atom with one 2D block store, 16 stores per sub-group, no C read.

Prefetch is **cooperative and two k-tiles ahead**: the group's A and B tiles for
k-tile `t+2` are split across the 32 sub-groups so each issues two small
messages and the whole 32 KB lands in L1 once. The loop body is: split barrier
arrive, load A and B for `t`, prefetch `t+2`, 32 `dpas`, split barrier wait.
The split barrier keeps the sub-groups within one k-tile of each other so their
L1 requests for shared lines coincide; it never waits on data.

Why two ahead and not more: at about 143 µs per group over 160 k-tiles, a
k-tile is about 0.9 µs, so two tiles cover about 1.8 µs of DRAM latency plus
contention. There was no GEMM-side distance measurement to appeal to, so two
was kept as the reference's own configuration and the distance was named as the
first post-probe knob rather than a probe variable.

### Why it can hit the rate

Per k-tile per thread: 3 block loads, 2 prefetches, 32 `dpas` of 4096 FLOP
each, about 2 barrier ops and a handful of coordinate adds. Four threads share
one engine's XMX, so per k-tile the engine must issue 128 `dpas`; at the 183.5
TFLOP/s peak that is one `dpas.8x8` per **5.7 ns per engine** (derived), about
730 ns of XMX work per k-tile round, during which the 12 loads per engine must
return from L1. Hundreds of cycles, well inside the window.

The steady state is XMX bound provided the prefetch keeps the loads in L1, the
16 accumulator chains keep the pipe from stalling on its own latency, and the
compiler keeps everything in 256 GRF. Those three are the load-bearing
assumptions, and all three are checked in the assembly before the clock runs.

### The assembly gate, and why it runs before the clock

The 256-GRF option must be on the **compiler's command line at build time**. A
runtime environment variable never reaches an ahead-of-time compile. That was a
lesson learned the expensive way on an earlier kernel, and the built artifact is
the evidence.

Before any timing counts, the dump is inspected for:

1. Metadata: `grf_count 256`, `simd_size 16`, `has_dpas true`, `spill_mem_size`
   and `private_size` **absent**, `barrier_count 1`, `slm_size 0`.
2. Assembly: **32 `dpas.8x8` per k-tile body**, in runs of four sharing one
   operand and marked as a macro; **3 data-returning 2D block loads** plus **2
   prefetch-form loads** per k-tile; **16 2D block stores** in the epilogue; no
   `send` with a per-lane address vector inside the loop; **no `mov` between a
   load's destination and the `dpas` operand that reads it**, because the
   fragments must alias the load payloads directly; no scratch access anywhere.
3. A 128-GRF build of the same source is **not** timed. "Zero spill" alone is
   not clearance: the earlier kernel's 128-GRF binary also had zero spill and a
   degenerate schedule. The `dpas`-per-k-tile and no-`mov` counts are the
   schedule check.

### The prediction, written before the measurement

Starting from the reference implementation's measured **153.69 TFLOP/s** at
gate‖up M=2048, which is 83.8% of the 183.5 peak. Our design issues the same
messages in the same counts, the same `dpas` in the same order, the same
224-GRF budget, the same work-group and raster, and the same split barrier.
The structural differences and their direction:

| difference | direction | estimate |
|---|:---:|---:|
| the `dpas` and 2D loads reach the compiler as **intrinsics** rather than as inline assembly blocks. It can schedule and macro-fuse intrinsics it understands; inline assembly is opaque to its higher passes but scheduled identically downstream. Could go either way | either | -3 to +3% |
| the 2D descriptor payload is an argument of every builtin call, where the reference keeps one live payload and increments its coordinate. If the compiler rebuilds payloads per iteration that is ALU work beside the `dpas`; if it fails to hoist the uniform parts it is worse | worse | -2 to -6% |
| no reorder stage, since the reference's is the identity for bf16 | none | 0 |
| own raster equals the reference's effective raster at every production shape | none | 0 |
| first hand-written kernel: prologue, epilogue, coordinate arithmetic, loop overhead | worse | -1 to -3% |
| **net** | | **-5% ± 5%** |

> **Predicted: 146 TFLOP/s at gate‖up M=2048, band 135 to 155.** Above the 140
> go-bar with the band's lower edge above break-even. A reading under 135 means
> one of the named risks fired and the assembly will say which. A 256-GRF
> failure is not inside this band: that is the 33x class and is caught before
> timing.

### Risks, each with the measurement that detects it

| # | risk | what it would cost | detector |
|---|---|---|---|
| R1 | **Register allocation at 224 live GRF.** A private-memory fallback or a conservative schedule | 33x, or a 10 to 30% scheduling tax | metadata plus 32 `dpas` per k-tile with no interleaved `mov` or scratch `send` |
| R2 | **2D descriptor overhead in the k-loop**, per-call payload construction or a failure to prove the payload uniform | -2 to -6%, possibly more | non-`dpas`, non-`send` instruction count per k-tile body; the fix would be loop-carried coordinates |
| R3 | **Split barrier unavailable** on this driver | compile failure, then a plain barrier at an estimated -3 to -8% | immediate at compile; visible in the assembly as one barrier instead of arrive and wait |
| R4 | **Prefetch not landing**: wrong hints, wrong distance, L2 only | loads become latency-exposed; rate falls toward 100 to 120 | 2 prefetch-form loads per k-tile with cache hints; symptom is a rate under 135 with R1 to R3 clean |
| R5 | **Fragment order differs from the measured DPAS layout** | a correctness failure, not a rate; one compile-time index permutation | the bitwise check against the reference fails; an identity-matrix probe isolates which operand |
| R6 | **L2 behaviour at M=4096**, where A exceeds L2 | the reference shows it already, at 119.73 | the M sweep itself; the grouped raster is the named lever |
| R7 | **Occupancy**: a 512-work-item group at 256 GRF must schedule as one Xe-core's 32 threads | launch failure or two-wave serialisation | a launch error is immediate; a rate near half of prediction with clean assembly points here |
| R8 | **Alignment**: 64-byte base, 16-byte pitch | a fault at launch | all production pitches are multiples of 64 (derived); a probe assertion on the pointers |

R1 and R2 are the top two: both are properties of what the compiler makes of a
hand-written 224-GRF loop, both are invisible in the source and visible in the
assembly, and both are why the assembly gate runs before the clock.

---

## Part 3: results

Every timed cell below is **RECORD grade**: before each, the box was checked for
containers, DRM fd holders and running compiles, and all three were empty.
Protocol for every cell: 8 replays, first 3 discarded, median of the last 5, 4
enqueues per replay, one discarded warm-up, incompressible inputs.

### Our GEMM

Two build-time defects were found and fixed before the first correct build,
both one-line, neither an algorithm change:

1. The 2D block write builtin declares its last parameter as a `private`
   pointer. Passing an unqualified cast let the frontend infer the generic
   address space, which segfaulted the backend with no diagnostic beyond a
   frontend error one line above the crash. Fix: qualify the cast `__private`.
2. An unused helper and an aggregate initialiser with fewer braces than members
   tripped this project's own `-Wall -Wextra -Werror`.

**Build gate: PASS, first build.** Every check in the list above was met:
256 GRF, SIMD16, `has_dpas`, no spill or private memory, one barrier, 32
`dpas.8x8` per k-tile confirmed b-outer a-inner (four consecutive `dpas` share
the same operand register while the fourth operand walks four others), 3 data
loads plus 2 prefetch loads per k-tile (9 total in the file: 3 in the loop, 4 in
the prologue, 2 for the loop's own prefetch), 16 stores in the epilogue, zero
scatter or gather sends in the loop, **zero `mov` between a load's destination
and the `dpas` that reads it**, and a genuine split barrier rather than two
plain ones.

One naming deviation, not a substantive one: this driver's disassembler names
the messages `load_block2d` and `store_block2d` rather than with an `lsc_`
prefix. Same messages, a disassembly spelling.

**R2 is present exactly where predicted**: a block of `mov` instructions
building the next prefetch's 2D descriptor sits between the barrier wait and
the next iteration's loads, outside the 32-`dpas` block, which contains zero
`mov` and zero `send`.

**Correctness: bitwise identical to the reference in every cell, on the first
build.** Determinism: bitwise across two independent runs, every cell.

**Rate.** The control is the reference GEMM re-run beside every cell, same
harness, same inputs, so session variance cancels:

| shape | K x N | M | ours ms | ours TFLOP/s | ref ms | ref TFLOP/s | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate‖up | 5120x34816 | 1024 | 2.283 | 159.88 | 2.350 | 155.36 | 1.029 |
| **gate‖up** | 5120x34816 | **2048** | **4.665** | **156.53** | 5.498 | 132.81 | **1.179** |
| gate‖up | 5120x34816 | 4096 | 11.772 | 124.05 | 12.440 | 117.39 | 1.057 |
| down | 17408x5120 | 2048 | 2.214 | 164.89 | 2.314 | 157.76 | 1.045 |

Our kernel **beats its own same-harness control at every cell**, by 2.9% to
17.9%. The control itself reads lower this session than its historical record
(132.81 against 153.69 at gate‖up M=2048), which is session-to-session variance
and precisely why the control is re-run beside every cell rather than quoted.

**Against the prediction: 156.53 against 146 predicted, band 135 to 155.** At
the top of the band, 1.0% above its upper edge. `down` at M=2048: 164.89
against 156 predicted. M=1024: 159.88 against 152. M=4096: 124.05 against 114,
so the L2 risk at M=4096 cost **less** than predicted, not more.

### The slab interleave

The same GEMM, plus a second entry point that takes a column window, so the
dequant and the GEMM are both raw Level Zero launches on the *same* in-order
list and the inner loop calls them back to back **with no wait between them at
all**, only one trailing wait after the whole battery. The two entry points
live in the same source file, and the dumped assemblies differ only in an
embedded build hash, so the measurement above stands unchanged.

**Correctness: bitwise identical**, windowed against full width over the same
4,194,304-element column window.

**Handoff verified, not assumed.** Device timestamps, last slab dequant's
kernel end to the following GEMM's kernel start on the same in-order list:
**1.667 µs**, against a bar of 5 µs and against the **22.35 µs host wait** it
replaces. The question was the device-side gap, not merely whether a `wait()`
appears in the source.

Controls, this probe's own harness:

| control | ms | note |
|---|---:|---|
| C1, full dequant | 1.546 | |
| C2, GEMM full width | 4.729 | 154.40 TFLOP/s |
| C3, two-pass with a host wait between | 6.294 | sum model 6.275, +0.29% |

The lever, with zero host waits within a pair:

| slab width | slabs | ms | overhead over C2 | overhead / C1 | chunk-equivalent | **saved** |
|---:|---:|---:|---:|---:|---:|---:|
| **1024** | 34 | 5.618 | 0.889 | 0.5750 | 120.82 | **+89.29 ms** |
| 2048 | 17 | 5.974 | 1.245 | 0.8053 | 169.21 | **+40.91 ms** |

The ratio form is used so that any harness-level GEMM difference cancels out.

**The payoff lives at the narrower slab**, exactly as the L2 residency argument
in Part 2 predicted: a 1024-column slab of B is about 10.5 MB, which stays
resident, and a 2048-column slab is not.

### Both together

The two effects are additive and were kept separable by design: both the
two-pass control and the interleaved lever already run our GEMM, so the
+89.29 ms is the interleave effect **alone**, on top of the GEMM's own credit.

```
GEMM effect (measured rate 156.53 against a 150 TFLOP/s reference point):
  656.3 x (150 / 156.53 - 1) = -27.4 ms/chunk
Interleave effect (measured, 1024-column slabs): -89.29 ms/chunk-equivalent
Combined: -116.7 ms
New chunk: 1456 - 116.7 = 1339.3 ms -> 1529 t/s
```

That was **77.5%** of the vLLM figure in use at the time, against 71.3% before.
It does not close the gap by itself, and the arithmetic said so before the
numbers existed: the remaining roughly 301 ms per chunk was the delta net and
the in-kernel int4 dequant, and both of those were separately measured here and
found not to support the path.

The direction was specced and built. The production rows are in
[BENCHMARKS.md](BENCHMARKS.md).

### oneDNN's int4 matmul

A probe calling oneDNN's C++ primitive API directly, over our own prefill
context's queue, with exactly the attributes vLLM's own kernel sets. No vLLM
machinery, no primitive cache, no torch linked against it.

| shape | K x N | M | ms | TFLOP/s |
|---|---:|---:|---:|---:|
| qkv‖z | 5120x16384 | 1024 | 1.567 | 109.66 |
| qkv‖z | 5120x16384 | 2048 | 3.327 | 103.28 |
| qkv‖z | 5120x16384 | 4096 | 6.470 | 106.22 |
| out proj | 6144x5120 | 1024 | 0.571 | 112.85 |
| out proj | 6144x5120 | 2048 | 1.182 | 109.02 |
| out proj | 6144x5120 | 4096 | 2.416 | 106.65 |
| **gate‖up** | 5120x34816 | 1024 | 3.455 | 105.65 |
| **gate‖up** | 5120x34816 | **2048** | **7.429** | **98.28** |
| gate‖up | 5120x34816 | 4096 | 16.636 | 87.78 |
| down | 17408x5120 | 1024 | 1.292 | 141.26 |
| down | 17408x5120 | 2048 | 2.829 | 129.06 |
| down | 17408x5120 | 4096 | 5.020 | 145.46 |
| q‖k‖v | 5120x14336 | 1024 | 1.360 | 110.52 |
| q‖k‖v | 5120x14336 | 2048 | 2.840 | 105.86 |
| q‖k‖v | 5120x14336 | 4096 | 5.818 | 103.35 |

`down` at 141 to 145 and `gate‖up` at 88 to 106 bracket the central estimate on
opposite sides, which is consistent with a real per-shape effect, the
weight-read to compute ratio, rather than a single systematic miscalibration.

**What it sets.** The break-even an in-kernel int4 GEMM of ours would need to
clear, to beat the two-pass dequant plus bf16 GEMM it would replace, is
**115.7 TFLOP/s** (derived: 99.64 TFLOP of chunk work over 0.8613 s). oneDNN's
measured 98.28 at the shape that dominates is **below that break-even**. The
vendor's own fused int4 primitive, on this box, does not clear the bar a
from-scratch in-kernel int4 GEMM of ours would have to. That reverses the
expectation going in: at the measured rate the four int4 linears would cost
*more* through oneDNN than through the two-pass path.

**Correctness: failed.** Identity extraction (feed an identity matrix, read the
weights back) against one real production matrix, layer 0's fused input
projection, K=5120 and N=16384, K-contiguous repacked, against our own dequant
oracle transcribed in C++: **33,198,358 of 83,886,080 weights (39.6%) differ**.
The known sign-of-zero class accounts for 5,372,359 of them (6.4%), which
matches the 6.642% reference almost exactly, so that part of the bar holds.

Every sampled mismatch is a **1-ULP bf16 neighbour**. For instance, at the very
first element, the oracle gives `0xbcac` and the device `0xbcab`, adjacent bf16
values. A wrong-group or wrong-column bug would produce large, structurally
patterned errors, not a uniform one-bit spread starting at element zero.

One diagnostic run to root-cause rather than to tune: re-running with strict
fp32 math in place of the production bf16 fpmath mode gives an **identical
mismatch count**, ruling out "oneDNN was permitted to drop precision". The
remaining consistent explanation is that the JIT generator's in-kernel dequant
computes `q*scale - zp*scale`, two independently rounded multiplies and a
subtract, rather than the oracle's `(q - zp)*scale`, one exact integer subtract
and one rounded multiply. Mathematically identical over the reals, a different
rounding path over bf16, and exactly the signature measured. That is the
vendor's own numerics from a different kernel generator, reported as a real
difference rather than as a probe defect.

A separate investigation later found the same generator **truncating** rather
than rounding to nearest on a different code path; see
[prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md).

### Intel's chunked delta net kernel

The probe calls the torch-free kernel launcher template directly, never
executing the file's torch-using outer wrapper.

**Two environment shims, both probe-side, neither touching vendor code.**

1. A minimal torch and ATen stand-in, just complete enough that the unused
   wrapper still typechecks. One bug found and fixed: the wrapper calls a
   `dtype()` helper on something that is already an options object, which real
   ATen has a distinct overload for.
2. **An unanticipated second shim, found only by attempting the build.** The
   vendor's own GEMM header declares a barrier scope constant and then calls
   the barrier functions unqualified through a `using namespace`. On this
   toolchain and our pinned reference library, the constant's type at those
   call sites is `int` rather than the scoped enum, and it **fails to compile
   as shipped**. Confirmed by compiling the unmodified vendor file alone with
   the project's exact include paths, independent of torch or this probe's
   code. Root cause not fully isolated: a hand-retyped copy of the same
   function, includes, namespace and template shape did not reproduce it, so
   something about the real complete file matters. The effect is exact and
   reproducible. Fixed probe-side with two `int`-argument overloads that
   forward to the real enum-typed functions, value-preserving, so no vendor
   tree was edited.

**One real numerics defect found and fixed the hard way.** The first successful
run produced a finite but astronomical state, max relative error 5.137e24. Root
cause: the probe fed the kernel **raw** post-convolution Q and K, reasoning from
three comments inside the vendor file that read as if the kernel normalizes
internally. That reasoning was wrong. Grepping the entire 1634-line file for
any sum-of-squares computation finds **nothing**: the comments document an
input precondition, they do not describe code. Fix: L2-normalize Q and K per
position per head before upload, replicating the CPU reference's exact formula.
One defect, one fix. It took the state from non-finite to finite.

**Numerics after the fix: still far outside a defensible band.** State against
the CPU reference over all 2048 positions: **max relative error 5.187e1, mean
relative error 9.604e-1**. That is not a reduction-order band; a mean relative
error near 1.0 means the two computations are not computing the same values
under this fixture.

**Root cause not found within this probe's scope.** Candidates named, none
confirmed: a residual sign or ordering convention in the beta or gate mapping
between our packed slots and the vendor's separate per-head tensors; a scale
convention on `v` or on the state update that the probe assumed rather than
verified against the roughly 1500 remaining lines of kernel body; or a
state-layout transpose beyond the one already applied. Reported as the finding,
not tuned, not re-fit.

**Time: 0.795 ms per chunk** at chunk width 2048 with 48 value heads, against
our own measured 303 ms per chunk for the same work. That is a **382x ratio**,
and the numerics finding means it **cannot be reported with confidence as an
apples-to-apples time**: if the unexplained divergence reflects the kernel
taking a data-dependent short circuit this fixture triggers, the time is an
artifact of that rather than of the algorithm. The measurement is recorded
exactly as taken, and it is reproducible at 0.794 and 0.795 ms across two runs
before and after the normalisation fix, which is consistent with fixed control
flow and argues mildly against a short circuit. It is **not** adopted as a
price.

The honest summary: the build and the torch-free shims succeeded and are a
real, reusable result. The numerics, and therefore the timing's validity, did
not clear this probe's own bar.

Separate, later work repaired the probe's input and reset contract and found
that the vendor kernel of the day also **fails a determinism check**, with
differing state bytes across independent repetitions of the same call, while a
newer revision passes. That is in
[prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md).

---

## Part 4: which oneDNN was actually measured

oneDNN's 98.28 TFLOP/s implies a chunk GEMM time of **1013.8 ms** (derived:
99.64 / 98.28), which would leave only **24.2 ms per chunk for everything
else**, the delta net, attention, small operations and HTTP, if vLLM really ran
at that rate end to end. That is far below the 305 ms or more those terms
plainly need. Something did not reconcile, so it was chased.

**1. It is the real JIT, not a fallback.** Printing the chosen implementation
at every primitive creation: all 16 primitives, across five shapes and three M
values plus the correctness cell, dispatched to the real JIT GEMM generator. A
reference or unresolved path would read differently. Verbose exec lines confirm
the same at execution, and a second diagnostic run confirms creation-time
dispatch matches, including the nested primitive the matmul delegates to
internally. Re-measured rate this session: 97.00 TFLOP/s, within the roughly
1.3% run-to-run spread.

**2. Configuration parity with vLLM: exact, field by field.** Joint dtype,
source dimensions and strides, weight dimensions and strides, destination,
scale mask and group and dtype, zero-point mask and dtype, fpmath mode,
scratchpad mode, bias, and the engine and stream construction pattern all
match. Primitive caching differs and is not rate-affecting once warm, since
both sides amortize the JIT compile before the timed region.

**3. One field does not match: the oneDNN build itself.** The probe links the
box's packaged oneAPI oneDNN, **v3.11.4**. vLLM's kernel package does not use
that installation at all: its CMake unconditionally fetches its own oneDNN from
source at a pinned tag, **v3.13**, and builds it statically with the SYCL GPU
runtime.

Between those two tags, on the same repository, **37 commits touch the JIT GEMM
generator**, of which **5 explicitly target the Xe int4 GEMM path** and are
present in v3.13 and absent from v3.11.4:

- upconvert int4 to int8 for mixed DPAS
- re-use registers in the int4 downconvert copy plan
- use block loads instead of VNNI loads for int4
- fix up int4 copy plan handling
- a bf16-with-int4-zero-points fix

That is a real, dated, on-topic version gap in exactly the kernel family under
test. It is **not** proof of a specific speedup magnitude: no build of v3.13
was measured.

A read-only search of the box found no usable matching build: the fetched
source tree present was at a stale tag, its build directory was never
configured, and the only other oneDNN on the machine was a CPU-only distribution
package with no SYCL GPU runtime.

**Verdict.** "An int4 GEMM with no dequant pass cannot pay on this box" is **not
established; it remains open.** What is settled: 98.28 TFLOP/s is a genuine JIT
rate, not a probe defect, from a configuration identical in every field to
vLLM's own call. What is not settled: whether vLLM's actual pinned build reaches
a materially different rate on the same hardware and the same primitive. The
figure should be read as "this oneAPI package's JIT rate for this primitive on
this box", not as "oneDNN's rate for vLLM's configuration". Those are not shown
to be the same number.

---

## Part 5: promotion to production

The probe kernel became `src/kernels/prefill/pf_gemm.cl`. The leading
dimensions and a batch stride became runtime arguments, and a second build
variant reads B stored `[N][K]` in place, which is what the attention KV cache
needs.

**One build-time defect, fixed once.** The transposed 2D read builtin at the
16-dword row width is **undeclared on this driver**. Only the 8-dword row is
exposed, and the compiler said so by name:

```
did you mean intel_sub_group_2d_block_read_transpose_32b_16r8x1c?
```

Fix: each n-atom's 16-dword B fragment becomes two 8-dword transposed loads,
one per k half-tile. Same coverage, same lane, same k order, doubling that
variant's B-load message count from 4 to 8. No other change.

**Gate: PASS, both variants.**

```
== T0 (B as [K][N])
      barrier_count: 1   grf_count: 256   has_dpas: true   simd_size: 16
      dpas 32  load2d 9   prefetch 6  store2d 16  gtwy 2  bar 1
== T1 (B as [N][K], transposed reads)
      barrier_count: 1   grf_count: 256   has_dpas: true   simd_size: 16
      dpas 32  load2d 15  prefetch 6  store2d 16  gtwy 2  bar 1
```

Spill and private size are absent from both. The `scratch` declaration present
in both is the universal per-kernel ABI register declaration every kernel on
this compiler carries, verbatim and at the same line in both files, not a real
scratch access.

The first variant's 9 loads (3 data plus 6 prefetch) match the probe's gate
exactly. The second's 15 (9 data plus 6 prefetch) is higher than the 11
predicted, because of the defect above: the two extra data loads anticipated
become six once each of the four transposed reads splits into two. Everything
else matches the probe's gate exactly in both variants.

---

## What this investigation decided

**Build our own.** The measured rate beat the library's at every shape, on the
first correct build, bitwise identical to it, with the assembly proving the
kernel is DPAS bound and not spilling. Owning the kernel then made the
interleave possible, which removed 512 host waits per chunk and is worth more
than the GEMM speedup itself.

**Do not build an in-kernel int4 GEMM against the vendor's measured rate.** It
is below the break-even, though the version gap leaves the question open rather
than closed.

**Do not treat the vendor delta net time as a target.** Its numerics did not
clear the bar, and a time from a kernel that may not be computing the right
thing is not a price.

The delta net was eventually addressed our own way instead, by making our scan
carry two BF16 limbs on a DPAS path, which is worth 2.33x on that row. See
[prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md).
