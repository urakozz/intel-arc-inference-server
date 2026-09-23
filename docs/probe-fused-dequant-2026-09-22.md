# Fusing the int4 dequant into the GEMM: rejected (2026-09-22)

**Verdict: REJECTED.** Both candidate kernels are **bitwise identical** to the
two-pass slab path over all 71,303,168 fp32 outputs, and both are **far below**
the break-even fixed before the probe was written. The faster of the two
reaches **100.37 TFLOP/s** against a break-even of **122.9**. The slab path
stands, unchanged. No tuning pass was run.

The interesting part is why. The dequant pass this was meant to delete already
runs at about **90% of hardware memory bandwidth**, so there is very little in
it to reclaim, and doing the same arithmetic inside the GEMM mainloop costs
**3.3x** the term it removes even when issued exactly once per weight.

Every number below is **measured** unless marked **derived**.

## 1. The break-even, stated before the numbers

Derived before measuring and unchanged since:

| outcome | fused in-kernel rate | linear-path time | change |
|---|---:|---:|---:|
| break-even | **122.9 TFLOP/s** | 1618.8 ms | 0 |
| target | 140 TFLOP/s or better | 1421 ms or less | -198 ms |
| ceiling, dequant free and GEMM unchanged | 155.7 TFLOP/s | 1278 ms | -341 ms |

122.9 TFLOP/s is 199 TFLOP of per-request linear work over the 1618.8 ms of
slab dequant plus slab GEMM measured in situ
([prefill-parity-2026-09-20.md](prefill-parity-2026-09-20.md), "The linear
path, itemised").

## 2. What was run

A new probe host and probe kernel, plus three build stanzas. **No production
file was edited.** The production GEMM and slab dequant sources are untouched,
and the control runs **those two production binaries themselves**, which is
what makes the byte comparison a gate rather than a paraphrase.

One shape: **gate‖up, K = 5120, N = 34816, M = 2048, layout 0**, which is
**730.1 GFLOP per call** (derived: 2 x 2048 x 5120 x 34816). Three measurements
in one process, same buffers, same device:

- **control**: slab dequant then GEMM per 1024-column slab, launch for launch
  what the production linear path does today. 34 slabs, 68 launches.
- **V1, registers only**: the production GEMM with **only the B path**
  replaced. Each sub-group reads the four qweight rows of its own 64 columns
  (four 32-bit block messages, 1 KB, against the 4 KB of bf16 the two VNNI
  transform loads read today) and dequantises straight into its B fragment
  registers. One launch, no slab, no SLM, no extra barrier.
- **V2, SLM staged, double buffered**: the work-group dequantises each k-tile's
  32k by 256n block of B **once** into a 32 KB double-buffered SLM image laid
  out exactly as the block-read builtin returns it, while the `dpas` block
  consumes the previous tile from the other buffer. One split barrier per
  k-tile. One launch.

Everything else in both variants is the production GEMM transcribed verbatim:
the 256x256x32 tile, the 512-work-item group, the 8x4 sub-group raster, the A
2D block load and its 2-deep prefetch, the 16 fp32 accumulator vectors, the 32
`dpas.8x8` per k-tile in b-outer a-inner order, the split barrier, and the
16-store epilogue.

**Environment.** One Arc Pro B70, the per-kernel card. ocloc and compute
runtime 26.35.39758.10, IGC 2.41.5. Both variants compiled ahead of time at 256
GRF per thread.

**Grade: iterate**, not record. At dispatch, load average 0.22 and zero
containers, but one DRM holder was present: a desktop terminal service that
holds render-node file descriptors and submits nothing. It was not killed. The
verdict does not turn on the grade: the three rows share one process and one
harness, two independent runs agree to within 1.9% on the worst row, and the
better variant misses break-even by 18%.

## 3. The three rates

Level Zero kernel timestamps, best of 5 after one discarded warm-up. The kernel
sum is the sum of per-launch device durations, the same quantity the linear
path attribution uses and therefore the one the break-even is derived from. The
span is first kernel start to last kernel end on the in-order list; for a
single-launch variant the two coincide.

| variant | launches | kernel ms | span ms | TFLOP/s | vs break-even |
|---|---:|---:|---:|---:|---:|
| **control**, dequant + GEMM, 34 slabs | 68 | **5.341** | 5.429 | **136.72** | +13.8 |
| **V1**, registers only | 1 | **11.117** | 11.117 | **65.68** | **-57.2** |
| **V2**, SLM staged, double buffered | 1 | **7.274** | 7.274 | **100.37** | **-22.5** |

Control split: dequant **0.826 ms**, GEMM **4.515 ms**, 34 launches each. The
GEMM half alone is therefore **161.7 TFLOP/s** (derived: 730.1 / 4.515), which
is the figure quoted elsewhere as this kernel's in-situ rate.

Repeat run, same binaries, minutes apart: control 5.346 ms and 136.58 TFLOP/s,
V1 11.318 ms and 64.51, V2 7.346 ms and 99.39. The spread is under 1.9% on the
worst row and changes nothing.

**The subtraction that matters.** Against the control's GEMM half (4.515 ms),
V1 adds **+6.602 ms** and V2 adds **+2.759 ms**, to delete a dequant pass that
costs **0.826 ms**. Even V2, which issues the dequant ALU exactly **once** per
weight per work-group, the minimum this design can reach, pays **3.3x** the
term it exists to remove.

## 4. Equality

`memcmp` of each fused variant's output against the control's over the full
71,303,168 fp32 values, compared as raw 32-bit words so a sign-of-zero
difference cannot hide behind `==`:

| variant | verdict | first differing index |
|---|---|---|
| V1, registers only | **bitwise identical** | none |
| V2, SLM staged | **bitwise identical** | none |

This was the predicted outcome and it held on the first correct build of both
kernels. **The arithmetic is not the reason the probe fails.**

## 5. Compiler evidence

Shader dumps taken around the identical `ocloc` command line the build uses:
kernel metadata first, then the Xe2 assembly.

### 5.1 The gate, both variants

| check | required | V2, the faster | V1 |
|---|---|---|---|
| `simd_size` | 16 | **16** | **16** |
| `grf_count` | 256 | **256** | **256** |
| `has_dpas` | true | **true** | **true** |
| `spill_mem_size` / `private_size` | absent | **absent** | **absent** |
| `barrier_count` | 1 | **1** | **1** |
| `slm_size` | | **32768** (2 x 16 KB, as designed) | **absent** (0) |
| `dpas.8x8` per k-tile | 32, b-outer a-inner | **32** | **32** |
| `store_block2d` in epilogue | 16 | **16** | **16** |
| scatter or gather `send` in the loop | 0 | **0** | **0** |
| split barrier | arrive and wait | **present** | **present** |

Both pass. `scratch` appears in both assemblies only as the universal
per-kernel ABI declaration the production GEMM's gate recorded verbatim, not as
a scratch-surface access; the authoritative spill signals are absent from both
metadata blocks. **Spill and scratch bytes: zero in both.**

`dpas` structure, V2, the same b-outer a-inner macro the production kernel
emits, four consecutive `dpas` sharing one operand and fused as a chain:

```
dpas.8x8 (16|M0)  r133:f  r133:f  r4:bf  r38.0:bf  {Atomic,Compacted,$15.dst}
dpas.8x8 (16|M0)  r165:f  r165:f  r4:bf  r42.0:bf  {Atomic,Compacted}
dpas.8x8 (16|M0)  r197:f  r197:f  r4:bf  r46.0:bf  {Atomic,Compacted}
dpas.8x8 (16|M0)  r229:f  r229:f  r4:bf  r50.0:bf  {Atomic,Compacted}
```

2D block IO per k-tile. V1 issues **5 data loads** (1 A, 4 qweight) plus 2
prefetches, 11 block loads in the file. V2 issues **2 data loads** (1 A, 1
qweight for the tile it stages) plus 2 prefetches, **9** in the file, which is
the production GEMM's own 9, because V2's B path costs one 2-row message where
the production kernel spends two VNNI transform messages. V2 additionally
issues 16 SLM loads and 1 SLM store per k-tile: each of the eight block-read
calls lowers to two SLM messages.

### 5.2 Which instruction produces the bf16 word, and that it rounds to nearest

This has to be stated against the generated code, because the investigation of
a vendor int4 path found it **truncating** instead of rounding. It does not
happen here. **No hardware fp32 to bf16 conversion instruction appears in
either binary**: there is no `mov … :bf` fed by an `:f` source anywhere. The
word is built by the integer sequence the production dequant kernel specifies,
and the compiler emits it literally. The nine instructions below are
transcribed from V2's assembly; they are one weight's chain, and they are not
contiguous lines in the dump because the scheduler interleaves sixteen such
chains through each other:

```
shl  (16|M0)  r11.0<1>:d   r12.0<1;1,0>:d  16:w      // (u << (28 - 4i)), i = 3 here
asr  (16|M0)  r11.0<1>:d   r11.0<1;1,0>:d  28:w      // …) >> 28   -> qm8, sign-extended
mov  (16|M0)  r16.0<1>:f   r11.0<1;1,0>:d            // (float)qm8
mul  (16|M0)  r16.0<1>:f   r16.0<1;1,0>:f  r6.0:f    // * scale   (r6 = the f16 group scale, widened)
shr  (16|M0)  r5.0<1>:ud   r16.0<1;1,0>:ud 16:w      // u >> 16
and  (16|M0)  acc2.0<1>:d  r5.0<1;1,0>:d   1:w       // … & 1      -> the destination's low bit
add3 (16|M0)  r9.0<1>:d    acc2.0:d  r16.0:d  32767:w // u + (lsb + 0x7FFF)   <-- RNE, one instruction
shr  (16|M0)  r4.0<1>:ud   r9.0<1;1,0>:ud  16:w      // … >> 16    -> the bf16 word
bfn.(s0&s1|s2) (16|M0)  r24.0:ud  r11.0:ud  r70.5:ud  r4.0:ud   // (hi & 0xFFFF0000) | lo
```

`32767` is `0x7FFF`. The `add3` adds the odd-tie bit **and** the half-ulp bias
in one instruction before the shift, which is exactly
`rounding = ((u >> 16) & 1) + 0x7FFF; (u + rounding) >> 16`, round to nearest
**even**. A truncation would be the bare shift with no `add3`. The `bfn` pack
folds the high half into place, an exact rewrite on the same bits. The
`memcmp` in section 4 is the end-to-end confirmation of the same claim.

### 5.3 The instruction count that explains the result

The `add3` count per k-tile is one per weight dequantised, so it counts the
work directly:

| | `add3` per sub-group per k-tile | weights dequantised | redundancy |
|---|---:|---:|---|
| V1 | **128** | 4 n-atoms x 32 k = 128 | **8x**: the 8 m-subgroups sharing an n-column block each repeat the whole dequant |
| V2 | **16** | 2 qweight words x 8 nibbles | **1x**: the work-group's tile is dequantised exactly once |

V1's loop body carries roughly **905** ALU instructions against its 32
`dpas.8x8`, about **7 SIMD16 instructions per weight** (derived from the
per-opcode counts: 128 `add3`, 129 `asr`, 130 `mul`, 109 `shl`, 196 `shr`, 131
`and`, 66 `bfn`, 16 `xor`). V2 carries one eighth of that, plus 17 SLM messages
and a barrier the production kernel does not have to honour.

## 6. What the numbers mean

The fused path does remove the traffic it was designed to remove: in situ it
would read about 12.2 GB of int4 per chunk instead of writing and re-reading
48.7 GB of bf16. That saving is real and it is **not large enough**.

At this shape the two-pass dequant already runs at about **546 GB/s** (derived:
89.1 MB of qweight and 5.6 MB of scales read, 356.6 MB of bf16 written, in
0.826 ms), which is roughly **90% of the 608 GB/s hardware peak**. The round
trip is already close to bandwidth-optimal, so there is only 0.826 ms in it at
this shape, and moving the same arithmetic inside the mainloop costs 2.759 ms
even when it is issued exactly once per weight.

Put as a requirement: to reach 140 TFLOP/s the fused kernel may add at most
**0.700 ms** over the GEMM half (derived: 730.1 / 140 - 4.515). It must
therefore perform the dequant **more cheaply than a dedicated,
bandwidth-saturated pass does**, while competing for issue slots with the
`dpas` stream. V2's measured 2.759 ms is 3.9x away from that, with the
redundancy already at its floor.

Extrapolated to the whole linear path (derived, and conditional on this shape's
rate transferring to the other linear shapes, which this probe did not
measure): 199 TFLOP at V2's 100.37 TFLOP/s is **1983 ms** against today's
1618.8 ms, **+364 ms**, not -198. At V1's rate it is 3030 ms.

This is the outcome the probe was written to look for: *the dequant ALU cost
inside the mainloop exceeds the traffic it removes.* It is recorded, not tuned
away.

## 7. Consequences

- The production linear path, GEMM and slab dequant are unchanged and stay
  unchanged. Nothing is adopted.
- The expected saving from fusing dequant comes off the plan. Nothing else
  depended on it, so the cost is exactly its own estimate.
- The 7,616 launches per request the fused path would also have removed are not
  recovered by any other change, and are not, on their own, worth 364 ms.

**What this does NOT license.** It measured one shape, one tile (256x256x32)
and one layout. It does not rule on: a different tile whose m-subgroup count is
smaller, since V1's 8x redundancy is a property of the 8x4 raster rather than of
fusion; a pre-swizzled int4 layout that would cut the shift and shift-right
chain; or a mixed-precision DPAS path that changes the rounding point, which is
a different question and was asked separately in
[probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md). Each of those is a **new**
probe with a new break-even, not a tuning pass on this one.

## 8. Files

- `tools/probe/probe_fused_dequant_gemm.cc`, probe host
- `tools/probe/probe_fused_dequant_gemm.cl`, probe kernels
  (`-DFUSED=1` for V1, `-DFUSED=2` for V2, both compiled ahead of time at 256
  GRF)
- `tools/probe/CMakeLists.txt`, one appended block

```
ZE_AFFINITY_MASK=1 tools/box.sh run ./build/tools/probe/probe_fused_dequant_gemm
```

The probe prints its own verdict line and returns non-zero only on a `memcmp`
difference, which did not occur.
