# Probe - int4-g64 through the BMG mixed-input mainloop (ruling A21)

grade: iterate (card 1, `ZE_AFFINITY_MASK=1`, card 0 may be held)

Brief: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/probe-int4-mixed-brief.md`.
Question: can the **existing layout-0 int4-g64 weights** be fed to
`MainloopIntelXeXMX16MixedPrecision` (bf16 A × int4 B, group 64, zero = 8,
scales N-major) **without a per-chunk bridge**, and at what TFLOP/s? This
decides whether the measured 210.116 ms/chunk dequant scratch disappears with
no format change, or needs a repack.

---

## PRE-REGISTRATION (written 2026-09-05, committed BEFORE any measurement)

Everything in this section is a prediction or a source reading taken from the
Mac checkout of `sycl-tla` at pin
`91e5bd735517d8e79591b41e0d0cd37a7bacdca7` (ruling A11) and from this repo's
own loader/oracle. Nothing below has been run on the box.

### 0. Source readings the predictions rest on (verified, not measured)

**S1 - the B tensor the mainloop builds is a rank-3 FLAT-strided tensor.**
`xe_mma_mixed_input.hpp:326-332`:

```cpp
auto mB_nkl = make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));
```

`args.dB` is `StrideB` = `TagToStrideB_t<LayoutB>`, and
`cutlass/detail/layout.hpp:79-89` fixes the only two options over modes
`(N, K, L)`:

| `LayoutB` | `StrideB` | 4-bit element index of (n, k) | memory shape |
|---|---|---|---|
| `RowMajor` | `Stride<_1, int64_t, int64_t>` | `p = n + k·ldb`, `ldb = N` | `[K][N]`, N contiguous |
| `ColumnMajor` | `Stride<int64_t, _1, int64_t>` | `p = n·ldb + k`, `ldb = K` | `[N][K]`, K contiguous |

For 4-bit `ElementB` the pointer becomes `cute::subbyte_iterator<const
ElementB>` (`:318-324`), i.e. element `p` is bits `[4p, 4p+4)` of the byte
array - **LSB nibble first**, the same convention `tools/oracle/dequant.py`
and `gemv.cl:66-73` use.

**S2 - layout 0's 4-bit address function.** `src/loader/quant.cc:120-142` and
`docs/02`: `qweight` is `int32 [K/8][N]` row-major; nibble `i` of word
`(r, n)` is `k = 8r + i`, low nibble first. Therefore the flat 4-bit index of
weight (k, n) in the checkpoint bytes is

```
p0(k, n) = 8·(⌊k/8⌋·N + n) + (k mod 8)
         = ⌊k/8⌋·8N  +  8n  +  (k mod 8)
```

**S3 - the scale tensor the mainloop builds is layout 0's scales exactly.**
`:341-343` builds `mScale` with shape `(N, ceil(K/g), L)` and stride
`args.dS`; the default `NonVoidStrideScale` is `Stride<_1, int64_t,
int64_t>`, so scale (n, g) sits at `g·N + n` - which **is** `scales f16
[K/64][N]` row-major, byte for byte. Scale bridge = 0.

**S4 - the 4-bit in-register transform is positionally an identity unpack.**
`transform_quant` (`:432-521`) reinterprets the 4-bit B fragment as
`unsigned short` words (`scalar = 16/4 = 4` nibbles per word) and writes
`out[i + vec_size·s + (loop_cnt)·n]` from nibble `i` of word `s` of column
`n`. With `vec_size == scalar == 4`, source 4-bit position and destination
element index are the same integer. The transform therefore adds no
permutation of its own; the **copy atom** decides the layout.
Its one shape constraint is `static_assert((scalar % N) == 0)` with
`N = size<1>(mma_B)` - the fragment's N-iteration count - so N ∈ {1, 2, 4}.

**S5 - Intel's own u4 configuration.**
`examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_f16.cpp:580-599` runs
the narrow operand as **B with `LayoutB = ColumnMajor`** and
`GmemTiledCopyB = XE_2D_U4x32x16_LD_T` - a *transposing* 4-bit block load.
The two non-transposing 4-bit atoms that exist (`XE_2D_U4x32x64_LD_N`,
`XE_2D_U4x16x64_LD_N`, `include/cute/arch/copy_xe_legacy_U4.hpp:108,158`)
carry Intel's own comment *"FIXME: the performance of shuffle algorithm here
is too bad"*. The file header also states the mainloop's two limitations:
*"group must be multiple of k-block size"* and *"scales & zeros must be
MN-major"*.

**S6 - the same tile shape this project already measured is the s8 example's
tile.** `02_bmg_gemm_bf16_s8_bf16.cpp:503-508` uses
`TileShape = Shape<_256,_256,_32>` with subgroup layout
`Shape<_8,_4,_1>, Stride<_4,_1,_0>` - identical to P2's and to
`src/sycl/xe_gemm_config.h`. At that shape `SG_N = 64`, `SG_K = 32`, and
`mma_B`'s N-iteration count is 4, satisfying S4's `scalar % N == 0`.

**S7 - the arithmetic is fp32 with one bf16 round, matching the oracle.**
`cutlass::half_t`'s `half_t(int)` constructor is **explicit**
(`include/cutlass/half.h:408`) while `operator float()` is **implicit**
(`:432`). So in `transform_quant` an `int`-valued nibble times a `half_t`
scale resolves as `float × float → float`, and the single
`static_cast<DstType>` is the only rounding - exactly
`dequant.py`'s `(q − 8) · float(scale)` in fp32 followed by one cast to
bf16.

**S8 - signed int4 makes the zero point free.** `cutlass::int4b_t` is
`integer_subbyte<4, true>`; its `operator int()` sign-extends
(`integer_subbyte.h:138-143`) and its explicit `int` constructor masks to 4
bits (`:97-99`). For every `q ∈ [0,16)`, `sign_extend₄(q ⊕ 8) == q − 8`.
This is the identity `gemv.cl:53` already relies on (`word ^ 0x88888888`)
and that `gemv_test` proves bit-identical. So a repack that stores `q ⊕ 8`
lets `ConvertAndScale` (scale only, **no zero tensor**) compute the GPTQ v1
zero-8 dequant exactly.

### 1. Layout fit - PREDICTION

**Direct consumption is impossible.** `p0(k, n)` (S2) is affine in `n` but
**piecewise in `k`**: `k mod 8` carries coefficient 1 and `⌊k/8⌋` carries
coefficient `8N`. No rank-3 flat `StrideB` (S1) can express it, and
`make_layout(make_shape(N,K,L), dB)` admits nothing else. Predicted verdict:
**a one-time repack is required.**

**Predicted repack target**: `B4[n][k]` 4-bit, **K contiguous**, i.e.
`LayoutB = ColumnMajor`, `ldb = K`, element (n,k) at 4-bit index `n·K + k`
(byte `(n·K + k)/2`, low nibble when `k` is even) - Intel's own u4
orientation (S5), with the nibble value stored as `q ⊕ 8` (S8).

**Predicted classification: a NEW layout, i.e. a second copy.** Decode's
`gemv` reads word `(r, n)` with lane = `n` (`gemv.cl:84-89`), so the 16 lanes
of a subgroup read 16 *consecutive u32* - fully coalesced. Under `[N][K]`
those same 16 lanes would be `K/8` u32 apart. Decode cannot adopt this layout
without a different GEMV geometry, so A21's "no decode change, Task 4's
retune untouched" survives only by paying for a second copy.

**Predicted price of the second copy** (derived from `model::Qwen35`'s table,
K·N/2 bytes each):

| group | matrices | weights | bytes |
|---|---:|---:|---:|
| 48 GDN layers (qkv‖z, out_proj, gate‖up, down) | 192 | 18,371,051,520 | 9,185,525,760 |
| 16 FA layers (q‖k‖v, o_proj, gate‖up, down) | 64 | 5,955,911,680 | 2,977,955,840 |
| **256 layer matrices** | **256** | **24,326,963,200** | **12,163,481,600** |
| int4 `lm_head` (RTN checkpoint) | 1 | 1,271,398,400 | 635,699,200 |
| **total with lm_head** | **257** | **25,598,361,600** | **12,799,180,800** |

12,163,481,600 B = **11.33 GiB** for the 256 layer matrices - A21's "+12.16
GB" reproduced exactly - or **11.92 GiB** including `lm_head`. Against plan
6f's stated 13.79 GiB headroom at max_len 16384 that leaves **2.46 GiB**
(layers only) or **1.87 GiB** (with lm_head), before the 356.5 MB dequant
scratch is handed back. Scales are shared with decode (S3) and cost nothing;
the `ConvertAndScale` route costs no zero tensor either (S8).

**Predicted alternative, priced not chosen**: a per-chunk repack instead of a
second copy moves `K·N/2` bytes in and `K·N/2` out per matrix per chunk =
24.33 GB/chunk of extra traffic versus the dequant's 60.8 GB
(12.16 read + 48.65 bf16 write). At the dequant's own measured efficiency
(210.116 ms for 60.8 GB = 289 GB/s) that is **≈84 ms/chunk** - better than
210 but not zero. Recorded as the fallback if the resident bytes are refused.

### 2. Correctness - PREDICTION

**Bit-exact.** By S4 + S7 + S8 the mainloop's in-register dequant produces
exactly `f32_to_bf16((q − 8) · f16_to_f32(scale))`, which is
`tools/oracle/dequant.py:44-45` verbatim. Predicted: an identity-A extraction
(`A = I`, so `C[k][n] = float(bf16 W[k][n])` exactly, every DPAS term either
`0·w` or `1·w`) reproduces the oracle's bf16 word for **every one of the
83,886,080 weights** of the real fused `layers.0.linear_attn.in_proj_qkv ‖
in_proj_z` matrix (K = 5120, N = 16384) from
`/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`, with **zero**
mismatching words.

**GEMM at (M=64, K=5120, N=16384)** against a CPU reference that dequantises
with the oracle's arithmetic and accumulates in fp64: predicted inside P2's
structural tolerance `5e-3·max|ref| + 1e-4` (`gemm_test` does not exist yet -
S1 has not landed - so P2's published bar is the precedent used).

**Two runs bitwise identical; workspace 0 B; no split-K atomics** - the
scheduler is the same data-parallel `PersistentScheduler` P2 named, and
`xe_gemm.hpp:82-87` still refuses any other.

### 3. Rate - PREDICTION

**gate‖up (K=5120, N=34816) at M=2048: ≥ 128 TFLOP/s** (the brief's bar,
0.85 × P2's corrected 256-GRF bf16 row of 150.19). Also measured: M ∈ {512,
1024, 4096} and the other five production shapes, as a matrix directly
comparable to P2's.

Secondary rate prediction (mechanism, not a bar): the mixed row will be
**below** the bf16 row at the same shape at large M and **above** it at small
M - the in-register unpack costs ALU that scales with the GEMM's FLOPs, while
B's DRAM traffic is 4× smaller, which is worth most where the GEMM is weight-
bandwidth-bound.

### 4. Composed consequence - DERIVED, written before measuring

`progress.md`'s last composition: **1126.958-1132.458 ms/chunk** at C=2048
(→ 1808.5-1817.3 t/s), of which dequant is **210.116 ms** and GEMM is
**60.2% = 678.4 ms**.

With bridge = 0 (a one-time repack) and the GEMM term unchanged:

```
916.842-922.342 ms/chunk  →  2048 / 0.922342 … 2048 / 0.916842
                          =  2220.5 - 2233.8 t/s
```

i.e. **112.5-113.2% of vLLM's 1973** - the same band plan 6f priced its
MXFP4 second-copy route at (2220-2234), reached without a new checkpoint, a
re-anchored oracle or a decode change. The brief's stated 2190-2230 is the
same arithmetic rounded; this document's 2220.5-2233.8 is the one scored.

Sensitivity, pre-registered so the post-measurement number cannot be tuned:
each **1% the mixed GEMM is slower than the bf16 matrix** adds 6.78 ms and
costs **≈16 t/s**. The bar 1973 t/s is reached as long as the mixed GEMM is
within **+13.5%** of the bf16 GEMM time (i.e. ≥ 88.1% of its rate).

### 5. Named risks and their pre-planned single responses

1. **`XE_2D_U4x32x16_LD_T` may not instantiate at TileShape<256,256,32> /
   8×4×1.** Intel's u4 example uses `<16,64,64>` / `1×2×1`. If the build
   fails there, the one fix is to fall back to the example's tile and report
   the rate at that tile as a deviation - not to sweep tiles.
2. **IGC.** The u4 example's header says
   `export IGC_allowDecompose2DBlockFuncs=0` is *"currently necessary"*. Our
   build is AOT (`-fsycl-targets=spir64_gen`), and P2's reusable lesson is
   that a runtime IGC variable does not reach an AOT device compile. This
   probe therefore sets `IGC_allowDecompose2DBlockFuncs=0` in the **build**
   environment as well as the run environment, and records it as a
   measurement condition.
3. **256-GRF.** `B70_SYCL_AOT_256_GRF` defaults ON and reaches this target
   through the same `-Xsycl-target-backend "-options
   -cl-intel-256-GRF-per-thread"` P2's correction added. It is asserted from
   the configure log, not assumed.
4. **Disk.** The box root is at **1.4 GB free / 100% used** at the start of
   this probe (the operator's 14 GB `qwen38-27b-mxfp4a16` checkpoint landed
   since L1-core's "15 GB free"). Nothing is deleted. `df` is checked before
   and after every build; any disk error stops the probe and is reported.

### 6. What is measured, exactly

Two `GemmUniversal` instantiations over `MainloopIntelXeXMX16MixedPrecision<3>`,
`TileShape<256,256,32>`, `TiledMma XE_DPAS_TT<8,float,bf16>` subgroup layout
8×4×1, A = bf16 RowMajor with `XE_2D_U16x32x32_LD_V`, B = 4-bit ColumnMajor
with `XE_2D_U4x32x16_LD_T`, `ElementScale = half_t`,
`StrideScale = Stride<_1,int64_t,int64_t>`, group 64, epilogue
`IntelXeGeneric` with `ElementC = void` (P2's corrected epilogue), scheduler
omitted (data-parallel `PersistentScheduler`):

- **`GemmScale`** - `ElementB = cutlass::int4b_t`, `ConvertAndScale`, weights
  repacked with the `⊕ 8` of S8. No zero tensor.
- **`GemmScaleZero`** - `ElementB = cutlass::uint4b_t`, `ConvertAndScaleWithZeroPoint`,
  `ElementZero = int8_t`, zero tensor `[K/64][N]` of the constant 8. The
  stock-semantics control; prices the zero tensor (`K·N/64` B = 1/32 of the
  weight bytes = 380.1 MB over the 256 layer matrices).

Method (identical to P2's, so the matrices compose): 8 replays, first 3
discarded, median of the last 5, 4 enqueues per replay,
`q.wait_and_throw()` after every replay; `ZE_AFFINITY_MASK=1`; a discarded
M=512 ramp control before each shape's recorded cells.

---

## RESULTS

*(everything below this line was written after the measurements it reports)*
