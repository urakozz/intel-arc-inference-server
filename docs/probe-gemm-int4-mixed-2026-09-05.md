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

All values below are **measured, iterate grade** (card 1, `ZE_AFFINITY_MASK=1`,
card 0 may be held), sycl-tla `91e5bd7...-content-verified`, on the
configuration named in §6 with **one deviation, recorded in D1**.

## D - deviations from the pre-registration

**D1 - the MMA atom had to change; the tile did not.** The pre-registered
configuration used P2's `XE_DPAS_TT<8, float, bfloat16_t>`. It does not
compile with any 4-bit copy atom, and the compiler says why:

```
copy_traits_xe_legacy.hpp:530: static assertion failed due to requirement
  'get<0>(cute::tuple<C<1>,C<32>>{}) >= get<0>(cute::tuple<C<8>,C<2>>{}) ||
   get<1>(cute::tuple<C<1>,C<32>>{}) <= get<1>(cute::tuple<C<8>,C<2>>{})':
  It is not currently supported to have MMA atom be bigger than copy atom in
  one dimension and smaller in other dimension!
```

`CopyValsShapeRegs` = `(1, 32)` is `shape_div(reverse(BlockShape), (16,1))`, a
property of `XE_2D_U4x32x16_LD_T` **alone**. `MmaValsShapeRegs2d` = `(8, 2)` is
`reverse(get<0>(mma_B.shape()))`, a property of the MMA atom **alone**: the new
`XE_DPAS_TT` describes its per-thread B fragment as the rank-2 VNNI shape
`(2, 8)`. Neither depends on `TileShape` or the subgroup layout, so **no tile
change can repair that pairing** - the pre-registered response ("fall back to
the example's tile") would not have worked and was not taken.

The fix is one line and stays inside Intel's own configurations: the **legacy**
`XE_8x16x16_F32BF16BF16F32_TT` atom, which is what `02_bmg_gemm_bf16_s8_bf16.cpp:507`
uses **at exactly this TileShape<256,256,32> and 8x4x1 subgroup layout**. Its
`BLayout` is `Shape<_16,_16>, Stride<_1,_16>` - a flat 16-value per-thread
fragment - so `MmaValsShapeRegs2d` is `(1, 16)` and `1 >= 1` holds. It is the
same DPAS instruction; only the CuTe fragment description differs. **Everything
else in §6 is as pre-registered: tile 256x256x32, subgroup layout 8x4x1, 32
subgroups, epilogue `IntelXeGeneric` with `ElementC = void`, data-parallel
`PersistentScheduler`.**

**D2 - the identity-A extraction cannot carry a negative zero, and this is
reported as its own column rather than hidden.** With `A = I` the fp32
accumulator sums `K-1` terms of `0*w` plus one `1*w`; that is exact for every
`w` except `w = -0`, since IEEE `(+0) + (-0) = +0`. A weight whose oracle value
is `-0` (`q = 8` with a negative f16 scale - this checkpoint has them) can only
come back as `+0`. 5,572,501 of 83,886,080 weights (6.642%) are in that class.
They are counted in a separate `+0/-0 only` column and excluded from
"mismatching".

**D3 - build/run conditions asserted, not assumed.** 256-GRF reached the AOT
device compile: `-cl-intel-256-GRF-per-thread` is on the probe's link line, and
IGC itself reports **`compiled SIMD16 allocated 256 regs and spilled around 3`**
and **`around 6`** for the two instantiations. P2's 128-GRF failure mode is
therefore excluded. The 3-6 register spill is recorded, not chased.

**D4 - Intel's IGC workaround makes no difference and is not the cause of the
rate.** Risk 2 pre-registered `IGC_allowDecompose2DBlockFuncs=0` for the AOT
build. A complete second battery was built and run with it **unset**:
gate‖up M=2048 **42.50** vs **42.21** TFLOP/s (+0.7%), every other cell within
1%, and correctness byte-identical (0 real mismatches, `max abs err`
3.57628e-07 in both). The workaround neither helps nor hurts here.

**D5 - box disk.** Root went 1.4 GB free -> 15 GB -> 738 MB during the probe
as the operator's `qwen38-27b-mxfp4a16` checkpoint (15 GB, 18 shards) was
rewritten. **No disk error occurred and nothing was deleted.** The probe's own
build footprint is 656 KB of objects and a 563 KB binary. The box ends the
probe at **738 MB free / 100% used** - reported, not acted on.

## 1. Layout verdict - REPACK REQUIRED, and it is exactly a u32 transpose

Predicted: no direct fit, repack to `[N][K]` K-contiguous, a NEW layout.
**All three confirmed, the first two by measurement.**

`StrideB` printed from the instantiated kernel at (N=16, K=64): **`(64,_1,0)`**
- `get<0>` is the ldb, so B is `[N][K]` with K contiguous, as §0/S1 read.

### The exact nibble/stride mapping

| | 4-bit element index of weight (k, n) | as bytes |
|---|---|---|
| **layout 0** (`qweight` u32 `[K/8][N]`) | `p0 = ⌊k/8⌋·8N + 8n + (k mod 8)` | word `(k/8)·N + n`, nibble `k mod 8`, LSB first |
| **what the mainloop needs** (`ColumnMajor`, `ldb = K`) | `p1 = n·K + k` | byte `(n·K + k)/2`, low nibble when `k` is even |

`p0` is affine in `n` but **piecewise in `k`** - coefficient 1 on `k mod 8` and
`8N` on `⌊k/8⌋` - so no rank-3 flat `StrideB` can express it, and
`make_layout(make_shape(N,K,L), dB)` admits nothing else. Direct consumption is
impossible for **any** `LayoutB`, RowMajor included; the exclusion is a
statement about every flat stride, not about the two tags.

**The repack is exactly a u32 TRANSPOSE of `qweight`. No nibble ever moves
inside a word:**

```
out_u32[n·(K/8) + r]  =  qweight[r·N + n]  ^  0x88888888
```

because **both** layouts pack eight consecutive `k` LSB-first into one 32-bit
word, and the 4-bit array's byte order is the word's own little-endian byte
order. The `^ 0x88888888` is the signed-int4 shift of S8 (omit it for the
unsigned + zero-tensor mode). Row pitch `K/2` bytes = 2560 / 3072 / 8704 - all
multiples of 64, and `ldb = K` clears `can_implement`'s 32-element alignment.

### Measured, on the real checkpoint

`layers.0.linear_attn.in_proj_qkv ‖ in_proj_z` (K = 5120, N = 16384), fused as
`model::Qwen35`'s table says, from
`/home/user/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`. `A = I` at
M = K = 5120, so `C[k][n]` is bit-exactly the mainloop's dequantised weight.

| B bytes fed | compared | mismatching | +0/-0 only (D2) | first real mismatch |
|---|---:|---:|---:|---|
| layout 0 verbatim (`qweight [K/8][N]`, values `q^8`) | 83,886,080 | **76,268,818** (90.92%) | 733,956 | `k=0 n=1 got=0.0126342773 want=0` |
| repacked `[N][K]` 4-bit, K contiguous, `q^8`, `ConvertAndScale` | 83,886,080 | **0** | 5,572,501 | (none) |
| same repack, unsigned nibbles + int8 zero=8, `ConvertAndScaleWithZeroPoint` | 83,886,080 | **0** | 5,572,501 | (none) |

The layout-0 row feeds the **same element values** (`q^8`) in the checkpoint's
own arrangement, so its 90.92% wrong is purely the layout, not the zero point.

### Classification: a NEW layout - second copy, not a bridge

Decode's `gemv` reads word `(r, n)` with lane = `n` (`gemv.cl:84-89`), so a
subgroup's 16 lanes read 16 **consecutive** u32. Under `[N][K/8]` those same
lanes are `K/8` u32 (10 KB at K=5120) apart. **Decode cannot adopt this layout**
without a different GEMV geometry, so A21's "no decode change, Task 4's retune
untouched" survives only by paying for a second copy:

| | bytes |
|---|---:|
| 256 layer matrices | **12,163,481,600** (11.33 GiB) |
| + int4 `lm_head` | 12,799,180,800 (11.92 GiB) |
| zero tensor, `ConvertAndScaleWithZeroPoint` only (int8 `[K/64][N]`) | +380,108,800 |
| zero tensor, `ConvertAndScale` (the `q^8` route) | **0** |
| scales | **0** - layout 0's `scales f16 [K/64][N]` are passed through unchanged (S3, confirmed by the 0-mismatch rows) |

Repack cost, measured on the host: **42.5-46.1 ms** for the 40 MB qkv‖z matrix
(1.82-1.97 GB/s; it is a strided u32 gather). Extrapolated one-time over the
12.16 GB of layer matrices: **~6.4 s on the host**, at load, once. A device
transpose kernel would be far cheaper if it ever mattered. **It never becomes a
per-chunk cost.**

## 2. Correctness - CONFIRMED, and better than the prediction

- **Bit-exact against `tools/oracle/dequant.py`: 0 real mismatches in
  83,886,080 real checkpoint weights**, in **both** conversion modes (the
  `q^8` + `ConvertAndScale` route and the stock unsigned + int8-zero route).
  The pre-registered claim was exactly this, and the arithmetic reason (S7:
  `half_t(int)` explicit, `operator float()` implicit, so the product is fp32
  with one bf16 round) is confirmed by the result.
- **GEMM at the brief's shape** (M=64, K=5120, N=16384, real layout-0 weights,
  CPU fp64 reference over oracle-dequantised bf16 weights, 4096 fixed-xorshift
  cells):

| M | K | N | can_implement | workspace | max abs err | tol | two runs |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 5120 | 16384 | 0 (`kSuccess`) | **0 B** | **3.57628e-07** | 4.30496e-03 | **bitwise identical** |

  **12,038x inside the tolerance.**
- **Two runs bitwise identical in every one of the 24 rate cells**, workspace
  **0 B** everywhere, **no split-K atomics** (data-parallel
  `PersistentScheduler`; `xe_gemm.hpp:82-87` admits no other).

**The format question is settled and durable: int4-g64 GPTQ v1 IS consumable by
this mainloop, exactly, after a transpose.** That result stands whatever
happens to the rate.

## 3. Rate - MISS by 3.03x

TFLOP/s, `ConvertAndScale` (signed int4, `q^8`). 8 replays, first 3 discarded,
median of the last 5, 4 enqueues per replay, `wait_and_throw` per replay, a
discarded M=512 ramp control before each shape.

| shape | K×N | M=512 | M=1024 | M=2048 | M=4096 |
|---|---:|---:|---:|---:|---:|
| qkv‖z | 5120×16384 | 45.06 | 43.51 | 43.03 | 41.56 |
| out/o_proj | 6144×5120 | 30.42 | 38.56 | **44.96** | 43.11 |
| gate‖up | 5120×34816 | 41.29 | 42.56 | **42.21** | 41.28 |
| down | 17408×5120 | 29.34 | 38.02 | 43.64 | 41.97 |
| q‖k‖v | 5120×14336 | 39.86 | 44.07 | 43.22 | 41.92 |
| lm_head | 5120×248320 | 41.39 | 41.37 | 41.38 | 40.89 |

**gate‖up, M=2048: 42.21 TFLOP/s against the pre-registered bar of 128 -
33.0% of it, a 3.03x miss.** Against P2's corrected bf16 row at the same cell
(150.19) it is **28.1%**.

Second battery, built and run **without** `IGC_allowDecompose2DBlockFuncs=0`
(D4): gate‖up M=2048 = **42.50**; the whole matrix within 1% of the first.

`ConvertAndScaleWithZeroPoint` control, gate‖up M=2048: **42.88** (and 42.99
without the IGC flag) - **the same rate**. The `q^8` simplification therefore
buys the 380 MB of zero tensor and nothing else; it costs nothing either.

### What the numbers themselves say about the mechanism (no tuning was done)

- **The rate is flat**: 40.9-45.1 TFLOP/s across all six shapes and all four M,
  spanning N from 5,120 to 248,320 and K from 5,120 to 17,408 - a 10% total
  range where the bf16 matrix spans 47.60-164.21. That is the signature of a
  fixed per-k-tile cost, not of any shape effect.
- **It is not bandwidth-bound.** gate‖up M=2048 moves A 20.97 + B4 89.13 +
  scales 5.57 + C 285.21 = **400.9 MB in 17.298 ms = 23.2 GB/s**, 3.9% of the
  device's measured 590 GB/s.
- **It is not the 128-GRF trap and not the IGC workaround** (D3, D4).
- It sits at **23.0%** of the 183.5 TFLOPS derived XMX bf16 peak, where P2's
  bf16 configuration reaches 89.5%.

**Named but NOT measured, and deliberately not chased** (the brief forbids the
sweep that would settle it): the per-k-tile cost of the 4-bit operand path.
`XE_2D_U4x32x16_LD_T` lowers to `XeSubgroup2DBlockLoadTranspose<4,4,16,1>` - a
transposing block load of 16 rows x 16 bytes, issued four times per subgroup
per k-tile - against the bf16 path's single wide VNNI load; on top of that sit
`transform_quant`'s 128-nibble in-register unpack and a scale reload every
`group_size / BLK_K = 2` k-tiles. Which term dominates is unmeasured here.

## 4. Composed consequence - the route is a 2.3x REGRESSION, not a gain

The chunk model below reproduces `progress.md`'s measured GEMM term
(**680.06 ms** derived here against its stated 678.4 ms / 60.2%), so the
substitution is like-for-like. C = 2048; rates are the M=2048 columns.

| linear | layers | TFLOP/chunk | bf16 ms | int4-mixed ms | delta |
|---|---:|---:|---:|---:|---:|
| qkv‖z | 48 | 16.493 | 103.77 | 383.28 | +279.52 |
| out/o_proj | 64 | 8.246 | 92.62 | 183.41 | +90.79 |
| gate‖up | 64 | 46.729 | 311.13 | 1107.07 | +795.93 |
| down | 64 | 23.365 | 142.29 | 535.39 | +393.11 |
| q‖k‖v | 16 | 4.810 | 30.25 | 111.30 | +81.05 |
| **total** | | **99.643** | **680.06** | **2320.46** | **+1640.40** |

Effective rate over the production mix: **42.94 TFLOP/s** against bf16's
**146.52** - **3.412x slower**.

```
dequant saved            -210.116 ms/chunk
GEMM term                +1640.40 ms/chunk
                        ------------------
net                      +1430.28 ms/chunk

chunk  1126.958-1132.458  ->  2557.24-2562.74 ms
t/s    1808.5-1817.3      ->  799.1-800.9
```

**799.1-800.9 t/s = 40.5% of vLLM's 1973 and 44.1% of the measured baseline.**
Against the pre-registered 2220.5-2233.8 this is a **2.78x miss**; the
pre-registered sensitivity (~16 t/s per 1% of GEMM slowdown) was written for a
few percent and is far outside its linear range here.

Two thresholds, derived from the same model:

- **Break-even** (chunk time unchanged): the mixed GEMM must run at
  **>= 111.9 TFLOP/s** over the production mix. Measured 42.94 - **38.4%** of
  break-even.
- **1973 t/s**: **>= 124.4-125.2 TFLOP/s**. This independently reproduces the
  brief's pre-registered bar of 128, which was therefore set correctly.

## 5. Decision-rule branch

> **Rate < 128 or correctness fails -> report plainly; do NOT tune; the
> controller compares against plan 6f's MXFP4 route.**

**This is the branch.** Correctness passed - perfectly - and the fit needs a
one-time repack to a new layout, but the rate misses by 3.03x, so neither of
the two "A21 confirmed" branches is reachable. No tile, atom, pipeline-stage or
copy-atom sweep was run.

**A21's premise is not confirmed by measurement.** The mainloop does consume
int4-g64 exactly, and the 210 ms dequant does disappear - but the GEMM it is
traded for costs 1640 ms.

### One derived consequence the controller needs, stated as a prior, not a measurement

**Plan 6f's MXFP4A16 route is priced on this same mainloop.** A21 records it as
"`MainloopIntelXeXMX16MixedPrecision` + an owned e8m0 scale path + a second
native weight copy", i.e. the same collective, the same 4-bit B operand, the
same legacy 4-bit copy-atom family, at the same tile. Its **2220-2234 t/s**
assumed the GEMM term unchanged from the bf16 matrix. This probe measures that
assumption **false by 3.412x** for the int4 instantiation. Plan 6f's number
should be treated as **unverified** until the same gate‖up M=2048 cell is
measured for `e2m1`; the cheapest possible check is one rate cell, and this
probe's harness is the shape of it. That is a prior from a neighbouring
instantiation, not a measurement of MXFP4.

## 6. Scorecard

| pre-registered claim | outcome |
|---|---|
| Direct consumption impossible; a one-time repack is required | **Hit.** Proven analytically for every rank-3 stride and measured: layout 0 verbatim is 90.92% wrong. |
| Repack target `[N][K]` 4-bit K-contiguous, `q^8` signed int4 | **Hit**, and sharper than predicted: the repack is exactly a u32 transpose of `qweight`; no nibble moves inside a word. |
| A NEW layout: second copy, 12,163,481,600 B for the 256 layer matrices; decode cannot adopt it | **Hit** (byte figure derived, adoption argument from `gemv.cl`'s lane mapping). |
| Bit-exact vs `tools/oracle/dequant.py` on all 83,886,080 real weights | **Hit.** 0 real mismatches, in both conversion modes. |
| M=64 GEMM inside `5e-3·max|ref| + 1e-4` | **Hit.** 3.57628e-07 vs 4.30496e-03, 12,038x margin. |
| Two runs bitwise identical; workspace 0 B; no split-K | **Hit** in all 24 cells + both correctness runs. |
| gate‖up M=2048 >= 128 TFLOP/s | **MISS. 42.21** (33.0%); 3.03x short. |
| Mixed below bf16 at large M, above it at small M | **Miss.** Below bf16 at every M and every shape; the flat ~42 has no crossover. |
| Composed ceiling 2220.5-2233.8 t/s | **MISS. 799.1-800.9** - a regression from the 1808.5-1817.3 baseline. |
| `XE_2D_U4x32x16_LD_T` may not instantiate at this tile (risk 1) | **Fired**, but the cause was the **MMA atom**, not the tile; fixed with Intel's own legacy atom at the unchanged tile (D1). |
| `IGC_allowDecompose2DBlockFuncs=0` needed (risk 2) | **Refuted.** No correctness or rate difference either way (D4). |
| 256-GRF reaches the AOT compile (risk 3) | **Confirmed** from the link line and IGC's own register report (D3). |
