# Fused int4 dequant in `pf_gemm` - probe design

**Status:** design, 2026-09-22. Probe first; no production file changes in this
document's scope. Adoption is a separate spec that may only be written after the
probe's numbers exist.

## 1. The question

The prefill linear path costs **1618.8 ms** of GPU time per 4096-id request,
measured and itemised on 2026-09-22 (`docs/prefill-parity-2026-09-20.md`, "The
linear path, itemised"):

| phase | L0 GPU ms |
|---|---:|
| `slab_dequant` (`pf_dequant_slab`) | 274.5 |
| `slab_gemm` (`pf_gemm`) | 1344.3 |

The GEMM half is within about 5% of its own derived floor (199 TFLOP at the
156.53 TFLOP/s P-A measured = 1278 ms), so the matrix math is not the problem.
The dequant half exists only to materialise bf16 weights that the GEMM reads
once and never reads again: about 48.7 GB written and re-read per chunk, where
the int4 source is 12.2 GB (derived from SUM(K.N) = 24.3 G weights).

**Question:** can `pf_gemm` read the int4 weights directly and produce its B
fragments in-kernel, so that the round-trip disappears, without changing a
single bit of C?

## 2. Fixed arithmetic - the bit that makes this cheap to gate

`pf_dequant_slab` produces, per weight, `rne_bf16((float)qm8 * scale)` where
`qm8` is the sign-extended nibble and `scale` is the group's f16 scale widened to
f32 (`src/kernels/prefill/pf_dequant_slab.cl`). The fused kernel **must produce
that same 16-bit word** and feed it to the same DPAS chain in the same k order.

If it does, C is **bitwise identical** to today's two-pass result, and the gate
is a `memcmp`, not a tolerance. This is the whole reason to prefer fusion over
any mixed-precision DPAS path: no rounding point moves, so none of the numerical
gates that killed the single-BF16 GDN scan can fire.

Any design that changes the dequant expression, the scale's widening, the k
order, or the accumulator type is **out of scope for this probe** and needs its
own pre-registration.

## 3. Break-even, stated before measuring

Effective end-to-end rate of the current linear path:
199 TFLOP / 1618.8 ms = **122.9 TFLOP/s** (derived).

| outcome | fused in-kernel rate | linear-path time | change |
|---|---:|---:|---:|
| break-even | 122.9 TFLOP/s | 1618.8 ms | 0 |
| target | **≥ 140 TFLOP/s** | ≤ 1421 ms | **−198 ms** |
| ceiling (dequant free, GEMM unchanged) | 155.7 TFLOP/s | 1278 ms | −341 ms |

The ceiling is unreachable in practice because the dequant ALU work moves into
the GEMM rather than vanishing. **The probe is a failure if the fused rate is
below 122.9 TFLOP/s**, and that verdict is recorded rather than tuned away.

## 4. What the probe builds

A standalone probe, `tools/probe/probe_fused_dequant_gemm.cc` plus its kernel,
modelled on `tools/probe/probe_pf_gemm.cc`. No production file is edited.

**One shape**: gate‖up, K = 5120, N = 34816, M = 2048, layout 0 - the largest
linear and 1/3 of all linear FLOPs. If it fails here it fails everywhere.

Three measurements in one process, same buffers, same device:

1. **Control**: `pf_dequant_slab` then `pf_gemm` per 1024-column slab, exactly
   what `linear_l0` does today. Report ms and TFLOP/s.
2. **Fused**: the candidate kernel, whole N, no slab buffer.
3. **Equality**: `memcmp` of the two C buffers over M x N fp32. Any difference,
   including sign-of-zero, is a failed probe - report the first index.

Each variant is timed with L0 kernel timestamps, best-of-5 after one warm-up,
on an idle device, and the report states device, IGC version and the grade word.

## 5. The candidate kernel

Start from `src/kernels/prefill/pf_gemm.cl` (256x256x32 tile, 256 GRF, split
barrier, 2D block loads) and replace **only the B path**:

- **B source**: `qweight` as `[K/8][N]` u32 (8 nibbles along k per word) plus
  `scales` as f16 `[K/64][N]`, the layout the loader already ships for layout 0.
- **Per k-tile (32 deep, 256 wide)**: the words needed are
  `qweight[k0/8 .. k0/8+3][n0 .. n0+255]` - 4 KB - and one scale row per 64-k
  group. Each subgroup dequantises the fragment it will feed to DPAS, in
  registers, in the same VNNI pair order the 2D block load produces today.
- **A path, accumulators, prefetch, split barrier, tile shape: unchanged.**

Two variants worth measuring, because the answer is not obvious and the probe is
where guessing stops:

- **V1, registers only.** Dequant straight into the B fragment registers. No SLM,
  no extra barrier. Risk: the dequant ALU is on the critical path of every
  k-tile, and the 2D block load it replaces was nearly free.
- **V2, SLM staged, double buffered.** Cooperative dequant of the next k-tile's
  B into SLM while DPAS consumes the current one. Costs SLM (4 KB per buffer per
  n-tile, so 8 KB double-buffered) and one barrier per k-tile; buys latency
  hiding. Risk: SLM pressure lowers occupancy, and the DPAS B operand now comes
  from SLM instead of a block load.

Report both. If V1 wins, the design is simpler than anyone expected and the
adoption spec is short.

## 6. Evidence required

- Rate and ms for control, V1, V2.
- `memcmp` equality for both variants (mandatory; a fast wrong kernel is a
  failed probe, not a trade-off).
- For the faster variant: the compiler resource report - SIMD width, GRF
  allocation, SLM bytes, **spill/scratch bytes**, and the DPAS count per k-tile,
  confirming the mainloop still issues the same 32 `dpas.8x8` it does today.
- The arithmetic identity argument stated against the generated code: which
  instruction produces the bf16 word, and that it is `rne`, not truncation. The
  oneDNN investigation found a vendor path that truncated instead of rounding
  (`docs/prefill-parity-2026-09-20.md`), which is exactly this failure mode.

## 7. Decision rule

- Fused rate **< 122.9 TFLOP/s**, or any `memcmp` difference → **rejected**,
  recorded, and the slab path stands. No tuning pass without a ruling.
- Fused rate **≥ 140 TFLOP/s** with bitwise equality → write the adoption spec:
  production kernel variant, `linear_l0` rewritten to one launch per linear
  (which also removes 7,616 launches per request), the full `pf_gemm_test` cell
  matrix extended to the fused variant, and the four model gates.
- Between 122.9 and 140 → operator ruling. The saving is real but smaller than
  the work, and the launch-count reduction may still justify it.

## 8. What this probe does not touch

The transposed-B attention path, decode, the dequant kernel itself (it stays for
the sycl-tla backend and for `pf_dequant_tile`'s tests), chunk size, the scale
layout, and every rounding point in the engine. Layout 1 (tiled, qkv‖z) is out of
scope for the probe and is the first thing the adoption spec must add, because
the production path needs both layouts.
