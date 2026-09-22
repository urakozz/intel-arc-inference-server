# Fused int4 dequant in `pf_gemm` - probe results (2026-09-22)

**Verdict: REJECTED.** Both candidate kernels are **bitwise identical** to the
two-pass slab path over all 71,303,168 fp32 outputs - the arithmetic identity
of design §2 holds exactly, in the assembly as well as in the memcmp - and both
are **far below** the pre-registered break-even. The faster of the two (V2, SLM
staged) reaches **100.37 TFLOP/s** against a break-even of **122.9**. Design §7
branch 1 applies: the probe is a failure, the slab path stands, and **stage S1
of the prefill parity program is dead**. No tuning pass was run and none is
proposed without an operator ruling.

Pre-registration: `docs/superpowers/specs/2026-09-22-fused-dequant-gemm-probe-design.md`,
committed at `107e132` before the probe source existed. Program context:
`2026-09-22-prefill-parity-program-design.md` §3, stage S1.

Every number below is **measured** unless marked **derived**.

## 1. The break-even, restated before the numbers

From design §3, derived before measuring and unchanged since:

| outcome | fused in-kernel rate | linear-path time | change |
|---|---:|---:|---:|
| break-even | **122.9 TFLOP/s** | 1618.8 ms | 0 |
| target | ≥ 140 TFLOP/s | ≤ 1421 ms | −198 ms |
| ceiling (dequant free, GEMM unchanged) | 155.7 TFLOP/s | 1278 ms | −341 ms |

122.9 TFLOP/s is 199 TFLOP of per-request linear work over the 1618.8 ms
`slab_dequant` + `slab_gemm` measured in situ
(`docs/prefill-parity-2026-09-20.md`, "The linear path, itemised").

## 2. What was run

`tools/probe/probe_fused_dequant_gemm.cc` + `tools/probe/probe_fused_dequant_gemm.cl`
(both new, probe-only) and three CMake stanzas appended to
`tools/probe/CMakeLists.txt`. **No production file was edited.**
`src/kernels/prefill/pf_gemm.cl` and `pf_dequant_slab.cl` are untouched; the
control runs those two production binaries themselves
(`pf_dequant_slab_K5120_N34816_L0`, `pf_gemm_T0`), which is what makes the
memcmp a gate rather than a paraphrase.

One shape, design §4: **gate‖up, K = 5120, N = 34816, M = 2048, layout 0** -
**730.1 GFLOP per call** (derived: 2 × 2048 × 5120 × 34816). Three measurements
in one process, same buffers, same device:

- **control** - `pf_dequant_slab` then `pf_gemm_T0` per 1024-column slab,
  launch for launch what `runtime::prefill::linear_l0` does today: 34 slabs,
  68 launches, `ldb` = 1024, `ldc` = N.
- **V1, registers only** - `pf_gemm`'s TRANSB=0 kernel with **only the B path**
  replaced. Each subgroup reads the four qweight rows of its own 64 columns
  (four `..._32b_4r16x1c` messages, 1 KB, against the 4 KB of bf16 the two VNNI
  transform loads read today) and dequantises straight into its B fragment
  registers. One launch, no slab, no SLM, no extra barrier.
- **V2, SLM staged, double buffered** - the work-group dequantises each k-tile's
  32k × 256n B **once** (subgroup `s` takes n-atom `s>>1`, k-half `s&1`) into a
  32 KB double-buffered SLM image laid out exactly as
  `intel_sub_group_block_read8` returns it, while the dpas block consumes the
  previous tile from the other buffer. One split barrier per k-tile. One launch.

Everything else in both variants is `pf_gemm.cl` transcribed verbatim: the
256×256×32 tile, the 512-work-item group, the 8×4 subgroup raster, the A 2D
block load and its 2-deep prefetch, the 16 `float8` accumulators, the 32
`dpas.8x8` per k-tile in b-outer/a-inner order, the split barrier, and the
16-store epilogue.

**Environment.** `ZE_AFFINITY_MASK=1` (device 1, the per-kernel device;
`Intel(R) Arc(TM) Pro B70 Graphics`). `ocloc`/NEO 26.35.39758.10, IGC 2.41.5,
kernel `7.3.0-rc4-b70-p2p-exp2-dirty`. Both variants AOT at
`-cl-intel-256-GRF-per-thread`.

**Grade: iterate**, not record. At dispatch (2026-09-22 15:28:27 UTC, load
average 0.22, zero containers) one DRM holder was present: pid 15288 `ptyxis`,
the GNOME terminal's `--gapplication-service` daemon that holds render-node fds
and submits nothing (`box-idle-protocol`). I did not kill it. The verdict does
not turn on the grade: the three rows share one process and one harness, two
independent runs agree to within 1.9 % on the worst row, and V2 misses
break-even by 18 %.

## 3. The three rates (measured)

L0 kernel timestamps, best of 5 after one discarded warm-up. **Σ kernel ms** is
the sum of the per-launch device durations - the same quantity
`docs/prefill-parity-2026-09-20.md` attributed as `slab_dequant` + `slab_gemm`,
and therefore the one the 122.9 TFLOP/s break-even is derived from. **span** is
first `kernelStart` to last `kernelEnd` on the in-order list; for a
single-launch variant the two coincide.

| variant | launches | Σ kernel ms | span ms | TFLOP/s | vs break-even |
|---|---:|---:|---:|---:|---:|
| **control** (dequant + GEMM, 34 slabs) | 68 | **5.341** | 5.429 | **136.72** | +13.8 |
| **V1** registers-only | 1 | **11.117** | 11.117 | **65.68** | **−57.2** |
| **V2** SLM staged, double buffered | 1 | **7.274** | 7.274 | **100.37** | **−22.5** |

Control split: `pf_dequant_slab` **0.826 ms**, `pf_gemm_T0` **4.515 ms**
(34 launches each).

Repeat run, same binaries, minutes apart: control 5.346 ms / 136.58 TFLOP/s,
V1 11.318 ms / 64.51, V2 7.346 ms / 99.39. The spread is under 1.9 % on the
worst row and changes nothing.

**The subtraction that matters.** Against the control's GEMM half (4.515 ms),
V1 adds **+6.602 ms** and V2 adds **+2.759 ms**, to delete a dequant pass that
costs **0.826 ms**. Even V2 - which issues the dequant ALU exactly **once** per
weight per work-group, the minimum this design can reach - pays **3.3×** the
term it exists to remove.

## 4. Equality (measured)

`memcmp` of each fused variant's C against the control's C over the full
M × N = 71,303,168 fp32 outputs, compared as raw 32-bit words so a sign-of-zero
difference cannot hide behind `==`:

| variant | verdict | first differing index |
|---|---|---|
| V1 registers-only | **bitwise identical** | - |
| V2 SLM staged | **bitwise identical** | - |

Nothing to report: there is no first differing index in either variant. This is
the predicted outcome of design §2 and it held on the first correct build of
both kernels. **The arithmetic is not the reason the probe fails.**

## 5. Compiler evidence

Method: plan 9a's assembly gate, `docs/probe-prefill-vllm-parity-2026-09-14.md`
§A4.8 - `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=…` around the `ocloc`
call, `.zeinfo` first, then the Xe2 `.asm`.

### 5.1 The gate, both variants

| check | required | V2 (faster) | V1 |
|---|---|---|---|
| `simd_size` | 16 | **16** | **16** |
| `grf_count` | 256 | **256** | **256** |
| `has_dpas` | true | **true** | **true** |
| `spill_mem_size` / `private_size` | absent | **absent** | **absent** |
| `barrier_count` | 1 | **1** | **1** |
| `slm_size` | - | **32768** (2 × 16 KB, as designed) | **absent** (0) |
| `dpas.8x8` per k-tile | 32, b-outer a-inner | **32** | **32** |
| `store_block2d` in epilogue | 16 | **16** | **16** |
| scatter/gather `send` in the loop | 0 | **0** | **0** |
| split barrier | `arrive`/`wait` | **present** | **present** |

Both pass. `scratch` appears in both `.asm` files only as
`//.declare %scratchloc (35) … IsBuiltin`, the universal per-kernel ABI
declaration the production `pf_gemm_T0`/`T1` gate recorded verbatim - not a
scratch-surface access; the authoritative spill signals are absent from both
`.zeinfo`s. **Spill/scratch bytes: zero in both.**

`dpas` structure, V2 (the same b-outer/a-inner macro the production kernel
emits, four consecutive `dpas` sharing one operand, `{Atomic}`-fused):

```
dpas.8x8 (16|M0)  r133:f  r133:f  r4:bf  r38.0:bf  {Atomic,Compacted,$15.dst}
dpas.8x8 (16|M0)  r165:f  r165:f  r4:bf  r42.0:bf  {Atomic,Compacted}
dpas.8x8 (16|M0)  r197:f  r197:f  r4:bf  r46.0:bf  {Atomic,Compacted}
dpas.8x8 (16|M0)  r229:f  r229:f  r4:bf  r50.0:bf  {Atomic,Compacted}
```

2D block IO per k-tile. V1 issues, per k-tile, **5 data loads**
(1 A `.d16`, 4 qweight `.d32`) + 2 prefetch - 11 `load_block2d` in the file
(5 loop data + 4 prologue prefetch + 2 loop prefetch). V2 issues **2 data
loads** (1 A `.d16`, 1 qweight `.d32` for the tile it stages) + 2 prefetch -
**9** in the file (2 loop data + 1 prologue qweight data + 4 prologue prefetch
+ 2 loop prefetch), which is the production `pf_gemm_T0` gate's own 9
(3 data + 6 prefetch), because V2's B path costs one 2-row message where the
production kernel spends two VNNI transform messages. V2 additionally issues **16 `load.slm.d32x64t.a32`**
and **1 `store.slm.d64x64t.a32`** per k-tile: the eight
`intel_sub_group_block_read8` calls each lower to two SLM messages.

V2's split barrier, verbatim: `sync.bar 0x0` at the loop top (the blocking
wait), `send.gtwy … signal barrier` at the loop bottom (the arrive), plus the
prologue arrive and the trailing wait that pairs the last one.

### 5.2 Which instruction produces the bf16 word, and that it is RNE

Design §6 requires this stated against the generated code, because the oneDNN
investigation found a vendor path that **truncated** instead of rounding. It
does not happen here. **No hardware f32→bf16 conversion instruction appears in
either binary** - there is no `mov … :bf` fed by an `:f` source anywhere. The
word is built by the integer sequence `pf_dequant_slab.cl` specifies, and IGC
emits it literally:

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
in one instruction before the `shr`, which is exactly
`rne_bf16`'s `rounding = ((u >> 16) & 1) + 0x7FFF; (u + rounding) >> 16` -
round-to-nearest-**even**, not truncation (a truncation would be the bare `shr`
with no `add3`). The `bfn` pack folds `(hi >> 16) << 16` into
`hi & 0xFFFF0000`, an exact rewrite on the same bits. The `^ 0x88888888`
appears as the `xor` that feeds every `shl` chain. `memcmp` §4 is the
end-to-end confirmation of the same claim.

### 5.3 The instruction count that explains the result

The `add3` count per k-tile is one per weight dequantised, so it counts the
work directly:

| | `add3` per subgroup per k-tile | weights dequantised per subgroup per k-tile | redundancy |
|---|---:|---:|---|
| V1 | **128** | 4 n-atoms × 32 k = 128 | **8×** - the 8 m-subgroups sharing an n-column block each repeat the whole dequant |
| V2 | **16** | 2 qweight words × 8 nibbles | **1×** - the work-group's 32k × 256n tile is dequantised exactly once |

V1's loop body carries roughly **905** ALU instructions (128 `add3`, 129 `asr`,
130 `mul`, 109 `shl`, 196 `shr`, 131 `and`, 66 `bfn`, 16 `xor`) against its 32
`dpas.8x8` - about **7 SIMD16 instructions per weight** (derived from those
counts). V2 carries one eighth of that, plus 17 SLM messages and a barrier the
production kernel does not have to honour.

## 6. What the numbers mean

The fused path does remove the traffic it was designed to remove: in situ it
would read about 12.2 GB of int4 per chunk instead of writing and re-reading
48.7 GB of bf16. That saving is real and it is **not large enough**. At this
shape the two-pass dequant already runs at about **546 GB/s** (derived:
89.1 MB of qweight + 5.6 MB of scales read and 356.6 MB of bf16 written, in
0.826 ms), i.e. ~90 % of the 608 GB/s hardware peak. The round trip is already
close to bandwidth-optimal, so there is only 0.826 ms in it at this shape - and
moving the same arithmetic inside the mainloop costs 2.759 ms even when it is
issued exactly once per weight.

Put as a requirement: to reach the §3 target of 140 TFLOP/s the fused kernel
may add at most **0.700 ms** over the GEMM half (derived: 730.1 / 140 − 4.515).
It must therefore perform the dequant **more cheaply than a dedicated,
bandwidth-saturated pass does**, while competing for issue slots with the dpas
stream. V2's measured 2.759 ms is 3.9× away from that, with the redundancy
already at its floor.

Extrapolated to the program (derived, and conditional on the gate‖up rate
transferring to the other linear shapes, which this probe did not measure):
199 TFLOP at V2's 100.37 TFLOP/s is **1983 ms** against today's 1618.8 ms -
**+364 ms**, not −198. At V1's 65.68 TFLOP/s it is 3030 ms.

This is the outcome the probe brief named in advance: *the dequant ALU cost
inside the mainloop exceeds the traffic it removes.* It is recorded, not tuned
away.

## 7. Decision

Design §7, branch 1: **fused rate < 122.9 TFLOP/s → rejected, recorded, and the
slab path stands. No tuning pass without a ruling.**

- `runtime::prefill::linear_l0`, `pf_gemm.cl` and `pf_dequant_slab.cl` are
  unchanged and stay unchanged. No adoption spec is written.
- **Stage S1 of `2026-09-22-prefill-parity-program-design.md` is dead**, and its
  −198…−270 ms comes off the program's expected total. Nothing downstream
  depends on S1 (program spec §3), so the cost to the program is exactly its own
  estimate. S2-S5 are unaffected and remain in their stated order.
- The 7,616 launches per request the fused path would also have removed are not
  recovered by any other stage and are not, on their own, worth 364 ms.

**What this probe does NOT license.** It measured one shape, one tile
(256×256×32) and one layout (0). It does not rule on: a different tile whose
m-subgroup count is smaller (V1's 8× redundancy is a property of the 8×4
raster, not of fusion); a pre-swizzled int4 layout that would cut the shl/asr
chain; or a mixed-precision DPAS path that changes the rounding point - that
last one is explicitly out of scope and needs its own pre-registration
(design §2). Any of those is a **new** probe with a new break-even, not a
tuning pass on this one.

## 8. Artifacts

- Probe host: `tools/probe/probe_fused_dequant_gemm.cc`
- Probe kernels: `tools/probe/probe_fused_dequant_gemm.cl` (`-DFUSED=1` → V1,
  `-DFUSED=2` → V2; both AOT at 256 GRF)
- Registration: the appended block at the end of `tools/probe/CMakeLists.txt`
- Run: `ZE_AFFINITY_MASK=1 tools/box.sh run ./build/tools/probe/probe_fused_dequant_gemm`
  (the probe prints its own verdict line and returns non-zero only on a memcmp
  difference, which did not occur)
