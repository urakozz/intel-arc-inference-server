# `pf_gdn_scan` on DPAS, and `pf_gdn_A2`'s staging back-port - ruling A28 option (c)

Ruling A28 named the DPAS scan as "a real task" and the operator's 2026-09-05
ruling took **(b+c)**; task (b) (`docs/prefill-gdn-a2a-simd32-2026-09-05.md`)
ran first and left one measured, un-taken lever behind (§9.1 of that document,
`pf_gdn_A2`'s staging loop). This task is that lever plus the DPAS scan.

It follows the shape the previous three documents use: **every pre-registration
section below was committed before the change it describes was written**,
everything after it is labelled measured / derived / estimated, and no quantity
appears twice with two values.

Grade: **iterate** throughout, and the reason is unchanged since gate row 1.
The two desktop processes named in `progress.md` - baobab (285644) and ptyxis
(285678) - still hold render-node fds on both B70s and on the iGPU, measured at
the start of this task by walking `/proc/*/fd` at **22:07:09**:

```
285644 baobab  /dev/dri/renderD128 + renderD129 + renderD130
285678 ptyxis  /dev/dri/renderD128 + renderD129 + renderD130
```

Four holders on the two B70s, so the record-grade condition is not met and
every row here is **iterate**. The holder count is measured again immediately
before the `--pp 4096` row and graded there. Load average 0.02, no container,
84 GB free.

Timed `--pp` rows are 8 independent runs, median, with min/max/spread beside
them; the drop-3 half of the project's iterate convention does not apply to a
`--pp` row (each run is a separate process with a cold model load, so there is
no warm-up sequence inside a process to drop -
`docs/prefill-pp-attribution-2026-09-05.md` §1). Per-kernel figures are single
`B70_PREFILL_PROFILE=1` runs, as in the previous three documents.

## 0. The control, measured before anything was changed

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8` at `a89311f`, one run, **measured**, beside task (b)'s published
figures (`b2c42c7`, §9 of that document, taken the same way on the same
device):

| ms/layer/chunk (column ÷ 96) | this session (`a89311f`) | task (b)'s (`b2c42c7`) | drift |
|---|---:|---:|---:|
| `pf_gdn_conv` | 0.3294 | 0.3296 | −0.06% |
| `pf_gdn_l2norm` | 0.1492 | 0.1486 | +0.40% |
| `pf_gdn_A` | 0.3289 | 0.3281 | +0.24% |
| `pf_gdn_solve` | 0.7966 | 0.7895 | +0.90% |
| `pf_gdn_wu` | 0.9343 | 0.9314 | +0.31% |
| **`pf_gdn_A2`** | **0.5962** | 0.5955 | +0.12% |
| **`pf_gdn_scan`** | **3.3937** | 3.3935 | +0.01% |
| `pf_gated_head` | 0.3204 | 0.3189 | +0.47% |
| profiled walk total, ms | **2882.7** | 2878.7 | +0.14% |

**There is no session drift to correct for**: the two sessions' controls agree
to 0.9% on every GDN kernel (the worst is `pf_gdn_solve`, which task (b) §8.3
already recorded as this box's noisiest kernel) and to 0.14% on the walk.

### 0.1 The two published values for `pf_gdn_A2`, and which one the bar uses

Task (b) reports `pf_gdn_A2` twice for two different runs: **0.5889** in its
§4.1 (the rewrite's own first measurement) and **0.5955** in its §9 (the final
profiled walk, after `pf_gdn_A` had also changed). They are the same kernel at
two instants of the same instrument, 1.1% apart. The brief fixes the bar
against **0.5889**, so that is what §1.2 scores against, and this session's
control (0.5962) is reported beside every result. No third value is coined.

### 0.2 The machine's own numbers, used by every derivation below

Fixed here so that no later section re-derives them differently (they are
`docs/prefill-gdn-scan-2026-09-05.md` §2.1 and
`docs/prefill-gdn-a2a-simd32-2026-09-05.md` §0.1; nothing new is claimed):

* **256 XVEs at 2800 MHz = 7.17e11 XVE-clocks/s**, 8 hardware threads per XVE =
  **2048 thread slots**; 32 Xe-cores of 8 XVEs × 8 threads = **64 thread slots
  and 128 KB of SLM per Xe-core**.
* A GRF register is **64 B**, so one fp32 vector value costs **1 register at
  SIMD16 and 2 at SIMD32**.
* The link line carries **`-abortOnSpill 4`**, so a spill is a **build
  failure**, not a silent regression.
* `pf_gdn_scan` does **9.66 GFLOP per GDN layer per chunk** at C = 2048
  (`docs/prefill-gdn-scan-2026-09-05.md` §1.1), and 96 = 48 GDN layers × 2
  chunks is the multiplier from ms/layer/chunk to ms on a `--pp 4096` run.

---

# PART 1 - item 0: `pf_gdn_A2`'s staging loop

## 1. PRE-REGISTRATION (committed before the change was built)

### 1.1 The defect, already measured

`docs/prefill-gdn-a2a-simd32-2026-09-05.md` §4.2 measured it inside the Xe2
ISA of the shipped kernel. `pf_gdn_A2`'s staging loop
(`src/kernels/prefill/pf_gdn_wy.cl:389-396`) is

```c
for (uint p = lid; p < AQ * DIM; p += WG_TRI) {        // 16 iterations
  const uint r = p / DIM, d = p % DIM;
  const uint gi = iz * AQ + r, gj = jz * AQ + r;
  qs[p] = gi < L ? bf16f(xb[... Q_OFF ...]) * Q_SCALE : 0.0f;
  ks[p] = gj < L ? bf16f(xb[... K_OFF ...])           : 0.0f;
}
```

- **one value per array per iteration, with the `< L` test written as a ternary
around a global load**. The measured body is **53 instructions per iteration**,
16 iterations = **848 instructions per work-item**, against ~986 for the whole
contraction: **47% of the kernel**. The itemised body (measured, task (b) §4.2):
2 scalar `load.ugm.d16u32`, ~14 instructions of 64-bit address arithmetic, **9
`goto`/`join` - the two guards, as BRANCHES**, ~9 `shl`/`mov`/`mul` for the
widen and `Q_SCALE`, 2 `store.slm.d32`.

`pf_gdn_A` has the identical loop shape and task (b) wrote it differently
**before** building it (§5.2 of that document): the row index is clamped
(`min(gi, L-1)`) so the global read is unconditional and the `< L` test becomes
a **select on the loaded value**, and four values move at a time through one
`vload4` of `ushort4` and one `vstore4` of `float4`. Measured on `pf_gdn_A`
(task (b) §6.2): staging fell from 848 instructions per work-item to **339**.

Task (b) deliberately did **not** back-port it, because `pf_gdn_A2` had by then
been measured and "one defect, one fix, one measurement" forbids re-tuning a
measured mapping. It priced the back-port at **~0.37 ms/layer/chunk (derived)**
and left it as this task's item 0. That is what §1.2 pre-registers.

### 1.2 The change, fixed here before it is built

`pf_gdn_A2`'s staging loop becomes `pf_gdn_A`'s, and **nothing else in the file
changes**:

```c
for (uint p = lid; p < AQ * DIM / 4; p += WG_TRI) {    // 4 iterations
  const uint r = p / (DIM / 4), d = (p % (DIM / 4)) * 4;
  const uint gi = iz * AQ + r, gj = jz * AQ + r;
  const ushort4 wq = vload4(0, xb + (base_m + min(gi, L - 1)) * CONV_ROWS + qbase + d);
  vstore4(gi < L ? (float4)(bf16f(wq.s0) * Q_SCALE, ... ) : zero4, 0, qs + r * DIM + d);
  const ushort4 wk = vload4(0, xb + (base_m + min(gj, L - 1)) * CONV_ROWS + kbase + d);
  vstore4(gj < L ? (float4)(bf16f(wk.s0), ... ) : zero4, 0, ks + r * DIM + d);
}
```

The loop covers `(r, d)` for `r ∈ [0,32)` and `d ∈ {0,4,…,124}` exactly once,
as the old one covered `(r, d)` for `d ∈ [0,128)` exactly once. Alignment: the
element index `(base_m+row)·10240 + {0,2048} + kh·128 + d` is a multiple of 4
for `d ≡ 0 (mod 4)`, so every `vload4` of `ushort4` is 8-byte aligned.
`min(gi, L-1)` is always a live row (`L ≥ 1`), so the global read is in bounds
and unconditional; rows past `L` store the same exact `0.0f` the ternary stored
before.

**Nothing else moves.** The grid is A28's quadrant grid, the work-group is 256,
`intel_reqd_sub_group_size(16)` stays (task (b) §2 measured the other width as
1.24× slower on `pf_gdn_wu` and reverted it - the lever is dead and is not
re-tried), the SLM stays at exactly 32,768 B, `tile_dot` is untouched.

### 1.3 Pre-registered outcomes

**Time.** `pf_gdn_A2` **≤ 0.40 ms per GDN layer per chunk** at C = 2048 (the
brief's bar, against task (b) §4.1's 0.5889). Point estimate **0.37 ms**,
task (b) §9.1's derivation, restated so it is checkable: the staging body goes
from 26.5 instructions per staged value to ~5, i.e. ~848 → ~170 per work-item,
which is ~680 of the ~1800 instructions a work-item issues - **1.61× fewer
instructions**, and 0.5889 / 1.61 = **0.366 ms** at an unchanged issue rate.
The issue rate should not fall: SLM, occupancy, sub-group width and the
contraction are all untouched.

Decision rule, fixed in advance (the brief's):

| outcome | verdict |
|---|---|
| ≤ 0.40 ms **and** bitwise identical | **adopt** |
| > 0.40 ms, or any word differs | **revert**, report |

**Numerics - the pre-registered outcome is BITWISE IDENTITY.** The staged value
is the same expression on the same word: `bf16f(word)` is exact in fp32 by
construction and `* Q_SCALE` is the same single fp32 multiply of the same two
values, in the same place. No sum is re-associated, no rounding point moves, so
the bar is bit equality and not a band:

* **`pf_gdn_A2` bit-identical to `pf_gdn_A2_legacy`** over every one of the
  `nchunks · 48 · 64 · 64` fp32 entries, at **C = 256, at the ragged C = 100 and
  at C = 4096** - the same three widths task (b) used. This is
  `gdn_wy_test` **case 7**, which already exists and already runs at all three
  widths with the two differing buffer fills (`0x11111111` / `0x22222222`), so
  an entry no quadrant wrote fails the `memcmp` rather than passing it. **No
  test change is needed and none is made.**
* `gdn_wy_test`'s case-4 bars unchanged: ≤ 4 fp32 ulp against
  `gdn_chunk_ref::mat_A2`, device non-zero count = `Σ 48·L(L+1)/2`, the diagonal
  non-zero on every row, exactly `0.0f` above the diagonal and outside `[0,L)²`.
* Because the bar is bit equality, **`gdn_chunk_test`'s band cannot move** and
  no band is re-pre-registered for this item. It is re-run anyway.
* A golden regression is a **finding**: diagnose it, report it, stop.

**`--pp 4096` consequence (derived).** 96 × (0.5889 − 0.40) = −18.1 ms at the
bar; 96 × (0.5889 − 0.366) = **−21.4 ms at the point estimate**. Composed with
the scan in §7; not taken as its own row.

`tests/prefill/gdn_chunk_ref.h` documents `pf_gdn_A2`'s reduction order and
carries a standing "must be edited together" rule with the kernel; **the order
does not change**, and the comment block records that this loop moved and the
order did not.

## 2. RESULT - `pf_gdn_A2` at 1.703×, bitwise, bar HIT

Everything in this section was measured after §1 was committed at `60ccad1`.
Kernel at `src/kernels/prefill/pf_gdn_wy.cl` (entry `pf_gdn_A2`, staging loop
only); bar asserted by `tests/prefill/gdn_wy_test.cc` case 7, unchanged.

### 2.1 Time, and the decision-rule verdict

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (33.2 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_A2`, ms per GDN layer per chunk | **0.3459** | |
| the same before the back-port, task (b) §4.1's | 0.5889 | **1.703×** |
| the same before the back-port, this session's control (§0) | 0.5962 | 1.724× |
| **pre-registered bar** | ≤ 0.40 | **HIT** |
| point estimate | 0.366 | beaten by 5.5% |
| rate on 0.818 GFLOP/layer/chunk (derived) | **2.365 TFLOP/s** | was 1.389 |

**Decision-rule verdict: ≤ 0.40 and bitwise → ADOPT.** The loop was not
iterated: §1.2's text is what was built and 0.3459 is that build's first and
only measurement.

Per-chunk consequence: 48 × (0.5889 − 0.3459) = **−11.7 ms/chunk**, **−23.3 ms
over a 4096-token prefill** (derived), against the −43 ms task (b) §9.1
projected - that projection scaled the whole kernel by 1.61× from an
instruction count, and the measured 1.70× arrives at 0.3459 rather than 0.366,
so the *speed-up* beat its estimate while the *saving* is half of §9.1's
because §9.1 mis-stated its own baseline as the walk figure 0.5955 minus 0.37
times 96 ≈ 43 when the correct arithmetic on those two numbers is 96 ×
(0.5955 − 0.37) = 21.9. The measured saving is 23.3 ms; the 43 ms figure was
arithmetic, not measurement, and is superseded here.

Nothing else moved (measured, same run): `pf_gdn_A` 0.3290 (+0.03% vs §0),
`pf_gdn_wu` 0.9342 (−0.01%), `pf_gdn_scan` 3.4072 (+0.40%), `pf_gdn_conv`
0.3291 (−0.09%), `pf_gdn_solve` 0.7932 (−0.43%), `pf_gated_head` 0.3198
(−0.19%). Profiled walk **2882.7 → 2860.0 ms (−22.7)**, which is the −23.3 the
per-kernel delta predicts to within 0.6 ms.

`pf_gdn_A2` (0.3459) is now within 5% of `pf_gdn_A` (0.3290), which is what the
two kernels' shapes predict: `A` stages one operand instead of two whenever
`iz == jz` and is otherwise the same code.

### 2.2 Numerics: bitwise, as pre-registered

`gdn_wy_test` at `ZE_AFFINITY_MASK=1`, **measured**, case 7 green at all three
widths and every existing bar unmoved:

| width | bar | measured |
|---|---|---|
| C = 256 | `A2` bit-identical to `pf_gdn_A2_legacy` | **786,432 fp32 entries identical** |
| C = 100 (ragged, L = 36) | the same | **393,216 entries identical** |
| C = 4096 | the same | **12,582,912 entries identical** |
| case 4, C = 4096 | ≤ 4 fp32 ulp vs `gdn_chunk_ref::mat_A2` | max **4** (max rel 3.031e-07), 10,076,482/12,582,912 exact |
| case 4 | device non-zero count = `Σ 48·L(L+1)/2` | **6,389,760 = 6,389,760** |
| case 4 | diagonal non-zero on every row | **196,608 of 196,608** |
| case 4/5 | exactly `0.0f` above the diagonal and outside `[0,L)²` | asserted, green |

Every host-side figure is task (b) §4.3's to the digit, which is what bit
equality predicts. Cases 1, 2, 3, 6 and 7a unchanged and green (`A` and `w`/`u`
still bit-identical to their own legacy kernels).

---

# PART 2 - the DPAS scan

## 3. PRE-REGISTRATION (committed before the kernel was built)

Everything in §3 was committed before a line of the DPAS kernel existed. §3.1 is
the one thing measured before it: the operand layout the builtin expects, which
is an API fact and not a design choice.

### 3.1 The builtin, and its three fragment layouts - MEASURED, not read

`intel_sub_group_bf16_bf16_matrix_mad_k16` is an unconditional IGC builtin
(`.superpowers/.../explorer-3-dpas-prior-art.md` §1): bf16 × bf16 → fp32,
`M ∈ {1,2,4,8}`, `K = 16`, `N = 16`, sub-group 16, and it lowers to a real
`dpas.8x8`. Its signature at `M = 8` is
`float8 (short8 a, int8 b, float8 acc)` (`~/sycl-tla/include/cute/arch/
mma_xe_legacy_builtin.hpp:34`).

The **fragment layouts** are the fact a design has to be built on, and reading
`MMA_Traits<XE_DPAS_TT<...>>` (`~/sycl-tla/include/cute/atom/mma_traits_xe.hpp:
86-95`) gives them only up to cute's codomain convention, which is exactly the
kind of thing that is worth ten minutes and not worth guessing. A standalone
probe (`/tmp/dpasprobe/dpas_probe.{cl,cc}` on the box: one `ocloc` binary, one
120-line Level Zero host, nothing in the repo) computed one 8×16×16 product
from small integers - every product and every 16-term sum exact in fp32, so the
comparison is bit-exact and a layout error cannot hide in a rounding
difference. **Measured, `ZE_AFFINITY_MASK=1`:**

| hypothesis | result |
|---|---|
| **A: lane `l` holds `A[0..M-1][l]`** (one K column per lane, register index = m) | |
| **B: lane `n` holds column `n`, VNNI pairs along K, `int r = (B[2r+1][n] << 16) \| B[2r][n]`** | |
| **C: lane `n` holds `C[0..M-1][n]`** (one N column per lane, register index = m) | **0 of 128 entries mismatch - CONFIRMED** |
| control: the same with the VNNI pair order swapped | 128/128 mismatch |
| control: the same with A's fragment transposed | 127/128 mismatch |

Both controls fail, so the pass is not an accident of the fixture. **Even `k`
goes in the LOW half of each `int`, which is what a little-endian `uint` read of
a k-contiguous bf16 row gives for free** - the fact the whole B-operand design
below rests on.

### 3.2 D1 is EMPTY, and this is stated plainly

The brief's D1 - "DPAS only where both operands are already bf16" - **does not
exist on this kernel.** Every one of the three stages has exactly one bf16
operand and one fp32 one:

| stage | matmul | bf16 today | fp32 today |
|---|---|---|---|
| 1 | `vn = u − W·S` | `W` (Q4, bf16 in memory), `u` (Q3) | **`S`** |
| 2 | `o = (Q·S)·expg + A2·vn` | `Q` (`xb`, P6/P7) | **`S`**, **`A2`**, **`vn`** |
| 3 | `S ← S·dl + Kᵀ·D` | `K` (`xb`, P7) | **`D = vn·exp(gl−gc)`** |

`S` is the fp32 recurrent state, and `vn`/`D` are the fp32 intermediates the
state is built from. D1 is therefore empty unless `S` is *carried* in bf16
across sub-chunks, which is not a rounding at the DPAS input but a permanent
truncation of the model's recurrent memory - a different kernel and a different
model, not an optimisation. **The design is D2.**

### 3.3 D2 - the design, fixed here before it is built

**Grid: unchanged.** (48 heads, 4 state-column chunks) = **192 work-groups of
256 = 16 subgroups × SIMD16**; work-group `(h, c)` owns state columns
`[32c, 32c+32)`. The brief made the grid free; it is kept, and the reason is
occupancy arithmetic rather than inertia: at 256 work-items an Xe-core holds
**4 work-groups (64 of 64 thread slots)** and the launch needs 6 per core, so
the wave structure is **1.5 waves - identical to A27's kernel**. Every grid that
removes that factor (fewer, fatter work-groups) multiplies the per-lane
accumulator count by 2 or 4; the state alone is already 16 fp32 per work-item,
and `-abortOnSpill 4` turns a spill into a build failure. Holding the grid also
keeps the two measurements comparable in the one variable that matters.

**Orientation: stages 1 and 2 are computed TRANSPOSED, stage 3 is not.** This is
the design's only non-obvious choice and it is what makes every fragment a
single contiguous per-lane load:

* stage 1 as `vnᵀ[32][64] = uᵀ − Sᵀ[32][128] · Wᵀ[128][64]`: `A = Sᵀ`
  (M = 32 state columns, K = 128 kdim), `B = Wᵀ` (K = kdim, N = 64 positions).
  The B fragment is then `W[position][16 consecutive kdim]` - **already VNNI
  order in memory**, so it is one `vload8` of `uint` straight off `w`, with no
  repack anywhere. The A fragment is 8 consecutive *columns* at one kdim row,
  one `vload8` of `ushort` out of SLM.
* stage 3 in the natural orientation `S[128][32] ← S·dl + Kᵀ[128][64]·D[64][32]`:
  `A = Kᵀ` (M = kdim, K = position) is `xb`'s k row for one position, 8
  consecutive kdim per lane - again one `vload8`; `B = D` (K = position,
  N = column) needs D **transposed** in SLM, which costs 8 scattered 16-bit SLM
  stores per work-item and buys a contiguous VNNI-ready `vload8` of `uint`.
  The transposed orientation for stage 3 was priced and rejected: its B operand
  would be `K` with positions on the K axis, which needs `k` staged transposed
  in **16 KB** more SLM and drops residency to 3 work-groups per Xe-core = 2
  waves.

**Tile ownership.**

| | C tile | M-tiles × N-tiles | per subgroup |
|---|---|---|---|
| stages 1-2 (`vnᵀ`, `oᵀ`) | 8 columns × 16 positions | 4 × 4 = 16 | **one** tile: `mt = sg >> 2` (columns `8mt..8mt+7`), `nt = sg & 3` (positions `16nt..16nt+15`); lane `n` ↔ position `16nt+n` |
| stage 3 (`S`) | 8 kdim × 16 columns | 16 × 2 = 32 | **two** tiles: M-tile `sg` (kdim `8sg..8sg+7`), both N-tiles; lane `n` ↔ column `16nt+n` |

Stage 3's C fragments **are the state**: 2 × `float8` = **16 fp32 per
work-item**, exactly what A27's kernel already holds in registers, and the
global `state` load/store is the same 16 coalesced 64 B lines it already is.

**SLM, itemised (22,400 B):**

| array | shape | bytes | why the stride |
|---|---|---:|---|
| `Sb` | `ushort[128][34]` - bf16 `S[k][x]` | 8,704 | 34 ushorts = 17 dwords, odd, so the 16 lanes' strided `vload8` hits 16 distinct banks |
| `VNb` | `ushort[64][34]` - bf16 `vn[i][x]` | 4,352 | same |
| `Dt` | `ushort[32][66]` - bf16 `D[i][x]`, **transposed** to `[x][i]` | 4,224 | 66 ushorts = 33 dwords, odd, same reason |
| `A2b` | `uint[10][16][8]` - bf16 A2, VNNI-packed, the 10 live `(nt,kb)` blocks | 5,120 | 6 of the 16 blocks are entirely above the diagonal and are not stored |

22,400 B is **below** A27's 25,088 B, so SLM is not the residency constraint at
either kernel (threads are: 4 work-groups per Xe-core both times).

**Barriers per sub-chunk: 3**, the same as A27's kernel - after staging
(`Sb`, `A2b`), after stage 1 fills `VNb`/`Dt`, and at the end of the sub-chunk
before the next one overwrites them.

**DPAS count per work-item per sub-chunk: 26.5** - 8 (stage 1) + 8 (stage 2's
`q·S`) + 2.5 average (stage 2's `A2·vn`; block `nt` needs `nt+1` of the 4
K-blocks because `A2[i][j]` is exactly `0.0f` for `j > i`, so the blocks above
the diagonal are skipped and the diagonal block's zeros contribute exact `+0`)
+ 8 (stage 3).

**The short final sub-chunk.** Every global read whose row index could pass `L`
is clamped with `min(pos, L-1)` - always a live row - and the results are simply
not stored; `VNb` and `Dt` are written **exactly `0.0f`** for positions `≥ L`,
which is what makes stage 3's clamped `Kᵀ` reads contribute exactly nothing.

**The k-loops are written rolled.** If IGC's unrolling spills, the build fails
under `-abortOnSpill 4` and the unroll factor is capped; that is a build fix,
not a mapping change, and it is said here so it cannot be presented as one
later.

**Entry points.** `pf_gdn_scan` is the DPAS kernel and is what `gdn.cc` binds;
**`pf_gdn_scan_vec` is A27's kernel kept verbatim in the same `.cl`**, launched
by nothing - the fallback the decision rule names and the vector reference,
exactly as `pf_gdn_wu_legacy` / `pf_gdn_A_legacy` / `pf_gdn_A2_legacy` /
`pf_gdn_conv_legacy` are kept. **On a revert the two entry points swap names**,
so the production kernel is always `pf_gdn_scan` and the test-only one is
`pf_gdn_scan_dpas`.

### 3.4 The FIVE new bf16 rounding points, named before they are measured

D2's whole cost is here. `S` stays **fp32 as the master state** - stage 3's
accumulation is fp32 out of DPAS and the register state, the global `gdn_state`
and the chunk-to-chunk carry are all untouched fp32. What becomes bf16 is the
**read** of `S` into the matmuls, and the intermediates on the DPAS inputs:

| | rounding | reaches |
|---|---|---|
| **R1** | `S[k][x] → bf16` for stage 1's `w·S` | `vn` → `gdn_state` **and** `gdn_o` |
| **R1′** | the same `Sb` for stage 2's `q·S` | `gdn_o` only |
| **R2** | `vn[i][x] → bf16` as stage 2's A operand | `gdn_o` only |
| **R3** | `vn[i][x]·exp(gl−gc[i]) → bf16` as stage 3's B operand | **`gdn_state` directly** |
| **R4** | `A2[i][j] → bf16` as stage 2's B operand | `gdn_o` only |

Decode has none of these and neither does the chunked CPU reference; they are
new with this kernel, they are the reason A22's band is not this kernel's bar,
and `tests/prefill/gdn_chunk_ref.h` is edited in the same commit to carry all
five.

**One rounding point MOVES and it moves in the safe direction.** P6 - the
`1/√128` q-scale, "in fp32, after the l2norm's bf16 round" - is today applied
**per term** (`bf16f(q) * Q_SCALE` inside the k-loop, 128 fp32 multiplies per
output). On DPAS the A operand must be the bf16 word itself, so the scale is
folded onto the **accumulated dot**: `o = (Σ q_raw·S)·(Q_SCALE·expg[i])`. That
is one fp32 multiply where there were 128, it keeps P6 in fp32 and after the
bf16 read exactly as A6 requires, and it **removes** roundings rather than
adding them. It is still a change of order and it is recorded.

### 3.5 The band: predicted, with the arithmetic, before it is measured

bf16 keeps 8 mantissa bits, so RNE gives a relative error ≤ **2⁻⁹ = 1.95e-03**
per rounded value. For a 128-term dot of random-sign terms the errors add in
quadrature against a result that is itself a random walk of the same terms, so
the *relative* error of `w·S` from R1 is **~2⁻⁹, not √128·2⁻⁹** - the √128
amplification the brief names (2.2e-02) is the **worst-cancellation** case,
where `|Σ|` is √128 below the term norm, and it is used below as the upper edge
rather than the centre.

The state already carries four 2⁻⁹ roundings with no twin in decode - Q1
(`vb`), Q2 (`kb`), Q3 (`u`), Q4 (`w`), all in `pf_gdn_wu` - and they measure
A22's **3.506e-02 max / 1.197e-03 mean** on `gdn_state`. R1 and R3 are two more
of the same size on the same path (R3 is the sharpest of the five: it is a
direct 2⁻⁹ relative rounding of the rank-1 update itself). Six independent
equal sources instead of four is √(6/4) = **1.22×**; allowing R3 to count double
gives **1.5×**. Predicted, therefore:

| tensor (case 1) | A22's | **predicted** | **pre-registered bar** |
|---|---:|---:|---:|
| `gdn_state` max rel | 3.506e-02 | **~5.3e-02** | **≤ 1.05e-01** |
| `gdn_state` mean rel | 1.197e-03 | **~1.8e-03** | **≤ 3.6e-03** |
| `gdn_o` max rel | 5.039e-02 | ~8e-02 | ≤ 1.5e-01 |
| `y` max rel | 9.567e-02 | ~1.5e-01 | ≤ 2.2e-01 |

The bars are 3× A22's, i.e. twice the prediction. Justification, so they are not
round numbers: the brief's worst-cancellation figure puts R1's contribution to
`vn` at up to 2.2e-02 relative; added to A22's 3.506e-02 **linearly** (not in
quadrature - the pessimistic composition) that is 5.7e-02, and 1.05e-01 leaves a
further 1.8× on top of the pessimistic case.

**Which case will show it, predicted now:** all three band cases compare against
a reference with **no** bf16 state rounding - case 1 against the CPU fp32
recurrent `gdn_ref::step`, cases 2 and 3 against decode's own `gdn_step_M1` on
the device. **Case 3 (`C = 1` × 4096) is the sharpest and is predicted to grow
the most in proportion**: there the chunk algebra collapses so that today's
2.089e-02 is almost purely Q1/Q2, while the scan still rounds `S` and `vn` at
every one of the 4096 single-position sub-chunks - the new roundings arrive
undiluted. Predicted case 3 `gdn_state` max rel **~4e-02** (from 2.089e-02).

**The `A2·vn` term, priced separately as the brief asks.** R2 and R4 are its
two roundings. They reach **`gdn_o` and `y` only** - the state's path never
touches `A2` - so **the A2·vn term cannot dominate `gdn_state`'s band, and the
prediction is that it does not dominate `o`'s either**: it is a 2⁻⁹ relative
perturbation of one of `o`'s two terms, against an `o` that already inherits the
state's 3.5e-02. If `gdn_o`'s band grows materially more than `gdn_state`'s,
that is this term and it will be named as such.

**`gdn_chunk_test`'s own 2.5e-1 tripwire in `report()` is NOT moved**, and the
named risk of this design is that `y` (9.567e-02 today) sits only **2.6×** below
it: a 2.6× growth fails the test outright. That would be a finding and a revert,
not a raised tripwire.

**The arbiter is the token gate, whatever the band does:** `prefill_gate_test`
**94/94 determined rows exact on Vishva and 93/93 on RTN**, every `gdn_state`
cosine **> 0.999**; `prefill_determinism_test` 9 cases × 3 runs bitwise;
`prefill_consistency_test` **18/18** under A26; `gdn_chunk_test` **6/6**,
including cases 4, 5 and 6 - multi-chunk == single-chunk, the ragged
`C = 100` == 64 + 36, and the same walk twice - which stay **bit-identical**
because nothing in this design is order-dependent on the chunking or
non-deterministic. Today's worst printed cosines are **0.999896667** (Vishva,
L32) and **0.999884022** (RTN, L48), i.e. `1−cos ≈ 1.0e-04 / 1.2e-04; a 1.5×
growth of the state's L2 error takes `1−cos` to ~2.3e-04 / 2.6e-04, so the
prediction is **cosines stay above 0.9995**, with the 0.999 bar reached only at
a 3× growth. **A golden regression is a finding**: diagnose which stage and
which of R1-R4 moved, report it, revert. It is not a tolerance to loosen.

### 3.6 Time, and the decision rule

Instruction accounting per work-item per sub-chunk (SIMD16 instructions per
thread; **derived**, and the point of it is that the DPAS pipe is not what this
kernel will be limited by):

| block | instructions |
|---|---:|
| `Sb` write: 16 `rne` + 16 `store.slm.d16` + addressing | ~45 |
| `A2b` staging: 5 uints × (`vload2` + 2 `rne` + pack + store) + addressing | ~50 |
| the gate: 1 gather of `g_cum`, `gl`, two `exp` | ~25 |
| stage 1: 8 × (SLM `vload8` + global `vload8` + `dpas`) + addressing | ~50 |
| stage 1 epilogue: `u` `vload8`, 8 widen, 8 sub, 8 `rne` + `vstore8` (`VNb`), 8 mul + 8 `rne` + 8 `store.slm.d16` (`Dt`) | ~65 |
| stage 2 `q·S`: 8 × 3 + addressing | ~40 |
| stage 2: scale, 2.5 × 3 for `A2·vn`, `o` `vstore8` | ~25 |
| stage 3: 16 `mul` by `dl`, 4 A `vload8`, 8 B `vload8`, 8 `dpas`, addressing | ~50 |
| barriers and loop overhead | ~15 |
| **total** | **≈ 365** |

against A27's measured **≈ 12,600** (`docs/prefill-gdn-scan-2026-09-05.md`
§2.1) - **34.5× fewer instructions issued per thread**.

* At A27's **measured** 0.52 instructions/XVE/clock: 3.3273 × 365/12,600 =
  **0.096 ms**.
* The DPAS pipe's own floor: 26.5 `dpas.8x8` × 32 sub-chunks × 3072 threads =
  2.60e6 instructions × 8 clocks / 7.17e11 XVE-clocks/s = **0.029 ms**.
* Neither is what will be measured, because with 34× fewer instructions between
  the same 3 barriers per sub-chunk the kernel stops being issue-bound and
  becomes latency- and barrier-bound, on a launch that is still 1.5 waves.

**Point estimate 0.45 ms (derived)** - 4.7× the pure-issue figure, priced for
exactly that. Sensitivity stated rather than hidden: at A27's issue rate 0.10 ms,
at half of it 0.19 ms, at a fifth of it 0.48 ms. The bar does not depend on the
estimate.

Decision rule, fixed in advance (the brief's):

| `pf_gdn_scan`, ms/layer/chunk | verdict |
|---|---|
| **≤ 1.0** | **adopt** |
| 1.0 - 2.0 | adopt **if** the band and every gate hold; attribute the shortfall from the assembly |
| > 2.0, **or** any band or gate failure | **REVERT** to A27's kernel; the DPAS kernel stays in-tree as `pf_gdn_scan_dpas`, launched only by tests; report |

**`--pp 4096` consequence (derived)**, from task (b)'s measured row
**2929.9 ms / 1398.00 t/s** (device 0, iterate), item 0's measured −23.3 ms, and
96 × (3.3937 − t) with 3.3937 this session's control:

| `t` | `--pp 4096` ms | t/s |
|---|---:|---:|
| 1.0 (the bar) | 2676.8 | **1529.8** |
| 0.45 (point estimate) | 2624.0 | **1560.6** |

So the pre-registered row is **1530-1561 t/s, point estimate 1561** (derived),
taken **unmasked on device 0** for series continuity with 978.07 → 1304.06 →
1375.65 → 1377.20 → 1398.00. A27's arithmetic is untouched by any of this: the
non-GDN terms alone exceed what 1973 t/s allows, so "beat vLLM on prefill" stays
closed.

### 3.7 Invariants that must survive

Full suite green (60 + the long gate, which may SKIP). Decode re-proven after
the last code commit: **774 kernels / 19 modules**, `kernel_table_test`,
`replay_determinism_test`, `golden_gate_test` 94/94, and a decode bench row.
Box: `ZE_AFFINITY_MASK=1` for every piece of device work except the
series-continuous `--pp` and decode bench rows; 84 GB free; any disk error stops
the task.

## 4. RESULT - the DPAS scan is 3.997× and its time bar is HIT; it is REVERTED on the token gate

Everything in this section was measured after §3 was committed at `9de7783`.
Kernel at `src/kernels/prefill/pf_gdn_scan.cl`, reference edited with it at
`tests/prefill/gdn_chunk_ref.h`. **The mapping was not iterated: §3.3's text is
what was built and every number below is that build's first and only
measurement.**

### 4.1 Time, and the decision-rule verdict on time

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (81.5 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_scan` (DPAS), ms per GDN layer per chunk | **0.8491** | |
| the same before the rewrite, A27's own figure | 3.3273 | **3.919×** |
| the same before the rewrite, this session's control (§0) | 3.3937 | **3.997×** |
| **pre-registered bar** | ≤ 1.0 | **HIT** |
| point estimate | 0.45 | missed by 1.89× |
| rate on 9.66 GFLOP/layer/chunk (derived) | **11.38 TFLOP/s** | was 2.90 |

**On time alone this is an adopt.** The 1.89× between 0.45 and 0.8491 is the
latency/barrier term §3.6 named and priced but under-priced: at 34.5× fewer
instructions between the same 3 barriers per sub-chunk, and with 32 serially
dependent sub-chunks per work-group on a 1.5-wave launch, the kernel issues at
an effective ~0.11 instructions/XVE/clock against A27's 0.52 (derived from
365 × 32 × 3072 / 0.8491 ms = 4.22e10 /s against 7.17e11 XVE-clocks/s). The
DPAS pipe itself accounts for 0.029 ms of the 0.8491 - **3.4%** - so this kernel
is not compute-bound on DPAS and a further tile change would not help it.

Nothing else moved by more than session drift (measured, same run): `pf_gdn_A`
0.3317 (+0.8% vs the item-0 run), `pf_gdn_A2` 0.3482 (+0.7%), `pf_gdn_wu`
0.9412 (+0.7%), `pf_gdn_conv` 0.3304 (+0.4%), `pf_gdn_solve` 0.8044 (+1.4%),
`gemm` 2.7499 (+1.2%). Profiled walk **2860.0 → 2636.9 ms**; the same process's
own plain companion figure was 2784.7 ms (one run, **not** a `--pp` row - no
8-run median was taken for this kernel, because it does not ship).

### 4.2 The assembly, confirmed

`ocloc` on the committed `.cl` with `IGC_ShaderDumpEnable=1` (**measured**
compiler artifact):

| | `pf_gdn_scan` (DPAS) | `pf_gdn_scan_vec` (A27's) |
|---|---|---|
| `has_dpas` | **true** | absent |
| `dpas.8x8 (16\|M0)` in the Xe2 ISA | **28 static instructions** | **0** |
| `simd_size` | 16 | 16 |
| `grf_count` | 128 | 128 |
| `slm_size` | **22,400 B - exactly §3.3's budget** | 25,088 B |
| `barrier_count` | 1 | 1 |
| spill | **none** (`-abortOnSpill 4` is on the link line and the build succeeded) | none |

The first emitted instruction is
`dpas.8x8 (16|M0) r20:f null:f r46:bf r20.0:bf`, i.e. a real 8×8 systolic
bf16 op with an fp32 destination - not a lowered vector sequence.

### 4.3 The band: measured, against the prediction

`gdn_chunk_test`, **measured**, all six cases green:

| case 1 | A22's | §3.5 predicted | **measured** | pre-registered bar |
|---|---:|---:|---:|---:|
| `gdn_state` max rel | 3.506e-02 | ~5.3e-02 | **3.534e-02** (+0.8%) | ≤ 1.05e-01 ✓ |
| `gdn_state` mean rel | 1.197e-03 | ~1.8e-03 | **1.457e-03** (+21.7%) | ≤ 3.6e-03 ✓ |
| `gdn_o` max rel | 5.039e-02 | ~8e-02 | **5.143e-02** (+2.1%) | ≤ 1.5e-01 ✓ |
| `y` max rel | 9.567e-02 | ~1.5e-01 | **1.317e-01** (+37.7%) | ≤ 2.2e-01 ✓ |

| case 3 (`C = 1` × 4096) | A22's | §3.5 predicted | **measured** |
|---|---:|---:|---:|
| `gdn_state` max rel | 2.089e-02 | ~4e-02 | **3.161e-02** (+51.3%) |
| `gdn_state` mean rel | 8.519e-04 | - | **1.192e-03** (+39.9%) |

**Every band bar HELD, and the qualitative prediction that case 3 would show it
most in proportion HIT** (+51% against case 1's +0.8% on the max). The
quantitative prediction was 2-4× too pessimistic on the max and about right on
the mean. Cases 4, 5 and 6 stayed **bit-identical** - multi-chunk == single
chunk, the ragged `C = 100` == 64 + 36, and the same walk twice.

### 4.4 The token gate: TWO pre-registered bars fell - this is the finding

| gate | pre-registered | **measured on the DPAS build** | |
|---|---|---|---|
| `prefill_gate_test` (Vishva) determined rows exact | 94/94 | **94/94** | ✓ |
| worst printed `gdn_state` cosine, Vishva | > 0.999 | **0.998499499** (`code`, L60) | **✗** |
| `prefill_gate_rtn_test` determined rows exact | 93/93 | **92/93** | **✗** |
| worst printed `gdn_state` cosine, RTN | > 0.999 | 0.999851577 (`code`, L49) | ✓ |
| `prefill_determinism_test` | 9 × 3 bitwise | **9 × 3 bitwise** | ✓ |
| `prefill_consistency_test` (A26) | 18/18 | **18/18**, 543 + 558 = **1101 determined rows exact, 0 mismatches** | ✓ |
| `gdn_chunk_test` | 6/6 | **6/6** | ✓ |

**The failing row, in full.** RTN, `prose` prompt, generated position **18**:
engine **10932**, golden **11362**. The row is DETERMINED - the golden argmax is
unique, so the tie rule does not reach it - and its logit cosine is a healthy
0.999968099. The walk was already teacher-forced from position 15, where a
genuine tie (`353` vs `271`, golden logits 19.75 vs 19.75) had gone the other
way; positions 19-31 then match again, so this is one token, not a divergence.
`code` 32/32 and `cjk` 32/32 on the same checkpoint.

**The cosine bar fell harder than the token bar, and it fell in ONE place.**
Scored against **this session's own reverted-state control** (§5.1, which is the
only baseline that is same-code and same-session; see the note below):

| worst printed `gdn_state` cosine, per prompt | reverted (control) | **DPAS** | `1−cos` ratio |
|---|---:|---:|---:|
| Vishva `prose` | 0.999876346 (L60) | 0.999857456 (L49) | 1.15× |
| **Vishva `code`** | **0.999780657 (L60)** | **0.998499499 (L60)** | **6.84×** |
| Vishva `cjk` | 0.999905334 (L33) | 0.999911193 (L33) | 0.94× |
| RTN `prose` | 0.999905880 (L60) | 0.999904933 (L60) | 1.01× |
| RTN `code` | 0.999847086 (L60) | 0.999851577 (L49) | 0.97× |
| RTN `cjk` | 0.999904818 (L33) | 0.999908522 (L33) | 0.96× |

Five of the six are unmoved. **One - Vishva's `code` prompt at L60 - loses
6.84× of `1−cos`, i.e. ≈ 2.6× of the state's L2 relative error**, against §3.5's
predicted 1.5×, and that is the one that crosses the pre-registered 0.999.
The token flip is on the *other* checkpoint (RTN `prose`), so the two failures
are two independent single-point sensitivities rather than one effect seen
twice.

**Note on the baseline, reported not resolved.** Task (b) §10.2 published
0.999896667 (L32) / 0.999884022 (L48) as the worst Vishva / RTN cosines. This
session's reverted tree - whose only difference from task (b)'s final commit is
item 0's *bitwise* staging change - reproduces **A27's** published figures
instead, 0.999780657 (L60) and 0.999847086 (L60), to all nine digits. The
comparison above therefore uses this session's own control and not task (b)'s
table. The discrepancy in task (b)'s table is recorded here and not chased.

Either way the verdict is unchanged: `0.998499499 < 0.999` against every
candidate baseline, and 92/93 is 92/93.

The wider point stands and is the single most useful thing this measurement
produced: **`gdn_chunk_test`'s synthetic band under-reports this design's cost.**
Its state band moved +0.8% on the max and +21.7% on the mean while the real
checkpoint's 60th GDN layer lost 2.6× of L2 agreement on one prompt and one
token flipped on another. That is why the token gate is the arbiter and not the
band.

### 4.5 Verdict: REVERT, by the rule fixed in §3.6

> "> 2.0, **or** any band or gate failure → **REVERT** to A27's kernel; the DPAS
> kernel stays in-tree as `pf_gdn_scan_dpas`, launched only by tests; report."

Two gate bars fell, so the rule fires. It is **not** a defect: the kernel is
correct (three bit-identity structural cases, bitwise determinism, 18/18
consistency, 1101 determined consistency rows exact, 94/94 + 92/93 + every band
bar met), and what it costs is exactly the arithmetic §3.4 named before it was
built. **D2 is measured and closed at this precision.**

**The A2·vn term, priced separately as the brief asked.** R2 and R4 are its two
roundings and they reach `gdn_o`/`y` only - the state's path never touches `A2`.
The measurement bears that out: `gdn_o`'s max rel moved +2.1% and `y`'s +37.7%
while `gdn_state`'s moved +0.8%, and the gate row that failed is downstream of
the *state*, not of `o` (the state cosine at L60 is what collapsed, and `o` is
consumed and discarded within a layer). **The A2·vn term does not dominate the
band and it is not what cost the token.** Had only R2/R4 existed - i.e. had `S`
and `D` stayed fp32 - the state would have been untouched; that is not a
buildable variant, because stages 1 and 3 are where the FLOPs are.

### 4.6 What would recover it, priced and NOT built

**D3 - split-bf16.** Carry `S` and `D` as a bf16 hi/lo pair (`hi = rne(x)`,
`lo = rne(x − hi)`) and issue two DPAS chains per contraction, so the operands
hold ~16 mantissa bits instead of 8 and R1/R3 fall from 2⁻⁹ to ~2⁻¹⁷ -
**below** Q1-Q4's own 2⁻⁹, i.e. invisible in the composed band. Cost, derived:

| | D2 (measured) | D3 (derived) |
|---|---:|---:|
| `dpas.8x8` per work-item per sub-chunk | 26.5 | **50.5** (stage 1 and 2's `q·S` and stage 3 all doubled; the `A2·vn` term stays single) |
| SLM | 22,400 B | **35,328 B** (`Sb` 17,408 + `VNb` 4,352 + `Dt` 8,448 + `A2b` 5,120) |
| resident work-groups per Xe-core | 4 (thread-limited) | **3 (SLM-limited)** → 2 waves instead of 1.5 |
| ms/layer/chunk | 0.8491 | **1.3-1.6 (estimated)** |

That is inside A28's 1.0-2.0 adopt-and-attribute band and still ≈ **2.2×** the
vector kernel, worth ≈ −180 ms on `--pp 4096` (derived). **It is a NEW design
and therefore a new pre-registration, not an iteration of this one**, and it is
not built here.

## 5. The revert, and the proof that it is complete

`pf_gdn_scan.cl` now holds two entry points, and §3.3's swap rule was followed:
**`pf_gdn_scan` is A27's vector kernel and is what `gdn.cc` binds**;
**`pf_gdn_scan_dpas` is the DPAS kernel, kept in-tree and launched by nothing**.
The production kernel's body is **byte-identical** to `a89311f`'s (checked with
`diff` over the extracted function: 149 lines, no difference), and
`src/runtime/prefill/gdn.cc` is untouched.

The proof that the revert is complete is the band coming back to the digit -
`gdn_chunk_test`, **measured** at the reverted state:

| case | tensor | max rel | mean rel | A22's / A27's / A28's / task (b)'s |
|---|---|---:|---:|---|
| 1 | `gdn_state` | **3.506e-02** | **1.197e-03** | 3.506e-02 / 1.197e-03 |
| 1 | `gdn_o` | 5.039e-02 | 1.228e-03 | identical |
| 1 | `y` | 9.567e-02 | 9.790e-04 | identical |
| 2 | `gdn_state` | 3.506e-02 | 1.197e-03 | identical |
| 3 | `gdn_state` | 2.089e-02 | 8.519e-04 | identical |

Even the diagnostic counts return: 17,234 / 17,236 `gdn_o` words over 1e-2 in
cases 1 / 2 and 237,692 / 237,664 for `y`, which are task (b)'s to the unit.
Cases 4, 5 and 6 bit-identical as before.
