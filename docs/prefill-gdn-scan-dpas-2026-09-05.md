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
