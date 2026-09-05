# `pf_gdn_wu` at SIMD32, and output tiles for `pf_gdn_A2` and `pf_gdn_A` - ruling A28 option (b)

Ruling A28 (`interfaces.md`) priced three GDN levers and built none of them; the
operator's 2026-09-05 ruling took option **(b+c)** and sent the first two, plus
the `pf_gdn_A` twin, to this task. It follows
`docs/prefill-gdn-wu-conv-2026-09-05.md`'s shape exactly: **every
pre-registration section below was committed before the change it describes was
written**, everything after it is labelled measured / derived / estimated, and
no quantity appears twice with two values.

Grade: **iterate** throughout, and the reason is the gate task's, not a new one.
The two desktop processes named in
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/progress.md` - baobab (285644)
and ptyxis (285678) - still hold render-node fds on **both B70s**, measured at
the start of this task by walking `/proc/*/fd`:

```
285644 baobab  /dev/dri/renderD129  2026-09-05 19:58:59
285644 baobab  /dev/dri/renderD130  2026-09-05 19:58:59
285678 ptyxis  /dev/dri/renderD129  2026-09-05 19:59:00
285678 ptyxis  /dev/dri/renderD130  2026-09-05 19:59:00
```

Four holders on the two cards, so the record-grade condition is not met and
every row here is **iterate**, as gate row 1 was. The holder count is measured
again immediately before the `--pp 4096` row (§8) and graded there.

Timed `--pp` rows are 8 independent runs, median, with min/max/spread beside
them; the drop-3 half of the project's iterate convention does not apply to a
`--pp` row, for the reason `docs/prefill-pp-attribution-2026-09-05.md` §1 gives
(each run is a separate process with a cold model load, so there is no warm-up
sequence inside a process to drop). Per-kernel figures are single
`B70_PREFILL_PROFILE=1` runs, as in the previous two documents.

## 0. The control, measured before anything was changed

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8` at `e6da455`, one run, **measured**, beside A28's published figures
(`42921d2`, taken the same way on the same device):

| ms/layer/chunk (column ÷ 96) | this session (`e6da455`) | A28's (`42921d2`) | drift |
|---|---:|---:|---:|
| `pf_gdn_A` | **0.6136** | 0.6136 | 0.00% |
| `pf_gdn_A2` | **1.0314** | 1.0313 | +0.01% |
| `pf_gdn_wu` | **0.9315** | 0.9298 | +0.18% |
| `pf_gdn_conv` | 0.3294 | 0.3287 | +0.21% |
| `pf_gdn_solve` | 0.7458 | 0.7528 | −0.93% |
| `pf_gdn_scan` | 3.3883 | 3.3968 | −0.25% |
| profiled walk total, ms | **2938.4** | 2938.7 | −0.01% |

**There is no session drift to correct for**: the two sessions' controls agree
to 0.25% on every GDN kernel and to 0.01% on the walk. Bars below are therefore
stated against A28's published figures, which is what the brief pre-registers,
and the same-session control is reported beside every result.

### 0.1 The machine's own numbers, used by every derivation below

Fixed here so that no later section re-derives them differently:

* **256 XVEs at 2800 MHz = 7.17e11 XVE-clocks/s**, 8 hardware threads per XVE =
  **2048 thread slots**, i.e. 32 Xe-cores of 8 XVEs × 8 threads = **64 thread
  slots and 128 KB of SLM per Xe-core** (`docs/prefill-gdn-scan-2026-09-05.md`
  §2.1 and `docs/prefill-gdn-wu-conv-2026-09-05.md` §1.1/§3.2 fix all of these;
  nothing new is claimed here).
* A GRF register is **64 B**, so one fp32 vector value costs **1 register at
  SIMD16 and 2 at SIMD32** - read off the ISA dumps used below, where a
  `load.slm.d32.a32 (32|M0) r8:2` returns 32 lanes × 4 B into two registers.
* The link line carries **`-abortOnSpill 4`** (measured, in every entry point's
  `.full_options` in the IGC dump), so a spill is a **build failure**, not a
  silent regression. That is what makes "any spill → revert" a decidable rule.

---

# PART 1 - `pf_gdn_wu` at SIMD32

## 1. PRE-REGISTRATION (committed before the change was built)

### 1.1 The defect, and the claim that has to be true before the line is changed

`docs/prefill-gdn-wu-conv-2026-09-05.md` §4.2 measured it: the tiled
`pf_gdn_wu` carries `__attribute__((intel_reqd_sub_group_size(SG)))` with
`SG = 16`, copied from `pf_gdn_scan.cl` without being priced, while the legacy
kernel in the same file compiles SIMD32. Per-lane work fell 5.06× and issued
instructions only 2.53×, and the whole of that gap is the sub-group width.

The brief requires the width-agnosticism claim to be **verified by reading the
kernel before anything is built**, and it is verified here. Every index in
`pf_gdn_wu` is arithmetic on `get_local_id(0)` and none of it calls a sub-group
builtin:

```
sgid = lid / 16 ;  lane = lid % 16 ;  cs = sgid & 1 ;  b = sgid >> 1
c0   = cs * 32 + lane * 2           ;  dcol = get_group_id(2) * 64
il   = b * 4 ;  ih = 60 - b * 4     ;  jl, jh derived from il, ih and L
```

At SIMD32 one hardware thread covers `lid ∈ [32t, 32t + 32)`, so:

* `b = (lid / 16) >> 1 = t` for **both** 16-lane halves - `b` is thread-uniform,
  therefore `il`, `ih`, `jl`, `jh` and both `vload4(TsT + j * 68 + il|ih)` reads
  stay thread-uniform, and the two phase loops have **identical trip counts on
  every lane**, so the widening introduces no divergence;
* `cs` is 0 on the low half and 1 on the high half, so `c0` runs
  `0, 2, … , 30` then `32, 34, … , 62`: the thread covers the work-group's **64
  contiguous columns exactly once, in lane order**. `vload2(vbs + j*64 + c0)` and
  the `kbs` twin therefore read 256 contiguous bytes across the thread instead
  of 128, and the `u`/`w` stores write 128 contiguous bytes per position instead
  of 64 - both stay coalesced, and neither the SLM footprint (50,688 B) nor the
  staging loops (`for (p = lid; p < …; p += 256)`) depend on the width at all.

**The claim is true**, so this item proceeds. `SG` keeps the value 16 because it
is the *mapping* granularity - 16 lids per column-half - and a new
`WU_SIMD = 32` is what the attribute takes; changing `SG` itself would move
`c0` off the end of the 64-column half and is not the change being made.

### 1.2 Pre-registered outcomes

**Time.** `pf_gdn_wu` **≤ 0.6 ms per GDN layer per chunk** at C = 2048 - the bar
A28 §4.1 missed, restated unchanged. Point estimate **0.465 ms**, the figure
`docs/prefill-gdn-wu-conv-2026-09-05.md` §4.2 derived from the measured issue
rate (24,576 threads × 1642 loop instructions = 4.03e7 instructions at the
measured 8.68e10 instructions/s). Decision rule, fixed in advance (the brief's):

| outcome | verdict |
|---|---|
| ≤ 0.6 ms **and** bitwise identical | **adopt** |
| > 0.6 ms **and** bitwise identical, but faster than 0.9298 | adopt, and attribute |
| any spill, or any non-identical word | **revert and report** |

**Numerics: the bar is bitwise identity.** Removing a sub-group-size attribute
moves no rounding point and re-associates nothing - the expression each work-item
evaluates is unchanged, only the number of lanes an issued instruction covers
changes. So the pre-registered outcome is that `w` and `u` are **bit-for-bit
identical to `pf_gdn_wu_legacy`** at every width `gdn_wy_test` runs (C = 256,
C = 100 ragged, C = 4096) - the same case-6 bar A28 already carries, which means
this item needs **no new test**: it either keeps passing or it is a finding.

**Named risk, stated before the build:** 32 fp32 accumulators (`au[8][2]`,
`aw[8][2]`) cost 2 registers each at SIMD32 = 64 of the 128 GRF, plus `tv[8]`
(16) and the per-`j` operands (8). If IGC cannot fit it, `-abortOnSpill 4` fails
the build; that is the revert-and-report branch, and it is a **build** outcome
rather than a measurement, so it costs no device time. The second half of the
risk is that halving the thread count halves what is available to hide the SLM
latency §4.2 named as term 2 - that one shows up as a time, not as an error.

**Assembly to be read and reported** (the brief asks for it by name): SIMD width,
`grf_count`, spill/scratch size, and the FMA density of both phase loops, from
`IGC_ShaderDumpEnable=1` on the committed `.cl`, with the legacy entry point in
the same binary as the control.

### 1.3 Invariants that must survive (all three items)

Full suite green (61 tests + whatever this task adds; `prefill_gate_long_test`
may SKIP - `oracle-out-long` is still the operator's `docker run`). Decode
re-proven after the last code commit: **774 kernels / 19 modules**,
`replay_determinism_test`, golden **94/94 + 93/93**, `prefill_determinism_test`
9 × 3, `prefill_consistency_test` 18/18, and a decode bench row. Box:
`ZE_AFFINITY_MASK=1` on every piece of device work except the `--pp 4096` row of
§8, which is taken **unmasked** so that it is series-continuous with the gate's
1377.20 t/s (device 0); 84 GB free at the start; any disk error stops the task.

## 2. RESULT - `pf_gdn_wu` at SIMD32 is 1.24× SLOWER, bitwise, and is REVERTED

Everything in this section was measured after §1 was committed at `37208a4`.
The change built and ran; it is the *time* that decided it.

### 2.1 Time, and the decision-rule verdict

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (110.5 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_wu` at SIMD32, ms per GDN layer per chunk | **1.1505** | |
| the same at SIMD16, A28's (`42921d2`) | 0.9298 | **1.237× slower** |
| the same at SIMD16, this session's control (§0) | 0.9315 | 1.235× slower |
| pre-registered bar | ≤ 0.6 | **MISS by 1.92×** |
| point estimate (A28 §4.2's derivation) | 0.465 | missed by 2.47× |
| rate on 1.636 GFLOP/layer/chunk (derived) | **1.42 TFLOP/s** | was 1.76 |

**Decision-rule verdict: > 0.6 ms and slower than 0.9298 → REVERT.** The
attribute is back at `SG` (16) in `src/kernels/prefill/pf_gdn_wy.cl`; the file
differs from `e6da455` only in the comment block that now records this
measurement, so that the next reader does not re-derive the same lever. The
mapping was not iterated: the one-line change was built once and measured once.

Nothing else in the walk moved (measured, same run): `pf_gdn_A` 0.6145,
`pf_gdn_A2` 1.0321, `pf_gdn_scan` 3.3989, `pf_gdn_conv` 0.3289 - all within
0.3% of §0's control, so the +0.22 ms is `pf_gdn_wu`'s and nothing else's.

### 2.2 Why the derivation was wrong, read off the Xe2 assembly

`ocloc` was re-run on the SIMD32 `pf_gdn_wy.cl` with `IGC_ShaderDumpEnable=1`,
and the numbers below are counted in the ISA, not inferred from the timing. The
legacy entry point is in the same binary, so this is a controlled comparison.

**No spill.** `-abortOnSpill 4` is on the link line and the build succeeded;
`grf_count 128`, `slm_size 50688`, `barrier_count 1`, `simd_size 32` - the
metadata says the kernel is exactly what was asked for. So the first half of
§1.2's named risk did **not** fire in the form it was named.

**It fired in a form the pre-registration did not name: register moves.**

| loop body | SIMD16 (A28 §4.2) | SIMD32 (measured here) |
|---|---:|---:|
| phase A - instructions | 45 | **84** |
| phase A - `mad` | 32 | 32 |
| phase A - FMA density | 71% | **38%** |
| phase A - added `mov` | 0 | **34** |
| phase B - instructions | 26 | **27** |
| phase B - `mad` | 16 | 17 |

Phase B is untouched: 1 × `load.slm.d32x4` (the block's four `T` values),
2 × `load.slm.d32x2` (the lane's two columns of `vb` and of `kb`), 16 `mad` - the
vector shapes §3.2 of the previous document fixed, at the new width. **Phase A
grew by 39 instructions, 34 of them plain `mov`.** At SIMD32 a fp32 vector value
is 2 GRF, so the 32 accumulators alone hold 64 of the 128 registers and the
eight `T` values another 16; IGC fits it without spilling by shuffling
registers around the two 16-`mad` groups, and those shuffles are issued
instructions like any other.

**The arithmetic, with both terms:**

| | SIMD16 | SIMD32 |
|---|---:|---:|
| threads per layer per chunk | 49,152 | 24,576 |
| loop instructions per thread, mean over `b` (derived) | 1,642 | **2,376** |
| loop instructions issued (derived) | 8.07e7 | **5.84e7** |
| ratio | | **1.38×**, not 2× |
| measured ms | 0.9298 | **1.1505** |
| issue rate, instructions/XVE/clock (derived) | 0.121 | **0.071** |

The per-thread count is `(4b+4)·A + (60−8b)·B` summed over the eight `b` values a
work-group's threads take, with `A`/`B` the phase bodies above; at SIMD32 one
thread covers two of the old `b`-pairs' lanes, so the thread count halves while
each thread's body grows 1.45×. The widening therefore bought only **1.38×** of
issued instructions where the pre-registration assumed 2×, and it **spent
1.71× of issue rate** to get it.

**The issue rate is the second half of the named risk, and it is the larger
term.** `pf_gdn_wu`'s SLM is 50,688 B, so an Xe-core holds 2 work-groups at
either width; that is **32 resident threads at SIMD16 and 16 at SIMD32** of the
64 the Xe-core has - 4 threads per XVE against 2. A28 §4.2 term 2 measured that
this kernel's `j` loops are **not unrolled** (their bounds `jl`/`jh` are runtime
values), so every iteration is one un-pipelined SLM round trip with 27-84 issue
slots to cover it, and halving the threads halves what is available to cover it
with. 0.121 → 0.071 is that, measured.

So the lever A28 priced at −45 ms/4096 is worth **+21 ms** instead, and the
reason is not the one that was named as most likely.

### 2.3 Numerics: bitwise, as pre-registered - which is why the revert is clean

`gdn_wy_test` at `ZE_AFFINITY_MASK=1`, **measured on the SIMD32 build**, all
five bars and case 6 green at all three widths:

| width | bar | measured |
|---|---|---|
| C = 256 | `w`, `u` bit-identical to `pf_gdn_wu_legacy` | **1,572,864 words each, identical** |
| C = 100 (ragged, L = 36) | the same | **614,400 words each, identical** |
| C = 4096 | the same | **25,165,824 words each, identical** |

`u` 0 bf16 ulp (bit-identical) against `gdn_chunk_ref::wu` at every width; `w`
max 1 ulp on `|ref| ≥ rms/4` at C = 256 and C = 100 and 2 at C = 4096, against
the bar of 2 - the same table A28 §4.3 recorded, to the word count. **The
sub-group width moves no rounding point, and the measurement says so**, which is
what makes the revert a pure performance decision with nothing to diagnose.

### 2.4 What this leaves

`pf_gdn_wu` stays at **0.9298 ms/layer/chunk** (A28's figure; §0's control
re-measured 0.9315 in this session and neither corrects the other). The
remaining named lever on this kernel is A28 §4.2's second: unrolling the `j`
loop by a fixed factor plus a remainder, so that the SLM round trips pipeline.
It is **not** attempted here - it is a different defect, and this task's budget
goes to items 2 and 3.

---

# PART 2 - `pf_gdn_A2`

## 3. PRE-REGISTRATION (committed before the rewrite was built)

### 3.1 The defect, read off the kernel AND off the Xe2 assembly

`src/kernels/prefill/pf_gdn_wy.cl:152-177`. Grid (48 heads, nchunks) of 256, and
the 64 × 64 triangular product is written `for (p = lid; p < CT*CT; p += 256)` -
**one output `A2[i][j]` per work-item per step, with no tile.** Each output runs
the whole 128-term `band_dot` out of bf16 SLM: per `k` a `qs` word and a `ks`
word, each widened where it is used, one `* Q_SCALE` and one `fma`.
`docs/prefill-gdn-wu-conv-2026-09-05.md` §6.1 already named it - *"the same
no-tile pattern `pf_gdn_wu` had, at the same 0.79 TFLOP/s"* - and the assembly
now says exactly where the time goes. `ocloc` on the unmodified `pf_gdn_wy.cl`
with `IGC_ShaderDumpEnable=1`, **measured**:

| | `pf_gdn_A2` | `pf_gdn_A` |
|---|---:|---:|
| `simd_size` / `grf_count` / `slm_size` | **32** / 128 / **33,024** | **16** / 128 / 16,896 |
| hot-loop body, instructions | **610** (8 bands) | **898** (16 bands) |
| `mad` in it | 64 | 130 |
| **FMA density** | **10.5%** | **14.5%** |
| bf16 widen (`mov` uw→d + `shl` 16) | 288 = **47%** | 516 = **57%** |
| `* Q_SCALE` (`mul`) | 72 = 12% | 5 (folded: `ascale` is 1.0f) |
| SLM `load.slm.d32` (2 words each) | 64 | 131 |
| addressing (`or`/`add`/`add3`/`shl`) | ~111 = 18% | ~80 = 9% |

Two mechanisms, both measurable, and the second is the one A28 did not have:

1. **No output tile → the widen is 47% of the loop.** Every operand word is
   widened at the point of use, and with one output per work-item nothing is
   reused: `q_i[k]` is re-widened for every `j` and `k_j[k]` for every `i`.
2. **`pf_gdn_A2` runs at 37.5% occupancy and `pf_gdn_A` at 100%.** SLM is
   33,024 B (two 64 × 128 bf16 stagings plus `gcs`), so an Xe-core's 128 KB
   holds **3** work-groups; at SIMD32 a 256-work-item group is 8 threads, so
   **24 of the Xe-core's 64 thread slots are resident**. `pf_gdn_A` stages one
   operand (16,896 B), is 4 work-groups × 16 threads at SIMD16, and fills all
   64. That is the whole reason two kernels doing the same 0.8 GFLOP measure
   1.0313 and 0.6136 ms.

The two combine into an issue-rate figure that closes the arithmetic
(**derived**, from the loop bodies above, the mask's live fraction, and the
measured times):

| | `pf_gdn_A2` | `pf_gdn_A` |
|---|---:|---:|
| threads per layer per chunk | 12,288 (SIMD32) | 24,576 (SIMD16) |
| loop iterations executed per thread (mask, derived) | 12 of 16 | 10 of 16 |
| instructions issued per layer per chunk | **1.90e8** | **2.21e8** |
| measured ms | 1.0314 | 0.6136 |
| **issue rate, instructions/XVE/clock** | **0.257** | **0.502** |

`pf_gdn_A` at 100% occupancy issues at 0.502 - the highest rate any GDN kernel
in this project has measured - and `pf_gdn_A2` at 37.5% issues at half of it.
**Occupancy is a lever here, and the rewrite has to keep it, not spend it.**

### 3.2 The rewrite, fixed here before it is built

Unchanged: the algebra; the mask `j <= i` **including the diagonal** (the header
block's mask pair); the `* Q_SCALE` applied to `q` in fp32 at read (P6); `k`
unscaled (P7); the 16 × 8 band tree and its collapse order; `exp(gc[i]-gc[j])`
as one device `exp` of one difference; the output layout `A2[i][j]` at
`((chunk*48 + h)*64 + i)*64 + j` and its exact `0.0f` outside the mask.

Changed:

* **The grid gains a third dimension: (48 heads, nch, 4).** `z` names a
  **quadrant** - `iz = z >> 1` picks `i ∈ [32·iz, 32·iz+32)`, `jz = z & 1` picks
  `j ∈ [32·jz, 32·jz+32)`. A work-group therefore owns 1024 of the 4096 outputs
  and stages only the 32 `q` rows and 32 `k` rows it needs.
* **The quadrant `(iz = 0, jz = 1)` is entirely masked** - every one of its
  outputs has `j ≥ 32 > i`, so `j <= i` is false - and it writes its 1024
  `0.0f`s and returns without staging anything. The other three quadrants
  compute 3 × 1024 = **3072 dots**, which is what the current kernel's
  predication already costs (12 of 16 iterations × 4096 = 3072, §3.1); the
  quadrant split makes that explicit instead of paying it in dead lanes.
* **Operands staged as fp32, so the widen leaves the inner loop entirely.**
  `qs[32][128]` holds `f32(q_word) * Q_SCALE` and `ks[32][128]` holds
  `f32(k_word)` - the same fp32 values the current kernel forms inside the
  `fma`'s first argument (the assembly shows the `mul` and the `mad` as separate
  instructions, so the product is rounded to fp32 before the `fma` either way).
  **SLM: 2 × 32 × 128 × 4 = exactly 32,768 B**, which is 4 work-groups per
  128 KB Xe-core.
* **`intel_reqd_sub_group_size(16)`, chosen deliberately** - the brief asks for
  the reason and this is it. At 32,768 B of SLM an Xe-core holds 4 work-groups
  either way, so the width decides residency: **4 × 16 = 64 threads at SIMD16,
  the Xe-core's whole budget, against 4 × 8 = 32 at SIMD32.** §3.1 measured what
  that is worth on this exact instruction mix - 0.502 against 0.257
  instructions/XVE/clock - and item 1 measured the other side of the same trade
  on `pf_gdn_wu`, where SIMD32 halved residency and cost 1.71× of issue rate for
  1.38× of issued instructions (§2.2). SIMD16 also halves the register cost of
  the band tree, which is what makes the tile fit with room to spare.
* **`gcs` leaves SLM** and is read from global per work-item (2 `i` values and 2
  `j` values, clamped to `< L`), because 32,768 + 256 B would be 3 work-groups
  per Xe-core instead of 4 - the same cliff §3.1 measured on the current kernel.
* **Output tile: 2 positions × 2 columns = 4 outputs per work-item.**
  `bi = lid >> 4` and `bj = lid & 15`, so `i ∈ {32iz+2bi, +1}` and
  `j ∈ {32jz+2bj, +1}`. `bi` is **uniform inside a SIMD16 thread**, so the `qs`
  reads are a broadcast and the 16 lanes' stores cover 32 contiguous fp32 of one
  `A2` row - a coalesced 128 B global write per position.
* **The contraction reads four fp32 rows per band with `vload4`-shaped access**
  (`qs[i0]`, `qs[i0+1]`, `ks[j0]`, `ks[j0+1]` at `k = 8b … 8b+7`), giving
  4 `fma` per 4 operand values. Density rises 10.5% → **~67%** (derived).
* **The band tree is written out as its own expression tree, not as an array.**
  `p[b] += p[b+8]`, then `+= p[b+4]`, `+= p[b+2]`, `+= p[b+1]` is exactly
  `((P0+P8)+(P4+P12)) + ((P2+P10)+(P6+P14))` plus
  `((P1+P9)+(P5+P13)) + ((P3+P11)+(P7+P15))`, summed. Written in that order the
  live set is **4 `float4` values**, i.e. 16 registers at SIMD16, instead of the
  16 the array form would pin. **The expression is the same one, term for term
  and parenthesis for parenthesis** - that is the whole numerics argument.

Nothing else changes. The rounding points do not move and **no sum is
re-associated**; this is a re-partition of a loop nest plus a change of
container for two operands, and every output is the same function of the same
inputs.

### 3.3 Pre-registered outcomes

**Time.** `pf_gdn_A2` **≤ 0.5 ms per GDN layer per chunk** at C = 2048 (≥ 2×
the measured 1.0313 - the brief's bar, derived from the wu result). Point
estimate **0.23 ms**, derived from the instruction count:

* per work-item: staging 16 × ~13 ≈ 208, contraction 16 bands × (32 `fma` +
  8 `vload4` + ~8 address) ≈ 828, tree 60, epilogue and prologue ≈ 70 →
  **~1106 instructions**;
* 4608 live work-groups (3 quadrants × 48 × 32) × 16 threads × 1106 =
  **8.15e7 instructions issued**, against the current 1.90e8;
* at `pf_gdn_A`'s **measured** 0.502 instructions/XVE/clock - the same 100%
  occupancy and a denser mix - that is **0.227 ms**.

The estimate's sensitivity is stated with it rather than hidden: at the current
kernel's 0.257 issue rate it would be 0.44 ms, still inside the bar, so the bar
does not depend on the issue rate improving.

Decision rule, fixed in advance (the brief's):

| speed-up vs 1.0313 ms | verdict |
|---|---|
| ≥ 2× (≤ 0.5157 ms) | adopt |
| 1.2-2× | adopt, and attribute the shortfall from the assembly |
| < 1.2× | report, do not tune |

**Numerics - the pre-registered outcome is BITWISE IDENTITY, and the brief's
band is kept as the fallback bar.** The brief allows a re-association if a tile
forces one; **under §3.2's mapping no tile forces one.** Each output is still
the same 16 × 8 band tree over the same fp32 operand values, in the same order,
times the same `exp`; what changed is which work-item owns which output, how
many times each operand is loaded, and whether the bf16→fp32 widen happens at
staging or at use (a bf16 value is exact in fp32 by construction, so the
container is not a rounding point). So:

* **Primary bar: `pf_gdn_A2` bit-identical to `pf_gdn_A2_legacy`** - the kernel
  exactly as it stands - over every one of the `nchunks · 48 · 64 · 64` fp32
  entries, at C = 256, at the ragged C = 100 and at C = 4096. Proved the way
  `pf_gdn_wu`'s was: the pre-rewrite kernel stays in the same `.cl`, launched by
  nothing but `gdn_wy_test`, and the two output buffers are pre-filled with
  **different** patterns so a word neither kernel wrote fails the `memcmp`
  rather than passing it - which makes the bar a coverage bar for the new 3-D
  grid as well as an equality one.
* **Fallback bar, kept because a pre-registration is not allowed to move after
  the fact:** if any word differs, `gdn_chunk_test` case 1's `gdn_state`
  **max rel ≤ 7.0e-02 and mean rel ≤ 2.4e-03** (A22's is 3.506e-02 /
  1.197e-03), and the arbiter is the token gate.
* `gdn_wy_test`'s existing case-4 bars unchanged: ≤ 4 fp32 ulp against
  `gdn_chunk_ref::mat_A2`, the device's non-zero count equal to
  `Σ 48·L(L+1)/2`, the diagonal non-zero on every row, and exactly `0.0f`
  everywhere above the diagonal and outside `[0,L)²`.
* **The arbiter, whatever the band does:** `prefill_gate_test` **94/94 on Vishva
  and 93/93 on RTN**, every `gdn_state` cosine > 0.999;
  `prefill_determinism_test` 9 × 3 bitwise; `prefill_consistency_test` 18/18
  under A26; `gdn_chunk_test` 6/6.
* **A golden regression is a finding**: diagnose it, report it, stop. It is not
  a tolerance to loosen.

`tests/prefill/gdn_chunk_ref.h` documents this kernel's reduction order and the
two files carry a standing "must be edited together" rule; **the order does not
change**, and that is what gets recorded there.

## 4. RESULT - `pf_gdn_A2` at 1.751×, bitwise

Everything in this section was measured after §3 was committed at `07cb7a9`.
Kernel at `src/kernels/prefill/pf_gdn_wy.cl` (entry `pf_gdn_A2`, helpers
`band4`/`tile_dot`), grid in `src/runtime/prefill/gdn.cc`, bar asserted by
`tests/prefill/gdn_wy_test.cc` case 7.

### 4.1 Time, and the decision-rule verdict

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (56.5 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_A2`, ms per GDN layer per chunk | **0.5889** | |
| the same before the rewrite, A28's (`42921d2`) | 1.0313 | **1.751×** |
| the same before the rewrite, this session's control (§0) | 1.0314 | 1.751× |
| pre-registered bar | ≤ 0.5157 (≥ 2×) | **MISS by 1.14×** |
| point estimate | 0.23 | missed by 2.56× |
| rate on 0.818 GFLOP/layer/chunk (derived) | **1.389 TFLOP/s** | was 0.79 |

**Decision-rule verdict: 1.751× falls in the 1.2-2× band → ADOPT, and attribute
the shortfall from the assembly.** The mapping was not iterated: §3.2's mapping
is what was built and 0.5889 is that build's first and only measurement.

Per-chunk consequence: 48 × (1.0313 − 0.5889) = **−21.2 ms/chunk**, **−42.5 ms
over a 4096-token prefill** (derived), against the −35 ms/chunk A28 §6.1 derived
for `A2` and `A` together.

Nothing else moved (measured, same run): `pf_gdn_A` 0.6157, `pf_gdn_wu` 0.9310,
`pf_gdn_scan` 3.3845, `pf_gdn_conv` 0.3289 - all within 0.4% of §0's control.

### 4.2 What the rewrite bought, and where the shortfall is - off the Xe2 ISA

`ocloc` on the committed `pf_gdn_wy.cl` with `IGC_ShaderDumpEnable=1`, with
`pf_gdn_A2_legacy` in the same binary as the control. **Measured:**

| | legacy | tiled |
|---|---:|---:|
| `simd_size` / `grf_count` / `slm_size` | 32 / 128 / 33,024 | **16 / 128 / 32,768** |
| resident work-groups per Xe-core (derived) | 3 | **4** |
| resident threads of the Xe-core's 64 (derived) | 24 (**37.5%**) | **64 (100%)** |
| spill | none | **none** |
| contraction block, instructions | 610 per 8 bands | **986 for all 16** |
| `mad` in it | 64 | **520** |
| **FMA density** | **10.5%** | **53%** |
| operand reads | 64 × `load.slm.d32` (2 bf16 words) | **128 × `load.slm.d32x4`** |
| bf16 widen in the loop | 288 instructions | **0** |
| `* Q_SCALE` in the loop | 72 | **0** (folded into staging) |

**Everything the tile was designed to do, it did.** The widen is gone from the
contraction, `Q_SCALE` is gone with it, all 128 operand reads are the four-wide
vector loads the 2 × 2 tile was for, and SLM landed at **exactly 32,768 B**, so
the kernel now fills all 64 of an Xe-core's thread slots instead of 24.

**The shortfall is the staging loop, and it is one number.** Its body is
**53 instructions per iteration** (measured), run 16 times = **848 instructions
per work-item**, against ~986 for the whole contraction. Per staged value that
is **26.5 instructions**, and the pre-registration priced it at 13:

| per staging iteration (one `q` value and one `k` value) | measured |
|---|---:|
| `load.ugm.d16u32` - one scalar 16-bit global load each | 2 |
| `mach` + `mul` + `add3`/`or`/`add` - the 64-bit global address | ~14 |
| `goto`/`join` - the `gi < L` and `gj < L` guards, as BRANCHES | **9** |
| `shl`/`mov` - the bf16 widen, and `mul` by `Q_SCALE` | ~9 |
| `store.slm.d32` | 2 |

Two things make it expensive and both were foreseeable and not foreseen: the
guard is written as a ternary around a *load*, so IGC has to branch rather than
select; and the loop stages **one value per array per iteration**, so the 64-bit
address arithmetic is paid per value instead of per four. On top of that the
quadrant split stages 3 × (32 + 32) rows per (head, chunk) where the old kernel
staged 1 × (64 + 64) - **1.5× more staged values in total**.

**The arithmetic, composed (derived, per head per chunk):**

| | legacy | tiled |
|---|---:|---:|
| work-groups | 1 | 4 (3 live + 1 zero-writer) |
| threads | 8 (SIMD32) | 64 (SIMD16) |
| staging instructions per work-item | ~640 | **848** |
| contraction + epilogue per work-item | ~15,480 (12 of 16 live) | **~986** |
| instructions issued | **123,700** | **~87,000** |
| ratio | | **1.42×** |
| measured ms | 1.0314 | **0.5889** |
| issue rate, instructions/XVE/clock (derived) | 0.257 | **0.316** |

1.42× of instructions × 1.23× of issue rate = **1.75×**, which is the measured
number. The occupancy lever paid (0.257 → 0.316, toward `pf_gdn_A`'s 0.502) and
the density lever paid (10.5% → 53%); **the staging spent 47% of the result.**

**Priced, and NOT applied to this kernel:** clamping the row index and using a
select instead of a branch, and staging four values per iteration with one
vector global load and one `store.slm.d32x4`, would take the staging body from
26.5 instructions per value to ~5. That is ~680 of the ~1800 instructions a
work-item issues, i.e. **~1.6× more on this kernel, ≈ 0.37 ms/layer/chunk
(derived)**. It is **not applied here**, because re-tuning a mapping after its
first measurement is exactly what "one defect, one fix, one measurement"
forbids. It is instead **pre-registered into item 3**, whose kernel has the same
staging loop and has not been built yet - that is a design decision made before
a build, not an iteration after a number, and §5.2 says so explicitly.

### 4.3 Numerics: bitwise, as pre-registered

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
| case 4 | exactly `0.0f` above the diagonal | asserted, green |
| case 5 | `0.0f` outside `[0,L)²` at the ragged tail | green |

The two runs are pre-filled `0x11111111` and `0x22222222`, so an entry **no
quadrant wrote** would fail the `memcmp` rather than pass it - the 3-D grid's
coverage is part of the same bar, including the `(iz = 0, jz = 1)` quadrant that
writes only zeros. The host-side figures are A28's to the digit, which is what
bit equality predicts. Cases 1, 2, 3 and 6 are unchanged and green.

---

# PART 3 - `pf_gdn_A`

## 5. PRE-REGISTRATION (committed before the rewrite was built)

### 5.1 The defect, already measured in §3.1

`src/kernels/prefill/pf_gdn_wy.cl:116-144`, `A[i][j] = beta[i]·(k_i·k_j)·
exp(gc[i]−gc[j])` for `i > j`. It is `pf_gdn_A2`'s code with one staged operand,
no `beta` on the right, `ascale = 1.0f` and the **strict** mask, and §3.1
measured it in the same ISA dump: `for (p = lid; p < CT*CT; p += 256)`, **one
output per work-item per step, with no tile**, 898-instruction hot loop, 130
`mad`, **FMA density 14.5%**, and 516 of those 898 instructions (57%) the bf16
widen. Its one advantage over `pf_gdn_A2` is the one the rewrite must keep: at
16,896 B of SLM and SIMD16 it is **4 work-groups × 16 threads = all 64 of an
Xe-core's slots**, and it issues at **0.502 instructions/XVE/clock** - the
highest rate measured anywhere in this project's GDN kernels. That is why it
costs 0.6136 ms against `pf_gdn_A2`'s 1.0313 for the same 0.8 GFLOP.

So the brief's "if reading the kernel shows it is not the no-tile mechanism,
report what it is instead of rewriting" does not apply: **it is exactly the
no-tile mechanism**, at 14.5% density, and the same treatment applies.

### 5.2 The rewrite, fixed here before it is built

`pf_gdn_A2`'s §3.2, with two changes and one thing deliberately kept.

* **Identical: the quadrant grid (48, nch, 4)**, `iz = z >> 1` / `jz = z & 1`;
  the `(iz = 0, jz = 1)` quadrant is entirely below the strict mask (`i < 32 ≤ j`
  gives `i > j` false), writes its 1024 exact `0.0f`s and returns; a 2 × 2
  output tile with `bi = lid >> 4`, `bj = lid & 15`; fp32 row-major staging;
  `intel_reqd_sub_group_size(16)` for the same reason (32,768 B of SLM is 4
  work-groups either way, and 4 × 16 = 64 resident threads against 4 × 8 = 32);
  `gcs`/`bts` read from global rather than staged, so SLM stays at exactly
  32,768 B; and `tile_dot`, the same band tree written as the same expression.
  `ascale` is `1.0f` here, so the staged value is plainly `f32(k_word)`.
* **Both operands are `k`, from different row ranges**, so the SLM is
  `kis[32][128]` (the quadrant's `i` rows) and `kjs[32][128]` (its `j` rows).
  **When `iz == jz` those are the same rows**, so `kjs` is not staged at all and
  the right-hand row pointers are aimed at `kis`; the test is work-group-uniform,
  so the barrier stays uniform. Two of the three live quadrants take that path.
* **The staging loop is written the way §4.2 measured it should have been, and
  this is a design decision made before a build rather than an iteration after a
  number.** §4.2 measured `pf_gdn_A2`'s staging at **53 instructions per
  iteration, 848 per work-item, 47% of the kernel**, and named the two causes:
  a ternary wrapped around a *load*, which IGC must implement as a branch, and
  one value staged per iteration, which pays the 64-bit global address per
  value. Both are fixed here, before anything is built:
  - **the row index is clamped, not the load** - `ri = min(gi, L−1)` is always a
    live row, so the global read is unconditional and in bounds, and the `< L`
    test becomes a `select` on the loaded fp32 value instead of a branch;
  - **four consecutive `d` per iteration** - one `vload4` of `ushort4` from
    global and one `vstore4` of `float4` into SLM, so 32 × 128 / 4 = 1024 groups
    over 256 work-items is **4 iterations**, and the address arithmetic is paid
    once per four values.
  Derived: ~46 instructions per iteration × 4 = **~184 per work-item**, against
  `pf_gdn_A2`'s measured 848.
  **This is NOT back-ported to `pf_gdn_A2` in this task**: that kernel has been
  measured and re-tuning it would be the iterate-until-it-looks-good the brief
  forbids. Its price there is recorded in §4.2 and carried to §9.

Nothing else changes. **No rounding point moves and no sum is re-associated.**

### 5.3 Pre-registered outcomes

**Time.** `pf_gdn_A` **≤ 0.4 ms per GDN layer per chunk** at C = 2048 (≥ 1.5×
the measured 0.6136 - the brief's bar). Point estimate **0.30 ms**, derived:

* per work-item: staging ~184, contraction + epilogue ~990 (`pf_gdn_A2`'s
  measured 986 plus the `beta` multiply), prologue ~40 → **~1214**;
* 3 live quadrants × 16 threads × 1214 + a zero-writing quadrant ≈ 58,900 per
  head per chunk → **9.05e7 instructions issued** per GDN layer per chunk,
  against the current 2.21e8;
* the issue rate is bracketed by two **measured** figures rather than assumed:
  the tiled `pf_gdn_A2`'s 0.316 (§4.2) and the current `pf_gdn_A`'s 0.502
  (§3.1). At 0.316 that is 0.400 ms, at 0.502 it is 0.252, and the midpoint
  0.42 gives **0.301 ms**. The bar sits at the pessimistic end of that bracket,
  which is stated here rather than discovered later.

Decision rule, fixed in advance (the brief's, with item 1's revert branch):

| speed-up vs 0.6136 ms | verdict |
|---|---|
| ≥ 1.5× (≤ 0.4091 ms) | adopt |
| 1.15-1.5× | adopt, and attribute the shortfall from the assembly |
| 1.0-1.15× | report, do not tune |
| slower than 0.6136 | **revert and report**, as item 1 was |

**Numerics - the pre-registered outcome is BITWISE IDENTITY**, for §3.3's
reasons applied unchanged, with the same fallback:

* **Primary bar: `pf_gdn_A` bit-identical to `pf_gdn_A_legacy`** over every
  `nchunks · 48 · 64 · 64` fp32 entry at C = 256, C = 100 (ragged) and C = 4096,
  the two buffers pre-filled with different patterns so the `memcmp` is a
  coverage bar for the quadrant grid as well as an equality one.
* **Fallback bar:** `gdn_chunk_test` case 1's `gdn_state` max rel ≤ 7.0e-02 and
  mean rel ≤ 2.4e-03, arbiter the token gate.
* `gdn_wy_test`'s existing case-1 bars unchanged: ≤ 4 fp32 ulp against
  `gdn_chunk_ref::mat_A`, device non-zero count `Σ 48·L(L−1)/2`, and exactly
  `0.0f` on every `i <= j` entry - the **strict** mask, which is what
  distinguishes this kernel from `pf_gdn_A2` and is the header block's mask
  pair. Case 2 grades `pf_gdn_solve` on this kernel's `A`, so a wrong `A` fails
  there too.
* **The arbiter:** `prefill_gate_test` 94/94 Vishva and 93/93 RTN,
  `prefill_determinism_test` 9 × 3 bitwise, `prefill_consistency_test` 18/18,
  `gdn_chunk_test` 6/6. A golden regression is a finding, not a tolerance.
