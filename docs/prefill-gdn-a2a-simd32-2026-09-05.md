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
