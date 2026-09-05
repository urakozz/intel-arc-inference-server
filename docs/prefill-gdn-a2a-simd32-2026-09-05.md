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
