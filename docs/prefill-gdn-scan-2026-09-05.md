# `pf_gdn_scan`, rewritten - ruling A25

Ruling A25 (`interfaces.md`) made this a NEW task with its own pre-registration
rather than a tuning pass, because the rewrite changes both of the scan's
reduction orders and therefore the numerics band `gdn_chunk_test` records.
**§1 below was committed before anything was built or measured.** Everything
after it is labelled measured / derived / estimated, and no quantity appears
twice with two values.

Grade: **iterate** throughout. Two desktop processes (baobab, ptyxis) hold DRM
fds on every card, so the "provably idle box" a record row needs is not
available. Timed rows are 8 independent runs, median, with min/max/spread
beside them (the drop-3 half of the project's iterate convention does not apply
to a `--pp` row - each run is a separate process with a cold model load, so
there is no warm-up sequence inside a process to drop; this is the same note
`docs/prefill-pp-attribution-2026-09-05.md` §1 makes, and it is not silently
re-claimed here).

---

## 1. PRE-REGISTRATION (committed before the rewrite was built)

### 1.1 The defect, read off the kernel

`src/kernels/prefill/pf_gdn_scan.cl` keeps decode's `gdn_step` tile mapping
deliberately - A22's numerics band was measured against exactly that choice.
Per 64-position sub-chunk the 256-lane work-group does three small matmuls:

| stage | shape per work-group (head `h`, 32 state columns) | as built |
|---|---|---|
| 1 | `vn[64][32] = u − W[64][128] · S[128][32]` | 64 sequential 256-lane tree reductions: band FMA + 16-way SLM tree, **5 barriers each**, epilogue on 1/16 of the lanes |
| 2 | `o[64][32] = (Q[64][128] · S) ∘ expg + A2[64][64] · vn` | the same, plus a serial `j ≤ i` loop on 1/16 of the lanes |
| 3 | `S ← S·dl + Kᵀ[128][64] · D[64][32]` | already per-work-item, no barriers |

**640 barriers per sub-chunk, 20,480 per work-group per layer**, 15 of 16
subgroups idle in every epilogue. Measured (attribution doc, `a663f7b`):
**15.071 ms per GDN layer per chunk**, **1446.8 ms of the 4187.9 ms `--pp 4096`
run**, 35.1% of the whole prefill; ~9.66 GFLOP/layer/chunk (derived) at
**0.64 TFLOP/s**.

### 1.2 The rewrite, fixed here before it is built

Unchanged: the grid (48 heads × 4 state-column chunks = 192 work-groups), the
work-group size (256 = 16 subgroups × 16 lanes), the SLM budget class, the
algebra, `o` reading the chunk-START `S`, the k-major state layout, the ONE
reassociation of stage 3 relative to decode, and every rounding point (P6's
q-scale in fp32, P7's bf16 `k`, P10's fp32 `gdn_o`; nothing in this kernel
rounds).

Changed - the work distribution inside stages 1 and 2, and nothing else:

* **`Ss[128][32]` fp32 (16 KB) in SLM**, written once per sub-chunk from the
  work-items' own state registers (not re-read from global) and read by stages
  1-2, which both use the chunk-start `S`.
* **Output tiling.** 64 positions × 32 columns = 2048 outputs / 256 lanes =
  **8 outputs per work-item**. Mapping, fixed now: subgroup `sgid` owns
  positions `i ∈ [4·sgid, 4·sgid+4)`; lane `l` owns state columns `l` and
  `l+16` - the same two columns that lane already owns for its state registers.
  A subgroup's 16 lanes therefore read `Ss[k][0..31]` = two full 64 B lines per
  `k`, and `w[i][k]` / `q[i][k]` are uniform across the subgroup.
* **Private ascending-k fp32 `fma` accumulation over all 128 k.** No tree, no
  `red[]`, no epilogue on 1/16 of the lanes. Stage 2's `A2 · vn` term keeps its
  ascending-`j` `fma` chain over `vn` in SLM.
* **`A2s` leaves SLM.** With this mapping each of the 64 `A2` rows is read by
  exactly one subgroup, so staging it was pure overhead; the `j ≤ i` loop reads
  `A2` from global at a subgroup-uniform address. SLM per work-group therefore
  **falls** 26.5 KB → 24.5 KB (`Ss` 16 KB + `vn` 8 KB + `gcv`/`expg` 512 B).
* **Stage 3 is untouched** and keeps `S` in registers; `Ss` is a copy of the
  chunk-start tile, so nothing is written back to it.
* **Barriers per sub-chunk: 640 → 3** (after the `Ss`/gate staging, after stage
  1 fills `vn`, and at the end of the sub-chunk).

This mapping is committed here **before** it is measured. Per the brief's rule
- one defect, one fix, one measurement - the number the first working build
produces is the number reported, whatever it is; the mapping is not iterated
until it looks good.

### 1.3 Pre-registered outcomes

**Time.** `pf_gdn_scan` **≤ 1.5 ms per GDN layer per chunk** at C = 2048
(≥ 10× the measured 15.071); point estimate **~1.0 ms** (9.66 GFLOP at
~10 TFLOP/s fp32 FMA with SLM operands). Decision rule, fixed in advance:

| speed-up vs 15.071 ms | verdict |
|---|---|
| ≥ 10× (≤ 1.507 ms) | adopt |
| 3-10× (1.507-5.024 ms) | adopt, and record the shortfall with its mechanism |
| < 3× (> 5.024 ms) | report, do not tune |

**`--pp 4096` consequence (derived, from the measured 4187.9 ms and 1446.8 ms
and 96 = 48 GDN layers × 2 chunks):** `4187.9 − 1446.8 + 96 × t` ms.

| pre-registered `t` (ms/layer/chunk) | `--pp 4096` ms | t/s |
|---|---:|---:|
| 1.5 (the bar) | 2885.1 | **1419.7** |
| 1.0 (point estimate) | 2837.1 | **1443.7** |

So the pre-registered row is **1420-1444 t/s, point estimate 1444** (derived).
This sits below A25's "≥ ~1450 t/s" because that figure is the per-chunk
ceiling; a 4096-token run also pays the ~170 ms one-time module-load and
first-launch cost the attribution measured.

**Numerics - the band moves, and is not held to A22's.** The reduction order of
both 128-term contractions changes from decode's band tree to an ascending
128-term chain, so A22's `gdn_state` max rel 3.506e-02 / mean 1.197e-03 is no
longer the right bar. Pre-registered instead:

* `gdn_chunk_test` case 1 (vs the CPU fp32 recurrent reference) `gdn_state`
  **max rel ≤ 7.0e-02 and mean rel ≤ 2.4e-03** - i.e. within 2× of A22's.
* **The arbiter is the token gate**: `golden_gate_test`-machinery
  `prefill_gate_test` **94/94 determined rows exact on Vishva and 93/93 on
  RTN**, every `gdn_state` cosine **> 0.999**; `prefill_determinism_test`
  9 cases × 3 runs bitwise.
* `gdn_chunk_test` case 6: the same 2 × 2048 walk twice from a zeroed state,
  `gdn_state` / `conv_ring` / `gdn_o` / `y` **bitwise identical**.
* A golden regression is a **finding**: diagnose which stage and which rounding
  point moved, report it, stop. It is not a tolerance to loosen.

`tests/prefill/gdn_chunk_ref.h`'s documented reduction order is updated in the
same commit as the kernel (the two files carry a standing "must be edited
together" rule).

**Attribution, priced not fixed.** In the same task, the remaining
~261 ms/chunk of GDN (`pf_gdn_wy`'s three launches, conv, l2norm, gate, seed,
plus `pf_gated_head` outside the GDN total) is re-measured from
`B70_PREFILL_PROFILE=1` and the top one or two get a named mechanism and a
price. No fixes.

### 1.4 Ruling A26 - the consistency gate

`tests/prefill/prefill_consistency_test.cc` imports the golden gate's two
mechanisms from `tests/golden/golden_common.h` (the controller's 2026-08-26 tie
ruling), exactly as `docs/prefill-consistency-finding-2026-09-05.md` §6 option
1 describes:

1. a row whose **decode-side** top-2 fp32-logit gap is under one bf16 ulp of the
   top value is **UNDETERMINED**; prefill's id must be a member of the set;
2. after the first divergence of either kind the prefill walk is
   **teacher-forced on decode's token**, so every later row is graded on a
   shared context.

The state diagnostics and the tie rows stay printed. Pre-registered:
**18/18 cases pass, with the undetermined count listed per case**. If any case
is still red, that is a real finding - reported with the row, both ids and the
gap, not absorbed. `golden_gate_test` and `prefill_gate_test` are not touched.

### 1.5 Invariants that must survive

Full suite green (60 + the long gate, which may SKIP). Decode re-proven after
the last commit: **774 kernels / 19 modules**, `replay_determinism_test`,
golden **94/94 + 93/93**, and a decode bench row. Box: `ZE_AFFINITY_MASK=1`,
84 GB free at the start; any disk error stops the task.

---

# RESULTS (everything below was measured after §1 was committed at `6056609`)

Kernel + reference at `82e52f8`, consistency gate at `bb5f1d5`. Box: B70,
two identical cards, 84 GB free at the start and unchanged at the end, no disk
error. All device work below is `ZE_AFFINITY_MASK=1` except the `--pp` bench
row - see §7's deviations.

## 2. The scan's time, and the decision-rule verdict

`B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096 --tg 8`, one run
(the instrument's own cost is reported below), **measured**:

| | measured | |
|---|---:|---|
| `pf_gdn_scan`, ms per GDN layer per chunk (319.4 ms / 96 waits) | **3.3273** | |
| the same, before the rewrite (`a663f7b`, 1446.8 / 96) | 15.071 | |
| **speed-up** | **4.53×** | |
| pre-registered bar | ≤ 1.5 (≥ 10×) | **MISS by 2.22×** |
| rate on 9.66 GFLOP/layer/chunk (derived) | **2.90 TFLOP/s** | was 0.64 |

**Decision-rule verdict: 4.53× falls in the 3-10× band → ADOPT, and record the
shortfall with its mechanism.** The mapping was not iterated: the number above
is the first working build's, exactly as §1.2 committed it.

### 2.1 The shortfall's mechanism, read off the generated code

`ocloc` was re-run on the committed kernel with `IGC_ShaderDumpEnable=1` and
the Xe2 assembly read. Nothing here is inferred from timing:

* **No spill, and the mapping fits.** `grf_count 128`, `simd_size 16`,
  `slm_size 25088` (= 24.5 KB, exactly §1.2's budget), `barrier_count 1`,
  `instCount 1986`. The build carries `-abortOnSpill 4`, so a spill would have
  failed the build rather than shown up as a slow kernel.
* **The hot loops are issuing ~3 instructions of overhead per fp32 `mad`.**
  Per-block opcode counts of the two k-loops and stage 3's body:

  | block | instructions | `mad` | FMA density | `load.slm` | `load.ugm` | `shl` | `mov` | `mul` |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | stage 1's k-loop (unrolled 32×) | 824 | 256 | **31%** | 64 | 64 | 161 | 128 | 0 |
  | stage 2's k-loop (unrolled 16×) | 479 | 128 | **27%** | 32 | 32 | 81 | 64 | 64 |
  | stage 3's per-position body | 74 | 18 | **24%** | 3 | 4 | 11 | 9 | 7 |

  Composed over one sub-chunk - staging 86, stage 1 `4 × 824`, stage 2's
  k-loop `8 × 479`, stage 2's triangular epilogue ≈ 650, stage 3 `64 × 74` -
  that is **≈ 12,600 instructions per work-item of which ≈ 3,460 are `mad`:
  27% FMA density** (derived).

  The other half of the gap is issue rate, and it is arithmetic on the device's
  own numbers rather than a guess: 12,600 × 32 sub-chunks × 3072 threads
  (192 work-groups × 16) = **1.24e9 SIMD16 instructions per layer per chunk**
  (derived) in 3.3273 ms = 3.72e11 /s, against 256 XVEs × 2800 MHz = 7.17e11
  XVE-clocks/s - **0.52 instructions per XVE per clock**. So the kernel issues
  at about half of one-instruction-per-clock, and only 27% of what it issues is
  the arithmetic. **0.27 × 0.52 = 14% of a one-`mad`-per-clock machine**, which
  is the 2.90 TFLOP/s. The pre-registered ~10 TFLOP/s point estimate needed
  both factors to be near 1.

  Two named terms account for the density, and both are structural rather than
  fixable by re-tiling this mapping:

  1. **The bf16 → fp32 widen is in the loop.** `bf16f()` is a `shl` by 16, and
     the packed pair a `d32` load delivers needs a `mov` to split. That is
     161 + 128 = 289 of stage 1's 824 instructions - **35%** - for 4 operands
     per `k`. Every operand of the two contractions except `Ss` is bf16 in
     memory, by design (P7 and the `xb` layout), so the widen cannot be
     hoisted out of the k-loop without staging widened copies in SLM, which is
     the budget this mapping deliberately gave to `Ss`.
  2. **8 outputs per work-item buys only 8 FMAs per 6 loads.** With `P`
     positions and `V` columns per work-item the loop needs `P + V` operand
     loads for `P · V` FMAs, and `P · V = 2048/256 = 8` is fixed by the grid
     and the work-group size, both of which A25 held constant. `P + V` is
     minimised at `P = 4, V = 2` - which is what §1.2 chose - giving a 8:6
     ceiling. The measured stage-1 loop hits 8 FMAs per 4 loads (the compiler
     merged each pair of consecutive `k` into one `d32` global load), so the
     tiling itself is at its ceiling; the widen is what is on top of it.

  A density near 1 on this shape needs a different instruction, not a
  different mapping: the operands are bf16 and the contraction is a matmul, so
  the machine has `DPAS` for exactly this and the vector pipe does not. That
  is a new task, not a tuning of this one, and it is not taken here.

* **And the 0.52 IPC has a named cause too: the launch is 1.5 waves.**
  192 work-groups × 16 threads = 3072 threads against 256 XVEs × 8 = 2048
  hardware thread slots (the device reports 256 compute units, 2800 MHz,
  128 KiB SLM per Xe-core; at 24.5 KB per work-group SLM is not the binding
  constraint). The first wave fills the machine and the second runs two-thirds
  empty, with too few threads left to cover SLM and L1 latency. **The grid is
  A25's and was held fixed** - 48 heads × 4 state-column chunks is what makes
  the state tile fit in registers and the scan sequential only where the
  recurrence requires - so this is a property of the launch the rewrite was
  told to keep, not of the rewrite.

### 2.2 What the barrier count bought, and what it did not

The rewrite did what it was designed to do - 640 barriers per sub-chunk became
3, and the epilogue that idled 15 of 16 subgroups is gone - and that is the
whole of the 4.53×. What it did not do is turn an fp32 vector-pipe kernel into
a DPAS one, and the 2.22× that separates 3.3273 from the pre-registered 1.5 is
that, measured at 25% FMA density in §2.1.

## 3. Numerics: the band did not move at the printed precision

`gdn_chunk_test`, **measured** (device, `ZE_AFFINITY_MASK=1`), all six cases
green:

| case | tensor | max rel | mean rel | A22's (pre-rewrite) |
|---|---|---:|---:|---|
| 1 - vs the CPU fp32 recurrent reference | `gdn_state` | **3.506e-02** | **1.197e-03** | 3.506e-02 / 1.197e-03 |
| | `gdn_o` | 5.039e-02 | 1.228e-03 | |
| | `y` | 9.567e-02 | 9.790e-04 | 9.567e-02 |
| 2 - vs DEVICE `gdn_step` × 4096 | `gdn_state` | 3.506e-02 | 1.197e-03 | |
| | `gdn_o` | 5.039e-02 | 1.228e-03 | |
| | `y` | 9.567e-02 | 9.790e-04 | |
| 3 - `C = 1` × 4096 vs that oracle | `gdn_state` | 2.089e-02 | 8.519e-04 | 2.089e-02 |
| | `gdn_o` | 2.669e-02 | 8.620e-04 | |

Pre-registered band: max rel ≤ 7.0e-02, mean ≤ 2.4e-03. **Measured 3.506e-02 /
1.197e-03 - inside it, and identical to A22's to four significant figures.**

That identity is the interesting part and it is not a copy-paste: **the band is
dominated by `pf_gdn_wu`'s Q1/Q2 bf16 roundings, which this task did not
touch.** Case 3 - the degenerate `C = 1` path where the chunk algebra collapses
and only Q1/Q2 survive - alone carries 2.089e-02 of the 3.506e-02, exactly as
L1-core measured. Re-associating a 128-term fp32 dot perturbs at ~1e-7
relative, five orders of magnitude below a cancellation that lands the worst
state word at |ref| = 0.052 · rms. The two references have NOT collapsed into
one, which the `> 1e-2` counts show: case 1 counts 17,234 `gdn_o` words over
1e-2 and case 2 counts 17,236; `y` counts 237,692 and 237,664.

Cases 4, 5 and 6 are the hard structural bars and all pass: 2048 in one call ==
2 × 1024 == 32 × 64 bit-identical in `gdn_state`/`conv_ring`/`gdn_o`/`y`;
`C = 100` == 64 + 36 bit-identical; and **the 2 × 2048 walk twice from a zeroed
state is bitwise identical in all four tensors** - the two-run `gdn_state`
determinism §1.3 pre-registered.

### 3.1 The arbiter: the token gates

`prefill_gate_test` grades the prefilled engine against the CPU oracle with the
golden gate's own machinery. **Measured, both checkpoints, unchanged from the
pre-rewrite counts:**

| gate | determined rows exact | undetermined | worst `gdn_state` cosine |
|---|---|---|---|
| Vishva (`oracle-out`) | **94/94** | (per prompt, printed) | > 0.999 |
| RTN (`oracle-out-rtn`) | **93/93** | | > 0.999 |

`prefill_determinism_test`: 9 cases × 3 runs, `gdn_state` / `conv_ring` /
`kv_k` / `kv_v` / control and `cur_token` bitwise identical. `replay_determinism_test`
(decode) green. **No golden regression to diagnose.**

## 4. The new `--pp 4096` row

`tools/bench_decode.sh --pp 4096 --tg 8 --runs 8 --model <RTN>` at `bb5f1d5`,
**measured, iterate grade**:

```
| b70-decode bb5f1d5 pp | 4096 | 2048 | 3141.0 | 1304.06 |
| b70-decode bb5f1d5    | 4096 |    8 |   32.40 |  30.87 |
```

| row | median | min | max | spread |
|---|---:|---:|---:|---:|
| **pp 4096, C = 2048** | **1304.06 t/s** (3141.0 ms) | 1299.46 | 1306.01 | 6.55 (**0.50%**) |
| tg 8 at depth 4096 | 32.40 t/s (30.87 ms/token) | 32.39 | 32.41 | 0.02 (0.06%) |

Against `a663f7b`'s **978.07 t/s / 4187.9 ms**: **+33.3%, −1046.9 ms**.

Scored against the pre-registration:

| | pre-registered (derived) | measured | |
|---|---:|---:|---|
| `--pp 4096` | 2885.1-2837.1 ms | **3141.0 ms** | MISS |
| | 1420-1444 t/s | **1304.06 t/s** | **MISS, 8.9% under the 1420 floor** |

The miss is exactly the scan's own miss and is not an additional finding.
Composing the measured scan time into the baseline gives
`4187.9 − 1446.8 + 96 × 3.3273 = 3060.5 ms` (derived) against the measured
3141.0 - an 80.5 ms (2.6%) residual, which §5 attributes: every *unchanged*
GDN kernel reads 4-10% higher in this session's profile than in `a663f7b`'s.
Holding the scan at its pre-registered 1.5 ms instead would have given
`3141.0 − 96 × (3.3273 − 1.5) = 2965.6 ms = 1381.2 t/s` (derived) - still under
1420 for the same session reason.

**Per-chunk ceiling as now built** (derived from §5's profiled total,
3097.4 ms / 2 chunks = 1548.7 ms/chunk): **1322.4 t/s = 67.0% of vLLM's 1973**,
against A25's 994 t/s = 50.4% before this task. The scan is no longer the
largest term in the prefill: `gemm_bf16` is, at 43.8%.

## 5. Where GDN's time goes now - priced, not fixed

Same profiled run, `--pp 4096`, C = 2048, two chunks (**measured**), with
`a663f7b`'s column beside it. `ms/layer/chunk` = the column ÷ 96.

| launch | ms (2 chunks) | ms/layer/chunk | `a663f7b` ms | Δ |
|---|---:|---:|---:|---:|
| `pf_gdn_seed` | 0.6 | 0.0059 | 0.5 | +20% |
| `pf_gdn_conv` | **126.5** | **1.3182** | 121.6 | +4.0% |
| `pf_gdn_l2norm` | 14.1 | 0.1473 | 13.5 | +4.4% |
| `pf_gdn_gate` | 2.1 | 0.0215 | 2.0 | +5.0% |
| `pf_gdn_A` | 57.6 | 0.5995 | 53.3 | +8.1% |
| `pf_gdn_solve` | 69.1 | 0.7200 | 62.7 | +10.2% |
| **`pf_gdn_wu`** | **197.7** | **2.0595** | 179.2 | +10.3% |
| `pf_gdn_A2` | 97.3 | 1.0138 | 88.6 | +9.8% |
| **`pf_gdn_scan`** | **319.4** | **3.3273** | **1446.8** | **−77.9%** |
| GDN total (the nine above) | 884.4 | - | 1968.2 | **−55.1%** |
| `pf_gated_head` (outside the GDN total) | 30.2 | 0.3148 | 27.8 | +8.6% |
| whole profiled walk | 3097.4 | - | 4119.1 | −24.8% |

Per chunk: GDN **442.2 ms** (was 984.1); GDN less the scan **282.5 ms** (was
260.7). **The +4 to +10% on every unchanged kernel is session-to-session drift
of the instrument on a box that is not idle**, and it is reported rather than
subtracted: it means the scan's 4.53× is measured across two sessions whose
controls disagree by up to 10%, so the true speed-up is between 4.53× and
~4.98×. Either way it is inside the 3-10× band.

### 5.1 `pf_gdn_wu` - 2.0595 ms/layer/chunk, 98.9 ms/chunk (the top one)

Mechanism, read off `src/kernels/prefill/pf_gdn_wy.cl:236-284`: grid
(48 heads, 32 sub-chunks) of 256, and the triangular product is written
`for (p = lid; p < L*DIM; p += 256)` - **one output `(i, x)` per work-item per
step, with no tile**. Each output streams its whole `j ≤ i` chain out of SLM:
per `j` it loads `Ts[i*64+j]` (fp32) plus `vbs[j*128+x]` and `kbs[j*128+x]`
(bf16, each needing the same `shl`/`mov` widen §2.1 priced) for **2 FMAs**.
`Ts[i][j]` is re-loaded for every `x` and `vbs[j][x]` for every `i`; neither is
reused. Work is 2080 `(i,j)` pairs × 128 × 2 outputs × 48 heads × 32 sub-chunks
= **1.636 GFLOP per layer per chunk** (derived), so 2.0595 ms is
**0.79 TFLOP/s** - 3.7× worse than the rewritten scan's 2.90.

Price (derived, **not built**): this is the same defect A25 named, in the same
family, and the same fix applies - give each work-item a tile of `(i, x)` so
`T[i][j]` amortises over several `x` and `vb[j][x]` over several `i`. At the
scan's post-rewrite 2.90 TFLOP/s the same 1.636 GFLOP costs 0.564 ms, i.e.
**−1.50 ms/layer/chunk = −72 ms/chunk = −144 ms on `--pp 4096` → ~1367 t/s**
(+4.8%). **The catch, and it is why this is priced rather than done:** Q1-Q4
live in this kernel, and A22 attributed the *entire* state band's tail to
cancellation at Q1/Q2. A re-tiling here moves the sums that feed those four
roundings, so it needs its own pre-registered band - unlike the scan, whose
band §3 shows did not move at all.

### 5.2 `pf_gdn_conv` - 1.3182 ms/layer/chunk, 63.3 ms/chunk (the second)

Mechanism, read off `src/kernels/prefill/pf_gdn_conv.cl:146-189`: the launch is
**40 work-groups of 256 = 10,240 work-items, one per conv channel**, and each
walks all `C` positions serially carrying the 4-tap window in three registers.
The machine has 256 XVEs × 8 threads = 2048 hardware thread slots; this launch
offers 10,240/16 = **640 threads, 31% of them**. Traffic is 83.9 MB read
(fp32 `qkvz_partials`, of whose 16,384-wide rows only 10,240 are touched) +
41.9 MB written (bf16 `xb`) = 125.8 MB per layer per chunk (derived), so
1.3182 ms is **95.4 GB/s - 16% of the 590 GB/s this device measures.** It is
neither bandwidth-bound nor compute-bound; it is under-occupied.

Price (derived, **not built**): the recurrence is only three positions deep and
its three predecessors are readable straight out of `qkvz_partials`, so the
`m` range blocks cleanly - a grid of (channels, position-blocks) would fill the
machine. Even taking occupancy 31% → ~93% and nothing else,
**−0.88 ms/layer/chunk = −42 ms/chunk = −84 ms on `--pp 4096`** (+2.8%). Its
rounding points are P1/P2/P3, which blocking does not move (each output is
still one 4-tap ascending `fma` chain over the same three predecessors), so
unlike §5.1 this one would need only the existing `gdn_conv_test` re-run - but
it is still not taken here.

Everything else is small: `pf_gdn_A2` 1.0138, `pf_gdn_solve` 0.7200,
`pf_gdn_A` 0.5995, `pf_gated_head` 0.3148, `pf_gdn_l2norm` 0.1473,
`pf_gdn_gate` 0.0215, `pf_gdn_seed` 0.0059 ms/layer/chunk.

## 6. The consistency gate after A26 - 18/18

`prefill_consistency_test` at `bb5f1d5`, run on **both** checkpoints
(`ctest` registers only the Vishva one; the RTN run is the same binary with the
RTN snapshot, which is how the finding's 18 cases were taken). **Measured:**

**Vishva** - `det-exact` is the gate; `undet` counts the rows where decode's own
top-2 gap is under one bf16 ulp; `lead` is the leading exact run before the walk
was teacher-forced.

| prompt | chunk | ids | det-exact | tie-agree | tie-member | undet | lead | first-diff | decode gap there |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prose | 2048 | 42 | **59/59** | 5 | 0 | 5 | 64/64 | - | - |
| prose | 1024 | 42 | **59/59** | 5 | 0 | 5 | 64/64 | - | - |
| prose | 16 | 42 | **59/59** | 5 | 0 | 5 | 64/64 | - | - |
| code | 2048 | 61 | **62/62** | 1 | 1 | 2 | 41/64 | 41 | 1.812e-03 (0.03 ulp) |
| code | 1024 | 61 | **62/62** | 1 | 1 | 2 | 41/64 | 41 | 1.812e-03 (0.03 ulp) |
| code | 16 | 61 | **62/62** | 1 | 1 | 2 | 41/64 | 41 | 1.812e-03 (0.03 ulp) |
| cjk | 2048 | 38 | **60/60** | 4 | 0 | 4 | 64/64 | - | - |
| cjk | 1024 | 38 | **60/60** | 4 | 0 | 4 | 64/64 | - | - |
| cjk | 16 | 38 | **60/60** | 3 | 1 | 4 | 23/64 | 23 | 3.675e-02 (0.63 ulp) |

543 determined rows, all exact; 33 undetermined, 4 of them resolved to the other
member of decode's tie set.

**RTN**

| prompt | chunk | ids | det-exact | tie-agree | tie-member | undet | lead | first-diff | decode gap there |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prose | 2048 | 42 | **60/60** | 2 | 2 | 4 | 15/64 | 15 | 2.502e-02 (0.32 ulp) |
| prose | 1024 | 42 | **60/60** | 2 | 2 | 4 | 15/64 | 15 | 2.502e-02 (0.32 ulp) |
| prose | 16 | 42 | **60/60** | 3 | 1 | 4 | 25/64 | 25 | 3.111e-03 (0.04 ulp) |
| code | 2048 | 61 | **63/63** | 1 | 0 | 1 | 64/64 | - | - |
| code | 1024 | 61 | **63/63** | 1 | 0 | 1 | 64/64 | - | - |
| code | 16 | 61 | **63/63** | 1 | 0 | 1 | 64/64 | - | - |
| cjk | 2048 | 38 | **63/63** | 0 | 1 | 1 | 47/64 | 47 | 1.010e-03 (0.02 ulp) |
| cjk | 1024 | 38 | **63/63** | 0 | 1 | 1 | 47/64 | 47 | 1.010e-03 (0.02 ulp) |
| cjk | 16 | 38 | **63/63** | 0 | 1 | 1 | 47/64 | 47 | 1.010e-03 (0.02 ulp) |

558 determined rows, all exact; 18 undetermined, 8 resolved to the other member.

**18 of 18 cases pass, exactly as pre-registered.** Across both checkpoints:
**1101 determined rows, zero mismatches**; 51 undetermined rows, of which 12
went to the other member of decode's own tie set and 39 agreed anyway. Every
divergence in the table sits on a row decode does not decide - 0.02 to 0.63
bf16 ulp - and none on a determined one. Nothing was widened: on a determined
row the gate is one id, element-exact, and it is 1101 for 1101.

Two second-order observations, recorded because they are evidence the gate is
measuring what it claims:

* **The rewrite moved two of the coin tosses and left two alone.** Against the
  finding's pre-rewrite rows: Vishva `cjk` at chunk 16 moved its divergence
  from step 13 to step 23, RTN `prose` at chunk 16 from 15 to 25, Vishva `code`
  at chunk 16 acquired one at 41 (it had none), while Vishva `code` at 2048/1024
  and RTN `cjk` and `prose` at 2048/1024 kept theirs at 41, 47 and 15. A
  reduction-order change is exactly what re-tosses a sub-ulp coin and leaves a
  clear row alone; a correctness change would not be selective like that.
* **`conv_ring`'s max rel is still exactly 1.0 in every row and is still not a
  finding** - the chunked path writes only the three live ring slots, so the
  comparison includes dead slots decode filled and prefill did not
  (`gdn_chunk_test` case 5's `live_slots_equal` is the standing proof). It is
  left unfiltered because a diagnostic that agrees by construction says nothing.

## 7. Invariants, and every deviation from the brief

### 7.1 Suite

`ctest` on the box, `ZE_AFFINITY_MASK=1`, at `bb5f1d5`: **60/60 pass, 0 fail,
1 skip** (`prefill_gate_long_test` - `oracle-out-long` still needs the
operator's `docker run`, so it skips cleanly with 77 as designed). Total 423 s.
`prefill_consistency_test` is green *inside* the suite for the first time since
it was written.

### 7.2 Decode, re-proven after the last code commit

| invariant | measured |
|---|---|
| kernel/module count | **774 kernels, 19 modules**, max_len 16384, persistent 1.24 GB (`replay_determinism_test`, and every `b70-decode` run's banner) |
| replay determinism | `replay_determinism_test` OK - 8 tokens × 3 runs bitwise identical |
| decode golden gate | `golden_gate_test` **94/94** determined rows exact, 2 undetermined (1 agree + 1 other member); `kernel_table_test` OK |
| prefill golden gate, Vishva | `prefill_gate_test` **94/94**, 2 undetermined; worst `gdn_state` cosine **0.999780657** (L60) |
| prefill golden gate, RTN | `prefill_gate_rtn_test` **93/93**, 3 undetermined; worst `gdn_state` cosine **0.999847086** (L60) |
| prefill determinism | `prefill_determinism_test` OK - 9 (prompt, chunk) cases × 3 runs, `gdn_state`/`conv_ring`/`kv_k`/`kv_v`/control and `cur_token` bitwise identical |
| decode bench | **32.21 t/s** (31.05 ms/token), median of 3, min 32.19 max 32.22, spread 0.09% - see below |

The decode bench row, `tools/bench_decode.sh --depth 4096 --tg 256 --runs 3
--model <RTN>` at `a52eea8` (measured; the `-dirty` suffix is this document,
the only uncommitted file at the time, and no source):

```
| b70-decode a52eea8-dirty | 4096 | 256 | 32.21 | 31.05 |
```

**32.21 t/s against `docs/BENCHMARKS.md`'s recorded 32.22 for this
checkpoint - 0.03% apart.** Decode is untouched, and the tg-8 row in §4
(32.40 t/s, spread 0.06%) says the same thing from the other harness. The
`a663f7b` session read 32.89 on the tg-8 row; that 1.5% is this box's
session-to-session spread at iterate grade, the same drift §5 measures on
the unchanged GDN kernels, and both readings sit at or above the record.

Every `gdn_state` cosine is above the pre-registered 0.999, on both
checkpoints, and both determined-row counts are exactly what they were before
the rewrite. **No golden regression to diagnose.**

### 7.3 Deviations

1. **`tools/bench_decode.sh` does not set `ZE_AFFINITY_MASK`, and it was not
   modified.** The box exposes two identical `Intel(R) Arc(TM) Pro B70` devices
   with the same two desktop DRM-fd holders on both; without the variable the
   CLI takes device 0. The `a663f7b` baseline row this task is scored against
   was taken by the same unmodified harness, so leaving it alone is what keeps
   978.07 and 1304.06 comparable - patching the harness would have made the new
   row a different measurement of a different thing. **Every other piece of
   device work in this task ran with `ZE_AFFINITY_MASK=1`**: the profiled walk
   of §2 and §5, `gdn_chunk_test`, both consistency runs, the gate re-runs, and
   the whole `ctest` suite.
2. **`A2s` was removed from SLM rather than kept.** The brief said to keep the
   SLM budget class; under the new mapping each `A2` row is read by exactly one
   subgroup, so staging it was pure overhead. SLM went **26.5 KB → 24.5 KB** -
   below the old budget, not above - and the compiler confirms 25,088 B. Fixed
   in §1.2 before anything was built.
3. **The scan's per-layer time is one profiled run, not eight.** `--pp` is a
   whole-process measurement with a cold 16 GB load; the per-kernel split comes
   from `B70_PREFILL_PROFILE=1`, which the attribution doc established costs
   −2.0% (and here −4.4%) against the same binary's plain run. The eight-run
   median is the `--pp` row in §4. §5 reports the unchanged kernels' 4-10%
   session drift rather than hiding it, which bounds the scan's speed-up at
   4.53-4.98× instead of asserting a single spuriously precise figure.
4. **The consistency gate's tie set.** A26 says "one of the two", so that is
   what `a52eea8` grades on. The wider band - every id within one bf16 ulp,
   which is the shape `golden_decision`'s set has - is kept and printed as a
   diagnostic. Measured on both checkpoints after the change: **identical
   counts, and no row ever printed a wider band**, so the two readings never
   differed on this data. (`bb5f1d5` graded on the band; `a52eea8` narrowed it.
   Both green, same numbers.)
5. **No fix was attempted for either priced GDN kernel** (§5.1, §5.2) and the
   scan's mapping was not iterated after its first measurement, per the brief's
   "one defect, one fix, one measurement".

### 7.4 Box

84 GB free at the start and at the end; no disk error. No container was
started, no process killed, nothing deleted, no server restarted. The only
files written outside the repo are the IGC dumps of §2.1 under `/tmp/gdnscan`.

## 8. What this leaves on the table

* `pf_gdn_scan` at 3.3273 ms is now **10.3% of the profiled walk**, down from
  35.1%. `gemm_bf16` is the largest term again at **43.8%**, and A23/A24 closed
  every lever on it.
* Getting the scan below ~1.5 ms needs DPAS, not another vector mapping: §2.1
  measures 27% FMA density and 0.52 instructions per XVE per clock, and the
  operands are bf16 in memory by design. That is a new pre-registered task.
* `pf_gdn_wu` (−144 ms) and `pf_gdn_conv` (−84 ms) are priced in §5 and would
  compose to roughly **1416 t/s** on `--pp 4096` (derived, neither built).
  `pf_gdn_wu`'s carries Q1-Q4 and would need its own band pre-registration;
  `pf_gdn_conv`'s does not move a rounding point.
* The 4096-id prefill is now **3.14 s** against the decode-replay ingest's
  **121 s** - a **38.5× cut**, measured, up from 28.9×.
