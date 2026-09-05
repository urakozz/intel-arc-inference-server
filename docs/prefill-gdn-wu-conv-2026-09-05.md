# `pf_gdn_conv` and `pf_gdn_wu`, rewritten - ruling A27

Ruling A27 (`interfaces.md`) sent the last two priced GDN kernels to one more
pre-registered task. `docs/prefill-gdn-scan-2026-09-05.md` §5.1 and §5.2 priced
them and did not build them; this document builds them. It follows that
document's shape exactly: **every pre-registration section below was committed
before the kernel it describes was written**, everything after it is labelled
measured / derived / estimated, and no quantity appears twice with two values.

Grade: **iterate** throughout. Two desktop processes (baobab, ptyxis) hold DRM
fds on every card, so the "provably idle box" a record row needs is not
available. Timed `--pp` rows are 8 independent runs, median, with min/max/spread
beside them (the drop-3 half of the project's iterate convention does not apply
to a `--pp` row - each run is a separate process with a cold model load, so
there is no warm-up sequence inside a process to drop; the same note
`docs/prefill-pp-attribution-2026-09-05.md` §1 and
`docs/prefill-gdn-scan-2026-09-05.md` §1 make, and it is not silently re-claimed
here).

## 0. The control, measured before anything was changed

A27's per-kernel figures were taken in the scan task's session, and that
document's §5 measured **+4 to +10% session-to-session drift** on kernels
nobody had touched. So this task's first act was to re-profile the **unchanged**
tree in this session, so that both a cross-session and a same-session scoring
exist and neither has to be assumed.

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8` at `4eca4c1`, one run, **measured**:

| ms/layer/chunk (column ÷ 96) | this session (`4eca4c1`) | A27's (`bb5f1d5`) | drift |
|---|---:|---:|---:|
| `pf_gdn_conv` | **1.3163** | 1.3182 | −0.14% |
| `pf_gdn_wu` | **2.0578** | 2.0595 | −0.08% |
| `pf_gdn_scan` | 3.3149 | 3.3273 | −0.37% |
| profiled walk total, ms | 3094.8 | 3097.4 | −0.08% |

**There is no session drift to correct for this time** - the two sessions'
controls agree to 0.4% on every GDN kernel. The bars below are therefore stated
against A27's published figures (1.3182 and 2.0595), which is what the brief
pre-registers, and the same-session control is reported beside every result so
the speed-up is not resting on a cross-session comparison.

---

# PART 1 - `pf_gdn_conv`

## 1. PRE-REGISTRATION (committed before the rewrite was built)

### 1.1 The defect, read off the kernel

`src/kernels/prefill/pf_gdn_conv.cl:145-189`. The launch is **40 work-groups of
256 = 10,240 work-items, one per conv channel**, and each walks all `C`
positions serially carrying the 4-tap window in three registers. The machine is
256 XVEs × 8 = 2048 hardware thread slots; this launch offers 10,240 / 16 =
**640 threads, 31% of them**.

Traffic per layer per chunk (derived): 83.9 MB read (fp32 `qkvz_partials`, of
whose 16,384-wide rows only the first 10,240 are touched) + 41.9 MB written
(bf16 `xb`) = **125.8 MB**. Measured 1.3182 ms is therefore **95.4 GB/s, 16% of
the 590 GB/s this device measures** - neither bandwidth-bound nor compute-bound.

The mechanism is **memory-level parallelism, and the arithmetic says so before
anything is built**: consecutive iterations of one work-item's `m` loop are
64 KB apart, so the bytes in flight at any instant are one iteration's worth of
the whole grid = 10,240 lanes × 4 B = **40 KB**. Sustaining 590 GB/s at this
part's ~500 ns DRAM latency needs ~295 KB in flight (derived, Little's law).
40/295 = 14%, which is the measured 16%. Nothing else needs to be wrong.

### 1.2 The rewrite, fixed here before it is built

The FIR is 3 deep, and **every one of a position's three predecessors is
readable straight out of `qkvz_partials`** - `raw(m) = rne_bf16(qkvz_partials[m
* 16384 + ch])` is exactly what the ring stores (P1) and exactly what the
serial walk carries in `win0..win2`. So the position range blocks with a
three-position halo that costs three loads, and the recurrence does not have to
be carried across the block boundary at all.

Fixed now, before measurement:

* **Grid (40 channel groups, `NB` position blocks, 1)**, work-group 256
  unchanged. `NB` is a **host-side count**, and the kernel derives its own block
  width from it - `blk = ceil(c_count / get_num_groups(1))`, `m0 = gid1 * blk`,
  `m1 = min(m0 + blk, c_count)` - so host and kernel cannot disagree about
  coverage the way a shared literal would let them. `gdn.cc` asks for
  `NB = ceil(C / 128)`: at C = 2048 that is 16 blocks = **640 work-groups =
  10,240 threads, 5 waves of the 2048-slot machine**, and ≈640 KB in flight
  against the ~295 KB Little's law asks for.
* **The three predecessors are re-read, not carried.** For `t = 0,1,2` the
  block's window slot is `raw(m0 - 3 + t)` when that position is ≥ 0 and
  `seed[(m0 - 3 + t) + 3]` when it is not - `seed[t]`'s definition is already
  "chunk-relative position `t - 3`" (`pf_gdn_seed`), so the two cases are one
  expression with a sign test. Three extra `qkvz_partials` loads per block per
  channel: **2.3% of a 128-position block** (derived).
* **The ring writeback is guarded by block ownership** rather than run by every
  work-item: position `m = C - 3 + t` is written by the one block whose
  `[m0, m1)` contains it. Every live slot is still written exactly once, with
  the same expression.
* **Nothing else changes.** The per-position body is the same four ascending
  `fma`s over the same widened bf16 words with the same weights, the same
  `rne_bf16(silu_f32(acc))` store (P3), and the same `xb` index. **No rounding
  point moves and no sum is re-associated**: this is a re-partition of a loop
  nest, and the output is a function of `(m, ch)` alone.

### 1.3 Pre-registered outcomes

**Time.** `pf_gdn_conv` **≤ 0.45 ms per GDN layer per chunk** at C = 2048
(≥ 2.9× the measured 1.3182; the −84 ms/4096 A27 derived). Point estimate
**~0.25 ms** (125.8 MB at ~500 GB/s, i.e. 85% of the measured device bandwidth).
Achieved GB/s reported beside it. Decision rule, fixed in advance:

| speed-up vs 1.3182 ms | verdict |
|---|---|
| ≥ 2× (≤ 0.659 ms) **and** bitwise identical | adopt |
| < 2× | report, do not tune |
| any non-identical word | **finding - report and stop** |

**Numerics: the bar is bitwise identity, not a band.** The rewrite moves no
rounding point, so the pre-registered outcome is that the new kernel's `xb` over
the whole chunk **and** the ring's live slots are **bit-for-bit identical** to
the current kernel's, at C = 2048 and at every ragged width
`gdn_conv_test` already exercises.

**How that is proved, stated in advance:** the pre-rewrite kernel is **kept in
the same `.cl` file as `pf_gdn_conv_legacy`**, launched by nothing but the test,
and `gdn_conv_test` gains a case that runs both over the same inputs and
`memcmp`s `xb` and the ring. This is the brief's "two-kernel comparison" option;
a golden dump was rejected because the artefact is 41.9 MB per width, and a
host-reference comparison cannot be the bar because `silu`'s `exp` is 3 ulp on
the device and correctly rounded on the host (which is why `gdn_conv_test`
case 1's existing bar is ≤ 2 bf16 ulp and not equality). The legacy entry point
costs one binary in a module the runtime already loads and it is a permanent
regression bar for any future re-partition of this loop.

`gdn_conv_test`'s existing six cases must stay green unchanged.

### 1.4 Invariants that must survive (both parts of this task)

Full suite green (60 + the long gate, which may SKIP). Decode re-proven after
the last commit: **774 kernels / 19 modules**, `replay_determinism_test`, golden
**94/94 + 93/93**, and a decode bench row. Box: `ZE_AFFINITY_MASK=1`, 84 GB free
at the start; any disk error stops the task.

## 2. RESULT - `pf_gdn_conv` at 4.00×, bitwise

Everything in this section was measured after §1 was committed at `c86a6ab`.
Kernel at `src/kernels/prefill/pf_gdn_conv.cl`, grid decided in
`src/runtime/prefill/gdn.cc`, bar asserted by `tests/prefill/gdn_conv_test.cc`
case 7.

### 2.1 Time, and the decision-rule verdict

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (31.6 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_conv`, ms per GDN layer per chunk | **0.3296** | |
| the same before the rewrite, A27's (`bb5f1d5`) | 1.3182 | **4.00×** |
| the same before the rewrite, this session's control (§0) | 1.3163 | 3.99× |
| pre-registered bar | ≤ 0.45 (≥ 2.9×) | **HIT** |
| point estimate | ~0.25 | missed by 32% |
| rate on 125,829,120 B (derived) | **381.8 GB/s** | was 95.4 |

**Decision-rule verdict: 4.00× ≥ 2× and bitwise identical → ADOPT.** The mapping
was not iterated: 128 positions per block was fixed in §1.2 and the number above
is that build's first and only measurement.

Per-chunk consequence: 48 layers × (1.3182 − 0.3296) = **−47.45 ms/chunk**,
**−94.9 ms over a 4096-token prefill** (derived), against the −84 ms A27 derived
from an assumed 31% → ~93% occupancy alone.

### 2.2 Where the remaining 35% of the bandwidth went

381.8 GB/s is **64.7% of the 590 GB/s this device measures** (derived, on the
same 125.83 MB of core traffic §1.1 priced; the three-position halo the blocking
re-reads adds 1.90 MB = **1.5%**, so it is not the residue). What the number
says plainly: the kernel has stopped being occupancy-bound - 640 work-groups is
5 waves, and the pre-registered mechanism moved it 4× - and is now
bandwidth-class on a 2:1 read/write mix. **The 1.5× that separates it from the
device's pure-read figure is not attributed here**, because attributing it would
mean a second mapping and a second measurement, which the brief's "one defect,
one fix, one measurement" forbids. It is recorded as what is left on this
kernel, and it is worth at most 0.12 ms/layer/chunk = 11 ms on a 4096-token
prefill.

### 2.3 Numerics: bitwise, as pre-registered

`gdn_conv_test` at `ZE_AFFINITY_MASK=1`, **measured**, all seven cases green:

| case | bar | measured |
|---|---|---|
| 7 - blocked vs `pf_gdn_conv_legacy`, C = 4096 (32 blocks) | bit equality | **41,943,040 `xb` words and 30,720 ring live words identical** |
| 7 - the same two kernels over the 1 + 2 + 64 ragged walk | bit equality | **identical, `xb` and ring** |
| 1 - vs `gdn_chunk_ref::conv` | ≤ 2 bf16 ulp | max **1** ulp, 41,942,899/41,943,040 exact (0.0003% differ) |
| 1 - ring live slots vs the reference | bit-exact | **bit-exact**, 30,720 words |
| 2 - 4 × 1024 vs 1 × 4096 | bit-identical | **bit-identical**, `xb` and ring |
| 3 - 1 + 2 + 64 vs the first 67 rows | bit-identical | **bit-identical** |
| 0, 4, 5, 6 | unchanged | unchanged and green |

**No non-identical word, so there is no finding to report and the task
continues to part 2.** Case 1's 141 words that differ from the *host* reference
by one ulp are `silu`'s `exp` (3 ulp on the device, correctly rounded on the
host) and are exactly the words that differed before the rewrite - case 7 is
what says so, because it compares the two devices' outputs rather than either
against the host.

---

# PART 2 - `pf_gdn_wu`

## 3. PRE-REGISTRATION (committed before the rewrite was built)

### 3.1 The defect, read off the kernel

`src/kernels/prefill/pf_gdn_wy.cl:236-284`. Grid (48 heads, 32 sub-chunks) of
256, and the triangular product is written `for (p = lid; p < L*DIM; p += 256)`
- **one output `(i, x)` per work-item per step, with no tile.** Each output
streams its whole `j ≤ i` chain out of SLM: per `j` it loads `Ts[i*64+j]`
(fp32) plus `vbs[j*128+x]` and `kbs[j*128+x]` (bf16, each needing the `shl`/`mov`
widen `docs/prefill-gdn-scan-2026-09-05.md` §2.1 priced at 35% of that kernel's
hot loop) for **2 FMAs**. `Ts[i][j]` is re-loaded for every `x` and `vb[j][x]`
for every `i`; neither is reused.

Work is 2080 `(i,j)` pairs × 128 `x` × 2 outputs × 48 heads × 32 sub-chunks =
**1.636 GFLOP per layer per chunk** (derived), so the measured 2.0595 ms is
**0.79 TFLOP/s** - 3.7× worse than the rewritten scan's 2.90 on the same vector
pipe. Counted as instructions: ~1040 `(i,j,x)` triples per work-item, each
costing 3 SLM loads + 2 widens (2 instructions each) + 2 `mad` + loop overhead
≈ 11 instructions for 2 FMAs - **an FMA density near 18%** (derived).

### 3.2 The rewrite, fixed here before it is built

A25's treatment, with the three complications this kernel has and the scan did
not: the sum is **triangular** (so a naive tile is load-imbalanced), the four
rounding points Q1-Q4 live here, and both `u` and `w` share one `T` operand.

Unchanged: the algebra; the work-group size (256 = 16 subgroups × 16 lanes);
Q1, Q2, Q3, Q4 - each still one bf16 RNE at the end of its own expression, in
bf16, in the same place; the **ascending-`j`, one-accumulator-per-output `fma`
chain** of Q3/Q4; `T` fp32 (ruling R8); the `w`/`u` output layout.

Changed:

* **The grid gains a third dimension: (48 heads, `nch` sub-chunks, 2).** The
  work-group now owns **64 of the 128 output columns** - `cxh = get_group_id(2)`
  picks the half. That is what lets the operands be staged in fp32 inside the
  same SLM budget class, and it doubles the launch to **3072 work-groups =
  49,152 threads (24 waves)** from 1536 / 24,576.
* **Operands staged in SLM as fp32, so the widen leaves the inner loop
  entirely.** `vbs[64][64]` and `kbs[64][64]` fp32 (16 KB each) hold
  `f32(rne(...))` - the **same rounded bf16 value**, widened once at staging
  instead of once per use. Q1 and Q2 are unmoved: what changes is the container,
  not the rounding, and a bf16 value is exact in fp32 by construction.
* **`T` is staged TRANSPOSED**, `TsT[j][i] = T[i][j]`, row stride 68 floats
  (17 KB). The stride is 68 and not 64 so that four consecutive `i` at one `j`
  are one 16 B aligned vector read, and not 65 (which would be conflict-free on
  the staging write but unaligned on every read); 68 leaves a 4-way bank
  conflict on 16 staging writes per work-item, which is ~3% of the compute.
  **SLM per work-group: 50,688 B (49.5 KB)**, against the current 49,152 B -
  the same budget class, and still two work-groups per 128 KB Xe-core.
* **Output tile: 8 positions × 2 columns = 16 `(i, x)` pairs = 32 fp32
  accumulators per work-item.** Mapping, fixed now: subgroup `sgid` takes
  `cs = sgid & 1` (which 32 of the work-group's 64 columns) and `b = sgid >> 1`
  (which positions); lane `l` owns the **two consecutive** columns
  `32·cs + 2l` and `+1`, so a subgroup reads 128 contiguous bytes of `vbs` and
  of `kbs` per `j` in one vector read each.
* **The positions are a MIRROR PAIR, `4b..4b+3` and `60−4b..63−4b`, and that is
  the answer to the triangular imbalance.** A tile of 8 *consecutive* positions
  would make the subgroup holding `i = 56..63` do 484 of the work-group's 2080
  `(i,j)` pairs against a mean of 260 - **1.86×**, and the work-group finishes
  when its slowest subgroup does, which would eat most of the gain. Pairing a
  low block of 4 with the mirror-image high block gives every subgroup
  `(16b+10) + (16(15−b)+10) = 260` pairs **exactly, for every `b`** (derived).
  The residual imbalance is in overhead only: 1.12× (derived, §3.3's
  instruction count).
* **Two phases, so the `j` loop is exactly as long as it must be.** Phase A runs
  `j = 0 .. min(4b+3, L−1)` with both blocks accumulating; phase B runs
  `j = jl+1 .. min(63−4b, L−1)` with the high block alone. Ascending `j` and one
  accumulator per output throughout, and the two phases are contiguous, so the
  accumulation order per output is **exactly the reference's**.
* **No mask on the triangle, because `T` already is one.** `pf_gdn_solve` writes
  **exactly `0.0f`** for `j > i` (`gdn_wy_test` case 2 asserts it bit-exactly),
  so a block's shared `j` range costs `fma(0.0f, vb, acc)` on the 12 of 260
  pairs (4.6%) it over-runs, and that is a **bitwise no-op**: `0·x` is `±0` for
  finite `x`, `acc` is never `−0.0f` (it starts at `+0.0f` and in
  round-to-nearest an exactly-cancelling sum returns `+0.0f`), and `acc + 0.0f`
  is `acc`. The `j` range is still clamped to `L−1` so no work-item reads a
  `vbs`/`kbs` row the staging loop did not write.

### 3.3 Pre-registered outcomes

**Time.** `pf_gdn_wu` **≤ 0.6 ms per GDN layer per chunk** at C = 2048 (≥ 3.4×
the measured 2.0595; the −144 ms/4096 A27 derived). Point estimate **~0.56 ms**,
and it is derived twice, from opposite directions, which is why it is stated to
two figures:

* *from the rate*: 1.636 GFLOP at the rewritten scan's measured 2.90 TFLOP/s =
  0.564 ms - A27 §5.1's own price;
* *from the instruction count*: the worst subgroup (`b = 0`) issues 4 phase-A
  steps of (2 vector operand reads + 2 vector `T` reads + 32 `mad` + ~4
  address/loop) and 60 phase-B steps of (2 + 1 + 16 + ~4), i.e. ≈ **1540
  instructions per 16 output pairs** against the current kernel's ≈ 5720 -
  **3.7×**, giving 0.557 ms. FMA density rises 18% → **~70%**.

Decision rule, fixed in advance (the brief's):

| speed-up vs 2.0595 ms | verdict |
|---|---|
| ≥ 3× (≤ 0.686 ms) | adopt |
| 1.5-3× (0.686-1.373 ms) | adopt, and attribute the shortfall from the assembly |
| < 1.5× (> 1.373 ms) | report, do not tune |

**Numerics - the pre-registered outcome is BITWISE IDENTITY, and the brief's
band is kept as the fallback bar.** The brief says "only the dot-product
association changes"; under the mapping fixed in §3.2 **the association does not
change at all**. Every output is still one accumulator over `j` ascending with
`fma`, over the same `T` values and the same Q1/Q2-rounded operands; what
changed is which work-item owns which output and how many times each operand is
loaded. So:

* **Primary bar: `pf_gdn_wu` bit-identical to `pf_gdn_wu_legacy`** - the kernel
  exactly as it stood - on `w` and `u` over every element, at C = 2048 and at
  the ragged `C = 100`. Proved the way part 1's was: the pre-rewrite kernel
  stays in the same `.cl`, launched by nothing but `gdn_wy_test`.
* **Fallback bar, kept because a pre-registration is not allowed to move after
  the fact:** if any word differs, `gdn_chunk_test` case 1's `gdn_state`
  **max rel ≤ 7.0e-02 and mean rel ≤ 2.4e-03** (within 2× of A22's 3.506e-02 /
  1.197e-03), and the arbiter is the token gate.
* `gdn_wy_test`'s existing bars unchanged: `u` **bit-identical** to
  `gdn_chunk_ref::wu` fed the device's own `T`, `w` ≤ 2 bf16 ulp on every
  element with `|ref| ≥ rms/4`.
* **The arbiter, whatever the band does:** `prefill_gate_test` **94/94
  determined rows exact on Vishva and 93/93 on RTN**, every `gdn_state` cosine
  > 0.999; `prefill_determinism_test` 9 cases × 3 runs bitwise;
  `prefill_consistency_test` **18/18 under A26**; `gdn_chunk_test` 6/6.
* **A golden regression is a finding**: diagnose which of Q1-Q4 moved, report,
  stop. It is not a tolerance to loosen.

`tests/prefill/gdn_chunk_ref.h`'s documented reduction order is updated in the
same commit as the kernel (the two files carry a standing "must be edited
together" rule). The **order itself does not change** - that is the point worth
recording there, and the reference already stores its `vb`/`kb` as widened fp32,
so it mirrors the new staging as written.

### 3.4 The `--pp 4096` row, pre-registered from the two measured savings

Composed from the measured `--pp 4096` baseline of **3141.0 ms / 1304.06 t/s**
(A27, 8-run median, iterate) and 96 = 48 GDN layers × 2 chunks:

| term | ms | source |
|---|---:|---|
| baseline | 3141.0 | measured, A27 |
| `pf_gdn_conv` 1.3182 → 0.3296 | −94.9 | **measured**, §2.1 |
| `pf_gdn_wu` 2.0595 → *t* | −96·(2.0595 − *t*) | *t* pre-registered ≤ 0.6 |

| pre-registered `t` (ms/layer/chunk) | `--pp 4096` ms | t/s |
|---|---:|---:|
| 0.6 (the bar) | 2906.4 | **1409.3** |
| 0.56 (the point estimate) | 2902.6 | **1411.2** |

So the pre-registered row is **1409-1411 t/s, point estimate 1411** (derived),
against A27 §8's "roughly 1416 t/s" for the two kernels composed. The row is
taken with the **same unmodified `tools/bench_decode.sh`** the 978.07 and
1304.06 rows were taken with - including its device-0 quirk (A27's harness
note), which is **not** fixed here, because patching the harness would make the
new row a measurement of a different thing.

## 4. RESULT - `pf_gdn_wu` at 2.21×, bitwise

Everything in this section was measured after §3 was committed at `696d50a`.

### 4.1 Time, and the decision-rule verdict

`ZE_AFFINITY_MASK=1 B70_PREFILL_PROFILE=1 b70-decode <RTN> --bench --pp 4096
--tg 8`, one run, **measured** (89.3 ms over 96 waits):

| | measured | |
|---|---:|---|
| `pf_gdn_wu`, ms per GDN layer per chunk | **0.9298** | |
| the same before the rewrite, A27's (`bb5f1d5`) | 2.0595 | **2.21×** |
| the same before the rewrite, this session's control (§0) | 2.0578 | 2.21× |
| pre-registered bar | ≤ 0.6 (≥ 3.4×) | **MISS by 1.55×** |
| point estimate | ~0.56 | missed by 1.66× |
| rate on 1.636 GFLOP/layer/chunk (derived) | **1.76 TFLOP/s** | was 0.79 |

**Decision-rule verdict: 2.21× falls in the 1.5-3× band → ADOPT, and attribute
the shortfall from the assembly.** The mapping was not iterated: §3.2's mapping
is what was built and 0.9298 is that build's first and only measurement.

Per-chunk consequence: 48 × (2.0595 − 0.9298) = **−54.2 ms/chunk**, **−108.5 ms
over a 4096-token prefill** (derived), against the −144 ms A27 derived from
assuming the rewritten scan's 2.90 TFLOP/s.

### 4.2 The shortfall, read off the Xe2 assembly

`ocloc` was re-run on the committed `pf_gdn_wy.cl` with `IGC_ShaderDumpEnable=1`
and the assembly of **both** entry points read - the tiled kernel and the legacy
one in the same binary, which makes this a controlled comparison rather than a
comparison against a remembered number. Nothing below is inferred from timing.

**The tiling did exactly what it was designed to do.** No spill (`-abortOnSpill 4`
is on the link line, so a spill would have failed the build), `grf_count 128`,
`slm_size 50688` - exactly §3.2's budget - `barrier_count 1`, and the hot loops
are the ones the mapping asked for:

| block | instructions | `mad` | FMA density | SLM loads | widens (`shl`) |
|---|---:|---:|---:|---:|---:|
| legacy, the `j` loop | **16** | 2 | **12.5%** | 3 (2 × `d16u32`, 1 × `d32`) | 5 |
| tiled, phase A (both blocks) | **45** | 32 | **71%** | 4 (2 × `d32x2`, 2 × `d32x4`) | **0** |
| tiled, phase B (high block) | **26** | 16 | **62%** | 3 (1 × `d32x4`, 2 × `d32x2`) | **0** |

The vector loads are the ones §3.2 fixed (`load.slm.d32x4` for the four `T`
values of a block, `load.slm.d32x2` for a lane's two columns of `vb` and of
`kb`), the widen is gone from the loop entirely, and the mirror pairing's
balance is visible in the dynamic count: subgroup `b` issues
`(4b+4)·45 + (60−8b)·26` loop instructions, i.e. **1740 at `b = 0` down to 1544
at `b = 7`, a 1.06× spread** (derived) where a tile of 8 consecutive positions
would have been 1.86×.

**Per lane, the work fell 5.06×. The measured time fell 2.21×. The whole gap is
two named terms, and the first is the larger:**

1. **The tiled kernel compiles SIMD16 where the legacy compiles SIMD32.** The
   `intel_reqd_sub_group_size(16)` attribute - copied from `pf_gdn_scan.cl`
   without being priced, and **not** part of what §3.2 pre-registered - halves
   the lanes an issued instruction covers. The machine's currency is issued
   instructions, not lane-operations:

   | | legacy | tiled |
   |---|---:|---:|
   | threads per layer per chunk | 12,288 (SIMD32) | 49,152 (SIMD16) |
   | loop instructions per thread | 16,640 | 1,642 (mean over `b`) |
   | **loop instructions issued** | **2.045e8** | **8.07e7** |
   | ratio | | **2.53×** |
   | per-lane instructions per `(i,x)` pair | 520 | 103 (**5.06×**) |

   2.53× of issued instructions against a measured 2.21× of time is the whole
   result. **The pre-registration's 3.7× was a count of lane-operations and
   silently assumed the SIMD width would not change; it did.**

2. **Issue rate is ~0.12-0.14 instructions per XVE per clock in BOTH kernels.**
   2.045e8 in 2.0578 ms = 9.94e10 /s and 8.07e7 in 0.9298 ms = 8.68e10 /s,
   against 256 XVEs × 2800 MHz = 7.17e11 XVE-clocks/s → **0.139 (legacy) and
   0.121 (tiled)** (derived). Both are ~4× below the rewritten scan's 0.52
   (`docs/prefill-gdn-scan-2026-09-05.md` §2.1), and the assembly says why:
   **the scan's `k` loops have the compile-time bound 128 and IGC unrolls them
   16-32×, while this kernel's `j` loops are bounded by the runtime `jl`/`jh`
   and are not unrolled at all.** Each iteration is therefore
   address → SLM load → dependent `mad` chain with a `goto` at the end, one
   un-pipelined SLM round trip per iteration and only 26-45 issue slots to
   cover it. This term did not get worse; it did not get better either, so the
   rewrite bought its 2.21× entirely from term 1's 2.53× of instruction count.

**What is left on this kernel, derived and NOT taken.** The sub-group attribute
is one line and the mapping does not depend on it: everything is computed from
`get_local_id(0)`, and at SIMD32 a thread's two 16-lane halves share the same
`b` (so `il`/`ih` and the `T` loads stay thread-uniform) and cover the same 64
contiguous columns (so `vbs`/`kbs` stay coalesced). At SIMD32 the same code
issues 24,576 × 1642 = **4.03e7 instructions, and at the measured issue rate
that is 0.465 ms** - under the pre-registered 0.6 bar. The estimate carries a
named risk: 32 fp32 accumulators cost twice the registers at SIMD32, so IGC may
refuse it at `grf_count 128` and fall back, and halving the thread count also
halves what is available to hide term 2's SLM latency. **It is not taken here**,
because the brief's rule is one defect, one fix, one measurement, and because
"remove an attribute and re-measure" is exactly the iterate-until-it-looks-good
the rule forbids. It is recorded as the next reader's cheapest lever, with
unrolling the `j` loop by a fixed factor plus a remainder as the second.

### 4.3 Numerics: bitwise, and therefore every gate is unchanged

**`gdn_wy_test`, measured** (`ZE_AFFINITY_MASK=1`), at all three widths `main`
runs, with the two kernels' outputs compared word for word:

| width | bar | measured |
|---|---|---|
| C = 256 | `w` and `u` bit-identical to `pf_gdn_wu_legacy` | **1,572,864 words each, identical** |
| C = 100 (ragged, L = 36) | the same | **614,400 words each, identical** |
| C = 4096 | the same | **25,165,824 words each, identical** |

The two runs are pre-filled with different patterns, so a word neither kernel
wrote would fail the comparison rather than pass it - the 3-D grid's coverage is
part of the same bar. The file's standing bars are unmoved: `u` **0 bf16 ulp,
bit-identical** to `gdn_chunk_ref::wu` fed the device's own `T` at every width;
`w` max **1 ulp** on `|ref| ≥ rms/4` at C = 256 and C = 100 and **2** at
C = 4096, against the bar of 2.

**`gdn_chunk_test`, measured, all six cases green** - and the band is identical
to A22's and A27's to four significant figures, which is what bit equality
predicts:

| case | tensor | max rel | mean rel | pre-registered fallback |
|---|---|---:|---:|---|
| 1 - vs the CPU fp32 recurrent reference | `gdn_state` | **3.506e-02** | **1.197e-03** | ≤ 7.0e-02 / ≤ 2.4e-03 |
| | `gdn_o` | 5.039e-02 | 1.228e-03 | |
| | `y` | 9.567e-02 | 9.790e-04 | |
| 2 - vs DEVICE `gdn_step` × 4096 | `gdn_state` | 3.506e-02 | 1.197e-03 | |
| 3 - `C = 1` × 4096 vs that oracle | `gdn_state` | 2.089e-02 | 8.519e-04 | |

Cases 4, 5 and 6 pass unchanged (2048 == 2 × 1024 == 32 × 64 bit-identical;
`C = 100` == 64 + 36; the 2 × 2048 walk twice from a zeroed state bitwise
identical in `gdn_state` / `conv_ring` / `gdn_o` / `y`). Even the diagnostic
counts are A27's: 17,234 / 17,236 `gdn_o` words over 1e-2 in cases 1 / 2 and
237,692 / 237,664 for `y`.

### 4.4 The arbiter: all four gates, both checkpoints

**Measured, `ZE_AFFINITY_MASK=1`, at the rewrite:**

| gate | Vishva | RTN |
|---|---|---|
| `prefill_gate_test` determined rows exact | **94/94** | **93/93** |
| undetermined | 2 (1 agree + 1 other member) | 3 (2 agree + 1 other member) |
| worst printed `gdn_state` cosine (layers 0, 8 … 56) | **0.999896667** (L32) | **0.999884022** (L48) |
| `prefill_consistency_test` (A26) | **9/9 cases**, 543 determined rows exact, 33 undetermined, 4 to the other member | **9/9 cases**, 558 determined rows exact, 18 undetermined, 8 to the other member |
| `prefill_determinism_test` | **9 cases × 3 runs bitwise** (`gdn_state`, `conv_ring`, `kv_k`, `kv_v`, control, `cur_token`) | - |

**18 of 18 consistency cases, 1101 determined rows across both checkpoints, zero
mismatches** - row for row the same table A27 §6 recorded, including which
divergence sits at which step (Vishva `code` at 41, `cjk` chunk 16 at 23; RTN
`prose` at 15 and 25, `cjk` at 47). A re-tiling that re-associated anything would
have re-tossed those sub-ulp coins, as A27 §6 observed the scan's did. **No
golden regression, so there is nothing to diagnose.**
