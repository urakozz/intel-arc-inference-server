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
