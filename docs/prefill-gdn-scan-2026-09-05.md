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
