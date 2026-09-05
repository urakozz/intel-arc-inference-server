# L1 core - pre-registration (buffer split totals, `gdn_chunk` numerics band)

Spec 2, plan 6b Tasks 1, 2, 6, 7, 8. Written **before** any of the three
artefacts exists and before any measurement, so that every number below can be
scored rather than rationalised. It lives in `docs/` and not in
`.superpowers/sdd/` because that directory's `.gitignore` is `*` and an
uncommitted pre-registration is not one (precedent: `4009100`,
`docs/probe-gemm-batched-2026-09-05.md`).

Conditions for everything measured against this file: the box
(box), card 1 (`ZE_AFFINITY_MASK=1`), `tools/box.sh`, JOBS 44,
Release, ocloc AOT `bmg-g31`, `-cl-fp32-correctly-rounded-divide-sqrt`,
`-cl-denorms-are-zero` absent.

---

## 1. Task 1 - `PrefillScratch` at `kC = 2048`, every byte recomputed

Ruling **A13** moved the chunk width from 4096 to 2048 and requires the byte
figures to be **recomputed, not scaled**; ruling **A14** retired the `M = 64`
attention route that plan 6b's `attn_q` / `attn_gate` / `attn_part` fields
existed to feed, and replaced it with the composed path whose scratch plan
6d-composed sizes. Both are applied here, so plan 6d does not have to resize
this struct when it lands.

Constants: `model::Qwen35` (`kHidden` 5120, `kIntermediate` 17408, gate‖up
`N` 34816, `kVocab` 248320, `kGdnVHeads` 48, `kGdnHeadDim` 128, `kFaQHeads` 24,
`kFaHeadDim` 256), `max_len = 16384`, `kC = 2048`, `kGdnChunk = 64`,
`kNormGroups = 20`, `kSHeads = 6` (one GQA group - plan 6d's `Lh`).

| field | shape | bytes |
|---|---|---:|
| `ids` | uint32 `[2048]` (Host) | 8,192 |
| `resid` | bf16 `[2048][5120]` | 20,971,520 |
| `x` | bf16 `[2048][17408]` | 71,303,168 |
| `partials` | fp32 `[2048][34816]` (R1: one S=1 rectangle) | 285,212,672 |
| `ab_out` | fp32 `[2048][128]` | 1,048,576 |
| `norm_sumsq` | fp32 `[20][2048]` | 163,840 |
| `gdn_o` | fp32 `[2048][48][128]` | 50,331,648 |
| `mixer_out` | bf16 `[2048][6144]` | 25,165,824 |
| `logits` | fp32 `[1][248320]` | 993,280 |
| `argmax_part` | fp32 `[1][243][2]` | 1,944 |
| `dequant` | bf16 `[5120][34816]` (chunk-independent) | 356,515,840 |
| `gdn_xb` | bf16 `[2048][10240]` | 41,943,040 |
| `gdn_seed` | bf16 `[3][10240]` | 61,440 |
| `gdn_g` | fp32 `[2048][48]` | 393,216 |
| `gdn_beta` | fp32 `[2048][48]` | 393,216 |
| `gdn_A` | fp32 `[32][48][64][64]` | 25,165,824 |
| `gdn_A2` | fp32 `[32][48][64][64]` | 25,165,824 |
| `gdn_w` | bf16 `[2048][48][128]` | 25,165,824 |
| `gdn_u` | bf16 `[2048][48][128]` | 25,165,824 |
| `pf_q` | bf16 `[2048][24][256]` (A9: bf16, not fp32) | 25,165,824 |
| `pf_attn` | bf16 `[2048][24][256]` | 25,165,824 |
| `pf_s` | fp32 `[6][2048][16384]` | 805,306,368 |
| `pf_p` | bf16 `[6][2048][16384]` | 402,653,184 |
| `pf_o` | fp32 `[24][2048][256]` | 50,331,648 |
| `pf_rowsum` | fp32 `[24][2048]` | 196,608 |
| **total** | | **2,263,990,168** |

Cross-checks, both of which must hold or the table is wrong:

- the 6b-owned rows alone (everything above except the six `pf_*` rows) sum to
  **955,170,712 B**;
- the six `pf_*` rows sum to **1,308,819,456 B**, which is exactly plan
  6d-composed's "total added" line (`pf_kt` excluded: the batched-GEMM probe
  measured `transB` native and bitwise identical, so the transpose fallback is
  not built).

The three retired fields would have added **408,944,640 B** (`attn_part`
405,798,912 + `attn_q` 1,572,864 + `attn_gate` 1,572,864), which is exactly
plan 6d's "retired" line; **2,263,990,168 = 955,170,712 + 408,944,640 +
899,874,816**, 6d's stated net.

**Not scaled from 4096.** For the record, plan 6b's 4096 table totalled
1,961,713,560 B; nothing here is that number halved.

**Residency prediction (derived, to be scored by `buffers_test`'s printout):**
persistent 1,240,465,536 + decode scratch 68,652,864 + prefill scratch
2,263,990,168 = **3,573,108,568 B of engine allocations**, on top of ~15.5 GB
of weights → **~19.1 GB resident** during prefill at `max_len` 16384. Ruling
R7 keeps `PrefillScratch` lazy, so a decode-only `Engine` is byte-identical to
today's.

**Decode invariants that must not move (pinned, not predicted):**
`PersistentBuffers::bytes() == 1,240,465,536`,
`DecodeScratch::bytes() == 68,652,864`, `kernel_count == 774`,
`modules.size() == 19`, both golden gates' determined rows exact,
`replay_determinism_test` bitwise across all three runs.

---

## 2. Task 8 - the `gdn_chunk` numerics band

### 2.1 The metric, fixed before the run

For tensors `got` and `ref`,

```
rms  = sqrt(mean(ref[i]^2))
rel  = |got[i] - ref[i]| / max(|ref[i]|, rms)      # RMS-floored
```

`max rel` and `mean rel` are over every element. The RMS floor is what makes
"max relative difference" meaningful on a 786,432-element state whose smallest
entries are ~0; it is the same device `attn_test.cc` uses for `attn_out`.

### 2.2 What actually differs, and by how much

The chunked (WY/UT) form is algebraically equal to the recurrence but is not
the same floating-point program. Three named differences, in decreasing order
of predicted contribution:

**(a) Q1-Q4 - four bf16 roundings the recurrence does not have.** The plan's
algorithm section keeps FLA's rounds: `vb = rne(v·β)` (Q1, `wy_fast.py:88`),
`kb = rne(k·β·e^{gc})` (Q2, `:110`), `u = rne(T·vb)` (Q3, `:89`),
`w = rne(T·kb)` (Q4, `:112`). bf16 has an 8-bit significand, so one rounding is
a relative perturbation of at most **2⁻⁹ = 1.95e-3**. The recurrence computes
the corresponding quantity - `Δ = (v − kv)·β` - entirely in fp32 from the same
bf16 `v` and `kf`, so these four are a genuine addition and **they dominate the
band**. Ruling R8's two deviations (fp32 `T`, fp32 `v_new`) remove two more
that FLA has; they are the reason the band is not larger still.

**(b) The decay/update reassociation.** Decode interleaves
`S ← S·e^{g_i}` with the rank-1 update per position (`gdn_step.cl:318-349`).
The chunk applies `S ← S·e^{gl}` once and then adds
`Σ_i kf_i ⊗ (vn_i·e^{gl−gc_i})`. Exact in real arithmetic; in fp32 it replaces
`L` sequential multiplies by one, and `e^{gl−gc_i}` by a product of `e^{g_j}`.
`exp` is 3 ulp in OpenCL, so this contributes **~3·2⁻²⁴ per position** - three
orders of magnitude below (a) and listed for completeness.

**(c) `o` reads the chunk-start state.** `o[i] = (q_i·S)·e^{gc_i} + Σ_{j≤i}
A2[i][j]·vn_j` is the same value decode computes as `q_i·S_new`, but reached by
a different route: decode contracts once against an already-updated 128×128
state, the chunk contracts against the chunk-start state and adds an
intra-chunk triangular term. Same 128 terms in the first part (identical band
tree, deliberately - `pf_gdn_scan` reuses `gdn_step.cl:52-65`'s trees), a new
`L`-term fp32 sum in the second.

### 2.3 The pre-registered numbers

Synthetic fixture (fixed here so the band is reproducible): `qkvz_partials`
fp32 `N(0, 0.5)`, `ab_out` fp32 `N(0, 1)`, conv weights `N(0, 0.5)`,
`negA ∈ [−4, −1]`, `dt_bias = −4` - i.e. a gate in the long-memory regime
(`decay ≈ 0.92…0.99`) that the real checkpoint is in, not a degenerate one
where the state resets every position and every path agrees trivially.

| comparison | quantity | predicted | band (pass without comment) |
|---|---|---:|---|
| **1.** `C = 4096` vs CPU fp32 recurrent `gdn_ref` | `gdn_state` max rel | **2e-3** | 3e-4 … 8e-3 |
| | `gdn_state` mean rel | **1e-4** | 1e-5 … 1e-3 |
| | `gdn_o` max rel | **3e-3** | 5e-4 … 1e-2 |
| | `y` max rel | **4e-3** | 5e-4 … 1.5e-2 |
| **2.** `C = 4096` vs device `gdn_step` ×4096 | `gdn_state` max rel | **2e-3** | 3e-4 … 8e-3 |
| | `gdn_state` mean rel | **1e-4** | 1e-5 … 1e-3 |
| **3.** `C = 1` ×4096 vs device `gdn_step` ×4096 | `gdn_state` max rel | **1e-3** | 1e-4 … 5e-3 |

Comparison 2 is predicted **at or below** comparison 1, because it removes the
host-vs-device `exp`/`log1p` slack (3 and 2 ulp) and leaves only the chunked-
vs-recurrent difference - that separation is the whole reason the plan asks for
both.

**Plan 6b's `C = 1` bar of ≤ 1e-5 is corrected here, before the measurement,
and the correction is derived rather than discovered.** At `C = 1` the chunk
algebra degenerates exactly as the plan says (`A` empty, `T = I`, `u = vb`,
`w = kb`, `A2` one diagonal entry) - but `u` and `w` are still Q1/Q2, i.e.
`u = rne(v·β)` and `w = rne(kf·β·e^{g})` where decode forms `(v − kv)·β` in
fp32 with no such rounding. One bf16 half-ulp on the `v·β` term is 1.95e-3
relative *to that term*; carried into the state through `S ← S·e^{g} + kf ⊗ vn`
and averaged over the ≈ 1/(1−decay) positions the gate keeps, it lands near
**1e-3**. 1e-5 would require dropping Q1/Q2, which is a change to the
algorithm the plan fixed, not a tolerance choice. The degenerate-path bug the
plan wanted this case to find is still found: a wrong `A2` diagonal, a wrong
`T` convention or a wrong chunk-start `S` all move this figure by orders of
magnitude, not by a factor of two.

### 2.4 Bars that are NOT bands (exact, and a failure is a bug)

These are pre-registered as **bit-exact** because every step is fp32 `fma` in a
stated order with at most one RNE at the end, evaluated on the same device in
both paths:

- `conv_ring`'s three live slots after a chunk vs the recurrent walk's ring -
  bit-exact (`raw_b` is `rne_bf16` of a buffer nothing else writes);
- multi-chunk == single-chunk: `C = 4096` in one call vs 4×1024 vs 64×64 -
  `gdn_state` and `conv_ring` bit-identical (this is what replaces the
  `M + 3 ≤ RING` argument);
- `C = 100` in one call vs 64 + 36 - bit-identical;
- determinism: the same call twice from a zeroed state - `gdn_state`,
  `conv_ring`, `gdn_o`, `y` all bit-identical (no fp atomic exists in any of
  the ten launches);
- `pf_gdn_solve`: `(I − A)·T == I` to `1e-9` in host fp64, `T[i][i] == 1.0f`
  and `T[i][j] == 0.0f` for `j > i`, exactly;
- `pf_gdn_l2norm` vs `gdn_ref`'s `qf`/`kf`: bit-exact, and the stored q word is
  the **unscaled** one (P6).

`xb` vs `gdn_ref`: **≤ 2 bf16 ulp** (the only transcendental in the chain is
`silu`'s `exp`).

### 2.5 The stop rule

If a measured figure exceeds its predicted value by **more than 10×**, the
brief forbids tuning: report the layer/position where the divergence first
exceeds the band and stop. The tripwire in the test itself stays at the plan's
**max rel > 1e-2 or any non-finite value**, which is a transcription-error
detector and not a grade.

---

## 3. Launch-count contract (Task 8, for Task 14's arithmetic)

`gdn_chunk` is **ten** launches per GDN layer per chunk, in this order:
`pf_gdn_seed`, `pf_gdn_conv`, `pf_gdn_l2norm`, `pf_gdn_gate`, `pf_gdn_A`,
`pf_gdn_solve`, `pf_gdn_wu`, `pf_gdn_A2`, `pf_gdn_scan`, `pf_gated_head`
(ruling R3 - the gated head is `gdn_chunk`'s, not the caller's). The test
asserts the count through `Context::launches()`.
