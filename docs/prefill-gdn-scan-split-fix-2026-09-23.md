# The split GDN scan's golden-gate failure: diagnosis and fix (2026-09-23)

Status: **the opt-in `dpas_split` scan now passes every gate it is registered
against, and is 2.33x faster than the vector scan on its own profile row.** The
production default is unchanged and still `pf_gdn_scan`; promoting it is a
separate decision with its own evidence.

Device 0 only, one job at a time (device 1's `zeInit` segfaults). Every number
below is labelled measured or derived; nothing here is estimated.

---

## 0. What was wrong, and the rule that comes first

`pf_gdn_scan_dpas_split` (`B70_PREFILL_GDN_SCAN=dpas_split`) failed
`prefill_gate_l0_test` - 92 of 93 determined rows, the code prompt's L60
`gdn_state` cosine **0.996344994** against a "> 0.999" bar - deterministically,
to nine digits, on two cards
(`docs/superpowers/specs/2026-09-22-prefill-parity-program-design.md` §11;
evidence `$HOME/split-dev0.log`).

That §11 also carries the expensive lesson: the 2026-09-21 record which declared
this kernel green was executing `pf_gdn_scan`. **So the first change here is not
arithmetic at all.** `gdn.cc` now records the entry string every scan launch is
built with, and exposes it beside the resolver:

```
const char* gdn_scan_entry_name();      // what the selector resolves to
const char* gdn_scan_launched_entry();  // what a launch was ACTUALLY built with
```

`prefill_gate_test`, `prefill_determinism_test`, `prefill_consistency_test`,
`prefill_replay_test` and `gdn_chunk_test`'s band print both, and the four
non-band tests `CHECK` that they agree. Every result in this document was
produced by a run that named its own kernel; the naming line is quoted with each
one. Neither accessor is a configuration API - the selector stays private to
`gdn.cc`, and the mutation-sensitive dispatch rows in `gdn_chunk_test` are
untouched.

## 1. Step 1 - the diagnosis, before any fix

The suspect was pre-identified: in
```
o[i][x] = ( SUM_k q[i][k]*S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
```
`S` and the update `D` carry hi/lo BF16 limbs with fp32 masters, but **both**
operands of the intra-chunk `A2*vn` term were single BF16 - and that term is a
direct additive contributor to `o`, which every later layer reads.

Four throwaway entries (commit `694b724`, removed in `the commit that adds this document`) computed that one
term in **plain scalar fp32, no DPAS**, from an fp32 `vn` staged in SLM and `A2`
read fp32 from global, each operand independently either left fp32 or put through
`bf16f(rne_bf16(.))` - the exact rounding the DPAS path applies. Nothing else in
the kernel changed: the same `S`/`D` limbs, the same RNE conversions, the same
grid and clamps. A 2x2, each cell one `prefill_gate_l0_test` run on device 0:

| A2 | vn | prose | code (L60) | cjk | tokens | log |
|---|---|---|---|---|---|---|
| bf16 | bf16 | 0.999904615 | **0.997517446** | 0.999879361 | **92/93** | `$HOME/fix-diag-bf16.log` |
| bf16 | fp32 | 0.999904615 | **0.997517446** | 0.999879361 | **92/93** | `$HOME/fix-diag-vn.log` |
| fp32 | bf16 | 0.999903435 | 0.999189068 | 0.999903208 | 93/93 | `$HOME/fix-diag-a2.log` |
| fp32 | fp32 | 0.999903435 | 0.999189068 | 0.999903208 | 93/93 | `$HOME/fix-diag-f32.log` |

Each run printed, e.g., `gdn scan entry LAUNCHED:
pf_gdn_scan_dpas_split_diag_f32`. Three things follow, in order of consequence:

1. **Diagnosis confirmed.** With `A2*vn` in fp32 the gate is 93/93 and every
   cosine clears the bar - the failure is entirely in that term.
2. **It is `A2`'s rounding, not `vn`'s.** The two fp32-`A2` rows are identical to
   each other and to the vector kernel's own printed cosines
   (0.999903435 / 0.999189068 / 0.999903208); the two bf16-`A2` rows are
   identical to each other and fail. `vn`'s BF16 rounding does not move a single
   printed digit in either `A2` state. This is measured; the mechanism is not
   established here - the naive error terms `SUM_j dA2*vn` and `SUM_j A2*dvn` are
   the same order, so something about `A2`'s row structure (its
   `exp(gc_i - gc_j)` decay spans decades inside one row) must be doing it.
3. **The control is faithful.** The bf16/bf16 cell reproduces the failure
   (0.997517446 vs the shipped kernel's 0.996344994, 92/93 with the same cjk row
   9 flip), so the scalar harness differs from the DPAS kernel only in summation
   order - which is worth 2.4e-3 of `1 - cos` on its own and is the reason the
   fixed kernel below lands slightly differently again.

## 2. Step 2 - what was split, and why only one operand

`A2` gets the treatment `S` and `D` already have - `hi = rne(x)`,
`lo = rne(x - bf16(hi))`, staged as two VNNI blocks - and **`vn` keeps its single
BF16**, because the 2x2 above measured `vn`'s contribution at exactly nil. That
is the cheaper two-chain option:

```
A2*vn ~= A2_hi*vn + A2_lo*vn        (both products exact: bf16 x bf16 -> fp32)
```

against the three chains a both-operand split would need. The low chain starts at
zero rather than on top of the accumulated `q.S` term, and the two meet once in
fp32 after the last position block - the same "combine exactly once" the state
chains use. Every existing limb scheme, fp32 master and RNE conversion is
untouched.

### Cost (MEASURED, from the built binary's ZEInfo and vISA dump)

| | before | after |
|---|---:|---:|
| `slm_size` | 35,328 B | **40,448 B** |
| ... `SbHi`/`SbLo` | 17,408 | 17,408 |
| ... `VNb` | 4,352 | 4,352 |
| ... `DtHi`/`DtLo` | 8,448 | 8,448 |
| ... `A2b` -> `A2bHi`/`A2bLo` | 5,120 | **10,240** |
| `grf_count` | 128 | **128** |
| `simd_size` / `barrier_count` / `has_dpas` | 16 / 1 / true | **16 / 1 / true** |
| spill | none | **none** |
| static `dpas.8x8` in the ISA | - | **50** |
| DPAS per work-item per sub-chunk (derived) | 50.5 avg | **53.0 avg** |

Splitting both operands would have cost 44,800 B and 55.5 DPAS; this is 4,352 B
and 2.5 DPAS cheaper for the same measured gate result. "No spill" is not an
absence of evidence: `-abortOnSpill 4` is on the vISA option line
(`//.full_options` in the IGC dump) and the build succeeded, the ZEInfo declares
no scratch or private memory buffer, and the only `spill`/`scratch` strings in
the 1,423-instruction listing are that option and vISA's standard `%scratchloc`
declare. Dump: `/tmp/igcdump-20260923` on the box; ZEInfo read straight out of
`build/kernels/pf_gdn_scan.bin`.

## 3. The gates

All on device 0 (`ZE_AFFINITY_MASK=0`), serialised, `tools/box.sh`, `REMOTE_DIR`
default, JOBS 44.

### 1. Dispatch proof - mandatory, and present in every row below

Each gate run prints `gdn scan selector: B70_PREFILL_GDN_SCAN=<v> -> entry <e>`
before the walk and `gdn scan entry LAUNCHED: <e>` after it, the latter read back
from the pointer `gdn_chunk` handed to `KernelCache` at the scan launch. The four
`gdn_scan_*_dispatch_test` rows (output-mutation-sensitive, unchanged) passed
alongside.

### 2. `prefill_gate_l0_test` under `dpas_split` - **PASS** (`$HOME/fix-split-gate.log`)

```
gdn scan selector: B70_PREFILL_GDN_SCAN=dpas_split -> entry pf_gdn_scan_dpas_split
gdn scan entry LAUNCHED: pf_gdn_scan_dpas_split      (printed for all three prompts)
  prose  worst: L33 at 0.999912369
  code   worst: L60 at 0.999712545
  cjk    worst: L33 at 0.999901750
  TOTAL: 93/93 determined rows exact, 3 undetermined (2 agree + 1 other member)
```

Every printed cosine is strictly above 0.999. The row that failed at
0.996344994 now reads 0.999712545 - **better than the vector kernel's
0.999189068**, and better than the fp32 scalar diagnostic, which reproduced the
vector value exactly. The remaining difference between them is summation order:
DPAS sums the A2 term in 16-wide blocks where both the vector kernel and the
diagnostic run one 64-long sequential `fma` chain. Measured; not a guarantee.

### 3. Determinism, consistency, replay under `dpas_split` - **5/5 PASS** (`$HOME/fix-split-suite.log`)

| test | result |
|---|---|
| `prefill_replay_test` | Passed 90.57 s |
| `prefill_consistency_l0_test` | Passed 65.31 s |
| `prefill_determinism_l0_test` | Passed 80.73 s - 9 cases x 3 runs bitwise |
| `prefill_consistency_test` (sycl-tla) | Passed 59.98 s |
| `prefill_determinism_test` (sycl-tla) | Passed 63.85 s |

Each printed `gdn scan entry LAUNCHED: pf_gdn_scan_dpas_split`.

### 4. `gdn_chunk_test` under both selectors - **PASS**, bands recorded

Both runs printed `scan entry: B70_PREFILL_GDN_SCAN=<v> -> LAUNCHED <e>`
(`$HOME/fix-split-band.log`, `$HOME/fix-vector-band.log`). Case 1, the chunked
4096-position walk against the CPU fp32 recurrence:

| tensor | vector | dpas_split (fixed) |
|---|---|---|
| `gdn_state` | max rel 3.506e-02, mean 1.197e-03 | max rel 3.506e-02, mean 1.197e-03 |
| `gdn_o` | max rel 5.039e-02, mean 1.228e-03 | max rel 7.097e-02, mean 1.443e-03 |
| `y` | max rel 9.567e-02, mean 9.790e-04 | max rel 1.067e-01, mean 1.132e-03 |

Case 3 (against decode's own `gdn_step_M1`): `gdn_state` 2.089e-02 / 2.087e-02,
`gdn_o` 2.669e-02 / 2.668e-02. The test's own gating tripwire is
`max_rel < 2.5e-1` and both selectors hold it with a wide margin. **Nothing was
loosened.**

Two honest notes on the band bar quoted for this gate:

* **`rel L2 <= 2.0e-3` holds** on the closest quantity `gdn_chunk_test` prints
  (mean rms-floored relative difference): 1.443e-03 split, 1.228e-03 vector.
* **`max rel <= 1.0e-2` is exceeded by the vector default too** (5.039e-02 on
  `gdn_o`) and has been since 2026-09-05 - a pre-registration MISS recorded in
  `report()`'s own comment in `gdn_chunk_test.cc` and in
  `docs/prefill-l1-preregistration-2026-09-05.md` §2.3. The meaningful reading is
  therefore split-vs-vector: the split's `gdn_o` max rel is 1.41x the vector's
  and its mean 1.18x, on a fixture whose synthetic state makes the BF16 limbs
  work harder than the real prompts do, while the real-prompt gate it exists to
  protect passes with margin.
* **A correction.** The note in `gdn.cc` that reverted the 2026-09-22 default
  promotion also blamed "`gdn_chunk_test`'s pre-registered band (rel L2
  2.7395e-03, max rel 2.9650e-02)". Those are `attn_chunk_test`'s numbers, not
  `gdn_chunk_test`'s: the identical pair is printed under the **vector** default
  in `$HOME/spec21-suite.log` (2026-09-18) and `$HOME/suite-igc2415.log`
  (2026-09-19), attention runs no GDN kernel, and `gdn_chunk_test` prints no
  rel L2 at all. One of that revert's two stated reasons never existed; the other
  one - the token gate - did, and is what this document fixes. The comment is
  corrected in the tree.

### 5. The vector default is untouched - **PASS** (`$HOME/fix-vector-gate.log`)

```
gdn scan selector: B70_PREFILL_GDN_SCAN=vector -> entry pf_gdn_scan
gdn scan entry LAUNCHED: pf_gdn_scan
  prose 0.999903435 (L33)   code 0.999189068 (L60)   cjk 0.999903208 (L60)
  TOTAL: 93/93 determined rows exact
```

Bit-identical to the pre-fix reference row, to nine digits, and re-run once more
on the final tree after the diagnostic entries were removed
(`$HOME/fix-final-vector.log`; the split side re-ran there too at 7/7,
`$HOME/fix-final-split.log`, same three cosines).
`gdn_scan_split_test` (which launches the split entry directly) also passed, with
a new fixture for the A2 chain: A2 = 22.65625 = 22.625 + 2^-5 makes `o`
0.000232638 where a high-only A2 chain gives 0.000354708 - 1.221e-04 apart, 24x
its 5e-6 bar. Verified RED by deleting the low chain (fails at
`gdn_scan_split_test.cc:125`, and only there), GREEN with it restored.

### 6. Speed - the point of the kernel

`b70-decode <snapshot> --bench --pp 4096 --pp-backend l0` with
`B70_PREFILL_PROFILE=1`, four processes in **vector, split, split, vector**
order, idle box, device 0. `gdn_scan` row, L0 GPU timestamp sum over 96 launches:

| run | selector | `gdn_scan` L0 GPU ms | instrumented call wall ms |
|---|---|---:|---:|
| 1 | vector | 297.5 | 2809.1 |
| 2 | dpas_split | 128.2 | 2595.7 |
| 3 | dpas_split | 127.1 | 2647.0 |
| 4 | vector | 297.7 | 2807.9 |

**Mean 297.6 ms vector vs 127.65 ms split: 2.33x, -170.0 ms** (derived from the
four measured rows). The fix costs essentially nothing against the broken split's
127.2 ms measured on 2026-09-22 - that comparison crosses days and builds, and
the vector control moved 304.7 -> 297.6 over the same interval, so read the
within-session pairs, not the cross-day delta. No other `gdn_*` row moved more
than 1.5 ms (`gdn_solve` 69.5 -> 71.0, `gdn_wu` 82.5 -> 83.5, the rest under
0.5 ms), which is the usual profiling coupling and not attributed to the scan.
Logs: `$HOME/fix-profile-{vector,dpas_split}-*.log`.

**It is not slower than vector. It has a reason to exist.**

## 4. Verdict

The diagnosis was right and narrower than expected: one operand of one term.
The fixed `pf_gdn_scan_dpas_split` clears every registered gate with its dispatch
proven inside each run, keeps its 2.33x on the scan row, costs 5,120 B more SLM,
128 GRF and no spill, and leaves the vector default bit-identical.

What it is **not**: an fp32-equivalence implementation. `vn` is still rounded once
to BF16 in the `A2*vn` term and the state path still carries two-limb BF16, all
measured as harmless on three prompts of one checkpoint and on `gdn_chunk_test`'s
fixture - not proven harmless in general. The default stays `vector`; anyone
proposing to promote it re-runs these gates and proves the dispatch, because a
green gate is not evidence unless the run says which kernel it dispatched.
