# The split delta net scan: a failed gate, its diagnosis, and the fix (2026-09-23)

The split-BF16 DPAS scan failed a golden token gate, was diagnosed to a single
operand of a single term, was fixed, and now clears every gate it is registered
against while running **2.33x faster** than the vector scan on its own profile
row. It is the default prefill scan as of this date, and it is the one
approximate kernel in that path.

The arithmetic being fixed, and the selector that switches it, are described in
[prefill-gdn-scan-split-2026-09-20.md](prefill-gdn-scan-split-2026-09-20.md).

Every number below is labelled measured or derived. Nothing here is estimated.

## 0. The rule that comes first: prove the dispatch

The kernel failed `prefill_gate_l0_test` at 92 of 93 determined rows, with the
code prompt's layer 60 `gdn_state` cosine at **0.996344994** against a bar of
0.999, deterministically, to nine digits, on two cards.

That failure was only visible because of something that had gone wrong two days
earlier: **a run that declared this kernel green had been executing the vector
kernel.** The selector resolved correctly, the gates ran, the cosines printed,
and none of it was about the kernel under test. Two runs, three days apart,
printed bit-identical cosines, and that is what gave it away.

So the first change here is not arithmetic at all. The scan dispatcher now
records the entry string every launch is actually built with, and exposes it
beside the resolver:

```c
const char* gdn_scan_entry_name();      // what the selector resolves to
const char* gdn_scan_launched_entry();  // what a launch was ACTUALLY built with
```

`prefill_gate_test`, `prefill_determinism_test`, `prefill_consistency_test`,
`prefill_replay_test` and `gdn_chunk_test` all print both, and the four
non-band tests assert that the two agree. Every result in this document was
produced by a run that named its own kernel, and the naming line is quoted with
each one.

Neither accessor is a configuration API: the selector stays private to the
dispatcher, and the mutation-sensitive dispatch tests are untouched.

## 1. Diagnosis, before any fix

The suspect was identified before it was tested. In

```
o[i][x] = ( SUM_k q[i][k]*S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
```

`S` and the update `D` carry hi/lo BF16 limbs with fp32 masters, but **both**
operands of the intra-chunk `A2*vn` term were single BF16, and that term adds
directly into `o`, which every later layer reads.

Four throwaway entries, since removed, computed that one term in plain scalar
fp32 with no DPAS, from an fp32 `vn` staged in SLM and `A2` read fp32 from
global, each operand independently either left fp32 or put through the exact
round-to-nearest-even conversion the DPAS path applies. Nothing else changed:
same `S` and `D` limbs, same conversions, same grid, same clamps. A 2x2, each
cell one full golden gate run:

| A2 | vn | prose | code (L60) | cjk | tokens |
|---|---|---|---|---|---|
| bf16 | bf16 | 0.999904615 | **0.997517446** | 0.999879361 | **92/93** |
| bf16 | fp32 | 0.999904615 | **0.997517446** | 0.999879361 | **92/93** |
| fp32 | bf16 | 0.999903435 | 0.999189068 | 0.999903208 | 93/93 |
| fp32 | fp32 | 0.999903435 | 0.999189068 | 0.999903208 | 93/93 |

Each run printed its own launched entry. Three things follow, in order of
consequence:

1. **Diagnosis confirmed.** With `A2*vn` in fp32 the gate is 93/93 and every
   cosine clears the bar. The failure is entirely in that term.
2. **It is `A2`'s rounding, not `vn`'s.** The two fp32-`A2` rows are identical
   to each other and to the vector kernel's own printed cosines; the two
   bf16-`A2` rows are identical to each other and both fail. `vn`'s BF16
   rounding does not move a single printed digit in either state. This is
   measured, and the mechanism is **not** established: the naive error terms
   `SUM_j dA2*vn` and `SUM_j A2*dvn` are the same order, so something about
   `A2`'s row structure, whose `exp(gc_i - gc_j)` decay spans decades inside one
   row, must be doing it.
3. **The control is faithful.** The bf16/bf16 cell reproduces the failure
   (0.997517446 against the shipped kernel's 0.996344994, 92/93 with the same
   flipped row), so the scalar harness differs from the DPAS kernel only in
   summation order, which is worth 2.4e-3 of `1 - cos` on its own.

## 2. The fix: one operand, not both

`A2` gets the treatment `S` and `D` already have: `hi = rne(x)`,
`lo = rne(x - bf16(hi))`, staged as two VNNI blocks. **`vn` keeps its single
BF16**, because the 2x2 measured its contribution at exactly nil. That is the
cheaper two-chain option:

```
A2*vn ~= A2_hi*vn + A2_lo*vn        (both products exact: bf16 x bf16 -> fp32)
```

against the three chains a both-operand split would need. The low chain starts
at zero rather than on top of the accumulated `q.S` term, and the two meet once
in fp32 after the last position block, the same combine-exactly-once the state
chains use. Every existing limb scheme, fp32 master and conversion is
untouched.

### Cost, measured from the built binary

| | before | after |
|---|---:|---:|
| `slm_size` | 35,328 B | **40,448 B** |
| of which `SbHi`/`SbLo` | 17,408 | 17,408 |
| of which `VNb` | 4,352 | 4,352 |
| of which `DtHi`/`DtLo` | 8,448 | 8,448 |
| of which `A2b` becoming `A2bHi`/`A2bLo` | 5,120 | **10,240** |
| `grf_count` | 128 | **128** |
| `simd_size` / `barrier_count` / `has_dpas` | 16 / 1 / true | **16 / 1 / true** |
| spill | none | **none** |
| static `dpas.8x8` in the ISA | | **50** |
| DPAS per work-item per sub-chunk (derived) | 50.5 avg | **53.0 avg** |

Splitting both operands would have cost 44,800 B and 55.5 DPAS: this is
4,352 B and 2.5 DPAS cheaper for the same measured gate result.

"No spill" is not an absence of evidence. `-abortOnSpill 4` is on the compiler
option line and the build succeeded, the binary metadata declares no scratch or
private memory buffer, and the only occurrences of `spill` or `scratch` in the
1,423-instruction listing are that option and the compiler's standard
`%scratchloc` declaration.

## 3. The gates

All serialised on one card, each with its dispatch printed.

### Dispatch proof, in every row below

Each run prints the selector resolution before the walk and the launched entry
after it, the latter read back from the pointer the launcher handed the kernel
cache:

```
gdn scan selector: B70_PREFILL_GDN_SCAN=dpas_split -> entry pf_gdn_scan_dpas_split
gdn scan entry LAUNCHED: pf_gdn_scan_dpas_split
```

The four output-mutation-sensitive dispatch tests passed alongside, unchanged.

### Golden token gate under the split entry: PASS

```
  prose  worst: L33 at 0.999912369
  code   worst: L60 at 0.999712545
  cjk    worst: L33 at 0.999901750
  TOTAL: 93/93 determined rows exact, 3 undetermined (2 agree + 1 other member)
```

Every printed cosine is strictly above 0.999. The row that failed at
0.996344994 now reads **0.999712545**, which is better than the vector kernel's
own 0.999189068 and better than the fp32 scalar diagnostic, which reproduced
the vector value exactly. The remaining difference between them is summation
order: DPAS sums the A2 term in 16-wide blocks where the vector kernel and the
diagnostic run one 64-long sequential `fma` chain. Measured, not guaranteed.

### Determinism, consistency and replay under the split entry: 5/5 PASS

| test | result |
|---|---|
| `prefill_replay_test` | Passed, 90.57 s |
| `prefill_consistency_l0_test` | Passed, 65.31 s |
| `prefill_determinism_l0_test` | Passed, 80.73 s, 9 cases by 3 runs bitwise |
| `prefill_consistency_test` (sycl-tla) | Passed, 59.98 s |
| `prefill_determinism_test` (sycl-tla) | Passed, 63.85 s |

### The chunk band test, under both selectors

The chunked 4096-position walk against the CPU fp32 recurrence:

| tensor | vector | split, fixed |
|---|---|---|
| `gdn_state` | max rel 3.506e-02, mean 1.197e-03 | max rel 3.506e-02, mean 1.197e-03 |
| `gdn_o` | max rel 5.039e-02, mean 1.228e-03 | max rel 7.097e-02, mean 1.443e-03 |
| `y` | max rel 9.567e-02, mean 9.790e-04 | max rel 1.067e-01, mean 1.132e-03 |

Against the decode path's own single-position kernel: `gdn_state` 2.089e-02 and
2.087e-02, `gdn_o` 2.669e-02 and 2.668e-02. The test's gating tripwire is
`max_rel < 2.5e-1` and both selectors hold it with a wide margin. **Nothing was
loosened.**

Two honest notes on the bounds quoted for this fixture:

- The relative-L2 bound of 2.0e-3 holds on the closest quantity the test
  prints: 1.443e-03 split, 1.228e-03 vector.
- The max-relative bound of 1.0e-2 is **exceeded by the vector default too**,
  at 5.039e-02 on `gdn_o`, and has been since 2026-09-05. The meaningful
  reading is therefore split against vector: the split's `gdn_o` max relative
  error is 1.41x the vector's and its mean 1.18x, on a fixture whose synthetic
  state works the BF16 limbs harder than the real prompts do, while the real
  prompt gate it exists to protect passes with margin.

### The vector entry is untouched: PASS

```
gdn scan entry LAUNCHED: pf_gdn_scan
  prose 0.999903435 (L33)   code 0.999189068 (L60)   cjk 0.999903208 (L60)
  TOTAL: 93/93 determined rows exact
```

Bit-identical to the pre-fix reference row, to nine digits.

The direct scan test gained a fixture aimed at the A2 chain specifically:
`A2 = 22.65625 = 22.625 + 2^-5` makes `o` read 0.000232638 where a high-only A2
chain gives 0.000354708, **1.221e-04 apart, 24x its 5e-6 bar.** Verified RED by
deleting the low chain, which fails at that assertion and only there, and GREEN
with it restored.

## 4. Speed

Four processes in **vector, split, split, vector** order, idle box, one card,
4096 ids, profiled. The `gdn_scan` row is the sum of L0 GPU timestamps over 96
launches:

| run | selector | `gdn_scan` L0 GPU ms | instrumented call wall ms |
|---|---|---:|---:|
| 1 | vector | 297.5 | 2809.1 |
| 2 | dpas_split | 128.2 | 2595.7 |
| 3 | dpas_split | 127.1 | 2647.0 |
| 4 | vector | 297.7 | 2807.9 |

**Mean 297.6 ms vector against 127.65 ms split: 2.33x, 170.0 ms** (derived from
the four measured rows). No other GDN row moved more than 1.5 ms, which is the
usual profiling coupling and is not attributed to the scan.

Read the within-session pairs, not any cross-day delta: the vector control
itself moved from 304.7 to 297.6 ms over the interval between builds.

## 5. What this is, and what it is not

The diagnosis was right and narrower than expected: one operand of one term.
The fixed entry clears every registered gate with its dispatch proven inside
each run, keeps 2.33x on the scan row, costs 5,120 B more SLM, stays at 128
GRF, and does not spill.

What it is **not**: an fp32-equivalent implementation. `vn` is still rounded
once to BF16 in the `A2*vn` term, and the state path still carries two-limb
BF16. All of that is measured harmless on three prompts of one checkpoint and
on one synthetic fixture. It is not proven harmless in general, and this is the
only default in the prefill path that is not bit-identical to what it replaced.

And the rule that made this document possible: **a green gate is not evidence
unless the run says which kernel it dispatched.**
