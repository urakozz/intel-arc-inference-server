# The split-BF16 DPAS delta net scan: design record (2026-09-20)

The delta net scan is the second largest block in a prefill chunk after the
linears. A first attempt to run it on DPAS in single BF16 was fast and failed a
real golden token gate. This record is the design of the second attempt: keep
the DPAS rate, buy the precision back with two BF16 limbs instead of one.

**Status when written:** an opt-in build behind a selector, default unchanged.
It became the default on 2026-09-23 after its arithmetic was corrected and its
dispatch proven. The correction is in
[prefill-gdn-scan-split-fix-2026-09-23.md](prefill-gdn-scan-split-fix-2026-09-23.md);
the resulting throughput row is in [BENCHMARKS.md](BENCHMARKS.md).

**Read the gate results in this file as void.** They were produced by runs that
were executing the vector kernel, not this one. That is the whole reason gate
runs now print the entry they launched. The arithmetic design below, the
selector design, and the throughput comparison stand.

## The arithmetic

The entry keeps fp32 register state and an fp32 global master state. At each
64-position sub-chunk it forms, with round-to-nearest-even BF16 conversion:

```
S_hi = rne(S)
S_lo = rne(S - bf16(S_hi))
```

Both `W.S` and `Q.S` run independent fp32-accumulating DPAS chains over the two
terms and combine the two chain results **once**, at the end. `vn` stays fp32
after that combined subtraction.

The state update is split the same way: `D = vn * exp(gl - gc)` becomes hi and
lo limbs; the high chain starts from the fp32-decayed master state, the low
chain starts at zero, and the two combine once into that fp32 state.

Each product is exact: a BF16 by BF16 multiply accumulating into fp32 loses
nothing, so the only rounding is the conversion into limbs, and two limbs carry
about 16 bits of mantissa instead of 8.

This is approximate split-BF16 arithmetic. It is **not** an fp32 equivalence
claim, and the entry is gated on tokens and state cosines rather than on
`memcmp`.

### The one term left at a single limb, and why that was a mistake

`A2*vn`, the intra-chunk term, deliberately kept D2's single-BF16 operands as a
controlled variable:

```
o[i][x] = ( SUM_k q[i][k]*S[k][x] ) * exp(gc[i]) + SUM_{j<=i} A2[i][j]*vn[j][x]
```

The reasoning was that the state path was where the precision had been lost.
It was not. That term is a direct additive contributor to `o`, which every
later layer reads, and it was the entire failure: with `A2` in fp32 the gate is
93/93, with `A2` in single BF16 it is 92/93. The fix doc has the 2x2 that
separates `A2`'s rounding from `vn`'s.

Expected resource price of the design as written here: about 35,328 B of SLM
and 50.5 DPAS per work-item per sub-chunk. Splitting `A2` as well took that to
40,448 B and 53.0 DPAS.

## The selector

`B70_PREFILL_GDN_SCAN` is read as a **process-lifetime** selector, before any
GDN launch:

| value | binds |
|---|---|
| unset or `vector` | `pf_gdn_scan`, the vector entry |
| `dpas_split` | `pf_gdn_scan_dpas_split` |
| anything else nonempty | throws, before any launch or buffer mutation |

Changing the environment inside a live engine process is unsupported, because
replay capture may retain its first kernel selection. Comparing selectors means
comparing separate processes.

The selector is private to the scan dispatcher. There is no public
resolver-string API to introspect, which means the tests could not take the
cheap route.

### Testing a selector without a test-only API

The dispatch tests exercise the **real** boundary instead of the resolver:

- An invalid value calls `gdn_chunk` with otherwise valid arguments and
  requires a `runtime_error`, **zero** launches on the context, and bytewise
  unchanged caller-owned state and ring sentinels. A selector that throws after
  mutating a buffer would fail this.
- For unset, `vector` and `dpas_split`, a nonzero-state fixture compares the
  state and output that `gdn_chunk` actually produced against the independently
  launched selected entry on the same saved pre-state, and requires it to
  **differ** from the opposite entry.

Both checks were proved RED before being trusted. Hard-wiring `dpas_split` to
the vector entry fails the split boundary test at the exact assertion that
compares against the direct split launch. That mutation was reverted before
verification.

**This is where the dispatch lesson was half-learned.** These tests catch a
mis-wired selector. They cannot catch a *build* whose split entry is stale or a
run whose launch went elsewhere, because two correct entries produce different
bytes here only by design, not by construction. What was missing, and was added
after this record, is the run printing the entry it actually launched.

## Throughput, measured ABBA

Four independent Level Zero processes in `vector, split, split, vector` order,
idle box, one card, 4096 ids in two 2048-token chunks, first-use run excluded
from every median. Each process asserted the position count, the chunk count
and the launch count, and every raw run returned first token **383**.

| process | warm ms (3 runs) | median | t/s |
|---|---|---:|---:|
| vector A | 2732.018, 2732.128, 2733.433 | 2732.128 | 1499.198 |
| split A | 2577.281, 2579.346, 2579.590 | 2579.346 | 1587.999 |
| split B | 2582.022, 2577.515, 2582.791 | 2582.022 | 1586.354 |
| vector B | 2746.171, 2741.451, 2743.357 | 2743.357 | 1493.061 |

Selector aggregate over all six warm samples each: vector **2737.442 ms /
1496.287 t/s**, split **2579.468 ms / 1587.924 t/s**. The split entry is
**157.974 ms and 91.637 t/s** ahead of vector, and at the time still 35.425 ms
behind the matched vLLM row of 2544.043 ms / 1610.04 t/s.

This table was measured on the pre-fix kernel, where the two selectors did
produce different wall times, so it stands as a selector comparison. The
post-fix profile row comparison is in the fix document, and it is tighter:
2.33x on the `gdn_scan` row itself.

A separate single instrumented process, profiling only and never a throughput
number, attributed the change: the `gdn_scan` row fell from 304.7 ms to
127.2 ms over 96 launches, a 177.5 ms reduction, while the other five GDN core
rows moved by +1.7 ms in total. Everything outside the GDN rows rose 18.0 ms,
chiefly the linear path at +15.7 ms, which is ordinary profiling coupling. A
single instrumented run establishes the scan-row change and nothing about the
whole pipeline.

## What the design does not claim

- Not fp32 equivalence. Two BF16 limbs are an approximation with a measured
  error, not an exact rewrite.
- Not general. The cosines that pass are measured on three prompts of one
  checkpoint.
- Not a promotion. Promoting an opt-in entry to the default means re-running
  every gate and proving the dispatch inside each run.
