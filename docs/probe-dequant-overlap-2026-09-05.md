# Probe - does the dequant hide behind the GEMM? (spec 2, L2 lever)

Plan 6c Task 4, promoted to a standalone probe by the controller's
probe-first ruling. Written BEFORE the probe is built or run. Every number in
this section is a **prediction** or a **derived** value from the named
measurements; nothing here is a measurement.

## What is being decided

`PrefillScratch::dequant` is one 356,515,840 B bf16 buffer today, so the
dequant of linear `i+1` cannot start until the GEMM of linear `i` has finished
reading it. A **second** slot removes that dependency: dequant `i` writes slot
`i&1` on the L0 immediate list while the GEMM of linear `i−1` reads slot
`(i−1)&1` on the SYCL queue. The question is whether the driver actually runs
the two concurrently on one compute engine, and how much of the dequant time
that hides.

## The two reference points, both already measured

| quantity | value | grade | source |
|---|---:|---|---|
| dequant, whole chunk | **210.116 ms/chunk** | derived from measured | `docs/probe-dequant-2026-09-04.md` |
| dequant, gate‖up L0 `[K][N]` (the probe's shape) | **1.5393 ms/launch** | measured, iterate | same |
| GEMM, gate‖up at M = 2048 | **150.19 TFLOP/s** | measured, iterate | `docs/probe-prefill-gemm-2026-09-04.md` (P2, 256-GRF) |
| GEMM, gate‖up at M = 2048, time | **4.862 ms** | derived: 730.145 GFLOP ÷ 150.19 TFLOP/s | - |
| GEMM, whole chunk | 680.1 ms/chunk | derived from measured | T6 composition |

Every per-matrix dequant is far shorter than the GEMM that follows it - at
gate‖up, **1.539 ms of dequant against 4.862 ms of GEMM (derived)** - so a
*perfect* overlap hides **all** of the dequant time.

## The pre-registered prediction

**The lever recovers 0.6 - 0.9 of the dequant time `D`.** At the measured
`D = 210.116 ms/chunk` that is a saving of **126 - 189 ms/chunk**, i.e.
**11 - 17% of the composed chunk** at the ~1130 ms operating point.

**Acceptance: ≥ 0.3 · D recovered.** Below that the lever is not built, the
356,515,840 B second slot is not spent, and the negative is recorded with its
price.

## The honest caveat, written down first

Overlap requires the driver to schedule the L0 immediate command list and the
SYCL in-order queue **concurrently** on the same compute engine. Two queues on
one engine may simply serialise. **This probe measures whether they do; it is
not an assumption.** A measured recovery of **0.0 is a finding, not a
failure** - it would mean the 210.116 ms dequant term is *structural* on this
single-context in-order execution model, and the composed ceiling must carry
it in full.

## What the probe does

`tools/probe/probe_dequant_overlap` - a plain g++ / Level Zero probe that
links `libb70_prefill.so` for `gemm_bf16`:

1. Allocates **two** 356,515,840 B bf16 scratches and one random int4 gate‖up
   weight (K = 5120, N = 34816, layout 0 - the layout the model selects for
   this matrix).
2. **Control A:** the dequant alone, through `runtime::prefill::Context::launch`
   on the L0 immediate list, ms/launch. Comparable to P3's 1.5393 ms.
3. **Control B:** `gemm_bf16` alone at M = 2048, K = 5120, N = 34816 on the
   SYCL queue, ms/launch. Comparable to P2's 150.19 TFLOP/s.
4. **Serial battery:** `N = 8` linears, one scratch, `Context::wait()` between
   every dequant and its GEMM - today's ordering.
5. **Overlapped battery:** the same 8 linears, two slots, plan 6c Task 4
   Step 4's sequence exactly - append dequant `i` into slot `i&1`, submit GEMM
   `i−1` reading slot `(i−1)&1`, then ONE `Context::wait()` per iteration that
   drains both queues.
6. Reports **`recovery = (T_serial − T_overlap) / (N · t_dequant)`**, the
   fraction of the dequant time hidden.

8 replays, first 3 discarded, median of the last 5, one discarded warm-up;
`ZE_AFFINITY_MASK=1`; **iterate grade**.

Correctness is not re-litigated here - `dequant_test` and P3 hold
`pf_dequant_tile` bit-exact - but the probe still checks that the overlapped
battery's final GEMM output is **bitwise identical** to the serial battery's,
because a lever that reorders queues and changes a bit is a race, not a
speed-up.

---

## Measured (2026-09-05, card 1, `ZE_AFFINITY_MASK=1`; iterate grade)

`tools/probe/probe_dequant_overlap` on Intel(R) Arc(TM) Pro B70 Graphics.
8 replays, first 3 discarded, median of the last 5, one discarded warm-up.

### Controls

| control | measured ms/launch | rate | reference | Δ |
|---|---:|---:|---|---:|
| dequant gate‖up L0 `[K][N]` | **1.539** | 293.1 GB/s (r+w) | P3's 1.5393 ms / 293.1 GB/s | **+0.0%** |
| `gemm_bf16` gate‖up M = 2048 | **5.554** | 131.47 TFLOP/s | P2's 150.19 TFLOP/s | **−12.5%** |

The dequant control reproduces P3 **to the digit** through a different host
path (`runtime::prefill::Context::launch` on the async immediate list, versus
P3's recorded `l0::CmdList` + `Queue::execute`). The batteries are comparable.

**The GEMM control is 12.5% below P2's cell for the same shape and M, and
that is a deviation, reported rather than absorbed.** It is not the per-call
`initialize`: `gemm_batched_test`'s single-head controls reproduce P2 at
+0.09% (QKᵀ) and +5.5% (PV) through exactly the same per-call path at
0.078-0.100 ms/launch, where a host-side tax would be catastrophic and is not
visible. Candidates, named and **not chased** (no tuning): this probe holds
~1.31 GB resident (two 356.5 MB scratches, a 285 MB fp32 output, two 94.7 MB
int4 sources) and runs control B immediately after a 174,080-work-group
dequant, so the memory system is in a different state than in P2's isolated
cell. It is an L2-stage item. **It does not weaken the verdict below: a
*slower* GEMM gives the dequant *more* room to hide, so 0.127 is if anything
generous.**

### The batteries

| battery | waits | ms total (8 linears) | ms/linear |
|---|---:|---:|---:|
| serial, one scratch (today's ordering) | 16 | 56.635 | 7.079 |
| **double-buffered, two scratches** | 8 | **55.075** | **6.884** |
| double-buffered, GEMM submitted first (diagnostic) | 8 | 55.137 | 6.892 |

- dequant time inside a battery: 8 × 1.539 = **12.315 ms**
- hidden: 56.635 − 55.075 = **1.560 ms**
- **recovery = 1.560 / 12.315 = 0.127**
- minus the 8 removed queue handoffs (8 × 8.569 µs = 0.069 ms, derived from
  P1's measured figure): **0.121**
- a replicate run of battery 2 earlier the same session gave **0.112**, so the
  measured recovery is **0.11 - 0.13**.

Serial matches the sum model to 0.2%: 8 × (1.539 + 5.554) = 56.746 ms
measured 56.635. Perfect overlap would be 8 × max(1.539, 5.554) = **44.431 ms
(derived)**; the double-buffered battery measured **55.075**.

The overlapped battery's fp32 output is **bitwise identical** to the serial
battery's, over two weight copies with deliberately *different* contents - so
the sequence is correct and the number is not a race.

## Verdict: MISS. The lever is rejected, and the dequant term is structural.

| | value |
|---|---|
| pre-registered | recovers **0.6 - 0.9** of D |
| acceptance bar | **≥ 0.3** |
| **measured** | **0.11 - 0.13** |
| ratio to the bar | **0.42×** - it misses the acceptance bar, not just the prediction |

**The two queues serialise on the device, and the submission order does not
change it.** The diagnostic battery - the same dependencies and the same two
slots with only the SYCL submission moved ahead of the L0 append - measures
**0.122**, indistinguishable from the pre-registered order's 0.127. Whichever
kernel the driver sees first, the other one waits. The ~0.12 that *is*
recovered is the tail: the GEMM's work-groups filling in as the dequant's
174,080 retire.

**Consequences, priced:**

- `PrefillScratch::dequant` stays **one** slot. The second **356,515,840 B**
  is **not spent**.
- The best case the lever could buy on the whole chunk is
  `0.127 × 210.116 = 26.7 ms/chunk (derived)` - about **2.4%** of a ~1130 ms
  composed chunk, ~43 t/s - for 356.5 MB on a part already resident at
  16.94-18.81 GB. Against a pre-registered 126-189 ms. It is not worth the
  memory at that price and the acceptance bar already says so.
- **The 210.116 ms/chunk dequant term is STRUCTURAL on this execution model**
  and the composed ceiling must carry it whole. The row
  "+ dequant hidden (plan 6c Task 4, unlanded)" in the T6 ledger is
  **withdrawn**, not deferred.
- P3's own corollary is now doubly confirmed: the dequant is irreducible by
  tuning *and* by overlap, so **the only route past it is eliminating the
  materialisation** - a register-only int4 unpack feeding DPAS, which spec §10
  keeps out of scope. That is the honest place to file the 210 ms, and it is
  now the largest single lever left on the prefill path after the GEMM itself.
- **What this does NOT say:** it does not say the L0 list and the SYCL queue
  cannot both be *used* - the composed attention path alternates them per
  kv-group and depends only on ordering, not on concurrency. It says only that
  submitting to both at once buys no wall-clock.
