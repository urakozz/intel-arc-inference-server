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
