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

---

# PROBE B - a second L0 queue on another engine (extension, 2026-09-05)

## Pre-registration - written BEFORE the probe is extended, built, enumerated or run

Nothing in this section is a measurement. Every value is a **prediction** or is
**derived** from a measurement already recorded above it.

### What the first probe did NOT test

The battery above ran the dequant on `runtime::prefill::Context`'s own
immediate command list, which `src/sycl/context.cc:68-76` creates with
`qd.ordinal = 0` and no `index` - i.e. **queue 0 of queue group 0**, whatever
that group turns out to be - and the GEMM on a `sycl::queue` built over the
same `ze_context`/`ze_device`, whose ordinal this project has never inspected.
The verdict "the two queues serialise on the device" is therefore, strictly,
"two queues *that may both be queue 0 of group 0* serialise". This extension
asks the only remaining version of the question: **does the serialisation
survive putting the dequant on a different hardware queue?**

### Step 1 - enumerate (read-only; no assumption, no timing)

`zeDeviceGetCommandQueueGroupProperties` on the masked device, printed
verbatim: for every ordinal, its `flags`
(`COMPUTE` / `COPY` / `COOPERATIVE_KERNELS` / `METRICS`), `numQueues`,
`maxMemoryFillPatternSize`. Whatever the driver reports is the record; this
probe does not assert in advance how many groups a B70 has.

### Step 2 - the binding rule, fixed in advance

The alternate list is one `zeCommandListCreateImmediate` with
`ZE_COMMAND_QUEUE_FLAG_IN_ORDER` + `ASYNCHRONOUS` (the same shape
`prefill::Context` builds), bound by this rule, applied to the enumeration
without further choice:

1. if a **second group with `ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE`**
   exists → `ordinal` = that group, `index = 0`;
2. else if the single compute group reports **`numQueues ≥ 2`** →
   `ordinal` = the compute group, **`index = 1`**;
3. else → there is exactly **one** compute queue on the device and the
   alternate binding is impossible; the probe reports that and the lever is
   dead by enumeration.

Copy-only groups are recorded but are **not** candidates: they carry no
`COMPUTE` flag, so `zeCommandListAppendLaunchKernel` has no engine there.

### Step 3 - the pre-registered prediction, both branches

The branch is selected by the enumeration, which is run and recorded **before**
the battery.

| enumeration outcome | pre-registered recovery |
|---|---|
| a second compute queue exists (rule 1 or rule 2) | **≥ 0.5 of `D`** |
| only one compute queue exists (rule 3) | **≈ 0.12 again** - the serialisation is device-wide and this probe cannot move it |

`D` is the dequant time inside the battery, `8 × t_dequant`, measured in the
same run by control A.

### Step 4 - the battery, unchanged

The exact battery above - 8 gate‖up linears, K = 5120, N = 34816, M = 2048,
serial vs double-buffered, 8 replays / drop 3 / median of 5 / one discarded
warm-up, `ZE_AFFINITY_MASK=1`, iterate grade - with **only** the dequant's
command list changed. Both batteries use the alternate list, so the serial
baseline and the overlapped cell differ in nothing but the overlap. The
per-iteration wait drains the SYCL queue **and** the alternate list
(`zeCommandListHostSynchronize`), so nothing is left in flight at a boundary.
The bitwise-identity check against the serial battery is kept: it is the only
thing separating a speed-up from a race, and a second engine makes a race
*more* plausible, not less.

The no-argument invocation still runs the original single-list path
unchanged, so the 2026-09-05 record above stays reproducible from the same
binary.

### Decision rule (plan 6c Task 4's original bar, restated)

- **recovery ≥ 0.3** → the lever is real; `runtime::prefill::Context` gains a
  second L0 immediate list on the alternate binding.
- **recovery < 0.3** → dead, and **the overlap question is closed for this
  device**: neither a second queue on the same engine nor a second engine
  recovers the dequant, and the 210.116 ms/chunk stays structural.

A clean miss is a valid result and will be reported as one.

---

## Measured - PROBE B (2026-09-05, card 1, `ZE_AFFINITY_MASK=1`; iterate grade)

Box state at every run below: **disk 83 GB free** (`/dev/nvme0n1p2 915G 786G
83G 91% /`); **DRM holders = 2** - pid 285644 `baobab` and pid 285678 `ptyxis`,
both desktop GUI processes holding `/dev/dri/card0` plus all three render
nodes. Not a provably idle box, so these rows are **iterate grade**, which is
what the probe is graded at; they are not record grade and are not offered as
such.

### Step 1 - the queue-group enumeration, verbatim

`zeDeviceGetCommandQueueGroupProperties`, Intel(R) Arc(TM) Pro B70 Graphics,
device 0 under `ZE_AFFINITY_MASK=1` (measured):

| ordinal | flags | numQueues | maxMemoryFillPatternSize |
|---:|---|---:|---:|
| 0 | COMPUTE + COPY + COOPERATIVE_KERNELS | 1 | 18446744073709551615 |
| 1 | COPY | 1 | 1 |

`compute groups: 1; first compute ordinal: 0; second compute ordinal: -1`

(The probe prints the flag names joined by `+` rather than `|`; the bits are
the driver's, the separator is this document's, because a pipe would split the
table row.)

### The binding availability check - asked of the driver, not inferred

Create-and-destroy only: one `zeCommandListCreateImmediate` per candidate,
`IN_ORDER | ASYNCHRONOUS`, destroyed immediately. **No kernel was appended and
nothing was submitted**, so this asks the queue-level question without putting
a compute kernel on an engine that may not have one (measured):

| ordinal | index | `zeCommandListCreateImmediate` |
|---:|---:|---|
| 0 | 0 | SUCCESS (0x0) |
| 0 | 1 | **refused** (0x78000004 = `ZE_RESULT_ERROR_INVALID_ARGUMENT`) |
| 1 | 0 | SUCCESS (0x0) |
| 1 | 1 | **refused** (0x78000004) |

`numQueues = 1` is therefore **enforced**, not merely advertised: index 1 is
refused on both groups. And ordinal 1 carries **no `COMPUTE` flag**, so
`zeCommandListAppendLaunchKernel` has no engine there - it is a copy engine and
was never a candidate for `pf_dequant_tile`.

### Step 2 - the binding rule resolves to rule 3

The pre-registered rule, applied without further choice:

1. a second COMPUTE group - **does not exist** (one compute group, ordinal 0);
2. `numQueues ≥ 2` on the compute group - **false** (`numQueues = 1`, and the
   driver refuses index 1);
3. → **exactly ONE compute queue exists on this device.**

`--auto` prints exactly that and stops:
`binding rule 3: exactly ONE compute queue on this device. The alternate
binding is impossible and the lever is DEAD BY ENUMERATION.`

**The pre-registered branch selected by the enumeration is therefore the
second one: "≈ 0.12 again - the serialisation is device-wide."** It was
selected before any battery was run, by the rule written down before the
enumeration.

### Step 3 - the only binding that exists, re-measured in this session

Since no alternate binding exists, the battery can only run on the one
compute queue. It was re-run through the extended binary as a **regression
control** - the claim "the no-argument path is unchanged" is verified rather
than asserted (measured, iterate grade):

| control | this session | 2026-09-05 record | Δ |
|---|---:|---:|---:|
| dequant gate‖up L0 `[K][N]` | 1.541 ms / 292.8 GB/s (r+w) | 1.539 / 293.1 | **+0.13%** |
| `gemm_bf16` gate‖up M = 2048 | 5.570 ms / 131.08 TFLOP/s | 5.554 / 131.47 | **+0.29%** |

| battery | waits | ms total | ms/linear |
|---|---:|---:|---:|
| serial, one scratch | 16 | 56.893 | 7.112 |
| double-buffered, two scratches | 8 | 55.199 | 6.900 |
| double-buffered, GEMM submitted first (diagnostic) | 8 | 55.158 | 6.895 |

- dequant inside the battery: 8 × 1.541 = **12.329 ms**
- hidden: 56.893 − 55.199 = **1.694 ms**
- **recovery = 0.137**; diagnostic (GEMM-first) 0.141
- output **bitwise identical** to the serial battery again, over the same two
  deliberately different weight copies

Three independent runs of this cell now read **0.112 / 0.127 / 0.137**. The
pre-registered branch said **≈ 0.12**; the measurement is **0.137**, inside
that band.

## Verdict - PROBE B: MISS, and by the cleanest possible route

| | value |
|---|---|
| pre-registered, branch "a second compute queue exists" | ≥ 0.5 of `D` |
| pre-registered, branch "only one compute queue exists" | ≈ 0.12 |
| **branch the enumeration selected** | **the second one** |
| acceptance bar | ≥ 0.3 |
| **measured on the only binding that exists** | **0.137** (0.46× of the bar) |

**The B70 exposes exactly one compute queue.** There is no second engine
ordinal and no second queue index to move the dequant to; the driver refuses
both. The serialisation the first probe measured is therefore not an artifact
of two queues landing on the same hardware queue by accident - **it is the
only arrangement this device has**.

**Consequences, priced:**

- `runtime::prefill::Context` does **not** gain a second L0 immediate list.
  There is nothing for it to bind to.
- **The overlap question is CLOSED for this device**, exactly as the decision
  rule said it would be at < 0.3. Neither double-buffering, nor submission
  order, nor a second command list on another ordinal or index recovers the
  dequant, because the hardware has one compute queue.
- The **210.116 ms/chunk dequant term stays structural** and the composed
  ceiling carries it whole. Nothing in the C = 2048 composition moves.
- The controller's "multi-engine overlap, a cheap third candidate, unpriced"
  is now priced: **cost one enumeration, value zero.**
