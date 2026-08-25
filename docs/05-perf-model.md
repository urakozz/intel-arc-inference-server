# Performance model

## The roofline

Decode at batch 1 reads every weight once per token, so:

```
tokens/sec_ceiling = read_bandwidth / bytes_read_per_token
                   = 600 GB/s / W
```

`600 GB/s` is **measured** (doc 01). `W` for the phase-1 model is **measured
from the safetensors headers**, deduplicated through the index manifest
(doc 03): **15.52 GB per token** - 12.16 GB int4 weights, 0.76 GB f16 scales,
2.54 GB bf16 `lm_head`, 0.05 GB of norms and small bf16 projections. Embedding
rows are gathered (~0), `qzeros`/`g_idx` are dropped at load, the vision tower
is skipped, the MTP head is phase 2.

```
ceiling(27B, no MTP) = 600 / 15.52 = 38.7 t/s
vLLM p314-t214-vxkp0  =               31.50 t/s  ->  489 GB/s  ->  81% MBU
                                       (600 GB/s / 15.52 GB denominators;
                                        83.0% on the measured 590 / 15.540 pair)
```

Resident bytes on load should equal this figure unless the loader pads; the
first load asserts it.

**Two constants, both measured, and which one to use.** The loader now reports
what it actually makes resident: **15,539,980,288 B = 15.540 GB per token**
(docs/13) - the 15.519 GB of doc-03 header arithmetic plus 0.016 GB of tiling
pad and 0.005 GB of fp32-widened norms. And doc 01 measures **590 GB/s** through
a plain Level Zero launch (`probe_bw`) against the 600 GB/s it recommends as the
general roofline denominator. **"Phase 1, measured" below, and the b70-decode
section of [BENCHMARKS.md](BENCHMARKS.md), use 15.540 GB and 590 GB/s
throughout**, because that pair describes the path this engine submits on, and
they apply it to vLLM's number too so the comparison is like-for-like. The rest
of this document - the tables in "Where the headroom actually is", the
specialisation list and the "what to measure" list - predates that pair and
still reads **600 GB/s / 15.52 GB**, which is why vLLM appears there as 81% MBU
and here as 83.0%. Those are the same measurement on two denominators, not two
measurements; each site that carries the older pair now says so inline. Do not
mix them inside one comparison:

```
ceiling(27B, no MTP) = 590 / 15.540 = 37.97 t/s   (26.34 ms/token)
vLLM p314-t214-vxkp0 =                31.50 t/s  ->  489 GB/s  ->  83.0% MBU
b70-decode ef6acb0   =                27.54 t/s  ->  428 GB/s  ->  72.5% MBU
b70-decode 62bdd4d   =                23.73 t/s  ->  369 GB/s  ->  62.5% MBU
                                       (62bdd4d is phase 1 before spec 1.5's
                                        lever ladder; ef6acb0 is its gate)
```

## Phase 1, measured - the honest verdict

**b70-decode does not beat vLLM. Measured 2026-08-25 at the spec 1.5 gate
(`ef6acb0`), median of three runs on an idle box: 27.54 t/s at tg256, depth
4096 - against vLLM's 31.50. That is 12.6% short, a factor of 1.144 the wrong
way.** Per token: 36.32 ms against vLLM's 31.75 and a 26.34 ms roofline; MBU
**428 GB/s of 590 = 72.5%**, against vLLM's 83.0%. Spread over the three runs
was 0.01 t/s (0.04%). Full rows, the depth experiment and the exact command are
in [BENCHMARKS.md](BENCHMARKS.md#b70-decode--this-project-phase-1).

**The phase-1 first measurement is kept, dated, because it is what spec 1.5 was
scoped from: 23.73 t/s / 42.141 ms/token, 24.7% short, 62.5% MBU - 2026-08-25,
`62bdd4d`, before any lever.** Nothing was tuned to produce *that* number and
nothing was tuned in the plan that measured it; what that plan owed was the
decomposition below, because the decomposition is what scoped spec 1.5.

**What spec 1.5 then bought, all measured, golden gate 96/96 at every step:**
`a‖b` GEMV K-split **−1.875 ms**, `prep_res_norm` two-stage **−2.220 ms**,
`ATTN_BLOCK` 256 → 64 **−1.726 ms** - 42.141 → 36.32 ms/token, +16.1%
throughput, 62.5% → 72.5% MBU. That closed **56% of the 10.395 ms/token gap**
this spec opened with, and it did **not** close the gate: **4.574 ms/token
remain**. Spec 1.5's stopping
rule fired at its own §6 gate and the spec closed short (tag `spec1.5-done`).
The per-lever ledger, the remaining gap and the priced menu of what a spec 1.6
would have to do are in
[the re-assessment memo](superpowers/specs/2026-08-25-spec1.5-reassessment.md);
its headline is that `lm_head` at int4 (~3.3 ms, **estimated**) is the largest
item left in the step and is **necessary but not sufficient** - 30.62 t/s,
0.88 t/s under the bar, so a second item is required and none has a measured
price.

### Where the 42.14 ms goes

A **partition**: every launch and the host appear once, and the rows sum to the
measured 42.141 ms exactly - 28.346 + 2.314 + 0.046 + 0.008 + 11.330 + 0.097 in
the original column, 29.005 + 5.782 + 0.017 + 6.768 + 0.473 + 0.097 = 42.142 in
the in-situ one (one µs of rounding) - which is why they carry three decimals
where the prose rounds to two.

The **in-situ** column was added 2026-08-25 by `b70-decode --profile --depth
4096 --steps 32` - Level Zero kernel timestamps on the replayed list, every one
of the 645 launches timed in the step it actually runs in. The full anatomy,
the method and its caveats are [15-step-anatomy.md](15-step-anatomy.md); the
"how it was obtained" column is kept as written because it is what the in-situ
run was checking.

| part | launches | ms/token | **in situ** | share | how it was obtained |
|---|---|---|---|---|---|
| GEMV - the int4 mixers/MLPs and bf16 `lm_head` | 257 | **28.346** | **29.005** | 68.8% | **measured per kernel**, `probe_gemv` 2026-08-24, at exactly the shapes, layout and `S` the model table binds; summed over the layer counts. **A floor, not an in-situ charge** - see below. In situ: **+2.3%**, so the floor was right |
| `attn_decode`'s per-block work at depth 4096 | (of 48) | **2.314** | **5.782** | 13.7% | **estimated**: a two-point extrapolation from the depth experiment (doc 07 #12, doc 12 `attn` → Measured) - 0.1361 ms/block × 17 blocks, with the measured 2.144 ms delta as its floor. In situ: **+145%**, the largest error the profiler found |
| the fixed attention grid's early-out | (of 48) | **0.046** | (inside the row above) | - | **measured**, the `--max-len` experiment (doc 07 #12). Re-confirmed per kernel in situ: 192 idle work-groups per layer cost **0.04%** of `attn_decode` |
| `embed_gather` + `argmax` | 3 | **0.008** | **0.017** | 0.04% | **measured per kernel** (doc 12) |
| `prep` + `gdn_step` + `attn_prep`/`attn_reduce` + the `a‖b` GEMV | 369 | **11.330** | **6.768** | 16.1% | **aggregate**, not separated: 30.7 µs per launch. In situ, separated: `prep` 3.573, `a‖b` 2.335, `gdn_step` 0.733, `attn_prep`+`attn_reduce` 0.127 |
| dispatch gap | - | (memo line) | **0.473** | 1.1% | **derived** in situ (bench fence − Σ kernel durations); it was a memo line rather than a row, so the bucket above silently carried it |
| host, outside the fence entirely | - | **0.097** | **0.097** | 0.23% | **measured**, `Engine::last_gen_ms() − last_fence_ms()` |
| **total** | **645** | **42.141** | **42.141** | 100% | **measured**, median of three |

> **This is the step at `43bb720`, before spec 1.5 cut a lever**, and it is kept
> whole because it is what the 42.141 ms bench row partitions. Since then lever
> L2 took the `a‖b` GEMV from 2.335 to 0.256 ms/token and lever L1 took
> `prep_res_norm` from **2.893 to 0.484** - that pair is L1's own before/after
> run (`b15f70f` → `b045e11`), not this table's 2.871, which is the anatomy
> run's reading of the same pre-lever row 0.8% away; a before/after is two rows
> of one comparison and the two must not be mixed inside one arrow. The `prep`
> share of the bucket row is 1.202 and the launch count is **774**, not 645; the
> attention family is 3.839, not 6.058 (lever L5, launch count unchanged). The
> recorded step is **36.32 ms** at `c746840` - BENCHMARKS.md, and
> [15-step-anatomy.md](15-step-anatomy.md) §L2, §L1 and §L5 for each
> before/after.

Two memo lines, not rows, because they overlap the rows above: **dispatch** was
645 × 0.52 µs = 0.335 ms (**estimated**, doc 07 #5) and is **0.473 ms derived in
situ** - 1.1% of the token, and now a row of its own above; and **everything
that is not a GEMV** is 42.141 − 29.005 = **13.136 ms** (31.2%) in situ
(13.795 against the floor), of which 13.039 is device time and 0.097 is host.
The second is the figure point 3 below compares against vLLM.

**The four things the in-situ column changed**, in size order: `attn_decode` is
2.45× its estimate and the second-largest item in the step; the unsplit bucket
is 6.768 ms rather than 11.330, because it had been carrying everyone else's
error; `gdn_step` is at **1.09× its own traffic floor** (540 GB/s, 92% of the
device) and is exonerated as a suspect; and the GEMV floor was accurate to 2.3%,
which resolves the "the two move in opposite directions" caveat below in favour
of the floor. Point 5's suspect ranking is superseded accordingly.

Read that table twice before proposing anything.

1. **The host-overhead thesis is finished for this engine.** 99.77% of the step
   is inside `execute` + `fence.wait()`; the host spends **97 µs per token**
   writing four bytes and submitting. There is no host work left to delete. The
   thesis was always about vLLM's Python/Triton launch path (doc 09), and
   replay did exactly what doc 08 said it would - which is precisely why the
   remaining gap is *not* excusable as overhead. It is kernel time.
2. **GEMV is not the problem - with one caveat that this section is the wrong
   place to omit.** 28.346 ms of measured GEMV is 89% of vLLM's *entire*
   31.75 ms step. Those kernels run at 89-97% of the 600 GB/s denominator the
   probe reports against (90-99% of 590), and the engine binds the probe's own
   best layout and `S` for every shape, so the "GEMV fill at the chosen `S` vs
   the probe matrix" suspect is **ruled out**: production and probe are the same
   configuration. There is ~2.0 ms of headroom between the GEMV sum and the
   26.34 ms roofline, and that is all there is.

   **The caveat (doc 12, `gemv` → Measured):** 28.346 ms is *transplanted*
   probe data - 40 independent launches replayed in their own in-order list -
   not GEMV time observed inside the decode list. It is therefore a **floor**.
   What the probe cannot see is a stall left behind by the *preceding* kernel: in
   the engine every GEMV is entered directly after a `prep` or `attn`/`gdn`
   kernel, and any drain or cache state that costs would land in GEMV's in-situ
   time, not in the probe's. Since the unsplit bucket below is defined as the
   step minus this number, **the two move in opposite directions**: if a profile
   finds GEMV above 28.346 in situ, the 11.33 ms bucket shrinks by exactly that
   much and the suspect ranking in point 5 changes with it. Spec 1.5's first
   task must measure both sides in the same run, not compare one measured in
   situ against one transplanted.

   **Answered 2026-08-25 (docs/15): 29.005 ms in situ, +2.3%.** The floor was a
   good floor. The bucket did shrink by exactly that 0.659 ms, and the stall the
   probe could not see is worth naming in one shape only - `out/o_proj`, +7.0%,
   entered directly from `attn_reduce`/`prep_gated_head`. The "~2.0 ms of
   headroom to the roofline" reading survives with 0.66 ms of it now spoken for.
3. **The 13.795 ms of non-GEMV work is the whole gap.** To reach 31.746 ms/token
   the budget for everything that is not a GEMV is **3.400 ms**; we spend
   13.795. Closing 10.4 of those milliseconds is exactly the phase-1 shortfall.
   (In situ the arithmetic shifts and the conclusion does not: GEMV is 29.005,
   so the non-GEMV budget is 31.746 − 29.005 = **2.741 ms** against 13.136 ms
   spent. The shortfall is 10.395 ms either way, because the total is the
   measured step.)
4. **The attention early-out is exonerated** (doc 07 #12, resolved): a grid
   sized for `max_len` 16384 costs **0.046 ms/token - 0.11% of the step** in
   idle work-groups. Context-bucketed lists would buy nothing.
5. **The named suspect inside the 11.33 ms is `prep_res_norm`**, on the strength
   of its own design note rather than a measurement: 129 launches per token, and
   each is a **single work-group** reducing 320 KB of split-K partials on one
   Xe-core (doc 12, "One work-group per token in `prep_res_norm` is the risk in
   this design, and it is unmeasured"). At 40 µs each that alone is 5.2 ms. The
   `in_proj_a‖b` GEMV is the second: N = 128 is 8 subgroups on a device with 32
   subslices (doc 01), 48 times per token, and it was never in the probe matrix.
   `gdn_step`'s 48 launches move 396 MB - 0.67 ms at the roofline - so if it is
   costing multiples of that, the cause is occupancy, not traffic.

   **Resolved 2026-08-25 (docs/15), and this point scored 2½ out of 3.**
   `prep_res_norm` is **2.871 ms** - the named mechanism exactly (one work-group
   at 17.0 GB/s), the magnitude 2.3× smaller than the 5.2 ms guessed. `a‖b` is
   **2.335 ms** at 27.0 GB/s, 21× its traffic - the second suspect, convicted.
   `gdn_step` is **0.733 ms**, 1.09× the 0.67 ms named right here as the test,
   so it is *not* costing multiples of its traffic and the occupancy charge is
   dropped. What none of the three anticipated is the item that outweighs all of
   them: **`attn_decode` at 5.782 ms.** The occupancy story generalises, but
   **not in the unit this paragraph first wrote it in.** It said one work-group
   is worth 12-17 GB/s (docs/15 §1, three points). Lever L2 tested that on `a‖b`
   by giving it 4× the work-groups at the same subgroup count and it bought
   **nothing**; splitting K to 16× the subgroups took the same launch from
   48.774 to 5.340 µs, i.e. that run's row from 2.341 to **0.256 ms/token**
   (docs/15 §L2; the 2.335 above is the anatomy run's reading of the same
   pre-lever row, 0.3% away - one run's pair per sentence, never a mix). The unit is the **subgroup**. `prep_res_norm`'s 2.871 ms was still a
   parallelism story and L1's case rested on that curve rather than on §1's
   table - **and the curve was right**: lever L1 split the kernel in two,
   16 → 320 subgroups, and took the row to **0.484 ms/token** (docs/15 §L1). `attn_decode` is a third failure again: its
   work-groups are nearly free (3.4× for +6.9%) and what costs is one
   work-group's serial walk of a 256-position block.

### What spec 1.5 is scoped to do

In this order, because that is the order the evidence supports:

1. **Get a per-kernel profile.** The 11.33 ms is an aggregate and no amount of
   arguing splits it. Level Zero kernel timestamps on a one-off instrumented
   capture, or a `probe_prep` / `probe_gdn` in the shape of `probe_gemv`.
   Nothing else should be attempted before this number exists.
2. **`prep_res_norm`'s single work-group**, if the profile confirms it: a
   two-stage reduction, or fold the split-K sum into the GEMV epilogue.
   129 launches is where the leverage is.
   **Done (2026-08-25, spec 1.5 lever L1):** the profile confirmed it at 22.4 µs
   and 17.0 GB/s per launch, and the two-stage reduction is what was built -
   `prep_res_fold` (20 work-groups, 320 subgroups) then `prep_norm_finish`
   (20 more). 2.893 → **0.484 ms/token**, golden gate 96/96, and the *launch*
   count deliberately went up: 645 → 774. Folding into the GEMV epilogue was
   never tried and is now worth ≤0.484 ms.
3. **`in_proj_a‖b`**: 48 launches of an 8-subgroup kernel. Fusing it into the
   `qkv‖z` GEMV (doc 04's fusion item 4) removes the launch and the fill
   problem at once - the one fusion the measurement now argues for.
   **Done, and without the fusion (2026-08-25, spec 1.5 lever L2):** the fill
   problem was the 8 subgroups, and a 16-way K split inside the work-group makes
   128 of them. 2.335 → **0.256 ms/token**, no fusion, no extra launch, golden
   gate 96/96. The fusion is now worth ≤0.256 ms and is not argued for.
4. **`gdn_step` occupancy**, if the profile puts it above ~1 ms.
5. Only then the ~2.0 ms of GEMV headroom, and the `lm_head` quantisation below,
   which is worth 4.35 → ~1.1 ms and needs no kernel work at all.

Fusion for its own sake stays rejected: 645 × 0.52 µs = 0.335 ms is 0.8% of the
step (doc 07 #5, estimated). Kernel *count* is not the problem; what those
kernels do while they run is.

**Step 1 is done, and it re-ordered steps 2-5** (2026-08-25,
[15-step-anatomy.md](15-step-anatomy.md) carries the ladder and the expected
yields). The profile ranks the levers by measured share as `attn_decode` 5.782 >
`prep_res_norm` 2.871 > `a‖b` 2.335 > `gdn_step` 0.733 > GEMV's in-situ excess
over its floor 0.659; the execution order docs/15 rules is **`a‖b` → `prep`
two-stage → attention** (`a‖b` is **cut**: −1.875 ms/token on the bench;
`prep` is **cut**: −2.220 ms/token; **attention is cut**: −1.73 ms/token, by an
`ATTN_BLOCK` 256 → 64 retile rather than the SLM staging the plan sketched, and
the step is **36.32 ms** at `c746840`, BENCHMARKS.md), with `gdn_step`
**skipped by ruling** (it is at 92% of
device bandwidth - there is 0.06 ms in the whole kernel) and the GEMV `S` retune
conditional on the gate being within reach. Fusion is re-priced too: the in-situ
dispatch gap is 0.473 ms (**derived**, doc 07 #5), so removing launches is worth
even less than the estimate said - but §1 of docs/15 shows what the fusion
candidates were really buying, which is work-groups, not launches.

One number the ladder does not touch and the gate cannot ignore: **`lm_head` is
4.376 ms in situ, 12.0% of the 36.32 ms step**, at **581 GB/s - 98.5% of the
measured 590** - and 0.6% above its probe floor. There is nothing to tune;
quantising it to int4 is worth ~3.3 ms (**estimated**) and is specialisation 1
below, deliberately outside spec 1.5.

**Closed 2026-08-25, short, at the §6 gate - 27.54 t/s / 36.32 ms/token
measured** (tag `spec1.5-done`). docs/15's gate arithmetic, written before the
first lever, put the whole ladder at ~34.4 ms optimistically; the measured
answer is 36.32 with L3 and L4 unspent and ruled not worth their golden-gate
runs (≤0.06 and ≤0.3 ms). **The claim this paragraph used to end on -
that `lm_head` is the difference between missing the bar and clearing it - is
withdrawn, and L5 is why:** it was written against a projection that assumed
attention would pay −3.0 ms, and attention paid **−1.726**. On the ladder's own
optimistic ceiling of 35.96 ms, `lm_head` at int4 lands at **32.66 ms =
30.62 t/s - still 0.88 t/s under the bar**. It is necessary and it is not
sufficient; a second item is required and none has a measured price yet. That
is the subject of
[the spec 1.5 re-assessment memo](superpowers/specs/2026-08-25-spec1.5-reassessment.md),
which prices every candidate - and puts a probe that names `attn_decode`'s
unexplained 224 µs launch ahead of every design, because four cost models have
now died on that kernel.

## Where the headroom actually is

The original claim was that the gap between the current stack and the ceiling is
**host overhead**, not kernel quality:

| Evidence | Reading |
|----------|---------|
| MXFP4 72.65 vs GPTQ-int4 73.31 t/s - two unrelated kernel paths within 1% | Not kernel-bound |
| Same model, XPU graphs off 10.55 vs on 31.5 t/s | ~3× of the cost is host-side, and graphs already recover most of it |
| Measured MBU 50-63% | The card is idle 40% of the time |

That evidence was gathered on the MoE models. Per model, with the 27B's `W`
now **measured** and the MoE `W` still an estimate (doc 07 #1):

| Model | t/s | `W` | Ceiling @ 600 GB/s | MBU |
|---|---|---|---|---|
| **Qwen3.8-27B dense int4 (phase 1)** | 31.50 | **15.52 GB, measured** | **38.7** | **81%** (83.0% at 590 / 15.540) |
| Ornith 35B-A3B MoE, MXFP4 / GPTQ | 72.65 / 73.31 | ~3B active × 4.1 bit + lm_head ≈ 4-5 GB, estimate | ~120-150 | ~50-60% |

The "50-63% MBU" figure describes the MoE models. On the dense 27B vLLM is
already at 81% - 83.0% on the 590 / 15.540 pair "Phase 1, measured" uses - so
**the entire remaining gap to the roofline - host overhead,
device under-fill, everything - is ×1.23.** The MoE number also has an obvious
mechanism (160 launches and 200 allocations per step, doc 03) that the dense
path under XPU graphs does not; vLLM's residual per-step scheduling is a few ms
against a ~32 ms step, which is exactly consistent with 81%. (Every percentage
in this section is on the 600 GB/s / 15.52 GB pair.)

So the honest projection for phase 1 is a product of small factors, not one
large one:

| Lever | Ceiling effect | Est. t/s |
|---|---|---|
| vLLM today | - | 31.50 |
| Fill the device (GEMV + split-K) and delete host work: 81% → ~95% MBU (600/15.52 basis) | ×1.17 | ~36.8 |
| Drop `qzeros` + `g_idx` | ×1.013 | ~37.3 |
| `lm_head` int8 | ×1.09 | ~40.6 |
| `lm_head` int4 (instead of int8) | ×1.14 | **~42.5** |

**~1.35× over vLLM at best without speculation (≈42 t/s against a 44 t/s
int4-`lm_head` roofline).** Phase 2 is where larger numbers live: with MTP the
verify step runs at `M = k + 1` for one weight read, and the draft step reads
0.85 GB (MTP head) + 2.54 GB (`lm_head`) - quantising both to int4 cuts that to
~0.9 GB. At an assumed acceptance of ~0.7 per draft token (**not measured**; vLLM's
45.23 at 2 drafts implies at least that) the phase-2 ceiling is ~55 t/s with
bf16 `lm_head` and ~70 t/s with everything int4, against vLLM's 45.23.

- **MoE (phases 3-4):** the host thesis is the whole game - ~×1.7-2 from replay
  alone if MBU moves from ~55% to ~90%.

**Not 5×** in any case. Anyone promising more is counting the graphs-off number
as the baseline, which is not the fair comparison.

Set expectations there and the project stays honest. The other three goals -
learning, dependency control, shipping something - are what justify the effort
beyond the multiplier.

## Specialisations a general engine will not take

Ranked by expected value.

### 1. Quantise `lm_head`

`lm_head.weight` in phase 1 is **BF16, 248320 × 5120 = 2.54 GB** (measured),
read in full on every token - **16.4% of `W`**. At int4 it is 0.66 GB; at int8
1.27 GB.

It costs one offline conversion pass (`tools/`) and **does not depend on the
host thesis being true**: ×1.14 on the ceiling at int4 (38.7 → 44.0 t/s),
×1.09 at int8 (42.1). Smaller than it would have been on a 9B, but it is the
only lever that does not need a kernel. Accuracy risk is real (`lm_head` is
sensitive) and must be checked against the unquantised baseline (doc 07 #6).
In phase 2 it pays twice per step - the MTP head reads `lm_head` for its draft
logits - and the 0.85 GB bf16 MTP head itself is the same kind of target.

### 2. Pre-swizzle weights at load

Rearranging weights once at load into the exact order the XMX mainloop wants is
free at run time. Every CUDA backend does it; vLLM's XPU MXFP4 path does not
(`compressed_tensors_moe_w4a4_mxfp4.py:190` is a bare `pass`). Size the win by
measuring the mainloop's cost with and without the ideal layout.

### 3. Delete per-step allocations

`vllm-xpu-kernels`' MoE path allocates **5 tensors per layer per step** -
200 allocations per decode step at 40 layers. Static allocation makes this zero.
Matters most from phase 3 onward.

### 4. Fuse the MoE launches (phase 3+)

Their MoE issues 4 kernel launches per layer per step - 160 per step at 40
layers. Fusing gate/up and the activation is a known ~2× reduction in launches.

### 5. Exploit 24 MB of L2

Unusually large for this class of GPU. A 4096×4096 int4 tile is 8 MB, so
multi-layer weight residency is at least arguable. Measure before designing
around it.

## What to measure, in order

1. ✅ **`W`** - 15.52 GB per token from the headers (doc 03). Confirm resident
   bytes on first load.
2. ✅ **vLLM baseline t/s** - pp4096 1973 / tg256 31.50, `p314-t214-vxkp0`, no
   speculation (BENCHMARKS.md). MTP: 42.56 / 45.23 at 1 / 2 drafts.
3. ✅ **Achieved MBU** - 81% on this model at 600 GB/s / 15.52 GB, **83.0%** on
   the measured 590 / 15.540 pair. The 50-63% figure is MoE-only.
4. ✅ **GDN vs GEMM time split** - **answered on our own engine 2026-08-25**
   (doc 07 #3, docs/15): GEMV **29.005 ms of a 42.141 ms step (68.8%)**,
   everything else 13.136 ms (31.2%). GEMM dominates as predicted. The estimate's
   *mechanism* was wrong: the non-GEMV third is not kernel count (0.473 ms of
   in-situ dispatch gap, 3.6% of it) but time inside `prep` / `gdn_step` /
   `attn` - and the per-kernel profile separates them: `attn_decode` **5.782**,
   `prep` **3.573**, `a‖b` **2.335**, `gdn_step` **0.733**, `attn_prep` +
   `attn_reduce` **0.127**. `gdn_step`, the kernel this row was written to
   worry about, is 1.7% of the step at 92% of device bandwidth. (Those shares
   are the 42.141 ms step the profile ran on. Spec 1.5's levers took the step to
   **36.32 ms** without touching a single GEMV, so GEMV's share is now
   **79.9%** - the split moved further towards GEMM, not away from it.) **Estimate before measuring** (2026-08-22, kept for the record): the
   recurrent state is 3 MB per layer, read and written once per token - ~150 MB
   across 48 layers, ~2% of `W`. Expect GEMM to dominate bandwidth and GDN to
   dominate *kernel count*; under replay the second is what the fusion list in
   doc 04 attacks. If a profile of vLLM says GDN dominates, suspect its Triton
   launch path
   (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`), not the
   arithmetic.
5. ✅ **Per-kernel fixed cost inside a replayed list** - **0.52 µs/kernel**
   (`noop`) and **0.63 µs/kernel** (`ctrl_read`) at N = 700, measured
   2026-08-22 (doc 07 #5, `tools/probe/probe_replay`). The empty submit +
   fence round trip - the floor no fusion removes - is **6.4 µs**. Both are
   under the 1 µs threshold, so the fusion list in doc 04 is *not* on the
   phase-1 critical path.
6. ✅ **GEMV bandwidth matrix** - the production `gemv.cl` at the five int4
   shapes the 27B runs after load-time fusion, both canonical layouts,
   `S` ∈ {1,2,4,8,16}, plus the bf16 `lm_head`: 51 configurations, measured
   **2026-08-24** with `tools/probe/probe_gemv` (re-run that day at the
   corrected `out/o_proj` shape 6144 × 5120 - spec §4.2 had said 5120 × 5120,
   which the model does not contain; doc 12). GB/s counts weight bytes
   (nibbles + f16 scales); each configuration is 40 launches in one replayed
   list cycling ≥ 72 MB of distinct weight copies, so the 24 MB L2 is missed
   and the number is DRAM bandwidth. **All 50 int4 configurations matched the
   CPU reference.** Best row per shape:

   | shape | K×N | layout | S | GB/s | % of 600 |
   |---|---|---|---|---|---|
   | out/o_proj | 6144×5120 | 0 | 4 | 549 | 92% |
   | q‖k‖v | 5120×14336 | 0 | 2 | 577 | 96% |
   | qkv‖z | 5120×16384 | **1** | 1 | 559 | 93% |
   | gate‖up | 5120×34816 | 0 | 8 | 555 | 93% |
   | down | 17408×5120 | 0 | 4 | 567 | 94% |
   | `lm_head` bf16 | 5120×248320 | tiled | - | **584** | **97%** |

   The probe's decision rule (spec §4.2) picks **layout 1** - by 1.7% summed
   over the five shapes, 3.6% in wall time - and `S` = 16 / 1 / 1 / 4 / 16 for
   the shapes in the order above. Doc 12 carries the full reasoning, including
   what layout 1 costs on the four shapes where layout 0 was ahead.

   Two conclusions. **The split-K argument in doc 08 is confirmed:** at
   N = 5120 - 320 subgroups, the worst fill case - the kernel reaches
   259 GB/s (43%) at `S` = 1 and 533 GB/s (89%) at `S` = 16, a 2.06× gain, and
   the same doubling appears on `down`. **`lm_head` needs no split-K:** 15520
   subgroups already reach 97% of the 600 GB/s denominator, which is 99% of the
   590 GB/s `probe_bw` measures through the same launch path. At that rate
   `lm_head` alone is 4.35 ms of a 25.8 ms token.

   ```bash
   tools/box.sh run "./build/tools/probe/probe_gemv" | tee docs/probe-gemv-2026-08-24.md
   ```

   The full 51-row matrix is committed verbatim as
   [probe-gemv-2026-08-24.md](probe-gemv-2026-08-24.md) (the superseded
   2026-08-23 run is kept alongside it). All of it is **M = 1**;
   the `S` picks are M = 1 picks (doc 12).

All six are done. 4 was half-done on 2026-08-25 (GEMM 67.3% / rest 32.7%, the
"rest" not split) and was finished the same day by spec 1.5's first task: the
per-kernel in-situ profile, [15-step-anatomy.md](15-step-anatomy.md). The
aggregate that was "the entire gap to vLLM" is now seven measured rows, and the
gap's largest addressable members were `attn_decode` (5.782 ms), `prep_res_norm`
(2.871) and the `a‖b` GEMV (2.335). **All three have since been cut** -
lever L2 took `a‖b` to 0.256 ms/token, lever L1 took `prep_res_norm` to 0.484,
and lever L5 took `attn_decode` to 3.585 - and the step is **36.32 ms** at
`c746840`. The three figures above are the pre-lever anatomy they were measured
in. **The list of addressable members is now empty at this spec's scope**, and
`attn_decode` is still its largest entry at 3.585 ms: what is left there is a
per-launch term of ~224 µs that four cost models have failed to explain
(docs/15 §L5), not a design anybody has costed. The next items are outside spec
1.5 - `lm_head` at int4 (~3.3 ms, docs/05 specialisation 1) above all.

## Benchmark

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

Reference numbers from the vLLM stack (image `p314-t214-vxkp0`, torch 2.14,
graphs on unless noted):

| Model | Config | pp4096 | tg256 |
|-------|--------|--------|-------|
| Qwen3.8-27B dense int4 | - | 1973 | 31.50 |
| Qwen3.8-27B dense int4 | MTP, 1 draft | 1931 | 42.56 |
| Qwen3.8-27B dense int4 | MTP, 2 draft | 1918 | 45.23 |
| Qwen3.8-27B MixedInt4 | **graphs off** | 2275 | **10.55** |
| Ornith MXFP4 MoE | - | ~9250 | 72.65 |
| Ornith GPTQ-int4 MoE | - | ~9232 | 73.31 |

Phase-1 baseline is the first row: **pp4096 1973 / tg256 31.50.** Full tables,
images and the exact serve and bench commands are in
[BENCHMARKS.md](BENCHMARKS.md); the dense 27B rows there use `--max-model-len
16k`, `pp4096`, `depth 1`, `concurrency 1`, which is the v1 definition of done.

**b70-decode's own row, measured 2026-08-25 at the spec 1.5 gate: tg256
27.54 t/s at depth 4096** - `tools/bench_decode.sh`, median of three, and 12.6%
short of that baseline (it was 23.73 t/s / 24.7% short at `62bdd4d`, before the
lever ladder). No `pp4096` figure exists yet: this engine has no prefill kernel
(spec 2), so a prompt costs one decode replay per id - 4096 ids in 141.0 s,
median **34.44 ms/token**, measured in the gate's own three runs.
