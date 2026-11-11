# The step, launch by launch

What one decode step and one prefill step do, and, for one card's decode step, where the microseconds go,
measured **inside the replayed command list**, kernel by kernel. [12-kernels.md](12-kernels.md) says what each kernel
does; this document says what a step is made of in each run mode and where the time went, and it is the reason the
kernel work that followed went where it did.

Part 1 is the step as the code runs it today (one card, several requests at once, replica, tensor parallel, pipeline
parallel). Part 2 is the one-card decode profile of 2026-08-25, which is the record of the method and of the lessons;
its absolute numbers are **from that day's stack** (a 42.141 ms step, taken to 36.32 ms by the three changes in it),
not today's. Current speeds are in [BENCHMARKS.md](BENCHMARKS.md).

## Part 1 - what a step is today

### Decode on one card, one request

The step is **captured once and replayed per token**: one in-order Level Zero command list with every argument baked
in, 774 kernel launches for Qwen3.8, replayed at about 0.7 us per launch (part 2 measures it). The three values that
change per token live in a device-resident control block (position, current token, active row count) that the kernels
read and the sampler writes, so the host's work per token is the fence wait and reading the answer
([04-architecture.md](04-architecture.md)).

With the MTP head (Qwen3.8's built-in draft head; the default, `--spec auto`) a step is an iteration of three things:

- **draft(k):** the head runs k times on one row each (k at most 3), each time proposing the next id;
- **verify(k):** one replay of the main model over the pending id plus the k drafts, **M = k + 1 rows** (at most 4),
  with the small-M kernels of [spec 28](superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md); the
  rows are checked in order and the first wrong draft ends the run;
- **commit(j, t):** row j's recurrent (GDN) state becomes the live one and the position moves on by the ids kept.

Prompt lookup (`--spec lookup`) proposes its drafts from the text instead of the head and verifies through the same
verify lists. K per iteration follows the request's recent acceptance (`--spec-max` caps it, 3 for MTP and lookup).

### Several requests at once on one card (sequence slots)

`--max-num-seqs N` above 1 gives the engine N **slots**. A slot is one request's private state: a range of positions
in the one KV allocation, its own GDN states, conv ring, control block and its own copy of every captured list.

- **One slot decoding:** that slot's own lists run, exactly the step above, so a lone request is bitwise what a
  one-slot engine computes.
- **Several slots decoding:** one replay of a **row list** at M = 2..8 (the engine captures one per M) over all their
  rows. A 256-byte row table, written by the host before the replay, says for each row its slot, position, KV base and
  which GDN state it reads and writes; rows of one slot are consecutive, a slot appears once, and rows never read each
  other's state. With MTP each slot first runs its own draft lists (one row each), then the batched verify checks
  everyone's drafts at once.
- **The row budget:** a batched step carries at most **8 rows**, and MTP drafts share them with the pending ids. Every
  decoding slot's pending id takes one row; the rest go to drafts, dealt one at a time round the slots, the
  longest-running request first, each up to its own K. (Measured at four requests: one draft each beats three drafts
  for the oldest request alone, [probe](probes/probe-batching-2026-10-23.md) section 8.) With more than one slot on a card, a
  DFlash drafter or draft vocabulary is refused at startup.
- **A batched row is not bitwise its lone run:** the M-row kernels round the last bf16 digit differently from the
  M = 1 GEMV, so a batched row is judged against its lone run by the f32 judge (a float32 CPU reference, doc 14).
  Slot isolation is bitwise: a rejected row writes only its own slot's speculative state.
- **The tick:** the scheduler runs at most one prompt chunk, then one decode step over every decoding slot. A chunk
  stalls the other slots while it runs, so the chunk is the engine's own prefill chunk, cut where a single-request
  prefill would cut it.

### Replica: one engine per card

`--parallel replica` runs one copy of the model per card, each with its own worker thread, KV and slots. A request is
sent to the slot that holds its prefix (prefix affinity), else to the card with the most free slots. `--max-num-seqs
N` is split between the cards (card 0 takes the odd one), up to 4 per card.

- **Each card runs the step above for its own slots**, with its own 8-row budget: a card's drafts are dealt against
  that card's rows only.
- **One prompt chunk in flight per card.** While a card prefills a chunk, its decoding rows sit out the step and nothing
  else touches that card; the other card keeps stepping, and both cards may prefill at once. A card with rows decoding
  starts its next chunk after its step (chunk, step, chunk, ...).
- **Cards fail apart:** a card that throws fails only its own rows. The prefix cache is the one shared piece: a
  snapshot saved by card 0 gives a prefix cache hit on card 1.

### Tensor parallel (`--parallel tp`): one request, both cards

Every layer is split over both cards: half the GDN and attention heads, half the MLP, each card's half of the
row-parallel linears (out_proj, o_proj, down) and half the vocabulary of the output head.

- **One token is one replay of each card's list**, submitted back to back. After each row-parallel linear a card
  writes its partial result straight into the other card's landing buffer (peer-to-peer), and the next residual
  fold waits for the other half there, so the lists order themselves with no host round trip. Both cards then hold
  the same residual, bitwise.
- **The output head:** each card's argmax names the best id of its half and that logit; after both fences the host takes
  the larger, ties to the lower id, and writes it into both control blocks. That is the one host action per token beside
  the two fence waits.
- **MTP:** the head sits whole on each card and runs the same input on both, so it needs no crossing. Draft lists
  take one fence each; the verify list runs the tensor-parallel walk at M = k + 1 rows with the per-row head merge.
- **Every wait is bounded:** a half that does not arrive in time fails the step with a message naming what is missing,
  and every later step throws until the engine is reset.
- **No slots:** tensor parallel serves one request at a time; `--max-num-seqs` above 1 is replica. Tensor parallel
  with slots was tried and ruled out (replica is faster at 2-4 requests with MTP;
  [low-priority-work.md](low-priority-work.md)).
- **Prefill** cuts the prompt into chunks and takes them **in pairs**: each card walks both chunks layer by layer,
  alternating, so while chunk A's half crosses to the other card on the copy engine, the compute engine runs chunk B's
  unit. Each unit then folds both halves.

### Pipeline parallel (`--parallel pp`): layers split between the cards

Card 0 holds the embedding and the first layers, card 1 the rest, the output head and the sampler. One token is one
replay of each list, submitted back to back; the hand-off between them is a copy or a peer write, and after the fence
the host mirrors card 1's control block into card 0's. Speed is about one card's; the gain is memory (models too big
for one card). Prefill is a chunk pipeline: card 0 runs a chunk's first layers and hands its rows to card 1 while
starting the next chunk.

## Part 2 - the one-card decode profile of 2026-08-25

### Read this before reading a number below

**Profile mode is not bench mode.** Every launch in a profiled list signals a host-visible kernel-timestamp event,
which carries a flush an unprofiled list never pays.

- **A per-kernel us here is directly comparable to a probe's.** The duration is `kernelStart` to `kernelEnd`; the
  flush lands after `kernelEnd`, so nothing about a kernel's own execution is inflated by being observed.
- **The inter-kernel gap is an upper bound:** each gap contains one flush.
- **The fence wall is not ms/token.** The step time is the `--bench` median in [BENCHMARKS.md](BENCHMARKS.md).
- **A mean is not an attribution.** A single run carries the instrument's drift; "The attribution floor" below
  prices it. Read it before believing any delta under about 0.3 ms.

Every number says which kind it is: **measured**, **derived** (arithmetic over two measurements) or **estimated**
(a model or an extrapolation).

### How it was measured

```bash
tools/box.sh run "./build/src/cli/b70-decode <checkpoint> --profile --depth 4096 --steps 32"
```

Idle box, 2026-08-25, Intel Arc Pro B70, `--max-len 16384`. `--profile` ingests 4096 synthetic ids on an
un-instrumented list (an ordinary decode), then builds a second, profiled list against the same buffers and replays 32
instrumented steps, resetting every event before each replay and reading every duration only after the fence.
Positions 4096 to 4127, so 17 live attention blocks at the then `ATTN_BLOCK` of 256.

**645 is this run's launch count, not today's.** The residual-norm split below made each of the 129 `prep_res_norm`
sites two launches, so the walk is now **774**; a run today prints 774 wherever 645 appears in the first tables.

**Reproducibility (measured):** two runs of the identical command three minutes apart agree on the sum of the kernel
durations to **0.006%** (41571.315 against 41568.965 us), on the fence wall to 0.031%; per-family spreads run from
0.02% (`attn_decode`) to 2.5% (`attn_prep`, the smallest family). Run A is quoted below; nothing that turns on a
difference under about 1% is claimed.

### The anatomy - in situ against the floors

Per kernel family, one decode step at depth 4096. "Probe floor" is the per-kernel measurement transplanted from
`probe_gemv` ([12-kernels.md](12-kernels.md)).

| family | launches | **in-situ ms** | probe-floor ms | delta |
|---|---|---|---|---|
| `gemv` (int4, five shapes) | 256 | **24.629** | 23.995 | +2.6% |
| `gemv_bf16`, `lm_head` | 1 | **4.376** | 4.351 | +0.6% |
| `attn_decode` | 16 | **5.782** | (2.360 estimated) | **+145%** |
| `prep_res_norm` | 129 | **2.871** | | |
| `gemv_bf16`, `a‖b` | 48 | **2.335** | (never probed) | |
| `gdn_step` | 48 | **0.733** | | |
| `prep_silu_mul` | 64 | **0.623** | | |
| `prep_gated_head`, `attn_reduce`, `attn_prep`, `embed_gather`, `argmax` | 131 | **0.223** | | |
| **sum of kernel durations** | **645** | **41.571** | | |

**The probe transplant was right.** The 257 GEMV launches cost 29.005 ms in situ against the probes' 28.346 ms floor,
**2.3% above it**; every shape is within 7% of its probe number. The one shape worth naming is `out_proj` / `o_proj`
(+7.0%), the one entered directly from `attn_reduce` or `prep_gated_head`, the two kernels with the least data in flight.

### The dispatch gap

| | us/step | us/launch | kind |
|---|---|---|---|
| profiled gap (fence wall - sum of durations) | **851.4** | **1.320** | measured, **upper bound** |
| un-instrumented gap | **473** | **0.733** | derived |

The derived row is the honest one: the bench step is 42.141 ms of which 0.097 ms is host outside the fence, so the
fence is 42.044 ms; the sum of kernel durations is 41.571 ms; the difference is 0.473 ms. **Kernel count is not a
lever:** 0.473 ms is 1.1% of the step, and fusing kernels to reduce their number is not worth it at 0.7 us each. The
instrument's own cost is 0.379 ms per step (derived), all of it in the gap, never in a kernel duration.

The step, re-partitioned (each launch and the host once, summing to the bench step): GEMV 68.8%, `attn_decode` 13.7%,
`prep` 8.5%, the `a‖b` GEMV 5.5%, `gdn_step` 1.7%, the other attention kernels 0.3%, embed and argmax 0.04%, dispatch
1.1%, host 0.2%. **98.6% of the step is kernel time.**

### What the measurement says that nothing else did

**1. The unit of parallelism is the subgroup, not the work-group.** Dividing bandwidth by work-groups suggested that one
Xe core cannot exceed about 15 GB/s and that `a‖b` would get four times the bandwidth from four times the work-groups.
That experiment was run and bought nothing (48.774 to 49.127 us at an unchanged subgroup count). Divided by subgroups
the kernels agree: a latency-bound kernel with one load in flight per thread gets rate = bytes per load / about
60-75 ns. The invariant held at 8, 16, 32, 128 and 320 subgroups across two kernels, sub-linearly at the top (16x the
subgroups bought 9.1x on `a‖b`).

**2. `attn_decode` was 2.45x its estimate, and four cost models died proving why.** 5.782 ms in situ against 2.360 ms
extrapolated from a two-point line. Four models, written down with predictions before the run that tested them, all
missed by 14% to 55%. The first model's "depth-independent term" was an artefact of fitting through a point where
the one live block held 65 of 256 positions. **Where a quantity has resisted four fits, stop fitting and remove
terms** (the battery below). The fixed grid is free: 192 idle work-groups per layer cost 0.04%, so
context-bucketed command lists are not needed.

**3. `gdn_step` is exonerated.** 0.733 ms against a 0.671 ms traffic floor, 540 GB/s effective, 92% of the device,
1.09x its own bound; making it perfect is worth 0.06 ms.

**4. `lm_head` is 10.4% of the step and nothing tunes it.** 4.376 ms in one launch, 581 GB/s in situ, 98.5% of the
measured 590 GB/s. The bytes are the lever (an int4 head), not the kernel; the published checkpoint ships a bf16 head,
so this project quotes only byte-matched comparisons with vLLM ([BENCHMARKS.md](BENCHMARKS.md) says why).

### The three changes of 2026-08-25

All three were run against this document's own ranking with its own instrument on the same day, in this order, because
each corrected the model the next was designed on.

| change | what shipped | in-situ site | bench step | **bench delta** |
|---|---|---|---|---|
| `a‖b` GEMV K split (`gemv_bf16.cl`) | tiling `{64,1}` to `{16,16}`: 8 to 128 subgroups | 2.341 to **0.256** ms | 42.141 to 40.266 | **-1.875** |
| residual-norm split (`prep.cl`) | one single-work-group kernel to `prep_res_fold` + `prep_norm_finish`, 20 work-groups each: 16 to 320 subgroups | 2.893 to **0.484** ms | 40.266 to 38.046 | **-2.220** |
| attention block retile | `ATTN_BLOCK` 256 to **64**: 68 to 260 live work-groups | attn family 6.058 to **3.839** ms | 38.046 to 36.32 | **-1.726** |
| **all three** | | | **42.141 to 36.32** | **-5.821** |

The golden gate was element-exact on every determined row before and after each one.

- **The K split:** `COLS_PER_WG` 16 with no K split (more work-groups, same eight subgroups) and loads-in-flight
  variants were worth exactly zero (the loads variant also cost `lm_head` 1.3% by perturbing its code); only splitting K
  makes more subgroups. 5.34 us shipped against a 2.22 us bandwidth floor, so 0.15 ms/token is all that is left.
- **The norm split:** RMSNorm's mean is over the whole row, so no tiling inside one launch could help; two launches
  did. The fold is 60% of the win; the rescale pass, which looked like a formality, is the other 40%, because one
  work-group walks it at 4.5 GB/s. The launch count rose 645 to 774 (0.095 ms derived) to save 2.409 ms, a 25:1 trade
  and the measured answer to the worry that kernel count matters.
- **The retile:** `ATTN_BLOCK` 32 was 0.070 ms/token better and rejected on a 101.4 MB `attn_part`.
- **The three deltas never agreed exactly** (the in-situ and bench deltas differ by 0.892 ms in total): untouched launches
  drifted +0.30% / +0.50% / +0.56% between runs, and after the retile 0.31 ms between the in-situ prediction and the
  bench stayed **unattributed**, recorded as such rather than rounded away.
- **Yield and mechanism:** three for three on yield, zero for three on mechanism; every change landed for a reason the
  ranking got wrong. That is an argument for the instrument, not for the ranking.

### The attribution floor

The profiler reproduces to 0.006% over two runs three minutes apart, yet the untouched launches drifted 0.12 to 0.18 ms
per comparison, in opposite directions. `b70-decode --profile --repeats R` therefore runs R sessions of instrumented
replays over one ingest, each rewinding only the position so every session profiles the identical launch shape, and
prints per-family mean, sd and a 2-sigma resolvable delta.

Three independent processes of `--repeats 5` (480 replays, 774 launches):

- **The step-level floor is 0.051 ms including `attn_reduce` and 0.011 ms without it.** Quote both together: the first
  number is dominated by that one family.
- **Within a process is not across processes.** `attn_reduce` reproduces to 0.3% inside every process and lands 31%
  apart between them (194.0, 193.6 and 254.1 us per step) on one binary and box in one twenty-minute window. A data
  dependence is ruled out (identical ids, identical arithmetic); the cause is unknown and likely outside the program.
  Repeats inside one process are therefore a lower bound on the floor: compare two binaries across processes, at least
  5 repeats each. Two binaries compared in one sitting resolve about 0.05 ms; across a day about 0.15 ms.
- **The 0.31 ms unattributed residual is real** (better than 6 sigma), cause unknown.
- `attn_decode` is 223.199 us per launch over the 480 replays (2-sigma 0.019 us), against 224.046 from the single run.
  The next section compares against 223.199.

### What `attn_decode`'s launch buys, named by subtraction

`tools/probe/probe_attn` is a copy of `attn_decode` with one term removed per compiled variant (each computes a wrong
answer on purpose; no runtime path binds one), so every number is a subtraction. Depth 4096, `ATTN_BLOCK` 64, base
207.396 us; the probe sits 7.6% under the live launch, and its fidelity varies by block size over a 6.4-point band
(-5.6% to -12.0%), which is carried as a caveat on every number here.

| what the variant removes | us/launch | share of the launch |
|---|---|---|
| **both KV global loads** | 92.396 | **-55.4%** |
| the whole online softmax update | 153.125 | -26.2% |
| all 6 work-group barriers per wave | 172.500 | -16.8% |
| `exp` alone | 175.312 | -15.5% |
| the loop-carried chain across waves | 204.792 | -1.3% |
| one wave of K prefetch | 206.354 | -0.5% |
| the 1.6 MB of `attn_part` stores | 206.562 | -0.4% |

- **The KV load path is a throughput term, not a latency chain.** Halving K's messages buys 10.8% and quartering them
  buys no more; the launch is linear in live work-groups (68 to 1004 is 14.8x for 11.8x the time); cutting the
  wave-to-wave dependency buys 1.3%. It splits into message issue (39.2%) and cache service (16.2%); the kernel issues
  3.19 M global load messages of 32 bytes per launch.
- **What is genuinely fixed costs about 4.2 us (2%):** the whole 1024-work-group grid with early-out is 1.667 us
  (1.63 ns per dead work-group), the q staging 2.5 us, the stores 0.8 us. The older "90.8 us fixed term" was the first
  of the six q-head passes being colder than the other five: the passes cost 62.6 then about 28.0 us each, so the
  six-pass reread was 67% of the launch.
- **That finding led to register-packed GQA** (each work-item loads its K / V once and serves all six q-heads from
  registers; [12-kernels.md](12-kernels.md)), which removes the messages instead of making them hit and measured
  **-1.075 ms/token** in situ, more than twice what this battery called reachable by tuning (about 0.45 ms). The
  decomposition (position x dim) still sets the message count within one pass.
- **Work-groups are not free at the shipped block size:** "work-group count is nearly free" held only at 320 to 1088
  hardware threads on a device with about 2048 slots. It is withdrawn for the shipped configuration; *dead*
  work-groups are free.
- **Two methodology traps nearly voided the battery.** The GPU clock ramp: the same binary read 264 then 206 us until
  about 100 warm-up launches (now 8 replays, first 3 dropped). And the compiler: an "all hits" ablation that held the
  address constant over a `restrict` pointer let IGC hoist the loads, so it measured message count and inverted the
  split (20.4% apart for V); the control makes the address a function of the loop variable.
- **What died:** the memory-latency model (predicted prefetch to gain 10% or more, measured 0.5%), the online-softmax
  model (predicted 123 us or less, measured 153.1) and the barrier model (157 or less, 172.5); the message-issue model
  was right about the mechanism and 8.6% off in its number.

### What part 2 does not settle

- The 0.31 ms outside the kernels, and `attn_reduce`'s across-process 56.8 us: causes unknown.
- Everything above is the M = 1 decode step on one card. The small kernels are the ones whose costs move most with M;
  the M = 2..8 row lists and the verify kernels are measured in
  [spec 28](superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md), the batching probes
  ([probe-batching-2026-10-23.md](probes/probe-batching-2026-10-23.md)) and BENCHMARKS.md, not here.

## The prefill walk

Decode's anatomy is per-kernel device timestamps off an instrumented capture. **The prefill walk is not a capture**,
so `--profile` cannot see it. `B70_PREFILL_PROFILE=1` times host waits around each phase and inserts one extra wait to
close each (its cost measured at -2.0%, inside the run-to-run spread).

### The launch arithmetic

Derived from the walk in `src/runtime/prefill/step.cc` and asserted by `prefill_smoke_test`. Prefill runs entirely on
Level Zero: a chunk at C = 2048 is **9073 launches with composed attention (8449 with the default flash attention)
and 0 host waits**; `step_head` adds 5 once per `prefill()`. A chunk of a few rows runs each int4 linear as one launch
instead of the slab walk, so a short chunk is far fewer launches.

A GDN layer is 2 norm launches, the slab dequant and GEMM pairs of three int4 linears, the `a‖b` projection, 10
`gdn_chunk` launches and 2 more norms. An attention layer is the same skeleton with `pf_attn_prep`, `pf_attn_gate` and
the attention itself. Only the composed attention depends on C (QK^T per 256-row block so the masked half is skipped);
the rest is C-free, so one binary set serves every `--prefill-chunk` width. The default chunk is 2048; 4096 is built
and opt-in (the default is open-work item 5's call, [open-work.md](open-work.md)).

The removed sycl-tla backend needed 656 host waits and ten times the scratch; moving the linears onto Level Zero was
worth +6.76% on prefill, all of it in the linears, and more launches did not cost host time
([04-architecture.md](04-architecture.md)).

### Where the time goes

Per-phase shares and the ladder of records are in [BENCHMARKS.md](BENCHMARKS.md). Two structural facts:

- **The linears are the walk.** `linear_l0` is over 60% of a chunk. With one wait per slab, a 4096-token prompt measures
  `slab_dequant` 274.5 ms and `slab_gemm` 1344.3 ms over 7616 launches each (an attribution, not a throughput figure:
  15,232 extra waits). The int8 prefill ([17-int8-prefill.md](17-int8-prefill.md)) attacks the GEMM half.
- **The delta-net scan was the biggest miss in this project's estimating record.** A projection that widened decode's
  `gdn_step` to M = 2048 said 15.4 ms per chunk; the first working version measured 984.1 ms, a 64x miss. Rewrites
  and then the split-bf16 DPAS scan took it down (BENCHMARKS.md). **A projection that widens a kernel written for one
  shape to a shape three orders of magnitude away is not a measurement of anything.**

### Chunk width, fixed and per-position cost

A single-chunk sweep at C = 256 / 512 / 1024 / 2048 (2026-09-09, older stack) measured **411.5 ms fixed per chunk**
and **0.852 ms per position** (derived, two-point fit). At C = 256 the fixed term is 70% of the chunk, which is why
throughput rose with the chunk width across the range then available. The widest chunk is now 4096 (the scratch is
sized by the chunk cap, so it is a session choice, not a free flag); its measured effect is in BENCHMARKS.md.
