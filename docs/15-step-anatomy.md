# The step, launch by launch

This document describes what one decode step and one prefill step are made of, in each run mode, and where the time
in a one-card decode step goes, measured kernel by kernel inside the replayed command list. [12-kernels.md](12-kernels.md)
says what each kernel does; this document says how they add up to a step. It is for readers who want to understand
why the kernel work went where it did, and how to measure a step without fooling themselves.

Part 1 is the step as the code runs it today: one card, several requests at once, replica, tensor parallel and
pipeline parallel. Part 2 is the one-card decode profile of 2026-08-25, kept as the record of the method and its
lessons. Its absolute numbers come from that day's software stack (a 42.141 ms step, taken to 36.32 ms by the three
changes it describes), not today's. Current speeds are in the benchmark record. The last section covers the
prefill walk.

## Part 1: what a step is today

### Decode on one card, one request

The step is captured once and replayed for every token. It is one in-order Level Zero command list (a recorded
sequence of kernel launches the driver can resubmit without the host rebuilding it), with every argument baked in:
774 kernel launches for Qwen3.8, replayed at about 0.7 us per launch (part 2 measures this). The three values that
change per token (position, current token, active row count) live in a control block in device memory that the kernels
read and the sampler writes. So the host's work per token is waiting on the fence and reading the answer
([04-architecture.md](04-architecture.md)).

With the MTP head (Qwen3.8's built-in draft head, used by default through `--spec auto`), a step is a loop of three
things:

- draft(k): the head runs k times on one row each, k at most 3, each time proposing the next id;
- verify(k): one replay of the main model over the pending id plus the k drafts, so M = k + 1 rows (at most 4),
  using the small-M verify kernels; the rows are checked in order and the first wrong draft ends the run;
- commit(j, t): row j's recurrent state (the GDN state, the gated delta net's fixed-size memory) becomes the live one,
  and the position moves on by the number of ids kept.

Prompt lookup (`--spec lookup`) proposes its drafts from earlier text instead of from the head and verifies them
through the same verify lists. The number of drafts per iteration follows the request's recent acceptance;
`--spec-max` caps it, at 3 for MTP and lookup.

### Several requests on one card (sequence slots)

`--max-num-seqs N` above 1 gives the engine N slots. A slot is one request's private state: a range of positions in
the one KV cache allocation, its own GDN states, convolution ring and control block, and its own copy of every captured
list.

When only one slot is decoding, that slot's own lists run, exactly the step above, so a lone request computes bitwise
what a one-slot engine would.

When several slots are decoding, the engine replays one row list at M = 2 to 8 rows (it captures one per M) over all
their rows. A 256-byte row table, written by the host before the replay, gives each row its slot, position, KV base
and which GDN state it reads and writes. Rows of one slot are consecutive, a slot appears once, and rows never read
each other's state. With MTP, each slot first runs its own draft lists (one row each), and then one batched verify
checks everyone's drafts at once.

A batched step carries at most 8 rows, and MTP drafts share them with the pending ids. Every decoding slot's pending
id takes one row. The remaining rows go to drafts, dealt one at a time round the slots, longest-running request first,
each slot up to its own draft limit. At four requests, one draft each measured faster than three drafts for the oldest
request alone. With more than one slot on a card, a DFlash drafter or a draft vocabulary is refused at startup.

A batched row is not bitwise equal to the same row run alone: the M-row kernels round the last bf16 digit differently
from the M = 1 GEMV. So a batched row is judged against its lone run by the f32 judge, a float32 CPU reference
([14-golden-gate.md](14-golden-gate.md)). Slot isolation is bitwise: a rejected row writes only its own slot's
speculative state.

The scheduler's tick runs at most one prompt chunk, then one decode step over every decoding slot. A chunk stalls the
other slots while it runs, so it is the engine's ordinary prefill chunk, cut where a single-request prefill would cut
it.

### Replica: one engine per card

`--parallel replica` runs one copy of the model per card, each with its own worker thread, KV cache and slots. A
request goes to the slot that holds its prefix (prefix affinity), and otherwise to the card with the most free slots.
`--max-num-seqs N` is split between the cards, card 0 taking the odd one, up to 4 per card.

Each card runs the step above for its own slots, with its own 8-row budget; a card's drafts are dealt against that
card's rows only. Each card has at most one prompt chunk in flight. While a card prefills a chunk, its decoding rows sit
out the step and nothing else touches that card; the other card keeps stepping, and both cards may prefill at once. A
card with rows decoding starts its next chunk after its step, so it alternates chunk, step, chunk.

The cards fail independently: a card that throws fails only its own rows. The prefix cache is the one shared piece, so
a snapshot saved by card 0 gives a prefix cache hit on card 1.

### Tensor parallel: one request, both cards

With `--parallel tp`, every layer is split over both cards: half the GDN and attention heads, half the MLP, each card's
half of the row-parallel linears (`out_proj`, `o_proj`, `down`, whose partial outputs must be summed across cards) and
half the vocabulary of the output head.

One token is one replay of each card's list, submitted back to back. After each row-parallel linear, a card writes its
partial result straight into the other card's landing buffer (a peer-to-peer write over PCIe), and the next residual
fold waits there for the other half. The two lists therefore order themselves with no host round trip, and both cards
end up holding the same residual, bitwise.

For the output head, each card's argmax names the best id in its half of the vocabulary and that id's logit. After
both fences the host takes the larger, breaking ties toward the lower id, and writes it into both control blocks. That
is the host's one action per token besides the two fence waits.

The MTP head sits whole on each card and runs the same input on both, so it needs nothing to cross between them. Each
draft list takes one fence; the verify list runs the tensor-parallel walk at M = k + 1 rows, with the head merge done
per row.

Every wait has a time limit. A half that does not arrive in time fails the step with a message naming what is missing,
and every later step throws until the engine is reset.

Tensor parallel has no slots and serves one request at a time; `--max-num-seqs` above 1 means replica. Tensor parallel
with slots was tried and ruled out, because replica is faster at 2 to 4 requests with MTP.

Prefill cuts the prompt into chunks and takes them in pairs. Each card walks both chunks layer by layer, alternating,
so that while chunk A's half crosses to the other card on the copy engine, the compute engine runs chunk B's unit. Each
unit then folds both halves.

### Pipeline parallel: the layers split between the cards

With `--parallel pp`, card 0 holds the embedding and the first layers, and card 1 the rest, the output head and the
sampler. One token is one replay of each card's list, submitted back to back; the hand-off between them is a copy or a
peer write, and after the fence the host mirrors card 1's control block into card 0's. Decode speed is about one
card's, since the cards take turns; the gain is memory, for models too big for one card. Prefill is a chunk pipeline:
card 0 runs a chunk's first layers and hands its rows to card 1 while starting on the next chunk.

## Part 2: the one-card decode profile of 2026-08-25

### Read this before any number below

This profile records the first build, not today's step. Two things it measures have since been replaced: the decode
attention (`attn_decode`, now `attn_decode_v2` and its int8-cache twin) and the output head (bf16 then, int8 per row
now). The method and the lessons still hold; the per-kernel times do not describe the current engine. Part 1 is the
current step.

Profile mode is not bench mode. In a profiled list every launch signals a host-visible kernel-timestamp event, which
carries a flush an unprofiled list never pays. Four consequences follow:

- A per-kernel time here is directly comparable with a probe's. The duration runs from `kernelStart` to `kernelEnd`;
  the flush lands after `kernelEnd`, so being observed inflates nothing about a kernel's own execution.
- The gap between kernels is an upper bound, since each gap contains one flush.
- The fence-to-fence wall time is not the time per token. The step time is the `--bench` median in
  the benchmark record.
- A single run's mean is not an attribution: it carries the instrument's drift. "The attribution floor" below prices
  that drift; read it before believing any difference under about 0.3 ms.

The numbers below are measurements unless a sentence says they are derived (arithmetic over two measurements) or
estimated (from a model or an extrapolation).

### How it was measured

```bash
tools/box.sh run "./build/src/cli/b70-decode <checkpoint> --profile --depth 4096 --steps 32"
```

On an idle box, 2026-08-25, one Intel Arc Pro B70, `--max-len 16384`. `--profile` first ingests 4096 synthetic ids on
an uninstrumented list (an ordinary decode), then builds a second, profiled list against the same buffers and replays
32 instrumented steps, resetting every event before each replay and reading the durations only after the fence. The
profiled positions were 4096 to 4127, so 17 attention blocks were live at that day's `ATTN_BLOCK` of 256 positions.

The launch count in the first tables, 645, is that run's. The residual-norm split described below turned each of the
129 `prep_res_norm` sites into two launches, so the walk is now 774, and a run today prints 774 wherever 645 appears.

Two runs of the same command three minutes apart agreed on the sum of kernel durations to 0.006% (41571.315 against
41568.965 us) and on the fence wall time to 0.031%. Per-family spreads ran from 0.02% (`attn_decode`) to 2.5%
(`attn_prep`, the smallest family). Run A is quoted below, and nothing that depends on a difference under about 1% is
claimed.

### The step against the probe floors

Per kernel family, one decode step at depth 4096. The probe floor is the per-kernel time `probe_gemv` measured in
isolation ([12-kernels.md](12-kernels.md)).

| family | launches | in-situ ms | probe-floor ms | difference |
|---|---|---|---|---|
| `gemv` (int4, five shapes) | 256 | 24.629 | 23.995 | +2.6% |
| `gemv_bf16`, `lm_head` | 1 | 4.376 | 4.351 | +0.6% |
| `attn_decode` | 16 | 5.782 | (2.360, estimated) | +145% |
| `prep_res_norm` | 129 | 2.871 | | |
| `gemv_bf16`, `a‖b` | 48 | 2.335 | (never probed) | |
| `gdn_step` | 48 | 0.733 | | |
| `prep_silu_mul` | 64 | 0.623 | | |
| `prep_gated_head`, `attn_reduce`, `attn_prep`, `embed_gather`, `argmax` | 131 | 0.223 | | |
| sum of kernel durations | 645 | 41.571 | | |

The probes predicted the GEMVs well. The 257 GEMV launches cost 29.005 ms in the step against the probes' 28.346 ms
floor, 2.3% above it, and every shape is within 7% of its probe number. The one shape worth naming is `out_proj` /
`o_proj` (+7.0%), the one entered straight from `attn_reduce` or `prep_gated_head`, the two kernels with the least data
in flight.

### The dispatch gap

| | us per step | us per launch | kind |
|---|---|---|---|
| profiled gap (fence wall minus sum of durations) | 851.4 | 1.320 | measured, upper bound |
| uninstrumented gap | 473 | 0.733 | derived |

The derived row is the honest one. The bench step is 42.141 ms, of which 0.097 ms is host time outside the fence, so
the fence covers 42.044 ms; the kernel durations sum to 41.571 ms; the difference is 0.473 ms. So kernel count is not a
lever: 0.473 ms is 1.1% of the step, and fusing kernels just to have fewer of them is not worth it at 0.7 us each. The
instrument's own cost is 0.379 ms per step (derived), all of it in the gap and none in a kernel's duration.

Split by share of the bench step: GEMV 68.8%, `attn_decode` 13.7%, `prep` 8.5%, the `a‖b` GEMV 5.5%, `gdn_step`
1.7%, the other attention kernels 0.3%, embedding and argmax 0.04%, dispatch 1.1%, host 0.2%. 98.6% of the step is
kernel time.

### What the measurement showed that nothing else did

The unit of parallelism is the subgroup, not the work-group. A subgroup is the 16 lanes an Xe core executes together;
a work-group is a set of subgroups that share local memory. Dividing bandwidth by work-groups suggested that one Xe
core could not exceed about 15 GB/s, and that `a‖b` would get four times the bandwidth from four times the work-groups.
That experiment was run and gained nothing (48.774 to 49.127 us at an unchanged subgroup count). Divided by subgroups,
the kernels agree: a latency-bound kernel with one load in flight per thread gets a rate of bytes per load divided by
about 60-75 ns. That held at 8, 16, 32, 128 and 320 subgroups across two kernels, sub-linearly at the top (16 times the
subgroups bought 9.1 times on `a‖b`).

`attn_decode` took 2.45 times its estimate, and four cost models failed to explain why. It measured 5.782 ms in the
step against 2.360 ms extrapolated from a two-point line. Four models, each written down with predictions before the
run that tested it, missed by 14% to 55%. The first model's "depth-independent term" was an artefact of fitting
through a point where the one live block held 65 of 256 positions. The lesson: where a quantity has resisted four fits,
stop fitting and remove terms one at a time (the subtraction battery below). One thing the battery confirmed: the
fixed launch grid is free, since 192 idle work-groups per layer cost 0.04%, so separate command lists per context
length are not needed.

`gdn_step` is not the problem. It measured 0.733 ms against a 0.671 ms memory-traffic floor: 540 GB/s effective, 92%
of the device's bandwidth, 1.09 times its own bound. Making it perfect would save 0.06 ms.

`lm_head` was 10.4% of the step, and no tuning could help it. It took 4.376 ms in one launch at 581 GB/s, 98.5% of the
measured 590 GB/s. The bytes are the lever, not the kernel. The published checkpoint ships a bf16 head, so this
project quotes only comparisons with vLLM that read the same bytes (the benchmark record says why); the
engine has since moved to an int8 head quantised at load ([13-loader.md](13-loader.md)).

### The three changes of 2026-08-25

All three were made against this profile's own ranking, measured with its own instrument on the same day, in this
order, because each one corrected the model the next was designed on.

| change | what shipped | site in the step | bench step, ms | bench difference, ms |
|---|---|---|---|---|
| `a‖b` GEMV K split (`gemv_bf16.cl`) | tiling `{64,1}` to `{16,16}`: 8 to 128 subgroups | 2.341 to 0.256 ms | 42.141 to 40.266 | -1.875 |
| residual-norm split (`prep.cl`) | one single-work-group kernel to `prep_res_fold` + `prep_norm_finish`, 20 work-groups each: 16 to 320 subgroups | 2.893 to 0.484 ms | 40.266 to 38.046 | -2.220 |
| attention block retile | `ATTN_BLOCK` 256 to 64: 68 to 260 live work-groups | attention family 6.058 to 3.839 ms | 38.046 to 36.32 | -1.726 |
| all three | | | 42.141 to 36.32 | -5.821 |

The golden gate was exact on every determined row before and after each change.

The K split: giving each work-group 16 columns with no K split (more work-groups, the same eight subgroups) gained
exactly nothing, and neither did variants with more loads in flight (one of them also cost `lm_head` 1.3% by
perturbing its code). Only splitting K makes more subgroups. It shipped at 5.34 us against a 2.22 us bandwidth floor,
so 0.15 ms per token is all that is left there.

The norm split: RMSNorm's mean is over the whole row, so no tiling inside one launch could help, but two launches did.
The fold is 60% of the win. The rescale pass, which looked like a formality, is the other 40%, because one work-group
walked it at 4.5 GB/s. The launch count rose from 645 to 774 (0.095 ms, derived) to save 2.409 ms, a 25:1 trade, and
the measured answer to the worry that kernel count matters.

The retile: `ATTN_BLOCK` 32 was 0.070 ms per token better and was rejected because it needs a 101.4 MB `attn_part`
buffer.

The in-step and bench differences never agreed exactly; they differ by 0.892 ms in total. Untouched launches drifted
+0.30%, +0.50% and +0.56% between runs, and after the retile 0.31 ms between the in-step prediction and the bench stayed
unexplained, recorded as such rather than rounded away.

All three changes delivered, and none of them for the reason the ranking predicted. That is an argument for the
instrument, not for the ranking.

### The attribution floor

The profiler reproduces to 0.006% over two runs three minutes apart, yet the untouched launches drifted 0.12 to 0.18 ms
per comparison, in opposite directions. So `b70-decode --profile --repeats R` runs R sessions of instrumented replays
over one ingest, each rewinding only the position so that every session profiles the identical launch shape, and
prints each family's mean, standard deviation and the smallest difference it can resolve at 2 sigma.

Three independent processes of `--repeats 5` (480 replays, 774 launches each) showed:

- The step-level floor is 0.051 ms including `attn_reduce` and 0.011 ms without it. Quote both together; the first is
  dominated by that one family.
- Repeats within a process are not repeats across processes. `attn_reduce` reproduces to 0.3% inside every process and
  lands 31% apart between them (194.0, 193.6 and 254.1 us per step) on one binary and one box within twenty minutes. A
  data dependence is ruled out (identical ids, identical arithmetic); the cause is unknown and probably outside the
  program. So repeats inside one process give only a lower bound on the floor: compare two binaries across processes,
  with at least 5 repeats each. Two binaries compared in one sitting resolve about 0.05 ms; across a day, about 0.15 ms.
- The 0.31 ms unexplained residual is real (better than 6 sigma); its cause is unknown.
- `attn_decode` is 223.199 us per launch over the 480 replays (2-sigma 0.019 us), against 224.046 from the single run.
  The next section compares against 223.199.

### What `attn_decode`'s time is spent on, found by subtraction

`tools/probe/probe_attn` is a copy of `attn_decode` with one term removed per compiled variant. Each variant computes a
wrong answer on purpose, and no runtime path binds one; every number is a subtraction from the full kernel. At depth
4096 and `ATTN_BLOCK` 64 the base is 207.396 us. The probe runs 7.6% under the live launch, and its fidelity varies by
block size over a 6.4-point band (-5.6% to -12.0%), which applies as a caveat to every number here.

| what the variant removes | us per launch | share of the launch |
|---|---|---|
| both KV global loads | 92.396 | -55.4% |
| the whole online softmax update | 153.125 | -26.2% |
| all 6 work-group barriers per wave | 172.500 | -16.8% |
| `exp` alone | 175.312 | -15.5% |
| the loop-carried chain across waves | 204.792 | -1.3% |
| one wave of K prefetch | 206.354 | -0.5% |
| the 1.6 MB of `attn_part` stores | 206.562 | -0.4% |

The KV load path is a throughput cost, not a latency chain. Halving K's load messages gains 10.8% and quartering them
gains no more; the launch time is linear in live work-groups (68 to 1004 is 14.8 times the work-groups for 11.8 times
the time); and cutting the wave-to-wave dependency gains 1.3%. The load cost splits into issuing messages (39.2%) and
the cache serving them (16.2%); the kernel issues 3.19 million global load messages of 32 bytes per launch.

What is truly fixed costs about 4.2 us (2%): the whole 1024-work-group grid with early exit is 1.667 us (1.63 ns per
dead work-group), staging q is 2.5 us, and the stores are 0.8 us. An older "90.8 us fixed term" turned out to be the
first of the six query-head passes running colder than the other five: the passes cost 62.6 us and then about 28.0 us
each, so re-reading K and V six times was 67% of the launch.

That finding led to register-packed GQA (grouped-query attention, where six query heads share one KV head): each
work-item loads its K and V once and serves all six query heads from registers ([12-kernels.md](12-kernels.md)). It
removes the messages instead of making them hit the cache, and measured -1.075 ms per token in the step, more than
twice what this battery called reachable by tuning (about 0.45 ms). The decomposition by position and dimension still
sets the message count within one pass.

Work-groups are not free at the shipped block size. "Work-group count is nearly free" held only at 320 to 1088
hardware threads on a device with about 2048 slots, and it is withdrawn for the shipped configuration. Dead
work-groups, which exit at once, are free.

Two measurement traps nearly voided the battery. The first was the GPU clock ramp: the same binary read 264 us and then
206 us until about 100 warm-up launches had run (the probe now does 8 replays and drops the first 3). The second was
the compiler: an "all hits" variant that held the address constant over a `restrict` pointer let IGC (Intel's GPU
compiler) hoist the loads out of the loop, so it measured message count and inverted the split (20.4% apart for V).
The control variant makes the address a function of the loop variable.

Three models died here: the memory-latency model (it predicted prefetch would gain 10% or more; it gained 0.5%), the
online-softmax model (it predicted 123 us or less; measured 153.1) and the barrier model (157 or less; measured
172.5). The message-issue model was right about the mechanism and 8.6% off in its number.

### What part 2 leaves open

The 0.31 ms outside the kernels, and `attn_reduce`'s 56.8 us spread across processes, have unknown causes. Everything
above is the M = 1 decode step on one card. The small kernels are the ones whose costs move most with M; the M = 2 to
8 row lists and the verify kernels are measured in the benchmark record, not here.

## The prefill walk

Decode's profile comes from per-kernel device timestamps on an instrumented capture. The prefill walk is not a
capture, so `--profile` cannot see it. Instead, `B70_PREFILL_PROFILE=1` times host waits around each phase, inserting
one extra wait to close each; that costs -2.0%, inside the run-to-run spread.

### The launch count

The counts below are derived from the walk in `src/runtime/prefill/step.cc` and checked by `prefill_smoke_test`.
Prefill runs entirely on Level Zero. A chunk of C = 2048 positions is 9073 launches with composed attention, or 8449
with the default flash attention, and 0 host waits; `step_head` adds 5 once per `prefill()`. A chunk of only a few rows
runs each int4 linear as one launch instead of walking it slab by slab, so a short chunk has far fewer launches.

A GDN layer is 2 norm launches, the slab dequantisation and GEMM pairs of three int4 linears, the `a‖b` projection, 10
`gdn_chunk` launches and 2 more norms. An attention layer has the same skeleton with `pf_attn_prep`, `pf_attn_gate` and
the attention itself. Only the composed attention's count depends on C (it computes QK^T per 256-row block, so the
masked half is skipped); the rest does not, so one set of binaries serves every `--prefill-chunk` width. The default
chunk is 2048; 4096 is built and opt-in.

The sycl-tla backend that came before needed 656 host waits per chunk and ten times the scratch memory. Moving the
linears onto Level Zero was worth +6.76% on prefill, all of it in the linears, and the extra launches cost no host
time ([04-architecture.md](04-architecture.md)).

### Where the time goes

Per-phase shares and the history of prefill records are in the benchmark record. Two structural facts
matter here.

The linears are the walk: `linear_l0` is over 60% of a chunk. With one wait per slab, a 4096-token prompt measures
`slab_dequant` at 274.5 ms and `slab_gemm` at 1344.3 ms, over 7616 launches each. That is an attribution, not a
throughput figure, since it adds 15,232 waits. The int8 prefill ([16-int8-prefill.md](16-int8-prefill.md)) attacks the
GEMM half.

The delta-net scan was the biggest miss in this project's estimates. A projection that widened decode's `gdn_step` to
M = 2048 said 15.4 ms per chunk; the first working version measured 984.1 ms, 64 times more. Rewrites, and then the
split-bf16 DPAS scan, brought it down (the benchmark record). A projection that stretches a kernel written for one shape to a
shape three orders of magnitude away measures nothing.

### Chunk width: fixed and per-position cost

A single-chunk sweep at C = 256, 512, 1024 and 2048 (2026-09-09, an older stack) measured 411.5 ms fixed per chunk and
0.852 ms per position (derived from a two-point fit). At C = 256 the fixed part is 70% of the chunk, which is why
throughput rose with chunk width across the range then available. The widest chunk is now 4096. The scratch memory is
sized by the chunk cap, so the width is chosen per session, not changed freely; its measured effect is in
the benchmark record.
