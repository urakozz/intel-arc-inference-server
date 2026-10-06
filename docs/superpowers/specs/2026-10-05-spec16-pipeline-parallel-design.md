# Spec 16 - pipeline parallel over two B70s

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.
16b (decode) built blind 2026-10-06, before 16a's probe: §8. 16c (the prefill chunk pipeline) built
blind 2026-10-06 on top of it: §9. 16d (b70-serve --pp 2, MTP across the split, P3 / S3 as box
stages) built blind 2026-10-06 on both: §10.

**Scope, set by the operator (2026-10-05):** multi-GPU is **pipeline parallel only** for now; tensor
parallel is not part of this spec.

**Hardware:** the box has two Arc Pro B70s, extendable to four (docs/10-the-box.md), on one CPU
root complex over PCIe; peer-to-peer between them is enabled at the OS level. The design is
written for two devices and keeps the device count a parameter. The second card is measured
3.2-3.4 % slower than the first on prefill and 0.2-0.4 % on decode (docs/10-the-box.md).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why, and what it does not do

Pipeline parallel (PP) splits the model's layers between the two cards: layers `[0, s)` on device 0,
`[s, 64)` on device 1 (Qwen3.8), and passes the residual stream from one to the other.

| | PP = 2 (estimates) |
|---|---|
| **single-stream decode** | **not faster**: the layers still run one after the other; the hand-off adds one ~10 KB transfer per token (5120 bf16), a few microseconds over PCIe |
| **memory** | **per card about halved**: ~9 GB of weights each instead of 18 (derived); room for Qwen3.8 at 262144 positions with bf16 KV (17.2 GB of KV split in two), Agnes 3.0 Flash at 128k without spec 12's int8 KV, larger models later |
| **long prefill** | **up to ~2x**: chunk `c` runs layers `[s, 64)` on device 1 while chunk `c + 1` runs `[0, s)` on device 0. The hand-off is ~20 MB per 2048-row chunk (2048 x 5120 bf16), ~1.7 ms at ~12 GB/s (a PCIe 3.0 link; P0 measures it), against ~1 s of compute per chunk (derived). This is the agentic case that matters: a prefix-cache miss on a long history |
| **concurrent requests** | ~2x aggregate once spec 13's batching exists (two micro-batches in flight); out of scope here |

## 2. The decisions

**1. One process, one Level Zero context holding both devices.** Not two processes with IPC (as
the TP all-reduce work needed for a multi-rank framework): our engine owns both devices, so
peer access, cross-device copies and events are available inside one context without exporting
handles.

**2. The split point `s` (decide).** Balance memory and time. The embedding (2.54 GB bf16) sits on
device 0, `lm_head` (2.54 GB bf16, 1.27 GB int8) and the MTP head (0.85 GB) on device 1; the FA
layers (the KV holders) are every fourth layer, so any split near the middle gives each card 8 FA
layers. Proposed: `s = 32` (16 FA layers split 8 / 8), revisited by P0 with device 1's 3 % slower
prefill (a split of 33 / 31 may balance time better). The card holding `lm_head` is the heavier one:
vLLM's PP = 2 run of K2-Horizon on these cards needed an uneven 25 / 23 layer partition to even the
per-card weights, because total KV is set by the smaller remainder. The split is chosen on bytes
(weights + KV per card) as well as time.

**3. The hand-off mechanism is chosen by P0** among:
- (a) a copy-engine copy of the residual row(s) from device 0 to a buffer on device 1, appended to
  device 0's captured list, signalling an event that device 1's captured list waits on (both lists
  captured once, no host round trip per token);
- (b) a peer write: device 0's last kernel writes the residual straight into device 1's buffer and
  raises a flag; device 1's first kernel polls it;
- (c) host-orchestrated (submit list 0, wait, submit list 1): the baseline.

**One more lesson from vLLM's PP on these cards:** with peer access enabled, a communication
backend that silently falls back (vLLM's oneCCL with socket IPC exchange) can return from a receive
having moved zero bytes, and the second card then decodes a stale buffer. Every hand-off is
therefore checked end to end in P1 (bitwise equality) and in a debug mode that stamps a sequence
number into the handed-off buffer.

**Two rules from peer-to-peer work on this exact card pair, applied to (b):** a flag written by the
peer over PCIe must be polled with **system-scope atomics** (a plain or `volatile` load spins on a
stale cache line and can miss the write indefinitely); and the buffer the peer writes into is an
**allocation of its own, page-aligned**, holding nothing the local device writes (peer writes over
PCIe have been seen to disturb neighbouring data in the same allocation). Any device-side sequence
counter lives in device memory and is advanced by the kernel, so a replayed list keeps making
progress with frozen arguments.

## 3. Design

### 3.1 Placement

- Device 0: embedding, layers `[0, s)` with their GDN state, conv ring and KV, decode list 0,
  prefill context 0.
- Device 1: layers `[s, L)` with their state, final norm, `lm_head`, argmax, the MTP head (spec 8)
  and its KV, decode list 1, prefill context 1. The MTP head needs `embed_tokens` for drafts:
  **(decide)** replicate the embedding on device 1 (2.54 GB, simplest) or gather the needed rows from
  device 0 over P2P per draft step.
- `Control` (pos, pending token): one per device, both advanced each step from the same values; the
  sampled / argmax token produced on device 1 reaches device 0's embedding gather through the
  hand-off path in the other direction (one int per step).

### 3.2 Decode

Per token: device 0 replays list 0 (embed, layers `[0, s)`, hand-off out), device 1 replays list 1
(hand-off in, layers `[s, L)`, head, argmax, token back to device 0). With (a) or (b) the host
submits both lists and waits once on device 1's fence. MTP's draft / verify / commit (spec 8) run on
both devices for the verify (M = K + 1 rows handed off together) and on device 1 for drafts.

### 3.3 Prefill: the chunk pipeline

Two prefill contexts, one per device, and double-buffered hand-off buffers on device 1
(`[2][C][5120]` bf16, ~40 MB). Chunk `c`: device 0 runs `[0, s)` and hands off into buffer
`c mod 2`; device 1 runs `[s, L)` on buffer `c mod 2` while device 0 starts chunk `c + 1`. The last
chunk's last row goes through the head on device 1. Prefill backends (`l0`, `l0-int8`, flash
attention) are unchanged per device.

### 3.4 What else changes

- **Spec 7 snapshots:** each device saves and restores the state of its own layers; the host store's
  entries become `{device 0 part, device 1 part}`; the per-block hook fires on device 1 when the
  block's last layer is done.
- **Spec 9 / 10 / 14 / 15:** per-device kernel variants are the same binaries; `ModelDesc` gains the
  split (`layers_on(device)`).
- **CLI:** `--pp N` / `--pipeline-parallel-size N` (vLLM's flag; N = 1, the default, or 2) and
  `--pipeline-split auto|N`; `ZE_AFFINITY_MASK` must expose both cards.
- **The box's GPU lock** (`~/b70-gpu.lock`) is taken once for both cards by a PP job.

## 4. Correctness gates

- **P1, bitwise equal to one card.** PP changes where layers run, not what they compute: the same
  kernels in the same order on the same data, plus a copy. Decode and prefill outputs, logits, KV and
  GDN state must be **bitwise identical** to the single-device run (on device 0) for the golden
  prompts, a 32k prompt, and 64 greedy tokens. Device 1 runs the same binaries; if a kernel's result
  depended on the device it would show here.
- **P2:** the golden gates, replay determinism, `prefill_split_*`, spec 7's C1/C2, spec 8's M2, all
  with `--pp 2`.
- **P3, long context:** Qwen3.8 at max_len 262144 loads with both cards (the memory report per
  device); passkey at 5 / 50 / 95 % of ~250k.
- **P4, failure modes:** a missing second device, P2P not available (`zeDeviceCanAccessPeer` false),
  or a hand-off timeout are refused with a clear error, never a hang.

## 5. Speed bars

Idle box (both cards free of other DRM holders), interleaved pairs against one card, median of 3.

- **P0 (first):** P2P capability and bandwidth both ways (copy engine and kernel peer writes), the
  three hand-off options' latency for 10 KB and 20 MB, and their cost inside a replayed list.
  **Plus one arm for spec 17 (tensor parallel):** the "remote partial" pattern: a decode GEMV on
  each card writes its split-K partial sums for a 5120-wide row both locally and into the peer's
  buffer (peer write + system-scope flag), and a `prep_res_fold`-style kernel on each card waits for
  the flag and folds local and remote partials in a fixed order; measured as the added latency per
  fold inside a replayed list, against a separate all-reduce kernel. It is spec 17's key number,
  taken here because the probe already has both cards and the peer-write machinery.
- **S1, decode:** `--pp 2` within **2 %** of one card at 4k and 32k depth (the hand-off must not cost
  more than that).
- **S2, long prefill:** pp32768 and pp65536 **>= 1.7x** one card; pp4096 (two chunks, so little
  overlap) **>= 1.2x** (estimates; P0 refines them).
- **S3, memory:** Qwen3.8 at 262144 fits with both cards (recorded per device).

## 6. Stages

- **16a, probe:** P0 on the box with both cards; the hand-off choice; the split proposal.
- **16b, decode PP:** placement, per-device lists, the hand-off, `--pp 2`; P1 (decode), P2 (decode
  parts), P4; S1.
- **16c, prefill pipeline:** two prefill contexts, double-buffered hand-off, the chunk pipeline;
  P1 (prefill), P2; S2.
- **16d, integration:** spec 7 snapshots per device, MTP across the split, the server flag, P3, S3,
  the record.

## 7. Out of scope

- Tensor parallel (operator, 2026-10-05): **spec 17**, built on this spec's multi-device foundation
  (one context with both cards, per-device buffers and capture, the peer-write rules) and P0's
  remote-partial arm. PP and TP are alternatives over the same two cards (`--pp` or `--tp`).
- More than two devices (the layout extends to four; the placement and hand-off generalise, but
  this spec builds and gates two).
- PP combined with batching (two micro-batches in flight): after spec 13.
- Splitting one layer across cards.

## 8. 16b as built blind (2026-10-06)

Written on the Mac without the box: multi-device Level Zero cannot run on a Mac, so 16b is
validated by host tests, syntax checks and reasoning only. **16a has not run**, so the two
choices it would have made are open and 16b builds both sides of each. Box queue row 22 is the
order to prove it on the cards.

**The CLI is `--pp N` / `--pipeline-parallel-size N`, vLLM's spelling** (the operator's
ruling, 2026-10-06). 16b was built as `--pipeline 1|2`, because b70-decode's `--pp N` was then
the bench's prefill length (llama-bench's "pp") beside `--pp-chunk` and `--pp-backend`. The
rename moved those to `--prefill-length N`, `--prefill-chunk C` and `--prefill-backend B` (no
aliases) and gave `--pp` vLLM's meaning; an old `--pp 4096` is refused naming
`--prefill-length` (any N but 1 or 2 is), never read as a 4096-way pipeline. So: `--pp N`
(N = 1 or 2), `--pipeline-split auto|N`, `--pipeline-handoff copy|peer` (b70-decode; b70-serve
is 16d). This text keeps "PP" for the concept.

**What 16a would have decided, and the defaults chosen.**

| decision | 16a's job | 16b ships |
|---|---|---|
| the hand-off (§2 decision 3) | latency of (a) copy + event, (b) peer write + flag, (c) host-orchestrated, inside replayed lists | (a) as `copy` (default) and (b) as `peer`; (c) not built. `copy` is the default because it needs no new device code to be correct: the driver orders the copy, the event and the cache flushes. `peer` carries the two P2P rules below and is first timed by row 22's S1 |
| the split (§2 decision 2) | per-layer timing on each card, `s` = 32 or 33 / 31 | `--pipeline-split auto` = equal **bytes** per card at the session's max_len (weights + RoPE + its layers' KV and GDN state + decode scratch + hand-off buffers), not equal layers. Decode is sequential, so time balance does not matter for 16b; 16c's prefill pipeline is where the second card's 3 % may move it |

**Placement (§3.1), as built.**
- One Level Zero context over both cards: `l0::Context(primary, 1)` is a view of device 1 in
  device 0's context (zeContextCreate's context already spans the driver's devices), so every
  existing buffer, module, list and queue class works per device unchanged, and cross-device
  copies, peer pointers and events need no export.
- **Load, then place** (runtime/pipeline_place.h): the loader is untouched and loads the
  whole checkpoint onto device 0; layers `[s, L)`, the final norm, `lm_head` (and an int4 a||b's
  prefill copies) move to device 1 through a 64 MiB pinned host buffer (no P2P needed). Every
  weight is byte-for-byte a one-card load's. Cost: device 0 briefly holds the whole model
  (18.1 GB on Qwen3.8 with the bf16 head; every supported model fits one card). A model larger
  than one card needs per-layer placement in the loader (16d or that model's spec).
- Each device: `PersistentBuffers`' stage constructor (its own GDN / FA layers only, the same
  layout, so stage 0's slices then stage 1's ARE the one-card layout), a whole
  `DecodeScratch`, its own queue, fence and Control block.

**The cut (capture.h, `build_stage`).** Device 0's list is `embed_gather`, layers `[0, s)`, then
layer `s`'s `prep_res_fold` - the fold reads no weight, so device 0 runs it and hands off the
**folded** residual plus its norm sums (M rows bf16 + the 640-byte `norm_sumsq`: 10.9 KB on
Qwen3.8) instead of the residual plus `down`'s split-K partials (90 KB). Device 1's list resumes
at layer `s`'s `prep_norm_finish`. Same kernels, arguments and order as the one-card list:
the two lists' compute launches add up to `decode_launches()` (774 / 870 / 526), asserted at
capture. The split comes from the descriptor (`pp_stages`), so Agnes and Ornith split without
code; each device must hold at least one GDN and one FA layer (Qwen3.8: 4 <= s <= 62).

**The hand-off.**
- `copy`: device 0's list ends with two device-to-device copies into device 1's landing buffer
  and a barrier signalling a cross-device event (`l0::SyncEvent`: a one-slot HOST_VISIBLE pool
  over both devices, HOST signal / wait scope); device 1's list starts with a wait on it and two
  local copies into its own `resid` / `norm_sumsq`. The host resets the event before each step.
- `peer` (`src/kernels/pp_handoff.cl`, the one new binary): `pp_send`, device 0's last launch,
  writes the rows, the sums and a stamp (the sequence number) into device 1's landing buffer,
  then publishes the sequence number in a flag with a system-scope release store. `pp_recv`,
  device 1's first launch, spins with system-scope acquire loads up to `spin_limit` times, checks
  the stamp, and copies the data in **with system-scope loads too** (device 1 read the same
  landing lines one step earlier - a cached copy of them is exactly the stale buffer of §2's vLLM
  lesson). Both counters live in device memory and are advanced by the kernels, so replays with
  frozen arguments progress. The landing buffer is its own 64 KiB-aligned allocation, only device
  0 writes it, the flag on a page of its own (§2's two rules). A failure (timeout; a flag whose
  stamp did not arrive) is recorded in device-1 state words the host reads after the fence. The
  protocol runs line for line on two host threads in `pp_protocol_test`.

**The token back, and `pos` (§3.2, plan Review Focus 2-3).** argmax on device 1 writes the id and
advances `pos` in device 1's Control. After the fence the host copies device 1's 128-byte Control
into device 0's: no extra wait or submission, the host already reads the block after every fence.
Nothing on device 0 writes its block, so the two are equal at every step boundary: reset, ingest,
generate, `load_state` (spec 7) all set both. A device-side return (a copy at the end of list 1)
was not chosen blind: device 0's Control is a shared allocation of device 0, and a peer write
into it is the least certain thing on this list.

**Bounded (P4).** Every step's fence wait has a bound (30 s default); a lost `copy` hand-off
host-signals the event so device 1 drains, then throws naming the event; `peer` throws with
pp_recv's record. A failed step marks the engine; every later step throws until `reset()`. A
missing second card and no peer access are refused before the load, by name.

**Spec 7 snapshots are in** (planned for 16d): `save_state` / `save_kv` under `--pp 2`
write the one-card host layouts byte for byte, so a snapshot moves between one card and two
unchanged - the refusal 16d planned between the two is not needed. P1 uses it: a one-card
prefill's snapshot is restored into the pipeline and decoded on.

**Not in 16b:** prefill across two cards (16c; `--pp 2` ingests one replay per id), MTP and
`b70-serve --pp 2` (16d), per-device step times (S1 records the total), K2-Horizon (its own
engine, `runtime/k2`; `pipeline_plan_test` gives its bytes per card through the descriptor-free
`pp_balance` - 24 + 24 layers - but K2 across two cards is future work).

**The planner's numbers** (derived, `pipeline_plan_test`; weights from the descriptor, which
sum to the measured Qwen3.8 load exactly):

| model, head, KV, max_len | split | device 0 / device 1 |
|---|---|---|
| Qwen3.8 bf16, bf16, 16384 | 32 (32 + 32) | 9.690 / 9.690 GB |
| Qwen3.8 int8, bf16, 16384 | 29 (29 + 35) | 9.009 / 9.101 GB |
| Qwen3.8 bf16, bf16, 262144 | 32 | 17.806 / 17.806 GB - fits two cards (one card's auto: 193792) |
| Qwen3.8 int8, int8, 262144 | 31 (31 + 33) | 12.806 / 13.013 GB |
| Agnes int8, bf16, 16384 | 33 (33 + 39) | 10.451 / 10.644 GB |
| Agnes int8, bf16, 131072 | 35 (35 + 37) | 14.688 / 14.921 GB |
| Ornith bf16, bf16, 16384 | 20 (20 + 20) | 10.203 / 10.203 GB |
| Ornith int8, bf16, 262144 | 20 | 12.782 / 12.275 GB |
| K2-Horizon int8, bf16, 131072 (weights + KV only) | 24 (24 + 24) | 23.216 / 23.715 GB |

**Gates as built:** P1 `pp_decode_test` (+ `_i8head`, `_kv8`), P4 `pp_fail_test` and
`cli_reject_pipeline_*`, S1 row 22's `r22.s1` (4k; 32k opt-in). P2 / P3 / S2 / S3 stay 16c / 16d.

## 9. 16c as built blind (2026-10-06)

Written on the Mac like 16b, on top of it (§8); box queue row 23 is the order to prove it on the
cards. `b70-decode --pp 2 --prefill` and `--bench --prefill-length N --pp 2` run it
(runtime::PipelineEngine::prefill, defined in b70_prefill_host as Engine::prefill is).

**Where 16b's build overrides plan 16c** (16b wins; the plan was written before it):

| plan 16c | as built, and why |
|---|---|
| hand off the chunk's hidden rows | 16b's cut: device 0 also runs layer s's `pf_res_fold` (it reads no weight) and hands off the FOLDED residual rows plus the norm sums (`[C][5120]` bf16 + 160 KB on Qwen3.8), not the residual and down's fp32 partials (twice the bytes); device 1 starts layer s at its `pf_norm_finish` |
| the hand-off | 16b's two, `--pipeline-handoff copy` (default) or `peer`. `peer` moves the rows by `pp_send`'s stores into device 1's slot and checks them with `pp_recv` (flag + stamp, system-scope loads) - but in prefill device 1 also waits on the ready event before `pp_recv`: a stage is ~0.5 s at 4k and seconds near 256k, and `pp_recv`'s bound counts flag loads, not time, so a spin there could time out on a slow chunk. The spin is then a check, not a wait |
| the split balances time (the second card ~3 % slower) | 16b's byte-balanced `--pipeline-split auto`, now planned with each card's prefill scratch; 16c reports each card's busy time per prefill instead of guessing the time balance, and row 23's opt-in sweep times the neighbouring cuts |
| `--pp 2` refuses the composed path | so it does, and sycl-tla (its walk waits on the host between runtimes), `B70_PREFILL_REPLAY=1` (recordings are one card's walk) and `B70_PREFILL_PROFILE=1` (its phase waits would serialise the pipeline) - each by name, before the device where the flags say it |
| the spec 7 hook fires on device 1 after the block's last layer | the hook is the host's (as on one card): it runs once BOTH devices have finished the block. Device 0 has moved on by then, so each device copies its GDN state and conv ring into a shadow after a hooked chunk, in order on its list, and `save_state` inside the hook reads the shadows; `save_kv` reads `[0, end)` live (later chunks write positions >= end) |

**The order** (runtime/pipeline_prefill_plan.h; one executor, `pp_prefill_run`, runs it in the
engine and in the host-thread protocol test). The chunks are one card's - `prefill_chunk_rows` is
one function, called by `Engine::prefill` and by the planner - so the gated delta rule's 64-row
chunks fall where they fall on one card. Every per-chunk resource is doubled and indexed by
chunk % 2: the landing slot, each device's prefill Control block (the walk's kernels read pos /
n_active while the host writes the next chunk's), device 0's ids buffer, the ready event
(device 0 -> 1), each device's done event and shadow. **The back-pressure rule:** chunk j is
appended to either device only after the host has seen BOTH finish chunk j - 2. Device 0 is
then never more than two chunks ahead of device 1, a slot is never rewritten while device 1 may
still read it, and an event is host-reset only when nothing waits on it or will signal it. The
plan's "device 0 waits only when both buffers are full" is this rule seen from device 0; the
difference is that device 0 waits for device 1 to *finish* chunk j - 2, not merely to copy it in -
the same rate in steady state (the slower card sets it), one fewer event, and every reuse a host
fact. Host loop, per chunk j: `[j >= 2: wait0 j-2, wait1 j-2, hook j-2]`, append device 0's
chunk (embed, layers [0, s), the cut's fold, timestamps, the hand-off out, signal ready, signal
done), append device 1's chunk (wait ready, the hand-off in, layers [s, L), signal done); at the
end chunk n - 2's waits and hook, the head on device 1, the last waits, the drain.
`pp_prefill_check` names every rule; the engine asserts it on the order it runs, and the host
tests show each single-step drop of the order (48) and six named reorderings are caught.

**Bitwise (P1), by construction:** `step_stage` is `step_chunk`'s loop (`walk_layers`) over the
stage's layers - the same launches, arguments and order per layer, the stage's own GDN / FA
slices (16b's layouts) - plus the hand-off's copies. The prefill Control blocks hold the same
pos / n_active one card's Control does; the head writes device 1's decode Control, and the host
mirrors it into device 0's as after a decode step. The launch arithmetic (`step_stage_launches`)
adds up to `step_chunk_launches` for both stages at every C (asserted at `prepare_prefill`).

**Bounded:** every host wait - a device's per-chunk done event, the final drain - has
`PipelineOptions::prefill_timeout_ms` (120 s; a hang, not a slow chunk). A wait that passes its
bound (or a `pp_recv` that reports a bad hand-off) host-signals every event device 1 may wait
on, lets both lists drain (bounded), marks the engine and throws naming the device, the chunk
and, when it is so, that the ready event was never signalled; `reset()` recovers. A hook that
throws is the hook's exception, not a failure: the lists drain, the shadows go back, pos is the
block end - one card's behaviour.

**Memory** (derived, `pipeline_prefill_plan_test`): each card holds a whole `PrefillScratch`
(0.70 GB on Qwen3.8; plus the l0 backend's slab) and on l0-int8 its own `Int8State` (the h8
scratch and ITS layers' column scales, 0.069 GB a card at the even split), device 1 the two
landing slots (42.3 MB), each card two Control blocks and its timestamps; with a block hook each
card also allocates two shadows of its state (166.7 MB a card at the even split; b70-decode
never sets a hook - the server's prefix cache would, 16d). Qwen3.8 bf16 head, l0-int8 prefill:
16384 -> split 32, 10.459 / 10.501 GB; 262144 -> split 32, 18.575 / 18.617 GB (fits two cards
with the default 1.5 GB reserve).

**Speed, expected** (derived; row 23's S2 measures): with two equal halves of a one-card chunk
time T, n chunks take (n + 1) T / 2 instead of n T - 1.33x at pp4096 (2 chunks), 1.88x at
pp32768, 1.94x at pp65536; the second card's ~3 % slower half takes ~0.05 off at n >= 16. The
copy hand-off is ~1.7 ms of 20 MB per chunk at ~12 GB/s on device 0's list (~0.4 % of a ~0.48 s
half-chunk). The peer path's `pp_send` / `pp_recv` are 16b's single work-group kernels: moving
20 MB through one work-group is the unmeasured term - if row 23's peer arms show it, a
multi-group variant is the fix (copy stays the default).

**Per-card busy time** (plan Review Focus 5): `zeCommandListAppendWriteGlobalTimestamp` before
and after each chunk's walk on each card; `last_prefill()` sums them, b70-decode prints
`pp: device N busy ... ms (x % of the wall)` per card.

**Not in 16c:** `b70-serve --pp 2` and its prefix cache (16d: the hook path it would use is built
and gated by `pp_prefill_test`), MTP (16d), the time-balanced split (measured first), a
multi-group peer copy, more than two chunks in flight per card.

**Gates as built:** host `pipeline_prefill_plan_test`, `pp_prefill_protocol_test` (TSan clean),
`pipeline_args_test`; on the cards P1 / P2 `pp_prefill_test` (+ `_l0`, `_i8head`, `_kv8`),
back-pressure and P4 `pp_prefill_fail_test`, `cli_reject_pipeline_*_sycl`, row 23's CLI stages
(Qwen3.8, Agnes, Ornith) and S2 (`r23.s2_4k` / `_32k` / `_64k`; 128k and the split sweep opt-in).

## 10. 16d as built blind (2026-10-06)

Written on the Mac like 16b and 16c, on top of both; box queue row 27 is the order to prove it on
the cards (`tools/box_validate.sh --only r27`). Nothing here has run on a B70: P2, P3 and S3 are
stages of that row, not results. **What works under `--pp 2` now** (by construction and host
tests): `b70-serve --pp 2` for Qwen3.8, Agnes and Ornith with `--max-len auto|N`,
`--mem-reserve-gb`, `--prefill-backend l0|l0-int8`, `--kv-cache bf16|int8`, `--lm-head`, the
prefix cache (`--prefix-cache-gb auto|N`, `--prefix-split-last`), `--mtp K|auto` / `--spec mtp`,
`--draft-vocab`, `--spec lookup`, `--pipeline-split`, `--pipeline-handoff`; and
`b70-decode --pp 2 --mtp K|auto` (16b's refusal lifted). **Refused by name, before any device**
(`cli/pipeline_serve.h`): K2-Horizon (its own engine; two-card K2 is future work - serve it on
one card), Kolibri-1 (not served at all; its serving, two cards included, is spec 20e),
`--device N`, a sycl-tla prefill, `B70_PREFILL_ATTN=composed`, `B70_PREFILL_REPLAY=1`,
`B70_PREFILL_PROFILE=1`; `--pipeline-split` / `--pipeline-handoff` without `--pp 2`. `--pp 1`
is today's engine and server: no binary added, the one-card paths untouched apart from the
verify walk binding its slots through one accessor (the same pointer) - G0 is the check.

**Where 16b / 16c's build overrides plan 16d** (the as-built code wins; the plan predates it):

| plan 16d | as built, and why |
|---|---|
| Review Focus 2: a snapshot saved under `--pp 2` refused when restored under `--pp 1` and vice versa | **moot**: 16b made the snapshots the one-card host layouts byte for byte, and 16d keeps that with MTP (the live verify slot per device in GDN order, the conv rings, the head's hidden row last; the KV per tensor with the head's layer after the last FA layer - `runtime::pp_kv_runs`, host-tested against Engine's order). An entry is the same bytes whichever wrote it, so the store keys stay 0 / 1 and nothing is refused (the store is per process anyway). `pp_serve_test` moves snapshots between a one-device and a two-device engine; `pp_mtp_test` continues one card's snapshot on two |
| Review Focus 1: the operator's choice - replicate the embedding (2.54 GB) or gather rows over P2P | the choice is still open; 16d builds **replicate**, as 16b's placement already anticipated (`place_stages` moved the head's weights and lm_head's companion there): the head's draft lists and its KV fill (verify and prefill) gather embedding rows of ids device 1 produced. The plan and the memory lines name it ("MTP: head 0.849 GB + embedding replica 2.543 GB in model"). A gather over P2P would save 2.54 GB on device 1 for one 10 KB peer read per row - worth it only if device 1's bytes ever bind (`--max-len auto` reaches 262144 either way on Qwen3.8, below) |
| "MTP head on device 1, verify across both with M = K + 1 rows handed off together" | as planned: `build_stage_verify` is `build_verify` cut at 16b's cut (layer s's fold on device 0, M rows + the norm sums through the landing buffer, which always held kM rows), and the head's KV fill appended on device 1; `build_stage_draft` is `build_draft` on device 1. **The per-row GDN slots need no new binary**: `gdn_step_slots` bakes one slot's stride as the whole model's (SPEC_SLOT_STRIDE), so each card's slots are an allocation of (kSlots - 2) whole slots + its own layers (`pp_gdn_spec_bytes`; device 1 uses its MtpBuffers' full slots) and binds its slice g as every stage buffer does. The gaps cost 0.151 GB a card on Qwen3.8 at the even split; a stage-stride variant would save them (one binary per split - a box decision, not built) |
| commit moves `pos` on both devices | and `gdn_live` (each card's verify slots hold its own layers, so the live index is one number for both); `verify` mirrors device 1's Control into device 0's as a step does; `pp_mtp_test` checks the two blocks equal after every call |
| (not in the plan) the head's KV during a two-card prefill | Engine::prefill fills the head's KV after each chunk (`step_mtp_kv`). On two cards it runs on device 1's list after the chunk's layers, in order: the chunk's ids copied into device 1's scratch (step_mtp_kv's embed reads them there), h_{pos-1} into the hidden rows' row 0, `step_mtp_kv` over a head Control block per chunk parity (16c's rule: the host writes chunk j + 1's while chunk j runs), the last row back into hh. A hooked chunk also shadows that row; `save_state` inside a mid-prompt hook reads it. One subtlety, named so it is not rediscovered: the head's KV row at `end - 1` is written by the NEXT chunk (it pairs h_{end-1} with x_end), which on two cards may already run while the hook copies `[0, end)`. That row is rewritten before anything reads it after a restore (the next prefill chunk or verify fills it first), so the bytes there are don't-care; `pp_mtp_test` blanks it in both records |
| Review Focus 4: 262144 - the RoPE table, the attention variants, the memory per device; passkey | decode attention v2 (the default) bakes no max_len; v1 under `--pp 2` takes its largest compiled length (`v1_compiled_at_most`, as one card). The memory per device is the planner's below; P3 is `r27.p3` (passkey at 5 / 50 / 95 % of 250000 ids at 262144, l0-int8), S3 `r27.s3` (both memory lines at 262144 with the bf16 head, and with `--mtp 3 --lm-head int8`) |
| Review Focus 5: `--pp 2` with the prefix cache and `--mtp` both on; one request end to end | `cli::pp::PipelineEngineAdapterT` is EngineAdapter's logic over the pipeline's host calls (`pending`, `set_token`, `set_draft_input`, `read_logits_into`, `read_draft_logits_into`, `host_rows` - pinned rows in the shared context); its host twin runs over a two-device fake through `PrefixSession` and the HTTP server (`pp_serve_test`); on the cards `pp_prefix_gpu_test` (C2's sequences, two cards bitwise one card, +- MTP) and `golden_server_test` with `B70_SERVE_ARGS='--pp 2 ...'` against one-card `b70-decode` (r27.golden). The prefix cache's block shadows (two per device, 0.167 GB a card on Qwen3.8 at split 32) are planned whenever the cache may be on (`auto` decides from host RAM only after the load) |
| Task 2's llama-benchy rows; Task 3, the record | stages (`r27.benchy`, opt-in; `serve_benchy.sh` gained `SERVE_AFFINITY`); the record (BENCHMARKS "Pipeline parallel (spec 16)", this spec's amendment with every gate's number, docs/10's two-card lock) is an edit after the box run |

**The planner's numbers** (derived, `pp_serve_plan_test`: b70-serve's terms - prefill l0-int8,
the prefix cache's shadows, 1.5 GB reserve a card; the dense MTP head at its checkpoint bytes):

| model, flags | `--pp 2` split, max_len | one card's auto | device 0 / device 1 |
|---|---|---|---|
| Qwen3.8, int8 head | 32, 262144 | 201216 | 18.741 / 17.513 GB |
| Qwen3.8, int8 head, `--mtp` | 36, 262144 | 169984 | 21.048 / 20.573 GB |
| Qwen3.8, int8 head, `--mtp`, `--kv-cache int8` | 36, 262144 | 262144 | 16.253 / 16.311 GB |
| Qwen3.8, bf16 head, `--mtp` | 38, 262144 | 151808 | 21.485 / 21.412 GB |
| Agnes, int8 head | 36, 262144 | 139520 | 21.313 / 20.085 GB |
| Agnes, int8 head, `--mtp` | 40, 262144 | 114176 | 23.733 / 23.138 GB |
| Ornith, int8 head | 20, 262144 | 262144 | 13.946 / 13.456 GB |

At 16384 with MTP, Qwen3.8 split 32: device 0 holds its verify slots (0.377 GB), device 1 the head
(0.849 GB), the replica (2.543 GB) and MtpBuffers (0.544 GB with the prefill rows).

**Gates as built:** host `pp_serve_plan_test`, `pp_serve_test`, `pipeline_args_test` (b70-serve's
refusals); on the cards `pp_mtp_test` (+ `_i8head`, `_kv8`: Review Focus 1-3, M2 across the
split, bitwise one card), `pp_prefix_gpu_test` (+ `_peer`, `_split`, `_mtp`, `_kv8`: P2's C2,
bitwise one card), `cli_reject_serve_pp*`, row 27's CLI, server, golden, Agnes / Ornith, S3 and
P3 stages; opt-in P3 with MTP and the llama-benchy rows.

**Not in 16d:** K2-Horizon and Kolibri-1 on two cards in the server (K2: its own engine, future
work; Kolibri: spec 20e), a stage-stride `gdn_step_slots` variant, the P2P embedding gather,
per-device step times in decode, more than two cards, concurrent requests (after spec 13), the
record (after row 27).
