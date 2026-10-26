# Spec 13 - batching: several sequences in one decode step

**Status:** design, 2026-09-28; **revised 2026-10-23 (§9, the operator's decisions; §10, the two-card
replica mode for later)** - §9 supersedes §2's "(decide)" on the layout, §3's "MTP: out of scope", and §7's
first line. Open decisions are marked **(decide)**.

**Order:** last of specs 10-13: it needs spec 10's decode attention (per-sequence depth
in the stride), and it is much cheaper in memory after spec 12 (int8 KV).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

`b70-serve` runs one request at a time (a bounded FIFO, `--queue 4`). opencode runs
subagents and side requests (titles, summaries) concurrently with the main thread;
today they wait in line, and each one also evicts the resident session (spec 7 then
restores it from host, ~0.35 s at 60k).

Decode is bandwidth-bound: one token reads ~14.3 GB of weights. A step with B rows
reads the weights once. Spec 8b measured the main list at M rows (consecutive positions)
at **1.17 / 1.52 / 1.74** plain steps for M = 2 / 3 / 4, so B sequences decoding
together would give, before attention (derived): **1.71x / 1.97x / 2.30x** aggregate
tokens per second. Attention does not share: each sequence reads its own KV.

## 2. The decision (proposed)

**Continuous batching of up to B sequences (B = 2..4) in the decode step, with
per-sequence state slots, prefill interleaved chunk by chunk with the batch's decode
steps.** One engine, one captured decode list per batch size, rows indexed by slot.

**(decide) B and the context per slot.** Memory at bf16 KV (derived; model 18.1 GB,
~2 GB of other buffers, 32.5 GB card):

| B | max_len per slot | KV | GDN + conv state | fits |
|---:|---:|---:|---:|---|
| 1 | 131072 | 8.6 GB | 0.17 GB | today |
| 2 | 65536 | 8.6 GB | 0.33 GB | yes |
| 4 | 32768 | 8.6 GB | 0.67 GB | yes |
| 2 | 131072 | 17.2 GB | 0.33 GB | no (int8 KV: 8.6 GB, yes) |

A mixed layout (one long slot for the main thread, small slots for subagents, e.g.
1 x 96k + 3 x 8k) matches opencode best but makes the KV addressing per slot.

## 3. Design

- **Slots:** `gdn_state`, `conv_ring`, KV and `Control` (pos, pending token) become
  per slot; each slot has a KV base and a max_len. The server maps a request to a free
  slot; spec 7's cache keys stay per request (the resident session is per slot).
- **Decode kernels:** the M loop today means consecutive positions of one sequence
  (MTP). Batching needs rows of different sequences: the GEMVs, norms, argmax and
  `lm_head` are row-agnostic already; `gdn_step` must index state by slot instead of
  carrying one state across rows; `attn_prep` / decode attention take each row's slot,
  pos and KV base (spec 10 v2's stride per row). A second compile-time mode
  (`ROWS=seqs` against `ROWS=positions`) or a per-row descriptor table.
- **Scheduling:** a step loop in the server: each tick runs one decode step over the
  active slots, and at most one prefill chunk of a waiting request between ticks (a
  chunk of 2048 costs ~1 s, so long prompts delay the batch's tokens: **(decide)** chunk
  size for interleaving, e.g. 512).
- **Sampling:** per row, host-side as today.
- **MTP:** out of scope at first (batch x (K+1) rows); `--mtp` and `--batch` are
  exclusive in the first version.

## 4. Correctness gates

- **B1:** a sequence decoded in a batch equals the same sequence decoded alone, bitwise
  where the row kernels are row-independent (GEMV rows are: the M = 1..4 verify rows are
  bitwise equal to M = 1, spec 8b), otherwise under the golden tie rule.
- **B2:** golden gates and A4 with 2 and 4 concurrent requests; replay determinism with
  a fixed arrival order.
- **B3:** spec 7's C2 per slot; a request finishing and a new one taking its slot.
- **B4, server:** concurrency tests (mock engine): fairness, cancellation, a stop in one
  row not affecting others, streaming per request.

## 5. Speed bars

- **Aggregate decode t/s** at B = 2 / 4, 4k depth each: >= 1.6x / 2.0x of one sequence.
- **Per-request latency:** a single request alone on the server within 2 % of today.
- **opencode-shaped replay** (spec 7's synthetic log with concurrent side requests):
  total wall time recorded against today's serial server.

## 6. Stages

- **13a, probe:** the batched GEMV and `gdn_step` at B rows of different sequences;
  measured step time at B = 2..4 (the table in §1 is derived from consecutive-position
  rows); the memory layout decision.
- **13b, engine:** slots, row descriptors, the batched list(s), B1.
- **13c, server:** the scheduler, slot mapping with spec 7, B2-B4, speed, record.

## 7. Out of scope

- MTP together with batching (a later spec).
- More than one GPU; B > 4.
- Prefix sharing between concurrent sequences (vLLM-style block tables).

## 8. Amendment - 2026-10-04: admission and measurement

- **§2 restated: never batch decode through the prefill path;** one captured decode list per
  batch size, and a single-request list for B = 1, so a lone request never pays for batching.
- **§3 scheduling, added: batched admission with a short burst hold.** Waiting prompts are admitted
  together in one prefill, with a hold of a few milliseconds to catch requests that arrive in a
  burst (opencode's subagents arrive like that). Trade-off against chunk interleaving: admission
  stalls the rows already decoding, so 13a measures both (one admission prefill for the burst
  against chunks interleaved with decode ticks) and the worst inter-token gap of a running request
  for each; **(decide)** after 13a.
- **For an MoE follow-on** (spec 15 batched): per-(token, expert) streaming at the decode kernels'
  bandwidth may beat a grouped small-M kernel up to a handful of rows; measured, not assumed. For
  dense Qwen3.8 the rows share weights and spec 8b's M-row step times apply as written.
- **§5, measurement:** llama-benchy's total tg at concurrency counts staggered admission inside the
  decode window as slow decode; every concurrency row states how admission was timed.

## 9. Revision - 2026-10-23: MTP in every slot, the layout, the models

What changed on main since §1-§8 were written: spec 12b's int8 KV cache, spec 14's Agnes, spec 19d's speculative
step under the scheduler (`server::BatchEngineIface::step_spec`, one proposer state per slot, a per-row K from
`server::AdaptiveK`, tested against the mock in `dflash_scheduler_test`), spec 16's pipeline parallel, and the MoE
models (specs 15, 18, 20-25). Plan 13c Task 1 (the scheduler against `MockBatchEngine`) is on main; 13a and 13b
are not built.

**Operator's decisions (2026-10-23):**

1. **MTP and batching together** (vLLM's shape: each sequence contributes K + 1 rows to one verify step). §3's
   "MTP: out of scope" and §7's "MTP together with batching (a later spec)" are withdrawn. The engine implements
   `step_spec` with the MTP head per slot; DFlash per slot (spec 19 §5 "Batching") follows on the same row table.
2. **Layout: one long slot and three short ones** on one card - the main session at 131072 and three subagent
   slots at 32768, **int8 KV** (spec 12b), MTP on in every slot. The KV form stays engine-wide (one form, so
   one `_KV8` kernel set); the addressing is per slot (a KV base and a max_len each).
3. **Models: Qwen3.8 and Agnes first** (the dense hybrids sharing the decode kernels; Agnes's 54 GDN + 18 FA
   layers from the descriptor, never a Qwen3.8 constant). The MoE models (Ornith, K2, Bolt) are
   per-model follow-ons: their rows route to different experts, so their cost model is the expert union of the M rows
   (spec 26's method; that spec was dropped 2026-10-27), not §1's dense M-row table.
4. **Concurrency on ONE card;** the second card stays free (pipeline parallel, long contexts, another model).
   The two-card replica mode is §10, written now and built later.

**The row table.** A decode step's rows are (slot, position) pairs: today's M rows of one sequence (MTP verify)
and §3's rows of different sequences are the same table, and so are their mixes (two slots x (K = 3 + 1) = 8
rows). Per slot: `Control` (pos, pending token, `gdn_live`), the GDN state and its speculative slots (spec 8's
`gdn_spec`, 4 per sequence today - the live one and 3), the conv ring, the main KV region, the MTP head's KV
layer and its `hh` row. `gdn_step` indexes state by the row's slot and speculative index; `attn_prep` /
decode attention take each row's slot, pos and KV base (v2's per-row stride).

**A row budget instead of an exclusion.** The decode kernels exist up to M = 8 (spec 19a's verify probe). The
scheduler hands out K under a cap of 8 rows a step (`AdaptiveK` per request, then clipped to the budget, the
longest-running main session first). Derived from spec 8b's verify costs (1.17 / 1.52 / 1.74 plain steps at
M = 2 / 3 / 4; M = 5..8 extrapolated at +0.22 a row, unmeasured - plan 19a Task 4, box Session 6b) and the
measured acceptance (K = 1: 1.92 ids, K = 3: 3.455):

| active slots | K each | rows | ids per plain-step cost (derived) |
|---|---|---:|---:|
| 1 | 3 | 4 | ~1.5x (today's `--mtp`) |
| 4 | 0 | 4 | ~2.3x |
| 4 | 1 | 8 | ~3.0x |

13a measures the M = 5..8 step and the mixed-row (different slots, different depths) step before 13b fixes the
budget rule.

**Memory (derived, Qwen3.8, one card: 32.53 GB, `--mem-reserve-gb` 1.5, model 18.1 GB, prefill scratch 0.7 GB;
int8 KV 33,024 B a position, x 17 / 16 with the MTP head's layer; ~0.62 GB of GDN state + conv ring a slot with
4 GDN state slots):**

| layout, int8 KV, MTP on | KV | state | fits |
|---|---:|---:|---|
| 1 x 131072 (today) | 4.6 GB | 0.6 GB | yes |
| **1 x 131072 + 3 x 32768** (decided) | 8.0 GB | 2.5 GB | yes, ~1 GB to spare - 13a confirms against `runtime::plan` |
| 2 x 131072 | 9.2 GB | 1.2 GB | yes |
| 4 x 65536 | 9.2 GB | 2.5 GB | marginal |

At bf16 KV the decided layout does not fit; `--batch` with bf16 KV shrinks the short slots (the planner's
auto max_len per slot), it never refuses the main slot's length silently. Agnes's numbers are its descriptor's
(18 FA layers, 54 GDN) and 13a prints them.

**Prefill while the batch decodes.** An opencode subagent's first prompt is 10-30k ids; a 2048-row chunk costs
~1 s (measured, spec 2), which stalls every decoding row. Two levers, both measured in 13a: the interleave chunk
(512 / 1024 / 2048) against the worst inter-token gap of the main session, and the prefix cache (spec 7) - the
subagents share opencode's system prompt, so a slot restores the shared prefix from host instead of prefilling it.
**(decide) after 13a:** the chunk size and whether a long admission waits for a quiet tick.

**Gates added to §4:** B1 at mixed K (a slot verifying K = 3 beside a slot at K = 0 equals each alone); B2 with
`--mtp` on (greedy MTP output identical to plain, per slot, at 4 concurrent requests); B3 for the speculative
state (a slot's rejected drafts leave every other slot's `gdn_spec` and head KV byte-identical).

**Plans.** 13a, 13b and 13c's remaining tasks are refreshed against this section before 13a starts (one
implementing agent at a time).

### 9.1 Addendum - 2026-10-23: the memory recomputed, Agnes's layout, the follow-on order

Recomputed from the code's own formulas (buffer_sizes.cc / buffer_sizes.h KvLayout / MtpDims, memory_plan.cc's
terms; the method reproduces spec 6's measured decode-state and prefill lines byte for byte). Budget 32.53 - 1.5
reserve = 31.03 GB (the reserve itself is unconfirmed on the box, queue row 7). Weights in the SERVING form: int8
head + the MTP head (0.849 GB). All derived; nothing batched has run.

| per slot, int8 KV, MTP on | Qwen3.8 | Agnes |
|---|---:|---:|
| KV a position (with the head's layer) | 35,088 B | 39,216 B |
| state a slot (4 GDN states + conv ring) | 0.620 GB | 0.697 GB |
| shared: weights + MTP head + RoPE at 128k | 17.70 GB | 20.52 GB |
| shared: decode scratch, MTP rows, prefill, int8 | 0.87 GB | 0.93 GB |

- **Qwen3.8, 1 x 131072 + 3 x 32768: total 29.10 GB, 1.93 GB spare** (the table above §9 said ~1 GB: it counted the
  bf16 head without the MTP head). 4 x 65536 leaves 0.80 GB.
- **Agnes, the same layout: 33.24 GB - 2.21 GB over.** **Operator's decision (2026-10-23, replacing the same day's
  1 x 98304 + 3 x 32768): Agnes batches at concurrency 2 on one card - 1 x 131072 + 1 x 32768, MTP in both slots -
  29.27 GB, 1.76 GB spare (derived).** The main session keeps its full 128k. Considered: 1 x 131072 + 2 x 32768 with MTP
  off in the short slots (30.10 GB, 0.93 GB spare), 1 x 98304 + 3 x 32768 (the shorter main slot). Qwen3.8 keeps
  1 x 131072 + 3 x 32768 with MTP in every slot. Short slots below 32k are refused as a layout (an opencode subagent's
  first prompt is 10-30k ids). Agnes at concurrency 4 is the pipeline follow-on below. Rejected: the embedding in host
  USM and every other host-RAM offload for batching (see the server option below), and a shared pool of speculative
  GDN states.
- **What this asks of 13b:** the slot count and lengths PER MODEL (Qwen3.8 four slots, Agnes two); a max_len PER SLOT (`BatchEngineIface::max_len()` is one value today,
  src/server/batch_engine.h:27); MTP on or off PER SLOT (a slot without speculative GDN states, `MtpDims` refuses
  fewer than 4 today, buffer_sizes.cc:273); the layout from the planner per model (`runtime::plan` grows a slot list),
  the extra captured lists (per batch size and per M) counted in it.
- **The dense gains are an upper bound at depth.** spec 8b's M-row costs are rows of ONE sequence, whose K/V wave
  attn_v2 shares across the rows (attn_v2.cl:35-36); rows of different slots each read their own KV (~1.08 GB a row at
  32k on Qwen3.8, derived). 13a measures the mixed-depth step.

**Follow-ons (operator, 2026-10-23), in order, each its own plan after 13b-13c** (Muse-Glimmer and LFM2.5 were
follow-ons 2 and 3 until their support was dropped, operator 2026-10-27):
1. **Ornith** (MoE, 256 experts top-8): shares Qwen's decode list and already has M = 2..4 MoE lists (spec 15);
   ~1.6x / 2.2x at B = 2 / 4 by bytes (estimated; rows of different sequences route nearly independently, so the gain is
   the shared non-expert weights, not an expert union).
2. **Batching over the pipeline (operator, 2026-10-23):** Agnes (and Qwen3.8 if it pays) at concurrency 4 on both cards
   under `--pp 2` - each card holds half the layers (~10 GB of Agnes's weights), so four 128k slots fit with room, and
   two groups of rows alternate between the stages so both cards work (the pipeline's only decode gain: at batch 1
   one card waits for the other). The row table in both stages of `runtime::PipelineEngine`, the group scheduling, the
   handoff per group. Begins only after spec 16's rows 22, 23 and 27 pass on the box and 13b-13c are done; per model
   and opt-in, since it takes the second card that §9 decision 4 otherwise keeps free.
Not in spec 13: Bolt and Kolibri (two cards only - revisit beside the pipeline), K2-Horizon (one card holds ~50k
positions in all), Qwen3.8-Flash-Next (offload-bound).

### 9.2 The server option (operator, 2026-10-23)

Batching is a `b70-serve` option, **off by default**: `--batch N` (N = 1, the default, is today's server - one request
decodes at a time, the rest queue, and every output stays bitwise today's). `--batch 2..4` turns the slots on, with the
layout the planner picks for the model (§9.1: Qwen3.8 four slots, Agnes two; a model without a batched engine refuses
`--batch` > 1 by name). **Batching never offloads to host RAM:** every slot's KV and state, the weights and the
embedding stay on the card; when a layout does not fit, the planner shortens the short slots or refuses, it does not
spill. (Host-RAM expert offload stays spec 22's, for Qwen3.8-Flash-Next alone.) `--batch` > 1 with `--pp 2` is §9.1
follow-on 2; with the two-card replica mode, §10.

**MTP trades against slots (operator, 2026-10-23).** With `--mtp off` a slot needs one GDN state instead of four, no
head KV layer, and the card drops the MTP head (0.849 GB), so a model fits more slots. Agnes on one card, int8 KV
(derived, the §9.1 method):

| Agnes, one card | total | spare | |
|---|---:|---:|---|
| `--batch 2` (128k + 32k), MTP on | 29.27 GB | 1.76 GB | the default batched layout |
| `--batch 4` (128k + 3 x 32k), `--mtp off` | ~29.9 GB | ~1.1 GB | allowed |
| `--batch 4`, MTP on in the main slot only | 31.50 GB | -0.47 GB | does not fit |
| `--batch 4`, MTP on everywhere | 33.24 GB | -2.21 GB | does not fit |

A `--batch` that does not fit is refused at startup with the planner's numbers and the ways out, never silently
shrunk below the slot lengths above. The message is built from `runtime::plan` (the model, the card's bytes, what the
asked layout needs, the largest batch that fits with MTP on), so it holds for every model and card; for Agnes:

```
b70-serve: --batch 4 with MTP does not fit Agnes on one card: 128k + 3 x 32k needs 33.24 GB, the card has 31.03 GB
(32.53 GB - 1.5 GB reserve). With MTP this card fits --batch 2. For --batch 4, either:
  - add the second card and run pipeline parallel: --pp 2 (follow-on 2; both cards), or
  - turn MTP off: --mtp off (128k + 3 x 32k fits, ~1.1 GB spare).
```

`--mtp auto` with `--batch` asks for the largest batch that fits WITH MTP and says so in the startup line; an explicit
`--batch` beyond it, with MTP on, is the refusal above. Qwen3.8 fits `--batch 4` with MTP (§9.1), so it never prints it.

## 10. Later: the two-card replica mode (written 2026-10-23, not built)

**The idea:** card 0 serves the main session with today's single-sequence engine (MTP, the whole card for
context), card 1 a batch of the subagents (§9's batched engine, or simply a second single-sequence engine).
No new kernels for the routing half: two engines, one per card, behind one `b70-serve`.

- **Routing:** the scheduler places a request by its conversation - the long-running session (the one whose
  prefix the cache holds) on card 0, the others on card 1. A request never moves between cards mid-generation;
  a prefix-cache entry is restorable on either card (the host-side layout is the same).
- **Memory:** each card loads the model (18.1 GB each for Qwen3.8); card 0 keeps today's full max_len, card 1
  splits its KV among its slots.
- **What it excludes:** pipeline parallel (spec 16) and a second model on card 1 while it runs; `--replica` and
  `--pp 2` are exclusive.
- **Gates:** B2's concurrency on two cards (each request's ids equal to running alone on its card); the prefix
  cache's C2 across cards; the device-1 prefill gap (3.4 % slower, measured) recorded, not hidden.
- **When:** after 13b-13c, on the operator's go.

