# Spec 19 v3 - DFlash in the modes we serve: the int8 KV cache, request slots, tensor parallel

**Status:** 2026-11-13: mode A implemented on `spec19v3a-dflash-int8-kv`, focused check pending; modes B/C remain design. Every speed number for a new mode is **derived** from numbers
already measured; each plan starts by measuring the one number its mode hinges on.

**Continues** spec 19 v2 (`2026-11-11-spec19v2-dflash-v2-design.md`), whose §4 parked these modes. Spec 19 v2 stays
the record of the one-card drafter; this spec adds modes and changes no draft arithmetic.
- **Plans:** `docs/superpowers/plans/2026-11-13-spec19v3a-dflash-int8-kv.md` (mode A),
  `2026-11-13-spec19v3b-dflash-slots.md` (mode B), `2026-11-13-spec19v3c-dflash-tensor-parallel.md` (mode C).
- **Open-work:** item 12.
- **Evidence:** `docs/BENCHMARKS.md` ("DFlash (spec 19)", "Tensor parallel (spec 17)"),
  `docs/probes/dflash-vs-sly-2026-11-11.md`, `docs/probes/probe-verify-attn-2026-11-03.md`,
  `docs/probes/probe-batching-2026-10-23.md`.

**Grades:** **measured** (with its source), **derived** (arithmetic on measured numbers), **estimated** (a guess with
its reason).

## 0. The short answer

`--spec dflash` is speculative decoding with a block drafter: a small model drafts 7 tokens in one pass, the target
checks 8 rows (the pending token and the 7 drafts) in one step, and the output is the target's own. Today it runs on
one card with the bf16 KV cache (the KV cache is the stored attention keys and values of the context), on the
two-card layer split (`--parallel pp`), and - being wired on branch `dflash-replica` - with one model copy per card.

| mode | what blocks it | derived speed against MTP in the same mode | worth building? |
|---|---|---|---|
| **A. `--kv-cache int8`** | three kernel builds and one table row set are missing; the source already supports them | **~1.15-1.18x** (one card's 1.19x less the int8 attention's cost) | **yes, first.** Small, and it lifts DFlash's context on one card from ~140k to ~245k tokens |
| **B. `--max-num-seqs` > 1** | the engine refuses a drafter beside slots; a drafter has one ring, one tap buffer, lists bound to one slot | one request active: **1.19x** (today's path); two: **~1.0-1.1x**; four: **~0.95x** | **yes, second, if its first task's arithmetic holds.** The gain is the lone request; it costs context |
| **C. `--parallel tp`** | no per-card kernels above 4 rows; the drafter and its head have no place in the split | **~0.93x** with the drafter whole on one card (range 0.9-1.04x) | **not as designed today.** Measure verify at 8 rows under TP first; build only if that surprises, or after the draft pass is cut (item 12) |

One plain statement the table does not show: **for one session on two cards, tensor parallel with MTP (123.1 t/s
measured) stays the fastest configuration.** One-card DFlash is 88.8 t/s on the same request shape. Mode C is the
only one that could change that, and it is derived to land at ~114 t/s.

## 1. Where we start

### 1.1 What is measured (one card, bf16 KV, int8 drafter, candidate head)

| quantity | value | source |
|---|---:|---|
| plain step at depth 4096 | 29.0-29.2 ms | BENCHMARKS "V2 baseline" |
| verify at 8 rows | 1.245 plain steps (4096), 1.272 (32768) | the same |
| draft pass, K = 7 | 8.43 ms = 0.287 plain steps | BENCHMARKS "Task 7" |
| one round (projection + draft + verify + accept) | 1.559 plain steps = 45.5 ms | the same |
| serving bench (3,425-token prompt, 512 tokens) | DFlash 88.8 t/s, MTP 73.2 t/s | BENCHMARKS "V2 baseline", README |
| tokens a round on that bench | ~4.04 (**derived**: 88.8 t/s x 45.5 ms) | - |
| the opencode replay (78 requests) | DFlash 1.193x MTP | BENCHMARKS "Task 8" |
| prefill, 4096-token prompt | DFlash 0.955x MTP (the open failure of spec 19 v2) | the same |

### 1.2 What a mode has to carry

Every mode needs the same five things the one-card engine has (`runtime/engine.h`, "spec 19c"):

1. **Verify lists at 1..8 rows** with 8 state slots for the delta-net layers (GDN layers keep a recurrent state per
   sequence; a verify row writes the state after it into its own slot, so accepting j drafts is choosing slot j).
2. **Taps:** the residual stream after 5 target layers, copied for every verify row into the drafter's tap buffer
   (`runtime::TapVerify`, bf16 `[8 rows][5 layers][5120]`).
3. **The projection:** after the host accepts j drafts, rows 0..j of the taps go through the drafter's `fc` and into
   its context ring (2048 positions of K and V per drafter layer). Spec 19 v2 defers it to the front of the next
   submit.
4. **The draft list:** one non-causal pass of 8 rows over the ring, ending in the target's head and the selector. It
   writes the drafts into `Control.cur_token[1..K]`, where the verify list reads them.
5. **One submit a round:** `Engine::verify(k, draft)` executes [pending projection, draft, verify] as one queue call.

Plus the session's edges: prefill fills the ring from the prompt's last 2048 positions, and a prefix cache snapshot
carries the ring (41.9 MB).

## 2. Mode A - the int8 KV cache

### 2.1 What blocks it

`cli::dflash::check_engine` (`src/cli/dflash_args.h`) throws for `--kv-cache int8`. The reason behind the message is
three missing binaries and one missing table section - not a missing kernel source.

| piece | what exists | what is missing |
|---|---|---|
| the int8 KV writer `attn_prep_kv8` (`src/kernels/kv8.cl`, `KV8_PREP`) | `attn_prep_M<M>_KV8` at M = 1..4. The source has no row limit: its row-table build `attn_prep_M<M>_KV8_ROWS` is built at M = 2..8 for the slots engine | `attn_prep_M<M>_KV8` at M = 5..8 |
| the int8 reader and its reduce (`kv8.cl`, `KV8_DEC`) | `attn_v2_M<M>_T32_KV8` at M = 1..4. The source stops the single-sequence build with `#error "kv8 dec: M is 1..4"`; under `BATCH_ROWS` it allows 1..8 and that build runs in production (slots) | the bound lifted to 8 and `attn_v2_M<M>_T32_KV8` at M = 5..8 (the verify binds its `attn_reduce_v2_kv8`) |
| the verify attention on the matrix engines (`src/kernels/attn_verify_dpas.cl`; DPAS is the card's integer dot-product instruction) | bf16 at M = 2..8; int8 (`KV8=1`) at M = 2..4. The source accepts `KV8=1` at M = 2..8 | `attn_verify_dpas_M<M>_T32_KV8` at M = 5..8 |
| the rule that picks the attention (`src/runtime/verify_attn.h`, `kVerifyAttnTable`) | rows for bf16 M = 2..8 and int8 M = 2..4. A missing row means "bind v2" - whose int8 binary does not exist at M = 5..8, so capture fails | rows `{5..8, int8}` |
| `runtime::TapVerify` lists (`build_verify_taps`, `build_stage_verify_taps`) | take the KV form from the buffers (`kv8_` in `capture.cc`); nothing in them is bf16-only | nothing |
| the drafter (ring, taps, draft list) | never reads the target's KV cache | nothing |

**The int8 attention at 5..8 rows is already measured once.** `probe-verify-attn-2026-11-03.md` §2 timed
`attn_verify_dpas` over int8 KV at M = 5..8 (busy box, before the kernel carried P as two halves, `PSPLIT=1`):

| per attention layer at M = 8, us | 512 keys | 1448 | 4096 | 11585 | 32768 | 130816 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 KV (bound today) | 48.7 | 79.2 | 107.5 | 232.8 | 460.1 | 1533 |
| int8 KV | 61.3 | 111.9 | 151.6 | 293.5 | 688.9 | 2605 |
| int8 costs more, x 16 layers, ms a verify | 0.2 | 0.5 | 0.7 | 1.0 | 3.7 | 17.1 |

The operator's decision 4 of that probe was "no binary" for int8 at M = 5..8 because DFlash did not serve int8.

### 2.2 The design

- **A1. Build the three binaries at M = 5..8 and add the table rows with `dpas_from = 0`** (DPAS at every depth), the
  bf16 rows' own form at M = 6..8.
  - *Rejected: a second list by depth with the v2 reader at shallow depth.* The probe shows v2 ahead only between
    ~1k and ~3k keys at M = 5..7, by at most 27 us a layer (0.4 ms a verify, ~1 % of a round). It would need the
    single-sequence int8 reader validated at 5..8 rows and a second captured list per M.
- **A2. Lift `kv8.cl`'s single-sequence bound to 8** so the M = 5..8 binary exists for its reduce kernel; the decode
  kernel in it is bound only by tests that force v2.
  - *Rejected: a reduce-only binary.* A new build form for one use.
- **A3. The int8 accuracy bar is the one already ruled** (operator, 2026-11-03): no further from the fp64 reference
  than the prefill's int8 flash attention on the same row. The probe's worst int8 row (0.99998797 at 4096 keys,
  query head 19, row 0) is the same value at every M, so the ruling's row covers M = 5..8; the plan's test states it.
- **A4. Nothing else changes:** the cost table of the K policy (`MtpCost::dflash`) keeps its rows unless the measured
  int8 verify at M = 5..8 differs from bf16 by more than 5 % at depth 4096.

### 2.3 Memory and `--max-len auto` (one 32.5 GB card, derived)

Per position: bf16 KV 65,536 B; int8 KV 33,024 B plus the int8 prefill's widening rows 4,096 B. The drafter's fixed
terms: weights 2.05 GB (int8), the candidate head 0.68 GB (an int4 copy of the head that picks 64 candidates a row,
re-ranked exactly; on whenever the head is int8), 7 more GDN slots 1.06 GB, buffers and prefill tap staging 0.17 GB.

| one card | MTP (today) | DFlash, bf16 KV (today) | DFlash, int8 KV (this mode) |
|---|---:|---:|---:|
| `--max-len auto` | 171,008 bf16 / 262,144 int8 | ~130-140k | **~225-245k** |

- **It does not reach the full 262,144 on one card.** It is short by ~0.7-1.4 GB. The tuned int4 drafter (open-work,
  "DFlash drafter quant") would free 0.84 GB.
- On `--parallel pp` with int8 KV the full context fits (each card holds half the target).
- The planner prints the real figure; the plan records it and gates on >= 200,000.

**As built (2026-11-13):** mode A keeps the int8 drafter, candidate64/exact-int8 head and bf16 drafter ring.
The host planner's printed int8 context is recorded after the focused check; allocation at that context and the
full checkpoint gates remain pending.

### 2.4 Expected speed (derived)

- At ~4k keys a round grows by 0.7 ms on 45.5 ms: -1.5 %. MTP's own int8 loss there is ~1 % (71.7 against 72.3 t/s,
  probe-batching "Engine"). **DFlash / MTP ~1.18x** on the replay.
- At ~32k keys: +3.7 ms on 51.4 ms, -7 %; MTP's verify at 4 rows pays ~1.4 ms on ~40 ms, -3.5 %. **~1.15x.**
- Past 100k keys the 8-row int8 attention is ~42 ms a verify against MTP's ~25 ms at 4 rows; DFlash still commits
  more tokens a round, but the ratio there is **estimated**, not derived. The plan measures 120,000 keys.
- **Least sure:** the int8 numbers predate `PSPLIT=1` and ran on a busy box.

**As built:** no new int8-KV DFlash speed measurements yet. The derived expectations above are unchanged;
paired costs, acceptance, serving/f32, PP and replay gates are pending under the operator's quick-check-only scope.

## 3. Mode B - several request slots on one card

### 3.1 What blocks it

`b70_serve.cc` refuses `--spec dflash` when a card gets more than one slot, and `runtime::Engine`'s constructor throws
"a tap drafter beside sequence slots is not built". Behind that:

- **How the slots engine steps** (`cli::SlotsEngine`, `Engine::verify_rows`, spec 13c):
  - **One request active:** the slot's own lists and its own `EngineAdapter` - bitwise the one-slot server.
  - **Several active:** one row list of up to 8 rows (`Engine::kMaxRows`), each row carrying its slot's position, KV
    base and GDN state indices in a row table (`runtime::BatchRowTable`). The 8 rows are the whole budget: with two
    requests each gets 3 drafts, with four each gets 1. Row lists bind the v2 attention only, never DPAS.
  - MTP drafts for several slots in one pass (`Engine::draft_rows`).
- **What a drafter has one of:**
  - one context ring and one tap buffer (`DFlashBuffers::ring_k / ring_v / taps`), one control block and readback;
  - a draft list bound to slot 0's `Control` and a projection list bound to that one ring;
  - `Engine::df_slots_` (the 7 extra GDN states) for slot 0 only (`Seq::spec`: "the drafter's (slot 0 only)");
  - `build_verify_taps` takes no slot binding (`SlotBind`), `build_rows` copies no taps;
  - the draft kernels assume one block: one anchor, one position, one ring (`df_embed`, the conv prepare,
    `df_qkv_prep`, `df_attn_part`, `df_select`), and `df_ctx_store` writes one ring.
- **The planner:** `runtime::plan_slots` and `cli::settle_slots` take no `DrafterPlan`.

### 3.2 The design

- **B1. Per slot: its own drafter session** - ring, tap buffer, control block, readback - and its own copy of the
  draft and projection lists bound to them and to the slot's `Control`, exactly as the engine keeps its own copy of
  every list per slot. Weights and scratch are shared (one list runs at a time on one queue).
  - So a lone request stays bitwise the one-slot DFlash server, and a slot's snapshot carries its own ring.
- **B2. GDN states: slot 0 holds 8 (K <= 7), every short slot holds 4 (K <= 3).** Slot 0 is the main session by the
  product's shape; a short slot's lone step runs DFlash at K <= 3.
  - *Rejected: 8 states in every slot.* It costs 0.60 GB more per short slot - at four slots 1.8 GB, ~27k tokens of
    slot 0's context in bf16 - for K above 3 on subagent slots only when they run alone.
- **B3. Several requests active: one batched draft pass.** The blocks of all active slots share the drafter's 8 rows
  (their rows sum to the step's 8-row budget), so the weights and the head are read once whatever the slot count.
  - The per-block kernels get a row-table build (as spec 13c gave the target's `gdn_step` and attention): each row
    names its block, position and ring.
  - The row list copies the taps of all rows into one rows tap buffer; a rows projection stores each committed row
    into its own slot's ring.
  - One submit a step: [pending projections, batched draft, row verify].
  - *Rejected: one draft pass per slot in turn.* 0.29 plain steps each: at four requests 1.16 plain steps of
    drafting a step, derived 0.72x MTP.
  - *Rejected: DFlash for the lone step and the MTP head for batched steps.* Two proposers: the head's weights, its
    KV layer and its hidden rows come back, and the rings still need the taps of every batched row.
- **B4. Policy stays the scheduler's:** the row budget and who gets drafts are spec 13c's; the K policy reads a
  drafter cost row for batched steps measured in the plan.
- **B5. On two cards** (`--max-num-seqs` up to 8, `--parallel replica`) each card runs this engine with its own
  drafter - branch `dflash-replica`'s shape, unchanged.

### 3.3 Memory (one card, derived)

Budget for positions, extra GDN states and rings once the drafter's fixed 2.9 GB is taken: ~10.3 GB. One GDN state
is 151 MB; a slot's ring 42 MB; a short slot's KV at 32,768 positions 2.17 GB (bf16) or 1.11 GB (int8).

| `--max-len auto`: slot 0 + short slots | MTP (today) | DFlash, B2's states | DFlash, 8 states everywhere |
|---|---:|---:|---:|
| 4 slots, bf16 | 46,080 + 3 x 32,768 | ~12k + 3 x 32,768 | does not fit |
| 4 slots, int8 | 167,936 + 3 x 32,768 | ~105k + 3 x 32,768 | ~57k + 3 x 32,768 |
| 2 slots, bf16 | - | ~97k + 32,768 | ~87k + 32,768 |
| 2 slots, int8 | - | ~198k + 32,768 | - |

**DFlash with four slots is an int8-KV configuration**: mode A comes first. In bf16 slot 0 keeps ~12k tokens.

### 3.4 Expected speed (derived)

Costs in plain steps: a row step of 8 rows from 4 slots is 1.53 (47.8 against 31.3 ms, measured, probe-batching);
the batched draft 0.29 (one card's draft pass: the same weights, head and rows); MTP's row draft ~0.15 a draft index.

| requests active | drafts each | DFlash step | MTP step | DFlash / MTP at equal tokens a draft |
|---|---:|---:|---:|---:|
| 1 | 7 / 3 | today's one-card round | today's | **1.19x** (measured on one slot) |
| 2 | 3 | ~1.4 + 0.29 | ~1.4 + 0.45 + head KV | **~1.0-1.1x** |
| 4 | 1 | 1.53 + 0.29 | 1.53 + 0.15 + head KV | **~0.95x** |

- "At equal tokens a draft" is the unknown: how many tokens DFlash commits at K = 1 and K = 3 against MTP on the
  same text. The numbers are on the box already (the K = 1..7 acceptance record of spec 19 v2 Task 1; MTP's
  acceptance in the server). **The plan's first task is that arithmetic, on the host, with a stop condition.**
- So the mode's worth is the lone request in a slots server: a coding agent's main session is alone most of the time.
- **Least sure:** the batched draft at mixed blocks keeping one card's acceptance - the drafter was trained on blocks
  of 8 and already runs short blocks by masking, but K <= 3 blocks are its weaker range.

## 4. Mode C - tensor parallel

### 4.1 What blocks it

`cli::check_tp` (`src/cli/pipeline_args.h`) refuses it, and `runtime::TpEngine`'s header says "no DFlash drafter (not
built)". Tensor parallel (TP) splits every layer over both cards; after each row-parallel linear a card writes its
half into the other card's landing buffer over peer-to-peer (P2P) and both fold the two halves - the all-reduce.

- **Verify lists** (`build_tp_verify` in `capture.cc`, `TpEngine::capture_mtp_lists` in `src/runtime/tp_engine.cc`):
  each card captures its own list at M = 1..4 over `MtpBuffers` (4 GDN slots at the card's half shapes), the pair is
  submitted back to back (`TpEngine::run_lists`), and the host merges the two vocabulary halves' argmax per row
  (`merge_heads`).
  - The crossing itself is ready for 8 rows: `tp.cl`'s `TP_MROWS` and `tp::kMRows` are 8, the landing buffers are
    sized for 8.
  - **Missing at M = 5..8, per card:** `gdn_step_slots_M<M>_GK8V24` with 8 slots, `prep_gated_head`,
    `prep_silu_mul_M<M>_I8704`, `attn_prep` / `attn_v2` / `attn_verify_dpas` at 12 q / 2 kv heads (the DPAS build at
    these heads uses the 2..4-row levers, which do not fit 6+ row blocks), `tp_cross`, `tp_argmax` for both halves,
    and the per-card small-M GEMM rows (`kSmallMTuning`'s spec 17d rows are M = 2..4).
  - `build_tp_verify` needs `MtpBuffers`; there is no tap form of it.
- **Where the taps live:** after every fold both cards hold the same residual, bitwise (TpEngine's property T1). So
  the 5 tapped layers' outputs exist whole, at hidden 5120, on **each** card. No crossing is needed for taps.
- **Which card holds the drafter and the head:** nothing today. Each card has the whole embedding but only its
  vocabulary half of the int8 head (124,160 rows). The draft list needs every row: the candidate head scores the
  full vocabulary and re-ranks against exact int8 rows.
- **The single submit:** `Engine::verify(k, draft)` runs draft then verify on one queue because the drafts land in the
  `Control` the verify reads. Under TP there are two queues and two `Control` blocks; the other card's verify must
  not start before the drafts are in its `Control`.
- **Prefill and snapshots:** the TP prefill (`runtime/prefill/tp_prefill.cc`) stages no taps; `TpEngine::save_state`
  writes the one-card layout without a ring.

### 4.2 The design

- **C1. The drafter is whole on card 1**, with the int4 candidate head and card 0's half of the int8 head beside it
  (+0.64 GB), so the draft list is one card's list unchanged.
  - *Rejected: the drafter split over both cards like the target.* It would halve the draft pass (the weights read
    is what costs), but needs every drafter linear in halves, crossings inside the draft list and the head merge
    on the device - a second kernel set. It is the follow-on if C3's measurement says the mode is close.
  - *Rejected: a drafter copy on each card, sharing rows.* Each card would still read all the weights: no gain.
- **C2. Taps are local copies on card 1** from its own residual (`build_tp_verify_taps`: `build_tp`'s walk at M rows,
  8 GDN slots at the card's shapes, taps on rank 1 only). Card 0's list is the same walk without taps.
- **C3. A round is two submits:**
  1. card 1: [pending projection, draft]; wait for its fence;
  2. the host mirrors `cur_token[1..K]` into card 0's `Control` (the same host step MTP's draft merge does);
  3. both cards' verify lists, the two fences, the per-row merge, the host's accept, the commit on both cards.
  - Cost of the extra submit: ~0.2-0.4 ms on a ~35 ms round (an empty submit and fence is 0.02 ms measured; the
    rest is the host's wake-up).
  - *Rejected: mirroring the drafts on the device* (a P2P write into card 0 and a wait kernel at the front of its
    verify list): a new protocol for under 1 %.
  - Card 0 is idle during the draft. That is the price of C1.
- **C4. Prefill:** card 1 stages the taps of the prompt's last 2048 positions in the wavefront and projects them 8
  rows a replay, as one card does. Snapshots keep the one-card host layout and append card 1's ring; the prefix
  cache key takes the drafter's key.
- **C5. The K policy** reads a TP drafter cost table measured in the plan. Sampled requests gather 8 full rows from
  the two halves (`TpEngine::read_logits_into`, today 4).
- **`--spec dflash-lookup` under TP stays refused** (it is built for one card's adapter).

### 4.3 Memory (per card, derived)

TP holds the full 262,144 context in bf16 today at ~21 of 31 GB usable a card. Without the MTP head each card frees
its head weights and the head's KV layer (1.07 GB at 262,144). Card 1 adds the drafter 2.05 GB, the candidate head
0.68 GB, the other head half 0.64 GB, tap staging and buffers 0.17 GB; each card's 8 GDN slots at half shapes cost
0.30 GB more than MTP's 4. **Card 1 ~24 GB, card 0 ~20 GB: `--max-len auto` stays 262,144.**

### 4.4 Expected speed (derived)

| quantity | value | grade |
|---|---:|---|
| TP plain step with the head, depth 4096 | 18.9 ms | measured (BENCHMARKS, spec 17d) |
| TP verify at 2 / 3 / 4 rows | x 1.09 / 1.13 / 1.20 | measured |
| TP verify at 8 rows | ~x 1.38 = ~26 ms (range 1.3-1.5) | **derived**: +0.045 a row past 4 |
| draft pass on card 1 | 8.43 ms = 0.45 TP plain steps | measured on one card; TP does not shrink it |
| projection, host, the extra submit | ~1.0 ms | derived |
| one round | ~35.4 ms | derived |
| serving bench, ~4.04 tokens a round | **~114 t/s against MTP's 123.1: 0.93x** | derived |

- MTP gained 1.68x from TP on this bench (73.2 -> 123.1). DFlash's round gains only 1.29x (45.5 -> 35.4 ms),
  because the draft pass is 24 % of a TP round and stays on one card.
- On the opencode replay the same arithmetic gives 0.91-1.04x.
- **What would make it win:** verify at 8 rows under TP at or below 1.16 plain steps (22 ms) - unlikely; or the draft
  pass at ~4.5 ms, spec 19 v2's open target (then ~128 t/s, 1.04x); or both that and the split drafter (~136 t/s,
  1.10x).
- **Prefill gets worse in percent:** the tail projection's ~120 ms a prompt (spec 19 v2's open failure) is 4.5 % of
  one card's 4096-token prefill and ~8 % of TP's shorter one.
- **Least sure:** verify at 8 rows under TP. It is one measurement, it needs both cards, and it is the plan's first
  task with a stop condition.

## 5. Recommended order

1. **Mode A (plan 19v3a).** Smallest (the source exists), needs one card, nearly doubles DFlash's context on one
   card, and mode B needs it to keep a usable slot 0.
2. **Mode C's first task only (plan 19v3c Task 1):** build the per-card binaries at 5..8 rows and measure verify at
   8 rows under TP. It is the configuration the operator serves, so the number is worth knowing early; it needs
   both cards, so it runs before or after the weekend's Flash-Next quantisation, not beside it.
3. **Mode B (plan 19v3b),** if its host-only first task projects >= 0.95x at two requests and >= 0.90x at four.
4. **Mode C's build** only if step 2 derives >= 1.05x MTP under TP, or after item 12 brings the draft pass down.

## 6. Gates (every mode)

- **Same output.** Greedy output identical to `--spec off` in the same mode. Where the mode's own verify rows are not
  bitwise its plain steps (DPAS attention, the small-M GEMM, TP's folded crossing, row lists), each first difference
  is judged by the f32 judge - a CPU reference in f32 that says whether the chosen token is the true top or a near
  tie (`tools/box_validate/serve_judge.py`, `tests/server/spec_judge.h`). No failure, no skipped request.
- **Nothing else moves.** Without `--spec dflash` in the mode, every list binds the binaries it bound before
  (`tools/kernel_cmdlines`: additions only; the G0 suite green).
- **Acceptance not lower than one card's.** Tokens a round on the tool-call set and golden code, generation stopped
  at EOS, within 0.15 of one card's bf16 run (6.65 and 6.08 measured, BENCHMARKS "V2 baseline"). A larger gap is a
  defect in the taps or the ring to root-cause, never a bar to widen.
- **Snapshots.** A turn served as a prefix cache hit drafts as the unbroken session does (the ring in the snapshot).
- **Speed, once at the end,** on an idle card (both cards for mode C), interleaved pairs after a warm-up, median of
  rounds: the mode's `--spec dflash` against the mode's `--spec mtp`, recorded with its grade. A mode that does not
  reach 1.0x is recorded as an option that is slower, and the docs say so.
- **Zero-config.** No new flag and no environment variable for users; the modes follow `--kv-cache`,
  `--max-num-seqs` and `--parallel`. Test and probe overrides go through `CaptureOptions` and the existing test
  seams (`B70_KV_CACHE`).

## 7. Out of scope

- **Making DFlash the default in any mode.** Spec 19 v2's G-speed still fails on prefill; `--spec auto` stays MTP.
- **The draft pass's cost** (the tuned int4 drafter, the small kernels, the tail projection's prefill cost): item 12
  and the drafter quant follow-up. This spec uses the pass as it is.
- **The drafter split over two cards** (§4.2 C1's rejected form) and **mirroring drafts on the device.**
- **Blocks above 8 rows, a tree verify, lossy acceptance:** spec 19 v2 §4, unchanged.
- **`--spec dflash-lookup`** in the new modes: it stays one card, one slot, bf16 or int8 KV as mode A leaves it.
- **Other models:** Ornith, Flash-Next, Kolibri and Bolt have no slots, TP or int8 KV path for a drafter here.
- **`--parallel replica`:** branch `dflash-replica`.
- **The int8 prompt pass** (`--prefill-backend l0-int8`) beside DFlash: as today.
