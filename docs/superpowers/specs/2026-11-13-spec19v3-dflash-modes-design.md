# Spec 19 v3 - DFlash: a cheaper round, then the modes we serve (int8 KV, tensor parallel, request slots)

**Status:** design, 2026-11-13 (revised the same day after reading two reference engines). Nothing here is built.
Every speed number for something not built is **derived** from numbers already measured; each plan starts by
measuring the number its work hinges on.

**The operator's target:** `--spec dflash` under tensor parallel at **170-190 t/s** for one session - a whole
speculation round of about 22-24 ms at 4.0-4.6 tokens a round. Today tensor parallel with MTP is 123.1 t/s.

**Continues** spec 19 v2 (`2026-11-11-spec19v2-dflash-v2-design.md`), whose §4 parked the modes and whose cost
targets (its C7) are still open. Spec 19 v2 stays the record of the one-card drafter.
- **Plans, in order:**
  1. `docs/superpowers/plans/2026-11-13-spec19v3d-dflash-round-cost.md` - the round's cost on one card (§3);
  2. `2026-11-13-spec19v3c-dflash-tensor-parallel.md` - tensor parallel (§6);
  3. `2026-11-13-spec19v3b-dflash-slots.md` - request slots (§5).
- **Not scheduled here:** `2026-11-13-spec19v3a-dflash-int8-kv.md` - the int8 KV cache (§4), in progress in
  another session (2026-11-13).
- **Open-work:** item 12.
- **Evidence:** `docs/BENCHMARKS.md` ("DFlash (spec 19)", "Tensor parallel (spec 17)"),
  `docs/probes/dflash-vs-sly-2026-11-11.md`, `docs/probes/probe-verify-attn-2026-11-03.md`,
  `docs/probes/probe-batching-2026-10-23.md`; for §2 the code and docs of vllm-sly-radiance ("sly") and
  saber-engine ("saber"), the two engines that probe already compares us with.

**Grades:** **measured** (with its source), **derived** (arithmetic on measured numbers), **estimated** (a guess with
its reason). For the reference engines: **[R]** read in their code, **[D]** stated in their docs or comments,
**[I]** inferred.

## 0. The short answer

`--spec dflash` is speculative decoding with a block drafter: a small model drafts 7 tokens in one pass, the target
checks 8 rows (the pending token and the 7 drafts) in one step, and the output is the target's own. One draft-and-
check is a **round**. Today it runs on one card with the bf16 KV cache (the stored attention keys and values of the
context), on the two-card layer split (`--parallel pp`), and - on branch `dflash-replica` - with a model copy per card.

| work | what it is | derived result | verdict |
|---|---|---|---|
| **D. the round's cost** (§3) | check 8 rows for ~1.05 plain steps (now 1.25); draft in <= 5 ms (now 8.4) | one card ~110 t/s on the serving bench (88.8 today; MTP 73.2) | **first.** It multiplies every mode; the verify half also speeds MTP and slots |
| **C. `--parallel tp`** (§6) | the drafter split over both cards like the target; a two-phase round | today's kernels ~127 t/s; after D ~157-166; with cheaper crossings ~166-173; **170 is within reach, 190 needs 4.6 tokens a round or the references' 22 ms round** | **second: measure first** (verify at 8 rows under TP), then build |
| **B. `--max-num-seqs` > 1** (§5) | a drafter session per slot; one batched draft pass | one request 1.19x MTP; two ~1.0-1.1x; four ~0.95x | **last**, and only if its host-only arithmetic holds |
| **A. `--kv-cache int8`** (§4) | three kernel builds and four table rows; the source exists | ~1.15-1.18x MTP over int8; context ~140k -> ~245k on one card | **in progress in another session (2026-11-13)**; not scheduled here |

**The reference engines reach ~186 t/s on two cards with the same drafter** (§2), on a plain step equal to ours
(18.7 against 18.9 ms). They are not faster hardware. Their round is cheap in two places where ours is not: checking
8 rows costs them about one plain step, and their draft pass costs ~5 ms **across both cards** against our 8.4 ms on
one.

## 1. Where we start

### 1.1 What is measured (one card, bf16 KV, int8 drafter, candidate head)

| quantity | value | source |
|---|---:|---|
| plain step at depth 4096 | 29.0-29.2 ms | BENCHMARKS "V2 baseline" |
| verify at 2 / 4 / 8 rows | 1.047 / 1.122 / 1.245 plain steps (4096); 1.272 at 8 rows (32768) | the same |
| draft pass, K = 7 | 8.43 ms = 0.287 plain steps | BENCHMARKS "Task 7" |
| ... its parts (device time, depth 4096) | int8 body GEMMs ~3.2 ms; candidate head path 1.79 ms (GEMM 1.21, top-64 0.44, re-rank 0.14); attention 2.08 ms for 5 layers; norms 0.13 ms | BENCHMARKS "V2 baseline", "Task 6", "Task 7" |
| one round (projection + draft + verify + accept) | 1.559 plain steps = 45.5 ms | BENCHMARKS "Task 7" |
| serving bench (3,425-token prompt, 512 tokens) | DFlash 88.8 t/s, MTP 73.2 t/s | BENCHMARKS "V2 baseline", README |
| tokens a round on that bench | ~4.04 (**derived**: 88.8 t/s x 45.5 ms); tool calls 6.65, golden code 6.08 measured | - |
| the opencode replay (78 requests) | DFlash 1.193x MTP | BENCHMARKS "Task 8" |
| prefill, 4096-token prompt | DFlash 0.955x MTP (the open failure of spec 19 v2) | the same |
| tensor parallel: plain step with the head, verify at 2 / 3 / 4 rows | 18.9 ms; x 1.09 / 1.13 / 1.20 | BENCHMARKS, spec 17d |

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

## 2. What the reference engines do

Both run the same drafter (`z-lab/Qwen3.8-27B-DFlash2`, block 8, K = 7) on AMD R9700 cards.

### 2.1 Their headline numbers

- **The comparable figure is 186.0 t/s:** the vLLM stack on two cards (tensor parallel), K = 7, 22.3 ms a round, a
  weighted benchmark mix sampled at temperature 0.7 [D]; that is ~4.2 tokens a round [I].
- saber's two-card round at 8 rows: 22.05 ms now, 24.19 ms earlier = 18.79 ms device step + ~5.4 ms proposal [D].
- saber's "229 t/s" is one README sentence with no workload, sampling or log [D]. Its workload-stamped numbers are
  one-card greedy runs of 256 tokens: 67 (prose) / 148 (code) / 231 (counting) t/s [D].
- **The base step is equal:** their plain two-card step is 18.7 ms, ours 18.9 ms. Their cards peak at 635 GB/s and
  decode at 520-590 GB/s [D]; ours decodes at ~560 GB/s. The one hardware difference that matters is the link
  between the cards: theirs carries ~28 GB/s, ours ~9.6 GB/s measured (PCIe 3.0).
- sly's sampled numbers include a lossy acceptance rule, on by default, worth +3..20 % tokens a round (~+10 %
  weighted) [D]. **We stay exact** (§9).

### 2.2 Their verify step at 8 rows under tensor parallel (saber, 18.79 ms device time) [D]

GEMMs 13.3 ms, GDN 1.8, attention 0.5, the all-reduces 1.8, small kernels 1.3. Rows are nearly free because:

- **One matrix tile holds all the rows.** The decode GEMM stages up to 16 rows into one 16-row tile of the matrix
  instruction: 1 row and 8 rows are the same kernel, the same grid and the same weight traffic [R]. Their head costs
  1088 us at 1 row and 1131 us at 8 [D].
- **Dequantise once per weight tile** into shared local memory, the block scale folded into the weight byte [R].
- **GDN:** all rows in one kernel, the state kept in registers, row t continuing from row t - 1; each row stores its
  own snapshot, and accepting `a` rows means the next step starts from snapshot a - 1 with no copy [R].
- **Attention:** one kernel for all rows of a sequence; the rows are extra lanes on one K / V stream; 8-bit KV [R].
- **Glue fused:** 802 body launches unfused, 483 served - the add, norm and quantise pairs folded into the GEMM
  before them, the small GDN projections folded, RoPE and the KV append fused [R].
- **The all-reduce** (adding the two cards' halves of a split linear) runs card to card inside the recorded graph:
  129 a step, ~14 us each; an 8-row message is 80 KiB [R/D].

### 2.3 Their round under tensor parallel (saber) [R]

- **The drafter is split over both cards** like the target: half of every linear per card, an all-reduce after the
  two row-parallel ones (11 reduces a draft pass), the drafter's KV split by head, small tensors replicated.
- **The draft head is split by vocabulary half.** Each card picks its top 16 and the two lists are merged on the
  device. The exact re-rank reads the target's head half that card already holds.
- **Taps are local:** each card copies its own rows; the residual is the same on both after every all-reduce.
- **A round is two phases, not one graph:** the verify step on both cards, the host accepts, then a separate
  proposal (context store, the drafter's backbone, the merge, a short tail). The drafted ids reach both cards through
  the host. About three host waits a card a round.
- Rejected verify rows go to a pad slot of the context store.
- At batch 1 the verify is a 10-row tree; a chain of 8 above batch 1. The tree's evidence is an offline simulation
  only (§9).

### 2.4 Their draft pass

| | saber [D] | sly [D / I] | ours (measured, one card) |
|---|---|---|---|
| drafter linears | 8-bit float, ~1.73 GB, ~590 GB/s | 4-bit, ~0.9 GB; ~0.6 ms of GEMMs at 8 rows | int8, 1.665 GB, ~3.2 ms (~520 GB/s: at bandwidth) |
| attention over the 2048 window, 5 layers | in the 90-launch graph | **38 us a layer** with a split kernel (133 us stock): 0.19 ms | **2.08 ms** (was 2.70) |
| draft head | 2-bit candidates (~358 MB) -> top 80 -> exact re-rank (11 us): ~0.5 ms | 4-bit head 656 MB, no re-rank: 1.3 ms | 4-bit candidates (0.64 GB) -> top 64 -> exact re-rank: **1.79 ms** |
| drafter KV | 8-bit | 8-bit | bf16 ring |
| launches | 90 + 12 | one graph | 88 + 8 (so the launch count is not the gap) |
| **draft pass** | **4.4 ms on one card, ~5.4 ms served on two** | ~3.3-3.8 ms [I] | **8.43 ms** |

### 2.5 Their round against ours

| | reference (two cards) | ours, one card (measured) | ours, two cards today (derived, §6.4) |
|---|---:|---:|---:|
| plain step | 18.7 ms | 29.2 ms | 18.9 ms |
| verify at 8 rows | 18.8 ms (~1.0 plain) | 36.3 ms (1.245 plain) | ~26 ms (~1.38 plain) |
| draft pass | ~5.4 ms | 8.4 ms | 8.4 ms whole on one card; ~4.6 ms split |
| projection + host | inside the phases; ~3 waits a card | 0.8 ms | ~1.0-1.2 ms |
| tokens a round | ~4.2 (sampled, lossy rule on) | ~4.04 (greedy, exact) | the same |
| **round** | **22.05-24.2 ms** | **45.5 ms** | **~31.8 ms** (split drafter) |
| **t/s** | **186** | **88.8** | **~127** |

## 3. The round's cost (plan 19v3d)

Done on one card, because it multiplies every mode. Two halves; each starts by measuring **our** lists per kernel
family, and each lever is kept only if its measured saving clears its bar.

### 3.1 Verify at 8 rows: 1.25 -> ~1.05 plain steps

**What we already have of §2.2** - checked in our code, not assumed:

| their lever | ours | have it? |
|---|---|---|
| all rows in one matrix tile, one weight read | `gemm_sm` (`src/kernels/gemm_sm.cl`): rows in blocks of 8 on DPAS (the card's dot-product matrix instruction), bound in every list of 2+ rows, the int8 head too (`gemm_sm_i8w`); its header's property R: a row's cost shape does not depend on the row count | **yes.** One difference: 1 row is another kernel (`gemv`), so "8 rows against a plain step" compares two kernels |
| dequantise once per weight tile | `gemm_sm`: "one dequant per weight tile for every row", in registers (the magic-exponent form) | **yes.** The scale is applied per 64-group after the DPAS chain; int4 with group scales cannot fold it into the weight |
| GDN: one kernel, state in registers, a snapshot per row, accept by index | `gdn_step.cl`'s `SPEC_SLOTS` build: the rows chain in registers, the state after row m goes to slot (live + m) % 8, `commit` sets `Control::gdn_live`, no copy | **yes.** 8 rows write 8 x 151 MB of state a verify - a suspect, to be measured |
| attention: one kernel for all rows on one K / V stream, then a combine | `attn_verify_dpas` + `attn_reduce_v2`: 107 us a layer at 4096 keys for 8 rows (1.7 ms a verify) | **yes.** Over int8 KV after plan 19v3a |
| glue folded into the GEMM before it | the one-card step is 774 launches, ~12 a layer: the residual fold and the norm are two launches, the SiLU its own, the taps 5 x M copies | **partly** - the main candidate |
| one recorded graph per shape | one captured list per row count, replayed | **yes** |
| the all-reduce inside the list | `tp_send` / `tp_res_fold` (the wait fused into the fold; the send its own launch) | **yes** (mode C) |
| a 10-row tree | no | not taken (§9) |

So the 0.25 is not a missing kernel family. Verify grows by ~1.0 ms a row from 2 to 8 rows (30.4 -> 36.3 ms), and
nobody has attributed it. **The plan's first task measures verify at 1, 2 and 8 rows by family** (linears per shape,
GDN, attention, head and argmax, glue, tap copies, the gaps between launches) and fills a stop / continue table.
Candidates it will judge, each with a bar of 0.3 ms (1 % of a plain step) at 8 rows:

- glue at M rows folded into its neighbour (fold + norm; SiLU into `gate||up`'s epilogue; the tap copies into the
  fold that produced the row);
- the GDN step's per-row snapshot writes and its row chain;
- any `gemm_sm` shape that is slower at 8 rows than `gemv` at 1 (a tuning row in `kernels::kSmallMTuning`);
- the head's argmax and logits rows at 8 rows.

**Target:** verify at 8 rows <= 1.10 plain steps as the gate, 1.05 as the aim. MTP's verify at 4 rows (1.12) and the
slots' row step (1.53 at 8 rows from 4 slots) bind the same kernels and are re-measured with every kept lever.

### 3.2 The draft pass: 8.4 -> <= 5 ms

The GEMMs already run at bandwidth (5.39 ms for 2.94 GB at the v2 baseline = ~545 GB/s). The time that is not
bandwidth is the attention, the top-k and the gaps. In the order of the measured gap:

| step | today | lever | have the pieces? | derived after |
|---|---:|---|---|---:|
| attention over the window | 2.08 ms | the drafter's attention on DPAS: `df_attn_part` is a scalar fp32 walk with its own `exp`; the target's `attn_verify_dpas` does 8 rows over 2048 keys in ~80-110 us a layer | **the kernel family exists for the target, not for the drafter** (32 query / 8 KV heads of 128, a ring, non-causal block keys) | ~0.5 ms |
| ... the ring's format | bf16 | int8 ring by the target's rotation scheme (`kv8`) | exists for the target; changes the snapshot bytes and the prefix cache key | only if the bf16 form leaves > 0.6 ms |
| candidate selection | 0.44 ms (two top-64 stages) | per-block maxima taken in the candidate GEMM's epilogue, then one small top-k | no | ~0.15 ms |
| candidate head bytes | 1.21 ms (int4, 0.64 GB) | 2-bit candidates with 64+ re-ranked exactly (the references restore acceptance at 64, not at 32) | the exact re-rank exists (`df_rerank`); the 2-bit form does not | ~0.6 ms |
| body bytes | ~3.2 ms (int8, 1.665 GB) | the tuned AutoRound int4 drafter (open-work, "DFlash drafter quant"; the at-load int4 form measured 1.91 ms and failed acceptance) | the int4 lists exist; the export and its loader do not | ~1.9 ms |

Derived draft pass: **~5.7 ms** with the attention and the selection alone, **~5.1 ms** with the 2-bit candidates,
**~4.4 ms** with the int4 body instead, **~3.8 ms** with both.

**Gates for every lever:** greedy output unchanged (the drafter never changes an output id); acceptance on all 35
corpus / K rows of spec 19 v2's record not lower by more than 0.05 tokens a round; the drafter's lists judged by the
f32 judge where arithmetic order changes; the saving at its bar in interleaved pairs.

### 3.3 What it is worth (derived)

One card after §3.1 and §3.2 at their gates: round = 1.10 x 29.2 + 5.0 + 0.8 = 37.9 ms -> **~107 t/s**; at the aims
(1.05, 4.4 ms) 35.9 ms -> **~113 t/s** on the serving bench, against 88.8 today and MTP's 73.2 (MTP gains a few
percent from §3.1 too).

## 4. Mode A - the int8 KV cache (plan 19v3a; in progress in another session, 2026-11-13)

### 4.1 What blocks it

`cli::dflash::check_engine` (`src/cli/dflash_args.h`) throws for `--kv-cache int8`. The reason behind the message is
three missing binaries and one missing table section - not a missing kernel source.

| piece | what exists | what is missing |
|---|---|---|
| the int8 KV writer `attn_prep_kv8` (`src/kernels/kv8.cl`, `KV8_PREP`) | `attn_prep_M<M>_KV8` at M = 1..4. The source has no row limit: its row-table build `attn_prep_M<M>_KV8_ROWS` is built at M = 2..8 for the slots engine | `attn_prep_M<M>_KV8` at M = 5..8 |
| the int8 reader and its reduce (`kv8.cl`, `KV8_DEC`) | `attn_v2_M<M>_T32_KV8` at M = 1..4. The source stops the single-sequence build with `#error "kv8 dec: M is 1..4"`; under `BATCH_ROWS` it allows 1..8 and that build runs in production (slots) | the bound lifted to 8 and `attn_v2_M<M>_T32_KV8` at M = 5..8 (the verify binds its `attn_reduce_v2_kv8`) |
| the verify attention on DPAS (`src/kernels/attn_verify_dpas.cl`) | bf16 at M = 2..8; int8 (`KV8=1`) at M = 2..4. The source accepts `KV8=1` at M = 2..8 | `attn_verify_dpas_M<M>_T32_KV8` at M = 5..8 |
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

### 4.2 The design

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

### 4.3 Memory and `--max-len auto` (one 32.5 GB card, derived)

Per position: bf16 KV 65,536 B; int8 KV 33,024 B plus the int8 prefill's widening rows 4,096 B. The drafter's fixed
terms: weights 2.05 GB (int8), the candidate head 0.68 GB (an int4 copy of the head that picks 64 candidates a row,
re-ranked exactly; on whenever the head is int8), 7 more GDN slots 1.06 GB, buffers and prefill tap staging 0.17 GB.

| one card | MTP (today) | DFlash, bf16 KV (today) | DFlash, int8 KV (this mode) |
|---|---:|---:|---:|
| `--max-len auto` | 171,008 bf16 / 262,144 int8 | ~130-140k | **~225-245k** |

- **It does not reach the full 262,144 on one card.** It is short by ~0.7-1.4 GB. The tuned int4 drafter would free
  0.84 GB, 2-bit candidates 0.32 GB (§3.2).
- On `--parallel pp` with int8 KV the full context fits (each card holds half the target).
- The planner prints the real figure; the plan records it and gates on >= 200,000.

### 4.4 Expected speed (derived)

- At ~4k keys a round grows by 0.7 ms on 45.5 ms: -1.5 %. MTP's own int8 loss there is ~1 % (71.7 against 72.3 t/s,
  probe-batching "Engine"). **DFlash / MTP ~1.18x** on the replay.
- At ~32k keys: +3.7 ms on 51.4 ms, -7 %; MTP's verify at 4 rows pays ~1.4 ms on ~40 ms, -3.5 %. **~1.15x.**
- Past 100k keys the 8-row int8 attention is ~42 ms a verify against MTP's ~25 ms at 4 rows; DFlash still commits
  more tokens a round, but the ratio there is **estimated**, not derived. The plan measures 120,000 keys.
- **Least sure:** the int8 numbers predate `PSPLIT=1` and ran on a busy box.

## 5. Mode B - several request slots on one card (plan 19v3b)

### 5.1 What blocks it

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

### 5.2 The design

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

### 5.3 Memory (one card, derived)

Budget for positions, extra GDN states and rings once the drafter's fixed 2.9 GB is taken: ~10.3 GB. One GDN state
is 151 MB; a slot's ring 42 MB; a short slot's KV at 32,768 positions 2.17 GB (bf16) or 1.11 GB (int8).

| `--max-len auto`: slot 0 + short slots | MTP (today) | DFlash, B2's states | DFlash, 8 states everywhere |
|---|---:|---:|---:|
| 4 slots, bf16 | 46,080 + 3 x 32,768 | ~12k + 3 x 32,768 | does not fit |
| 4 slots, int8 | 167,936 + 3 x 32,768 | ~105k + 3 x 32,768 | ~57k + 3 x 32,768 |
| 2 slots, bf16 | - | ~97k + 32,768 | ~87k + 32,768 |
| 2 slots, int8 | - | ~198k + 32,768 | - |

**DFlash with four slots is an int8-KV configuration**: mode A comes first. In bf16 slot 0 keeps ~12k tokens.

### 5.4 Expected speed (derived)

Costs in plain steps: a row step of 8 rows from 4 slots is 1.53 (47.8 against 31.3 ms, measured, probe-batching);
the batched draft 0.29 (one card's draft pass: the same weights, head and rows); MTP's row draft ~0.15 a draft index.

| requests active | drafts each | DFlash step | MTP step | DFlash / MTP at equal tokens a draft |
|---|---:|---:|---:|---:|
| 1 | 7 / 3 | today's one-card round | today's | **1.19x** (measured on one slot) |
| 2 | 3 | ~1.4 + 0.29 | ~1.4 + 0.45 + head KV | **~1.0-1.1x** |
| 4 | 1 | 1.53 + 0.29 | 1.53 + 0.15 + head KV | **~0.95x** |

- "At equal tokens a draft" is the unknown: how many tokens DFlash commits at K = 1 and K = 3 against MTP on the
  same text. The numbers are on the box already. **The plan's first task is that arithmetic, on the host, with a
  stop condition.**
- §3's cheaper draft pass moves the two- and four-request rows up by a few percent; the row step's own cost (rows of
  different slots read their own state and KV) is spec 13c's and not this spec's.
- So the mode's worth is the lone request in a slots server: a coding agent's main session is alone most of the time.
- **Least sure:** the batched draft at mixed blocks keeping one card's acceptance - K <= 3 blocks are the drafter's
  weaker range. The references confirm the direction: sly's longer blocks lost 12-17 % at 4-16 requests [D].

## 6. Mode C - tensor parallel (plan 19v3c)

### 6.1 What blocks it

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
  the 5 tapped layers' outputs exist whole, at hidden 5120, on **each** card.
- **The drafter and the head:** nothing places them today. Each card has the whole embedding but only its vocabulary
  half of the int8 head (124,160 rows).
- **The single submit:** `Engine::verify(k, draft)` runs draft then verify on one queue because the drafts land in the
  `Control` the verify reads. Under TP there are two queues and two `Control` blocks.
- **Prefill and snapshots:** the TP prefill (`runtime/prefill/tp_prefill.cc`) stages no taps; `TpEngine::save_state`
  writes the one-card layout without a ring.

### 6.2 The design

**This reverses the first version of this section.** It put the drafter whole on card 1, to reuse one card's draft
list unchanged. That leaves an 8.4 ms draft pass on one card while the other idles - 24 % of a TP round - and needs
a second copy of half the head. The reference engine that reaches the target splits the drafter (§2.3); with the
split the draft pass gains from TP like the target does, and each card re-ranks against the head half it already
holds.

- **C1. The drafter is split over both cards like the target.**
  - Column-parallel (each card computes its half of the outputs, no exchange): `q||k||v` by head (16 query / 4 KV
    heads a card), `gate||up`, the context `k||v`.
  - Row-parallel (each card computes a partial of every output; an all-reduce follows): `o` and `down` - two a
    layer, ten a draft pass, through the target's own `tp_send` / `tp_res_fold` at 8 rows.
  - Replicated whole on both cards: the norms, the conv kernel projections, `fc` (131 MB; each card has every tap
    locally), the selector's projection and codebooks.
  - The drafter's ring is split by KV head, as the target's KV is.
  - *Rejected: whole on one card* (above). *Rejected: a copy on each card sharing rows:* each card would still read
    all the weights.
- **C2. The draft head by vocabulary half.** Each card: its half of the candidate head, its top 64, the exact re-rank
  against its own half of the target's head, its top 16 with exact scores.
  - Two halves' top 64 contain the whole vocabulary's top 64, so the merged 16 are at least as good as one card's.
    The drafts may differ from one card's; the acceptance gate judges it.
  - **The merge of the two top-16 lists:** first on the host, between two submits (one extra wait, ~0.3 ms). Then a
    device merge - card 0 pushes its 16 into card 1's memory with the crossing's flag protocol and the selector runs
    on card 1 - kept only if it saves >= 0.2 ms measured.
- **C3. Taps are local copies on each card** from its own residual (`build_tp_verify_taps`: `build_tp`'s walk at M
  rows, 8 GDN slots at the card's shapes, the tap copies). Nothing crosses.
- **C4. A round is two phases:**
  1. both cards' verify lists; the two fences; the per-row merge; the host accepts and commits on both cards;
  2. both cards: [projection of the committed rows, draft]; the candidate merge; the selector; the host writes the
     drafted ids into both `Control` blocks.
  - This is the references' shape too. One submit for the whole round would need the verdict on the device.
  - Host cost: two submits and waits a card a round, ~0.6-1.0 ms (one submit's host gap is 0.38-0.42 ms measured).
- **C5. The context store: our deferred projection, not a pad slot.** saber runs its store over all 8 verify rows
  with launches whose arguments depend on the verdict, rejected rows going to a pad slot. Ours is one captured list
  that reads the row count from its control block and stores only the committed rows; it is built, tested with
  snapshots and truncation, and in a two-phase round it sits at the front of phase 2 at no extra wait. Each card
  projects with its replicated `fc` and stores its own KV heads.
- **C6. Prefill and snapshots:** each card stages the taps of the prompt's last 2048 positions and projects them 8
  rows a replay. Snapshots gather the two ring halves into the one-card layout (the way `runtime/tp_gather.h`
  gathers the target's state); the prefix cache key takes the drafter's key.
- **C7. The K policy** reads a TP drafter cost table measured in the plan. Sampled requests gather 8 full rows from
  the two halves (`TpEngine::read_logits_into`, today 4).
- **`--spec dflash-lookup` under TP stays refused** (it is built for one card's adapter).

### 6.3 Memory (per card, derived)

TP holds the full 262,144 context in bf16 today at ~21 of 31 GB usable a card. Without the MTP head each card frees
its head weights and the head's KV layer (1.07 GB at 262,144). Each card adds half the drafter ~1.03 GB, the
replicated small tensors ~0.15 GB, half the candidate head 0.34 GB, tap staging and buffers 0.15 GB, and 0.30 GB for
8 GDN slots at half shapes instead of MTP's 4. **Each card ~21.5 GB: `--max-len auto` stays 262,144.**

### 6.4 Expected speed (derived)

Parts:
- **Verify at 8 rows under TP** = the one-card ratio x 18.9 ms + the crossings' extra bytes. A crossing at 8 rows is
  160 KiB in fp32; 128 of them are 20 MB each way, ~2.1 ms at the measured ~9.6 GB/s, against 0.27 ms at 1 row. So
  today 1.245 x 18.9 + 1.8 = **~25.3-26 ms** (the measured 2..4-row slope gives the same 26). bf16 messages (the
  references' 80 KiB) would halve the crossing.
- **The split draft pass, today's kernels:** GEMMs (3.2 + 1.2) / 2 = 2.2 ms, attention 1.0, selection and re-rank
  0.3, ten reduces 0.3, the merge 0.3, the rest 0.5 = **~4.6 ms**. After §3.2: **~2.8-3.4 ms** (say 3.0).

| assumptions | verify | draft | host, projection | round | t/s at 4.04 tokens | at 4.2 | at 4.6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| today's kernels, drafter whole on one card (the first version) | 26 | 8.4 | 1.0 | 35.4 | 114 | 119 | 130 |
| today's kernels, drafter split | 26 | 4.6 | 1.2 | 31.8 | 127 | 132 | 145 |
| after plan 19v3d (verify 1.05, draft levers) | 21.7 | 3.0 | 1.0 | 25.7 | 157 | 163 | 179 |
| ... and bf16 crossing messages, host 0.6 ms | 20.7 | 3.0 | 0.6 | 24.3 | 166 | 173 | **189** |
| the references' costs (saber, 8 rows) | 18.8 | 5.4 | in the phases | 22.05-24.2 | 167-183 | 174-190 | 190-209 |

- **170 t/s** needs all of: verify at 8 rows at ~1.05 plain steps on one card (§3.1), the split drafter at ~3 ms
  (§3.2 and C1), and either bf16 crossing messages or 4.3+ tokens a round.
- **190 t/s** needs, on top, ~4.6 tokens a round (tool-call-heavy work: 6.65 measured; the serving bench gives 4.04)
  or the references' round of ~22 ms (verify under TP at ~1.0 plain step).
- Tokens a round are ours, greedy and exact. The references' 4.2 includes their lossy rule's ~10 %.
- **Prefill** pays the tail projection's ~120 ms a prompt (spec 19 v2's open failure): ~8 % of a 4096-token TP
  prefill until that is fixed.
- **Least sure:** verify at 8 rows under TP - both its ratio and the crossings' cost on our link. It is the plan's
  first task, with a stop rule.

## 7. Recommended order

1. **Plan 19v3d - the round's cost on one card.** Every mode multiplies it; the TP target cannot be met without it;
   its verify half also speeds MTP and slots. One card (card 0).
2. **Plan 19v3c Task 1 - verify at 8 rows under TP, measured.** Needs both cards: before or after the weekend's
   Flash-Next quantisation on card 1, never beside it. It can run before plan 19v3d ends; its stop rule is written
   for either case.
3. **Plan 19v3c's build** on a "go".
4. **Plan 19v3b - slots**, last, and only if its host-only first task projects >= 0.95x MTP at two requests and
   >= 0.90x at four.

**Not in this order: plan 19v3a (the int8 KV cache)** - in progress in another session (2026-11-13). Slots need it
merged; nothing else here waits for it.

## 8. Gates (every plan)

- **Same output.** Greedy output identical to `--spec off` in the same mode. Where the mode's own verify rows are not
  bitwise its plain steps (DPAS attention, the small-M GEMM, TP's folded crossing, row lists), each first difference
  is judged by the f32 judge - a CPU reference in f32 that says whether the chosen token is the true top or a near
  tie (`tools/box_validate/serve_judge.py`, `tests/server/spec_judge.h`). No failure, no skipped request.
- **Nothing else moves.** Without `--spec dflash` in the mode, every list binds the binaries it bound before
  (`tools/kernel_cmdlines`: additions only; the G0 suite green) - except plan 19v3d's verify levers, which change
  MTP's and the slots' lists on purpose and carry those lists' own gates.
- **Acceptance not lower than one card's.** Tokens a round on the tool-call set and golden code, generation stopped
  at EOS, within 0.15 of one card's bf16 run (6.65 and 6.08 measured, BENCHMARKS "V2 baseline"). A larger gap is a
  defect in the taps or the ring to root-cause, never a bar to widen.
- **Snapshots.** A turn served as a prefix cache hit drafts as the unbroken session does (the ring in the snapshot).
- **Speed, once at the end,** on an idle card (both cards for mode C), interleaved pairs after a warm-up, median of
  rounds, recorded with its grade. A mode that does not reach 1.0x MTP is recorded as an option that is slower.
- **Zero-config.** No new flag and no environment variable for users; the modes follow `--kv-cache`,
  `--max-num-seqs` and `--parallel`. Test and probe overrides go through `CaptureOptions` and the existing test
  seams (`B70_KV_CACHE`).

## 9. Considered, not taken now - and out of scope

**Considered, not taken now:**
- **A 10-row tree verify** (saber's default at batch 1): its evidence is an offline simulation only (+7.8 % tokens a
  round, oracle bound +39 %); no served measurement [D]. It needs tree forms of GDN, the conv and the attention mask.
- **K = 15 (16-row verify):** sly measured +10 % for one stream on code and JSON, but -12..-17 % at 4-16 requests,
  -9 % on chat, and one agent request at 19 against 84 t/s; its default went back to 7 [D]. The drafter is trained
  at 8 (4.205 accepted at 8 against 3.961 at 9).
- **Lossy acceptance** (sly's default rule, ~+10 % tokens a round weighted [D]): our output is exactly the target's.
- **Speculation under TP elsewhere:** the Intel-card engines we track have none; one keeps MTP as its default over
  this drafter family.

**Out of scope:**
- **Making DFlash the default in any mode.** Spec 19 v2's G-speed still fails on prefill; `--spec auto` stays MTP.
- **The prefill's tail projection cost** (spec 19 v2's open failure) and **quantising the drafter** (the AutoRound
  export is the open-work follow-up; plan 19v3d binds it when it exists).
- **bf16 crossing messages under TP** as a change to plain TP decode: plan 19v3c measures them for the 8-row verify
  only and brings the result to the operator.
- **`--spec dflash-lookup`** in the new modes: it stays one card, one slot.
- **Other models:** Ornith, Flash-Next, Kolibri and Bolt have no slots, TP or int8 KV path for a drafter here.
- **`--parallel replica`:** branch `dflash-replica`.
- **The int8 prompt pass** (`--prefill-backend l0-int8`) beside DFlash: as today.
