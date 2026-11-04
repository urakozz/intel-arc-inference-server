# Probe: batching - rows of different sequences in one decode step (spec 13a, 2026-10-23)

Spec 13 §9 / §9.1 / §9.2 (the operator's decisions of 2026-10-23) and plan 13a: which kernels of the decode, MTP verify
and draft lists assume one sequence, a probe build that runs rows of several slots in one step, its correctness check,
the step and chunk costs, the memory of the decided layouts, and the row table 13b builds. Qwen3.8 batches four slots
(1 x 131072 + 3 x 32768, MTP in every slot); Agnes two (1 x 131072 + 1 x 32768, MTP in both) or four with `--mtp off`.
Every number is **DERIVED** (from the code's formulas or spec 8b's measured costs), **ESTIMATED**, or **PENDING (box)** -
queue row 66; nothing batched has run on a card. Written on the Mac, branch `spec13a-batching-probe`.

## 1. The audit (Task 1)

A row of a step is today one of M consecutive positions of ONE sequence (spec 8's verify, spec 19c's tap verify). Spec 13
§9's row table makes it a (slot, position, speculative state index) triple. For every kernel the three lists bind
(`runtime::build`, `build_verify`, `build_draft`; src/runtime/capture.cc), on Qwen3.8 and Agnes alike (Agnes binds the
same sources at its shapes: `gdn_step_slots_M<M>_G54`, `argmax_stage1_M<M>_V248089`, `prep_silu_mul_M<M>_I19456`, the
gate||up' 5120 x 38912 / down' 19456 x 5120 GEMVs; its 54 GDN + 18 FA layers come from the descriptor), the question is
whether row m's output depends on anything but row m's inputs and the weights.

### 1.1 Row-agnostic: bound at M rows unchanged

| kernel | binding (capture.cc) | why rows of different slots need nothing | source |
|---|---|---|---|
| `embed_gather` | `embed_gather()` :596; head :1334 | row m gathers `cur_token[m]` of its control block | embed_gather.cl:51 |
| `prep_res_fold` / `prep_norm_finish` | `res_norm()` :640, `fold_norm()` :1293 | grid (G, M); row m reads and writes only resid / sumsq / x row m (`norm_sumsq` is [G][M]) | prep.cl:298-332, :361-374 |
| `gemv` (int4 g64) | `gemv()` :685 | one accumulator per row (`acc[m]`), weights shared; out `[(s M + m) N + n]` | gemv.cl:92-129 |
| `gemv_bf16` (a\|\|b, the head's linears, the bf16 lm_head) | `gemv_bf16()` :725, `head_gemv()` :1313 | per-row accumulators, as gemv | gemv_bf16.cl |
| `gemv_i8w` (the int8 lm_head) | `gemv_i8w()` :746 | `acc[m]`, row scale shared | gemv_i8w.cl:52-71 |
| `prep_silu_mul` | `mlp()` :769 | grid (chunks, M): row m only | prep.cl:392 |
| `prep_gated_head` | `gdn_layer()` :905 | grid (v-heads, M): z row m and gdn_o row m | prep.cl:436 |
| `argmax_stage1` | `head()` :1040 | grid (243, M): logits row m -> part row m | argmax.cl:83-113 |
| `moe_route` / `moe_gate_up` / `moe_down` (Ornith, follow-on 1) | `moe_block()` :809 | grid (.., M), each row routes on its own logits (spec 15e: M = 4 rows bitwise the M = 1 binary's) | moe.cl:195-324 |

Spec 8b measured the consequence for the GEMVs: the M = 1..4 verify rows are bitwise the M = 1 step, so a row's GEMV
output does not depend on what shares the step. The same holds for every kernel in this table by construction (no
cross-row reduction anywhere). These need **no change** for 13b; they need the M = 5..8 binaries at the model's shapes:
Qwen3.8's exist (`B70_VERIFY_M8`, plan 19a Task 4), Agnes's gate||up' / down' / SiLU / argmax at M = 5..8 do not (13a's
`B70_BATCH_PROBE` block adds them).

### 1.2 Per-sequence: what each needs for rows of different slots

| # | kernel / site | the one-sequence assumption | for rows of different slots | source |
|---|---|---|---|---|
| P1 | `gdn_step` (and `_slots`), `gdn_layer()` capture.cc:872-899 | (a) the conv window: rows m >= 1 take taps from the previous ROW's registers; only row 0 reads the ring at `pos - 1 .. pos - 3`, of ONE ring (the layer's slice); the ring write at `(pos + m) % 16`. (b) the state tile is loaded once from one slot (`state` or `live`), carried in registers across ALL rows and written once at the end (plain) or after each row to `(live + m) % N_SLOTS` | per row: its position, its ring (slot), at a slot's FIRST row the window reloaded from that slot's ring and the state tile loaded from the slot's live state index; after each row the tile written to the row's own destination index. Runs of one slot keep today's register carry (MTP rows) | gdn_step.cl:288 (pos), :313-316 (window), :324 (ring write), :343-354 (tile load, `live` :346), :448-456 (SPEC write), :463-468 (plain write) |
| P2 | `attn_prep` / `attn_prep_kv8` (main FA layers, `fa_layer()` :931-957) | RoPE row and KV row at `pos + m`, the layer's KV slice from row 0 of the cache | per row: RoPE at the row's position, the KV row at the slot's KV base + position | attn.cl:366, :395, :415; kv8.cl:237, :262, :287 |
| P3 | `attn_decode_v2` / `_kv8` (`attn_pair()` :447-498) | ONE `pos`: the rows' key counts are `pos + m + 1`, so they form at most two stride groups (`s0`, `s1`, `msplit`); every work-group loads ONE K/V wave and feeds all 6 x M (q head, row) pairs from it - the M-row cost model of spec 8b (attn_v2.cl:35-36) | per row: its position and KV base; groups = maximal runs of rows of ONE slot with one stride, each group walking its own slot's keys. Rows of different slots share no K/V wave: their attention costs add (spec 13 §9.1, "the dense gains are an upper bound at depth"). The kv8 decode is compiled for M <= 4 only | attn_v2.cl:170-183, :202, :242; kv8.cl:325 (M <= 4), :391-401, :421, :459 |
| P4 | `attn_reduce_v2` / `_kv8` | row m's block count from `pos + m` | the row's own position | attn_v2.cl:294-298; kv8.cl:511-515 |
| P5 | `argmax_stage2` (`head()` :1050) | `cur_token[0] = out[n - 1]`, `pos += n` - the chain of ONE sequence | nothing in the step reads them back: the host writes every row's id and position before the replay (the row table), and reads `out_token[0..M)`. A batched engine never relies on the in-list advance (per slot it is the host's commit) | argmax.cl:136-168 |
| P6 | the MTP head's KV fill (verify tail, capture.cc:253-261, `head_front()` :1330) | `hh` rows 0..M-1 are (h_{pos-1}, the rows' own hidden) of ONE sequence: row 0 reads THE h_{pos-1}, row r >= 1 the verify row r - 1's hidden (copied from `x` :258); the head's `attn_prep_s1` at `hctl.pos + m` | per run of a slot starting at row r0: hh[r0] <- that slot's h_{pos-1}, hh[r0 + j] <- x[r0 + j - 1] (copies, captured per run shape); the head's attn_prep at the row's position - 1, its KV at the slot's base | capture.cc:258-260, :1340-1376; attn.cl (QKV_S = 1) as P2 |
| P7 | the draft list (`build_draft`, capture.cc:1394) | M = 1 on `hctl` (pos, cur_token[0]), `dh`, logits row i; the chain advances `hctl.pos` and `cur_token[0]` in `argmax_stage2` | drafts of S slots either as S sequential M = 1 chains (today's list per slot; S x the draft's weight read, ~2.1 GB a draft on the int8 head) or ONE M = S draft list over the slots' rows: P2 / P3 / P4 on the head's KV per row, a `dh` row per slot, and a row-mode `argmax_stage2` that chains every row (`cur_token[m] = out[m]`, the row's position + 1). The latter reads the head and lm_head once per draft index for all slots (DERIVED: at 4 slots x K = 1 about 0.13-0.15 plain steps instead of ~0.5) | capture.cc:1394-1463; engine.cc:214-242 |
| P8 | `Control` and the host loop (engine.cc) | one `pos`, one `gdn_live`, one `hh` row 0 | per slot: pos, the live GDN index, the pending id, h_{pos-1}; commit per slot (engine.cc:272-290's three lines per slot) | control.h; engine.cc:244-290 |
| P9 | prefill (`runtime/prefill/step.cc` walk_layers) | binds the whole `gdn_state` / `conv_ring` / KV allocations and indexes KV by absolute position from 0 | a slot's prefill binds that slot's live state, ring and KV layer pointers offset by its base (the flash path reads `[0, pos + C)` relative to the pointer it is given) | prefill/step.cc:131-189 |

**Six kernels change** (gdn_step, attn_prep, attn_prep_kv8, attn_decode_v2 / reduce_v2, their kv8 twins), all at the
places listed; the MTP head's fill changes only its input gather (copies) and its attn_prep (P2's build at QKV_S = 1); the
draft list needs P7's decision; the rest is host bookkeeping (P5, P8, P9). Nothing in the GEMV, norm, SiLU, gated-head or
argmax-stage-1 kernels changes, so the step's bandwidth model is unchanged: one weight read per step whatever the rows'
slots, plus every row's own KV read (P3) and GDN state read / write (P1: ~151 MB a slot on Qwen3.8, ~170 MB on Agnes,
read once and written once per slot and step - DERIVED, the same bytes a 1-row step of that slot moves).

**Agnes against Qwen3.8:** no kernel differs in kind - the same sources at 54 GDN / 18 FA layers; the slot stride baked
into gdn_step's slot builds is per model (`_G54`), and the conv ring's per-slot stride follows the GDN layer count.

## 2. The probe and its check (Task 2)

Built only with `-DB70_BATCH_PROBE=ON` (default OFF: a default build compiles exactly main's binaries; `capture.cc`'s
default-build token stream is main's, and all 105 existing command lines of the four edited `.cl` files preprocess to
main's tokens - checked on the Mac with `clang -E -P`, whitespace aside):

- **Row builds** (`BATCH_ROWS` blocks, kernels/batch_kernels.h names; 80 variants, `tools/kernel_cmdlines` +80 / -0 / ~0):
  `gdn_step_rows_M<M>[_G54]`, `attn_prep_M<M>[_S1][_KV8]_ROWS`, `attn_v2_M<M>_T32[_KV8]_ROWS` at M = 1..8, and
  Agnes's row-agnostic M = 5..8 GEMVs / SiLU / argmax. None has been compiled by ocloc (the Mac's clang syntax check
  passes all 80).
- **`runtime::build_rows`** (capture.cc / capture.h under `#if B70_BATCH_PROBE`): the decode list at M rows over the row
  table, plus the MTP head's KV fill over the same rows with each run's hidden rows gathered by copies (P6).
- **`tools/probe/probe_batch_step`**: one engine at the slots' summed length; a slot is a position range of its KV
  cache; GDN states as pool indices (one source per slot + 8 destination scratch states), a conv ring per slot; every
  slot prefilled by `Engine::prefill` into slot 0's place and copied out. Its module comment is the contract.

**The check** (per arm, on the box): each slot's rows of a B-row step equal that slot's run alone - logits bitwise, ids,
every row's destination GDN state, the main KV rows written (K, V, int8 scales) and the head's KV rows; no source state
and no KV row of a window [depth - 8, depth + 16) of any slot moves but the step's own rows (B3 in probe form); and slot
0 alone at 32k equals the production lists bitwise (`Engine::verify(0)` and `verify(3)` with MTP, the plain list without).

| check | where | result |
|---|---|---|
| `gdn_step` rows (4 x 1; 2 x 4; 3 + 1 + 2 in slot order 0, 2, 1) against the plain M = 1 kernel per slot: outputs, dst states, rings | Mac, `tools/mac/clrun/batch_rows_run` (UHD 630, emulated subgroups - INDICATIVE) | **bitwise** (measured); a broken window reload or KV base is caught (mutation, measured) |
| `attn_prep` / `attn_prep_kv8` rows (5 rows, 3 slots) against the M = 1 kernel on the slot alone: q, gate, K / V rows, scales; no other row moved | Mac, the same | **bitwise** (measured) |
| attention v2 / kv8 decode rows (shuffles, 2D block reads: not runnable on the Mac) | box, row 66 `r66.check` | PENDING (box) |
| every arm: rows = runs alone; isolation; the production tie | box, `r66.check` (Qwen3.8 and Agnes, bf16 and int8 KV, Agnes also `--mtp off`) | PENDING (box) |

## 3. The step table

One step's wall time per arm as interleaved pairs against `1x1@4k` (one row of slot 1 at depth 4096), the int8 lm_head,
MTP on (every list carries the head's KV fill, as the engine's lists do). "ids / cost" is the arm's expected ids per
step (1 a plain row; 1.92 / 3.455 for K = 1 / 3, spec 8's measured acceptance) over its cost in plain steps.

**The DERIVED column** (Qwen3.8, int8 KV) is spec 8 / 9's verify table for the int8 head (1.00 / 1.18 / 1.56 / 1.79 at
M = 1..4, `MtpCost::int8_head`; M = 5..8 extrapolated at +0.23 a row, unmeasured) - the cost of M rows of ONE sequence
- plus what rows of other slots add at ~570 GB/s on the 32.65 ms int8 plain step: every extra slot's GDN state read and
written (2 x 151 MB = 0.302 GB) and its own KV read (0.135 GB at 4k int8, 0.268 at bf16), **0.0235 plain steps a slot at
4k int8** (0.0307 bf16); a 32k slot reads 0.95 GB more KV than a 4k one (+0.051 int8, +0.101 bf16). Agnes's extra a slot
is 0.340 GB of state + 37,152 B a position of KV (int8); its plain step is unmeasured, so its columns are PENDING.

| arm (Qwen3.8) | rows | cost, plain steps (DERIVED, int8 KV) | ids / cost (DERIVED) | ms (measured) | x `1x1@4k` (measured) | ids / cost (measured) |
|---|---:|---:|---:|---|---|---|
| `1x1@4k` | 1 | 1.000 | 1.00 | PENDING (box) | 1 | PENDING |
| `1x1@32k` | 1 | 1.051 | 0.95 | PENDING (box) | PENDING | PENDING |
| `2x1@4k` | 2 | 1.203 | 1.66 | PENDING (box) | PENDING | PENDING |
| `3x1@4k` | 3 | 1.607 | 1.87 | PENDING (box) | PENDING | PENDING |
| `4x1@4k` | 4 | 1.860 | 2.15 | PENDING (box) | PENDING | PENDING |
| `2x1@32k+4k` | 2 | 1.254 | 1.59 | PENDING (box) | PENDING | PENDING |
| `4x1@32k+4k` | 4 | 1.911 | 2.09 | PENDING (box) | PENDING | PENDING |
| `1x2@4k` (K = 1) | 2 | 1.180 | 1.63 | PENDING (box) | PENDING | PENDING |
| `1x4@4k` (K = 3, today's `--mtp`) | 4 | 1.790 | 1.93 | PENDING (box) | PENDING | PENDING |
| `2x2@4k` | 4 | 1.813 | 2.12 | PENDING (box) | PENDING | PENDING |
| `2x4@4k` | 8 | 2.733 | 2.53 | PENDING (box) | PENDING | PENDING |
| `4x2@4k` | 8 | 2.780 | 2.76 | PENDING (box) | PENDING | PENDING |
| `2x2@32k+4k` | 4 | 1.864 | 2.06 | PENDING (box) | PENDING | PENDING |
| `2x4@32k+4k` | 8 | 2.784 | 2.48 | PENDING (box) | PENDING | PENDING |
| `4x2@32k+4k` (the decided layout, K = 1 everywhere) | 8 | 2.831 | 2.71 | PENDING (box) | PENDING | PENDING |
| `1x4+3x1@32k+4k` (main K = 3, three plain) | 7 | 2.601 | 2.48 | PENDING (box) | PENDING | PENDING |
| `1x4+3x1@4k` | 7 | 2.550 | 2.53 | PENDING (box) | PENDING | PENDING |
| rows `1x1@32k` vs production `verify(0)` | 1 | 1.000 (the same kernels) | - | - | PENDING (box) | - |

At bf16 KV the derived costs move by +0.007 to +0.10 (the extra KV a slot and at 32k: `4x1@4k` 1.882, `4x2@32k+4k`
2.903). **Against spec 13 §1 / §9's figures** (DERIVED there from the bf16 head's M-row costs, no per-slot state or KV):
`4x1@4k` 2.15x here against 2.30x, `4x2` 2.71-2.76 against ~3.0 - the GDN state a slot moves (0.302 GB, read and
written once per slot and step) is the larger of the two terms the M-row table did not carry.

| arm (Agnes, MTP on: 2 slots) | rows | ms (measured) | x `1x1@4k` (measured) | ids / cost (measured) |
|---|---:|---|---|---|
| `1x1@4k`, `1x1@32k`, `2x1@4k`, `2x1@32k+4k` | 1-2 | PENDING (box) | PENDING | PENDING |
| `1x2@4k`, `1x4@4k`, `2x2@4k`, `2x4@4k` | 2-8 | PENDING (box) | PENDING | PENDING |
| `2x2@32k+4k`, `2x4@32k+4k` (the decided layout's shapes), `1x4+1x1@32k+4k` | 4-8 | PENDING (box) | PENDING | PENDING |
| **Agnes `--mtp off` (4 slots)**: `1x1@4k` .. `4x1@4k`, `2x1@32k+4k`, `4x1@32k+4k` | 1-4 | PENDING (box) | PENDING | PENDING |

## 4. The prefill chunk (the interleave decision)

`Engine::prefill` of ONE chunk into slot 0 at depth (the served path: l0-int8, flash attention, the MTP head's KV fill),
rotated rounds, median of 3. While a prompt prefills chunk by chunk between the batch's ticks, a decoding slot's worst
inter-token gap is one chunk plus one step. ESTIMATED from spec 2's ~1 s per 2048-row chunk (linear in rows; the
attention's growth with depth not estimated); the scheduler's mock (plan 13c Task 1, `scheduler_test`) gives the same
ordering in ticks (worst gap 17 ticks at chunk 512, 65 at 2048, 1025 with batched admission).

| chunk | depth | ms / chunk (ESTIMATED) | ms / chunk (measured) | worst gap, ms (measured chunk + step) |
|---:|---:|---:|---|---|
| 512 | 4096 | ~250 | PENDING (box) | PENDING (box) |
| 1024 | 4096 | ~500 | PENDING (box) | PENDING (box) |
| 2048 | 4096 | ~1000 | PENDING (box) | PENDING (box) |
| 512 / 1024 / 2048 | 32768 | more (attention over 32k keys) | PENDING (box) | PENDING (box) |

## 5. Memory (Task 3; the code's formulas, spec 13 §9.1's method)

`tests/runtime/batch_rows_test` (host, runs on the Mac) computes every layout from `runtime/buffer_sizes.h` /
`memory_plan.h` - per slot the KV (with the MTP head's layer when the slot has MTP) and the GDN states (4 with MTP, 1
without) and the ring; shared the weights in the serving form (memory_plan_test's derived int8-head bytes), the MTP head,
the RoPE table at the longest slot, and `runtime::plan`'s scratch (decode, MTP rows, prefill, int8). Budget: 32.53 GB -
the 1.5 GB reserve = 31.03 GB (the reserve is unconfirmed: queue row 7). All DERIVED:

| layout (int8 KV unless said) | KV GB | state | weights | scratch | total | spare |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8 1 x 131072 + 3 x 32768, MTP (decided) | 8.05 | 2.48 | 17.70 | 0.87 | 29.09 | +1.94 |
| Qwen3.8 4 x 65536, MTP | 9.20 | 2.48 | 17.68 | 0.87 | 30.22 | +0.81 |
| Qwen3.8 1 x 131072 + 3 x 32768, MTP, bf16 KV | 15.97 | 2.48 | 17.70 | 0.87 | 37.01 | -5.98 |
| Qwen3.8 1 x 131072 (today's engine), MTP | 4.60 | 0.62 | 17.70 | 0.87 | 23.78 | +7.25 |
| Agnes 1 x 131072 + 1 x 32768, MTP (decided, `--batch 2`) | 6.43 | 1.39 | 20.52 | 0.93 | 29.27 | +1.76 |
| Agnes 1 x 131072 + 3 x 32768, MTP | 9.00 | 2.79 | 20.52 | 0.93 | 33.24 | -2.21 |
| Agnes 1 x 131072 + 3 x 32768, `--mtp off` (`--batch 4`) | 8.52 | 0.75 | 19.67 | 0.86 | 29.81 | +1.22 |
| Agnes 1 x 131072 + 3 x 32768, MTP in the main slot only | 8.79 | 1.26 | 20.52 | 0.93 | 31.50 | -0.47 |
| Agnes 1 x 131072 + 2 x 32768, MTP in the main slot only | 7.57 | 1.07 | 20.52 | 0.93 | 30.10 | +0.93 |
| Agnes 1 x 131072 + 2 x 32768, `--mtp off` | 7.30 | 0.56 | 19.67 | 0.86 | 28.40 | +2.63 |
| Agnes 1 x 131072 + 1 x 32768, MTP, bf16 KV | 12.75 | 1.39 | 20.52 | 0.93 | 35.60 | -4.57 |

The largest batch of 1 x 131072 + short 32768 slots that fits (spec 13 §9.2's refusal numbers, `runtime::largest_batch`):
Qwen3.8 with MTP 4; Agnes with MTP 2, with `--mtp off` 4. Two deltas against the spec's text, both from rounding: §9.1
adds rounded parts (Qwen3.8's exact total is 29.09 GB, 1.94 spare, not 29.10 / 1.93); and §9.2's Agnes `--mtp off` row
says ~29.9 GB / ~1.1 GB spare where the code gives **29.81 GB / 1.22 GB** (DERIVED; the test pins the code's number).

The probe's own allocation (DERIVED, bf16 KV - the larger form): Qwen3.8 at 61,440 positions ~25.3 GB (KV 4.03 + the
head's 0.25, the engine's state with MTP 0.62, the probe's 12 pool states and a backup of slot 0's 1.81, rings 0.05);
Agnes at 45,056 ~27.4 GB; Agnes `--mtp off` at 61,440 ~27.3 GB. All under the budget; `probe_batch_step` prints the
engine's measured line and the pools.

## 6. The per-row descriptor for 13b

13b promotes the probe's arrangement; what it keeps, and what it adds:

- **The table** (`runtime::BatchRow`, 8 words a row, 256 B in shared memory, `runtime/batch_rows.h`): position, KV base,
  GDN src, GDN dst, slot. The host writes it before every replay with the ids (`Control::cur_token[m]`) and
  `Control::n_active`; nothing in a row list reads `Control::pos`. The rules (`runtime::check_rows`): a slot's rows one
  run at consecutive positions; at most 8 rows; destinations distinct and never another run's source; a row below its
  slot's max_len.
- **Slots as position ranges** of one KV allocation (and of the MTP head's KV layer), so the KV layout, the int8 scales'
  layout and the prefill path keep one allocation each; a slot's prefill binds pointers offset by its base (audit P9).
- **GDN state indices:** 13b gives each slot `MtpDims::kSlots` states (its live one and three speculative) with MTP on,
  one with `--mtp off` (spec 13 §9.2); a verify run of slot s reads src = its live index and writes dst = (live + j) mod 4
  in its own four (spec 8's commit as an index, per slot). The probe's shared 8-state scratch is not 13b's layout.
- **The head's hidden rows** per run by copies (P6), per (M, shape) list; the head's own table at position - 1.
- **One list per (M, shape)**: at most 8 rows in at most 4 slots - the shapes the budget allows (a few dozen); the plain
  M = 1 list stays today's for `--batch 1` (spec 13 §8, §9.2).
- **Commit and argmax_stage2:** the in-list `pos += n` / `cur_token[0] = out[n - 1]` are ignored by a row list's host
  (P5); per slot the host commits as Engine::commit does (pos + j + 1, live index, hh row 0 <- x[r0 + j]).
- **Drafts across slots** (P7): one M = S draft list over the slots' rows is the open design question 13b decides with
  the measured draft cost (it reads the head and lm_head once per draft index for all slots instead of once per slot).

## 7. What the box run (row 66) decides

| decision | read from | the rule (proposed; the operator rules) |
|---|---|---|
| the slots offered by `--batch` per model (Qwen3.8 up to 4; Agnes 2 with MTP, 4 with `--mtp off`) | `r66.cost`'s `B x 1` arms | keep a slot count when the aggregate (B / the measured x `1x1@4k`) is >= 1.6x at B = 2 and >= 2.0x at B = 4 (spec 13 §5's bars) at 4k, and the mixed `@32k+4k` arm within 10 % of the equal-depth one; otherwise the largest B that meets them |
| the row budget's allocation | the MTP-shaped arms (`2x4`, `4x2`, `1x4+3x1`, Agnes `2x4` / `2x2` / `1x4+1x1`) | every active slot gets its pending row first; the remaining rows of the 8 go where the measured ids / cost is highest (spec 13 §9's "the longest-running main session first" if `1x4+3x1` beats `4x2`, else K = 1 everywhere); a slot's K never exceeds its AdaptiveK |
| the interleave chunk (`SchedulerOptions::chunk`) | `r66.chunk`, with `1x1@4k` / `1x1@32k` | the smallest of 512 / 1024 / 2048 whose ms / id is within 15 % of 2048's (prefill throughput kept), and smaller still if its worst gap (chunk + step at 32k) exceeds the operator's bound (proposed 500 ms) |
| interleave vs batched admission (spec 13 §8) | `r66.chunk` and the mock's ticks | interleave unless a burst's batched prefill beats N interleaved chunks by more than its gap cost |
| the row build's overhead | `rows 1x1@32k vs production verify(0)` | <= 1.02 (spec 13 §5's "a single request within 2 %"); above it, 13b keeps the production M = 1 list for a lone request (it does anyway at `--batch 1`) |
| the check | `r66.check` | every arm PASS on both models and KV forms; a FAIL stops 13b until the row build is fixed |
