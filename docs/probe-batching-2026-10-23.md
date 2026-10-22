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
