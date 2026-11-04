# Spec 28m - the MoE engines at M rows: one probe per engine (expert GEMVs, distinct experts, verify cost)

**Status (2026-11-01): DESIGN, reviewed 2026-11-01 (corrections listed at the end); operator decisions of 2026-11-01 in
§7 - nothing built, no box work done.** Operator, 2026-11-01: "spec the MoE probe for all engines". This is the
follow-on spec 28 §8 left open ("the MoE expert GEMVs ... a follow-on per engine once 28b's numbers exist"). 28a's record (docs/probes/probe-small-m-gemm-2026-11-01.md, provisional) gives the dense numbers this builds
on: the dense GEMMs at M = 8 cost 1.08x the M = 1 sum with the stand-in kernel. Written Mac-only, from the code and docs.

**Plan:** `docs/superpowers/plans/2026-11-01-spec28m-moe-small-m-probe.md` (one probe tool, per-engine stages
described in prose). **No box queue row** is allocated (the operator has not asked; the next free row is 77: rows
68-75 are the speed track's, 76 the sycl-tla removal). **Name:** 28m sits beside 28a / 28b / 28c (the Qwen speed
track); the next free spec number is 29.

**Grades** as spec 28: **measured**, **derived** (arithmetic shown), **estimated** (a judgement, formula shown),
**published by others**. Anything marked derived or estimated below is not a measurement.

## 1. Why, and what the code says today

Spec 28's `gemm_sm` fixes the dense projections. A MoE layer has a second cost the dense kernel does not touch: the
**routed experts**. In a verify step of M rows each row routes to its own top-k experts. The best any kernel can do is
read each *distinct* expert once; the extra cost per row is then set by how many distinct experts M rows touch.

What our decode expert kernels do (code read, all five engines): the row is grid dimension 1 (`m = get_group_id(1)`)
in every kernel. **gate||up / up** launches one work-group per (slot, 4 n-tiles, row) - grid (SLOTS x NT_UP / 4, M);
**down** launches one work-group per (16-column n-tile, row) - grid (HIDDEN / 16, M) - whose sub-groups cover all of the
row's slots and do the combine in SLM (K2's MoVA `k2_mova_value` likewise, grid (MOVA_N / 16, M)). Each work-group reads
its row's route row and then that row's experts' tiles. No kernel and no list builder dedupes across rows (`capture.cc`'s
`moe_block` just launches with `kCapM` rows): two rows that pick the same expert read its block twice (only the 24 MB
L2 can help). Where the shared expert is a kernel slot (Ornith, Flash-Next, K2, Kolibri) it is read once per row;
**Bolt's shared expert is not in `nb_moe.cl`** - it is a dense int4 GEMV (`fc1_latent || shared_up`, `shared_down`,
23.4 MB) that `gemv` already reads once for all M rows. The dedupe code in the tree is prefill's (`pf_moe*.cl`,
`q4_pf_moe.cl`, `k2_pf_moe.cl`, `kol_pf_moe.cl`, `nb_pf_moe*.cl`: sort by expert, one read per chunk). **Bolt's decode
union kernel** (`src/kernels/bolt/nb_moe_union.cl`, spec 26c, Decision 10) was dropped from the tree with commit 0219cc2
(2026-10-26, "drop 26c", operator 2026-10-27); it is §3's U2 already written once - see there.

| engine | expert kernel | inner loop | combine order | `FP_CONTRACT` |
|---|---|---|---|---|
| Ornith (Qwen capture) | `src/kernels/moe.cl`: `moe_route`, `moe_gate_up`, `moe_down` (residual fold) | int4 g64 layout 1 (136 u32 a tile), `tile_dot` / `dot8` | **rank** (descending p), fp32 sum, one rounding; gated shared | default (not set) |
| Qwen3.8-Flash-Next | `qwen4exp/q4_moe.cl` (`Q4_EXPERT_GU/DN` is spec 22's indirection hook) | moe.cl's `tile_dot`; shared slot int4 (`_SH4`) or bf16 (`_SHB`, Intel's interim checkpoint) | **rank**, fp32; gated shared | OFF, whole file |
| K2-Horizon | `k2/k2_moe.cl` (routed experts and MoVA value experts) | moe.cl's `tile_dot` | **ascending id**, bf16 chain (every add rounds); ungated shared | default |
| Kolibri | `kolibri/kol_moe.cl` (shared expert int4 layout 1, operator 2026-10-31, commit 1124546) | moe.cl's `tile_dot` | **ascending id**, fp32, one rounding; ungated shared | OFF from the down combine on |
| Bolt | `bolt/nb_moe.cl` (router `nb_router` is its own bf16 kernel; 9 routed slots, no shared slot) | moe.cl's `tile_dot`, non-gated relu2 | **ascending id**, fp32, one rounding | OFF, whole file |

So one union form written against `moe.cl` ports to the other four by family copy - the inner loop is the same source
everywhere, but the route row, the combine and the `FP_CONTRACT` regime are each family's own and the copy keeps them.

**Speculative decoding per engine** (code read):

| engine | verify lists (M >= 2) | drafters | max M |
|---|---|---|---|
| Qwen3.8 dense, Agnes | yes (`build_verify`, `build_verify_taps`) | MTP, DFlash, lookup | 4 (MTP), 8 (DFlash) |
| **Ornith** | yes, built (536 launches at every M; `moe_M{2,3,4}_E256_T8_D2048_I512`); **has run on a B70 for correctness** (row 16's card stages in the 2026-10-23 sessions; r16.prefix with `--mtp 3`, 2026-11-01) - **no verify cost recorded** (r16.cost / r16.mtp_rows are postponed speed stages; BENCHMARKS "Ornith 1.5 MoE": nothing measured) | MTP head (one MoE layer), lookup; DFlash refused | **4** (`kMaxK` 3) |
| **Flash-Next** (qwen4exp) | yes, built (790-795 launches at M = 1..4, 48 layers); no card run recorded (r34.mtp pending) | MTP head, lookup | **4** (`kMaxDraft` 3) |
| **K2-Horizon** | **none** (kM = 1; `--mtp`, `--spec lookup`, dflash refused by name; EAGLE3 probed NO-GO) | none; prompt lookup is its named path | 1 |
| **Kolibri** | **none** (kM = 1; no MTP head) | none | 1 |
| **Bolt** | **none** since 2026-10-26 (commit 0219cc2 dropped the verify lists, Mamba row slots, union kernel, `--spec` on Bolt) | none | 1 |
| Flint | nothing built (spec 27 is a watch) | none published | - |
| Nanbeige | **not in the repo** (zero hits in docs, src, tools, tests); not MoE; no plain-Llama path today | - | - |

Consequence: today's verify cost can be measured only for Ornith and Flash-Next (Ornith's lists already run on the card;
the existing `probe_mtp_steps` on Ornith - stage r16.cost - times its verify at M = 1..4 today). For K2, Kolibri and
Bolt the probe measures the expert kernel at M rows in isolation and prices the step from bytes; it cannot attribute a
real list.

## 2. The bytes model

Per MoE layer: k = routed experts per row, E = routed experts, B = bytes per expert (int4 g64 layout 1), D(M) =
distinct experts over M rows.

- **Expert bytes (int4 g64 layout 1):** a [N][K] linear is N x K / 2 B of nibbles + 2 x N x K / 64 B of f16 scales
  (`eagle3_cost.py`'s `int4g64`). A gated expert is gate||up [2I][H] + down [H][I]; Bolt's non-gated latent expert is up
  [I][1024] + down [1024][I] = 1088 x I B.
- **Plain step (M = 1):** k + 1 expert reads (the shared expert; Bolt: k, its shared expert is a dense GEMV) plus the
  router.
- **Verify, today's kernels (upper bound, no L2 help):** M x (k + 1) reads (Bolt: M x k).
- **Verify, the ideal kernel:** D(M) + 1 reads (the shared expert once; Bolt: D(M)).
- **iid prior (estimated):** D(M) = E x (1 - (1 - k/E)^M).
- **Real routing is more local than iid - measured once, on K2's CPU reference.** docs/probes/probe-eagle3-k2-2026-10-08.md
  (`tools/oracle/eagle3_cost.py`, **measured** from routes recorded by `k2_taps.py` - the dequantised int4 checkpoint run
  by `k2_ref` on the Mac, **not the engine on the card**; golden prompts only (prose / code / cjk, up to 128 greedy ids
  each, 267 anchors; the A4 set was not in the run); the dumps are on the Mac in `oracle-out-eagle3-k2/dumps/`, not in the
  repo): M consecutive rows touch 12.46 / 15.75 / 18.55 distinct MoE experts per layer at M = 2 / 3 / 4 against iid's
  15.36 / 22.13 / 28.36 - **0.81x / 0.71x / 0.65x** of iid (the ratios are derived from those two measured / formula
  columns; MoVA: 7.16 / 9.70 / 11.96 against 7.75 / 11.27 / 14.56). Consecutive accepted rows stand in for a verify's
  rows (a verify also holds rejected drafts). No other model has routing statistics: Ornith's would come from its card
  runs (Task 2); `tools/spec/expert_union.py` (spec 26a's union counter) was dropped with LiLiCorr in commit 99adbb0 and
  its box tables never ran; Bolt's "u = 30 at M = 8" is an estimate (spec 26 §4).

Per engine at M = 4 (B by the formula above from each descriptor's widths - **derived**; iid by the formula; "loc"
applies K2's 0.65x of iid to the routed experts as a **guess** for the same kind of text, not measured for these models;
K2's own "loc" column is its measured 18.55). Reads in expert units; the ratio is against the plain step:

| | E / k / H / I | B (derived) | today (upper) / ideal iid / ideal loc | shared expert |
|---|---|---|---|---|
| **Ornith** | 256 / 8 / 2048 / 512 | 1,671,168 B (1,114,112 + 557,056) | 36 / 31.5 / ~21 (plain 9): 4.0x / 3.5x / 2.3x | 1, gated, a kernel slot |
| **Flash-Next** | 512 / 10 / 2560 / 640 | 2,611,200 B (1,740,800 + 870,400) | 44 / 39.8 / ~26 (plain 11): 4.0x / 3.6x / 2.4x | 1, gated, a kernel slot (int4 or bf16) |
| **K2** (MoE) | 100 / 8 / 2560 / 768 | 3,133,440 B (2,088,960 + 1,044,480) | 36 / 29.4 / 19.6 (18.55 **measured** + shared; plain 9): 4.0x / 3.3x / 2.2x | 1, ungated, a kernel slot |
| **Kolibri** | 384 / 6 / 2560 / 512 | 2,088,960 B (1,392,640 + 696,320) | 28 / 24.4 / ~16 (plain 7): 4.0x / 3.5x / 2.3x | 1, ungated, int4, a kernel slot |
| **Bolt** | 240-496 / 9 / latent 1024 / 1536-2560 | 1,671,168-2,785,280 B (1088 x I) | **routed only:** 36 / 34.0-35.0 / ~22-23 (plain 9): 4.0x / 3.8-3.9x / 2.5x | dense GEMV (4096 -> 5376 -> 4096, relu2, 23,396,352 B), already read once at M rows |

Reading it (estimated; formulas above):

1. **At large E the iid gain is small, and most of it is the shared expert.** Reads saved at M = 4, iid: Ornith 12 %
   (36 -> 31.5; routed alone 4.6 %), Flash-Next 9 % (routed 2.9 %), Kolibri 13 % (routed 2.3 %), Bolt 3-6 % (routed only:
   its shared expert is already dense); K2, at E = 100, 18 % (routed 11 %). Certain parts: the **shared expert read once,
   not M times** (where it is a kernel slot). The rest is locality, which only measurement gives.
2. **Even the ideal kernel does not make the MoE part flat:** with locality an MoE layer is still ~2.2-2.5x a plain
   step at M = 4. The dense part (spec 28) goes flat; this probe decides how far above that floor today's kernel sits.
3. **Ornith's whole step (derived, bytes only):** a MoE layer at M = 1 reads 16.15 MB (the bf16 router || shared gate,
   272 x 2048 x 2 = 1,114,112 B, plus 9 x 1,671,168 B), 40 layers = 0.646 GB of the ~1.86 GB a token reads (spec 15 §10;
   the rest, 1.21 GB, is the int8 head, the attention / GDN projections, norms). The router is one GEMV for all M rows, so
   the layer ratio is r = (1.114 + reads x 1.671) / 16.15 MB: 3.79 today (36 reads), 3.33 ideal iid (31.5), 2.24 ideal
   with the locality guess (~21). With the dense part at 1.1x (28a's Qwen3.8 GEMM-sum ratio at M = 8, **estimated** for
   Ornith's smaller shapes), verify(4) = (1.21 x 1.1 + 0.646 x r) / 1.86 plain steps: **2.0** with today's kernel's upper
   bound, **1.9** ideal iid, **1.5** ideal with locality. Attention / GDN state and KV bytes are not in it, and the step
   is launch-heavy at M = 1 (526 launches), so these are bytes-only bounds, not predictions.
4. **K2:** bytes-only verify with the measured routes is 1.26 / 1.46 / 1.63 plain steps at M = 2 / 3 / 4 (derived,
   probe-eagle3-k2: measured MoE **and** MoVA unions, shared expert once, int8 head, depth 0). Today's per-(row, slot)
   kernels read more at M = 4 (estimated, no L2 help, 45 sparse layers): routed 45 x (32 - 18.55) x 3.133 MB = 1.90 GB,
   the shared expert 45 x 3 x 3.133 MB = 0.42 GB, MoVA 45 x (16 - 11.96) x 1.393 MB = 0.25 GB - **2.57 GB** over the
   5.13 GB the measured-routes model reads, i.e. **~2.45 plain steps** (7.70 / 3.145 GB) against 1.63. K2 has the largest
   gap between today's kernel and the ideal of any engine.

## 3. The probe: one tool, one question per engine

**`probe_moe_small_m`** (`tools/probe/`): stand-in expert kernels of the **union** form, each engine's shapes as
compile-time parameters (one source, as `probe_small_m` is), today's expert kernels at M = 1..8 as the control
(probe-only builds where an engine stops earlier, as 28a did at M = 16).

**Stand-in forms** (timed in this order; measure, do not assume):

- **U1 - route-sorted order (gate||up only):** the work-group order cannot be set from outside a kernel, so U1 is today's
  gate||up with one indirection: work-group g reads its (row, slot) from a permutation table sorted by expert id
  (written by a variant of the route step, or a tiny extra launch over the M x k route words), so pairs that share an
  expert are dispatched adjacently and the second read can hit L2. Down is unchanged (its work-group owns all of a
  row's slots, so there is nothing to sort). **Bitwise equal to today's kernel** (no row's arithmetic changes). It
  measures what L2 alone recovers.
- **U2 - one work-group set per distinct expert:** **this is spec 26c's `nb_moe_union.cl`** (in git at `0219cc2^`; Mac
  run `nb_union_run` on UHD 630: bitwise the plain M-row form in 10 route cases; never run on a B70 - row 65 was
  dropped with it). Every work-group derives the union schedule alone, in SLM, from the M route rows (an M x k (row,
  slot) list, first occurrences, distinct experts ranked by id); work-group set u < M x k takes the expert of rank u and
  every (row, slot) pair of it, loading each weight tile and scale once and running each pair's `tile_dot` chain
  against its own activation row; sets past the union's size exit at once. Up / gate||up writes h[row][slot] as today;
  down writes each pair's bf16 output d[row][slot][n] (today's kernel rounds it to bf16 before the combine too, so this
  is exact), and a **third launch** does each row's combine in today's order - rank order with the gated shared expert
  and the residual fold for Ornith / Flash-Next, the ascending-id chain for K2 / Kolibri / Bolt - plus the shared expert
  where it is a slot. Fixed grid per M (one captured list per M), no host sync, no data-dependent launch count; **+1
  launch a MoE layer** (Ornith: +40 a verify list; the MTP head's MoE layer runs in the M = 1 draft list) and an M x SLOTS x H bf16 scratch (Ornith M = 4: 147
  KB). **Bitwise equal to today's kernel per row** if the per-pair chain is today's: the dot8 products (int4 x bf16) are
  exact in fp32, so contraction cannot change them, but `acc += gacc * scale` in `tile_dot` is contractible - in
  `moe.cl` and `k2_moe.cl` (no `FP_CONTRACT` pragma) whether it becomes a mad is the compiler's choice per code shape, so
  the U2 copy must either reproduce today's choice (checked in the ISA, Task 3) or the bitwise claim is tested, not
  assumed. Bolt's copy ran under `FP_CONTRACT OFF`, which is why it was bitwise by construction there.
- **U3 - DPAS over the rows of an expert** (`gemm_sm`'s form, rows on A): different accumulation order, so spec 28's
  N2 / N3 bars (bound, cosine, row independence), not bitwise. Timed only if U2 leaves the experts compute-bound
  (unlikely: an expert sees 1-3 rows at M = 4, so the 8-row DPAS block is mostly padding).

**Property R-moe:** row m's output does not depend on the other rows or on M (M >= 2). For U1 / U2 the per-row
arithmetic and the family's combine order (rank order for Ornith / Flash-Next, ascending id for K2 / Kolibri / Bolt) are
today's; only who loads the weights changes - so it holds if U2 is bitwise per row (above), and is held by test (10^3
trials per shape; today's `moe_m_test` - moe.cl at M = 2..4, every row bitwise M = 1's, rows reversed - is the
pattern). The router kernels are untouched; the route words are bitwise today's.

**For the build plan, not this probe** (the probe changes nothing in `src/`): every list built at M = 1 binds today's
expert binaries (U2 adds a launch, so M = 1 keeps today's list); M >= 2 binds the union kernel through one capture rule
(like `gemv_kind(M)`); a test-only capture option forces either. No CLI flag, no environment variable. The capture
(Level Zero command lists recorded once and replayed) already carries in-kernel early exits on a fixed grid -
`k2_mova_value`'s `m >= n_act` return - so U2's exit-past-the-union needs nothing new from it.

**What is measured, per engine:**

1. **Routing statistics** (where a verify list or real routes exist): D(M) for M = 2..4 per layer on the golden
   prompts and the A4 tool-call set, as a ratio to iid (the locality factor). Read from the engine's route buffers
   after the list runs (Ornith's MoE scratch keeps every layer's route rows for the step; the probe owns an engine as
   `probe_verify_m` does): no engine change. K2: the Mac's recorded CPU-reference routes, replayed.
2. **The expert block** (route + gate_up + down + shared + fold, all layers) at M = 1..4 (to 8 where lists reach it):
   today's kernel against U1 / U2 (/ U3), interleaved pairs after a warm-up, median of 5, on routes (a) recorded,
   (b) iid synthetic, (c) all rows distinct (worst) and all rows equal (best) - the bounds of the union gain.
3. **The attribution** (Ornith, Flash-Next): the real verify list at M = 1..4 split into GEMV, experts, attention,
   other; shares applied to the unprofiled step (as 28a). For Ornith reuse what exists: `probe_mtp_steps <ornith>`
   (r16.cost) for the unprofiled verify cost, and `probe_verify_m --verify8 1`'s profiled mode (the Qwen capture, so it
   builds Ornith's lists) capped at M = 4 with an "experts" family for `moe_*` names - two small probe edits.
4. **Numerics:** each form against today's kernel (bitwise for U1 / U2) and against an fp64 expert reference (N2's
   bound); property R-moe.

## 4. The go rule (per engine; thresholds decided by the operator 2026-11-01, §7.1)

For an engine with a verify list (Ornith, Flash-Next), at M = 4 (MTP's maximum):

- **GO** (the build plan may be written, the operator asked) when ALL hold:
  (a) **the expert block:** the union form at M = 4 is **>= 10 % below today's** on the recorded routes (measured;
      10 % = the 3 % noise line plus margin) and within 25 % of the bytes model's ideal for the measured D(4) (derived:
      ideal(4) = today's measured M = 1 block x (D(4) + 1) / (k + 1), the M = 1 block's time scaled by the expert
      reads; for Bolt D(4) / k);
  (b) **the whole step:** derived verify(4) = measured non-expert time + (a)'s expert block, dense part at 28b's cost,
      **<= 1.6 plain steps** of the engine's own M = 1 step (derived from measured parts; Qwen3.8's verify(4) is 1.74
      today, measured 2026-09-27; Ornith's bytes-only ideal-with-locality is ~1.5, §2.3, so 1.6 leaves little room for
      launches and attention - the operator's band 1.6-1.9 is where most outcomes are expected);
  (c) **numerics:** the form passes §3's bars (bitwise for U1 / U2).
- **The operator's call:** (b) between 1.6 and 1.9.
- **NO-GO** (recorded, the engine's MoE stays at today's kernel): (c) fails, or (a) misses 10 %, or (b) above 1.9.
  The record names the next lever (launch count, attention, the MTP head's MoE layer).
- **Recorded, not gating:** U1 vs U2 vs U3; the shared-expert saving alone; D(M) per prompt class; expert-block times
  at M = 5..8.

For an engine **without** a verify list (K2, Kolibri, Bolt, Flint): the verdict is **(a) and (c)** on its shapes plus a
bytes-model (b) with inputs labelled derived. It is a go for **building a verify list** only if the operator wants a
drafter there (K2: lookup; the others have none). The probe never builds a list.

## 5. Per engine

Order by expected value for the target workload (agentic coding): **Qwen3.8 is the served model** (docs/03-models.md,
"the model everything is tuned for"; the only one with speed numbers on the card) and is dense, so spec 28 covers it.
Every engine below is second tier and **has no speed number on a B70**; Ornith and K2 have run their correctness rows on
the card (the 2026-10-23 sessions, rows 10 / 13 / 16 and 14 / 15 / 21 / 25), Kolibri its first judged run (2026-10-31);
Flash-Next and Bolt have no card run recorded. Order decided by the operator 2026-11-01 (§7.2).

| order | engine | verify list | probe | worth running |
|---|---|---|---|---|
| 1 | Ornith | yes (M <= 4), already run on the card | full | **yes** - the only MoE engine on the Qwen capture with MTP and lookup built and run; dense part rides 28b (§7.4); the union kernel ports by copy |
| 2 | K2-Horizon | no | kernel + routing replay | **conditional, after Ornith's numbers** (§7.2-7.3) - the only measured routing data (CPU reference); largest gap between today's kernel and the ideal (~2.45 against 1.63 plain steps at M = 4, §2.4); matters only if a verify list (lookup) is wanted; lookup wins on code (S 1.18-1.68 at depth 4096, probe-eagle3-k2) |
| 3 | Flash-Next | yes (M <= 4) | full, truncated model | **later** - served only with `--layers N`; iid saves 9 % of reads at E = 512 (2.9 % of the routed ones), so value is the shared expert and locality |
| 4 | Bolt | no (removed) | shape rows | **no separate run** - nothing to verify; its 12 (E, I) pairs are rows of the shared probe; the smallest gain (routed only, iid 3-6 %) |
| 5 | Kolibri | no | shape rows | **no separate run** - same |
| - | Flint | nothing built | none | **no** - an architectural near-twin of Kolibri, not a kernel-level one (spec 27: hidden 2048, 256 experts top-6, expert width 512, 2 dense + 47 MoE; Kolibri's binaries and runtime are built at 2560 / 384 experts), so it needs its own descriptor and builds first. Its routed expert shape (H 2048, I 512, E 256) is Ornith's, so Ornith's rows at k = 6 cover it if wanted |
| - | Nanbeige | not in the repo | none | not MoE, no engine path; if added as plain dense Llama it would use `gemv.cl` and the Qwen capture, and 28b's shape list (Qwen3.8 + Agnes) would need its shapes |

### 5.1 Ornith (order 1)

Question: **how far above the ideal do `moe_gate_up` / `moe_down` sit at M = 2..4, and does a union form bring MTP's
verify(4) to <= 1.6 plain steps?**

| item | value |
|---|---|
| path | the Qwen capture (`capture.cc` branches on `is_moe()`); per layer router `gemv_bf16` (272 rows), `moe_route`, `moe_gate_up`, `moe_down` (residual fold); 526 launches a token, 536 a verify list at every M |
| MoE shapes | hidden 2048, E 256, k 8, inter 512; gate_up K 2048 x N 1024, down K 512 x N 2048; shared = slot 8 |
| dense shapes (28b) | int4 at M = 2..4 (spec 15e's CMake block): qkv||z 2048 -> 12288 (L1), out_proj / o_proj 4096 -> 2048, q||k||v 2048 -> 9216, a||b 2048 -> 128 (L1); int8 head 2048 -> 248320 (argmax masked from 248077); bf16 router 2048 -> 272 and the MTP head's bf16 linears stay `gemv_bf16` - **to be added to 28b** (operator 2026-11-01, §7.4) |
| M | 1..4 (MTP K <= 3, lookup K <= 3) |
| measured today | correctness only (rows 10 / 13 / 16 on 2026-10-23; r16.prefix with `--mtp 3` on 2026-11-01); no time (BENCHMARKS "Ornith 1.5 MoE": every row pending; r16.cost postponed) |
| expected upside | MoE-layer reads at M = 4: 36 -> 31.5 (iid) / ~21 (locality guess); verify(4) 2.0 -> 1.9 / 1.5 plain steps (estimated, bytes only, §2.3) |
| 28b reuse | dense parts: yes, `gemm_sm` once Ornith's shapes are added; routed experts: **no** (layout-1 tiles); router and head as in 28b |
| needs the card | routing statistics (real weights), all timing |

### 5.2 K2-Horizon (order 2)

Question: **how much more than the measured-routes bytes model (1.26 / 1.46 / 1.63 plain steps at M = 2 / 3 / 4)
does a per-(row, expert) kernel cost, and what does a union kernel get back?**

| item | value |
|---|---|
| path | own engine `runtime/k2`; dense int4 linears and sparse-attention GEMV on `gemv.cl` (layout 0), router and bf16 head `gemv_bf16`, int8 head `gemv_i8w`; experts and MoVA in `k2_moe.cl`; attention `k2_attn.cl`; 717 launches a token |
| shapes | hidden 2560; 45 sparse layers + 3 dense MLP (6144); 100 experts top 8, inter 768, shared 768; MoVA 64 value experts top 4, 2560 -> 1024 |
| M | 1 only today; probe M = 1..4 (8 for lookup's longest draft) |
| measured today | routing locality on the CPU reference's routes (§2); correctness rows on the card; **no decode or verify time** (BENCHMARKS: nothing measured) |
| expected upside | today's kernels at M = 4 read ~2.57 GB more than the measured-routes model (routed 1.90, shared 0.42, MoVA 0.25): ~2.45 against 1.63 plain steps (estimated, bytes only, §2.4) |
| 28b reuse | none until a K2 verify list exists (`static_assert(kM == 1)` in `k2_capture.cc`: nothing would bind a `gemm_sm` binary); experts and MoVA: no |
| precondition | a verify list for K2 (lookup's) - **not built**, not built here |
| needs the card | all timing; the routing replay runs on the Mac (`oracle-out-eagle3-k2/dumps/`) |

### 5.3 Flash-Next / qwen4exp (order 3)

Question: **is routing local enough at E = 512, k = 10 for a union form to matter (iid says 9 % of reads, 2.9 % of the
routed ones)?**

| item | value |
|---|---|
| path | own engine `runtime/qwen4exp`; int4 dense GEMVs through `gemv_variant`, `q4_moe.cl` experts, own HC / PLE / QSA kernels; MTP head with its own 512-expert MoE layer |
| shapes | 48 layers (36 GDN, 12 QSA), hidden 2560, E 512, k 10, inter 640 (~2.61 MB), shared 640 gated |
| M | 1..4 |
| measured today | nothing |
| expected upside | MoE-layer reads at M = 4: 44 -> 39.8 iid, ~26 only if locality is K2's (estimated) |
| 28b reuse | GDN `qkvz` / `out_proj` int4: only with a second `gemv_kind` call site (`qwen4exp_capture.cc` binds `gemv` itself) - a follow-on, not part of 28b (§7.4); experts: no |
| status | served only truncated; the whole model waits for spec 22's tier. **After Ornith, same tool, only if the operator wants Flash-Next's MTP** |

### 5.4 Bolt, Kolibri

| engine | what the probe does | why no more |
|---|---|---|
| Bolt (88 blocks: 40 Mamba, 40 MoE, 8 attention) | shape rows: the 12 (E, I) pairs (E 240..496, I 1536..2560, k 9, latent 1024) through the stand-in (U2 = the dropped `nb_moe_union.cl` itself); Mamba in_proj 4096 -> 18560 and out_proj 8192 -> 4096 as 28b-form dense shape rows | verify lists, per-row Mamba slots and the union kernel were dropped 2026-10-26 (0219cc2); Mamba verify needs a state per row, a different problem (spec 25 §6.1's stale verify text marked, §7.6) |
| Kolibri-1 (50 MoE layers, E 384, k 6, inter 512, hidden 2560) | shape rows: `kol_moe.cl`'s shapes, int4 shared expert; dense q||k||v and o_proj on `gemv.cl` are 28b-form shape rows | no MTP, no drafter, M = 1 only; two cards; correctness runs on the card since 2026-10-31, no speed number |

## 6. Rules that bind every stage

- **Timing:** interleaved control / candidate pairs after a warm-up, median of 5 (kernels) / 3 (steps); the 3 % line is
  noise. **One-card, busy-box timing is allowed but labelled "busy-box timing (one card), to be re-checked on a quiet
  box"** (operator, 2026-11-01); the record lists the re-check set. Grades on every number.
- **M = 1 bit for bit** (G0 checksums). This probe: `tools/kernel_cmdlines` +0 / -0 / ~0 for `src/` (stand-ins and
  probe-only builds live in `tools/probe`). A later build plan: only union-kernel lines added, +N / -0 / ~0.
- **No new CLI flag or environment variable** for production selection: the capture rule, a test-only capture option.
- **No host-RAM offload on the per-token path** (PCIe 3.0 box). Nothing here uses it; spec 22's tier is untouched.
- **256-GRF builds** carry the option on the ocloc line; the probe reads the built binary's GRF mode before a timed arm.
- Which experts repeat never changes a launch count: fixed grid, early exit.

## 7. Decisions (operator, 2026-11-01 - all decided)

1. **The thresholds of §4: as proposed.** (a) >= 10 % below today's expert block at M = 4 on recorded routes (and within
   25 % of the bytes ideal); (b) <= 1.6 plain steps GO, 1.6-1.9 the operator's call, > 1.9 NO-GO; (c) bitwise for U1 / U2,
   N2 / N3 for U3.
2. **Order and scope:** **Ornith first**; **K2 conditional, decided after Ornith's numbers**; then **Flash-Next**. **Bolt
   and Kolibri: shape rows only.**
3. **K2 speculative decoding** (a lookup verify list - the precondition for K2's probe to matter): **decided after
   Ornith's numbers.**
4. **Ornith / K2 / Flash-Next dense shapes in 28b: to be added.** Feasibility (review, 2026-11-01): **Ornith is a small
   addition** - its M >= 2 lists go through `capture.cc`'s `gemv()` / `gemv_i8w()` helpers, so 28b's `gemv_kind(M)`
   covers them with no capture change; what 28b gains is CMake lines for Ornith's four int4 shapes and its int8 head at
   M = 2..4 (§5.1), their rows in the per-shape S table and in `gemm_sm_test`, and one gate restated: r16.mtp's M2 holds
   Ornith's verify route rows bitwise to the plain rows for k = 1..3, which `gemm_sm`'s different rounding at M >= 2 ends
   - it becomes N4 + N3 like Qwen3.8's D3 (28b Review Focus 4), with the MoE f32-judge near-tie rule for flipped routes.
   **K2: nothing to add yet** - K2 has no M >= 2 list (`static_assert(kM == 1)`), so a `gemm_sm` binary at its shapes
   would bind nowhere; it goes in with the K2 verify list if decision 3 says so. **Flash-Next: not a small addition** -
   its own capture binds `gemv` directly (`qwen4exp_capture.cc`), so it needs a second `gemv_kind` call site, its shapes
   and r34's gates restated, on a truncated model; a follow-on task after 28b.
5. **Prompt set for routing traces:** the golden prompts plus the A4 tool-call set.
6. **Stale text:** fixed in this review (2026-11-01) - spec 25 §6.1's verify mentions marked as dropped; spec 20's
   shared-expert-bf16 passages marked superseded by the int4 ruling of 2026-10-31 (commit 1124546).

## 8. Out of scope

M > 8; building any verify list (U2 for production is a build plan after GO); Mamba row state slots (Bolt); the expert-offload tier (spec 22); acceptance and
drafting policy (specs 8, 19); the attention of MoE engines (28c's shape rule, per engine, later); prefill MoE GEMMs
(already per-expert grouped).

## 9. Verdict

Not run. Written by the plan's Task 6 when the card has run the stages.

## Review corrections (2026-11-01)

Checked against the code, git and the records (Mac only); changed in place above:

1. **Ornith's verify lists have run on a B70** (correctness: row 16's card stages, 2026-10-23 sessions; r16.prefix with
   `--mtp 3`, 2026-11-01) - "never run" removed. What is true: no MoE engine has a speed number on the card (BENCHMARKS), and
   Ornith's verify cost has an existing, postponed stage (r16.cost, `probe_mtp_steps`). Ornith, K2 and Kolibri have run
   correctness rows; "every engine below has never run on a B70" corrected (§1, §5).
2. **Kernel structure:** only gate||up is one work-group per (row, slot); down (and K2's MoVA) is one per (row, n-tile)
   with all slots inside. Nothing dedupes - confirmed for all five kernels and the list builders. **Bolt's shared expert
   is a dense GEMV, not a kernel slot**, so it is already read once at M rows; Bolt's union gain is routed-only (§1, §2).
3. **"moe.cl's, verbatim"** holds for the `tile_dot` inner loop only: combine order is rank order for Ornith /
   Flash-Next and ascending id for K2 / Kolibri / Bolt (the old text said "the ascending-id combine" for all), and the
   `FP_CONTRACT` regime differs per file (§1 table, §3).
4. **0219cc2** confirmed (2026-10-26, "drop 26c", operator 2026-10-27). Added: the dropped `nb_moe_union.cl` **is U2**,
   already written and bitwise on the Mac (spec 26 §14) - the probe's U2 starts from it (§3).
5. **K2's locality** is measured on routes recorded from the **CPU reference** (`k2_ref` on the dequantised int4
   checkpoint, golden prompts only, 267 anchors), not the engine; the 0.81 / 0.71 / 0.65 ratios are derived from that
   and the iid formula; the dumps are on the Mac (`oracle-out-eagle3-k2/dumps/`), not the box. `expert_union.py` existed
   and was dropped in 99adbb0 (§2, §5.2).
6. **Bytes model:** expert bytes now shown by formula (all five verified); K2's B is 3.133 MB (not ~3.2); **K2's iid
   at M = 4 is 29.4 reads, not 27.6** (3.3x, not 3.1x); **Flash-Next's iid is 39.8, not 40.9**; Bolt's row recounted
   without a shared slot (36 / 34-35 / ~22-23, plain 9); Kolibri's iid gain is 13 %, not under 10 %; Flash-Next's "~7 %"
   is 9 % (2.9 % routed). Ornith's step now counts the router once for all rows: r = 3.79 / 3.33 / 2.24, verify(4) =
   **2.0** / 1.9 / 1.5 (was 2.1 / 1.9 / 1.5). K2's today-kernel excess now includes the shared expert and MoVA: 2.57 GB,
   ~2.45 plain steps against 1.63 (§2).
7. **Go rule:** thresholds unchanged (operator); "within 25 % of the bytes ideal" given a formula; (b)'s 1.6 set against
   Ornith's ~1.5 bytes floor. Consistent with 28a's method (interleaved pairs, 3 % noise, labelled busy-box one-card
   timing, M = 1 bit for bit, no flags / env switches, no host-RAM offload); the M = 1 / capture-rule bullets marked as
   the build plan's, since this probe changes nothing in `src/` (§4, §6).
8. **U1 / U2 / U3:** U1 cannot be "same kernel and grid" - it needs a permutation indirection and applies to gate||up
   only. U2 needs a third launch (the per-row combine) because down's work-group owns a whole row today; bitwise per row
   holds only if `tile_dot`'s contractible `acc += gacc * scale` compiles as today's (moe.cl / k2_moe.cl set no
   `FP_CONTRACT`) - tested and ISA-checked, not assumed. Fixed grid + in-kernel early exit is realistic with our capture
   (`k2_mova_value`'s `n_act` exit is the precedent) (§3).
9. **Flint / Nanbeige** confirmed: Flint is an architectural near-twin of Kolibri, not a kernel-level one (H 2048, E 256
   against 2560 / 384); its routed expert shape equals Ornith's. Nanbeige: zero hits in the repo (§1, §5).
10. Small fixes: Ornith's int8 head binary is N 248320 (argmax masked from 248077); K2 gets nothing from 28b until it has
    an M >= 2 list; Flash-Next's 28b reuse needs a second `gemv_kind` call site; Ornith's attribution reuses
    `probe_mtp_steps` / `probe_verify_m`.
11. §7 rewritten with the operator's decisions of 2026-11-01, including the 28b feasibility note; spec 25 §6.1 and
    spec 20's stale passages fixed (§7.6).
