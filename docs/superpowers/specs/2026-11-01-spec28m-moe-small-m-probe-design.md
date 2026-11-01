# Spec 28m - the MoE engines at M rows: one probe per engine (expert GEMVs, distinct experts, verify cost)

**Status (2026-11-01): DESIGN - nothing built, no box work done.** Operator, 2026-11-01: "spec the MoE probe for all
engines". This is the follow-on spec 28 §8 left open ("the MoE expert GEMVs ... a follow-on per engine once 28b's
numbers exist"). 28a's record (docs/probe-small-m-gemm-2026-11-01.md, provisional) gives the dense numbers this builds
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

What our decode expert kernels do (code read, all five engines): **every one launches one work-group set per
(row, slot)** (`m = get_group_id(1)`), each reading its row's route row and then that expert's block. No kernel
dedupes across rows; two rows that pick the same expert read its block twice (only L2 can help). The shared expert,
the same for every row, is read once per row. The dedupe code that exists is prefill's (`pf_moe*.cl`,
`q4_pf_moe.cl`, `k2_pf_moe.cl`, `kol_pf_moe.cl`: sort by expert, one read per chunk). Bolt's decode union kernel
(`nb_moe_union.cl`) was dropped from the tree with commit 0219cc2.

| engine | expert kernel | inner loop |
|---|---|---|
| Ornith (Qwen capture) | `src/kernels/moe.cl`: `moe_route`, `moe_gate_up`, `moe_down` | int4 g64 layout 1 (136 u32 a tile), `tile_dot` / `dot8` |
| Qwen3.8-Flash-Next | `qwen4exp/q4_moe.cl` (`Q4_EXPERT_GU/DN` is spec 22's indirection hook) | moe.cl's, verbatim; the shared slot may be bf16 |
| K2-Horizon | `k2/k2_moe.cl` (routed experts and MoVA value experts) | moe.cl's, verbatim |
| Kolibri | `kolibri/kol_moe.cl` (shared expert int4 layout 1 too, operator 2026-10-31) | moe.cl's, verbatim |
| Bolt | `bolt/nb_moe.cl` + `nb_expert.h` (router is its own bf16 kernel) | moe.cl's, verbatim |

So one union kernel written against `moe.cl` ports by copy to the other four. **Speculative decoding per engine**
(code read):

| engine | verify lists (M >= 2) | drafters | max M |
|---|---|---|---|
| Qwen3.8 dense, Agnes | yes (`build_verify`, `build_verify_taps`) | MTP, DFlash, lookup | 4 (MTP), 8 (DFlash) |
| **Ornith** | yes, built (536 launches at every M), never run on a B70 | MTP head (one MoE layer), lookup; DFlash refused | **4** (`kMaxK` 3) |
| **Flash-Next** (qwen4exp) | yes, built, never run (790-795 launches at M = 1..4) | MTP head, lookup | **4** (`kMaxDraft` 3) |
| **K2-Horizon** | **none** (kM = 1; `--mtp`, `--spec lookup`, dflash refused by name; EAGLE3 probed NO-GO) | none; prompt lookup is its named path | 1 |
| **Kolibri** | **none** (kM = 1; no MTP head) | none | 1 |
| **Bolt** | **none** since 2026-10-26 (commit 0219cc2 dropped the verify lists, Mamba row slots, union kernel, drafter) | none | 1 |
| Flint | nothing built (spec 27 is a watch) | none published | - |
| Nanbeige | **not in the repo** (zero hits in docs, specs, src, tools); there is no plain-Llama path, the only dense family is `qwen3_5` | - | - |

Consequence: today's verify cost can be measured only for Ornith and Flash-Next. For K2, Kolibri and Bolt the probe
measures the expert kernel at M rows in isolation and prices the step from bytes; it cannot attribute a real list.

## 2. The bytes model

Per MoE layer: k = routed experts per row, E = routed experts, B = bytes per expert (int4 g64 layout 1), D(M) =
distinct experts over M rows.

- **Plain step (M = 1):** k + 1 expert reads (the shared expert) plus the router.
- **Verify, today's kernels (upper bound, no L2 help):** M x (k + 1) reads.
- **Verify, the ideal kernel:** D(M) + 1 reads (the shared expert once).
- **iid prior (estimated):** D(M) = E x (1 - (1 - k/E)^M).
- **Real routing is more local than iid.** The one measurement is K2's recorded routes (docs/probe-eagle3-k2-
  2026-10-08.md, **measured**; the routes themselves live on the box, not in the repo): 12.46 / 15.75 / 18.55
  distinct experts per layer at M = 2 / 3 / 4 against iid's 15.36 / 22.13 / 28.36 - **0.81x / 0.71x / 0.65x** of iid.
  No other model has routing statistics in the repo (Ornith's are box work of spec 15a; the LiLiCorr record's
  `tools/spec/expert_union.py` does not exist; Bolt's "u = 30 at M = 8" is an estimate, spec 26).

Per engine at M = 4 (B from descriptors and sweeps; iid by the formula; "loc" applies K2's 0.65x as a **guess** for
the same kind of text, not measured for these models). Reads in expert units; the ratio is against the plain step:

| | E / k | B | today (upper) / ideal iid / ideal loc | shared expert |
|---|---|---|---|---|
| **Ornith** | 256 / 8 | 1.671 MB (1,114,112 + 557,056) | 36 / 31.5 / 20.8 (plain 9): 4.0x / 3.5x / 2.3x | 1, gated, same for every row |
| **Flash-Next** | 512 / 10 | ~2.61 MB (derived: 64.17 GB / 24,576) | 44 / 40.9 / ~26 (plain 11): 4.0x / 3.7x / 2.4x | 1, gated |
| **K2** (MoE) | 100 / 8 | ~3.2 MB (320.3 MB a layer / 100) | 36 / 27.6 / 19.6 (18.55 measured + shared; plain 9): 4.0x / 3.1x / 2.2x | 1, ungated |
| **Kolibri** | 384 / 6 | 2.089 MB (1,392,640 + 696,320) | 28 / 24.4 / ~16 (plain 7): 4.0x / 3.5x / 2.3x | 1, ungated, int4 |
| **Bolt** | 240-496 / 9 (latent 1024) | 1.67-2.79 MB (1088 x I) | 40 / ~37 / ~24 (plain 10): 4.0x / 3.7x / 2.4x | 1 (on hidden, 5376) |

Reading it (estimated; formulas above):

1. **At large E the iid gain is small** (Ornith 36 -> 31.5 reads; Flash-Next, Kolibri, Bolt under 10 %). Certain parts:
   the **shared expert read once, not M times**, and the router. The rest is locality, which only measurement gives.
   K2's small E (100) has the largest dedupe even iid.
2. **Even the ideal kernel does not make the MoE part flat:** with locality an MoE layer is still ~2.2-2.4x a plain
   step at M = 4. The dense part (spec 28) goes flat; this probe decides how far above that floor today's kernel sits.
3. **Ornith's whole step (derived, bytes only):** 40 MoE layers x 16.15 MB = 0.646 GB of the ~1.86 GB a token reads
   (spec 15: the rest is the int8 head, attention / GDN projections, MTP head). With the dense part at 1.1x after
   spec 28, verify(4) = (1.21 x 1.1 + 0.646 x r) / 1.86 plain steps with r the MoE-layer ratio above: **2.1** with today's
   kernel's upper bound (r 4.0), **1.9** ideal iid (3.5), **1.5** ideal with locality (2.3). The step is launch-bound at
   M = 1 (526 launches), so these are bytes-only bounds, not predictions.
4. **K2:** bytes-only verify with the measured routes is 1.26 / 1.46 / 1.63 plain steps at M = 2 / 3 / 4 (derived,
   probe-eagle3-k2). A per-(row, expert) kernel reads more: MoE experts alone at M = 4 are 32 reads against 18.55
   measured; ~1.9 GB more on a 3.15 GB step (estimated: 45 sparse layers x (32 - 18.55) x 3.2 MB; MoVA's 64 value
   experts, top-4, ~1.4 MB each, have the same structure and are not in the sum).

## 3. The probe: one tool, one question per engine

**`probe_moe_small_m`** (`tools/probe/`): stand-in expert kernels of the **union** form, each engine's shapes as
compile-time parameters (one source, as `probe_small_m` is), today's expert kernels at M = 1..8 as the control
(probe-only builds where an engine stops earlier, as 28a did at M = 16).

**Stand-in forms** (timed in this order; measure, do not assume):

- **U1 - route-sorted launch order:** same kernel and grid; the (row, slot) work-groups are ordered by expert id (a
  tiny device sort of M x k route words) so rows sharing an expert run adjacently and hit L2. **Bitwise equal to
  today's kernel** (no row's arithmetic changes).
- **U2 - shared-weight work-group:** one work-group per *distinct* expert tile; it loads and dequantises the tile
  once and applies it to every row that chose the expert (<= M <= 8 rows), each row with **exactly today's per-row
  accumulation order**. Duplicate (row, slot) entries exit early, so the command list stays static (fixed grid of
  M x k slots, no host sync, no data-dependent launch count). **Bitwise equal to today's kernel per row.**
- **U3 - DPAS over the rows of an expert** (`gemm_sm`'s form, rows on A): different accumulation order, so spec 28's
  N2 / N3 bars (bound, cosine, row independence), not bitwise. Timed only if U2 leaves the experts compute-bound
  (unlikely: an expert sees 1-3 rows at M = 4).

**Property R-moe:** row m's output does not depend on the other rows or on M (M >= 2). For U1 / U2 it holds by
construction (per-row arithmetic and the ascending-id combine are fixed; only weight-load sharing changes); held by
test (10^3 trials per shape). The router stays `gemv_bf16`; the route words are bitwise today's.

**M = 1 bit for bit:** every list built at M = 1 binds today's expert binaries; M >= 2 binds the union kernel through
one capture rule (like `gemv_kind(M)`); a test-only capture option forces either. No CLI flag, no environment variable.

**What is measured, per engine:**

1. **Routing statistics** (where a verify list or real routes exist): D(M) for M = 2..4 per layer on the golden
   prompts and the A4 tool-call set, as a ratio to iid (the locality factor). Read from the engine's route buffers
   after the list runs (the probe owns an engine as `probe_verify_m` does): no engine change.
2. **The expert block** (route + gate_up + down + shared + fold, all layers) at M = 1..4 (to 8 where lists reach it):
   today's kernel against U1 / U2 (/ U3), interleaved pairs after a warm-up, median of 5, on routes (a) recorded,
   (b) iid synthetic, (c) all rows distinct (worst) and all rows equal (best) - the bounds of the union gain.
3. **The attribution** (Ornith, Flash-Next): the real verify list at M = 1..4 split into GEMV, experts, attention,
   other; shares applied to the unprofiled step (as 28a).
4. **Numerics:** each form against today's kernel (bitwise for U1 / U2) and against an fp64 expert reference (N2's
   bound); property R-moe.

## 4. The go rule (per engine; thresholds are proposals, open decision 1)

For an engine with a verify list (Ornith, Flash-Next), at M = 4 (MTP's maximum):

- **GO** (the build plan may be written, the operator asked) when ALL hold:
  (a) **the expert block:** the union form at M = 4 is **>= 10 % below today's** on the recorded routes (measured;
      10 % = the 3 % noise line plus margin) and within 25 % of the bytes model's ideal for the measured D(4) (derived);
  (b) **the whole step:** derived verify(4) = measured non-expert time + (a)'s expert block, dense part at 28b's cost,
      **<= 1.6 plain steps** of the engine's own M = 1 step (derived from measured parts; Qwen3.8's verify(4) is 1.74
      today);
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
"the model everything is tuned for"; the only one measured on the card) and is dense, so spec 28 covers it. Every
engine below is second tier and **has never run on a B70**.

| order | engine | verify list | probe | worth running |
|---|---|---|---|---|
| 1 | Ornith | yes (M <= 4) | full | **yes** - the only MoE engine on the Qwen capture with MTP and lookup built; dense part rides 28b; the union kernel ports by copy |
| 2 | K2-Horizon | no | kernel + routing replay | **yes, cheap, conditional** - the only measured routing data; largest dedupe (E = 100); matters only if a verify list (lookup) is wanted; lookup wins on code (1.18-1.68, probe-eagle3-k2) |
| 3 | Flash-Next | yes (M <= 4) | full, truncated model | **later** - served only with `--layers N`; iid dedupe ~7 % at E = 512, so value is locality only |
| 4 | Bolt | no (removed) | shape rows | **no separate run** - nothing to verify; its 12 (E, I) pairs are rows of the shared probe |
| 5 | Kolibri | no | shape rows | **no separate run** - same |
| - | Flint | nothing built | none | **no** - not a Kolibri twin (spec 27: hidden 2048, 256 experts top-6, 2 dense + 47 MoE, own shapes; Kolibri's kernels hardcode 2560 / 384 / 48 / 128000), so it needs its own descriptor first |
| - | Nanbeige | not in the repo | none | not MoE, no engine path; if added as plain dense Llama it would use `gemv.cl` and the Qwen capture, and 28b's shape list (Qwen3.8 + Agnes) would need its shapes |

### 5.1 Ornith (order 1)

Question: **how far above the ideal do `moe_gate_up` / `moe_down` sit at M = 2..4, and does a union form bring MTP's
verify(4) to <= 1.6 plain steps?**

| item | value |
|---|---|
| path | the Qwen capture (`capture.cc` branches on `is_moe()`); per layer router `gemv_bf16` (272 rows), `moe_route`, `moe_gate_up`, `moe_down` (residual fold); 526 launches a token |
| MoE shapes | hidden 2048, E 256, k 8, inter 512; gate_up K 2048 x N 1024, down K 512 x N 2048; shared = slot 8 |
| dense shapes (28b) | GDN / attention projections from Ornith's `LinearId` table (`qwen35.h`), int4 `a||b`, int8 head 2048 x 248077 - **not in 28b's list**; shape rows here |
| M | 1..4 (MTP K <= 3, lookup K <= 3) |
| measured today | nothing (BENCHMARKS "Ornith 1.5 MoE": pending) |
| expected upside | MoE-layer reads at M = 4: 36 -> 31.5 (iid) / ~21 (locality guess); verify(4) 2.1 -> 1.9 / 1.5 plain steps (estimated, bytes only, §2.3) |
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
| measured today | routing locality (above); **no decode or verify time** (BENCHMARKS: nothing measured) |
| expected upside | experts-only reads at M = 4: 32 (today, upper) against 18.55 + shared measured; ~1.9 GB of a 3.15 GB step (estimated, §2.4) |
| 28b reuse | dense linears (`gemv.cl`, layout 0): yes with K2's shapes added; experts and MoVA: no |
| precondition | a verify list for K2 (lookup's) - **not built**, not built here |
| needs the card | all timing; the routing replay needs the recorded routes (on the box) |

### 5.3 Flash-Next / qwen4exp (order 3)

Question: **is routing local enough at E = 512, k = 10 for a union form to matter (iid says ~7 % of reads)?**

| item | value |
|---|---|
| path | own engine `runtime/qwen4exp`; int4 dense GEMVs through `gemv_variant`, `q4_moe.cl` experts, own HC / PLE / QSA kernels; MTP head with its own 512-expert MoE layer |
| shapes | 48 layers (36 GDN, 12 QSA), hidden 2560, E 512, k 10, inter 640 (~2.61 MB), shared 640 gated |
| M | 1..4 |
| measured today | nothing |
| expected upside | MoE-layer reads at M = 4: 44 -> 40.9 iid, ~26 only if locality is K2's (estimated) |
| 28b reuse | GDN `qkvz` / `out_proj` int4: yes with its shapes; experts: no |
| status | served only truncated; the whole model waits for spec 22's tier. **After Ornith, same tool, only if the operator wants Flash-Next's MTP** |

### 5.4 Bolt, Kolibri

| engine | what the probe does | why no more |
|---|---|---|
| Bolt (88 blocks: 40 Mamba, 40 MoE, 8 attention) | shape rows: the 12 (E, I) pairs (E 240..496, I 1536..2560, k 9, latent 1024) through the stand-in; Mamba in_proj 4096 -> 18560 and out_proj 8192 -> 4096 as 28b-form dense shape rows | verify lists, per-row Mamba slots and the union kernel were dropped 2026-10-26; Mamba verify needs a state per row, a different problem (spec 25 §6.1 still describes the dropped verify: stale, open decision 6) |
| Kolibri-1 (50 MoE layers, E 384, k 6, inter 512, hidden 2560) | shape rows: `kol_moe.cl`'s shapes, int4 shared expert; dense q||k||v and o_proj on `gemv.cl` are 28b-form shape rows | no MTP, no drafter, M = 1 only; two cards; never run |

## 6. Rules that bind every stage

- **Timing:** interleaved control / candidate pairs after a warm-up, median of 5 (kernels) / 3 (steps); the 3 % line is
  noise. **One-card, busy-box timing is allowed but labelled "busy-box timing (one card), to be re-checked on a quiet
  box"** (operator, 2026-11-01); the record lists the re-check set. Grades on every number.
- **M = 1 bit for bit** (G0 checksums; `tools/kernel_cmdlines` +N / -0 / ~0, only union-kernel lines added).
- **No new CLI flag or environment variable** for production selection: the capture rule, a test-only capture option.
- **No host-RAM offload on the per-token path** (PCIe 3.0 box). Nothing here uses it; spec 22's tier is untouched.
- **256-GRF builds** carry the option on the ocloc line; the probe reads the built binary's GRF mode before a timed arm.
- Which experts repeat never changes a launch count: fixed grid, early exit.

## 7. Open decisions (operator)

1. **The thresholds of §4:** (a) >= 10 %, (b) <= 1.6 / 1.9 plain steps - proposed, 28a's shape with a floor the MoE part
   cannot go below (§2.2).
2. **Order and scope:** proposed Ornith -> K2 (conditional) -> Flash-Next (later); Bolt and Kolibri shape rows only.
3. **A K2 lookup verify list:** spec it at all? It is the precondition for K2's probe to matter. Proposed: decide
   after Ornith's numbers.
4. **Ornith's (and K2's, Flash-Next's) dense shapes in 28b:** add to 28b's variant list (same capture) or a follow-on?
   Proposed: add, if 28b is built.
5. **Prompt set for routing traces:** proposed golden prompts + the A4 set, as K2's record.
6. **Spec 25's stale verify text** (and spec 20's "shared expert stays bf16", superseded 2026-10-31): fix or leave?
   Not touched here.

## 8. Out of scope

M > 8; building any verify list; Mamba row state slots (Bolt); the expert-offload tier (spec 22); acceptance and
drafting policy (specs 8, 19); the attention of MoE engines (28c's shape rule, per engine, later); prefill MoE GEMMs
(already per-expert grouped).

## 9. Verdict

Not run. Written by the plan's Task 6 when the card has run the stages.
