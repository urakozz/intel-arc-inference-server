# Spec 17 - tensor parallel over two B70s

**Status:** **PARKED (operator, 2026-10-23): refreshed, retargeted at Nemotron 3.5 Bolt, parked.** Nothing of spec 17
is built; no plan of it has started. Written 2026-10-05 as a Qwen3.8-first design for review; that text is kept as
Appendix A, its stale claims marked in place. §R is the 2026-10-23 revision; §1-§9 are the corrected design and
supersede Appendix A wherever they differ.

**Plans:** `docs/superpowers/plans/2026-10-05-spec17{a,b,c,d}-*.md` (rewritten 2026-10-23, Bolt first, each PARKED):
17a the standalone P0 and the go / no-go rule, 17b decode on Bolt, 17c prefill on Bolt, 17d snapshots, speculation,
serving and the record. Box queue row **68** is proposed for 17a (the next free row on 2026-10-23: rows 64 / 65 are
spec 26's, 66 spec 13a's, 67 the prefill bandwidth profile); 17b-17d take the next free rows when they are built.
Like spec 25's plans, these documents do not edit `tools/box_validate/stages.sh`: each plan describes its stages and
adds them when it is built.

**Grades.** Every number is marked **measured** (on this box, by this repo, with the record cited), **derived** (from
descriptors, configs or measured numbers by arithmetic shown here), **estimated** (a judgement), **assumed** (a
placeholder for a number nobody has measured; every derived figure that uses one says so) or **published** (someone
else's figure, cited).

## R. Revision - 2026-10-23: refresh, retarget at Bolt, park

**The ruling (operator, 2026-10-23):** "Refresh, retarget at Bolt, park". Tensor parallel's first targets are
**Nemotron 3.5 Bolt 74B-A9B (spec 25)** and then **Kolibri-1 (spec 20)** - the models that need both cards anyway,
where pipeline parallel (spec 16) gives capacity but no decode speed. **Qwen3.8 / Agnes only if the two-card replica
mode (spec 13 §10) loses to TP.** The plan is **PARKED**:

1. **P0 (plan 17a) runs** only after spec 16's first two-card rows - **`r22.devices` and `r22.p1`** - have passed on
   the box (the first time two cards run in one process, peer access included).
2. **Building (plans 17b-17d) begins** only after **Bolt's rows 59-62 pass** AND only if **P0 measures the crossing
   cost at <= ~1 ms a token** (§3 states the rule exactly).

**What changed since 2026-10-05** (each checked against the repo on 2026-10-23):

| 2026-10-05 premise | the fact on 2026-10-23 |
|---|---|
| spec 16a's P0 measures TP's key number ("the remote-partial arm") | **16a never ran** (`tools/box_validate/stages.sh`, rownote 22: "16a's probe ... has not run - it is plan 16a's own probe, not a stage"); `docs/probe-multi-gpu-2026-10-05.md` and `tools/probe/probe_p2p.*` do not exist. **17a now owns its arm** (§3, plan 17a) |
| spec 16 gives the multi-device foundation | 16b-16d were **built blind and merged** (spec 16 §8-§10): one Level Zero context over both cards, per-device buffers / lists / queues, `pp_handoff.cl`'s system-scope flag protocol, the bounded fences; **none of it has run on a card** - rows 22 / 23 / 27 pending |
| TP is the one multi-GPU mode that shortens every turn (Qwen3.8 first) | spec 13 §9 decision 4 keeps the **second card free** for dense models (pipeline parallel, long contexts, another model) and §10 writes the **two-card replica mode** for later; spec 25 Decision 4 names TP "the decode lever after 25e" for Bolt. The targets are re-ordered (§2) |
| one engine (`src/runtime/capture.cc`, `prep_res_fold`) | every model family since has **its own engine and fold kernels**: `src/runtime/{bolt,k2,kolibri,lfm2moe,muse,qwen4exp,dflash}/`, `src/kernels/{bolt,k2,kolibri,lfm2moe,muse,qwen4exp,dflash}/`. TP is per-engine work; the target engines are named (§4.10) |
| MoE out of scope ("expert parallel ... a different scheme") | both targets are MoE: the expert split is now a decision with a measurement (§4.3) |
| no speculative decoding beside MTP | spec 19 (DFlash, prompt lookup, `--draft-vocab`) and spec 26 (LiLiCorr, Bolt's drafter) exist; their TP form is §4.8 |

**What this revision corrects** (a review of the 2026-10-05 text; each finding verified against the repo before it was
used):

| finding | where it is fixed |
|---|---|
| P0's dependency on 16a's arm and files that do not exist | §3, plan 17a (its own probe, its own row) |
| the status line | the Status block above, §R |
| 54.90 t/s and ~6.6-10.8 µs were written as if measured: they are **vLLM's published** figures (PR #570) | §1 (relabelled), Appendix A marks |
| "the other card's partial is one more partial": with split-K partials a fold is S x hidden fp32 a row - **80 KiB on Qwen3.8 at S4, ~10.5 MB a token each direction, ~0.87 ms at an assumed 12 GB/s** | §4.1 (three mechanisms, sizes from each descriptor, M <= 8 verify rows) |
| Qwen3.8 constants throughout | §4.2: every shape from the model's descriptor (`model::BoltDesc`, `Kolibri1Desc`, `ModelDesc`) |
| the GDN split ignored fused layouts (`qkv‖z` at layout 1, `a‖b` padded) | §4.2's dense table (conditional scope) and Bolt's fused `in_proj` / `q‖k‖v` / `fc1‖shared_up` |
| one engine | §4.10 |
| no MoE split decision; Bolt's "splits cleanly" (spec 25 §4.3) unproven | §4.3 (expert parallel vs intra-expert TP, measured from 25a's route records); §4.2 (what is derived, what is unproven) |
| placement: one load then slice | §4.6 (slice per device at load - Bolt's loader already uploads per device - and a TP memory planner) |
| snapshots as `{card 0 part, card 1 part}` | §4.7 (the one-card host layout, spec 16b's rule; spec 13 §10 needs entries restorable on either card) |
| spec 19 / 26 missing | §4.8 |
| `--tp` beside `--pp` only | §4.9 (`--tp`, `--pp`, `--replica` mutually exclusive; spec 13) |
| speed bars against one card | §6 (TP vs `--pp 2` for Bolt / Kolibri; vs `--mtp auto` one card for Qwen3.8; TP + speculation) |
| the prefill choice before PP prefill is measured | §4.5 (after row 23 for the main engine, row 61 for Bolt) |
| unmeasured link facts used as facts | §1 (each graded; ReBar recorded as measured; PCIe 3.0 x16 recorded as operator-stated; the peer rate still unmeasured) |
| the plans predate the runbook | plans 17a-17d (rows, stages in plan 24 / 25 / 26a style, `gpu` locks, signed commits after each step) |

## 1. What is known about the two cards

| fact | value | grade | source |
|---|---|---|---|
| cards | 2 x Arc Pro B70 (`bmg-g31`), 32 GB each; one CPU root complex | measured | docs/10-the-box.md; the box is a Dell T5810 (Xeon E5 v3 / v4 host, `tools/probe/offload_links.sh`'s header) |
| device 1 vs device 0 | prefill **-3.2 to -3.4 %**, decode **-0.2 to -0.4 %** (pp4096 1330.82 vs 1377.20 t/s; tg256 32.31 vs 32.37) | measured 2026-09-05 | docs/BENCHMARKS.md "The two cards are not interchangeable" |
| resizable BAR | **both cards have a 32 GB BAR mapped** (`lspci`: `[size=32G]`) | measured 2026-10-23 (operator, by hand) | recorded here first |
| P2P kernel patch | installed: 7.3.0-rc4 with `b70-p2p-exp2` | measured (installed) 2026-10-22 | docs/10-the-box.md "Versions on the box" |
| P2P exercised by our code | **never**: no two-card process of ours has run (rows 22 / 23 / 27 pending) | - | box-validation-queue.md |
| PCIe link generation and width per card | **PCIe 3.0 x16, both cards** - **operator-stated** 2026-10-23, not read by `lspci` / `offload_links.sh` (spec 22a's `r36.links` is pending; 17a's `r68.links` records it); theoretical ~15.75 GB/s per direction (derived); both cards under one CPU root complex (Broadwell-EP, operator-stated) | operator-stated | the operator, 2026-10-23 |
| host <-> device copy rate, device 0 | H2D **12.2-12.9 GB/s**, D2H **14.2 GB/s** (pinned host) | measured 2026-09-27 | docs/probe-prefix-cache-2026-09-27.md |
| peer copy rate (copy engine; kernel peer writes; **kernel peer reads**) | **not measured**. Peer traffic between two root ports of one Xeon E5 root complex can be far slower than host traffic, and peer reads slower than peer writes (estimated: a known risk of the platform) - P0 measures each | - | - |
| **12 GB/s**, used for every transfer time below | **assumed** for peer traffic: the measured host <-> device rate above, not a peer measurement; every time derived from it scales with the measured peer rate | assumed | - |
| a peer write + system-scope flag inside a replayed list | **not measured** (16b built `pp_send` / `pp_recv`; `r22.p1` under `--pipeline-handoff peer` is its first run) | - | spec 16 §8 |
| a two-rank all-reduce of 2-64 KiB inside an XPU graph on two B70s | ~6.6-10.8 µs | **published** (host platform not stated) | [vllm-project/vllm-xpu-kernels#570](https://github.com/vllm-project/vllm-xpu-kernels/pull/570), RFC [vllm-project/vllm#54766](https://github.com/vllm-project/vllm/issues/54766) |
| Qwen3.8-27B GPTQ int4, vLLM TP = 2, one request, `--max-model-len 8192` | 54.90 t/s decode | **published** | the same PR |
| Bolt (SergiioB's g128), vLLM 0.31 XPU TP = 2 | decode 60.5 / 50.7 / 33.5 / 23.0 t/s at 886 / 8k / 32k / 64k; prefill 2925 / 2014 / 871 / 486 t/s | **published** (his card, 2026-10-16) | spec 25 header |
| Qwen3.8 one card, int8 head | 31.21 t/s decode | measured | spec 9, README |
| Bolt / Kolibri on our engine | **nothing measured**: 25b-25e and 20c-20e are built blind (rows 59-62, 24 / 26 / 28 pending) | - | spec 25 §13-§17, spec 20 §11-§13 |

## 2. Why, and for which models

**Pipeline parallel does not make batch-1 decode faster** (spec 16 §1): the stages run one after the other, so a
token still reads every weight through one card's ~590 GB/s (docs/05). **Tensor parallel does**: each card reads its
half of every split matrix at the same time, so the bytes per card roughly halve, plus the cost of the cross-card
points ("crossings", §4.1). Launch counts do not halve - each card runs the whole list at half widths - so a
launch-bound step gains less.

**Targets, in order (the ruling):**

| model | why TP | one card | today's two-card mode | TP decode, derived / estimated |
|---|---|---|---|---|
| **Bolt 74B-A9B** (spec 25) | needs both cards (40.46 GB of weights, derived); bandwidth-bound: 5.48 GB a token incl. the fp32 SSM state (derived, spec 25 §3) | does not fit (spec 25 Decision 4) | `--pp 2` (25c-25e), ~84 t/s at short depth (estimated, spec 25 §3) | per card ~2.9-3.0 GB a token (derived, §4.3) -> **~1.5-1.7x `--pp 2`** at a crossing cost of 0.5-1.0 ms a token (estimated: 77.6 % of roofline as Qwen3.8 measured, Bolt's launch floor unchanged) |
| **Kolibri-1** (spec 20) | needs both cards (~42.5 GB int4, spec 20 §11) | does not fit | `--pp 2` (20c-20e), roofline ~250 t/s (derived, spec 20 §1) | ~1.0 GB a token a card (derived); the step is short (~4 ms), so 101 crossings and ~756 launches weigh more: **~1.3-1.5x** (estimated). Also needs spec 20b's real checkpoint (decision 1 pending) |
| Qwen3.8 / Agnes | **only if the replica mode (spec 13 §10) loses to TP** | fit one card | spec 13 §9 decision 4 keeps card 1 free; §10 replica mode (later) | Qwen3.8: ~1.6-1.8x one card at batch 1 (estimated, Appendix A §1 with the crossing cost of §4.1), against one card's `--mtp auto` (§6) |
| K2, Muse, LFM2, qwen4exp, Ornith | not targets | - | - | - |

## 3. The park, and P0's go / no-go rule

**When P0 (17a) runs:** after `r22.devices` and `r22.p1` PASS on the box. 17a is standalone: its own probe
(`tools/probe/probe_tp`, its own kernels in `tools/probe/`), its own row (68, proposed), no dependency on spec 16a.
It measures:

1. **Links:** each card's PCIe generation and width, idle and under load, and every bridge to the root port
   (`tools/probe/offload_links.sh`, spec 22a's script, run by 17a's own stage whether or not `r36.links` ran).
2. **Peer copy rate:** copy engine 0 -> 1, 1 -> 0 and both at once; kernel peer writes; kernel peer reads (a pull
   through the root complex, possibly much slower than a push); 4 KiB to 64 MiB; immediate and inside replayed
   lists. The link width is operator-stated (PCIe 3.0 x16); the peer rate is unmeasured until this arm.
3. **The crossing cost - its latency first** (§4.1's three mechanisms; each as a push - peer write + flag - and,
   where it applies, as a pull - peer read after a flag), inside two replayed lists (one per card), at each target's
   crossing count and bytes from its descriptor (Bolt: 88 + the head exchange; Kolibri: 100 + 1; Qwen3.8: 128 + 1),
   at M = 1, 4 and 8 rows; both cards' results bitwise equal in every arm; the cost is the difference against the same
   lists without crossings, interleaved pairs after a warm-up, median of 5, idle box.
4. **The MoE balance** (§4.3) from 25a's route records: expert parallel's per-card load under candidate placements,
   per M = 1..8, beside intra-expert TP's.
5. **The per-card step at TP shapes** for Bolt: 25c's `probe_bolt_step` at the halved shapes (stand-in kernels as 25c's
   P0), so the derived TP step is per-card time + crossing time, not bytes alone.

**The go / no-go rule** (17a states it and applies it; the operator rules on the band):

- **GO** - plans 17b-17d may start - when ALL hold:
  (a) **Bolt's rows 59-62 have passed** on the box;
  (b) **correctness:** the chosen mechanism's crossings are bitwise equal on both cards over 10^4 replays, and a
  stale or missing peer write is detected (the stamp check), never consumed;
  (c) **cost:** one Bolt token's crossings (89 at Bolt's sizes, M = 1, the chosen mechanism, inside replayed lists,
  both cards idle) cost **<= 1.0 ms**, measured.
- **The operator's call:** (c) between 1.0 and 1.2 ms ("~1 ms").
- **NO-GO** - TP stays parked, the record says which condition failed and by how much - when (b) fails, or (c) is
  above 1.2 ms, or the links train below Gen3 x8 (offload_links.sh's STOP line).
- **Kolibri's go** is the same rule at its own count (101 crossings at its sizes), evaluated after Bolt's record and
  after spec 20b's real checkpoint exists.
- **Recorded, not gating:** the derived TP decode for each target from (3) + (5); the M = 4 / 8 costs (speculation,
  §4.8); the prefill crossing (§4.5); the MoE choice (§4.3).

## 4. The design (corrected)

### 4.1 The crossing: three mechanisms, sized per model

A **crossing** is a point where each card holds a partial result the other needs: after every row-parallel linear
(the mixer's output projection, the FFN / MoE output) and once in the head. Each engine's decode GEMVs write fp32
**split-K partials** (S slices of the output row) and the next block's fold kernel sums them into the residual (main
engine `prep_res_fold`; Bolt `nb_res_norm`, which also folds the two slice sets a MoE block leaves; Kolibri's
`prep_res_fold _Z` + `kol_post_add`). The 2026-10-05 text called the peer's partial "one more partial"; with S slices
it is S more, and the size matters. Three mechanisms, all keeping **one fixed fold order on both cards** (card 0's
contribution, then card 1's) so both cards hold a **bitwise-identical residual** (T1):

| mechanism | what crosses, per row | extra launches | note |
|---|---|---|---|
| **(a) remote partials** | the row-parallel GEMV writes its S slices locally and into the peer's landing buffer; peer write + system-scope release flag; the fold waits (acquire) and folds 2S slices | 0 | bytes x S; the 2026-10-05 proposal |
| **(b) fold-then-send** | each card folds its own S slices first (in the GEMV's epilogue or a small kernel), then sends one fp32 row | 0 or 1 a crossing | bytes / S; the local fold's rounding is fp32, then the cross-card add in fixed order |
| **(c) a separate all-reduce kernel** | a two-rank exchange-and-add between the GEMV and the fold (the published prior art's shape; our own code) | 1 a crossing | the fallback if (a) / (b) lose |

**Sizes, derived from the descriptors (fp32, one direction, M = 1):**

| model | crossings a token | bytes a crossing, (a) | (b) | a token, (a) | (b) | (a) at 12 GB/s assumed |
|---|---|---|---|---|---|---|
| Qwen3.8 (`ModelDesc`: `down` and `out` / `o_proj` at **S4**, `src/model/model_desc.cc:30-31`; hidden 5120) | 64 x 2 = 128, + 1 | 4 x 5120 x 4 B = **80 KiB** | 20 KiB | **~10.5 MB** | ~2.6 MB | **~0.87 ms** ((b): ~0.22 ms) |
| Bolt (`BoltDesc::linears`: every output linear at **S1, PROVISIONAL** until 25c's P0 sweeps S, `src/model/bolt.cc:150-168`; hidden 4096) | 40 Mamba + 8 attention + 40 MoE = 88, + 1 | 16 KiB (S1); MoE adds the routed part (§4.3): +4 KiB (latent partial sum) or +36 KiB (9 per-expert latent rows, for a combine bitwise equal to one card's) | the same at S1 | ~1.6-2.9 MB at S1 (x S if 25c picks S > 1) | - | ~0.13-0.24 ms at S1 |
| Kolibri (`Kolibri1Desc`: `o_proj` at **S4, PROVISIONAL**; hidden 2560) | 50 attention + 50 MoE = 100, + 1 | 40 KiB (`o_proj`); MoE +10-60 KiB (§4.3) | 10 KiB | ~2.5-5 MB | - | ~0.2-0.4 ms |

**The crossing is latency-bound, not bandwidth-bound** (derived; the operator's note, 2026-10-23): with
fold-then-send, one Bolt crossing at M = 1 is 4096 x 2 B = **8 KiB in bf16** (16 KiB fp32): **~0.7 µs** of transfer at
12 GB/s (1.4 µs fp32), so 88 folds move ~0.7 MB, **~60 µs a token** (~120 µs fp32) (derived, assumed peer rate). What a
crossing really costs is its **latency** - the peer write landing, the system-scope release, the peer card noticing
the flag, the fold starting - times 89. At the published 6.6-10.8 µs a crossing that is **~0.6-1.0 ms** a token
(derived from published): the whole of §3's budget. So P0 measures the **per-crossing latency** inside replayed lists
at each model's count (not the bandwidth alone), and **peer write against peer read** (a pull from the other card's
memory) through the single root complex, where reads may be much slower; the go rule stays **<= ~1 ms a token in
total**. (A bf16 send rounds each card's half before the cross-card add: both cards still agree bitwise, T1; the cost
is T2's accuracy, recorded.)

**Verify rows scale it:** speculation verifies M <= 8 rows (spec 26: LiLiCorr's 8-row block; spec 19 DFlash K <= 7;
spec 8 MTP M <= 4); every crossing carries M rows. Bolt at M = 8: ~13-23 MB a verify, **~1.1-1.9 ms** at 12 GB/s
(derived, assumed rate) - on top of the latency. Qwen3.8 under (a) at M = 8: ~84 MB, ~7 ms (derived, assumed rate):
(a) is not viable for dense verify; (b) is 4x smaller. P0 records M = 1 / 4 / 8.

**The two P2P rules from spec 16 §2 apply to every mechanism:** the peer-written landing buffer is an allocation of
its own, page-aligned, holding nothing the local card writes; flags are polled with system-scope atomics; the
sequence counters live in device memory and advance in the kernels, so replays with frozen arguments progress; the
copied data is read with system-scope loads (16b's `pp_recv` lesson: a cached copy of the landing lines is the stale
buffer). A crossing that times out is recorded in device state and thrown by the host after the bounded fence (T4).

### 4.2 The split, per model, from its descriptor

Megatron-style: column-parallel linears split N (no crossing), row-parallel linears split K (a crossing after).
**Every shape comes from the model's descriptor** (`model::BoltDesc`, `model::Kolibri1Desc`, `model::ModelDesc`
for the main engine) through a `tp_rank` / `tp_size` view; no constant is written into kernels or plans.

**Bolt (spec 25 §1.1; 88 blocks: 40 Mamba-2, 8 attention, 40 LatentMoE).** Derived from the config; the "splits
cleanly" of spec 25 §4.3 is **derived, unproven in code** - 17b's host tests prove each line:

| block | column-parallel (per card) | row-parallel -> crossing | replicated |
|---|---|---|---|
| Mamba-2 | `in_proj` N 18560 = z 8192 ‖ x 8192 ‖ B 1024 ‖ C 1024 ‖ dt 128 -> per card **z 4096 ‖ x 4096 ‖ B 512 ‖ C 512 ‖ dt 64** (heads 64 of 128; B / C groups 4 of 8 - a group is 16 heads, so no group straddles the cards); the per-card N is a **gather of five column ranges, re-fused at load**, not a contiguous half; the depthwise conv over the card's xBC channels (5120 of 10240) and its raw-xBC ring; `A`, `D`, `dt_bias` per head; the gated RMSNorm's groups of 1024 channels = 16 heads, 4 per card; the SSM state 64 heads a card (2 MiB fp32 a layer) | `out_proj` K 8192 -> 4096 a card (64 g64 groups) | the block norm |
| attention (NoPE, GQA 16) | fused `q‖k‖v` N 4096 ‖ 256 ‖ 256 -> per card **q 2048 ‖ k 128 ‖ v 128** (16 q heads, 1 kv head), re-fused; KV 1 head a card | `o_proj` K 4096 -> 2048 (32 groups) | - |
| LatentMoE | the shared expert's up (5376 -> 2688 a card, 42 groups) out of the fused `fc1‖shared_up` (re-fused per card with the replicated fc1 part) | `shared_experts.down_proj` K 5376 -> 2688; the routed part per §4.3 - sent in the **same** crossing | the router (bf16 [E][4096], 2-4 MB a layer), `fc1_latent_proj` (2.2 MB), `fc2_latent_proj` (2.2 MB) - §4.3's rule |
| embedding | - | - | replicated (1.07 GB a card; a gather, no traffic) |
| `lm_head` | vocabulary split, 65536 rows a card | the argmax exchange (§4.4) | `norm_f` |

int4 g64 groups run along K, so any N slice of a column-parallel linear is group-safe; every K slice above is a
whole number of 64-groups (checked: 4096 / 2048 / 2688 / I / 2 for I in {1536, 2048, 2304, 2560} = 12 / 16 / 18 / 20
groups).

**Kolibri (spec 20 §1; 50 MoE layers, 40 sliding + 10 full attention, sandwich norms):** attention 24 of 48 q heads
and 2 of 4 kv heads a card; `o_proj` row-parallel -> crossing; **the sandwich norm** (`post_attn_norm`,
`post_ffn_norm` are RMSNorms over the whole 2560-wide mixer output, applied before the residual add) needs the full
output, so Kolibri's crossing completes the mixer output first, then norms, then adds - the same crossing count, a
different fold target than Bolt's. The MoE per §4.3 (384 experts top-6, I 512; the ungated shared expert, 2.6 MB a
layer, replicated). Router replicated.

**Qwen3.8 / Agnes (conditional on §2):** Appendix A §2 decision 2's table, corrected: **the fused layouts** - GDN's
`qkv‖z` (layout 1, S1, `model_desc.cc:31`) and the padded `a‖b` - must be re-fused per card from each card's heads,
not sliced from the one-card buffer; Agnes's counts (54 GDN + 18 FA, 72 layers) and its parallel-FFN fold come from
its descriptor. Ornith (MoE) follows §4.3 if it is ever a target.

### 4.3 MoE: expert parallel or intra-expert TP, per model, measured

Two schemes; the choice is per model and taken from P0's measurement:

- **Expert parallel (EP):** each card holds half of every layer's experts (Bolt: E even in every layer, 240-496;
  Kolibri: 192 of 384). A token's top-k experts fall unevenly between the cards; **the slower card sets the step.**
  With top-9 drawn uniformly and independently (an assumption the routes will correct), the busier card carries 5.73
  of 9 expert blocks on average (derived, binomial), 1.27x the even share; Kolibri's top-6: 3.94 of 6, 1.31x.
  Placement can do better than uniform: a frequency-balanced assignment computed from held-out routes.
- **Intra-expert TP:** every expert split in half (Bolt: `up` N I -> I / 2, `down` K I -> I / 2; Kolibri I 512 ->
  256); perfectly balanced, but each expert GEMV runs at half width (Bolt's are already small: latent 1024 x I), and
  `down`'s output becomes a K-partial that must cross.

**The routed part of the MoE crossing.** The reference combine (spec 25 §2.4) adds each expert's `d_e · w_e` in
ascending id with one rounding at the end, then `fc2_latent`. Either scheme can send (i) each card's latent partial
sum (Bolt 4 KiB a row; the combine is then a two-term fp32 sum - not bitwise one card's order) or (ii) the per-expert
rows (Bolt <= 9 x 1024 fp32 = 36 KiB; each card then runs the exact ascending-id combine - **bitwise one card's
combine**, so T2 can hold MoE outputs to one card more tightly). P0 costs both; the shared expert's down partial rides
in the same crossing, so a MoE block is **one** crossing.

**The routes must agree on both cards:** the router runs replicated on the bitwise-identical normed hidden (T1) with
the same binary, so both cards select the same set; a route mismatch between cards is a T1 failure, checked.

**Replicate small linears rather than add a crossing (derived rule):** one crossing at the published ~7 µs costs as
much time as reading ~4 MB at 590 GB/s (derived). A linear whose split would add a crossing and whose bytes a layer
are below that is replicated: Bolt's router, `fc1_latent_proj` and `fc2_latent_proj` (~0.29 GB a token together,
read on both cards), Kolibri's router and shared expert. P0's measured crossing cost resets the threshold.

**The measurement (P0, `tools/spec/tp_expert_balance.py`, box CPU):** from 25a's golden / A4 route records (ids per
row and MoE layer: `oracle-out-bolt`, row 57's `r57.golden`) and, when spec 20b exists, Kolibri's: for each placement
(even / odd ids, contiguous halves, frequency-balanced on a held-out half) the busier card's routed bytes per (row,
layer) - mean, p50, p90, per E / I bucket - at M = 1 and for M = 2..8 consecutive rows (the verify's per-card expert
union, plan 26a's `tools/spec/expert_union.py` split by card); beside intra-expert TP's even half. With P0's per-card
step (§3 item 5) at half-width expert GEMVs this gives each scheme's derived MoE time. **Rule (proposed):** the scheme
with the lower derived per-token time at M = 1 wins, unless the other wins at M = 8 by more than it loses at M = 1
(the operator's call when speculation is the default).

**Bytes a card a token for Bolt (derived from spec 25 §3's table, M = 1, int8 head, fp32 state):** Mamba 1.164 GB,
attention 0.076, `fc1` / `fc2` replicated 0.178, shared expert 0.468, routed 0.450 (intra-expert) or ~0.573 (EP,
uniform assumption), routers 0.113, head 0.268, small 0.011, SSM state 0.168 -> **~2.90 GB (intra-expert) / ~3.02 GB
(EP)** against 5.48 GB on one card's path under `--pp 2`: 1.81-1.89x fewer bytes on the busier card.

### 4.4 The head and sampling

`lm_head` split by vocabulary (Bolt 65536 rows a card; Kolibri 64000; Qwen3.8 124160). **Greedy:** each card's top-1
id and value cross once; the larger wins, ties to the lower id, identical on both cards. **Sampling (decide, kept
from 2026-10-05):** each card applies the shared top-k / top-p prefilter to its half and the host merges per-card top-k
plus the two partial log-sum-exps (exact), or the host reads both halves (two readbacks of the vocabulary). Proposed:
the merge. LiLiCorr's head (spec 26) needs the full-vocabulary top-8 candidates and their log-probs: the same merge.

### 4.5 Prefill

Prefill's crossings carry a chunk: Bolt at C = 2048 rows and fp32, 2048 x 4096 x 4 B = **32 MiB a crossing**, 88
crossings ~2.95 GB a chunk, **~0.25 s at 12 GB/s** (derived, assumed rate; bf16 halves it at an accuracy cost);
Qwen3.8 under (b) 40 MiB a crossing, 128 a chunk, ~0.45 s (Appendix A §2 decision 4 had the (a) form). Against it:
compute that halves. Bolt's `--pp 2` prefill is **sequential** (spec 25 §6.2: Kolibri's per-chunk hand-off, no chunk
pipeline), so TP prefill competes with one card's compute rate there - unlike the main engine, where spec 16c's
chunk pipeline approaches ~1.9x at long prompts (derived, spec 16 §9). **The choice is made only after the two-card
prefill is measured:** row 61's `r61.speed` for Bolt, row 23's `r23.s2_*` for the main engine. Options: TP prefill
with crossings overlapped on a copy engine; a hybrid (prefill as `--pp 2`, decode as TP: each card needs both
layouts or a reshuffle after prefill - memory decides); TP prefill with bf16 crossings. 17a records the prefill
crossing's measured rate (opt-in arm); 17c decides.

### 4.6 Placement and the memory planner

**Slice per device at load.** Bolt's loader already places each block straight onto its device
(`src/loader/bolt_loader.h`, "Placement at load"; `BoltDevicePart` per device), and Kolibri's does the same
(`load_kolibri1`); TP adds a placement kind whose parts are **the same blocks on both devices, each holding its slice**
(the per-card re-fused layouts of §4.2, the replicated parts on both). No one-card load then slice: no target fits
one card. **A TP memory planner** beside `runtime::pipeline_plan` / `runtime::memory_plan` and Bolt's
`runtime::bolt::plan`: per card the sliced and replicated weights, its heads' KV and state, the decode / prefill
scratch at per-card shapes, the crossing landing buffers (two per crossing point for double-buffering, page-aligned),
the reserve; `--max-len auto` from the tighter card; `plan == allocation` asserted as 25b / 25c do.

**Bolt, derived:** weights ~20.9 GB a card under EP (half of 40.46 GB plus the replicated embedding 1.07 GB, routers,
`fc1` / `fc2`); KV at 262144 1.07 GB a card (one kv head of two), SSM state 84 MB a card - **~22.5 GB of 31 GB usable a
card** (estimated with spec 25 §3's scratch), so the full context fits and spec 26's drafter has room (§4.8).

### 4.7 Snapshots and the prefix cache

**A snapshot taken under `--tp 2` is written in the one-card host layout**, byte for byte (spec 16b's ruling for
`--pp 2`, spec 25e's "one host layout for every split"): the host gathers each card's heads of KV and SSM / GDN state
and conv-ring channels into the one-card order on save and scatters them on restore. An entry is then the same bytes
whichever mode wrote it, restorable under `--tp 2`, `--pp 2`, one card, or either card of spec 13 §10's replica mode;
nothing is refused across modes. (The 2026-10-05 `{card 0 part, card 1 part}` entries are withdrawn.)

### 4.8 Speculation under TP (specs 8, 19, 26)

- **Taps are free:** both cards hold the identical residual (T1), so DFlash / LiLiCorr taps (Bolt's blocks 1, 18, 35,
  51, 68, 85) are read locally on either card - no copy (under `--pp 2` 26c copies taps 1 / 18 / 35 from device 0).
- **The drafter's placement (decide after P0 and 26a's `r63.cost`):** (i) replicated on both cards and run
  redundantly (no crossing, both produce the same drafts; LiLiCorr's int8 form ~1.1 GB a card; its block reads
  1.2-2.8 GB, spec 26 §4 - all of it on each card); (ii) TP'd itself (its own crossings, ~2 per layer + the head; half
  the bytes); (iii) on one card while the other idles. The draft head over the split vocabulary uses §4.4's merge.
- **Verify at M = 5..8 a card:** each card runs its half at M rows; every crossing carries M rows (§4.1); the MoE
  reads each card's share of the M rows' expert union (§4.3).
- **Prompt lookup** (spec 19e): host-side drafts; only the verify changes. **`--draft-vocab`** (the compact draft
  head, dense models only): its rows split at the same vocabulary boundary, merged as the argmax. **MTP** (spec 8,
  dense only; Bolt has none): the head TP'd like a main layer, M2 (verify rows bitwise M = 1) holds per card.
- **The bars include TP + speculation** (§6): speculation already multiplies tokens per step; TP must still pay off
  under it, at M = 8's crossing cost.

### 4.9 Flags, and the relation to batching

`--tp N` / `--tensor-parallel-size N` (vLLM's spelling; N = 1, the default, or 2) on `b70-decode` and `b70-serve`.
**`--tp 2`, `--pp 2` and `--replica` (spec 13 §10) are mutually exclusive**, refused together by name before any
device. `--tp 1` is today's engine bit for bit (G0). Batching (spec 13 §9) is one card's today; TP with batched rows
is the M-row crossing of §4.1 with rows from different sequences - after 13b-13c, out of scope here. For the dense
models the comparison that decides whether TP is built at all is TP vs the replica mode (§6).

### 4.10 Per-engine work: the target engines

TP is built into each engine, not once:

- **Bolt (first):** `src/runtime/bolt/` (`BoltEngine`, `bolt_capture`, `bolt_prefill*`, `bolt_buffers`, `bolt_sizes`),
  `src/loader/bolt_loader.*` / `bolt_repack.*` / `bolt_layout.h`, `src/model/bolt.*`, `src/kernels/bolt/` (`nb_res_norm`
  and its variants gain remote sources or the fold-then-send epilogue; per-card shape variants of `nb_*` - new binaries,
  the one-card ones unchanged), `src/cli/bolt_serve.h`.
- **Kolibri (second, a follow-on plan after Bolt's record):** `src/runtime/kolibri/`, `src/loader/kolibri1_*`,
  `src/model/kolibri1.*`, `src/kernels/kolibri/` (`kol_post_add`, the sandwich fold).
- **Main engine (Qwen3.8 / Agnes; only on §2's condition):** `src/runtime/{engine,capture,buffers,memory_plan}.*`,
  `src/kernels/prep.cl`'s `prep_res_fold`, `gdn_step.cl`, `attn*.cl`.
- **Shared:** a `tp` crossing protocol (host-tested on two threads like `pp_protocol_test`), the TP planner, the
  snapshot gather / scatter, the head's argmax exchange and sampling merge, the CLI refusals.
- **Not targets:** K2 (`runtime/k2`), Muse (`runtime/muse`), LFM2 (`runtime/lfm2moe`), qwen4exp (`runtime/qwen4exp`).

## 5. Correctness gates

- **T1, both cards agree bitwise:** after every block the two cards' residual streams are byte-identical, and so are
  the MoE route sets; golden prompts + 64 greedy tokens. The fixed fold order makes it hold by construction.
- **T2, against the reference mode:** TP adds partial products in another order, so it is **not** bitwise the
  reference. Bolt and Kolibri: against `--pp 2` (which is bitwise one card's order, spec 16 P1) under the tie-aware
  token gate and spec 25's B3 (route sets with near-ties, the SSM state / h cosine bars) - golden prompts, 4k, the
  agentic prompt. Dense: against one card under the golden tie rule.
- **T3:** determinism and replay bitwise under `--tp 2`; snapshots restored across `--tp 2` / `--pp 2` (§4.7) give the
  cold run's tokens; speculation's verify rows bitwise M = 1 per card (D3 of spec 26, M2 of spec 8).
- **T4, failure modes:** a missing card, no peer access, a crossing timeout or a stale stamp are refused or thrown with
  a clear error within a bound, never a hang; `--tp 2` with `--pp 2` / `--replica` refused by name.

## 6. Speed bars (the right baselines)

Idle box (both cards free of other DRM holders), interleaved pairs after a warm-up, median of 3; device 1's ~3.4 %
prefill gap recorded, not hidden.

- **D1-Bolt, decode:** `--tp 2` against **`--pp 2`** (25c's `r60.speed` arms) at depth 886 / 8192 / 32000 / 65536;
  bar **>= 1.4x at 8192** (proposed; estimated 1.5-1.7x, §2). SergiioB's published vLLM TP 2 row beside it, not a bar.
- **D1-Kolibri:** `--tp 2` against `--pp 2`; bar set after P0 (estimated 1.3-1.5x).
- **D1-dense (only on §2's condition):** `--tp 2` against **one card with `--mtp auto`** (the serving default, not plain
  decode), and `--tp 2 --mtp auto` against it; and against spec 13 §10's replica mode on the multi-session workload.
- **D4, TP + speculation:** Bolt `--tp 2` with spec 26's drafter against `--pp 2` with it (26c's arms); recorded with the
  verify's M.
- **D2, prefill:** recorded against `--pp 2` (Bolt `r61.speed`; dense `r23.s2_*`); the bar is 17c's, after those rows.
- **D3:** `--tp 1` unchanged: G0 bitwise, every existing kernel binary identical.

## 7. Stages

- **17a, P0 (row 68 proposed):** §3's measurements and the go / no-go line. Runs after `r22.devices`, `r22.p1`.
- **17b, decode on Bolt:** the TP placement and planner, per-card slicing and re-fused layouts, per-card `nb_*`
  variants, the crossing, the head exchange, `b70-decode <bolt> --tp 2`; T1, T2, T4; D1-Bolt, D3. After GO.
- **17c, prefill on Bolt:** the option §4.5 picks after row 61; T2 on prefill; D2.
- **17d, integration on Bolt:** snapshots in the one-card layout, LiLiCorr under TP, sampling merge, `b70-serve <bolt>
  --tp 2`, T3, D4, the record. Kolibri is a later plan (17e, not written) on 17b-17d's pieces.

## 8. Out of scope

- TP combined with PP or with the replica mode on the same two cards; more than two cards (the split generalises to
  four - 128 Mamba heads, 8 groups, E divisible by 4 in every layer - but this spec builds and gates two).
- TP with batching (spec 13): after 13b-13c.
- K2, Muse, LFM2, qwen4exp, Ornith; multi-tile devices.

## 9. P0 verdict

(Empty until 17a runs: the measured link, peer-rate and crossing tables, the MoE balance, the GO / NO-GO line.)

---

## Appendix A. The 2026-10-05 design as written (superseded 2026-10-23; kept for the record)

Marks added 2026-10-23 in square brackets name each stale claim and where it is corrected. Nothing else is changed.

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.
**[2026-10-23: stale - PARKED and retargeted at Bolt; see the Status block and §R.]**

**Order:** after spec 16 (pipeline parallel). Spec 17 reuses spec 16's multi-device foundation (one
Level Zero context holding both cards, per-device buffers and captured lists, the peer-write rules)
and spec 16 P0's "remote partial" arm, which measures this spec's key number. **[2026-10-23: 16a never ran and
has no such arm; 17a owns it - §3.]** PP and TP are alternatives over the same two cards: `--pp 2` or `--tp 2`,
not both.

**Prior art:** the two-rank peer-to-peer all-reduce over Level Zero IPC proposed for vLLM's XPU
kernels ([vllm-project/vllm-xpu-kernels#570](https://github.com/vllm-project/vllm-xpu-kernels/pull/570),
RFC [vllm-project/vllm#54766](https://github.com/vllm-project/vllm/issues/54766)), measured on two
B70s: ~6.6-10.8 us per all-reduce of 2-64 KiB inside an XPU graph, and Qwen3.8-27B GPTQ int4 at
**54.90 t/s** decode with TP = 2 at one request (vLLM, `--max-model-len 8192`). **[2026-10-23: both figures are
PUBLISHED (the PR's), not measured here - §1.]**

Every number is **measured** unless marked **derived** or **estimated**. **[2026-10-23: not so for the prior-art
figures and the 12 GB/s link rate (assumed); §1 grades each.]**

### A.1 Why

Single-stream decode is bandwidth-bound: one Qwen3.8 token reads ~14.3 GB (int8 head) at ~32 ms
(31.21 t/s, spec 9). Spec 16 (PP) does not make it faster. **Tensor parallel does:** each card reads
half of every weight matrix, so a step is ~16 ms plus the collectives. Two collectives per layer
(after the attention / GDN output projection and after the MLP down projection), 128 per token, at
~7 us each are ~0.9 ms: **~55-57 t/s (derived), ~1.8x one card**, in line with vLLM's 54.90 t/s on
the same cards. **[2026-10-23: the ~7 µs is published, not ours; the fold's bytes were left out (§4.1); the
baseline for a dense model is one card with `--mtp auto`, and the dense models are conditional (§2, §6).]** For long
agentic sessions this is the one multi-GPU mode that shortens every turn. **[2026-10-23: spec 13 §9 decision 4 /
§10 - the replica mode is the dense models' two-card mode.]**

TP also halves per-card KV (2 of 4 kv-heads each) and GDN state (24 of 48 v-heads each).

### A.2 The decisions

**1. The collective is folded into the residual fold, not a separate kernel.** Our decode GEMVs
already write fp32 split-K `partials`, and `prep_res_fold` (stage A of the norm pair) sums them into
the residual stream before the next layer (`src/runtime/capture.cc`). With TP the other card's
partial is **one more partial** **[2026-10-23: wrong - S more partials: 80 KiB a row a fold at S4 on Qwen3.8,
~10.5 MB a token; §4.1's three mechanisms]**: each row-parallel GEMV (o_proj / out_proj, down, and the MTP head's)
writes its partials locally **and** into the peer's buffer with a peer write and a system-scope flag;
each card's `prep_res_fold` waits for the peer's flag and folds **card 0's partials then card 1's**
in that fixed order on both cards. Both cards therefore compute a **bitwise-identical residual**, and
no launch is added. Spec 16 P0's remote-partial arm measures its cost against a separate all-reduce
kernel; if the fold variant loses, the separate kernel (as in the prior art) is the fallback.
**[2026-10-23: `prep_res_fold` is the main engine's; every other engine has its own fold (§4.10).]**

**2. The split (Megatron-style), per layer:** **[2026-10-23: Qwen3.8's constants; shapes now come from each
descriptor, and the GDN row ignores the fused `qkv‖z` / padded `a‖b` layouts - §4.2.]**

| block | column-parallel (split N, no collective) | row-parallel (split K, partials folded) |
|---|---|---|
| FA attention | q (12 of 24 heads + their gates), k / v (2 of 4 kv-heads) per card | o_proj (K split by the card's heads) |
| GDN | `in_proj_qkv` / `in_proj_z` by head: 8 of 16 k-heads, 24 of 48 v-heads per card; `in_proj_a/b` by v-head | out_proj (K split by v-heads) |
| MLP | gate‖up: 8704 of 17408 rows per card | down (K split) |
| embedding | replicated (gather, no traffic) | - |
| `lm_head` | vocab split: 124160 rows per card | argmax: each card's top-1 and value exchanged, the larger wins (ties to the lower id) |

Norms run replicated on the identical residual. Attention, the GDN recurrence and the conv ring are
per-head and need no communication.

**3. Sampling (decide):** with the vocabulary split, host-side sampling reads both halves of the
logits (two readbacks, ~0.5 MB each); or each card applies top-k to its half and the host merges.
Proposed: the merge (fewer bytes, same distribution after the shared filter). **[2026-10-23: kept - §4.4.]**

**4. Prefill:** the same split, but the collective is large: a 2048-row chunk's partials are
2048 x 5120 fp32 = ~42 MB per fold (~21 MB if sent as bf16, at an accuracy cost), 128 folds per
chunk: ~2.7-5.4 GB per chunk over PCIe 3.0, ~0.2-0.45 s at ~12 GB/s (derived), against compute that
halves from ~1 s to ~0.5 s. **Estimated 1.0-1.4x**, so prefill under `--tp 2` likely runs as spec
16's PP chunk pipeline instead (a hybrid for prefill only: layers split by depth for prefill, by
width for decode, which needs each card to hold both layouts' weights or a reshuffle; ~1.7x+ for
long prompts per spec 16). **(decide)** after P0 measures the transfer and the overlap. **[2026-10-23: the link is
operator-stated PCIe 3.0 x16 and the 12 GB/s is the measured host rate, not a peer rate (§1); the choice waits for rows 23 / 61's measured two-card prefill -
§4.5.]**

### A.3 Design

- **Weights:** the loader slices each linear by the table above at load (packed int4 g64 slices on
  group boundaries: 17408 / 2 = 8704 = 136 groups; head splits are whole heads); `ModelDesc` gains
  `tp_rank`, `tp_size` and derives per-card shapes. Kernels are the existing variants at the per-card
  shapes (e.g. 12 q-heads, 2 kv-heads, 24 GDN v-heads, gate‖up N 17408 per card). **[2026-10-23: per-engine
  descriptors and loaders; a TP memory planner - §4.6, §4.10.]**
- **Capture:** one decode list per card, both replayed for each token; the only cross-card points are
  the folds and the argmax exchange. The host submits both and waits on both fences.
- **Spec 7:** snapshots are per card (each holds its heads' KV and GDN state); the host store's
  entries become `{card 0 part, card 1 part}`. **[2026-10-23: withdrawn - the one-card host layout, §4.7.]**
- **Spec 8 MTP:** the head runs TP too (it is a dense layer at the main model's shapes); verify rows
  stay row-independent, so M2 (verify rows bitwise equal to M = 1 under TP) still holds. **[2026-10-23: and
  specs 19 / 26 - §4.8.]**
- **Spec 10:** decode attention v2 at 12 q-heads per card (6 per kv-head, unchanged ratio).
- **Spec 12 / 14 / 15:** int8 KV per card unchanged; Agnes and Ornith split the same way (Ornith's
  2 kv-heads give one per card; its MoE experts are split by expert, 128 per card, a different
  scheme: out of scope here). **[2026-10-23: the MoE split is now §4.3's decision.]**

### A.4 Correctness gates

- **T1, both cards agree bitwise:** after every layer the two cards' residual streams are
  byte-identical (the fixed fold order), checked for the golden prompts and 64 greedy tokens.
- **T2, against one card:** TP sums partial products in a different order than one card, so the
  result is **not** bitwise equal to one card: logits cosine and greedy tokens under the golden tie
  rule, as the chunked-prefill and split-prefill gates do; the golden gates on `l0` / `l0-int8` with
  `--tp 2`. **[2026-10-23: Bolt / Kolibri cannot run on one card - against `--pp 2`, §5.]**
- **T3:** determinism and replay bitwise with `--tp 2`; spec 7's C1/C2 per card; spec 8's M2.
- **T4, failure modes:** a missing card, no peer access, or a fold timeout refused with a clear
  error, never a hang.

### A.5 Speed bars

Idle box (both cards free of other DRM holders), interleaved pairs against one card, median of 3.
**[2026-10-23: the wrong baseline for Bolt / Kolibri (`--pp 2`) and for Qwen3.8 (`--mtp auto`) - §6.]**

- **P0:** spec 16 P0's remote-partial arm (the fold's added latency) plus, here: per-layer launch
  timing with TP shapes, the lm_head split and argmax exchange, the prefill collective's bandwidth.
  **[2026-10-23: §3.]**
- **D1, decode:** `--tp 2` **>= 1.6x** one card at 4k depth (derived ~1.8x), recorded at 32k / 128k.
- **D2, prefill:** recorded for both options (TP prefill with overlapped transfers, PP-hybrid prefill); the bar is set after 17a.
- **D3:** `--tp 1` unchanged within 1 % of today.

### A.6 Stages

- **17a, probe:** P0 beyond spec 16's arm; the fold-vs-kernel choice; the prefill choice.
- **17b, decode TP:** loader slicing, per-card variants, the folded collective, the lm_head split,
  `--tp 2` decode; T1, T2 (decode), T4; D1, D3.
- **17c, prefill TP:** the prefill path under the split (or the PP hybrid, per 17a); T2 (prefill);
  D2.
- **17d, integration:** spec 7 per card, MTP under TP, sampling merge, the server flag, T3, the
  record.

**[2026-10-23: rewritten Bolt-first - §7 and the plans.]**

### A.7 Out of scope

- TP combined with PP on the same cards; more than two cards (the layout extends to four; the fold
  generalises to more partials, but this spec builds and gates two).
- Expert parallel for MoE (spec 15's Ornith), multi-tile devices. **[2026-10-23: expert parallel is in scope for
  Bolt / Kolibri - §4.3.]**
