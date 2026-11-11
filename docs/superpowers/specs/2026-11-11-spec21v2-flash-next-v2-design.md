# Spec 21 v2 - Qwen3.8-Flash-Next whole on two B70s: expert tiering, the second card, the decode levers

**Status:** design, 2026-11-11, for the operator's review. It continues spec 21 (the `qwen4_exp` family, built) and
spec 22 (the expert-offload tier: its probe tools built, the tier not built), which stay as the historical record.
**Plan:** [2026-11-11-spec21v2a-whole-model-tiering.md](../plans/2026-11-11-spec21v2a-whole-model-tiering.md).

Every number is **measured** unless marked **derived** (from the checkpoint headers and the layout header),
**estimated**, **published** (another engine's own report) or **proposed** (a bar for the operator to approve).

## 1. Where Flash-Next stands today (2026-11-11)

**Built and on main** (spec 21 §12-§16, spec 22a):
- the CPU reference (`tools/oracle/qwen4exp_ref.py`, transformers 5.19.0 layer-streamed, the MTP head from vLLM), the
  descriptor and loader (Intel's g128 experts through the exact g64 expansion, the int8 PLE table pinned in host
  memory), decode, prefill and `b70-serve` with the prefix cache and the MTP head - all **truncated**: `--layers N`
  (18 of 48 layers fit one card, 37 fit two with `--parallel pp`);
- spec 22a's probe tools: `tools/probe/probe_offload` (copy rates, zero-copy expert reads, replay visibility, host
  bandwidth, pinned capacity, the alias check), `tools/probe/offload_links.sh`, `tools/oracle/offload_curve.py`
  (hit-rate curves, swap simulation, MTP verify unions, prefill bytes, the routing profile, the projection) and the
  planner's routed-expert budget (`runtime::qwen4exp::expert_budget`).

**Passed on the cards:** `r31.load`, `r32.k3` / `golden` / `pp` / `cli`, `r33.prefill` / `pp` (since 2026-11-07);
**prefill against an f32 reference** on 18 layers (one card) and 37 (two cards) by the operator's count rule - the
prefill's rejected expert routes at most 1.2x decode's, its f32-accepted tokens at least decode's (18 layers: ratio
1.17, tokens 32 / 32 both; 37 layers: ratio 1.06, tokens 29 / 32 prefill vs 28 / 32 decode;
[record](../../probes/q4exp-f32-aligned-routes-2026-11-10.md), merged 2026-11-10).

**Recorded:** teacher-forced routing traces on Intel's checkpoint, 39 sources, 474 MB (2026-11-09, 8095 s on the box
CPU, peak RSS 94.5 GiB) - the input of the hit-rate curve. Nothing of `probe_offload` has run on a card.

**Not passed or not run:** the golden-set remake with block scores (`r30.golden` / `r30.golden_layers`, held in the
CPU queue), and the two tests that wait for it (`r32.partial`, `r33.golden_real`); `r31.load_real`; the MTP CPU
stages `r34.fixture` / `r34.accept` / `r34.a4_ref`; `r34.a4` (needs the whole model); the PLE read rate
(`r32.ple_rate`); every speed row (none exists for Flash-Next).

**Open quality finding:** decode itself already disagrees with the f32 reference beyond the near-tie window on some
routes, from layer 0 (18 layers: 1089 of 81,576 layer-position routes rejected; 37 layers: 8413 of 167,684), and 4 of
32 decode tokens at 37 layers are not f32-accepted. The short diagnostic traces the first difference to bf16
activation rounding in layer 0's convolution, amplified where a route flips
([record](../../probes/q4exp-f32-decode-diagnostic-2026-11-10.md)); not attributed further. §7 proposes the absolute bar.

**What changed since spec 21 / 22 were written:**
1. **Intel's checkpoint is the weights** (operator 2026-11-07, plan 21q dropped): its dense layers are **bf16**, so a
   token reads ~9.3 GB, not ~5.0 (§2). Spec 22's cost table and its stopping rule assumed our int4 dense layers.
2. **Spec 22 was dropped on 2026-11-06** (host-RAM offload on PCIe 3.0) and came back as a probe on 2026-11-07
   (open-work item 4) after the other engines' tiering designs.
3. **The two-card modes exist:** `--parallel tp` (tensor parallel over P2P, Qwen3.8 only; an all-reduce costs
   3.5 µs inside replayed lists, measured), `replica` (a model copy per card + slots, `--max-num-seqs`) and `pp`
   (layers split; the one Flash-Next uses).
4. **Host memory both cards map stays at or below 4 GiB per allocation**: a 32 GiB pinned allocation imported by the
   second card made the kernel answer "busy" and the driver retry forever (the 2026-11-11 prefix cache hang); the
   prefix cache is now 2 GiB pinned pieces. Box RAM is **121 GB** (spec 22 said ~128).
5. **Zero-config engine:** no tuning flags; the engine sizes and chooses. Spec 22's planned `--expert-cache`,
   `--expert-profile`, `--expert-mirror`, `--expert-budget`, `--expert-swap*`, `--expert-stream` and `--expert-mask` are
   not built. Environment variables are allowed for experiments and tests only.
6. **Fixed slots, no paged KV pool** (operator ruling) - the slot design of `--max-num-seqs` applies.
7. **The quality bar is the f32 judge** (§7), not cosine bars against a bf16 reference.
8. **Removed 2026-11-07 (the never-run triage):** expert pruning - spec 22's P0.8, §4b's coding mask, plan 22m and their
   tools. They do not come back here (one format, best quality).

## 2. The bytes (derived)

Intel's forms, from `loader/qwen4exp_layout.h` (spec 21 §13): a GDN layer 1,492,583,040 B and a QSA layer
1,479,314,432 B, of which 1,336,934,400 B are the 512 routed experts.

| per decode token | GB |
|---|---:|
| non-routed layer weights (36 x 155.6 MB + 12 x 142.4 MB: bf16 dense projections, bf16 shared expert, routers, hyper-connections 1.28 GB) | 7.31 |
| head and fixed (int8 `lm_head` 0.637, PLE block 0.066, final mixer 0.013) | 0.72 |
| **D, non-routed** | **8.03** |
| **R, routed** (48 x 10 experts x 2.611 MB) | **1.253** |
| **total** | **9.28** |

- **Roofline** at 590 GB/s: 63.6 t/s with every expert in VRAM; at the 77.6 % of roofline Qwen3.8 reaches, **49 t/s**.
  Spec 21 §3's 119 / 93 t/s were for int4 dense layers, which no longer exist (open question 1).
- **Weights in VRAM for the whole model:** experts 64.17 + non-routed 7.31 + embedding 1.27 + head and fixed 0.72 +
  the MTP head 1.52 = **~75.0 GB**, against ~62 GB usable on two cards (32.53 GB each less the 1.5 GB reserve).
- **Resident fraction f** (the planner's `expert_budget`, plan 22a Task 1, Intel's forms, two cards, 32k, with the
  MTP head and the prefill scratch): **0.724**. About 17.7 GB of experts stay outside VRAM.
- **KV:** 25,344 B a position (12 QSA layers + indexer keys): 0.83 GB at 32k, 6.6 GB at 262,144 - each GB of KV is a GB
  of experts (spec 22 decision 5, still true).

## 3. The second card: layer split, tensor parallel or a peer expert tier

The cost model is spec 22 §2's, with the VRAM part at Qwen3.8's 77.6 % of roofline (457.8 GB/s), P = 12 GB/s per PCIe
3.0 x16 link (host to device, measured on device 0, 2026-09-27) and h the share of routed bytes found in VRAM:

- **(a) layer split + tier (`pp`, what exists).** Each card caches its own layers' experts; misses stream from pinned
  host memory over that card's link; the cards decode one after the other, so one link works at a time.
  `t = (D + h·R) / 457.8 + (1 - h)·R / P`.
- **(b) tensor parallel + tier (`tp`, Qwen3.8 only today).** Each card holds half of every dense projection and half
  of every expert's intermediate width (640 -> 320 a card: gate‖up split by columns, down by its K - 5 groups of 64),
  hyper-connections replicated (1.28 GB a token on each card), two all-reduces a layer (~98 a token x 3.5 µs =
  0.34 ms). Both links stream misses at once. `t = (4.66 + h·0.627) / 457.8 + (1 - h)·0.627 / P + 0.34 ms`. Per card
  ~38.8 GB of weights against ~29 GB of room: f ≈ 0.70 (derived, rough).
- **(c) card 1 as a peer expert tier** (basalt's design: card 0 runs everything, the cold experts live in card 1's
  VRAM and are read over P2P). Card 0 keeps only ~18 GB of experts beside the 10.8 GB of non-routed weights
  (f ≈ 0.28 on card 0), so most routed bytes cross the bus every token: at an h of 0.6-0.7 that is 0.38-0.50 GB a token,
  31-42 ms at ~12 GB/s - **15-20 t/s (estimated)**. With card 1 computing its own experts (expert parallel) the dense
  8 GB stays on card 0 alone, which caps it near (a). basalt's 46 -> 59 t/s came from a one-card baseline whose dense
  part is small; ours is dominated by 8 GB of bf16 dense weights.

| h | (a) `pp` + tier t/s | (b) `tp` + tier t/s |
|---:|---:|---:|
| 1.00 | 49.3 | 84 |
| 0.97 | 42.9 | 75 |
| 0.95 | 39.4 | 69 |
| 0.90 | 32.9 | 59 |
| 0.85 | 28.2 | 51 |

(derived; MTP off, one request, short context; no swap gains, no launch floor.)

**The ruling this design proposes:** (a) first - it reuses the engine that runs today and gets the whole model end to
end with the smallest build; (b) second, as the speed step, by its own plan once the probe confirms the link rates with
both cards streaming at once; (c) is not built unless the probe's P2P read rate is far above the host link's.

**The yardsticks (published, other engines on the same card):**
- Strata, ONE B70, Flash-Next IQ3_S (~3.4-bit experts) with about half the experts resident: **30-41 t/s** decode,
  **980-1000 t/s** prompt at 4k; two B70s split by layers: **66 t/s**.
- XeStrata, one B70: ~28 t/s decode, 2.4k t/s prefill at 32k (low-bit GGUF, CPU-computed misses).
- Our experts are int4 g64 (~1.24x IQ3_S's bytes) and our dense layers bf16, so 66 t/s is out of reach for (a) at any
  hit rate (49 t/s at h = 1) and within reach for (b).

## 4. The tier

Spec 22 §4's design holds - a byte-sized VRAM cache of whole experts per layer, filled from a routing profile; a
device-side table `[layer][1024]` u64 (512 gate‖up, 512 down pointers) read by the expert kernels through the existing
macro pair `Q4_EXPERT_GU` / `Q4_EXPERT_DN` (`q4_moe.cl`, `q4_pf_moe.cl`); misses read in place from pinned host memory
by the same kernels (zero-copy, spec 21's PLE gather mechanism); no host action between layers; the captured lists do
not change per step. The changes:

- **Pinned pieces of at most 2 GiB.** The mirror is per layer (a layer's non-resident experts: ~0.37 GB at f = 0.72,
  at most 1.34 GB), so it is under the 4 GiB rule by construction; under (b) each card maps only its own halves. Any
  range the probe or the engine maps on both cards stays at or below 4 GiB (2 GiB pieces, the prefix cache's form).
  The PLE table is 16 + 16 ranges of up to ~3.2 GB, mapped by the card that holds layer 1; under (b) card 0 gathers the
  16 rows and the 5 KB result crosses (no second import of the table).
- **The mirror holds only the non-resident experts** (spec 22 decision 4's "misses"): the full mirror (64.17 GB) plus
  the PLE table (51.8 GB) is 116 GB of the box's 121. With the misses mirror the GPU job's host side is ~70 GB (PLE
  51.8 + mirror ~17.7) before the prefix cache, so the prefix cache for the whole model is sized from what is left
  (its auto rule), and no CPU job above ~40 GB runs beside it.
- **The profile ships beside the checkpoint** (`<snapshot>-expert-profile.safetensors`, `offload_curve.py profile` on
  the 2026-11-09 traces) - saber keeps its captured route profile next to its expert pack the same way. No flag: the
  engine fills each card's cache from the profile and the planner's budget; without the file it fills by id order and
  says so at startup.
- **Swaps (only if the trace simulation shows they pay, ≥ 1 point of h over the static fill):** basalt's policy -
  per-expert hit counters written by the route kernel; an expert is promoted only when its count beats the victim's by
  an age margin (basalt: 1280 lookups), so the cache does not churn; the copy goes in 1 MB pieces on the copy engine
  between steps, in idle bus time, so it never delays a miss read for long; the table entry moves to the slot only
  after the copy's event (spec 22 §4's four steps). Tests pin the placement (an environment variable) so residency
  gates stay bitwise.
- **Prefill streams** a chunk's non-resident experts into a staging ring on the copy engine while the previous layer
  computes (spec 22 §4, plan 22c Task 3): a GEMM must not read host memory once per tile. Bus cap (derived): a
  2048-row chunk touches nearly every expert, 17.7 GB per chunk at f = 0.724, 1.5 s at 12 GB/s - ~1400 t/s on one link,
  ~2800 when both links stream (pipelined `pp` chunks or `tp`).
- **An expert pack beside the checkpoint** (saber's xtier: a fixed per-record table and 4096-aligned data) is an
  option for load time only - the loader expands Intel's g128 experts at every load today. Built only if the whole
  model's load time is a problem (it is measured in plan task 5).
- **Not taken:** CPU-computed misses (XeStrata, basalt: our replayed lists have no host hand-off inside a step), 2- or
  3-bit experts (one format), pruning (removed), expert parallel by id halves (saber's EP=2: top-10 routing loads the
  halves unevenly; (b) splits every expert evenly instead).

## 5. Decode levers after the whole model runs

1. **Shared-expert fusion is already built.** Strata fused the shared expert into the routed gate‖up launch (1 user
   115 -> 203 t/s, 8 users 211 -> 351, published, AMD); ours runs it as slot 10 of the 11-slot `q4_moe_gate_up` /
   `q4_moe_down` in decode and as block 512 of the grouped GEMM in prefill (spec 21 §4.4, §15). Nothing to gain there.
2. **Union-aware verify:** the MTP verify at M = K + 1 rows reads each row's experts; reading each distinct expert
   once a verify (Bolt's union MoE, spec 26) cuts the routed bytes, and under the tier it cuts the misses too.
3. **The dense bytes:** 7.31 GB of bf16 dense weights are 79 % of a token's bytes. int8 per channel at load would make
   them ~3.7 GB (39 -> ~57 t/s at h = 0.95 under (a), +~45 %, derived, before the VRAM it frees for experts), int4
   AutoRound ~2 GB - both change the weights, so the operator rules (open question 1). int8 hyper-connections alone:
   1.28 -> 0.64 GB (+~6 %).
4. **QSA, with OpenVINO as the Intel reference.** OpenVINO `3b0ee89623` added fused GPU ops for QSA (the indexer and
   the sparse attention, paged) with a quantised KV layout. Ours is five launches a QSA layer in decode (prep, score,
   select, attention, reduce) and bf16 KV only. Use it as the reference for a fused decode kernel and for int8 KV on
   the 12 QSA layers (int8 KV would free ~3.3 GB at 262k for experts); not its paging (fixed slots, ruled).
5. **The MTP head's experts:** ours are RTN int4 g64 at load (from Intel's bf16); AMD's Quark config keeps the MTP
   experts and the embeddings unquantised, which suggests acceptance is sensitive to them. Measure acceptance with
   the bf16 head and the RTN head on the CPU reference (`r34.accept` both ways) before deciding; the bf16 head costs
   ~3.5 GB of cache (~5 % of f).

## 6. Context against cache

Zero-config: the planner gives the cache every byte left after the non-routed weights, the scratch and the KV of the
chosen context. `--max-len auto` for the whole model is the largest of 32768 / 65536 / 131072 / 262144 whose projected
decode stays within 5 % of 32768's (spec 22 decision 5's rule, proposed); `--max-num-seqs N` slots follow the
`--max-num-seqs` design (context auto-fits N slots).

## 7. Quality bar (the f32 judge)

- **The tier changes no arithmetic:** with every expert resident it is bitwise today's engine (O0); forced residency
  of 0 / 50 / 100 %, random and adversarial, is bitwise the 100 % run in decode, prefill and verify (O1); swaps every
  step are bitwise the no-swap run (O2). A difference is a bug, never a tolerance.
- **The whole model against an f32 reference:** a CPU f32 reference of all 48 layers on `q4exp_4k` (37 layers peaked at
  68 GiB, so ~90 GiB estimated: it holds both cards' lock as a > 64 GB job). Prefill by the operator's count rule (its
  rejected routes ≤ 1.2x decode's, its f32-accepted tokens ≥ decode's).
- **Decode's own bar (proposed):** the twin count used for tensor parallel - the engine's f32-rejected tokens and routes
  on the golden prompts at most the bf16 CPU reference's own on the same prompts. This turns §1's open finding into a
  pass or a fail.
- **The tool-call test:** the 36 scenarios on the whole model against the CPU reference (`r34.a4_ref`), every first
  difference judged by the f32 judge (`serve_judge.py`), as for Qwen3.8.

## 8. Speed gates (proposed, from the yardsticks)

| stage | decode, one request, 4k depth, MTP off | prefill at 4k |
|---|---|---|
| (a) `pp` + tier, the whole model served | **≥ 30 t/s** (above XeStrata's ~28 on one B70, inside Strata's one-card 30-41) | recorded against Strata's 980-1000 |
| (b) `tp` + tier | **≥ 50 t/s** (reported against Strata's two-card 66) | **≥ 1000 t/s** |

**Stopping rule (replaces spec 22 §3's 40 t/s on int4 dense):** if the probe's projection at the measured h gives
(a) < 30 t/s and (b) < 50 t/s, stop before engine work and bring the dense-bytes question (§5.3) to the operator.

## 9. Open questions for the operator

1. **The dense layers:** keep Intel's bf16 (decode capped near 49 t/s at full residency under (a)), or allow int8 per
   channel at load / our own AutoRound of the dense linears only, judged by the f32 judge?
2. **The second card:** approve (a) first and (b) as the speed step (§3)?
3. **The gates of §7 and §8**, in particular decode's twin-count bar.
4. **The MTP head's experts:** decide RTN int4 vs bf16 from `r34.accept` (§5.5).
5. **Host RAM:** the whole model's GPU job pins ~70 GB before the prefix cache; accept that CPU jobs above ~40 GB do
   not run beside it?
